"""Lightweight runtime for model bundles produced by training_pipeline.py.

The feature schema, the transform and the version rules live in
feature_contract.py, which training imports too. This module loads a bundle,
runs it, and turns scores into predictions; it does not define what a feature
is or how it is scaled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from explain import DEFAULT_TOP_K, explain_rows
from feature_contract import (
    BUNDLE_FORMAT_VERSION,
    FeatureContract,
    check_bundle_version,
)


class HybridDetectorNetwork(nn.Module):
    """Shared classifier and benign-reconstruction network."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Iterable[int] = (256, 128),
        latent_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(int(width) for width in hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        encoder_layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        encoder_layers.extend([nn.Linear(previous, latent_dim), nn.GELU()])
        self.encoder = nn.Sequential(*encoder_layers)
        self.classifier = nn.Linear(latent_dim, num_classes)

        decoder_layers: list[nn.Module] = []
        previous = latent_dim
        for width in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(previous, width), nn.GELU()])
            previous = width
        decoder_layers.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(features)
        return self.classifier(latent), self.decoder(latent)


class Detector:
    """Load one immutable bundle and score flow rows through a small interface."""

    def __init__(self, bundle_dir: str | Path, device: str | None = None) -> None:
        self.bundle_dir = Path(bundle_dir)
        manifest_path = self.bundle_dir / "manifest.json"
        weights_path = self.bundle_dir / "model.pt"

        if not manifest_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Expected manifest.json and model.pt in {self.bundle_dir}"
            )

        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        check_bundle_version(self.manifest.get("bundle_format_version"))
        self.contract = FeatureContract.from_manifest(self.manifest["feature_contract"])
        self.feature_names = self.contract.feature_names

        self.classes = tuple(self.manifest["taxonomy"]["classes"])
        model_config = self.manifest["model"]
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = HybridDetectorNetwork(
            input_dim=len(self.feature_names),
            num_classes=len(self.classes),
            hidden_dims=model_config["hidden_dims"],
            latent_dim=model_config["latent_dim"],
            dropout=model_config["dropout"],
        )
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        thresholds = self.manifest["thresholds"]
        self.anomaly_threshold = float(thresholds["anomaly"])
        self.class_thresholds = {
            label: float(value)
            for label, value in thresholds["class_confidence"].items()
        }
        self.coverage = self.manifest["taxonomy"].get("reportable", {})

    @property
    def has_window_coverage(self) -> bool:
        """Whether this bundle was trained with temporal window features.

        A bundle without them cannot see rate-driven behaviour, and CONTEXT.md
        requires that missing coverage be visible rather than assumed.
        """

        return bool(self.contract.window_features)

    @property
    def window_coverage_reason(self) -> str | None:
        return self.contract.window_features_disabled_reason

    def detect(
        self,
        flows: pd.DataFrame,
        batch_size: int = 8192,
        include_probabilities: bool = False,
        explain: bool = False,
        explain_top_k: int = DEFAULT_TOP_K,
    ) -> pd.DataFrame:
        """Return one evidence-friendly detection row per input flow."""

        if flows.empty:
            columns = ["prediction", "confidence", "anomaly_score", "is_anomaly"]
            if explain:
                columns.append("explanation")
            return pd.DataFrame(columns=columns, index=flows.index)

        prepared = self.contract.prepare(flows)
        probabilities: list[np.ndarray] = []
        anomaly_scores: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(prepared), batch_size):
                batch = torch.from_numpy(prepared[start : start + batch_size]).to(
                    self.device
                )
                logits, reconstruction = self.model(batch)
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
                anomaly_scores.append(
                    torch.mean((batch - reconstruction) ** 2, dim=1).cpu().numpy()
                )

        probability_matrix = np.concatenate(probabilities)
        anomaly_score = np.concatenate(anomaly_scores)
        best_indices = probability_matrix.argmax(axis=1)
        best_labels = np.asarray(self.classes, dtype=object)[best_indices]
        confidence = probability_matrix[np.arange(len(best_indices)), best_indices]

        predictions: list[str] = []
        for label, probability, score in zip(best_labels, confidence, anomaly_score):
            threshold = self.class_thresholds.get(str(label), 1.0)
            reportable = bool(self.coverage.get(str(label), label == "Benign"))
            if label != "Benign" and reportable and probability >= threshold:
                predictions.append(str(label))
            elif score >= self.anomaly_threshold:
                predictions.append("Unknown anomaly")
            else:
                predictions.append("Benign")

        result = pd.DataFrame(
            {
                "prediction": predictions,
                "confidence": confidence.astype(float),
                "anomaly_score": anomaly_score.astype(float),
                "is_anomaly": np.asarray(predictions) != "Benign",
            },
            index=flows.index,
        )
        if include_probabilities:
            for index, label in enumerate(self.classes):
                result[f"probability_{label}"] = probability_matrix[:, index]
        if explain:
            result["explanation"] = self._explain(
                prepared, predictions, best_indices, explain_top_k
            )
        return result

    def _explain(
        self,
        prepared: np.ndarray,
        predictions: list[str],
        class_indices: np.ndarray,
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        """Attribute the flagged rows only.

        This is a separate pass rather than part of the scoring loop above:
        scoring runs under torch.inference_mode, and a tensor produced there
        cannot have requires_grad set afterwards. Restricting it to flagged
        rows is also what keeps the cost proportional to what was reported.
        """

        flagged = np.flatnonzero(np.asarray(predictions) != "Benign")
        explanations: list[list[dict[str, Any]]] = [[] for _ in predictions]
        if not flagged.size:
            return explanations

        unknown = np.asarray(
            [predictions[row] == "Unknown anomaly" for row in flagged], dtype=bool
        )
        produced = explain_rows(
            self.model,
            prepared[flagged],
            self.feature_names,
            class_indices=class_indices[flagged],
            use_reconstruction=unknown,
            top_k=top_k,
            device=self.device,
        )
        for row, explanation in zip(flagged, produced):
            explanations[row] = explanation
        return explanations


def run_detector(
    flows: pd.DataFrame,
    bundle_dir: str | Path,
    *,
    batch_size: int = 8192,
    device: str | None = None,
    include_probabilities: bool = False,
    explain: bool = False,
) -> pd.DataFrame:
    """Convenience function for one-off local flow detection."""

    detector = Detector(bundle_dir=bundle_dir, device=device)
    return detector.detect(
        flows,
        batch_size=batch_size,
        include_probabilities=include_probabilities,
        explain=explain,
    )


__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "Detector",
    "HybridDetectorNetwork",
    "run_detector",
]
