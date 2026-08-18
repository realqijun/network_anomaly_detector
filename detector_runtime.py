"""Lightweight runtime for model bundles produced by training_pipeline.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn


BUNDLE_FORMAT_VERSION = 1


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


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


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
        version = self.manifest.get("bundle_format_version")
        if version != BUNDLE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported bundle format {version!r}; "
                f"runtime expects {BUNDLE_FORMAT_VERSION}"
            )

        contract = self.manifest["feature_contract"]
        if contract.get("transform") != "signed_log1p_standardize":
            raise ValueError(f"Unsupported feature transform: {contract.get('transform')}")

        self.feature_names = tuple(contract["features"])
        self.mean = np.asarray(contract["mean"], dtype=np.float32)
        self.scale = np.asarray(contract["scale"], dtype=np.float32)
        if len(self.feature_names) != len(self.mean) or len(self.mean) != len(self.scale):
            raise ValueError("Feature names, means, and scales have different lengths")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("Feature contract contains non-finite scaling values")
        if np.any(self.scale <= 0):
            raise ValueError("Feature contract scale values must be positive")

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

    def _prepare(self, flows: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in flows.columns]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "..." if len(missing) > 8 else ""
            raise ValueError(f"Input is missing required features: {preview}{suffix}")

        numeric = flows.loc[:, self.feature_names].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=np.float32, copy=True)
        invalid = ~np.isfinite(values)
        if invalid.any():
            bad_columns = np.asarray(self.feature_names)[np.unique(np.where(invalid)[1])]
            preview = ", ".join(bad_columns[:8])
            raise ValueError(f"Input contains missing or non-finite values in: {preview}")

        transformed = _signed_log1p(values)
        return ((transformed - self.mean) / self.scale).astype(np.float32, copy=False)

    def detect(
        self,
        flows: pd.DataFrame,
        batch_size: int = 8192,
        include_probabilities: bool = False,
    ) -> pd.DataFrame:
        """Return one evidence-friendly detection row per input flow."""

        if flows.empty:
            columns = ["prediction", "confidence", "anomaly_score", "is_anomaly"]
            return pd.DataFrame(columns=columns, index=flows.index)

        prepared = self._prepare(flows)
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
        return result


def run_detector(
    flows: pd.DataFrame,
    bundle_dir: str | Path,
    *,
    batch_size: int = 8192,
    device: str | None = None,
    include_probabilities: bool = False,
) -> pd.DataFrame:
    """Convenience function for one-off local flow detection."""

    detector = Detector(bundle_dir=bundle_dir, device=device)
    return detector.detect(
        flows,
        batch_size=batch_size,
        include_probabilities=include_probabilities,
    )


__all__ = ["Detector", "HybridDetectorNetwork", "run_detector"]
