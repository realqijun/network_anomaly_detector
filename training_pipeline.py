"""Colab-friendly streaming trainer for CSE-CIC-IDS2018 flow CSV files.

The deployment runtime does not import this module. Training and inference share
the network definition and model-bundle contract through detector_runtime.py.
"""

from __future__ import annotations

import copy
import json
import math
import random
import shutil
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from detector_runtime import BUNDLE_FORMAT_VERSION, HybridDetectorNetwork


DEFAULT_CLASSES = (
    "Benign",
    "DDoS",
    "DoS",
    "Brute Force",
    "XSS",
    "Injection",
    "Bot",
    "Infiltration",
)

DEFAULT_EXCLUDED_COLUMNS = (
    "Label",
    "Flow ID",
    "Src IP",
    "Source IP",
    "Dst IP",
    "Destination IP",
    "Src Port",
    "Source Port",
    "Timestamp",
)


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str
    output_dir: str = "artifacts/cse_cic_ids2018_bundle"
    csv_pattern: str = "**/*.csv"
    validation_patterns: tuple[str, ...] = (
        "02-16-2018",
        "02-21-2018",
        "02-23-2018",
        "03-01-2018",
    )
    chunk_size: int = 100_000
    scaler_rows_per_class_per_chunk: int = 2_000
    train_rows_per_class_per_chunk: int = 20_000
    max_validation_rows_per_class: int = 100_000
    batch_size: int = 4096
    epochs: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    reconstruction_weight: float = 0.15
    hidden_dims: tuple[int, ...] = (256, 128)
    latent_dim: int = 64
    dropout: float = 0.1
    target_precision: float = 0.80
    anomaly_benign_quantile: float = 0.995
    random_seed: int = 42
    excluded_columns: tuple[str, ...] = DEFAULT_EXCLUDED_COLUMNS

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or self.batch_size <= 0:
            raise ValueError("chunk_size and batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0.5 <= self.target_precision < 1.0:
            raise ValueError("target_precision must be in [0.5, 1.0)")
        if not 0.9 <= self.anomaly_benign_quantile < 1.0:
            raise ValueError("anomaly_benign_quantile must be in [0.9, 1.0)")


def canonicalize_labels(labels: pd.Series) -> pd.Series:
    """Map original CSE-CIC labels to stable reportable attack families."""

    normalized = labels.astype(str).str.strip().str.upper()
    result = pd.Series(pd.NA, index=labels.index, dtype="string")
    result.loc[normalized.eq("BENIGN")] = "Benign"
    result.loc[normalized.str.contains("XSS", na=False)] = "XSS"
    result.loc[
        normalized.str.contains(r"SQL|INJECTION", regex=True, na=False)
    ] = "Injection"
    result.loc[normalized.str.contains("DDOS", na=False)] = "DDoS"
    result.loc[
        normalized.str.contains("DOS", na=False)
        & ~normalized.str.contains("DDOS", na=False)
    ] = "DoS"
    result.loc[
        normalized.str.contains(r"BRUTE|PATATOR", regex=True, na=False)
        & result.isna()
    ] = "Brute Force"
    result.loc[normalized.str.contains("BOT", na=False)] = "Bot"
    result.loc[normalized.str.contains("INFIL", na=False)] = "Infiltration"
    return result


def discover_csv_files(data_dir: str | Path, pattern: str = "**/*.csv") -> list[Path]:
    files = sorted(path for path in Path(data_dir).glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files matching {pattern!r} under {data_dir}")
    return files


def partition_csv_files(
    files: Sequence[Path], validation_patterns: Sequence[str]
) -> tuple[list[Path], list[Path]]:
    validation = [
        path for path in files if any(pattern in path.name for pattern in validation_patterns)
    ]
    training = [path for path in files if path not in validation]
    if not validation:
        names = "\n".join(f"- {path.name}" for path in files)
        raise ValueError(
            "None of the validation_patterns matched a CSV filename. "
            "Choose whole capture days from:\n" + names
        )
    if not training:
        raise ValueError("Validation patterns selected every CSV; no training files remain")
    return training, validation


def _normalized_columns(path: Path) -> list[str]:
    return [str(column).strip() for column in pd.read_csv(path, nrows=0).columns]


def infer_feature_columns(
    training_files: Sequence[Path], excluded_columns: Sequence[str]
) -> list[str]:
    common = set(_normalized_columns(training_files[0]))
    for path in training_files[1:]:
        common.intersection_update(_normalized_columns(path))

    excluded = {column.casefold() for column in excluded_columns}
    ordered = [
        column
        for column in _normalized_columns(training_files[0])
        if column in common and column.casefold() not in excluded
    ]
    if not ordered:
        raise ValueError("No common candidate features remain after exclusions")

    sample = pd.read_csv(training_files[0], nrows=5_000)
    sample.columns = sample.columns.str.strip()
    numeric_features = []
    for column in ordered:
        numeric = pd.to_numeric(sample[column], errors="coerce")
        if numeric.notna().mean() >= 0.95:
            numeric_features.append(column)
    if not numeric_features:
        raise ValueError("No consistently numeric feature columns were found")
    return numeric_features


def _read_chunks(
    paths: Sequence[Path], columns: Sequence[str], chunk_size: int
) -> Iterator[tuple[Path, pd.DataFrame]]:
    usecols = list(columns) + ["Label"]
    for path in paths:
        with warnings.catch_warnings():
            # Duplicate header rows make pandas infer mixed types.
            # _clean_chunk deliberately coerces numeric fields and removes them.
            warnings.simplefilter("ignore", pd.errors.DtypeWarning)
            reader = pd.read_csv(
                path,
                usecols=lambda name: name.strip() in usecols,
                chunksize=chunk_size,
            )
            for chunk in reader:
                chunk.columns = chunk.columns.str.strip()
                if "Label" not in chunk:
                    raise ValueError(f"{path} does not contain a Label column")
                yield path, chunk


def _clean_chunk(
    chunk: pd.DataFrame, feature_columns: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    labels = canonicalize_labels(chunk["Label"])
    numeric = chunk.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32, copy=True)
    valid = labels.notna().to_numpy() & np.isfinite(values).all(axis=1)
    values = values[valid]
    clean_labels = labels.loc[valid].astype(str).to_numpy()
    values = np.sign(values) * np.log1p(np.abs(values))
    return values.astype(np.float32, copy=False), clean_labels


def _sample_per_class(
    values: np.ndarray,
    labels: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[np.ndarray] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) > limit:
            indices = rng.choice(indices, size=limit, replace=False)
        selected.append(indices)
    if not selected:
        return values[:0], labels[:0]
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return values[indices], labels[indices]


def fit_feature_contract(
    training_files: Sequence[Path],
    feature_columns: Sequence[str],
    config: TrainingConfig,
) -> tuple[StandardScaler, Counter[str]]:
    scaler = StandardScaler()
    counts: Counter[str] = Counter()
    rng = np.random.default_rng(config.random_seed)

    for _, chunk in _read_chunks(training_files, feature_columns, config.chunk_size):
        values, labels = _clean_chunk(chunk, feature_columns)
        counts.update(labels.tolist())
        values, _ = _sample_per_class(
            values,
            labels,
            config.scaler_rows_per_class_per_chunk,
            rng,
        )
        if len(values):
            scaler.partial_fit(values)

    if not hasattr(scaler, "mean_"):
        raise ValueError("No valid training rows were available to fit preprocessing")
    return scaler, counts


def _active_classes(counts: Counter[str]) -> list[str]:
    classes = [label for label in DEFAULT_CLASSES if counts[label] > 0]
    if "Benign" not in classes:
        raise ValueError("Training data has no valid BENIGN rows")
    if len(classes) < 2:
        raise ValueError("Training data must contain benign and at least one attack family")
    return classes


def _class_weights(counts: Counter[str], classes: Sequence[str]) -> torch.Tensor:
    total = sum(counts[label] for label in classes)
    weights = [
        math.sqrt(total / (len(classes) * max(counts[label], 1))) for label in classes
    ]
    return torch.tensor(np.clip(weights, 0.25, 8.0), dtype=torch.float32)


def _batch_indices(length: int, batch_size: int) -> Iterator[slice]:
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


def _select_validation_indices(
    labels: np.ndarray,
    classes: Sequence[str],
    remaining: dict[str, int],
    rng: np.random.Generator,
) -> np.ndarray:
    selections: list[np.ndarray] = []
    for label in classes:
        available = remaining[label]
        if available <= 0:
            continue
        indices = np.flatnonzero(labels == label)
        if not len(indices):
            continue
        if len(indices) > available:
            indices = rng.choice(indices, size=available, replace=False)
        remaining[label] -= len(indices)
        selections.append(indices)

    if not selections:
        return np.empty(0, dtype=np.int64)
    selected = np.concatenate(selections)
    rng.shuffle(selected)
    return selected


def _train_epoch(
    model: HybridDetectorNetwork,
    files: Sequence[Path],
    feature_columns: Sequence[str],
    scaler: StandardScaler,
    classes: Sequence[str],
    optimizer: torch.optim.Optimizer,
    classification_loss: nn.Module,
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    rng = np.random.default_rng(config.random_seed + epoch)
    label_to_index = {label: index for index, label in enumerate(classes)}
    shuffled_files = list(files)
    random.Random(config.random_seed + epoch).shuffle(shuffled_files)
    total_loss = total_classification = total_reconstruction = 0.0
    batches = rows = 0

    for _, chunk in _read_chunks(shuffled_files, feature_columns, config.chunk_size):
        values, labels = _clean_chunk(chunk, feature_columns)
        known = np.isin(labels, classes)
        values, labels = values[known], labels[known]
        values, labels = _sample_per_class(
            values, labels, config.train_rows_per_class_per_chunk, rng
        )
        if not len(values):
            continue

        values = scaler.transform(values).astype(np.float32)
        order = rng.permutation(len(values))
        values, labels = values[order], labels[order]
        targets = np.fromiter(
            (label_to_index[label] for label in labels), dtype=np.int64, count=len(labels)
        )

        for selection in _batch_indices(len(values), config.batch_size):
            features = torch.from_numpy(values[selection]).to(device)
            target = torch.from_numpy(targets[selection]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, reconstruction = model(features)
            class_loss = classification_loss(logits, target)

            benign = target == label_to_index["Benign"]
            if benign.any():
                reconstruction_loss = nn.functional.huber_loss(
                    reconstruction[benign], features[benign], delta=1.0
                )
            else:
                reconstruction_loss = features.new_zeros(())
            loss = class_loss + config.reconstruction_weight * reconstruction_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.detach())
            total_classification += float(class_loss.detach())
            total_reconstruction += float(reconstruction_loss.detach())
            batches += 1
            rows += len(features)

    denominator = max(batches, 1)
    return {
        "loss": total_loss / denominator,
        "classification_loss": total_classification / denominator,
        "reconstruction_loss": total_reconstruction / denominator,
        "rows": float(rows),
    }


def _collect_validation(
    model: HybridDetectorNetwork,
    files: Sequence[Path],
    feature_columns: Sequence[str],
    scaler: StandardScaler,
    classes: Sequence[str],
    config: TrainingConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    label_to_index = {label: index for index, label in enumerate(classes)}
    remaining = {label: config.max_validation_rows_per_class for label in classes}
    probability_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    rng = np.random.default_rng(config.random_seed + 1000)

    with torch.inference_mode():
        for _, chunk in _read_chunks(files, feature_columns, config.chunk_size):
            values, labels = _clean_chunk(chunk, feature_columns)
            selected = _select_validation_indices(labels, classes, remaining, rng)
            if not len(selected):
                continue

            values, labels = values[selected], labels[selected]
            values = scaler.transform(values).astype(np.float32)
            targets = np.fromiter(
                (label_to_index[label] for label in labels),
                dtype=np.int64,
                count=len(labels),
            )

            for selection in _batch_indices(len(values), config.batch_size):
                features = torch.from_numpy(values[selection]).to(device)
                logits, reconstruction = model(features)
                probability_parts.append(torch.softmax(logits, dim=1).cpu().numpy())
                score_parts.append(
                    torch.mean((features - reconstruction) ** 2, dim=1).cpu().numpy()
                )
                target_parts.append(targets[selection])

    if not target_parts:
        raise ValueError("Validation files produced no recognized, finite rows")
    return (
        np.concatenate(target_parts),
        np.concatenate(probability_parts),
        np.concatenate(score_parts),
    )


def _calibrate_class_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
    target_precision: float,
) -> tuple[dict[str, float], dict[str, bool]]:
    thresholds: dict[str, float] = {"Benign": 0.0}
    reportable: dict[str, bool] = {"Benign": True}
    grid = np.linspace(0.20, 0.99, 160)

    for class_index, label in enumerate(classes):
        if label == "Benign":
            continue
        truth = targets == class_index
        if not truth.any():
            thresholds[label] = 1.0
            reportable[label] = False
            continue

        selected_threshold = 1.0
        selected_recall = -1.0
        for threshold in grid:
            predicted = probabilities[:, class_index] >= threshold
            true_positive = np.count_nonzero(predicted & truth)
            false_positive = np.count_nonzero(predicted & ~truth)
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / np.count_nonzero(truth)
            if precision >= target_precision and recall > selected_recall:
                selected_threshold = float(threshold)
                selected_recall = recall
        thresholds[label] = selected_threshold
        reportable[label] = selected_recall >= 0.0
    return thresholds, reportable


def _evaluate(
    targets: np.ndarray,
    probabilities: np.ndarray,
    anomaly_scores: np.ndarray,
    classes: Sequence[str],
    anomaly_quantile: float,
) -> tuple[dict, float]:
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        targets,
        predictions,
        labels=np.arange(len(classes)),
        target_names=list(classes),
        output_dict=True,
        zero_division=0,
    )
    benign_index = classes.index("Benign")
    benign_scores = anomaly_scores[targets == benign_index]
    if not len(benign_scores):
        raise ValueError("Validation data has no benign rows for anomaly calibration")
    anomaly_threshold = float(np.quantile(benign_scores, anomaly_quantile))

    binary_truth = (targets != benign_index).astype(int)
    if len(np.unique(binary_truth)) == 2:
        report["anomaly_roc_auc"] = float(roc_auc_score(binary_truth, anomaly_scores))
    report["validation_rows"] = int(len(targets))
    return report, anomaly_threshold


def _write_bundle(
    output_dir: Path,
    model: HybridDetectorNetwork,
    scaler: StandardScaler,
    feature_columns: Sequence[str],
    classes: Sequence[str],
    class_thresholds: dict[str, float],
    reportable: dict[str, bool],
    anomaly_threshold: float,
    metrics: dict,
    counts: Counter[str],
    training_files: Sequence[Path],
    validation_files: Sequence[Path],
    config: TrainingConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale <= 0] = 1.0
    manifest = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_contract": {
            "features": list(feature_columns),
            "transform": "signed_log1p_standardize",
            "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "scale": scale.tolist(),
            "missing_value_policy": "reject",
        },
        "taxonomy": {
            "classes": list(classes),
            "reportable": reportable,
            "unknown_label": "Unknown anomaly",
        },
        "model": {
            "type": "hybrid_mlp_classifier_benign_autoencoder",
            "hidden_dims": list(config.hidden_dims),
            "latent_dim": config.latent_dim,
            "dropout": config.dropout,
        },
        "thresholds": {
            "anomaly": anomaly_threshold,
            "class_confidence": class_thresholds,
            "target_precision": config.target_precision,
        },
        "training": {
            "dataset": "CSE-CIC-IDS2018",
            "label_counts": dict(counts),
            "training_files": [path.name for path in training_files],
            "validation_files": [path.name for path in validation_files],
            "config": asdict(config),
        },
        "metrics": metrics,
    }
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def train_detector(config: TrainingConfig) -> dict:
    """Train, calibrate, and save one versioned detector bundle."""

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    all_files = discover_csv_files(config.data_dir, config.csv_pattern)
    training_files, validation_files = partition_csv_files(
        all_files, config.validation_patterns
    )
    print("Training files:", [path.name for path in training_files])
    print("Validation files:", [path.name for path in validation_files])

    feature_columns = infer_feature_columns(training_files, config.excluded_columns)
    print(f"Feature contract: {len(feature_columns)} common numeric fields")
    scaler, counts = fit_feature_contract(training_files, feature_columns, config)
    classes = _active_classes(counts)
    print("Training label counts:", dict(counts))

    model = HybridDetectorNetwork(
        input_dim=len(feature_columns),
        num_classes=len(classes),
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    classification_loss = nn.CrossEntropyLoss(
        weight=_class_weights(counts, classes).to(device)
    )

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_macro_f1 = -1.0
    for epoch in range(config.epochs):
        summary = _train_epoch(
            model,
            training_files,
            feature_columns,
            scaler,
            classes,
            optimizer,
            classification_loss,
            config,
            device,
            epoch,
        )
        targets, probabilities, anomaly_scores = _collect_validation(
            model,
            validation_files,
            feature_columns,
            scaler,
            classes,
            config,
            device,
        )
        metrics, _ = _evaluate(
            targets,
            probabilities,
            anomaly_scores,
            classes,
            config.anomaly_benign_quantile,
        )
        macro_f1 = float(metrics["macro avg"]["f1-score"])
        summary["validation_macro_f1"] = macro_f1
        history.append(summary)
        print(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"loss={summary['loss']:.4f}, val_macro_f1={macro_f1:.4f}, "
            f"sampled_rows={int(summary['rows']):,}"
        )
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    targets, probabilities, anomaly_scores = _collect_validation(
        model,
        validation_files,
        feature_columns,
        scaler,
        classes,
        config,
        device,
    )
    metrics, anomaly_threshold = _evaluate(
        targets,
        probabilities,
        anomaly_scores,
        classes,
        config.anomaly_benign_quantile,
    )
    metrics["history"] = history
    class_thresholds, reportable = _calibrate_class_thresholds(
        targets,
        probabilities,
        classes,
        config.target_precision,
    )

    output_dir = Path(config.output_dir)
    _write_bundle(
        output_dir,
        model,
        scaler,
        feature_columns,
        classes,
        class_thresholds,
        reportable,
        anomaly_threshold,
        metrics,
        counts,
        training_files,
        validation_files,
        config,
    )
    archive_path = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
    print(f"Saved model bundle to {output_dir}")
    print(f"Saved downloadable archive to {archive_path}")
    return {
        "bundle_dir": str(output_dir),
        "archive_path": archive_path,
        "classes": classes,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "class_thresholds": class_thresholds,
        "anomaly_threshold": anomaly_threshold,
    }


__all__ = [
    "TrainingConfig",
    "canonicalize_labels",
    "discover_csv_files",
    "partition_csv_files",
    "train_detector",
]
