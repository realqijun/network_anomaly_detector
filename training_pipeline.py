"""Colab-friendly streaming trainer for CSE-CIC-IDS2018 flow CSV files.

The deployment runtime does not import this module. Training and inference share
the network definition and model-bundle contract through detector_runtime.py.
"""

from __future__ import annotations

import copy
import json
import math
import random
import zipfile
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

from detector_runtime import HybridDetectorNetwork
from evaluation import (
    BENIGN_QUANTILE,
    FALSE_POSITIVE_BUDGET,
    operational_metrics,
    select_anomaly_threshold,
)
from explain import EXPLANATION_METHOD, EXPLANATION_VERSION
from feature_contract import (
    BUNDLE_FORMAT_VERSION,
    IDENTITY_COLUMNS,
    WINDOW_FEATURE_NAMES,
    WINDOW_SECONDS,
    FeatureContract,
    build_window_sidecar,
    canonical_name,
    epoch_seconds,
    normalize_columns,
    parse_timestamps,
    resolve_window_cache_dir,
    window_inputs_available,
)
from splits import build_grouped_split, split_manifest, validate_split_manifest


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
    # Normalized capture-group keys, so they select the same days whether the
    # corpus names files day-first (16-02-2018) or month-first (02-16-2018).
    validation_patterns: tuple[str, ...] = (
        "2018-02-16",
        "2018-02-21",
        "2018-02-23",
        "2018-03-01",
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

    # Window features. "require" fails when the corpus carries no addresses,
    # "auto" degrades and records why, "off" is explicit. The default is loud
    # because a silently empty window block would close issue #6 on paper.
    enable_window_features: str = "require"
    window_cache_dir: str | None = None
    rebuild_window_cache: bool = False

    # Operational evaluation. Thresholds come from an alert budget rather than
    # a ROC-derived cutoff, because the deployment constraint is how many
    # alerts a person can actually read.
    threshold_policy: str = FALSE_POSITIVE_BUDGET
    max_false_positives_per_hour: float = 10.0
    top_k_values: tuple[int, ...] = (10, 50, 100)
    benign_hours_override: float | None = None
    max_evaluation_rows: int = 5_000_000
    min_family_support: int = 50

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or self.batch_size <= 0:
            raise ValueError("chunk_size and batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0.5 <= self.target_precision < 1.0:
            raise ValueError("target_precision must be in [0.5, 1.0)")
        if not 0.9 <= self.anomaly_benign_quantile < 1.0:
            raise ValueError("anomaly_benign_quantile must be in [0.9, 1.0)")
        if self.enable_window_features not in ("require", "auto", "off"):
            raise ValueError(
                "enable_window_features must be 'require', 'auto' or 'off'"
            )
        if self.threshold_policy not in (FALSE_POSITIVE_BUDGET, BENIGN_QUANTILE):
            raise ValueError(f"Unknown threshold_policy {self.threshold_policy!r}")
        if self.max_evaluation_rows <= 0:
            raise ValueError("max_evaluation_rows must be positive")

    @property
    def resolved_window_cache_dir(self) -> str:
        return self.window_cache_dir or str(Path(self.data_dir) / ".window_cache")


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
    """Training and validation files, split by whole capture group.

    Kept as a two-tuple because the Colab notebook unpacks it that way. The
    test and external-test partitions are reachable through
    splits.build_grouped_split.
    """

    split = build_grouped_split(files, validation_patterns)
    return split.train, split.validation


def _normalized_columns(path: Path) -> list[str]:
    """Header names as the contract knows them, not as the file spells them.

    Everything downstream selects and validates on canonical names, so a
    contract built from raw spellings would name columns that _read_chunks
    never loads. That silently broke every corpus using the CIC-IDS2017 long
    headers or CICFlowMeter output, while leaving CSE-CIC-IDS2018 working
    because there the raw and canonical spellings happen to coincide.
    """

    return [canonical_name(column) for column in pd.read_csv(path, nrows=0).columns]


def infer_feature_columns(
    training_files: Sequence[Path], excluded_columns: Sequence[str]
) -> list[str]:
    common = set(_normalized_columns(training_files[0]))
    for path in training_files[1:]:
        common.intersection_update(_normalized_columns(path))

    excluded = {canonical_name(column).casefold() for column in excluded_columns}
    ordered = [
        column
        for column in _normalized_columns(training_files[0])
        if column in common and column.casefold() not in excluded
    ]
    if not ordered:
        raise ValueError("No common candidate features remain after exclusions")

    sample = normalize_columns(pd.read_csv(training_files[0], nrows=5_000))
    numeric_features = []
    for column in ordered:
        numeric = pd.to_numeric(sample[column], errors="coerce")
        if numeric.notna().mean() >= 0.95:
            numeric_features.append(column)
    if not numeric_features:
        raise ValueError("No consistently numeric feature columns were found")
    return numeric_features


def _read_chunks(
    paths: Sequence[Path],
    columns: Sequence[str],
    chunk_size: int,
    identity_columns: Sequence[str] = (),
    sidecars: dict[Path, Path] | None = None,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Stream capture files in chunks, joining each file's window sidecar.

    usecols stays a callable rather than a list. A callable silently ignores
    names a file does not have, while a list raises, and that tolerance is what
    lets identity columns be requested unconditionally, including on corpus days
    that ship without addresses.

    The sidecar is joined by row position, which is only sound because it is
    built at full file length. Filtering happens downstream, after the join.
    """

    wanted = set(columns) | {"Label"} | set(identity_columns)
    for path in paths:
        sidecar = None
        if sidecars and path in sidecars:
            sidecar = pd.read_parquet(sidecars[path])

        offset = 0
        for chunk in pd.read_csv(
            path,
            usecols=lambda name: canonical_name(name) in wanted,
            chunksize=chunk_size,
            low_memory=False,
        ):
            chunk = normalize_columns(chunk).reset_index(drop=True)
            if "Label" not in chunk:
                raise ValueError(f"{path} does not contain a Label column")

            if sidecar is not None:
                block = sidecar.iloc[offset : offset + len(chunk)].reset_index(drop=True)
                if len(block) != len(chunk):
                    raise ValueError(
                        f"Window sidecar for {path} has {len(sidecar)} rows but the "
                        "capture has more; the cache is stale, rebuild it"
                    )
                chunk = pd.concat([chunk, block], axis=1)

            offset += len(chunk)
            yield path, chunk

        if sidecar is not None and len(sidecar) != offset:
            # A sidecar longer than its capture shifts every row's window
            # values without tripping the per-slice guard above, which only
            # sees a short final slice.
            raise ValueError(
                f"Window sidecar for {path} has {len(sidecar)} rows but the "
                f"capture has {offset}; the cache is stale, rebuild it"
            )


def _clean_chunk(
    chunk: pd.DataFrame, feature_columns: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Drop unusable rows and apply the contract transform's first half.

    Filtering happens here, after the sidecar join, so a dropped row takes its
    window values with it and the remaining rows keep the file-global values
    inference will compute for them.
    """

    labels = canonicalize_labels(chunk["Label"])
    missing = [name for name in feature_columns if name not in chunk.columns]
    if missing:
        raise ValueError(
            f"Chunk is missing contract features: {', '.join(missing[:8])}"
        )
    numeric = chunk.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32, copy=True)
    valid = labels.notna().to_numpy() & np.isfinite(values).all(axis=1)
    values = values[valid]
    clean_labels = labels.loc[valid].astype(str).to_numpy()
    values = np.sign(values) * np.log1p(np.abs(values))
    return values.astype(np.float32, copy=False), clean_labels


def build_window_sidecars(
    files: Sequence[Path], config: TrainingConfig
) -> dict[Path, Path]:
    """Precompute one whole-file window block per capture file."""

    cache = resolve_window_cache_dir(config.resolved_window_cache_dir)
    sidecars: dict[Path, Path] = {}
    total = 0
    for path in files:
        sidecar = build_window_sidecar(path, cache, rebuild=config.rebuild_window_cache)
        sidecars[path] = sidecar
        total += sidecar.stat().st_size
    print(f"Window cache: {len(sidecars)} sidecars, {total / 1024**2:,.1f} MiB in {cache}")
    return sidecars


def resolve_window_support(
    training_files: Sequence[Path], config: TrainingConfig
) -> tuple[bool, str | None]:
    """Decide whether this run gets a window block, and record why if not."""

    if config.enable_window_features == "off":
        return False, "disabled by configuration (enable_window_features='off')"

    sample = normalize_columns(pd.read_csv(training_files[0], nrows=1))
    available, missing = window_inputs_available(sample)
    if available:
        return True, None

    reason = (
        "capture files do not carry the flow identity columns window features "
        f"are computed from; missing: {', '.join(missing)}"
    )
    if config.enable_window_features == "require":
        raise ValueError(
            "enable_window_features='require' but " + reason + ". Use 'auto' to "
            "train without the window block, or point at a corpus that carries "
            "addresses and timestamps."
        )
    print(f"Window features disabled: {reason}")
    return False, reason


def iter_prepared_chunks(
    files: Sequence[Path],
    contract: FeatureContract,
    config: TrainingConfig,
    sidecars: dict[Path, Path] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield the exact model-input matrices training sees, chunk by chunk.

    This is the trainer's view of the data, exposed so that it can be compared
    against what Detector computes for the same capture read whole. If those
    two disagree on one value, every metric a bundle reports is measured
    against inputs the deployment will never see.
    """

    if sidecars is None and contract.window_features:
        sidecars = build_window_sidecars(files, config)

    for _, chunk in _read_chunks(
        files,
        contract.flow_features,
        config.chunk_size,
        identity_columns=IDENTITY_COLUMNS,
        sidecars=sidecars,
    ):
        values, labels = _clean_chunk(chunk, contract.feature_names)
        if len(values):
            yield contract.standardize(values), labels


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
    window_features: Sequence[str] = (),
    window_disabled_reason: str | None = None,
    sidecars: dict[Path, Path] | None = None,
) -> tuple[FeatureContract, Counter[str]]:
    """Fit the scaler by streaming, and return the contract it belongs to."""

    scaler = StandardScaler()
    counts: Counter[str] = Counter()
    rng = np.random.default_rng(config.random_seed)
    names = list(feature_columns) + list(window_features)

    for _, chunk in _read_chunks(
        training_files,
        feature_columns,
        config.chunk_size,
        identity_columns=IDENTITY_COLUMNS,
        sidecars=sidecars,
    ):
        values, labels = _clean_chunk(chunk, names)
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

    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale <= 0] = 1.0
    contract = FeatureContract(
        flow_features=tuple(feature_columns),
        window_features=tuple(window_features),
        window_seconds=tuple(WINDOW_SECONDS) if window_features else (),
        window_features_disabled_reason=window_disabled_reason,
        mean=tuple(np.asarray(scaler.mean_, dtype=float).tolist()),
        scale=tuple(scale.tolist()),
    )
    return contract, counts


@dataclass(frozen=True)
class ValidationProfile:
    """A cheap, complete picture of the validation corpus.

    Both the false-positive denominator and the evaluation sampling rate need
    totals over every row, and neither can come from a sample: the rate cannot
    be chosen before the row count is known, and the span of a sample is not
    the duration of a capture.
    """

    benign_hours: float
    benign_hours_source: str
    total_rows: int
    class_counts: dict[str, int]

    def sampling_rate(self, max_rows: int) -> float:
        if self.total_rows <= max_rows:
            return 1.0
        return float(max_rows / self.total_rows)


_BENIGN_BUCKET_SECONDS = 60


def scan_validation_profile(
    files: Sequence[Path], config: TrainingConfig
) -> ValidationProfile:
    """Read Timestamp and Label only, and total up what evaluation needs.

    Benign hours are the union of one-minute buckets containing at least one
    benign row, not the benign min-to-max span. These days interleave benign
    and attack traffic throughout, so that span is essentially the whole day
    and would count every attack hour against a benign-only metric.
    """

    counts: Counter[str] = Counter()
    benign_buckets: set[int] = set()
    total_rows = 0
    saw_timestamp = False

    wanted = {"Timestamp", "Label"}
    for path in files:
        for chunk in pd.read_csv(
            path,
            usecols=lambda name: canonical_name(name) in wanted,
            chunksize=config.chunk_size,
            low_memory=False,
        ):
            chunk = normalize_columns(chunk)
            labels = canonicalize_labels(chunk["Label"])
            total_rows += len(chunk)
            counts.update(labels.dropna().astype(str).tolist())

            if "Timestamp" not in chunk.columns:
                continue
            saw_timestamp = True
            seconds = epoch_seconds(parse_timestamps(chunk["Timestamp"], strict=False))
            benign = labels.eq("Benign").to_numpy() & seconds.notna().to_numpy()
            if benign.any():
                buckets = (seconds[benign] // _BENIGN_BUCKET_SECONDS).astype("int64")
                benign_buckets.update(buckets.tolist())

    if not total_rows:
        raise ValueError("Validation files contain no rows")

    if config.benign_hours_override is not None:
        benign_hours = float(config.benign_hours_override)
        source = "config.benign_hours_override"
    elif saw_timestamp and benign_buckets:
        benign_hours = len(benign_buckets) * _BENIGN_BUCKET_SECONDS / 3600.0
        source = "union of benign-occupied 60s buckets in the validation files"
    else:
        raise ValueError(
            "Cannot measure benign hours: validation files carry no usable "
            "Timestamp column. Set TrainingConfig.benign_hours_override."
        )

    return ValidationProfile(
        benign_hours=benign_hours,
        benign_hours_source=source,
        total_rows=total_rows,
        class_counts=dict(counts),
    )


def _collect_operational(
    model: HybridDetectorNetwork,
    files: Sequence[Path],
    contract: FeatureContract,
    classes: Sequence[str],
    config: TrainingConfig,
    device: torch.device,
    sampling_rate: float,
    sidecars: dict[Path, Path] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score validation rows with uniform, prevalence-preserving sampling.

    The per-epoch pass caps each class, which is right for macro F1 and wrong
    for everything operational: benign vastly outnumbers each attack family
    here, so a per-class cap destroys the prevalence that false positives per
    hour, top-K precision and PR-AUC are all measured against.
    """

    model.eval()
    label_to_index = {label: index for index, label in enumerate(classes)}
    rng = np.random.default_rng(config.random_seed + 2000)
    probability_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []

    with torch.inference_mode():
        for _, chunk in _read_chunks(
            files,
            contract.flow_features,
            config.chunk_size,
            identity_columns=IDENTITY_COLUMNS,
            sidecars=sidecars,
        ):
            values, labels = _clean_chunk(chunk, contract.feature_names)
            known = np.isin(labels, classes)
            values, labels = values[known], labels[known]
            if not len(values):
                continue

            if sampling_rate < 1.0:
                keep = rng.random(len(values)) < sampling_rate
                values, labels = values[keep], labels[keep]
                if not len(values):
                    continue

            values = contract.standardize(values)
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
        raise ValueError("Operational evaluation produced no recognized rows")
    return (
        np.concatenate(target_parts),
        np.concatenate(probability_parts),
        np.concatenate(score_parts),
    )


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


def _train_epoch(
    model: HybridDetectorNetwork,
    files: Sequence[Path],
    contract: FeatureContract,
    classes: Sequence[str],
    optimizer: torch.optim.Optimizer,
    classification_loss: nn.Module,
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
    sidecars: dict[Path, Path] | None = None,
) -> dict[str, float]:
    model.train()
    rng = np.random.default_rng(config.random_seed + epoch)
    label_to_index = {label: index for index, label in enumerate(classes)}
    shuffled_files = list(files)
    random.Random(config.random_seed + epoch).shuffle(shuffled_files)
    total_loss = total_classification = total_reconstruction = 0.0
    batches = rows = 0

    for _, chunk in _read_chunks(
        shuffled_files,
        contract.flow_features,
        config.chunk_size,
        identity_columns=IDENTITY_COLUMNS,
        sidecars=sidecars,
    ):
        values, labels = _clean_chunk(chunk, contract.feature_names)
        known = np.isin(labels, classes)
        values, labels = values[known], labels[known]
        values, labels = _sample_per_class(
            values, labels, config.train_rows_per_class_per_chunk, rng
        )
        if not len(values):
            continue

        values = contract.standardize(values)
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
    contract: FeatureContract,
    classes: Sequence[str],
    config: TrainingConfig,
    device: torch.device,
    sidecars: dict[Path, Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    label_to_index = {label: index for index, label in enumerate(classes)}
    remaining = {label: config.max_validation_rows_per_class for label in classes}
    probability_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    rng = np.random.default_rng(config.random_seed + 1000)

    with torch.inference_mode():
        for _, chunk in _read_chunks(
            files,
            contract.flow_features,
            config.chunk_size,
            identity_columns=IDENTITY_COLUMNS,
            sidecars=sidecars,
        ):
            values, labels = _clean_chunk(chunk, contract.feature_names)
            selections: list[np.ndarray] = []
            for label in classes:
                available = remaining[label]
                if available <= 0:
                    continue
                indices = np.flatnonzero(labels == label)
                if len(indices) > available:
                    indices = rng.choice(indices, size=available, replace=False)
                remaining[label] -= len(indices)
                selections.append(indices)
            if not selections:
                continue

            selected = np.concatenate(selections)
            rng.shuffle(selected)
            values, labels = values[selected], labels[selected]
            values = contract.standardize(values)
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
        reportable[label] = bool(selected_recall >= 0.0)
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


def _json_default(value):
    """Make numpy scalars and Paths serializable.

    sklearn metrics and numpy comparisons return numpy scalars, which json
    refuses, and the failure only appears at the very end of a training run.
    """

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_bundle(
    output_dir: Path,
    model: HybridDetectorNetwork,
    contract: FeatureContract,
    classes: Sequence[str],
    class_thresholds: dict[str, float],
    reportable: dict[str, bool],
    threshold_policy,
    metrics: dict,
    counts: Counter[str],
    split_block: dict,
    config: TrainingConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_contract": contract.to_manifest(),
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
            "anomaly": threshold_policy.threshold,
            "class_confidence": class_thresholds,
            "target_precision": config.target_precision,
            "policy": threshold_policy.to_manifest(),
        },
        "explanation": {
            "version": EXPLANATION_VERSION,
            "methods": EXPLANATION_METHOD,
        },
        "training": {
            "dataset": "CSE-CIC-IDS2018",
            "label_counts": dict(counts),
            "split_manifest": split_block,
            "config": asdict(config),
        },
        "metrics": metrics,
    }
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default)
    )


def _archive_bundle(output_dir: Path) -> str:
    """Zip only the bundle's own files.

    make_archive over the directory would sweep in anything that happens to sit
    beside them, which is how a multi-gigabyte window cache could end up inside
    a downloadable model bundle.
    """

    archive_path = str(output_dir) + ".zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("manifest.json", "model.pt"):
            archive.write(output_dir / name, arcname=name)
    return archive_path


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
    split = build_grouped_split(all_files, config.validation_patterns)
    split_block = split_manifest(split)
    # Run the leakage guard against real runs, not only against tests.
    validate_split_manifest(split_block)
    training_files, validation_files = split.train, split.validation
    print("Training groups:", split.train_groups)
    print("Validation groups:", split.validation_groups)

    feature_columns = infer_feature_columns(training_files, config.excluded_columns)
    use_windows, window_reason = resolve_window_support(training_files, config)
    window_features = tuple(WINDOW_FEATURE_NAMES) if use_windows else ()

    sidecars = (
        build_window_sidecars(list(training_files) + list(validation_files), config)
        if use_windows
        else None
    )
    print(
        f"Feature contract: {len(feature_columns)} flow fields, "
        f"{len(window_features)} window fields"
    )

    contract, counts = fit_feature_contract(
        training_files,
        feature_columns,
        config,
        window_features=window_features,
        window_disabled_reason=window_reason,
        sidecars=sidecars,
    )
    classes = _active_classes(counts)
    print("Training label counts:", dict(counts))

    model = HybridDetectorNetwork(
        input_dim=len(contract.feature_names),
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
            contract,
            classes,
            optimizer,
            classification_loss,
            config,
            device,
            epoch,
            sidecars=sidecars,
        )
        # Model selection uses the per-class capped pass. Macro F1 is
        # prevalence-insensitive, so capping is right here and wrong for the
        # operational metrics below.
        targets, probabilities, anomaly_scores = _collect_validation(
            model, validation_files, contract, classes, config, device, sidecars
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
        model, validation_files, contract, classes, config, device, sidecars
    )
    metrics, quantile_threshold = _evaluate(
        targets,
        probabilities,
        anomaly_scores,
        classes,
        config.anomaly_benign_quantile,
    )
    metrics["history"] = history
    class_thresholds, reportable = _calibrate_class_thresholds(
        targets, probabilities, classes, config.target_precision
    )

    # Operational evaluation is its own pass, over an unbiased sample.
    profile = scan_validation_profile(validation_files, config)
    sampling_rate = profile.sampling_rate(config.max_evaluation_rows)
    print(
        f"Operational evaluation: {profile.total_rows:,} validation rows, "
        f"sampling rate {sampling_rate:.4f}, "
        f"benign hours {profile.benign_hours:.2f} ({profile.benign_hours_source})"
    )
    operational_targets, operational_probabilities, operational_scores = (
        _collect_operational(
            model,
            validation_files,
            contract,
            classes,
            config,
            device,
            sampling_rate,
            sidecars,
        )
    )
    benign_index = classes.index("Benign")
    binary_truth = (operational_targets != benign_index).astype(int)
    threshold_policy = select_anomaly_threshold(
        config.threshold_policy,
        truth=binary_truth,
        scores=operational_scores,
        benign_hours=profile.benign_hours,
        budget_per_hour=config.max_false_positives_per_hour,
        quantile=config.anomaly_benign_quantile,
        sampling_rate=sampling_rate,
    )
    metrics["operational"] = operational_metrics(
        operational_targets,
        operational_probabilities,
        operational_scores,
        classes,
        benign_hours=profile.benign_hours,
        benign_hours_source=profile.benign_hours_source,
        anomaly_threshold=threshold_policy.threshold,
        sampling_rate=sampling_rate,
        top_k_values=config.top_k_values,
        min_family_support=config.min_family_support,
    )
    metrics["benign_quantile_threshold"] = quantile_threshold

    output_dir = Path(config.output_dir)
    _write_bundle(
        output_dir,
        model,
        contract,
        classes,
        class_thresholds,
        reportable,
        threshold_policy,
        metrics,
        counts,
        split_block,
        config,
    )
    archive_path = _archive_bundle(output_dir)
    print(f"Saved model bundle to {output_dir}")
    print(f"Saved downloadable archive to {archive_path}")
    return {
        "bundle_dir": str(output_dir),
        "archive_path": archive_path,
        "classes": classes,
        "feature_columns": list(contract.feature_names),
        "metrics": metrics,
        "class_thresholds": class_thresholds,
        "anomaly_threshold": threshold_policy.threshold,
        "threshold_policy": threshold_policy.to_manifest(),
        "split_manifest": split_block,
    }


__all__ = [
    "TrainingConfig",
    "ValidationProfile",
    "build_window_sidecars",
    "canonicalize_labels",
    "discover_csv_files",
    "fit_feature_contract",
    "infer_feature_columns",
    "iter_prepared_chunks",
    "partition_csv_files",
    "resolve_window_support",
    "scan_validation_profile",
    "train_detector",
]
