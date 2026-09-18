"""The authoritative, versioned flow feature contract.

Training and inference both import this module. Neither computes a feature
value or a transform itself, so there is exactly one definition of what a
feature is called, what units it carries, and how it is scaled.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Manifest layout version. Bump when the shape of manifest.json changes.
BUNDLE_FORMAT_VERSION = 2

# Feature list, ordering and transform semantics. Bump when those change.
# These are separate axes on purpose: a feature change must not force a
# manifest-layout bump, and a layout change must not invalidate a feature set.
FEATURE_CONTRACT_VERSION = 1
COMPATIBLE_CONTRACT_VERSIONS = frozenset({1})

TRANSFORM = "signed_log1p_standardize"
MISSING_VALUE_POLICY = "reject"


# Flow identity and event time. Never model inputs on their own, but required
# to compute window features and to join evidence back to a scored row.
# `Protocol` and `Dst Port` stay eligible as model features; the rest do not.
IDENTITY_COLUMNS = (
    "Timestamp",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Dst Port",
    "Protocol",
)

# Measurement columns the window block aggregates.
WINDOW_SOURCE_COLUMNS = (
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "SYN Flag Cnt",
    "ACK Flag Cnt",
)

# Observation windows, in seconds. Buckets are floor(epoch_seconds / W), so an
# offline replay and a live feed bucket identically with no capture start time.
WINDOW_SECONDS = (1, 10, 60)

_WINDOW_FIELDS = (
    "dst_flow_count",
    "dst_unique_src",
    "dst_unique_port",
    "src_flow_count",
    "src_unique_dst",
    "pair_flow_count",
    "service_flow_count",
    "dst_bytes_rate",
    "dst_pkt_rate",
    "syn_ack_imbalance",
    "dst_flow_count_delta",
)

WINDOW_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"win{seconds}s_{field_name}"
    for seconds in WINDOW_SECONDS
    for field_name in _WINDOW_FIELDS
)


def _cic_ids2017_aliases() -> dict[str, str]:
    """CIC-IDS2017 long headers, mapped to the canonical short names."""

    aliases = {
        "Source IP": "Src IP",
        "Destination IP": "Dst IP",
        "Source Port": "Src Port",
        "Destination Port": "Dst Port",
        "Total Fwd Packets": "Tot Fwd Pkts",
        "Total Backward Packets": "Tot Bwd Pkts",
        "Total Length of Fwd Packets": "TotLen Fwd Pkts",
        "Total Length of Bwd Packets": "TotLen Bwd Pkts",
        "Flow Bytes/s": "Flow Byts/s",
        "Flow Packets/s": "Flow Pkts/s",
        "Fwd IAT Total": "Fwd IAT Tot",
        "Bwd IAT Total": "Bwd IAT Tot",
        "Fwd Header Length": "Fwd Header Len",
        "Bwd Header Length": "Bwd Header Len",
        "Fwd Packets/s": "Fwd Pkts/s",
        "Bwd Packets/s": "Bwd Pkts/s",
        "Min Packet Length": "Pkt Len Min",
        "Max Packet Length": "Pkt Len Max",
        "Packet Length Mean": "Pkt Len Mean",
        "Packet Length Std": "Pkt Len Std",
        "Packet Length Variance": "Pkt Len Var",
        "Average Packet Size": "Pkt Size Avg",
        "Avg Fwd Segment Size": "Fwd Seg Size Avg",
        "Avg Bwd Segment Size": "Bwd Seg Size Avg",
        "Init_Win_bytes_forward": "Init Fwd Win Byts",
        "Init_Win_bytes_backward": "Init Bwd Win Byts",
        "act_data_pkt_fwd": "Fwd Act Data Pkts",
        "min_seg_size_forward": "Fwd Seg Size Min",
        "CWE Flag Count": "CWE Flag Count",
    }
    for direction in ("Fwd", "Bwd"):
        for statistic in ("Max", "Min", "Mean", "Std"):
            aliases[f"{direction} Packet Length {statistic}"] = (
                f"{direction} Pkt Len {statistic}"
            )
    for flag in ("FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE"):
        aliases[f"{flag} Flag Count"] = f"{flag} Flag Cnt"
    for direction in ("Fwd", "Bwd"):
        aliases[f"Subflow {direction} Packets"] = f"Subflow {direction} Pkts"
        aliases[f"Subflow {direction} Bytes"] = f"Subflow {direction} Byts"
        aliases[f"{direction} Avg Bytes/Bulk"] = f"{direction} Byts/b Avg"
        aliases[f"{direction} Avg Packets/Bulk"] = f"{direction} Pkts/b Avg"
        aliases[f"{direction} Avg Bulk Rate"] = f"{direction} Blk Rate Avg"
    return aliases


def _cicflowmeter_cli_aliases() -> dict[str, str]:
    """CICFlowMeter CLI snake_case output, mapped to canonical names.

    The flow-measurement half of this table previously lived in
    deploy/pcap_parser.py. The identity entries are new: the CLI emits
    src_ip, dst_ip, src_port and timestamp, which the old table dropped, so
    a PCAP upload could never have supported the window block.
    """

    aliases = {
        "src_ip": "Src IP",
        "dst_ip": "Dst IP",
        "src_port": "Src Port",
        "dst_port": "Dst Port",
        "protocol": "Protocol",
        "timestamp": "Timestamp",
        "flow_duration": "Flow Duration",
        "tot_fwd_pkts": "Tot Fwd Pkts",
        "tot_bwd_pkts": "Tot Bwd Pkts",
        "totlen_fwd_pkts": "TotLen Fwd Pkts",
        "totlen_bwd_pkts": "TotLen Bwd Pkts",
        "flow_byts_s": "Flow Byts/s",
        "flow_pkts_s": "Flow Pkts/s",
        "flow_iat_mean": "Flow IAT Mean",
        "flow_iat_std": "Flow IAT Std",
        "flow_iat_max": "Flow IAT Max",
        "flow_iat_min": "Flow IAT Min",
        "fwd_iat_tot": "Fwd IAT Tot",
        "bwd_iat_tot": "Bwd IAT Tot",
        "fwd_psh_flags": "Fwd PSH Flags",
        "bwd_psh_flags": "Bwd PSH Flags",
        "fwd_urg_flags": "Fwd URG Flags",
        "bwd_urg_flags": "Bwd URG Flags",
        "fwd_header_len": "Fwd Header Len",
        "bwd_header_len": "Bwd Header Len",
        "fwd_pkts_s": "Fwd Pkts/s",
        "bwd_pkts_s": "Bwd Pkts/s",
        "pkt_len_min": "Pkt Len Min",
        "pkt_len_max": "Pkt Len Max",
        "pkt_len_mean": "Pkt Len Mean",
        "pkt_len_std": "Pkt Len Std",
        "pkt_len_var": "Pkt Len Var",
        "cwr_flag_count": "CWE Flag Count",
        "down_up_ratio": "Down/Up Ratio",
        "pkt_size_avg": "Pkt Size Avg",
        "fwd_seg_size_avg": "Fwd Seg Size Avg",
        "bwd_seg_size_avg": "Bwd Seg Size Avg",
        "init_fwd_win_byts": "Init Fwd Win Byts",
        "init_bwd_win_byts": "Init Bwd Win Byts",
        "fwd_act_data_pkts": "Fwd Act Data Pkts",
        "fwd_seg_size_min": "Fwd Seg Size Min",
    }
    for direction, short in (("fwd", "Fwd"), ("bwd", "Bwd")):
        for statistic in ("max", "min", "mean", "std"):
            aliases[f"{direction}_pkt_len_{statistic}"] = (
                f"{short} Pkt Len {statistic.capitalize()}"
            )
            aliases[f"{direction}_iat_{statistic}"] = (
                f"{short} IAT {statistic.capitalize()}"
            )
        aliases[f"subflow_{direction}_pkts"] = f"Subflow {short} Pkts"
        aliases[f"subflow_{direction}_byts"] = f"Subflow {short} Byts"
        aliases[f"{direction}_byts_b_avg"] = f"{short} Byts/b Avg"
        aliases[f"{direction}_pkts_b_avg"] = f"{short} Pkts/b Avg"
        aliases[f"{direction}_blk_rate_avg"] = f"{short} Blk Rate Avg"
    for flag in ("fin", "syn", "rst", "psh", "ack", "urg", "ece"):
        aliases[f"{flag}_flag_cnt"] = f"{flag.upper()} Flag Cnt"
    for state in ("active", "idle"):
        for statistic in ("max", "min", "mean", "std"):
            aliases[f"{state}_{statistic}"] = (
                f"{state.capitalize()} {statistic.capitalize()}"
            )
    return aliases


COLUMN_ALIASES: dict[str, str] = {
    **_cic_ids2017_aliases(),
    **_cicflowmeter_cli_aliases(),
}


def normalize_columns(flows: pd.DataFrame) -> pd.DataFrame:
    """Strip header padding and fold known spellings onto canonical names.

    Columns with no known alias are left untouched. This is the first call in
    every entry point, so only one spelling ever reaches the contract.
    """

    stripped = {column: str(column).strip() for column in flows.columns}
    renamed = {
        column: COLUMN_ALIASES.get(bare, bare) for column, bare in stripped.items()
    }
    return flows.rename(columns=renamed)


# Ordered candidates. Day-first and month-first are both listed so that a value
# satisfying both can be detected and rejected rather than silently resolved.
_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%d/%m/%Y %I:%M:%S %p",
    "%m/%d/%Y %I:%M:%S %p",
    "%d-%m-%Y %H:%M:%S",
    "%m-%d-%Y %H:%M:%S",
)


def parse_timestamps(values: pd.Series, strict: bool = True) -> pd.Series:
    """Parse capture timestamps with an explicit format, never by guessing.

    These CSVs carry a local-time string with no zone, and pandas will happily
    read 03/01/2018 as 3 January under `dayfirst=True` and as 1 March without
    it. Either choice silently reorders every window bucket, so this function
    accepts a format only if it parses every value, and refuses when two
    formats both parse and disagree.

    `strict=False` is used only by the sidecar and profile passes, which read a
    whole file to produce a row-aligned result. Those files carry repeated
    header rows, whose Timestamp is the literal string "Timestamp". Dropping
    such rows would make the output shorter than the file, and because the
    sidecar is joined by row position every later row would silently receive
    another row's window values. So non-strict mode yields NaT for them and
    keeps the length, letting the caller's own non-finite filter remove them.
    """

    text = values.astype("string").str.strip()
    present = text.notna() & text.ne("")
    if not present.any():
        raise ValueError("Timestamp column contains no parseable values")

    threshold = 1.0 if strict else 0.5
    candidates: dict[str, pd.Series] = {}
    for fmt in _TIMESTAMP_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        share = float(parsed[present].notna().mean())
        if share >= threshold:
            candidates[fmt] = parsed

    if not candidates:
        sample = text[present].iloc[0]
        raise ValueError(
            f"Could not parse Timestamp with any known format; first value is {sample!r}"
        )

    resolved = sorted(
        candidates.items(), key=lambda item: float(item[1].notna().mean()), reverse=True
    )
    reference_format, reference = resolved[0]
    for other_format, other in resolved[1:]:
        comparable = present & reference.notna() & other.notna()
        if not comparable.any():
            continue
        if not reference[comparable].equals(other[comparable]):
            disagreement = text[comparable][
                reference[comparable].ne(other[comparable])
            ].iloc[0]
            raise ValueError(
                "Timestamp day/month order is ambiguous: "
                f"{disagreement!r} parses under both {reference_format!r} and "
                f"{other_format!r} with different results. Normalize the capture "
                "to ISO 8601 before training or scoring."
            )
    return reference


def canonical_name(column: str) -> str:
    """The contract name a raw header maps to."""

    bare = str(column).strip()
    return COLUMN_ALIASES.get(bare, bare)


WINDOW_INPUT_COLUMNS: tuple[str, ...] = IDENTITY_COLUMNS + WINDOW_SOURCE_COLUMNS


def window_inputs_available(flows: pd.DataFrame) -> tuple[bool, list[str]]:
    """Report whether a frame can support the window block, and what is missing."""

    present = {canonical_name(column) for column in flows.columns}
    missing = [column for column in WINDOW_INPUT_COLUMNS if column not in present]
    return not missing, missing


def window_coverage_warning(flows: pd.DataFrame) -> str | None:
    """Warn when a frame is too small for its window features to mean anything.

    A handful of hand-picked flows produce near-1 counts for every window
    field, which is far outside the training distribution, and the model would
    otherwise score them confidently with nothing indicating the problem.
    """

    available, _ = window_inputs_available(flows)
    if not available or flows.empty:
        return None

    widest = max(WINDOW_SECONDS)
    normalized = normalize_columns(flows)
    seconds = epoch_seconds(parse_timestamps(normalized["Timestamp"], strict=False))
    finite = seconds.dropna()
    if finite.empty:
        return None

    span = float(finite.max() - finite.min())
    if span < widest:
        return (
            f"This capture spans {span:.1f}s, shorter than the widest observation "
            f"window of {widest}s, so window features describe a partial window "
            "and scores from them are unreliable."
        )
    return None


def _window_block(frame: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """Compute one window's eleven fields over a complete, time-valid frame."""

    bucket = np.floor(frame["_epoch"] / seconds).astype("int64")
    work = frame.assign(_bucket=bucket)

    by_dst = work.groupby(["_bucket", "Dst IP"], sort=False)
    by_src = work.groupby(["_bucket", "Src IP"], sort=False)
    by_pair = work.groupby(["_bucket", "Src IP", "Dst IP"], sort=False)
    by_service = work.groupby(["_bucket", "Dst IP", "Dst Port"], sort=False)

    syn = by_dst["SYN Flag Cnt"].transform("sum")
    ack = by_dst["ACK Flag Cnt"].transform("sum")

    block = pd.DataFrame(index=frame.index)
    block[f"win{seconds}s_dst_flow_count"] = by_dst["Dst IP"].transform("size")
    block[f"win{seconds}s_dst_unique_src"] = by_dst["Src IP"].transform("nunique")
    block[f"win{seconds}s_dst_unique_port"] = by_dst["Dst Port"].transform("nunique")
    block[f"win{seconds}s_src_flow_count"] = by_src["Src IP"].transform("size")
    block[f"win{seconds}s_src_unique_dst"] = by_src["Dst IP"].transform("nunique")
    block[f"win{seconds}s_pair_flow_count"] = by_pair["Dst IP"].transform("size")
    block[f"win{seconds}s_service_flow_count"] = by_service["Dst IP"].transform("size")
    block[f"win{seconds}s_dst_bytes_rate"] = (
        by_dst["_bytes"].transform("sum") / seconds
    )
    block[f"win{seconds}s_dst_pkt_rate"] = by_dst["_packets"].transform("sum") / seconds
    block[f"win{seconds}s_syn_ack_imbalance"] = (syn - ack) / (syn + ack + 1.0)

    # Baseline delta against the immediately preceding bucket for the same
    # destination. An absent predecessor counts as zero, so a destination's
    # first observed bucket reports its own load as the change.
    counts = (
        work.groupby(["Dst IP", "_bucket"], sort=False)
        .size()
        .rename("_count")
        .reset_index()
    )
    previous = counts.rename(columns={"_count": "_previous"})
    previous["_bucket"] = previous["_bucket"] + 1
    joined = counts.merge(previous, on=["Dst IP", "_bucket"], how="left")
    joined["_delta"] = joined["_count"] - joined["_previous"].fillna(0.0)
    lookup = work[["Dst IP", "_bucket"]].merge(
        joined[["Dst IP", "_bucket", "_delta"]], on=["Dst IP", "_bucket"], how="left"
    )
    block[f"win{seconds}s_dst_flow_count_delta"] = lookup["_delta"].to_numpy()

    return block.astype(np.float32)


def compute_window_features(flows: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Compute the window block over one complete frame.

    This is the only implementation, called by training and inference alike.
    Aggregates are scoped to the frame handed in, so the caller is responsible
    for handing in a complete capture rather than an arbitrary slice of one.

    The result always has one row per input row. Rows whose timestamp could not
    be parsed carry NaN window values rather than being dropped, so a caller
    joining by row position stays aligned.
    """

    normalized = normalize_columns(flows)
    available, missing = window_inputs_available(normalized)
    if not available:
        raise ValueError(
            "Cannot compute window features; frame is missing: " + ", ".join(missing)
        )

    seconds = epoch_seconds(parse_timestamps(normalized["Timestamp"], strict=strict))
    work = pd.DataFrame(
        {
            "_epoch": seconds.to_numpy(),
            "Src IP": normalized["Src IP"].astype("string"),
            "Dst IP": normalized["Dst IP"].astype("string"),
            "Dst Port": pd.to_numeric(normalized["Dst Port"], errors="coerce"),
            "SYN Flag Cnt": pd.to_numeric(normalized["SYN Flag Cnt"], errors="coerce"),
            "ACK Flag Cnt": pd.to_numeric(normalized["ACK Flag Cnt"], errors="coerce"),
            "_bytes": pd.to_numeric(normalized["TotLen Fwd Pkts"], errors="coerce")
            + pd.to_numeric(normalized["TotLen Bwd Pkts"], errors="coerce"),
            "_packets": pd.to_numeric(normalized["Tot Fwd Pkts"], errors="coerce")
            + pd.to_numeric(normalized["Tot Bwd Pkts"], errors="coerce"),
        },
        index=normalized.index,
    )

    usable = work["_epoch"].notna() & work["Src IP"].notna() & work["Dst IP"].notna()
    result = pd.DataFrame(
        np.nan, index=flows.index, columns=list(WINDOW_FEATURE_NAMES), dtype=np.float32
    )
    if not usable.any():
        return result

    valid = work.loc[usable].fillna({"SYN Flag Cnt": 0.0, "ACK Flag Cnt": 0.0,
                                     "_bytes": 0.0, "_packets": 0.0, "Dst Port": -1})
    for seconds_in_window in WINDOW_SECONDS:
        block = _window_block(valid, seconds_in_window)
        result.loc[usable, block.columns] = block.to_numpy()
    return result


def _sidecar_key(csv_path: Path) -> str:
    """Identify a sidecar by its source file and the definition that built it.

    The resolved absolute path is part of the key because discover_csv_files
    globs recursively, so two capture files can share a stem; a stem-keyed
    cache would join one file's window values onto another file's rows with no
    error anywhere. The contract and window definition are part of it so that
    editing WINDOW_SECONDS does not leave stale sidecars looking fresh.
    """

    stats = csv_path.stat()
    material = "|".join(
        [
            str(csv_path.resolve()),
            str(stats.st_mtime_ns),
            str(stats.st_size),
            str(FEATURE_CONTRACT_VERSION),
            str(WINDOW_SECONDS),
            str(WINDOW_FEATURE_NAMES),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def resolve_window_cache_dir(preferred: str | Path) -> Path:
    """Return a writable cache directory, falling back to a temporary one.

    On Colab the natural location sits inside the kagglehub download cache,
    which is not guaranteed writable. Losing the cache is cheap; failing the
    training run over it is not.
    """

    candidate = Path(preferred)
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write-probe"
        probe.touch()
        probe.unlink()
        return candidate
    except OSError as error:
        fallback = Path(tempfile.gettempdir()) / "network-anomaly-window-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "Window cache directory %s is not writable (%s); using %s instead",
            candidate,
            error,
            fallback,
        )
        return fallback


def build_window_sidecar(
    csv_path: str | Path, cache_dir: str | Path, rebuild: bool = False
) -> Path:
    """Compute a whole-file window block once and cache it beside the input.

    Window aggregates cannot be computed per chunk: a groupby inside a 100k-row
    slice is not the same number as over the file, and the baseline delta needs
    a bucket that lives in the previous slice. So the whole file is read once,
    through the narrow set of identity and measurement columns, and the result
    is joined back to each training chunk by row position.
    """

    csv_path = Path(csv_path)
    cache = resolve_window_cache_dir(cache_dir)
    sidecar = cache / f"{csv_path.stem}.{_sidecar_key(csv_path)}.window.parquet"
    if sidecar.is_file() and not rebuild:
        return sidecar

    wanted = set(WINDOW_INPUT_COLUMNS)
    # A callable usecols silently ignores names a file does not have, while a
    # list raises. That tolerance is what lets identity columns be requested
    # unconditionally, including on corpus days that ship without addresses.
    frame = pd.read_csv(
        csv_path,
        usecols=lambda name: canonical_name(name) in wanted,
        low_memory=False,
    )
    computed = compute_window_features(frame, strict=False)

    handle, staging = tempfile.mkstemp(dir=cache, suffix=".parquet")
    os.close(handle)
    try:
        computed.to_parquet(staging, index=False)
        os.replace(staging, sidecar)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise
    return sidecar


class ContractMismatchError(ValueError):
    """Raised when a bundle declares a contract this runtime cannot honour."""


def check_bundle_version(found: Any) -> None:
    """Check the manifest layout version.

    This is the first of two checks and it never substitutes for the second:
    a bundle whose layout is current can still declare a stale feature
    contract, and that case must fail too.
    """

    if found != BUNDLE_FORMAT_VERSION:
        raise ContractMismatchError(
            f"Unsupported bundle format {found!r}; runtime expects "
            f"{BUNDLE_FORMAT_VERSION}. Retrain to produce a current bundle."
        )


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


@dataclass(frozen=True)
class FeatureContract:
    """The ordered feature schema and the transform that produced its scaling.

    One instance describes exactly what a model bundle was trained on, so
    training and inference cannot drift: both call `prepare`.
    """

    flow_features: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    window_features: tuple[str, ...] = ()
    window_seconds: tuple[int, ...] = ()
    version: int = FEATURE_CONTRACT_VERSION
    transform: str = TRANSFORM
    missing_value_policy: str = MISSING_VALUE_POLICY
    window_features_disabled_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "flow_features", tuple(self.flow_features))
        object.__setattr__(self, "window_features", tuple(self.window_features))
        object.__setattr__(self, "window_seconds", tuple(int(w) for w in self.window_seconds))
        object.__setattr__(self, "mean", tuple(float(value) for value in self.mean))
        object.__setattr__(self, "scale", tuple(float(value) for value in self.scale))

        expected = len(self.feature_names)
        if len(self.mean) != expected or len(self.scale) != expected:
            raise ValueError(
                f"Feature names, means and scales have different length: "
                f"{expected}, {len(self.mean)}, {len(self.scale)}"
            )
        if not all(np.isfinite(self.mean)) or not all(np.isfinite(self.scale)):
            raise ValueError("Feature contract contains non-finite scaling values")
        if any(value <= 0 for value in self.scale):
            raise ValueError("Feature contract scale values must be positive")

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The ordered model input: flow fields first, then the window block."""

        return self.flow_features + self.window_features

    def to_manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "features": list(self.feature_names),
            "flow_features": list(self.flow_features),
            "window_features": list(self.window_features),
            "window_seconds": list(self.window_seconds),
            "window_features_disabled_reason": self.window_features_disabled_reason,
            "transform": self.transform,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "missing_value_policy": self.missing_value_policy,
        }

    @classmethod
    def from_manifest(cls, block: dict[str, Any]) -> "FeatureContract":
        """Rebuild a contract, failing closed on anything this runtime cannot honour."""

        version = block.get("version")
        if version not in COMPATIBLE_CONTRACT_VERSIONS:
            raise ContractMismatchError(
                f"Unsupported feature contract version {version!r}; this runtime "
                f"expects {FEATURE_CONTRACT_VERSION} and accepts "
                f"{sorted(COMPATIBLE_CONTRACT_VERSIONS)}. Retrain to produce a "
                "current bundle."
            )

        transform = block.get("transform")
        if transform != TRANSFORM:
            raise ContractMismatchError(
                f"Unsupported feature transform {transform!r}; runtime implements "
                f"{TRANSFORM!r}"
            )

        policy = block.get("missing_value_policy", MISSING_VALUE_POLICY)
        if policy != MISSING_VALUE_POLICY:
            raise ContractMismatchError(
                f"Unsupported missing value policy {policy!r}; runtime implements "
                f"{MISSING_VALUE_POLICY!r}"
            )

        return cls(
            flow_features=tuple(block.get("flow_features", block.get("features", ()))),
            window_features=tuple(block.get("window_features", ())),
            window_seconds=tuple(block.get("window_seconds", ())),
            window_features_disabled_reason=block.get("window_features_disabled_reason"),
            mean=tuple(block["mean"]),
            scale=tuple(block["scale"]),
            version=version,
            transform=transform,
            missing_value_policy=policy,
        )

    def select(self, flows: pd.DataFrame) -> np.ndarray:
        """Return the contract's features as a float32 matrix, or raise.

        Missing and non-finite values are errors, never silently zero-filled,
        so an upload that cannot be scored honestly is refused instead.
        """

        missing = [name for name in self.feature_names if name not in flows.columns]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "..." if len(missing) > 8 else ""
            raise ValueError(f"Input is missing required features: {preview}{suffix}")

        numeric = flows.loc[:, list(self.feature_names)].apply(
            pd.to_numeric, errors="coerce"
        )
        values = numeric.to_numpy(dtype=np.float32, copy=True)
        invalid = ~np.isfinite(values)
        if invalid.any():
            bad = np.asarray(self.feature_names)[np.unique(np.where(invalid)[1])]
            preview = ", ".join(bad[:8])
            raise ValueError(
                f"Input contains missing or non-finite values in: {preview}"
            )
        return values

    def standardize(self, transformed: np.ndarray) -> np.ndarray:
        """Centre and scale values that have already been signed-log1p'd.

        Training fits its scaler on signed-log1p values while streaming, so it
        reaches this half on its own; everything else goes through
        transform_values. Both share these fitted constants.
        """

        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        return ((transformed - mean) / scale).astype(np.float32, copy=False)

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        """Apply the fitted transform. The only implementation of it."""

        return self.standardize(_signed_log1p(values))

    def with_window_features(self, flows: pd.DataFrame) -> pd.DataFrame:
        """Attach the declared window block, unless the caller already did.

        Training joins a precomputed sidecar and passes the result through, so
        columns already present are trusted rather than recomputed. That is what
        keeps the trainer and the runtime on one set of values.
        """

        if not self.window_features:
            return flows
        if all(name in flows.columns for name in self.window_features):
            return flows

        available, missing = window_inputs_available(flows)
        if not available:
            raise ValueError(
                "This bundle declares a window feature block, so scoring needs "
                "the flow identity columns it is computed from. Missing: "
                + ", ".join(missing)
            )
        computed = compute_window_features(flows)
        return pd.concat([flows, computed], axis=1)

    def prepare(self, flows: pd.DataFrame) -> np.ndarray:
        """Normalize, validate and transform a batch into model input."""

        normalized = self.with_window_features(normalize_columns(flows))
        return self.transform_values(self.select(normalized))


def epoch_seconds(timestamps: pd.Series) -> pd.Series:
    """Convert parsed timestamps to float seconds since the epoch.

    Window bucketing divides by this, so it must not assume a resolution.
    pandas resolves datetimes to datetime64[us] in 3.x and datetime64[ns] in
    2.x, so a hardcoded division by 1e9 is wrong on one of them and would
    silently misbucket every row rather than raising.

    NaT maps to NaN, not to the int64 sentinel that a bare astype produces.
    Without that, an unparseable timestamp becomes a large negative epoch that
    reads as a perfectly valid row and lands in a bucket of its own.

    Subtracting the epoch rather than casting to datetime64[ns] is what keeps
    the promise above. A nanosecond cast overflows outside roughly 1678-2262,
    so a single out-of-range date in a capture would raise and abort the whole
    file, which is exactly what the NaN-preserving contract exists to prevent.
    """

    return (timestamps - pd.Timestamp("1970-01-01")).dt.total_seconds()


__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "COLUMN_ALIASES",
    "ContractMismatchError",
    "FeatureContract",
    "check_bundle_version",
    "COMPATIBLE_CONTRACT_VERSIONS",
    "FEATURE_CONTRACT_VERSION",
    "IDENTITY_COLUMNS",
    "MISSING_VALUE_POLICY",
    "TRANSFORM",
    "WINDOW_FEATURE_NAMES",
    "WINDOW_INPUT_COLUMNS",
    "WINDOW_SECONDS",
    "WINDOW_SOURCE_COLUMNS",
    "build_window_sidecar",
    "canonical_name",
    "compute_window_features",
    "epoch_seconds",
    "normalize_columns",
    "parse_timestamps",
    "resolve_window_cache_dir",
    "window_coverage_warning",
    "window_inputs_available",
]
