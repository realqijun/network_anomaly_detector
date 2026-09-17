"""The authoritative, versioned flow feature contract.

Training and inference both import this module. Neither computes a feature
value or a transform itself, so there is exactly one definition of what a
feature is called, what units it carries, and how it is scaled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


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


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse capture timestamps with an explicit format, never by guessing.

    These CSVs carry a local-time string with no zone, and pandas will happily
    read 03/01/2018 as 3 January under `dayfirst=True` and as 1 March without
    it. Either choice silently reorders every window bucket, so this function
    accepts a format only if it parses every value, and refuses when two
    formats both parse and disagree.
    """

    text = values.astype("string").str.strip()
    present = text.notna() & text.ne("")
    if not present.any():
        raise ValueError("Timestamp column contains no parseable values")

    candidates: dict[str, pd.Series] = {}
    for fmt in _TIMESTAMP_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if parsed[present].notna().all():
            candidates[fmt] = parsed

    if not candidates:
        sample = text[present].iloc[0]
        raise ValueError(
            f"Could not parse Timestamp with any known format; first value is {sample!r}"
        )

    resolved = list(candidates.items())
    reference_format, reference = resolved[0]
    for other_format, other in resolved[1:]:
        if not reference[present].equals(other[present]):
            disagreement = text[present][
                reference[present].ne(other[present])
            ].iloc[0]
            raise ValueError(
                "Timestamp day/month order is ambiguous: "
                f"{disagreement!r} parses under both {reference_format!r} and "
                f"{other_format!r} with different results. Normalize the capture "
                "to ISO 8601 before training or scoring."
            )
    return reference


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

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        """Apply the fitted transform. The only implementation of it."""

        transformed = _signed_log1p(values)
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        return ((transformed - mean) / scale).astype(np.float32, copy=False)

    def prepare(self, flows: pd.DataFrame) -> np.ndarray:
        """Normalize, validate and transform a batch into model input."""

        normalized = normalize_columns(flows)
        return self.transform_values(self.select(normalized))


def epoch_seconds(timestamps: pd.Series) -> pd.Series:
    """Convert parsed timestamps to float seconds since the epoch.

    Window bucketing divides by this, so it must not assume a resolution.
    pandas resolves datetimes to datetime64[us] in 3.x and datetime64[ns] in
    2.x, so a hardcoded division by 1e9 is wrong on one of them and would
    silently misbucket every row rather than raising.
    """

    return timestamps.astype("datetime64[ns]").astype("int64") / 1e9


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
    "WINDOW_SOURCE_COLUMNS",
    "epoch_seconds",
    "normalize_columns",
    "parse_timestamps",
]
