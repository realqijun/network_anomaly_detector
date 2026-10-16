"""Map flow-level detector results back to packet evidence in a capture."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from pcap_parser import FlowBatch, parse_pcap_to_flow_batch


@dataclass(frozen=True)
class PacketRecord:
    frame_number: int
    flow_key: tuple[str, int, str, int, int]
    timestamp: float | None = None


def _normalize_ip(value: object) -> str:
    return str(ip_address(str(value).strip()))


def _normalize_protocol(value: object) -> int:
    text = str(value).strip().casefold()
    aliases = {"tcp": 6, "udp": 17, "icmp": 1, "icmpv6": 58}
    if text in aliases:
        return aliases[text]
    return int(float(text))


def _canonical_flow_key(
    source_ip: object,
    source_port: object,
    destination_ip: object,
    destination_port: object,
    protocol: object,
) -> tuple[str, int, str, int, int]:
    first = (_normalize_ip(source_ip), int(float(source_port)))
    second = (_normalize_ip(destination_ip), int(float(destination_port)))
    low, high = sorted((first, second))
    return low[0], low[1], high[0], high[1], _normalize_protocol(protocol)


def read_packet_records(
    capture_path: str | Path,
    frame_numbers: Sequence[int],
) -> list[PacketRecord]:
    """Read packet identities in capture order while retaining Wireshark numbers."""

    try:
        from scapy.all import IP, IPv6, TCP, UDP, PcapReader
    except ImportError as exc:
        raise RuntimeError("Scapy is required to map flow scores to packets") from exc

    records: list[PacketRecord] = []
    packet_count = 0
    with PcapReader(str(capture_path)) as packets:
        for packet_count, packet in enumerate(packets, start=1):
            if packet_count > len(frame_numbers):
                raise ValueError("Capture contains more packets than the frame manifest")

            if IP in packet:
                network = packet[IP]
            elif IPv6 in packet:
                network = packet[IPv6]
            else:
                continue

            source_port = 0
            destination_port = 0
            if TCP in packet:
                source_port = int(packet[TCP].sport)
                destination_port = int(packet[TCP].dport)
                protocol = 6
            elif UDP in packet:
                source_port = int(packet[UDP].sport)
                destination_port = int(packet[UDP].dport)
                protocol = 17
            else:
                protocol = int(network.nh if IPv6 in packet else network.proto)

            records.append(
                PacketRecord(
                    frame_number=int(frame_numbers[packet_count - 1]),
                    flow_key=_canonical_flow_key(
                        network.src,
                        source_port,
                        network.dst,
                        destination_port,
                        protocol,
                    ),
                    timestamp=float(packet.time),
                )
            )

    if packet_count != len(frame_numbers):
        raise ValueError(
            "Frame manifest length does not match the number of captured packets"
        )
    return records


def _score_flows(detector: object, features: pd.DataFrame) -> pd.DataFrame:
    if hasattr(detector, "detect"):
        scored = detector.detect(features)
        required = {"prediction", "anomaly_score", "is_anomaly"}
        missing = required.difference(scored.columns)
        if missing:
            raise ValueError(
                "Detector result is missing columns: " + ", ".join(sorted(missing))
            )
        if "confidence" not in scored:
            scored = scored.assign(confidence=np.nan)
        return scored.reset_index(drop=True)

    if hasattr(detector, "predict"):
        predictions, scores = detector.predict(features)
        predictions = np.asarray(predictions)
        scores = np.asarray(scores, dtype=float)
        if len(predictions) != len(features) or len(scores) != len(features):
            raise ValueError("Detector returned a different number of rows than it received")
        is_anomaly = predictions.astype(bool)
        return pd.DataFrame(
            {
                "prediction": np.where(is_anomaly, "Unknown anomaly", "Benign"),
                "confidence": np.nan,
                "anomaly_score": scores,
                "is_anomaly": is_anomaly,
            }
        )

    raise TypeError("Detector must expose detect(DataFrame) or predict(DataFrame)")


def _flow_start_sort_value(value: object) -> tuple[int, object]:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return 1, str(value)
    return 0, int(parsed.value)


def _flow_time_bounds(
    started_at: object,
    duration_microseconds: object,
) -> tuple[float, float] | None:
    parsed = pd.to_datetime(started_at, errors="coerce")
    duration = pd.to_numeric(pd.Series([duration_microseconds]), errors="coerce").iloc[0]
    if pd.isna(parsed) or pd.isna(duration):
        return None
    start = float(parsed.to_pydatetime().timestamp())
    return start, start + max(0.0, float(duration) / 1_000_000)


def assign_flow_scores_to_packets(
    packet_records: Iterable[PacketRecord],
    flow_batch: FlowBatch,
    flow_scores: pd.DataFrame,
) -> list[dict[str, object]]:
    """Assign each packet the result for its bidirectional CIC flow."""

    if len(flow_batch.features) != len(flow_batch.metadata):
        raise ValueError("Flow feature and metadata row counts do not match")
    if len(flow_scores) != len(flow_batch.features):
        raise ValueError("Flow score and feature row counts do not match")

    flows_by_key: dict[tuple[str, int, str, int, int], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for index, metadata in flow_batch.metadata.reset_index(drop=True).iterrows():
        key = _canonical_flow_key(
            metadata["source_ip"],
            metadata["source_port"],
            metadata["destination_ip"],
            metadata["destination_port"],
            metadata["protocol"],
        )
        expected_packets = int(
            float(flow_batch.features.iloc[index]["Total Fwd Packets"])
            + float(flow_batch.features.iloc[index]["Total Backward Packets"])
        )
        bounds = _flow_time_bounds(
            metadata.get("timestamp", ""),
            flow_batch.features.iloc[index].get("Flow Duration", np.nan),
        )
        score = flow_scores.iloc[index]
        flows_by_key[key].append(
            {
                "index": index,
                "remaining": max(1, expected_packets),
                "started_at": metadata.get("timestamp", ""),
                "bounds": bounds,
                "score": score,
            }
        )

    for candidates in flows_by_key.values():
        candidates.sort(key=lambda item: _flow_start_sort_value(item["started_at"]))

    cursors: dict[tuple[str, int, str, int, int], int] = defaultdict(int)
    annotations: list[dict[str, object]] = []
    for packet in packet_records:
        candidates = flows_by_key.get(packet.flow_key)
        if not candidates:
            continue

        cursor = cursors[packet.flow_key]
        if packet.timestamp is not None:
            for candidate_index in range(cursor, len(candidates)):
                bounds = candidates[candidate_index]["bounds"]
                if bounds is None:
                    continue
                start, end = bounds
                if start - 0.001 <= packet.timestamp <= end + 0.001:
                    cursor = candidate_index
                    break
        while cursor < len(candidates) and int(candidates[cursor]["remaining"]) <= 0:
            cursor += 1
        if cursor >= len(candidates):
            continue
        cursors[packet.flow_key] = cursor

        flow = candidates[cursor]
        flow["remaining"] = int(flow["remaining"]) - 1
        score = flow["score"]
        confidence = score["confidence"]
        annotations.append(
            {
                "frame_number": packet.frame_number,
                "flow_id": f"flow-{int(flow['index']) + 1}",
                "anomaly_score": float(score["anomaly_score"]),
                "is_anomaly": bool(score["is_anomaly"]),
                "prediction": str(score["prediction"]),
                "confidence": None if pd.isna(confidence) else float(confidence),
            }
        )
    return annotations


def score_capture_file(
    detector: object,
    capture_path: str | Path,
    frame_numbers: Sequence[int],
    *,
    flow_parser: Callable[[str | Path], FlowBatch] = parse_pcap_to_flow_batch,
    packet_reader: Callable[[str | Path, Sequence[int]], list[PacketRecord]] = (
        read_packet_records
    ),
) -> list[dict[str, object]]:
    flow_batch = flow_parser(capture_path)
    if flow_batch.features.empty:
        return []
    flow_scores = _score_flows(detector, flow_batch.features)
    packet_records = packet_reader(capture_path, frame_numbers)
    return assign_flow_scores_to_packets(packet_records, flow_batch, flow_scores)


__all__ = [
    "PacketRecord",
    "assign_flow_scores_to_packets",
    "read_packet_records",
    "score_capture_file",
]
