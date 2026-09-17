"""Evidence for PCAP triage, gathered after model scoring.

A flow anomaly score prioritizes traffic. An investigation, and a packet
analysis CTF, needs the follow-up: which packets, which stream, what was
actually requested. This module is the second stage, deliberately kept apart
from the detector so an anomaly claim and the observations backing it never
get merged into one number.

Payload bytes are measured and parsed, never retained.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_MAX_PACKETS_PER_FLOW = 50
_MAX_HTTP_RECORDS_PER_FLOW = 20
_HTTP_METHODS = (
    b"GET", b"POST", b"HEAD", b"PUT", b"DELETE", b"OPTIONS", b"PATCH", b"TRACE", b"CONNECT",
)


@dataclass(frozen=True)
class FlowKey:
    """A direction-normalized 5-tuple.

    Both halves of one conversation map to the same key, so evidence for a
    request and its response is gathered together rather than split.
    """

    low_address: str
    low_port: int
    high_address: str
    high_port: int
    protocol: int

    @classmethod
    def build(
        cls, source: str, source_port: int, destination: str, destination_port: int, protocol: int
    ) -> "FlowKey":
        first = (str(source), int(source_port))
        second = (str(destination), int(destination_port))
        low, high = sorted([first, second])
        return cls(low[0], low[1], high[0], high[1], int(protocol))

    def as_text(self) -> str:
        return (
            f"{self.low_address}:{self.low_port}-"
            f"{self.high_address}:{self.high_port}/{self.protocol}"
        )

    @classmethod
    def parse(cls, text: str) -> "FlowKey":
        try:
            endpoints, protocol = text.rsplit("/", 1)
            left, right = endpoints.split("-", 1)
            source, source_port = left.rsplit(":", 1)
            destination, destination_port = right.rsplit(":", 1)
        except ValueError as error:
            raise ValueError(
                f"Could not parse flow key {text!r}; expected "
                "address:port-address:port/protocol"
            ) from error
        return cls.build(
            source, int(source_port), destination, int(destination_port), int(protocol)
        )


@dataclass
class FlowEvidence:
    """What was observed for one flow, separate from what was scored."""

    key: FlowKey
    stream_id: int
    packet_indices: list[int] = field(default_factory=list)
    packet_count: int = 0
    first_seen: float | None = None
    last_seen: float | None = None
    http_requests: list[dict[str, Any]] = field(default_factory=list)
    http_responses: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow": self.key.as_text(),
            "stream_id": self.stream_id,
            "packet_indices": list(self.packet_indices),
            "packet_count": self.packet_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "http_requests": list(self.http_requests),
            "http_responses": list(self.http_responses),
        }


def _header_value(headers: dict[str, str], name: str) -> str | None:
    return headers.get(name.lower())


def _parse_headers(lines: Sequence[bytes]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in lines:
        if b":" not in line:
            continue
        name, _, value = line.partition(b":")
        headers[name.decode("latin-1").strip().lower()] = value.decode("latin-1").strip()
    return headers


def _parse_http(payload: bytes, packet_index: int) -> tuple[dict | None, dict | None]:
    """Recover a request or response head from one packet's plaintext payload.

    Only a payload that begins a message is parsed. Reassembling a head split
    across segments is deliberately out of scope here.
    """

    if not payload:
        return None, None
    head = payload.split(b"\r\n\r\n", 1)[0]
    lines = head.split(b"\r\n")
    if not lines:
        return None, None

    start = lines[0]
    headers = _parse_headers(lines[1:])

    if start.startswith(b"HTTP/"):
        parts = start.split(b" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            return None, None
        length = _header_value(headers, "content-length")
        return None, {
            "packet_index": packet_index,
            "status": int(parts[1]),
            "content_type": _header_value(headers, "content-type"),
            "content_length": int(length) if length and length.isdigit() else None,
        }

    if any(start.startswith(method + b" ") for method in _HTTP_METHODS):
        parts = start.split(b" ")
        if len(parts) < 3:
            return None, None
        return {
            "packet_index": packet_index,
            "method": parts[0].decode("latin-1"),
            "path": parts[1].decode("latin-1"),
            "host": _header_value(headers, "host"),
            "user_agent": _header_value(headers, "user-agent"),
        }, None

    return None, None


def collect_evidence(
    pcap_path: str | Path,
    flow_keys: Iterable[FlowKey] | None = None,
    max_packets_per_flow: int = DEFAULT_MAX_PACKETS_PER_FLOW,
) -> dict[FlowKey, FlowEvidence]:
    """Gather packet-level evidence for the given flows.

    `flow_keys=None` means every flow in the capture. A requested flow that is
    absent yields no entry, rather than a fabricated empty one.
    """

    # Imported lazily so a missing scapy degrades the evidence stage rather
    # than breaking inference, which does not need it.
    from scapy.all import IP, IPv6, TCP, UDP, PcapReader, Raw

    wanted = set(flow_keys) if flow_keys is not None else None
    found: dict[FlowKey, FlowEvidence] = {}

    with PcapReader(str(pcap_path)) as reader:
        for index, packet in enumerate(reader):
            network = packet.getlayer(IP) or packet.getlayer(IPv6)
            if network is None:
                continue
            transport = packet.getlayer(TCP) or packet.getlayer(UDP)
            if transport is None:
                continue

            protocol = 6 if packet.haslayer(TCP) else 17
            key = FlowKey.build(
                network.src, transport.sport, network.dst, transport.dport, protocol
            )
            if wanted is not None and key not in wanted:
                continue

            evidence = found.get(key)
            if evidence is None:
                evidence = FlowEvidence(key=key, stream_id=len(found))
                found[key] = evidence

            evidence.packet_count += 1
            if len(evidence.packet_indices) < max_packets_per_flow:
                evidence.packet_indices.append(index)

            moment = float(packet.time)
            if evidence.first_seen is None or moment < evidence.first_seen:
                evidence.first_seen = moment
            if evidence.last_seen is None or moment > evidence.last_seen:
                evidence.last_seen = moment

            if protocol == 6 and packet.haslayer(Raw):
                request, response = _parse_http(bytes(packet[Raw].load), index)
                if request and len(evidence.http_requests) < _MAX_HTTP_RECORDS_PER_FLOW:
                    evidence.http_requests.append(request)
                if response and len(evidence.http_responses) < _MAX_HTTP_RECORDS_PER_FLOW:
                    evidence.http_responses.append(response)

    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evidence",
        description=(
            "Gather packet evidence for flows in a capture. Prints JSON so a "
            "triage result can be joined back to concrete observations."
        ),
    )
    parser.add_argument("pcap", help="Path to a pcap or pcapng file")
    parser.add_argument(
        "--flow",
        action="append",
        default=None,
        metavar="A:PORT-B:PORT/PROTO",
        help="Restrict to one flow; repeatable. Omit for every flow.",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=DEFAULT_MAX_PACKETS_PER_FLOW,
        help="Cap on packet indices kept per flow",
    )
    arguments = parser.parse_args(argv)

    keys = [FlowKey.parse(text) for text in arguments.flow] if arguments.flow else None
    evidence = collect_evidence(arguments.pcap, keys, arguments.max_packets)
    json.dump([value.to_dict() for value in evidence.values()], sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


__all__ = [
    "DEFAULT_MAX_PACKETS_PER_FLOW",
    "FlowEvidence",
    "FlowKey",
    "collect_evidence",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
