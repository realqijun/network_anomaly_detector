"""HTTP adapter used by the Wireshark Lua plugin."""

from __future__ import annotations

import csv
import io
import os
import tempfile
from collections.abc import Callable

from flask import Blueprint, Response, jsonify, request

from capture_scoring import score_capture_file


TSV_COLUMNS = (
    "frame_number",
    "flow_id",
    "anomaly_score",
    "is_anomaly",
    "prediction",
    "confidence",
)


def _parse_frame_manifest(raw: bytes) -> list[int]:
    try:
        frames = [int(line.strip()) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Frame manifest must contain one integer per line") from exc
    if not frames:
        raise ValueError("Frame manifest is empty")
    if any(frame <= 0 for frame in frames):
        raise ValueError("Frame numbers must be positive")
    if len(set(frames)) != len(frames):
        raise ValueError("Frame numbers must be unique")
    return frames


def _render_tsv(rows: list[dict[str, object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=TSV_COLUMNS,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        serialized = dict(row)
        serialized["is_anomaly"] = "true" if row["is_anomaly"] else "false"
        serialized["confidence"] = "" if row["confidence"] is None else row["confidence"]
        writer.writerow(serialized)
    return output.getvalue()


def create_wireshark_blueprint(
    detector_provider: Callable[[], object | None],
) -> Blueprint:
    blueprint = Blueprint("wireshark_api", __name__)

    @blueprint.post("/api/v1/captures/score")
    def score_capture() -> Response | tuple[Response, int]:
        detector = detector_provider()
        if detector is None:
            return jsonify(error="Detector is not available"), 503
        capture = request.files.get("capture")
        frame_manifest = request.files.get("frames")
        if capture is None or frame_manifest is None:
            return jsonify(error="Both capture and frames files are required"), 400

        suffix = os.path.splitext(capture.filename or "capture.pcapng")[1].lower()
        if suffix not in {".pcap", ".pcapng"}:
            return jsonify(error="Capture must be a PCAP or PCAPNG file"), 400

        capture_path = None
        try:
            frame_numbers = _parse_frame_manifest(frame_manifest.read())
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                capture.save(temporary.name)
                capture_path = temporary.name
            rows = score_capture_file(detector, capture_path, frame_numbers)
            return Response(_render_tsv(rows), mimetype="text/tab-separated-values")
        except ValueError as exc:
            return jsonify(error=str(exc)), 422
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 500
        finally:
            if capture_path and os.path.exists(capture_path):
                os.remove(capture_path)

    return blueprint


__all__ = ["TSV_COLUMNS", "create_wireshark_blueprint"]
