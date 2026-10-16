import io
import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd
from flask import Flask

from capture_scoring import PacketRecord, assign_flow_scores_to_packets
from pcap_parser import FlowBatch
from wireshark_api import create_wireshark_blueprint


FLOW_KEY = ("10.0.0.1", 49152, "10.0.0.2", 443, 6)


class CaptureScoringTests(unittest.TestCase):
    def test_repeated_five_tuple_is_split_using_flow_packet_counts(self):
        batch = FlowBatch(
            features=pd.DataFrame(
                {
                    "Total Fwd Packets": [2, 1],
                    "Total Backward Packets": [1, 1],
                }
            ),
            metadata=pd.DataFrame(
                {
                    "source_ip": ["10.0.0.1", "10.0.0.1"],
                    "destination_ip": ["10.0.0.2", "10.0.0.2"],
                    "source_port": [49152, 49152],
                    "destination_port": [443, 443],
                    "protocol": [6, 6],
                    "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:01:00"],
                }
            ),
        )
        scores = pd.DataFrame(
            {
                "prediction": ["Benign", "DDoS"],
                "confidence": [0.9, 0.95],
                "anomaly_score": [0.1, 4.2],
                "is_anomaly": [False, True],
            }
        )
        packets = [PacketRecord(index, FLOW_KEY) for index in range(1, 6)]

        result = assign_flow_scores_to_packets(packets, batch, scores)

        self.assertEqual([row["flow_id"] for row in result], [
            "flow-1",
            "flow-1",
            "flow-1",
            "flow-2",
            "flow-2",
        ])
        self.assertEqual([row["is_anomaly"] for row in result], [
            False,
            False,
            False,
            True,
            True,
        ])

    def test_reverse_direction_packet_matches_the_same_flow(self):
        batch = FlowBatch(
            features=pd.DataFrame(
                {"Total Fwd Packets": [1], "Total Backward Packets": [1]}
            ),
            metadata=pd.DataFrame(
                {
                    "source_ip": ["10.0.0.2"],
                    "destination_ip": ["10.0.0.1"],
                    "source_port": [443],
                    "destination_port": [49152],
                    "protocol": ["TCP"],
                    "timestamp": ["2026-01-01 00:00:00"],
                }
            ),
        )
        scores = pd.DataFrame(
            {
                "prediction": ["Unknown anomaly"],
                "confidence": [float("nan")],
                "anomaly_score": [1.5],
                "is_anomaly": [True],
            }
        )

        result = assign_flow_scores_to_packets(
            [PacketRecord(7, FLOW_KEY)], batch, scores
        )

        self.assertEqual(result[0]["frame_number"], 7)
        self.assertIsNone(result[0]["confidence"])

    def test_timestamp_selects_a_later_reused_five_tuple(self):
        batch = FlowBatch(
            features=pd.DataFrame(
                {
                    "Total Fwd Packets": [99, 1],
                    "Total Backward Packets": [0, 0],
                    "Flow Duration": [1_000_000, 1_000_000],
                }
            ),
            metadata=pd.DataFrame(
                {
                    "source_ip": ["10.0.0.1", "10.0.0.1"],
                    "destination_ip": ["10.0.0.2", "10.0.0.2"],
                    "source_port": [49152, 49152],
                    "destination_port": [443, 443],
                    "protocol": [6, 6],
                    "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:01:00"],
                }
            ),
        )
        scores = pd.DataFrame(
            {
                "prediction": ["Benign", "DDoS"],
                "confidence": [0.9, 0.95],
                "anomaly_score": [0.1, 4.2],
                "is_anomaly": [False, True],
            }
        )
        later_timestamp = datetime.fromisoformat("2026-01-01 00:01:00").timestamp()

        result = assign_flow_scores_to_packets(
            [PacketRecord(20, FLOW_KEY, later_timestamp)], batch, scores
        )

        self.assertEqual(result[0]["flow_id"], "flow-2")
        self.assertTrue(result[0]["is_anomaly"])


class WiresharkApiTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(create_wireshark_blueprint(lambda: object()))
        app.config.update(TESTING=True)
        self.client = app.test_client()

    @patch("wireshark_api.score_capture_file")
    def test_capture_endpoint_returns_packet_annotations_as_tsv(self, score_capture):
        score_capture.return_value = [
            {
                "frame_number": 12,
                "flow_id": "flow-1",
                "anomaly_score": 2.25,
                "is_anomaly": True,
                "prediction": "Unknown anomaly",
                "confidence": None,
            }
        ]

        response = self.client.post(
            "/api/v1/captures/score",
            data={
                "capture": (io.BytesIO(b"pcap bytes"), "capture.pcapng"),
                "frames": (io.BytesIO(b"12\n"), "frames.txt"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/tab-separated-values")
        self.assertIn(b"12\tflow-1\t2.25\ttrue\tUnknown anomaly\t", response.data)
        score_capture.assert_called_once()

    def test_capture_endpoint_rejects_invalid_frame_manifest(self):
        response = self.client.post(
            "/api/v1/captures/score",
            data={
                "capture": (io.BytesIO(b"pcap bytes"), "capture.pcap"),
                "frames": (io.BytesIO(b"not-a-number\n"), "frames.txt"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("one integer per line", response.json["error"])


if __name__ == "__main__":
    unittest.main()
