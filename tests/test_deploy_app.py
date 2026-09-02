import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

from app import create_app
from detector_runtime import HybridDetectorNetwork


def _write_bundle(directory: Path) -> None:
    model = HybridDetectorNetwork(
        input_dim=2,
        num_classes=2,
        hidden_dims=(4,),
        latent_dim=2,
        dropout=0.0,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(model.state_dict(), directory / "model.pt")
    manifest = {
        "bundle_format_version": 1,
        "feature_contract": {
            "features": ["Flow Duration", "Tot Fwd Pkts"],
            "transform": "signed_log1p_standardize",
            "mean": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "missing_value_policy": "reject",
        },
        "taxonomy": {
            "classes": ["Benign", "DDoS"],
            "reportable": {"Benign": True, "DDoS": True},
            "unknown_label": "Unknown anomaly",
        },
        "model": {
            "hidden_dims": [4],
            "latent_dim": 2,
            "dropout": 0.0,
        },
        "thresholds": {
            "anomaly": 0.1,
            "class_confidence": {"Benign": 0.0, "DDoS": 0.9},
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def _post_csv(client, csv_bytes: bytes):
    return client.post(
        "/predict",
        data={"file": (io.BytesIO(csv_bytes), "flows.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


class DeployAppPredictTests(unittest.TestCase):
    def test_predict_scores_uploaded_csv_through_the_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()

            response = _post_csv(
                client, b"Flow Duration,Tot Fwd Pkts\n0.0,0.0\n2.0,2.0\n"
            )

            body = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Benign", body)
            self.assertIn("Unknown anomaly", body)

    def test_predict_flashes_error_for_missing_contract_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()

            response = _post_csv(client, b"Flow Duration\n1.0\n")

            body = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn("missing required features", body)


class DeployAppFailClosedTests(unittest.TestCase):
    def test_create_app_serves_but_refuses_predict_when_bundle_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing_bundle = Path(temporary) / "does-not-exist"
            client = create_app(missing_bundle).test_client()

            response = _post_csv(client, b"Flow Duration,Tot Fwd Pkts\n0.0,0.0\n")

            body = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Detector not available", body)

    def test_create_app_serves_but_refuses_predict_when_weights_do_not_match_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            # Overwrite model.pt with weights shaped for a different input_dim,
            # so torch's load_state_dict raises RuntimeError, not one of the
            # exception types Detector's own validation raises deliberately.
            mismatched = HybridDetectorNetwork(
                input_dim=5, num_classes=2, hidden_dims=(4,), latent_dim=2, dropout=0.0
            )
            torch.save(mismatched.state_dict(), bundle / "model.pt")

            client = create_app(bundle).test_client()
            response = _post_csv(client, b"Flow Duration,Tot Fwd Pkts\n0.0,0.0\n")

            body = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Detector not available", body)

    def test_pcap_upload_surfaces_processing_errors_instead_of_claiming_no_flows(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()

            with patch(
                "app.parse_pcap_to_dataframe",
                side_effect=RuntimeError("PCAP processing requires Docker support"),
            ):
                response = client.post(
                    "/predict",
                    data={"file": (io.BytesIO(b"pcap-bytes"), "capture.pcap")},
                    content_type="multipart/form-data",
                    follow_redirects=True,
                )

            body = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn("PCAP processing requires Docker support", body)
            self.assertNotIn("no valid network flows were extracted", body.lower())


if __name__ == "__main__":
    unittest.main()
