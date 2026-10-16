import io
import unittest
from pathlib import Path

from anomaly_detector_service import EXPECTED_FEATURES_CICFLOWMETER
from app import app, detector


class DeploymentFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_application_starts_with_detector_loaded(self):
        response = self.client.get("/")
        health = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json, {"status": "ok"})
        self.assertIsNotNone(detector)
        self.assertNotIn(b"Detector not loaded", response.data)

    def test_csv_upload_returns_predictions(self):
        csv_data = ",".join(EXPECTED_FEATURES_CICFLOWMETER) + "\n"
        csv_data += ",".join("0" for _ in EXPECTED_FEATURES_CICFLOWMETER) + "\n"

        response = self.client.post(
            "/predict",
            data={"file": (io.BytesIO(csv_data.encode()), "flows.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Anomaly Detection Results", response.data)
        self.assertIn(b"Processed 1 samples", response.data)

    def test_pcap_upload_returns_flow_predictions(self):
        fixture = Path(__file__).resolve().parents[1] / "dhcp.pcap"

        with fixture.open("rb") as capture:
            response = self.client.post(
                "/predict",
                data={"file": (capture, fixture.name)},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Anomaly Detection Results", response.data)
        self.assertIn(b"Processed 2 samples", response.data)
        self.assertNotIn(b"no valid network flows", response.data.lower())


if __name__ == "__main__":
    unittest.main()
