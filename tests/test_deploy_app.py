import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

from app import _resolve_bundle_dir, create_app
from detector_runtime import HybridDetectorNetwork
from tests.bundle_fixture import write_bundle as _write_bundle

# A capture the window block can actually be computed from: two sources onto
# one destination inside one second. Identity columns are not contract feature
# names, so a CSV without them would never exercise the window path.
WINDOW_CSV = (
    b"Flow Duration,Tot Fwd Pkts,Timestamp,Src IP,Dst IP,Src Port,Dst Port,"
    b"Protocol,SYN Flag Cnt,ACK Flag Cnt,TotLen Fwd Pkts,TotLen Bwd Pkts,Tot Bwd Pkts\n"
    b"0.0,0.0,2018-02-14 08:00:00,10.0.0.1,10.0.0.9,1111,80,6,1,0,100,0,0\n"
    b"9.0,9.0,2018-02-14 08:00:00,10.0.0.2,10.0.0.9,2222,80,6,1,0,100,0,0\n"
)


def _post_csv(client, csv_bytes: bytes):
    return client.post(
        "/predict",
        data={"file": (io.BytesIO(csv_bytes), "flows.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


class DeployAppPredictTests(unittest.TestCase):
    def test_relative_bundle_paths_resolve_under_deploy_directory(self):
        resolved = Path(_resolve_bundle_dir("model_bundle"))

        self.assertEqual(resolved.name, "model_bundle")
        self.assertEqual(resolved.parent.name, "deploy")

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
            self.assertIn("n/a", body)

    def test_a_flagged_row_carries_its_contributing_features_to_the_page(self):
        """The end-to-end path: bundle load, score, attribute, render."""

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()

            response = _post_csv(
                client, b"Flow Duration,Tot Fwd Pkts\n0.0,0.0\n9.0,9.0\n"
            )

            body = response.get_data(as_text=True)
            self.assertIn("Top contributing features", body)
            self.assertIn("Flow Duration", body)
            self.assertIn("standard", body)

    def test_scores_a_window_bundle_end_to_end_from_identity_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle, with_window_block=True)
            client = create_app(bundle).test_client()

            response = _post_csv(client, WINDOW_CSV)

            body = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Flows scored", body)
            self.assertNotIn("missing required features", body)

    def test_a_window_bundle_refuses_a_csv_without_identity_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle, with_window_block=True)
            client = create_app(bundle).test_client()

            response = _post_csv(
                client, b"Flow Duration,Tot Fwd Pkts\n0.0,0.0\n2.0,2.0\n"
            )

            self.assertIn("Dst IP", response.get_data(as_text=True))

    def test_reports_missing_window_coverage_rather_than_hiding_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle, disabled_reason="capture files carry no addresses")
            client = create_app(bundle).test_client()

            response = _post_csv(
                client, b"Flow Duration,Tot Fwd Pkts\n0.0,0.0\n2.0,2.0\n"
            )

            body = response.get_data(as_text=True)
            self.assertIn("Coverage gaps", body)
            self.assertIn("capture files carry no addresses", body)

    def test_warns_when_a_capture_is_too_short_for_the_widest_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle, with_window_block=True)
            client = create_app(bundle).test_client()

            response = _post_csv(client, WINDOW_CSV)

            self.assertIn("Coverage gaps", response.get_data(as_text=True))

    def test_predict_flashes_error_for_missing_contract_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()

            response = _post_csv(client, b"Flow Duration\n1.0\n")

            body = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn("missing required features", body)

    def test_predict_rejects_pcap_uploads_when_the_runtime_has_not_enabled_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle, enable_pcap_uploads=False).test_client()

            response = client.post(
                "/predict",
                data={"file": (io.BytesIO(b"pcap-bytes"), "capture.pcap")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )

            body = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn("CSV uploads only", body)

    def test_requested_pcap_uploads_stay_disabled_when_runtime_probe_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            with patch.dict(os.environ, {"ENABLE_PCAP_UPLOADS": "1"}, clear=False), patch(
                "app.probe_pcap_runtime",
                return_value=(False, "Docker daemon is unavailable"),
            ):
                app = create_app(bundle)

        self.assertFalse(app.config["PCAP_UPLOADS_ENABLED"])


class DeployAppFailClosedTests(unittest.TestCase):
    def test_an_outdated_bundle_says_so_instead_of_looking_like_a_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            manifest = json.loads((bundle / "manifest.json").read_text())
            manifest["bundle_format_version"] = 1
            (bundle / "manifest.json").write_text(json.dumps(manifest))

            client = create_app(bundle).test_client()
            response = client.get("/", follow_redirects=True)

            self.assertIn("not compatible with this runtime", response.get_data(as_text=True))

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
                input_dim=7, num_classes=2, hidden_dims=(4,), latent_dim=2, dropout=0.0
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
            client = create_app(bundle, enable_pcap_uploads=True).test_client()

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


class UploadGuardTests(unittest.TestCase):
    """The limits that stand between an upload and the worker.

    Each of these was a surviving mutant: the guard could be deleted outright
    and the whole suite still passed.
    """

    def test_rejects_an_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()

            response = client.post(
                "/predict",
                data={"file": (io.BytesIO(b"Flow Duration,Tot Fwd Pkts\n1.0,1.0\n"), "flows.txt")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )

            body = response.get_data(as_text=True)
            self.assertIn("Unsupported file type", body)
            self.assertNotIn("Flows scored", body)

    def test_rejects_an_executable_masquerading_by_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()

            response = client.post(
                "/predict",
                data={"file": (io.BytesIO(b"\x7fELF\x02\x01\x01"), "payload.elf")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )

            self.assertIn("Unsupported file type", response.get_data(as_text=True))

    def test_a_binary_csv_is_refused_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()

            response = _post_csv(client, b"\xff\xfe\x00\x01\x02\x03")

            self.assertEqual(response.status_code, 200)
            self.assertIn("not readable as CSV text", response.get_data(as_text=True))

    def test_declares_an_upload_cap_the_worker_can_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            app = create_app(bundle)

            cap = app.config["MAX_CONTENT_LENGTH"]

            self.assertIsNotNone(cap)
            self.assertLessEqual(cap, 16 * 1024 * 1024)

    def test_an_upload_over_the_cap_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            app = create_app(bundle)
            app.config["MAX_CONTENT_LENGTH"] = 512
            client = app.test_client()

            oversized = b"Flow Duration,Tot Fwd Pkts\n" + b"1.0,1.0\n" * 500

            response = _post_csv(client, oversized)

            self.assertEqual(response.status_code, 413)

    def test_the_temporary_upload_is_always_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()
            staging = Path(tempfile.gettempdir())
            before = set(staging.glob("*.csv"))

            _post_csv(client, b"Flow Duration,Tot Fwd Pkts\n0.0,0.0\n")

            self.assertEqual(set(staging.glob("*.csv")) - before, set())

    def test_a_failed_save_leaves_no_temporary_file_behind(self):
        """The leak: tmp_path was bound after save(), so cleanup was skipped."""

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()
            staging = Path(tempfile.gettempdir())
            before = set(staging.glob("*.csv"))

            with patch(
                "werkzeug.datastructures.FileStorage.save",
                side_effect=OSError("No space left on device"),
            ):
                response = _post_csv(client, b"Flow Duration,Tot Fwd Pkts\n0.0,0.0\n")

            self.assertEqual(response.status_code, 200)
            self.assertIn("No space left on device", response.get_data(as_text=True))
            self.assertEqual(set(staging.glob("*.csv")) - before, set())


class ResultsRenderingTests(unittest.TestCase):
    def test_counts_every_flow_but_renders_a_bounded_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            app = create_app(bundle)
            app.config["MAX_RENDERED_ROWS"] = 5
            client = app.test_client()

            rows = b"Flow Duration,Tot Fwd Pkts\n" + b"9.0,9.0\n" * 40

            body = _post_csv(client, rows).get_data(as_text=True)

            self.assertIn("40", body)
            self.assertIn("Showing the", body)
            self.assertEqual(body.count("<tr>"), 5 + 1)

    def test_a_small_upload_is_not_marked_truncated(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()

            body = _post_csv(
                client, b"Flow Duration,Tot Fwd Pkts\n0.0,0.0\n2.0,2.0\n"
            ).get_data(as_text=True)

            self.assertNotIn("Showing the", body)

    def test_a_pending_flash_is_consumed_by_the_results_page(self):
        """Keeps the flash block live: a message set but never rendered here
        is what leaked the run summary onto whatever page came next."""

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()

            with client.session_transaction() as session:
                session["_flashes"] = [("warning", "a pending notice")]

            body = _post_csv(
                client, b"Flow Duration,Tot Fwd Pkts\n0.0,0.0\n"
            ).get_data(as_text=True)

            self.assertIn("a pending notice", body)

            later = client.get("/", follow_redirects=True).get_data(as_text=True)
            self.assertNotIn("a pending notice", later)

    def test_the_run_summary_does_not_leak_onto_the_next_page(self):
        """It was flashed but never rendered, so it surfaced on a later page."""

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            _write_bundle(bundle)
            client = create_app(bundle).test_client()

            _post_csv(client, b"Flow Duration,Tot Fwd Pkts\n9.0,9.0\n")
            later = client.get("/", follow_redirects=True).get_data(as_text=True)

            self.assertNotIn("Flows scored", later)
            self.assertNotIn("anomalies", later)


if __name__ == "__main__":
    unittest.main()
