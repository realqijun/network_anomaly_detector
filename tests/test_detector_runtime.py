import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from detector_runtime import Detector
from feature_contract import ContractMismatchError
from tests.bundle_fixture import write_bundle


class BundleLoadingTests(unittest.TestCase):
    def test_loads_a_current_bundle(self):
        with TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_bundle(bundle)

            detector = Detector(bundle, device="cpu")

            self.assertEqual(detector.feature_names, ("Flow Duration", "Tot Fwd Pkts"))

    def test_refuses_a_legacy_bundle_format(self):
        with TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_bundle(bundle)
            manifest = json.loads((bundle / "manifest.json").read_text())
            manifest["bundle_format_version"] = 1
            (bundle / "manifest.json").write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ContractMismatchError, "bundle format"):
                Detector(bundle, device="cpu")

    def test_refuses_a_stale_feature_contract_under_a_current_format(self):
        with TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_bundle(bundle)
            manifest = json.loads((bundle / "manifest.json").read_text())
            manifest["feature_contract"]["version"] = 99
            (bundle / "manifest.json").write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ContractMismatchError, "99"):
                Detector(bundle, device="cpu")

    def test_reports_a_missing_bundle_clearly(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "manifest.json"):
                Detector(Path(temporary) / "absent", device="cpu")


class DetectTests(unittest.TestCase):
    def _detector(self, directory, **kwargs):
        write_bundle(directory, **kwargs)
        return Detector(directory, device="cpu")

    def test_separates_benign_rows_from_unknown_anomalies(self):
        with TemporaryDirectory() as temporary:
            detector = self._detector(Path(temporary))
            flows = pd.DataFrame(
                {"Flow Duration": [0.0, 2.0], "Tot Fwd Pkts": [0.0, 2.0]}
            )

            result = detector.detect(flows)

            self.assertEqual(result["prediction"].tolist(), ["Benign", "Unknown anomaly"])
            self.assertEqual(result["is_anomaly"].tolist(), [False, True])

    def test_rejects_missing_contract_features_by_name(self):
        with TemporaryDirectory() as temporary:
            detector = self._detector(Path(temporary))

            with self.assertRaisesRegex(ValueError, "Tot Fwd Pkts"):
                detector.detect(pd.DataFrame({"Flow Duration": [1.0]}))

    def test_rejects_non_finite_values_rather_than_zero_filling(self):
        with TemporaryDirectory() as temporary:
            detector = self._detector(Path(temporary))
            flows = pd.DataFrame(
                {"Flow Duration": [1.0], "Tot Fwd Pkts": [float("nan")]}
            )

            with self.assertRaisesRegex(ValueError, "Tot Fwd Pkts"):
                detector.detect(flows)

    def test_an_empty_frame_returns_the_result_columns(self):
        with TemporaryDirectory() as temporary:
            detector = self._detector(Path(temporary))

            result = detector.detect(pd.DataFrame(columns=["Flow Duration", "Tot Fwd Pkts"]))

            self.assertEqual(len(result), 0)
            self.assertIn("prediction", result.columns)

    def test_explanations_are_attached_only_to_flagged_rows(self):
        with TemporaryDirectory() as temporary:
            detector = self._detector(Path(temporary))
            flows = pd.DataFrame(
                {"Flow Duration": [0.0, 9.0], "Tot Fwd Pkts": [0.0, 9.0]}
            )

            result = detector.detect(flows, explain=True)

            self.assertEqual(result["explanation"].iloc[0], [])
            self.assertTrue(result["explanation"].iloc[1])
            self.assertIn("feature", result["explanation"].iloc[1][0])

    def test_explanations_are_absent_unless_requested(self):
        with TemporaryDirectory() as temporary:
            detector = self._detector(Path(temporary))
            flows = pd.DataFrame({"Flow Duration": [9.0], "Tot Fwd Pkts": [9.0]})

            result = detector.detect(flows)

            self.assertNotIn("explanation", result.columns)


class WindowBlockBundleTests(unittest.TestCase):
    """A bundle declaring a window block must get one computed for it."""

    def test_computes_window_features_from_identity_columns(self):
        with TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_bundle(bundle, with_window_block=True)
            detector = Detector(bundle, device="cpu")
            flows = pd.DataFrame(
                {
                    "Flow Duration": [1.0, 1.0],
                    "Tot Fwd Pkts": [1.0, 1.0],
                    "Timestamp": ["2018-02-14 08:00:00", "2018-02-14 08:00:00"],
                    "Src IP": ["10.0.0.1", "10.0.0.2"],
                    "Dst IP": ["10.0.0.9", "10.0.0.9"],
                    "Src Port": [1111, 2222],
                    "Dst Port": [80, 80],
                    "Protocol": [6, 6],
                    "SYN Flag Cnt": [1, 1],
                    "ACK Flag Cnt": [0, 0],
                    "TotLen Fwd Pkts": [100.0, 100.0],
                    "TotLen Bwd Pkts": [0.0, 0.0],
                    "Tot Bwd Pkts": [0.0, 0.0],
                }
            )

            result = detector.detect(flows)

            self.assertEqual(len(result), 2)
            self.assertIn("prediction", result.columns)

    def test_refuses_to_score_without_the_identity_columns(self):
        with TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_bundle(bundle, with_window_block=True)
            detector = Detector(bundle, device="cpu")

            with self.assertRaisesRegex(ValueError, "Dst IP"):
                detector.detect(
                    pd.DataFrame({"Flow Duration": [1.0], "Tot Fwd Pkts": [1.0]})
                )

    def test_exposes_whether_the_bundle_covers_window_behaviour(self):
        with TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "a", Path(temporary) / "b"
            first.mkdir()
            second.mkdir()
            write_bundle(first, with_window_block=True)
            write_bundle(second, disabled_reason="capture has no addresses")

            self.assertTrue(Detector(first, device="cpu").has_window_coverage)
            covered = Detector(second, device="cpu")
            self.assertFalse(covered.has_window_coverage)
            self.assertEqual(
                covered.window_coverage_reason, "capture has no addresses"
            )


if __name__ == "__main__":
    unittest.main()
