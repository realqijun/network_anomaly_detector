import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from detector_runtime import Detector, HybridDetectorNetwork
from training_pipeline import canonicalize_labels, partition_csv_files


class LabelMappingTests(unittest.TestCase):
    def test_original_labels_map_to_stable_families(self):
        labels = pd.Series(
            [
                "BENIGN",
                "DDOS attack-HOIC",
                "DoS attacks-Hulk",
                "FTP-BruteForce",
                "Brute Force -XSS",
                "SQL Injection",
                "Bot",
                "Infilteration",
                "unrecognized",
            ]
        )

        mapped = canonicalize_labels(labels).tolist()

        self.assertEqual(
            mapped[:8],
            [
                "Benign",
                "DDoS",
                "DoS",
                "Brute Force",
                "XSS",
                "Injection",
                "Bot",
                "Infiltration",
            ],
        )
        self.assertTrue(pd.isna(mapped[8]))

    def test_files_are_partitioned_by_whole_capture_name(self):
        paths = [Path("day-20.csv"), Path("day-21.csv"), Path("day-22.csv")]

        training, validation = partition_csv_files(paths, ("21",))

        self.assertEqual(training, [Path("day-20.csv"), Path("day-22.csv")])
        self.assertEqual(validation, [Path("day-21.csv")])


class DetectorRuntimeTests(unittest.TestCase):
    def _write_bundle(self, directory: Path) -> None:
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

    def test_detector_returns_benign_and_unknown_anomaly(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            self._write_bundle(bundle)
            detector = Detector(bundle, device="cpu")
            flows = pd.DataFrame(
                {
                    "Flow Duration": [0.0, 2.0],
                    "Tot Fwd Pkts": [0.0, 2.0],
                }
            )

            result = detector.detect(flows)

            self.assertEqual(result["prediction"].tolist(), ["Benign", "Unknown anomaly"])
            self.assertEqual(result["is_anomaly"].tolist(), [False, True])

    def test_detector_rejects_missing_contract_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            self._write_bundle(bundle)
            detector = Detector(bundle, device="cpu")

            with self.assertRaisesRegex(ValueError, "Tot Fwd Pkts"):
                detector.detect(pd.DataFrame({"Flow Duration": [1.0]}))


class NotebookTests(unittest.TestCase):
    def test_colab_notebook_is_valid_and_requests_a_gpu(self):
        path = Path("notebooks/train_cse_cic_ids2018_colab.ipynb")
        notebook = json.loads(path.read_text())

        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        source = "\n".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("solarmainframe/ids-intrusion-csv", source)
        self.assertIn("kagglehub.dataset_download", source)
        self.assertIn("train_detector(config)", source)


if __name__ == "__main__":
    unittest.main()
