import json
import unittest
from pathlib import Path

import pandas as pd

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
