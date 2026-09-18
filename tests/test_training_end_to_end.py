"""Train a real bundle on a tiny corpus and check what it promises.

Everything below runs the public entry point, so the manifest asserted here is
the manifest a deployment would load.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from detector_runtime import Detector
from feature_contract import BUNDLE_FORMAT_VERSION, FEATURE_CONTRACT_VERSION
from splits import SplitLeakageError
from training_pipeline import TrainingConfig, train_detector

BASE = pd.Timestamp("2018-02-14 08:00:00")


def _day(path: Path, rows: int, seed: int, attack_share: float = 0.3) -> None:
    rng = np.random.default_rng(seed)
    is_attack = rng.random(rows) < attack_share
    frame = pd.DataFrame(
        {
            "Flow Duration": np.where(is_attack, rng.normal(900, 40, rows), rng.normal(70, 8, rows)),
            "Tot Fwd Pkts": np.where(is_attack, rng.normal(220, 12, rows), rng.normal(14, 3, rows)),
            "Flow Byts/s": np.where(is_attack, rng.normal(6000, 200, rows), rng.normal(320, 40, rows)),
            "Timestamp": [
                (BASE + pd.Timedelta(seconds=float(index) * 1.7)).strftime("%Y-%m-%d %H:%M:%S")
                for index in range(rows)
            ],
            "Src IP": [f"10.0.0.{index % 23}" for index in range(rows)],
            "Dst IP": ["10.0.1.9" if attack else f"10.0.2.{index % 6}"
                       for index, attack in enumerate(is_attack)],
            "Src Port": [40000 + index % 9000 for index in range(rows)],
            "Dst Port": [80 if attack else 443 for attack in is_attack],
            "Protocol": [6] * rows,
            "SYN Flag Cnt": np.where(is_attack, 1, 1),
            "ACK Flag Cnt": np.where(is_attack, 0, 1),
            "TotLen Fwd Pkts": np.where(is_attack, 9000.0, 800.0),
            "TotLen Bwd Pkts": np.where(is_attack, 60.0, 900.0),
            "Tot Bwd Pkts": np.where(is_attack, 2.0, 11.0),
            "Label": ["DDOS attack-HOIC" if attack else "Benign" for attack in is_attack],
        }
    )
    frame.to_csv(path, index=False)


def _corpus(root: Path) -> None:
    _day(root / "02-14-2018.csv", 900, seed=1)
    _day(root / "02-15-2018.csv", 900, seed=2)
    _day(root / "02-16-2018.csv", 900, seed=3)


def _config(root: Path, output: Path, **overrides) -> TrainingConfig:
    settings = dict(
        data_dir=str(root),
        output_dir=str(output),
        validation_patterns=("2018-02-16",),
        chunk_size=200,
        batch_size=64,
        epochs=1,
        max_evaluation_rows=10_000,
        max_false_positives_per_hour=10_000.0,
        min_family_support=1,
        window_cache_dir=str(root / ".cache"),
    )
    settings.update(overrides)
    return TrainingConfig(**settings)


class TrainDetectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "bundle"
        _corpus(self.root)

    def _train(self, **overrides):
        return train_detector(_config(self.root, self.output, **overrides))

    def _manifest(self):
        return json.loads((self.output / "manifest.json").read_text())

    def test_writes_a_loadable_bundle(self):
        self._train()

        detector = Detector(self.output, device="cpu")

        self.assertIn("Benign", detector.classes)
        self.assertTrue(detector.has_window_coverage)

    def test_the_bundle_declares_both_versions(self):
        self._train()
        manifest = self._manifest()

        self.assertEqual(manifest["bundle_format_version"], BUNDLE_FORMAT_VERSION)
        self.assertEqual(
            manifest["feature_contract"]["version"], FEATURE_CONTRACT_VERSION
        )

    def test_the_contract_carries_the_window_block(self):
        self._train()
        contract = self._manifest()["feature_contract"]

        self.assertTrue(contract["window_features"])
        self.assertIn("win1s_dst_flow_count", contract["window_features"])
        self.assertEqual(
            contract["features"],
            contract["flow_features"] + contract["window_features"],
        )

    def test_the_bundle_records_the_split_it_was_trained_on(self):
        self._train()
        split = self._manifest()["training"]["split_manifest"]

        self.assertEqual(list(split["validation"]), ["2018-02-16"])
        self.assertEqual(sorted(split["train"]), ["2018-02-14", "2018-02-15"])
        self.assertEqual(split["group_key"], "capture_day")

    def test_the_bundle_records_operational_metrics_not_only_an_auc(self):
        self._train()
        operational = self._manifest()["metrics"]["operational"]

        self.assertIn("false_positives_per_hour", operational)
        self.assertIn("anomaly_pr_auc", operational)
        self.assertIn("top_k_precision", operational)
        self.assertIn("per_family", operational)
        self.assertEqual(operational["prevalence_basis"]["sampling"], "uniform")
        self.assertGreater(operational["benign_hours"], 0)

    def test_the_bundle_records_why_its_threshold_is_what_it_is(self):
        self._train()
        policy = self._manifest()["thresholds"]["policy"]

        self.assertEqual(policy["policy"], "false_positive_budget")
        self.assertIn("realized_false_positives_per_hour", policy)

    def test_the_bundle_records_how_explanations_were_produced(self):
        self._train()
        explanation = self._manifest()["explanation"]

        self.assertEqual(explanation["version"], 1)
        self.assertIn("gradient_x_input", explanation["methods"])

    def test_the_archive_contains_only_the_bundle_files(self):
        """A window cache beside the bundle must not end up in the download."""

        import zipfile

        result = self._train()

        with zipfile.ZipFile(result["archive_path"]) as archive:
            self.assertEqual(sorted(archive.namelist()), ["manifest.json", "model.pt"])

    def test_the_trained_bundle_scores_a_held_out_capture(self):
        self._train()
        detector = Detector(self.output, device="cpu")
        flows = pd.read_csv(self.root / "02-16-2018.csv")

        detections = detector.detect(flows, explain=True)

        self.assertEqual(len(detections), len(flows))
        self.assertTrue(set(detections["prediction"]) <= set(detector.classes) | {"Unknown anomaly"})

    def test_a_strict_alert_budget_is_honoured_by_the_exported_threshold(self):
        """The budget is a constraint on the bundle, not a suggestion.

        Refusal when a budget can only be met by flagging nothing is covered
        deterministically in tests.test_evaluation; here the corpus separates
        well, so a strict budget should be met rather than refused.
        """

        self._train(max_false_positives_per_hour=1e-6)
        policy = self._manifest()["thresholds"]["policy"]

        self.assertLessEqual(policy["realized_false_positives_per_hour"], 1e-6)


class WindowSupportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "bundle"
        _corpus(self.root)
        for path in sorted(self.root.glob("*.csv")):
            frame = pd.read_csv(path)
            frame.drop(columns=["Src IP", "Dst IP"]).to_csv(path, index=False)

    def test_require_refuses_a_corpus_without_addresses(self):
        with self.assertRaisesRegex(ValueError, "Src IP|Dst IP"):
            train_detector(_config(self.root, self.output))

    def test_auto_degrades_and_records_why(self):
        train_detector(
            _config(self.root, self.output, enable_window_features="auto")
        )
        contract = json.loads((self.output / "manifest.json").read_text())["feature_contract"]

        self.assertEqual(contract["window_features"], [])
        self.assertIn("identity", contract["window_features_disabled_reason"])


class SplitGuardTests(unittest.TestCase):
    def test_a_pattern_matching_nothing_lists_the_real_group_keys(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _corpus(root)

            with self.assertRaises(ValueError) as caught:
                train_detector(
                    _config(root, root / "bundle", validation_patterns=("16-02-2018",))
                )

            self.assertIn("2018-02-16", str(caught.exception))


class AliasedCorpusTests(unittest.TestCase):
    """A corpus using CIC-IDS2017 long headers must train, not abort.

    The contract advertises those aliases, so a corpus spelling its columns
    that way has to reach the trainer. Regression guard: the feature contract
    is built from canonical names, and _read_chunks selects on canonical
    names, so raw spellings anywhere between them mean no column is ever read.
    """

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "bundle"
        _corpus(self.root)
        renames = {
            "Tot Fwd Pkts": "Total Fwd Packets",
            "Tot Bwd Pkts": "Total Backward Packets",
            "TotLen Fwd Pkts": "Total Length of Fwd Packets",
            "TotLen Bwd Pkts": "Total Length of Bwd Packets",
            "SYN Flag Cnt": "SYN Flag Count",
            "ACK Flag Cnt": "ACK Flag Count",
            "Flow Byts/s": "Flow Bytes/s",
            "Src IP": "Source IP",
            "Dst IP": "Destination IP",
            "Src Port": "Source Port",
            "Dst Port": "Destination Port",
        }
        for path in sorted(self.root.glob("*.csv")):
            pd.read_csv(path).rename(columns=renames).to_csv(path, index=False)

    def test_trains_and_records_canonical_feature_names(self):
        train_detector(_config(self.root, self.output))
        contract = json.loads((self.output / "manifest.json").read_text())["feature_contract"]

        self.assertIn("Tot Fwd Pkts", contract["flow_features"])
        self.assertNotIn("Total Fwd Packets", contract["flow_features"])
        self.assertTrue(contract["window_features"])

    def test_the_bundle_scores_a_capture_in_either_spelling(self):
        train_detector(_config(self.root, self.output))
        detector = Detector(self.output, device="cpu")
        long_form = pd.read_csv(self.root / "02-16-2018.csv")

        detections = detector.detect(long_form)

        self.assertEqual(len(detections), len(long_form))


if __name__ == "__main__":
    unittest.main()
