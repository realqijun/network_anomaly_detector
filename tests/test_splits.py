import unittest
from pathlib import Path

from splits import (
    SplitLeakageError,
    build_grouped_split,
    capture_group,
    resolve_capture_convention,
    split_manifest,
    validate_split_manifest,
)

MIRROR = [Path(f"data/02-{day}-2018.csv") for day in (14, 15, 16, 20, 21, 22, 23, 28)]
MIRROR += [Path("data/03-01-2018.csv"), Path("data/03-02-2018.csv")]

AWS = [Path(f"data/{day}-02-2018.csv") for day in (14, 15, 16, 20, 21, 22, 23, 28)]
AWS += [Path("data/01-03-2018.csv"), Path("data/02-03-2018.csv")]


class CaptureConventionTests(unittest.TestCase):
    """Resolve the day/month order once per corpus, never per file."""

    def test_resolves_a_mirror_corpus_as_month_first(self):
        self.assertEqual(resolve_capture_convention(MIRROR), "month_first")

    def test_resolves_an_aws_style_corpus_as_day_first(self):
        self.assertEqual(resolve_capture_convention(AWS), "day_first")

    def test_an_ambiguous_corpus_falls_back_to_stem_grouping(self):
        paths = [Path("data/03-01-2018.csv"), Path("data/04-02-2018.csv")]

        self.assertEqual(resolve_capture_convention(paths), "stem")

    def test_a_corpus_exhibiting_both_conventions_raises_and_names_the_files(self):
        paths = [Path("data/02-14-2018.csv"), Path("data/14-02-2018.csv")]

        with self.assertRaises(ValueError) as caught:
            resolve_capture_convention(paths)

        message = str(caught.exception)
        self.assertIn("02-14-2018.csv", message)
        self.assertIn("14-02-2018.csv", message)

    def test_both_conventions_normalize_a_day_to_the_same_key(self):
        """This is what lets the split defaults be correct either way."""

        mirror = capture_group(Path("02-16-2018.csv"), "month_first")
        aws = capture_group(Path("16-02-2018.csv"), "day_first")

        self.assertEqual(mirror, "2018-02-16")
        self.assertEqual(aws, "2018-02-16")

    def test_the_whole_mirror_corpus_lands_in_one_namespace(self):
        """Per-file disambiguation split 03-01-2018 away from its neighbours."""

        convention = resolve_capture_convention(MIRROR)
        groups = [capture_group(path, convention) for path in MIRROR]

        self.assertTrue(all(group.startswith("2018-") for group in groups))
        self.assertIn("2018-03-01", groups)
        self.assertIn("2018-02-14", groups)

    def test_shards_of_one_capture_day_share_a_group(self):
        convention = "month_first"

        first = capture_group(Path("02-14-2018_part1.csv"), convention)
        second = capture_group(Path("02-14-2018_TrafficForML_CICFlowMeter.csv"), convention)

        self.assertEqual(first, second)

    def test_a_corpus_without_dates_groups_by_stem_consistently(self):
        paths = [Path("day-20.csv"), Path("day-21.csv")]
        convention = resolve_capture_convention(paths)

        self.assertEqual(
            [capture_group(path, convention) for path in paths], ["day-20", "day-21"]
        )


class GroupedSplitTests(unittest.TestCase):
    def test_assigns_whole_groups_to_validation(self):
        split = build_grouped_split(MIRROR, validation_patterns=("2018-02-16",))

        self.assertEqual([path.name for path in split.validation], ["02-16-2018.csv"])
        self.assertNotIn(Path("data/02-16-2018.csv"), split.train)

    def test_the_normalized_defaults_select_the_same_days_in_either_corpus(self):
        patterns = ("2018-02-16", "2018-02-21", "2018-02-23", "2018-03-01")

        mirror = build_grouped_split(MIRROR, validation_patterns=patterns)
        aws = build_grouped_split(AWS, validation_patterns=patterns)

        self.assertEqual(sorted(mirror.validation_groups), sorted(aws.validation_groups))
        self.assertEqual(len(mirror.validation), 4)

    def test_every_shard_of_a_validation_day_goes_to_validation(self):
        paths = [
            Path("data/02-14-2018_part1.csv"),
            Path("data/02-14-2018_part2.csv"),
            Path("data/02-16-2018.csv"),
        ]

        split = build_grouped_split(paths, validation_patterns=("2018-02-14",))

        self.assertEqual(len(split.validation), 2)
        self.assertEqual(len(split.train), 1)

    def test_still_matches_a_bare_filename_pattern(self):
        """Keeps tests/test_training_pipeline.py's day-20/day-21 case working."""

        paths = [Path("day-20.csv"), Path("day-21.csv"), Path("day-22.csv")]

        split = build_grouped_split(paths, validation_patterns=("21",))

        self.assertEqual([p.name for p in split.train], ["day-20.csv", "day-22.csv"])
        self.assertEqual([p.name for p in split.validation], ["day-21.csv"])

    def test_matches_a_filename_even_when_the_group_key_does_not(self):
        """The filename arm: the pattern appears only in the shard suffix."""

        paths = [
            Path("data/02-14-2018_TrafficForML.csv"),
            Path("data/02-15-2018_WebAttacks.csv"),
        ]

        split = build_grouped_split(paths, validation_patterns=("WebAttacks",))

        self.assertEqual([p.name for p in split.validation], ["02-15-2018_WebAttacks.csv"])
        self.assertEqual([p.name for p in split.train], ["02-14-2018_TrafficForML.csv"])
        # And the group key itself carries no trace of the pattern.
        self.assertEqual(split.validation_groups, ["2018-02-15"])

    def test_reports_the_available_group_keys_when_nothing_matches(self):
        with self.assertRaises(ValueError) as caught:
            build_grouped_split(MIRROR, validation_patterns=("16-02-2018",))

        self.assertIn("2018-02-16", str(caught.exception))

    def test_refuses_to_leave_no_training_files(self):
        with self.assertRaisesRegex(ValueError, "training"):
            build_grouped_split(MIRROR, validation_patterns=("2018",))

    def test_keeps_an_external_corpus_in_its_own_partition(self):
        external = [Path("other/cicddos2019.csv")]

        split = build_grouped_split(
            MIRROR, validation_patterns=("2018-02-16",), external_test_files=external
        )

        self.assertEqual(split.external_test, external)
        self.assertNotIn(external[0], split.train)


class SplitManifestTests(unittest.TestCase):
    """The validator is the regression guard, so the trainer runs it too."""

    def _manifest(self):
        return split_manifest(
            build_grouped_split(MIRROR, validation_patterns=("2018-02-16",))
        )

    def test_records_the_group_key_and_the_groups_per_partition(self):
        manifest = self._manifest()

        self.assertEqual(manifest["group_key"], "capture_day")
        self.assertIn("2018-02-16", manifest["validation"])
        self.assertIn("2018-02-14", manifest["train"])

    def test_records_the_files_under_each_group(self):
        manifest = self._manifest()

        self.assertEqual(manifest["validation"]["2018-02-16"], ["02-16-2018.csv"])

    def test_a_valid_manifest_passes(self):
        validate_split_manifest(self._manifest())

    def test_a_group_in_two_partitions_is_leakage(self):
        manifest = self._manifest()
        manifest["train"]["2018-02-16"] = ["02-16-2018.csv"]

        with self.assertRaises(SplitLeakageError) as caught:
            validate_split_manifest(manifest)

        self.assertIn("2018-02-16", str(caught.exception))

    def test_an_empty_validation_partition_is_rejected(self):
        manifest = self._manifest()
        manifest["validation"] = {}

        with self.assertRaisesRegex(SplitLeakageError, "validation"):
            validate_split_manifest(manifest)

    def test_an_empty_training_partition_is_rejected(self):
        manifest = self._manifest()
        manifest["train"] = {}

        with self.assertRaisesRegex(SplitLeakageError, "train"):
            validate_split_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
