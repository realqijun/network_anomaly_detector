import unittest

import pandas as pd

from feature_contract import (
    BUNDLE_FORMAT_VERSION,
    COLUMN_ALIASES,
    COMPATIBLE_CONTRACT_VERSIONS,
    FEATURE_CONTRACT_VERSION,
    TRANSFORM,
    ContractMismatchError,
    FeatureContract,
    check_bundle_version,
    IDENTITY_COLUMNS,
    WINDOW_SOURCE_COLUMNS,
    epoch_seconds,
    normalize_columns,
    parse_timestamps,
)


class ColumnNormalizationTests(unittest.TestCase):
    """One spelling reaches the contract, whichever convention the source uses."""

    def test_strips_padding_the_datasets_ship_with(self):
        frame = pd.DataFrame({" Flow Duration ": [1.0], "Tot Fwd Pkts\t": [2.0]})

        normalized = normalize_columns(frame)

        self.assertEqual(
            sorted(normalized.columns), ["Flow Duration", "Tot Fwd Pkts"]
        )

    def test_folds_cic_ids2017_long_names_to_canonical_short_names(self):
        frame = pd.DataFrame(
            {
                "Total Fwd Packets": [1],
                "Total Length of Bwd Packets": [2],
                "Init_Win_bytes_forward": [3],
                "Source IP": ["10.0.0.1"],
                "Destination Port": [80],
            }
        )

        normalized = normalize_columns(frame)

        self.assertEqual(
            normalized.columns.tolist(),
            ["Tot Fwd Pkts", "TotLen Bwd Pkts", "Init Fwd Win Byts", "Src IP", "Dst Port"],
        )

    def test_folds_cicflowmeter_cli_names_including_identity_fields(self):
        frame = pd.DataFrame(
            {
                "src_ip": ["10.0.0.1"],
                "dst_ip": ["10.0.0.2"],
                "src_port": [51000],
                "dst_port": [80],
                "protocol": [6],
                "timestamp": ["2018-02-14 08:00:00"],
                "tot_fwd_pkts": [4],
            }
        )

        normalized = normalize_columns(frame)

        self.assertEqual(
            normalized.columns.tolist(),
            ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol", "Timestamp", "Tot Fwd Pkts"],
        )

    def test_leaves_unrecognized_columns_untouched(self):
        frame = pd.DataFrame({"some_unmapped_column": [1]})

        self.assertEqual(
            normalize_columns(frame).columns.tolist(), ["some_unmapped_column"]
        )

    def test_every_identity_and_window_source_column_is_reachable_by_alias(self):
        """A canonical name the aliases cannot produce is a hole in the contract."""

        produced = set(COLUMN_ALIASES.values())

        for column in tuple(IDENTITY_COLUMNS) + tuple(WINDOW_SOURCE_COLUMNS):
            with self.subTest(column=column):
                self.assertIn(column, produced)


class TimestampParsingTests(unittest.TestCase):
    """Guessing the day/month order silently reorders every window bucket."""

    def test_parses_an_unambiguous_day_first_capture(self):
        parsed = parse_timestamps(pd.Series(["14/02/2018 08:31:02", "28/02/2018 09:00:00"]))

        self.assertEqual(parsed.iloc[0].day, 14)
        self.assertEqual(parsed.iloc[0].month, 2)

    def test_parses_iso_timestamps(self):
        parsed = parse_timestamps(pd.Series(["2018-02-14 08:31:02"]))

        self.assertEqual(parsed.iloc[0].month, 2)
        self.assertEqual(parsed.iloc[0].day, 14)

    def test_refuses_to_guess_when_the_order_is_ambiguous(self):
        """03/01/2018 is a real date under both conventions, so it must not parse."""

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            parse_timestamps(pd.Series(["03/01/2018 08:31:02"]))

    def test_reports_the_offending_value_when_nothing_parses(self):
        with self.assertRaisesRegex(ValueError, "not-a-timestamp"):
            parse_timestamps(pd.Series(["not-a-timestamp"]))

    def test_epoch_seconds_is_independent_of_the_pandas_datetime_unit(self):
        """Window bucketing divides by this, so a resolution assumption misbuckets."""

        parsed = parse_timestamps(pd.Series(["2018-02-14 00:00:00", "2018-02-14 00:01:00"]))

        seconds = epoch_seconds(parsed)

        self.assertEqual(seconds.iloc[1] - seconds.iloc[0], 60.0)
        self.assertEqual(seconds.iloc[0], 1518566400.0)



class ContractManifestTests(unittest.TestCase):
    """A bundle must fail closed on any contract it was not built for."""

    def _contract(self, **overrides):
        defaults = dict(
            flow_features=("Flow Duration", "Tot Fwd Pkts"),
            mean=(0.0, 0.0),
            scale=(1.0, 1.0),
        )
        defaults.update(overrides)
        return FeatureContract(**defaults)

    def test_round_trips_through_a_manifest_block(self):
        contract = self._contract()

        restored = FeatureContract.from_manifest(contract.to_manifest())

        self.assertEqual(restored, contract)

    def test_records_the_current_versions(self):
        manifest = self._contract().to_manifest()

        self.assertEqual(manifest["version"], FEATURE_CONTRACT_VERSION)
        self.assertEqual(manifest["transform"], TRANSFORM)
        self.assertEqual(manifest["missing_value_policy"], "reject")

    def test_rejects_a_contract_version_outside_the_compatible_set(self):
        manifest = self._contract().to_manifest()
        manifest["version"] = 99

        with self.assertRaises(ContractMismatchError) as caught:
            FeatureContract.from_manifest(manifest)

        message = str(caught.exception)
        self.assertIn("99", message)
        self.assertIn(str(sorted(COMPATIBLE_CONTRACT_VERSIONS)), message)

    def test_rejects_an_unknown_transform(self):
        manifest = self._contract().to_manifest()
        manifest["transform"] = "minmax"

        with self.assertRaisesRegex(ContractMismatchError, "minmax"):
            FeatureContract.from_manifest(manifest)

    def test_rejects_scaling_vectors_that_do_not_match_the_feature_count(self):
        with self.assertRaisesRegex(ValueError, "length"):
            self._contract(scale=(1.0,))

    def test_rejects_a_non_positive_scale(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self._contract(scale=(1.0, 0.0))


class BundleVersionPrecedenceTests(unittest.TestCase):
    """Bundle layout is checked first, contract second, and both always."""

    def test_rejects_a_stale_bundle_format(self):
        with self.assertRaises(ContractMismatchError) as caught:
            check_bundle_version(1)

        self.assertIn(str(BUNDLE_FORMAT_VERSION), str(caught.exception))

    def test_a_current_bundle_format_with_a_stale_contract_still_fails(self):
        """The divergent case: bumping both at once would leave this untested."""

        manifest = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "feature_contract": {
                "version": 99,
                "features": ["Flow Duration"],
                "transform": TRANSFORM,
                "mean": [0.0],
                "scale": [1.0],
                "missing_value_policy": "reject",
                "window_seconds": [],
            },
        }

        check_bundle_version(manifest["bundle_format_version"])
        with self.assertRaises(ContractMismatchError) as caught:
            FeatureContract.from_manifest(manifest["feature_contract"])

        message = str(caught.exception)
        self.assertIn("99", message)
        self.assertIn(str(FEATURE_CONTRACT_VERSION), message)


class PrepareTests(unittest.TestCase):
    """One transform, shared by training and inference."""

    def _contract(self):
        return FeatureContract(
            flow_features=("Flow Duration", "Tot Fwd Pkts"),
            mean=(0.0, 0.0),
            scale=(1.0, 1.0),
        )

    def test_applies_signed_log1p_then_standardizes(self):
        prepared = self._contract().prepare(
            pd.DataFrame({"Flow Duration": [9.0], "Tot Fwd Pkts": [-9.0]})
        )

        # log(10) from an independent source, and the sign is preserved.
        self.assertAlmostEqual(float(prepared[0, 0]), 2.302585, places=5)
        self.assertAlmostEqual(float(prepared[0, 1]), -2.302585, places=5)

    def test_divides_by_the_fitted_scale(self):
        contract = FeatureContract(
            flow_features=("Flow Duration",), mean=(2.302585,), scale=(2.0,)
        )

        prepared = contract.prepare(pd.DataFrame({"Flow Duration": [9.0]}))

        self.assertAlmostEqual(float(prepared[0, 0]), 0.0, places=5)

    def test_accepts_the_long_dataset_spelling_of_a_contract_feature(self):
        prepared = self._contract().prepare(
            pd.DataFrame({"Flow Duration": [9.0], "Total Fwd Packets": [-9.0]})
        )

        self.assertAlmostEqual(float(prepared[0, 1]), -2.302585, places=5)

    def test_orders_columns_by_the_contract_not_by_the_input(self):
        prepared = self._contract().prepare(
            pd.DataFrame({"Tot Fwd Pkts": [-9.0], "Flow Duration": [9.0]})
        )

        self.assertAlmostEqual(float(prepared[0, 0]), 2.302585, places=5)

    def test_rejects_a_missing_contract_feature_by_name(self):
        with self.assertRaisesRegex(ValueError, "Tot Fwd Pkts"):
            self._contract().prepare(pd.DataFrame({"Flow Duration": [1.0]}))

    def test_rejects_non_finite_values_by_column_name(self):
        frame = pd.DataFrame({"Flow Duration": [1.0], "Tot Fwd Pkts": [float("inf")]})

        with self.assertRaisesRegex(ValueError, "Tot Fwd Pkts"):
            self._contract().prepare(frame)

    def test_rejects_unparseable_values_rather_than_zero_filling_them(self):
        frame = pd.DataFrame({"Flow Duration": [1.0], "Tot Fwd Pkts": ["n/a"]})

        with self.assertRaisesRegex(ValueError, "Tot Fwd Pkts"):
            self._contract().prepare(frame)



class EpochSecondsRangeTests(unittest.TestCase):
    """A date outside the nanosecond range must not abort a whole capture."""

    def test_converts_dates_beyond_the_nanosecond_range(self):
        parsed = parse_timestamps(
            pd.Series(["2018-02-14 08:00:00", "9999-01-01 00:00:00"])
        )

        seconds = epoch_seconds(parsed)

        self.assertAlmostEqual(seconds.iloc[0], 1518595200.0, places=1)
        self.assertGreater(seconds.iloc[1], 2.5e11)

    def test_converts_dates_before_the_nanosecond_range(self):
        parsed = parse_timestamps(pd.Series(["1500-01-01 00:00:00"]))

        self.assertLess(epoch_seconds(parsed).iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
