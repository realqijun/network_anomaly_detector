import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from feature_contract import (
    WINDOW_FEATURE_NAMES,
    FeatureContract,
    WINDOW_SECONDS,
    build_window_sidecar,
    compute_window_features,
    window_coverage_warning,
    window_inputs_available,
)

BASE = 1_518_566_400.0  # 2018-02-14 00:00:00Z


def _flows(rows):
    """Build a normalized flow frame from (offset, src, dst, port, syn, ack) tuples."""

    return pd.DataFrame(
        {
            "Timestamp": [
                pd.Timestamp(BASE + offset, unit="s").strftime("%Y-%m-%d %H:%M:%S")
                for offset, *_ in rows
            ],
            "Src IP": [row[1] for row in rows],
            "Dst IP": [row[2] for row in rows],
            "Src Port": [40000 + index for index, _ in enumerate(rows)],
            "Dst Port": [row[3] for row in rows],
            "Protocol": [6] * len(rows),
            "SYN Flag Cnt": [row[4] for row in rows],
            "ACK Flag Cnt": [row[5] for row in rows],
            "TotLen Fwd Pkts": [100.0] * len(rows),
            "TotLen Bwd Pkts": [50.0] * len(rows),
            "Tot Fwd Pkts": [2.0] * len(rows),
            "Tot Bwd Pkts": [1.0] * len(rows),
        }
    )


class WindowFeatureShapeTests(unittest.TestCase):
    def test_returns_exactly_the_contract_window_columns_in_order(self):
        computed = compute_window_features(_flows([(0, "a", "z", 80, 1, 1)]))

        self.assertEqual(computed.columns.tolist(), list(WINDOW_FEATURE_NAMES))

    def test_defines_eleven_fields_for_each_window(self):
        self.assertEqual(len(WINDOW_FEATURE_NAMES), 11 * len(WINDOW_SECONDS))

    def test_preserves_the_input_index_one_to_one(self):
        flows = _flows([(0, "a", "z", 80, 1, 1), (5, "b", "z", 80, 1, 1)])

        computed = compute_window_features(flows)

        self.assertEqual(len(computed), len(flows))
        self.assertTrue(computed.index.equals(flows.index))


class WindowBehaviourTests(unittest.TestCase):
    """The attack behaviours issue #6 names, made measurable."""

    def test_counts_fan_in_from_many_sources_onto_one_destination(self):
        rows = [(0.0, f"src{index}", "victim", 80, 1, 0) for index in range(30)]
        rows += [(0.5, f"src{index}", "victim", 80, 1, 0) for index in range(30)]

        computed = compute_window_features(_flows(rows))

        self.assertEqual(computed["win1s_dst_unique_src"].iloc[0], 30)
        self.assertEqual(computed["win1s_dst_flow_count"].iloc[0], 60)
        self.assertEqual(computed["win60s_dst_flow_count"].iloc[0], 60)

    def test_counts_repetition_between_one_source_and_one_destination(self):
        rows = [(index * 0.1, "attacker", "victim", 22, 1, 1) for index in range(9)]
        rows.append((0.05, "other", "victim", 22, 1, 1))

        computed = compute_window_features(_flows(rows))

        self.assertEqual(computed["win1s_pair_flow_count"].iloc[0], 9)
        self.assertEqual(computed["win1s_dst_flow_count"].iloc[0], 10)

    def test_counts_fan_out_from_one_source_across_destinations(self):
        rows = [(index * 0.1, "scanner", f"host{index}", 80, 1, 0) for index in range(7)]

        computed = compute_window_features(_flows(rows))

        self.assertEqual(computed["win1s_src_unique_dst"].iloc[0], 7)

    def test_counts_port_spread_against_one_destination(self):
        rows = [(index * 0.01, "scanner", "victim", 20 + index, 1, 0) for index in range(12)]

        computed = compute_window_features(_flows(rows))

        self.assertEqual(computed["win1s_dst_unique_port"].iloc[0], 12)

    def test_syn_ack_imbalance_is_positive_for_a_half_open_flood(self):
        flood = compute_window_features(
            _flows([(index * 0.01, f"s{index}", "victim", 80, 1, 0) for index in range(20)])
        )
        handshaken = compute_window_features(
            _flows([(index * 0.01, f"s{index}", "victim", 80, 1, 1) for index in range(20)])
        )

        self.assertGreater(flood["win1s_syn_ack_imbalance"].iloc[0], 0.9)
        self.assertAlmostEqual(handshaken["win1s_syn_ack_imbalance"].iloc[0], 0.0, places=2)

    def test_rates_are_per_second_not_per_window(self):
        rows = [(index * 0.1, f"s{index}", "victim", 80, 1, 1) for index in range(10)]

        computed = compute_window_features(_flows(rows))

        # 10 flows, 150 bytes each, all inside one 1s bucket and one 10s bucket.
        self.assertAlmostEqual(computed["win1s_dst_bytes_rate"].iloc[0], 1500.0)
        self.assertAlmostEqual(computed["win10s_dst_bytes_rate"].iloc[0], 150.0)
        self.assertAlmostEqual(computed["win1s_dst_pkt_rate"].iloc[0], 30.0)

    def test_baseline_delta_is_the_change_against_the_preceding_window(self):
        rows = [(0.1, "a", "victim", 80, 1, 1)]
        rows += [(1.1 + index * 0.01, f"s{index}", "victim", 80, 1, 1) for index in range(5)]

        computed = compute_window_features(_flows(rows))

        # First bucket has no predecessor, so its delta is its own count.
        self.assertEqual(computed["win1s_dst_flow_count_delta"].iloc[0], 1)
        # Second bucket has 5 flows against the previous bucket's 1.
        self.assertEqual(computed["win1s_dst_flow_count_delta"].iloc[1], 4)

    def test_delta_is_negative_when_pressure_drops(self):
        rows = [(0.0 + index * 0.01, f"s{index}", "victim", 80, 1, 1) for index in range(6)]
        rows += [(1.5, "a", "victim", 80, 1, 1)]

        computed = compute_window_features(_flows(rows))

        self.assertEqual(computed["win1s_dst_flow_count_delta"].iloc[-1], -5)


class WindowInputTests(unittest.TestCase):
    def test_reports_the_columns_a_frame_is_missing(self):
        flows = _flows([(0, "a", "z", 80, 1, 1)]).drop(columns=["Timestamp", "Src IP"])

        available, missing = window_inputs_available(flows)

        self.assertFalse(available)
        self.assertIn("Timestamp", missing)
        self.assertIn("Src IP", missing)

    def test_reports_available_for_a_complete_frame(self):
        available, missing = window_inputs_available(_flows([(0, "a", "z", 80, 1, 1)]))

        self.assertTrue(available)
        self.assertEqual(missing, [])

    def test_accepts_the_long_dataset_spelling(self):
        flows = _flows([(0, "a", "z", 80, 1, 1)]).rename(
            columns={"Src IP": "Source IP", "Dst IP": "Destination IP"}
        )

        available, _ = window_inputs_available(flows)

        self.assertTrue(available)

    def test_refuses_to_compute_windows_without_identity_columns(self):
        flows = _flows([(0, "a", "z", 80, 1, 1)]).drop(columns=["Dst IP"])

        with self.assertRaisesRegex(ValueError, "Dst IP"):
            compute_window_features(flows)


class WindowCoverageTests(unittest.TestCase):
    """A frame too short to support the window block must say so."""

    def test_warns_when_the_capture_is_shorter_than_the_widest_window(self):
        rows = [(index * 0.01, f"s{index}", "victim", 80, 1, 1) for index in range(40)]

        warning = window_coverage_warning(_flows(rows))

        self.assertIsNotNone(warning)
        self.assertIn(str(max(WINDOW_SECONDS)), warning)

    def test_does_not_warn_for_a_capture_spanning_the_widest_window(self):
        rows = [(index * 10.0, f"s{index}", "victim", 80, 1, 1) for index in range(40)]

        self.assertIsNone(window_coverage_warning(_flows(rows)))


class WindowSidecarTests(unittest.TestCase):
    """The sidecar joins by row position, so its length must match the file."""

    def _write_csv(self, path, rows, header_row_at=None):
        frame = _flows(rows)
        if header_row_at is None:
            frame.to_csv(path, index=False)
            return len(frame)
        head = frame.iloc[:header_row_at]
        tail = frame.iloc[header_row_at:]
        repeated = pd.DataFrame([dict(zip(frame.columns, frame.columns))])
        combined = pd.concat([head, repeated, tail], ignore_index=True)
        combined.to_csv(path, index=False)
        return len(combined)

    def test_sidecar_has_one_row_per_csv_row(self):
        rows = [(index * 0.5, f"s{index % 4}", "victim", 80, 1, 1) for index in range(20)]
        with TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "capture.csv"
            expected = self._write_csv(csv_path, rows)

            sidecar = pd.read_parquet(
                build_window_sidecar(csv_path, Path(temporary) / "cache")
            )

            self.assertEqual(len(sidecar), expected)
            self.assertEqual(sidecar.columns.tolist(), list(WINDOW_FEATURE_NAMES))

    def test_a_repeated_header_row_does_not_shorten_the_sidecar(self):
        """Dropping the bad row would misalign every row after it, silently."""

        rows = [(index * 0.5, f"s{index % 4}", "victim", 80, 1, 1) for index in range(20)]
        with TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "capture.csv"
            expected = self._write_csv(csv_path, rows, header_row_at=10)

            sidecar = pd.read_parquet(
                build_window_sidecar(csv_path, Path(temporary) / "cache")
            )

            self.assertEqual(len(sidecar), expected)
            self.assertTrue(np.isnan(sidecar["win1s_dst_flow_count"].iloc[10]))
            self.assertFalse(np.isnan(sidecar["win1s_dst_flow_count"].iloc[0]))

    def test_window_values_are_file_global_not_chunk_local(self):
        """The whole point of the sidecar: a value must not depend on chunking."""

        rows = [(index * 0.01, f"s{index}", "victim", 80, 1, 1) for index in range(50)]
        with TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "capture.csv"
            self._write_csv(csv_path, rows)

            sidecar = pd.read_parquet(
                build_window_sidecar(csv_path, Path(temporary) / "cache")
            )

            self.assertTrue((sidecar["win1s_dst_flow_count"] == 50).all())

    def test_files_sharing_a_stem_in_different_directories_do_not_collide(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            first_dir = root / "one"
            second_dir = root / "two"
            first_dir.mkdir()
            second_dir.mkdir()
            self._write_csv(
                first_dir / "day.csv",
                [(0.0, "a", "victim", 80, 1, 1)],
            )
            self._write_csv(
                second_dir / "day.csv",
                [(index * 0.01, f"s{index}", "victim", 80, 1, 1) for index in range(9)],
            )

            first = pd.read_parquet(build_window_sidecar(first_dir / "day.csv", cache))
            second = pd.read_parquet(build_window_sidecar(second_dir / "day.csv", cache))

            self.assertEqual(first["win1s_dst_flow_count"].iloc[0], 1)
            self.assertEqual(second["win1s_dst_flow_count"].iloc[0], 9)

    def test_reuses_a_cached_sidecar_and_rebuilds_on_request(self):
        rows = [(0.0, "a", "victim", 80, 1, 1)]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "capture.csv"
            self._write_csv(csv_path, rows)
            cache = root / "cache"

            first = build_window_sidecar(csv_path, cache)
            marker = first.stat().st_mtime_ns
            second = build_window_sidecar(csv_path, cache)

            self.assertEqual(first, second)
            self.assertEqual(second.stat().st_mtime_ns, marker)

            rebuilt = build_window_sidecar(csv_path, cache, rebuild=True)
            self.assertEqual(rebuilt, first)


class ContractWindowBlockTests(unittest.TestCase):
    """A bundle declaring a window block must get one, or refuse to score."""

    def _contract(self):
        window = ("win1s_dst_flow_count", "win60s_dst_unique_src")
        return FeatureContract(
            flow_features=("Flow Duration",),
            window_features=window,
            window_seconds=WINDOW_SECONDS,
            mean=(0.0,) * (1 + len(window)),
            scale=(1.0,) * (1 + len(window)),
        )

    def test_feature_names_put_the_window_block_after_the_flow_block(self):
        self.assertEqual(
            self._contract().feature_names,
            ("Flow Duration", "win1s_dst_flow_count", "win60s_dst_unique_src"),
        )

    def test_computes_the_window_block_from_identity_columns(self):
        rows = [(index * 0.01, f"s{index}", "victim", 80, 1, 1) for index in range(4)]
        flows = _flows(rows).assign(**{"Flow Duration": 0.0})

        prepared = self._contract().prepare(flows)

        # 4 flows to one destination inside one 1s bucket, signed-log1p'd.
        self.assertAlmostEqual(float(prepared[0, 1]), float(np.log1p(4)), places=5)

    def test_refuses_to_score_without_the_identity_columns_it_needs(self):
        flows = pd.DataFrame({"Flow Duration": [1.0]})

        with self.assertRaisesRegex(ValueError, "Dst IP"):
            self._contract().prepare(flows)

    def test_trusts_window_columns_the_caller_already_attached(self):
        """Training joins a sidecar; recomputing would defeat the whole design."""

        flows = pd.DataFrame(
            {
                "Flow Duration": [0.0],
                "win1s_dst_flow_count": [999.0],
                "win60s_dst_unique_src": [0.0],
            }
        )

        prepared = self._contract().prepare(flows)

        self.assertAlmostEqual(float(prepared[0, 1]), float(np.log1p(999)), places=4)


if __name__ == "__main__":
    unittest.main()
