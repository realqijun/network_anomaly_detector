"""The guarantee the window sidecar exists to provide.

Training streams a capture in chunks; inference scores it whole. If those two
paths disagree on a single feature value, every metric a bundle reports is
measured against inputs the deployment will never see.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from feature_contract import WINDOW_SECONDS, FeatureContract
from training_pipeline import (
    TrainingConfig,
    build_window_sidecars,
    iter_prepared_chunks,
)

FLOW_FEATURES = ("Flow Duration", "Tot Fwd Pkts")
WINDOW_FEATURES = (
    "win1s_dst_flow_count",
    "win1s_dst_unique_src",
    "win60s_dst_flow_count",
    "win60s_dst_flow_count_delta",
)


def _capture(rows: int, repeated_header_at: int | None = None) -> pd.DataFrame:
    base = pd.Timestamp("2018-02-14 08:00:00")
    frame = pd.DataFrame(
        {
            "Flow Duration": np.arange(rows, dtype=float),
            "Tot Fwd Pkts": np.arange(rows, dtype=float) % 7,
            "Timestamp": [
                (base + pd.Timedelta(seconds=index * 0.3)).strftime("%Y-%m-%d %H:%M:%S")
                for index in range(rows)
            ],
            "Src IP": [f"10.0.0.{index % 5}" for index in range(rows)],
            "Dst IP": ["10.0.1.9" if index % 3 else "10.0.1.8" for index in range(rows)],
            "Src Port": [40000 + index for index in range(rows)],
            "Dst Port": [80 if index % 2 else 443 for index in range(rows)],
            "Protocol": [6] * rows,
            "SYN Flag Cnt": [1] * rows,
            "ACK Flag Cnt": [index % 2 for index in range(rows)],
            "TotLen Fwd Pkts": [100.0 + index for index in range(rows)],
            "TotLen Bwd Pkts": [50.0] * rows,
            "Tot Bwd Pkts": [1.0] * rows,
            "Label": ["Benign" if index % 4 else "DDoS" for index in range(rows)],
        }
    )
    if repeated_header_at is None:
        return frame
    header = pd.DataFrame([dict(zip(frame.columns, frame.columns))])
    return pd.concat(
        [frame.iloc[:repeated_header_at], header, frame.iloc[repeated_header_at:]],
        ignore_index=True,
    )


class TrainingInferenceParityTests(unittest.TestCase):
    def _contract(self):
        names = FLOW_FEATURES + WINDOW_FEATURES
        return FeatureContract(
            flow_features=FLOW_FEATURES,
            window_features=WINDOW_FEATURES,
            window_seconds=WINDOW_SECONDS,
            mean=(0.0,) * len(names),
            scale=(1.0,) * len(names),
        )

    def _trainer_matrix(self, csv_path, cache, chunk_size):
        config = TrainingConfig(
            data_dir=str(csv_path.parent),
            chunk_size=chunk_size,
            window_cache_dir=str(cache),
        )
        parts = [
            values
            for values, _ in iter_prepared_chunks([csv_path], self._contract(), config)
        ]
        return np.vstack(parts)

    def test_chunked_training_matches_whole_file_inference(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "02-14-2018.csv"
            frame = _capture(120)
            frame.to_csv(csv_path, index=False)

            trainer = self._trainer_matrix(csv_path, root / "cache", chunk_size=40)
            inference = self._contract().prepare(pd.read_csv(csv_path))

            self.assertEqual(trainer.shape, inference.shape)
            np.testing.assert_allclose(trainer, inference, rtol=1e-6, atol=1e-6)

    def test_the_chunk_size_does_not_change_a_single_value(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "02-14-2018.csv"
            _capture(120).to_csv(csv_path, index=False)

            small = self._trainer_matrix(csv_path, root / "a", chunk_size=17)
            large = self._trainer_matrix(csv_path, root / "b", chunk_size=1000)

            np.testing.assert_allclose(small, large, rtol=1e-6, atol=1e-6)

    def test_parity_survives_a_repeated_header_row_mid_file(self):
        """The row that would otherwise shorten the sidecar and misalign the join."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "02-14-2018.csv"
            _capture(120, repeated_header_at=57).to_csv(csv_path, index=False)

            trainer = self._trainer_matrix(csv_path, root / "cache", chunk_size=25)
            whole = pd.read_csv(csv_path)
            keep = pd.to_numeric(whole["Flow Duration"], errors="coerce").notna()
            inference = self._contract().prepare(whole[keep].reset_index(drop=True))

            self.assertEqual(trainer.shape[0], int(keep.sum()))
            np.testing.assert_allclose(trainer, inference, rtol=1e-6, atol=1e-6)

    def test_a_window_value_is_file_global_not_chunk_local(self):
        """Guards the specific bug: aggregates recomputed inside each chunk."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "02-14-2018.csv"
            _capture(120).to_csv(csv_path, index=False)

            trainer = self._trainer_matrix(csv_path, root / "cache", chunk_size=10)
            column = FLOW_FEATURES.index("Tot Fwd Pkts") + 1 + WINDOW_FEATURES.index(
                "win60s_dst_flow_count"
            )

            # Every row of one destination shares one file-global 60s count, so
            # a chunked computation would produce several distinct values here.
            self.assertLessEqual(len(np.unique(np.round(trainer[:, column], 6))), 2)


class SidecarGuardTests(unittest.TestCase):
    """The join is positional, so a wrong-length sidecar must never be used."""

    def _prepare(self, root, rows=60):
        csv_path = root / "02-14-2018.csv"
        _capture(rows).to_csv(csv_path, index=False)
        config = TrainingConfig(
            data_dir=str(root), chunk_size=25, window_cache_dir=str(root / "cache")
        )
        contract = FeatureContract(
            flow_features=FLOW_FEATURES,
            window_features=WINDOW_FEATURES,
            window_seconds=WINDOW_SECONDS,
            mean=(0.0,) * (len(FLOW_FEATURES) + len(WINDOW_FEATURES)),
            scale=(1.0,) * (len(FLOW_FEATURES) + len(WINDOW_FEATURES)),
        )
        sidecars = build_window_sidecars([csv_path], config)
        return csv_path, config, contract, sidecars

    def test_rejects_a_sidecar_shorter_than_its_capture(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path, config, contract, sidecars = self._prepare(root)
            truncated = pd.read_parquet(sidecars[csv_path]).iloc[:-5]
            truncated.to_parquet(sidecars[csv_path], index=False)

            with self.assertRaisesRegex(ValueError, "stale"):
                list(iter_prepared_chunks([csv_path], contract, config, sidecars))

    def test_rejects_a_sidecar_longer_than_its_capture(self):
        """A long sidecar shifts every row without tripping a per-slice check."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path, config, contract, sidecars = self._prepare(root)
            block = pd.read_parquet(sidecars[csv_path])
            pd.concat([block.iloc[:1], block], ignore_index=True).to_parquet(
                sidecars[csv_path], index=False
            )

            with self.assertRaisesRegex(ValueError, "stale"):
                list(iter_prepared_chunks([csv_path], contract, config, sidecars))


if __name__ == "__main__":
    unittest.main()
