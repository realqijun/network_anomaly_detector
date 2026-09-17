"""Public local inference interface.

Training lives in training_pipeline.py and notebooks/. Local callers only need
the Detector class or run_detector function exported here, or the command line
below for a one-off scoring run.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from detector_runtime import Detector, run_detector

__all__ = ["Detector", "main", "run_detector"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python detector.py",
        description="Score a CSV of flows through one immutable model bundle.",
    )
    parser.add_argument("flows", help="CSV whose columns are the bundle's contract features")
    parser.add_argument("--bundle", default="model_bundle", help="Model bundle directory")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Attach the top contributing features to each flagged row",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="How many features to attribute per row"
    )
    parser.add_argument(
        "--anomalies-only", action="store_true", help="Print only flagged rows"
    )
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    arguments = parser.parse_args(argv)

    import pandas as pd

    flows = pd.read_csv(arguments.flows)
    detector = Detector(arguments.bundle, device=arguments.device)
    detections = detector.detect(
        flows, explain=arguments.explain, explain_top_k=arguments.top_k
    )
    if arguments.anomalies_only:
        detections = detections[detections["is_anomaly"]]

    if detections.empty:
        print("No rows to report.")
        return 0

    columns = ["prediction", "confidence", "anomaly_score"]
    print(detections[columns].to_string())

    if arguments.explain:
        print()
        for position, explanation in detections["explanation"].items():
            if not explanation:
                continue
            print(f"Row {position}: {detections.loc[position, 'prediction']}")
            for item in explanation:
                side = "above" if item["direction"] >= 0 else "below"
                print(
                    f"  {item['feature']}: {item['observed_z']:.2f} standard "
                    f"deviations {side} the training baseline "
                    f"({item['basis']})"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
