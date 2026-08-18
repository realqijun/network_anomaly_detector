"""Public local inference interface.

Training lives in training_pipeline.py and notebooks/. Local callers only need
the Detector class or run_detector function exported here.
"""

from detector_runtime import Detector, run_detector

__all__ = ["Detector", "run_detector"]
