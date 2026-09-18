# Legacy CTF-era autoencoder pipeline

This directory holds the original single-flow `ConvAutoencoder` pipeline this project started with, before the bundle-based runtime in `training_pipeline.py` and `detector_runtime.py` replaced it (issue #3).
Nothing outside this directory imports these files.
They are kept only because `README.md` narrates this pipeline as the project's origin story, and because `legacy/results/` and `legacy/deploy_working/` are the only surviving record of a trained run of it.

- `models.py`, `dataset.py` - the `ConvAutoencoder` network and the windowed `TemporalDataset` it trained on.
- `parser.py`, `scale_temporal.py` - the raw-pcap-to-scaled-tensor preprocessing steps `train.py` expects to have already run.
- `train.py` - the training script that reads `working/X_train.npy` etc. and writes a `.pth` checkpoint plus an ROC-tuned threshold.
- `results/` - a trained `ConvAutoencoder` checkpoint, its error distribution plot, and the threshold and metrics from one training run.
- `deploy_working/` - the model, scaler, and threshold artifacts the old `deploy/anomaly_detector_service.py` loaded before deployment moved to `Detector` and one model-bundle directory.

`AnomalyDetector`/`Autoencoder`, a second, unreachable detector class that referenced a `models.Autoencoder` class that never existed in this repo, was deleted rather than moved here.
It never ran, so it is not part of this project's history.
