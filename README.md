## Network Anomaly Detector

### Description

Network Anomaly Detector labels network flows with an attack family (DDoS, DoS, Brute Force, XSS, Injection, Bot, Infiltration), "Unknown anomaly", or "Benign".
It is hosted on a simple Flask app: upload a packet capture or flow CSV file, and the model processes it and displays the results.

---

### Project Details

This project is structured as so:
- The root directory holds the current bundle-based runtime.
  - `training_pipeline.py` trains `HybridDetectorNetwork` on CSE-CIC-IDS2018 flow CSVs and writes a versioned model bundle (`manifest.json` + `model.pt`).
  - `detector_runtime.py` loads one bundle and scores flow rows through `Detector`; `detector.py` re-exports it as the small public interface for local scripts.
  - `notebooks/train_cse_cic_ids2018_colab.ipynb` runs the same training code on Colab.
  - `tests/` covers the label mapping, the bundle contract, and the deployed Flask path.
- `deploy/` holds the Flask app.
  It loads the same `Detector` from a bundle directory (`MODEL_BUNDLE_DIR`, default `model_bundle`) and calls `detector.detect(...)` for every upload, so local inference and the website can never silently diverge.
- `legacy/` holds the original CTF-era `ConvAutoencoder` pipeline this project started with, kept for history.
  Nothing outside `legacy/` imports it; see `legacy/README.md`.
- `datasets/` holds training datasets.
  They are too large for GitHub; if you want the actual data, it is open-sourced, see [references](#references).

This project was started to try to game the network packet analysis CTFs, where I could quickly find out which packets may contain useful data on the flag.
Later I found out that the rate of false positives was too high without further filtering and analysis of the anomalous packets, which motivated moving to a calibrated, multi-class classifier over a single fixed-threshold autoencoder.

---

### Limitations

1. The program currently only supports "manual" detection, meaning that you would have to capture packets on your own, and upload it to the website to check for anomalies.
   A planned enhancement for this is to make it a program that runs locally and has access to the Network Interface Card (NIC), which allows the program to capture packets and label them in real time.

2. As this is an anomaly detector, false positives are expected, and network flows can be complex enough that a single model will not catch every attack family.
   Per-class confidence thresholds are calibrated during training to target a chosen precision, but real-world traffic still needs a human reviewing flagged flows rather than trusting a label outright.

---

The model was trained using the Intrusion detection evaluation dataset (CIC-IDS2017).

### References

1. Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.

**If you have any enquiries, or suggestions, I would love it hear it! Kindly reach out to me**

### Train on Google Colab

Open [the Colab training notebook](notebooks/train_cse_cic_ids2018_colab.ipynb).
Select a GPU runtime, and run its cells in order.
It uses KaggleHub to download the CSE-CIC-IDS2018 CSV mirror, holds out complete capture days, streams the dataset in bounded chunks, and downloads a versioned model-bundle ZIP when training is complete.
The notebook can clone a pushed GitHub branch or accept direct uploads of `training_pipeline.py` and `detector_runtime.py` for unpushed local work.

The reviewable source copy of the notebook's training implementation is [`training_pipeline.py`](training_pipeline.py).
Deployment does not import that module.
Local inference uses only the exported bundle and the small interface in [`detector.py`](detector.py):

```python
import pandas as pd

from detector import run_detector

flows = pd.read_csv("flows.csv")
findings = run_detector(flows, "model_bundle")
print(findings[["prediction", "confidence", "anomaly_score"]])
```

The input must contain the exact ordered feature contract recorded in the bundle's `manifest.json`.
Missing or non-finite values are rejected instead of being silently replaced with zero.

### Deployment

`deploy/app.py` serves the same `Detector` behind a Flask upload form.
`create_app(bundle_dir)` loads one bundle directory at startup and stores it on the app; every `/predict` request calls `detector.detect(...)` on the uploaded CSV or PCAP, so the website scores flows exactly the way `detector.py` does locally.
Point it at a trained bundle with the `MODEL_BUNDLE_DIR` environment variable.
That variable defaults to `model_bundle` in the app's working directory.
`deploy.sh <path-to-bundle>` copies a bundle into `deploy/model_bundle` for the Docker build.
Build the image from the repo root so the container also includes `detector_runtime.py`:
`docker build -f deploy/Dockerfile -t network-anomaly-detector .`
If the configured bundle directory is missing or malformed, the app still starts but flashes an error and refuses every `/predict` request rather than falling back to a legacy detector.
A CSV upload with missing or non-finite contract columns is rejected with the same error the local runtime raises, surfaced as a flash message instead of a 500 or a silently zero-filled prediction.
PCAP uploads go through `deploy/pcap_parser.py`, which renames CICFlowMeter's CLI output columns to the bundle's contract names.
That path requires a local Docker client plus a sibling `cfm` image and Docker socket access.
If those dependencies are missing, the app now fails closed with an explicit processing error instead of pretending the upload contained no flows.
That column mapping is derived from the published CSE-CIC-IDS2018 schema and has not yet been verified against a live `cicflowmeter` run, so prefer CSV uploads until that is confirmed against a real bundle.
