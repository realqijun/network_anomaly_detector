# Evidence-first network anomaly detector redesign

## Outcome

The redesigned system detects suspicious behavior across network flows and time
windows, links the behavior back to implicated packets, and presents findings
whose attack labels are constrained by validated detector coverage. It supports
both PCAP replay and a local live sensor without maintaining separate analysis
paths.

The first release is local-first, advisory-only, and optimized for a learner or
analyst. It stores metadata and findings by default; uploaded captures and
payloads expire after analysis.

## Product rules

1. A high anomaly score means unusual, not malicious.
2. DDoS, brute force, and similar temporal attacks are classified from flows and
   observation windows, not isolated packets.
3. Packets are marked as evidence for a finding. Only a packet-specific detector
   may make a claim about one packet by itself.
4. XSS is reported with high confidence only when plaintext HTTP evidence or an
   application/WAF signal is available. Encrypted traffic alone is insufficient.
5. HTTP status and application latency come from an optional application-signal
   adapter. The dashboard shows that coverage as unavailable when it is absent.
6. The narrative layer may rephrase structured findings but may not create facts,
   attack labels, severities, or remediation actions.

## System shape

```text
PCAP replay ─┐
             ├─> observation ingestion ─> feature contract ─> detector suite
Zeek live ───┘                                      │               │
                                                    │               v
Application metrics/logs ─> application signals ───┴────> correlation
                                                                    │
                                                                    v
packet/flow evidence <─ finding store <─ structured findings <─ narrator
                                                                    │
                                                                    v
                                                               dashboard
```

The external analysis module exposes one small interface:

```python
report = analyzer.analyze(traffic_source, application_signal_source=None)
```

It hides ingestion, normalization, feature calculation, model execution,
correlation, persistence, and narration. The report is the same shape for a
completed PCAP and a periodically refreshed live analysis run.

## Modules and seams

### Traffic source

Interface: yield normalized packet and flow observations in event-time order.

Adapters:

- `PcapReplaySource` reads an uploaded PCAP/PCAPNG.
- `ZeekLiveSource` tails a local Zeek sensor.

Both adapters must populate the same flow identity, timestamps, counters, TCP
state, and packet evidence references. Source-specific fields remain optional
metadata and never silently become model inputs.

### Feature contract

Interface: transform observations into a versioned batch of flow and window
features.

This is the deepest module in the system. It owns field names, types, units,
missing-value policy, flow timeout behavior, window definitions, categorical
encoding, transformations, and validation. Training and inference import this
same implementation. A batch is rejected if its contract version does not match
the model bundle.

Suggested windows are 1, 10, and 60 seconds, grouped at least by destination,
destination service, source-to-destination pair, and source. They include rate,
cardinality, fan-out, TCP-state, response, and baseline-delta measurements.

### Detector suite

Interface: accept a feature batch and return detector signals with evidence
references.

The suite contains three detector types:

- A calibrated supervised classifier recognizes only dataset-supported attack
  families. A gradient-boosted tree model is the first baseline because the
  inputs are tabular and feature attribution is straightforward.
- A benign-only anomaly detector identifies behavior outside the learned normal
  distribution. Its output remains `unusual` unless other evidence supports a
  known family.
- Deterministic pattern detectors measure behaviors such as rate bursts, source
  fan-out, SYN/ACK imbalance, repeated authentication failures, and correlated
  application degradation.

Each signal contains the detector and model version, score, calibrated
probability when applicable, threshold, candidate family, feature contributions,
coverage requirements, and evidence references.

### Correlation

Interface: turn detector signals and application signals into findings and
incidents.

Correlation operates by target, attack family, and time. It suppresses repeated
row-level alerts and produces one finding for one behavior episode. Confidence
depends on independent evidence agreement and telemetry availability; severity
depends on magnitude, duration, affected assets, and application impact.

Examples:

- A DDoS finding combines a traffic-rate deviation, source cardinality, protocol
  behavior, and optionally increased latency or error rate.
- A brute-force finding combines repeated attempts with authentication failures.
  Without auth telemetry it is labeled a lower-confidence credential-guessing
  pattern.
- An XSS finding requires decoded HTTP request evidence or an application/WAF
  event. A flow classifier by itself produces only a web-attack candidate.

### Narrator

Interface: render a structured finding without changing its claims.

`TemplateNarrator` is always available and produces deterministic explanations.
An optional local-LLM adapter can improve readability. Before display, its output
is checked against the finding's allowed facts and falls back to the template on
failure or timeout.

### Finding store

Interface: save and query analysis runs, observations, signals, findings,
incidents, and evidence references.

SQLite is sufficient for the first local release. Raw uploads live in a temporary
area and are deleted after analysis; a short opt-in ring buffer may retain raw
packets for live forensic drill-down. Stored IP addresses should support optional
redaction or keyed pseudonymization.

## Dataset strategy

No single public dataset is sufficient evidence for all three target behaviors
in a real deployment. Use datasets for distinct jobs:

1. Start the deployable flow classifier with **NF-CSE-CIC-IDS2018-v2**. Its
   standardized 43-field NetFlow schema contains strong DDoS, DoS, brute-force,
   bot, and infiltration coverage, although its aggregated web-attack class is
   too small to support a trustworthy XSS claim by itself.
2. Use **NF-UQ-NIDS-v2** for broad pretraining and leave-one-source-out
   generalization tests. Preserve the dataset-of-origin column for grouping and
   auditing, never as a model input, and never randomly split its rows.
3. Use raw **CSE-CIC-IDS2018** PCAPs and its published attack schedule to test
   PCAP replay and episode-level correlation for DDoS, brute force, XSS, and SQL
   injection. Regenerate the production feature contract from these PCAPs.
4. Use **CICIoT2023** PCAP subsets to add explicit XSS, injection, brute-force,
   and modern flood diversity. Its supplied CSV rows are fixed packet windows,
   not normal bidirectional flows, so they remain a separate baseline.
5. Use **CICDDoS2019** as a DDoS stress and held-out generalization set rather than
   allowing its large attack classes to dominate general training.
6. Treat an HTTP-request dataset as a separate application-layer experiment for
   XSS/injection. It must not be mixed into a flow-feature table.
7. Learn the unknown-anomaly baseline from clean local traffic after excluding
   identifiers and validating the collection period. Do not continuously retrain
   from unreviewed live traffic, which would let attacks poison the baseline.

The counts, access notes, source citations, and additional specialist datasets
are recorded in [the dataset research note](../research/network-anomaly-datasets.md).

## Model bundle contract

Every deployable model is one immutable bundle containing:

- feature-contract and extractor versions;
- ordered feature schema, types, and units;
- fitted preprocessing objects;
- model weights;
- label taxonomy and coverage declaration;
- calibrated thresholds per class;
- training dataset manifests and split identifiers;
- evaluation metrics and creation timestamp.

Startup fails closed when any artifact is missing or incompatible. Missing input
features are errors unless the contract explicitly declares them optional; they
are never silently filled with zero.

## Training and evaluation

Training uses group-based splits by capture day, scenario, attacker/victim pair,
or source dataset. This prevents nearly identical flows from the same attack run
appearing in both training and test sets.

Report per attack family:

- precision, recall, F1, and precision-recall AUC;
- false positives per hour on benign-only captures;
- recall at a fixed false-positive budget;
- probability calibration error;
- episode detection latency;
- results on a dataset or campaign never used for training.

Accuracy and binary ROC-AUC are secondary. Release gates should include a
minimum precision per reportable family; classes that fail the gate remain
visible as experimental coverage and cannot create named findings.

## Findings experience

The default screen is incident-first rather than a table of every flow:

- timeline of traffic, anomaly, and application-health measurements;
- finding cards with family, confidence, severity, target, interval, and status;
- a plain-language “why this was flagged” summary;
- measured evidence with baseline comparisons;
- coverage gaps, including TLS or missing application telemetry;
- expandable flows and implicated packet indices;
- safe investigation steps, with no automatic blocking in the first release.

Raw scores remain available in technical detail but are not the headline.

## Local deployment

- Zeek runs on the monitored host or a sensor host with capture privileges.
- The unprivileged Python application performs ingestion, scoring, storage, and
  serves the browser interface.
- PCAP replay uses the same feature contract as live ingestion.
- Template explanations work offline. A local Ollama-compatible narrator is an
  optional enhancement, not a runtime dependency.
- The application-metrics adapter accepts a small normalized event format first;
  OpenTelemetry and Prometheus adapters can be added behind that seam.

## Delivery sequence

### Phase 1: trustworthy offline analysis

- Freeze the feature contract and model-bundle format.
- Build PCAP replay, flow/window features, and packet evidence references.
- Train and calibrate the family classifier and anomaly baseline.
- Add deterministic correlation and the incident-first findings UI.
- Validate on capture-day and dataset-held-out tests.

### Phase 2: live observation

- Add the Zeek live adapter and bounded event-time buffering.
- Add SQLite retention, sensor health, backpressure, and replay-safe ingestion.
- Verify that the same saved PCAP produces equivalent features in replay and
  live-sensor paths.

### Phase 3: application evidence and narration

- Add auth/HTTP/application-health signal ingestion.
- Raise confidence for brute force, XSS, 503, and latency findings only when the
  required evidence is present.
- Add the optional local narrator with grounding and fallback tests.

### Phase 4: hardening

- Add drift reports, per-family release gates, model rollback, retention controls,
  IP redaction, and cross-dataset regression evaluation.

## Acceptance criteria for the first release

- Training and inference produce identical feature values for a golden PCAP.
- A model bundle with a mismatched schema cannot load.
- The UI never labels an isolated high anomaly score as a named attack.
- DDoS and brute-force fixture captures produce one correlated incident each,
  with measurable evidence and implicated flows/packets.
- XSS over TLS is reported as unavailable without application evidence.
- Benign-only evaluation reports false findings per hour and stays within the
  configured alert budget.
- The system remains useful with no LLM installed.
