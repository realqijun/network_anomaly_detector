# 0001 - Operational detection upgrade

Issues: #4, #5 (partial), #6, #7 (partial), #8, #9, #10.
Status: plan, revision 2 after review.
Branch: `claude/dazzling-clarke-200d7v`.
Base: `fix/issue-3-unify-runtime` (open PR #11), because issue #3 is complete but not yet merged into `main`.

Revision 2 reworks the window-feature mechanism, the operational-metric sampling basis, the explanation basis for unknown anomalies, the commit order, and the scope of issues #5 and #7.
The rejected designs are recorded at the end so they are not re-proposed.

## Why these issues are one change

Issue #9 asks for a frozen, versioned feature contract that lives in one authoritative place.
Every other issue in this set reads or writes that contract.

- #6 adds temporal window features, which are new contract fields that training and inference must compute identically.
- #5 changes how rows are grouped into splits, and the split manifest belongs in the bundle next to the contract.
- #4 changes which metrics and which threshold policy the bundle records, also manifest content.
- #8 needs the ordered contract feature names and the fitted scaler baseline to express an attribution.
- #7 needs flow identity fields that the contract must name canonically so evidence can be joined back to scored rows.
- #10 is the test harness that keeps all of the above from silently regressing.

Splitting these would mean six pull requests churning the same manifest schema and the same two wiring points.

## What this PR does not close

Stated up front so the pull request does not overclaim.

- Issue #5 requires an untouched external benchmark reserved for final testing.
  This PR builds the grouped-split machinery, the split manifest, and the leakage validator, and plumbs an `external_test` partition, but names no second corpus and adds no download for one.
  Issue #5 stays open with that one criterion outstanding.
- Issue #7 requires exported objects and a demonstrated analyst workflow on a PCAP.
  This PR ships the evidence library and its CLI (packet indices, stream identifiers, first and last seen, plaintext HTTP request and response lines).
  Object export, TCP reassembly across segments, and the web wiring are deferred, because the only end-to-end path for them is the PCAP upload, which needs a Docker `cfm` image this repository does not ship (`deploy/cicflowmeter-docker/` is empty).
  Issue #7 stays open with those criteria outstanding.
- No trained v2 bundle exists, so no real CSE-CIC-IDS2018 numbers are refreshed here.
  Everything is validated on synthetic fixtures.
  Retraining through the Colab notebook is the follow-up.

Per `CONTEXT.md`, what this PR produces are **detector signals** with **evidence** references and per-row explanations.
Correlation, **findings**, **confidence** and **severity** remain future work.
The README wording follows that glossary rather than calling a scored row a finding.

## Current state this plan builds on

`detector_runtime.py` owns the inference-side contract inline: it reads `manifest["feature_contract"]`, validates the transform name, and implements `_prepare` (signed log1p, then standardize, reject non-finite) at lines 63-64 and 126-142.
`training_pipeline.py` owns the training side separately: `infer_feature_columns`, `_clean_chunk` (which re-implements the same signed log1p at line 195), `fit_feature_contract`, and `_write_bundle`.
The transform is written twice, which is the drift issue #9 describes.

`deploy/pcap_parser.py:21-100` holds a third partial copy of the schema as `CLI_TO_CONTRACT_FEATURE`.

Splits are filename substring matching in `partition_csv_files` (`training_pipeline.py:124-139`), with no manifest and no validator.
Evaluation is `classification_report` plus a single `anomaly_roc_auc`, and the anomaly threshold is a fixed benign quantile (`training_pipeline.py:433-459`).
There is no explanation path, no evidence path, and no `.github/` directory.

## Target architecture

```text
                      feature_contract.py  (authoritative, versioned)
                       |          |            |
        normalize +    |          |            |  FeatureContract.prepare()
        alias columns  |          |            |  (one transform, both sides)
                       v          v            v
 training_pipeline.py            detector_runtime.py           deploy/pcap_parser.py
   |  splits.py                    |  explain.py                 (alias table moves
   |  evaluation.py                |                              into the contract)
   v                               v
 model bundle  ------------------> Detector.detect(explain=True)
 (manifest v2)
                              evidence.py  (standalone library + CLI, joined by flow key)
```

The rule that makes the contract authoritative: `training_pipeline.py` and `detector_runtime.py` may not compute a feature value or a transform themselves.
They call `feature_contract.py`.

## New modules

### `feature_contract.py` - issues #9 and #6

Versioning, with an explicit precedence rule:

- `BUNDLE_FORMAT_VERSION = 2` is the manifest layout version.
  It moves here from `detector_runtime.py`, which re-exports it so `training_pipeline.py:27` keeps working.
- `FEATURE_CONTRACT_VERSION = 1` is a monotonic integer describing the feature list, its order, and the transform semantics.
  `COMPATIBLE_CONTRACT_VERSIONS = frozenset({1})` is the documented compatible set, so a future additive change can widen the set without a container bump.
- These are separate axes on purpose: a feature change must not force a manifest-layout bump, and vice versa.
- Load order is fixed and both are always checked: bundle format first, then contract version, then transform name.
  A bundle whose format is 2 but whose contract version is stale must still be rejected, and the error names both the found and the expected contract version.
  There is a test for exactly that divergent case, because bumping both at once would otherwise leave the interesting path untested.

Canonical identity fields, never model inputs, required to compute window features and to join evidence:

```python
IDENTITY_COLUMNS = ("Timestamp", "Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol")
```

`Protocol` and `Dst Port` stay eligible as model features, matching `DEFAULT_EXCLUDED_COLUMNS` today; the rest are identity only.

`WINDOW_SOURCE_COLUMNS` names the measurement columns the window block aggregates, so `window_inputs_available` has something concrete to check:
`TotLen Fwd Pkts`, `TotLen Bwd Pkts`, `Tot Fwd Pkts`, `Tot Bwd Pkts`, `SYN Flag Cnt`, `ACK Flag Cnt`.

`COLUMN_ALIASES` folds three naming conventions into the canonical CSE-CIC-IDS2018 short names:

- the CIC-IDS2017 long names (`Total Fwd Packets` to `Tot Fwd Pkts`, `Source IP` to `Src IP`);
- the CICFlowMeter CLI snake_case names currently in `deploy/pcap_parser.CLI_TO_CONTRACT_FEATURE`;
- whitespace-padded variants, which the datasets ship with.

Note that the existing CLI table has no identity entries, so `src_ip`, `dst_ip`, `src_port`, `dst_port`, `timestamp` and `protocol` are new aliases, not moved ones.

`normalize_columns(frame)` strips header whitespace and applies the alias table.
Every entry point calls it first, so one spelling reaches the contract.

Timestamp parsing gets its own function, `parse_timestamps(series)`, because getting it wrong silently reorders every window bucket.
These CSVs carry a local-time string with no zone, and `pd.to_datetime(..., dayfirst=True)` will silently accept `03/01/2018` as 3 January.
`parse_timestamps` tries an explicit ordered list of `format=` strings, takes the first that parses every non-null value, and raises naming the offending sample if none does.
It never guesses with `dayfirst`.

#### Window features

`WINDOW_SECONDS = (1, 10, 60)`.
Each bucket is `floor(epoch_seconds / W)`, so offline replay and a future live feed bucket identically without a capture start time.
For each `W` the contract defines eleven fields named `win{W}s_<name>`, 33 in total:

| field | grouping | source columns | meaning |
| --- | --- | --- | --- |
| `dst_flow_count` | `Dst IP` | - | destination pressure |
| `dst_unique_src` | `Dst IP` | `Src IP` | fan-in, the DDoS signal |
| `dst_unique_port` | `Dst IP` | `Dst Port` | port spread, the scan signal |
| `src_flow_count` | `Src IP` | - | source activity |
| `src_unique_dst` | `Src IP` | `Dst IP` | fan-out, the sweep signal |
| `pair_flow_count` | `Src IP`, `Dst IP` | - | repetition, the brute-force signal |
| `service_flow_count` | `Dst IP`, `Dst Port` | - | service pressure |
| `dst_bytes_rate` | `Dst IP` | `TotLen Fwd Pkts`, `TotLen Bwd Pkts` | bytes per second into the destination |
| `dst_pkt_rate` | `Dst IP` | `Tot Fwd Pkts`, `Tot Bwd Pkts` | packets per second into the destination |
| `syn_ack_imbalance` | `Dst IP` | `SYN Flag Cnt`, `ACK Flag Cnt` | `(SYN - ACK) / (SYN + ACK + 1)`, the half-open signal |
| `dst_flow_count_delta` | `Dst IP` | - | this bucket minus the same destination's previous bucket |

`compute_window_features(frame)` returns exactly `WINDOW_FEATURE_NAMES` in order.
It is deterministic, pure pandas, and the only implementation.
`window_inputs_available(frame)` returns `(bool, missing_columns)` checking identity and source columns together.

`syn_ack_imbalance` is already bounded to `[-1, 1]`, so the global `signed_log1p` compresses it to about `[-0.69, 0.69]`.
That is accepted deliberately: one global transform keeps the contract simple and the compression is monotonic, so it costs resolution but not ordering.
A per-field transform is recorded as future work rather than added now.

#### How window features survive chunked training

This is the part revision 1 got wrong, so it is specified precisely.

`training_pipeline._read_chunks` yields 100k-row slices.
A `groupby` inside a chunk is not the same number as over the file: measured on 250k synthetic flows to one destination at 1000 flows per second, `win60s_dst_flow_count` takes values `{10000, 60000}` over the whole frame and `{10000, 20000, 40000, 60000}` per 100k-row chunk, with 120,000 of 250,000 rows differing.
`dst_flow_count_delta` is worse: it needs the previous bucket for the same destination, which is in the previous chunk.

The mechanism is a **whole-file window pre-pass writing a per-file sidecar**:

1. `build_window_sidecar(csv_path, cache_dir)` reads only `IDENTITY_COLUMNS + WINDOW_SOURCE_COLUMNS` from the file, which is about nine columns of roughly eighty, and casts IP columns to pandas categoricals to bound memory.
2. It calls `compute_window_features` once over the complete file and writes the 33 float32 columns to `cache_dir/<file stem>.window.parquet`, indexed by original row position.
3. `_read_chunks` gains an explicit `identity_columns` argument unioned into `usecols`, because `infer_feature_columns` has already dropped `Timestamp`, `Src IP` and `Dst IP` via `DEFAULT_EXCLUDED_COLUMNS`, so today they are never loaded at all.
   Each chunk then joins its slice of the sidecar by row position.
4. `_clean_chunk` filters rows (unmapped label, non-finite features) **after** the join, so a dropped row takes its sidecar row with it.
   Window values are therefore always file-global and never depend on which rows survived filtering, which matches inference, where `Detector` drops nothing and raises instead.

The sidecar is a cache: `TrainingConfig.window_cache_dir` defaults to `<output_dir>/.window_cache`, it is keyed by file stem plus source mtime and size, and it is rebuilt when stale.

Because both sides now call `compute_window_features` over a complete frame, parity is exact by construction rather than by hope.
The parity test enforces it the way the reviewer specified: the trainer path over a file deliberately read in at least three chunks, versus `Detector.prepare` over the same file read whole, asserting matrix equality, not two calls into the same helper.

`enable_window_features` is tri-state, because a silent degrade would let the default configuration produce a bundle that closes #6 on paper while containing zero window features:

- `"require"` is the default.
  Training fails with a message naming the missing identity columns.
- `"auto"` degrades to no window block and must write `window_features_disabled_reason` into the manifest.
- `"off"` is explicit and also records a reason.

`deploy/app.py` renders a coverage banner when the loaded bundle declares no window block, satisfying `CONTEXT.md`'s rule that missing telemetry must be visible to the user.

#### The contract object

`FeatureContract` is a frozen dataclass holding `version`, `flow_features`, `window_features`, `transform`, `mean`, `scale`, `missing_value_policy`, `window_seconds`, and `window_features_disabled_reason`.

- `feature_names` is `flow_features + window_features`, the ordered model input.
- `prepare(frame)` is the single transform: normalize columns, compute or accept window features, check for missing columns, coerce to numeric, reject non-finite, signed log1p, standardize.
- `to_manifest()` and `from_manifest()` round-trip, with `from_manifest` applying the version precedence above.

Non-finite handling stays `reject`, matching what issue #3 landed.

### `splits.py` - issue #5 (partial)

`capture_group(path)` derives a stable whole-capture group key from a filename.
The corpus naming is an **unverified assumption** and the plan says so rather than hiding it: the notebook downloads `solarmainframe/ids-intrusion-csv`, commonly mirrored as `02-14-2018.csv` (MM-DD-YYYY), while `TrainingConfig.validation_patterns` defaults to `("16-02-2018", ...)` (DD-MM-YYYY).
If the mirror is MM-DD-YYYY then `partition_csv_files` raises "None of the validation_patterns matched" before training starts today, which would be a pre-existing bug this PR must not preserve behind a new module.
No Kaggle credentials are available here, so `capture_group` is written to handle the ambiguity rather than to bet on one convention:

- match any `NN-NN-YYYY` or `YYYY-NN-NN` token and disambiguate by magnitude, first field above 12 means day-first, second field above 12 means month-first;
- when genuinely ambiguous, fall back to the raw filename stem, which is still a valid whole-file group key;
- normalize a recognized date to `YYYY-MM-DD` so both conventions collapse to one key.

`validation_patterns` are matched against the **group key and the filename**, and the "nothing matched" error lists the actual group keys found, so a wrong convention is diagnosable in one run instead of guessed at.
The defaults are updated to the MM-DD-YYYY spellings with a comment recording why, and both spellings are covered by tests.

- `GroupedSplit` holds `train`, `validation`, `test`, `external_test`, and `group_key`.
- `build_grouped_split(...)` assigns whole groups, never individual files from a shared group, so two shards of one capture day cannot land on opposite sides.
- `split_manifest(split)` is the serializable record for the bundle.
- `validate_split_manifest(manifest)` raises `SplitLeakageError` when a group appears in more than one partition, or when training or validation is empty.
  The trainer calls it before the first epoch, not only in tests.

`partition_csv_files` stays as a thin wrapper returning the same 2-tuple the notebook unpacks (cell 7); the `test` and `external_test` partitions are reachable only through `build_grouped_split`.
The existing assertion in `tests/test_training_pipeline.py:46-52` uses `day-20.csv` with pattern `"21"`, which survives the stem fallback unchanged, so that test keeps passing as written.

### `evaluation.py` - issue #4

Pure functions over `(targets, probabilities, anomaly_scores, classes)`:

- `precision_recall_auc(truth, scores)` via `average_precision_score`.
- `false_positives_per_hour(truth, flagged, benign_hours)`.
- `recall_at_false_positive_budget(truth, scores, budget_per_hour, benign_hours)`.
- `top_k_precision(truth, scores, k)` for `k` in `(10, 50, 100)`.
- `expected_calibration_error(truth, probabilities)`.

#### Prevalence basis

Every one of those metrics is prevalence-dependent.
`_collect_validation` currently caps each class at `max_validation_rows_per_class = 100_000` (`training_pipeline.py:348, 358-368`), and in CSE-CIC-IDS2018 benign vastly outnumbers each attack family, so capping both destroys prevalence and would write numerically meaningless numbers into the manifest as operational truth.
That is the exact failure issue #4 exists to prevent.

The fix separates two passes that were conflated:

- **Model selection**, per epoch, keeps the existing per-class cap.
  It optimizes macro F1, which is prevalence-insensitive by design.
- **Operational evaluation**, once, at the end, uses **uniform** subsampling, which preserves prevalence, bounded by `max_evaluation_rows` (default 5,000,000).
  The manifest records `metrics.operational.prevalence_basis = {"sampling": "uniform", "rate": r, "rows_seen": n}`, so a reader can always tell what the numbers stand on.

#### Benign hours

`benign_hours` is never taken from the sampled evaluation frame, whose span is not the capture duration.
`scan_benign_hours(validation_files)` is a dedicated cheap pass reading only `Timestamp` and `Label`, taking the min and max over benign rows per file and summing the per-file spans.
`TrainingConfig.benign_hours_override` is the only other source, and the manifest records which was used under `metrics.operational.benign_hours_source`.

#### Threshold policy

`select_anomaly_threshold(policy, ...)` supports `"false_positive_budget"` (the new default) and `"benign_quantile"` (the current behavior, retained for reproducibility).

The budget policy is defined exactly, because revision 1 left the degenerate cases open:

> Search the sorted unique benign anomaly scores, not a fixed grid.
> Choose the smallest threshold whose realized false positives per hour is at or below `max_false_positives_per_hour`.
> If no threshold satisfies the budget, raise and refuse to write the bundle, naming the best achievable rate.

Refusing to write is deliberate: a bundle whose `thresholds.anomaly` is whatever a loop left behind is worse than no bundle.
There is a test for the unreachable-budget case.

The returned `ThresholdPolicy` record (policy, driving parameter, threshold, realized rate) is written to `manifest["thresholds"]["policy"]`.

`operational_metrics(...)` assembles the block, including per-family PR-AUC, into `manifest["metrics"]["operational"]`.
`TrainingConfig` gains `threshold_policy`, `max_false_positives_per_hour`, `top_k_values`, `benign_hours_override`, and `max_evaluation_rows`.

### `explain.py` - issue #8

Attribution keeps the trained hybrid network rather than swapping in a gradient-boosted model.
`docs/architecture/redesign.md:94-96` names a gradient-boosted tree as the first baseline, so this is a **recorded deviation**, not an oversight: issue #8's acceptance criteria are model-agnostic and its proposed change says "if needed", and replacing the model family would invalidate the bundle this PR is already migrating.
The deviation and its rationale go in the README next to the explanation limitation.

There are **two bases**, because the runtime produces two kinds of non-benign row and one method does not cover both:

- A **named-family** row comes from a class logit, so the basis is gradient times input on that logit: `EXPLANATION_BASIS_CLASS = "gradient_x_input"`.
- An **`Unknown anomaly`** row comes from reconstruction error, not from any logit (`detector_runtime.py:183-184`), so there is no index to differentiate.
  Its basis is the per-feature decomposition of the score itself, `(z_i - reconstruction_i) ** 2`, which is exactly faithful because the score is the mean of those terms (`detector_runtime.py:168`): `EXPLANATION_BASIS_ANOMALY = "reconstruction_error_decomposition"`.

`explanation_basis` is recorded per row, and `EXPLANATION_METHOD` is a mapping from basis key to method name, written into the manifest along with `EXPLANATION_VERSION = 1` so an explanation is tied to the model that produced it.

`explain_rows(...)` returns, per row, a list of `{feature, contribution, observed_z, direction}` sorted by absolute contribution, `top_k` default 5.
`observed_z` is the standardized value, so "3.4 standard deviations above the training baseline" reads directly, satisfying the observed-versus-baseline requirement without shipping training data.

Gradients cannot be taken inside the existing scoring loop: `Detector.detect` runs under `torch.inference_mode()` (`detector_runtime.py:160`), and a tensor produced there cannot later set `requires_grad_(True)` (`RuntimeError: Setting requires_grad=True on inference tensor outside InferenceMode is not allowed`).
So `explain=True` runs a **second pass** under `torch.enable_grad()` over freshly constructed tensors, restricted to the flagged subset.
The main scoring loop is unchanged, and the "only flagged rows pay" cost story is therefore true rather than aspirational.

Documented limitation, repeated in the README: attribution supports analyst interpretation and is not proof.

### `evidence.py` - issue #7 (partial)

Scoped down after review.
Ships as a standalone library plus a CLI; model scores and evidence stay separate structures joined by flow key.

- `FlowKey` is the canonical 5-tuple, direction-normalized so both directions of a conversation map to one key.
- `FlowEvidence` holds `packet_indices` (bounded, default 50), `packet_count`, `stream_id`, `first_seen`, `last_seen`, `http_requests` (method, path, host, user agent, plaintext only) and `http_responses` (status, content type, content length).
- `collect_evidence(pcap_path, flow_keys=None, max_packets_per_flow=50)`.
  `flow_keys=None` means **all flows**, defined explicitly rather than left undefined.
- `python -m evidence <pcap> [--flow a:p-b:q/proto]` prints the same structure as JSON, and is the documented sample workflow.

Cut from this PR and deferred: `exported_objects` (which needs TCP reassembly plus chunked and gzip handling), and the `deploy/app.py` evidence wiring.

Payload bytes are never retained; only lengths and parsed request or status lines, matching the redesign note's retention rule.

Fixtures: `deploy/dhcp.pcap` is four DHCP-over-UDP packets and `deploy/ipv4frags.pcap` is three ICMP packets, so neither contains HTTP and neither can prove the HTTP path.
Tests therefore build their own PCAP with scapy, and the repository fixtures are used only to prove the reader handles a real file without HTTP.

### `.github/workflows/ci.yml` and `requirements-dev.txt` - issue #10

Workflow `CI` on `push` and `pull_request`, `ubuntu-latest`, Python 3.11:

1. checkout;
2. `pip install -r requirements-dev.txt`, with pip caching and the CPU torch index, because the default `torch` wheel is roughly 800 MB per run;
3. `python -m compileall -q . -x '(legacy|\.git|\.venv|\.agents|\.claude)'`;
4. `python -m unittest discover -s tests -t . -v`.

`requirements-dev.txt` is the root runtime pins **plus Flask and scapy**, because `requirements.txt` contains neither, while `tests/test_deploy_app.py:14` imports `app` (which imports Flask) and `tests/test_evidence.py` needs scapy.
Revision 1's "runtime pins plus nothing else" would have gone red on a fresh runner.
Verification includes creating a clean venv from this file, not reusing a developer's existing one.

`tests/__init__.py` is added, because `python -m unittest discover -s tests -t .` fails today with `ImportError: Start directory is not importable`.
Confirmed: adding the empty file makes that exact command pass with all 19 existing tests.

## Changes to existing files

| File | Change |
| --- | --- |
| `detector_runtime.py` | Import `FeatureContract`; delete the inline `_prepare` and inline contract validation; `BUNDLE_FORMAT_VERSION` becomes a re-export; `detect` gains `explain`, implemented as a second `enable_grad` pass over flagged rows. |
| `training_pipeline.py` | `_read_chunks` gains `identity_columns`; window sidecar build and per-chunk join; filtering moves after the join; `splits.build_grouped_split` plus `validate_split_manifest`; separate operational evaluation pass with uniform sampling; `evaluation.operational_metrics` and `select_anomaly_threshold`; v2 manifest. |
| `detector.py` | Keep the re-export, add an argparse `main()` so `--explain` is reachable from a CLI, which #8 requires. |
| `deploy/pcap_parser.py` | Delete `CLI_TO_CONTRACT_FEATURE`; `rename_cli_columns_to_contract` delegates to `feature_contract.normalize_columns`. |
| `deploy/app.py` | Request `explain=True`; pass explanations to the template; render the window-coverage banner; surface a contract-mismatch load failure distinctly. No evidence wiring (deferred). |
| `deploy/templates/results.html` | Per-row expandable "why this was flagged" list; coverage banner. |
| `deploy/Dockerfile` | Copy `feature_contract.py` and `explain.py` next to `detector_runtime.py`; without this the image breaks on import (`deploy/Dockerfile:13-14` copies only `deploy/` and `detector_runtime.py`). |
| `deploy/requirements.txt` | Unchanged; scapy stays pinned here. Root `requirements.txt` **adds** scapy rather than moving it, so the container keeps working. |
| `notebooks/train_cse_cic_ids2018_colab.ipynb` | Cell 11 displays the operational block; cell 13 stops hand-preparing features now that the contract does it. |
| `tests/test_training_pipeline.py` | v2 fixture; assert the new manifest blocks. |
| `tests/test_deploy_app.py` | v2 fixture; end-to-end smoke assertion that an explanation reaches the rendered page. |
| `README.md` | Contract version and schema policy; operational metrics replacing the ROC-AUC framing; split policy; explanation limitation and the recorded deviation from the redesign note; the evidence CLI; local check commands. Also fix `README.md:41`, which still says CIC-IDS2017 while everything else says CSE-CIC-IDS2018. |
| `.gitignore` | Root-anchor `artifacts/` to `/artifacts/` so `docs/artifacts/` stays tracked. Already done. |

## Commit points

CI moves first, so every later commit is validated by the workflow rather than none of them.
Each commit leaves `python -m unittest discover -s tests -t .` green.

1. `ci: add unit test workflow and document local checks`
   `.github/workflows/ci.yml`, `requirements-dev.txt`, `tests/__init__.py`, README check section.
   Self-contained, and it surfaces the discovery problem immediately.
   Issue #10.
2. `feat: add authoritative versioned feature contract`
   `feature_contract.py` (versions and precedence, identity and source columns, aliases, `normalize_columns`, `parse_timestamps`, `FeatureContract`, `prepare`, manifest round-trip) and `tests/test_feature_contract.py`.
   Issue #9.
3. `feat: add temporal window aggregation features`
   `WINDOW_SECONDS`, `WINDOW_FEATURE_NAMES`, `compute_window_features`, `window_inputs_available`, `build_window_sidecar`, and pure unit tests only.
   The parity test is **not** here, because the trainer is not wired until commit 7.
   Issue #6.
4. `feat: add leakage-resistant grouped split manifests`
   `splits.py` and `tests/test_splits.py`.
   Issue #5, partial.
5. `feat: add operational detection metrics and threshold policy`
   `evaluation.py` and `tests/test_evaluation.py`.
   Issue #4.
6. `feat: add feature attribution for named detector signals`
   `explain.py` plus `tests/test_explain.py` only; the `Detector.detect(explain=True)` wiring lands in commit 7 so `detector_runtime.py` is touched once.
   Issue #8.
7. `refactor: adopt feature contract v2 across training and runtime`
   `detector_runtime.py`, `training_pipeline.py`, `detector.py`, `deploy/pcap_parser.py`, `deploy/app.py`, `deploy/templates/results.html`, `deploy/Dockerfile`, the notebook, the two updated test modules, and the train-versus-inference parity test.
8. `feat: add pcap evidence library and cli`
   `evidence.py` and `tests/test_evidence.py`.
   Issue #7, partial.
9. `docs: document contract, metrics, splits and evidence`
   README sections and the `CONTEXT.md`-aligned wording.

## How to test and verify

### Setup

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
```

A clean venv is part of the verification, not an aside: revision 1's dev requirements would have passed on a developer machine and failed in CI.

### The full suite

```sh
python -m unittest discover -s tests -t . -v
```

Same command CI runs, satisfying #10's "README documents how to run the same checks locally".

### Per issue

**Issue #9, frozen versioned contract.**

```sh
python -m unittest tests.test_feature_contract -v
```

Verifies: alias folding produces canonical names from all three conventions; `feature_names` ordering is stable; `to_manifest` / `from_manifest` round-trips exactly; a stale `bundle_format_version` raises; a **matching** bundle format with a stale `feature_contract.version` also raises and names both found and expected, which is the divergence case that would otherwise go untested; an unknown transform raises; `parse_timestamps` raises rather than guessing on an ambiguous `03/01/2018` and succeeds on an explicit format.

Authoritativeness check:

```sh
grep -rn "log1p" --include=*.py . | grep -v legacy | grep -v tests | grep -v '\.agents'
```

All hits are in `feature_contract.py`.

**Issue #6, temporal window features.**

```sh
python -m unittest tests.test_feature_contract.WindowFeatureTests -v
```

Verifies: 60 flows from 30 sources onto one destination within one second give `win1s_dst_unique_src == 30` and `win60s_dst_flow_count == 60`; a repeated pair raises `pair_flow_count`; a SYN-flood fixture drives `syn_ack_imbalance` positive; `dst_flow_count_delta` is zero in a destination's first bucket and positive in a growing one; a frame missing `Timestamp` raises naming the column instead of scoring.

The parity test (in commit 7, `tests.test_training_pipeline`) is the one that matters:
write a fixture CSV, run the trainer path over it with `chunk_size` forced small enough to produce at least three chunks, run `Detector`'s prepare path over the same file read whole, and assert the two matrices are equal.

Degrade visibility:
`enable_window_features="require"` on a frame without identity columns fails with the missing columns named; `"auto"` writes `window_features_disabled_reason`; the deploy app renders the coverage banner for a bundle with no window block.

**Issue #5, grouped splits.**

```sh
python -m unittest tests.test_splits -v
```

Verifies: two shards of one capture day always land in the same partition; `validate_split_manifest` raises `SplitLeakageError` when a group is in both training and validation; an empty validation partition raises; `capture_group` normalizes `02-14-2018` and `14-02-2018` to the same `YYYY-MM-DD` key where unambiguous and falls back to the stem where not; `day-20.csv` with pattern `"21"` still behaves as `tests/test_training_pipeline.py:46-52` asserts; the "nothing matched" error lists actual group keys.

**Issue #4, operational metrics.**

```sh
python -m unittest tests.test_evaluation -v
```

Verifies: `precision_recall_auc` matches `sklearn.average_precision_score` on a known vector; `false_positives_per_hour` matches a hand-computed value for a known hour span; `recall_at_false_positive_budget` returns a threshold whose realized rate is at or under budget; `top_k_precision` matches a hand-computed ranking; `select_anomaly_threshold` **raises** when the budget is unreachable rather than writing a bundle; the `ThresholdPolicy` record names which policy ran; `prevalence_basis` and `benign_hours_source` appear in the manifest.

**Issue #8, explainability.**

```sh
python -m unittest tests.test_explain -v
```

Verifies: for a network whose first layer weights exactly one feature, `explain_rows` ranks that feature first; `top_k` is respected; `Benign` rows carry no explanation; an `Unknown anomaly` row gets the reconstruction basis and its contributions sum to the reported `anomaly_score` times the feature count, which is the faithfulness check the decomposition makes possible; a named-family row gets the gradient basis; the manifest records the method mapping and version; the deploy smoke test asserts a contributing feature name appears in the rendered HTML.

**Issue #7, evidence.**

```sh
python -m unittest tests.test_evidence -v
python -m evidence deploy/dhcp.pcap
```

Verifies against a scapy-built in-test PCAP with one plaintext HTTP conversation and one unrelated UDP flow: packet indices index back into the capture; both directions share one `FlowKey` and one `stream_id`; method, path and host are recovered; a response reports status and content type with no payload; an absent flow key yields no entry rather than a fabricated empty one; `flow_keys=None` returns every flow.
The CLI run against the repository's DHCP fixture proves the reader handles a real file that contains no HTTP.

**Issue #10, CI.**

Verified by the workflow running on the pull request and by the local command above.
To confirm the failure mode is real rather than decorative, break a contract assertion, confirm the suite goes red, and revert.

### End-to-end smoke

```sh
python -m unittest tests.test_deploy_app -v
```

Builds a v2 bundle in a temporary directory, starts the Flask test client, posts a CSV whose columns are the contract feature names, and asserts a results page containing a prediction, an anomaly score, and for a flagged row an explanation.
That is the single end-to-end inference path #10 asks for.

## Rejected alternatives

Recorded so they are not re-proposed.

- **Per-chunk window aggregates.** Measured wrong: 120,000 of 250,000 rows differ from whole-file values, and `dst_flow_count_delta` is not computable inside a chunk at all.
- **Bucket-aligned chunking with a one-bucket lookback.** This does give exact parity, including the delta feature, and was verified experimentally.
  It is rejected anyway because it requires each file to be time-sorted, which these CSVs do not guarantee, and verifying sortedness costs a pass that the sidecar approach spends more usefully.
- **Dropping `dst_flow_count_delta` to 30 window features.** Unnecessary once the sidecar makes whole-file computation the norm, and the baseline-delta signal is explicitly requested by issue #6.
- **Swapping the model family to gradient-boosted trees for explainability.** Defensible under the redesign note, but it would invalidate the bundle this PR is migrating and is not required by issue #8's acceptance criteria.
- **Per-class capped sampling for operational metrics.** Destroys prevalence, which makes every metric in issue #4 meaningless.

## Risks

Bumping `bundle_format_version` to 2 invalidates every existing v1 bundle.
That is intended by #9, and the error names the version and points at retraining.

The window sidecar adds a cache with a lifecycle.
It is keyed by source mtime and size and rebuilt when stale, and a corrupt or unreadable sidecar is rebuilt rather than trusted.

The capture-day filename convention is assumed, not verified, because no Kaggle credentials are available here.
`capture_group` is written to handle both conventions and the failure message lists what it actually found, so the first real training run diagnoses it immediately instead of failing opaquely.
