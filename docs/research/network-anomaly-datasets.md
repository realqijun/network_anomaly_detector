# Dataset strategy for an AI-assisted network-forensics anomaly detector

Research date: 2026-08-18

## Recommendation in one paragraph

Do not choose one benchmark and call the problem solved. Use **NF-CSE-CIC-IDS2018-v2 as the first deployable flow-model corpus**, because its 43-field NetFlow schema is much closer to telemetry that can be produced online than the original CICFlowMeter feature set. Use **NF-UQ-NIDS-v2 for broad pretraining and source-held-out generalization experiments**, not as a randomly shuffled benchmark. Add **CICIoT2023 PCAPs** for explicit XSS, SQL/command injection, brute force, and modern flood variants; **CIC-DDoS2019 PCAPs** for DDoS specialization; and **IoT-23 plus CIC-Bell-DNS-EXF-2021** for malware and exfiltration evidence. Keep CIC-IDS2017, UNSW-NB15, and whole capture scenarios out of training as external tests. Finally, build a small, versioned CTF evidence corpus because intrusion labels do not teach the system to recover flags, credentials, transferred files, DNS payloads, or reconstructed streams.

## What each dataset is good for

| Dataset | Raw evidence and prepared features | Relevant labels / scale | Access and constraints | Fit for this system |
|---|---|---|---|---|
| **CIC-IDS2017** | Full-payload PCAPs and labeled CICFlowMeter CSVs with 80+ flow features. Five days total; the owner lists about 51.1 GB by day. | FTP/SSH brute force, DoS, DDoS, web attacks, Heartbleed, bot, infiltration, scan. | Public research download; citation requested. Synthetic attacks and B-Profile-generated activity from only 25 modeled users in one 2017 testbed. Day, IP, and attack-time structure can become shortcuts. | Good small raw-PCAP baseline and external test; too old and testbed-specific to be the primary corpus. [UNB dataset page](https://www.unb.ca/cic/datasets/ids-2017.html) |
| **CSE-CIC-IDS2018** | Per-machine PCAPs and host logs plus CICFlowMeter-V3 CSVs. More than 80 bidirectional-flow statistics; configurable flow timeouts. Anonymous AWS S3 download is documented. | Brute force, Heartbleed, botnet, DoS, DDoS, web attacks, infiltration across 420 clients and 30 servers. Explicit events include FTP/SSH brute force, LOIC/HOIC, web/XSS brute force, and SQL injection. | The owner explicitly permits redistribution/mirroring with citation. Large AWS corpus and per-machine layout add storage/ETL friction. Labels derive from known IP/port/time schedules. | Best source PCAP collection for enterprise-style coverage, but regenerate a deployable schema rather than training directly on supplied CICFlowMeter CSVs. [UNB dataset page](https://www.unb.ca/cic/datasets/ids-2018.html) |
| **NF-CSE-CIC-IDS2018-v2** | CSV only; 43 extended NetFlow fields regenerated from the CSE-CIC-IDS2018 PCAPs. The original raw evidence remains available from UNB. | 18,893,708 flows: 16,635,567 benign; 1,390,270 DDoS; 483,999 DoS; 120,912 brute force; 143,097 bot; 116,361 infiltration; only 3,502 aggregated web-attack flows. | Direct University of Queensland research-data download. The standardized schema is easier to deploy, but the published corpus alone cannot support payload explanation or stream reconstruction. | **Primary phase-1 model corpus.** Excellent for binary ranking and coarse family prediction; weak supervision for XSS/SQL specifics. [UQ dataset collection](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) |
| **NF-UQ-NIDS-v2** | CSV only; the same 43-field schema, merging NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2, and NF-CSE-CIC-IDS2018-v2. Includes a source-dataset field. | 75,987,976 flows; 25,165,295 benign and 50,822,681 attacks. Includes DDoS, DoS, brute force, injection, XSS, reconnaissance, password, malware, and other families. The published counts include 2,455,020 XSS and 684,897 injection records. | Large and heavily imbalanced; harmonized parent labels erase attack-tool detail. Samples from one original capture are highly related. | Best broad pretraining pool, but **split by source dataset and capture/scenario**, never random rows. Keep the source identifier for grouping and auditing, never as a model input. [UQ dataset collection](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) |
| **CIC-DDoS2019** | Full PCAPs, host event logs, and CICFlowMeter-V3 CSVs with 80+ fields. The owner permits redistribution/mirroring with citation. | Two experiment days; NTP, DNS, LDAP, MSSQL, NetBIOS, SNMP, SSDP, UDP/UDP-Lag, SYN, TFTP, WebDDoS, and PortMap/PortScan. The owner notes WebDDoS traffic was very small. | Straightforward download, but it is a purpose-built testbed with attack windows and fixed endpoints; these fields can leak the label. | DDoS specialist fine-tuning and whole-day external evaluation, not a general anomaly corpus. [UNB dataset page](https://www.unb.ca/cic/datasets/ddos-2019.html) |
| **CICIoT2023** | Original PCAPs plus CSVs. Supplementary source describes Mergecap/TCPDump/PySpark and a DPKT extractor. CSV records are fixed packet windows, not conventional bidirectional flows. | 33 attacks over 105 real IoT devices: 11 DDoS variants, four DoS variants, dictionary brute force, XSS, SQL/command injection, upload/backdoor/browser-hijack attacks, recon, spoofing, and Mirai. The paper reports 46,686,579 extracted records. | The owner page provides a download and citation but does not state a reusable-data license; confirm intended redistribution before bundling. Huge class imbalance: the paper groups lower-volume attacks in 10-packet windows and floods in 100-packet windows. | Strongest named coverage for the requested XSS/web and modern flood families. Use PCAPs to regenerate the production schema; do not mix its supplied window rows directly with NetFlow rows. [UNB dataset page](https://www.unb.ca/cic/datasets/iotdataset-2023.html), [authors' dataset paper](https://www.mdpi.com/1424-8220/23/13/5941) |
| **UNSW-NB15** | About 100 GB of PCAPs plus Bro/Zeek, Argus, CSV, reports, and a 49-feature labeled schema. 2,540,044 CSV records; an official 175,341/82,332 train/test subset also exists. | Fuzzers, analysis, backdoors, DoS, exploits, generic attacks, reconnaissance, shellcode, and worms. It does **not** provide dedicated DDoS, brute-force, or XSS classes. | Academic use is granted in perpetuity; commercial use requires agreement with the authors. Traffic combines normal activity with IXIA-generated attacks. | Valuable source-held-out test for unseen families and toolchain diversity, not a primary corpus for the requested classes. [UNSW owner page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| **IoT-23** | Full 21 GB download contains PCAPs and analyst-labeled Zeek `conn.log` files; an 8.8 GB flow-only option exists. Twenty malware and three benign IoT capture scenarios. | Labels include benign, C&C, file download, DDoS, horizontal scan, and named malware behavior. | No permission request is needed if cited. Complete original PCAPs are supplied for IoT-23. | Especially useful for training/testing the path from anomalous flow to Zeek DNS/HTTP/TLS evidence and malware narrative. [Stratosphere owner page](https://www.stratosphereips.org/datasets-iot23), [owner FAQ](https://www.stratosphereips.org/datasets-faq) |
| **CIC-Bell-DNS-EXF-2021** | PCAPs and structured data from a published 30-feature DNS extractor; 270.8 MB of DNS traffic. | 323,698 heavy-attack, 53,978 light-attack, and 641,642 benign samples; six exfiltrated file types. | Redistribution/mirroring allowed with citation. Testbed uses DNSExfiltrator, one encoding configuration, and Alexa-derived benign domains, so it is a technique exemplar rather than broad reality. | High-value forensic specialist set: detect and explain DNS tunneling, then separately reconstruct/decode the carried data. [UNB dataset page](https://www.unb.ca/cic/datasets/dns-exf-2021.html) |

## Feature schema and live reproducibility

There is no lossless interchangeability among these prepared tables:

- **CICFlowMeter** creates bidirectional flows and over 80 statistics; direction is defined by the first packet, and UDP/TCP behavior depends on configured timeouts. UNB documents the schema and its Java implementation, but exact compatibility requires pinning the same implementation, timeout, feature names, and bug behavior. [CSE-CIC-IDS2018 feature documentation](https://www.unb.ca/cic/datasets/ids-2018.html), [CICFlowMeter source](https://github.com/ahlashkari/CICFlowMeter)
- **NetFlow V2** is the better model boundary. The authors designed the 43-field schema specifically to standardize datasets and make header-derived telemetry more feasible in live networks. Their original conversion used **nProbe/NetFlow v9**, not Zeek or CICFlowMeter. A Zeek `conn.log` mapping can cover core five-tuple, duration, bytes, packets, protocol, state, and service, but it will not automatically reproduce all 43 values. [standard feature-set paper](https://arxiv.org/abs/2101.11315), [conversion paper](https://arxiv.org/abs/2011.09144)
- **CICIoT2023 CSVs** use DPKT-derived fixed packet windows, including different window sizes for flood and non-flood attacks. They are unsuitable for a single model trained as though each row were a NetFlow record. Regenerate NetFlow/Zeek features from the PCAPs instead, while keeping the supplied rows as a separate specialist baseline. [authors' paper](https://www.mdpi.com/1424-8220/23/13/5941)

Recommended production contract:

1. Store the original PCAP and a stable packet/capture identifier whenever the input is offline CTF evidence.
2. Produce one canonical, versioned **bidirectional flow schema** from both PCAP and live interfaces. Start with fields common to NetFlow V2 and Zeek; add first-N packet sizes/directions/timing as an optional early-flow block.
3. Run Zeek in parallel for `conn`, `dns`, `http`, `ssl`/TLS, `files`, and notice logs. The anomaly model ranks flows; protocol analyzers supply human-readable evidence.
4. Pin extractor version, active/idle timeout, TCP teardown behavior, time units, missing-value rules, and categorical vocabulary in model metadata. Validate PCAP and live-interface output against golden captures before training.
5. Exclude source/destination IP, raw timestamps, capture/day IDs, labels, and dataset-of-origin from model inputs. Keep them only for grouping, joins, explanation, and leakage audits. The NetFlow authors likewise removed identifiers and TTL features that were highly correlated with labels. [authors' NetFlow paper](https://arxiv.org/abs/2011.09144)

## Staged training and evaluation portfolio

### Stage 1: fast flow triage

- Train binary anomaly ranking on a class-balanced, source-aware sample of **NF-CSE-CIC-IDS2018-v2**. Use all benign traffic for density/representation learning, but cap repeated flood flows per capture window so one flood does not dominate.
- Add a coarse multi-label/family head for DDoS, DoS, brute force, web/injection, bot/malware, infiltration/exfiltration, and recon. Do not promise exact exploit attribution from header-only features.
- Pretrain or stress-test with **NF-UQ-NIDS-v2**, preserving its original-dataset field only as a group key. This corpus is attack-heavy (about 67% attack), so calibrated probabilities must be re-estimated on a realistic, mostly benign validation stream.

### Stage 2: requested specialists

- DDoS: regenerate canonical features from **CIC-DDoS2019**, holding out an entire day and unseen attack tools/protocols.
- Brute force and web: regenerate from **CSE-CIC-IDS2018** and **CICIoT2023** PCAPs. Keep an entire experiment/day/device group out of training. Because NF-CSE-CIC-IDS2018-v2 has only 3,502 aggregated web flows, do not rely on it alone for XSS/SQL.
- Malware/exfiltration: use whole **IoT-23 scenarios** and whole **CIC-Bell-DNS-EXF-2021 days/file types** as grouped examples. Train protocol-specific evidence modules, not just another flow classifier.

### Stage 3: explicit split that resists leakage

Create a manifest assigning every row by `dataset -> capture/scenario -> time block -> flow`, then enforce:

- **Training:** selected CSE-CIC-IDS2018 capture days/scenarios plus selected NF-UQ parent sources; never allow rows from one capture/time window into multiple splits.
- **Validation:** later, whole CSE-CIC-IDS2018 days and whole CICIoT2023 device/attack runs not used for training; use this only for thresholds, calibration, and model selection.
- **In-domain test:** sealed whole days/scenarios from the same sources. Report per-family precision/recall, precision at top-K flows, alert volume per GB/hour, and time-to-first-relevant-evidence.
- **Cross-dataset test:** all of **CIC-IDS2017** and **UNSW-NB15**, plus a sealed CIC-DDoS2019 day and sealed IoT-23 malware scenarios. Do not tune after looking at these results.
- **Novel-family test:** exclude one attack family/tool entirely during training (for example DNS reflection, HOIC, XSS, or one malware family), then measure binary anomaly ranking separately from family naming.
- **CTF test:** hold out complete challenge PCAPs by challenge/event. Never split packets or flows from one challenge. Score whether the correct artifact is surfaced in top-K, whether the answer is correct, supporting packet/stream references, and elapsed analyst time.

Random row splits are invalid for the main claim: adjacent packets/windows, repeated five-tuples, fixed endpoints, scheduled attack periods, and duplicated flood behavior let a model memorize the lab rather than recognize an attack. CIC pages publish IPs and exact attack schedules, while CICIoT2023 aggregates neighboring packets into windows; these structures are useful for group manifests and dangerous as inputs.

## The CTF gap: anomaly detection is only the first half

The benchmark labels above answer “does this flow resemble a known attack?” They generally do **not** supervise the tasks that make network-forensics CTFs slow:

- locating a flag string in an HTTP body or reassembled TCP stream;
- extracting FTP/HTTP/SMB objects and archives;
- finding plaintext credentials, cookies, tokens, or commands;
- reconstructing DNS/ICMP covert-channel content;
- correlating ARP spoofing with later sessions;
- explaining a multi-step timeline with packet-level citations.

Implement this as two connected systems:

1. **Triage model:** scores and clusters packets/flows, estimates a broad family, and prioritizes sessions.
2. **Evidence engine:** runs deterministic Wireshark/tshark and Zeek-style reassembly, object extraction, protocol parsing, entropy/encoding checks, credential heuristics, and searchable strings; an LLM summarizes only retrieved evidence and cites capture/packet/stream IDs.

Build a separate CTF corpus from redistributable or locally generated challenge-like PCAPs. Each case should have structured answers: relevant packet and stream IDs, protocol, endpoints, artifact hashes, decoded payload, credentials/flag (stored securely), timeline, and acceptable explanation. Keep generated variants and their parent capture in one split. This evidence corpus—not higher benchmark classification accuracy—is what should be optimized against “solve network-forensics questions faster.”

## Practical acquisition order

1. Download **NF-CSE-CIC-IDS2018-v2** first and establish the canonical schema and grouped evaluation harness.
2. Fetch a limited set of **CSE-CIC-IDS2018 PCAP days** matching DDoS, brute force, and web attacks; prove identical offline/live extraction.
3. Add **CICIoT2023 PCAP subsets** for explicit XSS/injection and flood diversity.
4. Add **CIC-DDoS2019**, **IoT-23**, and **CIC-Bell-DNS-EXF-2021** specialists.
5. Use **NF-UQ-NIDS-v2** only after source-aware sampling and split manifests work; otherwise its 76 million rows mostly make leakage and imbalance faster.
6. Seal **CIC-IDS2017**, **UNSW-NB15**, and complete CTF challenges as untouched external tests.
