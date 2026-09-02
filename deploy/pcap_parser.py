import os
import shutil
import subprocess
import sys
from typing import Sequence

import pandas as pd

CICFLOWMETER_CMD = "cicflowmeter"

# CICFlowMeter's CLI output column names, mapped to the raw
# CSE-CIC-IDS2018 CSV header names training_pipeline.py infers a bundle's
# feature contract from (short forms like "Tot Fwd Pkts", "Dst Port").
# Derived from the published CSE-CIC-IDS2018 CSV schema; not yet verified
# against a live `cicflowmeter` CLI run or a real trained bundle (issue #3
# risky-seam note: run cicflowmeter against deploy/dhcp.pcap and
# deploy/ipv4frags.pcap and diff the output columns before trusting this
# mapping for a production PCAP upload).
CLI_TO_CONTRACT_FEATURE = {
    "dst_port": "Dst Port",
    "protocol": "Protocol",
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Tot Fwd Pkts",
    "tot_bwd_pkts": "Tot Bwd Pkts",
    "totlen_fwd_pkts": "TotLen Fwd Pkts",
    "totlen_bwd_pkts": "TotLen Bwd Pkts",
    "fwd_pkt_len_max": "Fwd Pkt Len Max",
    "fwd_pkt_len_min": "Fwd Pkt Len Min",
    "fwd_pkt_len_mean": "Fwd Pkt Len Mean",
    "fwd_pkt_len_std": "Fwd Pkt Len Std",
    "bwd_pkt_len_max": "Bwd Pkt Len Max",
    "bwd_pkt_len_min": "Bwd Pkt Len Min",
    "bwd_pkt_len_mean": "Bwd Pkt Len Mean",
    "bwd_pkt_len_std": "Bwd Pkt Len Std",
    "flow_byts_s": "Flow Byts/s",
    "flow_pkts_s": "Flow Pkts/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_tot": "Fwd IAT Tot",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Tot",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fwd_header_len": "Fwd Header Len",
    "bwd_header_len": "Bwd Header Len",
    "fwd_pkts_s": "Fwd Pkts/s",
    "bwd_pkts_s": "Bwd Pkts/s",
    "pkt_len_min": "Pkt Len Min",
    "pkt_len_max": "Pkt Len Max",
    "pkt_len_mean": "Pkt Len Mean",
    "pkt_len_std": "Pkt Len Std",
    "pkt_len_var": "Pkt Len Var",
    "fin_flag_cnt": "FIN Flag Cnt",
    "syn_flag_cnt": "SYN Flag Cnt",
    "rst_flag_cnt": "RST Flag Cnt",
    "psh_flag_cnt": "PSH Flag Cnt",
    "ack_flag_cnt": "ACK Flag Cnt",
    "urg_flag_cnt": "URG Flag Cnt",
    "cwr_flag_count": "CWE Flag Count",
    "ece_flag_cnt": "ECE Flag Cnt",
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Pkt Size Avg",
    "fwd_seg_size_avg": "Fwd Seg Size Avg",
    "bwd_seg_size_avg": "Bwd Seg Size Avg",
    "fwd_byts_b_avg": "Fwd Byts/b Avg",
    "fwd_pkts_b_avg": "Fwd Pkts/b Avg",
    "fwd_blk_rate_avg": "Fwd Blk Rate Avg",
    "bwd_byts_b_avg": "Bwd Byts/b Avg",
    "bwd_pkts_b_avg": "Bwd Pkts/b Avg",
    "bwd_blk_rate_avg": "Bwd Blk Rate Avg",
    "subflow_fwd_pkts": "Subflow Fwd Pkts",
    "subflow_bwd_pkts": "Subflow Bwd Pkts",
    "subflow_fwd_byts": "Subflow Fwd Byts",
    "subflow_bwd_byts": "Subflow Bwd Byts",
    "init_fwd_win_byts": "Init Fwd Win Byts",
    "init_bwd_win_byts": "Init Bwd Win Byts",
    "fwd_act_data_pkts": "Fwd Act Data Pkts",
    "fwd_seg_size_min": "Fwd Seg Size Min",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
}


def rename_cli_columns_to_contract(flows: pd.DataFrame) -> pd.DataFrame:
    """Rename CICFlowMeter CLI output columns to a bundle's contract names.

    Columns with no known mapping are left untouched. Missing or
    non-finite contract features are not filled in here; `Detector`
    rejects them at scoring time so PCAP uploads and CSV uploads share one
    missing-value policy.
    """

    return flows.rename(columns=CLI_TO_CONTRACT_FEATURE)


def parse_pcap_to_dataframe(pcap_path: str, expected_features: Sequence[str]) -> pd.DataFrame:
    """Extract flow features from a PCAP via a sibling `cfm` CICFlowMeter container.

    Requires a `cfm` Docker image providing the `cicflowmeter` CLI and
    Docker socket access from the caller; neither is provided by this
    repository (`deploy/cicflowmeter-docker/` is currently empty), so this
    path is not exercised by the test suite.
    """

    host_base_dir = "/tmp/pcap_data"
    host_upload_dir = os.path.join(host_base_dir, "uploads")
    host_output_dir = os.path.join(host_base_dir, "outputs")
    os.makedirs(host_upload_dir, exist_ok=True)
    os.makedirs(host_output_dir, exist_ok=True)

    pcap_filename = os.path.basename(pcap_path)
    host_pcap_path = os.path.join(host_upload_dir, pcap_filename)
    output_csv_name = f"{pcap_filename}.csv"
    local_csv_path = os.path.join(host_output_dir, output_csv_name)

    command = [
        "docker", "run", "--rm",
        "-v", f"{host_upload_dir}:/data/input",
        "-v", f"{host_output_dir}:/data/output",
        "cfm", CICFLOWMETER_CMD,
        "-f", f"/data/input/{pcap_filename}",
        "-c", f"/data/output/{output_csv_name}"
    ]

    try:
        shutil.copy(pcap_path, host_pcap_path)
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)

        if not os.path.exists(local_csv_path) or os.path.getsize(local_csv_path) == 0:
            return pd.DataFrame(columns=list(expected_features))

        df_flows = pd.read_csv(local_csv_path)
        df_flows.columns = df_flows.columns.str.strip()
        return rename_cli_columns_to_contract(df_flows)
    except FileNotFoundError as error:
        raise RuntimeError(
            "PCAP processing requires a local Docker client and the sibling "
            "'cfm' image; neither is available in this runtime"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("PCAP processing timed out while running CICFlowMeter") from error
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(f"PCAP processing failed while running CICFlowMeter: {details}") from error
    finally:
        for path in (host_pcap_path, local_csv_path):
            if os.path.exists(path):
                os.remove(path)


if __name__ == '__main__':
    test_pcap = "dhcp.pcap"

    if not os.path.exists(test_pcap):
        print(f"Error: {test_pcap} not found in current directory.")
        sys.exit(1)

    try:
        df_parsed = parse_pcap_to_dataframe(test_pcap, list(CLI_TO_CONTRACT_FEATURE.values()))

        print(f"\n--- Extraction Successful ---")
        print(f"Rows (Flows): {df_parsed.shape[0]}")
        print(f"Columns: {df_parsed.shape[1]}")
        print("\nTop 5 Rows:\n", df_parsed.head())

    except Exception as e:
        print(f"Testing failed: {e}")
