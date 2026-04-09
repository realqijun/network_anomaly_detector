import pandas as pd
import os
import sys
import subprocess
import shutil

CICFLOWMETER_CMD = "cicflowmeter"


MAPPING_CLI_TO_MODEL_FEATURES = {
    'dst_port': 'Destination Port',
    'flow_duration': 'Flow Duration',
    'tot_fwd_pkts': 'Total Fwd Packets',
    'tot_bwd_pkts': 'Total Backward Packets',
    'totlen_fwd_pkts': 'Total Length of Fwd Packets',
    'totlen_bwd_pkts': 'Total Length of Bwd Packets',
    'fwd_pkt_len_max': 'Fwd Packet Length Max',
    'fwd_pkt_len_min': 'Fwd Packet Length Min',
    'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
    'fwd_pkt_len_std': 'Fwd Packet Length Std',
    'bwd_pkt_len_max': 'Bwd Packet Length Max',
    'bwd_pkt_len_min': 'Bwd Packet Length Min',
    'bwd_pkt_len_mean': 'Bwd Packet Length Mean',
    'bwd_pkt_len_std': 'Bwd Packet Length Std',
    'flow_byts_s': 'Flow Bytes/s',
    'flow_pkts_s': 'Flow Packets/s',
    'flow_iat_mean': 'Flow IAT Mean',
    'flow_iat_std': 'Flow IAT Std',
    'flow_iat_max': 'Flow IAT Max',
    'flow_iat_min': 'Flow IAT Min',
    'fwd_iat_tot': 'Fwd IAT Total',
    'fwd_iat_mean': 'Fwd IAT Mean',
    'fwd_iat_std': 'Fwd IAT Std',
    'fwd_iat_max': 'Fwd IAT Max',
    'fwd_iat_min': 'Fwd IAT Min',
    'bwd_iat_tot': 'Bwd IAT Total',
    'bwd_iat_mean': 'Bwd IAT Mean',
    'bwd_iat_std': 'Bwd IAT Std',
    'bwd_iat_max': 'Bwd IAT Max',
    'bwd_iat_min': 'Bwd IAT Min',
    'fwd_psh_flags': 'Fwd PSH Flags',
    'bwd_psh_flags': 'Bwd PSH Flags',
    'fwd_urg_flags': 'Fwd URG Flags',
    'bwd_urg_flags': 'Bwd URG Flags',
    # This will map to the first 'Fwd Header Length'
    'fwd_header_len': 'Fwd Header Length',
    'bwd_header_len': 'Bwd Header Length',
    'fwd_pkts_s': 'Fwd Packets/s',
    'bwd_pkts_s': 'Bwd Packets/s',
    'pkt_len_min': 'Min Packet Length',
    'pkt_len_max': 'Max Packet Length',
    'pkt_len_mean': 'Packet Length Mean',
    'pkt_len_std': 'Packet Length Std',
    'pkt_len_var': 'Packet Length Variance',
    'fin_flag_cnt': 'FIN Flag Count',
    'syn_flag_cnt': 'SYN Flag Count',
    'rst_flag_cnt': 'RST Flag Count',
    'psh_flag_cnt': 'PSH Flag Count',
    'ack_flag_cnt': 'ACK Flag Count',
    'urg_flag_cnt': 'URG Flag Count',
    # CWR Flag Count vs CWE Flag Count, often related/aliased. Verify.
    'cwr_flag_count': 'CWE Flag Count',
    'ece_flag_cnt': 'ECE Flag Count',
    'down_up_ratio': 'Down/Up Ratio',
    'pkt_size_avg': 'Average Packet Size',
    # This replaces 'Fwd Header Length.1' from your old list
    'fwd_seg_size_avg': 'Avg Fwd Segment Size',
    'bwd_seg_size_avg': 'Avg Bwd Segment Size',
    'fwd_byts_b_avg': 'Fwd Avg Bytes/Bulk',
    'fwd_pkts_b_avg': 'Fwd Avg Packets/Bulk',
    'fwd_blk_rate_avg': 'Fwd Avg Bulk Rate',
    'bwd_byts_b_avg': 'Bwd Avg Bytes/Bulk',
    'bwd_pkts_b_avg': 'Bwd Avg Packets/Bulk',
    'bwd_blk_rate_avg': 'Bwd Avg Bulk Rate',
    'subflow_fwd_pkts': 'Subflow Fwd Packets',
    'subflow_bwd_pkts': 'Subflow Bwd Packets',
    'subflow_fwd_byts': 'Subflow Fwd Bytes',
    'subflow_bwd_byts': 'Subflow Bwd Bytes',
    'init_fwd_win_byts': 'Init_Win_bytes_forward',
    'init_bwd_win_byts': 'Init_Win_bytes_backward',
    'fwd_act_data_pkts': 'act_data_pkt_fwd',
    # This is fwd_seg_size_min, not fwd_seg_size_avg
    'fwd_seg_size_min': 'min_seg_size_forward',
    'active_max': 'Active Max',
    'active_min': 'Active Min',
    'active_mean': 'Active Mean',
    'active_std': 'Active Std',
    'idle_max': 'Idle Max',
    'idle_min': 'Idle Min',
    'idle_mean': 'Idle Mean',
    'idle_std': 'Idle Std'
}

EXPECTED_FEATURES_CICFLOWMETER = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
    'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std',
    'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]


def parse_pcap_to_dataframe(pcap_filename: str, expected_features: list) -> pd.DataFrame:
    host_base_dir = "/tmp/pcap_data"
    host_upload_dir = os.path.join(host_base_dir, "uploads")
    host_output_dir = os.path.join(host_base_dir, "outputs")

    output_csv_name = f"{pcap_filename}.csv"
    local_csv_path = os.path.join(host_output_dir, output_csv_name)

    command = [
        "docker", "run", "--rm",
        "-v", f"{host_upload_dir}:/data/input",
        "-v", f"{host_output_dir}:/data/output",
        "cfm", "cicflowmeter",
        "-f", f"/data/input/{pcap_filename}",
        "-c", f"/data/output/{output_csv_name}"
    ]

    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)

        if not os.path.exists(local_csv_path) or os.path.getsize(local_csv_path) == 0:
            return pd.DataFrame(columns=expected_features)

        df_flows = pd.read_csv(local_csv_path)
        df_flows.columns = df_flows.columns.str.strip()

        reverse_mapping = {cli_name: model_name for cli_name, model_name in MAPPING_CLI_TO_MODEL_FEATURES.items()}
        cols_to_rename = {col: reverse_mapping[col] for col in df_flows.columns if col in reverse_mapping}
        df_flows.rename(columns=cols_to_rename, inplace=True)

        if 'Fwd Header Length' in df_flows.columns and 'Fwd Header Length.1' in expected_features:
            df_flows['Fwd Header Length.1'] = df_flows['Fwd Header Length']

        for col in expected_features:
            if col not in df_flows.columns:
                df_flows[col] = 0

        return df_flows[expected_features]

    except Exception as e:
        print(f"Error during processing: {e}")
        return pd.DataFrame(columns=expected_features)
    finally:
        # Cleanup file, but DON'T return anything here
        if os.path.exists(local_csv_path):
            os.remove(local_csv_path)


if __name__ == '__main__':
    test_pcap = "dhcp.pcap"
    host_upload_dir = "/tmp/pcap_data/uploads"

    if not os.path.exists(test_pcap):
        print(f"Error: {test_pcap} not found in current directory.")
        sys.exit(1)

    # Move the test file to the shared volume
    # In a web app, this is where you'd save the uploaded file
    os.makedirs(host_upload_dir, exist_ok=True)
    shutil.copy(test_pcap, os.path.join(host_upload_dir, test_pcap))

    try:
        df_parsed = parse_pcap_to_dataframe(test_pcap, EXPECTED_FEATURES_CICFLOWMETER)

        print(f"\n--- Extraction Successful ---")
        print(f"Rows (Flows): {df_parsed.shape[0]}")
        print(f"Columns: {df_parsed.shape[1]}")
        print("\nTop 5 Rows:\n", df_parsed.head())

    except Exception as e:
        print(f"Testing failed: {e}")