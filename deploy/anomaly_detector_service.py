import os

import joblib
import torch
import pandas as pd
import numpy as np
import sys
from models import Autoencoder

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
    'fwd_header_len': 'Fwd Header Length', # This will map to the first 'Fwd Header Length'
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
    'cwr_flag_count': 'CWE Flag Count', # CWR Flag Count vs CWE Flag Count, often related/aliased. Verify.
    'ece_flag_cnt': 'ECE Flag Count',
    'down_up_ratio': 'Down/Up Ratio',
    'pkt_size_avg': 'Average Packet Size',
    'fwd_seg_size_avg': 'Avg Fwd Segment Size', # This replaces 'Fwd Header Length.1' from your old list
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
    'fwd_seg_size_min': 'min_seg_size_forward', # This is fwd_seg_size_min, not fwd_seg_size_avg
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

WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')

class AnomalyDetector:
    def __init__(self, model_path=None,
                 scaler_path=None,
                 threshold_path=None,
                 feature_columns=None):

        self.model_path = model_path if model_path else os.path.join(WORKING_DIR, "anomaly_detector_model.pth")
        self.scaler_path = scaler_path if scaler_path else os.path.join(WORKING_DIR, "min_max_scaler.pkl")
        self.threshold_path = threshold_path if threshold_path else os.path.join(WORKING_DIR, "optimal_threshold.npy")
        self.feature_columns = feature_columns if feature_columns is not None else EXPECTED_FEATURES_CICFLOWMETER

        self.scaler = None
        self.model = None
        self.threshold = None
        self.input_dim = None

        self._load_assets()

    def _load_assets(self):
        try:
            self.scaler = joblib.load(self.scaler_path)
            print(f"Loaded scaler from {self.scaler_path}")

            if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ is not None:
                self.input_dim = self.scaler.n_features_in_
            elif self.feature_columns is not None:
                 self.input_dim = len(self.feature_columns)
            else:
                raise ValueError("Cannot determine input dimension. Provide feature_columns or ensure scaler has n_features_in_.")

            self.model = Autoencoder(self.input_dim)
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            print(f"Loaded model from {self.model_path}")

            self.threshold = np.load(self.threshold_path).item()
            print(f"Loaded optimal threshold from {self.threshold_path}: {self.threshold:.4f}")

        except FileNotFoundError as e:
            print(f"Error loading asset: {e}. Make sure training has been run and 'working/' directory exists.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during asset loading: {e}")
            sys.exit(1)

    def predict(self, raw_data_df: pd.DataFrame):
        """
        Predicts anomalies for a given DataFrame of raw data.

        Args:
            raw_data_df (pd.DataFrame): DataFrame containing the raw features.
                                        It should include all necessary columns,
                                        even if some are non-features.

        Returns:
            tuple: (predictions, anomaly_scores)
                   predictions (np.ndarray): 0 for benign, 1 for anomaly.
                   anomaly_scores (np.ndarray): The reconstruction error (MSE) for each sample.
        """
        if raw_data_df.empty:
            return np.array([]), np.array([])

        processed_df = raw_data_df.copy()
        for col in self.feature_columns:
            if col not in processed_df.columns:
                processed_df[col] = np.nan

        features_df = processed_df[self.feature_columns]
        features_df = features_df.apply(pd.to_numeric, errors='coerce')
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.fillna(0, inplace=True)
        if features_df.empty:
            print("Warning: No valid numerical data left after all preprocessing steps.")
            return np.array([]), np.array([])

        # print("\n--- DEBUG: features_df after cleaning (before scaling) ---")
        # print("Columns:", features_df.columns.tolist())
        # print("Shape:", features_df.shape)
        # print("Data Types:\n", features_df.dtypes)
        # print("Head (first 5 rows):\n", features_df.head())
        # print("Describe (summary stats):\n", features_df.describe())
        # print("Check for NaNs (should be 0):\n", features_df.isnull().sum().sum()) # Total NaNs
        # print("Check for Infinities (should be 0):\n", features_df.isin([np.inf, -np.inf]).sum().sum()) # Total Infinities
        # print("--- END DEBUG: features_df ---")

        scaled_data = self.scaler.transform(features_df.values.astype(np.float32))
        scaled_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # print("\n--- DEBUG: scaled_tensor (before model prediction) ---")
        # print("Shape:", scaled_tensor.shape)
        # print("Sample (first 5 rows):\n", scaled_tensor[:5])
        # print(f"Min scaled value: {scaled_tensor.min().item()}")
        # print(f"Max scaled value: {scaled_tensor.max().item()}")
        # print(f"Mean scaled value: {scaled_tensor.mean().item()}")
        # print(f"Std scaled value: {scaled_tensor.std().item()}")
        # print("--- END DEBUG: scaled_tensor ---")
        with torch.no_grad():
            reconstructed_data = self.model(scaled_tensor)
            anomaly_scores = torch.mean((scaled_tensor - reconstructed_data) ** 2, dim=1).numpy()

        predictions = (anomaly_scores > self.threshold).astype(int)

        return predictions, anomaly_scores