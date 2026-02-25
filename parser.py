import ipaddress
import os
import numpy as np
import pandas as pd


input_dir = "datasets"
output_dir = "working"
os.makedirs(output_dir, exist_ok=True)


def parse_ip(ip_str):
    """
    Converts an IP string (e.g., '192.168.1.1') to a 32-bit integer.
    Returns 0 if the IP is invalid.
    """
    try:
        if isinstance(ip_str, (int, float)):
            return int(ip_str)
        return int(ipaddress.IPv4Address(ip_str))
    except:
        return 0


def check_private_ip(ip_int):
    """
    Checks if a 32-bit integer IP is private.
    Returns 1 if private, 0 if public.
    """
    try:
        return 1 if ipaddress.IPv4Address(ip_int).is_private else 0
    except:
        return 0


def load_data(input_dir):
    """
    takes all the .csv files in datasets/ and prunes nas and combines them into 1 big dataframe
    :return: pandas dataframe
    """
    drop_cols = ['Flow ID']

    df_list = []

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            path = os.path.join(input_dir, file)

            df_temp = pd.read_csv(path, encoding='cp1252', low_memory=False)
            df_temp.columns = df_temp.columns.str.strip()

            if 'Timestamp' in df_temp.columns:
                df_temp['Timestamp'] = pd.to_datetime(
                    df_temp['Timestamp'], errors='coerce')

                df_temp.dropna(subset=['Timestamp'], inplace=True)

                df_temp['HourOfDay'] = df_temp['Timestamp'].dt.hour
                df_temp['DayOfWeek'] = df_temp['Timestamp'].dt.dayofweek

                df_temp['hour_sin'] = np.sin(
                    2 * np.pi * df_temp['HourOfDay']/23.0)
                df_temp['hour_cos'] = np.cos(
                    2 * np.pi * df_temp['HourOfDay']/23.0)

                df_temp.drop(columns=['Timestamp'], inplace=True)

            # NOTE IP AND PORTS ARE LARGE NUMBERS AND CAN DOMINATE SCALE, REMEMBER TO NORMALIZE
            if 'Source IP' in df_temp.columns and 'Destination IP' in df_temp.columns:
                df_temp['Source IP_Int'] = df_temp['Source IP'].astype(
                    str).apply(parse_ip)
                df_temp['Dest IP_Int'] = df_temp['Destination IP'].astype(
                    str).apply(parse_ip)

                df_temp['Source_Is_Private'] = df_temp['Source IP_Int'].apply(
                    check_private_ip)
                df_temp['Dest_Is_Private'] = df_temp['Dest IP_Int'].apply(
                    check_private_ip)

                # Identify "External-to-Internal" flows (common attack vector)
                df_temp['Inbound_External'] = ((df_temp['Source_Is_Private'] == 0) &
                                               (df_temp['Dest_Is_Private'] == 1)).astype(int)

                df_temp.drop(
                    columns=['Source IP', 'Destination IP'], inplace=True)

            if 'Source Port' in df_temp.columns and 'Destination Port' in df_temp.columns:
                df_temp['Is_Well_Known_Port'] = (
                    df_temp['Source Port'].isin(range(0, 1024)) |
                    df_temp['Destination Port'].isin(range(0, 1024))
                ).astype(int)

            df_temp['Label'] = df_temp['Label'].astype(str).str.strip()

            # Drop non-feature columns
            df_temp.drop(columns=drop_cols, errors='ignore', inplace=True)
            # Handle Infinity / NaN
            df_temp.replace(["Infinity", "Inf"], np.nan, inplace=True)

            # Convert numeric columns to float32
            numeric_cols = [col for col in df_temp.columns if col != 'Label']

            for col in numeric_cols:
                df_temp[col] = pd.to_numeric(
                    df_temp[col], errors='coerce').astype(np.float32)

            df_temp.dropna(inplace=True)
            df_list.append(df_temp)

    if not df_list:
        raise ValueError("No valid CSV files found in datasets directory.")
    df_final = pd.concat(df_list, ignore_index=True)

    # sanity check
    df_final = df_final[df_final['Label'] != 'Label']
    return df_final


if __name__ == "__main__":
    df = load_data(input_dir)
    output_path = os.path.join(output_dir, "processed_data.parquet")
    df.to_parquet(output_path, index=False)
