import pandas as pd
import os
import shutil
import sys
import subprocess

CICFLOWMETER_CMD = "cicflowmeter"

try:
    from anomaly_detector_service import EXPECTED_FEATURES_CICFLOWMETER, MAPPING_CLI_TO_MODEL_FEATURES
except ImportError:
    print("Error: Could not import EXPECTED_FEATURES_CICFLOWMETER or MAPPING_CLI_TO_MODEL_FEATURES from anomaly_detector_service.py.")
    sys.exit(1)

def parse_pcap_to_dataframe(pcap_filepath: str, output_dir: str, expected_features: list) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    pcap_filename_base = os.path.splitext(os.path.basename(pcap_filepath))[0]
    output_csv_filename = f"{pcap_filename_base}_flows.csv"
    output_csv_path = os.path.join(output_dir, output_csv_filename)

    print(f"Parsing PCAP: {pcap_filepath} using cicflowmeter CLI...")
    print(f"Output CSV will be saved temporarily to: {output_csv_path}")

    command = [
        CICFLOWMETER_CMD,
        "-f", pcap_filepath,
        "-c",
        output_csv_path
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        print("cicflowmeter stdout:\n", result.stdout)
        if result.stderr:
            print("cicflowmeter stderr:\n", result.stderr)

        if not os.path.exists(output_csv_path) or os.path.getsize(output_csv_path) == 0:
            print("Warning: cicflowmeter CLI did not generate a CSV file or it was empty.")
            return pd.DataFrame(columns=expected_features)

        df_flows = pd.read_csv(output_csv_path)
        df_flows.columns = df_flows.columns.str.strip()


        reverse_mapping = {cli_name: model_name for cli_name, model_name in MAPPING_CLI_TO_MODEL_FEATURES.items()}

        cols_to_rename = {col: reverse_mapping[col] for col in df_flows.columns if col in reverse_mapping}

        df_flows.rename(columns=cols_to_rename, inplace=True)
        if 'Fwd Header Length' in df_flows.columns and 'Fwd Header Length.1' in expected_features:  # Adjust name if 'Fwd Header Length.1' etc.
            if 'Fwd Header Length.1' not in df_flows.columns:  # Only duplicate if the second one is truly missing
                df_flows['Fwd Header Length.1'] = df_flows['Fwd Header Length']
                print(
                    "Warning: Duplicated 'Fwd Header Length' to satisfy 'Fwd Header Length.1' feature requirement.")

        missing_cols_in_pcap_output = [col for col in expected_features if col not in df_flows.columns]
        if missing_cols_in_pcap_output:
            print(f"Warning: Missing expected features in cicflowmeter output after renaming: {missing_cols_in_pcap_output}. Filling with 0.")
            for col in missing_cols_in_pcap_output:
                df_flows[col] = 0
        df_final = df_flows[expected_features]

        return df_final

    except FileNotFoundError:
        print(f"Error: The command '{CICFLOWMETER_CMD}' was not found. ")
        return pd.DataFrame(columns=expected_features)
    except subprocess.CalledProcessError as e:
        print(f"Error running cicflowmeter CLI: Command '{' '.join(e.cmd)}' returned non-zero exit code {e.returncode}.")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
        return pd.DataFrame(columns=expected_features)
    except subprocess.TimeoutExpired:
        print(f"Error: cicflowmeter CLI timed out after {300} seconds for {pcap_filepath}.")
        return pd.DataFrame(columns=expected_features)
    except pd.errors.EmptyDataError:
        print(f"Warning: cicflowmeter produced an empty CSV file for {pcap_filepath}. No flows found or parsed.")
        return pd.DataFrame(columns=expected_features)
    except Exception as e:
        print(f"An unexpected error occurred during PCAP parsing: {e}")
        return pd.DataFrame(columns=expected_features)
    finally:
        if os.path.exists(output_csv_path):
            print(f"Cleaning up generated CSV: {output_csv_path}")
            os.remove(output_csv_path)

if __name__ == '__main__':
    test_pcap = "test.pcap"
    if not os.path.exists(test_pcap):
        print(f"Please create a dummy '{test_pcap}' file or point to an existing one for testing.")
        sys.exit(1)

    output_test_dir = "temp_pcap_output_test"
    os.makedirs(output_test_dir, exist_ok=True)

    df_parsed = parse_pcap_to_dataframe(test_pcap, output_test_dir, EXPECTED_FEATURES_CICFLOWMETER)
    print(f"\nParsed DataFrame shape: {df_parsed.shape}")
    print("First 5 rows of parsed DataFrame:\n", df_parsed.head())
    print(f"Columns: {df_parsed.columns.tolist()}")

    if os.path.exists(output_test_dir) and not os.listdir(output_test_dir):
        print(f"Removing empty test output directory: {output_test_dir}")
        shutil.rmtree(output_test_dir)