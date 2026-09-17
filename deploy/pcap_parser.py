import os
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_contract import COLUMN_ALIASES, normalize_columns

CICFLOWMETER_CMD = "cicflowmeter"
DOCKER_IMAGE = "cfm"


def rename_cli_columns_to_contract(flows: pd.DataFrame) -> pd.DataFrame:
    """Rename CICFlowMeter CLI output columns to a bundle's contract names.

    The mapping itself lives in feature_contract.COLUMN_ALIASES, so a PCAP
    upload, a CSV upload and the trainer all agree on what a column is called.
    Keeping a private copy here is what let the identity fields go missing:
    the old table dropped src_ip, dst_ip, src_port and timestamp, so a PCAP
    upload could never have supported window features.

    Columns with no known mapping are left untouched. Missing or non-finite
    contract features are not filled in here; `Detector` rejects them at
    scoring time so every upload path shares one missing-value policy.
    """

    return normalize_columns(flows)


def probe_pcap_runtime() -> tuple[bool, str | None]:
    docker_path = shutil.which("docker")
    if docker_path is None:
        return False, "Docker CLI is not installed"

    try:
        subprocess.run(
            [docker_path, "info"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        details = getattr(error, "stderr", None) or getattr(error, "stdout", None) or str(error)
        return False, f"Docker daemon is unavailable: {details.strip()}"

    try:
        subprocess.run(
            [docker_path, "image", "inspect", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or str(error)).strip()
        return False, f"Required Docker image '{DOCKER_IMAGE}' is unavailable: {details}"
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return False, f"Could not verify required Docker image '{DOCKER_IMAGE}': {error}"

    return True, None


def parse_pcap_to_dataframe(pcap_path: str, expected_features: Sequence[str]) -> pd.DataFrame:
    """Extract flow features from a PCAP via a sibling `cfm` CICFlowMeter container.

    Requires a `cfm` Docker image providing the `cicflowmeter` CLI and
    Docker socket access from the caller; neither is provided by this
    repository (`deploy/cicflowmeter-docker/` is currently empty), so this
    path is not exercised by the test suite.
    """

    ready, reason = probe_pcap_runtime()
    if not ready:
        raise RuntimeError(f"PCAP processing is unavailable: {reason}")

    pcap_filename = os.path.basename(pcap_path)
    with tempfile.TemporaryDirectory(prefix="pcap-data-") as host_base_dir:
        host_upload_dir = os.path.join(host_base_dir, "uploads")
        host_output_dir = os.path.join(host_base_dir, "outputs")
        os.makedirs(host_upload_dir, exist_ok=True)
        os.makedirs(host_output_dir, exist_ok=True)

        host_pcap_path = os.path.join(host_upload_dir, pcap_filename)
        output_csv_name = f"{pcap_filename}.csv"
        local_csv_path = os.path.join(host_output_dir, output_csv_name)
        command = [
            "docker", "run", "--rm",
            "-v", f"{host_upload_dir}:/data/input",
            "-v", f"{host_output_dir}:/data/output",
            DOCKER_IMAGE,
            CICFLOWMETER_CMD,
            "-f", f"/data/input/{pcap_filename}",
            "-c", f"/data/output/{output_csv_name}",
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
                f"'{DOCKER_IMAGE}' image; neither is available in this runtime"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("PCAP processing timed out while running CICFlowMeter") from error
        except subprocess.CalledProcessError as error:
            details = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError(f"PCAP processing failed while running CICFlowMeter: {details}") from error


if __name__ == '__main__':
    test_pcap = "dhcp.pcap"

    if not os.path.exists(test_pcap):
        print(f"Error: {test_pcap} not found in current directory.")
        sys.exit(1)

    try:
        df_parsed = parse_pcap_to_dataframe(test_pcap, sorted(set(COLUMN_ALIASES.values())))

        print(f"\n--- Extraction Successful ---")
        print(f"Rows (Flows): {df_parsed.shape[0]}")
        print(f"Columns: {df_parsed.shape[1]}")
        print("\nTop 5 Rows:\n", df_parsed.head())

    except Exception as e:
        print(f"Testing failed: {e}")
