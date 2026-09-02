import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

from pcap_parser import parse_pcap_to_dataframe, probe_pcap_runtime, rename_cli_columns_to_contract


class RenameCliColumnsToContractTests(unittest.TestCase):
    def test_renames_cicflowmeter_cli_columns_to_bundle_contract_names(self):
        cli_flows = pd.DataFrame(
            {
                "dst_port": [80],
                "protocol": [6],
                "flow_duration": [123.0],
                "tot_fwd_pkts": [4.0],
            }
        )

        renamed = rename_cli_columns_to_contract(cli_flows)

        self.assertEqual(renamed["Dst Port"].tolist(), [80])
        self.assertEqual(renamed["Protocol"].tolist(), [6])
        self.assertEqual(renamed["Flow Duration"].tolist(), [123.0])
        self.assertEqual(renamed["Tot Fwd Pkts"].tolist(), [4.0])

    def test_leaves_unrecognized_columns_untouched(self):
        cli_flows = pd.DataFrame({"some_unmapped_column": [1]})

        renamed = rename_cli_columns_to_contract(cli_flows)

        self.assertEqual(renamed.columns.tolist(), ["some_unmapped_column"])

    def test_probe_pcap_runtime_reports_missing_docker_cli(self):
        with patch("pcap_parser.shutil.which", return_value=None):
            ready, reason = probe_pcap_runtime()

        self.assertFalse(ready)
        self.assertIn("Docker CLI", reason)

    def test_parse_pcap_raises_clear_runtime_error_when_docker_is_unavailable(self):
        with patch("pcap_parser.probe_pcap_runtime", return_value=(False, "Docker CLI is not installed")):
            with self.assertRaisesRegex(RuntimeError, "PCAP processing is unavailable"):
                parse_pcap_to_dataframe("capture.pcap", ["Flow Duration"])

    def test_parse_pcap_surfaces_cicflowmeter_failures(self):
        process_error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["docker", "run"],
            stderr="cfm image missing",
        )
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "capture.pcap"
            capture.write_bytes(b"pcap")
            with patch("pcap_parser.probe_pcap_runtime", return_value=(True, None)), patch(
                "pcap_parser.subprocess.run",
                side_effect=process_error,
            ):
                with self.assertRaisesRegex(RuntimeError, "cfm image missing"):
                    parse_pcap_to_dataframe(str(capture), ["Flow Duration"])

    def test_parse_pcap_uses_isolated_temp_directories_per_request(self):
        commands = []

        def fake_run(command, capture_output, text, check, timeout):
            commands.append(command)
            output_mount = next(part for part in command if part.endswith(":/data/output"))
            output_dir = output_mount.split(":", 1)[0]
            csv_path = os.path.join(output_dir, f"{Path(command[-3]).name}.csv")
            Path(csv_path).write_text("dst_port,protocol,flow_duration\n80,6,123.0\n")

        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "capture.pcap"
            capture.write_bytes(b"pcap")
            with patch("pcap_parser.probe_pcap_runtime", return_value=(True, None)), patch(
                "pcap_parser.subprocess.run",
                side_effect=fake_run,
            ):
                first = parse_pcap_to_dataframe(str(capture), ["Dst Port", "Protocol", "Flow Duration"])
                second = parse_pcap_to_dataframe(str(capture), ["Dst Port", "Protocol", "Flow Duration"])

        self.assertEqual(first["Dst Port"].tolist(), [80])
        self.assertEqual(second["Protocol"].tolist(), [6])
        first_input_mount = next(part for part in commands[0] if part.endswith(":/data/input"))
        second_input_mount = next(part for part in commands[1] if part.endswith(":/data/input"))
        self.assertNotEqual(first_input_mount, second_input_mount)


if __name__ == "__main__":
    unittest.main()
