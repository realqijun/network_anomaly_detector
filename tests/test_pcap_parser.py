import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

from pcap_parser import parse_pcap_to_dataframe, rename_cli_columns_to_contract


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

    def test_parse_pcap_raises_clear_runtime_error_when_docker_is_unavailable(self):
        with patch("pcap_parser.shutil.copy"), patch(
            "pcap_parser.subprocess.run",
            side_effect=FileNotFoundError("docker"),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires a local Docker client"):
                parse_pcap_to_dataframe("capture.pcap", ["Flow Duration"])

    def test_parse_pcap_surfaces_cicflowmeter_failures(self):
        process_error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["docker", "run"],
            stderr="cfm image missing",
        )
        with patch("pcap_parser.shutil.copy"), patch(
            "pcap_parser.subprocess.run",
            side_effect=process_error,
        ):
            with self.assertRaisesRegex(RuntimeError, "cfm image missing"):
                parse_pcap_to_dataframe("capture.pcap", ["Flow Duration"])


if __name__ == "__main__":
    unittest.main()
