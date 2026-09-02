import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

from pcap_parser import rename_cli_columns_to_contract


class RenameCliColumnsToContractTests(unittest.TestCase):
    def test_renames_cicflowmeter_cli_columns_to_bundle_contract_names(self):
        cli_flows = pd.DataFrame(
            {
                "dst_port": [80],
                "flow_duration": [123.0],
                "tot_fwd_pkts": [4.0],
            }
        )

        renamed = rename_cli_columns_to_contract(cli_flows)

        self.assertEqual(renamed["Dst Port"].tolist(), [80])
        self.assertEqual(renamed["Flow Duration"].tolist(), [123.0])
        self.assertEqual(renamed["Tot Fwd Pkts"].tolist(), [4.0])

    def test_leaves_unrecognized_columns_untouched(self):
        cli_flows = pd.DataFrame({"some_unmapped_column": [1]})

        renamed = rename_cli_columns_to_contract(cli_flows)

        self.assertEqual(renamed.columns.tolist(), ["some_unmapped_column"])


if __name__ == "__main__":
    unittest.main()
