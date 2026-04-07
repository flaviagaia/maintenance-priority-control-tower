from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.modeling import run_pipeline


class MaintenancePriorityControlTowerTestCase(unittest.TestCase):
    def test_pipeline_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_pipeline(temp_dir)
            self.assertEqual(summary["dataset_source"], "maintenance_control_tower_ai4i_style")
            self.assertEqual(summary["asset_count"], 10)
            self.assertGreaterEqual(summary["roc_auc"], 0.90)
            self.assertGreaterEqual(summary["average_precision"], 0.84)
            self.assertGreaterEqual(summary["f1"], 0.76)

            tower = pd.read_csv(Path(summary["control_tower_artifact"]))
            self.assertEqual(len(tower), 10)
            self.assertTrue(tower["priority_band"].isin(["P1", "P2", "P3", "P4"]).all())


if __name__ == "__main__":
    unittest.main()
