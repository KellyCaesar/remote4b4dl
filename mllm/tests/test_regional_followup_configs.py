"""Validate the intended v2 structure comparisons without starting training."""
import unittest
from pathlib import Path

from scripts.preflight_regional_followup import config, verify_configs


class RegionalFollowupConfigTests(unittest.TestCase):
    def test_v2_experiment_differences_are_isolated(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        rows = [config(root / ("core_regional_" + name + "_v2.json")) for name in ("s2", "s3", "s4")]
        verify_configs(*rows)
        for index, key in ((1, "use_temporal"), (2, "centroid_policy")):
            changed = [dict(row) for row in rows]
            changed[index][key] = False if key == "use_temporal" else "strict_cell"
            with self.assertRaises(ValueError):
                verify_configs(*changed)


if __name__ == "__main__":
    unittest.main()
