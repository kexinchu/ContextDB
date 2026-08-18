from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.hybrid_vector_db.scripts import build_q5k_throughput_overrides as subject


class BuildQ5kThroughputOverridesTests(unittest.TestCase):
    def test_overlay_replaces_one_pair_and_emits_complete_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            overlay = root / "overlay.json"
            base.write_text(json.dumps({"amazon": {
                "f1": {
                    "stock": {"ef_search": 20},
                    "sqlens": {"ef_search": 11, "traversal_guided_target": 11},
                },
                "f2": {
                    "stock": {"ef_search": 30},
                    "sqlens": {"ef_search": 30},
                },
            }}), encoding="utf-8")
            overlay.write_text(json.dumps({"amazon": {
                "f1": {
                    "stock": {"ef_search": 40, "iterative_scan": "strict_order"},
                    "sqlens": {"ef_search": 12, "traversal_guided_target": 12},
                }
            }}), encoding="utf-8")
            result = subject.build(base, "amazon", [overlay])
            self.assertEqual(result["original"]["f1"]["ef_search"], 40)
            self.assertEqual(
                result[subject.SQLENS_MODE]["f1"]["traversal_guided_target"],
                12,
            )
            self.assertEqual(result["original"]["f2"]["ef_search"], 30)
            self.assertIn("max_scan_tuples", result[subject.SQLENS_MODE]["f2"])


if __name__ == "__main__":
    unittest.main()
