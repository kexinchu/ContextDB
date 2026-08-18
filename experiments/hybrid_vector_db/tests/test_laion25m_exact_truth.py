from __future__ import annotations

import argparse
import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import laion25m_exact_truth as truth  # noqa: E402


class LaionExactTruthTest(unittest.TestCase):
    def test_parse_labels_preserves_frequency_order_and_deduplicates(self) -> None:
        self.assertEqual(truth.parse_labels("175 197 20 175"), (175, 197, 20))
        self.assertEqual(truth.parse_labels("{175,197,20,197}"), (175, 197, 20))

    def test_formal_filter_atoms_follow_predicate_label_order(self) -> None:
        row = {
            "filter_name": "labelor_top3",
            "target_band_pct": 10.0,
            "actual_pct": 9.02,
            "filter_rows": 2_256_181,
            "predicate": "labels && ARRAY[175,197,20]::int[]",
            "labels_tuple": truth.parse_labels("175 197 20"),
        }

        [formal] = truth.formal_filter_rows([row])

        self.assertEqual(
            formal["atoms"].split("||OR||"),
            [
                "sql:labels @> ARRAY[175]::int[]",
                "sql:labels @> ARRAY[197]::int[]",
                "sql:labels @> ARRAY[20]::int[]",
            ],
        )

    def test_load_selected_keeps_csv_label_order(self) -> None:
        with TemporaryDirectory() as tmp:
            selected = Path(tmp) / "selected.csv"
            with selected.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "workload",
                        "filter_name",
                        "target_band_pct",
                        "actual_pct",
                        "filter_rows",
                        "qid",
                        "labels",
                        "range_l",
                        "range_r",
                        "predicate",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "workload": "label_or",
                        "filter_name": "labelor_top3",
                        "target_band_pct": "10.0",
                        "actual_pct": "9.02",
                        "filter_rows": "2256181",
                        "qid": "5329",
                        "labels": "175 197 20",
                        "range_l": "",
                        "range_r": "",
                        "predicate": "labels && ARRAY[175,197,20]::int[]",
                    }
                )
            args = argparse.Namespace(workloads=[], target_bands=[], limit_per_group=0)

            [loaded] = truth.load_selected(selected, args)

            self.assertEqual(loaded["labels_tuple"], (175, 197, 20))


if __name__ == "__main__":
    unittest.main()
