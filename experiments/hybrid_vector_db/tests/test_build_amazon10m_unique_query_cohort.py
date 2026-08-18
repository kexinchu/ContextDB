from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_amazon10m_unique_query_cohort as cohort  # noqa: E402


class UniqueQueryCohortTest(unittest.TestCase):
    def test_singleton_fingerprints_exclude_exact_duplicates(self) -> None:
        vectors = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        )
        first, second = cohort.projected_fingerprints(
            vectors, (0, 1), chunk_rows=2
        )
        self.assertEqual(
            cohort.singleton_fingerprint_ids(first, second).tolist(), [0, 2]
        )

    def test_projection_collision_can_only_reject_rows(self) -> None:
        vectors = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 30.0, 40.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=np.float32,
        )
        first, second = cohort.projected_fingerprints(
            vectors, (0,), chunk_rows=10
        )
        self.assertEqual(cohort.singleton_fingerprint_ids(first, second).tolist(), [2])

    def test_word_positions_are_deterministic_and_cover_width(self) -> None:
        self.assertEqual(cohort.fingerprint_word_positions(8, 4), (0, 1, 2, 3))
        self.assertEqual(cohort.fingerprint_word_positions(128, 4), (0, 16, 32, 48))
        with self.assertRaises(ValueError):
            cohort.fingerprint_word_positions(127, 4)

    def test_cohort_csv_uses_strict_dedicated_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cohort.csv"
            cohort.write_cohort_csv(
                path,
                np.asarray([19, 23], dtype=np.int64),
                calibration_queries=1,
                candidate_validity_predicate="embedding_valid",
            )
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                [
                    "query_no,query_id,query_split,candidate_validity_predicate,query_validity_predicate",
                    "0,19,calibration,embedding_valid,embedding_valid",
                    "1,23,final,embedding_valid,embedding_valid",
                ],
            )


if __name__ == "__main__":
    unittest.main()
