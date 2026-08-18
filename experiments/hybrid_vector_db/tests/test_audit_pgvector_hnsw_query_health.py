from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.hybrid_vector_db.scripts import audit_pgvector_hnsw_query_health as health


class CohortTests(unittest.TestCase):
    def test_hash_bound_split_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / "cohort.csv"
            with cohort.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=(
                        "query_no", "query_id", "query_split",
                        "candidate_validity_predicate", "query_validity_predicate",
                    ),
                )
                writer.writeheader()
                writer.writerow({
                    "query_no": 100, "query_id": 7, "query_split": "final",
                    "candidate_validity_predicate": "embedding_valid",
                    "query_validity_predicate": "embedding_valid",
                })
            digest = hashlib.sha256(cohort.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "artifact_valid": True,
                "outputs": {"cohort_csv": {"path": str(cohort), "sha256": digest}},
            }), encoding="utf-8")
            rows, identity = health.load_cohort(cohort, manifest, "final", 1)
            self.assertEqual(rows[0]["query_id"], 7)
            self.assertEqual(identity["cohort_csv_sha256"], digest)

    def test_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / "cohort.csv"
            cohort.write_text(
                "query_no,query_id,query_split,candidate_validity_predicate,query_validity_predicate\n"
                "100,7,final,embedding_valid,embedding_valid\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "artifact_valid": True,
                "outputs": {"cohort_csv": {"path": str(cohort), "sha256": "0" * 64}},
            }), encoding="utf-8")
            with self.assertRaisesRegex(health.HealthAuditError, "SHA256"):
                health.load_cohort(cohort, manifest, "final", 1)

    def test_explicit_query_number_slice_is_exact_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / "cohort.csv"
            cohort.write_text(
                "query_no,query_id,query_split,candidate_validity_predicate,query_validity_predicate\n"
                "100,7,final,embedding_valid,embedding_valid\n"
                "101,8,final,embedding_valid,embedding_valid\n"
                "102,9,final,embedding_valid,embedding_valid\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "artifact_valid": True,
                "outputs": {"cohort_csv": {
                    "path": str(cohort),
                    "sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
                }},
            }), encoding="utf-8")

            rows, identity = health.load_cohort(
                cohort, manifest, "final", 2, 101, 103
            )

            self.assertEqual([row["query_no"] for row in rows], [101, 102])
            self.assertEqual(identity["query_no_start"], 101)
            self.assertEqual(identity["query_no_end_exclusive"], 103)


class SummaryTests(unittest.TestCase):
    def row(self, index: str, query_no: int, **overrides: object) -> dict[str, object]:
        value: dict[str, object] = {
            "index": index,
            "query_no": query_no,
            "latency_ms": 2.0,
            "returned": 10,
            "visited_tuples": 1200,
            "exhausted_terminations": 0,
            "plan_index_verified": True,
            "ids_unique": True,
            "self_excluded": True,
            "profile_valid": True,
            "profile_final_path": "stock",
            "error": "",
        }
        value.update(overrides)
        return value

    def test_complete_non_exhausted_indexes_pass(self) -> None:
        rows = [self.row("source", 1), self.row("clone", 1)]
        summary = health.summarize_rows(rows, ("source", "clone"), 10)
        self.assertTrue(summary["artifact_valid"])

    def test_exhausted_or_short_query_fails(self) -> None:
        rows = [
            self.row("source", 1, exhausted_terminations=1),
            self.row("clone", 1, returned=9),
        ]
        summary = health.summarize_rows(rows, ("source", "clone"), 10)
        self.assertFalse(summary["artifact_valid"])
        self.assertEqual(summary["indexes"]["source"]["exhausted_queries"], [1])
        self.assertEqual(summary["indexes"]["clone"]["incomplete_topk_queries"], [1])

    def test_plan_index_names_walks_explain_tree(self) -> None:
        plan = [{"Plan": {"Node Type": "Limit", "Plans": [
            {"Node Type": "Index Scan", "Index Name": "expected_idx"}
        ]}}]
        self.assertEqual(health.plan_index_names(plan), ["expected_idx"])


if __name__ == "__main__":
    unittest.main()
