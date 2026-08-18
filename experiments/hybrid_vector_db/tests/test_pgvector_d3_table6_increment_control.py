from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from experiments.hybrid_vector_db.scripts import (
    pgvector_d2_cache_isolation_control as d2,
)
from experiments.hybrid_vector_db.scripts import (
    pgvector_d3_table6_increment_control as control,
)


class Table6D3IncrementControlTests(unittest.TestCase):
    def setUp(self):
        self.config = d2.MatchedConfig(
            filter_name="filter_a",
            target_recall=0.9,
            ef_search=250,
            max_scan_tuples=5_000_000,
            scan_mem_multiplier=32.0,
            iterative_scan="strict_order",
            guided_collect_target=1,
        )
        self.args = argparse.Namespace(
            queries=1,
            query_offset=100,
            repeats=1,
            bfs_index="public.bfs_idx",
            expected_sqlens_build_id="sqlens-v16-test",
            expected_vector_so_sha256="a" * 64,
            d1_guidance_kind="auto",
            guidance_max_atoms=128,
        )

    def row(self, mode: str) -> dict[str, str]:
        d3 = mode.endswith("_d3")
        return {
            "filter_name": "filter_a",
            "mode": mode,
            "error": "",
            "error_detail": "",
            "index": "public.bfs_idx",
            "planner_proof_verified": "True",
            "backend_cpu_exact_match": "True",
            "sqlens_build_id": "sqlens-v16-test",
            "vector_so_sha256": "a" * 64,
            "ef_search": "250",
            "max_scan_tuples": "5000000",
            "iterative_scan": "strict_order",
            "pair_key": "filter_a|q100|r0",
            "query_no": "100",
            "repeat": "0",
            "ids": "1,2,3",
            "result_distances": "0.1,0.2,0.3",
            "recall": "0.9",
            "selectivity": "1.0",
            "end_to_end_ms": "9" if d3 else "10",
            "query_latency_ms": "8" if d3 else "9",
            "activation_ms": "1",
            "returned_tuples": "10",
            "d3_initialization": "admitted_warm_reuse" if d3 else "",
            "d3_fragment_store_namespace": "namespace" if d3 else "",
            "d3_phase": "warm" if d3 else "",
            "d3_active_guidance_reused": "True" if d3 else "False",
            "d3_state_after": "exact" if d3 else "",
            "d3_fragment_builds_delta": "0",
            "d3_composed_guide_hits_delta": "1" if d3 else "0",
        }

    def test_namespace_is_bounded_and_filter_specific(self):
        first = control.namespace_for("abc123", "filter_a")
        second = control.namespace_for("abc123", "filter_b")
        self.assertNotEqual(first, second)
        self.assertLessEqual(len(first), 64)

    def test_validate_child_requires_identical_warm_reuse(self):
        rows = [
            self.row("design1_bloom_bfs_layout"),
            self.row("design1_bloom_bfs_layout_d3"),
        ]
        validated = control.validate_child(
            rows, self.args, "filter_a", "namespace", self.config
        )
        self.assertEqual(len(validated), 2)

        rows[1]["d3_phase"] = "admission"
        with self.assertRaisesRegex(control.ControlError, "not warm"):
            control.validate_child(
                rows, self.args, "filter_a", "namespace", self.config
            )

    def test_summary_uses_query_paired_delta(self):
        rows = [
            self.row("design1_bloom_bfs_layout"),
            self.row("design1_bloom_bfs_layout_d3"),
        ]
        summary = control.summarize(rows, ["filter_a"], 7)[0]
        self.assertEqual(summary["d123_speedup_over_d12"], 10 / 9)
        self.assertEqual(summary["d123_minus_d12_query_cluster_mean_ms"], -1.0)
        self.assertTrue(summary["statistically_positive"])

    def test_runner_command_uses_external_split_and_truth_contract(self):
        args = argparse.Namespace(
            python=Path("python"),
            filters_csv=Path("filters.csv"),
            truth_csv=Path("truth.csv"),
            table="public.external_items",
            source_index="public.external_source_idx",
            bfs_index="public.bfs_idx",
            candidate_validity_predicate="TRUE",
            schedule_seed=9,
            queries=7,
            query_offset=80,
            repeats=3,
            k=10,
            d1_guidance_kind="auto",
            guidance_max_atoms=128,
            d1_cache_mb=1024,
            d3_cache_mb=1024,
            d3_probe_requests=2,
            d3_min_benefit_per_byte=0.0,
            d3_max_fragment_mb=256,
            d3_page_min_skip_rate=0.8,
            d2_graph_proof_json=Path("proof.json"),
            statement_timeout_ms=300_000,
            progress_queries=0,
            expected_sqlens_build_id="sqlens-v16-test",
            expected_vector_so_sha256="a" * 64,
            backend_cpu=8,
            expected_truth_self_excluded=False,
            query_table="public.external_queries",
            query_id_column="qid",
            query_vector_column="embedding",
            matched_configs_csv=Path("matched-configs.csv"),
            matched_recall_manifest=Path("legacy.json"),
        )

        command = control.build_runner_command(
            args,
            Path("child.csv"),
            "filter_a",
            "namespace",
            self.config,
        )

        self.assertEqual(command[command.index("--query-offset") + 1], "80")
        self.assertEqual(command[command.index("--queries") + 1], "7")
        self.assertEqual(command[command.index("--repeats") + 1], "3")
        self.assertIn("--no-expected-truth-self-excluded", command)
        self.assertEqual(command[command.index("--guidance-max-atoms") + 1], "128")
        self.assertEqual(
            command[command.index("--query-table") + 1], "public.external_queries"
        )
        protocol = control.protocol(args)
        self.assertEqual(protocol["configuration_source"], "matched-configs.csv")
        self.assertEqual(protocol["measurement"]["guidance_max_atoms"], 128)
        self.assertFalse(protocol["measurement"]["truth_self_excluded"])

    def test_validate_child_rejects_rows_outside_cli_split(self):
        rows = [
            self.row("design1_bloom_bfs_layout"),
            self.row("design1_bloom_bfs_layout_d3"),
        ]
        rows[0]["query_no"] = "99"
        rows[1]["query_no"] = "99"

        with self.assertRaisesRegex(control.ControlError, "outside the CLI split"):
            control.validate_child(
                rows, self.args, "filter_a", "namespace", self.config
            )


if __name__ == "__main__":
    unittest.main()
