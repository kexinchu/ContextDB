from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from experiments.hybrid_vector_db.scripts import (
    pgvector_d2_cache_isolation_control as d2,
)
from experiments.hybrid_vector_db.scripts import (
    pgvector_four_arm_table7_control as control,
)


class FourArmTable7ControlTests(unittest.TestCase):
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
            query_offset=80,
            repeats=1,
            source_index="public.source_idx",
            bfs_index="public.bfs_idx",
            expected_sqlens_build_id="sqlens-v16-test",
            expected_vector_so_sha256="a" * 64,
            d1_guidance_kind="auto",
            matched_target_recall=0.9,
        )

    def row(self, mode: str, latency: float) -> dict[str, str]:
        d3 = mode.endswith("_d3")
        bfs = "bfs_layout" in mode
        return {
            "filter_name": "filter_a",
            "mode": mode,
            "error": "",
            "error_detail": "",
            "index": "public.bfs_idx" if bfs else "public.source_idx",
            "planner_proof_verified": "True",
            "backend_cpu_exact_match": "True",
            "sqlens_build_id": "sqlens-v16-test",
            "vector_so_sha256": "a" * 64,
            "ef_search": "250",
            "max_scan_tuples": "5000000",
            "scan_mem_multiplier": "32.0",
            "iterative_scan": "strict_order",
            "guided_collect_target": "1",
            "pair_key": "filter_a|q80|r0",
            "query_no": "80",
            "repeat": "0",
            "ids": "1,2,3",
            "result_distances": "[0.1,0.2,0.3]",
            "recall": "0.9",
            "returned": "3",
            "selectivity": "1.0",
            "end_to_end_ms": str(latency),
            "query_latency_ms": str(latency - 1),
            "activation_ms": "1",
            "d3_initialization": "admitted_warm_reuse" if d3 else "",
            "d3_fragment_store_namespace": "namespace" if d3 else "",
            "d3_phase": "warm" if d3 else "",
            "d3_active_guidance_reused": "True" if d3 else "False",
            "d3_state_after": "exact" if d3 else "",
            "d3_fragment_builds_delta": "0",
            "d3_composed_guide_hits_delta": "1" if d3 else "0",
        }

    def rows(self) -> list[dict[str, str]]:
        return [
            self.row("original", 16),
            self.row("design1_bloom", 12),
            self.row("design1_bloom_bfs_layout", 10),
            self.row("design1_bloom_bfs_layout_d3", 8),
        ]

    def test_validate_child_requires_one_identical_four_arm_pair(self):
        validated = control.validate_child(
            self.rows(), self.args, "filter_a", "namespace", self.config
        )
        self.assertEqual(len(validated), 4)

        broken = self.rows()
        broken[-1]["ids"] = "9"
        with self.assertRaisesRegex(control.ControlError, "changed ids"):
            control.validate_child(
                broken, self.args, "filter_a", "namespace", self.config
            )

    def test_summary_reports_all_absolute_and_incremental_speedups(self):
        summary = control.summarize(
            self.rows(), ["filter_a"], seed=7, target_recall=0.9
        )[0]
        self.assertEqual(summary["d1_speedup_over_stock"], 16 / 12)
        self.assertEqual(summary["d12_speedup_over_stock"], 16 / 10)
        self.assertEqual(summary["d123_speedup_over_stock"], 2.0)
        self.assertEqual(summary["d2_increment_speedup"], 12 / 10)
        self.assertEqual(summary["d3_increment_speedup"], 10 / 8)
        self.assertTrue(summary["d1_statistically_positive"])
        self.assertTrue(summary["d2_statistically_positive"])
        self.assertTrue(summary["d3_statistically_positive"])

    def test_summary_rejects_held_out_recall_below_target(self):
        rows = self.rows()
        for row in rows:
            row["recall"] = "0.89"
        with self.assertRaisesRegex(control.ControlError, "below 0.90"):
            control.summarize(
                rows, ["filter_a"], seed=7, target_recall=0.9
            )

    def test_runner_command_uses_all_modes_and_one_shared_config(self):
        args = argparse.Namespace(
            python=Path("python"),
            filters_csv=Path("filters.csv"),
            truth_csv=Path("truth.csv"),
            table="public.items",
            source_index="public.source_idx",
            bfs_index="public.bfs_idx",
            candidate_validity_predicate="TRUE",
            schedule_seed=9,
            queries=7,
            query_offset=80,
            repeats=3,
            k=10,
            d1_guidance_kind="auto",
            guidance_max_atoms=256,
            d1_cache_mb=1024,
            d3_cache_mb=1024,
            d3_probe_requests=2,
            d3_min_benefit_per_byte=0.0,
            d3_max_fragment_mb=256,
            d3_page_min_skip_rate=1.0,
            d2_graph_proof_json=Path("proof.json"),
            statement_timeout_ms=600_000,
            progress_queries=25,
            expected_sqlens_build_id="sqlens-v16-test",
            expected_vector_so_sha256="a" * 64,
            backend_cpu=56,
            expected_truth_self_excluded=False,
            query_table="public.queries",
            query_id_column="qid",
            query_vector_column="embedding",
        )
        command = control.build_runner_command(
            args,
            Path("child.csv"),
            "filter_a",
            "namespace",
            self.config,
        )
        modes_start = command.index("--modes") + 1
        modes_end = command.index("--execution-order")
        self.assertEqual(tuple(command[modes_start:modes_end]), control.MODES)
        self.assertEqual(command[command.index("--query-offset") + 1], "80")
        self.assertEqual(command[command.index("--queries") + 1], "7")
        self.assertIn("--warmup-all-queries", command)
        self.assertIn("--no-expected-truth-self-excluded", command)
        self.assertEqual(
            command[command.index("--d3-page-min-skip-rate") + 1], "1.0"
        )

        config_json = command[command.index("--mode-configs-json") + 1]
        self.assertEqual(config_json.count('"ef_search": 250'), 4)

    def test_namespace_and_attempt_paths_are_stable(self):
        namespace = control.namespace_for("abc123", "filter_a")
        self.assertLessEqual(len(namespace), 64)
        self.assertEqual(
            control.child_path(Path("out.csv"), 2, "filter_a", 3).name,
            "out.f02.filter_a.a003.csv",
        )


if __name__ == "__main__":
    unittest.main()
