from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.hybrid_vector_db.scripts import (
    pgvector_d1_stock_increment_control as control,
)


class StockD1IncrementControlTests(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def config_rows(self) -> list[dict[str, object]]:
        return [
            {
                "filter_name": "filter_a",
                "mode": mode,
                "target_recall": 0.9,
                "ef_search": 100 if mode == "original" else 200,
                "max_scan_tuples": 5000000,
                "scan_mem_multiplier": 32,
                "iterative_scan": "off" if mode == "original" else "strict_order",
                "qualification": "lcb95",
                "calibration_recall_mean": 0.93,
                "calibration_recall_lcb95": 0.91,
            }
            for mode in control.MODES
        ]

    def configs(self) -> dict[str, control.SearchConfig]:
        return {
            "original": control.SearchConfig(
                "filter_a", "original", 0.9, 100, 5_000_000, 32.0, "off"
            ),
            "design1_bloom": control.SearchConfig(
                "filter_a", "design1_bloom", 0.9, 200, 5_000_000, 32.0, "strict_order"
            ),
        }

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            queries=1,
            repeats=1,
            k=10,
            table="public.items",
            source_index="public.items_hnsw",
            bfs_index="public.items_hnsw_bfs",
            query_table="public.queries",
            query_id_column="qid",
            query_vector_column="embedding",
            candidate_validity_predicate="embedding IS NOT NULL",
            expected_truth_self_excluded=False,
            backend_cpu=7,
            expected_sqlens_build_id="sqlens-v16-test",
            expected_vector_so_sha256="a" * 64,
            recall_tolerance=1e-12,
            config_csv=Path("config.csv"),
            config_manifest=Path("config.manifest.json"),
            filters_csv=Path("filters.csv"),
            truth_csv=Path("truth.csv"),
            truth_manifest=Path("truth.manifest.json"),
            python=Path("/usr/bin/python3"),
            query_offset=0,
            schedule_seed=17,
            d1_guidance_kind="auto",
            d1_exact_max_selectivity_pct=2.5,
            d1_cache_mb=1024,
            guidance_max_atoms=128,
            statement_timeout_ms=300000,
            progress_queries=0,
            bootstrap_samples=100,
            matched_target_recall=0.9,
        )

    def row(self, mode: str, recall: float = 0.9) -> dict[str, str]:
        d1 = mode == "design1_bloom"
        config = self.configs()[mode]
        return {
            "filter_name": "filter_a",
            "mode": mode,
            "error": "",
            "error_detail": "",
            "table": "public.items",
            "index": "public.items_hnsw",
            "query_table": "public.queries",
            "query_id_column": "qid",
            "query_vector_column": "embedding",
            "candidate_validity_predicate": "embedding IS NOT NULL",
            "planner_proof_verified": "True",
            "backend_cpu_exact_match": "True",
            "backend_cpu_requested": "7",
            "backend_cpu_observed": "7",
            "sqlens_build_id": "sqlens-v16-test",
            "vector_so_sha256": "a" * 64,
            "guidance_filter_strategy": "safe_guided",
            "guidance_enabled": str(d1),
            "guidance_scan_verified": "True",
            "guidance_binding_verified": "True",
            "final_path": "validation_only" if d1 else "stock",
            "warmup_all_queries": "True",
            "truth_self_excluded": "False",
            "ef_search": str(config.ef_search),
            "max_scan_tuples": str(config.max_scan_tuples),
            "scan_mem_multiplier": str(config.scan_mem_multiplier),
            "iterative_scan": config.iterative_scan,
            "query_no": "80",
            "query_id": "1000",
            "repeat": "0",
            "pair_key": "filter_a|q80|r0",
            "truth_filtered_rows": "100",
            "truth_kth_distance_sq": "1.0",
            "truth_tie_tolerance": "0.0",
            "recall": str(recall),
            "end_to_end_ms": "8" if d1 else "10",
            "query_latency_ms": "7" if d1 else "9",
            "activation_ms": "1",
        }

    def test_load_configs_requires_exact_two_arm_matched_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "configs.csv"
            rows = self.config_rows()
            self.write_csv(path, list(rows[0]), rows)
            configs, order = control.load_configs(path)
            self.assertEqual(order, ["filter_a"])
            self.assertEqual(configs["filter_a"]["design1_bloom"].ef_search, 200)

            self.write_csv(path, list(rows[0]), rows[:1])
            with self.assertRaisesRegex(control.ControlError, "exactly"):
                control.load_configs(path)

            rows[1]["target_recall"] = 0.95
            rows[1]["calibration_recall_lcb95"] = 0.96
            self.write_csv(path, list(rows[0]), rows)
            with self.assertRaisesRegex(control.ControlError, "matched recall target"):
                control.load_configs(path)

    def test_external_query_command_propagates_false_and_atom_limit(self):
        args = self.args()
        command = control.build_runner_command(
            args, Path("child.csv"), "filter_a", self.configs()
        )
        self.assertIn("--no-expected-truth-self-excluded", command)
        atom_position = command.index("--guidance-max-atoms")
        self.assertEqual(command[atom_position + 1], "128")
        query_position = command.index("--query-table")
        self.assertEqual(command[query_position + 1], "public.queries")
        config_position = command.index("--mode-configs-json")
        mode_configs = json.loads(command[config_position + 1])
        self.assertFalse(
            mode_configs["design1_bloom"]["traversal_guided_prioritization"]
        )

    def test_parse_args_infers_external_query_not_self_excluded_and_default_128(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "config.csv",
                "config.manifest.json",
                "filters.csv",
                "truth.csv",
                "truth.manifest.json",
                "python",
            ):
                (root / name).write_text("x\n", encoding="utf-8")
            args = control.parse_args(
                [
                    "--out",
                    str(root / "out.csv"),
                    "--config-csv",
                    str(root / "config.csv"),
                    "--config-manifest",
                    str(root / "config.manifest.json"),
                    "--filters-csv",
                    str(root / "filters.csv"),
                    "--truth-csv",
                    str(root / "truth.csv"),
                    "--truth-provenance-manifest",
                    str(root / "truth.manifest.json"),
                    "--table",
                    "items",
                    "--source-index",
                    "items_hnsw",
                    "--bfs-index",
                    "items_hnsw_bfs",
                    "--query-table",
                    "queries",
                    "--backend-cpu",
                    "7",
                    "--query-offset",
                    "80",
                    "--python",
                    str(root / "python"),
                    "--expected-sqlens-build-id",
                    "sqlens-v16-test",
                    "--expected-vector-so-sha256",
                    "a" * 64,
                ]
            )
            self.assertFalse(args.expected_truth_self_excluded)
            self.assertEqual(args.guidance_max_atoms, 128)
            self.assertEqual(args.table, "public.items")

    def test_validate_child_accepts_external_truth_and_rejects_recall_miss(self):
        args = self.args()
        rows = [self.row(mode) for mode in control.MODES]
        validated = control.validate_child(
            rows, args, "filter_a", self.configs(), {80}
        )
        self.assertEqual(len(validated), 2)
        self.assertEqual(validated[0]["configured_target_recall"], 0.9)

        rows[1]["recall"] = "0.8"
        with self.assertRaisesRegex(control.ControlError, "below target"):
            control.validate_child(rows, args, "filter_a", self.configs(), {80})

    def test_validate_child_rejects_silent_d1_atom_admission_bypass(self):
        args = self.args()
        rows = [self.row(mode) for mode in control.MODES]
        rows[1]["guidance_enabled"] = "False"
        with self.assertRaisesRegex(control.ControlError, "guidance activation"):
            control.validate_child(rows, args, "filter_a", self.configs(), {80})

    def test_truth_audit_covers_external_query_slice_and_every_filter_cell(self):
        args = self.args()
        args.query_offset = 1
        query_by_no = {10: 1000, 20: 2000, 30: 3000}
        truth = {
            (name, query_no): object()
            for name in ("filter_a", "filter_b")
            for query_no in query_by_no
        }
        fake_runner = SimpleNamespace(
            load_tie_aware_truth=mock.Mock(return_value=(truth, query_by_no)),
            effective_candidate_validity_predicate=(
                control.effective_candidate_validity_predicate
            ),
        )
        args.truth_csv = Path(__file__)
        with (
            mock.patch.object(control.d2, "load_runner", return_value=fake_runner),
            mock.patch.object(
                control.d2,
                "audit_exact_truth_manifest",
                return_value={"artifact_valid": True},
            ),
        ):
            audit, query_nos = control.audit_truth(
                args, ["filter_a", "filter_b"]
            )
        self.assertEqual(query_nos, {20})
        self.assertEqual(audit["query_ids"], [2000])
        self.assertFalse(audit["self_excluded"])
        fake_runner.load_tie_aware_truth.assert_called_once_with(
            args.truth_csv,
            expected_self_excluded=False,
            expected_candidate_validity_predicate="embedding IS NOT NULL",
        )

    def test_config_provenance_requires_audited_d1_to_match_csv(self):
        args = self.args()
        configs = {"filter_a": self.configs()}
        audited = control.d2.MatchedConfig(
            "filter_a", 0.9, 200, 5_000_000, 32.0, "strict_order", 1,
            "lcb95", 0.93, 0.91,
        )
        configs["filter_a"]["original"] = control.SearchConfig(
            "filter_a", "original", 0.9, 100, 5_000_000, 32.0, "off",
            "lcb95", 0.93, 0.91,
        )
        configs["filter_a"]["design1_bloom"] = control.SearchConfig(
            "filter_a", "design1_bloom", 0.9, 200, 5_000_000, 32.0,
            "strict_order", "lcb95", 0.93, 0.91,
        )
        with (
            mock.patch.object(
                control.d2, "audit_matched_configs_csv", return_value={"filter_a": audited}
            ),
            mock.patch.object(Path, "resolve", return_value=Path("/tmp/config.manifest.json")),
            mock.patch.object(control.d2, "sha256_file", return_value="a" * 64),
        ):
            evidence = control.audit_config_provenance(args, ["filter_a"], configs)
        self.assertEqual(evidence["qualification"], "one-sided bootstrap Recall@10 LCB95")

    def test_paired_summary_reports_latency_recall_and_speedup_ci(self):
        rows = [self.row(mode) for mode in control.MODES]
        summary = control.summarize(
            rows,
            ["filter_a"],
            {"filter_a": self.configs()},
            {"filter_a": 50.0},
            seed=3,
            bootstrap_samples=100,
        )[0]
        self.assertEqual(summary["stock_end_to_end_ms_mean"], 10.0)
        self.assertEqual(summary["d1_end_to_end_ms_mean"], 8.0)
        self.assertEqual(summary["d1_speedup_over_stock"], 1.25)
        self.assertEqual(summary["d1_speedup_ci95_low"], 1.25)
        self.assertEqual(summary["d1_minus_stock_latency_ci95_high_ms"], -2.0)
        self.assertTrue(summary["statistically_positive"])

    def test_plan_evidence_validates_index_build_cpu_and_warmup(self):
        args = self.args()
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp) / "child.csv"
            child.write_text("value\n1\n", encoding="utf-8")
            plan = child.with_suffix(child.suffix + ".plan.json")
            checks = []
            for mode in control.MODES:
                checks.append(
                    {
                        "mode": mode,
                        "filter_name": "filter_a",
                        "passed": True,
                        "expected_table_identity": "public.items",
                        "expected_index_identity": "public.items_hnsw",
                        "expected_index_oid": 42,
                        "catalog_index_oid": 42,
                        "preferred_index_current_setting": "public.items_hnsw",
                        "catalog_index_predicate_matches": True,
                        "candidate_validity_predicate": "embedding IS NOT NULL",
                        "backend_cpu_provenance": {
                            "requested_cpu_list": "7",
                            "observed_cpu_list": "7",
                            "exact_match": True,
                        },
                        "sqlens_runtime_identity": {
                            "expected_build_id": "sqlens-v16-test",
                            "expected_vector_so_sha256": "a" * 64,
                            "exact_match": True,
                        },
                    }
                )
            payload = {
                "status": "complete",
                "output": str(child),
                "output_rows": 2,
                "output_sha256": control.d2.sha256_file(child),
                "query_contract": {
                    "query_table": "public.queries",
                    "query_id_column": "qid",
                    "query_vector_column": "embedding",
                    "self_excluded": False,
                    "candidate_validity_predicate": "embedding IS NOT NULL",
                },
                "checks": checks,
                "execution_lifecycle": {
                    "warmup_complete": True,
                    "warmup_observed": 2,
                    "backend_cpu_provenance_complete": True,
                    "runtime_sqlens_identity_complete": True,
                },
            }
            plan.write_text(json.dumps(payload), encoding="utf-8")
            evidence = control.validate_plan_evidence(
                plan, child, args, "filter_a", {"oid": 42}
            )
            self.assertEqual(evidence["checks"], 2)

            payload["checks"][1]["expected_index_oid"] = 43
            plan.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(control.ControlError, "gate failed"):
                control.validate_plan_evidence(
                    plan, child, args, "filter_a", {"oid": 42}
                )


if __name__ == "__main__":
    unittest.main()
