from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.hybrid_vector_db.scripts import (
    run_pgvector_three_arm_matched_recall as controller,
)


def graph_payload(m: int = 32) -> dict[str, object]:
    source = "public.amazon10m_embedding_valid_hnsw_source_idx"
    clone = "public.amazon10m_embedding_valid_hnsw_bfs_clone_idx"

    def index(role: str, layout: str) -> dict[str, object]:
        contract: dict[str, object] = {
            "role": role,
            "table": "public.amazon_grocery_reviews_10m_pgvector",
            "predicate": "embedding_valid",
            "opclass": "vector_l2_ops",
            "m": m,
            "build_page_order": layout,
            "require_full_memory_build": role == "clone",
        }
        if role == "clone":
            contract["clone_source"] = source
        return {
            "definition_diff": {},
            "build_contract": contract,
            "state": {
                "predicate": "embedding_valid",
                "reloptions": [f"m={m}", "ef_construction=200"],
            },
        }

    return {
        "artifact_valid": True,
        "source_index": source,
        "clone_index": clone,
        "stable_fingerprint_sha256": "a" * 64,
        "comparison": {
            "format": "sqlens-hnsw-compare-v2",
            "same_heap": True,
            "entry_equal": True,
            "logical_equal": True,
            "definition_equal": True,
            "tuple_coverage_equal": True,
            "physical_equal": False,
            "left_definition_digest": "definition",
            "right_definition_digest": "definition",
            "left_tuple_coverage_digest": "coverage",
            "right_tuple_coverage_digest": "coverage",
            "left_logical_digest": "logical",
            "right_logical_digest": "logical",
            "left_physical_digest": "physical-source",
            "right_physical_digest": "physical-clone",
        },
        "preparation": {
            "indexes": {
                "source": index("source", "insertion"),
                "clone": index("clone", "bfs"),
            }
        },
    }


def calibration_rows(
    arm: str = "official",
    filter_name: str = "f0",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    configs = {
        "slow_high_recall": (12.0, 1.0),
        "fast_qualified": (7.0, 0.95),
        "fast_unqualified": (4.0, 0.89),
    }
    for label, (latency, recall) in configs.items():
        for query_no in controller.CALIBRATION_QUERY_NOS:
            for repeat in range(controller.CALIBRATION_REPEATS):
                rows.append(
                    {
                        "arm": arm,
                        "filter_name": filter_name,
                        "config_label": label,
                        "query_no": query_no,
                        "repeat": repeat,
                        "latency_ms": latency,
                        "recall_at_10": recall,
                        "valid": True,
                        "error": "",
                    }
                )
    return rows


def selection_rows(filter_name: str = "f0") -> list[dict[str, object]]:
    return [
        {
            "arm": arm,
            "filter_name": filter_name,
            "target_recall": target,
            "config_label": f"{arm}_{target:g}",
        }
        for arm in controller.ARMS
        for target in controller.TARGET_RECALLS
    ]


def final_rows(filter_name: str = "f0") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selections = {
        (str(row["arm"]), float(row["target_recall"])): str(row["config_label"])
        for row in selection_rows(filter_name)
    }
    latency = {"official": 10.0, "sqlens_disabled": 12.0, "sqlens_full": 5.0}
    for target in controller.TARGET_RECALLS:
        for query_no in controller.FINAL_QUERY_NOS:
            for arm in controller.ARMS:
                for repeat in range(controller.FINAL_REPEATS):
                    rows.append(
                        {
                            "arm": arm,
                            "filter_name": filter_name,
                            "target_recall": target,
                            "query_no": query_no,
                            "query_id": 1000 + query_no,
                            "repeat": repeat,
                            "config_label": selections[(arm, target)],
                            "latency_ms": latency[arm] + (query_no - 100) / 100.0,
                            "recall_at_10": target,
                            "valid": True,
                            "error": "",
                        }
                    )
    return rows


class ThreeArmMatchedRecallTests(unittest.TestCase):
    def test_formal_protocol_fixes_splits_repeats_targets_and_arms(self) -> None:
        protocol = controller.formal_protocol()

        self.assertEqual(protocol["arms"], list(controller.ARMS))
        self.assertEqual(protocol["hnsw"]["m"], 32)
        self.assertEqual(
            protocol["ground_truth"]["candidate_validity_predicate"],
            "embedding_valid",
        )
        self.assertEqual(protocol["query_splits"]["screen"], list(range(20)))
        self.assertEqual(protocol["query_splits"]["calibration"], list(range(20, 100)))
        self.assertEqual(protocol["query_splits"]["final"], list(range(100, 200)))
        self.assertEqual(protocol["repeats"], {"screen": 1, "calibration": 2, "final": 6})
        self.assertEqual(protocol["target_recalls"], [0.90, 0.95, 0.99])
        self.assertEqual(
            protocol["calibration_selection_policy"], "lcb_then_max_recall"
        )
        self.assertEqual(
            protocol["bootstrap"]["calibration_unit"],
            "query cluster after averaging two repeats",
        )

    def test_default_filters_match_embedding_valid_candidate_universe(self) -> None:
        args = controller.build_parser().parse_args(["--dry-run"])

        self.assertEqual(
            args.filters_csv,
            controller.ROOT
            / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_valid_embeddings_filters.csv",
        )

    def test_dry_run_does_not_touch_files_docker_or_database(self) -> None:
        args = controller.build_parser().parse_args(["--dry-run", "--schedule-seed", "17"])
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("file read")), mock.patch.object(
            controller.subprocess, "run", side_effect=AssertionError("process run")
        ):
            payload = controller.dry_run_payload(args)

        self.assertFalse(payload["file_access"])
        self.assertFalse(payload["docker_access"])
        self.assertFalse(payload["database_access"])
        self.assertFalse(payload["experiment_started"])

    def test_final_schedule_is_seeded_deterministic_and_rotating(self) -> None:
        schedule, audit = controller.rotating_final_schedule("run-123", 19)

        self.assertEqual(len(schedule), 18)
        self.assertEqual(audit["arm_counts"], {arm: 6 for arm in controller.ARMS})
        self.assertNotEqual(audit["block_orders"][0], audit["block_orders"][1])
        self.assertTrue(audit["seeded_rotation_verified"])
        for block in range(6):
            block_rows = [row for row in schedule if row["final_block"] == block]
            self.assertEqual({row["arm"] for row in block_rows}, set(controller.ARMS))
            self.assertEqual({row["repeats"] for row in block_rows}, {1})
        for arm in controller.ARMS:
            self.assertEqual(sorted(audit["positions_by_arm"][arm]), [0, 0, 1, 1, 2, 2])
        self.assertEqual(
            controller.rotating_final_schedule("run-123", 19),
            (schedule, audit),
        )

    def test_canonical_final_arm_requires_exact_query_repeat_block_coverage(self) -> None:
        schedule, _ = controller.rotating_final_schedule("run-123", 19)
        selections = selection_rows()
        positions = {
            (int(row["final_block"]), str(row["arm"])): int(row["position"])
            for row in schedule
        }
        selected = {
            (str(row["arm"]), float(row["target_recall"])): str(row["config_label"])
            for row in selections
        }
        rows: list[dict[str, object]] = []
        for target in controller.TARGET_RECALLS:
            for query_no in controller.FINAL_QUERY_NOS:
                for repeat in range(controller.FINAL_REPEATS):
                    block = repeat // controller.FINAL_REPEATS_PER_BLOCK
                    rows.append(
                        {
                            "arm": "official",
                            "filter_name": "f0",
                            "target_recall": target,
                            "query_no": query_no,
                            "query_id": 1000 + query_no,
                            "repeat": repeat,
                            "final_block": block,
                            "arm_order_position": positions[(block, "official")],
                            "config_label": selected[("official", target)],
                            "valid": True,
                            "error": "",
                        }
                    )

        controller.validate_canonical_final_arm_rows(
            rows, "official", selections, schedule
        )
        with self.assertRaisesRegex(controller.CheckpointError, "coverage is not exact"):
            controller.validate_canonical_final_arm_rows(
                [*rows, dict(rows[0])], "official", selections, schedule
            )

    def test_m32_graph_gate_accepts_canonical_source_bfs_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "graph.json"
            path.write_text(json.dumps(graph_payload()), encoding="utf-8")

            proof = controller.validate_m32_same_graph_proof(
                path,
                "public.amazon10m_embedding_valid_hnsw_source_idx",
                "public.amazon10m_embedding_valid_hnsw_bfs_clone_idx",
            )

        self.assertTrue(proof["formal_gate_passed"])
        self.assertEqual(proof["hnsw_m"], 32)
        self.assertFalse(proof["physical_equal"])

    def test_m32_graph_gate_rejects_other_m(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "graph.json"
            path.write_text(json.dumps(graph_payload(m=16)), encoding="utf-8")

            with self.assertRaisesRegex(controller.ProtocolError, "expected 32"):
                controller.validate_m32_same_graph_proof(
                    path,
                    "public.amazon10m_embedding_valid_hnsw_source_idx",
                    "public.amazon10m_embedding_valid_hnsw_bfs_clone_idx",
                )

    def test_selection_uses_lowest_latency_among_lcb_confirmed_qualifiers(self) -> None:
        summaries = [
            {
                "config_label": "high-recall",
                "complete": True,
                "errors": 0,
                "recall_mean": 1.0,
                "recall_lcb95": 1.0,
                "latency_mean_ms": 12.0,
            },
            {
                "config_label": "fast-qualified",
                "complete": True,
                "errors": 0,
                "recall_mean": 0.95,
                "recall_lcb95": 0.80,
                "latency_mean_ms": 7.0,
            },
            {
                "config_label": "fast-unqualified",
                "complete": True,
                "errors": 0,
                "recall_mean": 0.94,
                "recall_lcb95": 0.94,
                "latency_mean_ms": 4.0,
            },
        ]

        selected = controller.select_fastest_qualifying_config(summaries, 0.95)

        self.assertIsNotNone(selected)
        self.assertEqual(selected["config_label"], "high-recall")
        self.assertEqual(selected["selection_fallback"], "none")

    def test_selection_falls_back_to_max_mean_recall_then_latency(self) -> None:
        summaries = [
            {
                "config_label": "lower-fast",
                "complete": True,
                "errors": 0,
                "recall_mean": 0.96,
                "recall_lcb95": 0.90,
                "latency_mean_ms": 2.0,
            },
            {
                "config_label": "best-slow",
                "complete": True,
                "errors": 0,
                "recall_mean": 0.98,
                "recall_lcb95": 0.94,
                "latency_mean_ms": 8.0,
            },
            {
                "config_label": "best-fast",
                "complete": True,
                "errors": 0,
                "recall_mean": 0.98,
                "recall_lcb95": 0.93,
                "latency_mean_ms": 5.0,
            },
        ]

        selected = controller.select_fastest_qualifying_config(summaries, 0.95)

        self.assertIsNotNone(selected)
        self.assertEqual(selected["config_label"], "best-fast")
        self.assertEqual(selected["selection_fallback"], "max_mean_recall")
        self.assertEqual(selected["lcb95_qualified_configs"], 0)

    def test_calibration_summary_requires_exact_q20_99_r2_coverage(self) -> None:
        rows = calibration_rows()[: 80 * 2]
        summary = controller.summarize_config_measurements(
            rows,
            expected_query_nos=controller.CALIBRATION_QUERY_NOS,
            expected_repeats=2,
        )
        self.assertEqual(summary["queries"], 80)
        self.assertEqual(summary["samples"], 160)
        self.assertEqual(summary["recall_lcb95"], summary["recall_mean"])
        self.assertEqual(
            summary["recall_bootstrap_unit"],
            "query_cluster_after_repeat_mean",
        )
        self.assertEqual(
            summary,
            controller.summarize_config_measurements(
                rows,
                expected_query_nos=controller.CALIBRATION_QUERY_NOS,
                expected_repeats=2,
            ),
        )

        varied = [
            {
                **row,
                "recall_at_10": 1.0 if int(row["query_no"]) < 90 else 0.5,
            }
            for row in rows
        ]
        canonical = controller.summarize_config_measurements(
            varied,
            expected_query_nos=controller.CALIBRATION_QUERY_NOS,
            expected_repeats=2,
            bootstrap_samples=500,
            bootstrap_seed=37,
        )
        upstream = controller.upstream_runner.summarize_rows(
            varied,
            expected_queries=80,
            expected_repeats=2,
            bootstrap_samples=500,
            bootstrap_seed=37,
        )
        self.assertEqual(canonical["recall_lcb95"], upstream["recall_lcb95"])

        duplicate = [*rows, dict(rows[0])]
        with self.assertRaisesRegex(controller.ProtocolError, "duplicate"):
            controller.summarize_config_measurements(
                duplicate,
                expected_query_nos=controller.CALIBRATION_QUERY_NOS,
                expected_repeats=2,
            )

    def test_calibrated_selection_is_recomputed_from_raw_rows(self) -> None:
        selected = controller.select_calibrated_configs(
            calibration_rows(),
            ["f0"],
            arms=("official",),
            targets=(0.90, 0.95),
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["config_label"], "fast_qualified")
        self.assertEqual(selected[1]["config_label"], "fast_qualified")
        self.assertEqual(
            selected[0]["calibration_selection_policy"], "lcb_then_max_recall"
        )
        self.assertEqual(selected[0]["selection_fallback"], "none")
        self.assertEqual(
            selected[0]["calibration_recall_lcb95"],
            selected[0]["recall_lcb95"],
        )

    def test_query_pairing_requires_all_three_arms_and_six_repeats(self) -> None:
        selections = selection_rows()
        paired = controller.build_query_level_pairs(final_rows(), selections, ["f0"])

        self.assertEqual(len(paired), 3 * 100)
        self.assertEqual(paired[0]["official_repeats"], 6)
        self.assertEqual(paired[0]["sqlens_disabled_repeats"], 6)
        self.assertEqual(paired[0]["sqlens_full_repeats"], 6)
        self.assertEqual(paired[0]["query_id"], 1100)

        incomplete = final_rows()
        incomplete.pop()
        with self.assertRaisesRegex(controller.FinalizationError, "repeat coverage"):
            controller.build_query_level_pairs(incomplete, selections, ["f0"])

    def test_paired_statistics_report_percentiles_and_reproducible_bootstrap(self) -> None:
        paired = controller.build_query_level_pairs(
            final_rows(), selection_rows(), ["f0"]
        )

        first = controller.summarize_paired_final(
            paired,
            ["f0"],
            bootstrap_samples=100,
            bootstrap_seed=23,
        )
        second = controller.summarize_paired_final(
            paired,
            ["f0"],
            bootstrap_samples=100,
            bootstrap_seed=23,
        )

        summaries, pairwise = first
        self.assertEqual(first, second)
        self.assertEqual(len(summaries), 9)
        self.assertEqual(len(pairwise), 9)
        full = next(
            row
            for row in summaries
            if row["target_recall"] == 0.90 and row["arm"] == "sqlens_full"
        )
        self.assertIn("latency_p50_ms", full)
        self.assertIn("latency_p95_ms", full)
        self.assertIn("latency_p99_ms", full)
        comparison = next(
            row
            for row in pairwise
            if row["target_recall"] == 0.90
            and row["baseline_arm"] == "official"
            and row["contender_arm"] == "sqlens_full"
        )
        self.assertGreater(comparison["speedup_mean"], 1.9)

    def test_runtime_identity_requires_exact_server_sha_and_sqlens_build(self) -> None:
        args = SimpleNamespace(
            server_container="pg",
            pg_user="postgres",
            pg_database="db",
            expected_sqlens_build_id="sqlens-v11-exact",
        )
        source = {"expected_digest": "a" * 64}
        with mock.patch.object(
            controller.binary_controller,
            "server_binary_digest",
            return_value="a" * 64,
        ), mock.patch.object(
            controller,
            "_docker_psql_scalar",
            side_effect=["0.8.2", "sqlens-v11-exact"],
        ):
            identity = controller.verify_runtime_identity(
                args, "sqlens_full", source, "/usr/lib/vector.so"
            )

        self.assertTrue(identity["sha256_exact_match"])
        self.assertTrue(identity["build_id_exact_match"])
        self.assertEqual(identity["loaded_build_id"], "sqlens-v11-exact")

    def test_runtime_identity_rejects_wrong_exact_build(self) -> None:
        args = SimpleNamespace(
            server_container="pg",
            pg_user="postgres",
            pg_database="db",
            expected_sqlens_build_id="sqlens-v11-expected",
        )
        with mock.patch.object(
            controller.binary_controller,
            "server_binary_digest",
            return_value="a" * 64,
        ), mock.patch.object(
            controller,
            "_docker_psql_scalar",
            side_effect=["0.8.2", "sqlens-v11-other"],
        ):
            with self.assertRaisesRegex(controller.RuntimeIdentityError, "build ID mismatch"):
                controller.verify_runtime_identity(
                    args,
                    "sqlens_disabled",
                    {"expected_digest": "a" * 64},
                    "/usr/lib/vector.so",
                )

    def test_cache_protocol_records_every_synchronous_relation_prewarm(self) -> None:
        args = SimpleNamespace(
            prewarm_relations=["public.items", "public.items_hnsw_idx"],
            server_container="pg",
            pg_user="postgres",
            pg_database="db",
        )
        with mock.patch.object(
            controller, "_docker_psql_scalar", side_effect=["100", "40"]
        ):
            evidence = controller.execute_cache_protocol(
                args, "official", "final", 0
            )

        self.assertTrue(evidence["complete"])
        self.assertEqual([row["blocks"] for row in evidence["records"]], [100, 40])
        self.assertEqual(
            evidence["protocol_sha256"],
            controller.cache_protocol_spec(args.prewarm_relations)["sha256"],
        )

    def test_checkpoint_claim_refuses_overwrite_and_cross_spec_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            old = {
                "run_uuid": "run",
                "controller_spec_sha256": "a" * 64,
            }
            path.write_text(json.dumps(old), encoding="utf-8")

            with self.assertRaises(FileExistsError):
                controller.claim_manifest(path, old, resume=False)
            with self.assertRaisesRegex(controller.CheckpointError, "does not match"):
                controller.claim_manifest(
                    path,
                    {
                        "run_uuid": "run",
                        "controller_spec_sha256": "b" * 64,
                    },
                    resume=True,
                )

    def test_upstream_argv_preserves_fixed_formal_contract(self) -> None:
        args = controller.build_parser().parse_args(
            [
                "--server-container",
                "pg",
                "--official-vector-so",
                "/tmp/official.so",
                "--sqlens-vector-so",
                "/tmp/sqlens.so",
                "--sqlens-vector-so-sha256",
                "a" * 64,
                "--run-uuid",
                "run",
                "--data-epoch",
                "amazon10m-v1",
                "--graph-identity-json",
                "/tmp/graph.json",
                "--prewarm-relation",
                "public.items",
            ]
        )
        args.prewarm_relations = ["public.items"]

        argv = controller.build_upstream_runner_argv(
            args, "official", "final", 1
        )

        pairs = list(zip(argv, argv[1:]))
        self.assertIn(("--screen-repeats", "1"), pairs)
        self.assertIn(("--verification-repeats", "2"), pairs)
        self.assertIn(("--final-repeats", "6"), pairs)
        self.assertIn(("--candidate-validity-predicate", "embedding_valid"), pairs)
        self.assertIn(("--prewarm-relation", "public.items"), pairs)
        self.assertIn(("--final-block", "1"), pairs)

    def test_full_screen_promotion_keeps_fast_near_target_and_boundary(self) -> None:
        configs = [
            SimpleNamespace(
                ef_search=250,
                max_scan_tuples=1000,
                scan_mem_multiplier=1.0,
                iterative_scan="off",
                guided_collect_target=250,
                label="ef250",
            ),
            SimpleNamespace(
                ef_search=500,
                max_scan_tuples=1000,
                scan_mem_multiplier=1.0,
                iterative_scan="off",
                guided_collect_target=500,
                label="ef500",
            ),
            SimpleNamespace(
                ef_search=1000,
                max_scan_tuples=1000,
                scan_mem_multiplier=1.0,
                iterative_scan="off",
                guided_collect_target=1000,
                label="ef1000",
            ),
        ]
        rows = [
            {
                "filter_name": "f0",
                "config": configs[0].label,
                "ok": 20,
                "errors": 0,
                "rows_complete": True,
                "recall_mean": 0.89,
                "latency_mean_ms": 3.0,
            },
            {
                "filter_name": "f0",
                "config": configs[1].label,
                "ok": 20,
                "errors": 0,
                "rows_complete": True,
                "recall_mean": 0.96,
                "latency_mean_ms": 5.0,
            },
            {
                "filter_name": "f0",
                "config": configs[2].label,
                "ok": 20,
                "errors": 0,
                "rows_complete": True,
                "recall_mean": 1.0,
                "latency_mean_ms": 10.0,
            },
        ]

        promoted, proof = controller.promote_full_sqlens_configs(
            rows, configs, "f0", 0.02
        )

        labels = {config.label for config in promoted}
        self.assertIn(configs[1].label, labels)
        self.assertIn(configs[2].label, labels)
        self.assertEqual(len(proof), len(promoted))


if __name__ == "__main__":
    unittest.main()
