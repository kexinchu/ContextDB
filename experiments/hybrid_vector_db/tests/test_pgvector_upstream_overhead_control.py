import argparse
import contextlib
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock


from experiments.hybrid_vector_db.scripts import pgvector_upstream_overhead_control as runner


class Cursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.statements = []

    def execute(self, sql, *_args):
        self.statements.append(sql)

    def fetchone(self):
        return self.rows.pop(0)

    def fetchall(self):
        return self.rows.pop(0)


def completed_summary(config, recall, latency):
    return {
        "config_label": config.label,
        "config_family": config.family,
        "complete": True,
        "recall_mean": recall,
        "recall_lcb95": recall,
        "latency_mean_ms": latency,
        "latency_p95_ms": latency,
        "latency_p99_ms": latency,
    }


class PgvectorUpstreamOverheadControlTests(unittest.TestCase):
    def test_relation_prewarm_is_synchronous_complete_and_hashed(self):
        args = SimpleNamespace(
            prewarm_relations=["public.items", "public.items_hnsw_idx"]
        )
        cur = Cursor(
            [
                (11, 21, 16_384, 8_192),
                (2,),
                (12, 22, 24_576, 8_192),
                (3,),
            ]
        )

        evidence = runner.prewarm_relations(cur, args)

        self.assertTrue(evidence["complete"])
        self.assertEqual(evidence["mode"], "read")
        self.assertEqual(evidence["fork"], "main")
        self.assertEqual([row["warmed_blocks"] for row in evidence["records"]], [2, 3])
        self.assertRegex(evidence["prewarm_spec_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            sum("pg_prewarm" in statement for statement in cur.statements), 2
        )

    def test_relation_prewarm_rejects_partial_block_coverage(self):
        args = SimpleNamespace(prewarm_relations=["public.items"])
        cur = Cursor([(11, 21, 16_384, 8_192), (1,)])

        with self.assertRaisesRegex(RuntimeError, "block count mismatch"):
            runner.prewarm_relations(cur, args)

    def test_default_ladder_uses_only_the_official_upstream_ef_range(self):
        raw = runner.default_config_ladder()
        effective, proof = runner.effective_config_grid(raw)

        self.assertEqual(len(raw), 77)
        self.assertEqual(len(effective), 77)
        self.assertEqual(proof["dropped_equivalent_configs"], 0)
        self.assertLessEqual(
            max(config.ef_search for config in effective),
            runner.UPSTREAM_MAX_EF_SEARCH,
        )
        self.assertEqual(
            {family: sum(config.family == family for config in effective) for family in {config.family for config in effective}},
            {"off": 11, "strict_order": 33, "relaxed_order": 33},
        )
        self.assertEqual(
            len({(config.max_scan_tuples, config.scan_mem_multiplier) for config in effective if config.family == "strict_order"}),
            3,
        )
        self.assertTrue(set(runner.LOW_EF_VALUES).issubset({config.ef_search for config in effective}))

    def test_off_configs_dedup_max_scan_and_memory_as_semantically_irrelevant(self):
        configs = runner.default_config_ladder()
        configs.extend(
            [
                runner.Config(250, "off", 5_000_000, 32.0, 9),
                runner.Config(250, "off", 1_000_000, 8.0, 4),
            ]
        )

        effective, proof = runner.effective_config_grid(configs)

        self.assertEqual(len(effective), 77)
        self.assertEqual(proof["dropped_equivalent_configs"], 2)
        off_250 = [config for config in effective if config.family == "off" and config.ef_search == 250]
        self.assertEqual(len(off_250), 1)
        self.assertEqual(off_250[0].max_scan_tuples, runner.OFF_REPRESENTATIVE_MAX_SCAN)
        self.assertEqual(off_250[0].scan_mem_multiplier, runner.OFF_REPRESENTATIVE_SCAN_MEM)

        conflicting = runner.default_config_ladder() + [
            runner.Config(250, "strict_order", 100_000, 1.0, 99)
        ]
        with self.assertRaisesRegex(ValueError, "conflicting budget_rank"):
            runner.effective_config_grid(conflicting)

    def test_formal_grid_does_not_require_exploratory_relaxed_order(self):
        configs = [
            runner.Config(100, "off", 100_000, 1.0, 0),
            runner.Config(100, "strict_order", 100_000, 1.0, 1),
            runner.Config(100, "strict_order", 5_000_000, 32.0, 3),
        ]

        effective, proof = runner.effective_config_grid(configs)
        max_budget = runner.family_max_budget_configs(effective)

        self.assertEqual(proof["families"], ["off", "strict_order"])
        self.assertEqual(proof["required_formal_families"], ["off", "strict_order"])
        self.assertEqual(set(max_budget), {"off", "strict_order"})
        self.assertEqual(max_budget["strict_order"].budget_rank, 3)

    def test_default_query_count_bound_is_53200_per_implementation(self):
        counts = runner.default_query_count_bounds(28)

        self.assertEqual(counts["screen_queries"], 7_840)
        self.assertEqual(counts["max_promoted_configs_per_filter"], 28)
        self.assertEqual(counts["verification_query_upper_bound"], 62_720)
        self.assertEqual(counts["final_query_upper_bound"], 21_000)
        self.assertEqual(counts["total_query_upper_bound"], 91_560)

    def test_custom_ladder_rejects_values_above_the_declared_ceiling(self):
        with self.assertRaisesRegex(ValueError, "official pgvector"):
            runner.config_from_mapping(
                {
                    "ef_search": 1500,
                    "iterative_scan": "strict_order",
                    "max_scan_tuples": 5_000_000,
                    "scan_mem_multiplier": 32,
                    "budget_rank": 3,
                }
            )

        accepted = runner.config_from_mapping(
            {
                "ef_search": 100_000,
                "iterative_scan": "strict_order",
                "max_scan_tuples": 5_000_000,
                "scan_mem_multiplier": 32,
                "budget_rank": 3,
            },
            max_ef_search=100_000,
        )
        self.assertEqual(accepted.ef_search, 100_000)

    def test_ef100000_formal_ladder_uses_sparse_high_ef_anchors(self):
        configs = runner.load_config_ladder(
            Path(
                "experiments/hybrid_vector_db/configs/"
                "pgvector_v082_ef100000_formal_ladder.csv"
            ),
            max_ef_search=100_000,
        )
        high_ef_values = sorted(
            {config.ef_search for config in configs if config.ef_search > 10_000}
        )

        self.assertEqual(
            high_ef_values,
            [15_000, 20_000, 30_000, 50_000, 75_000, 100_000],
        )
        self.assertEqual(max(config.ef_search for config in configs), 100_000)
        proof = runner.validate_formal_ladder(configs, 100_000)
        self.assertEqual(proof["required_low_ef_values"], list(runner.LOW_EF_VALUES))
        self.assertEqual(proof["observed_ef_values"][: len(runner.LOW_EF_VALUES)], list(runner.LOW_EF_VALUES))
        self.assertTrue(
            any(
                config.ef_search == 100_000
                and config.family == "strict_order"
                and config.budget_rank == 3
                for config in configs
            )
        )

    def test_formal_ladder_rejects_a_missing_low_ef_or_strict_budget_rung(self):
        configs = runner.load_config_ladder(
            Path("experiments/hybrid_vector_db/configs/pgvector_v082_ef10000_formal_ladder.csv"),
            max_ef_search=10_000,
        )
        without_low = [config for config in configs if config.ef_search != 20]
        with self.assertRaisesRegex(runner.ProvenanceGateError, "required low-ef"):
            runner.validate_formal_ladder(without_low, 10_000)

        without_rung = [
            config
            for config in configs
            if not (config.ef_search == 100 and config.family == "strict_order" and config.budget_rank == 3)
        ]
        with self.assertRaisesRegex(runner.ProvenanceGateError, "rungs are incomplete"):
            runner.validate_formal_ladder(without_rung, 10_000)

    def test_candidate_validity_is_part_of_sql_but_not_the_workload_predicate(self):
        sql = runner.build_hybrid_sql(
            "public.items",
            "rating = 5",
            10,
            "embedding_valid",
        )
        normalized = " ".join(sql.split())
        self.assertIn(
            "WHERE (rating = 5) AND (embedding_valid) AND id <> %s",
            normalized,
        )
        with self.assertRaisesRegex(ValueError, "candidate-validity"):
            runner.build_hybrid_sql("public.items", "rating = 5", 10, "true; DROP")

    def test_upstream_evaluation_patch_must_equal_the_canonical_two_file_diff(self):
        canonical = Path("patches/pgvector-v0.8.2-ef-search-100000.patch").read_bytes()
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            patch = Path(temporary) / "ceiling.patch"
            patch.write_bytes(canonical)
            command_results = [
                SimpleNamespace(returncode=0, stdout=canonical, stderr=b""),
                SimpleNamespace(
                    returncode=0,
                    stdout="src/hnsw.c\nsrc/hnsw.h\n",
                    stderr="",
                ),
                SimpleNamespace(returncode=0, stdout="", stderr=""),
            ]
            with mock.patch.object(runner.subprocess, "run", side_effect=command_results):
                proof = runner.upstream_parameter_ceiling_provenance(
                    source, patch, 100_000
                )

            self.assertFalse(proof["algorithm_change"])
            self.assertEqual(
                proof["patch_sha256"], runner.EVALUATION_EF_PATCH_SHA256[100_000]
            )
            self.assertEqual(proof["changed_files"], ["src/hnsw.c", "src/hnsw.h"])

            patch.write_bytes(canonical + b"\n")
            with self.assertRaisesRegex(runner.ProvenanceGateError, "canonical"):
                runner.upstream_parameter_ceiling_provenance(source, patch, 100_000)

    def test_legacy_ef10000_patch_remains_provenance_valid(self):
        canonical = Path("patches/pgvector-v0.8.2-ef-search-10000.patch")
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            with mock.patch.object(
                runner.subprocess,
                "run",
                side_effect=[
                    SimpleNamespace(
                        returncode=0,
                        stdout=canonical.read_bytes(),
                        stderr=b"",
                    ),
                    SimpleNamespace(
                        returncode=0,
                        stdout="src/hnsw.c\nsrc/hnsw.h\n",
                        stderr="",
                    ),
                    SimpleNamespace(returncode=0, stdout="", stderr=""),
                ],
            ):
                proof = runner.upstream_parameter_ceiling_provenance(
                    source, canonical, 10_000
                )

        self.assertEqual(
            proof["patch_sha256"], runner.EVALUATION_EF_PATCH_SHA256[10_000]
        )

    def test_source_tag_must_resolve_to_the_declared_build_commit(self):
        with mock.patch.object(
            runner.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="abc123\n", stderr=""),
        ):
            proof = runner.source_tag_provenance(Path("."), "v0.8.2", "abc123")
        self.assertEqual(proof["source_tag_commit"], "abc123")

        with mock.patch.object(
            runner.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="different\n", stderr=""),
        ):
            with self.assertRaisesRegex(runner.ProvenanceGateError, "tag/commit mismatch"):
                runner.source_tag_provenance(Path("."), "v0.8.2", "abc123")

    def test_promotion_includes_margin_winner_family_recall_and_max_budget_proofs(self):
        configs = [
            runner.Config(100, "off", 100_000, 1.0, 0),
            runner.Config(200, "off", 100_000, 1.0, 0),
            runner.Config(100, "strict_order", 100_000, 1.0, 1),
            runner.Config(200, "strict_order", 5_000_000, 32.0, 3),
            runner.Config(100, "relaxed_order", 100_000, 1.0, 1),
            runner.Config(200, "relaxed_order", 5_000_000, 32.0, 3),
        ]
        recalls = [0.90, 0.91, 0.96, 0.97, 0.94, 0.95]
        latencies = [1.0, 3.0, 5.0, 8.0, 2.0, 6.0]
        summaries = [
            completed_summary(config, recall, latency)
            for config, recall, latency in zip(configs, recalls, latencies)
        ]

        promoted, proof = runner.build_promotion_set(summaries, configs, [0.95], margin=0.02)
        reasons = {row["config_label"]: row["promotion_reasons"] for row in proof}

        self.assertEqual(
            {config.label for config in promoted},
            {configs[1].label, configs[3].label, configs[4].label, configs[5].label},
        )
        self.assertIn("fastest_screen_target_0.95_minus_margin_0.02", reasons[configs[4].label])
        self.assertIn("family_strict_order_max_screen_recall", reasons[configs[3].label])
        self.assertIn("global_max_screen_recall", reasons[configs[3].label])
        self.assertIn("family_off_maximum_budget_verification_boundary", reasons[configs[1].label])

    def test_promotion_and_selection_never_cross_the_declared_family(self):
        off = runner.Config(100, "off", 100_000, 1.0, 0)
        strict = runner.Config(100, "strict_order", 100_000, 1.0, 1)
        summaries = [
            completed_summary(off, 0.99, 100.0),
            completed_summary(strict, 0.96, 5.0),
        ]

        promoted, _proof = runner.build_promotion_set(
            summaries, [off, strict], [0.95], margin=0.02, family="strict_order"
        )
        selected, status, _proof = runner.select_verified_config(
            summaries,
            0.95,
            verified_config_labels=[strict.label],
            family="strict_order",
        )

        self.assertEqual(promoted, [strict])
        self.assertEqual(status, "selected")
        self.assertEqual(selected["config_label"], strict.label)

    def test_no_mean_qualified_config_is_unattainable_on_calibration_grid(self):
        labels = {"off": "off-max", "strict_order": "strict-max", "relaxed_order": "relaxed-max"}
        summaries = [
            {
                "config_label": label,
                "complete": True,
                "recall_mean": 0.94,
                "recall_lcb95": 0.94,
                "latency_mean_ms": position + 1.0,
            }
            for position, label in enumerate(labels.values())
        ]

        selected, status, proof = runner.select_verified_config(
            summaries, 0.95, verified_config_labels=list(labels.values())
        )
        self.assertIsNone(selected)
        self.assertEqual(status, "unattainable_on_calibration_grid")
        self.assertTrue(proof["claims_unattainable"])
        self.assertEqual(proof["selection_fallback"], "none_lcb95_qualified")
        self.assertEqual(
            proof["calibration_selection_policy"], "lcb_then_max_recall"
        )
        self.assertEqual(proof["verified_configs"], sorted(labels.values()))

        selected, status, proof = runner.select_verified_config(
            summaries[:-1], 0.95, verified_config_labels=list(labels.values())
        )
        self.assertIsNone(selected)
        self.assertEqual(status, "incomplete_verification")
        self.assertEqual(proof["missing_verified_configs"], ["relaxed-max"])

    def test_verification_selection_prefers_lcb_confirmed_then_latency(self):
        summaries = [
            {"config_label": "slow", "complete": True, "recall_mean": 0.98, "recall_lcb95": 0.94, "latency_mean_ms": 8.0},
            {"config_label": "confirmed", "complete": True, "recall_mean": 0.97, "recall_lcb95": 0.96, "latency_mean_ms": 6.0},
            {"config_label": "fast", "complete": True, "recall_mean": 0.96, "recall_lcb95": 0.90, "latency_mean_ms": 3.0},
            {"config_label": "uncertain", "complete": True, "recall_mean": 0.94, "recall_lcb95": 0.93, "latency_mean_ms": 1.0},
        ]
        selected, status, proof = runner.select_verified_config(
            summaries,
            0.95,
            verified_config_labels=["slow", "confirmed", "fast", "uncertain"],
        )
        self.assertEqual(status, "selected")
        self.assertEqual(selected["config_label"], "confirmed")
        self.assertEqual(proof["selection_fallback"], "none")
        self.assertEqual(proof["mean_qualified_configs"], 3)
        self.assertEqual(proof["lcb95_qualified_configs"], 1)

    def test_verification_selection_rejects_mean_only_recall(self):
        summaries = [
            {"config_label": "fast-lower", "complete": True, "recall_mean": 0.96, "recall_lcb95": 0.90, "latency_mean_ms": 2.0},
            {"config_label": "slow-best", "complete": True, "recall_mean": 0.98, "recall_lcb95": 0.94, "latency_mean_ms": 8.0},
            {"config_label": "fast-best", "complete": True, "recall_mean": 0.98, "recall_lcb95": 0.93, "latency_mean_ms": 5.0},
        ]

        selected, status, proof = runner.select_verified_config(
            summaries,
            0.95,
            verified_config_labels=["fast-lower", "slow-best", "fast-best"],
        )

        self.assertIsNone(selected)
        self.assertEqual(status, "unattainable_on_calibration_grid")
        self.assertEqual(proof["selection_fallback"], "none_lcb95_qualified")
        self.assertEqual(proof["lcb95_qualified_configs"], 0)
        self.assertTrue(proof["claims_unattainable"])

    def test_heldout_final_mean_recall_miss_is_never_reported_as_success(self):
        metrics = {"complete": True, "recall_mean": 0.94, "recall_lcb95": 0.90}

        self.assertEqual(
            runner.heldout_final_status("selected", 0.95, metrics),
            "missed_target",
        )
        self.assertEqual(
            runner.heldout_final_status(
                "selected", 0.95, {"complete": True, "recall_mean": 0.96, "recall_lcb95": 0.90}
            ),
            "confirmed",
        )

    def test_query_splits_are_disjoint_and_cover_q0_through_q199(self):
        contract = runner.validate_split_contract()

        self.assertEqual(contract["screen"], {"first": 0, "last": 19, "queries": 20})
        self.assertEqual(contract["verification"], {"first": 20, "last": 99, "queries": 80})
        self.assertEqual(contract["final"], {"first": 100, "last": 199, "queries": 100})
        self.assertFalse(set(runner.SCREEN_QUERY_NOS) & set(runner.VERIFICATION_QUERY_NOS))
        self.assertFalse(set(runner.VERIFICATION_QUERY_NOS) & set(runner.FINAL_QUERY_NOS))

    def test_official_runtime_gate_uses_extension_version_but_not_stale_sql_declarations(self):
        cursor = Cursor([("0.8.2",)])

        provenance = runner.gate_implementation(cursor, "official")

        self.assertFalse(provenance["runtime_sql_declarations_used_as_identity"])
        self.assertEqual(len(cursor.statements), 1)
        self.assertNotIn("vector_sqlens_build_id", cursor.statements[0])

    def test_sqlens_gate_defaults_to_current_v15_and_profile_semantics(self):
        profile = {
            "profile_semantics_version": 9,
            "graph_elements_visited": 10,
            "raw_index_tids_returned": 4,
            "hnsw_am_callback_ms": 1.0,
            "executor_residual_ms": 0.5,
        }
        profile.update({field: 0 for field in runner.SQLENS_PROFILE_FIELDS if field not in profile})
        build_id = "sqlens-v16-d3-full-materialization-persisted-reuse-amazon"
        cursor = Cursor([("0.8.2",), (build_id,), (json.dumps(profile),)])

        provenance = runner.gate_implementation(cursor, "sqlens_disabled")

        self.assertEqual(provenance["loaded_vector_sqlens_build_id"], build_id)
        self.assertEqual(
            provenance["profile_gate"]["required_build_prefix"],
            "sqlens-v16-d3-full-materialization-persisted-reuse-",
        )
        self.assertEqual(provenance["profile_gate"]["minimum_profile_semantics_version"], 9.0)

        old = profile | {"profile_semantics_version": 8}
        with self.assertRaises(runner.ProvenanceGateError):
            runner.gate_implementation(
                Cursor([("0.8.2",), (build_id,), (json.dumps(old),)]),
                "sqlens_disabled",
            )

    def test_sqlens_exact_r36_identity_is_equality_not_prefix(self):
        profile = {
            "profile_semantics_version": 12,
            **{field: 0 for field in runner.SQLENS_PROFILE_FIELDS},
        }
        exact = runner.DEFAULT_SQLENS_BUILD_ID
        cursor = Cursor([("0.8.2",), (exact,), (json.dumps(profile),)])

        provenance = runner.gate_implementation(
            cursor,
            "sqlens_disabled",
            expected_sqlens_build_id=exact,
        )

        self.assertEqual(provenance["loaded_vector_sqlens_build_id"], exact)
        self.assertTrue(provenance["profile_gate"]["exact_build_id_match"])
        with self.assertRaisesRegex(runner.ProvenanceGateError, "exact build ID"):
            runner.gate_implementation(
                Cursor(
                    [
                        ("0.8.2",),
                        (exact + "-different",),
                        (json.dumps(profile),),
                    ]
                ),
                "sqlens_disabled",
                expected_sqlens_build_id=exact,
            )

    def test_sqlens_disabled_per_query_profile_fails_closed_on_d1_d2_or_d3(self):
        scan = {
            "valid": True,
            "final_path": "stock",
            "filter_strategy": "off",
            "traversal_guidance_scope": "none",
            "approximate_prioritization_attempted": False,
            "traversal_order_changed": False,
            "guidance_checks": 0,
            "traversal_guidance_checks": 0,
            "neighbor_expansion_guidance_checks": 0,
            "priority_reorders": 0,
            "page_access_prefetches": 0,
            "index_page_prefetches": 0,
        }
        guidance = {
            "active": False,
            "effective_active": False,
            "composed_exact_active": False,
        }

        proof = runner.assert_sqlens_disabled_query_profile(
            Cursor([(json.dumps(scan),), (json.dumps(guidance),)])
        )

        self.assertTrue(proof["verified"])
        self.assertEqual(proof["final_path"], "stock")
        self.assertRegex(proof["profile_sha256"], r"^[0-9a-f]{64}$")
        for tampered_scan, tampered_guidance in (
            (scan | {"final_path": "validation_only"}, guidance),
            (scan | {"priority_reorders": 1}, guidance),
            (scan | {"index_page_prefetches": 1}, guidance),
            (scan, guidance | {"composed_exact_active": True}),
        ):
            with self.assertRaisesRegex(
                runner.ProvenanceGateError, "stock-path gate failed"
            ):
                runner.assert_sqlens_disabled_query_profile(
                    Cursor(
                        [
                            (json.dumps(tampered_scan),),
                            (json.dumps(tampered_guidance),),
                        ]
                    )
                )

    def test_frozen_workloads_use_request_number_as_pairing_identity(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "workload.csv"
            path.write_text(
                "request_no,query_no,query_id,filter_name,split\n"
                "0,200,900,f0,measurement\n"
                "1,201,901,f1,measurement\n",
                encoding="utf-8",
            )

            requests = runner.load_frozen_workload(
                path,
                expected_requests=2,
                expected_split="measurement",
                filter_names={"f0", "f1"},
            )

        self.assertEqual(
            requests,
            [
                runner.WorkloadRequest(0, 200, 900, "f0"),
                runner.WorkloadRequest(1, 201, 901, "f1"),
            ],
        )
        config = runner.Config(100, "off", 100_000, 1.0)
        row = {field: "" for field in runner.RAW_FIELDS} | {
            "implementation": "official",
            "phase": "final",
            "query_split": "final",
            "filter_name": "f0",
            "request_no": 0,
            "query_no": 200,
            "query_id": 900,
            "repeat": 2,
            "config_label": config.label,
            "config_family": config.family,
            "budget_rank": config.budget_rank,
            "ef_search": config.ef_search,
            "iterative_scan": config.iterative_scan,
            "max_scan_tuples": config.max_scan_tuples,
            "scan_mem_multiplier": config.scan_mem_multiplier,
            "measurement_key": runner.measurement_key(
                "official", "final", "f0", 0, 2, config.label
            ),
        }
        self.assertEqual(
            runner.validate_stage_checkpoint(
                [row],
                "official",
                "final",
                {"f0": [config]},
                [200],
                3,
                repeat_values=[2],
                workload_requests=requests[:1],
            ),
            {("final", "f0")},
        )
        with self.assertRaises(runner.CheckpointContractError):
            runner.validate_stage_checkpoint(
                [row | {"query_id": 999}],
                "official",
                "final",
                {"f0": [config]},
                [200],
                3,
                repeat_values=[2],
                workload_requests=requests[:1],
            )

    def test_frozen_stage_replays_global_request_order_without_filter_grouping(self):
        config = runner.Config(100, "off", 100_000, 1.0)
        requests = [
            runner.WorkloadRequest(1, 201, 901, "f0"),
            runner.WorkloadRequest(0, 200, 900, "f1"),
        ]
        truth = {
            ("f0", 201): runner.TruthEntry(
                901, 1.0, 0.0, tuple(range(10)), True
            ),
            ("f1", 200): runner.TruthEntry(
                900, 1.0, 0.0, tuple(range(10)), True
            ),
        }
        observed: list[int] = []

        def measurement(*args, **_kwargs):
            request_no = int(args[-1])
            observed.append(request_no)
            return {field: "" for field in runner.RAW_FIELDS} | {
                "measurement_key": f"m{request_no}",
            }

        with TemporaryDirectory() as temporary, mock.patch.object(
            runner, "measurement_row", side_effect=measurement
        ), mock.patch.object(
            runner,
            "enforce_hnsw_guc_allowlist",
            return_value={"actions": [], "after": {}},
        ), mock.patch.object(runner, "configure_stock"):
            args = SimpleNamespace(
                table="public.items",
                k=10,
                candidate_validity_predicate="",
                schedule_seed=7,
                implementation="official",
                run_uuid="run",
                execution_stage="final",
                planner_mode="auto",
                paths={"raw": Path(temporary) / "raw.csv"},
                guc_block_audits=[],
            )
            rows: list[dict[str, object]] = []
            completed: set[tuple[str, str]] = set()
            runner.run_stage_blocks(
                mock.MagicMock(),
                args,
                "final",
                [
                    {"filter_name": "f0", "predicate": "id >= 0"},
                    {"filter_name": "f1", "predicate": "id >= 0"},
                ],
                {"f0": [config], "f1": [config]},
                [200, 201],
                1,
                truth,
                {900: "[0,1]", 901: "[0,1]"},
                rows,
                completed,
                workload_requests=requests,
            )

        self.assertEqual(observed, [0, 1])
        self.assertEqual(completed, {("final", "f0"), ("final", "f1")})

    def test_official_index_build_manifest_binds_live_source_index(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "index-build.json"
            payload = {
                "artifact_valid": True,
                "table": "public.items",
                "index": "public.items_hnsw",
                "builder": {
                    "vector_so_sha256": runner.OFFICIAL_UPSTREAM_VECTOR_SO_SHA256,
                    "source_tag": "v0.8.2",
                    "source_commit": "cab9",
                },
                "index_fingerprint": {
                    "index_oid": 12,
                    "index_relfilenode": 34,
                    "indexdef_sha256": "a" * 64,
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            identity = runner.load_official_index_build_identity(
                path,
                table="public.items",
                index="public.items_hnsw",
                official_source_commit="cab9",
            )

        runner.verify_live_official_index_identity(
            identity,
            {
                "index_oid": 12,
                "index_relfilenode": 34,
                "indexdef_sha256": "a" * 64,
            },
        )
        with self.assertRaisesRegex(
            runner.ProvenanceGateError, "relfilenode"
        ):
            runner.verify_live_official_index_identity(
                identity,
                {
                    "index_oid": 12,
                    "index_relfilenode": 35,
                    "indexdef_sha256": "a" * 64,
                },
            )

    def test_frozen_q10k_runtime_gate_accepts_only_all_r3_and_shared_index(self):
        with TemporaryDirectory() as temporary:
            index_build = Path(temporary) / "index-build.json"
            index_build.write_text("{}\n", encoding="utf-8")
            args = runner.build_parser().parse_args(
                [
                    "--implementation",
                    "sqlens_disabled",
                    "--dsn",
                    "postgresql://postgres@localhost/hybrid",
                    "--server-container",
                    "pg-server",
                    "--expected-vector-so-sha256",
                    "b" * 64,
                    "--vector-source-tag",
                    "sqlens-r36",
                    "--vector-source-commit",
                    "sqlens-commit",
                    "--vector-source-repo",
                    str(Path(__file__).parent),
                    "--vector-build-recipe",
                    "make",
                    "--vector-compiler-flags=-O3",
                    "--run-uuid",
                    "p0-2",
                    "--data-epoch",
                    "amazon10m-v1",
                    "--filters-csv",
                    "experiments/hybrid_vector_db/configs/"
                    "amazon10m_selectivity14_valid_embeddings_filters.csv",
                    "--truth-csv",
                    "results/hybrid_vector_db/"
                    "amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal.csv",
                    "--calibration-workload-csv",
                    "results/hybrid_vector_db/figure5_r35_amazon_calibration.csv",
                    "--measurement-workload-csv",
                    "results/hybrid_vector_db/figure5_r35_amazon_measurement.csv",
                    "--index",
                    "public.shared_official_idx",
                    "--source-index",
                    "public.shared_official_idx",
                    "--official-index-build-manifest",
                    str(index_build),
                    "--official-index-source-commit",
                    "cab9da72",
                    "--formal-family",
                    "all",
                    "--final-repeats",
                    "3",
                    "--expected-sqlens-build-id",
                    runner.DEFAULT_SQLENS_BUILD_ID,
                ]
            )

            runner.validate_runtime_args(args)

            args.formal_family = "off"
            with self.assertRaisesRegex(
                runner.ProvenanceGateError, "formal-family all"
            ):
                runner.validate_runtime_args(args)

    def test_sqlens_disabled_resets_every_current_v11_extension_guc(self):
        cursor = Cursor([])

        statements = runner.disable_sqlens_gucs(cursor)

        self.assertEqual(statements, cursor.statements)
        for guc in (
            "hnsw.filter_strategy",
            "hnsw.page_access",
            "hnsw.index_page_access",
            "hnsw.guidance_compose_exact_or",
            "hnsw.guidance_require_epoch",
            "hnsw.require_full_memory_build",
        ):
            self.assertIn(f"SET {guc} = off", statements)
        for guc in ("hnsw.metadata_cache_max_mb", "hnsw.build_page_order", "hnsw.build_seed"):
            self.assertIn(f"RESET {guc}", statements)
        for guc in ("hnsw.clone_source", "hnsw.preferred_index"):
            self.assertIn(f"SET {guc} = ''", statements)

    def test_runtime_hnsw_guc_inventory_forces_every_nonstock_knob_safe(self):
        class GucCursor:
            def __init__(self):
                self.statements = []
                self.inventory_reads = 0

            def execute(self, sql, *_args):
                self.statements.append(sql)
                if "FROM pg_settings" in sql:
                    self.inventory_reads += 1

            def fetchall(self):
                if self.inventory_reads == 1:
                    return [
                        ("hnsw.ef_search", "integer", "40", "40"),
                        ("hnsw.clone_source", "string", "public.old", ""),
                        ("hnsw.preferred_index", "string", "public.idx", ""),
                        ("hnsw.traversal_guidance", "bool", "on", "off"),
                        ("hnsw.experimental_budget", "integer", "9", "0"),
                    ]
                return [
                    ("hnsw.ef_search", "integer", "40", "40"),
                    ("hnsw.clone_source", "string", "", ""),
                    ("hnsw.preferred_index", "string", "", ""),
                    ("hnsw.traversal_guidance", "bool", "off", "off"),
                    ("hnsw.experimental_budget", "integer", "0", "0"),
                ]

        cursor = GucCursor()
        audit = runner.enforce_hnsw_guc_allowlist(cursor)

        self.assertEqual(audit["stock_allowlist"], sorted(runner.STOCK_HNSW_GUCS))
        self.assertEqual(audit["unhandled_nonstock_gucs"], [])
        self.assertIn("SET hnsw.clone_source = ''", cursor.statements)
        self.assertIn("SET hnsw.preferred_index = ''", cursor.statements)
        self.assertIn("SET hnsw.traversal_guidance = off", cursor.statements)
        self.assertIn("RESET hnsw.experimental_budget", cursor.statements)

    def test_measurement_timer_stops_immediately_after_fetchall(self):
        config = runner.Config(100, "off", 100_000, 1.0)
        truth = runner.TruthEntry(7, 1.0, 0.0, tuple(range(10)), True)

        class MeasurementCursor:
            def execute(self, *_args):
                return None

            def fetchall(self):
                return [(1, 0.5)]

        with mock.patch.object(runner.time, "perf_counter", side_effect=[10.0, 10.125]), \
                mock.patch.object(runner, "tie_aware_recall", return_value=0.9) as recall:
            row = runner.measurement_row(
                "official", "final", "f", 100, 7, 0, config, 1,
                "[0,1]", truth, MeasurementCursor(), "SELECT 1", 10,
            )

        self.assertEqual(row["latency_ms"], 125.0)
        recall.assert_called_once()

    def test_database_fingerprint_binds_cluster_database_relations_and_epoch(self):
        cursor = Cursor([
            ("cluster-123", "db", 42, "postgres", "127.0.0.1", 55432, "17.5", "0.8.2"),
            (10_000_000, 0, 9_999_999, 100, 101),
            (200, 201, "CREATE INDEX idx ON public.items USING hnsw (embedding vector_l2_ops)"),
        ])

        fingerprint = runner.database_fingerprint(
            cursor, "public.items", "public.idx", "amazon10m-v1"
        )

        self.assertEqual(fingerprint["system_identifier"], "cluster-123")
        self.assertEqual(fingerprint["database_oid"], 42)
        self.assertEqual(fingerprint["table_oid"], 100)
        self.assertEqual(fingerprint["index_oid"], 200)
        self.assertEqual(fingerprint["data_epoch"], "amazon10m-v1")
        self.assertEqual(
            fingerprint["indexdef_sha256"],
            hashlib.sha256(
                b"CREATE INDEX idx ON public.items USING hnsw (embedding vector_l2_ops)"
            ).hexdigest(),
        )

    def test_graph_identity_requires_same_heap_logical_equivalence(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "graph.json"
            payload = {
                "source_index": "public.source_idx",
                "clone_index": "public.clone_idx",
                "stable_fingerprint_sha256": "f" * 64,
                "comparison": {
                    "format": "sqlens-hnsw-compare-v2",
                    "same_heap": True,
                    "entry_equal": True,
                    "logical_equal": True,
                    "definition_equal": True,
                    "tuple_coverage_equal": True,
                    "physical_equal": False,
                    "left_definition_digest": "sha256:def",
                    "right_definition_digest": "sha256:def",
                    "left_tuple_coverage_digest": "sha256:tid",
                    "right_tuple_coverage_digest": "sha256:tid",
                    "left_logical_digest": "sha256:logical",
                    "right_logical_digest": "sha256:logical",
                    "left_physical_digest": "sha256:left",
                    "right_physical_digest": "sha256:right",
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            identity = runner.load_graph_identity(
                path, "public.source_idx", "public.clone_idx"
            )
            self.assertTrue(identity["logical_equal"])
            self.assertFalse(identity["physical_equal"])
            self.assertEqual(identity["logical_digest"], "sha256:logical")

            payload["comparison"]["logical_equal"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(runner.ProvenanceGateError):
                runner.load_graph_identity(
                    path, "public.source_idx", "public.clone_idx"
                )

            payload["comparison"]["logical_equal"] = True
            payload["comparison"]["right_logical_digest"] = "sha256:other"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(runner.ProvenanceGateError):
                runner.load_graph_identity(
                    path, "public.source_idx", "public.clone_idx"
                )

    def test_formal_design_is_fixed_to_fourteen_filters_three_targets_and_declared_family(self):
        filters = [{"filter_name": f"f{i}"} for i in range(14)]
        design = runner.validate_formal_design(filters, [0.90, 0.95, 0.99], "strict_order")

        self.assertEqual(design["cell_count"], 42)
        self.assertEqual(design["formal_family"], "strict_order")
        self.assertEqual(design["target_metric"], "query_level_mean_recall_at_10")
        self.assertEqual(
            design["calibration_selection_policy"], "lcb_then_max_recall"
        )
        self.assertIn("LCB95", design["calibration_selection_rule"])
        with self.assertRaisesRegex(ValueError, "exactly 14"):
            runner.validate_formal_design(filters[:-1], [0.90, 0.95, 0.99], "off")
        with self.assertRaisesRegex(ValueError, "exactly 0.90,0.95,0.99"):
            runner.validate_formal_design(filters, [0.90, 0.95], "off")
        with self.assertRaisesRegex(ValueError, "relaxed_order"):
            runner.validate_formal_design(filters, [0.90, 0.95, 0.99], "relaxed_order")

    def test_checkpoint_spec_includes_runtime_binding_and_run_uuid(self):
        spec = runner.build_checkpoint_spec(
            run_uuid="run-123",
            base_spec={"filters": ["f"]},
            database_fingerprint={"system_identifier": "cluster", "database_oid": 7},
            binary_provenance={"vector_so_sha256": "a" * 64},
            settings_audit={"after": {"hnsw.clone_source": ""}},
        )

        self.assertEqual(spec["run_uuid"], "run-123")
        self.assertEqual(spec["database_fingerprint"]["database_oid"], 7)
        self.assertEqual(spec["binary_provenance"]["vector_so_sha256"], "a" * 64)
        self.assertEqual(spec["checkpoint_spec_sha256"], runner.sha256_json(spec | {"checkpoint_spec_sha256": None}))

    def test_server_binary_gate_hashes_pg_config_vector_so_inside_container(self):
        digest = runner.OFFICIAL_UPSTREAM_VECTOR_SO_SHA256
        command = mock.Mock(
            side_effect=[
                SimpleNamespace(returncode=0, stdout="pgvector-upstream:0.8.2\n", stderr=""),
                SimpleNamespace(returncode=0, stdout="/usr/lib/postgresql/17/lib\n", stderr=""),
                SimpleNamespace(
                    returncode=0,
                    stdout=f"{digest}  /usr/lib/postgresql/17/lib/vector.so\n",
                    stderr="",
                ),
                SimpleNamespace(
                    returncode=0,
                    stdout=f"sha256:{'c' * 64}\n",
                    stderr="",
                ),
                SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "CpusetCpus": "48-63",
                            "CpuPeriod": 100000,
                            "CpuQuota": 0,
                            "NanoCpus": 0,
                            "Memory": 68719476736,
                            "MemorySwap": 68719476736,
                        }
                    ),
                    stderr="",
                ),
            ]
        )

        provenance = runner.server_vector_binary_provenance("pg-server", digest, command)

        self.assertEqual(provenance["vector_so_sha256"], digest)
        self.assertEqual(provenance["server_image"], "pgvector-upstream:0.8.2")
        self.assertEqual(provenance["server_image_id"], f"sha256:{'c' * 64}")
        self.assertEqual(provenance["server_resource_limits"]["cpuset_cpus"], "48-63")
        self.assertEqual(
            provenance["server_resource_limits"]["memory_bytes"], 68719476736
        )
        self.assertEqual(
            command.call_args_list[2].args[0],
            [
                "docker",
                "exec",
                "pg-server",
                "sha256sum",
                "/usr/lib/postgresql/17/lib/vector.so",
            ],
        )

    def test_server_binary_gate_fails_closed_on_digest_mismatch(self):
        actual = "a" * 64
        expected = "b" * 64
        command = mock.Mock(
            side_effect=[
                SimpleNamespace(returncode=0, stdout="image\n", stderr=""),
                SimpleNamespace(returncode=0, stdout="/pkglib\n", stderr=""),
                SimpleNamespace(returncode=0, stdout=f"{actual}  /pkglib/vector.so\n", stderr=""),
            ]
        )
        with self.assertRaisesRegex(runner.ProvenanceGateError, "SHA-256 mismatch"):
            runner.server_vector_binary_provenance("pg-server", expected, command)

    def test_official_formal_args_require_pinned_digest_container_and_source(self):
        args = argparse.Namespace(
            implementation="official",
            dsn="postgresql://upstream.example/amazon",
            server_container="pg-server",
            expected_vector_so_sha256=runner.OFFICIAL_UPSTREAM_VECTOR_SO_SHA256,
            vector_source_tag="v0.8.2",
            vector_source_commit="abc123",
            vector_build_recipe="make",
            vector_compiler_flags="-O3",
            filters_csv=Path(__file__),
            truth_csv=Path(__file__),
            graph_identity_json=Path(__file__),
            vector_source_repo=Path(__file__).parent,
            source_index="public.source_idx",
            clone_index="public.clone_idx",
            run_uuid="run-123",
            data_epoch="amazon10m-v1",
            target_recalls=[0.90, 0.95, 0.99],
            formal_family="off",
            final_repeats=5,
            execution_stage="calibration",
            final_block=None,
            promotion_margin=0.02,
            minimum_sqlens_profile_semantics=9.0,
        )
        with mock.patch.object(
            runner,
            "sha256_file",
            side_effect=[
                runner.FORMAL_FILTERS_SHA256,
                runner.FORMAL_TRUTH_COHORT_SHA256,
            ],
        ):
            runner.validate_runtime_args(args)

        args.expected_vector_so_sha256 = "a" * 64
        with (
            mock.patch.object(
                runner,
                "sha256_file",
                side_effect=[
                    runner.FORMAL_FILTERS_SHA256,
                    runner.FORMAL_TRUTH_COHORT_SHA256,
                ],
            ),
            self.assertRaisesRegex(runner.ProvenanceGateError, "pinned upstream"),
        ):
            runner.validate_runtime_args(args)

    def test_formal_runtime_rejects_implicit_environment_dsn(self):
        with self.assertRaisesRegex(runner.ProvenanceGateError, "explicit non-empty --dsn"):
            runner.dsn_fingerprint("")
        self.assertEqual(
            runner.dsn_fingerprint("postgresql://upstream.example/amazon"),
            runner.dsn_fingerprint("postgresql://upstream.example/amazon"),
        )
        self.assertNotEqual(
            runner.dsn_fingerprint("postgresql://upstream.example/amazon"),
            runner.dsn_fingerprint("postgresql://sqlens.example/amazon"),
        )

    def test_final_stage_uses_five_one_repeat_blocks(self):
        args = argparse.Namespace(
            implementation="sqlens_disabled",
            dsn="postgresql://sqlens-control.example/amazon",
            server_container="pg-server",
            expected_vector_so_sha256="a" * 64,
            vector_source_tag="sqlens",
            vector_source_commit="abc123",
            vector_build_recipe="make",
            vector_compiler_flags="-O3",
            filters_csv=Path(__file__),
            truth_csv=Path(__file__),
            graph_identity_json=Path(__file__),
            vector_source_repo=Path(__file__).parent,
            source_index="public.source_idx",
            clone_index="public.clone_idx",
            run_uuid="run-123",
            data_epoch="amazon10m-v1",
            target_recalls=[0.90, 0.95, 0.99],
            formal_family="off",
            final_repeats=5,
            execution_stage="final",
            final_block=4,
            promotion_margin=0.02,
            minimum_sqlens_profile_semantics=9.0,
            expected_sqlens_build_prefix="sqlens-v12-",
        )
        with mock.patch.object(
            runner,
            "sha256_file",
            side_effect=[
                runner.FORMAL_FILTERS_SHA256,
                runner.FORMAL_TRUTH_COHORT_SHA256,
            ],
        ):
            runner.validate_runtime_args(args)
        args.final_block = 5
        with (
            mock.patch.object(
                runner,
                "sha256_file",
                side_effect=[
                    runner.FORMAL_FILTERS_SHA256,
                    runner.FORMAL_TRUTH_COHORT_SHA256,
                ],
            ),
            self.assertRaisesRegex(runner.ProvenanceGateError, "0..4"),
        ):
            runner.validate_runtime_args(args)

    def test_stock_sql_is_marker_free_and_hnsw_plan_is_required(self):
        sql = runner.build_hybrid_sql("public.items", "rating = 5 AND price <= 10", 10)
        normalized = " ".join(sql.lower().split())

        self.assertIn("where (rating = 5 and price <= 10) and id <> %s", normalized)
        self.assertIn("order by embedding <-> %s::vector limit 10", normalized)
        self.assertNotIn("sqlens", normalized)
        self.assertNotIn("guidance", normalized)

        with self.assertRaisesRegex(RuntimeError, "HNSW EXPLAIN gate failed"):
            runner.assert_hnsw_explain_gate(
                [{"Plan": {"Node Type": "Seq Scan"}}], "public.items_hnsw"
            )

    def test_planner_auto_records_route_without_forcing_hnsw(self):
        cursor = mock.MagicMock()
        cursor.fetchone.return_value = (
            [{"Plan": {"Node Type": "Seq Scan", "Relation Name": "items"}}],
        )

        audit = runner.explain_hybrid(
            cursor,
            runner.build_hybrid_sql("public.items", "rating = 5", 10),
            "[0,1]",
            7,
            "public.items_hnsw",
            planner_mode="auto",
            query_no=20,
        )

        self.assertEqual(audit["route"], "sequential_exact")
        self.assertEqual(audit["representative_query_no"], 20)
        self.assertEqual(audit["index_names"], [])
        with self.assertRaisesRegex(RuntimeError, "HNSW EXPLAIN gate failed"):
            runner.explain_hybrid(
                cursor,
                runner.build_hybrid_sql("public.items", "rating = 5", 10),
                "[0,1]",
                7,
                "public.items_hnsw",
                planner_mode="forced_hnsw",
            )

    def test_planner_auto_resets_planner_switches(self):
        cursor = mock.MagicMock()
        cursor.fetchone.side_effect = [("on",), ("8",), ("on",)]

        settings = runner.configure_planner_session(cursor, "auto", 300_000)

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertIn("RESET enable_seqscan", statements)
        self.assertIn("RESET max_parallel_workers_per_gather", statements)
        self.assertIn("RESET jit", statements)
        self.assertNotIn("SET enable_seqscan = off", statements)
        self.assertEqual(settings["planner_mode"], "auto")

    def test_checkpoint_resume_accepts_only_exact_complete_measurement_key_blocks(self):
        config = runner.Config(100, "off", 100_000, 1.0, 0)
        query_nos = [0, 1]

        def row(query_no):
            key = runner.measurement_key("official", "screen", "f", query_no, 0, config.label)
            return {field: "" for field in runner.RAW_FIELDS} | {
                "implementation": "official",
                "phase": "screen",
                "query_split": "screen",
                "filter_name": "f",
                "query_no": query_no,
                "repeat": 0,
                "config_label": config.label,
                "config_family": config.family,
                "budget_rank": config.budget_rank,
                "ef_search": config.ef_search,
                "iterative_scan": config.iterative_scan,
                "max_scan_tuples": config.max_scan_tuples,
                "scan_mem_multiplier": config.scan_mem_multiplier,
                "measurement_key": key,
            }

        rows = [row(0), row(1)]
        completed = runner.validate_stage_checkpoint(
            rows, "official", "screen", {"f": [config]}, query_nos, 1
        )
        self.assertEqual(completed, {("screen", "f")})

        with self.assertRaises(runner.CheckpointContractError):
            runner.validate_stage_checkpoint(
                rows[:1], "official", "screen", {"f": [config]}, query_nos, 1
            )
        foreign = rows + [row(1) | {"measurement_key": "foreign"}]
        with self.assertRaises(runner.CheckpointContractError):
            runner.validate_stage_checkpoint(
                foreign, "official", "screen", {"f": [config]}, query_nos, 1
            )

    def test_resume_requires_recorded_promotion_hash_before_later_rows(self):
        with self.assertRaises(runner.CheckpointContractError):
            runner.validate_derived_resume_hash(
                {"promotion_set_sha256": "old"},
                "promotion_set_sha256",
                "new",
                later_rows_exist=True,
            )
        with self.assertRaises(runner.CheckpointContractError):
            runner.validate_derived_resume_hash(
                {}, "promotion_set_sha256", "new", later_rows_exist=True
            )

    def test_manifest_only_current_protocol_zero_row_checkpoint_can_resume(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "raw": root / "raw.csv",
                "manifest": root / "manifest.json",
            }
            runner.atomic_write_json(
                paths["manifest"],
                {
                    "base_run_spec_hash": "spec",
                    "formal_protocol_version": runner.FORMAL_PROTOCOL_VERSION,
                    "calibration_selection": {
                        "policy": runner.CALIBRATION_SELECTION_POLICY,
                        "lcb95_required": True,
                        "mean_only_fallback_allowed": False,
                    },
                },
            )

            rows, manifest = runner.load_checkpoint(paths, "spec", resume=True)

        self.assertEqual(rows, [])
        self.assertEqual(manifest["base_run_spec_hash"], "spec")

    def test_legacy_mean_only_or_unbound_raw_checkpoint_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {"raw": root / "raw.csv", "manifest": root / "manifest.json"}
            runner.atomic_write_json(paths["manifest"], {"base_run_spec_hash": "spec"})
            with self.assertRaisesRegex(runner.CheckpointContractError, "legacy or mean-only"):
                runner.load_checkpoint(paths, "spec", resume=True)

            paths["raw"].write_text("measurement_key\nrow\n", encoding="utf-8")
            runner.atomic_write_json(
                paths["manifest"],
                {
                    "base_run_spec_hash": "spec",
                    "formal_protocol_version": runner.FORMAL_PROTOCOL_VERSION,
                    "calibration_selection": {
                        "policy": runner.CALIBRATION_SELECTION_POLICY,
                        "lcb95_required": True,
                        "mean_only_fallback_allowed": False,
                    },
                },
            )
            with self.assertRaisesRegex(runner.CheckpointContractError, "not hash-bound"):
                runner.load_checkpoint(paths, "spec", resume=True)

    def test_resume_append_only_audit_rejects_mutation(self):
        before = [{"measurement_key": "m1", "latency_ms": "1.0"}]
        after = before + [{"measurement_key": "m2", "latency_ms": 2.0}]

        audit = runner.resume_append_only_audit(before, after)

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["new_measurements"], 1)
        with self.assertRaises(runner.CheckpointContractError):
            runner.resume_append_only_audit(
                before,
                [{"measurement_key": "m1", "latency_ms": "9.0"}],
            )

    def test_dry_run_has_no_file_docker_or_database_access(self):
        argv = ["runner", "--implementation", "official", "--dry-run"]
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(output),
            mock.patch.object(runner.Path, "open", side_effect=AssertionError("file access")),
            mock.patch.object(runner.Path, "exists", side_effect=AssertionError("file access")),
            mock.patch.object(runner.subprocess, "run", side_effect=AssertionError("external command")),
            mock.patch.dict(sys.modules, {"psycopg": None}),
        ):
            runner.main()

        payload = json.loads(output.getvalue())
        bounds = payload["default_query_count_bounds_per_implementation"]
        self.assertEqual(payload["default_effective_config_count"], 77)
        self.assertEqual(payload["formal_family_effective_config_count"], 11)
        self.assertEqual(payload["formal_cell_count"], 42)
        self.assertEqual(bounds["screen_queries"], 3_080)
        self.assertEqual(bounds["verification_query_upper_bound"], 24_640)
        self.assertEqual(bounds["total_query_upper_bound"], 48_720)
        self.assertFalse(payload["file_access"])
        self.assertFalse(payload["docker_access"])
        self.assertFalse(payload["database_access"])


if __name__ == "__main__":
    unittest.main()
