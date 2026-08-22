import argparse
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import amazon10m_sql_native_benchmark as benchmark  # noqa: E402
import amazon10m_sql_native_exact_truth as exact_truth  # noqa: E402


class Amazon10MSqlNativeBenchmarkTests(unittest.TestCase):
    def _binary_identity_gate(self):
        build_id = benchmark.SQLENS_R43_BUILD_ID
        vector_sha256 = benchmark.SQLENS_R43_VECTOR_SO_SHA256
        stages_and_connections = [
            ("experiment_start", "fragment_store"),
            ("connection_open", "data_guard"),
            *(("connection_open", mode) for mode in benchmark.MODES),
            ("pre_exact_truth", "data_guard"),
            ("pre_calibration", "data_guard"),
            ("pre_final", "data_guard"),
            ("manifest_finalization", "data_guard"),
        ]
        evidence = [
            {
                "sequence": sequence,
                "stage": stage,
                "connection": connection,
                "backend_pid": 1000 + sequence,
                "database": "benchmark",
                "expected_sqlens_build_id": build_id,
                "observed_sqlens_build_id": build_id,
                "expected_vector_so_sha256": vector_sha256,
                "observed_vector_so_sha256": vector_sha256,
                "observed_vector_so_path": "/server/lib/vector.so",
                "build_id_exact_match": True,
                "vector_so_sha256_exact_match": True,
                "path_valid": True,
                "observed_sha256_valid": True,
                "exact_match": True,
            }
            for sequence, (stage, connection) in enumerate(
                stages_and_connections, start=1
            )
        ]
        return benchmark.binary_identity_gate_summary(
            build_id, vector_sha256, evidence
        )

    def _external_truth_fixture(self, directory: Path):
        fbin = directory / "vectors.fbin"
        filters_csv = directory / "filters.csv"
        query_ids_csv = directory / "queries.csv"
        query_cohort_manifest = directory / "queries.manifest.json"
        candidate = directory / "candidate.ids"
        truth_csv = directory / "truth.csv"
        manifest_path = directory / "truth.manifest.json"
        fbin.write_bytes(b"fixture-fbin")
        filters_csv.write_text("fixture-filters\n", encoding="utf-8")
        query_ids_csv.write_text("fixture-queries\n", encoding="utf-8")
        query_cohort_manifest.write_text("{}\n", encoding="utf-8")
        candidate.write_text("1\n2\n3\n", encoding="ascii")
        workload = benchmark.WorkloadSpec("acl_only", "fixture", 50.0, False)
        spec = benchmark.FilterSpec("f", "1%", "rating = 5", ("sql:rating = 5",), 3, 1.0)
        query_ids = {0: 10}
        query_splits = {0: "calibration"}
        cohort_hash = benchmark.query_cohort_sha256(query_ids, query_splits)
        validity_predicate = "embedding_valid"
        validity_hash = benchmark.candidate_universe_predicate_sha256(
            validity_predicate
        )
        scalar_hash = benchmark.workload_scalar_predicate_sha256(spec.predicate)
        as_of = {workload.name: 123}
        candidate_hash = benchmark.sha256_file(candidate)
        distance_sq = 4.0
        row = {
            "workload": workload.name, "filter_name": spec.name, "predicate": spec.predicate,
            "workload_scalar_predicate_sha256": scalar_hash,
            "candidate_universe_predicate": validity_predicate,
            "candidate_universe_predicate_sha256": validity_hash,
            "query_no": 0, "query_id": 10, "query_split": "calibration", "k": 1,
            "as_of": 123, "self_excluded": True, "candidate_count": 3,
            "candidate_min_id": 1, "candidate_max_id": 3, "candidate_ids_sha256": candidate_hash,
            "exact_topk_ids": "1", "exact_topk_distances_sq": "4",
            "exact_topk_plus_one_ids": "1,2", "exact_topk_plus_one_distances_sq": "4,9",
            "kth_distance_sq": 4.0, "tie_tolerance": benchmark.distance_tolerance(4.0),
            "strict_closer_count": 0, "boundary_tied": False,
        }
        with truth_csv.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        source_hashes = {
            "script": "0" * 64,
            "filters_csv": benchmark.sha256_file(filters_csv),
            "query_ids_csv": benchmark.sha256_file(query_ids_csv),
            "query_cohort_manifest": benchmark.sha256_file(query_cohort_manifest),
            "fbin": benchmark.sha256_file(fbin),
        }
        run_spec = {
            "vector_table": "t", "principal": "principal", "k": 1,
            "calibration_queries": 1, "final_queries": 0,
            "filters": [{"name": spec.name, "target_rate": spec.target_rate, "predicate": spec.predicate,
                         "expected_rows": spec.expected_rows, "actual_pct": spec.actual_pct}],
            "workloads": [{"name": workload.name, "bucket_pct": workload.bucket_pct, "temporal_kind": "none"}],
            "query_ids": {"0": 10}, "query_splits": {"0": "calibration"},
            "query_cohort_sha256": cohort_hash,
            "query_cohort_hash_contract": exact_truth.QUERY_COHORT_HASH_CONTRACT,
            "query_cohort": {
                "source_csv_sha256": source_hashes["query_ids_csv"],
                "source_manifest": {
                    "path": str(query_cohort_manifest),
                    "sha256": source_hashes["query_cohort_manifest"],
                },
                "query_count": 1,
                "query_cohort_sha256": cohort_hash,
                "query_cohort_hash_contract": exact_truth.QUERY_COHORT_HASH_CONTRACT,
            },
            "candidate_universe": {
                "predicate": validity_predicate,
                "predicate_sha256": validity_hash,
                "sql_role": "candidate_relation_only; separate from workload scalar predicate",
            },
            "workload_scalar_predicates": [{
                "filter_name": spec.name,
                "predicate": spec.predicate,
                "predicate_sha256": scalar_hash,
            }],
            "source_hashes": source_hashes,
        }
        backend = {
            "backend": "faiss", "class": "IndexFlatL2", "faiss_version": "1.14.2",
            "threads": 1, "exact": True, "formal_default": True,
        }
        base_sample_ids = [0, 10]
        checked_ids = [0, 10]
        mapping = {
            "base_sample_size_requested": 2,
            "base_sample_ids": base_sample_ids,
            "base_sample_ids_sha256": benchmark.canonical_sha256(base_sample_ids),
            "query_ids_included": [10],
            "checked_ids": checked_ids,
            "checked_ids_sha256": benchmark.canonical_sha256(checked_ids),
            "checked_rows": 2,
            "comparison": "float32_allclose",
            "rtol": 1e-6,
            "atol": 1e-7,
            "max_abs_error": 0.0,
        }
        run_spec["backend"] = backend
        run_spec["base_table_mapping"] = mapping
        exact_workload = exact_truth.WorkloadSpec(
            workload.name, workload.description, workload.bucket_pct, "none"
        )
        candidate_sql = exact_truth.build_candidate_sql(
            "t", spec.predicate, exact_workload, validity_predicate
        )
        spot_sql = exact_truth.build_spot_check_sql(
            "t", spec.predicate, exact_workload, validity_predicate
        )
        trigger = [[
            "amazon_sql_native_epoch_bump",
            "O",
            "public.amazon_sql_native_bump_relation_epoch()",
            "INSERT INTO public.amazon_sql_native_relation_epoch VALUES (1); "
            "ON CONFLICT DO UPDATE SET epoch = epoch + 1",
            "CREATE TRIGGER amazon_sql_native_epoch_bump AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE "
            "ON t FOR EACH STATEMENT EXECUTE FUNCTION amazon_sql_native_bump_relation_epoch()",
        ]]
        relation_data = {
            "t": {"oid": 1, "data_epoch": 11, "policies": [], "triggers": trigger},
            "public.amazon_review_facts": {"oid": 2, "data_epoch": 12, "policies": [], "triggers": trigger},
            "public.amazon_product_dim": {"oid": 3, "data_epoch": 13, "triggers": trigger},
            "public.amazon_principal_tenant_grants": {"oid": 4, "data_epoch": 14, "triggers": trigger},
            "public.amazon_sql_native_buckets": {"oid": 5, "data_epoch": 15, "triggers": trigger},
        }
        pair = {
            "workload": workload.name, "filter": {"name": spec.name}, "as_of": 123,
            "workload_scalar_predicate": spec.predicate,
            "workload_scalar_predicate_sha256": scalar_hash,
            "candidate_universe_predicate": validity_predicate,
            "candidate_universe_predicate_sha256": validity_hash,
            "session": {"current_user": "principal"},
            "relations": relation_data, "candidate": {"count": 3, "min_id": 1, "max_id": 3,
                                                        "sha256": candidate_hash, "path": str(candidate)},
            "candidate_sql": candidate_sql, "candidate_sql_sha256": hashlib.sha256(candidate_sql.encode()).hexdigest(),
            "candidate_explain_gate": {"valid": True, "index_names": []},
            "spot_check_sql": spot_sql, "spot_check_sql_sha256": hashlib.sha256(spot_sql.encode()).hexdigest(),
            "spot_check_explain_gate": {"valid": True, "index_names": []},
            "exact_backend": {
                "backend": "faiss", "class": "IndexFlatL2", "faiss_version": "1.14.2",
                "threads": 1, "index_ntotal": 3, "index_add_ms": 1.0,
                "search_ms": 2.0, "elapsed_ms": 3.0, "search_calls": 1,
                "local_positions_mapped_to_global_ids": True,
                "order": "squared_l2_then_global_id", "exact": True,
            },
            "spot_checks": [{"valid": True, "query_no": 0, "query_id": 10, "limit": 2,
                             "sql_ids": [1, 2], "sql_distances": [4.0, 9.0]}],
        }
        manifest = {
            "artifact_valid": True, "artifact": "amazon10m_sql_native_exact_truth", "version": 4,
            "run_spec": run_spec, "run_spec_hash": benchmark.canonical_sha256(run_spec),
            "query_cohort": run_spec["query_cohort"],
            "query_cohort_sha256": cohort_hash,
            "candidate_universe": run_spec["candidate_universe"],
            "candidate_universe_predicate_sha256": validity_hash,
            "relation_epoch": benchmark.relation_epoch_contract(relation_data),
            "source_hashes": source_hashes,
            "fbin": {"path": str(fbin)},
            "backend": backend, "base_table_mapping": mapping,
            "outputs": {"truth_csv_sha256": benchmark.sha256_file(truth_csv)}, "pairs": [pair],
            "data_version_proof": {
                "valid": True,
                "start_relations": relation_data,
                "end_relations": relation_data,
                "start_hash": benchmark.canonical_sha256(relation_data),
                "end_hash": benchmark.canonical_sha256(relation_data),
            },
            "rls_security_proof": {
                "current_user": "principal", "is_superuser": False,
                "bypass_rls": False, "owns_facts": False,
                "reader_membership": True, "rls_enabled": True,
                "policy_hash": benchmark.canonical_sha256([]),
                "positive_probe_visible": True, "negative_probe_hidden": True,
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return {
            "fbin": fbin, "filters_csv": filters_csv, "query_ids_csv": query_ids_csv,
            "query_cohort_manifest": query_cohort_manifest,
            "truth_csv": truth_csv, "manifest_path": manifest_path, "manifest": manifest,
            "workload": workload, "spec": spec, "query_ids": query_ids, "query_splits": query_splits,
            "as_of": as_of, "relations": relation_data,
        }

    def _load_external_fixture(self, fixture):
        return benchmark.load_external_exact_truth(
            fixture["truth_csv"], fixture["manifest_path"], fixture["fbin"], fixture["filters_csv"],
            fixture["query_ids_csv"], [fixture["workload"]], [fixture["spec"]], fixture["query_ids"],
            fixture["query_splits"], fixture["as_of"], "t", "principal", 1, fixture["relations"],
            require_formal_keyspace=False,
            query_cohort_manifest=fixture["query_cohort_manifest"],
        )

    def test_defaults_are_formal_disjoint_q100_q100_and_r2_r5(self):
        args = benchmark.create_argument_parser().parse_args([])
        self.assertEqual(args.calibration_query_offset, 20)
        self.assertEqual(args.calibration_queries, 80)
        self.assertEqual(args.final_queries, 100)
        self.assertEqual(args.calibration_repeats, 2)
        self.assertEqual(args.final_repeats, 5)
        self.assertEqual(args.targets, [0.90, 0.95, 0.99])
        self.assertEqual(args.bootstrap_samples, 10_000)
        self.assertEqual(
            args.ef_search_values,
            [20, 40, 60, 80, 100, 150, 200, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
        )
        self.assertIsNone(args.expected_sqlens_build_id)
        self.assertIsNone(args.expected_vector_so_sha256)
        self.assertEqual(args.max_scan_tuples_values, [5_000_000])
        self.assertEqual(args.scan_mem_multiplier_values, [32.0])
        self.assertEqual(tuple(benchmark.MODES), ("stock", "d1", "d1_d2", "d1_d2_d3"))
        self.assertEqual(args.source_index, benchmark.DEFAULT_SOURCE_INDEX)
        self.assertEqual(args.clone_index, benchmark.DEFAULT_CLONE_INDEX)
        self.assertNotEqual(args.source_index, args.clone_index)
        self.assertEqual(args.execution_engine, "sqlens")
        self.assertEqual(args.official_planner_mode, "auto")
        self.assertEqual(args.d3_probe_requests, 2)
        self.assertEqual(args.exact_truth_csv, benchmark.DEFAULT_EXACT_TRUTH_CSV)
        self.assertEqual(args.exact_truth_manifest, benchmark.DEFAULT_EXACT_TRUTH_MANIFEST)
        self.assertFalse(args.debug_compute_exact_truth)
        self.assertEqual(args.candidate_validity_predicate, "embedding_valid")
        self.assertIn("unique_embeddings_formal", args.query_ids_csv.name)
        filters = benchmark.read_filters(
            ROOT / "configs" / "amazon10m_selectivity14_filters.csv"
        )
        benchmark.validate_formal_dimensions(args, filters)
        args.final_repeats = 4
        with self.assertRaisesRegex(RuntimeError, "final r5"):
            benchmark.validate_formal_dimensions(args, filters)

    def test_q10200_protocol_resolves_p0_dimensions_arms_and_r43_identity(self):
        args = benchmark.resolve_protocol_args(
            benchmark.create_argument_parser().parse_args(
                ["--protocol", "q10200"]
            )
        )
        source_filters = benchmark.read_filters(benchmark.DEFAULT_FILTERS)
        filters = benchmark.read_filters(
            benchmark.DEFAULT_FILTERS, set(args.filter_names)
        )
        workloads = benchmark.select_workloads(
            args.workload_names, args.protocol
        )
        benchmark.validate_formal_dimensions(
            args, filters, workloads, source_filters
        )
        self.assertEqual(args.calibration_query_offset, 20)
        self.assertEqual(args.calibration_queries, 80)
        self.assertEqual(args.calibration_repeats, 2)
        self.assertEqual(args.final_query_offset, 200)
        self.assertEqual(args.final_queries, 10_000)
        self.assertEqual(args.final_repeats, 3)
        self.assertEqual(args.targets, [0.90])
        self.assertEqual(benchmark.P0_MODES, (
            "stock",
            "sql_first_forced_indexed_exact",
            "d1_d2_d3",
        ))
        self.assertEqual(
            args.expected_sqlens_build_id, benchmark.SQLENS_R43_BUILD_ID
        )
        self.assertEqual(
            args.expected_vector_so_sha256,
            benchmark.SQLENS_R43_VECTOR_SO_SHA256,
        )
        self.assertEqual(
            args.exact_truth_csv.name,
            "amazon10m_sql_native_exact_truth_q10200.csv",
        )
        self.assertEqual(
            args.exact_truth_csv.parent.name,
            "amazon10m_sql_native_q10200_r43_sqlops_join",
        )
        self.assertEqual(
            args.exact_truth_manifest.name,
            "amazon10m_sql_native_exact_truth_manifest.json",
        )
        self.assertEqual(
            args.workload_names,
            list(benchmark.P0_WORKLOAD_NAMES),
        )
        self.assertEqual(
            args.filter_names,
            list(benchmark.P0_FILTER_NAMES),
        )
        self.assertGreaterEqual(min(args.ef_search_values), benchmark.P0_MIN_EF_SEARCH)
        self.assertNotIn("popular_ge1000", args.filter_names)
        self.assertNotIn("price_10_to_20", args.filter_names)
        self.assertNotIn("acl_only", args.workload_names)
        self.assertNotIn("boolean_complex_narrow_exists", args.workload_names)
        self.assertNotIn("fact_temporal_selectivity", args.workload_names)
        self.assertEqual(args.workload_names, ["join_facts", "join_catalog", "join_acl"])

    def test_q10200_rejects_legacy_acl_popular_matrix(self):
        args = benchmark.resolve_protocol_args(
            benchmark.create_argument_parser().parse_args(
                [
                    "--protocol",
                    "q10200",
                    "--workload-names",
                    "acl_only",
                    "grant_temporal_selectivity",
                    "fact_temporal_selectivity",
                    "--filter-names",
                    "popular_ge1000",
                    "price_10_to_20",
                    "rating5_price_le10",
                    "helpful_ge20",
                ]
            )
        )
        source_filters = benchmark.read_filters(benchmark.DEFAULT_FILTERS)
        filters = benchmark.read_filters(
            benchmark.DEFAULT_FILTERS, set(args.filter_names)
        )
        workloads = benchmark.select_workloads(
            args.workload_names, args.protocol
        )
        with self.assertRaisesRegex(RuntimeError, "facts-join, catalog-join, and ACL-join"):
            benchmark.validate_formal_dimensions(
                args, filters, workloads, source_filters
            )

    def test_q10200_confirmation_resolves_q2k_r1_and_parallel_sql_first(self):
        args = benchmark.resolve_protocol_args(
            benchmark.create_argument_parser().parse_args(
                ["--protocol", "q10200", "--confirmation"]
            )
        )
        source_filters = benchmark.read_filters(benchmark.DEFAULT_FILTERS)
        filters = benchmark.read_filters(
            benchmark.DEFAULT_FILTERS, set(args.filter_names)
        )
        workloads = benchmark.select_workloads(
            args.workload_names, args.protocol
        )
        benchmark.validate_formal_dimensions(
            args, filters, workloads, source_filters
        )
        self.assertTrue(args.confirmation)
        self.assertEqual(args.final_query_offset, 200)
        self.assertEqual(args.final_queries, 2_000)
        self.assertEqual(args.final_repeats, 1)
        self.assertEqual(
            args.sql_first_workers, benchmark.P0_CONFIRMATION_SQL_FIRST_WORKERS
        )
        self.assertEqual(
            args.out.name,
            "amazon10m_sql_native_p0_r43_q2k_r1_sqlops_join_confirm.csv",
        )
        self.assertEqual(
            args.workload_names,
            [
                "join_facts",
                "join_catalog",
                "join_acl",
            ],
        )
        self.assertEqual(
            args.filter_names,
            ["grocery_helpful", "helpful_ge20", "grocery_long500"],
        )
        self.assertNotIn("popular_ge1000", args.filter_names)
        self.assertGreaterEqual(min(args.ef_search_values), benchmark.P0_MIN_EF_SEARCH)

    def test_q10200_screening_resolves_q1k_r1(self):
        args = benchmark.resolve_protocol_args(
            benchmark.create_argument_parser().parse_args(
                ["--protocol", "q10200", "--screening"]
            )
        )
        source_filters = benchmark.read_filters(benchmark.DEFAULT_FILTERS)
        filters = benchmark.read_filters(
            benchmark.DEFAULT_FILTERS, set(args.filter_names)
        )
        workloads = benchmark.select_workloads(
            args.workload_names, args.protocol
        )
        benchmark.validate_formal_dimensions(
            args, filters, workloads, source_filters
        )
        self.assertTrue(args.screening)
        self.assertFalse(args.confirmation)
        self.assertEqual(args.final_queries, 1_000)
        self.assertEqual(args.final_repeats, 1)
        self.assertEqual(
            args.out.name,
            "amazon10m_sql_native_p0_r43_q1k_r1_sqlops_join_screen.csv",
        )

    def test_p0_sql_exposes_facts_catalog_and_acl_join_depths(self):
        table = "public.amazon_grocery_reviews_10m_pgvector"
        predicate = "v.rating = 5 AND v.has_price AND v.price <= 10"
        by_name = {item.name: item for item in benchmark.WORKLOADS}
        facts_sql = benchmark.build_hybrid_sql(
            table, predicate, workload=by_name["join_facts"]
        )
        catalog_sql = benchmark.build_hybrid_sql(
            table, predicate, workload=by_name["join_catalog"]
        )
        acl_sql = benchmark.build_hybrid_sql(
            table, predicate, workload=by_name["join_acl"]
        )
        for sql_text in (facts_sql, catalog_sql, acl_sql):
            self.assertIn("JOIN public.amazon_review_facts", sql_text)
            self.assertIn("ORDER BY", sql_text)
        self.assertNotIn("JOIN public.amazon_product_dim", facts_sql)
        self.assertNotIn("CURRENT_USER", facts_sql)
        self.assertIn("JOIN public.amazon_product_dim", catalog_sql)
        self.assertNotIn("CURRENT_USER", catalog_sql)
        self.assertIn("JOIN public.amazon_principal_tenant_grants", acl_sql)
        self.assertIn("CURRENT_USER", acl_sql)

    def test_attributes_shape_is_where_only_hybrid_search(self):
        table = "public.amazon_grocery_reviews_10m_pgvector"
        predicate = "v.main_category = 'Grocery' AND v.helpful_vote >= 1"
        attributes = benchmark.WorkloadSpec(
            "attributes",
            "row-local WHERE hybrid search",
            50.0,
            False,
            "base",
            "",
            "none",
            "none",
        )
        sql_text = benchmark.build_hybrid_sql(
            table, predicate, workload=attributes
        )
        self.assertIn("ORDER BY", sql_text)
        self.assertIn("LIMIT", sql_text)
        self.assertNotIn("JOIN public.amazon_review_facts", sql_text)
        self.assertNotIn("CURRENT_USER", sql_text)
        candidate = exact_truth.build_candidate_sql(table, predicate, attributes)
        self.assertIn("ORDER BY v.id", candidate)
        self.assertNotIn("<->", candidate)
        self.assertNotIn("JOIN public.amazon_review_facts", candidate)

    def test_binding_atoms_stay_filter_local_for_join_workloads(self):
        filters = benchmark.read_filters(
            benchmark.DEFAULT_FILTERS, {"helpful_ge20"}
        )
        by_name = {item.name: item for item in benchmark.WORKLOADS}
        for name in ("join_facts", "join_catalog", "join_acl"):
            self.assertEqual(
                benchmark.binding_atoms_for(by_name[name], filters[0]),
                filters[0].atoms,
            )

    def test_partition_measurement_schedule_keeps_sqlens_sequential(self):
        keys = [("acl_only", "popular_ge1000", 200, 0)]
        schedule = benchmark.interleaved_schedule(
            keys, benchmark.P0_MODES, seed=7
        )
        sequential, parallel = benchmark.partition_measurement_schedule(
            schedule, benchmark.SQL_FIRST_MODE
        )
        self.assertEqual(len(schedule), 3)
        self.assertEqual(len(parallel), 1)
        self.assertEqual(len(sequential), 2)
        self.assertEqual(parallel[0][2], benchmark.SQL_FIRST_MODE)
        self.assertNotIn(
            benchmark.SQL_FIRST_MODE, [mode for _, _, mode in sequential]
        )
        self.assertEqual(
            {item[0] for item in sequential + parallel}, {0, 1, 2}
        )

    def test_run_spec_integrity_hash_survives_json_integer_key_roundtrip(self):
        run_spec = {
            "k": 10,
            "query_ids": {0: 10, 2: 20, 10: 30},
            "query_splits": {0: "calibration", 2: "calibration", 10: "final"},
        }
        stored = benchmark.canonical_sha256(run_spec)
        loaded = json.loads(json.dumps(run_spec))
        self.assertNotEqual(stored, benchmark.canonical_sha256(loaded))
        self.assertEqual(
            stored,
            benchmark.canonical_sha256(
                benchmark._run_spec_for_integrity_hash(loaded)
            ),
        )

    def test_tie_metadata_recompute_uses_producer_float32_precision(self):
        # Real Amazon-10M edge row (acl_only/popular_ge1000/q5677): two of the
        # top-k distances sit within one float32 ULP of ``kth - tolerance``.
        # The producer counts them with faiss float32 semantics (0 strict), so
        # a naive float64 recompute (2 strict) would wrongly reject the CSV.
        plus_one = [
            0.118575215, 0.118575215, 0.118575335, 0.118575335, 0.118575335,
            0.118575335, 0.118575335, 0.118575335, 0.118575335, 0.118575335,
            0.118575335,
        ]
        k = 10
        kth = plus_one[k - 1]
        tolerance = benchmark.distance_tolerance(kth)
        naive_strict = sum(value < kth - tolerance for value in plus_one[:k])
        self.assertEqual(naive_strict, 2)
        recomputed = exact_truth.truth_metadata(
            benchmark._np.asarray(plus_one, dtype=benchmark._np.float32), k
        )
        self.assertEqual(recomputed["strict_closer_count"], 0)

    def test_external_truth_loads_squared_l2_as_l2_and_binds_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._external_truth_fixture(Path(tmp))
            truth, provenance = self._load_external_fixture(fixture)
            entry = truth[("acl_only", "f", 0)]
            self.assertEqual(entry.ids, (1,))
            self.assertEqual(entry.kth_distance, 2.0)
            self.assertEqual(entry.tie_tolerance, benchmark.distance_tolerance(2.0))
            self.assertNotEqual(entry.tie_tolerance, benchmark.distance_tolerance(4.0))
            self.assertEqual(provenance["truth_csv_sha256"], benchmark.sha256_file(fixture["truth_csv"]))

    def test_external_truth_rejects_stale_malformed_incomplete_duplicate_and_plan_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._external_truth_fixture(Path(tmp))
            mutations = {
                "artifact_valid": lambda data: data["manifest"].update({"artifact_valid": False}),
                "source": lambda data: data["manifest"]["source_hashes"].update({"fbin": "f" * 64}),
                "spot": lambda data: data["manifest"]["pairs"][0]["spot_checks"][0].update({"valid": False}),
                "plan": lambda data: data["manifest"]["pairs"][0]["candidate_explain_gate"].update({"index_names": ["hnsw_bad"]}),
                "candidate": lambda data: data["manifest"]["pairs"][0]["candidate"].update({"sha256": "a" * 64}),
                "backend": lambda data: data["manifest"]["pairs"][0]["exact_backend"].update({"exact": False}),
                "mapping": lambda data: data["manifest"]["base_table_mapping"].update({"checked_rows": 1}),
                "data_version": lambda data: data["manifest"]["data_version_proof"].update({"end_hash": "d" * 64}),
                "candidate_universe_manifest": lambda data: data["manifest"].update(
                    {"candidate_universe_predicate_sha256": "f" * 64}
                ),
                "query_cohort_manifest": lambda data: data["manifest"].update(
                    {"query_cohort_sha256": "e" * 64}
                ),
                "relation_epoch_manifest": lambda data: data["manifest"]["relation_epoch"].update(
                    {"sha256": "d" * 64}
                ),
                "rls_probe": lambda data: data["manifest"]["rls_security_proof"].update({"negative_probe_hidden": False}),
                "stale": lambda data: data["truth_csv"].write_text(data["truth_csv"].read_text(encoding="utf-8").replace(",4.0,", ",5.0,"), encoding="utf-8"),
                "incomplete": lambda data: data["truth_csv"].write_text(data["truth_csv"].read_text(encoding="utf-8").split("\n", 1)[0] + "\n", encoding="utf-8"),
                "duplicate": lambda data: data["truth_csv"].write_text(data["truth_csv"].read_text(encoding="utf-8") + data["truth_csv"].read_text(encoding="utf-8").split("\n", 1)[1], encoding="utf-8"),
                "schema": lambda data: data["truth_csv"].write_text("workload,filter_name\nacl_only,f\n", encoding="utf-8"),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    case_dir = Path(tmp) / name
                    case_dir.mkdir()
                    current = self._external_truth_fixture(case_dir)
                    mutate(current)
                    current["manifest"]["outputs"]["truth_csv_sha256"] = benchmark.sha256_file(current["truth_csv"])
                    current["manifest_path"].write_text(json.dumps(current["manifest"]), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "exact-truth artifact rejected"):
                        self._load_external_fixture(current)

    def test_external_truth_rejects_query_cohort_mismatch_before_loading_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._external_truth_fixture(Path(tmp))
            fixture["query_ids"] = {0: 11}
            with self.assertRaisesRegex(RuntimeError, "query cohort"):
                self._load_external_fixture(fixture)

    def test_external_truth_explicitly_rejects_legacy_artifact_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._external_truth_fixture(Path(tmp))
            fixture["manifest"]["version"] = 3
            fixture["manifest_path"].write_text(
                json.dumps(fixture["manifest"]), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "legacy/incompatible.*observed=3"):
                self._load_external_fixture(fixture)

    def test_external_truth_rejects_manifest_and_record_checkpoint_staleness(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._external_truth_fixture(Path(tmp))
            truth, provenance = self._load_external_fixture(fixture)
            self.assertEqual(len(truth), 1)
            args = benchmark.create_argument_parser().parse_args([])
            run_spec = benchmark.build_run_spec(
                args, [], [], {}, {}, {"relations": {}}, {}, provenance
            )
            checkpoint = {
                "checkpoint_version": benchmark.CHECKPOINT_VERSION,
                "run_spec": run_spec, "run_spec_sha256": benchmark.canonical_sha256(run_spec),
            }
            benchmark.validate_checkpoint_run_spec(checkpoint, run_spec)
            stale = {**provenance, "records_sha256": "0" * 64}
            with self.assertRaisesRegex(RuntimeError, "run-spec mismatch"):
                benchmark.validate_checkpoint_run_spec(checkpoint, benchmark.build_run_spec(
                    args, [], [], {}, {}, {"relations": {}}, {}, stale
                ))

    def test_query_split_loader_rejects_overlap_and_duplicate_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.csv"
            with path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=[
                        "query_no",
                        "query_id",
                        "method",
                        "query_split",
                        "self_excluded",
                        "kth_distance_sq",
                        "candidate_validity_predicate",
                        "query_validity_predicate",
                    ],
                )
                writer.writeheader()
                for query_no, query_id in [(0, 10), (1, 11), (2, 12)]:
                    writer.writerow(
                        {
                            "query_no": query_no,
                            "query_id": query_id,
                            "method": "pre_filter_exact",
                            "query_split": "calibration" if query_no < 2 else "final",
                            "self_excluded": True,
                            "kth_distance_sq": 1.0,
                            "candidate_validity_predicate": "embedding_valid",
                            "query_validity_predicate": "embedding_valid",
                        }
                    )
            calibration = benchmark.load_query_ids(path, 0, 2)
            final = benchmark.load_query_ids(path, 2, 1)
            benchmark.validate_query_splits(calibration, final)
            with self.assertRaisesRegex(ValueError, "query IDs"):
                benchmark.validate_query_splits({0: 10}, {1: 10})

    def test_approx_and_exact_sql_have_the_required_ordering_boundaries(self):
        sql_text = benchmark.build_hybrid_sql("public.amazon_grocery_reviews_10m_pgvector", "rating = 5")
        exact_text = benchmark.build_hybrid_sql("public.amazon_grocery_reviews_10m_pgvector", "rating = 5", exact=True)
        normalized = " ".join(sql_text.lower().split())
        self.assertIn(
            "order by v.embedding <-> %(query_embedding)s::vector",
            normalized,
        )
        self.assertIn("v.id <> %(query_id)s", normalized)
        self.assertIn("v.embedding_valid", normalized)
        self.assertNotIn("query_vector", normalized)
        self.assertIn("vector_hnsw_guidance_bind", normalized)
        self.assertIn("%(binding_kind)s", sql_text)
        self.assertNotIn("vector_hnsw_guidance_bind", exact_text.lower())
        self.assertNotIn("hnsw", exact_text.lower())
        benchmark.validate_exact_sql_text(exact_text)
        with self.assertRaisesRegex(RuntimeError, "guidance/HNSW"):
            benchmark.validate_exact_sql_text(exact_text + " SELECT vector_hnsw_guidance_bind()")
        self.assertNotIn("order by v.embedding <-> %(query_embedding)s::vector, v.id", normalized)
        self.assertIn("valid as materialized", exact_text.lower())
        exact_normalized = " ".join(exact_text.lower().split())
        self.assertIn(
            "join public.amazon_principal_tenant_grants as grant_row "
            "on grant_row.tenant_id = product.tenant_id cross join query_vector where",
            exact_normalized,
        )
        self.assertIn("order by distance, valid.id", exact_text.lower())
        self.assertNotIn("enable_indexscan = off", exact_text.lower())
        self.assertIn("join public.amazon_review_facts", normalized)
        self.assertIn("join public.amazon_product_dim", normalized)
        self.assertIn("join public.amazon_principal_tenant_grants", normalized)
        self.assertIn("current_user", normalized)
        self.assertIn("fact.valid_from <= %(as_of)s", normalized)
        self.assertIn("v.rating = 5", normalized)
        self.assertIn("v.embedding_valid", exact_normalized)
        self.assertEqual(normalized.count("select"), 2)

    def test_candidate_validity_predicate_rejects_nonformal_expressions(self):
        for predicate in (
            "embedding_valid; DROP TABLE x",
            "embedding_valid -- comment",
            "embedding_valid /* comment */",
            "embedding_valid) OR true OR (embedding_valid",
        ):
            with self.subTest(predicate=predicate):
                with self.assertRaisesRegex(argparse.ArgumentTypeError, "must be exactly"):
                    benchmark.validate_candidate_validity_predicate(predicate)

    def test_sqlens_provenance_gate_rejects_old_or_incomplete_profiles(self):
        profile = {
            "profile_semantics_version": 9,
            "graph_elements_visited": 0,
            "raw_index_tids_returned": 0,
            "hnsw_am_callback_ms": 0.0,
            "executor_residual_ms": 0.0,
        }
        profile.update({field: 0 for field in benchmark.SQLENS_PROFILE_FIELDS if field not in profile})
        build_id, validated = benchmark.validate_sqlens_provenance(
            benchmark.SQLENS_R43_BUILD_ID, profile
        )
        self.assertEqual(build_id, benchmark.SQLENS_R43_BUILD_ID)
        self.assertEqual(validated, profile)

        with self.assertRaisesRegex(RuntimeError, "expected exact r43"):
            benchmark.validate_sqlens_provenance("sqlens-v10-old", profile)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            benchmark.validate_sqlens_provenance(
                benchmark.SQLENS_R43_BUILD_ID,
                {"profile_semantics_version": 9},
            )
        with self.assertRaisesRegex(RuntimeError, "minimum=9"):
            benchmark.validate_sqlens_provenance(
                benchmark.SQLENS_R43_BUILD_ID,
                profile | {"profile_semantics_version": 8},
            )

    def test_formal_execute_requires_exact_binary_identity_cli(self):
        incomplete = (
            ["--execute"],
            [
                "--execute",
                "--expected-sqlens-build-id",
                benchmark.SQLENS_R43_BUILD_ID,
            ],
            [
                "--execute",
                "--expected-vector-so-sha256",
                "a" * 64,
            ],
        )
        for argv in incomplete:
            with self.subTest(argv=argv):
                with mock.patch.object(benchmark, "run_benchmark") as run:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "requires --expected-sqlens-build-id and --expected-vector-so-sha256",
                    ):
                        benchmark.main(argv)
                    run.assert_not_called()

        with mock.patch.object(benchmark, "run_benchmark", return_value=7) as run:
            status = benchmark.main(
                [
                    "--execute",
                    "--expected-sqlens-build-id",
                    benchmark.SQLENS_R43_BUILD_ID,
                    "--expected-vector-so-sha256",
                    benchmark.SQLENS_R43_VECTOR_SO_SHA256,
                ]
            )
        self.assertEqual(status, 7)
        run.assert_called_once()

        for value in ("", "A" * 64, "a" * 63, "g" * 64):
            with self.subTest(sha256=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    benchmark.expected_sha256_arg(value)
        for value in (
            "",
            "sqlens-v11-old",
            " sqlens-v14-profiled-target-bounded-prioritization-build",
        ):
            with self.subTest(build_id=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    benchmark.expected_sqlens_build_id_arg(value)

    def test_serving_binary_identity_gate_reads_server_and_requires_exact_match(self):
        build_id = benchmark.SQLENS_BUILD_PREFIX + "-test-build"
        vector_sha256 = "a" * 64
        cursor = mock.MagicMock()
        cursor.fetchone.return_value = (
            build_id,
            "/server/lib/vector.so",
            vector_sha256,
            1234,
            "benchmark",
        )
        evidence = benchmark.verify_serving_binary_identity(
            cursor,
            build_id,
            vector_sha256,
            stage="experiment_start",
            connection="stock",
        )
        self.assertTrue(evidence["exact_match"])
        self.assertEqual(evidence["backend_pid"], 1234)
        self.assertIn("pg_read_binary_file(path)", cursor.execute.call_args.args[0])
        self.assertIn("vector_sqlens_build_id()", cursor.execute.call_args.args[0])

        mismatches = (
            (
                benchmark.SQLENS_BUILD_PREFIX + "-other",
                "/server/lib/vector.so",
                vector_sha256,
            ),
            (build_id, "/server/lib/vector.so", "b" * 64),
            (build_id, "/server/lib/not-vector", vector_sha256),
        )
        for observed_build, path, observed_sha in mismatches:
            with self.subTest(
                observed_build=observed_build,
                path=path,
                observed_sha=observed_sha,
            ):
                cursor.fetchone.return_value = (
                    observed_build,
                    path,
                    observed_sha,
                    1234,
                    "benchmark",
                )
                with self.assertRaisesRegex(RuntimeError, "binary identity gate failed"):
                    benchmark.verify_serving_binary_identity(
                        cursor,
                        build_id,
                        vector_sha256,
                        stage="pre_calibration",
                        connection="data_guard",
                    )

        cursor.execute.side_effect = PermissionError("denied")
        with self.assertRaisesRegex(RuntimeError, "gate unavailable"):
            benchmark.verify_serving_binary_identity(
                cursor,
                build_id,
                vector_sha256,
                stage="pre_final",
                connection="data_guard",
            )

    def test_binary_identity_summary_fails_closed_without_all_stage_evidence(self):
        valid = self._binary_identity_gate()
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["observed_sqlens_build_id"], valid["expected_sqlens_build_id"])
        self.assertEqual(
            valid["observed_vector_so_sha256"],
            valid["expected_vector_so_sha256"],
        )
        missing_finalization = benchmark.binary_identity_gate_summary(
            valid["expected_sqlens_build_id"],
            valid["expected_vector_so_sha256"],
            [
                item
                for item in valid["gate_evidence"]
                if item["stage"] != "manifest_finalization"
            ],
        )
        self.assertFalse(missing_finalization["valid"])
        self.assertEqual(
            missing_finalization["missing_required_stages"],
            ["manifest_finalization"],
        )
        forged = dict(valid)
        forged_evidence = [dict(item) for item in valid["gate_evidence"]]
        forged_evidence[0]["observed_vector_so_sha256"] = "b" * 64
        forged["gate_evidence"] = forged_evidence
        recomputed = benchmark.binary_identity_gate_summary(
            forged["expected_sqlens_build_id"],
            forged["expected_vector_so_sha256"],
            forged_evidence,
        )
        self.assertFalse(recomputed["valid"])
        self.assertFalse(recomputed["all_exact_match"])
        self.assertNotEqual(recomputed, forged)

    def test_explain_gate_requires_named_hnsw_for_approx_and_forbids_hnsw_for_exact(self):
        hnsw_plan = [{"Plan": {"Node Type": "Index Scan", "Index Name": "vector_hnsw_idx"}}]
        btree_plan = [{"Plan": {"Node Type": "Index Scan", "Index Name": "facts_parent_time_idx"}}]
        approx = benchmark.validate_explain_gate(hnsw_plan, "public.vector_hnsw_idx", require_hnsw=True)
        exact = benchmark.validate_explain_gate(btree_plan, "public.vector_hnsw_idx", require_hnsw=False)
        self.assertTrue(approx["valid"])
        self.assertTrue(exact["valid"])
        with self.assertRaisesRegex(RuntimeError, "EXPLAIN gate failed"):
            benchmark.validate_explain_gate(btree_plan, "public.vector_hnsw_idx", require_hnsw=True)
        with self.assertRaisesRegex(RuntimeError, "EXPLAIN gate failed"):
            benchmark.validate_explain_gate(hnsw_plan, "public.vector_hnsw_idx", require_hnsw=False)

    def test_exact_query_keeps_index_scans_available(self):
        class Cursor:
            def __init__(self):
                self.calls = []

            def execute(self, sql, params=None):
                self.calls.append(str(sql))

            def fetchall(self):
                return [(1, 0.25)]

        cursor = Cursor()
        self.assertEqual(benchmark.query_rows(cursor, "SELECT 1", {}, exact=True), [1])
        self.assertIn("SET LOCAL enable_indexscan = on", cursor.calls)
        self.assertIn("SET LOCAL enable_bitmapscan = on", cursor.calls)
        self.assertNotIn("SET LOCAL enable_indexscan = off", cursor.calls)
        self.assertIn("COMMIT", cursor.calls)

    def test_predicate_qualification_does_not_corrupt_compound_column_names(self):
        self.assertEqual(
            benchmark.qualify_predicate("item_rating_number >= 1000 AND rating = 5"),
            "v.item_rating_number >= 1000 AND v.rating = 5",
        )

    def test_tie_aware_recall_accepts_different_exact_boundary_id(self):
        truth = benchmark.ExactTruth((1, 2), 1.0, 1e-9, True)
        self.assertEqual(
            benchmark.tie_aware_recall_at_k([(1, 0.0), (3, 1.0)], truth, 0, 2),
            1.0,
        )
        self.assertEqual(
            benchmark.tie_aware_recall_at_k([(1, 0.0), (4, 2.0)], truth, 0, 2),
            0.5,
        )

    def test_artifact_gate_rejects_missing_or_failed_held_out_comparisons(self):
        self.assertEqual(benchmark.artifact_validation_errors(0, []), [])
        paired = [
            {
                "phase": "final",
                "mode": f"paired_{mode}",
                "workload": "join_acl",
                "filter_name": "f",
                "target_recall": 0.95,
                "status": "complete",
            }
            for mode in benchmark.SQLENS_MODES
        ]
        failed = [dict(row) for row in paired]
        failed[0]["status"] = benchmark.NA
        errors = benchmark.artifact_validation_errors(1, failed)
        self.assertEqual(len(errors), 1)
        self.assertIn("held-out matched-recall validation failed", errors[0])
        self.assertEqual(
            benchmark.artifact_validation_errors(1, paired),
            [],
        )
        self.assertTrue(benchmark.artifact_validation_errors(1, paired[:2]))
    def test_three_workload_classes_have_distinct_sql_temporal_contracts(self):
        sql_by_workload = {
            workload.name: benchmark.build_hybrid_sql("t", "rating = 5", workload=workload)
            for workload in benchmark.WORKLOADS
        }
        self.assertEqual(
            sql_by_workload["join_acl"], sql_by_workload["acl_only"]
        )
        distinct = {
            name: sql
            for name, sql in sql_by_workload.items()
            if name != "join_acl"
        }
        self.assertEqual(len(set(distinct.values())), len(distinct))
        self.assertNotIn("valid_from <= %(as_of)s", sql_by_workload["acl_only"])
        self.assertIn("grant_row.valid_from <= %(as_of)s", sql_by_workload["grant_temporal_selectivity"])
        self.assertIn("fact.valid_from <= %(as_of)s", sql_by_workload["fact_temporal_selectivity"])

    def test_official_compatible_sql_has_no_sqlens_profile_or_binding(self):
        workload = next(
            item
            for item in benchmark.WORKLOADS
            if item.name == "boolean_complex_narrow_not_exists"
        )

        sql = benchmark.build_hybrid_sql(
            "public.vectors",
            "rating = 5",
            workload=workload,
            official_compatible=True,
        )
        normalized = sql.lower()

        self.assertIn("not exists", normalized)
        self.assertNotIn("vector_hnsw_guidance_bind", normalized)
        self.assertNotIn("vector_hnsw_guidance_profile", normalized)
        self.assertNotIn("vector_sqlens", normalized)
        self.assertEqual(sql.count("SELECT v.id"), 1)

    def test_as_of_uses_parameterized_set_config_not_parameterized_set(self):
        cursor = mock.MagicMock()
        benchmark.set_as_of(cursor, 123456)
        cursor.execute.assert_called_once_with(
            "SELECT set_config('app.as_of', %s, false)",
            ("123456",),
        )

    def test_set_preferred_index_resets_role_before_privileged_guc(self):
        class _Cursor:
            def __init__(self) -> None:
                self.role = "amazon10m_sql_native_benchmark"
                self.preferred = ""
                self.statements: list[str] = []
                self._row: tuple[object, ...] | None = None

            def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
                normalized = " ".join(sql.split())
                self.statements.append(normalized)
                if normalized == "SELECT current_user, session_user":
                    self._row = (self.role, "postgres")
                elif normalized == "RESET ROLE":
                    self.role = "postgres"
                    self._row = None
                elif normalized.startswith("SET ROLE"):
                    self.role = "amazon10m_sql_native_benchmark"
                    self._row = None
                elif "set_config('hnsw.preferred_index'" in normalized:
                    if self.role != "postgres":
                        raise AssertionError("hnsw.preferred_index must be set as postgres")
                    self.preferred = str(params[0] if params else "")
                    self._row = (self.preferred,)
                elif normalized == "SELECT current_setting('hnsw.preferred_index')":
                    self._row = (self.preferred,)
                else:
                    raise AssertionError(f"unexpected SQL {sql!r}")

            def fetchone(self) -> tuple[object, ...] | None:
                return self._row

        cursor = _Cursor()
        current = benchmark.set_preferred_index(cursor, "public.source_idx")
        self.assertEqual(current, "public.source_idx")
        self.assertEqual(cursor.role, "amazon10m_sql_native_benchmark")
        self.assertEqual(
            cursor.statements,
            [
                "SELECT current_user, session_user",
                "RESET ROLE",
                "SELECT set_config('hnsw.preferred_index', %s, false)",
                'SET ROLE "amazon10m_sql_native_benchmark"',
                "SELECT current_setting('hnsw.preferred_index')",
            ],
        )

    def test_set_preferred_index_skips_reset_when_already_session_user(self):
        class _Cursor:
            def __init__(self) -> None:
                self.statements: list[str] = []
                self._row: tuple[object, ...] | None = None

            def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
                normalized = " ".join(sql.split())
                self.statements.append(normalized)
                if normalized == "SELECT current_user, session_user":
                    self._row = ("postgres", "postgres")
                elif "set_config('hnsw.preferred_index'" in normalized:
                    self._row = (str(params[0] if params else ""),)
                elif normalized == "SELECT current_setting('hnsw.preferred_index')":
                    self._row = ("public.source_idx",)
                else:
                    raise AssertionError(f"unexpected SQL {sql!r}")

            def fetchone(self) -> tuple[object, ...] | None:
                return self._row

        cursor = _Cursor()
        current = benchmark.set_preferred_index(cursor, "public.source_idx")
        self.assertEqual(current, "public.source_idx")
        self.assertNotIn("RESET ROLE", cursor.statements)
        self.assertFalse(any(item.startswith("SET ROLE") for item in cursor.statements))

    def test_ensure_fragment_catalog_resets_role_then_grants_principal(self):
        class _Cursor:
            def __init__(self) -> None:
                self.role = "amazon10m_sql_native_benchmark"
                self.statements: list[str] = []
                self._row: tuple[object, ...] | None = None

            def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
                normalized = " ".join(sql.split())
                self.statements.append(normalized)
                if normalized == "SELECT current_user, session_user":
                    self._row = (self.role, "postgres")
                elif normalized == "RESET ROLE":
                    self.role = "postgres"
                    self._row = None
                elif normalized.startswith("SET ROLE"):
                    self.role = "amazon10m_sql_native_benchmark"
                    self._row = None
                elif self.role != "postgres":
                    raise AssertionError(f"catalog DDL must run as postgres: {sql!r}")
                else:
                    self._row = None

            def fetchone(self) -> tuple[object, ...] | None:
                return self._row

        cursor = _Cursor()
        proof = benchmark.ensure_sqlens_fragment_catalog(
            cursor,
            "amazon10m_sql_native_benchmark",
            "public.amazon_grocery_reviews_10m_pgvector",
            enable_tracking=True,
        )
        self.assertTrue(proof["valid"])
        self.assertEqual(cursor.role, "amazon10m_sql_native_benchmark")
        self.assertEqual(cursor.statements[0], "SELECT current_user, session_user")
        self.assertEqual(cursor.statements[1], "RESET ROLE")
        self.assertTrue(
            any("CREATE TABLE IF NOT EXISTS public.pgvector_hnsw_fragment_store" in sql for sql in cursor.statements)
        )
        self.assertTrue(
            any(
                'GRANT SELECT, INSERT, UPDATE, DELETE ON public.pgvector_hnsw_fragment_store TO "amazon10m_sql_native_benchmark"'
                in sql
                for sql in cursor.statements
            )
        )
        self.assertTrue(
            any(
                'GRANT CREATE ON SCHEMA public TO "amazon10m_sql_native_benchmark"' in sql
                for sql in cursor.statements
            )
        )
        self.assertIn(
            "SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)",
            cursor.statements,
        )
        self.assertEqual(
            cursor.statements[-1],
            'SET ROLE "amazon10m_sql_native_benchmark"',
        )

    def test_ensure_fragment_catalog_rejects_unsafe_principal(self):
        cursor = mock.MagicMock()
        with self.assertRaises(RuntimeError):
            benchmark.ensure_sqlens_fragment_catalog(
                cursor, 'evil"; DROP TABLE x; --', "public.vectors"
            )
        cursor.execute.assert_not_called()

    def test_filter_loader_preserves_config_predicate_and_atoms(self):
        filters = benchmark.read_filters(ROOT / "configs" / "amazon10m_selectivity14_filters.csv")
        self.assertEqual(len(filters), 14)
        self.assertEqual(filters[0].name, "popular_ge1000")
        self.assertEqual(filters[0].predicate, "item_rating_number >= 1000")
        self.assertEqual(filters[0].atoms, ("sql:item_rating_number >= 1000",))
        self.assertEqual(filters[-1].name, "grocery_long500")

    def test_safe_guided_and_stock_use_distinct_settings_but_same_sql_contract(self):
        cursor = mock.MagicMock()
        config = benchmark.Config(500, 10000, 8.0, "strict_order", 500)
        with mock.patch.object(benchmark, "set_preferred_index", return_value="source"):
            benchmark.set_mode(cursor, "stock", config, "source")
        stock_calls = [call.args[0] for call in cursor.execute.call_args_list]
        cursor.reset_mock()
        with mock.patch.object(benchmark, "set_preferred_index", return_value="source"):
            benchmark.set_mode(cursor, "d1", config, "source")
        guided_calls = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertIn("SET hnsw.filter_strategy = off", stock_calls)
        self.assertIn("SET hnsw.filter_strategy = safe_guided", guided_calls)
        self.assertIn("SET hnsw.page_access = off", stock_calls)
        self.assertIn("SET hnsw.index_page_access = off", stock_calls)
        self.assertIn("SET hnsw.page_access = off", guided_calls)
        self.assertIn("SET hnsw.index_page_access = off", guided_calls)
        for calls in (stock_calls, guided_calls):
            self.assertIn("SET enable_seqscan = off", calls)
            self.assertIn("SET enable_bitmapscan = off", calls)
            self.assertIn("SET join_collapse_limit = 1", calls)
            self.assertIn("SET from_collapse_limit = 1", calls)
        self.assertEqual(
            benchmark.build_hybrid_sql("t", "rating = 5"),
            benchmark.build_hybrid_sql("t", "rating = 5"),
        )

    def test_heap_competing_indexes_are_toggled_as_postgres(self):
        class Cursor:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def execute(self, sql: str, params=None) -> None:
                self.calls.append(" ".join(str(sql).split()))
                if params is not None:
                    self.params = params

            def fetchone(self):
                return ("amazon10m_sql_native_benchmark", "postgres")

            def fetchall(self):
                return [("amazon_grocery_reviews_10m_pgvector_helpful_vote_idx",)]

        cursor = Cursor()
        names = benchmark.set_heap_competing_indexes_valid(
            cursor, "public.amazon_grocery_reviews_10m_pgvector", valid=False
        )
        self.assertEqual(
            names, ["amazon_grocery_reviews_10m_pgvector_helpful_vote_idx"]
        )
        self.assertTrue(any("RESET ROLE" in call for call in cursor.calls))
        self.assertTrue(
            any(
                "UPDATE pg_index" in call
                and "am.amname <> 'hnsw'" in call
                and "indisprimary" not in call
                for call in cursor.calls
            )
        )
        self.assertEqual(cursor.params, (False, "amazon_grocery_reviews_10m_pgvector"))
        self.assertTrue(
            any(
                'SET ROLE "amazon10m_sql_native_benchmark"' in call
                for call in cursor.calls
            )
        )

    def test_bootstrap_lcb_is_reproducible_and_mean_recall_selects_config(self):
        values = [0.91, 0.96, 0.99, 1.0]
        self.assertEqual(benchmark.bootstrap_bounds(values, 500, 17), benchmark.bootstrap_bounds(values, 500, 17))
        base = {
            "workload": "join_acl",
            "filter_name": "f",
            "complete": True,
            "target_met": True,
            "recall_mean": 0.96,
            "recall_lcb95": 0.96,
            "latency_mean_ms": 12.0,
            "config": "fast",
        }
        slower = {**base, "config": "slow", "latency_mean_ms": 20.0}
        self.assertEqual(benchmark.select_config([slower, base], 0.95)["config"], "fast")
        self.assertIsNone(benchmark.select_config([{**base, "recall_lcb95": 0.94, "target_met": False}], 0.95))

    def test_calibration_early_stop_waits_for_full_ef_group_and_selects_fastest(self):
        fast_low = benchmark.Config(250, 1000, 8.0, "strict_order", 250)
        reaches_high = benchmark.Config(250, 1000, 8.0, "relaxed_order", 250)
        unexecuted = benchmark.Config(500, 1000, 8.0, "strict_order", 500)
        configs = sorted(
            [fast_low, reaches_high, unexecuted],
            key=lambda config: (config.ef_search, config.label),
        )

        def summary(config, target, recall_mean, latency):
            met = recall_mean >= target
            return {
                "target_recall": target,
                "complete": True,
                "errors": 0,
                "target_met": met,
                "recall_mean": recall_mean,
                "recall_lcb95": recall_mean,
                "latency_mean_ms": latency if met else benchmark.NA,
                "config": config.label,
            }

        summaries = [
            summary(fast_low, 0.90, 0.96, 5.0),
            summary(fast_low, 0.99, 0.96, 5.0),
            summary(reaches_high, 0.90, 1.0, 10.0),
            summary(reaches_high, 0.99, 1.0, 10.0),
        ]
        executed = [config.label for config in configs if config.ef_search == 250]
        outcome = benchmark.calibration_outcome(
            summaries, configs, executed, [0.90, 0.99]
        )
        self.assertTrue(outcome["stopped"])
        self.assertFalse(outcome["grid_exhausted"])
        self.assertEqual(outcome["executed_blocks"], 2)
        self.assertEqual(outcome["selected"][0.90]["config"], fast_low.label)
        self.assertEqual(outcome["selected"][0.99]["config"], reaches_high.label)

    def test_unattainable_requires_error_free_full_grid_exhaustion(self):
        configs = [
            benchmark.Config(250, 1000, 8.0, "strict_order", 250),
            benchmark.Config(500, 1000, 8.0, "strict_order", 500),
        ]
        summaries = [
            {
                "target_recall": 0.99,
                "complete": True,
                "errors": 0,
                "target_met": False,
                "recall_lcb95": 0.8,
                "latency_mean_ms": benchmark.NA,
                "config": config.label,
            }
            for config in configs
        ]
        exhausted = benchmark.calibration_outcome(
            summaries, configs, [config.label for config in configs], [0.99]
        )
        self.assertEqual(exhausted["unattainable_on_grid"], [0.99])
        with_error = [dict(row) for row in summaries]
        with_error[-1].update({"complete": False, "errors": 1})
        indeterminate = benchmark.calibration_outcome(
            with_error, configs, [config.label for config in configs], [0.99]
        )
        self.assertEqual(indeterminate["unattainable_on_grid"], [])
        self.assertEqual(indeterminate["indeterminate_targets"], [0.99])

    def test_exact_checkpoint_round_trip_and_tie_fields_are_strict(self):
        workload = benchmark.WORKLOADS[0]
        spec = benchmark.FilterSpec(
            "f", "1%", "rating = 5", ("sql:rating = 5",), 10, 1.0
        )
        sql_text = benchmark.build_hybrid_sql("t", spec.predicate, workload=workload, exact=True)
        truth = benchmark.ExactTruth((1, 2), 0.5, benchmark.distance_tolerance(0.5), True)
        record = benchmark.exact_truth_record(
            workload, spec, 0, 10, "calibration", 123, "table-fp", sql_text, truth
        )
        restored = benchmark.restore_exact_truth(
            [record],
            [workload],
            [spec],
            {0: 10},
            {0: "calibration"},
            {workload.name: 123},
            "t",
            "table-fp",
            2,
            require_complete=True,
        )
        self.assertEqual(restored[(workload.name, spec.name, 0)], truth)
        stale = {**record, "tie_tolerance": truth.tie_tolerance * 2}
        with self.assertRaisesRegex(RuntimeError, "incomplete or stale"):
            benchmark.restore_exact_truth(
                [stale],
                [workload],
                [spec],
                {0: 10},
                {0: "calibration"},
                {workload.name: 123},
                "t",
                "table-fp",
                2,
            )

    def test_checkpoint_run_spec_mismatch_is_rejected(self):
        run_spec = {"k": 10, "queries": [[0, 10]]}
        checkpoint = {
            "checkpoint_version": benchmark.CHECKPOINT_VERSION,
            "run_spec": run_spec,
            "run_spec_sha256": benchmark.canonical_sha256(run_spec),
        }
        benchmark.validate_checkpoint_run_spec(checkpoint, run_spec)
        with self.assertRaisesRegex(RuntimeError, "run-spec mismatch"):
            benchmark.validate_checkpoint_run_spec(checkpoint, {"k": 20, "queries": [[0, 10]]})
        runner_only = {**run_spec, "runner_sha256": "ab" * 32}
        benchmark.validate_checkpoint_run_spec(checkpoint, runner_only)

    def test_checkpoint_directory_uses_atomic_immutable_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.checkpoint"
            run_spec = {"k": 10}
            checkpoint = {
                "run_spec": run_spec,
                "loaded_sessions": [],
            }
            benchmark.initialize_checkpoint(path, checkpoint)
            record = {"workload": "join_acl", "filter_name": "f", "query_no": 0}
            benchmark.persist_checkpoint_entry(
                path, "exact_truth", "join_acl|f|q0", record
            )
            loaded = benchmark.load_checkpoint(path)
            benchmark.validate_checkpoint_run_spec(loaded, run_spec)
            self.assertEqual(loaded["exact_truth"], [record])
            with self.assertRaisesRegex(RuntimeError, "immutable key"):
                benchmark.persist_checkpoint_entry(
                    path,
                    "exact_truth",
                    "join_acl|f|q0",
                    {**record, "query_no": 1},
                )
            benchmark.remove_checkpoint(path)
            self.assertFalse(path.exists())

    def test_measurement_checkpoint_rejects_missing_repeats_and_query_id_changes(self):
        workload = benchmark.WORKLOADS[0]
        spec = benchmark.FilterSpec(
            "f", "1%", "rating = 5", ("sql:rating = 5",), 10, 1.0
        )
        config = benchmark.Config(250, 1000, 8.0, "strict_order", 250)
        truth = benchmark.ExactTruth((1,), 0.5, benchmark.distance_tolerance(0.5), False)
        sql_text = benchmark.build_hybrid_sql("t", spec.predicate, workload=workload)
        sql_hash = hashlib.sha256(sql_text.encode()).hexdigest()
        activation = {
            "guidance_enabled": False,
            "guidance_route": "stock",
            "before": {
                "binding_attempts": 4,
                "binding_matches": 3,
                "binding_scan_checks": 2,
                "binding_scan_matches": 2,
                "binding_scan_bypasses": 0,
            },
        }
        execution_profile = {
            "active": False,
            "effective_active": False,
            "statement_bound": False,
            "binding_attempts": 5,
            "binding_matches": 3,
            "binding_scan_checks": 2,
            "binding_scan_matches": 2,
            "binding_scan_bypasses": 1,
        }
        scan_profile = {
            "valid": True,
            "guidance_checks": 0,
            "final_path": "stock_bypass",
        }
        execution_proof = benchmark.guidance_execution_proof(
            "stock", activation, execution_profile, scan_profile
        )
        row = {
            "phase": "calibration",
            "target_recall": "",
            "workload": workload.name,
            "filter_name": spec.name,
            "predicate": spec.predicate,
            "workload_scalar_predicate_sha256": benchmark.workload_scalar_predicate_sha256(
                spec.predicate
            ),
            "candidate_universe_predicate": "embedding_valid",
            "candidate_universe_predicate_sha256": benchmark.candidate_universe_predicate_sha256(
                "embedding_valid"
            ),
            "as_of": 123,
            "principal": "principal",
            "snapshot_as_of": 123,
            "mode": "stock",
            "selected_vector_index": "public.source_hnsw",
            "preferred_index_current_setting": "public.source_hnsw",
            "filter_strategy": "off",
            "guidance_semantics": benchmark.MODE_SPECS["stock"].guidance_semantics,
            "hard_traversal_used": False,
            "guidance_activation_profile": activation,
            "execution_guidance_profile": execution_profile,
            "scan_profile": scan_profile,
            "guidance_execution_proof": execution_proof,
            "guidance_binding_matched": execution_proof["binding_matched"],
            "guidance_effective_active": execution_proof["effective_active"],
            "guidance_checks": execution_proof["guidance_checks"],
            "guidance_final_path": execution_proof["final_path"],
            "config": config.label,
            "query_no": 0,
            "query_id": 10,
            "repeat": 0,
            "pair_key": f"{workload.name}|{spec.name}|q0|r0",
            "query_sql": sql_text,
            "query_sql_sha256": sql_hash,
            "exact_gt_ids": "1",
            "exact_gt_kth_distance": truth.kth_distance,
            "exact_gt_tie_tolerance": truth.tie_tolerance,
            "exact_gt_boundary_tied": False,
        }
        block = {
            "rows": [row],
            "plans": [
                {
                    "phase": "calibration",
                    "workload": workload.name,
                    "filter_name": spec.name,
                    "mode": "stock",
                    "config": config.label,
                    "sql_sha256": sql_hash,
                    "as_of": 123,
                    "principal": "principal",
                    "snapshot_as_of": 123,
                    "selected_vector_index": "public.source_hnsw",
                    "preferred_index_current_setting": "public.source_hnsw",
                    "filter_strategy": "off",
                    "hard_traversal_used": False,
                    "explain_order": "after_all_timed_requests_in_block",
                    "plan_state_proof": {"valid": True},
                    "explain_gate": {
                        "valid": True,
                        "require_hnsw": True,
                        "expected_index_qualified": "public.source_hnsw",
                    },
                }
            ],
        }
        kwargs = {
            "phase": "calibration",
            "workload": workload,
            "spec": spec,
            "query_ids": {0: 10},
            "repeats": 1,
            "modes": ("stock",),
            "configs": {"stock": config},
            "target_recall": None,
            "truth": {(workload.name, spec.name, 0): truth},
            "table": "t",
            "principal": "principal",
            "source_index": "public.source_hnsw",
            "clone_index": "public.clone_hnsw",
        }
        benchmark.validate_measurement_block(block, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "row count/query IDs/repeats"):
            benchmark.validate_measurement_block(block, **{**kwargs, "repeats": 2})
        stale_row = {**row, "query_id": 11}
        with self.assertRaisesRegex(RuntimeError, "incomplete or stale"):
            benchmark.validate_measurement_block({**block, "rows": [stale_row]}, **kwargs)
        forged_proof = {**execution_proof, "binding_attempts_delta": 999}
        with self.assertRaisesRegex(RuntimeError, "incomplete or stale"):
            benchmark.validate_measurement_block(
                {
                    **block,
                    "rows": [
                        {**row, "guidance_execution_proof": forged_proof}
                    ],
                },
                **kwargs,
            )
        for field, value in (
            ("principal", "wrong_role"),
            ("snapshot_as_of", 124),
            ("preferred_index_current_setting", "public.clone_hnsw"),
            ("hard_traversal_used", True),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "incomplete or stale"):
                    benchmark.validate_measurement_block(
                        {**block, "rows": [{**row, field: value}]}, **kwargs
                    )
        wrong_plan = {
            **block["plans"][0],
            "selected_vector_index": "public.clone_hnsw",
        }
        with self.assertRaisesRegex(RuntimeError, "EXPLAIN plan"):
            benchmark.validate_measurement_block(
                {**block, "plans": [wrong_plan]}, **kwargs
            )

    def test_final_targets_are_the_stock_sqlens_common_attainable_set(self):
        stock = {"selected": {0.90: {"config": "s90"}, 0.95: {"config": "s95"}, 0.99: None}}
        sqlens = {"selected": {0.90: {"config": "m90"}, 0.95: None, 0.99: {"config": "m99"}}}
        self.assertEqual(
            benchmark.common_attainable_targets([stock, sqlens], [0.90, 0.95, 0.99]),
            [0.90],
        )

    def test_missing_pair_makes_latency_and_pair_metrics_na(self):
        expected = {("join_acl", "f", 0, 0), ("join_acl", "f", 1, 0)}
        rows = [{"workload": "join_acl", "filter_name": "f", "query_no": 0, "repeat": 0, "pair_key": "join_acl|f|q0|r0", "query_id": 10, "predicate": "rating = 5", "query_sql_sha256": "same", "exact_gt_ids": "1", "query_ms": 10.0, "recall": 1.0, "error": ""}]
        summary = benchmark.summarize_rows(rows, expected_keys=expected, target_recall=0.90, bootstrap_samples=100, seed=1)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["latency_mean_ms"], benchmark.NA)
        self.assertEqual(summary["status"], benchmark.NA)
        paired = benchmark.paired_summary(rows, rows, expected_keys=expected, target_recall=0.90, bootstrap_samples=100, seed=1)
        self.assertEqual(paired["speedup_vs_stock"], benchmark.NA)
        self.assertEqual(paired["status"], benchmark.NA)

    def test_paired_summary_can_skip_per_cell_adaptive_evidence(self):
        expected = {("grant_temporal_selectivity", "helpful_ge20", q, 0) for q in (0, 1)}
        stock = [
            {
                "workload": "grant_temporal_selectivity",
                "filter_name": "helpful_ge20",
                "query_no": q,
                "repeat": 0,
                "pair_key": f"grant_temporal_selectivity|helpful_ge20|q{q}|r0",
                "query_id": 10 + q,
                "predicate": "helpful_vote >= 20",
                "query_sql_sha256": "same",
                "exact_gt_ids": "1",
                "e2e_ms": 40.0,
                "query_ms": 40.0,
                "recall": 1.0,
                "error": "",
                "mode": "stock",
            }
            for q in (0, 1)
        ]
        sqlens = [
            {
                **row,
                "mode": "d1_d2_d3",
                "e2e_ms": 20.0,
                "query_ms": 20.0,
                "adaptive_probe_observed": False,
                "adaptive_materialized": False,
                "adaptive_active": True,
                "adaptive_admission_observed": True,
                "hidden_prebuilt_fragment_reused": False,
            }
            for row in stock
        ]
        blocked = benchmark.paired_summary(
            stock, sqlens, expected_keys=expected, target_recall=0.90,
            bootstrap_samples=100, seed=1, method_mode="d1_d2_d3",
        )
        self.assertEqual(blocked["status"], benchmark.NA)
        allowed = benchmark.paired_summary(
            stock, sqlens, expected_keys=expected, target_recall=0.90,
            bootstrap_samples=100, seed=1, method_mode="d1_d2_d3",
            require_adaptive_evidence=False,
        )
        self.assertEqual(allowed["status"], "complete")
        self.assertAlmostEqual(float(allowed["speedup_vs_stock"]), 2.0)

    def test_paired_sql_contract_rejects_different_query_or_exact_gt(self):
        stock = [{"pair_key": "p", "query_id": 1, "predicate": "rating = 5", "query_sql_sha256": "sql", "exact_gt_ids": "1,2"}]
        method = [{"pair_key": "p", "query_id": 1, "predicate": "rating = 5", "query_sql_sha256": "sql", "exact_gt_ids": "2,3"}]
        with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
            benchmark.validate_paired_query_contract(stock, method)

    def test_recall_below_target_makes_latency_na(self):
        expected = {("temporal_as_of", "f", 0, 0), ("temporal_as_of", "f", 1, 0)}
        rows = [
            {"workload": "temporal_as_of", "filter_name": "f", "query_no": q, "repeat": 0, "query_ms": 10.0 + q, "recall": 0.5, "error": ""}
            for q in (0, 1)
        ]
        summary = benchmark.summarize_rows(rows, expected_keys=expected, target_recall=0.90, bootstrap_samples=100, seed=1)
        self.assertTrue(summary["complete"])
        self.assertLess(float(summary["recall_lcb95"]), 0.90)
        self.assertEqual(summary["latency_p50_ms"], benchmark.NA)

    def test_target_matching_uses_query_mean_and_reports_lcb_separately(self):
        expected = {("join_acl", "f", q, 0) for q in (0, 1)}
        rows = [
            {
                "workload": "join_acl",
                "filter_name": "f",
                "query_no": q,
                "repeat": 0,
                "e2e_ms": 10.0,
                "query_ms": 10.0,
                "recall": recall,
                "error": "",
            }
            for q, recall in ((0, 1.0), (1, 0.8))
        ]
        summary = benchmark.summarize_rows(
            rows,
            expected_keys=expected,
            target_recall=0.90,
            bootstrap_samples=500,
            seed=4,
        )
        self.assertAlmostEqual(float(summary["recall_mean"]), 0.90)
        self.assertLess(float(summary["recall_lcb95"]), 0.90)
        self.assertFalse(summary["target_met"])
        self.assertEqual(summary["latency_mean_ms"], benchmark.NA)

    def test_scan_profile_export_is_explicit_and_complete(self):
        profile = {field: index for index, field in enumerate(benchmark.SQLENS_PROFILE_EXPORT_FIELDS)}
        self.assertEqual(benchmark.scan_profile_export(profile), profile)
        missing = benchmark.scan_profile_export({})
        self.assertEqual(set(missing), set(benchmark.SQLENS_PROFILE_EXPORT_FIELDS))
        self.assertTrue(all(value == benchmark.NA for value in missing.values()))

    def test_primary_summary_uses_e2e_and_keeps_activation_and_query_components(self):
        expected = {("join_acl", "f", 0, 0), ("join_acl", "f", 1, 0)}
        rows = [
            {
                "workload": "join_acl",
                "filter_name": "f",
                "query_no": q,
                "repeat": 0,
                "activation_ms": 2.0,
                "query_ms": 8.0,
                "e2e_ms": 10.0,
                "recall": 1.0,
                "error": "",
            }
            for q in (0, 1)
        ]
        summary = benchmark.summarize_rows(rows, expected_keys=expected, target_recall=0.90, bootstrap_samples=100, seed=1)
        self.assertEqual(summary["primary_timing_field"], "e2e_ms")
        self.assertEqual(summary["latency_mean_ms"], 10.0)
        self.assertEqual(summary["activation_mean_ms"], 2.0)
        self.assertEqual(summary["query_mean_ms"], 8.0)
        self.assertEqual(summary["latency_ci95_low_ms"], 10.0)
        self.assertEqual(summary["latency_ci95_high_ms"], 10.0)

    def test_interleaved_schedule_pairs_every_key_and_is_seeded(self):
        keys = [("join_acl", "f", 0, 0), ("temporal_as_of", "f", 1, 0)]
        first = benchmark.interleaved_schedule(keys, benchmark.MODES, 11)
        self.assertEqual(first, benchmark.interleaved_schedule(keys, benchmark.MODES, 11))
        self.assertEqual(len(first), len(keys) * len(benchmark.MODES))
        for key in keys:
            self.assertEqual({mode for scheduled_key, mode in first if scheduled_key == key}, set(benchmark.MODES))

    def test_balanced_mixed_trace_is_seeded_unique_and_nearly_uniform(self):
        query_ids = {query_no: 100_000 + query_no for query_no in range(10_000)}
        workloads = list(benchmark.WORKLOADS[:3])
        filters = [
            benchmark.FilterSpec(
                f"f{index}", "1%", "rating = 5",
                ("sql:rating = 5",), 1, 1.0,
            )
            for index in range(4)
        ]
        first = benchmark.build_balanced_mixed_trace(
            workloads, filters, query_ids, 19
        )
        second = benchmark.build_balanced_mixed_trace(
            workloads, filters, query_ids, 19
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10_000)
        self.assertEqual(len({row["query_no"] for row in first}), 10_000)
        counts = {}
        for row in first:
            key = (row["workload"], row["filter_name"])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 12)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(
            benchmark.mixed_trace_sha256(first),
            benchmark.mixed_trace_sha256(second),
        )

    def test_sql_first_explain_gate_requires_registered_scalar_and_no_hnsw(self):
        scalar_plan = [
            {
                "Plan": {
                    "Node Type": "Index Scan",
                    "Index Name": "amazon_review_facts_parent_time_idx",
                }
            }
        ]
        gate = benchmark.validate_sql_first_explain_gate(
            scalar_plan, ["amazon_review_facts_parent_time_idx"]
        )
        self.assertTrue(gate["valid"])
        self.assertTrue(gate["forced_indexed"])
        self.assertEqual(gate["vector_hnsw_index_names"], [])
        with self.assertRaisesRegex(RuntimeError, "SQL-first EXPLAIN"):
            benchmark.validate_sql_first_explain_gate(
                [{"Plan": {"Index Name": "amazon10m_hnsw_idx"}}],
                ["amazon_review_facts_parent_time_idx"],
            )

    def test_p0_requested_slice_completion_requires_every_mode_repeat_key(self):
        trace = [
            {
                "request_no": 0,
                "workload": "acl_only",
                "filter_name": "f",
                "query_no": 200,
                "query_id": 1,
            },
            {
                "request_no": 1,
                "workload": "acl_only",
                "filter_name": "f",
                "query_no": 201,
                "query_id": 2,
            },
        ]
        rows = []
        for repeat in range(2):
            for request in trace:
                for mode in benchmark.P0_MODES:
                    rows.append(
                        {
                            "phase": "measurement",
                            "mode": mode,
                            "repeat": repeat,
                            "pair_key": (
                                f"{request['workload']}|{request['filter_name']}|"
                                f"q{request['query_no']}|r{repeat}"
                            ),
                            "error": "",
                        }
                    )
        complete = benchmark.p0_requested_slice_completion(
            rows, trace, benchmark.P0_MODES, 2
        )
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["expected_measurement_rows"], 12)
        incomplete = benchmark.p0_requested_slice_completion(
            rows[:-1], trace, benchmark.P0_MODES, 2
        )
        self.assertFalse(incomplete["complete"])

    def test_run_measurements_selected_modes_avoids_running_unselected_mode(self):
        class FakeCursor:
            def __init__(self):
                self.calls = []

            def execute(self, sql, params=None):
                self.calls.append((str(sql), params))

            def fetchone(self):
                return ([{"Plan": {"Node Type": "Index Scan", "Index Name": "vector_hnsw_idx"}}],)

            def fetchall(self):
                return [
                    (
                        1,
                        0.25,
                        {
                            "active": False,
                            "effective_active": False,
                            "statement_bound": False,
                            "binding_attempts": 1,
                            "binding_matches": 0,
                            "binding_scan_checks": 0,
                            "binding_scan_matches": 0,
                            "binding_scan_bypasses": 1,
                        },
                    )
                ]

            def close(self):
                pass

        class FakeConnection:
            def __init__(self):
                self.cursor_value = FakeCursor()

            def cursor(self):
                return self.cursor_value

        connection = FakeConnection()
        spec = benchmark.FilterSpec("f", "1%", "rating = 5", ("sql:rating = 5",), 1, 1.0)
        config = benchmark.Config(250, 5_000_000, 32.0, "strict_order", 250)
        context = {
            "current_user": "principal",
            "app_as_of": "1000",
            "preferred_index": "public.vector_hnsw_idx",
            "filter_strategy": "off",
            "page_access": "off",
            "index_page_access": "off",
        }
        with (
            mock.patch.object(benchmark, "set_mode"),
            mock.patch.object(
                benchmark,
                "configure_guidance",
                return_value={
                    "guidance_enabled": False,
                    "guidance_route": "stock",
                    "activation_atom_count": 0,
                    "activation_ms": 0.25,
                    "before": {},
                    "after_activation": {},
                },
            ) as configure,
            mock.patch.object(benchmark, "runtime_sql_context", return_value=context),
            mock.patch.object(
                benchmark,
                "explain",
                return_value=(
                    [{"Plan": {"Index Name": "vector_hnsw_idx"}}],
                    {
                        "valid": True,
                        "require_hnsw": True,
                        "expected_index_qualified": "public.vector_hnsw_idx",
                    },
                ),
            ),
            mock.patch.object(
                benchmark,
                "fetch_json_object",
                side_effect=lambda _cur, sql: (
                    {"valid": True, "final_path": "stock"}
                    if "last_scan" in sql
                    else {}
                ),
            ),
        ):
            rows, plans = benchmark.run_measurements(
                {"stock": connection},
                {"stock": config},
                [benchmark.WORKLOADS[0]],
                [spec],
                {0: 1},
                {("acl_only", "f", 0): (1,)},
                {"acl_only": 1000},
                "t",
                "public.vector_hnsw_idx",
                "public.vector_hnsw_clone_idx",
                "principal",
                1,
                1,
                "calibration",
                None,
                7,
                selected_modes=("stock",),
                query_embeddings={1: "[1,0,0]"},
            )
        self.assertEqual({row["mode"] for row in rows}, {"stock"})
        self.assertEqual(len(plans), 1)
        configure.assert_called_once()
        self.assertEqual(plans[0]["explain_order"], "after_all_timed_requests_in_block")
        self.assertNotIn("safe_guided", " ".join(call[0] for call in connection.cursor_value.calls))
        self.assertGreaterEqual(rows[0]["e2e_ms"], rows[0]["query_ms"])
        self.assertNotEqual(
            rows[0]["e2e_ms"],
            rows[0]["activation_ms"] + rows[0]["query_ms"],
        )

    def test_sql_first_workers_execute_on_pooled_cursors(self):
        class NamedCursor:
            def __init__(self, name):
                self.name = name
                self.calls = []

            def execute(self, sql, params=None):
                self.calls.append((str(sql), params))

            def fetchone(self):
                return (
                    [{"Plan": {"Node Type": "Index Scan", "Index Name": "price_idx"}}],
                )

            def fetchall(self):
                return [(1, 0.25)]

            def close(self):
                pass

        class NamedConnection:
            def __init__(self, name):
                self.name = name
                self.cursors = []

            def cursor(self):
                cursor = NamedCursor(self.name)
                self.cursors.append(cursor)
                return cursor

        main = NamedConnection("main")
        worker = NamedConnection("worker")
        spec = benchmark.FilterSpec("f", "1%", "rating = 5", ("sql:rating = 5",), 1, 1.0)
        context = {
            "current_user": "principal",
            "app_as_of": "1000",
            "preferred_index": "public.vector_hnsw_idx",
            "filter_strategy": "forced_indexed_exact",
            "page_access": "off",
            "index_page_access": "off",
        }
        with (
            mock.patch.object(benchmark, "set_mode"),
            mock.patch.object(
                benchmark,
                "configure_guidance",
                return_value={
                    "sql_first_forced_indexed_exact": True,
                    "guidance_enabled": False,
                    "guidance_route": benchmark.SQL_FIRST_MODE,
                    "activation_atom_count": 0,
                    "activation_ms": 0.0,
                    "before": {},
                    "after_activation": {},
                },
            ),
            mock.patch.object(benchmark, "runtime_sql_context", return_value=context),
            mock.patch.object(
                benchmark,
                "explain",
                return_value=(
                    [{"Plan": {"Index Name": "price_idx"}}],
                    {"valid": True, "require_hnsw": False},
                ),
            ),
            mock.patch.object(
                benchmark,
                "validate_sql_first_explain_gate",
                return_value={"valid": True, "require_hnsw": False},
            ),
            mock.patch.object(
                benchmark,
                "prepare_explain_without_runtime_state",
                return_value={"guidance": {}, "cache": {}},
            ),
            mock.patch.object(
                benchmark,
                "finish_explain_without_runtime_state",
                return_value={"valid": True},
            ),
        ):
            rows, plans = benchmark.run_measurements(
                {benchmark.SQL_FIRST_MODE: main},
                {benchmark.SQL_FIRST_MODE: benchmark.SQL_FIRST_CONFIG},
                [benchmark.WORKLOADS[0]],
                [spec],
                {0: 1},
                {("acl_only", "f", 0): (1,)},
                {"acl_only": 1000},
                "t",
                "public.vector_hnsw_idx",
                "public.vector_hnsw_clone_idx",
                "principal",
                1,
                1,
                "measurement",
                0.9,
                7,
                selected_modes=(benchmark.SQL_FIRST_MODE,),
                registered_scalar_indexes=("public.price_idx",),
                sql_first_workers=2,
                sql_first_connections=[worker],
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], benchmark.SQL_FIRST_MODE)
        self.assertFalse(rows[0]["error"])
        worker_sql = [sql for cursor in worker.cursors for sql, _ in cursor.calls]
        shared_sql = [sql for sql, _ in main.cursors[0].calls]
        self.assertTrue(
            any("FROM" in sql and "rating = 5" in sql for sql in worker_sql),
            worker_sql,
        )
        self.assertFalse(
            any("FROM" in sql and "rating = 5" in sql for sql in shared_sql),
            shared_sql,
        )
        self.assertEqual(len(plans), 1)

    def test_dry_run_does_not_read_missing_inputs_or_connect(self):
        output = io.StringIO()
        with (
            mock.patch.object(benchmark, "run_benchmark") as run,
            mock.patch("sys.stdout", output),
        ):
            status = benchmark.main(["--dry-run", "--filters-csv", "/missing/filters.csv"])
        self.assertEqual(status, 0)
        run.assert_not_called()
        text = output.getvalue()
        self.assertIn("database=not_opened", text)
        self.assertIn("calibration=q80/r2 (q20..q99); final=q100/r5", text)
        self.assertIn("acl_only", text)

    def test_manifest_contract_records_fingerprint_and_timing_fields(self):
        args = benchmark.create_argument_parser().parse_args([])
        binary_identity_gate = self._binary_identity_gate()
        manifest = benchmark._manifest(
            args,
            [benchmark.FilterSpec("f", "1%", "rating = 5", ("sql:rating = 5",), 10, 1.0)],
            {0: 10},
            {100: 20},
            {
                "sqlens_build_id": "build-x",
                "relations": {"t": {"oid": 1, "data_epoch": 9}},
                "formal_data_version_proof": {
                    "start_relations": {"t": {"oid": 1, "data_epoch": 9}}
                },
                "binary_identity_gate": binary_identity_gate,
            },
            {"join_acl": 1000},
        )
        self.assertEqual(manifest["database"]["sqlens_build_id"], "build-x")
        self.assertEqual(manifest["binary_identity_gate"], binary_identity_gate)
        self.assertEqual(
            manifest["binary_identity_gate"]["observed_vector_so_sha256"],
            binary_identity_gate["expected_vector_so_sha256"],
        )
        self.assertIn("schema_sql_sha256", manifest)
        self.assertIn("timing_definition", manifest)
        self.assertIn("candidate-admission/validation", manifest["sqlens_filter_strategy"])
        self.assertFalse(manifest["rls_and_guidance_contract"]["hard_traversal_pruning"])
        self.assertEqual(manifest["source_index"], benchmark.DEFAULT_SOURCE_INDEX)
        self.assertEqual(manifest["clone_index"], benchmark.DEFAULT_CLONE_INDEX)
        self.assertTrue(manifest["sql_contract"]["single_select"])
        self.assertEqual(manifest["relation_epoch"]["relations"], {"t": 9})
        self.assertEqual(
            manifest["candidate_universe"]["predicate"], "embedding_valid"
        )
        self.assertTrue(manifest["rls_and_guidance_contract"]["facts_policy_always_enforced"])
        self.assertEqual(
            manifest["rls_and_guidance_contract"]["executor_recheck"],
            ["JOIN", "ACL", "temporal", "RLS"],
        )

    def test_mode_contract_uses_source_for_stock_d1_and_clone_for_d2_d3(self):
        source = "public.source_hnsw"
        clone = "public.clone_hnsw"
        self.assertEqual(benchmark.mode_index("stock", source, clone), source)
        self.assertEqual(benchmark.mode_index("d1", source, clone), source)
        self.assertEqual(benchmark.mode_index("d1_d2", source, clone), clone)
        self.assertEqual(benchmark.mode_index("d1_d2_d3", source, clone), clone)
        for mode in benchmark.SQLENS_MODES:
            self.assertEqual(benchmark.MODE_SPECS[mode].filter_strategy, "safe_guided")
            self.assertIn("candidate_admission", benchmark.MODE_SPECS[mode].guidance_semantics)
        self.assertEqual(benchmark.MODE_SPECS["d1_d2_d3"].guidance_kind, "adaptive")

    def test_graph_clone_proof_raises_session_maintenance_work_mem(self):
        class _Cursor:
            def __init__(self) -> None:
                self.statements: list[tuple[str, tuple[object, ...] | None]] = []
                self._row: tuple[object, ...] | None = None

            def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
                self.statements.append((" ".join(sql.split()), params))
                if "set_config('maintenance_work_mem'" in sql:
                    self._row = (params[0] if params else "",)
                elif "vector_hnsw_graph_compare" in sql:
                    self._row = (
                        {
                            "same_heap": True,
                            "logical_equal": True,
                            "entry_equal": True,
                            "tuple_coverage_equal": True,
                            "definition_equal": True,
                            "physical_equal": False,
                        },
                    )
                else:
                    raise AssertionError(f"unexpected SQL {sql!r}")

            def fetchone(self) -> tuple[object, ...] | None:
                return self._row

        cursor = _Cursor()
        proof = benchmark.graph_clone_proof(
            cursor, "public.source_hnsw", "public.clone_hnsw"
        )
        self.assertTrue(proof["valid"])
        self.assertEqual(proof["comparison"]["physical_equal"], False)
        self.assertEqual(
            cursor.statements[0],
            (
                "SELECT set_config('maintenance_work_mem', %s, false)",
                (benchmark.D2_GRAPH_PROOF_MAINTENANCE_WORK_MEM,),
            ),
        )

    def test_same_graph_clone_proof_is_strict_and_requires_physical_difference(self):
        proof = {
            "same_heap": True,
            "logical_equal": True,
            "entry_equal": True,
            "tuple_coverage_equal": True,
            "definition_equal": True,
            "physical_equal": False,
        }
        validated = benchmark.validate_graph_compare(
            proof, "public.source_hnsw", "public.clone_hnsw"
        )
        self.assertTrue(validated["valid"])
        for field in (
            "same_heap",
            "logical_equal",
            "entry_equal",
            "tuple_coverage_equal",
            "definition_equal",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "same-heap/same-logical-graph"):
                    benchmark.validate_graph_compare(
                        {**proof, field: False},
                        "public.source_hnsw",
                        "public.clone_hnsw",
                    )
        with self.assertRaisesRegex(RuntimeError, "same-heap/same-logical-graph"):
            benchmark.validate_graph_compare(
                {**proof, "physical_equal": True},
                "public.source_hnsw",
                "public.clone_hnsw",
            )
        with self.assertRaisesRegex(RuntimeError, "same-heap/same-logical-graph"):
            benchmark.validate_graph_compare(proof, "public.same", "public.same")

    def test_explain_gate_rejects_other_hnsw_even_when_expected_is_present(self):
        competing = [
            {
                "Plan": {
                    "Node Type": "Nested Loop",
                    "Plans": [
                        {"Node Type": "Index Scan", "Index Name": "clone_hnsw"},
                        {"Node Type": "Index Scan", "Index Name": "source_hnsw"},
                    ],
                }
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "EXPLAIN gate failed"):
            benchmark.validate_explain_gate(
                competing, "public.clone_hnsw", require_hnsw=True
            )

    def test_runtime_context_fails_closed_on_principal_as_of_or_prefetch_mismatch(self):
        cursor = mock.MagicMock()
        cursor.fetchone.return_value = (
            "principal",
            "123",
            "public.source_hnsw",
            "safe_guided",
            "off",
            "off",
        )
        context = benchmark.runtime_sql_context(cursor, "principal", 123)
        self.assertEqual(context["app_as_of"], "123")
        cursor.fetchone.return_value = (
            "wrong_role",
            "123",
            "public.source_hnsw",
            "safe_guided",
            "off",
            "off",
        )
        with self.assertRaisesRegex(RuntimeError, "principal/snapshot/prefetch gate"):
            benchmark.runtime_sql_context(cursor, "principal", 123)
        cursor.fetchone.return_value = (
            "principal",
            "124",
            "public.source_hnsw",
            "safe_guided",
            "off",
            "off",
        )
        with self.assertRaisesRegex(RuntimeError, "principal/snapshot/prefetch gate"):
            benchmark.runtime_sql_context(cursor, "principal", 123)
        cursor.fetchone.return_value = (
            "principal",
            "123",
            "public.source_hnsw",
            "safe_guided",
            "off",
            "on",
        )
        with self.assertRaisesRegex(RuntimeError, "principal/snapshot/prefetch gate"):
            benchmark.runtime_sql_context(cursor, "principal", 123)

    def test_d3_summary_requires_probe_materialize_admission_and_active(self):
        expected = {("join_acl", "f", query_no, 0) for query_no in range(3)}
        rows = []
        for query_no in range(3):
            rows.append(
                {
                    "workload": "join_acl",
                    "filter_name": "f",
                    "mode": "d1_d2_d3",
                    "query_no": query_no,
                    "repeat": 0,
                    "query_ms": 10.0,
                    "e2e_ms": 11.0,
                    "recall": 1.0,
                    "error": "",
                    "adaptive_probe_observed": query_no == 0,
                    "adaptive_materialized": query_no == 2,
                    "adaptive_admission_observed": query_no == 2,
                    "adaptive_active": query_no == 2,
                }
            )
        summary = benchmark.summarize_rows(
            rows,
            expected_keys=expected,
            target_recall=0.90,
            bootstrap_samples=100,
            seed=1,
        )
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["adaptive_mode_active"])
        missing_materialization = [dict(row, adaptive_materialized=False) for row in rows]
        rejected = benchmark.summarize_rows(
            missing_materialization,
            expected_keys=expected,
            target_recall=0.90,
            bootstrap_samples=100,
            seed=1,
        )
        self.assertEqual(rejected["status"], benchmark.NA)
        self.assertFalse(rejected["adaptive_mode_active"])
        hidden_reuse = [dict(row) for row in rows]
        hidden_reuse[2]["hidden_prebuilt_fragment_reused"] = True
        rejected = benchmark.summarize_rows(
            hidden_reuse,
            expected_keys=expected,
            target_recall=0.90,
            bootstrap_samples=100,
            seed=1,
        )
        self.assertEqual(rejected["status"], benchmark.NA)

    def test_artifact_gate_rejects_unsafe_hard_traversal_and_wrong_index(self):
        summaries = [
            {
                "phase": "final",
                "mode": "paired_d1",
                "workload": "join_acl",
                "filter_name": "f",
                "target_recall": 0.90,
                "status": "complete",
            }
        ]
        row = {
            "mode": "d1",
            "pair_key": "join_acl|f|q0|r0",
            "filter_strategy": "traversal_guided",
            "guidance_semantics": benchmark.MODE_SPECS["d1"].guidance_semantics,
            "hard_traversal_used": True,
        }
        plan = {
            "phase": "final",
            "workload": "join_acl",
            "filter_name": "f",
            "mode": "d1",
            "selected_vector_index": "public.source_hnsw",
            "preferred_index_current_setting": "public.clone_hnsw",
            "explain_gate": {
                "valid": True,
                "expected_index_qualified": "public.source_hnsw",
            },
        }
        errors = benchmark.artifact_validation_errors(
            1, summaries, [row], [plan]
        )
        self.assertTrue(any("unsafe" in error for error in errors))
        self.assertTrue(any("wrong or unproven" in error for error in errors))

    def test_reset_adaptive_state_rejects_nonempty_cache(self):
        cursor = mock.MagicMock()
        with mock.patch.object(
            benchmark,
            "fetch_json_object",
            side_effect=[{"resident_entries": 2}, {"resident_entries": 0}],
        ):
            evidence = benchmark.reset_adaptive_state(
                cursor,
                {
                    "valid": True,
                    "prebuilt_fragments": 0,
                    "before": {"count": 0},
                    "after": {"count": 0},
                },
            )
        self.assertTrue(evidence["after_reset_empty"])
        self.assertEqual(evidence["prebuilt_fragments"], 0)
        with mock.patch.object(
            benchmark,
            "fetch_json_object",
            side_effect=[{}, {"resident_entries": 1}],
        ):
            with self.assertRaisesRegex(RuntimeError, "cold-start gate"):
                benchmark.reset_adaptive_state(cursor)

    def test_adaptive_activation_reports_probe_as_inactive_and_active_admission(self):
        cursor = mock.MagicMock()
        cursor.fetchone.return_value = (0,)
        probing = {
            "active": False,
            "adaptive_state": "probing",
            "adaptive_probes": 0,
        }
        with mock.patch.object(
            benchmark, "fetch_json_object", side_effect=[{}, probing]
        ):
            result = benchmark.configure_guidance(
                cursor, "d1_d2_d3", "public.clone_hnsw", ("sql:rating = 5",)
            )
        self.assertFalse(result["guidance_enabled"])
        self.assertEqual(result["guidance_route"], "d3_probing")
        self.assertTrue(
            any(call.args[0] == "SET hnsw.filter_strategy = off" for call in cursor.execute.call_args_list)
        )

        cursor.reset_mock()
        cursor.fetchone.return_value = (1,)
        active = {
            "active": True,
            "adaptive_state": "page",
            "adaptive_admissions": 1,
            "adaptive_page_builds": 1,
        }
        with mock.patch.object(
            benchmark, "fetch_json_object", side_effect=[probing, active]
        ):
            result = benchmark.configure_guidance(
                cursor, "d1_d2_d3", "public.clone_hnsw", ("sql:rating = 5",)
            )
        self.assertTrue(result["guidance_enabled"])
        self.assertEqual(result["guidance_route"], "d3_page")

    def test_database_artifact_gate_binds_graph_settings_and_empty_d3_start(self):
        graph = {
            "valid": True,
            "comparison": {
                "same_heap": True,
                "logical_equal": True,
                "entry_equal": True,
                "tuple_coverage_equal": True,
                "definition_equal": True,
                "physical_equal": False,
            },
        }
        indexes = {
            "stock": "source",
            "d1": "source",
            "d1_d2": "clone",
            "d1_d2_d3": "clone",
        }
        database = {
            "binary_identity_gate": self._binary_identity_gate(),
            "d2_graph_proof": graph,
            "d2_graph_proof_end": graph,
            "d2_index_names": ["source", "clone"],
            "relations": {"source": {"oid": 1}, "clone": {"oid": 2}},
            "d2_index_fingerprints_end": {
                "source": {"oid": 1}, "clone": {"oid": 2},
            },
            "mode_indexes": indexes,
            "preferred_index_current_settings": dict(indexes),
            "principal": "principal",
            "rls_security_proofs": {
                mode: {
                    "current_user": "principal", "is_superuser": False,
                    "bypass_rls": False, "owns_facts": False,
                    "reader_membership": True, "rls_enabled": True,
                    "policy_hash": "f" * 64,
                    "positive_probe_visible": True,
                    "negative_probe_hidden": True,
                }
                for mode in benchmark.MODES
            },
            "d3_startup_reset_evidence": {
                "after_reset_empty": True,
                "prebuilt_fragments": 0,
            },
            "d3_persistent_fragment_reset": {
                "valid": True, "prebuilt_fragments": 0,
            },
            "d3_fragment_store_end": {"count": 3, "content_sha256": "x"},
            "formal_data_version_proof": {
                "valid": True, "start_hash": "x", "end_hash": "x",
            },
        }
        self.assertEqual(benchmark.database_contract_errors(database), [])
        for mutation in (
            {"d2_graph_proof": {**graph, "valid": False}},
            {"preferred_index_current_settings": {**indexes, "d1_d2": "source"}},
            {
                "d3_startup_reset_evidence": {
                    "after_reset_empty": False,
                    "prebuilt_fragments": 1,
                }
            },
            {
                "binary_identity_gate": {
                    **self._binary_identity_gate(),
                    "all_exact_match": False,
                    "valid": False,
                }
            },
        ):
            with self.subTest(mutation=mutation):
                self.assertTrue(benchmark.database_contract_errors({**database, **mutation}))

    def test_each_measurement_requires_binding_effective_scan_and_final_path_proof(self):
        activation = {
            "before": {"binding_attempts": 4, "binding_matches": 3},
            "after_activation": {"active": True},
            "guidance_enabled": True,
            "guidance_route": "safe_guided_candidate_validation",
        }
        post = {
            "active": True,
            "effective_active": True,
            "statement_bound": True,
            "binding_attempts": 5,
            "binding_matches": 4,
            "binding_scan_checks": 1,
            "binding_scan_matches": 1,
            "binding_scan_bypasses": 0,
        }
        scan = {
            "valid": True,
            "guidance_checks": 17,
            "final_path": "validation_only",
        }
        proof = benchmark.guidance_execution_proof("d1", activation, post, scan)
        self.assertTrue(proof["valid"])
        self.assertTrue(proof["execution_profile_complete"])
        self.assertTrue(proof["binding_matched"])
        self.assertTrue(proof["effective_active"])
        self.assertEqual(proof["guidance_checks"], 17)
        self.assertEqual(proof["final_path"], "validation_only")
        self.assertFalse(proof["d3_probe_exception"])

        for broken_post, broken_scan in (
            ({**post, "effective_active": False}, scan),
            ({**post, "binding_matches": 3}, scan),
            (post, {**scan, "guidance_checks": 0}),
            (post, {**scan, "final_path": "guided"}),
        ):
            with self.subTest(post=broken_post, scan=broken_scan):
                self.assertFalse(
                    benchmark.guidance_execution_proof(
                        "d1", activation, broken_post, broken_scan
                    )["valid"]
                )

    def test_d3_probe_is_explicit_exception_but_cannot_be_reported_active(self):
        activation = {
            "before": {"binding_attempts": 7, "binding_matches": 7},
            "after_activation": {"active": False, "adaptive_state": "probing"},
            "guidance_enabled": False,
            "guidance_route": "d3_probing",
        }
        post = {
            "active": False,
            "effective_active": False,
            "statement_bound": False,
            "binding_attempts": 8,
            "binding_matches": 7,
            "binding_scan_checks": 0,
            "binding_scan_matches": 0,
            "binding_scan_bypasses": 0,
            "adaptive_state": "probing",
            "adaptive_probes": 1,
        }
        scan = {"valid": True, "guidance_checks": 0, "final_path": "stock"}
        proof = benchmark.guidance_execution_proof(
            "d1_d2_d3", activation, post, scan
        )
        self.assertTrue(proof["valid"])
        self.assertTrue(proof["d3_probe_exception"])
        self.assertFalse(proof["reported_active"])
        self.assertEqual(proof["exception_reason"], "workload_driven_probe_stock_route")

    def test_d3_probe_counts_r43_post_scan_adaptive_probes(self):
        activation = {
            "before": {"binding_attempts": 0, "binding_matches": 0, "adaptive_probes": 0},
            "after_activation": {"active": False, "adaptive_state": "probing"},
            "guidance_enabled": False,
            "guidance_route": "d3_probing",
        }
        execution = {
            "active": False,
            "effective_active": False,
            "statement_bound": False,
            "binding_attempts": 1,
            "binding_matches": 0,
            "binding_scan_checks": 0,
            "binding_scan_matches": 0,
            "binding_scan_bypasses": 0,
            "adaptive_state": "probing",
            "adaptive_probes": 0,
        }
        post = {**execution, "adaptive_probes": 1, "binding_attempts": 1}
        scan = {"valid": True, "guidance_checks": 0, "final_path": "stock"}
        self.assertFalse(
            benchmark.guidance_execution_proof(
                "d1_d2_d3", activation, execution, scan
            )["valid"]
        )
        proof = benchmark.guidance_execution_proof(
            "d1_d2_d3", activation, execution, scan, post
        )
        self.assertTrue(proof["valid"])
        self.assertTrue(proof["d3_probe_exception"])

    def test_invalid_artifact_does_not_publish_csv_or_summary_and_keeps_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint = directory / "formal.checkpoint"
            checkpoint.mkdir()
            args = SimpleNamespace(
                out=directory / "formal.csv",
                manifest=directory / "formal.manifest.json",
                plans=directory / "formal.plans.json",
            )
            manifest = {"artifact_valid": False, "artifact_errors": ["bad proof"]}
            status = benchmark.publish_benchmark_artifacts(
                args, [{"row": 1}], [{"summary": 1}], [{"plan": 1}], manifest,
                checkpoint,
            )
            self.assertNotEqual(status, 0)
            self.assertFalse(args.out.exists())
            self.assertFalse(directory.joinpath("formal_summary.csv").exists())
            self.assertTrue(checkpoint.exists())
            self.assertTrue(args.manifest.exists())
            self.assertFalse(json.loads(args.manifest.read_text())["artifact_valid"])

    def test_producer_and_consumer_share_fingerprint_and_minimal_loader(self):
        self.assertIs(benchmark.relation_fingerprint, exact_truth.relation_fingerprint)
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._external_truth_fixture(Path(tmp))
            fixture["manifest"] = exact_truth.build_artifact_manifest(
                run_spec=fixture["manifest"]["run_spec"],
                source_hashes=fixture["manifest"]["source_hashes"],
                fbin={"path": str(fixture["fbin"])},
                base_table_mapping=fixture["manifest"]["base_table_mapping"],
                outputs={"truth_csv_sha256": benchmark.sha256_file(fixture["truth_csv"])},
                backend=fixture["manifest"]["backend"],
                pairs=fixture["manifest"]["pairs"],
                data_version_proof={
                    "valid": True,
                    "start_relations": fixture["relations"],
                    "end_relations": fixture["relations"],
                    "start_hash": benchmark.canonical_sha256(fixture["relations"]),
                    "end_hash": benchmark.canonical_sha256(fixture["relations"]),
                },
                rls_security_proof={
                    "current_user": "principal", "is_superuser": False,
                    "bypass_rls": False, "owns_facts": False,
                    "reader_membership": True, "rls_enabled": True,
                    "policy_hash": benchmark.canonical_sha256([]),
                    "positive_probe_visible": True,
                    "negative_probe_hidden": True,
                },
            )
            fixture["manifest_path"].write_text(
                json.dumps(fixture["manifest"]), encoding="utf-8"
            )
            loaded, _ = self._load_external_fixture(fixture)
            self.assertEqual(loaded[(fixture["workload"].name, "f", 0)].ids, (1,))

    def test_fragment_store_reset_is_measured_not_hard_coded(self):
        before = {"exists": True, "count": 2, "content_sha256": "before"}
        after = {"exists": True, "count": 0, "content_sha256": "after"}
        proof = benchmark.validate_fragment_store_reset(before, 2, after)
        self.assertTrue(proof["valid"])
        self.assertEqual(proof["deleted_count"], 2)
        self.assertEqual(proof["prebuilt_fragments"], 0)
        with self.assertRaisesRegex(RuntimeError, "persistent fragment store"):
            benchmark.validate_fragment_store_reset(before, 1, after)

    def test_security_proof_rejects_owner_bypass_membership_and_probe_failures(self):
        proof = {
            "current_user": "principal",
            "is_superuser": False,
            "bypass_rls": False,
            "owns_facts": False,
            "reader_membership": True,
            "rls_enabled": True,
            "policy_hash": "f" * 64,
            "positive_probe_visible": True,
            "negative_probe_hidden": True,
        }
        self.assertTrue(benchmark.validate_rls_security_proof(proof, "principal")["valid"])
        for field, value in (
            ("owns_facts", True),
            ("bypass_rls", True),
            ("reader_membership", False),
            ("negative_probe_hidden", False),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "RLS security proof"):
                    benchmark.validate_rls_security_proof({**proof, field: value}, "principal")
        for missing in ("policy_hash", "positive_probe_visible", "negative_probe_hidden"):
            with self.subTest(missing=missing):
                incomplete = dict(proof)
                incomplete.pop(missing)
                with self.assertRaisesRegex(RuntimeError, "RLS security proof"):
                    benchmark.validate_rls_security_proof(incomplete, "principal")

    def test_full_matrix_is_preregistered_and_unattainable_is_explicit_na(self):
        filters = [
            benchmark.FilterSpec(f"f{number}", "1%", "rating = 5", ("sql:rating = 5",), 1, 1.0)
            for number in range(14)
        ]
        matrix = benchmark.preregister_formal_matrix(
            benchmark.WORKLOADS, filters, (0.90, 0.95, 0.99)
        )
        self.assertEqual(len(matrix), len(benchmark.WORKLOADS) * 14 * 3)
        self.assertTrue(all(cell["status"] == "preregistered" for cell in matrix))
        outcomes = {
            (workload.name, spec.name, mode): {
                "selected": {0.90: None, 0.95: None, 0.99: None},
                "unattainable_on_grid": [0.90, 0.95, 0.99],
                "indeterminate_targets": [],
            }
            for workload in benchmark.WORKLOADS
            for spec in filters
            for mode in benchmark.MODES
        }
        finalized = benchmark.finalize_formal_matrix(matrix, outcomes, set())
        self.assertTrue(all(cell["status"] == benchmark.NA for cell in finalized))
        self.assertTrue(all(cell["reason"] == "unattainable_on_grid" for cell in finalized))

    def test_official_matrix_requires_final_target_or_complete_grid_proof(self):
        workload = benchmark.WorkloadSpec("w", "fixture", 50.0, False)
        spec = benchmark.FilterSpec("f", "1%", "rating = 5", (), 1, 1.0)
        config = benchmark.Config(100, 1000, 8.0, "strict_order", 100)
        matrix = benchmark.preregister_official_formal_matrix([workload], [spec], [0.90])
        outcome = {
            "selected": {0.90: {"config": config.label}},
            "planned_blocks": 1,
            "executed_blocks": 1,
            "grid_exhausted": True,
            "error_free_grid_exhaustion": True,
            "unattainable_on_grid": [],
        }
        final = {
            "complete": True,
            "errors": 0,
            "target_met": True,
            "recall_mean": 0.91,
            "recall_lcb95": 0.90,
            "config": config.label,
        }
        complete = benchmark.finalize_official_formal_matrix(
            matrix, {("w", "f"): outcome}, {("w", "f", 0.90): final}
        )
        self.assertEqual(complete[0]["status"], "complete")
        self.assertTrue(benchmark.formal_matrix_coverage(complete, 1)["all_targets_attained"])

        missing_final = benchmark.finalize_official_formal_matrix(
            matrix, {("w", "f"): outcome}, {}
        )
        self.assertEqual(missing_final[0]["status"], "invalid")

        unattainable = benchmark.finalize_official_formal_matrix(
            matrix,
            {
                ("w", "f"): {
                    **outcome,
                    "selected": {0.90: None},
                    "unattainable_on_grid": [0.90],
                }
            },
            {},
        )
        self.assertEqual(unattainable[0]["status"], benchmark.NA)
        coverage = benchmark.formal_matrix_coverage(unattainable, 1)
        self.assertTrue(coverage["coverage_complete"])
        self.assertFalse(coverage["all_targets_attained"])

    def test_official_execution_requires_explicit_independent_contract(self):
        args = benchmark.create_argument_parser().parse_args(
            ["--execution-engine", "official"]
        )
        with self.assertRaisesRegex(RuntimeError, "independent DSN/container contract"):
            benchmark.validate_official_contract_args(args)
        valid = benchmark.create_argument_parser().parse_args(
            [
                "--execution-engine", "official",
                "--official-dsn", "postgresql://upstream.example/amazon",
                "--official-server-container", "upstream-pgvector",
                "--official-image-digest", "sha256:" + "b" * 64,
                "--official-pgvector-commit", "c" * 40,
                "--official-vector-so-sha256", "d" * 64,
            ]
        )
        benchmark.validate_official_contract_args(valid)
        self.assertNotEqual(
            benchmark.dsn_fingerprint(valid.official_dsn),
            benchmark.dsn_fingerprint("postgresql://sqlens.example/amazon"),
        )


if __name__ == "__main__":
    unittest.main()
