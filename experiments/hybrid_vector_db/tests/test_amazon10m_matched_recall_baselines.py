import csv
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.hybrid_vector_db.scripts.amazon10m_matched_recall_baselines import (
    DEFAULT_FAISS_INDEX,
    DEFAULT_FAISS_INDEX_MANIFEST,
    DEFAULT_EF_SEARCH,
    DEFAULT_CALIBRATION_WORKLOAD,
    DEFAULT_MEASUREMENT_WORKLOAD,
    DEFAULT_TRUTH,
    DEFAULT_TRUTH_MANIFEST,
    CURRENT_CALIBRATION_REQUESTS,
    CURRENT_MEASUREMENT_REPEATS,
    CURRENT_MEASUREMENT_REQUESTS,
    CURRENT_PROTOCOL,
    FAISS_METHOD,
    FORMAL_CALIBRATION_WORKLOAD_SHA256,
    FORMAL_FAISS_INDEX_MANIFEST_SHA256,
    FORMAL_FAISS_INDEX_SHA256,
    FORMAL_FILTERS_SHA256,
    FORMAL_FBIN_SHA256,
    FORMAL_METHODS,
    FORMAL_MEASUREMENT_WORKLOAD_SHA256,
    FORMAL_QUERY_COHORT_MANIFEST_SHA256,
    FORMAL_QUERY_COHORT_SHA256,
    FORMAL_TRUTH_MANIFEST_SHA256,
    FORMAL_TRUTH_SHA256,
    NA,
    LEGACY_PROTOCOL,
    SQL_FIRST_CONTROL_METHOD,
    SQL_FIRST_FORCED_METHOD,
    SQL_FIRST_PLANNER_METHOD,
    AllowList,
    FilterSpec,
    TruthEntry,
    WorkloadRequest,
    aggregate_measurements,
    artifact_validity_flags,
    artifact_validation_errors,
    allowlist_id_sql,
    assert_no_hnsw_index,
    assert_scalar_index_plan,
    balanced_order,
    bitmap_contains,
    build_allow_list,
    calibration_table,
    build_parser,
    completion_gate,
    dry_run_payload,
    execute_checkpointed_cell,
    exact_sql,
    exact_sql_for_method,
    final_summary_table,
    faiss_index_metadata,
    formal_protocol_errors,
    formal_input_hash_errors,
    full_setup_search_row,
    load_workload,
    load_truth,
    measurement_row,
    materialized_exact_sql,
    parse_methods,
    result_membership_errors,
    search_faiss,
    sha256_file,
    set_bitmap_ids,
    tie_aware_recall_at_k,
    verify_faiss_build_manifest,
    verify_truth_manifest,
    validate_checkpoint_prefix,
    validate_workload_pair,
    workload_query_nos_by_filter,
    write_csv,
)
from experiments.hybrid_vector_db.scripts.finalize_amazon10m_matched_recall_baselines import (
    FinalizationFailure,
    finalize_existing,
)


SPEC = FilterSpec(
    name="filter_a",
    target_rate="10.0%",
    predicate="helpful_vote >= 1",
    expected_rows=20,
    actual_pct=10.0,
)


def measured_row(
    phase: str,
    method: str,
    query_no: int,
    repeat: int,
    latency: float,
    recall: float,
    ef_search=100,
):
    return {
        "phase": phase,
        "method": method,
        "filter_name": SPEC.name,
        "ef_search": ef_search,
        "query_no": query_no,
        "repeat": repeat,
        "search_latency_ms": latency,
        "recall_at_10": recall,
        "valid": True,
        "error": "",
    }


class Amazon10mMatchedRecallBaselineTests(unittest.TestCase):
    def test_formal_default_table_matches_the_exact_gt_and_fbin_id_space(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.table, "amazon_grocery_reviews_10m_pgvector")
        self.assertEqual(args.faiss_index, DEFAULT_FAISS_INDEX)
        self.assertEqual(args.faiss_index_manifest, DEFAULT_FAISS_INDEX_MANIFEST)
        self.assertIn("q10200", DEFAULT_TRUTH.name)
        self.assertIn("q10200", DEFAULT_TRUTH_MANIFEST.name)
        self.assertEqual(args.truth_manifest, DEFAULT_TRUTH_MANIFEST)
        self.assertEqual(args.protocol, CURRENT_PROTOCOL)
        self.assertEqual(args.calibration_workload_csv, DEFAULT_CALIBRATION_WORKLOAD)
        self.assertEqual(args.measurement_workload_csv, DEFAULT_MEASUREMENT_WORKLOAD)
        self.assertEqual(args.measurement_repeats, CURRENT_MEASUREMENT_REPEATS)
        self.assertEqual(parse_methods(args.methods), FORMAL_METHODS)
        self.assertIn("m32_efc200", args.faiss_index.name)
        self.assertEqual(args.calibration_query_offset, 20)
        self.assertEqual(args.calibration_queries, 80)
        self.assertEqual(args.final_query_offset, 100)
        self.assertEqual(args.final_queries, 100)
        self.assertEqual(DEFAULT_EF_SEARCH[:7], (20, 40, 60, 80, 100, 150, 200))
        self.assertEqual(DEFAULT_EF_SEARCH[-1], 100000)
        self.assertEqual(args.target_recalls, "0.9,0.95,0.99")
        self.assertEqual(args.calibration_selection_policy, "lcb_then_max_recall")
        self.assertEqual(formal_protocol_errors(args), [])

    def test_formal_protocol_is_fail_closed_and_dry_run_has_no_io(self) -> None:
        args = build_parser().parse_args([
            "--dry-run", "--filters-csv", "/missing/filters.csv",
            "--truth-csv", "/missing/truth.csv", "--fbin", "/missing/vectors.fbin",
            "--faiss-index", "/missing/index.faiss",
            "--faiss-index-manifest", "/missing/index.manifest.json",
        ])
        with mock.patch.object(Path, "open", side_effect=AssertionError("file read")):
            payload = dry_run_payload(args)
        self.assertTrue(payload["formal_protocol_valid"])
        self.assertEqual(payload["protocol"], CURRENT_PROTOCOL)
        self.assertEqual(payload["calibration"]["requests"], 200)
        self.assertEqual(payload["calibration"]["repeats"], 2)
        self.assertEqual(payload["final"]["requests"], 10_000)
        self.assertEqual(payload["final"]["repeats"], 3)
        self.assertTrue(payload["checkpoint"]["resumable"])
        self.assertEqual(payload["calibration_selection_policy"], "lcb_then_max_recall")
        self.assertIn("LCB95", payload["target_selection"])
        self.assertIn("--calibration-selection-policy", payload["command"])
        self.assertIn("--truth-manifest", payload["command"])
        self.assertEqual(payload["methods"][FAISS_METHOD]["index"], {
            "m": 32, "ef_construction": 200, "manifest_required": True,
        })
        self.assertIn("untruncated", payload["methods"][FAISS_METHOD]["semantics"])
        self.assertIn("end-to-end", payload["methods"][SQL_FIRST_CONTROL_METHOD]["timing"])
        self.assertIn("direct exact SQL", payload["methods"][SQL_FIRST_PLANNER_METHOD]["semantics"])
        self.assertIn("enable_seqscan=off", payload["methods"][SQL_FIRST_FORCED_METHOD]["semantics"])
        self.assertFalse(payload["throughput"]["qps_reported"])
        self.assertFalse(payload["throughput"]["latency_reciprocal_used_as_qps"])
        altered = build_parser().parse_args(["--calibration-repeats", "1"])
        self.assertTrue(any("calibration_repeats" in item for item in formal_protocol_errors(altered)))
        altered_measurement = build_parser().parse_args(
            ["--measurement-repeats", "2"]
        )
        self.assertTrue(
            any(
                "measurement_repeats" in item
                for item in formal_protocol_errors(altered_measurement)
            )
        )

        legacy = build_parser().parse_args(
            ["--protocol", LEGACY_PROTOCOL, "--dry-run"]
        )
        legacy_payload = dry_run_payload(legacy)
        self.assertEqual(
            legacy_payload["calibration"],
            {"query_nos": [20, 99], "queries": 80, "repeats": 2},
        )
        self.assertEqual(
            legacy_payload["final"],
            {"query_nos": [100, 199], "queries": 100, "repeats": 5},
        )
        self.assertFalse(legacy_payload["checkpoint"]["resumable"])

    def test_methods_are_independently_selectable_and_canonicalized(self) -> None:
        selected = parse_methods(
            f"{FAISS_METHOD},{SQL_FIRST_PLANNER_METHOD},{FAISS_METHOD}"
        )
        self.assertEqual(selected, (SQL_FIRST_PLANNER_METHOD, FAISS_METHOD))
        args = build_parser().parse_args(
            ["--methods", SQL_FIRST_FORCED_METHOD, "--dry-run"]
        )
        payload = dry_run_payload(args)
        self.assertTrue(payload["formal_protocol_valid"])
        self.assertEqual(payload["requested_methods"], [SQL_FIRST_FORCED_METHOD])
        with self.assertRaisesRegex(Exception, "unknown methods"):
            parse_methods("sql_first,not_a_method")

    def test_current_workloads_are_frozen_disjoint_mixed_traces(self) -> None:
        calibration = load_workload(
            DEFAULT_CALIBRATION_WORKLOAD,
            expected_rows=CURRENT_CALIBRATION_REQUESTS,
            expected_split="calibration",
            filter_names=set(
                row["filter_name"]
                for row in csv.DictReader(
                    DEFAULT_CALIBRATION_WORKLOAD.open(
                        newline="", encoding="utf-8"
                    )
                )
            ),
        )
        measurement = load_workload(
            DEFAULT_MEASUREMENT_WORKLOAD,
            expected_rows=CURRENT_MEASUREMENT_REQUESTS,
            expected_split="measurement",
            filter_names=set(request.filter_name for request in calibration),
        )
        validate_workload_pair(calibration, measurement)
        self.assertEqual(
            [request.request_no for request in calibration],
            list(range(CURRENT_CALIBRATION_REQUESTS)),
        )
        self.assertEqual(
            [request.request_no for request in measurement],
            list(range(CURRENT_MEASUREMENT_REQUESTS)),
        )
        self.assertEqual(
            sha256_file(DEFAULT_CALIBRATION_WORKLOAD),
            FORMAL_CALIBRATION_WORKLOAD_SHA256,
        )
        self.assertEqual(
            sha256_file(DEFAULT_MEASUREMENT_WORKLOAD),
            FORMAL_MEASUREMENT_WORKLOAD_SHA256,
        )
        self.assertEqual(len({request.filter_name for request in calibration}), 14)
        self.assertEqual(len({request.filter_name for request in measurement}), 14)

    def test_workload_loader_rejects_noncontiguous_or_overlapping_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "trace.csv"
            write_csv(path, [{
                "request_no": 1,
                "query_no": 10,
                "query_id": 20,
                "filter_name": SPEC.name,
                "trace_cycle": 0,
                "split": "calibration",
            }])
            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_workload(
                    path,
                    expected_rows=1,
                    expected_split="calibration",
                    filter_names={SPEC.name},
                )
        request = WorkloadRequest(0, 10, 20, SPEC.name, 0, "calibration")
        overlap = WorkloadRequest(0, 11, 20, SPEC.name, 0, "measurement")
        with self.assertRaisesRegex(ValueError, "query_id"):
            validate_workload_pair([request], [overlap])

    def test_checkpoint_cell_resumes_only_a_valid_contiguous_prefix(self) -> None:
        requests = [
            WorkloadRequest(i, 100 + i, 200 + i, SPEC.name, 0, "measurement")
            for i in range(3)
        ]
        cell_id = "final__sql_first_exact__r0"
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            prefix_row = {
                "request_no": 0,
                "query_no": 100,
                "query_id": 200,
                "filter_name": SPEC.name,
                "repeat": 0,
                "checkpoint_cell": cell_id,
                "valid": True,
                "error": "",
            }
            write_csv(checkpoint_dir / f"{cell_id}.csv", [prefix_row])
            executed: list[int] = []

            def execute(request, _position):
                executed.append(request.request_no)
                return {
                    "request_no": request.request_no,
                    "query_no": request.query_no,
                    "query_id": request.query_id,
                    "filter_name": request.filter_name,
                    "repeat": 0,
                    "valid": True,
                    "error": "",
                }

            rows, record = execute_checkpointed_cell(
                checkpoint_dir=checkpoint_dir,
                cell_id=cell_id,
                requests=requests,
                repeat=0,
                checkpoint_every=1,
                resume=True,
                execute_request=execute,
            )
            self.assertEqual(executed, [1, 2])
            self.assertEqual(len(rows), 3)
            self.assertEqual(record["resumed_rows"], 1)
            self.assertTrue(record["complete"])
            self.assertEqual(record["sha256"], sha256_file(Path(record["path"])))

            rows[0]["query_id"] = 999
            with self.assertRaisesRegex(ValueError, "contiguous workload prefix"):
                validate_checkpoint_prefix(
                    rows, requests, cell_id=cell_id, repeat=0
                )

    def test_sql_only_slice_does_not_require_faiss_calibration(self) -> None:
        final_rows = [
            measurement_row(
                phase="final", method=SQL_FIRST_PLANNER_METHOD, spec=SPEC,
                query_no=100, query_id=20, repeat=repeat, schedule_position=1,
                block_no=repeat, ef_search=NA, result_ids=list(range(10)),
                truth_ids=list(range(10)), latency_ms=10.0,
            )
            for repeat in range(2)
        ]
        methods = (SQL_FIRST_PLANNER_METHOD,)
        summary = final_summary_table(
            final_rows, [SPEC], [0.9], {}, [100], repeats=2,
            bootstrap_samples=20, bootstrap_seed=57, methods=methods,
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["status"], "valid")
        self.assertFalse(summary[0]["matched_recall_comparison_valid"])
        self.assertEqual(
            artifact_validation_errors([], summary, [SPEC], [100], [0.9], methods),
            [],
        )
        gate = completion_gate([], summary, [SPEC], [100], [0.9], methods)
        self.assertEqual(gate["expected_calibration_cells"], 0)
        self.assertTrue(gate["requested_slice_complete"])
        self.assertFalse(gate["publishable_matched_recall"])

    def test_current_input_hash_gate_rejects_legacy_gt_and_m16(self) -> None:
        current = {
            "filters": FORMAL_FILTERS_SHA256,
            "truth": FORMAL_TRUTH_SHA256,
            "truth_manifest": FORMAL_TRUTH_MANIFEST_SHA256,
            "query_cohort_csv": FORMAL_QUERY_COHORT_SHA256,
            "query_cohort_manifest": FORMAL_QUERY_COHORT_MANIFEST_SHA256,
            "calibration_workload": FORMAL_CALIBRATION_WORKLOAD_SHA256,
            "measurement_workload": FORMAL_MEASUREMENT_WORKLOAD_SHA256,
            "fbin": FORMAL_FBIN_SHA256,
            "faiss_index": FORMAL_FAISS_INDEX_SHA256,
            "faiss_index_manifest": FORMAL_FAISS_INDEX_MANIFEST_SHA256,
        }
        self.assertEqual(
            formal_input_hash_errors(
                current, FORMAL_METHODS, CURRENT_PROTOCOL
            ),
            {},
        )
        stale = {**current, "truth": "old-q200", "faiss_index": "old-m16"}
        errors = formal_input_hash_errors(
            stale, FORMAL_METHODS, CURRENT_PROTOCOL
        )
        self.assertEqual(set(errors), {"truth", "faiss_index"})
        flags = artifact_validity_flags(
            [],
            {
                "requested_slice_complete": True,
                "publishable_matched_recall": True,
            },
            formal_provenance_valid=False,
        )
        self.assertTrue(flags["artifact_valid"])
        self.assertFalse(flags["paper_eligible"])
        subset = build_parser().parse_args(["--filter-names", "popular_ge1000"])
        self.assertTrue(any("filter_names" in item for item in formal_protocol_errors(subset)))

    def test_result_membership_gate_rejects_ids_outside_bitmap(self) -> None:
        bitmap = np.zeros(2, dtype=np.uint8)
        set_bitmap_ids(bitmap, [1, 4, 9], total_rows=10)
        self.assertEqual(result_membership_errors(bitmap, [1, 9]), [])
        self.assertEqual(result_membership_errors(bitmap, [1, 3, 9]), [3])

    def test_formal_faiss_metadata_requires_m32_l2(self) -> None:
        class FakeHnsw:
            def __init__(self, m: int):
                self.m = m
                self.efConstruction = 200

            def nb_neighbors(self, level: int) -> int:
                return 2 * self.m if level == 0 else self.m

        class FakeIndex:
            ntotal = 10
            d = 4
            metric_type = 1

            def __init__(self, m: int):
                self.hnsw = FakeHnsw(m)

        class FakeFaiss:
            METRIC_L2 = 1

        metadata = faiss_index_metadata(FakeIndex(32), FakeFaiss, 10, 4)
        self.assertEqual(metadata["m"], 32)
        self.assertEqual(metadata["ef_construction"], 200)
        self.assertEqual(metadata["level0_neighbors"], 64)
        with self.assertRaisesRegex(ValueError, "not formal M32"):
            faiss_index_metadata(FakeIndex(16), FakeFaiss, 10, 4)
        bad_ef = FakeIndex(32)
        bad_ef.hnsw.efConstruction = 128
        with self.assertRaisesRegex(ValueError, "efConstruction=200"):
            faiss_index_metadata(bad_ef, FakeFaiss, 10, 4)

    def test_faiss_build_manifest_binds_m32_index_and_fbin_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "index.manifest.json"
            index_identity = {"sha256": "a" * 64, "bytes": 4096}
            fbin_identity = {"sha256": "b" * 64, "bytes": 8192}
            payload = {
                "artifact": "faiss_hnsw_index_build",
                "status": "complete",
                "artifact_valid": True,
                "configuration": {
                    "m": 32,
                    "ef_construction": 200,
                    "rows": 10,
                    "dimensions": 4,
                },
                "index_contract": {
                    "type": "IndexHNSWFlat",
                    "metric": "l2",
                    "m": 32,
                    "ef_construction": 200,
                    "rows": 10,
                    "dimensions": 4,
                },
                "output_identity": {"sha256": "a" * 64, "size_bytes": 4096},
                "inputs": {
                    "fbin": {"sha256": "b" * 64, "size_bytes": 8192},
                },
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            verified = verify_faiss_build_manifest(
                manifest_path, index_identity, fbin_identity, 10, 4
            )
            self.assertTrue(verified["artifact_valid"])

            payload["configuration"]["m"] = 16
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest contract failed"):
                verify_faiss_build_manifest(
                    manifest_path, index_identity, fbin_identity, 10, 4
                )

            payload["configuration"]["m"] = 32
            payload["output_identity"]["sha256"] = "c" * 64
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest contract failed"):
                verify_faiss_build_manifest(
                    manifest_path, index_identity, fbin_identity, 10, 4
                )

    def test_truth_manifest_binds_truth_fbin_and_unique_query_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = root / "truth.csv"
            fbin = root / "vectors.fbin"
            cohort = root / "cohort.csv"
            truth.write_bytes(b"truth")
            fbin.write_bytes(b"fbin")
            cohort.write_bytes(b"cohort")
            cohort_manifest = root / "cohort_manifest.json"
            cohort_manifest.write_text(json.dumps({
                "artifact_valid": True,
                "selection": {"disjoint": True},
                "outputs": {"cohort_csv": {"path": str(cohort), "sha256": sha256_file(cohort)}},
            }), encoding="utf-8")
            truth_manifest = root / "truth_manifest.json"
            truth_manifest.write_text(json.dumps({
                "artifact_valid": True,
                "method": "exact_filtered_l2_tie_aware",
                "k": 10,
                "rows": 10_000_000,
                "filters": 14,
                "query_ids_disjoint": True,
                "calibration": {"queries": 100},
                "final": {"queries": 100},
                "outputs": {"truth_csv": {"path": str(truth), "sha256": sha256_file(truth)}},
                "inputs": {
                    "fbin": {"path": str(fbin), "sha256": sha256_file(fbin)},
                    "postgres": {
                        "table": "amazon_grocery_reviews_10m_pgvector",
                        "table_oid": 1,
                        "table_relfilenode": 2,
                        "rows": 10_000_000,
                    },
                },
                "query_source": {
                    "cohort_csv": {"path": str(cohort), "sha256": sha256_file(cohort)},
                    "manifest": {"path": str(cohort_manifest), "sha256": sha256_file(cohort_manifest)},
                },
            }), encoding="utf-8")
            truth_identity = {"sha256": sha256_file(truth), "bytes": truth.stat().st_size}
            fbin_identity = {"sha256": sha256_file(fbin), "bytes": fbin.stat().st_size}
            bound = verify_truth_manifest(truth_manifest, truth_identity, fbin_identity)
            self.assertTrue(bound["artifact_valid"])
            self.assertEqual(bound["query_cohort_csv"]["sha256"], sha256_file(cohort))

            cohort.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "query cohort CSV identity"):
                verify_truth_manifest(truth_manifest, truth_identity, fbin_identity)

    def test_target_eligibility_uses_query_mean_and_reports_lcb_only(self):
        rows = [
            measured_row("calibration", "faiss_allowlist", 20, 0, 1.0, 1.0, 100),
            measured_row("calibration", "faiss_allowlist", 20, 1, 1.0, 0.0, 100),
        ]
        table, selected = calibration_table(
            rows, [SPEC], [100], [0.5], [20], repeats=2,
            bootstrap_samples=100, bootstrap_seed=57,
        )
        self.assertTrue(table[0]["eligible"])
        self.assertEqual(selected[(SPEC.name, 0.5)], 100)
        self.assertIn("recall_lcb95", table[0])

    def test_conservative_selection_prefers_lcb_confirmed_configuration(self):
        rows = [
            measured_row(
                "calibration", "faiss_allowlist", query_no, 0, 1.0,
                recall, 100,
            )
            for query_no, recall in ((20, 1.0), (21, 0.8))
        ]
        rows.extend(
            measured_row(
                "calibration", "faiss_allowlist", query_no, 0, 2.0,
                1.0, 200,
            )
            for query_no in (20, 21)
        )

        table, selected = calibration_table(
            rows, [SPEC], [100, 200], [0.9], [20, 21], repeats=1,
            bootstrap_samples=100, bootstrap_seed=57,
            selection_policy="lcb_then_max_recall",
        )

        self.assertEqual(selected[(SPEC.name, 0.9)], 200)
        winner = next(row for row in table if row["selected"])
        self.assertTrue(winner["lcb95_eligible"])
        self.assertEqual(winner["selection_fallback"], "none")

    def test_calibration_selects_each_filter_from_its_own_mixed_trace_queries(self):
        spec_b = FilterSpec(
            name="filter_b",
            target_rate="5.0%",
            predicate="helpful_vote >= 2",
            expected_rows=10,
            actual_pct=5.0,
        )
        rows = []
        for repeat in range(2):
            rows.append(
                measured_row(
                    "calibration",
                    FAISS_METHOD,
                    10,
                    repeat,
                    2.0,
                    1.0,
                    100,
                )
            )
            row_b = measured_row(
                "calibration",
                FAISS_METHOD,
                20,
                repeat,
                3.0,
                1.0,
                100,
            )
            row_b["filter_name"] = spec_b.name
            rows.append(row_b)
        table, selected = calibration_table(
            rows,
            [SPEC, spec_b],
            [100],
            [0.9],
            {SPEC.name: [10], spec_b.name: [20]},
            repeats=2,
            bootstrap_samples=20,
            bootstrap_seed=57,
            selection_policy="lcb_then_max_recall",
        )
        self.assertEqual(
            selected,
            {(SPEC.name, 0.9): 100, (spec_b.name, 0.9): 100},
        )
        self.assertTrue(all(row["status"] == "valid" for row in table))

    def test_formal_lcb_policy_never_selects_a_mean_only_fallback(self):
        rows = [
            measured_row("calibration", "faiss_allowlist", query_no, 0, 1.0, recall, 100)
            for query_no, recall in ((20, 1.0), (21, 0.8))
        ]
        rows.extend(
            measured_row("calibration", "faiss_allowlist", query_no, 0, 2.0, 0.85, 200)
            for query_no in (20, 21)
        )
        table, selected = calibration_table(
            rows, [SPEC], [100, 200], [0.9], [20, 21], repeats=1,
            bootstrap_samples=100, bootstrap_seed=57,
            selection_policy="lcb_then_max_recall",
        )
        self.assertEqual(selected, {})
        self.assertTrue(all(row["calibration_ladder_complete"] for row in table))
        self.assertTrue(all(row["outcome"] == "lcb95_unattained_on_grid" for row in table))
        self.assertTrue(all(row["selection_fallback"] == "no_lcb95_qualified_config" for row in table))
        self.assertFalse(any(row["selected"] for row in table))

    def test_direct_exact_sql_and_materialized_control_are_distinct(self):
        sql = exact_sql("samegraph_insert", SPEC.predicate, 10)
        control = materialized_exact_sql("samegraph_insert", SPEC.predicate, 10)

        self.assertNotIn("MATERIALIZED", sql)
        self.assertIn("FROM samegraph_insert", sql)
        self.assertIn("id <> %s", sql)
        self.assertIn(
            "ORDER BY (embedding <-> %s::vector) + 0.0::double precision, id",
            sql,
        )
        self.assertIn("LIMIT 10", sql)
        self.assertIn("WITH filtered AS MATERIALIZED", control)
        self.assertEqual(
            exact_sql_for_method(
                SQL_FIRST_CONTROL_METHOD, "samegraph_insert", SPEC.predicate, 10
            ),
            control,
        )
        self.assertEqual(
            exact_sql_for_method(
                SQL_FIRST_PLANNER_METHOD, "samegraph_insert", SPEC.predicate, 10
            ),
            sql,
        )
        self.assertEqual(
            exact_sql_for_method(
                SQL_FIRST_FORCED_METHOD, "samegraph_insert", SPEC.predicate, 10
            ),
            sql,
        )

        scalar_plan = {
            "Node Type": "Limit",
            "Plans": [
                {
                    "Node Type": "CTE Scan",
                    "Plans": [
                        {"Node Type": "Index Scan", "Index Name": "helpful_vote_idx"}
                    ],
                }
            ],
        }
        self.assertEqual(
            assert_no_hnsw_index(scalar_plan, ["samegraph_insert_hnsw"]),
            {"helpful_vote_idx"},
        )

        hnsw_plan = {
            "Node Type": "Index Scan",
            "Index Name": "samegraph_insert_hnsw",
        }
        with self.assertRaisesRegex(RuntimeError, "used HNSW"):
            assert_no_hnsw_index(hnsw_plan, ["public.samegraph_insert_hnsw"])
        self.assertEqual(
            assert_scalar_index_plan(
                scalar_plan,
                ["samegraph_insert_hnsw"],
                ["public.helpful_vote_idx"],
            ),
            {"helpful_vote_idx"},
        )
        with self.assertRaisesRegex(RuntimeError, "did not use"):
            assert_scalar_index_plan(
                {"Node Type": "Seq Scan"},
                ["samegraph_insert_hnsw"],
                ["public.helpful_vote_idx"],
            )

    def test_streaming_bitmap_sets_faiss_little_endian_bits(self):
        bitmap = np.zeros(3, dtype=np.uint8)

        count = set_bitmap_ids(bitmap, [0, 1, 7, 8, 19], total_rows=20)

        self.assertEqual(count, 5)
        self.assertEqual(bitmap.tolist(), [0b10000011, 0b00000001, 0b00001000])
        self.assertTrue(all(bitmap_contains(bitmap, value) for value in [0, 1, 7, 8, 19]))
        self.assertFalse(bitmap_contains(bitmap, 18))
        with self.assertRaisesRegex(ValueError, "outside"):
            set_bitmap_ids(bitmap, [20], total_rows=20)

    def test_allowlist_id_stream_is_complete_and_untruncated(self):
        sql = allowlist_id_sql("samegraph_insert", SPEC.predicate)
        self.assertIn("SELECT id FROM samegraph_insert", sql)
        self.assertIn("embedding_valid", sql)
        self.assertNotIn("ORDER BY", sql.upper())
        self.assertNotIn("LIMIT", sql.upper())
        self.assertNotIn("OFFSET", sql.upper())

    def test_allowlist_builder_fetches_batches_and_keeps_bitmap_backing(self):
        class FakeCursor:
            def __init__(self):
                self.batches = [[(0,), (7,)], [(8,), (19,)], []]
                self.fetchmany_calls = 0
                self.sql = ""

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, sql):
                self.sql = sql

            def fetchmany(self, size):
                self.fetchmany_calls += 1
                self.fetch_size = size
                return self.batches.pop(0)

        class FakeConnection:
            def __init__(self):
                self.server_cursor = FakeCursor()
                self.control_cursor = FakeCursor()
                self.control_cursor.rowcount = 4

            def transaction(self):
                return nullcontext()

            def cursor(self, name=None):
                if name is None:
                    return self.control_cursor
                self.cursor_name = name
                return self.server_cursor

        class FakeSelector:
            def __init__(self, size, bitmap):
                self.size = size
                self.bitmap = bitmap

        class FakeFaiss:
            IDSelectorBitmap = FakeSelector

            @staticmethod
            def swig_ptr(bitmap):
                return bitmap

        conn = FakeConnection()
        small_spec = FilterSpec(SPEC.name, SPEC.target_rate, SPEC.predicate, 4, SPEC.actual_pct)

        allow_list = build_allow_list(conn, FakeFaiss, "samegraph_insert", small_spec, 20, 2)

        self.assertTrue(allow_list.valid)
        self.assertEqual(allow_list.rows, 4)
        self.assertEqual(allow_list.bitmap_bytes, 3)
        self.assertEqual(allow_list.selector.size, 20)
        self.assertIs(allow_list.selector.bitmap, allow_list.bitmap)
        self.assertEqual(conn.server_cursor.fetchmany_calls, 3)
        self.assertEqual(conn.server_cursor.fetch_size, 2)
        self.assertIn("CREATE TEMP TABLE", conn.control_cursor.sql)
        self.assertIn("SELECT id FROM samegraph_insert", conn.control_cursor.sql)
        self.assertIn("SELECT id FROM allowlist_materialized_", conn.server_cursor.sql)
        self.assertGreaterEqual(allow_list.full_setup_ms, 0.0)
        self.assertGreaterEqual(allow_list.server_execution_ms, 0.0)
        self.assertGreaterEqual(allow_list.row_transfer_ms, 0.0)
        self.assertGreaterEqual(allow_list.bitmap_construction_ms, 0.0)
        self.assertGreaterEqual(allow_list.selector_construction_ms, 0.0)
        self.assertTrue(bitmap_contains(allow_list.bitmap, 19))

    def test_faiss_search_passes_bitmap_selector_to_hnsw_parameters(self):
        class Params:
            efSearch = 0
            sel = None

        class FakeFaiss:
            SearchParametersHNSW = Params

        class FakeIndex:
            def search(self, query, k, params):
                self.query = query
                self.k = k
                self.params = params
                return np.zeros((1, k), dtype=np.float32), np.asarray([[4, 2, -1]])

        index = FakeIndex()
        selector = object()

        ids, latency_ms = search_faiss(
            index,
            FakeFaiss,
            np.asarray([1.0, 2.0], dtype=np.float32),
            selector,
            ef_search=500,
            k=3,
        )

        self.assertEqual(ids, [4, 2])
        self.assertGreater(latency_ms, 0.0)
        self.assertEqual(index.params.efSearch, 500)
        self.assertIs(index.params.sel, selector)
        self.assertEqual(index.query.shape, (1, 2))

    def test_faiss_search_requests_extra_row_and_excludes_query_id(self):
        class Params:
            efSearch = 0
            sel = None

        class FakeFaiss:
            SearchParametersHNSW = Params

        class FakeIndex:
            def search(self, query, k, params):
                self.k = k
                return np.zeros((1, k), dtype=np.float32), np.asarray([[7, 4, 2, 1]])

        index = FakeIndex()
        ids, _ = search_faiss(
            index,
            FakeFaiss,
            np.asarray([1.0, 2.0], dtype=np.float32),
            object(),
            ef_search=500,
            k=3,
            query_id=7,
        )
        self.assertEqual(index.k, 4)
        self.assertEqual(ids, [4, 2, 1])

    def test_full_setup_search_e2e_is_continuously_timed(self):
        bitmap = np.zeros(2, dtype=np.uint8)
        set_bitmap_ids(bitmap, [1, 2], total_rows=10)
        allow_list = AllowList(
            selector=object(),
            bitmap=bitmap,
            rows=2,
            build_ms=5.0,
            bitmap_bytes=2,
            valid=True,
            server_execution_ms=2.0,
            row_transfer_ms=1.5,
            bitmap_construction_ms=1.0,
            selector_construction_ms=0.5,
            full_setup_ms=5.0,
        )
        module = "experiments.hybrid_vector_db.scripts.amazon10m_matched_recall_baselines"
        with (
            mock.patch(f"{module}.build_allow_list", return_value=allow_list),
            mock.patch(f"{module}.search_faiss", return_value=([1, 2], 0.25)),
            mock.patch(f"{module}.time.perf_counter", side_effect=[10.0, 10.007]),
        ):
            row = full_setup_search_row(
                conn=object(),
                faiss_module=object(),
                index=object(),
                table="samegraph_insert",
                spec=SPEC,
                total_rows=10,
                fetch_rows=2,
                query=np.asarray([1.0, 2.0], dtype=np.float32),
                query_no=100,
                query_id=7,
                ef_search=500,
                k=2,
            )

        self.assertTrue(row["valid"])
        self.assertAlmostEqual(row["full_setup_plus_search_e2e_ms"], 7.0)
        self.assertEqual(row["allowlist_full_setup_ms"], 5.0)
        self.assertEqual(row["cached_ann_search_ms"], 0.25)
        self.assertNotEqual(
            row["full_setup_plus_search_e2e_ms"],
            row["allowlist_full_setup_ms"] + row["cached_ann_search_ms"],
        )

    def test_formal_summary_separates_faiss_setup_cached_search_and_full_e2e(self):
        final_rows = []
        truth_ids = list(range(10))
        for repeat in range(2):
            final_rows.append(measurement_row(
                phase="final", method=SQL_FIRST_CONTROL_METHOD, spec=SPEC,
                query_no=100, query_id=20, repeat=repeat, schedule_position=1,
                block_no=repeat, ef_search=NA, result_ids=truth_ids,
                truth_ids=truth_ids, latency_ms=24.0,
            ))
            final_rows.append(measurement_row(
                phase="final", method=SQL_FIRST_PLANNER_METHOD, spec=SPEC,
                query_no=100, query_id=20, repeat=repeat, schedule_position=2,
                block_no=repeat, ef_search=NA, result_ids=truth_ids,
                truth_ids=truth_ids, latency_ms=20.0,
            ))
            final_rows.append(measurement_row(
                phase="final", method=SQL_FIRST_FORCED_METHOD, spec=SPEC,
                query_no=100, query_id=20, repeat=repeat, schedule_position=3,
                block_no=repeat, ef_search=NA, result_ids=truth_ids,
                truth_ids=truth_ids, latency_ms=18.0,
            ))
            final_rows.append(measurement_row(
                phase="final", method=FAISS_METHOD, spec=SPEC, query_no=100,
                query_id=20, repeat=repeat, schedule_position=4, block_no=repeat,
                ef_search=500, result_ids=truth_ids, truth_ids=truth_ids,
                latency_ms=5.0,
            ))
        allow_list = AllowList(
            selector=object(), bitmap=np.zeros(3, dtype=np.uint8), rows=20,
            build_ms=13.0, bitmap_bytes=3, valid=True,
            server_execution_ms=4.0, row_transfer_ms=5.0,
            bitmap_construction_ms=3.0, selector_construction_ms=1.0,
            full_setup_ms=13.0,
        )
        summary = final_summary_table(
            final_rows, [SPEC], [0.9], {(SPEC.name, 0.9): 500}, [100],
            repeats=2, bootstrap_samples=20, bootstrap_seed=57,
            allow_lists={SPEC.name: allow_list},
            calibration_outcomes={(SPEC.name, 0.9): "selected_pending_final"},
            methods=FORMAL_METHODS,
            setup_search_rows=[{
                "phase": "setup_search_e2e", "filter_name": SPEC.name,
                "ef_search": 500, "continuous_full_e2e_ms": 19.0,
                "valid": True, "error": "",
            }],
        )
        self.assertEqual({row["method"] for row in summary}, set(FORMAL_METHODS))
        shapes = {
            row["method"]: row["sql_shape"]
            for row in final_rows
            if row["repeat"] == 0 and row["method"] != FAISS_METHOD
        }
        self.assertEqual(shapes[SQL_FIRST_CONTROL_METHOD], "materialized_cte_control")
        self.assertEqual(shapes[SQL_FIRST_PLANNER_METHOD], "direct_exact_sql")
        self.assertEqual(shapes[SQL_FIRST_FORCED_METHOD], "direct_exact_sql")
        faiss_row = next(row for row in summary if row["method"] == FAISS_METHOD)
        self.assertEqual(faiss_row["allowlist_sql_materialization_ms_one_time"], 4.0)
        self.assertEqual(faiss_row["allowlist_row_transfer_ms_one_time"], 5.0)
        self.assertEqual(faiss_row["allowlist_bitmap_build_ms_one_time"], 3.0)
        self.assertEqual(faiss_row["cached_allowlist_search_mean_ms"], 5.0)
        self.assertEqual(faiss_row["continuous_full_e2e_mean_ms"], 19.0)
        self.assertEqual(faiss_row["continuous_full_e2e_samples"], 1)

    def test_current_faiss_targets_keep_distinct_cached_and_full_e2e_samples(self):
        rows = []
        truth_ids = list(range(10))
        for target, cached_ms, full_ms in ((0.90, 5.0, 20.0), (0.95, 8.0, 25.0)):
            for repeat in range(2):
                row = measurement_row(
                    phase="final",
                    method=FAISS_METHOD,
                    spec=SPEC,
                    query_no=100,
                    query_id=20,
                    repeat=repeat,
                    schedule_position=repeat,
                    block_no=0,
                    ef_search=500,
                    result_ids=truth_ids,
                    truth_ids=truth_ids,
                    latency_ms=cached_ms,
                    target_recall=target,
                    request_no=0,
                    trace_cycle=0,
                )
                row.update({
                    "continuous_full_e2e_ms": full_ms,
                    "continuous_full_e2e_valid": True,
                    "continuous_recall_at_10": 1.0,
                    "per_request_allowlist_sql_materialization_ms": 10.0,
                    "per_request_allowlist_row_transfer_ms": 4.0,
                    "per_request_allowlist_bitmap_build_ms": 2.0,
                    "per_request_allowlist_selector_construction_ms": 1.0,
                    "per_request_allowlist_full_setup_ms": 17.0,
                    "per_request_full_path_search_ms": full_ms - 17.0,
                })
                rows.append(row)
        summary = final_summary_table(
            rows,
            [SPEC],
            [0.90, 0.95],
            {(SPEC.name, 0.90): 500, (SPEC.name, 0.95): 500},
            {SPEC.name: [100]},
            repeats=2,
            bootstrap_samples=20,
            bootstrap_seed=57,
            calibration_outcomes={
                (SPEC.name, 0.90): "selected_pending_final",
                (SPEC.name, 0.95): "selected_pending_final",
            },
            methods=(FAISS_METHOD,),
        )
        by_target = {float(row["target_recall"]): row for row in summary}
        self.assertEqual(by_target[0.90]["cached_allowlist_search_mean_ms"], 5.0)
        self.assertEqual(by_target[0.95]["cached_allowlist_search_mean_ms"], 8.0)
        self.assertEqual(by_target[0.90]["continuous_full_e2e_mean_ms"], 20.0)
        self.assertEqual(by_target[0.95]["continuous_full_e2e_mean_ms"], 25.0)
        self.assertEqual(by_target[0.90]["continuous_full_e2e_samples"], 2)
        self.assertTrue(by_target[0.90]["continuous_full_e2e_complete"])
        self.assertEqual(by_target[0.90]["continuous_recall_lcb95"], 1.0)
        self.assertEqual(
            by_target[0.90][
                "per_request_allowlist_sql_materialization_mean_ms"
            ],
            10.0,
        )
        self.assertEqual(
            by_target[0.90]["per_request_full_path_search_mean_ms"], 3.0
        )

    def test_tie_aware_recall_accepts_different_boundary_ids(self):
        vectors = np.asarray(
            [[0.0], [0.0], [1.0], [-1.0], [2.0]], dtype=np.float32
        )
        entry = TruthEntry(0, 0, SPEC.name, "calibration", (1, 2), 20, 1.0, 1e-9, True)
        self.assertEqual(tie_aware_recall_at_k([1, 3], 0, vectors, entry, 2), 1.0)
        self.assertEqual(tie_aware_recall_at_k([1, 4], 0, vectors, entry, 2), 0.5)

    def test_balanced_order_rotates_each_config_through_each_position(self):
        configs = [250, 500, 1000]
        orders = [balanced_order(configs, block, seed=57) for block in range(3)]

        self.assertTrue(all(set(order) == set(configs) for order in orders))
        for position in range(3):
            self.assertEqual({order[position] for order in orders}, set(configs))

    def test_truth_loader_requires_disjoint_complete_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.csv"
            fieldnames = [
                "query_no",
                "query_id",
                "filter_name",
                "predicate",
                "method",
                "exact_filtered_topk_ids",
                "recall_at_10_exact_filtered",
                "filtered_rows",
                "k",
                "kth_distance_sq",
                "tie_tolerance",
                "self_excluded",
                "query_split",
                "candidate_validity_predicate",
                "query_validity_predicate",
                "candidate_rows",
            ]
            with path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "query_no": 0,
                        "query_id": 3,
                        "filter_name": SPEC.name,
                        "predicate": SPEC.predicate,
                        "method": "pre_filter_exact",
                        "exact_filtered_topk_ids": "1,2",
                        "recall_at_10_exact_filtered": 1.0,
                        "filtered_rows": 20,
                        "k": 2,
                        "kth_distance_sq": 1.0,
                        "tie_tolerance": 1e-9,
                        "self_excluded": True,
                        "query_split": "calibration",
                        "candidate_validity_predicate": "embedding_valid",
                        "query_validity_predicate": "embedding_valid",
                        "candidate_rows": 20,
                    }
                )
                writer.writerow(
                    {
                        "query_no": 1,
                        "query_id": 4,
                        "filter_name": SPEC.name,
                        "predicate": SPEC.predicate,
                        "method": "pre_filter_exact",
                        "exact_filtered_topk_ids": "5,6",
                        "recall_at_10_exact_filtered": 1.0,
                        "filtered_rows": 20,
                        "k": 2,
                        "kth_distance_sq": 1.0,
                        "tie_tolerance": 1e-9,
                        "self_excluded": True,
                        "query_split": "final",
                        "candidate_validity_predicate": "embedding_valid",
                        "query_validity_predicate": "embedding_valid",
                        "candidate_rows": 20,
                    }
                )

            truth, query_ids = load_truth(path, [SPEC], [0], [1], k=2)

            self.assertEqual(query_ids, {0: 3, 1: 4})
            self.assertEqual(truth[(SPEC.name, 1)].ids, (5, 6))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_truth(path, [SPEC], [0], [0], k=2)

    def test_truth_loader_rejects_legacy_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.csv"
            path.write_text(
                "query_no,query_id,filter_name,method,exact_filtered_topk_ids,candidates\n"
                "0,3,filter_a,pre_filter_exact,1\"\"2,20\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "retired schema"):
                load_truth(path, [SPEC], [0], [], k=2)

    def test_calibration_uses_lcb_and_then_lowest_mean_latency(self):
        rows = []
        for query_no in range(4):
            for repeat in range(2):
                rows.append(
                    measured_row(
                        "calibration", "faiss_allowlist", query_no, repeat, 20.0, 1.0, 100
                    )
                )
                rows.append(
                    measured_row(
                        "calibration", "faiss_allowlist", query_no, repeat, 10.0, 0.95, 200
                    )
                )
                rows.append(
                    measured_row(
                        "calibration", "faiss_allowlist", query_no, repeat, 1.0, 0.90, 300
                    )
                )

        table, selected = calibration_table(
            rows,
            [SPEC],
            [100, 200, 300],
            [0.95, 0.99],
            [0, 1, 2, 3],
            repeats=2,
            bootstrap_samples=100,
            bootstrap_seed=57,
        )

        self.assertEqual(selected[(SPEC.name, 0.95)], 200)
        self.assertEqual(selected[(SPEC.name, 0.99)], 100)
        selected_rows = [row for row in table if row["selected"]]
        self.assertEqual({(row["target_recall"], row["ef_search"]) for row in selected_rows}, {(0.95, 200), (0.99, 100)})

    def test_incomplete_measurement_grid_is_invalid_with_na_metrics(self):
        rows = [
            measured_row("calibration", "faiss_allowlist", 0, 0, 1.0, 1.0),
            measured_row("calibration", "faiss_allowlist", 0, 1, 1.0, 1.0),
            measured_row("calibration", "faiss_allowlist", 1, 0, 1.0, 1.0),
        ]

        summary = aggregate_measurements(
            rows,
            phase="calibration",
            method="faiss_allowlist",
            filter_name=SPEC.name,
            ef_search=100,
            query_nos=[0, 1],
            repeats=2,
            bootstrap_samples=10,
            bootstrap_seed=57,
        )

        self.assertEqual(summary["status"], "invalid")
        self.assertEqual(summary["missing_pairs"], 1)
        self.assertEqual(summary["recall_mean"], NA)
        self.assertEqual(summary["latency_mean_ms"], NA)

    def test_complete_summary_reports_p99_and_query_mean_ci(self):
        rows = [
            measured_row("final", "faiss_allowlist", query_no, repeat, 1.0 + query_no + repeat, 1.0)
            for query_no in (0, 1)
            for repeat in (0, 1)
        ]
        summary = aggregate_measurements(
            rows,
            phase="final",
            method="faiss_allowlist",
            filter_name=SPEC.name,
            ef_search=100,
            query_nos=[0, 1],
            repeats=2,
            bootstrap_samples=100,
            bootstrap_seed=57,
        )
        self.assertEqual(summary["status"], "valid")
        self.assertIn("latency_p99_ms", summary)
        self.assertIn("latency_query_mean_ci95_low_ms", summary)

    def test_final_summary_rejects_missing_matched_pair(self):
        final_rows = []
        truth_ids = list(range(10))
        for repeat in range(2):
            final_rows.append(
                measurement_row(
                    phase="final",
                    method="sql_first_exact",
                    spec=SPEC,
                    query_no=100,
                    query_id=10,
                    repeat=repeat,
                    schedule_position=1,
                    block_no=repeat,
                    ef_search=NA,
                    result_ids=truth_ids,
                    truth_ids=truth_ids,
                    latency_ms=20.0,
                )
            )
        final_rows.append(
            measurement_row(
                phase="final",
                method="faiss_allowlist",
                spec=SPEC,
                query_no=100,
                query_id=10,
                repeat=0,
                schedule_position=2,
                block_no=0,
                ef_search=500,
                result_ids=truth_ids,
                truth_ids=truth_ids,
                latency_ms=5.0,
            )
        )

        summary = final_summary_table(
            final_rows,
            [SPEC],
            [0.95],
            {(SPEC.name, 0.95): 500},
            [100],
            repeats=2,
            bootstrap_samples=10,
            bootstrap_seed=57,
            allow_lists={
                SPEC.name: AllowList(object(), np.zeros(3, dtype=np.uint8), 20, 3.0, 3, True)
            },
        )

        self.assertEqual(len(summary), 2)
        sql_row = next(row for row in summary if row["method"] == SQL_FIRST_CONTROL_METHOD)
        faiss_row = next(row for row in summary if row["method"] == FAISS_METHOD)
        self.assertEqual(sql_row["status"], "valid")
        self.assertEqual(sql_row["search_latency_mean_ms"], 20.0)
        self.assertEqual(faiss_row["status"], "invalid")
        self.assertEqual(faiss_row["search_latency_mean_ms"], NA)
        self.assertTrue(all(not row["matched_recall_comparison_valid"] for row in summary))
        self.assertTrue(all(row["speedup_vs_sql_first_exact"] == NA for row in summary))

    def test_complete_ladder_below_target_is_valid_unattainable_without_faiss_final(self):
        calibration = [
            measured_row("calibration", "faiss_allowlist", query_no, repeat, 2.0, 0.5, ef)
            for ef in (100, 200)
            for query_no in (0, 1)
            for repeat in (0, 1)
        ]
        table, selected = calibration_table(
            calibration, [SPEC], [100, 200], [0.9], [0, 1], 2, 50, 57
        )
        self.assertEqual(selected, {})
        self.assertTrue(all(row["outcome"] == "unattainable_on_grid" for row in table))
        self.assertTrue(all(row["calibration_ladder_complete"] for row in table))
        self.assertTrue(all(row["max_ef_search"] == 200 for row in table))

        final_rows = [
            measurement_row(
                phase="final", method="sql_first_exact", spec=SPEC, query_no=query_no,
                query_id=10 + query_no, repeat=repeat, schedule_position=1, block_no=repeat,
                ef_search=NA, result_ids=list(range(10)), truth_ids=list(range(10)), latency_ms=20.0,
            )
            for query_no in (100, 101)
            for repeat in (0, 1)
        ]
        summary = final_summary_table(
            final_rows, [SPEC], [0.9], selected, [100, 101], 2, 50, 57,
            calibration_outcomes={(SPEC.name, 0.9): "unattainable_on_grid"},
        )
        faiss = next(row for row in summary if row["method"] == "faiss_allowlist")
        self.assertEqual(faiss["outcome"], "unattainable_on_grid")
        self.assertEqual(faiss["status"], "valid")
        self.assertEqual(faiss["samples"], 0)
        self.assertEqual(faiss["expected_samples"], 0)
        self.assertEqual(faiss["missing_pairs"], 0)
        self.assertEqual(faiss["recall_mean"], NA)
        self.assertFalse(faiss["matched_recall_comparison_valid"])

    def test_incomplete_max_ef_is_not_unattainable(self):
        rows = [
            measured_row("calibration", "faiss_allowlist", query_no, repeat, 2.0, 0.5, ef)
            for ef in (100, 200)
            for query_no in (0, 1)
            for repeat in (0, 1)
            if not (ef == 200 and query_no == 1 and repeat == 1)
        ]
        table, selected = calibration_table(rows, [SPEC], [100, 200], [0.9], [0, 1], 2, 50, 57)
        self.assertEqual(selected, {})
        self.assertTrue(all(row["outcome"] == "calibration_invalid" for row in table))
        self.assertTrue(any(row["status"] == "invalid" for row in table))
        self.assertTrue(any(not row["calibration_ladder_complete"] for row in table))

    def test_selected_complete_final_target_miss_is_valid_unconfirmed(self):
        calibration = [
            measured_row("calibration", "faiss_allowlist", query_no, repeat, 2.0, 1.0, 100)
            for query_no in (0, 1)
            for repeat in (0, 1)
        ]
        table, selected = calibration_table(calibration, [SPEC], [100], [0.9], [0, 1], 2, 50, 57)
        final_rows = []
        for query_no in (100, 101):
            for repeat in (0, 1):
                final_rows.append(measurement_row(
                    phase="final", method="sql_first_exact", spec=SPEC, query_no=query_no,
                    query_id=10 + query_no, repeat=repeat, schedule_position=1, block_no=repeat,
                    ef_search=NA, result_ids=list(range(10)), truth_ids=list(range(10)), latency_ms=20.0,
                ))
                final_rows.append(measurement_row(
                    phase="final", method="faiss_allowlist", spec=SPEC, query_no=query_no,
                    query_id=10 + query_no, repeat=repeat, schedule_position=2, block_no=repeat,
                    ef_search=100, result_ids=list(range(8)), truth_ids=list(range(10)), latency_ms=5.0,
                ))
        summary = final_summary_table(
            final_rows, [SPEC], [0.9], selected, [100, 101], 2, 50, 57,
            calibration_outcomes={(SPEC.name, 0.9): "selected_pending_final"},
        )
        self.assertTrue(all(row["status"] == "valid" for row in summary))
        outcomes = {row["method"]: row["outcome"] for row in summary}
        self.assertEqual(outcomes[SQL_FIRST_CONTROL_METHOD], "selected_and_confirmed")
        self.assertEqual(outcomes[FAISS_METHOD], "selected_but_final_unconfirmed")
        self.assertTrue(all(not row["matched_recall_comparison_valid"] for row in summary))
        self.assertFalse(artifact_validation_errors(table, summary, [SPEC], [100], [0.9]))

    def test_completion_gate_separates_finished_slice_from_publishable_release(self):
        calibration = [
            measured_row("calibration", "faiss_allowlist", 0, 0, 2.0, 0.5, 100)
        ]
        table, selected = calibration_table(
            calibration, [SPEC], [100], [0.9], [0], 1, 20, 57,
            selection_policy="lcb_then_max_recall",
        )
        final_rows = [measurement_row(
            phase="final", method="sql_first_exact", spec=SPEC, query_no=100,
            query_id=10, repeat=0, schedule_position=1, block_no=0, ef_search=NA,
            result_ids=list(range(10)), truth_ids=list(range(10)), latency_ms=20.0,
        )]
        summary = final_summary_table(
            final_rows, [SPEC], [0.9], selected, [100], 1, 20, 57,
            calibration_outcomes={(SPEC.name, 0.9): "lcb95_unattained_on_grid"},
        )
        gate = completion_gate(table, summary, [SPEC], [100], [0.9])
        self.assertTrue(gate["requested_slice_complete"])
        self.assertFalse(gate["full_release_complete"])
        self.assertFalse(gate["publishable_matched_recall"])

    def test_current_release_gate_requires_per_request_full_e2e_and_recall(self):
        calibration = [{
            "filter_name": SPEC.name,
            "target_recall": 0.9,
            "ef_search": 100,
            "status": "valid",
        }]
        common = {
            "filter_name": SPEC.name,
            "target_recall": 0.9,
            "status": "valid",
            "outcome": "selected_and_confirmed",
            "matched_recall_comparison_valid": True,
            "target_confirmed_in_final": True,
        }
        summary = [
            {**common, "method": SQL_FIRST_PLANNER_METHOD},
            {
                **common,
                "method": FAISS_METHOD,
                "continuous_full_e2e_samples": 6,
                "continuous_full_e2e_expected_samples": 6,
                "continuous_full_e2e_errors": 0,
                "continuous_full_e2e_complete": True,
                "continuous_recall_lcb95": 0.91,
            },
        ]
        module = (
            "experiments.hybrid_vector_db.scripts."
            "amazon10m_matched_recall_baselines"
        )
        with (
            mock.patch(f"{module}.FORMAL_FILTER_NAMES", (SPEC.name,)),
            mock.patch(f"{module}.DEFAULT_EF_SEARCH", (100,)),
            mock.patch(f"{module}.DEFAULT_TARGETS", (0.9,)),
        ):
            gate = completion_gate(
                calibration,
                summary,
                [SPEC],
                [100],
                [0.9],
                (SQL_FIRST_PLANNER_METHOD, FAISS_METHOD),
                CURRENT_PROTOCOL,
            )
            self.assertTrue(gate["full_release_complete"])
            summary[1]["continuous_recall_lcb95"] = 0.89
            rejected = completion_gate(
                calibration,
                summary,
                [SPEC],
                [100],
                [0.9],
                (SQL_FIRST_PLANNER_METHOD, FAISS_METHOD),
                CURRENT_PROTOCOL,
            )
            self.assertFalse(rejected["full_release_complete"])

    def test_duplicate_or_error_calibration_cannot_be_unattainable(self):
        rows = [
            measured_row("calibration", "faiss_allowlist", query_no, repeat, 2.0, 0.5, 100)
            for query_no in (0, 1)
            for repeat in (0, 1)
        ]
        duplicate = dict(rows[0])
        rows.append(duplicate)
        table, _ = calibration_table(rows, [SPEC], [100], [0.9], [0, 1], 2, 50, 57)
        self.assertTrue(all(row["outcome"] == "calibration_invalid" for row in table))
        rows[-1] = {**rows[-1], "valid": False, "error": "search failed"}
        table, _ = calibration_table(rows, [SPEC], [100], [0.9], [0, 1], 2, 50, 57)
        self.assertTrue(all(row["outcome"] == "calibration_invalid" for row in table))

    def test_finalizer_rejects_tamper_and_preserves_outputs_until_atomic_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filters = root / "filters.csv"
            filters.write_text(
                "filter_name,target_rate,predicate,count,actual_pct\n"
                "filter_a,10.0%,helpful_vote >= 1,20,10.0\n", encoding="utf-8"
            )
            truth = root / "truth.csv"
            fields = ["query_no", "query_id", "filter_name", "predicate", "method", "exact_filtered_topk_ids",
                      "recall_at_10_exact_filtered", "filtered_rows", "k", "kth_distance_sq", "tie_tolerance",
                      "self_excluded", "query_split", "candidate_validity_predicate",
                      "query_validity_predicate", "candidate_rows"]
            with truth.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                for query_no, split in ((0, "calibration"), (1, "final")):
                    writer.writerow({"query_no": query_no, "query_id": query_no + 2,
                        "filter_name": SPEC.name, "predicate": SPEC.predicate, "method": "pre_filter_exact", "exact_filtered_topk_ids": "5,6",
                        "recall_at_10_exact_filtered": 1.0, "filtered_rows": 20, "k": 2, "kth_distance_sq": 1.0,
                        "tie_tolerance": 1e-9, "self_excluded": True, "query_split": split,
                        "candidate_validity_predicate": "embedding_valid",
                        "query_validity_predicate": "embedding_valid", "candidate_rows": 20})
            fbin, faiss = root / "vectors.fbin", root / "index.faiss"
            fbin.write_bytes(b"vectors")
            faiss.write_bytes(b"index")
            raw = root / "raw.csv"
            calibration = root / "calibration.csv"
            final = root / "final.csv"
            raw_rows = [measured_row("calibration", "faiss_allowlist", 0, repeat, 2.0, 0.5, ef)
                        for ef in (100, 200) for repeat in (0, 1)]
            final_rows = [measurement_row(
                phase="final", method="sql_first_exact", spec=SPEC, query_no=1, query_id=3,
                repeat=repeat, schedule_position=1, block_no=repeat, ef_search=NA,
                result_ids=list(range(10)), truth_ids=list(range(10)), latency_ms=20.0,
            ) for repeat in (0, 1)]
            raw_rows.extend(final_rows)
            write_csv(raw, raw_rows)
            table, _ = calibration_table(raw_rows, [SPEC], [100, 200], [0.9], [0], 2, 20, 57)
            write_csv(calibration, table)
            write_csv(final, final_rows)
            manifest = root / "legacy.json"
            payload = {
                "status": "invalid", "finished_at_utc": "2026-07-18T00:00:00+00:00",
                "args": {"filter_names": [SPEC.name], "calibration_query_offset": 0,
                    "calibration_queries": 1, "calibration_repeats": 2, "final_query_offset": 1,
                    "final_queries": 1, "final_repeats": 2, "target_recalls": "0.9",
                    "ef_search_values": "100,200", "k": 2, "bootstrap_samples": 20, "bootstrap_seed": 57,
                    "tag": "only-this-shard", "out_dir": str(root), "overwrite": True},
                "inputs": {"filters": {"path": str(filters)}, "truth": {"path": str(truth)},
                    "fbin": {"path": str(fbin)}, "faiss_index": {"path": str(faiss)},
                    "runner": {"path": "runner.py", "sha256": "a" * 64}},
                "postgres": {"table_oid": 1}, "outputs": {"raw": str(raw), "calibration": str(calibration), "final": str(final)},
                "query_splits": {"calibration_query_nos": [0], "final_query_nos": [1]},
                "source_hashes": {"truth": sha256_file(truth), "fbin": sha256_file(fbin), "faiss": sha256_file(faiss)},
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            prefix = root / "finalized"
            summary_path = root / "finalized_summary.csv"
            manifest_path = root / "finalized_manifest.json"
            summary_path.write_text("old-summary", encoding="utf-8")
            manifest_path.write_text("old-manifest", encoding="utf-8")
            fbin.write_bytes(b"tampered")
            with self.assertRaisesRegex(FinalizationFailure, "hash changed"):
                finalize_existing(manifest, raw, calibration, final, prefix)
            self.assertEqual(summary_path.read_text(encoding="utf-8"), "old-summary")
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), "old-manifest")
            fbin.write_bytes(b"vectors")
            outputs = finalize_existing(manifest, raw, calibration, final, prefix)
            self.assertTrue(outputs["summary"].is_file())
            finalized = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertTrue(finalized["artifact_valid"])
            self.assertEqual(finalized["status"], "complete")
            self.assertEqual(finalized["software_versions"]["measurement_runner_sha256"], "a" * 64)
            self.assertNotIn("filter_names", finalized["run_contract"])
            self.assertNotIn("tag", finalized["run_contract"])
            self.assertNotIn("out_dir", finalized["run_contract"])
            self.assertTrue(Path(finalized["outputs"]["summary"]["path"]).is_absolute())
            self.assertTrue(Path(finalized["outputs"]["manifest"]).is_absolute())
            write_csv(calibration, table[:-1])
            with self.assertRaisesRegex(FinalizationFailure, "calibration key coverage"):
                finalize_existing(manifest, raw, calibration, final, root / "rejected")

    def test_baseline_artifact_flags_do_not_promote_a_diagnostic_slice(self):
        partial = artifact_validity_flags(
            [],
            {
                "requested_slice_complete": False,
                "publishable_matched_recall": False,
            },
        )
        self.assertTrue(partial["diagnostic_valid"])
        self.assertFalse(partial["artifact_valid"])
        self.assertFalse(partial["paper_eligible"])
        complete = artifact_validity_flags(
            [],
            {
                "requested_slice_complete": True,
                "publishable_matched_recall": True,
            },
        )
        self.assertTrue(complete["artifact_valid"])
        self.assertTrue(complete["paper_eligible"])
        invalid = artifact_validity_flags(
            ["missing final cell"],
            {
                "requested_slice_complete": True,
                "publishable_matched_recall": True,
            },
        )
        self.assertFalse(invalid["diagnostic_valid"])
        self.assertFalse(invalid["artifact_valid"])


if __name__ == "__main__":
    unittest.main()
