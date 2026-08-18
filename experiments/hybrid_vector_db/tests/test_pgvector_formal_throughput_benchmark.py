from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from experiments.hybrid_vector_db.scripts import pgvector_formal_throughput_benchmark as runner


@dataclass(frozen=True)
class FakeTruth:
    filtered_rows: int


def query_ids(first: int = 0, last: int = 200) -> dict[int, int]:
    return {query_no: 100_000 + query_no for query_no in range(first, last)}


def filter_spec(name: str = "f", rows: int = 10) -> runner.FilterSpec:
    return runner.FilterSpec(name, "rating = 5", ("sql:rating = 5",), rows, 1.0)


def write_query_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_q10200_fixture(directory: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    path = directory / "amazon10m_unique_embedding_query_cohort_q10200.csv"
    rows = [
        {
            "query_no": query_no,
            "query_id": 50_000 + query_no,
            "query_split": "calibration" if query_no < 100 else "final",
        }
        for query_no in range(10_200)
    ]
    write_query_csv(path, rows)
    manifest = {
        "artifact_valid": True,
        "candidate_validity_predicate": runner.DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
        "selection": {
            "calibration": {"queries": 100},
            "final": {"queries": 10_100},
            "disjoint": True,
        },
        "outputs": {
            "cohort_csv": {
                "path": str(path),
                "sha256": runner.sha256_file(path),
                "rows": 10_200,
            }
        },
    }
    manifest_path = directory / "amazon10m_unique_embedding_query_cohort_q10200_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest_path, rows


def write_matched_recall_fixture(
    directory: Path,
    *,
    filters: list[runner.FilterSpec],
    truth_csv: Path,
    filters_csv: Path,
    targets: tuple[float, ...] = runner.FORMAL_TARGETS,
    status: str = "complete",
    complete: bool = True,
    policy: str = "lcb_then_max_recall",
    lcb95: float = 0.995,
    guidance_filter_strategy: str = "safe_guided",
) -> Path:
    selected_path = directory / "matched_selected.csv"
    rows: list[dict[str, object]] = []
    for item in filters:
        for method, mode in runner.MODE_BY_METHOD.items():
            for target in targets:
                rows.append(
                    {
                        "target_recall": target,
                        "target_met_in_calibration": True,
                        "target_confirmed_in_calibration": True,
                        "target_lcb95_met_in_calibration": True,
                        "selection_status": "selected",
                        "filter_name": item.name,
                        "mode": mode,
                        "config": f"{method}-{target}",
                        "ef_search": 1000,
                        "max_scan_tuples": 200_000,
                        "scan_mem_multiplier": 32.0,
                        "iterative_scan": "strict_order" if method == "sqlens_d1" else "off",
                        "guided_collect_target": 1000,
                        "traversal_guided_target": 11,
                        "samples": 200,
                        "errors": 0,
                        "recall_mean": max(target, 0.995),
                        "recall_lcb95": lcb95,
                        "rows_complete": True,
                        "truth_self_excluded": True,
                        "plan_gate_passed": True,
                        "guidance_filter_strategy": guidance_filter_strategy,
                        "latency_mean_ms": 5.0,
                    }
                )
    runner.write_csv_atomic(selected_path, rows)
    selected_metadata = {
        "path": str(selected_path),
        "sha256": runner.sha256_file(selected_path),
        "bytes": selected_path.stat().st_size,
        "row_count": len(rows),
    }
    database = {
        "candidate_validity_predicate": runner.DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
        "candidate_validity_predicate_sha256": "predicate-sha",
        "sqlens_build_id": runner.REQUIRED_SQLENS_BUILD_PREFIXES[0] + "test",
        "relations": {
            runner.DEFAULT_TABLE: {"oid": 10, "relfilenode": 20},
            runner.DEFAULT_SOURCE_INDEX: {
                "oid": 11,
                "relfilenode": 21,
                "valid": True,
                "ready": True,
                "candidate_validity_predicate_matches": True,
                "candidate_validity_predicate_sha256": "predicate-sha",
            },
        },
        "query_table": {
            "name": runner.DEFAULT_TABLE,
            "oid": 10,
            "relfilenode": 20,
            "row_count": 10_000_000,
            "columns": ["id:bigint", "embedding:vector"],
        },
    }
    manifest = {
        "status": status,
        "matrix_complete": complete,
        "measurement_complete": complete,
        "comparison_valid": complete,
        "targets": list(targets),
        "modes": list(runner.MODE_BY_METHOD.values()),
        "calibration_policy": {
            "calibration_selection_policy": policy,
            "selection": (
                "lowest latency among recall_lcb95-qualified configurations"
                if policy == "lcb_then_max_recall"
                else "lowest latency among mean recall-qualified configurations; LCB report-only"
            ),
            "stop_metric": "recall_lcb95" if policy == "lcb_then_max_recall" else "recall_mean",
        },
        "run_spec": {
            "truth_sha256": runner.sha256_file(truth_csv),
            "filters_sha256": runner.sha256_file(filters_csv),
            "args": {
                "truth_csv": str(truth_csv.resolve()),
                "filters_csv": str(filters_csv.resolve()),
                "insertion_table": runner.DEFAULT_TABLE,
                "insertion_index": runner.DEFAULT_SOURCE_INDEX,
                "candidate_validity_predicate": runner.DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
                "guidance_filter_strategy": guidance_filter_strategy,
                "traversal_guided_prioritization": guidance_filter_strategy == "traversal_guided",
                "traversal_guided_burst": 8,
                "calibration_selection_policy": policy,
            },
            "sqlens_runtime_provenance": {
                "loaded_vector_sqlens_build_id": runner.REQUIRED_SQLENS_BUILD_PREFIXES[0] + "test",
                "loaded_vector_so_sha256": "a" * 64,
            },
            "database": database,
        },
        "outputs": {"selected": selected_metadata},
    }
    manifest_path = directory / "matched_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_parser_defaults_are_the_formal_matrix_and_dry_run_is_explicit() -> None:
    args = runner.create_argument_parser().parse_args([])
    payload = runner.dry_run_payload(args)

    assert payload["target_recalls"] == [0.90, 0.95, 0.99]
    assert payload["clients"] == [1, 4, 8, 16, 32, 64]
    assert payload["methods"] == ["stock", "sqlens_d1"]
    assert payload["internal_calibration_enabled"] is False
    assert payload["configuration_source"].startswith("independently_audited")
    assert payload["workload_requests"] == 10_000
    assert payload["measurement_query_cohort"] == "q200..q10199"
    assert payload["measurement_unique_query_vectors"] == 10_000
    assert payload["measurement_trace_replay"] is False
    assert payload["debug_replay_query_cohort"] == "q100..q199"
    assert payload["debug_replay_only"] is True
    assert payload["formal_ready"] is False
    assert payload["formal_prerequisites"]["status"] == "prerequisite_missing"
    assert "matched_recall_manifest" in payload["formal_prerequisites"]["missing"]
    assert payload["throughput_definition"] == "completed_queries / measurement_wall_clock_seconds"
    assert payload["measurement_repeats"] == 6
    assert payload["independent_connection_per_client"] is True
    assert payload["throughput_ci_unit"] == "measurement_repeat_completed_wall_pair"
    assert payload["measurement_protocol"]["cache"]["reset_between_repeats"] is False
    assert payload["measurement_protocol"]["request_order"]["distinct_seed_per_repeat"] is True
    assert payload["measurement_protocol"]["guidance"]["safe_guided_preserves_graph_traversal"] is True
    assert payload["evaluation_scope"] == "representative_filters"
    assert args.guidance_filter_strategy == "safe_guided"


def test_formal_matrix_requires_target_099_but_debug_remains_backward_compatible() -> None:
    args = runner.create_argument_parser().parse_args(["--target-recalls", "0.90,0.95"])
    with pytest.raises(runner.BenchmarkContractError, match="0.90/0.95/0.99"):
        runner.validate_formal_args(args)

    args.allow_nonformal_debug = True
    assert runner.validate_formal_args(args)[0] == [0.90, 0.95]


def test_nonformal_request_count_cannot_be_labelled_formal() -> None:
    args = runner.create_argument_parser().parse_args(["--requests", "400"])
    with pytest.raises(runner.BenchmarkContractError, match="formal runs require requests=10000"):
        runner.validate_formal_args(args)

    args.allow_nonformal_debug = True
    payload = runner.dry_run_payload(args)
    assert payload["formal"] is False
    assert payload["workload_requests"] == 400
    assert payload["measurement_unique_query_vectors"] == 10_000


def test_nonformal_diagnostic_can_complete_without_becoming_a_paper_artifact() -> None:
    args = runner.create_argument_parser().parse_args([])
    coverage = {"passed": True}
    formal = runner.artifact_validity_flags(
        args, diagnostic_complete=True, completion_coverage=coverage
    )
    assert formal == {
        "diagnostic_valid": True,
        "formal_protocol_complete": True,
        "paper_eligible": True,
        "artifact_valid": True,
    }

    args.allow_nonformal_debug = True
    diagnostic = runner.artifact_validity_flags(
        args, diagnostic_complete=True, completion_coverage=coverage
    )
    assert diagnostic == {
        "diagnostic_valid": True,
        "formal_protocol_complete": False,
        "paper_eligible": False,
        "artifact_valid": False,
    }


def test_formal_artifact_gate_requires_completion_coverage() -> None:
    args = runner.create_argument_parser().parse_args([])
    flags = runner.artifact_validity_flags(
        args, diagnostic_complete=True, completion_coverage={"passed": False}
    )
    assert flags["diagnostic_valid"] is True
    assert flags["formal_protocol_complete"] is False
    assert flags["artifact_valid"] is False


def test_formal_scope_is_explicit_and_rejects_cherry_picked_filter_subsets() -> None:
    all_filters = runner.load_filters(runner.DEFAULT_FILTERS)
    args = runner.create_argument_parser().parse_args([])
    representative = runner.resolve_evaluation_filters(args, all_filters)
    assert [item.name for item in representative] == list(runner.REPRESENTATIVE_FILTERS)

    args.evaluation_scope = "full_matrix"
    assert [item.name for item in runner.resolve_evaluation_filters(args, all_filters)] == [
        item.name for item in all_filters
    ]
    args.filter_names = ["popular_ge1000"]
    with pytest.raises(runner.BenchmarkContractError, match="forbids --filter-names"):
        runner.resolve_evaluation_filters(args, all_filters)

    args.evaluation_scope = "representative_filters"
    with pytest.raises(runner.BenchmarkContractError, match="canonical"):
        runner.resolve_evaluation_filters(args, all_filters)

    args.allow_nonformal_debug = True
    narrowed = runner.resolve_evaluation_filters(args, all_filters)
    assert [item.name for item in narrowed] == ["popular_ge1000"]


def test_formal_resume_is_allowed_but_cannot_overwrite() -> None:
    args = runner.create_argument_parser().parse_args(["--resume"])
    assert runner.validate_formal_args(args)[0] == list(runner.FORMAL_TARGETS)
    args.overwrite = True
    with pytest.raises(runner.BenchmarkContractError, match="mutually exclusive"):
        runner.validate_formal_args(args)


def test_formal_artifact_name_binds_the_guidance_strategy() -> None:
    args = runner.create_argument_parser().parse_args(
        ["--guidance-filter-strategy", "traversal_guided"]
    )
    with pytest.raises(runner.BenchmarkContractError, match="output path must name"):
        runner.validate_formal_args(args)
    args.out = Path("/tmp/amazon10m_throughput_traversal_guided_raw.csv")
    assert runner.validate_formal_args(args)[0] == list(runner.FORMAL_TARGETS)


def test_replay_is_exactly_q100_q199_for_100_cycles() -> None:
    workload = runner.build_replay_workload(query_ids())

    assert len(workload.requests) == 10_000
    assert workload.unique_query_vectors == 100
    assert workload.trace_replay is True
    assert workload.query_cohort == "q100..q199"
    assert {item.query_no for item in workload.requests} == set(range(100, 200))
    assert [item.query_no for item in workload.requests[:100]] == list(range(100, 200))
    assert workload.requests[9_999].trace_cycle == 99
    assert len({item.query_id for item in workload.requests}) == 100


def test_replay_requires_100_distinct_heldout_query_vectors() -> None:
    ids = query_ids()
    ids[199] = ids[198]
    with pytest.raises(runner.BenchmarkContractError, match="100 unique query vectors"):
        runner.build_replay_workload(ids)


def test_q10200_cohort_returns_10000_unique_measurement_queries_and_checks_actual_id_overlap() -> None:
    with TemporaryDirectory() as temporary:
        path, manifest, rows = write_q10200_fixture(Path(temporary))
        calibration_ids = range(1000, 1100)
        workload = runner.load_true_query_workload(
            path,
            calibration_ids,
            query_manifest=manifest,
        )
        assert workload.trace_replay is False
        assert workload.unique_query_vectors == 10_000
        assert workload.query_cohort == "q200..q10199"
        assert workload.source_kind == "q10200_cohort_measurement_split"
        assert workload.source_sha256 == runner.sha256_file(path)
        assert {item.query_no for item in workload.requests} == set(range(200, 10_200))

        rows[-1]["query_id"] = rows[-2]["query_id"]
        write_query_csv(path, rows)
        with pytest.raises(runner.BenchmarkContractError, match="unique query_id"):
            runner.load_true_query_workload(path, calibration_ids, query_manifest=None)

        rows[-1]["query_id"] = 50_000 + 10_199
        write_query_csv(path, rows)
        with pytest.raises(runner.BenchmarkContractError, match="overlap matched-recall selection"):
            runner.load_true_query_workload(
                path,
                [*calibration_ids, int(rows[200]["query_id"])],
                query_manifest=None,
            )


def test_q200_selection_prefix_is_disjoint_from_formal_measurement() -> None:
    with TemporaryDirectory() as temporary:
        path, _, rows = write_q10200_fixture(Path(temporary))
        for row in rows:
            query_no = int(row["query_no"])
            row["query_id"] = 1_000 + query_no
        write_query_csv(path, rows)

        q100_calibration_ids = {1_000 + query_no for query_no in range(100)}
        q200_selection_ids = {1_000 + query_no for query_no in range(200)}
        workload = runner.load_true_query_workload(
            path,
            q100_calibration_ids,
            query_manifest=None,
        )
        assert workload.unique_query_vectors == 10_000
        workload = runner.load_true_query_workload(
            path,
            q200_selection_ids,
            query_manifest=None,
        )
        assert workload.query_cohort == "q200..q10199"


def test_discovered_real_query_file_is_preferred_over_replay() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        path, manifest, rows = write_q10200_fixture(directory)
        ids = {number: 50_000 + number for number in range(10_200)}

        workload = runner.choose_workload(
            ids,
            None,
            directory,
            runner.FORMAL_REQUESTS,
            calibration_query_ids=range(1000, 1100),
            query_manifest=manifest,
        )
        assert workload.source_kind == "q10200_cohort_measurement_split"
        assert workload.source_path == str(path.resolve())


def test_q100_replay_is_debug_only_and_formal_mode_reports_missing_prerequisite() -> None:
    ids = query_ids(0, 200)
    with TemporaryDirectory() as temporary:
        with pytest.raises(runner.BenchmarkContractError, match="prerequisite missing"):
            runner.choose_workload(
                ids,
                None,
                Path(temporary),
                runner.FORMAL_REQUESTS,
                formal=True,
            )
        workload = runner.choose_workload(
            ids,
            None,
            Path(temporary),
            runner.FORMAL_REQUESTS,
            formal=False,
        )
        assert workload.trace_replay is True
        assert workload.source_kind == "heldout_q100_q199_trace_replay"


def test_measurement_truth_manifest_binds_the_q10200_cohort_and_split_contract() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        query_path, query_manifest, _ = write_q10200_fixture(directory)
        truth_path = directory / "truth.csv"
        truth_path.write_text("query_no,query_id\n100,50100\n", encoding="utf-8")
        manifest_path = directory / "truth_manifest.json"
        payload = {
            "artifact_valid": True,
            "self_excluded": True,
            "calibration": {"queries": 100},
            "final": {"queries": 10_100},
            "query_ids_disjoint": True,
            "validity_contract": {
                "candidate_validity_predicate": runner.DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
            },
            "candidate_universe": {
                "candidate_validity_predicate": runner.DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
                "rows": 123,
            },
            "query_source": {
                "cohort_csv": {
                    "path": str(query_path),
                    "sha256": runner.sha256_file(query_path),
                }
            },
            "outputs": {"truth_csv": {"sha256": runner.sha256_file(truth_path)}},
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        evidence = runner.verify_measurement_truth_manifest(
            truth_path,
            manifest_path,
            runner.DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
            123,
            query_path,
        )
        assert evidence["measurement_queries"] == 10_000
        assert evidence["query_ids_disjoint"] is True

        payload["query_source"]["cohort_csv"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(runner.BenchmarkContractError, match="cohort SHA256"):
            runner.verify_measurement_truth_manifest(
                truth_path,
                manifest_path,
                runner.DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
                123,
                query_path,
            )


def test_workload_query_ids_must_match_exact_gt_mapping() -> None:
    workload = runner.build_replay_workload(query_ids())
    ids = query_ids()
    ids[150] += 1
    with pytest.raises(runner.BenchmarkContractError, match="disagrees with exact GT"):
        runner.validate_workload_query_mapping(workload, ids)


def test_truth_coverage_enforces_disjoint_cohorts_and_filter_counts() -> None:
    item = filter_spec(rows=7)
    truth = {(item.name, query_no): FakeTruth(7) for query_no in range(200)}
    runner.validate_truth_coverage(
        truth,
        query_ids(),
        [item],
        range(100),
        range(100, 200),
    )

    with pytest.raises(runner.BenchmarkContractError, match="overlap"):
        runner.validate_truth_coverage(truth, query_ids(), [item], range(100), range(99, 200))

    truth[(item.name, 150)] = FakeTruth(8)
    with pytest.raises(runner.BenchmarkContractError, match="candidate count mismatch"):
        runner.validate_truth_coverage(
            truth,
            query_ids(),
            [item],
            range(200),
            (),
        )


def test_split_truth_validation_uses_actual_query_ids_when_query_numbers_overlap() -> None:
    item = filter_spec(rows=7)
    calibration_ids = {query_no: 10_000 + query_no for query_no in runner.SELECTION_QUERY_NOS}
    measurement_ids = {
        query_no: 20_000 + query_no for query_no in runner.MEASUREMENT_QUERY_NOS
    }
    calibration_truth = {
        (item.name, query_no): FakeTruth(7) for query_no in runner.SELECTION_QUERY_NOS
    }
    measurement_truth = {
        (item.name, query_no): FakeTruth(7) for query_no in runner.MEASUREMENT_QUERY_NOS
    }
    evidence = runner.validate_calibration_measurement_split(
        calibration_truth,
        calibration_ids,
        measurement_truth,
        measurement_ids,
        [item],
    )
    assert evidence["actual_query_id_disjoint"] is True
    assert evidence["selection_query_numbers"] == 200
    assert evidence["measurement_query_numbers"] == 10_000

    measurement_ids[200] = calibration_ids[0]
    with pytest.raises(runner.BenchmarkContractError, match="query IDs overlap"):
        runner.validate_calibration_measurement_split(
            calibration_truth,
            calibration_ids,
            measurement_truth,
            measurement_ids,
            [item],
        )


def test_truth_manifest_binds_sha_candidate_universe_and_self_exclusion() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        truth = directory / "truth.csv"
        manifest = directory / "truth_manifest.json"
        cohort_manifest = directory / "cohort_manifest.json"
        truth.write_text("query_no\n0\n", encoding="utf-8")
        digest = hashlib.sha256(truth.read_bytes()).hexdigest()
        cohort_payload = {
            "artifact_valid": True,
            "candidate_validity_predicate": "embedding_valid",
            "eligible_query_population": {
                "rows": 7,
                "embedding_valid_rows": 123,
            },
        }
        cohort_manifest.write_text(json.dumps(cohort_payload), encoding="utf-8")
        payload = {
            "artifact_valid": True,
            "self_excluded": True,
            "validity_contract": {"candidate_validity_predicate": "embedding_valid"},
            "eligible_query_population": {"eligible_rows": 123},
            "query_source": {
                "manifest": {
                    "path": str(cohort_manifest),
                    "sha256": runner.sha256_file(cohort_manifest),
                }
            },
            "outputs": {"truth_csv": {"sha256": digest}},
        }
        manifest.write_text(json.dumps(payload), encoding="utf-8")

        evidence = runner.verify_truth_manifest(truth, manifest, "embedding_valid", 123)
        assert evidence["artifact_valid"] is True
        assert evidence["eligible_candidate_rows"] == 123
        assert evidence["candidate_rows_source"].startswith("bound_query_cohort")

        payload["validity_contract"]["candidate_validity_predicate"] = "TRUE"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(runner.BenchmarkContractError, match="predicate mismatch"):
            runner.verify_truth_manifest(truth, manifest, "embedding_valid", 123)

        payload["validity_contract"]["candidate_validity_predicate"] = "embedding_valid"
        payload["candidate_universe"] = {
            "candidate_validity_predicate": "embedding_valid",
            "rows": 123,
        }
        payload["query_source"] = {}
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        evidence = runner.verify_truth_manifest(truth, manifest, "embedding_valid", 123)
        assert evidence["candidate_rows_source"] == "truth_manifest.candidate_universe"


def test_real_valid_embedding_filters_are_loaded_without_synthesis() -> None:
    specs = runner.load_filters(runner.DEFAULT_FILTERS)
    assert len(specs) == 14
    assert specs[0].name == "popular_ge1000"
    assert specs[0].expected_rows == 5_019_997
    assert specs[-1].atoms == ("sql:main_category = 'Grocery'", "sql:review_text_len >= 500")
    assert all("%" not in item.predicate for item in specs)


def test_seeded_method_order_is_deterministic_and_balanced() -> None:
    one = [runner.interleaved_method_order(block, 81) for block in range(8)]
    two = [runner.interleaved_method_order(block, 81) for block in range(8)]
    assert one == two
    assert all(set(order) == set(runner.METHODS) for order in one)
    assert all(one[index] == tuple(reversed(one[index + 1])) for index in range(0, 8, 2))

    schedule = runner.build_measurement_schedule(
        [0.90, 0.95], [1, 4], [filter_spec("a"), filter_spec("b")], 81
    )
    assert len(schedule) == 16
    for offset in range(0, len(schedule), 2):
        pair = schedule[offset : offset + 2]
        assert {item["method"] for item in pair} == set(runner.METHODS)
        assert pair[0]["block_no"] == pair[1]["block_no"]


def load_fixture_bundle(
    directory: Path,
    *,
    status: str = "complete",
    complete: bool = True,
    policy: str = "lcb_then_max_recall",
    lcb95: float = 0.995,
    targets: tuple[float, ...] = runner.FORMAL_TARGETS,
    guidance_filter_strategy: str = "safe_guided",
) -> tuple[Path, Path, Path, runner.FilterSpec]:
    truth_csv = directory / "truth.csv"
    filters_csv = directory / "filters.csv"
    truth_csv.write_text("query_no\n0\n", encoding="utf-8")
    filters_csv.write_text("name,predicate\nf,rating = 5\n", encoding="utf-8")
    item = filter_spec()
    manifest = write_matched_recall_fixture(
        directory,
        filters=[item],
        truth_csv=truth_csv,
        filters_csv=filters_csv,
        targets=targets,
        status=status,
        complete=complete,
        policy=policy,
        lcb95=lcb95,
        guidance_filter_strategy=guidance_filter_strategy,
    )
    return manifest, truth_csv, filters_csv, item


def configure_prerequisite_args(
    directory: Path,
    *,
    matched_manifest: Path,
    truth_csv: Path,
    filters_csv: Path,
) -> argparse.Namespace:
    """Point dry-run at small fixtures while keeping its formal CLI contract."""
    query_csv = directory / "amazon10m_unique_embedding_query_cohort_q10200.csv"
    query_manifest = directory / "amazon10m_unique_embedding_query_cohort_q10200_manifest.json"
    measurement_truth = directory / "amazon_selectivity14_exact_truth_q10200.csv"
    measurement_manifest = directory / "amazon_selectivity14_exact_truth_q10200_manifest.json"
    query_csv.write_text("query_no,query_id\n0,100000\n", encoding="utf-8")
    measurement_truth.write_text("query_no\n0\n", encoding="utf-8")
    for path in (
        directory / "truth_manifest.json",
        query_manifest,
        measurement_manifest,
    ):
        path.write_text(json.dumps({"artifact_valid": True}), encoding="utf-8")

    args = runner.create_argument_parser().parse_args([])
    args.filters_csv = filters_csv
    args.calibration_truth_csv = truth_csv
    args.calibration_truth_manifest = directory / "truth_manifest.json"
    args.measurement_query_file = query_csv
    args.measurement_query_manifest = query_manifest
    args.measurement_truth_csv = measurement_truth
    args.measurement_truth_manifest = measurement_manifest
    args.matched_recall_manifest = matched_manifest
    return args


def patch_static_fixture_gates(item: runner.FilterSpec):
    """Keep prerequisite tests focused on the matched-recall static gate."""
    return mock.patch.multiple(
        runner,
        load_filters=mock.DEFAULT,
        resolve_evaluation_filters=mock.DEFAULT,
        verify_truth_manifest=mock.DEFAULT,
        verify_measurement_query_manifest=mock.DEFAULT,
        verify_measurement_truth_manifest=mock.DEFAULT,
        audit_matched_recall_manifest=mock.DEFAULT,
    )


def prepare_static_fixture_patches(item: runner.FilterSpec):
    patches = patch_static_fixture_gates(item)
    mocks = patches.start()
    mocks["load_filters"].return_value = [item]
    mocks["resolve_evaluation_filters"].return_value = [item]
    mocks["verify_truth_manifest"].return_value = {"artifact_valid": True}
    mocks["verify_measurement_query_manifest"].return_value = {"artifact_valid": True}
    mocks["verify_measurement_truth_manifest"].return_value = {"artifact_valid": True}
    mocks["audit_matched_recall_manifest"].return_value = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }
    return patches, mocks


def test_dry_run_rejects_existing_old_t90_manifest_as_invalid() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        manifest, truth_csv, filters_csv, item = load_fixture_bundle(
            directory, targets=(0.90,)
        )
        args = configure_prerequisite_args(
            directory,
            matched_manifest=manifest,
            truth_csv=truth_csv,
            filters_csv=filters_csv,
        )
        patches, mocks = prepare_static_fixture_patches(item)
        try:
            payload = runner.dry_run_payload(args)
        finally:
            patches.stop()

        assert mocks["audit_matched_recall_manifest"].called
        assert payload["formal_ready"] is False
        assert payload["formal_prerequisites"]["status"] == "invalid"
        assert payload["formal_prerequisites"]["missing"] == []
        assert any(
            "does not contain every requested" in error
            for error in payload["formal_prerequisites"]["validation_errors"]
        )


def test_dry_run_accepts_complete_static_matched_recall_fixture_without_db_fingerprint() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        manifest, truth_csv, filters_csv, item = load_fixture_bundle(directory)
        args = configure_prerequisite_args(
            directory,
            matched_manifest=manifest,
            truth_csv=truth_csv,
            filters_csv=filters_csv,
        )
        patches, _ = prepare_static_fixture_patches(item)
        try:
            with mock.patch.object(
                runner,
                "database_fingerprint",
                side_effect=AssertionError("dry-run must not fingerprint PostgreSQL"),
            ):
                payload = runner.dry_run_payload(args)
        finally:
            patches.stop()

        assert payload["formal_ready"] is True
        assert payload["formal_prerequisites"]["status"] == "ready"
        assert payload["formal_prerequisites"]["validation_errors"] == []
        assert payload["formal_prerequisites"]["static_matched_recall_gate"][
            "selected_config_cells"
        ] == len(runner.METHODS) * len(runner.FORMAL_TARGETS)


def test_audited_manifest_supplies_explicit_lcb_configs_for_every_target() -> None:
    with TemporaryDirectory() as temporary:
        manifest, truth_csv, filters_csv, item = load_fixture_bundle(Path(temporary))
        with mock.patch.object(
            runner,
            "audit_matched_recall_manifest",
            return_value={"valid": True, "errors": [], "warnings": []},
        ):
            bundle = runner.load_audited_matched_recall_configs(
                manifest,
                truth_csv=truth_csv,
                filters_csv=filters_csv,
                filters=[item],
                targets=runner.FORMAL_TARGETS,
            )
        assert set(bundle.configs) == {
            (item.name, method, target)
            for method in runner.METHODS
            for target in runner.FORMAL_TARGETS
        }
        assert all(row["recall_lcb95"] >= row["target_recall"] for row in bundle.evidence)
        assert bundle.provenance["requested_slice_complete"] is True
        assert bundle.guidance_filter_strategy == "safe_guided"
        assert bundle.configs[(item.name, "sqlens_d1", 0.95)].iterative_scan == "strict_order"


def test_throughput_rejects_manifest_or_cli_guidance_strategy_mismatch() -> None:
    with TemporaryDirectory() as temporary:
        manifest, truth_csv, filters_csv, item = load_fixture_bundle(Path(temporary))
        with mock.patch.object(
            runner,
            "audit_matched_recall_manifest",
            return_value={"valid": True, "errors": [], "warnings": []},
        ):
            bundle = runner.load_audited_matched_recall_configs(
                manifest,
                truth_csv=truth_csv,
                filters_csv=filters_csv,
                filters=[item],
                targets=runner.FORMAL_TARGETS,
            )
        args = runner.create_argument_parser().parse_args(
            ["--guidance-filter-strategy", "traversal_guided"]
        )
        with pytest.raises(runner.BenchmarkContractError, match="guidance_filter_strategy differs"):
            runner.bind_matched_recall_provenance(args, bundle)

        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["run_spec"]["args"]["guidance_filter_strategy"] = "traversal_guided"
        payload["run_spec"]["args"]["traversal_guided_prioritization"] = True
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with mock.patch.object(
            runner,
            "audit_matched_recall_manifest",
            return_value={"valid": True, "errors": [], "warnings": []},
        ), pytest.raises(runner.BenchmarkContractError, match="guidance_filter_strategy does not bind"):
            runner.load_audited_matched_recall_configs(
                manifest,
                truth_csv=truth_csv,
                filters_csv=filters_csv,
                filters=[item],
                targets=runner.FORMAL_TARGETS,
            )


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"status": "incomplete"}, "status is not complete"),
        ({"complete": False}, "requested slice is incomplete"),
        ({"policy": "mean_latency"}, "mean-only"),
        ({"lcb95": 0.94}, "mean-only or below target"),
        ({"targets": (0.90, 0.95)}, "does not contain every requested"),
    ],
)
def test_formal_config_loader_rejects_incomplete_mean_only_or_missing_target_artifacts(
    fixture_kwargs: dict[str, object], message: str
) -> None:
    with TemporaryDirectory() as temporary:
        manifest, truth_csv, filters_csv, item = load_fixture_bundle(
            Path(temporary), **fixture_kwargs
        )
        with mock.patch.object(
            runner,
            "audit_matched_recall_manifest",
            return_value={"valid": True, "errors": [], "warnings": []},
        ), pytest.raises(runner.BenchmarkContractError, match=message):
            runner.load_audited_matched_recall_configs(
                manifest,
                truth_csv=truth_csv,
                filters_csv=filters_csv,
                filters=[item],
                targets=runner.FORMAL_TARGETS,
            )


def test_formal_config_loader_requires_independent_audit_and_gt_hash_binding() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        manifest, truth_csv, filters_csv, item = load_fixture_bundle(directory)
        with mock.patch.object(
            runner,
            "audit_matched_recall_manifest",
            return_value={"valid": False, "errors": ["bad plan evidence"]},
        ), pytest.raises(runner.BenchmarkContractError, match="independent audit"):
            runner.load_audited_matched_recall_configs(
                manifest,
                truth_csv=truth_csv,
                filters_csv=filters_csv,
                filters=[item],
                targets=runner.FORMAL_TARGETS,
            )

        truth_csv.write_text("query_no\n1\n", encoding="utf-8")
        with mock.patch.object(
            runner,
            "audit_matched_recall_manifest",
            return_value={"valid": True, "errors": []},
        ), pytest.raises(runner.BenchmarkContractError, match="GT provenance"):
            runner.load_audited_matched_recall_configs(
                manifest,
                truth_csv=truth_csv,
                filters_csv=filters_csv,
                filters=[item],
                targets=runner.FORMAL_TARGETS,
            )


def test_matched_recall_runtime_and_live_index_identity_are_bound_exactly() -> None:
    with TemporaryDirectory() as temporary:
        manifest, truth_csv, filters_csv, item = load_fixture_bundle(Path(temporary))
        with mock.patch.object(
            runner,
            "audit_matched_recall_manifest",
            return_value={"valid": True, "errors": [], "warnings": []},
        ):
            bundle = runner.load_audited_matched_recall_configs(
                manifest,
                truth_csv=truth_csv,
                filters_csv=filters_csv,
                filters=[item],
                targets=runner.FORMAL_TARGETS,
            )
        args = runner.create_argument_parser().parse_args([])
        runner.bind_matched_recall_provenance(args, bundle)
        assert args.expected_sqlens_build_id == runner.REQUIRED_SQLENS_BUILD_PREFIXES[0] + "test"
        assert args.expected_vector_so_sha256 == "a" * 64
        assert args.bfs_index == args.insertion_index

        live = dict(bundle.provenance["database"])
        live["relations"] = {
            name: dict(value)
            for name, value in bundle.provenance["database"]["relations"].items()
        }
        evidence = runner.validate_live_matched_recall_provenance(bundle, live, args)
        assert evidence["relation_identity_exact_match"] is True
        assert evidence["query_relation_exact_match"] is True

        live["relations"][args.insertion_index]["relfilenode"] = 999
        with pytest.raises(runner.BenchmarkContractError, match="relfilenode"):
            runner.validate_live_matched_recall_provenance(bundle, live, args)

        live = dict(bundle.provenance["database"])
        live["relations"] = {
            name: dict(value)
            for name, value in bundle.provenance["database"]["relations"].items()
        }
        live["query_table"] = dict(bundle.provenance["database"]["query_table"])
        live["query_table"]["row_count"] = 9_999_999
        with pytest.raises(runner.BenchmarkContractError, match="query relation row_count"):
            runner.validate_live_matched_recall_provenance(bundle, live, args)


def measured_row(
    request_no: int,
    query_no: int,
    latency: float,
    completed_offset_ms: float,
    recall: float = 0.96,
    error: str = "",
    measurement_repeat: int = 0,
    clients: int = 1,
    dispatch_position: int | None = None,
    method: str = "stock",
    trace_permutation_seed: int | None = None,
) -> dict[str, object]:
    position = request_no if dispatch_position is None else dispatch_position
    order_seed = 10_000 + measurement_repeat if trace_permutation_seed is None else trace_permutation_seed
    completed = max(completed_offset_ms, latency)
    return {
        "method": method,
        "guidance_filter_strategy": "safe_guided",
        "evaluation_scope": "representative_filters",
        "clients": clients,
        "filter_name": "a",
        "target_recall": 0.95,
        "request_no": request_no,
        "query_no": query_no,
        "query_id": 1_000 + query_no % 100,
        "trace_cycle": request_no // 2,
        "measurement_repeat": measurement_repeat,
        "dispatch_position": position,
        "trace_permutation_seed": order_seed,
        "client_id": position % clients,
        "latency_ms": latency,
        "activation_ms": latency * 0.1,
        "query_ms": latency * 0.9,
        "started_offset_ms": max(0.0, completed - latency),
        "completed_offset_ms": completed,
        "recall_at_10": recall,
        "error_type": "QueryCanceled" if error else "",
        "error": error,
    }


def tiny_workload() -> runner.Workload:
    requests = tuple(
        runner.WorkloadRequest(number, 100 + number % 2, 1_000 + number % 2, number // 2)
        for number in range(4)
    )
    return runner.Workload(requests, "test_replay", "", "", "q100..q101", True, 2)


def matched_config_evidence(recall: float, latency: float, samples: int = 200) -> dict[str, object]:
    return {
        "recall_mean": recall,
        "recall_lcb95": recall,
        "latency_mean_ms": latency,
        "samples": samples,
        "selection_source": "independently_audited_matched_recall_selected_artifact",
    }


def arm_telemetry(scale: float = 1.0) -> dict[str, object]:
    return {
        "devices": ["sda4"],
        "host": {
            "cpu": {
                "utilization_pct": 50.0,
                "user_pct": 30.0,
                "system_pct": 15.0,
                "iowait_pct": 5.0,
            },
            "disk_total": {
                "reads_completed": 10.0 * scale,
                "read_bytes": 4096.0 * scale,
                "read_time_ms": 4.0 * scale,
                "writes_completed": 2.0 * scale,
                "write_bytes": 1024.0 * scale,
                "write_time_ms": 1.0 * scale,
                "io_time_ms": 5.0 * scale,
                "weighted_io_time_ms": 6.0 * scale,
            },
        },
        "postgresql": {
            "database": {
                "blks_read": 3.0 * scale,
                "blks_hit": 30.0 * scale,
                "temp_files": 0.0,
                "temp_bytes": 0.0,
                "blk_read_time": 2.0 * scale,
                "blk_write_time": 0.5 * scale,
            },
            "io_total": {
                "reads": 4.0 * scale,
                "read_bytes": 8192.0 * scale,
                "read_time": 2.5 * scale,
                "writes": 1.0 * scale,
                "write_bytes": 2048.0 * scale,
                "write_time": 0.5 * scale,
                "hits": 20.0 * scale,
                "evictions": 0.0,
            },
            "relations": {
                "target_table": "public.t",
                "target_index": "public.t_hnsw_idx",
                "tracking_complete": True,
                "table": {
                    "relid": 10,
                    "schemaname": "public",
                    "relname": "t",
                    "heap_blks_read": 3.0 * scale,
                    "heap_blks_hit": 30.0 * scale,
                    "idx_blks_read": 4.0 * scale,
                    "idx_blks_hit": 40.0 * scale,
                },
                "index": {
                    "relid": 10,
                    "indexrelid": 11,
                    "schemaname": "public",
                    "relname": "t",
                    "indexrelname": "t_hnsw_idx",
                    "idx_blks_read": 4.0 * scale,
                    "idx_blks_hit": 40.0 * scale,
                },
            },
        },
        "backend_cpu": {
            "backend_pids": [123],
            "tracking_complete": True,
            "per_backend": {
                "123": {
                    "pid": 123.0,
                    "user_cpu_ms": 2.0 * scale,
                    "system_cpu_ms": 1.0 * scale,
                    "total_cpu_ms": 3.0 * scale,
                }
            },
            "total": {
                "user_cpu_ms": 2.0 * scale,
                "system_cpu_ms": 1.0 * scale,
                "total_cpu_ms": 3.0 * scale,
            },
        },
    }


def test_summary_reports_tail_qps_bootstrap_recall_and_error_counts() -> None:
    item = filter_spec()
    config = runner.SearchConfig(100, 1_000, 8.0, "off", 100)
    rows = [
        measured_row(0, 100, 1.0, 300.0, clients=2),
        measured_row(1, 101, 2.0, 700.0, clients=2),
        measured_row(2, 100, 3.0, 1_400.0, clients=2),
        measured_row(3, 101, 99.0, 1_800.0, error="timeout", clients=2),
    ]
    summary = runner.summarize_arm(
        rows,
        2.0,
        0.95,
        2,
        "stock",
        config,
        item,
        tiny_workload(),
        matched_config_evidence(0.97, 4.0),
        100,
        17,
    )

    assert summary["completed_queries"] == 3
    assert summary["error_count"] == 1
    assert json.loads(summary["error_counts_json"]) == {"QueryCanceled": 1}
    assert summary["throughput_qps"] == 1.5
    assert summary["latency_p50_ms"] == 2.0
    assert summary["latency_p95_ms"] == 3.0
    assert summary["latency_p99_ms"] == 3.0
    assert summary["status"] == "invalid"
    assert summary["throughput_qps_ci95_low"] <= summary["throughput_qps_ci95_high"]
    assert summary["latency_recall_bootstrap_unit"] == "unique_query_vector_cluster"


def test_valid_summary_uses_completed_queries_over_wall_clock() -> None:
    item = filter_spec()
    config = runner.SearchConfig(100, 1_000, 8.0, "off", 100)
    rows = [measured_row(number, 100 + number % 2, number + 1.0, (number + 1) * 400.0) for number in range(4)]
    summary = runner.summarize_arm(
        rows,
        2.0,
        0.95,
        1,
        "sqlens_d1",
        config,
        item,
        tiny_workload(),
        matched_config_evidence(0.96, 3.0),
        50,
        23,
        telemetry=arm_telemetry(),
    )
    assert summary["status"] == "valid"
    assert summary["throughput_qps"] == 2.0
    assert summary["workload_requests"] == 4
    assert summary["unique_query_vectors"] == 2
    assert summary["trace_replay"] is True
    assert summary["telemetry_collected"] is True
    assert summary["host_disk_read_bytes"] == 4096.0
    assert summary["pg_io_reads"] == 4.0
    assert json.loads(summary["telemetry_json"])["devices"] == ["sda4"]


def test_measurement_mean_can_pass_while_unique_query_cluster_lcb95_fails() -> None:
    item = filter_spec()
    config = runner.SearchConfig(100, 1_000, 8.0, "off", 100)
    rows = [
        measured_row(0, 100, 1.0, 100.0, recall=1.0),
        measured_row(1, 101, 1.0, 200.0, recall=0.9),
        measured_row(2, 100, 1.0, 300.0, recall=1.0),
        measured_row(3, 101, 1.0, 400.0, recall=0.9),
    ]
    summary = runner.summarize_arm(
        rows,
        1.0,
        0.94,
        1,
        "stock",
        config,
        item,
        tiny_workload(),
        matched_config_evidence(0.99, 1.0),
        1000,
        71,
        telemetry=arm_telemetry(),
    )
    assert summary["recall_mean"] == pytest.approx(0.95)
    assert summary["target_met_measurement"] is True
    assert summary["recall_query_cluster_ci95_low"] < 0.94
    assert summary["target_lcb95_met_measurement"] is False
    assert summary["status"] == "invalid"
    assert "unique query-vector clusters" in summary["recall_lcb95_definition"]


def test_summary_rejects_nonfinite_or_out_of_wall_request_timing() -> None:
    item = filter_spec()
    config = runner.SearchConfig(100, 1_000, 8.0, "off", 100)
    rows = [measured_row(number, 100 + number % 2, 2.0, 10.0) for number in range(4)]
    rows[0]["completed_offset_ms"] = 2_000.0
    with pytest.raises(runner.BenchmarkContractError, match="exceeds"):
        runner.summarize_arm(
            rows, 1.0, 0.95, 1, "stock", config, item, tiny_workload(),
            matched_config_evidence(0.96, 3.0), 50, 23,
        )

    rows = [measured_row(number, 100 + number % 2, 2.0, 10.0) for number in range(4)]
    rows[0]["latency_ms"] = float("nan")
    with pytest.raises(runner.BenchmarkContractError, match="non-finite"):
        runner.summarize_arm(
            rows, 1.0, 0.95, 1, "stock", config, item, tiny_workload(),
            matched_config_evidence(0.96, 3.0), 50, 23,
        )


def test_bootstrap_functions_are_seeded() -> None:
    values = [1.0, 2.0, 3.0, 10.0]
    assert runner.bootstrap_mean_ci(values, 100, 42) == runner.bootstrap_mean_ci(values, 100, 42)
    rows = [measured_row(number, 100, 1.0, (number + 1) * 100.0) for number in range(20)]
    assert runner.cluster_bootstrap_percentile_ci(rows, 0.95, 100, 42) == runner.cluster_bootstrap_percentile_ci(rows, 0.95, 100, 42)
    ratio = runner.bootstrap_pooled_ratio_ci([10, 10, 10], [1.0, 2.0, 4.0], 100, 42)
    assert ratio == runner.bootstrap_pooled_ratio_ci([10, 10, 10], [1.0, 2.0, 4.0], 100, 42)
    assert ratio[0] <= 30.0 / 7.0 <= ratio[1]


def test_formal_execute_requires_audited_manifest_and_six_repeats() -> None:
    args = runner.create_argument_parser().parse_args(["--execute"])
    with pytest.raises(runner.BenchmarkContractError, match="matched-recall-manifest"):
        runner.validate_formal_args(args)

    args.matched_recall_manifest = Path("matched.json")
    args.measurement_repeats = 2
    with pytest.raises(runner.BenchmarkContractError, match="six measurement repeats"):
        runner.validate_formal_args(args)


def test_runtime_configuration_binds_safe_guided_or_traversal_guided_semantics() -> None:
    args = runner.create_argument_parser().parse_args([])
    config = runner.SearchConfig(100, 1_000, 8.0, "off", 100, True, 7)

    assert runner.configure_args_for_runtime(args, "sqlens_d1", config) == runner.MODE_BY_METHOD["sqlens_d1"]
    d1 = args.mode_configs_json[runner.MODE_BY_METHOD["sqlens_d1"]]
    assert d1["traversal_guided_prioritization"] is False
    assert d1["traversal_guided_burst"] == 7

    args.guidance_filter_strategy = "traversal_guided"
    runner.configure_args_for_runtime(args, "sqlens_d1", config)
    assert args.mode_configs_json[runner.MODE_BY_METHOD["sqlens_d1"]]["traversal_guided_prioritization"] is True

    assert runner.configure_args_for_runtime(args, "stock", config) == runner.MODE_BY_METHOD["stock"]
    stock = args.mode_configs_json[runner.MODE_BY_METHOD["stock"]]
    assert stock["traversal_guided_prioritization"] is False
    assert stock["traversal_guided_burst"] == 7


def test_plan_evidence_requires_one_passing_exact_index_check_per_backend() -> None:
    runtimes = [
        argparse.Namespace(backend_cpu_provenance={"backend_pid": pid})
        for pid in (11, 12)
    ]
    item = filter_spec("half")
    rows = [
        {
            "mode": runner.MODE_BY_METHOD["sqlens_d1"],
            "filter_name": item.name,
            "expected_index": "source_idx",
            "passed": True,
            "backend_cpu_provenance": {"backend_pid": pid},
        }
        for pid in (11, 12)
    ]
    evidence = runner.validate_plan_evidence(
        rows, runtimes, runner.MODE_BY_METHOD["sqlens_d1"], item, "source_idx"
    )
    assert evidence["passed"] is True
    assert evidence["observed_count"] == 2

    with pytest.raises(runner.BenchmarkContractError, match="per-client EXPLAIN"):
        runner.validate_plan_evidence(
            rows[:1], runtimes, runner.MODE_BY_METHOD["sqlens_d1"], item, "source_idx"
        )


def test_runtime_canary_binds_safe_guided_and_traversal_guided_paths() -> None:
    class Cursor:
        def __init__(self, strategy: str, prioritization: str) -> None:
            self.strategy = strategy
            self.prioritization = prioritization
            self.sql: list[str] = []

        def execute(self, sql: str, *args: object) -> None:
            self.sql.append(sql)

        def fetchone(self) -> tuple[str, str, str]:
            return self.strategy, self.prioritization, "8"

    args = runner.create_argument_parser().parse_args([])
    args.traversal_guided_burst = 8
    item = filter_spec("half")
    truth = FakeTruth(10)
    runtime = argparse.Namespace(
        cur=Cursor("traversal_guided", "on"), mode=runner.MODE_BY_METHOD["sqlens_d1"],
        backend_cpu_provenance={"backend_pid": 71},
    )
    args.guidance_filter_strategy = "traversal_guided"
    profile = {
        "final_path": "approximate_traversal_prioritization",
        "planner_proof_attempted": True,
        "planner_proof_succeeded": True,
        "approximate_ann_path": True,
        "approximate_prioritization_attempted": True,
        "traversal_order_changed": True,
        "priority_reorders": 1,
        "match_frontier_pops": 4,
        "no_bridge_frontier_pops": 1,
        "traversal_prioritization_burst": 8,
        "graph_expansion_pruned": False,
        "distance_computations_pruned": False,
        "stock_bypass_requests": 0,
        "fallback_requests": 0,
    }
    with mock.patch.object(runner, "activate", return_value={"table": "t"}), \
         mock.patch.object(runner, "activation_binding", return_value=None), \
         mock.patch.object(runner, "candidate_self_exclusion", return_value=False), \
         mock.patch.object(runner, "query_table_for_candidate", return_value="t"), \
         mock.patch.object(runner, "tie_aware_recall", return_value=1.0), \
         mock.patch.object(runner, "run_query", return_value=([1] * 10, [0.1] * 10, profile)):
        evidence = runner.runtime_canary(args, runtime, "sqlens_d1", item, 100, 1_000, truth)
    assert evidence["profile"]["final_path"] == "approximate_traversal_prioritization"

    runtime.cur.strategy = "safe_guided"
    runtime.cur.prioritization = "off"
    args.guidance_filter_strategy = "safe_guided"
    with mock.patch.object(runner, "activate", return_value={"table": "t"}), \
         mock.patch.object(runner, "activation_binding", return_value=None), \
         mock.patch.object(runner, "candidate_self_exclusion", return_value=False), \
         mock.patch.object(runner, "query_table_for_candidate", return_value="t"), \
         mock.patch.object(runner, "tie_aware_recall", return_value=1.0), \
         mock.patch.object(runner, "run_query", return_value=([1] * 10, [0.1] * 10, {"final_path": "stock"})):
        evidence = runner.runtime_canary(args, runtime, "stock", item, 100, 1_000, truth)
    assert evidence["gucs"]["hnsw.traversal_guided_prioritization"] == "off"

    safe_profile = {
        "final_path": "validation_only",
        "graph_expansion_pruned": False,
        "distance_computations_pruned": False,
        "traversal_order_changed": False,
        "priority_reorders": 0,
        "guidance_checks": 4,
    }
    runtime.mode = runner.MODE_BY_METHOD["sqlens_d1"]
    with mock.patch.object(runner, "activate", return_value={"table": "t"}), \
         mock.patch.object(runner, "activation_binding", return_value=None), \
         mock.patch.object(runner, "candidate_self_exclusion", return_value=False), \
         mock.patch.object(runner, "query_table_for_candidate", return_value="t"), \
         mock.patch.object(runner, "tie_aware_recall", return_value=1.0), \
         mock.patch.object(runner, "run_query", return_value=([1] * 10, [0.1] * 10, safe_profile)):
        evidence = runner.runtime_canary(args, runtime, "sqlens_d1", item, 100, 1_000, truth)
    assert evidence["gucs"]["hnsw.filter_strategy"] == "safe_guided"


def test_measurement_repeats_are_interleaved_and_aggregate_repeat_qps_ci() -> None:
    item = filter_spec("a")
    schedule = runner.build_measurement_schedule([0.95], [1], [item], 81, measurement_repeats=6)
    assert len(schedule) == 12
    assert {row["measurement_repeat"] for row in schedule} == {0, 1, 2, 3, 4, 5}
    for offset in range(0, len(schedule), 2):
        assert {row["method"] for row in schedule[offset : offset + 2]} == set(runner.METHODS)

    config = runner.SearchConfig(100, 1_000, 8.0, "off", 100)
    repeats = []
    combined = []
    for repeat, wall in enumerate((1.0, 1.5, 2.0, 2.5, 3.0, 3.5)):
        rows = [
            measured_row(
                number, 100 + number % 2, number + 1.0, 10.0,
                measurement_repeat=repeat,
            )
            for number in range(4)
        ]
        combined.extend(rows)
        repeats.append(runner.summarize_arm(
            rows, wall, 0.95, 1, "stock", config, item, tiny_workload(),
            matched_config_evidence(0.96, 3.0), 50, 23, repeat,
            arm_telemetry(repeat + 1),
        ))
    aggregate = runner.aggregate_measurement_cell(repeats, combined, 100, 23)
    assert aggregate["summary_type"] == "aggregate"
    assert aggregate["throughput_bootstrap_unit"] == "measurement_repeat_completed_wall_pair"
    assert aggregate["throughput_qps"] == pytest.approx(24.0 / 13.5)
    assert aggregate["throughput_qps_ci95_low"] <= aggregate["throughput_qps_ci95_high"]
    assert aggregate["latency_p95_query_cluster_ci95_low_ms"] <= aggregate["latency_p95_query_cluster_ci95_high_ms"]
    assert aggregate["telemetry_collected"] is True
    assert aggregate["host_disk_reads_completed"] == 210.0
    assert len(json.loads(aggregate["telemetry_json"])["repeats"]) == 6


def test_measurement_trace_permutation_is_paired_across_methods_and_changes_by_repeat() -> None:
    workload = tiny_workload()
    seed0, dispatch0 = runner.measurement_dispatch(workload, 57, 0.95, 4, "f", 0)
    seed0_again, dispatch0_again = runner.measurement_dispatch(workload, 57, 0.95, 4, "f", 0)
    seed1, dispatch1 = runner.measurement_dispatch(workload, 57, 0.95, 4, "f", 1)

    order0 = [request.request_no for _, request in dispatch0]
    assert seed0 == seed0_again
    assert order0 == [request.request_no for _, request in dispatch0_again]
    assert seed1 != seed0
    assert [request.request_no for _, request in dispatch1] != order0


def test_aggregate_recomputes_mean_recall_and_rejects_tampered_raw_coverage() -> None:
    item = filter_spec("a")
    config = runner.SearchConfig(100, 1_000, 8.0, "off", 100)
    repeat_summaries = []
    combined: list[dict[str, object]] = []
    for repeat, latency in enumerate((1.0, 10.0, 100.0, 1.0, 10.0, 100.0)):
        rows = [
            measured_row(
                number,
                100 + number % 2,
                latency,
                10.0,
                recall=0.96 + min(repeat, 4) * 0.01,
                measurement_repeat=repeat,
            )
            for number in range(4)
        ]
        combined.extend(rows)
        repeat_summaries.append(
            runner.summarize_arm(
                rows, 1.0, 0.95, 1, "stock", config, item, tiny_workload(),
                matched_config_evidence(0.99, 1.0),
                50, 19, repeat,
            )
        )

    aggregate = runner.aggregate_measurement_cell(repeat_summaries, combined, 50, 19)
    assert aggregate["latency_mean_ms"] == pytest.approx(37.0)
    assert aggregate["recall_mean"] == pytest.approx(0.9833333333333333)
    assert aggregate["latency_p50_ms"] == 10.0
    assert aggregate["latency_p95_ms"] == 100.0
    assert aggregate["latency_p99_ms"] == 100.0
    assert aggregate["target_met_measurement"] is True
    assert aggregate["target_met_each_repeat"] is True

    tampered_summaries = [dict(row, recall_mean=0.0, status="invalid") for row in repeat_summaries]
    recomputed = runner.aggregate_measurement_cell(tampered_summaries, combined, 50, 19)
    assert recomputed["recall_mean"] == pytest.approx(0.9833333333333333)
    assert recomputed["status"] == "valid"

    tampered = combined[:-1] + [dict(combined[-2])]
    with pytest.raises(runner.BenchmarkContractError, match="coverage"):
        runner.aggregate_measurement_cell(repeat_summaries, tampered, 50, 19)


def test_aggregate_target_gate_requires_every_repeat_and_scores_errors_as_zero() -> None:
    item = filter_spec("a")
    config = runner.SearchConfig(100, 1_000, 8.0, "off", 100)
    repeat_summaries = []
    combined: list[dict[str, object]] = []
    recalls = (0.94, 0.96, 0.96, 0.96, 0.96, 0.96)
    for repeat, recall in enumerate(recalls):
        rows = [
            measured_row(
                number,
                100 + number % 2,
                2.0,
                10.0,
                recall=recall,
                measurement_repeat=repeat,
            )
            for number in range(4)
        ]
        combined.extend(rows)
        repeat_summaries.append(
            runner.summarize_arm(
                rows, 1.0, 0.95, 1, "stock", config, item, tiny_workload(),
                matched_config_evidence(0.99, 1.0),
                20, 5, repeat,
            )
        )
    aggregate = runner.aggregate_measurement_cell(repeat_summaries, combined, 20, 5)
    assert aggregate["recall_mean"] == pytest.approx(sum(recalls) / len(recalls))
    assert aggregate["target_met_measurement"] is True
    assert aggregate["target_met_each_repeat"] is False
    assert aggregate["status"] == "invalid"

    failed = [dict(row) for row in combined]
    failed[0]["error"] = "timeout"
    failed[0]["error_type"] = "QueryCanceled"
    failed_summaries = [dict(row) for row in repeat_summaries]
    failed_summaries[0]["completed_queries"] = 3
    failed_summaries[0]["error_count"] = 1
    failed_aggregate = runner.aggregate_measurement_cell(failed_summaries, failed, 20, 5)
    assert failed_aggregate["recall_mean"] < aggregate["recall_mean"]
    assert failed_aggregate["status"] == "invalid"


def test_aggregate_rejects_cross_strategy_or_config_mixing() -> None:
    item = filter_spec("a")
    config = runner.SearchConfig(100, 1_000, 8.0, "off", 100)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for repeat in range(6):
        repeat_rows = [
            measured_row(number, 100 + number % 2, 2.0, 10.0, measurement_repeat=repeat)
            for number in range(4)
        ]
        rows.extend(repeat_rows)
        summaries.append(
            runner.summarize_arm(
                repeat_rows, 1.0, 0.95, 1, "stock", config, item, tiny_workload(),
                matched_config_evidence(0.99, 1.0), 20, 5, repeat,
            )
        )
    summaries[-1]["guidance_filter_strategy"] = "traversal_guided"
    with pytest.raises(runner.BenchmarkContractError, match="guidance_filter_strategy"):
        runner.aggregate_measurement_cell(summaries, rows, 20, 5)


def test_measurement_cell_requires_exact_paired_repeat_client_and_seed_coverage() -> None:
    workload = tiny_workload()
    rows = [
        measured_row(
            request_no,
            100 + request_no % 2,
            1.0,
            10.0,
            measurement_repeat=repeat,
            clients=2,
            method=method,
            trace_permutation_seed=700 + repeat,
        )
        for repeat in range(3)
        for method in runner.METHODS
        for request_no in range(4)
    ]
    evidence = runner.validate_measurement_cell_rows(rows, workload, 0.95, 2, "a", 3)
    assert evidence["no_duplicate_or_missing_requests"] is True
    assert evidence["repeat_order_seeds"] == [700, 701, 702]

    duplicate = [dict(row) for row in rows]
    duplicate[-1] = dict(duplicate[-2])
    with pytest.raises(runner.BenchmarkContractError, match="request_no coverage"):
        runner.validate_measurement_cell_rows(duplicate, workload, 0.95, 2, "a", 3)

    reused_seed = [dict(row) for row in rows]
    for row in reused_seed:
        if row["measurement_repeat"] == 2:
            row["trace_permutation_seed"] = 701
    with pytest.raises(runner.BenchmarkContractError, match="distinct request-order seeds"):
        runner.validate_measurement_cell_rows(reused_seed, workload, 0.95, 2, "a", 3)

    wrong_client = [dict(row) for row in rows]
    wrong_client[0]["client_id"] = 1
    with pytest.raises(runner.BenchmarkContractError, match="client coverage"):
        runner.validate_measurement_cell_rows(wrong_client, workload, 0.95, 2, "a", 3)


def test_completion_coverage_rejects_duplicate_or_missing_summary_and_evidence_keys() -> None:
    item = filter_spec("a")
    schedule = runner.build_measurement_schedule(
        [0.95], [1], [item], 81, measurement_repeats=3
    )
    repeat_rows = [
        {
            "summary_type": "repeat",
            "target_recall": arm["target_recall"],
            "clients": arm["clients"],
            "filter_name": arm["filter_name"],
            "method": arm["method"],
            "measurement_repeat": arm["measurement_repeat"],
            "workload_requests": 4,
            "unique_query_vectors": 2,
            "telemetry_collected": True,
            "target_lcb95_met_measurement": True,
        }
        for arm in schedule
    ]
    aggregate_rows = [
        {
            "summary_type": "aggregate",
            "target_recall": 0.95,
            "clients": 1,
            "filter_name": "a",
            "method": method,
            "workload_requests": 4,
            "unique_query_vectors": 2,
            "measurement_repeats": 3,
            "coverage_gate_passed": True,
            "telemetry_collected": True,
            "target_lcb95_met_measurement": True,
            "target_lcb95_met_each_repeat": True,
        }
        for method in runner.METHODS
    ]
    evidence = []
    for arm in schedule:
        repeat = int(arm["measurement_repeat"])
        evidence.append(
            {
                "arm_key": runner.measurement_arm_key(
                    0.95, 1, "a", str(arm["method"]), repeat
                ),
                "arm_order": arm["arm_order"],
                "block_no": arm["block_no"],
                "method_position": arm["method_position"],
                "method": arm["method"],
                "target_recall": arm["target_recall"],
                "clients": arm["clients"],
                "filter_name": arm["filter_name"],
                "trace_permutation_seed": 900 + repeat,
                "trace_order_sha256": f"order-{repeat}",
                "backend_pids": [123],
                "telemetry": arm_telemetry(),
            }
        )
    gate = runner.validate_completion_coverage(
        repeat_rows + aggregate_rows,
        schedule,
        evidence,
        [0.95],
        [1],
        [item],
        3,
        tiny_workload(),
        {runner.measurement_cell_key(0.95, 1, "a"): {"passed": True}},
    )
    assert gate["passed"] is True
    assert gate["cell_raw_coverage_evidence_bound"] is True

    with pytest.raises(runner.BenchmarkContractError, match="duplicate or missing arms"):
        runner.validate_completion_coverage(
            repeat_rows[:-1] + [dict(repeat_rows[-2])] + aggregate_rows,
            schedule,
            evidence,
            [0.95],
            [1],
            [item],
            3,
            tiny_workload(),
        )


def test_atomic_json_and_csv_fsync_temporary_file_and_parent() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        calls: list[int] = []
        original = runner.os.fsync
        with mock.patch.object(runner.os, "fsync", side_effect=lambda descriptor: (calls.append(descriptor), original(descriptor))[1]):
            runner.atomic_json(directory / "evidence.json", {"ok": True})
            runner.write_csv_atomic(directory / "evidence.csv", [{"ok": True}])
        assert len(calls) >= 4


def test_each_client_must_have_an_independent_backend() -> None:
    def runtime(pid: int) -> argparse.Namespace:
        return argparse.Namespace(backend_cpu_provenance={"backend_pid": pid})

    assert runner.validate_independent_backends([runtime(10), runtime(11)], 2) == [10, 11]
    with pytest.raises(runner.BenchmarkContractError, match="independent PostgreSQL"):
        runner.validate_independent_backends([runtime(10), runtime(10)], 2)


def test_catalog_index_gate_requires_one_ready_shared_stock_d1_index() -> None:
    def relation(oid: int) -> dict[str, object]:
        return {
            "oid": oid,
            "relfilenode": oid + 100,
            "bytes": 1_000,
            "valid": True,
            "ready": True,
            "candidate_validity_predicate_matches": True,
        }

    database = {"relations": {"stock_idx": relation(10)}}
    evidence = runner.validate_database_index_gate(database, "stock_idx")
    assert evidence["passed"] is True
    assert evidence["same_hnsw_index_for_stock_and_d1"] is True
    assert evidence["per_client_exact_hnsw_explain_gate_required"] is True

    database["relations"]["stock_idx"]["ready"] = False
    with pytest.raises(runner.BenchmarkContractError, match="index identity/readiness"):
        runner.validate_database_index_gate(database, "stock_idx")


def test_client_cpu_affinity_parser_expands_ranges() -> None:
    assert runner.parse_cpu_set("1-3,7") == (1, 2, 3, 7)
    assert runner.parse_cpu_set(None) == ()


def test_host_proc_cpu_and_diskstats_deltas_are_persistable() -> None:
    before = {
        "monotonic_ns": 1_000_000_000,
        "cpu": {
            "user": 10,
            "nice": 0,
            "system": 10,
            "idle": 70,
            "iowait": 10,
            "irq": 0,
            "softirq": 0,
            "steal": 0,
        },
        "disk": {
            "sda4": dict(
                zip(
                    runner.DISK_COUNTER_NAMES,
                    (10, 1, 100, 20, 5, 1, 40, 10, 0, 20, 30),
                )
            )
        },
    }
    after = {
        "monotonic_ns": 3_000_000_000,
        "cpu": {
            "user": 30,
            "nice": 0,
            "system": 20,
            "idle": 120,
            "iowait": 20,
            "irq": 0,
            "softirq": 0,
            "steal": 0,
        },
        "disk": {
            "sda4": dict(
                zip(
                    runner.DISK_COUNTER_NAMES,
                    (14, 2, 108, 26, 8, 1, 44, 13, 1, 25, 38),
                )
            )
        },
    }
    delta = runner.host_telemetry_delta(before, after)
    assert delta["window_seconds"] == 2.0
    assert delta["cpu"]["utilization_pct"] == pytest.approx(100.0 / 3.0)
    assert delta["disk_total"]["reads_completed"] == 4.0
    assert delta["disk_total"]["read_bytes"] == 8 * 512
    assert delta["disk_devices"]["sda4"]["io_in_progress_end"] == 1.0


def test_explicit_telemetry_device_resolution_uses_proc_diskstats() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        diskstats = directory / "diskstats"
        diskstats.write_text(
            "   8       4 sda4 1 0 2 3 4 0 5 6 0 7 8\n",
            encoding="utf-8",
        )
        devices, evidence = runner.resolve_telemetry_devices(
            "/dev/sda4",
            [directory],
            diskstats_path=diskstats,
            sys_dev_block_path=directory / "missing-sysfs",
        )
        assert devices == ("sda4",)
        assert evidence["explicit_devices"] == ["sda4"]
        assert runner.read_diskstats(devices, diskstats)["sda4"][
            "sectors_written"
        ] == 5


def test_backend_cpu_proc_parser_and_delta_bind_the_same_postgres_process() -> None:
    with TemporaryDirectory() as temporary:
        proc_root = Path(temporary)
        stat_path = proc_root / "4321" / "stat"
        stat_path.parent.mkdir()

        def write_stat(utime: int, stime: int, starttime: int = 777) -> None:
            fields = ["0"] * 20
            fields[0] = "S"
            fields[11] = str(utime)
            fields[12] = str(stime)
            fields[19] = str(starttime)
            stat_path.write_text(
                "4321 (postgres: client backend) " + " ".join(fields),
                encoding="utf-8",
            )

        write_stat(100, 20)
        before = runner.backend_cpu_snapshot(
            [4321], proc_root=proc_root, clock_ticks_per_second=100
        )
        write_stat(125, 35)
        after = runner.backend_cpu_snapshot(
            [4321], proc_root=proc_root, clock_ticks_per_second=100
        )
        delta = runner.backend_cpu_delta(before, after)
        assert delta["backend_pids"] == [4321]
        assert delta["per_backend"]["4321"]["user_cpu_ms"] == 250.0
        assert delta["per_backend"]["4321"]["system_cpu_ms"] == 150.0
        assert delta["total"]["total_cpu_ms"] == 400.0

        write_stat(130, 36, starttime=778)
        reused = runner.backend_cpu_snapshot(
            [4321], proc_root=proc_root, clock_ticks_per_second=100
        )
        with pytest.raises(runner.BenchmarkContractError, match="reused"):
            runner.backend_cpu_delta(after, reused)


def test_completion_telemetry_rejects_unbound_backend_or_relation_stats() -> None:
    telemetry = arm_telemetry()
    runner.validate_arm_telemetry(telemetry, [123])

    missing_backend = json.loads(json.dumps(telemetry))
    missing_backend["backend_cpu"]["backend_pids"] = [999]
    with pytest.raises(runner.BenchmarkContractError, match="backend CPU telemetry"):
        runner.validate_arm_telemetry(missing_backend, [123])

    missing_relation = json.loads(json.dumps(telemetry))
    del missing_relation["postgresql"]["relations"]["index"]["idx_blks_hit"]
    with pytest.raises(runner.BenchmarkContractError, match="lacks target counters"):
        runner.validate_arm_telemetry(missing_relation, [123])


def test_postgres_database_and_io_counter_deltas_include_bytes() -> None:
    database_before = {name: 0 for name in runner.PG_STAT_DATABASE_COUNTERS}
    database_before.update({"datid": 1, "datname": "db", "stats_reset": "t0"})
    database_after = dict(database_before)
    database_after.update({"blks_read": 3, "blks_hit": 12, "blk_read_time": 1.5})
    io_before = {name: 0 for name in runner.PG_STAT_IO_COUNTERS}
    io_before.update(
        {
            "backend_type": "client backend",
            "object": "relation",
            "context": "normal",
            "op_bytes": 8192,
        }
    )
    io_after = dict(io_before)
    io_after.update({"reads": 4, "writes": 2, "hits": 9})
    relations_before = {
        "target_table": "public.t",
        "target_index": "public.t_hnsw_idx",
        "table": {
            "relid": 10,
            "schemaname": "public",
            "relname": "t",
            "heap_blks_read": 0,
            "heap_blks_hit": 0,
            "idx_blks_read": 0,
            "idx_blks_hit": 0,
        },
        "index": {
            "relid": 10,
            "indexrelid": 11,
            "schemaname": "public",
            "relname": "t",
            "indexrelname": "t_hnsw_idx",
            "idx_blks_read": 0,
            "idx_blks_hit": 0,
        },
    }
    relations_after = json.loads(json.dumps(relations_before))
    relations_after["table"].update({"heap_blks_read": 3, "idx_blks_hit": 12})
    relations_after["index"].update({"idx_blks_read": 4, "idx_blks_hit": 12})
    delta = runner.postgres_telemetry_delta(
        {
            "monotonic_ns": 1,
            "database": database_before,
            "io": [io_before],
            "relations": relations_before,
        },
        {
            "monotonic_ns": 1_000_000_001,
            "database": database_after,
            "io": [io_after],
            "relations": relations_after,
        },
    )
    assert delta["database"]["blks_read"] == 3.0
    assert delta["io_total"]["reads"] == 4.0
    assert delta["io_total"]["read_bytes"] == 4 * 8192
    assert delta["io_total"]["write_bytes"] == 2 * 8192
    assert delta["relations"]["table"]["heap_blks_read"] == 3.0
    assert delta["relations"]["index"]["idx_blks_read"] == 4.0


def test_checkpoint_binds_run_spec_and_truncates_partial_raw_tail() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        raw = directory / "raw.csv"
        checkpoint_path = directory / "checkpoint.json"
        offset = runner.initialize_raw_csv(raw, runner.RAW_FIELDS)
        checkpoint = {
            "schema_version": runner.CHECKPOINT_SCHEMA_VERSION,
            "run_spec_hash": "abc",
            "raw_byte_offset": offset,
            "raw_pair_artifacts": {},
        }
        runner.atomic_json(checkpoint_path, checkpoint)
        with raw.open("ab") as target:
            target.write(b"partial,row")

        loaded = runner.load_checkpoint(checkpoint_path, "abc")
        runner.prepare_resume_raw(raw, loaded, runner.RAW_FIELDS)
        assert raw.stat().st_size == offset
        with pytest.raises(runner.BenchmarkContractError, match="run-spec"):
            runner.load_checkpoint(checkpoint_path, "different")


def test_resume_raw_verifies_each_committed_pair_segment_hash() -> None:
    with TemporaryDirectory() as temporary:
        raw = Path(temporary) / "raw.csv"
        runner.initialize_raw_csv(raw, runner.RAW_FIELDS)
        rows = []
        arm_keys = []
        for method in runner.METHODS:
            row = {field: "" for field in runner.RAW_FIELDS}
            row.update(
                {
                    "method": method,
                    "request_no": 0,
                    "query_no": 100,
                    "query_id": 1_000,
                }
            )
            rows.append(row)
            arm_keys.append(runner.measurement_arm_key(0.95, 1, "a", method, 0))
        artifact = runner.append_csv_rows(raw, rows, runner.RAW_FIELDS)
        artifact.update({"methods": list(runner.METHODS), "arm_keys": arm_keys})
        checkpoint = {
            "raw_byte_offset": artifact["end_offset"],
            "raw_pair_artifacts": {
                runner.measurement_pair_key(0.95, 1, "a", 0): artifact
            },
        }
        runner.prepare_resume_raw(raw, checkpoint, runner.RAW_FIELDS)

        with raw.open("r+b") as target:
            target.seek(artifact["start_offset"])
            target.write(b"X")
            target.flush()
        with pytest.raises(runner.BenchmarkContractError, match="segment hash mismatch"):
            runner.prepare_resume_raw(raw, checkpoint, runner.RAW_FIELDS)


def test_resume_checkpoint_accepts_only_complete_committed_method_pairs() -> None:
    item = filter_spec("a")
    workload = tiny_workload()
    schedule = runner.build_measurement_schedule(
        [0.95], [1], [item], 9, measurement_repeats=1
    )
    pair_key = runner.measurement_pair_key(0.95, 1, "a", 0)
    arm_keys = [
        runner.measurement_arm_key(0.95, 1, "a", method, 0)
        for method in runner.METHODS
    ]
    summaries = [
        {
            "summary_type": "repeat",
            "target_recall": 0.95,
            "clients": 1,
            "filter_name": "a",
            "method": method,
            "measurement_repeat": 0,
        }
        for method in runner.METHODS
    ]
    checkpoint = {
        "completed_measurement_pairs": [pair_key],
        "completed_measurement_cells": [],
        "measurement_rows": len(workload.requests) * len(runner.METHODS),
        "raw_pair_artifacts": {
            pair_key: {
                "arm_keys": arm_keys,
                "methods": list(runner.METHODS),
                "rows": len(workload.requests) * len(runner.METHODS),
            }
        },
        "pair_evidence": {
            pair_key: {
                "committed": True,
                "arm_keys": arm_keys,
                "methods": list(runner.METHODS),
            }
        },
        "arm_evidence": [{"arm_key": arm_key} for arm_key in arm_keys],
        "cell_coverage_evidence": {},
    }
    evidence = runner.validate_resume_checkpoint(
        checkpoint, schedule, summaries, workload, 1
    )
    assert evidence["committed_pairs"] == 1
    assert evidence["half_pairs"] == 0

    with pytest.raises(runner.BenchmarkContractError, match="half method pair"):
        runner.validate_resume_checkpoint(
            checkpoint, schedule, summaries[:1], workload, 1
        )

    obsolete = dict(checkpoint, completed_measurement_arms=arm_keys)
    with pytest.raises(runner.BenchmarkContractError, match="obsolete arm-level"):
        runner.validate_resume_checkpoint(
            obsolete, schedule, summaries, workload, 1
        )


def test_resume_restores_configuration_and_summary_to_committed_row_prefix() -> None:
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "summary.csv"
        runner.write_csv_atomic(path, [{"arm": 0}, {"arm": 1}, {"arm": 2}])
        committed = runner.restore_csv_row_prefix(path, 2)
        assert [row["arm"] for row in committed] == ["0", "1"]
        assert len(runner.read_csv(path)) == 2

        with pytest.raises(runner.BenchmarkContractError, match="shorter"):
            runner.restore_csv_row_prefix(path, 3)


def test_stable_runtime_identity_excludes_mutable_profile_values() -> None:
    base = {
        "loaded_vector_sqlens_build_id": "sqlens-v11-a",
        "loaded_vector_so_path": "/lib/vector.so",
        "loaded_vector_so_sha256": "a" * 64,
        "required_build_prefix": "sqlens-v11-",
        "minimum_profile_semantics_version": 9,
        "profile_semantics_version": 9,
        "required_profile_fields": {"visited": 1, "returned": 2},
    }
    changed = {**base, "required_profile_fields": {"visited": 999, "returned": 0}}
    assert runner.stable_runtime_identity(base) == runner.stable_runtime_identity(changed)


def test_output_paths_keep_raw_summary_manifest_and_checkpoint_separate() -> None:
    paths = runner.output_paths(Path("/tmp/formal_raw.csv"))
    assert paths["raw"].name == "formal_raw.csv"
    assert paths["configuration"].name == "formal_matched_recall_configs.csv"
    assert paths["summary"].name == "formal_summary.csv"
    assert paths["manifest"].name == "formal_manifest.json"
    assert paths["checkpoint"].name == "formal_checkpoint.json"


def test_main_defaults_to_dry_run_and_never_starts_experiment(capsys: pytest.CaptureFixture[str]) -> None:
    with mock.patch.object(runner, "execute_experiment") as execute:
        assert runner.main([]) == 0
    execute.assert_not_called()
    payload = json.loads(capsys.readouterr().out)
    assert payload["database_connected"] is False
    assert payload["files_written"] is False
