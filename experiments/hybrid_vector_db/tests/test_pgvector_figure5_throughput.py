from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.hybrid_vector_db.scripts import figure5_frontier_artifact
from experiments.hybrid_vector_db.scripts import pgvector_figure5_throughput as runner


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(root: Path) -> runner.DatasetBinding:
    return runner.DatasetBinding(
        key="amazon",
        dataset_id="amazon10m",
        label="Amazon-10M",
        table="public.items",
        query_table="public.items",
        query_id_column="id",
        query_vector_column="embedding",
        source_index="public.source_idx",
        bfs_index="public.bfs_idx",
        candidate_validity_predicate="embedding_valid",
        truth_self_excluded=True,
        filters_csv=root / "filters.csv",
        truth_csv=root / "truth.csv",
        workload_csv=root / "workload.csv",
        d2_graph_proof_json=root / "proof.json",
    )


def settings() -> runner.SearchSettings:
    return runner.SearchSettings(
        config_id="pair-r90",
        pair_id="pair-r90",
        target_recall=0.90,
        stock=runner.ArmSearchSettings(
            ef_search=100,
            iterative_scan="strict_order",
            max_scan_tuples=5_000_000,
            scan_mem_multiplier=32.0,
            guided_collect_target=100,
            traversal_guided_target=40,
            traversal_guided_burst=8,
        ),
        sqlens=runner.ArmSearchSettings(
            ef_search=250,
            iterative_scan="off",
            max_scan_tuples=200_000,
            scan_mem_multiplier=16.0,
            guided_collect_target=250,
            traversal_guided_target=80,
            traversal_guided_burst=16,
        ),
    )


def write_fixture(root: Path) -> tuple[runner.DatasetBinding, Path]:
    item = binding(root)
    filter_names = [f"filter_{index:02d}" for index in range(runner.EXPECTED_FILTERS)]
    with item.filters_csv.open("w", newline="", encoding="utf-8") as target:
        fields = ["filter_name", "actual_pct", "predicate", "atoms"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for index, name in enumerate(filter_names):
            writer.writerow(
                {
                    "filter_name": name,
                    "actual_pct": 50.0 - index,
                    "predicate": f"category = {index}",
                    "atoms": f"sql:category = {index}",
                }
            )

    with item.workload_csv.open("w", newline="", encoding="utf-8") as target:
        fields = [
            "request_no",
            "query_no",
            "query_id",
            "filter_name",
            "trace_cycle",
            "split",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for request_no in range(runner.EXPECTED_REQUESTS):
            writer.writerow(
                {
                    "request_no": request_no,
                    "query_no": request_no,
                    "query_id": 100_000 + request_no,
                    "filter_name": filter_names[request_no % len(filter_names)],
                    "trace_cycle": 0,
                    "split": "measurement",
                }
            )

    with item.truth_csv.open("w", newline="", encoding="utf-8") as target:
        fields = [
            "query_no",
            "query_id",
            "filter_name",
            "method",
            "filtered_rows",
            "kth_distance_sq",
            "tie_tolerance",
            "strict_closer_count",
            "boundary_tied",
            "self_excluded",
            "candidate_validity_predicate",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for request_no in range(runner.EXPECTED_REQUESTS):
            writer.writerow(
                {
                    "query_no": request_no,
                    "query_id": 100_000 + request_no,
                    "filter_name": filter_names[request_no % len(filter_names)],
                    "method": "pre_filter_exact",
                    "filtered_rows": 1_000,
                    "kth_distance_sq": 1.0,
                    "tie_tolerance": 1e-6,
                    "strict_closer_count": 9,
                    "boundary_tied": "false",
                    "self_excluded": "true",
                    "candidate_validity_predicate": "embedding_valid",
                }
            )

    item.d2_graph_proof_json.write_text(
        json.dumps({"fixture": True}), encoding="utf-8"
    )
    manifest_path = root / "workload_manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_type": "figure5_frontier_workload",
        "artifact_valid": True,
        "gates": {
            name: True
            for name in (
                "exactly_14_filters",
                "input_sha256_bound",
                "measurement_filter_balance",
                "measurement_filter_coverage",
                "measurement_query_no_uniqueness",
                "measurement_query_vector_uniqueness",
                "measurement_request_count",
                "output_sha256_verified",
                "truth_pair_coverage",
                "truth_tie_aware",
            )
        },
        "inputs": {
            "truth_csv": {
                "path": str(item.truth_csv),
                "rows": runner.EXPECTED_REQUESTS,
                "sha256": sha(item.truth_csv),
            },
            "filters_csv": {
                "path": str(item.filters_csv),
                "rows": runner.EXPECTED_FILTERS,
                "sha256": sha(item.filters_csv),
            },
        },
        "outputs": {
            "measurement_workload_csv": {
                "path": str(item.workload_csv),
                "rows": runner.EXPECTED_REQUESTS,
                "sha256": sha(item.workload_csv),
            }
        },
        "truth": {"valid": True},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return item, manifest_path


def test_loads_frozen_mixed_q10k_with_full_assigned_truth(tmp_path: Path) -> None:
    item, manifest = write_fixture(tmp_path)
    workload = runner.load_frozen_workload(item, manifest)

    assert len(workload.requests) == runner.EXPECTED_REQUESTS
    assert len({request.query_id for request in workload.requests}) == 10_000
    assert len(workload.filters) == runner.EXPECTED_FILTERS
    assert len(
        {(request.filter_name, request.query_no) for request in workload.requests}
    ) == 10_000
    counts = {
        name: sum(request.filter_name == name for request in workload.requests)
        for name in workload.filters
    }
    assert max(counts.values()) - min(counts.values()) == 1


def test_missing_assigned_truth_and_nonunique_query_are_rejected(
    tmp_path: Path,
) -> None:
    item, manifest = write_fixture(tmp_path)
    truth_rows = list(csv.DictReader(item.truth_csv.open(newline="", encoding="utf-8")))
    with item.truth_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(truth_rows[0]))
        writer.writeheader()
        writer.writerows(truth_rows[:-1])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inputs"]["truth_csv"]["sha256"] = sha(item.truth_csv)
    payload["inputs"]["truth_csv"]["rows"] -= 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((runner.Figure5ThroughputError, ValueError), match="truth|Truth"):
        runner.load_frozen_workload(item, manifest)

    item, manifest = write_fixture(tmp_path)
    rows = list(csv.DictReader(item.workload_csv.open(newline="", encoding="utf-8")))
    rows[-1]["query_id"] = rows[-2]["query_id"]
    rows[-1]["query_no"] = rows[-2]["query_no"]
    rows[-1]["filter_name"] = rows[-2]["filter_name"]
    with item.workload_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["outputs"]["measurement_workload_csv"]["sha256"] = sha(item.workload_csv)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unique query"):
        runner.load_frozen_workload(item, manifest)


def test_request_permutation_is_paired_and_distinct_across_repeats(
    tmp_path: Path,
) -> None:
    item, manifest = write_fixture(tmp_path)
    workload = runner.load_frozen_workload(item, manifest)
    first = runner.request_dispatch(
        workload,
        schedule_seed=57,
        dataset_id="amazon10m",
        config_id="ef100",
        clients=8,
        repeat_id=0,
    )
    paired = runner.request_dispatch(
        workload,
        schedule_seed=57,
        dataset_id="amazon10m",
        config_id="ef100",
        clients=8,
        repeat_id=0,
    )
    second = runner.request_dispatch(
        workload,
        schedule_seed=57,
        dataset_id="amazon10m",
        config_id="ef100",
        clients=8,
        repeat_id=1,
    )
    assert first[0:2] == paired[0:2]
    assert [request.request_no for _, request in first[2]] == [
        request.request_no for _, request in paired[2]
    ]
    assert first[0] != second[0]
    assert first[1] != second[1]


def test_arm_schedule_is_balanced_and_modes_are_exact() -> None:
    schedule = runner.validate_balanced_schedule(7, 20260728)
    first = Counter(order[0] for order in schedule)
    assert max(first.values()) - min(first.values()) == 1
    assert all(set(order) == set(runner.MODES) for order in schedule)
    assert runner.ARM_BY_MODE == {
        "original": "stock_pgvector",
        "design1_bloom_bfs_layout_d3": "sqlens_full",
    }
    with pytest.raises(runner.Figure5ThroughputError, match="at least 6"):
        runner.validate_balanced_schedule(5, 1)
    single = runner.validate_balanced_schedule(1, 1, allow_single_pass=True)
    assert len(single) == 1
    assert set(single[0]) == set(runner.MODES)


def test_search_config_binds_full_d3_and_stock_without_method_mixing(
    tmp_path: Path,
) -> None:
    item = binding(tmp_path)
    config, digest = runner.config_identity(item, settings(), 16)
    assert len(digest) == 64
    assert config["pair_id"] == "pair-r90"
    assert config["target_recall"] == 0.90
    assert config["search"]["original"]["ef_search"] == 100
    assert config["search"]["design1_bloom_bfs_layout_d3"]["ef_search"] == 250
    assert config["search"]["original"]["iterative_scan"] == "strict_order"
    assert config["search"]["original"]["traversal_guided_prioritization"] is False
    assert config["search"]["design1_bloom_bfs_layout_d3"][
        "traversal_guided_prioritization"
    ] is True
    assert config["guidance_policy"] == {
        "guidance_selectivity_min_pct": 0.0,
        "guidance_selectivity_max_pct": 6.0,
        "guidance_composite_max_selectivity_pct": 100.0,
        "guidance_max_atoms": 160,
        "d1_exact_max_selectivity_pct": 6.0,
            "collapse_exact_and_guidance": True,
            "d2_source_on_guidance_bypass": True,
            "guidance_bypass_ef_search": 0,
            "guidance_low_selectivity_bypass_ef_search": 0,
        }
    assert config["search"]["design1_bloom_bfs_layout_d3"]["iterative_scan"] == "off"
    assert config["d3_measurement_policy"] == "workload_driven_adaptive"
    assert set(config["modes"]) == set(runner.MODES)
    assert runner.arm_config_sha256(settings(), "stock_pgvector") != runner.arm_config_sha256(
        settings(), "sqlens_full"
    )
    _, other_clients_digest = runner.config_identity(item, settings(), 8)
    assert other_clients_digest == digest


def test_current_yfcc_sqlens_settings_bind_early_stop_target_burst_and_policy(
    tmp_path: Path,
) -> None:
    args = runner.create_argument_parser().parse_args(
        [
            "--dataset", "yfcc",
            "--workload-manifest", str(tmp_path / "workload.json"),
            "--pair-id", "yfcc-r39-t130",
            "--target-recall", "0.90",
            "--stock-ef-search", "100",
            "--sqlens-ef-search", "130",
            "--sqlens-traversal-guided-target", "130",
            "--sqlens-traversal-guided-burst", "1",
            "--sqlens-traversal-guided-early-stop",
            "--clients", "1", "--out-prefix", str(tmp_path / "out"),
        ]
    )
    resolved = runner.resolve_search_settings(
        args,
        {"max_scan_tuples": 5_000_000, "scan_mem_multiplier": 32.0},
        minimum_traversal_target=10,
    )

    assert resolved.sqlens.ef_search == 130
    assert resolved.sqlens.traversal_guided_target == 130
    assert resolved.sqlens.traversal_guided_burst == 1
    assert resolved.sqlens.traversal_guided_early_stop is True
    assert resolved.d1_exact_max_selectivity_pct == 6.0
    assert resolved.collapse_exact_and_guidance is True
    assert resolved.guidance_selectivity_max_pct == 6.0
    assert resolved.d2_source_on_guidance_bypass is True
    assert resolved.mode_configs()["design1_bloom_bfs_layout_d3"][
        "traversal_guided_early_stop"
    ] is True

    runtime = runner._runtime_args(
        args,
        binding(tmp_path),
        runner.FrozenWorkload(
            requests=(),
            truth={},
            filters={},
            filter_tuples=(),
            filter_atoms={},
            trace_sha256="",
            truth_sha256="",
            filters_sha256="",
            workload_manifest={},
            workload_manifest_sha256="",
        ),
        resolved,
        {
            "expected_sqlens_build_id": "sqlens-v16-guided-test",
            "expected_vector_so_sha256": "a" * 64,
        },
    )
    assert runtime.d1_exact_max_selectivity_pct == 6.0
    assert runtime.collapse_exact_and_guidance is True
    assert runtime.d2_source_on_guidance_bypass is True
    assert runtime.mode_configs_json["design1_bloom_bfs_layout_d3"][
        "traversal_guided_target"
    ] == 130


def test_per_filter_search_settings_are_parsed_and_sha_bound(tmp_path: Path) -> None:
    ef_path = tmp_path / "ef.json"
    target_path = tmp_path / "target.json"
    ef_path.write_text(
        json.dumps(
            {
                "original": {"filter_00": 700},
                "design1_bloom_bfs_layout_d3": {"filter_00": 800},
            }
        ),
        encoding="utf-8",
    )
    target_path.write_text(
        json.dumps(
            {"design1_bloom_bfs_layout_d3": {"filter_00": 55}}
        ),
        encoding="utf-8",
    )
    args = runner.create_argument_parser().parse_args(
        [
            "--dataset", "laion",
            "--workload-manifest", str(tmp_path / "workload.json"),
            "--pair-id", "laion-per-filter-r90",
            "--target-recall", "0.90",
            "--stock-ef-search", "1000",
            "--sqlens-ef-search", "1000",
            "--filter-ef-search-json", str(ef_path),
            "--filter-traversal-target-json", str(target_path),
            "--clients", "16",
            "--out-prefix", str(tmp_path / "out"),
        ]
    )
    resolved = runner.resolve_search_settings(
        args,
        {"max_scan_tuples": 5_000_000, "scan_mem_multiplier": 32.0},
        minimum_traversal_target=10,
    )

    assert resolved.filter_ef_search["original"]["filter_00"] == 700
    assert (
        resolved.filter_traversal_target["design1_bloom_bfs_layout_d3"][
            "filter_00"
        ]
        == 55
    )
    config, _ = runner.config_identity(binding(tmp_path), resolved, 16)
    assert config["per_filter_search"]["stock_pgvector"]["ef_search"] == {
        "filter_00": 700
    }
    assert config["per_filter_search"]["sqlens_full"][
        "traversal_guided_target"
    ] == {"filter_00": 55}
    without_overrides = settings()
    assert runner.arm_config_sha256(resolved, "stock_pgvector") != (
        runner.arm_config_sha256(without_overrides, "stock_pgvector")
    )

    runtime = runner._runtime_args(
        args,
        binding(tmp_path),
        runner.FrozenWorkload(
            requests=(),
            truth={},
            filters={},
            filter_tuples=(),
            filter_atoms={},
            trace_sha256="",
            truth_sha256="",
            filters_sha256="",
            workload_manifest={},
            workload_manifest_sha256="",
        ),
        resolved,
        {
            "expected_sqlens_build_id": "sqlens-v16-guided-test",
            "expected_vector_so_sha256": "a" * 64,
        },
    )
    assert runtime.filter_ef_search_json == resolved.filter_ef_search
    assert (
        runtime.filter_traversal_target_json
        == resolved.filter_traversal_target
    )


def test_measurement_plan_consumes_independent_per_arm_settings_and_binds_manifest(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "measurement-plan.json"
    plan.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": "r95-a",
                        "target_recall": 0.95,
                        "stock_ef_search": 80,
                        "stock_iterative_scan": "strict_order",
                        "stock_max_scan_tuples": 100_000,
                        "stock_scan_mem_multiplier": 8.0,
                        "stock_guided_collect_target": 80,
                        "stock_traversal_guided_target": 40,
                        "stock_traversal_guided_burst": 8,
                        "stock_traversal_guided_early_stop": False,
                        "sqlens_ef_search": 400,
                        "sqlens_iterative_scan": "off",
                        "sqlens_max_scan_tuples": 300_000,
                        "sqlens_scan_mem_multiplier": 24.0,
                        "sqlens_guided_collect_target": 400,
                        "sqlens_traversal_guided_target": 100,
                        "sqlens_traversal_guided_burst": 16,
                        "sqlens_traversal_guided_early_stop": True,
                        "guidance_policy": {
                            "guidance_selectivity_min_pct": 1.0,
                            "guidance_selectivity_max_pct": 6.0,
                            "guidance_bypass_ef_search": 500,
                            "guidance_low_selectivity_bypass_ef_search": 2000,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = runner.create_argument_parser().parse_args(
        [
            "--dataset",
            "amazon",
            "--workload-manifest",
            str(tmp_path / "workload-manifest.json"),
            "--measurement-plan",
            str(plan),
            "--pair-id",
            "r95-a",
            "--clients",
            "1",
            "--out-prefix",
            str(tmp_path / "out"),
        ]
    )
    resolved = runner.resolve_search_settings(
        args,
        {"max_scan_tuples": 5_000_000, "scan_mem_multiplier": 32.0},
        minimum_traversal_target=10,
    )
    assert resolved.pair_id == "r95-a"
    assert resolved.target_recall == 0.95
    assert resolved.stock.ef_search == 80
    assert resolved.sqlens.ef_search == 400
    assert resolved.sqlens.traversal_guided_early_stop is True
    assert resolved.stock.max_scan_tuples != resolved.sqlens.max_scan_tuples
    assert resolved.guidance_selectivity_min_pct == 1.0
    assert resolved.guidance_selectivity_max_pct == 6.0
    assert resolved.guidance_bypass_ef_search == 500
    assert resolved.guidance_low_selectivity_bypass_ef_search == 2000

    item, manifest_path = write_fixture(tmp_path)
    workload = runner.load_frozen_workload(item, manifest_path)
    plan_manifest = runner.prospective_manifest(
        args,
        item,
        {},
        {"release": {"expected_sqlens_build_id": "sqlens-v16-d3-test"}},
        workload,
        resolved,
        {"stable_fingerprint_sha256": "a" * 64},
        6,
        (),
    )
    configuration = plan_manifest["configuration"]
    assert configuration["pair_id"] == "r95-a"
    assert configuration["target_recall"] == 0.95
    assert configuration["stock_config_sha256"] != configuration["sqlens_config_sha256"]
    assert plan_manifest["measurement_pair"]["pair_id"] == "r95-a"
    assert plan_manifest["measurement_pair"]["sha256"] == sha(plan)


def test_legacy_single_ef_requires_explicit_equal_arm_opt_in(tmp_path: Path) -> None:
    parser = runner.create_argument_parser()
    args = parser.parse_args(
        [
            "--dataset", "amazon", "--workload-manifest", str(tmp_path / "workload.json"),
            "--pair-id", "equal-r90", "--target-recall", "0.90", "--ef-search", "100",
            "--clients", "1", "--out-prefix", str(tmp_path / "out"),
        ]
    )
    with pytest.raises(runner.Figure5ThroughputError, match="legacy --ef-search"):
        runner.resolve_search_settings(
            args,
            {"max_scan_tuples": 5_000_000, "scan_mem_multiplier": 32.0},
            minimum_traversal_target=10,
        )
    args.allow_equal_arm_settings = True
    resolved = runner.resolve_search_settings(
        args,
        {"max_scan_tuples": 5_000_000, "scan_mem_multiplier": 32.0},
        minimum_traversal_target=10,
    )
    assert resolved.stock.ef_search == resolved.sqlens.ef_search == 100


def test_measurement_plan_csv_is_supported(tmp_path: Path) -> None:
    plan = tmp_path / "measurement-plan.csv"
    fields = [
        "pair_id", "target_recall",
        *(f"{arm}_{field}" for arm in ("stock", "sqlens") for field in runner.ARM_SEARCH_FIELDS),
    ]
    row = {
        "pair_id": "r90-csv",
        "target_recall": "0.90",
        "stock_ef_search": "60",
        "stock_iterative_scan": "relaxed_order",
        "stock_max_scan_tuples": "100000",
        "stock_scan_mem_multiplier": "8",
        "stock_guided_collect_target": "60",
        "stock_traversal_guided_target": "30",
        "stock_traversal_guided_burst": "4",
        "stock_traversal_guided_early_stop": False,
        "sqlens_ef_search": "180",
        "sqlens_iterative_scan": "off",
        "sqlens_max_scan_tuples": "200000",
        "sqlens_scan_mem_multiplier": "16",
        "sqlens_guided_collect_target": "180",
        "sqlens_traversal_guided_target": "80",
        "sqlens_traversal_guided_burst": "8",
        "sqlens_traversal_guided_early_stop": True,
    }
    with plan.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    args = runner.create_argument_parser().parse_args(
        [
            "--dataset", "amazon", "--workload-manifest", str(tmp_path / "workload.json"),
            "--measurement-plan", str(plan), "--pair-id", "r90-csv",
            "--clients", "1", "--out-prefix", str(tmp_path / "out"),
        ]
    )
    resolved = runner.resolve_search_settings(
        args,
        {"max_scan_tuples": 5_000_000, "scan_mem_multiplier": 32.0},
        minimum_traversal_target=10,
    )
    assert resolved.stock.iterative_scan == "relaxed_order"
    assert resolved.stock.ef_search == 60
    assert resolved.sqlens.ef_search == 180


def test_client_cpu_assignment_supports_client_ids_zero_through_31() -> None:
    cpus = runner.client_cpu_assignment("32-63", 32)
    assert len(cpus) == 32
    assert cpus[0] == 32
    assert cpus[31] == 63
    with pytest.raises(runner.Figure5ThroughputError, match="\\[1, 32\\]"):
        runner.client_cpu_assignment("0-63", 33)
    with pytest.raises(runner.Figure5ThroughputError, match="CPUs"):
        runner.client_cpu_assignment("0-3", 8)


def test_concurrency_and_independent_backend_gates_are_fail_closed() -> None:
    overlapping = [
        {
            "client_id": 0,
            "started_offset_ms": 0.0,
            "completed_offset_ms": 10.0,
        },
        {
            "client_id": 1,
            "started_offset_ms": 1.0,
            "completed_offset_ms": 11.0,
        },
    ]
    serialized = [
        {
            "client_id": 0,
            "started_offset_ms": 0.0,
            "completed_offset_ms": 1.0,
        },
        {
            "client_id": 1,
            "started_offset_ms": 2.0,
            "completed_offset_ms": 3.0,
        },
    ]
    assert runner._has_cross_client_overlap(overlapping, 2) is True
    assert runner._has_cross_client_overlap(serialized, 2) is False

    runtimes = [
        argparse.Namespace(backend_cpu_provenance={"backend_pid": pid})
        for pid in (101, 102)
    ]
    assert runner.validate_independent_backends(runtimes, 2) == [101, 102]
    runtimes[1].backend_cpu_provenance["backend_pid"] = 101
    with pytest.raises(runner.Figure5ThroughputError, match="independent"):
        runner.validate_independent_backends(runtimes, 2)


def test_repeat_output_schema_is_directly_accepted_by_figure5_artifact() -> None:
    assert figure5_frontier_artifact.REQUIRED_INPUT_FIELDS <= set(
        runner.REPEAT_FIELDS
    )
    assert not [
        field
        for field in runner.REPEAT_FIELDS
        if "qps" in field and field != "throughput_qps"
    ]


def request_rows(count: int = 10_000) -> list[dict[str, object]]:
    return [
        {
            "query_id": index,
            "latency_ms": 1.0 + (index % 100),
            "recall_at_10": 0.9,
            "error": "",
        }
        for index in range(count)
    ]


def telemetry_fixture() -> dict[str, object]:
    return {
        "host": {
            "cpu": {
                "utilization_pct": 50.0,
                "user_pct": 30.0,
                "system_pct": 10.0,
                "iowait_pct": 5.0,
            },
            "disk_total": {
                "reads_completed": 1.0,
                "read_bytes": 4096.0,
                "read_time_ms": 1.0,
                "writes_completed": 0.0,
                "write_bytes": 0.0,
                "write_time_ms": 0.0,
                "io_time_ms": 1.0,
                "weighted_io_time_ms": 1.0,
            },
        },
        "postgresql": {
            "database": {
                "blks_read": 1.0,
                "blks_hit": 10.0,
                "temp_files": 0.0,
                "temp_bytes": 0.0,
                "blk_read_time": 1.0,
                "blk_write_time": 0.0,
            },
            "io_total": {
                "reads": 1.0,
                "read_bytes": 8192.0,
                "read_time": 1.0,
                "writes": 0.0,
                "write_bytes": 0.0,
                "write_time": 0.0,
                "hits": 10.0,
                "evictions": 0.0,
            },
            "relations": {
                "table": {
                    "relid": 1,
                    "heap_blks_read": 1.0,
                    "heap_blks_hit": 2.0,
                    "idx_blks_read": 3.0,
                    "idx_blks_hit": 4.0,
                },
                "index": {
                    "relid": 1,
                    "indexrelid": 2,
                    "idx_blks_read": 3.0,
                    "idx_blks_hit": 4.0,
                },
            },
        },
        "backend_cpu": {
            "backend_pids": [101],
            "total": {
                "user_cpu_ms": 10.0,
                "system_cpu_ms": 2.0,
                "total_cpu_ms": 12.0,
            },
        },
        "devices": ["nvme0n1"],
    }


def test_repeat_summary_uses_only_completed_over_barrier_wall(
    tmp_path: Path,
) -> None:
    rows = request_rows()
    summary = runner.summarize_repeat(
        rows,
        wall_seconds=20.0,
        run_id="run",
        binding=binding(tmp_path),
        settings=settings(),
        config_sha256="1" * 64,
        release_identity_sha256="2" * 64,
        arm_id="stock_pgvector",
        mode_id="original",
        arm_order=0,
        repeat_id=0,
        clients=1,
        request_trace_sha256="3" * 64,
        trace_seed=9,
        trace_order_sha256="4" * 64,
        backend_pids=[101],
        backend_cpu_provenance=[
            {
                "backend_pid": 101,
                "requested_cpu_list": "48-63",
                "observed_cpu_list": "48-63",
                "exact_match": True,
            }
        ],
        client_affinity=[
            {
                "client_id": 0,
                "native_tid": 1,
                "requested_cpu": 0,
                "affinity_applied": True,
            }
        ],
        true_concurrency_observed=True,
        namespace="",
        namespace_rows_before=0,
        namespace_rows_after=0,
        arm_telemetry=telemetry_fixture(),
        bootstrap_samples=20,
        bootstrap_seed=5,
    )
    assert summary["throughput_qps"] == 500.0
    assert summary["throughput_source"] == runner.THROUGHPUT_SOURCE
    assert summary["pair_id"] == "pair-r90"
    assert summary["target_recall"] == 0.90
    assert summary["stock_config_sha256"] != summary["sqlens_config_sha256"]
    assert summary["arm_config_sha256"] == summary["stock_config_sha256"]
    assert summary["completed_queries"] == 10_000
    assert summary["latency_p95_ms"] == 95.0
    assert summary["latency_p99_ms"] == 99.0
    assert summary["status"] == "valid"
    assert "single_client_throughput" not in summary


def test_paired_repeat_gate_rejects_trace_drift(tmp_path: Path) -> None:
    base = {
        "run_id": "run",
        "dataset": "amazon10m",
        "pair_id": "pair-r90",
        "target_recall": 0.90,
        "config_id": "ef100",
        "config_sha256": "1" * 64,
        "stock_config_sha256": "4" * 64,
        "sqlens_config_sha256": "5" * 64,
        "release_identity_sha256": "2" * 64,
        "clients": 8,
        "request_trace_sha256": "3" * 64,
        "requests": 10_000,
        "unique_queries": 10_000,
    }
    summaries = []
    arm_order = 0
    for repeat in range(6):
        for arm in runner.balanced_arm_order(repeat, 7):
            summaries.append(
                {
                    **base,
                    "repeat_id": repeat,
                    "arm_id": runner.ARM_BY_MODE[arm],
                    "trace_permutation_seed": repeat,
                    "trace_order_sha256": f"{repeat:064x}",
                    "arm_order": arm_order,
                }
            )
            arm_order += 1
    assert runner.validate_paired_repeats(summaries, 6)["passed"] is True
    summaries[1]["trace_order_sha256"] = "f" * 64
    with pytest.raises(runner.Figure5ThroughputError, match="trace_order_sha256"):
        runner.validate_paired_repeats(summaries, 6)


def test_release_and_d2_proof_gates_are_fail_closed(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_sqlens_build_id": "sqlens-v16-d3-test",
                "expected_vector_so_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    assert runner.validate_release_contract(release)["sha256"] == sha(release)
    payload = json.loads(release.read_text(encoding="utf-8"))
    payload["expected_sqlens_build_id"] = (
        "sqlens-v16-guided-early-stop-batched-route-ef500k-20260801-r39"
    )
    release.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.validate_release_contract(release)["expected_sqlens_build_id"].startswith(
        "sqlens-v16-guided-"
    )
    payload["expected_sqlens_build_id"] = "sqlens-v16-distance-aware-test"
    release.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.validate_release_contract(release)["expected_sqlens_build_id"].startswith(
        "sqlens-v16-distance-aware-"
    )
    payload = json.loads(release.read_text(encoding="utf-8"))
    payload["expected_vector_so_sha256"] = "bad"
    release.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.Figure5ThroughputError, match="sha256|SHA"):
        runner.validate_release_contract(release)

    item = binding(tmp_path)
    item.d2_graph_proof_json.write_text("{}", encoding="utf-8")
    calls = []

    def validator(proof, source, clone):
        calls.append((proof, source, clone))
        return {"stable_fingerprint_sha256": "b" * 64}

    result = runner.load_delegated_d2_proof(
        item.d2_graph_proof_json,
        item,
        validator=validator,
    )
    assert result["stable_fingerprint_sha256"] == "b" * 64
    assert calls[0][1:] == (item.source_index, item.bfs_index)


def test_runtime_binary_identity_comparison_ignores_only_check_timestamp() -> None:
    identity = {
        "expected_build_id": "sqlens-v16-d3-test",
        "expected_vector_so_sha256": "a" * 64,
        "observed_build_id": "sqlens-v16-d3-test",
        "observed_vector_so_path": "/usr/lib/postgresql/vector.so",
        "observed_vector_so_sha256": "a" * 64,
        "exact_match": True,
        "checked_at": "2026-07-28T01:00:00Z",
    }
    later = {**identity, "checked_at": "2026-07-28T02:00:00Z"}
    assert runner.stable_runtime_identity(identity) == runner.stable_runtime_identity(
        later
    )

    changed = {**later, "observed_vector_so_sha256": "b" * 64}
    with pytest.raises(runner.Figure5ThroughputError, match="exact loaded-binary"):
        runner.stable_runtime_identity(changed)


def test_d3_namespace_is_repeat_fresh_and_protocol_has_no_unmeasured_query() -> None:
    first = runner.d3_namespace("run", "amazon10m", "ef100", 8, 0)
    second = runner.d3_namespace("run", "amazon10m", "ef100", 8, 1)
    assert first != second
    assert len(first) <= 64
    assert runner.MODE_BY_ARM["sqlens_full"] == "design1_bloom_bfs_layout_d3"


def test_execute_requires_separate_client_backend_and_telemetry_binding() -> None:
    args = argparse.Namespace(
        schedule_seed=1,
        client_cpu_list="0-7",
        clients=8,
        execute=True,
        backend_cpu_list=None,
        telemetry_devices="nvme0n1",
        telemetry_path=[],
        d3_min_benefit_per_byte=0.0,
        d3_page_min_skip_rate=0.05,
    )
    with pytest.raises(runner.Figure5ThroughputError, match="backend-cpu-list"):
        runner.validate_execution_args(args, 6)
    args.backend_cpu_list = "48-63"
    args.telemetry_devices = None
    with pytest.raises(runner.Figure5ThroughputError, match="telemetry"):
        runner.validate_execution_args(args, 6)
    args.telemetry_devices = "nvme0n1"
    assert runner.validate_execution_args(args, 6) == tuple(range(8))


def test_throughput_search_routes_before_activation(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    args = argparse.Namespace(
        k=10,
        guidance_filter_strategy="traversal_guided",
        candidate_validity_predicate="TRUE",
        query_id_column="qid",
        query_vector_column="embedding",
    )
    runtime = SimpleNamespace(cur=object(), mode="design1_bloom_bfs_layout_d3")
    request = runner.core.WorkloadRequest(0, 7, 10007, "wide", 0, "measurement")
    workload = SimpleNamespace(
        filters={
            "wide": runner.FilterSpec(
                name="wide",
                predicate="tags && ARRAY[1]::int[]",
                actual_pct=50.0,
                atoms=("sql:tags @> ARRAY[1]::int[]",),
            )
        },
        truth={("wide", 7): object()},
    )

    def route(_args, _runtime, filter_name):
        calls.append(("route", filter_name))
        return "public.items", "public.source_idx", True, True

    def activate(_cur, _args, _mode, filter_name, **kwargs):
        calls.append(("activate", (filter_name, kwargs)))
        return {"table": "public.items", "index": "public.source_idx"}

    monkeypatch.setattr(runner.core, "route_runtime_request", route)
    monkeypatch.setattr(runner.core, "activate", activate)
    monkeypatch.setattr(runner.core, "activation_binding", lambda *args: None)
    monkeypatch.setattr(runner.core, "candidate_self_exclusion", lambda *args: False)
    monkeypatch.setattr(runner.core, "query_table_for_candidate", lambda *args: "q")
    monkeypatch.setattr(
        runner.core,
        "run_query",
        lambda *args, **kwargs: ([1, 2], [0.1, 0.2], {}),
    )
    monkeypatch.setattr(runner.core, "tie_aware_recall", lambda *args: 0.95)

    result = runner._execute_search(args, runtime, request, workload)

    assert [name for name, _ in calls] == ["route", "activate"]
    _, activation = calls[1]
    assert activation[1]["configure_search_strategy"] is False
    assert activation[1]["reset_bypass_guidance"] is False
    assert result["error"] == ""
    assert result["recall_at_10"] == 0.95
