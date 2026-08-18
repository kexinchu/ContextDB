from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from experiments.hybrid_vector_db.scripts import (
    pgvector_design1_design2_design3_selectivity_benchmark as benchmark,
)


def write_workload(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truth_entry(query_id: int) -> benchmark.TruthEntry:
    return benchmark.TruthEntry(
        query_id=query_id,
        filtered_rows=10,
        kth_distance_sq=1.0,
        tie_tolerance=1e-6,
        self_excluded=True,
    )


def test_load_workload_requests_requires_exact_unique_q10k_contract(tmp_path: Path) -> None:
    path = tmp_path / "workload.csv"
    rows = [
        {
            "request_no": 0,
            "query_no": 200,
            "query_id": 1200,
            "filter_name": "f1",
            "trace_cycle": 0,
            "split": "measurement",
        },
        {
            "request_no": 1,
            "query_no": 201,
            "query_id": 1201,
            "filter_name": "f2",
            "trace_cycle": 0,
            "split": "measurement",
        },
    ]
    write_workload(path, rows)
    filters = [("f1", 1.0, "a = 1"), ("f2", 2.0, "b = 2")]
    truth = {
        ("f1", 200): truth_entry(1200),
        ("f2", 201): truth_entry(1201),
    }

    requests = benchmark.load_workload_requests(
        path,
        query_by_no={200: 1200, 201: 1201},
        filters=filters,
        truth=truth,
        expected_requests=2,
    )

    assert [request.request_no for request in requests] == [0, 1]
    assert {request.filter_name for request in requests} == {"f1", "f2"}

    rows[1]["query_no"] = 200
    rows[1]["query_id"] = 1200
    write_workload(path, rows)
    truth[("f2", 200)] = truth_entry(1200)
    with pytest.raises(ValueError, match="unique query vector"):
        benchmark.load_workload_requests(
            path,
            query_by_no={200: 1200},
            filters=filters,
            truth=truth,
            expected_requests=2,
        )


def test_load_workload_requests_validates_source_then_limits_prefix(tmp_path: Path) -> None:
    path = tmp_path / "workload.csv"
    rows = [
        {
            "request_no": request_no,
            "query_no": 200 + request_no,
            "query_id": 1200 + request_no,
            "filter_name": "f1" if request_no % 2 == 0 else "f2",
            "trace_cycle": 0,
            "split": "measurement",
        }
        for request_no in range(4)
    ]
    write_workload(path, rows)
    filters = [("f1", 1.0, "a = 1"), ("f2", 2.0, "b = 2")]
    truth = {
        (row["filter_name"], row["query_no"]): truth_entry(int(row["query_id"]))
        for row in rows
    }

    requests = benchmark.load_workload_requests(
        path,
        query_by_no={int(row["query_no"]): int(row["query_id"]) for row in rows},
        filters=filters,
        truth=truth,
        expected_requests=4,
        request_limit=2,
    )

    assert [request.request_no for request in requests] == [0, 1]
    assert {request.filter_name for request in requests} == {"f1", "f2"}

    selected = benchmark.load_workload_requests(
        path,
        query_by_no={int(row["query_no"]): int(row["query_id"]) for row in rows},
        filters=filters,
        truth=truth,
        expected_requests=4,
        request_limit=3,
        selected_filter_names={"f1"},
    )

    assert [request.request_no for request in selected] == [0, 1]
    assert [request.query_no for request in selected] == [200, 202]


def test_mixed_workload_interleaves_the_same_request_across_modes() -> None:
    modes = ["original", "design1_bloom_bfs_layout_d3"]
    filters = [("f1", 1.0, "a = 1"), ("f2", 2.0, "b = 2")]
    requests = [
        benchmark.WorkloadRequest(0, 200, 1200, "f1", 0, "measurement"),
        benchmark.WorkloadRequest(1, 201, 1201, "f2", 0, "measurement"),
        benchmark.WorkloadRequest(2, 202, 1202, "f1", 0, "measurement"),
    ]
    args = argparse.Namespace(
        modes=modes,
        d3_measurement_policy="workload_driven_adaptive",
        warmup_queries=0,
        warmup_all_queries=False,
        repeats=2,
        schedule_seed=17,
        progress_queries=0,
    )

    def fake_measured(
        _args,
        runtime,
        filter_name,
        _selectivity,
        _predicate,
        query_no,
        _query_id,
        repeat,
        _truth,
        schedule_position,
        block_no=0,
        query_order_position=0,
    ):
        return {
            "mode": runtime.mode,
            "filter_name": filter_name,
            "query_no": query_no,
            "repeat": repeat,
            "schedule_position": schedule_position,
            "block_no": block_no,
            "query_order_position": query_order_position,
            "error": "",
        }

    with (
        mock.patch.object(
            benchmark,
            "open_mode_runtime",
            side_effect=lambda _args, mode, _filters: SimpleNamespace(mode=mode),
        ),
        mock.patch.object(benchmark, "close_mode_runtime"),
        mock.patch.object(
            benchmark, "run_measured_query", side_effect=fake_measured
        ),
    ):
        rows = benchmark.run_interleaved(
            args,
            filters,
            [200, 201, 202],
            {200: 1200, 201: 1201, 202: 1202},
            truth={},
            workload_requests=requests,
        )

    assert len(rows) == len(modes) * len(requests) * args.repeats
    for repeat in range(args.repeats):
        repeat_rows = [row for row in rows if row["repeat"] == repeat]
        for request_no in range(len(requests)):
            pair = [row for row in repeat_rows if row["request_no"] == request_no]
            assert {row["mode"] for row in pair} == set(modes)
            assert len({row["query_no"] for row in pair}) == 1
            assert len({row["trace_cycle"] for row in pair}) == 1


def test_workload_lifecycle_accepts_admitted_and_bypassed_d3_filters() -> None:
    modes = ["original", "design1_bloom_bfs_layout_d3"]
    requests = [
        benchmark.WorkloadRequest(0, 200, 1200, "admitted", 0, "measurement"),
        benchmark.WorkloadRequest(1, 201, 1201, "admitted", 0, "measurement"),
        benchmark.WorkloadRequest(2, 202, 1202, "admitted", 0, "measurement"),
        benchmark.WorkloadRequest(3, 203, 1203, "bypassed", 0, "measurement"),
        benchmark.WorkloadRequest(4, 204, 1204, "bypassed", 0, "measurement"),
    ]
    args = argparse.Namespace(
        modes=modes,
        d3_measurement_policy="workload_driven_adaptive",
        warmup_all_queries=False,
        warmup_queries=0,
        warmup_evidence=[],
        d3_phase_evidence=[
            {"filter_name": "admitted", "d3_phase": "probe"},
            {"filter_name": "admitted", "d3_phase": "admission"},
            {"filter_name": "admitted", "d3_phase": "warm"},
            {"filter_name": "bypassed", "d3_phase": "probe"},
            {"filter_name": "bypassed", "d3_phase": "bypass"},
        ],
        d3_warmup_phase_evidence=[],
        repeats=1,
        backend_cpu_list="48-51",
        backend_cpu_evidence=[
            {
                "mode": mode,
                "backend_pid": 100 + position,
                "requested_cpu_list": "48-51",
                "observed_cpu_list": "48-51",
                "exact_match": True,
                "pinning_attempted_by_runner": False,
            }
            for position, mode in enumerate(modes)
        ],
        expected_sqlens_build_id="sqlens-v16-test",
        expected_vector_so_sha256="a" * 64,
        runtime_sqlens_identity_evidence=[
            {
                "mode": mode,
                "exact_match": True,
                "expected_build_id": "sqlens-v16-test",
                "expected_vector_so_sha256": "a" * 64,
            }
            for mode in modes
        ],
    )

    evidence = benchmark.validate_execution_lifecycle(
        args,
        [("admitted", 1.0, "a = 1"), ("bypassed", 50.0, "b = 1")],
        [request.query_no for request in requests],
        requests,
    )

    assert evidence["d3_expected_measured_requests"] == len(requests)
    assert evidence["d3_phase_counts"]["admitted"]["warm"] == 1
    assert evidence["d3_phase_counts"]["bypassed"]["bypass"] == 1


def test_relation_prewarm_records_complete_main_forks() -> None:
    connection = mock.MagicMock()
    cursor = connection.cursor.return_value
    cursor.fetchone.side_effect = [
        (101, 201, 16_384, 8_192),
        (2,),
        (102, 202, 24_576, 8_192),
        (3,),
    ]
    with mock.patch.object(benchmark.psycopg, "connect", return_value=connection):
        evidence = benchmark.prewarm_relations(["public.heap", "public.hnsw"])

    assert evidence["enabled"] is True
    assert evidence["complete"] is True
    assert [item["warmed_blocks"] for item in evidence["records"]] == [2, 3]
    assert all(item["elapsed_ms"] >= 0 for item in evidence["records"])
    connection.close.assert_called_once()


def test_relation_prewarm_rejects_duplicate_relations() -> None:
    with pytest.raises(RuntimeError, match="duplicates"):
        benchmark.prewarm_relations(["public.hnsw", "public.hnsw"])


def test_d3_switched_predicate_cache_reactivation_is_warm() -> None:
    evidence = benchmark.d3_phase_evidence(
        {
            "active": True,
            "adaptive_state": "page",
            "adaptive_admissions": 2,
            "fragment_cache_hits": 0,
        },
        {
            "active": True,
            "adaptive_state": "page",
            "adaptive_admissions": 2,
            "fragment_cache_hits": 1,
        },
        {"resident_entries": 2, "resident_bytes": 4096, "composed_guide_hits": 0},
        {"resident_entries": 2, "resident_bytes": 4096, "composed_guide_hits": 0},
        {
            "guidance_enabled": True,
            "guidance_route": "d3_adaptive",
            "activation_atom_count": 1,
            "fragment_cache_hits": 1,
            "fragment_store_hits": 0,
            "fast_reactivation_hits": 1,
        },
        same_predicate_before=False,
    )

    assert evidence["d3_phase"] == "warm"
    assert evidence["d3_active_guidance_reused"] is True
    assert evidence["d3_adaptive_admissions_delta"] == 0


def test_d3_sticky_admission_rejection_is_bypass() -> None:
    evidence = benchmark.d3_phase_evidence(
        {
            "active": False,
            "adaptive_state": "probing",
            "adaptive_admissions": 0,
            "adaptive_rejections": 0,
        },
        {
            "active": False,
            "adaptive_state": "rejected",
            "adaptive_admissions": 0,
            "adaptive_rejections": 1,
        },
        {"resident_entries": 0, "resident_bytes": 0, "composed_guide_hits": 0},
        {"resident_entries": 2, "resident_bytes": 43_735_492, "composed_guide_hits": 0},
        {
            "guidance_enabled": False,
            "guidance_route": "d3_admission_bypass",
            "activation_atom_count": 0,
            "fragment_cache_hits": 0,
            "fragment_store_hits": 0,
            "fast_reactivation_hits": 0,
        },
        same_predicate_before=False,
    )

    assert evidence["d3_phase"] == "bypass"
    assert evidence["d3_adaptive_rejections_delta"] == 1
    assert evidence["d3_active_guidance_reused"] is False
