from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import pgvector_d2_cache_isolation_control as control


SOURCE = "public.source_idx"
BFS = "public.bfs_idx"
TABLE = "public.items"
SHA = "a" * 64
BUILD = "sqlens-v15-d3-preflight-namespaced-store-test"
GRAPH = "b" * 64


def write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def args(tmp_path: Path) -> argparse.Namespace:
    filters = tmp_path / "filters.csv"
    truth = tmp_path / "truth.csv"
    truth_manifest = tmp_path / "truth.csv.manifest.json"
    matched_manifest = tmp_path / "matched.json"
    proof = tmp_path / "proof.json"
    filters.write_text("filter_name,target_rate,predicate,atoms\nf,50%,x=1,x=1\n")
    truth.write_text("query_no,id,filter_name\n100,1,f\n")
    truth_manifest.write_text("{}\n")
    matched_manifest.write_text("{}\n")
    proof.write_text("{}\n")
    return argparse.Namespace(
        out=tmp_path / "control.csv",
        filters_csv=filters,
        truth_csv=truth,
        truth_manifest=truth_manifest,
        matched_recall_manifest=matched_manifest,
        matched_configs_csv=None,
        matched_configs_manifest=None,
        filter_names=["f"],
        matched_mode="design1_bloom",
        matched_target_recall=0.90,
        matched_config_index_policy="exact",
        allow_mean_qualified_matched_config=False,
        d2_graph_proof_json=proof,
        live_graph_proof_policy="full",
        table=TABLE,
        source_index=SOURCE,
        bfs_index=BFS,
        query_table=None,
        query_id_column="id",
        query_vector_column="embedding",
        candidate_validity_predicate="embedding_valid",
        expected_truth_self_excluded=True,
        container="pg",
        backend_cpu=7,
        query_offset=100,
        queries=100,
        repeats=5,
        cache_regime="warm_resident",
        cold_block_queries=1,
        prewarm_index_blocks=123,
        prewarm_common_relation=[],
        k=10,
        d1_cache_mb=1024,
        d1_guidance_kind="auto",
        guidance_max_atoms=64,
        statement_timeout_ms=300_000,
        schedule_seed=20260721,
        readiness_timeout_s=60.0,
        python=Path("python"),
        expected_sqlens_build_id=BUILD,
        expected_vector_so_sha256=SHA,
        expected_candidate_rows=9_979_556,
        i_understand_container_restarts=False,
        i_understand_relation_cache_eviction=False,
        dry_run=True,
    )


def config(name: str = "f") -> control.MatchedConfig:
    return control.MatchedConfig(name, 0.90, 750, 5_000_000, 32.0, "strict_order", 1)


def row(
    arm: control.Arm,
    query_no: int,
    *,
    ids: str = "1,2,3",
    latency: float = 10.0,
    filter_name: str = "f",
) -> dict[str, object]:
    return {
        "selectivity": "50%",
        "filter_name": filter_name,
        "mode": arm.mode,
        "index": arm.expected_index,
        "backend_pid": "91",
        "backend_cpu_requested": "7",
        "backend_cpu_observed": "7",
        "backend_cpu_exact_match": "True",
        "query_no": str(query_no),
        "repeat": "0",
        "ef_search": "750",
        "max_scan_tuples": "5000000",
        "scan_mem_multiplier": "32.0",
        "iterative_scan": "strict_order",
        "guided_collect_target": "1",
        "guidance_filter_strategy": "safe_guided",
        "recall": "0.9",
        "end_to_end_ms": str(latency),
        "activation_ms": "1.0",
        "query_latency_ms": str(latency - 1.0),
        "vector_search_ms": "7.0",
        "ids": ids,
        "sqlens_build_id": BUILD,
        "vector_so_sha256": SHA,
        "index_page_loads": "20",
        "index_page_runs": "18",
        "index_page_distinct_pages": "17",
        "index_page_distinct_pages_exact": "True",
        "index_page_profile_scope": "scan_local_all_hnsw_reads",
        "profile_semantics_version": "12",
        "index_readbuffer_calls": "20",
        "index_readbuffer_ms": "2.0",
        "index_readbuffer_shared_read_calls": "5",
        "index_readbuffer_shared_read_ms": "1.5",
        "index_readbuffer_shared_hit_calls": "15",
        "index_readbuffer_shared_hit_ms": "0.5",
        "index_readbuffer_unclassified_calls": "0",
        "index_readbuffer_unclassified_ms": "0.0",
        "index_readbuffer_timing_scope": "all_profiled_hnsw_readbuffer_calls",
        "index_readbuffer_classification_scope": "per_call_pg_buffer_usage_delta",
        "distance_compute_timed_calls": "22",
        "distance_compute_ms": "3.0",
        "distance_compute_timing_scope": "all_profiled_hnsw_distance_calls",
        "hnsw_am_callback_ms": "7.0",
        "hnsw_remaining_ms": "2.0",
        "hnsw_remaining_ms_is_residual": "True",
        "hnsw_remaining_scope": "callback_minus_readbuffer_minus_distance",
        "profile_timer_overhead_scope": "included_in_timed_components",
        "index_page_transition_count": "19",
        "index_page_same_block_transitions": "5",
        "index_page_within_1_page_transitions": "7",
        "index_page_within_4_pages_transitions": "12",
        "index_page_within_16_pages_transitions": "18",
        "index_page_backward_transitions": "8",
        "index_page_total_abs_block_delta": "100",
        "index_page_max_abs_block_delta": "20",
        "index_page_trace_statistics_scope": "all_actual_search_readbuffer_transitions_excluding_cross_scan_boundaries",
        "index_page_trace_sample_limit": "64",
        "index_page_trace_sample_count": "20",
        "index_page_trace_sample_truncated": "False",
        "index_page_trace_sample_scope": "concatenated_prefix_of_actual_search_readbuffer_blocks",
        "index_page_trace_sample": json.dumps(list(range(20))),
        "index_page_prefetches": "3",
        "page_access_prefetches": "4",
        "idx_blks_hit": "15",
        "idx_blks_read": "5",
        "heap_blks_hit": "13",
        "heap_blks_read": "2",
        "heap_blks_are_exact_heap_io": "False",
        "heap_tid_page_runs": "10",
        "preferred_index_current_setting": arm.expected_index,
        "guidance_enabled": "True",
        "guidance_scan_verified": "True",
        "guidance_binding_verified": "True",
        "guidance_route": "safe_guided_candidate_validation",
        "guidance_kind": "bloom",
        "activation_atom_count": "1",
        "final_path": "validation_only",
        "visited_tuples": "20",
        "returned_tuples": "10",
        "distance_compute_count": "22",
        "guidance_checks": "20",
        "guidance_matches": "10",
        "traversal_guidance_matches": "10",
        "guidance_skips": "10",
        "traversal_expanded_nodes": "8",
        "traversal_neighbors_examined": "20",
        "error": "",
        "error_detail": "",
    }


def identity(index: str) -> dict[str, object]:
    return {
        "oid": 1 if index == SOURCE else 2,
        "relfilenode": 11 if index == SOURCE else 12,
        "heap_oid": 99,
        "size_bytes": 8192 * 200,
        "blocks": 200,
    }


def graph_proof() -> dict[str, object]:
    return {
        "stable_fingerprint_sha256": GRAPH,
        "relations": {
            "source": {"name": SOURCE, "oid": 1, "relfilenode": 11, "heap_oid": 99},
            "clone": {"name": BFS, "oid": 2, "relfilenode": 12, "heap_oid": 99},
        },
        "comparison": {"logical_equal": True, "physical_equal": False},
    }


def resume_schedule(count: int = 2) -> list[dict[str, object]]:
    return [
        {
            "sequence": sequence,
            "control_repeat": 0,
            "position": sequence + 1,
            "arm": "d1_source" if sequence % 2 == 0 else "d1_bfs",
            "filter_name": "f",
        }
        for sequence in range(count)
    ]


def write_resume_manifest(
    tmp_path: Path,
    value: argparse.Namespace,
    schedule: list[dict[str, object]],
    records: list[dict[str, object]],
    *,
    argv: list[str] | None = None,
) -> tuple[Path, list[str]]:
    manifest_path = value.out.with_suffix(value.out.suffix + ".manifest.json")
    current_argv = argv or ["controller.py", "--out", str(value.out), "--resume"]
    recorded_argv = [item for item in current_argv if item != "--resume"]
    exact_truth = {"artifact_valid": True, "truth": str(value.truth_csv.resolve())}
    matched_source = {"kind": "test", "configs": {"f": config().as_dict()}}
    manifest_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "protocol": control.protocol_spec(value),
                "argv": recorded_argv,
                "controller_sha256": control.sha256_file(Path(control.__file__)),
                "runner_sha256": control.sha256_file(control.RUNNER_PATH),
                "exact_truth_audit": exact_truth,
                "matched_config_source": matched_source,
                "runtime_input_identity": control.resume_runtime_input_identity(value),
                "schedule": schedule,
                "invocations": records,
            }
        )
    )
    return manifest_path, current_argv


def completed_resume_record(
    value: argparse.Namespace,
    schedule_item: dict[str, object],
) -> dict[str, object]:
    child = control.child_output_path(
        value.out,
        int(schedule_item["control_repeat"]),
        str(schedule_item["arm"]),
        str(schedule_item["filter_name"]),
    )
    child_sha = write_csv(child, [{"query_no": 100}])
    plan = child.with_suffix(child.suffix + ".plan.json")
    plan.write_text(json.dumps({"status": "complete", "checks": [{"passed": True}]}))
    return {
        **schedule_item,
        "status": "complete",
        "artifact": {"path": str(child), "rows": 1, "sha256": child_sha},
        "plan_evidence": {
            "path": str(plan),
            "sha256": control.sha256_file(plan),
        },
    }


def validate_resume_fixture(
    value: argparse.Namespace,
    manifest_path: Path,
    schedule: list[dict[str, object]],
    current_argv: list[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    exact_truth = {"artifact_valid": True, "truth": str(value.truth_csv.resolve())}
    matched_source = {"kind": "test", "configs": {"f": config().as_dict()}}
    return control.validate_resume_manifest(
        value,
        manifest_path,
        schedule,
        control.protocol_spec(value),
        exact_truth,
        matched_source,
        current_argv=current_argv,
    )


def test_default_truth_manifest_matches_formal_artifact_naming() -> None:
    assert control.DEFAULT_TRUTH_MANIFEST == control.DEFAULT_TRUTH.with_name(
        control.DEFAULT_TRUTH.stem + "_manifest.json"
    )
    assert control.DEFAULT_BFS_INDEX == (
        "public.amazon10m_hnsw_m32ef200_dupbridge_r29_bfs_idx"
    )


def test_two_arm_code_path_has_one_source_and_one_bfs_consumer(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRunner:
        @staticmethod
        def mode_table_index(value: argparse.Namespace, mode: str) -> tuple[str, str]:
            return (TABLE, SOURCE) if mode == "design1_bloom" else (TABLE, BFS)

    monkeypatch.setattr(control, "load_runner", lambda: FakeRunner)
    evidence = control.verify_two_arm_code_path(SOURCE, BFS)
    assert evidence["mode_to_index"] == {"design1_bloom": SOURCE, "design1_bloom_bfs_layout": BFS}
    assert evidence["physical_index_consumer_counts"] == {SOURCE: 1, BFS: 1}
    assert evidence["cache_confound_present"] is False


def test_resume_reuses_only_complete_prefix_and_leaves_next_invocation_unstarted(
    tmp_path: Path,
) -> None:
    value = args(tmp_path)
    schedule = resume_schedule()
    records = [completed_resume_record(value, schedule[0])]
    manifest_path, current_argv = write_resume_manifest(tmp_path, value, schedule, records)

    _manifest, reused = validate_resume_fixture(value, manifest_path, schedule, current_argv)

    assert len(reused) == 1
    assert reused[0]["schedule"] == schedule[0]
    assert not control.child_output_path(value.out, 0, "d1_bfs", "f").exists()


def test_resume_rejects_argv_drift(tmp_path: Path) -> None:
    value = args(tmp_path)
    schedule = resume_schedule(1)
    records = [completed_resume_record(value, schedule[0])]
    manifest_path, current_argv = write_resume_manifest(tmp_path, value, schedule, records)
    drifted = [*current_argv, "--queries", "2"]

    with pytest.raises(control.ControlError, match="argv differs"):
        validate_resume_fixture(value, manifest_path, schedule, drifted)


def test_resume_rejects_completed_child_hash_drift(tmp_path: Path) -> None:
    value = args(tmp_path)
    schedule = resume_schedule(1)
    records = [completed_resume_record(value, schedule[0])]
    manifest_path, current_argv = write_resume_manifest(tmp_path, value, schedule, records)
    child = control.child_output_path(value.out, 0, "d1_source", "f")
    child.write_text(child.read_text(encoding="utf-8") + "\n")

    with pytest.raises(control.ControlError, match="SHA256 mismatch"):
        validate_resume_fixture(value, manifest_path, schedule, current_argv)


def test_resume_rejects_incomplete_child_for_next_invocation(tmp_path: Path) -> None:
    value = args(tmp_path)
    schedule = resume_schedule()
    records = [completed_resume_record(value, schedule[0])]
    manifest_path, current_argv = write_resume_manifest(tmp_path, value, schedule, records)
    pending = control.child_output_path(value.out, 0, "d1_bfs", "f")
    pending.write_text("query_no\n", encoding="utf-8")

    with pytest.raises(control.ControlError, match="existing or incomplete"):
        validate_resume_fixture(value, manifest_path, schedule, current_argv)


def test_resume_rejects_running_child_instead_of_overwriting(tmp_path: Path) -> None:
    value = args(tmp_path)
    schedule = resume_schedule(1)
    running = {**schedule[0], "status": "running"}
    manifest_path, current_argv = write_resume_manifest(tmp_path, value, schedule, [running])

    with pytest.raises(control.ControlError, match="current running invocation"):
        validate_resume_fixture(value, manifest_path, schedule, current_argv)


def test_schedule_is_two_arm_seeded_paired_q100_r5() -> None:
    schedule = control.rotating_schedule(5, 19)
    assert len(schedule) == 10
    for block in range(5):
        rows = [item for item in schedule if item["control_repeat"] == block]
        assert {item["arm"] for item in rows} == {"d1_source", "d1_bfs"}
        assert {item["position"] for item in rows} == {1, 2}


def test_expanded_schedule_places_each_filter_pair_adjacent() -> None:
    schedule = control.paired_filter_schedule(["f1", "f2", "f3"], 5, 19)
    assert len(schedule) == 30
    for offset in range(0, len(schedule), 2):
        pair = schedule[offset : offset + 2]
        assert pair[0]["filter_name"] == pair[1]["filter_name"]
        assert pair[0]["control_repeat"] == pair[1]["control_repeat"]
        assert {item["arm"] for item in pair} == {"d1_source", "d1_bfs"}


def test_runner_command_uses_per_filter_config_and_safe_guided_only(tmp_path: Path) -> None:
    value = args(tmp_path)
    source, bfs = control.arm_specs(SOURCE, BFS)
    command = control.build_runner_command(value, source, tmp_path / "child.csv", tmp_path / "proof.json", config())
    assert command[command.index("--queries") + 1] == "100"
    assert command[command.index("--query-offset") + 1] == "100"
    assert command[command.index("--repeats") + 1] == "1"
    assert command[command.index("--modes") + 1] == "design1_bloom"
    assert "--guidance-filter-strategy" in command
    assert command[command.index("--guidance-filter-strategy") + 1] == "safe_guided"
    config_json = json.loads(command[command.index("--mode-configs-json") + 1])
    assert config_json["design1_bloom"]["traversal_guided_prioritization"] is False
    assert command[command.index("--ef-search") + 1] == "750"
    assert command[command.index("--filter-names") + 1] == "f"
    assert command[command.index("--guidance-max-atoms") + 1] == "64"
    assert "--expected-truth-self-excluded" in command
    assert bfs.index_role == "bfs"
    assert command == control.build_runner_command(
        value,
        source,
        tmp_path / "child.csv",
        tmp_path / "proof.json",
        config(),
        4,
    )


def test_child_row_gate_requires_exact_cache_metrics_and_config(tmp_path: Path) -> None:
    value = args(tmp_path)
    arm = control.arm_specs(SOURCE, BFS)[0]
    rows = [row(arm, query_no) for query_no in range(100, 200)]
    enriched = control.validate_child_rows(rows, value, arm, 2, 7, 1, identity(SOURCE), config(), GRAPH)
    assert len(enriched) == 100
    assert enriched[0]["control_graph_semantic_fingerprint"] == GRAPH
    rows[0]["heap_blks_read"] = ""
    with pytest.raises(control.ControlError, match="heap_blks_read"):
        control.validate_child_rows(rows, value, arm, 2, 7, 1, identity(SOURCE), config(), GRAPH)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"profile_semantics_version": "11"}, "profile semantics >=12"),
        ({"index_readbuffer_ms": "-0.1"}, "index_readbuffer_ms"),
        ({"index_readbuffer_shared_hit_calls": "14"}, "call classification"),
        ({"index_readbuffer_shared_hit_ms": "0.4"}, "timing classification"),
        ({"distance_compute_timed_calls": "21"}, "timed distance calls"),
        ({"index_readbuffer_calls": "19", "index_readbuffer_shared_hit_calls": "14"}, "index_page_loads"),
        ({"hnsw_remaining_ms": "1.5"}, "callback breakdown"),
        ({"hnsw_remaining_ms_is_residual": "False"}, "as a residual"),
        ({"distance_compute_timing_scope": ""}, "distance_compute_timing_scope"),
    ],
)
def test_child_row_gate_rejects_invalid_r33_profile_breakdown(
    tmp_path: Path, changes: dict[str, object], error: str
) -> None:
    value = args(tmp_path)
    arm = control.arm_specs(SOURCE, BFS)[0]
    rows = [row(arm, query_no) for query_no in range(100, 200)]
    rows[0].update(changes)

    with pytest.raises(control.ControlError, match=error):
        control.validate_child_rows(
            rows, value, arm, 0, 0, 1, identity(SOURCE), config(), GRAPH
        )


def test_child_row_gate_preserves_censored_distinct_count_with_exact_trace(
    tmp_path: Path,
) -> None:
    value = args(tmp_path)
    arm = control.arm_specs(SOURCE, BFS)[0]
    rows = [row(arm, query_no) for query_no in range(100, 200)]
    rows[15]["index_page_distinct_pages"] = "-1"
    rows[15]["index_page_distinct_pages_exact"] = "False"

    enriched = control.validate_child_rows(
        rows, value, arm, 2, 7, 1, identity(SOURCE), config(), GRAPH
    )

    assert enriched[15]["index_page_distinct_pages"] == "-1"
    assert enriched[15]["index_page_transition_count"] == "19"


def test_child_row_gate_rejects_non_safe_guided_or_config_drift(tmp_path: Path) -> None:
    value = args(tmp_path)
    arm = control.arm_specs(SOURCE, BFS)[1]
    rows = [row(arm, query_no) for query_no in range(100, 200)]
    rows[0]["guidance_filter_strategy"] = "traversal_guided"
    with pytest.raises(control.ControlError, match="safe_guided"):
        control.validate_child_rows(rows, value, arm, 0, 0, 2, identity(BFS), config(), GRAPH)
    rows[0]["guidance_filter_strategy"] = "safe_guided"
    rows[0]["ef_search"] = "1000"
    with pytest.raises(control.ControlError, match="ef_search"):
        control.validate_child_rows(rows, value, arm, 0, 0, 2, identity(BFS), config(), GRAPH)


def test_pair_gate_allows_physical_counters_to_differ_but_not_semantics(tmp_path: Path) -> None:
    value = args(tmp_path)
    arms = control.arm_specs(SOURCE, BFS)
    rows: list[dict[str, object]] = []
    for repeat in range(5):
        for position, arm in enumerate(arms, 1):
            child = [row(arm, query_no, latency=12.0 if arm.index_role == "source" else 10.0) for query_no in range(100, 200)]
            if arm.index_role == "bfs":
                child[0]["idx_blks_read"] = "1"
                child[0]["heap_blks_read"] = "7"
                child[0]["index_page_prefetches"] = "9"
                child[0]["index_readbuffer_shared_read_calls"] = "1"
                child[0]["index_readbuffer_shared_read_ms"] = "0.2"
                child[0]["index_readbuffer_shared_hit_calls"] = "19"
                child[0]["index_readbuffer_shared_hit_ms"] = "0.8"
                child[0]["index_readbuffer_ms"] = "1.0"
                child[0]["hnsw_remaining_ms"] = "3.0"
            rows.extend(control.validate_child_rows(child, value, arm, repeat, repeat * 2 + position, position, identity(arm.expected_index), config(), GRAPH))
    control.validate_paired_rows(rows)
    summary = control.summarize(rows, 11)
    assert summary[0]["d1_bfs_speedup_over_source"] == pytest.approx(1.2)
    assert summary[0]["d1_bfs_heap_blks_read_mean"] > summary[0]["d1_source_heap_blks_read_mean"]
    assert summary[0]["d1_bfs_index_page_run_reduction"] == pytest.approx(0.0)
    assert summary[0]["d1_source_index_readbuffer_calls_mean"] == pytest.approx(20.0)
    assert summary[0]["d1_bfs_index_readbuffer_calls_mean"] == pytest.approx(20.0)
    assert summary[0]["d1_bfs_index_readbuffer_total_time_reduction_ms"] > 0
    assert summary[0]["d1_bfs_index_readbuffer_shared_read_time_reduction_ms"] > 0
    assert "d1_bfs_index_readbuffer_shared_hit_time_reduction_ms" in summary[0]
    for arm_name in control.ARMS:
        for field in (
            *control.R33_PROFILE_COUNT_FIELDS,
            *control.R33_PROFILE_TIME_FIELDS,
        ):
            assert f"{arm_name}_{field}_mean" in summary[0]
    assert summary[0]["d1_bfs_distance_compute_timed_calls_delta"] == pytest.approx(0.0)
    assert summary[0]["d1_bfs_distance_compute_ms_delta"] == pytest.approx(0.0)
    assert summary[0]["d1_bfs_hnsw_remaining_ms_delta"] > 0
    assert (
        summary[0]["d1_source_index_readbuffer_timing_scope"]
        == "all_profiled_hnsw_readbuffer_calls"
    )


def test_summary_reports_page_run_reduction_as_one_minus_ratio(tmp_path: Path) -> None:
    value = args(tmp_path)
    source, bfs = control.arm_specs(SOURCE, BFS)
    rows: list[dict[str, object]] = []
    for repeat in range(5):
        for position, arm in enumerate((source, bfs), 1):
            child = [row(arm, query_no) for query_no in range(100, 200)]
            if arm.index_role == "bfs":
                for item in child:
                    item["index_page_runs"] = "9"
            rows.extend(
                control.validate_child_rows(
                    child,
                    value,
                    arm,
                    repeat,
                    repeat * 2 + position,
                    position,
                    identity(arm.expected_index),
                    config(),
                    GRAPH,
                )
            )

    summary = control.summarize(rows, 11)
    assert summary[0]["d1_source_index_page_runs_mean"] == pytest.approx(18.0)
    assert summary[0]["d1_bfs_index_page_runs_mean"] == pytest.approx(9.0)
    assert summary[0]["d1_bfs_index_page_run_reduction"] == pytest.approx(0.5)


def test_cold_summary_pairs_independent_eviction_repeats(tmp_path: Path) -> None:
    value = args(tmp_path)
    value.cache_regime = "cold_io"
    value.prewarm_index_blocks = None
    source, bfs = control.arm_specs(SOURCE, BFS)
    rows: list[dict[str, object]] = []
    for repeat in range(5):
        for position, arm in enumerate((source, bfs), 1):
            latency = 100.0 + repeat * 10.0 if arm.index_role == "source" else 20.0
            rows.extend(
                control.validate_child_rows(
                    [row(arm, 100 + repeat, latency=latency)],
                    value,
                    arm,
                    repeat,
                    repeat * 2 + position,
                    position,
                    identity(arm.expected_index),
                    config(),
                    GRAPH,
                )
            )

    control.validate_paired_rows(rows, queries=1, repeats=5)
    summary = control.summarize(rows, 11, queries=1, repeats=5)[0]
    assert summary["paired_cluster_unit"] == "cold_eviction_block"
    assert summary["paired_clusters"] == 5
    assert summary["d1_bfs_minus_source_ci95_low_ms"] < summary[
        "d1_bfs_minus_source_ci95_high_ms"
    ]


def test_pair_gate_rejects_semantic_drift(tmp_path: Path) -> None:
    value = args(tmp_path)
    arms = control.arm_specs(SOURCE, BFS)
    rows: list[dict[str, object]] = []
    for repeat in range(5):
        for position, arm in enumerate(arms, 1):
            rows.extend(control.validate_child_rows([row(arm, query_no) for query_no in range(100, 200)], value, arm, repeat, repeat * 2 + position, position, identity(arm.expected_index), config(), GRAPH))
    changed = next(item for item in rows if item["control_arm"] == "d1_bfs")
    changed["visited_tuples"] = "21"
    with pytest.raises(control.ControlError, match="execution semantics differ"):
        control.validate_paired_rows(rows)


def test_pair_gate_rejects_r33_logical_work_drift(tmp_path: Path) -> None:
    value = args(tmp_path)
    arms = control.arm_specs(SOURCE, BFS)
    rows: list[dict[str, object]] = []
    for repeat in range(5):
        for position, arm in enumerate(arms, 1):
            rows.extend(
                control.validate_child_rows(
                    [row(arm, query_no) for query_no in range(100, 200)],
                    value,
                    arm,
                    repeat,
                    repeat * 2 + position,
                    position,
                    identity(arm.expected_index),
                    config(),
                    GRAPH,
                )
            )
    changed = next(item for item in rows if item["control_arm"] == "d1_bfs")
    changed["index_readbuffer_calls"] = "21"
    changed["index_readbuffer_shared_hit_calls"] = "16"
    changed["index_page_loads"] = "21"

    with pytest.raises(control.ControlError, match="execution semantics differ"):
        control.validate_paired_rows(rows)


def test_pair_gate_uses_exported_traversal_guidance_match_counter(
    tmp_path: Path,
) -> None:
    value = args(tmp_path)
    arms = control.arm_specs(SOURCE, BFS)
    rows: list[dict[str, object]] = []
    for repeat in range(5):
        for position, arm in enumerate(arms, 1):
            child = [row(arm, query_no) for query_no in range(100, 200)]
            for item in child:
                item.pop("guidance_matches", None)
                item["traversal_guidance_matches"] = "10"
            rows.extend(
                control.validate_child_rows(
                    child,
                    value,
                    arm,
                    repeat,
                    repeat * 2 + position,
                    position,
                    identity(arm.expected_index),
                    config(),
                    GRAPH,
                )
            )
    control.validate_paired_rows(rows)


def test_graph_identity_gate_binds_relfilenode_and_semantic_fingerprint() -> None:
    proof = graph_proof()
    identities = {SOURCE: identity(SOURCE), BFS: identity(BFS)}
    control.validate_graph_relation_identities(proof, identities, SOURCE, BFS)
    proof["relations"]["clone"]["relfilenode"] = 13
    with pytest.raises(control.ControlError, match="relfilenode"):
        control.validate_graph_relation_identities(proof, identities, SOURCE, BFS)


def test_live_graph_proof_rejects_fingerprint_drift() -> None:
    delegated = {"stable_fingerprint_sha256": GRAPH}
    assert control.validate_live_graph_proof(delegated, {"stable_fingerprint_sha256": GRAPH})["stable_fingerprint_sha256"] == GRAPH
    with pytest.raises(control.ControlError, match="graph proof drifted"):
        control.validate_live_graph_proof(delegated, {"stable_fingerprint_sha256": "c" * 64})


def test_plan_evidence_gate_requires_exact_index_and_filter(tmp_path: Path) -> None:
    arm = control.arm_specs(SOURCE, BFS)[0]
    path = tmp_path / "child.csv.plan.json"
    path.write_text(json.dumps({
        "status": "complete",
        "checks": [{
            "passed": True,
            "filter_name": "f",
            "mode": arm.mode,
            "expected_table_identity": TABLE,
            "expected_index_identity": SOURCE,
            "expected_index_oid": 1,
            "catalog_index_predicate_matches": True,
            "preferred_index_current_setting": SOURCE,
        }],
    }))
    evidence = control.validate_plan_evidence(path, arm, "f", identity(SOURCE), TABLE)
    assert evidence["checks"] == 1
    payload = json.loads(path.read_text())
    payload["checks"][0]["expected_index_oid"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(control.ControlError, match="OID"):
        control.validate_plan_evidence(path, arm, "f", identity(SOURCE), TABLE)


def test_dry_run_is_two_arm_per_filter_and_does_not_touch_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = args(tmp_path)
    matched = config()
    monkeypatch.setattr(control, "load_runner", lambda: object())
    monkeypatch.setattr(control, "load_graph_proof", lambda *_args: {"stable_fingerprint_sha256": GRAPH, "source_index": SOURCE, "clone_index": BFS, "comparison": {"same_heap": True, "logical_equal": True, "physical_equal": False}})
    monkeypatch.setattr(control, "audit_exact_truth_manifest", lambda *_args, **_kwargs: {"artifact_valid": True})
    monkeypatch.setattr(control, "audit_matched_recall_manifest", lambda *_args: {"f": matched})
    monkeypatch.setattr(control, "verify_two_arm_code_path", lambda *_args: {"cache_confound_present": False})
    monkeypatch.setattr(control, "run_command", lambda _command: pytest.fail("dry-run executed a command"))
    payload = control.dry_run_payload(value)
    assert len(payload["invocations"]) == 10
    assert {item["arm"] for item in payload["invocations"]} == {"d1_source", "d1_bfs"}
    assert all(item["filter_name"] == "f" for item in payload["invocations"])
    assert payload["protocol"]["measurement"]["queries"] == 100
    assert payload["protocol"]["measurement"]["repeats"] == 5


def test_default_cache_protocol_prewarms_full_target_relation(tmp_path: Path) -> None:
    value = args(tmp_path)
    value.prewarm_index_blocks = None
    plan = control.docker_command_plan(value, control.arm_specs(SOURCE, BFS)[1], "FULL_TARGET_INDEX_BLOCKS")
    assert plan[-1] == f"SELECT pg_prewarm('{BFS}'::regclass, 'read', 'main');"
    protocol = control.protocol_spec(value)
    assert protocol["name"] == control.PROTOCOL_NAME
    assert protocol["version"] == control.PROTOCOL_VERSION
    assert protocol["cache"]["prewarm_index_blocks"] == "full_target_index"
    assert protocol["profile_contract"]["required_profile_semantics_min"] == 12
    assert set(control.R33_PROFILE_COUNT_FIELDS).issubset(
        protocol["required_per_query_metrics"]
    )
    assert set(control.R33_PROFILE_TIME_FIELDS).issubset(
        protocol["required_per_query_metrics"]
    )
    assert set(control.R33_PROFILE_SCOPE_FIELDS).issubset(
        protocol["required_per_query_metrics"]
    )


def test_cold_protocol_evicts_only_target_relation_and_never_warms_it(tmp_path: Path) -> None:
    value = args(tmp_path)
    value.cache_regime = "cold_io"
    value.prewarm_index_blocks = None
    arm = control.arm_specs(SOURCE, BFS)[1]
    plan = control.docker_command_plan(value, arm, "FULL_TARGET_INDEX_BLOCKS")
    assert any("relation-scoped-posix-fadvise-DONTNEED" in step for step in plan)
    assert not any("drop_caches" in step for step in plan)
    assert not any(BFS in step and "pg_prewarm" in step for step in plan)
    command = control.build_runner_command(
        value, arm, tmp_path / "child.csv", tmp_path / "proof.json", config()
    )
    assert "--warmup-all-queries" not in command
    assert command[command.index("--warmup-queries") + 1] == "0"
    assert command[command.index("--queries") + 1] == "1"
    assert command[command.index("--query-offset") + 1] == "100"
    source = control.arm_specs(SOURCE, BFS)[0]
    source_repeat_3 = control.build_runner_command(
        value,
        source,
        tmp_path / "source-r3.csv",
        tmp_path / "proof.json",
        config(),
        3,
    )
    bfs_repeat_3 = control.build_runner_command(
        value,
        arm,
        tmp_path / "bfs-r3.csv",
        tmp_path / "proof.json",
        config(),
        3,
    )
    assert source_repeat_3[source_repeat_3.index("--query-offset") + 1] == "103"
    assert bfs_repeat_3[bfs_repeat_3.index("--query-offset") + 1] == "103"
    protocol = control.protocol_spec(value)
    assert protocol["name"] == control.PROTOCOL_NAME
    assert protocol["version"] == 5
    assert protocol["measurement"]["cold_protocol_semantics_version"] == 4
    assert protocol["measurement"]["queries"] == 1
    assert protocol["measurement"]["queries_per_block"] == 1
    assert protocol["measurement"]["total_distinct_queries"] == 5
    assert protocol["measurement"]["query_split"] == "q100..q104"
    assert protocol["measurement"]["query_slice_policy"] == (
        control.DISTINCT_COLD_QUERY_SLICE_POLICY
    )
    assert [item["query_offset"] for item in protocol["measurement"]["block_query_slices"]] == [
        100,
        101,
        102,
        103,
        104,
    ]
    assert protocol["cache"]["os_page_cache_dropped"] is False
    assert protocol["cache"]["target_relation_os_pages_evicted"] is True
    assert protocol["cache"]["target_index_prewarm"] == "none"
    assert protocol["cache"]["cold_block_interpretation"] == (
        "each q1/r1 block is an independent cold-start measurement after "
        "target-relation eviction and uses a distinct contiguous query slice; "
        "both source/BFS arms use the same slice; this preserves cold v4 "
        "independent-block semantics"
    )


def test_cold_multi_query_blocks_use_contiguous_nonoverlapping_slices(
    tmp_path: Path,
) -> None:
    value = args(tmp_path)
    value.cache_regime = "cold_io"
    value.cold_block_queries = 2
    value.repeats = 3
    source, bfs = control.arm_specs(SOURCE, BFS)

    for repeat, expected_offset in enumerate((100, 102, 104)):
        for arm in (source, bfs):
            command = control.build_runner_command(
                value,
                arm,
                tmp_path / f"{arm.name}-{repeat}.csv",
                tmp_path / "proof.json",
                config(),
                repeat,
            )
            assert command[command.index("--query-offset") + 1] == str(expected_offset)
            child = [row(arm, query_no) for query_no in range(expected_offset, expected_offset + 2)]
            assert len(
                control.validate_child_rows(
                    child,
                    value,
                    arm,
                    repeat,
                    repeat * 2,
                    1,
                    identity(arm.expected_index),
                    config(),
                    GRAPH,
                )
            ) == 2

    with pytest.raises(control.ControlError, match="expected q102..q103"):
        control.validate_child_rows(
            [row(source, 100), row(source, 101)],
            value,
            source,
            1,
            2,
            1,
            identity(source.expected_index),
            config(),
            GRAPH,
        )

    protocol = control.protocol_spec(value)
    assert protocol["measurement"]["query_split"] == "q100..q105"
    assert protocol["measurement"]["total_distinct_queries"] == 6


def test_cold_truth_audit_covers_every_distinct_block_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = args(tmp_path)
    value.cache_regime = "cold_io"
    value.cold_block_queries = 2
    value.repeats = 3
    observed: dict[str, object] = {}

    def fake_audit(*_args: object, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"artifact_valid": True}

    monkeypatch.setattr(control, "audit_exact_truth_manifest", fake_audit)
    control.audit_truth_for_args(value)

    assert observed["query_offset"] == 100
    assert observed["queries"] == 6


def test_refresh_summary_preserves_raw_and_records_revision(tmp_path: Path) -> None:
    value = args(tmp_path)
    value.cache_regime = "cold_io"
    value.prewarm_index_blocks = None
    source, bfs = control.arm_specs(SOURCE, BFS)
    rows: list[dict[str, object]] = []
    for repeat in range(5):
        for position, arm in enumerate((source, bfs), 1):
            rows.extend(
                control.validate_child_rows(
                    [
                        row(
                            arm,
                            100 + repeat,
                            latency=30.0 if arm.index_role == "source" else 10.0,
                        )
                    ],
                    value,
                    arm,
                    repeat,
                    repeat * 2 + position,
                    position,
                    identity(arm.expected_index),
                    config(),
                    GRAPH,
                )
            )
    raw_sha = write_csv(value.out, rows)
    manifest_path = value.out.with_suffix(value.out.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_valid": True,
                "paired_gate_passed": True,
                "argv": [str(control.Path(control.__file__))],
                "protocol": control.protocol_spec(value),
                "outputs": {
                    "raw": {
                        "path": str(value.out),
                        "rows": len(rows),
                        "sha256": raw_sha,
                    },
                    "summary": {"path": "old.csv", "rows": 1, "sha256": "old"},
                },
            }
        ),
        encoding="utf-8",
    )

    control.refresh_completed_summary(value)

    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert refreshed["outputs"]["raw"]["sha256"] == raw_sha
    assert refreshed["summary_revisions"][-1]["measurement_commands_rerun"] == 0
    assert refreshed["summary_revisions"][-1]["raw_sha256_unchanged"] == raw_sha
    assert refreshed["outputs"]["summary"]["sha256"] != "old"
    assert refreshed["protocol"]["measurement"]["schedule_seed"] == value.schedule_seed


def test_refresh_v5_rejects_raw_rows_missing_r33_fields(tmp_path: Path) -> None:
    value = args(tmp_path)
    value.cache_regime = "cold_io"
    value.prewarm_index_blocks = None
    source, bfs = control.arm_specs(SOURCE, BFS)
    rows: list[dict[str, object]] = []
    for repeat in range(5):
        for position, arm in enumerate((source, bfs), 1):
            enriched = control.validate_child_rows(
                [row(arm, 100 + repeat)],
                value,
                arm,
                repeat,
                repeat * 2 + position,
                position,
                identity(arm.expected_index),
                config(),
                GRAPH,
            )[0]
            enriched.pop("index_readbuffer_ms")
            rows.append(enriched)
    raw_sha = write_csv(value.out, rows)
    manifest_path = value.out.with_suffix(value.out.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_valid": True,
                "paired_gate_passed": True,
                "argv": [str(control.Path(control.__file__))],
                "protocol": control.protocol_spec(value),
                "outputs": {
                    "raw": {
                        "path": str(value.out),
                        "rows": len(rows),
                        "sha256": raw_sha,
                    },
                    "summary": {"path": "old.csv", "rows": 1, "sha256": "old"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(control.ControlError, match="index_readbuffer_ms"):
        control.refresh_completed_summary(value)


def test_refresh_legacy_cold_manifest_preserves_repeated_slice_semantics(
    tmp_path: Path,
) -> None:
    value = args(tmp_path)
    value.cache_regime = "cold_io"
    value.prewarm_index_blocks = None
    source, bfs = control.arm_specs(SOURCE, BFS)
    rows: list[dict[str, object]] = []
    for repeat in range(5):
        for position, arm in enumerate((source, bfs), 1):
            enriched = control.validate_child_rows(
                [row(arm, 100, latency=30.0 if arm.index_role == "source" else 10.0)],
                value,
                arm,
                0,
                repeat * 2 + position,
                position,
                identity(arm.expected_index),
                config(),
                GRAPH,
            )[0]
            enriched["control_repeat"] = repeat
            enriched["control_pair_key"] = f"f|100|{repeat}"
            enriched["profile_semantics_version"] = "11"
            for field in (
                *control.R33_PROFILE_COUNT_FIELDS,
                *control.R33_PROFILE_TIME_FIELDS,
                *control.R33_PROFILE_SCOPE_FIELDS,
                "hnsw_remaining_ms_is_residual",
            ):
                enriched.pop(field)
            rows.append(enriched)

    raw_sha = write_csv(value.out, rows)
    legacy_protocol = control.protocol_spec(value)
    legacy_protocol["name"] = "sqlens-d2-cache-isolation-v3"
    legacy_protocol.pop("version")
    legacy_protocol.pop("profile_contract")
    legacy_protocol["required_per_query_metrics"] = [
        field
        for field in legacy_protocol["required_per_query_metrics"]
        if field
        not in {
            *control.R33_PROFILE_COUNT_FIELDS,
            *control.R33_PROFILE_TIME_FIELDS,
            *control.R33_PROFILE_SCOPE_FIELDS,
            "hnsw_remaining_ms_is_residual",
        }
    ]
    legacy_protocol["measurement"]["query_split"] = "q100..q100"
    for field in (
        "queries_per_block",
        "total_distinct_queries",
        "query_slice_policy",
        "block_query_slices",
    ):
        legacy_protocol["measurement"].pop(field)
    legacy_interpretation = (
        "q1/r1 independent cold-start measurement after target-relation eviction"
    )
    legacy_protocol["cache"]["cold_block_interpretation"] = legacy_interpretation
    manifest_path = value.out.with_suffix(value.out.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_valid": True,
                "paired_gate_passed": True,
                "argv": [str(control.Path(control.__file__))],
                "protocol": legacy_protocol,
                "outputs": {
                    "raw": {
                        "path": str(value.out),
                        "rows": len(rows),
                        "sha256": raw_sha,
                    },
                    "summary": {"path": "old.csv", "rows": 1, "sha256": "old"},
                },
            }
        ),
        encoding="utf-8",
    )

    control.refresh_completed_summary(value)

    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert refreshed["protocol"]["name"] == "sqlens-d2-cache-isolation-v3"
    assert refreshed["protocol"]["cache"]["cold_block_interpretation"] == (
        legacy_interpretation
    )
    assert refreshed["outputs"]["raw"]["sha256"] == raw_sha
    assert refreshed["summary_revisions"][-1]["r33_profile_fields_present"] is False
    summary_rows = control.read_csv(
        value.out.with_name(value.out.stem + "_summary.csv")
    )
    assert "d1_source_index_readbuffer_ms_mean" not in summary_rows[0]


def test_cold_eviction_must_cover_every_target_index_byte() -> None:
    payload = {"files": [{"path": "idx", "bytes": 8192}], "file_count": 1, "bytes": 8192}
    audited = control.validate_eviction_coverage(payload, 8192)
    assert audited["coverage_ratio"] == 1.0
    assert audited["expected_bytes"] == 8192
    with pytest.raises(control.ControlError, match="complete target index"):
        control.validate_eviction_coverage(payload, 16384)


def test_legacy_raw_is_rejected() -> None:
    with pytest.raises(control.ControlError, match="legacy raw"):
        control.reject_legacy_raw(Path("old.csv"))


def test_source_contains_no_legacy_traversal_or_fixed_ef_assumption() -> None:
    source = control.Path(control.__file__).read_text(encoding="utf-8")
    assert "traversal_guided_target" not in source
    assert "traversal_guided_burst" not in source
    assert "stock_source" not in source
    assert "q400" not in source
    assert "--ef-search" in source
    assert "args.ef_search" not in source


def test_audited_truth_and_matched_config_bind_same_files_and_per_filter_config(tmp_path: Path) -> None:
    value = args(tmp_path)
    truth_rows = [
        {
            "query_no": query_no,
            "id": query_no,
            "filter_name": "f",
            "candidate_validity_predicate": "embedding_valid",
            "self_excluded": "True",
            "query_split": "final",
        }
        for query_no in range(100, 200)
    ]
    truth_sha = write_csv(value.truth_csv, truth_rows)
    filters_sha = hashlib.sha256(value.filters_csv.read_bytes()).hexdigest()
    value.truth_manifest.write_text(json.dumps({
        "artifact_valid": True,
        "recall_contract": "distance_squared_threshold_tie_aware_v1",
        "self_excluded": True,
        "inputs": {
            "filters_csv": {"sha256": filters_sha},
            "postgres": {
                "table": TABLE,
                "query_population": {"candidate_validity_predicate": "embedding_valid"},
            },
        },
        "outputs": {"truth_csv": {"sha256": truth_sha}},
    }))
    selected = tmp_path / "selected.csv"
    selected_sha = write_csv(selected, [{
        "selection_status": "selected",
        "target_recall": "0.9",
        "target_lcb95_met_in_calibration": "True",
        "filter_name": "f",
        "mode": "design1_bloom",
        "guidance_filter_strategy": "safe_guided",
        "ef_search": "750",
        "max_scan_tuples": "5000000",
        "scan_mem_multiplier": "32.0",
        "iterative_scan": "strict_order",
    }])
    value.matched_recall_manifest.write_text(json.dumps({
        "status": "complete",
        "comparison_valid": True,
        "self_excluded": True,
        "run_spec": {"args": {
            "guidance_filter_strategy": "safe_guided",
            "final_query_offset": 100,
            "final_queries": 100,
            "final_repeats": 5,
            "insertion_table": TABLE,
            "insertion_index": SOURCE,
            "bfs_table": TABLE,
            "bfs_index": BFS,
            "query_table": None,
            "query_id_column": "id",
            "query_vector_column": "embedding",
            "candidate_validity_predicate": "embedding_valid",
            "expected_truth_self_excluded": True,
            "truth_csv": str(value.truth_csv),
            "filters_csv": str(value.filters_csv),
        }},
        "outputs": {"selected": {"path": str(selected), "sha256": selected_sha}},
    }))
    exact = control.audit_exact_truth_manifest(
        value.truth_manifest,
        value.truth_csv,
        value.filters_csv,
        expected_table=TABLE,
        expected_candidate_validity_predicate="embedding_valid",
    )
    assert exact["artifact_valid"] is True
    configs = control.audit_matched_recall_manifest(
        value.matched_recall_manifest, value, value.filters_csv, value.truth_csv
    )
    assert configs["f"].ef_search == 750
    assert configs["f"].iterative_scan == "strict_order"


def test_external_launch_truth_and_matched_manifest_follow_cli_split_and_schema(
    tmp_path: Path,
) -> None:
    value = args(tmp_path)
    value.query_offset = 80
    value.queries = 3
    value.repeats = 2
    value.query_table = "public.external_queries"
    value.query_id_column = "qid"
    value.query_vector_column = "vector"
    value.candidate_validity_predicate = "TRUE"
    value.expected_truth_self_excluded = False
    value.guidance_max_atoms = 128

    truth_rows = [
        {
            "query_no": query_no,
            "query_id": 1000 + query_no,
            "filter_name": "f",
            "candidate_validity_predicate": "TRUE",
            "self_excluded": "False",
            "query_split": "final",
        }
        for query_no in range(80, 83)
    ]
    truth_sha = write_csv(value.truth_csv, truth_rows)
    filters_sha = hashlib.sha256(value.filters_csv.read_bytes()).hexdigest()
    selected = tmp_path / "selected.csv"
    selected_sha = write_csv(
        selected,
        [
            {
                "selection_status": "selected",
                "target_recall": "0.9",
                "target_lcb95_met_in_calibration": "True",
                "filter_name": "f",
                "mode": "design1_bloom",
                "guidance_filter_strategy": "safe_guided",
                "ef_search": "250",
                "max_scan_tuples": "5000000",
                "scan_mem_multiplier": "32.0",
                "iterative_scan": "strict_order",
            }
        ],
    )
    value.matched_recall_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "comparison_valid": True,
                "self_excluded": False,
                "run_spec": {
                    "args": {
                        "guidance_filter_strategy": "safe_guided",
                        "final_query_offset": 80,
                        "final_queries": 3,
                        "final_repeats": 2,
                        "insertion_table": TABLE,
                        "insertion_index": SOURCE,
                        "bfs_table": TABLE,
                        "bfs_index": BFS,
                        "query_table": value.query_table,
                        "query_id_column": "qid",
                        "query_vector_column": "vector",
                        "candidate_validity_predicate": "TRUE",
                        "expected_truth_self_excluded": False,
                        "truth_csv": str(value.truth_csv),
                        "filters_csv": str(value.filters_csv),
                    }
                },
                "outputs": {
                    "selected": {"path": str(selected), "sha256": selected_sha}
                },
            }
        ),
        encoding="utf-8",
    )
    matched_sha = hashlib.sha256(value.matched_recall_manifest.read_bytes()).hexdigest()
    value.truth_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "ready": True,
                "dataset": {
                    "table": TABLE,
                    "index": SOURCE,
                    "query_table": value.query_table,
                    "query_id_column": "qid",
                    "query_vector_column": "vector",
                    "filter_names": ["f"],
                },
                "database": {
                    "ready": True,
                    "errors": [],
                    "index": SOURCE,
                    "query_rows": 1000,
                    "relations": {TABLE: {}, value.query_table: {}},
                },
                "truth": {
                    "ready": True,
                    "errors": [],
                    "path": str(value.truth_csv),
                    "sha256": truth_sha,
                    "query_count": 3,
                    "row_count": 3,
                },
                "filters": {
                    "errors": [],
                    "path": str(value.filters_csv),
                    "sha256": filters_sha,
                    "count": 1,
                },
                "protocol": {
                    "candidate_validity_predicate": "TRUE",
                    "truth_self_excluded": False,
                    "final": {"offset": 80, "queries": 3, "repeats": 2},
                },
                "generic_manifest": {
                    "path": str(value.matched_recall_manifest),
                    "sha256": matched_sha,
                },
            }
        ),
        encoding="utf-8",
    )

    truth = control.audit_truth_for_args(value)
    configs = control.audit_matched_recall_manifest(
        value.matched_recall_manifest, value, value.filters_csv, value.truth_csv
    )
    command = control.build_runner_command(
        value,
        control.arm_specs(SOURCE, BFS)[0],
        tmp_path / "child.csv",
        tmp_path / "proof.json",
        configs["f"],
    )

    assert truth["provenance_kind"] == "external_launch_manifest"
    assert truth["self_excluded"] is False
    assert command[command.index("--query-offset") + 1] == "80"
    assert command[command.index("--queries") + 1] == "3"
    assert command[command.index("--guidance-max-atoms") + 1] == "128"
    assert "--no-expected-truth-self-excluded" in command

    matched = json.loads(value.matched_recall_manifest.read_text(encoding="utf-8"))
    matched["run_spec"]["args"]["query_table"] = "public.wrong_queries"
    value.matched_recall_manifest.write_text(json.dumps(matched), encoding="utf-8")
    with pytest.raises(control.ControlError, match="query table"):
        control.audit_matched_recall_manifest(
            value.matched_recall_manifest, value, value.filters_csv, value.truth_csv
        )


def test_external_launch_truth_fails_closed_on_index_and_self_exclusion(
    tmp_path: Path,
) -> None:
    value = args(tmp_path)
    value.query_offset = 80
    value.queries = 1
    value.repeats = 1
    value.query_table = "public.external_queries"
    value.candidate_validity_predicate = "TRUE"
    value.expected_truth_self_excluded = False
    truth_sha = write_csv(
        value.truth_csv,
        [
            {
                "query_no": 80,
                "filter_name": "f",
                "candidate_validity_predicate": "TRUE",
                "self_excluded": "False",
                "query_split": "final",
            }
        ],
    )
    filters_sha = hashlib.sha256(value.filters_csv.read_bytes()).hexdigest()
    value.truth_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "ready": True,
                "dataset": {
                    "table": TABLE,
                    "index": "public.wrong_idx",
                    "query_table": value.query_table,
                    "query_id_column": "id",
                    "query_vector_column": "embedding",
                    "filter_names": ["f"],
                },
                "database": {
                    "ready": True,
                    "errors": [],
                    "index": "public.wrong_idx",
                    "relations": {TABLE: {}, value.query_table: {}},
                },
                "truth": {
                    "ready": True,
                    "errors": [],
                    "path": str(value.truth_csv),
                    "sha256": truth_sha,
                },
                "filters": {
                    "errors": [],
                    "path": str(value.filters_csv),
                    "sha256": filters_sha,
                },
                "protocol": {
                    "candidate_validity_predicate": "TRUE",
                    "truth_self_excluded": False,
                    "final": {"offset": 80, "queries": 1},
                },
                "generic_manifest": {
                    "path": str(value.matched_recall_manifest),
                    "sha256": hashlib.sha256(
                        value.matched_recall_manifest.read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(control.ControlError, match="source index"):
        control.audit_truth_for_args(value)

    launch = json.loads(value.truth_manifest.read_text(encoding="utf-8"))
    launch["dataset"]["index"] = SOURCE
    launch["database"]["index"] = SOURCE
    launch["protocol"]["truth_self_excluded"] = True
    value.truth_manifest.write_text(json.dumps(launch), encoding="utf-8")
    with pytest.raises(control.ControlError, match="self-exclusion"):
        control.audit_truth_for_args(value)


def test_matched_configs_csv_selects_only_d1_and_pins_runtime_split_and_provenance(
    tmp_path: Path,
) -> None:
    value = args(tmp_path)
    configs_csv = tmp_path / "matched-configs.csv"
    config_rows = [
        {
            "filter_name": "f",
            "target_recall": "0.9",
            "mode": "original",
            "ef_search": "20",
            "max_scan_tuples": "5000000",
            "scan_mem_multiplier": "32",
            "iterative_scan": "off",
            "qualification": "lcb95",
            "calibration_recall_mean": "0.91",
            "calibration_recall_lcb95": "0.90",
        },
        {
            "filter_name": "f",
            "target_recall": "0.9",
            "mode": "design1_bloom",
            "ef_search": "333",
            "max_scan_tuples": "4000000",
            "scan_mem_multiplier": "16",
            "iterative_scan": "strict_order",
            "qualification": "lcb95",
            "calibration_recall_mean": "0.92",
            "calibration_recall_lcb95": "0.905",
        },
    ]
    configs_sha = write_csv(configs_csv, config_rows)
    configs_manifest = tmp_path / "matched-configs.manifest.json"
    configs_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_valid": True,
                "outputs": {
                    "matched_configs_csv": {
                        "path": str(configs_csv),
                        "sha256": configs_sha,
                    }
                },
                "runtime": {
                    "sqlens_build_id": BUILD,
                    "vector_so_sha256": SHA,
                },
                "protocol": {
                    "mode": "design1_bloom",
                    "table": TABLE,
                    "source_index": SOURCE,
                    "bfs_index": BFS,
                    "query_table": None,
                    "query_id_column": "id",
                    "query_vector_column": "embedding",
                    "candidate_validity_predicate": "embedding_valid",
                    "expected_truth_self_excluded": True,
                    "guidance_filter_strategy": "safe_guided",
                    "guidance_max_atoms": 64,
                    "query_offset": 100,
                    "queries": 100,
                    "repeats": 5,
                },
                "inputs": {
                    "truth_csv": {
                        "path": str(value.truth_csv),
                        "sha256": hashlib.sha256(value.truth_csv.read_bytes()).hexdigest(),
                    },
                    "filters_csv": {
                        "path": str(value.filters_csv),
                        "sha256": hashlib.sha256(value.filters_csv.read_bytes()).hexdigest(),
                    },
                    "truth_provenance_manifest": {
                        "path": str(value.truth_manifest),
                        "sha256": hashlib.sha256(
                            value.truth_manifest.read_bytes()
                        ).hexdigest(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    value.matched_configs_csv = configs_csv
    value.matched_configs_manifest = configs_manifest

    selected = control.load_matched_configs(
        value, value.filters_csv, value.truth_csv
    )

    assert selected["f"].ef_search == 333
    assert selected["f"].max_scan_tuples == 4_000_000
    evidence = control.matched_config_source_evidence(value, selected)
    assert evidence["kind"] == "audited_matched_configs_csv"
    assert evidence["mode"] == "design1_bloom"

    manifest = json.loads(configs_manifest.read_text(encoding="utf-8"))
    manifest["runtime"]["vector_so_sha256"] = "b" * 64
    configs_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(control.ControlError, match="vector.so SHA256"):
        control.load_matched_configs(value, value.filters_csv, value.truth_csv)

    value.matched_configs_manifest = None
    with pytest.raises(control.ControlError, match="requires --matched-configs-manifest"):
        control.load_matched_configs(value, value.filters_csv, value.truth_csv)
