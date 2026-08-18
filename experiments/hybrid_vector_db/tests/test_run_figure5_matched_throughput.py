from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import (
    pgvector_figure5_throughput as core,
)
from experiments.hybrid_vector_db.scripts import (
    run_figure5_matched_latency as latency,
)
from experiments.hybrid_vector_db.scripts import (
    run_figure5_matched_throughput as runner,
)


BUILD = "sqlens-v16-test-r35"
VECTOR_SHA = "a" * 64
RELEASE_SHA = "b" * 64
TRACE_SHA = "c" * 64


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair(
    pair_id: str = "amazon:recall_0.900",
    dataset: str = "amazon",
    target_recall: float = 0.9,
) -> latency.SelectedPair:
    stock = {
        "config_id": "stock-ef100",
        "config_sha256": "d" * 64,
        "ef_search": 100,
        "iterative_scan": "strict_order",
        "max_scan_tuples": 5_000_000,
        "scan_mem_multiplier": 32.0,
        "guided_collect_target": 100,
        "traversal_guided_target": 40,
        "d2_page_access": "off",
        "d2_index_page_access": "off",
        "table": "public.items",
        "index": "public.source_idx",
    }
    sqlens = {
        "config_id": "sqlens-ef250",
        "config_sha256": "e" * 64,
        "ef_search": 250,
        "iterative_scan": "off",
        "max_scan_tuples": 200_000,
        "scan_mem_multiplier": 16.0,
        "guided_collect_target": 250,
        "traversal_guided_target": 80,
        "d2_page_access": "off",
        "d2_index_page_access": "off",
        "table": "public.items",
        "index": "public.bfs_idx",
    }
    return latency.SelectedPair(
        pair_id, dataset, target_recall, stock, sqlens
    )


def config(root: Path, measurement: Path | None = None) -> dict[str, object]:
    release = root / "release.json"
    release.write_text("{}\n", encoding="utf-8")
    measurement = measurement or root / "figure5_r35_amazon_measurement.csv"
    return {
        "release_contract_path": str(release),
        "release_contract_sha256": RELEASE_SHA,
        "release_identity": {
            "contract_id": "figure5-r35-test",
            "expected_sqlens_build_id": BUILD,
            "expected_vector_so_sha256": VECTOR_SHA,
        },
        "protocol": {
            "schedule_seed": 20260728,
            "guidance_max_atoms": 160,
            "d2_page_access": "off",
            "d2_index_page_access": "off",
        },
        "datasets": {
            "amazon": {
                "measurement_workload_csv": str(measurement),
                "table": "public.items",
                "source_index": "public.source_idx",
                "bfs_index": "public.bfs_idx",
                "query_table": "public.items",
            }
        },
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_workload_fixture(
    root: Path, requests: int
) -> tuple[Path, Path]:
    measurement = root / "figure5_r35_amazon_measurement.csv"
    write_csv(
        measurement,
        ["request_no", "query_id", "query_no", "filter_name"],
        [
            {
                "request_no": request_no,
                "query_id": f"q{request_no}",
                "query_no": request_no,
                "filter_name": f"f{request_no}",
            }
            for request_no in range(requests)
        ],
    )
    manifest = root / "figure5_r35_amazon_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "figure5_frontier_workload",
                "artifact_valid": True,
                "outputs": {
                    "measurement_workload_csv": {
                        "path": str(measurement),
                        "rows": requests,
                        "sha256": sha(measurement),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return measurement, manifest


def test_normalized_plan_and_command_keep_independent_arm_settings(
    tmp_path: Path,
) -> None:
    selected = pair()
    cfg = config(tmp_path)
    payload = runner.normalized_measurement_plan(
        [selected],
        config_path=tmp_path / "frontier.json",
        config_sha256="f" * 64,
        selection_csv=tmp_path / "selected.csv",
        selection_plan=tmp_path / "selected.json",
        selection_manifest=tmp_path / "selected.manifest.json",
        selection_bindings={
            "selection_csv_sha256": "1" * 64,
            "selection_plan_sha256": "2" * 64,
            "selection_manifest_sha256": "3" * 64,
            "required_grid_contract": {
                "path": str(tmp_path / "required-grid.json"),
                "sha256": "4" * 64,
            },
        },
        release={"sha256": RELEASE_SHA, **cfg["release_identity"]},
    )
    row = payload["pairs"][0]
    assert row["stock"]["ef_search"] == 100
    assert row["sqlens"]["ef_search"] == 250
    assert row["stock"]["iterative_scan"] == "strict_order"
    assert row["sqlens"]["iterative_scan"] == "off"
    assert row["stock"]["traversal_guided_burst"] == 8
    assert row["sqlens"]["traversal_guided_burst"] == 8
    assert payload["required_grid_contract"]["sha256"] == "4" * 64

    workload = runner.WorkloadBinding(
        path=tmp_path / "workload.manifest.json",
        sha256="4" * 64,
        measurement_path=tmp_path / "measurement.csv",
        measurement_sha256="5" * 64,
    )
    plan = tmp_path / "measurement-plan.json"
    paths = runner.cell_paths(tmp_path, selected, 8)
    command = runner.build_cell_command(
        config_path=tmp_path / "frontier.json",
        config=cfg,
        pair=selected,
        clients=8,
        workload=workload,
        normalized_plan=plan,
        paths=paths,
        run_id="fresh-run-id",
        repeats=3,
        client_cpu_list="0-31",
        backend_cpu_list="48-63",
        backend_proc_root=Path("/proc"),
        telemetry_devices="sda",
        telemetry_paths=(),
        pg_prewarm=True,
        overwrite=False,
        execute=True,
    )
    assert command[command.index("--measurement-plan") + 1] == str(plan)
    assert command[command.index("--pair-id") + 1] == selected.pair_id
    assert command[command.index("--clients") + 1] == "8"
    assert command[command.index("--repeats") + 1] == "3"
    assert command[command.index("--backend-proc-root") + 1] == "/proc"
    assert command[1] == str(Path(runner.__file__).resolve())
    assert command[2] == "--delegate-core"
    assert "--execute" in command
    assert "--ef-search" not in command
    assert "--stock-ef-search" not in command
    assert "--sqlens-ef-search" not in command
    delegated = runner.build_cell_command(
        config_path=tmp_path / "frontier.json",
        config=cfg,
        pair=selected,
        clients=64,
        workload=workload,
        normalized_plan=plan,
        paths=runner.cell_paths(tmp_path, selected, 64),
        run_id="fresh-run-id-64",
        repeats=6,
        client_cpu_list="0-31",
        backend_cpu_list="32-63",
        backend_proc_root=Path("/proc"),
        telemetry_devices="sda",
        telemetry_paths=(),
        pg_prewarm=True,
        overwrite=False,
        execute=True,
    )
    assert delegated[1] == str(Path(runner.__file__).resolve())
    assert delegated[2] == "--delegate-core"
    assert delegated[delegated.index("--clients") + 1] == "64"


def test_delegate_allows_only_preregistered_r3_below_core_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[int, list[str]]] = []

    def fake_main(argv: list[str]) -> int:
        observed.append((core.MIN_REPEATS, list(argv)))
        return 0

    monkeypatch.setattr(core, "main", fake_main)
    monkeypatch.setattr(core, "MIN_REPEATS", runner.CORE_FORMAL_MIN_REPEATS)
    assert runner.delegate_core(
        ["--repeats", "3", "--clients", "16"]
    ) == 0
    assert observed == [
        (3, ["--repeats", "3", "--clients", "16"])
    ]

    monkeypatch.setattr(core, "MIN_REPEATS", runner.CORE_FORMAL_MIN_REPEATS)
    assert runner.delegate_core(
        ["--repeats", "4", "--clients", "16"]
    ) == 2


def test_r36_protocol_slices_freeze_selector_clients_and_repeats() -> None:
    distinct = runner.PROTOCOL_SLICES[runner.DISTINCT_C16_PROTOCOL]
    assert distinct.selection_csv.name == "figure5_r36_matched_configs.csv"
    assert distinct.clients == (16,)
    assert distinct.repeats == 3
    assert distinct.expected_pairs == 32

    fixed = runner.PROTOCOL_SLICES[runner.FIXED_R090_PROTOCOL]
    assert (
        fixed.selection_csv.name
        == "figure5_r36_fixed_target_configs.csv"
    )
    assert fixed.clients == (1, 4, 8, 16, 32, 64)
    assert fixed.repeats == 6
    assert fixed.expected_pairs == 3
    assert fixed.fixed_target_recall == 0.90

    selected = [
        pair(target_recall=0.90),
        pair("amazon:recall_0.950", target_recall=0.95),
    ]
    assert runner.protocol_pairs(selected, fixed) == [selected[0]]

    fixed_targets = runner.PROTOCOL_SLICES[
        runner.FIXED_TARGETS_C16_PROTOCOL
    ]
    assert (
        fixed_targets.selection_csv.name
        == "figure5_r36_fixed_target_configs.csv"
    )
    assert fixed_targets.clients == (16,)
    assert fixed_targets.repeats == 3
    assert fixed_targets.expected_pairs is None
    assert fixed_targets.fixed_target_recall is None
    assert fixed_targets.fixed_targets == (0.90, 0.95, 0.99)
    assert fixed_targets.client_cpu_list == "0-31"
    assert fixed_targets.backend_cpu_list == "48-63"
    assert runner.protocol_pairs(selected, fixed_targets) == selected


def test_parser_rejects_non_preregistered_client_grid(
    tmp_path: Path,
) -> None:
    args = runner.create_parser().parse_args(
        ["--clients", "1", "--out-dir", str(tmp_path)]
    )
    protocol = runner.selected_protocol_slice(args.protocol_slice)
    with pytest.raises(
        runner.MatchedThroughputError,
        match="requires clients",
    ):
        runner.protocol_client_grid(args.clients, protocol)


def test_final_throughput_run_requires_explicit_required_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "frontier.json"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(latency, "load_config", lambda path: config(tmp_path))
    args = runner.create_parser().parse_args(["--config", str(config_path)])

    with pytest.raises(
        runner.MatchedThroughputError,
        match="--required-grid-contract is mandatory",
    ):
        runner.run(args)


def test_explicit_workload_manifest_mapping_and_sha_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 2)
    measurement, manifest = write_workload_fixture(tmp_path, 2)
    binding = runner.validate_workload_manifest(
        {"measurement_workload_csv": str(measurement)}, manifest
    )
    assert binding.path == manifest.resolve()
    assert binding.measurement_sha256 == sha(measurement)
    overrides = runner.workload_manifest_overrides(
        [f"amazon={manifest}", "yfcc=other.json"]
    )
    assert overrides["amazon"] == manifest.resolve()
    assert overrides["yfcc"] == (runner.ROOT / "other.json").resolve()
    with pytest.raises(runner.MatchedThroughputError, match="duplicate"):
        runner.workload_manifest_overrides(
            [f"amazon={manifest}", f"amazon={manifest}"]
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["outputs"]["measurement_workload_csv"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.MatchedThroughputError, match="binding failed"):
        runner.validate_workload_manifest(
            {"measurement_workload_csv": str(measurement)}, manifest
        )


def completion_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    runner.CellPaths,
    latency.SelectedPair,
    dict[str, object],
    Path,
    Path,
    runner.WorkloadBinding,
]:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 2)
    monkeypatch.setattr(runner, "EXPECTED_REPEATS", 2)
    monkeypatch.setattr(runner, "EXPECTED_REQUEST_ROWS", 8)
    monkeypatch.setattr(runner, "EXPECTED_REPEAT_ROWS", 4)
    monkeypatch.setattr(runner, "EXPECTED_FORMAL_PREDICATES", 2)
    monkeypatch.setattr(latency, "MIN_FORMAL_PREDICATE_SAMPLES", 1)
    selected = pair()
    config_path = root / "frontier.json"
    config_path.write_text("{}\n", encoding="utf-8")
    cfg = config(root)
    normalized = root / "normalized.json"
    normalized.write_text(
        json.dumps({"pairs": [runner.normalized_pair(selected)]}) + "\n",
        encoding="utf-8",
    )
    workload_manifest = root / "workload.json"
    workload_manifest.write_text("{}\n", encoding="utf-8")
    measurement = root / "measurement.csv"
    measurement.write_text(
        "request_no,query_id,query_no,filter_name\n"
        "0,q0,0,f0\n"
        "1,q1,1,f1\n",
        encoding="utf-8",
    )
    workload = runner.WorkloadBinding(
        path=workload_manifest,
        sha256=sha(workload_manifest),
        measurement_path=measurement,
        measurement_sha256=sha(measurement),
    )
    paths = runner.cell_paths(root, selected, 2)
    settings = runner.expected_search_settings(selected)
    arm_shas = {
        arm: core.arm_config_sha256(settings, arm)
        for arm in runner.MODES_BY_ARM
    }
    run_id = "fixture-run"
    request_rows: list[dict[str, object]] = []
    repeat_rows: list[dict[str, object]] = []
    schedule_seed = int(cfg["protocol"]["schedule_seed"])
    for repeat_id in range(2):
        trace_seed, trace_order_sha, positions = runner.expected_request_dispatch(
            schedule_seed=schedule_seed,
            dataset_id="amazon10m",
            config_id=settings.config_id,
            clients=2,
            repeat_id=repeat_id,
            requests=2,
        )
        arm_order = core.balanced_arm_order(repeat_id, schedule_seed)
        for arm, mode in runner.MODES_BY_ARM.items():
            for request_no in range(2):
                position = positions[request_no]
                request_rows.append(
                    {
                        "runner_version": core.RUNNER_VERSION,
                        "run_id": run_id,
                        "dataset": "amazon10m",
                        "pair_id": selected.pair_id,
                        "arm_id": arm,
                        "mode_id": mode,
                        "repeat_id": repeat_id,
                        "clients": 2,
                        "release_identity_sha256": RELEASE_SHA,
                        "arm_config_sha256": arm_shas[arm],
                        "request_no": request_no,
                        "dispatch_position": position,
                        "query_id": f"q{request_no}",
                        "query_no": request_no,
                        "filter_name": f"f{request_no}",
                        "client_id": position % 2,
                        "recall_at_10": 0.95,
                        "trace_permutation_seed": trace_seed,
                        "trace_order_sha256": trace_order_sha,
                        "error_type": "",
                        "error": "",
                    }
                )
            wall = 0.5 + repeat_id * 0.1
            repeat_rows.append(
                {
                    "runner_version": core.RUNNER_VERSION,
                    "run_id": run_id,
                    "dataset": "amazon10m",
                    "pair_id": selected.pair_id,
                    "arm_id": arm,
                    "mode_id": mode,
                    "repeat_id": repeat_id,
                    "clients": 2,
                    "release_identity_sha256": RELEASE_SHA,
                    "arm_config_sha256": arm_shas[arm],
                    "status": "valid",
                    "throughput_source": core.THROUGHPUT_SOURCE,
                    "requests": 2,
                    "unique_queries": 2,
                    "completed_queries": 2,
                    "error_count": 0,
                    "wall_clock_seconds": wall,
                    "throughput_qps": 2 / wall,
                    "recall_ci95_low": 0.95,
                    "arm_order": (
                        repeat_id * len(runner.MODES_BY_ARM)
                        + arm_order.index(mode)
                    ),
                    "trace_permutation_seed": trace_seed,
                    "trace_order_sha256": trace_order_sha,
                    "telemetry_collected": "true",
                    "true_concurrency_observed": "true",
                    "request_trace_sha256": TRACE_SHA,
                    "d3_measurement_policy": (
                        "workload_driven_adaptive"
                        if arm == "sqlens_full"
                        else ""
                    ),
                    "d3_namespace_rows_before": (
                        0 if arm == "sqlens_full" else ""
                    ),
                    "d3_online_cost_charged": (
                        "true" if arm == "sqlens_full" else ""
                    ),
                }
            )
    write_csv(paths.requests, list(request_rows[0]), request_rows)
    write_csv(paths.repeats, list(repeat_rows[0]), repeat_rows)
    identity = {
        "exact_match": True,
        "expected_build_id": BUILD,
        "observed_build_id": BUILD,
        "expected_vector_so_sha256": VECTOR_SHA,
        "observed_vector_so_sha256": VECTOR_SHA,
    }
    cell_manifest = {
        "artifact_type": "sqlens_figure5_mixed_q10k_throughput_cell",
        "runner_version": core.RUNNER_VERSION,
        "artifact_valid": True,
        "paper_eligible": True,
        "run_id": run_id,
        "dataset": {"key": "amazon", "dataset_id": "amazon10m"},
        "configuration": {
            "pair_id": selected.pair_id,
            "target_recall": selected.target_recall,
            "stock_config_sha256": arm_shas["stock_pgvector"],
            "sqlens_config_sha256": arm_shas["sqlens_full"],
        },
        "methods": {
            "stock_pgvector": {
                "mode_id": "original",
                "search": runner.expected_search_settings(selected).stock.mode_config(
                    guidance_enabled=False
                ),
                "config_sha256": arm_shas["stock_pgvector"],
            },
            "sqlens_full": {
                "mode_id": "design1_bloom_bfs_layout_d3",
                "search": runner.expected_search_settings(selected).sqlens.mode_config(
                    guidance_enabled=True
                ),
                "config_sha256": arm_shas["sqlens_full"],
                "d3_measurement_policy": "workload_driven_adaptive",
                "unmeasured_query_count": 0,
            },
        },
        "release_contract": {
            "sha256": RELEASE_SHA,
            "expected_sqlens_build_id": BUILD,
            "expected_vector_so_sha256": VECTOR_SHA,
        },
        "protocol": {
            "requests_per_arm_repeat": 2,
            "unique_queries_per_arm_repeat": 2,
            "filters": 14,
            "repeats": 2,
            "clients": 2,
            "schedule_seed": schedule_seed,
            "throughput_source": core.THROUGHPUT_SOURCE,
            "throughput_formula": (
                "completed_queries / barrier_wall_clock_seconds"
            ),
            "independently_tuned_arms": True,
            "independent_connection_per_client": True,
            "pg_prewarm": True,
            "client_cpu_list": "0-31",
            "client_cpu_assignment": [0, 1],
            "backend_cpu_list": "32-63",
        },
        "inputs": {
            "execution_sources": {
                "orchestrator": {
                    "path": str(Path(runner.__file__).resolve()),
                    "sha256": sha(Path(runner.__file__).resolve()),
                },
                "throughput_core": {
                    "path": str(Path(core.__file__).resolve()),
                    "sha256": sha(Path(core.__file__).resolve()),
                },
            },
            "measurement_pair": {
                "source": "measurement_plan",
                "path": str(normalized),
                "sha256": sha(normalized),
                "pair_id": selected.pair_id,
                "target_recall": selected.target_recall,
            },
            "frontier_config": {
                "path": str(config_path),
                "sha256": sha(config_path),
            },
            "workload_manifest": {
                "path": str(workload_manifest),
                "sha256": sha(workload_manifest),
            },
        },
        "gates": {"paired": True, "wall_qps": True},
        "evidence": {
            "runtime_binary_identity_start": identity,
            "runtime_binary_identity_end": identity,
            "prewarm": {
                "enabled": True,
                "complete": True,
                "method": "pg_prewarm(regclass,'read','main')",
                "records": [
                    {
                        "relation": "public.items",
                        "expected_blocks": 10,
                        "warmed_blocks": 10,
                    },
                    {
                        "relation": "public.source_idx",
                        "expected_blocks": 20,
                        "warmed_blocks": 20,
                    },
                    {
                        "relation": "public.bfs_idx",
                        "expected_blocks": 30,
                        "warmed_blocks": 30,
                    },
                ],
            },
        },
        "outputs": {
            "requests": {
                "path": str(paths.requests),
                "rows": 8,
                "sha256": sha(paths.requests),
            },
            "repeats": {
                "path": str(paths.repeats),
                "rows": 4,
                "sha256": sha(paths.repeats),
            },
        },
    }
    paths.manifest.write_text(
        json.dumps(cell_manifest), encoding="utf-8"
    )
    return (
        paths,
        selected,
        cfg,
        config_path,
        normalized,
        workload,
    )


def test_completion_gate_recomputes_wall_clock_qps_and_binds_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        paths,
        selected,
        cfg,
        config_path,
        normalized,
        workload,
    ) = completion_fixture(tmp_path, monkeypatch)
    evidence = runner.cell_completion_evidence(
        paths,
        selected,
        2,
        config=cfg,
        config_path=config_path,
        normalized_plan=normalized,
        normalized_plan_sha=sha(normalized),
        workload=workload,
    )
    assert evidence["complete"] is True
    assert evidence["outputs"]["requests"]["rows"] == 8

    rows = list(csv.DictReader(paths.repeats.open(newline="", encoding="utf-8")))
    rows[0]["throughput_qps"] = "999"
    write_csv(paths.repeats, list(rows[0]), rows)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["outputs"]["repeats"]["sha256"] = sha(paths.repeats)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    evidence = runner.cell_completion_evidence(
        paths,
        selected,
        2,
        config=cfg,
        config_path=config_path,
        normalized_plan=normalized,
        normalized_plan_sha=sha(normalized),
        workload=workload,
    )
    assert evidence["complete"] is False
    assert any(
        "barrier_wall_clock" in reason for reason in evidence["reasons"]
    )


def test_completion_gate_uses_request_cluster_bootstrap_not_core_normal_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, selected, cfg, config_path, normalized, workload = (
        completion_fixture(tmp_path, monkeypatch)
    )
    repeat_rows = list(
        csv.DictReader(paths.repeats.open(newline="", encoding="utf-8"))
    )
    for row in repeat_rows:
        row["recall_ci95_low"] = "0.0"
    write_csv(paths.repeats, list(repeat_rows[0]), repeat_rows)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["outputs"]["repeats"]["sha256"] = sha(paths.repeats)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    evidence = runner.cell_completion_evidence(
        paths,
        selected,
        2,
        config=cfg,
        config_path=config_path,
        normalized_plan=normalized,
        normalized_plan_sha=sha(normalized),
        workload=workload,
    )

    assert evidence["complete"] is True
    assert evidence["recall_gate"]["passed"] is True
    assert (
        evidence["recall_gate"]["recall_ci_method"]
        == "query_id_cluster_stratified_predicate_percentile_bootstrap_95"
    )


def test_throughput_recall_gate_uses_query_cluster_bootstrap_for_six_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 42)
    monkeypatch.setattr(runner, "EXPECTED_REPEATS", 6)
    monkeypatch.setattr(latency, "MIN_FORMAL_PREDICATE_SAMPLES", 3)
    selected = pair()
    rows = [
        {
            "arm_id": arm,
            "repeat_id": repeat,
            "filter_name": f"f{filter_no}",
            "query_id": f"q{filter_no}-{sample_no}",
            "recall_at_10": value,
        }
        for arm in runner.MODES_BY_ARM
        for repeat in range(6)
        for filter_no in range(14)
        for sample_no, value in enumerate((0.9, 0.9, 0.9))
    ]
    gate = runner.matched_recall_gate(rows, selected)
    assert gate["passed"] is True
    assert (
        gate["aggregate"]["stock_pgvector/repeat=5"]["lower"]
        == pytest.approx(0.9)
    )
    assert gate["observed_predicate_count"] == 14
    assert gate["formal_predicate_sample_floor"] == 3
    assert (
        gate["recall_ci_method"]
        == "query_id_cluster_stratified_predicate_percentile_bootstrap_95"
    )

    rows[-1]["recall_at_10"] = 0.8
    gate = runner.matched_recall_gate(rows, selected)
    assert gate["passed"] is False
    assert (
        gate["per_predicate"]["sqlens_full/repeat=5"]["f13"]["lower"]
        < selected.target_recall
    )


def test_throughput_recall_gate_rejects_predicate_hidden_by_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 140)
    monkeypatch.setattr(latency, "MIN_FORMAL_PREDICATE_SAMPLES", 10)
    selected = pair(target_recall=0.90)
    rows = [
        {
            "arm_id": arm,
            "repeat_id": repeat,
            "filter_name": f"f{filter_no}",
            "query_id": f"q{filter_no}-{sample_no}",
            "recall_at_10": 0.80 if filter_no == 13 else 1.0,
        }
        for arm in runner.MODES_BY_ARM
        for repeat in range(3)
        for filter_no in range(14)
        for sample_no in range(10)
    ]

    gate = runner.matched_recall_gate(rows, selected, repeats=3)

    assert all(
        stats["lower"] >= selected.target_recall
        for stats in gate["aggregate"].values()
    )
    assert gate["passed"] is False
    assert gate["paper_eligible"] is False
    assert "per-predicate" in gate["reason"]
    assert gate["worst_predicate"]["filter_name"] == "f13"


def test_failed_completion_retains_per_predicate_recall_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, selected, cfg, config_path, normalized, workload = (
        completion_fixture(tmp_path, monkeypatch)
    )
    rows = list(
        csv.DictReader(paths.requests.open(newline="", encoding="utf-8"))
    )
    rows[-1]["recall_at_10"] = "0.0"
    write_csv(paths.requests, list(rows[0]), rows)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["outputs"]["requests"]["sha256"] = sha(paths.requests)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    evidence = runner.cell_completion_evidence(
        paths,
        selected,
        2,
        config=cfg,
        config_path=config_path,
        normalized_plan=normalized,
        normalized_plan_sha=sha(normalized),
        workload=workload,
    )

    assert evidence["complete"] is False
    assert evidence["recall_gate"]["passed"] is False
    assert evidence["recall_gate"]["per_predicate"]


def test_completion_gate_rejects_frozen_identity_or_client_dispatch_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, selected, cfg, config_path, normalized, workload = completion_fixture(
        tmp_path, monkeypatch
    )
    rows = list(csv.DictReader(paths.requests.open(newline="", encoding="utf-8")))
    original_client_id = rows[1]["client_id"]
    rows[1]["client_id"] = str(1 - int(original_client_id))
    write_csv(paths.requests, list(rows[0]), rows)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["outputs"]["requests"]["sha256"] = sha(paths.requests)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    evidence = runner.cell_completion_evidence(
        paths,
        selected,
        2,
        config=cfg,
        config_path=config_path,
        normalized_plan=normalized,
        normalized_plan_sha=sha(normalized),
        workload=workload,
    )
    assert evidence["complete"] is False
    assert any("client dispatch" in reason for reason in evidence["reasons"])

    rows[1]["client_id"] = original_client_id
    rows[1]["filter_name"] = "wrong-filter"
    write_csv(paths.requests, list(rows[0]), rows)
    manifest["outputs"]["requests"]["sha256"] = sha(paths.requests)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    evidence = runner.cell_completion_evidence(
        paths,
        selected,
        2,
        config=cfg,
        config_path=config_path,
        normalized_plan=normalized,
        normalized_plan_sha=sha(normalized),
        workload=workload,
    )
    assert evidence["complete"] is False
    assert any("frozen workload" in reason for reason in evidence["reasons"])


def test_completion_gate_rejects_effective_mode_config_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, selected, cfg, config_path, normalized, workload = completion_fixture(
        tmp_path, monkeypatch
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["methods"]["sqlens_full"]["search"]["ef_search"] = 999
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    evidence = runner.cell_completion_evidence(
        paths,
        selected,
        2,
        config=cfg,
        config_path=config_path,
        normalized_plan=normalized,
        normalized_plan_sha=sha(normalized),
        workload=workload,
    )
    assert evidence["complete"] is False
    assert any("effective sqlens_full search" in reason for reason in evidence["reasons"])


def test_resume_overwrite_lock_and_client_grid_contract(
    tmp_path: Path,
) -> None:
    assert (
        runner.parse_client_grid([1, 4, 8, 16, 32, 64])
        == runner.DEFAULT_CLIENTS
    )
    with pytest.raises(runner.MatchedThroughputError, match="duplicates"):
        runner.parse_client_grid([1, 4, 4])
    with pytest.raises(runner.MatchedThroughputError, match="increasing"):
        runner.parse_client_grid([2, 1])

    run_manifest = tmp_path / "run.json"
    run_manifest.write_text(
        json.dumps(
            {
                "artifact_type": "sqlens_figure5_matched_throughput_run",
                "runner_version": runner.RUNNER_VERSION,
                "protocol_fingerprint_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    assert (
        runner.validate_existing_run_manifest(
            run_manifest,
            "f" * 64,
            resume=True,
            overwrite=False,
        )
        is not None
    )
    with pytest.raises(runner.MatchedThroughputError, match="incompatible"):
        runner.validate_existing_run_manifest(
            run_manifest,
            "e" * 64,
            resume=True,
            overwrite=False,
        )
    assert (
        runner.validate_existing_run_manifest(
            run_manifest,
            "e" * 64,
            resume=True,
            overwrite=True,
        )
        is None
    )

    completed_manifest = tmp_path / "completed.json"
    completed_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_valid": True,
                "paper_eligible": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(runner.MatchedThroughputError, match="immutable"):
        runner.validate_existing_run_manifest(
            completed_manifest,
            "f" * 64,
            resume=True,
            overwrite=True,
        )

    lock_path = tmp_path / "run.lock"
    first = runner.acquire_lock(lock_path)
    try:
        with pytest.raises(runner.MatchedThroughputError, match="owns"):
            runner.acquire_lock(lock_path)
    finally:
        first.close()

    cpu = runner.validate_cpu_lists(
        "0-31", "32-63", runner.DEFAULT_CLIENTS
    )
    assignment = cpu["client_cpus_for_max_clients"]
    assert len(assignment) == 64
    assert assignment[:32] == list(range(32))
    assert assignment[32:] == list(range(32))
    assert cpu["backend_cpus"] == list(range(32, 64))


def test_full_release_scope_distinguishes_requested_slice() -> None:
    protocol = runner.PROTOCOL_SLICES[runner.FIXED_R090_PROTOCOL]
    selected = [
        pair(),
        pair("yfcc:recall_0.900", "yfcc"),
        pair("laion:recall_0.900", "laion"),
    ]
    bindings = {
        "target_policy": "fixed",
        "target_rows": 9,
        "selected_pairs": 8,
        "unattainable_pairs": 1,
    }
    full_args = runner.create_parser().parse_args(
        ["--protocol-slice", runner.FIXED_R090_PROTOCOL]
    )
    scope = runner.full_release_scope(
        full_args,
        protocol,
        protocol.clients,
        selected,
        selected,
        selection_bindings=bindings,
        enforce_frozen_selector=True,
    )
    assert scope["requested"] is True
    assert all(scope["checks"].values())

    slice_args = runner.create_parser().parse_args(
        [
            "--protocol-slice",
            runner.FIXED_R090_PROTOCOL,
            "--datasets",
            "amazon",
        ]
    )
    scope = runner.full_release_scope(
        slice_args,
        protocol,
        protocol.clients,
        selected[:1],
        selected,
        selection_bindings=bindings,
        enforce_frozen_selector=True,
    )
    assert scope["requested"] is False
    assert scope["checks"]["all_datasets_requested"] is False
    assert scope["checks"]["all_selected_pairs_requested"] is False
    assert scope["checks"]["protocol_client_grid"] is True


def test_all_fixed_targets_full_release_skips_declared_unattainable() -> None:
    protocol = runner.PROTOCOL_SLICES[
        runner.FIXED_TARGETS_C16_PROTOCOL
    ]
    selected = [
        pair(
            f"{dataset}:recall_{target:.3f}",
            dataset,
            target,
        )
        for dataset in ("amazon", "yfcc", "laion")
        for target in (0.90, 0.95, 0.99)
        if not (dataset == "laion" and target == 0.99)
    ]
    bindings = {
        "qualification_scope": latency.QUALIFICATION_SCOPE_FORMAL,
        "target_policy": "fixed",
        "targets_by_dataset": {
            dataset: [0.90, 0.95, 0.99]
            for dataset in ("amazon", "yfcc", "laion")
        },
        "target_rows": 9,
        "selected_pairs": 8,
        "unattainable_pairs": 1,
    }
    args = runner.create_parser().parse_args(
        [
            "--protocol-slice",
            runner.FIXED_TARGETS_C16_PROTOCOL,
            "--backend-cpu-list",
            "48-63",
        ]
    )
    scope = runner.full_release_scope(
        args,
        protocol,
        protocol.clients,
        selected,
        selected,
        selection_bindings=bindings,
        enforce_frozen_selector=True,
    )
    assert scope["requested"] is True
    assert scope["checks"]["protocol_pair_count"] is True
    assert scope["checks"]["selector_resolves_every_fixed_target"] is True
    assert scope["checks"]["selected_pairs_are_unique_fixed_targets"] is True


def test_all_fixed_targets_subset_and_selector_drift_fail_closed() -> None:
    protocol = runner.PROTOCOL_SLICES[
        runner.FIXED_TARGETS_C16_PROTOCOL
    ]
    selected = [
        pair(
            f"{dataset}:recall_{target:.3f}",
            dataset,
            target,
        )
        for dataset in ("amazon", "yfcc", "laion")
        for target in (0.90, 0.95, 0.99)
    ]
    bindings = {
        "qualification_scope": latency.QUALIFICATION_SCOPE_FORMAL,
        "target_policy": "fixed",
        "targets_by_dataset": {
            dataset: [0.90, 0.95, 0.99]
            for dataset in ("amazon", "yfcc", "laion")
        },
        "target_rows": 9,
        "selected_pairs": 9,
        "unattainable_pairs": 0,
    }
    args = runner.create_parser().parse_args(
        [
            "--protocol-slice",
            runner.FIXED_TARGETS_C16_PROTOCOL,
            "--datasets",
            "amazon",
            "--backend-cpu-list",
            "48-63",
        ]
    )
    scope = runner.full_release_scope(
        args,
        protocol,
        protocol.clients,
        selected[:3],
        selected,
        selection_bindings=bindings,
        enforce_frozen_selector=True,
    )
    assert scope["requested"] is False
    assert scope["checks"]["all_datasets_requested"] is False
    assert scope["checks"]["all_selected_pairs_requested"] is False

    drifted = dict(bindings)
    drifted["qualification_scope"] = latency.QUALIFICATION_SCOPE_AGGREGATE
    drifted_scope = runner.full_release_scope(
        runner.create_parser().parse_args(
            [
                "--protocol-slice",
                runner.FIXED_TARGETS_C16_PROTOCOL,
                "--backend-cpu-list",
                "48-63",
            ]
        ),
        protocol,
        protocol.clients,
        selected,
        selected,
        selection_bindings=drifted,
        enforce_frozen_selector=True,
    )
    assert drifted_scope["requested"] is False
    assert (
        drifted_scope["checks"][
            "selector_uses_formal_predicate_qualification"
        ]
        is False
    )


def test_all_fixed_targets_plan_binds_config_selector_and_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 2)
    measurement, workload_manifest = write_workload_fixture(tmp_path, 2)
    cfg = config(tmp_path, measurement)
    dataset_cfg = dict(cfg["datasets"]["amazon"])
    cfg["datasets"] = {
        dataset: dict(dataset_cfg)
        for dataset in ("amazon", "yfcc", "laion")
    }
    config_path = tmp_path / "figure5_r37_formal_datasets.json"
    config_path.write_text('{"release":"r36"}\n', encoding="utf-8")
    selection_csv = tmp_path / "fixed-targets.csv"
    selection_csv.write_text("pair_id\n", encoding="utf-8")
    selection_plan = tmp_path / "fixed-targets.json"
    selection_plan.write_text("{}\n", encoding="utf-8")
    selection_manifest = tmp_path / "fixed-targets.manifest.json"
    selection_manifest.write_text("{}\n", encoding="utf-8")
    required_grid = tmp_path / "required-grid.json"
    required_grid.write_text("{}\n", encoding="utf-8")
    selected = [
        pair(
            f"{dataset}:recall_{target:.3f}",
            dataset,
            target,
        )
        for dataset in ("amazon", "yfcc", "laion")
        for target in (0.90, 0.95, 0.99)
        if not (dataset == "laion" and target == 0.99)
    ]
    bindings = {
        "selection_csv_sha256": "1" * 64,
        "selection_plan_sha256": "2" * 64,
        "selection_manifest_sha256": "3" * 64,
        "qualification_scope": latency.QUALIFICATION_SCOPE_FORMAL,
        "target_policy": "fixed",
        "targets_by_dataset": {
            dataset: [0.90, 0.95, 0.99]
            for dataset in ("amazon", "yfcc", "laion")
        },
        "target_rows": 9,
        "selected_pairs": 8,
        "unattainable_pairs": 1,
        "required_grid_contract": {
            "path": str(required_grid),
            "sha256": sha(required_grid),
        },
    }
    monkeypatch.setattr(latency, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        latency,
        "validate_selection_artifacts",
        lambda *args, **kwargs: bindings,
    )
    monkeypatch.setattr(
        latency,
        "load_selected_pairs",
        lambda *args, **kwargs: selected,
    )
    out_dir = tmp_path / "out"
    argv = [
        "--protocol-slice",
        runner.FIXED_TARGETS_C16_PROTOCOL,
        "--config",
        str(config_path),
        "--selection-csv",
        str(selection_csv),
        "--selection-plan",
        str(selection_plan),
        "--selection-manifest",
        str(selection_manifest),
        "--required-grid-contract",
        str(required_grid),
        "--out-dir",
        str(out_dir),
        "--backend-cpu-list",
        "48-63",
    ]
    for dataset in ("amazon", "yfcc", "laion"):
        argv.extend([
            "--workload-manifest",
            f"{dataset}={workload_manifest}",
        ])
    assert runner.run(runner.create_parser().parse_args(argv)) == 0

    manifest = json.loads(
        (
            out_dir
            / (
                "figure5_r36_fixed-targets-c16-q10k-r3_"
                "throughput_run_manifest.json"
            )
        ).read_text(encoding="utf-8")
    )
    assert manifest["protocol_slice"] == runner.FIXED_TARGETS_C16_PROTOCOL
    assert manifest["full_release_scope"]["requested"] is True
    assert manifest["execution"]["client_grid"] == [16]
    assert manifest["execution"]["repeats"] == 3
    assert manifest["cells_total"] == 8
    assert {
        (cell["dataset"], cell["target_recall"], cell["clients"])
        for cell in manifest["schedule"]
    } == {
        (pair.dataset, pair.target_recall, 16)
        for pair in selected
    }
    assert manifest["frontier_config"]["sha256"] == sha(config_path)
    assert (
        manifest["required_grid_contract"]["sha256"]
        == sha(required_grid)
    )
    assert manifest["selector"]["selection_csv_sha256"] == "1" * 64
    assert manifest["release_contract"]["expected_sqlens_build_id"] == BUILD
    assert manifest["paper_eligible"] is False


def test_plan_run_records_default_grid_serial_reason_and_manifest_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 2)
    measurement, workload_manifest = write_workload_fixture(tmp_path, 2)
    cfg = config(tmp_path, measurement)
    config_path = tmp_path / "frontier.json"
    config_path.write_text("{}\n", encoding="utf-8")
    selection_csv = tmp_path / "selection.csv"
    selection_csv.write_text("pair_id\n", encoding="utf-8")
    selection_plan = tmp_path / "selection.json"
    selection_plan.write_text("{}\n", encoding="utf-8")
    selection_manifest = tmp_path / "selection.manifest.json"
    selection_manifest.write_text("{}\n", encoding="utf-8")
    required_grid = tmp_path / "required-grid.json"
    required_grid.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(latency, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        latency,
        "validate_selection_artifacts",
        lambda *args, **kwargs: {
            "selection_csv_sha256": "1" * 64,
            "selection_plan_sha256": "2" * 64,
            "selection_manifest_sha256": "3" * 64,
            "required_grid_contract": {
                "path": str(required_grid),
                "sha256": sha(required_grid),
            },
        },
    )
    monkeypatch.setattr(
        latency,
        "load_selected_pairs",
        lambda *args, **kwargs: [pair()],
    )
    out_dir = tmp_path / "out"
    args = runner.create_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--selection-csv",
            str(selection_csv),
            "--selection-plan",
            str(selection_plan),
            "--selection-manifest",
            str(selection_manifest),
            "--required-grid-contract",
            str(required_grid),
            "--workload-manifest",
            f"amazon={workload_manifest}",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert runner.run(args) == 0
    manifest = json.loads(
        (
            out_dir
            / "figure5_r36_distinct-c16-q10k-r3_throughput_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["protocol_slice"] == runner.DISTINCT_C16_PROTOCOL
    assert manifest["execution"]["client_grid"] == [16]
    assert manifest["execution"]["repeats"] == 3
    assert manifest["execution"]["client_grid_matches_protocol"] is True
    assert manifest["execution"]["backend_proc_root"] == "/proc"
    assert manifest["execution"]["database_cells_parallel"] is False
    assert (
        manifest["required_grid_contract"]["sha256"]
        == sha(required_grid)
    )
    assert "buffer pool" in manifest["execution"]["why_not_parallel"]
    assert manifest["execution"]["cpu"]["client_cpu_list"] == "0-31"
    assert manifest["execution"]["cpu"]["backend_cpu_list"] == "32-63"
    assert manifest["requested_slice_complete"] is False
    assert manifest["full_release_complete"] is False
    assert manifest["full_release_scope"]["requested"] is False
    assert (
        manifest["datasets"]["amazon"]["workload_manifest_sha256"]
        == sha(workload_manifest)
    )
