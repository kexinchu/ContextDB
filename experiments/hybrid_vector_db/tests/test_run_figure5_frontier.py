from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import run_figure5_frontier as runner


def write_csv(path: Path, count: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["request_no"])
        writer.writeheader()
        for value in range(count):
            writer.writerow({"request_no": value})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_formal_workload_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    calibration = tmp_path / "calibration.csv"
    measurement = tmp_path / "measurement.csv"
    manifest_path = tmp_path / "workload_manifest.json"
    write_csv(calibration, runner.FORMAL_CALIBRATION_REQUESTS)
    write_csv(measurement, 1)
    filter_counts = {
        f"filter_{index}": runner.FORMAL_CALIBRATION_PER_FILTER
        for index in range(runner.FORMAL_CALIBRATION_FILTERS)
    }
    manifest = {
        "artifact_type": "figure5_frontier_workload",
        "artifact_valid": True,
        "construction": {
            "calibration": {
                "protocol": runner.FORMAL_CALIBRATION_PROTOCOL,
                "per_predicate_cartesian": True,
                "query_count": runner.FORMAL_CALIBRATION_PER_FILTER,
                "requests": runner.FORMAL_CALIBRATION_REQUESTS,
            }
        },
        "distribution": {
            "calibration": {
                "filter_counts": filter_counts,
                "cartesian_coverage": {
                    "complete": True,
                    "expected_pairs": runner.FORMAL_CALIBRATION_REQUESTS,
                    "observed_rows": runner.FORMAL_CALIBRATION_REQUESTS,
                    "observed_unique_pairs": runner.FORMAL_CALIBRATION_REQUESTS,
                    "missing_pairs": 0,
                    "duplicate_pairs": 0,
                    "canonical_pair_sha256": "c" * 64,
                },
            }
        },
        "formal_paper_calibration": {"passed": True},
        "outputs": {
            "calibration_workload_csv": {
                "path": str(calibration.resolve()),
                "sha256": sha(calibration),
                "rows": runner.FORMAL_CALIBRATION_REQUESTS,
            },
            "measurement_workload_csv": {
                "path": str(measurement.resolve()),
                "sha256": sha(measurement),
                "rows": 1,
            },
            "manifest_json": {"path": str(manifest_path.resolve())},
        },
    }
    manifest["outputs"]["manifest_json"]["content_sha256"] = (
        runner.manifest_content_sha256(manifest)
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return calibration, measurement, manifest_path


def formal_protocol() -> dict[str, object]:
    return {
        "calibration_requests": runner.FORMAL_CALIBRATION_REQUESTS,
        "calibration_protocol": runner.FORMAL_CALIBRATION_PROTOCOL,
        "qualification_scope": runner.FORMAL_QUALIFICATION_SCOPE,
    }


def test_calibration_command_binds_warm_cache_without_hiding_d3(tmp_path: Path) -> None:
    workload = tmp_path / "workload.csv"
    truth = tmp_path / "truth.csv"
    filters = tmp_path / "filters.csv"
    proof = tmp_path / "proof.json"
    write_csv(workload, 200)
    for path in (truth, filters):
        path.write_text("header\nvalue\n", encoding="utf-8")
    proof.write_text("{}\n", encoding="utf-8")
    config = {
        "protocol": {
            "calibration_requests": 200,
            "latency_repeats": 3,
            "schedule_seed": 7,
            "guidance_filter_strategy": "traversal_guided",
            "d2_page_access": "off",
            "d2_index_page_access": "off",
            "d3_measurement_policy": "workload_driven_adaptive",
            "guidance_max_atoms": 160,
        },
        "search_grid": {
            "max_scan_tuples": 5_000_000,
            "scan_mem_multiplier": 32,
        },
        "release_identity": {
            "contract_id": "sigmod-p0-r36-20260729",
            "expected_sqlens_build_id": "sqlens-v16-test",
            "expected_vector_so_sha256": "a" * 64,
        },
        "release_contract_sha256": "b" * 64,
        "datasets": {
            "amazon": {
                "label": "Amazon-10M",
                "table": "public.heap",
                "query_table": "public.heap",
                "query_id_column": "id",
                "query_vector_column": "embedding",
                "candidate_validity_predicate": "embedding_valid",
                "truth_self_excluded": True,
                "source_index": "public.source_hnsw",
                "bfs_index": "public.bfs_hnsw",
                "calibration_workload_csv": str(workload),
                "truth_csv": str(truth),
                "filters_csv": str(filters),
                "d2_graph_proof_json": str(proof),
            }
        },
    }

    command, provenance = runner.build_cell_command(
        config,
        "amazon",
        "calibration",
        "both_off",
        20,
        tmp_path / "out.csv",
        "48-63",
        calibration_repeats=3,
    )

    assert "--no-require-unique-workload-queries" in command
    assert "--warmup-all-queries" not in command
    positions = [
        index for index, value in enumerate(command) if value == "--prewarm-relation"
    ]
    assert [command[index + 1] for index in positions] == [
        "public.heap",
        "public.source_hnsw",
        "public.bfs_hnsw",
    ]
    assert provenance["cache_protocol"]["d3_materialization_excluded"] is False
    assert provenance["repeats"] == 3
    assert provenance["expected_rows"] == 1200
    repeats_position = command.index("--repeats")
    assert command[repeats_position + 1] == "3"
    assert "--isolate-repeat-runtimes" in command
    assert provenance["isolated_repeat_runtimes"] is True
    assert provenance["d3_fragment_store_namespace"] == (
        "fig5-r36-amazon-calibration-both_off-ef20"
    )
    namespace_position = command.index("--d3-fragment-store-namespace")
    assert command[namespace_position + 1] == provenance["d3_fragment_store_namespace"]
    assert provenance["mode_configs"]["design1_bloom_bfs_layout_d3"][
        "traversal_guided_target"
    ] == 20
    assert runner.mode_configs("both_off", 200, 5_000_000, 32)[
        "design1_bloom_bfs_layout_d3"
    ]["traversal_guided_target"] == 200

    single_repeat_command, single_repeat_provenance = runner.build_cell_command(
        config,
        "amazon",
        "calibration",
        "both_off",
        20,
        tmp_path / "single.csv",
        "48-63",
    )
    assert "--isolate-repeat-runtimes" not in single_repeat_command
    assert single_repeat_provenance["isolated_repeat_runtimes"] is False
    assert single_repeat_provenance["workload_contract"]["required"] is False


def test_formal_workload_contract_requires_audited_cartesian_bundle(
    tmp_path: Path,
) -> None:
    calibration, measurement, manifest_path = write_formal_workload_bundle(tmp_path)
    dataset = {
        "calibration_workload_csv": str(calibration),
        "measurement_workload_csv": str(measurement),
        "workload_manifest_json": str(manifest_path),
    }
    contract = runner.formal_workload_contract("amazon", dataset, formal_protocol())

    assert contract is not None
    assert contract["manifest"]["path"] == str(manifest_path.resolve())
    assert contract["manifest"]["file_sha256"] == sha(manifest_path)
    assert contract["protocol"]["calibration_requests"] == 2800
    assert contract["formal_paper_calibration"]["cartesian_complete"] is True
    assert contract["outputs"]["calibration_workload_csv"]["sha256"] == sha(
        calibration
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_valid"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(runner.Figure5ContractError, match="artifact_valid"):
        runner.formal_workload_contract("amazon", dataset, formal_protocol())


def test_formal_workload_contract_rejects_missing_manifest_and_sha_drift(
    tmp_path: Path,
) -> None:
    calibration, measurement, manifest_path = write_formal_workload_bundle(tmp_path)
    dataset = {
        "calibration_workload_csv": str(calibration),
        "measurement_workload_csv": str(measurement),
    }
    with pytest.raises(runner.Figure5ContractError, match="workload_manifest_json"):
        runner.formal_workload_contract("amazon", dataset, formal_protocol())

    dataset["workload_manifest_json"] = str(manifest_path)
    calibration.write_text("request_no\nchanged\n", encoding="utf-8")
    with pytest.raises(runner.Figure5ContractError, match="SHA-mismatched"):
        runner.formal_workload_contract("amazon", dataset, formal_protocol())


def test_formal_workload_contract_rejects_q200_and_scope_protocol_mismatch(
    tmp_path: Path,
) -> None:
    calibration, measurement, manifest_path = write_formal_workload_bundle(tmp_path)
    dataset = {
        "calibration_workload_csv": str(calibration),
        "measurement_workload_csv": str(measurement),
        "workload_manifest_json": str(manifest_path),
    }
    protocol = formal_protocol()
    protocol["calibration_requests"] = 200
    with pytest.raises(runner.Figure5ContractError, match="calibration_requests"):
        runner.formal_workload_contract("amazon", dataset, protocol)

    protocol = {
        "calibration_protocol": runner.FORMAL_CALIBRATION_PROTOCOL,
        "calibration_requests": runner.FORMAL_CALIBRATION_REQUESTS,
    }
    with pytest.raises(runner.Figure5ContractError, match="qualification_scope"):
        runner.formal_workload_contract("amazon", dataset, protocol)


def test_formal_run_manifest_binds_workload_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration, measurement, manifest_path = write_formal_workload_bundle(tmp_path)
    truth = tmp_path / "truth.csv"
    filters = tmp_path / "filters.csv"
    graph_proof = tmp_path / "graph_proof.json"
    for path in (truth, filters):
        path.write_text("header\nvalue\n", encoding="utf-8")
    graph_proof.write_text("{}\n", encoding="utf-8")
    config = {
        "protocol": {
            **formal_protocol(),
            "latency_repeats": 3,
            "measurement_requests": 1,
            "schedule_seed": 7,
            "guidance_filter_strategy": "traversal_guided",
            "d2_page_access": "off",
            "d2_index_page_access": "off",
            "d3_measurement_policy": "workload_driven_adaptive",
            "guidance_max_atoms": 160,
        },
        "search_grid": {
            "ef_search": [20],
            "extension_ef_search": [],
            "max_scan_tuples": 5_000_000,
            "scan_mem_multiplier": 32,
        },
        "release_identity": {
            "contract_id": "test-r36",
            "expected_sqlens_build_id": "sqlens-v16-test",
            "expected_vector_so_sha256": "a" * 64,
        },
        "release_contract_sha256": "b" * 64,
        "release_contract_path": str(tmp_path / "release.json"),
        "datasets": {
            "amazon": {
                "label": "Amazon-10M",
                "table": "public.heap",
                "query_table": "public.heap",
                "query_id_column": "id",
                "query_vector_column": "embedding",
                "candidate_validity_predicate": "embedding_valid",
                "truth_self_excluded": True,
                "source_index": "public.source_hnsw",
                "bfs_index": "public.bfs_hnsw",
                "calibration_workload_csv": str(calibration),
                "measurement_workload_csv": str(measurement),
                "workload_manifest_json": str(manifest_path),
                "truth_csv": str(truth),
                "filters_csv": str(filters),
                "d2_graph_proof_json": str(graph_proof),
            }
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "load_config", lambda path: config)
    args = runner.create_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--phase",
            "calibration",
            "--datasets",
            "amazon",
            "--ef-search-values",
            "20",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert runner.run(args) == 0
    run_manifest = json.loads(
        (tmp_path / "out/figure5_r35_calibration_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    cell_contract = run_manifest["schedule"][0]["workload_contract"]
    assert run_manifest["workload_contracts"]["amazon"] == cell_contract
    assert cell_contract["manifest"]["file_sha256"] == sha(manifest_path)
    assert cell_contract["outputs"]["measurement_workload_csv"]["path"] == str(
        measurement.resolve()
    )
    assert "database_isolation" not in run_manifest
    assert all(
        "database_isolation" not in cell
        for cell in run_manifest["schedule"]
    )


def test_release_namespace_requires_explicit_release_tag() -> None:
    assert runner.release_namespace_prefix(
        {"contract_id": "sigmod-p0-r36-20260729"}
    ) == "fig5-r36"
    with pytest.raises(runner.Figure5ContractError, match="explicit rNN tag"):
        runner.release_namespace_prefix({"contract_id": "development"})


def test_isolated_namespace_evidence_matches_each_repeat() -> None:
    provenance = {
        "d3_fragment_store_namespace": "fig5-r36-canary",
        "isolated_repeat_runtimes": True,
        "repeats": 3,
    }
    assert runner.fragment_store_namespaces(provenance) == [
        "fig5-r36-canary-r0",
        "fig5-r36-canary-r1",
        "fig5-r36-canary-r2",
    ]
    evidence = {
        "isolated_repeats": True,
        "base_namespace": "fig5-r36-canary",
        "records": [
            {"namespace": namespace, "empty": True, "rows_before": 0}
            for namespace in runner.fragment_store_namespaces(provenance)
        ],
    }
    assert runner.namespace_start_matches(evidence, provenance)
    evidence["records"][1]["rows_before"] = 1
    assert not runner.namespace_start_matches(evidence, provenance)


def test_cell_complete_requires_exact_prewarm_evidence(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    write_csv(raw, 2)
    plan = tmp_path / "raw.csv.plan.json"
    payload = {
        "status": "complete",
        "output_rows": 2,
        "output_sha256": sha(raw),
        "query_error_summary": {"error_rows": 0},
        "relation_prewarm": {
            "enabled": True,
            "complete": True,
            "records": [
                {"expected_blocks": 10, "warmed_blocks": 10}
                for _ in range(3)
            ],
        },
    }
    plan.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.cell_complete(raw, plan, 2)

    payload["relation_prewarm"]["records"][1]["warmed_blocks"] = 9
    plan.write_text(json.dumps(payload), encoding="utf-8")
    assert not runner.cell_complete(raw, plan, 2)


def test_cell_complete_rejects_effective_config_drift(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    fields = [
        "mode",
        "ef_search",
        "max_scan_tuples",
        "scan_mem_multiplier",
        "guided_collect_target",
        "traversal_guided_target",
        "iterative_scan",
        "sqlens_build_id",
        "vector_so_sha256",
        "error",
    ]
    configs = runner.mode_configs("both_off", 200, 5_000_000, 32)
    rows = [
        {
            "mode": mode,
            **{
                field: config[field]
                for field in fields[1:-3]
            },
            "sqlens_build_id": "sqlens-v16-test-r36",
            "vector_so_sha256": "a" * 64,
            "error": "",
        }
        for mode, config in configs.items()
    ]
    with raw.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "modes": list(configs),
        "mode_configs": configs,
        "requests": 1,
        "repeats": 1,
        "inputs": {
            "workload": {"sha256": "c" * 64},
            "truth": {"sha256": "d" * 64},
            "filters": {"sha256": "e" * 64},
            "d2_graph_proof": {"canonical_json_sha256": "f" * 64},
        },
        "execution_sources": {
            "core_runner": {"path": "/core.py", "sha256": "1" * 64},
            "orchestrator": {"path": "/runner.py", "sha256": "2" * 64},
        },
        "d3_fragment_store_namespace": "fresh-ns",
        "release_identity": {
            "contract_id": "test-r36",
            "expected_sqlens_build_id": "sqlens-v16-test-r36",
            "expected_vector_so_sha256": "a" * 64,
            "contract_sha256": "b" * 64,
        },
        "release_contract_sha256": "b" * 64,
    }
    plan = tmp_path / "raw.csv.plan.json"
    payload = {
        "status": "complete",
        "output_rows": 2,
        "output_sha256": sha(raw),
        "query_error_summary": {"error_rows": 0},
        "query_contract": {
            "workload_sha256": "c" * 64,
            "truth_sha256": "d" * 64,
            "filters_sha256": "e" * 64,
            "d2_graph_proof_input_sha256": "f" * 64,
        },
        "execution_sources": provenance["execution_sources"],
        "d3_fragment_store_start": {
            "namespace": "fresh-ns",
            "empty": True,
            "rows_before": 0,
        },
        "sqlens_runtime_identity_startup": {
            "expected_build_id": "sqlens-v16-test-r36",
            "expected_vector_so_sha256": "a" * 64,
            "observed_build_id": "sqlens-v16-test-r36",
            "observed_vector_so_sha256": "a" * 64,
            "exact_match": True,
        },
        "sqlens_runtime_identity_final": {
            "expected_build_id": "sqlens-v16-test-r36",
            "expected_vector_so_sha256": "a" * 64,
            "observed_build_id": "sqlens-v16-test-r36",
            "observed_vector_so_sha256": "a" * 64,
            "exact_match": True,
        },
        "runtime_sqlens_identity_evidence": [
            {
                "expected_build_id": "sqlens-v16-test-r36",
                "expected_vector_so_sha256": "a" * 64,
                "observed_build_id": "sqlens-v16-test-r36",
                "observed_vector_so_sha256": "a" * 64,
                "exact_match": True,
            }
            for _ in range(2)
        ],
        "relation_prewarm": {
            "enabled": True,
            "complete": True,
            "records": [
                {"expected_blocks": 10, "warmed_blocks": 10}
                for _ in range(3)
            ],
        },
    }
    plan.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.cell_complete(raw, plan, 2, provenance)

    payload["sqlens_runtime_identity_final"]["observed_build_id"] = (
        "sqlens-v16-stale-r35"
    )
    plan.write_text(json.dumps(payload), encoding="utf-8")
    assert not runner.cell_complete(raw, plan, 2, provenance)
    payload["sqlens_runtime_identity_final"]["observed_build_id"] = (
        "sqlens-v16-test-r36"
    )

    rows[1]["traversal_guided_target"] = 40
    with raw.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload["output_sha256"] = sha(raw)
    plan.write_text(json.dumps(payload), encoding="utf-8")
    assert not runner.cell_complete(raw, plan, 2, provenance)


def test_cell_complete_rejects_any_query_error_row(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["request_no", "error"])
        writer.writeheader()
        writer.writerow({"request_no": 0, "error": ""})
        writer.writerow({"request_no": 1, "error": "RuntimeError"})
    plan = tmp_path / "raw.csv.plan.json"
    plan.write_text(
        json.dumps(
            {
                "status": "complete",
                "output_rows": 2,
                "output_sha256": sha(raw),
                "query_error_summary": {"error_rows": 1},
                "relation_prewarm": {
                    "enabled": True,
                    "complete": True,
                    "records": [
                        {"expected_blocks": 10, "warmed_blocks": 10}
                        for _ in range(3)
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    assert not runner.cell_complete(raw, plan, 2)


def test_both_off_calibration_rejects_unbounded_high_guided_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda path: {
            "datasets": {"amazon": {}},
            "search_grid": {
                "ef_search": [1000, 1500],
                "extension_ef_search": [],
            },
        },
    )
    args = runner.create_parser().parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--datasets",
            "amazon",
            "--scan-families",
            "both_off",
        ]
    )
    with pytest.raises(runner.Figure5ContractError, match="capped"):
        runner.run(args)


def test_expensive_sqlens_calibration_requires_explicit_admission() -> None:
    with pytest.raises(runner.Figure5ContractError, match="explicit"):
        runner.validate_calibration_budget_policy(
            "calibration",
            ("both_off",),
            (1500,),
            allow_expensive_sqlens_calibration=False,
        )

    runner.validate_calibration_budget_policy(
        "calibration",
        ("both_off",),
        (1500, 2000),
        allow_expensive_sqlens_calibration=True,
    )


def test_formal_calibration_suite_has_exactly_146_unique_cells() -> None:
    config = {
        "datasets": {
            "amazon": {},
            "yfcc": {},
            "laion": {},
        },
        "search_grid": {
            "ef_search": [
                20,
                40,
                60,
                80,
                100,
                150,
                200,
                250,
                500,
                750,
                1000,
                1500,
                2000,
                3000,
                4000,
                5000,
                7000,
                8500,
                10000,
            ]
        },
    }
    cells = runner.formal_calibration_cells(config)
    assert len(cells) == 146
    assert len(set(cells)) == 146
    assert ("laion", "both_off", 1500, None) in cells
    assert ("laion", "both_off", 2000, None) in cells
    assert ("amazon", "both_off", 1500, None) not in cells
    assert ("yfcc", "both_off", 2000, None) not in cells
    assert sum(family == "both_off" for _, family, _, _ in cells) == 50
    assert sum(family == "stock_strict" for _, family, _, _ in cells) == 72
    assert sum(
        family == runner.SQLENS_CAP_FAMILY
        for _, family, _, _ in cells
    ) == 24
    assert (
        "amazon",
        runner.SQLENS_CAP_FAMILY,
        11,
        500,
    ) in cells


def test_sqlens_cap_has_distinct_path_and_mode_budget(tmp_path: Path) -> None:
    raw, plan = runner.cell_paths(
        tmp_path,
        "amazon",
        "calibration",
        runner.SQLENS_CAP_FAMILY,
        11,
        1000,
    )
    assert raw.name.endswith("sqlens_cap_ef11_cap1000.csv")
    assert plan.name.endswith("sqlens_cap_ef11_cap1000.csv.plan.json")

    configs = runner.mode_configs(
        runner.SQLENS_CAP_FAMILY,
        11,
        5_000_000,
        32,
        1000,
    )
    assert configs["original"]["max_scan_tuples"] == 5_000_000
    assert (
        configs["design1_bloom_bfs_layout_d3"]["max_scan_tuples"]
        == 1000
    )


def test_cell_paths_support_release_specific_artifact_prefix(tmp_path: Path) -> None:
    raw, plan = runner.cell_paths(
        tmp_path,
        "laion",
        "calibration",
        "both_off",
        1000,
        artifact_prefix="figure5_r41",
    )

    assert raw.name == "figure5_r41_laion_calibration_both_off_ef1000.csv"
    assert plan.name == raw.name + ".plan.json"


def prepare_minimal_locked_frontier_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    execute: bool,
) -> tuple[argparse.Namespace, Path]:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    config = {
        "protocol": {},
        "search_grid": {
            "ef_search": [20],
            "extension_ef_search": [],
        },
        "release_contract_path": str(tmp_path / "release.json"),
        "release_contract_sha256": "a" * 64,
        "release_identity": {"contract_id": "test-r36"},
        "datasets": {"amazon": {}},
    }
    monkeypatch.setattr(runner, "load_config", lambda path: config)

    def fake_build_cell_command(
        config: dict[str, object],
        dataset_name: str,
        phase: str,
        family: str,
        ef_search: int,
        raw: Path,
        backend_cpu_list: str,
        calibration_repeats: int,
        sqlens_scan_cap: int | None,
    ) -> tuple[list[str], dict[str, object]]:
        del (
            config,
            phase,
            backend_cpu_list,
            calibration_repeats,
            sqlens_scan_cap,
        )
        return ["fake-db-cell", str(raw)], {
            "dataset": dataset_name,
            "scan_family": family,
            "ef_search": ef_search,
            "expected_rows": 1,
            "modes": ["design1_bloom_bfs_layout_d3"],
            "workload_contract": {"required": False},
            "d3_fragment_store_table": "public.fragments",
            "d3_fragment_store_namespace": "test-r36-cell",
            "isolated_repeat_runtimes": False,
        }

    monkeypatch.setattr(runner, "build_cell_command", fake_build_cell_command)
    monkeypatch.setattr(
        runner,
        "cell_complete",
        lambda raw, plan, expected_rows, provenance=None: raw.is_file(),
    )
    lock_path = tmp_path / "shared-formal-db.lock"
    argv = [
        "--config",
        str(config_path),
        "--phase",
        "calibration",
        "--datasets",
        "amazon",
        "--ef-search-values",
        "20",
        "--scan-families",
        "stock_strict",
        "--out-dir",
        str(tmp_path / "out"),
        "--overwrite",
        "--no-resume",
        "--require-global-db-lock",
        "--global-db-lock-path",
        str(lock_path),
    ]
    if execute:
        argv.append("--execute")
    return runner.create_parser().parse_args(argv), lock_path


def test_locked_run_records_manifest_and_cell_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, lock_path = prepare_minimal_locked_frontier_run(
        tmp_path,
        monkeypatch,
        execute=True,
    )
    lock_seen_during_reset = False
    lock_seen_during_cell = False
    lock_seen_during_final_manifest = False

    def assert_lock_owned() -> None:
        with pytest.raises(runner.Figure5ContractError, match="already owned"):
            runner.acquire_global_db_lock(lock_path, "conflicting-runner")

    original_atomic_json = runner.atomic_json

    def audited_atomic_json(
        path: Path,
        payload: dict[str, object],
    ) -> None:
        nonlocal lock_seen_during_final_manifest
        isolation = payload.get("database_isolation")
        if (
            payload.get("status") == "complete"
            and isinstance(isolation, dict)
            and isolation.get("held_through_completion") is True
        ):
            assert_lock_owned()
            lock_seen_during_final_manifest = True
        original_atomic_json(path, payload)

    def fake_reset(table: str, namespace: str) -> int:
        nonlocal lock_seen_during_reset
        assert table == "public.fragments"
        assert namespace == "test-r36-cell"
        assert_lock_owned()
        lock_seen_during_reset = True
        return 0

    def fake_subprocess(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal lock_seen_during_cell
        del kwargs
        assert_lock_owned()
        lock_seen_during_cell = True
        Path(command[-1]).write_text("result\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner, "clear_fragment_store_namespace", fake_reset)
    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(runner, "atomic_json", audited_atomic_json)

    assert runner.run(args) == 0
    assert lock_seen_during_reset is True
    assert lock_seen_during_cell is True
    assert lock_seen_during_final_manifest is True
    manifest = json.loads(
        (tmp_path / "out/figure5_r35_calibration_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    top = manifest["database_isolation"]
    cell = manifest["schedule"][0]["database_isolation"]
    assert manifest["status"] == "complete"
    assert top["parallel_db_cells"] is False
    assert top["lock_required"] is True
    assert top["lock_path"] == str(lock_path.resolve())
    assert top["held_through_completion"] is True
    assert cell["held_through_completion"] is True
    assert cell["lock_owner_token"] == top["lock_owner_token"]

    released = runner.acquire_global_db_lock(lock_path, "after-completion")
    released.close()


def test_locked_dry_run_does_not_claim_lock_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, lock_path = prepare_minimal_locked_frontier_run(
        tmp_path,
        monkeypatch,
        execute=False,
    )

    assert runner.run(args) == 0
    manifest = json.loads(
        (tmp_path / "out/figure5_r35_calibration_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = manifest["database_isolation"]
    assert evidence["parallel_db_cells"] is None
    assert evidence["lock_required"] is True
    assert evidence["lock_path"] == str(lock_path.resolve())
    assert evidence["lock_acquired"] is False
    assert evidence["held_through_completion"] is False
    assert "database_isolation" not in manifest["schedule"][0]


def test_custom_global_lock_path_requires_explicit_lock_flag(
    tmp_path: Path,
) -> None:
    args = runner.create_parser().parse_args(
        ["--global-db-lock-path", str(tmp_path / "shared.lock")]
    )
    with pytest.raises(
        runner.Figure5ContractError,
        match="requires --require-global-db-lock",
    ):
        runner.global_db_lock_path(args)


def test_locked_run_releases_after_unexpected_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, lock_path = prepare_minimal_locked_frontier_run(
        tmp_path,
        monkeypatch,
        execute=True,
    )
    monkeypatch.setattr(
        runner,
        "clear_fragment_store_namespace",
        lambda table, namespace: 0,
    )

    def fail_subprocess(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("injected runner failure")

    monkeypatch.setattr(runner.subprocess, "run", fail_subprocess)
    with pytest.raises(RuntimeError, match="injected runner failure"):
        runner.run(args)

    manifest = json.loads(
        (tmp_path / "out/figure5_r35_calibration_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "failed"
    assert (
        manifest["database_isolation"]["held_through_completion"] is True
    )
    reacquired = runner.acquire_global_db_lock(lock_path, "after-failure")
    reacquired.close()
