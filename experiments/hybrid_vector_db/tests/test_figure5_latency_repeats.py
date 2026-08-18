from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import figure5_latency_repeats as repeats


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_query_cluster_bootstrap_resamples_query_ids_not_repeat_rows() -> None:
    rows = [
        {
            "query_id": query_id,
            "filter_name": filter_name,
            "recall": recall,
        }
        for query_id, filter_name, recall in (
            ("q0", "f0", 0.0),
            ("q1", "f0", 1.0),
            ("q2", "f1", 0.8),
            ("q3", "f1", 1.0),
        )
        for _ in range(3)
    ]

    first = repeats.query_cluster_bootstrap_recall(
        rows, value_field="recall", samples=500, seed=7, seed_label="cell"
    )
    second = repeats.query_cluster_bootstrap_recall(
        rows, value_field="recall", samples=500, seed=7, seed_label="cell"
    )

    assert first == second
    assert first["sample_count"] == 12
    assert first["query_cluster_count"] == 4
    assert first["predicate_count"] == 2
    assert first["per_predicate"]["f0"]["query_cluster_count"] == 2
    assert first["method"] == repeats.CLUSTER_BOOTSTRAP_METHOD


def test_query_cluster_bootstrap_rejects_unequal_repeat_coverage() -> None:
    rows = [
        {"query_id": "q0", "filter_name": "f0", "recall": 1.0},
        {"query_id": "q0", "filter_name": "f0", "recall": 1.0},
        {"query_id": "q1", "filter_name": "f0", "recall": 1.0},
    ]

    with pytest.raises(
        repeats.LatencyRepeatError,
        match="equal repeat coverage",
    ):
        repeats.query_cluster_bootstrap_recall(rows, value_field="recall")


def make_cell(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    raw = tmp_path / "cell.csv"
    rows: list[dict[str, object]] = []
    for repeat in range(2):
        for request_no in range(3):
            for mode in repeats.MODE_ARMS:
                rows.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "request_no": request_no,
                        "query_id": 100 + request_no,
                        "filter_name": f"f{request_no}",
                        "recall": 0.8 + 0.01 * request_no,
                        "end_to_end_ms": 10 + request_no,
                        "error": "",
                        "sqlens_build_id": "sqlens-v16-test",
                        "vector_so_sha256": "a" * 64,
                    }
                )
    write_csv(raw, rows)
    plan = raw.with_suffix(".plan.json")
    plan.write_text(
        json.dumps(
            {
                "status": "complete",
                "output_rows": len(rows),
                "output_sha256": file_sha(raw),
                "relation_prewarm": {
                    "enabled": True,
                    "complete": True,
                    "records": [
                        {
                            "relation": f"r{index}",
                            "expected_blocks": 10,
                            "warmed_blocks": 10,
                        }
                        for index in range(3)
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    cell: dict[str, object] = {
        "dataset": "amazon",
        "phase": "measurement",
        "raw": str(raw),
        "plan": str(plan),
        "expected_rows": len(rows),
        "repeats": 2,
        "requests": 3,
        "scan_family": "both_off",
        "ef_search": 100,
        "mode_configs": {
            "original": {"ef_search": 100, "iterative_scan": "off"},
            "design1_bloom_bfs_layout_d3": {
                "ef_search": 100,
                "iterative_scan": "off",
            },
        },
        "inputs": {"workload": {"sha256": "b" * 64}},
        "cache_protocol": {"state": "warm"},
    }
    release = {
        "expected_sqlens_build_id": "sqlens-v16-test",
        "expected_vector_so_sha256": "a" * 64,
    }
    return cell, release


def test_convert_cell_builds_paired_repeat_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repeats.artifact, "EXPECTED_REQUESTS", 3)
    monkeypatch.setitem(repeats.artifact.MIN_REPEATS, "latency", 2)
    cell, release = make_cell(tmp_path)

    rows = repeats.convert_cell(
        cell, release_sha256="c" * 64, release=release
    )

    assert len(rows) == 4
    assert {row["arm_id"] for row in rows} == {
        "stock_pgvector",
        "sqlens_full",
    }
    assert {row["run_id"] for row in rows} == {rows[0]["run_id"]}
    assert {row["repeat_id"] for row in rows} == {0, 1}
    assert all(row["completed_queries"] == 3 for row in rows)
    assert all(row["throughput_qps"] == "" for row in rows)


def test_convert_manifest_preserves_legacy_frontier_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repeats.artifact, "EXPECTED_REQUESTS", 3)
    monkeypatch.setitem(repeats.artifact.MIN_REPEATS, "latency", 2)
    cell, release = make_cell(tmp_path)
    cell["status"] = "complete"
    manifest = tmp_path / "legacy-run.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "sqlens_figure5_frontier_run",
                "phase": "measurement",
                "release_contract": {"sha256": "c" * 64, **release},
                "schedule": [cell],
            }
        ),
        encoding="utf-8",
    )

    rows = repeats.convert_manifest(manifest)

    assert len(rows) == 4
    assert {row["config_id"] for row in rows} == {"both_off_ef100"}


def test_convert_cell_rejects_unpaired_request_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repeats.artifact, "EXPECTED_REQUESTS", 3)
    monkeypatch.setitem(repeats.artifact.MIN_REPEATS, "latency", 2)
    cell, release = make_cell(tmp_path)
    raw = Path(str(cell["raw"]))
    with raw.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    target = next(
        row
        for row in rows
        if row["mode"] == "design1_bloom_bfs_layout_d3"
        and row["repeat"] == "0"
        and row["request_no"] == "0"
    )
    target["filter_name"] = "different"
    write_csv(raw, rows)
    plan = Path(str(cell["plan"]))
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["output_sha256"] = file_sha(raw)
    plan.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(repeats.LatencyRepeatError, match="paired trace mismatch"):
        repeats.convert_cell(cell, release_sha256="c" * 64, release=release)


def test_convert_cell_rejects_incomplete_prewarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repeats.artifact, "EXPECTED_REQUESTS", 3)
    monkeypatch.setitem(repeats.artifact.MIN_REPEATS, "latency", 2)
    cell, release = make_cell(tmp_path)
    plan = Path(str(cell["plan"]))
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["relation_prewarm"]["records"][0]["warmed_blocks"] = 9
    plan.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(repeats.LatencyRepeatError, match="warm-cache evidence"):
        repeats.convert_cell(cell, release_sha256="c" * 64, release=release)


MATCHED_BUILD = "sqlens-v16-test-r35"
MATCHED_VECTOR_SHA = "d" * 64
MATCHED_PAIR_ID = "amazon:recall_0.900"


def matched_selector_row() -> dict[str, str]:
    row = {
        "pair_id": MATCHED_PAIR_ID,
        "dataset": "amazon",
        "target_recall": "0.9",
        "selection_status": "selected",
        "qualification_scope": (
            repeats.matched_latency.QUALIFICATION_SCOPE_FORMAL
        ),
    }
    fields = (
        "config_id",
        "config_sha256",
        "ef_search",
        "iterative_scan",
        "max_scan_tuples",
        "scan_mem_multiplier",
        "guided_collect_target",
        "traversal_guided_target",
        "d2_page_access",
        "d2_index_page_access",
        "table",
        "index",
    )
    values = {
        "stock": (
            "stock-ef100",
            "a" * 64,
            "100",
            "strict_order",
            "5000000",
            "32",
            "100",
            "40",
            "off",
            "off",
            "public.items",
            "public.source_idx",
        ),
        "sqlens": (
            "sqlens-ef250",
            "b" * 64,
            "250",
            "off",
            "200000",
            "16",
            "250",
            "80",
            "off",
            "off",
            "public.items",
            "public.bfs_idx",
        ),
    }
    for arm, arm_values in values.items():
        row.update(
            {
                f"{arm}_{field}": value
                for field, value in zip(fields, arm_values)
            }
        )
    return row


def matched_identity() -> dict[str, object]:
    return {
        "exact_match": True,
        "expected_build_id": MATCHED_BUILD,
        "observed_build_id": MATCHED_BUILD,
        "expected_vector_so_sha256": MATCHED_VECTOR_SHA,
        "observed_vector_so_sha256": MATCHED_VECTOR_SHA,
    }


def use_small_matched_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repeats.matched_latency, "EXPECTED_REQUESTS", 3)
    monkeypatch.setattr(repeats.matched_latency, "EXPECTED_REPEATS", 2)
    monkeypatch.setattr(repeats.matched_latency, "EXPECTED_ROWS", 12)
    monkeypatch.setattr(
        repeats.matched_latency, "EXPECTED_FORMAL_PREDICATES", 3
    )
    monkeypatch.setattr(
        repeats.matched_latency, "MIN_FORMAL_PREDICATE_SAMPLES", 1
    )


def make_matched_manifest(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], Path, Path]:
    release_path = tmp_path / "release.json"
    release_path.write_text('{"release":"r35"}\n', encoding="utf-8")
    frontier_config = tmp_path / "frontier.json"
    frontier_config.write_text("{}\n", encoding="utf-8")
    required_grid = tmp_path / "required-grid.json"
    required_grid.write_text(
        json.dumps(
            {
                "contract_type": (
                    repeats.matched_latency.REQUIRED_GRID_CONTRACT_TYPE
                ),
                "grid_complete": True,
                "qualification_scope": (
                    repeats.matched_latency.QUALIFICATION_SCOPE_FORMAL
                ),
                "dataset_config": {
                    "path": str(frontier_config),
                    "sha256": file_sha(frontier_config),
                },
            }
        ),
        encoding="utf-8",
    )
    required_grid_binding = {
        "path": str(required_grid),
        "sha256": file_sha(required_grid),
    }
    workload = tmp_path / "workload.csv"
    write_csv(
        workload,
        [{"request_no": request_no} for request_no in range(3)],
    )
    truth = tmp_path / "truth.csv"
    truth.write_text("truth\n", encoding="utf-8")
    filters = tmp_path / "filters.csv"
    filters.write_text("filters\n", encoding="utf-8")
    proof = tmp_path / "proof.json"
    proof.write_text('{"proof":"same-graph"}\n', encoding="utf-8")

    selector_row = matched_selector_row()
    selector_csv = tmp_path / "selector.csv"
    write_csv(selector_csv, [selector_row])
    release = {
        "path": str(release_path),
        "sha256": file_sha(release_path),
        "contract_id": "sqlens-r35-test-contract",
        "expected_sqlens_build_id": MATCHED_BUILD,
        "expected_vector_so_sha256": MATCHED_VECTOR_SHA,
    }
    selector_plan = tmp_path / "selector.json"
    selector_plan.write_text(
        json.dumps(
            {
                "artifact_valid": True,
                "qualification_scope": (
                    repeats.matched_latency.QUALIFICATION_SCOPE_FORMAL
                ),
                "execution_source": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": file_sha(Path(__file__).resolve()),
                },
                "measurement_plan_csv": {"sha256": file_sha(selector_csv)},
                "release_contract": release,
                "required_grid_contract": required_grid_binding,
            }
        ),
        encoding="utf-8",
    )
    selector_manifest = tmp_path / "selector.manifest.json"
    selector_manifest.write_text(
        json.dumps(
            {
                "artifact_valid": True,
                "qualification_scope": (
                    repeats.matched_latency.QUALIFICATION_SCOPE_FORMAL
                ),
                "release_contract": release,
                "required_grid_contract": required_grid_binding,
                "outputs": {
                    "measurement_plan_csv": {
                        "sha256": file_sha(selector_csv)
                    },
                    "measurement_plan_json": {
                        "sha256": file_sha(selector_plan)
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    stock = repeats.matched_latency.arm_config(selector_row, "stock")
    sqlens = repeats.matched_latency.arm_config(selector_row, "sqlens")
    selected_pair = repeats.matched_latency.SelectedPair(
        MATCHED_PAIR_ID,
        "amazon",
        0.9,
        stock,
        sqlens,
    )
    base_namespace = repeats.matched_latency.pair_namespace(selected_pair)
    repeat_namespaces = repeats.matched_latency.pair_repeat_namespaces(
        selected_pair
    )
    execution_sources = {
        "core_runner": {"path": "/core.py", "sha256": "1" * 64},
        "orchestrator": {"path": "/runner.py", "sha256": "2" * 64},
    }
    mode_configs = {
        "original": {
            **{
                field: stock[field]
                for field in (
                    "ef_search",
                    "iterative_scan",
                    "max_scan_tuples",
                    "scan_mem_multiplier",
                    "guided_collect_target",
                    "traversal_guided_target",
                    "d2_page_access",
                    "d2_index_page_access",
                )
            },
            "traversal_guided_burst": 8,
            "traversal_guided_prioritization": False,
        },
        "design1_bloom_bfs_layout_d3": {
            **{
                field: sqlens[field]
                for field in (
                    "ef_search",
                    "iterative_scan",
                    "max_scan_tuples",
                    "scan_mem_multiplier",
                    "guided_collect_target",
                    "traversal_guided_target",
                    "d2_page_access",
                    "d2_index_page_access",
                )
            },
            "traversal_guided_burst": 8,
            "traversal_guided_prioritization": True,
        },
    }
    raw = tmp_path / "matched.csv"
    rows: list[dict[str, object]] = []
    for repeat in range(2):
        for request_no in range(3):
            for mode, selected in (
                ("original", stock),
                ("design1_bloom_bfs_layout_d3", sqlens),
            ):
                rows.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "request_no": request_no,
                        "query_id": 100 + request_no,
                        "filter_name": f"f{request_no}",
                        "recall": 0.9,
                        "end_to_end_ms": 10 + request_no,
                        "error": "",
                        "sqlens_build_id": MATCHED_BUILD,
                        "vector_so_sha256": MATCHED_VECTOR_SHA,
                        "ef_search": selected["ef_search"],
                        "iterative_scan": selected["iterative_scan"],
                        "max_scan_tuples": selected["max_scan_tuples"],
                        "scan_mem_multiplier": selected[
                            "scan_mem_multiplier"
                        ],
                        "guided_collect_target": selected[
                            "guided_collect_target"
                        ],
                        "traversal_guided_target": selected[
                            "traversal_guided_target"
                        ],
                        "d2_page_access": selected["d2_page_access"],
                        "d2_index_page_access": selected[
                            "d2_index_page_access"
                        ],
                        "d3_fragment_store_namespace": repeat_namespaces[
                            repeat
                        ],
                        "block_no": repeat * 3 + request_no,
                        "schedule_position": (
                            1 if mode == "original" else 2
                        ),
                        "query_order_position": request_no + 1,
                        "execution_order": "interleaved",
                    }
                )
    write_csv(raw, rows)
    plan = tmp_path / "matched.plan.json"
    plan.write_text(
        json.dumps(
            {
                "status": "complete",
                "output_rows": 12,
                "output_sha256": file_sha(raw),
                "relation_prewarm": {
                    "enabled": True,
                    "complete": True,
                    "records": [
                        {
                            "expected_blocks": 10,
                            "warmed_blocks": 10,
                        }
                        for _ in range(3)
                    ],
                },
                "sqlens_runtime_identity_startup": matched_identity(),
                "sqlens_runtime_identity_final": matched_identity(),
                "runtime_sqlens_identity_evidence": [
                    matched_identity() for _ in range(4)
                ],
                "d3_fragment_store_start": {
                    "isolated_repeats": True,
                    "base_namespace": base_namespace,
                    "records": [
                        {
                            "namespace": namespace,
                            "empty": True,
                            "rows_before": 0,
                        }
                        for namespace in repeat_namespaces
                    ],
                },
                "query_contract": {
                    "workload_sha256": file_sha(workload),
                    "truth_sha256": file_sha(truth),
                    "filters_sha256": file_sha(filters),
                    "d2_graph_proof_input_sha256": (
                        repeats.matched_latency.frontier.sha256_json(
                            json.loads(proof.read_text(encoding="utf-8"))
                        )
                    ),
                    "expected_workload_requests": 3,
                    "workload_unique_queries": 3,
                    "require_unique_workload_queries": True,
                },
                "query_error_summary": {"error_rows": 0},
                "execution_sources": execution_sources,
                "execution_lifecycle": {
                    "repeat_runtime_isolation": True,
                    "runtime_openings": 4,
                },
                "d2_graph_proof": {},
                "d2_graph_proof_final": {},
            }
        ),
        encoding="utf-8",
    )
    cell = {
        "pair_id": MATCHED_PAIR_ID,
        "dataset": "amazon",
        "target_recall": 0.9,
        "expected_rows": 12,
        "expected_requests": 3,
        "expected_repeats": 2,
        "stock_config": stock,
        "sqlens_config": sqlens,
        "mode_configs": mode_configs,
        "d3_fragment_store_namespace": base_namespace,
        "d3_repeat_namespaces": repeat_namespaces,
        "execution_sources": execution_sources,
        "input_bindings": {
            "measurement_workload_csv": {
                "path": str(workload),
                "sha256": file_sha(workload),
            },
            "truth_csv": {
                "path": str(truth),
                "sha256": file_sha(truth),
            },
            "filters_csv": {
                "path": str(filters),
                "sha256": file_sha(filters),
            },
            "d2_graph_proof_json": {
                "path": str(proof),
                "sha256": file_sha(proof),
                "canonical_json_sha256": (
                    repeats.matched_latency.frontier.sha256_json(
                        json.loads(proof.read_text(encoding="utf-8"))
                    )
                ),
            },
        },
        "raw": str(raw),
        "plan": str(plan),
        "raw_sha256": file_sha(raw),
        "plan_sha256": file_sha(plan),
        "status": "complete",
        "qualification_scope": (
            repeats.matched_latency.QUALIFICATION_SCOPE_FORMAL
        ),
        "predicate_completion": {
            "expected_predicate_count": 3,
            "observed_predicate_count": 3,
            "predicate_names": ["f0", "f1", "f2"],
            "exact_coverage": True,
        },
    }
    manifest: dict[str, object] = {
        "artifact_type": "sqlens_figure5_matched_latency_run",
        "status": "complete",
        "artifact_valid": True,
        "requested_slice_complete": True,
        "full_release_complete": True,
        "paper_eligible": True,
        "execution": {
            "execution_order": "paired_interleaved",
            "requests": 3,
            "repeats": 2,
            "expected_rows_per_pair": 12,
            "expected_predicate_count": 3,
        },
        "frontier_config": {
            "path": str(frontier_config),
            "sha256": file_sha(frontier_config),
        },
        "release_contract": release,
        "required_grid_contract": required_grid_binding,
        "selector": {
            "csv": str(selector_csv),
            "plan": str(selector_plan),
            "manifest": str(selector_manifest),
            "selection_csv_sha256": file_sha(selector_csv),
            "selection_plan_sha256": file_sha(selector_plan),
            "selection_manifest_sha256": file_sha(selector_manifest),
        },
        "schedule": [cell],
        "pairs_total": 1,
        "pairs_complete": 1,
    }
    manifest_path = tmp_path / "matched-run.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest, raw, plan


def publish_matched_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_convert_matched_manifest_uses_independent_arm_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_small_matched_contract(monkeypatch)
    manifest_path, manifest, _, _ = make_matched_manifest(tmp_path)

    rows = repeats.convert_manifest(manifest_path)

    assert len(rows) == 4
    assert {row["config_id"] for row in rows} == {MATCHED_PAIR_ID}
    assert {row["clients"] for row in rows} == {1}
    hashes = {row["arm_id"]: row["config_sha256"] for row in rows}
    assert hashes["stock_pgvector"] != hashes["sqlens_full"]
    assert len(set(hashes.values())) == 2
    cell = manifest["schedule"][0]
    settings = repeats._matched_search_settings(
        cell, cell["stock_config"], cell["sqlens_config"]
    )
    assert hashes == {
        arm: repeats.throughput.arm_config_sha256(settings, arm)
        for arm in ("stock_pgvector", "sqlens_full")
    }


@pytest.mark.parametrize(
    "partial_kind",
    ("manifest", "release_gate", "cell", "pair_count"),
)
def test_convert_matched_manifest_rejects_partial_or_incomplete_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial_kind: str,
) -> None:
    use_small_matched_contract(monkeypatch)
    manifest_path, manifest, _, _ = make_matched_manifest(tmp_path)
    if partial_kind == "manifest":
        manifest["status"] = "running"
    elif partial_kind == "release_gate":
        manifest["full_release_complete"] = False
        manifest["paper_eligible"] = False
    elif partial_kind == "cell":
        manifest["schedule"][0]["status"] = "pending"
    else:
        manifest["pairs_complete"] = 0
    publish_matched_manifest(manifest_path, manifest)

    with pytest.raises(repeats.LatencyRepeatError, match="incomplete"):
        repeats.convert_manifest(manifest_path)


def test_convert_matched_manifest_rejects_error_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_small_matched_contract(monkeypatch)
    manifest_path, manifest, raw, plan = make_matched_manifest(tmp_path)
    with raw.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    rows[0]["error"] = "query failed"
    write_csv(raw, rows)
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    plan_payload["output_sha256"] = file_sha(raw)
    plan.write_text(json.dumps(plan_payload), encoding="utf-8")
    cell = manifest["schedule"][0]
    cell["raw_sha256"] = file_sha(raw)
    cell["plan_sha256"] = file_sha(plan)
    publish_matched_manifest(manifest_path, manifest)

    with pytest.raises(repeats.LatencyRepeatError, match="completion gates"):
        repeats.convert_manifest(manifest_path)


@pytest.mark.parametrize("drifted", ("raw", "plan"))
def test_convert_matched_manifest_rejects_raw_or_plan_sha_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drifted: str,
) -> None:
    use_small_matched_contract(monkeypatch)
    manifest_path, _, raw, plan = make_matched_manifest(tmp_path)
    target = raw if drifted == "raw" else plan
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(repeats.LatencyRepeatError, match="SHA drifted"):
        repeats.convert_manifest(manifest_path)


def test_main_publishes_sha_bound_converter_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_small_matched_contract(monkeypatch)
    manifest_path, _, _, _ = make_matched_manifest(tmp_path)
    output = tmp_path / "latency-repeats.csv"

    assert repeats.main(
        [
            "--run-manifest",
            str(manifest_path),
            "--out",
            str(output),
        ]
    ) == 0

    binding_path = output.with_suffix(output.suffix + ".manifest.json")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["paper_eligible"] is True
    assert (
        binding["converter_binding"]["source_manifest"]["sha256"]
        == file_sha(manifest_path)
    )
    assert (
        binding["converter_binding"]["output"]["sha256"]
        == file_sha(output)
    )
    with output.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert {row["source_manifest_sha256"] for row in rows} == {
        file_sha(manifest_path)
    }
