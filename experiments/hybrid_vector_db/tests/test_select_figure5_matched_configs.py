from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import select_figure5_matched_configs as selector


BUILD = "sqlens-v16-d3-sticky-rejection-mixed-predicate-reuse-d2-edge-trace-readbuffer-profile-orderchangefix-ef500k-20260729-r36"
VECTOR_SHA = "d32dc122a35180f1e28617fcaf9f83ec8b639c8a497d0d8bf7d935950939d56a"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_contract(root: Path) -> Path:
    path = root / "p0_release_contract.json"
    path.write_text(json.dumps({
        "contract_id": "sigmod-p0-r36-20260729",
        "expected_sqlens_build_id": BUILD,
        "expected_vector_so_sha256": VECTOR_SHA,
    }), encoding="utf-8")
    return path


def test_release_contract_accepts_new_release_and_rejects_tag_drift(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "r36.json"
    contract.write_text(
        json.dumps(
            {
                "contract_id": "sigmod-p0-r36-20260729",
                "expected_sqlens_build_id": "sqlens-orderchangefix-r36",
                "expected_vector_so_sha256": VECTOR_SHA,
            }
        ),
        encoding="utf-8",
    )
    release = selector.load_release_contract(contract)
    assert release.contract_id == "sigmod-p0-r36-20260729"
    assert release.build_id == "sqlens-orderchangefix-r36"

    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["expected_sqlens_build_id"] = "sqlens-orderchangefix-r35"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="same explicit release tag"):
        selector.load_release_contract(contract)


def rows(
    mode: str,
    ef: int,
    recall: float,
    latency: float,
    *,
    request_count: int = selector.FORMAL_EXPECTED_REQUESTS,
) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for request in range(request_count):
        activation_ms = 2.0
        query_no = request // selector.EXPECTED_FILTERS
        values.append({
            "mode": mode,
            "error": "",
            "recall": recall,
            "end_to_end_ms": latency + request / 1000.0,
            "query_latency_ms": latency - activation_ms + request / 1000.0,
            "activation_ms": activation_ms,
            "request_no": request,
            "query_id": 1_000_000 + query_no,
            "filter_name": f"f{request % selector.EXPECTED_FILTERS:02d}",
            "ef_search": ef,
            "iterative_scan": "off" if mode == selector.MODE_SQLENS else "strict_order",
            "max_scan_tuples": 5_000_000,
            "scan_mem_multiplier": 32.0,
            "guided_collect_target": ef,
            "traversal_guided_target": min(ef, 40),
            "d2_page_access": "off",
            "d2_index_page_access": "off",
            "table": "public.items",
            "index": "public.bfs_idx" if mode == selector.MODE_SQLENS else "public.source_idx",
            "candidate_validity_predicate": "embedding_valid",
            "candidate_validity_predicate_sha256": "a" * 64,
            "self_exclusion_contract": "limit_k_plus_1_client_remove_query_id",
            "scan_limit": 11,
            "sqlens_build_id": BUILD,
            "vector_so_sha256": VECTOR_SHA,
            "d3_adaptive_page_builds_delta": 0,
            "d3_adaptive_bloom_builds_delta": 0,
            "d3_adaptive_exact_builds_delta": 0,
            "d3_adaptive_refinements_delta": 0,
            "d3_adaptive_rejections_delta": 0,
            "d3_fragment_builds_delta": 0,
        })
    return values


def test_formal_selection_latency_uses_observed_q2800_mean_for_sqlens() -> None:
    sample = [
        {
            "end_to_end_ms": "12",
            "query_latency_ms": "10",
            "activation_ms": "2",
            **{field: "0" for field in selector.FIXED_D3_EVENT_FIELDS},
        },
        {
            "end_to_end_ms": "112",
            "query_latency_ms": "10",
            "activation_ms": "102",
            **{
                field: ("1" if field == "d3_fragment_builds_delta" else "0")
                for field in selector.FIXED_D3_EVENT_FIELDS
            },
        },
        {
            "end_to_end_ms": "12",
            "query_latency_ms": "10",
            "activation_ms": "2",
            **{field: "0" for field in selector.FIXED_D3_EVENT_FIELDS},
        },
    ]

    score, metric, recurring, fixed_excess = selector._selection_latency(
        selector.MODE_SQLENS,
        sample,
        label="fixture",
    )

    assert recurring == pytest.approx(2.0)
    assert fixed_excess == pytest.approx(100.0)
    assert score == pytest.approx((12.0 + 112.0 + 12.0) / 3.0)
    assert metric == "observed_q2800_mean_end_to_end_ms"


def write_calibration(
    root: Path,
    *,
    dataset: str,
    family: str,
    ef: int,
    mode_rows: list[dict[str, object]],
    prewarm: bool = True,
    cap: int | None = None,
    traversal_target: int | None = None,
    artifact_tag: str = "r36",
) -> Path:
    workload = root / "workload.csv"
    truth = root / "truth.csv"
    filters = root / "filters.csv"
    core_source = root / "core_runner.py"
    orchestrator_source = root / "orchestrator.py"
    for path, content in (
        (truth, "truth\n"),
        (filters, "filters\n"),
        (core_source, "# core\n"),
        (orchestrator_source, "# orchestrator\n"),
    ):
        path.write_text(content, encoding="utf-8")
    d2_proof = {"required": False}
    if family == selector.FAMILY_BOTH_OFF:
        for row in mode_rows:
            if row["mode"] == selector.MODE_STOCK:
                row["iterative_scan"] = "off"
    if family == selector.FAMILY_SQLENS_CAP:
        if cap is None:
            raise ValueError("sqlens_cap fixture requires cap")
        for row in mode_rows:
            row["iterative_scan"] = "off"
            row["max_scan_tuples"] = cap
    if family == selector.FAMILY_SQLENS_TARGET:
        if traversal_target is None:
            raise ValueError("sqlens_target fixture requires target")
        for row in mode_rows:
            row["iterative_scan"] = "off"
            row["guided_collect_target"] = ef
            row["traversal_guided_target"] = traversal_target
    suffix = f"_cap{cap}" if cap is not None else ""
    suffix += (
        f"_target{traversal_target}"
        if traversal_target is not None
        else ""
    )
    path = root / (
        f"figure5_{artifact_tag}_{dataset}_calibration_{family}_ef{ef}{suffix}.csv"
    )
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(mode_rows[0]))
        writer.writeheader()
        writer.writerows(mode_rows)
    identity = {
        "exact_match": True,
        "expected_build_id": BUILD,
        "observed_build_id": BUILD,
        "expected_vector_so_sha256": VECTOR_SHA,
        "observed_vector_so_sha256": VECTOR_SHA,
    }
    records = [
        {"expected_blocks": 10, "warmed_blocks": 10},
        {"expected_blocks": 11, "warmed_blocks": 11},
        {"expected_blocks": 12, "warmed_blocks": 12},
    ]
    expected_modes = selector._family_modes(family)
    requests_per_mode = len(mode_rows) // len(expected_modes)
    if any(
        sum(row["mode"] == mode for row in mode_rows) != requests_per_mode
        for mode in expected_modes
    ):
        raise ValueError("fixture modes must have equal request counts")
    workload_rows: dict[int, tuple[object, object]] = {}
    for row in mode_rows:
        request_no = int(row["request_no"])
        signature = (row["query_id"], row["filter_name"])
        previous = workload_rows.setdefault(request_no, signature)
        if previous != signature:
            raise ValueError("fixture modes must share request identities")
    with workload.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=["request_no", "query_id", "filter_name"]
        )
        writer.writeheader()
        for request_no in sorted(workload_rows):
            query_id, filter_name = workload_rows[request_no]
            writer.writerow(
                {
                    "request_no": request_no,
                    "query_id": query_id,
                    "filter_name": filter_name,
                }
            )
    iterative = (
        "strict_order"
        if family in {
            selector.FAMILY_STOCK_STRICT,
            selector.FAMILY_STOCK_CAP,
        }
        else "off"
    )
    expected_warmups = (
        selector.EXPECTED_FILTERS
        if selector.MODE_STOCK in expected_modes
        else 0
    )
    plan = {
        "status": "complete",
        "output_sha256": sha(path),
        "output_rows": len(mode_rows),
        "d3_fragment_store_namespace": (
            f"fig5-r36-{dataset}-calibration-{family}-ef{ef}"
            + (f"-cap{cap}" if cap is not None else "")
            + (
                f"-target{traversal_target}"
                if traversal_target is not None
                else ""
            )
        ),
        "warmup_evidence": [
            {"filter_name": f"f{index:02d}", "status": "complete"}
            for index in range(expected_warmups)
        ],
        "execution_lifecycle": {
            "warmup_expected": expected_warmups,
            "warmup_observed": expected_warmups,
        },
        "checks": [
            {
                "mode": mode,
                "config": {
                    "ef_search": ef,
                    "iterative_scan": iterative,
                    **(
                        {
                            "guided_collect_target": ef,
                            "traversal_guided_target": traversal_target,
                        }
                        if traversal_target is not None
                        else {}
                    ),
                    **(
                        {"max_scan_tuples": cap}
                        if cap is not None
                        else {}
                    ),
                },
            }
            for mode in sorted(expected_modes)
        ],
        "relation_prewarm": {"enabled": True, "complete": prewarm, "records": records},
        "sqlens_runtime_identity_startup": identity,
        "sqlens_runtime_identity_final": identity,
        "runtime_sqlens_identity_evidence": [identity],
        "query_contract": {
            "query_table": f"public.{dataset}_items",
            "expected_workload_requests": requests_per_mode,
            "workload_requests": requests_per_mode,
            "workload_csv": str(workload),
            "workload_sha256": sha(workload),
            "truth_csv": str(truth),
            "truth_sha256": sha(truth),
            "filters_csv": str(filters),
            "filters_sha256": sha(filters),
            "d2_graph_proof_input_sha256": selector.sha256_json(d2_proof),
        },
        "d2_graph_proof_input": d2_proof,
        "execution_sources": {
            "core_runner": {
                "path": str(core_source),
                "sha256": sha(core_source),
            },
            "orchestrator": {
                "path": str(orchestrator_source),
                "sha256": sha(orchestrator_source),
            },
        },
    }
    Path(str(path) + ".plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return path


def write_required_grid_contract(
    root: Path,
    release_contract: Path,
    raws: list[Path],
    targets: tuple[float, ...],
    *,
    parallel_db_cells: bool = False,
    cell_raw_sha_override: str | None = None,
    include_normalized_manifest_fields: bool = True,
    manifest_cell_key_override: str | None = None,
    manifest_plan_sha_override: str | None = None,
) -> Path:
    runner_manifest = root / "serial_calibration_manifest.json"
    schedule = []
    isolation = {
        "parallel_db_cells": parallel_db_cells,
        "lock_required": True,
        "lock_acquired": True,
        "lock_path": str((root / "formal.lock").resolve()),
        "lock_protocol": "fcntl_flock_exclusive_nonblocking_v1",
        "lock_owner_token": "fixture-owner-token",
        "held_through_completion": True,
    }
    for index, raw in enumerate(raws):
        plan = Path(str(raw) + ".plan.json")
        cell = {
            "status": "complete",
            "raw": str(raw.resolve()),
            "raw_sha256": (
                cell_raw_sha_override
                if index == 0 and cell_raw_sha_override is not None
                else sha(raw)
            ),
            "plan": str(plan.resolve()),
            "database_isolation": isolation,
        }
        if include_normalized_manifest_fields:
            cell["cell_key"] = (
                manifest_cell_key_override
                if index == 0 and manifest_cell_key_override is not None
                else selector.calibration_cell_key(raw)
            )
            cell["plan_sha256"] = (
                manifest_plan_sha_override
                if index == 0 and manifest_plan_sha_override is not None
                else sha(plan)
            )
        schedule.append(cell)
    dataset_bindings: dict[str, dict[str, str]] = {}
    for raw in raws:
        dataset = selector._filename_metadata(raw).group("dataset")
        plan = json.loads(
            Path(str(raw) + ".plan.json").read_text(encoding="utf-8")
        )
        query_contract = plan["query_contract"]
        dataset_bindings.setdefault(
            dataset,
            {
                "calibration_workload_csv": query_contract["workload_csv"],
                "truth_csv": query_contract["truth_csv"],
                "filters_csv": query_contract["filters_csv"],
            },
        )
    dataset_config = root / "formal_datasets.json"
    dataset_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_contract": str(release_contract.resolve()),
                "protocol": {
                    "qualification_scope": selector.QUALIFICATION_SCOPE_FORMAL,
                    "calibration_requests": selector.FORMAL_EXPECTED_REQUESTS,
                    "calibration_observations_per_predicate": (
                        selector.FORMAL_OBSERVATIONS_PER_FILTER
                    ),
                },
                "datasets": dataset_bindings,
            }
        ),
        encoding="utf-8",
    )
    source_grid_plan = root / "isolated_grid_plan.json"
    source_grid_plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plan_id": "selector-test-isolated-grid",
                "release_contract": str(release_contract.resolve()),
                "dataset_config": str(dataset_config.resolve()),
                "qualification_scope": selector.QUALIFICATION_SCOPE_FORMAL,
                "targets": list(targets),
            }
        ),
        encoding="utf-8",
    )
    runner_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "requested_slice_complete": True,
                "release_contract": {
                    "path": str(release_contract.resolve()),
                    "sha256": sha(release_contract),
                },
                "source_grid_plan": {
                    "path": str(source_grid_plan.resolve()),
                    "sha256": sha(source_grid_plan),
                    "plan_id": "selector-test-isolated-grid",
                },
                "dataset_config": {
                    "path": str(dataset_config.resolve()),
                    "sha256": sha(dataset_config),
                },
                "database_isolation": isolation,
                "schedule": schedule,
            }
        ),
        encoding="utf-8",
    )
    contract = root / "required_grid_contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract_type": selector.REQUIRED_GRID_CONTRACT_TYPE,
                "grid_complete": True,
                "release_contract": {
                    "path": str(release_contract.resolve()),
                    "sha256": sha(release_contract),
                },
                "source_grid_plan": {
                    "path": str(source_grid_plan.resolve()),
                    "sha256": sha(source_grid_plan),
                    "plan_id": "selector-test-isolated-grid",
                },
                "dataset_config": {
                    "path": str(dataset_config.resolve()),
                    "sha256": sha(dataset_config),
                },
                "qualification_scope": selector.QUALIFICATION_SCOPE_FORMAL,
                "targets": list(targets),
                "cells": [
                    {
                        "cell_key": selector.calibration_cell_key(raw),
                        "raw_csv": {
                            "path": str(raw.resolve()),
                            "sha256": sha(raw),
                        },
                        "input_plan": {
                            "path": str(Path(str(raw) + ".plan.json").resolve()),
                            "sha256": sha(Path(str(raw) + ".plan.json")),
                        },
                        "serial_runner_manifest": {
                            "path": str(runner_manifest.resolve()),
                            "sha256": sha(runner_manifest),
                        },
                    }
                    for raw in raws
                ],
            }
        ),
        encoding="utf-8",
    )
    return contract


def load_required_grid(
    contract: Path,
    release: selector.ReleaseContract,
    targets: tuple[float, ...],
    raws: list[Path],
) -> selector.RequiredGridEvidence:
    return selector.load_required_grid_contract(
        contract,
        release,
        selector.QUALIFICATION_SCOPE_FORMAL,
        targets,
        raws,
    )


def test_sqlens_target_family_binds_filename_plan_and_config(
    tmp_path: Path,
) -> None:
    contract = write_contract(tmp_path)
    release = selector.load_release_contract(contract)
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_SQLENS_TARGET,
        ef=150,
        traversal_target=11,
        mode_rows=rows(selector.MODE_SQLENS, 150, 0.90, 12.0),
    )

    configs = selector.load_calibration_csv(
        raw,
        release,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert len(configs) == 1
    assert configs[0].config_id == "sqlens_target_ef150_target11"
    assert configs[0].config["guided_collect_target"] == 150
    assert configs[0].config["traversal_guided_target"] == 11


def test_sqlens_target_rejects_plan_target_drift(tmp_path: Path) -> None:
    contract = write_contract(tmp_path)
    release = selector.load_release_contract(contract)
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_SQLENS_TARGET,
        ef=150,
        traversal_target=11,
        mode_rows=rows(selector.MODE_SQLENS, 150, 0.90, 12.0),
    )
    plan_path = Path(str(raw) + ".plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["checks"][0]["config"]["traversal_guided_target"] = 20
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(selector.SelectionError, match="target disagrees"):
        selector.load_calibration_csv(
            raw,
            release,
            bootstrap_samples=100,
            bootstrap_seed=7,
        )


def test_legacy_artifact_tag_uses_current_release_namespace(
    tmp_path: Path,
) -> None:
    contract = write_contract(tmp_path)
    path = write_calibration(
        tmp_path,
        dataset="amazon",
        family="both_off",
        ef=100,
        mode_rows=(
            rows(selector.MODE_STOCK, 100, 0.72, 10)
            + rows(selector.MODE_SQLENS, 100, 0.82, 19)
        ),
        artifact_tag="r35",
    )
    release = selector.load_release_contract(contract)

    configs = selector.load_calibration_csv(
        path,
        release,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    assert {config.artifact_tag for config in configs} == {"r35"}

    plan_path = Path(str(path) + ".plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["d3_fragment_store_namespace"] = (
        "fig5-r35-amazon-calibration-both_off-ef100"
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="namespace release tag"):
        selector.load_calibration_csv(
            path,
            release,
            bootstrap_samples=100,
            bootstrap_seed=7,
        )


def test_stock_only_calibration_does_not_require_d3_delta_columns(
    tmp_path: Path,
) -> None:
    contract = selector.load_release_contract(write_contract(tmp_path))
    stock_rows = rows(selector.MODE_STOCK, 100, 0.72, 10)
    for row in stock_rows:
        for field in selector.FIXED_D3_EVENT_FIELDS:
            row.pop(field)
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_STOCK_STRICT,
        ef=100,
        mode_rows=stock_rows,
    )

    configs = selector.load_calibration_csv(
        raw,
        contract,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert len(configs) == 1
    assert configs[0].mode == selector.MODE_STOCK


def test_selects_independent_lowest_latency_configs_and_publishes_atomically(tmp_path: Path) -> None:
    contract = write_contract(tmp_path)
    files = [
        write_calibration(tmp_path, dataset="amazon", family="both_off", ef=100, mode_rows=rows(selector.MODE_STOCK, 100, 0.72, 10) + rows(selector.MODE_SQLENS, 100, 0.82, 19)),
        write_calibration(tmp_path, dataset="amazon", family="both_off", ef=200, mode_rows=rows(selector.MODE_STOCK, 200, 0.84, 23) + rows(selector.MODE_SQLENS, 200, 0.93, 11)),
        write_calibration(tmp_path, dataset="amazon", family="both_off", ef=300, mode_rows=rows(selector.MODE_STOCK, 300, 0.96, 40) + rows(selector.MODE_SQLENS, 300, 0.98, 29)),
        write_calibration(tmp_path, dataset="amazon", family="stock_strict", ef=150, mode_rows=rows(selector.MODE_STOCK, 150, 0.91, 14)),
    ]
    release = selector.load_release_contract(contract)
    configs = [config for path in files for config in selector.load_calibration_csv(path, release, bootstrap_samples=100, bootstrap_seed=7)]
    targets = (0.70, 0.85, 0.99)
    grid_contract = write_required_grid_contract(
        tmp_path, contract, files, targets
    )
    required_grid = load_required_grid(
        grid_contract, release, targets, files
    )
    result, plan = selector.build_measurement_plan(
        configs,
        release,
        targets,
        bootstrap_samples=100,
        bootstrap_seed=7,
        required_grid=required_grid,
    )

    assert len(result) == 3
    low, high, unattainable = result
    assert low["selection_status"] == "selected"
    assert low["stock_config_id"] == "both_off_ef100"
    assert low["sqlens_config_id"] == "both_off_ef200"
    assert high["stock_config_id"] == "stock_strict_ef150"
    assert high["sqlens_config_id"] == "both_off_ef200"
    assert unattainable["selection_status"] == "unattainable_on_calibration_grid"
    assert unattainable["stock_config_id"] == ""
    assert plan["summary"]["selected_pairs"] == 2
    assert plan["qualification_scope"] == selector.QUALIFICATION_SCOPE_FORMAL
    assert plan["qualification_metric"] == (
        "bootstrap_aggregate_recall_ci95_low_and_"
        "bootstrap_min_per_filter_recall_ci95_low"
    )
    assert all(
        row["qualification_scope"] == selector.QUALIFICATION_SCOPE_FORMAL
        and ":scope_global_min_predicate_lcb:" in row["pair_id"]
        for row in result
    )

    prefix = tmp_path / "matched"
    paths = selector.publish_atomically(prefix, result, plan)
    with paths["csv"].open(newline="", encoding="utf-8") as source:
        published_rows = list(csv.DictReader(source))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert {row["qualification_scope"] for row in published_rows} == {
        selector.QUALIFICATION_SCOPE_FORMAL
    }
    assert manifest["outputs"]["measurement_plan_csv"]["sha256"] == sha(paths["csv"])
    assert manifest["outputs"]["measurement_plan_json"]["sha256"] == sha(paths["plan"])
    assert manifest["qualification_scope"] == selector.QUALIFICATION_SCOPE_FORMAL
    assert manifest["selection_provenance"]["measurement_pair_scope_bound"] is True
    assert (
        manifest["required_grid_contract"]["sha256"]
        == sha(grid_contract)
    )
    assert manifest["exhaustion_proof"]["unattainable_arms"]


def test_formal_scope_rejects_an_aggregate_only_recall_pass(tmp_path: Path) -> None:
    release_contract = write_contract(tmp_path)
    contract = selector.load_release_contract(release_contract)
    stock_rows = rows(selector.MODE_STOCK, 100, 0.96, 10)
    sqlens_rows = rows(selector.MODE_SQLENS, 100, 0.96, 8)
    for row in (*stock_rows, *sqlens_rows):
        if row["filter_name"] == "f00":
            row["recall"] = 0.40
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family="both_off",
        ef=100,
        mode_rows=stock_rows + sqlens_rows,
    )

    configs = selector.load_calibration_csv(
        raw,
        contract,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    stock = [item for item in configs if item.mode == selector.MODE_STOCK]
    assert stock[0].recall_ci95_low > 0.80
    assert stock[0].per_filter_recall_min_ci95_low < 0.80
    assert selector.select_config(stock, 0.80) is None
    assert selector.select_config(
        stock,
        0.80,
        selector.QUALIFICATION_SCOPE_AGGREGATE,
    ) is stock[0]

    targets = (0.80,)
    grid_contract = write_required_grid_contract(
        tmp_path, release_contract, [raw], targets
    )
    required_grid = load_required_grid(
        grid_contract, contract, targets, [raw]
    )
    formal_rows, formal_plan = selector.build_measurement_plan(
        configs,
        contract,
        targets,
        bootstrap_samples=100,
        bootstrap_seed=7,
        required_grid=required_grid,
    )
    assert formal_rows[0]["selection_status"] == "unattainable_on_calibration_grid"
    assert formal_plan["qualification_scope"] == selector.QUALIFICATION_SCOPE_FORMAL

    legacy_rows, legacy_plan = selector.build_measurement_plan(
        configs,
        contract,
        (0.80,),
        bootstrap_samples=100,
        bootstrap_seed=7,
        qualification_scope=selector.QUALIFICATION_SCOPE_AGGREGATE,
    )
    assert legacy_rows[0]["selection_status"] == "selected"
    assert legacy_rows[0]["qualification_scope"] == selector.QUALIFICATION_SCOPE_AGGREGATE
    assert ":scope_aggregate_lcb:" in str(legacy_rows[0]["pair_id"])
    assert legacy_plan["qualification_metric"] == "bootstrap_aggregate_recall_ci95_low"


def test_legacy_scope_explicitly_admits_old_q200_calibration_only(tmp_path: Path) -> None:
    contract = selector.load_release_contract(write_contract(tmp_path))
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family="both_off",
        ef=100,
        mode_rows=(
            rows(
                selector.MODE_STOCK,
                100,
                0.90,
                10,
                request_count=selector.LEGACY_EXPECTED_REQUESTS,
            )
            + rows(
                selector.MODE_SQLENS,
                100,
                0.90,
                8,
                request_count=selector.LEGACY_EXPECTED_REQUESTS,
            )
        ),
    )

    with pytest.raises(selector.SelectionError, match="2800.*global_min_predicate_lcb"):
        selector.load_calibration_csv(
            raw,
            contract,
            bootstrap_samples=100,
            bootstrap_seed=7,
        )

    configs = selector.load_calibration_csv(
        raw,
        contract,
        bootstrap_samples=100,
        bootstrap_seed=7,
        qualification_scope=selector.QUALIFICATION_SCOPE_AGGREGATE,
    )
    rows_out, plan = selector.build_measurement_plan(
        configs,
        contract,
        (0.80,),
        bootstrap_samples=100,
        bootstrap_seed=7,
        qualification_scope=selector.QUALIFICATION_SCOPE_AGGREGATE,
    )
    assert rows_out[0]["selection_status"] == "selected"
    assert plan["calibration_coverage_contract"] == {
        "requests_per_mode": 200,
        "filters": 14,
        "observations_per_filter": None,
    }


def test_loads_sqlens_only_scan_cap_configuration(tmp_path: Path) -> None:
    contract = selector.load_release_contract(write_contract(tmp_path))
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_SQLENS_CAP,
        ef=11,
        cap=1000,
        mode_rows=rows(selector.MODE_SQLENS, 11, 0.75, 20),
    )

    configs = selector.load_calibration_csv(
        raw,
        contract,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert len(configs) == 1
    assert configs[0].mode == selector.MODE_SQLENS
    assert configs[0].config_id == "sqlens_cap_ef11_cap1000"
    assert configs[0].config["max_scan_tuples"] == 1000


def test_loads_stock_only_scan_cap_configuration(tmp_path: Path) -> None:
    contract = selector.load_release_contract(write_contract(tmp_path))
    stock_rows = rows(selector.MODE_STOCK, 11, 0.86, 15)
    for row in stock_rows:
        row["iterative_scan"] = "strict_order"
        row["max_scan_tuples"] = 20_000
        for field in selector.FIXED_D3_EVENT_FIELDS:
            row.pop(field)
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_STOCK_CAP,
        ef=11,
        cap=20_000,
        mode_rows=stock_rows,
    )

    configs = selector.load_calibration_csv(
        raw,
        contract,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert len(configs) == 1
    assert configs[0].mode == selector.MODE_STOCK
    assert configs[0].config_id == "stock_cap_ef11_cap20000"
    assert configs[0].config["iterative_scan"] == "strict_order"
    assert configs[0].config["max_scan_tuples"] == 20_000


def test_rejects_incomplete_prewarm_and_wrong_runtime_identity(tmp_path: Path) -> None:
    contract = selector.load_release_contract(write_contract(tmp_path))
    raw = write_calibration(
        tmp_path, dataset="amazon", family="both_off", ef=100,
        mode_rows=rows(selector.MODE_STOCK, 100, 0.9, 10) + rows(selector.MODE_SQLENS, 100, 0.9, 8),
        prewarm=False,
    )
    with pytest.raises(selector.SelectionError, match="prewarm"):
        selector.load_calibration_csv(raw, contract, bootstrap_samples=100, bootstrap_seed=1)

    plan_path = Path(str(raw) + ".plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["relation_prewarm"]["complete"] = True
    plan["runtime_sqlens_identity_evidence"][0]["observed_build_id"] = "wrong-r35"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="runtime release evidence"):
        selector.load_calibration_csv(raw, contract, bootstrap_samples=100, bootstrap_seed=1)


def test_dry_run_validates_without_publishing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    contract = write_contract(tmp_path)
    raw = write_calibration(
        tmp_path, dataset="amazon", family="both_off", ef=100,
        mode_rows=rows(selector.MODE_STOCK, 100, 0.9, 10) + rows(selector.MODE_SQLENS, 100, 0.9, 8),
    )
    prefix = tmp_path / "never-published"
    grid_contract = write_required_grid_contract(
        tmp_path, contract, [raw], (0.70, 0.99)
    )
    assert selector.main([
        "--input", str(raw), "--contract", str(contract), "--out-prefix", str(prefix),
        "--targets", "0.70,0.99", "--bootstrap-samples", "100",
        "--required-grid-contract", str(grid_contract),
    ]) == 0
    assert not Path(str(prefix) + ".csv").exists()
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"


def test_required_grid_rejects_missing_and_extra_inputs(tmp_path: Path) -> None:
    release_contract = write_contract(tmp_path)
    release = selector.load_release_contract(release_contract)
    first = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_STOCK_STRICT,
        ef=100,
        mode_rows=rows(selector.MODE_STOCK, 100, 0.90, 10),
    )
    second = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_STOCK_STRICT,
        ef=200,
        mode_rows=rows(selector.MODE_STOCK, 200, 0.95, 20),
    )
    targets = (0.90, 0.95, 0.99)
    complete = write_required_grid_contract(
        tmp_path, release_contract, [first, second], targets
    )
    with pytest.raises(selector.SelectionError, match=r"missing=.*ef200"):
        load_required_grid(complete, release, targets, [first])

    partial = write_required_grid_contract(
        tmp_path, release_contract, [first], targets
    )
    with pytest.raises(selector.SelectionError, match=r"extra=.*ef200"):
        load_required_grid(partial, release, targets, [first, second])


def test_required_grid_rejects_parallel_database_manifest(tmp_path: Path) -> None:
    release_contract = write_contract(tmp_path)
    release = selector.load_release_contract(release_contract)
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family="both_off",
        ef=100,
        mode_rows=(
            rows(selector.MODE_STOCK, 100, 0.90, 10)
            + rows(selector.MODE_SQLENS, 100, 0.90, 8)
        ),
    )
    targets = (0.90,)
    grid = write_required_grid_contract(
        tmp_path,
        release_contract,
        [raw],
        targets,
        parallel_db_cells=True,
    )

    with pytest.raises(
        selector.SelectionError,
        match="parallel_db_cells=false",
    ):
        load_required_grid(grid, release, targets, [raw])


def test_required_grid_rejects_wrong_manifest_cell_sha(tmp_path: Path) -> None:
    release_contract = write_contract(tmp_path)
    release = selector.load_release_contract(release_contract)
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family="both_off",
        ef=100,
        mode_rows=(
            rows(selector.MODE_STOCK, 100, 0.90, 10)
            + rows(selector.MODE_SQLENS, 100, 0.90, 8)
        ),
    )
    targets = (0.90,)
    grid = write_required_grid_contract(
        tmp_path,
        release_contract,
        [raw],
        targets,
        cell_raw_sha_override="0" * 64,
    )

    with pytest.raises(selector.SelectionError, match="cell status/path/SHA"):
        load_required_grid(grid, release, targets, [raw])


def test_required_grid_accepts_runner_manifest_without_normalized_fields(
    tmp_path: Path,
) -> None:
    release_contract = write_contract(tmp_path)
    release = selector.load_release_contract(release_contract)
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_STOCK_STRICT,
        ef=100,
        mode_rows=rows(selector.MODE_STOCK, 100, 0.90, 10),
    )
    grid = write_required_grid_contract(
        tmp_path,
        release_contract,
        [raw],
        (0.90,),
        include_normalized_manifest_fields=False,
    )

    evidence = load_required_grid(grid, release, (0.90,), [raw])

    assert evidence.cells[0]["cell_key"] == selector.calibration_cell_key(raw)
    assert evidence.cells[0]["plan_sha256"] == sha(
        Path(str(raw) + ".plan.json")
    )


@pytest.mark.parametrize(
    ("override", "error"),
    (
        ({"manifest_cell_key_override": "amazon:wrong"}, "cell_key mismatch"),
        ({"manifest_plan_sha_override": "0" * 64}, "plan SHA mismatch"),
    ),
)
def test_required_grid_rejects_present_normalized_manifest_field_drift(
    tmp_path: Path,
    override: dict[str, str],
    error: str,
) -> None:
    release_contract = write_contract(tmp_path)
    release = selector.load_release_contract(release_contract)
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_STOCK_STRICT,
        ef=100,
        mode_rows=rows(selector.MODE_STOCK, 100, 0.90, 10),
    )
    grid = write_required_grid_contract(
        tmp_path,
        release_contract,
        [raw],
        (0.90,),
        **override,
    )

    with pytest.raises(selector.SelectionError, match=error):
        load_required_grid(grid, release, (0.90,), [raw])


@pytest.mark.parametrize("field", ("query_id", "filter_name"))
def test_calibration_rows_must_match_bound_workload_request_mapping(
    tmp_path: Path,
    field: str,
) -> None:
    release = selector.load_release_contract(write_contract(tmp_path))
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_STOCK_STRICT,
        ef=100,
        mode_rows=rows(selector.MODE_STOCK, 100, 0.90, 10),
    )
    with raw.open(newline="", encoding="utf-8") as source:
        records = list(csv.DictReader(source))
        fieldnames = list(records[0])
    records[0][field] = (
        "999999999" if field == "query_id" else "wrong_filter"
    )
    with raw.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    plan_path = Path(str(raw) + ".plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["output_sha256"] = sha(raw)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(
        selector.SelectionError,
        match="does not match bound workload request_no mapping",
    ):
        selector.load_calibration_csv(
            raw,
            release,
            bootstrap_samples=100,
            bootstrap_seed=7,
        )


def test_measurement_plan_rejects_cross_cell_input_binding_drift(
    tmp_path: Path,
) -> None:
    release_contract = write_contract(tmp_path)
    release = selector.load_release_contract(release_contract)
    stock_raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_STOCK_STRICT,
        ef=100,
        mode_rows=rows(selector.MODE_STOCK, 100, 0.95, 10),
    )
    sqlens_raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_SQLENS_TARGET,
        ef=100,
        traversal_target=20,
        mode_rows=rows(selector.MODE_SQLENS, 100, 0.95, 8),
    )
    alternate_truth = tmp_path / "alternate_truth.csv"
    alternate_truth.write_text("different truth\n", encoding="utf-8")
    sqlens_plan_path = Path(str(sqlens_raw) + ".plan.json")
    sqlens_plan = json.loads(sqlens_plan_path.read_text(encoding="utf-8"))
    sqlens_plan["query_contract"]["truth_csv"] = str(alternate_truth)
    sqlens_plan["query_contract"]["truth_sha256"] = sha(alternate_truth)
    sqlens_plan_path.write_text(json.dumps(sqlens_plan), encoding="utf-8")
    raws = [stock_raw, sqlens_raw]
    grid_path = write_required_grid_contract(
        tmp_path, release_contract, raws, (0.90,)
    )
    required_grid = load_required_grid(
        grid_path, release, (0.90,), raws
    )
    configs = [
        config
        for raw in raws
        for config in selector.load_calibration_csv(
            raw,
            release,
            bootstrap_samples=100,
            bootstrap_seed=7,
        )
    ]

    with pytest.raises(
        selector.SelectionError,
        match="do not share workload/truth/filter bindings",
    ):
        selector.build_measurement_plan(
            configs,
            release,
            (0.90,),
            bootstrap_samples=100,
            bootstrap_seed=7,
            required_grid=required_grid,
        )


@pytest.mark.parametrize(
    "binding_name", ("source_grid_plan", "dataset_config")
)
def test_required_grid_rejects_changed_source_config_or_plan(
    tmp_path: Path,
    binding_name: str,
) -> None:
    release_contract = write_contract(tmp_path)
    release = selector.load_release_contract(release_contract)
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_STOCK_STRICT,
        ef=100,
        mode_rows=rows(selector.MODE_STOCK, 100, 0.90, 10),
    )
    grid = write_required_grid_contract(
        tmp_path, release_contract, [raw], (0.90,)
    )
    payload = json.loads(grid.read_text(encoding="utf-8"))
    bound_path = Path(payload[binding_name]["path"])
    bound_path.write_text(
        bound_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(selector.SelectionError, match="path/SHA mismatch"):
        load_required_grid(grid, release, (0.90,), [raw])


def test_required_grid_rejects_semantic_source_plan_config_drift(
    tmp_path: Path,
) -> None:
    release_contract = write_contract(tmp_path)
    release = selector.load_release_contract(release_contract)
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family=selector.FAMILY_STOCK_STRICT,
        ef=100,
        mode_rows=rows(selector.MODE_STOCK, 100, 0.90, 10),
    )
    grid = write_required_grid_contract(
        tmp_path, release_contract, [raw], (0.90,)
    )
    payload = json.loads(grid.read_text(encoding="utf-8"))
    source_path = Path(payload["source_grid_plan"]["path"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["dataset_config"] = str(tmp_path / "wrong_config.json")
    source_path.write_text(json.dumps(source), encoding="utf-8")
    payload["source_grid_plan"]["sha256"] = sha(source_path)
    grid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        selector.SelectionError, match="source grid plan binding mismatch"
    ):
        load_required_grid(grid, release, (0.90,), [raw])


def test_calibration_bootstrap_resamples_whole_query_clusters() -> None:
    recalls: list[float] = []
    filters: list[str] = []
    query_ids: list[str] = []
    for query_no in range(selector.FORMAL_OBSERVATIONS_PER_FILTER):
        query_recall = 0.0 if query_no < 100 else 1.0
        for filter_no in range(selector.EXPECTED_FILTERS):
            recalls.append(query_recall)
            filters.append(f"f{filter_no:02d}")
            query_ids.append(f"q{query_no:03d}")

    mean, low, _, per_filter_min, per_filter_low = (
        selector._bootstrap_metrics(
            recalls,
            filters,
            query_ids,
            samples=1000,
            seed=17,
            require_formal_cartesian=True,
        )
    )

    assert mean == pytest.approx(0.5)
    assert per_filter_min == pytest.approx(0.5)
    assert per_filter_low == pytest.approx(low)


def test_complete_required_grid_proves_unattainable_target(
    tmp_path: Path,
) -> None:
    release_contract = write_contract(tmp_path)
    release = selector.load_release_contract(release_contract)
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family="both_off",
        ef=100,
        mode_rows=(
            rows(selector.MODE_STOCK, 100, 0.96, 10)
            + rows(selector.MODE_SQLENS, 100, 0.97, 8)
        ),
    )
    targets = (0.99,)
    grid_contract = write_required_grid_contract(
        tmp_path, release_contract, [raw], targets
    )
    required_grid = load_required_grid(
        grid_contract, release, targets, [raw]
    )
    configs = selector.load_calibration_csv(
        raw,
        release,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    selected, plan = selector.build_measurement_plan(
        configs,
        release,
        targets,
        bootstrap_samples=100,
        bootstrap_seed=7,
        required_grid=required_grid,
    )

    assert selected[0]["selection_status"] == (
        "unattainable_on_calibration_grid"
    )
    proof = plan["exhaustion_proof"]
    assert proof["required_grid_complete"] is True
    assert proof["input_set_exact"] is True
    assert proof["required_grid_contract_sha256"] == sha(grid_contract)
    assert proof["required_grid_cell_keys_sha256"] == selector.sha256_json(
        list(required_grid.cell_keys)
    )
    assert {
        item["arm"] for item in proof["unattainable_arms"]
    } == {"stock", "sqlens"}
    proof_body = dict(proof)
    proof_sha = proof_body.pop("proof_sha256")
    assert proof_sha == selector.sha256_json(proof_body)


def test_formal_fixed_target_publication_requires_required_grid(
    tmp_path: Path,
) -> None:
    release = selector.load_release_contract(write_contract(tmp_path))
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family="both_off",
        ef=100,
        mode_rows=(
            rows(selector.MODE_STOCK, 100, 0.95, 10)
            + rows(selector.MODE_SQLENS, 100, 0.95, 8)
        ),
    )
    configs = selector.load_calibration_csv(
        raw,
        release,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    result, plan = selector.build_measurement_plan(
        configs,
        release,
        (0.90,),
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    with pytest.raises(
        selector.SelectionError,
        match="publication requires a complete required-grid",
    ):
        selector.publish_atomically(tmp_path / "forbidden", result, plan)


def test_formal_selection_metric_is_identical_for_stock_and_sqlens() -> None:
    stock_rows = [
        {
            "end_to_end_ms": value,
            "activation_ms": "2",
        }
        for value in ("10", "20", "30")
    ]
    sqlens_rows = [
        {
            "end_to_end_ms": value,
            "activation_ms": "2",
            **{field: "0" for field in selector.FIXED_D3_EVENT_FIELDS},
        }
        for value in ("10", "20", "30")
    ]

    stock = selector._selection_latency(
        selector.MODE_STOCK, stock_rows, label="stock"
    )
    sqlens = selector._selection_latency(
        selector.MODE_SQLENS, sqlens_rows, label="sqlens"
    )

    assert stock[0] == sqlens[0] == pytest.approx(20.0)
    assert stock[1] == sqlens[1] == (
        "observed_q2800_mean_end_to_end_ms"
    )


def test_rejects_any_request_level_runtime_error(tmp_path: Path) -> None:
    contract = selector.load_release_contract(write_contract(tmp_path))
    records = rows(selector.MODE_STOCK, 100, 0.9, 10) + rows(selector.MODE_SQLENS, 100, 0.9, 8)
    records[-1]["error"] = "RuntimeError"
    raw = write_calibration(
        tmp_path, dataset="amazon", family="both_off", ef=100, mode_rows=records,
    )
    with pytest.raises(selector.SelectionError, match="reports an error"):
        selector.load_calibration_csv(raw, contract, bootstrap_samples=100, bootstrap_seed=1)


def test_directory_discovery_ignores_derived_calibration_csvs(tmp_path: Path) -> None:
    raw = tmp_path / "figure5_r36_amazon_calibration_both_off_ef100.csv"
    raw.touch()
    (tmp_path / "figure5_r36_amazon_calibration_both_off_ef100_table.csv").touch()
    (tmp_path / "figure5_r36_amazon_calibration_both_off_ef100_profile_summary.csv").touch()

    assert selector._discover_csvs(tmp_path, ()) == [raw.resolve()]


def test_directory_discovery_includes_extension_directories(tmp_path: Path) -> None:
    base = tmp_path / "base"
    extension = tmp_path / "extension"
    base.mkdir()
    extension.mkdir()
    first = base / "figure5_r36_amazon_calibration_both_off_ef100.csv"
    second = extension / "figure5_r36_laion_calibration_stock_strict_ef20000.csv"
    first.touch()
    second.touch()

    assert selector._discover_csvs(base, (), (extension,)) == [
        first.resolve(),
        second.resolve(),
    ]


def test_explicit_inputs_reject_extension_directories(tmp_path: Path) -> None:
    raw = tmp_path / "figure5_r36_amazon_calibration_both_off_ef100.csv"
    raw.touch()

    with pytest.raises(selector.SelectionError, match="cannot be combined"):
        selector._discover_csvs(tmp_path, (raw,), (tmp_path,))


def test_explicit_derived_calibration_csv_remains_fail_closed(tmp_path: Path) -> None:
    derived = tmp_path / "figure5_r36_amazon_calibration_both_off_ef100_table.csv"
    derived.touch()

    with pytest.raises(selector.SelectionError, match="violates the figure5_rNN contract"):
        selector._discover_csvs(tmp_path, (derived,))


def test_discovery_rejects_mixed_artifact_prefixes(tmp_path: Path) -> None:
    r36 = tmp_path / "figure5_r36_amazon_calibration_both_off_ef100.csv"
    r35 = tmp_path / "figure5_r35_amazon_calibration_both_off_ef200.csv"
    r36.touch()
    r35.touch()

    with pytest.raises(selector.SelectionError, match="mixed calibration artifact prefixes"):
        selector._discover_csvs(tmp_path, (r36, r35))


def test_plan_filename_rename_is_rejected_against_actual_plan_metadata(tmp_path: Path) -> None:
    contract = selector.load_release_contract(write_contract(tmp_path))
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family="both_off",
        ef=100,
        mode_rows=rows(selector.MODE_STOCK, 100, 0.9, 10) + rows(selector.MODE_SQLENS, 100, 0.9, 8),
    )
    renamed = tmp_path / "figure5_r36_yfcc_calibration_both_off_ef100.csv"
    renamed_plan = Path(str(renamed) + ".plan.json")
    raw_plan = Path(str(raw) + ".plan.json")
    raw.rename(renamed)
    raw_plan.rename(renamed_plan)

    with pytest.raises(selector.SelectionError, match="plan dataset disagrees"):
        selector.load_calibration_csv(renamed, contract, bootstrap_samples=100, bootstrap_seed=1)


def test_plan_request_count_drift_is_rejected(tmp_path: Path) -> None:
    contract = selector.load_release_contract(write_contract(tmp_path))
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family="both_off",
        ef=100,
        mode_rows=rows(selector.MODE_STOCK, 100, 0.9, 10) + rows(selector.MODE_SQLENS, 100, 0.9, 8),
    )
    plan_path = Path(str(raw) + ".plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["query_contract"]["workload_requests"] = selector.FORMAL_EXPECTED_REQUESTS - 1
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(selector.SelectionError, match="workload_requests"):
        selector.load_calibration_csv(raw, contract, bootstrap_samples=100, bootstrap_seed=1)


def test_publish_lock_rejects_a_concurrent_selector(tmp_path: Path) -> None:
    prefix = tmp_path / "matched"
    with selector.acquire_publish_lock(prefix):
        with pytest.raises(selector.SelectionError, match="another selector owns publish lock"):
            with selector.acquire_publish_lock(prefix):
                pass


def test_distinct_pair_policy_rejects_a_grid_with_too_few_points(tmp_path: Path) -> None:
    contract = selector.load_release_contract(write_contract(tmp_path))
    raw = write_calibration(
        tmp_path,
        dataset="amazon",
        family="both_off",
        ef=100,
        mode_rows=(
            rows(selector.MODE_STOCK, 100, 0.9, 10)
            + rows(selector.MODE_SQLENS, 100, 0.9, 8)
        ),
    )
    configs = selector.load_calibration_csv(
        raw, contract, bootstrap_samples=100, bootstrap_seed=1
    )

    with pytest.raises(selector.SelectionError, match="distinct matched pairs"):
        selector.build_measurement_plan(
            configs,
            contract,
            (),
            bootstrap_samples=100,
            bootstrap_seed=1,
            target_policy="distinct_pairs",
            min_points_per_dataset=2,
            max_points_per_dataset=3,
        )


def test_coverage_aware_downsampling_keeps_endpoints_and_each_arm() -> None:
    states = [
        (position / 11, f"stock-{position}", f"sqlens-{position}")
        for position in range(12)
    ]

    selected = selector._coverage_aware_states(
        states,
        min_points=10,
        max_points=10,
    )

    assert len(selected) == 10
    assert selected[0] == states[0]
    assert selected[-1] == states[-1]
    assert len({stock for _, stock, _ in selected}) == 10
    assert len({sqlens for _, _, sqlens in selected}) == 10


def test_coverage_aware_downsampling_rejects_an_insufficient_cap() -> None:
    states = [
        (position / 18, f"stock-{min(position, 9)}", "sqlens-0")
        for position in range(10)
    ]
    states.extend(
        (
            position / 18,
            "stock-9",
            f"sqlens-{position - 9}",
        )
        for position in range(10, 19)
    )

    with pytest.raises(selector.SelectionError, match="requires more than 10"):
        selector._coverage_aware_states(
            states,
            min_points=10,
            max_points=10,
        )
