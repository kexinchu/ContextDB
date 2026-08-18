from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.hybrid_vector_db.scripts import (
    build_figure5_required_grid_contract as builder,
)
from experiments.hybrid_vector_db.scripts import (
    build_table6_matched_recall_summary as table6,
)
from experiments.hybrid_vector_db.scripts import (
    run_figure5_frontier as frontier,
)
from experiments.hybrid_vector_db.scripts import (
    select_figure5_matched_configs as selector,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def isolated() -> dict[str, object]:
    return {
        "parallel_db_cells": False,
        "lock_required": True,
        "lock_path": "/tmp/figure5-formal-test.lock",
        "lock_protocol": "fcntl_flock_exclusive_nonblocking_v1",
        "lock_acquired": True,
        "lock_owner_runner": "run_figure5_frontier",
        "lock_owner_pid": 12345,
        "lock_owner_token": "test-owner-token",
        "lock_acquired_at": "2026-07-31T00:00:00+00:00",
        "held_through_completion": True,
    }


def fixture(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    release = tmp_path / "release.json"
    config = tmp_path / "config.json"
    plan_path = tmp_path / "grid.json"
    write_json(
        release,
        {
            "schema_version": 1,
            "contract_id": "test-release-r36",
            "expected_sqlens_build_id": "test-build-r36",
            "expected_vector_so_sha256": "a" * 64,
        },
    )
    write_json(
        config,
        {
            "schema_version": 1,
            "release_contract": str(release),
            "protocol": {
                "qualification_scope": "global_min_predicate_lcb",
                "calibration_requests": 2800,
                "calibration_observations_per_predicate": 200,
            },
            "search_grid": {},
            "datasets": {
                "amazon": {},
                "yfcc": {},
                "laion": {},
            },
        },
    )
    definitions = {
        ("amazon", "stock"): [20, 60, 50000, 100000],
        ("amazon", "sqlens"): ["200:200", "500:500", "5000:1000", "10000:2000"],
        ("yfcc", "stock"): [250, 2000, 20000, 50000],
        ("yfcc", "sqlens"): ["100:20", "200:80", "5000:1000", "10000:2000"],
        ("laion", "stock"): [20000, 50000, 100000],
        ("laion", "sqlens"): ["3000:250", "5000:500", "10000:1000"],
    }
    groups: list[dict[str, object]] = []
    manifests: list[Path] = []
    for (dataset, mode), settings in definitions.items():
        family = "stock_strict" if mode == "stock" else "sqlens_target"
        output_dir = tmp_path / f"{dataset}_{mode}"
        group: dict[str, object] = {
            "dataset": dataset,
            "mode": mode,
            "family": family,
            "output_dir": str(output_dir),
        }
        if mode == "stock":
            group["ef_search_values"] = settings
        else:
            group["settings"] = settings
        groups.append(group)

        schedule: list[dict[str, object]] = []
        parsed_settings = (
            [(int(value), None) for value in settings]
            if mode == "stock"
            else [
                tuple(int(piece) for piece in str(value).split(":"))
                for value in settings
            ]
        )
        for ef_search, target in parsed_settings:
            raw = output_dir / builder.raw_name(
                dataset, family, ef_search, target
            )
            input_plan = raw.with_name(raw.name + ".plan.json")
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text("request_no,mode\n0,original\n", encoding="utf-8")
            write_json(input_plan, {"cell": raw.name})
            schedule.append(
                {
                    "dataset": dataset,
                    "phase": "calibration",
                    "scan_family": family,
                    "ef_search": ef_search,
                    "sqlens_scan_cap": None,
                    "sqlens_traversal_target": target,
                    "modes": [
                        "original"
                        if mode == "stock"
                        else "design1_bloom_bfs_layout_d3"
                    ],
                    "expected_rows": 2800,
                    "status": "complete",
                    "raw": str(raw.resolve()),
                    "raw_sha256": sha(raw),
                    "plan": str(input_plan.resolve()),
                    "plan_sha256": sha(input_plan),
                    "database_isolation": isolated(),
                }
            )
        manifest_path = output_dir / builder.RUNNER_MANIFEST_NAMES[mode]
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": builder.RUNNER_ARTIFACT_TYPES[mode],
            "status": "complete",
            "requested_slice_complete": True,
            "cells_total": len(schedule),
            "cells_complete": len(schedule),
            "config": {"path": str(config.resolve()), "sha256": sha(config)},
            "release_contract": {
                "path": str(release.resolve()),
                "sha256": sha(release),
            },
            "database_isolation": isolated(),
            "schedule": schedule,
        }
        if mode == "stock":
            manifest.update(
                {
                    "phase": "calibration",
                    "search_grid": {
                        "budgets": settings,
                        "scan_families": [family],
                        "calibration_repeats": 1,
                    },
                }
            )
        else:
            manifest.update(
                {
                    "datasets": [dataset],
                    "settings": [
                        {
                            "ef_search": ef_search,
                            "traversal_guided_target": target,
                        }
                        for ef_search, target in parsed_settings
                    ],
                }
            )
        write_json(manifest_path, manifest)
        manifests.append(manifest_path)

    write_json(
        plan_path,
        {
            "schema_version": 1,
            "plan_id": "test-table6-isolated-grid",
            "release_contract": str(release),
            "dataset_config": str(config),
            "qualification_scope": "global_min_predicate_lcb",
            "targets": [0.90, 0.95, 0.99],
            "protocol": {
                "calibration_requests": 2800,
                "observations_per_predicate": 200,
                "predicates": 14,
                "parallel_db_cells": False,
                "require_global_db_lock": True,
                "cache_state": "warm",
                "screening_latency_eligible": False,
            },
            "groups": groups,
        },
    )
    return plan_path, release, manifests


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def mutate(path: Path, callback) -> None:
    value = load(path)
    callback(value)
    write_json(path, value)


def write_calibration_raw(
    path: Path,
    *,
    mode: str,
    ef_search: int,
    traversal_target: int | None,
    recall: float = 0.5,
) -> str:
    rows: list[dict[str, object]] = []
    request_no = 0
    for query_id in ("q0", "q1"):
        for filter_name in ("f0", "f1"):
            rows.append(
                {
                    "mode": mode,
                    "error": "",
                    "recall": recall,
                    "request_no": request_no,
                    "query_id": query_id,
                    "filter_name": filter_name,
                    "sqlens_build_id": "test-build-r36",
                    "vector_so_sha256": "a" * 64,
                    "ef_search": ef_search,
                    "iterative_scan": (
                        "strict_order" if mode == "original" else "off"
                    ),
                    "max_scan_tuples": 200000,
                    "scan_mem_multiplier": 1.0,
                    "guided_collect_target": ef_search,
                    "traversal_guided_target": traversal_target or 11,
                    "d2_page_access": "off",
                    "d2_index_page_access": "off",
                    "table": "items",
                    "index": "items_hnsw",
                    "candidate_validity_predicate": "true",
                    "candidate_validity_predicate_sha256": "b" * 64,
                    "self_exclusion_contract": "query_id",
                    "scan_limit": 10,
                }
            )
            request_no += 1
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    normalized = selector._normalized_config(rows[0], "test row")
    return selector.sha256_json(normalized)


def test_builds_complete_contract_with_canonical_cell_keys(
    tmp_path: Path,
) -> None:
    plan, _, _ = fixture(tmp_path)

    contract = builder.build_contract(plan)

    assert contract["contract_type"] == (
        "figure5_formal_fixed_target_required_grid"
    )
    assert contract["grid_complete"] is True
    assert contract["groups"] == 6
    assert contract["cell_count"] == 22
    assert contract["targets"] == [0.90, 0.95, 0.99]
    assert contract["qualification_scope"] == "global_min_predicate_lcb"
    cells = contract["cells"]
    assert isinstance(cells, list)
    assert len({cell["cell_key"] for cell in cells}) == 22
    assert all(
        set(cell) == {
            "cell_key",
            "dataset",
            "arm",
            "mode",
            "family",
            "ef_search",
            "traversal_guided_target",
            "raw_csv",
            "input_plan",
            "serial_runner_manifest",
        }
        for cell in cells
    )
    contract_path = tmp_path / "required-grid.json"
    builder.atomic_write_json(contract_path, contract)
    persisted = load(contract_path)
    assert persisted["cells"] == cells
    assert all(
        cell["cell_key"]
        == selector.calibration_cell_key(Path(cell["raw_csv"]["path"]))
        for cell in cells
    )


def test_cli_writes_atomically_and_reports_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan, _, _ = fixture(tmp_path)
    output = tmp_path / "nested/required-grid.json"

    assert builder.main(
        ["--grid-plan", str(plan), "--output", str(output)]
    ) == 0

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["cells"] == 22
    assert summary["groups"] == 6
    assert summary["grid_complete"] is True
    assert load(output)["cell_count"] == 22
    assert not list(output.parent.glob("*.tmp"))


def test_accepts_manifest_structure_emitted_by_frontier_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, release_path, manifests = fixture(tmp_path)
    plan_payload = load(plan)
    config_path = Path(plan_payload["dataset_config"])
    output_dir = manifests[0].parent
    fake_config = {
        "release_contract_path": str(release_path.resolve()),
        "release_contract_sha256": sha(release_path),
        "release_identity": load(release_path),
        "protocol": {},
        "datasets": {"amazon": {}},
        "search_grid": {
            "ef_search": [],
            "extension_ef_search": [],
        },
    }

    def fake_build_cell_command(
        config: dict[str, Any],
        dataset: str,
        phase: str,
        family: str,
        ef_search: int,
        raw: Path,
        backend_cpu_list: str,
        calibration_repeats: int,
        sqlens_scan_cap: int | None,
    ) -> tuple[list[str], dict[str, Any]]:
        del config, raw, backend_cpu_list
        return [], {
            "dataset": dataset,
            "dataset_label": "Amazon-10M",
            "phase": phase,
            "scan_family": family,
            "ef_search": ef_search,
            "sqlens_scan_cap": sqlens_scan_cap,
            "sqlens_traversal_target": None,
            "requests": 2800,
            "repeats": calibration_repeats,
            "expected_rows": 2800,
            "modes": ["original"],
            "workload_contract": {},
        }

    monkeypatch.setattr(frontier, "load_config", lambda path: fake_config)
    monkeypatch.setattr(frontier, "build_cell_command", fake_build_cell_command)
    monkeypatch.setattr(frontier, "cell_complete", lambda *args: False)
    args = frontier.create_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--phase",
            "calibration",
            "--datasets",
            "amazon",
            "--ef-search-values",
            "20,60,50000,100000",
            "--scan-families",
            "stock_strict",
            "--out-dir",
            str(output_dir),
        ]
    )

    assert frontier.run(args) == 0
    generated_path = (
        output_dir / "figure5_r35_calibration_run_manifest.json"
    )
    generated = load(generated_path)
    assert generated["artifact_type"] == "sqlens_figure5_frontier_run"
    assert "cell_key" not in generated["schedule"][0]
    assert "plan_sha256" not in generated["schedule"][0]

    evidence = isolated()
    generated["status"] = "complete"
    generated["requested_slice_complete"] = True
    generated["cells_complete"] = generated["cells_total"]
    generated["database_isolation"] = evidence
    for cell in generated["schedule"]:
        raw = Path(cell["raw"])
        input_plan = Path(cell["plan"])
        assert raw.is_file()
        assert input_plan.is_file()
        cell["status"] = "complete"
        cell["raw_sha256"] = sha(raw)
        cell["database_isolation"] = copy.deepcopy(evidence)
    write_json(generated_path, generated)

    contract = builder.build_contract(plan)
    assert contract["grid_complete"] is True
    assert contract["cell_count"] == 22


def test_runner_shaped_manifests_flow_to_table6_exhaustion_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid_plan, release_path, manifests = fixture(tmp_path)
    config_sha_by_raw: dict[Path, str] = {}
    selected_config_by_arm: dict[tuple[str, str], str] = {}
    for manifest_path in manifests:
        manifest = load(manifest_path)
        arm = (
            "stock"
            if manifest["schedule"][0]["modes"][0] == "original"
            else "sqlens"
        )
        dataset = manifest["schedule"][0]["dataset"]
        for cell_no, cell in enumerate(manifest["schedule"]):
            raw = Path(cell["raw"])
            config_sha_by_raw[raw.resolve()] = write_calibration_raw(
                raw,
                mode=cell["modes"][0],
                ef_search=cell["ef_search"],
                traversal_target=cell["sqlens_traversal_target"],
                recall=0.92 if cell_no == 0 else 0.5,
            )
            if cell_no == 0:
                selected_config_by_arm[(dataset, arm)] = (
                    config_sha_by_raw[raw.resolve()]
                )
            cell["raw_sha256"] = sha(raw)
            cell.pop("cell_key", None)
            cell.pop("plan_sha256", None)
        write_json(manifest_path, manifest)

    contract_payload = builder.build_contract(grid_plan)
    contract_path = tmp_path / "required-grid.json"
    builder.atomic_write_json(contract_path, contract_payload)
    monkeypatch.setattr(table6, "EXPECTED_CALIBRATION_REQUESTS", 4)
    monkeypatch.setattr(
        table6, "EXPECTED_CALIBRATION_OBSERVATIONS_PER_FILTER", 2
    )
    monkeypatch.setattr(table6, "EXPECTED_FILTERS", 2)
    monkeypatch.setattr(selector, "FORMAL_OBSERVATIONS_PER_FILTER", 2)
    monkeypatch.setattr(selector, "EXPECTED_FILTERS", 2)
    required_grid = table6.load_required_grid_contract(contract_path)

    runner_bindings: list[dict[str, object]] = []
    seen_runners: set[str] = set()
    for cell in contract_payload["cells"]:
        binding = cell["serial_runner_manifest"]
        if binding["path"] not in seen_runners:
            seen_runners.add(binding["path"])
            runner_bindings.append(dict(binding))
    grid_binding = {
        "path": str(contract_path),
        "sha256": sha(contract_path),
        "cell_count": 22,
        "cell_keys": sorted(
            cell["cell_key"] for cell in contract_payload["cells"]
        ),
        "serial_runner_manifests": runner_bindings,
    }
    selection_inputs = [
        {
            "raw_csv": cell["raw_csv"]["path"],
            "raw_csv_sha256": cell["raw_csv"]["sha256"],
            "input_plan": cell["input_plan"]["path"],
            "input_plan_sha256": cell["input_plan"]["sha256"],
            "dataset": cell["dataset"],
            "family": cell["family"],
            "mode": cell["mode"],
            "config_sha256": config_sha_by_raw[
                Path(cell["raw_csv"]["path"]).resolve()
            ],
        }
        for cell in contract_payload["cells"]
    ]
    candidates: dict[tuple[str, str], list[str]] = {}
    for cell in contract_payload["cells"]:
        key = (cell["dataset"], cell["arm"])
        candidates.setdefault(key, []).append(
            config_sha_by_raw[Path(cell["raw_csv"]["path"]).resolve()]
        )

    pairs: list[table6.SelectionPair] = []
    measurement_pairs: list[dict[str, object]] = []
    unattainable: list[dict[str, object]] = []
    for dataset in ("amazon", "yfcc", "laion"):
        for target in (0.90, 0.95, 0.99):
            pair_id = f"{dataset}:recall_{target:.2f}:unattainable"
            status = (
                table6.SELECTED
                if target == 0.90
                else table6.UNATTAINABLE
            )
            pairs.append(
                table6.SelectionPair(
                    dataset=dataset,
                    dataset_id=table6.DATASET_IDS[dataset],
                    target=target,
                    pair_id=pair_id,
                    status=status,
                    stock=None,
                    sqlens=None,
                    stock_selection_sha="",
                    sqlens_selection_sha="",
                    stock_arm_sha="",
                    sqlens_arm_sha="",
                    stock_status=status,
                    sqlens_status=status,
                )
            )
            measurement_row: dict[str, object] = {
                "pair_id": pair_id,
                "dataset": dataset,
                "target_recall": target,
                "selection_status": status,
                "stock_status": status,
                "sqlens_status": status,
            }
            if status == table6.SELECTED:
                for arm in ("stock", "sqlens"):
                    measurement_row[f"{arm}_config_sha256"] = (
                        selected_config_by_arm[(dataset, arm)]
                    )
                    measurement_row[
                        f"{arm}_calibration_"
                        "per_filter_recall_min_ci95_low"
                    ] = 0.92
            measurement_pairs.append(measurement_row)
            for arm in ("stock", "sqlens"):
                if status == table6.SELECTED:
                    continue
                config_shas = sorted(candidates[(dataset, arm)])
                unattainable.append(
                    {
                        "dataset": dataset,
                        "target_recall": target,
                        "arm": arm,
                        "status": (
                            "unattainable_on_complete_required_grid"
                        ),
                        "candidate_configs": len(config_shas),
                        "candidate_config_sha256s": config_shas,
                        "maximum_qualification_floor": 0.92,
                    }
                )
    proof_body = {
        "required_grid_contract_present": True,
        "required_grid_complete": True,
        "input_set_exact": True,
        "required_grid_contract_sha256": sha(contract_path),
        "required_grid_cell_keys_sha256": table6.sha256_json(
            grid_binding["cell_keys"]
        ),
        "qualification_scope": "global_min_predicate_lcb",
        "targets": [0.90, 0.95, 0.99],
        "unattainable_arms": unattainable,
    }
    proof = {
        **proof_body,
        "proof_sha256": table6.sha256_json(proof_body),
    }
    selection_plan = {
        "required_grid_contract": grid_binding,
        "bootstrap": {"samples": 100, "seed": 17},
        "inputs": selection_inputs,
        "measurement_pairs": measurement_pairs,
        "exhaustion_proof": proof,
    }
    selection_manifest = {
        "required_grid_contract": grid_binding,
        "exhaustion_proof": proof,
    }
    release = {
        "path": str(release_path.resolve()),
        "sha256": sha(release_path),
        "expected_sqlens_build_id": "test-build-r36",
        "expected_vector_so_sha256": "a" * 64,
    }

    evidence = table6.audit_selection_grid_and_exhaustion(
        selection_plan,
        selection_manifest,
        pairs,
        required_grid,
        release,
    )

    assert len(evidence) == 22
    assert {item.qualification_lcb95 for item in evidence} == {0.5, 0.92}

    bad_grid_binding = copy.deepcopy(selection_plan)
    bad_grid_binding["required_grid_contract"]["sha256"] = "0" * 64
    with pytest.raises(
        table6.Table6SummaryError,
        match="required-grid path/SHA/cell count differs",
    ):
        table6.audit_selection_grid_and_exhaustion(
            bad_grid_binding,
            selection_manifest,
            pairs,
            required_grid,
            release,
        )

    bad_input = copy.deepcopy(selection_plan)
    bad_input["inputs"][0]["raw_csv_sha256"] = "0" * 64
    with pytest.raises(
        table6.Table6SummaryError,
        match="selection input binding differs",
    ):
        table6.audit_selection_grid_and_exhaustion(
            bad_input,
            selection_manifest,
            pairs,
            required_grid,
            release,
        )

    bad_selected_lcb = copy.deepcopy(selection_plan)
    bad_selected_lcb["measurement_pairs"][0][
        "stock_calibration_per_filter_recall_min_ci95_low"
    ] = 0.91
    with pytest.raises(
        table6.Table6SummaryError,
        match="selected LCB evidence is invalid",
    ):
        table6.audit_selection_grid_and_exhaustion(
            bad_selected_lcb,
            selection_manifest,
            pairs,
            required_grid,
            release,
        )

    bad_plan = copy.deepcopy(selection_plan)
    bad_manifest = copy.deepcopy(selection_manifest)
    bad_plan["exhaustion_proof"]["unattainable_arms"][0][
        "maximum_qualification_floor"
    ] = 0.6
    bad_body = dict(bad_plan["exhaustion_proof"])
    bad_body.pop("proof_sha256")
    bad_plan["exhaustion_proof"]["proof_sha256"] = table6.sha256_json(
        bad_body
    )
    bad_manifest["exhaustion_proof"] = copy.deepcopy(
        bad_plan["exhaustion_proof"]
    )
    with pytest.raises(
        table6.Table6SummaryError,
        match="exhaustion LCB evidence is invalid",
    ):
        table6.audit_selection_grid_and_exhaustion(
            bad_plan,
            bad_manifest,
            pairs,
            required_grid,
            release,
        )


def test_rejects_missing_and_extra_schedule_cells(tmp_path: Path) -> None:
    plan, _, manifests = fixture(tmp_path)
    manifest = manifests[0]
    original = load(manifest)

    missing = copy.deepcopy(original)
    missing["schedule"].pop()
    missing["cells_total"] -= 1
    missing["cells_complete"] -= 1
    write_json(manifest, missing)
    with pytest.raises(builder.RequiredGridBuildError, match="cells_total mismatch"):
        builder.build_contract(plan)

    extra = copy.deepcopy(original)
    duplicate = copy.deepcopy(extra["schedule"][0])
    duplicate["raw"] = str(tmp_path / "extra.csv")
    duplicate["plan"] = str(tmp_path / "extra.csv.plan.json")
    duplicate["ef_search"] = 40
    extra["schedule"].append(duplicate)
    write_json(manifest, extra)
    with pytest.raises(
        builder.RequiredGridBuildError, match="does not exactly match grid"
    ):
        builder.build_contract(plan)


def test_rejects_duplicate_schedule_cell(tmp_path: Path) -> None:
    plan, _, manifests = fixture(tmp_path)
    manifest = manifests[0]

    def duplicate(value: dict[str, object]) -> None:
        value["schedule"][1]["raw"] = value["schedule"][0]["raw"]

    mutate(manifest, duplicate)
    with pytest.raises(
        builder.RequiredGridBuildError, match="duplicate runner schedule cell"
    ):
        builder.build_contract(plan)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "running", "not complete"),
        ("requested_slice_complete", False, "requested slice is incomplete"),
    ],
)
def test_rejects_incomplete_manifest(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    plan, _, manifests = fixture(tmp_path)
    mutate(manifests[0], lambda manifest: manifest.__setitem__(field, value))
    with pytest.raises(builder.RequiredGridBuildError, match=message):
        builder.build_contract(plan)


@pytest.mark.parametrize(
    "isolation",
    [
        None,
        {
            "parallel_db_cells": True,
            **{
                key: value
                for key, value in isolated().items()
                if key != "parallel_db_cells"
            },
        },
        {
            "parallel_db_cells": False,
            **{
                key: value
                for key, value in isolated().items()
                if key not in {"parallel_db_cells", "lock_required"}
            },
            "lock_required": False,
        },
        {
            "parallel_db_cells": False,
            **{
                key: value
                for key, value in isolated().items()
                if key not in {"parallel_db_cells", "held_through_completion"}
            },
            "held_through_completion": False,
        },
    ],
)
def test_rejects_nonisolated_manifest(
    tmp_path: Path, isolation: object
) -> None:
    plan, _, manifests = fixture(tmp_path)

    def change(value: dict[str, object]) -> None:
        if isolation is None:
            value.pop("database_isolation")
        else:
            value["database_isolation"] = isolation

    mutate(manifests[0], change)
    with pytest.raises(
        builder.RequiredGridBuildError,
        match="database_isolation|not globally isolated",
    ):
        builder.build_contract(plan)


def test_accepts_resumed_cell_lock_epoch(tmp_path: Path) -> None:
    plan, _, manifests = fixture(tmp_path)

    def drift(value: dict[str, object]) -> None:
        evidence = value["schedule"][0]["database_isolation"]
        evidence["lock_owner_pid"] = 54321
        evidence["lock_owner_token"] = "resumed-owner"
        evidence["lock_acquired_at"] = "2026-07-31T01:00:00+00:00"

    mutate(manifests[0], drift)
    contract = builder.build_contract(plan)
    assert contract["grid_complete"] is True


def test_rejects_cell_lock_namespace_drift(tmp_path: Path) -> None:
    plan, _, manifests = fixture(tmp_path)

    def drift(value: dict[str, object]) -> None:
        value["schedule"][0]["database_isolation"]["lock_path"] = (
            "/tmp/different-formal.lock"
        )

    mutate(manifests[0], drift)
    with pytest.raises(
        builder.RequiredGridBuildError,
        match="does not use the manifest global lock namespace",
    ):
        builder.build_contract(plan)


def test_rejects_wrong_runner_family_and_settings(tmp_path: Path) -> None:
    plan, _, manifests = fixture(tmp_path)
    stock_manifest, sqlens_manifest = manifests[0], manifests[1]

    mutate(
        stock_manifest,
        lambda manifest: manifest.__setitem__(
            "artifact_type", "sqlens_figure5_sqlens_target_extension"
        ),
    )
    with pytest.raises(builder.RequiredGridBuildError, match="wrong runner type"):
        builder.build_contract(plan)

    fixture_plan, _, fresh_manifests = fixture(tmp_path / "fresh")
    mutate(
        fresh_manifests[1],
        lambda manifest: manifest["settings"][0].__setitem__(
            "traversal_guided_target", 199
        ),
    )
    with pytest.raises(builder.RequiredGridBuildError, match="settings mismatch"):
        builder.build_contract(fixture_plan)


def test_rejects_release_and_config_binding_drift(tmp_path: Path) -> None:
    plan, _, manifests = fixture(tmp_path)
    manifest = manifests[0]

    mutate(
        manifest,
        lambda value: value["config"].__setitem__("sha256", "0" * 64),
    )
    with pytest.raises(builder.RequiredGridBuildError, match="config SHA mismatch"):
        builder.build_contract(plan)

    fresh_plan, _, fresh_manifests = fixture(tmp_path / "fresh")
    mutate(
        fresh_manifests[0],
        lambda value: value["release_contract"].__setitem__(
            "sha256", "0" * 64
        ),
    )
    with pytest.raises(builder.RequiredGridBuildError, match="release SHA mismatch"):
        builder.build_contract(fresh_plan)


@pytest.mark.parametrize("artifact", ["raw", "plan"])
def test_rejects_raw_or_plan_sha_drift(
    tmp_path: Path, artifact: str
) -> None:
    plan, _, manifests = fixture(tmp_path)
    manifest = manifests[0]
    value = load(manifest)
    cell = value["schedule"][0]
    field = "raw_sha256" if artifact == "raw" else "plan_sha256"
    cell[field] = "0" * 64
    write_json(manifest, value)

    with pytest.raises(builder.RequiredGridBuildError, match="SHA mismatch"):
        builder.build_contract(plan)


def test_rejects_grid_group_count_and_duplicate_setting(tmp_path: Path) -> None:
    plan, _, _ = fixture(tmp_path)
    payload = load(plan)
    payload["groups"].pop()
    write_json(plan, payload)
    with pytest.raises(builder.RequiredGridBuildError, match="exactly 6 groups"):
        builder.build_contract(plan)

    fresh_plan, _, _ = fixture(tmp_path / "fresh")
    payload = load(fresh_plan)
    payload["groups"][0]["ef_search_values"][1] = 20
    write_json(fresh_plan, payload)
    with pytest.raises(builder.RequiredGridBuildError, match="duplicate settings"):
        builder.build_contract(fresh_plan)
