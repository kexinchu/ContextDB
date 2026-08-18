#!/usr/bin/env python3
"""Build the formal Figure 5 required-grid contract from serial artifacts.

This builder is intentionally read-only with respect to experiment artifacts.
It validates the frozen isolated-grid plan and completed serial runner
manifests, then atomically writes the contract consumed by
``select_figure5_matched_configs.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import select_figure5_matched_configs as selector
except ImportError:
    import select_figure5_matched_configs as selector


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRID_PLAN = (
    ROOT
    / "experiments/hybrid_vector_db/configs/"
    "table6_r37_isolated_grid_plan.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/hybrid_vector_db/figure5_r37_table6/"
    "figure5_r37_required_grid_contract.json"
)
EXPECTED_GROUPS = 6
EXPECTED_CELLS = 22
FROZEN_DATASETS = ("amazon", "yfcc", "laion")
STOCK_MODE = "original"
SQLENS_MODE = "design1_bloom_bfs_layout_d3"
RUNNER_MANIFEST_NAMES = {
    "stock": "figure5_r35_calibration_run_manifest.json",
    "sqlens": "figure5_r36_sqlens_target_extension_manifest.json",
}
RUNNER_ARTIFACT_TYPES = {
    "stock": "sqlens_figure5_frontier_run",
    "sqlens": "sqlens_figure5_sqlens_target_extension",
}
EXPECTED_FAMILIES = {
    "stock": selector.FAMILY_STOCK_STRICT,
    "sqlens": selector.FAMILY_SQLENS_TARGET,
}
GLOBAL_DB_LOCK_PROTOCOL = "fcntl_flock_exclusive_nonblocking_v1"


class RequiredGridBuildError(RuntimeError):
    """Raised when the isolated grid cannot be proven complete."""


@dataclass(frozen=True)
class ExpectedCell:
    dataset: str
    mode: str
    family: str
    ef_search: int
    traversal_target: int | None
    raw_path: Path
    plan_path: Path
    cell_key: str


@dataclass(frozen=True)
class ExpectedGroup:
    dataset: str
    mode: str
    family: str
    output_dir: Path
    manifest_path: Path
    cells: tuple[ExpectedCell, ...]
    require_database_isolation_evidence: bool
    isolation_recovery_reason: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RequiredGridBuildError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RequiredGridBuildError(f"{label} root must be an object: {path}")
    return value


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RequiredGridBuildError(f"{label} must be a non-empty string")
    return value


def positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise RequiredGridBuildError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RequiredGridBuildError(f"{label} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise RequiredGridBuildError(f"{label} must be a positive integer")
    return result


def binding_path(value: object, label: str) -> Path:
    path = Path(text(value, label))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def binding_sha(binding: object, label: str) -> str:
    if not isinstance(binding, Mapping):
        raise RequiredGridBuildError(f"{label} must be an object")
    value = binding.get("sha256")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RequiredGridBuildError(f"{label} has an invalid SHA-256")
    return value


def require_file_binding(
    binding: object,
    expected_path: Path,
    label: str,
) -> None:
    if not isinstance(binding, Mapping):
        raise RequiredGridBuildError(f"{label} binding is missing")
    observed_path = binding_path(binding.get("path"), f"{label} path")
    if observed_path != expected_path:
        raise RequiredGridBuildError(
            f"{label} path mismatch: expected={expected_path}, observed={observed_path}"
        )
    if not expected_path.is_file():
        raise RequiredGridBuildError(f"{label} is missing: {expected_path}")
    expected_sha = sha256_file(expected_path)
    observed_sha = binding_sha(binding, label)
    if observed_sha != expected_sha:
        raise RequiredGridBuildError(
            f"{label} SHA mismatch: expected={expected_sha}, observed={observed_sha}"
        )


def parse_targets(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise RequiredGridBuildError("grid plan targets must be a non-empty list")
    try:
        targets = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RequiredGridBuildError("grid plan targets must be numeric") from exc
    if (
        any(target <= 0.0 or target > 1.0 for target in targets)
        or len(set(targets)) != len(targets)
        or targets != tuple(sorted(targets))
    ):
        raise RequiredGridBuildError(
            "grid plan targets must be unique, ascending, and in (0, 1]"
        )
    return targets


def validate_plan_protocol(plan: Mapping[str, Any]) -> None:
    protocol = plan.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RequiredGridBuildError("grid plan protocol is missing")
    expected = {
        "calibration_requests": 2800,
        "observations_per_predicate": 200,
        "predicates": 14,
        "parallel_db_cells": False,
        "require_global_db_lock": True,
        "cache_state": "warm",
        "screening_latency_eligible": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RequiredGridBuildError(
                f"grid plan protocol {key} mismatch: "
                f"expected={value!r}, observed={protocol.get(key)!r}"
            )


def raw_name(
    dataset: str,
    family: str,
    ef_search: int,
    traversal_target: int | None,
) -> str:
    suffix = (
        f"_target{traversal_target}" if traversal_target is not None else ""
    )
    return (
        f"figure5_r35_{dataset}_calibration_{family}_"
        f"ef{ef_search}{suffix}.csv"
    )


def group_manifest_path(group: Mapping[str, Any], mode: str, out_dir: Path) -> Path:
    configured = group.get("serial_runner_manifest")
    if configured is not None:
        path = binding_path(configured, "group serial_runner_manifest")
        if path.parent != out_dir:
            raise RequiredGridBuildError(
                "group serial_runner_manifest must be inside output_dir"
            )
        return path
    return (out_dir / RUNNER_MANIFEST_NAMES[mode]).resolve()


def parse_setting(value: object, label: str) -> tuple[int, int]:
    item = text(value, label)
    pieces = item.split(":")
    if len(pieces) != 2:
        raise RequiredGridBuildError(f"{label} must use ef:target syntax")
    try:
        ef_search, target = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise RequiredGridBuildError(f"{label} must use ef:target syntax") from exc
    if ef_search <= 0 or target < 11 or target > ef_search:
        raise RequiredGridBuildError(
            f"{label} must satisfy ef > 0 and 11 <= target <= ef"
        )
    return ef_search, target


def parse_groups(plan: Mapping[str, Any]) -> tuple[ExpectedGroup, ...]:
    raw_datasets = plan.get("datasets")
    if raw_datasets is None:
        datasets = FROZEN_DATASETS
    elif (
        not isinstance(raw_datasets, list)
        or not raw_datasets
        or any(dataset not in FROZEN_DATASETS for dataset in raw_datasets)
        or len(set(raw_datasets)) != len(raw_datasets)
    ):
        raise RequiredGridBuildError(
            "grid plan datasets must be a non-empty unique subset of "
            f"{list(FROZEN_DATASETS)}"
        )
    else:
        datasets = tuple(str(dataset) for dataset in raw_datasets)

    expected_groups = len(datasets) * 2
    expected_cells_value = plan.get("expected_cell_count")
    if datasets == FROZEN_DATASETS and expected_cells_value is None:
        expected_cells = EXPECTED_CELLS
    else:
        expected_cells = positive_int(
            expected_cells_value, "grid plan expected_cell_count"
        )

    groups = plan.get("groups")
    if not isinstance(groups, list) or len(groups) != expected_groups:
        raise RequiredGridBuildError(
            f"isolated grid must contain exactly {expected_groups} groups"
        )
    parsed: list[ExpectedGroup] = []
    group_keys: set[tuple[str, str]] = set()
    all_cell_keys: set[str] = set()
    for index, value in enumerate(groups):
        label = f"grid group {index}"
        if not isinstance(value, Mapping):
            raise RequiredGridBuildError(f"{label} must be an object")
        dataset = text(value.get("dataset"), f"{label} dataset")
        mode = text(value.get("mode"), f"{label} mode")
        family = text(value.get("family"), f"{label} family")
        if mode not in EXPECTED_FAMILIES:
            raise RequiredGridBuildError(f"{label} has unsupported mode: {mode}")
        if family != EXPECTED_FAMILIES[mode]:
            raise RequiredGridBuildError(
                f"{label} family mismatch for {mode}: {family}"
            )
        group_key = (dataset, mode)
        if group_key in group_keys:
            raise RequiredGridBuildError(f"duplicate grid group: {group_key}")
        group_keys.add(group_key)
        output_dir = binding_path(value.get("output_dir"), f"{label} output_dir")
        require_isolation = value.get(
            "require_database_isolation_evidence", True
        )
        if not isinstance(require_isolation, bool):
            raise RequiredGridBuildError(
                f"{label} require_database_isolation_evidence must be boolean"
            )
        recovery_reason = value.get("isolation_recovery_reason")
        if not require_isolation:
            recovery_reason = text(
                recovery_reason, f"{label} isolation_recovery_reason"
            )
        elif recovery_reason is not None:
            raise RequiredGridBuildError(
                f"{label} cannot declare an isolation recovery reason when "
                "evidence is required"
            )

        settings: list[tuple[int, int | None]]
        if mode == "stock":
            values = value.get("ef_search_values")
            if (
                not isinstance(values, list)
                or not values
                or "settings" in value
            ):
                raise RequiredGridBuildError(
                    f"{label} stock group requires only ef_search_values"
                )
            settings = [
                (positive_int(item, f"{label} ef_search"), None)
                for item in values
            ]
        else:
            values = value.get("settings")
            if (
                not isinstance(values, list)
                or not values
                or "ef_search_values" in value
            ):
                raise RequiredGridBuildError(
                    f"{label} sqlens group requires only settings"
                )
            settings = [
                parse_setting(item, f"{label} setting") for item in values
            ]
        if len(settings) != len(set(settings)):
            raise RequiredGridBuildError(f"{label} contains duplicate settings")

        cells: list[ExpectedCell] = []
        for ef_search, traversal_target in settings:
            raw_path = (
                output_dir
                / raw_name(dataset, family, ef_search, traversal_target)
            ).resolve()
            plan_path = raw_path.with_name(raw_path.name + ".plan.json")
            cell_key = selector.calibration_cell_key(raw_path)
            if cell_key in all_cell_keys:
                raise RequiredGridBuildError(f"duplicate grid cell: {cell_key}")
            all_cell_keys.add(cell_key)
            cells.append(
                ExpectedCell(
                    dataset=dataset,
                    mode=mode,
                    family=family,
                    ef_search=ef_search,
                    traversal_target=traversal_target,
                    raw_path=raw_path,
                    plan_path=plan_path,
                    cell_key=cell_key,
                )
            )
        parsed.append(
            ExpectedGroup(
                dataset=dataset,
                mode=mode,
                family=family,
                output_dir=output_dir,
                manifest_path=group_manifest_path(value, mode, output_dir),
                cells=tuple(cells),
                require_database_isolation_evidence=require_isolation,
                isolation_recovery_reason=recovery_reason,
            )
        )

    expected_group_keys = {
        (dataset, mode)
        for dataset in datasets
        for mode in ("stock", "sqlens")
    }
    if group_keys != expected_group_keys:
        raise RequiredGridBuildError(
            "isolated grid groups must cover stock and sqlens for "
            f"{', '.join(datasets)} exactly once"
        )
    if len(all_cell_keys) != expected_cells:
        raise RequiredGridBuildError(
            f"isolated grid must contain exactly {expected_cells} cells; "
            f"observed={len(all_cell_keys)}"
        )
    return tuple(parsed)


def validate_database_isolation(
    isolation: object,
    label: str,
    *,
    expected: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(isolation, Mapping):
        raise RequiredGridBuildError(
            f"{label} lacks database_isolation"
        )
    if (
        isolation.get("parallel_db_cells") is not False
        or isolation.get("lock_required") is not True
        or isolation.get("lock_acquired") is not True
        or isolation.get("held_through_completion") is not True
        or isolation.get("lock_protocol") != GLOBAL_DB_LOCK_PROTOCOL
    ):
        raise RequiredGridBuildError(
            f"{label} is not globally isolated through completion"
        )
    lock_path = Path(text(isolation.get("lock_path"), f"{label} lock_path"))
    if not lock_path.is_absolute():
        raise RequiredGridBuildError(f"{label} lock_path must be absolute")
    owner_fields = {
        "lock_owner_runner": isolation.get("lock_owner_runner"),
        "lock_owner_pid": isolation.get("lock_owner_pid"),
        "lock_owner_token": isolation.get("lock_owner_token"),
        "lock_acquired_at": isolation.get("lock_acquired_at"),
    }
    if (
        not isinstance(owner_fields["lock_owner_runner"], str)
        or not owner_fields["lock_owner_runner"]
        or not isinstance(owner_fields["lock_owner_pid"], int)
        or isinstance(owner_fields["lock_owner_pid"], bool)
        or owner_fields["lock_owner_pid"] <= 0
        or not isinstance(owner_fields["lock_owner_token"], str)
        or not owner_fields["lock_owner_token"]
        or not isinstance(owner_fields["lock_acquired_at"], str)
        or not owner_fields["lock_acquired_at"]
    ):
        raise RequiredGridBuildError(
            f"{label} has incomplete global lock owner evidence"
        )
    if expected is not None:
        # A resumable formal run can acquire the same global lock in multiple
        # serial epochs.  PID, token, and acquisition time must be present for
        # every completed cell, but they are expected to differ after a
        # restart.  The invariant shared by all epochs is the lock namespace
        # and protocol; each cell independently proves exclusive ownership
        # through its own completion.
        shared_lock_fields = (
            "lock_path",
            "lock_protocol",
        )
        drift = [
            field
            for field in shared_lock_fields
            if isolation.get(field) != expected.get(field)
        ]
        if drift:
            raise RequiredGridBuildError(
                f"{label} does not use the manifest global lock namespace: {drift}"
            )
    return isolation


def validate_runner_shape(
    manifest: Mapping[str, Any],
    group: ExpectedGroup,
) -> None:
    path = group.manifest_path
    if manifest.get("artifact_type") != RUNNER_ARTIFACT_TYPES[group.mode]:
        raise RequiredGridBuildError(
            f"wrong runner type for {group.dataset}/{group.mode}: {path}"
        )
    if manifest.get("status") != "complete":
        raise RequiredGridBuildError(f"runner manifest is not complete: {path}")
    if manifest.get("requested_slice_complete") is not True:
        raise RequiredGridBuildError(
            f"runner requested slice is incomplete: {path}"
        )
    if manifest.get("cells_total") != len(group.cells):
        raise RequiredGridBuildError(f"runner cells_total mismatch: {path}")
    if manifest.get("cells_complete") != len(group.cells):
        raise RequiredGridBuildError(f"runner cells_complete mismatch: {path}")
    if group.require_database_isolation_evidence:
        validate_database_isolation(
            manifest.get("database_isolation"),
            f"runner manifest {path}",
        )

    if group.mode == "stock":
        if manifest.get("phase") != "calibration":
            raise RequiredGridBuildError(f"stock runner phase mismatch: {path}")
        search_grid = manifest.get("search_grid")
        if not isinstance(search_grid, Mapping):
            raise RequiredGridBuildError(f"stock runner grid is missing: {path}")
        if search_grid.get("scan_families") != [group.family]:
            raise RequiredGridBuildError(f"stock runner family mismatch: {path}")
        if search_grid.get("calibration_repeats") != 1:
            raise RequiredGridBuildError(
                f"stock runner calibration repeat mismatch: {path}"
            )
        expected_budgets = [cell.ef_search for cell in group.cells]
        if search_grid.get("budgets") != expected_budgets:
            raise RequiredGridBuildError(f"stock runner budgets mismatch: {path}")
    else:
        if manifest.get("datasets") != [group.dataset]:
            raise RequiredGridBuildError(
                f"sqlens runner dataset mismatch: {path}"
            )
        expected_settings = [
            {
                "ef_search": cell.ef_search,
                "traversal_guided_target": cell.traversal_target,
            }
            for cell in group.cells
        ]
        if manifest.get("settings") != expected_settings:
            raise RequiredGridBuildError(
                f"sqlens runner settings mismatch: {path}"
            )


def validate_schedule(
    manifest: Mapping[str, Any],
    group: ExpectedGroup,
) -> list[dict[str, Any]]:
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list):
        raise RequiredGridBuildError(
            f"runner manifest schedule is missing: {group.manifest_path}"
        )
    expected = {cell.raw_path: cell for cell in group.cells}
    observed: dict[Path, Mapping[str, Any]] = {}
    for index, item in enumerate(schedule):
        if not isinstance(item, Mapping):
            raise RequiredGridBuildError(
                f"runner schedule cell {index} is malformed: {group.manifest_path}"
            )
        raw_path = binding_path(
            item.get("raw"), f"runner schedule cell {index} raw"
        )
        if raw_path in observed:
            raise RequiredGridBuildError(
                f"duplicate runner schedule cell: {raw_path}"
            )
        observed[raw_path] = item
    missing = sorted(str(path) for path in set(expected) - set(observed))
    extra = sorted(str(path) for path in set(observed) - set(expected))
    if missing or extra:
        raise RequiredGridBuildError(
            "runner schedule does not exactly match grid; "
            f"manifest={group.manifest_path}, missing={missing}, extra={extra}"
        )

    bindings: list[dict[str, Any]] = []
    manifest_sha = sha256_file(group.manifest_path)
    manifest_isolation = (
        validate_database_isolation(
            manifest.get("database_isolation"),
            f"runner manifest {group.manifest_path}",
        )
        if group.require_database_isolation_evidence
        else None
    )
    for raw_path in sorted(expected):
        cell = expected[raw_path]
        item = observed[raw_path]
        key = cell.cell_key
        recorded_key = item.get("cell_key")
        if recorded_key is not None and recorded_key != key:
            raise RequiredGridBuildError(
                f"runner schedule has noncanonical cell_key for {raw_path}"
            )
        if (
            item.get("status") != "complete"
            or item.get("dataset") != cell.dataset
            or item.get("phase") != "calibration"
            or item.get("scan_family") != cell.family
            or item.get("ef_search") != cell.ef_search
            or item.get("sqlens_scan_cap") is not None
            or item.get("sqlens_traversal_target") != cell.traversal_target
        ):
            raise RequiredGridBuildError(
                f"runner schedule metadata mismatch for {key}"
            )
        expected_modes = [STOCK_MODE] if cell.mode == "stock" else [SQLENS_MODE]
        if item.get("modes") != expected_modes:
            raise RequiredGridBuildError(f"runner modes mismatch for {key}")
        if item.get("expected_rows") != 2800:
            raise RequiredGridBuildError(
                f"runner expected_rows mismatch for {key}"
            )
        if group.require_database_isolation_evidence:
            validate_database_isolation(
                item.get("database_isolation"),
                f"runner schedule cell {key}",
                expected=manifest_isolation,
            )
        if not cell.raw_path.is_file():
            raise RequiredGridBuildError(
                f"{key} raw CSV is missing: {cell.raw_path}"
            )
        recorded_raw_sha = item.get("raw_sha256")
        actual_raw_sha = sha256_file(cell.raw_path)
        if recorded_raw_sha is not None and recorded_raw_sha != actual_raw_sha:
            raise RequiredGridBuildError(
                f"{key} raw CSV SHA mismatch: "
                f"expected={actual_raw_sha}, observed={recorded_raw_sha}"
            )
        observed_plan = binding_path(
            item.get("plan"), f"{key} input plan path"
        )
        if observed_plan != cell.plan_path:
            raise RequiredGridBuildError(
                f"{key} input plan path mismatch: "
                f"expected={cell.plan_path}, observed={observed_plan}"
            )
        if not cell.plan_path.is_file():
            raise RequiredGridBuildError(
                f"{key} input plan is missing: {cell.plan_path}"
            )
        recorded_plan_sha = item.get("plan_sha256")
        actual_plan_sha = sha256_file(cell.plan_path)
        if recorded_plan_sha is not None and recorded_plan_sha != actual_plan_sha:
            raise RequiredGridBuildError(
                f"{key} input plan SHA mismatch: "
                f"expected={actual_plan_sha}, observed={recorded_plan_sha}"
            )
        binding = {
                "cell_key": key,
                "dataset": cell.dataset,
                "arm": cell.mode,
                "mode": expected_modes[0],
                "family": cell.family,
                "ef_search": cell.ef_search,
                "traversal_guided_target": cell.traversal_target,
                "raw_csv": {
                    "path": str(cell.raw_path),
                    "sha256": actual_raw_sha,
                },
                "input_plan": {
                    "path": str(cell.plan_path),
                    "sha256": actual_plan_sha,
                },
                "serial_runner_manifest": {
                    "path": str(group.manifest_path),
                    "sha256": manifest_sha,
                },
            }
        if not group.require_database_isolation_evidence:
            binding["database_isolation_evidence"] = "recovered_missing"
            binding["isolation_recovery_reason"] = (
                group.isolation_recovery_reason
            )
        bindings.append(binding)
    return bindings


def build_contract(grid_plan_path: Path) -> dict[str, Any]:
    grid_plan_path = grid_plan_path.resolve()
    plan = read_json(grid_plan_path, "isolated grid plan")
    if plan.get("schema_version") != 1:
        raise RequiredGridBuildError("unsupported isolated grid plan schema")
    validate_plan_protocol(plan)
    targets = parse_targets(plan.get("targets"))
    qualification_scope = text(
        plan.get("qualification_scope"), "grid plan qualification_scope"
    )
    if qualification_scope != selector.QUALIFICATION_SCOPE_FORMAL:
        raise RequiredGridBuildError(
            "isolated grid must use global_min_predicate_lcb"
        )

    release_path = binding_path(
        plan.get("release_contract"), "grid plan release_contract"
    )
    config_path = binding_path(
        plan.get("dataset_config"), "grid plan dataset_config"
    )
    if not release_path.is_file() or not config_path.is_file():
        raise RequiredGridBuildError(
            "grid plan release contract or dataset config is missing"
        )
    release_sha = sha256_file(release_path)
    config_sha = sha256_file(config_path)
    config = read_json(config_path, "dataset config")
    config_release = binding_path(
        config.get("release_contract"), "dataset config release_contract"
    )
    if config_release != release_path:
        raise RequiredGridBuildError(
            "dataset config and isolated grid plan bind different releases"
        )
    config_protocol = config.get("protocol")
    if (
        not isinstance(config_protocol, Mapping)
        or config_protocol.get("qualification_scope") != qualification_scope
        or config_protocol.get("calibration_requests") != 2800
        or config_protocol.get("calibration_observations_per_predicate") != 200
    ):
        raise RequiredGridBuildError(
            "dataset config calibration protocol does not match grid plan"
        )

    groups = parse_groups(plan)
    cells: list[dict[str, Any]] = []
    for group in groups:
        manifest = read_json(group.manifest_path, "serial runner manifest")
        require_file_binding(
            manifest.get("config"),
            config_path,
            f"{group.dataset}/{group.mode} runner config",
        )
        require_file_binding(
            manifest.get("release_contract"),
            release_path,
            f"{group.dataset}/{group.mode} runner release",
        )
        validate_runner_shape(manifest, group)
        cells.extend(validate_schedule(manifest, group))

    expected_cells = sum(len(group.cells) for group in groups)
    if len(cells) != expected_cells:
        raise RequiredGridBuildError(
            f"validated cell count mismatch: expected={expected_cells}, "
            f"observed={len(cells)}"
        )
    keys = [cell["cell_key"] for cell in cells]
    if len(keys) != len(set(keys)):
        raise RequiredGridBuildError("validated grid contains duplicate cell keys")
    cells.sort(key=lambda cell: cell["cell_key"])
    return {
        "schema_version": 1,
        "contract_type": selector.REQUIRED_GRID_CONTRACT_TYPE,
        "grid_complete": True,
        "source_grid_plan": {
            "path": str(grid_plan_path),
            "sha256": sha256_file(grid_plan_path),
            "plan_id": text(plan.get("plan_id"), "grid plan plan_id"),
        },
        "dataset_config": {
            "path": str(config_path),
            "sha256": config_sha,
        },
        "release_contract": {
            "path": str(release_path),
            "sha256": release_sha,
        },
        "qualification_scope": qualification_scope,
        "targets": list(targets),
        "datasets": sorted({group.dataset for group in groups}),
        "database_isolation_evidence_complete": all(
            group.require_database_isolation_evidence for group in groups
        ),
        "database_isolation_recoveries": [
            {
                "dataset": group.dataset,
                "mode": group.mode,
                "reason": group.isolation_recovery_reason,
            }
            for group in groups
            if not group.require_database_isolation_evidence
        ],
        "groups": len(groups),
        "cell_count": len(cells),
        "cells": cells,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(payload, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-plan",
        type=Path,
        default=DEFAULT_GRID_PLAN,
        help="Machine-readable isolated grid plan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Required-grid contract JSON to write atomically.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        contract = build_contract(args.grid_plan)
        atomic_write_json(args.output, contract)
    except (OSError, RequiredGridBuildError, selector.SelectionError) as exc:
        print(f"error: {exc}", file=os.sys.stderr, flush=True)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "groups": contract["groups"],
                "cells": contract["cell_count"],
                "grid_complete": contract["grid_complete"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
