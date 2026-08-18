#!/usr/bin/env python3
"""Build the fail-closed machine-readable Table 6 summary.

The builder joins independently selected fixed-recall pairs to formal q10k/r3
paired latency evidence and formal c16/q10k/r3 throughput evidence.  It never
accepts an unbound CSV, derives QPS from latency, or fabricates values for an
unattainable target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in the minimal test venv
    np = None  # type: ignore[assignment]

try:
    from . import figure5_converter_binding as converter_binding
    from . import figure5_latency_repeats as latency_converter
    from . import pgvector_figure5_throughput as throughput
    from . import run_figure5_matched_latency as matched_latency
    from . import run_figure5_matched_throughput as matched_throughput
    from . import select_figure5_matched_configs as selector
except ImportError:
    import figure5_converter_binding as converter_binding
    import figure5_latency_repeats as latency_converter
    import pgvector_figure5_throughput as throughput
    import run_figure5_matched_latency as matched_latency
    import run_figure5_matched_throughput as matched_throughput
    import select_figure5_matched_configs as selector


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "sqlens_table6_matched_recall_summary"
RUNNER_VERSION = "sqlens-table6-matched-recall-summary-v1"
EXPECTED_REQUESTS = 10_000
EXPECTED_LATENCY_REPEATS = 3
EXPECTED_THROUGHPUT_REPEATS = 3
EXPECTED_CLIENTS = 16
EXPECTED_FILTERS = 14
EXPECTED_CALIBRATION_REQUESTS = 2_800
EXPECTED_CALIBRATION_OBSERVATIONS_PER_FILTER = 200
EXPECTED_GRID_CELLS = 22
EXPECTED_ARMS = ("stock_pgvector", "sqlens_full")
MODE_TO_ARM = {
    "original": "stock_pgvector",
    "design1_bloom_bfs_layout_d3": "sqlens_full",
}
ARM_TO_MODE = {arm: mode for mode, arm in MODE_TO_ARM.items()}
DATASET_IDS = {
    "amazon": "amazon10m",
    "yfcc": "yfcc10m",
    "laion": "laion25m",
}
SELECTED = "selected"
UNATTAINABLE = "unattainable_on_calibration_grid"
BOOTSTRAP_METHOD = (
    "paired_query_cluster_bootstrap_within_filter_then_14_filter_log_ratio_"
    "geomean"
)
QUALIFICATION_SCOPE = "global_min_predicate_lcb"
SUMMARY_FIELDS = (
    "schema_version",
    "dataset",
    "target_recall",
    "pair_id",
    "status",
    "status_detail",
    "stock_selection_config_sha256",
    "sqlens_selection_config_sha256",
    "stock_arm_config_sha256",
    "sqlens_arm_config_sha256",
    "stock_recall",
    "sqlens_recall",
    "stock_mean_latency_ms",
    "sqlens_mean_latency_ms",
    "stock_latency_p95_ms",
    "sqlens_latency_p95_ms",
    "stock_latency_p99_ms",
    "sqlens_latency_p99_ms",
    "stock_qps",
    "sqlens_qps",
    "speedup_geomean",
    "speedup_ci95_low",
    "speedup_ci95_high",
    "wins",
    "wins_denominator",
    "filters",
    "latency_requests_per_arm",
    "latency_repeats",
    "throughput_clients",
    "throughput_requests_per_arm",
    "throughput_repeats",
    "bootstrap_method",
    "bootstrap_samples",
    "bootstrap_seed",
    "release_contract_sha256",
    "dataset_config_sha256",
    "required_grid_contract_sha256",
    "required_grid_cell_keys_sha256",
    "latency_workload_sha256",
    "throughput_workload_sha256",
    "selection_csv_sha256",
    "selection_plan_sha256",
    "selection_manifest_sha256",
    "latency_source_manifest_sha256",
    "throughput_source_manifest_sha256",
    "throughput_protocol_fingerprint_sha256",
)


class Table6SummaryError(RuntimeError):
    """An input cannot be admitted to the formal Table 6 summary."""


@dataclass(frozen=True)
class SelectionPair:
    dataset: str
    dataset_id: str
    target: float
    pair_id: str
    status: str
    stock: Mapping[str, object] | None
    sqlens: Mapping[str, object] | None
    stock_selection_sha: str
    sqlens_selection_sha: str
    stock_arm_sha: str
    sqlens_arm_sha: str
    stock_status: str = SELECTED
    sqlens_status: str = SELECTED


@dataclass(frozen=True)
class SelectionEvidence:
    pairs: tuple[SelectionPair, ...]
    release: Mapping[str, str]
    qualification_scope: str
    bindings: Mapping[str, str]
    config: Mapping[str, str] | None = None
    required_grid: Mapping[str, object] | None = None


@dataclass(frozen=True)
class RequiredGridCell:
    cell_key: str
    dataset: str
    arm: str
    mode: str
    family: str
    ef_search: int
    traversal_guided_target: int | None
    raw_path: Path
    raw_sha256: str
    plan_path: Path
    plan_sha256: str
    runner_manifest_path: Path
    runner_manifest_sha256: str


@dataclass(frozen=True)
class RequiredGridEvidence:
    path: Path
    sha256: str
    config_path: Path
    config_sha256: str
    release_path: Path
    release_sha256: str
    qualification_scope: str
    targets: tuple[float, ...]
    cell_keys_sha256: str
    cells: tuple[RequiredGridCell, ...]


@dataclass(frozen=True)
class CalibrationCellEvidence:
    cell_key: str
    dataset: str
    arm: str
    config_sha256: str
    qualification_lcb95: float


@dataclass
class LatencyPairEvidence:
    dataset: str
    target: float
    pair_id: str
    by_filter: dict[
        str,
        dict[tuple[int, str], dict[str, tuple[float, ...]]],
    ]
    recall_by_arm: dict[str, Sequence[float]]
    latency_by_arm: dict[str, Sequence[float]]
    workload_sha256: str
    filters_sha256: str
    source_manifest_sha256: str


@dataclass(frozen=True)
class ThroughputPairEvidence:
    dataset: str
    target: float
    pair_id: str
    qps_by_arm: Mapping[str, float]
    workload_sha256: str
    source_manifest_sha256: str
    protocol_fingerprint_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Table6SummaryError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Table6SummaryError(f"{label} root is not an object: {path}")
    return value


def read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise Table6SummaryError(f"cannot read {label} {path}: {exc}") from exc
    if not fields:
        raise Table6SummaryError(f"{label} has no header: {path}")
    if any(None in row for row in rows):
        raise Table6SummaryError(f"{label} has a row wider than its header: {path}")
    return fields, rows


def require_fields(
    fields: Sequence[str], required: Sequence[str], label: str
) -> None:
    missing = sorted(set(required) - set(fields))
    if missing:
        raise Table6SummaryError(f"{label} is missing fields {missing}")


def require_sha(value: object, label: str) -> str:
    text = str(value or "").strip().lower()
    if not throughput.SHA256_RE.fullmatch(text):
        raise Table6SummaryError(f"{label} is not a SHA-256 value")
    return text


def require_float(
    value: object,
    label: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise Table6SummaryError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise Table6SummaryError(f"{label} is not finite")
    if lower is not None and number < lower:
        raise Table6SummaryError(f"{label} is below {lower}: {number}")
    if upper is not None and number > upper:
        raise Table6SummaryError(f"{label} exceeds {upper}: {number}")
    return number


def require_int(value: object, label: str, *, lower: int = 0) -> int:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise Table6SummaryError(f"{label} is not an integer: {value!r}") from exc
    if number < lower:
        raise Table6SummaryError(f"{label} is below {lower}: {number}")
    return number


def require_true(value: object, label: str) -> None:
    if value is True:
        return
    if str(value).strip().lower() in {"true", "1", "t", "yes"}:
        return
    raise Table6SummaryError(f"{label} is not true")


def resolve_bound_path(value: object, *, base: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise Table6SummaryError(f"{label} path is empty")
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise Table6SummaryError(f"{label} is missing: {path}")
    return path


def audit_file_binding(
    binding: Mapping[str, Any],
    *,
    base: Path,
    label: str,
    path_key: str = "path",
    sha_key: str = "sha256",
) -> tuple[Path, str]:
    path = resolve_bound_path(binding.get(path_key), base=base, label=label)
    expected = require_sha(binding.get(sha_key), f"{label} SHA")
    observed = sha256_file(path)
    if observed != expected:
        raise Table6SummaryError(
            f"{label} SHA drifted: expected={expected}, observed={observed}"
        )
    return path, observed


def load_required_grid_contract(path: Path) -> RequiredGridEvidence:
    path = path.resolve()
    payload = read_json(path, "required-grid contract")
    if (
        payload.get("schema_version") != 1
        or payload.get("contract_type")
        != selector.REQUIRED_GRID_CONTRACT_TYPE
        or payload.get("grid_complete") is not True
    ):
        raise Table6SummaryError(
            "required-grid contract is not a complete formal grid"
        )
    if payload.get("qualification_scope") != QUALIFICATION_SCOPE:
        raise Table6SummaryError(
            "required-grid qualification scope is not "
            "global_min_predicate_lcb"
        )
    targets_value = payload.get("targets")
    if not isinstance(targets_value, list) or not targets_value:
        raise Table6SummaryError("required-grid targets are missing")
    targets = tuple(
        require_float(
            value,
            "required-grid target",
            lower=0.0,
            upper=1.0,
        )
        for value in targets_value
    )
    if targets != tuple(sorted(set(targets))):
        raise Table6SummaryError(
            "required-grid targets must be unique and ascending"
        )

    source_grid = payload.get("source_grid_plan")
    config_binding = payload.get("dataset_config")
    release_binding = payload.get("release_contract")
    if not all(
        isinstance(binding, Mapping)
        for binding in (source_grid, config_binding, release_binding)
    ):
        raise Table6SummaryError(
            "required-grid source/config/release bindings are incomplete"
        )
    assert isinstance(source_grid, Mapping)
    assert isinstance(config_binding, Mapping)
    assert isinstance(release_binding, Mapping)
    audit_file_binding(
        source_grid,
        base=path.parent,
        label="required-grid source plan",
    )
    config_path, config_sha = audit_file_binding(
        config_binding,
        base=path.parent,
        label="required-grid dataset config",
    )
    release_path, release_sha = audit_file_binding(
        release_binding,
        base=path.parent,
        label="required-grid release contract",
    )
    config = read_json(config_path, "required-grid dataset config")
    config_release = resolve_bound_path(
        config.get("release_contract"),
        base=config_path.parent,
        label="required-grid config release contract",
    )
    if config_release != release_path:
        raise Table6SummaryError(
            "required-grid dataset config binds a different release"
        )

    cells_value = payload.get("cells")
    if (
        not isinstance(cells_value, list)
        or require_int(
            payload.get("cell_count"), "required-grid cell_count"
        )
        != len(cells_value)
        or len(cells_value) != EXPECTED_GRID_CELLS
        or require_int(payload.get("groups"), "required-grid groups") != 6
    ):
        raise Table6SummaryError(
            f"required-grid must contain exactly {EXPECTED_GRID_CELLS} cells "
            "from six serial groups"
        )
    cells: list[RequiredGridCell] = []
    seen_keys: set[str] = set()
    seen_raw: set[Path] = set()
    manifest_shas: dict[Path, str] = {}
    for index, value in enumerate(cells_value):
        label = f"required-grid cell {index}"
        if not isinstance(value, Mapping):
            raise Table6SummaryError(f"{label} is malformed")
        cell_key = str(value.get("cell_key") or "").strip()
        dataset = str(value.get("dataset") or "").strip()
        arm = str(value.get("arm") or "").strip()
        mode = str(value.get("mode") or "").strip()
        family = str(value.get("family") or "").strip()
        if (
            not cell_key
            or dataset not in DATASET_IDS
            or arm not in {"stock", "sqlens"}
            or mode
            != (
                selector.MODE_STOCK
                if arm == "stock"
                else selector.MODE_SQLENS
            )
            or family
            != (
                selector.FAMILY_STOCK_STRICT
                if arm == "stock"
                else selector.FAMILY_SQLENS_TARGET
            )
        ):
            raise Table6SummaryError(f"{label} identity is invalid")
        ef_search = require_int(
            value.get("ef_search"), f"{label} ef_search", lower=1
        )
        traversal_value = value.get("traversal_guided_target")
        traversal_target = (
            None
            if traversal_value is None
            else require_int(
                traversal_value,
                f"{label} traversal target",
                lower=11,
            )
        )
        if (arm == "stock") != (traversal_target is None):
            raise Table6SummaryError(
                f"{label} traversal target does not match its arm"
            )
        raw_binding = value.get("raw_csv")
        plan_binding = value.get("input_plan")
        runner_binding = value.get("serial_runner_manifest")
        if not all(
            isinstance(binding, Mapping)
            for binding in (raw_binding, plan_binding, runner_binding)
        ):
            raise Table6SummaryError(f"{label} artifact bindings are missing")
        assert isinstance(raw_binding, Mapping)
        assert isinstance(plan_binding, Mapping)
        assert isinstance(runner_binding, Mapping)
        raw_path, raw_sha = audit_file_binding(
            raw_binding,
            base=path.parent,
            label=f"{label} raw CSV",
        )
        plan_path, plan_sha = audit_file_binding(
            plan_binding,
            base=path.parent,
            label=f"{label} input plan",
        )
        runner_path, runner_sha = audit_file_binding(
            runner_binding,
            base=path.parent,
            label=f"{label} serial runner manifest",
        )
        if (
            plan_path
            != raw_path.with_name(raw_path.name + ".plan.json")
            or cell_key != selector.calibration_cell_key(raw_path)
            or cell_key in seen_keys
            or raw_path in seen_raw
        ):
            raise Table6SummaryError(
                f"{label} has duplicate or noncanonical path/cell_key"
            )
        previous_runner_sha = manifest_shas.setdefault(
            runner_path, runner_sha
        )
        if previous_runner_sha != runner_sha:
            raise Table6SummaryError(
                "required-grid binds one runner manifest to multiple SHAs"
            )
        seen_keys.add(cell_key)
        seen_raw.add(raw_path)
        cells.append(
            RequiredGridCell(
                cell_key=cell_key,
                dataset=dataset,
                arm=arm,
                mode=mode,
                family=family,
                ef_search=ef_search,
                traversal_guided_target=traversal_target,
                raw_path=raw_path,
                raw_sha256=raw_sha,
                plan_path=plan_path,
                plan_sha256=plan_sha,
                runner_manifest_path=runner_path,
                runner_manifest_sha256=runner_sha,
            )
        )
    if len(manifest_shas) != 6:
        raise Table6SummaryError(
            "required-grid does not bind exactly six serial runner manifests"
        )
    ordered = tuple(sorted(cells, key=lambda item: item.cell_key))
    return RequiredGridEvidence(
        path=path,
        sha256=sha256_file(path),
        config_path=config_path,
        config_sha256=config_sha,
        release_path=release_path,
        release_sha256=release_sha,
        qualification_scope=QUALIFICATION_SCOPE,
        targets=targets,
        cell_keys_sha256=sha256_json(
            [cell.cell_key for cell in ordered]
        ),
        cells=ordered,
    )


def _same_target(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise Table6SummaryError("cannot summarize an empty numeric sample")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def expected_arm_shas(
    pair_id: str,
    target: float,
    stock: Mapping[str, object],
    sqlens: Mapping[str, object],
) -> tuple[str, str]:
    selected = matched_latency.SelectedPair(
        pair_id=pair_id,
        dataset="unused",
        target_recall=target,
        stock=dict(stock),
        sqlens=dict(sqlens),
    )
    settings = matched_throughput.expected_search_settings(selected)
    return (
        throughput.arm_config_sha256(settings, "stock_pgvector"),
        throughput.arm_config_sha256(settings, "sqlens_full"),
    )


def _required_grid_binding_gate(
    binding: object,
    required_grid: RequiredGridEvidence,
    *,
    label: str,
) -> None:
    if not isinstance(binding, Mapping):
        raise Table6SummaryError(f"{label} has no required-grid binding")
    path = resolve_bound_path(
        binding.get("path"),
        base=required_grid.path.parent,
        label=f"{label} required-grid contract",
    )
    if (
        path != required_grid.path
        or require_sha(
            binding.get("sha256"), f"{label} required-grid SHA"
        )
        != required_grid.sha256
        or require_int(
            binding.get("cell_count"), f"{label} required-grid cell count"
        )
        != len(required_grid.cells)
    ):
        raise Table6SummaryError(
            f"{label} required-grid path/SHA/cell count differs"
        )
    keys = binding.get("cell_keys")
    if not isinstance(keys, list) or sorted(str(key) for key in keys) != [
        cell.cell_key for cell in required_grid.cells
    ]:
        raise Table6SummaryError(
            f"{label} required-grid cell keys differ"
        )
    runner_bindings = binding.get("serial_runner_manifests")
    expected_runners = {
        (
            cell.runner_manifest_path,
            cell.runner_manifest_sha256,
        )
        for cell in required_grid.cells
    }
    if not isinstance(runner_bindings, list):
        raise Table6SummaryError(
            f"{label} has no serial runner manifest bindings"
        )
    observed_runners: set[tuple[Path, str]] = set()
    for index, runner in enumerate(runner_bindings):
        if not isinstance(runner, Mapping):
            raise Table6SummaryError(
                f"{label} runner binding {index} is malformed"
            )
        runner_path = resolve_bound_path(
            runner.get("path"),
            base=required_grid.path.parent,
            label=f"{label} runner manifest {index}",
        )
        runner_sha = require_sha(
            runner.get("sha256"),
            f"{label} runner manifest {index} SHA",
        )
        if sha256_file(runner_path) != runner_sha:
            raise Table6SummaryError(
                f"{label} runner manifest {index} SHA drifted"
            )
        observed_runners.add((runner_path, runner_sha))
    if observed_runners != expected_runners:
        raise Table6SummaryError(
            f"{label} serial runner manifest set differs"
        )


def _selection_input_map(
    plan: Mapping[str, Any],
    required_grid: RequiredGridEvidence,
) -> dict[Path, Mapping[str, Any]]:
    inputs = plan.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != len(
        required_grid.cells
    ):
        raise Table6SummaryError(
            "selection plan inputs do not exactly cover required-grid cells"
        )
    result: dict[Path, Mapping[str, Any]] = {}
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping):
            raise Table6SummaryError(
                f"selection input {index} is malformed"
            )
        raw_path = resolve_bound_path(
            item.get("raw_csv"),
            base=required_grid.path.parent,
            label=f"selection input {index} raw CSV",
        )
        if raw_path in result:
            raise Table6SummaryError(
                f"selection repeats calibration input {raw_path}"
            )
        result[raw_path] = item
    expected = {cell.raw_path for cell in required_grid.cells}
    if set(result) != expected:
        raise Table6SummaryError(
            "selection input set is not the complete required grid"
        )
    return result


def _audit_calibration_cell(
    cell: RequiredGridCell,
    selection_input: Mapping[str, Any],
    release: Mapping[str, str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> CalibrationCellEvidence:
    label = f"required-grid cell {cell.cell_key}"
    if (
        require_sha(
            selection_input.get("raw_csv_sha256"),
            f"{label} selection raw SHA",
        )
        != cell.raw_sha256
        or resolve_bound_path(
            selection_input.get("input_plan"),
            base=cell.plan_path.parent,
            label=f"{label} selection input plan",
        )
        != cell.plan_path
        or require_sha(
            selection_input.get("input_plan_sha256"),
            f"{label} selection plan SHA",
        )
        != cell.plan_sha256
        or str(selection_input.get("dataset") or "") != cell.dataset
        or str(selection_input.get("family") or "") != cell.family
        or str(selection_input.get("mode") or "") != cell.mode
    ):
        raise Table6SummaryError(
            f"{label} selection input binding differs from required grid"
        )
    fields, rows = read_csv(cell.raw_path, f"{label} raw CSV")
    required_fields = {
        "mode",
        "error",
        "recall",
        "request_no",
        "query_id",
        "filter_name",
        "sqlens_build_id",
        "vector_so_sha256",
        *selector.CONFIG_FIELDS,
    }
    require_fields(fields, sorted(required_fields), f"{label} raw CSV")
    if len(rows) != EXPECTED_CALIBRATION_REQUESTS:
        raise Table6SummaryError(
            f"{label} has {len(rows)} calibration rows, expected "
            f"{EXPECTED_CALIBRATION_REQUESTS}"
        )
    filters: list[str] = []
    query_ids: list[str] = []
    recalls: list[float] = []
    request_keys: set[tuple[int, str, str]] = set()
    filter_counts: dict[str, int] = {}
    config_shas: set[str] = set()
    normalized_config: Mapping[str, object] | None = None
    for row_no, row in enumerate(rows, start=2):
        row_label = f"{label} row {row_no}"
        if (
            row.get("mode") != cell.mode
            or str(row.get("error") or "").strip()
            or row.get("sqlens_build_id")
            != release["expected_sqlens_build_id"]
            or row.get("vector_so_sha256")
            != release["expected_vector_so_sha256"]
        ):
            raise Table6SummaryError(
                f"{row_label} mode/error/release identity is invalid"
            )
        request_key = (
            require_int(
                row.get("request_no"),
                f"{row_label} request_no",
            ),
            str(row.get("query_id") or "").strip(),
            str(row.get("filter_name") or "").strip(),
        )
        if (
            not request_key[1]
            or not request_key[2]
            or request_key in request_keys
        ):
            raise Table6SummaryError(
                f"{row_label} request identity is empty or duplicated"
            )
        request_keys.add(request_key)
        recall = require_float(
            row.get("recall"),
            f"{row_label} recall",
            lower=0.0,
            upper=1.0,
        )
        recalls.append(recall)
        filters.append(request_key[2])
        query_ids.append(request_key[1])
        filter_counts[request_key[2]] = (
            filter_counts.get(request_key[2], 0) + 1
        )
        try:
            normalized = selector._normalized_config(row, row_label)
            config_shas.add(selector.sha256_json(normalized))
            normalized_config = normalized
        except selector.SelectionError as exc:
            raise Table6SummaryError(
                f"{row_label} calibration config is invalid: {exc}"
            ) from exc
    if (
        len(request_keys) != EXPECTED_CALIBRATION_REQUESTS
        or len(filter_counts) != EXPECTED_FILTERS
        or set(filter_counts.values())
        != {EXPECTED_CALIBRATION_OBSERVATIONS_PER_FILTER}
        or len(config_shas) != 1
        or normalized_config is None
    ):
        raise Table6SummaryError(
            f"{label} does not satisfy q2800/14x200/config-stability"
        )
    if (
        int(normalized_config["ef_search"]) != cell.ef_search
        or (
            cell.arm == "stock"
            and normalized_config["iterative_scan"] != "strict_order"
        )
        or (
            cell.arm == "sqlens"
            and (
                normalized_config["iterative_scan"] != "off"
                or int(normalized_config["guided_collect_target"])
                != cell.ef_search
                or int(normalized_config["traversal_guided_target"])
                != cell.traversal_guided_target
            )
        )
    ):
        raise Table6SummaryError(
            f"{label} raw config differs from required-grid search setting"
        )
    config_sha = next(iter(config_shas))
    if require_sha(
        selection_input.get("config_sha256"),
        f"{label} selection config SHA",
    ) != config_sha:
        raise Table6SummaryError(
            f"{label} selection config SHA differs from raw CSV"
        )
    try:
        _, _, _, _, qualification_lcb = selector._bootstrap_metrics(
            recalls,
            filters,
            query_ids,
            samples=bootstrap_samples,
            seed=(
                bootstrap_seed
                + cell.ef_search
                + (0 if cell.arm == "stock" else 100_000)
            ),
            require_formal_cartesian=True,
        )
    except selector.SelectionError as exc:
        raise Table6SummaryError(
            f"{label} LCB calculation failed: {exc}"
        ) from exc
    return CalibrationCellEvidence(
        cell_key=cell.cell_key,
        dataset=cell.dataset,
        arm=cell.arm,
        config_sha256=config_sha,
        qualification_lcb95=qualification_lcb,
    )


def audit_selection_grid_and_exhaustion(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    pairs: Sequence[SelectionPair],
    required_grid: RequiredGridEvidence,
    release: Mapping[str, str],
) -> tuple[CalibrationCellEvidence, ...]:
    _required_grid_binding_gate(
        plan.get("required_grid_contract"),
        required_grid,
        label="selection plan",
    )
    _required_grid_binding_gate(
        manifest.get("required_grid_contract"),
        required_grid,
        label="selection manifest",
    )
    if (
        required_grid.release_sha256 != release["sha256"]
        or required_grid.release_path != Path(release["path"]).resolve()
        or required_grid.qualification_scope != QUALIFICATION_SCOPE
    ):
        raise Table6SummaryError(
            "selection release/scope differs from required grid"
        )
    if {
        (pair.dataset, pair.target) for pair in pairs
    } != {
        (dataset, target)
        for dataset in DATASET_IDS
        for target in required_grid.targets
    }:
        raise Table6SummaryError(
            "selection targets do not cover every required-grid dataset/target"
        )

    bootstrap = plan.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise Table6SummaryError("selection plan has no bootstrap contract")
    bootstrap_samples = require_int(
        bootstrap.get("samples"), "selection bootstrap samples", lower=100
    )
    bootstrap_seed = require_int(
        bootstrap.get("seed"), "selection bootstrap seed"
    )
    inputs = _selection_input_map(plan, required_grid)
    evidence = tuple(
        _audit_calibration_cell(
            cell,
            inputs[cell.raw_path],
            release,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for cell in required_grid.cells
    )

    proof = plan.get("exhaustion_proof")
    manifest_proof = manifest.get("exhaustion_proof")
    if not isinstance(proof, Mapping) or manifest_proof != proof:
        raise Table6SummaryError(
            "selection plan/manifest exhaustion proofs differ or are missing"
        )
    proof_body = dict(proof)
    proof_sha = proof_body.pop("proof_sha256", None)
    proof_targets = proof.get("targets")
    if not isinstance(proof_targets, list):
        raise Table6SummaryError(
            "selection exhaustion proof targets are malformed"
        )
    if (
        require_sha(proof_sha, "selection exhaustion proof SHA")
        != sha256_json(proof_body)
        or proof.get("required_grid_contract_present") is not True
        or proof.get("required_grid_complete") is not True
        or proof.get("input_set_exact") is not True
        or require_sha(
            proof.get("required_grid_contract_sha256"),
            "selection exhaustion required-grid SHA",
        )
        != required_grid.sha256
        or require_sha(
            proof.get("required_grid_cell_keys_sha256"),
            "selection exhaustion cell-key SHA",
        )
        != required_grid.cell_keys_sha256
        or proof.get("qualification_scope") != QUALIFICATION_SCOPE
        or tuple(
            require_float(
                value,
                "selection exhaustion target",
                lower=0.0,
                upper=1.0,
            )
            for value in proof_targets
        )
        != required_grid.targets
    ):
        raise Table6SummaryError(
            "selection exhaustion proof does not bind the complete grid"
        )

    candidates: dict[tuple[str, str], list[CalibrationCellEvidence]] = {}
    for item in evidence:
        candidates.setdefault((item.dataset, item.arm), []).append(item)
    proof_items = proof.get("unattainable_arms")
    if not isinstance(proof_items, list):
        raise Table6SummaryError(
            "selection exhaustion proof has no arm evidence"
        )
    proof_map: dict[tuple[str, float, str], Mapping[str, Any]] = {}
    for index, item in enumerate(proof_items):
        if not isinstance(item, Mapping):
            raise Table6SummaryError(
                f"selection exhaustion arm {index} is malformed"
            )
        key = (
            str(item.get("dataset") or ""),
            require_float(
                item.get("target_recall"),
                f"selection exhaustion arm {index} target",
                lower=0.0,
                upper=1.0,
            ),
            str(item.get("arm") or ""),
        )
        if key in proof_map:
            raise Table6SummaryError(
                f"selection exhaustion repeats arm/target {key!r}"
            )
        proof_map[key] = item

    expected_unattainable: set[tuple[str, float, str]] = set()
    measurement_pairs = plan.get("measurement_pairs")
    if not isinstance(measurement_pairs, list):
        raise Table6SummaryError(
            "selection plan has no measurement pair evidence"
        )
    measurement_map: dict[tuple[str, float], Mapping[str, Any]] = {}
    for index, row in enumerate(measurement_pairs):
        if not isinstance(row, Mapping):
            raise Table6SummaryError(
                f"selection measurement pair {index} is malformed"
            )
        key = (
            str(row.get("dataset") or ""),
            require_float(
                row.get("target_recall"),
                f"selection measurement pair {index} target",
                lower=0.0,
                upper=1.0,
            ),
        )
        if key in measurement_map:
            raise Table6SummaryError(
                f"selection repeats measurement pair {key!r}"
            )
        measurement_map[key] = row
    if len(measurement_map) != len(pairs):
        raise Table6SummaryError(
            "selection measurement-pair coverage is incomplete"
        )
    for pair in pairs:
        row = measurement_map.get((pair.dataset, pair.target))
        if row is None:
            raise Table6SummaryError(
                f"selection plan lacks measurement pair {pair.pair_id}"
            )
        if (
            row.get("pair_id") != pair.pair_id
            or row.get("selection_status") != pair.status
            or row.get("stock_status") != pair.stock_status
            or row.get("sqlens_status") != pair.sqlens_status
        ):
            raise Table6SummaryError(
                f"selection plan identity/status differs for {pair.pair_id}"
            )
        for arm, arm_status in (
            ("stock", pair.stock_status),
            ("sqlens", pair.sqlens_status),
        ):
            arm_candidates = candidates.get((pair.dataset, arm), [])
            if not arm_candidates:
                raise Table6SummaryError(
                    f"required grid lacks {pair.dataset}/{arm} candidates"
                )
            maximum = max(
                item.qualification_lcb95 for item in arm_candidates
            )
            if arm_status == SELECTED:
                config_sha = require_sha(
                    row.get(f"{arm}_config_sha256"),
                    f"{pair.pair_id}/{arm} selected config SHA",
                )
                matching = [
                    item
                    for item in arm_candidates
                    if item.config_sha256 == config_sha
                ]
                if len(matching) != 1:
                    raise Table6SummaryError(
                        f"{pair.pair_id}/{arm} selected config is not unique "
                        "on the required grid"
                    )
                selected_lcb = require_float(
                    row.get(
                        f"{arm}_calibration_per_filter_recall_min_ci95_low"
                    ),
                    f"{pair.pair_id}/{arm} selected LCB",
                    lower=0.0,
                    upper=1.0,
                )
                if (
                    not math.isclose(
                        selected_lcb,
                        matching[0].qualification_lcb95,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or selected_lcb + 1e-12 < pair.target
                ):
                    raise Table6SummaryError(
                        f"{pair.pair_id}/{arm} selected LCB evidence is invalid"
                    )
            elif arm_status == UNATTAINABLE:
                key = (pair.dataset, pair.target, arm)
                expected_unattainable.add(key)
                item = proof_map.get(key)
                candidate_shas = sorted(
                    candidate.config_sha256
                    for candidate in arm_candidates
                )
                if (
                    item is None
                    or item.get("status")
                    != "unattainable_on_complete_required_grid"
                    or require_int(
                        item.get("candidate_configs"),
                        f"{pair.pair_id}/{arm} candidate count",
                    )
                    != len(arm_candidates)
                    or sorted(
                        str(value)
                        for value in item.get(
                            "candidate_config_sha256s", ()
                        )
                    )
                    != candidate_shas
                    or not math.isclose(
                        require_float(
                            item.get("maximum_qualification_floor"),
                            f"{pair.pair_id}/{arm} maximum LCB",
                            lower=0.0,
                            upper=1.0,
                        ),
                        maximum,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or maximum + 1e-12 >= pair.target
                ):
                    raise Table6SummaryError(
                        f"{pair.pair_id}/{arm} exhaustion LCB evidence is invalid"
                    )
            else:
                raise Table6SummaryError(
                    f"{pair.pair_id}/{arm} has unsupported arm status"
                )
        expected_pair_status = (
            SELECTED
            if pair.stock_status == pair.sqlens_status == SELECTED
            else UNATTAINABLE
        )
        if pair.status != expected_pair_status:
            raise Table6SummaryError(
                f"{pair.pair_id} pair status disagrees with arm statuses"
            )
    if set(proof_map) != expected_unattainable:
        raise Table6SummaryError(
            "selection exhaustion proof arm set differs from recomputed grid"
        )
    return evidence


def load_selection(
    selection_csv: Path,
    selection_plan: Path,
    selection_manifest: Path,
    required_grid: RequiredGridEvidence,
) -> SelectionEvidence:
    selection_csv = selection_csv.resolve()
    selection_plan = selection_plan.resolve()
    selection_manifest = selection_manifest.resolve()
    plan = read_json(selection_plan, "selection plan")
    manifest = read_json(selection_manifest, "selection manifest")
    release_value = plan.get("release_contract")
    if not isinstance(release_value, Mapping):
        raise Table6SummaryError("selection plan has no release contract")
    release_path, release_sha = audit_file_binding(
        release_value,
        base=selection_plan.parent,
        label="selection release contract",
    )
    release = {
        "path": str(release_path),
        "sha256": release_sha,
        "contract_id": str(release_value.get("contract_id") or ""),
        "expected_sqlens_build_id": str(
            release_value.get("expected_sqlens_build_id") or ""
        ),
        "expected_vector_so_sha256": require_sha(
            release_value.get("expected_vector_so_sha256"),
            "selection expected vector.so SHA",
        ),
    }
    if not release["contract_id"] or not release["expected_sqlens_build_id"]:
        raise Table6SummaryError("selection release identity is incomplete")
    config = {
        "release_identity": {
            "expected_sqlens_build_id": release[
                "expected_sqlens_build_id"
            ],
            "expected_vector_so_sha256": release[
                "expected_vector_so_sha256"
            ],
        },
        "release_contract_sha256": release_sha,
    }
    try:
        bindings = matched_latency.validate_selection_artifacts(
            selection_csv, selection_plan, selection_manifest, config
        )
    except matched_latency.MatchedLatencyError as exc:
        raise Table6SummaryError(f"selection audit failed: {exc}") from exc
    if bindings.get("target_policy") != "fixed":
        raise Table6SummaryError("Table 6 requires selector target_policy=fixed")
    scope = str(bindings.get("qualification_scope") or "")
    if scope != matched_latency.QUALIFICATION_SCOPE_FORMAL:
        raise Table6SummaryError(
            "Table 6 requires qualification_scope=global_min_predicate_lcb"
        )

    fields, rows = read_csv(selection_csv, "selection CSV")
    require_fields(
        fields,
        (
            "pair_id",
            "qualification_scope",
            "dataset",
            "target_recall",
            "selection_status",
            "stock_status",
            "sqlens_status",
        ),
        "selection CSV",
    )
    if len(rows) != int(bindings.get("target_rows") or -1):
        raise Table6SummaryError("selection CSV row count differs from selector plan")

    expected_targets = {
        (str(dataset), float(target))
        for dataset, targets in dict(
            bindings.get("targets_by_dataset") or {}
        ).items()
        for target in targets
    }
    result: list[SelectionPair] = []
    seen: set[tuple[str, float]] = set()
    for row_no, row in enumerate(rows, start=2):
        dataset = str(row.get("dataset") or "").strip()
        if dataset not in DATASET_IDS:
            raise Table6SummaryError(
                f"selection row {row_no} has unknown dataset {dataset!r}"
            )
        target = require_float(
            row.get("target_recall"),
            f"selection row {row_no} target",
            lower=0.0,
            upper=1.0,
        )
        key = (dataset, target)
        if key in seen:
            raise Table6SummaryError(f"selection repeats dataset/target {key!r}")
        seen.add(key)
        status = str(row.get("selection_status") or "").strip()
        stock_status = str(row.get("stock_status") or "").strip()
        sqlens_status = str(row.get("sqlens_status") or "").strip()
        pair_id = str(row.get("pair_id") or "").strip()
        if not pair_id:
            raise Table6SummaryError(
                f"selection row {row_no} has no pair_id"
            )
        if status == SELECTED:
            if stock_status != SELECTED or sqlens_status != SELECTED:
                raise Table6SummaryError(
                    f"selected row {row_no} lacks two selected arms or pair_id"
                )
            try:
                stock = matched_latency.arm_config(row, "stock")
                sqlens = matched_latency.arm_config(row, "sqlens")
            except matched_latency.MatchedLatencyError as exc:
                raise Table6SummaryError(
                    f"selected row {row_no} config is invalid: {exc}"
                ) from exc
            stock_arm_sha, sqlens_arm_sha = expected_arm_shas(
                pair_id, target, stock, sqlens
            )
            stock_selection_sha = require_sha(
                row.get("stock_config_sha256"),
                f"selection row {row_no} Stock config SHA",
            )
            sqlens_selection_sha = require_sha(
                row.get("sqlens_config_sha256"),
                f"selection row {row_no} SQLens config SHA",
            )
        elif status == UNATTAINABLE:
            if stock_status not in {SELECTED, UNATTAINABLE} or sqlens_status not in {
                SELECTED,
                UNATTAINABLE,
            }:
                raise Table6SummaryError(
                    f"unattainable row {row_no} has invalid arm status"
                )
            if stock_status == SELECTED and sqlens_status == SELECTED:
                raise Table6SummaryError(
                    f"unattainable row {row_no} has two selected arms"
                )
            stock = sqlens = None
            stock_selection_sha = sqlens_selection_sha = ""
            stock_arm_sha = sqlens_arm_sha = ""
        else:
            raise Table6SummaryError(
                f"selection row {row_no} has unsupported status {status!r}"
            )
        result.append(
            SelectionPair(
                dataset=dataset,
                dataset_id=DATASET_IDS[dataset],
                target=target,
                pair_id=pair_id,
                status=status,
                stock=stock,
                sqlens=sqlens,
                stock_selection_sha=stock_selection_sha,
                sqlens_selection_sha=sqlens_selection_sha,
                stock_arm_sha=stock_arm_sha,
                sqlens_arm_sha=sqlens_arm_sha,
                stock_status=stock_status,
                sqlens_status=sqlens_status,
            )
        )
    if expected_targets and {
        (dataset, target) for dataset, target in seen
    } != expected_targets:
        raise Table6SummaryError(
            "selection CSV does not exactly cover targets_by_dataset"
        )
    pairs = tuple(
        sorted(result, key=lambda item: (item.dataset, item.target))
    )
    audit_selection_grid_and_exhaustion(
        plan,
        manifest,
        pairs,
        required_grid,
        release,
    )
    return SelectionEvidence(
        pairs=pairs,
        release=release,
        qualification_scope=scope,
        bindings={
            "selection_csv_sha256": require_sha(
                bindings["selection_csv_sha256"], "selection CSV SHA"
            ),
            "selection_plan_sha256": require_sha(
                bindings["selection_plan_sha256"], "selection plan SHA"
            ),
            "selection_manifest_sha256": require_sha(
                bindings["selection_manifest_sha256"],
                "selection manifest SHA",
            ),
        },
        config={
            "path": str(required_grid.config_path),
            "sha256": required_grid.config_sha256,
        },
        required_grid={
            "path": str(required_grid.path),
            "sha256": required_grid.sha256,
            "cell_keys_sha256": required_grid.cell_keys_sha256,
        },
    )


def audit_converter_binding(
    output: Path,
    binding_path: Path,
    *,
    experiment_kind: str,
    service_output: Path | None = None,
) -> tuple[Path, str, Mapping[str, Any]]:
    output = output.resolve()
    binding_path = binding_path.resolve()
    binding = read_json(binding_path, f"{experiment_kind} converter binding")
    if (
        binding.get("artifact_type")
        != "sqlens_figure5_converter_binding"
        or binding.get("experiment_kind") != experiment_kind
        or binding.get("status") != "complete"
    ):
        raise Table6SummaryError(
            f"{experiment_kind} converter binding identity is invalid"
        )
    for field in (
        "artifact_valid",
        "requested_slice_complete",
        "full_release_complete",
        "paper_eligible",
    ):
        if binding.get(field) is not True:
            raise Table6SummaryError(
                f"{experiment_kind} converter binding failed {field}"
            )
    converter = binding.get("converter_binding")
    if not isinstance(converter, Mapping):
        raise Table6SummaryError(
            f"{experiment_kind} converter binding has no converter_binding"
        )
    output_binding = converter.get("output")
    source_binding = converter.get("source_manifest")
    if not isinstance(output_binding, Mapping) or not isinstance(
        source_binding, Mapping
    ):
        raise Table6SummaryError(
            f"{experiment_kind} converter binding is incomplete"
        )
    bound_output, _ = audit_file_binding(
        output_binding,
        base=binding_path.parent,
        label=f"{experiment_kind} converter output",
    )
    if bound_output != output:
        raise Table6SummaryError(
            f"{experiment_kind} converter binds a different output"
        )
    source, source_sha = audit_file_binding(
        source_binding,
        base=binding_path.parent,
        label=f"{experiment_kind} source manifest",
    )
    if service_output is not None:
        service = binding.get("service_aggregate")
        if not isinstance(service, Mapping):
            raise Table6SummaryError(
                "throughput converter binding has no service aggregate"
            )
        bound_service, _ = audit_file_binding(
            service,
            base=binding_path.parent,
            label="throughput service aggregate",
        )
        if bound_service != service_output.resolve():
            raise Table6SummaryError(
                "throughput converter binds a different service aggregate"
            )
        if service.get("qps_source") != throughput.THROUGHPUT_SOURCE:
            raise Table6SummaryError("throughput service QPS source is invalid")
        if service.get("qps_from_latency_forbidden") is not True:
            raise Table6SummaryError(
                "throughput service does not forbid latency-derived QPS"
            )
    return source, source_sha, binding


def _manifest_selector_gate(
    manifest: Mapping[str, Any],
    selection: SelectionEvidence,
    *,
    label: str,
) -> None:
    selector = manifest.get("selector")
    if not isinstance(selector, Mapping):
        raise Table6SummaryError(f"{label} has no selector binding")
    for field, expected in selection.bindings.items():
        observed = require_sha(selector.get(field), f"{label} {field}")
        if observed != expected:
            raise Table6SummaryError(
                f"{label} {field} does not match the supplied selector"
            )


def _manifest_release_gate(
    release: Mapping[str, Any],
    selection: SelectionEvidence,
    *,
    label: str,
) -> None:
    expected = selection.release
    checks = (
        ("sha256", expected["sha256"]),
        ("contract_id", expected["contract_id"]),
        (
            "expected_sqlens_build_id",
            expected["expected_sqlens_build_id"],
        ),
        (
            "expected_vector_so_sha256",
            expected["expected_vector_so_sha256"],
        ),
    )
    for field, expected_value in checks:
        if str(release.get(field) or "") != expected_value:
            raise Table6SummaryError(
                f"{label} release {field} differs from selection"
            )


def _manifest_config_gate(
    binding: object,
    selection: SelectionEvidence,
    *,
    base: Path,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(selection.config, Mapping):
        raise Table6SummaryError(
            "selection evidence has no required-grid dataset config"
        )
    if not isinstance(binding, Mapping):
        raise Table6SummaryError(f"{label} has no frontier config binding")
    path, observed_sha = audit_file_binding(
        binding,
        base=base,
        label=f"{label} frontier config",
    )
    expected_path = Path(str(selection.config["path"])).resolve()
    expected_sha = require_sha(
        selection.config["sha256"], "selection dataset config SHA"
    )
    if path != expected_path or observed_sha != expected_sha:
        raise Table6SummaryError(
            f"{label} frontier config differs from required grid"
        )
    return path, observed_sha


def _compare_latency_repeat_rows(
    repeat_csv: Path, converted: Sequence[Mapping[str, object]]
) -> None:
    fields, rows = read_csv(repeat_csv, "latency repeat CSV")
    required = (
        "dataset",
        "experiment_kind",
        "arm_id",
        "config_id",
        "config_sha256",
        "release_identity_sha256",
        "clients",
        "repeat_id",
        "request_trace_sha256",
        "requests",
        "unique_queries",
        "completed_queries",
        "error_count",
        "recall_mean",
        "latency_mean_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "status",
    )
    require_fields(fields, required, "latency repeat CSV")
    if len(rows) != len(converted):
        raise Table6SummaryError(
            "latency repeat CSV row count differs from source conversion"
        )
    keys = ("dataset", "arm_id", "config_id", "repeat_id")
    observed = {tuple(row[key] for key in keys): row for row in rows}
    if len(observed) != len(rows):
        raise Table6SummaryError("latency repeat CSV has duplicate rows")
    numeric = (
        "recall_mean",
        "latency_mean_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    )
    identity = (
        "experiment_kind",
        "config_sha256",
        "release_identity_sha256",
        "clients",
        "request_trace_sha256",
        "requests",
        "unique_queries",
        "completed_queries",
        "error_count",
        "status",
    )
    for source in converted:
        key = tuple(str(source[item]) for item in keys)
        row = observed.get(key)
        if row is None:
            raise Table6SummaryError(
                f"latency repeat CSV lacks converted row {key!r}"
            )
        if any(str(source[item]) != row[item] for item in identity):
            raise Table6SummaryError(
                f"latency repeat identity differs for {key!r}"
            )
        for field in numeric:
            if not math.isclose(
                float(source[field]),
                require_float(row[field], f"latency repeat {field}"),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise Table6SummaryError(
                    f"latency repeat metric {field} differs for {key!r}"
                )


def resolve_latency_manifest(
    *,
    run_manifest: Path | None,
    repeat_csv: Path | None,
    binding_manifest: Path | None,
) -> tuple[Path, str, Sequence[Mapping[str, object]]]:
    if (run_manifest is None) == (repeat_csv is None):
        raise Table6SummaryError(
            "provide exactly one of --latency-run-manifest or "
            "--latency-repeat-csv"
        )
    if run_manifest is not None:
        source = run_manifest.resolve()
        source_sha = sha256_file(source)
        converted = latency_converter.convert_manifest(source)
        return source, source_sha, converted
    assert repeat_csv is not None
    repeat_csv = repeat_csv.resolve()
    binding = (
        binding_manifest.resolve()
        if binding_manifest is not None
        else repeat_csv.with_suffix(repeat_csv.suffix + ".manifest.json")
    )
    source, source_sha, _ = audit_converter_binding(
        repeat_csv, binding, experiment_kind="latency"
    )
    converted = latency_converter.convert_manifest(source)
    _compare_latency_repeat_rows(repeat_csv, converted)
    return source, source_sha, converted


def load_latency(
    source_manifest: Path,
    source_sha: str,
    converted_rows: Sequence[Mapping[str, object]],
    selection: SelectionEvidence,
) -> dict[str, LatencyPairEvidence]:
    manifest = read_json(source_manifest, "latency run manifest")
    if manifest.get("artifact_type") != "sqlens_figure5_matched_latency_run":
        raise Table6SummaryError("latency source is not a matched-latency run")
    _manifest_selector_gate(manifest, selection, label="latency manifest")
    release = manifest.get("release_contract")
    if not isinstance(release, Mapping):
        raise Table6SummaryError("latency manifest has no release contract")
    _manifest_release_gate(release, selection, label="latency manifest")
    _manifest_config_gate(
        manifest.get("frontier_config"),
        selection,
        base=source_manifest.parent,
        label="latency manifest",
    )
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping) or (
        require_int(execution.get("requests"), "latency requests")
        != EXPECTED_REQUESTS
        or require_int(execution.get("repeats"), "latency repeats")
        != EXPECTED_LATENCY_REPEATS
        or require_int(
            execution.get("expected_rows_per_pair"),
            "latency rows per pair",
        )
        != EXPECTED_REQUESTS * EXPECTED_LATENCY_REPEATS * len(EXPECTED_ARMS)
        or execution.get("execution_order") != "paired_interleaved"
    ):
        raise Table6SummaryError("latency manifest does not bind q10k/r3")

    selected = {pair.pair_id: pair for pair in selection.pairs if pair.status == SELECTED}
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or {
        str(cell.get("pair_id") or "")
        for cell in schedule
        if isinstance(cell, Mapping)
    } != set(selected):
        raise Table6SummaryError(
            "latency schedule does not exactly cover selected pairs"
        )
    converted: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for row in converted_rows:
        key = (
            str(row.get("config_id") or ""),
            str(row.get("arm_id") or ""),
            require_int(row.get("repeat_id"), "latency repeat id"),
        )
        if key in converted:
            raise Table6SummaryError(f"latency converter repeats row {key!r}")
        converted[key] = row

    result: dict[str, LatencyPairEvidence] = {}
    for cell in schedule:
        if not isinstance(cell, Mapping):
            raise Table6SummaryError("latency schedule contains a malformed cell")
        pair_id = str(cell.get("pair_id") or "")
        pair = selected[pair_id]
        if str(cell.get("dataset") or "") != pair.dataset or not _same_target(
            require_float(cell.get("target_recall"), f"{pair_id} latency target"),
            pair.target,
        ):
            raise Table6SummaryError(
                f"latency cell identity differs for pair {pair_id!r}"
            )
        for name, expected in (
            ("stock_config", pair.stock_selection_sha),
            ("sqlens_config", pair.sqlens_selection_sha),
        ):
            value = cell.get(name)
            if not isinstance(value, Mapping) or require_sha(
                value.get("config_sha256"), f"{pair_id} {name} SHA"
            ) != expected:
                raise Table6SummaryError(
                    f"latency cell {name} differs from selection for {pair_id!r}"
                )
        for arm, expected_sha in (
            ("stock_pgvector", pair.stock_arm_sha),
            ("sqlens_full", pair.sqlens_arm_sha),
        ):
            for repeat in range(EXPECTED_LATENCY_REPEATS):
                row = converted.get((pair_id, arm, repeat))
                if row is None:
                    raise Table6SummaryError(
                        f"latency converter lacks {(pair_id, arm, repeat)!r}"
                    )
                if require_sha(
                    row.get("config_sha256"),
                    f"{pair_id} {arm} latency arm SHA",
                ) != expected_sha:
                    raise Table6SummaryError(
                        f"latency arm SHA differs for {pair_id}/{arm}"
                    )
                if str(row.get("dataset") or "") != pair.dataset_id:
                    raise Table6SummaryError(
                        f"latency dataset differs for {pair_id}/{arm}"
                    )

        inputs = cell.get("input_bindings")
        if not isinstance(inputs, Mapping):
            raise Table6SummaryError(f"latency cell {pair_id} has no inputs")
        workload = inputs.get("measurement_workload_csv")
        filters = inputs.get("filters_csv")
        if not isinstance(workload, Mapping) or not isinstance(filters, Mapping):
            raise Table6SummaryError(
                f"latency cell {pair_id} lacks workload/filter bindings"
            )
        workload_path, workload_sha = audit_file_binding(
            workload,
            base=source_manifest.parent,
            label=f"{pair_id} latency workload",
        )
        if matched_latency.frontier.count_csv_rows(workload_path) != EXPECTED_REQUESTS:
            raise Table6SummaryError(f"{pair_id} latency workload is not q10k")
        _, filters_sha = audit_file_binding(
            filters,
            base=source_manifest.parent,
            label=f"{pair_id} latency filters",
        )

        raw_path = resolve_bound_path(
            cell.get("raw"),
            base=source_manifest.parent,
            label=f"{pair_id} latency raw CSV",
        )
        fields, raw_rows = read_csv(raw_path, f"{pair_id} latency raw CSV")
        require_fields(
            fields,
            (
                "mode",
                "repeat",
                "request_no",
                "query_id",
                "filter_name",
                "recall",
                "end_to_end_ms",
                "error",
            ),
            f"{pair_id} latency raw CSV",
        )
        expected_rows = (
            EXPECTED_REQUESTS * EXPECTED_LATENCY_REPEATS * len(EXPECTED_ARMS)
        )
        if len(raw_rows) != expected_rows:
            raise Table6SummaryError(
                f"{pair_id} latency raw rows={len(raw_rows)}, expected={expected_rows}"
            )
        keyed: dict[str, dict[tuple[int, int, str, str], tuple[float, float]]] = {
            arm: {} for arm in EXPECTED_ARMS
        }
        repeat_counts = {
            (arm, repeat): 0
            for arm in EXPECTED_ARMS
            for repeat in range(EXPECTED_LATENCY_REPEATS)
        }
        for row_no, row in enumerate(raw_rows, start=2):
            arm = MODE_TO_ARM.get(str(row.get("mode") or ""))
            if arm is None:
                raise Table6SummaryError(
                    f"{pair_id} latency row {row_no} has unknown mode"
                )
            repeat = require_int(
                row.get("repeat"), f"{pair_id} latency row {row_no} repeat"
            )
            if repeat >= EXPECTED_LATENCY_REPEATS:
                raise Table6SummaryError(
                    f"{pair_id} latency row {row_no} repeat is out of range"
                )
            request_no = require_int(
                row.get("request_no"),
                f"{pair_id} latency row {row_no} request_no",
            )
            query_id = str(row.get("query_id") or "").strip()
            filter_name = str(row.get("filter_name") or "").strip()
            if not query_id or not filter_name:
                raise Table6SummaryError(
                    f"{pair_id} latency row {row_no} has empty query/filter"
                )
            if str(row.get("error") or "").strip():
                raise Table6SummaryError(
                    f"{pair_id} latency row {row_no} contains an error"
                )
            latency = require_float(
                row.get("end_to_end_ms"),
                f"{pair_id} latency row {row_no}",
                lower=0.0,
            )
            if latency <= 0.0:
                raise Table6SummaryError(
                    f"{pair_id} latency row {row_no} is not positive"
                )
            recall = require_float(
                row.get("recall"),
                f"{pair_id} recall row {row_no}",
                lower=0.0,
                upper=1.0,
            )
            key = (repeat, request_no, query_id, filter_name)
            if key in keyed[arm]:
                raise Table6SummaryError(
                    f"{pair_id}/{arm} repeats paired request key {key!r}"
                )
            keyed[arm][key] = (latency, recall)
            repeat_counts[(arm, repeat)] += 1
        if any(value != EXPECTED_REQUESTS for value in repeat_counts.values()):
            raise Table6SummaryError(
                f"{pair_id} does not contain q10k in every arm/repeat"
            )
        if set(keyed["stock_pgvector"]) != set(keyed["sqlens_full"]):
            raise Table6SummaryError(
                f"{pair_id} Stock/SQLens request keys cannot be strictly paired"
            )
        filter_names = sorted({key[3] for key in keyed["stock_pgvector"]})
        if len(filter_names) != EXPECTED_FILTERS:
            raise Table6SummaryError(
                f"{pair_id} has {len(filter_names)} filters, expected 14"
            )
        by_filter: dict[
            str,
            dict[tuple[int, str], dict[str, tuple[float, ...]]],
        ] = {}
        for filter_name in filter_names:
            keys = sorted(
                key
                for key in keyed["stock_pgvector"]
                if key[3] == filter_name
            )
            cluster_repeats: dict[tuple[int, str], set[int]] = {}
            for repeat, request_no, query_id, _ in keys:
                cluster_repeats.setdefault(
                    (request_no, query_id), set()
                ).add(repeat)
            expected_repeats = set(range(EXPECTED_LATENCY_REPEATS))
            if any(
                repeats != expected_repeats
                for repeats in cluster_repeats.values()
            ):
                raise Table6SummaryError(
                    f"{pair_id}/{filter_name} query cluster does not cover "
                    "exactly all repeats"
                )
            by_filter[filter_name] = {}
            for cluster_key in sorted(cluster_repeats):
                request_no, query_id = cluster_key
                repeat_keys = tuple(
                    (
                        repeat,
                        request_no,
                        query_id,
                        filter_name,
                    )
                    for repeat in range(EXPECTED_LATENCY_REPEATS)
                )
                by_filter[filter_name][cluster_key] = {
                    arm: tuple(
                        keyed[arm][repeat_key][0]
                        for repeat_key in repeat_keys
                    )
                    for arm in EXPECTED_ARMS
                }
        ordered_keys = sorted(keyed["stock_pgvector"])
        latency_by_arm = {
            arm: tuple(keyed[arm][key][0] for key in ordered_keys)
            for arm in EXPECTED_ARMS
        }
        recall_by_arm = {
            arm: tuple(keyed[arm][key][1] for key in ordered_keys)
            for arm in EXPECTED_ARMS
        }
        result[pair_id] = LatencyPairEvidence(
            dataset=pair.dataset,
            target=pair.target,
            pair_id=pair_id,
            by_filter=by_filter,
            recall_by_arm=recall_by_arm,
            latency_by_arm=latency_by_arm,
            workload_sha256=workload_sha,
            filters_sha256=filters_sha,
            source_manifest_sha256=source_sha,
        )
    if set(converted) != {
        (pair_id, arm, repeat)
        for pair_id in selected
        for arm in EXPECTED_ARMS
        for repeat in range(EXPECTED_LATENCY_REPEATS)
    }:
        raise Table6SummaryError("latency converter contains unexpected rows")
    return result


def _core_throughput_gate(
    cell: Mapping[str, Any],
    pair: SelectionPair,
    source_manifest: Path,
    selection: SelectionEvidence,
) -> tuple[str, Path]:
    completion = cell.get("completion_audit")
    if (
        cell.get("status") != "complete"
        or not isinstance(completion, Mapping)
        or completion.get("complete") is not True
    ):
        raise Table6SummaryError(
            f"throughput cell {pair.pair_id} is incomplete"
        )
    outputs = completion.get("outputs")
    if not isinstance(outputs, Mapping):
        raise Table6SummaryError(
            f"throughput cell {pair.pair_id} has no output bindings"
        )
    expected_rows = {
        "requests": EXPECTED_REQUESTS
        * EXPECTED_THROUGHPUT_REPEATS
        * len(EXPECTED_ARMS),
        "repeats": EXPECTED_THROUGHPUT_REPEATS * len(EXPECTED_ARMS),
    }
    repeat_output_path: Path | None = None
    for name, rows in expected_rows.items():
        binding = outputs.get(name)
        if not isinstance(binding, Mapping):
            raise Table6SummaryError(
                f"throughput cell {pair.pair_id} lacks {name} binding"
            )
        output_path, _ = audit_file_binding(
            binding,
            base=source_manifest.parent,
            label=f"{pair.pair_id} throughput {name}",
        )
        if require_int(
            binding.get("rows"), f"{pair.pair_id} throughput {name} rows"
        ) != rows:
            raise Table6SummaryError(
                f"{pair.pair_id} throughput {name} row count is invalid"
            )
        if name == "repeats":
            repeat_output_path = output_path
    core_binding = outputs.get("manifest")
    if not isinstance(core_binding, Mapping):
        raise Table6SummaryError(
            f"throughput cell {pair.pair_id} lacks core manifest binding"
        )
    core_path, _ = audit_file_binding(
        core_binding,
        base=source_manifest.parent,
        label=f"{pair.pair_id} throughput core manifest",
    )
    core = read_json(core_path, f"{pair.pair_id} throughput core manifest")
    if (
        core.get("artifact_type")
        != "sqlens_figure5_mixed_q10k_throughput_cell"
        or core.get("artifact_valid") is not True
        or core.get("paper_eligible") is not True
    ):
        raise Table6SummaryError(
            f"{pair.pair_id} throughput core manifest is not paper eligible"
        )
    protocol = core.get("protocol")
    if not isinstance(protocol, Mapping) or (
        require_int(
            protocol.get("requests_per_arm_repeat"),
            f"{pair.pair_id} core requests",
        )
        != EXPECTED_REQUESTS
        or require_int(protocol.get("filters"), f"{pair.pair_id} core filters")
        != EXPECTED_FILTERS
        or require_int(protocol.get("repeats"), f"{pair.pair_id} core repeats")
        != EXPECTED_THROUGHPUT_REPEATS
        or require_int(protocol.get("clients"), f"{pair.pair_id} core clients")
        != EXPECTED_CLIENTS
        or protocol.get("throughput_source") != throughput.THROUGHPUT_SOURCE
        or protocol.get("throughput_formula")
        != "completed_queries / barrier_wall_clock_seconds"
    ):
        raise Table6SummaryError(
            f"{pair.pair_id} throughput core protocol is invalid"
        )
    configuration = core.get("configuration")
    if not isinstance(configuration, Mapping) or (
        configuration.get("pair_id") != pair.pair_id
        or require_sha(
            configuration.get("stock_config_sha256"),
            f"{pair.pair_id} core Stock arm SHA",
        )
        != pair.stock_arm_sha
        or require_sha(
            configuration.get("sqlens_config_sha256"),
            f"{pair.pair_id} core SQLens arm SHA",
        )
        != pair.sqlens_arm_sha
    ):
        raise Table6SummaryError(
            f"{pair.pair_id} throughput core config differs from selection"
        )
    release = core.get("release_contract")
    if not isinstance(release, Mapping):
        raise Table6SummaryError(
            f"{pair.pair_id} throughput core has no release contract"
        )
    _manifest_release_gate(
        release, selection, label=f"{pair.pair_id} throughput core"
    )
    gates = core.get("gates")
    if not isinstance(gates, Mapping) or not gates or any(
        value is not True for value in gates.values()
    ):
        raise Table6SummaryError(
            f"{pair.pair_id} throughput core completion gate failed"
        )

    inputs = cell.get("inputs")
    workload = (
        inputs.get("measurement_workload")
        if isinstance(inputs, Mapping)
        else None
    )
    if not isinstance(workload, Mapping):
        raise Table6SummaryError(
            f"{pair.pair_id} throughput cell has no workload binding"
        )
    workload_path, workload_sha = audit_file_binding(
        workload,
        base=source_manifest.parent,
        label=f"{pair.pair_id} throughput workload",
    )
    if (
        require_int(
            workload.get("rows"), f"{pair.pair_id} throughput workload rows"
        )
        != EXPECTED_REQUESTS
        or matched_latency.frontier.count_csv_rows(workload_path)
        != EXPECTED_REQUESTS
    ):
        raise Table6SummaryError(
            f"{pair.pair_id} throughput workload is not q10k"
        )
    if repeat_output_path is None:
        raise Table6SummaryError(
            f"{pair.pair_id} throughput repeat output was not bound"
        )
    return workload_sha, repeat_output_path


def load_throughput(
    repeat_csv: Path,
    service_csv: Path,
    binding_manifest: Path | None,
    selection: SelectionEvidence,
) -> dict[str, ThroughputPairEvidence]:
    repeat_csv = repeat_csv.resolve()
    service_csv = service_csv.resolve()
    binding_path = (
        binding_manifest.resolve()
        if binding_manifest is not None
        else repeat_csv.with_suffix(repeat_csv.suffix + ".manifest.json")
    )
    source_manifest, source_sha, binding = audit_converter_binding(
        repeat_csv,
        binding_path,
        experiment_kind="throughput",
        service_output=service_csv,
    )
    manifest, release, audited_source_sha = converter_binding.audited_run_manifest(
        source_manifest,
        expected_artifact_type="sqlens_figure5_matched_throughput_run",
    )
    if audited_source_sha != source_sha:
        raise Table6SummaryError("throughput source manifest changed during audit")
    _manifest_release_gate(release, selection, label="throughput manifest")
    _manifest_selector_gate(manifest, selection, label="throughput manifest")
    protocol_name = str(manifest.get("protocol_slice") or "")
    if str(binding.get("protocol_slice") or "") != protocol_name:
        raise Table6SummaryError(
            "throughput binding protocol differs from source manifest"
        )
    fingerprint = require_sha(
        manifest.get("protocol_fingerprint_sha256"),
        "throughput protocol fingerprint",
    )
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping) or (
        require_int(
            execution.get("requests_per_arm_repeat"),
            "throughput requests per repeat",
        )
        != EXPECTED_REQUESTS
        or require_int(execution.get("repeats"), "throughput repeats")
        != EXPECTED_THROUGHPUT_REPEATS
        or require_int(
            execution.get("expected_repeat_rows_per_cell"),
            "throughput repeat rows per cell",
        )
        != EXPECTED_THROUGHPUT_REPEATS * len(EXPECTED_ARMS)
        or list(execution.get("client_grid") or []) != [EXPECTED_CLIENTS]
        or execution.get("throughput_source") != throughput.THROUGHPUT_SOURCE
        or execution.get("throughput_formula")
        != "completed_queries / barrier_wall_clock_seconds"
        or execution.get("qps_from_latency_forbidden") is not True
    ):
        raise Table6SummaryError(
            "throughput manifest does not bind fixed-target c16/q10k/r3"
        )
    scope = manifest.get("full_release_scope")
    if not isinstance(scope, Mapping) or (
        scope.get("requested") is not True
        or list(scope.get("required_clients") or []) != [EXPECTED_CLIENTS]
        or require_int(scope.get("required_repeats"), "throughput scope repeats")
        != EXPECTED_THROUGHPUT_REPEATS
    ):
        raise Table6SummaryError("throughput full-release scope is invalid")
    frontier = manifest.get("frontier_config")
    normalized_plan = manifest.get("normalized_measurement_plan")
    if not isinstance(frontier, Mapping) or not isinstance(
        normalized_plan, Mapping
    ):
        raise Table6SummaryError(
            "throughput manifest lacks frontier/normalized-plan bindings"
        )
    _, frontier_sha = _manifest_config_gate(
        frontier,
        selection,
        base=source_manifest.parent,
        label="throughput frontier config",
    )
    normalized_plan_path, normalized_plan_sha = audit_file_binding(
        normalized_plan,
        base=source_manifest.parent,
        label="throughput normalized measurement plan",
    )
    normalized_payload = read_json(
        normalized_plan_path,
        "throughput normalized measurement plan",
    )
    _manifest_config_gate(
        normalized_payload.get("frontier_config"),
        selection,
        base=source_manifest.parent,
        label="throughput normalized measurement plan",
    )
    _manifest_selector_gate(
        normalized_payload,
        selection,
        label="throughput normalized measurement plan",
    )

    selected = {pair.pair_id: pair for pair in selection.pairs if pair.status == SELECTED}
    schedule = manifest.get("schedule")
    if (
        not isinstance(schedule, list)
        or require_int(manifest.get("cells_total"), "throughput cells_total")
        != len(selected)
        or require_int(
            manifest.get("cells_complete"), "throughput cells_complete"
        )
        != len(selected)
        or {
            str(cell.get("pair_id") or "")
            for cell in schedule
            if isinstance(cell, Mapping)
        }
        != set(selected)
    ):
        raise Table6SummaryError(
            "throughput schedule does not exactly cover selected pairs"
        )
    workload_by_pair: dict[str, str] = {}
    detailed_repeat_path_by_pair: dict[str, Path] = {}
    for cell in schedule:
        if not isinstance(cell, Mapping):
            raise Table6SummaryError(
                "throughput schedule contains a malformed cell"
            )
        pair_id = str(cell.get("pair_id") or "")
        pair = selected[pair_id]
        if (
            str(cell.get("dataset") or "") != pair.dataset
            or require_int(cell.get("clients"), f"{pair_id} clients")
            != EXPECTED_CLIENTS
            or not _same_target(
                require_float(cell.get("target_recall"), f"{pair_id} target"),
                pair.target,
            )
        ):
            raise Table6SummaryError(
                f"throughput cell identity differs for {pair_id!r}"
            )
        workload_sha, repeat_path = _core_throughput_gate(
            cell, pair, source_manifest, selection
        )
        workload_by_pair[pair_id] = workload_sha
        detailed_repeat_path_by_pair[pair_id] = repeat_path

    repeat_fields, repeat_rows = read_csv(
        repeat_csv, "throughput repeat CSV"
    )
    require_fields(
        repeat_fields,
        (
            "dataset",
            "experiment_kind",
            "arm_id",
            "mode_id",
            "config_id",
            "config_sha256",
            "release_identity_sha256",
            "clients",
            "repeat_id",
            "request_trace_sha256",
            "requests",
            "unique_queries",
            "completed_queries",
            "error_count",
            "wall_clock_seconds",
            "throughput_qps",
            "throughput_source",
            "status",
        ),
        "throughput repeat CSV",
    )
    expected_repeat_rows = (
        len(selected) * len(EXPECTED_ARMS) * EXPECTED_THROUGHPUT_REPEATS
    )
    if len(repeat_rows) != expected_repeat_rows:
        raise Table6SummaryError(
            f"throughput repeat rows={len(repeat_rows)}, "
            f"expected={expected_repeat_rows}"
        )
    aggregate_keys: set[tuple[str, str, int]] = set()
    aggregate_pair_config_sha: dict[str, str] = {}
    for row in repeat_rows:
        pair_id = str(row.get("config_id") or "")
        arm = str(row.get("arm_id") or "")
        if pair_id not in selected or arm not in EXPECTED_ARMS:
            raise Table6SummaryError(
                f"throughput repeat has unexpected pair/arm {(pair_id, arm)!r}"
            )
        repeat_id = require_int(
            row.get("repeat_id"), f"{pair_id}/{arm} aggregate repeat"
        )
        key = (pair_id, arm, repeat_id)
        if key in aggregate_keys:
            raise Table6SummaryError(
                f"throughput aggregate repeats row {key!r}"
            )
        aggregate_keys.add(key)
        if (
            str(row.get("dataset") or "") != selected[pair_id].dataset_id
            or row.get("experiment_kind") != "throughput"
            or row.get("mode_id") != ARM_TO_MODE[arm]
            or require_int(
                row.get("clients"), f"{pair_id}/{arm} aggregate clients"
            )
            != EXPECTED_CLIENTS
            or require_int(
                row.get("requests"), f"{pair_id}/{arm} aggregate requests"
            )
            != EXPECTED_REQUESTS
            or require_int(
                row.get("completed_queries"),
                f"{pair_id}/{arm} aggregate completed",
            )
            != EXPECTED_REQUESTS
            or require_int(
                row.get("error_count"), f"{pair_id}/{arm} aggregate errors"
            )
            != 0
            or row.get("throughput_source") != throughput.THROUGHPUT_SOURCE
            or row.get("status") != "valid"
        ):
            raise Table6SummaryError(
                f"throughput aggregate protocol differs for {pair_id}/{arm}"
            )
        if require_sha(
            row.get("release_identity_sha256"),
            f"{pair_id}/{arm} aggregate release SHA",
        ) != selection.release["sha256"]:
            raise Table6SummaryError(
                f"throughput aggregate release differs for {pair_id}/{arm}"
            )
        observed_pair_sha = require_sha(
            row.get("config_sha256"),
            f"{pair_id}/{arm} aggregate pair config SHA",
        )
        previous_pair_sha = aggregate_pair_config_sha.setdefault(
            pair_id, observed_pair_sha
        )
        if previous_pair_sha != observed_pair_sha:
            raise Table6SummaryError(
                f"throughput aggregate pair config SHA differs for {pair_id}"
            )
        source_row_sha = str(row.get("source_manifest_sha256") or "")
        if source_row_sha and require_sha(
            source_row_sha, f"{pair_id}/{arm} aggregate source SHA"
        ) != source_sha:
            raise Table6SummaryError(
                f"throughput aggregate source differs for {pair_id}/{arm}"
            )
    expected_aggregate_keys = {
        (pair_id, arm, repeat)
        for pair_id in selected
        for arm in EXPECTED_ARMS
        for repeat in range(EXPECTED_THROUGHPUT_REPEATS)
    }
    if aggregate_keys != expected_aggregate_keys:
        raise Table6SummaryError(
            "throughput aggregate repeat coverage is incomplete"
        )

    repeat_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for pair_id, detailed_path in detailed_repeat_path_by_pair.items():
        detailed_fields, detailed_rows = read_csv(
            detailed_path, f"{pair_id} detailed throughput repeats"
        )
        require_fields(
            detailed_fields,
            (
                "dataset",
                "experiment_kind",
                "arm_id",
                "mode_id",
                "pair_id",
                "target_recall",
                "config_sha256",
                "stock_config_sha256",
                "sqlens_config_sha256",
                "arm_config_sha256",
                "release_identity_sha256",
                "clients",
                "repeat_id",
                "request_trace_sha256",
                "requests",
                "unique_queries",
                "completed_queries",
                "error_count",
                "wall_clock_seconds",
                "throughput_qps",
                "throughput_source",
                "status",
            ),
            f"{pair_id} detailed throughput repeats",
        )
        if len(detailed_rows) != (
            EXPECTED_THROUGHPUT_REPEATS * len(EXPECTED_ARMS)
        ):
            raise Table6SummaryError(
                f"{pair_id} detailed throughput repeat coverage is incomplete"
            )
        for row in detailed_rows:
            observed_pair_id = str(row.get("pair_id") or "")
            arm = str(row.get("arm_id") or "")
            if observed_pair_id != pair_id or arm not in EXPECTED_ARMS:
                raise Table6SummaryError(
                    f"{pair_id} detailed repeat has unexpected identity"
                )
            repeat_groups.setdefault((pair_id, arm), []).append(row)

    service_fields, service_rows = read_csv(
        service_csv, "throughput service CSV"
    )
    require_fields(
        service_fields,
        (
            "protocol_slice",
            "dataset",
            "pair_id",
            "target_recall",
            "arm_id",
            "mode_id",
            "config_sha256",
            "arm_config_sha256",
            "stock_config_sha256",
            "sqlens_config_sha256",
            "clients",
            "repeats",
            "requests_per_repeat",
            "total_requests",
            "completed_queries",
            "error_count",
            "timeout_count",
            "total_barrier_wall_clock_seconds",
            "throughput_qps",
            "throughput_source",
            "target_lcb95_met",
            "selection_csv_sha256",
            "selection_plan_sha256",
            "selection_manifest_sha256",
            "frontier_config_sha256",
            "normalized_measurement_plan_sha256",
            "protocol_fingerprint_sha256",
            "release_contract_sha256",
            "source_manifest_sha256",
        ),
        "throughput service CSV",
    )
    if len(service_rows) != len(selected) * len(EXPECTED_ARMS):
        raise Table6SummaryError("throughput service row coverage is incomplete")
    service_map: dict[tuple[str, str], dict[str, str]] = {}
    for row in service_rows:
        key = (str(row.get("pair_id") or ""), str(row.get("arm_id") or ""))
        if key in service_map:
            raise Table6SummaryError(
                f"throughput service repeats pair/arm {key!r}"
            )
        service_map[key] = row

    result: dict[str, ThroughputPairEvidence] = {}
    for pair_id, pair in selected.items():
        qps_by_arm: dict[str, float] = {}
        pair_config_sha: str | None = None
        for arm, expected_arm_sha in (
            ("stock_pgvector", pair.stock_arm_sha),
            ("sqlens_full", pair.sqlens_arm_sha),
        ):
            group = repeat_groups.get((pair_id, arm), [])
            if len(group) != EXPECTED_THROUGHPUT_REPEATS or {
                require_int(row.get("repeat_id"), f"{pair_id}/{arm} repeat")
                for row in group
            } != set(range(EXPECTED_THROUGHPUT_REPEATS)):
                raise Table6SummaryError(
                    f"throughput repeats are incomplete for {pair_id}/{arm}"
                )
            for row in group:
                if (
                    str(row.get("dataset") or "") != pair.dataset_id
                    or row.get("experiment_kind") != "throughput"
                    or row.get("mode_id") != ARM_TO_MODE[arm]
                    or require_int(row.get("clients"), f"{pair_id}/{arm} clients")
                    != EXPECTED_CLIENTS
                    or require_int(
                        row.get("requests"), f"{pair_id}/{arm} requests"
                    )
                    != EXPECTED_REQUESTS
                    or require_int(
                        row.get("unique_queries"), f"{pair_id}/{arm} unique"
                    )
                    != EXPECTED_REQUESTS
                    or require_int(
                        row.get("completed_queries"),
                        f"{pair_id}/{arm} completed",
                    )
                    != EXPECTED_REQUESTS
                    or require_int(
                        row.get("error_count"), f"{pair_id}/{arm} errors"
                    )
                    != 0
                    or row.get("throughput_source")
                    != throughput.THROUGHPUT_SOURCE
                    or row.get("status") != "valid"
                ):
                    raise Table6SummaryError(
                        f"throughput repeat protocol differs for {pair_id}/{arm}"
                    )
                observed_pair_config_sha = require_sha(
                    row.get("config_sha256"),
                    f"{pair_id}/{arm} pair config SHA",
                )
                if pair_config_sha is None:
                    pair_config_sha = observed_pair_config_sha
                elif observed_pair_config_sha != pair_config_sha:
                    raise Table6SummaryError(
                        f"throughput pair config SHA differs for {pair_id}"
                    )
                if require_sha(
                    row.get("arm_config_sha256"),
                    f"{pair_id}/{arm} arm config SHA",
                ) != expected_arm_sha:
                    raise Table6SummaryError(
                        f"throughput repeat arm SHA differs for {pair_id}/{arm}"
                    )
                if require_sha(
                    row.get("stock_config_sha256"),
                    f"{pair_id}/{arm} Stock SHA",
                ) != pair.stock_arm_sha or require_sha(
                    row.get("sqlens_config_sha256"),
                    f"{pair_id}/{arm} SQLens SHA",
                ) != pair.sqlens_arm_sha:
                    raise Table6SummaryError(
                        f"throughput repeat pair SHA differs for {pair_id}/{arm}"
                    )
                if require_sha(
                    row.get("release_identity_sha256"),
                    f"{pair_id}/{arm} release SHA",
                ) != selection.release["sha256"]:
                    raise Table6SummaryError(
                        f"throughput repeat release differs for {pair_id}/{arm}"
                    )
            service = service_map.get((pair_id, arm))
            if service is None:
                raise Table6SummaryError(
                    f"throughput service lacks {pair_id}/{arm}"
                )
            if (
                service.get("protocol_slice") != protocol_name
                or str(service.get("dataset") or "") != pair.dataset_id
                or service.get("mode_id") != ARM_TO_MODE[arm]
                or not _same_target(
                    require_float(
                        service.get("target_recall"),
                        f"{pair_id}/{arm} service target",
                    ),
                    pair.target,
                )
                or require_int(
                    service.get("clients"), f"{pair_id}/{arm} service clients"
                )
                != EXPECTED_CLIENTS
                or require_int(
                    service.get("repeats"), f"{pair_id}/{arm} service repeats"
                )
                != EXPECTED_THROUGHPUT_REPEATS
                or require_int(
                    service.get("requests_per_repeat"),
                    f"{pair_id}/{arm} service requests",
                )
                != EXPECTED_REQUESTS
                or require_int(
                    service.get("total_requests"),
                    f"{pair_id}/{arm} service total",
                )
                != EXPECTED_REQUESTS * EXPECTED_THROUGHPUT_REPEATS
                or require_int(
                    service.get("completed_queries"),
                    f"{pair_id}/{arm} service completed",
                )
                != EXPECTED_REQUESTS * EXPECTED_THROUGHPUT_REPEATS
                or require_int(
                    service.get("error_count"),
                    f"{pair_id}/{arm} service errors",
                )
                != 0
                or require_int(
                    service.get("timeout_count"),
                    f"{pair_id}/{arm} service timeouts",
                )
                != 0
                or service.get("throughput_source")
                != throughput.THROUGHPUT_SOURCE
            ):
                raise Table6SummaryError(
                    f"throughput service protocol differs for {pair_id}/{arm}"
                )
            require_true(
                service.get("target_lcb95_met"),
                f"{pair_id}/{arm} service recall target",
            )
            for field, expected in (
                ("arm_config_sha256", expected_arm_sha),
                ("stock_config_sha256", pair.stock_arm_sha),
                ("sqlens_config_sha256", pair.sqlens_arm_sha),
                (
                    "release_contract_sha256",
                    selection.release["sha256"],
                ),
                ("source_manifest_sha256", source_sha),
                ("protocol_fingerprint_sha256", fingerprint),
                (
                    "selection_csv_sha256",
                    selection.bindings["selection_csv_sha256"],
                ),
                (
                    "selection_plan_sha256",
                    selection.bindings["selection_plan_sha256"],
                ),
                (
                    "selection_manifest_sha256",
                    selection.bindings["selection_manifest_sha256"],
                ),
                ("frontier_config_sha256", frontier_sha),
                (
                    "normalized_measurement_plan_sha256",
                    normalized_plan_sha,
                ),
            ):
                if require_sha(
                    service.get(field), f"{pair_id}/{arm} service {field}"
                ) != expected:
                    raise Table6SummaryError(
                        f"throughput service {field} differs for {pair_id}/{arm}"
                    )
            if pair_config_sha is None or require_sha(
                service.get("config_sha256"),
                f"{pair_id}/{arm} service pair config SHA",
            ) != pair_config_sha:
                raise Table6SummaryError(
                    f"throughput service pair config SHA differs for "
                    f"{pair_id}/{arm}"
                )
            if aggregate_pair_config_sha.get(pair_id) != pair_config_sha:
                raise Table6SummaryError(
                    f"throughput aggregate/source pair config SHA differs "
                    f"for {pair_id}"
                )
            pooled_completed = sum(
                require_int(
                    row.get("completed_queries"),
                    f"{pair_id}/{arm} repeat completed",
                )
                for row in group
            )
            pooled_wall = sum(
                require_float(
                    row.get("wall_clock_seconds"),
                    f"{pair_id}/{arm} repeat wall",
                    lower=0.0,
                )
                for row in group
            )
            if pooled_wall <= 0.0:
                raise Table6SummaryError(
                    f"throughput repeat wall time is zero for {pair_id}/{arm}"
                )
            qps = require_float(
                service.get("throughput_qps"),
                f"{pair_id}/{arm} service QPS",
                lower=0.0,
            )
            if not math.isclose(
                qps,
                pooled_completed / pooled_wall,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise Table6SummaryError(
                    f"throughput service QPS is not completed/barrier wall "
                    f"for {pair_id}/{arm}"
                )
            qps_by_arm[arm] = qps
        result[pair_id] = ThroughputPairEvidence(
            dataset=pair.dataset,
            target=pair.target,
            pair_id=pair_id,
            qps_by_arm=qps_by_arm,
            workload_sha256=workload_by_pair[pair_id],
            source_manifest_sha256=source_sha,
            protocol_fingerprint_sha256=fingerprint,
        )
    if set(repeat_groups) != {
        (pair_id, arm) for pair_id in selected for arm in EXPECTED_ARMS
    } or set(service_map) != {
        (pair_id, arm) for pair_id in selected for arm in EXPECTED_ARMS
    }:
        raise Table6SummaryError(
            "throughput artifacts contain unexpected pair/arm rows"
        )
    return result


def paired_stratified_speedup(
    by_filter: Mapping[
        str,
        Mapping[
            tuple[int, str],
            Mapping[str, Sequence[float]],
        ],
    ],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float, int]:
    if samples < 100:
        raise Table6SummaryError("bootstrap requires at least 100 samples")
    if len(by_filter) != EXPECTED_FILTERS:
        raise Table6SummaryError(
            f"speedup requires exactly {EXPECTED_FILTERS} filters"
        )
    central_logs: list[float] = []
    wins = 0
    bootstrap_logs = [0.0] * samples
    numpy_rng = np.random.default_rng(seed) if np is not None else None
    python_rng = random.Random(seed)
    for filter_name in sorted(by_filter):
        clusters = by_filter[filter_name]
        if not clusters:
            raise Table6SummaryError(
                f"filter {filter_name!r} has no query clusters"
            )
        stock_cluster_means: list[float] = []
        sqlens_cluster_means: list[float] = []
        repeat_count: int | None = None
        for cluster_key in sorted(clusters):
            arms = clusters[cluster_key]
            if set(arms) != set(EXPECTED_ARMS):
                raise Table6SummaryError(
                    f"filter {filter_name!r} cluster {cluster_key!r} "
                    "cannot be strictly paired"
                )
            stock_values = tuple(
                float(value) for value in arms["stock_pgvector"]
            )
            sqlens_values = tuple(
                float(value) for value in arms["sqlens_full"]
            )
            if (
                not stock_values
                or len(stock_values) != len(sqlens_values)
                or any(
                    not math.isfinite(value) or value <= 0.0
                    for value in (*stock_values, *sqlens_values)
                )
            ):
                raise Table6SummaryError(
                    f"filter {filter_name!r} cluster {cluster_key!r} "
                    "cannot be strictly paired"
                )
            if len(stock_values) != EXPECTED_LATENCY_REPEATS:
                raise Table6SummaryError(
                    f"filter {filter_name!r} cluster {cluster_key!r} has "
                    f"{len(stock_values)} repeats, expected "
                    f"{EXPECTED_LATENCY_REPEATS}"
                )
            if repeat_count is None:
                repeat_count = len(stock_values)
            elif len(stock_values) != repeat_count:
                raise Table6SummaryError(
                    f"filter {filter_name!r} query clusters have unequal "
                    "repeat coverage"
                )
            stock_cluster_means.append(statistics.fmean(stock_values))
            sqlens_cluster_means.append(statistics.fmean(sqlens_values))
        stock_mean = statistics.fmean(stock_cluster_means)
        sqlens_mean = statistics.fmean(sqlens_cluster_means)
        central_logs.append(math.log(stock_mean / sqlens_mean))
        wins += int(sqlens_mean < stock_mean)
        if np is not None and numpy_rng is not None:
            stock = np.asarray(stock_cluster_means, dtype=np.float64)
            sqlens = np.asarray(sqlens_cluster_means, dtype=np.float64)
            chunk = 128
            for start in range(0, samples, chunk):
                stop = min(samples, start + chunk)
                indexes = numpy_rng.integers(
                    0, stock.size, size=(stop - start, stock.size)
                )
                stock_means = np.mean(stock[indexes], axis=1)
                sqlens_means = np.mean(sqlens[indexes], axis=1)
                filter_logs = np.log(stock_means / sqlens_means)
                for offset, value in enumerate(filter_logs, start=start):
                    bootstrap_logs[offset] += float(value)
        else:
            count = len(stock_cluster_means)
            for sample in range(samples):
                stock_sum = 0.0
                sqlens_sum = 0.0
                for _ in range(count):
                    index = python_rng.randrange(count)
                    stock_sum += stock_cluster_means[index]
                    sqlens_sum += sqlens_cluster_means[index]
                bootstrap_logs[sample] += math.log(stock_sum / sqlens_sum)
    center = math.exp(statistics.fmean(central_logs))
    distribution = [
        math.exp(value / EXPECTED_FILTERS) for value in bootstrap_logs
    ]
    low = percentile(distribution, 0.025)
    high = percentile(distribution, 0.975)
    return center, low, high, wins


def summarize(
    selection: SelectionEvidence,
    latency: Mapping[str, LatencyPairEvidence],
    service: Mapping[str, ThroughputPairEvidence],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    selected_ids = {
        pair.pair_id for pair in selection.pairs if pair.status == SELECTED
    }
    if set(latency) != selected_ids or set(service) != selected_ids:
        raise Table6SummaryError(
            "latency/throughput evidence does not exactly cover selected pairs"
        )
    output: list[dict[str, object]] = []
    for pair in selection.pairs:
        base: dict[str, object] = {
            field: "" for field in SUMMARY_FIELDS
        }
        base.update(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": pair.dataset,
                "target_recall": pair.target,
                "pair_id": pair.pair_id,
                "status": pair.status,
                "selection_csv_sha256": selection.bindings[
                    "selection_csv_sha256"
                ],
                "selection_plan_sha256": selection.bindings[
                    "selection_plan_sha256"
                ],
                "selection_manifest_sha256": selection.bindings[
                    "selection_manifest_sha256"
                ],
                "release_contract_sha256": selection.release["sha256"],
                "dataset_config_sha256": (
                    selection.config["sha256"]
                    if isinstance(selection.config, Mapping)
                    else ""
                ),
                "required_grid_contract_sha256": (
                    selection.required_grid["sha256"]
                    if isinstance(selection.required_grid, Mapping)
                    else ""
                ),
                "required_grid_cell_keys_sha256": (
                    selection.required_grid["cell_keys_sha256"]
                    if isinstance(selection.required_grid, Mapping)
                    else ""
                ),
            }
        )
        if pair.status == UNATTAINABLE:
            base["status_detail"] = UNATTAINABLE
            output.append(base)
            continue
        latency_pair = latency[pair.pair_id]
        service_pair = service[pair.pair_id]
        if (
            latency_pair.dataset != pair.dataset
            or service_pair.dataset != pair.dataset
            or not _same_target(latency_pair.target, pair.target)
            or not _same_target(service_pair.target, pair.target)
        ):
            raise Table6SummaryError(
                f"joined identity differs for pair {pair.pair_id!r}"
            )
        if latency_pair.workload_sha256 != service_pair.workload_sha256:
            raise Table6SummaryError(
                f"latency/throughput workload SHA differs for {pair.pair_id}"
            )
        speedup, speedup_low, speedup_high, wins = (
            paired_stratified_speedup(
                latency_pair.by_filter,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        )
        stock_latency = latency_pair.latency_by_arm["stock_pgvector"]
        sqlens_latency = latency_pair.latency_by_arm["sqlens_full"]
        stock_recall = latency_pair.recall_by_arm["stock_pgvector"]
        sqlens_recall = latency_pair.recall_by_arm["sqlens_full"]
        base.update(
            {
                "status_detail": "complete_paper_eligible",
                "stock_selection_config_sha256": pair.stock_selection_sha,
                "sqlens_selection_config_sha256": pair.sqlens_selection_sha,
                "stock_arm_config_sha256": pair.stock_arm_sha,
                "sqlens_arm_config_sha256": pair.sqlens_arm_sha,
                "stock_recall": statistics.fmean(stock_recall),
                "sqlens_recall": statistics.fmean(sqlens_recall),
                "stock_mean_latency_ms": statistics.fmean(stock_latency),
                "sqlens_mean_latency_ms": statistics.fmean(sqlens_latency),
                "stock_latency_p95_ms": percentile(stock_latency, 0.95),
                "sqlens_latency_p95_ms": percentile(sqlens_latency, 0.95),
                "stock_latency_p99_ms": percentile(stock_latency, 0.99),
                "sqlens_latency_p99_ms": percentile(sqlens_latency, 0.99),
                "stock_qps": service_pair.qps_by_arm["stock_pgvector"],
                "sqlens_qps": service_pair.qps_by_arm["sqlens_full"],
                "speedup_geomean": speedup,
                "speedup_ci95_low": speedup_low,
                "speedup_ci95_high": speedup_high,
                "wins": wins,
                "wins_denominator": EXPECTED_FILTERS,
                "filters": EXPECTED_FILTERS,
                "latency_requests_per_arm": EXPECTED_REQUESTS
                * EXPECTED_LATENCY_REPEATS,
                "latency_repeats": EXPECTED_LATENCY_REPEATS,
                "throughput_clients": EXPECTED_CLIENTS,
                "throughput_requests_per_arm": EXPECTED_REQUESTS
                * EXPECTED_THROUGHPUT_REPEATS,
                "throughput_repeats": EXPECTED_THROUGHPUT_REPEATS,
                "bootstrap_method": BOOTSTRAP_METHOD,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
                "latency_workload_sha256": latency_pair.workload_sha256,
                "throughput_workload_sha256": service_pair.workload_sha256,
                "latency_source_manifest_sha256": (
                    latency_pair.source_manifest_sha256
                ),
                "throughput_source_manifest_sha256": (
                    service_pair.source_manifest_sha256
                ),
                "throughput_protocol_fingerprint_sha256": (
                    service_pair.protocol_fingerprint_sha256
                ),
            }
        )
        output.append(base)
    return output


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--required-grid-contract",
        type=Path,
        required=True,
    )
    latency = parser.add_mutually_exclusive_group(required=True)
    latency.add_argument("--latency-run-manifest", type=Path)
    latency.add_argument("--latency-repeat-csv", type=Path)
    parser.add_argument("--latency-binding-manifest", type=Path)
    parser.add_argument("--throughput-repeat-csv", type=Path, required=True)
    parser.add_argument("--throughput-service-csv", type=Path, required=True)
    parser.add_argument("--throughput-binding-manifest", type=Path)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        required_grid = load_required_grid_contract(
            args.required_grid_contract
        )
        selection = load_selection(
            args.selection_csv,
            args.selection_plan,
            args.selection_manifest,
            required_grid,
        )
        latency_manifest, latency_source_sha, converted = (
            resolve_latency_manifest(
                run_manifest=args.latency_run_manifest,
                repeat_csv=args.latency_repeat_csv,
                binding_manifest=args.latency_binding_manifest,
            )
        )
        latency = load_latency(
            latency_manifest, latency_source_sha, converted, selection
        )
        service = load_throughput(
            args.throughput_repeat_csv,
            args.throughput_service_csv,
            args.throughput_binding_manifest,
            selection,
        )
        rows = summarize(
            selection,
            latency,
            service,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        out_csv = args.out_csv.resolve()
        out_json = args.out_json.resolve()
        if out_csv == out_json:
            raise Table6SummaryError("CSV and JSON outputs must differ")
        atomic_write_csv(out_csv, rows)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "runner_version": RUNNER_VERSION,
            "status": "complete",
            "artifact_valid": True,
            "paper_eligible": True,
            "created_at": utc_now(),
            "protocol": {
                "latency": "paired_interleaved_q10k_r3",
                "throughput": "fixed_target_c16_q10k_r3",
                "filters": EXPECTED_FILTERS,
                "speedup": "14_filter_geometric_mean_stock_over_sqlens",
                "wins": "pooled_filter_mean_sqlens_lt_stock",
                "bootstrap_method": BOOTSTRAP_METHOD,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
            },
            "release_contract": dict(selection.release),
            "dataset_config": dict(selection.config or {}),
            "required_grid_contract": dict(selection.required_grid or {}),
            "inputs": {
                "required_grid_contract": {
                    "path": str(required_grid.path),
                    "sha256": required_grid.sha256,
                    "cell_keys_sha256": required_grid.cell_keys_sha256,
                },
                "dataset_config": {
                    "path": str(required_grid.config_path),
                    "sha256": required_grid.config_sha256,
                },
                "selection_csv": {
                    "path": str(args.selection_csv.resolve()),
                    "sha256": selection.bindings["selection_csv_sha256"],
                },
                "selection_plan": {
                    "path": str(args.selection_plan.resolve()),
                    "sha256": selection.bindings["selection_plan_sha256"],
                },
                "selection_manifest": {
                    "path": str(args.selection_manifest.resolve()),
                    "sha256": selection.bindings[
                        "selection_manifest_sha256"
                    ],
                },
                "latency_source_manifest": {
                    "path": str(latency_manifest),
                    "sha256": latency_source_sha,
                },
                "throughput_repeat_csv": {
                    "path": str(args.throughput_repeat_csv.resolve()),
                    "sha256": sha256_file(
                        args.throughput_repeat_csv.resolve()
                    ),
                },
                "throughput_service_csv": {
                    "path": str(args.throughput_service_csv.resolve()),
                    "sha256": sha256_file(
                        args.throughput_service_csv.resolve()
                    ),
                },
            },
            "outputs": {
                "csv": {
                    "path": str(out_csv),
                    "sha256": sha256_file(out_csv),
                    "rows": len(rows),
                    "fields": list(SUMMARY_FIELDS),
                }
            },
            "summary": {
                "rows": len(rows),
                "selected": sum(row["status"] == SELECTED for row in rows),
                "unattainable": sum(
                    row["status"] == UNATTAINABLE for row in rows
                ),
            },
            "rows": rows,
        }
        atomic_write_json(out_json, payload)
    except (
        Table6SummaryError,
        converter_binding.ConverterBindingError,
        latency_converter.LatencyRepeatError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        f"wrote {out_csv} rows={len(rows)} json={out_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
