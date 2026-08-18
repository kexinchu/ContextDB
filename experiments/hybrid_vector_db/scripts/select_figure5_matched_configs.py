#!/usr/bin/env python3
"""Select audited, independently tuned Figure 5 matched-recall configurations.

This bounded sidecar consumes only Figure 5 calibration request CSVs.  It does
not start PostgreSQL, mutate a database, or infer a result from partial runs.
For each recall target it independently selects the lowest-mean-latency Stock
and full-SQLens configuration under an explicit qualification scope. The
formal default requires both aggregate and worst-predicate bootstrap recall
lower confidence bounds to meet the target. Formal fixed-target publication
also requires an explicit, serially executed required-grid contract. The
aggregate-only policy is kept only as an explicit legacy-audit mode. A target
with no qualifying configuration is preserved as
``unattainable_on_calibration_grid`` only when the complete required grid has
been audited; it is never inferred from a partial input set.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = ROOT / "results/hybrid_vector_db/figure5_r35"
DEFAULT_CONTRACT = ROOT / "experiments/hybrid_vector_db/configs/p0_release_contract.json"
DEFAULT_TARGETS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.97, 0.99)
TARGET_POLICIES = ("fixed", "distinct_pairs")
QUALIFICATION_SCOPE_AGGREGATE = "aggregate_lcb"
QUALIFICATION_SCOPE_FORMAL = "global_min_predicate_lcb"
QUALIFICATION_SCOPES = (
    QUALIFICATION_SCOPE_AGGREGATE,
    QUALIFICATION_SCOPE_FORMAL,
)
DEFAULT_QUALIFICATION_SCOPE = QUALIFICATION_SCOPE_FORMAL
DEFAULT_MAX_POINTS_PER_DATASET = 12
DEFAULT_MIN_POINTS_PER_DATASET = 10
RUNNER_VERSION = "sqlens-figure5-matched-config-selector-v7"
REQUIRED_GRID_CONTRACT_TYPE = "figure5_formal_fixed_target_required_grid"
EXPECTED_FILTERS = 14
LEGACY_EXPECTED_REQUESTS = 200
FORMAL_OBSERVATIONS_PER_FILTER = 200
FORMAL_EXPECTED_REQUESTS = EXPECTED_FILTERS * FORMAL_OBSERVATIONS_PER_FILTER
# Compatibility alias for fixtures and explicit aggregate_lcb audits only.
EXPECTED_REQUESTS = LEGACY_EXPECTED_REQUESTS
MODE_STOCK = "original"
MODE_SQLENS = "design1_bloom_bfs_layout_d3"
FAMILY_BOTH_OFF = "both_off"
FAMILY_STOCK_STRICT = "stock_strict"
FAMILY_SQLENS_CAP = "sqlens_cap"
FAMILY_STOCK_CAP = "stock_cap"
FAMILY_SQLENS_TARGET = "sqlens_target"
CAP_FAMILIES = {FAMILY_SQLENS_CAP, FAMILY_STOCK_CAP}
TARGET_FAMILIES = {FAMILY_SQLENS_TARGET}
CSV_RE = re.compile(
    r"^figure5_(?P<artifact_tag>r[0-9]+)_(?P<dataset>[a-z0-9]+)_calibration_"
    r"(?P<family>both_off|stock_strict|sqlens_cap|stock_cap|sqlens_target)_"
    r"ef(?P<ef>[1-9][0-9]*)"
    r"(?:_cap(?P<cap>[1-9][0-9]*))?"
    r"(?:_target(?P<target>[1-9][0-9]*))?\.csv$"
)
PLAN_NAMESPACE_RE = re.compile(
    r"^fig5-(?P<artifact_tag>r[0-9]+)-(?P<dataset>[a-z0-9]+)-"
    r"(?P<phase>[a-z0-9]+)-"
    r"(?P<family>both_off|stock_strict|sqlens_cap|stock_cap|sqlens_target)-"
    r"ef(?P<ef>[1-9][0-9]*)(?:-cap(?P<cap>[1-9][0-9]*))?"
    r"(?:-target(?P<target>[1-9][0-9]*))?$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RELEASE_TAG_RE = re.compile(r"(?:^|-)r([0-9]+)(?:-|$)")

RAW_REQUIRED_FIELDS = {
    "mode",
    "error",
    "recall",
    "end_to_end_ms",
    "query_latency_ms",
    "activation_ms",
    "request_no",
    "query_id",
    "filter_name",
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
    "candidate_validity_predicate",
    "candidate_validity_predicate_sha256",
    "self_exclusion_contract",
    "scan_limit",
    "sqlens_build_id",
    "vector_so_sha256",
}
FIXED_D3_EVENT_FIELDS = (
    "d3_adaptive_page_builds_delta",
    "d3_adaptive_bloom_builds_delta",
    "d3_adaptive_exact_builds_delta",
    "d3_adaptive_refinements_delta",
    "d3_adaptive_rejections_delta",
    "d3_fragment_builds_delta",
)
CONFIG_FIELDS = (
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
    "candidate_validity_predicate",
    "candidate_validity_predicate_sha256",
    "self_exclusion_contract",
    "scan_limit",
)
CSV_FIELDS = (
    "schema_version",
    "pair_id",
    "qualification_scope",
    "quality_gate_override",
    "dataset",
    "target_recall",
    "selection_status",
    "stock_status",
    "sqlens_status",
    "stock_config_id",
    "stock_config_sha256",
    "stock_ef_search",
    "stock_iterative_scan",
    "stock_max_scan_tuples",
    "stock_scan_mem_multiplier",
    "stock_guided_collect_target",
    "stock_traversal_guided_target",
    "stock_d2_page_access",
    "stock_d2_index_page_access",
    "stock_table",
    "stock_index",
    "stock_calibration_recall_mean",
    "stock_calibration_recall_ci95_low",
    "stock_calibration_recall_ci95_high",
    "stock_calibration_per_filter_recall_min",
    "stock_calibration_per_filter_recall_min_ci95_low",
    "stock_calibration_latency_mean_ms",
    "stock_calibration_latency_p95_ms",
    "stock_calibration_selection_latency_ms",
    "stock_calibration_selection_latency_metric",
    "stock_calibration_recurring_activation_ms",
    "stock_calibration_fixed_activation_excess_ms",
    "sqlens_config_id",
    "sqlens_config_sha256",
    "sqlens_ef_search",
    "sqlens_iterative_scan",
    "sqlens_max_scan_tuples",
    "sqlens_scan_mem_multiplier",
    "sqlens_guided_collect_target",
    "sqlens_traversal_guided_target",
    "sqlens_d2_page_access",
    "sqlens_d2_index_page_access",
    "sqlens_table",
    "sqlens_index",
    "sqlens_calibration_recall_mean",
    "sqlens_calibration_recall_ci95_low",
    "sqlens_calibration_recall_ci95_high",
    "sqlens_calibration_per_filter_recall_min",
    "sqlens_calibration_per_filter_recall_min_ci95_low",
    "sqlens_calibration_latency_mean_ms",
    "sqlens_calibration_latency_p95_ms",
    "sqlens_calibration_selection_latency_ms",
    "sqlens_calibration_selection_latency_metric",
    "sqlens_calibration_recurring_activation_ms",
    "sqlens_calibration_fixed_activation_excess_ms",
)


class SelectionError(RuntimeError):
    """Calibration evidence cannot be admitted to the formal selector."""


@dataclass(frozen=True)
class ReleaseContract:
    path: Path
    sha256: str
    contract_id: str
    build_id: str
    vector_so_sha256: str


@dataclass(frozen=True)
class CalibrationConfig:
    artifact_tag: str
    dataset: str
    family: str
    mode: str
    config_id: str
    config: Mapping[str, object]
    config_sha256: str
    raw_path: Path
    raw_sha256: str
    plan_path: Path
    plan_sha256: str
    workload_path: Path
    workload_sha256: str
    truth_path: Path
    truth_sha256: str
    filters_path: Path
    filters_sha256: str
    requests: int
    filters: tuple[str, ...]
    recall_mean: float
    recall_ci95_low: float
    recall_ci95_high: float
    per_filter_recall_min: float
    per_filter_recall_min_ci95_low: float
    latency_mean_ms: float
    latency_p95_ms: float
    selection_latency_ms: float
    selection_latency_metric: str
    recurring_activation_ms: float
    fixed_activation_excess_ms: float


@dataclass(frozen=True)
class RequiredGridEvidence:
    path: Path
    sha256: str
    schema_version: int
    cell_keys: tuple[str, ...]
    raw_paths: tuple[Path, ...]
    serial_runner_manifests: tuple[Mapping[str, object], ...]
    cells: tuple[Mapping[str, object], ...]
    source_grid_plan: Mapping[str, object]
    dataset_config: Mapping[str, object]
    dataset_inputs: Mapping[str, Mapping[str, object]]


def _validate_qualification_scope(scope: str) -> str:
    if scope not in QUALIFICATION_SCOPES:
        raise SelectionError(f"unknown qualification scope: {scope}")
    return scope


def qualification_metric(scope: str) -> str:
    """Return the auditable quality condition selected by ``scope``."""
    _validate_qualification_scope(scope)
    if scope == QUALIFICATION_SCOPE_AGGREGATE:
        return "bootstrap_aggregate_recall_ci95_low"
    return (
        "bootstrap_aggregate_recall_ci95_low_and_"
        "bootstrap_min_per_filter_recall_ci95_low"
    )


def qualifies(config: CalibrationConfig, target: float, scope: str) -> bool:
    """Apply the requested admission contract to one calibrated config."""
    _validate_qualification_scope(scope)
    if config.recall_ci95_low < target:
        return False
    return (
        scope == QUALIFICATION_SCOPE_AGGREGATE
        or config.per_filter_recall_min_ci95_low >= target
    )


def qualification_floor(config: CalibrationConfig, scope: str) -> float:
    """Largest target admissible by a configuration under ``scope``."""
    _validate_qualification_scope(scope)
    if scope == QUALIFICATION_SCOPE_AGGREGATE:
        return config.recall_ci95_low
    return min(
        config.recall_ci95_low,
        config.per_filter_recall_min_ci95_low,
    )


def expected_requests_for_scope(scope: str) -> int:
    _validate_qualification_scope(scope)
    if scope == QUALIFICATION_SCOPE_AGGREGATE:
        return LEGACY_EXPECTED_REQUESTS
    return FORMAL_EXPECTED_REQUESTS


def expected_observations_per_filter(scope: str) -> int | None:
    _validate_qualification_scope(scope)
    if scope == QUALIFICATION_SCOPE_AGGREGATE:
        return None
    return FORMAL_OBSERVATIONS_PER_FILTER


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    )


def _number(value: object, label: str, *, lower: float | None = None) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number) or (lower is not None and number < lower):
        raise SelectionError(f"{label} is invalid: {value!r}")
    return number


def _integer(value: object, label: str, *, lower: int = 0) -> int:
    text = str(value).strip()
    try:
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{label} is not an integer: {value!r}") from exc
    if str(number) != text or number < lower:
        raise SelectionError(f"{label} is invalid: {value!r}")
    return number


def _text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SelectionError(f"{label} is empty")
    return text


def _sha(value: object, label: str) -> str:
    text = _text(value, label).lower()
    if not SHA256_RE.fullmatch(text):
        raise SelectionError(f"{label} is not a SHA-256 value")
    return text


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise SelectionError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def _format(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectionError(f"{label} must contain a JSON object: {path}")
    return value


def load_release_contract(path: Path) -> ReleaseContract:
    payload = read_json(path, "release contract")
    contract_id = _text(payload.get("contract_id"), "release contract ID")
    build_id = _text(payload.get("expected_sqlens_build_id"), "expected SQLens build ID")
    vector_sha = _sha(payload.get("expected_vector_so_sha256"), "expected vector.so SHA")
    contract_match = RELEASE_TAG_RE.search(contract_id)
    build_match = RELEASE_TAG_RE.search(build_id)
    if (
        contract_match is None
        or build_match is None
        or contract_match.group(1) != build_match.group(1)
    ):
        raise SelectionError(
            "release contract ID and SQLens build ID must bind the same "
            "explicit release tag"
        )
    return ReleaseContract(path.resolve(), sha256_file(path), contract_id, build_id, vector_sha)


def _read_csv(
    path: Path,
    required_fields: set[str],
) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise SelectionError(f"cannot read calibration CSV {path}: {exc}") from exc
    if not fields or not rows or len(fields) != len(set(fields)):
        raise SelectionError(f"calibration CSV is empty or malformed: {path}")
    if any(None in row for row in rows):
        raise SelectionError(f"calibration CSV has rows wider than its header: {path}")
    missing = sorted(required_fields - set(fields))
    if missing:
        raise SelectionError(f"calibration CSV is missing fields {missing}: {path}")
    return fields, rows


def _prewarm_complete(plan: Mapping[str, Any]) -> bool:
    prewarm = plan.get("relation_prewarm")
    records = prewarm.get("records") if isinstance(prewarm, Mapping) else None
    return (
        isinstance(prewarm, Mapping)
        and prewarm.get("enabled") is True
        and prewarm.get("complete") is True
        and isinstance(records, list)
        and len(records) == 3
        and all(
            isinstance(record, Mapping)
            and _integer(record.get("expected_blocks"), "prewarm expected blocks", lower=1)
            == _integer(record.get("warmed_blocks"), "prewarm warmed blocks", lower=1)
            for record in records
        )
    )


def _identity_matches(identity: object, release: ReleaseContract, label: str) -> bool:
    if not isinstance(identity, Mapping):
        return False
    return (
        identity.get("exact_match") is True
        and identity.get("expected_build_id") == release.build_id
        and identity.get("observed_build_id") == release.build_id
        and identity.get("expected_vector_so_sha256") == release.vector_so_sha256
        and identity.get("observed_vector_so_sha256") == release.vector_so_sha256
    )


def _filename_metadata(path: Path) -> re.Match[str]:
    match = CSV_RE.fullmatch(path.name)
    if match is None:
        raise SelectionError(f"invalid calibration filename: {path}")
    if (match.group("family") in CAP_FAMILIES) != bool(match.group("cap")):
        raise SelectionError(
            f"calibration filename has an invalid SQLens scan cap: {path}"
        )
    if (match.group("family") in TARGET_FAMILIES) != bool(
        match.group("target")
    ):
        raise SelectionError(
            f"calibration filename has an invalid SQLens traversal target: {path}"
        )
    return match


def calibration_cell_key(path: Path) -> str:
    match = _filename_metadata(path)
    return (
        f"{match.group('dataset')}:{match.group('family')}:"
        f"ef{match.group('ef')}:cap{match.group('cap') or 'none'}:"
        f"target{match.group('target') or 'none'}"
    )


def _contract_path(value: object, contract_path: Path, label: str) -> Path:
    path = Path(_text(value, label))
    if not path.is_absolute():
        path = contract_path.parent / path
    return path.resolve()


def _rooted_path(value: object, label: str) -> Path:
    path = Path(_text(value, label))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _bound_file(
    binding: object,
    contract_path: Path,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(binding, Mapping):
        raise SelectionError(f"required-grid contract lacks {label} binding")
    path = _contract_path(binding.get("path"), contract_path, f"{label} path")
    expected_sha = _sha(binding.get("sha256"), f"{label} SHA")
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise SelectionError(f"{label} path/SHA mismatch: {path}")
    return path, expected_sha


def _validate_required_grid_sources(
    payload: Mapping[str, Any],
    contract_path: Path,
    release: ReleaseContract,
    qualification_scope: str,
    targets: Sequence[float],
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, Mapping[str, object]],
]:
    grid_plan_path, grid_plan_sha = _bound_file(
        payload.get("source_grid_plan"),
        contract_path,
        "required-grid source_grid_plan",
    )
    dataset_config_path, dataset_config_sha = _bound_file(
        payload.get("dataset_config"),
        contract_path,
        "required-grid dataset_config",
    )
    source_binding = payload["source_grid_plan"]
    assert isinstance(source_binding, Mapping)
    source_plan = read_json(grid_plan_path, "required-grid source grid plan")
    if (
        source_plan.get("schema_version") != 1
        or _text(source_plan.get("plan_id"), "source grid plan plan_id")
        != _text(source_binding.get("plan_id"), "source_grid_plan plan_id")
        or _rooted_path(
            source_plan.get("dataset_config"),
            "source grid plan dataset_config",
        )
        != dataset_config_path
        or _rooted_path(
            source_plan.get("release_contract"),
            "source grid plan release_contract",
        )
        != release.path
        or source_plan.get("qualification_scope") != qualification_scope
    ):
        raise SelectionError(
            f"required-grid source grid plan binding mismatch: {grid_plan_path}"
        )
    source_targets = tuple(
        sorted(
            _number(value, "source grid plan target", lower=0.0)
            for value in source_plan.get("targets", ())
        )
    )
    if source_targets != tuple(sorted(targets)):
        raise SelectionError(
            f"required-grid source grid plan targets mismatch: {grid_plan_path}"
        )

    dataset_config = read_json(
        dataset_config_path, "required-grid dataset config"
    )
    if (
        _rooted_path(
            dataset_config.get("release_contract"),
            "dataset config release_contract",
        )
        != release.path
    ):
        raise SelectionError(
            f"required-grid dataset config release mismatch: {dataset_config_path}"
        )
    protocol = dataset_config.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("qualification_scope") != qualification_scope
        or protocol.get("calibration_requests") != FORMAL_EXPECTED_REQUESTS
        or protocol.get("calibration_observations_per_predicate")
        != FORMAL_OBSERVATIONS_PER_FILTER
    ):
        raise SelectionError(
            f"required-grid dataset config protocol mismatch: {dataset_config_path}"
        )
    datasets = dataset_config.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise SelectionError(
            f"required-grid dataset config has no datasets: {dataset_config_path}"
        )
    dataset_inputs: dict[str, Mapping[str, object]] = {}
    for dataset, raw_config in datasets.items():
        label = f"dataset config {dataset}"
        if not isinstance(dataset, str) or not dataset or not isinstance(
            raw_config, Mapping
        ):
            raise SelectionError(
                f"required-grid dataset config entry is malformed: {dataset_config_path}"
            )
        bindings: dict[str, object] = {}
        for name, field in (
            ("workload", "calibration_workload_csv"),
            ("truth", "truth_csv"),
            ("filters", "filters_csv"),
        ):
            artifact_path = _rooted_path(raw_config.get(field), f"{label} {field}")
            if not artifact_path.is_file():
                raise SelectionError(
                    f"{label} {name} file is missing: {artifact_path}"
                )
            bindings[f"{name}_path"] = artifact_path
            bindings[f"{name}_sha256"] = sha256_file(artifact_path)
        dataset_inputs[dataset] = bindings

    return (
        {
            "path": str(grid_plan_path),
            "sha256": grid_plan_sha,
            "plan_id": _text(
                source_binding.get("plan_id"), "source_grid_plan plan_id"
            ),
        },
        {
            "path": str(dataset_config_path),
            "sha256": dataset_config_sha,
        },
        dataset_inputs,
    )


def _release_sha_from_runner_manifest(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> str:
    release_binding = manifest.get("release_contract")
    if not isinstance(release_binding, Mapping):
        raise SelectionError(
            f"serial runner manifest lacks release contract binding: {manifest_path}"
        )
    return _sha(
        release_binding.get("sha256"),
        f"serial runner manifest release SHA: {manifest_path}",
    )


def _validate_global_db_isolation(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    isolation = manifest.get("database_isolation")
    if not isinstance(isolation, Mapping):
        raise SelectionError(
            f"serial runner manifest lacks global DB isolation evidence: {manifest_path}"
        )
    if (
        isolation.get("parallel_db_cells") is not False
        or isolation.get("lock_required") is not True
        or isolation.get("lock_acquired") is not True
        or isolation.get("held_through_completion") is not True
        or not str(isolation.get("lock_path") or "").strip()
        or not str(isolation.get("lock_protocol") or "").strip()
        or not str(isolation.get("lock_owner_token") or "").strip()
    ):
        raise SelectionError(
            "serial runner manifest does not prove parallel_db_cells=false "
            f"with a required lock held through completion: {manifest_path}"
        )


def load_required_grid_contract(
    path: Path,
    release: ReleaseContract,
    qualification_scope: str,
    targets: Sequence[float],
    raw_paths: Sequence[Path],
) -> RequiredGridEvidence:
    """Validate the frozen formal grid and its serial execution evidence."""
    _validate_qualification_scope(qualification_scope)
    path = path.resolve()
    payload = read_json(path, "required-grid contract")
    if payload.get("contract_type") != REQUIRED_GRID_CONTRACT_TYPE:
        raise SelectionError(
            f"required-grid contract has an invalid contract_type: {path}"
        )
    schema_version = _integer(
        payload.get("schema_version"),
        "required-grid contract schema_version",
        lower=1,
    )
    if payload.get("grid_complete") is not True:
        raise SelectionError(f"required-grid contract is not complete: {path}")
    if payload.get("qualification_scope") != qualification_scope:
        raise SelectionError(
            f"required-grid qualification scope mismatch: {path}"
        )
    contract_targets = tuple(
        sorted(
            _number(value, "required-grid target", lower=0.0)
            for value in payload.get("targets", ())
        )
    )
    if (
        contract_targets != tuple(sorted(targets))
        or len(contract_targets) != len(set(contract_targets))
        or any(value > 1.0 for value in contract_targets)
    ):
        raise SelectionError(f"required-grid targets mismatch: {path}")
    release_binding = payload.get("release_contract")
    if not isinstance(release_binding, Mapping):
        raise SelectionError(
            f"required-grid contract lacks release binding: {path}"
        )
    bound_release_path = _contract_path(
        release_binding.get("path"),
        path,
        "required-grid release contract path",
    )
    if (
        bound_release_path != release.path
        or _sha(
            release_binding.get("sha256"),
            "required-grid release contract SHA",
        )
        != release.sha256
    ):
        raise SelectionError(
            f"required-grid release contract path/SHA mismatch: {path}"
        )
    (
        source_grid_plan,
        dataset_config,
        dataset_inputs,
    ) = _validate_required_grid_sources(
        payload,
        path,
        release,
        qualification_scope,
        targets,
    )

    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise SelectionError(f"required-grid contract has no cells: {path}")
    expected_raw_paths: set[Path] = set()
    cell_keys: set[str] = set()
    manifest_cells: dict[Path, set[Path]] = defaultdict(set)
    manifest_expected_sha: dict[Path, str] = {}
    normalized_cells: dict[Path, Mapping[str, object]] = {}
    recovered_isolation_paths: set[Path] = set()
    for index, cell in enumerate(cells):
        label = f"required-grid cell {index}"
        if not isinstance(cell, Mapping):
            raise SelectionError(f"{label} is malformed: {path}")
        raw_binding = cell.get("raw_csv")
        plan_binding = cell.get("input_plan")
        runner_binding = cell.get("serial_runner_manifest")
        if not all(
            isinstance(binding, Mapping)
            for binding in (raw_binding, plan_binding, runner_binding)
        ):
            raise SelectionError(f"{label} lacks artifact bindings: {path}")
        assert isinstance(raw_binding, Mapping)
        assert isinstance(plan_binding, Mapping)
        assert isinstance(runner_binding, Mapping)
        raw_path = _contract_path(
            raw_binding.get("path"), path, f"{label} raw path"
        )
        plan_path = _contract_path(
            plan_binding.get("path"), path, f"{label} plan path"
        )
        manifest_path = _contract_path(
            runner_binding.get("path"), path, f"{label} runner manifest path"
        )
        if raw_path in expected_raw_paths:
            raise SelectionError(f"duplicate required-grid raw path: {raw_path}")
        expected_raw_paths.add(raw_path)
        expected_plan_path = raw_path.with_name(raw_path.name + ".plan.json")
        if plan_path != expected_plan_path:
            raise SelectionError(
                f"{label} input plan path does not match raw path: {path}"
            )
        for artifact_path, binding, artifact_label in (
            (raw_path, raw_binding, "raw"),
            (plan_path, plan_binding, "plan"),
            (manifest_path, runner_binding, "runner manifest"),
        ):
            if (
                not artifact_path.is_file()
                or _sha(
                    binding.get("sha256"),
                    f"{label} {artifact_label} SHA",
                )
                != sha256_file(artifact_path)
            ):
                raise SelectionError(
                    f"{label} {artifact_label} path/SHA mismatch: {artifact_path}"
                )
        cell_key = _text(cell.get("cell_key"), f"{label} cell_key")
        if cell_key != calibration_cell_key(raw_path) or cell_key in cell_keys:
            raise SelectionError(
                f"{label} has a duplicate or noncanonical cell_key: {cell_key}"
            )
        cell_keys.add(cell_key)
        normalized_cells[raw_path] = {
            "cell_key": cell_key,
            "raw_path": raw_path,
            "raw_sha256": _sha(
                raw_binding.get("sha256"), f"{label} raw SHA"
            ),
            "plan_path": plan_path,
            "plan_sha256": _sha(
                plan_binding.get("sha256"), f"{label} plan SHA"
            ),
            "runner_manifest_path": manifest_path,
        }
        if cell.get("database_isolation_evidence") == "recovered_missing":
            recovery_reason = cell.get("isolation_recovery_reason")
            if not isinstance(recovery_reason, str) or not recovery_reason:
                raise SelectionError(
                    f"{label} recovered isolation evidence lacks a reason"
                )
            recovered_isolation_paths.add(raw_path)
        manifest_cells[manifest_path].add(raw_path)
        manifest_sha = _sha(
            runner_binding.get("sha256"),
            f"{label} runner manifest SHA",
        )
        previous = manifest_expected_sha.setdefault(manifest_path, manifest_sha)
        if previous != manifest_sha:
            raise SelectionError(
                f"required-grid binds one runner manifest to multiple SHAs: {manifest_path}"
            )

    observed_raw_paths = {raw.resolve() for raw in raw_paths}
    missing = sorted(str(value) for value in expected_raw_paths - observed_raw_paths)
    extra = sorted(str(value) for value in observed_raw_paths - expected_raw_paths)
    if missing or extra:
        raise SelectionError(
            "required-grid input set mismatch; "
            f"missing={missing}, extra={extra}"
        )

    manifest_bindings: list[Mapping[str, object]] = []
    for manifest_path in sorted(manifest_cells):
        manifest = read_json(manifest_path, "serial runner manifest")
        if (
            manifest.get("status") != "complete"
            or manifest.get("requested_slice_complete") is not True
        ):
            raise SelectionError(
                f"serial runner manifest requested slice is incomplete: {manifest_path}"
            )
        if _release_sha_from_runner_manifest(manifest, manifest_path) != release.sha256:
            raise SelectionError(
                f"serial runner manifest release SHA mismatch: {manifest_path}"
            )
        manifest_expected_paths = manifest_cells[manifest_path]
        manifest_is_recovered = manifest_expected_paths.issubset(
            recovered_isolation_paths
        )
        if not manifest_is_recovered:
            _validate_global_db_isolation(manifest, manifest_path)
        schedule = manifest.get("schedule")
        if not isinstance(schedule, list) or not schedule:
            raise SelectionError(
                f"serial runner manifest has no schedule: {manifest_path}"
            )
        schedule_paths: set[Path] = set()
        for cell in schedule:
            if not isinstance(cell, Mapping):
                raise SelectionError(
                    f"serial runner manifest has malformed schedule: {manifest_path}"
                )
            raw_path = Path(
                _text(cell.get("raw"), "serial runner cell raw path")
            ).resolve()
            if raw_path not in recovered_isolation_paths:
                _validate_global_db_isolation(cell, manifest_path)
            if raw_path in schedule_paths:
                raise SelectionError(
                    f"serial runner manifest has duplicate cell: {raw_path}"
                )
            schedule_paths.add(raw_path)
            plan_path = Path(
                _text(cell.get("plan"), "serial runner cell plan path")
            ).resolve()
            normalized = normalized_cells.get(raw_path)
            if normalized is None:
                raise SelectionError(
                    f"serial runner manifest cell is outside required grid: {raw_path}"
                )
            if "cell_key" in cell and (
                _text(cell.get("cell_key"), "serial runner cell_key")
                != normalized["cell_key"]
            ):
                raise SelectionError(
                    f"serial runner manifest cell_key mismatch: {raw_path}"
                )
            if "plan_sha256" in cell and (
                _sha(
                    cell.get("plan_sha256"),
                    "serial runner cell plan SHA",
                )
                != normalized["plan_sha256"]
            ):
                raise SelectionError(
                    f"serial runner manifest plan SHA mismatch: {raw_path}"
                )
            recorded_raw_sha = cell.get("raw_sha256")
            raw_sha_matches = (
                recorded_raw_sha is None
                and raw_path in recovered_isolation_paths
            ) or (
                recorded_raw_sha is not None
                and _sha(
                    recorded_raw_sha,
                    "serial runner cell raw SHA",
                )
                == normalized["raw_sha256"]
            )
            if (
                cell.get("status") != "complete"
                or raw_path not in manifest_cells[manifest_path]
                or plan_path != raw_path.with_name(raw_path.name + ".plan.json")
                or not raw_sha_matches
            ):
                raise SelectionError(
                    f"serial runner manifest cell status/path/SHA mismatch: {raw_path}"
                )
        if schedule_paths != manifest_cells[manifest_path]:
            raise SelectionError(
                "serial runner manifest requested slice does not equal its "
                f"required-grid cells: {manifest_path}"
            )
        manifest_bindings.append(
            {
                "path": str(manifest_path),
                "sha256": manifest_expected_sha[manifest_path],
                "cells": sorted(
                    calibration_cell_key(raw_path)
                    for raw_path in manifest_cells[manifest_path]
                ),
            }
        )

    return RequiredGridEvidence(
        path=path,
        sha256=sha256_file(path),
        schema_version=schema_version,
        cell_keys=tuple(sorted(cell_keys)),
        raw_paths=tuple(sorted(expected_raw_paths)),
        serial_runner_manifests=tuple(manifest_bindings),
        cells=tuple(
            normalized_cells[path] for path in sorted(normalized_cells)
        ),
        source_grid_plan=source_grid_plan,
        dataset_config=dataset_config,
        dataset_inputs=dataset_inputs,
    )


def _family_modes(family: str) -> set[str]:
    if family == FAMILY_BOTH_OFF:
        return {MODE_STOCK, MODE_SQLENS}
    if family == FAMILY_STOCK_STRICT:
        return {MODE_STOCK}
    if family == FAMILY_SQLENS_CAP:
        return {MODE_SQLENS}
    if family == FAMILY_STOCK_CAP:
        return {MODE_STOCK}
    if family == FAMILY_SQLENS_TARGET:
        return {MODE_SQLENS}
    raise SelectionError(f"unknown calibration family: {family}")


def _plan_namespace_metadata(plan: Mapping[str, Any], plan_path: Path) -> re.Match[str]:
    namespace = _text(
        plan.get("d3_fragment_store_namespace"),
        "calibration plan fragment-store namespace",
    )
    match = PLAN_NAMESPACE_RE.fullmatch(namespace)
    if match is None:
        raise SelectionError(
            f"calibration plan has malformed fragment-store namespace: {plan_path}"
        )
    if match.group("phase") != "calibration":
        raise SelectionError(
            f"calibration plan namespace is not a calibration artifact: {plan_path}"
        )
    return match


def _validate_plan_workload_shape(
    plan: Mapping[str, Any],
    plan_path: Path,
    raw_rows: int,
    family: str,
    qualification_scope: str,
) -> None:
    query_contract = plan.get("query_contract")
    if not isinstance(query_contract, Mapping):
        raise SelectionError(f"input plan lacks query input bindings: {plan_path}")

    expected_requests = expected_requests_for_scope(qualification_scope)
    for field in ("expected_workload_requests", "workload_requests"):
        if _integer(query_contract.get(field), f"input plan {field}", lower=1) != expected_requests:
            raise SelectionError(
                f"input plan {field} does not match {expected_requests} for "
                f"{qualification_scope}: {plan_path}"
            )

    expected_modes = _family_modes(family)
    expected_rows = expected_requests * len(expected_modes)
    if raw_rows != expected_rows:
        raise SelectionError(
            f"input plan/raw row count disagrees with filename family: {plan_path}"
        )

    warmup = plan.get("warmup_evidence")
    lifecycle = plan.get("execution_lifecycle")
    expected_warmups = (
        EXPECTED_FILTERS if MODE_STOCK in expected_modes else 0
    )
    if not isinstance(warmup, list) or len(warmup) != expected_warmups:
        raise SelectionError(
            f"input plan warmup count does not match {expected_warmups}: {plan_path}"
        )
    warmup_filters = {
        _text(item.get("filter_name"), "input plan warmup filter name")
        for item in warmup
        if isinstance(item, Mapping)
    }
    if len(warmup_filters) != expected_warmups or any(
        not isinstance(item, Mapping) or item.get("status") != "complete"
        for item in warmup
    ):
        raise SelectionError(f"input plan warmup filter evidence is incomplete: {plan_path}")
    if not isinstance(lifecycle, Mapping):
        raise SelectionError(f"input plan lacks execution lifecycle evidence: {plan_path}")
    if _integer(lifecycle.get("warmup_expected"), "input plan warmup_expected", lower=0) != expected_warmups:
        raise SelectionError(f"input plan warmup_expected disagrees with filter count: {plan_path}")
    if _integer(lifecycle.get("warmup_observed"), "input plan warmup_observed", lower=0) != expected_warmups:
        raise SelectionError(f"input plan warmup_observed disagrees with filter count: {plan_path}")


def validate_plan(
    plan_path: Path,
    raw_path: Path,
    raw_rows: int,
    raw_sha256: str,
    release: ReleaseContract,
    qualification_scope: str = DEFAULT_QUALIFICATION_SCOPE,
) -> dict[str, Any]:
    _validate_qualification_scope(qualification_scope)
    plan = read_json(plan_path, "calibration input plan")
    filename = _filename_metadata(raw_path)
    namespace = _plan_namespace_metadata(plan, plan_path)
    release_match = RELEASE_TAG_RE.search(release.contract_id)
    if release_match is None:
        raise SelectionError("release contract lacks an explicit release tag")
    expected_namespace_tag = f"r{release_match.group(1)}"
    if namespace.group("artifact_tag") != expected_namespace_tag:
        raise SelectionError(
            "input plan namespace release tag disagrees with release contract: "
            f"{plan_path}"
        )
    for field in ("dataset", "family", "ef", "cap", "target"):
        if namespace.group(field) != filename.group(field):
            raise SelectionError(
                f"input plan {field} disagrees with calibration filename: {plan_path}"
            )
    query_contract = plan.get("query_contract")
    query_table = query_contract.get("query_table") if isinstance(query_contract, Mapping) else None
    query_table_name = _text(query_table, "input plan query_table").lower().rsplit(".", 1)[-1]
    if not query_table_name.startswith(filename.group("dataset")):
        raise SelectionError(
            f"input plan dataset disagrees with calibration filename: {plan_path}"
        )
    checks = plan.get("checks")
    if not isinstance(checks, list) or not checks:
        raise SelectionError(f"input plan lacks plan checks: {plan_path}")
    family = filename.group("family")
    expected_modes = _family_modes(family)
    observed_modes: set[str] = set()
    observed_efs: set[int] = set()
    observed_iterative: set[str] = set()
    observed_max_scan_tuples: set[int] = set()
    observed_traversal_targets: set[int] = set()
    observed_guided_collect_targets: set[int] = set()
    for check in checks:
        if not isinstance(check, Mapping):
            raise SelectionError(f"input plan has malformed plan check: {plan_path}")
        observed_modes.add(_text(check.get("mode"), "input plan check mode"))
        config = check.get("config")
        if not isinstance(config, Mapping):
            raise SelectionError(f"input plan check lacks config: {plan_path}")
        observed_efs.add(_integer(config.get("ef_search"), "input plan check ef_search", lower=1))
        observed_iterative.add(_text(config.get("iterative_scan"), "input plan check iterative_scan"))
        if family in CAP_FAMILIES:
            observed_max_scan_tuples.add(
                _integer(
                    config.get("max_scan_tuples"),
                    "input plan check max_scan_tuples",
                    lower=1,
                )
            )
        if family in TARGET_FAMILIES:
            observed_traversal_targets.add(
                _integer(
                    config.get("traversal_guided_target"),
                    "input plan check traversal_guided_target",
                    lower=1,
                )
            )
            observed_guided_collect_targets.add(
                _integer(
                    config.get("guided_collect_target"),
                    "input plan check guided_collect_target",
                    lower=1,
                )
            )
    if observed_modes != expected_modes or observed_efs != {int(filename.group("ef"))}:
        raise SelectionError(
            f"input plan scan family/ef evidence disagrees with calibration filename: {plan_path}"
        )
    expected_iterative = (
        "strict_order"
        if family in {FAMILY_STOCK_STRICT, FAMILY_STOCK_CAP}
        else "off"
    )
    if observed_iterative != {expected_iterative}:
        raise SelectionError(f"input plan iterative scan family disagrees with filename: {plan_path}")
    if family in CAP_FAMILIES and observed_max_scan_tuples != {
        int(filename.group("cap"))
    }:
        raise SelectionError(
            f"input plan scan cap disagrees with filename: {plan_path}"
        )
    if family in TARGET_FAMILIES and (
        observed_traversal_targets != {int(filename.group("target"))}
        or observed_guided_collect_targets != {int(filename.group("ef"))}
    ):
        raise SelectionError(
            f"input plan traversal target disagrees with filename: {plan_path}"
        )
    _validate_plan_workload_shape(
        plan,
        plan_path,
        raw_rows,
        family,
        qualification_scope,
    )
    if plan.get("status") != "complete":
        raise SelectionError(f"input plan is not complete: {plan_path}")
    if plan.get("output_sha256") != raw_sha256 or _integer(
        plan.get("output_rows"), "input plan output rows", lower=1
    ) != raw_rows:
        raise SelectionError(f"input plan output SHA/row binding failed: {plan_path}")
    if not _prewarm_complete(plan):
        raise SelectionError(f"input plan lacks complete relation prewarm evidence: {plan_path}")
    query_contract = plan.get("query_contract")
    if not isinstance(query_contract, Mapping):
        raise SelectionError(f"input plan lacks query input bindings: {plan_path}")
    for label, path_field, sha_field in (
        ("workload", "workload_csv", "workload_sha256"),
        ("truth", "truth_csv", "truth_sha256"),
        ("filters", "filters_csv", "filters_sha256"),
    ):
        bound_path = Path(
            _text(query_contract.get(path_field), f"{label} path")
        ).resolve()
        if (
            not bound_path.is_file()
            or _sha(query_contract.get(sha_field), f"{label} SHA")
            != sha256_file(bound_path)
        ):
            raise SelectionError(
                f"input plan {label} path/SHA binding failed: {plan_path}"
            )
    d2_proof = plan.get("d2_graph_proof_input")
    if (
        not isinstance(d2_proof, Mapping)
        or _sha(
            query_contract.get("d2_graph_proof_input_sha256"),
            "D2 graph proof input SHA",
        )
        != sha256_json(d2_proof)
    ):
        raise SelectionError(
            f"input plan D2 graph proof binding failed: {plan_path}"
        )
    execution_sources = plan.get("execution_sources")
    if not isinstance(execution_sources, Mapping) or set(execution_sources) != {
        "core_runner",
        "orchestrator",
    }:
        raise SelectionError(
            f"input plan lacks exact execution-source bindings: {plan_path}"
        )
    for source_name, binding in execution_sources.items():
        if not isinstance(binding, Mapping):
            raise SelectionError(
                f"input plan source binding is malformed: {plan_path}:{source_name}"
            )
        source_path = Path(
            _text(binding.get("path"), f"{source_name} source path")
        ).resolve()
        if (
            not source_path.is_file()
            or _sha(binding.get("sha256"), f"{source_name} source SHA")
            != sha256_file(source_path)
        ):
            raise SelectionError(
                f"input plan source path/SHA binding failed: "
                f"{plan_path}:{source_name}"
            )
    for field in ("sqlens_runtime_identity_startup", "sqlens_runtime_identity_final"):
        if not _identity_matches(plan.get(field), release, field):
            raise SelectionError(f"input plan release identity failed: {plan_path}:{field}")
    runtime = plan.get("runtime_sqlens_identity_evidence")
    if not isinstance(runtime, list) or not runtime or not all(
        _identity_matches(item, release, "runtime identity") for item in runtime
    ):
        raise SelectionError(f"input plan lacks exact runtime release evidence: {plan_path}")
    return plan


def _normalized_config(row: Mapping[str, str], label: str) -> dict[str, object]:
    return {
        "ef_search": _integer(row["ef_search"], f"{label}:ef_search", lower=1),
        "iterative_scan": _text(row["iterative_scan"], f"{label}:iterative_scan"),
        "max_scan_tuples": _integer(row["max_scan_tuples"], f"{label}:max_scan_tuples", lower=1),
        "scan_mem_multiplier": _number(row["scan_mem_multiplier"], f"{label}:scan_mem_multiplier", lower=0.0),
        "guided_collect_target": _integer(row["guided_collect_target"], f"{label}:guided_collect_target", lower=1),
        "traversal_guided_target": _integer(row["traversal_guided_target"], f"{label}:traversal_guided_target", lower=1),
        "d2_page_access": _text(row["d2_page_access"], f"{label}:d2_page_access"),
        "d2_index_page_access": _text(row["d2_index_page_access"], f"{label}:d2_index_page_access"),
        "table": _text(row["table"], f"{label}:table"),
        "index": _text(row["index"], f"{label}:index"),
        "candidate_validity_predicate": _text(row["candidate_validity_predicate"], f"{label}:candidate predicate"),
        "candidate_validity_predicate_sha256": _sha(row["candidate_validity_predicate_sha256"], f"{label}:candidate predicate SHA"),
        "self_exclusion_contract": _text(row["self_exclusion_contract"], f"{label}:self exclusion"),
        "scan_limit": _integer(row["scan_limit"], f"{label}:scan_limit", lower=1),
    }


def _bootstrap_metrics(
    recalls: Sequence[float],
    filters: Sequence[str],
    query_ids: Sequence[str],
    *,
    samples: int,
    seed: int,
    require_formal_cartesian: bool,
) -> tuple[float, float, float, float, float]:
    if (
        len(recalls) != len(filters)
        or len(recalls) != len(query_ids)
        or not recalls
    ):
        raise SelectionError("bootstrap inputs are empty or inconsistent")
    mean = statistics.fmean(recalls)
    per_filter: dict[str, list[float]] = defaultdict(list)
    for recall, filter_name in zip(recalls, filters):
        per_filter[filter_name].append(recall)
    per_filter_min = min(statistics.fmean(values) for values in per_filter.values())
    grouped_indexes: dict[str, list[int]] = defaultdict(list)
    for index, query_id in enumerate(query_ids):
        grouped_indexes[query_id].append(index)
    if require_formal_cartesian:
        expected_filters = set(per_filter)
        if (
            len(grouped_indexes) != FORMAL_OBSERVATIONS_PER_FILTER
            or len(expected_filters) != EXPECTED_FILTERS
            or any(
                len(indexes) != EXPECTED_FILTERS
                or {filters[index] for index in indexes} != expected_filters
                for indexes in grouped_indexes.values()
            )
        ):
            raise SelectionError(
                "formal calibration is not a 200-query x 14-predicate "
                "Cartesian workload"
            )
    clusters = [
        grouped_indexes[query_id] for query_id in sorted(grouped_indexes)
    ]
    random_source = random.Random(seed)
    mean_samples: list[float] = []
    min_samples: list[float] = []
    for _ in range(samples):
        chosen = [
            index
            for _ in clusters
            for index in clusters[random_source.randrange(len(clusters))]
        ]
        mean_samples.append(statistics.fmean(recalls[index] for index in chosen))
        sampled_filters: dict[str, list[float]] = defaultdict(list)
        for index in chosen:
            sampled_filters[filters[index]].append(recalls[index])
        min_samples.append(min(statistics.fmean(values) for values in sampled_filters.values()))
    return (
        mean,
        _percentile(mean_samples, 0.025),
        _percentile(mean_samples, 0.975),
        per_filter_min,
        _percentile(min_samples, 0.025),
    )


def _calibration_input_bindings(
    plan: Mapping[str, Any],
    plan_path: Path,
) -> Mapping[str, object]:
    query_contract = plan.get("query_contract")
    if not isinstance(query_contract, Mapping):
        raise SelectionError(f"input plan lacks query input bindings: {plan_path}")
    result: dict[str, object] = {}
    for name, path_field, sha_field in (
        ("workload", "workload_csv", "workload_sha256"),
        ("truth", "truth_csv", "truth_sha256"),
        ("filters", "filters_csv", "filters_sha256"),
    ):
        bound_path = Path(
            _text(query_contract.get(path_field), f"{name} path")
        ).resolve()
        result[f"{name}_path"] = bound_path
        result[f"{name}_sha256"] = _sha(
            query_contract.get(sha_field), f"{name} SHA"
        )
    return result


def _load_calibration_workload(
    bindings: Mapping[str, object],
    *,
    qualification_scope: str,
) -> Mapping[int, tuple[str, str]]:
    workload_path = bindings["workload_path"]
    assert isinstance(workload_path, Path)
    _, rows = _read_csv(
        workload_path, {"request_no", "query_id", "filter_name"}
    )
    expected_requests = expected_requests_for_scope(qualification_scope)
    if len(rows) != expected_requests:
        raise SelectionError(
            f"calibration workload must contain {expected_requests} rows: "
            f"{workload_path}"
        )
    requests: dict[int, tuple[str, str]] = {}
    for row_no, row in enumerate(rows, start=2):
        label = f"{workload_path}:{row_no}"
        request_no = _integer(
            row["request_no"], f"{label}:request_no", lower=0
        )
        signature = (
            _text(row["query_id"], f"{label}:query_id"),
            _text(row["filter_name"], f"{label}:filter_name"),
        )
        if request_no in requests:
            raise SelectionError(
                f"duplicate calibration workload request_no: {label}"
            )
        requests[request_no] = signature
    if set(requests) != set(range(expected_requests)):
        raise SelectionError(
            "calibration workload request_no values must be contiguous "
            f"0..{expected_requests - 1}: {workload_path}"
        )
    return requests


def _selection_latency(
    mode: str,
    rows: Sequence[Mapping[str, str]],
    *,
    label: str,
    qualification_scope: str = DEFAULT_QUALIFICATION_SCOPE,
) -> tuple[float, str, float, float]:
    _validate_qualification_scope(qualification_scope)
    observed = [
        _number(row["end_to_end_ms"], f"{label}:end_to_end_ms", lower=0.0)
        for row in rows
    ]
    activations = [
        _number(row["activation_ms"], f"{label}:activation_ms", lower=0.0)
        for row in rows
    ]
    observed_metric = (
        "observed_q2800_mean_end_to_end_ms"
        if qualification_scope == QUALIFICATION_SCOPE_FORMAL
        else "observed_q200_mean_end_to_end_ms"
    )
    if mode == MODE_STOCK:
        return (
            statistics.fmean(observed),
            observed_metric,
            statistics.fmean(activations),
            0.0,
        )

    fixed = [
        any(
            _integer(row[field], f"{label}:{field}", lower=0) > 0
            for field in FIXED_D3_EVENT_FIELDS
        )
        for row in rows
    ]
    recurring_samples = [
        activation
        for activation, is_fixed in zip(activations, fixed)
        if not is_fixed
    ]
    recurring_activation = statistics.median(
        recurring_samples or activations
    )
    fixed_excess = sum(
        max(activation - recurring_activation, 0.0)
        for activation, is_fixed in zip(activations, fixed)
        if is_fixed
    )
    return (
        statistics.fmean(observed),
        observed_metric,
        recurring_activation,
        fixed_excess,
    )


def _discover_csvs(
    input_dir: Path,
    explicit: Sequence[Path],
    extra_input_dirs: Sequence[Path] = (),
) -> list[Path]:
    if explicit:
        if extra_input_dirs:
            raise SelectionError(
                "--input cannot be combined with --extra-input-dir"
            )
        candidates = list(explicit)
    else:
        candidates = []
        for directory in (input_dir, *extra_input_dirs):
            candidates.extend(
                path
                for path in directory.glob("figure5_r*_calibration_*_ef*.csv")
                if CSV_RE.fullmatch(path.name)
            )
        candidates.sort()
    if not candidates:
        raise SelectionError(f"no figure5_rNN calibration CSVs found in {input_dir}")
    seen: set[Path] = set()
    result: list[Path] = []
    for raw in candidates:
        raw = raw.resolve()
        if raw in seen:
            raise SelectionError(f"duplicate calibration input: {raw}")
        seen.add(raw)
        if not raw.is_file():
            raise SelectionError(f"calibration CSV does not exist: {raw}")
        if CSV_RE.fullmatch(raw.name) is None:
            raise SelectionError(f"calibration CSV name violates the figure5_rNN contract: {raw.name}")
        result.append(raw)
    artifact_tags = {_filename_metadata(path).group("artifact_tag") for path in result}
    if len(artifact_tags) != 1:
        raise SelectionError(
            "mixed calibration artifact prefixes are not allowed: "
            + ", ".join(sorted(artifact_tags))
        )
    return result


def load_calibration_csv(
    raw_path: Path,
    release: ReleaseContract,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    qualification_scope: str = DEFAULT_QUALIFICATION_SCOPE,
) -> list[CalibrationConfig]:
    _validate_qualification_scope(qualification_scope)
    expected_requests = expected_requests_for_scope(qualification_scope)
    expected_per_filter = expected_observations_per_filter(qualification_scope)
    match = _filename_metadata(raw_path)
    artifact_tag = match.group("artifact_tag")
    dataset, family, filename_ef = match.group("dataset", "family", "ef")
    required_fields = set(RAW_REQUIRED_FIELDS)
    if MODE_SQLENS in _family_modes(family):
        required_fields.update(FIXED_D3_EVENT_FIELDS)
    fields, rows = _read_csv(raw_path, required_fields)
    raw_sha = sha256_file(raw_path)
    plan_path = raw_path.with_name(raw_path.name + ".plan.json")
    if not plan_path.is_file():
        raise SelectionError(f"missing input plan for calibration CSV: {plan_path}")
    plan = validate_plan(
        plan_path,
        raw_path,
        len(rows),
        raw_sha,
        release,
        qualification_scope,
    )
    input_bindings = _calibration_input_bindings(plan, plan_path)
    workload_requests = _load_calibration_workload(
        input_bindings,
        qualification_scope=qualification_scope,
    )
    expected_modes = _family_modes(family)
    modes = {row["mode"] for row in rows}
    if modes != expected_modes:
        raise SelectionError(
            f"{raw_path} must contain exactly modes {sorted(expected_modes)}, observed {sorted(modes)}"
        )
    output: list[CalibrationConfig] = []
    for mode in sorted(modes):
        mode_rows = [row for row in rows if row["mode"] == mode]
        signatures: set[tuple[int, str, str]] = set()
        request_numbers: set[int] = set()
        filter_counts: dict[str, int] = defaultdict(int)
        filters: list[str] = []
        query_ids: list[str] = []
        recalls: list[float] = []
        latencies: list[float] = []
        configs: set[str] = set()
        normalized_config: dict[str, object] | None = None
        for row_no, row in enumerate(mode_rows, start=2):
            label = f"{raw_path}:{row_no}:{mode}"
            if str(row["error"] or "").strip():
                raise SelectionError(f"calibration row reports an error: {label}")
            if row["sqlens_build_id"] != release.build_id or row["vector_so_sha256"] != release.vector_so_sha256:
                raise SelectionError(f"calibration row release identity failed: {label}")
            request_no = _integer(row["request_no"], f"{label}:request_no", lower=0)
            query_id = _text(row["query_id"], f"{label}:query_id")
            filter_name = _text(row["filter_name"], f"{label}:filter_name")
            signature = (request_no, query_id, filter_name)
            if signature in signatures:
                raise SelectionError(f"duplicate calibration request identity: {label}")
            signatures.add(signature)
            request_numbers.add(request_no)
            expected_signature = workload_requests.get(request_no)
            if expected_signature != (query_id, filter_name):
                raise SelectionError(
                    "calibration row does not match bound workload "
                    f"request_no mapping: {label}"
                )
            recall = _number(row["recall"], f"{label}:recall", lower=0.0)
            if recall > 1.0:
                raise SelectionError(f"recall exceeds one: {label}")
            recalls.append(recall)
            latencies.append(_number(row["end_to_end_ms"], f"{label}:end_to_end_ms", lower=0.0))
            filters.append(filter_name)
            query_ids.append(query_id)
            filter_counts[filter_name] += 1
            current = _normalized_config(row, label)
            configs.add(sha256_json(current))
            normalized_config = current
        if len(signatures) != expected_requests or len(request_numbers) != expected_requests:
            raise SelectionError(
                f"{raw_path}:{mode} must contain {expected_requests} unique calibration "
                f"requests for {qualification_scope}"
            )
        observed_filters = tuple(sorted(set(filters)))
        if len(observed_filters) != EXPECTED_FILTERS:
            raise SelectionError(
                f"{raw_path}:{mode} must contain {EXPECTED_FILTERS} filters, observed={len(observed_filters)}"
            )
        if expected_per_filter is not None and any(
            filter_counts[filter_name] != expected_per_filter
            for filter_name in observed_filters
        ):
            observed = ", ".join(
                f"{filter_name}={filter_counts[filter_name]}"
                for filter_name in observed_filters
            )
            raise SelectionError(
                f"{raw_path}:{mode} must contain exactly {expected_per_filter} "
                f"observations per filter for {qualification_scope}, observed: {observed}"
            )
        if len(configs) != 1 or normalized_config is None:
            raise SelectionError(f"exact config fields drift within {raw_path}:{mode}")
        if int(normalized_config["ef_search"]) != int(filename_ef):
            raise SelectionError(f"filename ef and CSV ef_search disagree: {raw_path}:{mode}")
        if family == FAMILY_BOTH_OFF and mode == MODE_STOCK and normalized_config["iterative_scan"] != "off":
            raise SelectionError(f"both_off Stock config is not iterative_scan=off: {raw_path}")
        if family == FAMILY_BOTH_OFF and mode == MODE_SQLENS and normalized_config["iterative_scan"] != "off":
            raise SelectionError(f"both_off SQLens config is not iterative_scan=off: {raw_path}")
        if family == FAMILY_SQLENS_CAP and (
            mode != MODE_SQLENS
            or normalized_config["iterative_scan"] != "off"
            or int(normalized_config["max_scan_tuples"])
            != int(match.group("cap"))
        ):
            raise SelectionError(
                f"sqlens_cap config disagrees with filename: {raw_path}"
            )
        if family == FAMILY_STOCK_CAP and (
            mode != MODE_STOCK
            or normalized_config["iterative_scan"] != "strict_order"
            or int(normalized_config["max_scan_tuples"])
            != int(match.group("cap"))
        ):
            raise SelectionError(
                f"stock_cap config disagrees with filename: {raw_path}"
            )
        if family == FAMILY_SQLENS_TARGET and (
            mode != MODE_SQLENS
            or normalized_config["iterative_scan"] != "off"
            or int(normalized_config["guided_collect_target"])
            != int(filename_ef)
            or int(normalized_config["traversal_guided_target"])
            != int(match.group("target"))
        ):
            raise SelectionError(
                f"sqlens_target config disagrees with filename: {raw_path}"
            )
        mean, low, high, per_filter_min, per_filter_low = _bootstrap_metrics(
            recalls,
            filters,
            query_ids,
            samples=bootstrap_samples,
            seed=bootstrap_seed + int(normalized_config["ef_search"]) + (0 if mode == MODE_STOCK else 100_000),
            require_formal_cartesian=(
                qualification_scope == QUALIFICATION_SCOPE_FORMAL
            ),
        )
        (
            selection_latency,
            selection_metric,
            recurring_activation,
            fixed_activation_excess,
        ) = _selection_latency(
            mode,
            mode_rows,
            label=f"{raw_path}:{mode}",
            qualification_scope=qualification_scope,
        )
        config_id = f"{family}_ef{normalized_config['ef_search']}"
        if family in CAP_FAMILIES:
            config_id += f"_cap{normalized_config['max_scan_tuples']}"
        if family in TARGET_FAMILIES:
            config_id += (
                f"_target{normalized_config['traversal_guided_target']}"
            )
        output.append(CalibrationConfig(
            artifact_tag=artifact_tag,
            dataset=dataset,
            family=family,
            mode=mode,
            config_id=config_id,
            config=normalized_config,
            config_sha256=next(iter(configs)),
            raw_path=raw_path,
            raw_sha256=raw_sha,
            plan_path=plan_path.resolve(),
            plan_sha256=sha256_file(plan_path),
            workload_path=input_bindings["workload_path"],
            workload_sha256=str(input_bindings["workload_sha256"]),
            truth_path=input_bindings["truth_path"],
            truth_sha256=str(input_bindings["truth_sha256"]),
            filters_path=input_bindings["filters_path"],
            filters_sha256=str(input_bindings["filters_sha256"]),
            requests=len(signatures),
            filters=observed_filters,
            recall_mean=mean,
            recall_ci95_low=low,
            recall_ci95_high=high,
            per_filter_recall_min=per_filter_min,
            per_filter_recall_min_ci95_low=per_filter_low,
            latency_mean_ms=statistics.fmean(latencies),
            latency_p95_ms=_percentile(latencies, 0.95),
            selection_latency_ms=selection_latency,
            selection_latency_metric=selection_metric,
            recurring_activation_ms=recurring_activation,
            fixed_activation_excess_ms=fixed_activation_excess,
        ))
    if family == FAMILY_BOTH_OFF:
        stock_signatures = {
            (row["request_no"], row["query_id"], row["filter_name"])
            for row in rows if row["mode"] == MODE_STOCK
        }
        sqlens_signatures = {
            (row["request_no"], row["query_id"], row["filter_name"])
            for row in rows if row["mode"] == MODE_SQLENS
        }
        if stock_signatures != sqlens_signatures:
            raise SelectionError(f"both_off modes are not paired over the same calibration trace: {raw_path}")
    return output


def nondominated_frontier(
    configs: Sequence[CalibrationConfig],
    qualification_scope: str = DEFAULT_QUALIFICATION_SCOPE,
) -> list[CalibrationConfig]:
    """Return configs not dominated under the requested quality contract."""
    _validate_qualification_scope(qualification_scope)
    frontier: list[CalibrationConfig] = []
    for candidate in configs:
        dominated = any(
            other is not candidate
            and other.recall_ci95_low >= candidate.recall_ci95_low
            and (
                qualification_scope == QUALIFICATION_SCOPE_AGGREGATE
                or other.per_filter_recall_min_ci95_low
                >= candidate.per_filter_recall_min_ci95_low
            )
            and other.selection_latency_ms <= candidate.selection_latency_ms
            and (
                other.recall_ci95_low > candidate.recall_ci95_low
                or (
                    qualification_scope != QUALIFICATION_SCOPE_AGGREGATE
                    and other.per_filter_recall_min_ci95_low
                    > candidate.per_filter_recall_min_ci95_low
                )
                or other.selection_latency_ms < candidate.selection_latency_ms
            )
            for other in configs
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda item: (
            qualification_floor(item, qualification_scope),
            item.selection_latency_ms,
            item.config_id,
        ),
    )


def select_config(
    configs: Sequence[CalibrationConfig],
    target: float,
    qualification_scope: str = DEFAULT_QUALIFICATION_SCOPE,
) -> CalibrationConfig | None:
    qualified = [
        item for item in configs
        if qualifies(item, target, qualification_scope)
    ]
    if not qualified:
        return None
    return min(
        qualified,
        key=lambda item: (
            item.selection_latency_ms,
            -qualification_floor(item, qualification_scope),
            item.config_id,
        ),
    )


def _evenly_spaced(values: Sequence[float], count: int) -> list[float]:
    if len(values) <= count:
        return list(values)
    indexes = {
        round(index * (len(values) - 1) / (count - 1))
        for index in range(count)
    }
    return [values[index] for index in sorted(indexes)]


def _coverage_aware_states(
    states: Sequence[tuple[float, str, str]],
    *,
    min_points: int,
    max_points: int,
) -> list[tuple[float, str, str]]:
    """Keep a compact, deterministic set covering both arms' configurations."""
    if len(states) <= max_points:
        return list(states)
    selected = {0, len(states) - 1}
    stock_seen = {states[index][1] for index in selected}
    sqlens_seen = {states[index][2] for index in selected}
    target_min = states[0][0]
    target_span = states[-1][0] - target_min

    def spacing(index: int) -> float:
        if target_span > 0:
            position = (states[index][0] - target_min) / target_span
            selected_positions = [
                (states[item][0] - target_min) / target_span
                for item in selected
            ]
        else:
            position = index / max(1, len(states) - 1)
            selected_positions = [
                item / max(1, len(states) - 1)
                for item in selected
            ]
        return min(abs(position - item) for item in selected_positions)

    while len(stock_seen) < min_points or len(sqlens_seen) < min_points:
        candidates: list[tuple[int, float, int]] = []
        for index, (_, stock_sha, sqlens_sha) in enumerate(states):
            if index in selected:
                continue
            gain = int(
                len(stock_seen) < min_points and stock_sha not in stock_seen
            ) + int(
                len(sqlens_seen) < min_points and sqlens_sha not in sqlens_seen
            )
            if gain:
                candidates.append((gain, spacing(index), index))
        if not candidates:
            raise SelectionError(
                "coverage-aware downsampling cannot reach the per-arm "
                f"distinct-point gate: stock={len(stock_seen)}, "
                f"sqlens={len(sqlens_seen)}, required={min_points}"
            )
        _, _, chosen = max(
            candidates,
            key=lambda item: (item[0], item[1], -item[2]),
        )
        selected.add(chosen)
        stock_seen.add(states[chosen][1])
        sqlens_seen.add(states[chosen][2])
        if len(selected) > max_points:
            raise SelectionError(
                "coverage-aware downsampling requires more than "
                f"{max_points} points to preserve {min_points} distinct "
                "configurations per arm; increase --max-points-per-dataset"
            )
    return [states[index] for index in sorted(selected)]


def distinct_pair_targets(
    stock: Sequence[CalibrationConfig],
    sqlens: Sequence[CalibrationConfig],
    *,
    min_points: int,
    max_points: int,
    target_floor: float,
    qualification_scope: str = DEFAULT_QUALIFICATION_SCOPE,
) -> list[float]:
    """Choose targets that map to distinct independently selected arm pairs."""
    if min_points < 2 or max_points < min_points:
        raise SelectionError("distinct-pair point bounds are invalid")
    _validate_qualification_scope(qualification_scope)
    common_max = min(
        max(qualification_floor(item, qualification_scope) for item in stock),
        max(qualification_floor(item, qualification_scope) for item in sqlens),
    )
    if not 0.0 <= target_floor <= common_max:
        raise SelectionError(
            f"target floor {target_floor:.6f} exceeds the common attainable "
            f"recall LCB {common_max:.6f}"
        )
    boundaries = sorted({
        qualification_floor(item, qualification_scope)
        for item in (*stock, *sqlens)
        if target_floor <= qualification_floor(item, qualification_scope) < common_max
    })
    candidates = [target_floor]
    candidates.extend(
        math.nextafter(boundary, math.inf)
        for boundary in boundaries
        if math.nextafter(boundary, math.inf) <= common_max
    )
    candidates.append(common_max)

    states: list[tuple[float, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for target in candidates:
        stock_choice = select_config(stock, target, qualification_scope)
        sqlens_choice = select_config(sqlens, target, qualification_scope)
        if stock_choice is None or sqlens_choice is None:
            continue
        pair = (stock_choice.config_sha256, sqlens_choice.config_sha256)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        states.append(
            (target, stock_choice.config_sha256, sqlens_choice.config_sha256)
        )
    if len(states) < min_points:
        raise SelectionError(
            "calibration grid yields only "
            f"{len(states)} distinct matched pairs; at least {min_points} are required"
        )
    stock_points = len({stock_sha for _, stock_sha, _ in states})
    sqlens_points = len({sqlens_sha for _, _, sqlens_sha in states})
    if min(stock_points, sqlens_points) < min_points:
        raise SelectionError(
            "calibration grid does not yield enough distinct points per arm: "
            f"stock={stock_points}, sqlens={sqlens_points}, required={min_points}"
        )
    selected_states = _coverage_aware_states(
        states,
        min_points=min_points,
        max_points=max_points,
    )
    selected_stock = len({stock_sha for _, stock_sha, _ in selected_states})
    selected_sqlens = len({sqlens_sha for _, _, sqlens_sha in selected_states})
    if min(selected_stock, selected_sqlens) < min_points:
        raise SelectionError(
            "point downsampling drops below the per-arm distinct-point gate: "
            f"stock={selected_stock}, sqlens={selected_sqlens}, "
            f"required={min_points}; increase --max-points-per-dataset"
        )
    return [target for target, _, _ in selected_states]


def _config_cells(prefix: str, config: CalibrationConfig | None) -> dict[str, object]:
    fields = {
        f"{prefix}_config_id": "",
        f"{prefix}_config_sha256": "",
        f"{prefix}_ef_search": "",
        f"{prefix}_iterative_scan": "",
        f"{prefix}_max_scan_tuples": "",
        f"{prefix}_scan_mem_multiplier": "",
        f"{prefix}_guided_collect_target": "",
        f"{prefix}_traversal_guided_target": "",
        f"{prefix}_d2_page_access": "",
        f"{prefix}_d2_index_page_access": "",
        f"{prefix}_table": "",
        f"{prefix}_index": "",
        f"{prefix}_calibration_recall_mean": "",
        f"{prefix}_calibration_recall_ci95_low": "",
        f"{prefix}_calibration_recall_ci95_high": "",
        f"{prefix}_calibration_per_filter_recall_min": "",
        f"{prefix}_calibration_per_filter_recall_min_ci95_low": "",
        f"{prefix}_calibration_latency_mean_ms": "",
        f"{prefix}_calibration_latency_p95_ms": "",
        f"{prefix}_calibration_selection_latency_ms": "",
        f"{prefix}_calibration_selection_latency_metric": "",
        f"{prefix}_calibration_recurring_activation_ms": "",
        f"{prefix}_calibration_fixed_activation_excess_ms": "",
    }
    if config is None:
        return fields
    config_fields = {
        "ef_search", "iterative_scan", "max_scan_tuples", "scan_mem_multiplier",
        "guided_collect_target", "traversal_guided_target", "d2_page_access",
        "d2_index_page_access", "table", "index",
    }
    fields.update({
        f"{prefix}_config_id": config.config_id,
        f"{prefix}_config_sha256": config.config_sha256,
        **{f"{prefix}_{name}": config.config[name] for name in config_fields},
        f"{prefix}_calibration_recall_mean": config.recall_mean,
        f"{prefix}_calibration_recall_ci95_low": config.recall_ci95_low,
        f"{prefix}_calibration_recall_ci95_high": config.recall_ci95_high,
        f"{prefix}_calibration_per_filter_recall_min": config.per_filter_recall_min,
        f"{prefix}_calibration_per_filter_recall_min_ci95_low": config.per_filter_recall_min_ci95_low,
        f"{prefix}_calibration_latency_mean_ms": config.latency_mean_ms,
        f"{prefix}_calibration_latency_p95_ms": config.latency_p95_ms,
        f"{prefix}_calibration_selection_latency_ms": config.selection_latency_ms,
        f"{prefix}_calibration_selection_latency_metric": config.selection_latency_metric,
        f"{prefix}_calibration_recurring_activation_ms": config.recurring_activation_ms,
        f"{prefix}_calibration_fixed_activation_excess_ms": config.fixed_activation_excess_ms,
    })
    return fields


def build_measurement_plan(
    configs: Sequence[CalibrationConfig],
    release: ReleaseContract,
    targets: Sequence[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    target_policy: str = "fixed",
    qualification_scope: str = DEFAULT_QUALIFICATION_SCOPE,
    min_points_per_dataset: int = DEFAULT_MIN_POINTS_PER_DATASET,
    max_points_per_dataset: int = DEFAULT_MAX_POINTS_PER_DATASET,
    target_floor: float = min(DEFAULT_TARGETS),
    required_grid: RequiredGridEvidence | None = None,
    aggregate_lcb_override_targets: Sequence[float] = (),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if target_policy not in TARGET_POLICIES:
        raise SelectionError(f"unknown target policy: {target_policy}")
    _validate_qualification_scope(qualification_scope)
    override_targets = {float(target) for target in aggregate_lcb_override_targets}
    if target_policy != "fixed" and override_targets:
        raise SelectionError("quality-gate overrides require fixed targets")
    if not override_targets.issubset({float(target) for target in targets}):
        raise SelectionError("quality-gate override targets must be selected targets")
    artifact_tags = {item.artifact_tag for item in configs}
    if len(artifact_tags) != 1:
        raise SelectionError(
            "measurement plan requires one calibration artifact prefix, observed: "
            + ", ".join(sorted(artifact_tags))
        )
    if required_grid is not None and {
        item.raw_path.resolve() for item in configs
    } != set(required_grid.raw_paths):
        raise SelectionError(
            "required-grid evidence does not exactly cover measurement-plan inputs"
        )
    cells_by_raw = (
        {
            cell["raw_path"]: cell
            for cell in required_grid.cells
        }
        if required_grid is not None
        else {}
    )
    by_dataset: dict[str, list[CalibrationConfig]] = defaultdict(list)
    for config in configs:
        by_dataset[config.dataset].append(config)
    for dataset, items in by_dataset.items():
        input_signatures = {
            (
                item.workload_path,
                item.workload_sha256,
                item.truth_path,
                item.truth_sha256,
                item.filters_path,
                item.filters_sha256,
            )
            for item in items
        }
        if len(input_signatures) != 1:
            raise SelectionError(
                f"{dataset} calibration cells do not share workload/truth/filter bindings"
            )
        if required_grid is None:
            continue
        expected = required_grid.dataset_inputs.get(dataset)
        if expected is None:
            raise SelectionError(
                f"{dataset} is absent from required-grid dataset config"
            )
        observed = next(iter(input_signatures))
        expected_signature = (
            expected["workload_path"],
            expected["workload_sha256"],
            expected["truth_path"],
            expected["truth_sha256"],
            expected["filters_path"],
            expected["filters_sha256"],
        )
        if observed != expected_signature:
            raise SelectionError(
                f"{dataset} calibration inputs disagree with required-grid dataset config"
            )
        for item in items:
            cell = cells_by_raw.get(item.raw_path.resolve())
            if (
                cell is None
                or cell["cell_key"] != calibration_cell_key(item.raw_path)
                or cell["raw_sha256"] != item.raw_sha256
                or cell["plan_path"] != item.plan_path
                or cell["plan_sha256"] != item.plan_sha256
            ):
                raise SelectionError(
                    f"{dataset} calibration artifact disagrees with normalized required-grid cell"
                )
    rows: list[dict[str, object]] = []
    frontier_by_dataset: dict[str, dict[str, list[dict[str, object]]]] = {}
    for dataset in sorted(by_dataset):
        items = by_dataset[dataset]
        stock = [item for item in items if item.mode == MODE_STOCK]
        sqlens = [item for item in items if item.mode == MODE_SQLENS]
        if not stock or not sqlens:
            raise SelectionError(f"{dataset} needs Stock and full SQLens calibration candidates")
        frontier_by_dataset[dataset] = {}
        for arm, candidates in (("stock", stock), ("sqlens", sqlens)):
            frontier_by_dataset[dataset][arm] = [
                {
                    "config_id": item.config_id,
                    "config_sha256": item.config_sha256,
                    "recall_ci95_low": item.recall_ci95_low,
                    "recall_mean": item.recall_mean,
                    "per_filter_recall_min": item.per_filter_recall_min,
                    "per_filter_recall_min_ci95_low": item.per_filter_recall_min_ci95_low,
                    "qualification_floor": qualification_floor(
                        item, qualification_scope
                    ),
                    "observed_latency_mean_ms": item.latency_mean_ms,
                    "selection_latency_ms": item.selection_latency_ms,
                    "selection_latency_metric": item.selection_latency_metric,
                }
                for item in nondominated_frontier(candidates, qualification_scope)
            ]
        dataset_targets = (
            list(targets)
            if target_policy == "fixed"
            else distinct_pair_targets(
                stock,
                sqlens,
                min_points=min_points_per_dataset,
                max_points=max_points_per_dataset,
                target_floor=target_floor,
                qualification_scope=qualification_scope,
            )
        )
        for target in dataset_targets:
            target_scope = (
                QUALIFICATION_SCOPE_AGGREGATE
                if target in override_targets
                else qualification_scope
            )
            stock_choice = select_config(stock, target, target_scope)
            sqlens_choice = select_config(sqlens, target, target_scope)
            if (
                target_policy == "fixed"
                and qualification_scope == QUALIFICATION_SCOPE_FORMAL
                and (stock_choice is None or sqlens_choice is None)
                and required_grid is None
            ):
                raise SelectionError(
                    "formal fixed-target selector cannot emit unattainable "
                    "without a complete required-grid contract"
                )
            stock_status = "selected" if stock_choice else "unattainable_on_calibration_grid"
            sqlens_status = "selected" if sqlens_choice else "unattainable_on_calibration_grid"
            status = "selected" if stock_choice and sqlens_choice else "unattainable_on_calibration_grid"
            stock_token = (
                stock_choice.config_sha256[:10] if stock_choice else "unattainable"
            )
            sqlens_token = (
                sqlens_choice.config_sha256[:10] if sqlens_choice else "unattainable"
            )
            pair_id = (
                f"{dataset}:scope_{qualification_scope}:recall_{target:.9f}:"
                f"{stock_token}:{sqlens_token}"
            )
            row: dict[str, object] = {
                "schema_version": 2,
                "pair_id": pair_id,
                "qualification_scope": qualification_scope,
                "quality_gate_override": (
                    "aggregate_lcb" if target in override_targets else ""
                ),
                "dataset": dataset,
                "target_recall": target,
                "selection_status": status,
                "stock_status": stock_status,
                "sqlens_status": sqlens_status,
            }
            row.update(_config_cells("stock", stock_choice))
            row.update(_config_cells("sqlens", sqlens_choice))
            rows.append(row)
    inputs = [
        {
            "raw_csv": str(item.raw_path),
            "raw_csv_sha256": item.raw_sha256,
            "input_plan": str(item.plan_path),
            "input_plan_sha256": item.plan_sha256,
            "artifact_tag": item.artifact_tag,
            "dataset": item.dataset,
            "family": item.family,
            "mode": item.mode,
            "config_id": item.config_id,
            "config_sha256": item.config_sha256,
        }
        for item in sorted(configs, key=lambda item: (item.dataset, item.mode, item.config_id, item.raw_path.name))
    ]
    required_grid_binding: Mapping[str, object] | None = None
    if required_grid is not None:
        required_grid_binding = {
            "path": str(required_grid.path),
            "sha256": required_grid.sha256,
            "schema_version": required_grid.schema_version,
            "cell_count": len(required_grid.cell_keys),
            "cell_keys": list(required_grid.cell_keys),
            "serial_runner_manifests": list(
                required_grid.serial_runner_manifests
            ),
            "source_grid_plan": dict(required_grid.source_grid_plan),
            "dataset_config": dict(required_grid.dataset_config),
        }
    unattainable_arms: list[dict[str, object]] = []
    for row in rows:
        dataset = str(row["dataset"])
        target = float(row["target_recall"])
        candidates = by_dataset[dataset]
        for arm, mode, status_field in (
            ("stock", MODE_STOCK, "stock_status"),
            ("sqlens", MODE_SQLENS, "sqlens_status"),
        ):
            if row[status_field] == "selected":
                continue
            arm_candidates = [item for item in candidates if item.mode == mode]
            unattainable_arms.append(
                {
                    "dataset": dataset,
                    "target_recall": target,
                    "arm": arm,
                    "status": "unattainable_on_complete_required_grid",
                    "candidate_configs": len(arm_candidates),
                    "candidate_config_sha256s": sorted(
                        item.config_sha256 for item in arm_candidates
                    ),
                    "maximum_qualification_floor": max(
                        qualification_floor(item, qualification_scope)
                        for item in arm_candidates
                    ),
                }
            )
    exhaustion_base: dict[str, object] = {
        "required_grid_contract_present": required_grid is not None,
        "required_grid_complete": required_grid is not None,
        "input_set_exact": required_grid is not None,
        "required_grid_contract_sha256": (
            required_grid.sha256 if required_grid is not None else None
        ),
        "required_grid_cell_keys_sha256": (
            sha256_json(list(required_grid.cell_keys))
            if required_grid is not None
            else None
        ),
        "qualification_scope": qualification_scope,
        "quality_gate_overrides": {
            f"{target:.9f}": "aggregate_lcb"
            for target in sorted(override_targets)
        },
        "targets": list(targets) if target_policy == "fixed" else [],
        "unattainable_arms": unattainable_arms,
    }
    exhaustion_proof = {
        **exhaustion_base,
        "proof_sha256": sha256_json(exhaustion_base),
    }
    observed_metric = (
        "observed_q2800_mean_end_to_end_ms"
        if qualification_scope == QUALIFICATION_SCOPE_FORMAL
        else "observed_q200_mean_end_to_end_ms"
    )
    plan: dict[str, object] = {
        "schema_version": 2,
        "runner_version": RUNNER_VERSION,
        "created_at": utc_now(),
        "artifact_valid": True,
        "artifact_prefix": {
            "format": "figure5_rNN",
            "tag": next(iter(artifact_tags)),
        },
        "execution_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "selection_policy": (
            "independent_lowest_calibration_score_with_"
            f"{qualification_metric(qualification_scope)};"
            f"stock={observed_metric};"
            f"sqlens={observed_metric}"
        ),
        "qualification_scope": qualification_scope,
        "target_policy": target_policy,
        "target_floor": target_floor if target_policy == "distinct_pairs" else None,
        "qualification_metric": qualification_metric(qualification_scope),
        "calibration_coverage_contract": {
            "requests_per_mode": expected_requests_for_scope(
                qualification_scope
            ),
            "filters": EXPECTED_FILTERS,
            "observations_per_filter": expected_observations_per_filter(
                qualification_scope
            ),
        },
        "targets": list(targets) if target_policy == "fixed" else [],
        "targets_by_dataset": {
            dataset: [row["target_recall"] for row in rows if row["dataset"] == dataset]
            for dataset in sorted(by_dataset)
        },
        "point_contract": {
            "min_distinct_pairs_per_dataset": (
                min_points_per_dataset if target_policy == "distinct_pairs" else None
            ),
            "max_distinct_pairs_per_dataset": (
                max_points_per_dataset if target_policy == "distinct_pairs" else None
            ),
        },
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": 0.95,
            "resampling_unit": "query_id_cluster",
            "formal_cluster_contract": "200_queries_x_14_predicates",
        },
        "release_contract": {
            "path": str(release.path),
            "sha256": release.sha256,
            "contract_id": release.contract_id,
            "expected_sqlens_build_id": release.build_id,
            "expected_vector_so_sha256": release.vector_so_sha256,
        },
        "required_grid_contract": required_grid_binding,
        "exhaustion_proof": exhaustion_proof,
        "inputs": inputs,
        "frontiers": frontier_by_dataset,
        "measurement_pairs": rows,
        "summary": {
            "datasets": sorted(by_dataset),
            "target_rows": len(rows),
            "selected_pairs": sum(row["selection_status"] == "selected" for row in rows),
            "unattainable_pairs": sum(row["selection_status"] != "selected" for row in rows),
        },
    }
    return rows, plan


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    import io
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=CSV_FIELDS, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _format(row.get(field)) for field in CSV_FIELDS})
    return target.getvalue().encode("utf-8")


def output_paths(prefix: Path) -> dict[str, Path]:
    return {
        "csv": Path(str(prefix) + ".csv"),
        "plan": Path(str(prefix) + ".json"),
        "manifest": Path(str(prefix) + ".manifest.json"),
    }


def publish_atomically(prefix: Path, rows: Sequence[Mapping[str, object]], plan: Mapping[str, object]) -> dict[str, Path]:
    with acquire_publish_lock(prefix):
        return _publish_atomically_locked(prefix, rows, plan)


@contextmanager
def acquire_publish_lock(prefix: Path) -> Iterable[object]:
    lock_path = Path(str(prefix) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SelectionError(f"another selector owns publish lock: {lock_path}") from exc
    try:
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _publish_atomically_locked(prefix: Path, rows: Sequence[Mapping[str, object]], plan: Mapping[str, object]) -> dict[str, Path]:
    if (
        plan.get("target_policy") == "fixed"
        and plan.get("qualification_scope") == QUALIFICATION_SCOPE_FORMAL
    ):
        required_grid = plan.get("required_grid_contract")
        exhaustion = plan.get("exhaustion_proof")
        if (
            not isinstance(required_grid, Mapping)
            or not isinstance(exhaustion, Mapping)
            or exhaustion.get("required_grid_complete") is not True
            or exhaustion.get("input_set_exact") is not True
        ):
            raise SelectionError(
                "formal fixed-target publication requires a complete "
                "required-grid contract and exhaustion proof"
            )
        required_grid_path = Path(
            _text(
                required_grid.get("path"),
                "formal required-grid contract path",
            )
        ).resolve()
        if (
            not required_grid_path.is_file()
            or _sha(
                required_grid.get("sha256"),
                "formal required-grid contract SHA",
            )
            != sha256_file(required_grid_path)
        ):
            raise SelectionError(
                "formal required-grid contract changed before publication"
            )
        runner_manifests = required_grid.get("serial_runner_manifests")
        if not isinstance(runner_manifests, list) or not runner_manifests:
            raise SelectionError(
                "formal required-grid contract lacks serial runner bindings"
            )
        for binding in runner_manifests:
            if not isinstance(binding, Mapping):
                raise SelectionError(
                    "formal serial runner manifest binding is malformed"
                )
            manifest_path = Path(
                _text(
                    binding.get("path"),
                    "formal serial runner manifest path",
                )
            ).resolve()
            if (
                not manifest_path.is_file()
                or _sha(
                    binding.get("sha256"),
                    "formal serial runner manifest SHA",
                )
                != sha256_file(manifest_path)
            ):
                raise SelectionError(
                    "formal serial runner manifest changed before publication"
                )
        proof_body = dict(exhaustion)
        proof_sha = proof_body.pop("proof_sha256", None)
        if proof_sha != sha256_json(proof_body):
            raise SelectionError(
                "formal fixed-target exhaustion proof SHA is invalid"
            )
        if plan.get("measurement_pairs") != list(rows):
            raise SelectionError(
                "formal measurement rows disagree with the audited plan"
            )
        expected_unattainable = {
            (
                str(row["dataset"]),
                float(row["target_recall"]),
                arm,
            )
            for row in rows
            for arm, status_field in (
                ("stock", "stock_status"),
                ("sqlens", "sqlens_status"),
            )
            if row[status_field] != "selected"
        }
        proof_unattainable = exhaustion.get("unattainable_arms")
        if not isinstance(proof_unattainable, list) or {
            (
                str(item.get("dataset")),
                float(item.get("target_recall")),
                str(item.get("arm")),
            )
            for item in proof_unattainable
            if isinstance(item, Mapping)
        } != expected_unattainable:
            raise SelectionError(
                "formal exhaustion proof does not match measurement rows"
            )
    paths = output_paths(prefix)
    parent = paths["manifest"].parent
    parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = _csv_bytes(rows)
    plan_with_binding = dict(plan)
    plan_with_binding["measurement_plan_csv"] = {
        "path": str(paths["csv"].resolve()),
        "sha256": sha256_bytes(csv_bytes),
        "rows": len(rows),
    }
    plan_bytes = (json.dumps(plan_with_binding, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest = {
        "schema_version": 2,
        "runner_version": RUNNER_VERSION,
        "artifact_valid": True,
        "created_at": utc_now(),
        "release_contract": plan_with_binding["release_contract"],
        "artifact_prefix": plan_with_binding.get("artifact_prefix"),
        "qualification_scope": plan_with_binding["qualification_scope"],
        "qualification_metric": plan_with_binding["qualification_metric"],
        "selection_provenance": {
            "qualification_scope": plan_with_binding["qualification_scope"],
            "qualification_metric": plan_with_binding["qualification_metric"],
            "measurement_pair_scope_bound": True,
        },
        "required_grid_contract": plan_with_binding.get(
            "required_grid_contract"
        ),
        "exhaustion_proof": plan_with_binding.get("exhaustion_proof"),
        "outputs": {
            "measurement_plan_csv": {"path": str(paths["csv"].resolve()), "sha256": sha256_bytes(csv_bytes), "rows": len(rows)},
            "measurement_plan_json": {"path": str(paths["plan"].resolve()), "sha256": sha256_bytes(plan_bytes)},
        },
        "input_bindings": plan_with_binding["inputs"],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    stage = Path(tempfile.mkdtemp(prefix=".figure5-selector-", dir=parent))
    try:
        staged = {"csv": stage / paths["csv"].name, "plan": stage / paths["plan"].name, "manifest": stage / paths["manifest"].name}
        staged["csv"].write_bytes(csv_bytes)
        staged["plan"].write_bytes(plan_bytes)
        staged["manifest"].write_bytes(manifest_bytes)
        for name in ("csv", "plan", "manifest"):
            os.replace(staged[name], paths[name])
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return paths


def parse_targets(value: str) -> tuple[float, ...]:
    targets = tuple(_number(item, "target recall", lower=0.0) for item in value.split(",") if item.strip())
    if not targets or any(target > 1.0 for target in targets) or len(set(targets)) != len(targets):
        raise argparse.ArgumentTypeError("targets must be distinct recall values in [0, 1]")
    return tuple(sorted(targets))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--extra-input-dir",
        type=Path,
        action="append",
        default=[],
        help="additional calibration directory; may be repeated",
    )
    parser.add_argument("--input", type=Path, action="append", default=[], help="explicit calibration CSV; disables directory discovery")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--required-grid-contract",
        type=Path,
        help=(
            "explicit serial required-grid JSON; mandatory for formal "
            "fixed-target publication"
        ),
    )
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_INPUT_DIR / "figure5_r35_matched_configs")
    parser.add_argument("--targets", type=parse_targets, default=DEFAULT_TARGETS)
    parser.add_argument(
        "--aggregate-lcb-override-targets",
        type=parse_targets,
        default=(),
        help=(
            "Explicit fixed targets admitted by aggregate calibration LCB "
            "instead of the formal minimum-predicate LCB."
        ),
    )
    parser.add_argument("--target-policy", choices=TARGET_POLICIES, default="fixed")
    parser.add_argument(
        "--qualification-scope",
        choices=QUALIFICATION_SCOPES,
        default=DEFAULT_QUALIFICATION_SCOPE,
        help=(
            "quality admission contract; global_min_predicate_lcb is the "
            "formal default, aggregate_lcb is legacy audit only"
        ),
    )
    parser.add_argument("--target-floor", type=float, default=min(DEFAULT_TARGETS))
    parser.add_argument(
        "--min-points-per-dataset",
        type=int,
        default=DEFAULT_MIN_POINTS_PER_DATASET,
    )
    parser.add_argument(
        "--max-points-per-dataset",
        type=int,
        default=DEFAULT_MAX_POINTS_PER_DATASET,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--execute", action="store_true", help="atomically publish CSV, plan JSON, and manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_samples < 100:
        raise SelectionError("bootstrap-samples must be at least 100")
    release = load_release_contract(args.contract.resolve())
    raw_paths = _discover_csvs(
        args.input_dir.resolve(),
        args.input,
        tuple(path.resolve() for path in args.extra_input_dir),
    )
    required_grid: RequiredGridEvidence | None = None
    if args.required_grid_contract is not None:
        required_grid = load_required_grid_contract(
            args.required_grid_contract.resolve(),
            release,
            args.qualification_scope,
            args.targets,
            raw_paths,
        )
    if (
        args.execute
        and args.target_policy == "fixed"
        and args.qualification_scope == QUALIFICATION_SCOPE_FORMAL
        and required_grid is None
    ):
        raise SelectionError(
            "formal fixed-target publication requires "
            "--required-grid-contract"
        )
    configs: list[CalibrationConfig] = []
    for raw_path in raw_paths:
        configs.extend(load_calibration_csv(
            raw_path, release,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            qualification_scope=args.qualification_scope,
        ))
    rows, plan = build_measurement_plan(
        configs, release, args.targets,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        target_policy=args.target_policy,
        qualification_scope=args.qualification_scope,
        min_points_per_dataset=args.min_points_per_dataset,
        max_points_per_dataset=args.max_points_per_dataset,
        target_floor=args.target_floor,
        required_grid=required_grid,
        aggregate_lcb_override_targets=args.aggregate_lcb_override_targets,
    )
    if args.execute:
        paths = publish_atomically(args.out_prefix.resolve(), rows, plan)
        print(json.dumps({"status": "published", **{name: str(path) for name, path in paths.items()}}, sort_keys=True))
    else:
        print(json.dumps({"status": "dry_run", "summary": plan["summary"], "out_prefix": str(args.out_prefix.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SelectionError as exc:
        print(f"selection error: {exc}", file=sys.stderr)
        raise SystemExit(2)
