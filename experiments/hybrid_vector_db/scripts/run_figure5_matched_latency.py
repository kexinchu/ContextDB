#!/usr/bin/env python3
"""Run formal q10k/r3 matched-latency Figure 5 pairs.

This is deliberately a small orchestration layer over
``pgvector_design1_design2_design3_selectivity_benchmark.py``.  It consumes
the selector's published CSV rather than retuning either arm: each selected
row supplies independently chosen Stock and full-SQLens search settings.
Every cell runs serially, uses the frozen q10k trace, and is admitted only
when its raw request evidence is complete and paired.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import run_figure5_frontier as frontier
    from .common_pg import pg_config_from_env
except ImportError:
    import run_figure5_frontier as frontier
    from common_pg import pg_config_from_env

import psycopg


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT / "experiments/hybrid_vector_db/configs/figure5_frontier_datasets.json"
)
DEFAULT_SELECTION_CSV = (
    ROOT / "results/hybrid_vector_db/figure5_r35/figure5_r35_matched_configs.csv"
)
RESULTS = ROOT / "results/hybrid_vector_db"
MODES = ("original", "design1_bloom_bfs_layout_d3")
EXPECTED_REQUESTS = 10_000
EXPECTED_REPEATS = 3
EXPECTED_ROWS = EXPECTED_REQUESTS * EXPECTED_REPEATS * len(MODES)
DEFAULT_BACKEND_CPU_LIST = "48-63"
FROZEN_DATASETS = ("amazon", "yfcc", "laion")
MIN_FORMAL_POINTS_PER_ARM_DATASET = 10
EXPECTED_FORMAL_PREDICATES = 14
REQUIRED_GRID_CONTRACT_TYPE = "figure5_formal_fixed_target_required_grid"
RUNNER_VERSION = "sqlens-figure5-matched-latency-orchestrator-r36-v5"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
QUALIFICATION_SCOPE_AGGREGATE = "aggregate_lcb"
QUALIFICATION_SCOPE_FORMAL = "global_min_predicate_lcb"
QUALIFICATION_SCOPES = (
    QUALIFICATION_SCOPE_AGGREGATE,
    QUALIFICATION_SCOPE_FORMAL,
)
# A q10k final workload must provide enough independent requests for every
# predicate before a per-predicate confidence interval can admit a paper cell.
MIN_FORMAL_PREDICATE_SAMPLES = 100
VALIDATOR_ONLY_COMPATIBLE_ORCHESTRATOR_SHA256 = {
    # The q10k Amazon 0.90 cell used this source.  The only subsequent change
    # replaced the offline O(n^2) paired-row audit with an equivalent O(n)
    # grouping pass; query construction and execution were unchanged.
    "86f3cb56896206cd5cf6ab6a973207cf6fb8b3fd0aaad75c9c9944d51a882eed",
}


class MatchedLatencyError(RuntimeError):
    """A matched-latency cell violates the frozen Figure 5 protocol."""


@dataclass(frozen=True)
class SelectedPair:
    pair_id: str
    dataset: str
    target_recall: float
    stock: dict[str, object]
    sqlens: dict[str, object]
    quality_gate_override: str = ""


def sha256_file(path: Path) -> str:
    return frontier.sha256_file(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    frontier.atomic_json(path, payload)


def utc_now() -> str:
    return frontier.utc_now()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatchedLatencyError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatchedLatencyError(f"{label} must be a JSON object: {path}")
    return value


def read_raw_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))
    except (OSError, csv.Error) as exc:
        raise MatchedLatencyError(f"cannot read raw CSV {path}: {exc}") from exc


def resolve_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def require_sha(value: object, label: str) -> str:
    text = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(text):
        raise MatchedLatencyError(f"{label} is not a SHA-256 value")
    return text


def require_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise MatchedLatencyError(f"{label} is empty")
    return text


def require_int(value: object, label: str, *, lower: int = 0) -> int:
    text = str(value).strip()
    try:
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise MatchedLatencyError(f"{label} is not an integer: {value!r}") from exc
    if str(number) != text or number < lower:
        raise MatchedLatencyError(f"{label} is invalid: {value!r}")
    return number


def require_float(value: object, label: str, *, lower: float = 0.0) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise MatchedLatencyError(f"{label} is not numeric: {value!r}") from exc
    if not number >= lower:
        raise MatchedLatencyError(f"{label} is invalid: {value!r}")
    return number


def require_qualification_scope(value: object, label: str) -> str:
    scope = require_text(value, label)
    if scope not in QUALIFICATION_SCOPES:
        raise MatchedLatencyError(
            f"{label} is unsupported: {scope!r}; expected one of "
            f"{list(QUALIFICATION_SCOPES)}"
        )
    return scope


def inferred_selection_paths(selection_csv: Path) -> tuple[Path, Path]:
    return (
        selection_csv.with_suffix(".json"),
        selection_csv.with_name(selection_csv.stem + ".manifest.json"),
    )


def require_matching_release_tag(contract_id: str, build_id: str) -> str:
    pattern = re.compile(r"(?:^|-)r([0-9]+)(?:-|$)")
    contract_match = pattern.search(contract_id)
    build_match = pattern.search(build_id)
    if (
        contract_match is None
        or build_match is None
        or contract_match.group(1) != build_match.group(1)
    ):
        raise MatchedLatencyError(
            "release contract ID and SQLens build ID must bind the same "
            "explicit release tag"
        )
    return f"r{contract_match.group(1)}"


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = frontier.load_config(path)
    except frontier.Figure5ContractError as exc:
        raise MatchedLatencyError(str(exc)) from exc
    identity = config["release_identity"]
    contract_id = require_text(identity.get("contract_id"), "release contract ID")
    build_id = require_text(
        identity.get("expected_sqlens_build_id"), "expected SQLens build ID"
    )
    require_sha(identity.get("expected_vector_so_sha256"), "expected vector.so SHA")
    require_matching_release_tag(contract_id, build_id)
    return config


def validate_required_grid_binding(
    *,
    required_grid_contract: Path,
    config_path: Path,
    selection_plan: Mapping[str, object],
    selection_manifest: Mapping[str, object],
    qualification_scope: str,
) -> dict[str, object]:
    """Bind selector, required grid, and active dataset config fail-closed."""
    required_grid_contract = required_grid_contract.resolve()
    config_path = config_path.resolve()
    if not required_grid_contract.is_file():
        raise MatchedLatencyError(
            f"required-grid contract is missing: {required_grid_contract}"
        )
    if not config_path.is_file():
        raise MatchedLatencyError(f"frontier config is missing: {config_path}")
    required_grid_sha = sha256_file(required_grid_contract)
    config_sha = sha256_file(config_path)
    plan_binding = selection_plan.get("required_grid_contract")
    manifest_binding = selection_manifest.get("required_grid_contract")
    if not isinstance(plan_binding, Mapping) or not isinstance(
        manifest_binding, Mapping
    ):
        raise MatchedLatencyError(
            "selector plan/manifest must both bind the required-grid contract"
        )
    for label, binding in (
        ("selector plan", plan_binding),
        ("selector manifest", manifest_binding),
    ):
        bound_path = resolve_path(
            require_text(
                binding.get("path"), f"{label} required-grid contract path"
            )
        ).resolve()
        if (
            bound_path != required_grid_contract
            or require_sha(
                binding.get("sha256"),
                f"{label} required-grid contract SHA",
            )
            != required_grid_sha
        ):
            raise MatchedLatencyError(
                f"{label} required-grid path/SHA binding failed"
            )
    grid = read_json(required_grid_contract, "required-grid contract")
    dataset_config = grid.get("dataset_config")
    if (
        grid.get("contract_type") != REQUIRED_GRID_CONTRACT_TYPE
        or grid.get("grid_complete") is not True
        or grid.get("qualification_scope") != qualification_scope
        or not isinstance(dataset_config, Mapping)
    ):
        raise MatchedLatencyError(
            "required-grid contract identity/status/scope is invalid"
        )
    grid_config_path = resolve_path(
        require_text(
            dataset_config.get("path"), "required-grid dataset config path"
        )
    ).resolve()
    if (
        grid_config_path != config_path
        or require_sha(
            dataset_config.get("sha256"), "required-grid dataset config SHA"
        )
        != config_sha
    ):
        raise MatchedLatencyError(
            "selection, required-grid, and active --config SHA do not agree"
        )
    return {
        "path": str(required_grid_contract),
        "sha256": required_grid_sha,
        "dataset_config_path": str(config_path),
        "dataset_config_sha256": config_sha,
        "contract_type": REQUIRED_GRID_CONTRACT_TYPE,
        "grid_complete": True,
        "qualification_scope": qualification_scope,
    }


def validate_selection_artifacts(
    selection_csv: Path,
    selection_plan: Path,
    selection_manifest: Path,
    config: Mapping[str, object],
    *,
    config_path: Path | None = None,
    required_grid_contract: Path | None = None,
) -> dict[str, object]:
    for label, path in (
        ("selection CSV", selection_csv),
        ("selection plan", selection_plan),
        ("selection manifest", selection_manifest),
    ):
        if not path.is_file():
            raise MatchedLatencyError(f"{label} is missing: {path}")
    csv_sha = sha256_file(selection_csv)
    plan_sha = sha256_file(selection_plan)
    manifest_sha = sha256_file(selection_manifest)
    plan = read_json(selection_plan, "selection plan")
    manifest = read_json(selection_manifest, "selection manifest")
    plan_csv = plan.get("measurement_plan_csv")
    outputs = manifest.get("outputs")
    if not isinstance(plan_csv, Mapping) or not isinstance(outputs, Mapping):
        raise MatchedLatencyError("selector artifacts lack CSV output bindings")
    manifest_csv = outputs.get("measurement_plan_csv")
    manifest_plan = outputs.get("measurement_plan_json")
    if not isinstance(manifest_csv, Mapping) or not isinstance(manifest_plan, Mapping):
        raise MatchedLatencyError("selector manifest lacks output SHA bindings")
    if (
        require_sha(plan_csv.get("sha256"), "selection plan CSV SHA") != csv_sha
        or require_sha(manifest_csv.get("sha256"), "selection manifest CSV SHA") != csv_sha
        or require_sha(manifest_plan.get("sha256"), "selection manifest plan SHA") != plan_sha
    ):
        raise MatchedLatencyError("selector CSV/plan SHA binding failed")
    if plan.get("artifact_valid") is not True or manifest.get("artifact_valid") is not True:
        raise MatchedLatencyError("selector artifacts are not marked valid")
    qualification_scope = require_qualification_scope(
        plan.get("qualification_scope"), "selector qualification_scope"
    )
    manifest_scope = manifest.get("qualification_scope")
    if manifest_scope is not None and require_qualification_scope(
        manifest_scope, "selector manifest qualification_scope"
    ) != qualification_scope:
        raise MatchedLatencyError("selector plan/manifest qualification scopes differ")
    plan_grid_binding = plan.get("required_grid_contract")
    if required_grid_contract is None:
        if not isinstance(plan_grid_binding, Mapping):
            raise MatchedLatencyError(
                "selector plan does not bind a required-grid contract"
            )
        required_grid_contract = resolve_path(
            require_text(
                plan_grid_binding.get("path"),
                "selector plan required-grid contract path",
            )
        )
    if config_path is None:
        grid_payload = read_json(
            required_grid_contract.resolve(), "required-grid contract"
        )
        dataset_config = grid_payload.get("dataset_config")
        if not isinstance(dataset_config, Mapping):
            raise MatchedLatencyError(
                "required-grid contract lacks dataset config binding"
            )
        config_path = resolve_path(
            require_text(
                dataset_config.get("path"),
                "required-grid dataset config path",
            )
        )
    required_grid_binding = validate_required_grid_binding(
        required_grid_contract=required_grid_contract,
        config_path=config_path,
        selection_plan=plan,
        selection_manifest=manifest,
        qualification_scope=qualification_scope,
    )
    with selection_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if "qualification_scope" not in set(reader.fieldnames or ()):
            raise MatchedLatencyError("selector CSV lacks qualification_scope")
        for row_no, row in enumerate(reader, start=2):
            if require_qualification_scope(
                row.get("qualification_scope"),
                f"selector CSV row {row_no} qualification_scope",
            ) != qualification_scope:
                raise MatchedLatencyError(
                    f"selector CSV row {row_no} qualification_scope disagrees with selector plan"
                )
    selector_source = plan.get("execution_source")
    if not isinstance(selector_source, Mapping):
        raise MatchedLatencyError("selector plan lacks execution-source binding")
    selector_source_path = resolve_path(selector_source.get("path")).resolve()
    if (
        not selector_source_path.is_file()
        or require_sha(
            selector_source.get("sha256"), "selector source SHA"
        )
        != sha256_file(selector_source_path)
    ):
        raise MatchedLatencyError("selector execution-source binding failed")
    expected = config["release_identity"]
    for label, identity in (
        ("selector plan", plan.get("release_contract")),
        ("selector manifest", manifest.get("release_contract")),
    ):
        if not isinstance(identity, Mapping):
            raise MatchedLatencyError(f"{label} lacks release identity")
        if (
            identity.get("expected_sqlens_build_id")
            != expected["expected_sqlens_build_id"]
            or identity.get("expected_vector_so_sha256")
            != expected["expected_vector_so_sha256"]
            or require_sha(identity.get("sha256"), f"{label} contract SHA")
            != config["release_contract_sha256"]
        ):
            raise MatchedLatencyError(f"{label} does not bind the active release")
    point_contract = plan.get("point_contract")
    min_points = (
        point_contract.get("min_distinct_pairs_per_dataset")
        if isinstance(point_contract, Mapping)
        else None
    )
    try:
        min_points_value = int(min_points) if min_points is not None else 0
    except (TypeError, ValueError):
        min_points_value = 0
    try:
        target_floor = float(plan.get("target_floor"))
    except (TypeError, ValueError):
        target_floor = -1.0
    summary = plan.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}
    targets_by_dataset = plan.get("targets_by_dataset")
    if not isinstance(targets_by_dataset, Mapping):
        targets_by_dataset = {}
    normalized_targets_by_dataset: dict[str, list[float]] = {}
    for dataset, values in targets_by_dataset.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        try:
            normalized_targets_by_dataset[str(dataset)] = sorted(
                float(value) for value in values
            )
        except (TypeError, ValueError):
            continue
    return {
        "selection_csv_sha256": csv_sha,
        "selection_plan_sha256": plan_sha,
        "selection_manifest_sha256": manifest_sha,
        "qualification_scope": qualification_scope,
        "target_policy": str(plan.get("target_policy") or ""),
        "min_distinct_pairs_per_dataset": min_points_value,
        "target_floor": target_floor,
        "targets_by_dataset": normalized_targets_by_dataset,
        "target_rows": int(summary.get("target_rows") or 0),
        "selected_pairs": int(summary.get("selected_pairs") or 0),
        "unattainable_pairs": int(summary.get("unattainable_pairs") or 0),
        "required_grid_contract": required_grid_binding,
    }


def arm_config(row: Mapping[str, str], arm: str) -> dict[str, object]:
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
    prefix = f"{arm}_"
    missing = [prefix + name for name in fields if prefix + name not in row]
    if missing:
        raise MatchedLatencyError(f"selector CSV is missing fields: {missing}")
    return {
        "config_id": require_text(row[prefix + "config_id"], prefix + "config_id"),
        "config_sha256": require_sha(row[prefix + "config_sha256"], prefix + "config_sha256"),
        "ef_search": require_int(row[prefix + "ef_search"], prefix + "ef_search", lower=1),
        "iterative_scan": require_text(row[prefix + "iterative_scan"], prefix + "iterative_scan"),
        "max_scan_tuples": require_int(row[prefix + "max_scan_tuples"], prefix + "max_scan_tuples", lower=1),
        "scan_mem_multiplier": require_float(row[prefix + "scan_mem_multiplier"], prefix + "scan_mem_multiplier"),
        "guided_collect_target": require_int(row[prefix + "guided_collect_target"], prefix + "guided_collect_target", lower=1),
        "traversal_guided_target": require_int(row[prefix + "traversal_guided_target"], prefix + "traversal_guided_target", lower=1),
        "d2_page_access": require_text(row[prefix + "d2_page_access"], prefix + "d2_page_access"),
        "d2_index_page_access": require_text(row[prefix + "d2_index_page_access"], prefix + "d2_index_page_access"),
        "table": require_text(row[prefix + "table"], prefix + "table"),
        "index": require_text(row[prefix + "index"], prefix + "index"),
    }


def load_selected_pairs(
    selection_csv: Path,
    config: Mapping[str, object],
    *,
    datasets: Sequence[str],
    pair_ids: Sequence[str],
    qualification_scope: str | None = None,
) -> list[SelectedPair]:
    with selection_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        required = {
            "pair_id",
            "qualification_scope",
            "dataset",
            "target_recall",
            "selection_status",
        }
        if not required.issubset(fields):
            raise MatchedLatencyError(
                f"selector CSV is missing fields: {sorted(required - fields)}"
            )
        rows = list(reader)
    if not rows:
        raise MatchedLatencyError("selector CSV has no rows")
    requested_datasets = set(datasets)
    requested_pairs = set(pair_ids)
    result: list[SelectedPair] = []
    seen: set[str] = set()
    for row_no, row in enumerate(rows, start=2):
        row_scope = require_qualification_scope(
            row.get("qualification_scope"), f"row {row_no}:qualification_scope"
        )
        if qualification_scope is not None and row_scope != qualification_scope:
            raise MatchedLatencyError(
                f"row {row_no}:qualification_scope {row_scope!r} does not match "
                f"selector plan {qualification_scope!r}"
            )
        status = str(row.get("selection_status") or "").strip()
        if status != "selected":
            continue
        pair_id = require_text(row.get("pair_id"), f"row {row_no}:pair_id")
        dataset = require_text(row.get("dataset"), f"row {row_no}:dataset")
        if requested_datasets and dataset not in requested_datasets:
            continue
        if requested_pairs and pair_id not in requested_pairs:
            continue
        if dataset not in config["datasets"]:
            raise MatchedLatencyError(f"row {row_no} uses unknown dataset {dataset!r}")
        if pair_id in seen:
            raise MatchedLatencyError(f"selector CSV repeats selected pair {pair_id!r}")
        seen.add(pair_id)
        target = require_float(row.get("target_recall"), f"row {row_no}:target_recall")
        if target > 1.0:
            raise MatchedLatencyError(f"row {row_no}:target_recall exceeds one")
        stock = arm_config(row, "stock")
        sqlens = arm_config(row, "sqlens")
        dataset_config = config["datasets"][dataset]
        checks = (
            (stock["table"], dataset_config["table"], "stock table"),
            (stock["index"], dataset_config["source_index"], "stock index"),
            (sqlens["table"], dataset_config["table"], "SQLens table"),
            (sqlens["index"], dataset_config["bfs_index"], "SQLens index"),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise MatchedLatencyError(
                    f"row {row_no}:{label} {actual!r} does not match frozen config {expected!r}"
                )
        if sqlens["iterative_scan"] != "off":
            raise MatchedLatencyError(
                f"row {row_no}: full SQLens traversal_guided arm must use iterative_scan=off"
            )
        quality_gate_override = str(
            row.get("quality_gate_override") or ""
        ).strip()
        if quality_gate_override not in ("", "aggregate_lcb"):
            raise MatchedLatencyError(
                f"row {row_no}: unsupported quality_gate_override "
                f"{quality_gate_override!r}"
            )
        result.append(
            SelectedPair(
                pair_id,
                dataset,
                target,
                stock,
                sqlens,
                quality_gate_override,
            )
        )
    if requested_pairs and requested_pairs - seen:
        raise MatchedLatencyError(
            f"requested pair IDs are absent or not selected: {sorted(requested_pairs - seen)}"
        )
    if not result:
        raise MatchedLatencyError("no selected pairs match the requested filters")
    return result


def pair_stem(pair: SelectedPair) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", pair.pair_id).strip(".-")
    if not SAFE_ID_RE.fullmatch(safe):
        raise MatchedLatencyError(f"unsafe pair ID for output/namespace: {pair.pair_id!r}")
    return f"figure5_r35_{pair.dataset}_matched_latency_{safe}"


def pair_namespace(pair: SelectedPair) -> str:
    digest = hashlib.sha256(pair.pair_id.encode("utf-8")).hexdigest()[:20]
    namespace = f"fig5-r35-matched-{pair.dataset}-{digest}"
    if not SAFE_ID_RE.fullmatch(namespace):
        raise MatchedLatencyError(f"unsafe D3 namespace {namespace!r}")
    return namespace


def pair_repeat_namespaces(pair: SelectedPair) -> list[str]:
    base = pair_namespace(pair)
    namespaces = [f"{base}-r{repeat}" for repeat in range(EXPECTED_REPEATS)]
    if any(not SAFE_ID_RE.fullmatch(namespace) for namespace in namespaces):
        raise MatchedLatencyError("unsafe per-repeat D3 namespace")
    return namespaces


def frozen_workload_binding(provenance: Mapping[str, object]) -> dict[int, tuple[str, str, str]]:
    """Load the immutable request identity used by every measured arm."""
    inputs = provenance.get("input_bindings")
    if not isinstance(inputs, Mapping):
        raise MatchedLatencyError("cell provenance has no input bindings")
    workload_binding = inputs.get("measurement_workload_csv")
    if not isinstance(workload_binding, Mapping):
        raise MatchedLatencyError("cell provenance has no measurement workload binding")
    path = resolve_path(workload_binding.get("path"))
    if not path.is_file():
        raise MatchedLatencyError(f"frozen workload is missing: {path}")
    expected_sha = require_sha(workload_binding.get("sha256"), "workload SHA")
    if sha256_file(path) != expected_sha:
        raise MatchedLatencyError(f"frozen workload SHA drifted: {path}")
    required = {"request_no", "query_no", "query_id", "filter_name"}
    result: dict[int, tuple[str, str, str]] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not required.issubset(set(reader.fieldnames or ())):
            raise MatchedLatencyError(
                f"frozen workload is missing fields: {sorted(required - set(reader.fieldnames or ()))}"
            )
        for row in reader:
            request_no = require_int(row.get("request_no"), "workload request_no", lower=0)
            if request_no in result:
                raise MatchedLatencyError(f"frozen workload repeats request_no={request_no}")
            result[request_no] = (
                str(row.get("query_id") or ""),
                str(row.get("query_no") or ""),
                str(row.get("filter_name") or ""),
            )
    if set(result) != set(range(EXPECTED_REQUESTS)):
        raise MatchedLatencyError("frozen workload request_no coverage is not exactly 0..9999")
    if any(not all(value for value in identity) for identity in result.values()):
        raise MatchedLatencyError("frozen workload contains an empty request identity")
    return result


def expected_schedule_position(block_no: int, schedule_seed: int, mode: str) -> int:
    """Mirror the core runner's deterministic balanced interleaving schedule."""
    order = list(MODES)
    random.Random(schedule_seed).shuffle(order)
    offset = block_no % len(order)
    order = order[offset:] + order[:offset]
    try:
        return order.index(mode) + 1
    except ValueError as exc:
        raise MatchedLatencyError(f"unknown measured mode {mode!r}") from exc


def expected_query_positions(
    repeat: int,
    schedule_seed: int,
    *,
    requests: int = EXPECTED_REQUESTS,
) -> dict[int, int]:
    """Map frozen request numbers to the seeded 1-based execution position."""
    request_order = list(range(requests))
    random.Random(schedule_seed + 104729 * repeat).shuffle(request_order)
    return {
        request_no: position
        for position, request_no in enumerate(request_order, start=1)
    }


def matched_recall_gate(
    rows: Sequence[Mapping[str, str]],
    pair: SelectedPair,
    qualification_scope: str,
) -> dict[str, object]:
    """Evaluate aggregate and per-predicate Recall@10 admission evidence.

    ``global_min_predicate_lcb`` is the only paper-admissible contract: every
    mode/repeat/filter cell must meet the target and contain enough requests
    for a stable interval.  ``aggregate_lcb`` remains available solely to
    audit legacy artifacts selected under the old aggregate-only contract.
    """
    # Import lazily: figure5_latency_repeats imports this orchestrator.
    try:
        from .figure5_latency_repeats import query_cluster_bootstrap_recall
    except ImportError:
        from figure5_latency_repeats import query_cluster_bootstrap_recall

    try:
        qualification_scope = require_qualification_scope(
            qualification_scope, "matched-latency qualification_scope"
        )
    except MatchedLatencyError as exc:
        return {"passed": False, "reason": str(exc), "arms": {}}

    expected = {("original", repeat) for repeat in range(EXPECTED_REPEATS)} | {
        ("design1_bloom_bfs_layout_d3", repeat)
        for repeat in range(EXPECTED_REPEATS)
    }
    grouped: dict[tuple[str, int], list[Mapping[str, str]]] = {
        key: [] for key in expected
    }
    for row in rows:
        try:
            key = (
                str(row.get("mode") or ""),
                require_int(row.get("repeat"), "recall repeat", lower=0),
            )
            value = float(str(row.get("recall") or ""))
        except (TypeError, ValueError, MatchedLatencyError):
            return {
                "passed": False,
                "reason": "invalid recall row",
                "aggregate": {},
                "per_predicate": {},
            }
        filter_name = str(row.get("filter_name") or "").strip()
        query_id = str(row.get("query_id") or "").strip()
        if (
            key not in grouped
            or not filter_name
            or not query_id
            or not 0.0 <= value <= 1.0
        ):
            return {
                "passed": False,
                "reason": "recall row is outside the expected domain",
                "aggregate": {},
                "per_predicate": {},
            }
        grouped[key].append(row)

    filter_names = sorted(
        {
            str(row.get("filter_name") or "").strip()
            for group in grouped.values()
            for row in group
        }
    )
    if len(filter_names) != EXPECTED_FORMAL_PREDICATES:
        return {
            "qualification_scope": qualification_scope,
            "formal_predicate_sample_floor": MIN_FORMAL_PREDICATE_SAMPLES,
            "expected_predicate_count": EXPECTED_FORMAL_PREDICATES,
            "observed_predicate_count": len(filter_names),
            "filter_names": filter_names,
            "passed": False,
            "paper_eligible": False,
            "reason": (
                "formal recall coverage does not contain exactly "
                f"{EXPECTED_FORMAL_PREDICATES} predicates"
            ),
            "aggregate": {},
            "per_predicate": {},
        }

    aggregate: dict[str, dict[str, object]] = {}
    per_predicate: dict[str, dict[str, dict[str, object]]] = {}
    worst_by_arm: dict[str, dict[str, object]] = {}
    aggregate_passed = True
    predicate_passed = True
    coverage_complete = True
    for mode in MODES:
        for repeat in range(EXPECTED_REPEATS):
            arm_key = f"{mode}/repeat={repeat}"
            group = grouped[(mode, repeat)]
            if len(group) != EXPECTED_REQUESTS:
                return {
                    "passed": False,
                    "reason": f"{arm_key} aggregate recall coverage is incomplete",
                    "aggregate": aggregate,
                    "per_predicate": per_predicate,
                }
            try:
                bootstrap = query_cluster_bootstrap_recall(
                    group,
                    value_field="recall",
                    seed_label=f"{pair.pair_id}:{arm_key}",
                )
            except Exception as exc:
                return {
                    "passed": False,
                    "reason": f"{arm_key} cluster bootstrap failed: {exc}",
                    "aggregate": aggregate,
                    "per_predicate": per_predicate,
                }
            mean = float(bootstrap["mean"])
            lower = float(bootstrap["lower"])
            upper = float(bootstrap["upper"])
            aggregate[arm_key] = {
                "sample_count": len(group),
                "query_cluster_count": bootstrap["query_cluster_count"],
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "target": pair.target_recall,
                "passed": lower >= pair.target_recall,
                "ci_method": bootstrap["method"],
                "bootstrap_samples": bootstrap["samples"],
                "bootstrap_seed": bootstrap["seed"],
                "bootstrap_seed_label": bootstrap["seed_label"],
            }
            aggregate_passed = aggregate_passed and lower >= pair.target_recall

            per_predicate[arm_key] = {}
            for filter_name in filter_names:
                bootstrap_stats = bootstrap["per_predicate"].get(filter_name)
                if not isinstance(bootstrap_stats, Mapping):
                    bootstrap_stats = {}
                sample_count = int(bootstrap_stats.get("sample_count") or 0)
                sufficient = sample_count >= MIN_FORMAL_PREDICATE_SAMPLES
                filter_mean = bootstrap_stats.get("mean")
                filter_lower = bootstrap_stats.get("lower")
                filter_upper = bootstrap_stats.get("upper")
                filter_passed = bool(
                    sufficient
                    and filter_lower is not None
                    and float(filter_lower) >= pair.target_recall
                )
                stats: dict[str, object] = {
                    "sample_count": sample_count,
                    "query_cluster_count": int(
                        bootstrap_stats.get("query_cluster_count") or 0
                    ),
                    "sample_count_sufficient": sufficient,
                    "mean": filter_mean,
                    "lower": filter_lower,
                    "upper": filter_upper,
                    "target": pair.target_recall,
                    "passed": filter_passed,
                    "ci_method": bootstrap["method"],
                }
                per_predicate[arm_key][filter_name] = stats
                coverage_complete = coverage_complete and sample_count > 0
                predicate_passed = predicate_passed and filter_passed

            worst_by_arm[arm_key] = min(
                (
                    {
                        "filter_name": filter_name,
                        **stats,
                    }
                    for filter_name, stats in per_predicate[arm_key].items()
                ),
                key=lambda item: (
                    float(item["lower"])
                    if item["lower"] is not None
                    else float("-inf"),
                    float(item["mean"])
                    if item["mean"] is not None
                    else float("-inf"),
                    str(item["filter_name"]),
                ),
            )

    worst_candidates = [
        {"mode_repeat": arm_key, **stats}
        for arm_key, stats in worst_by_arm.items()
    ]
    worst_predicate = min(
        worst_candidates,
        key=lambda item: (
            float(item["lower"])
            if item["lower"] is not None
            else float("-inf"),
            float(item["mean"])
            if item["mean"] is not None
            else float("-inf"),
            str(item["mode_repeat"]),
            str(item["filter_name"]),
        ),
    )
    if pair.quality_gate_override == "aggregate_lcb":
        passed = aggregate_passed
        reason = (
            "explicit aggregate-LCB target override passes"
            if passed
            else "aggregate recall LCB misses overridden target"
        )
    elif qualification_scope == QUALIFICATION_SCOPE_FORMAL:
        passed = aggregate_passed and coverage_complete and predicate_passed
        if not coverage_complete:
            reason = "per-predicate recall coverage is incomplete"
        elif not predicate_passed:
            reason = "per-predicate recall LCB or sample-count gate misses target"
        elif not aggregate_passed:
            reason = "aggregate recall LCB misses target"
        else:
            reason = "ok"
    else:
        passed = aggregate_passed
        reason = "legacy aggregate audit passes" if passed else "aggregate recall LCB misses target"
    return {
        "qualification_scope": qualification_scope,
        "quality_gate_override": pair.quality_gate_override,
        "formal_predicate_sample_floor": MIN_FORMAL_PREDICATE_SAMPLES,
        "expected_predicate_count": EXPECTED_FORMAL_PREDICATES,
        "observed_predicate_count": len(filter_names),
        "filter_names": filter_names,
        "recall_ci_method": (
            "query_id_cluster_stratified_predicate_percentile_bootstrap_95"
        ),
        "passed": passed,
        "paper_eligible": (
            qualification_scope == QUALIFICATION_SCOPE_FORMAL
            and not pair.quality_gate_override
            and passed
        ),
        "reason": reason,
        "aggregate": aggregate,
        "per_predicate": per_predicate,
        "worst_predicate_by_arm": worst_by_arm,
        "worst_predicate": worst_predicate,
    }


def quality_gate_matches_plan(
    plan: Mapping[str, object], quality_gate: Mapping[str, object]
) -> bool:
    """Require the core plan to carry the runner's raw-derived quality proof."""
    recorded = plan.get("matched_latency_quality_gate")
    recorded_sha = plan.get("matched_latency_quality_gate_sha256")
    predicate_completion = plan.get("matched_latency_predicate_completion")
    if not isinstance(recorded, Mapping) or not isinstance(recorded_sha, str):
        return False
    return (
        recorded.get("qualification_scope")
        == quality_gate.get("qualification_scope")
        and recorded_sha == sha256_json(recorded)
        and recorded_sha == sha256_json(quality_gate)
        and predicate_completion
        == predicate_completion_contract(quality_gate)
    )


def predicate_completion_contract(
    quality_gate: Mapping[str, object],
) -> dict[str, object]:
    names = quality_gate.get("filter_names")
    if not isinstance(names, list):
        names = []
    return {
        "expected_predicate_count": EXPECTED_FORMAL_PREDICATES,
        "observed_predicate_count": len(names),
        "predicate_names": list(names),
        "exact_coverage": (
            len(names) == EXPECTED_FORMAL_PREDICATES
            and len(set(str(name) for name in names))
            == EXPECTED_FORMAL_PREDICATES
        ),
    }


def write_quality_gate_to_plan(plan_path: Path, quality_gate: Mapping[str, object]) -> None:
    """Attach final raw-derived quality evidence without altering core output."""
    plan = read_json(plan_path, "cell plan")
    plan["matched_latency_quality_gate"] = dict(quality_gate)
    plan["matched_latency_quality_gate_sha256"] = sha256_json(quality_gate)
    plan["matched_latency_qualification_scope"] = quality_gate[
        "qualification_scope"
    ]
    plan["matched_latency_predicate_completion"] = (
        predicate_completion_contract(quality_gate)
    )
    atomic_json(plan_path, plan)


def full_release_scope(
    args: argparse.Namespace,
    pairs: Sequence[SelectedPair],
    all_pairs: Sequence[SelectedPair],
    *,
    selection_bindings: Mapping[str, object] | None = None,
    enforce_frozen_selector: bool = False,
) -> dict[str, object]:
    checks = {
        "all_datasets_requested": (
            not args.datasets or set(args.datasets) == set(FROZEN_DATASETS)
        ),
        "all_selected_pairs_requested": (
            not args.pair_ids
            and {pair.pair_id for pair in pairs}
            == {pair.pair_id for pair in all_pairs}
        ),
        "default_backend_cpu_partition": (
            args.backend_cpu_list == DEFAULT_BACKEND_CPU_LIST
        ),
    }
    if enforce_frozen_selector:
        bindings = selection_bindings or {}
        target_policy = bindings.get("target_policy")
        qualification_scope = bindings.get("qualification_scope")
        targets_by_dataset = bindings.get("targets_by_dataset")
        if not isinstance(targets_by_dataset, Mapping):
            targets_by_dataset = {}
        selector_datasets = set(targets_by_dataset)
        checks.update({
            "selector_uses_formal_predicate_qualification": (
                qualification_scope == QUALIFICATION_SCOPE_FORMAL
            ),
        })
        if target_policy == "fixed":
            expected_targets = [0.90, 0.95, 0.99]
            target_rows = int(bindings.get("target_rows") or 0)
            selected_pairs = int(bindings.get("selected_pairs") or 0)
            unattainable_pairs = int(bindings.get("unattainable_pairs") or 0)
            checks.update({
                "selector_covers_frozen_datasets": (
                    selector_datasets == set(FROZEN_DATASETS)
                ),
                "selector_uses_fixed_target_policy": True,
                "selector_uses_formal_fixed_targets": all(
                    list(targets_by_dataset.get(dataset, ()))
                    == expected_targets
                    for dataset in FROZEN_DATASETS
                ),
                "selector_resolves_every_fixed_target": (
                    target_rows == len(FROZEN_DATASETS) * len(expected_targets)
                    and selected_pairs + unattainable_pairs == target_rows
                    and selected_pairs == len(all_pairs)
                ),
            })
        else:
            datasets = {pair.dataset for pair in all_pairs}
            per_dataset = {
                dataset: [pair for pair in all_pairs if pair.dataset == dataset]
                for dataset in FROZEN_DATASETS
            }
            checks.update({
                "selector_covers_frozen_datasets": datasets == set(FROZEN_DATASETS),
                "selector_uses_distinct_pair_policy": (
                    target_policy == "distinct_pairs"
                ),
                "selector_declares_minimum_point_gate": (
                    int(bindings.get("min_distinct_pairs_per_dataset") or 0)
                    >= MIN_FORMAL_POINTS_PER_ARM_DATASET
                ),
                "selector_uses_formal_target_floor": (
                    float(bindings.get("target_floor") or -1.0) >= 0.70
                ),
                "selector_has_minimum_distinct_stock_points": all(
                    len({pair.stock.get("config_sha256") for pair in dataset_pairs})
                    >= MIN_FORMAL_POINTS_PER_ARM_DATASET
                    for dataset_pairs in per_dataset.values()
                ),
                "selector_has_minimum_distinct_sqlens_points": all(
                    len({pair.sqlens.get("config_sha256") for pair in dataset_pairs})
                    >= MIN_FORMAL_POINTS_PER_ARM_DATASET
                    for dataset_pairs in per_dataset.values()
                ),
            })
    return {
        "requested": all(checks.values()),
        "kind": (
            "matched_targets"
            if (selection_bindings or {}).get("target_policy") == "fixed"
            else "frontier"
        ),
        "checks": checks,
        "required_pairs": sorted(pair.pair_id for pair in all_pairs),
        "requested_pairs": sorted(pair.pair_id for pair in pairs),
    }


def input_bindings(dataset: Mapping[str, object]) -> dict[str, dict[str, object]]:
    paths = {
        "truth_csv": resolve_path(dataset["truth_csv"]),
        "measurement_workload_csv": resolve_path(dataset["measurement_workload_csv"]),
        "filters_csv": resolve_path(dataset["filters_csv"]),
        "d2_graph_proof_json": resolve_path(dataset["d2_graph_proof_json"]),
    }
    missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise MatchedLatencyError("matched-latency inputs are missing: " + ", ".join(missing))
    bindings = {
        name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    bindings["d2_graph_proof_json"]["canonical_json_sha256"] = (
        frontier.sha256_json(
            read_json(paths["d2_graph_proof_json"], "D2 graph proof")
        )
    )
    return bindings


def build_pair_command(
    config: Mapping[str, object],
    pair: SelectedPair,
    raw: Path,
    backend_cpu_list: str,
) -> tuple[list[str], dict[str, Any]]:
    dataset = config["datasets"][pair.dataset]
    protocol = config["protocol"]
    inputs = input_bindings(dataset)
    workload = Path(str(inputs["measurement_workload_csv"]["path"]))
    if frontier.count_csv_rows(workload) != EXPECTED_REQUESTS:
        raise MatchedLatencyError(f"measurement workload must contain {EXPECTED_REQUESTS} rows: {workload}")
    mode_configs = {
        "original": {
            **{key: pair.stock[key] for key in (
                "ef_search", "iterative_scan", "max_scan_tuples", "scan_mem_multiplier",
                "guided_collect_target", "traversal_guided_target",
            )},
            "traversal_guided_burst": 8,
            "traversal_guided_prioritization": False,
        },
        "design1_bloom_bfs_layout_d3": {
            **{key: pair.sqlens[key] for key in (
                "ef_search", "iterative_scan", "max_scan_tuples", "scan_mem_multiplier",
                "guided_collect_target", "traversal_guided_target",
            )},
            "traversal_guided_burst": 8,
            "traversal_guided_prioritization": True,
        },
    }
    release = config["release_identity"]
    relations = [str(dataset["table"]), str(dataset["source_index"]), str(dataset["bfs_index"])]
    namespace = pair_namespace(pair)
    repeat_namespaces = pair_repeat_namespaces(pair)
    command = [
        sys.executable, str(frontier.CORE_RUNNER),
        "--insertion-table", str(dataset["table"]),
        "--insertion-index", str(dataset["source_index"]),
        "--bfs-table", str(dataset["table"]),
        "--bfs-index", str(dataset["bfs_index"]),
        "--query-table", str(dataset["query_table"]),
        "--query-id-column", str(dataset["query_id_column"]),
        "--query-vector-column", str(dataset["query_vector_column"]),
        "--candidate-validity-predicate", str(dataset["candidate_validity_predicate"]),
        "--expected-truth-self-excluded" if bool(dataset["truth_self_excluded"]) else "--no-expected-truth-self-excluded",
        "--truth-csv", str(inputs["truth_csv"]["path"]),
        "--workload-csv", str(workload),
        "--expected-workload-requests", str(EXPECTED_REQUESTS),
        "--require-unique-workload-queries",
        "--filters-csv", str(inputs["filters_csv"]["path"]),
        *[item for relation in relations for item in ("--prewarm-relation", relation)],
        "--modes", *MODES,
        "--execution-order", "interleaved",
        "--schedule-seed", str(int(protocol["schedule_seed"])),
        "--mode-configs-json", json.dumps(mode_configs, separators=(",", ":"), sort_keys=True),
        "--repeats", str(EXPECTED_REPEATS),
        "--isolate-repeat-runtimes",
        "--warmup-queries", "1", "--no-warmup-all-queries",
        "--ef-search", str(pair.stock["ef_search"]),
        "--guided-collect-target", str(pair.stock["guided_collect_target"]),
        "--traversal-guided-target", str(pair.stock["traversal_guided_target"]),
        "--no-traversal-guided-prioritization",
        "--guidance-filter-strategy", str(protocol["guidance_filter_strategy"]),
        "--iterative-scan", str(pair.stock["iterative_scan"]),
        "--max-scan-tuples", str(pair.stock["max_scan_tuples"]),
        "--scan-mem-multiplier", str(pair.stock["scan_mem_multiplier"]),
        "--d2-page-access", str(pair.stock["d2_page_access"]),
        "--d2-index-page-access", str(pair.stock["d2_index_page_access"]),
        "--d1-guidance-kind", "auto",
        "--d3-measurement-policy", str(protocol["d3_measurement_policy"]),
        "--d3-fragment-store-namespace", namespace,
        "--guidance-selectivity-max-pct", "100",
        "--guidance-max-atoms", str(int(protocol["guidance_max_atoms"])),
        "--statement-timeout-ms", "7200000", "--force-hnsw", "--require-preferred-index-guc",
        "--d2-graph-proof-json", str(inputs["d2_graph_proof_json"]["path"]),
        "--expected-sqlens-build-id", str(release["expected_sqlens_build_id"]),
        "--expected-vector-so-sha256", str(release["expected_vector_so_sha256"]),
        "--backend-cpu-list", backend_cpu_list,
        "--progress-queries", "250", "--out", str(raw),
        "--orchestrator-source", str(Path(__file__).resolve()),
    ]
    provenance = {
        "pair_id": pair.pair_id,
        "dataset": pair.dataset,
        "target_recall": pair.target_recall,
        "expected_rows": EXPECTED_ROWS,
        "expected_requests": EXPECTED_REQUESTS,
        "expected_repeats": EXPECTED_REPEATS,
        "modes": list(MODES),
        "schedule_seed": int(protocol["schedule_seed"]),
        "stock_config": pair.stock,
        "sqlens_config": pair.sqlens,
        "mode_configs": mode_configs,
        "d3_fragment_store_namespace": namespace,
        "d3_repeat_namespaces": repeat_namespaces,
        "d3_fragment_store_table": str(dataset["table"]),
        "input_bindings": inputs,
        "execution_sources": {
            "core_runner": {
                "path": str(frontier.CORE_RUNNER.resolve()),
                "sha256": sha256_file(frontier.CORE_RUNNER.resolve()),
            },
            "orchestrator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "relation_prewarm": {"method": "pg_prewarm(regclass,'read','main')", "relations": relations},
    }
    return command, provenance


def namespace_rows(table: str, namespace: str) -> int:
    prefix = namespace + "\x1f"
    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM public.pgvector_hnsw_fragment_store "
                "WHERE heap_oid = %s::regclass "
                "AND left(filter_name, char_length(%s)) = %s",
                (table, prefix, prefix),
            )
            row = cur.fetchone()
    return int(row[0]) if row else -1


def ensure_fresh_namespace(table: str, namespace: str, *, overwrite: bool) -> int:
    current = namespace_rows(table, namespace)
    if current < 0:
        raise MatchedLatencyError(f"could not inspect D3 namespace {namespace!r}")
    if current == 0:
        return 0
    if not overwrite:
        raise MatchedLatencyError(
            f"D3 namespace {namespace!r} is not fresh ({current} rows); use --overwrite"
        )
    try:
        deleted = frontier.clear_fragment_store_namespace(table, namespace)
    except frontier.Figure5ContractError as exc:
        raise MatchedLatencyError(str(exc)) from exc
    if namespace_rows(table, namespace) != 0:
        raise MatchedLatencyError(f"D3 namespace remained nonempty after clear: {namespace!r}")
    return deleted


def identity_matches(value: object, release: Mapping[str, object]) -> bool:
    return isinstance(value, Mapping) and (
        value.get("exact_match") is True
        and value.get("expected_build_id") == release["expected_sqlens_build_id"]
        and value.get("observed_build_id") == release["expected_sqlens_build_id"]
        and value.get("expected_vector_so_sha256") == release["expected_vector_so_sha256"]
        and value.get("observed_vector_so_sha256") == release["expected_vector_so_sha256"]
    )


def execution_sources_compatible(
    observed: object, expected: object
) -> bool:
    if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
        return False
    if observed.get("core_runner") != expected.get("core_runner"):
        return False
    observed_orchestrator = observed.get("orchestrator")
    expected_orchestrator = expected.get("orchestrator")
    if not isinstance(observed_orchestrator, Mapping) or not isinstance(
        expected_orchestrator, Mapping
    ):
        return False
    if observed_orchestrator.get("path") != expected_orchestrator.get("path"):
        return False
    observed_sha = str(observed_orchestrator.get("sha256") or "")
    expected_sha = str(expected_orchestrator.get("sha256") or "")
    return observed_sha == expected_sha or (
        observed_sha in VALIDATOR_ONLY_COMPATIBLE_ORCHESTRATOR_SHA256
    )


def complete_prewarm(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("enabled") is not True or value.get("complete") is not True:
        return False
    records = value.get("records")
    if not isinstance(records, list) or len(records) != 3:
        return False
    try:
        return all(
            isinstance(record, Mapping)
            and require_int(record.get("expected_blocks"), "prewarm expected", lower=1)
            == require_int(record.get("warmed_blocks"), "prewarm warmed", lower=1)
            for record in records
        )
    except MatchedLatencyError:
        return False


def _legacy_raw_rows_match_pair(
    rows: list[dict[str, str]], pair: SelectedPair
) -> bool:
    """Read old r35 cells so existing artifact-converter tests remain usable."""
    if len(rows) != EXPECTED_ROWS or any(str(row.get("error") or "").strip() for row in rows):
        return False
    by_mode: dict[str, list[dict[str, str]]] = {mode: [] for mode in MODES}
    for row in rows:
        if row.get("mode") not in by_mode:
            return False
        by_mode[row["mode"]].append(row)
    expected_configs = {"original": pair.stock, "design1_bloom_bfs_layout_d3": pair.sqlens}
    signatures: dict[tuple[str, int], set[tuple[str, str, str]]] = {}
    for mode, expected in expected_configs.items():
        if len(by_mode[mode]) != EXPECTED_REQUESTS * EXPECTED_REPEATS:
            return False
        for repeat in range(EXPECTED_REPEATS):
            subset = [row for row in by_mode[mode] if str(row.get("repeat")) == str(repeat)]
            if len(subset) != EXPECTED_REQUESTS:
                return False
            for field in (
                "ef_search", "iterative_scan", "max_scan_tuples", "guided_collect_target",
                "scan_mem_multiplier", "traversal_guided_target",
                "d2_page_access", "d2_index_page_access",
            ):
                if any(str(row.get(field)) != str(expected[field]) for row in subset):
                    return False
            signatures[(mode, repeat)] = {
                (str(row.get("query_id")), str(row.get("query_no", "")), str(row.get("filter_name")))
                for row in subset
            }
        repeat_rows = [
            row for mode_rows in by_mode.values() for row in mode_rows
            if str(row.get("repeat")) == str(repeat)
        ]
        by_request: dict[str, list[dict[str, str]]] = {}
        for row in repeat_rows:
            by_request.setdefault(str(row.get("request_no")), []).append(row)
        if len(by_request) != EXPECTED_REQUESTS:
            return False
        for request_rows in by_request.values():
            if len(request_rows) != len(MODES):
                return False
            if len({str(row.get("block_no")) for row in request_rows}) != 1:
                return False
            if {str(row.get("schedule_position")) for row in request_rows} != {"1", "2"}:
                return False
            if len({str(row.get("query_order_position")) for row in request_rows}) != 1:
                return False
            if len({(str(row.get("query_id")), str(row.get("filter_name"))) for row in request_rows}) != 1:
                return False
            if any(row.get("execution_order") != "interleaved" for row in request_rows):
                return False
    return all(
        signatures[("original", repeat)] == signatures[("design1_bloom_bfs_layout_d3", repeat)]
        for repeat in range(EXPECTED_REPEATS)
    )


def raw_rows_match_pair(
    rows: list[dict[str, str]],
    pair: SelectedPair,
    provenance: Mapping[str, object],
) -> bool:
    if "schedule_seed" not in provenance:
        return _legacy_raw_rows_match_pair(rows, pair)
    if len(rows) != EXPECTED_ROWS or any(str(row.get("error") or "").strip() for row in rows):
        return False
    workload = frozen_workload_binding(provenance)
    by_mode: dict[str, list[dict[str, str]]] = {mode: [] for mode in MODES}
    for row in rows:
        mode = row.get("mode")
        if mode not in by_mode:
            return False
        by_mode[mode].append(row)
    expected_configs = {"original": pair.stock, "design1_bloom_bfs_layout_d3": pair.sqlens}
    signatures: dict[tuple[str, int], dict[int, tuple[str, str, str]]] = {}
    schedule_seed = require_int(
        provenance.get("schedule_seed"), "schedule seed", lower=0
    )
    query_positions_by_repeat = {
        repeat: expected_query_positions(
            repeat, schedule_seed, requests=EXPECTED_REQUESTS
        )
        for repeat in range(EXPECTED_REPEATS)
    }
    for mode, expected in expected_configs.items():
        if len(by_mode[mode]) != EXPECTED_REQUESTS * EXPECTED_REPEATS:
            return False
        for repeat in range(EXPECTED_REPEATS):
            subset = [row for row in by_mode[mode] if str(row.get("repeat")) == str(repeat)]
            if len(subset) != EXPECTED_REQUESTS:
                return False
            request_nos = {
                require_int(row.get("request_no"), "raw request_no", lower=0)
                for row in subset
            }
            if request_nos != set(range(EXPECTED_REQUESTS)):
                return False
            for field in (
                "ef_search", "iterative_scan", "max_scan_tuples", "guided_collect_target",
                "scan_mem_multiplier", "traversal_guided_target",
                "d2_page_access", "d2_index_page_access",
            ):
                if any(str(row.get(field)) != str(expected[field]) for row in subset):
                    return False
            expected_namespace = pair_repeat_namespaces(pair)[repeat]
            if any(
                str(row.get("d3_fragment_store_namespace"))
                != expected_namespace
                for row in subset
            ):
                return False
            signature: dict[int, tuple[str, str, str]] = {}
            query_positions: set[int] = set()
            blocks: set[int] = set()
            for row in subset:
                request_no = require_int(row.get("request_no"), "raw request_no", lower=0)
                identity = (
                    str(row.get("query_id") or ""),
                    str(row.get("query_no") or ""),
                    str(row.get("filter_name") or ""),
                )
                if identity != workload[request_no] or request_no in signature:
                    return False
                signature[request_no] = identity
                if row.get("execution_order") != "interleaved":
                    return False
                if str(row.get("schedule_seed")) != str(schedule_seed):
                    return False
                query_position = require_int(
                    row.get("query_order_position"), "query order position", lower=1
                )
                if query_position != query_positions_by_repeat[repeat][request_no]:
                    return False
                block_no = require_int(row.get("block_no"), "block number", lower=0)
                if query_position not in range(1, EXPECTED_REQUESTS + 1):
                    return False
                if block_no != repeat * EXPECTED_REQUESTS + query_position - 1:
                    return False
                if expected_schedule_position(block_no, schedule_seed, mode) != require_int(
                    row.get("schedule_position"), "schedule position", lower=1
                ):
                    return False
                query_positions.add(query_position)
                blocks.add(block_no)
            if query_positions != set(range(1, EXPECTED_REQUESTS + 1)):
                return False
            if blocks != set(
                range(repeat * EXPECTED_REQUESTS, (repeat + 1) * EXPECTED_REQUESTS)
            ):
                return False
            signatures[(mode, repeat)] = signature

        # The core plan is the authoritative effective configuration.  Raw rows
        # additionally expose the effective prioritization burst for each query.
        mode_configs = provenance.get("mode_configs")
        expected_mode_config = (
            mode_configs.get(mode) if isinstance(mode_configs, Mapping) else None
        )
        if not isinstance(expected_mode_config, Mapping):
            return False
        expected_burst = (
            require_int(expected_mode_config.get("traversal_guided_burst"), "guided burst", lower=0)
            if bool(expected_mode_config.get("traversal_guided_prioritization"))
            else 0
        )
        if any(
            require_int(row.get("traversal_prioritization_burst"), "raw guided burst", lower=0)
            != expected_burst
            for row in by_mode[mode]
        ):
            return False
    rows_by_repeat_request: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        repeat = require_int(row.get("repeat"), "raw repeat", lower=0)
        request_no = require_int(row.get("request_no"), "raw request_no", lower=0)
        rows_by_repeat_request.setdefault((repeat, request_no), []).append(row)

    for repeat in range(EXPECTED_REPEATS):
        if signatures[("original", repeat)] != signatures[("design1_bloom_bfs_layout_d3", repeat)]:
            return False
        for request_no in range(EXPECTED_REQUESTS):
            request_rows = rows_by_repeat_request.get((repeat, request_no), [])
            if len(request_rows) != len(MODES):
                return False
            if {require_int(row.get("schedule_position"), "schedule position", lower=1) for row in request_rows} != {1, 2}:
                return False
            if len({require_int(row.get("block_no"), "block number", lower=0) for row in request_rows}) != 1:
                return False
            if len({require_int(row.get("query_order_position"), "query order position", lower=1) for row in request_rows}) != 1:
                return False
    return True


def effective_mode_config_complete(
    plan: Mapping[str, object], provenance: Mapping[str, object]
) -> bool:
    expected = provenance.get("mode_configs")
    checks = plan.get("checks")
    if not isinstance(expected, Mapping) or not isinstance(checks, list):
        return False
    observed_modes: set[str] = set()
    for check in checks:
        if not isinstance(check, Mapping):
            return False
        mode = str(check.get("mode") or "")
        config = check.get("config")
        if mode not in MODES or not isinstance(config, Mapping):
            return False
        if dict(config) != dict(expected.get(mode) or {}):
            return False
        observed_modes.add(mode)
    return observed_modes == set(MODES)


def cell_complete(
    raw: Path,
    plan_path: Path,
    pair: SelectedPair,
    config: Mapping[str, object],
    provenance: Mapping[str, object],
) -> bool:
    if not raw.is_file() or not plan_path.is_file():
        return False
    try:
        plan = read_json(plan_path, "cell plan")
        rows = read_raw_rows(raw)
        runtime = plan.get("runtime_sqlens_identity_evidence")
        start = plan.get("d3_fragment_store_start")
        query_contract = plan.get("query_contract")
        errors = plan.get("query_error_summary")
        execution_sources = plan.get("execution_sources")
        lifecycle = plan.get("execution_lifecycle")
        bindings = provenance["input_bindings"]
        strict_protocol = "schedule_seed" in provenance
        qualification_scope = require_qualification_scope(
            provenance.get("qualification_scope"), "cell qualification_scope"
        )
        quality_gate = matched_recall_gate(rows, pair, qualification_scope)
        start_records = (
            start.get("records") if isinstance(start, Mapping) else None
        )
        return (
            plan.get("status") == "complete"
            and plan.get("output_sha256") == sha256_file(raw)
            and require_int(plan.get("output_rows"), "output rows", lower=1) == EXPECTED_ROWS
            and raw_rows_match_pair(rows, pair, provenance)
            and (
                not strict_protocol
                or (
                    effective_mode_config_complete(plan, provenance)
                    and bool(quality_gate["passed"])
                    and quality_gate_matches_plan(plan, quality_gate)
                )
            )
            and isinstance(errors, Mapping) and errors.get("error_rows") == 0
            and complete_prewarm(plan.get("relation_prewarm"))
            and identity_matches(plan.get("sqlens_runtime_identity_startup"), config["release_identity"])
            and identity_matches(plan.get("sqlens_runtime_identity_final"), config["release_identity"])
            and isinstance(runtime, list) and bool(runtime)
            and all(identity_matches(item, config["release_identity"]) for item in runtime)
            and isinstance(start, Mapping)
            and start.get("isolated_repeats") is True
            and start.get("base_namespace")
            == provenance["d3_fragment_store_namespace"]
            and isinstance(start_records, list)
            and len(start_records) == EXPECTED_REPEATS
            and {
                record.get("namespace")
                for record in start_records
                if isinstance(record, Mapping)
            }
            == set(provenance["d3_repeat_namespaces"])
            and all(
                isinstance(record, Mapping)
                and record.get("empty") is True
                and require_int(
                    record.get("rows_before"),
                    "namespace rows",
                    lower=0,
                )
                == 0
                for record in start_records
            )
            and isinstance(query_contract, Mapping)
            and query_contract.get("workload_sha256") == bindings["measurement_workload_csv"]["sha256"]
            and query_contract.get("truth_sha256")
            == bindings["truth_csv"]["sha256"]
            and query_contract.get("filters_sha256")
            == bindings["filters_csv"]["sha256"]
            and query_contract.get("d2_graph_proof_input_sha256")
            == bindings["d2_graph_proof_json"]["canonical_json_sha256"]
            and require_int(query_contract.get("expected_workload_requests"), "workload requests", lower=1) == EXPECTED_REQUESTS
            and require_int(query_contract.get("workload_unique_queries"), "workload unique queries", lower=1) == EXPECTED_REQUESTS
            and query_contract.get("require_unique_workload_queries") is True
            and execution_sources_compatible(
                execution_sources, provenance["execution_sources"]
            )
            and isinstance(lifecycle, Mapping)
            and lifecycle.get("repeat_runtime_isolation") is True
            and require_int(
                lifecycle.get("runtime_openings"),
                "runtime openings",
                lower=1,
            )
            == EXPECTED_REPEATS * len(MODES)
            and isinstance(plan.get("d2_graph_proof"), Mapping)
            and isinstance(plan.get("d2_graph_proof_final"), Mapping)
        )
    except (MatchedLatencyError, OSError, ValueError, csv.Error, KeyError):
        return False


def acquire_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise MatchedLatencyError(f"another matched-latency runner owns {path}") from exc
    return handle


def protocol_fingerprint(
    *,
    config_path: Path,
    selection_bindings: Mapping[str, object],
    pairs: Sequence[SelectedPair],
    all_pairs: Sequence[SelectedPair],
    release_scope: Mapping[str, object],
    backend_cpu_list: str,
) -> str:
    return sha256_json(
        {
            "runner_version": RUNNER_VERSION,
            "orchestrator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "selector": dict(selection_bindings),
            "requested_pairs": [
                {
                    "pair_id": pair.pair_id,
                    "dataset": pair.dataset,
                    "target_recall": pair.target_recall,
                    "stock": pair.stock,
                    "sqlens": pair.sqlens,
                }
                for pair in pairs
            ],
            "full_selected_pair_ids": [pair.pair_id for pair in all_pairs],
            "release_scope": dict(release_scope),
            "backend_cpu_list": backend_cpu_list,
            "execution": {
                "requests": EXPECTED_REQUESTS,
                "repeats": EXPECTED_REPEATS,
                "modes": list(MODES),
                "order": "paired_interleaved",
            },
        }
    )


def validate_existing_run_manifest(
    path: Path,
    fingerprint: str,
    release_scope: Mapping[str, object],
    *,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    existing = read_json(path, "matched-latency run manifest")
    existing_is_full = bool(
        existing.get("full_release_complete") is True
        or existing.get("paper_eligible") is True
        or (
            isinstance(existing.get("full_release_scope"), Mapping)
            and existing["full_release_scope"].get("requested") is True
        )
    )
    if not release_scope.get("requested") and existing_is_full:
        raise MatchedLatencyError(
            "a subset run cannot overwrite an existing full-release manifest; "
            "use a new --out-dir"
        )
    compatible = (
        existing.get("artifact_type") == "sqlens_figure5_matched_latency_run"
        and existing.get("runner_version") == RUNNER_VERSION
        and existing.get("protocol_fingerprint_sha256") == fingerprint
    )
    if not compatible and not overwrite:
        raise MatchedLatencyError(
            f"existing run manifest is incompatible: {path}; use --overwrite or a new --out-dir"
        )
    if compatible and not resume and not overwrite:
        raise MatchedLatencyError(
            f"run manifest already exists: {path}; use --resume or --overwrite"
        )
    return existing if compatible and not overwrite else None


def run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.required_grid_contract is None:
        raise MatchedLatencyError(
            "--required-grid-contract is mandatory for final matched latency"
        )
    required_grid_contract = args.required_grid_contract.resolve()
    selection_csv = args.selection_csv.resolve()
    inferred_plan, inferred_manifest = inferred_selection_paths(selection_csv)
    selection_plan = (args.selection_plan or inferred_plan).resolve()
    selection_manifest = (args.selection_manifest or inferred_manifest).resolve()
    selection_bindings = validate_selection_artifacts(
        selection_csv,
        selection_plan,
        selection_manifest,
        config,
        config_path=config_path,
        required_grid_contract=required_grid_contract,
    )
    pairs = load_selected_pairs(
        selection_csv,
        config,
        datasets=args.datasets,
        pair_ids=args.pair_ids,
        qualification_scope=str(selection_bindings["qualification_scope"]),
    )
    all_pairs = load_selected_pairs(
        selection_csv,
        config,
        datasets=(),
        pair_ids=(),
        qualification_scope=str(selection_bindings["qualification_scope"]),
    )
    release_scope = full_release_scope(
        args,
        pairs,
        all_pairs,
        selection_bindings=selection_bindings,
        enforce_frozen_selector=True,
    )
    out_dir = args.out_dir.resolve()
    manifest_path = out_dir / "figure5_r35_matched_latency_run_manifest.json"
    lock = acquire_lock(manifest_path.with_suffix(".lock"))
    try:
        schedule: list[dict[str, Any]] = []
        for pair in pairs:
            raw = out_dir / f"{pair_stem(pair)}.csv"
            command, provenance = build_pair_command(config, pair, raw, args.backend_cpu_list)
            provenance["qualification_scope"] = selection_bindings[
                "qualification_scope"
            ]
            complete = cell_complete(raw, raw.with_suffix(raw.suffix + ".plan.json"), pair, config, provenance)
            cell = {
                **provenance,
                "raw": str(raw),
                "plan": str(raw.with_suffix(raw.suffix + ".plan.json")),
                "command": command,
                "status": "complete" if complete else "pending",
            }
            if complete:
                existing_plan = read_json(
                    raw.with_suffix(raw.suffix + ".plan.json"), "cell plan"
                )
                cell["observed_execution_sources"] = existing_plan.get(
                    "execution_sources"
                )
                cell["quality_gate"] = existing_plan.get(
                    "matched_latency_quality_gate"
                )
                cell["predicate_completion"] = predicate_completion_contract(
                    cell["quality_gate"]
                    if isinstance(cell["quality_gate"], Mapping)
                    else {}
                )
            schedule.append(cell)
        fingerprint = protocol_fingerprint(
            config_path=config_path,
            selection_bindings=selection_bindings,
            pairs=pairs,
            all_pairs=all_pairs,
            release_scope=release_scope,
            backend_cpu_list=args.backend_cpu_list,
        )
        validate_existing_run_manifest(
            manifest_path,
            fingerprint,
            release_scope,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "sqlens_figure5_matched_latency_run",
            "runner_version": RUNNER_VERSION,
            "protocol_fingerprint_sha256": fingerprint,
            "status": "planned",
            "artifact_valid": False,
            "paper_eligible": False,
            "requested_slice_complete": False,
            "full_release_complete": False,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "full_release_scope": release_scope,
            "execution": {
                "parallel_db_cells": False,
                "execution_order": "paired_interleaved",
                "requests": EXPECTED_REQUESTS,
                "repeats": EXPECTED_REPEATS,
                "expected_rows_per_pair": EXPECTED_ROWS,
                "expected_predicate_count": EXPECTED_FORMAL_PREDICATES,
                "recall_ci_method": (
                    "query_id_cluster_stratified_predicate_"
                    "percentile_bootstrap_95"
                ),
            },
            "frontier_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "required_grid_contract": dict(
                selection_bindings["required_grid_contract"]
            ),
            "release_contract": {"path": config["release_contract_path"], "sha256": config["release_contract_sha256"], **config["release_identity"]},
            "validator_compatibility": {
                "current_orchestrator_sha256": sha256_file(Path(__file__).resolve()),
                "accepted_validator_only_orchestrator_sha256": sorted(
                    VALIDATOR_ONLY_COMPATIBLE_ORCHESTRATOR_SHA256
                ),
            },
            "selector": {"csv": str(selection_csv), "plan": str(selection_plan), "manifest": str(selection_manifest), **selection_bindings},
            "schedule": schedule,
            "pairs_total": len(schedule),
            "pairs_complete": sum(cell["status"] == "complete" for cell in schedule),
            "paper_eligible": False,
        }
        atomic_json(manifest_path, manifest)
        if not args.execute:
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        manifest["status"] = "running"
        for cell in schedule:
            if cell["status"] == "complete" and args.resume:
                print(f"resume: complete pair={cell['pair_id']}", flush=True)
                continue
            raw = Path(cell["raw"])
            plan_path = Path(cell["plan"])
            if (raw.exists() or plan_path.exists()) and not args.overwrite:
                raise MatchedLatencyError(
                    f"incomplete output exists for pair {cell['pair_id']}; use --overwrite: {raw}"
                )
            if args.overwrite:
                for path in (raw, plan_path, raw.with_suffix(raw.suffix + ".log")):
                    if path.exists():
                        path.unlink()
            cell["d3_namespace_rows_deleted"] = {
                namespace: ensure_fresh_namespace(
                    str(cell["d3_fragment_store_table"]),
                    namespace,
                    overwrite=args.overwrite,
                )
                for namespace in cell["d3_repeat_namespaces"]
            }
            cell["status"] = "running"
            cell["started_at"] = utc_now()
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            log = raw.with_suffix(raw.suffix + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"running pair={cell['pair_id']} target={cell['target_recall']:.3f}", flush=True)
            with log.open("w", encoding="utf-8") as output:
                completed = subprocess.run(
                    cell["command"], cwd=ROOT, env=os.environ.copy(), stdout=output,
                    stderr=subprocess.STDOUT, check=False,
                )
            cell["returncode"] = completed.returncode
            cell["completed_at"] = utc_now()
            pair = next(item for item in pairs if item.pair_id == cell["pair_id"])
            if completed.returncode == 0 and raw.is_file() and plan_path.is_file():
                quality_gate = matched_recall_gate(
                    read_raw_rows(raw),
                    pair,
                    str(cell["qualification_scope"]),
                )
                write_quality_gate_to_plan(plan_path, quality_gate)
                cell["quality_gate"] = quality_gate
                cell["predicate_completion"] = predicate_completion_contract(
                    quality_gate
                )
            if completed.returncode != 0 or not cell_complete(raw, plan_path, pair, config, cell):
                cell["status"] = "failed"
                cell["log"] = str(log)
                manifest["status"] = "failed"
                manifest["updated_at"] = utc_now()
                atomic_json(manifest_path, manifest)
                raise MatchedLatencyError(f"pair failed: {cell['pair_id']}; see {log}")
            cell["status"] = "complete"
            cell["raw_sha256"] = sha256_file(raw)
            cell["plan_sha256"] = sha256_file(plan_path)
            cell["observed_execution_sources"] = read_json(
                plan_path, "cell plan"
            ).get("execution_sources")
            manifest["pairs_complete"] = sum(item["status"] == "complete" for item in schedule)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
        manifest["status"] = "complete"
        manifest["artifact_valid"] = True
        manifest["requested_slice_complete"] = True
        manifest["full_release_complete"] = bool(release_scope["requested"])
        manifest["paper_eligible"] = bool(
            manifest["full_release_complete"]
            and selection_bindings["qualification_scope"]
            == QUALIFICATION_SCOPE_FORMAL
        )
        if not manifest["paper_eligible"]:
            if selection_bindings["qualification_scope"] == QUALIFICATION_SCOPE_AGGREGATE:
                manifest["paper_eligible_reason"] = (
                    "aggregate_lcb is legacy/audit-only and cannot produce a paper artifact"
                )
            else:
                manifest["paper_eligible_reason"] = (
                    "requested slice is complete, but the run does not cover the "
                    "full frozen pair and backend-CPU release protocol"
                )
        manifest["completed_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        print(f"wrote {manifest_path}", flush=True)
        return 0
    finally:
        lock.close()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--selection-plan", type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument(
        "--required-grid-contract",
        type=Path,
        help=(
            "Explicit complete required-grid contract. Final runs fail closed "
            "when it is absent or disagrees with the selector or --config."
        ),
    )
    parser.add_argument("--datasets", nargs="*", choices=("amazon", "yfcc", "laion"), default=[])
    parser.add_argument("--pair-ids", nargs="*", default=[])
    parser.add_argument("--backend-cpu-list", default=DEFAULT_BACKEND_CPU_LIST)
    parser.add_argument("--out-dir", type=Path, default=RESULTS / "figure5_r35/matched_latency")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_parser().parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatchedLatencyError as exc:
        print(f"matched-latency error: {exc}", file=sys.stderr)
        raise SystemExit(2)
