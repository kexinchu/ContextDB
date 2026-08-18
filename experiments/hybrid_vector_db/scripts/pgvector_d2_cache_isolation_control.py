"""Fail-closed cache-isolation control for the SQLens D1/D2 crossover.

The experiment has exactly two arms: D1 safe-guided on the source HNSW
relation and the same D1 safe-guided query on its same-graph BFS clone.  The
matched-recall artifact supplies one audited configuration per filter; the
controller never substitutes one global ef-search value.  Every invocation
uses either a full target-index warm-resident protocol or an explicitly
authorized relation-scoped page-cache eviction protocol, and rotates the two
arms over five seeded paired blocks.  Any cache, plan, exact-truth, runtime,
relation, or graph-semantic gate failure invalidates the artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = Path(__file__).with_name(
    "pgvector_design1_design2_design3_selectivity_benchmark.py"
)
DEFAULT_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
DEFAULT_SOURCE_INDEX = "public.amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx"
DEFAULT_BFS_INDEX = "public.amazon10m_hnsw_m32ef200_dupbridge_r29_bfs_idx"
DEFAULT_FILTERS = (
    ROOT
    / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_valid_embeddings_filters.csv"
)
DEFAULT_TRUTH = (
    ROOT
    / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_unique_embeddings_formal.csv"
)
DEFAULT_TRUTH_MANIFEST = DEFAULT_TRUTH.with_name(DEFAULT_TRUTH.stem + "_manifest.json")
QUERY_OFFSET = 100
QUERIES = 100
REPEATS = 5
COLD_BLOCK_QUERIES = 1
SCHEDULE_SEED = 20260721
ARMS = ("d1_source", "d1_bfs")
PROTOCOL_NAME = "sqlens-d2-cache-isolation-v5"
PROTOCOL_VERSION = 5
REQUIRED_PROFILE_SEMANTICS_VERSION = 12
COLD_PROTOCOL_SEMANTICS_VERSION = 4
PROFILE_TIME_ABS_TOLERANCE_MS = 0.001
LEGACY_PROTOCOL_NAMES = frozenset(
    {"sqlens-d2-cache-isolation-v3", "sqlens-d2-cache-isolation-v4"}
)
R33_PROFILE_COUNT_FIELDS = (
    "index_readbuffer_calls",
    "index_readbuffer_shared_read_calls",
    "index_readbuffer_shared_hit_calls",
    "index_readbuffer_unclassified_calls",
    "distance_compute_timed_calls",
)
R33_PROFILE_TIME_FIELDS = (
    "index_readbuffer_ms",
    "index_readbuffer_shared_read_ms",
    "index_readbuffer_shared_hit_ms",
    "index_readbuffer_unclassified_ms",
    "distance_compute_ms",
    "hnsw_remaining_ms",
)
R33_PROFILE_SCOPE_FIELDS = (
    "index_readbuffer_timing_scope",
    "index_readbuffer_classification_scope",
    "distance_compute_timing_scope",
    "hnsw_remaining_scope",
    "profile_timer_overhead_scope",
)
REQUIRED_PROFILE_FIELDS = (
    "activation_ms",
    "query_latency_ms",
    "vector_search_ms",
    "hnsw_am_callback_ms",
    "distance_compute_count",
    "index_page_loads",
    "index_page_runs",
    "index_page_distinct_pages",
    "index_page_distinct_pages_exact",
    "index_page_profile_scope",
    "index_page_prefetches",
    "page_access_prefetches",
    "idx_blks_hit",
    "idx_blks_read",
    "heap_blks_hit",
    "heap_blks_read",
    "heap_blks_are_exact_heap_io",
    "heap_tid_page_runs",
    "profile_semantics_version",
    *R33_PROFILE_COUNT_FIELDS,
    *R33_PROFILE_TIME_FIELDS,
    *R33_PROFILE_SCOPE_FIELDS,
    "hnsw_remaining_ms_is_residual",
    "index_page_transition_count",
    "index_page_same_block_transitions",
    "index_page_within_1_page_transitions",
    "index_page_within_4_pages_transitions",
    "index_page_within_16_pages_transitions",
    "index_page_backward_transitions",
    "index_page_total_abs_block_delta",
    "index_page_max_abs_block_delta",
    "index_page_trace_statistics_scope",
    "index_page_trace_sample_limit",
    "index_page_trace_sample_count",
    "index_page_trace_sample_truncated",
    "index_page_trace_sample_scope",
    "index_page_trace_sample",
)
CACHE_REGIMES = ("warm_resident", "cold_io")
PAIRED_D1_SEMANTIC_FIELDS = (
    "guidance_enabled",
    "guidance_scan_verified",
    "guidance_binding_verified",
    "guidance_route",
    "guidance_kind",
    "activation_atom_count",
    "guidance_filter_strategy",
    "final_path",
    "visited_tuples",
    "returned_tuples",
    "distance_compute_count",
    "guidance_checks",
    "traversal_guidance_matches",
    "guidance_skips",
    "traversal_expanded_nodes",
    "traversal_neighbors_examined",
    "index_readbuffer_calls",
    "distance_compute_timed_calls",
    *R33_PROFILE_SCOPE_FIELDS,
)


class ControlError(RuntimeError):
    """The cache-isolation contract could not be established."""


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    index_role: str
    expected_index: str


@dataclass(frozen=True)
class MatchedConfig:
    filter_name: str
    target_recall: float
    ef_search: int
    max_scan_tuples: int
    scan_mem_multiplier: float
    iterative_scan: str
    guided_collect_target: int
    qualification: str = "lcb95"
    calibration_recall_mean: float | None = None
    calibration_recall_lcb95: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ControlError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def resolve_artifact_path(value: object, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def canonical_public_relation(value: object) -> str:
    name = str(value or "")
    return name if "." in name else f"public.{name}"


def require_sha256(path: Path, expected: object, label: str) -> None:
    if not path.is_file():
        raise ControlError(f"{label} does not exist: {path}")
    observed = sha256_file(path)
    if str(expected or "").lower() != observed:
        raise ControlError(
            f"{label} SHA256 mismatch: expected={expected!r}, observed={observed!r}"
        )


def canonical_optional_relation(value: object) -> str | None:
    return canonical_public_relation(value) if value not in (None, "") else None


def require_bound_artifact(
    identity: Mapping[str, Any],
    actual_path: Path,
    label: str,
    *,
    require_path: bool,
) -> None:
    bound_path = identity.get("path")
    if require_path and not bound_path:
        raise ControlError(f"{label} provenance has no path")
    if bound_path and resolve_artifact_path(bound_path, ROOT) != actual_path.resolve():
        raise ControlError(
            f"{label} path mismatch: expected={actual_path.resolve()}, "
            f"observed={resolve_artifact_path(bound_path, ROOT)}"
        )
    require_sha256(actual_path, identity.get("sha256"), label)


def require_manifest_value(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise ControlError(
            f"{label} mismatch: expected={expected!r}, observed={observed!r}"
        )


def audit_exact_truth_manifest(
    manifest_path: Path,
    truth_csv: Path,
    filters_csv: Path,
    *,
    expected_table: str | None = None,
    expected_index: str | None = None,
    expected_query_table: str | None = None,
    expected_query_id_column: str = "id",
    expected_query_vector_column: str = "embedding",
    expected_candidate_validity_predicate: str | None = None,
    expected_self_excluded: bool = True,
    query_offset: int = QUERY_OFFSET,
    queries: int = QUERIES,
    expected_filter_names: Sequence[str] | None = None,
    expected_matched_manifest: Path | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlError(f"cannot read exact-truth manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ControlError("truth provenance manifest is not a JSON object")
    external_launch = all(
        isinstance(payload.get(field), dict)
        for field in ("dataset", "database", "truth", "filters", "protocol")
    )
    supported_recall_contracts = {
        "distance_squared_threshold_tie_aware_v1",
        "returned SQL-valid IDs with squared L2 <= kth_distance_sq + tie_tolerance, capped at k",
    }
    if external_launch:
        if payload.get("status") != "complete" or payload.get("ready") is not True:
            raise ControlError("external truth launch manifest is not complete and ready")
        dataset = payload["dataset"]
        database = payload["database"]
        truth_identity = payload["truth"]
        filter_identity = payload["filters"]
        protocol = payload["protocol"]
        for label, section in (
            ("database", database),
            ("truth", truth_identity),
            ("filters", filter_identity),
        ):
            if section.get("errors") not in (None, []):
                raise ControlError(f"external truth launch {label} section reports errors")
        if database.get("ready") is not True or truth_identity.get("ready") is not True:
            raise ControlError("external truth launch database/truth readiness gate failed")
        require_bound_artifact(
            truth_identity, truth_csv, "external exact truth CSV", require_path=True
        )
        require_bound_artifact(
            filter_identity, filters_csv, "external truth filters CSV", require_path=True
        )
        require_manifest_value(
            canonical_optional_relation(dataset.get("table")),
            canonical_optional_relation(expected_table),
            "external truth table",
        )
        require_manifest_value(
            canonical_optional_relation(dataset.get("index")),
            canonical_optional_relation(expected_index),
            "external truth source index",
        )
        require_manifest_value(
            canonical_optional_relation(dataset.get("query_table")),
            canonical_optional_relation(expected_query_table),
            "external truth query table",
        )
        require_manifest_value(
            dataset.get("query_id_column"),
            expected_query_id_column,
            "external truth query ID column",
        )
        require_manifest_value(
            dataset.get("query_vector_column"),
            expected_query_vector_column,
            "external truth query vector column",
        )
        require_manifest_value(
            canonical_optional_relation(database.get("index")),
            canonical_optional_relation(expected_index),
            "external launch database index",
        )
        relations = database.get("relations")
        if not isinstance(relations, dict):
            raise ControlError("external truth launch has no database relation inventory")
        expected_relations = {
            canonical_optional_relation(expected_table),
            canonical_optional_relation(expected_query_table),
        }
        observed_relations = {
            canonical_optional_relation(name) for name in relations
        }
        if not {name for name in expected_relations if name}.issubset(observed_relations):
            raise ControlError("external truth launch relation inventory is incomplete")
        final_split = protocol.get("final")
        if not isinstance(final_split, dict):
            raise ControlError("external truth launch has no final split")
        for field, expected in (
            ("offset", query_offset),
            ("queries", queries),
        ):
            try:
                observed = int(final_split.get(field, -1))
            except (TypeError, ValueError) as exc:
                raise ControlError(f"external truth final {field} is invalid") from exc
            require_manifest_value(observed, expected, f"external truth final {field}")
        require_manifest_value(
            protocol.get("candidate_validity_predicate"),
            expected_candidate_validity_predicate,
            "external truth candidate-validity predicate",
        )
        require_manifest_value(
            protocol.get("truth_self_excluded"),
            expected_self_excluded,
            "external truth self-exclusion contract",
        )
        if expected_filter_names is not None:
            launch_filters = dataset.get("filter_names")
            if not isinstance(launch_filters, list) or not set(expected_filter_names).issubset(
                {str(name) for name in launch_filters}
            ):
                raise ControlError("external truth launch does not cover requested filters")
        if expected_matched_manifest is not None:
            generic = payload.get("generic_manifest")
            if not isinstance(generic, dict):
                raise ControlError("external truth launch does not bind a matched manifest")
            require_bound_artifact(
                generic,
                expected_matched_manifest,
                "external launch matched manifest",
                require_path=True,
            )
        recall_contract = "distance_squared_threshold_tie_aware_v1"
        manifest_self_excluded = protocol.get("truth_self_excluded")
        provenance_kind = "external_launch_manifest"
    else:
        if payload.get("artifact_valid") is not True:
            raise ControlError("exact-truth manifest is not an audited valid artifact")
        outputs = payload.get("outputs")
        truth_identity = outputs.get("truth_csv") if isinstance(outputs, dict) else None
        if not isinstance(truth_identity, dict):
            raise ControlError("exact-truth manifest has no truth_csv output identity")
        require_bound_artifact(
            truth_identity, truth_csv, "exact truth CSV", require_path=False
        )
        inputs = payload.get("inputs")
        filter_identity = inputs.get("filters_csv") if isinstance(inputs, dict) else None
        if isinstance(filter_identity, dict) and filter_identity.get("sha256"):
            require_bound_artifact(
                filter_identity,
                filters_csv,
                "exact-truth filters CSV",
                require_path=False,
            )
        recall_contract = payload.get("recall_contract")
        if recall_contract not in supported_recall_contracts:
            raise ControlError("exact-truth manifest uses an unsupported recall contract")
        manifest_self_excluded = payload.get("self_excluded")
        require_manifest_value(
            manifest_self_excluded,
            expected_self_excluded,
            "exact-truth self-exclusion contract",
        )
        postgres = inputs.get("postgres") if isinstance(inputs, dict) else None
        if expected_table is not None and (
            not isinstance(postgres, dict)
            or canonical_public_relation(postgres.get("table"))
            != canonical_public_relation(expected_table)
        ):
            raise ControlError("exact-truth PostgreSQL table does not match D2 table")
        if expected_candidate_validity_predicate is not None:
            observed_predicate = (
                postgres.get("query_population", {}).get(
                    "candidate_validity_predicate"
                )
                if isinstance(postgres, dict)
                and isinstance(postgres.get("query_population"), dict)
                else None
            )
            require_manifest_value(
                observed_predicate,
                expected_candidate_validity_predicate,
                "exact-truth candidate-validity predicate",
            )
        provenance_kind = "exact_truth_manifest"

    rows = read_csv(truth_csv)
    try:
        query_nos = sorted(
            {int(row["query_no"]) for row in rows if row.get("query_no") is not None}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ControlError("exact truth contains an invalid query_no") from exc
    expected_query_nos = set(range(query_offset, query_offset + queries))
    if not expected_query_nos.issubset(query_nos):
        raise ControlError(
            f"exact truth does not cover q{query_offset}..q{query_offset + queries - 1}"
        )
    requested_filters = (
        set(expected_filter_names)
        if expected_filter_names is not None
        else {str(row.get("filter_name", "")) for row in rows}
    )
    requested_rows = [
        row
        for row in rows
        if int(row["query_no"]) in expected_query_nos
        and str(row.get("filter_name", "")) in requested_filters
    ]
    expected_cells = {
        (filter_name, query_no)
        for filter_name in requested_filters
        for query_no in expected_query_nos
    }
    observed_cells = {
        (str(row.get("filter_name", "")), int(row["query_no"]))
        for row in requested_rows
    }
    if observed_cells != expected_cells:
        raise ControlError("exact truth does not cover every requested filter/query cell")
    for row in requested_rows:
        if "self_excluded" not in row or parse_bool(row["self_excluded"]) is not expected_self_excluded:
            raise ControlError("truth CSV self_excluded value differs from the CLI contract")
        if expected_candidate_validity_predicate is not None and row.get(
            "candidate_validity_predicate"
        ) != expected_candidate_validity_predicate:
            raise ControlError(
                "truth CSV candidate-validity predicate differs from the CLI contract"
            )
        if row.get("query_split") not in (None, "", "final"):
            raise ControlError("truth CSV requested rows are not in the final split")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "artifact_valid": True,
        "provenance_kind": provenance_kind,
        "truth_csv": {"path": str(truth_csv.resolve()), "sha256": sha256_file(truth_csv)},
        "filters_csv_sha256": sha256_file(filters_csv),
        "recall_contract": recall_contract,
        "normalized_recall_contract": "distance_squared_threshold_tie_aware_v1",
        "self_excluded": manifest_self_excluded,
        "truth_rows": len(rows),
        "query_no_min": min(query_nos),
        "query_no_max": max(query_nos),
        "measurement_query_offset": query_offset,
        "measurement_queries": queries,
        "measurement_filter_count": len(requested_filters),
    }


def audit_matched_recall_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    filters_csv: Path,
    truth_csv: Path,
) -> dict[str, MatchedConfig]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlError(f"cannot read matched-recall manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ControlError("matched-recall manifest is not complete")
    if payload.get("artifact_valid") is False or payload.get("comparison_valid") is not True:
        raise ControlError("matched-recall manifest failed its independent comparison gate")
    run_spec = payload.get("run_spec")
    run_args = run_spec.get("args") if isinstance(run_spec, dict) else None
    if not isinstance(run_args, dict):
        raise ControlError("matched-recall manifest has no audited run_spec.args")
    if run_args.get("guidance_filter_strategy") != "safe_guided":
        raise ControlError("D2 control accepts only a safe_guided matched-recall configuration")
    for field, expected in (
        ("final_query_offset", args.query_offset),
        ("final_queries", args.queries),
        ("final_repeats", args.repeats),
    ):
        try:
            observed = int(run_args.get(field, -1))
        except (TypeError, ValueError) as exc:
            raise ControlError(f"matched-recall {field} is invalid") from exc
        require_manifest_value(observed, expected, f"matched-recall {field}")
    expected_pairs = {"insertion_table": args.table, "bfs_table": args.table}
    if args.matched_config_index_policy == "exact":
        expected_pairs.update(
            {"insertion_index": args.source_index, "bfs_index": args.bfs_index}
        )
    for field, expected in expected_pairs.items():
        observed = run_args.get(field)
        if canonical_optional_relation(observed) != canonical_optional_relation(expected):
            raise ControlError(
                f"matched-recall {field} mismatch: expected={expected!r}, observed={observed!r}"
            )
    for field in ("insertion_index", "bfs_index"):
        if canonical_optional_relation(run_args.get(field)) is None:
            raise ControlError(f"matched-recall manifest does not bind {field}")
    require_manifest_value(
        canonical_optional_relation(run_args.get("query_table")),
        canonical_optional_relation(args.query_table),
        "matched-recall query table",
    )
    require_manifest_value(
        run_args.get("query_id_column"),
        args.query_id_column,
        "matched-recall query ID column",
    )
    require_manifest_value(
        run_args.get("query_vector_column"),
        args.query_vector_column,
        "matched-recall query vector column",
    )
    if run_args.get("candidate_validity_predicate") != args.candidate_validity_predicate:
        raise ControlError("matched-recall candidate-validity predicate differs from D2 query")
    require_manifest_value(
        run_args.get("expected_truth_self_excluded"),
        args.expected_truth_self_excluded,
        "matched-recall truth self-exclusion contract",
    )
    require_manifest_value(
        payload.get("self_excluded"),
        args.expected_truth_self_excluded,
        "matched-recall top-level self-exclusion contract",
    )
    bound_truth = run_args.get("truth_csv")
    if not bound_truth:
        raise ControlError("matched-recall manifest does not bind an exact truth CSV")
    require_bound_artifact(
        {"path": bound_truth, "sha256": sha256_file(truth_csv)},
        truth_csv,
        "matched-recall truth CSV",
        require_path=True,
    )
    bound_filters = run_args.get("filters_csv")
    if not bound_filters:
        raise ControlError("matched-recall manifest does not bind a filters CSV")
    require_bound_artifact(
        {"path": bound_filters, "sha256": sha256_file(filters_csv)},
        filters_csv,
        "matched-recall filters CSV",
        require_path=True,
    )
    outputs = payload.get("outputs")
    selected_identity = outputs.get("selected") if isinstance(outputs, dict) else None
    if not isinstance(selected_identity, dict):
        raise ControlError("matched-recall manifest has no selected configuration artifact")
    selected_path = resolve_artifact_path(selected_identity.get("path"), ROOT)
    require_sha256(selected_path, selected_identity.get("sha256"), "matched-recall selected CSV")
    selected_rows = read_csv(selected_path)
    wanted_filters = set(args.filter_names)
    configs: dict[str, MatchedConfig] = {}
    for row in selected_rows:
        if row.get("mode") != args.matched_mode or row.get("guidance_filter_strategy") != "safe_guided":
            continue
        if str(row.get("selection_status")) != "selected":
            continue
        try:
            target = float(row["target_recall"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlError(f"invalid matched-recall target in {selected_path}") from exc
        if not math.isclose(target, args.matched_target_recall, rel_tol=0.0, abs_tol=1e-9):
            continue
        lcb_qualified = row.get("target_lcb95_met_in_calibration") == "True"
        if not lcb_qualified:
            mean_qualified = (
                row.get("target_met_in_calibration") == "True"
                and row.get("target_confirmed_in_calibration") == "True"
            )
            if not args.allow_mean_qualified_matched_config or not mean_qualified:
                raise ControlError(
                    f"matched-recall row for {row.get('filter_name')} is not LCB95-qualified"
                )
        name = str(row.get("filter_name", ""))
        if name not in wanted_filters:
            continue
        try:
            config = MatchedConfig(
                filter_name=name,
                target_recall=target,
                ef_search=int(row["ef_search"]),
                max_scan_tuples=int(row["max_scan_tuples"]),
                scan_mem_multiplier=float(row["scan_mem_multiplier"]),
                iterative_scan=str(row["iterative_scan"]),
                guided_collect_target=1,
                qualification=("lcb95" if lcb_qualified else "mean_confirmed"),
                calibration_recall_mean=(
                    float(row["recall_mean"]) if row.get("recall_mean") else None
                ),
                calibration_recall_lcb95=(
                    float(row["recall_lcb95"]) if row.get("recall_lcb95") else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlError(f"invalid matched-recall config for {name}") from exc
        if config.ef_search <= 0 or config.max_scan_tuples <= 0 or config.scan_mem_multiplier <= 0 or config.iterative_scan not in {"off", "strict_order"}:
            raise ControlError(f"matched-recall config for {name} has invalid search parameters")
        if name in configs:
            raise ControlError(f"matched-recall selected CSV has duplicate config for {name}")
        configs[name] = config
    missing = sorted(wanted_filters - set(configs))
    if missing:
        raise ControlError(f"matched-recall selected CSV lacks D1 configs for filters: {missing}")
    return configs


MATCHED_CONFIG_COLUMNS = (
    "filter_name",
    "target_recall",
    "mode",
    "ef_search",
    "max_scan_tuples",
    "scan_mem_multiplier",
    "iterative_scan",
    "qualification",
    "calibration_recall_mean",
    "calibration_recall_lcb95",
)


def optional_finite_float(value: object, label: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ControlError(f"{label} is not numeric") from exc
    if not math.isfinite(parsed):
        raise ControlError(f"{label} is not finite")
    return parsed


def manifest_contract_args(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = payload.get("protocol")
    if isinstance(protocol, Mapping):
        return protocol
    run_spec = payload.get("run_spec")
    run_args = run_spec.get("args") if isinstance(run_spec, Mapping) else None
    if isinstance(run_args, Mapping):
        return run_args
    raise ControlError("matched-configs manifest has no protocol or run_spec.args")


def contract_value(
    contract: Mapping[str, Any], primary: str, compatibility: str | None = None
) -> object:
    if primary in contract:
        return contract.get(primary)
    return contract.get(compatibility) if compatibility else None


def audit_matched_configs_csv(
    csv_path: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    filters_csv: Path,
    truth_csv: Path,
) -> dict[str, MatchedConfig]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlError(
            f"cannot read matched-configs manifest {manifest_path}: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "complete"
        or payload.get("artifact_valid") is not True
    ):
        raise ControlError("matched-configs manifest is not a complete valid artifact")

    outputs = payload.get("outputs")
    csv_identity = None
    if isinstance(outputs, Mapping):
        csv_identity = outputs.get("matched_configs_csv") or outputs.get("configs_csv")
    if not isinstance(csv_identity, Mapping):
        raise ControlError("matched-configs manifest has no CSV output identity")
    require_bound_artifact(
        csv_identity, csv_path, "matched-configs CSV", require_path=True
    )

    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        runtime = payload.get("sqlens_runtime_provenance")
    if not isinstance(runtime, Mapping):
        raise ControlError("matched-configs manifest has no runtime provenance")
    observed_build = runtime.get("sqlens_build_id") or runtime.get(
        "loaded_vector_sqlens_build_id"
    )
    observed_binary = runtime.get("vector_so_sha256") or runtime.get(
        "loaded_vector_so_sha256"
    )
    require_manifest_value(
        observed_build,
        args.expected_sqlens_build_id,
        "matched-configs SQLens build ID",
    )
    require_manifest_value(
        str(observed_binary or "").lower(),
        args.expected_vector_so_sha256,
        "matched-configs vector.so SHA256",
    )

    contract = manifest_contract_args(payload)
    mode = contract.get("mode")
    modes = contract.get("modes")
    if mode != "design1_bloom" and not (
        isinstance(modes, list) and "design1_bloom" in modes
    ):
        raise ControlError("matched-configs manifest does not authorize design1_bloom")
    relation_contract = {
        "table": ("table", "insertion_table", args.table),
        "source index": ("source_index", "insertion_index", args.source_index),
        "BFS index": ("bfs_index", None, args.bfs_index),
        "query table": ("query_table", None, args.query_table),
    }
    for label, (primary, compatibility, expected) in relation_contract.items():
        observed = contract_value(contract, primary, compatibility)
        if canonical_optional_relation(observed) != canonical_optional_relation(expected):
            raise ControlError(
                f"matched-configs {label} mismatch: expected={expected!r}, "
                f"observed={observed!r}"
            )
    for field, expected in (
        ("query_id_column", args.query_id_column),
        ("query_vector_column", args.query_vector_column),
        ("candidate_validity_predicate", args.candidate_validity_predicate),
        ("expected_truth_self_excluded", args.expected_truth_self_excluded),
        ("guidance_filter_strategy", "safe_guided"),
        ("guidance_max_atoms", args.guidance_max_atoms),
    ):
        require_manifest_value(
            contract.get(field), expected, f"matched-configs {field}"
        )
    for field, compatibility, expected in (
        ("query_offset", "final_query_offset", args.query_offset),
        ("queries", "final_queries", args.queries),
        ("repeats", "final_repeats", args.repeats),
    ):
        try:
            observed = int(contract_value(contract, field, compatibility))
        except (TypeError, ValueError) as exc:
            raise ControlError(f"matched-configs {field} is invalid") from exc
        require_manifest_value(observed, expected, f"matched-configs {field}")

    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ControlError("matched-configs manifest has no input provenance")
    for field, path, label in (
        ("truth_csv", truth_csv, "matched-configs truth CSV"),
        ("filters_csv", filters_csv, "matched-configs filters CSV"),
        (
            "truth_provenance_manifest",
            args.truth_manifest,
            "matched-configs truth provenance manifest",
        ),
    ):
        identity = inputs.get(field)
        if not isinstance(identity, Mapping):
            raise ControlError(f"matched-configs manifest does not bind {field}")
        require_bound_artifact(identity, path, label, require_path=True)

    with csv_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or [])
        missing_columns = sorted(set(MATCHED_CONFIG_COLUMNS) - fields)
        if missing_columns:
            raise ControlError(
                f"matched-configs CSV is missing required columns: {missing_columns}"
            )
        rows = list(reader)
    wanted_filters = set(args.filter_names)
    configs: dict[str, MatchedConfig] = {}
    for row in rows:
        if row.get("mode") != "design1_bloom":
            continue
        name = str(row.get("filter_name") or "")
        if name not in wanted_filters:
            continue
        try:
            target = float(row["target_recall"])
        except (TypeError, ValueError) as exc:
            raise ControlError(f"matched-configs target_recall is invalid for {name}") from exc
        if not math.isfinite(target) or not math.isclose(
            target, args.matched_target_recall, rel_tol=0.0, abs_tol=1e-9
        ):
            continue
        qualification = str(row.get("qualification") or "")
        if qualification not in {"lcb95", "mean_confirmed"}:
            raise ControlError(
                f"matched-configs qualification is invalid for {name}: {qualification!r}"
            )
        if qualification == "mean_confirmed" and not args.allow_mean_qualified_matched_config:
            raise ControlError(
                f"matched-configs row for {name} is not LCB95-qualified"
            )
        try:
            config = MatchedConfig(
                filter_name=name,
                target_recall=target,
                ef_search=int(row["ef_search"]),
                max_scan_tuples=int(row["max_scan_tuples"]),
                scan_mem_multiplier=float(row["scan_mem_multiplier"]),
                iterative_scan=str(row["iterative_scan"]),
                guided_collect_target=1,
                qualification=qualification,
                calibration_recall_mean=optional_finite_float(
                    row.get("calibration_recall_mean"),
                    f"matched-configs calibration_recall_mean for {name}",
                ),
                calibration_recall_lcb95=optional_finite_float(
                    row.get("calibration_recall_lcb95"),
                    f"matched-configs calibration_recall_lcb95 for {name}",
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ControlError(f"matched-configs row is invalid for {name}") from exc
        if (
            config.ef_search <= 0
            or config.max_scan_tuples <= 0
            or not math.isfinite(config.scan_mem_multiplier)
            or config.scan_mem_multiplier <= 0
            or config.iterative_scan not in {"off", "strict_order"}
        ):
            raise ControlError(f"matched-configs search parameters are invalid for {name}")
        if name in configs:
            raise ControlError(f"matched-configs CSV has duplicate D1 config for {name}")
        configs[name] = config
    missing = sorted(wanted_filters - set(configs))
    if missing:
        raise ControlError(f"matched-configs CSV lacks D1 configs for filters: {missing}")
    return configs


def load_matched_configs(
    args: argparse.Namespace, filters_csv: Path, truth_csv: Path
) -> dict[str, MatchedConfig]:
    csv_path = getattr(args, "matched_configs_csv", None)
    if csv_path is not None:
        manifest_path = getattr(args, "matched_configs_manifest", None)
        if manifest_path is None:
            raise ControlError("--matched-configs-csv requires --matched-configs-manifest")
        return audit_matched_configs_csv(
            csv_path, manifest_path, args, filters_csv, truth_csv
        )
    manifest_path = getattr(args, "matched_recall_manifest", None)
    if manifest_path is None:
        raise ControlError(
            "provide --matched-recall-manifest or the matched-configs CSV/manifest pair"
        )
    return audit_matched_recall_manifest(
        manifest_path, args, filters_csv, truth_csv
    )


def matched_config_source_evidence(
    args: argparse.Namespace, configs: Mapping[str, MatchedConfig]
) -> dict[str, Any]:
    csv_path = getattr(args, "matched_configs_csv", None)
    if csv_path is not None:
        manifest_path = args.matched_configs_manifest
        return {
            "kind": "audited_matched_configs_csv",
            "csv": {"path": str(csv_path.resolve()), "sha256": sha256_file(csv_path)},
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "mode": "design1_bloom",
            "configs": {name: value.as_dict() for name, value in configs.items()},
        }
    manifest_path = args.matched_recall_manifest
    return {
        "kind": "legacy_matched_recall_manifest",
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "mode": "design1_bloom",
        "configs": {name: value.as_dict() for name, value in configs.items()},
    }


def legacy_matched_recall_evidence(
    args: argparse.Namespace, configs: Mapping[str, MatchedConfig]
) -> dict[str, Any] | None:
    if getattr(args, "matched_configs_csv", None) is not None:
        return None
    manifest_path = args.matched_recall_manifest
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "configs": {name: value.as_dict() for name, value in configs.items()},
    }


def resume_runtime_input_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Return immutable command/runtime inputs bound to a resumable run."""
    path_identities = {}
    for field, path in (
        ("filters_csv", args.filters_csv),
        ("truth_csv", args.truth_csv),
        ("truth_manifest", args.truth_manifest),
        ("d2_graph_proof_json", args.d2_graph_proof_json),
    ):
        path_identities[field] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    return {
        "table": args.table,
        "source_index": args.source_index,
        "bfs_index": args.bfs_index,
        "query_table": args.query_table,
        "query_id_column": args.query_id_column,
        "query_vector_column": args.query_vector_column,
        "candidate_validity_predicate": args.candidate_validity_predicate,
        "expected_truth_self_excluded": args.expected_truth_self_excluded,
        "expected_sqlens_build_id": args.expected_sqlens_build_id,
        "expected_vector_so_sha256": args.expected_vector_so_sha256,
        "cache_regime": args.cache_regime,
        "inputs": path_identities,
    }


def argv_without_resume(argv: Sequence[object]) -> list[str]:
    """Normalize only the explicit resume switch for strict argv comparison."""
    return [str(value) for value in argv if str(value) != "--resume"]


def validate_generic_plan_artifact(path: Path) -> None:
    """Validate immutable child plan evidence before any database work."""
    if not path.is_file():
        raise ControlError(f"missing delegated EXPLAIN evidence: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlError(f"invalid delegated EXPLAIN evidence {path}: {exc}") from exc
    checks = payload.get("checks")
    if payload.get("status") != "complete" or not isinstance(checks, list) or not checks:
        raise ControlError(f"delegated EXPLAIN evidence is incomplete: {path}")
    if any(not isinstance(check, Mapping) or check.get("passed") is not True for check in checks):
        raise ControlError(f"delegated EXPLAIN evidence has a failed check: {path}")


def validate_resume_manifest(
    args: argparse.Namespace,
    manifest_path: Path,
    schedule: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    exact_truth: Mapping[str, Any],
    matched_source: Mapping[str, Any],
    *,
    current_argv: Sequence[object] | None = None,
    current_controller_sha256: str | None = None,
    current_runner_sha256: str | None = None,
    current_two_arm_code_path: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate a resumable manifest without connecting to PostgreSQL.

    Only a complete invocation prefix may be reused.  A running/failed child,
    an artifact left beside an unstarted invocation, or an aggregate output
    that could be overwritten invalidates the resume request.
    """
    if not manifest_path.is_file():
        raise ControlError(f"--resume requires an existing manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlError(f"cannot read resume manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ControlError("resume manifest is not a JSON object")
    if manifest.get("status") not in {"running", "failed"}:
        raise ControlError(
            "--resume accepts only a running or failed manifest; completed runs are immutable"
        )
    if manifest.get("protocol") != dict(protocol):
        raise ControlError("resume run protocol differs from the existing manifest")

    observed_argv = manifest.get("argv")
    if not isinstance(observed_argv, list):
        raise ControlError("resume manifest has no original argv")
    expected_argv = list(current_argv) if current_argv is not None else list(sys.argv)
    if argv_without_resume(observed_argv) != argv_without_resume(expected_argv):
        raise ControlError("resume argv differs from the existing run")

    if manifest.get("controller_sha256") != (
        current_controller_sha256 or sha256_file(Path(__file__))
    ):
        raise ControlError("resume controller source identity differs from the run")
    if manifest.get("runner_sha256") != (
        current_runner_sha256 or sha256_file(RUNNER_PATH)
    ):
        raise ControlError("resume delegated runner identity differs from the run")
    if current_two_arm_code_path is not None and manifest.get("two_arm_code_path") != dict(
        current_two_arm_code_path
    ):
        raise ControlError("resume two-arm code path differs from the run")
    if manifest.get("exact_truth_audit") != dict(exact_truth):
        raise ControlError("resume exact-truth input identity differs from the run")
    if manifest.get("matched_config_source") != dict(matched_source):
        raise ControlError("resume matched-configuration input identity differs from the run")
    if manifest.get("runtime_input_identity") != resume_runtime_input_identity(args):
        raise ControlError("resume runtime/input identity differs from the run")
    if manifest.get("schedule") != [dict(item) for item in schedule]:
        raise ControlError("resume schedule differs from the existing run")

    output_path = args.out.resolve()
    summary_path = args.out.with_name(args.out.stem + "_summary.csv").resolve()
    if args.out.exists() or summary_path.exists():
        raise ControlError(
            "resume refuses to overwrite an existing aggregate raw/summary artifact"
        )

    records = manifest.get("invocations")
    if not isinstance(records, list) or len(records) > len(schedule):
        raise ControlError("resume manifest has an invalid invocation prefix")
    reused: list[dict[str, Any]] = []
    for expected, record in zip(schedule, records, strict=False):
        if not isinstance(record, Mapping):
            raise ControlError("resume invocation record is not an object")
        for field in ("sequence", "control_repeat", "position", "arm", "filter_name"):
            if record.get(field) != expected.get(field):
                raise ControlError(f"resume invocation schedule drifted at {field}")
        status = record.get("status")
        if status in {"running", "failed"}:
            raise ControlError(
                f"resume refuses to overwrite the current {status} invocation at "
                f"sequence {expected['sequence']}"
            )
        if status != "complete":
            raise ControlError(f"resume invocation has unsupported status: {status!r}")

        arm = str(expected["arm"])
        filter_name = str(expected["filter_name"])
        child_out = child_output_path(
            args.out, int(expected["control_repeat"]), arm, filter_name
        )
        plan_path = child_out.with_suffix(child_out.suffix + ".plan.json")
        artifact = record.get("artifact")
        plan_record = record.get("plan_evidence")
        if not isinstance(artifact, Mapping) or not isinstance(plan_record, Mapping):
            raise ControlError(f"completed invocation lacks artifact evidence: {child_out}")
        if Path(str(artifact.get("path", ""))).resolve() != child_out.resolve():
            raise ControlError(f"completed invocation child path drifted: {child_out}")
        if Path(str(plan_record.get("path", ""))).resolve() != plan_path.resolve():
            raise ControlError(f"completed invocation plan path drifted: {plan_path}")
        if not child_out.is_file():
            raise ControlError(f"completed child artifact is missing: {child_out}")
        require_sha256(child_out, artifact.get("sha256"), "completed child artifact")
        child_rows = read_csv(child_out)
        if len(child_rows) != int(artifact.get("rows", -1)):
            raise ControlError(f"completed child artifact row count drifted: {child_out}")
        if not plan_path.is_file():
            raise ControlError(f"completed child plan evidence is missing: {plan_path}")
        require_sha256(plan_path, plan_record.get("sha256"), "completed child plan evidence")
        validate_generic_plan_artifact(plan_path)
        reused.append(
            {
                "schedule": dict(expected),
                "child_path": child_out,
                "plan_path": plan_path,
                "rows": child_rows,
            }
        )

    for expected in schedule[len(records) :]:
        child_out = child_output_path(
            args.out,
            int(expected["control_repeat"]),
            str(expected["arm"]),
            str(expected["filter_name"]),
        )
        plan_path = child_out.with_suffix(child_out.suffix + ".plan.json")
        if child_out.exists() or plan_path.exists():
            raise ControlError(
                f"unstarted invocation has an existing or incomplete child artifact: "
                f"{child_out}"
            )
    return manifest, reused


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ControlError(f"invalid boolean value {value!r}")


def nonnegative_number(row: Mapping[str, object], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ControlError(f"row has no numeric {field}: {row.get(field)!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ControlError(f"row has invalid {field}: {value!r}")
    return value


def nonnegative_integer(row: Mapping[str, object], field: str) -> int:
    value = nonnegative_number(row, field)
    if not value.is_integer():
        raise ControlError(f"row has non-integral {field}: {value!r}")
    return int(value)


def profile_times_close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=1e-9,
        abs_tol=PROFILE_TIME_ABS_TOLERANCE_MS,
    )


def validate_r33_profile_row(
    row: Mapping[str, object], arm_name: str = "measured"
) -> None:
    semantics = nonnegative_integer(row, "profile_semantics_version")
    if semantics < REQUIRED_PROFILE_SEMANTICS_VERSION:
        raise ControlError(
            f"{arm_name} row requires profile semantics >="
            f"{REQUIRED_PROFILE_SEMANTICS_VERSION}, observed={semantics}"
        )

    counts = {
        field: nonnegative_integer(row, field) for field in R33_PROFILE_COUNT_FIELDS
    }
    times = {
        field: nonnegative_number(row, field) for field in R33_PROFILE_TIME_FIELDS
    }
    for field in R33_PROFILE_SCOPE_FIELDS:
        if not str(row.get(field, "")).strip():
            raise ControlError(f"{arm_name} row is missing {field}")
    if parse_bool(row.get("hnsw_remaining_ms_is_residual", "")) is not True:
        raise ControlError(
            f"{arm_name} row does not identify hnsw_remaining_ms as a residual"
        )

    classified_calls = (
        counts["index_readbuffer_shared_read_calls"]
        + counts["index_readbuffer_shared_hit_calls"]
        + counts["index_readbuffer_unclassified_calls"]
    )
    if counts["index_readbuffer_calls"] != classified_calls:
        raise ControlError(
            f"{arm_name} row ReadBuffer call classification does not sum: "
            f"total={counts['index_readbuffer_calls']}, classified={classified_calls}"
        )
    classified_ms = (
        times["index_readbuffer_shared_read_ms"]
        + times["index_readbuffer_shared_hit_ms"]
        + times["index_readbuffer_unclassified_ms"]
    )
    if not profile_times_close(times["index_readbuffer_ms"], classified_ms):
        raise ControlError(
            f"{arm_name} row ReadBuffer timing classification does not sum within "
            f"{PROFILE_TIME_ABS_TOLERANCE_MS} ms: "
            f"total={times['index_readbuffer_ms']}, classified={classified_ms}"
        )

    distance_compute_count = nonnegative_integer(row, "distance_compute_count")
    if counts["distance_compute_timed_calls"] != distance_compute_count:
        raise ControlError(
            f"{arm_name} row timed distance calls differ from distance_compute_count: "
            f"timed={counts['distance_compute_timed_calls']}, "
            f"logical={distance_compute_count}"
        )
    index_page_loads = nonnegative_integer(row, "index_page_loads")
    if counts["index_readbuffer_calls"] != index_page_loads:
        raise ControlError(
            f"{arm_name} row ReadBuffer calls differ from index_page_loads: "
            f"calls={counts['index_readbuffer_calls']}, loads={index_page_loads}"
        )

    callback_ms = nonnegative_number(row, "hnsw_am_callback_ms")
    callback_components_ms = (
        times["index_readbuffer_ms"]
        + times["distance_compute_ms"]
        + times["hnsw_remaining_ms"]
    )
    if not profile_times_close(callback_ms, callback_components_ms):
        raise ControlError(
            f"{arm_name} row HNSW callback breakdown does not sum within "
            f"{PROFILE_TIME_ABS_TOLERANCE_MS} ms: "
            f"callback={callback_ms}, components={callback_components_ms}"
        )


def protocol_requires_r33_profile(protocol: Mapping[str, Any]) -> bool:
    name = str(protocol.get("name", ""))
    if name in LEGACY_PROTOCOL_NAMES:
        return False
    if name != PROTOCOL_NAME:
        raise ControlError(f"unsupported cache-isolation protocol for refresh: {name!r}")
    try:
        version = int(protocol["version"])
        required_semantics = int(
            protocol["profile_contract"]["required_profile_semantics_min"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ControlError("v5 protocol is missing its r33 profile contract") from exc
    if (
        version != PROTOCOL_VERSION
        or required_semantics < REQUIRED_PROFILE_SEMANTICS_VERSION
    ):
        raise ControlError("v5 protocol has an invalid r33 profile contract")
    return True


def load_runner() -> Any:
    module_name = (
        "experiments.hybrid_vector_db.scripts."
        "pgvector_design1_design2_design3_selectivity_benchmark"
    )
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != "experiments":
            raise
        return importlib.import_module(
            "pgvector_design1_design2_design3_selectivity_benchmark"
        )


def arm_specs(source_index: str, bfs_index: str) -> tuple[Arm, ...]:
    if source_index == bfs_index:
        raise ControlError("source and BFS indexes must be distinct")
    return (
        Arm("d1_source", "design1_bloom", "source", source_index),
        Arm("d1_bfs", "design1_bloom_bfs_layout", "bfs", bfs_index),
    )


def verify_two_arm_code_path(source_index: str, bfs_index: str) -> dict[str, Any]:
    """Prove that both measured arms use D1 and differ only in index layout."""
    runner = load_runner()
    args = argparse.Namespace(
        insertion_table=DEFAULT_TABLE,
        insertion_index=source_index,
        bfs_table=DEFAULT_TABLE,
        bfs_index=bfs_index,
    )
    expected = {
        "design1_bloom": source_index,
        "design1_bloom_bfs_layout": bfs_index,
    }
    observed = {
        mode: str(runner.mode_table_index(args, mode)[1]) for mode in expected
    }
    if observed != expected:
        raise ControlError(
            f"delegated runner mode/index path changed: expected={expected}, observed={observed}"
        )
    source_line = inspect.getsourcelines(runner.mode_table_index)[1]
    return {
        "runner": str(RUNNER_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "mode_table_index_first_line": source_line,
        "mode_to_index": observed,
        "physical_index_consumer_counts": {source_index: 1, bfs_index: 1},
        "source_to_bfs_consumer_ratio": 1.0,
        "cache_confound_present": False,
        "reason": "both arms are D1 safe_guided; only the physical HNSW relation changes",
    }


def reject_legacy_raw(path: Path | None) -> None:
    """Do not let an old three-arm artifact enter this control."""
    if path is not None:
        raise ControlError(
            "legacy raw artifacts are not accepted; rerun the two-arm safe_guided protocol"
        )


def rotating_schedule(repeats: int = REPEATS, seed: int = 20260721) -> list[dict[str, Any]]:
    if repeats <= 0:
        raise ControlError("repeat count must be positive")
    first = list(ARMS)
    random.Random(seed).shuffle(first)
    schedule: list[dict[str, Any]] = []
    for block in range(repeats):
        offset = block % len(first)
        order = first[offset:] + first[:offset]
        for position, arm in enumerate(order, start=1):
            schedule.append(
                {
                    "sequence": len(schedule),
                    "control_repeat": block,
                    "position": position,
                    "arm": arm,
                }
            )
    for block in range(repeats):
        block_arms = {
            row["arm"] for row in schedule if row["control_repeat"] == block
        }
        if block_arms != set(ARMS):
            raise ControlError(f"schedule block {block} is incomplete")
    return schedule


def paired_filter_schedule(
    filter_names: Sequence[str],
    repeats: int = REPEATS,
    seed: int = 20260721,
) -> list[dict[str, Any]]:
    """Place the two independently reset arms next to each other per filter."""
    arm_schedule = rotating_schedule(repeats, seed)
    schedule: list[dict[str, Any]] = []
    for block in range(repeats):
        filters = list(filter_names)
        random.Random(seed + block + 1).shuffle(filters)
        arms = [
            item for item in arm_schedule if int(item["control_repeat"]) == block
        ]
        for filter_position, filter_name in enumerate(filters, start=1):
            for arm_item in arms:
                schedule.append(
                    {
                        **arm_item,
                        "sequence": len(schedule),
                        "filter_position": filter_position,
                        "filter_name": filter_name,
                    }
                )
    return schedule


def mode_config(config: MatchedConfig) -> dict[str, object]:
    return {
        "ef_search": config.ef_search,
        "max_scan_tuples": config.max_scan_tuples,
        "scan_mem_multiplier": config.scan_mem_multiplier,
        "iterative_scan": config.iterative_scan,
        "guided_collect_target": 1,
        # safe_guided is candidate validation only.  The delegated runner has
        # a legacy traversal-prioritization default, so pin that compatibility
        # GUC off instead of inheriting it accidentally.
        "traversal_guided_prioritization": False,
    }


def measurement_queries(args: argparse.Namespace) -> int:
    return args.cold_block_queries if args.cache_regime == "cold_io" else args.queries


DISTINCT_COLD_QUERY_SLICE_POLICY = "distinct_contiguous_per_eviction_block"
LEGACY_COLD_QUERY_SLICE_POLICY = "repeated_slice_per_eviction_block"


def measurement_query_offset(
    args: argparse.Namespace, control_repeat: int
) -> int:
    if args.cache_regime != "cold_io":
        return args.query_offset
    return args.query_offset + control_repeat * measurement_queries(args)


def truth_measurement_queries(args: argparse.Namespace) -> int:
    if args.cache_regime != "cold_io":
        return args.queries
    return measurement_queries(args) * args.repeats


def audit_truth_for_args(args: argparse.Namespace) -> dict[str, Any]:
    return audit_exact_truth_manifest(
        args.truth_manifest,
        args.truth_csv,
        args.filters_csv,
        expected_table=args.table,
        expected_index=args.source_index,
        expected_query_table=args.query_table,
        expected_query_id_column=args.query_id_column,
        expected_query_vector_column=args.query_vector_column,
        expected_candidate_validity_predicate=args.candidate_validity_predicate,
        expected_self_excluded=args.expected_truth_self_excluded,
        query_offset=args.query_offset,
        queries=truth_measurement_queries(args),
        expected_filter_names=args.filter_names,
        expected_matched_manifest=(
            None
            if getattr(args, "matched_configs_csv", None) is not None
            else args.matched_recall_manifest
        ),
    )


def build_runner_command(
    args: argparse.Namespace,
    arm: Arm,
    child_out: Path,
    graph_proof_path: Path,
    config: MatchedConfig,
    control_repeat: int = 0,
) -> list[str]:
    config_json = json.dumps({arm.mode: mode_config(config)}, sort_keys=True)
    command = [
        str(args.python),
        str(RUNNER_PATH),
        "--out",
        str(child_out),
        "--filters-csv",
        str(args.filters_csv),
        "--truth-csv",
        str(args.truth_csv),
        "--insertion-table",
        args.table,
        "--insertion-index",
        args.source_index,
        "--bfs-table",
        args.table,
        "--bfs-index",
        args.bfs_index,
        "--candidate-validity-predicate",
        args.candidate_validity_predicate,
        "--modes",
        arm.mode,
        "--execution-order",
        "mode_major",
        "--mode-configs-json",
        config_json,
        "--queries",
        str(measurement_queries(args)),
        "--query-offset",
        str(measurement_query_offset(args, control_repeat)),
        "--repeats",
        "1",
        "--k",
        str(args.k),
        "--ef-search",
        str(config.ef_search),
        "--guided-collect-target",
        "1",
        "--guidance-filter-strategy",
        "safe_guided",
        "--iterative-scan",
        config.iterative_scan,
        "--max-scan-tuples",
        str(config.max_scan_tuples),
        "--scan-mem-multiplier",
        str(config.scan_mem_multiplier),
        "--d2-page-access",
        "off",
        "--d2-index-page-access",
        "off",
        "--d1-cache-mb",
        str(args.d1_cache_mb),
        "--d1-guidance-kind",
        args.d1_guidance_kind,
        "--guidance-selectivity-max-pct",
        "100",
        "--guidance-max-atoms",
        str(args.guidance_max_atoms),
        "--statement-timeout-ms",
        str(args.statement_timeout_ms),
        "--progress-queries",
        "0",
        "--expected-sqlens-build-id",
        args.expected_sqlens_build_id,
        "--expected-vector-so-sha256",
        args.expected_vector_so_sha256,
        "--backend-cpu-list",
        str(args.backend_cpu),
        "--fragment-tracking-prepared",
    ]
    command.append(
        "--expected-truth-self-excluded"
        if args.expected_truth_self_excluded
        else "--no-expected-truth-self-excluded"
    )
    if args.cache_regime == "warm_resident":
        command.append("--warmup-all-queries")
    else:
        command.extend(["--warmup-queries", "0"])
    if arm.index_role == "bfs":
        command.extend(["--d2-graph-proof-json", str(graph_proof_path)])
    if args.filter_names:
        command.extend(["--filter-names", config.filter_name])
    if args.query_table:
        command.extend(["--query-table", args.query_table])
    command.extend(["--query-id-column", args.query_id_column])
    command.extend(["--query-vector-column", args.query_vector_column])
    return command


def docker_command_plan(args: argparse.Namespace, arm: Arm, prewarm_blocks: int | str) -> list[str]:
    relation = arm.expected_index.replace("'", "''")
    common = [
        f"SELECT pg_prewarm('{item.replace(chr(39), chr(39) * 2)}'::regclass, 'read', 'main');"
        for item in args.prewarm_common_relation
    ]
    lifecycle = [
        f"docker update --cpuset-cpus={args.backend_cpu} {shlex.quote(args.container)}"
    ]
    if args.cache_regime == "cold_io":
        lifecycle.extend(
            [
                f"docker stop {shlex.quote(args.container)}",
                f"sudo -n python3 -c <relation-scoped-posix-fadvise-DONTNEED> {relation}",
                f"docker start {shlex.quote(args.container)}",
                *common,
            ]
        )
        return lifecycle
    if prewarm_blocks == "FULL_TARGET_INDEX_BLOCKS":
        index_sql = f"SELECT pg_prewarm('{relation}'::regclass, 'read', 'main');"
    else:
        index_sql = (
            f"SELECT pg_prewarm('{relation}'::regclass, 'read', 'main', 0, "
            f"{prewarm_blocks}-1);"
        )
    lifecycle.extend(
        [
            f"docker restart {shlex.quote(args.container)}",
            *common,
            index_sql,
        ]
    )
    return lifecycle


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    runner = load_runner()
    proof = load_graph_proof(
        args.d2_graph_proof_json,
        runner,
        args.source_index,
        args.bfs_index,
        args.expected_candidate_rows,
    )
    truth_evidence = audit_truth_for_args(args)
    configs = load_matched_configs(args, args.filters_csv, args.truth_csv)
    arms = {arm.name: arm for arm in arm_specs(args.source_index, args.bfs_index)}
    proof_path = args.out.with_suffix(args.out.suffix + ".d2_graph_proof.json")
    prewarm_blocks: int | str = args.prewarm_index_blocks or "FULL_TARGET_INDEX_BLOCKS"
    schedule = paired_filter_schedule(args.filter_names, args.repeats, args.schedule_seed)
    invocations = []
    for item in schedule:
        arm = arms[str(item["arm"])]
        filter_name = str(item["filter_name"])
        config = configs[filter_name]
        child_out = child_output_path(
            args.out, int(item["control_repeat"]), arm.name, filter_name
        )
        command = build_runner_command(
            args,
            arm,
            child_out,
            proof_path,
            config,
            int(item["control_repeat"]),
        )
        invocations.append({
            **item,
            "matched_config": config.as_dict(),
            "mode": arm.mode,
            "target_index": arm.expected_index,
            "lifecycle": docker_command_plan(args, arm, prewarm_blocks),
            "runner_argv": command,
            "runner_shell": shlex.join(command),
        })
    payload = {
        "status": "dry_run",
        "protocol": protocol_spec(args),
        "two_arm_code_path": verify_two_arm_code_path(args.source_index, args.bfs_index),
        "exact_truth_audit": truth_evidence,
        "matched_config_source": matched_config_source_evidence(args, configs),
        "delegated_d2_graph_proof": {
            "source_index": proof["source_index"],
            "clone_index": proof["clone_index"],
            "stable_fingerprint_sha256": proof["stable_fingerprint_sha256"],
            "same_heap": proof["comparison"]["same_heap"],
            "logical_equal": proof["comparison"]["logical_equal"],
            "physical_equal": proof["comparison"]["physical_equal"],
        },
        "schedule": schedule,
        "invocations": invocations,
    }
    legacy_evidence = legacy_matched_recall_evidence(args, configs)
    if legacy_evidence is not None:
        payload["matched_recall_audit"] = legacy_evidence
    return payload


def protocol_spec(args: argparse.Namespace) -> dict[str, Any]:
    warm = args.cache_regime == "warm_resident"
    protocol = {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "cache_regime": args.cache_regime,
        "profile_contract": {
            "required_profile_semantics_min": REQUIRED_PROFILE_SEMANTICS_VERSION,
            "fail_closed": True,
            "raw_breakdown_validation": (
                "nonnegative counts/times; classified ReadBuffer calls and time "
                "sum to totals; timed distance calls equal logical distance count; "
                "ReadBuffer calls equal index-page loads; HNSW callback equals "
                "ReadBuffer plus distance plus residual"
            ),
            "time_sum_absolute_tolerance_ms": PROFILE_TIME_ABS_TOLERANCE_MS,
            "legacy_completed_artifact_policy": (
                "v3/v4 artifacts may refresh only their originally measured fields; "
                "missing r33 fields are never synthesized or accepted as a v5 run"
            ),
        },
        "arms": [asdict(arm) for arm in arm_specs(args.source_index, args.bfs_index)],
        "measurement": {
            "query_split": (
                f"q{args.query_offset}.."
                f"q{args.query_offset + measurement_queries(args) - 1}"
            ),
            "queries": measurement_queries(args),
            "repeats": args.repeats,
            "schedule_seed": args.schedule_seed,
            "pairing_unit": "filter_name,query_no,control_repeat",
            "child_repeats_per_invocation": 1,
            "warmup": (
                (
                    f"one unmeasured q{args.query_offset}.."
                    f"q{args.query_offset + args.queries - 1} pass before each "
                    f"measured q{args.queries}/r1 block"
                )
                if warm
                else f"none; each q{measurement_queries(args)} block begins after target-relation page-cache eviction"
            ),
            "configuration": "one independently audited D1 safe_guided matched-recall config per filter",
            "guidance_max_atoms": args.guidance_max_atoms,
            "truth_self_excluded": args.expected_truth_self_excluded,
        },
        "cache": {
            "protocol": (
                "restart_full_target_index_read_prewarm"
                if warm
                else "stop_relation_scoped_posix_fadvise_dontneed_start_no_target_prewarm"
            ),
            "postgres_restart_before_every_arm_invocation": True,
            "postgres_shared_buffers_cleared": True,
            "os_page_cache_dropped": False,
            "target_relation_os_pages_evicted": not warm,
            "eviction_scope": None if warm else "target HNSW main-fork segment files only",
            "target_index_prewarm": (
                (
                    "full target relation by default; same explicit prefix when "
                    "--prewarm-index-blocks is supplied; first block 0, mode=read"
                )
                if warm
                else "none"
            ),
            "prewarm_index_blocks": (
                args.prewarm_index_blocks or "full_target_index" if warm else 0
            ),
            "common_relations": list(args.prewarm_common_relation),
            "prewarm_excluded_from_latency": True,
            "cold_block_interpretation": (
                None
                if warm
                else (
                    "q1/r1 independent cold-start measurement after target-relation eviction"
                    if measurement_queries(args) == 1
                    else (
                        f"q{measurement_queries(args)}/r1 cold-start block; first-query and "
                        "prefix metrics are reported separately because later queries "
                        "progressively warm the index"
                    )
                )
            ),
        },
        "cpu": {
            "container_cpuset": str(args.backend_cpu),
            "single_cpu": True,
            "same_cpu_for_every_arm": True,
            "child_backend_must_report_exact_match": True,
        },
        "d2_isolation": {
            "d1_source_mode": "design1_bloom",
            "d1_bfs_mode": "design1_bloom_bfs_layout",
            "stock_arm_present": False,
            "same_d1_search_config": True,
            "page_access": "off",
            "index_page_access": "off",
            "same_heap_same_logical_graph_required": True,
            "physical_layout_must_differ": True,
            "d1_result_ids_must_match": True,
            "live_graph_proof_policy": args.live_graph_proof_policy,
            "matched_config_index_policy": args.matched_config_index_policy,
            "mean_qualified_config_allowed": args.allow_mean_qualified_matched_config,
            "configuration_role": (
                "current-index matched-recall proof"
                if args.matched_config_index_policy == "exact"
                else "borrowed tuning seed; observed recall is reported and no matched-recall claim is made"
            ),
        },
        "required_per_query_metrics": list(REQUIRED_PROFILE_FIELDS),
    }
    if not warm:
        block_queries = measurement_queries(args)
        total_queries = truth_measurement_queries(args)
        protocol["measurement"].update(
            {
                "cold_protocol_semantics_version": COLD_PROTOCOL_SEMANTICS_VERSION,
                "query_split": (
                    f"q{args.query_offset}.."
                    f"q{args.query_offset + total_queries - 1}"
                ),
                "queries_per_block": block_queries,
                "total_distinct_queries": total_queries,
                "query_slice_policy": DISTINCT_COLD_QUERY_SLICE_POLICY,
                "block_query_slices": [
                    {
                        "control_repeat": repeat,
                        "query_offset": measurement_query_offset(args, repeat),
                        "queries": block_queries,
                    }
                    for repeat in range(args.repeats)
                ],
            }
        )
        protocol["cache"]["cold_block_interpretation"] = (
            f"each q{block_queries}/r1 block is an independent cold-start "
            "measurement after target-relation eviction and uses a distinct "
            "contiguous query slice; both source/BFS arms use the same slice; "
            "this preserves cold v4 independent-block semantics"
        )
    return protocol


def child_output_path(base: Path, control_repeat: int, arm: str, filter_name: str) -> Path:
    safe_filter = re.sub(r"[^A-Za-z0-9_.-]+", "_", filter_name)
    return base.with_name(f"{base.stem}.block{control_repeat}.{arm}.{safe_filter}.csv")


def validate_plan_evidence(
    path: Path,
    arm: Arm,
    filter_name: str,
    index_identity: Mapping[str, Any],
    table: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise ControlError(f"missing delegated EXPLAIN evidence: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlError(f"invalid delegated EXPLAIN evidence {path}: {exc}") from exc
    checks = payload.get("checks")
    if payload.get("status") != "complete" or not isinstance(checks, list) or not checks:
        raise ControlError(f"delegated EXPLAIN evidence is incomplete: {path}")
    if any(not isinstance(check, dict) or check.get("passed") is not True for check in checks):
        raise ControlError(f"delegated EXPLAIN evidence has a failed check: {path}")
    selected = [check for check in checks if check.get("filter_name") == filter_name]
    if len(selected) != 1:
        raise ControlError(f"delegated EXPLAIN evidence does not cover exactly {filter_name}")
    check = selected[0]
    if check.get("mode") != arm.mode:
        raise ControlError(f"delegated EXPLAIN evidence mode mismatch for {filter_name}")
    if check.get("expected_table_identity") != table:
        raise ControlError(f"delegated EXPLAIN evidence table mismatch for {filter_name}")
    if check.get("expected_index_oid") != index_identity["oid"]:
        raise ControlError(f"delegated EXPLAIN evidence OID mismatch for {filter_name}")
    if check.get("expected_index_identity") != arm.expected_index:
        raise ControlError(f"delegated EXPLAIN evidence index mismatch for {filter_name}")
    if check.get("catalog_index_predicate_matches") is not True:
        raise ControlError(f"delegated EXPLAIN evidence predicate mismatch for {filter_name}")
    if check.get("preferred_index_current_setting") != arm.expected_index:
        raise ControlError(f"delegated EXPLAIN preferred-index proof mismatch for {filter_name}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": payload["status"],
        "checks": len(checks),
        "filter_name": filter_name,
        "expected_index": arm.expected_index,
        "expected_index_oid": index_identity["oid"],
    }


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ControlError(f"command failed ({completed.returncode}): {shlex.join(command)}\n{detail}")
    return completed


def wait_for_postgres(timeout_s: float) -> None:
    runner = load_runner()
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            connection = runner.psycopg.connect(
                runner.pg_config_from_env().conninfo, autocommit=True
            )
            connection.close()
            return
        except Exception as exc:  # noqa: BLE001 - readiness polling
            last_error = str(exc)
            time.sleep(0.25)
    raise ControlError(f"PostgreSQL did not become ready: {last_error}")


def inspect_cpuset(container: str) -> str:
    completed = run_command(
        ["docker", "inspect", "-f", "{{.HostConfig.CpusetCpus}}", container]
    )
    return completed.stdout.strip()


def set_container_cpu(container: str, cpu: int) -> None:
    run_command(["docker", "update", f"--cpuset-cpus={cpu}", container])
    observed = inspect_cpuset(container)
    if observed != str(cpu):
        raise ControlError(
            f"container CPU gate failed: requested={cpu}, observed={observed!r}"
        )


def restart_postgres(args: argparse.Namespace) -> dict[str, Any]:
    before = run_command(
        ["docker", "inspect", "-f", "{{.RestartCount}}", args.container]
    ).stdout.strip()
    completed = run_command(["docker", "restart", args.container])
    wait_for_postgres(args.readiness_timeout_s)
    after = run_command(
        ["docker", "inspect", "-f", "{{.RestartCount}}", args.container]
    ).stdout.strip()
    return {
        "started_at": utc_now(),
        "container": completed.stdout.strip(),
        "restart_count_before": int(before),
        "restart_count_after": int(after),
        "cpuset": inspect_cpuset(args.container),
        "ready": True,
    }


def relation_main_fork_host_path(
    args: argparse.Namespace, arm: Arm
) -> tuple[str, str]:
    runner = load_runner()
    connection = runner.psycopg.connect(
        runner.pg_config_from_env().conninfo, autocommit=True
    )
    try:
        cur = connection.cursor()
        cur.execute("SELECT pg_relation_filepath(%s::regclass)", (arm.expected_index,))
        row = cur.fetchone()
        relative = str(row[0]) if row and row[0] else ""
    finally:
        connection.close()
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise ControlError(
            f"invalid target-index relation filepath for cold eviction: {relative!r}"
        )
    mount = run_command(
        [
            "docker",
            "inspect",
            "-f",
            '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Source}}{{end}}{{end}}',
            args.container,
        ]
    ).stdout.strip()
    if not mount or not Path(mount).is_absolute():
        raise ControlError(f"cannot resolve PostgreSQL host data mount: {mount!r}")
    return str(Path(mount) / relative), relative


def evict_relation_main_fork(host_base: str) -> dict[str, Any]:
    helper = (
        "import glob,json,os,re,sys;"
        "b=sys.argv[1];"
        "p=[b]+sorted(x for x in glob.glob(b+'.*') if re.fullmatch(re.escape(b)+r'\\.[0-9]+',x));"
        "r=[];"
        "[(lambda f:(os.posix_fadvise(f,0,0,os.POSIX_FADV_DONTNEED),r.append({'path':x,'bytes':os.fstat(f).st_size}),os.close(f)))(os.open(x,os.O_RDONLY|os.O_CLOEXEC)) for x in p if os.path.isfile(x)];"
        "print(json.dumps({'files':r,'file_count':len(r),'bytes':sum(x['bytes'] for x in r)}))"
    )
    completed = run_command(["sudo", "-n", "python3", "-c", helper, host_base])
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ControlError("relation-scoped cache eviction returned invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or int(payload.get("file_count", 0)) <= 0
        or int(payload.get("bytes", 0)) <= 0
    ):
        raise ControlError(f"relation-scoped cache eviction found no index files: {payload}")
    return payload


def validate_eviction_coverage(
    payload: Mapping[str, Any], expected_size_bytes: int
) -> dict[str, Any]:
    observed = int(payload.get("bytes", -1))
    if observed != expected_size_bytes:
        raise ControlError(
            "relation-scoped cache eviction did not cover the complete target index: "
            f"expected={expected_size_bytes}, observed={observed}"
        )
    return {**payload, "expected_bytes": expected_size_bytes, "coverage_ratio": 1.0}


def cold_restart_postgres(
    args: argparse.Namespace, arm: Arm, expected_size_bytes: int
) -> dict[str, Any]:
    if not args.i_understand_relation_cache_eviction:
        raise ControlError(
            "cold-I/O execution requires --i-understand-relation-cache-eviction"
        )
    started_at = utc_now()
    host_base, relative_path = relation_main_fork_host_path(args, arm)
    run_command(["docker", "stop", args.container])
    try:
        eviction = validate_eviction_coverage(
            evict_relation_main_fork(host_base), expected_size_bytes
        )
    except BaseException:
        run_command(["docker", "start", args.container])
        wait_for_postgres(args.readiness_timeout_s)
        raise
    completed = run_command(["docker", "start", args.container])
    wait_for_postgres(args.readiness_timeout_s)
    return {
        "started_at": started_at,
        "completed_at": utc_now(),
        "container": completed.stdout.strip(),
        "cpuset": inspect_cpuset(args.container),
        "ready": True,
        "eviction_method": "posix_fadvise(POSIX_FADV_DONTNEED)",
        "eviction_scope": "target HNSW main-fork segment files only",
        "target_relation": arm.expected_index,
        "target_relation_filepath": relative_path,
        "target_relation_host_base": host_base,
        "eviction": eviction,
        "target_index_prewarmed": False,
    }


def reset_cache_for_arm(
    args: argparse.Namespace, arm: Arm, expected_size_bytes: int
) -> dict[str, Any]:
    if args.cache_regime == "cold_io":
        return cold_restart_postgres(args, arm, expected_size_bytes)
    return restart_postgres(args)


def catalog_index_identities(cur: Any, source_index: str, bfs_index: str) -> dict[str, Any]:
    cur.execute(
        "SELECT c.oid::bigint, n.nspname || '.' || c.relname, c.relfilenode::bigint, "
        "i.indrelid::bigint, pg_relation_size(c.oid)::bigint, "
        "current_setting('block_size')::bigint, am.amname, "
        "i.indisvalid, i.indisready, i.indislive, pg_get_expr(i.indpred, i.indrelid) "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_index i ON i.indexrelid=c.oid JOIN pg_am am ON am.oid=c.relam "
        "WHERE c.oid IN (%s::regclass, %s::regclass) ORDER BY c.oid",
        (source_index, bfs_index),
    )
    records = {}
    for row in cur.fetchall():
        size = int(row[4])
        block_size = int(row[5])
        name = str(row[1])
        records[name] = {
            "oid": int(row[0]),
            "name": name,
            "relfilenode": int(row[2]),
            "heap_oid": int(row[3]),
            "size_bytes": size,
            "block_size": block_size,
            "blocks": (size + block_size - 1) // block_size,
            "access_method": str(row[6]),
            "indisvalid": bool(row[7]),
            "indisready": bool(row[8]),
            "indislive": bool(row[9]),
            "predicate": row[10],
        }
    if set(records) != {source_index, bfs_index}:
        raise ControlError(f"index catalog gate returned {sorted(records)}")
    source = records[source_index]
    bfs = records[bfs_index]
    if source["oid"] == bfs["oid"] or source["relfilenode"] == bfs["relfilenode"]:
        raise ControlError("source and BFS index identities are not physically distinct")
    if source["heap_oid"] != bfs["heap_oid"]:
        raise ControlError("source and BFS indexes do not share one heap")
    for record in records.values():
        if record["access_method"] != "hnsw" or not all(
            record[field] for field in ("indisvalid", "indisready", "indislive")
        ):
            raise ControlError(f"index is not a live HNSW relation: {record}")
    return records


def load_graph_proof(
    path: Path,
    runner: Any,
    source: str,
    bfs: str,
    expected_heap_tids: int | None = None,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    candidates = (
        value,
        value.get("delegated_d2_graph_proof") if isinstance(value, dict) else None,
        value.get("d2_graph_proof_final") if isinstance(value, dict) else None,
        value.get("run_spec", {}).get("d2_graph_proof")
        if isinstance(value, dict) and isinstance(value.get("run_spec"), dict)
        else None,
    )
    proof = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, dict) and "comparison" in candidate
        ),
        None,
    )
    if proof is None:
        raise ControlError(f"no D2 graph proof found in {path}")
    try:
        validated = runner.validate_d2_graph_proof(
            proof, source, bfs, expected_heap_tids=expected_heap_tids
        )
        if validated.get("proof_contract") != (
            "sqlens_same_heap_same_logical_graph_physical_layout_v3"
        ):
            raise ControlError("D2 cache-isolation control requires a v3 graph proof")
        return validated
    except Exception as exc:  # noqa: BLE001 - convert delegated gate failures
        raise ControlError(f"delegated D2 graph proof failed: {exc}") from exc


def validate_live_graph_proof(
    delegated: Mapping[str, Any], live: Mapping[str, Any]
) -> dict[str, Any]:
    expected = str(delegated.get("stable_fingerprint_sha256", ""))
    observed = str(live.get("stable_fingerprint_sha256", ""))
    if not expected or not observed or observed != expected:
        raise ControlError(
            "D2 graph proof drifted: the live logical graph, physical layout, "
            "tuple coverage, entry point, or relation identity differs from the "
            "delegated artifact"
        )
    return dict(live)


def validate_graph_relation_identities(
    proof: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
    source_index: str,
    bfs_index: str,
) -> None:
    relations = proof.get("relations")
    if not isinstance(relations, Mapping):
        raise ControlError("D2 graph proof has no stable relation identities")
    for role, name in (("source", source_index), ("clone", bfs_index)):
        relation = relations.get(role)
        if not isinstance(relation, Mapping):
            raise ControlError(f"D2 graph proof has no {role} relation identity")
        requested = next(
            (identity for key, identity in identities.items() if key == name), None
        )
        if requested is None:
            requested = identities.get(name)
        if requested is None:
            # Callers using non-default index names pass the relation through the
            # proof's name; the name check below still prevents cross-binding.
            requested = next(
                (
                    identity
                    for key, identity in identities.items()
                    if str(relation.get("name")) == key
                ),
                None,
            )
        if requested is None:
            raise ControlError(f"D2 graph proof relation {role} is not a requested index")
        for field in ("oid", "relfilenode", "heap_oid"):
            if int(relation.get(field, 0)) != int(requested[field]):
                raise ControlError(
                    f"D2 graph proof {role} {field} does not match live catalog identity"
                )


def prepare_database_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    runner = load_runner()
    connection = runner.psycopg.connect(runner.pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = connection.cursor()
        runner.require_exact_sqlens_identity(
            cur, args.expected_sqlens_build_id, args.expected_vector_so_sha256
        )
        runner.ensure_functions(cur)
        runner.ensure_tracking(cur, args.table)
        proof = load_graph_proof(
            args.d2_graph_proof_json,
            runner,
            args.source_index,
            args.bfs_index,
            args.expected_candidate_rows,
        )
        identities = catalog_index_identities(cur, args.source_index, args.bfs_index)
        if args.live_graph_proof_policy == "full":
            live = runner.require_d2_graph_proof(
                cur,
                args.source_index,
                args.bfs_index,
                expected_heap_tids=args.expected_candidate_rows,
            )
            live = validate_live_graph_proof(proof, live)
        else:
            live = {
                **proof,
                "runtime_validation": {
                    "checked_at": utc_now(),
                    "policy": "delegated_immutable",
                    "full_graph_rescan": False,
                    "binary_identity_exact": True,
                    "relation_identity_and_size_checked": True,
                    "dedicated_read_only_experiment_required": True,
                },
            }
        validate_graph_relation_identities(
            live, identities, args.source_index, args.bfs_index
        )
        return live, identities
    finally:
        connection.close()


def require_dedicated_server() -> dict[str, Any]:
    """Refuse to restart a PostgreSQL server with any other client session."""
    runner = load_runner()
    connection = runner.psycopg.connect(runner.pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = connection.cursor()
        cur.execute(
            "SELECT pid::bigint, application_name, state, backend_start::text "
            "FROM pg_stat_activity WHERE datname=current_database() "
            "AND pid<>pg_backend_pid() AND backend_type='client backend' ORDER BY pid"
        )
        sessions = [
            {
                "pid": int(row[0]),
                "application_name": str(row[1] or ""),
                "state": str(row[2] or ""),
                "backend_start": str(row[3] or ""),
            }
            for row in cur.fetchall()
        ]
        if sessions:
            raise ControlError(
                "cache-isolation control requires a dedicated PostgreSQL server; "
                f"found client sessions: {sessions}"
            )
        return {"checked_at": utc_now(), "other_client_sessions": [], "passed": True}
    finally:
        connection.close()


def prewarm_for_arm(
    args: argparse.Namespace,
    arm: Arm,
    identities: Mapping[str, Mapping[str, Any]],
    blocks: int,
) -> dict[str, Any]:
    runner = load_runner()
    connection = runner.psycopg.connect(runner.pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = connection.cursor()
        current = catalog_index_identities(cur, args.source_index, args.bfs_index)
        if current != identities:
            raise ControlError("index identity or size changed before prewarm")
        common_records = []
        for relation in args.prewarm_common_relation:
            cur.execute(
                "SELECT pg_prewarm(%s::regclass, 'read', 'main')::bigint",
                (relation,),
            )
            warmed = int(cur.fetchone()[0])
            if warmed <= 0:
                raise ControlError(f"common-relation prewarm returned {warmed} for {relation}")
            common_records.append({"relation": relation, "blocks": warmed})
        if args.cache_regime == "cold_io":
            return {
                "completed_at": utc_now(),
                "backend": "dedicated common-relation prewarm connection, closed before measured backend",
                "mode": "read",
                "fork": "main",
                "target_index": arm.expected_index,
                "target_index_blocks": 0,
                "target_index_prewarmed": False,
                "common_relations": common_records,
            }
        if blocks <= 0 or blocks > int(identities[arm.expected_index]["blocks"]):
            raise ControlError(f"invalid equal prewarm block count {blocks} for {arm.expected_index}")
        cur.execute(
            "SELECT pg_prewarm(%s::regclass, 'read', 'main', 0, %s)::bigint",
            (arm.expected_index, blocks - 1),
        )
        warmed = int(cur.fetchone()[0])
        if warmed != blocks:
            raise ControlError(
                f"target-index prewarm mismatch for {arm.name}: expected={blocks}, got={warmed}"
            )
        return {
            "completed_at": utc_now(),
            "backend": "dedicated prewarm connection, closed before measured backend",
            "mode": "read",
            "fork": "main",
            "target_index": arm.expected_index,
            "first_block": 0,
            "last_block": blocks - 1,
            "blocks": warmed,
            "coverage_ratio": warmed / int(identities[arm.expected_index]["blocks"]),
            "target_index_prewarmed": True,
            "common_relations": common_records,
        }
    finally:
        connection.close()


def validate_child_rows(
    rows: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
    arm: Arm,
    control_repeat: int,
    sequence: int,
    position: int,
    index_identity: Mapping[str, Any],
    config: MatchedConfig,
    graph_fingerprint: str,
) -> list[dict[str, Any]]:
    queries = measurement_queries(args)
    query_offset = measurement_query_offset(args, control_repeat)
    if len(rows) != queries:
        raise ControlError(f"{arm.name} block {control_repeat} has {len(rows)} rows, need {queries}")
    query_nos = {int(row.get("query_no", -1)) for row in rows}
    if query_nos != set(range(query_offset, query_offset + queries)):
        raise ControlError(
            f"{arm.name} block {control_repeat} has incomplete query coverage: "
            f"expected q{query_offset}..q{query_offset + queries - 1}"
        )
    backend_pids = {int(row.get("backend_pid", 0)) for row in rows}
    if len(backend_pids) != 1 or next(iter(backend_pids)) <= 0:
        raise ControlError(f"{arm.name} block {control_repeat} did not use one measured backend")
    output = []
    for row in rows:
        if len({str(item.get("filter_name", "")) for item in rows}) != 1 or row.get("filter_name") != config.filter_name:
            raise ControlError(f"{arm.name} block {control_repeat} has unexpected filter coverage")
        if row.get("mode") != arm.mode or row.get("index") != arm.expected_index:
            raise ControlError(f"{arm.name} executed an unexpected mode/index")
        if str(row.get("error", "")):
            raise ControlError(f"{arm.name} measured row failed: {row.get('error_detail')}")
        if int(row.get("repeat", -1)) != 0:
            raise ControlError(f"{arm.name} child repeat is not zero")
        if str(row.get("backend_cpu_requested", "")) != str(args.backend_cpu):
            raise ControlError(f"{arm.name} did not request the fixed CPU")
        if str(row.get("backend_cpu_observed", "")) != str(args.backend_cpu):
            raise ControlError(f"{arm.name} backend affinity differs from the fixed CPU")
        if parse_bool(row.get("backend_cpu_exact_match", "")) is not True:
            raise ControlError(f"{arm.name} backend CPU exact-match gate failed")
        if row.get("sqlens_build_id") != args.expected_sqlens_build_id:
            raise ControlError(f"{arm.name} SQLens build ID changed")
        if row.get("vector_so_sha256") != args.expected_vector_so_sha256:
            raise ControlError(f"{arm.name} vector.so SHA256 changed")
        if row.get("guidance_filter_strategy") != "safe_guided":
            raise ControlError(f"{arm.name} is not using safe_guided")
        for field, expected in (
            ("ef_search", config.ef_search),
            ("max_scan_tuples", config.max_scan_tuples),
            ("iterative_scan", config.iterative_scan),
        ):
            if str(row.get(field)) != str(expected):
                raise ControlError(
                    f"{arm.name} filter {config.filter_name} used unexpected {field}: "
                    f"expected={expected!r}, observed={row.get(field)!r}"
                )
        for field in REQUIRED_PROFILE_FIELDS:
            if field in {
                "index_page_profile_scope",
                "index_page_trace_statistics_scope",
                "index_page_trace_sample_scope",
                *R33_PROFILE_SCOPE_FIELDS,
            }:
                if not str(row.get(field, "")):
                    raise ControlError(f"{arm.name} row is missing {field}")
            elif field == "index_page_trace_sample":
                try:
                    trace_sample = json.loads(str(row.get(field, "")))
                except json.JSONDecodeError as exc:
                    raise ControlError(f"{arm.name} row has invalid index trace JSON") from exc
                if not isinstance(trace_sample, list) or any(
                    isinstance(block, bool) or not isinstance(block, int) or block < 0
                    for block in trace_sample
                ):
                    raise ControlError(f"{arm.name} row has an invalid index trace sample")
            elif field == "index_page_distinct_pages_exact":
                parse_bool(row.get(field, ""))
            elif field == "index_page_distinct_pages":
                distinct_exact = parse_bool(row.get("index_page_distinct_pages_exact", ""))
                distinct_pages = float(row.get(field, "nan"))
                if distinct_exact is True:
                    if not math.isfinite(distinct_pages) or distinct_pages < 0:
                        raise ControlError(
                            f"{arm.name} row has invalid exact {field}: {distinct_pages}"
                        )
                elif distinct_pages != -1:
                    raise ControlError(
                        f"{arm.name} row must use -1 when {field} exceeds the profiler cap"
                    )
            elif field == "heap_blks_are_exact_heap_io":
                if parse_bool(row.get(field, "")) is not False:
                    raise ControlError(
                        f"{arm.name} row incorrectly claims exact heap I/O accounting"
                    )
            elif field == "index_page_trace_sample_truncated":
                parse_bool(row.get(field, ""))
            elif field == "hnsw_remaining_ms_is_residual":
                parse_bool(row.get(field, ""))
            else:
                nonnegative_number(row, field)
        validate_r33_profile_row(row, arm.name)
        loads = int(float(row["index_page_loads"]))
        transitions = int(float(row["index_page_transition_count"]))
        transition_bands = [
            int(float(row["index_page_same_block_transitions"])),
            int(float(row["index_page_within_1_page_transitions"])),
            int(float(row["index_page_within_4_pages_transitions"])),
            int(float(row["index_page_within_16_pages_transitions"])),
            transitions,
        ]
        if transitions != max(loads - 1, 0):
            raise ControlError(
                f"{arm.name} actual ReadBuffer transition count does not match one scan"
            )
        if transition_bands != sorted(transition_bands):
            raise ControlError(f"{arm.name} actual ReadBuffer transition bands disagree")
        trace_limit = int(float(row["index_page_trace_sample_limit"]))
        trace_count = int(float(row["index_page_trace_sample_count"]))
        trace_sample = json.loads(str(row["index_page_trace_sample"]))
        if trace_limit != 64 or trace_count != min(loads, trace_limit):
            raise ControlError(f"{arm.name} actual ReadBuffer trace bound is invalid")
        if len(trace_sample) != trace_count:
            raise ControlError(f"{arm.name} actual ReadBuffer trace sample is incomplete")
        if parse_bool(row["index_page_trace_sample_truncated"]) is not (
            loads > trace_limit
        ):
            raise ControlError(f"{arm.name} actual ReadBuffer trace truncation is invalid")
        if str(row.get("preferred_index_current_setting", "")) != arm.expected_index:
            raise ControlError(
                f"{arm.name} preferred-index proof does not name {arm.expected_index}"
            )
        if arm.name.startswith("d1_"):
            for field in (
                "guidance_enabled",
                "guidance_scan_verified",
                "guidance_binding_verified",
            ):
                if parse_bool(row.get(field, "")) is not True:
                    raise ControlError(
                        f"{arm.name} did not execute active D1 guidance: {field}"
                    )
            if row.get("final_path") != "validation_only":
                raise ControlError(
                    f"{arm.name} did not execute the safe-guided validation path"
                )
        enriched = dict(row)
        enriched.update(
            {
                "control_arm": arm.name,
                "control_repeat": control_repeat,
                "control_sequence": sequence,
                "control_position": position,
                "control_pair_key": (
                    f"{row['filter_name']}|{int(row['query_no'])}|{control_repeat}"
                ),
                "control_index_role": arm.index_role,
                "control_index_oid": index_identity["oid"],
                "control_index_relfilenode": index_identity["relfilenode"],
                "control_index_size_bytes": index_identity["size_bytes"],
                "control_index_blocks": index_identity["blocks"],
                "control_cache_regime": args.cache_regime,
                "control_graph_semantic_fingerprint": graph_fingerprint,
                "control_matched_config": config.as_dict(),
            }
        )
        output.append(enriched)
    return output


def validate_paired_rows(
    rows: Sequence[Mapping[str, Any]],
    queries: int = QUERIES,
    repeats: int = REPEATS,
    *,
    require_r33_profile: bool = True,
) -> None:
    if require_r33_profile:
        for row in rows:
            validate_r33_profile_row(row, str(row.get("control_arm", "measured")))
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["control_pair_key"]), []).append(row)
    expected_pairs = len({str(row["filter_name"]) for row in rows}) * queries * repeats
    if len(groups) != expected_pairs:
        raise ControlError(f"paired coverage mismatch: got={len(groups)}, expected={expected_pairs}")
    for key, group in groups.items():
        by_arm = {str(row["control_arm"]): row for row in group}
        if set(by_arm) != set(ARMS) or len(group) != len(ARMS):
            raise ControlError(f"pair {key} is missing or duplicates an arm")
        source = by_arm["d1_source"]
        bfs = by_arm["d1_bfs"]
        if source["ids"] != bfs["ids"]:
            raise ControlError(f"D1 source/BFS result IDs differ for pair {key}")
        if not math.isclose(
            float(source["recall"]), float(bfs["recall"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ControlError(f"D1 source/BFS recall differs for pair {key}")
        config_fields = (
            "ef_search",
            "max_scan_tuples",
            "scan_mem_multiplier",
            "iterative_scan",
            "guided_collect_target",
            "guidance_filter_strategy",
        )
        if any(source[field] != bfs[field] for field in config_fields):
            raise ControlError(f"D1 source/BFS search configuration differs for pair {key}")
        semantic_fields = (
            PAIRED_D1_SEMANTIC_FIELDS
            if require_r33_profile
            else tuple(
                field
                for field in PAIRED_D1_SEMANTIC_FIELDS
                if field
                not in {
                    "index_readbuffer_calls",
                    "distance_compute_timed_calls",
                    *R33_PROFILE_SCOPE_FIELDS,
                }
            )
        )
        missing_semantics = [
            field
            for field in semantic_fields
            if field not in source or field not in bfs
        ]
        if missing_semantics:
            raise ControlError(
                f"D1 source/BFS pair {key} is missing semantic fields: {missing_semantics}"
            )
        changed_semantics = [
            field
            for field in semantic_fields
            if source[field] != bfs[field]
        ]
        if changed_semantics:
            raise ControlError(
                f"D1 source/BFS execution semantics differ for pair {key}: "
                f"{changed_semantics}"
            )
        if source.get("control_graph_semantic_fingerprint") != bfs.get(
            "control_graph_semantic_fingerprint"
        ):
            raise ControlError(f"D1 source/BFS graph semantic fingerprint differs for pair {key}")


def validate_measurement_query_slices(
    rows: Sequence[Mapping[str, Any]],
    *,
    cache_regime: str,
    queries: int,
    repeats: int,
    query_offset: int,
    cold_query_slice_policy: str | None = None,
) -> None:
    policy = cold_query_slice_policy
    if cache_regime == "cold_io":
        policy = policy or DISTINCT_COLD_QUERY_SLICE_POLICY
        if policy not in {
            DISTINCT_COLD_QUERY_SLICE_POLICY,
            LEGACY_COLD_QUERY_SLICE_POLICY,
        }:
            raise ControlError(f"unsupported cold query-slice policy: {policy}")
    for filter_name in {str(row["filter_name"]) for row in rows}:
        for repeat in range(repeats):
            expected_offset = query_offset
            if (
                cache_regime == "cold_io"
                and policy == DISTINCT_COLD_QUERY_SLICE_POLICY
            ):
                expected_offset += repeat * queries
            expected = set(range(expected_offset, expected_offset + queries))
            for arm in ARMS:
                observed = {
                    int(row["query_no"])
                    for row in rows
                    if str(row["filter_name"]) == filter_name
                    and int(row["control_repeat"]) == repeat
                    and str(row["control_arm"]) == arm
                }
                if observed != expected:
                    raise ControlError(
                        "measurement query-slice coverage mismatch for "
                        f"{filter_name}/{arm}/repeat={repeat}: "
                        f"expected q{expected_offset}..q{expected_offset + queries - 1}"
                    )


def recorded_cold_query_slice_policy(protocol: Mapping[str, Any]) -> str | None:
    if protocol.get("cache_regime") != "cold_io":
        return None
    measurement = protocol.get("measurement")
    if not isinstance(measurement, Mapping):
        raise ControlError("completed manifest has no measurement protocol")
    policy = measurement.get("query_slice_policy")
    if policy is None:
        if protocol.get("name") == "sqlens-d2-cache-isolation-v3":
            return LEGACY_COLD_QUERY_SLICE_POLICY
        raise ControlError(
            "cold completed manifest does not identify its query-slice policy"
        )
    if policy != DISTINCT_COLD_QUERY_SLICE_POLICY:
        raise ControlError(
            f"cold completed manifest uses unsupported query-slice policy: {policy}"
        )
    return str(policy)


def bootstrap_mean_ci(values: Sequence[float], seed: int, samples: int = 5000) -> tuple[float, float]:
    if not values:
        raise ControlError("cannot bootstrap an empty sample")
    randomizer = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(values[randomizer.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]


def summarize(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    queries: int = QUERIES,
    repeats: int = REPEATS,
    query_offset: int = QUERY_OFFSET,
    cold_query_slice_policy: str | None = None,
    include_r33_profile: bool = True,
) -> list[dict[str, Any]]:
    filters = sorted({str(row["filter_name"]) for row in rows})
    summaries = []
    for filter_name in filters:
        selected = [row for row in rows if row["filter_name"] == filter_name]
        by_arm = {
            arm: [row for row in selected if row["control_arm"] == arm] for arm in ARMS
        }
        cache_regime = str(selected[0]["control_cache_regime"])
        query_deltas = []
        if cache_regime == "cold_io":
            policy = cold_query_slice_policy or DISTINCT_COLD_QUERY_SLICE_POLICY
            if policy not in {
                DISTINCT_COLD_QUERY_SLICE_POLICY,
                LEGACY_COLD_QUERY_SLICE_POLICY,
            }:
                raise ControlError(f"unsupported cold query-slice policy: {policy}")
            paired_cluster_unit = "cold_eviction_block"
            for repeat in range(repeats):
                block_offset = (
                    query_offset + repeat * queries
                    if policy == DISTINCT_COLD_QUERY_SLICE_POLICY
                    else query_offset
                )
                block_deltas = []
                for query_no in range(block_offset, block_offset + queries):
                    source = [
                        float(row["end_to_end_ms"])
                        for row in by_arm["d1_source"]
                        if int(row["control_repeat"]) == repeat
                        and int(row["query_no"]) == query_no
                    ]
                    bfs = [
                        float(row["end_to_end_ms"])
                        for row in by_arm["d1_bfs"]
                        if int(row["control_repeat"]) == repeat
                        and int(row["query_no"]) == query_no
                    ]
                    if len(source) != 1 or len(bfs) != 1:
                        raise ControlError(
                            "cold-block pairing incomplete for "
                            f"{filter_name}/{repeat}/q{query_no}"
                        )
                    block_deltas.append(bfs[0] - source[0])
                query_deltas.append(statistics.fmean(block_deltas))
        else:
            paired_cluster_unit = "query"
            for query_no in range(query_offset, query_offset + queries):
                source = [
                    float(row["end_to_end_ms"])
                    for row in by_arm["d1_source"]
                    if int(row["query_no"]) == query_no
                ]
                bfs = [
                    float(row["end_to_end_ms"])
                    for row in by_arm["d1_bfs"]
                    if int(row["query_no"]) == query_no
                ]
                if len(source) != repeats or len(bfs) != repeats:
                    raise ControlError(f"query-cluster pairing incomplete for {filter_name}/{query_no}")
                query_deltas.append(statistics.fmean(bfs) - statistics.fmean(source))
        ci_low, ci_high = bootstrap_mean_ci(
            query_deltas,
            seed + int.from_bytes(hashlib.sha256(filter_name.encode()).digest()[:4], "big"),
        )
        item: dict[str, Any] = {
            "filter_name": filter_name,
            "selectivity": selected[0]["selectivity"],
            "cache_regime": cache_regime,
            "queries": queries,
            "repeats": repeats,
            "paired_rows_per_arm": len(by_arm["d1_source"]),
            "paired_cluster_unit": paired_cluster_unit,
            "paired_clusters": len(query_deltas),
            "d1_bfs_minus_source_query_cluster_mean_ms": statistics.fmean(query_deltas),
            "d1_bfs_minus_source_ci95_low_ms": ci_low,
            "d1_bfs_minus_source_ci95_high_ms": ci_high,
        }
        for arm in ARMS:
            arm_rows = by_arm[arm]
            for field in (
                "end_to_end_ms",
                "activation_ms",
                "query_latency_ms",
                "vector_search_ms",
                "recall",
                "idx_blks_hit",
                "idx_blks_read",
                "index_page_runs",
                "index_page_loads",
                "index_page_transition_count",
                "index_page_same_block_transitions",
                "index_page_within_1_page_transitions",
                "index_page_within_4_pages_transitions",
                "index_page_within_16_pages_transitions",
                "index_page_backward_transitions",
                "index_page_total_abs_block_delta",
                "index_page_max_abs_block_delta",
                "index_page_prefetches",
                "page_access_prefetches",
                "heap_blks_hit",
                "heap_blks_read",
                "heap_tid_page_runs",
                *(
                    (*R33_PROFILE_COUNT_FIELDS, *R33_PROFILE_TIME_FIELDS)
                    if include_r33_profile
                    else ()
                ),
            ):
                item[f"{arm}_{field}_mean"] = statistics.fmean(
                    float(row[field]) for row in arm_rows
                )
            if include_r33_profile:
                residual_flags = [
                    parse_bool(row["hnsw_remaining_ms_is_residual"])
                    for row in arm_rows
                ]
                if not all(residual_flags):
                    raise ControlError(
                        f"{arm} summary contains a non-residual hnsw_remaining_ms row"
                    )
                item[f"{arm}_hnsw_remaining_ms_is_residual_rate"] = (
                    statistics.fmean(float(flag) for flag in residual_flags)
                )
                for field in R33_PROFILE_SCOPE_FIELDS:
                    scopes = {str(row[field]) for row in arm_rows}
                    if len(scopes) != 1:
                        raise ControlError(
                            f"{arm} summary has mixed {field} values: {sorted(scopes)}"
                        )
                    item[f"{arm}_{field}"] = next(iter(scopes))
            exact_distinct_rows = [
                row
                for row in arm_rows
                if parse_bool(row["index_page_distinct_pages_exact"]) is True
            ]
            item[f"{arm}_index_page_distinct_pages_exact_rows"] = len(
                exact_distinct_rows
            )
            item[f"{arm}_index_page_distinct_pages_exact_rate"] = (
                len(exact_distinct_rows) / len(arm_rows)
            )
            item[f"{arm}_index_page_distinct_pages_mean"] = (
                statistics.fmean(
                    float(row["index_page_distinct_pages"])
                    for row in exact_distinct_rows
                )
                if exact_distinct_rows
                else None
            )
            transitions = sum(
                float(row["index_page_transition_count"]) for row in arm_rows
            )
            for label, field in (
                ("same_block", "index_page_same_block_transitions"),
                ("within_1_page", "index_page_within_1_page_transitions"),
                ("within_4_pages", "index_page_within_4_pages_transitions"),
                ("within_16_pages", "index_page_within_16_pages_transitions"),
                ("backward", "index_page_backward_transitions"),
            ):
                item[f"{arm}_index_page_{label}_transition_rate"] = (
                    sum(float(row[field]) for row in arm_rows) / transitions
                    if transitions > 0
                    else 0.0
                )
            item[f"{arm}_index_page_mean_abs_block_delta"] = (
                sum(
                    float(row["index_page_total_abs_block_delta"])
                    for row in arm_rows
                )
                / transitions
                if transitions > 0
                else 0.0
            )
            for prefix in (prefix for prefix in (1, 5, 10) if prefix <= queries):
                if (
                    cache_regime == "cold_io"
                    and (cold_query_slice_policy or DISTINCT_COLD_QUERY_SLICE_POLICY)
                    == DISTINCT_COLD_QUERY_SLICE_POLICY
                ):
                    prefix_rows = [
                        row
                        for row in arm_rows
                        if int(row["query_no"])
                        < query_offset + int(row["control_repeat"]) * queries + prefix
                    ]
                else:
                    prefix_rows = [
                        row
                        for row in arm_rows
                        if int(row["query_no"]) < query_offset + prefix
                    ]
                item[f"{arm}_cold_prefix_q{prefix}_end_to_end_ms_mean"] = (
                    statistics.fmean(float(row["end_to_end_ms"]) for row in prefix_rows)
                )
        source_ms = float(item["d1_source_end_to_end_ms_mean"])
        bfs_ms = float(item["d1_bfs_end_to_end_ms_mean"])
        item["d1_bfs_speedup_over_source"] = source_ms / bfs_ms
        item["d1_bfs_query_speedup_over_source"] = (
            float(item["d1_source_query_latency_ms_mean"])
            / float(item["d1_bfs_query_latency_ms_mean"])
        )
        item["d1_bfs_index_page_run_reduction"] = 1.0 - (
            float(item["d1_bfs_index_page_runs_mean"])
            / float(item["d1_source_index_page_runs_mean"])
        )
        source_distinct = item["d1_source_index_page_distinct_pages_mean"]
        bfs_distinct = item["d1_bfs_index_page_distinct_pages_mean"]
        distinct_fully_exact = all(
            float(item[f"{arm}_index_page_distinct_pages_exact_rate"]) == 1.0
            for arm in ARMS
        )
        item["d1_bfs_distinct_index_page_reduction"] = (
            1.0 - float(bfs_distinct) / float(source_distinct)
            if distinct_fully_exact
            and source_distinct not in (None, 0)
            and bfs_distinct is not None
            else None
        )
        if include_r33_profile:
            for label, field in (
                ("index_readbuffer_total", "index_readbuffer_ms"),
                (
                    "index_readbuffer_shared_read",
                    "index_readbuffer_shared_read_ms",
                ),
                (
                    "index_readbuffer_shared_hit",
                    "index_readbuffer_shared_hit_ms",
                ),
            ):
                source_value = float(item[f"d1_source_{field}_mean"])
                bfs_value = float(item[f"d1_bfs_{field}_mean"])
                item[f"d1_bfs_{label}_time_reduction_ms"] = source_value - bfs_value
                item[f"d1_bfs_{label}_time_reduction_fraction"] = (
                    1.0 - bfs_value / source_value if source_value > 0 else None
                )
            item["d1_bfs_distance_compute_timed_calls_delta"] = (
                float(item["d1_bfs_distance_compute_timed_calls_mean"])
                - float(item["d1_source_distance_compute_timed_calls_mean"])
            )
            item["d1_bfs_distance_compute_ms_delta"] = (
                float(item["d1_bfs_distance_compute_ms_mean"])
                - float(item["d1_source_distance_compute_ms_mean"])
            )
            item["d1_bfs_hnsw_remaining_ms_delta"] = (
                float(item["d1_bfs_hnsw_remaining_ms_mean"])
                - float(item["d1_source_hnsw_remaining_ms_mean"])
            )
        summaries.append(item)
    return summaries


def execute(args: argparse.Namespace) -> None:
    if not args.i_understand_container_restarts:
        raise ControlError("execution requires --i-understand-container-restarts")
    if args.cache_regime == "cold_io" and not args.i_understand_relation_cache_eviction:
        raise ControlError(
            "cold-I/O execution requires --i-understand-relation-cache-eviction"
        )
    runner = load_runner()
    reject_legacy_raw(None)
    exact_truth = audit_truth_for_args(args)
    matched_configs = load_matched_configs(args, args.filters_csv, args.truth_csv)
    two_arm_code_path = verify_two_arm_code_path(args.source_index, args.bfs_index)
    original_cpuset = inspect_cpuset(args.container)
    started = utc_now()
    run_uuid = str(uuid.uuid4())
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    proof_path = args.out.with_suffix(args.out.suffix + ".d2_graph_proof.json")
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    schedule = paired_filter_schedule(args.filter_names, args.repeats, args.schedule_seed)
    protocol = protocol_spec(args)
    matched_source = matched_config_source_evidence(args, matched_configs)
    runtime_input_identity = resume_runtime_input_identity(args)
    if args.resume:
        manifest, reused_children = validate_resume_manifest(
            args,
            manifest_path,
            schedule,
            protocol,
            exact_truth,
            matched_source,
            current_argv=sys.argv,
            current_two_arm_code_path=two_arm_code_path,
        )
        invocation_records = manifest["invocations"]
        combined: list[dict[str, Any]] = []
    else:
        invocation_records = []
        combined = []
        manifest = {
            "status": "running",
            "run_uuid": run_uuid,
            "started_at": started,
            "protocol": protocol,
            "argv": sys.argv,
            "controller_sha256": sha256_file(Path(__file__)),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "two_arm_code_path": two_arm_code_path,
            "exact_truth_audit": exact_truth,
            "matched_config_source": matched_source,
            "runtime_input_identity": runtime_input_identity,
            "original_container_cpuset": original_cpuset,
            "schedule": schedule,
            "invocations": invocation_records,
        }
    legacy_evidence = legacy_matched_recall_evidence(args, matched_configs)
    if legacy_evidence is not None and not args.resume:
        manifest["matched_recall_audit"] = legacy_evidence
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        manifest["status"] = "running"
        manifest["resume_started_at"] = utc_now()
        manifest["resume_controller_sha256"] = sha256_file(Path(__file__))
        atomic_write_json(manifest_path, manifest)
    else:
        atomic_write_json(manifest_path, manifest)
    try:
        manifest["dedicated_server_gate"] = require_dedicated_server()
        atomic_write_json(manifest_path, manifest)
        set_container_cpu(args.container, args.backend_cpu)
        restart_postgres(args)
        proof, identities = prepare_database_contract(args)
        if args.resume and proof_path.exists():
            existing_proof = load_graph_proof(
                proof_path,
                runner,
                args.source_index,
                args.bfs_index,
                args.expected_candidate_rows,
            )
            if existing_proof.get("stable_fingerprint_sha256") != proof.get(
                "stable_fingerprint_sha256"
            ):
                raise ControlError("resume D2 graph-proof artifact drifted")
        else:
            atomic_write_json(proof_path, proof)
        prewarm_blocks_by_arm = {
            arm.name: (
                args.prewarm_index_blocks
                or int(identities[arm.expected_index]["blocks"])
                if args.cache_regime == "warm_resident"
                else 0
            )
            for arm in arm_specs(args.source_index, args.bfs_index)
        }
        if args.cache_regime == "warm_resident" and any(
            blocks <= 0 for blocks in prewarm_blocks_by_arm.values()
        ):
            raise ControlError("target-index prewarm block count is not positive")
        if args.resume:
            if manifest.get("index_identities_start") != identities:
                raise ControlError("resume live index identities differ from the run")
            if manifest.get("d2_graph_proof", {}).get("stable_fingerprint_sha256") != proof.get(
                "stable_fingerprint_sha256"
            ):
                raise ControlError("resume live graph fingerprint differs from the run")
            if manifest.get("prewarm_blocks_by_arm") != prewarm_blocks_by_arm:
                raise ControlError("resume prewarm schedule differs from the run")
        else:
            manifest.update(
                {
                    "d2_graph_proof": proof,
                    "index_identities_start": identities,
                    "prewarm_blocks_by_arm": prewarm_blocks_by_arm,
                }
            )
        atomic_write_json(manifest_path, manifest)
        arms = {arm.name: arm for arm in arm_specs(args.source_index, args.bfs_index)}
        if args.resume:
            for reused in reused_children:
                item = reused["schedule"]
                arm = arms[str(item["arm"])]
                filter_name = str(item["filter_name"])
                record = next(
                    record
                    for record in invocation_records
                    if int(record["sequence"]) == int(item["sequence"])
                )
                expected_command = build_runner_command(
                    args,
                    arm,
                    reused["child_path"],
                    proof_path,
                    matched_configs[filter_name],
                    int(item["control_repeat"]),
                )
                if record.get("runner_argv") != expected_command:
                    raise ControlError(
                        f"resume delegated runner argv differs for sequence {item['sequence']}"
                    )
                if record.get("arm_spec") != asdict(arm):
                    raise ControlError(
                        f"resume arm identity differs for sequence {item['sequence']}"
                    )
                if record.get("matched_config") != matched_configs[filter_name].as_dict():
                    raise ControlError(
                        f"resume matched configuration differs for sequence {item['sequence']}"
                    )
                plan_evidence = validate_plan_evidence(
                    reused["plan_path"],
                    arm,
                    filter_name,
                    identities[arm.expected_index],
                    args.table,
                )
                combined.extend(
                    validate_child_rows(
                        reused["rows"],
                        args,
                        arm,
                        int(item["control_repeat"]),
                        int(item["sequence"]),
                        int(item["position"]),
                        identities[arm.expected_index],
                        matched_configs[filter_name],
                        proof["stable_fingerprint_sha256"],
                    )
                )
                record["resume_reused_at"] = utc_now()
                record["resume_revalidated_plan_evidence"] = plan_evidence
            atomic_write_json(manifest_path, manifest)
        for item in schedule:
            if args.resume and int(item["sequence"]) < len(reused_children):
                continue
            arm = arms[str(item["arm"])]
            filter_name = str(item["filter_name"])
            config = matched_configs[filter_name]
            record: dict[str, Any] = {
                **item,
                "matched_config": config.as_dict(),
                "arm_spec": asdict(arm),
                "status": "running",
            }
            invocation_records.append(record)
            atomic_write_json(manifest_path, manifest)
            set_container_cpu(args.container, args.backend_cpu)
            record["cache_reset"] = reset_cache_for_arm(
                args, arm, int(identities[arm.expected_index]["size_bytes"])
            )
            record["prewarm"] = prewarm_for_arm(
                args, arm, identities, int(prewarm_blocks_by_arm[arm.name])
            )
            child_out = child_output_path(
                args.out, int(item["control_repeat"]), arm.name, filter_name
            )
            if child_out.exists():
                raise ControlError(f"refusing to overwrite child artifact: {child_out}")
            command = build_runner_command(
                args,
                arm,
                child_out,
                proof_path,
                config,
                int(item["control_repeat"]),
            )
            record["runner_argv"] = command
            record["runner_shell"] = shlex.join(command)
            completed = run_command(command)
            if not child_out.is_file():
                raise ControlError(f"delegated runner did not create {child_out}")
            child_rows = read_csv(child_out)
            plan_evidence = validate_plan_evidence(
                child_out.with_suffix(child_out.suffix + ".plan.json"),
                arm,
                filter_name,
                identities[arm.expected_index],
                args.table,
            )
            enriched = validate_child_rows(
                child_rows,
                args,
                arm,
                int(item["control_repeat"]),
                int(item["sequence"]),
                int(item["position"]),
                identities[arm.expected_index],
                config,
                proof["stable_fingerprint_sha256"],
            )
            combined.extend(enriched)
            record.update(
                {
                    "status": "complete",
                    "completed_at": utc_now(),
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "plan_evidence": plan_evidence,
                    "artifact": {
                        "path": str(child_out),
                        "rows": len(child_rows),
                        "sha256": sha256_file(child_out),
                    },
                    "measured_backend_pid": int(child_rows[0]["backend_pid"]),
                }
            )
            atomic_write_json(manifest_path, manifest)
        if len(invocation_records) != len(schedule) or any(
            record.get("status") != "complete" for record in invocation_records
        ):
            raise ControlError(
                "cannot finalize cache-isolation control: the raw artifact does not "
                f"cover all {len(schedule)} invocations"
            )
        validate_paired_rows(combined, measurement_queries(args), args.repeats)
        validate_measurement_query_slices(
            combined,
            cache_regime=args.cache_regime,
            queries=measurement_queries(args),
            repeats=args.repeats,
            query_offset=args.query_offset,
        )
        summary = summarize(
            combined,
            args.schedule_seed,
            measurement_queries(args),
            args.repeats,
            args.query_offset,
        )
        atomic_write_csv(args.out, combined)
        atomic_write_csv(summary_path, summary)
        restart_postgres(args)
        proof_final, identities_final = prepare_database_contract(args)
        if identities_final != identities:
            raise ControlError("index identity or size changed during the control")
        if proof_final["stable_fingerprint_sha256"] != proof["stable_fingerprint_sha256"]:
            raise ControlError("D2 graph or physical layout changed during the control")
        validate_graph_relation_identities(
            proof_final, identities_final, args.source_index, args.bfs_index
        )
        manifest.update(
            {
                "status": "complete",
                "artifact_valid": True,
                "completed_at": utc_now(),
                "index_identities_final": identities_final,
                "d2_graph_proof_final": proof_final,
                "outputs": {
                    "raw": {
                        "path": str(args.out),
                        "rows": len(combined),
                        "sha256": sha256_file(args.out),
                    },
                    "summary": {
                        "path": str(summary_path),
                        "rows": len(summary),
                        "sha256": sha256_file(summary_path),
                    },
                    "d2_graph_proof": {
                        "path": str(proof_path),
                        "sha256": sha256_file(proof_path),
                    },
                },
                "paired_gate_passed": True,
            }
        )
        atomic_write_json(manifest_path, manifest)
        print(json.dumps({"manifest": str(manifest_path), "summary": summary}, indent=2))
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "artifact_valid": False,
                "completed_at": utc_now(),
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        if original_cpuset != str(args.backend_cpu):
            run_command(
                ["docker", "update", f"--cpuset-cpus={original_cpuset}", args.container]
            )


def finalize_completed_invocations(args: argparse.Namespace) -> None:
    """Re-audit immutable child artifacts after a post-processing-only failure."""
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    proof_path = args.out.with_suffix(args.out.suffix + ".d2_graph_proof.json")
    if not manifest_path.is_file() or not proof_path.is_file():
        raise ControlError("finalization requires the existing manifest and graph proof")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_error = manifest.get("error")
    if manifest.get("status") != "failed" or not isinstance(prior_error, Mapping):
        raise ControlError("finalization accepts only a failed, fully measured run")
    if prior_error.get("type") != "ControlError" or "missing semantic fields" not in str(
        prior_error.get("message", "")
    ):
        raise ControlError(f"failure is not an approved post-processing schema error: {prior_error}")

    schedule = paired_filter_schedule(args.filter_names, args.repeats, args.schedule_seed)
    records = manifest.get("invocations")
    if not isinstance(records, list) or len(records) != len(schedule):
        raise ControlError("existing run does not contain the complete invocation schedule")
    for expected, record in zip(schedule, records, strict=True):
        if not isinstance(record, Mapping) or record.get("status") != "complete":
            raise ControlError("existing run contains an incomplete invocation")
        for field in ("sequence", "control_repeat", "position", "arm", "filter_name"):
            if record.get(field) != expected.get(field):
                raise ControlError(f"existing invocation schedule drifted at {field}")

    runner = load_runner()
    if manifest.get("runner_sha256") != sha256_file(RUNNER_PATH):
        raise ControlError("delegated runner changed since the measured run")
    exact_truth = audit_truth_for_args(args)
    matched_configs = load_matched_configs(args, args.filters_csv, args.truth_csv)
    expected_source = matched_config_source_evidence(args, matched_configs)
    recorded_source = manifest.get("matched_config_source")
    if recorded_source is None and getattr(args, "matched_configs_csv", None) is None:
        recorded_source = {
            "kind": "legacy_matched_recall_manifest",
            "manifest": {
                "path": manifest.get("matched_recall_audit", {}).get("path"),
                "sha256": manifest.get("matched_recall_audit", {}).get("sha256"),
            },
            "mode": "design1_bloom",
            "configs": manifest.get("matched_recall_audit", {}).get("configs"),
        }
    if recorded_source != expected_source:
        raise ControlError("matched configuration source differs from the measured run")
    original_cpuset = inspect_cpuset(args.container)
    finalizer_started = utc_now()
    try:
        require_dedicated_server()
        set_container_cpu(args.container, args.backend_cpu)
        restart_postgres(args)
        proof, identities = prepare_database_contract(args)
        recorded_identities = manifest.get("index_identities_start")
        if identities != recorded_identities:
            raise ControlError("live index identity or size differs from the measured run")
        if proof.get("stable_fingerprint_sha256") != manifest.get(
            "d2_graph_proof", {}
        ).get("stable_fingerprint_sha256"):
            raise ControlError("live graph fingerprint differs from the measured run")

        arms = {arm.name: arm for arm in arm_specs(args.source_index, args.bfs_index)}
        combined: list[dict[str, Any]] = []
        for record in records:
            arm = arms[str(record["arm"])]
            filter_name = str(record["filter_name"])
            artifact = record.get("artifact")
            if not isinstance(artifact, Mapping):
                raise ControlError("completed invocation has no artifact evidence")
            child_out = Path(str(artifact.get("path", "")))
            if not child_out.is_file() or sha256_file(child_out) != artifact.get("sha256"):
                raise ControlError(f"child artifact hash mismatch: {child_out}")
            child_rows = read_csv(child_out)
            if len(child_rows) != int(artifact.get("rows", -1)):
                raise ControlError(f"child artifact row count mismatch: {child_out}")
            plan_record = record.get("plan_evidence")
            if not isinstance(plan_record, Mapping):
                raise ControlError(f"child artifact has no plan evidence: {child_out}")
            plan_path = Path(str(plan_record.get("path", "")))
            if not plan_path.is_file() or sha256_file(plan_path) != plan_record.get("sha256"):
                raise ControlError(f"plan evidence hash mismatch: {plan_path}")
            validate_plan_evidence(
                plan_path,
                arm,
                filter_name,
                identities[arm.expected_index],
                args.table,
            )
            combined.extend(
                validate_child_rows(
                    child_rows,
                    args,
                    arm,
                    int(record["control_repeat"]),
                    int(record["sequence"]),
                    int(record["position"]),
                    identities[arm.expected_index],
                    matched_configs[filter_name],
                    str(proof["stable_fingerprint_sha256"]),
                )
            )

        validate_paired_rows(combined, measurement_queries(args), args.repeats)
        validate_measurement_query_slices(
            combined,
            cache_regime=args.cache_regime,
            queries=measurement_queries(args),
            repeats=args.repeats,
            query_offset=args.query_offset,
        )
        summary = summarize(
            combined,
            args.schedule_seed,
            measurement_queries(args),
            args.repeats,
            args.query_offset,
        )
        atomic_write_csv(args.out, combined)
        atomic_write_csv(summary_path, summary)
        previous_failure = {
            "status": "failed",
            "completed_at": manifest.get("completed_at"),
            "error": prior_error,
        }
        manifest.pop("error", None)
        manifest.update(
            {
                "status": "complete",
                "artifact_valid": True,
                "completed_at": utc_now(),
                "failure_history": [
                    *(manifest.get("failure_history") or []),
                    previous_failure,
                ],
                "finalization": {
                    "started_at": finalizer_started,
                    "completed_at": utc_now(),
                    "finalizer_sha256": sha256_file(Path(__file__)),
                    "reason": "re-audit immutable completed children after correcting legacy guidance_matches to traversal_guidance_matches",
                    "measurement_commands_rerun": 0,
                    "completed_invocations_reused": len(records),
                    "exact_truth_audit": exact_truth,
                },
                "index_identities_final": identities,
                "d2_graph_proof_final": proof,
                "outputs": {
                    "raw": {
                        "path": str(args.out),
                        "rows": len(combined),
                        "sha256": sha256_file(args.out),
                    },
                    "summary": {
                        "path": str(summary_path),
                        "rows": len(summary),
                        "sha256": sha256_file(summary_path),
                    },
                    "d2_graph_proof": {
                        "path": str(proof_path),
                        "sha256": sha256_file(proof_path),
                    },
                },
                "paired_gate_passed": True,
            }
        )
        atomic_write_json(manifest_path, manifest)
        print(json.dumps({"manifest": str(manifest_path), "summary": summary}, indent=2))
    except BaseException as exc:
        manifest["finalization_error"] = {
            "at": utc_now(),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "finalizer_sha256": sha256_file(Path(__file__)),
        }
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        if original_cpuset != str(args.backend_cpu):
            run_command(
                ["docker", "update", f"--cpuset-cpus={original_cpuset}", args.container]
            )


def refresh_completed_summary(args: argparse.Namespace) -> None:
    """Recompute a completed run's summary without touching measurements."""
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    if not manifest_path.is_file() or not args.out.is_file():
        raise ControlError("summary refresh requires an existing raw artifact and manifest")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("artifact_valid") is not True
        or manifest.get("paired_gate_passed") is not True
    ):
        raise ControlError("summary refresh accepts only a completed, paired-gate artifact")

    outputs = manifest.get("outputs")
    raw_evidence = outputs.get("raw") if isinstance(outputs, Mapping) else None
    if not isinstance(raw_evidence, Mapping):
        raise ControlError("completed manifest has no raw-output evidence")
    if Path(str(raw_evidence.get("path", ""))).resolve() != args.out.resolve():
        raise ControlError("manifest raw-output path does not match --out")
    if sha256_file(args.out) != raw_evidence.get("sha256"):
        raise ControlError("raw artifact hash differs from the completed manifest")

    protocol = manifest.get("protocol")
    measurement = protocol.get("measurement") if isinstance(protocol, Mapping) else None
    if not isinstance(measurement, Mapping):
        raise ControlError("completed manifest has no measurement protocol")
    queries = int(measurement.get("queries", -1))
    repeats = int(measurement.get("repeats", -1))
    if queries != measurement_queries(args) or repeats != args.repeats:
        raise ControlError("refresh arguments differ from the measured query/repeat protocol")
    if protocol.get("cache_regime") != args.cache_regime:
        raise ControlError("refresh cache regime differs from the measured protocol")
    require_r33_profile = protocol_requires_r33_profile(protocol)
    cold_query_slice_policy = recorded_cold_query_slice_policy(protocol)
    current_protocol = protocol_spec(args)
    if cold_query_slice_policy == DISTINCT_COLD_QUERY_SLICE_POLICY:
        current_measurement = current_protocol["measurement"]
        for field in (
            "query_split",
            "queries_per_block",
            "total_distinct_queries",
            "block_query_slices",
        ):
            if measurement.get(field) != current_measurement.get(field):
                raise ControlError(
                    f"refresh {field} differs from the measured cold query slices"
                )

    recorded_argv = manifest.get("argv")
    if not isinstance(recorded_argv, list):
        raise ControlError("completed manifest has no original argv")
    if "--schedule-seed" in recorded_argv:
        position = recorded_argv.index("--schedule-seed")
        try:
            recorded_schedule_seed = int(recorded_argv[position + 1])
        except (IndexError, TypeError, ValueError) as exc:
            raise ControlError("completed manifest has an invalid schedule seed") from exc
    else:
        recorded_schedule_seed = SCHEDULE_SEED
    if args.schedule_seed != recorded_schedule_seed:
        raise ControlError("refresh schedule seed differs from the measured run")

    recorded_arms = protocol.get("arms")
    expected_arms = {
        arm.name: arm.expected_index
        for arm in arm_specs(args.source_index, args.bfs_index)
    }
    if not isinstance(recorded_arms, list) or {
        str(arm.get("name")): str(arm.get("expected_index"))
        for arm in recorded_arms
        if isinstance(arm, Mapping)
    } != expected_arms:
        raise ControlError("refresh source/BFS arms differ from the measured run")

    rows = read_csv(args.out)
    if len(rows) != int(raw_evidence.get("rows", -1)):
        raise ControlError("raw artifact row count differs from the completed manifest")
    if {str(row["filter_name"]) for row in rows} != set(args.filter_names):
        raise ControlError("refresh filter set differs from the measured run")
    validate_paired_rows(
        rows,
        queries,
        repeats,
        require_r33_profile=require_r33_profile,
    )
    validate_measurement_query_slices(
        rows,
        cache_regime=args.cache_regime,
        queries=queries,
        repeats=repeats,
        query_offset=args.query_offset,
        cold_query_slice_policy=cold_query_slice_policy,
    )
    summary = summarize(
        rows,
        args.schedule_seed,
        queries,
        repeats,
        args.query_offset,
        cold_query_slice_policy,
        include_r33_profile=require_r33_profile,
    )

    old_summary = outputs.get("summary")
    old_summary_sha = (
        old_summary.get("sha256") if isinstance(old_summary, Mapping) else None
    )
    atomic_write_csv(summary_path, summary)
    new_summary_sha = sha256_file(summary_path)

    if cold_query_slice_policy != LEGACY_COLD_QUERY_SLICE_POLICY:
        protocol["cache"]["cold_block_interpretation"] = current_protocol["cache"][
            "cold_block_interpretation"
        ]
    protocol["measurement"]["schedule_seed"] = args.schedule_seed
    revision = {
        "at": utc_now(),
        "reason": (
            "recompute summary with query-cluster warm CI or independent-eviction "
            "cold CI; correct dynamic cold-block description; preserve the measured "
            + (
                "r33 internal timing breakdown"
                if require_r33_profile
                else "legacy field set without synthesizing r33 metrics"
            )
        ),
        "revision_controller_sha256": sha256_file(Path(__file__)),
        "measurement_commands_rerun": 0,
        "raw_sha256_unchanged": raw_evidence["sha256"],
        "old_summary_sha256": old_summary_sha,
        "new_summary_sha256": new_summary_sha,
        "r33_profile_fields_present": require_r33_profile,
        "r33_profile_fields_synthesized": False,
    }
    manifest["summary_revisions"] = [
        *(manifest.get("summary_revisions") or []),
        revision,
    ]
    outputs["summary"] = {
        "path": str(summary_path),
        "rows": len(summary),
        "sha256": new_summary_sha,
    }
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "summary": str(summary_path),
                "revision": revision,
            },
            indent=2,
            sort_keys=True,
        )
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def sha256_arg(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise argparse.ArgumentTypeError("expected a 64-character SHA256")
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fail-closed D1 safe_guided source-vs-BFS cache-isolation control."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument(
        "--truth-provenance-manifest",
        "--truth-manifest",
        dest="truth_manifest",
        type=Path,
        default=DEFAULT_TRUTH_MANIFEST,
        help=(
            "Audited exact-truth manifest or an external-dataset launch manifest; "
            "--truth-manifest remains a compatibility alias."
        ),
    )
    parser.add_argument(
        "--matched-recall-manifest",
        type=Path,
        help=(
            "Legacy independent audited safe_guided matched-recall manifest. Required "
            "unless --matched-configs-csv is supplied."
        ),
    )
    parser.add_argument(
        "--matched-configs-csv",
        type=Path,
        help="Current-build per-filter matched configuration CSV.",
    )
    parser.add_argument(
        "--matched-configs-manifest",
        type=Path,
        help="Fail-closed provenance manifest required with --matched-configs-csv.",
    )
    parser.add_argument("--filter-names", nargs="+", required=True)
    parser.add_argument("--d2-graph-proof-json", type=Path, required=True)
    parser.add_argument(
        "--live-graph-proof-policy",
        choices=("full", "delegated_immutable"),
        default="full",
    )
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--source-index", default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--bfs-index", default=DEFAULT_BFS_INDEX)
    parser.add_argument("--query-table")
    parser.add_argument("--query-id-column", default="id")
    parser.add_argument("--query-vector-column", default="embedding")
    parser.add_argument("--candidate-validity-predicate", default="embedding_valid")
    parser.add_argument(
        "--expected-truth-self-excluded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require this exact self_excluded value in the truth provenance and CSV.",
    )
    parser.add_argument("--matched-mode", choices=("design1_bloom",), default="design1_bloom")
    parser.add_argument("--matched-target-recall", type=float, default=0.90)
    parser.add_argument(
        "--matched-config-index-policy",
        choices=("exact", "same_table_borrowed"),
        default="exact",
    )
    parser.add_argument("--allow-mean-qualified-matched-config", action="store_true")
    parser.add_argument("--container", default="hybrid-pgvector")
    parser.add_argument("--backend-cpu", type=int, required=True)
    parser.add_argument("--query-offset", type=int, default=QUERY_OFFSET)
    parser.add_argument("--queries", type=positive_int, default=QUERIES)
    parser.add_argument("--repeats", type=positive_int, default=REPEATS)
    parser.add_argument("--cache-regime", choices=CACHE_REGIMES, required=True)
    parser.add_argument(
        "--cold-block-queries", type=positive_int, default=COLD_BLOCK_QUERIES
    )
    parser.add_argument("--prewarm-index-blocks", type=positive_int)
    parser.add_argument("--prewarm-common-relation", action="append", default=[])
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--d1-cache-mb", type=positive_int, default=1024)
    parser.add_argument("--d1-guidance-kind", choices=("auto", "exact", "bloom"), default="auto")
    parser.add_argument("--guidance-max-atoms", type=positive_int, default=64)
    parser.add_argument("--statement-timeout-ms", type=positive_int, default=300_000)
    parser.add_argument("--schedule-seed", type=int, default=SCHEDULE_SEED)
    parser.add_argument("--readiness-timeout-s", type=float, default=60.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--expected-sqlens-build-id", required=True)
    parser.add_argument("--expected-vector-so-sha256", type=sha256_arg, required=True)
    parser.add_argument("--expected-candidate-rows", type=positive_int, default=9_979_556)
    parser.add_argument("--i-understand-container-restarts", action="store_true")
    parser.add_argument("--i-understand-relation-cache-eviction", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only a running/failed manifest at the same --out; completed "
            "children are revalidated and never rerun."
        ),
    )
    parser.add_argument("--finalize-complete-invocations", action="store_true")
    parser.add_argument("--refresh-completed-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.backend_cpu < 0:
        parser.error("--backend-cpu must be nonnegative")
    if args.query_offset < 0:
        parser.error("--query-offset must be nonnegative")
    if not 0.0 < args.matched_target_recall <= 1.0 or not math.isfinite(args.matched_target_recall):
        parser.error("--matched-target-recall must be finite and within (0, 1]")
    if args.cache_regime == "cold_io" and args.prewarm_index_blocks is not None:
        parser.error("--prewarm-index-blocks is incompatible with --cache-regime cold_io")
    if args.resume and (args.dry_run or args.finalize_complete_invocations or args.refresh_completed_summary):
        parser.error("--resume is only valid for the measurement execution path")
    if args.matched_configs_csv is None and args.matched_recall_manifest is None:
        parser.error(
            "provide --matched-recall-manifest or --matched-configs-csv with its manifest"
        )
    if (args.matched_configs_csv is None) != (args.matched_configs_manifest is None):
        parser.error(
            "--matched-configs-csv and --matched-configs-manifest must be supplied together"
        )
    required_paths = [
        args.filters_csv,
        args.truth_csv,
        args.truth_manifest,
        args.d2_graph_proof_json,
    ]
    if args.matched_configs_csv is not None:
        required_paths.extend([args.matched_configs_csv, args.matched_configs_manifest])
    else:
        required_paths.append(args.matched_recall_manifest)
    for path in required_paths:
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if not RUNNER_PATH.is_file():
        parser.error(f"delegated runner does not exist: {RUNNER_PATH}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_payload(args), indent=2, sort_keys=True))
        return
    if args.finalize_complete_invocations:
        finalize_completed_invocations(args)
        return
    if args.refresh_completed_summary:
        refresh_completed_summary(args)
        return
    execute(args)


if __name__ == "__main__":
    main()
