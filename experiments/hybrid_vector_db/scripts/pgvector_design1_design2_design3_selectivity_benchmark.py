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
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import psycopg
from psycopg import errors

try:
    from .common_pg import pg_config_from_env
    from .faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS
    from .pgvector_predicate_guidance_benchmark import FILTER_ATOMS
except ImportError:  # Direct script execution puts this directory on sys.path.
    from common_pg import pg_config_from_env
    from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS
    from pgvector_predicate_guidance_benchmark import FILTER_ATOMS


INSERTION_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
INSERTION_INDEX = "public.amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx"
BFS_TABLE = INSERTION_TABLE
BFS_INDEX = "public.amazon10m_embedding_valid_hnsw_m32ef200_fullmem_bfs_idx"

MODES = [
    "original",
    "design1_bloom",
    "design1_bloom_bfs_layout",
    "design1_bloom_bfs_layout_d3",
]

MODE_LABELS = {
    "original": "Original pgvector",
    "design1_bloom": "Design 1",
    "design1_bloom_bfs_layout": "Design 1 + Design 2",
    "design1_bloom_bfs_layout_d3": "Design 1 + Design 2 + Design 3",
}

MODE_CONFIG_FIELDS = (
    "ef_search",
    "max_scan_tuples",
    "scan_mem_multiplier",
    "iterative_scan",
    "guided_collect_target",
    "traversal_guided_target",
    "traversal_guided_prioritization",
    "traversal_guided_burst",
    "traversal_guided_early_stop",
    "traversal_guided_early_stop_distance_ratio",
)
MODE_CONFIG_DEFAULTS = {
    "ef_search": 1000,
    "max_scan_tuples": 20000,
    "scan_mem_multiplier": 1.0,
    "iterative_scan": "off",
    "guided_collect_target": 1000,
    "traversal_guided_target": 40,
    "traversal_guided_prioritization": True,
    "traversal_guided_burst": 8,
    "traversal_guided_early_stop": False,
    "traversal_guided_early_stop_distance_ratio": 0.0,
}
ITERATIVE_SCAN_VALUES = {"off", "strict_order", "relaxed_order"}
D3_MEASUREMENT_POLICIES = {"workload_driven_adaptive", "admitted_warm_reuse"}
D3_PHASES = ("probe", "admission", "refinement", "warm", "bypass")
SQLENS_V11_BUILD_PREFIX = "sqlens-v11-"
SQLENS_V12_BUILD_PREFIX = "sqlens-v12-"
SQLENS_V13_BUILD_PREFIX = "sqlens-v13-"
SQLENS_V14_BUILD_PREFIX = "sqlens-v14-"
SQLENS_V15_BUILD_PREFIX = "sqlens-v15-"
SQLENS_V16_BUILD_PREFIX = "sqlens-v16-"
SQLENS_V17_BUILD_PREFIX = "sqlens-v17-"
SQLENS_SUPPORTED_BUILD_PREFIXES = (
    SQLENS_V11_BUILD_PREFIX,
    SQLENS_V12_BUILD_PREFIX,
    SQLENS_V13_BUILD_PREFIX,
    SQLENS_V14_BUILD_PREFIX,
    SQLENS_V15_BUILD_PREFIX,
    SQLENS_V16_BUILD_PREFIX,
    SQLENS_V17_BUILD_PREFIX,
)
SQLENS_MIN_PROFILE_SEMANTICS = 12.0
SQLENS_WRAPPER_DDL_ADVISORY_LOCK = 0x53514C454E535744
SQLENS_PROFILE_FIELDS = (
    "graph_elements_visited",
    "raw_index_tids_returned",
    "hnsw_am_callback_ms",
    "executor_residual_ms",
    "index_readbuffer_calls",
    "index_readbuffer_ms",
    "index_readbuffer_shared_read_calls",
    "index_readbuffer_shared_read_ms",
    "index_readbuffer_shared_hit_calls",
    "index_readbuffer_shared_hit_ms",
    "index_readbuffer_unclassified_calls",
    "index_readbuffer_unclassified_ms",
    "index_readbuffer_timing_scope",
    "index_readbuffer_classification_scope",
    "distance_compute_timed_calls",
    "distance_compute_ms",
    "distance_compute_timing_scope",
    "hnsw_remaining_ms",
    "profile_timer_overhead_scope",
    "index_page_loads",
    "index_page_runs",
    "index_page_distinct_pages",
    "index_page_distinct_pages_exact",
    "index_page_profile_scope",
    "heap_tid_returns",
    "heap_tid_page_runs",
    "heap_tid_distinct_pages",
    "heap_tid_distinct_pages_exact",
    "heap_tid_sequence_scope",
    "heap_blks_are_exact_heap_io",
)
SQLENS_TRAVERSAL_PROFILE_FIELDS = (
    "traversal_result_target",
    "traversal_guided_result_count",
    "traversal_max_scan_reached",
    "final_path",
    "planner_proof_attempted",
    "planner_proof_succeeded",
    "planner_proof_bypass_reason",
    "traversal_guidance_scope",
    "graph_expansion_pruned",
    "distance_computations_pruned",
    "pre_distance_membership_checks",
    "pre_distance_membership_matches",
    "pre_distance_membership_misses",
    "distance_computations_avoided_attempted",
    "distance_computations_avoided",
    "neighbor_expansion_guidance_checks",
    "neighbor_expansion_guidance_matches",
    "neighbor_expansion_guidance_misses",
    "traversal_guided_admissions",
    "traversal_guided_suppressions",
    "traversal_heap_tids_suppressed",
    "guided_expanded_nodes",
    "guided_phase_distance_computations",
    "stock_phase_expanded_nodes",
    "stock_phase_distance_computations",
    "stock_bypass_requests",
    "stock_bypass_reason",
    "fallback_requests",
    "fallback_reason",
    "fallback_stock_expanded_nodes",
    "fallback_stock_distance_computations",
    "fallback_iterative_scan_enabled",
    "traversal_estimated_skip_rate_valid",
    "traversal_estimated_skip_rate",
    "approximate_prioritization_attempted",
    "traversal_order_changed",
    "approximate_ann_path",
    "priority_reorders",
    "match_frontier_pops",
    "no_bridge_frontier_pops",
)
D2_GRAPH_PROOF_FIELDS = (
    "same_heap",
    "logical_equal",
    "entry_equal",
    "tuple_coverage_equal",
    "physical_equal",
)
D2_STABLE_COMPARISON_FIELDS = (
    "format",
    "same_heap",
    "logical_equal",
    "physical_equal",
    "entry_equal",
    "definition_equal",
    "tuple_coverage_equal",
    "left_definition_digest",
    "right_definition_digest",
    "left_tuple_coverage_digest",
    "right_tuple_coverage_digest",
    "left_logical_digest",
    "right_logical_digest",
    "left_physical_digest",
    "right_physical_digest",
)
D2_V3_COVERAGE_FIELDS = (
    "left_nodes",
    "right_nodes",
    "left_heap_tids",
    "right_heap_tids",
    "left_tombstones",
    "right_tombstones",
)
D2_RELATION_IDENTITY_FIELDS = ("name", "oid", "relfilenode", "heap_oid")
D2_BFS_LOCALITY_COMPARISON_FIELDS = (
    "left_bfs_locality",
    "right_bfs_locality",
)
D2_EDGE_SPAN_COMPARISON_FIELDS = (
    "left_edge_span",
    "right_edge_span",
)
D2_GRAPH_FINGERPRINT_ROLES = ("source", "clone")


class SqlensProvenanceGateError(RuntimeError):
    """Raised when the formal runner is not connected to the required SQLens ABI."""


class D2GraphProofGateError(RuntimeError):
    """Raised when D2 is not a same-heap, same-logical-graph layout comparison."""


class BackendAffinityGateError(RuntimeError):
    """Raised when a production PostgreSQL backend is not on the requested CPUs."""


@dataclass(frozen=True)
class TruthEntry:
    query_id: int
    filtered_rows: int
    kth_distance_sq: float | None
    tie_tolerance: float
    self_excluded: bool
    strict_closer_count: int | None = None
    boundary_tied: bool | None = None


@dataclass(frozen=True)
class WorkloadRequest:
    request_no: int
    query_no: int
    query_id: int
    filter_name: str
    trace_cycle: int
    split: str


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def validate_candidate_validity_predicate(value: str) -> str:
    """Accept one SQL expression, not a statement or comment-bearing fragment."""
    predicate = str(value or "").strip()
    forbidden = (";", "--", "/*", "*/", "\x00")
    token = next((token for token in forbidden if token in predicate), None)
    if token is not None:
        raise argparse.ArgumentTypeError(
            "candidate validity predicate must be a single comment-free SQL expression; "
            f"found forbidden token {token!r}"
        )
    return predicate


def effective_candidate_validity_predicate(value: object = "") -> str:
    predicate = validate_candidate_validity_predicate(str(value or ""))
    return predicate or "TRUE"


def candidate_validity_sha256(value: object = "") -> str:
    return hashlib.sha256(
        effective_candidate_validity_predicate(value).encode("utf-8")
    ).hexdigest()


def normalized_sql_predicate(value: object = "") -> str:
    """Normalize catalog-rendered SQL enough for a fail-closed predicate bind."""
    text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        balanced = True
        for position, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0 or (depth == 0 and position != len(text) - 1):
                    balanced = False
                    break
        if balanced and depth == 0:
            text = text[1:-1].strip()
        else:
            break
    return text


def candidate_validity_index_predicate_matches(
    catalog_predicate: object,
    candidate_validity_predicate: object = "",
) -> bool:
    """Bind TRUE to a full index and every other value to its exact partial qual."""
    expected = normalized_sql_predicate(
        effective_candidate_validity_predicate(candidate_validity_predicate)
    )
    observed = normalized_sql_predicate(catalog_predicate)
    if expected == "true":
        return not observed
    return bool(observed) and observed == expected


def validate_guidance_atoms(
    atoms: list[str],
    candidate_validity_predicate: object = "",
) -> list[str]:
    """Reject global candidate validity quals from the D1 atom channel."""
    validity = normalized_sql_predicate(
        effective_candidate_validity_predicate(candidate_validity_predicate)
    )
    if not validity:
        return atoms
    pattern = re.compile(
        r"(?<![a-z0-9_$])" + re.escape(validity) + r"(?![a-z0-9_$])",
        re.IGNORECASE,
    )
    invalid = [atom for atom in atoms if pattern.search(normalized_sql_predicate(atom))]
    if invalid:
        raise ValueError(
            "D1 guidance atoms must not contain the global candidate validity predicate; "
            f"invalid atoms={invalid!r}, predicate={validity!r}"
        )
    return atoms


def _cpu_set(value: str) -> set[int]:
    cpus: set[int] = set()
    for token in str(value).strip().split(","):
        token = token.strip()
        if not token:
            raise ValueError("CPU list contains an empty range")
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            first = int(first_text)
            last = int(last_text)
        else:
            first = last = int(token)
        if first < 0 or last < first:
            raise ValueError(f"invalid CPU range: {token!r}")
        cpus.update(range(first, last + 1))
    if not cpus:
        raise ValueError("CPU list is empty")
    return cpus


def normalize_cpu_list(value: str) -> str:
    try:
        cpus = sorted(_cpu_set(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid CPU list {value!r}: {exc}") from exc
    ranges: list[str] = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def fragment_store_namespace_arg(value: str) -> str:
    namespace = str(value or "").strip()
    if namespace and not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", namespace):
        raise argparse.ArgumentTypeError(
            "fragment-store namespace must match [A-Za-z0-9_.-]{1,64}"
        )
    return namespace


def d3_measurement_policy(args: argparse.Namespace) -> str:
    policy = str(
        getattr(args, "d3_measurement_policy", "workload_driven_adaptive")
    )
    if policy not in D3_MEASUREMENT_POLICIES:
        raise ValueError(f"unsupported D3 measurement policy: {policy!r}")
    return policy


def d3_initialization_label(args: argparse.Namespace) -> str:
    return d3_measurement_policy(args)


def repeat_fragment_store_namespace(base: str, repeat: int) -> str:
    namespace = f"{base}-r{repeat}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", namespace):
        raise ValueError(
            "per-repeat fragment-store namespace must match "
            "[A-Za-z0-9_.-]{1,64}"
        )
    return namespace


def expected_plan_evidence_count(
    args: argparse.Namespace,
    filters: Sequence[tuple[str, float, str]],
) -> int:
    return len(args.modes) * len(filters) * int(args.repeats)


def mode_uses_unmeasured_warmup(args: argparse.Namespace, mode: str) -> bool:
    return (
        mode != "design1_bloom_bfs_layout_d3"
        or d3_measurement_policy(args) == "admitted_warm_reuse"
    )


def backend_cpu_provenance(
    cur: psycopg.Cursor,
    requested_cpu_list: str | None,
) -> dict[str, object]:
    cur.execute("SELECT pg_backend_pid(), pg_read_file('/proc/self/status')")
    row = cur.fetchone()
    if not row or row[0] is None or not isinstance(row[1], str):
        raise BackendAffinityGateError(
            "could not read pg_backend_pid()/DB-side /proc/self/status affinity"
        )
    observed_raw = ""
    for line in row[1].splitlines():
        if line.startswith("Cpus_allowed_list:"):
            observed_raw = line.split(":", 1)[1].strip()
            break
    if not observed_raw:
        raise BackendAffinityGateError(
            "DB-side /proc/self/status is missing Cpus_allowed_list"
        )
    try:
        observed = normalize_cpu_list(observed_raw)
    except argparse.ArgumentTypeError as exc:
        raise BackendAffinityGateError(
            f"DB-side Cpus_allowed_list is invalid: {observed_raw!r}"
        ) from exc
    requested = normalize_cpu_list(requested_cpu_list) if requested_cpu_list else ""
    exact_match = _cpu_set(observed) == _cpu_set(requested) if requested else None
    return {
        "backend_pid": int(row[0]),
        "pid_namespace": "postgresql_container_namespace",
        "requested_cpu_list": requested,
        "observed_cpu_list": observed,
        "exact_match": exact_match,
        "pinning_attempted_by_runner": False,
        "mapping_trust": "db_side_proc_self_status",
        "checked_at": utc_now(),
    }


def enforce_backend_cpu_provenance(provenance: dict[str, object]) -> None:
    if provenance.get("requested_cpu_list") and provenance.get("exact_match") is not True:
        raise BackendAffinityGateError(
            "PostgreSQL backend CPU affinity mismatch: "
            f"backend_pid={provenance.get('backend_pid')}, "
            f"requested={provenance.get('requested_cpu_list')!r}, "
            f"observed={provenance.get('observed_cpu_list')!r}. "
            "Pin the trustworthy host PostgreSQL PID in orchestration; the runner will not "
            "apply taskset to a Docker namespace PID."
        )


def load_tie_aware_truth(
    path: Path,
    method: str = "pre_filter_exact",
    expected_self_excluded: bool = True,
    expected_candidate_validity_predicate: str | None = None,
) -> tuple[dict[tuple[str, int], TruthEntry], dict[int, int]]:
    truth: dict[tuple[str, int], TruthEntry] = {}
    query_by_no: dict[int, int] = {}
    required = {
        "filtered_rows",
        "kth_distance_sq",
        "tie_tolerance",
        "self_excluded",
    }
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"truth CSV is missing tie-aware fields: {sorted(missing)}")
        if (
            expected_candidate_validity_predicate is not None
            and "candidate_validity_predicate" not in set(reader.fieldnames or ())
        ):
            raise ValueError(
                "truth CSV is missing candidate_validity_predicate required by the "
                "explicit candidate-validity contract"
            )
        for row in reader:
            if row.get("method") != method:
                continue
            query_no = int(row["query_no"])
            query_id = int(row["query_id"])
            previous = query_by_no.setdefault(query_no, query_id)
            if previous != query_id:
                raise ValueError(f"query_no={query_no} maps to multiple query IDs")
            self_excluded = parse_bool(row["self_excluded"])
            if self_excluded != expected_self_excluded:
                raise ValueError(
                    f"truth row {(row['filter_name'], query_no)} self_excluded="
                    f"{self_excluded!r} does not match expected {expected_self_excluded!r}"
                )
            if expected_candidate_validity_predicate is not None:
                expected_validity = effective_candidate_validity_predicate(
                    expected_candidate_validity_predicate
                )
                observed_validity = effective_candidate_validity_predicate(
                    row.get("candidate_validity_predicate", "")
                )
                if observed_validity != expected_validity:
                    raise ValueError(
                        f"truth row {(row['filter_name'], query_no)} candidate_validity_predicate="
                        f"{observed_validity!r} does not match expected {expected_validity!r}"
                    )
            filtered_rows = int(row["filtered_rows"])
            kth_distance_sq = (
                float(row["kth_distance_sq"]) if row["kth_distance_sq"].strip() else None
            )
            tie_tolerance = float(row["tie_tolerance"])
            strict_closer_count = (
                int(row["strict_closer_count"])
                if row.get("strict_closer_count", "").strip()
                else None
            )
            boundary_tied = (
                parse_bool(row["boundary_tied"])
                if row.get("boundary_tied", "").strip()
                else None
            )
            if filtered_rows < 0 or (strict_closer_count is not None and strict_closer_count < 0):
                raise ValueError("tie-aware truth counts must be non-negative")
            if tie_tolerance < 0:
                raise ValueError("tie_tolerance must be non-negative")
            if filtered_rows and kth_distance_sq is None:
                raise ValueError("non-empty formal truth requires kth_distance_sq")
            key = (row["filter_name"], query_no)
            if key in truth:
                raise ValueError(f"duplicate truth row: {key}")
            truth[key] = TruthEntry(
                query_id=query_id,
                filtered_rows=filtered_rows,
                kth_distance_sq=kth_distance_sq,
                tie_tolerance=tie_tolerance,
                self_excluded=self_excluded,
                strict_closer_count=strict_closer_count,
                boundary_tied=boundary_tied,
            )
    return truth, query_by_no


def load_workload_requests(
    path: Path,
    *,
    query_by_no: dict[int, int],
    filters: list[tuple[str, float, str]],
    truth: dict[tuple[str, int], TruthEntry],
    expected_requests: int = 0,
    request_limit: int = 0,
    selected_filter_names: set[str] | None = None,
    require_unique_queries: bool = True,
) -> list[WorkloadRequest]:
    filter_names = {name for name, _, _ in filters}
    requests: list[WorkloadRequest] = []
    required = {"request_no", "query_no", "query_id", "filter_name", "trace_cycle", "split"}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"workload CSV is missing fields: {sorted(missing)}")
        for row in reader:
            request = WorkloadRequest(
                request_no=int(row["request_no"]),
                query_no=int(row["query_no"]),
                query_id=int(row["query_id"]),
                filter_name=str(row["filter_name"]),
                trace_cycle=int(row["trace_cycle"]),
                split=str(row["split"]),
            )
            if request.filter_name not in filter_names:
                raise ValueError(
                    f"workload request {request.request_no} uses unknown filter "
                    f"{request.filter_name!r}"
                )
            expected_query_id = query_by_no.get(request.query_no)
            if expected_query_id is None or expected_query_id != request.query_id:
                raise ValueError(
                    f"workload request {request.request_no} query mapping does not match truth: "
                    f"q{request.query_no}={request.query_id}, expected={expected_query_id}"
                )
            if (request.filter_name, request.query_no) not in truth:
                raise ValueError(
                    f"workload request {request.request_no} has no tie-aware truth row"
                )
            requests.append(request)

    request_nos = [request.request_no for request in requests]
    if request_nos != list(range(len(requests))):
        raise ValueError("workload request_no must be contiguous and ordered from zero")
    if expected_requests and len(requests) != expected_requests:
        raise ValueError(
            f"workload has {len(requests)} requests, expected {expected_requests}"
        )
    if request_limit:
        if request_limit < 1 or request_limit > len(requests):
            raise ValueError(
                f"workload request limit must be in [1, {len(requests)}], "
                f"got {request_limit}"
            )
        requests = requests[:request_limit]
    if selected_filter_names is not None:
        unknown_selected = selected_filter_names - filter_names
        if unknown_selected:
            raise ValueError(
                f"selected workload filters are unknown: {sorted(unknown_selected)}"
            )
        requests = [
            WorkloadRequest(
                request_no=selected_request_no,
                query_no=request.query_no,
                query_id=request.query_id,
                filter_name=request.filter_name,
                trace_cycle=request.trace_cycle,
                split=request.split,
            )
            for selected_request_no, request in enumerate(
                request
                for request in requests
                if request.filter_name in selected_filter_names
            )
        ]
    if require_unique_queries:
        query_nos = [request.query_no for request in requests]
        query_ids = [request.query_id for request in requests]
        if len(set(query_nos)) != len(requests) or len(set(query_ids)) != len(requests):
            raise ValueError("formal workload must contain one unique query vector per request")
    expected_filter_names = (
        selected_filter_names
        if selected_filter_names is not None
        else filter_names
    )
    observed_filters = {request.filter_name for request in requests}
    if observed_filters != expected_filter_names:
        raise ValueError(
            "workload filter coverage is incomplete: "
            f"missing={sorted(expected_filter_names - observed_filters)} "
            f"unexpected={sorted(observed_filters - expected_filter_names)}"
        )
    return requests


def tie_aware_recall(result_distances: list[float], truth: TruthEntry, k: int) -> float:
    denominator = min(k, truth.filtered_rows)
    if denominator == 0:
        return 0.0
    if truth.kth_distance_sq is None:
        raise ValueError("formal truth is missing kth_distance_sq")
    if truth.strict_closer_count is not None and truth.strict_closer_count > denominator:
        raise ValueError("strict_closer_count exceeds the recall denominator")
    credit = min(
        denominator,
        k,
        sum(
            distance * distance <= truth.kth_distance_sq + truth.tie_tolerance
            for distance in result_distances[:k]
        ),
    )
    return credit / denominator


def parse_mode_configs_json(value: str) -> dict[str, dict[str, object]]:
    source = value
    if not value.lstrip().startswith("{"):
        try:
            source = Path(value).read_text(encoding="utf-8")
        except OSError as exc:
            raise argparse.ArgumentTypeError(f"cannot read mode config JSON from {value}: {exc}") from exc
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid mode config JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("mode config JSON must be an object")

    configs: dict[str, dict[str, object]] = {}
    for mode, overrides in parsed.items():
        if mode not in MODES:
            raise argparse.ArgumentTypeError(f"unknown mode in mode config JSON: {mode}")
        if not isinstance(overrides, dict):
            raise argparse.ArgumentTypeError(f"mode config for {mode} must be an object")
        unknown = sorted(set(overrides) - set(MODE_CONFIG_FIELDS))
        if unknown:
            raise argparse.ArgumentTypeError(f"unknown config field for {mode}: {unknown[0]}")

        normalized: dict[str, object] = {}
        for field, field_value in overrides.items():
            if field in {
                "ef_search",
                "max_scan_tuples",
                "guided_collect_target",
                "traversal_guided_target",
                "traversal_guided_burst",
            }:
                if isinstance(field_value, bool) or not isinstance(field_value, int):
                    raise argparse.ArgumentTypeError(f"{mode}.{field} must be an integer")
            elif field in {
                "traversal_guided_prioritization",
                "traversal_guided_early_stop",
            }:
                if not isinstance(field_value, bool):
                    raise argparse.ArgumentTypeError(f"{mode}.{field} must be a boolean")
            elif field in {
                "scan_mem_multiplier",
                "traversal_guided_early_stop_distance_ratio",
            }:
                if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
                    raise argparse.ArgumentTypeError(f"{mode}.{field} must be a number")
                field_value = float(field_value)
            elif field == "iterative_scan":
                if field_value not in ITERATIVE_SCAN_VALUES:
                    choices = ", ".join(sorted(ITERATIVE_SCAN_VALUES))
                    raise argparse.ArgumentTypeError(f"{mode}.{field} must be one of: {choices}")
            normalized[field] = field_value
        configs[mode] = normalized
    return configs


def parse_filter_ef_search_json(value: str) -> dict[str, dict[str, int]]:
    source = value
    if not value.lstrip().startswith("{"):
        try:
            source = Path(value).read_text(encoding="utf-8")
        except OSError as exc:
            raise argparse.ArgumentTypeError(
                f"cannot read per-filter ef_search JSON from {value}: {exc}"
            ) from exc
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid per-filter ef_search JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("per-filter ef_search JSON must be an object")

    configs: dict[str, dict[str, int]] = {}
    for mode, overrides in parsed.items():
        if mode not in MODES:
            raise argparse.ArgumentTypeError(
                f"unknown mode in per-filter ef_search JSON: {mode}"
            )
        if not isinstance(overrides, dict):
            raise argparse.ArgumentTypeError(
                f"per-filter ef_search config for {mode} must be an object"
            )
        normalized: dict[str, int] = {}
        for filter_name, ef_search in overrides.items():
            if (
                not isinstance(filter_name, str)
                or not filter_name
                or isinstance(ef_search, bool)
                or not isinstance(ef_search, int)
                or ef_search < 1
            ):
                raise argparse.ArgumentTypeError(
                    f"{mode}.{filter_name} ef_search must be a positive integer"
                )
            normalized[filter_name] = ef_search
        configs[mode] = normalized
    return configs


def parse_filter_traversal_target_json(value: str) -> dict[str, dict[str, int]]:
    source = value
    if not value.lstrip().startswith("{"):
        try:
            source = Path(value).read_text(encoding="utf-8")
        except OSError as exc:
            raise argparse.ArgumentTypeError(
                f"cannot read per-filter traversal target JSON from {value}: {exc}"
            ) from exc
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid per-filter traversal target JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            "per-filter traversal target JSON must be an object"
        )

    configs: dict[str, dict[str, int]] = {}
    for mode, overrides in parsed.items():
        if mode not in MODES:
            raise argparse.ArgumentTypeError(
                f"unknown mode in per-filter traversal target JSON: {mode}"
            )
        if not isinstance(overrides, dict):
            raise argparse.ArgumentTypeError(
                f"per-filter traversal target config for {mode} must be an object"
            )
        normalized: dict[str, int] = {}
        for filter_name, target in overrides.items():
            if (
                not isinstance(filter_name, str)
                or not filter_name
                or isinstance(target, bool)
                or not isinstance(target, int)
                or target < 1
            ):
                raise argparse.ArgumentTypeError(
                    f"{mode}.{filter_name} traversal target must be a positive integer"
                )
            normalized[filter_name] = target
        configs[mode] = normalized
    return configs


def parse_filter_mode_configs_json(
    value: str,
) -> dict[str, dict[str, dict[str, object]]]:
    """Parse complete per-mode, per-filter search configuration overrides."""
    source = value
    if not value.lstrip().startswith("{"):
        try:
            source = Path(value).read_text(encoding="utf-8")
        except OSError as exc:
            raise argparse.ArgumentTypeError(
                f"cannot read per-filter mode config JSON from {value}: {exc}"
            ) from exc
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid per-filter mode config JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            "per-filter mode config JSON must be an object"
        )
    result: dict[str, dict[str, dict[str, object]]] = {}
    for mode, by_filter in parsed.items():
        if mode not in MODES:
            raise argparse.ArgumentTypeError(
                f"unknown mode in per-filter mode config JSON: {mode}"
            )
        if not isinstance(by_filter, dict):
            raise argparse.ArgumentTypeError(
                f"per-filter mode config for {mode} must be an object"
            )
        normalized: dict[str, dict[str, object]] = {}
        for filter_name, overrides in by_filter.items():
            if not isinstance(filter_name, str) or not filter_name:
                raise argparse.ArgumentTypeError("filter names must be nonempty strings")
            if not isinstance(overrides, dict):
                raise argparse.ArgumentTypeError(
                    f"{mode}.{filter_name} config must be an object"
                )
            unknown = sorted(set(overrides) - set(MODE_CONFIG_FIELDS))
            if unknown:
                raise argparse.ArgumentTypeError(
                    f"{mode}.{filter_name} has unknown fields: {', '.join(unknown)}"
                )
            # Reuse the canonical mode parser so type/range semantics cannot drift.
            parsed_mode = parse_mode_configs_json(
                json.dumps({mode: overrides}, separators=(",", ":"))
            )[mode]
            normalized[filter_name] = parsed_mode
        result[mode] = normalized
    return result


def effective_mode_config(args: argparse.Namespace, mode: str) -> dict[str, object]:
    config = {
        field: getattr(args, field, MODE_CONFIG_DEFAULTS.get(field))
        for field in MODE_CONFIG_FIELDS
    }
    config.update(getattr(args, "mode_configs_json", {}).get(mode, {}))
    if mode != "original" and bool(config["traversal_guided_prioritization"]):
        target = int(config["traversal_guided_target"])
        k = int(getattr(args, "k", 10))
        ef_search = int(config["ef_search"])
        table, _ = mode_table_index(args, mode)
        client_self_exclusion = (
            uses_exact_predicate_scan_contract(
                getattr(args, "guidance_filter_strategy", "traversal_guided")
            )
            and candidate_self_exclusion(args, table)
        )
        stop_ratio = float(config["traversal_guided_early_stop_distance_ratio"])
        if stop_ratio < 0 or stop_ratio > 1:
            raise ValueError(
                f"{mode}.traversal_guided_early_stop_distance_ratio must be in [0, 1]"
            )
        minimum_target = k + int(client_self_exclusion)
        if target < minimum_target or target > ef_search:
            raise ValueError(
                f"{mode}.traversal_guided_target must satisfy "
                f"k(+client_self_exclusion) <= target <= ef_search; got "
                f"k={k}, client_self_exclusion={client_self_exclusion}, "
                f"target={target}, ef_search={ef_search}"
            )
    return config


def configured_ef_search_for_filter(
    args: argparse.Namespace,
    runtime: "ModeRuntime",
    filter_name: str,
) -> int:
    default = int(
        runtime.config.get("ef_search", getattr(args, "ef_search", 1000))
    )
    full = getattr(args, "filter_mode_configs_json", {}).get(runtime.mode, {})
    if filter_name in full and "ef_search" in full[filter_name]:
        return int(full[filter_name]["ef_search"])
    by_mode = getattr(args, "filter_ef_search_json", {}).get(runtime.mode, {})
    return int(by_mode.get(filter_name, default))


def configured_traversal_target_for_filter(
    args: argparse.Namespace,
    runtime: "ModeRuntime",
    filter_name: str,
) -> int:
    default = int(
        runtime.config.get(
            "traversal_guided_target",
            getattr(args, "traversal_guided_target", 40),
        )
    )
    full = getattr(args, "filter_mode_configs_json", {}).get(runtime.mode, {})
    if filter_name in full and "traversal_guided_target" in full[filter_name]:
        return int(full[filter_name]["traversal_guided_target"])
    by_mode = getattr(args, "filter_traversal_target_json", {}).get(
        runtime.mode, {}
    )
    return int(by_mode.get(filter_name, default))


def configured_mode_value_for_filter(
    args: argparse.Namespace,
    runtime: "ModeRuntime",
    filter_name: str,
    field: str,
) -> object:
    by_mode = getattr(args, "filter_mode_configs_json", {}).get(runtime.mode, {})
    overrides = by_mode.get(filter_name, {})
    if field in overrides:
        return overrides[field]
    if field in runtime.config:
        return runtime.config[field]
    if hasattr(args, field) and getattr(args, field) is not None:
        return getattr(args, field)
    if field in MODE_CONFIG_DEFAULTS:
        return MODE_CONFIG_DEFAULTS[field]
    raise RuntimeError(f"missing mode configuration field {field!r}")


def initialize_routed_search_state(runtime: "ModeRuntime") -> None:
    """Backfill state for older callers that construct ModeRuntime directly."""
    fields = (
        ("max_scan_tuples_current_setting", "max_scan_tuples"),
        ("scan_mem_multiplier_current_setting", "scan_mem_multiplier"),
        ("guided_collect_target_current_setting", "guided_collect_target"),
        ("traversal_guided_burst_current_setting", "traversal_guided_burst"),
        (
            "traversal_guided_early_stop_current_setting",
            "traversal_guided_early_stop",
        ),
        (
            "traversal_guided_early_stop_distance_ratio_current_setting",
            "traversal_guided_early_stop_distance_ratio",
        ),
    )
    for attribute, field in fields:
        if getattr(runtime, attribute) is None:
            setattr(
                runtime,
                attribute,
                runtime.config.get(field, MODE_CONFIG_DEFAULTS[field]),
            )


def search_configuration_evidence(args: argparse.Namespace) -> dict[str, object]:
    filter_ef_search_overrides = dict(
        getattr(args, "filter_ef_search_json", {}) or {}
    )
    filter_traversal_target_overrides = dict(
        getattr(args, "filter_traversal_target_json", {}) or {}
    )
    filter_mode_config_overrides = dict(
        getattr(args, "filter_mode_configs_json", {}) or {}
    )
    configured_scope = (
        "per_filter"
        if (
            filter_ef_search_overrides
            or filter_traversal_target_overrides
            or filter_mode_config_overrides
        )
        else "global_policy"
    )
    return {
        "schema_version": 1,
        "configured_scope": configured_scope,
        "mode_defaults": dict(getattr(args, "mode_configs_json", {}) or {}),
        "filter_ef_search_overrides": filter_ef_search_overrides,
        "filter_traversal_target_overrides": filter_traversal_target_overrides,
        "filter_mode_config_overrides": filter_mode_config_overrides,
        "guidance_bypass_policy": {
            "ef_search": int(
                getattr(args, "guidance_bypass_ef_search", 0) or 0
            ),
            "low_selectivity_ef_search": int(
                getattr(args, "guidance_low_selectivity_bypass_ef_search", 0)
                or 0
            ),
            "iterative_scan": str(
                getattr(args, "guidance_bypass_iterative_scan", "")
            ),
            "selectivity_min_pct": float(
                getattr(args, "guidance_selectivity_min_pct", 0.0)
            ),
            "selectivity_max_pct": float(
                getattr(args, "guidance_selectivity_max_pct", 100.0)
            ),
            "composite_max_selectivity_pct": float(
                getattr(
                    args,
                    "guidance_composite_max_selectivity_pct",
                    100.0,
                )
            ),
        },
        "effective_settings_recorded_per_request": True,
    }


def shuffled_modes(modes: list[str], rng: random.Random) -> list[str]:
    scheduled = list(modes)
    rng.shuffle(scheduled)
    return scheduled


def balanced_mode_order(modes: list[str], block_no: int, seed: int) -> list[str]:
    base = list(modes)
    random.Random(seed).shuffle(base)
    if not base:
        return base
    offset = block_no % len(base)
    return base[offset:] + base[:offset]


def parse_atoms(text: str) -> list[str]:
    atoms = [part.strip() for part in str(text or "").split("||") if part.strip()]
    if not atoms:
        raise ValueError("empty atoms field")
    separators = {"|", "OR"}
    if atoms[0].upper() in separators or atoms[-1].upper() in separators:
        raise ValueError("atom composition cannot start or end with OR")
    for previous, current in zip(atoms, atoms[1:]):
        if previous.upper() in separators and current.upper() in separators:
            raise ValueError("atom composition cannot contain adjacent OR separators")
    return atoms


def load_filter_specs(path: Path | None) -> tuple[list[tuple[str, str, str]], dict[str, list[str]]]:
    if path is None:
        return ATTR_FILTERS, dict(FILTER_ATOMS)
    filters: list[tuple[str, str, str]] = []
    atoms_by_filter: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["filter_name"]
            filters.append((name, row.get("actual_pct") or row["target_rate"], row["predicate"]))
            atoms_by_filter[name] = parse_atoms(row["atoms"])
    if not filters:
        raise SystemExit(f"no filters loaded from {path}")
    return filters, atoms_by_filter


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def parse_pct(value: object) -> float:
    text = str(value).strip().replace("%", "")
    return float(text)


def sqlens_build_prefix(build_id: str) -> str | None:
    return next(
        (prefix for prefix in SQLENS_SUPPORTED_BUILD_PREFIXES if build_id.startswith(prefix)),
        None,
    )


def require_sqlens_provenance(cur: psycopg.Cursor) -> tuple[str, dict[str, Any]]:
    """Verify a supported SQLens ABI before installing C-backed SQL wrappers."""
    try:
        cur.execute("SELECT vector_sqlens_build_id()")
        row = cur.fetchone()
        build_id = str(row[0]) if row and row[0] is not None else ""
    except Exception as exc:  # noqa: BLE001 - missing SQL must fail closed
        raise SqlensProvenanceGateError(
            "SQLens provenance gate failed: vector_sqlens_build_id() is unavailable. "
            "Install/reload a supported SQLens extension and reconnect before running this benchmark."
        ) from exc
    matched_prefix = sqlens_build_prefix(build_id)
    if matched_prefix is None:
        raise SqlensProvenanceGateError(
            f"SQLens provenance gate failed: vector_sqlens_build_id() returned {build_id!r}; "
            f"expected one of {SQLENS_SUPPORTED_BUILD_PREFIXES!r}. "
            "Rebuild/reload the intended SQLens extension and reconnect."
        )

    try:
        cur.execute("SELECT vector_hnsw_last_scan_profile()")
        row = cur.fetchone()
        raw_profile = row[0] if row else None
        profile = json.loads(raw_profile) if isinstance(raw_profile, str) else raw_profile
    except Exception as exc:  # noqa: BLE001 - missing SQL or invalid JSON must fail closed
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_hnsw_last_scan_profile() is unavailable or is not valid JSON. "
            "Load the SQLens v11 extension and reconnect before running this formal benchmark."
        ) from exc
    if not isinstance(profile, dict):
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_hnsw_last_scan_profile() did not return a JSON object. "
            "Load the SQLens v11 extension and reconnect before running this formal benchmark."
        )

    try:
        profile_version = float(profile["profile_semantics_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_hnsw_last_scan_profile() is missing a numeric "
            "profile_semantics_version. Load the SQLens v11 extension and reconnect."
        ) from exc

    required_fields = SQLENS_PROFILE_FIELDS + SQLENS_TRAVERSAL_PROFILE_FIELDS
    missing = [field for field in required_fields if field not in profile]
    instrumentation_errors: list[str] = []
    if not missing:
        try:
            readbuffer_calls = int(profile["index_readbuffer_calls"])
            readbuffer_classes = sum(
                int(profile[field])
                for field in (
                    "index_readbuffer_shared_read_calls",
                    "index_readbuffer_shared_hit_calls",
                    "index_readbuffer_unclassified_calls",
                )
            )
            if readbuffer_calls != readbuffer_classes:
                instrumentation_errors.append(
                    f"ReadBuffer calls do not sum ({readbuffer_calls} != {readbuffer_classes})"
                )
            distance_calls = int(profile["distance_compute_count"])
            timed_distance_calls = int(profile["distance_compute_timed_calls"])
            if distance_calls != timed_distance_calls:
                instrumentation_errors.append(
                    f"distance calls do not match timed calls ({distance_calls} != {timed_distance_calls})"
                )
            hnsw_callback = float(profile["hnsw_am_callback_ms"])
            breakdown = (
                float(profile["index_readbuffer_ms"])
                + float(profile["distance_compute_ms"])
                + float(profile["hnsw_remaining_ms"])
            )
            if not math.isfinite(breakdown) or abs(hnsw_callback - breakdown) >= 0.01:
                instrumentation_errors.append(
                    f"HNSW timing breakdown does not close ({hnsw_callback:.6f} != {breakdown:.6f})"
                )
        except (TypeError, ValueError, KeyError) as exc:
            instrumentation_errors.append(f"invalid instrumentation value: {exc}")
    if (
        not math.isfinite(profile_version)
        or profile_version < SQLENS_MIN_PROFILE_SEMANTICS
        or missing
        or instrumentation_errors
    ):
        details = []
        if not math.isfinite(profile_version) or profile_version < SQLENS_MIN_PROFILE_SEMANTICS:
            details.append(
                f"profile_semantics_version={profile.get('profile_semantics_version')!r} "
                f"(need >= {SQLENS_MIN_PROFILE_SEMANTICS:g})"
            )
        if missing:
            details.append(f"missing fields={missing!r}")
        details.extend(instrumentation_errors)
        raise SqlensProvenanceGateError(
            "SQLens r33 profile semantics gate failed: vector_hnsw_last_scan_profile() is incompatible: "
            + "; ".join(details)
            + ". Load the SQLens v11 extension and reconnect before running this formal benchmark."
        )
    return build_id, profile


def require_exact_sqlens_identity(
    cur: psycopg.Cursor,
    expected_build_id: str,
    expected_vector_so_sha256: str,
) -> dict[str, object]:
    if not expected_build_id or len(expected_vector_so_sha256) != 64:
        raise SqlensProvenanceGateError(
            "exact SQLens identity gate requires a parent-provided build ID and vector.so SHA256"
        )
    try:
        cur.execute(
            "WITH lib AS ("
            "SELECT setting || '/vector.so' AS path "
            "FROM pg_config WHERE name = 'PKGLIBDIR'"
            ") SELECT vector_sqlens_build_id(), path, "
            "encode(sha256(pg_read_binary_file(path)), 'hex') FROM lib"
        )
        row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 - binary identity must fail closed
        raise SqlensProvenanceGateError(
            "exact SQLens identity gate could not read the server-side vector.so"
        ) from exc
    observed_build_id = str(row[0]) if row and row[0] is not None else ""
    observed_path = str(row[1]) if row and row[1] is not None else ""
    observed_sha = str(row[2]) if row and row[2] is not None else ""
    if observed_build_id != expected_build_id:
        raise SqlensProvenanceGateError(
            "SQLens build ID mismatch: "
            f"expected {expected_build_id!r}, observed {observed_build_id!r}"
        )
    if observed_sha != expected_vector_so_sha256:
        raise SqlensProvenanceGateError(
            "server-side vector.so SHA256 mismatch: "
            f"expected {expected_vector_so_sha256!r}, observed {observed_sha!r}"
        )
    if not observed_path.endswith("/vector.so"):
        raise SqlensProvenanceGateError(
            f"server-side vector.so path is invalid: {observed_path!r}"
        )
    return {
        "expected_build_id": expected_build_id,
        "expected_vector_so_sha256": expected_vector_so_sha256,
        "observed_build_id": observed_build_id,
        "observed_vector_so_path": observed_path,
        "observed_vector_so_sha256": observed_sha,
        "exact_match": True,
        "checked_at": utc_now(),
    }


def parse_json_object(value: str) -> dict[str, object]:
    source = value
    if not value.lstrip().startswith("{"):
        try:
            source = Path(value).read_text(encoding="utf-8")
        except OSError as exc:
            raise argparse.ArgumentTypeError(f"cannot read JSON object from {value}: {exc}") from exc
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def stable_d2_graph_proof(proof: dict[str, object]) -> dict[str, object]:
    comparison = proof.get("comparison")
    relations = proof.get("relations")
    if not isinstance(comparison, dict) or not isinstance(relations, dict):
        raise D2GraphProofGateError("D2 stable proof is missing comparison/relation identity")
    stable_relations: dict[str, dict[str, object]] = {}
    for role in ("source", "clone"):
        relation = relations.get(role)
        if not isinstance(relation, dict):
            raise D2GraphProofGateError(f"D2 proof is missing {role} relation identity")
        missing = [field for field in D2_RELATION_IDENTITY_FIELDS if field not in relation]
        if missing:
            raise D2GraphProofGateError(
                f"D2 {role} relation identity is missing fields: {missing}"
            )
        stable_relations[role] = {
            field: relation[field] for field in D2_RELATION_IDENTITY_FIELDS
        }
    missing_comparison = [
        field for field in D2_STABLE_COMPARISON_FIELDS if field not in comparison
    ]
    if missing_comparison:
        raise D2GraphProofGateError(
            f"D2 graph proof is missing stable comparison fields: {missing_comparison}"
        )
    stable_comparison = {
        field: comparison[field] for field in D2_STABLE_COMPARISON_FIELDS
    }
    v3_coverage_present = [
        field for field in D2_V3_COVERAGE_FIELDS if field in comparison
    ]
    if comparison.get("format") == "sqlens-hnsw-compare-v3":
        missing_v3_coverage = [
            field for field in D2_V3_COVERAGE_FIELDS if field not in comparison
        ]
        if missing_v3_coverage:
            raise D2GraphProofGateError(
                f"D2 v3 graph proof is missing coverage fields: {missing_v3_coverage}"
            )
    elif v3_coverage_present:
        raise D2GraphProofGateError(
            "D2 legacy graph proof unexpectedly contains partial v3 coverage"
        )
    for field in D2_V3_COVERAGE_FIELDS:
        if field in comparison:
            stable_comparison[field] = comparison[field]
    locality_fields_present = [
        field for field in D2_BFS_LOCALITY_COMPARISON_FIELDS if field in comparison
    ]
    if locality_fields_present and len(locality_fields_present) != len(
        D2_BFS_LOCALITY_COMPARISON_FIELDS
    ):
        raise D2GraphProofGateError(
            "D2 graph proof has only one side of the BFS locality comparison"
        )
    for field in D2_BFS_LOCALITY_COMPARISON_FIELDS:
        locality = comparison.get(field)
        if locality is not None:
            validate_d2_bfs_locality(locality, field)
            stable_comparison[field] = locality
    edge_fields_present = [
        field for field in D2_EDGE_SPAN_COMPARISON_FIELDS if field in comparison
    ]
    if edge_fields_present and len(edge_fields_present) != len(
        D2_EDGE_SPAN_COMPARISON_FIELDS
    ):
        raise D2GraphProofGateError(
            "D2 graph proof has only one side of the edge-span comparison"
        )
    for field in D2_EDGE_SPAN_COMPARISON_FIELDS:
        edge_span = comparison.get(field)
        if edge_span is not None:
            validate_d2_edge_span(edge_span, field)
            stable_comparison[field] = edge_span
    stable: dict[str, object] = {
        "proof_contract": (
            "sqlens_same_heap_same_logical_graph_physical_layout_v3"
            if stable_comparison["format"] == "sqlens-hnsw-compare-v3"
            else "sqlens_same_heap_same_logical_graph_physical_layout_v2"
        ),
        "source_index": proof.get("source_index"),
        "clone_index": proof.get("clone_index"),
        "relations": stable_relations,
        "comparison": stable_comparison,
    }
    fingerprints = proof.get("fingerprints")
    if fingerprints is not None:
        if not isinstance(fingerprints, dict):
            raise D2GraphProofGateError("D2 graph fingerprints are not a JSON object")
        stable_fingerprints: dict[str, object] = {}
        for role in D2_GRAPH_FINGERPRINT_ROLES:
            fingerprint = fingerprints.get(role)
            validate_d2_graph_fingerprint(fingerprint, role)
            stable_fingerprints[role] = fingerprint
        stable["fingerprints"] = stable_fingerprints
    return stable


def _d2_nonnegative_int(value: Mapping[str, object], name: str, label: str) -> int:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise D2GraphProofGateError(f"D2 {label} has invalid {name}")
    return item


def validate_d2_edge_span(value: object, field: str) -> None:
    """Validate complete full-graph physical edge-span statistics."""
    if not isinstance(value, dict):
        raise D2GraphProofGateError(f"D2 {field} is not a JSON object")
    required = (
        "format",
        "graph_nodes",
        "index_blocks",
        "source_page_scope",
        "target_page_scope",
        "edge_scope",
        "full_statistics",
        "all_layers",
        "level_zero",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise D2GraphProofGateError(f"D2 {field} is missing fields: {missing}")
    if value["format"] != "sqlens-hnsw-edge-span-v1":
        raise D2GraphProofGateError(f"D2 {field} has an unsupported format")
    if value["full_statistics"] is not True:
        raise D2GraphProofGateError(f"D2 {field} is not a complete statistic")
    if value["source_page_scope"] != "owner_neighbor_tuple_page" or value[
        "target_page_scope"
    ] != "destination_element_page":
        raise D2GraphProofGateError(f"D2 {field} has an unsupported page scope")
    if value["edge_scope"] != "complete_directed_adjacency_with_level_duplicates":
        raise D2GraphProofGateError(f"D2 {field} has an unsupported edge scope")
    graph_nodes = _d2_nonnegative_int(value, "graph_nodes", field)
    index_blocks = _d2_nonnegative_int(value, "index_blocks", field)
    if graph_nodes <= 0 or index_blocks <= 0:
        raise D2GraphProofGateError(f"D2 {field} has an empty graph or index")

    edge_counts: dict[str, int] = {}
    for band in ("all_layers", "level_zero"):
        stats = value.get(band)
        if not isinstance(stats, dict):
            raise D2GraphProofGateError(f"D2 {field}.{band} is not a JSON object")
        int_fields = (
            "directed_edges",
            "same_page_edges",
            "within_1_page_edges",
            "within_4_pages_edges",
            "within_16_pages_edges",
            "p50_abs_block_delta",
            "p95_abs_block_delta",
            "p99_abs_block_delta",
            "max_abs_block_delta",
        )
        missing_stats = [name for name in int_fields if name not in stats]
        ratio_fields = (
            "same_page_ratio",
            "within_1_page_ratio",
            "within_4_pages_ratio",
            "within_16_pages_ratio",
            "mean_abs_block_delta",
        )
        missing_stats.extend(name for name in ratio_fields if name not in stats)
        if missing_stats:
            raise D2GraphProofGateError(
                f"D2 {field}.{band} is missing fields: {missing_stats}"
            )
        counts = {name: _d2_nonnegative_int(stats, name, f"{field}.{band}") for name in int_fields}
        directed = counts["directed_edges"]
        if directed <= 0:
            raise D2GraphProofGateError(f"D2 {field}.{band} has no graph edges")
        ordered_counts = (
            counts["same_page_edges"],
            counts["within_1_page_edges"],
            counts["within_4_pages_edges"],
            counts["within_16_pages_edges"],
            directed,
        )
        if list(ordered_counts) != sorted(ordered_counts):
            raise D2GraphProofGateError(f"D2 {field}.{band} edge bands disagree")
        percentiles = (
            counts["p50_abs_block_delta"],
            counts["p95_abs_block_delta"],
            counts["p99_abs_block_delta"],
            counts["max_abs_block_delta"],
        )
        if list(percentiles) != sorted(percentiles) or percentiles[-1] >= index_blocks:
            raise D2GraphProofGateError(f"D2 {field}.{band} percentiles disagree")
        for ratio_name, numerator in zip(ratio_fields[:4], ordered_counts[:4]):
            ratio = stats[ratio_name]
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(float(ratio))
                or not math.isclose(
                    float(ratio), numerator / directed, rel_tol=1e-15, abs_tol=1e-15
                )
            ):
                raise D2GraphProofGateError(
                    f"D2 {field}.{band} has invalid {ratio_name}"
                )
        mean = stats["mean_abs_block_delta"]
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(float(mean)) or float(mean) < 0 or float(mean) > counts["max_abs_block_delta"]:
            raise D2GraphProofGateError(
                f"D2 {field}.{band} has invalid mean_abs_block_delta"
            )
        edge_counts[band] = directed
    if edge_counts["all_layers"] < edge_counts["level_zero"]:
        raise D2GraphProofGateError(f"D2 {field} all-layer edge count is incomplete")


def validate_d2_graph_fingerprint(value: object, role: str) -> None:
    if not isinstance(value, dict):
        raise D2GraphProofGateError(f"D2 {role} graph fingerprint is not a JSON object")
    required = (
        "format",
        "definition_digest",
        "tuple_coverage_digest",
        "logical_digest",
        "physical_digest",
        "nodes",
        "heap_tids",
        "tombstones",
        "edge_span",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise D2GraphProofGateError(
            f"D2 {role} graph fingerprint is missing fields: {missing}"
        )
    if value["format"] != "sqlens-hnsw-graph-v3":
        raise D2GraphProofGateError(f"D2 {role} graph fingerprint format is unsupported")
    for field in ("definition_digest", "tuple_coverage_digest", "logical_digest", "physical_digest"):
        digest = str(value[field])
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise D2GraphProofGateError(
                f"D2 {role} graph fingerprint has an invalid {field}"
            )
    nodes = _d2_nonnegative_int(value, "nodes", f"{role} graph fingerprint")
    heap_tids = _d2_nonnegative_int(value, "heap_tids", f"{role} graph fingerprint")
    if nodes <= 0 or heap_tids <= 0:
        raise D2GraphProofGateError(f"D2 {role} graph fingerprint is empty")
    _d2_nonnegative_int(value, "tombstones", f"{role} graph fingerprint")
    validate_d2_edge_span(value["edge_span"], f"{role}.edge_span")
    if int(value["edge_span"]["graph_nodes"]) != nodes:
        raise D2GraphProofGateError(
            f"D2 {role} graph fingerprint edge-span node count differs"
        )


def validate_d2_bfs_locality(value: object, field: str) -> None:
    """Validate the C proof's complete counters and bounded rank evidence."""
    if not isinstance(value, dict):
        raise D2GraphProofGateError(f"D2 {field} locality is not a JSON object")
    required = (
        "format",
        "rank_base",
        "graph_nodes",
        "reachable_nodes",
        "fallback_nodes",
        "sequence_nodes",
        "adjacent_pairs",
        "same_block_pairs",
        "next_block_pairs",
        "same_or_next_page_pairs",
        "nondecreasing_pairs",
        "backward_pairs",
        "total_abs_block_delta",
        "max_abs_block_delta",
        "page_runs",
        "same_block_ratio",
        "same_or_next_page_ratio",
        "nondecreasing_ratio",
        "full_statistics",
        "sample_limit",
        "sample_count",
        "sample_truncated",
        "sample_strategy",
        "rank_samples",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise D2GraphProofGateError(
            f"D2 {field} locality is missing fields: {missing}"
        )
    if value["format"] != "sqlens-hnsw-bfs-locality-v1":
        raise D2GraphProofGateError(f"D2 {field} locality has an unsupported format")
    if value["rank_base"] != 0 or value["full_statistics"] is not True:
        raise D2GraphProofGateError(
            f"D2 {field} locality must use zero-based complete statistics"
        )

    def nonnegative_int(name: str) -> int:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise D2GraphProofGateError(f"D2 {field} locality has invalid {name}")
        return item

    graph_nodes = nonnegative_int("graph_nodes")
    reachable_nodes = nonnegative_int("reachable_nodes")
    fallback_nodes = nonnegative_int("fallback_nodes")
    sequence_nodes = nonnegative_int("sequence_nodes")
    adjacent_pairs = nonnegative_int("adjacent_pairs")
    same_block_pairs = nonnegative_int("same_block_pairs")
    next_block_pairs = nonnegative_int("next_block_pairs")
    same_or_next_pairs = nonnegative_int("same_or_next_page_pairs")
    nondecreasing_pairs = nonnegative_int("nondecreasing_pairs")
    backward_pairs = nonnegative_int("backward_pairs")
    total_abs_block_delta = nonnegative_int("total_abs_block_delta")
    max_abs_block_delta = nonnegative_int("max_abs_block_delta")
    page_runs = nonnegative_int("page_runs")
    sample_limit = nonnegative_int("sample_limit")
    sample_count = nonnegative_int("sample_count")
    if graph_nodes != sequence_nodes or reachable_nodes + fallback_nodes != sequence_nodes:
        raise D2GraphProofGateError(f"D2 {field} locality sequence coverage is incomplete")
    if adjacent_pairs != max(sequence_nodes - 1, 0):
        raise D2GraphProofGateError(f"D2 {field} locality adjacent-pair count is invalid")
    if same_block_pairs + next_block_pairs != same_or_next_pairs:
        raise D2GraphProofGateError(f"D2 {field} locality same/next counters disagree")
    if same_or_next_pairs > adjacent_pairs:
        raise D2GraphProofGateError(f"D2 {field} locality has too many same/next pairs")
    if nondecreasing_pairs + backward_pairs != adjacent_pairs:
        raise D2GraphProofGateError(f"D2 {field} locality monotonicity counters disagree")
    if page_runs != sequence_nodes - same_block_pairs:
        raise D2GraphProofGateError(f"D2 {field} locality page-run count is invalid")
    if next_block_pairs > nondecreasing_pairs:
        raise D2GraphProofGateError(f"D2 {field} locality forward counters disagree")
    if max_abs_block_delta > total_abs_block_delta:
        raise D2GraphProofGateError(f"D2 {field} locality block-delta counters disagree")
    if adjacent_pairs == same_block_pairs and (
        total_abs_block_delta != 0 or max_abs_block_delta != 0
    ):
        raise D2GraphProofGateError(f"D2 {field} locality zero-delta counters disagree")
    if sample_limit != 256 or sample_count != min(sample_limit, sequence_nodes):
        raise D2GraphProofGateError(f"D2 {field} locality sample bound is invalid")
    if value["sample_truncated"] is not (sample_count < sequence_nodes):
        raise D2GraphProofGateError(f"D2 {field} locality sample truncation is invalid")
    if value["sample_strategy"] != "evenly_spaced_inclusive":
        raise D2GraphProofGateError(f"D2 {field} locality sample strategy is invalid")
    samples = value["rank_samples"]
    if not isinstance(samples, list) or len(samples) != sample_count:
        raise D2GraphProofGateError(f"D2 {field} locality rank samples are incomplete")
    previous_rank = -1
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise D2GraphProofGateError(f"D2 {field} locality has an invalid rank sample")
        rank = sample.get("rank")
        block = sample.get("block")
        offset = sample.get("offset")
        expected_rank = (
            0
            if sample_count == 1
            else sample_index * (sequence_nodes - 1) // (sample_count - 1)
        )
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= previous_rank
            or rank < 0
            or rank >= sequence_nodes
            or rank != expected_rank
            or isinstance(block, bool)
            or not isinstance(block, int)
            or block < 0
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset <= 0
        ):
            raise D2GraphProofGateError(f"D2 {field} locality has invalid rank samples")
        previous_rank = rank
    if sample_count and (samples[0]["rank"] != 0 or samples[-1]["rank"] != sequence_nodes - 1):
        raise D2GraphProofGateError(f"D2 {field} locality samples do not cover sequence ends")
    ratio_counters = {
        "same_block_ratio": same_block_pairs,
        "same_or_next_page_ratio": same_or_next_pairs,
        "nondecreasing_ratio": nondecreasing_pairs,
    }
    denominator = adjacent_pairs if adjacent_pairs else 1
    for ratio_name, numerator in ratio_counters.items():
        ratio = value[ratio_name]
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 <= ratio <= 1
            or not math.isclose(
                float(ratio), numerator / denominator, rel_tol=1e-15, abs_tol=1e-15
            )
        ):
            raise D2GraphProofGateError(f"D2 {field} locality has invalid {ratio_name}")


def d2_stable_fingerprint(proof: dict[str, object]) -> str:
    encoded = json.dumps(
        stable_d2_graph_proof(proof), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_d2_graph_proof(
    proof: dict[str, object],
    source_index: str,
    clone_index: str,
    expected_heap_tids: int | None = None,
) -> dict[str, object]:
    if source_index == clone_index:
        raise D2GraphProofGateError("D2 source and clone index must be different relations")
    comparison_value = proof.get("comparison", proof)
    if not isinstance(comparison_value, dict):
        raise D2GraphProofGateError("D2 graph proof comparison is not a JSON object")
    missing = [field for field in D2_GRAPH_PROOF_FIELDS if field not in comparison_value]
    if missing:
        raise D2GraphProofGateError(f"D2 graph proof is missing fields: {missing}")
    required_true = (
        "same_heap",
        "logical_equal",
        "entry_equal",
        "definition_equal",
        "tuple_coverage_equal",
    )
    failed = [field for field in required_true if comparison_value.get(field) is not True]
    if failed:
        raise D2GraphProofGateError(
            "D2 graph proof failed required equivalence checks: " + ", ".join(failed)
        )
    if comparison_value.get("physical_equal") is not False:
        raise D2GraphProofGateError(
            "D2 graph proof is not a meaningful layout experiment: physical_equal must be false"
        )
    expected_source = proof.get("source_index")
    expected_clone = proof.get("clone_index")
    if expected_source is not None and str(expected_source) != source_index:
        raise D2GraphProofGateError(
            f"D2 proof source index mismatch: proof={expected_source!r}, requested={source_index!r}"
        )
    if expected_clone is not None and str(expected_clone) != clone_index:
        raise D2GraphProofGateError(
            f"D2 proof clone index mismatch: proof={expected_clone!r}, requested={clone_index!r}"
        )
    stable = stable_d2_graph_proof(
        {
            **proof,
            "source_index": source_index,
            "clone_index": clone_index,
            "comparison": comparison_value,
        }
    )
    stable_comparison = stable["comparison"]
    if stable_comparison["format"] not in {
        "sqlens-hnsw-compare-v2",
        "sqlens-hnsw-compare-v3",
    }:
        raise D2GraphProofGateError("D2 graph proof has an unsupported comparison format")
    for field in D2_STABLE_COMPARISON_FIELDS[7:]:
        digest = str(stable_comparison[field])
        if (
            not digest.startswith("sha256:")
            or len(digest) != 71
            or any(char not in "0123456789abcdef" for char in digest[7:])
        ):
            raise D2GraphProofGateError(f"D2 graph proof has an invalid digest in {field}")
    for left, right in (
        ("left_definition_digest", "right_definition_digest"),
        ("left_tuple_coverage_digest", "right_tuple_coverage_digest"),
        ("left_logical_digest", "right_logical_digest"),
    ):
        if stable_comparison[left] != stable_comparison[right]:
            raise D2GraphProofGateError(f"D2 equal graph proof has mismatched {left}/{right}")
    if stable_comparison["left_physical_digest"] == stable_comparison["right_physical_digest"]:
        raise D2GraphProofGateError("D2 physical digests are equal despite physical_equal=false")
    source_relation = stable["relations"]["source"]
    clone_relation = stable["relations"]["clone"]
    if source_relation["name"] != source_index or clone_relation["name"] != clone_index:
        raise D2GraphProofGateError("D2 relation identity names do not match requested indexes")
    for role, relation in (("source", source_relation), ("clone", clone_relation)):
        for field in ("oid", "relfilenode", "heap_oid"):
            if int(relation[field]) <= 0:
                raise D2GraphProofGateError(
                    f"D2 {role} relation identity has invalid {field}={relation[field]!r}"
                )
    if int(source_relation["heap_oid"]) != int(clone_relation["heap_oid"]):
        raise D2GraphProofGateError("D2 source and clone relation identities do not share a heap")
    if stable_comparison["format"] == "sqlens-hnsw-compare-v3":
        for left_field, right_field in (
            ("left_nodes", "right_nodes"),
            ("left_heap_tids", "right_heap_tids"),
            ("left_tombstones", "right_tombstones"),
        ):
            left_value = stable_comparison[left_field]
            right_value = stable_comparison[right_field]
            if (
                isinstance(left_value, bool)
                or not isinstance(left_value, int)
                or left_value < 0
                or isinstance(right_value, bool)
                or not isinstance(right_value, int)
                or right_value < 0
            ):
                raise D2GraphProofGateError(
                    f"D2 graph comparison has invalid {left_field}/{right_field}"
                )
            if left_value != right_value:
                raise D2GraphProofGateError(
                    f"D2 source/clone graph comparison disagrees on {left_field}/{right_field}"
                )
        if int(stable_comparison["left_nodes"]) <= 0 or int(
            stable_comparison["left_heap_tids"]
        ) <= 0:
            raise D2GraphProofGateError("D2 graph comparison has empty tuple coverage")
        if int(stable_comparison["left_tombstones"]) != 0:
            raise D2GraphProofGateError(
                "D2 formal layout control requires zero graph tombstones"
            )
        if expected_heap_tids is not None and int(
            stable_comparison["left_heap_tids"]
        ) != int(expected_heap_tids):
            raise D2GraphProofGateError(
                "D2 graph tuple coverage differs from the expected candidate rows: "
                f"expected={expected_heap_tids}, observed={stable_comparison['left_heap_tids']}"
            )
        left_edge = stable_comparison.get("left_edge_span")
        right_edge = stable_comparison.get("right_edge_span")
        if not isinstance(left_edge, dict) or not isinstance(right_edge, dict):
            raise D2GraphProofGateError("D2 v3 proof requires both edge-span summaries")
        if int(left_edge["graph_nodes"]) != int(stable_comparison["left_nodes"]) or int(
            right_edge["graph_nodes"]
        ) != int(stable_comparison["right_nodes"]):
            raise D2GraphProofGateError(
                "D2 edge-span graph coverage does not match the comparison"
            )
    elif expected_heap_tids is not None:
        raise D2GraphProofGateError(
            "D2 tuple-coverage gate requires a v3 graph proof"
        )
    stable_fingerprint = d2_stable_fingerprint(stable)
    delegated_fingerprint = proof.get("stable_fingerprint_sha256")
    if delegated_fingerprint is not None and str(delegated_fingerprint) != stable_fingerprint:
        raise D2GraphProofGateError("D2 delegated stable fingerprint does not match proof fields")
    return {
        **stable,
        "checked_at": proof.get("checked_at") or utc_now(),
        "stable_fingerprint_sha256": stable_fingerprint,
    }


def require_d2_graph_proof(
    cur: psycopg.Cursor,
    source_index: str,
    clone_index: str,
    expected_heap_tids: int | None = None,
) -> dict[str, object]:
    try:
        cur.execute(
            "SELECT vector_hnsw_graph_compare(%s::regclass, %s::regclass), "
            "source.oid::bigint, source.relfilenode::bigint, source_index.indrelid::bigint, "
            "clone.oid::bigint, clone.relfilenode::bigint, clone_index.indrelid::bigint "
            "FROM pg_class source "
            "JOIN pg_index source_index ON source_index.indexrelid = source.oid "
            "JOIN pg_class clone ON clone.oid = %s::regclass "
            "JOIN pg_index clone_index ON clone_index.indexrelid = clone.oid "
            "WHERE source.oid = %s::regclass",
            (
                source_index,
                clone_index,
                clone_index,
                source_index,
            ),
        )
        row = cur.fetchone()
        raw = row[0] if row else None
        comparison = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:  # noqa: BLE001 - a formal D2 gate must fail closed
        raise D2GraphProofGateError(
            "D2 graph proof gate failed: vector_hnsw_graph_compare(source, clone) is unavailable "
            "or could not fingerprint both indexes"
        ) from exc
    if not isinstance(comparison, dict):
        raise D2GraphProofGateError(
            "D2 graph proof gate failed: vector_hnsw_graph_compare() did not return a JSON object"
        )
    return validate_d2_graph_proof(
        {
            "checked_at": utc_now(),
            "source_index": source_index,
            "clone_index": clone_index,
            "relations": {
                "source": {
                    "name": source_index,
                    "oid": row[1],
                    "relfilenode": row[2],
                    "heap_oid": row[3],
                },
                "clone": {
                    "name": clone_index,
                    "oid": row[4],
                    "relfilenode": row[5],
                    "heap_oid": row[6],
                },
            },
            "comparison": comparison,
        },
        source_index,
        clone_index,
        expected_heap_tids=expected_heap_tids,
    )


def require_d2_relation_identity(
    cur: psycopg.Cursor,
    source_index: str,
    clone_index: str,
) -> dict[str, dict[str, object]]:
    cur.execute(
        "SELECT source.oid::bigint, source.relfilenode::bigint, source_index.indrelid::bigint, "
        "source_index.indisvalid, source_index.indisready, source_index.indislive, "
        "clone.oid::bigint, clone.relfilenode::bigint, clone_index.indrelid::bigint, "
        "clone_index.indisvalid, clone_index.indisready, clone_index.indislive "
        "FROM pg_class source "
        "JOIN pg_index source_index ON source_index.indexrelid = source.oid "
        "JOIN pg_class clone ON clone.oid = %s::regclass "
        "JOIN pg_index clone_index ON clone_index.indexrelid = clone.oid "
        "WHERE source.oid = %s::regclass",
        (clone_index, source_index),
    )
    row = cur.fetchone()
    if row is None:
        raise D2GraphProofGateError("D2 source or clone index identity is unavailable")
    relations = {
        "source": {
            "name": source_index,
            "oid": int(row[0]),
            "relfilenode": int(row[1]),
            "heap_oid": int(row[2]),
            "indisvalid": bool(row[3]),
            "indisready": bool(row[4]),
            "indislive": bool(row[5]),
        },
        "clone": {
            "name": clone_index,
            "oid": int(row[6]),
            "relfilenode": int(row[7]),
            "heap_oid": int(row[8]),
            "indisvalid": bool(row[9]),
            "indisready": bool(row[10]),
            "indislive": bool(row[11]),
        },
    }
    if relations["source"]["heap_oid"] != relations["clone"]["heap_oid"]:
        raise D2GraphProofGateError("D2 source and clone no longer share the same heap")
    for role, relation in relations.items():
        if not all(bool(relation[field]) for field in ("indisvalid", "indisready", "indislive")):
            raise D2GraphProofGateError(f"D2 {role} index is no longer valid, ready, and live")
    return relations


def require_d2_graph_proof_from_env(
    args: argparse.Namespace,
    delegated_proof: dict[str, object] | None = None,
) -> dict[str, object]:
    delegated = (
        validate_d2_graph_proof(
            delegated_proof,
            args.insertion_index,
            args.bfs_index,
        )
        if delegated_proof is not None
        else None
    )
    conn = psycopg.connect(pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = conn.cursor()
        try:
            ensure_functions(cur)
            if delegated is None:
                live = require_d2_graph_proof(cur, args.insertion_index, args.bfs_index)
                live["live_revalidated"] = True
                live["full_graph_fingerprint_recomputed"] = True
                return live
            live_relations = require_d2_relation_identity(
                cur, args.insertion_index, args.bfs_index
            )
            delegated_relations = delegated["relations"]
            for role in ("source", "clone"):
                for field in D2_RELATION_IDENTITY_FIELDS:
                    if live_relations[role][field] != delegated_relations[role][field]:
                        raise D2GraphProofGateError(
                            "D2 live revalidation changed delegated relation identity: "
                            f"{role}.{field}"
                        )
            return {
                **delegated,
                "delegated_checked_at": delegated.get("checked_at"),
                "live_identity_checked_at": utc_now(),
                "live_relation_identity": live_relations,
                "live_revalidated": True,
                "full_graph_fingerprint_recomputed": False,
            }
        finally:
            cur.close()
    finally:
        conn.close()


def require_sqlens_provenance_from_env() -> tuple[str, dict[str, Any]]:
    """Run the formal-entry gate on a short-lived connection before any wrapper DDL."""
    conn = psycopg.connect(pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = conn.cursor()
        try:
            return require_sqlens_provenance(cur)
        finally:
            cur.close()
    finally:
        conn.close()


def require_exact_sqlens_identity_from_env(
    expected_build_id: str,
    expected_vector_so_sha256: str,
) -> dict[str, object]:
    conn = psycopg.connect(pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = conn.cursor()
        try:
            return require_exact_sqlens_identity(
                cur,
                expected_build_id,
                expected_vector_so_sha256,
            )
        finally:
            cur.close()
    finally:
        conn.close()


def ensure_functions(cur: psycopg.Cursor) -> None:
    require_sqlens_provenance(cur)
    functions = [
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_activate(regclass, text[], text) "
        "RETURNS int4 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_bind(regclass, text[], text) "
        "RETURNS boolean AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_sqlens_build_id() "
        "RETURNS text AS 'vector' LANGUAGE C IMMUTABLE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_fragment_epoch_bump_trigger() "
        "RETURNS trigger AS 'vector' LANGUAGE C SECURITY DEFINER "
        "SET search_path = pg_catalog, pg_temp",
        "CREATE OR REPLACE FUNCTION vector_hnsw_fragment_tracking_enable(regclass) "
        "RETURNS int8 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_last_scan_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_reset_scan_profile() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_graph_compare(regclass, regclass) "
        "RETURNS jsonb AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
    ]
    cur.execute(
        "SELECT pg_catalog.pg_advisory_lock(%s)",
        (SQLENS_WRAPPER_DDL_ADVISORY_LOCK,),
    )
    try:
        for sql in functions:
            try:
                cur.execute(sql)
            except Exception as exc:  # noqa: BLE001 - tolerate pre-lock runners
                if "tuple concurrently updated" not in str(exc):
                    raise
                cur.connection.rollback()
        cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    finally:
        cur.execute(
            "SELECT pg_catalog.pg_advisory_unlock(%s)",
            (SQLENS_WRAPPER_DDL_ADVISORY_LOCK,),
        )


def ensure_tracking(cur: psycopg.Cursor, *tables: str) -> None:
    cur.execute(
        "SELECT p.oid IS NOT NULL, p.prosecdef, "
        "COALESCE(p.proconfig @> ARRAY['search_path=pg_catalog, pg_temp']::text[], false), "
        "EXISTS ("
        "SELECT 1 FROM pg_catalog.pg_depend AS d "
        "JOIN pg_catalog.pg_extension AS x ON x.oid = d.refobjid "
        "WHERE d.classid = 'pg_catalog.pg_proc'::pg_catalog.regclass "
        "AND d.objid = p.oid "
        "AND d.refclassid = 'pg_catalog.pg_extension'::pg_catalog.regclass "
        "AND d.deptype = 'e' AND x.extname = 'vector'"
        ") "
        "FROM pg_catalog.pg_proc AS p "
        "WHERE p.oid = pg_catalog.to_regprocedure("
        "'vector_hnsw_fragment_epoch_bump_trigger()')"
    )
    contract = cur.fetchone()
    if contract != (True, True, True, True):
        raise RuntimeError(
            "SQLens fragment tracking trigger is not an extension-owned SECURITY DEFINER "
            "function with a fixed pg_catalog, pg_temp search_path; install the complete "
            "SQLens extension SQL before running formal experiments"
        )
    for table in dict.fromkeys(tables):
        cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", (table,))


def mode_uses_d2(mode: str) -> bool:
    return mode in {"design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3"}


def mode_uses_guidance(mode: str) -> bool:
    return mode != "original"


def configure(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    cache_mb: int,
    mode: str = "original",
    mode_config: dict[str, object] | None = None,
) -> None:
    config = mode_config or effective_mode_config(args, mode)
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(config['ef_search'])}")
    cur.execute(f"SET hnsw.iterative_scan = {config['iterative_scan']}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(config['max_scan_tuples'])}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(config['scan_mem_multiplier'])}")
    cur.execute(f"SET hnsw.guided_collect_target = {int(config['guided_collect_target'])}")
    cur.execute(f"SET hnsw.traversal_guided_target = {int(config['traversal_guided_target'])}")
    prioritization = bool(config["traversal_guided_prioritization"]) and mode_uses_guidance(mode)
    cur.execute(
        "SET hnsw.traversal_guided_prioritization = "
        + ("on" if prioritization else "off")
    )
    cur.execute(f"SET hnsw.traversal_guided_burst = {int(config['traversal_guided_burst'])}")
    cur.execute(
        "SET hnsw.traversal_guided_early_stop = "
        + ("on" if bool(config["traversal_guided_early_stop"]) else "off")
    )
    cur.execute(
        "SET hnsw.traversal_guided_early_stop_distance_ratio = "
        f"{float(config['traversal_guided_early_stop_distance_ratio'])}"
    )
    cur.execute(f"SET hnsw.metadata_cache_max_mb = {int(cache_mb)}")
    if mode == "design1_bloom_bfs_layout_d3":
        cur.execute(
            "SELECT set_config('hnsw.fragment_store_namespace', %s, false)",
            (str(getattr(args, "d3_fragment_store_namespace", "")),),
        )
        cur.execute(
            f"SET hnsw.d3_probe_requests = {int(getattr(args, 'd3_probe_requests', 2))}"
        )
        cur.execute(
            "SET hnsw.d3_min_benefit_per_byte = "
            f"{float(getattr(args, 'd3_min_benefit_per_byte', 0.0))}"
        )
        cur.execute(
            f"SET hnsw.d3_max_fragment_mb = {int(getattr(args, 'd3_max_fragment_mb', 16))}"
        )
        cur.execute(
            "SET hnsw.d3_page_min_skip_rate = "
            f"{float(getattr(args, 'd3_page_min_skip_rate', 0.05))}"
        )
    cur.execute(
        "SET hnsw.filter_strategy = "
        + (
            str(getattr(args, "guidance_filter_strategy", "traversal_guided"))
            if mode_uses_guidance(mode)
            else "off"
        )
    )
    cur.execute(f"SET hnsw.page_access = {args.d2_page_access if mode_uses_d2(mode) else 'off'}")
    cur.execute(f"SET hnsw.index_page_access = {args.d2_index_page_access if mode_uses_d2(mode) else 'off'}")
    cur.execute(f"SET hnsw.page_window = {int(args.d2_page_window)}")
    cur.execute(f"SET hnsw.page_prefetch_min_items = {int(args.d2_page_prefetch_min_items)}")
    cur.execute(f"SET hnsw.page_disable_after_no_merge = {int(args.d2_page_disable_after_no_merge)}")
    cur.execute("SET jit = off")
    if args.force_hnsw:
        cur.execute("SET enable_sort = off")


def mode_table_index(
    args: argparse.Namespace,
    mode: str,
    filter_name: str | None = None,
) -> tuple[str, str]:
    if mode in {"design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3"}:
        if (
            filter_name is not None
            and bool(getattr(args, "d2_source_on_guidance_bypass", False))
            and not should_enable_guidance(args, filter_name)[0]
        ):
            return (
                getattr(args, "insertion_table", INSERTION_TABLE),
                getattr(args, "insertion_index", INSERTION_INDEX),
            )
        return (
            getattr(args, "bfs_table", BFS_TABLE),
            getattr(args, "bfs_index", BFS_INDEX),
        )
    return (
        getattr(args, "insertion_table", INSERTION_TABLE),
        getattr(args, "insertion_index", INSERTION_INDEX),
    )


def set_preferred_index_if_supported(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    expected_index: str,
) -> str | None:
    guc = str(getattr(args, "preferred_index_guc", "hnsw.preferred_index"))
    cur.execute("SELECT current_setting(%s, true)", (guc,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    cur.execute("SELECT set_config(%s, %s, false)", (guc, expected_index))
    cur.execute(
        "SELECT current_setting(%s), current_setting(%s)::regclass = %s::regclass",
        (guc, guc, expected_index),
    )
    observed = cur.fetchone()
    if not observed or observed[0] is None or observed[1] is not True:
        raise RuntimeError(
            f"{guc} did not resolve to expected index {expected_index!r}: {observed!r}"
        )
    return str(observed[0])


def uses_exact_predicate_scan_contract(filter_strategy: str) -> bool:
    return filter_strategy == "traversal_guided"


def query_table_for_candidate(args: argparse.Namespace, candidate_table: str) -> str:
    """Use the candidate heap as the query source unless an external query heap is supplied."""
    return str(getattr(args, "query_table", None) or candidate_table)


def candidate_self_exclusion(args: argparse.Namespace, candidate_table: str) -> bool:
    return query_table_for_candidate(args, candidate_table) == candidate_table


def validate_query_source_contract(args: argparse.Namespace) -> None:
    observed = {
        candidate_self_exclusion(args, mode_table_index(args, mode)[0])
        for mode in args.modes
    }
    if len(observed) != 1:
        raise RuntimeError(
            "query source has inconsistent self-exclusion semantics across candidate tables; "
            "supply a query table that is either external to every mode or the candidate table"
        )
    actual = observed.pop()
    if actual != args.expected_truth_self_excluded:
        raise RuntimeError(
            "query/truth self-exclusion contract mismatch: "
            f"candidate_self_excluded={actual!r}, "
            f"expected_truth_self_excluded={args.expected_truth_self_excluded!r}"
        )


def quoted_column(identifier: str) -> str:
    if not identifier or "." in identifier or "\x00" in identifier:
        raise ValueError(f"invalid column identifier: {identifier!r}")
    return '"' + identifier.replace('"', '""') + '"'


def search_query_sql(
    table: str,
    predicate: str,
    k: int,
    bind_guidance: bool = False,
    client_self_exclusion: bool = False,
    *,
    candidate_validity_predicate: str = "",
    query_table: str | None = None,
    query_id_column: str = "id",
    query_vector_column: str = "embedding",
    self_exclusion: bool = True,
) -> str:
    binding = (
        "(SELECT vector_hnsw_guidance_bind(%s::regclass, %s::text[], %s) OFFSET 0) AND "
        if bind_guidance
        else ""
    )
    source_table = query_table or table
    query_id = quoted_column(query_id_column)
    query_vector = quoted_column(query_vector_column)
    validity_predicate = effective_candidate_validity_predicate(
        candidate_validity_predicate
    )
    self_qual = "" if client_self_exclusion or not self_exclusion else " AND id <> %s"
    scan_limit = int(k) + 1 if client_self_exclusion else int(k)
    return f"""
        SELECT id,
               embedding <-> (
                   SELECT q.{query_vector}
                   FROM {source_table} AS q
                   WHERE q.{query_id} = %s
               ) AS distance
        FROM {table}
        WHERE {binding}({predicate}) AND ({validity_predicate}){self_qual}
        ORDER BY distance
        LIMIT {scan_limit}
    """


def plan_index_nodes(value: object) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if isinstance(value, dict):
        if "Index Name" in value:
            found.append(
                {
                    key: value.get(key)
                    for key in ("Node Type", "Index Name", "Schema", "Relation Name", "Alias")
                }
            )
        for child in value.values():
            found.extend(plan_index_nodes(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(plan_index_nodes(child))
    return found


def explain_hnsw_plan(
    cur: psycopg.Cursor,
    table: str,
    expected_index: str,
    predicate: str,
    query_id: int,
    k: int,
    binding: tuple[str, list[str], str] | None = None,
    client_self_exclusion: bool = False,
    *,
    candidate_validity_predicate: str = "",
    query_table: str | None = None,
    query_id_column: str = "id",
    query_vector_column: str = "embedding",
    self_exclusion: bool = True,
) -> dict[str, object]:
    cur.execute(
        "SELECT idx.oid::bigint, idx.relname, idx_ns.nspname, am.amname, "
        "tbl.oid::bigint, tbl.relname, tbl_ns.nspname "
        ", pg_get_expr(ix.indpred, ix.indrelid) "
        "FROM pg_class idx "
        "JOIN pg_namespace idx_ns ON idx_ns.oid = idx.relnamespace "
        "JOIN pg_am am ON am.oid = idx.relam "
        "JOIN pg_index ix ON ix.indexrelid = idx.oid "
        "JOIN pg_class tbl ON tbl.oid = ix.indrelid "
        "JOIN pg_namespace tbl_ns ON tbl_ns.oid = tbl.relnamespace "
        "WHERE idx.oid = to_regclass(%s) AND tbl.oid = to_regclass(%s)",
        (expected_index, table),
    )
    metadata = cur.fetchone()
    if metadata is None:
        return {
            "passed": False,
            "expected_index": expected_index,
            "expected_table": table,
            "failure": "expected index/table metadata not found",
            "observed_index_nodes": [],
        }

    if len(metadata) < 8:
        raise RuntimeError("index catalog metadata is missing pg_index.indpred")
    (
        index_oid,
        index_name,
        index_schema,
        access_method,
        table_oid,
        table_name,
        table_schema,
        catalog_predicate,
    ) = metadata
    sql = search_query_sql(
        table,
        predicate,
        k,
        binding is not None,
        client_self_exclusion,
        candidate_validity_predicate=candidate_validity_predicate,
        query_table=query_table,
        query_id_column=query_id_column,
        query_vector_column=query_vector_column,
        self_exclusion=self_exclusion,
    )
    params: tuple[object, ...] = (int(query_id),)
    if binding is not None:
        params += binding
    if self_exclusion and not client_self_exclusion:
        params += (int(query_id),)
    cur.execute("EXPLAIN (FORMAT JSON, VERBOSE) " + sql, params)
    explain_value: Any = cur.fetchone()[0]
    if isinstance(explain_value, str):
        explain_value = json.loads(explain_value)
    observed = plan_index_nodes(explain_value)
    matched = [
        node
        for node in observed
        if node.get("Node Type") in {"Index Scan", "Index Only Scan"}
        and node.get("Index Name") == index_name
        and node.get("Relation Name") == table_name
        and node.get("Schema") == table_schema
    ]
    expected_predicate = effective_candidate_validity_predicate(
        candidate_validity_predicate
    )
    expected_is_partial = not candidate_validity_index_predicate_matches(
        None, expected_predicate
    )
    predicate_matches = candidate_validity_index_predicate_matches(
        catalog_predicate, expected_predicate
    )
    passed = access_method == "hnsw" and bool(matched) and predicate_matches
    if access_method != "hnsw" or not matched:
        failure = "EXPLAIN did not use the expected HNSW index"
    elif not predicate_matches:
        failure = (
            "expected index pg_index.indpred does not match candidate validity predicate: "
            f"catalog={catalog_predicate!r}, expected={expected_predicate!r}"
        )
    else:
        failure = ""
    return {
        "passed": passed,
        "expected_index": expected_index,
        "expected_index_oid": index_oid,
        "expected_index_identity": f"{index_schema}.{index_name}",
        "expected_index_access_method": access_method,
        "expected_index_predicate": expected_predicate,
        "expected_index_predicate_sha256": candidate_validity_sha256(expected_predicate),
        "expected_index_is_partial": expected_is_partial,
        "catalog_index_oid": index_oid,
        "catalog_index_predicate": catalog_predicate,
        "catalog_index_predicate_sha256": (
            candidate_validity_sha256(catalog_predicate)
            if catalog_predicate is not None
            else candidate_validity_sha256("TRUE")
        ),
        "catalog_index_is_partial": catalog_predicate is not None,
        "catalog_index_predicate_matches": predicate_matches,
        "expected_table": table,
        "expected_table_oid": table_oid,
        "expected_table_identity": f"{table_schema}.{table_name}",
        "query_id": query_id,
        "query_table": query_table or table,
        "query_id_column": query_id_column,
        "query_vector_column": query_vector_column,
        "self_excluded": self_exclusion,
        "self_exclusion_contract": (
            "limit_k_plus_1_client_remove_query_id"
            if client_self_exclusion
            else "sql_residual_id_not_equal" if self_exclusion else "none_external_query_source"
        ),
        "scan_limit": int(k) + 1 if client_self_exclusion else int(k),
        "residual_self_qual_present": self_exclusion and not client_self_exclusion,
        "candidate_validity_predicate": effective_candidate_validity_predicate(
            candidate_validity_predicate
        ),
        "candidate_validity_predicate_sha256": candidate_validity_sha256(
            candidate_validity_predicate
        ),
        "statement_binding_present": binding is not None,
        "observed_index_nodes": observed,
        "matched_index_nodes": matched,
        "plan": explain_value,
        "failure": failure,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def acquire_database_experiment_lock() -> tuple[Any, dict[str, object]]:
    port = str(os.environ.get("PGPORT", "5432"))
    root = Path(__file__).resolve().parents[3]
    path = root / "results/hybrid_vector_db" / f".pg{port}_experiment.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle, {
        "enabled": True,
        "protocol": "fcntl_flock_exclusive_v1",
        "path": str(path),
        "pgport": port,
        "acquired_at": utc_now(),
        "pid": os.getpid(),
    }


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def prewarm_relations(relations: Sequence[str]) -> dict[str, object]:
    """Synchronously load relation main forks into the OS page cache."""
    records: list[dict[str, object]] = []
    if not relations:
        return {
            "enabled": False,
            "method": "pg_prewarm(regclass,'read','main')",
            "records": records,
            "complete": True,
        }
    if len(set(relations)) != len(relations):
        raise RuntimeError("prewarm relation list contains duplicates")

    conn = psycopg.connect(pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = conn.cursor()
        for relation in relations:
            cur.execute(
                "SELECT c.oid::bigint, c.relfilenode::bigint, "
                "pg_relation_size(c.oid)::bigint, "
                "current_setting('block_size')::bigint "
                "FROM pg_class c WHERE c.oid = %s::regclass",
                (relation,),
            )
            identity = cur.fetchone()
            if identity is None:
                raise RuntimeError(f"prewarm relation does not exist: {relation}")
            oid, relfilenode, relation_bytes, block_size = map(int, identity)
            expected_blocks = (
                (relation_bytes + block_size - 1) // block_size
                if relation_bytes
                else 0
            )
            started = time.perf_counter()
            cur.execute(
                "SELECT pg_prewarm(%s::regclass, 'read', 'main')::bigint",
                (relation,),
            )
            warmed = cur.fetchone()
            warmed_blocks = int(warmed[0]) if warmed else -1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if warmed_blocks != expected_blocks:
                raise RuntimeError(
                    f"pg_prewarm block count mismatch for {relation}: "
                    f"expected={expected_blocks}, observed={warmed_blocks}"
                )
            records.append(
                {
                    "relation": relation,
                    "oid": oid,
                    "relfilenode": relfilenode,
                    "relation_bytes": relation_bytes,
                    "block_size": block_size,
                    "expected_blocks": expected_blocks,
                    "warmed_blocks": warmed_blocks,
                    "elapsed_ms": elapsed_ms,
                }
            )
    finally:
        conn.close()
    return {
        "enabled": True,
        "method": "pg_prewarm(regclass,'read','main')",
        "cache_scope": "synchronous_os_page_cache_before_measured_runtimes",
        "records": records,
        "complete": len(records) == len(relations),
    }


def fragment_store_namespace_evidence(
    table: str,
    namespace: str,
) -> dict[str, object]:
    if not namespace:
        raise RuntimeError("formal D3 execution requires a fragment-store namespace")
    prefix = namespace + "\x1f"
    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT count(*) "
            "FROM public.pgvector_hnsw_fragment_store "
            "WHERE heap_oid = %s::regclass "
            "AND left(filter_name, char_length(%s)) = %s",
            (table, prefix, prefix),
        )
        row = cur.fetchone()
    count = int(row[0]) if row else -1
    return {
        "required_empty": True,
        "table": table,
        "namespace": namespace,
        "prefix_encoding": "namespace + ASCII unit separator",
        "rows_before": count,
        "empty": count == 0,
        "checked_at": utc_now(),
    }


def write_plan_evidence(
    args: argparse.Namespace,
    status: str,
    error: BaseException | None = None,
) -> None:
    path = args.plan_evidence_out
    payload = {
        "status": status,
        "started_at": args.plan_started_at,
        "completed_at": utc_now() if status in {"complete", "failed"} else None,
        "output": str(args.out),
        "output_rows": getattr(args, "output_rows", 0),
        "output_sha256": sha256_file(args.out),
        "d3_initialization": d3_initialization_label(args),
        "d3_fragment_store_namespace": str(
            getattr(args, "d3_fragment_store_namespace", "")
        ),
        "d3_fragment_store_start": getattr(
            args, "d3_fragment_store_start_evidence", None
        ),
        "repeat_runtime_isolation": bool(
            getattr(args, "isolate_repeat_runtimes", False)
        ),
        "d3_tuning": {
            "probe_requests": int(getattr(args, "d3_probe_requests", 2)),
            "min_benefit_per_byte": float(
                getattr(args, "d3_min_benefit_per_byte", 0.0)
            ),
            "max_fragment_mb": int(getattr(args, "d3_max_fragment_mb", 16)),
            "page_min_skip_rate": float(
                getattr(args, "d3_page_min_skip_rate", 0.05)
            ),
        },
        "prebuilt_fragments": 0,
        "relation_prewarm": getattr(
            args,
            "relation_prewarm_evidence",
            {
                "enabled": False,
                "method": "pg_prewarm(regclass,'read','main')",
                "records": [],
                "complete": True,
            },
        ),
        "warmup_all_queries": bool(getattr(args, "warmup_all_queries", False)),
        "warmup_evidence": getattr(args, "warmup_evidence", []),
        "execution_lifecycle": getattr(args, "execution_lifecycle", None),
        "query_error_summary": getattr(args, "query_error_summary", None),
        "database_experiment_lock": getattr(
            args, "database_experiment_lock_evidence", {"enabled": False}
        ),
        "guidance_filter_strategy": args.guidance_filter_strategy,
        "search_configuration": search_configuration_evidence(args),
        "query_contract": {
            "query_table": args.query_table or "candidate_table_per_mode",
            "query_id_column": args.query_id_column,
            "query_vector_column": args.query_vector_column,
            "workload_csv": str(getattr(args, "workload_csv", "") or ""),
            "workload_sha256": sha256_file(
                Path(str(getattr(args, "workload_csv", "") or ""))
            ),
            "truth_csv": str(args.truth_csv.resolve()),
            "truth_sha256": sha256_file(args.truth_csv.resolve()),
            "filters_csv": (
                str(args.filters_csv.resolve())
                if args.filters_csv is not None
                else ""
            ),
            "filters_sha256": (
                sha256_file(args.filters_csv.resolve())
                if args.filters_csv is not None
                else ""
            ),
            "d2_graph_proof_input_sha256": sha256_json(
                args.d2_graph_proof_json
            ),
            "expected_workload_requests": int(
                getattr(args, "expected_workload_requests", 0)
            ),
            "workload_request_limit": int(
                getattr(args, "workload_request_limit", 0)
            ),
            "workload_requests": int(
                getattr(args, "workload_request_count", 0)
            ),
            "workload_unique_queries": int(
                getattr(args, "workload_unique_query_count", 0)
            ),
            "require_unique_workload_queries": bool(
                getattr(args, "require_unique_workload_queries", False)
            ),
            "self_excluded": args.expected_truth_self_excluded,
            "candidate_validity_predicate": effective_candidate_validity_predicate(
                getattr(args, "candidate_validity_predicate", "")
            ),
            "candidate_validity_predicate_explicit": bool(
                getattr(args, "candidate_validity_predicate_explicit", False)
            ),
            "candidate_validity_predicate_sha256": candidate_validity_sha256(
                getattr(args, "candidate_validity_predicate", "")
            ),
            "candidate_validity_contract": (
                "planner_partial_index_predicate_and_sql_candidate_qual_not_guidance_atom"
            ),
            "predicate_contract": (
                "exact_activated_workload_predicate_plus_candidate_validity_sql_qual"
                if uses_exact_predicate_scan_contract(args.guidance_filter_strategy)
                else "diagnostic_workload_plus_candidate_validity_sql_quals"
            ),
            "self_exclusion": (
                "limit_k_plus_1_client_remove_query_id"
                if (
                    uses_exact_predicate_scan_contract(args.guidance_filter_strategy)
                    and args.expected_truth_self_excluded
                )
                else (
                    "sql_residual_id_not_equal"
                    if args.expected_truth_self_excluded
                    else "none_external_query_source"
                )
            ),
            "measured_latency_includes_client_self_exclusion": bool(
                uses_exact_predicate_scan_contract(args.guidance_filter_strategy)
                and args.expected_truth_self_excluded
            ),
        },
        "d2_graph_proof": getattr(args, "d2_graph_proof", {"required": False}),
        "d2_graph_proof_input": getattr(
            args,
            "d2_graph_proof_json",
            {},
        ),
        "d2_graph_proof_final": getattr(
            args, "d2_graph_proof_final", {"required": False}
        ),
        "sqlens_runtime_identity_startup": getattr(
            args, "sqlens_runtime_identity", None
        ),
        "sqlens_runtime_identity_final": getattr(
            args, "sqlens_runtime_identity_final", None
        ),
        "backend_cpu_evidence": getattr(args, "backend_cpu_evidence", []),
        "runtime_sqlens_identity_evidence": getattr(
            args, "runtime_sqlens_identity_evidence", []
        ),
        "execution_sources": getattr(args, "execution_source_evidence", {}),
        "checks": args.plan_evidence,
        "error": (
            {"type": error.__class__.__name__, "message": str(error)}
            if error is not None
            else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def should_enable_guidance(args: argparse.Namespace, filter_name: str) -> tuple[bool, str]:
    selectivity = float(
        getattr(args, "filter_selectivity_by_name", {}).get(filter_name, 100.0)
    )
    atom_count = len(getattr(args, "filter_atoms", {}).get(filter_name, []))
    minimum = float(getattr(args, "guidance_selectivity_min_pct", 0.0))
    if selectivity < minimum:
        return False, f"selectivity<{minimum:g}%"
    maximum = float(getattr(args, "guidance_selectivity_max_pct", 100.0))
    if selectivity > maximum:
        return False, f"selectivity>{maximum:g}%"
    composite_limit = float(
        getattr(args, "guidance_composite_max_selectivity_pct", 100.0)
    )
    if atom_count > 1 and selectivity > composite_limit:
        return False, f"composite_selectivity>{composite_limit:g}%"
    maximum_atoms = int(getattr(args, "guidance_max_atoms", 64))
    if atom_count > maximum_atoms:
        return False, f"atoms>{maximum_atoms}"
    return True, "enabled"


def bypass_ef_search_for_filter(
    args: argparse.Namespace,
    filter_name: str,
    configured_ef_search: int,
) -> int:
    selectivity = float(args.filter_selectivity_by_name.get(filter_name, 100.0))
    minimum = float(getattr(args, "guidance_selectivity_min_pct", 0.0))
    if minimum > 0.0 and selectivity < minimum:
        low_budget = int(
            getattr(args, "guidance_low_selectivity_bypass_ef_search", 0) or 0
        )
        if low_budget:
            return low_budget
    return int(
        getattr(args, "guidance_bypass_ef_search", 0) or configured_ef_search
    )


def d1_guidance_kind(args: argparse.Namespace, filter_name: str) -> str:
    requested = str(getattr(args, "d1_guidance_kind", "auto"))
    if requested in {"exact", "bloom"}:
        return requested
    if requested != "auto":
        raise ValueError(f"unsupported D1 guidance kind: {requested!r}")
    selectivity = float(args.filter_selectivity_by_name.get(filter_name, 100.0))
    threshold = float(getattr(args, "d1_exact_max_selectivity_pct", 2.5))
    return "exact" if selectivity <= threshold else "bloom"


def activation_atoms(
    args: argparse.Namespace, mode: str, filter_name: str
) -> list[str]:
    atoms = list(args.filter_atoms[filter_name])
    if (
        mode == "design1_bloom_bfs_layout_d3"
        and d1_guidance_kind(args, filter_name) == "exact"
    ):
        if (
            len(atoms) > 1
            and bool(getattr(args, "collapse_exact_and_guidance", True))
        ):
            predicate = str(args.filter_predicate_by_name[filter_name]).strip()
            if not predicate:
                raise ValueError(f"empty predicate for exact composite {filter_name!r}")
            return [f"exact:sql:{predicate}"]
        return [
            atom
            if atom == "|" or atom.upper() == "OR" or atom.startswith("exact:")
            else f"exact:{atom}"
            for atom in atoms
        ]
    return atoms


def read_guidance_profile(cur: psycopg.Cursor) -> dict[str, object]:
    cur.execute("SELECT vector_hnsw_guidance_profile()")
    raw = cur.fetchone()[0]
    profile = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(profile, dict):
        raise RuntimeError("guidance profile is not a JSON object")
    return profile


def read_scan_profile(cur: psycopg.Cursor) -> dict[str, object]:
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    raw = cur.fetchone()[0]
    profile = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(profile, dict):
        raise RuntimeError("scan profile is not a JSON object")
    return profile


def read_cache_profile(cur: psycopg.Cursor) -> dict[str, object]:
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    raw = cur.fetchone()[0]
    profile = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(profile, dict):
        raise RuntimeError("metadata cache profile is not a JSON object")
    return profile


def activate(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    mode: str,
    filter_name: str,
    *,
    read_profile: bool = True,
    reset_bypass_guidance: bool = True,
    configure_search_strategy: bool = True,
) -> dict[str, object]:
    table, index = mode_table_index(args, mode, filter_name)
    if mode == "original":
        return {"table": table, "index": index, "guidance_enabled": False, "guidance_route": "stock"}
    enabled, route = should_enable_guidance(args, filter_name)
    if not enabled:
        if reset_bypass_guidance:
            cur.execute("SELECT vector_hnsw_guidance_reset()")
        return {"table": table, "index": index, "guidance_enabled": False, "guidance_route": route}
    atoms = activation_atoms(args, mode, filter_name)
    validate_guidance_atoms(
        atoms, getattr(args, "candidate_validity_predicate", "")
    )
    if configure_search_strategy:
        cur.execute(f"SET hnsw.filter_strategy = {args.guidance_filter_strategy}")
    if args.reset_cache_per_query and mode in {"design1_bloom", "design1_bloom_bfs_layout"}:
        cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    kind = (
        "adaptive"
        if mode == "design1_bloom_bfs_layout_d3"
        else d1_guidance_kind(args, filter_name)
    )
    # These atoms are only the workload predicate from filters CSV. A broad
    # partial-index predicate (for example embedding_valid) is enforced by the
    # planner/index and SQL candidate qual, and must never become D1 guidance.
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
        (index, atoms, kind),
    )
    activation_row = cur.fetchone()
    activated_atoms = int(activation_row[0]) if activation_row and activation_row[0] is not None else 0
    profile = read_guidance_profile(cur) if read_profile else {}
    profile["table"] = table
    profile["index"] = index
    profile["activation_atom_count"] = activated_atoms
    profile["guidance_kind"] = kind
    active = bool(profile.get("active", activated_atoms > 0))
    if (
        mode == "design1_bloom_bfs_layout_d3"
        and str(profile.get("adaptive_state", "")) == "rejected"
    ):
        profile["guidance_enabled"] = False
        profile["guidance_route"] = "d3_admission_bypass"
    elif mode == "design1_bloom_bfs_layout_d3" and (activated_atoms <= 0 or not active):
        profile["guidance_enabled"] = False
        profile["guidance_route"] = "d3_stock_probe"
    else:
        profile["guidance_enabled"] = True
        profile["guidance_route"] = route
    return profile


def activation_binding(
    args: argparse.Namespace,
    mode: str,
    filter_name: str,
    activation_profile: dict[str, object],
) -> tuple[str, list[str], str] | None:
    if not activation_profile.get("guidance_enabled"):
        return None
    kind = str(
        activation_profile.get(
            "guidance_kind",
            "adaptive"
            if mode == "design1_bloom_bfs_layout_d3"
            else d1_guidance_kind(args, filter_name),
        )
    )
    return str(activation_profile["index"]), activation_atoms(args, mode, filter_name), kind


def run_query(
    cur: psycopg.Cursor,
    table: str,
    predicate: str,
    query_id: int,
    k: int,
    binding: tuple[str, list[str], str] | None = None,
    client_self_exclusion: bool = False,
    *,
    candidate_validity_predicate: str = "",
    query_table: str | None = None,
    query_id_column: str = "id",
    query_vector_column: str = "embedding",
    self_exclusion: bool = True,
    reset_profile: bool = True,
    read_profile: bool = True,
) -> tuple[list[int], list[float], dict[str, object]]:
    if reset_profile:
        cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    params: tuple[object, ...] = (int(query_id),)
    if binding is not None:
        params += binding
    if self_exclusion and not client_self_exclusion:
        params += (int(query_id),)
    cur.execute(
        search_query_sql(
            table,
            predicate,
            k,
            binding is not None,
            client_self_exclusion,
            candidate_validity_predicate=candidate_validity_predicate,
            query_table=query_table,
            query_id_column=query_id_column,
            query_vector_column=query_vector_column,
            self_exclusion=self_exclusion,
        ),
        params,
    )
    result_rows = cur.fetchall()
    raw_returned = len(result_rows)
    if client_self_exclusion:
        result_rows = [row for row in result_rows if int(row[0]) != int(query_id)][:k]
    ids = [int(row[0]) for row in result_rows[:k]]
    distances = [float(row[1]) for row in result_rows[:k]]
    profile = read_scan_profile(cur) if read_profile else {}
    profile["sqlens_raw_returned_before_self_exclusion"] = raw_returned
    return ids, distances, profile


@dataclass
class ModeRuntime:
    mode: str
    config: dict[str, object]
    cache_mb: int
    conn: psycopg.Connection
    cur: psycopg.Cursor
    planner_proof_verified: bool = False
    preferred_index_current_setting: str | None = None
    backend_cpu_provenance: dict[str, object] | None = None
    sqlens_runtime_identity: dict[str, object] | None = None
    d3_last_filter_name: str | None = None
    filter_strategy_current_setting: str | None = None
    iterative_scan_current_setting: str | None = None
    ef_search_current_setting: int | None = None
    traversal_guided_target_current_setting: int | None = None
    max_scan_tuples_current_setting: int | None = None
    scan_mem_multiplier_current_setting: float | None = None
    guided_collect_target_current_setting: int | None = None
    traversal_guided_burst_current_setting: int | None = None
    traversal_guided_early_stop_current_setting: bool | None = None
    traversal_guided_early_stop_distance_ratio_current_setting: float | None = None
    guidance_policy_enabled: bool = False


def route_runtime_index(
    args: argparse.Namespace,
    runtime: ModeRuntime,
    filter_name: str,
) -> tuple[str, str]:
    table, expected_index = mode_table_index(args, runtime.mode, filter_name)
    if not bool(getattr(args, "d2_source_on_guidance_bypass", False)):
        return table, expected_index
    if runtime.preferred_index_current_setting != expected_index:
        observed = set_preferred_index_if_supported(
            runtime.cur,
            args,
            expected_index,
        )
        if observed is None and bool(
            getattr(args, "require_preferred_index_guc", True)
        ):
            raise RuntimeError(
                "formal index-selection gate requires "
                f"{getattr(args, 'preferred_index_guc', 'hnsw.preferred_index')}"
            )
        runtime.preferred_index_current_setting = observed
    return table, expected_index


def route_runtime_search_settings(
    args: argparse.Namespace,
    runtime: ModeRuntime,
    filter_name: str,
) -> bool:
    """Apply adaptive search GUCs only when the route changes."""
    initialize_routed_search_state(runtime)
    previous_enabled = runtime.guidance_policy_enabled
    enabled = mode_uses_guidance(runtime.mode) and should_enable_guidance(
        args, filter_name
    )[0]
    strategy = (
        str(getattr(args, "guidance_filter_strategy", "traversal_guided"))
        if enabled
        else "off"
    )
    if not mode_uses_guidance(runtime.mode) or enabled:
        iterative_scan = str(
            configured_mode_value_for_filter(
                args, runtime, filter_name, "iterative_scan"
            )
        )
    else:
        iterative_scan = str(
            getattr(args, "guidance_bypass_iterative_scan", "strict_order")
        )
    configured_ef_search = configured_ef_search_for_filter(
        args, runtime, filter_name
    )
    bypass_ef_search = (
        bypass_ef_search_for_filter(args, filter_name, configured_ef_search)
        if mode_uses_guidance(runtime.mode)
        else configured_ef_search
    )
    ef_search = (
        bypass_ef_search
        if mode_uses_guidance(runtime.mode) and not enabled
        else configured_ef_search
    )
    traversal_target = configured_traversal_target_for_filter(
        args, runtime, filter_name
    )
    max_scan_tuples = int(configured_mode_value_for_filter(
        args, runtime, filter_name, "max_scan_tuples"
    ))
    scan_mem_multiplier = float(configured_mode_value_for_filter(
        args, runtime, filter_name, "scan_mem_multiplier"
    ))
    guided_collect_target = int(configured_mode_value_for_filter(
        args, runtime, filter_name, "guided_collect_target"
    ))
    traversal_guided_burst = int(configured_mode_value_for_filter(
        args, runtime, filter_name, "traversal_guided_burst"
    ))
    traversal_guided_early_stop = bool(configured_mode_value_for_filter(
        args, runtime, filter_name, "traversal_guided_early_stop"
    ))
    traversal_guided_early_stop_distance_ratio = float(
        configured_mode_value_for_filter(
            args,
            runtime,
            filter_name,
            "traversal_guided_early_stop_distance_ratio",
        )
    )
    k = int(getattr(args, "k", 10))
    if enabled and not k <= traversal_target <= ef_search:
        raise RuntimeError(
            f"per-filter traversal target for {runtime.mode}/{filter_name} must "
            f"satisfy k <= target <= ef_search; got k={k}, "
            f"target={traversal_target}, ef_search={ef_search}"
        )
    if runtime.filter_strategy_current_setting != strategy:
        runtime.cur.execute(f"SET hnsw.filter_strategy = {strategy}")
        runtime.filter_strategy_current_setting = strategy
    if runtime.iterative_scan_current_setting != iterative_scan:
        runtime.cur.execute(f"SET hnsw.iterative_scan = {iterative_scan}")
        runtime.iterative_scan_current_setting = iterative_scan
    if runtime.ef_search_current_setting != ef_search:
        runtime.cur.execute(f"SET hnsw.ef_search = {ef_search}")
        runtime.ef_search_current_setting = ef_search
    if runtime.traversal_guided_target_current_setting != traversal_target:
        runtime.cur.execute(
            f"SET hnsw.traversal_guided_target = {traversal_target}"
        )
        runtime.traversal_guided_target_current_setting = traversal_target
    per_filter_gucs = (
        (
            "max_scan_tuples_current_setting", "hnsw.max_scan_tuples",
            max_scan_tuples,
        ),
        (
            "scan_mem_multiplier_current_setting", "hnsw.scan_mem_multiplier",
            scan_mem_multiplier,
        ),
        (
            "guided_collect_target_current_setting", "hnsw.guided_collect_target",
            guided_collect_target,
        ),
        (
            "traversal_guided_burst_current_setting", "hnsw.traversal_guided_burst",
            traversal_guided_burst,
        ),
        (
            "traversal_guided_early_stop_current_setting",
            "hnsw.traversal_guided_early_stop",
            traversal_guided_early_stop,
        ),
        (
            "traversal_guided_early_stop_distance_ratio_current_setting",
            "hnsw.traversal_guided_early_stop_distance_ratio",
            traversal_guided_early_stop_distance_ratio,
        ),
    )
    for attribute, guc, value in per_filter_gucs:
        if getattr(runtime, attribute) != value:
            runtime.cur.execute("SELECT set_config(%s, %s, false)", (guc, str(value).lower()))
            setattr(runtime, attribute, value)
    runtime.guidance_policy_enabled = enabled
    return previous_enabled


def route_runtime_request(
    args: argparse.Namespace,
    runtime: ModeRuntime,
    filter_name: str,
) -> tuple[str, str, bool, bool]:
    """Route one request with a single PostgreSQL protocol round trip."""
    initialize_routed_search_state(runtime)
    table, expected_index = mode_table_index(args, runtime.mode, filter_name)
    previous_enabled = runtime.guidance_policy_enabled
    enabled = mode_uses_guidance(runtime.mode) and should_enable_guidance(
        args, filter_name
    )[0]
    strategy = (
        str(getattr(args, "guidance_filter_strategy", "traversal_guided"))
        if enabled
        else "off"
    )
    if not mode_uses_guidance(runtime.mode) or enabled:
        iterative_scan = str(
            configured_mode_value_for_filter(
                args, runtime, filter_name, "iterative_scan"
            )
        )
    else:
        iterative_scan = str(
            getattr(args, "guidance_bypass_iterative_scan", "strict_order")
        )
    configured_ef_search = configured_ef_search_for_filter(
        args, runtime, filter_name
    )
    bypass_ef_search = (
        bypass_ef_search_for_filter(args, filter_name, configured_ef_search)
        if mode_uses_guidance(runtime.mode)
        else configured_ef_search
    )
    ef_search = (
        bypass_ef_search
        if mode_uses_guidance(runtime.mode) and not enabled
        else configured_ef_search
    )
    traversal_target = configured_traversal_target_for_filter(
        args, runtime, filter_name
    )
    max_scan_tuples = int(configured_mode_value_for_filter(
        args, runtime, filter_name, "max_scan_tuples"
    ))
    scan_mem_multiplier = float(configured_mode_value_for_filter(
        args, runtime, filter_name, "scan_mem_multiplier"
    ))
    guided_collect_target = int(configured_mode_value_for_filter(
        args, runtime, filter_name, "guided_collect_target"
    ))
    traversal_guided_burst = int(configured_mode_value_for_filter(
        args, runtime, filter_name, "traversal_guided_burst"
    ))
    traversal_guided_early_stop = bool(configured_mode_value_for_filter(
        args, runtime, filter_name, "traversal_guided_early_stop"
    ))
    traversal_guided_early_stop_distance_ratio = float(
        configured_mode_value_for_filter(
            args,
            runtime,
            filter_name,
            "traversal_guided_early_stop_distance_ratio",
        )
    )
    k = int(getattr(args, "k", 10))
    if enabled and not k <= traversal_target <= ef_search:
        raise RuntimeError(
            f"per-filter traversal target for {runtime.mode}/{filter_name} must "
            f"satisfy k <= target <= ef_search; got k={k}, "
            f"target={traversal_target}, ef_search={ef_search}"
        )

    expressions: list[str] = []
    params: list[object] = []
    manages_preferred_index = (
        runtime.preferred_index_current_setting is not None
        or bool(getattr(args, "d2_source_on_guidance_bypass", False))
    )
    if manages_preferred_index and runtime.preferred_index_current_setting is None:
        observed = set_preferred_index_if_supported(
            runtime.cur, args, expected_index
        )
        if observed is None and bool(
            getattr(args, "require_preferred_index_guc", False)
        ):
            raise RuntimeError(
                "formal index-selection gate requires "
                f"{getattr(args, 'preferred_index_guc', 'hnsw.preferred_index')}"
            )
        runtime.preferred_index_current_setting = observed
    if (
        manages_preferred_index
        and runtime.preferred_index_current_setting != expected_index
    ):
        if runtime.preferred_index_current_setting is not None:
            expressions.append("set_config(%s, %s, false)")
            params.extend(
                [
                    str(
                        getattr(
                            args, "preferred_index_guc", "hnsw.preferred_index"
                        )
                    ),
                    expected_index,
                ]
            )
            runtime.preferred_index_current_setting = expected_index
    if runtime.filter_strategy_current_setting != strategy:
        expressions.append("set_config('hnsw.filter_strategy', %s, false)")
        params.append(strategy)
        runtime.filter_strategy_current_setting = strategy
    if runtime.iterative_scan_current_setting != iterative_scan:
        expressions.append("set_config('hnsw.iterative_scan', %s, false)")
        params.append(iterative_scan)
        runtime.iterative_scan_current_setting = iterative_scan
    if runtime.ef_search_current_setting != ef_search:
        expressions.append("set_config('hnsw.ef_search', %s, false)")
        params.append(str(ef_search))
        runtime.ef_search_current_setting = ef_search
    if runtime.traversal_guided_target_current_setting != traversal_target:
        expressions.append(
            "set_config('hnsw.traversal_guided_target', %s, false)"
        )
        params.append(str(traversal_target))
        runtime.traversal_guided_target_current_setting = traversal_target

    routed = (
        (
            "max_scan_tuples_current_setting", "hnsw.max_scan_tuples",
            max_scan_tuples,
        ),
        (
            "scan_mem_multiplier_current_setting", "hnsw.scan_mem_multiplier",
            scan_mem_multiplier,
        ),
        (
            "guided_collect_target_current_setting", "hnsw.guided_collect_target",
            guided_collect_target,
        ),
        (
            "traversal_guided_burst_current_setting", "hnsw.traversal_guided_burst",
            traversal_guided_burst,
        ),
        (
            "traversal_guided_early_stop_current_setting",
            "hnsw.traversal_guided_early_stop",
            traversal_guided_early_stop,
        ),
        (
            "traversal_guided_early_stop_distance_ratio_current_setting",
            "hnsw.traversal_guided_early_stop_distance_ratio",
            traversal_guided_early_stop_distance_ratio,
        ),
    )
    for attribute, guc, value in routed:
        if getattr(runtime, attribute) != value:
            expressions.append("set_config(%s, %s, false)")
            params.extend((guc, str(value).lower()))
            setattr(runtime, attribute, value)

    reset_performed = previous_enabled and not enabled
    if reset_performed:
        expressions.append("vector_hnsw_guidance_reset()")
    if expressions:
        runtime.cur.execute("SELECT " + ", ".join(expressions), tuple(params))
    runtime.guidance_policy_enabled = enabled
    return table, expected_index, previous_enabled, reset_performed


def gate_runtime_plans(
    args: argparse.Namespace,
    runtime: ModeRuntime,
    filters: list[tuple[str, float, str]],
    query_id: int,
) -> None:
    try:
        for filter_name, _, predicate in filters:
            table, expected_index, _, _ = route_runtime_request(
                args, runtime, filter_name
            )
            query_table = query_table_for_candidate(args, table)
            self_exclusion = candidate_self_exclusion(args, table)
            client_self_exclusion = (
                uses_exact_predicate_scan_contract(args.guidance_filter_strategy)
                and self_exclusion
            )
            try:
                activation = activate(runtime.cur, args, runtime.mode, filter_name)
                binding = activation_binding(args, runtime.mode, filter_name, activation)
                evidence = explain_hnsw_plan(
                    runtime.cur,
                    table,
                    expected_index,
                    predicate,
                    query_id,
                    args.k,
                    binding,
                    client_self_exclusion,
                    candidate_validity_predicate=getattr(
                        args, "candidate_validity_predicate", ""
                    ),
                    query_table=query_table,
                    query_id_column=getattr(args, "query_id_column", "id"),
                    query_vector_column=getattr(args, "query_vector_column", "embedding"),
                    self_exclusion=self_exclusion,
                )
            except Exception as exc:
                evidence = {
                    "passed": False,
                    "mode": runtime.mode,
                    "filter_name": filter_name,
                    "expected_index": expected_index,
                    "expected_table": table,
                    "query_id": query_id,
                    "query_table": query_table,
                    "self_excluded": self_exclusion,
                    "candidate_validity_predicate": effective_candidate_validity_predicate(
                        getattr(args, "candidate_validity_predicate", "")
                    ),
                    "candidate_validity_predicate_sha256": candidate_validity_sha256(
                        getattr(args, "candidate_validity_predicate", "")
                    ),
                    "failure": f"{exc.__class__.__name__}: {exc}",
                }
                args.plan_evidence.append(evidence)
                raise RuntimeError(
                    f"HNSW plan gate failed for mode={runtime.mode} filter={filter_name}: {exc}"
                ) from exc
            evidence.update(
                {
                    "mode": runtime.mode,
                    "filter_name": filter_name,
                    "config": runtime.config,
                    "planner_proof_verified": bool(evidence["passed"]),
                    "d3_initialization": d3_initialization_label(args),
                    "prebuilt_fragments": 0,
                    "preferred_index_guc": getattr(
                        args, "preferred_index_guc", "hnsw.preferred_index"
                    ),
                    "preferred_index_guc_available": (
                        runtime.preferred_index_current_setting is not None
                    ),
                    "preferred_index_current_setting": runtime.preferred_index_current_setting,
                    "backend_cpu_provenance": runtime.backend_cpu_provenance,
                    "sqlens_runtime_identity": runtime.sqlens_runtime_identity,
                    "candidate_validity_predicate": effective_candidate_validity_predicate(
                        getattr(args, "candidate_validity_predicate", "")
                    ),
                    "candidate_validity_predicate_sha256": candidate_validity_sha256(
                        getattr(args, "candidate_validity_predicate", "")
                    ),
                }
            )
            args.plan_evidence.append(evidence)
            if not evidence["passed"]:
                raise RuntimeError(
                    f"HNSW plan gate failed for mode={runtime.mode} filter={filter_name}: "
                    f"{evidence['failure']}"
                )
    finally:
        runtime.cur.execute("SELECT vector_hnsw_guidance_reset()")


def open_mode_runtime(
    args: argparse.Namespace,
    mode: str,
    filters: list[tuple[str, float, str]],
) -> ModeRuntime:
    cache_mb = args.d3_cache_mb if mode == "design1_bloom_bfs_layout_d3" else args.d1_cache_mb
    config = effective_mode_config(args, mode)
    conn = psycopg.connect(pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = conn.cursor()
        cpu_provenance = backend_cpu_provenance(
            cur,
            getattr(args, "backend_cpu_list", None),
        )
        if not hasattr(args, "backend_cpu_evidence"):
            args.backend_cpu_evidence = []
        args.backend_cpu_evidence.append({"mode": mode, **cpu_provenance})
        enforce_backend_cpu_provenance(cpu_provenance)
        runtime_identity = require_exact_sqlens_identity(
            cur,
            args.expected_sqlens_build_id,
            args.expected_vector_so_sha256,
        )
        if not hasattr(args, "runtime_sqlens_identity_evidence"):
            args.runtime_sqlens_identity_evidence = []
        args.runtime_sqlens_identity_evidence.append(
            {"mode": mode, "backend_pid": cpu_provenance["backend_pid"], **runtime_identity}
        )
        ensure_functions(cur)
        if mode_uses_guidance(mode) and not bool(
            getattr(args, "fragment_tracking_prepared", False)
        ):
            ensure_tracking(cur, args.insertion_table, args.bfs_table)
        configure(cur, args, cache_mb, mode, config)
        _, expected_index = mode_table_index(args, mode)
        preferred_index_current_setting = set_preferred_index_if_supported(
            cur,
            args,
            expected_index,
        )
        if (
            preferred_index_current_setting is None
            and bool(getattr(args, "require_preferred_index_guc", True))
        ):
            raise RuntimeError(
                f"formal index-selection gate requires {args.preferred_index_guc}; "
                "load the SQLens build that exposes the session preferred-index GUC"
            )
        cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
        runtime = ModeRuntime(
            mode=mode,
            config=config,
            cache_mb=cache_mb,
            conn=conn,
            cur=cur,
            preferred_index_current_setting=preferred_index_current_setting,
            backend_cpu_provenance=cpu_provenance,
            sqlens_runtime_identity=runtime_identity,
            filter_strategy_current_setting=(
                str(getattr(args, "guidance_filter_strategy", "traversal_guided"))
                if mode_uses_guidance(mode)
                else "off"
            ),
            iterative_scan_current_setting=str(config["iterative_scan"]),
            ef_search_current_setting=int(config["ef_search"]),
            traversal_guided_target_current_setting=int(
                config.get("traversal_guided_target", 40)
            ),
            max_scan_tuples_current_setting=int(
                config.get("max_scan_tuples", MODE_CONFIG_DEFAULTS["max_scan_tuples"])
            ),
            scan_mem_multiplier_current_setting=float(
                config.get("scan_mem_multiplier", MODE_CONFIG_DEFAULTS["scan_mem_multiplier"])
            ),
            guided_collect_target_current_setting=int(
                config.get("guided_collect_target", MODE_CONFIG_DEFAULTS["guided_collect_target"])
            ),
            traversal_guided_burst_current_setting=int(
                config.get("traversal_guided_burst", MODE_CONFIG_DEFAULTS["traversal_guided_burst"])
            ),
            traversal_guided_early_stop_current_setting=bool(
                config.get("traversal_guided_early_stop", MODE_CONFIG_DEFAULTS["traversal_guided_early_stop"])
            ),
            traversal_guided_early_stop_distance_ratio_current_setting=float(
                config.get("traversal_guided_early_stop_distance_ratio", MODE_CONFIG_DEFAULTS["traversal_guided_early_stop_distance_ratio"])
            ),
            guidance_policy_enabled=mode_uses_guidance(mode),
        )
        plan_query_id = getattr(args, "plan_query_id", None)
        if plan_query_id is not None:
            gate_runtime_plans(args, runtime, filters, int(plan_query_id))
            runtime.planner_proof_verified = True
            # Gate activations must not seed the workload-driven adaptive state.
            cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
            configure(cur, args, cache_mb, mode, config)
            runtime.preferred_index_current_setting = set_preferred_index_if_supported(
                cur,
                args,
                expected_index,
            )
            runtime.filter_strategy_current_setting = (
                str(getattr(args, "guidance_filter_strategy", "traversal_guided"))
                if mode_uses_guidance(mode)
                else "off"
            )
            runtime.iterative_scan_current_setting = str(config["iterative_scan"])
            runtime.ef_search_current_setting = int(config["ef_search"])
            runtime.traversal_guided_target_current_setting = int(
                config.get("traversal_guided_target", 40)
            )
            runtime.max_scan_tuples_current_setting = int(
                config.get("max_scan_tuples", MODE_CONFIG_DEFAULTS["max_scan_tuples"])
            )
            runtime.scan_mem_multiplier_current_setting = float(
                config.get("scan_mem_multiplier", MODE_CONFIG_DEFAULTS["scan_mem_multiplier"])
            )
            runtime.guided_collect_target_current_setting = int(
                config.get("guided_collect_target", MODE_CONFIG_DEFAULTS["guided_collect_target"])
            )
            runtime.traversal_guided_burst_current_setting = int(
                config.get("traversal_guided_burst", MODE_CONFIG_DEFAULTS["traversal_guided_burst"])
            )
            runtime.traversal_guided_early_stop_current_setting = bool(
                config.get("traversal_guided_early_stop", MODE_CONFIG_DEFAULTS["traversal_guided_early_stop"])
            )
            runtime.traversal_guided_early_stop_distance_ratio_current_setting = float(
                config.get("traversal_guided_early_stop_distance_ratio", MODE_CONFIG_DEFAULTS["traversal_guided_early_stop_distance_ratio"])
            )
            runtime.guidance_policy_enabled = mode_uses_guidance(mode)
        return runtime
    except Exception:
        conn.close()
        raise


def close_mode_runtime(runtime: ModeRuntime) -> None:
    try:
        runtime.cur.execute("SELECT vector_hnsw_guidance_reset()")
    finally:
        runtime.cur.close()
        runtime.conn.close()


def recover_runtime(args: argparse.Namespace, runtime: ModeRuntime) -> None:
    try:
        runtime.cur.execute("ROLLBACK")
    except Exception:
        pass
    configure(runtime.cur, args, runtime.cache_mb, runtime.mode, runtime.config)
    _, expected_index = mode_table_index(args, runtime.mode)
    runtime.preferred_index_current_setting = set_preferred_index_if_supported(
        runtime.cur,
        args,
        expected_index,
    )
    runtime.filter_strategy_current_setting = (
        str(getattr(args, "guidance_filter_strategy", "traversal_guided"))
        if mode_uses_guidance(runtime.mode)
        else "off"
    )
    runtime.iterative_scan_current_setting = str(runtime.config["iterative_scan"])
    runtime.ef_search_current_setting = int(runtime.config["ef_search"])
    runtime.traversal_guided_target_current_setting = int(
        runtime.config.get("traversal_guided_target", 40)
    )
    runtime.max_scan_tuples_current_setting = int(
        runtime.config.get("max_scan_tuples", MODE_CONFIG_DEFAULTS["max_scan_tuples"])
    )
    runtime.scan_mem_multiplier_current_setting = float(
        runtime.config.get("scan_mem_multiplier", MODE_CONFIG_DEFAULTS["scan_mem_multiplier"])
    )
    runtime.guided_collect_target_current_setting = int(
        runtime.config.get("guided_collect_target", MODE_CONFIG_DEFAULTS["guided_collect_target"])
    )
    runtime.traversal_guided_burst_current_setting = int(
        runtime.config.get("traversal_guided_burst", MODE_CONFIG_DEFAULTS["traversal_guided_burst"])
    )
    runtime.traversal_guided_early_stop_current_setting = bool(
        runtime.config.get("traversal_guided_early_stop", MODE_CONFIG_DEFAULTS["traversal_guided_early_stop"])
    )
    runtime.traversal_guided_early_stop_distance_ratio_current_setting = float(
        runtime.config.get("traversal_guided_early_stop_distance_ratio", MODE_CONFIG_DEFAULTS["traversal_guided_early_stop_distance_ratio"])
    )
    runtime.guidance_policy_enabled = mode_uses_guidance(runtime.mode)


def run_warmup(
    args: argparse.Namespace,
    runtime: ModeRuntime,
    filter_name: str,
    predicate: str,
    query_id: int,
) -> None:
    evidence: dict[str, object] = {
        "mode": runtime.mode,
        "filter_name": filter_name,
        "query_id": query_id,
        "status": "running",
        "error": "",
    }
    try:
        _, _, previous_guidance_policy, reset_performed = route_runtime_request(
            args, runtime, filter_name
        )
        evidence["guidance_before"] = read_guidance_profile(runtime.cur)
        evidence["cache_before"] = read_cache_profile(runtime.cur)
        same_predicate_before = (
            runtime.mode == "design1_bloom_bfs_layout_d3"
            and runtime.d3_last_filter_name == filter_name
        )
        activation_profile = activate(
            runtime.cur,
            args,
            runtime.mode,
            filter_name,
            reset_bypass_guidance=(
                previous_guidance_policy and not reset_performed
            ),
            configure_search_strategy=False,
        )
        if runtime.mode == "design1_bloom_bfs_layout_d3":
            runtime.d3_last_filter_name = filter_name
        binding = activation_binding(args, runtime.mode, filter_name, activation_profile)
        candidate_table = str(activation_profile["table"])
        self_exclusion = candidate_self_exclusion(args, candidate_table)
        run_query(
            runtime.cur,
            candidate_table,
            predicate,
            query_id,
            args.k,
            binding,
            uses_exact_predicate_scan_contract(args.guidance_filter_strategy) and self_exclusion,
            candidate_validity_predicate=getattr(
                args, "candidate_validity_predicate", ""
            ),
            query_table=query_table_for_candidate(args, candidate_table),
            query_id_column=getattr(args, "query_id_column", "id"),
            query_vector_column=getattr(args, "query_vector_column", "embedding"),
            self_exclusion=self_exclusion,
        )
        evidence["guidance_after"] = read_guidance_profile(runtime.cur)
        evidence["cache_after"] = read_cache_profile(runtime.cur)
        if runtime.mode == "design1_bloom_bfs_layout_d3":
            merged_activation = {
                **evidence["guidance_after"],
                **activation_profile,
            }
            phase = d3_phase_evidence(
                evidence["guidance_before"],
                evidence["guidance_after"],
                evidence["cache_before"],
                evidence["cache_after"],
                merged_activation,
                same_predicate_before=same_predicate_before,
            )
            evidence.update(phase)
            if not hasattr(args, "d3_warmup_phase_evidence"):
                args.d3_warmup_phase_evidence = []
            args.d3_warmup_phase_evidence.append(
                {
                    "filter_name": filter_name,
                    "query_id": query_id,
                    **phase,
                }
            )
        evidence["status"] = "complete"
        getattr(args, "warmup_evidence").append(evidence)
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = f"{exc.__class__.__name__}: {exc}"
        getattr(args, "warmup_evidence").append(evidence)
        recover_runtime(args, runtime)
        raise RuntimeError(
            f"warmup failed for mode={runtime.mode} filter={filter_name} query_id={query_id}: {exc}"
        ) from exc


def pair_key(filter_name: str, query_no: int, repeat: int) -> str:
    return f"{filter_name}|q{query_no}|r{repeat}"


def guidance_scan_contract_satisfied(
    scan_profile: dict[str, object],
    filter_strategy: str,
    expected_prioritization: bool | None = None,
    expected_burst: int | None = None,
) -> bool:
    return guidance_scan_contract_failure(
        scan_profile,
        filter_strategy,
        expected_prioritization,
        expected_burst,
    ) == ""


def guidance_scan_contract_failure(
    scan_profile: dict[str, object],
    filter_strategy: str,
    expected_prioritization: bool | None = None,
    expected_burst: int | None = None,
) -> str:
    if filter_strategy == "traversal_guided":
        missing = [field for field in SQLENS_TRAVERSAL_PROFILE_FIELDS if field not in scan_profile]
        if missing:
            return f"traversal profile is missing fields: {missing}"
        final_path = scan_profile.get("final_path")
        if final_path == "stock_bypass":
            if scan_profile.get("planner_proof_attempted") is not True:
                return "stock bypass planner proof was not attempted"
            if scan_profile.get("planner_proof_succeeded") is not True:
                return (
                    "stock bypass planner proof failed: "
                    f"{scan_profile.get('planner_proof_bypass_reason', 'unknown')}"
                )
            if int(scan_profile.get("stock_bypass_requests", 0) or 0) != 1:
                return "stock bypass request count is not exactly one"
            if str(scan_profile.get("stock_bypass_reason", "")) != (
                "low_estimated_skip_rate"
            ):
                return (
                    "formal active-guidance stock bypass has an unsupported reason: "
                    f"{scan_profile.get('stock_bypass_reason')!r}"
                )
            if int(scan_profile.get("fallback_requests", 0) or 0) != 0:
                return "admission stock bypass unexpectedly used fallback"
            if scan_profile.get("traversal_estimated_skip_rate_valid") is not True:
                return "admission stock bypass lacks a valid skip-rate estimate"
            try:
                estimated_skip_rate = float(
                    scan_profile["traversal_estimated_skip_rate"]
                )
            except (KeyError, TypeError, ValueError):
                return "admission stock bypass skip-rate estimate is invalid"
            if not math.isfinite(estimated_skip_rate) or not (
                0.0 <= estimated_skip_rate <= 1.0
            ):
                return "admission stock bypass skip-rate is outside [0, 1]"
            for field in (
                "guidance_checks",
                "traversal_guidance_checks",
                "neighbor_expansion_guidance_checks",
                "guided_expanded_nodes",
                "guided_phase_distance_computations",
            ):
                if int(scan_profile.get(field, 0) or 0) != 0:
                    return f"admission stock bypass recorded guided work ({field})"
            if min(
                int(scan_profile.get("stock_phase_expanded_nodes", 0) or 0),
                int(scan_profile.get("stock_phase_distance_computations", 0) or 0),
            ) <= 0:
                return "admission stock bypass recorded no stock traversal work"
            if scan_profile.get("traversal_guidance_scope") != "none":
                return "admission stock bypass reported a guided scope"
            if scan_profile.get("graph_expansion_pruned") is not False:
                return "admission stock bypass claimed graph-expansion pruning"
            if scan_profile.get("distance_computations_pruned") is not False:
                return "admission stock bypass claimed distance pruning"
            return ""
        if final_path == "fresh_stock_fallback":
            if expected_prioritization is not True:
                return "fresh stock fallback requires an admitted prioritization attempt"
            if scan_profile.get("planner_proof_attempted") is not True:
                return "fallback planner proof was not attempted"
            if scan_profile.get("planner_proof_succeeded") is not True:
                return (
                    "fallback planner proof failed: "
                    f"{scan_profile.get('planner_proof_bypass_reason', 'unknown')}"
                )
            if scan_profile.get("approximate_prioritization_attempted") is not True:
                return "fresh fallback recorded no guided prioritization attempt"
            if scan_profile.get("fallback_iterative_scan_enabled") is not True:
                return "fresh fallback did not enable internal strict iterative scan"
            if int(scan_profile.get("fallback_requests", 0) or 0) <= 0:
                return "fresh fallback recorded no fallback request"
            if str(scan_profile.get("fallback_reason", "none")) == "none":
                return "fresh fallback recorded no reason"
            guided_expanded = int(scan_profile.get("guided_expanded_nodes", 0) or 0)
            guided_distances = int(
                scan_profile.get("guided_phase_distance_computations", 0) or 0
            )
            stock_expanded = int(scan_profile.get("stock_phase_expanded_nodes", 0) or 0)
            stock_distances = int(
                scan_profile.get("stock_phase_distance_computations", 0) or 0
            )
            fallback_expanded = int(
                scan_profile.get("fallback_stock_expanded_nodes", 0) or 0
            )
            fallback_distances = int(
                scan_profile.get("fallback_stock_distance_computations", 0) or 0
            )
            if min(guided_expanded, guided_distances, stock_expanded, stock_distances) <= 0:
                return "fresh fallback omitted guided or stock work counters"
            if (stock_expanded, stock_distances) != (
                fallback_expanded,
                fallback_distances,
            ):
                return "fresh fallback stock counters are internally inconsistent"
            neighbor_checks = int(
                scan_profile.get("neighbor_expansion_guidance_checks", 0) or 0
            )
            neighbor_matches = int(
                scan_profile.get("neighbor_expansion_guidance_matches", 0) or 0
            )
            neighbor_misses = int(
                scan_profile.get("neighbor_expansion_guidance_misses", 0) or 0
            )
            if neighbor_checks <= 0 or neighbor_checks != neighbor_matches + neighbor_misses:
                return "fresh fallback has invalid guided neighbor accounting"
            if int(scan_profile.get("guidance_checks", 0) or 0) != 0:
                return "fresh stock fallback unexpectedly applied guide validation"
            if scan_profile.get("traversal_guidance_scope") != "none":
                return "fresh stock fallback reported a guided final scope"
            if scan_profile.get("graph_expansion_pruned") is not False:
                return "fresh fallback claimed graph-expansion pruning"
            if scan_profile.get("distance_computations_pruned") is not False:
                return "fresh fallback claimed distance pruning"
            if scan_profile.get("traversal_estimated_skip_rate_valid") is not True:
                return "fresh fallback lacks its admission skip-rate estimate"
            return ""

        if int(scan_profile.get("guidance_checks", 0) or 0) <= 0:
            return "heap validation did not consult the active guide"
        if expected_prioritization is True:
            if final_path != "approximate_traversal_prioritization":
                return (
                    f"traversal final_path={final_path!r}, expected "
                    "'approximate_traversal_prioritization'"
                )
            if scan_profile.get("approximate_ann_path") is not True:
                return "approximate traversal path was not reported"
            if scan_profile.get("approximate_prioritization_attempted") is not True:
                return "dual-frontier traversal did not report a prioritization attempt"
            priority_reorders = int(scan_profile.get("priority_reorders", 0) or 0)
            order_changed = scan_profile.get("traversal_order_changed") is True
            if order_changed != (priority_reorders > 0):
                return "traversal_order_changed disagrees with priority_reorders"
            frontier_pops = int(scan_profile.get("match_frontier_pops", 0) or 0) + int(
                scan_profile.get("no_bridge_frontier_pops", 0) or 0
            )
            if frontier_pops <= 0:
                return "dual-frontier traversal reported no frontier pops"
            if expected_burst is not None and int(
                scan_profile.get("traversal_prioritization_burst", 0) or 0
            ) != int(expected_burst):
                return (
                    "traversal burst mismatch: "
                    f"observed={scan_profile.get('traversal_prioritization_burst')!r}, "
                    f"expected={expected_burst}"
                )
        elif expected_prioritization is False:
            if final_path not in {"candidate_admission_validation_only", "guided"}:
                return (
                    f"traversal final_path={final_path!r}, expected candidate-admission "
                    "validation-only"
                )
            if scan_profile.get("approximate_ann_path") is True:
                return "validation-only traversal unexpectedly used an approximate ANN path"
        elif final_path not in {
            "guided",
            "candidate_admission_validation_only",
            "approximate_traversal_prioritization",
        }:
            return f"unrecognized traversal final_path={final_path!r}"
        if scan_profile.get("planner_proof_attempted") is not True:
            return "planner proof was not attempted"
        if scan_profile.get("planner_proof_succeeded") is not True:
            return (
                "planner proof failed: "
                f"{scan_profile.get('planner_proof_bypass_reason', 'unknown')}"
            )
        for field in ("stock_bypass_requests", "fallback_requests"):
            if int(scan_profile.get(field, 0) or 0) != 0:
                return f"traversal used a stock bypass/fallback ({field}={scan_profile.get(field)})"
        for field in (
            "stock_phase_expanded_nodes",
            "stock_phase_distance_computations",
            "fallback_stock_expanded_nodes",
            "fallback_stock_distance_computations",
        ):
            if int(scan_profile.get(field, 0) or 0) != 0:
                return f"guided final path contains stock work ({field}={scan_profile.get(field)})"
        pre_distance_checks = int(scan_profile.get("pre_distance_membership_checks", 0) or 0)
        attempted_avoided = int(scan_profile.get("distance_computations_avoided_attempted", 0) or 0)
        avoided = int(scan_profile.get("distance_computations_avoided", 0) or 0)
        neighbor_checks = int(scan_profile.get("neighbor_expansion_guidance_checks", 0) or 0)
        neighbor_matches = int(scan_profile.get("neighbor_expansion_guidance_matches", 0) or 0)
        neighbor_misses = int(scan_profile.get("neighbor_expansion_guidance_misses", 0) or 0)
        guided_admissions = int(scan_profile.get("traversal_guided_admissions", 0) or 0)
        guided_suppressions = int(scan_profile.get("traversal_guided_suppressions", 0) or 0)
        heap_tids_suppressed = int(scan_profile.get("traversal_heap_tids_suppressed", 0) or 0)
        expanded = int(scan_profile.get("guided_expanded_nodes", 0) or 0)
        distance_calls = int(scan_profile.get("guided_phase_distance_computations", 0) or 0)
        total_distance_calls = int(scan_profile.get("distance_compute_count", 0) or 0)
        total_expanded = int(scan_profile.get("traversal_expanded_nodes", 0) or 0)
        try:
            estimated_skip_rate = float(scan_profile["traversal_estimated_skip_rate"])
        except (KeyError, TypeError, ValueError):
            return "traversal skip-rate estimate is missing or invalid"
        if scan_profile.get("traversal_estimated_skip_rate_valid") is not True:
            return "traversal skip-rate estimate was not valid for formal admission"
        if not math.isfinite(estimated_skip_rate) or not 0.0 <= estimated_skip_rate <= 1.0:
            return "traversal skip-rate estimate is outside [0, 1]"
        scope = scan_profile.get("traversal_guidance_scope")
        expected_scopes = {
            "approximate_traversal_prioritization": {
                "approximate_traversal_prioritization_and_candidate_admission"
            },
            "candidate_admission_validation_only": {
                "candidate_admission_and_pre_heap_tid_validation"
            },
            # Kept for v11 artifacts while formal v12 runs use one of the two
            # explicit paths above.
            "guided": {
                "candidate_admission_and_validation",
                "legacy_experimental_guidance",
            },
        }
        if scope not in expected_scopes.get(str(final_path), set()):
            return (
                "traversal guidance scope does not match final_path: "
                f"final_path={final_path!r}, scope={scope!r}"
            )
        if scan_profile.get("graph_expansion_pruned") is not False:
            return "formal candidate admission must not claim graph-expansion pruning"
        if scan_profile.get("distance_computations_pruned") is not False:
            return "formal candidate admission must not claim distance pruning"
        if pre_distance_checks != 0 or attempted_avoided != 0 or avoided != 0:
            return "candidate admission unexpectedly recorded pre-distance pruning"
        if neighbor_checks <= 0 or neighbor_checks != neighbor_matches + neighbor_misses:
            return "invalid or empty neighbor-expansion membership accounting"
        if guided_admissions <= 0:
            return "candidate admission recorded no guided admissions"
        if heap_tids_suppressed < guided_suppressions:
            return "heap-TID suppression count is smaller than suppressed HNSW elements"
        if expanded <= 0 or distance_calls <= 0:
            return "guided path recorded no expansions or no distance calls"
        if total_expanded < expanded or total_distance_calls < distance_calls:
            return "guided expansion/distance counters exceed total scan counters"
        return ""
    if int(scan_profile.get("guidance_checks", 0) or 0) <= 0:
        return "heap validation did not consult the active guide"
    if filter_strategy in {"guided_collect", "acorn1"}:
        if int(scan_profile.get("traversal_guidance_checks", 0) or 0) <= 0:
            return "legacy diagnostic strategy recorded no traversal guidance checks"
        if scan_profile.get("final_path") not in {None, "legacy_guided"}:
            return f"unexpected legacy diagnostic final_path={scan_profile.get('final_path')!r}"
    elif filter_strategy == "safe_guided" and scan_profile.get("final_path") not in {
        None,
        "validation_only",
    }:
        return f"unexpected safe_guided final_path={scan_profile.get('final_path')!r}"
    return ""


def _counter(profile: dict[str, object], field: str) -> int:
    return int(profile.get(field, 0) or 0)


def _safe_counter_delta(
    before: dict[str, object], after: dict[str, object], field: str
) -> tuple[int, bool]:
    """Return a non-negative snapshot delta and flag a counter reset.

    D3 profiles are cumulative snapshots, but replacing an active guide can
    reset descriptor-local counters.  A negative delta is therefore not a
    cache hit or a build; it is unusable telemetry and must fail closed.
    """
    before_value = _counter(before, field)
    after_value = _counter(after, field)
    if after_value < before_value:
        return 0, True
    return after_value - before_value, False


def _safe_counter_delta_with_fallback(
    before: dict[str, object],
    after: dict[str, object],
    field: str,
    fallback_before: dict[str, object],
    fallback_after: dict[str, object],
    fallback_field: str | None = None,
) -> tuple[int, bool, str]:
    """Read a monotonic event counter, falling back to the other profile.

    The extension exposes adaptive counters in the metadata-cache profile and
    descriptor counters in the guidance profile.  Tests and older binaries may
    expose only one of those views, so the fallback is used only when the
    primary field is absent from both snapshots.  A present-but-decreasing
    primary counter is intentionally not hidden by the fallback.
    """
    primary_field = field
    if primary_field in before or primary_field in after:
        delta, invalid = _safe_counter_delta(before, after, primary_field)
        return delta, invalid, primary_field
    selected = fallback_field or field
    if selected in fallback_before or selected in fallback_after:
        delta, invalid = _safe_counter_delta(
            fallback_before, fallback_after, selected
        )
        return delta, invalid, selected
    return 0, False, ""


def _explicit_reuse_event(
    activation_profile: dict[str, object], keys: tuple[str, ...]
) -> bool:
    """Accept only explicit per-request event fields, never cumulative totals."""
    for key in keys:
        value = activation_profile.get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (int, float)) and value > 0:
            return True
        if isinstance(value, str) and value.strip().lower() in {
            "true",
            "hit",
            "reused",
            "reuse",
        }:
            return True
    return False


def d3_phase_evidence(
    guidance_before: dict[str, object],
    guidance_after: dict[str, object],
    cache_before: dict[str, object],
    cache_after: dict[str, object],
    activation_profile: dict[str, object],
    *,
    same_predicate_before: bool,
) -> dict[str, object]:
    state_before = str(guidance_before.get("adaptive_state", "stock"))
    state_after = str(guidance_after.get("adaptive_state", "stock"))
    global_active_before = bool(guidance_before.get("active", False))
    active_before = global_active_before and same_predicate_before
    active_after = bool(guidance_after.get("active", False))
    admissions_before = _counter(guidance_before, "adaptive_admissions")
    admissions_after = _counter(guidance_after, "adaptive_admissions")
    counter_reset_fields: list[str] = []

    def delta(field: str, *, fallback_field: str | None = None) -> int:
        value, invalid, source = _safe_counter_delta_with_fallback(
            guidance_before,
            guidance_after,
            field,
            cache_before,
            cache_after,
            fallback_field,
        )
        if invalid:
            counter_reset_fields.append(source or field)
        return value

    adaptive_admissions_delta = delta("adaptive_admissions")
    adaptive_refinements_delta = delta("adaptive_refinements")
    adaptive_page_builds_delta = delta("adaptive_page_builds")
    adaptive_bloom_builds_delta = delta("adaptive_bloom_builds")
    adaptive_exact_builds_delta = delta("adaptive_exact_builds")
    adaptive_fragment_builds_delta = delta(
        "adaptive_fragment_builds", fallback_field="fragment_builds"
    )
    fragment_cache_hits_delta = delta(
        "fragment_cache_hits", fallback_field="adaptive_fragment_cache_hits"
    )
    fragment_store_hits_delta = delta(
        "fragment_store_hits", fallback_field="adaptive_fragment_store_hits"
    )
    fast_reactivation_hits_delta = delta(
        "fast_reactivation_hits", fallback_field="adaptive_fast_reactivation_hits"
    )
    fragment_builds_delta = delta(
        "fragment_builds", fallback_field="adaptive_fragment_builds"
    )
    cache_composed_hits_delta, cache_composed_invalid = _safe_counter_delta(
        cache_before, cache_after, "composed_guide_hits"
    )
    if cache_composed_invalid:
        counter_reset_fields.append("composed_guide_hits")
    guidance_composed_hits_delta, guidance_composed_invalid = _safe_counter_delta(
        guidance_before, guidance_after, "composed_guide_hits"
    )
    if guidance_composed_invalid:
        counter_reset_fields.append("composed_guide_hits")
    composed_guide_hits_delta = max(
        cache_composed_hits_delta, guidance_composed_hits_delta
    )
    # A reset in any reuse/build counter invalidates the request-level reuse
    # claim.  Deduplicate only for readable artifact evidence.
    counter_reset_fields = sorted(set(counter_reset_fields))
    counters_reset = bool(counter_reset_fields)

    admitted_before = active_before and admissions_before > 0 and state_before not in {
        "stock",
        "probing",
        "rejected",
    }
    admitted_after = active_after and admissions_after > 0 and state_after not in {
        "stock",
        "probing",
        "rejected",
    }
    if active_before and not admitted_before and not counters_reset:
        raise RuntimeError(
            "D3 active state before request lacks a prior adaptive admission proof"
        )
    if active_after and not admitted_after and not counters_reset:
        raise RuntimeError(
            "D3 active state after request lacks an adaptive admission proof"
        )
    route = str(activation_profile.get("guidance_route", ""))
    guidance_enabled = activation_profile.get("guidance_enabled") is True
    build_observed = bool(
        fragment_builds_delta
        or adaptive_fragment_builds_delta
        or adaptive_page_builds_delta
        or adaptive_bloom_builds_delta
        or adaptive_exact_builds_delta
    )
    refinement_observed = bool(adaptive_refinements_delta)
    prior_admitted_state = state_before not in {"stock", "probing", "rejected"}
    refinement_observed = bool(
        refinement_observed
        or (
            build_observed
            and (
                prior_admitted_state
                or state_before == "page" and state_after in {"bloom", "exact"}
                or not adaptive_admissions_delta
            )
        )
    )
    cache_event = bool(
        fragment_cache_hits_delta
        or composed_guide_hits_delta
        or _explicit_reuse_event(
            activation_profile,
            (
                "fragment_cache_reactivation",
                "cache_reuse_event",
                "fast_cache_hit",
            ),
        )
    )
    store_event = bool(
        fragment_store_hits_delta
        or _explicit_reuse_event(
            activation_profile,
            ("fragment_store_reactivation", "store_reuse_event"),
        )
    )
    fast_event = bool(
        fast_reactivation_hits_delta
        or _explicit_reuse_event(
            activation_profile,
            ("fast_reactivation_event", "fast_reactivation"),
        )
    )
    reuse_event_trusted = not counters_reset and not build_observed and not refinement_observed
    if not reuse_event_trusted:
        cache_event = store_event = fast_event = False
    if route == "d3_stock_probe":
        phase_detail = "probe"
    elif not guidance_enabled:
        if active_after or admitted_after:
            raise RuntimeError("D3 bypass unexpectedly left guidance active")
        phase_detail = "bypass"
    elif build_observed or refinement_observed:
        phase_detail = "refinement" if prior_admitted_state or refinement_observed else "admission"
    elif adaptive_admissions_delta > 0:
        phase_detail = "admission"
    elif store_event:
        phase_detail = "fragment_store_reactivation"
    elif cache_event or fast_event:
        phase_detail = "fragment_cache_reactivation"
    elif active_before and admitted_before:
        # Active guidance is not proof of a reuse event.  Keep this distinct
        # from cache/store reactivation and report reused=False.
        phase_detail = "warm_active"
    else:
        phase_detail = "bypass"

    reused = bool(
        reuse_event_trusted
        and phase_detail
        in {"fragment_cache_reactivation", "fragment_store_reactivation"}
        and (cache_event or store_event or fast_event)
        and guidance_enabled
        and active_after
        and _counter(activation_profile, "activation_atom_count") > 0
    )
    # Preserve the legacy phase vocabulary consumed by existing lifecycle
    # gates, while exposing the precise mutually-exclusive phase separately.
    phase = (
        "probe"
        if phase_detail == "probe"
        else "bypass"
        if phase_detail == "bypass"
        else "admission"
        if phase_detail == "admission"
        else "refinement"
        if phase_detail == "refinement"
        else "warm"
    )

    fields: dict[str, object] = {
        "d3_phase": phase,
        "d3_guidance_route": route,
        "d3_state_before": state_before,
        "d3_state_after": state_after,
        "d3_global_active_before": global_active_before,
        "d3_same_predicate_before": same_predicate_before,
        "d3_active_before": active_before,
        "d3_active_after": active_after,
        "d3_admitted_before": admitted_before,
        "d3_admitted_after": admitted_after,
        "d3_active_guidance_reused": reused,
        "d3_phase_detail": phase_detail,
        "d3_reuse_event": (
            "fragment_store"
            if store_event and reuse_event_trusted
            else "fragment_cache"
            if cache_event and reuse_event_trusted
            else "fast_reactivation"
            if fast_event and reuse_event_trusted
            else ""
        ),
        "d3_reuse_event_trusted": reuse_event_trusted,
        "d3_counter_reset_observed": counters_reset,
        "d3_counter_reset_fields": json.dumps(counter_reset_fields),
        "d3_build_observed": build_observed,
        "d3_refinement_observed": refinement_observed,
    }
    for field in (
        "adaptive_requests",
        "adaptive_probes",
        "adaptive_admissions",
        "adaptive_page_builds",
        "adaptive_bloom_builds",
        "adaptive_exact_builds",
        "adaptive_refinements",
        "adaptive_rejections",
        "fragment_cache_hits",
        "fragment_store_hits",
        "fragment_builds",
        "fast_reactivation_hits",
    ):
        before = _counter(guidance_before, field)
        after = _counter(guidance_after, field)
        output = f"d3_{field}"
        fields[f"{output}_before"] = before
        fields[f"{output}_after"] = after
        safe_delta, invalid = _safe_counter_delta(guidance_before, guidance_after, field)
        fields[f"{output}_delta"] = safe_delta
        fields[f"{output}_delta_invalid"] = invalid
    for field in ("resident_entries", "resident_bytes", "composed_guide_hits", "evictions"):
        before = _counter(cache_before, field)
        after = _counter(cache_after, field)
        output = f"d3_cache_{field}" if not field.startswith("composed") else "d3_composed_guide_hits"
        fields[f"{output}_before"] = before
        fields[f"{output}_after"] = after
        safe_delta, invalid = _safe_counter_delta(cache_before, cache_after, field)
        fields[f"{output}_delta"] = safe_delta
        fields[f"{output}_delta_invalid"] = invalid
    fields["d3_cache_reuse_observed"] = bool(
        reused
        and (
            int(fields["d3_fragment_cache_hits_delta"]) > 0
            or int(fields["d3_fragment_store_hits_delta"]) > 0
            or int(fields["d3_composed_guide_hits_delta"]) > 0
            or int(fields["d3_fast_reactivation_hits_delta"]) > 0
        )
    )
    return fields


def run_measured_query(
    args: argparse.Namespace,
    runtime: ModeRuntime,
    filter_name: str,
    selectivity: float,
    predicate: str,
    query_no: int,
    query_id: int,
    repeat: int,
    truth: dict[tuple[str, int], TruthEntry],
    schedule_position: int,
    block_no: int = 0,
    query_order_position: int = 0,
) -> dict[str, object]:
    mode = runtime.mode
    error = ""
    ids: list[int] = []
    distances: list[float] = []
    activation_profile: dict[str, object] = {}
    scan_profile: dict[str, object] = {}
    cache_profile: dict[str, object] = {}
    guidance_before: dict[str, object] = {}
    guidance_after: dict[str, object] = {}
    cache_before: dict[str, object] = {}
    d3_evidence: dict[str, object] = {}
    activation_ms = 0.0
    query_ms = 0.0
    end_to_end_ms = 0.0
    error_detail = ""
    table, index = mode_table_index(args, mode, filter_name)
    query_table = query_table_for_candidate(args, table)
    self_exclusion = candidate_self_exclusion(args, table)
    client_self_exclusion = (
        uses_exact_predicate_scan_contract(args.guidance_filter_strategy) and self_exclusion
    )
    try:
        runtime.cur.execute("SELECT vector_hnsw_reset_scan_profile()")
        guidance_before = read_guidance_profile(runtime.cur)
        cache_before = read_cache_profile(runtime.cur)
        e2e_started = time.perf_counter()
        table, index, previous_guidance_policy, reset_performed = (
            route_runtime_request(
            args, runtime, filter_name
            )
        )
        same_predicate_before = (
            mode == "design1_bloom_bfs_layout_d3"
            and runtime.d3_last_filter_name == filter_name
        )
        activation_profile = activate(
            runtime.cur,
            args,
            mode,
            filter_name,
            read_profile=False,
            reset_bypass_guidance=(
                previous_guidance_policy and not reset_performed
            ),
            configure_search_strategy=False,
        )
        if mode == "design1_bloom_bfs_layout_d3":
            runtime.d3_last_filter_name = filter_name
        activation_completed = time.perf_counter()
        table = str(activation_profile["table"])
        index = str(activation_profile["index"])
        binding = activation_binding(args, mode, filter_name, activation_profile)
        ids, distances, query_metadata = run_query(
            runtime.cur,
            table,
            predicate,
            query_id,
            args.k,
            binding,
            client_self_exclusion,
            candidate_validity_predicate=getattr(
                args, "candidate_validity_predicate", ""
            ),
            query_table=query_table,
            query_id_column=getattr(args, "query_id_column", "id"),
            query_vector_column=getattr(args, "query_vector_column", "embedding"),
            self_exclusion=self_exclusion,
            reset_profile=False,
            read_profile=False,
        )
        query_completed = time.perf_counter()
        activation_ms = (activation_completed - e2e_started) * 1000.0
        query_ms = (query_completed - activation_completed) * 1000.0
        end_to_end_ms = (query_completed - e2e_started) * 1000.0
        scan_profile = read_scan_profile(runtime.cur)
        scan_profile.update(query_metadata)
        guidance_after = read_guidance_profile(runtime.cur)
        cache_profile = read_cache_profile(runtime.cur)
        activation_profile = {**guidance_after, **activation_profile}
        if (
            mode == "design1_bloom_bfs_layout_d3"
            and str(guidance_after.get("adaptive_state", "")) == "rejected"
            and not bool(guidance_after.get("active", False))
        ):
            # Activation avoids a profile round-trip inside the timed path, so
            # activated_atoms=0 cannot distinguish a probe from a sticky
            # conservative rejection until the post-query profile is read.
            activation_profile["guidance_enabled"] = False
            activation_profile["guidance_route"] = "d3_admission_bypass"
        if (
            args.guidance_filter_strategy == "traversal_guided"
            and mode in {"design1_bloom", "design1_bloom_bfs_layout"}
            and not activation_profile.get("guidance_enabled")
        ):
            raise RuntimeError(
                f"formal {mode} measurement disabled traversal guidance: "
                f"{activation_profile.get('guidance_route', 'unknown')}"
            )
        if mode == "design1_bloom_bfs_layout_d3":
            d3_evidence = d3_phase_evidence(
                guidance_before,
                guidance_after,
                cache_before,
                cache_profile,
                activation_profile,
                same_predicate_before=same_predicate_before,
            )
            getattr(args, "d3_phase_evidence", []).append(
                {
                    "filter_name": filter_name,
                    "query_no": query_no,
                    "repeat": repeat,
                    **d3_evidence,
                }
            )
        if activation_profile.get("guidance_enabled"):
            contract_failure = guidance_scan_contract_failure(
                scan_profile,
                args.guidance_filter_strategy,
                bool(runtime.config.get("traversal_guided_prioritization", False)),
                int(runtime.config.get("traversal_guided_burst", 8)),
            )
            if contract_failure:
                raise RuntimeError(
                    "active guidance did not execute the required measured HNSW path: "
                    + contract_failure
                )
    except errors.QueryCanceled as exc:
        error = exc.__class__.__name__
        error_detail = str(exc)
        recover_runtime(args, runtime)
    except Exception as exc:  # noqa: BLE001
        error = exc.__class__.__name__
        error_detail = str(exc)
        recover_runtime(args, runtime)

    truth_entry = truth[(filter_name, query_no)]
    return {
        "selectivity": selectivity,
        "filter_name": filter_name,
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "table": table,
        "index": index,
        "query_table": query_table,
        "query_id_column": getattr(args, "query_id_column", "id"),
        "query_vector_column": getattr(args, "query_vector_column", "embedding"),
        "candidate_validity_predicate": effective_candidate_validity_predicate(
            getattr(args, "candidate_validity_predicate", "")
        ),
        "candidate_validity_predicate_sha256": candidate_validity_sha256(
            getattr(args, "candidate_validity_predicate", "")
        ),
        "d2_page_access": args.d2_page_access if mode_uses_d2(mode) else "off",
        "d2_index_page_access": args.d2_index_page_access if mode_uses_d2(mode) else "off",
        "preferred_index_guc": getattr(args, "preferred_index_guc", "hnsw.preferred_index"),
        "preferred_index_current_setting": runtime.preferred_index_current_setting or "",
        "backend_pid": int((runtime.backend_cpu_provenance or {}).get("backend_pid", 0)),
        "backend_cpu_requested": str(
            (runtime.backend_cpu_provenance or {}).get("requested_cpu_list", "")
        ),
        "backend_cpu_observed": str(
            (runtime.backend_cpu_provenance or {}).get("observed_cpu_list", "")
        ),
        "backend_cpu_exact_match": (
            (runtime.backend_cpu_provenance or {}).get("exact_match")
        ),
        "backend_cpu_pinning_attempted_by_runner": False,
        "ef_search": runtime.config["ef_search"],
        "effective_ef_search": runtime.ef_search_current_setting,
        "max_scan_tuples": (
            runtime.max_scan_tuples_current_setting
            if runtime.max_scan_tuples_current_setting is not None
            else runtime.config.get("max_scan_tuples", MODE_CONFIG_DEFAULTS["max_scan_tuples"])
        ),
        "scan_mem_multiplier": (
            runtime.scan_mem_multiplier_current_setting
            if runtime.scan_mem_multiplier_current_setting is not None
            else runtime.config.get("scan_mem_multiplier", MODE_CONFIG_DEFAULTS["scan_mem_multiplier"])
        ),
        "iterative_scan": (
            runtime.iterative_scan_current_setting
            if runtime.iterative_scan_current_setting is not None
            else runtime.config.get("iterative_scan", MODE_CONFIG_DEFAULTS["iterative_scan"])
        ),
        "guided_collect_target": (
            runtime.guided_collect_target_current_setting
            if runtime.guided_collect_target_current_setting is not None
            else runtime.config.get("guided_collect_target", MODE_CONFIG_DEFAULTS["guided_collect_target"])
        ),
        "traversal_guided_target": runtime.traversal_guided_target_current_setting,
        "mode_default_traversal_guided_target": runtime.config.get(
            "traversal_guided_target", 40
        ),
        "traversal_guided_early_stop": (
            runtime.traversal_guided_early_stop_current_setting
        ),
        "traversal_guided_early_stop_distance_ratio": (
            runtime.traversal_guided_early_stop_distance_ratio_current_setting
        ),
        "pair_key": pair_key(filter_name, query_no, repeat),
        "block_no": block_no,
        "query_order_position": query_order_position,
        "execution_order": getattr(args, "execution_order", "mode_major"),
        "schedule_seed": getattr(args, "schedule_seed", 20260718),
        "schedule_position": schedule_position,
        "query_no": query_no,
        "query_id": query_id,
        "repeat": repeat,
        "k": args.k,
        "scan_limit": (
            args.k + 1
            if client_self_exclusion
            else args.k
        ),
        "self_exclusion_contract": (
            "limit_k_plus_1_client_remove_query_id"
            if client_self_exclusion
            else "sql_residual_id_not_equal" if self_exclusion else "none_external_query_source"
        ),
        "recall": tie_aware_recall(distances, truth_entry, args.k) if not error else 0.0,
        "recall_contract": "distance_squared_threshold_tie_aware_v1",
        "truth_filtered_rows": truth_entry.filtered_rows,
        "truth_kth_distance_sq": truth_entry.kth_distance_sq,
        "truth_tie_tolerance": truth_entry.tie_tolerance,
        "truth_strict_closer_count": truth_entry.strict_closer_count,
        "truth_boundary_tied": truth_entry.boundary_tied,
        "truth_self_excluded": truth_entry.self_excluded,
        "truth_candidate_validity_predicate": effective_candidate_validity_predicate(
            getattr(args, "candidate_validity_predicate", "")
        ),
        "truth_candidate_validity_predicate_sha256": candidate_validity_sha256(
            getattr(args, "candidate_validity_predicate", "")
        ),
        "activation_ms": activation_ms,
        "query_latency_ms": query_ms,
        "end_to_end_ms": end_to_end_ms,
        "guidance_enabled": bool(activation_profile.get("guidance_enabled", mode != "original")),
        "guidance_scan_verified": (
            not bool(activation_profile.get("guidance_enabled", mode != "original"))
            or guidance_scan_contract_satisfied(
                scan_profile,
                args.guidance_filter_strategy,
                bool(runtime.config.get("traversal_guided_prioritization", False)),
                int(runtime.config.get("traversal_guided_burst", 8)),
            )
        ),
        # Kept for CSV consumers that predate the clearer scan-specific name.
        "guidance_binding_verified": (
            not bool(activation_profile.get("guidance_enabled", mode != "original"))
            or guidance_scan_contract_satisfied(
                scan_profile,
                args.guidance_filter_strategy,
                bool(runtime.config.get("traversal_guided_prioritization", False)),
                int(runtime.config.get("traversal_guided_burst", 8)),
            )
        ),
        "planner_proof_verified": runtime.planner_proof_verified,
        "guidance_route": str(activation_profile.get("guidance_route", "")),
        "guidance_kind": str(activation_profile.get("guidance_kind", "")),
        "activation_atom_count": activation_profile.get("activation_atom_count", 0),
        "d3_active_guidance_reused": bool(
            d3_evidence.get("d3_active_guidance_reused", False)
        ),
        **d3_evidence,
        "d3_initialization": d3_initialization_label(args),
        "d3_fragment_store_namespace": str(
            getattr(args, "d3_fragment_store_namespace", "")
        ),
        "prebuilt_fragments": 0,
        "warmup_all_queries": bool(getattr(args, "warmup_all_queries", False)),
        "guidance_filter_strategy": args.guidance_filter_strategy,
        "total_scan_ms": scan_profile.get("total_scan_ms", 0.0),
        "hnsw_search_ms": scan_profile.get("hnsw_search_ms", 0.0),
        "heap_fetch_ms": scan_profile.get("heap_fetch_ms", 0.0),
        "vector_search_ms": scan_profile.get("vector_search_ms", 0.0),
        "hnsw_am_callback_ms": scan_profile.get("hnsw_am_callback_ms", 0.0),
        "executor_residual_ms": scan_profile.get("executor_residual_ms", 0.0),
        "index_readbuffer_calls": scan_profile.get("index_readbuffer_calls", 0),
        "index_readbuffer_ms": scan_profile.get("index_readbuffer_ms", 0.0),
        "index_readbuffer_shared_read_calls": scan_profile.get("index_readbuffer_shared_read_calls", 0),
        "index_readbuffer_shared_read_ms": scan_profile.get("index_readbuffer_shared_read_ms", 0.0),
        "index_readbuffer_shared_hit_calls": scan_profile.get("index_readbuffer_shared_hit_calls", 0),
        "index_readbuffer_shared_hit_ms": scan_profile.get("index_readbuffer_shared_hit_ms", 0.0),
        "index_readbuffer_unclassified_calls": scan_profile.get("index_readbuffer_unclassified_calls", 0),
        "index_readbuffer_unclassified_ms": scan_profile.get("index_readbuffer_unclassified_ms", 0.0),
        "index_readbuffer_timing_scope": scan_profile.get("index_readbuffer_timing_scope", ""),
        "index_readbuffer_classification_scope": scan_profile.get("index_readbuffer_classification_scope", ""),
        "distance_compute_timed_calls": scan_profile.get("distance_compute_timed_calls", 0),
        "distance_compute_ms": scan_profile.get("distance_compute_ms", 0.0),
        "distance_compute_timing_scope": scan_profile.get("distance_compute_timing_scope", ""),
        "hnsw_remaining_ms": scan_profile.get("hnsw_remaining_ms", 0.0),
        "hnsw_remaining_ms_is_residual": scan_profile.get(
            "hnsw_remaining_ms_is_residual", False
        ),
        "hnsw_remaining_scope": scan_profile.get("hnsw_remaining_scope", ""),
        "profile_timer_overhead_scope": scan_profile.get("profile_timer_overhead_scope", ""),
        "heap_fetch_ms_is_residual_proxy": scan_profile.get(
            "heap_fetch_ms_is_residual_proxy", False
        ),
        "visited_tuples": scan_profile.get("visited_tuples", 0),
        "returned_tuples": scan_profile.get("returned_tuples", 0),
        "distance_compute_count": scan_profile.get("distance_compute_count", 0),
        "traversal_result_target": scan_profile.get("traversal_result_target", 0),
        "traversal_guided_result_count": scan_profile.get(
            "traversal_guided_result_count", 0
        ),
        "traversal_max_scan_reached": scan_profile.get(
            "traversal_max_scan_reached", False
        ),
        "traversal_guided_early_stop_enabled": scan_profile.get(
            "traversal_guided_early_stop_enabled", False
        ),
        "traversal_guided_early_stop_distance_ratio_effective": scan_profile.get(
            "traversal_guided_early_stop_distance_ratio", 0.0
        ),
        "traversal_guided_early_stop_terminations": scan_profile.get(
            "traversal_guided_early_stop_terminations", 0
        ),
        "page_access_batches": scan_profile.get("page_access_batches", 0),
        "page_access_candidates": scan_profile.get("page_access_candidates", 0),
        "page_access_prefetches": scan_profile.get("page_access_prefetches", 0),
        "page_access_distinct_pages": scan_profile.get("page_access_distinct_pages", 0),
        "index_page_prefetches": scan_profile.get("index_page_prefetches", 0),
        "profile_semantics_version": scan_profile.get("profile_semantics_version", 0),
        "index_page_loads": scan_profile.get("index_page_loads", 0),
        "index_page_runs": scan_profile.get("index_page_runs", 0),
        "index_page_distinct_pages": scan_profile.get("index_page_distinct_pages", 0),
        "index_page_distinct_pages_exact": scan_profile.get("index_page_distinct_pages_exact", False),
        "index_page_profile_scope": scan_profile.get("index_page_profile_scope", ""),
        "index_page_transition_count": scan_profile.get("index_page_transition_count", 0),
        "index_page_same_block_transitions": scan_profile.get("index_page_same_block_transitions", 0),
        "index_page_within_1_page_transitions": scan_profile.get("index_page_within_1_page_transitions", 0),
        "index_page_within_4_pages_transitions": scan_profile.get("index_page_within_4_pages_transitions", 0),
        "index_page_within_16_pages_transitions": scan_profile.get("index_page_within_16_pages_transitions", 0),
        "index_page_backward_transitions": scan_profile.get("index_page_backward_transitions", 0),
        "index_page_total_abs_block_delta": scan_profile.get("index_page_total_abs_block_delta", 0),
        "index_page_max_abs_block_delta": scan_profile.get("index_page_max_abs_block_delta", 0),
        "index_page_trace_statistics_scope": scan_profile.get("index_page_trace_statistics_scope", ""),
        "index_page_trace_sample_limit": scan_profile.get("index_page_trace_sample_limit", 0),
        "index_page_trace_sample_count": scan_profile.get("index_page_trace_sample_count", 0),
        "index_page_trace_sample_truncated": scan_profile.get("index_page_trace_sample_truncated", False),
        "index_page_trace_sample_scope": scan_profile.get("index_page_trace_sample_scope", ""),
        "index_page_trace_sample": json.dumps(
            scan_profile.get("index_page_trace_sample", []), separators=(",", ":")
        ),
        "heap_tid_returns": scan_profile.get("heap_tid_returns", 0),
        "heap_tid_page_runs": scan_profile.get("heap_tid_page_runs", 0),
        "heap_tid_distinct_pages": scan_profile.get("heap_tid_distinct_pages", 0),
        "heap_tid_distinct_pages_exact": scan_profile.get("heap_tid_distinct_pages_exact", False),
        "heap_tid_sequence_scope": scan_profile.get("heap_tid_sequence_scope", ""),
        "idx_blks_hit": scan_profile.get("idx_blks_hit", 0),
        "idx_blks_read": scan_profile.get("idx_blks_read", 0),
        "heap_blks_hit": scan_profile.get("heap_blks_hit", 0),
        "heap_blks_read": scan_profile.get("heap_blks_read", 0),
        "heap_blks_are_exact_heap_io": scan_profile.get("heap_blks_are_exact_heap_io", True),
        "guidance_checks": scan_profile.get("guidance_checks", 0),
        "guidance_skips": scan_profile.get("guidance_skips", 0),
        "traversal_expanded_nodes": scan_profile.get("traversal_expanded_nodes", 0),
        "traversal_neighbors_examined": scan_profile.get("traversal_neighbors_examined", 0),
        "traversal_guidance_checks": scan_profile.get("traversal_guidance_checks", 0),
        "traversal_guidance_matches": scan_profile.get("traversal_guidance_matches", 0),
        "traversal_guidance_misses": scan_profile.get("traversal_guidance_misses", 0),
        "traversal_matching_expanded": scan_profile.get("traversal_matching_expanded", 0),
        "traversal_bridge_expanded": scan_profile.get("traversal_bridge_expanded", 0),
        "traversal_candidate_admissions": scan_profile.get("traversal_candidate_admissions", 0),
        "traversal_result_admissions": scan_profile.get("traversal_result_admissions", 0),
        "traversal_guided_admissions": scan_profile.get("traversal_guided_admissions", 0),
        "traversal_guided_suppressions": scan_profile.get("traversal_guided_suppressions", 0),
        "traversal_heap_tids_suppressed": scan_profile.get("traversal_heap_tids_suppressed", 0),
        "traversal_stop_deferrals": scan_profile.get("traversal_stop_deferrals", 0),
        "traversal_discarded_pushes": scan_profile.get("traversal_discarded_pushes", 0),
        "traversal_discarded_pops": scan_profile.get("traversal_discarded_pops", 0),
        "traversal_initial_batches": scan_profile.get("traversal_initial_batches", 0),
        "traversal_resume_batches": scan_profile.get("traversal_resume_batches", 0),
        "traversal_strict_order_drops": scan_profile.get("traversal_strict_order_drops", 0),
        "traversal_stock_terminations": scan_profile.get("traversal_stock_terminations", 0),
        "traversal_max_scan_terminations": scan_profile.get("traversal_max_scan_terminations", 0),
        "traversal_exhausted_terminations": scan_profile.get("traversal_exhausted_terminations", 0),
        "neighbor_expansion_guidance_checks": scan_profile.get("neighbor_expansion_guidance_checks", 0),
        "neighbor_expansion_guidance_matches": scan_profile.get("neighbor_expansion_guidance_matches", 0),
        "neighbor_expansion_guidance_misses": scan_profile.get("neighbor_expansion_guidance_misses", 0),
        "pre_distance_membership_checks": scan_profile.get("pre_distance_membership_checks", 0),
        "pre_distance_membership_matches": scan_profile.get("pre_distance_membership_matches", 0),
        "pre_distance_membership_misses": scan_profile.get("pre_distance_membership_misses", 0),
        "distance_computations_avoided_attempted": scan_profile.get("distance_computations_avoided_attempted", 0),
        "distance_computations_avoided": scan_profile.get("distance_computations_avoided", 0),
        "guided_expanded_nodes": scan_profile.get("guided_expanded_nodes", 0),
        "guided_phase_distance_computations": scan_profile.get("guided_phase_distance_computations", 0),
        "stock_phase_expanded_nodes": scan_profile.get("stock_phase_expanded_nodes", 0),
        "stock_phase_distance_computations": scan_profile.get("stock_phase_distance_computations", 0),
        "stock_bypass_requests": scan_profile.get("stock_bypass_requests", 0),
        "stock_bypass_reason": scan_profile.get("stock_bypass_reason", ""),
        "fallback_requests": scan_profile.get("fallback_requests", 0),
        "fallback_reason": scan_profile.get("fallback_reason", ""),
        "fallback_stock_expanded_nodes": scan_profile.get("fallback_stock_expanded_nodes", 0),
        "fallback_stock_distance_computations": scan_profile.get("fallback_stock_distance_computations", 0),
        "fallback_iterative_scan_enabled": scan_profile.get(
            "fallback_iterative_scan_enabled", False
        ),
        "effective_iterative_scan": scan_profile.get(
            "effective_iterative_scan", scan_profile.get("iterative_scan", "off")
        ),
        "traversal_estimated_skip_rate_valid": scan_profile.get("traversal_estimated_skip_rate_valid", False),
        "traversal_estimated_skip_rate": scan_profile.get("traversal_estimated_skip_rate", 0.0),
        "traversal_prioritization_burst": scan_profile.get("traversal_prioritization_burst", 0),
        "approximate_prioritization_attempted": scan_profile.get("approximate_prioritization_attempted", False),
        "traversal_order_changed": scan_profile.get("traversal_order_changed", False),
        "approximate_ann_path": scan_profile.get("approximate_ann_path", False),
        "match_frontier_pops": scan_profile.get("match_frontier_pops", 0),
        "no_bridge_frontier_pops": scan_profile.get("no_bridge_frontier_pops", 0),
        "no_bridge_deferred": scan_profile.get("no_bridge_deferred", 0),
        "priority_reorders": scan_profile.get("priority_reorders", 0),
        "max_no_bridge_debt": scan_profile.get("max_no_bridge_debt", 0),
        "dual_frontier_termination_checks": scan_profile.get("dual_frontier_termination_checks", 0),
        "dual_frontier_terminations": scan_profile.get("dual_frontier_terminations", 0),
        "traversal_guidance_scope": scan_profile.get("traversal_guidance_scope", ""),
        "graph_expansion_pruned": scan_profile.get("graph_expansion_pruned", False),
        "distance_computations_pruned": scan_profile.get("distance_computations_pruned", False),
        "final_path": scan_profile.get("final_path", ""),
        "planner_proof_attempted": scan_profile.get("planner_proof_attempted", False),
        "planner_proof_succeeded": scan_profile.get("planner_proof_succeeded", False),
        "planner_proof_bypass_reason": scan_profile.get("planner_proof_bypass_reason", ""),
        "fragment_cache_hits": activation_profile.get("fragment_cache_hits", 0),
        "fragment_cache_misses": activation_profile.get("fragment_cache_misses", 0),
        "fragment_store_hits": activation_profile.get("fragment_store_hits", 0),
        "fragment_builds": activation_profile.get("fragment_builds", 0),
        "fast_reactivation_hits": activation_profile.get("fast_reactivation_hits", 0),
        "composed_guide_hit": activation_profile.get("composed_guide_hit", False),
        "activation_build_ms": activation_profile.get("last_cache_build_ms", 0.0),
        "activation_memory_bytes": activation_profile.get("last_cache_memory_bytes", 0),
        "cache_resident_bytes": cache_profile.get("resident_bytes", 0),
        "cache_resident_entries": cache_profile.get("resident_entries", 0),
        "cache_evictions": cache_profile.get("evictions", 0),
        "composed_guide_entries": cache_profile.get("composed_guide_entries", 0),
        "composed_guide_hits_total": cache_profile.get("composed_guide_hits", 0),
        "adaptive_state": activation_profile.get("adaptive_state", "stock"),
        "adaptive_requests": activation_profile.get("adaptive_requests", 0),
        "adaptive_probes": activation_profile.get("adaptive_probes", 0),
        "adaptive_admissions": activation_profile.get("adaptive_admissions", 0),
        "adaptive_page_builds": activation_profile.get("adaptive_page_builds", 0),
        "adaptive_bloom_builds": activation_profile.get("adaptive_bloom_builds", 0),
        "adaptive_refinements": activation_profile.get("adaptive_refinements", 0),
        "adaptive_rejections": activation_profile.get("adaptive_rejections", 0),
        "adaptive_bytes": activation_profile.get("adaptive_bytes", cache_profile.get("adaptive_bytes", 0)),
        "adaptive_score": activation_profile.get("adaptive_score", cache_profile.get("adaptive_score", 0.0)),
        "sqlens_build_id": str(
            (runtime.sqlens_runtime_identity or {}).get("observed_build_id", "")
        ),
        "vector_so_sha256": str(
            (runtime.sqlens_runtime_identity or {}).get("observed_vector_so_sha256", "")
        ),
        "returned": len(ids),
        "raw_returned_before_self_exclusion": scan_profile.get(
            "sqlens_raw_returned_before_self_exclusion", len(ids)
        ),
        "ids": ",".join(str(x) for x in ids),
        "result_distances": json.dumps(distances, separators=(",", ":")),
        "error": error,
        "error_detail": error_detail,
    }


def print_progress(
    rows: list[dict[str, object]],
    mode: str,
    filter_name: str,
    query_index: int,
    query_count: int,
) -> None:
    ok = [r for r in rows if r["mode"] == mode and r["filter_name"] == filter_name and not r["error"]]
    if ok:
        print(
            f"progress mode={mode} filter={filter_name} queries={query_index}/{query_count} "
            f"e2e={statistics.fmean(float(r['end_to_end_ms']) for r in ok):.2f}ms",
            flush=True,
        )


def print_workload_progress(
    rows: list[dict[str, object]],
    mode: str,
    request_index: int,
    request_count: int,
) -> None:
    ok = [r for r in rows if r["mode"] == mode and not r["error"]]
    if ok:
        print(
            f"progress mode={mode} workload_requests={request_index}/{request_count} "
            f"e2e={statistics.fmean(float(r['end_to_end_ms']) for r in ok):.2f}ms",
            flush=True,
        )


def workload_warmup_requests(
    workload_requests: list[WorkloadRequest],
    warmup_queries: int,
    warmup_all_queries: bool,
) -> list[WorkloadRequest]:
    if warmup_all_queries:
        return list(workload_requests)
    if warmup_queries <= 0:
        return []
    counts: dict[str, int] = {}
    selected: list[WorkloadRequest] = []
    for request in workload_requests:
        count = counts.get(request.filter_name, 0)
        if count >= warmup_queries:
            continue
        counts[request.filter_name] = count + 1
        selected.append(request)
    return selected


def run_mode(
    args: argparse.Namespace,
    mode: str,
    filters: list[tuple[str, float, str]],
    query_nos: list[int],
    query_by_no: dict[int, int],
    truth: dict[tuple[str, int], TruthEntry],
    workload_requests: list[WorkloadRequest] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    runtime = open_mode_runtime(args, mode, filters)
    try:
        if workload_requests is not None:
            filter_by_name = {
                name: (selectivity, predicate)
                for name, selectivity, predicate in filters
            }
            if mode_uses_unmeasured_warmup(args, mode):
                for request in workload_warmup_requests(
                    workload_requests,
                    args.warmup_queries,
                    args.warmup_all_queries,
                ):
                    _, predicate = filter_by_name[request.filter_name]
                    run_warmup(
                        args,
                        runtime,
                        request.filter_name,
                        predicate,
                        request.query_id,
                    )

            schedule_position = getattr(args, "modes", [mode]).index(mode) + 1
            block_no = 0
            for repeat in range(args.repeats):
                scheduled = list(workload_requests)
                random.Random(args.schedule_seed + 104729 * repeat).shuffle(scheduled)
                for query_position, request in enumerate(scheduled, start=1):
                    selectivity, predicate = filter_by_name[request.filter_name]
                    row = run_measured_query(
                        args,
                        runtime,
                        request.filter_name,
                        selectivity,
                        predicate,
                        request.query_no,
                        request.query_id,
                        repeat,
                        truth,
                        schedule_position,
                        block_no,
                        query_position,
                    )
                    row.update(
                        {
                            "request_no": request.request_no,
                            "trace_cycle": request.trace_cycle,
                            "workload_split": request.split,
                        }
                    )
                    rows.append(row)
                    block_no += 1
                    if args.progress_queries and query_position % args.progress_queries == 0:
                        print_progress(
                            rows,
                            mode,
                            request.filter_name,
                            query_position,
                            len(scheduled),
                        )
            return rows

        warm_nos = query_nos if args.warmup_all_queries else query_nos[: args.warmup_queries]
        if mode_uses_unmeasured_warmup(args, mode):
            for filter_name, _, predicate in filters:
                for qno in warm_nos:
                    run_warmup(args, runtime, filter_name, predicate, query_by_no[qno])

        schedule_position = getattr(args, "modes", [mode]).index(mode) + 1
        for filter_name, selectivity, predicate in filters:
            for idx, qno in enumerate(query_nos, start=1):
                for repeat in range(args.repeats):
                    rows.append(
                        run_measured_query(
                            args,
                            runtime,
                            filter_name,
                            selectivity,
                            predicate,
                            qno,
                            query_by_no[qno],
                            repeat,
                            truth,
                            schedule_position,
                        )
                    )
                if args.progress_queries and idx % args.progress_queries == 0:
                    print_progress(rows, mode, filter_name, idx, len(query_nos))
    finally:
        close_mode_runtime(runtime)
    return rows


def run_interleaved(
    args: argparse.Namespace,
    filters: list[tuple[str, float, str]],
    query_nos: list[int],
    query_by_no: dict[int, int],
    truth: dict[tuple[str, int], TruthEntry],
    workload_requests: list[WorkloadRequest] | None = None,
) -> list[dict[str, object]]:
    if bool(getattr(args, "isolate_repeat_runtimes", False)):
        if workload_requests is None:
            raise RuntimeError(
                "--isolate-repeat-runtimes requires a frozen --workload-csv"
            )
        return run_interleaved_isolated_repeats(
            args,
            filters,
            query_by_no,
            truth,
            workload_requests,
        )

    rows: list[dict[str, object]] = []
    runtimes: dict[str, ModeRuntime] = {}
    warmup_block = 0
    measured_block = 0
    try:
        for mode in args.modes:
            runtimes[mode] = open_mode_runtime(args, mode, filters)

        if workload_requests is not None:
            filter_by_name = {
                name: (selectivity, predicate)
                for name, selectivity, predicate in filters
            }
            warm_requests = workload_warmup_requests(
                workload_requests,
                args.warmup_queries,
                args.warmup_all_queries,
            )
            for request in warm_requests:
                _, predicate = filter_by_name[request.filter_name]
                warmup_modes = [
                    mode
                    for mode in args.modes
                    if mode_uses_unmeasured_warmup(args, mode)
                ]
                for mode in balanced_mode_order(
                    warmup_modes, warmup_block, args.schedule_seed
                ):
                    run_warmup(
                        args,
                        runtimes[mode],
                        request.filter_name,
                        predicate,
                        request.query_id,
                    )
                warmup_block += 1

            for repeat in range(args.repeats):
                scheduled = list(workload_requests)
                random.Random(args.schedule_seed + 104729 * repeat).shuffle(scheduled)
                for query_position, request in enumerate(scheduled, start=1):
                    selectivity, predicate = filter_by_name[request.filter_name]
                    mode_order = balanced_mode_order(
                        args.modes, measured_block, args.schedule_seed
                    )
                    for position, mode in enumerate(mode_order, start=1):
                        row = run_measured_query(
                            args,
                            runtimes[mode],
                            request.filter_name,
                            selectivity,
                            predicate,
                            request.query_no,
                            request.query_id,
                            repeat,
                            truth,
                            position,
                            measured_block,
                            query_position,
                        )
                        row.update(
                            {
                                "request_no": request.request_no,
                                "trace_cycle": request.trace_cycle,
                                "workload_split": request.split,
                            }
                        )
                        rows.append(row)
                    measured_block += 1
                    if (
                        args.progress_queries
                        and query_position % args.progress_queries == 0
                    ):
                        for mode in args.modes:
                            print_workload_progress(
                                rows,
                                mode,
                                query_position,
                                len(scheduled),
                            )
            return rows

        warm_nos = query_nos if args.warmup_all_queries else query_nos[: args.warmup_queries]
        for filter_name, _, predicate in filters:
            for qno in warm_nos:
                warmup_modes = [
                    mode
                    for mode in args.modes
                    if mode_uses_unmeasured_warmup(args, mode)
                ]
                for mode in balanced_mode_order(warmup_modes, warmup_block, args.schedule_seed):
                    run_warmup(args, runtimes[mode], filter_name, predicate, query_by_no[qno])
                warmup_block += 1

        for filter_no, (filter_name, selectivity, predicate) in enumerate(filters):
            completed_queries = 0
            for repeat in range(args.repeats):
                repeat_query_nos = list(query_nos)
                random.Random(args.schedule_seed + 1009 * filter_no + 104729 * repeat).shuffle(repeat_query_nos)
                for query_position, qno in enumerate(repeat_query_nos, start=1):
                    mode_order = balanced_mode_order(args.modes, measured_block, args.schedule_seed)
                    for position, mode in enumerate(mode_order, start=1):
                        rows.append(
                            run_measured_query(
                                args,
                                runtimes[mode],
                                filter_name,
                                selectivity,
                                predicate,
                                qno,
                                query_by_no[qno],
                                repeat,
                                truth,
                                position,
                                measured_block,
                                query_position,
                            )
                        )
                    measured_block += 1
                    completed_queries += 1
                if args.progress_queries and completed_queries % args.progress_queries == 0:
                    for mode in args.modes:
                        print_progress(rows, mode, filter_name, completed_queries, len(query_nos) * args.repeats)
    finally:
        for runtime in reversed(list(runtimes.values())):
            close_mode_runtime(runtime)
    return rows


def run_interleaved_isolated_repeats(
    args: argparse.Namespace,
    filters: list[tuple[str, float, str]],
    query_by_no: dict[int, int],
    truth: dict[tuple[str, int], TruthEntry],
    workload_requests: list[WorkloadRequest],
) -> list[dict[str, object]]:
    """Run each repeat with fresh sessions and a disjoint D3 namespace."""
    rows = list(getattr(args, "workload_checkpoint_rows", []))
    completed_by_repeat = dict(
        getattr(args, "workload_checkpoint_completed_by_repeat", {})
    )
    filter_by_name = {
        name: (selectivity, predicate)
        for name, selectivity, predicate in filters
    }
    warm_requests = workload_warmup_requests(
        workload_requests,
        args.warmup_queries,
        args.warmup_all_queries,
    )
    base_namespace = str(args.d3_fragment_store_namespace)
    warmup_block = 0
    measured_block = 0
    try:
        for repeat in range(args.repeats):
            args.d3_fragment_store_namespace = repeat_fragment_store_namespace(
                base_namespace,
                repeat,
            )
            runtimes: dict[str, ModeRuntime] = {}
            try:
                for mode in args.modes:
                    runtimes[mode] = open_mode_runtime(args, mode, filters)

                for request in warm_requests:
                    _, predicate = filter_by_name[request.filter_name]
                    warmup_modes = [
                        mode
                        for mode in args.modes
                        if mode_uses_unmeasured_warmup(args, mode)
                    ]
                    for mode in balanced_mode_order(
                        warmup_modes,
                        warmup_block,
                        args.schedule_seed,
                    ):
                        run_warmup(
                            args,
                            runtimes[mode],
                            request.filter_name,
                            predicate,
                            request.query_id,
                        )
                    warmup_block += 1

                scheduled = list(workload_requests)
                random.Random(
                    args.schedule_seed + 104729 * repeat
                ).shuffle(scheduled)
                completed_prefix = int(completed_by_repeat.get(repeat, 0))
                if completed_prefix:
                    print(
                        "resuming repeat="
                        f"{repeat} after {completed_prefix}/{len(scheduled)} "
                        "complete workload requests",
                        flush=True,
                    )
                measured_block += completed_prefix
                for query_position, request in enumerate(scheduled, start=1):
                    if query_position <= completed_prefix:
                        continue
                    selectivity, predicate = filter_by_name[request.filter_name]
                    mode_order = balanced_mode_order(
                        args.modes,
                        measured_block,
                        args.schedule_seed,
                    )
                    for position, mode in enumerate(mode_order, start=1):
                        row = run_measured_query(
                            args,
                            runtimes[mode],
                            request.filter_name,
                            selectivity,
                            predicate,
                            request.query_no,
                            request.query_id,
                            repeat,
                            truth,
                            position,
                            measured_block,
                            query_position,
                        )
                        row.update(
                            {
                                "request_no": request.request_no,
                                "trace_cycle": request.trace_cycle,
                                "workload_split": request.split,
                            }
                        )
                        rows.append(row)
                    measured_block += 1
                    checkpoint_every = int(
                        getattr(args, "checkpoint_every_workload_requests", 0)
                    )
                    if checkpoint_every and (
                        query_position % checkpoint_every == 0
                        or query_position == len(scheduled)
                    ):
                        write_workload_checkpoint(args, rows)
                    if (
                        args.progress_queries
                        and query_position % args.progress_queries == 0
                    ):
                        for mode in args.modes:
                            print_workload_progress(
                                rows,
                                mode,
                                query_position,
                                len(scheduled),
                            )
            finally:
                for runtime in reversed(list(runtimes.values())):
                    close_mode_runtime(runtime)
    finally:
        args.d3_fragment_store_namespace = base_namespace
    return rows


def workload_checkpoint_path(args: argparse.Namespace) -> Path:
    return args.out.with_suffix(args.out.suffix + ".checkpoint.csv")


def write_workload_checkpoint(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
) -> None:
    """Atomically persist only complete workload request pairs."""
    if not rows:
        return
    modes = list(args.modes)
    if len(rows) % len(modes):
        raise RuntimeError(
            "refusing to checkpoint a partial interleaved request pair"
        )
    path = workload_checkpoint_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(dict.fromkeys(field for row in rows for field in row))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    args.workload_checkpoint_last_rows = len(rows)


def load_workload_checkpoint(
    args: argparse.Namespace,
    workload_requests: list[WorkloadRequest],
) -> tuple[list[dict[str, object]], dict[int, int]]:
    """Load a checkpoint only when it is a complete schedule prefix."""
    path = workload_checkpoint_path(args)
    if not path.exists():
        return [], {}
    if not bool(getattr(args, "resume_from_checkpoint", False)):
        raise RuntimeError(
            f"workload checkpoint already exists; pass --resume-from-checkpoint "
            f"or remove it: {path}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "repeat",
            "request_no",
            "query_order_position",
            "block_no",
            "mode",
            "error",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                f"workload checkpoint is missing columns: {sorted(missing)}"
            )
        rows = list(reader)
    modes = list(args.modes)
    if len(rows) % len(modes):
        raise RuntimeError("workload checkpoint ends with a partial mode pair")
    completed_by_repeat: dict[int, int] = {}
    offset = 0
    for repeat in range(args.repeats):
        scheduled = list(workload_requests)
        random.Random(args.schedule_seed + 104729 * repeat).shuffle(scheduled)
        completed = 0
        for query_position, request in enumerate(scheduled, start=1):
            if offset >= len(rows):
                break
            block = rows[offset : offset + len(modes)]
            if len(block) != len(modes):
                raise RuntimeError("workload checkpoint contains a partial mode pair")
            expected_order = balanced_mode_order(
                modes,
                repeat * len(scheduled) + query_position - 1,
                args.schedule_seed,
            )
            observed_order = [str(row["mode"]) for row in block]
            if observed_order != expected_order:
                raise RuntimeError(
                    "workload checkpoint mode order differs from the deterministic "
                    f"schedule at repeat={repeat}, position={query_position}: "
                    f"expected={expected_order}, observed={observed_order}"
                )
            for row in block:
                if int(row["repeat"]) != repeat:
                    raise RuntimeError("workload checkpoint repeat order is not contiguous")
                if int(row["request_no"]) != request.request_no:
                    raise RuntimeError(
                        "workload checkpoint is not a prefix of the deterministic schedule"
                    )
                if int(row["query_order_position"]) != query_position:
                    raise RuntimeError(
                        "workload checkpoint query position does not match its schedule"
                    )
                if str(row.get("error", "")).strip():
                    raise RuntimeError("workload checkpoint contains a query error")
            offset += len(modes)
            completed += 1
        if completed:
            completed_by_repeat[repeat] = completed
        if completed < len(scheduled):
            break
    if offset != len(rows):
        raise RuntimeError(
            "workload checkpoint contains rows after an incomplete schedule prefix"
        )
    return rows, completed_by_repeat


def validate_execution_lifecycle(
    args: argparse.Namespace,
    filters: list[tuple[str, float, str]],
    query_nos: list[int],
    workload_requests: list[WorkloadRequest] | None = None,
) -> dict[str, object]:
    backend_evidence = getattr(args, "backend_cpu_evidence", [])
    runtime_multiplier = (
        args.repeats
        if bool(getattr(args, "isolate_repeat_runtimes", False))
        else 1
    )
    expected_runtime_count = len(args.modes) * runtime_multiplier
    if len(backend_evidence) != expected_runtime_count or any(
        int(item.get("backend_pid") or 0) <= 0
        or not item.get("observed_cpu_list")
        or item.get("pinning_attempted_by_runner") is not False
        for item in backend_evidence
    ):
        raise RuntimeError(
            "production backend CPU provenance is incomplete: "
            f"expected {expected_runtime_count}, observed {len(backend_evidence)}"
        )
    requested_cpu_list = str(getattr(args, "backend_cpu_list", None) or "")
    if requested_cpu_list and any(
        item.get("exact_match") is not True for item in backend_evidence
    ):
        raise RuntimeError("one or more production backends failed the requested CPU affinity gate")
    runtime_identities = getattr(args, "runtime_sqlens_identity_evidence", [])
    if len(runtime_identities) != expected_runtime_count or any(
        item.get("exact_match") is not True
        or item.get("expected_build_id") != args.expected_sqlens_build_id
        or item.get("expected_vector_so_sha256") != args.expected_vector_so_sha256
        for item in runtime_identities
    ):
        raise RuntimeError(
            "production backend SQLens identity evidence is incomplete or mismatched"
        )
    warm_modes = [
        mode for mode in args.modes if mode_uses_unmeasured_warmup(args, mode)
    ]
    if workload_requests is None:
        warm_query_count = len(query_nos) if args.warmup_all_queries else min(
            len(query_nos), args.warmup_queries
        )
        expected_warmups = len(filters) * warm_query_count * len(warm_modes)
        requests_per_filter = {
            filter_name: len(query_nos)
            for filter_name, _, _ in filters
        }
        measured_requests_per_mode = len(filters) * len(query_nos) * args.repeats
    else:
        warm_request_count = len(
            workload_warmup_requests(
                workload_requests,
                args.warmup_queries,
                args.warmup_all_queries,
            )
        )
        expected_warmups = warm_request_count * len(warm_modes)
        requests_per_filter = {
            filter_name: sum(
                request.filter_name == filter_name
                for request in workload_requests
            )
            for filter_name, _, _ in filters
        }
        measured_requests_per_mode = len(workload_requests) * args.repeats
    if bool(getattr(args, "isolate_repeat_runtimes", False)):
        expected_warmups *= args.repeats
    warmup_evidence = getattr(args, "warmup_evidence", [])
    if len(warmup_evidence) != expected_warmups or any(
        item.get("status") != "complete" for item in warmup_evidence
    ):
        raise RuntimeError(
            "warmup evidence is incomplete or failed: "
            f"expected {expected_warmups}, observed {len(warmup_evidence)}"
        )

    phase_counts: dict[str, dict[str, int]] = {}
    d3_admission_outcomes: dict[str, str] = {}
    d3_evidence = getattr(args, "d3_phase_evidence", [])
    if "design1_bloom_bfs_layout_d3" in args.modes:
        policy = d3_measurement_policy(args)
        expected_d3_requests = measured_requests_per_mode
        if len(d3_evidence) != expected_d3_requests:
            raise RuntimeError(
                "D3 phase evidence is incomplete: "
                f"expected {expected_d3_requests}, observed {len(d3_evidence)}"
            )
        for filter_name, selectivity, _ in filters:
            counts = {phase: 0 for phase in D3_PHASES}
            for item in d3_evidence:
                if item.get("filter_name") == filter_name:
                    phase = str(item.get("d3_phase") or "")
                    if phase in counts:
                        counts[phase] += 1
            if policy == "workload_driven_adaptive":
                expected_per_filter = requests_per_filter[filter_name] * args.repeats
                policy_guidance_enabled = should_enable_guidance(
                    args, filter_name
                )[0]
                policy_bypass = (
                    not policy_guidance_enabled
                    and counts == {
                        "probe": 0,
                        "admission": 0,
                        "refinement": 0,
                        "warm": 0,
                        "bypass": expected_per_filter,
                    }
                )
                guided_transition = (
                    counts["admission"] > 0 or counts["refinement"] > 0
                )
                admitted = guided_transition or counts["warm"] > 0
                complete_admission = (
                    counts["probe"] > 0
                    and guided_transition
                    and counts["warm"] > 0
                )
                complete_bypass = (
                    counts["probe"] > 0
                    and counts["bypass"] > 0
                    and not admitted
                )
                if not (complete_admission or complete_bypass or policy_bypass):
                    raise RuntimeError(
                        f"D3 lifecycle is incomplete for filter={filter_name}: {counts}"
                    )
                d3_admission_outcomes[filter_name] = (
                    "policy_bypass"
                    if policy_bypass
                    else "adaptive_admission"
                    if complete_admission
                    else "adaptive_bypass"
                )
            else:
                expected_per_filter = requests_per_filter[filter_name] * args.repeats
                if counts != {
                    "probe": 0,
                    "admission": 0,
                    "refinement": 0,
                    "warm": expected_per_filter,
                    "bypass": 0,
                }:
                    raise RuntimeError(
                        "D3 admitted-warm measured rows are not exclusively warm for "
                        f"filter={filter_name}: {counts}"
                    )
                d3_admission_outcomes[filter_name] = "admitted_warm_reuse"
            phase_counts[filter_name] = counts
        if sum(sum(counts.values()) for counts in phase_counts.values()) != (
            expected_d3_requests
        ):
            raise RuntimeError(
                "D3 lifecycle phase counts do not close to measured requests"
            )
        if policy == "admitted_warm_reuse":
            warmup_phases = getattr(args, "d3_warmup_phase_evidence", [])
            for filter_name, _, _ in filters:
                observed = {
                    str(item.get("d3_phase"))
                    for item in warmup_phases
                    if item.get("filter_name") == filter_name
                }
                if not {"probe", "admission", "warm"}.issubset(observed):
                    raise RuntimeError(
                        "D3 unmeasured admission lifecycle is incomplete for "
                        f"filter={filter_name}: {sorted(observed)}"
                    )
    return {
        "warmup_policy": d3_measurement_policy(args),
        "repeat_runtime_isolation": bool(
            getattr(args, "isolate_repeat_runtimes", False)
        ),
        "runtime_openings": expected_runtime_count,
        "backend_cpu_requested": requested_cpu_list,
        "backend_cpu_evidence_count": len(backend_evidence),
        "backend_cpu_provenance_complete": True,
        "runtime_sqlens_identity_evidence_count": len(runtime_identities),
        "runtime_sqlens_identity_complete": True,
        "warmup_expected": expected_warmups,
        "warmup_observed": len(warmup_evidence),
        "warmup_complete": True,
        "d3_expected_measured_requests": (
            measured_requests_per_mode
            if "design1_bloom_bfs_layout_d3" in args.modes
            else 0
        ),
        "d3_phase_counts": phase_counts,
        "d3_admission_outcomes": d3_admission_outcomes,
        "d3_warmup_phase_evidence_count": len(
            getattr(args, "d3_warmup_phase_evidence", [])
        ),
        "d3_lifecycle_complete": (
            "design1_bloom_bfs_layout_d3" not in args.modes or bool(phase_counts)
        ),
    }


def write_summary(rows: list[dict[str, object]], out: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["filter_name"]), str(row["mode"])), []).append(row)

    mode_mean: dict[tuple[str, str], float] = {}
    for key, items in grouped.items():
        ok = [r for r in items if not r["error"]]
        mode_mean[key] = statistics.fmean(float(r["end_to_end_ms"]) for r in ok) if ok else 0.0

    table_out = out.with_name(out.stem + "_table.csv")
    fields = [
        "Selectivity",
        "Filter",
        "Original pgvector",
        "Design 1",
        "Design 1 + Design 2",
        "Design 1 + Design 2 + Design 3",
        "D1 speedup",
        "D1+D2 speedup",
        "D1+D2 + D3 speedup",
    ]
    with table_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        seen_filters = []
        for row in rows:
            key = (str(row["filter_name"]), str(row["selectivity"]))
            if key not in seen_filters:
                seen_filters.append(key)
        for filter_name, selectivity in seen_filters:
            if (filter_name, "original") not in mode_mean:
                continue
            original = mode_mean[(filter_name, "original")]
            d1 = mode_mean.get((filter_name, "design1_bloom"), 0.0)
            d12 = mode_mean.get((filter_name, "design1_bloom_bfs_layout"), 0.0)
            d123 = mode_mean.get((filter_name, "design1_bloom_bfs_layout_d3"), 0.0)
            writer.writerow(
                {
					"Selectivity": str(selectivity),
                    "Filter": filter_name,
                    "Original pgvector": f"{original:.4f}",
                    "Design 1": f"{d1:.4f}",
                    "Design 1 + Design 2": f"{d12:.4f}",
                    "Design 1 + Design 2 + Design 3": f"{d123:.4f}",
                    "D1 speedup": f"{(original / d1):.4f}" if d1 else "0.0000",
                    "D1+D2 speedup": f"{(original / d12):.4f}" if d12 else "0.0000",
                    "D1+D2 + D3 speedup": f"{(original / d123):.4f}" if d123 else "0.0000",
                }
            )

    profile_out = out.with_name(out.stem + "_profile_summary.csv")
    profile_fields = [
        "filter_name",
        "mode",
        "ok",
        "errors",
        "recall_mean",
        "end_to_end_mean_ms",
        "activation_mean_ms",
        "query_latency_mean_ms",
        "total_scan_mean_ms",
        "hnsw_search_mean_ms",
        "heap_fetch_mean_ms",
        "vector_search_mean_ms",
        "hnsw_am_callback_mean_ms",
        "executor_residual_mean_ms",
        "index_readbuffer_calls_mean",
        "index_readbuffer_mean_ms",
        "index_readbuffer_shared_read_calls_mean",
        "index_readbuffer_shared_read_mean_ms",
        "index_readbuffer_shared_hit_calls_mean",
        "index_readbuffer_shared_hit_mean_ms",
        "index_readbuffer_unclassified_calls_mean",
        "index_readbuffer_unclassified_mean_ms",
        "distance_compute_timed_calls_mean",
        "distance_compute_mean_ms",
        "hnsw_remaining_mean_ms",
        "guidance_enabled_rate",
        "cache_resident_bytes_max",
        "fragment_cache_hits_mean",
        "fragment_store_hits_mean",
        "fragment_builds_mean",
        "composed_guide_hit_rate",
        "guidance_skip_rate",
        "traversal_expanded_nodes_mean",
        "traversal_neighbors_examined_mean",
        "traversal_guidance_checks_mean",
        "traversal_guidance_match_rate",
        "traversal_matching_expanded_mean",
        "traversal_bridge_expanded_mean",
        "traversal_candidate_admissions_mean",
        "traversal_result_admissions_mean",
        "traversal_guided_admissions_mean",
        "traversal_guided_suppressions_mean",
        "traversal_heap_tids_suppressed_mean",
        "traversal_stop_deferrals_mean",
        "traversal_discarded_pushes_mean",
        "traversal_discarded_pops_mean",
        "traversal_initial_batches_mean",
        "traversal_resume_batches_mean",
        "traversal_strict_order_drops_mean",
        "guided_final_path_rate",
        "planner_proof_success_rate",
        "pre_distance_membership_checks_mean",
        "pre_distance_membership_misses_mean",
        "distance_computations_avoided_mean",
        "guided_expanded_nodes_mean",
        "guided_phase_distance_computations_mean",
        "stock_bypass_requests_mean",
        "fallback_requests_mean",
        "index_page_loads_mean",
        "index_page_runs_mean",
        "index_page_distinct_pages_mean",
        "index_page_distinct_pages_exact_rate",
        "index_page_transition_count_mean",
        "index_page_same_block_transitions_mean",
        "index_page_within_1_page_transitions_mean",
        "index_page_within_4_pages_transitions_mean",
        "index_page_within_16_pages_transitions_mean",
        "index_page_backward_transitions_mean",
        "index_page_total_abs_block_delta_mean",
        "index_page_max_abs_block_delta_mean",
        "heap_tid_returns_mean",
        "heap_tid_page_runs_mean",
        "heap_tid_distinct_pages_mean",
        "heap_tid_distinct_pages_exact_rate",
        "idx_blks_hit_mean",
        "idx_blks_read_mean",
        "heap_blks_hit_mean",
        "heap_blks_read_mean",
        "heap_blks_exact_io_claim_rate",
    ]
    d3_phases = D3_PHASES
    for phase in d3_phases:
        profile_fields.extend(
            [
                f"d3_{phase}_count",
                f"d3_{phase}_recall_mean",
                f"d3_{phase}_end_to_end_mean_ms",
                f"d3_{phase}_activation_mean_ms",
                f"d3_{phase}_query_latency_mean_ms",
            ]
        )
    with profile_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=profile_fields)
        writer.writeheader()
        for (filter_name, mode), items in sorted(grouped.items()):
            ok = [r for r in items if not r["error"]]
            checks = statistics.fmean(float(r["guidance_checks"]) for r in ok) if ok else 0.0
            skips = statistics.fmean(float(r["guidance_skips"]) for r in ok) if ok else 0.0
            traversal_checks = statistics.fmean(float(r["traversal_guidance_checks"]) for r in ok) if ok else 0.0
            traversal_matches = statistics.fmean(float(r["traversal_guidance_matches"]) for r in ok) if ok else 0.0
            summary_row: dict[str, object] = {
                    "filter_name": filter_name,
                    "mode": mode,
                    "ok": len(ok),
                    "errors": len(items) - len(ok),
                    "recall_mean": statistics.fmean(float(r["recall"]) for r in ok) if ok else 0.0,
                    "end_to_end_mean_ms": statistics.fmean(float(r["end_to_end_ms"]) for r in ok) if ok else 0.0,
                    "activation_mean_ms": statistics.fmean(float(r["activation_ms"]) for r in ok) if ok else 0.0,
                    "query_latency_mean_ms": statistics.fmean(float(r["query_latency_ms"]) for r in ok) if ok else 0.0,
                    "total_scan_mean_ms": statistics.fmean(float(r["total_scan_ms"]) for r in ok) if ok else 0.0,
                    "hnsw_search_mean_ms": statistics.fmean(float(r["hnsw_search_ms"]) for r in ok) if ok else 0.0,
                    "heap_fetch_mean_ms": statistics.fmean(float(r["heap_fetch_ms"]) for r in ok) if ok else 0.0,
                    "vector_search_mean_ms": statistics.fmean(float(r["vector_search_ms"]) for r in ok) if ok else 0.0,
                    "hnsw_am_callback_mean_ms": statistics.fmean(float(r["hnsw_am_callback_ms"]) for r in ok) if ok else 0.0,
                    "executor_residual_mean_ms": statistics.fmean(float(r["executor_residual_ms"]) for r in ok) if ok else 0.0,
                    "index_readbuffer_calls_mean": statistics.fmean(float(r["index_readbuffer_calls"]) for r in ok) if ok else 0.0,
                    "index_readbuffer_mean_ms": statistics.fmean(float(r["index_readbuffer_ms"]) for r in ok) if ok else 0.0,
                    "index_readbuffer_shared_read_calls_mean": statistics.fmean(float(r["index_readbuffer_shared_read_calls"]) for r in ok) if ok else 0.0,
                    "index_readbuffer_shared_read_mean_ms": statistics.fmean(float(r["index_readbuffer_shared_read_ms"]) for r in ok) if ok else 0.0,
                    "index_readbuffer_shared_hit_calls_mean": statistics.fmean(float(r["index_readbuffer_shared_hit_calls"]) for r in ok) if ok else 0.0,
                    "index_readbuffer_shared_hit_mean_ms": statistics.fmean(float(r["index_readbuffer_shared_hit_ms"]) for r in ok) if ok else 0.0,
                    "index_readbuffer_unclassified_calls_mean": statistics.fmean(float(r["index_readbuffer_unclassified_calls"]) for r in ok) if ok else 0.0,
                    "index_readbuffer_unclassified_mean_ms": statistics.fmean(float(r["index_readbuffer_unclassified_ms"]) for r in ok) if ok else 0.0,
                    "distance_compute_timed_calls_mean": statistics.fmean(float(r["distance_compute_timed_calls"]) for r in ok) if ok else 0.0,
                    "distance_compute_mean_ms": statistics.fmean(float(r["distance_compute_ms"]) for r in ok) if ok else 0.0,
                    "hnsw_remaining_mean_ms": statistics.fmean(float(r["hnsw_remaining_ms"]) for r in ok) if ok else 0.0,
                    "guidance_enabled_rate": statistics.fmean(float(r["guidance_enabled"]) for r in ok) if ok else 0.0,
                    "cache_resident_bytes_max": max((int(r["cache_resident_bytes"]) for r in ok), default=0),
                    "fragment_cache_hits_mean": statistics.fmean(float(r["fragment_cache_hits"]) for r in ok) if ok else 0.0,
                    "fragment_store_hits_mean": statistics.fmean(float(r["fragment_store_hits"]) for r in ok) if ok else 0.0,
                    "fragment_builds_mean": statistics.fmean(float(r["fragment_builds"]) for r in ok) if ok else 0.0,
                    "composed_guide_hit_rate": statistics.fmean(1.0 if r["composed_guide_hit"] else 0.0 for r in ok) if ok else 0.0,
                    "guidance_skip_rate": skips / checks if checks else 0.0,
                    "traversal_expanded_nodes_mean": statistics.fmean(float(r["traversal_expanded_nodes"]) for r in ok) if ok else 0.0,
                    "traversal_neighbors_examined_mean": statistics.fmean(float(r["traversal_neighbors_examined"]) for r in ok) if ok else 0.0,
                    "traversal_guidance_checks_mean": traversal_checks,
                    "traversal_guidance_match_rate": traversal_matches / traversal_checks if traversal_checks else 0.0,
                    "traversal_matching_expanded_mean": statistics.fmean(float(r["traversal_matching_expanded"]) for r in ok) if ok else 0.0,
                    "traversal_bridge_expanded_mean": statistics.fmean(float(r["traversal_bridge_expanded"]) for r in ok) if ok else 0.0,
                    "traversal_candidate_admissions_mean": statistics.fmean(float(r["traversal_candidate_admissions"]) for r in ok) if ok else 0.0,
                    "traversal_result_admissions_mean": statistics.fmean(float(r["traversal_result_admissions"]) for r in ok) if ok else 0.0,
                    "traversal_guided_admissions_mean": statistics.fmean(float(r["traversal_guided_admissions"]) for r in ok) if ok else 0.0,
                    "traversal_guided_suppressions_mean": statistics.fmean(float(r["traversal_guided_suppressions"]) for r in ok) if ok else 0.0,
                    "traversal_heap_tids_suppressed_mean": statistics.fmean(float(r["traversal_heap_tids_suppressed"]) for r in ok) if ok else 0.0,
                    "traversal_stop_deferrals_mean": statistics.fmean(float(r["traversal_stop_deferrals"]) for r in ok) if ok else 0.0,
                    "traversal_discarded_pushes_mean": statistics.fmean(float(r["traversal_discarded_pushes"]) for r in ok) if ok else 0.0,
                    "traversal_discarded_pops_mean": statistics.fmean(float(r["traversal_discarded_pops"]) for r in ok) if ok else 0.0,
                    "traversal_initial_batches_mean": statistics.fmean(float(r["traversal_initial_batches"]) for r in ok) if ok else 0.0,
                    "traversal_resume_batches_mean": statistics.fmean(float(r["traversal_resume_batches"]) for r in ok) if ok else 0.0,
                    "traversal_strict_order_drops_mean": statistics.fmean(float(r["traversal_strict_order_drops"]) for r in ok) if ok else 0.0,
                    "guided_final_path_rate": statistics.fmean(1.0 if r["final_path"] == "guided" else 0.0 for r in ok) if ok else 0.0,
                    "planner_proof_success_rate": statistics.fmean(1.0 if r["planner_proof_succeeded"] else 0.0 for r in ok) if ok else 0.0,
                    "pre_distance_membership_checks_mean": statistics.fmean(float(r["pre_distance_membership_checks"]) for r in ok) if ok else 0.0,
                    "pre_distance_membership_misses_mean": statistics.fmean(float(r["pre_distance_membership_misses"]) for r in ok) if ok else 0.0,
                    "distance_computations_avoided_mean": statistics.fmean(float(r["distance_computations_avoided"]) for r in ok) if ok else 0.0,
                    "guided_expanded_nodes_mean": statistics.fmean(float(r["guided_expanded_nodes"]) for r in ok) if ok else 0.0,
                    "guided_phase_distance_computations_mean": statistics.fmean(float(r["guided_phase_distance_computations"]) for r in ok) if ok else 0.0,
                    "stock_bypass_requests_mean": statistics.fmean(float(r["stock_bypass_requests"]) for r in ok) if ok else 0.0,
                    "fallback_requests_mean": statistics.fmean(float(r["fallback_requests"]) for r in ok) if ok else 0.0,
                    "index_page_loads_mean": statistics.fmean(float(r["index_page_loads"]) for r in ok) if ok else 0.0,
                    "index_page_runs_mean": statistics.fmean(float(r["index_page_runs"]) for r in ok) if ok else 0.0,
                    "index_page_distinct_pages_mean": statistics.fmean(float(r["index_page_distinct_pages"]) for r in ok) if ok else 0.0,
                    "index_page_distinct_pages_exact_rate": statistics.fmean(1.0 if r["index_page_distinct_pages_exact"] else 0.0 for r in ok) if ok else 0.0,
                    "index_page_transition_count_mean": statistics.fmean(float(r["index_page_transition_count"]) for r in ok) if ok else 0.0,
                    "index_page_same_block_transitions_mean": statistics.fmean(float(r["index_page_same_block_transitions"]) for r in ok) if ok else 0.0,
                    "index_page_within_1_page_transitions_mean": statistics.fmean(float(r["index_page_within_1_page_transitions"]) for r in ok) if ok else 0.0,
                    "index_page_within_4_pages_transitions_mean": statistics.fmean(float(r["index_page_within_4_pages_transitions"]) for r in ok) if ok else 0.0,
                    "index_page_within_16_pages_transitions_mean": statistics.fmean(float(r["index_page_within_16_pages_transitions"]) for r in ok) if ok else 0.0,
                    "index_page_backward_transitions_mean": statistics.fmean(float(r["index_page_backward_transitions"]) for r in ok) if ok else 0.0,
                    "index_page_total_abs_block_delta_mean": statistics.fmean(float(r["index_page_total_abs_block_delta"]) for r in ok) if ok else 0.0,
                    "index_page_max_abs_block_delta_mean": statistics.fmean(float(r["index_page_max_abs_block_delta"]) for r in ok) if ok else 0.0,
                    "heap_tid_returns_mean": statistics.fmean(float(r["heap_tid_returns"]) for r in ok) if ok else 0.0,
                    "heap_tid_page_runs_mean": statistics.fmean(float(r["heap_tid_page_runs"]) for r in ok) if ok else 0.0,
                    "heap_tid_distinct_pages_mean": statistics.fmean(float(r["heap_tid_distinct_pages"]) for r in ok) if ok else 0.0,
                    "heap_tid_distinct_pages_exact_rate": statistics.fmean(1.0 if r["heap_tid_distinct_pages_exact"] else 0.0 for r in ok) if ok else 0.0,
                    "idx_blks_hit_mean": statistics.fmean(float(r["idx_blks_hit"]) for r in ok) if ok else 0.0,
                    "idx_blks_read_mean": statistics.fmean(float(r["idx_blks_read"]) for r in ok) if ok else 0.0,
                    "heap_blks_hit_mean": statistics.fmean(float(r["heap_blks_hit"]) for r in ok) if ok else 0.0,
                    "heap_blks_read_mean": statistics.fmean(float(r["heap_blks_read"]) for r in ok) if ok else 0.0,
                    "heap_blks_exact_io_claim_rate": statistics.fmean(1.0 if r["heap_blks_are_exact_heap_io"] else 0.0 for r in ok) if ok else 0.0,
                }
            for phase in d3_phases:
                phase_rows = [r for r in ok if r.get("d3_phase") == phase]
                summary_row[f"d3_{phase}_count"] = len(phase_rows)
                for field, output in (
                    ("recall", "recall_mean"),
                    ("end_to_end_ms", "end_to_end_mean_ms"),
                    ("activation_ms", "activation_mean_ms"),
                    ("query_latency_ms", "query_latency_mean_ms"),
                ):
                    summary_row[f"d3_{phase}_{output}"] = (
                        statistics.fmean(float(r[field]) for r in phase_rows)
                        if phase_rows
                        else 0.0
                    )
            writer.writerow(summary_row)
    print(f"wrote {table_out}", flush=True)
    print(f"wrote {profile_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Original, D1, D1+D2, and D1+D2+D3 pgvector variants.")
    parser.add_argument("--insertion-table", default=INSERTION_TABLE)
    parser.add_argument("--insertion-index", default=INSERTION_INDEX)
    parser.add_argument("--bfs-table", default=BFS_TABLE)
    parser.add_argument("--bfs-index", default=BFS_INDEX)
    parser.add_argument(
        "--query-table",
        help="External query relation. Omit to read each mode's query vector from its candidate table.",
    )
    parser.add_argument("--query-id-column", default="id")
    parser.add_argument("--query-vector-column", default="embedding")
    parser.add_argument(
        "--candidate-validity-predicate",
        type=validate_candidate_validity_predicate,
        default="embedding_valid",
        help=(
            "Global candidate validity expression implied by a partial HNSW index, such as "
            "embedding_valid. It is a SQL/planner qual and is never added to guidance atoms."
        ),
    )
    parser.add_argument(
        "--expected-truth-self-excluded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require this exact self_excluded value in every formal truth row.",
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path("results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_unique_embeddings_formal.csv"),
    )
    parser.add_argument(
        "--workload-csv",
        type=Path,
        help=(
            "Optional frozen mixed-filter request trace. When provided, each CSV row "
            "selects one (query, filter) pair and replaces the Cartesian filter/query schedule."
        ),
    )
    parser.add_argument(
        "--expected-workload-requests",
        type=int,
        default=0,
        help="Fail unless --workload-csv contains exactly this many requests; zero disables the count gate.",
    )
    parser.add_argument(
        "--workload-request-limit",
        type=int,
        default=0,
        help=(
            "After validating the complete frozen workload, execute only its "
            "first N requests; zero executes the full trace."
        ),
    )
    parser.add_argument(
        "--require-unique-workload-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require one unique query vector per mixed-workload request.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--orchestrator-source",
        type=Path,
        help="Parent runner source file bound by path and startup SHA-256.",
    )
    parser.add_argument(
        "--prewarm-relation",
        dest="prewarm_relations",
        action="append",
        default=[],
        help=(
            "Relation synchronously loaded with pg_prewarm(..., 'read', 'main') "
            "before measured runtimes open; repeatable."
        ),
    )
    parser.add_argument("--filters-csv", type=Path)
    parser.add_argument("--modes", nargs="*", choices=MODES, default=MODES)
    parser.add_argument("--execution-order", choices=["mode_major", "interleaved"], default="mode_major")
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument(
        "--mode-configs-json",
        type=parse_mode_configs_json,
        default={},
        help="JSON object or JSON file mapping modes to per-mode search-setting overrides.",
    )
    parser.add_argument(
        "--filter-ef-search-json",
        type=parse_filter_ef_search_json,
        default={},
        help=(
            "JSON object or JSON file mapping mode names, then filter names, to "
            "independently calibrated ef_search values."
        ),
    )
    parser.add_argument(
        "--filter-traversal-target-json",
        type=parse_filter_traversal_target_json,
        default={},
        help=(
            "JSON object or JSON file mapping mode names, then filter names, to "
            "independently calibrated traversal-guided result targets."
        ),
    )
    parser.add_argument(
        "--filter-mode-configs-json",
        type=parse_filter_mode_configs_json,
        default={},
        help=(
            "JSON object or file mapping mode, predicate, and any complete "
            "search configuration overrides used for that request."
        ),
    )
    parser.add_argument("--filter-names", nargs="*")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--isolate-repeat-runtimes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Open fresh PostgreSQL sessions and use a disjoint D3 fragment-store "
            "namespace for every repeat. Formal q10k latency uses this so repeats "
            "are independent and match the throughput lifecycle."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-workload-requests",
        type=int,
        default=0,
        help=(
            "Atomically checkpoint complete interleaved request pairs every N "
            "requests; zero disables checkpointing."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume a frozen interleaved workload from its validated checkpoint.",
    )
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument(
        "--warmup-all-queries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one unmeasured pass over every measured query for each filter before recording latency.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--guided-collect-target", type=int, default=1000)
    parser.add_argument(
        "--traversal-guided-target",
        type=int,
        default=40,
        help=(
            "Predicate-matching result target for prioritized traversal; must be at least "
            "k+1 when client-side self-exclusion is active, otherwise k, and no larger "
            "than ef_search. Tune it explicitly for matched recall."
        ),
    )
    parser.add_argument(
        "--traversal-guided-prioritization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable r11 bounded traversal prioritization for D1 modes. Original pgvector is "
            "always forced off; disabling this flag makes D1 validation/admission-only."
        ),
    )
    parser.add_argument(
        "--traversal-guided-burst",
        type=int,
        default=8,
        help="Maximum consecutive predicate-MAYBE frontier pops before expanding a NO bridge.",
    )
    parser.add_argument(
        "--traversal-guided-early-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop approximate guided traversal after collecting the configured target.",
    )
    parser.add_argument(
        "--traversal-guided-early-stop-distance-ratio",
        type=float,
        default=0.0,
        help=(
            "Distance-aware early-stop ratio in [0,1]. Zero is count-only; one "
            "matches stock distance termination, and intermediate values trade work for recall."
        ),
    )
    parser.add_argument(
        "--guidance-filter-strategy",
        default="traversal_guided",
        choices=["traversal_guided", "safe_guided", "guided_collect", "acorn1"],
        help=(
            "safe_guided is the formal candidate-validation D1 and preserves stock graph "
            "traversal/termination. traversal_guided is formal only when work counters prove "
            "traversal pruning. guided_collect and acorn1 remain diagnostic modes."
        ),
    )
    parser.add_argument("--iterative-scan", default="off", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument(
        "--guidance-bypass-iterative-scan",
        default="strict_order",
        choices=["off", "strict_order", "relaxed_order"],
        help="Iterative-scan mode restored when adaptive guidance is policy-bypassed.",
    )
    parser.add_argument(
        "--guidance-bypass-ef-search",
        type=int,
        default=0,
        help=(
            "Fixed ef_search for policy-bypassed requests; zero reuses the mode's "
            "configured ef_search. This is one global route policy, not per-filter tuning."
        ),
    )
    parser.add_argument(
        "--guidance-low-selectivity-bypass-ef-search",
        type=int,
        default=0,
        help=(
            "ef_search used when guidance is bypassed below the minimum selectivity; "
            "zero reuses --guidance-bypass-ef-search."
        ),
    )
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--d2-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--d2-index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument(
        "--preferred-index-guc",
        default="hnsw.preferred_index",
        help=(
            "Optional SQLens planner preference GUC. If the loaded C build exposes it, the runner "
            "sets it to the mode's expected index; EXPLAIN remains the fail-closed source of truth."
        ),
    )
    parser.add_argument(
        "--require-preferred-index-guc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail formal runs when the preferred-index GUC is unavailable. Disabling this is "
            "diagnostic only; the exact EXPLAIN Index Name gate still applies."
        ),
    )
    parser.add_argument("--d2-page-window", type=int, default=128)
    parser.add_argument("--d2-page-prefetch-min-items", type=int, default=2)
    parser.add_argument("--d2-page-disable-after-no-merge", type=int, default=2)
    parser.add_argument(
        "--d2-source-on-guidance-bypass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Route policy-bypassed predicates to the source HNSW layout while "
            "retaining BFS layout for admitted guided traversal."
        ),
    )
    parser.add_argument("--d1-cache-mb", type=int, default=1024)
    parser.add_argument(
        "--d1-guidance-kind",
        choices=["auto", "exact", "bloom"],
        default="auto",
        help="D1 fragment representation; auto uses exact TID sets for selective filters.",
    )
    parser.add_argument(
        "--d1-exact-max-selectivity-pct",
        type=float,
        default=2.5,
        help="Maximum observed selectivity routed to exact TID guidance in auto mode.",
    )
    parser.add_argument(
        "--collapse-exact-and-guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Materialize an exact multi-atom AND predicate as one final-result fragment. "
            "This avoids constructing broad component TID sets while preserving exact membership."
        ),
    )
    parser.add_argument("--d3-cache-mb", type=int, default=1024)
    parser.add_argument(
        "--d3-measurement-policy",
        choices=sorted(D3_MEASUREMENT_POLICIES),
        default="workload_driven_adaptive",
        help=(
            "workload_driven_adaptive charges probe/admission in measured requests; "
            "admitted_warm_reuse completes that lifecycle in the untimed warmup and "
            "requires every measured D3 request to reuse admitted guidance"
        ),
    )
    parser.add_argument(
        "--d3-fragment-store-namespace",
        type=fragment_store_namespace_arg,
        default="",
    )
    parser.add_argument("--d3-probe-requests", type=int, default=2)
    parser.add_argument("--d3-min-benefit-per-byte", type=float, default=0.0)
    parser.add_argument("--d3-max-fragment-mb", type=int, default=16)
    parser.add_argument("--d3-page-min-skip-rate", type=float, default=0.05)
    parser.add_argument(
        "--fragment-tracking-prepared",
        action="store_true",
        help=(
            "Assert that the parent prepared fragment epoch tracking before acquiring "
            "its long-lived data guard; child mode sessions must not run tracking DDL."
        ),
    )
    parser.add_argument(
        "--guidance-selectivity-min-pct",
        type=float,
        default=0.0,
        help=(
            "Disable predicate guidance below this filter percentage; zero keeps "
            "all low-selectivity predicates eligible."
        ),
    )
    parser.add_argument(
        "--guidance-selectivity-max-pct",
        type=float,
        default=100.0,
        help="Disable predicate guidance above this filter percentage; D1+D2 then runs as D2-only.",
    )
    parser.add_argument(
        "--guidance-composite-max-selectivity-pct",
        type=float,
        default=100.0,
        help=(
            "Disable guidance for multi-atom predicates above this percentage; "
            "this avoids constructing broad component fragments for a selective conjunction."
        ),
    )
    parser.add_argument(
        "--guidance-max-atoms",
        type=int,
        default=64,
        help="Disable predicate guidance when a query decomposes into more atoms than this.",
    )
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument(
        "--database-experiment-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Serialize benchmark processes that target the same PGPORT so "
            "latency runs cannot overlap on one PostgreSQL instance."
        ),
    )
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-queries", type=int, default=10)
    parser.add_argument("--reset-cache-per-query", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--d2-graph-proof-json",
        type=parse_json_object,
        help=(
            "Delegated proof produced by vector_hnsw_graph_compare at the parent formal-runner "
            "startup. Standalone runs compute the proof directly."
        ),
    )
    parser.add_argument("--expected-sqlens-build-id", required=True)
    parser.add_argument("--expected-vector-so-sha256", required=True)
    parser.add_argument(
        "--backend-cpu-list",
        type=normalize_cpu_list,
        help=(
            "Required DB-side Cpus_allowed_list for every production backend. The runner records "
            "pg_backend_pid() but never tasksets a Docker namespace PID."
        ),
    )
    args = parser.parse_args()
    args.candidate_validity_predicate_explicit = (
        "--candidate-validity-predicate" in sys.argv
    )
    args.candidate_validity_predicate = effective_candidate_validity_predicate(
        args.candidate_validity_predicate
    )
    args.plan_started_at = utc_now()
    core_source = Path(__file__).resolve()
    args.execution_source_evidence = {
        "core_runner": {
            "path": str(core_source),
            "sha256": sha256_file(core_source),
        }
    }
    if args.orchestrator_source is not None:
        orchestrator_source = args.orchestrator_source.resolve()
        if not orchestrator_source.is_file():
            raise RuntimeError(
                f"orchestrator source does not exist: {orchestrator_source}"
            )
        args.execution_source_evidence["orchestrator"] = {
            "path": str(orchestrator_source),
            "sha256": sha256_file(orchestrator_source),
        }
    args.plan_evidence_out = args.out.with_suffix(args.out.suffix + ".plan.json")
    args.plan_evidence = []
    args.output_rows = 0
    args.warmup_evidence = []
    args.d3_phase_evidence = []
    args.d3_warmup_phase_evidence = []
    args.backend_cpu_evidence = []
    args.runtime_sqlens_identity_evidence = []
    args.workload_checkpoint_rows = []
    args.workload_checkpoint_completed_by_repeat = {}
    args.workload_checkpoint_last_rows = 0
    args.database_experiment_lock_handle = None
    args.database_experiment_lock_evidence = {"enabled": False}
    if args.database_experiment_lock:
        (
            args.database_experiment_lock_handle,
            args.database_experiment_lock_evidence,
        ) = acquire_database_experiment_lock()
    try:
        if args.guidance_bypass_ef_search < 0:
            raise RuntimeError("guidance_bypass_ef_search must be non-negative")
        if args.checkpoint_every_workload_requests < 0:
            raise RuntimeError(
                "checkpoint_every_workload_requests must be non-negative"
            )
        if args.checkpoint_every_workload_requests and not (
            args.execution_order == "interleaved"
            and args.isolate_repeat_runtimes
            and args.workload_csv is not None
        ):
            raise RuntimeError(
                "workload checkpointing requires an isolated interleaved frozen workload"
            )
        if args.guidance_low_selectivity_bypass_ef_search < 0:
            raise RuntimeError(
                "guidance_low_selectivity_bypass_ef_search must be non-negative"
            )
        if not 0.0 <= args.guidance_selectivity_min_pct <= 100.0:
            raise RuntimeError("guidance_selectivity_min_pct must be in [0, 100]")
        if args.guidance_selectivity_min_pct > args.guidance_selectivity_max_pct:
            raise RuntimeError(
                "guidance_selectivity_min_pct must not exceed "
                "guidance_selectivity_max_pct"
            )
        quoted_column(args.query_id_column)
        quoted_column(args.query_vector_column)
        validate_query_source_contract(args)
        require_sqlens_provenance_from_env()
        args.sqlens_runtime_identity = require_exact_sqlens_identity_from_env(
            args.expected_sqlens_build_id,
            args.expected_vector_so_sha256,
        )
        if args.guidance_filter_strategy == "traversal_guided":
            invalid_modes = [
                mode
                for mode in args.modes
                if mode != "original"
                and effective_mode_config(args, mode)["iterative_scan"] != "off"
            ]
            if invalid_modes:
                raise RuntimeError(
                    "formal traversal_guided measurements require iterative_scan=off; "
                    f"invalid modes: {invalid_modes}"
                )
        if any(mode_uses_d2(mode) for mode in args.modes):
            args.d2_graph_proof = require_d2_graph_proof_from_env(
                args,
                args.d2_graph_proof_json,
            )
        else:
            args.d2_graph_proof = {"required": False}
        truth, query_by_no = load_tie_aware_truth(
            args.truth_csv,
            expected_self_excluded=args.expected_truth_self_excluded,
            expected_candidate_validity_predicate=(
                args.candidate_validity_predicate
                if args.candidate_validity_predicate_explicit
                else None
            ),
        )
        all_filters, args.filter_atoms = load_filter_specs(args.filters_csv)
        selected = set(args.filter_names or [])
        filters = [(name, target, pred) for name, target, pred in all_filters if not selected or name in selected]
        if not filters:
            raise RuntimeError("no benchmark filters selected")
        args.filter_predicate_by_name = {
            name: predicate for name, _, predicate in filters
        }
        args.filter_selectivity_by_name = {name: parse_pct(target) for name, target, _ in filters}
        workload_requests: list[WorkloadRequest] | None = None
        if args.workload_csv is not None:
            workload_requests = load_workload_requests(
                args.workload_csv,
                query_by_no=query_by_no,
                filters=all_filters,
                truth=truth,
                expected_requests=args.expected_workload_requests,
                request_limit=args.workload_request_limit,
                selected_filter_names=(
                    {name for name, _, _ in filters}
                    if selected
                    else None
                ),
                require_unique_queries=args.require_unique_workload_queries,
            )
            query_nos = sorted({request.query_no for request in workload_requests})
            args.workload_request_count = len(workload_requests)
            args.workload_unique_query_count = len(
                {request.query_id for request in workload_requests}
            )
        else:
            query_nos = sorted(query_by_no)[
                args.query_offset : args.query_offset + args.queries
            ]
            if len(query_nos) != args.queries:
                raise RuntimeError(
                    f"requested {args.queries} queries, found {len(query_nos)}"
                )
            args.workload_request_count = 0
            args.workload_unique_query_count = 0
        if workload_requests is not None:
            (
                args.workload_checkpoint_rows,
                args.workload_checkpoint_completed_by_repeat,
            ) = load_workload_checkpoint(args, workload_requests)
            args.workload_checkpoint_last_rows = len(
                args.workload_checkpoint_rows
            )
            args.d3_phase_evidence.extend(
                {
                    "filter_name": row["filter_name"],
                    "query_no": int(row["query_no"]),
                    "repeat": int(row["repeat"]),
                    "d3_phase": row.get("d3_phase", ""),
                }
                for row in args.workload_checkpoint_rows
                if row.get("mode") == "design1_bloom_bfs_layout_d3"
            )
        args.plan_query_id = query_by_no[query_nos[0]]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if "design1_bloom_bfs_layout_d3" in args.modes:
            if args.isolate_repeat_runtimes:
                repeat_records = [
                    fragment_store_namespace_evidence(
                        args.bfs_table,
                        repeat_fragment_store_namespace(
                            args.d3_fragment_store_namespace,
                            repeat,
                        ),
                    )
                    for repeat in range(args.repeats)
                ]
                args.d3_fragment_store_start_evidence = {
                    "required_empty": not bool(args.workload_checkpoint_rows),
                    "isolated_repeats": True,
                    "base_namespace": args.d3_fragment_store_namespace,
                    "records": repeat_records,
                }
                nonempty = [
                    record
                    for record in repeat_records
                    if record.get("empty") is not True
                ]
                if nonempty and not args.workload_checkpoint_rows:
                    raise RuntimeError(
                        "one or more per-repeat D3 fragment-store namespaces "
                        f"are not fresh: {nonempty}"
                    )
            else:
                args.d3_fragment_store_start_evidence = (
                    fragment_store_namespace_evidence(
                        args.bfs_table,
                        args.d3_fragment_store_namespace,
                    )
                )
                if not args.d3_fragment_store_start_evidence["empty"]:
                    raise RuntimeError(
                        "D3 fragment-store namespace is not fresh: "
                        f"{args.d3_fragment_store_namespace!r} has "
                        f"{args.d3_fragment_store_start_evidence['rows_before']} rows"
                    )
        args.relation_prewarm_evidence = prewarm_relations(args.prewarm_relations)

        rows: list[dict[str, object]] = []
        if args.execution_order == "interleaved":
            print(f"running interleaved modes={','.join(args.modes)} seed={args.schedule_seed}", flush=True)
            rows = run_interleaved(
                args,
                filters,
                query_nos,
                query_by_no,
                truth,
                workload_requests,
            )
        else:
            for mode in args.modes:
                print(f"running mode={mode}", flush=True)
                rows.extend(
                    run_mode(
                        args,
                        mode,
                        filters,
                        query_nos,
                        query_by_no,
                        truth,
                        workload_requests,
                    )
                )
        expected_plan_checks = expected_plan_evidence_count(args, filters)
        if len(args.plan_evidence) != expected_plan_checks or not all(
            bool(item.get("passed")) for item in args.plan_evidence
        ):
            raise RuntimeError(
                f"HNSW plan evidence incomplete: expected {expected_plan_checks}, got {len(args.plan_evidence)}"
            )
        query_errors = [row for row in rows if row.get("error")]
        args.query_error_summary = {
            "rows": len(rows),
            "error_rows": len(query_errors),
            "error_types": dict(
                Counter(str(row.get("error")) for row in query_errors)
            ),
        }
        if query_errors:
            failed_out = args.out.with_suffix(args.out.suffix + ".failed.csv")
            fieldnames = list(
                dict.fromkeys(field for row in rows for field in row)
            )
            with failed_out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            first = query_errors[0]
            raise RuntimeError(
                "formal query execution produced error rows: "
                f"count={len(query_errors)}, first_mode={first.get('mode')}, "
                f"first_filter={first.get('filter_name')}, "
                f"first_error={first.get('error_detail') or first.get('error')}"
            )
        args.execution_lifecycle = validate_execution_lifecycle(
            args,
            filters,
            query_nos,
            workload_requests,
        )
        args.sqlens_runtime_identity_final = require_exact_sqlens_identity_from_env(
            args.expected_sqlens_build_id,
            args.expected_vector_so_sha256,
        )
        if any(mode_uses_d2(mode) for mode in args.modes):
            args.d2_graph_proof_final = require_d2_graph_proof_from_env(
                args,
                args.d2_graph_proof,
            )
        else:
            args.d2_graph_proof_final = {"required": False}

        fieldnames = list(dict.fromkeys(field for row in rows for field in row))
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        args.output_rows = len(rows)
        print(f"wrote {args.out}", flush=True)
        write_summary(rows, args.out)
        write_plan_evidence(args, "complete")
    except BaseException as exc:
        write_plan_evidence(args, "failed", exc)
        raise
    finally:
        if args.database_experiment_lock_handle is not None:
            fcntl.flock(
                args.database_experiment_lock_handle.fileno(),
                fcntl.LOCK_UN,
            )
            args.database_experiment_lock_handle.close()


if __name__ == "__main__":
    main()
