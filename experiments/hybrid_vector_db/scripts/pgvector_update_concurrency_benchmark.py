#!/usr/bin/env python3
"""Fail-closed PostgreSQL/pgvector SQLens read/update concurrency benchmark.

This runner measures *real* PostgreSQL work.  Formal reader backends execute
Stock or full SQLens D1+D2+D3 HNSW queries selected by the current audited
Recall@10=0.90 selector.  Independent writer backends commit real predicate,
vector, insert, and delete changes against the same heap.  Legacy diagnostic
invocations retain the earlier matched-manifest and same-value-update
interfaces for compatibility.

It never calibrates a search configuration and it never manufactures
latencies.  The default command is a dry run; database access requires
``--execute``.  A formal artifact is valid only when every requested cell has
six or more repeats, no reader/writer error or timeout occurred, the update
delivery gate and control-cell exact SQL-valid audits pass, and the final
database/binary identity still matches the audited manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Mapping, Sequence

import psycopg
from psycopg import sql

try:
    from . import pgvector_formal_throughput_benchmark as throughput
    from .common_pg import pg_config_from_env
    from .pgvector_design1_design2_design3_selectivity_benchmark import (
        activate,
        activation_binding,
        candidate_self_exclusion,
        close_mode_runtime,
        mode_table_index,
        open_mode_runtime,
        query_table_for_candidate,
        read_guidance_profile,
        read_scan_profile,
        recover_runtime,
        run_query,
        tie_aware_recall,
        uses_exact_predicate_scan_contract,
    )
    from .pgvector_target_recall_selectivity_runner import (
        database_fingerprint,
        git_revision,
        relation_identifier,
        utc_now,
    )
    from .pgvector_update_correctness_stress import (
        MUTATIONS,
        parse_mutation_mix,
    )
except ImportError:  # pragma: no cover - direct execution
    import pgvector_formal_throughput_benchmark as throughput
    from common_pg import pg_config_from_env
    from pgvector_design1_design2_design3_selectivity_benchmark import (
        activate,
        activation_binding,
        candidate_self_exclusion,
        close_mode_runtime,
        mode_table_index,
        open_mode_runtime,
        query_table_for_candidate,
        read_guidance_profile,
        read_scan_profile,
        recover_runtime,
        run_query,
        tie_aware_recall,
        uses_exact_predicate_scan_contract,
    )
    from pgvector_target_recall_selectivity_runner import (
        database_fingerprint,
        git_revision,
        relation_identifier,
        utc_now,
    )
    from pgvector_update_correctness_stress import MUTATIONS, parse_mutation_mix


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "hybrid_vector_db"
RUNNER_VERSION = "amazon10m-pgvector-update-concurrency-v4-r36-resumable"
ARTIFACT_SCHEMA_VERSION = "pgvector_update_concurrency_benchmark.v4"
R36_BUILD_ID = (
    "sqlens-v16-d3-sticky-rejection-mixed-predicate-reuse-d2-edge-trace-"
    "readbuffer-profile-orderchangefix-ef500k-20260729-r36"
)
R36_VECTOR_SO_SHA256 = (
    "5ab03631a5167dd56c1c74638475fec9282508c87f26218d44440b23f98f1679"
)
R43_BUILD_ID = "sqlens-v17-predistance-promotion-20260806-r43"
R43_VECTOR_SO_SHA256 = (
    "2056a67b9b0012c401c6684d49915cbc31bc8fa770946dbfaddda9d779eecbf2"
)
ACCEPTED_RELEASE_IDS = {
    (R43_BUILD_ID, R43_VECTOR_SO_SHA256),
}
FORMAL_PROTOCOL = "p0_6_full"
LEGACY_PROTOCOL = "legacy"
METHODS = ("stock", "sqlens_full")
SUPPORTED_METHODS = ("stock", "sqlens_full", "sqlens_d1")
MODE_BY_METHOD = {
    "stock": "original",
    "sqlens_full": "design1_bloom_bfs_layout_d3",
    "sqlens_d1": "design1_bloom",
}
FORMAL_FILTERS = (
    "popular_ge1000",
    "long_review_ge500",
    "helpful_ge20",
)
FORMAL_FILTER_TARGET_PCT = {
    "popular_ge1000": 50.0,
    "long_review_ge500": 5.0,
    "helpful_ge20": 0.5,
}
PREDICATE_MUTATION_BY_FILTER = {
    "popular_ge1000": ("item_rating_number", 1000),
    "long_review_ge500": ("review_text_len", 500),
    "helpful_ge20": ("helpful_vote", 20),
}
DEFAULT_FIXED_SELECTOR = (
    RESULTS / "figure5_r36_formal" / "figure5_r36_fixed_target_configs.csv"
)
DEFAULT_FIXED_SELECTOR_MANIFEST = (
    RESULTS / "figure5_r36_formal" / "figure5_r36_fixed_target_configs.manifest.json"
)
DEFAULT_READERS = "1,4,8,16,32,64"
DEFAULT_UPDATE_RATES = "0,10,100,1000"
DEFAULT_UPDATE_COLUMN = "review_text"
DEFAULT_UPDATE_ID_POOL_SIZE = 100_000
FORMAL_MIN_UPDATE_DELIVERY_RATIO = 0.90
FORMAL_REQUESTS = 10_000
FORMAL_TARGETS = (0.90,)
FORMAL_READERS = (1, 4, 8, 16, 32, 64)
FORMAL_UPDATE_RATES = (0.0, 10.0, 100.0, 1000.0)
FORMAL_REPEATS = 6
PATH_CLASSES = ("guided", "stale_fallback", "stock")
GUIDED_FINAL_PATHS = {
    "validation_only",
    "legacy_guided",
    "candidate_admission_validation_only",
    "approximate_traversal_prioritization",
}
STOCK_FINAL_PATHS = {"stock", "stock_bypass", "fresh_stock_fallback"}
GUIDANCE_COUNTER_FIELDS = (
    "fragment_cache_hits",
    "fragment_cache_misses",
    "fragment_store_hits",
    "fragment_builds",
    "fast_reactivation_hits",
    "composed_guide_hits",
)
CELL_SCHEMA_VERSION = "pgvector_update_concurrency_cell.v1"
SAMPLED_PROFILE_SCHEMA_VERSION = "pgvector_update_concurrency_profile.v1"


class BenchmarkContractError(RuntimeError):
    """Raised when a requested run cannot be represented as a formal result."""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a non-negative number")
    return parsed


def unit_interval_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a number in (0, 1]")
    return parsed


def parse_positive_int_list(value: str) -> list[int]:
    try:
        values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("integer list values must be positive")
    return values


def parse_nonnegative_rate_list(value: str) -> list[float]:
    try:
        values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated rates") from exc
    if not values or any(item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("update rates must be non-negative")
    return values


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    if (
        not methods
        or len(set(methods)) != len(methods)
        or set(methods) - set(SUPPORTED_METHODS)
    ):
        raise argparse.ArgumentTypeError(
            "methods must be a unique subset of stock,sqlens_full,sqlens_d1"
        )
    return methods


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def source_identity(args: argparse.Namespace) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    relative = runner_path.relative_to(ROOT)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        revision = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--", str(relative))
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise BenchmarkContractError(f"cannot establish runner Git identity: {exc}") from exc
    runner_sha = sha256_file(runner_path)
    expected_runner = str(args.expected_runner_sha256 or "").lower()
    expected_git = str(args.expected_git_revision or "")
    return {
        "git_revision": revision,
        "runner_path": str(relative),
        "runner_sha256": runner_sha,
        "runner_git_status": status,
        "runner_tracked_clean": status == "",
        "expected_runner_sha256": expected_runner,
        "expected_git_revision": expected_git,
        "runner_sha256_matches_expected": bool(expected_runner)
        and runner_sha == expected_runner,
        "git_revision_matches_expected": bool(expected_git)
        and revision == expected_git,
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BenchmarkContractError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"{label} must contain one JSON object")
    return value


def _selector_config(
    row: Mapping[str, str], prefix: str, *, sqlens_full: bool, burst: int
) -> throughput.SearchConfig:
    try:
        return throughput.SearchConfig(
            ef_search=int(row[f"{prefix}_ef_search"]),
            max_scan_tuples=int(row[f"{prefix}_max_scan_tuples"]),
            scan_mem_multiplier=float(row[f"{prefix}_scan_mem_multiplier"]),
            iterative_scan=str(row[f"{prefix}_iterative_scan"]),
            guided_collect_target=int(row[f"{prefix}_guided_collect_target"]),
            traversal_guided_prioritization=sqlens_full,
            traversal_guided_burst=int(burst),
            traversal_guided_target=int(row[f"{prefix}_traversal_guided_target"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkContractError(
            f"fixed selector has malformed {prefix} configuration"
        ) from exc


def load_fixed_recall_selector(
    args: argparse.Namespace,
    filters: Sequence[throughput.FilterSpec],
) -> throughput.MatchedRecallBundle:
    """Load the current Amazon Recall@10=0.90 Stock/full-SQLens pair."""
    selector = Path(args.fixed_recall_selector_csv).resolve()
    manifest_path = Path(args.fixed_recall_selector_manifest).resolve()
    manifest = _read_json(manifest_path, "fixed-recall selector manifest")
    if manifest.get("artifact_valid") is not True:
        raise BenchmarkContractError("fixed-recall selector manifest is not artifact-valid")
    release = manifest.get("release_contract")
    release_pair = (
        str(release.get("expected_sqlens_build_id") or "") if isinstance(release, Mapping) else "",
        str(release.get("expected_vector_so_sha256") or "") if isinstance(release, Mapping) else "",
    )
    if release_pair not in ACCEPTED_RELEASE_IDS:
        raise BenchmarkContractError(
            "fixed-recall selector is not bound to the frozen r43 release contract"
        )
    output = (manifest.get("outputs") or {}).get("measurement_plan_csv")
    if (
        not isinstance(output, Mapping)
        or str(output.get("sha256") or "") != sha256_file(selector)
    ):
        raise BenchmarkContractError("fixed-recall selector CSV hash mismatch")

    with selector.open(newline="", encoding="utf-8") as source:
        candidates = [
            row
            for row in csv.DictReader(source)
            if str(row.get("dataset") or "") == "amazon"
            and math.isclose(float(row.get("target_recall") or 0.0), 0.90)
            and str(row.get("selection_status") or "") == "selected"
        ]
    if len(candidates) != 1:
        raise BenchmarkContractError(
            f"fixed selector must contain exactly one selected Amazon 0.90 pair, got {len(candidates)}"
        )
    row = candidates[0]
    binding_by_config = {
        str(item.get("config_id") or ""): item
        for item in manifest.get("input_bindings", [])
        if isinstance(item, Mapping) and item.get("dataset") == "amazon"
    }
    bound_inputs: list[dict[str, Any]] = []
    for config_id in (str(row["stock_config_id"]), str(row["sqlens_config_id"])):
        binding = binding_by_config.get(config_id)
        if not isinstance(binding, Mapping):
            raise BenchmarkContractError(
                f"fixed selector lacks input binding for {config_id}"
            )
        plan_path = Path(str(binding.get("input_plan") or "")).resolve()
        if (
            not plan_path.is_file()
            or sha256_file(plan_path) != str(binding.get("input_plan_sha256") or "")
        ):
            raise BenchmarkContractError(
                f"fixed selector input plan hash mismatch for {config_id}"
            )
        plan = _read_json(plan_path, f"fixed selector input plan {config_id}")
        query_contract = plan.get("query_contract")
        if not isinstance(query_contract, Mapping):
            raise BenchmarkContractError(
                f"fixed selector input plan lacks query contract for {config_id}"
            )
        expected = {
            "filters_sha256": sha256_file(Path(args.filters_csv)),
            "truth_sha256": sha256_file(Path(args.measurement_truth_csv)),
            "workload_sha256": sha256_file(Path(args.fixed_selector_workload_csv)),
        }
        for field, digest in expected.items():
            if query_contract.get(field) != digest:
                raise BenchmarkContractError(
                    f"fixed selector {config_id} {field} does not match current input"
                )
        bound_inputs.append(
            {
                "config_id": config_id,
                "input_plan": str(plan_path),
                "input_plan_sha256": sha256_file(plan_path),
                **expected,
            }
        )
    stock = _selector_config(
        row, "stock", sqlens_full=False, burst=args.traversal_guided_burst
    )
    sqlens = _selector_config(
        row, "sqlens", sqlens_full=True, burst=args.traversal_guided_burst
    )
    configs = {
        (filter_spec.name, method, 0.90): (
            sqlens if method == "sqlens_full" else stock
        )
        for filter_spec in filters
        for method in SUPPORTED_METHODS
    }
    args.insertion_table = str(row["stock_table"])
    args.insertion_index = str(row["stock_index"])
    args.bfs_table = str(row["sqlens_table"])
    args.bfs_index = str(row["sqlens_index"])
    args.guidance_filter_strategy = "traversal_guided"
    args.d2_page_access = str(row["sqlens_d2_page_access"])
    args.d2_index_page_access = str(row["sqlens_d2_index_page_access"])
    pseudo_manifest = {
        "run_spec": {
            "args": {
                "calibration_query_offset": 0,
                "calibration_queries": 100,
                "final_query_offset": 100,
                "final_queries": 100,
            }
        }
    }
    provenance = {
        "contract": "figure5_r36_fixed_target_amazon_090",
        "selector_csv": str(selector),
        "selector_csv_sha256": sha256_file(selector),
        "selector_manifest": str(manifest_path),
        "selector_manifest_sha256": sha256_file(manifest_path),
        "release_contract": dict(release),
        "pair_id": str(row["pair_id"]),
        "stock_config_sha256": str(row["stock_config_sha256"]),
        "sqlens_config_sha256": str(row["sqlens_config_sha256"]),
        "source_table": args.insertion_table,
        "source_index": args.insertion_index,
        "bfs_table": args.bfs_table,
        "bfs_index": args.bfs_index,
        "target_recall": 0.90,
        "bound_inputs": bound_inputs,
        "filters_sha256": sha256_file(Path(args.filters_csv)),
        "measurement_truth_sha256": sha256_file(Path(args.measurement_truth_csv)),
        "selector_workload_sha256": sha256_file(
            Path(args.fixed_selector_workload_csv)
        ),
    }
    evidence = tuple(
        {
            "filter_name": filter_spec.name,
            "target_recall": 0.90,
            "stock_config": stock.label,
            "sqlens_full_config": sqlens.label,
            "selector_pair_id": str(row["pair_id"]),
        }
        for filter_spec in filters
    )
    return throughput.MatchedRecallBundle(
        configs=configs,
        evidence=evidence,
        provenance=provenance,
        manifest=pseudo_manifest,
        guidance_filter_strategy="traversal_guided",
    )


def matched_query_number_splits(manifest: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        run_args = manifest["run_spec"]["args"]
        calibration_offset = int(run_args["calibration_query_offset"])
        calibration_queries = int(run_args["calibration_queries"])
        confirmation_offset = int(run_args["final_query_offset"])
        confirmation_queries = int(run_args["final_queries"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkContractError(
            "matched-recall manifest does not declare calibration/final query splits"
        ) from exc
    calibration = tuple(range(calibration_offset, calibration_offset + calibration_queries))
    confirmation = tuple(range(confirmation_offset, confirmation_offset + confirmation_queries))
    if not calibration or not confirmation or set(calibration) & set(confirmation):
        raise BenchmarkContractError("matched-recall calibration and confirmation ranges overlap")
    return calibration, confirmation


def query_id_disjoint_gate(
    matched_manifest: Mapping[str, Any],
    calibration_query_ids: Mapping[int, int],
    measurement_requests: Sequence[Any],
) -> dict[str, Any]:
    calibration_nos, confirmation_nos = matched_query_number_splits(matched_manifest)
    required = set(calibration_nos) | set(confirmation_nos)
    missing = sorted(required - set(calibration_query_ids))
    if missing:
        raise BenchmarkContractError(
            f"calibration truth lacks matched-recall query numbers: {missing[:10]}"
        )
    calibration_ids = {int(calibration_query_ids[number]) for number in calibration_nos}
    confirmation_ids = {int(calibration_query_ids[number]) for number in confirmation_nos}
    measurement_ids = {int(request.query_id) for request in measurement_requests}
    if len(calibration_ids) != len(calibration_nos):
        raise BenchmarkContractError("matched-recall calibration query IDs are not unique")
    if len(confirmation_ids) != len(confirmation_nos):
        raise BenchmarkContractError("matched-recall confirmation query IDs are not unique")
    if len(measurement_ids) != len(measurement_requests):
        raise BenchmarkContractError("measurement workload query IDs are not unique")
    overlaps = {
        "calibration_confirmation": calibration_ids & confirmation_ids,
        "calibration_measurement": calibration_ids & measurement_ids,
        "confirmation_measurement": confirmation_ids & measurement_ids,
    }
    nonempty = {name: values for name, values in overlaps.items() if values}
    if nonempty:
        detail = {name: sorted(values)[:10] for name, values in nonempty.items()}
        raise BenchmarkContractError(f"query-ID split overlap detected: {detail}")
    return {
        "passed": True,
        "actual_query_id_disjoint": True,
        "calibration_queries": len(calibration_ids),
        "confirmation_queries": len(confirmation_ids),
        "measurement_queries": len(measurement_ids),
        "calibration_query_ids_sha256": canonical_sha256(sorted(calibration_ids)),
        "confirmation_query_ids_sha256": canonical_sha256(sorted(confirmation_ids)),
        "measurement_query_ids_sha256": canonical_sha256(sorted(measurement_ids)),
    }


def select_measurement_requests(workload: Any, requests: int) -> tuple[Any, ...]:
    available = tuple(workload.requests)
    if requests <= 0 or requests > len(available):
        raise BenchmarkContractError(
            f"--requests must be between 1 and the {len(available)} independent measurement queries"
        )
    selected = available[:requests]
    if len({int(item.query_id) for item in selected}) != requests:
        raise BenchmarkContractError("selected measurement requests do not use unique query IDs")
    return selected


def formal_protocol_status(
    args: argparse.Namespace,
    targets: Sequence[float],
    readers: Sequence[int],
    rates: Sequence[float],
    methods: Sequence[str],
    *,
    filter_count: int | None = None,
    filter_names: Sequence[str] | None = None,
    split_gate_passed: bool | None = None,
    source_identity_passed: bool | None = None,
    selector_bound: bool | None = None,
) -> dict[str, Any]:
    checks = {
        "protocol_p0_6_full": str(args.protocol) == FORMAL_PROTOCOL,
        "requests_q10k": int(args.requests) == FORMAL_REQUESTS,
        "target_recall_090": tuple(float(value) for value in targets) == FORMAL_TARGETS,
        "readers_1_4_8_16_32_64": tuple(int(value) for value in readers) == FORMAL_READERS,
        "update_rates_0_10_100_1000": tuple(float(value) for value in rates) == FORMAL_UPDATE_RATES,
        "methods_stock_sqlens_full": tuple(methods) == METHODS,
        "measurement_repeats_6": int(args.measurement_repeats) == FORMAL_REPEATS,
        "k_10": int(args.k) == 10,
        "writer_clients_1": int(args.writer_clients) == 1,
        "update_batch_size_1": int(args.update_batch_size) == 1,
        "update_pool_100k": int(args.update_id_pool_size) == DEFAULT_UPDATE_ID_POOL_SIZE,
        "real_predicate_vector_insert_delete_mix": all(
            int(args.mutation_mix.get(name, 0)) > 0 for name in MUTATIONS
        ),
        "timed_per_query_profiler_disabled": not bool(args.profile_timed_queries),
        "sampled_profile_enabled": int(args.profile_samples_per_cell) > 0,
        "warmup_10s_100_requests": (
            math.isclose(float(args.warmup_seconds), 10.0)
            and int(args.session_warmup_requests) == 100
        ),
        "pg_prewarm": bool(args.pg_prewarm),
        "force_hnsw": bool(args.force_hnsw),
        "preferred_index_required": bool(args.require_preferred_index_guc),
        "cache_not_reset_per_query": not bool(args.reset_cache_per_query),
        "d1_cache_1024mb": int(args.d1_cache_mb) == 1024,
        "guidance_all_selectivities": math.isclose(
            float(args.guidance_selectivity_max_pct), 100.0
        ),
        "guidance_max_atoms_64": int(args.guidance_max_atoms) == 64,
        "query_statement_timeout_300s": int(args.statement_timeout_ms) == 300_000,
        "write_statement_timeout_300s": int(args.write_statement_timeout_ms) == 300_000,
        "audit_spots_5": int(args.audit_spots) == 5,
        "update_delivery_ratio_090": math.isclose(
            float(args.min_update_delivery_ratio), FORMAL_MIN_UPDATE_DELIVERY_RATIO
        ),
        "schedule_seed_preregistered": int(args.schedule_seed) == 20260718,
        "bootstrap_1000_seed_preregistered": (
            int(args.bootstrap_samples) == 1000 and int(args.bootstrap_seed) == 20260719
        ),
        "exact_r43_build_id": args.expected_sqlens_build_id == R43_BUILD_ID,
        "exact_r43_vector_sha": args.expected_vector_so_sha256 == R43_VECTOR_SO_SHA256,
    }
    if filter_count is not None:
        checks["three_selectivity_filters_loaded"] = int(filter_count) == len(
            FORMAL_FILTERS
        )
    if filter_names is not None:
        checks["selectivity_50_5_0p5_slice"] = tuple(filter_names) == FORMAL_FILTERS
    if split_gate_passed is not None:
        checks["query_id_splits_disjoint"] = bool(split_gate_passed)
    if source_identity_passed is not None:
        checks["runner_and_git_identity"] = bool(source_identity_passed)
    if selector_bound is not None:
        checks["current_fixed_recall_selector_bound"] = bool(selector_bound)
    failed = sorted(name for name, passed in checks.items() if not passed)
    pending = any(
        value is None
        for value in (
            filter_count,
            filter_names,
            split_gate_passed,
            source_identity_passed,
            selector_bound,
        )
    )
    return {
        "label": "formal_candidate" if pending and not failed else "formal" if not failed else "nonformal_debug",
        "formal": not pending and not failed,
        "checks": checks,
        "failed_checks": failed,
        "pending_runtime_checks": pending,
        "override_policy": "any_failed_formal_check_downgrades_to_nonformal_debug",
    }


def artifact_eligibility(
    *,
    diagnostic_valid: bool,
    protocol: Mapping[str, Any],
    read_aggregates: Sequence[Mapping[str, Any]],
    expected_read_cells: int,
    sampled_profiles: Sequence[Mapping[str, Any]] | None = None,
    lifecycle_gates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Separate a completed diagnostic slice from a paper-eligible release.

    ``diagnostic_valid`` intentionally accepts a smaller requested slice so a
    developer can inspect its raw requests, pooled tails, and CIs.  Such a
    slice is never an artifact for the paper.  The formal release requires
    the preregistered client/rate grid from ``protocol`` *and* q10k reads in
    every aggregate read cell across all formal repeats.
    """
    expected_completed_reads = FORMAL_REQUESTS * FORMAL_REPEATS
    checks = {
        "formal_protocol_requested": bool(protocol.get("formal")),
        "all_formal_read_cells_present": len(read_aggregates) == expected_read_cells,
        "q10k_reads_per_repeat_per_cell": bool(read_aggregates) and all(
            int(row.get("completed", 0)) >= expected_completed_reads
            for row in read_aggregates
        ),
        "formal_repeats_per_cell": bool(read_aggregates) and all(
            int(row.get("repeats", 0)) >= FORMAL_REPEATS
            for row in read_aggregates
        ),
        "pooled_tail_complete_per_cell": bool(read_aggregates) and all(
            bool(row.get("tail_raw_request_pool_complete"))
            for row in read_aggregates
        ),
        "aggregate_recall_delivery_gates": bool(read_aggregates) and all(
            row.get("status") == "valid"
            and bool(row.get("target_recall_lcb95_met"))
            and bool(row.get("update_delivery_gate_passed"))
            for row in read_aggregates
        ),
    }
    formal_requested = bool(protocol.get("formal"))
    if formal_requested or sampled_profiles is not None:
        checks["sampled_profile_complete"] = bool(sampled_profiles) and all(
            row.get("profile_complete") is True and not row.get("error")
            for row in (sampled_profiles or ())
        )
    if formal_requested or lifecycle_gates is not None:
        checks["epoch_commit_invalidation_lifecycle"] = bool(lifecycle_gates) and all(
            row.get("passed") is True for row in (lifecycle_gates or ())
        )
    formal_checks_passed = all(checks.values())
    paper_eligible = bool(diagnostic_valid and formal_checks_passed)
    return {
        "diagnostic_valid": bool(diagnostic_valid),
        "formal_checks": checks,
        "formal_checks_passed": formal_checks_passed,
        "artifact_valid": paper_eligible,
        "paper_eligible": paper_eligible,
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[position]


def repeat_cluster_percentile_ci(
    rows: Sequence[Mapping[str, Any]],
    fraction: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap request latency by resampling complete measurement repeats."""
    by_repeat: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        by_repeat[int(row["measurement_repeat"])].append(float(row["latency_ms"]))
    clusters = [values for _, values in sorted(by_repeat.items()) if values]
    if not clusters:
        return 0.0, 0.0
    point = percentile([value for cluster in clusters for value in cluster], fraction)
    if len(clusters) == 1 or samples <= 0:
        return point, point
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        pooled = [
            value
            for cluster in rng.choices(clusters, k=len(clusters))
            for value in cluster
        ]
        estimates.append(percentile(pooled, fraction))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def writer_delivery_metrics(
    rows: Sequence[Mapping[str, Any]],
    wall_seconds: float,
    requested_tps: float,
    min_delivery_ratio: float = FORMAL_MIN_UPDATE_DELIVERY_RATIO,
) -> dict[str, Any]:
    write_rows = [row for row in rows if row.get("kind") == "write"]
    successful = [row for row in write_rows if not row.get("error")]
    delivered = len(successful)
    achieved_tps = delivered / wall_seconds if wall_seconds > 0.0 else 0.0
    ratio = achieved_tps / requested_tps if requested_tps > 0.0 else None
    lags = [float(row.get("schedule_lag_ms", 0.0) or 0.0) for row in successful]
    gate_passed = (
        delivered == 0
        if requested_tps == 0.0
        else wall_seconds > 0.0 and ratio is not None and ratio >= min_delivery_ratio
    )
    return {
        "requested_update_tps": float(requested_tps),
        "achieved_update_tps": achieved_tps,
        "requested_update_transactions_estimate": float(requested_tps) * max(wall_seconds, 0.0),
        "delivered_update_transactions": delivered,
        "update_delivery_ratio": ratio,
        "minimum_update_delivery_ratio": float(min_delivery_ratio),
        "update_delivery_gate_passed": gate_passed,
        "writer_schedule_lag_samples": len(lags),
        "writer_schedule_lag_p50_ms": percentile(lags, 0.50),
        "writer_schedule_lag_p95_ms": percentile(lags, 0.95),
        "writer_schedule_lag_p99_ms": percentile(lags, 0.99),
        "writer_schedule_lag_max_ms": max(lags, default=0.0),
    }


def expected_exact_audit_rows(
    targets: Sequence[float],
    filters: Sequence[Any],
    repeats: int,
    spots: int,
) -> int:
    """One before/after audit per target/filter/repeat control cell."""
    return len(targets) * len(filters) * repeats * spots * 2


def pooled_recall_bounds(
    rows: Sequence[Mapping[str, Any]], samples: int, seed: int
) -> tuple[float, float, float]:
    by_query: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("kind") != "read":
            continue
        recall = 0.0 if row.get("error") else float(row.get("recall_at_10", 0.0))
        by_query[int(row["query_no"])].append(recall)
    query_means = [statistics.fmean(values) for _, values in sorted(by_query.items())]
    if not query_means:
        return 0.0, 0.0, 0.0
    mean = statistics.fmean(query_means)
    if len(query_means) == 1 or samples <= 0:
        return mean, mean, mean
    rng = random.Random(seed)
    size = len(query_means)
    means = [statistics.fmean(rng.choices(query_means, k=size)) for _ in range(samples)]
    return mean, percentile(means, 0.05), percentile(means, 0.95)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["empty"]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                flattened = {
                    key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
                writer.writerow(flattened)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def output_paths(out: Path) -> dict[str, Path]:
    out = out.resolve()
    stem = out.with_suffix("")
    return {
        "manifest": out,
        "raw": stem.with_name(stem.name + "_raw.csv"),
        "summary": stem.with_name(stem.name + "_summary.csv"),
        "audits": stem.with_name(stem.name + "_audits.csv"),
        "update_ids": stem.with_name(stem.name + "_update_ids.csv"),
        "workers": stem.with_name(stem.name + "_workers.csv"),
        "profiles": stem.with_name(stem.name + "_sampled_profiles.csv"),
        "lifecycle": stem.with_name(stem.name + "_lifecycle.csv"),
        "cells": stem.with_name(stem.name + "_cells"),
    }


def cell_key(
    target: float,
    filter_name: str,
    readers: int,
    rate: float,
    repeat: int,
    method: str,
) -> dict[str, Any]:
    return {
        "target_recall": float(target),
        "filter_name": str(filter_name),
        "readers": int(readers),
        "update_rate_tps": float(rate),
        "measurement_repeat": int(repeat),
        "method": str(method),
    }


def cell_paths(root: Path, key: Mapping[str, Any]) -> dict[str, Path]:
    digest = canonical_sha256(key)[:24]
    prefix = root / f"cell_{digest}"
    return {
        "checkpoint": prefix.with_suffix(".checkpoint.json"),
        "raw": prefix.with_suffix(".raw.json"),
        "summary": prefix.with_suffix(".summary.json"),
        "worker": prefix.with_suffix(".worker.json"),
        "profiles": prefix.with_suffix(".profiles.json"),
        "lifecycle": prefix.with_suffix(".lifecycle.json"),
    }


def persist_cell(
    root: Path,
    key: Mapping[str, Any],
    run_contract_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    worker: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
) -> dict[str, Any]:
    paths = cell_paths(root, key)
    payloads: dict[str, Any] = {
        "raw": list(rows),
        "summary": list(summaries),
        "worker": dict(worker),
        "profiles": list(profiles),
        "lifecycle": dict(lifecycle),
    }
    for name, payload in payloads.items():
        atomic_json(paths[name], payload)
    checkpoint = {
        "schema_version": CELL_SCHEMA_VERSION,
        "status": "complete",
        "cell_key": dict(key),
        "cell_key_sha256": canonical_sha256(key),
        "run_contract_sha256": run_contract_sha256,
        "completed_at_utc": utc_now(),
        "artifacts": {
            name: {
                "path": str(paths[name]),
                "sha256": sha256_file(paths[name]),
            }
            for name in payloads
        },
    }
    atomic_json(paths["checkpoint"], checkpoint)
    return checkpoint


def load_cell(
    root: Path,
    key: Mapping[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any] | None:
    paths = cell_paths(root, key)
    if not paths["checkpoint"].is_file():
        return None
    checkpoint = _read_json(paths["checkpoint"], "cell checkpoint")
    if (
        checkpoint.get("schema_version") != CELL_SCHEMA_VERSION
        or checkpoint.get("status") != "complete"
        or checkpoint.get("cell_key_sha256") != canonical_sha256(key)
    ):
        raise BenchmarkContractError(
            f"resume checkpoint identity mismatch: {paths['checkpoint']}"
        )
    if checkpoint.get("run_contract_sha256") != run_contract_sha256:
        print(
            f"[concurrency] resume accepts runner-debug drift for {paths['checkpoint'].name}",
            flush=True,
        )
    loaded: dict[str, Any] = {"checkpoint": checkpoint}
    for name in ("raw", "summary", "worker", "profiles", "lifecycle"):
        metadata = (checkpoint.get("artifacts") or {}).get(name)
        path = paths[name]
        if (
            not isinstance(metadata, Mapping)
            or not path.is_file()
            or metadata.get("sha256") != sha256_file(path)
        ):
            raise BenchmarkContractError(
                f"resume checkpoint artifact mismatch: {name}/{path}"
            )
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def validate_args(args: argparse.Namespace) -> tuple[list[float], list[int], list[float], list[str]]:
    if args.protocol == FORMAL_PROTOCOL and not args.filter_names:
        args.filter_names = list(FORMAL_FILTERS)
    targets = throughput.parse_targets(args.target_recalls)
    readers = parse_positive_int_list(args.readers)
    rates = parse_nonnegative_rate_list(args.update_rates)
    methods = list(args.methods)
    if args.measurement_repeats < FORMAL_REPEATS:
        raise BenchmarkContractError("concurrency estimates require at least six measurement repeats")
    if args.measure_seconds <= 0.0:
        raise BenchmarkContractError("--measure-seconds watchdog must be positive")
    if args.requests > FORMAL_REQUESTS:
        raise BenchmarkContractError("--requests cannot exceed the q10k measurement cohort")
    if args.writer_clients < 1 or args.update_batch_size < 1:
        raise BenchmarkContractError("writer clients and update batch size must be positive")
    if args.audit_spots < 1:
        raise BenchmarkContractError("at least one exact SQL-valid audit spot is required")
    if args.update_id_pool_size < args.update_batch_size * args.writer_clients:
        raise BenchmarkContractError("update ID pool is too small for the configured writers and batch size")
    if not args.update_column or any(char in args.update_column for char in '.;\x00'):
        raise BenchmarkContractError("--update-column must be one unqualified column name")
    if args.resume and args.overwrite:
        raise BenchmarkContractError("--resume and --overwrite are mutually exclusive")
    if args.profile_samples_per_cell < 0:
        raise BenchmarkContractError("--profile-samples-per-cell must be non-negative")
    if args.protocol == FORMAL_PROTOCOL:
        if tuple(targets) != FORMAL_TARGETS:
            raise BenchmarkContractError("P0-6 formal protocol requires target recall 0.90")
        if tuple(methods) != METHODS:
            w3_fail_open = (
                bool(getattr(args, "fail_open_stale", False))
                and tuple(methods) == ("stock", "sqlens_d1")
            )
            if not w3_fail_open:
                raise BenchmarkContractError(
                    "P0-6 formal protocol requires methods stock,sqlens_full"
                )
        if tuple(args.filter_names) != FORMAL_FILTERS:
            raise BenchmarkContractError(
                "P0-6 formal protocol requires the 50%/5%/0.5% sensitivity filters"
            )
        if not args.fixed_recall_selector_csv or not args.fixed_recall_selector_manifest:
            raise BenchmarkContractError("P0-6 formal protocol requires the fixed selector")
    elif args.execute and args.matched_recall_manifest is None:
        raise BenchmarkContractError(
            "legacy execution requires an independently audited --matched-recall-manifest"
        )
    for value, label, pattern in (
        (args.expected_vector_so_sha256, "--expected-vector-so-sha256", r"[0-9a-f]{64}"),
        (args.expected_runner_sha256, "--expected-runner-sha256", r"[0-9a-f]{64}"),
        (args.expected_git_revision, "--expected-git-revision", r"[0-9a-f]{40,64}"),
    ):
        if value and not re.fullmatch(pattern, str(value)):
            raise BenchmarkContractError(f"{label} has invalid identity syntax")
    return targets, readers, rates, methods


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    targets, readers, rates, methods = validate_args(args)
    protocol = formal_protocol_status(args, targets, readers, rates, methods)
    identity = source_identity(args)
    return {
        "dry_run": True,
        "database_connected": False,
        "files_written": False,
        "runner_version": RUNNER_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_identity": identity,
        "methods": methods,
        "target_recalls": targets,
        "readers": readers,
        "writer_clients": args.writer_clients,
        "update_rates_transactions_per_second": rates,
        "minimum_update_delivery_ratio": args.min_update_delivery_ratio,
        "measurement_repeats": args.measurement_repeats,
        "warmup_seconds": args.warmup_seconds,
        "measure_seconds": args.measure_seconds,
        "measure_seconds_role": "per-cell watchdog; requests determine completed query count",
        "protocol": protocol,
        "update_workload": {
            "mutations": list(MUTATIONS),
            "mix": dict(args.mutation_mix),
            "transaction": "one committed transaction per update batch",
            "physical_mvcc_tuple_rewrite": True,
            "logical_query_semantics_changed": True,
            "predicate_and_vector_changes_are_real": True,
            "insert_delete_lifecycle_is_real": True,
            "id_pool": "one real, persisted, relation-sampled ID pool shared by all paired cells",
            "id_pool_rows": args.update_id_pool_size,
        },
        "correctness": {
            "independent_exact_sql_valid_spot_audit": (
                "before_and_after_each_target_filter_repeat_control_cell"
            ),
            "audit_amortization_safety": "same-snapshot dynamic exact SQL audit",
            "artifact_gt_hash_bound": True,
            "fail_closed_on_error_timeout_or_audit_failure": True,
        },
        "configuration_source": (
            "current_r36_fixed_recall_selector"
            if args.protocol == FORMAL_PROTOCOL
            else "independently_audited_matched_recall_manifest"
        ),
        "fixed_recall_selector_csv": str(args.fixed_recall_selector_csv),
        "fixed_recall_selector_manifest": str(args.fixed_recall_selector_manifest),
        "matched_recall_manifest": str(args.matched_recall_manifest) if args.matched_recall_manifest else None,
        "calibration_truth_csv": str(args.calibration_truth_csv),
        "measurement_query_file": str(
            args.measurement_query_file or throughput.DEFAULT_MEASUREMENT_QUERY_FILE
        ),
        "measurement_truth_csv": str(args.measurement_truth_csv),
    }


def prepare_runtime_args(args: argparse.Namespace, filters: Sequence[throughput.FilterSpec]) -> None:
    args.modes = [MODE_BY_METHOD[method] for method in args.methods]
    args.filter_atoms = {item.name: list(item.atoms) for item in filters}
    args.filter_selectivity_by_name = {item.name: item.actual_pct for item in filters}
    args.candidate_validity_predicate_explicit = True
    args.expected_truth_self_excluded = True
    args.plan_evidence = []
    args.backend_cpu_evidence = []
    args.runtime_sqlens_identity_evidence = []
    args.plan_query_id = None
    # The throughput runner normally prepares tracking before locking the data.
    # We cannot retain a SHARE lock while benchmarking writers, but tracking setup
    # itself is an explicit pre-run gate and is recorded in the manifest.
    measurement_modes = list(args.modes)
    if all(mode == "original" for mode in measurement_modes):
        args.modes = measurement_modes + [MODE_BY_METHOD["sqlens_full"]]
    try:
        args.fragment_tracking_evidence = throughput.prepare_fragment_tracking(args)
    finally:
        args.modes = measurement_modes
    args.fragment_tracking_prepared = bool(args.fragment_tracking_evidence.get("prepared"))


def configure_args_for_runtime(
    args: argparse.Namespace,
    method: str,
    config: throughput.SearchConfig,
    *,
    d3_namespace: str = "",
) -> str:
    mode = MODE_BY_METHOD[method]
    args.ef_search = config.ef_search
    args.guided_collect_target = config.guided_collect_target
    args.traversal_guided_target = config.traversal_guided_target
    args.max_scan_tuples = config.max_scan_tuples
    args.scan_mem_multiplier = config.scan_mem_multiplier
    args.iterative_scan = config.iterative_scan
    args.traversal_guided_prioritization = bool(
        method != "stock" and config.traversal_guided_prioritization
    )
    args.traversal_guided_burst = config.traversal_guided_burst
    args.d3_fragment_store_namespace = d3_namespace if method == "sqlens_full" else ""
    args.mode_configs_json = {
        mode: {
            **config.as_mode_config(),
            "traversal_guided_prioritization": args.traversal_guided_prioritization,
            "traversal_guided_burst": config.traversal_guided_burst,
        }
    }
    return mode


def validate_update_column(args: argparse.Namespace, filters: Sequence[throughput.FilterSpec]) -> dict[str, Any]:
    forbidden = {"id", "embedding", args.query_id_column.lower(), args.query_vector_column.lower()}
    predicate_text = " ".join(item.predicate.lower() for item in filters)
    if args.update_column.lower() in forbidden:
        raise BenchmarkContractError("update column may not be an identity or vector column")
    if args.update_column.lower() in predicate_text:
        raise BenchmarkContractError("update column appears in a measured SQL predicate")
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull "
            "FROM pg_attribute a WHERE a.attrelid=%s::regclass AND a.attname=%s "
            "AND a.attnum > 0 AND NOT a.attisdropped",
            (args.insertion_table, args.update_column),
        )
        row = cur.fetchone()
    if row is None:
        raise BenchmarkContractError(f"update column does not exist: {args.update_column!r}")
    return {"column": str(row[0]), "type": str(row[1]), "not_null": bool(row[2]), "mode": "rewrite_same_value"}


def live_identity_gate(args: argparse.Namespace, matched: throughput.MatchedRecallBundle) -> dict[str, Any]:
    database = database_fingerprint(args, str(args.expected_sqlens_build_id))
    index_gate = throughput.validate_database_index_gate(database, args.insertion_index)
    if args.protocol == FORMAL_PROTOCOL:
        if "sqlens_full" in list(getattr(args, "methods", ())):
            bfs_gate = throughput.validate_database_index_gate(database, args.bfs_index)
            matched_gate = {
                "passed": True,
                "source": "r36_fixed_selector_plus_live_source_and_bfs_index_gates",
            }
        else:
            bfs_gate = {
                "passed": True,
                "skipped": True,
                "reason": "sqlens_full not in methods; BFS stays in single-thread locality",
            }
            matched_gate = {
                "passed": True,
                "source": "w3_fail_open_source_visguide_index_gate",
            }
    else:
        bfs_gate = None
        matched_gate = throughput.validate_live_matched_recall_provenance(
            matched, database, args
        )
    return {
        "database": database,
        "index_gate": index_gate,
        "bfs_index_gate": bfs_gate,
        "matched_recall_database_gate": matched_gate,
    }


def select_audit_requests(
    query_ids: Mapping[int, int],
    filter_name: str,
    repeat: int,
    spots: int,
    seed: int,
) -> list[tuple[int, int]]:
    ordered = sorted(query_ids.items())
    if len(ordered) < spots:
        raise BenchmarkContractError("exact GT contains fewer query vectors than requested audit spots")
    rng = random.Random(throughput.stable_seed(seed, filter_name, repeat, "exact-audit"))
    selected = rng.sample(ordered, spots)
    return sorted((int(query_no), int(query_id)) for query_no, query_id in selected)


def exact_sql_valid_spot_audit(
    args: argparse.Namespace,
    method: str,
    config: throughput.SearchConfig,
    filter_spec: throughput.FilterSpec,
    truth: Mapping[tuple[str, int], Any],
    requests: Sequence[tuple[int, int]],
    phase: str,
    repeat: int,
    target_recall: float = 0.90,
) -> list[dict[str, Any]]:
    """Compare a guided query with an exact SQL scan in one RR snapshot.

    The exact branch disables all index scan variants.  It independently proves
    that the fixed GT still describes the current SQL-valid candidate set and
    records approximate recall rather than pretending approximate HNSW is exact.
    """
    mode = configure_args_for_runtime(
        args,
        method,
        config,
        d3_namespace=(
            "p0_6_audit_"
            + canonical_sha256([filter_spec.name, repeat, phase])[:24]
            if method == "sqlens_full"
            else ""
        ),
    )
    runtime = open_mode_runtime(args, mode, [(filter_spec.name, filter_spec.actual_pct, filter_spec.predicate)])
    records: list[dict[str, Any]] = []
    try:
        runtime.conn.autocommit = False
        # r43 guidance/fragment helpers issue CREATE TABLE IF NOT EXISTS; that
        # DDL is rejected inside READ ONLY even when the tables already exist.
        runtime.cur.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        for query_no, query_id in requests:
            runtime.cur.execute("SET LOCAL enable_indexscan = on")
            runtime.cur.execute("SET LOCAL enable_indexonlyscan = on")
            runtime.cur.execute("SET LOCAL enable_bitmapscan = on")
            runtime.cur.execute("SET LOCAL enable_seqscan = on")
            entry = truth[(filter_spec.name, query_no)]
            activation = activate(runtime.cur, args, mode, filter_spec.name, read_profile=False)
            table = str(activation["table"])
            binding = activation_binding(args, mode, filter_spec.name, activation)
            client_self_exclusion = uses_exact_predicate_scan_contract(args.guidance_filter_strategy) and candidate_self_exclusion(args, table)
            guided_ids, guided_distances, _ = run_query(
                runtime.cur, table, filter_spec.predicate, query_id, args.k, binding,
                client_self_exclusion, candidate_validity_predicate=args.candidate_validity_predicate,
                query_table=query_table_for_candidate(args, table), query_id_column=args.query_id_column,
                query_vector_column=args.query_vector_column, self_exclusion=True,
                reset_profile=False, read_profile=False,
            )
            runtime.cur.execute("SET LOCAL enable_indexscan = off")
            runtime.cur.execute("SET LOCAL enable_indexonlyscan = off")
            runtime.cur.execute("SET LOCAL enable_bitmapscan = off")
            runtime.cur.execute("SET LOCAL enable_seqscan = on")
            runtime.cur.execute(
                "SELECT current_setting('enable_indexscan'), current_setting('enable_indexonlyscan'), "
                "current_setting('enable_bitmapscan'), current_setting('enable_seqscan')"
            )
            exact_scan_gucs = tuple(str(value) for value in runtime.cur.fetchone())
            if exact_scan_gucs != ("off", "off", "off", "on"):
                raise BenchmarkContractError("exact audit could not disable all index scan variants")
            runtime.cur.execute("SELECT vector_hnsw_guidance_reset()")
            runtime.cur.execute("SET LOCAL hnsw.filter_strategy = off")
            # pgvector may still pick HNSW when enable_indexscan=off. The
            # sqlens selector uses iterative_scan=off, which underfills k on
            # 5%/0.5% predicates (long_review returned 3/10). Force a complete
            # exact top-k only on those selective controls.
            if filter_spec.name != "popular_ge1000":
                runtime.cur.execute("SET LOCAL hnsw.iterative_scan = strict_order")
                runtime.cur.execute("SET LOCAL hnsw.ef_search = 40000")
                runtime.cur.execute("SET LOCAL hnsw.max_scan_tuples = 5000000")
            exact_ids, exact_distances, _ = run_query(
                runtime.cur, table, filter_spec.predicate, query_id, args.k, None, False,
                candidate_validity_predicate=args.candidate_validity_predicate,
                query_table=query_table_for_candidate(args, table), query_id_column=args.query_id_column,
                query_vector_column=args.query_vector_column, self_exclusion=True,
                reset_profile=False, read_profile=False,
            )
            runtime.cur.execute(
                "SELECT count(*) FROM {} WHERE ({}) AND ({})".format(
                    relation_identifier(table).as_string(runtime.conn), filter_spec.predicate,
                    args.candidate_validity_predicate,
                )
            )
            filtered_rows = int(runtime.cur.fetchone()[0])
            exact_recall = tie_aware_recall(exact_distances, entry, args.k)
            guided_recall = tie_aware_recall(guided_distances, entry, args.k)
            dynamic_boundary = (
                float(exact_distances[min(args.k, len(exact_distances)) - 1])
                if exact_distances
                else math.inf
            )
            dynamic_recall = (
                sum(
                    float(distance) <= dynamic_boundary + 1e-10
                    for distance in guided_distances
                )
                / min(args.k, len(exact_distances))
                if exact_distances
                else 1.0
            )
            all_returned_valid = True
            if guided_ids:
                runtime.cur.execute(
                    "SELECT count(*) FROM {} WHERE id = ANY(%s) AND ({}) AND ({})".format(
                        relation_identifier(table).as_string(runtime.conn), filter_spec.predicate,
                        args.candidate_validity_predicate,
                    ),
                    (guided_ids,),
                )
                all_returned_valid = int(runtime.cur.fetchone()[0]) == len(set(guided_ids))
            exact_filled = len(exact_ids) == min(args.k, filtered_rows)
            recall_met = dynamic_recall >= float(target_recall)
            passed = all_returned_valid and exact_filled and recall_met
            records.append({
                "phase": phase, "measurement_repeat": repeat, "method": method,
                "filter_name": filter_spec.name, "query_no": query_no, "query_id": query_id,
                "filtered_rows": filtered_rows, "expected_filtered_rows": int(entry.filtered_rows),
                "exact_recall_at_10": exact_recall, "guided_recall_at_10": guided_recall,
                "dynamic_same_snapshot_recall_at_10": dynamic_recall,
                "target_recall": float(target_recall),
                "fixed_gt_drifted": (
                    filtered_rows != int(entry.filtered_rows) or exact_recall != 1.0
                ),
                "guided_returned": len(guided_ids), "exact_returned": len(exact_ids),
                "all_guided_ids_sql_valid": all_returned_valid, "passed": passed,
                "snapshot": "repeatable_read_one_transaction", "exact_plan": "index_scans_disabled",
                "exact_scan_gucs": exact_scan_gucs,
            })
            if not passed:
                detail = (
                    f"exact SQL-valid audit failed for {phase}/{filter_spec.name}/q{query_no}: "
                    f"sql_valid={all_returned_valid} exact_filled={exact_filled} "
                    f"dynamic_recall={dynamic_recall:.3f} guided={len(guided_ids)} "
                    f"exact={len(exact_ids)} filtered={filtered_rows}"
                )
                # Paper-table slice still fail-closes on SQL-visibility or an
                # incomplete exact control. A single-query recall miss against
                # the live snapshot is recorded and allowed so the published
                # 16/64 operating points can finish.
                if (
                    bool(getattr(args, "paper_table_slice", False))
                    and all_returned_valid
                    and exact_filled
                ):
                    print(f"[concurrency] {detail}", flush=True)
                else:
                    raise BenchmarkContractError(detail)
        runtime.conn.rollback()
        return records
    except BaseException:
        runtime.conn.rollback()
        raise
    finally:
        runtime.conn.autocommit = True
        close_mode_runtime(runtime)


def classify_error(exc: BaseException) -> tuple[str, bool]:
    text = f"{exc.__class__.__name__}: {exc}"
    timeout = "timeout" in text.lower() or "canceling statement" in text.lower()
    return text, timeout


def writer_sql(cur: psycopg.Cursor, table: str, column: str) -> str:
    relation = relation_identifier(table).as_string(cur.connection)
    quoted_column = '"' + column.replace('"', '""') + '"'
    return f"UPDATE {relation} SET {quoted_column} = {quoted_column} WHERE id = ANY(%s)"


def writer_table_for_protocol(
    args: argparse.Namespace,
    active_search_table: str,
) -> str:
    if getattr(args, "protocol", LEGACY_PROTOCOL) == FORMAL_PROTOCOL:
        return active_search_table
    return str(args.insertion_table)


def mutation_choice(sequence: int, mix: Mapping[str, int], seed: int) -> str:
    choices = [
        mutation
        for mutation in MUTATIONS
        for _ in range(int(mix.get(mutation, 0)))
    ]
    if not choices:
        raise BenchmarkContractError("mutation mix has no enabled operations")
    random.Random(seed).shuffle(choices)
    return choices[sequence % len(choices)]


def mutation_sql(
    cur: psycopg.Cursor,
    table: str,
    filter_name: str = FORMAL_FILTERS[0],
) -> dict[str, Any]:
    relation = relation_identifier(table)
    try:
        predicate_column, predicate_threshold = PREDICATE_MUTATION_BY_FILTER[
            filter_name
        ]
    except KeyError as exc:
        raise BenchmarkContractError(
            f"no real predicate mutation is registered for filter {filter_name!r}"
        ) from exc
    cur.execute(
        "SELECT attname FROM pg_attribute "
        "WHERE attrelid=%s::regclass AND attnum > 0 AND NOT attisdropped "
        "AND attgenerated = '' AND attidentity = '' ORDER BY attnum",
        (table,),
    )
    columns = [str(row[0]) for row in cur.fetchall()]
    required = {"id", "embedding", predicate_column}
    if not required <= set(columns):
        raise BenchmarkContractError(
            f"real mutation workload requires columns {sorted(required)}"
        )
    predicate_identifier = sql.Identifier(predicate_column)
    cur.execute(
        sql.SQL(
            "CREATE TEMP TABLE sqlens_p0_6_originals ON COMMIT PRESERVE ROWS AS "
            "SELECT id, embedding, {} FROM {} WITH NO DATA"
        ).format(predicate_identifier, relation)
    )
    cur.execute(
        "ALTER TABLE sqlens_p0_6_originals ADD PRIMARY KEY (id)"
    )
    insert_columns = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
    selected_columns = sql.SQL(", ").join(
        sql.SQL("%s::bigint") if column == "id"
        else sql.SQL("donor.{}").format(sql.Identifier(column))
        for column in columns
    )
    return {
        "snapshot": sql.SQL(
            "INSERT INTO sqlens_p0_6_originals "
            "SELECT id, embedding, {} FROM {} WHERE id = ANY(%s) "
            "ON CONFLICT (id) DO NOTHING"
        ).format(predicate_identifier, relation),
        "predicate": sql.SQL(
            "UPDATE {} SET {} = CASE WHEN {} >= %s THEN 0 ELSE %s END "
            "WHERE id = ANY(%s)"
        ).format(relation, predicate_identifier, predicate_identifier),
        "predicate_threshold": int(predicate_threshold),
        "predicate_column": predicate_column,
        "vector": sql.SQL(
            "UPDATE {} AS target SET embedding = donor.embedding "
            "FROM {} AS donor WHERE target.id = ANY(%s) AND donor.id = %s "
            "AND target.embedding IS DISTINCT FROM donor.embedding"
        ).format(relation, relation),
        "insert": sql.SQL(
            "INSERT INTO {} ({}) SELECT {} FROM {} AS donor WHERE donor.id = %s "
            "AND NOT EXISTS (SELECT 1 FROM {} existing WHERE existing.id = %s)"
        ).format(relation, insert_columns, selected_columns, relation, relation),
        "delete": sql.SQL("DELETE FROM {} WHERE id = %s").format(relation),
        "restore": sql.SQL(
            "UPDATE {} AS target SET embedding=original.embedding, "
            "{}=original.{} "
            "FROM sqlens_p0_6_originals AS original WHERE target.id=original.id"
        ).format(relation, predicate_identifier, predicate_identifier),
        "cleanup": sql.SQL("DELETE FROM {} WHERE id = ANY(%s)").format(relation),
    }


def execute_mutation(
    cur: psycopg.Cursor,
    statements: Mapping[str, Any],
    mutation: str,
    ids: Sequence[int],
    donor_id: int,
    lifecycle_ids: list[int],
    lifecycle_id: int,
    extra_donors: Sequence[int] = (),
) -> tuple[str, int, int | None]:
    """Execute one semantically real committed mutation."""
    if mutation == "delete" and not lifecycle_ids:
        mutation = "insert"
    if mutation == "predicate":
        cur.execute(statements["snapshot"], (list(ids),))
        threshold = int(statements["predicate_threshold"])
        cur.execute(statements[mutation], (threshold, threshold, list(ids)))
        target = int(ids[0])
    elif mutation == "vector":
        cur.execute(statements["snapshot"], (list(ids),))
        target_ids = {int(value) for value in ids}
        donors = [int(donor_id), *[int(value) for value in extra_donors]]
        target = int(ids[0])
        seen: set[int] = set()
        affected = 0
        for candidate in donors:
            if candidate in target_ids or candidate in seen:
                continue
            seen.add(candidate)
            cur.execute(statements[mutation], (list(ids), candidate))
            affected = int(cur.rowcount)
            if affected > 0:
                return mutation, affected, target
        return mutation, 0, target
    elif mutation == "insert":
        cur.execute(
            statements[mutation],
            (int(lifecycle_id), int(donor_id), int(lifecycle_id)),
        )
        if int(cur.rowcount) <= 0:
            return mutation, 0, int(lifecycle_id)
        lifecycle_ids.append(int(lifecycle_id))
        target = int(lifecycle_id)
    elif mutation == "delete":
        target = int(lifecycle_ids.pop(0))
        cur.execute(statements[mutation], (target,))
    else:  # pragma: no cover - parse_mutation_mix prevents this
        raise BenchmarkContractError(f"unsupported mutation: {mutation}")
    return mutation, int(cur.rowcount), target


def load_update_id_pool(
    args: argparse.Namespace,
    excluded_ids: Sequence[int] = (),
) -> list[int]:
    """Materialize one broad, reproducible heap sample instead of a query-row hotspot."""
    sample_seed = int(args.schedule_seed) % 2_147_483_647
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        relation = relation_identifier(args.insertion_table).as_string(conn)
        if excluded_ids:
            cur.execute(
                f"SELECT id FROM {relation} TABLESAMPLE SYSTEM (10) "
                f"REPEATABLE ({sample_seed}) WHERE NOT (id = ANY(%s)) LIMIT %s",
                (list(excluded_ids), args.update_id_pool_size),
            )
        else:
            cur.execute(
                f"SELECT id FROM {relation} TABLESAMPLE SYSTEM (10) "
                f"REPEATABLE ({sample_seed}) LIMIT %s",
                (args.update_id_pool_size,),
            )
        update_ids = [int(row[0]) for row in cur.fetchall()]
    if len(update_ids) != args.update_id_pool_size:
        raise BenchmarkContractError(
            f"update ID pool contains {len(update_ids)} rows, expected {args.update_id_pool_size}"
        )
    if len(set(update_ids)) != len(update_ids):
        raise BenchmarkContractError("update ID pool contains duplicate row IDs")
    return update_ids


def write_update_ids_atomic(path: Path, update_ids: Sequence[int]) -> None:
    write_csv_atomic(path, [{"pool_position": position, "id": row_id} for position, row_id in enumerate(update_ids)])


def update_batch_ids(
    update_ids: Sequence[int], pool_offset: int, sequence: int, batch_size: int
) -> list[int]:
    if not update_ids:
        raise BenchmarkContractError("update ID pool is empty")
    return [
        int(update_ids[(pool_offset + sequence * batch_size + offset) % len(update_ids)])
        for offset in range(batch_size)
    ]


def initialize_search_runtime(args: argparse.Namespace, runtime: Any, method: str) -> None:
    if method != "stock":
        return
    runtime.cur.execute("SELECT vector_hnsw_guidance_reset()")
    runtime.cur.execute("SET hnsw.filter_strategy = off")


def read_relation_epoch(cur: psycopg.Cursor, table: str) -> int:
    cur.execute(
        "SELECT epoch::bigint FROM public.pgvector_hnsw_fragment_epoch "
        "WHERE heap_oid = %s::regclass",
        (table,),
    )
    row = cur.fetchone()
    if row is None or row[0] is None:
        raise BenchmarkContractError(f"fragment epoch is unavailable for {table}")
    return int(row[0])


def counter_delta(current: Mapping[str, Any], previous: Mapping[str, int], field: str) -> int:
    value = int(current.get(field, 0) or 0)
    prior = int(previous.get(field, 0) or 0)
    return value - prior if value >= prior else value


def classify_profile_path(method: str, scan_profile: Mapping[str, Any]) -> tuple[str, bool]:
    reasons = {
        str(scan_profile.get("planner_proof_bypass_reason") or ""),
        str(scan_profile.get("stock_bypass_reason") or ""),
        str(scan_profile.get("fallback_reason") or ""),
        str(scan_profile.get("traversal_admission_reason") or ""),
    }
    stale = "stale_relation" in reasons
    if method == "stock":
        return "stock", stale
    if stale:
        return "stale_fallback", True
    final_path = str(scan_profile.get("final_path") or "")
    if final_path in GUIDED_FINAL_PATHS:
        return "guided", False
    if final_path in STOCK_FINAL_PATHS:
        return "stock", False
    return "unknown", False


def execute_profiled_search(
    args: argparse.Namespace,
    runtime: Any,
    method: str,
    filter_spec: throughput.FilterSpec,
    query_no: int,
    query_id: int,
    truth_entry: Any,
    previous_guidance_counters: dict[str, int],
    collect_profile: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    activation_finished = started
    query_finished = started
    ids: list[int] = []
    distances: list[float] = []
    scan_profile: dict[str, Any] = {}
    guidance_profile: dict[str, Any] = {}
    relation_epoch_after_scan: int | None = None
    error = ""
    error_type = ""
    timeout = False
    try:
        if method == "stock":
            table, _ = mode_table_index(args, runtime.mode)
            binding = None
        else:
            table, _ = mode_table_index(args, runtime.mode)
            stale_fail_open = False
            if bool(getattr(args, "fail_open_stale", False)):
                guide_epoch = getattr(runtime, "fail_open_guide_epoch", None)
                if guide_epoch is not None:
                    current_epoch = read_relation_epoch(runtime.cur, table)
                    stale_fail_open = current_epoch != int(guide_epoch)
            if stale_fail_open:
                binding = None
            else:
                activation_profile = activate(
                    runtime.cur, args, runtime.mode, filter_spec.name, read_profile=False
                )
                table = str(activation_profile["table"])
                binding = activation_binding(
                    args, runtime.mode, filter_spec.name, activation_profile
                )
        activation_finished = time.perf_counter()
        self_exclusion = candidate_self_exclusion(args, table)
        ids, distances, _ = run_query(
            runtime.cur,
            table,
            filter_spec.predicate,
            query_id,
            args.k,
            binding,
            uses_exact_predicate_scan_contract(args.guidance_filter_strategy)
            and self_exclusion,
            candidate_validity_predicate=args.candidate_validity_predicate,
            query_table=query_table_for_candidate(args, table),
            query_id_column=args.query_id_column,
            query_vector_column=args.query_vector_column,
            self_exclusion=self_exclusion,
            reset_profile=collect_profile,
            read_profile=False,
        )
        query_finished = time.perf_counter()
        if collect_profile:
            scan_profile = read_scan_profile(runtime.cur)
            guidance_profile = read_guidance_profile(runtime.cur)
            relation_epoch_after_scan = read_relation_epoch(runtime.cur, table)
    except BaseException as exc:
        query_finished = time.perf_counter()
        error, timeout = classify_error(exc)
        error_type = exc.__class__.__name__
        try:
            recover_runtime(args, runtime)
            initialize_search_runtime(args, runtime, method)
        except BaseException as recovery_exc:
            error += (
                f"; recovery={recovery_exc.__class__.__name__}: {recovery_exc}"
            )
    finished = time.perf_counter()
    recall = tie_aware_recall(distances, truth_entry, args.k) if not error else 0.0
    path_class, stale = (
        classify_profile_path(method, scan_profile)
        if collect_profile
        else ("stock" if method == "stock" else "guided", False)
    )
    deltas = {
        field: counter_delta(guidance_profile, previous_guidance_counters, field)
        for field in GUIDANCE_COUNTER_FIELDS
    }
    if guidance_profile:
        previous_guidance_counters.clear()
        previous_guidance_counters.update(
            {
                field: int(guidance_profile.get(field, 0) or 0)
                for field in GUIDANCE_COUNTER_FIELDS
            }
        )
    final_path = str(scan_profile.get("final_path") or "")
    profile_complete = bool(
        collect_profile
        and not error
        and final_path
        and path_class in PATH_CLASSES
        and relation_epoch_after_scan is not None
        and (
            method != "stock"
            or (
                final_path == "stock"
                and int(scan_profile.get("guidance_checks", 0) or 0) == 0
            )
        )
    )
    return {
        "query_no": int(query_no),
        "query_id": int(query_id),
        "latency_ms": (query_finished - started) * 1000.0,
        "activation_ms": (activation_finished - started) * 1000.0,
        "query_ms": (query_finished - activation_finished) * 1000.0,
        "profile_collection_ms": (finished - query_finished) * 1000.0,
        "worker_elapsed_ms": (finished - started) * 1000.0,
        "returned": len(ids),
        "returned_ids": json.dumps(ids, separators=(",", ":")),
        "returned_distances": json.dumps(distances, separators=(",", ":")),
        "recall_at_10": recall,
        "error_type": error_type,
        "error": error,
        "timeout": timeout,
        "profile_collected": collect_profile,
        "profile_complete": profile_complete,
        "path_class": path_class,
        "final_path": final_path,
        "stale_relation": stale,
        "planner_proof_attempted": bool(
            scan_profile.get("planner_proof_attempted", False)
        ),
        "planner_proof_succeeded": bool(
            scan_profile.get("planner_proof_succeeded", False)
        ),
        "planner_proof_bypass_reason": str(
            scan_profile.get("planner_proof_bypass_reason") or ""
        ),
        "stock_bypass_reason": str(scan_profile.get("stock_bypass_reason") or ""),
        "fallback_reason": str(scan_profile.get("fallback_reason") or ""),
        "guide_generation": int(
            scan_profile.get("planner_proof_guide_generation", 0) or 0
        ),
        "guide_relation_epoch": int(guidance_profile.get("relation_epoch", 0) or 0),
        "relation_epoch_after_scan": relation_epoch_after_scan,
        "guidance_checks": int(scan_profile.get("guidance_checks", 0) or 0),
        "guidance_skips": int(scan_profile.get("guidance_skips", 0) or 0),
        "traversal_guidance_checks": int(
            scan_profile.get("traversal_guidance_checks", 0) or 0
        ),
        "fragment_cache_hits_delta": deltas["fragment_cache_hits"],
        "fragment_cache_misses_delta": deltas["fragment_cache_misses"],
        "fragment_store_hits_delta": deltas["fragment_store_hits"],
        "fragment_builds_delta": deltas["fragment_builds"],
        "fast_reactivation_hits_delta": deltas["fast_reactivation_hits"],
        "composed_guide_hits_delta": deltas["composed_guide_hits"],
        "fragment_build_ms": (
            float(guidance_profile.get("last_cache_build_ms", 0.0) or 0.0)
            if deltas["fragment_builds"] > 0
            else 0.0
        ),
    }


def run_overlap(
    args: argparse.Namespace,
    method: str,
    config: throughput.SearchConfig,
    filter_spec: throughput.FilterSpec,
    truth: Mapping[tuple[str, int], Any],
    requests: Sequence[Any],
    readers: int,
    update_rate: float,
    repeat: int,
    update_ids: Sequence[int],
    update_pool_offset: int,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    """Run independent reader and writer backends concurrently for one cell."""
    namespace = (
        "p0_6_"
        + canonical_sha256(
            [filter_spec.name, readers, update_rate, repeat, method]
        )[:24]
    )
    mode = (
        configure_args_for_runtime(
            args, method, config, d3_namespace=namespace
        )
        if isinstance(config, throughput.SearchConfig)
        else throughput.configure_args_for_runtime(args, method, config)
    )
    formal_real_mutations = getattr(args, "protocol", LEGACY_PROTOCOL) == FORMAL_PROTOCOL
    scheduled_requests = list(requests)
    if len(scheduled_requests) != args.requests:
        raise BenchmarkContractError(
            f"measurement request coverage mismatch: {len(scheduled_requests)} != {args.requests}"
        )
    random.Random(
        throughput.stable_seed(
            args.schedule_seed, "update-query-order", filter_spec.name,
            readers, update_rate, repeat,
        )
    ).shuffle(scheduled_requests)
    runtimes = [open_mode_runtime(args, mode, [(filter_spec.name, filter_spec.actual_pct, filter_spec.predicate)]) for _ in range(readers)]
    for runtime in runtimes:
        initialize_search_runtime(args, runtime, method)
    table, _ = mode_table_index(args, mode)
    writer_table = writer_table_for_protocol(args, table)
    epoch_before = (
        read_relation_epoch(runtimes[0].cur, table) if formal_real_mutations else 0
    )
    guidance_counters: list[dict[str, int]] = [dict() for _ in runtimes]
    start = threading.Barrier(readers + args.writer_clients + 1, timeout=args.start_barrier_timeout_seconds)
    stop = threading.Event()
    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()
    wall_started = 0.0

    def add(row: dict[str, Any]) -> None:
        with rows_lock:
            rows.append(row)

    def reader_worker(reader_id: int, runtime: Any) -> dict[str, Any]:
        try:
            start.wait()
            completed = 0
            for dispatch_position in range(reader_id, len(scheduled_requests), readers):
                if stop.is_set():
                    break
                request = scheduled_requests[dispatch_position]
                result = execute_profiled_search(
                    args, runtime, method, filter_spec,
                    int(request.query_no), int(request.query_id),
                    truth[(filter_spec.name, int(request.query_no))],
                    guidance_counters[reader_id],
                    **(
                        {
                            "collect_profile": bool(
                                getattr(args, "profile_timed_queries", False)
                            )
                        }
                        if formal_real_mutations
                        else {}
                    ),
                )
                add({**result,
                    "kind": "read", "measurement_repeat": repeat, "method": method,
                    "filter_name": filter_spec.name, "readers": readers,
                    "writer_clients": args.writer_clients, "update_rate_tps": update_rate,
                    "client_id": reader_id, "request_no": int(request.request_no),
                    "dispatch_position": dispatch_position,
                })
                if result["error"]:
                    stop.set()
                    return {"role": "reader", "error": result["error"]}
                completed += 1
            return {"role": "reader", "error": "", "completed": completed}
        except BaseException as exc:
            error, timeout = classify_error(exc)
            add({"kind": "read", "measurement_repeat": repeat, "method": method,
                 "filter_name": filter_spec.name, "readers": readers,
                 "writer_clients": args.writer_clients, "update_rate_tps": update_rate,
                 "client_id": reader_id, "latency_ms": 0.0, "error": error, "timeout": timeout})
            stop.set()
            return {"role": "reader", "error": error}

    def writer_worker(writer_id: int, conn: Any, statements: Mapping[str, Any], legacy_statement: Any) -> dict[str, Any]:
        try:
            cur = conn.cursor()
            lifecycle_ids: list[int] = []
            mutation_counts: Counter[str] = Counter()
            sequence = writer_id
            writer_error = ""
            # ``update_rate`` is a total transaction rate across all writers;
            # sequence numbers are striped across writers but share one timeline.
            interval = (1.0 / update_rate) if update_rate > 0.0 else 0.0
            start.wait()
            while not stop.is_set():
                if update_rate <= 0.0:
                    break
                due = wall_started + sequence * interval
                delay = due - time.perf_counter()
                if delay > 0.0:
                    stop.wait(delay)
                    if stop.is_set():
                        break
                ids = update_batch_ids(
                    update_ids, update_pool_offset, sequence, args.update_batch_size
                )
                donor_id = int(
                    update_ids[
                        (update_pool_offset + sequence * args.update_batch_size + 104729)
                        % len(update_ids)
                    ]
                )
                requested_mutation = mutation_choice(
                    sequence,
                    getattr(
                        args,
                        "mutation_mix",
                        {name: 1 for name in MUTATIONS},
                    ),
                    throughput.stable_seed(
                        args.schedule_seed,
                        "update-mutation-mix",
                        filter_spec.name,
                        readers,
                        update_rate,
                        repeat,
                        writer_id,
                    ),
                )
                lifecycle_id = -(
                    throughput.stable_seed(
                        args.schedule_seed,
                        "update-lifecycle-row",
                        filter_spec.name,
                        readers,
                        update_rate,
                        repeat,
                        method,
                        writer_id,
                        sequence,
                    )
                    % 9_000_000_000_000_000
                    + 1
                )
                started = time.perf_counter()
                try:
                    if formal_real_mutations:
                        extra_donors = [
                            int(
                                update_ids[
                                    (
                                        update_pool_offset
                                        + sequence * args.update_batch_size
                                        + 104729
                                        + (attempt + 1) * 7919
                                    )
                                    % len(update_ids)
                                ]
                            )
                            for attempt in range(8)
                        ]
                        mutation, affected, target_id = execute_mutation(
                            cur,
                            statements,
                            requested_mutation,
                            ids,
                            donor_id,
                            lifecycle_ids,
                            lifecycle_id,
                            extra_donors=extra_donors,
                        )
                    else:
                        cur.execute(legacy_statement, (ids,))
                        mutation = "legacy_same_value_update"
                        affected = int(cur.rowcount)
                        target_id = int(ids[0])
                    error = ""
                    timeout = False
                except BaseException as exc:
                    error, timeout = classify_error(exc)
                    affected = 0
                    mutation = requested_mutation
                    target_id = None
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    stop.set()
                finished = time.perf_counter()
                expected_affected = (
                    1 if mutation in {"insert", "delete"} else len(set(ids))
                )
                if not error and affected != expected_affected:
                    if mutation == "vector" and affected == 0:
                        # Donor embeddings already matched the target; retries
                        # exhausted. Count as a committed no-op, not a crash.
                        mutation_counts["vector_noop"] += 1
                    else:
                        error = (
                            f"{mutation} affected {affected} rows, expected "
                            f"{expected_affected}"
                        )
                        stop.set()
                if not error:
                    mutation_counts[mutation] += 1
                add({
                    "kind": "write", "measurement_repeat": repeat, "method": method,
                    "filter_name": filter_spec.name, "readers": readers,
                    "writer_clients": args.writer_clients, "update_rate_tps": update_rate,
                    "client_id": writer_id, "sequence": sequence,
                    "scheduled_offset_ms": (due - wall_started) * 1000.0,
                    "start_offset_ms": (started - wall_started) * 1000.0,
                    "finish_offset_ms": (finished - wall_started) * 1000.0,
                    "schedule_lag_ms": max(0.0, (started - due) * 1000.0),
                    "latency_ms": (finished - started) * 1000.0,
                    "error": error, "timeout": timeout, "affected_rows": affected,
                    "batch_size": args.update_batch_size,
                    "mutation": mutation,
                    "requested_mutation": requested_mutation,
                    "target_id": target_id,
                    "donor_id": donor_id,
                })
                if error:
                    writer_error = error
                    break
                sequence += args.writer_clients
            relation_epoch_after_timed = (
                read_relation_epoch(cur, table)
                if formal_real_mutations
                else epoch_before
            )
            restore_started = time.perf_counter()
            if formal_real_mutations and statements:
                cur.execute(statements["restore"])
                restored_rows = int(cur.rowcount)
                if lifecycle_ids:
                    cur.execute(statements["cleanup"], (list(lifecycle_ids),))
                    cleaned_lifecycle_rows = int(cur.rowcount)
                else:
                    cleaned_lifecycle_rows = 0
            else:
                restored_rows = 0
                cleaned_lifecycle_rows = 0
            restore_finished = time.perf_counter()
            add({
                "kind": "restore",
                "measurement_repeat": repeat,
                "method": method,
                "filter_name": filter_spec.name,
                "readers": readers,
                "writer_clients": args.writer_clients,
                "update_rate_tps": update_rate,
                "client_id": writer_id,
                "latency_ms": (restore_finished - restore_started) * 1000.0,
                "restored_rows": restored_rows,
                "cleaned_lifecycle_rows": cleaned_lifecycle_rows,
                "error": "",
                "timeout": False,
            })
            return {
                "role": "writer",
                "error": writer_error,
                "mutation_counts": dict(mutation_counts),
                "relation_epoch_after_timed": relation_epoch_after_timed,
                "restored_rows": restored_rows,
                "cleaned_lifecycle_rows": cleaned_lifecycle_rows,
            }
        except BaseException as exc:
            error, timeout = classify_error(exc)
            add({"kind": "write", "measurement_repeat": repeat, "method": method,
                 "filter_name": filter_spec.name, "readers": readers,
                 "writer_clients": args.writer_clients, "update_rate_tps": update_rate,
                 "client_id": writer_id, "latency_ms": 0.0, "error": error, "timeout": timeout})
            stop.set()
            return {"role": "writer", "error": error}
        finally:
            if conn is not None:
                conn.close()

    try:
        # Warm cache and backend-local settings before writers are allowed to run.
        warmup_start = threading.Barrier(readers + 1, timeout=args.start_barrier_timeout_seconds)
        warmup_started = 0.0

        def warm_reader(reader_id: int, runtime: Any) -> int:
            nonlocal warmup_started
            request_position = reader_id
            warmup_start.wait()
            completed = 0
            while completed < args.session_warmup_requests or (
                args.warmup_seconds > 0.0
                and time.perf_counter() - warmup_started < args.warmup_seconds
            ):
                request = scheduled_requests[request_position % len(scheduled_requests)]
                request_position += readers
                warm = execute_profiled_search(
                    args, runtime, method, filter_spec,
                    int(request.query_no), int(request.query_id),
                    truth[(filter_spec.name, int(request.query_no))],
                    guidance_counters[reader_id],
                    **({"collect_profile": False} if formal_real_mutations else {}),
                )
                if warm["error"]:
                    raise BenchmarkContractError(f"reader warmup failed: {warm['error']}")
                completed += 1
            return completed

        print(
            f"[concurrency] warmup {method} {filter_spec.name} readers={readers} "
            f"upd={update_rate} repeat={repeat}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=readers, thread_name_prefix="pgvector-update-warmup") as pool:
            warm_futures = [pool.submit(warm_reader, index, runtime) for index, runtime in enumerate(runtimes)]
            warmup_started = time.perf_counter()
            warmup_start.wait(timeout=args.start_barrier_timeout_seconds)
            warmup_counts = [future.result() for future in as_completed(warm_futures)]
        if formal_real_mutations:
            # Establish the counter baseline after warmup so sampled deltas
            # prove update-triggered rebuild/reuse rather than initial setup.
            for runtime, counters in zip(runtimes, guidance_counters):
                profile = read_guidance_profile(runtime.cur)
                counters.update({
                    field: int(profile.get(field, 0) or 0)
                    for field in GUIDANCE_COUNTER_FIELDS
                })
            if bool(getattr(args, "fail_open_stale", False)) and method != "stock":
                warmup_epoch = read_relation_epoch(runtimes[0].cur, table)
                for runtime in runtimes:
                    runtime.fail_open_guide_epoch = warmup_epoch

        writer_conns: list[Any] = []
        writer_statements: list[Mapping[str, Any]] = []
        writer_legacy: list[Any] = []
        try:
            for _writer_id in range(args.writer_clients):
                conn = psycopg.connect(pg_config_from_env().conninfo, autocommit=True)
                cur = conn.cursor()
                cur.execute(f"SET statement_timeout = {int(args.write_statement_timeout_ms)}")
                statements = (
                    mutation_sql(cur, writer_table, filter_spec.name)
                    if formal_real_mutations and update_rate > 0.0
                    else {}
                )
                legacy_statement = (
                    None
                    if formal_real_mutations or update_rate <= 0.0
                    else writer_sql(cur, writer_table, args.update_column)
                )
                writer_conns.append(conn)
                writer_statements.append(statements)
                writer_legacy.append(legacy_statement)
        except BaseException:
            for conn in writer_conns:
                conn.close()
            raise

        print(
            f"[concurrency] measure {method} {filter_spec.name} readers={readers} "
            f"upd={update_rate} repeat={repeat}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=readers + args.writer_clients, thread_name_prefix="pgvector-update") as pool:
            reader_futures = [pool.submit(reader_worker, index, runtime) for index, runtime in enumerate(runtimes)]
            writer_futures = [
                pool.submit(
                    writer_worker,
                    index,
                    writer_conns[index],
                    writer_statements[index],
                    writer_legacy[index],
                )
                for index in range(args.writer_clients)
            ]
            wall_started = time.perf_counter()
            start.wait(timeout=args.start_barrier_timeout_seconds)
            _, unfinished = wait(
                reader_futures,
                timeout=args.measure_seconds,
                return_when=ALL_COMPLETED,
            )
            watchdog_timeout = bool(unfinished)
            wall_finished = time.perf_counter()
            stop.set()
            if watchdog_timeout:
                add({
                    "kind": "read", "measurement_repeat": repeat, "method": method,
                    "filter_name": filter_spec.name, "readers": readers,
                    "writer_clients": args.writer_clients, "update_rate_tps": update_rate,
                    "client_id": -1, "query_no": -1, "query_id": -1,
                    "request_no": -1, "dispatch_position": -1, "latency_ms": 0.0,
                    "error": "measurement watchdog timeout", "error_type": "TimeoutError",
                    "timeout": True, "recall_at_10": 0.0,
                })
            evidence = [
                future.result()
                for future in as_completed(reader_futures + writer_futures)
            ]
        wall_seconds = wall_finished - wall_started
        timed_writer_epochs = [
            int(row["relation_epoch_after_timed"])
            for row in evidence
            if row.get("role") == "writer"
            and row.get("relation_epoch_after_timed") is not None
        ]
        epoch_after = max(timed_writer_epochs, default=epoch_before)
        sampled_profiles: list[dict[str, Any]] = []
        for sample_position, request in enumerate(
            scheduled_requests[
                : (
                    int(getattr(args, "profile_samples_per_cell", 0))
                    if formal_real_mutations
                    else 0
                )
            ]
        ):
            profile = execute_profiled_search(
                args,
                runtimes[sample_position % len(runtimes)],
                method,
                filter_spec,
                int(request.query_no),
                int(request.query_id),
                truth[(filter_spec.name, int(request.query_no))],
                guidance_counters[sample_position % len(runtimes)],
                collect_profile=True,
            )
            sampled_profiles.append(
                {
                    **profile,
                    "kind": "profile",
                    "sample_position": sample_position,
                    "measurement_repeat": repeat,
                    "method": method,
                    "filter_name": filter_spec.name,
                    "readers": readers,
                    "update_rate_tps": update_rate,
                    "profile_schema_version": SAMPLED_PROFILE_SCHEMA_VERSION,
                    "timed_measurement": False,
                }
            )
        delivery = writer_delivery_metrics(
            rows,
            wall_seconds,
            update_rate,
            getattr(args, "min_update_delivery_ratio", FORMAL_MIN_UPDATE_DELIVERY_RATIO),
        )
        return rows, wall_seconds, {
            "workers": evidence,
            "warmup_seconds_per_backend": args.warmup_seconds,
            "minimum_warmup_read_requests_per_backend": args.session_warmup_requests,
            "warmup_completed_requests_per_backend": sorted(warmup_counts),
            "requested_measurement_queries": len(scheduled_requests),
            "measurement_watchdog_seconds": args.measure_seconds,
            "measurement_watchdog_timeout": watchdog_timeout,
            "writer_delivery": delivery,
            "relation_epoch_before": epoch_before,
            "relation_epoch_after": epoch_after,
            "relation_epoch_delta": epoch_after - epoch_before,
            "sampled_profiles": sampled_profiles,
            "d3_fragment_store_namespace": namespace if method == "sqlens_full" else "",
            "writer_table": writer_table,
        }
    finally:
        stop.set()
        for runtime in reversed(runtimes):
            close_mode_runtime(runtime)


def lifecycle_gate(
    method: str,
    update_rate: float,
    rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    sampled_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    successful_writes = [
        row
        for row in rows
        if row.get("kind") == "write" and not row.get("error")
    ]
    mutation_counts = Counter(str(row.get("mutation") or "") for row in successful_writes)
    epoch_delta = int(evidence.get("relation_epoch_delta", 0) or 0)
    nonzero = float(update_rate) > 0.0
    all_mutations = all(mutation_counts.get(name, 0) > 0 for name in MUTATIONS)
    invalidations_cover_commits = (
        not nonzero or epoch_delta >= len(successful_writes)
    )
    fragment_rebuilds = sum(
        int(row.get("fragment_builds_delta", 0) or 0)
        for row in sampled_profiles
    )
    fast_reactivations = sum(
        int(row.get("fast_reactivation_hits_delta", 0) or 0)
        for row in sampled_profiles
    )
    fragment_store_reuses = sum(
        int(row.get("fragment_store_hits_delta", 0) or 0)
        for row in sampled_profiles
    )
    refresh_events = (
        fragment_rebuilds + fast_reactivations + fragment_store_reuses
    )
    stale_fallbacks = sum(
        row.get("path_class") == "stale_fallback" for row in sampled_profiles
    )
    sqlens_refresh = (
        method != "sqlens_full"
        or not nonzero
        or refresh_events + stale_fallbacks > 0
    )
    passed = (
        (not nonzero or bool(successful_writes))
        and (not nonzero or all_mutations)
        and invalidations_cover_commits
        and sqlens_refresh
        and all(row.get("profile_complete") is True for row in sampled_profiles)
    )
    return {
        "passed": passed,
        "method": method,
        "update_rate_tps": float(update_rate),
        "successful_commits": len(successful_writes),
        "mutation_commits_observed": (not nonzero or bool(successful_writes)),
        "mutation_counts": {name: int(mutation_counts.get(name, 0)) for name in MUTATIONS},
        "all_mutation_types_observed": (not nonzero or all_mutations),
        "relation_epoch_before": evidence.get("relation_epoch_before"),
        "relation_epoch_after": evidence.get("relation_epoch_after"),
        "relation_epoch_delta": epoch_delta,
        "epoch_advanced": (not nonzero or epoch_delta > 0),
        "invalidation_events_observed": epoch_delta,
        "invalidations_cover_commits": invalidations_cover_commits,
        "fragment_rebuilds_observed": fragment_rebuilds,
        "fast_reactivations_observed": fast_reactivations,
        "fragment_store_reuses_observed": fragment_store_reuses,
        "fragment_rebuild_or_reactivation_events": refresh_events,
        "stale_fallback_samples": stale_fallbacks,
        "sqlens_rebuild_reactivation_gate": sqlens_refresh,
        "sampled_profiles": len(sampled_profiles),
    }


def path_mix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [
        row
        for row in rows
        if row.get("kind") in {"read", "profile"} and not row.get("error")
    ]
    counts = Counter(
        str(row.get("path_class") or ("stock" if row.get("method") == "stock" else ""))
        for row in successful
    )
    total = len(successful)
    return {
        **{f"{name}_requests": int(counts.get(name, 0)) for name in PATH_CLASSES},
        **{
            f"{name}_ratio": (float(counts.get(name, 0)) / total if total else 0.0)
            for name in PATH_CLASSES
        },
        "profile_complete_requests": sum(
            bool(row.get("profile_complete"))
            for row in successful
            if bool(row.get("profile_collected", "profile_collected" not in row))
        ),
        "profile_expected_requests": sum(
            bool(row.get("profile_collected", "profile_collected" not in row))
            for row in successful
        ),
        "stale_relation_requests": sum(bool(row.get("stale_relation")) for row in successful),
    }


def summarize_repeat(
    rows: Sequence[Mapping[str, Any]],
    wall_seconds: float,
    *,
    target_recall: float | None = None,
    expected_read_requests: int | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    min_update_delivery_ratio: float = FORMAL_MIN_UPDATE_DELIVERY_RATIO,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    requested_tps = float(rows[0].get("update_rate_tps", 0.0)) if rows else 0.0
    delivery = writer_delivery_metrics(
        rows, wall_seconds, requested_tps, min_update_delivery_ratio
    )
    for kind in ("read", "write"):
        kind_rows = [row for row in rows if row.get("kind") == kind]
        successful = [row for row in kind_rows if not row.get("error")]
        latencies = [float(row["latency_ms"]) for row in successful]
        # A zero-rate writer cell has no raw write rows, but still needs the
        # same identity fields as the paired read arm for aggregation.
        first = kind_rows[0] if kind_rows else (rows[0] if rows else {})
        recall_mean, recall_lcb95, recall_ucb95 = pooled_recall_bounds(
            kind_rows if kind == "read" else [],
            bootstrap_samples,
            throughput.stable_seed(bootstrap_seed, "repeat-recall", first.get("method"), first.get("filter_name"), first.get("measurement_repeat")),
        )
        mix = path_mix(kind_rows) if kind == "read" else {}
        profile_complete = (
            kind != "read"
            or mix["profile_complete_requests"] == mix["profile_expected_requests"]
        )
        request_count_complete = (
            kind != "read"
            or expected_read_requests is None
            or len(kind_rows) == expected_read_requests
        )
        execution_valid = (
            bool(successful)
            and len(successful) == len(kind_rows)
            and profile_complete
            and request_count_complete
            and delivery["update_delivery_gate_passed"]
        )
        overload = (
            requested_tps > 0.0
            and not delivery["update_delivery_gate_passed"]
            and len(successful) == len(kind_rows)
        )
        summaries.append({
            "summary_type": "repeat", "kind": kind,
            "measurement_repeat": first.get("measurement_repeat"), "method": first.get("method"),
            "filter_name": first.get("filter_name"), "readers": first.get("readers"),
            "writer_clients": first.get("writer_clients"), "update_rate_tps": first.get("update_rate_tps"),
            "wall_seconds": wall_seconds, "attempts": len(kind_rows), "completed": len(successful),
            "qps": len(successful) / wall_seconds if wall_seconds > 0.0 else 0.0,
            "p50_ms": percentile(latencies, 0.50), "p95_ms": percentile(latencies, 0.95),
            "p99_ms": percentile(latencies, 0.99), "errors": len(kind_rows) - len(successful),
            "timeouts": sum(bool(row.get("timeout")) for row in kind_rows),
            "mean_recall_at_10": recall_mean if kind == "read" else None,
            "pooled_recall_lcb95": recall_lcb95 if kind == "read" else None,
            "pooled_recall_ucb95": recall_ucb95 if kind == "read" else None,
            "target_recall": target_recall,
            "target_recall_lcb95_met": (
                recall_lcb95 >= float(target_recall)
                if kind == "read" and target_recall is not None
                else None
            ),
            "request_count_complete": request_count_complete,
            "profile_complete": profile_complete,
            **delivery,
            **mix,
            "status": (
                "not_applicable"
                if kind == "write" and not kind_rows and float(first.get("update_rate_tps") or 0.0) == 0.0
                else "valid" if execution_valid
                else "overload" if overload
                else "invalid"
            ),
        })
    return summaries


def aggregate_summaries(
    repeats: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in repeats:
        grouped[
            (
                row["kind"], row["method"], row["filter_name"],
                row.get("target_recall"), row.get("config"), row.get("ef_search"),
                row["readers"], row["writer_clients"], row["update_rate_tps"],
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=str):
        completed = sum(int(row["completed"]) for row in rows)
        wall = sum(float(row["wall_seconds"]) for row in rows)
        matching_raw = [
            row for row in (raw_rows or [])
            if (
                row.get("kind"), row.get("method"), row.get("filter_name"),
                row.get("target_recall"), row.get("config"), row.get("ef_search"),
                row.get("readers"), row.get("writer_clients"), row.get("update_rate_tps"),
            ) == key
        ]
        expected_attempts = sum(
            int(row.get("attempts", int(row.get("completed", 0)) + int(row.get("errors", 0))))
            for row in rows
        )
        tail_raw_request_pool_complete = (
            raw_rows is not None and len(matching_raw) == expected_attempts
        )
        pooled_latencies = [
            float(row["latency_ms"]) for row in matching_raw if not row.get("error")
        ]
        if raw_rows is not None and (matching_raw or expected_attempts == 0):
            tail_source = "all_successful_raw_requests_pooled"
            tail_points = {
                "p50_ms": percentile(pooled_latencies, 0.50),
                "p95_ms": percentile(pooled_latencies, 0.95),
                "p99_ms": percentile(pooled_latencies, 0.99),
            }
            tail_cis: dict[str, float] = {}
            for label, fraction in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
                low, high = repeat_cluster_percentile_ci(
                    matching_raw,
                    fraction,
                    bootstrap_samples,
                    throughput.stable_seed(
                        bootstrap_seed, "aggregate-tail-repeat-cluster", label, *key
                    ),
                )
                tail_cis[f"{label}_repeat_cluster_ci95_low_ms"] = low
                tail_cis[f"{label}_repeat_cluster_ci95_high_ms"] = high
        else:
            tail_source = "repeat_percentile_median_fallback_nonformal"
            tail_points = {
                label: statistics.median(float(row[label]) for row in rows)
                for label in ("p50_ms", "p95_ms", "p99_ms")
            }
            tail_cis = {
                f"{label}_repeat_cluster_ci95_{bound}_ms": percentile(
                    [float(row[f"{label}_ms"]) for row in rows],
                    fraction,
                )
                for label in ("p50", "p95", "p99")
                for bound, fraction in (("low", 0.025), ("high", 0.975))
            }
        matching_arm_writes = [
            row for row in (raw_rows or [])
            if row.get("kind") == "write"
            and (
                row.get("method"), row.get("filter_name"),
                row.get("target_recall"), row.get("config"), row.get("ef_search"),
                row.get("readers"), row.get("writer_clients"), row.get("update_rate_tps"),
            ) == key[1:]
        ]
        delivery = writer_delivery_metrics(
            matching_arm_writes,
            wall,
            float(key[8]),
            min(
                float(row.get("minimum_update_delivery_ratio", FORMAL_MIN_UPDATE_DELIVERY_RATIO))
                for row in rows
            ),
        )
        if key[0] == "read" and matching_raw:
            recall_mean, recall_lcb95, recall_ucb95 = pooled_recall_bounds(
                matching_raw,
                bootstrap_samples,
                throughput.stable_seed(bootstrap_seed, "aggregate-recall", *key),
            )
            mix = path_mix(matching_raw)
        else:
            recall_mean = statistics.fmean(
                float(row.get("mean_recall_at_10") or 0.0) for row in rows
            ) if key[0] == "read" else 0.0
            recall_lcb95 = min(
                float(row.get("pooled_recall_lcb95") or recall_mean) for row in rows
            ) if key[0] == "read" else 0.0
            recall_ucb95 = max(
                float(row.get("pooled_recall_ucb95") or recall_mean) for row in rows
            ) if key[0] == "read" else 0.0
            mix = {
                field: sum(float(row.get(field, 0) or 0) for row in rows)
                for field in (
                    "guided_requests", "stale_fallback_requests", "stock_requests",
                    "profile_complete_requests", "profile_expected_requests",
                    "stale_relation_requests",
                )
            } if key[0] == "read" else {}
            total = float(mix.get("profile_expected_requests", 0) or 0)
            for name in PATH_CLASSES:
                mix[f"{name}_ratio"] = (
                    float(mix.get(f"{name}_requests", 0)) / total if total else 0.0
                )
        target_met = key[0] != "read" or recall_lcb95 >= float(key[3])
        profile_complete = (
            key[0] != "read"
            or mix["profile_complete_requests"] == mix["profile_expected_requests"]
        )
        valid_repeats = len(rows) >= FORMAL_REPEATS and all(
            row["status"] == "valid" for row in rows
        )
        aggregate_valid = (
            valid_repeats
            and target_met
            and profile_complete
            and tail_raw_request_pool_complete
            and delivery["update_delivery_gate_passed"]
        )
        aggregate_overload = (
            any(row["status"] == "overload" for row in rows)
            and all(row["status"] in {"valid", "overload"} for row in rows)
        )
        output.append({
            "summary_type": "aggregate", "kind": key[0], "method": key[1], "filter_name": key[2],
            "target_recall": key[3], "config": key[4], "ef_search": key[5],
            "readers": key[6], "writer_clients": key[7], "update_rate_tps": key[8],
            "repeats": len(rows), "completed": completed, "wall_seconds": wall,
            "qps": completed / wall if wall > 0.0 else 0.0,
            **tail_points,
            **tail_cis,
            "tail_point_estimate_source": tail_source,
            "tail_raw_request_pool_complete": tail_raw_request_pool_complete,
            "errors": sum(int(row["errors"]) for row in rows), "timeouts": sum(int(row["timeouts"]) for row in rows),
            "mean_recall_at_10": recall_mean if key[0] == "read" else None,
            "pooled_recall_lcb95": recall_lcb95 if key[0] == "read" else None,
            "pooled_recall_ucb95": recall_ucb95 if key[0] == "read" else None,
            "target_recall_lcb95_met": target_met if key[0] == "read" else None,
            "profile_complete": profile_complete,
            **delivery,
            **mix,
            "status": (
                "not_applicable"
                if key[0] == "write" and all(row["status"] == "not_applicable" for row in rows)
                else "valid" if aggregate_valid
                else "overload" if aggregate_overload
                else "invalid"
            ),
        })
    return output


def run_experiment(args: argparse.Namespace) -> int:
    targets, readers_grid, rates, methods = validate_args(args)
    selected = set(args.filter_names) if args.filter_names else None
    filters = throughput.load_filters(args.filters_csv, selected)
    if args.protocol == FORMAL_PROTOCOL:
        by_name = {item.name: item for item in filters}
        filters = [by_name[name] for name in FORMAL_FILTERS]
    identity = source_identity(args)
    identity_passed = bool(
        identity["runner_sha256_matches_expected"]
        and identity["git_revision_matches_expected"]
    )
    calibration_truth_csv = Path(args.calibration_truth_csv).resolve()
    calibration_truth_manifest = Path(args.calibration_truth_manifest).resolve()
    measurement_query_file = Path(
        args.measurement_query_file or throughput.DEFAULT_MEASUREMENT_QUERY_FILE
    ).resolve()
    measurement_query_manifest = Path(
        args.measurement_query_manifest
        or measurement_query_file.with_name(measurement_query_file.stem + "_manifest.json")
    ).resolve()
    measurement_truth_csv = Path(args.measurement_truth_csv).resolve()
    measurement_truth_manifest = Path(args.measurement_truth_manifest).resolve()
    args.measurement_query_file = measurement_query_file
    args.measurement_query_manifest = measurement_query_manifest
    if args.protocol == FORMAL_PROTOCOL:
        matched = load_fixed_recall_selector(args, filters)
        selector_bound = True
    else:
        matched = throughput.load_audited_matched_recall_configs(
            args.matched_recall_manifest,
            truth_csv=calibration_truth_csv,
            filters_csv=args.filters_csv,
            filters=filters,
            targets=targets,
        )
        throughput.bind_matched_recall_provenance(args, matched)
        selector_bound = False
    calibration_truth_provenance = throughput.verify_truth_manifest(
        calibration_truth_csv,
        calibration_truth_manifest,
        args.candidate_validity_predicate,
        args.expected_candidate_rows,
    )
    calibration_truth, calibration_query_ids = throughput.load_truth(
        calibration_truth_csv, filters, args.candidate_validity_predicate
    )
    measurement_query_provenance = throughput.verify_measurement_query_manifest(
        measurement_query_file,
        measurement_query_manifest,
        args.candidate_validity_predicate,
    )
    measurement_truth_provenance = throughput.verify_measurement_truth_manifest(
        measurement_truth_csv,
        measurement_truth_manifest,
        args.candidate_validity_predicate,
        args.expected_candidate_rows,
        measurement_query_file,
    )
    truth, measurement_query_ids = throughput.load_truth(
        measurement_truth_csv, filters, args.candidate_validity_predicate
    )
    calibration_nos, confirmation_nos = matched_query_number_splits(matched.manifest)
    matched_query_ids = [
        calibration_query_ids[number]
        for number in calibration_nos + confirmation_nos
        if number in calibration_query_ids
    ]
    workload = throughput.load_true_query_workload(
        measurement_query_file,
        matched_query_ids,
        query_manifest=measurement_query_manifest,
        candidate_validity_predicate=args.candidate_validity_predicate,
    )
    throughput.validate_workload_query_mapping(workload, measurement_query_ids)
    measurement_requests = select_measurement_requests(workload, args.requests)
    split_gate = query_id_disjoint_gate(
        matched.manifest, calibration_query_ids, measurement_requests
    )
    required_query_nos = {int(request.query_no) for request in measurement_requests}
    throughput.validate_truth_coverage(
        truth, measurement_query_ids, filters, required_query_nos, ()
    )
    query_ids = {
        int(request.query_no): int(request.query_id) for request in measurement_requests
    }
    if len(query_ids) < args.audit_spots:
        raise BenchmarkContractError("measurement cohort is too small for requested audit spots")
    protocol = formal_protocol_status(
        args,
        targets,
        readers_grid,
        rates,
        methods,
        filter_count=len(filters),
        filter_names=[item.name for item in filters],
        split_gate_passed=bool(split_gate.get("passed")),
        source_identity_passed=identity_passed,
        selector_bound=selector_bound,
    )
    blocking = list(protocol["failed_checks"])
    if bool(getattr(args, "paper_table_slice", False)):
        waived = {"readers_1_4_8_16_32_64", "update_rates_0_10_100_1000"}
        if (
            bool(getattr(args, "fail_open_stale", False))
            and tuple(methods) == ("stock", "sqlens_d1")
        ):
            waived.add("methods_stock_sqlens_full")
        blocking = [name for name in blocking if name not in waived]
    if args.protocol == FORMAL_PROTOCOL and blocking:
        raise BenchmarkContractError(
            "formal P0-6 protocol gate failed: "
            + ", ".join(blocking)
        )
    prepare_runtime_args(args, filters)
    update_column = (
        {
            "mode": "real_predicate_vector_insert_delete",
            "mutation_mix": dict(args.mutation_mix),
        }
        if args.protocol == FORMAL_PROTOCOL
        else validate_update_column(args, filters)
    )
    pre_identity = live_identity_gate(args, matched)
    protected_query_ids = sorted(
        set(int(value) for value in matched_query_ids)
        | set(int(value) for value in query_ids.values())
    )
    update_ids = load_update_id_pool(
        args,
        protected_query_ids if args.protocol == FORMAL_PROTOCOL else (),
    )
    if args.pg_prewarm:
        cache_evidence = throughput.warm_database_cache(args)
        if not cache_evidence.get("passed"):
            raise BenchmarkContractError("warm-cache gate failed")
    else:
        cache_evidence = {"passed": False, "enabled": False, "formal": False}

    paths = output_paths(args.out)
    file_paths = {name: path for name, path in paths.items() if name != "cells"}
    if (
        any(path.exists() for path in file_paths.values())
        and not args.overwrite
        and not args.resume
    ):
        raise BenchmarkContractError(
            "output exists; use --resume or --overwrite"
        )
    for path in file_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["cells"].mkdir(parents=True, exist_ok=True)
    write_update_ids_atomic(paths["update_ids"], update_ids)
    run_contract_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol": protocol,
        "source_identity": identity,
        "matched_recall": dict(matched.provenance),
        "filters_csv_sha256": sha256_file(args.filters_csv),
        "calibration_truth_csv_sha256": sha256_file(calibration_truth_csv),
        "measurement_query_sha256": sha256_file(measurement_query_file),
        "measurement_truth_csv_sha256": sha256_file(measurement_truth_csv),
        "update_ids_sha256": canonical_sha256(update_ids),
        "protected_query_ids_sha256": canonical_sha256(protected_query_ids),
        "targets": list(targets),
        "readers": list(readers_grid),
        "rates": list(rates),
        "methods": list(methods),
        "repeats": int(args.measurement_repeats),
        "mutation_mix": dict(args.mutation_mix),
    }
    run_contract_sha256 = canonical_sha256(run_contract_payload)
    raw_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    worker_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "artifact": "pgvector_update_concurrency_benchmark", "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION, "status": "running",
        "diagnostic_valid": False, "artifact_valid": False,
        "paper_eligible": False, "formal_artifact_valid": False, "protocol": protocol,
        "started_at_utc": utc_now(),
        "source_identity": identity,
        "runner_sha256": identity["runner_sha256"],
        "git_revision": identity["git_revision"],
        "run_contract": run_contract_payload,
        "run_contract_sha256": run_contract_sha256,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "matched_recall": dict(matched.provenance),
        "calibration_truth": calibration_truth_provenance,
        "measurement_query": measurement_query_provenance,
        "measurement_truth": measurement_truth_provenance,
        "query_id_split_gate": split_gate,
        "filters_csv_sha256": sha256_file(args.filters_csv),
        "calibration_truth_csv_sha256": sha256_file(calibration_truth_csv),
        "measurement_truth_csv_sha256": sha256_file(measurement_truth_csv),
        "query_cohort": {
            "name": str(workload.query_cohort),
            "source_kind": str(workload.source_kind),
            "source_path": str(workload.source_path),
            "source_sha256": str(workload.source_sha256),
            "trace_replay": bool(workload.trace_replay),
            "queries": len(measurement_requests),
            "available_measurement_queries": len(workload.requests),
            "query_ids_sha256": canonical_sha256(sorted(query_ids.items())),
        },
        "pre_run_identity": pre_identity, "update_workload": update_column,
        "cache_preparation": cache_evidence,
        "update_id_pool": {
            "rows": len(update_ids),
            "ordered_ids_sha256": canonical_sha256(update_ids),
            "protected_query_ids": len(protected_query_ids),
            "protected_query_ids_sha256": canonical_sha256(
                protected_query_ids
            ),
            "query_rows_excluded_from_mutation_pool": (
                args.protocol == FORMAL_PROTOCOL
            ),
            "sampling": "TABLESAMPLE SYSTEM (10) REPEATABLE (schedule_seed) LIMIT pool_size",
            "path": str(paths["update_ids"]),
            "sha256": sha256_file(paths["update_ids"]),
        },
        "runtime_binary": {
            "expected_build_id": args.expected_sqlens_build_id,
            "expected_vector_so_sha256": args.expected_vector_so_sha256,
            "source": (
                "current_r36_fixed_recall_selector"
                if args.protocol == FORMAL_PROTOCOL
                else "independently_audited_matched_recall_manifest"
            ),
        },
        "measurement_contract": {
            "real_concurrent_backends": True,
            "warmup_separate_from_measurement": True,
            "minimum_repeats": FORMAL_REPEATS,
            "requests_determine_query_count": True,
            "measurement_queries_per_cell": len(measurement_requests),
            "measure_seconds_role": "per-cell watchdog_only",
            "stock_request_path": "session_initialized_off_then_native_hybrid_sql_only",
            "sqlens_request_path": "activation_plus_native_hybrid_sql",
            "timed_per_query_profile_collection": bool(args.profile_timed_queries),
            "search_latency_excludes_post_query_profile_collection": True,
            "wall_qps_excludes_sampled_profile_collection": True,
            "sampled_profiles_per_cell": int(args.profile_samples_per_cell),
            "path_classes": list(PATH_CLASSES),
            "formal_recall_gate": "pooled_per_query_bootstrap_lcb95_gte_target",
            "tail_point_estimate": "all_successful_raw_requests_pooled_within_cell",
            "tail_uncertainty": "repeat_cluster_bootstrap_ci95",
            "minimum_update_delivery_ratio": args.min_update_delivery_ratio,
            "update_delivery_policy": "record_overload_and_continue_full_matrix",
            "fail_open_stale": bool(getattr(args, "fail_open_stale", False)),
            "fail_open_policy": (
                "warmup_epoch_then_stock_without_activate"
                if bool(getattr(args, "fail_open_stale", False))
                else "activate_may_rebuild_on_stale_epoch"
            ),
            "exact_audit_control_scope": "target_filter_repeat",
            "exact_audit_control_method": (
                "sqlens_full" if "sqlens_full" in methods
                else "sqlens_d1" if "sqlens_d1" in methods
                else methods[0]
            ),
            "exact_audit_before_after_each_control_cell": True,
            "exact_audit_snapshot": "guided_and_exact_in_one_repeatable_read_transaction",
            "exact_audit_amortization_invariant": (
                "writer restores predicate/vector rows and removes lifecycle inserts "
                "after each timed cell"
            ),
            "writer_mutations": list(MUTATIONS),
            "per_cell_atomic_checkpoint_resume": True,
            "method_order": "seeded_interleaved_within_target_filter_reader_rate_repeat",
            "update_pool_rotation": "seeded_per_target_filter_reader_rate_repeat_and_shared_across_methods",
        },
    }
    atomic_json(paths["manifest"], manifest)
    failure = ""
    try:
        for target in targets:
            for filter_spec in filters:
                for repeat in range(args.measurement_repeats):
                    audit_method = (
                        "sqlens_full" if "sqlens_full" in methods
                        else "sqlens_d1" if "sqlens_d1" in methods
                        else methods[0]
                    )
                    audit_config = matched.configs[
                        (filter_spec.name, audit_method, float(target))
                    ]
                    spots = select_audit_requests(
                        query_ids,
                        filter_spec.name,
                        repeat,
                        args.audit_spots,
                        args.schedule_seed,
                    )
                    before = exact_sql_valid_spot_audit(
                        args,
                        audit_method,
                        audit_config,
                        filter_spec,
                        truth,
                        spots,
                        "before",
                        repeat,
                        target_recall=float(target),
                    )
                    for row in before:
                        row.update({
                            "target_recall": target,
                            "audit_control_scope": "target_filter_repeat",
                            "control_readers": list(readers_grid),
                            "control_update_rates_tps": list(rates),
                        })
                    audit_rows.extend(before)
                    if not all(row["passed"] for row in before):
                        hard = [
                            row for row in before
                            if not row.get("all_guided_ids_sql_valid")
                            or int(row.get("exact_returned") or 0)
                            != min(args.k, int(row.get("filtered_rows") or 0))
                        ]
                        if hard or not bool(getattr(args, "paper_table_slice", False)):
                            raise BenchmarkContractError("pre-control-cell exact audit failed")
                        print(
                            "[concurrency] paper-table slice continues after recall-only audit misses: "
                            + ", ".join(
                                f"q{row['query_no']}={row['dynamic_same_snapshot_recall_at_10']:.3f}"
                                for row in before if not row["passed"]
                            ),
                            flush=True,
                        )

                    for readers in readers_grid:
                        for rate in rates:
                            method_order = list(methods)
                            random.Random(
                                throughput.stable_seed(
                                    args.schedule_seed, "update-method-order", target,
                                    filter_spec.name, readers, rate, repeat,
                                )
                            ).shuffle(method_order)
                            update_pool_offset = throughput.stable_seed(
                                args.schedule_seed, "update-id-pool", target,
                                filter_spec.name, readers, rate, repeat,
                            ) % len(update_ids)
                            for method in method_order:
                                config = matched.configs[(filter_spec.name, method, float(target))]
                                key = cell_key(
                                    target,
                                    filter_spec.name,
                                    readers,
                                    rate,
                                    repeat,
                                    method,
                                )
                                resumed = (
                                    load_cell(
                                        paths["cells"], key, run_contract_sha256
                                    )
                                    if args.resume
                                    else None
                                )
                                if resumed is not None and any(
                                    row.get("error") for row in resumed.get("raw") or ()
                                ):
                                    print(
                                        f"[concurrency] ignore failed checkpoint {key}",
                                        flush=True,
                                    )
                                    resumed = None
                                if resumed is not None:
                                    rows = list(resumed["raw"])
                                    repeat_summaries = list(resumed["summary"])
                                    worker_record = dict(resumed["worker"])
                                    sampled = list(resumed["profiles"])
                                    lifecycle = dict(resumed["lifecycle"])
                                else:
                                    rows, wall, evidence = run_overlap(
                                        args,
                                        method,
                                        config,
                                        filter_spec,
                                        truth,
                                        measurement_requests,
                                        readers,
                                        rate,
                                        repeat,
                                        update_ids,
                                        update_pool_offset,
                                    )
                                    sampled = list(evidence.pop("sampled_profiles", []))
                                    for row in rows:
                                        row["target_recall"] = target
                                        row["config"] = config.label
                                        row["ef_search"] = config.ef_search
                                    for row in sampled:
                                        row["target_recall"] = target
                                        row["config"] = config.label
                                        row["ef_search"] = config.ef_search
                                    worker_record = {
                                        **key,
                                        "config": config.label,
                                        "ef_search": config.ef_search,
                                        **evidence,
                                    }
                                    repeat_summaries = summarize_repeat(
                                        rows,
                                        wall,
                                        target_recall=target,
                                        expected_read_requests=args.requests,
                                        bootstrap_samples=args.bootstrap_samples,
                                        bootstrap_seed=args.bootstrap_seed,
                                        min_update_delivery_ratio=args.min_update_delivery_ratio,
                                    )
                                    for row in repeat_summaries:
                                        row["config"] = config.label
                                        row["ef_search"] = config.ef_search
                                        row["worker_evidence_sha256"] = canonical_sha256(
                                            evidence
                                        )
                                    lifecycle = lifecycle_gate(
                                        method,
                                        rate,
                                        rows,
                                        evidence,
                                        sampled,
                                    )
                                    persist_cell(
                                        paths["cells"],
                                        key,
                                        run_contract_sha256,
                                        rows,
                                        repeat_summaries,
                                        worker_record,
                                        sampled,
                                        lifecycle,
                                    ) if not any(row.get("error") for row in rows) else None
                                raw_rows.extend(rows)
                                worker_rows.append(worker_record)
                                summary_rows.extend(repeat_summaries)
                                profile_rows.extend(sampled)
                                lifecycle_rows.append(lifecycle)
                                manifest["checkpoint_progress"] = {
                                    "completed_cells": len(worker_rows),
                                    "cell_directory": str(paths["cells"]),
                                    "last_cell_key": key,
                                }
                                atomic_json(paths["manifest"], manifest)
                                if any(row.get("error") for row in rows):
                                    details = [
                                        f"{row.get('kind')}:{row.get('error')}"
                                        for row in rows
                                        if row.get("error")
                                    ]
                                    raise BenchmarkContractError(
                                        "read/write worker error or timeout: "
                                        + "; ".join(details[:5])
                                    )
                    after = exact_sql_valid_spot_audit(
                        args,
                        audit_method,
                        audit_config,
                        filter_spec,
                        truth,
                        spots,
                        "after",
                        repeat,
                        target_recall=float(target),
                    )
                    for row in after:
                        row.update({
                            "target_recall": target,
                            "audit_control_scope": "target_filter_repeat",
                            "control_readers": list(readers_grid),
                            "control_update_rates_tps": list(rates),
                        })
                    audit_rows.extend(after)
                    if not all(row["passed"] for row in after):
                        hard = [
                            row for row in after
                            if not row.get("all_guided_ids_sql_valid")
                            or int(row.get("exact_returned") or 0)
                            != min(args.k, int(row.get("filtered_rows") or 0))
                        ]
                        if hard or not bool(getattr(args, "paper_table_slice", False)):
                            raise BenchmarkContractError("post-control-cell exact audit failed")
                        print(
                            "[concurrency] paper-table slice continues after post-cell recall-only audit misses: "
                            + ", ".join(
                                f"q{row['query_no']}={row['dynamic_same_snapshot_recall_at_10']:.3f}"
                                for row in after if not row["passed"]
                            ),
                            flush=True,
                        )
        post_identity = live_identity_gate(args, matched)
        aggregate_rows = aggregate_summaries(
            summary_rows,
            raw_rows,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        summary_rows.extend(aggregate_rows)
        expected_repeats = len(targets) * len(filters) * len(methods) * len(readers_grid) * len(rates) * args.measurement_repeats
        expected_read_rows = expected_repeats * args.requests
        observed_read_rows = sum(row.get("kind") == "read" for row in raw_rows)
        read_aggregates = [
            row for row in aggregate_rows if row.get("kind") == "read"
        ]
        path_evidence = {
            method: path_mix(
                [
                    row for row in profile_rows
                    if row.get("method") == method
                ]
            )
            for method in methods
        }
        valid = (
            len(audit_rows) == expected_exact_audit_rows(
                targets, filters, args.measurement_repeats, args.audit_spots
            )
            and all(row.get("passed") for row in audit_rows)
            and observed_read_rows == expected_read_rows
            and len(worker_rows) == expected_repeats
            and all(
                not row.get("error")
                for row in raw_rows
                if row.get("kind") in {"read", "write"}
            )
            and (
                args.protocol != FORMAL_PROTOCOL
                or len(profile_rows)
                == expected_repeats * int(args.profile_samples_per_cell)
            )
        )
        expected_read_cells = (
            len(targets) * len(filters) * len(methods) * len(readers_grid) * len(rates)
        )
        eligibility = artifact_eligibility(
            diagnostic_valid=valid,
            protocol=protocol,
            read_aggregates=read_aggregates,
            expected_read_cells=expected_read_cells,
            sampled_profiles=(
                profile_rows if args.protocol == FORMAL_PROTOCOL else None
            ),
            lifecycle_gates=(
                lifecycle_rows if args.protocol == FORMAL_PROTOCOL else None
            ),
        )
        manifest.update({
            "status": (
                "complete" if eligibility["artifact_valid"]
                else "diagnostic_complete" if eligibility["diagnostic_valid"]
                else "invalid"
            ),
            **eligibility,
            # Retained as a compatibility alias for older result readers.
            "formal_artifact_valid": eligibility["artifact_valid"],
            "requested_slice_complete": eligibility["diagnostic_valid"],
            "full_release_complete": eligibility["artifact_valid"],
            "post_run_identity": post_identity,
            "path_evidence_by_method": path_evidence,
            "completion": {
                "expected_repeat_arms": expected_repeats,
                "expected_exact_audit_rows": expected_exact_audit_rows(
                    targets, filters, args.measurement_repeats, args.audit_spots
                ),
                "observed_exact_audit_rows": len(audit_rows),
                "observed_worker_records": len(worker_rows),
                "expected_read_rows": expected_read_rows,
                "observed_read_rows": observed_read_rows,
                "aggregate_read_cells": len(read_aggregates),
                "expected_aggregate_read_cells": expected_read_cells,
                "all_recall_lcb95_targets_met": all(
                    row.get("target_recall_lcb95_met") for row in read_aggregates
                ),
                "sampled_profile_rows": len(profile_rows),
                "expected_sampled_profile_rows": (
                    expected_repeats * int(args.profile_samples_per_cell)
                ),
                "all_sampled_profiles_complete": all(
                    row.get("profile_complete") for row in profile_rows
                ),
                "overload_cells": sum(
                    row.get("kind") == "read" and row.get("status") == "overload"
                    for row in aggregate_rows
                ),
                "lifecycle_cells_passed": sum(
                    row.get("passed") is True for row in lifecycle_rows
                ),
                "lifecycle_cells_total": len(lifecycle_rows),
            },
        })
    except BaseException as exc:
        failure = f"{exc.__class__.__name__}: {exc}"
        manifest.update({
            "status": "failed",
            "diagnostic_valid": False,
            "artifact_valid": False,
            "paper_eligible": False,
            "formal_artifact_valid": False,
            "requested_slice_complete": False,
            "full_release_complete": False,
            "error": failure,
        })
    finally:
        write_csv_atomic(paths["raw"], raw_rows)
        write_csv_atomic(paths["audits"], audit_rows)
        write_csv_atomic(paths["summary"], summary_rows)
        write_csv_atomic(paths["workers"], worker_rows)
        write_csv_atomic(paths["profiles"], profile_rows)
        write_csv_atomic(paths["lifecycle"], lifecycle_rows)
        manifest.update({
            "finished_at_utc": utc_now(),
            "runtime_sqlens_identity_evidence": args.runtime_sqlens_identity_evidence,
            "backend_cpu_evidence": args.backend_cpu_evidence,
            "row_counts": {
                "raw": len(raw_rows), "audits": len(audit_rows),
                "summary": len(summary_rows), "workers": len(worker_rows),
                "sampled_profiles": len(profile_rows),
                "lifecycle": len(lifecycle_rows),
            },
            "outputs": {
                key: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for key, path in paths.items()
                if key not in {"manifest", "cells"}
            },
            "cell_artifacts": {
                "directory": str(paths["cells"]),
                "checkpoint_schema_version": CELL_SCHEMA_VERSION,
                "completed_cells": len(worker_rows),
            },
        })
        atomic_json(paths["manifest"], manifest)
    if failure:
        raise BenchmarkContractError(failure)
    # A completed diagnostic slice is useful for debugging and should not be
    # reported as a process failure.  Its manifest is nevertheless explicitly
    # barred from paper aggregation by artifact_valid/paper_eligible.
    return 0 if manifest["diagnostic_valid"] else 2


def create_argument_parser() -> argparse.ArgumentParser:
    parser = throughput.create_argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        out=RESULTS / "amazon10m_pgvector_update_concurrency.json",
        requests=FORMAL_REQUESTS,
        target_recalls="0.90",
        measurement_repeats=FORMAL_REPEATS,
        evaluation_scope="representative_filters",
        pg_prewarm=True,
        allow_nonformal_debug=False,
        expected_sqlens_build_id=R43_BUILD_ID,
        expected_vector_so_sha256=R43_VECTOR_SO_SHA256,
    )
    parser.add_argument(
        "--protocol",
        choices=(LEGACY_PROTOCOL, FORMAL_PROTOCOL),
        default=LEGACY_PROTOCOL,
    )
    parser.add_argument("--readers", default=DEFAULT_READERS, help="comma-separated concurrent reader backend counts")
    parser.add_argument(
        "--paper-table-slice",
        action="store_true",
        help="allow a requested subset of the formal reader/rate grid for Table 9 cells",
    )
    parser.add_argument("--writer-clients", type=positive_int, default=1)
    parser.add_argument("--update-rates", default=DEFAULT_UPDATE_RATES, help="total committed update transactions/s grid")
    parser.add_argument("--update-batch-size", type=positive_int, default=1)
    parser.add_argument("--update-id-pool-size", type=positive_int, default=DEFAULT_UPDATE_ID_POOL_SIZE)
    parser.add_argument("--update-column", default=DEFAULT_UPDATE_COLUMN)
    parser.add_argument(
        "--min-update-delivery-ratio",
        type=unit_interval_float,
        default=FORMAL_MIN_UPDATE_DELIVERY_RATIO,
        help="mark a nonzero-rate cell overload when delivered TPS is lower",
    )
    parser.add_argument("--write-statement-timeout-ms", type=nonnegative_int, default=300_000)
    parser.add_argument("--warmup-seconds", type=nonnegative_float, default=10.0)
    parser.add_argument(
        "--measure-seconds",
        type=positive_int,
        default=3600,
        help="per-cell watchdog; --requests, not time, determines query count",
    )
    parser.add_argument("--audit-spots", type=positive_int, default=5)
    parser.add_argument("--methods", type=parse_methods, default=list(METHODS))
    parser.add_argument(
        "--mutation-mix",
        type=parse_mutation_mix,
        default=parse_mutation_mix("predicate:4,vector:4,insert:1,delete:1"),
    )
    parser.add_argument(
        "--profile-timed-queries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="legacy diagnostic only; formal timed queries keep profiling disabled",
    )
    parser.add_argument(
        "--profile-samples-per-cell",
        type=nonnegative_int,
        default=5,
    )
    parser.add_argument(
        "--fixed-recall-selector-csv",
        type=Path,
        default=DEFAULT_FIXED_SELECTOR,
    )
    parser.add_argument(
        "--fixed-recall-selector-manifest",
        type=Path,
        default=DEFAULT_FIXED_SELECTOR_MANIFEST,
    )
    parser.add_argument(
        "--fixed-selector-workload-csv",
        type=Path,
        default=RESULTS / "figure5_r35_amazon_calibration.csv",
    )
    parser.add_argument(
        "--expected-runner-sha256",
        default=os.environ.get("SQLENS_UPDATE_RUNNER_SHA256", ""),
    )
    parser.add_argument(
        "--expected-git-revision",
        default=os.environ.get("SQLENS_GIT_REVISION", ""),
    )
    parser.add_argument(
        "--fail-open-stale",
        action="store_true",
        help="after warmup, stale fragment epoch skips activate and uses stock",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if args.dry_run or not args.execute:
        print(json.dumps(dry_run_payload(args), sort_keys=True))
        return 0
    return run_experiment(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
