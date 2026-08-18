#!/usr/bin/env python3
"""Formal Amazon-10M deterministic PostgreSQL online D3 replay.

This runner deliberately distinguishes request-driven D3 admission from an
eagerly materialized control.  It runs the modes on three independent,
persistent PostgreSQL sessions in deterministic paired windows.  It never
invents predicates: every request uses one of the fourteen observed Amazon
predicates and one of 10,000 preregistered, unique query vectors with bound
tie-aware exact truth.  Cross-process resume fails closed because the D3/cache
lifecycle is backend-local.  This is a generated, deterministic q10k replay,
not a captured production trace.
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
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_valid_embeddings_filters.csv"
DEFAULT_QUERY_COHORT = ROOT / "results/hybrid_vector_db/amazon10m_unique_embedding_query_cohort_q10200.csv"
DEFAULT_QUERY_COHORT_MANIFEST = ROOT / "results/hybrid_vector_db/amazon10m_unique_embedding_query_cohort_q10200_manifest.json"
DEFAULT_TRUTH = ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal.csv"
DEFAULT_TRUTH_MANIFEST = ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal_manifest.json"
DEFAULT_INDEX_HEALTH_MANIFEST = ROOT / "results/hybrid_vector_db/amazon10m_hnsw_query_health_q10200_measurement_manifest.json"
DEFAULT_TABLE = "amazon_grocery_reviews_10m_pgvector"
DEFAULT_INDEX = "amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx"
DEFAULT_CANDIDATE_VALIDITY_PREDICATE = "embedding_valid"
SQLENS_SOURCE_DIR = ROOT / "third_party/pgvector-sqlens/src"
SQLENS_LOCAL_SHARED_OBJECT = ROOT / "third_party/pgvector-sqlens/vector.so"
MODES = ("stock", "adaptive", "eager_prebuilt")
FORMAL_REQUESTS = 10_000
FORMAL_WINDOW = 100
FORMAL_CALIBRATION_QUERY_COUNT = 100
FORMAL_FINAL_QUERY_COUNT = 10_100
FORMAL_TRUTH_QUERY_COUNT = FORMAL_CALIBRATION_QUERY_COUNT + FORMAL_FINAL_QUERY_COUNT
FORMAL_MEASUREMENT_QUERY_OFFSET = 200
FORMAL_SEED = 20260718
FORMAL_K = 10
FORMAL_EF_SEARCH = 10_000
FORMAL_MAX_SCAN_TUPLES = 5_000_000
FORMAL_SCAN_MEM_MULTIPLIER = 32.0
FORMAL_RECALL_TARGET = 0.90
FORMAL_RECALL_DELTA = 0.01
FORMAL_D3_PROBE_REQUESTS = 2
FORMAL_D3_MIN_BENEFIT_PER_BYTE = 2e-7
FORMAL_D3_MAX_FRAGMENT_MB = 256
FORMAL_D3_PAGE_MIN_SKIP_RATE = 0.80
FORMAL_CACHE_MB = 1024
FORMAL_INPUT_SHA256 = {
    "filters_csv": "ae07c4d94450958f2071bf54f5db48d26c55328538087629cb1375c09bd4bcec",
    "query_cohort": "c25e942bda9f45e435f000eeb938eaecce8e9fc562291bf6f56a57e0ced6a73f",
    "query_cohort_manifest": "bdcfc34d46eddffa70e24cea7cd197df851274eba68ebf6291b1fc407569a8fc",
    "truth": "62e7f280f953828b680b2ae069de221bd6d593e42b241cd3d699ea870a1bfb5b",
    "truth_manifest": "0a6ab22579a8cf01eaa29889bf6ee2e822336d6d1c580b15697b7148a149bff2",
}
CHECKPOINT_SCHEMA_VERSION = 6
PAIRING_SCHEDULE = "deterministic_request_interleaved_round_robin"
FRAGMENT_STORE_RELATION = "public.pgvector_hnsw_fragment_store"
FRAGMENT_EPOCH_RELATION = "public.pgvector_hnsw_fragment_epoch"
FORMAL_WORKLOAD_MANIFEST_NAME = "amazon10m_d3_deterministic_postgresql_online_replay_q10k_unique"
TRACE_KIND = "deterministic_q10k_unique_postgresql_online_replay"


@dataclass(frozen=True)
class FilterSpec:
    name: str
    predicate: str
    atoms: tuple[str, ...]
    expected_rows: int
    actual_pct: float


@dataclass(frozen=True)
class TruthEntry:
    filter_name: str
    query_no: int
    query_id: int
    ids: tuple[int, ...]
    kth_distance_sq: float
    tie_tolerance: float
    strict_closer_count: int = 9
    boundary_tied: bool = False
    query_split: str = "final"


@dataclass(frozen=True)
class Request:
    request_no: int
    phase: str
    window: int
    filter_name: str
    query_no: int
    query_id: int
    reuse_distance: int | None


class Session(Protocol):
    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None: ...

    def one(self) -> Any: ...

    def row(self) -> Any: ...

    def all(self) -> Sequence[Any]: ...


class CursorSession:
    """Small adapter so the lifecycle code is both fakeable and psycopg-neutral."""

    def __init__(self, cursor: Any) -> None:
        self.cursor = cursor

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        self.cursor.execute(sql, params)

    def one(self) -> Any:
        return self.cursor.fetchone()[0]

    def row(self) -> Any:
        return self.cursor.fetchone()

    def all(self) -> Sequence[Any]:
        return self.cursor.fetchall()


@dataclass
class ModeBackend:
    """One long-lived PostgreSQL backend, dedicated to one experimental mode."""

    mode: str
    connection: Any
    session: Session
    backend_pid: int
    database: dict[str, Any]


class BenchmarkContractError(RuntimeError):
    """A run or checkpoint no longer satisfies the formal experiment contract."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_resume_contract() -> dict[str, Any]:
    """Describe the deliberately conservative cross-process recovery policy."""
    return {
        "checkpoint_unit": "complete_cross_mode_paired_window",
        "checkpoint_storage": "per_complete_window_atomic_shards_with_lightweight_manifest",
        "cross_process_resume": "forbidden",
        "policy": "fail_closed",
        "reason": "D3 lifecycle, metadata cache, and persistent fragment-store state are backend-local and have no portable restore API",
        "cache_lifecycle_fingerprints": "audit_only_not_replayable",
        "timed_replay": "not_implemented",
    }


def reject_cross_process_resume(resume_requested: bool) -> None:
    if resume_requested:
        raise BenchmarkContractError(
            "cross-process --resume is disabled: a checkpoint cannot restore backend-local D3/cache state; "
            "start a fresh run after preserving or removing the checkpoint"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_query_ids_sha256(ids: Sequence[int]) -> str:
    """Match the cohort builder's typed, order-sensitive ID digest without NumPy."""
    digest = hashlib.sha256()
    digest.update(b"sqlens-ordered-id-population-v1\0")
    for query_id in ids:
        digest.update(int(query_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def formal_protocol(args: argparse.Namespace) -> bool:
    return (
        args.requests == FORMAL_REQUESTS
        and args.window_size == FORMAL_WINDOW
        and args.truth_query_count == FORMAL_TRUTH_QUERY_COUNT
        and args.seed == FORMAL_SEED
        and args.k == FORMAL_K
        and args.ef_search == FORMAL_EF_SEARCH
        and args.iterative_scan == "strict_order"
        and args.max_scan_tuples == FORMAL_MAX_SCAN_TUPLES
        and math.isclose(args.scan_mem_multiplier, FORMAL_SCAN_MEM_MULTIPLIER)
        and args.force_hnsw is True
        and args.guidance_filter_strategy == "safe_guided"
        and args.d3_probe_requests == FORMAL_D3_PROBE_REQUESTS
        and math.isclose(args.d3_min_benefit_per_byte, FORMAL_D3_MIN_BENEFIT_PER_BYTE)
        and args.d3_max_fragment_mb == FORMAL_D3_MAX_FRAGMENT_MB
        and math.isclose(args.d3_page_min_skip_rate, FORMAL_D3_PAGE_MIN_SKIP_RATE)
        and args.cache_mb == FORMAL_CACHE_MB
        and math.isclose(args.absolute_recall_target, FORMAL_RECALL_TARGET)
        and math.isclose(args.recall_delta, FORMAL_RECALL_DELTA)
        and args.table == DEFAULT_TABLE
        and args.index == DEFAULT_INDEX
        and args.candidate_validity_predicate == DEFAULT_CANDIDATE_VALIDITY_PREDICATE
        and args.filters_csv.resolve() == DEFAULT_FILTERS.resolve()
        and args.query_cohort.resolve() == DEFAULT_QUERY_COHORT.resolve()
        and args.query_cohort_manifest.resolve() == DEFAULT_QUERY_COHORT_MANIFEST.resolve()
        and args.truth.resolve() == DEFAULT_TRUTH.resolve()
        and args.truth_manifest.resolve() == DEFAULT_TRUTH_MANIFEST.resolve()
        and args.index_health_manifest.resolve() == DEFAULT_INDEX_HEALTH_MANIFEST.resolve()
    )


def aggregate_source_file_sha256(file_sha256: Mapping[str, str]) -> str:
    """Hash a path-sensitive, deterministic manifest of source-file hashes."""
    digest = hashlib.sha256()
    for relative_path, file_hash in sorted(file_sha256.items()):
        if not relative_path or not valid_sha256(file_hash):
            raise BenchmarkContractError("SQLens source file manifest contains an invalid path or SHA256")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    return digest.hexdigest()


def sqlens_source_tree_provenance(source_dir: Path = SQLENS_SOURCE_DIR) -> dict[str, Any]:
    paths = sorted((*source_dir.glob("*.c"), *source_dir.glob("*.h")))
    if not paths:
        raise BenchmarkContractError(f"SQLens source tree has no C/H files: {source_dir}")
    file_sha256 = {path.name: sha256_file(path) for path in paths}
    try:
        source_root = str(source_dir.relative_to(ROOT))
    except ValueError:
        source_root = str(source_dir.resolve())
    build_header = source_dir / "hnsw.h"
    build_id_match = re.search(
        r'^#define\s+SQLENS_BUILD_ID\s+"([^"]+)"',
        build_header.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if build_id_match is None:
        raise BenchmarkContractError("SQLens source tree does not declare SQLENS_BUILD_ID")
    return {
        "source_root": source_root,
        "file_globs": ["*.c", "*.h"],
        "file_count": len(file_sha256),
        "file_sha256": file_sha256,
        "aggregate_sha256": aggregate_source_file_sha256(file_sha256),
        "declared_build_id": build_id_match.group(1),
        "latest_source_mtime_ns": max(path.stat().st_mtime_ns for path in paths),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def parse_atoms(value: str) -> tuple[str, ...]:
    atoms = tuple(part.strip() for part in value.split("||") if part.strip())
    if not atoms or any(not atom.startswith("sql:") for atom in atoms):
        raise BenchmarkContractError("filter atoms must be nonempty sql: atoms")
    return atoms


def load_filters(path: Path) -> list[FilterSpec]:
    rows = read_csv(path)
    required = {"filter_name", "predicate", "atoms", "count", "actual_pct", "source"}
    if len(rows) != 14 or not rows or not required <= set(rows[0]):
        raise BenchmarkContractError("formal benchmark requires the real fourteen-filter Amazon CSV")
    result = [
        FilterSpec(
            name=row["filter_name"],
            predicate=row["predicate"],
            atoms=parse_atoms(row["atoms"]),
            expected_rows=int(row["count"]),
            actual_pct=float(row["actual_pct"]),
        )
        for row in rows
    ]
    if len({item.name for item in result}) != 14 or any("%" in item.predicate for item in result):
        raise BenchmarkContractError("filters must be the fourteen distinct real predicates, without modulo synthesis")
    return result


def parse_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part) for part in value.split(",") if part.strip())
    if len(ids) != 10 or len(set(ids)) != 10:
        raise BenchmarkContractError("exact truth must provide ten distinct IDs")
    return ids


def load_truth(path: Path, filters: Sequence[FilterSpec], *, expected_query_count: int) -> dict[tuple[str, int], TruthEntry]:
    if expected_query_count <= 0:
        raise ValueError("expected_query_count must be positive")
    rows = read_csv(path)
    required = {
        "filter_name", "query_no", "query_id", "exact_filtered_topk_ids",
        "kth_distance_sq", "tie_tolerance", "strict_closer_count",
        "boundary_tied", "self_excluded", "query_split",
    }
    if not rows or not required <= set(rows[0]):
        raise BenchmarkContractError("exact truth is missing the fixed tie-aware schema")
    wanted = {item.name for item in filters}
    filters_by_name = {item.name: item for item in filters}
    truth: dict[tuple[str, int], TruthEntry] = {}
    query_ids: dict[int, int] = {}
    for row in rows:
        if row.get("method") not in (None, "", "pre_filter_exact") or row["filter_name"] not in wanted:
            continue
        query_no = int(row["query_no"])
        if not 0 <= query_no < expected_query_count:
            raise BenchmarkContractError(
                f"exact truth query_no={query_no} is outside requested q{expected_query_count} grid"
            )
        active_filter = filters_by_name[row["filter_name"]]
        if row.get("predicate") not in (None, "", active_filter.predicate):
            raise BenchmarkContractError(
                f"exact truth predicate differs for filter={active_filter.name}"
            )
        if row.get("candidate_validity_predicate") not in (
            None, "", DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
        ):
            raise BenchmarkContractError("exact truth candidate universe is incompatible")
        if row.get("filtered_rows") not in (None, "") and int(row["filtered_rows"]) != active_filter.expected_rows:
            raise BenchmarkContractError(
                f"exact truth candidate count differs for filter={active_filter.name}"
            )
        key = (row["filter_name"], query_no)
        if key in truth:
            raise BenchmarkContractError(f"duplicate exact truth pair: {key}")
        query_id = int(row["query_id"])
        old = query_ids.setdefault(query_no, query_id)
        if old != query_id:
            raise BenchmarkContractError(f"query_no={query_no} maps to multiple IDs")
        if str(row.get("self_excluded", "true")).lower() != "true":
            raise BenchmarkContractError("exact truth must exclude each query row")
        ids = parse_ids(row["exact_filtered_topk_ids"])
        kth_distance_sq = float(row["kth_distance_sq"])
        tie_tolerance = float(row["tie_tolerance"])
        strict_closer_count = int(row["strict_closer_count"])
        boundary_tied = str(row["boundary_tied"]).strip().lower() == "true"
        query_split = str(row["query_split"]).strip()
        if str(row["self_excluded"]).strip().lower() != "true":
            raise BenchmarkContractError("exact truth must explicitly bind self_excluded=true")
        if query_split not in {"calibration", "final"}:
            raise BenchmarkContractError("exact truth query_split must be calibration or final")
        if len(ids) != FORMAL_K or not math.isfinite(kth_distance_sq) or kth_distance_sq < 0:
            raise BenchmarkContractError("exact truth has an invalid top-k distance boundary")
        if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
            raise BenchmarkContractError("exact truth has an invalid tie tolerance")
        if not 0 <= strict_closer_count <= FORMAL_K:
            raise BenchmarkContractError("exact truth strict_closer_count is outside [0, k]")
        truth[key] = TruthEntry(
            row["filter_name"], query_no, query_id, ids, kth_distance_sq,
            tie_tolerance, strict_closer_count, boundary_tied, query_split,
        )
    expected = {
        (item.name, query_no)
        for item in filters
        for query_no in range(expected_query_count)
    }
    missing = expected - set(truth)
    if (
        missing
        or len(truth) != len(expected)
        or set(query_ids) != set(range(expected_query_count))
        or len(set(query_ids.values())) != expected_query_count
    ):
        raise BenchmarkContractError(
            f"fixed q{expected_query_count} truth grid is incomplete or non-unique; "
            f"missing={len(missing)} observed={len(truth)} expected={len(expected)}"
        )
    for query_no in range(expected_query_count):
        splits = {truth[(item.name, query_no)].query_split for item in filters}
        if len(splits) != 1:
            raise BenchmarkContractError(
                f"exact truth has filter-dependent query_split at query_no={query_no}"
            )
    return truth


def truth_query_count(
    filters: Sequence[FilterSpec], truth: Mapping[tuple[str, int], TruthEntry]
) -> int:
    names = {item.name for item in filters}
    query_nos = {query_no for filter_name, query_no in truth if filter_name in names}
    if not query_nos or query_nos != set(range(max(query_nos) + 1)):
        raise BenchmarkContractError("truth query_no values must be contiguous from zero")
    query_count = len(query_nos)
    expected = {(name, query_no) for name in names for query_no in range(query_count)}
    if set(truth) != expected:
        raise BenchmarkContractError("truth must contain a complete filter-by-query grid")
    return query_count


def _weighted_pick(rng: random.Random, names: Sequence[str], weights: Sequence[float]) -> str:
    return names[rng.choices(range(len(names)), weights=weights, k=1)[0]]


def build_trace(filters: Sequence[FilterSpec], truth: Mapping[tuple[str, int], TruthEntry], *,
                requests: int = FORMAL_REQUESTS, window_size: int = FORMAL_WINDOW, seed: int = FORMAL_SEED,
                hot_reuse_probability: float = 0.78) -> list[Request]:
    """Create a deterministic unique-vector trace with a predicate hot-set shift."""
    if requests <= 0 or window_size <= 0 or requests % window_size:
        raise ValueError("requests must be positive and divisible by window_size")
    if len(filters) != 14:
        raise BenchmarkContractError("trace needs exactly fourteen real filters")
    filter_names = [item.name for item in filters]
    query_count = truth_query_count(filters, truth)
    query_split_by_no = {
        query_no: truth[(filter_names[0], query_no)].query_split
        for query_no in range(query_count)
    }
    final_query_nos = [
        query_no for query_no, query_split in query_split_by_no.items()
        if query_split == "final"
    ]
    if query_count == FORMAL_TRUTH_QUERY_COUNT and requests == FORMAL_REQUESTS:
        final_query_nos = [
            query_no for query_no in final_query_nos
            if query_no >= FORMAL_MEASUREMENT_QUERY_OFFSET
        ]
    if requests > len(final_query_nos):
        raise BenchmarkContractError(
            f"unique-vector trace needs at least one final-split truth query per request: "
            f"requests={requests} final_truth_queries={len(final_query_nos)}"
        )
    rng = random.Random(seed)
    query_order = final_query_nos[:]
    random.Random(seed ^ 0xD3A0A17).shuffle(query_order)
    ranked = filter_names[:]
    rng.shuffle(ranked)
    phase_hot = (set(ranked[:4]), set(ranked[4:8]))
    background = ranked[8:]
    previous: dict[str, int] = {}
    trace: list[Request] = []
    half = requests // 2
    for request_no in range(requests):
        phase_index = 0 if request_no < half else 1
        phase = "steady_hot" if phase_index == 0 else "phase_shift_hot"
        hot = phase_hot[phase_index]
        recent = trace[-1].filter_name if trace else None
        if recent in hot and rng.random() < hot_reuse_probability:
            filter_name = recent
        elif rng.random() < 0.88:
            ordered_hot = [name for name in ranked if name in hot]
            filter_name = _weighted_pick(rng, ordered_hot, [1.0 / (rank + 1) for rank in range(len(ordered_hot))])
        else:
            filter_name = _weighted_pick(
                rng, background, [1.0 / (rank + 1) for rank in range(len(background))]
            )
        query_no = query_order[request_no]
        query_id = truth[(filter_name, query_no)].query_id
        old = previous.get(filter_name)
        trace.append(Request(request_no, phase, request_no // window_size, filter_name, query_no, query_id,
                             None if old is None else request_no - old))
        previous[filter_name] = request_no
    return trace


def trace_contract_summary(trace: Sequence[Request]) -> dict[str, Any]:
    """Describe the unique-vector and predicate-reuse properties of a replay."""
    ordered = sorted(trace, key=lambda request: request.request_no)
    midpoint = len(ordered) // 2
    first = Counter(request.filter_name for request in ordered[:midpoint])
    second = Counter(request.filter_name for request in ordered[midpoint:])
    first_hot = [name for name, _ in first.most_common(4)]
    second_hot = [name for name, _ in second.most_common(4)]
    return {
        "requests": len(ordered),
        "request_numbers_contiguous": [request.request_no for request in ordered] == list(range(len(ordered))),
        "unique_query_vectors": len({request.query_id for request in ordered}),
        "one_request_per_query_vector": len({request.query_id for request in ordered}) == len(ordered),
        "final_query_slice_only": sorted(request.query_no for request in ordered)
        == list(range(FORMAL_MEASUREMENT_QUERY_OFFSET, FORMAL_TRUTH_QUERY_COUNT)),
        "distinct_predicates": len({request.filter_name for request in ordered}),
        "predicate_counts": dict(sorted(Counter(request.filter_name for request in ordered).items())),
        "repeated_predicates": sum(count > 1 for count in Counter(request.filter_name for request in ordered).values()),
        "first_phase_predicate_counts": dict(sorted(first.items())),
        "second_phase_predicate_counts": dict(sorted(second.items())),
        "first_phase_hot_predicates": first_hot,
        "second_phase_hot_predicates": second_hot,
        "hot_sets_disjoint": bool(first_hot and second_hot and set(first_hot).isdisjoint(second_hot)),
        "phase_boundary": midpoint,
        "phase_labels_valid": all(
            request.phase == ("steady_hot" if request.request_no < midpoint else "phase_shift_hot")
            for request in ordered
        ),
    }


def formal_trace_contract_errors(trace: Sequence[Request]) -> list[str]:
    """Fail closed if the q10k workload no longer demonstrates online reuse."""
    if len(trace) != FORMAL_REQUESTS:
        return []
    summary = trace_contract_summary(trace)
    errors: list[str] = []
    if not summary["request_numbers_contiguous"]:
        errors.append("trace_request_numbers_not_contiguous")
    if not summary["one_request_per_query_vector"]:
        errors.append("trace_query_vectors_not_unique")
    if not summary["final_query_slice_only"]:
        errors.append("trace_contains_calibration_or_missing_final_queries")
    if int(summary["distinct_predicates"]) != 14:
        errors.append("trace_does_not_cover_all_fourteen_predicates")
    if int(summary["repeated_predicates"]) != 14:
        errors.append("trace_does_not_repeat_every_predicate")
    if not summary["phase_labels_valid"]:
        errors.append("trace_phase_labels_invalid")
    if not summary["hot_sets_disjoint"]:
        errors.append("trace_hot_set_shift_not_observed")
    return errors


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def block_bootstrap_ci(blocks: Sequence[Sequence[float]], *, statistic: Any,
                       samples: int, seed: int) -> tuple[float, float]:
    """Bootstrap a statistic by resampling complete temporal blocks.

    The online cache state makes adjacent requests dependent.  Formal phase and
    cross-mode summaries therefore use a circular moving-block bootstrap over
    complete paired windows instead of pretending that individual requests are
    IID.  A single block intentionally yields a degenerate interval rather
    than fabricated precision.
    """
    normalized = [tuple(float(value) for value in block) for block in blocks if block]
    if not normalized:
        return (0.0, 0.0)
    observed = [value for block in normalized for value in block]
    if len(normalized) == 1:
        value = float(statistic(observed))
        return (value, value)
    rng = random.Random(seed)
    span = max(1, math.ceil(math.sqrt(len(normalized))))
    estimates: list[float] = []
    for _ in range(samples):
        sampled: list[float] = []
        while len(sampled) < len(observed):
            start = rng.randrange(len(normalized))
            for offset in range(span):
                sampled.extend(normalized[(start + offset) % len(normalized)])
                if len(sampled) >= len(observed):
                    break
        estimates.append(float(statistic(sampled[:len(observed)])))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def cache_is_empty(profile: Mapping[str, Any]) -> bool:
    return all(int(profile.get(key, 0) or 0) == 0 for key in (
        "entries", "resident_entries", "resident_bytes", "composed_guide_entries",
        "composed_exact_entries", "adaptive_cache_entries", "adaptive_bytes",
    ))


def _counter(profile: Mapping[str, Any], key: str) -> int:
    return int(profile.get(key, 0) or 0)


def _counter_delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> int:
    delta = _counter(after, key) - _counter(before, key)
    if delta < 0:
        raise BenchmarkContractError(f"non-monotonic lifecycle counter {key}: {before.get(key)} -> {after.get(key)}")
    return delta


def _float_counter_delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> float:
    old = float(before.get(key, 0.0) or 0.0)
    new = float(after.get(key, 0.0) or 0.0)
    delta = new - old
    if delta < -1e-9:
        raise BenchmarkContractError(f"non-monotonic lifecycle timer {key}: {old} -> {new}")
    return max(0.0, delta)


def audit_fragment_store(session: Session, table: str, namespace: str | None = None) -> dict[str, Any]:
    """Audit one heap and, when supplied, one experiment-arm namespace."""
    session.execute(
        f"SELECT to_regclass(%s), %s::regclass::oid::bigint, "
        f"pg_relation_filenode(%s::regclass)::bigint, "
        f"coalesce((SELECT epoch FROM {FRAGMENT_EPOCH_RELATION} "
        "WHERE heap_oid = %s::regclass::oid), 0)::bigint, "
        f"EXISTS (SELECT 1 FROM {FRAGMENT_EPOCH_RELATION} "
        "WHERE heap_oid = %s::regclass::oid)",
        (FRAGMENT_STORE_RELATION, table, table, table, table),
    )
    store_name, heap_oid, relfilenode, epoch, epoch_present = session.row()
    store_exists = store_name is not None
    records: list[dict[str, Any]] = []
    if store_exists:
        namespace_clause = ""
        params: tuple[Any, ...] = (table,)
        if namespace:
            # Do not use LIKE: a permitted run ID may contain `_`, which is a
            # wildcard there and would make one arm audit another arm's rows.
            namespace_clause = " AND left(store_row.filter_name, length(%s) + 1) = %s || chr(31)"
            params = (table, namespace, namespace)
        session.execute(
            f"SELECT row_to_json(store_row)::text FROM {FRAGMENT_STORE_RELATION} AS store_row "
            "WHERE store_row.heap_oid = %s::regclass::oid"
            f"{namespace_clause} ORDER BY row_to_json(store_row)::text",
            params,
        )
        records = [json.loads(str(row[0])) for row in session.all()]
    epoch_value = int(epoch or 0)
    relfilenode_value = int(relfilenode)
    epoch_matches = sum(int(record.get("build_epoch", -1)) == epoch_value for record in records)
    relfilenode_matches = sum(int(record.get("relfilenode", -1)) == relfilenode_value for record in records)
    proof = {
        "valid": all(
            int(record.get("heap_oid", -1)) == int(heap_oid)
            and int(record.get("build_epoch", -1)) == epoch_value
            and int(record.get("relfilenode", -1)) == relfilenode_value
            for record in records
        ) and bool(epoch_present),
        "heap_oid": int(heap_oid),
        "epoch": epoch_value,
        "epoch_present": bool(epoch_present),
        "relfilenode": relfilenode_value,
        "rows_checked": len(records),
        "rows_epoch_match": epoch_matches,
        "rows_relfilenode_match": relfilenode_matches,
        "fragment_store_namespace": namespace or "",
    }
    return {
        "exists": store_exists,
        "count": len(records),
        "content_sha256": canonical_sha256(records),
        "heap_oid": int(heap_oid),
        "epoch": epoch_value,
        "relfilenode": relfilenode_value,
        "epoch_proof": proof,
        "fragment_store_namespace": namespace or "",
    }


def validate_fragment_store_reset(before: Mapping[str, Any], deleted_count: int,
                                  after: Mapping[str, Any]) -> dict[str, Any]:
    before_count = int(before.get("count", -1))
    after_count = int(after.get("count", -1))
    if before_count < 0 or after_count < 0 or int(deleted_count) != before_count or after_count != 0:
        raise BenchmarkContractError(
            "persistent fragment store reset failed: "
            f"before={before_count} deleted={deleted_count} after={after_count}"
        )
    if int(before.get("heap_oid", -1)) != int(after.get("heap_oid", -2)):
        raise BenchmarkContractError("persistent fragment store reset changed target heap identity")
    if str(before.get("fragment_store_namespace", "")) != str(after.get("fragment_store_namespace", "")):
        raise BenchmarkContractError("persistent fragment store reset changed namespace")
    epoch_proof = dict(after.get("epoch_proof") or {})
    if epoch_proof.get("valid") is not True or int(epoch_proof.get("rows_checked", -1)) != 0:
        raise BenchmarkContractError("persistent fragment store reset has incomplete epoch proof")
    return {
        "valid": True,
        "before": dict(before),
        "deleted": int(deleted_count),
        "after": dict(after),
        "heap_oid": int(after["heap_oid"]),
        "epoch_proof": epoch_proof,
        "prebuilt_fragments": after_count,
        "fragment_store_namespace": str(after.get("fragment_store_namespace", "")),
    }


def clear_fragment_store(session: Session, table: str, namespace: str | None = None) -> dict[str, Any]:
    """Clear one heap/namespace slice while retaining an epoch proof."""
    before = audit_fragment_store(session, table, namespace)
    if before["exists"]:
        namespace_clause = ""
        params: tuple[Any, ...] = (table,)
        if namespace:
            namespace_clause = " AND left(filter_name, length(%s) + 1) = %s || chr(31)"
            params = (table, namespace, namespace)
        session.execute(
            f"DELETE FROM {FRAGMENT_STORE_RELATION} "
            "WHERE heap_oid = %s::regclass::oid"
            f"{namespace_clause} RETURNING heap_oid",
            params,
        )
        deleted = len(session.all())
    else:
        deleted = 0
    after = audit_fragment_store(session, table, namespace)
    return validate_fragment_store_reset(before, deleted, after)


def lifecycle_classification(before: Mapping[str, Any], after: Mapping[str, Any], guidance: Mapping[str, Any], *,
                             admitted: bool, reason: str) -> dict[str, Any]:
    builds = int(guidance.get("fragment_builds", 0) or 0)
    store_hits = int(guidance.get("fragment_store_hits", 0) or 0)
    cache_hits = int(guidance.get("fragment_cache_hits", 0) or 0)
    evicted = int(after.get("evictions", 0) or 0) > int(before.get("evictions", 0) or 0)
    created = admitted and (builds > 0 or int(after.get("entries", 0) or 0) > int(before.get("entries", 0) or 0))
    # `active` only says that the extension accepted the guidance binding.  It
    # is not evidence that a previously materialized fragment was reused.
    reused = admitted and not created and (store_hits > 0 or cache_hits > 0 or bool(guidance.get("composed_guide_hit", False)))
    return {"fragment_created": created, "fragment_reused": reused, "fragment_evicted": evicted,
            "admission_reason": reason, "fragment_builds": builds, "fragment_store_hits": store_hits,
            "fragment_cache_hits": cache_hits}


def tie_aware_result_quality(
    returned_ids: Sequence[int], returned_distances_sq: Sequence[float],
    truth: TruthEntry, *, k: int,
) -> dict[str, Any]:
    """Validate top-k using the exact SQL-valid squared-L2 boundary."""
    if len(returned_ids) != len(returned_distances_sq):
        raise BenchmarkContractError("returned IDs and distances have different cardinalities")
    finite = all(math.isfinite(float(value)) and float(value) >= 0.0 for value in returned_distances_sq)
    unique = len(returned_ids) == len(set(int(value) for value in returned_ids))
    threshold = truth.kth_distance_sq + truth.tie_tolerance
    qualifying = sum(
        math.isfinite(float(distance)) and float(distance) <= threshold
        for distance in returned_distances_sq[:k]
    )
    strict_ids = set(int(value) for value in truth.ids[:truth.strict_closer_count])
    returned_id_set = set(int(value) for value in returned_ids[:k])
    strict_returned_ids = strict_ids & returned_id_set
    strict_missing_ids = sorted(strict_ids - returned_id_set)
    strict_returned = len(strict_returned_ids)
    order_valid = all(
        float(returned_distances_sq[index])
        <= float(returned_distances_sq[index + 1]) + truth.tie_tolerance
        for index in range(max(0, len(returned_distances_sq) - 1))
    )
    denominator = min(k, len(truth.ids))
    boundary_slots = max(0, denominator - truth.strict_closer_count)
    boundary_credit = min(boundary_slots, max(0, qualifying - strict_returned))
    recall = (strict_returned + boundary_credit) / denominator if denominator else 0.0
    return {
        "recall": recall,
        "finite_distances": finite,
        "unique_ids": unique,
        "distance_order_valid": order_valid,
        "strict_closer_returned": strict_returned,
        "strict_closer_required": truth.strict_closer_count,
        "strict_closer_missing_ids": strict_missing_ids,
        "all_strict_closer_returned": not strict_missing_ids,
        "all_returned_within_boundary": qualifying == len(returned_distances_sq[:k]),
        "kth_distance_sq": truth.kth_distance_sq,
        "tie_tolerance": truth.tie_tolerance,
        "boundary_tied": truth.boundary_tied,
        "contract": "exact_strict_prefix_ids_and_distance_boundary_ties_v2",
    }


def audit_result_correctness(session: Session, rows_by_mode: Mapping[str, Sequence[dict[str, Any]]], *,
                             filters_by_name: Mapping[str, FilterSpec], table: str,
                             candidate_validity_predicate: str, k: int,
                             truth: Mapping[tuple[str, int], TruthEntry]) -> dict[str, Any]:
    """Independently recheck returned IDs after the timed replay has finished."""
    started = time.perf_counter()
    returned_by_filter: dict[str, set[int]] = {name: set() for name in filters_by_name}
    for rows in rows_by_mode.values():
        for row in rows:
            filter_name = str(row.get("filter_name", ""))
            if filter_name not in returned_by_filter:
                raise BenchmarkContractError(f"result row references unknown filter: {filter_name}")
            returned_by_filter[filter_name].update(int(value) for value in row.get("returned_ids", []))

    valid_by_filter: dict[str, set[int]] = {}
    sql_checks = 0
    for filter_name, filter_spec in filters_by_name.items():
        returned_ids = sorted(returned_by_filter[filter_name])
        if not returned_ids:
            valid_by_filter[filter_name] = set()
            continue
        session.execute(
            f"SELECT id FROM {table} WHERE id = ANY(%s::bigint[]) "
            f"AND ({filter_spec.predicate}) AND ({candidate_validity_predicate})",
            (returned_ids,),
        )
        valid_by_filter[filter_name] = {int(row[0]) for row in session.all()}
        sql_checks += 1

    per_mode_filter: dict[str, dict[str, dict[str, Any]]] = {}
    total_rows = 0
    correct_rows = 0
    for mode, rows in rows_by_mode.items():
        mode_summary: dict[str, dict[str, Any]] = {}
        for filter_name in filters_by_name:
            filter_rows = [row for row in rows if row.get("filter_name") == filter_name]
            invalid_ids: set[int] = set()
            filter_correct = 0
            for row in filter_rows:
                ids = [int(value) for value in row.get("returned_ids", [])]
                distances_sq = [float(value) for value in row.get("returned_distances_sq", [])]
                unique = len(ids) == len(set(ids))
                cardinality = len(ids) == k and int(row.get("returned", len(ids))) == k
                self_excluded = int(row.get("query_id", -1)) not in ids
                invalid = set(ids) - valid_by_filter[filter_name]
                sql_valid = not invalid
                truth_entry = truth[(filter_name, int(row["query_no"]))]
                quality = tie_aware_result_quality(ids, distances_sq, truth_entry, k=k)
                metric_valid = math.isclose(
                    float(row.get("recall_at_10", -1.0)), float(quality["recall"]),
                    rel_tol=0.0, abs_tol=1e-12,
                )
                exact_topk = bool(
                    quality["all_strict_closer_returned"]
                    and quality["all_returned_within_boundary"]
                )
                semantic_correct = bool(
                    not row.get("error") and unique and cardinality and self_excluded
                    and sql_valid and metric_valid and quality["finite_distances"]
                    and quality["distance_order_valid"]
                )
                row["result_ids_unique"] = unique
                row["result_topk_cardinality_correct"] = cardinality
                row["result_self_excluded"] = self_excluded
                row["result_sql_filter_correct"] = sql_valid
                row["result_tie_aware_metric_correct"] = metric_valid
                row["result_distance_order_correct"] = quality["distance_order_valid"]
                row["result_all_strict_closer_returned"] = quality["all_strict_closer_returned"]
                row["result_all_within_exact_boundary"] = quality["all_returned_within_boundary"]
                row["result_exact_topk"] = exact_topk
                row["result_semantically_correct"] = semantic_correct
                # Keep the legacy field as the SQL/result-semantics gate. ANN quality is
                # reported separately and must not require exact top-k at a 0.90 target.
                row["result_correct"] = semantic_correct
                row["result_invalid_ids"] = sorted(invalid)
                invalid_ids.update(invalid)
                filter_correct += int(semantic_correct)
                total_rows += 1
                correct_rows += int(semantic_correct)
            mode_summary[filter_name] = {
                "requests": len(filter_rows),
                "correct_requests": filter_correct,
                "all_requests_correct": bool(filter_rows) and filter_correct == len(filter_rows),
                "invalid_ids": sorted(invalid_ids),
            }
        per_mode_filter[mode] = mode_summary
    return {
        "method": "post_replay_batched_postgresql_semantic_recheck_plus_separate_tie_aware_ann_quality_audit",
        "included_in_online_latency": False,
        "sql_checks": sql_checks,
        "audit_ms": (time.perf_counter() - started) * 1000.0,
        "rows_checked": total_rows,
        "correct_rows": correct_rows,
        "all_rows_correct": total_rows > 0 and correct_rows == total_rows,
        "per_mode_filter": per_mode_filter,
    }


def quality_summary(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]], trace: Sequence[Request], *,
                    absolute_recall_target: float, k: int) -> dict[str, Any]:
    expected_by_filter = Counter(request.filter_name for request in trace)
    by_mode: dict[str, Any] = {}
    for mode in MODES:
        rows = list(rows_by_mode.get(mode, []))
        ok = [row for row in rows if not row.get("error")]
        filter_quality: dict[str, Any] = {}
        for filter_name, expected in sorted(expected_by_filter.items()):
            selected = [row for row in rows if row.get("filter_name") == filter_name]
            selected_ok = [row for row in selected if not row.get("error")]
            recall_mean = statistics.fmean(float(row.get("recall_at_10", 0.0)) for row in selected_ok) if selected_ok else 0.0
            correct = sum(
                1 for row in selected
                if row.get("result_semantically_correct", row.get("result_correct")) is True
                and int(row.get("returned", -1)) == k
                and len(row.get("returned_ids", [])) == k
            )
            exact_topk = sum(
                1 for row in selected
                if row.get("result_exact_topk") is True
            )
            filter_quality[filter_name] = {
                "expected_requests": expected,
                "observed_requests": len(selected),
                "successful_requests": len(selected_ok),
                "recall_mean": recall_mean,
                "recall_min": min((float(row.get("recall_at_10", 0.0)) for row in selected_ok), default=0.0),
                "absolute_recall_target_met": len(selected_ok) == expected and recall_mean >= absolute_recall_target,
                "correct_requests": correct,
                "correctness_met": len(selected) == expected and correct == expected,
                "exact_topk_requests": exact_topk,
                "exact_topk_rate": exact_topk / expected if expected else 0.0,
            }
        overall_recall = statistics.fmean(float(row.get("recall_at_10", 0.0)) for row in ok) if ok else 0.0
        by_mode[mode] = {
            "requests": len(rows),
            "successful_requests": len(ok),
            "recall_mean": overall_recall,
            "absolute_recall_target_met": len(ok) == len(trace) and overall_recall >= absolute_recall_target,
            "all_filters_recall_target_met": bool(filter_quality) and all(
                item["absolute_recall_target_met"] for item in filter_quality.values()
            ),
            "all_filters_correct": bool(filter_quality) and all(
                item["correctness_met"] for item in filter_quality.values()
            ),
            "filters": filter_quality,
        }
    return {"absolute_recall_target": absolute_recall_target, "by_mode": by_mode}


def adaptive_lifecycle_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted((row for row in rows if not row.get("error")), key=lambda row: int(row["request_no"]))
    probes = [row for row in ordered if row.get("probe_observed")]
    materializations = [row for row in ordered if row.get("materialization_observed")]
    reuses = [
        row for row in ordered
        if row.get("reuse_observed")
        and int(row.get("online_materializations_before_for_filter", 0) or 0) > 0
        and bool(row.get("direct_reuse_signal", False))
        and not row.get("hidden_prebuilt_fragment_reused")
    ]
    first_probe = int(probes[0]["request_no"]) if probes else None
    first_materialization = int(materializations[0]["request_no"]) if materializations else None
    first_reuse = int(reuses[0]["request_no"]) if reuses else None
    sequence_complete = bool(
        first_probe is not None
        and first_materialization is not None
        and first_reuse is not None
        and first_probe <= first_materialization < first_reuse
    )
    timed_materializations = sum(float(row.get("materialization_ms", 0.0) or 0.0) > 0.0 for row in materializations)
    return {
        "probe_count": len(probes),
        "materialization_count": len(materializations),
        "reuse_count_after_online_materialization": len(reuses),
        "first_probe_request": first_probe,
        "first_materialization_request": first_materialization,
        "first_reuse_request": first_reuse,
        "sequence_complete": sequence_complete,
        "materialization_timer_coverage": timed_materializations,
        "all_materializations_timed": bool(materializations) and timed_materializations == len(materializations),
    }


def amortization_summary(rows: Sequence[Mapping[str, Any]],
                         stock_by_request: Mapping[int, Mapping[str, Any]], *,
                         bootstrap_samples: int = 1000, bootstrap_seed: int = 20260719) -> dict[str, Any]:
    paired = [
        row for row in rows
        if not row.get("error")
        and int(row.get("request_no", -1)) in stock_by_request
        and not stock_by_request[int(row["request_no"])].get("error")
    ]
    materialization_costs = [
        float(row.get("materialization_ms", 0.0) or 0.0)
        for row in paired if row.get("materialization_observed")
    ]
    reuse_rows = [
        row for row in paired
        if row.get("reuse_observed")
        and int(row.get("online_materializations_before_for_filter", 0) or 0) > 0
        and bool(row.get("direct_reuse_signal", False))
        and not row.get("hidden_prebuilt_fragment_reused")
    ]
    reuse_savings = [
        float(stock_by_request[int(row["request_no"])] ["e2e_ms"]) - float(row["e2e_ms"])
        for row in reuse_rows
    ]
    cumulative_net_savings = sum(
        float(stock_by_request[int(row["request_no"])] ["e2e_ms"]) - float(row["e2e_ms"])
        for row in paired
    )
    deltas = [
        float(stock_by_request[int(row["request_no"])] ["e2e_ms"]) - float(row["e2e_ms"])
        for row in paired
    ]
    blocks_by_window: dict[int, list[float]] = {}
    for row in paired:
        blocks_by_window.setdefault(int(row.get("window", 0)), []).append(
            float(stock_by_request[int(row["request_no"])] ["e2e_ms"]) - float(row["e2e_ms"])
        )
    blocks = [blocks_by_window[window] for window in sorted(blocks_by_window)]
    mean_low, mean_high = block_bootstrap_ci(
        blocks, statistic=statistics.fmean, samples=bootstrap_samples, seed=bootstrap_seed
    )
    p95_low, p95_high = block_bootstrap_ci(
        blocks, statistic=lambda values: percentile(values, .95),
        samples=bootstrap_samples, seed=bootstrap_seed + 1,
    )
    p99_low, p99_high = block_bootstrap_ci(
        blocks, statistic=lambda values: percentile(values, .99),
        samples=bootstrap_samples, seed=bootstrap_seed + 2,
    )
    return {
        "materialization_events": len(materialization_costs),
        "materialization_cost_ms": sum(materialization_costs),
        "materialization_cost_mean_ms": statistics.fmean(materialization_costs) if materialization_costs else 0.0,
        "reuse_requests": len(reuse_rows),
        "reuse_savings_vs_stock_ms": sum(reuse_savings),
        "reuse_savings_mean_ms": statistics.fmean(reuse_savings) if reuse_savings else 0.0,
        "reuse_positive_savings_requests": sum(value > 0.0 for value in reuse_savings),
        "cumulative_net_savings_vs_stock_ms": cumulative_net_savings,
        "cumulative_break_even_request": break_even_request(rows, stock_by_request),
        "paired_e2e_savings_mean_ms": statistics.fmean(deltas) if deltas else 0.0,
        "paired_e2e_savings_mean_ci95_low_ms": mean_low,
        "paired_e2e_savings_mean_ci95_high_ms": mean_high,
        "paired_e2e_savings_p95_ms": percentile(deltas, .95),
        "paired_e2e_savings_p95_ci95_low_ms": p95_low,
        "paired_e2e_savings_p95_ci95_high_ms": p95_high,
        "paired_e2e_savings_p99_ms": percentile(deltas, .99),
        "paired_e2e_savings_p99_ci95_low_ms": p99_low,
        "paired_e2e_savings_p99_ci95_high_ms": p99_high,
        "paired_ci_method": "paired_window_circular_moving_block_bootstrap",
        "paired_ci_blocks": len(blocks),
        "paired_ci_block_span_windows": max(1, math.ceil(math.sqrt(len(blocks)))) if blocks else 0,
        "accounting_contract": "paired stock e2e minus mode e2e; online materialization is already included in mode e2e and is not subtracted twice",
    }


def summary_for_window(rows: Sequence[Mapping[str, Any]], *, bootstrap_samples: int, bootstrap_seed: int,
                       bootstrap_blocks: Sequence[Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    ok = [row for row in rows if not row.get("error")]
    e2e = [float(row["e2e_ms"]) for row in ok]
    query = [float(row["query_ms"]) for row in ok]
    recalls = [float(row["recall_at_10"]) for row in ok]
    blocks = (
        [[float(row["e2e_ms"]) for row in block if not row.get("error")] for block in bootstrap_blocks]
        if bootstrap_blocks is not None else [[value] for value in e2e]
    )
    blocks = [block for block in blocks if block]
    low, high = block_bootstrap_ci(blocks, statistic=statistics.fmean, samples=bootstrap_samples, seed=bootstrap_seed)
    p95_low, p95_high = block_bootstrap_ci(
        blocks, statistic=lambda values: percentile(values, .95), samples=bootstrap_samples, seed=bootstrap_seed + 1,
    )
    p99_low, p99_high = block_bootstrap_ci(
        blocks, statistic=lambda values: percentile(values, .99), samples=bootstrap_samples, seed=bootstrap_seed + 2,
    )
    checks = sum(float(row.get("guidance_checks", 0) or 0) for row in ok)
    skips = sum(float(row.get("guidance_skips", 0) or 0) for row in ok)
    hits = sum(1 for row in ok if row.get("fragment_reused"))
    lifecycle_fields = {
        "probe": "probe_observed",
        "materialize": "materialization_observed",
        "reuse": "reuse_observed",
        "refine": "refine_observed",
        "evict": "evict_observed",
    }
    return {
        "requests": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
        "e2e_mean_ms": statistics.fmean(e2e) if e2e else 0.0, "e2e_p50_ms": percentile(e2e, .50),
        "e2e_p95_ms": percentile(e2e, .95), "e2e_p99_ms": percentile(e2e, .99),
        "e2e_mean_ci95_low_ms": low, "e2e_mean_ci95_high_ms": high,
        "e2e_p95_ci95_low_ms": p95_low, "e2e_p95_ci95_high_ms": p95_high,
        "e2e_p99_ci95_low_ms": p99_low, "e2e_p99_ci95_high_ms": p99_high,
        "latency_ci_method": "paired_window_circular_moving_block_bootstrap" if bootstrap_blocks is not None else "request_circular_moving_block_bootstrap",
        "latency_ci_blocks": len(blocks),
        "latency_ci_block_span_windows": max(1, math.ceil(math.sqrt(len(blocks)))) if blocks else 0,
        "latency_quantile_method": "nearest_rank",
        "query_mean_ms": statistics.fmean(query) if query else 0.0, "query_p50_ms": percentile(query, .50),
        "query_p95_ms": percentile(query, .95), "query_p99_ms": percentile(query, .99),
        "recall_mean": statistics.fmean(recalls) if recalls else 0.0,
        "cache_hit_rate": hits / len(ok) if ok else 0.0,
        "memory_bytes_end": int(ok[-1].get("cache_resident_bytes_after", 0) or 0) if ok else 0,
        "guidance_skip_rate": skips / checks if checks else 0.0,
        "lifecycle_event_counts": {
            event: sum(1 for row in ok if row.get(field)) for event, field in lifecycle_fields.items()
        },
        "probe_count": sum(1 for row in ok if row.get("probe_observed")),
        "materialization_count": sum(1 for row in ok if row.get("materialization_observed")),
        "materialization_cost_ms": sum(
            float(row.get("materialization_ms", 0.0) or 0.0)
            for row in ok if row.get("materialization_observed")
        ),
        "reuse_count": sum(1 for row in ok if row.get("reuse_observed")),
        "refine_count": sum(1 for row in ok if row.get("refine_observed")),
        "evict_count": sum(1 for row in ok if row.get("evict_observed")),
        "fragment_store_hit_delta": sum(_counter(row, "fragment_store_hit_delta") for row in ok),
        "hidden_prebuilt_reuse_count": sum(1 for row in ok if row.get("hidden_prebuilt_fragment_reused")),
        "lifecycle_paths": dict(Counter(str(row.get("lifecycle_path", "unknown")) for row in ok)),
    }


def break_even_request(rows: Sequence[Mapping[str, Any]], stock_by_request: Mapping[int, Mapping[str, Any]]) -> int | None:
    cumulative_by_request: list[tuple[int, float]] = []
    cumulative = 0.0
    for row in sorted(rows, key=lambda item: int(item["request_no"])):
        stock = stock_by_request.get(int(row["request_no"]))
        if not stock or row.get("error") or stock.get("error"):
            continue
        cumulative += float(stock["e2e_ms"]) - float(row["e2e_ms"])
        cumulative_by_request.append((int(row["request_no"]), cumulative))
    if not cumulative_by_request or cumulative_by_request[-1][1] <= 0.0:
        return None
    # A transient crossing caused by request-level noise is not amortization.
    # Report the first point after which cumulative savings stay nonnegative.
    suffix_minimum = math.inf
    stable_request: int | None = None
    for request_no, value in reversed(cumulative_by_request):
        suffix_minimum = min(suffix_minimum, value)
        if value >= 0.0 and suffix_minimum >= 0.0:
            stable_request = request_no
    if stable_request is not None:
        return stable_request
    return None


def paired_execution_errors(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
                            trace: Sequence[Request]) -> list[str]:
    """Verify the recorded schedule rather than trusting self-reported fields."""
    errors: list[str] = []
    backend_pids: set[int] = set()
    expected_by_request = {request.request_no: paired_request_mode_order(request.request_no) for request in trace}
    for mode in MODES:
        pids = {int(row["backend_pid"]) for row in rows_by_mode.get(mode, []) if row.get("backend_pid") is not None}
        if len(pids) != 1:
            errors.append(f"backend_pid_not_persistent:{mode}")
        backend_pids.update(pids)
        for row in rows_by_mode.get(mode, []):
            request_no = int(row.get("request_no", -1))
            expected_order = expected_by_request.get(request_no)
            if expected_order is None:
                continue
            if row.get("backend_mode") != mode:
                errors.append(f"backend_mode_mismatch:{mode}")
                break
            if tuple(row.get("paired_request_mode_order", ())) != expected_order:
                errors.append(f"paired_order_mismatch:{mode}")
                break
            if int(row.get("paired_request_mode_rank", -1)) != expected_order.index(mode):
                errors.append(f"paired_rank_mismatch:{mode}")
                break
    if backend_pids and len(backend_pids) != len(MODES):
        errors.append("backends_not_mode_isolated")
    return errors


def validate_artifact(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]], trace: Sequence[Request], *,
                      recall_delta: float, provenance: Mapping[str, Any],
                      source: Mapping[str, Any] | None = None,
                      absolute_recall_target: float = 0.90, k: int = 10,
                      formal: bool = True,
                      persisted_reuse_evidence: Mapping[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    expected = {request.request_no for request in trace}
    stock = {int(row["request_no"]): row for row in rows_by_mode.get("stock", [])}
    runtime_identity: dict[str, str] = {}
    source_aggregate_sha256 = ""
    if formal:
        if len(trace) == FORMAL_REQUESTS:
            errors.extend(formal_trace_contract_errors(trace))
            errors.extend(paired_execution_errors(rows_by_mode, trace))
        runtime_identity = {
            "runtime_build_id": str(provenance.get("loaded_vector_sqlens_build_id") or ""),
            "loaded_vector_so_path": str(provenance.get("loaded_vector_so_path") or ""),
            "loaded_vector_so_sha256": str(provenance.get("loaded_vector_so_sha256") or ""),
            "database_index_fingerprint": str(provenance.get("database_index_fingerprint") or ""),
        }
        if (
            not runtime_identity["runtime_build_id"]
            or not Path(runtime_identity["loaded_vector_so_path"]).is_absolute()
            or not runtime_identity["loaded_vector_so_path"].endswith("/vector.so")
            or not valid_sha256(runtime_identity["loaded_vector_so_sha256"])
            or not valid_sha256(runtime_identity["database_index_fingerprint"])
        ):
            errors.append("runtime_provenance_incomplete")
        source_tree = source.get("sqlens_source") if isinstance(source, Mapping) else None
        if not isinstance(source_tree, Mapping):
            errors.append("source_provenance_missing")
        else:
            file_sha256 = source_tree.get("file_sha256")
            source_aggregate_sha256 = str(source_tree.get("aggregate_sha256") or "")
            if not isinstance(file_sha256, Mapping) or not file_sha256:
                errors.append("source_file_manifest_missing")
            else:
                try:
                    recomputed = aggregate_source_file_sha256(
                        {str(path): str(file_hash) for path, file_hash in file_sha256.items()}
                    )
                except BenchmarkContractError:
                    recomputed = ""
                if (
                    not valid_sha256(source_aggregate_sha256)
                    or recomputed != source_aggregate_sha256
                    or int(source_tree.get("file_count", -1)) != len(file_sha256)
                ):
                    errors.append("source_provenance_invalid")
            local_binary = source.get("local_vector_so") if isinstance(source, Mapping) else None
            if (
                not isinstance(local_binary, Mapping)
                or not valid_sha256(local_binary.get("sha256"))
                or local_binary.get("sha256") != runtime_identity["loaded_vector_so_sha256"]
                or source_tree.get("declared_build_id") != runtime_identity["runtime_build_id"]
                or local_binary.get("built_after_source_tree") is not True
            ):
                errors.append("source_runtime_binary_binding_invalid")
    for mode in MODES:
        rows = list(rows_by_mode.get(mode, []))
        observed = {int(row.get("request_no", -1)) for row in rows}
        if observed != expected or len(rows) != len(trace):
            errors.append(f"missing_or_duplicate_windows:{mode}")
        if any(row.get("error") for row in rows):
            errors.append(f"request_errors:{mode}")
        if any(
            row.get("hnsw_scan_profile_required") is not True
            or row.get("hnsw_scan_profile_valid") is not True
            for row in rows
        ):
            errors.append(f"hnsw_scan_profile_failure:{mode}")
        namespaces = {str(row.get("fragment_store_namespace") or "") for row in rows}
        if formal and (len(namespaces) != 1 or "" in namespaces):
            errors.append(f"fragment_store_namespace_invalid:{mode}")
        if formal and any(
            any(str(row.get(field) or "") != expected_value for field, expected_value in runtime_identity.items())
            for row in rows
        ):
            errors.append(f"runtime_identity_mismatch:{mode}")
        if formal and any(
            str(row.get("sqlens_source_aggregate_sha256") or "") != source_aggregate_sha256
            for row in rows
        ):
            errors.append(f"source_binding_mismatch:{mode}")
        if mode != "stock":
            planner_failed = False
            result_equivalence_failed = False
            for row in rows:
                if row.get("planner_proof_required") and not row.get("planner_proof_verified"):
                    planner_failed = True
                stock_row = stock.get(int(row.get("request_no", -1)))
                if stock_row and (
                    row.get("returned_ids") != stock_row.get("returned_ids")
                    or row.get("returned_distances_sq") != stock_row.get("returned_distances_sq")
                ):
                    result_equivalence_failed = True
                if stock_row and float(row.get("recall_at_10", 0.0)) + recall_delta < float(stock_row.get("recall_at_10", 0.0)):
                    errors.append(f"recall_regression:{mode}")
                    break
                if int(row.get("fragment_store_hit_delta", 0) or 0) < 0:
                    errors.append(f"negative_fragment_store_hit_delta:{mode}")
                    break
            if planner_failed:
                errors.append(f"planner_proof_failure:{mode}")
            if result_equivalence_failed:
                errors.append(f"result_equivalence_failure:{mode}")
    adaptive_rows = rows_by_mode.get("adaptive", [])
    all_namespaces = {
        str(row.get("fragment_store_namespace") or "")
        for mode in MODES for row in rows_by_mode.get(mode, [])
    }
    if formal and len(all_namespaces) != len(MODES):
        errors.append("fragment_store_namespaces_not_mode_isolated")
    if adaptive_rows and not bool(adaptive_rows[0].get("adaptive_cache_started_empty")):
        errors.append("preexisting_adaptive_cache")
    if adaptive_rows:
        reset_proof = adaptive_rows[0].get("persistent_fragment_reset_proof")
        if not isinstance(reset_proof, Mapping) or reset_proof.get("valid") is not True:
            errors.append("missing_adaptive_fragment_store_reset_proof")
        elif int(reset_proof.get("prebuilt_fragments", -1)) != 0:
            errors.append("prebuilt_adaptive_fragments")
        if any(row.get("hidden_prebuilt_fragment_reused") for row in adaptive_rows):
            errors.append("hidden_prebuilt_fragment_reuse:adaptive")
        if any(not row.get("online_arm") for row in adaptive_rows):
            errors.append("adaptive_arm_contract_missing")
        if any(
            row.get("materialization_observed")
            and not (row.get("activation_materialization_observed") or row.get("query_materialization_observed"))
            for row in adaptive_rows
        ):
            errors.append("adaptive_materialization_phase_unattributed")
        if formal and any(row.get("truth_query_split") != "final" for row in adaptive_rows):
            errors.append("adaptive_trace_contains_non_final_queries")
        if formal and any(row.get("query_materialization_observed") for row in adaptive_rows):
            errors.append("adaptive_materialization_unexpectedly_occurred_during_search")
        if formal and any(
            row.get("materialization_observed")
            and (
                float(row.get("adaptive_fragment_build_ms_delta", 0.0) or 0.0) <= 0.0
                or float(row.get("materialization_ms", 0.0) or 0.0)
                != float(row.get("adaptive_fragment_build_ms_delta", 0.0) or 0.0)
            )
            for row in adaptive_rows
        ):
            errors.append("adaptive_internal_build_timer_missing_or_inconsistent")
        if any(
            row.get("reuse_observed")
            and (
                not row.get("direct_reuse_signal")
                or int(row.get("online_materializations_before_for_filter", 0) or 0) <= 0
            )
            for row in adaptive_rows
        ):
            errors.append("adaptive_reuse_not_bound_to_prior_same_filter_materialization")
    eager_rows = rows_by_mode.get("eager_prebuilt", [])
    if eager_rows and any(not row.get("explicit_eager_control") for row in eager_rows):
        errors.append("eager_control_contract_missing")
    if eager_rows and any(row.get("materialization_observed") for row in eager_rows):
        errors.append("eager_materialization_leaked_into_timed_request")
    if eager_rows:
        eager_setup = eager_rows[0].get("eager_prebuild_evidence")
        if formal and (
            not isinstance(eager_setup, Mapping)
            or eager_setup.get("setup_outside_timed_requests") is not True
            or eager_setup.get("all_filters_prebuilt") is not True
            or not isinstance(eager_setup.get("common_post_setup_warmup"), Mapping)
            or eager_setup["common_post_setup_warmup"].get("executed_after_eager_prebuild") is not True
            or eager_setup["common_post_setup_warmup"].get("all_mode_backends_executed") is not True
            or eager_setup["common_post_setup_warmup"].get("all_backend_pids_distinct") is not True
            or eager_setup["common_post_setup_warmup"].get("all_backend_d3_state_untouched") is not True
            or int(eager_setup["common_post_setup_warmup"].get("calibration_queries_per_backend", -1))
            != FORMAL_CALIBRATION_QUERY_COUNT
        ):
            errors.append("eager_prebuild_or_common_warmup_evidence_missing")
    if formal:
        if (
            not isinstance(persisted_reuse_evidence, Mapping)
            or persisted_reuse_evidence.get("artifact_valid") is not True
            or persisted_reuse_evidence.get("fresh_backend_distinct") is not True
            or persisted_reuse_evidence.get("cache_started_empty") is not True
            or persisted_reuse_evidence.get("all_materialized_filters_reloaded") is not True
            or persisted_reuse_evidence.get("store_unchanged") is not True
            or int(persisted_reuse_evidence.get("materialized_filter_count", 0)) <= 0
        ):
            errors.append("fresh_backend_persistent_fragment_reuse_evidence_missing")
        quality = quality_summary(
            rows_by_mode, trace, absolute_recall_target=absolute_recall_target, k=k
        )
        for mode, mode_quality in quality["by_mode"].items():
            if not mode_quality["absolute_recall_target_met"]:
                errors.append(f"absolute_recall_target_not_met:{mode}")
            for filter_name, filter_quality in mode_quality["filters"].items():
                if not filter_quality["absolute_recall_target_met"]:
                    errors.append(f"per_filter_recall_target_not_met:{mode}:{filter_name}")
                if not filter_quality["correctness_met"]:
                    errors.append(f"per_filter_correctness_failure:{mode}:{filter_name}")
        lifecycle = adaptive_lifecycle_summary(adaptive_rows)
        if not lifecycle["sequence_complete"]:
            errors.append("adaptive_lifecycle_incomplete:probe_materialize_reuse")
        if not lifecycle["all_materializations_timed"]:
            errors.append("adaptive_materialization_cost_missing")
        if not any(row.get("admission_observed") for row in adaptive_rows):
            errors.append("adaptive_admission_event_missing")
        if not any(
            int(row.get("adaptive_fast_reactivation_hits_delta", 0) or 0) > 0
            for row in adaptive_rows
        ):
            errors.append("adaptive_fast_reactivation_evidence_missing")
        if len(trace) == FORMAL_REQUESTS:
            for phase in ("steady_hot", "phase_shift_hot"):
                phase_rows = [row for row in adaptive_rows if row.get("phase") == phase]
                if not any(row.get("admission_observed") for row in phase_rows):
                    errors.append(f"adaptive_phase_admission_missing:{phase}")
                if not any(row.get("materialization_observed") for row in phase_rows):
                    errors.append(f"adaptive_phase_materialization_missing:{phase}")
                if not any(row.get("reuse_observed") for row in phase_rows):
                    errors.append(f"adaptive_phase_reuse_missing:{phase}")
        amortization = amortization_summary(adaptive_rows, stock)
        if amortization["reuse_savings_vs_stock_ms"] <= 0.0:
            errors.append("adaptive_reuse_savings_not_positive")
        if amortization["cumulative_net_savings_vs_stock_ms"] < 0.0:
            errors.append("adaptive_cumulative_savings_negative")
        if amortization["cumulative_break_even_request"] is None:
            errors.append("adaptive_cumulative_break_even_not_reached")
        if amortization["paired_e2e_savings_mean_ci95_low_ms"] <= 0.0:
            errors.append("adaptive_paired_mean_savings_ci_not_positive")
    return errors


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_path(out: Path) -> Path:
    return out.with_name(out.stem + "_checkpoint.json")


def checkpoint_shard_directory(path: Path) -> Path:
    return path.with_name(path.stem + "_shards")


def checkpoint_exists(path: Path) -> bool:
    """Treat an orphan shard directory as a checkpoint so fresh runs fail closed."""
    return path.exists() or checkpoint_shard_directory(path).exists()


def cleanup_checkpoint(path: Path) -> None:
    """Remove only this runner's manifest and per-run shard directory."""
    path.unlink(missing_ok=True)
    shard_directory = checkpoint_shard_directory(path)
    if shard_directory.is_dir():
        shutil.rmtree(shard_directory)
    elif shard_directory.exists():
        shard_directory.unlink()


def paired_request_mode_order(request_no: int) -> tuple[str, ...]:
    """Rotate first position for every request so no mode owns a fixed time slot."""
    if request_no < 0:
        raise ValueError("request_no must be nonnegative")
    offset = request_no % len(MODES)
    return MODES[offset:] + MODES[:offset]


def validate_independent_mode_sessions(backends: Mapping[str, ModeBackend]) -> None:
    if set(backends) != set(MODES):
        raise BenchmarkContractError("paired execution requires exactly stock, adaptive, and eager backends")
    if len({id(backends[mode].session) for mode in MODES}) != len(MODES):
        raise BenchmarkContractError("each mode requires an independent persistent session/cache")
    pids = [int(backends[mode].backend_pid) for mode in MODES]
    if len(set(pids)) != len(MODES):
        raise BenchmarkContractError("each mode requires a distinct PostgreSQL backend PID")


def _completed_paired_windows(value: Any) -> list[int]:
    if not isinstance(value, list) or any(isinstance(window, bool) for window in value):
        raise BenchmarkContractError("checkpoint completed-paired-window schema is invalid")
    try:
        completed = [int(window) for window in value]
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError("checkpoint completed-paired-window schema is invalid") from exc
    if completed != list(range(len(completed))):
        raise BenchmarkContractError("checkpoint paired windows are not a complete prefix")
    return completed


def validate_checkpoint_rows(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]], completed_windows: Sequence[int],
                             window_size: int) -> None:
    if window_size <= 0:
        raise BenchmarkContractError("checkpoint window size is invalid")
    completed = list(completed_windows)
    if completed != list(range(len(completed))):
        raise BenchmarkContractError("checkpoint paired windows are not a complete prefix")
    for mode in MODES:
        rows = list(rows_by_mode.get(mode, []))
        grouped: dict[int, list[Mapping[str, Any]]] = {}
        try:
            for row in rows:
                grouped.setdefault(int(row["window"]), []).append(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkContractError(f"checkpoint rows are invalid for {mode}") from exc
        if set(grouped) != set(completed):
            raise BenchmarkContractError(f"checkpoint is not a complete paired window set: {mode}")
        for window in completed:
            block = grouped[window]
            try:
                request_numbers = {int(row["request_no"]) for row in block}
            except (KeyError, TypeError, ValueError) as exc:
                raise BenchmarkContractError(f"checkpoint rows are invalid for {mode}/{window}") from exc
            if len(block) != window_size or len(request_numbers) != window_size:
                raise BenchmarkContractError(f"checkpoint has partial paired window: {mode}/{window}")


def validate_checkpoint_window_rows(
    rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]], window: int, window_size: int,
) -> None:
    """Validate one complete paired window without scanning prior windows."""
    if window < 0 or window_size <= 0:
        raise BenchmarkContractError("checkpoint window or window size is invalid")
    if set(rows_by_mode) != set(MODES):
        raise BenchmarkContractError("checkpoint shard requires exactly three mode row blocks")
    for mode in MODES:
        rows = list(rows_by_mode[mode])
        if len(rows) != window_size:
            raise BenchmarkContractError(f"checkpoint shard has partial paired window: {mode}/{window}")
        try:
            observed_windows = {int(row["window"]) for row in rows}
            request_numbers = {int(row["request_no"]) for row in rows}
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkContractError(f"checkpoint shard rows are invalid for {mode}/{window}") from exc
        if observed_windows != {window} or len(request_numbers) != window_size:
            raise BenchmarkContractError(f"checkpoint shard has invalid request/window rows: {mode}/{window}")


def paired_window_fingerprints(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
                               completed_windows: Sequence[int]) -> dict[str, str]:
    return {
        str(window): canonical_sha256({
            mode: sorted((row for row in rows_by_mode.get(mode, []) if int(row["window"]) == window),
                         key=lambda row: int(row["request_no"]))
            for mode in MODES
        })
        for window in completed_windows
    }


def backend_lifecycle_fingerprints(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for mode in MODES:
        rows = sorted(rows_by_mode.get(mode, []), key=lambda row: int(row["request_no"]))
        last = rows[-1] if rows else {}
        fingerprints[mode] = canonical_sha256({
            "request_no": last.get("request_no"),
            "cache_profile_after": last.get("cache_profile_after"),
            "guidance_profile": last.get("guidance_profile"),
            "adaptive_state": last.get("adaptive_state"),
            "fragment_created": last.get("fragment_created"),
            "fragment_reused": last.get("fragment_reused"),
            "fragment_store_hit_delta": last.get("fragment_store_hit_delta"),
            "lifecycle_path": last.get("lifecycle_path"),
        })
    return fingerprints


def _read_checkpoint_manifest(path: Path, run_spec_hash: str) -> tuple[dict[str, Any], list[int], int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BenchmarkContractError(f"cannot read checkpoint: {exc}") from exc
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise BenchmarkContractError("checkpoint schema does not support paired-window recovery evidence")
    if payload.get("run_spec_hash") != run_spec_hash:
        raise BenchmarkContractError("checkpoint run-spec/source/database/index mismatch")
    if payload.get("resume_contract") != checkpoint_resume_contract():
        raise BenchmarkContractError("checkpoint resume contract is incompatible")
    completed = _completed_paired_windows(payload.get("completed_paired_windows", []))
    try:
        window_size = int(payload["window_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkContractError("checkpoint window size is invalid") from exc
    shards = payload.get("shards")
    if not isinstance(shards, Mapping):
        raise BenchmarkContractError("checkpoint shard manifest is invalid")
    if set(shards) != {str(window) for window in completed}:
        raise BenchmarkContractError("checkpoint shard manifest does not cover the complete window prefix")
    return payload, completed, window_size


def load_checkpoint(path: Path, run_spec_hash: str) -> dict[str, Any]:
    """Load and audit all immutable window shards, reconstructing the old API shape."""
    payload, completed, window_size = _read_checkpoint_manifest(path, run_spec_hash)
    shard_directory = checkpoint_shard_directory(path)
    rows_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    paired_fingerprints: dict[str, str] = {}
    lifecycle_fingerprints: dict[str, dict[str, str]] = {}
    for window in completed:
        descriptor = payload["shards"].get(str(window))
        if not isinstance(descriptor, Mapping):
            raise BenchmarkContractError(f"checkpoint shard descriptor is invalid: window={window}")
        shard_name = str(descriptor.get("path", ""))
        if not shard_name or Path(shard_name).name != shard_name or Path(shard_name).suffix != ".json":
            raise BenchmarkContractError(f"checkpoint shard path is invalid: window={window}")
        shard_path = shard_directory / shard_name
        if not shard_path.is_file():
            raise BenchmarkContractError(f"checkpoint shard is missing: window={window}")
        if sha256_file(shard_path) != descriptor.get("sha256"):
            raise BenchmarkContractError(f"checkpoint shard SHA mismatch: window={window}")
        try:
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise BenchmarkContractError(f"cannot read checkpoint shard window={window}: {exc}") from exc
        if (
            shard.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
            or shard.get("run_spec_hash") != run_spec_hash
            or int(shard.get("window", -1)) != window
            or int(shard.get("window_size", -1)) != window_size
        ):
            raise BenchmarkContractError(f"checkpoint shard contract mismatch: window={window}")
        shard_rows = shard.get("rows_by_mode")
        if not isinstance(shard_rows, Mapping):
            raise BenchmarkContractError(f"checkpoint shard rows are invalid: window={window}")
        validate_checkpoint_window_rows(shard_rows, window, window_size)
        paired = paired_window_fingerprints(shard_rows, [window])[str(window)]
        lifecycle = backend_lifecycle_fingerprints(shard_rows)
        if shard.get("paired_window_fingerprint") != paired:
            raise BenchmarkContractError(f"checkpoint shard paired fingerprint mismatch: window={window}")
        if shard.get("backend_lifecycle_fingerprints") != lifecycle:
            raise BenchmarkContractError(f"checkpoint shard lifecycle fingerprint mismatch: window={window}")
        row_counts = descriptor.get("row_counts")
        if not isinstance(row_counts, Mapping):
            raise BenchmarkContractError(f"checkpoint shard row counts are invalid: window={window}")
        if int(descriptor.get("row_count", -1)) != sum(len(shard_rows[mode]) for mode in MODES):
            raise BenchmarkContractError(f"checkpoint shard row count mismatch: window={window}")
        for mode in MODES:
            if int(row_counts.get(mode, -1)) != len(shard_rows[mode]):
                raise BenchmarkContractError(f"checkpoint shard mode row count mismatch: {mode}/{window}")
            rows_by_mode[mode].extend(dict(row) for row in shard_rows[mode])
        paired_fingerprints[str(window)] = paired
        lifecycle_fingerprints[str(window)] = lifecycle
    validate_checkpoint_rows(rows_by_mode, completed, window_size)
    if payload.get("paired_window_fingerprints") != paired_fingerprints:
        raise BenchmarkContractError("checkpoint manifest paired-window fingerprint mismatch")
    if payload.get("backend_lifecycle_fingerprints") != lifecycle_fingerprints:
        raise BenchmarkContractError("checkpoint manifest lifecycle fingerprint mismatch")
    return {
        **payload,
        "rows_by_mode": rows_by_mode,
        "paired_window_fingerprints": paired_fingerprints,
        "backend_lifecycle_fingerprints": lifecycle_fingerprints,
    }


def write_checkpoint(path: Path, run_spec_hash: str, rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
                     completed_paired_windows: Sequence[int], window_size: int) -> None:
    """Atomically append one window shard and rewrite only the small manifest."""
    completed = [int(window) for window in completed_paired_windows]
    if not completed:
        raise BenchmarkContractError("checkpoint requires at least one complete paired window")
    window = completed[-1]
    window_rows = {
        mode: [dict(row) for row in rows_by_mode.get(mode, []) if int(row.get("window", -1)) == window]
        for mode in MODES
    }
    validate_checkpoint_window_rows(window_rows, window, window_size)
    expected_total_rows = len(completed) * window_size
    if any(len(rows_by_mode.get(mode, [])) != expected_total_rows for mode in MODES):
        raise BenchmarkContractError("checkpoint cumulative row counts do not match complete paired windows")

    existing_manifest: dict[str, Any] | None = None
    if path.exists():
        existing_manifest, existing_completed, existing_window_size = _read_checkpoint_manifest(path, run_spec_hash)
        if existing_window_size != window_size or existing_completed != completed[:-1]:
            raise BenchmarkContractError("checkpoint manifest does not precede the new complete window")
    elif checkpoint_shard_directory(path).exists():
        raise BenchmarkContractError("checkpoint shard directory exists without a manifest")
    elif completed != [0]:
        raise BenchmarkContractError("checkpoint manifest is missing before a later window")

    shard_directory = checkpoint_shard_directory(path)
    shard_name = f"window_{window:06d}.json"
    shard_path = shard_directory / shard_name
    if shard_path.exists():
        raise BenchmarkContractError(f"checkpoint shard already exists: window={window}")
    paired = paired_window_fingerprints(window_rows, [window])[str(window)]
    lifecycle = backend_lifecycle_fingerprints(window_rows)
    atomic_json(shard_path, {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_spec_hash": run_spec_hash,
        "window_size": window_size,
        "window": window,
        "rows_by_mode": window_rows,
        "paired_window_fingerprint": paired,
        "backend_lifecycle_fingerprints": lifecycle,
        "resume_contract": checkpoint_resume_contract(),
    })
    descriptor = {
        "path": shard_name,
        "sha256": sha256_file(shard_path),
        "row_count": sum(len(window_rows[mode]) for mode in MODES),
        "row_counts": {mode: len(window_rows[mode]) for mode in MODES},
        "paired_window_fingerprint": paired,
        "backend_lifecycle_fingerprints": lifecycle,
    }
    shards = dict((existing_manifest or {}).get("shards") or {})
    shards[str(window)] = descriptor
    paired_fingerprints = dict((existing_manifest or {}).get("paired_window_fingerprints") or {})
    paired_fingerprints[str(window)] = paired
    lifecycle_fingerprints = dict((existing_manifest or {}).get("backend_lifecycle_fingerprints") or {})
    lifecycle_fingerprints[str(window)] = lifecycle
    atomic_json(path, {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_spec_hash": run_spec_hash,
        "window_size": window_size,
        "completed_paired_windows": completed,
        "shards": shards,
        "paired_window_fingerprints": paired_fingerprints,
        "backend_lifecycle_fingerprints": lifecycle_fingerprints,
        "resume_contract": checkpoint_resume_contract(),
    })


def json_profile(session: Session, sql: str) -> dict[str, Any]:
    session.execute(sql)
    value = session.one()
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def configure(session: Session, args: argparse.Namespace, mode: str) -> None:
    json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
    session.execute(
        "SELECT set_config('hnsw.preferred_index', %s, false)",
        (args.index,),
    )
    session.execute(
        "SELECT set_config('hnsw.fragment_store_namespace', %s, false)",
        (fragment_store_namespace(args, mode),),
    )
    session.one()
    settings = [
        "SET jit = off", f"SET statement_timeout = {int(args.statement_timeout_ms)}",
        f"SET hnsw.ef_search = {int(args.ef_search)}", f"SET hnsw.iterative_scan = {args.iterative_scan}",
        f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}", f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}",
        f"SET hnsw.metadata_cache_max_mb = {int(args.cache_mb)}", "SET hnsw.page_access = off", "SET hnsw.index_page_access = off",
        f"SET hnsw.d3_probe_requests = {int(args.d3_probe_requests)}",
        f"SET hnsw.d3_min_benefit_per_byte = {float(args.d3_min_benefit_per_byte)}",
        f"SET hnsw.d3_max_fragment_mb = {int(args.d3_max_fragment_mb)}",
        f"SET hnsw.d3_page_min_skip_rate = {float(args.d3_page_min_skip_rate)}",
        f"SET hnsw.filter_strategy = {'off' if mode == 'stock' else args.guidance_filter_strategy}",
    ]
    if args.force_hnsw:
        settings.append("SET enable_sort = off")
    for statement in settings:
        session.execute(statement)


def reset_guidance(session: Session) -> None:
    session.execute("SELECT vector_hnsw_guidance_reset()")


def adaptive_cache_empty_gate(session: Session) -> tuple[bool, dict[str, Any]]:
    reset_guidance(session)
    before = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
    session.execute("SELECT vector_hnsw_metadata_cache_reset()")
    after = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
    evidence = {
        "before_reset": before,
        "after_reset": after,
        "before_reset_empty": cache_is_empty(before),
        "after_reset_empty": cache_is_empty(after),
    }
    return bool(evidence["after_reset_empty"]), evidence


def activate(session: Session, index: str, atoms: Sequence[str], kind: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    session.execute("SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)", (index, list(atoms), kind))
    session.one()
    activation_ms = (time.perf_counter() - started) * 1000.0
    return json_profile(session, "SELECT vector_hnsw_guidance_profile()"), activation_ms


def run_search(
    session: Session, table: str, predicate: str, candidate_validity_predicate: str,
    query_id: int, k: int, *, guidance_binding: tuple[str, Sequence[str], str] | None = None,
) -> tuple[list[int], list[float], dict[str, Any], str, float]:
    session.execute("SELECT vector_hnsw_reset_scan_profile()")
    binding = ""
    params: tuple[Any, ...]
    if guidance_binding is not None:
        index, atoms, kind = guidance_binding
        binding = (
            "(SELECT vector_hnsw_guidance_bind(%s::regclass, %s::text[], %s) OFFSET 0) AND "
        )
        params = (query_id, index, list(atoms), kind, query_id)
    else:
        params = (query_id, query_id)
    started = time.perf_counter()
    try:
        session.execute(
            f"SELECT id, embedding <-> (SELECT embedding FROM {table} WHERE id = %s) AS distance "
            f"FROM {table} WHERE {binding}({predicate}) "
            f"AND ({candidate_validity_predicate}) AND id <> %s "
            f"ORDER BY distance LIMIT {int(k)}",
            params,
        )
        result_rows = list(session.all())
        ids = [int(row[0]) for row in result_rows]
        distances_sq = [float(row[1]) * float(row[1]) for row in result_rows]
        error = ""
    except Exception as exc:  # The row stays in the artifact and invalidates it later.
        ids, distances_sq, error = [], [], exc.__class__.__name__
    query_ms = (time.perf_counter() - started) * 1000.0
    profile = json_profile(session, "SELECT vector_hnsw_last_scan_profile()")
    return ids, distances_sq, profile, error, query_ms


def reported_profile_build_id(profiles: Sequence[Mapping[str, Any]]) -> str:
    for profile in profiles:
        for key in ("profile_build_id", "build_id", "sqlens_build_id"):
            if profile.get(key) not in (None, ""):
                return str(profile[key])
    return "unreported"


def fragment_store_namespace(args: argparse.Namespace, mode: str) -> str:
    run_id = str(getattr(args, "fragment_store_run_id", "") or "").strip()
    if not run_id:
        run_id = canonical_sha256({
            "out": str(args.out.resolve()),
            "table": args.table,
            "index": args.index,
            "requests": args.requests,
            "seed": args.seed,
        })[:16]
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", run_id):
        raise BenchmarkContractError("--fragment-store-run-id must match [A-Za-z0-9_.-]{1,64}")
    if mode not in MODES:
        raise BenchmarkContractError(f"unknown D3 mode for fragment namespace: {mode}")
    return f"sqlens_d3_{run_id}_{mode}"


def run_request(session: Session, args: argparse.Namespace, mode: str, request: Request, filter_spec: FilterSpec,
                truth: TruthEntry, provenance: Mapping[str, Any], *, adaptive_started_empty: bool,
                online_materializations_before: int = 0,
                online_materializations_before_for_filter: int = 0,
                previous_filter_name: str | None = None) -> dict[str, Any]:
    cache_before = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
    guidance_before = (
        json_profile(session, "SELECT vector_hnsw_guidance_profile()")
        if mode != "stock" else {}
    )
    activation_ms = 0.0
    guidance: dict[str, Any] = {}
    activation_attempted = False
    guidance_active = False
    reason = "stock_no_fragment_cache" if mode == "stock" else "eager_prebuilt_request_activation"
    if mode == "adaptive":
        # Every request enters the extension's D3 state machine.  The extension,
        # not this runner, decides whether to probe, admit page guidance, refine
        # to Bloom, reject, or reuse a resident fragment.
        activation_attempted = True
        activation_kind = "adaptive"
        reason = "extension_adaptive_state_machine"
    elif mode == "eager_prebuilt":
        activation_attempted = True
        activation_kind = args.eager_kind
    if activation_attempted:
        guidance, activation_ms = activate(session, args.index, filter_spec.atoms, activation_kind)
        guidance_active = bool(guidance.get("active", False))
        if mode == "adaptive":
            reason = f"extension_adaptive_{guidance.get('adaptive_state', 'unknown')}"
    # This profile read is deliberately outside the timer.  It lets us attribute
    # a materialization to activation versus the in-query guidance bind without
    # silently charging an in-query build to the wrong phase.
    cache_after_activation = (
        json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
        if activation_attempted else cache_before
    )
    ids, distances_sq, scan, error, query_ms = run_search(
        session, args.table, filter_spec.predicate, args.candidate_validity_predicate,
        request.query_id, args.k,
        guidance_binding=(args.index, filter_spec.atoms, activation_kind)
        if guidance_active else None,
    )
    cache_after = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
    guidance_after = (
        json_profile(session, "SELECT vector_hnsw_guidance_profile()")
        if mode != "stock" else guidance
    )
    e2e_ms = activation_ms + query_ms
    lifecycle_guidance = dict(guidance)
    lifecycle_guidance.update(guidance_after)
    lifecycle = lifecycle_classification(
        cache_before, cache_after, lifecycle_guidance, admitted=guidance_active, reason=reason
    )
    global_counter_fields = (
        "adaptive_probes", "adaptive_admissions", "adaptive_page_builds", "adaptive_bloom_builds",
        "adaptive_refinements", "adaptive_rejections", "adaptive_evictions",
        "adaptive_fragment_cache_hits", "adaptive_fragment_store_hits", "adaptive_fragment_builds",
        "adaptive_fast_reactivation_hits", "adaptive_event_sequence",
    )
    global_deltas = {
        f"{field}_delta": _counter_delta(cache_before, cache_after, field)
        for field in global_counter_fields
    }
    activation_global_deltas = {
        f"{field}_activation_delta": _counter_delta(cache_before, cache_after_activation, field)
        for field in global_counter_fields
    }
    query_global_deltas = {
        f"{field}_query_delta": _counter_delta(cache_after_activation, cache_after, field)
        for field in global_counter_fields
    }
    build_ms_delta = _float_counter_delta(
        cache_before, cache_after, "adaptive_fragment_build_ms"
    )
    build_ms_activation_delta = _float_counter_delta(
        cache_before, cache_after_activation, "adaptive_fragment_build_ms"
    )
    build_ms_query_delta = _float_counter_delta(
        cache_after_activation, cache_after, "adaptive_fragment_build_ms"
    )
    if mode == "adaptive":
        fragment_cache_hits_delta = global_deltas["adaptive_fragment_cache_hits_delta"]
        fragment_store_hits_delta = global_deltas["adaptive_fragment_store_hits_delta"]
        fragment_builds_delta = global_deltas["adaptive_fragment_builds_delta"]
    elif mode == "eager_prebuilt":
        fragment_cache_hits_delta = _counter(guidance, "fragment_cache_hits")
        fragment_store_hits_delta = _counter(guidance, "fragment_store_hits")
        fragment_builds_delta = _counter(guidance, "fragment_builds")
    else:
        fragment_cache_hits_delta = fragment_store_hits_delta = fragment_builds_delta = 0
    deltas = {
        **global_deltas,
        **activation_global_deltas,
        **query_global_deltas,
        "fragment_cache_hits_delta": fragment_cache_hits_delta,
        "fragment_store_hits_delta": fragment_store_hits_delta,
        "fragment_builds_delta": fragment_builds_delta,
    }
    materialization_observed = bool(
        deltas["fragment_builds_delta"]
        or deltas["adaptive_page_builds_delta"]
        or deltas["adaptive_bloom_builds_delta"]
    )
    activation_materialization_observed = bool(
        activation_global_deltas["adaptive_fragment_builds_activation_delta"]
        or activation_global_deltas["adaptive_page_builds_activation_delta"]
        or activation_global_deltas["adaptive_bloom_builds_activation_delta"]
    )
    query_materialization_observed = bool(
        query_global_deltas["adaptive_fragment_builds_query_delta"]
        or query_global_deltas["adaptive_page_builds_query_delta"]
        or query_global_deltas["adaptive_bloom_builds_query_delta"]
    )
    payload_build_ms = (
        float(lifecycle_guidance.get("last_cache_build_ms", 0.0) or 0.0)
        if materialization_observed else 0.0
    )
    materialization_ms = build_ms_delta if mode == "adaptive" else payload_build_ms
    probe_observed = bool(deltas["adaptive_probes_delta"])
    admission_observed = bool(deltas["adaptive_admissions_delta"])
    refine_observed = bool(deltas["adaptive_refinements_delta"])
    evict_observed = bool(deltas["adaptive_evictions_delta"] or _counter_delta(cache_before, cache_after, "evictions"))
    direct_reuse_signal = bool(
        deltas["fragment_cache_hits_delta"]
        or deltas["fragment_store_hits_delta"]
        or deltas["adaptive_fast_reactivation_hits_delta"]
        or guidance_after.get("composed_guide_hit", False)
    )
    reuse_observed = bool(
        direct_reuse_signal
        and not materialization_observed
        and (mode != "adaptive" or online_materializations_before_for_filter > 0)
    )
    # Keep aggregate cache-hit reporting on the same evidence standard as the
    # formal lifecycle gate: adaptive reuse must refer to a fragment previously
    # materialized for this predicate, not merely any earlier fragment.
    lifecycle["fragment_created"] = materialization_observed
    lifecycle["fragment_reused"] = reuse_observed
    lifecycle["fragment_evicted"] = evict_observed
    hidden_prebuilt_reused = bool(
        mode == "adaptive"
        and deltas["fragment_store_hits_delta"] > 0
        and online_materializations_before == 0
    )
    lifecycle_events = [
        name for name, observed in (
            ("probe", probe_observed), ("admit", admission_observed),
            ("materialize", materialization_observed),
            ("reuse", reuse_observed), ("refine", refine_observed), ("evict", evict_observed),
        ) if observed
    ]
    lifecycle_path = "->".join(lifecycle_events) if lifecycle_events else "none"
    proof = bool(scan.get("planner_proof_succeeded", False)) if guidance_active else True
    hnsw_scan_profile_valid = bool(scan.get("valid", False))
    result_quality = tie_aware_result_quality(ids, distances_sq, truth, k=args.k)
    return {
        "mode": mode, "request_no": request.request_no, "phase": request.phase, "window": request.window,
        "filter_name": request.filter_name, "predicate": filter_spec.predicate, "atoms": list(filter_spec.atoms),
        "query_no": request.query_no, "query_id": request.query_id, "reuse_distance": request.reuse_distance,
        "e2e_ms": e2e_ms, "query_ms": query_ms, "activation_ms": activation_ms,
        "materialization_ms": materialization_ms, "payload_build_ms": payload_build_ms,
        "materialization_activation_wall_ms": activation_ms if activation_materialization_observed else 0.0,
        "adaptive_fragment_build_ms_delta": build_ms_delta,
        "adaptive_fragment_build_ms_activation_delta": build_ms_activation_delta,
        "adaptive_fragment_build_ms_query_delta": build_ms_query_delta,
        "returned": len(ids),
        "returned_ids": ids, "returned_distances_sq": distances_sq,
        "recall_at_10": result_quality["recall"], "tie_aware_quality": result_quality,
        "truth_query_split": truth.query_split,
        "truth_strict_closer_count": truth.strict_closer_count,
        "truth_kth_distance_sq": truth.kth_distance_sq,
        "truth_tie_tolerance": truth.tie_tolerance,
        "strict_closer_correct": result_quality["all_strict_closer_returned"],
        "distance_order_correct": result_quality["distance_order_valid"],
        "distance_boundary_correct": result_quality["all_returned_within_boundary"],
        "error": error,
        "activation_attempted": activation_attempted, "guidance_active": guidance_active,
        "planner_proof_required": guidance_active, "planner_proof_verified": proof,
        "planner_proof_attempted": scan.get("planner_proof_attempted", False),
        "planner_proof_bypass_reason": scan.get("planner_proof_bypass_reason", ""),
        "hnsw_scan_profile_required": True, "hnsw_scan_profile_valid": hnsw_scan_profile_valid,
        "visited": scan.get("visited_tuples", 0), "returned_profile": scan.get("returned_tuples", 0),
        "guidance_checks": scan.get("guidance_checks", 0), "guidance_skips": scan.get("guidance_skips", 0),
        "cache_entries_before": cache_before.get("entries", 0), "cache_entries_after": cache_after.get("entries", 0),
        "cache_fragments_before": cache_before.get("composed_guide_entries", 0), "cache_fragments_after": cache_after.get("composed_guide_entries", 0),
        "cache_resident_bytes_before": cache_before.get("resident_bytes", 0), "cache_resident_bytes_after": cache_after.get("resident_bytes", 0),
        "cache_profile_before": cache_before, "cache_profile_after_activation": cache_after_activation,
        "cache_profile_after": cache_after, "guidance_profile": guidance,
        "adaptive_state": guidance.get("adaptive_state", "not_adaptive"),
        "adaptive_requests": guidance.get("adaptive_requests", 0),
        "adaptive_probes": guidance.get("adaptive_probes", 0),
        "adaptive_admissions": guidance.get("adaptive_admissions", 0),
        "adaptive_refinements": guidance.get("adaptive_refinements", 0),
        "adaptive_rejections": guidance.get("adaptive_rejections", 0),
        "adaptive_score": guidance.get("adaptive_score", 0.0),
        "runtime_build_id": provenance["loaded_vector_sqlens_build_id"],
        "loaded_vector_so_path": provenance["loaded_vector_so_path"],
        "loaded_vector_so_sha256": provenance["loaded_vector_so_sha256"],
        "database_index_fingerprint": provenance["database_index_fingerprint"],
        "sqlens_source_aggregate_sha256": provenance["sqlens_source_aggregate_sha256"],
        "profile_reported_build_id": reported_profile_build_id((scan, guidance, cache_after)),
        "adaptive_cache_started_empty": adaptive_started_empty,
        "online_arm": mode == "adaptive", "explicit_eager_control": mode == "eager_prebuilt",
        "guidance_profile_before": guidance_before, "guidance_profile_after": guidance_after,
        **deltas,
        "probe_observed": probe_observed, "admission_observed": admission_observed,
        "materialization_observed": materialization_observed,
        "reuse_observed": reuse_observed, "refine_observed": refine_observed,
        "evict_observed": evict_observed, "lifecycle_events": lifecycle_events,
        "lifecycle_path": lifecycle_path,
        "fragment_store_hit_delta": deltas["fragment_store_hits_delta"],
        "online_materializations_before": online_materializations_before,
        "online_materializations_before_for_filter": online_materializations_before_for_filter,
        "activation_materialization_observed": activation_materialization_observed,
        "query_materialization_observed": query_materialization_observed,
        "direct_reuse_signal": direct_reuse_signal,
        "hidden_prebuilt_fragment_reused": hidden_prebuilt_reused,
        "previous_filter_name": previous_filter_name or "",
        "fragment_store_namespace": fragment_store_namespace(args, mode),
        **lifecycle,
    }


def run_paired_window(backends: Mapping[str, ModeBackend], args: argparse.Namespace, trace: Sequence[Request],
                      filters_by_name: Mapping[str, FilterSpec], truth: Mapping[tuple[str, int], TruthEntry],
                      provenance: Mapping[str, Any], *, window: int,
                      adaptive_started_empty: bool,
                      adaptive_lifecycle_state: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Run one complete trace window once per mode on isolated persistent backends."""
    validate_independent_mode_sessions(backends)
    window_trace = [request for request in trace if request.window == window]
    if len(window_trace) != args.window_size:
        raise BenchmarkContractError(f"trace does not contain one full paired window: {window}")
    if adaptive_lifecycle_state is None:
        adaptive_lifecycle_state = {"online_materializations": 0, "online_materializations_by_filter": {}}
    blocks: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    previous_filter_by_mode = adaptive_lifecycle_state.setdefault(
        "previous_filter_by_mode", {mode: None for mode in MODES}
    )
    materializations_by_filter = adaptive_lifecycle_state.setdefault("online_materializations_by_filter", {})
    for request in window_trace:
        mode_order = paired_request_mode_order(request.request_no)
        for rank, mode in enumerate(mode_order):
            backend = backends[mode]
            mode_provenance = {
                **backend.database,
                "sqlens_source_aggregate_sha256": provenance["sqlens_source_aggregate_sha256"],
            }
            row = run_request(
                backend.session, args, mode, request, filters_by_name[request.filter_name],
                truth[(request.filter_name, request.query_no)], mode_provenance,
                adaptive_started_empty=adaptive_started_empty,
                online_materializations_before=(
                    int(adaptive_lifecycle_state.get("online_materializations", 0))
                    if mode == "adaptive" else 0
                ),
                online_materializations_before_for_filter=(
                    int(materializations_by_filter.get(request.filter_name, 0))
                    if mode == "adaptive" else 0
                ),
                previous_filter_name=previous_filter_by_mode[mode],
            )
            previous_filter_by_mode[mode] = request.filter_name
            if mode == "adaptive":
                builds = int(row.get("fragment_builds_delta", 0) or 0)
                if builds <= 0 and row.get("materialization_observed"):
                    builds = 1
                adaptive_lifecycle_state["online_materializations"] = (
                    int(adaptive_lifecycle_state.get("online_materializations", 0)) + builds
                )
                materializations_by_filter[request.filter_name] = (
                    int(materializations_by_filter.get(request.filter_name, 0)) + builds
                )
                row["online_materializations_after"] = adaptive_lifecycle_state["online_materializations"]
                row["online_materializations_after_for_filter"] = materializations_by_filter[request.filter_name]
            row["backend_mode"] = mode
            row["backend_pid"] = backend.backend_pid
            row["paired_request_mode_order"] = list(mode_order)
            row["paired_request_mode_rank"] = rank
            row["measurement_schedule"] = PAIRING_SCHEDULE
            blocks[mode].append(row)
    if set(blocks) != set(MODES):
        raise BenchmarkContractError(f"window did not execute every mode: {window}")
    return blocks


def eager_prebuild(session: Session, args: argparse.Namespace, filters: Sequence[FilterSpec]) -> dict[str, Any]:
    """This is intentionally outside timed requests and only used for the eager control."""
    reset_guidance(session)
    session.execute("SELECT vector_hnsw_metadata_cache_reset()")
    total_ms = 0.0
    activated_filters: list[str] = []
    for item in filters:
        reset_guidance(session)
        profile, activation_ms = activate(session, args.index, item.atoms, args.eager_kind)
        if not bool(profile.get("active", False)):
            raise BenchmarkContractError(
                f"eager {args.eager_kind} prebuild did not activate filter {item.name}"
            )
        total_ms += activation_ms
        activated_filters.append(item.name)
    reset_guidance(session)
    cache_profile = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
    if cache_is_empty(cache_profile):
        raise BenchmarkContractError("eager prebuild left no resident metadata fragments")
    return {
        "eager_prebuild_ms": total_ms,
        "cache_profile": cache_profile,
        "prebuilt_filters": activated_filters,
        "all_filters_prebuilt": set(activated_filters) == {item.name for item in filters},
        "setup_outside_timed_requests": True,
    }


def common_post_setup_warmup(
    backends: Mapping[str, ModeBackend], args: argparse.Namespace,
    filters: Sequence[FilterSpec], truth: Mapping[tuple[str, int], TruthEntry],
) -> dict[str, Any]:
    """Warm every persistent mode backend after eager setup without changing D3 state."""
    started = time.perf_counter()
    calibration_query_nos = [
        query_no
        for query_no in range(truth_query_count(filters, truth))
        if truth[(filters[0].name, query_no)].query_split == "calibration"
    ]
    if formal_protocol(args) and calibration_query_nos != list(range(FORMAL_CALIBRATION_QUERY_COUNT)):
        raise BenchmarkContractError("formal common warmup requires the preregistered q0..q99 calibration split")
    validate_independent_mode_sessions(backends)
    per_backend: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    state_fields = (
        "entries", "resident_entries", "resident_bytes", "composed_guide_entries",
        "adaptive_probes", "adaptive_admissions", "adaptive_fragment_builds",
        "adaptive_fragment_store_hits", "adaptive_fast_reactivation_hits",
    )
    for mode in MODES:
        backend = backends[mode]
        session = backend.session
        reset_guidance(session)
        cache_before = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
        prefetched: dict[str, int] = {}
        for relation_name, relation in (("heap", args.table), ("hnsw", args.index)):
            session.execute("SELECT pg_prewarm(%s::regclass, 'prefetch')", (relation,))
            prefetched[relation_name] = int(session.one())
        session.execute("SET hnsw.filter_strategy = off")
        returned_topk = 0
        mode_started = time.perf_counter()
        for position, query_no in enumerate(calibration_query_nos):
            filter_spec = filters[position % len(filters)]
            truth_entry = truth[(filter_spec.name, query_no)]
            ids, distances_sq, profile, error, _ = run_search(
                session, args.table, filter_spec.predicate,
                args.candidate_validity_predicate, truth_entry.query_id, args.k,
            )
            quality = tie_aware_result_quality(ids, distances_sq, truth_entry, k=args.k)
            if error or len(ids) != args.k or profile.get("valid") is not True:
                errors.append({
                    "mode": mode, "query_no": query_no,
                    "filter_name": filter_spec.name, "error": error,
                    "returned": len(ids), "profile_valid": profile.get("valid"),
                })
            returned_topk += int(len(ids) == args.k)
            if not quality["finite_distances"] or not quality["distance_order_valid"]:
                errors.append({
                    "mode": mode, "query_no": query_no,
                    "filter_name": filter_spec.name,
                    "error": "invalid_warmup_distances",
                })
        configure(session, args, mode)
        reset_guidance(session)
        cache_after = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
        state_before = {field: cache_before.get(field) for field in state_fields}
        state_after = {field: cache_after.get(field) for field in state_fields}
        per_backend[mode] = {
            "backend_pid": backend.backend_pid,
            "calibration_queries": len(calibration_query_nos),
            "returned_topk_queries": returned_topk,
            "prefetched_blocks": prefetched,
            "cache_state_before": state_before,
            "cache_state_after": state_after,
            "d3_state_untouched": state_before == state_after,
            "elapsed_ms": (time.perf_counter() - mode_started) * 1000.0,
        }
    if errors:
        raise BenchmarkContractError(f"common post-setup warmup failed: {errors[:5]}")
    unique_pids = {int(item["backend_pid"]) for item in per_backend.values()}
    return {
        "method": "per_backend_pg_prewarm_then_stock_calibration_queries",
        "executed_after_eager_prebuild": True,
        "all_mode_backends_executed": set(per_backend) == set(MODES),
        "all_backend_pids_distinct": len(unique_pids) == len(MODES),
        "all_backend_d3_state_untouched": all(
            item["d3_state_untouched"] for item in per_backend.values()
        ),
        "calibration_only": True,
        "calibration_queries_per_backend": len(calibration_query_nos),
        "total_calibration_queries": len(calibration_query_nos) * len(per_backend),
        "per_backend": per_backend,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def database_provenance(session: Session, table: str, index: str) -> dict[str, Any]:
    session.execute(
        "WITH lib AS ("
        "SELECT setting || '/vector.so' AS path "
        "FROM pg_config WHERE name = 'PKGLIBDIR'"
        ") SELECT vector_sqlens_build_id(), path, "
        "encode(sha256(pg_read_binary_file(path)), 'hex') FROM lib"
    )
    runtime_build_id, vector_so_path, vector_so_sha256 = session.row()
    runtime_build_id = str(runtime_build_id or "")
    vector_so_path = str(vector_so_path or "")
    vector_so_sha256 = str(vector_so_sha256 or "")
    if (
        not runtime_build_id
        or not Path(vector_so_path).is_absolute()
        or not vector_so_path.endswith("/vector.so")
        or not valid_sha256(vector_so_sha256)
    ):
        raise BenchmarkContractError("loaded SQLens runtime build ID/vector.so server provenance is invalid")
    session.execute(
        "SELECT current_setting('server_version'), "
        "coalesce((SELECT extversion FROM pg_extension WHERE extname = 'vector'), 'missing'), "
        "%s::regclass::oid::bigint, pg_relation_filenode(%s::regclass)::bigint, "
        "%s::regclass::oid::bigint, pg_relation_filenode(%s::regclass)::bigint, pg_get_indexdef(%s::regclass), "
        "(SELECT pg_get_expr(i.indpred, i.indrelid) FROM pg_index AS i WHERE i.indexrelid = %s::regclass)",
        (table, table, index, index, index, index),
    )
    server, extension, table_oid, table_node, index_oid, index_node, indexdef, index_predicate = session.row()
    database_identity = {
        "server_version": str(server), "vector_extension": str(extension),
        "table": table, "table_oid": int(table_oid), "table_relfilenode": int(table_node),
        "index": index, "index_oid": int(index_oid), "index_relfilenode": int(index_node),
        "indexdef": str(indexdef), "index_predicate": index_predicate,
    }
    return {
        **database_identity,
        "loaded_vector_sqlens_build_id": runtime_build_id,
        "loaded_vector_so_path": vector_so_path,
        "loaded_vector_so_sha256": vector_so_sha256,
        "database_index_fingerprint": canonical_sha256(database_identity),
    }


def normalized_predicate(value: object) -> str:
    """Normalize the simple partial-index predicate used by the formal cohort."""
    return "".join(str(value or "").split()).strip("()")


def validate_database_contract(
    backends: Mapping[str, ModeBackend],
    args: argparse.Namespace,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate runtime/index identity without changing any database state."""
    validate_independent_mode_sessions(backends)
    databases = {mode: backends[mode].database for mode in MODES}
    runtime_identities = {
        (
            database.get("loaded_vector_sqlens_build_id"),
            database.get("loaded_vector_so_path"),
            database.get("loaded_vector_so_sha256"),
        )
        for database in databases.values()
    }
    if len(runtime_identities) != 1:
        raise BenchmarkContractError("mode backends do not observe the same loaded SQLens runtime")
    database_fingerprints = {
        database.get("database_index_fingerprint") for database in databases.values()
    }
    if len(database_fingerprints) != 1:
        raise BenchmarkContractError("mode backends do not observe the same database/index fingerprint")
    database = databases["stock"]
    if database["loaded_vector_so_sha256"] != source["local_vector_so"]["sha256"]:
        raise BenchmarkContractError(
            "server-loaded vector.so does not match the locally built SQLens vector.so"
        )
    if database["loaded_vector_sqlens_build_id"] != source["sqlens_source"]["declared_build_id"]:
        raise BenchmarkContractError(
            "server-loaded SQLens build ID does not match the current source declaration"
        )
    if database["table"] != args.table or database["index"] != args.index:
        raise BenchmarkContractError("database provenance is not bound to the requested table/index")
    if "using hnsw" not in database["indexdef"].lower():
        raise BenchmarkContractError("target index is not an HNSW index")
    observed_predicate = normalized_predicate(database.get("index_predicate"))
    expected_predicate = normalized_predicate(args.candidate_validity_predicate)
    if observed_predicate != expected_predicate:
        raise BenchmarkContractError(
            "target HNSW partial-index predicate does not match the exact GT candidate universe: "
            f"index={database.get('index_predicate')!r} expected={args.candidate_validity_predicate!r}"
        )
    if formal_protocol(args):
        health = source.get("index_query_health") or {}
        identity = health.get("index_identity") or {}
        if (
            health.get("artifact_valid") is not True
            or int(identity.get("oid", -1)) != int(database["index_oid"])
            or int(identity.get("relfilenode", -1)) != int(database["index_relfilenode"])
            or str(identity.get("definition", "")) != str(database["indexdef"])
        ):
            raise BenchmarkContractError(
                "server-selected HNSW index identity differs from the q10k health artifact"
            )
    return {
        "database": database,
        "backend_sessions": {
            mode: {
                "backend_pid": backends[mode].backend_pid,
                "runtime_build_id": backends[mode].database["loaded_vector_sqlens_build_id"],
                "loaded_vector_so_path": backends[mode].database["loaded_vector_so_path"],
                "loaded_vector_so_sha256": backends[mode].database["loaded_vector_so_sha256"],
                "database_index_fingerprint": backends[mode].database["database_index_fingerprint"],
            }
            for mode in MODES
        },
        "three_independent_backend_identities": True,
        "runtime_identity_consistent": True,
        "database_index_identity_consistent": True,
        "partial_index_predicate": database.get("index_predicate"),
        "partial_index_predicate_matches_candidate_universe": True,
    }


def open_mode_backends(psycopg: Any, conninfo: str, *, table: str, index: str) -> dict[str, ModeBackend]:
    """Open three distinct, long-lived sessions before any timed request begins."""
    backends: dict[str, ModeBackend] = {}
    try:
        for mode in MODES:
            connection = psycopg.connect(conninfo, autocommit=True)
            try:
                session: Session = CursorSession(connection.cursor())
                database = database_provenance(session, table, index)
                session.execute("SELECT pg_backend_pid()")
                backends[mode] = ModeBackend(mode, connection, session, int(session.one()), database)
            except Exception:
                connection.close()
                raise
        validate_independent_mode_sessions(backends)
        runtime_identities = {
            (
                backend.database["loaded_vector_sqlens_build_id"],
                backend.database["loaded_vector_so_path"],
                backend.database["loaded_vector_so_sha256"],
            )
            for backend in backends.values()
        }
        if len(runtime_identities) != 1:
            raise BenchmarkContractError("mode backends do not observe the same loaded SQLens runtime")
        database_fingerprints = {
            backend.database["database_index_fingerprint"] for backend in backends.values()
        }
        if len(database_fingerprints) != 1:
            raise BenchmarkContractError("mode backends do not observe the same database/index fingerprint")
        return backends
    except Exception:
        close_mode_backends(backends)
        raise


def close_mode_backends(backends: Mapping[str, ModeBackend]) -> None:
    for mode in reversed(MODES):
        backend = backends.get(mode)
        if backend is None:
            continue
        try:
            backend.connection.close()
        except Exception:
            pass


def materialized_filter_kinds(
    adaptive_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Select the last persisted page/Bloom materialization for each predicate."""
    selected: dict[str, dict[str, Any]] = {}
    unresolved: set[str] = set()
    for row in adaptive_rows:
        if not row.get("materialization_observed"):
            continue
        filter_name = str(row.get("filter_name") or "")
        profiles = (
            row.get("guidance_profile_after"),
            row.get("guidance_profile"),
        )
        kind = next((
            str(profile.get("kind"))
            for profile in profiles
            if isinstance(profile, Mapping) and str(profile.get("kind")) in {"page", "bloom"}
        ), "")
        if not filter_name or not kind:
            unresolved.add(filter_name or "<missing_filter_name>")
            continue
        unresolved.discard(filter_name)
        selected[filter_name] = {
            "kind": kind,
            "request_no": int(row.get("request_no", -1)),
            "query_no": int(row.get("query_no", -1)),
            "query_id": int(row.get("query_id", -1)),
            "expected_returned_ids": list(row.get("returned_ids", [])),
            "expected_returned_distances_sq": list(row.get("returned_distances_sq", [])),
            "expected_recall_at_10": float(row.get("recall_at_10", 0.0)),
        }
    return selected, sorted(unresolved)


def audit_persisted_fragment_reload(
    psycopg_module: Any,
    conninfo: str,
    args: argparse.Namespace,
    *,
    adaptive_rows: Sequence[Mapping[str, Any]],
    filters_by_name: Mapping[str, FilterSpec],
    truth: Mapping[tuple[str, int], TruthEntry],
    existing_backend_pids: Sequence[int],
    expected_database: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that online fragments survive and are reused by a fresh backend."""
    selected, unresolved = materialized_filter_kinds(adaptive_rows)
    namespace = fragment_store_namespace(args, "adaptive")
    evidence: dict[str, Any] = {
        "contract": "fresh_backend_persistent_fragment_reload_v1",
        "fragment_store_namespace": namespace,
        "materialized_filters": sorted(selected),
        "materialized_filter_count": len(selected),
        "unresolved_materialization_filters": unresolved,
        "fresh_backend_pid": None,
        "fresh_backend_distinct": False,
        "cache_started_empty": False,
        "all_materialized_filters_reloaded": False,
        "store_unchanged": False,
        "per_filter": [],
        "artifact_valid": False,
    }
    connection: Any | None = None
    try:
        if not selected or unresolved:
            raise BenchmarkContractError(
                "online replay did not expose a persisted page/Bloom kind for every materialized predicate"
            )
        connection = psycopg_module.connect(conninfo, autocommit=True)
        session: Session = CursorSession(connection.cursor())
        database = database_provenance(session, args.table, args.index)
        if database.get("database_index_fingerprint") != expected_database.get("database_index_fingerprint"):
            raise BenchmarkContractError("fresh backend observes a different table/index identity")
        if database.get("loaded_vector_so_sha256") != expected_database.get("loaded_vector_so_sha256"):
            raise BenchmarkContractError("fresh backend observes a different SQLens binary")
        session.execute("SELECT pg_backend_pid()")
        fresh_pid = int(session.one())
        evidence["fresh_backend_pid"] = fresh_pid
        evidence["fresh_backend_distinct"] = fresh_pid not in {int(value) for value in existing_backend_pids}
        if not evidence["fresh_backend_distinct"]:
            raise BenchmarkContractError("persistent reload control did not open a fresh PostgreSQL backend")

        configure(session, args, "adaptive")
        reset_guidance(session)
        session.execute("SELECT vector_hnsw_metadata_cache_reset()")
        cache_before = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
        evidence["cache_profile_before"] = cache_before
        evidence["cache_started_empty"] = cache_is_empty(cache_before)
        if not evidence["cache_started_empty"]:
            raise BenchmarkContractError("fresh backend metadata cache is not empty")

        store_before = audit_fragment_store(session, args.table, namespace)
        evidence["fragment_store_before"] = store_before
        if store_before.get("exists") is not True or int(store_before.get("count", 0)) < len(selected):
            raise BenchmarkContractError("persistent fragment store does not cover the materialized predicates")
        if (store_before.get("epoch_proof") or {}).get("valid") is not True:
            raise BenchmarkContractError("persistent fragment store epoch proof is invalid")

        per_filter: list[dict[str, Any]] = []
        for filter_name, descriptor in selected.items():
            filter_spec = filters_by_name[filter_name]
            query_no = int(descriptor["query_no"])
            truth_entry = truth[(filter_name, query_no)]
            kind = str(descriptor["kind"])
            reset_guidance(session)
            cache_pre_activation = json_profile(
                session, "SELECT vector_hnsw_metadata_cache_profile()"
            )
            guidance, activation_ms = activate(session, args.index, filter_spec.atoms, kind)
            cache_post_activation = json_profile(
                session, "SELECT vector_hnsw_metadata_cache_profile()"
            )
            ids, distances_sq, scan, error, query_ms = run_search(
                session, args.table, filter_spec.predicate,
                args.candidate_validity_predicate, truth_entry.query_id, args.k,
                guidance_binding=(args.index, filter_spec.atoms, kind),
            )
            quality = tie_aware_result_quality(ids, distances_sq, truth_entry, k=args.k)
            session.execute(
                f"SELECT count(*) FROM {args.table} WHERE id = ANY(%s::bigint[]) "
                f"AND ({filter_spec.predicate}) AND ({args.candidate_validity_predicate})",
                (ids,),
            )
            sql_valid_rows = int(session.one())
            store_hits = int(guidance.get("fragment_store_hits", 0) or 0)
            fragment_builds = int(guidance.get("fragment_builds", 0) or 0)
            expected_ids = [int(value) for value in descriptor["expected_returned_ids"]]
            expected_distances_sq = [
                float(value) for value in descriptor["expected_returned_distances_sq"]
            ]
            result_equivalent = bool(
                ids == expected_ids
                and distances_sq == expected_distances_sq
                and math.isclose(
                    float(quality["recall"]),
                    float(descriptor["expected_recall_at_10"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            row_valid = bool(
                not error
                and guidance.get("active") is True
                and str(guidance.get("kind")) == kind
                and store_hits > 0
                and fragment_builds == 0
                and len(ids) == args.k
                and sql_valid_rows == len(ids)
                and scan.get("valid") is True
                and scan.get("planner_proof_succeeded") is True
                and quality["finite_distances"]
                and quality["unique_ids"]
                and quality["distance_order_valid"]
                and truth_entry.query_id not in ids
                and result_equivalent
            )
            per_filter.append({
                "filter_name": filter_name,
                **descriptor,
                "query_id": truth_entry.query_id,
                "fresh_backend_pid": fresh_pid,
                "activation_ms": activation_ms,
                "query_ms": query_ms,
                "guidance_active": guidance.get("active") is True,
                "guidance_kind": guidance.get("kind"),
                "fragment_store_hits": store_hits,
                "fragment_builds": fragment_builds,
                "cache_entries_before_activation": cache_pre_activation.get("entries", 0),
                "cache_entries_after_activation": cache_post_activation.get("entries", 0),
                "returned": len(ids),
                "sql_valid_rows": sql_valid_rows,
                "scan_profile_valid": scan.get("valid") is True,
                "planner_proof_succeeded": scan.get("planner_proof_succeeded") is True,
                "tie_aware_quality": quality,
                "result_equivalent_to_online_materialization": result_equivalent,
                "error": error,
                "valid": row_valid,
            })

        store_after = audit_fragment_store(session, args.table, namespace)
        evidence["fragment_store_after"] = store_after
        evidence["store_unchanged"] = bool(
            store_before.get("count") == store_after.get("count")
            and store_before.get("content_sha256") == store_after.get("content_sha256")
            and store_before.get("epoch") == store_after.get("epoch")
            and store_before.get("relfilenode") == store_after.get("relfilenode")
        )
        evidence["per_filter"] = per_filter
        evidence["all_materialized_filters_reloaded"] = bool(
            len(per_filter) == len(selected) and all(row["valid"] for row in per_filter)
        )
        evidence["artifact_valid"] = bool(
            evidence["fresh_backend_distinct"]
            and evidence["cache_started_empty"]
            and evidence["all_materialized_filters_reloaded"]
            and evidence["store_unchanged"]
        )
    except Exception as exc:
        evidence["error"] = f"{exc.__class__.__name__}: {exc}"
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
    return evidence


def initialize_mode_backends(backends: Mapping[str, ModeBackend], args: argparse.Namespace,
                             filters: Sequence[FilterSpec],
                             truth: Mapping[tuple[str, int], TruthEntry]) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Configure modes, retain eager's local prebuild, then cold-start online persistently."""
    validate_independent_mode_sessions(backends)
    for mode in MODES:
        configure(backends[mode].session, args, mode)
    reset_guidance(backends["stock"].session)
    backends["stock"].session.execute("SELECT vector_hnsw_metadata_cache_reset()")
    stock_persistent_reset = clear_fragment_store(
        backends["stock"].session, args.table, fragment_store_namespace(args, "stock")
    )
    adaptive_started_empty, adaptive_reset_evidence = adaptive_cache_empty_gate(backends["adaptive"].session)
    if not adaptive_started_empty:
        raise BenchmarkContractError("adaptive cold-start reset did not leave an empty metadata cache")
    eager_namespace = fragment_store_namespace(args, "eager_prebuilt")
    adaptive_namespace = fragment_store_namespace(args, "adaptive")
    eager_persistent_reset = clear_fragment_store(
        backends["eager_prebuilt"].session, args.table, eager_namespace
    )
    persistent_reset = clear_fragment_store(
        backends["adaptive"].session, args.table, adaptive_namespace
    )
    if int(eager_persistent_reset["prebuilt_fragments"]) != 0:
        raise BenchmarkContractError("eager-control fragment-store namespace is not empty before prebuild")
    if int(persistent_reset["prebuilt_fragments"]) != 0:
        raise BenchmarkContractError("online fragment-store namespace is not empty before run")
    if int(stock_persistent_reset["prebuilt_fragments"]) != 0:
        raise BenchmarkContractError("stock fragment-store namespace is not empty before run")
    eager_prebuild_evidence = eager_prebuild(backends["eager_prebuilt"].session, args, filters)
    common_warmup = common_post_setup_warmup(backends, args, filters, truth)
    adaptive_reset_evidence["persistent_fragment_store_reset"] = persistent_reset
    adaptive_reset_evidence["prebuilt_fragments"] = persistent_reset["prebuilt_fragments"]
    adaptive_reset_evidence["fragment_store_namespace"] = adaptive_namespace
    eager_prebuild_evidence["persistent_fragment_store_reset"] = eager_persistent_reset
    eager_prebuild_evidence["fragment_store_namespace"] = eager_namespace
    eager_prebuild_evidence["persistent_store_namespaced_from_adaptive"] = True
    eager_prebuild_evidence["stock_fragment_store_reset"] = stock_persistent_reset
    eager_prebuild_evidence["common_post_setup_warmup"] = common_warmup
    adaptive_reset_evidence["common_post_setup_warmup"] = common_warmup
    return adaptive_started_empty, adaptive_reset_evidence, eager_prebuild_evidence


def load_query_cohort_provenance(
    args: argparse.Namespace,
    exact_manifest: Mapping[str, Any],
    truth: Mapping[tuple[str, int], TruthEntry],
    filters: Sequence[FilterSpec],
) -> dict[str, Any]:
    for path, label in (
        (args.query_cohort, "query cohort CSV"),
        (args.query_cohort_manifest, "query cohort manifest"),
    ):
        if not path.exists():
            raise BenchmarkContractError(f"{label} is required: {path}")
    expected_manifest_name = args.query_cohort.with_name(
        args.query_cohort.stem + "_manifest.json"
    ).name
    if args.query_cohort_manifest.name != expected_manifest_name:
        raise BenchmarkContractError(
            f"query cohort manifest name must match its CSV: expected {expected_manifest_name}"
        )
    with args.query_cohort.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        expected_fields = [
            "query_no", "query_id", "query_split",
            "candidate_validity_predicate", "query_validity_predicate",
        ]
        if list(reader.fieldnames or ()) != expected_fields:
            raise BenchmarkContractError("query cohort does not use the strict dedicated schema")
        rows = list(reader)
    if len(rows) != args.truth_query_count:
        raise BenchmarkContractError(
            f"query cohort row mismatch: expected={args.truth_query_count} observed={len(rows)}"
        )
    cohort_query_ids: list[int] = []
    cohort_splits: list[str] = []
    for query_no, row in enumerate(rows):
        try:
            observed_no = int(row["query_no"])
            query_id = int(row["query_id"])
        except (TypeError, ValueError) as exc:
            raise BenchmarkContractError("query cohort contains a non-integer mapping") from exc
        if observed_no != query_no:
            raise BenchmarkContractError("query cohort query_no values are not contiguous from zero")
        if row["query_split"] not in {"calibration", "final"}:
            raise BenchmarkContractError("query cohort split must be calibration or final")
        if (
            row["candidate_validity_predicate"] != args.candidate_validity_predicate
            or row["query_validity_predicate"] != args.candidate_validity_predicate
        ):
            raise BenchmarkContractError("query cohort validity predicate is incompatible")
        cohort_query_ids.append(query_id)
        cohort_splits.append(row["query_split"])
    if len(set(cohort_query_ids)) != len(cohort_query_ids):
        raise BenchmarkContractError("formal query cohort must contain unique query IDs")

    try:
        cohort_manifest = json.loads(args.query_cohort_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"cannot read query cohort manifest: {exc}") from exc
    cohort_sha256 = sha256_file(args.query_cohort)
    cohort_manifest_sha256 = sha256_file(args.query_cohort_manifest)
    cohort_output = ((cohort_manifest.get("outputs") or {}).get("cohort_csv") or {})
    selection = cohort_manifest.get("selection") or {}
    calibration = selection.get("calibration") or {}
    final = selection.get("final") or {}
    uniqueness = cohort_manifest.get("uniqueness_contract") or {}
    eligible_population = cohort_manifest.get("eligible_query_population") or {}
    if (
        cohort_manifest.get("artifact_valid") is not True
        or cohort_manifest.get("candidate_validity_predicate") != args.candidate_validity_predicate
        or cohort_output.get("sha256") != cohort_sha256
        or int(cohort_output.get("rows", -1)) != args.truth_query_count
        or int(calibration.get("queries", -1)) + int(final.get("queries", -1))
        != args.truth_query_count
        or cohort_manifest.get("schema_version") != 1
        or cohort_manifest.get("method") != "deterministic_unique_projection_fingerprint_cohort_v1"
        or cohort_manifest.get("query_validity_predicate") != args.candidate_validity_predicate
        or selection.get("disjoint") is not True
        or selection.get("query_ids_sha256") != ordered_query_ids_sha256(cohort_query_ids)
        or uniqueness.get("all_rows_fingerprinted") is not True
        or uniqueness.get("duplicate_admission_false_negative_only") is not True
        or int(uniqueness.get("hashes", -1)) != 2
        or int(eligible_population.get("singleton_fingerprint_rows", -1)) < args.truth_query_count
    ):
        raise BenchmarkContractError("query cohort manifest does not bind the selected unique-vector q10k cohort")
    if formal_protocol(args) and (
        int(calibration.get("queries", -1)) != FORMAL_CALIBRATION_QUERY_COUNT
        or int(final.get("queries", -1)) != FORMAL_FINAL_QUERY_COUNT
        or cohort_splits[:FORMAL_CALIBRATION_QUERY_COUNT]
        != ["calibration"] * FORMAL_CALIBRATION_QUERY_COUNT
        or cohort_splits[FORMAL_CALIBRATION_QUERY_COUNT:]
        != ["final"] * FORMAL_FINAL_QUERY_COUNT
    ):
        raise BenchmarkContractError(
            "formal D3 cohort must contain calibration q0..99 followed by exactly 10,100 final queries"
        )

    truth_query_ids: dict[int, int] = {}
    for item in filters:
        for query_no in range(args.truth_query_count):
            query_id = truth[(item.name, query_no)].query_id
            previous = truth_query_ids.setdefault(query_no, query_id)
            if previous != query_id:
                raise BenchmarkContractError("exact truth has filter-dependent query mappings")
            if truth[(item.name, query_no)].query_split != cohort_splits[query_no]:
                raise BenchmarkContractError(
                    "exact truth query split does not match the query cohort"
                )
    if [truth_query_ids[number] for number in range(args.truth_query_count)] != cohort_query_ids:
        raise BenchmarkContractError("exact truth query mapping does not match the query cohort")

    exact_query_source = exact_manifest.get("query_source") or {}
    exact_cohort = exact_query_source.get("cohort_csv") or {}
    exact_cohort_manifest = exact_query_source.get("manifest") or {}
    if (
        exact_query_source.get("kind") != "external_unique_vector_query_cohort"
        or exact_cohort.get("sha256") != cohort_sha256
        or exact_cohort_manifest.get("sha256") != cohort_manifest_sha256
    ):
        raise BenchmarkContractError("exact GT manifest is not bound to the selected query cohort")
    return {
        "csv_path": str(args.query_cohort),
        "csv_sha256": cohort_sha256,
        "manifest_path": str(args.query_cohort_manifest),
        "manifest_sha256": cohort_manifest_sha256,
        "rows": args.truth_query_count,
        "unique_query_ids": len(set(cohort_query_ids)),
        "calibration_queries": int(calibration["queries"]),
        "final_queries": int(final["queries"]),
        "exact_truth_binding_verified": True,
    }


def source_provenance(
    args: argparse.Namespace,
    truth: Mapping[tuple[str, int], TruthEntry],
    filters: Sequence[FilterSpec],
) -> dict[str, Any]:
    truth_manifest = args.truth_manifest
    if not truth_manifest.exists():
        raise BenchmarkContractError("fixed exact GT manifest is required for strict source provenance")
    expected_manifest_name = args.truth.with_name(args.truth.stem + "_manifest.json").name
    if truth_manifest.name != expected_manifest_name:
        raise BenchmarkContractError(
            f"truth manifest name must match its truth CSV: expected {expected_manifest_name}"
        )
    try:
        manifest = json.loads(truth_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"cannot read exact GT manifest: {exc}") from exc
    truth_sha256 = sha256_file(args.truth)
    manifest_truth = ((manifest.get("outputs") or {}).get("truth_csv") or {})
    manifest_predicate = ((manifest.get("validity_contract") or {}).get("candidate_validity_predicate"))
    if manifest.get("artifact_valid") is not True:
        raise BenchmarkContractError("exact GT manifest is not valid")
    if manifest_truth.get("sha256") != truth_sha256:
        raise BenchmarkContractError("exact GT manifest does not bind the selected truth CSV")
    if manifest_predicate != args.candidate_validity_predicate:
        raise BenchmarkContractError(
            "exact GT candidate universe does not match --candidate-validity-predicate"
        )
    calibration_queries = int((manifest.get("calibration") or {}).get("queries", -1))
    final_queries = int((manifest.get("final") or {}).get("queries", -1))
    if (
        calibration_queries + final_queries != args.truth_query_count
        or int(manifest.get("truth_rows", -1)) != len(filters) * args.truth_query_count
    ):
        raise BenchmarkContractError("exact GT manifest does not describe the requested truth grid")
    if formal_protocol(args) and (
        calibration_queries != FORMAL_CALIBRATION_QUERY_COUNT
        or final_queries != FORMAL_FINAL_QUERY_COUNT
    ):
        raise BenchmarkContractError(
            "formal exact GT must bind 100 calibration and 10,100 final queries"
        )
    query_cohort = load_query_cohort_provenance(args, manifest, truth, filters)
    source_tree = sqlens_source_tree_provenance()
    if not SQLENS_LOCAL_SHARED_OBJECT.is_file():
        raise BenchmarkContractError(
            f"locally built SQLens vector.so is missing: {SQLENS_LOCAL_SHARED_OBJECT}"
        )
    local_vector_mtime_ns = SQLENS_LOCAL_SHARED_OBJECT.stat().st_mtime_ns
    built_after_source_tree = local_vector_mtime_ns >= int(source_tree["latest_source_mtime_ns"])
    if not built_after_source_tree:
        raise BenchmarkContractError(
            "local SQLens vector.so predates the recorded C/H source tree; rebuild before execution"
        )
    source = {"script_sha256": sha256_file(Path(__file__)), "filters_sha256": sha256_file(args.filters_csv),
            "truth_sha256": truth_sha256, "truth_manifest_sha256": sha256_file(truth_manifest),
            "truth_manifest_name": truth_manifest.name,
            "candidate_validity_predicate": args.candidate_validity_predicate,
            "truth_manifest_artifact_valid": True,
            "query_cohort": query_cohort,
            "sqlens_source": source_tree,
            "local_vector_so": {
                "path": str(SQLENS_LOCAL_SHARED_OBJECT.relative_to(ROOT)),
                "sha256": sha256_file(SQLENS_LOCAL_SHARED_OBJECT),
                "mtime_ns": local_vector_mtime_ns,
                "built_after_source_tree": built_after_source_tree,
            }}
    if formal_protocol(args):
        observed_hashes = {
            "filters_csv": source["filters_sha256"],
            "query_cohort": query_cohort["csv_sha256"],
            "query_cohort_manifest": query_cohort["manifest_sha256"],
            "truth": source["truth_sha256"],
            "truth_manifest": source["truth_manifest_sha256"],
        }
        if observed_hashes != FORMAL_INPUT_SHA256:
            raise BenchmarkContractError(
                f"formal D3 input hashes differ from the preregistered contract: {observed_hashes}"
            )
        source["formal_input_hash_contract"] = dict(FORMAL_INPUT_SHA256)
        health = load_index_health_provenance(args)
        validate_index_health_binary_binding(health, source)
        source["index_query_health"] = health
    return source


def load_index_health_provenance(args: argparse.Namespace) -> dict[str, Any]:
    path = args.index_health_manifest
    if not path.is_file():
        raise BenchmarkContractError(f"formal q10k HNSW health manifest is required: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"cannot read HNSW health manifest: {exc}") from exc
    qualified_index = args.index if "." in args.index else f"public.{args.index}"
    identities = manifest.get("index_identities") or {}
    summary = ((manifest.get("summary") or {}).get("indexes") or {}).get(qualified_index) or {}
    settings = manifest.get("settings") or {}
    inputs = manifest.get("inputs") or {}
    runtime = manifest.get("runtime") or {}
    identity = identities.get(qualified_index) or {}
    if (
        manifest.get("artifact_contract") != "sqlens_hnsw_query_health_v1"
        or manifest.get("artifact_valid") is not True
        or int(inputs.get("queries", -1)) != FORMAL_REQUESTS
        or inputs.get("split") != "final"
        or int(inputs.get("query_no_start", -1)) != FORMAL_MEASUREMENT_QUERY_OFFSET
        or int(inputs.get("query_no_end_exclusive", -1)) != FORMAL_TRUTH_QUERY_COUNT
        or inputs.get("cohort_csv_sha256") != FORMAL_INPUT_SHA256["query_cohort"]
        or inputs.get("cohort_manifest_sha256") != FORMAL_INPUT_SHA256["query_cohort_manifest"]
        or settings.get("query_shape") != "unfiltered_partial_index_self_excluded_topk"
        or settings.get("query_split") != "final"
        or str(settings.get("table", "")).split(".")[-1]
        != str(args.table).split(".")[-1]
        or qualified_index not in set(settings.get("indexes") or ())
        or int(settings.get("k", -1)) != FORMAL_K
        or int(settings.get("ef_search", -1)) != 1000
        or settings.get("iterative_scan") != "off"
        or settings.get("filter_strategy") != "off"
        or int(summary.get("queries", -1)) != FORMAL_REQUESTS
        or summary.get("valid") is not True
        or summary.get("error_queries")
        or summary.get("exhausted_queries")
        or summary.get("incomplete_topk_queries")
        or summary.get("plan_failure_queries")
        or not identity
        or not valid_sha256(str(runtime.get("vector_so_sha256", "")))
        or not str(runtime.get("sqlens_build_id", ""))
    ):
        raise BenchmarkContractError(
            "formal HNSW health manifest does not prove zero-error, zero-exhaustion q10k health"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "qualified_index": qualified_index,
        "index_identity": dict(identity),
        "summary": dict(summary),
        "settings": dict(settings),
        "runtime": dict(runtime),
        "artifact_valid": True,
    }


def validate_index_health_binary_binding(
    health: Mapping[str, Any], source: Mapping[str, Any]
) -> None:
    runtime = health.get("runtime") or {}
    local_binary = source.get("local_vector_so") or {}
    source_tree = source.get("sqlens_source") or {}
    if (
        runtime.get("vector_so_sha256") != local_binary.get("sha256")
        or runtime.get("sqlens_build_id") != source_tree.get("declared_build_id")
    ):
        raise BenchmarkContractError(
            "formal HNSW health manifest was generated by a different SQLens binary"
        )


def _preflight_check(
    checks: dict[str, Any], errors: list[dict[str, str]], name: str, operation: Any,
    summarize: Any | None = None,
) -> Any:
    """Record one independent check while keeping the JSON report complete."""
    try:
        value = operation()
    except Exception as exc:
        checks[name] = {"ready": False, "error": str(exc)}
        errors.append({"check": name, "error": str(exc)})
        return None
    checks[name] = {"ready": True, "details": summarize(value) if summarize else value}
    return value


def preflight_experiment(
    args: argparse.Namespace, *, psycopg_module: Any | None = None, conninfo: str | None = None,
) -> dict[str, Any]:
    """Read-only formal readiness audit; never touches cache/store or result paths."""
    checks: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    filters = _preflight_check(
        checks, errors, "filters",
        lambda: load_filters(args.filters_csv),
        lambda value: {
            "count": len(value),
            "names": [item.name for item in value],
            "csv_sha256": sha256_file(args.filters_csv),
        },
    )
    truth = None
    if filters is not None:
        truth = _preflight_check(
            checks, errors, "exact_truth",
            lambda: load_truth(args.truth, filters, expected_query_count=args.truth_query_count),
            lambda value: {
                "rows": len(value),
                "query_count": truth_query_count(filters, value),
                "filter_count": len({filter_name for filter_name, _ in value}),
                "truth_csv_sha256": sha256_file(args.truth),
            },
        )
    trace = None
    if filters is not None and truth is not None:
        trace = _preflight_check(
            checks, errors, "trace_contract",
            lambda: _build_formal_preflight_trace(args, filters, truth),
        )
    source = None
    if filters is not None and truth is not None:
        source = _preflight_check(
            checks, errors, "source_provenance",
            lambda: source_provenance(args, truth, filters),
        )

    backends: dict[str, ModeBackend] = {}
    database_result = None
    if psycopg_module is None:
        try:
            import psycopg as psycopg_module  # type: ignore[no-redef]
            from common_pg import pg_config_from_env
            if conninfo is None:
                conninfo = pg_config_from_env().conninfo
        except Exception as exc:
            checks["database"] = {"ready": False, "error": f"execution needs psycopg/common_pg: {exc}"}
            errors.append({"check": "database", "error": checks["database"]["error"]})
    if psycopg_module is not None and conninfo is None:
        checks["database"] = {"ready": False, "error": "PostgreSQL connection info is required for preflight"}
        errors.append({"check": "database", "error": checks["database"]["error"]})
    elif psycopg_module is not None and conninfo is not None:
        def connect_and_validate() -> dict[str, Any]:
            nonlocal backends
            backends = open_mode_backends(
                psycopg_module, conninfo, table=args.table, index=args.index,
            )
            if source is None:
                raise BenchmarkContractError("database identity cannot be matched because source provenance failed")
            return validate_database_contract(backends, args, source)

        try:
            database_result = _preflight_check(
                checks, errors, "database", connect_and_validate,
            )
        finally:
            close_mode_backends(backends)

    return {
        "preflight": True,
        "ready": not errors,
        "errors": errors,
        "checks": checks,
        "database_connected": database_result is not None,
        "files_written": False,
        "timed_requests_executed": False,
        "cache_or_fragment_store_modified": False,
        "formal": formal_protocol(args),
        "requested_contract": {
            "requests": args.requests,
            "window_size": args.window_size,
            "truth_query_count": args.truth_query_count,
            "modes": list(MODES),
            "trace_kind": TRACE_KIND,
            "table": args.table,
            "index": args.index,
            "candidate_validity_predicate": args.candidate_validity_predicate,
        },
    }


def _build_formal_preflight_trace(
    args: argparse.Namespace, filters: Sequence[FilterSpec],
    truth: Mapping[tuple[str, int], TruthEntry],
) -> dict[str, Any]:
    if not formal_protocol(args):
        raise BenchmarkContractError(
            "preflight requires the formal q10k/100-window request contract"
        )
    trace = build_trace(
        filters, truth, requests=args.requests, window_size=args.window_size, seed=args.seed,
    )
    trace_errors = formal_trace_contract_errors(trace)
    if trace_errors:
        raise BenchmarkContractError("formal trace contract failed: " + ", ".join(trace_errors))
    summary = trace_contract_summary(trace)
    return {"summary": summary, "trace_sha256": canonical_sha256([asdict(item) for item in trace])}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def replay_disclosure(args: argparse.Namespace) -> dict[str, Any]:
    formal = formal_protocol(args)
    if formal:
        return {
            "trace_kind": TRACE_KIND,
            "trace_contract": (
                "deterministic 10,000-request PostgreSQL online replay over 10,000 "
                "preregistered unique query vectors; every vector is used exactly once; "
                "predicate reuse and a mid-trace hot-set shift drive D3 adaptation"
            ),
            "workload_manifest_name": FORMAL_WORKLOAD_MANIFEST_NAME,
        }
    return {
        "trace_kind": "deterministic_non_formal_postgresql_online_replay",
        "trace_contract": (
            f"deterministic {args.requests}-request non-formal PostgreSQL online replay "
            f"using distinct vectors from a q{args.truth_query_count} fixed truth grid"
        ),
        "workload_manifest_name": "amazon10m_d3_non_formal_postgresql_online_replay",
    }


def make_run_spec(args: argparse.Namespace, source: Mapping[str, Any], database: Mapping[str, Any], trace: Sequence[Request]) -> dict[str, Any]:
    disclosure = replay_disclosure(args)
    unique_query_vectors = len({item.query_id for item in trace})
    return {"formal": formal_protocol(args), "requests": args.requests,
            "window_size": args.window_size, "seed": args.seed, "phase_boundary": args.requests // 2,
            **disclosure,
            "trace_source": "generated_deterministically_by_this_runner",
            "production_trace": False,
            "postgresql_online_execution": True,
            "fixed_exact_truth_query_count": args.truth_query_count,
            "effective_unique_queries": unique_query_vectors,
            "unique_query_vectors": unique_query_vectors,
            "one_request_per_query_vector": unique_query_vectors == len(trace),
            "database_cracking": False,
            "candidate_validity_predicate": args.candidate_validity_predicate,
            "search_configuration": {
                "k": args.k,
                "ef_search": args.ef_search,
                "iterative_scan": args.iterative_scan,
                "max_scan_tuples": args.max_scan_tuples,
                "scan_mem_multiplier": args.scan_mem_multiplier,
                "force_hnsw": args.force_hnsw,
                "statement_timeout_ms": args.statement_timeout_ms,
            },
            "absolute_recall_target": args.absolute_recall_target,
            "per_filter_recall_gate": True,
            "post_replay_sql_correctness_audit": True,
            "modes": list(MODES),
            "single_client_sequential": True, "measurement_schedule": PAIRING_SCHEDULE,
            "mode_backend_topology": "three_independent_persistent_postgresql_backends",
            "fragment_store_isolation": {
                mode: fragment_store_namespace(args, mode) for mode in MODES
            },
            "paired_request_mode_order": "round_robin rotation by request number",
            "checkpoint_resume_contract": checkpoint_resume_contract(),
            "adaptive_admission_owner": "pgvector_extension",
            "adaptive_kind": "adaptive", "eager_kind": args.eager_kind,
            "persistent_reuse_control": "fresh_postgresql_backend_after_online_replay",
            "d3_probe_requests": args.d3_probe_requests,
            "d3_min_benefit_per_byte": args.d3_min_benefit_per_byte,
            "d3_max_fragment_mb": args.d3_max_fragment_mb,
            "d3_page_min_skip_rate": args.d3_page_min_skip_rate,
            "cache_mb": args.cache_mb, "guidance_filter_strategy": args.guidance_filter_strategy,
            "source": source, "database": database,
            "trace_summary": trace_contract_summary(trace),
            "trace_sha256": canonical_sha256([asdict(item) for item in trace])}


def execute_experiment(args: argparse.Namespace) -> int:
    reject_cross_process_resume(args.resume)
    try:
        import psycopg
        from common_pg import pg_config_from_env
    except ImportError as exc:
        raise BenchmarkContractError("execution needs psycopg and common_pg") from exc
    filters = load_filters(args.filters_csv)
    truth = load_truth(args.truth, filters, expected_query_count=args.truth_query_count)
    trace = build_trace(filters, truth, requests=args.requests, window_size=args.window_size, seed=args.seed)
    if formal_protocol(args):
        trace_errors = formal_trace_contract_errors(trace)
        if trace_errors:
            raise BenchmarkContractError("formal trace contract failed: " + ", ".join(trace_errors))
    source = source_provenance(args, truth, filters)
    filters_by_name = {item.name: item for item in filters}
    rows_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    completed_paired_windows: list[int] = []
    adaptive_lifecycle_state = {"online_materializations": 0}
    adaptive_reset_evidence: dict[str, Any] | None = None
    eager_prebuild_evidence: dict[str, Any] | None = None
    backends: dict[str, ModeBackend] = {}
    database: dict[str, Any] = {}
    backend_sessions: dict[str, dict[str, Any]] = {}
    correctness_audit: dict[str, Any] = {}
    persisted_reuse_evidence: dict[str, Any] = {}
    checkpoint = checkpoint_path(args.out)
    try:
        backends = open_mode_backends(psycopg, pg_config_from_env().conninfo, table=args.table, index=args.index)
        database_contract = validate_database_contract(backends, args, source)
        database = dict(database_contract["database"])
        backend_sessions = dict(database_contract["backend_sessions"])
        run_spec = make_run_spec(args, source, database, trace)
        spec_hash = canonical_sha256(run_spec)
        if checkpoint_exists(checkpoint):
            if checkpoint.exists():
                load_checkpoint(checkpoint, spec_hash)
            raise BenchmarkContractError(
                "checkpoint exists after a prior interrupted run; it is complete paired-window evidence only and cannot be resumed "
                "across newly opened mode backends"
            )
        row_provenance = {
            **database,
            "sqlens_source_aggregate_sha256": source["sqlens_source"]["aggregate_sha256"],
        }
        adaptive_started_empty, adaptive_reset_evidence, eager_prebuild_evidence = initialize_mode_backends(
            backends, args, filters, truth
        )
        persistent_reset = adaptive_reset_evidence.get("persistent_fragment_store_reset", {})
        if int(persistent_reset.get("heap_oid", -1)) != int(database.get("table_oid", -2)):
            raise BenchmarkContractError("online fragment reset heap_oid does not match target table")
        for window in range(args.requests // args.window_size):
            blocks = run_paired_window(
                backends, args, trace, filters_by_name, truth, row_provenance, window=window,
                adaptive_started_empty=adaptive_started_empty,
                adaptive_lifecycle_state=adaptive_lifecycle_state,
            )
            if not rows_by_mode["adaptive"] and blocks["adaptive"]:
                blocks["adaptive"][0]["adaptive_reset_evidence"] = adaptive_reset_evidence
                blocks["adaptive"][0]["persistent_fragment_reset_proof"] = persistent_reset
            if not rows_by_mode["eager_prebuilt"] and blocks["eager_prebuilt"]:
                blocks["eager_prebuilt"][0]["eager_prebuild_evidence"] = eager_prebuild_evidence
            for mode in MODES:
                rows_by_mode[mode].extend(blocks[mode])
            completed_paired_windows.append(window)
            # The only durable state transition occurs after all three isolated caches completed this trace window.
            write_checkpoint(checkpoint, spec_hash, rows_by_mode, completed_paired_windows, args.window_size)
        correctness_audit = audit_result_correctness(
            backends["stock"].session, rows_by_mode, filters_by_name=filters_by_name,
            table=args.table, candidate_validity_predicate=args.candidate_validity_predicate,
            k=args.k, truth=truth,
        )
        persisted_reuse_evidence = audit_persisted_fragment_reload(
            psycopg, pg_config_from_env().conninfo, args,
            adaptive_rows=rows_by_mode["adaptive"], filters_by_name=filters_by_name,
            truth=truth,
            existing_backend_pids=[backend.backend_pid for backend in backends.values()],
            expected_database=database,
        )
    finally:
        close_mode_backends(backends)
    formal = formal_protocol(args)
    errors = validate_artifact(
        rows_by_mode, trace, recall_delta=args.recall_delta, provenance=database,
        source=source,
        absolute_recall_target=args.absolute_recall_target, k=args.k, formal=formal,
        persisted_reuse_evidence=persisted_reuse_evidence,
    )
    all_rows = [row for mode in MODES for row in rows_by_mode[mode]]
    stock_by_request = {int(row["request_no"]): row for row in rows_by_mode["stock"]}
    windows = [{"mode": mode, "window": window, "phase": next(request.phase for request in trace if request.window == window),
                **summary_for_window([row for row in rows_by_mode[mode] if row["window"] == window], bootstrap_samples=args.bootstrap_samples,
                                     bootstrap_seed=args.bootstrap_seed + window,
                                     bootstrap_blocks=[[row for row in rows_by_mode[mode] if row["window"] == window]])}
               for mode in MODES for window in range(args.requests // args.window_size)]
    for item in windows:
        if item["mode"] != "stock":
            mode_rows = [row for row in rows_by_mode[item["mode"]] if row["window"] == item["window"]]
            item["cumulative_savings_vs_stock_ms"] = sum(
                float(stock_by_request[int(row["request_no"])] ["e2e_ms"]) - float(row["e2e_ms"])
                for row in mode_rows if int(row["request_no"]) in stock_by_request and not row.get("error")
            )
            item["cumulative_break_even_request"] = break_even_request(mode_rows, stock_by_request)
        else:
            item["cumulative_savings_vs_stock_ms"] = 0.0
            item["cumulative_break_even_request"] = None
    timeline = []
    for mode in MODES:
        cumulative_build = 0.0
        cumulative_savings = 0.0
        for row in sorted(rows_by_mode[mode], key=lambda item: item["request_no"]):
            cumulative_build += float(row["materialization_ms"])
            if mode != "stock" and int(row["request_no"]) in stock_by_request:
                cumulative_savings += float(stock_by_request[int(row["request_no"])] ["e2e_ms"]) - float(row["e2e_ms"])
            timeline.append({"mode": mode, "request_no": row["request_no"], "phase": row["phase"],
                             "cumulative_build_ms": cumulative_build, "cache_resident_bytes": row["cache_resident_bytes_after"],
                             "fragment_created": row["fragment_created"], "fragment_reused": row["fragment_reused"],
                             "fragment_store_hit_delta": row.get("fragment_store_hit_delta", 0),
                             "lifecycle_path": row.get("lifecycle_path", "none"),
                             "cumulative_savings_vs_stock_ms": cumulative_savings})
    for item in windows:
        stock_window = next((candidate for candidate in windows if candidate["mode"] == "stock" and candidate["window"] == item["window"]), None)
        item["benefit_vs_stock_mean_ms"] = (float(stock_window["e2e_mean_ms"]) - float(item["e2e_mean_ms"])) if stock_window else 0.0
    cumulative_build_cost = {
        mode: sum(float(row["materialization_ms"]) for row in rows_by_mode[mode]) for mode in MODES
    }
    quality = quality_summary(
        rows_by_mode, trace, absolute_recall_target=args.absolute_recall_target, k=args.k
    )
    lifecycle = adaptive_lifecycle_summary(rows_by_mode["adaptive"])
    amortization = {
        mode: amortization_summary(
            rows_by_mode[mode], stock_by_request,
            bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
        )
        for mode in MODES if mode != "stock"
    }
    method_summaries = [
        {
            "mode": mode,
            **summary_for_window(
                rows_by_mode[mode], bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_blocks=[
                    [row for row in rows_by_mode[mode] if row["window"] == window]
                    for window in range(args.requests // args.window_size)
                ],
            ),
        }
        for mode in MODES
    ]
    phase_shift_recovery = {
        mode: {
            "first_shift_window": next((item for item in windows if item["mode"] == mode and item["phase"] == "phase_shift_hot"), None),
            "shift_windows": [item for item in windows if item["mode"] == mode and item["phase"] == "phase_shift_hot"],
        }
        for mode in MODES
    }
    phase_summaries = []
    for mode in MODES:
        for phase in ("steady_hot", "phase_shift_hot"):
            phase_rows = [row for row in rows_by_mode[mode] if row["phase"] == phase]
            phase_summary = summary_for_window(
                phase_rows, bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + (0 if phase == "steady_hot" else 1),
                bootstrap_blocks=[
                    [row for row in phase_rows if row["window"] == window]
                    for window in sorted({int(row["window"]) for row in phase_rows})
                ],
            )
            if mode != "stock":
                phase_summary["cumulative_savings_vs_stock_ms"] = sum(
                    float(stock_by_request[int(row["request_no"])] ["e2e_ms"]) - float(row["e2e_ms"])
                    for row in phase_rows if int(row["request_no"]) in stock_by_request and not row.get("error")
                )
                phase_summary["cumulative_break_even_request"] = break_even_request(phase_rows, stock_by_request)
            else:
                phase_summary["cumulative_savings_vs_stock_ms"] = 0.0
                phase_summary["cumulative_break_even_request"] = None
            phase_summaries.append({"mode": mode, "phase": phase, **phase_summary})
    summary = {"artifact_valid": not errors, "formal_artifact_valid": formal and not errors,
               "validation_errors": errors, "run_spec": run_spec, "run_spec_hash": spec_hash,
               "artifact_kind": run_spec["trace_kind"],
               "production_trace": False,
               "trace_disclosure": "deterministic q10k PostgreSQL online replay; not a captured production trace",
               "effective_unique_queries": len({item.query_id for item in trace}),
               "unique_query_vectors": len({item.query_id for item in trace}),
               "one_request_per_query_vector": len({item.query_id for item in trace}) == len(trace),
               "database_cracking": False,
               "non_formal_debug_override": not formal,
               "window_summaries": windows, "phase_summaries": phase_summaries,
               "method_summaries": method_summaries,
               "cumulative_build_cost_ms": cumulative_build_cost,
               "quality_gate": quality,
               "correctness_audit": correctness_audit,
               "persisted_fragment_reuse": persisted_reuse_evidence,
               "adaptive_lifecycle": lifecycle,
               "amortization": amortization,
               "formal_artifact_gate": {
                   "enforced": formal,
                   "passed": formal and not errors,
                   "absolute_recall_target": args.absolute_recall_target,
                   "per_filter_recall_and_correctness_required": True,
                   "unique_q10k_trace_hot_shift_and_paired_schedule_required": True,
                   "probe_materialization_reuse_required": True,
                   "same_predicate_direct_reuse_evidence_required": True,
                   "fresh_backend_persistent_fragment_reuse_required": True,
                   "materialization_phase_attribution_required": True,
                   "eager_setup_separate_from_timed_requests_required": True,
                   "per_backend_common_warmup_required": True,
                   "positive_reuse_savings_required": True,
                   "paired_mean_savings_ci95_lower_bound_positive_required": True,
                   "cumulative_break_even_required": True,
                   "latency_ci_method": "paired_window_circular_moving_block_bootstrap",
               },
               "break_even_request": {mode: break_even_request(rows_by_mode[mode], stock_by_request) for mode in MODES if mode != "stock"},
               "cumulative_break_even_request": {mode: break_even_request(rows_by_mode[mode], stock_by_request) for mode in MODES if mode != "stock"},
               "phase_shift_recovery": phase_shift_recovery,
               "adaptive_reset_evidence": adaptive_reset_evidence,
               "eager_prebuild_evidence": eager_prebuild_evidence,
               "backend_sessions": backend_sessions,
               "checkpoint_resume_contract": checkpoint_resume_contract(),
               "measurement_mode": PAIRING_SCHEDULE}
    write_csv(args.out, all_rows)
    windows_path = args.out.with_name(args.out.stem + "_windows.csv")
    timeline_path = args.out.with_name(args.out.stem + "_timeline.csv")
    summary_path = args.out.with_name(args.out.stem + "_summary.json")
    manifest_path = args.out.with_name(args.out.stem + "_manifest.json")
    write_csv(windows_path, windows)
    write_csv(timeline_path, timeline)
    atomic_json(summary_path, summary)
    outputs = {
        "raw_csv": {"path": str(args.out), "rows": len(all_rows), "sha256": sha256_file(args.out)},
        "windows_csv": {"path": str(windows_path), "rows": len(windows), "sha256": sha256_file(windows_path)},
        "timeline_csv": {"path": str(timeline_path), "rows": len(timeline), "sha256": sha256_file(timeline_path)},
        "summary_json": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
    }
    atomic_json(manifest_path, {
        "artifact_contract": "sqlens_d3_online_replay_v2",
        "status": "complete" if not errors else "invalid",
        "artifact_valid": not errors,
        "formal_release_complete": formal and not errors,
        "requested_trace_complete": all(
            len(rows_by_mode[mode]) == args.requests for mode in MODES
        ),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_spec_hash": spec_hash,
        "run_spec": run_spec,
        "validation_errors": errors,
        "persisted_fragment_reuse": persisted_reuse_evidence,
        "outputs": outputs,
        "source": source,
        "database": database,
        "checkpoint_contract": checkpoint_resume_contract(),
        "checkpoint_cleaned_after_immutable_outputs": not errors,
    })
    if errors:
        return 2
    cleanup_checkpoint(checkpoint_path(args.out))
    return 0


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Formal Amazon-10M deterministic PostgreSQL online D3 replay")
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--query-cohort", type=Path, default=DEFAULT_QUERY_COHORT)
    parser.add_argument("--query-cohort-manifest", type=Path, default=DEFAULT_QUERY_COHORT_MANIFEST)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--truth-manifest", type=Path, default=DEFAULT_TRUTH_MANIFEST)
    parser.add_argument("--index-health-manifest", type=Path, default=DEFAULT_INDEX_HEALTH_MANIFEST)
    parser.add_argument(
        "--truth-query-count", type=int, default=FORMAL_TRUTH_QUERY_COUNT,
        help="debug override for smaller truth grids; formal evidence requires q10200 with q200..q10199 measured",
    )
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--candidate-validity-predicate", default=DEFAULT_CANDIDATE_VALIDITY_PREDICATE)
    parser.add_argument("--out", type=Path, default=ROOT / "results/hybrid_vector_db/amazon10m_d3_adaptation_lifecycle.csv")
    parser.add_argument(
        "--fragment-store-run-id", default="",
        help="optional run identifier used to isolate each mode's persisted fragment-store keys",
    )
    parser.add_argument("--execute", action="store_true", help="run the database experiment; dry-run is the default")
    parser.add_argument("--dry-run", action="store_true", help="print the formal contract without reading inputs or connecting")
    parser.add_argument(
        "--preflight", action="store_true",
        help="read-only formal readiness audit; connect three backends but do not run requests or mutate cache/store",
    )
    parser.add_argument("--resume", action="store_true",
                        help="rejected: backend-local D3/cache state cannot be restored across processes")
    parser.add_argument("--requests", type=int, default=FORMAL_REQUESTS, help="debug only when not 10000; labels output non-formal")
    parser.add_argument("--window-size", type=int, default=FORMAL_WINDOW, help="debug only when not 100; labels output non-formal")
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--d3-probe-requests", "--admission-reuse-threshold", type=int, default=FORMAL_D3_PROBE_REQUESTS,
                        help="stock probes observed by the extension before adaptive admission")
    parser.add_argument("--eager-kind", choices=("bloom", "page"), default="bloom")
    parser.add_argument("--d3-min-benefit-per-byte", type=float, default=FORMAL_D3_MIN_BENEFIT_PER_BYTE)
    parser.add_argument("--d3-max-fragment-mb", type=int, default=FORMAL_D3_MAX_FRAGMENT_MB)
    parser.add_argument(
        "--d3-page-min-skip-rate", "--d3-refine-skip-rate",
        dest="d3_page_min_skip_rate", type=float, default=FORMAL_D3_PAGE_MIN_SKIP_RATE,
        help="refine page guidance to Bloom when its measured skip rate is below this value",
    )
    parser.add_argument("--cache-mb", type=int, default=FORMAL_CACHE_MB)
    parser.add_argument("--guidance-filter-strategy", choices=("guided_collect", "safe_guided"), default="safe_guided")
    parser.add_argument("--k", type=int, default=FORMAL_K)
    parser.add_argument("--ef-search", type=int, default=FORMAL_EF_SEARCH)
    parser.add_argument("--iterative-scan", choices=("off", "relaxed_order", "strict_order"), default="strict_order")
    parser.add_argument("--max-scan-tuples", type=int, default=FORMAL_MAX_SCAN_TUPLES)
    parser.add_argument("--scan-mem-multiplier", type=float, default=FORMAL_SCAN_MEM_MULTIPLIER)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recall-delta", type=float, default=FORMAL_RECALL_DELTA)
    parser.add_argument("--absolute-recall-target", type=float, default=FORMAL_RECALL_TARGET)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    return parser


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    disclosure = replay_disclosure(args)
    formal = formal_protocol(args)
    return {"dry_run": True, "database_connected": False, "inputs_read": False, "files_written": False,
            "modes": list(MODES), "requests": args.requests, "window_size": args.window_size,
            "formal": formal,
            "debug_override_labeled_non_formal": not formal,
            "fixed_exact_truth_query_count": args.truth_query_count,
            "unique_query_vectors": args.requests,
            "one_request_per_query_vector": True,
            "candidate_validity_predicate": args.candidate_validity_predicate,
            **disclosure,
            "trace_source": "generated_deterministically_by_this_runner",
            "production_trace": False, "postgresql_online_execution": True,
            "database_cracking": False, "single_client_sequential": True,
            "search_configuration": {
                "k": args.k,
                "ef_search": args.ef_search,
                "iterative_scan": args.iterative_scan,
                "max_scan_tuples": args.max_scan_tuples,
                "scan_mem_multiplier": args.scan_mem_multiplier,
                "force_hnsw": args.force_hnsw,
                "statement_timeout_ms": args.statement_timeout_ms,
            },
            "absolute_recall_target": args.absolute_recall_target,
            "measurement_schedule": PAIRING_SCHEDULE,
            "mode_backend_topology": "three_independent_persistent_postgresql_backends",
            "checkpoint_resume_contract": checkpoint_resume_contract(),
            "adaptive_contract": "reset empty metadata cache and target-heap persistent fragment store; no activate/prewarm outside timed requests"}


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if args.requests <= 0 or args.window_size <= 0 or args.requests % args.window_size:
        raise SystemExit("--requests must be positive and divisible by --window-size")
    if args.truth_query_count <= 0 or args.requests > args.truth_query_count:
        raise SystemExit("--truth-query-count must be positive and at least --requests")
    if args.d3_probe_requests < 1:
        raise SystemExit("--d3-probe-requests must be at least one")
    if not 0.0 < args.absolute_recall_target <= 1.0:
        raise SystemExit("--absolute-recall-target must be in (0, 1]")
    if args.preflight:
        payload = preflight_experiment(args)
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["ready"] else 2
    if args.dry_run or not args.execute:
        print(json.dumps(dry_run_payload(args), sort_keys=True))
        return 0
    return execute_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
