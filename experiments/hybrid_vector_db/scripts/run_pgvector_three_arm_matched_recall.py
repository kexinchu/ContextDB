"""Formal Amazon-10M three-arm matched-recall pgvector controller.

The controller compares exactly three arms:

* the pinned official pgvector binary;
* the SQLens binary with every non-stock HNSW control disabled/reset; and
* the full SQLens D1+D2+D3 path on the BFS physical clone.

It deliberately reuses the binary switcher, stock overhead runner, and SQLens
target-recall runner.  This module owns the cross-arm contract: fixed query
splits, binary/build identity after every restart, cache treatment, rotating
final blocks, canonical query-level pairing, and publication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

try:
    from . import pgvector_upstream_overhead_control as upstream_runner
    from . import run_pgvector_binary_ab_control as binary_controller
except ImportError:
    import pgvector_upstream_overhead_control as upstream_runner
    import run_pgvector_binary_ab_control as binary_controller


class _LazyTargetRunner:
    """Keep dry-run and pure finalization independent of psycopg availability."""

    _module: Any = None

    def _load(self) -> Any:
        if self._module is None:
            if __package__:
                self._module = importlib.import_module(
                    ".pgvector_target_recall_selectivity_runner", __package__
                )
            else:
                self._module = importlib.import_module(
                    "pgvector_target_recall_selectivity_runner"
                )
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


target_runner = _LazyTargetRunner()


ROOT = Path(__file__).resolve().parents[3]
ARMS = ("official", "sqlens_disabled", "sqlens_full")
SQLENS_ARMS = frozenset({"sqlens_disabled", "sqlens_full"})
TARGET_RECALLS = (0.90, 0.95, 0.99)
SCREEN_QUERY_NOS = tuple(range(0, 20))
CALIBRATION_QUERY_NOS = tuple(range(20, 100))
FINAL_QUERY_NOS = tuple(range(100, 200))
SCREEN_REPEATS = 1
CALIBRATION_REPEATS = 2
FINAL_REPEATS = 6
FINAL_BLOCKS = 6
FINAL_REPEATS_PER_BLOCK = FINAL_REPEATS // FINAL_BLOCKS
HNSW_M = 32
CANDIDATE_VALIDITY_PREDICATE = "embedding_valid"
FULL_SQLENS_MODE = "design1_bloom_bfs_layout_d3"
REQUIRED_TRAVERSAL_BUILD_PREFIX = (
    "sqlens-v16-d3-full-materialization-persisted-reuse-"
)
FULL_EF_SEARCH_VALUES = (
    "20,40,60,80,100,150,200,250,500,750,1000,1500,2000,3000,4000,5000,7000,8500,10000,20000,50000,100000"
)
FORMAL_FILTER_COUNT = 14
FORMAL_CELL_COUNT = FORMAL_FILTER_COUNT * len(TARGET_RECALLS)
SCHEMA_VERSION = 2
CACHE_PROTOCOL_VERSION = "warm-os-pgprewarm-fixed-query-warmup-v1"
CONTROLLER_NAME = "run_pgvector_three_arm_matched_recall"
CALIBRATION_SELECTION_POLICY = "lcb_then_max_recall"
CALIBRATION_SELECTION_RULE = (
    "mean Recall@10 >= target is required; among complete configurations whose "
    "query-cluster bootstrap Recall@10 LCB95 also reaches target, select the "
    "lowest mean latency; when none is LCB-confirmed, select the highest mean "
    "recall and break exact recall ties by lowest mean latency"
)
DEFAULT_FILTERS = (
    ROOT
    / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_valid_embeddings_filters.csv"
)

CANONICAL_FIELDS = (
    "run_uuid",
    "phase",
    "arm",
    "filter_name",
    "target_recall",
    "query_no",
    "query_id",
    "repeat",
    "final_block",
    "arm_order_position",
    "config_label",
    "ef_search",
    "guided_collect_target",
    "iterative_scan",
    "max_scan_tuples",
    "scan_mem_multiplier",
    "latency_ms",
    "recall_at_10",
    "final_path",
    "planner_proof_succeeded",
    "approximate_ann_path",
    "approximate_prioritization_attempted",
    "traversal_order_changed",
    "priority_reorders",
    "match_frontier_pops",
    "no_bridge_frontier_pops",
    "traversal_prioritization_burst",
    "valid",
    "error",
    "source_path",
    "source_sha256",
    "source_row_no",
    "vector_so_sha256",
    "build_id",
    "pair_key",
    "measurement_key",
)


class ThreeArmError(RuntimeError):
    """A formal three-arm protocol or execution gate failed."""


class ProtocolError(ThreeArmError):
    """A fixed workload, graph, cache, or row contract failed."""


class RuntimeIdentityError(ThreeArmError):
    """The restarted server did not load the requested vector binary/build."""


class CheckpointError(ThreeArmError):
    """A checkpoint is stale, incomplete, or belongs to another run."""


class FinalizationError(ThreeArmError):
    """The three arm outputs cannot form a publishable paired result."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return binary_controller.sha256_file(path)


def sha256_json(value: Any) -> str:
    return binary_controller.sha256_json(value)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    binary_controller.atomic_write_json(path, value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    preferred: Sequence[str] = (),
) -> None:
    upstream_runner.write_csv_atomic(path, rows, preferred)


def artifact_entry(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointError(f"artifact is missing: {path}")
    entry: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        entry["rows"] = rows
    return entry


def validate_artifact_entry(value: Mapping[str, Any]) -> Path:
    path = Path(str(value.get("path", "")))
    digest = str(value.get("sha256", ""))
    if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CheckpointError(f"checkpoint artifact identity is incomplete: {path}")
    if sha256_file(path) != digest:
        raise CheckpointError(f"checkpoint artifact SHA256 changed: {path}")
    if "bytes" in value and path.stat().st_size != int(value["bytes"]):
        raise CheckpointError(f"checkpoint artifact size changed: {path}")
    if "rows" in value and len(read_csv(path)) != int(value["rows"]):
        raise CheckpointError(f"checkpoint artifact row count changed: {path}")
    return path


def formal_protocol() -> dict[str, Any]:
    """Return the immutable paper-facing design contract."""
    return {
        "dataset": "Amazon10M",
        "arms": list(ARMS),
        "hnsw": {
            "m": HNSW_M,
            "source_layout": "insertion",
            "clone_layout": "bfs",
            "same_heap_same_logical_graph_required": True,
            "physical_layout_must_differ": True,
        },
        "ground_truth": {
            "candidate_validity_predicate": CANDIDATE_VALIDITY_PREDICATE,
            "self_excluded": True,
            "recall": "tie-aware Recall@10",
        },
        "query_splits": {
            "screen": list(SCREEN_QUERY_NOS),
            "calibration": list(CALIBRATION_QUERY_NOS),
            "final": list(FINAL_QUERY_NOS),
        },
        "repeats": {
            "screen": SCREEN_REPEATS,
            "calibration": CALIBRATION_REPEATS,
            "final": FINAL_REPEATS,
        },
        "target_recalls": list(TARGET_RECALLS),
        "calibration_selection_policy": CALIBRATION_SELECTION_POLICY,
        "selection_rule": CALIBRATION_SELECTION_RULE,
        "final_schedule": {
            "blocks": FINAL_BLOCKS,
            "repeats_per_block": FINAL_REPEATS_PER_BLOCK,
            "arm_order": (
                "six binary-switched one-repeat blocks; seeded initial permutation, "
                "cyclic rotation by block"
            ),
            "interleaving_unit": "one complete q100..q199 repeat per arm invocation",
        },
        "bootstrap": {
            "calibration_unit": "query cluster after averaging two repeats",
            "calibration_lcb_confidence": 0.95,
            "unit": "paired query cluster after averaging six repeats",
            "confidence": 0.95,
        },
    }


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join(str(part) for part in (seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def rotating_final_schedule(
    run_uuid: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build six seeded, cyclically rotated one-repeat binary-switch blocks."""
    if not run_uuid:
        raise ProtocolError("final schedule requires a nonempty run UUID")
    first = list(ARMS)
    random.Random(_stable_seed(seed, run_uuid, "three-arm-final-order")).shuffle(first)
    direction = 1 if _stable_seed(seed, run_uuid, "rotation-direction") & 1 else -1
    block_orders: list[list[str]] = []
    schedule: list[dict[str, Any]] = []
    sequence = 0
    for block in range(FINAL_BLOCKS):
        offset = (direction * block) % len(first)
        order = first[offset:] + first[:offset]
        block_orders.append(order)
        for position, arm in enumerate(order):
            sequence += 1
            schedule.append(
                {
                    "sequence": sequence,
                    "final_block": block,
                    "repeat_first": block * FINAL_REPEATS_PER_BLOCK,
                    "repeat_last": (block + 1) * FINAL_REPEATS_PER_BLOCK - 1,
                    "repeats": FINAL_REPEATS_PER_BLOCK,
                    "position": position,
                    "arm": arm,
                }
            )
    arm_counts = {arm: sum(row["arm"] == arm for row in schedule) for arm in ARMS}
    positions = {
        arm: [row["position"] for row in schedule if row["arm"] == arm]
        for arm in ARMS
    }
    passed = (
        all(set(order) == set(ARMS) and len(order) == len(ARMS) for order in block_orders)
        and set(arm_counts.values()) == {FINAL_BLOCKS}
        and block_orders[0] != block_orders[1]
        and all(Counter(items) == Counter({0: 2, 1: 2, 2: 2}) for items in positions.values())
        and all(sorted(items) == list(range(FINAL_REPEATS)) for items in (
            [repeat for block in range(FINAL_BLOCKS) for repeat in range(
                block * FINAL_REPEATS_PER_BLOCK,
                (block + 1) * FINAL_REPEATS_PER_BLOCK,
            )]
            for _arm in ARMS
        ))
    )
    audit = {
        "seed": seed,
        "run_uuid": run_uuid,
        "initial_order": first,
        "rotation_direction": direction,
        "block_orders": block_orders,
        "arm_counts": arm_counts,
        "positions_by_arm": positions,
        "total_invocations": len(schedule),
        "seeded_rotation_verified": passed,
    }
    if not passed:
        raise ProtocolError(f"internal three-arm schedule audit failed: {audit}")
    return schedule, audit


def calibration_order(run_uuid: str, seed: int) -> list[str]:
    order = list(ARMS)
    random.Random(_stable_seed(seed, run_uuid, "three-arm-calibration-order")).shuffle(order)
    return order


def _required_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _index_build_contract(
    payload: Mapping[str, Any], role: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    preparation = _required_mapping(payload.get("preparation"), "graph preparation")
    indexes = _required_mapping(preparation.get("indexes"), "graph preparation indexes")
    index = _required_mapping(indexes.get(role), f"graph {role} index evidence")
    contract = _required_mapping(index.get("build_contract"), f"graph {role} build contract")
    state = _required_mapping(index.get("state"), f"graph {role} index state")
    differences = index.get("definition_diff")
    if differences not in ({}, None):
        raise ProtocolError(f"graph {role} index has a nonempty definition diff")
    return contract, state


def validate_m32_same_graph_proof(
    path: Path,
    source_index: str,
    clone_index: str,
) -> dict[str, Any]:
    """Validate the canonical proof plus M32/build provenance embedded in it."""
    try:
        canonical = upstream_runner.load_graph_identity(path, source_index, clone_index)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, upstream_runner.ProvenanceGateError) as exc:
        raise ProtocolError(f"invalid source/BFS graph proof: {exc}") from exc
    if payload.get("artifact_valid") is not True:
        raise ProtocolError("graph proof artifact_valid is not true")
    source_contract, source_state = _index_build_contract(payload, "source")
    clone_contract, clone_state = _index_build_contract(payload, "clone")
    expected = {
        "m": HNSW_M,
        "predicate": CANDIDATE_VALIDITY_PREDICATE,
        "opclass": "vector_l2_ops",
    }
    for role, contract, state in (
        ("source", source_contract, source_state),
        ("clone", clone_contract, clone_state),
    ):
        for key, value in expected.items():
            if contract.get(key) != value:
                raise ProtocolError(
                    f"graph {role} build contract {key}={contract.get(key)!r}; "
                    f"expected {value!r}"
                )
        if contract.get("role") != role:
            raise ProtocolError(f"graph {role} build role is not {role!r}")
        reloptions = upstream_runner.safe_predicate(
            str(state.get("predicate", "")), f"graph {role} predicate"
        )
        if reloptions != CANDIDATE_VALIDITY_PREDICATE:
            raise ProtocolError(f"graph {role} state predicate is not embedding_valid")
        options = state.get("reloptions") or []
        parsed_options = {}
        for item in options:
            key, separator, value = str(item).partition("=")
            if separator:
                parsed_options[key.strip()] = value.strip()
        if parsed_options.get("m") != str(HNSW_M):
            raise ProtocolError(f"graph {role} state does not prove m={HNSW_M}")
    if source_contract.get("build_page_order") != "insertion":
        raise ProtocolError("source graph was not built in insertion order")
    if clone_contract.get("build_page_order") != "bfs":
        raise ProtocolError("clone graph was not built in BFS order")
    if clone_contract.get("clone_source") != source_index:
        raise ProtocolError("BFS clone build contract is not bound to the source index")
    if clone_contract.get("require_full_memory_build") is not True:
        raise ProtocolError("BFS clone does not prove a full-memory same-graph build")
    if source_contract.get("table") != clone_contract.get("table"):
        raise ProtocolError("source and BFS build contracts name different heaps")
    fingerprint = str(canonical.get("stable_fingerprint_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ProtocolError("graph proof stable fingerprint is missing or invalid")
    return {
        **canonical,
        "hnsw_m": HNSW_M,
        "candidate_validity_predicate": CANDIDATE_VALIDITY_PREDICATE,
        "source_build_contract": dict(source_contract),
        "clone_build_contract": dict(clone_contract),
        "formal_gate_passed": True,
    }


def validate_truth_and_filters(
    filters_path: Path,
    truth_path: Path,
    selected_filters: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    try:
        filters = upstream_runner.load_filters(filters_path, selected_filters)
        if len(filters) != FORMAL_FILTER_COUNT:
            raise ProtocolError(
                f"formal run requires {FORMAL_FILTER_COUNT} filters, got {len(filters)}"
            )
        truth = upstream_runner.load_truth(
            truth_path,
            range(200),
            {row["filter_name"] for row in filters},
            10,
            CANDIDATE_VALIDITY_PREDICATE,
        )
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"invalid embedding_valid exact truth contract: {exc}") from exc
    query_ids: dict[int, int] = {}
    for query_no in range(200):
        ids = {
            truth[(row["filter_name"], query_no)].query_id
            for row in filters
        }
        if len(ids) != 1:
            raise ProtocolError(f"query_no={query_no} maps to multiple query IDs")
        query_ids[query_no] = ids.pop()
    if len(set(query_ids.values())) != len(query_ids):
        raise ProtocolError("formal q0..q199 must map to 200 distinct query IDs")
    return filters, {
        "filters_sha256": sha256_file(filters_path),
        "truth_sha256": sha256_file(truth_path),
        "candidate_validity_predicate": CANDIDATE_VALIDITY_PREDICATE,
        "query_ids": query_ids,
        "query_count": len(query_ids),
        "truth_rows": len(truth),
        "self_excluded": True,
        "tie_aware": True,
    }


def _explicit_bool(value: object, field: str) -> bool:
    if value is True or str(value).strip().lower() == "true":
        return True
    if value is False or str(value).strip().lower() == "false":
        return False
    raise ProtocolError(f"{field} must be an explicit boolean")


def _finite_float(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field} is not numeric") from exc
    if not math.isfinite(number):
        raise ProtocolError(f"{field} is not finite")
    return number


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise FinalizationError("cannot compute a percentile from an empty sample")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _bootstrap_mean_ci(
    values: Sequence[float], samples: int, seed: int
) -> tuple[float, float]:
    _lcb95, low, high = _bootstrap_mean_bounds(values, samples, seed)
    return low, high


def _bootstrap_mean_bounds(
    values: Sequence[float], samples: int, seed: int
) -> tuple[float, float, float]:
    if not values or samples <= 0:
        raise FinalizationError("bootstrap requires nonempty values and positive samples")
    observed = [float(value) for value in values]
    rng = random.Random(seed)
    draws = [
        statistics.fmean(observed[rng.randrange(len(observed))] for _ in observed)
        for _ in range(samples)
    ]
    return (
        _percentile(draws, 0.05),
        _percentile(draws, 0.025),
        _percentile(draws, 0.975),
    )


def _calibration_bootstrap_mean_bounds(
    values: Sequence[float], samples: int, seed: int
) -> tuple[float, float, float]:
    if not values or samples <= 0:
        raise ProtocolError(
            "calibration bootstrap requires nonempty values and positive samples"
        )
    draws = upstream_runner.bootstrap_means(values, samples, seed)
    return (
        _percentile(draws, 0.05),
        _percentile(draws, 0.025),
        _percentile(draws, 0.975),
    )


def calibration_bootstrap_seed(
    base_seed: int,
    arm: str,
    filter_name: str,
    config_label: str,
) -> int:
    material = (
        f"{base_seed}|calibration-selection|{arm}|{filter_name}|{config_label}"
    )
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def summarize_config_measurements(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_query_nos: Sequence[int],
    expected_repeats: int,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260718,
) -> dict[str, Any]:
    if not rows:
        raise ProtocolError("configuration summary received no measurements")
    expected_queries = set(map(int, expected_query_nos))
    expected_keys = {
        (query_no, repeat)
        for query_no in expected_queries
        for repeat in range(expected_repeats)
    }
    seen: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["query_no"]), int(row["repeat"]))
        if key in seen:
            raise ProtocolError(f"duplicate configuration measurement key {key}")
        seen[key] = row
        if not _explicit_bool(row.get("valid", False), "valid") or row.get("error"):
            raise ProtocolError(f"invalid configuration measurement at {key}")
        latency = _finite_float(row["latency_ms"], "latency_ms")
        recall = _finite_float(row["recall_at_10"], "recall_at_10")
        if latency <= 0 or not 0 <= recall <= 1:
            raise ProtocolError(f"invalid latency/recall at {key}")
    if set(seen) != expected_keys:
        missing = sorted(expected_keys - set(seen))[:5]
        extra = sorted(set(seen) - expected_keys)[:5]
        raise ProtocolError(
            f"configuration measurement coverage mismatch; missing={missing}, extra={extra}"
        )
    query_latency: list[float] = []
    query_recall: list[float] = []
    for query_no in sorted(expected_queries):
        query_latency.append(
            statistics.fmean(
                _finite_float(seen[(query_no, repeat)]["latency_ms"], "latency_ms")
                for repeat in range(expected_repeats)
            )
        )
        query_recall.append(
            statistics.fmean(
                _finite_float(seen[(query_no, repeat)]["recall_at_10"], "recall_at_10")
                for repeat in range(expected_repeats)
            )
        )
    recall_lcb95, recall_ci_low, recall_ci_high = _calibration_bootstrap_mean_bounds(
        query_recall,
        bootstrap_samples,
        bootstrap_seed,
    )
    return {
        "queries": len(query_latency),
        "repeats": expected_repeats,
        "samples": len(rows),
        "complete": True,
        "recall_mean": statistics.fmean(query_recall),
        "recall_lcb95": recall_lcb95,
        "recall_ci_low": recall_ci_low,
        "recall_ci_high": recall_ci_high,
        "recall_bootstrap_unit": "query_cluster_after_repeat_mean",
        "recall_bootstrap_samples": bootstrap_samples,
        "recall_bootstrap_seed": bootstrap_seed,
        "recall_lcb_confidence": 0.95,
        "latency_mean_ms": statistics.fmean(query_latency),
        "latency_p50_ms": _percentile(query_latency, 0.50),
        "latency_p95_ms": _percentile(query_latency, 0.95),
        "latency_p99_ms": _percentile(query_latency, 0.99),
    }


def select_fastest_qualifying_config(
    summaries: Sequence[Mapping[str, Any]], target: float
) -> dict[str, Any] | None:
    if not 0 < target <= 1:
        raise ProtocolError("target recall must be in (0, 1]")
    labels: set[str] = set()
    mean_eligible: list[dict[str, Any]] = []
    for source in summaries:
        row = dict(source)
        label = str(row.get("config_label") or row.get("config") or "")
        if not label or label in labels:
            raise ProtocolError("calibration summaries contain a missing/duplicate config")
        labels.add(label)
        complete = row.get("complete") is True or str(row.get("complete", "")).lower() == "true"
        if not complete or int(row.get("errors", 0) or 0) != 0:
            raise ProtocolError(f"calibration config {label} is incomplete or failed")
        recall = _finite_float(row.get("recall_mean"), "recall_mean")
        recall_lcb95 = _finite_float(row.get("recall_lcb95"), "recall_lcb95")
        latency = _finite_float(row.get("latency_mean_ms"), "latency_mean_ms")
        if latency <= 0 or not 0 <= recall <= 1 or not 0 <= recall_lcb95 <= 1:
            raise ProtocolError(f"calibration config {label} has invalid metrics")
        row["config_label"] = label
        if recall >= target:
            mean_eligible.append(row)
    if not mean_eligible:
        return None
    lcb95_eligible = [
        row for row in mean_eligible if float(row["recall_lcb95"]) >= target
    ]
    if lcb95_eligible:
        selection_pool = lcb95_eligible
        selection_fallback = "none"
    else:
        best_recall = max(float(row["recall_mean"]) for row in mean_eligible)
        selection_pool = [
            row for row in mean_eligible if float(row["recall_mean"]) == best_recall
        ]
        selection_fallback = "max_mean_recall"
    selected = min(
        selection_pool,
        key=lambda row: (
            float(row["latency_mean_ms"]),
            str(row["config_label"]),
        ),
    )
    return {
        **selected,
        "calibration_selection_policy": CALIBRATION_SELECTION_POLICY,
        "selection_fallback": selection_fallback,
        "mean_qualified_configs": len(mean_eligible),
        "lcb95_qualified_configs": len(lcb95_eligible),
    }


def select_calibrated_configs(
    rows: Sequence[Mapping[str, Any]],
    filters: Sequence[str],
    *,
    arms: Sequence[str] = ARMS,
    targets: Sequence[float] = TARGET_RECALLS,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260718,
) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for arm in arms:
        if arm not in ARMS:
            raise ProtocolError(f"unknown arm {arm!r}")
        for filter_name in filters:
            candidates = [
                row
                for row in rows
                if str(row.get("arm")) == arm
                and str(row.get("filter_name")) == filter_name
            ]
            if not candidates:
                raise ProtocolError(f"missing calibration candidates for {arm}/{filter_name}")
            by_config: dict[str, list[Mapping[str, Any]]] = {}
            for row in candidates:
                by_config.setdefault(str(row["config_label"]), []).append(row)
            summaries = []
            for label, config_rows in sorted(by_config.items()):
                config_fields = {
                    key: config_rows[0].get(key, "")
                    for key in (
                        "ef_search",
                        "guided_collect_target",
                        "iterative_scan",
                        "max_scan_tuples",
                        "scan_mem_multiplier",
                    )
                }
                if any(
                    row.get(key, "") != value
                    for row in config_rows
                    for key, value in config_fields.items()
                ):
                    raise ProtocolError(
                        f"config label {label} maps to inconsistent parameter values"
                    )
                summaries.append(
                    {
                        "arm": arm,
                        "filter_name": filter_name,
                        "config_label": label,
                        "errors": 0,
                        **config_fields,
                        **summarize_config_measurements(
                            config_rows,
                            expected_query_nos=CALIBRATION_QUERY_NOS,
                            expected_repeats=CALIBRATION_REPEATS,
                            bootstrap_samples=bootstrap_samples,
                            bootstrap_seed=calibration_bootstrap_seed(
                                bootstrap_seed,
                                arm,
                                filter_name,
                                label,
                            ),
                        ),
                    }
                )
            for target in targets:
                selected = select_fastest_qualifying_config(summaries, float(target))
                if selected is None:
                    raise ProtocolError(
                        f"unattainable_on_calibration_grid: no complete calibrated "
                        f"config reaches mean recall {target:g} for {arm}/{filter_name}"
                    )
                selections.append(
                    {
                        "arm": arm,
                        "filter_name": filter_name,
                        "target_recall": float(target),
                        "selection_status": "selected",
                        "selection_metric": "query_level_mean_recall_at_10",
                        "calibration_selection_policy": CALIBRATION_SELECTION_POLICY,
                        "selection_rule": CALIBRATION_SELECTION_RULE,
                        "selection_fallback": selected["selection_fallback"],
                        "calibration_recall_lcb95": selected["recall_lcb95"],
                        **{
                            key: value
                            for key, value in selected.items()
                            if key not in {"arm", "filter_name"}
                        },
                    }
                )
    expected = len(arms) * len(filters) * len(targets)
    if len(selections) != expected:
        raise ProtocolError(f"selection matrix has {len(selections)} rows, expected {expected}")
    return selections


def _selection_map(
    selections: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, float], str]:
    result: dict[tuple[str, str, float], str] = {}
    for row in selections:
        key = (
            str(row["arm"]),
            str(row["filter_name"]),
            float(row["target_recall"]),
        )
        label = str(row["config_label"])
        if key in result or not label:
            raise FinalizationError(f"duplicate or empty selection {key}")
        result[key] = label
    return result


def build_query_level_pairs(
    final_rows: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    filters: Sequence[str],
) -> list[dict[str, Any]]:
    selected = _selection_map(selections)
    expected_selection_keys = {
        (arm, filter_name, target)
        for arm in ARMS
        for filter_name in filters
        for target in TARGET_RECALLS
    }
    if set(selected) != expected_selection_keys:
        raise FinalizationError("final selection matrix is not the complete three-arm design")
    cells: dict[tuple[str, float, int, str], list[Mapping[str, Any]]] = {}
    seen_measurements: set[tuple[str, str, float, int, int]] = set()
    for row in final_rows:
        arm = str(row.get("arm", ""))
        filter_name = str(row.get("filter_name", ""))
        target = float(row.get("target_recall", 0.0))
        query_no = int(row.get("query_no", -1))
        repeat = int(row.get("repeat", -1))
        if (arm, filter_name, target) not in selected:
            raise FinalizationError("final row is outside the selected matrix")
        if str(row.get("config_label", "")) != selected[(arm, filter_name, target)]:
            raise FinalizationError("final row does not use its calibrated selected config")
        key = (arm, filter_name, target, query_no, repeat)
        if key in seen_measurements:
            raise FinalizationError(f"duplicate final measurement {key}")
        seen_measurements.add(key)
        if not _explicit_bool(row.get("valid", False), "valid") or row.get("error"):
            raise FinalizationError(f"invalid final measurement {key}")
        latency = _finite_float(row.get("latency_ms"), "latency_ms")
        recall = _finite_float(row.get("recall_at_10"), "recall_at_10")
        if latency <= 0 or not 0 <= recall <= 1:
            raise FinalizationError(f"invalid final latency/recall {key}")
        cells.setdefault((filter_name, target, query_no, arm), []).append(row)

    paired: list[dict[str, Any]] = []
    for filter_name in filters:
        for target in TARGET_RECALLS:
            for query_no in FINAL_QUERY_NOS:
                output: dict[str, Any] = {
                    "filter_name": filter_name,
                    "target_recall": target,
                    "query_no": query_no,
                    "pair_key": f"{filter_name}|{target:g}|q{query_no}",
                }
                query_ids: set[int] = set()
                for arm in ARMS:
                    rows = cells.get((filter_name, target, query_no, arm), [])
                    repeats = sorted(int(row["repeat"]) for row in rows)
                    if repeats != list(range(FINAL_REPEATS)):
                        raise FinalizationError(
                            f"final repeat coverage mismatch for {arm}/{filter_name}/"
                            f"{target:g}/q{query_no}: {repeats}"
                        )
                    query_ids.update(int(row["query_id"]) for row in rows)
                    output[f"{arm}_config_label"] = selected[(arm, filter_name, target)]
                    output[f"{arm}_latency_mean_ms"] = statistics.fmean(
                        float(row["latency_ms"]) for row in rows
                    )
                    output[f"{arm}_recall_mean"] = statistics.fmean(
                        float(row["recall_at_10"]) for row in rows
                    )
                    output[f"{arm}_repeats"] = len(rows)
                if len(query_ids) != 1:
                    raise FinalizationError(
                        f"paired arms disagree on query ID for {filter_name}/{target:g}/q{query_no}"
                    )
                output["query_id"] = query_ids.pop()
                paired.append(output)
    expected_rows = len(filters) * len(TARGET_RECALLS) * len(FINAL_QUERY_NOS)
    if len(paired) != expected_rows:
        raise FinalizationError(f"paired query rows={len(paired)}, expected={expected_rows}")
    return paired


def summarize_paired_final(
    paired_rows: Sequence[Mapping[str, Any]],
    filters: Sequence[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if bootstrap_samples <= 0:
        raise FinalizationError("bootstrap_samples must be positive")
    arm_summaries: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    pairs = (
        ("official", "sqlens_disabled"),
        ("official", "sqlens_full"),
        ("sqlens_disabled", "sqlens_full"),
    )
    for cell_no, (filter_name, target) in enumerate(
        (item for name in filters for item in ((name, value) for value in TARGET_RECALLS))
    ):
        rows = [
            row
            for row in paired_rows
            if str(row["filter_name"]) == filter_name
            and float(row["target_recall"]) == target
        ]
        if len(rows) != len(FINAL_QUERY_NOS):
            raise FinalizationError(f"paired cell {filter_name}/{target:g} is incomplete")
        for arm_no, arm in enumerate(ARMS):
            latencies = [float(row[f"{arm}_latency_mean_ms"]) for row in rows]
            recalls = [float(row[f"{arm}_recall_mean"]) for row in rows]
            latency_ci = _bootstrap_mean_ci(
                latencies,
                bootstrap_samples,
                _stable_seed(bootstrap_seed, "arm", cell_no, arm_no),
            )
            recall_ci = _bootstrap_mean_ci(
                recalls,
                bootstrap_samples,
                _stable_seed(bootstrap_seed, "recall", cell_no, arm_no),
            )
            recall_mean = statistics.fmean(recalls)
            arm_summaries.append(
                {
                    "filter_name": filter_name,
                    "target_recall": target,
                    "arm": arm,
                    "queries": len(rows),
                    "repeats_per_query": FINAL_REPEATS,
                    "config_label": str(rows[0][f"{arm}_config_label"]),
                    "latency_mean_ms": statistics.fmean(latencies),
                    "latency_p50_ms": _percentile(latencies, 0.50),
                    "latency_p95_ms": _percentile(latencies, 0.95),
                    "latency_p99_ms": _percentile(latencies, 0.99),
                    "latency_bootstrap_ci95_low_ms": latency_ci[0],
                    "latency_bootstrap_ci95_high_ms": latency_ci[1],
                    "recall_mean": recall_mean,
                    "recall_bootstrap_ci95_low": recall_ci[0],
                    "recall_bootstrap_ci95_high": recall_ci[1],
                    "heldout_target_met": recall_mean >= target,
                }
            )
        for pair_no, (baseline, contender) in enumerate(pairs):
            baseline_values = [float(row[f"{baseline}_latency_mean_ms"]) for row in rows]
            contender_values = [float(row[f"{contender}_latency_mean_ms"]) for row in rows]
            observed_delta = statistics.fmean(
                contender_value - baseline_value
                for baseline_value, contender_value in zip(
                    baseline_values, contender_values, strict=True
                )
            )
            observed_speedup = statistics.fmean(baseline_values) / statistics.fmean(
                contender_values
            )
            rng = random.Random(
                _stable_seed(bootstrap_seed, "pair", cell_no, pair_no)
            )
            deltas: list[float] = []
            speedups: list[float] = []
            for _ in range(bootstrap_samples):
                sample = [rng.randrange(len(rows)) for _row in rows]
                base_mean = statistics.fmean(baseline_values[index] for index in sample)
                contender_mean = statistics.fmean(
                    contender_values[index] for index in sample
                )
                deltas.append(contender_mean - base_mean)
                speedups.append(base_mean / contender_mean)
            pairwise.append(
                {
                    "filter_name": filter_name,
                    "target_recall": target,
                    "baseline_arm": baseline,
                    "contender_arm": contender,
                    "paired_queries": len(rows),
                    "latency_delta_direction": f"{contender}_minus_{baseline}",
                    "latency_delta_mean_ms": observed_delta,
                    "latency_delta_bootstrap_ci95_low_ms": _percentile(deltas, 0.025),
                    "latency_delta_bootstrap_ci95_high_ms": _percentile(deltas, 0.975),
                    "speedup_direction": f"{baseline}_over_{contender}",
                    "speedup_mean": observed_speedup,
                    "speedup_bootstrap_ci95_low": _percentile(speedups, 0.025),
                    "speedup_bootstrap_ci95_high": _percentile(speedups, 0.975),
                }
            )
    return arm_summaries, pairwise


def controller_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.out_dir / "staging" / args.run_uuid
    return {
        "root": root,
        "manifest": args.manifest or root / "three_arm_controller.json",
        "selection": root / "three_arm_calibration_selection.csv",
        "calibration": root / "three_arm_calibration_raw.csv",
        "screen": root / "sqlens_full_screen_summary.csv",
        "final_raw": root / "three_arm_final_raw.csv",
        "paired": root / "three_arm_final_query_pairs.csv",
        "summary": root / "three_arm_final_summary.csv",
        "pairwise": root / "three_arm_final_pairwise.csv",
        "full_manifest": root / "sqlens_full_manifest.json",
        "full_final": root / "sqlens_full_final_raw.csv",
        "recovery_dir": root / "recovery",
        "published": (
            args.publish_path
            or args.out_dir
            / "published"
            / f"pgvector_three_arm_matched_recall_{args.tag}_{args.run_uuid}.json"
        ),
    }


def _docker_psql_scalar(args: argparse.Namespace, sql: str, appname: str) -> str:
    command = [
        "docker",
        "exec",
        "--env",
        f"PGAPPNAME={appname}",
        args.server_container,
        "psql",
        "--no-psqlrc",
        "--quiet",
        "--no-align",
        "--tuples-only",
        "--set",
        "ON_ERROR_STOP=1",
        "--username",
        args.pg_user,
        "--dbname",
        args.pg_database,
        "--command",
        sql,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = str(result.stderr or result.stdout or "").strip()
        raise RuntimeIdentityError(f"PostgreSQL identity query failed: {detail}")
    output = str(result.stdout or "").strip()
    if not output or "\n" in output:
        raise RuntimeIdentityError("PostgreSQL identity query returned an invalid scalar")
    return output


def verify_runtime_identity(
    args: argparse.Namespace,
    arm: str,
    source: Mapping[str, Any],
    binary_path: str,
) -> dict[str, Any]:
    """Verify the server file and loaded ABI after every arm restart."""
    if arm not in ARMS:
        raise RuntimeIdentityError(f"unknown arm {arm!r}")
    expected_sha = str(source.get("expected_digest", ""))
    observed_sha = binary_controller.server_binary_digest(
        args.server_container, binary_path
    )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or observed_sha != expected_sha:
        raise RuntimeIdentityError(
            f"{arm} vector.so SHA256 mismatch: expected {expected_sha}, got {observed_sha}"
        )
    extension_version = _docker_psql_scalar(
        args,
        "SELECT COALESCE((SELECT extversion FROM pg_extension "
        "WHERE extname='vector'), '');",
        f"pgvector-three-arm-identity-{arm}",
    )
    if extension_version != "0.8.2":
        raise RuntimeIdentityError(
            f"{arm} requires vector extension 0.8.2, got {extension_version!r}"
        )
    if arm == "official":
        build_id = f"official-pgvector-{extension_version}-sha256:{observed_sha}"
        build_method = "extension version plus exact loaded vector.so SHA256"
    else:
        build_id = _docker_psql_scalar(
            args,
            "SELECT vector_sqlens_build_id();",
            f"pgvector-three-arm-build-{arm}",
        )
        if build_id != args.expected_sqlens_build_id:
            raise RuntimeIdentityError(
                f"{arm} SQLens build ID mismatch: expected "
                f"{args.expected_sqlens_build_id!r}, got {build_id!r}"
            )
        build_method = "exact vector_sqlens_build_id() equality"
    return {
        "arm": arm,
        "checked_at_utc": utc_now(),
        "vector_so_path": binary_path,
        "expected_vector_so_sha256": expected_sha,
        "loaded_vector_so_sha256": observed_sha,
        "sha256_exact_match": True,
        "vector_extension_version": extension_version,
        "expected_build_id": (
            build_id if arm == "official" else args.expected_sqlens_build_id
        ),
        "loaded_build_id": build_id,
        "build_id_exact_match": True,
        "build_identity_method": build_method,
    }


def cache_protocol_spec(relations: Sequence[str]) -> dict[str, Any]:
    spec = {
        "version": CACHE_PROTOCOL_VERSION,
        "cache_state": "warm",
        "drop_os_caches": False,
        "postgres_restart_before_every_arm_invocation": True,
        "restart_effect": "clears PostgreSQL shared buffers; OS page cache is retained",
        "relation_prewarm": {
            "method": "synchronous pg_prewarm(regclass, 'read', 'main')",
            "relations": list(relations),
            "timing": "after runtime identity gate and before deterministic query warmup",
        },
        "query_warmup": {
            "official_and_disabled": "upstream runner fixed q0 warmup per filter",
            "sqlens_full": "target runner --warmup-all-queries before measured rows",
            "excluded_from_latency": True,
        },
        "measured_latency": "end-to-end query execution only; prewarm/warmup/output excluded",
    }
    return spec | {"sha256": sha256_json(spec)}


def execute_cache_protocol(
    args: argparse.Namespace, arm: str, execution_stage: str, block: int | None
) -> dict[str, Any]:
    spec = cache_protocol_spec(args.prewarm_relations)
    records: list[dict[str, Any]] = []
    for relation in args.prewarm_relations:
        upstream_runner.validate_identifier(relation)
        escaped = relation.replace("'", "''")
        output = _docker_psql_scalar(
            args,
            f"SELECT pg_prewarm('{escaped}'::regclass, 'read', 'main')::bigint;",
            f"pgvector-three-arm-prewarm-{arm}",
        )
        try:
            blocks = int(output)
        except ValueError as exc:
            raise ProtocolError(f"pg_prewarm returned a non-integer for {relation}") from exc
        if blocks <= 0:
            raise ProtocolError(f"pg_prewarm read no blocks for {relation}")
        records.append({"relation": relation, "blocks": blocks})
    return {
        "arm": arm,
        "execution_stage": execution_stage,
        "final_block": block,
        "completed_at_utc": utc_now(),
        "protocol_sha256": spec["sha256"],
        "records": records,
        "complete": len(records) == len(args.prewarm_relations),
    }


def source_specs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    official = binary_controller.source_spec(args, "official")
    disabled = binary_controller.source_spec(args, "sqlens_disabled")
    full = dict(disabled)
    full["implementation"] = "sqlens_full"
    return {"official": official, "sqlens_disabled": disabled, "sqlens_full": full}


def _runtime_from_upstream_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    binary = _required_mapping(
        manifest.get("server_binary_provenance"), "upstream binary provenance"
    )
    runtime = manifest.get("runtime_provenance")
    runtime_map = runtime if isinstance(runtime, Mapping) else {}
    sha = str(binary.get("vector_so_sha256", ""))
    arm = str(manifest.get("implementation", ""))
    build = str(runtime_map.get("loaded_vector_sqlens_build_id", ""))
    if arm == "official":
        extension = str(runtime_map.get("vector_extension_version", "0.8.2"))
        build = f"official-pgvector-{extension}-sha256:{sha}"
    if not re.fullmatch(r"[0-9a-f]{64}", sha) or not build:
        raise ProtocolError(f"upstream manifest runtime identity is incomplete for {arm}")
    return {"vector_so_sha256": sha, "build_id": build}


def canonicalize_upstream_rows(
    path: Path,
    arm: str,
    run_uuid: str,
    *,
    phase: str,
    runtime: Mapping[str, str],
    selections: Sequence[Mapping[str, Any]] | None = None,
    final_schedule: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    source_rows = read_csv(path)
    source_sha = sha256_file(path)
    schedule_positions = {
        (int(row["final_block"]), str(row["arm"])): int(row["position"])
        for row in final_schedule
    }
    selected_by_filter_config: dict[tuple[str, str], list[float]] = {}
    if selections is not None:
        for row in selections:
            if str(row["arm"]) == arm:
                selected_by_filter_config.setdefault(
                    (str(row["filter_name"]), str(row["config_label"])), []
                ).append(float(row["target_recall"]))
    output: list[dict[str, Any]] = []
    for row_no, source in enumerate(source_rows, start=2):
        if str(source.get("phase")) != phase:
            continue
        label = str(source.get("config_label", ""))
        filter_name = str(source.get("filter_name", ""))
        targets: Sequence[float | str] = (
            selected_by_filter_config.get((filter_name, label), [])
            if selections is not None
            else ("",)
        )
        if selections is not None and not targets:
            continue
        valid = _explicit_bool(source.get("valid", ""), "valid")
        if _explicit_bool(source.get("truth_self_excluded", ""), "truth_self_excluded") is not True:
            raise ProtocolError("upstream row is not bound to self-excluded truth")
        block_text = str(source.get("final_block", ""))
        block = int(block_text) if block_text not in {"", "None"} else None
        for target in targets:
            query_no = int(source["query_no"])
            repeat = int(source["repeat"])
            target_text = "" if target == "" else format(float(target), "g")
            pair_key = f"{filter_name}|{target_text}|q{query_no}|r{repeat}"
            output.append(
                {
                    "run_uuid": run_uuid,
                    "phase": "calibration" if phase == "verification" else phase,
                    "arm": arm,
                    "filter_name": filter_name,
                    "target_recall": target,
                    "query_no": query_no,
                    "query_id": int(source["query_id"]),
                    "repeat": repeat,
                    "final_block": "" if block is None else block,
                    "arm_order_position": (
                        "" if block is None else schedule_positions.get((block, arm), "")
                    ),
                    "config_label": label,
                    "ef_search": source.get("ef_search", ""),
                    "guided_collect_target": "",
                    "iterative_scan": source.get("iterative_scan", ""),
                    "max_scan_tuples": source.get("max_scan_tuples", ""),
                    "scan_mem_multiplier": source.get("scan_mem_multiplier", ""),
                    "latency_ms": _finite_float(source["latency_ms"], "latency_ms"),
                    "recall_at_10": _finite_float(
                        source["recall_at_10"], "recall_at_10"
                    ),
                    "valid": valid,
                    "error": str(source.get("error", "")),
                    "source_path": str(path),
                    "source_sha256": source_sha,
                    "source_row_no": row_no,
                    "vector_so_sha256": runtime["vector_so_sha256"],
                    "build_id": runtime["build_id"],
                    "pair_key": pair_key,
                    "measurement_key": f"{arm}|{pair_key}|{label}",
                }
            )
    return output


def canonicalize_target_rows(
    path: Path,
    run_uuid: str,
    config_label: str,
    runtime: Mapping[str, str],
    *,
    phase: str,
    target_recall: float | str = "",
    repeat_offset: int = 0,
    final_block: int | None = None,
    arm_order_position: int | str = "",
    config: Mapping[str, Any] | object | None = None,
) -> list[dict[str, Any]]:
    rows = read_csv(path)
    source_sha = sha256_file(path)
    config_values: dict[str, Any] = {}
    for key in (
        "ef_search",
        "guided_collect_target",
        "iterative_scan",
        "max_scan_tuples",
        "scan_mem_multiplier",
    ):
        if isinstance(config, Mapping):
            config_values[key] = config.get(key, "")
        else:
            config_values[key] = getattr(config, key, "")
    output: list[dict[str, Any]] = []
    for row_no, source in enumerate(rows, start=2):
        if str(source.get("mode", "")) != FULL_SQLENS_MODE:
            raise ProtocolError(f"full SQLens raw row uses unexpected mode in {path}")
        query_no = int(source["query_no"])
        repeat = int(source["repeat"]) + repeat_offset
        filter_name = str(source["filter_name"])
        target_text = (
            "" if target_recall == "" else format(float(target_recall), "g")
        )
        pair_key = f"{filter_name}|{target_text}|q{query_no}|r{repeat}"
        error = str(source.get("error", ""))
        if not error:
            if source.get("final_path") != "approximate_traversal_prioritization":
                raise ProtocolError(
                    f"full SQLens row did not use the r11 dual-frontier path in {path}"
                )
            if not _explicit_bool(
                source.get("planner_proof_succeeded", ""), "planner_proof_succeeded"
            ) or not _explicit_bool(
                source.get("approximate_ann_path", ""), "approximate_ann_path"
            ) or not _explicit_bool(
                source.get("approximate_prioritization_attempted", ""),
                "approximate_prioritization_attempted",
            ):
                raise ProtocolError(
                    f"full SQLens row lacks planner/approximate-path proof in {path}"
                )
            if int(source.get("match_frontier_pops", 0) or 0) + int(
                source.get("no_bridge_frontier_pops", 0) or 0
            ) <= 0:
                raise ProtocolError(
                    f"full SQLens row reports no dual-frontier work in {path}"
                )
            priority_reorders = int(source.get("priority_reorders", 0) or 0)
            if _explicit_bool(
                source.get("traversal_order_changed", ""),
                "traversal_order_changed",
            ) != (priority_reorders > 0):
                raise ProtocolError(
                    f"full SQLens row has inconsistent priority-reorder evidence in {path}"
                )
        output.append(
            {
                "run_uuid": run_uuid,
                "phase": phase,
                "arm": "sqlens_full",
                "filter_name": filter_name,
                "target_recall": target_recall,
                "query_no": query_no,
                "query_id": int(source["query_id"]),
                "repeat": repeat,
                "final_block": "" if final_block is None else final_block,
                "arm_order_position": arm_order_position,
                "config_label": config_label,
                **config_values,
                "latency_ms": _finite_float(source["end_to_end_ms"], "end_to_end_ms"),
                "recall_at_10": _finite_float(source["recall"], "recall"),
                "final_path": source.get("final_path", ""),
                "planner_proof_succeeded": source.get("planner_proof_succeeded", ""),
                "approximate_ann_path": source.get("approximate_ann_path", ""),
                "approximate_prioritization_attempted": source.get(
                    "approximate_prioritization_attempted", ""
                ),
                "traversal_order_changed": source.get(
                    "traversal_order_changed", ""
                ),
                "priority_reorders": int(
                    source.get("priority_reorders", 0) or 0
                ),
                "match_frontier_pops": int(source.get("match_frontier_pops", 0) or 0),
                "no_bridge_frontier_pops": int(source.get("no_bridge_frontier_pops", 0) or 0),
                "traversal_prioritization_burst": int(
                    source.get("traversal_prioritization_burst", 0) or 0
                ),
                "valid": not error,
                "error": error,
                "source_path": str(path),
                "source_sha256": source_sha,
                "source_row_no": row_no,
                "vector_so_sha256": runtime["vector_so_sha256"],
                "build_id": runtime["build_id"],
                "pair_key": pair_key,
                "measurement_key": f"sqlens_full|{pair_key}|{config_label}",
            }
        )
    return output


@contextmanager
def pg_environment(args: argparse.Namespace) -> Iterator[None]:
    values = {
        "PGHOST": str(args.pg_host),
        "PGPORT": str(args.pg_port),
        "PGDATABASE": str(args.pg_database),
        "PGUSER": str(args.pg_user),
        "PGAPPNAME": "pgvector-three-arm-sqlens-full",
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_full_runner_args(
    args: argparse.Namespace,
    filters: Sequence[str],
) -> argparse.Namespace:
    return argparse.Namespace(
        tag=args.tag,
        target_recalls=",".join(format(value, "g") for value in TARGET_RECALLS),
        target_recall=None,
        filters=list(filters),
        modes=[FULL_SQLENS_MODE],
        calibration_queries=100,
        calibration_repeats=CALIBRATION_REPEATS,
        calibration_query_offset=0,
        final_queries=len(FINAL_QUERY_NOS),
        final_repeats=FINAL_REPEATS,
        final_query_offset=FINAL_QUERY_NOS[0],
        final_execution_order="interleaved",
        schedule_seed=args.schedule_seed,
        allow_overlapping_query_splits=False,
        ef_search_values=args.full_ef_search_values,
        guided_collect_target_values=args.full_guided_collect_target_values,
        max_scan_tuples_values=args.full_max_scan_tuples_values,
        scan_mem_multiplier_values=args.full_scan_mem_multiplier_values,
        iterative_scan_values="off",
        stock_iterative_scan_values="off",
        filters_csv=args.filters_csv,
        truth_csv=args.truth_csv,
        insertion_table=args.table,
        insertion_index=args.source_index,
        bfs_table=args.table,
        bfs_index=args.clone_index,
        query_table=args.query_table,
        query_id_column=args.query_id_column,
        query_vector_column=args.query_vector_column,
        candidate_validity_predicate=CANDIDATE_VALIDITY_PREDICATE,
        candidate_validity_predicate_explicit=True,
        expected_truth_self_excluded=True,
        guidance_filter_strategy="traversal_guided",
        traversal_guided_prioritization=True,
        traversal_guided_burst=args.traversal_guided_burst,
        guidance_selectivity_max_pct=100.0,
        guidance_max_atoms=args.guidance_max_atoms,
        d2_page_access=args.d2_page_access,
        d2_index_page_access=args.d2_index_page_access,
        preferred_index_guc=args.preferred_index_guc,
        require_preferred_index_guc=True,
        d1_cache_mb=args.d1_cache_mb,
        d3_cache_mb=args.d3_cache_mb,
        backend_cpu_list=args.backend_cpu_list,
        warmup_all_queries=True,
        force_hnsw=True,
        resume=args.resume,
        skip_final=True,
        statement_timeout_ms=args.statement_timeout_ms,
        progress_queries=args.progress_queries,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


def _target_row_complete(row: Mapping[str, Any]) -> bool:
    return (
        int(row.get("ok", 0) or 0) > 0
        and int(row.get("errors", 0) or 0) == 0
        and (row.get("rows_complete") is True or str(row.get("rows_complete", "")).lower() == "true")
    )


def promote_full_sqlens_configs(
    screen_rows: Sequence[Mapping[str, Any]],
    configs: Sequence[target_runner.Config],
    filter_name: str,
    margin: float,
) -> tuple[list[target_runner.Config], list[dict[str, Any]]]:
    candidates = [row for row in screen_rows if str(row.get("filter_name")) == filter_name]
    by_label = {str(row.get("config")): row for row in candidates}
    config_by_label = {config.label: config for config in configs}
    if set(by_label) != set(config_by_label) or any(
        not _target_row_complete(row) for row in candidates
    ):
        raise ProtocolError(f"full SQLens screening is incomplete for {filter_name}")
    reasons: dict[str, set[str]] = {}

    def add(label: str, reason: str) -> None:
        reasons.setdefault(label, set()).add(reason)

    for target in TARGET_RECALLS:
        threshold = max(0.0, target - margin)
        eligible = [
            row for row in candidates if float(row["recall_mean"]) >= threshold
        ]
        if eligible:
            winner = min(
                eligible,
                key=lambda row: (float(row["latency_mean_ms"]), str(row["config"])),
            )
            add(str(winner["config"]), f"fastest_target_{target:g}_minus_margin")
    max_recall = max(
        candidates,
        key=lambda row: (
            float(row["recall_mean"]),
            -float(row["latency_mean_ms"]),
            str(row["config"]),
        ),
    )
    add(str(max_recall["config"]), "maximum_screen_recall")
    max_budget = max(
        configs,
        key=lambda config: (
            config.ef_search,
            config.guided_collect_target,
            config.max_scan_tuples,
            config.scan_mem_multiplier,
            config.label,
        ),
    )
    add(max_budget.label, "declared_grid_maximum_budget")
    promoted = [config for config in configs if config.label in reasons]
    proof = [
        {
            "filter_name": filter_name,
            "config_label": config.label,
            "promotion_reasons": "|".join(sorted(reasons[config.label])),
            "screen_recall_mean": by_label[config.label]["recall_mean"],
            "screen_latency_mean_ms": by_label[config.label]["latency_mean_ms"],
        }
        for config in promoted
    ]
    if not promoted:
        raise ProtocolError(f"screening promoted no full SQLens config for {filter_name}")
    return promoted, proof


def _stable_database_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    relations = value.get("relations")
    stable_relations: dict[str, Any] = {}
    if isinstance(relations, Mapping):
        for name, raw in sorted(relations.items()):
            relation = _required_mapping(raw, f"database relation {name}")
            stable_relations[str(name)] = {
                key: relation.get(key)
                for key in (
                    "oid",
                    "relfilenode",
                    "valid",
                    "ready",
                    "indpred",
                    "candidate_validity_predicate_sha256",
                    "candidate_validity_predicate_matches",
                )
            }
    query = value.get("query_table")
    query_map = query if isinstance(query, Mapping) else {}
    return {
        "relations": stable_relations,
        "candidate_validity_predicate": value.get(
            "candidate_validity_predicate"
        ),
        "candidate_validity_predicate_sha256": value.get(
            "candidate_validity_predicate_sha256"
        ),
        "query_table": {
            key: query_map.get(key) for key in ("name", "oid", "relfilenode", "row_count", "columns")
        },
    }


def _bind_full_runtime(
    args: argparse.Namespace,
    full_args: argparse.Namespace,
    controller_graph: Mapping[str, Any],
    expected_runtime: Mapping[str, str],
) -> tuple[Any, dict[str, Any]]:
    tracking = target_runner.prepare_fragment_tracking(full_args)
    full_args.fragment_tracking_prepared = bool(tracking["prepared"])
    full_args.fragment_tracking_evidence = tracking
    guard, guard_evidence = target_runner.acquire_formal_data_guard(full_args)
    try:
        run_spec = target_runner.build_run_spec(full_args)
    except BaseException:
        guard.rollback()
        guard.close()
        raise
    runtime = _required_mapping(
        run_spec.get("sqlens_runtime_provenance"), "full SQLens runtime provenance"
    )
    observed_sha = str(runtime.get("loaded_vector_so_sha256", ""))
    observed_build = str(runtime.get("loaded_vector_sqlens_build_id", ""))
    if (
        observed_sha != expected_runtime["vector_so_sha256"]
        or observed_build != expected_runtime["build_id"]
        or observed_build != args.expected_sqlens_build_id
    ):
        guard.rollback()
        guard.close()
        raise RuntimeIdentityError(
            "full SQLens child runtime is not bound to the controller SHA/build ID"
        )
    graph = _required_mapping(run_spec.get("d2_graph_proof"), "live D2 graph proof")
    if graph.get("stable_fingerprint_sha256") != controller_graph.get(
        "stable_fingerprint_sha256"
    ):
        guard.rollback()
        guard.close()
        raise ProtocolError("live full SQLens D2 graph proof differs from the M32 input proof")
    full_args.d2_graph_proof = graph
    full_args.sqlens_runtime_provenance = dict(runtime)
    full_args.database_fingerprint = run_spec["database"]
    full_args.run_spec_hash = str(run_spec["run_spec_hash"])
    return guard, {
        "run_spec": run_spec,
        "formal_data_guard": guard_evidence,
        "fragment_tracking_preparation": tracking,
    }


def _full_runtime_mapping(run_spec: Mapping[str, Any]) -> dict[str, str]:
    runtime = _required_mapping(
        run_spec.get("sqlens_runtime_provenance"), "full SQLens runtime provenance"
    )
    sha = str(runtime.get("loaded_vector_so_sha256", ""))
    build = str(runtime.get("loaded_vector_sqlens_build_id", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", sha) or not build:
        raise ProtocolError("full SQLens run spec has incomplete runtime identity")
    return {"vector_so_sha256": sha, "build_id": build}


def _validate_full_resume_manifest(
    path: Path, controller_spec_hash: str
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("controller_spec_sha256") != controller_spec_hash:
        raise CheckpointError("full SQLens checkpoint belongs to a different controller spec")
    if manifest.get("status") not in {
        "calibration_complete",
        "final_in_progress",
        "final_failed",
        "arm_ready",
    }:
        raise CheckpointError("full SQLens checkpoint did not complete calibration")
    outputs = _required_mapping(manifest.get("outputs"), "full SQLens outputs")
    calibration_path = validate_artifact_entry(
        _required_mapping(outputs.get("calibration"), "full calibration artifact")
    )
    selection_path = validate_artifact_entry(
        _required_mapping(outputs.get("selection"), "full selection artifact")
    )
    return manifest, read_csv(calibration_path), read_csv(selection_path)


def run_full_sqlens_calibration(
    args: argparse.Namespace,
    filters: Sequence[str],
    controller_graph: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    controller_spec_hash: str,
) -> dict[str, Any]:
    paths = controller_paths(args)
    manifest_path = paths["full_manifest"]
    prior_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"full SQLens checkpoint exists: {manifest_path}")
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_manifest.get("controller_spec_sha256") != controller_spec_hash:
            raise CheckpointError(
                "full SQLens checkpoint belongs to a different controller spec"
            )
        if prior_manifest.get("status") != "failed":
            manifest, calibration_rows, selections = _validate_full_resume_manifest(
                manifest_path, controller_spec_hash
            )
            return {
                "manifest": manifest,
                "calibration_rows": calibration_rows,
                "selections": selections,
                "resumed": True,
            }

    full_args = build_full_runner_args(args, filters)
    full_root = paths["root"] / "sqlens_full"
    previous_results = target_runner.RESULTS
    target_runner.RESULTS = full_root
    guard = None
    manifest: dict[str, Any] = {
        **(prior_manifest or {}),
        "schema_version": SCHEMA_VERSION,
        "artifact": "amazon10m_pgvector_three_arm_sqlens_full",
        "arm": "sqlens_full",
        "run_uuid": args.run_uuid,
        "status": "running",
        "artifact_valid": False,
        "controller_spec_sha256": controller_spec_hash,
        "started_at_utc": (prior_manifest or {}).get("started_at_utc", utc_now()),
        "last_attempt_started_at_utc": utc_now(),
        "protocol": formal_protocol(),
        "calibration_selection": {
            "policy": CALIBRATION_SELECTION_POLICY,
            "rule": CALIBRATION_SELECTION_RULE,
        },
        "outputs": {},
    }
    atomic_write_json(manifest_path, manifest)
    try:
        with pg_environment(args):
            guard, binding = _bind_full_runtime(
                args,
                full_args,
                controller_graph,
                {
                    "vector_so_sha256": str(
                        runtime_identity["loaded_vector_so_sha256"]
                    ),
                    "build_id": str(runtime_identity["loaded_build_id"]),
                },
            )
            run_spec = _required_mapping(binding["run_spec"], "full run spec")
            runtime = _full_runtime_mapping(run_spec)
            configs = target_runner.build_configs(full_args)
            if not configs:
                raise ProtocolError("full SQLens configuration grid is empty")
            manifest.update(
                {
                    **binding,
                    "run_spec_sha256": run_spec["run_spec_hash"],
                    "phase_execution_contract": {
                        "binding_run_spec_scope": (
                            "immutable database/index/binary/query-universe binding; the "
                            "controller supplies the disjoint execution slices below"
                        ),
                        "binding_query_union": {
                            "calibration": list(range(0, 100)),
                            "final": list(FINAL_QUERY_NOS),
                        },
                        "screen": {
                            "query_nos": list(SCREEN_QUERY_NOS),
                            "repeats": SCREEN_REPEATS,
                        },
                        "calibration": {
                            "query_nos": list(CALIBRATION_QUERY_NOS),
                            "repeats": CALIBRATION_REPEATS,
                        },
                        "final": {
                            "query_nos": list(FINAL_QUERY_NOS),
                            "repeats": FINAL_REPEATS,
                            "blocks": FINAL_BLOCKS,
                            "repeats_per_block": FINAL_REPEATS_PER_BLOCK,
                        },
                        "screen_calibration_final_disjoint": True,
                    },
                    "runtime_identity": runtime_identity,
                    "configuration_grid": [asdict(config) for config in configs],
                    "screen_policy": {
                        "query_nos": list(SCREEN_QUERY_NOS),
                        "repeats": SCREEN_REPEATS,
                        "complete_grid_required": True,
                        "promotion_margin": args.promotion_margin,
                    },
                    "calibration_policy": {
                        "query_nos": list(CALIBRATION_QUERY_NOS),
                        "repeats": CALIBRATION_REPEATS,
                        "all_promoted_configs_required": True,
                        "selection": formal_protocol()["selection_rule"],
                    },
                }
            )
            atomic_write_json(manifest_path, manifest)

            screen_rows: list[dict[str, Any]] = []
            promotion_rows: list[dict[str, Any]] = []
            promoted_by_filter: dict[str, list[target_runner.Config]] = {}
            full_args.calibration_query_offset = SCREEN_QUERY_NOS[0]
            full_args.calibration_queries = len(SCREEN_QUERY_NOS)
            full_args.calibration_repeats = SCREEN_REPEATS
            full_args.tag = f"{args.tag}_screen"
            for filter_name in filters:
                rows, evidence = target_runner.calibrate_mode_filter(
                    filter_name,
                    FULL_SQLENS_MODE,
                    configs,
                    full_args,
                    [math.nextafter(1.0, math.inf)],
                )
                if evidence.get("calibration_failed") is True:
                    raise ProtocolError(f"full SQLens screening failed for {filter_name}")
                screen_rows.extend(rows)
                promoted, proof = promote_full_sqlens_configs(
                    rows, configs, filter_name, args.promotion_margin
                )
                promoted_by_filter[filter_name] = promoted
                promotion_rows.extend(proof)

            calibration_summaries: list[dict[str, Any]] = []
            canonical_calibration: list[dict[str, Any]] = []
            full_args.calibration_query_offset = CALIBRATION_QUERY_NOS[0]
            full_args.calibration_queries = len(CALIBRATION_QUERY_NOS)
            full_args.calibration_repeats = CALIBRATION_REPEATS
            full_args.tag = f"{args.tag}_calibration"
            for filter_name in filters:
                rows, evidence = target_runner.calibrate_mode_filter(
                    filter_name,
                    FULL_SQLENS_MODE,
                    promoted_by_filter[filter_name],
                    full_args,
                    [math.nextafter(1.0, math.inf)],
                )
                if evidence.get("calibration_failed") is True:
                    raise ProtocolError(f"full SQLens calibration failed for {filter_name}")
                calibration_summaries.extend(rows)
                for row in rows:
                    raw = Path(str(row["raw"]))
                    target_runner.require_plan_evidence(
                        raw,
                        CANDIDATE_VALIDITY_PREDICATE,
                        full_args.database_fingerprint,
                    )
                    raw_rows = canonicalize_target_rows(
                        raw,
                        args.run_uuid,
                        str(row["config"]),
                        runtime,
                        phase="calibration",
                        config=row,
                    )
                    summarize_config_measurements(
                        raw_rows,
                        expected_query_nos=CALIBRATION_QUERY_NOS,
                        expected_repeats=CALIBRATION_REPEATS,
                    )
                    canonical_calibration.extend(raw_rows)

            selections = select_calibrated_configs(
                canonical_calibration,
                list(filters),
                arms=("sqlens_full",),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + 100_000,
            )
            screen_path = paths["screen"]
            promotion_path = full_root / "promotion.csv"
            calibration_path = full_root / "calibration_raw.csv"
            calibration_summary_path = full_root / "calibration_summary.csv"
            selection_path = full_root / "selection.csv"
            write_csv_atomic(screen_path, screen_rows)
            write_csv_atomic(promotion_path, promotion_rows)
            write_csv_atomic(calibration_path, canonical_calibration, CANONICAL_FIELDS)
            write_csv_atomic(calibration_summary_path, calibration_summaries)
            write_csv_atomic(selection_path, selections)
            manifest.update(
                {
                    "status": "calibration_complete",
                    "artifact_valid": True,
                    "calibration_completed_at_utc": utc_now(),
                    "promotion_proof": promotion_rows,
                    "outputs": {
                        "screen": artifact_entry(screen_path, rows=len(screen_rows)),
                        "promotion": artifact_entry(
                            promotion_path, rows=len(promotion_rows)
                        ),
                        "calibration": artifact_entry(
                            calibration_path, rows=len(canonical_calibration)
                        ),
                        "calibration_summary": artifact_entry(
                            calibration_summary_path,
                            rows=len(calibration_summaries),
                        ),
                        "selection": artifact_entry(
                            selection_path, rows=len(selections)
                        ),
                    },
                    "completed_final_blocks": [],
                }
            )
            atomic_write_json(manifest_path, manifest)
            return {
                "manifest": manifest,
                "calibration_rows": canonical_calibration,
                "selections": selections,
                "resumed": False,
            }
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "artifact_valid": False,
                "failed_at_utc": utc_now(),
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        if guard is not None:
            guard.rollback()
            guard.close()
        target_runner.RESULTS = previous_results


def _config_from_selection(row: Mapping[str, Any]) -> target_runner.Config:
    return target_runner.Config(
        ef_search=int(row["ef_search"]),
        max_scan_tuples=int(row["max_scan_tuples"]),
        scan_mem_multiplier=float(row["scan_mem_multiplier"]),
        iterative_scan=str(row["iterative_scan"]),
        guided_collect_target=int(row["guided_collect_target"]),
    )


def run_full_sqlens_final_block(
    args: argparse.Namespace,
    filters: Sequence[str],
    controller_graph: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    controller_spec_hash: str,
    selections: Sequence[Mapping[str, Any]],
    block: int,
    position: int,
) -> dict[str, Any]:
    if block not in range(FINAL_BLOCKS):
        raise ProtocolError(f"invalid full SQLens final block {block}")
    paths = controller_paths(args)
    manifest, _calibration, checkpoint_selections = _validate_full_resume_manifest(
        paths["full_manifest"], controller_spec_hash
    )
    selected_rows = [
        dict(row) for row in selections if str(row.get("arm")) == "sqlens_full"
    ]
    if not _same_selection_matrix(selected_rows, checkpoint_selections):
        raise CheckpointError("full SQLens final selections differ from calibration checkpoint")
    completed = {int(value) for value in manifest.get("completed_final_blocks", [])}
    block_path = paths["root"] / "sqlens_full" / f"final_block_{block}.csv"
    if block in completed:
        entry = _required_mapping(
            _required_mapping(manifest.get("final_block_artifacts"), "full final artifacts").get(
                str(block)
            ),
            f"full final block {block}",
        )
        validate_artifact_entry(entry)
        return {"artifact": dict(entry), "resumed": True}
    if block_path.exists():
        raise CheckpointError(f"unclaimed full SQLens final block exists: {block_path}")

    full_args = build_full_runner_args(args, filters)
    full_root = paths["root"] / "sqlens_full" / f"block_{block}"
    previous_results = target_runner.RESULTS
    target_runner.RESULTS = full_root
    guard = None
    try:
        with pg_environment(args):
            guard, binding = _bind_full_runtime(
                args,
                full_args,
                controller_graph,
                {
                    "vector_so_sha256": str(
                        runtime_identity["loaded_vector_so_sha256"]
                    ),
                    "build_id": str(runtime_identity["loaded_build_id"]),
                },
            )
            run_spec = _required_mapping(binding["run_spec"], "full final run spec")
            calibration_run_spec = _required_mapping(
                manifest.get("run_spec"), "full calibration run spec"
            )
            if _stable_database_identity(
                _required_mapping(run_spec.get("database"), "full final database")
            ) != _stable_database_identity(
                _required_mapping(
                    calibration_run_spec.get("database"), "full calibration database"
                )
            ):
                raise CheckpointError("database identity changed since full SQLens calibration")
            runtime = _full_runtime_mapping(run_spec)
            full_args.final_query_offset = FINAL_QUERY_NOS[0]
            full_args.final_queries = len(FINAL_QUERY_NOS)
            full_args.final_repeats = FINAL_REPEATS_PER_BLOCK
            full_args.tag = f"{args.tag}_final_b{block}"

            targets_by_filter_config: dict[tuple[str, str], list[float]] = {}
            config_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
            for row in selected_rows:
                key = (str(row["filter_name"]), str(row["config_label"]))
                targets_by_filter_config.setdefault(key, []).append(
                    float(row["target_recall"])
                )
                config_rows.setdefault(key, row)

            canonical: list[dict[str, Any]] = []
            evidence: list[dict[str, Any]] = []
            for filter_name, config_label in sorted(config_rows):
                selection = config_rows[(filter_name, config_label)]
                config = _config_from_selection(selection)
                raw = full_root / f"{filter_name}_{config.label}.csv"
                log = full_root / "logs" / f"{filter_name}_{config.label}.log"
                summary = (
                    target_runner.reusable_summary(
                        raw,
                        args.bootstrap_samples,
                        args.bootstrap_seed,
                        len(FINAL_QUERY_NOS),
                        FINAL_REPEATS_PER_BLOCK,
                        True,
                    )
                    if args.resume
                    else None
                )
                if summary is not None:
                    try:
                        target_runner.require_plan_evidence(
                            raw,
                            CANDIDATE_VALIDITY_PREDICATE,
                            full_args.database_fingerprint,
                        )
                    except RuntimeError:
                        summary = None
                if summary is None:
                    if raw.exists():
                        raise CheckpointError(
                            f"unusable unclaimed full final raw exists: {raw}"
                        )
                    target_runner.run_d123(
                        raw,
                        filter_name,
                        FULL_SQLENS_MODE,
                        FINAL_QUERY_NOS[0],
                        len(FINAL_QUERY_NOS),
                        FINAL_REPEATS_PER_BLOCK,
                        config,
                        full_args,
                        log,
                    )
                    summary = target_runner.summarize_raw(
                        raw,
                        args.bootstrap_samples,
                        args.bootstrap_seed,
                        len(FINAL_QUERY_NOS),
                        FINAL_REPEATS_PER_BLOCK,
                        True,
                    )
                if len(summary) != 1 or not summary[0].get("rows_complete"):
                    raise ProtocolError(f"full SQLens final raw is incomplete: {raw}")
                plan = target_runner.plan_evidence_manifest_entry(
                    raw,
                    CANDIDATE_VALIDITY_PREDICATE,
                    full_args.database_fingerprint,
                )
                evidence.append(plan)
                for target in targets_by_filter_config[(filter_name, config_label)]:
                    canonical.extend(
                        canonicalize_target_rows(
                            raw,
                            args.run_uuid,
                            config_label,
                            runtime,
                            phase="final",
                            target_recall=target,
                            repeat_offset=block * FINAL_REPEATS_PER_BLOCK,
                            final_block=block,
                            arm_order_position=position,
                            config=config,
                        )
                    )
            write_csv_atomic(block_path, canonical, CANONICAL_FIELDS)
            entry = artifact_entry(block_path, rows=len(canonical))
            block_artifacts = dict(manifest.get("final_block_artifacts", {}))
            block_artifacts[str(block)] = entry
            completed.add(block)
            manifest.update(
                {
                    "status": (
                        "arm_ready" if completed == set(range(FINAL_BLOCKS)) else "final_in_progress"
                    ),
                    "completed_final_blocks": sorted(completed),
                    "final_block_artifacts": block_artifacts,
                    "final_plan_evidence": [
                        *manifest.get("final_plan_evidence", []),
                        *evidence,
                    ],
                    "last_final_block_completed_at_utc": utc_now(),
                }
            )
            atomic_write_json(paths["full_manifest"], manifest)
            return {"artifact": entry, "resumed": False}
    except BaseException as exc:
        manifest.update(
            {
                "status": "final_failed",
                "artifact_valid": False,
                "failed_at_utc": utc_now(),
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        atomic_write_json(paths["full_manifest"], manifest)
        raise
    finally:
        if guard is not None:
            guard.rollback()
            guard.close()
        target_runner.RESULTS = previous_results


def load_upstream_calibration_rows(
    args: argparse.Namespace, arm: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = upstream_runner.output_paths(args.out_dir, arm, args.tag, args.run_uuid)
    if not paths["manifest"].is_file() or not paths["raw"].is_file():
        raise CheckpointError(f"upstream calibration artifacts are missing for {arm}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("implementation") != arm or manifest.get("run_uuid") != args.run_uuid:
        raise CheckpointError(f"upstream calibration manifest identity mismatch for {arm}")
    if manifest.get("status") not in {
        "calibration_complete",
        "final_in_progress",
        "arm_ready",
        "staging_unconfirmed",
    }:
        raise CheckpointError(f"upstream calibration is incomplete for {arm}")
    runtime = _runtime_from_upstream_manifest(manifest)
    rows = canonicalize_upstream_rows(
        paths["raw"],
        arm,
        args.run_uuid,
        phase="verification",
        runtime=runtime,
    )
    return rows, manifest


def assert_upstream_selection_matches(
    manifest: Mapping[str, Any], selections: Sequence[Mapping[str, Any]], arm: str
) -> None:
    calibration_contract = _required_mapping(
        manifest.get("calibration_selection"), f"{arm} calibration selection"
    )
    if calibration_contract.get("policy") != CALIBRATION_SELECTION_POLICY:
        raise ProtocolError(f"{arm} calibration selection policy mismatch")
    recorded = _required_mapping(
        manifest.get("target_selection"), f"{arm} target selection"
    )
    for row in selections:
        if str(row.get("arm")) != arm:
            continue
        filter_name = str(row["filter_name"])
        target = format(float(row["target_recall"]), "g")
        filter_values = _required_mapping(recorded.get(filter_name), f"{arm}/{filter_name}")
        value = filter_values.get(target)
        if isinstance(value, Mapping):
            label = str(value.get("config_label", ""))
        elif isinstance(value, str) and value.startswith("ef"):
            label = value
        else:
            label = ""
        if label != str(row["config_label"]):
            raise ProtocolError(
                f"unified selection differs from upstream runner for "
                f"{arm}/{filter_name}/{target}"
            )
        if not isinstance(value, Mapping):
            raise ProtocolError(
                f"{arm}/{filter_name}/{target} lacks audited selection metadata"
            )
        if value.get("calibration_selection_policy") != CALIBRATION_SELECTION_POLICY:
            raise ProtocolError(
                f"{arm}/{filter_name}/{target} selection policy mismatch"
            )
        if str(value.get("selection_fallback", "")) != str(
            row.get("selection_fallback", "")
        ):
            raise ProtocolError(
                f"{arm}/{filter_name}/{target} selection fallback mismatch"
            )
        if _finite_float(
            value.get("verification_recall_lcb95"),
            "verification_recall_lcb95",
        ) != _finite_float(row.get("recall_lcb95"), "recall_lcb95"):
            raise ProtocolError(
                f"{arm}/{filter_name}/{target} selection LCB95 mismatch"
            )


def load_upstream_final_rows(
    args: argparse.Namespace,
    arm: str,
    selections: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    paths = upstream_runner.output_paths(args.out_dir, arm, args.tag, args.run_uuid)
    if not paths["manifest"].is_file() or not paths["raw"].is_file():
        raise CheckpointError(f"upstream final artifacts are missing for {arm}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("implementation") != arm or manifest.get("run_uuid") != args.run_uuid:
        raise CheckpointError(f"upstream final manifest identity mismatch for {arm}")
    if manifest.get("status") not in {"arm_ready", "staging_unconfirmed"}:
        raise CheckpointError(f"upstream final arm {arm} is incomplete")
    if set(map(int, manifest.get("completed_final_blocks", []))) != set(
        range(FINAL_BLOCKS)
    ):
        raise CheckpointError(f"upstream final block coverage is incomplete for {arm}")
    schedule_contract = _required_mapping(
        manifest.get("schedule_contract"), f"{arm} schedule contract"
    )
    if (
        int(schedule_contract.get("final_blocks", 0)) != FINAL_BLOCKS
        or int(schedule_contract.get("final_repeats", 0)) != FINAL_REPEATS
    ):
        raise CheckpointError(f"upstream final schedule contract mismatch for {arm}")
    raw_identity = _required_mapping(
        _required_mapping(manifest.get("output_hashes"), f"{arm} output hashes").get(
            "raw"
        ),
        f"{arm} raw output hash",
    )
    recorded_path = Path(str(raw_identity.get("path", "")))
    if recorded_path.resolve() != paths["raw"].resolve():
        raise CheckpointError(f"upstream final raw path mismatch for {arm}")
    recorded_sha = str(raw_identity.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", recorded_sha) or sha256_file(
        paths["raw"]
    ) != recorded_sha:
        raise CheckpointError(f"upstream final raw SHA256 mismatch for {arm}")
    runtime = _runtime_from_upstream_manifest(manifest)
    rows = canonicalize_upstream_rows(
        paths["raw"],
        arm,
        args.run_uuid,
        phase="final",
        runtime=runtime,
        selections=selections,
        final_schedule=schedule,
    )
    validate_canonical_final_arm_rows(rows, arm, selections, schedule)
    return rows


def validate_canonical_final_arm_rows(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    selections: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
) -> None:
    selected = {
        (str(row["filter_name"]), float(row["target_recall"])): str(
            row["config_label"]
        )
        for row in selections
        if str(row.get("arm")) == arm
    }
    expected = {
        (filter_name, target, query_no, repeat)
        for filter_name, target in selected
        for query_no in FINAL_QUERY_NOS
        for repeat in range(FINAL_REPEATS)
    }
    observed: list[tuple[str, float, int, int]] = []
    positions = {
        (int(row["final_block"]), str(row["arm"])): int(row["position"])
        for row in schedule
    }
    for row in rows:
        key = (
            str(row["filter_name"]),
            float(row["target_recall"]),
            int(row["query_no"]),
            int(row["repeat"]),
        )
        observed.append(key)
        repeat = key[3]
        block = int(row["final_block"])
        if block != repeat // FINAL_REPEATS_PER_BLOCK:
            raise CheckpointError(f"{arm} final row repeat/block mapping is invalid")
        if int(row["arm_order_position"]) != positions[(block, arm)]:
            raise CheckpointError(f"{arm} final row arm-order position is invalid")
        if str(row["config_label"]) != selected.get((key[0], key[1])):
            raise CheckpointError(f"{arm} final row uses a non-selected config")
        if _explicit_bool(row.get("valid", ""), "valid") is not True or str(
            row.get("error", "")
        ):
            raise CheckpointError(f"{arm} final row is invalid")
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise CheckpointError(
            f"{arm} final query/repeat coverage is not exact: "
            f"observed={len(observed)} unique={len(set(observed))} expected={len(expected)}"
        )


def load_full_final_rows(
    args: argparse.Namespace,
    selections: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest = json.loads(
        controller_paths(args)["full_manifest"].read_text(encoding="utf-8")
    )
    if manifest.get("status") != "arm_ready" or set(
        map(int, manifest.get("completed_final_blocks", []))
    ) != set(range(FINAL_BLOCKS)):
        raise CheckpointError("full SQLens final arm is incomplete")
    rows: list[dict[str, Any]] = []
    artifacts = _required_mapping(
        manifest.get("final_block_artifacts"), "full final block artifacts"
    )
    for block in range(FINAL_BLOCKS):
        path = validate_artifact_entry(
            _required_mapping(artifacts.get(str(block)), f"full final block {block}")
        )
        rows.extend(read_csv(path))
    validate_canonical_final_arm_rows(rows, "sqlens_full", selections, schedule)
    return rows


def _normalized_controller_args(args: argparse.Namespace) -> dict[str, Any]:
    ignored = {
        "dry_run",
        "resume",
        "manifest",
        "publish_path",
        "recovery_journal",
    }
    return {
        key: (
            [str(item) if isinstance(item, Path) else item for item in value]
            if isinstance(value, list)
            else str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in sorted(vars(args).items())
        if key not in ignored
    }


def build_controller_spec(
    args: argparse.Namespace,
    filters: Sequence[Mapping[str, str]],
    graph: Mapping[str, Any],
    truth: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    schedule, schedule_audit = rotating_final_schedule(
        args.run_uuid, args.schedule_seed
    )
    cache = cache_protocol_spec(args.prewarm_relations)
    module_paths = {
        "controller": Path(__file__),
        "binary_controller": Path(binary_controller.__file__),
        "upstream_runner": Path(upstream_runner.__file__),
        "target_runner": Path(target_runner.__file__),
    }
    spec = {
        "schema_version": SCHEMA_VERSION,
        "controller": CONTROLLER_NAME,
        "run_uuid": args.run_uuid,
        "args": _normalized_controller_args(args),
        "protocol": formal_protocol(),
        "filters": [
            {
                "filter_name": row["filter_name"],
                "predicate": row["predicate"],
                "target_rate": row["target_rate"],
            }
            for row in filters
        ],
        "truth_contract": dict(truth),
        "graph_contract": dict(graph),
        "binary_sources": {arm: dict(sources[arm]) for arm in ARMS},
        "expected_sqlens_build_id": args.expected_sqlens_build_id,
        "cache_protocol": cache,
        "calibration_order": calibration_order(args.run_uuid, args.schedule_seed),
        "final_schedule": schedule,
        "final_schedule_audit": schedule_audit,
        "source_hashes": {
            name: sha256_file(path) for name, path in module_paths.items()
        }
        | {
            "filters": sha256_file(args.filters_csv),
            "truth": sha256_file(args.truth_csv),
            "graph_identity": sha256_file(args.graph_identity_json),
        },
    }
    spec["controller_spec_sha256"] = sha256_json(spec)
    return spec


def validate_formal_args(args: argparse.Namespace) -> None:
    args.target_recalls = list(map(float, args.target_recalls))
    if tuple(args.target_recalls) != TARGET_RECALLS:
        raise ProtocolError("formal targets must be exactly 0.90,0.95,0.99")
    if (
        args.screen_repeats != SCREEN_REPEATS
        or args.verification_repeats != CALIBRATION_REPEATS
        or args.final_repeats != FINAL_REPEATS
    ):
        raise ProtocolError("formal repeats must be screen r1, calibration r2, final r6")
    if args.hnsw_m != HNSW_M:
        raise ProtocolError(f"formal source/BFS graph must use M={HNSW_M}")
    if args.candidate_validity_predicate.strip() != CANDIDATE_VALIDITY_PREDICATE:
        raise ProtocolError("formal candidate validity predicate must be embedding_valid")
    if not args.expected_sqlens_build_id or not args.expected_sqlens_build_id.startswith(
        REQUIRED_TRAVERSAL_BUILD_PREFIX
    ):
        raise ProtocolError(
            "formal SQLens requires the current profiled D1/D2/D3 build ID beginning with "
            f"{REQUIRED_TRAVERSAL_BUILD_PREFIX!r}"
        )
    if args.required_sqlens_build_prefix != REQUIRED_TRAVERSAL_BUILD_PREFIX:
        raise ProtocolError("formal SQLens build-prefix gate cannot be relaxed")
    if not 1 <= args.traversal_guided_burst <= 1024:
        raise ProtocolError("traversal-guided burst must be in [1, 1024]")
    if args.full_mode != FULL_SQLENS_MODE:
        raise ProtocolError(f"formal full SQLens mode must be {FULL_SQLENS_MODE}")
    if not args.prewarm_relations:
        args.prewarm_relations = [args.table, args.source_index, args.clone_index]
    if len(set(args.prewarm_relations)) != len(args.prewarm_relations):
        raise ProtocolError("cache prewarm relation list contains duplicates")
    for relation in args.prewarm_relations:
        upstream_runner.validate_identifier(relation)
    if not math.isfinite(args.promotion_margin) or not 0 <= args.promotion_margin < 1:
        raise ProtocolError("promotion margin must be in [0, 1)")
    if args.bootstrap_samples <= 0:
        raise ProtocolError("bootstrap samples must be positive")
    binary_controller.validate_runtime_args(args)


def claim_manifest(
    path: Path, initial: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite controller checkpoint {path}")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("run_uuid") != initial.get("run_uuid")
            or existing.get("controller_spec_sha256")
            != initial.get("controller_spec_sha256")
        ):
            raise CheckpointError("resume controller spec/run UUID does not match checkpoint")
        return existing
    if resume:
        raise FileNotFoundError(f"resume requested but checkpoint does not exist: {path}")
    atomic_write_json(path, initial)
    return dict(initial)


def persist_manifest(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    atomic_write_json(args.manifest, manifest)
    journal_path = getattr(args, "recovery_journal", None)
    if journal_path:
        journal = {
            "schema_version": SCHEMA_VERSION,
            "controller": CONTROLLER_NAME,
            "run_uuid": manifest.get("run_uuid"),
            "status": manifest.get("status"),
            "binary_path": manifest.get("binary_path"),
            "initial_binary": manifest.get("initial_binary"),
            "switches": manifest.get("switches", []),
            "restoration": manifest.get("restoration"),
            "updated_at_utc": utc_now(),
        }
        atomic_write_json(Path(journal_path), journal)


def _step_key(stage: str, arm: str, block: int | None) -> str:
    return f"{stage}|{arm}|{'-' if block is None else block}"


def audit_execution_journal(manifest: Mapping[str, Any]) -> dict[str, Any]:
    spec = _required_mapping(manifest.get("controller_spec"), "controller spec")
    expected = [
        _step_key("calibration", arm, None)
        for arm in spec.get("calibration_order", [])
    ] + [
        _step_key("final", str(row["arm"]), int(row["final_block"]))
        for row in spec.get("final_schedule", [])
    ]
    successful = [
        row for row in manifest.get("runner_runs", []) if row.get("exit_code") == 0
    ]
    actual = [str(row.get("controller_step_key", "")) for row in successful]
    identities = [row.get("runtime_identity") for row in successful]
    cache_records = [row.get("cache_protocol") for row in successful]
    source_digests = {
        arm: str(spec.get("binary_sources", {}).get(arm, {}).get("expected_digest", ""))
        for arm in ARMS
    }
    identity_passed = all(
        isinstance(identity, Mapping)
        and identity.get("sha256_exact_match") is True
        and identity.get("build_id_exact_match") is True
        and identity.get("loaded_vector_so_sha256")
        == source_digests.get(str(record.get("arm")))
        for identity, record in zip(identities, successful, strict=True)
    )
    cache_passed = all(
        isinstance(record, Mapping)
        and record.get("complete") is True
        and record.get("protocol_sha256")
        == spec.get("cache_protocol", {}).get("sha256")
        for record in cache_records
    )
    passed = (
        actual == expected
        and identity_passed
        and cache_passed
        and spec.get("final_schedule_audit", {}).get("seeded_rotation_verified") is True
    )
    audit = {
        "expected_steps": expected,
        "actual_steps": actual,
        "runtime_identity_passed": identity_passed,
        "cache_protocol_passed": cache_passed,
        "seeded_rotation_passed": spec.get("final_schedule_audit", {}).get(
            "seeded_rotation_verified"
        ),
        "passed": passed,
    }
    if not passed:
        raise CheckpointError(f"controller execution journal audit failed: {audit}")
    return audit


def _run_arm_step(
    args: argparse.Namespace,
    manifest: MutableMapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    binary_path: str,
    arm: str,
    stage: str,
    block: int | None,
    graph: Mapping[str, Any],
    filters: Sequence[str],
    controller_spec_hash: str,
    selections: Sequence[Mapping[str, Any]] = (),
    position: int | None = None,
) -> dict[str, Any]:
    source = sources[arm]
    switch = binary_controller.switch_binary(
        args, manifest, binary_path, source
    )
    runtime = verify_runtime_identity(args, arm, source, binary_path)
    switch["runtime_identity"] = runtime
    persist_manifest(args, manifest)
    cache = execute_cache_protocol(args, arm, stage, block)
    started = utc_now()
    if arm in {"official", "sqlens_disabled"}:
        runner_record = binary_controller.run_external_runner(
            args,
            arm,
            execution_stage=stage,
            final_block=block,
        )
        result: dict[str, Any] = dict(runner_record)
    elif stage == "calibration":
        full = run_full_sqlens_calibration(
            args,
            filters,
            graph,
            runtime,
            controller_spec_hash,
        )
        result = {
            "exit_code": 0,
            "full_manifest": artifact_entry(controller_paths(args)["full_manifest"]),
            "resumed": full["resumed"],
        }
    else:
        if block is None or position is None:
            raise ProtocolError("full SQLens final step requires block and position")
        full = run_full_sqlens_final_block(
            args,
            filters,
            graph,
            runtime,
            controller_spec_hash,
            selections,
            block,
            position,
        )
        result = {
            "exit_code": 0,
            "full_final_block": full["artifact"],
            "resumed": full["resumed"],
        }
    result.update(
        {
            "arm": arm,
            "implementation": arm,
            "execution_stage": stage,
            "final_block": block,
            "position": position,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "runtime_identity": runtime,
            "cache_protocol": cache,
            "controller_step_key": _step_key(stage, arm, block),
            "run_uuid": args.run_uuid,
        }
    )
    return result


def _same_selection_matrix(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> bool:
    def keys(rows: Sequence[Mapping[str, Any]]) -> set[tuple[Any, ...]]:
        return {
            (
                str(row["arm"]),
                str(row["filter_name"]),
                float(row["target_recall"]),
                str(row["config_label"]),
                str(row.get("calibration_selection_policy", "")),
                str(row.get("selection_fallback", "")),
                float(row.get("recall_lcb95", row.get("calibration_recall_lcb95", -1))),
            )
            for row in rows
        }

    return keys(left) == keys(right) and len(left) == len(right)


def _stable_upstream_database(value: Mapping[str, Any]) -> dict[str, Any]:
    graph = _required_mapping(
        value.get("source_clone_graph_identity"), "upstream graph database binding"
    )
    source = _required_mapping(graph.get("source_index"), "upstream source index")
    clone = _required_mapping(graph.get("clone_index"), "upstream clone index")
    return {
        key: value.get(key)
        for key in (
            "system_identifier",
            "database_oid",
            "table_oid",
            "table_relfilenode",
            "table_rows",
            "table_min_id",
            "table_max_id",
            "data_epoch",
        )
    } | {
        "source": {
            key: source.get(key)
            for key in ("index", "index_oid", "index_relfilenode", "heap_oid")
        },
        "clone": {
            key: clone.get(key)
            for key in ("index", "index_oid", "index_relfilenode", "heap_oid")
        },
    }


def validate_cross_arm_provenance(
    args: argparse.Namespace,
    controller_manifest: Mapping[str, Any],
    filters: Sequence[str],
) -> dict[str, Any]:
    """Bind all arm checkpoints to one database, truth, graph, and binary run."""
    spec = _required_mapping(controller_manifest.get("controller_spec"), "controller spec")
    expected_hashes = _required_mapping(spec.get("source_hashes"), "controller hashes")
    upstream_manifests: dict[str, dict[str, Any]] = {}
    stable_databases: dict[str, dict[str, Any]] = {}
    for arm in ("official", "sqlens_disabled"):
        path = upstream_runner.output_paths(
            args.out_dir, arm, args.tag, args.run_uuid
        )["manifest"]
        arm_manifest = json.loads(path.read_text(encoding="utf-8"))
        upstream_manifests[arm] = arm_manifest
        if (
            arm_manifest.get("run_uuid") != args.run_uuid
            or arm_manifest.get("implementation") != arm
            or arm_manifest.get("query_splits")
            != {
                "screen": {"first": 0, "last": 19, "queries": 20},
                "verification": {"first": 20, "last": 99, "queries": 80},
                "final": {"first": 100, "last": 199, "queries": 100},
            }
        ):
            raise FinalizationError(f"{arm} run/query-split identity is invalid")
        design = _required_mapping(arm_manifest.get("formal_design"), f"{arm} design")
        if (
            list(design.get("filters", [])) != list(filters)
            or tuple(map(float, design.get("target_recalls", []))) != TARGET_RECALLS
            or design.get("cell_count") != FORMAL_CELL_COUNT
        ):
            raise FinalizationError(f"{arm} formal matrix differs from the controller")
        hashes = _required_mapping(arm_manifest.get("source_hashes"), f"{arm} hashes")
        if (
            hashes.get("filters_sha256") != expected_hashes.get("filters")
            or hashes.get("truth_sha256") != expected_hashes.get("truth")
            or hashes.get("graph_identity_sha256")
            != expected_hashes.get("graph_identity")
        ):
            raise FinalizationError(f"{arm} input hashes differ from the controller")
        arm_args = _required_mapping(arm_manifest.get("args"), f"{arm} args")
        if (
            arm_args.get("candidate_validity_predicate")
            != CANDIDATE_VALIDITY_PREDICATE
            or arm_args.get("screen_repeats") != SCREEN_REPEATS
            or arm_args.get("verification_repeats") != CALIBRATION_REPEATS
            or arm_args.get("final_repeats") != FINAL_REPEATS
        ):
            raise FinalizationError(f"{arm} workload contract differs from the controller")
        binary = _required_mapping(
            arm_manifest.get("server_binary_provenance"), f"{arm} binary"
        )
        expected_sha = spec.get("binary_sources", {}).get(arm, {}).get(
            "expected_digest"
        )
        if (
            binary.get("binary_hash_matches_expected") is not True
            or binary.get("vector_so_sha256") != expected_sha
            or binary.get("expected_vector_so_sha256") != expected_sha
        ):
            raise FinalizationError(f"{arm} server binary is not exactly bound")
        database = _required_mapping(
            arm_manifest.get("database_fingerprint"), f"{arm} database"
        )
        graph_binding = _required_mapping(
            database.get("source_clone_graph_identity"), f"{arm} graph binding"
        )
        proof = _required_mapping(graph_binding.get("proof"), f"{arm} graph proof")
        if proof.get("stable_fingerprint_sha256") != spec.get(
            "graph_contract", {}
        ).get("stable_fingerprint_sha256"):
            raise FinalizationError(f"{arm} graph fingerprint differs from M32 proof")
        stable_databases[arm] = _stable_upstream_database(database)
    if stable_databases["official"] != stable_databases["sqlens_disabled"]:
        raise FinalizationError("official and SQLens-disabled database identities differ")

    full_path = controller_paths(args)["full_manifest"]
    full = json.loads(full_path.read_text(encoding="utf-8"))
    if (
        full.get("run_uuid") != args.run_uuid
        or full.get("arm") != "sqlens_full"
        or full.get("status") != "arm_ready"
    ):
        raise FinalizationError("full SQLens arm checkpoint is not arm_ready")
    run_spec = _required_mapping(full.get("run_spec"), "full SQLens run spec")
    if (
        run_spec.get("filters_sha256") != expected_hashes.get("filters")
        or run_spec.get("truth_sha256") != expected_hashes.get("truth")
    ):
        raise FinalizationError("full SQLens input hashes differ from the controller")
    query_contract = _required_mapping(
        run_spec.get("query_contract"), "full SQLens query contract"
    )
    if (
        query_contract.get("candidate_validity_predicate")
        != CANDIDATE_VALIDITY_PREDICATE
        or query_contract.get("self_excluded") is not True
    ):
        raise FinalizationError("full SQLens truth/candidate contract differs")
    full_graph = _required_mapping(run_spec.get("d2_graph_proof"), "full live graph")
    if full_graph.get("stable_fingerprint_sha256") != spec.get(
        "graph_contract", {}
    ).get("stable_fingerprint_sha256"):
        raise FinalizationError("full SQLens live graph differs from the M32 proof")
    runtime = _required_mapping(
        run_spec.get("sqlens_runtime_provenance"), "full SQLens runtime"
    )
    sqlens_sha = spec.get("binary_sources", {}).get("sqlens_full", {}).get(
        "expected_digest"
    )
    if (
        runtime.get("loaded_vector_so_sha256") != sqlens_sha
        or runtime.get("loaded_vector_sqlens_build_id")
        != spec.get("expected_sqlens_build_id")
    ):
        raise FinalizationError("full SQLens exact SHA/build identity differs")

    full_database = _required_mapping(run_spec.get("database"), "full SQLens database")
    relations = _required_mapping(
        full_database.get("relations"), "full SQLens database relations"
    )
    upstream_database = stable_databases["official"]
    for role, name in (
        ("source", args.source_index),
        ("clone", args.clone_index),
    ):
        relation = _required_mapping(relations.get(name), f"full SQLens {role} relation")
        expected_relation = upstream_database[role]
        if (
            int(relation.get("oid", 0)) != int(expected_relation["index_oid"])
            or int(relation.get("relfilenode", 0))
            != int(expected_relation["index_relfilenode"])
            or relation.get("candidate_validity_predicate_matches") is not True
        ):
            raise FinalizationError(
                f"full SQLens {role} index differs from upstream arm identity"
            )
    evidence = {
        "upstream_database_identity": stable_databases["official"],
        "full_database_identity_sha256": sha256_json(
            _stable_database_identity(full_database)
        ),
        "graph_stable_fingerprint_sha256": full_graph[
            "stable_fingerprint_sha256"
        ],
        "filters_sha256": expected_hashes["filters"],
        "truth_sha256": expected_hashes["truth"],
        "official_vector_so_sha256": spec["binary_sources"]["official"][
            "expected_digest"
        ],
        "sqlens_vector_so_sha256": sqlens_sha,
        "sqlens_build_id": spec["expected_sqlens_build_id"],
        "passed": True,
    }
    evidence["sha256"] = sha256_json(evidence)
    return evidence


def finalize_three_arm_artifacts(
    args: argparse.Namespace,
    manifest: MutableMapping[str, Any],
    filters: Sequence[str],
    schedule: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paths = controller_paths(args)
    cross_arm_provenance = validate_cross_arm_provenance(args, manifest, filters)
    final_rows = [
        *load_upstream_final_rows(args, "official", selections, schedule),
        *load_upstream_final_rows(args, "sqlens_disabled", selections, schedule),
        *load_full_final_rows(args, selections, schedule),
    ]
    write_csv_atomic(paths["final_raw"], final_rows, CANONICAL_FIELDS)
    paired = build_query_level_pairs(final_rows, selections, filters)
    summaries, pairwise = summarize_paired_final(
        paired,
        filters,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_csv_atomic(paths["paired"], paired)
    write_csv_atomic(paths["summary"], summaries)
    write_csv_atomic(paths["pairwise"], pairwise)
    misses = [
        {
            "arm": row["arm"],
            "filter_name": row["filter_name"],
            "target_recall": row["target_recall"],
            "final_recall_mean": row["recall_mean"],
        }
        for row in summaries
        if row["heldout_target_met"] is not True
    ]
    outputs = {
        "selection": artifact_entry(paths["selection"], rows=len(selections)),
        "calibration": artifact_entry(
            paths["calibration"], rows=len(read_csv(paths["calibration"]))
        ),
        "final_raw": artifact_entry(paths["final_raw"], rows=len(final_rows)),
        "paired_query_rows": artifact_entry(paths["paired"], rows=len(paired)),
        "summary": artifact_entry(paths["summary"], rows=len(summaries)),
        "pairwise": artifact_entry(paths["pairwise"], rows=len(pairwise)),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "amazon10m_pgvector_three_arm_matched_recall",
        "artifact_valid": not misses,
        "run_uuid": args.run_uuid,
        "controller_spec_sha256": manifest["controller_spec_sha256"],
        "protocol": formal_protocol(),
        "calibration_selection": {
            "policy": CALIBRATION_SELECTION_POLICY,
            "rule": CALIBRATION_SELECTION_RULE,
        },
        "execution_journal_audit": manifest["execution_journal_audit"],
        "cross_arm_provenance": cross_arm_provenance,
        "query_level_paired": True,
        "heldout_target_misses": misses,
        "outputs": outputs,
        "completed_at_utc": utc_now(),
    }
    if misses:
        manifest.update(
            {
                "status": "staging_unconfirmed",
                "artifact_valid": False,
                "heldout_target_misses": misses,
                "outputs": outputs,
                "finished_at_utc": utc_now(),
            }
        )
        persist_manifest(args, manifest)
        raise FinalizationError(
            f"held-out final mean recall missed {len(misses)} arm/filter/target cells"
        )
    if paths["published"].exists():
        raise FileExistsError(f"refusing to overwrite published artifact {paths['published']}")
    atomic_write_json(paths["published"], report)
    return report | {"published": artifact_entry(paths["published"])}


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    validate_formal_args(args)
    filters, truth = validate_truth_and_filters(args.filters_csv, args.truth_csv)
    filter_names = [row["filter_name"] for row in filters]
    graph = validate_m32_same_graph_proof(
        args.graph_identity_json, args.source_index, args.clone_index
    )
    sources = source_specs(args)
    prevalidated = {
        arm: binary_controller.validate_host_binary(source)
        for arm, source in sources.items()
    }
    preflight_active = binary_controller.enforce_active_session_gate(args)
    binary_path = binary_controller.discover_vector_so(args.server_container)
    spec = build_controller_spec(args, filters, graph, truth, sources)
    paths = controller_paths(args)
    args.manifest = paths["manifest"]
    args.recovery_journal = paths["recovery_dir"] / "journal.json"
    backup_path = paths["recovery_dir"] / "vector.so.initial"
    initial = {
        "schema_version": SCHEMA_VERSION,
        "controller": CONTROLLER_NAME,
        "run_uuid": args.run_uuid,
        "status": "backing_up_initial_binary",
        "artifact_valid": False,
        "started_at_utc": utc_now(),
        "controller_spec": spec,
        "controller_spec_sha256": spec["controller_spec_sha256"],
        "calibration_selection": {
            "policy": CALIBRATION_SELECTION_POLICY,
            "rule": CALIBRATION_SELECTION_RULE,
        },
        "binary_path": binary_path,
        "preflight_active_session_gate": preflight_active,
        "prevalidated_host_digests": prevalidated,
        "switches": [],
        "runner_runs": [],
    }
    manifest = claim_manifest(args.manifest, initial, resume=args.resume)
    if manifest.get("binary_path") != binary_path:
        raise CheckpointError("resume discovered a different server vector.so path")
    if args.resume:
        initial_binary = _required_mapping(
            manifest.get("initial_binary"), "initial binary checkpoint"
        )
        if initial_binary.get("host_path") != str(backup_path) or not backup_path.is_file():
            raise CheckpointError("persistent initial vector.so backup is missing")
        original_sha = sha256_file(backup_path)
        if original_sha != initial_binary.get("sha256"):
            raise CheckpointError("persistent initial vector.so backup SHA256 changed")
    else:
        if backup_path.exists():
            raise FileExistsError(f"recovery backup already exists: {backup_path}")
        paths["recovery_dir"].mkdir(parents=True, exist_ok=True)
        binary_controller.docker_copy(
            f"{args.server_container}:{binary_path}", str(backup_path)
        )
        binary_controller.fsync_existing_file(backup_path)
        original_sha = sha256_file(backup_path)
        manifest["initial_binary"] = {
            "host_path": str(backup_path),
            "sha256": original_sha,
            "fsync_verified": True,
            "scope": "persistent run UUID recovery journal",
        }
        manifest["status"] = "running"
        persist_manifest(args, manifest)

    restore_source = {
        "implementation": "restore_initial",
        "source_tag": "initial-container-binary",
        "source_commit": "",
        "expected_digest": original_sha,
        "host_path": str(backup_path),
    }
    previous_sigterm: Any = None
    signal_installed = False

    def request_termination(_signum: int, _frame: Any) -> None:
        raise binary_controller.TerminationRequested(
            "SIGTERM requested controlled three-arm restoration"
        )

    try:
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, request_termination)
        signal_installed = True
    except ValueError:
        signal_installed = False

    failure: BaseException | None = None
    selections: list[dict[str, Any]] = []
    try:
        completed = {
            str(row.get("controller_step_key"))
            for row in manifest.get("runner_runs", [])
            if row.get("exit_code") == 0
        }
        for arm in spec["calibration_order"]:
            key = _step_key("calibration", arm, None)
            if key in completed:
                continue
            record = _run_arm_step(
                args,
                manifest,
                sources,
                binary_path,
                arm,
                "calibration",
                None,
                graph,
                filter_names,
                spec["controller_spec_sha256"],
            )
            manifest["runner_runs"].append(record)
            completed.add(key)
            persist_manifest(args, manifest)

        official_calibration, official_manifest = load_upstream_calibration_rows(
            args, "official"
        )
        disabled_calibration, disabled_manifest = load_upstream_calibration_rows(
            args, "sqlens_disabled"
        )
        _full_manifest, full_calibration, full_selection = _validate_full_resume_manifest(
            paths["full_manifest"], spec["controller_spec_sha256"]
        )
        calibration_rows = [
            *official_calibration,
            *disabled_calibration,
            *full_calibration,
        ]
        selections = select_calibrated_configs(
            calibration_rows,
            filter_names,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + 100_000,
        )
        assert_upstream_selection_matches(official_manifest, selections, "official")
        assert_upstream_selection_matches(
            disabled_manifest, selections, "sqlens_disabled"
        )
        if not _same_selection_matrix(
            [row for row in selections if row["arm"] == "sqlens_full"],
            full_selection,
        ):
            raise ProtocolError("unified full SQLens selection differs from its checkpoint")
        write_csv_atomic(paths["calibration"], calibration_rows, CANONICAL_FIELDS)
        write_csv_atomic(paths["selection"], selections)
        selection_hash = sha256_json(
            sorted(
                (
                    row["arm"],
                    row["filter_name"],
                    float(row["target_recall"]),
                    row["config_label"],
                    row["calibration_selection_policy"],
                    row["selection_fallback"],
                    float(row["recall_lcb95"]),
                )
                for row in selections
            )
        )
        if manifest.get("selection_sha256") not in {None, selection_hash}:
            raise CheckpointError("calibration selection changed on resume")
        manifest.update(
            {
                "status": "calibration_complete",
                "selection_sha256": selection_hash,
                "calibration_artifact": artifact_entry(
                    paths["calibration"], rows=len(calibration_rows)
                ),
                "selection_artifact": artifact_entry(
                    paths["selection"], rows=len(selections)
                ),
            }
        )
        persist_manifest(args, manifest)

        for item in spec["final_schedule"]:
            arm = str(item["arm"])
            block = int(item["final_block"])
            key = _step_key("final", arm, block)
            if key in completed:
                continue
            record = _run_arm_step(
                args,
                manifest,
                sources,
                binary_path,
                arm,
                "final",
                block,
                graph,
                filter_names,
                spec["controller_spec_sha256"],
                selections,
                int(item["position"]),
            )
            manifest["runner_runs"].append(record)
            completed.add(key)
            persist_manifest(args, manifest)
        manifest["execution_journal_audit"] = audit_execution_journal(manifest)
        manifest["status"] = "measurements_complete_restoration_pending"
        persist_manifest(args, manifest)
    except BaseException as exc:
        failure = exc
        manifest.update(
            {
                "status": "failed_restoration_pending",
                "fatal_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        persist_manifest(args, manifest)
    finally:
        restore_required = any(
            event.get("replacement_attempted")
            for event in manifest.get("switches", [])
            if event.get("recovery") is not True
        )
        if restore_required:
            try:
                binary_controller.switch_binary(
                    args,
                    manifest,
                    binary_path,
                    restore_source,
                    recovery=True,
                )
                manifest["restoration"] = {
                    "status": "verified",
                    "sha256": original_sha,
                    "finished_at_utc": utc_now(),
                }
            except BaseException as restore_error:
                manifest["restoration"] = {
                    "status": "failed",
                    "error": f"{restore_error.__class__.__name__}: {restore_error}",
                    "finished_at_utc": utc_now(),
                }
                failure = binary_controller.RecoveryFailedError(
                    f"initial binary restoration failed: {restore_error}", failure
                )
        else:
            manifest["restoration"] = {
                "status": "not_required",
                "sha256": original_sha,
            }
        persist_manifest(args, manifest)
        if signal_installed:
            signal.signal(signal.SIGTERM, previous_sigterm)

    if failure is not None:
        manifest.update(
            {
                "status": (
                    "recovery_failed"
                    if isinstance(failure, binary_controller.RecoveryFailedError)
                    else "failed"
                ),
                "finished_at_utc": utc_now(),
            }
        )
        persist_manifest(args, manifest)
        raise failure

    report = finalize_three_arm_artifacts(
        args,
        manifest,
        filter_names,
        spec["final_schedule"],
        selections,
    )
    manifest.update(
        {
            "status": "completed",
            "artifact_valid": True,
            "finished_at_utc": utc_now(),
            "published_artifact": report["published"],
            "outputs": report["outputs"],
        }
    )
    persist_manifest(args, manifest)
    return manifest


def build_upstream_runner_argv(
    args: argparse.Namespace,
    arm: str,
    execution_stage: str = "calibration",
    final_block: int | None = None,
) -> list[str]:
    if arm not in {"official", "sqlens_disabled"}:
        raise ProtocolError("upstream runner only owns official/sqlens_disabled")
    return binary_controller.build_runner_argv(
        args, arm, execution_stage, final_block
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Formal Amazon-10M matched-recall controller for official pgvector, "
            "SQLens-disabled, and full SQLens."
        )
    )
    parser.add_argument("--server-container", type=binary_controller.validate_container)
    parser.add_argument(
        "--official-vector-so",
        "--official-binary",
        dest="official_vector_so",
        type=Path,
    )
    parser.add_argument(
        "--official-vector-so-sha256",
        type=binary_controller.validate_sha256,
        default=binary_controller.OFFICIAL_VECTOR_SO_SHA256,
    )
    parser.add_argument(
        "--sqlens-vector-so",
        "--sqlens-binary",
        dest="sqlens_vector_so",
        type=Path,
    )
    parser.add_argument(
        "--sqlens-vector-so-sha256",
        "--sqlens-digest",
        dest="sqlens_vector_so_sha256",
        type=binary_controller.validate_sha256,
    )
    parser.add_argument("--expected-sqlens-build-id", default="")
    parser.add_argument(
        "--required-sqlens-build-prefix",
        default=REQUIRED_TRAVERSAL_BUILD_PREFIX,
    )
    parser.add_argument("--traversal-guided-burst", type=int, default=8)
    parser.add_argument(
        "--minimum-sqlens-profile-semantics",
        type=float,
        default=binary_controller.DEFAULT_SQLENS_PROFILE_SEMANTICS,
    )
    parser.add_argument("--official-vector-source-tag", default="")
    parser.add_argument("--official-vector-source-commit", default="")
    parser.add_argument("--official-vector-build-recipe", default="")
    parser.add_argument("--official-vector-compiler-flags", default="")
    parser.add_argument("--official-vector-source-repo", type=Path)
    parser.add_argument("--sqlens-vector-source-tag", default="")
    parser.add_argument("--sqlens-vector-source-commit", default="")
    parser.add_argument("--sqlens-vector-build-recipe", default="")
    parser.add_argument("--sqlens-vector-compiler-flags", default="")
    parser.add_argument("--sqlens-vector-source-repo", type=Path)

    parser.add_argument(
        "--filters-csv",
        type=Path,
        default=DEFAULT_FILTERS,
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=ROOT
        / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_unique_embeddings_formal.csv",
    )
    parser.add_argument(
        "--graph-identity-json",
        type=Path,
        default=ROOT
        / "results/hybrid_vector_db/amazon10m_valid_embedding_m32ef200_d2_graph_proof.json",
    )
    parser.add_argument(
        "--table",
        type=binary_controller.validate_identifier,
        default="public.amazon_grocery_reviews_10m_pgvector",
    )
    parser.add_argument(
        "--index",
        type=binary_controller.validate_identifier,
        default="public.amazon10m_embedding_valid_hnsw_m32ef200_source_idx",
    )
    parser.add_argument(
        "--source-index",
        type=binary_controller.validate_identifier,
        default="public.amazon10m_embedding_valid_hnsw_m32ef200_source_idx",
    )
    parser.add_argument(
        "--clone-index",
        type=binary_controller.validate_identifier,
        default="public.amazon10m_embedding_valid_hnsw_m32ef200_bfs_clone_idx",
    )
    parser.add_argument("--query-table")
    parser.add_argument("--query-id-column", default="id")
    parser.add_argument("--query-vector-column", default="embedding")
    parser.add_argument(
        "--candidate-validity-predicate",
        default=CANDIDATE_VALIDITY_PREDICATE,
    )
    parser.add_argument("--hnsw-m", type=binary_controller.positive_int, default=HNSW_M)
    parser.add_argument("--data-epoch", default="")
    parser.add_argument(
        "--planner-mode", choices=("auto", "forced_hnsw"), default="auto"
    )
    parser.add_argument(
        "--dsn",
        default="",
        help="explicit DSN passed to the shared upstream runner",
    )
    parser.add_argument("--k", type=binary_controller.positive_int, default=10)
    parser.add_argument(
        "--target-recalls",
        type=binary_controller.parse_target_recalls,
        default=list(TARGET_RECALLS),
    )
    parser.add_argument("--promotion-margin", type=float, default=0.02)
    parser.add_argument(
        "--screen-repeats",
        type=binary_controller.positive_int,
        default=SCREEN_REPEATS,
    )
    parser.add_argument(
        "--verification-repeats",
        type=binary_controller.positive_int,
        default=CALIBRATION_REPEATS,
    )
    parser.add_argument(
        "--final-repeats",
        type=binary_controller.positive_int,
        default=FINAL_REPEATS,
    )
    parser.add_argument("--formal-family", choices=("off", "strict_order"), default="off")
    parser.add_argument("--config-ladder", type=Path)
    parser.add_argument(
        "--max-ef-search", type=binary_controller.positive_int, default=1000
    )
    parser.add_argument("--upstream-evaluation-patch", type=Path)

    parser.add_argument("--full-mode", default=FULL_SQLENS_MODE)
    parser.add_argument(
        "--full-ef-search-values", default=FULL_EF_SEARCH_VALUES
    )
    parser.add_argument("--full-guided-collect-target-values", default="ef")
    parser.add_argument("--full-max-scan-tuples-values", default="5000000")
    parser.add_argument("--full-scan-mem-multiplier-values", default="32")
    parser.add_argument("--guidance-max-atoms", type=int, default=64)
    parser.add_argument(
        "--d2-page-access", choices=("off", "prefetch", "reorder"), default="off"
    )
    parser.add_argument(
        "--d2-index-page-access", choices=("off", "prefetch"), default="off"
    )
    parser.add_argument("--preferred-index-guc", default="hnsw.preferred_index")
    parser.add_argument("--d1-cache-mb", type=binary_controller.positive_int, default=1024)
    parser.add_argument("--d3-cache-mb", type=binary_controller.positive_int, default=1024)
    parser.add_argument("--backend-cpu-list")

    parser.add_argument("--warmup-queries", type=binary_controller.positive_int, default=5)
    parser.add_argument(
        "--prewarm-relation",
        dest="prewarm_relations",
        action="append",
        type=binary_controller.validate_identifier,
        default=[],
    )
    parser.add_argument(
        "--cache-protocol",
        choices=(CACHE_PROTOCOL_VERSION,),
        default=CACHE_PROTOCOL_VERSION,
    )
    parser.add_argument(
        "--statement-timeout-ms",
        type=binary_controller.nonnegative_int,
        default=300_000,
    )
    parser.add_argument("--progress-queries", type=int, default=10)
    parser.add_argument(
        "--bootstrap-samples", type=binary_controller.positive_int, default=10_000
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--schedule-seed", type=int, default=20260718)

    parser.add_argument("--pg-host", default=os.environ.get("PGHOST", "127.0.0.1"))
    parser.add_argument(
        "--pg-port",
        type=binary_controller.positive_int,
        default=int(os.environ.get("PGPORT", "55432")),
    )
    parser.add_argument(
        "--pg-database", default=os.environ.get("PGDATABASE", "hybrid_vector")
    )
    parser.add_argument("--pg-user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument(
        "--pg-isready-timeout-seconds",
        type=binary_controller.positive_int,
        default=60,
    )
    parser.add_argument("--pg-isready-poll-seconds", type=float, default=1.0)
    parser.add_argument("--allow-active-sessions", action="store_true")

    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results/hybrid_vector_db"
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--publish-path", type=Path)
    parser.add_argument("--tag", default="20260718")
    parser.add_argument("--run-uuid", default="")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    # The shared A/B runner accepts this optional list. Formal three-arm runs use all 14.
    parser.set_defaults(filter_names=[])
    return parser


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    run_uuid = args.run_uuid or "<generated-at-runtime>"
    schedule, audit = rotating_final_schedule(run_uuid, args.schedule_seed)
    relations = list(args.prewarm_relations) or [
        args.table,
        args.source_index,
        args.clone_index,
    ]
    manifest = (
        args.manifest
        or args.out_dir / "staging" / run_uuid / "three_arm_controller.json"
    )
    return {
        "controller": CONTROLLER_NAME,
        "arms": list(ARMS),
        "protocol": formal_protocol(),
        "run_uuid": run_uuid,
        "manifest": str(manifest),
        "calibration_order": calibration_order(run_uuid, args.schedule_seed),
        "final_schedule": schedule,
        "final_schedule_audit": audit,
        "cache_protocol": cache_protocol_spec(relations),
        "full_sqlens_mode": args.full_mode,
        "expected_sqlens_build_id": args.expected_sqlens_build_id,
        "official_vector_so_sha256": args.official_vector_so_sha256,
        "sqlens_vector_so_sha256": args.sqlens_vector_so_sha256,
        "candidate_validity_predicate": args.candidate_validity_predicate,
        "hnsw_m": args.hnsw_m,
        "graph_identity_json": str(args.graph_identity_json),
        "file_access": False,
        "docker_access": False,
        "database_access": False,
        "experiment_started": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_payload(args), sort_keys=True))
        return 0
    if not args.run_uuid:
        if args.resume:
            print("controller failed: --resume requires --run-uuid", file=sys.stderr)
            return 1
        args.run_uuid = str(uuid.uuid4())
    if args.manifest is None:
        args.manifest = (
            args.out_dir / "staging" / args.run_uuid / "three_arm_controller.json"
        )
    try:
        run_controller(args)
    except binary_controller.TerminationRequested as exc:
        print(f"controller terminated after restoration: {exc}", file=sys.stderr)
        return 128
    except Exception as exc:
        print(f"controller failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote controller manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
