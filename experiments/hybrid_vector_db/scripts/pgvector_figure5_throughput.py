#!/usr/bin/env python3
"""Formal mixed-q10K Stock-vs-full-SQLens throughput runner for Figure 5.

One invocation measures one dataset, one frozen search configuration, and one
client count.  It executes exactly two paired arms:

* ``original`` -> ``stock_pgvector``
* ``design1_bloom_bfs_layout_d3`` -> ``sqlens_full``

Each repeat uses one method-independent seeded permutation of the frozen q10K
trace.  Client threads own independent PostgreSQL connections and wait on a
shared start barrier.  Throughput is completed requests divided by the measured
barrier wall-clock interval; it is never derived from request latency.

The SQLens arm uses a fresh persistent-fragment namespace for every repeat.
There are no unmeasured SQLens warmup or canary queries: D3 probe, admission,
materialization, and reuse all happen inside timed requests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from . import pgvector_design1_design2_design3_selectivity_benchmark as core
    from . import pgvector_formal_throughput_benchmark as telemetry
    from .common_pg import pg_config_from_env
    from .pgvector_target_recall_selectivity_runner import (
        acquire_formal_data_guard,
        database_fingerprint,
        prepare_fragment_tracking,
    )
except ImportError:
    import pgvector_design1_design2_design3_selectivity_benchmark as core
    import pgvector_formal_throughput_benchmark as telemetry
    from common_pg import pg_config_from_env
    from pgvector_target_recall_selectivity_runner import (
        acquire_formal_data_guard,
        database_fingerprint,
        prepare_fragment_tracking,
    )


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FRONTIER_CONFIG = (
    ROOT / "experiments/hybrid_vector_db/configs/figure5_frontier_datasets.json"
)
RUNNER_VERSION = "sqlens-figure5-mixed-q10k-throughput-v1"
EXPECTED_REQUESTS = 10_000
EXPECTED_FILTERS = 14
MIN_REPEATS = 6
MAX_CLIENTS = 32
K = 10
MODES = ("original", "design1_bloom_bfs_layout_d3")
ARM_BY_MODE = {
    "original": "stock_pgvector",
    "design1_bloom_bfs_layout_d3": "sqlens_full",
}
MODE_BY_ARM = {arm: mode for mode, arm in ARM_BY_MODE.items()}
DATASET_IDS = {
    "amazon": "amazon10m",
    "yfcc": "yfcc10m",
    "laion": "laion25m",
}
THROUGHPUT_SOURCE = "measured_completed_over_barrier_wall_clock"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")

REQUEST_FIELDS = (
    "runner_version",
    "run_id",
    "dataset",
    "pair_id",
    "target_recall",
    "config_id",
    "config_sha256",
    "stock_config_sha256",
    "sqlens_config_sha256",
    "arm_config_sha256",
    "release_identity_sha256",
    "arm_id",
    "mode_id",
    "arm_order",
    "repeat_id",
    "clients",
    "trace_permutation_seed",
    "trace_order_sha256",
    "request_trace_sha256",
    "dispatch_position",
    "request_no",
    "trace_cycle",
    "filter_name",
    "query_no",
    "query_id",
    "client_id",
    "backend_pid",
    "client_native_tid",
    "client_requested_cpu",
    "client_affinity_applied",
    "started_offset_ms",
    "completed_offset_ms",
    "latency_ms",
    "activation_ms",
    "query_ms",
    "returned",
    "returned_ids",
    "recall_at_10",
    "d3_fragment_store_namespace",
    "error_type",
    "error",
)

REPEAT_BASE_FIELDS = (
    "schema_version",
    "run_id",
    "dataset",
    "experiment_kind",
    "arm_id",
    "mode_id",
    "pair_id",
    "target_recall",
    "config_id",
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
    "recall_mean",
    "recall_ci95_low",
    "recall_ci95_high",
    "latency_mean_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_qps",
    "throughput_ci95_low",
    "throughput_ci95_high",
    "throughput_source",
    "status",
)

REPEAT_EVIDENCE_FIELDS = (
    "runner_version",
    "arm_order",
    "trace_permutation_seed",
    "trace_order_sha256",
    "backend_pids_json",
    "backend_cpu_provenance_json",
    "client_affinity_json",
    "true_concurrency_observed",
    "d3_measurement_policy",
    "d3_fragment_store_namespace",
    "d3_namespace_rows_before",
    "d3_namespace_rows_after",
    "d3_online_cost_charged",
    "telemetry_collected",
    "telemetry_devices_json",
    "host_cpu_utilization_pct",
    "host_cpu_user_pct",
    "host_cpu_system_pct",
    "host_cpu_iowait_pct",
    "host_disk_reads_completed",
    "host_disk_read_bytes",
    "host_disk_read_time_ms",
    "host_disk_writes_completed",
    "host_disk_write_bytes",
    "host_disk_write_time_ms",
    "host_disk_io_time_ms",
    "host_disk_weighted_io_time_ms",
    "pg_database_blks_read",
    "pg_database_blks_hit",
    "pg_database_temp_files",
    "pg_database_temp_bytes",
    "pg_database_blk_read_time_ms",
    "pg_database_blk_write_time_ms",
    "pg_io_reads",
    "pg_io_read_bytes",
    "pg_io_read_time_ms",
    "pg_io_writes",
    "pg_io_write_bytes",
    "pg_io_write_time_ms",
    "pg_io_hits",
    "pg_io_evictions",
    "pg_target_table_heap_blks_read",
    "pg_target_table_heap_blks_hit",
    "pg_target_table_idx_blks_read",
    "pg_target_table_idx_blks_hit",
    "pg_target_index_blks_read",
    "pg_target_index_blks_hit",
    "pg_backend_cpu_processes",
    "pg_backend_cpu_user_ms",
    "pg_backend_cpu_system_ms",
    "pg_backend_cpu_total_ms",
    "telemetry_json",
)
REPEAT_FIELDS = REPEAT_BASE_FIELDS + REPEAT_EVIDENCE_FIELDS


class Figure5ThroughputError(RuntimeError):
    """A run cannot satisfy the formal Figure 5 throughput contract."""


@dataclass(frozen=True)
class DatasetBinding:
    key: str
    dataset_id: str
    label: str
    table: str
    query_table: str
    query_id_column: str
    query_vector_column: str
    source_index: str
    bfs_index: str
    candidate_validity_predicate: str
    truth_self_excluded: bool
    filters_csv: Path
    truth_csv: Path
    workload_csv: Path
    d2_graph_proof_json: Path


@dataclass(frozen=True)
class FilterSpec:
    name: str
    predicate: str
    actual_pct: float
    atoms: tuple[str, ...]


@dataclass(frozen=True)
class ArmSearchSettings:
    ef_search: int
    iterative_scan: str
    max_scan_tuples: int
    scan_mem_multiplier: float
    guided_collect_target: int
    traversal_guided_target: int
    traversal_guided_burst: int
    traversal_guided_early_stop: bool = False
    traversal_guided_early_stop_distance_ratio: float = 0.0

    def mode_config(self, *, guidance_enabled: bool) -> dict[str, object]:
        return {
            "ef_search": self.ef_search,
            "iterative_scan": self.iterative_scan,
            "max_scan_tuples": self.max_scan_tuples,
            "scan_mem_multiplier": self.scan_mem_multiplier,
            "guided_collect_target": self.guided_collect_target,
            "traversal_guided_target": self.traversal_guided_target,
            "traversal_guided_burst": self.traversal_guided_burst,
            "traversal_guided_early_stop": self.traversal_guided_early_stop,
            "traversal_guided_early_stop_distance_ratio": (
                self.traversal_guided_early_stop_distance_ratio
            ),
            "traversal_guided_prioritization": guidance_enabled,
        }


@dataclass(frozen=True)
class SearchSettings:
    """One independently tuned, paired Stock-vs-SQLens operating point."""

    config_id: str
    pair_id: str
    target_recall: float
    stock: ArmSearchSettings
    sqlens: ArmSearchSettings
    d1_exact_max_selectivity_pct: float = 6.0
    collapse_exact_and_guidance: bool = True
    guidance_selectivity_min_pct: float = 0.0
    guidance_selectivity_max_pct: float = 6.0
    guidance_composite_max_selectivity_pct: float = 100.0
    guidance_max_atoms: int = 160
    d2_source_on_guidance_bypass: bool = True
    guidance_bypass_ef_search: int = 0
    guidance_low_selectivity_bypass_ef_search: int = 0
    filter_ef_search: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    filter_traversal_target: Mapping[str, Mapping[str, int]] = field(
        default_factory=dict
    )
    filter_mode_configs: Mapping[
        str, Mapping[str, Mapping[str, object]]
    ] = field(default_factory=dict)

    def mode_configs(self) -> dict[str, dict[str, object]]:
        return {
            "original": self.stock.mode_config(guidance_enabled=False),
            "design1_bloom_bfs_layout_d3": self.sqlens.mode_config(
                guidance_enabled=True
            ),
        }

    def arm_config(self, arm_id: str) -> dict[str, object]:
        if arm_id == "stock_pgvector":
            return self.mode_configs()["original"]
        if arm_id == "sqlens_full":
            return self.mode_configs()["design1_bloom_bfs_layout_d3"]
        raise Figure5ThroughputError(f"unknown arm: {arm_id}")

    def arm_filter_config(self, arm_id: str) -> dict[str, object]:
        mode_id = MODE_BY_ARM.get(arm_id)
        if mode_id is None:
            raise Figure5ThroughputError(f"unknown arm: {arm_id}")
        return {
            "ef_search": dict(self.filter_ef_search.get(mode_id, {})),
            "traversal_guided_target": dict(
                self.filter_traversal_target.get(mode_id, {})
            ),
            "mode_configs": {
                name: dict(config)
                for name, config in self.filter_mode_configs.get(
                    mode_id, {}
                ).items()
            },
        }


@dataclass(frozen=True)
class FrozenWorkload:
    requests: tuple[core.WorkloadRequest, ...]
    truth: Mapping[tuple[str, int], Any]
    filters: Mapping[str, FilterSpec]
    filter_tuples: tuple[tuple[str, float, str], ...]
    filter_atoms: Mapping[str, list[str]]
    trace_sha256: str
    truth_sha256: str
    filters_sha256: str
    workload_manifest: Mapping[str, Any]
    workload_manifest_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stable_runtime_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the timestamp-free fields that identify the loaded SQLens binary."""
    required = (
        "expected_build_id",
        "expected_vector_so_sha256",
        "observed_build_id",
        "observed_vector_so_path",
        "observed_vector_so_sha256",
        "exact_match",
    )
    missing = [field for field in required if field not in identity]
    if missing:
        raise Figure5ThroughputError(
            f"runtime SQLens identity is missing fields: {missing}"
        )
    stable = {field: identity[field] for field in required}
    if (
        stable["exact_match"] is not True
        or stable["expected_build_id"] != stable["observed_build_id"]
        or stable["expected_vector_so_sha256"]
        != stable["observed_vector_so_sha256"]
        or not SHA256_RE.fullmatch(str(stable["observed_vector_so_sha256"]))
        or not str(stable["observed_vector_so_path"]).endswith("/vector.so")
    ):
        raise Figure5ThroughputError(
            "runtime SQLens identity is not an exact loaded-binary match"
        )
    return stable


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Figure5ThroughputError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Figure5ThroughputError(f"{label} root is not an object: {path}")
    return value


def resolve_path(value: object, *, base: Path = ROOT) -> Path:
    path = Path(str(value or ""))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _metadata_path(
    manifest_path: Path,
    metadata: object,
    label: str,
) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(metadata, Mapping):
        raise Figure5ThroughputError(f"{label} metadata is missing")
    path = resolve_path(metadata.get("path"), base=manifest_path.parent)
    if not path.is_file():
        raise Figure5ThroughputError(f"{label} does not exist: {path}")
    expected_sha = str(metadata.get("sha256") or "")
    if not SHA256_RE.fullmatch(expected_sha) or sha256_file(path) != expected_sha:
        raise Figure5ThroughputError(f"{label} SHA-256 is missing or mismatched")
    return path, metadata


def validate_release_contract(path: Path) -> dict[str, Any]:
    payload = read_json(path, "release contract")
    if payload.get("schema_version") != 1:
        raise Figure5ThroughputError("release contract schema_version must be 1")
    build_id = str(payload.get("expected_sqlens_build_id") or "")
    vector_sha = str(payload.get("expected_vector_so_sha256") or "")
    if not (
        build_id.startswith("sqlens-v16-d3-")
        or build_id.startswith("sqlens-v16-guided-")
        or build_id.startswith("sqlens-v16-distance-aware-")
    ):
        raise Figure5ThroughputError(
            "release contract does not identify a supported full SQLens v16 D3 build"
        )
    if not SHA256_RE.fullmatch(vector_sha):
        raise Figure5ThroughputError(
            "release contract expected_vector_so_sha256 is invalid"
        )
    return {
        **payload,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def load_frontier_config(
    path: Path,
    dataset_key: str,
    release_override: Path | None = None,
) -> tuple[DatasetBinding, dict[str, Any], dict[str, Any]]:
    config = read_json(path, "Figure 5 frontier config")
    if config.get("schema_version") != 1:
        raise Figure5ThroughputError("unsupported Figure 5 frontier config schema")
    protocol = config.get("protocol")
    datasets = config.get("datasets")
    grid = config.get("search_grid")
    if not all(isinstance(value, Mapping) for value in (protocol, datasets, grid)):
        raise Figure5ThroughputError(
            "Figure 5 config must contain protocol, datasets, and search_grid objects"
        )
    if tuple(protocol.get("modes") or ()) != MODES:
        raise Figure5ThroughputError(
            "Figure 5 config methods must be exactly original/full D1+D2+D3"
        )
    if protocol.get("d3_measurement_policy") != "workload_driven_adaptive":
        raise Figure5ThroughputError(
            "Figure 5 throughput requires workload_driven_adaptive D3"
        )
    if dataset_key not in DATASET_IDS or dataset_key not in datasets:
        raise Figure5ThroughputError(f"unknown Figure 5 dataset: {dataset_key!r}")
    row = datasets[dataset_key]
    if not isinstance(row, Mapping):
        raise Figure5ThroughputError(f"dataset config is not an object: {dataset_key}")
    required = {
        "label",
        "table",
        "query_table",
        "query_id_column",
        "query_vector_column",
        "source_index",
        "bfs_index",
        "candidate_validity_predicate",
        "truth_self_excluded",
        "filters_csv",
        "truth_csv",
        "measurement_workload_csv",
        "d2_graph_proof_json",
    }
    missing = sorted(required - set(row))
    if missing:
        raise Figure5ThroughputError(
            f"dataset {dataset_key} config is missing fields: {missing}"
        )
    binding = DatasetBinding(
        key=dataset_key,
        dataset_id=DATASET_IDS[dataset_key],
        label=str(row["label"]),
        table=str(row["table"]),
        query_table=str(row["query_table"]),
        query_id_column=str(row["query_id_column"]),
        query_vector_column=str(row["query_vector_column"]),
        source_index=str(row["source_index"]),
        bfs_index=str(row["bfs_index"]),
        candidate_validity_predicate=str(row["candidate_validity_predicate"]),
        truth_self_excluded=(
            row["truth_self_excluded"]
            if isinstance(row["truth_self_excluded"], bool)
            else _invalid_boolean("truth_self_excluded")
        ),
        filters_csv=resolve_path(row["filters_csv"]),
        truth_csv=resolve_path(row["truth_csv"]),
        workload_csv=resolve_path(row["measurement_workload_csv"]),
        d2_graph_proof_json=resolve_path(row["d2_graph_proof_json"]),
    )
    release_path = (
        release_override.resolve()
        if release_override
        else resolve_path(config.get("release_contract"))
    )
    release = validate_release_contract(release_path)
    return binding, dict(protocol), {
        "config": config,
        "grid": dict(grid),
        "release": release,
        "config_path": str(path.resolve()),
        "config_sha256": sha256_file(path),
    }


def validate_workload_manifest(
    manifest_path: Path,
    binding: DatasetBinding,
    *,
    allowed_request_counts: set[int] | None = None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path, "frozen workload manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "figure5_frontier_workload"
        or manifest.get("artifact_valid") is not True
    ):
        raise Figure5ThroughputError(
            "frozen workload manifest identity/validity gate failed"
        )
    gates = manifest.get("gates")
    required_gates = {
        "exactly_14_filters",
        "input_sha256_bound",
        "measurement_filter_balance",
        "measurement_filter_coverage",
        "measurement_query_no_uniqueness",
        "measurement_query_vector_uniqueness",
        "measurement_request_count",
        "output_sha256_verified",
        "truth_pair_coverage",
        "truth_tie_aware",
    }
    if not isinstance(gates, Mapping) or any(
        gates.get(gate) is not True for gate in required_gates
    ):
        failed = sorted(gate for gate in required_gates if not gates or gates.get(gate) is not True)
        raise Figure5ThroughputError(
            f"frozen workload manifest lacks required gates: {failed}"
        )
    outputs = manifest.get("outputs")
    inputs = manifest.get("inputs")
    if not isinstance(outputs, Mapping) or not isinstance(inputs, Mapping):
        raise Figure5ThroughputError("frozen workload manifest lacks input/output bindings")
    workload_path, workload_meta = _metadata_path(
        manifest_path,
        outputs.get("measurement_workload_csv"),
        "measurement workload CSV",
    )
    truth_path, truth_meta = _metadata_path(
        manifest_path, inputs.get("truth_csv"), "truth CSV"
    )
    filters_path, filters_meta = _metadata_path(
        manifest_path, inputs.get("filters_csv"), "filters CSV"
    )
    expected = {
        "measurement workload CSV": (workload_path, binding.workload_csv),
        "truth CSV": (truth_path, binding.truth_csv),
        "filters CSV": (filters_path, binding.filters_csv),
    }
    for label, (observed, configured) in expected.items():
        if observed != configured.resolve():
            raise Figure5ThroughputError(
                f"{label} path differs from dataset config: {observed} != {configured}"
            )
    request_count = int(workload_meta.get("rows") or -1)
    allowed_counts = allowed_request_counts or {EXPECTED_REQUESTS}
    if request_count not in allowed_counts:
        raise Figure5ThroughputError(
            "measurement workload manifest request count is outside the "
            f"requested contract: observed={request_count}, "
            f"allowed={sorted(allowed_counts)}"
        )
    if int(filters_meta.get("rows") or -1) != EXPECTED_FILTERS:
        raise Figure5ThroughputError("filters manifest does not bind exactly 14 filters")
    truth_contract = manifest.get("truth")
    if not isinstance(truth_contract, Mapping) or truth_contract.get("valid") is not True:
        raise Figure5ThroughputError("frozen workload has no valid exact-truth contract")
    if int(truth_meta.get("rows") or 0) <= 0:
        raise Figure5ThroughputError("truth CSV manifest row count is invalid")
    return manifest


def _load_filters(binding: DatasetBinding) -> tuple[
    dict[str, FilterSpec],
    tuple[tuple[str, float, str], ...],
    dict[str, list[str]],
]:
    raw_filters, atoms_by_filter = core.load_filter_specs(binding.filters_csv)
    if len(raw_filters) != EXPECTED_FILTERS:
        raise Figure5ThroughputError(
            f"formal mixed workload requires 14 filters, observed={len(raw_filters)}"
        )
    if len({name for name, _, _ in raw_filters}) != EXPECTED_FILTERS:
        raise Figure5ThroughputError("filter names are duplicated")
    filters: dict[str, FilterSpec] = {}
    tuples: list[tuple[str, float, str]] = []
    for name, selectivity, predicate in raw_filters:
        if "%" in predicate:
            raise Figure5ThroughputError("synthetic modulo predicates are forbidden")
        actual_pct = core.parse_pct(selectivity)
        atoms = tuple(atoms_by_filter[name])
        filters[name] = FilterSpec(name, predicate, actual_pct, atoms)
        tuples.append((name, actual_pct, predicate))
    return filters, tuple(tuples), atoms_by_filter


def load_frozen_workload(
    binding: DatasetBinding,
    workload_manifest_path: Path,
    *,
    request_limit: int = 0,
) -> FrozenWorkload:
    allowed_counts = {EXPECTED_REQUESTS}
    if request_limit:
        allowed_counts.add(request_limit)
    manifest = validate_workload_manifest(
        workload_manifest_path,
        binding,
        allowed_request_counts=allowed_counts,
    )
    source_requests = int(
        manifest["outputs"]["measurement_workload_csv"]["rows"]
    )
    filters, filter_tuples, atoms = _load_filters(binding)
    truth, query_by_no = core.load_tie_aware_truth(
        binding.truth_csv,
        expected_self_excluded=binding.truth_self_excluded,
        expected_candidate_validity_predicate=binding.candidate_validity_predicate,
    )
    requests = core.load_workload_requests(
        binding.workload_csv,
        query_by_no=query_by_no,
        filters=list(filter_tuples),
        truth=truth,
        expected_requests=source_requests,
        require_unique_queries=True,
    )
    if request_limit:
        if request_limit > len(requests):
            raise Figure5ThroughputError(
                f"workload request limit {request_limit} exceeds audited source "
                f"requests {len(requests)}"
            )
        requests = requests[:request_limit]
    expected_requests = len(requests)
    if any(request.split != "measurement" for request in requests):
        raise Figure5ThroughputError(
            "formal throughput accepts only the frozen measurement split"
        )
    assigned_pairs = {(request.filter_name, request.query_no) for request in requests}
    if len(assigned_pairs) != expected_requests:
        raise Figure5ThroughputError(
            "mixed workload does not contain one unique assigned "
            "filter/query pair per request"
        )
    missing_pairs = assigned_pairs - set(truth)
    if missing_pairs:
        raise Figure5ThroughputError(
            f"exact truth is missing {len(missing_pairs)} assigned workload pairs"
        )
    counts = Counter(request.filter_name for request in requests)
    if set(counts) != set(filters):
        raise Figure5ThroughputError("mixed q10K does not cover all fourteen filters")
    if not request_limit and max(counts.values()) - min(counts.values()) > 1:
        raise Figure5ThroughputError(
            "mixed q10K filter assignment is not balanced within one request"
        )
    for request in requests:
        entry = truth[(request.filter_name, request.query_no)]
        if int(entry.query_id) != request.query_id:
            raise Figure5ThroughputError(
                "assigned truth/query identity changed after workload validation"
            )
    return FrozenWorkload(
        requests=tuple(requests),
        truth=truth,
        filters=filters,
        filter_tuples=filter_tuples,
        filter_atoms=atoms,
        trace_sha256=sha256_file(binding.workload_csv),
        truth_sha256=sha256_file(binding.truth_csv),
        filters_sha256=sha256_file(binding.filters_csv),
        workload_manifest=manifest,
        workload_manifest_sha256=sha256_file(workload_manifest_path),
    )


def _invalid_boolean(label: str) -> bool:
    raise Figure5ThroughputError(f"{label} must be a JSON boolean")


def validate_arm_search_settings(
    settings: ArmSearchSettings,
    *,
    minimum_traversal_target: int = K,
    arm_name: str,
) -> None:
    if not isinstance(settings.traversal_guided_early_stop, bool):
        raise Figure5ThroughputError(
            f"{arm_name} traversal_guided_early_stop must be boolean"
        )
    if not 0.0 <= settings.traversal_guided_early_stop_distance_ratio <= 1.0:
        raise Figure5ThroughputError(
            f"{arm_name} traversal_guided_early_stop_distance_ratio must be in [0, 1]"
        )
    if settings.iterative_scan not in core.ITERATIVE_SCAN_VALUES:
        raise Figure5ThroughputError(f"{arm_name} iterative-scan value is invalid")
    integer_values = (
        settings.ef_search,
        settings.max_scan_tuples,
        settings.guided_collect_target,
        settings.traversal_guided_target,
        settings.traversal_guided_burst,
    )
    if any(value <= 0 for value in integer_values):
        raise Figure5ThroughputError(f"{arm_name} search settings must be positive")
    if not math.isfinite(settings.scan_mem_multiplier) or settings.scan_mem_multiplier <= 0:
        raise Figure5ThroughputError(
            f"{arm_name} scan_mem_multiplier must be finite and positive"
        )
    if settings.guided_collect_target > settings.ef_search:
        raise Figure5ThroughputError(
            f"{arm_name} guided_collect_target cannot exceed ef_search"
        )
    if not minimum_traversal_target <= settings.traversal_guided_target <= settings.ef_search:
        raise Figure5ThroughputError(
            f"{arm_name} traversal_guided_target must lie between the result/self-exclusion "
            "minimum and ef_search"
        )


def validate_search_settings(
    settings: SearchSettings,
    *,
    minimum_traversal_target: int = K,
) -> None:
    if not SAFE_ID_RE.fullmatch(settings.config_id):
        raise Figure5ThroughputError("config_id must match [A-Za-z0-9_.-]{1,96}")
    if not SAFE_ID_RE.fullmatch(settings.pair_id):
        raise Figure5ThroughputError("pair_id must match [A-Za-z0-9_.-]{1,96}")
    if not math.isfinite(settings.target_recall) or not 0.0 < settings.target_recall <= 1.0:
        raise Figure5ThroughputError("target_recall must be finite and in (0, 1]")
    if (
        not math.isfinite(settings.d1_exact_max_selectivity_pct)
        or settings.d1_exact_max_selectivity_pct <= 0
    ):
        raise Figure5ThroughputError(
            "d1_exact_max_selectivity_pct must be finite and positive"
        )
    if (
        not math.isfinite(settings.guidance_selectivity_min_pct)
        or settings.guidance_selectivity_min_pct < 0
    ):
        raise Figure5ThroughputError(
            "guidance_selectivity_min_pct must be finite and non-negative"
        )
    if (
        not math.isfinite(settings.guidance_selectivity_max_pct)
        or settings.guidance_selectivity_max_pct <= 0
    ):
        raise Figure5ThroughputError(
            "guidance_selectivity_max_pct must be finite and positive"
        )
    if settings.guidance_selectivity_min_pct > settings.guidance_selectivity_max_pct:
        raise Figure5ThroughputError(
            "guidance_selectivity_min_pct must not exceed guidance_selectivity_max_pct"
        )
    if (
        not math.isfinite(settings.guidance_composite_max_selectivity_pct)
        or settings.guidance_composite_max_selectivity_pct <= 0
    ):
        raise Figure5ThroughputError(
            "guidance_composite_max_selectivity_pct must be finite and positive"
        )
    if settings.guidance_max_atoms <= 0:
        raise Figure5ThroughputError("guidance_max_atoms must be positive")
    if not isinstance(settings.collapse_exact_and_guidance, bool):
        raise Figure5ThroughputError("collapse_exact_and_guidance must be boolean")
    if not isinstance(settings.d2_source_on_guidance_bypass, bool):
        raise Figure5ThroughputError(
            "d2_source_on_guidance_bypass must be boolean"
        )
    if settings.guidance_bypass_ef_search < 0:
        raise Figure5ThroughputError(
            "guidance_bypass_ef_search must be non-negative"
        )
    if settings.guidance_low_selectivity_bypass_ef_search < 0:
        raise Figure5ThroughputError(
            "guidance_low_selectivity_bypass_ef_search must be non-negative"
        )
    if settings.sqlens.iterative_scan != "off":
        raise Figure5ThroughputError("full SQLens iterative_scan must be off")
    validate_arm_search_settings(
        settings.stock,
        minimum_traversal_target=minimum_traversal_target,
        arm_name="stock",
    )
    validate_arm_search_settings(
        settings.sqlens,
        minimum_traversal_target=minimum_traversal_target,
        arm_name="sqlens",
    )


def config_identity(
    binding: DatasetBinding,
    settings: SearchSettings,
    clients: int,
) -> tuple[dict[str, Any], str]:
    # ``clients`` is deliberately not part of the search configuration hash.
    # Figure 5 measures the same config at several concurrency levels and binds
    # client count as a separate operating-point dimension.
    _ = clients
    value = {
        "dataset": binding.dataset_id,
        "pair_id": settings.pair_id,
        "target_recall": settings.target_recall,
        "modes": list(MODES),
        "arms": ARM_BY_MODE,
        "search": settings.mode_configs(),
        "arm_configs": {
            arm_id: settings.arm_config(arm_id) for arm_id in MODE_BY_ARM
        },
        "per_filter_search": {
            arm_id: settings.arm_filter_config(arm_id) for arm_id in MODE_BY_ARM
        },
        "guidance_filter_strategy": "traversal_guided",
        "d3_measurement_policy": "workload_driven_adaptive",
        "guidance_policy": {
            "guidance_selectivity_min_pct": settings.guidance_selectivity_min_pct,
            "guidance_selectivity_max_pct": settings.guidance_selectivity_max_pct,
            "guidance_composite_max_selectivity_pct": (
                settings.guidance_composite_max_selectivity_pct
            ),
            "guidance_max_atoms": settings.guidance_max_atoms,
            "d1_exact_max_selectivity_pct": settings.d1_exact_max_selectivity_pct,
            "collapse_exact_and_guidance": settings.collapse_exact_and_guidance,
            "d2_source_on_guidance_bypass": settings.d2_source_on_guidance_bypass,
            "guidance_bypass_ef_search": settings.guidance_bypass_ef_search,
            "guidance_low_selectivity_bypass_ef_search": (
                settings.guidance_low_selectivity_bypass_ef_search
            ),
        },
        "source_index": binding.source_index,
        "bfs_index": binding.bfs_index,
        "candidate_validity_predicate": core.effective_candidate_validity_predicate(
            binding.candidate_validity_predicate
        ),
    }
    return value, canonical_sha256(value)


def arm_config_sha256(settings: SearchSettings, arm_id: str) -> str:
    """Bind a per-arm row to its exact independently tuned search settings."""
    return canonical_sha256(
        {
            "pair_id": settings.pair_id,
            "target_recall": settings.target_recall,
            "arm_id": arm_id,
            "mode_id": MODE_BY_ARM[arm_id],
            "search": settings.arm_config(arm_id),
            "per_filter_search": settings.arm_filter_config(arm_id),
        }
    )


def balanced_arm_order(repeat_id: int, seed: int) -> tuple[str, str]:
    if repeat_id < 0:
        raise Figure5ThroughputError("repeat_id must be non-negative")
    base = list(ARM_BY_MODE)
    random.Random(telemetry.stable_seed(seed, "figure5_arm_order")).shuffle(base)
    if repeat_id % 2:
        base.reverse()
    return tuple(base)  # type: ignore[return-value]


def validate_balanced_schedule(
    repeats: int,
    seed: int,
    *,
    allow_single_pass: bool = False,
) -> list[tuple[str, str]]:
    minimum_repeats = 1 if allow_single_pass else MIN_REPEATS
    if repeats < minimum_repeats:
        raise Figure5ThroughputError(
            f"formal throughput requires at least {MIN_REPEATS} repeats"
        )
    schedule = [balanced_arm_order(repeat, seed) for repeat in range(repeats)]
    if repeats == 1 and allow_single_pass:
        return schedule
    first_counts = Counter(order[0] for order in schedule)
    if set(first_counts) != set(ARM_BY_MODE) or (
        max(first_counts.values()) - min(first_counts.values()) > 1
    ):
        raise Figure5ThroughputError("paired arm order is not balanced")
    return schedule


def request_dispatch(
    workload: FrozenWorkload,
    *,
    schedule_seed: int,
    dataset_id: str,
    config_id: str,
    clients: int,
    repeat_id: int,
) -> tuple[int, str, list[tuple[int, core.WorkloadRequest]]]:
    seed = telemetry.stable_seed(
        schedule_seed,
        "figure5_mixed_q10k",
        dataset_id,
        config_id,
        clients,
        repeat_id,
    )
    requests = list(workload.requests)
    random.Random(seed).shuffle(requests)
    dispatch = list(enumerate(requests))
    order_sha = canonical_sha256([request.request_no for _, request in dispatch])
    return seed, order_sha, dispatch


def client_cpu_assignment(client_cpu_list: str | None, clients: int) -> tuple[int, ...]:
    if not 1 <= clients <= MAX_CLIENTS:
        raise Figure5ThroughputError(
            f"clients must be in [1, {MAX_CLIENTS}], observed={clients}"
        )
    cpus = telemetry.parse_cpu_set(client_cpu_list)
    if not cpus:
        return ()
    if len(cpus) < clients:
        raise Figure5ThroughputError(
            f"client CPU set has {len(cpus)} CPUs for {clients} clients"
        )
    return tuple(cpus[:clients])


def d3_namespace(run_id: str, dataset_id: str, config_id: str, clients: int, repeat: int) -> str:
    digest = canonical_sha256(
        [run_id, dataset_id, config_id, clients, repeat]
    )[:20]
    namespace = f"f5t_{dataset_id}_{digest}_r{repeat}"
    if len(namespace) > 64 or not re.fullmatch(r"[A-Za-z0-9_.-]+", namespace):
        raise Figure5ThroughputError("generated D3 namespace is invalid")
    return namespace


def load_delegated_d2_proof(
    path: Path,
    binding: DatasetBinding,
    validator: Callable[[dict[str, object], str, str], dict[str, object]] = (
        core.validate_d2_graph_proof
    ),
) -> dict[str, object]:
    proof = read_json(path, "delegated D2 graph proof")
    try:
        return validator(proof, binding.source_index, binding.bfs_index)
    except Exception as exc:
        raise Figure5ThroughputError(f"D2 graph proof gate failed: {exc}") from exc


def fragment_store_count(table: str, namespace: str) -> int:
    with telemetry.psycopg.connect(
        pg_config_from_env().conninfo, autocommit=True
    ) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT count(*)::bigint FROM public.pgvector_hnsw_fragment_store "
            "WHERE heap_oid=%s::regclass::oid "
            "AND left(filter_name, length(%s) + 1) = %s || chr(31)",
            (table, namespace, namespace),
        )
        return int(cursor.fetchone()[0])


def _runtime_args(
    args: argparse.Namespace,
    binding: DatasetBinding,
    workload: FrozenWorkload,
    settings: SearchSettings,
    release: Mapping[str, Any],
) -> argparse.Namespace:
    runtime = argparse.Namespace(**vars(args))
    runtime.insertion_table = binding.table
    runtime.insertion_index = binding.source_index
    runtime.bfs_table = binding.table
    runtime.bfs_index = binding.bfs_index
    runtime.query_table = binding.query_table
    runtime.query_id_column = binding.query_id_column
    runtime.query_vector_column = binding.query_vector_column
    runtime.candidate_validity_predicate = (
        core.effective_candidate_validity_predicate(
            binding.candidate_validity_predicate
        )
    )
    runtime.candidate_validity_predicate_explicit = True
    runtime.expected_truth_self_excluded = binding.truth_self_excluded
    runtime.expected_sqlens_build_id = str(release["expected_sqlens_build_id"])
    runtime.expected_vector_so_sha256 = str(release["expected_vector_so_sha256"])
    runtime.modes = list(MODES)
    runtime.mode_configs_json = settings.mode_configs()
    runtime.filter_ef_search_json = {
        mode: dict(overrides)
        for mode, overrides in settings.filter_ef_search.items()
    }
    runtime.filter_traversal_target_json = {
        mode: dict(overrides)
        for mode, overrides in settings.filter_traversal_target.items()
    }
    runtime.filter_mode_configs_json = {
        mode: {
            filter_name: dict(config)
            for filter_name, config in overrides.items()
        }
        for mode, overrides in settings.filter_mode_configs.items()
    }
    # Core opens each mode with ``mode_configs_json`` below.  These defaults
    # only satisfy its generic Namespace contract; they are not shared tuning.
    runtime.ef_search = settings.stock.ef_search
    runtime.guided_collect_target = settings.stock.guided_collect_target
    runtime.traversal_guided_target = settings.stock.traversal_guided_target
    runtime.traversal_guided_burst = settings.stock.traversal_guided_burst
    runtime.traversal_guided_early_stop = settings.stock.traversal_guided_early_stop
    runtime.traversal_guided_early_stop_distance_ratio = (
        settings.stock.traversal_guided_early_stop_distance_ratio
    )
    runtime.traversal_guided_prioritization = True
    runtime.iterative_scan = "off"
    runtime.max_scan_tuples = settings.stock.max_scan_tuples
    runtime.scan_mem_multiplier = settings.stock.scan_mem_multiplier
    runtime.guidance_filter_strategy = "traversal_guided"
    runtime.filter_atoms = dict(workload.filter_atoms)
    runtime.filter_predicate_by_name = {
        name: item.predicate for name, item in workload.filters.items()
    }
    runtime.filter_selectivity_by_name = {
        name: item.actual_pct for name, item in workload.filters.items()
    }
    runtime.guidance_selectivity_min_pct = settings.guidance_selectivity_min_pct
    runtime.guidance_selectivity_max_pct = settings.guidance_selectivity_max_pct
    runtime.guidance_composite_max_selectivity_pct = (
        settings.guidance_composite_max_selectivity_pct
    )
    runtime.guidance_max_atoms = settings.guidance_max_atoms
    runtime.d1_guidance_kind = "auto"
    runtime.d1_exact_max_selectivity_pct = settings.d1_exact_max_selectivity_pct
    runtime.collapse_exact_and_guidance = settings.collapse_exact_and_guidance
    runtime.d1_cache_mb = args.d1_cache_mb
    runtime.d3_cache_mb = args.d3_cache_mb
    runtime.d3_measurement_policy = "workload_driven_adaptive"
    runtime.d3_probe_requests = args.d3_probe_requests
    runtime.d3_min_benefit_per_byte = args.d3_min_benefit_per_byte
    runtime.d3_max_fragment_mb = args.d3_max_fragment_mb
    runtime.d3_page_min_skip_rate = args.d3_page_min_skip_rate
    runtime.d3_fragment_store_namespace = ""
    runtime.d2_page_access = args.d2_page_access
    runtime.d2_index_page_access = args.d2_index_page_access
    runtime.d2_page_window = args.d2_page_window
    runtime.d2_page_prefetch_min_items = args.d2_page_prefetch_min_items
    runtime.d2_page_disable_after_no_merge = args.d2_page_disable_after_no_merge
    runtime.d2_source_on_guidance_bypass = settings.d2_source_on_guidance_bypass
    runtime.guidance_bypass_iterative_scan = args.guidance_bypass_iterative_scan
    runtime.guidance_bypass_ef_search = settings.guidance_bypass_ef_search
    runtime.guidance_low_selectivity_bypass_ef_search = (
        settings.guidance_low_selectivity_bypass_ef_search
    )
    runtime.preferred_index_guc = "hnsw.preferred_index"
    runtime.require_preferred_index_guc = True
    runtime.statement_timeout_ms = args.statement_timeout_ms
    runtime.force_hnsw = True
    runtime.reset_cache_per_query = False
    runtime.fragment_tracking_prepared = True
    runtime.plan_query_id = None
    runtime.plan_evidence = []
    runtime.warmup_evidence = []
    runtime.d3_phase_evidence = []
    runtime.d3_warmup_phase_evidence = []
    runtime.backend_cpu_evidence = []
    runtime.runtime_sqlens_identity_evidence = []
    runtime.k = K
    return runtime


def validate_execution_args(args: argparse.Namespace, repeats: int) -> tuple[int, ...]:
    schedule = validate_balanced_schedule(
        repeats,
        args.schedule_seed,
        allow_single_pass=bool(getattr(args, "allow_single_pass", False)),
    )
    if len(schedule) != repeats:
        raise Figure5ThroughputError("internal arm schedule length mismatch")
    cpus = client_cpu_assignment(args.client_cpu_list, args.clients)
    if (
        not math.isfinite(args.d3_min_benefit_per_byte)
        or args.d3_min_benefit_per_byte < 0
    ):
        raise Figure5ThroughputError(
            "d3_min_benefit_per_byte must be finite and non-negative"
        )
    if (
        not math.isfinite(args.d3_page_min_skip_rate)
        or not 0.0 <= args.d3_page_min_skip_rate <= 1.0
    ):
        raise Figure5ThroughputError(
            "d3_page_min_skip_rate must be finite and in [0, 1]"
        )
    if args.execute:
        if not cpus:
            raise Figure5ThroughputError(
                "formal --execute requires explicit client CPU pinning"
            )
        if not args.backend_cpu_list:
            raise Figure5ThroughputError(
                "formal --execute requires separately configured --backend-cpu-list"
            )
        if not args.telemetry_devices and not args.telemetry_path:
            raise Figure5ThroughputError(
                "formal --execute requires telemetry device or path binding"
            )
    return cpus


def validate_independent_backends(runtimes: Sequence[Any], clients: int) -> list[int]:
    pids = [
        int((runtime.backend_cpu_provenance or {}).get("backend_pid") or 0)
        for runtime in runtimes
    ]
    if (
        len(pids) != clients
        or any(pid <= 0 for pid in pids)
        or len(set(pids)) != clients
    ):
        raise Figure5ThroughputError(
            f"each client must own an independent PostgreSQL backend: {pids}"
        )
    return pids


def _has_cross_client_overlap(rows: Sequence[Mapping[str, Any]], clients: int) -> bool:
    if clients == 1:
        return True
    intervals = sorted(
        (
            float(row["started_offset_ms"]),
            float(row["completed_offset_ms"]),
            int(row["client_id"]),
        )
        for row in rows
    )
    if {client_id for _, _, client_id in intervals} != set(range(clients)):
        return False
    latest_completion_by_client: dict[int, float] = {}
    for started, completed, client_id in intervals:
        if any(
            other_client != client_id and other_completed > started
            for other_client, other_completed in latest_completion_by_client.items()
        ):
            return True
        latest_completion_by_client[client_id] = max(
            completed,
            latest_completion_by_client.get(client_id, 0.0),
        )
    return False


def validate_arm_rows(
    rows: Sequence[Mapping[str, Any]],
    workload: FrozenWorkload,
    clients: int,
    repeat_id: int,
    trace_seed: int,
) -> bool:
    expected_requests = len(workload.requests)
    if len(rows) != expected_requests:
        raise Figure5ThroughputError(
            f"arm has {len(rows)} rows, expected {expected_requests}"
        )
    by_request = {request.request_no: request for request in workload.requests}
    request_nos = [int(row["request_no"]) for row in rows]
    positions = [int(row["dispatch_position"]) for row in rows]
    if (
        len(set(request_nos)) != expected_requests
        or set(request_nos) != set(range(expected_requests))
        or len(set(positions)) != expected_requests
        or set(positions) != set(range(expected_requests))
    ):
        raise Figure5ThroughputError("arm request/dispatch coverage is not exact q10K")
    expected_client_counts = Counter(
        position % clients for position in range(expected_requests)
    )
    observed_client_counts = Counter(int(row["client_id"]) for row in rows)
    if observed_client_counts != expected_client_counts:
        raise Figure5ThroughputError("deterministic per-client assignment is incomplete")
    for row in rows:
        request = by_request[int(row["request_no"])]
        if (
            int(row["repeat_id"]) != repeat_id
            or int(row["trace_permutation_seed"]) != trace_seed
            or int(row["query_no"]) != request.query_no
            or int(row["query_id"]) != request.query_id
            or str(row["filter_name"]) != request.filter_name
            or int(row["trace_cycle"]) != request.trace_cycle
            or int(row["client_id"]) != int(row["dispatch_position"]) % clients
        ):
            raise Figure5ThroughputError(
                "request row differs from its frozen workload/dispatch identity"
            )
    overlap = _has_cross_client_overlap(rows, clients)
    if clients > 1 and not overlap:
        raise Figure5ThroughputError(
            "multi-client arm has no observed cross-client request overlap"
        )
    return overlap


def summarize_repeat(
    rows: Sequence[Mapping[str, Any]],
    *,
    wall_seconds: float,
    run_id: str,
    binding: DatasetBinding,
    settings: SearchSettings,
    config_sha256: str,
    release_identity_sha256: str,
    arm_id: str,
    mode_id: str,
    arm_order: int,
    repeat_id: int,
    clients: int,
    request_trace_sha256: str,
    trace_seed: int,
    trace_order_sha256: str,
    backend_pids: Sequence[int],
    backend_cpu_provenance: Sequence[Mapping[str, Any]],
    client_affinity: Sequence[Mapping[str, Any]],
    true_concurrency_observed: bool,
    namespace: str,
    namespace_rows_before: int,
    namespace_rows_after: int,
    arm_telemetry: Mapping[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
    expected_requests: int = EXPECTED_REQUESTS,
) -> dict[str, Any]:
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise Figure5ThroughputError("repeat wall-clock must be finite and positive")
    if arm_id not in MODE_BY_ARM or MODE_BY_ARM[arm_id] != mode_id:
        raise Figure5ThroughputError("repeat arm/mode identity is invalid")
    completed_rows = [row for row in rows if not row.get("error")]
    completed = len(completed_rows)
    errors = len(rows) - completed
    recalls = [
        0.0 if row.get("error") else float(row["recall_at_10"]) for row in rows
    ]
    latencies = [float(row["latency_ms"]) for row in rows]
    if not recalls or not latencies or any(
        not math.isfinite(value) or value < 0 for value in (*recalls, *latencies)
    ):
        raise Figure5ThroughputError("repeat contains invalid recall/latency samples")
    recall_low, recall_high = telemetry.bootstrap_mean_ci(
        recalls,
        bootstrap_samples,
        telemetry.stable_seed(
            bootstrap_seed,
            binding.dataset_id,
            settings.config_id,
            clients,
            repeat_id,
            arm_id,
            "recall",
        ),
    )
    telemetry_fields = telemetry.telemetry_summary_fields(arm_telemetry)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": binding.dataset_id,
        "experiment_kind": "throughput",
        "arm_id": arm_id,
        "mode_id": mode_id,
        "pair_id": settings.pair_id,
        "target_recall": settings.target_recall,
        "config_id": settings.config_id,
        "config_sha256": config_sha256,
        "stock_config_sha256": arm_config_sha256(settings, "stock_pgvector"),
        "sqlens_config_sha256": arm_config_sha256(settings, "sqlens_full"),
        "arm_config_sha256": arm_config_sha256(settings, arm_id),
        "release_identity_sha256": release_identity_sha256,
        "clients": clients,
        "repeat_id": repeat_id,
        "request_trace_sha256": request_trace_sha256,
        "requests": len(rows),
        "unique_queries": len({int(row["query_id"]) for row in rows}),
        "completed_queries": completed,
        "error_count": errors,
        "wall_clock_seconds": wall_seconds,
        "recall_mean": statistics.fmean(recalls),
        "recall_ci95_low": recall_low,
        "recall_ci95_high": recall_high,
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p95_ms": telemetry.percentile(latencies, 0.95),
        "latency_p99_ms": telemetry.percentile(latencies, 0.99),
        "throughput_qps": completed / wall_seconds,
        "throughput_ci95_low": "",
        "throughput_ci95_high": "",
        "throughput_source": THROUGHPUT_SOURCE,
        "status": (
            "valid"
            if len(rows) == expected_requests
            and len({int(row["query_id"]) for row in rows}) == expected_requests
            and completed == expected_requests
            and errors == 0
            and recall_low >= settings.target_recall
            else "invalid"
        ),
        "runner_version": RUNNER_VERSION,
        "arm_order": arm_order,
        "trace_permutation_seed": trace_seed,
        "trace_order_sha256": trace_order_sha256,
        "backend_pids_json": json.dumps([int(pid) for pid in backend_pids]),
        "backend_cpu_provenance_json": json.dumps(
            [dict(item) for item in backend_cpu_provenance],
            sort_keys=True,
        ),
        "client_affinity_json": json.dumps(
            sorted(
                (dict(item) for item in client_affinity),
                key=lambda item: int(item["client_id"]),
            ),
            sort_keys=True,
        ),
        "true_concurrency_observed": true_concurrency_observed,
        "d3_measurement_policy": "workload_driven_adaptive",
        "d3_fragment_store_namespace": namespace,
        "d3_namespace_rows_before": namespace_rows_before,
        "d3_namespace_rows_after": namespace_rows_after,
        "d3_online_cost_charged": mode_id == "design1_bloom_bfs_layout_d3",
        **telemetry_fields,
    }


def _execute_search(
    runtime_args: argparse.Namespace,
    runtime: Any,
    request: core.WorkloadRequest,
    workload: FrozenWorkload,
) -> dict[str, Any]:
    filter_spec = workload.filters[request.filter_name]
    truth_entry = workload.truth[(request.filter_name, request.query_no)]
    started = time.perf_counter()
    activation_finished = started
    ids: list[int] = []
    distances: list[float] = []
    error = ""
    error_type = ""
    try:
        _, _, previous_guidance_policy, reset_performed = (
            core.route_runtime_request(runtime_args, runtime, filter_spec.name)
        )
        activation_profile = core.activate(
            runtime.cur,
            runtime_args,
            runtime.mode,
            filter_spec.name,
            read_profile=False,
            reset_bypass_guidance=(
                previous_guidance_policy and not reset_performed
            ),
            configure_search_strategy=False,
        )
        activation_finished = time.perf_counter()
        table = str(activation_profile["table"])
        binding = core.activation_binding(
            runtime_args,
            runtime.mode,
            filter_spec.name,
            activation_profile,
        )
        self_exclusion = core.candidate_self_exclusion(runtime_args, table)
        ids, distances, _ = core.run_query(
            runtime.cur,
            table,
            filter_spec.predicate,
            request.query_id,
            runtime_args.k,
            binding,
            core.uses_exact_predicate_scan_contract(
                runtime_args.guidance_filter_strategy
            )
            and self_exclusion,
            candidate_validity_predicate=(
                runtime_args.candidate_validity_predicate
            ),
            query_table=core.query_table_for_candidate(runtime_args, table),
            query_id_column=runtime_args.query_id_column,
            query_vector_column=runtime_args.query_vector_column,
            self_exclusion=self_exclusion,
            reset_profile=False,
            read_profile=False,
        )
    except Exception as exc:  # noqa: BLE001 - errors are counted per request
        error_type = exc.__class__.__name__
        error = f"{error_type}: {exc}"
        try:
            core.recover_runtime(runtime_args, runtime)
        except Exception as recovery_exc:  # noqa: BLE001
            error += (
                f"; recovery={recovery_exc.__class__.__name__}: {recovery_exc}"
            )
    finished = time.perf_counter()
    recall = (
        core.tie_aware_recall(distances, truth_entry, runtime_args.k)
        if not error
        else 0.0
    )
    return {
        "query_no": request.query_no,
        "query_id": request.query_id,
        "latency_ms": (finished - started) * 1000.0,
        "activation_ms": (activation_finished - started) * 1000.0,
        "query_ms": (finished - activation_finished) * 1000.0,
        "returned": len(ids),
        "returned_ids": ",".join(str(value) for value in ids),
        "recall_at_10": recall,
        "error_type": error_type,
        "error": error,
    }


def run_arm(
    args: argparse.Namespace,
    runtime_args: argparse.Namespace,
    binding: DatasetBinding,
    workload: FrozenWorkload,
    settings: SearchSettings,
    *,
    run_id: str,
    config_sha256: str,
    release_identity_sha256: str,
    mode_id: str,
    arm_order: int,
    repeat_id: int,
    trace_seed: int,
    trace_order_sha256: str,
    dispatch: Sequence[tuple[int, core.WorkloadRequest]],
    client_cpus: tuple[int, ...],
    namespace: str,
    namespace_rows_before: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if mode_id not in ARM_BY_MODE:
        raise Figure5ThroughputError(f"unsupported arm mode: {mode_id}")
    arm_id = ARM_BY_MODE[mode_id]
    runtime_args.d3_fragment_store_namespace = namespace if arm_id == "sqlens_full" else ""
    runtimes: list[Any] = []
    try:
        for _ in range(args.clients):
            runtimes.append(
                core.open_mode_runtime(
                    runtime_args,
                    mode_id,
                    list(workload.filter_tuples),
                )
            )
        backend_pids = validate_independent_backends(runtimes, args.clients)
        backend_cpu_provenance = [
            dict(runtime.backend_cpu_provenance or {}) for runtime in runtimes
        ]
        _, expected_index = core.mode_table_index(runtime_args, mode_id)
        barrier = threading.Barrier(args.clients + 1)
        arm_started = 0.0

        def worker(
            client_id: int,
            runtime: Any,
        ) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
            try:
                affinity = telemetry.pin_current_thread(client_cpus, client_id)
                barrier.wait(timeout=args.start_barrier_timeout_seconds)
            except BaseException:
                barrier.abort()
                raise
            client_rows: list[dict[str, Any]] = []
            for dispatch_position, request in dispatch[client_id::args.clients]:
                started_offset = (time.perf_counter() - arm_started) * 1000.0
                result = _execute_search(runtime_args, runtime, request, workload)
                completed_offset = (time.perf_counter() - arm_started) * 1000.0
                client_rows.append(
                    {
                        "runner_version": RUNNER_VERSION,
                        "run_id": run_id,
                        "dataset": binding.dataset_id,
                        "pair_id": settings.pair_id,
                        "target_recall": settings.target_recall,
                        "config_id": settings.config_id,
                        "config_sha256": config_sha256,
                        "stock_config_sha256": arm_config_sha256(
                            settings, "stock_pgvector"
                        ),
                        "sqlens_config_sha256": arm_config_sha256(
                            settings, "sqlens_full"
                        ),
                        "arm_config_sha256": arm_config_sha256(settings, arm_id),
                        "release_identity_sha256": release_identity_sha256,
                        "arm_id": arm_id,
                        "mode_id": mode_id,
                        "arm_order": arm_order,
                        "repeat_id": repeat_id,
                        "clients": args.clients,
                        "trace_permutation_seed": trace_seed,
                        "trace_order_sha256": trace_order_sha256,
                        "request_trace_sha256": workload.trace_sha256,
                        "dispatch_position": dispatch_position,
                        "request_no": request.request_no,
                        "trace_cycle": request.trace_cycle,
                        "filter_name": request.filter_name,
                        "client_id": client_id,
                        "backend_pid": backend_pids[client_id],
                        "client_native_tid": affinity["native_tid"],
                        "client_requested_cpu": affinity["requested_cpu"],
                        "client_affinity_applied": affinity["affinity_applied"],
                        "started_offset_ms": started_offset,
                        "completed_offset_ms": completed_offset,
                        "d3_fragment_store_namespace": (
                            namespace if arm_id == "sqlens_full" else ""
                        ),
                        **result,
                    }
                )
            return (
                client_rows,
                {
                    "client_id": client_id,
                    **affinity,
                },
                time.perf_counter(),
            )

        telemetry.flush_postgres_stats(runtimes)
        postgres_before = telemetry.postgres_telemetry_snapshot(
            runtimes[0].cur,
            binding.table,
            expected_index,
        )
        backend_before = telemetry.backend_cpu_snapshot(
            backend_pids,
            proc_root=args.backend_proc_root,
        )
        rows: list[dict[str, Any]] = []
        affinity_evidence: list[dict[str, Any]] = []
        worker_finished_at: list[float] = []
        with ThreadPoolExecutor(
            max_workers=args.clients,
            thread_name_prefix="figure5-throughput",
        ) as pool:
            futures = [
                pool.submit(worker, client_id, runtime)
                for client_id, runtime in enumerate(runtimes)
            ]
            host_before = telemetry.host_telemetry_snapshot(
                args.telemetry_devices_resolved
            )
            arm_started = time.perf_counter()
            try:
                barrier.wait(timeout=args.start_barrier_timeout_seconds)
            except threading.BrokenBarrierError as exc:
                raise Figure5ThroughputError("client start barrier failed") from exc
            for future in as_completed(futures):
                client_rows, affinity, finished_at = future.result()
                rows.extend(client_rows)
                affinity_evidence.append(affinity)
                worker_finished_at.append(finished_at)
        if len(worker_finished_at) != args.clients:
            raise Figure5ThroughputError(
                "not every client reported its barrier-window completion"
            )
        wall_seconds = max(worker_finished_at) - arm_started
        host_after = telemetry.host_telemetry_snapshot(
            args.telemetry_devices_resolved
        )
        telemetry.flush_postgres_stats(runtimes)
        postgres_after = telemetry.postgres_telemetry_snapshot(
            runtimes[0].cur,
            binding.table,
            expected_index,
        )
        backend_after = telemetry.backend_cpu_snapshot(
            backend_pids,
            proc_root=args.backend_proc_root,
        )
        rows.sort(key=lambda row: int(row["dispatch_position"]))
        true_overlap = validate_arm_rows(
            rows,
            workload,
            args.clients,
            repeat_id,
            trace_seed,
        )
        telemetry.validate_measurement_arm_timing(rows, wall_seconds)
        arm_telemetry = {
            "host": telemetry.host_telemetry_delta(host_before, host_after),
            "postgresql": telemetry.postgres_telemetry_delta(
                postgres_before, postgres_after
            ),
            "backend_cpu": telemetry.backend_cpu_delta(
                backend_before, backend_after
            ),
            "backend_proc_root": str(args.backend_proc_root),
            "devices": list(args.telemetry_devices_resolved),
            "postgres_stats_force_flushed_per_backend": True,
            "measurement_wall_clock_seconds": wall_seconds,
        }
        telemetry.validate_arm_telemetry(arm_telemetry, backend_pids)
        namespace_rows_after = (
            fragment_store_count(binding.table, namespace)
            if arm_id == "sqlens_full"
            else namespace_rows_before
        )
        summary = summarize_repeat(
            rows,
            wall_seconds=wall_seconds,
            run_id=run_id,
            binding=binding,
            settings=settings,
            config_sha256=config_sha256,
            release_identity_sha256=release_identity_sha256,
            arm_id=arm_id,
            mode_id=mode_id,
            arm_order=arm_order,
            repeat_id=repeat_id,
            clients=args.clients,
            request_trace_sha256=workload.trace_sha256,
            trace_seed=trace_seed,
            trace_order_sha256=trace_order_sha256,
            backend_pids=backend_pids,
            backend_cpu_provenance=backend_cpu_provenance,
            client_affinity=affinity_evidence,
            true_concurrency_observed=true_overlap,
            namespace=(namespace if arm_id == "sqlens_full" else ""),
            namespace_rows_before=namespace_rows_before,
            namespace_rows_after=namespace_rows_after,
            arm_telemetry=arm_telemetry,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            expected_requests=len(workload.requests),
        )
        evidence = {
            "arm_id": arm_id,
            "mode_id": mode_id,
            "repeat_id": repeat_id,
            "arm_order": arm_order,
            "backend_pids": backend_pids,
            "backend_cpu_provenance": backend_cpu_provenance,
            "independent_connections": True,
            "true_concurrency_observed": true_overlap,
            "trace_permutation_seed": trace_seed,
            "trace_order_sha256": trace_order_sha256,
            "client_affinity": affinity_evidence,
            "d3_namespace": namespace if arm_id == "sqlens_full" else "",
            "d3_namespace_rows_before": namespace_rows_before,
            "d3_namespace_rows_after": namespace_rows_after,
            "unmeasured_query_count": 0,
            "telemetry": arm_telemetry,
        }
        return rows, summary, evidence
    finally:
        for runtime in reversed(runtimes):
            core.close_mode_runtime(runtime)


def validate_paired_repeats(
    summaries: Sequence[Mapping[str, Any]],
    repeats: int,
) -> dict[str, Any]:
    expected = {
        (repeat, arm)
        for repeat in range(repeats)
        for arm in ("stock_pgvector", "sqlens_full")
    }
    observed = [(int(row["repeat_id"]), str(row["arm_id"])) for row in summaries]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise Figure5ThroughputError("paired repeat/arm coverage is incomplete")
    by_repeat: dict[int, list[Mapping[str, Any]]] = {}
    for repeat in range(repeats):
        rows = [row for row in summaries if int(row["repeat_id"]) == repeat]
        by_repeat[repeat] = rows
        if len(rows) != 2:
            raise Figure5ThroughputError(f"repeat {repeat} does not contain two arms")
        for field in (
            "run_id",
            "dataset",
            "pair_id",
            "target_recall",
            "config_id",
            "config_sha256",
            "stock_config_sha256",
            "sqlens_config_sha256",
            "release_identity_sha256",
            "clients",
            "request_trace_sha256",
            "trace_permutation_seed",
            "trace_order_sha256",
            "requests",
            "unique_queries",
        ):
            if len({str(row[field]) for row in rows}) != 1:
                raise Figure5ThroughputError(
                    f"repeat {repeat} paired identity mismatch: {field}"
                )
    first_positions = Counter(
        min(rows, key=lambda row: int(row["arm_order"]))["arm_id"]
        for rows in by_repeat.values()
    )
    if max(first_positions.values()) - min(first_positions.values()) > 1:
        raise Figure5ThroughputError("measured arm order is not balanced")
    return {
        "passed": True,
        "paired_repeats": repeats,
        "paired_arms": len(summaries),
        "first_position_counts": dict(first_positions),
        "identical_trace_permutation_within_repeat": True,
    }


def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        serialized = {}
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, float):
                value = format(value, ".17g")
            serialized[field] = value
        writer.writerow(serialized)
    return output.getvalue().encode("utf-8")


def output_paths(prefix: Path) -> dict[str, Path]:
    return {
        "requests": Path(str(prefix) + ".requests.csv"),
        "repeats": Path(str(prefix) + ".repeats.csv"),
        "manifest": Path(str(prefix) + ".manifest.json"),
    }


def execution_source_bindings(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    core_source = Path(__file__).resolve()
    bindings = {
        "throughput_core": {
            "path": str(core_source),
            "sha256": sha256_file(core_source),
        }
    }
    orchestrator = getattr(args, "orchestrator_source", None)
    if orchestrator is not None:
        orchestrator_source = Path(orchestrator).resolve()
        if not orchestrator_source.is_file():
            raise Figure5ThroughputError(
                f"orchestrator source is missing: {orchestrator_source}"
            )
        bindings["orchestrator"] = {
            "path": str(orchestrator_source),
            "sha256": sha256_file(orchestrator_source),
        }
    return bindings


def publish_bundle(
    paths: Mapping[str, Path],
    payloads: Mapping[str, bytes],
    *,
    overwrite: bool,
) -> None:
    parent = paths["manifest"].parent
    parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise Figure5ThroughputError(
            "output already exists; pass --overwrite only after auditing: "
            + ", ".join(str(path) for path in existing)
        )
    stage = Path(tempfile.mkdtemp(prefix=".figure5-throughput-", dir=parent))
    backup = stage / "backup"
    backup.mkdir()
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    installed: list[str] = []
    try:
        for name, payload in payloads.items():
            path = stage / paths[name].name
            with path.open("wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            staged[name] = path
        for name in ("requests", "repeats", "manifest"):
            destination = paths[name]
            if destination.exists():
                saved = backup / destination.name
                os.replace(destination, saved)
                backups[name] = saved
            os.replace(staged[name], destination)
            installed.append(name)
    except Exception:
        for name in reversed(installed):
            if paths[name].exists():
                paths[name].unlink()
            if name in backups and backups[name].exists():
                os.replace(backups[name], paths[name])
        for name, saved in backups.items():
            if name not in installed and saved.exists():
                os.replace(saved, paths[name])
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def prewarm_relations(binding: DatasetBinding) -> dict[str, Any]:
    return core.prewarm_relations(
        list(
            dict.fromkeys(
                (
                    binding.table,
                    binding.source_index,
                    binding.bfs_index,
                    binding.query_table,
                )
            )
        )
    )


def execute_benchmark(
    args: argparse.Namespace,
    binding: DatasetBinding,
    protocol: Mapping[str, Any],
    metadata: Mapping[str, Any],
    workload: FrozenWorkload,
    settings: SearchSettings,
    delegated_proof: Mapping[str, Any],
    repeats: int,
    client_cpus: tuple[int, ...],
) -> tuple[dict[str, Path], dict[str, Any]]:
    release = metadata["release"]
    runtime_args = _runtime_args(args, binding, workload, settings, release)
    runtime_args.client_cpu_list = args.client_cpu_list
    runtime_args.backend_proc_root = args.backend_proc_root
    runtime_args.telemetry_devices_resolved, telemetry_binding = (
        telemetry.resolve_telemetry_devices(
            args.telemetry_devices,
            args.telemetry_path,
        )
    )
    args.telemetry_devices_resolved = runtime_args.telemetry_devices_resolved
    runtime_args.telemetry_devices_resolved = runtime_args.telemetry_devices_resolved
    try:
        core.validate_query_source_contract(runtime_args)
    except Exception as exc:
        raise Figure5ThroughputError(f"query source/truth contract failed: {exc}") from exc
    runtime_identity_start = core.require_exact_sqlens_identity_from_env(
        str(release["expected_sqlens_build_id"]),
        str(release["expected_vector_so_sha256"]),
    )
    tracking = prepare_fragment_tracking(runtime_args)
    runtime_args.fragment_tracking_prepared = True
    live_d2_start = core.require_d2_graph_proof_from_env(
        runtime_args, dict(delegated_proof)
    )
    guard_connection = None
    requests_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    arm_evidence: list[dict[str, Any]] = []
    namespaces: list[dict[str, Any]] = []
    config_value, config_sha = config_identity(binding, settings, args.clients)
    run_id = args.run_id or (
        f"f5t-{binding.dataset_id}-{settings.config_id}-c{args.clients}-"
        f"{int(time.time())}-{os.getpid()}"
    )
    if not SAFE_ID_RE.fullmatch(run_id):
        raise Figure5ThroughputError(
            "run_id must match [A-Za-z0-9_.-]{1,96}"
        )
    try:
        guard_connection, guard = acquire_formal_data_guard(runtime_args)
        build_id = str(release["expected_sqlens_build_id"])
        database_start = database_fingerprint(runtime_args, build_id)
        prewarm = (
            prewarm_relations(binding)
            if args.pg_prewarm
            else {"enabled": False, "complete": True, "records": []}
        )
        schedule = validate_balanced_schedule(
            repeats,
            args.schedule_seed,
            allow_single_pass=bool(getattr(args, "allow_single_pass", False)),
        )
        arm_number = 0
        for repeat_id, order in enumerate(schedule):
            namespace = d3_namespace(
                run_id,
                binding.dataset_id,
                settings.config_id,
                args.clients,
                repeat_id,
            )
            before_rows = fragment_store_count(binding.table, namespace)
            if before_rows != 0:
                raise Figure5ThroughputError(
                    f"fresh D3 namespace is not empty: {namespace} rows={before_rows}"
                )
            trace_seed, trace_order_sha, dispatch = request_dispatch(
                workload,
                schedule_seed=args.schedule_seed,
                dataset_id=binding.dataset_id,
                config_id=settings.config_id,
                clients=args.clients,
                repeat_id=repeat_id,
            )
            namespace_evidence = {
                "repeat_id": repeat_id,
                "namespace": namespace,
                "rows_before": before_rows,
                "unmeasured_materialization_requests": 0,
            }
            for mode_id in order:
                arm_rows, summary, evidence = run_arm(
                    args,
                    runtime_args,
                    binding,
                    workload,
                    settings,
                    run_id=run_id,
                    config_sha256=config_sha,
                    release_identity_sha256=str(release["sha256"]),
                    mode_id=mode_id,
                    arm_order=arm_number,
                    repeat_id=repeat_id,
                    trace_seed=trace_seed,
                    trace_order_sha256=trace_order_sha,
                    dispatch=dispatch,
                    client_cpus=client_cpus,
                    namespace=namespace,
                    namespace_rows_before=before_rows,
                )
                requests_rows.extend(arm_rows)
                repeat_rows.append(summary)
                arm_evidence.append(evidence)
                arm_number += 1
            namespace_evidence["rows_after"] = fragment_store_count(
                binding.table, namespace
            )
            namespaces.append(namespace_evidence)
        pair_gate = validate_paired_repeats(repeat_rows, repeats)
        database_end = database_fingerprint(runtime_args, build_id)
    finally:
        if guard_connection is not None:
            try:
                guard_connection.rollback()
            finally:
                guard_connection.close()
    live_d2_end = core.require_d2_graph_proof_from_env(
        runtime_args, dict(delegated_proof)
    )
    runtime_identity_end = core.require_exact_sqlens_identity_from_env(
        str(release["expected_sqlens_build_id"]),
        str(release["expected_vector_so_sha256"]),
    )
    if (
        live_d2_start.get("stable_fingerprint_sha256")
        != live_d2_end.get("stable_fingerprint_sha256")
    ):
        raise Figure5ThroughputError("D2 graph proof changed during the run")
    if stable_runtime_identity(runtime_identity_start) != stable_runtime_identity(
        runtime_identity_end
    ):
        raise Figure5ThroughputError("loaded SQLens binary identity changed during the run")

    request_bytes = _csv_bytes(REQUEST_FIELDS, requests_rows)
    repeat_bytes = _csv_bytes(REPEAT_FIELDS, repeat_rows)
    paths = output_paths(args.out_prefix.resolve())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": (
            "sqlens_figure5_mixed_q10k_throughput_cell"
            if len(workload.requests) == EXPECTED_REQUESTS
            else f"sqlens_figure5_mixed_q{len(workload.requests)}_throughput_cell"
        ),
        "runner_version": RUNNER_VERSION,
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "artifact_valid": all(row["status"] == "valid" for row in repeat_rows),
        "paper_eligible": all(row["status"] == "valid" for row in repeat_rows),
        "dataset": {
            "key": binding.key,
            "dataset_id": binding.dataset_id,
            "label": binding.label,
        },
        "configuration": {
            "config_id": settings.config_id,
            "pair_id": settings.pair_id,
            "target_recall": settings.target_recall,
            "config_sha256": config_sha,
            "stock_config_sha256": arm_config_sha256(settings, "stock_pgvector"),
            "sqlens_config_sha256": arm_config_sha256(settings, "sqlens_full"),
            "value": config_value,
        },
        "release_contract": dict(release),
        "runtime_binary": {
            "expected_build_id": str(release["expected_sqlens_build_id"]),
            "expected_vector_so_sha256": str(release["expected_vector_so_sha256"]),
        },
        "methods": {
            "stock_pgvector": {
                "mode_id": "original",
                "search": settings.arm_config("stock_pgvector"),
                "config_sha256": arm_config_sha256(settings, "stock_pgvector"),
            },
            "sqlens_full": {
                "mode_id": "design1_bloom_bfs_layout_d3",
                "search": settings.arm_config("sqlens_full"),
                "config_sha256": arm_config_sha256(settings, "sqlens_full"),
                "d3_measurement_policy": "workload_driven_adaptive",
                "unmeasured_query_count": 0,
            },
        },
        "protocol": {
            "requests_per_arm_repeat": len(workload.requests),
            "unique_queries_per_arm_repeat": len(workload.requests),
            "filters": EXPECTED_FILTERS,
            "repeats": repeats,
            "clients": args.clients,
            "schedule_seed": args.schedule_seed,
            "balanced_arm_order": True,
            "same_seeded_permutation_across_paired_arms": True,
            "independently_tuned_arms": True,
            "target_recall": settings.target_recall,
            "client_cpu_list": args.client_cpu_list,
            "client_cpu_assignment": list(client_cpus),
            "backend_cpu_list": args.backend_cpu_list,
            "independent_connection_per_client": True,
            "start_barrier": "threading.Barrier(clients+1)",
            "throughput_source": THROUGHPUT_SOURCE,
            "throughput_formula": "completed_queries / barrier_wall_clock_seconds",
            "latency_fields": ["per-request", "mean", "p95", "p99"],
            "recall": "tie-aware exact SQL-valid Recall@10",
            "d3_online_cost": (
                "fresh namespace per repeat; no unmeasured SQLens query; "
                "activation/probe/admission/materialization are inside per-request latency "
                "and barrier wall-clock"
            ),
            "pg_prewarm": bool(args.pg_prewarm),
        },
        "inputs": {
            "execution_sources": getattr(
                args,
                "execution_sources",
                execution_source_bindings(args),
            ),
            "measurement_pair": measurement_plan_binding(args, settings),
            "frontier_config": {
                "path": metadata["config_path"],
                "sha256": metadata["config_sha256"],
            },
            "workload_manifest": {
                "path": str(args.workload_manifest.resolve()),
                "sha256": workload.workload_manifest_sha256,
            },
            "workload_csv": {
                "path": str(binding.workload_csv),
                "sha256": workload.trace_sha256,
                "rows": len(workload.requests),
            },
            "truth_csv": {
                "path": str(binding.truth_csv),
                "sha256": workload.truth_sha256,
                "assigned_pairs": len(workload.requests),
            },
            "filters_csv": {
                "path": str(binding.filters_csv),
                "sha256": workload.filters_sha256,
                "rows": EXPECTED_FILTERS,
            },
            "d2_graph_proof": {
                "path": str(binding.d2_graph_proof_json),
                "sha256": sha256_file(binding.d2_graph_proof_json),
                "stable_fingerprint_sha256": live_d2_start.get(
                    "stable_fingerprint_sha256"
                ),
            },
        },
        "gates": {
            "release_contract": True,
            "runtime_binary_identity_start_end": True,
            "mixed_q10k_unique_queries": True,
            "full_assigned_pair_truth": True,
            "exact_arm_identity": True,
            "independent_per_arm_search_settings": True,
            "minimum_six_repeats": repeats >= MIN_REPEATS,
            "single_pass_override": bool(
                getattr(args, "allow_single_pass", False)
            ),
            "paired_request_permutation": True,
            "balanced_arm_order": True,
            "independent_client_backends": True,
            "barrier_wall_clock_qps": True,
            "telemetry_complete": True,
            "d2_graph_proof_start_end": True,
            "fresh_d3_namespace_per_repeat": True,
            "no_unmeasured_d3_materialization": True,
        },
        "evidence": {
            "pairing": pair_gate,
            "arm_runs": arm_evidence,
            "d3_namespaces": namespaces,
            "fragment_tracking": tracking,
            "runtime_binary_identity_start": runtime_identity_start,
            "runtime_binary_identity_end": runtime_identity_end,
            "formal_data_guard": guard,
            "database_start": database_start,
            "database_end": database_end,
            "d2_graph_proof_start": live_d2_start,
            "d2_graph_proof_end": live_d2_end,
            "telemetry_binding": telemetry_binding,
            "prewarm": prewarm,
        },
        "outputs": {
            "requests": {
                "path": str(paths["requests"]),
                "rows": len(requests_rows),
                "sha256": hashlib.sha256(request_bytes).hexdigest(),
            },
            "repeats": {
                "path": str(paths["repeats"]),
                "rows": len(repeat_rows),
                "sha256": hashlib.sha256(repeat_bytes).hexdigest(),
            },
        },
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_bundle(
        paths,
        {
            "requests": request_bytes,
            "repeats": repeat_bytes,
            "manifest": manifest_bytes,
        },
        overwrite=args.overwrite,
    )
    return paths, manifest


def prospective_manifest(
    args: argparse.Namespace,
    binding: DatasetBinding,
    protocol: Mapping[str, Any],
    metadata: Mapping[str, Any],
    workload: FrozenWorkload,
    settings: SearchSettings,
    delegated_proof: Mapping[str, Any],
    repeats: int,
    client_cpus: tuple[int, ...],
) -> dict[str, Any]:
    config_value, config_sha = config_identity(binding, settings, args.clients)
    schedule = validate_balanced_schedule(
        repeats,
        args.schedule_seed,
        allow_single_pass=bool(getattr(args, "allow_single_pass", False)),
    )
    return {
        "schema_version": 1,
        "artifact_type": "sqlens_figure5_mixed_q10k_throughput_plan",
        "runner_version": RUNNER_VERSION,
        "database_executed": False,
        "dataset": binding.dataset_id,
        "configuration": {
            "config_id": settings.config_id,
            "pair_id": settings.pair_id,
            "target_recall": settings.target_recall,
            "config_sha256": config_sha,
            "stock_config_sha256": arm_config_sha256(settings, "stock_pgvector"),
            "sqlens_config_sha256": arm_config_sha256(settings, "sqlens_full"),
            "value": config_value,
        },
        "release_contract": metadata["release"],
        "runtime_binary": {
            "expected_build_id": str(
                metadata["release"]["expected_sqlens_build_id"]
            ),
            "expected_vector_so_sha256": str(
                metadata["release"].get("expected_vector_so_sha256", "")
            ),
        },
        "execution_sources": getattr(
            args,
            "execution_sources",
            execution_source_bindings(args),
        ),
        "measurement_pair": measurement_plan_binding(args, settings),
        "workload": {
            "requests": len(workload.requests),
            "unique_queries": len({request.query_id for request in workload.requests}),
            "filters": len(workload.filters),
            "trace_sha256": workload.trace_sha256,
            "assigned_truth_pairs": len(
                {(request.filter_name, request.query_no) for request in workload.requests}
            ),
        },
        "d2_graph_proof": {
            "path": str(binding.d2_graph_proof_json),
            "sha256": sha256_file(binding.d2_graph_proof_json),
            "stable_fingerprint_sha256": delegated_proof.get(
                "stable_fingerprint_sha256"
            ),
        },
        "protocol": {
            "clients": args.clients,
            "client_cpu_assignment": list(client_cpus),
            "backend_cpu_list": args.backend_cpu_list,
            "repeats": repeats,
            "arm_order": schedule,
            "same_request_permutation_per_pair": True,
            "throughput_source": THROUGHPUT_SOURCE,
            "d3_measurement_policy": "workload_driven_adaptive",
            "d3_unmeasured_query_count": 0,
        },
        "outputs": {
            name: str(path)
            for name, path in output_paths(args.out_prefix.resolve()).items()
        },
    }


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def unit_interval_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number in [0, 1]") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a finite number in [0, 1]")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


ARM_SEARCH_FIELDS = (
    "ef_search",
    "iterative_scan",
    "max_scan_tuples",
    "scan_mem_multiplier",
    "guided_collect_target",
    "traversal_guided_target",
    "traversal_guided_burst",
    "traversal_guided_early_stop",
    "traversal_guided_early_stop_distance_ratio",
)
_LEGACY_COMMON_ARGUMENTS = (
    "ef_search",
    "max_scan_tuples",
    "scan_mem_multiplier",
    "guided_collect_target",
    "traversal_guided_target",
    "traversal_guided_burst",
)


def _measurement_plan_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise Figure5ThroughputError(f"measurement plan does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as source:
            return [dict(row) for row in csv.DictReader(source)]
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Figure5ThroughputError(f"cannot read measurement plan {path}: {exc}") from exc
    if isinstance(payload, list):
        rows: object = payload
    elif isinstance(payload, Mapping):
        rows = payload.get("pairs", payload.get("rows", payload.get("measurement_plan")))
    else:
        rows = None
    if (
        rows is None
        and isinstance(payload, Mapping)
        and all(isinstance(value, Mapping) for value in payload.values())
    ):
        # Also accept {"pair-a": {...}, "pair-b": {...}} for hand-authored plans.
        rows = [dict(value, pair_id=value.get("pair_id", key)) for key, value in payload.items()]
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise Figure5ThroughputError(
            "measurement plan JSON must contain a pairs/rows array of objects"
        )
    return [dict(row) for row in rows]


def _plan_arm_value(row: Mapping[str, Any], arm: str, field: str) -> Any:
    nested = row.get(arm)
    if isinstance(nested, Mapping) and field in nested:
        return nested[field]
    key = f"{arm}_{field}"
    if key in row:
        return row[key]
    raise Figure5ThroughputError(
        f"measurement plan pair {row.get('pair_id')!r} is missing {key}"
    )


def _plan_arm_value_default(
    row: Mapping[str, Any], arm: str, field: str, default: Any
) -> Any:
    nested = row.get(arm)
    if isinstance(nested, Mapping) and field in nested:
        return nested[field]
    return row.get(f"{arm}_{field}", default)


def _as_plan_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise Figure5ThroughputError(f"measurement plan {field} must be an integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise Figure5ThroughputError(
            f"measurement plan {field} must be an integer"
        ) from exc
    if parsed <= 0:
        raise Figure5ThroughputError(f"measurement plan {field} must be positive")
    return parsed


def _as_plan_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise Figure5ThroughputError(f"measurement plan {field} must be a number")
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise Figure5ThroughputError(
            f"measurement plan {field} must be a number"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise Figure5ThroughputError(
            f"measurement plan {field} must be finite and positive"
        )
    return parsed


def _as_plan_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise Figure5ThroughputError(f"measurement plan {field} must be a boolean")


def _arm_settings_from_plan(row: Mapping[str, Any], arm: str) -> ArmSearchSettings:
    prefix = f"{arm}_"
    iterative_scan = str(_plan_arm_value(row, arm, "iterative_scan"))
    return ArmSearchSettings(
        ef_search=_as_plan_int(_plan_arm_value(row, arm, "ef_search"), prefix + "ef_search"),
        iterative_scan=iterative_scan,
        max_scan_tuples=_as_plan_int(
            _plan_arm_value(row, arm, "max_scan_tuples"), prefix + "max_scan_tuples"
        ),
        scan_mem_multiplier=_as_plan_float(
            _plan_arm_value(row, arm, "scan_mem_multiplier"),
            prefix + "scan_mem_multiplier",
        ),
        guided_collect_target=_as_plan_int(
            _plan_arm_value(row, arm, "guided_collect_target"),
            prefix + "guided_collect_target",
        ),
        traversal_guided_target=_as_plan_int(
            _plan_arm_value(row, arm, "traversal_guided_target"),
            prefix + "traversal_guided_target",
        ),
        traversal_guided_burst=_as_plan_int(
            _plan_arm_value(row, arm, "traversal_guided_burst"),
            prefix + "traversal_guided_burst",
        ),
        traversal_guided_early_stop=_as_plan_bool(
            _plan_arm_value(row, arm, "traversal_guided_early_stop"),
            prefix + "traversal_guided_early_stop",
        ),
        traversal_guided_early_stop_distance_ratio=float(
            _plan_arm_value_default(
                row, arm, "traversal_guided_early_stop_distance_ratio", 0.0
            ) or 0.0
        ),
    )


def _cli_arm_settings(
    args: argparse.Namespace,
    arm: str,
    grid: Mapping[str, Any],
    minimum_traversal_target: int,
) -> ArmSearchSettings:
    prefix = f"{arm}_"
    legacy_used = [
        name for name in _LEGACY_COMMON_ARGUMENTS if getattr(args, name, None) is not None
    ]
    if legacy_used and not args.allow_equal_arm_settings:
        raise Figure5ThroughputError(
            "legacy common search options require --allow-equal-arm-settings: "
            + ", ".join("--" + name.replace("_", "-") for name in legacy_used)
        )

    def value(name: str, default: Any) -> Any:
        explicit = getattr(args, prefix + name)
        if explicit is not None:
            return explicit
        legacy = getattr(args, name, None)
        if legacy is not None:
            return legacy
        return default

    ef_search = value("ef_search", None)
    if ef_search is None:
        raise Figure5ThroughputError(
            "explicit CLI fallback requires both --stock-ef-search and --sqlens-ef-search"
        )
    iterative_scan = getattr(args, prefix + "iterative_scan")
    if iterative_scan is None:
        iterative_scan = "off"
    return ArmSearchSettings(
        ef_search=int(ef_search),
        iterative_scan=str(iterative_scan),
        max_scan_tuples=int(value("max_scan_tuples", grid["max_scan_tuples"])),
        scan_mem_multiplier=float(
            value("scan_mem_multiplier", grid["scan_mem_multiplier"])
        ),
        guided_collect_target=int(value("guided_collect_target", ef_search)),
        traversal_guided_target=int(
            value(
                "traversal_guided_target",
                max(minimum_traversal_target, min(40, int(ef_search))),
            )
        ),
        traversal_guided_burst=int(value("traversal_guided_burst", 8)),
        traversal_guided_early_stop=bool(
            value("traversal_guided_early_stop", False)
        ),
        traversal_guided_early_stop_distance_ratio=float(
            value("traversal_guided_early_stop_distance_ratio", 0.0)
        ),
    )


def _guidance_settings_from_args(
    args: argparse.Namespace,
    plan_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = (
        plan_row.get("guidance_policy", {})
        if isinstance(plan_row, Mapping)
        else {}
    )

    def value(name: str, default: Any) -> Any:
        return policy.get(name, getattr(args, name, default))

    return {
        "d1_exact_max_selectivity_pct": float(
            value("d1_exact_max_selectivity_pct", 6.0)
        ),
        "collapse_exact_and_guidance": bool(
            value("collapse_exact_and_guidance", True)
        ),
        "guidance_selectivity_min_pct": float(
            value("guidance_selectivity_min_pct", 0.0)
        ),
        "guidance_selectivity_max_pct": float(
            value("guidance_selectivity_max_pct", 6.0)
        ),
        "guidance_composite_max_selectivity_pct": float(
            value("guidance_composite_max_selectivity_pct", 100.0)
        ),
        "guidance_max_atoms": int(value("guidance_max_atoms", 160)),
        "d2_source_on_guidance_bypass": bool(
            value("d2_source_on_guidance_bypass", True)
        ),
        "guidance_bypass_ef_search": int(
            value("guidance_bypass_ef_search", 0)
        ),
        "guidance_low_selectivity_bypass_ef_search": int(
            value("guidance_low_selectivity_bypass_ef_search", 0)
        ),
    }


def resolve_search_settings(
    args: argparse.Namespace,
    grid: Mapping[str, Any],
    *,
    minimum_traversal_target: int,
) -> SearchSettings:
    """Resolve exactly one immutable independently tuned measurement pair."""
    if args.measurement_plan is not None:
        if not args.pair_id:
            raise Figure5ThroughputError("--measurement-plan requires --pair-id")
        conflicting = [
            name
            for name in (*_LEGACY_COMMON_ARGUMENTS, *(f"{arm}_{field}" for arm in ("stock", "sqlens") for field in ARM_SEARCH_FIELDS))
            if getattr(args, name, None) is not None
        ]
        if conflicting:
            raise Figure5ThroughputError(
                "measurement-plan cannot be mixed with explicit search settings: "
                + ", ".join("--" + name.replace("_", "-") for name in conflicting)
            )
        matches = [
            row
            for row in _measurement_plan_rows(args.measurement_plan.resolve())
            if str(row.get("pair_id") or "") == args.pair_id
        ]
        if len(matches) != 1:
            raise Figure5ThroughputError(
                f"measurement plan must contain exactly one pair_id={args.pair_id!r}; found {len(matches)}"
            )
        row = matches[0]
        target_recall = _as_plan_float(row.get("target_recall"), "target_recall")
        settings = SearchSettings(
            config_id=str(row.get("config_id") or args.config_id or args.pair_id),
            pair_id=args.pair_id,
            target_recall=target_recall,
            stock=_arm_settings_from_plan(row, "stock"),
            sqlens=_arm_settings_from_plan(row, "sqlens"),
            filter_ef_search=args.filter_ef_search_json,
            filter_traversal_target=args.filter_traversal_target_json,
            filter_mode_configs=getattr(args, "filter_mode_configs_json", {}),
            **_guidance_settings_from_args(args, row),
        )
    else:
        if args.ef_search is not None and not args.allow_equal_arm_settings:
            raise Figure5ThroughputError(
                "legacy --ef-search is forbidden for formal execution; use per-arm "
                "--stock-ef-search/--sqlens-ef-search, a --measurement-plan, or "
                "explicit --allow-equal-arm-settings"
            )
        if not args.pair_id:
            raise Figure5ThroughputError("explicit CLI fallback requires --pair-id")
        if args.target_recall is None:
            raise Figure5ThroughputError("explicit CLI fallback requires --target-recall")
        settings = SearchSettings(
            config_id=str(args.config_id or args.pair_id),
            pair_id=str(args.pair_id),
            target_recall=float(args.target_recall),
            stock=_cli_arm_settings(args, "stock", grid, minimum_traversal_target),
            sqlens=_cli_arm_settings(args, "sqlens", grid, minimum_traversal_target),
            filter_ef_search=args.filter_ef_search_json,
            filter_traversal_target=args.filter_traversal_target_json,
            filter_mode_configs=getattr(args, "filter_mode_configs_json", {}),
            **_guidance_settings_from_args(args),
        )
    validate_search_settings(settings, minimum_traversal_target=minimum_traversal_target)
    return settings


def measurement_plan_binding(args: argparse.Namespace, settings: SearchSettings) -> dict[str, Any]:
    """Record whether the immutable pair came from a plan or explicit flags."""
    if args.measurement_plan is None:
        return {
            "source": "explicit_per_arm_cli",
            "pair_id": settings.pair_id,
            "target_recall": settings.target_recall,
        }
    path = args.measurement_plan.resolve()
    return {
        "source": "measurement_plan",
        "path": str(path),
        "sha256": sha256_file(path),
        "pair_id": settings.pair_id,
        "target_recall": settings.target_recall,
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-config", type=Path, default=DEFAULT_FRONTIER_CONFIG)
    parser.add_argument(
        "--orchestrator-source",
        type=Path,
        help="Parent service-curve runner source bound at process startup.",
    )
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_IDS))
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument(
        "--workload-request-limit",
        type=nonnegative_int,
        default=0,
        help=(
            "Execute only the first N requests after validating the complete "
            "frozen source workload; zero preserves the formal q10K default."
        ),
    )
    parser.add_argument("--release-contract", type=Path)
    parser.add_argument("--config-id")
    parser.add_argument("--measurement-plan", type=Path)
    parser.add_argument("--pair-id")
    parser.add_argument("--target-recall", type=positive_float)
    parser.add_argument(
        "--allow-equal-arm-settings",
        action="store_true",
        help="Explicitly permit deprecated shared search options for an equal-config control.",
    )
    # Deprecated common options are retained only for an explicitly opted-in
    # equal-config control. Formal matched-recall points use the per-arm flags
    # or an immutable measurement-plan row below.
    parser.add_argument("--ef-search", type=positive_int)
    parser.add_argument("--stock-ef-search", type=positive_int)
    parser.add_argument("--sqlens-ef-search", type=positive_int)
    parser.add_argument(
        "--stock-iterative-scan",
        choices=sorted(core.ITERATIVE_SCAN_VALUES),
    )
    parser.add_argument(
        "--sqlens-iterative-scan",
        choices=("off",),
    )
    parser.add_argument("--max-scan-tuples", type=positive_int)
    parser.add_argument("--stock-max-scan-tuples", type=positive_int)
    parser.add_argument("--sqlens-max-scan-tuples", type=positive_int)
    parser.add_argument("--scan-mem-multiplier", type=positive_float)
    parser.add_argument("--stock-scan-mem-multiplier", type=positive_float)
    parser.add_argument("--sqlens-scan-mem-multiplier", type=positive_float)
    parser.add_argument("--guided-collect-target", type=positive_int)
    parser.add_argument("--stock-guided-collect-target", type=positive_int)
    parser.add_argument("--sqlens-guided-collect-target", type=positive_int)
    parser.add_argument("--traversal-guided-target", type=positive_int)
    parser.add_argument("--stock-traversal-guided-target", type=positive_int)
    parser.add_argument("--sqlens-traversal-guided-target", type=positive_int)
    parser.add_argument(
        "--filter-ef-search-json",
        type=core.parse_filter_ef_search_json,
        default={},
        help=(
            "JSON object or file mapping mode and predicate names to the "
            "independently calibrated ef_search used for each request."
        ),
    )
    parser.add_argument(
        "--filter-traversal-target-json",
        type=core.parse_filter_traversal_target_json,
        default={},
        help=(
            "JSON object or file mapping mode and predicate names to the "
            "independently calibrated traversal result target."
        ),
    )
    parser.add_argument(
        "--filter-mode-configs-json",
        type=core.parse_filter_mode_configs_json,
        default={},
        help=(
            "JSON object or file mapping mode, predicate, and complete "
            "per-request search configuration overrides."
        ),
    )
    parser.add_argument("--traversal-guided-burst", type=positive_int)
    parser.add_argument("--stock-traversal-guided-burst", type=positive_int)
    parser.add_argument("--sqlens-traversal-guided-burst", type=positive_int)
    parser.add_argument(
        "--stock-traversal-guided-early-stop",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--sqlens-traversal-guided-early-stop",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--stock-traversal-guided-early-stop-distance-ratio",
        type=unit_interval_float,
    )
    parser.add_argument(
        "--sqlens-traversal-guided-early-stop-distance-ratio",
        type=unit_interval_float,
    )
    parser.add_argument("--clients", type=positive_int, required=True)
    parser.add_argument("--repeats", type=positive_int)
    parser.add_argument(
        "--allow-single-pass",
        action="store_true",
        help=(
            "Permit one paired c16 pass for time-critical reporting. The "
            "artifact is explicitly marked and does not satisfy the default "
            "six-repeat formal gate."
        ),
    )
    parser.add_argument("--schedule-seed", type=int)
    parser.add_argument("--client-cpu-list", default="0-31")
    parser.add_argument("--backend-cpu-list")
    parser.add_argument("--backend-proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--telemetry-devices")
    parser.add_argument(
        "--telemetry-path",
        action="append",
        default=[],
        type=Path,
    )
    parser.add_argument(
        "--pg-prewarm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--d1-cache-mb", type=positive_int, default=1024)
    parser.add_argument("--d3-cache-mb", type=positive_int, default=1024)
    parser.add_argument("--d1-exact-max-selectivity-pct", type=positive_float, default=6.0)
    parser.add_argument(
        "--collapse-exact-and-guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--guidance-selectivity-min-pct",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--guidance-selectivity-max-pct",
        type=positive_float,
        default=6.0,
    )
    parser.add_argument(
        "--guidance-composite-max-selectivity-pct",
        type=positive_float,
        default=100.0,
    )
    parser.add_argument("--guidance-max-atoms", type=positive_int, default=160)
    parser.add_argument("--d3-probe-requests", type=positive_int, default=2)
    parser.add_argument("--d3-min-benefit-per-byte", type=float, default=0.0)
    parser.add_argument("--d3-max-fragment-mb", type=positive_int, default=16)
    parser.add_argument("--d3-page-min-skip-rate", type=float, default=0.05)
    parser.add_argument(
        "--d2-source-on-guidance-bypass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Route guidance-policy bypasses to the source HNSW index.",
    )
    parser.add_argument(
        "--guidance-bypass-iterative-scan",
        choices=sorted(core.ITERATIVE_SCAN_VALUES),
        default="strict_order",
        help="Iterative-scan mode used by SQLens policy-bypass requests.",
    )
    parser.add_argument(
        "--guidance-bypass-ef-search",
        type=nonnegative_int,
        default=0,
        help="Fixed ef_search used by SQLens policy-bypass requests; zero reuses SQLens ef_search.",
    )
    parser.add_argument(
        "--guidance-low-selectivity-bypass-ef-search",
        type=nonnegative_int,
        default=0,
        help=(
            "ef_search used by low-selectivity policy bypasses; zero reuses "
            "--guidance-bypass-ef-search."
        ),
    )
    parser.add_argument("--d2-page-access", choices=("off", "prefetch", "reorder"), default="off")
    parser.add_argument("--d2-index-page-access", choices=("off", "prefetch"), default="off")
    parser.add_argument("--d2-page-window", type=positive_int, default=128)
    parser.add_argument("--d2-page-prefetch-min-items", type=positive_int, default=2)
    parser.add_argument("--d2-page-disable-after-no-merge", type=positive_int, default=2)
    parser.add_argument("--statement-timeout-ms", type=positive_int, default=7_200_000)
    parser.add_argument("--start-barrier-timeout-seconds", type=positive_float, default=120.0)
    parser.add_argument("--bootstrap-samples", type=positive_int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--run-id")
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    try:
        args.execution_sources = execution_source_bindings(args)
        binding, protocol, metadata = load_frontier_config(
            args.frontier_config.resolve(),
            args.dataset,
            args.release_contract,
        )
        grid = metadata["grid"]
        repeats = args.repeats or int(protocol.get("throughput_repeats") or 0)
        args.schedule_seed = (
            args.schedule_seed
            if args.schedule_seed is not None
            else int(protocol.get("schedule_seed") or 0)
        )
        settings = resolve_search_settings(
            args,
            grid,
            minimum_traversal_target=K + int(binding.truth_self_excluded),
        )
        client_cpus = validate_execution_args(args, repeats)
        workload = load_frozen_workload(
            binding,
            args.workload_manifest.resolve(),
            request_limit=args.workload_request_limit,
        )
        delegated_proof = load_delegated_d2_proof(
            binding.d2_graph_proof_json,
            binding,
        )
        if not args.execute:
            print(
                json.dumps(
                    prospective_manifest(
                        args,
                        binding,
                        protocol,
                        metadata,
                        workload,
                        settings,
                        delegated_proof,
                        repeats,
                        client_cpus,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        paths, manifest = execute_benchmark(
            args,
            binding,
            protocol,
            metadata,
            workload,
            settings,
            delegated_proof,
            repeats,
            client_cpus,
        )
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return 0 if manifest["artifact_valid"] else 2
    except (Figure5ThroughputError, core.D2GraphProofGateError) as exc:
        print(f"Figure 5 throughput rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
