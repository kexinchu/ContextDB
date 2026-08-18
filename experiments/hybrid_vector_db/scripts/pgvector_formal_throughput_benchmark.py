#!/usr/bin/env python3
"""Formal warm-cache Amazon-10M Stock vs SQLens-D1 throughput benchmark.

Formal measurement has two deliberately separate artifacts.  The q200 exact
truth/manifest calibrates and audits matched-recall configurations.  A q10200
cohort contains q0..q10199; q0..q199 are the selection/confirmation prefix and
only q200..q10199 form the 10,000-query formal workload, with a separately bound
exact-truth manifest.  The two sets are disjoint by actual PostgreSQL query ID.

The old q100..q199 replay remains available only for explicitly labelled
non-formal debugging.  A formal run never falls back to it.

Search settings are never calibrated here.  Every (filter, method, target)
configuration must come from an independently audited, requested-slice-complete
matched-recall artifact whose calibration Recall@10 LCB95 reaches the explicit
0.90, 0.95, or 0.99 target. Formal measurement independently rechecks both
mean Recall@10 and a unique-query-cluster bootstrap LCB95. Each measurement
arm also records host CPU/block-I/O and PostgreSQL buffer/I/O counter deltas.
Resume commits only an adjacent Stock/SQLens-D1 pair for one cell/repeat.
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
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import psycopg

try:
    from .audit_sigmod_matched_recall_artifact import (
        audit_manifest as audit_matched_recall_manifest,
    )
    from .common_pg import pg_config_from_env
    from .pgvector_design1_design2_design3_selectivity_benchmark import (
        activate,
        activation_binding,
        candidate_self_exclusion,
        close_mode_runtime,
        mode_table_index,
        normalize_cpu_list,
        open_mode_runtime,
        query_table_for_candidate,
        recover_runtime,
        run_query,
        tie_aware_recall,
        uses_exact_predicate_scan_contract,
    )
    from .pgvector_target_recall_selectivity_runner import (
        acquire_formal_data_guard,
        database_fingerprint,
        git_revision,
        prepare_fragment_tracking,
        relation_identifier,
        sha256_file,
        utc_now,
    )
except ImportError:
    from audit_sigmod_matched_recall_artifact import (
        audit_manifest as audit_matched_recall_manifest,
    )
    from common_pg import pg_config_from_env
    from pgvector_design1_design2_design3_selectivity_benchmark import (
        activate,
        activation_binding,
        candidate_self_exclusion,
        close_mode_runtime,
        mode_table_index,
        normalize_cpu_list,
        open_mode_runtime,
        query_table_for_candidate,
        recover_runtime,
        run_query,
        tie_aware_recall,
        uses_exact_predicate_scan_contract,
    )
    from pgvector_target_recall_selectivity_runner import (
        acquire_formal_data_guard,
        database_fingerprint,
        git_revision,
        prepare_fragment_tracking,
        relation_identifier,
        sha256_file,
        utc_now,
    )


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/hybrid_vector_db"
DEFAULT_FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_valid_embeddings_filters.csv"
DEFAULT_CALIBRATION_TRUTH = RESULTS / "amazon_selectivity14_exact_truth_q200_unique_embeddings_formal.csv"
DEFAULT_CALIBRATION_TRUTH_MANIFEST = RESULTS / "amazon_selectivity14_exact_truth_q200_unique_embeddings_formal_manifest.json"
DEFAULT_MEASUREMENT_QUERY_FILE = RESULTS / "amazon10m_unique_embedding_query_cohort_q10200.csv"
DEFAULT_MEASUREMENT_QUERY_MANIFEST = RESULTS / "amazon10m_unique_embedding_query_cohort_q10200_manifest.json"
DEFAULT_MEASUREMENT_TRUTH = RESULTS / "amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal.csv"
DEFAULT_MEASUREMENT_TRUTH_MANIFEST = RESULTS / "amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal_manifest.json"
DEFAULT_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
DEFAULT_SOURCE_INDEX = "public.amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx"
DEFAULT_BFS_INDEX = "public.amazon10m_embedding_valid_hnsw_m32ef200_fullmem_bfs_idx"
DEFAULT_CANDIDATE_VALIDITY_PREDICATE = "embedding_valid"
EXPECTED_CANDIDATE_ROWS = 9_979_556

METHODS = ("stock", "sqlens_d1")
MODE_BY_METHOD = {"stock": "original", "sqlens_d1": "design1_bloom"}
FORMAL_D1_GUIDANCE_STRATEGY = "safe_guided"
SUPPORTED_FORMAL_D1_STRATEGIES = ("safe_guided", "traversal_guided")
FORMAL_CLIENTS = (1, 4, 8, 16, 32, 64)
FORMAL_TARGETS = (0.90, 0.95, 0.99)
FORMAL_REQUESTS = 10_000
FORMAL_FILTER_COUNT = 14
# This is a fixed stratified slice for the paper's throughput/tail figure.  It
# spans wide, medium, narrow, and highly selective predicates without letting a
# caller silently choose favorable filters.  It is not a substitute for the
# complete 14-filter matrix, which has its own explicit execution mode.
REPRESENTATIVE_FILTERS = (
    "popular_ge1000",
    "rating5_price_le10",
    "grocery_rating5",
    "grocery_long500",
)
EVALUATION_SCOPES = ("representative_filters", "full_matrix")
SELECTION_QUERY_NOS = tuple(range(0, 200))
MEASUREMENT_QUERY_NOS = tuple(range(200, 10_200))
REPLAY_QUERY_NOS = tuple(range(100, 200))
CHECKPOINT_SCHEMA_VERSION = 10
RUNNER_VERSION = "amazon10m-pgvector-formal-throughput-v12-paired-resume-relation-telemetry"
FORMAL_MEASUREMENT_REPEATS = 6
REQUIRED_SQLENS_BUILD_PREFIXES = (
    "sqlens-v16-d3-full-materialization-persisted-reuse-",
    "sqlens-v16-d3-representation-preserving-exact-d2-edge-trace-",
)

TRUE_QUERY_GLOBS = ("amazon10m_unique_embedding_query_cohort_q10200.csv",)
PROC_STAT_PATH = Path("/proc/stat")
PROC_DISKSTATS_PATH = Path("/proc/diskstats")
PROC_ROOT_PATH = Path("/proc")
SYS_DEV_BLOCK_PATH = Path("/sys/dev/block")
CPU_COUNTER_NAMES = (
    "user",
    "nice",
    "system",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
)
DISK_COUNTER_NAMES = (
    "reads_completed",
    "reads_merged",
    "sectors_read",
    "read_time_ms",
    "writes_completed",
    "writes_merged",
    "sectors_written",
    "write_time_ms",
    "io_in_progress",
    "io_time_ms",
    "weighted_io_time_ms",
)
PG_STAT_DATABASE_COUNTERS = (
    "xact_commit",
    "xact_rollback",
    "blks_read",
    "blks_hit",
    "tup_returned",
    "tup_fetched",
    "tup_inserted",
    "tup_updated",
    "tup_deleted",
    "temp_files",
    "temp_bytes",
    "deadlocks",
    "checksum_failures",
    "blk_read_time",
    "blk_write_time",
    "session_time",
    "active_time",
    "idle_in_transaction_time",
    "sessions",
    "sessions_abandoned",
    "sessions_fatal",
    "sessions_killed",
)
PG_STAT_IO_COUNTERS = (
    "reads",
    "read_time",
    "writes",
    "write_time",
    "writebacks",
    "writeback_time",
    "extends",
    "extend_time",
    "hits",
    "evictions",
    "reuses",
    "fsyncs",
    "fsync_time",
)
PG_STATIO_TABLE_COUNTERS = (
    "heap_blks_read",
    "heap_blks_hit",
    "idx_blks_read",
    "idx_blks_hit",
)
PG_STATIO_INDEX_COUNTERS = ("idx_blks_read", "idx_blks_hit")


class BenchmarkContractError(RuntimeError):
    """The requested run cannot be labelled as a formal benchmark."""


@dataclass(frozen=True)
class FilterSpec:
    name: str
    predicate: str
    atoms: tuple[str, ...]
    expected_rows: int
    actual_pct: float


@dataclass(frozen=True)
class WorkloadRequest:
    request_no: int
    query_no: int
    query_id: int
    trace_cycle: int


@dataclass(frozen=True)
class Workload:
    requests: tuple[WorkloadRequest, ...]
    source_kind: str
    source_path: str
    source_sha256: str
    query_cohort: str
    trace_replay: bool
    unique_query_vectors: int


@dataclass(frozen=True)
class SearchConfig:
    ef_search: int
    max_scan_tuples: int
    scan_mem_multiplier: float
    iterative_scan: str
    guided_collect_target: int
    traversal_guided_prioritization: bool = True
    traversal_guided_burst: int = 8
    traversal_guided_target: int = 11

    def as_mode_config(self) -> dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        mem = str(self.scan_mem_multiplier).replace(".", "p")
        return (
            f"ef{self.ef_search}_target{self.guided_collect_target}_"
            f"traverse{self.traversal_guided_target}_"
            f"max{self.max_scan_tuples}_mem{mem}_{self.iterative_scan}"
        )


@dataclass(frozen=True)
class MatchedRecallBundle:
    configs: Mapping[tuple[str, str, float], SearchConfig]
    evidence: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    manifest: Mapping[str, Any]
    guidance_filter_strategy: str


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def stable_runtime_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude mutable last-scan counters from the durable run identity."""
    fields = (
        "loaded_vector_sqlens_build_id",
        "loaded_vector_so_path",
        "loaded_vector_so_sha256",
        "required_build_prefix",
        "minimum_profile_semantics_version",
        "profile_semantics_version",
    )
    return {
        **{field: provenance[field] for field in fields},
        "required_profile_field_names": sorted(
            (provenance.get("required_profile_fields") or {}).keys()
        ),
    }


def stable_fragment_tracking(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in provenance.items() if key != "prepared_at"}


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


def parse_int_list(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("integer list values must be positive")
    return sorted(set(parsed))


def parse_targets(value: str) -> list[float]:
    try:
        targets = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated recall list") from exc
    if not targets or any(not 0.0 < target <= 1.0 for target in targets):
        raise argparse.ArgumentTypeError("recall targets must be in (0, 1]")
    return targets


def parse_device_names(value: str | None) -> tuple[str, ...]:
    devices = {
        Path(item.strip()).name
        for item in str(value or "").split(",")
        if item.strip()
    }
    if any("/" in device or device in {".", ".."} for device in devices):
        raise argparse.ArgumentTypeError("telemetry devices must be Linux block-device names")
    return tuple(sorted(devices))


def _existing_path(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise BenchmarkContractError(f"cannot resolve telemetry path: {path}")
    return candidate


def block_device_for_path(
    path: Path,
    sys_dev_block_path: Path = SYS_DEV_BLOCK_PATH,
) -> str | None:
    """Resolve a host path's Linux block device without shelling out."""
    existing = _existing_path(path)
    device_number = existing.stat().st_dev
    link = sys_dev_block_path / f"{os.major(device_number)}:{os.minor(device_number)}"
    if not link.exists():
        return None
    try:
        return link.resolve(strict=True).name
    except OSError:
        return None


def read_diskstats(
    devices: Sequence[str],
    path: Path = PROC_DISKSTATS_PATH,
) -> dict[str, dict[str, int]]:
    requested = set(devices)
    observed: dict[str, dict[str, int]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkContractError(f"cannot read {path}: {exc}") from exc
    for line in lines:
        fields = line.split()
        if len(fields) < 14 or fields[2] not in requested:
            continue
        values = [int(value) for value in fields[3:14]]
        observed[fields[2]] = dict(zip(DISK_COUNTER_NAMES, values))
    missing = sorted(requested - set(observed))
    if missing:
        raise BenchmarkContractError(
            f"telemetry devices are absent from {path}: {missing}"
        )
    return observed


def resolve_telemetry_devices(
    explicit: str | None,
    paths: Sequence[Path],
    *,
    diskstats_path: Path = PROC_DISKSTATS_PATH,
    sys_dev_block_path: Path = SYS_DEV_BLOCK_PATH,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    explicit_devices = parse_device_names(explicit)
    path_devices: dict[str, str | None] = {}
    for path in paths:
        path_devices[str(path)] = block_device_for_path(path, sys_dev_block_path)
    devices = tuple(
        sorted(set(explicit_devices) | {value for value in path_devices.values() if value})
    )
    if not devices:
        raise BenchmarkContractError(
            "formal host I/O telemetry could not bind a block device; pass "
            "--telemetry-devices or --telemetry-path"
        )
    read_diskstats(devices, diskstats_path)
    return devices, {
        "resolved_devices": list(devices),
        "explicit_devices": list(explicit_devices),
        "path_device_resolution": path_devices,
        "sector_bytes": 512,
        "source": str(diskstats_path),
    }


def read_proc_stat(path: Path = PROC_STAT_PATH) -> dict[str, int]:
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0].split()
    except (OSError, IndexError) as exc:
        raise BenchmarkContractError(f"cannot read {path}: {exc}") from exc
    if not first or first[0] != "cpu" or len(first) < len(CPU_COUNTER_NAMES) + 1:
        raise BenchmarkContractError(f"{path} has no aggregate CPU counters")
    return {
        name: int(value)
        for name, value in zip(CPU_COUNTER_NAMES, first[1 : len(CPU_COUNTER_NAMES) + 1])
    }


def _monotonic_counter_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    names: Sequence[str],
    label: str,
) -> dict[str, float]:
    delta: dict[str, float] = {}
    for name in names:
        start = float(before.get(name) or 0.0)
        finish = float(after.get(name) or 0.0)
        if not math.isfinite(start) or not math.isfinite(finish) or finish < start:
            raise BenchmarkContractError(
                f"{label} counter is non-monotonic or non-finite: {name}"
            )
        delta[name] = finish - start
    return delta


def host_telemetry_snapshot(
    devices: Sequence[str],
    *,
    proc_stat_path: Path = PROC_STAT_PATH,
    diskstats_path: Path = PROC_DISKSTATS_PATH,
) -> dict[str, Any]:
    return {
        "monotonic_ns": time.monotonic_ns(),
        "cpu": read_proc_stat(proc_stat_path),
        "disk": read_diskstats(devices, diskstats_path),
    }


def host_telemetry_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    cpu = _monotonic_counter_delta(
        _require_mapping(before.get("cpu"), "host CPU before snapshot"),
        _require_mapping(after.get("cpu"), "host CPU after snapshot"),
        CPU_COUNTER_NAMES,
        "host CPU",
    )
    cpu_total = sum(cpu.values())
    if cpu_total <= 0:
        raise BenchmarkContractError("host CPU telemetry window has no elapsed ticks")
    idle = cpu["idle"] + cpu["iowait"]
    cpu_percent = {
        f"{name}_pct": 100.0 * value / cpu_total for name, value in cpu.items()
    }
    cpu_percent["utilization_pct"] = 100.0 * (cpu_total - idle) / cpu_total

    before_disk = _require_mapping(before.get("disk"), "disk before snapshot")
    after_disk = _require_mapping(after.get("disk"), "disk after snapshot")
    if set(before_disk) != set(after_disk):
        raise BenchmarkContractError("host disk device set changed during an arm")
    disk_delta: dict[str, dict[str, float]] = {}
    for device in sorted(before_disk):
        counters = _monotonic_counter_delta(
            _require_mapping(before_disk[device], f"{device} before counters"),
            _require_mapping(after_disk[device], f"{device} after counters"),
            tuple(name for name in DISK_COUNTER_NAMES if name != "io_in_progress"),
            f"disk {device}",
        )
        counters["read_bytes"] = counters["sectors_read"] * 512.0
        counters["write_bytes"] = counters["sectors_written"] * 512.0
        counters["io_in_progress_end"] = float(
            _require_mapping(after_disk[device], f"{device} after counters").get(
                "io_in_progress"
            )
            or 0
        )
        disk_delta[device] = counters
    disk_total = {
        name: sum(values[name] for values in disk_delta.values())
        for name in (
            "reads_completed",
            "read_bytes",
            "read_time_ms",
            "writes_completed",
            "write_bytes",
            "write_time_ms",
            "io_time_ms",
            "weighted_io_time_ms",
        )
    }
    return {
        "window_seconds": (
            int(after["monotonic_ns"]) - int(before["monotonic_ns"])
        )
        / 1_000_000_000.0,
        "cpu_tick_delta": cpu,
        "cpu": cpu_percent,
        "disk_devices": disk_delta,
        "disk_total": disk_total,
    }


def read_backend_proc_stat(
    backend_pid: int,
    *,
    proc_root: Path = PROC_ROOT_PATH,
) -> dict[str, int]:
    """Read the stable CPU counters for one PostgreSQL backend process."""
    if backend_pid <= 0:
        raise BenchmarkContractError(f"invalid PostgreSQL backend PID: {backend_pid}")
    path = proc_root / str(backend_pid) / "stat"
    try:
        line = path.read_text(encoding="utf-8").strip()
        prefix, suffix = line.rsplit(")", 1)
        fields = suffix.split()
    except (OSError, ValueError) as exc:
        raise BenchmarkContractError(
            f"cannot read tracked PostgreSQL backend stat {path}: {exc}"
        ) from exc
    # After the executable name, Linux field 3 is at index 0.  utime/stime are
    # fields 14/15 and starttime is field 22, which detects PID reuse.
    if len(fields) <= 19:
        raise BenchmarkContractError(f"malformed PostgreSQL backend stat: {path}")
    comm = prefix.rsplit("(", 1)[-1]
    if "postgres" not in comm.lower():
        raise BenchmarkContractError(
            f"tracked backend PID does not name a PostgreSQL process: {path} ({comm!r})"
        )
    try:
        return {
            "pid": backend_pid,
            "comm": comm,
            "utime_ticks": int(fields[11]),
            "stime_ticks": int(fields[12]),
            "starttime_ticks": int(fields[19]),
        }
    except ValueError as exc:
        raise BenchmarkContractError(f"non-integer PostgreSQL backend stat: {path}") from exc


def backend_cpu_snapshot(
    backend_pids: Sequence[int],
    *,
    proc_root: Path = PROC_ROOT_PATH,
    clock_ticks_per_second: int | None = None,
) -> dict[str, Any]:
    pids = [int(pid) for pid in backend_pids]
    if not pids or len(set(pids)) != len(pids):
        raise BenchmarkContractError("backend CPU telemetry requires unique tracked backend PIDs")
    ticks = int(clock_ticks_per_second or os.sysconf("SC_CLK_TCK"))
    if ticks <= 0:
        raise BenchmarkContractError("SC_CLK_TCK must be positive for backend CPU telemetry")
    return {
        "monotonic_ns": time.monotonic_ns(),
        "clock_ticks_per_second": ticks,
        "backend": {str(pid): read_backend_proc_stat(pid, proc_root=proc_root) for pid in pids},
    }


def backend_cpu_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    before_ticks = int(before.get("clock_ticks_per_second") or 0)
    after_ticks = int(after.get("clock_ticks_per_second") or 0)
    if before_ticks <= 0 or before_ticks != after_ticks:
        raise BenchmarkContractError("backend CPU clock-tick identity changed during an arm")
    before_rows = _require_mapping(before.get("backend"), "backend CPU before snapshot")
    after_rows = _require_mapping(after.get("backend"), "backend CPU after snapshot")
    if set(before_rows) != set(after_rows) or not before_rows:
        raise BenchmarkContractError("tracked PostgreSQL backend PID set changed during an arm")
    per_backend: dict[str, dict[str, float]] = {}
    for key in sorted(before_rows, key=int):
        start = _require_mapping(before_rows[key], f"backend {key} before snapshot")
        finish = _require_mapping(after_rows[key], f"backend {key} after snapshot")
        if int(start.get("pid") or -1) != int(finish.get("pid") or -1):
            raise BenchmarkContractError(f"PostgreSQL backend PID changed during an arm: {key}")
        if int(start.get("starttime_ticks") or -1) != int(finish.get("starttime_ticks") or -1):
            raise BenchmarkContractError(f"PostgreSQL backend PID was reused during an arm: {key}")
        if start.get("comm") != finish.get("comm"):
            raise BenchmarkContractError(f"PostgreSQL backend identity changed during an arm: {key}")
        counters = _monotonic_counter_delta(
            start, finish, ("utime_ticks", "stime_ticks"), f"backend CPU {key}"
        )
        user_ms = counters["utime_ticks"] * 1000.0 / before_ticks
        system_ms = counters["stime_ticks"] * 1000.0 / before_ticks
        per_backend[key] = {
            "pid": float(int(finish["pid"])),
            "comm": str(finish["comm"]),
            "starttime_ticks": float(int(finish["starttime_ticks"])),
            **counters,
            "user_cpu_ms": user_ms,
            "system_cpu_ms": system_ms,
            "total_cpu_ms": user_ms + system_ms,
        }
    return {
        "window_seconds": (
            int(after["monotonic_ns"]) - int(before["monotonic_ns"])
        ) / 1_000_000_000.0,
        "clock_ticks_per_second": before_ticks,
        "backend_pids": [int(key) for key in sorted(per_backend, key=int)],
        "per_backend": per_backend,
        "total": {
            "user_cpu_ms": sum(row["user_cpu_ms"] for row in per_backend.values()),
            "system_cpu_ms": sum(row["system_cpu_ms"] for row in per_backend.values()),
            "total_cpu_ms": sum(row["total_cpu_ms"] for row in per_backend.values()),
        },
        "scope": "tracked_postgresql_client_backend_processes",
        "tracking_complete": True,
    }


def _json_value(value: Any, label: str) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"PostgreSQL returned invalid {label} JSON") from exc


def _postgres_relation_stat_snapshot(
    cursor: Any,
    target_table: str,
    target_index: str,
) -> dict[str, Any]:
    cursor.execute(
        "SELECT row_to_json(s)::text FROM "
        "(SELECT relid::bigint AS relid, schemaname, relname, "
        f"{', '.join(PG_STATIO_TABLE_COUNTERS)} "
        "FROM pg_statio_user_tables WHERE relid = to_regclass(%s)) AS s",
        (target_table,),
    )
    row = cursor.fetchone()
    table = _json_value(row[0], "pg_statio_user_tables") if row and row[0] else None
    cursor.execute(
        "SELECT row_to_json(s)::text FROM "
        "(SELECT relid::bigint AS relid, indexrelid::bigint AS indexrelid, "
        "schemaname, relname, indexrelname, "
        f"{', '.join(PG_STATIO_INDEX_COUNTERS)} "
        "FROM pg_statio_user_indexes WHERE indexrelid = to_regclass(%s)) AS s",
        (target_index,),
    )
    row = cursor.fetchone()
    index = _json_value(row[0], "pg_statio_user_indexes") if row and row[0] else None
    if not isinstance(table, Mapping) or not isinstance(index, Mapping):
        raise BenchmarkContractError(
            "target table/index are absent from pg_statio_user_tables/indexes: "
            f"{target_table}/{target_index}"
        )
    return {
        "target_table": target_table,
        "target_index": target_index,
        "table": dict(table),
        "index": dict(index),
    }


def postgres_telemetry_snapshot(
    cursor: Any,
    target_table: str,
    target_index: str,
) -> dict[str, Any]:
    database_columns = ", ".join(PG_STAT_DATABASE_COUNTERS)
    cursor.execute(
        "SELECT row_to_json(s)::text FROM "
        f"(SELECT datid, datname, stats_reset, {database_columns} "
        "FROM pg_stat_database WHERE datname = current_database()) AS s"
    )
    row = cursor.fetchone()
    if not row or row[0] is None:
        raise BenchmarkContractError("pg_stat_database has no current-database row")
    database = _json_value(row[0], "pg_stat_database")

    io_columns = ", ".join(PG_STAT_IO_COUNTERS)
    cursor.execute(
        "SELECT COALESCE(json_agg(row_to_json(s) ORDER BY "
        "s.backend_type, s.object, s.context), '[]'::json)::text FROM "
        f"(SELECT backend_type, object, context, op_bytes, {io_columns} "
        "FROM pg_stat_io) AS s"
    )
    row = cursor.fetchone()
    io_rows = _json_value(row[0] if row else "[]", "pg_stat_io")
    if not isinstance(database, Mapping) or not isinstance(io_rows, list):
        raise BenchmarkContractError("PostgreSQL telemetry snapshots have invalid shapes")
    return {
        "monotonic_ns": time.monotonic_ns(),
        "database": dict(database),
        "io": io_rows,
        "relations": _postgres_relation_stat_snapshot(
            cursor, target_table, target_index
        ),
    }


def _pg_io_key(row: Mapping[str, Any]) -> str:
    return "|".join(str(row.get(name) or "") for name in ("backend_type", "object", "context"))


def _relation_counter_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    identity_fields: Sequence[str],
    counters: Sequence[str],
    label: str,
) -> dict[str, Any]:
    for field in identity_fields:
        if before.get(field) != after.get(field):
            raise BenchmarkContractError(f"{label} identity changed during an arm: {field}")
    return {
        **{field: after.get(field) for field in identity_fields},
        **_monotonic_counter_delta(before, after, counters, label),
    }


def postgres_telemetry_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    before_database = _require_mapping(before.get("database"), "pg_stat_database before")
    after_database = _require_mapping(after.get("database"), "pg_stat_database after")
    if (
        before_database.get("datid") != after_database.get("datid")
        or before_database.get("stats_reset") != after_database.get("stats_reset")
    ):
        raise BenchmarkContractError("pg_stat_database identity/reset changed during an arm")
    database_delta = _monotonic_counter_delta(
        before_database,
        after_database,
        PG_STAT_DATABASE_COUNTERS,
        "pg_stat_database",
    )

    before_rows = {
        _pg_io_key(_require_mapping(row, "pg_stat_io before row")): row
        for row in before.get("io", [])
    }
    after_rows = {
        _pg_io_key(_require_mapping(row, "pg_stat_io after row")): row
        for row in after.get("io", [])
    }
    if set(before_rows) != set(after_rows):
        raise BenchmarkContractError("pg_stat_io dimensions changed during an arm")
    io_delta: dict[str, dict[str, float]] = {}
    for key in sorted(before_rows):
        if before_rows[key].get("op_bytes") != after_rows[key].get("op_bytes"):
            raise BenchmarkContractError(f"pg_stat_io op_bytes changed during an arm: {key}")
        counters = _monotonic_counter_delta(
            before_rows[key],
            after_rows[key],
            PG_STAT_IO_COUNTERS,
            f"pg_stat_io {key}",
        )
        op_bytes = float(after_rows[key].get("op_bytes") or 0)
        counters["op_bytes"] = op_bytes
        counters["read_bytes"] = counters["reads"] * op_bytes
        counters["write_bytes"] = counters["writes"] * op_bytes
        io_delta[key] = counters
    io_total = {
        name: sum(row[name] for row in io_delta.values())
        for name in (*PG_STAT_IO_COUNTERS, "read_bytes", "write_bytes")
    }
    before_relations = _require_mapping(before.get("relations"), "relation stats before")
    after_relations = _require_mapping(after.get("relations"), "relation stats after")
    if (
        before_relations.get("target_table") != after_relations.get("target_table")
        or before_relations.get("target_index") != after_relations.get("target_index")
    ):
        raise BenchmarkContractError("target relation binding changed during an arm")
    relation_delta = {
        "target_table": after_relations.get("target_table"),
        "target_index": after_relations.get("target_index"),
        "table": _relation_counter_delta(
            _require_mapping(before_relations.get("table"), "table stats before"),
            _require_mapping(after_relations.get("table"), "table stats after"),
            identity_fields=("relid", "schemaname", "relname"),
            counters=PG_STATIO_TABLE_COUNTERS,
            label="pg_statio_user_tables target table",
        ),
        "index": _relation_counter_delta(
            _require_mapping(before_relations.get("index"), "index stats before"),
            _require_mapping(after_relations.get("index"), "index stats after"),
            identity_fields=("relid", "indexrelid", "schemaname", "relname", "indexrelname"),
            counters=PG_STATIO_INDEX_COUNTERS,
            label="pg_statio_user_indexes target index",
        ),
        "scope": "target_table_and_hnsw_index_relation_statistics",
        "tracking_complete": True,
    }
    return {
        "window_seconds": (
            int(after["monotonic_ns"]) - int(before["monotonic_ns"])
        )
        / 1_000_000_000.0,
        "database": database_delta,
        "io_by_backend_object_context": io_delta,
        "io_total": io_total,
        "relations": relation_delta,
        "scope": {
            "pg_stat_database": "current_database_cluster_wide",
            "pg_stat_io": "postgresql_cluster_wide",
            "pg_statio_user_tables_indexes": "target_table_and_hnsw_index",
        },
    }


def flush_postgres_stats(runtimes: Sequence[Any]) -> None:
    for runtime in runtimes:
        runtime.cur.execute("SELECT pg_stat_force_next_flush()")
        runtime.cur.fetchone()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkContractError(f"{label} root must be a JSON object")
    return payload


def _is_explicit_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkContractError(f"{label} is missing or is not an object")
    return value


def _matched_artifact_path(manifest_path: Path, artifact: Any, label: str) -> Path:
    metadata = _require_mapping(artifact, label)
    raw_path = Path(str(metadata.get("path") or ""))
    path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
    path = path.resolve()
    if not path.is_file():
        raise BenchmarkContractError(f"{label} file does not exist: {path}")
    expected_sha = str(metadata.get("sha256") or "")
    if len(expected_sha) != 64 or sha256_file(path) != expected_sha:
        raise BenchmarkContractError(f"{label} SHA256 is missing or does not match")
    if int(metadata.get("bytes") or -1) != path.stat().st_size:
        raise BenchmarkContractError(f"{label} byte count does not match")
    return path


def _selected_config_from_row(
    row: Mapping[str, Any],
    *,
    method: str,
    target: float,
    traversal_guided_burst: int,
    guidance_filter_strategy: str,
) -> SearchConfig:
    context = f"{row.get('filter_name')}/{method}/target={target:.2f}"
    required_true = (
        "target_met_in_calibration",
        "target_confirmed_in_calibration",
        "target_lcb95_met_in_calibration",
        "rows_complete",
        "truth_self_excluded",
        "plan_gate_passed",
    )
    missing = [name for name in required_true if not _is_explicit_true(row.get(name))]
    if missing:
        raise BenchmarkContractError(
            f"matched-recall config {context} lacks required gates: {missing}"
        )
    if str(row.get("selection_status") or "") != "selected":
        raise BenchmarkContractError(f"matched-recall config {context} was not selected")
    if int(row.get("errors") or 0) != 0:
        raise BenchmarkContractError(f"matched-recall config {context} reports errors")
    if str(row.get("guidance_filter_strategy") or "") != guidance_filter_strategy:
        raise BenchmarkContractError(
            f"matched-recall config {context} guidance_filter_strategy does not bind "
            f"the manifest strategy {guidance_filter_strategy!r}"
        )
    try:
        recall_mean = float(row["recall_mean"])
        recall_lcb95 = float(row["recall_lcb95"])
        ef_search = int(row["ef_search"])
        max_scan_tuples = int(row["max_scan_tuples"])
        scan_mem_multiplier = float(row["scan_mem_multiplier"])
        guided_collect_target = int(row["guided_collect_target"])
        traversal_guided_target = int(row["traversal_guided_target"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkContractError(
            f"matched-recall config {context} has incomplete numeric configuration"
        ) from exc
    if not all(math.isfinite(value) for value in (recall_mean, recall_lcb95, scan_mem_multiplier)):
        raise BenchmarkContractError(f"matched-recall config {context} has non-finite values")
    if recall_mean < target or recall_lcb95 < target:
        raise BenchmarkContractError(
            f"matched-recall config {context} is mean-only or below target: "
            f"mean={recall_mean:.6f} lcb95={recall_lcb95:.6f}"
        )
    if min(ef_search, max_scan_tuples, guided_collect_target, traversal_guided_target) <= 0:
        raise BenchmarkContractError(f"matched-recall config {context} has non-positive search settings")
    if scan_mem_multiplier <= 0:
        raise BenchmarkContractError(f"matched-recall config {context} has invalid scan memory")
    iterative_scan = str(row.get("iterative_scan") or "")
    if iterative_scan not in {"off", "strict_order", "relaxed_order"}:
        raise BenchmarkContractError(
            f"matched-recall config {context} has invalid iterative_scan={iterative_scan!r}"
        )
    return SearchConfig(
        ef_search=ef_search,
        max_scan_tuples=max_scan_tuples,
        scan_mem_multiplier=scan_mem_multiplier,
        iterative_scan=iterative_scan,
        guided_collect_target=guided_collect_target,
        traversal_guided_prioritization=(
            method == "sqlens_d1" and guidance_filter_strategy == "traversal_guided"
        ),
        traversal_guided_burst=traversal_guided_burst,
        traversal_guided_target=traversal_guided_target,
    )


def load_audited_matched_recall_configs(
    manifest_path: Path,
    *,
    truth_csv: Path,
    filters_csv: Path,
    filters: Sequence[FilterSpec],
    targets: Sequence[float],
    require_runtime_provenance: bool = True,
) -> MatchedRecallBundle:
    """Load only LCB-qualified configs from a separately audited formal artifact.

    ``require_runtime_provenance`` controls the persisted database-relation
    fingerprint gate.  Dry-run uses the static artifact gates only; execute
    still requires the live database fingerprint and exact relation identity.
    The recorded SQLens build/vector.so provenance remains mandatory in both
    modes because it is part of the matched-recall artifact identity.
    """
    manifest_path = manifest_path.resolve()
    manifest = _read_json_object(manifest_path, "matched-recall manifest")
    audit = audit_matched_recall_manifest(
        manifest_path,
        truth_csv=truth_csv,
        filters_csv=filters_csv,
        recall_tolerance=0.0,
        require_complete=True,
    )
    if audit.get("valid") is not True or audit.get("errors"):
        raise BenchmarkContractError(
            "matched-recall manifest failed independent audit: "
            + "; ".join(str(value) for value in audit.get("errors", []))
        )
    for field in ("matrix_complete", "measurement_complete", "comparison_valid"):
        if manifest.get(field) is not True:
            raise BenchmarkContractError(
                f"matched-recall requested slice is incomplete: {field}=false"
            )
    if manifest.get("status") != "complete":
        raise BenchmarkContractError("matched-recall requested slice status is not complete")

    run_spec = _require_mapping(manifest.get("run_spec"), "matched-recall run_spec")
    run_args = _require_mapping(run_spec.get("args"), "matched-recall run_spec.args")
    guidance_filter_strategy = str(run_args.get("guidance_filter_strategy") or "")
    if guidance_filter_strategy not in SUPPORTED_FORMAL_D1_STRATEGIES:
        raise BenchmarkContractError(
            "matched-recall manifest must bind one supported formal D1 guidance strategy; "
            f"observed {guidance_filter_strategy!r}"
        )
    policy = _require_mapping(
        manifest.get("calibration_policy"), "matched-recall calibration_policy"
    )
    effective_policy = str(
        policy.get("calibration_selection_policy")
        or run_args.get("calibration_selection_policy")
        or ""
    )
    selection_text = str(policy.get("selection") or "").lower().replace("_", " ")
    if (
        effective_policy != "lcb_then_max_recall"
        or str(policy.get("stop_metric") or "") != "recall_lcb95"
        or "lcb" not in selection_text
        or "report-only" in selection_text
        or "report only" in selection_text
    ):
        raise BenchmarkContractError(
            "formal throughput rejects mean-only or non-LCB matched-recall manifests"
        )

    manifest_targets = {float(value) for value in manifest.get("targets", [])}
    if not set(float(value) for value in targets) <= manifest_targets:
        raise BenchmarkContractError(
            "matched-recall manifest does not contain every requested 0.90/0.95/0.99 target"
        )
    manifest_modes = {str(value) for value in manifest.get("modes", [])}
    expected_modes = set(MODE_BY_METHOD.values())
    missing_modes = expected_modes - manifest_modes
    extra_modes = manifest_modes - expected_modes
    if missing_modes or extra_modes:
        raise BenchmarkContractError(
            "matched-recall manifest methods must be exactly Stock/SQLens-D1: "
            f"missing={sorted(missing_modes)} extra={sorted(extra_modes)}"
        )

    if sha256_file(truth_csv) != str(run_spec.get("truth_sha256") or ""):
        raise BenchmarkContractError("matched-recall GT provenance does not match --truth-csv")
    if sha256_file(filters_csv) != str(run_spec.get("filters_sha256") or ""):
        raise BenchmarkContractError("matched-recall filter provenance does not match --filters-csv")
    for argument, path in (("truth_csv", truth_csv), ("filters_csv", filters_csv)):
        recorded = Path(str(run_args.get(argument) or "")).resolve()
        if recorded != path.resolve():
            raise BenchmarkContractError(
                f"matched-recall {argument} path does not match the throughput input"
            )

    runtime = _require_mapping(
        run_spec.get("sqlens_runtime_provenance"),
        "matched-recall SQLens runtime provenance",
    )
    build_id = str(runtime.get("loaded_vector_sqlens_build_id") or "")
    vector_sha = str(runtime.get("loaded_vector_so_sha256") or "")
    if not build_id.startswith(REQUIRED_SQLENS_BUILD_PREFIXES):
        raise BenchmarkContractError(
            "matched-recall runtime build is not a supported SQLens release"
        )
    if len(vector_sha) != 64 or any(char not in "0123456789abcdef" for char in vector_sha):
        raise BenchmarkContractError("matched-recall runtime vector.so SHA256 is invalid")

    source_index = str(run_args.get("insertion_index") or "")
    source_table = str(run_args.get("insertion_table") or "")
    source_query_table = str(run_args.get("query_table") or source_table)
    if require_runtime_provenance:
        database = _require_mapping(
            run_spec.get("database"), "matched-recall database provenance"
        )
        relations = _require_mapping(
            database.get("relations"), "matched-recall relation provenance"
        )
        for relation_name in (source_table, source_index):
            relation = _require_mapping(
                relations.get(relation_name), f"matched-recall relation {relation_name}"
            )
            if min(int(relation.get("oid") or 0), int(relation.get("relfilenode") or 0)) <= 0:
                raise BenchmarkContractError(
                    f"matched-recall relation provenance is incomplete for {relation_name}"
                )
        index_relation = _require_mapping(
            relations.get(source_index), "matched-recall source index"
        )
        if (
            index_relation.get("valid") is not True
            or index_relation.get("ready") is not True
            or index_relation.get("candidate_validity_predicate_matches") is not True
        ):
            raise BenchmarkContractError("matched-recall source HNSW index provenance is invalid")
        query_relation = _require_mapping(
            database.get("query_table"), "matched-recall query relation provenance"
        )
        if str(query_relation.get("name") or "") != source_query_table:
            raise BenchmarkContractError(
                "matched-recall query relation name does not match its query-table argument"
            )
        for field in ("oid", "relfilenode", "row_count"):
            if int(query_relation.get(field) or 0) <= 0:
                raise BenchmarkContractError(
                    f"matched-recall query relation provenance is incomplete: {field}"
                )
        if not isinstance(query_relation.get("columns"), list) or not query_relation["columns"]:
            raise BenchmarkContractError(
                "matched-recall query relation has no bound column signature"
            )
    else:
        database = dict(run_spec.get("database")) if isinstance(run_spec.get("database"), Mapping) else {}

    outputs = _require_mapping(manifest.get("outputs"), "matched-recall outputs")
    selected_path = _matched_artifact_path(
        manifest_path, outputs.get("selected"), "matched-recall selected artifact"
    )
    selected_rows = read_csv(selected_path)
    selected_metadata = _require_mapping(
        outputs.get("selected"), "matched-recall selected artifact"
    )
    if int(selected_metadata.get("row_count") or -1) != len(selected_rows):
        raise BenchmarkContractError("matched-recall selected row count does not match")

    rows_by_key: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        mode = str(row.get("mode") or "")
        for method, expected_mode in MODE_BY_METHOD.items():
            if mode == expected_mode:
                rows_by_key[
                    (str(row.get("filter_name") or ""), method, float(row["target_recall"]))
                ].append(row)
    traversal_burst = int(run_args.get("traversal_guided_burst") or 0)
    manifest_prioritization = run_args.get("traversal_guided_prioritization")
    if traversal_burst <= 0:
        raise BenchmarkContractError("matched-recall D1 traversal burst provenance is incomplete")
    if guidance_filter_strategy == "safe_guided" and manifest_prioritization is not False:
        raise BenchmarkContractError(
            "safe_guided matched-recall provenance must disable dual-frontier prioritization"
        )
    if guidance_filter_strategy == "traversal_guided" and manifest_prioritization is not True:
        raise BenchmarkContractError(
            "traversal_guided matched-recall provenance must enable dual-frontier prioritization"
        )

    configs: dict[tuple[str, str, float], SearchConfig] = {}
    evidence: list[Mapping[str, Any]] = []
    for item in filters:
        for method in METHODS:
            for target in targets:
                key = (item.name, method, float(target))
                rows = rows_by_key.get(key, [])
                if len(rows) != 1:
                    raise BenchmarkContractError(
                        f"matched-recall selected config coverage is not exact for {key}: {len(rows)}"
                    )
                row = rows[0]
                config = _selected_config_from_row(
                    row,
                    method=method,
                    target=float(target),
                    traversal_guided_burst=traversal_burst,
                    guidance_filter_strategy=guidance_filter_strategy,
                )
                configs[key] = config
                evidence.append(
                    {
                        "filter_name": item.name,
                        "method": method,
                        "mode": MODE_BY_METHOD[method],
                        "guidance_filter_strategy": guidance_filter_strategy,
                        "target_recall": float(target),
                        "config": config.label,
                        "ef_search": config.ef_search,
                        "max_scan_tuples": config.max_scan_tuples,
                        "scan_mem_multiplier": config.scan_mem_multiplier,
                        "iterative_scan": config.iterative_scan,
                        "guided_collect_target": config.guided_collect_target,
                        "traversal_guided_target": config.traversal_guided_target,
                        "traversal_guided_burst": config.traversal_guided_burst,
                        "recall_mean": float(row["recall_mean"]),
                        "recall_lcb95": float(row["recall_lcb95"]),
                        "latency_mean_ms": float(row["latency_mean_ms"]),
                        "samples": int(row["samples"]),
                        "selection_source": "independently_audited_matched_recall_selected_artifact",
                    }
                )

    provenance = {
        "contract": "independently_audited_lcb_matched_recall_configs_v1",
        "requested_slice_complete": True,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_artifact": str(selected_path),
        "selected_artifact_sha256": sha256_file(selected_path),
        "independent_audit": audit,
        "truth_csv": str(truth_csv.resolve()),
        "truth_sha256": sha256_file(truth_csv),
        "filters_csv": str(filters_csv.resolve()),
        "filters_sha256": sha256_file(filters_csv),
        "runtime": dict(runtime),
        "database": dict(database),
        "source_table": source_table,
        "source_index": source_index,
        "source_query_table": source_query_table,
        "guidance_filter_strategy": guidance_filter_strategy,
    }
    return MatchedRecallBundle(
        configs, tuple(evidence), provenance, manifest, guidance_filter_strategy
    )


def bind_matched_recall_provenance(
    args: argparse.Namespace, bundle: MatchedRecallBundle
) -> None:
    run_spec = _require_mapping(bundle.manifest.get("run_spec"), "matched-recall run_spec")
    run_args = _require_mapping(run_spec.get("args"), "matched-recall run_spec.args")
    expected_table = str(bundle.provenance.get("source_table") or "")
    expected_index = str(bundle.provenance.get("source_index") or "")
    expected_query_table = str(bundle.provenance.get("source_query_table") or "")
    if args.insertion_table != expected_table or args.insertion_index != expected_index:
        raise BenchmarkContractError(
            "throughput table/index do not match matched-recall provenance: "
            f"expected={expected_table}/{expected_index} "
            f"observed={args.insertion_table}/{args.insertion_index}"
        )
    if str(args.query_table or args.insertion_table) != expected_query_table:
        raise BenchmarkContractError(
            "throughput query table differs from matched-recall provenance: "
            f"expected={expected_query_table} observed={args.query_table or args.insertion_table}"
        )
    expected_predicate = str(run_args.get("candidate_validity_predicate") or "")
    if args.candidate_validity_predicate != expected_predicate:
        raise BenchmarkContractError(
            "throughput candidate-validity predicate differs from matched-recall provenance"
        )
    runtime = _require_mapping(bundle.provenance.get("runtime"), "matched-recall runtime")
    if args.guidance_filter_strategy != bundle.guidance_filter_strategy:
        raise BenchmarkContractError(
            "throughput guidance_filter_strategy differs from matched-recall provenance: "
            f"requested={args.guidance_filter_strategy!r} "
            f"matched_recall={bundle.guidance_filter_strategy!r}"
        )
    build_id = str(runtime.get("loaded_vector_sqlens_build_id") or "")
    vector_sha = str(runtime.get("loaded_vector_so_sha256") or "")
    if args.expected_sqlens_build_id and args.expected_sqlens_build_id != build_id:
        raise BenchmarkContractError(
            "explicit SQLens build ID conflicts with matched-recall provenance"
        )
    if args.expected_vector_so_sha256 and args.expected_vector_so_sha256 != vector_sha:
        raise BenchmarkContractError(
            "explicit vector.so SHA256 conflicts with matched-recall provenance"
        )
    args.expected_sqlens_build_id = build_id
    args.expected_vector_so_sha256 = vector_sha
    # D1 and Stock deliberately share the exact HNSW index. These aliases keep
    # the common pgvector runtime helper from fingerprinting or prewarming D2.
    args.bfs_table = args.insertion_table
    args.bfs_index = args.insertion_index


def parse_cpu_set(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    normalized = normalize_cpu_list(value)
    cpus: list[int] = []
    for token in normalized.split(","):
        if "-" in token:
            first, last = (int(item) for item in token.split("-", 1))
            cpus.extend(range(first, last + 1))
        else:
            cpus.append(int(token))
    return tuple(cpus)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[position]


def bootstrap_mean_ci(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    numeric = [float(value) for value in values]
    if not numeric:
        return 0.0, 0.0
    if len(numeric) == 1 or samples <= 0:
        return numeric[0], numeric[0]
    rng = random.Random(seed)
    size = len(numeric)
    means = [statistics.fmean(rng.choices(numeric, k=size)) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def bootstrap_pooled_ratio_ci(
    numerators: Sequence[float],
    denominators: Sequence[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a pooled ratio by resampling whole repeat-level pairs."""
    if len(numerators) != len(denominators) or not numerators:
        raise BenchmarkContractError("ratio CI requires nonempty paired repeat samples")
    pairs = [(float(numerator), float(denominator)) for numerator, denominator in zip(numerators, denominators)]
    if any(denominator <= 0.0 for _, denominator in pairs):
        raise BenchmarkContractError("ratio CI requires positive repeat wall-clock durations")
    point = sum(numerator for numerator, _ in pairs) / sum(
        denominator for _, denominator in pairs
    )
    if len(pairs) == 1 or samples <= 0:
        return point, point
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = rng.choices(pairs, k=len(pairs))
        estimates.append(
            sum(numerator for numerator, _ in selected)
            / sum(denominator for _, denominator in selected)
        )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def cluster_bootstrap_percentile_ci(
    rows: Sequence[Mapping[str, Any]],
    fraction: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap latency percentiles by resampling whole query-vector clusters."""
    clusters: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if not row.get("error"):
            clusters[int(row["query_no"])].append(float(row["latency_ms"]))
    values = list(clusters.values())
    if not values:
        return 0.0, 0.0
    point = percentile([latency for cluster in values for latency in cluster], fraction)
    if len(values) == 1 or samples <= 0:
        return point, point
    rng = random.Random(seed)
    estimates = [
        percentile([latency for cluster in rng.choices(values, k=len(values)) for latency in cluster], fraction)
        for _ in range(samples)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def interleaved_method_order(block_no: int, seed: int) -> tuple[str, ...]:
    methods = list(METHODS)
    random.Random(seed).shuffle(methods)
    offset = block_no % len(methods)
    return tuple(methods[offset:] + methods[:offset])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_filters(path: Path, selected: set[str] | None = None) -> list[FilterSpec]:
    rows = read_csv(path)
    required = {"filter_name", "predicate", "atoms", "count", "actual_pct", "source"}
    if not rows or not required <= set(rows[0]):
        raise BenchmarkContractError("Amazon filter CSV is missing formal fields")
    filters: list[FilterSpec] = []
    for row in rows:
        if selected and row["filter_name"] not in selected:
            continue
        atoms = tuple(part.strip() for part in row["atoms"].split("||") if part.strip())
        if not atoms or any(not atom.startswith("sql:") for atom in atoms):
            raise BenchmarkContractError(f"invalid guidance atoms for {row['filter_name']}")
        filters.append(
            FilterSpec(
                row["filter_name"],
                row["predicate"],
                atoms,
                int(row["count"]),
                float(row["actual_pct"]),
            )
        )
    if not filters or len({item.name for item in filters}) != len(filters):
        raise BenchmarkContractError("selected filters must be nonempty and unique")
    if selected and {item.name for item in filters} != selected:
        missing = sorted(selected - {item.name for item in filters})
        raise BenchmarkContractError(f"unknown selected filters: {missing}")
    if not selected and len(filters) != 14:
        raise BenchmarkContractError("formal all-filter run requires fourteen Amazon cohorts")
    if any("%" in item.predicate for item in filters):
        raise BenchmarkContractError("synthetic modulo predicates are forbidden")
    return filters


def resolve_evaluation_filters(
    args: argparse.Namespace,
    available: Sequence[FilterSpec],
) -> list[FilterSpec]:
    """Resolve an explicit formal scope without allowing cherry-picked subsets."""
    names = tuple(item.name for item in available)
    if len(names) != len(set(names)):
        raise BenchmarkContractError("filter CSV contains duplicate filter names")
    requested = tuple(str(name) for name in args.filter_names)
    if len(requested) != len(set(requested)):
        raise BenchmarkContractError("--filter-names contains duplicates")

    if args.allow_nonformal_debug:
        if not requested:
            return list(available)
        selected = set(requested)
        resolved = [item for item in available if item.name in selected]
        if {item.name for item in resolved} != selected:
            missing = sorted(selected - {item.name for item in resolved})
            raise BenchmarkContractError(f"unknown requested filters: {missing}")
        return resolved

    if len(available) != FORMAL_FILTER_COUNT:
        raise BenchmarkContractError(
            f"formal throughput requires exactly {FORMAL_FILTER_COUNT} Amazon filters; "
            f"observed {len(available)}"
        )
    if args.evaluation_scope == "full_matrix":
        if requested:
            raise BenchmarkContractError(
                "full_matrix formal mode forbids --filter-names; it always executes all 14 filters"
            )
        return list(available)

    if args.evaluation_scope == "representative_filters":
        if set(REPRESENTATIVE_FILTERS) - set(names):
            raise BenchmarkContractError(
                "formal filter CSV lacks the canonical representative-filter set"
            )
        if requested and requested != REPRESENTATIVE_FILTERS:
            raise BenchmarkContractError(
                "representative_filters formal mode requires exactly the canonical "
                f"filter order: {list(REPRESENTATIVE_FILTERS)}"
            )
        selected = set(REPRESENTATIVE_FILTERS)
        return [item for item in available if item.name in selected]

    raise BenchmarkContractError(
        f"unsupported evaluation scope {args.evaluation_scope!r}; expected one of {EVALUATION_SCOPES}"
    )


def load_truth(
    path: Path,
    filters: Sequence[FilterSpec],
    candidate_validity_predicate: str,
) -> tuple[dict[tuple[str, int], Any], dict[int, int]]:
    # Importing the formal loader here keeps this module's public data classes
    # independent while preserving the exact tie-aware recall implementation.
    try:
        from .pgvector_design1_design2_design3_selectivity_benchmark import load_tie_aware_truth
    except ImportError:
        from pgvector_design1_design2_design3_selectivity_benchmark import load_tie_aware_truth

    truth, query_ids = load_tie_aware_truth(
        path,
        expected_self_excluded=True,
        expected_candidate_validity_predicate=candidate_validity_predicate,
    )
    wanted = {item.name for item in filters}
    truth = {key: value for key, value in truth.items() if key[0] in wanted}
    return truth, query_ids


def validate_truth_coverage(
    truth: Mapping[tuple[str, int], Any],
    query_ids: Mapping[int, int],
    filters: Sequence[FilterSpec],
    calibration_query_nos: Iterable[int],
    measurement_query_nos: Iterable[int],
) -> None:
    calibration = set(calibration_query_nos)
    measurement = set(measurement_query_nos)
    if calibration & measurement:
        raise BenchmarkContractError("calibration and measurement query cohorts overlap")
    required_queries = calibration | measurement
    if any(query_no not in query_ids for query_no in required_queries):
        missing = sorted(required_queries - set(query_ids))
        raise BenchmarkContractError(f"GT is missing query IDs: {missing[:10]}")
    if len({query_ids[query_no] for query_no in required_queries}) != len(required_queries):
        raise BenchmarkContractError("GT query_no values do not map to distinct query IDs")
    expected = {(item.name, query_no) for item in filters for query_no in required_queries}
    missing_pairs = expected - set(truth)
    if missing_pairs:
        raise BenchmarkContractError(f"exact GT lacks {len(missing_pairs)} filter/query pairs")
    for item in filters:
        counts = {int(truth[(item.name, query_no)].filtered_rows) for query_no in required_queries}
        if counts != {item.expected_rows}:
            raise BenchmarkContractError(
                f"GT/filter candidate count mismatch for {item.name}: "
                f"config={item.expected_rows} truth={sorted(counts)}"
            )


def validate_calibration_measurement_split(
    calibration_truth: Mapping[tuple[str, int], Any],
    calibration_query_ids: Mapping[int, int],
    measurement_truth: Mapping[tuple[str, int], Any],
    measurement_query_ids: Mapping[int, int],
    filters: Sequence[FilterSpec],
) -> dict[str, Any]:
    """Validate both exact GT artifacts and disjointness by actual row ID."""
    calibration_nos = set(SELECTION_QUERY_NOS)
    measurement_nos = set(MEASUREMENT_QUERY_NOS)
    validate_truth_coverage(
        calibration_truth,
        calibration_query_ids,
        filters,
        calibration_nos,
        (),
    )
    validate_truth_coverage(
        measurement_truth,
        measurement_query_ids,
        filters,
        measurement_nos,
        (),
    )
    calibration_ids = {
        int(calibration_query_ids[query_no]) for query_no in calibration_nos
    }
    measurement_ids = {
        int(measurement_query_ids[query_no]) for query_no in measurement_nos
    }
    overlap = calibration_ids & measurement_ids
    if overlap:
        raise BenchmarkContractError(
            "calibration and measurement query IDs overlap: "
            f"{len(overlap)} IDs, examples={sorted(overlap)[:10]}"
        )
    return {
        "passed": True,
        "selection_query_numbers": len(calibration_nos),
        "measurement_query_numbers": len(measurement_nos),
        "selection_query_ids": len(calibration_ids),
        "measurement_query_ids": len(measurement_ids),
        "actual_query_id_disjoint": True,
    }


def validate_workload_query_mapping(
    workload: Workload,
    query_ids: Mapping[int, int],
) -> None:
    for request in workload.requests:
        if query_ids.get(request.query_no) != request.query_id:
            raise BenchmarkContractError(
                f"workload query mapping disagrees with exact GT for query_no={request.query_no}"
            )


def verify_truth_manifest(
    truth_path: Path,
    manifest_path: Path,
    candidate_validity_predicate: str,
    expected_candidate_rows: int,
) -> dict[str, Any]:
    if manifest_path.name != truth_path.with_name(truth_path.stem + "_manifest.json").name:
        raise BenchmarkContractError("truth manifest filename must be derived from the truth CSV")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"cannot read exact GT manifest: {exc}") from exc
    truth_sha = sha256_file(truth_path)
    manifest_sha = ((manifest.get("outputs") or {}).get("truth_csv") or {}).get("sha256")
    predicate = ((manifest.get("validity_contract") or {}).get("candidate_validity_predicate"))
    candidate_universe = manifest.get("candidate_universe") or {}
    candidate_rows = candidate_universe.get("rows")
    candidate_rows_source = "truth_manifest.candidate_universe"
    candidate_source_manifest: dict[str, Any] | None = None
    if candidate_rows is not None and candidate_universe.get(
        "candidate_validity_predicate"
    ) != candidate_validity_predicate:
        raise BenchmarkContractError(
            "GT candidate-universe predicate does not match its validity contract"
        )
    if candidate_rows is None:
        source_ref = ((manifest.get("query_source") or {}).get("manifest") or {})
        source_path_value = source_ref.get("path")
        source_sha = source_ref.get("sha256")
        if not source_path_value or not source_sha:
            raise BenchmarkContractError(
                "GT manifest does not bind a candidate-universe row count"
            )
        source_path = Path(str(source_path_value))
        if not source_path.is_file() or sha256_file(source_path) != source_sha:
            raise BenchmarkContractError(
                "GT query-cohort manifest identity does not match its bound SHA256"
            )
        try:
            candidate_source_manifest = json.loads(
                source_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkContractError(
                f"cannot read GT query-cohort manifest: {exc}"
            ) from exc
        source_predicate = candidate_source_manifest.get(
            "candidate_validity_predicate"
        )
        if (
            candidate_source_manifest.get("artifact_valid") is not True
            or source_predicate != candidate_validity_predicate
        ):
            raise BenchmarkContractError(
                "GT query-cohort candidate-universe contract mismatch"
            )
        candidate_rows = (
            (candidate_source_manifest.get("eligible_query_population") or {}).get(
                "embedding_valid_rows"
            )
        )
        candidate_rows_source = "bound_query_cohort_manifest.embedding_valid_rows"
    if manifest.get("artifact_valid") is not True:
        raise BenchmarkContractError("exact GT manifest is not artifact_valid")
    if manifest_sha != truth_sha:
        raise BenchmarkContractError("exact GT manifest does not bind the truth CSV SHA256")
    if predicate != candidate_validity_predicate:
        raise BenchmarkContractError("GT candidate universe predicate mismatch")
    try:
        observed_candidate_rows = int(candidate_rows)
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError(
            "GT candidate-universe row count is missing or invalid"
        ) from exc
    if observed_candidate_rows != expected_candidate_rows:
        raise BenchmarkContractError("GT eligible candidate row count mismatch")
    if manifest.get("self_excluded") is not True:
        raise BenchmarkContractError("formal GT must use self-excluded queries")
    return {
        "artifact_valid": True,
        "truth_csv": str(truth_path),
        "truth_sha256": truth_sha,
        "truth_manifest": str(manifest_path),
        "truth_manifest_sha256": sha256_file(manifest_path),
        "candidate_validity_predicate": predicate,
        "eligible_candidate_rows": observed_candidate_rows,
        "candidate_rows_source": candidate_rows_source,
        "self_excluded": True,
    }


def discover_true_query_file(results_dir: Path = RESULTS) -> Path | None:
    """Discover only the preregistered q10200 cohort, never legacy replays."""
    candidates = {
        path.resolve()
        for pattern in TRUE_QUERY_GLOBS
        for path in results_dir.glob(pattern)
        if path.is_file()
    }
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _resolve_manifest_artifact_path(manifest_path: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path or ""))
    return (path if path.is_absolute() else manifest_path.parent / path).resolve()


def verify_measurement_query_manifest(
    query_path: Path,
    manifest_path: Path,
    candidate_validity_predicate: str,
) -> dict[str, Any]:
    """Verify that the input is the complete, split-aware q10200 cohort."""
    manifest_path = manifest_path.resolve()
    query_path = query_path.resolve()
    manifest = _read_json_object(manifest_path, "q10200 query-cohort manifest")
    if manifest.get("artifact_valid") is not True:
        raise BenchmarkContractError("q10200 query-cohort manifest is not artifact_valid")
    if manifest.get("candidate_validity_predicate") != candidate_validity_predicate:
        raise BenchmarkContractError("q10200 query-cohort candidate predicate mismatch")
    output = _require_mapping(
        (manifest.get("outputs") or {}).get("cohort_csv"),
        "q10200 query-cohort output",
    )
    bound_path = _resolve_manifest_artifact_path(manifest_path, output.get("path"))
    if bound_path != query_path:
        raise BenchmarkContractError(
            "q10200 query-cohort manifest does not bind the measurement query file"
        )
    if str(output.get("sha256") or "") != sha256_file(query_path):
        raise BenchmarkContractError("q10200 query-cohort SHA256 does not match")
    if int(output.get("rows") or -1) != 10_200:
        raise BenchmarkContractError("q10200 query-cohort must contain exactly 10200 rows")
    selection = _require_mapping(manifest.get("selection"), "q10200 query-cohort selection")
    calibration = _require_mapping(selection.get("calibration"), "q10200 calibration selection")
    final = _require_mapping(selection.get("final"), "q10200 final selection")
    if int(calibration.get("queries") or -1) != 100 or int(final.get("queries") or -1) != 10_100:
        raise BenchmarkContractError(
            "q10200 query-cohort must declare 100 calibration and 10100 final queries"
        )
    if selection.get("disjoint") is not True:
        raise BenchmarkContractError("q10200 query-cohort selection is not disjoint")
    return {
        "artifact_valid": True,
        "path": str(query_path),
        "sha256": sha256_file(query_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": 10_200,
        "calibration_queries": 100,
        "measurement_queries": 10_000,
        "selection_disjoint": True,
        "candidate_validity_predicate": candidate_validity_predicate,
    }


def verify_measurement_truth_manifest(
    truth_path: Path,
    manifest_path: Path,
    candidate_validity_predicate: str,
    expected_candidate_rows: int,
    query_path: Path,
) -> dict[str, Any]:
    """Verify the exact truth artifact corresponding to q10200."""
    evidence = verify_truth_manifest(
        truth_path,
        manifest_path,
        candidate_validity_predicate,
        expected_candidate_rows,
    )
    manifest = _read_json_object(manifest_path.resolve(), "measurement exact-truth manifest")
    calibration = _require_mapping(manifest.get("calibration"), "measurement truth calibration")
    final = _require_mapping(manifest.get("final"), "measurement truth final split")
    if int(calibration.get("queries") or -1) != 100:
        raise BenchmarkContractError("measurement exact truth must declare 100 calibration queries")
    if int(final.get("queries") or -1) != 10_100:
        raise BenchmarkContractError("measurement exact truth must declare 10100 final queries")
    if manifest.get("query_ids_disjoint") is not True:
        raise BenchmarkContractError("measurement exact truth calibration/final IDs are not disjoint")
    source = _require_mapping(manifest.get("query_source"), "measurement exact-truth query source")
    cohort = _require_mapping(source.get("cohort_csv"), "measurement exact-truth cohort source")
    bound_query_path = _resolve_manifest_artifact_path(manifest_path.resolve(), cohort.get("path"))
    if bound_query_path != query_path.resolve():
        raise BenchmarkContractError(
            "measurement exact truth is not generated from the supplied q10200 cohort"
        )
    if str(cohort.get("sha256") or "") != sha256_file(query_path):
        raise BenchmarkContractError("measurement exact truth cohort SHA256 does not match")
    return {
        **evidence,
        "query_cohort": str(query_path.resolve()),
        "query_cohort_sha256": sha256_file(query_path),
        "calibration_queries": 100,
        "measurement_queries": 10_000,
        "query_ids_disjoint": True,
    }


def load_true_query_workload(
    path: Path,
    calibration_query_ids: Iterable[int],
    *,
    query_manifest: Path | None = None,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> Workload:
    """Load q10200 and return only its q200..q10199 measurement split."""
    if query_manifest is not None:
        verify_measurement_query_manifest(path, query_manifest, candidate_validity_predicate)
    rows = read_csv(path)
    required = {"query_no", "query_id"}
    if len(rows) != 10_200 or not rows or not required <= set(rows[0]):
        raise BenchmarkContractError(
            "q10200 cohort must contain exactly 10200 rows with query_no/query_id"
        )
    by_query_no: dict[int, dict[str, str]] = {}
    for row in rows:
        query_no = int(row["query_no"])
        if query_no in by_query_no:
            raise BenchmarkContractError("q10200 cohort has duplicate query_no values")
        by_query_no[query_no] = row
    if set(by_query_no) != set(range(10_200)):
        raise BenchmarkContractError("q10200 cohort query_no coverage must be exactly 0..10199")
    all_query_ids = [int(by_query_no[query_no]["query_id"]) for query_no in range(10_200)]
    if len(set(all_query_ids)) != 10_200:
        raise BenchmarkContractError("q10200 cohort must contain unique query_id values")
    measurement_rows = [by_query_no[query_no] for query_no in MEASUREMENT_QUERY_NOS]
    measurement_ids = {int(row["query_id"]) for row in measurement_rows}
    calibration_ids = {int(value) for value in calibration_query_ids}
    if measurement_ids & calibration_ids:
        raise BenchmarkContractError(
            "q10200 measurement query IDs overlap matched-recall selection query IDs"
        )
    if len(measurement_ids) != FORMAL_REQUESTS:
        raise BenchmarkContractError(
            "q10200 measurement split must contain 10,000 unique query IDs"
        )
    requests = tuple(
        WorkloadRequest(
            request_no,
            MEASUREMENT_QUERY_NOS[request_no],
            int(by_query_no[MEASUREMENT_QUERY_NOS[request_no]]["query_id"]),
            0,
        )
        for request_no in range(FORMAL_REQUESTS)
    )
    return Workload(
        requests,
        "q10200_cohort_measurement_split",
        str(path.resolve()),
        sha256_file(path),
        "q200..q10199",
        False,
        FORMAL_REQUESTS,
    )


def build_replay_workload(query_ids: Mapping[int, int], requests: int = FORMAL_REQUESTS) -> Workload:
    if requests <= 0:
        raise ValueError("requests must be positive")
    missing = set(REPLAY_QUERY_NOS) - set(query_ids)
    if missing:
        raise BenchmarkContractError(f"GT is missing held-out replay query IDs: {sorted(missing)}")
    if len({query_ids[query_no] for query_no in REPLAY_QUERY_NOS}) != len(REPLAY_QUERY_NOS):
        raise BenchmarkContractError("q100..q199 must map to 100 unique query vectors")
    trace = tuple(
        WorkloadRequest(
            request_no,
            REPLAY_QUERY_NOS[request_no % len(REPLAY_QUERY_NOS)],
            query_ids[REPLAY_QUERY_NOS[request_no % len(REPLAY_QUERY_NOS)]],
            request_no // len(REPLAY_QUERY_NOS),
        )
        for request_no in range(requests)
    )
    return Workload(
        trace,
        "heldout_q100_q199_trace_replay",
        "",
        "",
        "q100..q199",
        True,
        len(REPLAY_QUERY_NOS),
    )


def choose_workload(
    query_ids: Mapping[int, int],
    explicit_query_file: Path | None,
    results_dir: Path,
    requests: int,
    *,
    calibration_query_ids: Iterable[int] = (),
    query_manifest: Path | None = None,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
    formal: bool = True,
) -> Workload:
    discovered = explicit_query_file or discover_true_query_file(results_dir)
    if discovered is not None and not discovered.is_file():
        if formal:
            raise BenchmarkContractError(
                f"formal prerequisite missing: q10200 measurement cohort does not exist: {discovered}"
            )
        discovered = None
    if discovered is not None:
        manifest = query_manifest
        if manifest is None:
            manifest = discovered.with_name(discovered.stem + "_manifest.json")
        if requests != FORMAL_REQUESTS:
            raise BenchmarkContractError("q10200 measurement workload cannot be truncated")
        return load_true_query_workload(
            discovered,
            calibration_query_ids,
            query_manifest=manifest,
            candidate_validity_predicate=candidate_validity_predicate,
        )
    if formal:
        raise BenchmarkContractError(
            "formal prerequisite missing: q10200 measurement cohort CSV and manifest "
            "are required; q100..q199 replay is debug-only"
        )
    return build_replay_workload(query_ids, requests)


def output_paths(raw_path: Path) -> dict[str, Path]:
    stem = raw_path.stem[:-4] if raw_path.stem.endswith("_raw") else raw_path.stem
    return {
        "raw": raw_path,
        "configuration": raw_path.with_name(stem + "_matched_recall_configs.csv"),
        "summary": raw_path.with_name(stem + "_summary.csv"),
        "manifest": raw_path.with_name(stem + "_manifest.json"),
        "checkpoint": raw_path.with_name(stem + "_checkpoint.json"),
    }


def fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)
    fsync_parent(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_replace_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8"),
    )


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], include_header: bool) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=list(fields), extrasaction="ignore")
    if include_header:
        writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
        )
    return target.getvalue().encode("utf-8")


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    resolved_fields = list(fields or [])
    if not resolved_fields:
        for row in rows:
            for key in row:
                if key not in resolved_fields:
                    resolved_fields.append(key)
    atomic_replace_bytes(path, csv_bytes(rows, resolved_fields, True))


def append_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[str, Any]:
    payload = csv_bytes(rows, fields, False)
    with path.open("ab") as target:
        start = target.tell()
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
        end = target.tell()
    return {
        "start_offset": start,
        "end_offset": end,
        "bytes": len(payload),
        "rows": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def initialize_raw_csv(path: Path, fields: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = csv_bytes([], fields, True)
    with path.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    fsync_parent(path)
    return len(payload)


def load_checkpoint(path: Path, run_spec_hash: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"cannot read checkpoint: {exc}") from exc
    if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise BenchmarkContractError("checkpoint schema version mismatch")
    if value.get("run_spec_hash") != run_spec_hash:
        raise BenchmarkContractError("checkpoint run-spec/provenance mismatch")
    return value


def prepare_resume_raw(path: Path, checkpoint: Mapping[str, Any], fields: Sequence[str]) -> None:
    expected_offset = int(checkpoint["raw_byte_offset"])
    if not path.exists() or path.stat().st_size < expected_offset:
        raise BenchmarkContractError("raw CSV is shorter than its durable checkpoint")
    expected_header = csv_bytes([], fields, True)
    with path.open("r+b") as target:
        if target.read(len(expected_header)) != expected_header:
            raise BenchmarkContractError("raw CSV header does not match this runner")
        target.truncate(expected_offset)
        target.flush()
        os.fsync(target.fileno())
    fsync_parent(path)
    pair_artifacts = _require_mapping(
        checkpoint.get("raw_pair_artifacts"), "checkpoint raw_pair_artifacts"
    )
    ranges: list[tuple[int, int, str]] = []
    with path.open("rb") as source:
        for pair_key, artifact_value in pair_artifacts.items():
            artifact = _require_mapping(
                artifact_value, f"checkpoint raw pair artifact {pair_key}"
            )
            start = int(artifact["start_offset"])
            end = int(artifact["end_offset"])
            if start < len(expected_header) or end <= start or end > expected_offset:
                raise BenchmarkContractError(f"invalid raw byte range for pair {pair_key}")
            if (
                sorted(str(value) for value in artifact.get("methods", []))
                != sorted(METHODS)
                or len(set(str(value) for value in artifact.get("arm_keys", []))) != 2
            ):
                raise BenchmarkContractError(
                    f"raw artifact does not bind one complete method pair: {pair_key}"
                )
            source.seek(start)
            payload = source.read(end - start)
            if len(payload) != end - start or hashlib.sha256(payload).hexdigest() != artifact.get(
                "sha256"
            ):
                raise BenchmarkContractError(f"raw CSV pair segment hash mismatch: {pair_key}")
            ranges.append((start, end, str(pair_key)))
    expected_start = len(expected_header)
    for start, end, pair_key in sorted(ranges):
        if start != expected_start:
            raise BenchmarkContractError(
                f"raw CSV committed pair segments are not contiguous at {pair_key}"
            )
        expected_start = end
    if expected_start != expected_offset:
        raise BenchmarkContractError(
            "raw CSV durable byte offset is not the end of committed pair segments"
        )


def restore_csv_row_prefix(
    path: Path,
    committed_rows: int,
    expected_sha256: str = "",
) -> list[dict[str, str]]:
    if committed_rows < 0:
        raise BenchmarkContractError("checkpoint contains a negative CSV row count")
    if not path.exists():
        if committed_rows:
            raise BenchmarkContractError(f"checkpoint expects missing CSV rows in {path}")
        return []
    rows = read_csv(path)
    if len(rows) < committed_rows:
        raise BenchmarkContractError(f"{path} is shorter than its durable checkpoint")
    committed = rows[:committed_rows]
    if len(rows) != committed_rows:
        write_csv_atomic(path, committed)
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise BenchmarkContractError(f"{path} committed-prefix SHA256 mismatch")
    return committed


def load_measurement_cell_rows(
    path: Path,
    target: float,
    clients: int,
    filter_name: str,
) -> list[dict[str, str]]:
    """Stream only one resumable cell instead of loading a multi-million-row raw CSV."""
    with path.open(newline="", encoding="utf-8") as source:
        return [
            row
            for row in csv.DictReader(source)
            if float(row["target_recall"]) == target
            and int(row["clients"]) == clients
            and row["filter_name"] == filter_name
        ]


def measurement_arm_key(target: float, clients: int, filter_name: str, method: str, repeat: int) -> str:
    return (
        f"target={target:.6f}|clients={clients}|filter={filter_name}|method={method}|repeat={repeat}"
    )


def measurement_pair_key(target: float, clients: int, filter_name: str, repeat: int) -> str:
    return (
        f"target={target:.6f}|clients={clients}|filter={filter_name}|repeat={repeat}"
    )


def measurement_cell_key(target: float, clients: int, filter_name: str) -> str:
    return f"target={target:.6f}|clients={clients}|filter={filter_name}"


def validate_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    workload: Workload,
    measurement_repeats: int,
) -> dict[str, Any]:
    """Accept only durably committed Stock/D1 pairs; reject every half-pair shape."""
    if "completed_measurement_arms" in checkpoint or "raw_arm_artifacts" in checkpoint:
        raise BenchmarkContractError("checkpoint uses obsolete arm-level resume state")
    expected_pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for arm in schedule:
        expected_pairs[
            measurement_pair_key(
                float(arm["target_recall"]),
                int(arm["clients"]),
                str(arm["filter_name"]),
                int(arm["measurement_repeat"]),
            )
        ].append(arm)
    for pair_key, arms in expected_pairs.items():
        if (
            len(arms) != len(METHODS)
            or {str(arm["method"]) for arm in arms} != set(METHODS)
            or len({int(arm["block_no"]) for arm in arms}) != 1
        ):
            raise BenchmarkContractError(
                f"schedule does not define one adjacent method pair: {pair_key}"
            )

    completed_list = [
        str(value) for value in checkpoint.get("completed_measurement_pairs", [])
    ]
    completed_pairs = set(completed_list)
    if len(completed_list) != len(completed_pairs) or not completed_pairs <= set(expected_pairs):
        raise BenchmarkContractError(
            "checkpoint completed-pair coverage is duplicate or outside this schedule"
        )
    artifacts = _require_mapping(
        checkpoint.get("raw_pair_artifacts"), "checkpoint raw_pair_artifacts"
    )
    pair_evidence = _require_mapping(
        checkpoint.get("pair_evidence"), "checkpoint pair_evidence"
    )
    if set(artifacts) != completed_pairs or set(pair_evidence) != completed_pairs:
        raise BenchmarkContractError(
            "checkpoint pair artifacts/evidence do not exactly match committed pairs"
        )

    expected_arm_keys: set[str] = set()
    for pair_key in completed_pairs:
        scheduled = expected_pairs[pair_key]
        arm_keys = {
            measurement_arm_key(
                float(arm["target_recall"]),
                int(arm["clients"]),
                str(arm["filter_name"]),
                str(arm["method"]),
                int(arm["measurement_repeat"]),
            )
            for arm in scheduled
        }
        expected_arm_keys.update(arm_keys)
        artifact = _require_mapping(artifacts[pair_key], f"raw pair {pair_key}")
        evidence = _require_mapping(pair_evidence[pair_key], f"pair evidence {pair_key}")
        if (
            set(str(value) for value in artifact.get("arm_keys", [])) != arm_keys
            or set(str(value) for value in evidence.get("arm_keys", [])) != arm_keys
            or set(str(value) for value in evidence.get("methods", [])) != set(METHODS)
            or int(artifact.get("rows") or 0) != len(workload.requests) * len(METHODS)
            or evidence.get("committed") is not True
        ):
            raise BenchmarkContractError(
                f"checkpoint does not bind a complete committed method pair: {pair_key}"
            )

    arm_evidence = list(checkpoint.get("arm_evidence") or [])
    observed_arm_keys = [str(row.get("arm_key") or "") for row in arm_evidence]
    if (
        len(observed_arm_keys) != len(set(observed_arm_keys))
        or set(observed_arm_keys) != expected_arm_keys
    ):
        raise BenchmarkContractError(
            "checkpoint arm evidence contains a missing or half method pair"
        )
    expected_measurement_rows = (
        len(completed_pairs) * len(METHODS) * len(workload.requests)
    )
    if int(checkpoint.get("measurement_rows") or 0) != expected_measurement_rows:
        raise BenchmarkContractError(
            "checkpoint raw row count does not match its committed method pairs"
        )
    repeat_rows = [row for row in summary_rows if row.get("summary_type") == "repeat"]
    repeat_keys = [
        measurement_arm_key(
            float(row["target_recall"]),
            int(row["clients"]),
            str(row["filter_name"]),
            str(row["method"]),
            int(row["measurement_repeat"]),
        )
        for row in repeat_rows
    ]
    if len(repeat_keys) != len(set(repeat_keys)) or set(repeat_keys) != expected_arm_keys:
        raise BenchmarkContractError(
            "checkpoint repeat summaries contain a missing or half method pair"
        )

    completed_cell_list = [
        str(value) for value in checkpoint.get("completed_measurement_cells", [])
    ]
    completed_cells = set(completed_cell_list)
    if len(completed_cell_list) != len(completed_cells):
        raise BenchmarkContractError("checkpoint completed-cell coverage is duplicated")
    expected_completed_cells = {
        measurement_cell_key(
            float(arms[0]["target_recall"]),
            int(arms[0]["clients"]),
            str(arms[0]["filter_name"]),
        )
        for pair_key, arms in expected_pairs.items()
        if pair_key in completed_pairs
        and all(
            measurement_pair_key(
                float(arms[0]["target_recall"]),
                int(arms[0]["clients"]),
                str(arms[0]["filter_name"]),
                repeat,
            )
            in completed_pairs
            for repeat in range(measurement_repeats)
        )
    }
    if not completed_cells <= expected_completed_cells:
        raise BenchmarkContractError(
            "checkpoint marks a cell complete before all of its repeat pairs commit"
        )
    aggregate_rows = [
        row for row in summary_rows if row.get("summary_type") == "aggregate"
    ]
    if len(summary_rows) != len(repeat_rows) + len(aggregate_rows):
        raise BenchmarkContractError("checkpoint summary contains an unknown row type")
    aggregate_cells = [
        measurement_cell_key(
            float(row["target_recall"]),
            int(row["clients"]),
            str(row["filter_name"]),
        )
        for row in aggregate_rows
    ]
    if (
        len(aggregate_rows) != len(completed_cells) * len(METHODS)
        or set(aggregate_cells) != completed_cells
        or any(aggregate_cells.count(cell) != len(METHODS) for cell in completed_cells)
    ):
        raise BenchmarkContractError(
            "checkpoint aggregate summaries do not exactly match completed cells"
        )
    cell_coverage = _require_mapping(
        checkpoint.get("cell_coverage_evidence"),
        "checkpoint cell_coverage_evidence",
    )
    if set(cell_coverage) != completed_cells or any(
        _require_mapping(value, f"cell coverage {key}").get("passed") is not True
        for key, value in cell_coverage.items()
    ):
        raise BenchmarkContractError(
            "checkpoint cell coverage evidence does not match completed cells"
        )
    return {
        "passed": True,
        "committed_pairs": len(completed_pairs),
        "committed_arms": len(expected_arm_keys),
        "completed_cells": len(completed_cells),
        "half_pairs": 0,
        "commit_unit": "cell_repeat_method_pair",
    }


def execute_search(
    args: argparse.Namespace,
    runtime: Any,
    filter_spec: FilterSpec,
    query_no: int,
    query_id: int,
    truth_entry: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    activation_finished = started
    ids: list[int] = []
    distances: list[float] = []
    error = ""
    error_type = ""
    try:
        activation_profile = activate(
            runtime.cur,
            args,
            runtime.mode,
            filter_spec.name,
            read_profile=False,
        )
        activation_finished = time.perf_counter()
        table = str(activation_profile["table"])
        binding = activation_binding(args, runtime.mode, filter_spec.name, activation_profile)
        self_exclusion = candidate_self_exclusion(args, table)
        ids, distances, _ = run_query(
            runtime.cur,
            table,
            filter_spec.predicate,
            query_id,
            args.k,
            binding,
            uses_exact_predicate_scan_contract(args.guidance_filter_strategy) and self_exclusion,
            candidate_validity_predicate=args.candidate_validity_predicate,
            query_table=query_table_for_candidate(args, table),
            query_id_column=args.query_id_column,
            query_vector_column=args.query_vector_column,
            self_exclusion=self_exclusion,
            reset_profile=False,
            read_profile=False,
        )
    except Exception as exc:  # noqa: BLE001 - errors are counted per request
        error_type = exc.__class__.__name__
        error = f"{error_type}: {exc}"
        try:
            recover_runtime(args, runtime)
        except Exception as recovery_exc:  # noqa: BLE001
            error += f"; recovery={recovery_exc.__class__.__name__}: {recovery_exc}"
    finished = time.perf_counter()
    recall = tie_aware_recall(distances, truth_entry, args.k) if not error else 0.0
    return {
        "query_no": query_no,
        "query_id": query_id,
        "latency_ms": (finished - started) * 1000.0,
        "activation_ms": (activation_finished - started) * 1000.0,
        "query_ms": (finished - activation_finished) * 1000.0,
        "returned": len(ids),
        "returned_ids": ",".join(str(value) for value in ids),
        "recall_at_10": recall,
        "error_type": error_type,
        "error": error,
    }


def configure_args_for_runtime(args: argparse.Namespace, method: str, config: SearchConfig) -> str:
    mode = MODE_BY_METHOD[method]
    args.ef_search = config.ef_search
    args.guided_collect_target = config.guided_collect_target
    args.traversal_guided_target = config.traversal_guided_target
    args.max_scan_tuples = config.max_scan_tuples
    args.scan_mem_multiplier = config.scan_mem_multiplier
    args.iterative_scan = config.iterative_scan
    prioritization = (
        method == "sqlens_d1" and args.guidance_filter_strategy == "traversal_guided"
    )
    args.traversal_guided_prioritization = prioritization
    args.traversal_guided_burst = config.traversal_guided_burst
    args.mode_configs_json = {
        mode: {
            **config.as_mode_config(),
            "traversal_guided_prioritization": prioritization,
            "traversal_guided_burst": config.traversal_guided_burst,
        }
    }
    return mode


def runtime_canary(
    args: argparse.Namespace,
    runtime: Any,
    method: str,
    filter_spec: FilterSpec,
    query_no: int,
    query_id: int,
    truth_entry: Any,
) -> dict[str, Any]:
    """Run one unmeasured profiled request after warmup and fail closed on r11 semantics."""
    expected_prioritization = (
        "on"
        if method == "sqlens_d1" and args.guidance_filter_strategy == "traversal_guided"
        else "off"
    )
    try:
        runtime.cur.execute(
            "SELECT current_setting('hnsw.filter_strategy'), "
            "current_setting('hnsw.traversal_guided_prioritization'), "
            "current_setting('hnsw.traversal_guided_burst')"
        )
        observed_strategy, observed_prioritization, observed_burst = (
            str(value) for value in runtime.cur.fetchone()
        )
        if observed_strategy != args.guidance_filter_strategy:
            raise BenchmarkContractError(
                f"canary strategy mismatch for {method}: strategy={observed_strategy!r}, "
                f"expected={args.guidance_filter_strategy!r}"
            )
        if observed_prioritization != expected_prioritization:
            raise BenchmarkContractError(
                f"canary GUC mismatch for {method}: prioritization="
                f"{observed_prioritization!r}, expected={expected_prioritization!r}"
            )
        if observed_burst != str(args.traversal_guided_burst):
            raise BenchmarkContractError(
                f"canary GUC mismatch for {method}: burst={observed_burst!r}, "
                f"expected={args.traversal_guided_burst!r}"
            )
        activation_profile = activate(
            runtime.cur, args, runtime.mode, filter_spec.name, read_profile=False
        )
        table = str(activation_profile["table"])
        binding = activation_binding(args, runtime.mode, filter_spec.name, activation_profile)
        self_exclusion = candidate_self_exclusion(args, table)
        ids, distances, profile = run_query(
            runtime.cur,
            table,
            filter_spec.predicate,
            query_id,
            args.k,
            binding,
            uses_exact_predicate_scan_contract(args.guidance_filter_strategy) and self_exclusion,
            candidate_validity_predicate=args.candidate_validity_predicate,
            query_table=query_table_for_candidate(args, table),
            query_id_column=args.query_id_column,
            query_vector_column=args.query_vector_column,
            self_exclusion=self_exclusion,
            reset_profile=True,
            read_profile=True,
        )
        recall = tie_aware_recall(distances, truth_entry, args.k)
        if len(ids) != args.k or recall < 0.0:
            raise BenchmarkContractError("canary request did not return a valid top-k result")
        final_path = str(profile.get("final_path", ""))
        if method == "sqlens_d1" and args.guidance_filter_strategy == "traversal_guided":
            required = {
                "final_path": "approximate_traversal_prioritization",
                "planner_proof_attempted": True,
                "planner_proof_succeeded": True,
            }
            if any(profile.get(key) != value for key, value in required.items()):
                raise BenchmarkContractError(
                    "SQLens D1 traversal-prioritization canary failed: "
                    + json.dumps({key: profile.get(key) for key in required}, sort_keys=True)
                )
            if profile.get("approximate_ann_path") is not True or profile.get(
                "approximate_prioritization_attempted"
            ) is not True:
                raise BenchmarkContractError(
                    "SQLens D1 canary did not prove the approximate dual-frontier path"
                )
            priority_reorders = int(profile.get("priority_reorders", 0) or 0)
            if (profile.get("traversal_order_changed") is True) != (
                priority_reorders > 0
            ):
                raise BenchmarkContractError(
                    "SQLens D1 canary order flag disagrees with priority_reorders"
                )
            if int(profile.get("match_frontier_pops", 0) or 0) + int(
                profile.get("no_bridge_frontier_pops", 0) or 0
            ) <= 0:
                raise BenchmarkContractError(
                    "SQLens D1 canary reported no dual-frontier pops"
                )
            if int(profile.get("traversal_prioritization_burst", 0) or 0) != int(
                args.traversal_guided_burst
            ):
                raise BenchmarkContractError(
                    "SQLens D1 canary profile burst does not match the session GUC"
                )
            if profile.get("graph_expansion_pruned") is not False:
                raise BenchmarkContractError(
                    "traversal_guided canary claimed graph-expansion pruning"
                )
            if profile.get("distance_computations_pruned") is not False:
                raise BenchmarkContractError(
                    "traversal_guided canary claimed distance-computation pruning"
                )
            if int(profile.get("stock_bypass_requests", 0) or 0) or int(profile.get("fallback_requests", 0) or 0):
                raise BenchmarkContractError("SQLens D1 canary used a stock bypass/fallback")
        elif method == "sqlens_d1":
            if final_path not in {None, "", "validation_only"}:
                raise BenchmarkContractError(
                    f"safe_guided canary final_path={final_path!r}, expected validation-only"
                )
            if profile.get("graph_expansion_pruned") is not False:
                raise BenchmarkContractError(
                    "safe_guided canary claimed graph-expansion pruning"
                )
            if profile.get("distance_computations_pruned") is not False:
                raise BenchmarkContractError(
                    "safe_guided canary claimed distance-computation pruning"
                )
            if profile.get("traversal_order_changed") is True or int(
                profile.get("priority_reorders", 0) or 0
            ) != 0:
                raise BenchmarkContractError(
                    "safe_guided canary unexpectedly changed HNSW traversal order"
                )
            if int(profile.get("guidance_checks", 0) or 0) <= 0:
                raise BenchmarkContractError(
                    "safe_guided canary did not perform candidate-admission validation"
                )
        elif final_path != "stock":
            raise BenchmarkContractError(
                f"stock canary final_path={final_path!r}, expected 'stock'"
            )
        return {
            "passed": True,
            "backend_pid": int(runtime.backend_cpu_provenance["backend_pid"]),
            "method": method,
            "mode": runtime.mode,
            "filter_name": filter_spec.name,
            "query_no": query_no,
            "query_id": query_id,
            "gucs": {
                "hnsw.filter_strategy": observed_strategy,
                "hnsw.traversal_guided_prioritization": observed_prioritization,
                "hnsw.traversal_guided_burst": observed_burst,
            },
            "low_skip_bypass_expected": False,
            "low_skip_bypass_observed": bool(
                int(profile.get("stock_bypass_requests", 0) or 0)
                or int(profile.get("fallback_requests", 0) or 0)
            ),
            "profile": profile,
        }
    finally:
        runtime.cur.execute("SELECT vector_hnsw_guidance_reset()")


def validate_plan_evidence(
    plan_rows: Sequence[Mapping[str, Any]],
    runtimes: Sequence[Any],
    mode: str,
    filter_spec: FilterSpec,
    expected_index: str,
) -> dict[str, Any]:
    expected_pids = {int(runtime.backend_cpu_provenance["backend_pid"]) for runtime in runtimes}
    matched = [
        row for row in plan_rows
        if row.get("mode") == mode
        and row.get("filter_name") == filter_spec.name
        and row.get("expected_index") == expected_index
        and int((row.get("backend_cpu_provenance") or {}).get("backend_pid", -1)) in expected_pids
    ]
    observed_pids = {
        int((row.get("backend_cpu_provenance") or {}).get("backend_pid", -1)) for row in matched
    }
    passed = (
        len(matched) == len(expected_pids)
        and observed_pids == expected_pids
        and all(row.get("passed") is True for row in matched)
    )
    evidence = {
        "passed": passed,
        "mode": mode,
        "filter_name": filter_spec.name,
        "expected_index": expected_index,
        "expected_backend_pids": sorted(expected_pids),
        "observed_backend_pids": sorted(observed_pids),
        "expected_count": len(expected_pids),
        "observed_count": len(matched),
        "checks": matched,
    }
    if not passed:
        raise BenchmarkContractError(
            f"per-client EXPLAIN evidence is incomplete for mode={mode} filter={filter_spec.name} "
            f"index={expected_index}: expected={sorted(expected_pids)} observed={sorted(observed_pids)}"
        )
    return evidence


def pin_current_thread(client_cpu_set: tuple[int, ...], client_id: int) -> dict[str, Any]:
    native_id = threading.get_native_id()
    if not client_cpu_set:
        return {"native_tid": native_id, "requested_cpu": "", "affinity_applied": False}
    cpu = client_cpu_set[client_id % len(client_cpu_set)]
    os.sched_setaffinity(native_id, {cpu})
    observed = sorted(os.sched_getaffinity(native_id))
    if observed != [cpu]:
        raise BenchmarkContractError(
            f"client thread affinity mismatch: tid={native_id} requested={cpu} observed={observed}"
        )
    return {"native_tid": native_id, "requested_cpu": cpu, "affinity_applied": True}


def validate_independent_backends(runtimes: Sequence[Any], clients: int) -> list[int]:
    pids = [int(runtime.backend_cpu_provenance["backend_pid"]) for runtime in runtimes]
    if len(pids) != clients or len(set(pids)) != clients:
        raise BenchmarkContractError(
            f"each client must own one independent PostgreSQL connection/backend: {pids}"
        )
    return pids


def measurement_dispatch(
    workload: Workload,
    schedule_seed: int,
    target: float,
    clients: int,
    filter_name: str,
    measurement_repeat: int,
) -> tuple[int, list[tuple[int, WorkloadRequest]]]:
    """Return a method-independent seeded request permutation for one paired trial."""
    seed = stable_seed(
        schedule_seed,
        "request_trace",
        f"{target:.6f}",
        clients,
        filter_name,
        measurement_repeat,
    )
    requests = list(workload.requests)
    random.Random(seed).shuffle(requests)
    return seed, list(enumerate(requests))


def validate_measurement_arm_rows(
    rows: Sequence[Mapping[str, Any]],
    workload: Workload,
    clients: int,
    measurement_repeat: int,
) -> None:
    if clients <= 0:
        raise BenchmarkContractError("measurement arm requires at least one client")
    expected_requests = {request.request_no: request for request in workload.requests}
    if (
        len(expected_requests) != len(workload.requests)
        or set(expected_requests) != set(range(len(workload.requests)))
    ):
        raise BenchmarkContractError("bound workload request_no coverage is not exact")
    observed_request_nos = [int(row["request_no"]) for row in rows]
    if (
        len(rows) != len(expected_requests)
        or len(observed_request_nos) != len(set(observed_request_nos))
        or set(observed_request_nos) != set(expected_requests)
    ):
        raise BenchmarkContractError("measurement arm request_no coverage is not exact")
    dispatch_positions = [int(row["dispatch_position"]) for row in rows]
    if (
        len(dispatch_positions) != len(set(dispatch_positions))
        or set(dispatch_positions) != set(range(len(expected_requests)))
    ):
        raise BenchmarkContractError("measurement arm dispatch-position coverage is not exact")
    expected_client_counts = Counter(position % clients for position in range(len(expected_requests)))
    observed_client_counts = Counter(int(row["client_id"]) for row in rows)
    if observed_client_counts != expected_client_counts:
        raise BenchmarkContractError(
            "measurement arm client coverage is not exact: "
            f"expected={dict(expected_client_counts)} observed={dict(observed_client_counts)}"
        )
    trace_seeds = {int(row["trace_permutation_seed"]) for row in rows}
    if len(trace_seeds) != 1:
        raise BenchmarkContractError("measurement arm must bind exactly one trace permutation seed")
    for row in rows:
        request = expected_requests[int(row["request_no"])]
        if (
            int(row["query_no"]) != request.query_no
            or int(row["query_id"]) != request.query_id
            or int(row["trace_cycle"]) != request.trace_cycle
        ):
            raise BenchmarkContractError("measurement row does not match the bound workload request")
        if int(row["measurement_repeat"]) != measurement_repeat:
            raise BenchmarkContractError("measurement row repeat does not match its arm")
        if int(row["clients"]) != clients:
            raise BenchmarkContractError("measurement row client-count identity does not match its arm")
        dispatch_position = int(row["dispatch_position"])
        client_id = int(row["client_id"])
        if not 0 <= client_id < clients or client_id != dispatch_position % clients:
            raise BenchmarkContractError("measurement row violates deterministic client dispatch")


def validate_measurement_arm_timing(
    rows: Sequence[Mapping[str, Any]], wall_seconds: float
) -> None:
    """Reject malformed request timing before it can enter QPS or tail metrics."""
    if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise BenchmarkContractError("measurement arm wall-clock duration must be finite and positive")
    wall_ms = wall_seconds * 1000.0
    latest_completion = 0.0
    for row in rows:
        try:
            latency = float(row["latency_ms"])
            activation = float(row["activation_ms"])
            query = float(row["query_ms"])
            started = float(row["started_offset_ms"])
            completed = float(row["completed_offset_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkContractError("measurement row has incomplete timing fields") from exc
        if not all(math.isfinite(value) and value >= 0.0 for value in (
            latency, activation, query, started, completed,
        )):
            raise BenchmarkContractError("measurement row has non-finite or negative timing")
        if completed < started or completed + 1e-6 < started + latency:
            raise BenchmarkContractError("measurement row timing is internally inconsistent")
        # The per-request latency is execute_search() time; its interval must
        # finish before the arm-level wall-clock interval closes.
        latest_completion = max(latest_completion, completed)
    if latest_completion > wall_ms + 1.0:
        raise BenchmarkContractError(
            "measurement request completion exceeds the recorded arm wall-clock interval"
        )


def validate_measurement_cell_rows(
    rows: Sequence[Mapping[str, Any]],
    workload: Workload,
    target: float,
    clients: int,
    filter_name: str,
    measurement_repeats: int,
) -> dict[str, Any]:
    """Validate paired methods, repeats, request orders, and client assignments."""
    expected_arms = {
        (method, repeat)
        for method in METHODS
        for repeat in range(measurement_repeats)
    }
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        identity = (str(row["method"]), int(row["measurement_repeat"]))
        if (
            float(row["target_recall"]) != target
            or int(row["clients"]) != clients
            or str(row["filter_name"]) != filter_name
        ):
            raise BenchmarkContractError("measurement cell contains a row from another cell")
        grouped[identity].append(row)
    if set(grouped) != expected_arms:
        raise BenchmarkContractError(
            "measurement cell method/repeat coverage is not exact: "
            f"missing={sorted(expected_arms - set(grouped))} "
            f"unexpected={sorted(set(grouped) - expected_arms)}"
        )

    repeat_seeds: dict[int, int] = {}
    repeat_orders: dict[int, dict[str, tuple[int, ...]]] = defaultdict(dict)
    for (method, repeat), arm_rows in grouped.items():
        validate_measurement_arm_rows(arm_rows, workload, clients, repeat)
        seeds = {int(row["trace_permutation_seed"]) for row in arm_rows}
        seed = next(iter(seeds))
        previous_seed = repeat_seeds.setdefault(repeat, seed)
        if previous_seed != seed:
            raise BenchmarkContractError(
                f"paired methods use different request-order seeds in repeat {repeat}"
            )
        ordered = tuple(
            int(row["request_no"])
            for row in sorted(arm_rows, key=lambda value: int(value["dispatch_position"]))
        )
        repeat_orders[repeat][method] = ordered

    if len(set(repeat_seeds.values())) != measurement_repeats:
        raise BenchmarkContractError("measurement repeats do not use distinct request-order seeds")
    for repeat, orders in repeat_orders.items():
        if len(set(orders.values())) != 1:
            raise BenchmarkContractError(
                f"paired methods do not replay the same request order in repeat {repeat}"
            )
    return {
        "passed": True,
        "workload_requests_per_arm": len(workload.requests),
        "unique_query_vectors_per_arm": workload.unique_query_vectors,
        "measurement_repeats": measurement_repeats,
        "methods": list(METHODS),
        "clients": clients,
        "repeat_order_seeds": [repeat_seeds[repeat] for repeat in range(measurement_repeats)],
        "independent_order_seed_per_repeat": True,
        "paired_identical_order_across_methods": True,
        "no_duplicate_or_missing_requests": True,
    }


def run_measurement_arm(
    args: argparse.Namespace,
    target: float,
    clients: int,
    method: str,
    config: SearchConfig,
    filter_spec: FilterSpec,
    workload: Workload,
    truth: Mapping[tuple[str, int], Any],
    arm_order: int,
    measurement_repeat: int,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    mode = configure_args_for_runtime(args, method, config)
    args.plan_query_id = workload.requests[0].query_id
    trace_permutation_seed, dispatch = measurement_dispatch(
        workload,
        args.schedule_seed,
        target,
        clients,
        filter_spec.name,
        measurement_repeat,
    )
    runtimes: list[Any] = []
    plan_start = len(args.plan_evidence)
    try:
        for _client_id in range(clients):
            runtimes.append(
                open_mode_runtime(
                    args,
                    mode,
                    [(filter_spec.name, filter_spec.actual_pct, filter_spec.predicate)],
                )
            )
        backend_pids = validate_independent_backends(runtimes, clients)
        _, expected_index = mode_table_index(args, mode)
        plan_gate = validate_plan_evidence(
            args.plan_evidence[plan_start:], runtimes, mode, filter_spec, expected_index
        )

        warmup_requests = list(workload.requests[: args.session_warmup_requests])
        for runtime in runtimes:
            for request in warmup_requests:
                warm = execute_search(
                    args,
                    runtime,
                    filter_spec,
                    request.query_no,
                    request.query_id,
                    truth[(filter_spec.name, request.query_no)],
                )
                if warm["error"]:
                    raise BenchmarkContractError(
                        f"session warm-cache gate failed for {filter_spec.name}/{method}: {warm['error']}"
                    )
        canary_request = workload.requests[0]
        canary_evidence = [
            runtime_canary(
                args,
                runtime,
                method,
                filter_spec,
                canary_request.query_no,
                canary_request.query_id,
                truth[(filter_spec.name, canary_request.query_no)],
            )
            for runtime in runtimes
        ]

        barrier = threading.Barrier(clients + 1)
        client_cpu_set = parse_cpu_set(args.client_cpu_list)
        arm_started = 0.0

        def worker(client_id: int, runtime: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            try:
                affinity = pin_current_thread(client_cpu_set, client_id)
                barrier.wait()
            except BaseException:
                barrier.abort()
                raise
            client_rows: list[dict[str, Any]] = []
            for dispatch_position, request in dispatch[client_id::clients]:
                started_offset = (time.perf_counter() - arm_started) * 1000.0
                result = execute_search(
                    args,
                    runtime,
                    filter_spec,
                    request.query_no,
                    request.query_id,
                    truth[(filter_spec.name, request.query_no)],
                )
                completed_offset = (time.perf_counter() - arm_started) * 1000.0
                client_rows.append(
                    {
                        "phase": "measurement",
                        "runner_version": RUNNER_VERSION,
                        "evaluation_scope": args.evaluation_scope,
                        "target_recall": target,
                        "filter_name": filter_spec.name,
                        "query_cohort": workload.query_cohort,
                        "workload_source_kind": workload.source_kind,
                        "trace_replay": workload.trace_replay,
                        "workload_requests": len(workload.requests),
                        "unique_query_vectors": workload.unique_query_vectors,
                        "method": method,
                        "mode": mode,
                        "guidance_filter_strategy": args.guidance_filter_strategy,
                        "clients": clients,
                        "arm_order": arm_order,
                        "measurement_repeat": measurement_repeat,
                        "dispatch_position": dispatch_position,
                        "trace_permutation_seed": trace_permutation_seed,
                        "request_no": request.request_no,
                        "trace_cycle": request.trace_cycle,
                        "client_id": client_id,
                        "backend_pid": backend_pids[client_id],
                        "client_native_tid": affinity["native_tid"],
                        "client_requested_cpu": affinity["requested_cpu"],
                        "client_affinity_applied": affinity["affinity_applied"],
                        "started_offset_ms": started_offset,
                        "completed_offset_ms": completed_offset,
                        "config": config.label,
                        "ef_search": config.ef_search,
                        "guided_collect_target": config.guided_collect_target,
                        "max_scan_tuples": config.max_scan_tuples,
                        "scan_mem_multiplier": config.scan_mem_multiplier,
                        "iterative_scan": config.iterative_scan,
                        "traversal_guided_prioritization": config.traversal_guided_prioritization,
                        "traversal_guided_burst": config.traversal_guided_burst,
                        "traversal_guided_target": config.traversal_guided_target,
                        **result,
                    }
                )
            return client_rows, affinity

        affinity_evidence: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        flush_postgres_stats(runtimes)
        postgres_before = postgres_telemetry_snapshot(
            runtimes[0].cur,
            args.insertion_table,
            expected_index,
        )
        backend_cpu_before = backend_cpu_snapshot(
            backend_pids, proc_root=args.backend_proc_root
        )
        with ThreadPoolExecutor(max_workers=clients, thread_name_prefix="pgvector-throughput") as pool:
            futures = [pool.submit(worker, client_id, runtime) for client_id, runtime in enumerate(runtimes)]
            host_before = host_telemetry_snapshot(args.telemetry_devices_resolved)
            arm_started = time.perf_counter()
            try:
                barrier.wait(timeout=args.start_barrier_timeout_seconds)
            except threading.BrokenBarrierError as exc:
                raise BenchmarkContractError("client start barrier failed") from exc
            for future in as_completed(futures):
                client_rows, affinity = future.result()
                rows.extend(client_rows)
                affinity_evidence.append(affinity)
        wall_seconds = time.perf_counter() - arm_started
        host_after = host_telemetry_snapshot(args.telemetry_devices_resolved)
        flush_postgres_stats(runtimes)
        postgres_after = postgres_telemetry_snapshot(
            runtimes[0].cur,
            args.insertion_table,
            expected_index,
        )
        backend_cpu_after = backend_cpu_snapshot(
            backend_pids, proc_root=args.backend_proc_root
        )
        rows.sort(key=lambda row: int(row["dispatch_position"]))
        validate_measurement_arm_rows(rows, workload, clients, measurement_repeat)
        validate_measurement_arm_timing(rows, wall_seconds)
        telemetry = {
            "host": host_telemetry_delta(host_before, host_after),
            "postgresql": postgres_telemetry_delta(postgres_before, postgres_after),
            "backend_cpu": backend_cpu_delta(backend_cpu_before, backend_cpu_after),
            "backend_proc_root": str(args.backend_proc_root),
            "devices": list(args.telemetry_devices_resolved),
            "postgres_stats_force_flushed_per_backend": True,
            "measurement_wall_clock_seconds": wall_seconds,
        }
        evidence = {
            "backend_pids": backend_pids,
            "independent_connections": len(set(backend_pids)) == clients,
            "session_warmup_requests_per_client": len(warmup_requests),
            "client_affinity": sorted(affinity_evidence, key=lambda row: int(row["native_tid"])),
            "plan_evidence": plan_gate,
            "canary_evidence": canary_evidence,
            "measurement_repeat": measurement_repeat,
            "trace_permutation_seed": trace_permutation_seed,
            "trace_order_sha256": canonical_sha256(
                [request.request_no for _, request in dispatch]
            ),
            "trial_protocol": "paired_method_arm_with_repeat_specific_request_order_seed",
            "cache_protocol": {
                "classification": "warm_cache",
                "session_warmup_requests_per_client": len(warmup_requests),
                "reset_before_arm": False,
                "shared_cache_state_retained": True,
            },
            "telemetry": telemetry,
        }
        return rows, wall_seconds, evidence
    finally:
        for runtime in reversed(runtimes):
            close_mode_runtime(runtime)


def telemetry_summary_fields(telemetry: Mapping[str, Any] | None) -> dict[str, Any]:
    if not telemetry:
        return {
            "telemetry_collected": False,
            "telemetry_json": "",
        }
    host = _require_mapping(telemetry.get("host"), "host arm telemetry")
    cpu = _require_mapping(host.get("cpu"), "host CPU arm telemetry")
    disk = _require_mapping(host.get("disk_total"), "host disk arm telemetry")
    postgres = _require_mapping(telemetry.get("postgresql"), "PostgreSQL arm telemetry")
    database = _require_mapping(
        postgres.get("database"), "pg_stat_database arm telemetry"
    )
    io_total = _require_mapping(postgres.get("io_total"), "pg_stat_io arm telemetry")
    relations = _require_mapping(postgres.get("relations"), "relation stats arm telemetry")
    relation_table = _require_mapping(relations.get("table"), "target table arm telemetry")
    relation_index = _require_mapping(relations.get("index"), "target index arm telemetry")
    backend_cpu = _require_mapping(telemetry.get("backend_cpu"), "backend CPU arm telemetry")
    backend_cpu_total = _require_mapping(backend_cpu.get("total"), "backend CPU total")
    return {
        "telemetry_collected": True,
        "telemetry_devices_json": json.dumps(
            list(telemetry.get("devices") or []), sort_keys=True
        ),
        "host_cpu_utilization_pct": cpu["utilization_pct"],
        "host_cpu_user_pct": cpu["user_pct"],
        "host_cpu_system_pct": cpu["system_pct"],
        "host_cpu_iowait_pct": cpu["iowait_pct"],
        "host_disk_reads_completed": disk["reads_completed"],
        "host_disk_read_bytes": disk["read_bytes"],
        "host_disk_read_time_ms": disk["read_time_ms"],
        "host_disk_writes_completed": disk["writes_completed"],
        "host_disk_write_bytes": disk["write_bytes"],
        "host_disk_write_time_ms": disk["write_time_ms"],
        "host_disk_io_time_ms": disk["io_time_ms"],
        "host_disk_weighted_io_time_ms": disk["weighted_io_time_ms"],
        "pg_database_blks_read": database["blks_read"],
        "pg_database_blks_hit": database["blks_hit"],
        "pg_database_temp_files": database["temp_files"],
        "pg_database_temp_bytes": database["temp_bytes"],
        "pg_database_blk_read_time_ms": database["blk_read_time"],
        "pg_database_blk_write_time_ms": database["blk_write_time"],
        "pg_io_reads": io_total["reads"],
        "pg_io_read_bytes": io_total["read_bytes"],
        "pg_io_read_time_ms": io_total["read_time"],
        "pg_io_writes": io_total["writes"],
        "pg_io_write_bytes": io_total["write_bytes"],
        "pg_io_write_time_ms": io_total["write_time"],
        "pg_io_hits": io_total["hits"],
        "pg_io_evictions": io_total["evictions"],
        "pg_target_table_heap_blks_read": relation_table["heap_blks_read"],
        "pg_target_table_heap_blks_hit": relation_table["heap_blks_hit"],
        "pg_target_table_idx_blks_read": relation_table["idx_blks_read"],
        "pg_target_table_idx_blks_hit": relation_table["idx_blks_hit"],
        "pg_target_index_blks_read": relation_index["idx_blks_read"],
        "pg_target_index_blks_hit": relation_index["idx_blks_hit"],
        "pg_backend_cpu_processes": len(backend_cpu.get("backend_pids") or []),
        "pg_backend_cpu_user_ms": backend_cpu_total["user_cpu_ms"],
        "pg_backend_cpu_system_ms": backend_cpu_total["system_cpu_ms"],
        "pg_backend_cpu_total_ms": backend_cpu_total["total_cpu_ms"],
        "telemetry_json": json.dumps(telemetry, sort_keys=True),
    }


def aggregate_telemetry_summary(
    repeat_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not repeat_summaries or not all(
        _is_explicit_true(row.get("telemetry_collected")) for row in repeat_summaries
    ):
        return telemetry_summary_fields(None)
    telemetry_rows = [
        _json_value(row["telemetry_json"], "repeat telemetry")
        for row in repeat_summaries
    ]
    walls = [float(row["wall_clock_seconds"]) for row in repeat_summaries]
    wall_total = sum(walls)
    cpu_fields = (
        "host_cpu_utilization_pct",
        "host_cpu_user_pct",
        "host_cpu_system_pct",
        "host_cpu_iowait_pct",
    )
    summed_fields = (
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
        "pg_backend_cpu_user_ms",
        "pg_backend_cpu_system_ms",
        "pg_backend_cpu_total_ms",
    )
    devices = sorted(
        {
            device
            for telemetry in telemetry_rows
            for device in telemetry.get("devices", [])
        }
    )
    result: dict[str, Any] = {
        "telemetry_collected": True,
        "telemetry_devices_json": json.dumps(devices),
        "telemetry_json": json.dumps(
            {
                "aggregation": (
                    "counter deltas summed across repeats; CPU percentages weighted "
                    "by repeat measurement wall-clock"
                ),
                "repeats": telemetry_rows,
            },
            sort_keys=True,
        ),
    }
    for field in cpu_fields:
        result[field] = (
            sum(float(row[field]) * wall for row, wall in zip(repeat_summaries, walls))
            / wall_total
            if wall_total > 0
            else 0.0
        )
    for field in summed_fields:
        result[field] = sum(float(row[field]) for row in repeat_summaries)
    result["pg_backend_cpu_processes"] = max(
        int(float(row["pg_backend_cpu_processes"])) for row in repeat_summaries
    )
    return result


def summarize_arm(
    rows: Sequence[Mapping[str, Any]],
    wall_seconds: float,
    target: float,
    clients: int,
    method: str,
    config: SearchConfig,
    filter_spec: FilterSpec,
    workload: Workload,
    matched_recall_evidence: Mapping[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
    measurement_repeat: int = 0,
    telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_measurement_arm_rows(rows, workload, clients, measurement_repeat)
    validate_measurement_arm_timing(rows, wall_seconds)
    ok = [row for row in rows if not row.get("error")]
    latencies = [float(row["latency_ms"]) for row in ok]
    by_query_latency: dict[int, list[float]] = defaultdict(list)
    by_query_recall: dict[int, list[float]] = defaultdict(list)
    recalls: list[float] = []
    for row in rows:
        query_no = int(row["query_no"])
        if not row.get("error"):
            by_query_latency[query_no].append(float(row["latency_ms"]))
        recall = 0.0 if row.get("error") else float(row["recall_at_10"])
        recalls.append(recall)
        by_query_recall[query_no].append(recall)
    query_latency_means = [statistics.fmean(values) for values in by_query_latency.values()]
    query_recall_means = [statistics.fmean(values) for values in by_query_recall.values()]
    latency_low, latency_high = bootstrap_mean_ci(
        query_latency_means,
        bootstrap_samples,
        stable_seed(
            bootstrap_seed,
            "latency",
            method,
            clients,
            filter_spec.name,
            target,
            measurement_repeat,
        ),
    )
    recall_low, recall_high = bootstrap_mean_ci(
        query_recall_means,
        bootstrap_samples,
        stable_seed(
            bootstrap_seed,
            "recall",
            method,
            clients,
            filter_spec.name,
            target,
            measurement_repeat,
        ),
    )
    errors = Counter(str(row.get("error_type") or "unknown") for row in rows if row.get("error"))
    completed = len(ok)
    recall_mean = statistics.fmean(recalls) if recalls else 0.0
    trace_permutation_seed = next(
        iter({int(row["trace_permutation_seed"]) for row in rows})
    )
    mean_target_met = recall_mean >= target
    lcb95_target_met = recall_low >= target
    valid = (
        len(rows) == len(workload.requests)
        and completed == len(workload.requests)
        and not errors
        and len(by_query_recall) == workload.unique_query_vectors
        and mean_target_met
        and lcb95_target_met
    )
    return {
        "summary_type": "repeat",
        "measurement_repeat": measurement_repeat,
        "trace_permutation_seed": trace_permutation_seed,
        "dataset": "Amazon-10M",
        "evaluation_scope": str(rows[0]["evaluation_scope"]),
        "status": "valid" if valid else "invalid",
        "target_recall": target,
        "target_met_measurement": mean_target_met,
        "target_lcb95_met_measurement": lcb95_target_met,
        "filter_name": filter_spec.name,
        "filter_candidate_rows": filter_spec.expected_rows,
        "query_cohort": workload.query_cohort,
        "workload_source_kind": workload.source_kind,
        "trace_replay": workload.trace_replay,
        "workload_requests": len(workload.requests),
        "unique_query_vectors": workload.unique_query_vectors,
        "method": method,
        "mode": MODE_BY_METHOD[method],
        "guidance_filter_strategy": str(rows[0]["guidance_filter_strategy"]),
        "clients": clients,
        "completed_queries": completed,
        "error_count": len(rows) - completed,
        "error_counts_json": json.dumps(dict(sorted(errors.items())), sort_keys=True),
        "wall_clock_seconds": wall_seconds,
        "throughput_definition": "completed_queries / measurement_wall_clock_seconds",
        "throughput_qps": completed / wall_seconds if wall_seconds > 0 else 0.0,
        "throughput_qps_ci95_low": "",
        "throughput_qps_ci95_high": "",
        "throughput_bootstrap_bins": 0,
        "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "latency_query_cluster_ci95_low_ms": latency_low,
        "latency_query_cluster_ci95_high_ms": latency_high,
        "recall_mean": recall_mean,
        "recall_query_cluster_ci95_low": recall_low,
        "recall_query_cluster_ci95_high": recall_high,
        "recall_lcb95_definition": (
            "2.5th percentile of a seeded nonparametric bootstrap over unique "
            "query-vector clusters; each cluster is its mean Recall@10 across "
            "occurrences and failed requests score zero"
        ),
        "bootstrap_samples": bootstrap_samples,
        "latency_recall_bootstrap_unit": "unique_query_vector_cluster",
        "latency_sample_scope": "successful_requests_only",
        "recall_sample_scope": "all_requests_with_failed_requests_scored_zero",
        "throughput_bootstrap_unit": "not_applicable_single_repeat",
        "config": config.label,
        "ef_search": config.ef_search,
        "guided_collect_target": config.guided_collect_target,
        "max_scan_tuples": config.max_scan_tuples,
        "scan_mem_multiplier": config.scan_mem_multiplier,
        "iterative_scan": config.iterative_scan,
        "traversal_guided_prioritization": config.traversal_guided_prioritization,
        "traversal_guided_burst": config.traversal_guided_burst,
        "traversal_guided_target": config.traversal_guided_target,
        "matched_recall_calibration_mean": matched_recall_evidence["recall_mean"],
        "matched_recall_calibration_lcb95": matched_recall_evidence["recall_lcb95"],
        "matched_recall_calibration_latency_ms": matched_recall_evidence["latency_mean_ms"],
        "matched_recall_calibration_samples": matched_recall_evidence["samples"],
        "config_source": matched_recall_evidence["selection_source"],
        "measurement_queries_overlap_matched_recall_calibration": False,
        "warm_cache": True,
        "independent_connection_per_client": True,
        **telemetry_summary_fields(telemetry),
    }


def aggregate_measurement_cell(
    repeat_summaries: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(repeat_summaries) < FORMAL_MEASUREMENT_REPEATS:
        raise BenchmarkContractError("aggregate requires at least six measurement repeats")
    first = dict(repeat_summaries[0])
    repeats = sorted(int(row["measurement_repeat"]) for row in repeat_summaries)
    if repeats != list(range(len(repeat_summaries))) or len(repeats) != len(set(repeats)):
        raise BenchmarkContractError("repeat summaries do not have exact contiguous repeat coverage")
    identity_fields = (
        "evaluation_scope", "method", "guidance_filter_strategy", "clients", "filter_name", "target_recall", "workload_requests",
        "unique_query_vectors", "query_cohort", "workload_source_kind", "trace_replay",
        "config", "ef_search", "guided_collect_target", "max_scan_tuples",
        "scan_mem_multiplier", "iterative_scan", "traversal_guided_prioritization",
        "traversal_guided_burst", "traversal_guided_target", "config_source",
    )
    for field in identity_fields:
        if len({str(row[field]) for row in repeat_summaries}) != 1:
            raise BenchmarkContractError(f"repeat summary identity mismatch: {field}")
    expected_requests = int(first["workload_requests"])
    expected_unique_queries = int(first["unique_query_vectors"])
    clients = int(first["clients"])
    method = str(first["method"])
    filter_name = str(first["filter_name"])
    target = float(first["target_recall"])
    grouped_rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[int(row["measurement_repeat"])].append(row)
    if set(grouped_rows) != set(repeats):
        raise BenchmarkContractError("aggregate raw rows do not cover the expected repeats")
    if len(rows) != expected_requests * len(repeats):
        raise BenchmarkContractError("aggregate raw row count does not match request/repeat coverage")

    summaries_by_repeat = {
        int(summary["measurement_repeat"]): summary for summary in repeat_summaries
    }
    request_mapping: dict[int, tuple[int, int, int]] = {}
    query_frequency: Counter[int] | None = None
    repeat_seeds: dict[int, int] = {}
    repeat_completed: list[float] = []
    repeat_walls: list[float] = []
    repeat_qps: list[float] = []
    repeat_mean_target_gate: dict[int, bool] = {}
    repeat_lcb95_target_gate: dict[int, bool] = {}
    for repeat, repeat_rows in grouped_rows.items():
        request_nos = [int(row["request_no"]) for row in repeat_rows]
        dispatch_positions = [int(row["dispatch_position"]) for row in repeat_rows]
        if (
            len(repeat_rows) != expected_requests
            or len(request_nos) != len(set(request_nos))
            or set(request_nos) != set(range(expected_requests))
            or len(dispatch_positions) != len(set(dispatch_positions))
            or set(dispatch_positions) != set(range(expected_requests))
        ):
            raise BenchmarkContractError(f"aggregate raw coverage is incomplete for repeat {repeat}")
        observed_query_frequency = Counter(int(row["query_no"]) for row in repeat_rows)
        if len(observed_query_frequency) != expected_unique_queries:
            raise BenchmarkContractError(
                f"aggregate raw unique-query coverage is incomplete for repeat {repeat}"
            )
        if query_frequency is None:
            query_frequency = observed_query_frequency
        elif observed_query_frequency != query_frequency:
            raise BenchmarkContractError("query frequency coverage changed across repeats")
        expected_client_counts = Counter(position % clients for position in range(expected_requests))
        observed_client_counts = Counter(int(row["client_id"]) for row in repeat_rows)
        if observed_client_counts != expected_client_counts:
            raise BenchmarkContractError(
                f"aggregate raw client coverage is incomplete for repeat {repeat}"
            )
        seeds = {int(row["trace_permutation_seed"]) for row in repeat_rows}
        if len(seeds) != 1:
            raise BenchmarkContractError(
                f"aggregate raw rows do not bind one order seed for repeat {repeat}"
            )
        repeat_seeds[repeat] = next(iter(seeds))
        for row in repeat_rows:
            if (
                str(row["method"]) != method
                or int(row["clients"]) != clients
                or str(row["filter_name"]) != filter_name
                or float(row["target_recall"]) != target
            ):
                raise BenchmarkContractError("aggregate raw row identity does not match its cell")
            dispatch_position = int(row["dispatch_position"])
            if int(row["client_id"]) != dispatch_position % clients:
                raise BenchmarkContractError("aggregate raw row violates client dispatch")
            request_no = int(row["request_no"])
            mapping = (int(row["query_no"]), int(row["query_id"]), int(row["trace_cycle"]))
            previous = request_mapping.setdefault(request_no, mapping)
            if previous != mapping:
                raise BenchmarkContractError("workload request mapping changed across repeats")
        summary = summaries_by_repeat[repeat]
        if int(summary.get("trace_permutation_seed", -1)) != repeat_seeds[repeat]:
            raise BenchmarkContractError("repeat summary order seed does not match raw rows")
        validate_measurement_arm_timing(repeat_rows, float(summary["wall_clock_seconds"]))
        completed = sum(1 for row in repeat_rows if not row.get("error"))
        errors = len(repeat_rows) - completed
        if (
            int(summary["completed_queries"]) != completed
            or int(summary["error_count"]) != errors
        ):
            raise BenchmarkContractError("repeat summary completion counts do not match raw rows")
        wall_seconds = float(summary["wall_clock_seconds"])
        if wall_seconds <= 0.0:
            raise BenchmarkContractError("repeat summary wall-clock duration must be positive")
        repeat_completed.append(float(completed))
        repeat_walls.append(wall_seconds)
        repeat_qps.append(completed / wall_seconds)
        repeat_recalls = [
            0.0 if row.get("error") else float(row["recall_at_10"])
            for row in repeat_rows
        ]
        repeat_by_query: dict[int, list[float]] = defaultdict(list)
        for row, recall in zip(repeat_rows, repeat_recalls):
            repeat_by_query[int(row["query_no"])].append(recall)
        repeat_query_means = [
            statistics.fmean(values) for values in repeat_by_query.values()
        ]
        repeat_recall_low, _ = bootstrap_mean_ci(
            repeat_query_means,
            bootstrap_samples,
            stable_seed(
                bootstrap_seed,
                "recall",
                method,
                clients,
                filter_name,
                target,
                repeat,
            ),
        )
        repeat_mean_target_gate[repeat] = (
            errors == 0 and statistics.fmean(repeat_recalls) >= target
        )
        repeat_lcb95_target_gate[repeat] = errors == 0 and repeat_recall_low >= target
    if len(set(repeat_seeds.values())) != len(repeats):
        raise BenchmarkContractError("aggregate repeats do not use distinct request-order seeds")

    low, high = bootstrap_pooled_ratio_ci(
        repeat_completed,
        repeat_walls,
        bootstrap_samples,
        stable_seed(bootstrap_seed, "repeat_qps", method, clients, filter_name, target),
    )
    ok = [row for row in rows if not row.get("error")]
    latencies = [float(row["latency_ms"]) for row in ok]
    by_query_latency: dict[int, list[float]] = defaultdict(list)
    by_query_recall: dict[int, list[float]] = defaultdict(list)
    recalls: list[float] = []
    for row in rows:
        query_no = int(row["query_no"])
        if not row.get("error"):
            by_query_latency[query_no].append(float(row["latency_ms"]))
        recall = 0.0 if row.get("error") else float(row["recall_at_10"])
        recalls.append(recall)
        by_query_recall[query_no].append(recall)
    query_latency_means = [statistics.fmean(values) for values in by_query_latency.values()]
    query_recall_means = [statistics.fmean(values) for values in by_query_recall.values()]
    latency_low, latency_high = bootstrap_mean_ci(
        query_latency_means,
        bootstrap_samples,
        stable_seed(bootstrap_seed, "aggregate_latency", method, clients, filter_name, target),
    )
    recall_low, recall_high = bootstrap_mean_ci(
        query_recall_means,
        bootstrap_samples,
        stable_seed(bootstrap_seed, "aggregate_recall", method, clients, filter_name, target),
    )
    recall_mean = statistics.fmean(recalls) if recalls else 0.0
    errors = Counter(str(row.get("error_type") or "unknown") for row in rows if row.get("error"))
    p95_low, p95_high = cluster_bootstrap_percentile_ci(
        rows, 0.95, bootstrap_samples,
        stable_seed(bootstrap_seed, "p95", method, clients, filter_name, target),
    )
    p99_low, p99_high = cluster_bootstrap_percentile_ci(
        rows, 0.99, bootstrap_samples,
        stable_seed(bootstrap_seed, "p99", method, clients, filter_name, target),
    )
    first.update(
        {
            "summary_type": "aggregate",
            "measurement_repeat": "all",
            "measurement_repeats": len(repeat_summaries),
            "repeat_keys_json": json.dumps(repeats),
            "repeat_order_seeds_json": json.dumps(
                [repeat_seeds[repeat] for repeat in repeats]
            ),
            "completed_queries": len(ok),
            "error_count": len(rows) - len(ok),
            "wall_clock_seconds": sum(repeat_walls),
            "throughput_qps": (
                sum(repeat_completed) / sum(repeat_walls)
            ),
            "throughput_qps_repeat_mean": statistics.fmean(repeat_qps),
            "throughput_qps_ci95_low": low,
            "throughput_qps_ci95_high": high,
            "throughput_bootstrap_bins": 0,
            "throughput_bootstrap_unit": "measurement_repeat_completed_wall_pair",
            "throughput_qps_ci_method": "nonparametric_bootstrap_of_pooled_repeat_ratio",
            "throughput_qps_point_estimate": "sum_completed_queries / sum_repeat_wall_clock_seconds",
            "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "latency_p99_ms": percentile(latencies, 0.99),
            "latency_query_cluster_ci95_low_ms": latency_low,
            "latency_query_cluster_ci95_high_ms": latency_high,
            "latency_p95_query_cluster_ci95_low_ms": p95_low,
            "latency_p95_query_cluster_ci95_high_ms": p95_high,
            "latency_p99_query_cluster_ci95_low_ms": p99_low,
            "latency_p99_query_cluster_ci95_high_ms": p99_high,
            "recall_mean": recall_mean,
            "recall_query_cluster_ci95_low": recall_low,
            "recall_query_cluster_ci95_high": recall_high,
            "target_met_measurement": recall_mean >= target,
            "target_lcb95_met_measurement": recall_low >= target,
            "target_met_each_repeat": all(repeat_mean_target_gate.values()),
            "target_lcb95_met_each_repeat": all(
                repeat_lcb95_target_gate.values()
            ),
            "repeat_target_gate_json": json.dumps(
                repeat_mean_target_gate, sort_keys=True
            ),
            "repeat_lcb95_target_gate_json": json.dumps(
                repeat_lcb95_target_gate, sort_keys=True
            ),
            "recall_lcb95_definition": (
                "2.5th percentile of a seeded nonparametric bootstrap over unique "
                "query-vector clusters; each cluster is its mean Recall@10 across "
                "all measurement repeats and failed requests score zero"
            ),
            "error_counts_json": json.dumps(dict(sorted(errors.items())), sort_keys=True),
            "latency_sample_scope": "successful_requests_only",
            "recall_sample_scope": "all_requests_with_failed_requests_scored_zero",
            "coverage_gate_passed": True,
            "status": "valid" if (
                len(ok) == len(rows)
                and recall_mean >= target
                and recall_low >= target
                and all(repeat_mean_target_gate.values())
                and all(repeat_lcb95_target_gate.values())
            ) else "invalid",
            **aggregate_telemetry_summary(repeat_summaries),
        }
    )
    return first


RAW_FIELDS = (
    "phase", "runner_version", "evaluation_scope", "target_recall", "filter_name", "query_cohort",
    "workload_source_kind", "trace_replay", "workload_requests", "unique_query_vectors",
    "method", "mode", "guidance_filter_strategy", "clients", "arm_order", "measurement_repeat", "dispatch_position",
    "trace_permutation_seed", "request_no", "trace_cycle", "query_no",
    "query_id", "client_id", "backend_pid", "client_native_tid", "client_requested_cpu",
    "client_affinity_applied", "started_offset_ms", "completed_offset_ms", "latency_ms",
    "activation_ms", "query_ms", "returned", "returned_ids", "recall_at_10", "config",
    "ef_search", "guided_collect_target", "max_scan_tuples", "scan_mem_multiplier",
    "iterative_scan", "traversal_guided_prioritization", "traversal_guided_burst",
    "traversal_guided_target", "error_type", "error",
)


def candidate_universe_gate(
    args: argparse.Namespace,
    filters: Sequence[FilterSpec],
    measurement_truth: Mapping[tuple[str, int], Any],
    calibration_query_ids: Mapping[int, int],
    measurement_query_ids: Mapping[int, int],
    measurement_query_nos: Iterable[int],
) -> dict[str, Any]:
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT count(*) FROM {args.insertion_table} "
            f"WHERE ({args.candidate_validity_predicate})"
        )
        candidate_rows = int(cur.fetchone()[0])
        if candidate_rows != args.expected_candidate_rows:
            raise BenchmarkContractError(
                f"candidate universe row mismatch: expected={args.expected_candidate_rows} observed={candidate_rows}"
            )
        observed_filter_rows: dict[str, int] = {}
        for item in filters:
            cur.execute(
                f"SELECT count(*) FROM {args.insertion_table} WHERE "
                f"({item.predicate}) AND ({args.candidate_validity_predicate})"
            )
            observed = int(cur.fetchone()[0])
            observed_filter_rows[item.name] = observed
            if observed != item.expected_rows:
                raise BenchmarkContractError(
                    f"filter candidate universe mismatch for {item.name}: "
                    f"expected={item.expected_rows} observed={observed}"
                )
        measurement_query_set = set(measurement_query_nos)
        calibration_ids = [calibration_query_ids[query_no] for query_no in SELECTION_QUERY_NOS]
        measurement_ids = [measurement_query_ids[query_no] for query_no in sorted(measurement_query_set)]
        requested_query_ids = sorted(set(calibration_ids) | set(measurement_ids))
        query_table = str(args.query_table or args.insertion_table)
        query_id_column = psycopg.sql.Identifier(args.query_id_column)
        cur.execute(
            psycopg.sql.SQL("SELECT count(*), count(DISTINCT {}) FROM {} WHERE {} = ANY(%s)").format(
                query_id_column,
                relation_identifier(query_table),
                query_id_column,
            ),
            (requested_query_ids,),
        )
        found, distinct = (int(value) for value in cur.fetchone())
        if found != len(requested_query_ids) or distinct != len(requested_query_ids):
            raise BenchmarkContractError(
                f"query relation does not contain each measurement query exactly once: "
                f"expected={len(requested_query_ids)} found={found} distinct={distinct}"
            )
    return {
        "passed": True,
        "predicate": args.candidate_validity_predicate,
        "expected_rows": args.expected_candidate_rows,
        "observed_rows": candidate_rows,
        "filter_rows": observed_filter_rows,
        "query_relation": query_table,
        "selection_query_ids": len(SELECTION_QUERY_NOS),
        "measurement_query_ids": len(measurement_query_set),
        "actual_query_id_disjoint": not bool(set(calibration_ids) & set(measurement_ids)),
        "required_query_ids_found": found,
        "truth_filter_count_gate": all(
            int(measurement_truth[(item.name, query_no)].filtered_rows) == item.expected_rows
            for item in filters
            for query_no in measurement_query_nos
        ),
    }


def validate_database_index_gate(
    database: Mapping[str, Any],
    source_index: str,
) -> dict[str, Any]:
    relations = database.get("relations")
    if not isinstance(relations, Mapping):
        raise BenchmarkContractError("database fingerprint has no relation identities")
    relation = relations.get(source_index)
    if not isinstance(relation, Mapping):
        raise BenchmarkContractError(f"database fingerprint is missing index {source_index}")
    passed = (
        relation.get("valid") is True
        and relation.get("ready") is True
        and relation.get("candidate_validity_predicate_matches") is True
        and int(relation.get("bytes") or 0) > 0
        and int(relation.get("oid") or 0) > 0
        and int(relation.get("relfilenode") or 0) > 0
    )
    if not passed:
        raise BenchmarkContractError(
            f"formal index identity/readiness/candidate-predicate gate failed for {source_index}"
        )
    return {
        "passed": True,
        "catalog_indexes": {
            "shared_stock_d1": {
                "relation": source_index,
                "oid": int(relation["oid"]),
                "relfilenode": int(relation["relfilenode"]),
                "bytes": int(relation["bytes"]),
                "valid": True,
                "ready": True,
                "candidate_validity_predicate_matches": True,
            }
        },
        "same_hnsw_index_for_stock_and_d1": True,
        "per_client_exact_hnsw_explain_gate_required": True,
    }


def validate_live_matched_recall_provenance(
    bundle: MatchedRecallBundle,
    database: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected = _require_mapping(
        bundle.provenance.get("database"), "matched-recall database provenance"
    )
    expected_runtime = _require_mapping(
        bundle.provenance.get("runtime"), "matched-recall runtime provenance"
    )
    if database.get("sqlens_build_id") != expected_runtime.get("loaded_vector_sqlens_build_id"):
        raise BenchmarkContractError("live SQLens build ID differs from matched-recall provenance")
    for field in (
        "candidate_validity_predicate",
        "candidate_validity_predicate_sha256",
    ):
        if database.get(field) != expected.get(field):
            raise BenchmarkContractError(
                f"live database {field} differs from matched-recall provenance"
            )
    expected_relations = _require_mapping(
        expected.get("relations"), "matched-recall relation provenance"
    )
    live_relations = _require_mapping(database.get("relations"), "live relation provenance")
    identities: dict[str, Any] = {}
    for relation_name in (args.insertion_table, args.insertion_index):
        expected_relation = _require_mapping(
            expected_relations.get(relation_name), f"matched-recall relation {relation_name}"
        )
        live_relation = _require_mapping(
            live_relations.get(relation_name), f"live relation {relation_name}"
        )
        for field in ("oid", "relfilenode"):
            if int(live_relation.get(field) or 0) != int(expected_relation.get(field) or -1):
                raise BenchmarkContractError(
                    f"live {relation_name} {field} differs from matched-recall provenance"
                )
        if relation_name == args.insertion_index:
            for field in (
                "valid",
                "ready",
                "candidate_validity_predicate_sha256",
                "candidate_validity_predicate_matches",
            ):
                if live_relation.get(field) != expected_relation.get(field):
                    raise BenchmarkContractError(
                        f"live source HNSW {field} differs from matched-recall provenance"
                    )
        identities[relation_name] = {
            "oid": int(live_relation["oid"]),
            "relfilenode": int(live_relation["relfilenode"]),
        }
    expected_query = _require_mapping(
        expected.get("query_table"), "matched-recall query relation provenance"
    )
    live_query = _require_mapping(database.get("query_table"), "live query relation provenance")
    for field in ("name", "oid", "relfilenode", "row_count", "columns"):
        if live_query.get(field) != expected_query.get(field):
            raise BenchmarkContractError(
                f"live query relation {field} differs from matched-recall provenance"
            )
    return {
        "passed": True,
        "runtime_build_exact_match": True,
        "candidate_universe_exact_match": True,
        "relation_identity_exact_match": True,
        "query_relation_exact_match": True,
        "relations": identities,
    }


def warm_database_cache(args: argparse.Namespace) -> dict[str, Any]:
    if not args.pg_prewarm:
        return {"passed": False, "enabled": False, "formal": False}
    relations = list(
        dict.fromkeys(
            [
                args.insertion_table,
                args.insertion_index,
                args.query_table or args.insertion_table,
            ]
        )
    )
    loaded: dict[str, int] = {}
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        for relation in relations:
            try:
                cur.execute("SELECT pg_prewarm(%s::regclass, 'buffer', 'read')", (relation,))
            except Exception as exc:  # noqa: BLE001
                raise BenchmarkContractError(
                    "warm-cache gate requires the pg_prewarm extension and readable benchmark relations"
                ) from exc
            loaded[relation] = int(cur.fetchone()[0])
    return {
        "passed": True,
        "enabled": True,
        "mode": "buffer/read",
        "relations": loaded,
        "completed_at_utc": utc_now(),
    }


def build_measurement_schedule(
    targets: Sequence[float],
    clients: Sequence[int],
    filters: Sequence[FilterSpec],
    seed: int,
    measurement_repeats: int = 1,
) -> list[dict[str, Any]]:
    if measurement_repeats <= 0:
        raise ValueError("measurement repeats must be positive")
    schedule: list[dict[str, Any]] = []
    block_no = 0
    for target in targets:
        for client_count in clients:
            for item in filters:
                for measurement_repeat in range(measurement_repeats):
                    order = interleaved_method_order(block_no, stable_seed(seed, "measurement", measurement_repeat))
                    for position, method in enumerate(order):
                        schedule.append(
                            {
                                "arm_order": len(schedule),
                                "block_no": block_no,
                                "method_position": position,
                                "measurement_repeat": measurement_repeat,
                                "target_recall": target,
                                "clients": client_count,
                                "filter_name": item.name,
                                "method": method,
                            }
                        )
                    block_no += 1
    return schedule


def measurement_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "guidance": {
            "guidance_filter_strategy": args.guidance_filter_strategy,
            "stock_and_d1_share_source_hnsw": True,
            "safe_guided_preserves_graph_traversal": (
                args.guidance_filter_strategy == "safe_guided"
            ),
            "traversal_guided_requires_dual_frontier_canary": (
                args.guidance_filter_strategy == "traversal_guided"
            ),
        },
        "request_order": {
            "seed_derivation": (
                "sha256(schedule_seed,target_recall,clients,filter_name,measurement_repeat)"
            ),
            "distinct_seed_per_repeat": True,
            "identical_permutation_paired_across_methods": True,
            "method_included_in_seed": False,
        },
        "cache": {
            "classification": "warm_cache",
            "relation_prewarm": "once_before_measurement",
            "session_warmup_requests_per_client_before_each_arm": (
                args.session_warmup_requests
            ),
            "reset_between_method_arms": False,
            "reset_between_repeats": False,
            "independence_scope": (
                "repeat request-order seeds are distinct; PostgreSQL shared-buffer and OS "
                "cache state is intentionally retained"
            ),
        },
        "throughput": {
            "point_estimate": "sum_completed_queries / sum_repeat_wall_clock_seconds",
            "ci_method": "nonparametric_bootstrap_of_pooled_repeat_ratio",
            "ci_resampling_unit": "measurement_repeat_completed_wall_pair",
            "repeat_pairing": "Stock and SQLens-D1 use the same request permutation per repeat",
        },
        "latency": {
            "point_estimates": "recomputed_from_all_successful_raw_requests",
            "reported": ["mean", "p50", "p95", "p99"],
        },
        "recall": {
            "point_estimate": "mean_over_all_raw_requests",
            "failed_request_score": 0.0,
            "lcb95": (
                "seeded nonparametric bootstrap over unique query-vector clusters; "
                "the lower endpoint is the 2.5th percentile"
            ),
            "target_gate": (
                "aggregate mean and aggregate LCB95 plus every-repeat mean and "
                "every-repeat LCB95 must meet target"
            ),
        },
        "telemetry": {
            "window": "measurement arm only; warmup and canary excluded",
            "host_cpu": "/proc/stat aggregate CPU counter delta",
            "host_block_io": "/proc/diskstats selected-device counter delta",
            "postgresql": [
                "pg_stat_io cluster-wide counter delta",
                "pg_stat_database current-database counter delta",
                "pg_statio_user_tables target-table counter delta",
                "pg_statio_user_indexes target-HNSW-index counter delta",
            ],
            "backend_cpu": "/proc/<tracked PostgreSQL backend PID>/stat CPU counter delta",
            "backend_proc_root": str(args.backend_proc_root),
            "backend_stats_flush": "pg_stat_force_next_flush on every measured backend",
        },
        "resume": {
            "commit_unit": "adjacent Stock/SQLens-D1 method pair for one cell/repeat",
            "half_pair_recovery": "forbidden; uncommitted bytes/summary rows are truncated",
            "validation": "run-spec, raw segment SHA256, summary SHA256, and pair coverage",
        },
    }


def validate_arm_telemetry(
    telemetry: Mapping[str, Any],
    expected_backend_pids: Sequence[int],
) -> None:
    """Require arm telemetry that can be attributed to the measured backends/index."""
    backend_cpu = _require_mapping(telemetry.get("backend_cpu"), "arm backend CPU telemetry")
    observed_pids = {int(pid) for pid in backend_cpu.get("backend_pids") or []}
    if (
        observed_pids != {int(pid) for pid in expected_backend_pids}
        or backend_cpu.get("tracking_complete") is not True
        or not isinstance(backend_cpu.get("per_backend"), Mapping)
    ):
        raise BenchmarkContractError("arm backend CPU telemetry is incomplete or misbound")
    postgres = _require_mapping(telemetry.get("postgresql"), "arm PostgreSQL telemetry")
    relations = _require_mapping(postgres.get("relations"), "arm relation telemetry")
    if (
        not relations.get("target_table")
        or not relations.get("target_index")
        or relations.get("tracking_complete") is not True
    ):
        raise BenchmarkContractError("arm relation-level telemetry is incomplete")
    table = _require_mapping(relations.get("table"), "arm target table telemetry")
    index = _require_mapping(relations.get("index"), "arm target index telemetry")
    required_table = ("relid", "heap_blks_read", "heap_blks_hit", "idx_blks_read", "idx_blks_hit")
    required_index = ("relid", "indexrelid", "idx_blks_read", "idx_blks_hit")
    if any(field not in table for field in required_table) or any(
        field not in index for field in required_index
    ):
        raise BenchmarkContractError("arm relation-level telemetry lacks target counters")


def validate_completion_coverage(
    summary_rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    arm_evidence: Sequence[Mapping[str, Any]],
    targets: Sequence[float],
    clients: Sequence[int],
    filters: Sequence[FilterSpec],
    measurement_repeats: int,
    workload: Workload,
    cell_coverage_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    expected_repeat_keys = {
        measurement_arm_key(
            float(arm["target_recall"]),
            int(arm["clients"]),
            str(arm["filter_name"]),
            str(arm["method"]),
            int(arm["measurement_repeat"]),
        )
        for arm in schedule
    }
    if len(expected_repeat_keys) != len(schedule):
        raise BenchmarkContractError("measurement schedule contains duplicate arms")
    schedule_by_key = {
        measurement_arm_key(
            float(arm["target_recall"]),
            int(arm["clients"]),
            str(arm["filter_name"]),
            str(arm["method"]),
            int(arm["measurement_repeat"]),
        ): arm
        for arm in schedule
    }
    repeat_rows = [row for row in summary_rows if row.get("summary_type") == "repeat"]
    observed_repeat_keys = [
        measurement_arm_key(
            float(row["target_recall"]),
            int(row["clients"]),
            str(row["filter_name"]),
            str(row["method"]),
            int(row["measurement_repeat"]),
        )
        for row in repeat_rows
    ]
    if (
        len(observed_repeat_keys) != len(set(observed_repeat_keys))
        or set(observed_repeat_keys) != expected_repeat_keys
    ):
        raise BenchmarkContractError("repeat summary coverage has duplicate or missing arms")
    for row in repeat_rows:
        if (
            int(row["workload_requests"]) != len(workload.requests)
            or int(row["unique_query_vectors"]) != workload.unique_query_vectors
            or not _is_explicit_true(row.get("telemetry_collected"))
            or not _is_explicit_true(row.get("target_lcb95_met_measurement"))
        ):
            raise BenchmarkContractError(
                "repeat summary workload/query/telemetry/LCB coverage is inconsistent"
            )

    expected_aggregate_keys = {
        (target, client_count, item.name, method)
        for target in targets
        for client_count in clients
        for item in filters
        for method in METHODS
    }
    aggregate_rows = [row for row in summary_rows if row.get("summary_type") == "aggregate"]
    observed_aggregate_keys = [
        (
            float(row["target_recall"]),
            int(row["clients"]),
            str(row["filter_name"]),
            str(row["method"]),
        )
        for row in aggregate_rows
    ]
    if (
        len(observed_aggregate_keys) != len(set(observed_aggregate_keys))
        or set(observed_aggregate_keys) != expected_aggregate_keys
    ):
        raise BenchmarkContractError("aggregate summary coverage has duplicate or missing cells")
    for row in aggregate_rows:
        if (
            int(row["workload_requests"]) != len(workload.requests)
            or int(row["unique_query_vectors"]) != workload.unique_query_vectors
            or int(row["measurement_repeats"]) != measurement_repeats
            or str(row.get("coverage_gate_passed", "")).lower() != "true"
            or not _is_explicit_true(row.get("telemetry_collected"))
            or not _is_explicit_true(row.get("target_lcb95_met_measurement"))
            or not _is_explicit_true(row.get("target_lcb95_met_each_repeat"))
        ):
            raise BenchmarkContractError("aggregate summary coverage gate is inconsistent")

    evidence_keys = [str(row["arm_key"]) for row in arm_evidence]
    if len(evidence_keys) != len(set(evidence_keys)) or set(evidence_keys) != expected_repeat_keys:
        raise BenchmarkContractError("measurement arm evidence has duplicate or missing arms")
    evidence_by_key = {str(row["arm_key"]): row for row in arm_evidence}
    for arm_key, scheduled in schedule_by_key.items():
        evidence = evidence_by_key[arm_key]
        telemetry = evidence.get("telemetry")
        if not isinstance(telemetry, Mapping):
            raise BenchmarkContractError(
                f"measurement arm evidence has no telemetry: {arm_key}"
            )
        validate_arm_telemetry(
            telemetry,
            [int(pid) for pid in evidence.get("backend_pids") or []],
        )
        for field in ("arm_order", "block_no", "method_position", "method", "target_recall", "clients", "filter_name"):
            if str(evidence.get(field)) != str(scheduled.get(field)):
                raise BenchmarkContractError(
                    f"measurement arm evidence does not bind scheduled {field}: {arm_key}"
                )
    expected_cell_keys = {
        measurement_cell_key(target, client_count, item.name)
        for target in targets
        for client_count in clients
        for item in filters
    }
    if cell_coverage_evidence is not None:
        if set(cell_coverage_evidence) != expected_cell_keys or any(
            row.get("passed") is not True for row in cell_coverage_evidence.values()
        ):
            raise BenchmarkContractError(
                "measurement cell coverage evidence has duplicate, missing, or failed cells"
            )
    for target in targets:
        for client_count in clients:
            for item in filters:
                repeat_seeds: list[int] = []
                for repeat in range(measurement_repeats):
                    scheduled_pair = [
                        schedule_by_key[
                            measurement_arm_key(
                                target, client_count, item.name, method, repeat
                            )
                        ]
                        for method in METHODS
                    ]
                    if (
                        len({int(row["block_no"]) for row in scheduled_pair}) != 1
                        or sorted(int(row["method_position"]) for row in scheduled_pair) != [0, 1]
                        or abs(int(scheduled_pair[0]["arm_order"]) - int(scheduled_pair[1]["arm_order"])) != 1
                    ):
                        raise BenchmarkContractError(
                            "paired method arms are not adjacent in one scheduled block"
                        )
                    pair = [
                        evidence_by_key[
                            measurement_arm_key(
                                target, client_count, item.name, method, repeat
                            )
                        ]
                        for method in METHODS
                    ]
                    seeds = {int(row["trace_permutation_seed"]) for row in pair}
                    orders = {str(row["trace_order_sha256"]) for row in pair}
                    if len(seeds) != 1 or len(orders) != 1:
                        raise BenchmarkContractError(
                            "paired method arms do not share one request order"
                        )
                    repeat_seeds.append(next(iter(seeds)))
                if len(set(repeat_seeds)) != measurement_repeats:
                    raise BenchmarkContractError(
                        "measurement arm evidence reuses a request-order seed across repeats"
                    )
    return {
        "passed": True,
        "workload_requests_per_arm": len(workload.requests),
        "unique_query_vectors_per_arm": workload.unique_query_vectors,
        "measurement_repeats": measurement_repeats,
        "repeat_arms": len(expected_repeat_keys),
        "aggregate_rows": len(expected_aggregate_keys),
        "validated_cells": len(expected_cell_keys),
        "no_duplicate_or_missing_repeat_arms": True,
        "no_duplicate_or_missing_aggregate_cells": True,
        "no_duplicate_or_missing_arm_evidence": True,
        "distinct_order_seed_per_repeat": True,
        "paired_identical_order_across_methods": True,
        "paired_method_arms_adjacent_in_schedule": True,
        "cell_raw_coverage_evidence_bound": cell_coverage_evidence is not None,
        "telemetry_complete_for_every_arm": True,
        "measurement_lcb95_gate_complete": True,
    }


def normalized_args(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {
        "resume", "overwrite", "execute", "dry_run", "plan_evidence",
        "backend_cpu_evidence", "runtime_sqlens_identity_evidence", "plan_query_id",
        "mode_configs_json", "expected_sqlens_build_id", "expected_vector_so_sha256",
        "fragment_tracking_evidence", "fragment_tracking_prepared",
    }
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }


def validate_formal_args(args: argparse.Namespace) -> tuple[list[float], list[int]]:
    targets = parse_targets(args.target_recalls)
    clients = parse_int_list(args.clients)
    formal_matrix = (
        args.requests == FORMAL_REQUESTS
        and tuple(targets) == FORMAL_TARGETS
        and tuple(clients) == FORMAL_CLIENTS
        and args.pg_prewarm
    )
    if not formal_matrix and not args.allow_nonformal_debug:
        raise BenchmarkContractError(
            "formal runs require requests=10000, recall targets exactly 0.90/0.95/0.99, "
            "clients=1,4,8,16,32,64, and pg_prewarm; use --allow-nonformal-debug for labelled diagnostics"
        )
    if args.bootstrap_samples <= 0:
        raise BenchmarkContractError("bootstrap samples must be positive")
    if args.measurement_repeats < FORMAL_MEASUREMENT_REPEATS:
        raise BenchmarkContractError("throughput estimates require at least six measurement repeats per cell")
    if (
        not args.allow_nonformal_debug
        and args.guidance_filter_strategy not in args.out.name
    ):
        raise BenchmarkContractError(
            "formal output path must name its bound guidance_filter_strategy to keep "
            "safe_guided and traversal_guided artifacts separate"
        )
    if args.resume and args.overwrite:
        raise BenchmarkContractError(
            "--resume and --overwrite are mutually exclusive"
        )
    if not args.allow_nonformal_debug and args.execute:
        if args.matched_recall_manifest is None:
            raise BenchmarkContractError(
                "formal runs require --matched-recall-manifest; throughput configuration "
                "cannot be calibrated or inferred inside this runner"
            )
    return targets, clients


def formal_protocol_configured(args: argparse.Namespace) -> bool:
    """Whether CLI settings are eligible for a paper artifact, independent of results."""
    return (
        not args.allow_nonformal_debug
        and args.requests == FORMAL_REQUESTS
        and tuple(parse_targets(args.target_recalls)) == FORMAL_TARGETS
        and tuple(parse_int_list(args.clients)) == FORMAL_CLIENTS
        and args.pg_prewarm
        and args.measurement_repeats >= FORMAL_MEASUREMENT_REPEATS
    )


def artifact_validity_flags(
    args: argparse.Namespace,
    *,
    diagnostic_complete: bool,
    completion_coverage: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    """Keep successful diagnostics distinct from formal, paper-eligible artifacts."""
    coverage_complete = (
        completion_coverage is not None and completion_coverage.get("passed") is True
    )
    formal_protocol_complete = formal_protocol_configured(args) and coverage_complete
    paper_eligible = bool(diagnostic_complete and formal_protocol_complete)
    return {
        "diagnostic_valid": bool(diagnostic_complete),
        "formal_protocol_complete": formal_protocol_complete,
        "paper_eligible": paper_eligible,
        "artifact_valid": paper_eligible,
    }


def measurement_prerequisite_status(args: argparse.Namespace) -> dict[str, Any]:
    """Report artifact availability without connecting to PostgreSQL."""
    discovered_query = (
        Path(args.measurement_query_file)
        if args.measurement_query_file is not None
        else discover_true_query_file(Path(args.query_search_dir))
    )
    query_path = discovered_query or DEFAULT_MEASUREMENT_QUERY_FILE
    query_manifest = (
        Path(args.measurement_query_manifest)
        if args.measurement_query_manifest is not None
        else query_path.with_name(query_path.stem + "_manifest.json")
    )
    paths = {
        "filters_csv": Path(args.filters_csv),
        "calibration_truth_csv": Path(args.calibration_truth_csv),
        "calibration_truth_manifest": Path(args.calibration_truth_manifest),
        "query_cohort_csv": query_path,
        "query_cohort_manifest": query_manifest,
        "measurement_truth_csv": Path(args.measurement_truth_csv),
        "measurement_truth_manifest": Path(args.measurement_truth_manifest),
    }
    matched_manifest = (
        Path(args.matched_recall_manifest) if args.matched_recall_manifest is not None else None
    )
    missing = [name for name, path in paths.items() if not path.is_file()]
    if matched_manifest is None or not matched_manifest.is_file():
        missing.append("matched_recall_manifest")
    manifest_flags: dict[str, Any] = {}
    for name in (
        "calibration_truth_manifest",
        "query_cohort_manifest",
        "measurement_truth_manifest",
    ):
        path = paths[name]
        if not path.is_file():
            continue
        try:
            payload = _read_json_object(path, name)
            manifest_flags[name] = {
                "artifact_valid": payload.get("artifact_valid") is True,
                "schema_version": payload.get("schema_version"),
            }
        except BenchmarkContractError as exc:
            manifest_flags[name] = {"artifact_valid": False, "error": str(exc)}
    validation_errors: list[str] = []
    static_gate: dict[str, Any] = {
        "passed": False,
        "requested_scope_filters": [],
        "requested_methods": list(METHODS),
        "requested_targets": parse_targets(args.target_recalls),
        "runtime_database_fingerprint_required": False,
    }
    filters: list[FilterSpec] | None = None
    targets = parse_targets(args.target_recalls)
    if not missing:
        try:
            filters = resolve_evaluation_filters(args, load_filters(Path(args.filters_csv)))
            static_gate["requested_scope_filters"] = [item.name for item in filters]
            verify_truth_manifest(
                paths["calibration_truth_csv"],
                paths["calibration_truth_manifest"],
                args.candidate_validity_predicate,
                args.expected_candidate_rows,
            )
            bundle = load_audited_matched_recall_configs(
                matched_manifest,
                truth_csv=paths["calibration_truth_csv"],
                filters_csv=paths["filters_csv"],
                filters=filters,
                targets=targets,
                require_runtime_provenance=False,
            )
            static_gate.update(
                {
                    "passed": True,
                    "requested_slice_complete": bool(
                        bundle.provenance.get("requested_slice_complete") is True
                    ),
                    "guidance_filter_strategy": bundle.guidance_filter_strategy,
                    "selected_config_cells": len(bundle.configs),
                    "matched_recall_manifest_sha256": sha256_file(matched_manifest),
                }
            )
        except (BenchmarkContractError, OSError, ValueError) as exc:
            validation_errors.append(str(exc))
    for name, flag in manifest_flags.items():
        if flag.get("artifact_valid") is not True:
            validation_errors.append(f"{name} is not artifact_valid")
    if not missing:
        try:
            verify_measurement_query_manifest(
                paths["query_cohort_csv"],
                paths["query_cohort_manifest"],
                args.candidate_validity_predicate,
            )
            verify_measurement_truth_manifest(
                paths["measurement_truth_csv"],
                paths["measurement_truth_manifest"],
                args.candidate_validity_predicate,
                args.expected_candidate_rows,
                paths["query_cohort_csv"],
            )
        except (BenchmarkContractError, OSError, ValueError) as exc:
            validation_errors.append(str(exc))
    passed = not missing and static_gate["passed"] and not validation_errors
    return {
        "passed": passed,
        "status": "ready" if passed else ("prerequisite_missing" if missing else "invalid"),
        "missing": sorted(set(missing)),
        "validation_errors": validation_errors,
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "manifest_flags": manifest_flags,
        "matched_recall_manifest": str(matched_manifest) if matched_manifest else None,
        "static_matched_recall_gate": static_gate,
    }


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    targets, clients = validate_formal_args(args)
    measurement_prerequisites = measurement_prerequisite_status(args)
    formal_ready = not args.allow_nonformal_debug and measurement_prerequisites["passed"]
    return {
        "dry_run": True,
        "runner_version": RUNNER_VERSION,
        "formal": not args.allow_nonformal_debug,
        "database_connected": False,
        "files_written": False,
        "dataset": "Amazon-10M",
        "evaluation_scope": args.evaluation_scope,
        "formal_filter_contract": {
            "full_matrix_filters": FORMAL_FILTER_COUNT,
            "representative_filters": list(REPRESENTATIVE_FILTERS),
            "representative_filters_are_a_fixed_stratified_slice": True,
            "cross_filter_or_cross_target_aggregation": "forbidden",
        },
        "methods": list(METHODS),
        "guidance_filter_strategy": args.guidance_filter_strategy,
        "internal_modes": MODE_BY_METHOD,
        "target_recalls": targets,
        "clients": clients,
        "configuration_source": "independently_audited_lcb_matched_recall_manifest",
        "matched_recall_manifest": (
            str(args.matched_recall_manifest) if args.matched_recall_manifest else None
        ),
        "formal_ready": formal_ready,
        "formal_prerequisites": measurement_prerequisites,
        "artifact_gate": {
            "diagnostic_valid": "complete diagnostic slice regardless of formal eligibility",
            "paper_eligible": "complete formal protocol plus all completion gates",
            "artifact_valid": "identical to paper_eligible",
        },
        "workload_requests": args.requests,
        "measurement_query_cohort": "q200..q10199",
        "measurement_unique_query_vectors": FORMAL_REQUESTS,
        "measurement_trace_replay": False,
        "debug_replay_query_cohort": "q100..q199",
        "debug_replay_only": True,
        "internal_calibration_enabled": False,
        "throughput_definition": "completed_queries / measurement_wall_clock_seconds",
        "warm_cache": args.pg_prewarm,
        "independent_connection_per_client": True,
        "seeded_interleaved_method_order": True,
        "measurement_repeats": args.measurement_repeats,
        "measurement_protocol": measurement_protocol(args),
        "throughput_ci_unit": "measurement_repeat_completed_wall_pair",
        "throughput_ci_method": "nonparametric_bootstrap_of_pooled_repeat_ratio",
        "measurement_recall_gate": (
            "mean and unique-query-cluster bootstrap LCB95 must both reach target "
            "for every repeat and the aggregate"
        ),
        "telemetry": {
            "host_cpu": "/proc/stat",
            "host_disk": "/proc/diskstats",
            "postgresql": [
                "pg_stat_io",
                "pg_stat_database",
                "pg_statio_user_tables target table",
                "pg_statio_user_indexes target HNSW index",
            ],
            "backend_cpu": "/proc/<tracked PostgreSQL backend PID>/stat",
            "backend_proc_root": str(args.backend_proc_root),
            "explicit_devices": list(parse_device_names(args.telemetry_devices)),
            "auto_paths": [str(args.out.parent), *(str(path) for path in args.telemetry_path)],
        },
        "resume_commit_unit": "complete Stock/SQLens-D1 pair for one cell/repeat",
    }


def execute_experiment(args: argparse.Namespace) -> int:
    targets, clients = validate_formal_args(args)
    filters = resolve_evaluation_filters(args, load_filters(args.filters_csv))
    if args.matched_recall_manifest is None:
        raise BenchmarkContractError("--matched-recall-manifest is required for execution")
    matched_recall = load_audited_matched_recall_configs(
        args.matched_recall_manifest,
        truth_csv=args.calibration_truth_csv,
        filters_csv=args.filters_csv,
        filters=filters,
        targets=targets,
    )
    bind_matched_recall_provenance(args, matched_recall)

    paths = output_paths(args.out)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_paths = [args.out.parent, *(Path(path) for path in args.telemetry_path)]
    (
        args.telemetry_devices_resolved,
        args.telemetry_device_resolution,
    ) = resolve_telemetry_devices(args.telemetry_devices, telemetry_paths)
    if args.overwrite and not args.resume:
        for path in paths.values():
            path.unlink(missing_ok=True)
    elif not args.resume and any(path.exists() for path in paths.values()):
        existing = [str(path) for path in paths.values() if path.exists()]
        raise BenchmarkContractError(f"output artifacts already exist: {existing}")

    calibration_truth_provenance = verify_truth_manifest(
        args.calibration_truth_csv,
        args.calibration_truth_manifest,
        args.candidate_validity_predicate,
        args.expected_candidate_rows,
    )
    calibration_truth, calibration_query_ids = load_truth(
        args.calibration_truth_csv, filters, args.candidate_validity_predicate
    )
    discovered_query = (
        Path(args.measurement_query_file)
        if args.measurement_query_file is not None
        else discover_true_query_file(args.query_search_dir)
    )
    measurement_query_path = (discovered_query or DEFAULT_MEASUREMENT_QUERY_FILE).resolve()
    measurement_query_manifest = (
        Path(args.measurement_query_manifest)
        if args.measurement_query_manifest is not None
        else measurement_query_path.with_name(measurement_query_path.stem + "_manifest.json")
    ).resolve()
    measurement_truth_path = Path(args.measurement_truth_csv).resolve()
    measurement_truth_manifest = Path(args.measurement_truth_manifest).resolve()
    required_measurement_paths = (
        measurement_query_path,
        measurement_query_manifest,
        measurement_truth_path,
        measurement_truth_manifest,
    )
    missing_measurement = [str(path) for path in required_measurement_paths if not path.is_file()]
    if missing_measurement:
        raise BenchmarkContractError(
            "formal prerequisite missing: q10200 measurement artifacts are absent: "
            + ", ".join(missing_measurement)
        )
    measurement_query_provenance = verify_measurement_query_manifest(
        measurement_query_path,
        measurement_query_manifest,
        args.candidate_validity_predicate,
    )
    measurement_truth_provenance = verify_measurement_truth_manifest(
        measurement_truth_path,
        measurement_truth_manifest,
        args.candidate_validity_predicate,
        args.expected_candidate_rows,
        measurement_query_path,
    )
    measurement_truth, measurement_query_ids = load_truth(
        measurement_truth_path, filters, args.candidate_validity_predicate
    )
    workload = choose_workload(
        measurement_query_ids,
        measurement_query_path,
        args.query_search_dir,
        args.requests,
        calibration_query_ids=(
            calibration_query_ids[query_no] for query_no in SELECTION_QUERY_NOS
        ),
        query_manifest=measurement_query_manifest,
        candidate_validity_predicate=args.candidate_validity_predicate,
        formal=not args.allow_nonformal_debug,
    )
    validate_workload_query_mapping(workload, measurement_query_ids)
    measurement_query_nos = {request.query_no for request in workload.requests}
    split_evidence = validate_calibration_measurement_split(
        calibration_truth,
        calibration_query_ids,
        measurement_truth,
        measurement_query_ids,
        filters,
    )

    args.modes = list(MODE_BY_METHOD.values())
    args.filter_atoms = {item.name: list(item.atoms) for item in filters}
    args.candidate_validity_predicate_explicit = True
    args.expected_truth_self_excluded = True
    args.plan_evidence = []
    args.backend_cpu_evidence = []
    args.runtime_sqlens_identity_evidence = []
    args.fragment_tracking_evidence = prepare_fragment_tracking(args)
    args.fragment_tracking_prepared = bool(args.fragment_tracking_evidence.get("prepared"))

    manifest: dict[str, Any] = {
        "artifact": "pgvector_formal_throughput_benchmark",
        "runner_version": RUNNER_VERSION,
        "status": "starting",
        "diagnostic_valid": False,
        "formal_protocol_complete": False,
        "paper_eligible": False,
        "artifact_valid": False,
        "started_at_utc": utc_now(),
        "paths": {key: str(path) for key, path in paths.items()},
    }
    atomic_json(paths["manifest"], manifest)
    guard_connection: Any = None
    try:
        guard_connection, guard_evidence = acquire_formal_data_guard(args)
        database = database_fingerprint(args, str(args.expected_sqlens_build_id))
        index_gate = validate_database_index_gate(database, args.insertion_index)
        matched_database_gate = validate_live_matched_recall_provenance(
            matched_recall, database, args
        )
        candidate_gate = candidate_universe_gate(
            args,
            filters,
            measurement_truth,
            calibration_query_ids,
            measurement_query_ids,
            measurement_query_nos,
        )
        warm_cache = warm_database_cache(args)
        if not warm_cache.get("passed") and not args.allow_nonformal_debug:
            raise BenchmarkContractError("formal run did not pass warm-cache prewarm gate")

        source = {
            "runner_sha256": sha256_file(Path(__file__)),
            "filters_csv": str(args.filters_csv),
            "filters_sha256": sha256_file(args.filters_csv),
            "calibration_truth": calibration_truth_provenance,
            "measurement_truth": measurement_truth_provenance,
            "measurement_query_cohort": measurement_query_provenance,
            "query_file": {
                "kind": workload.source_kind,
                "path": workload.source_path,
                "sha256": workload.source_sha256,
            },
            "git_revision": git_revision(),
            "matched_recall": dict(matched_recall.provenance),
        }
        schedule = build_measurement_schedule(
            targets, clients, filters, args.schedule_seed, args.measurement_repeats
        )
        run_spec = {
            "runner_version": RUNNER_VERSION,
            "args": normalized_args(args),
            "formal": not args.allow_nonformal_debug,
            "dataset": "Amazon-10M",
            "methods": list(METHODS),
            "matched_recall_configuration": {
                "source": "independently_audited_lcb_matched_recall_manifest",
                "internal_calibration": False,
                "targets": targets,
                "selected_cells": len(matched_recall.evidence),
                "provenance": dict(matched_recall.provenance),
            },
            "truth_contract": {
                "calibration": {
                    "query_numbers": "q0..q199",
                    "truth_csv": calibration_truth_provenance,
                },
                "measurement": {
                    "query_numbers": "q200..q10199",
                    "truth_csv": measurement_truth_provenance,
                    "query_cohort": measurement_query_provenance,
                },
                "actual_query_id_disjoint": split_evidence["actual_query_id_disjoint"],
            },
            "workload": {
                "workload_requests": len(workload.requests),
                "unique_query_vectors": workload.unique_query_vectors,
                "query_cohort": workload.query_cohort,
                "trace_replay": workload.trace_replay,
                "source_kind": workload.source_kind,
                "request_trace_sha256": canonical_sha256([asdict(item) for item in workload.requests]),
            },
            "throughput_definition": "completed_queries / measurement_wall_clock_seconds",
            "measurement_repeats": args.measurement_repeats,
            "throughput_ci_unit": "measurement_repeat_completed_wall_pair",
            "throughput_ci_method": "nonparametric_bootstrap_of_pooled_repeat_ratio",
            "warm_cache": True,
            "independent_connection_per_client": True,
            "measurement_schedule": "seeded_adjacent_interleaved_method_blocks",
            "guidance_semantics": {
                "guidance_filter_strategy": args.guidance_filter_strategy,
                "safe_guided_contract": (
                    "candidate_admission_validation_only_preserves_stock_graph_traversal"
                    if args.guidance_filter_strategy == "safe_guided"
                    else None
                ),
                "traversal_guided_contract": (
                    "dual_frontier_prioritization_requires_its_own_manifest_and_canary"
                    if args.guidance_filter_strategy == "traversal_guided"
                    else None
                ),
            },
            "measurement_protocol": measurement_protocol(args),
            "schedule": schedule,
            "source": source,
            "runtime_binary": {
                "expected_build_id": args.expected_sqlens_build_id,
                "expected_vector_so_sha256": args.expected_vector_so_sha256,
                "identity_source": "independently_audited_matched_recall_manifest",
            },
            "database": database,
            "index_gate": index_gate,
            "matched_recall_database_gate": matched_database_gate,
            "candidate_universe_gate": candidate_gate,
            "fragment_tracking_preparation": stable_fragment_tracking(
                args.fragment_tracking_evidence
            ),
        }
        run_spec_hash = canonical_sha256(run_spec)
        manifest.update(
            {
                "status": "running",
                "run_spec_hash": run_spec_hash,
                "run_spec": run_spec,
                "provenance": {
                    "runtime_binary": {
                        "expected_build_id": args.expected_sqlens_build_id,
                        "expected_vector_so_sha256": args.expected_vector_so_sha256,
                        "identity_source": "independently_audited_matched_recall_manifest",
                    },
                    "formal_data_guard": guard_evidence,
                    "matched_recall": dict(matched_recall.provenance),
                    "matched_recall_database_gate": matched_database_gate,
                    "warm_cache_gate": warm_cache,
                },
                "gates": {
                    "binary": {
                        "passed": True,
                        "expected_build_id": args.expected_sqlens_build_id,
                        "expected_vector_so_sha256": args.expected_vector_so_sha256,
                        "identity_source": "independently_audited_matched_recall_manifest",
                    },
                    "indexes": index_gate,
                    "ground_truth": {
                        "passed": True,
                        "calibration": calibration_truth_provenance,
                        "measurement": measurement_truth_provenance,
                        "query_cohort": measurement_query_provenance,
                        "split": split_evidence,
                    },
                    "candidate_universe": candidate_gate,
                    "warm_cache": warm_cache,
                    "matched_recall_artifact": {
                        "passed": True,
                        **dict(matched_recall.provenance),
                    },
                    "matched_recall_database": matched_database_gate,
                },
            }
        )
        atomic_json(paths["manifest"], manifest)

        if args.resume:
            checkpoint = load_checkpoint(paths["checkpoint"], run_spec_hash)
            prepare_resume_raw(paths["raw"], checkpoint, RAW_FIELDS)
            restore_csv_row_prefix(
                paths["configuration"],
                int(checkpoint.get("configuration_rows", 0)),
                str(checkpoint.get("configuration_sha256", "")),
            )
            resumed_summary_rows = restore_csv_row_prefix(
                paths["summary"],
                int(checkpoint.get("summary_rows", 0)),
                str(checkpoint.get("summary_sha256", "")),
            )
            resume_evidence = validate_resume_checkpoint(
                checkpoint,
                schedule,
                resumed_summary_rows,
                workload,
                args.measurement_repeats,
            )
            checkpoint.setdefault("resume_evidence", []).append(
                {"resumed_at_utc": utc_now(), **resume_evidence}
            )
            checkpoint["updated_at_utc"] = utc_now()
            atomic_json(paths["checkpoint"], checkpoint)
            manifest["resume_evidence"] = checkpoint["resume_evidence"]
            atomic_json(paths["manifest"], manifest)
        else:
            write_csv_atomic(paths["configuration"], list(matched_recall.evidence))
            raw_offset = initialize_raw_csv(paths["raw"], RAW_FIELDS)
            checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "run_spec_hash": run_spec_hash,
                "status": "running",
                "configuration_rows": len(matched_recall.evidence),
                "configuration_sha256": sha256_file(paths["configuration"]),
                "completed_measurement_pairs": [],
                "completed_measurement_cells": [],
                "measurement_rows": 0,
                "summary_rows": 0,
                "summary_sha256": "",
                "raw_byte_offset": raw_offset,
                "raw_pair_artifacts": {},
                "pair_evidence": {},
                "arm_evidence": [],
                "cell_coverage_evidence": {},
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
            }
            atomic_json(paths["checkpoint"], checkpoint)

        selected = dict(matched_recall.configs)
        selection_evidence = list(matched_recall.evidence)
        selection_by_key = {
            (row["filter_name"], row["method"], float(row["target_recall"])): row
            for row in selection_evidence
        }
        manifest["matched_recall_configuration"] = {
            "phase_complete": True,
            "internal_calibration": False,
            "independent_audit_passed": True,
            "requested_slice_complete": True,
            "rows": len(selection_evidence),
            "selected_configs": selection_evidence,
            "provenance": dict(matched_recall.provenance),
        }
        atomic_json(paths["manifest"], manifest)

        completed_pairs = set(checkpoint.get("completed_measurement_pairs", []))
        completed_cells = set(checkpoint.get("completed_measurement_cells", []))
        summary_rows = read_csv(paths["summary"]) if paths["summary"].exists() else []
        filters_by_name = {item.name: item for item in filters}
        if len(schedule) % len(METHODS):
            raise BenchmarkContractError("measurement schedule does not contain whole pairs")
        for pair_offset in range(0, len(schedule), len(METHODS)):
            pair_schedule = schedule[pair_offset : pair_offset + len(METHODS)]
            first_arm = pair_schedule[0]
            target = float(first_arm["target_recall"])
            client_count = int(first_arm["clients"])
            filter_name = str(first_arm["filter_name"])
            measurement_repeat = int(first_arm["measurement_repeat"])
            pair_key = measurement_pair_key(
                target, client_count, filter_name, measurement_repeat
            )
            if (
                {str(arm["method"]) for arm in pair_schedule} != set(METHODS)
                or len({int(arm["block_no"]) for arm in pair_schedule}) != 1
                or any(
                    (
                        float(arm["target_recall"]),
                        int(arm["clients"]),
                        str(arm["filter_name"]),
                        int(arm["measurement_repeat"]),
                    )
                    != (target, client_count, filter_name, measurement_repeat)
                    for arm in pair_schedule
                )
            ):
                raise BenchmarkContractError(
                    f"scheduled block is not one complete method pair: {pair_key}"
                )
            if pair_key in completed_pairs:
                continue

            pair_raw_rows: list[dict[str, Any]] = []
            pair_summaries: list[dict[str, Any]] = []
            pair_arm_evidence: list[dict[str, Any]] = []
            pair_arm_keys: list[str] = []
            for arm in pair_schedule:
                method = str(arm["method"])
                arm_key = measurement_arm_key(
                    target,
                    client_count,
                    filter_name,
                    method,
                    measurement_repeat,
                )
                config = selected[(filter_name, method, target)]
                arm_rows, wall_seconds, gate_evidence = run_measurement_arm(
                    args,
                    target,
                    client_count,
                    method,
                    config,
                    filters_by_name[filter_name],
                    workload,
                    measurement_truth,
                    int(arm["arm_order"]),
                    measurement_repeat,
                )
                gate_evidence.update(
                    {
                        "arm_order": int(arm["arm_order"]),
                        "block_no": int(arm["block_no"]),
                        "method_position": int(arm["method_position"]),
                        "method": method,
                        "target_recall": target,
                        "clients": client_count,
                        "filter_name": filter_name,
                    }
                )
                summary = summarize_arm(
                    arm_rows,
                    wall_seconds,
                    target,
                    client_count,
                    method,
                    config,
                    filters_by_name[filter_name],
                    workload,
                    selection_by_key[(filter_name, method, target)],
                    args.bootstrap_samples,
                    args.bootstrap_seed,
                    measurement_repeat,
                    gate_evidence["telemetry"],
                )
                pair_raw_rows.extend(arm_rows)
                pair_summaries.append(summary)
                pair_arm_evidence.append({"arm_key": arm_key, **gate_evidence})
                pair_arm_keys.append(arm_key)

            if (
                len({row["trace_permutation_seed"] for row in pair_arm_evidence}) != 1
                or len({row["trace_order_sha256"] for row in pair_arm_evidence}) != 1
            ):
                raise BenchmarkContractError(
                    f"method pair does not share one request trace: {pair_key}"
                )
            raw_artifact = append_csv_rows(paths["raw"], pair_raw_rows, RAW_FIELDS)
            raw_artifact.update(
                {
                    "pair_key": pair_key,
                    "arm_keys": pair_arm_keys,
                    "methods": [str(arm["method"]) for arm in pair_schedule],
                    "target_recall": target,
                    "clients": client_count,
                    "filter_name": filter_name,
                    "measurement_repeat": measurement_repeat,
                }
            )
            checkpoint["raw_byte_offset"] = raw_artifact["end_offset"]
            checkpoint["measurement_rows"] = (
                int(checkpoint["measurement_rows"]) + len(pair_raw_rows)
            )
            completed_pairs.add(pair_key)
            checkpoint["completed_measurement_pairs"] = sorted(completed_pairs)
            checkpoint["raw_pair_artifacts"][pair_key] = raw_artifact
            checkpoint["pair_evidence"][pair_key] = {
                "committed": True,
                "commit_unit": "cell_repeat_method_pair",
                "committed_at_utc": utc_now(),
                "arm_keys": pair_arm_keys,
                "methods": [str(arm["method"]) for arm in pair_schedule],
                "trace_permutation_seed": pair_arm_evidence[0][
                    "trace_permutation_seed"
                ],
                "trace_order_sha256": pair_arm_evidence[0]["trace_order_sha256"],
            }
            checkpoint["arm_evidence"].extend(pair_arm_evidence)
            summary_rows.extend(pair_summaries)
            checkpoint["summary_rows"] = len(summary_rows)
            checkpoint["updated_at_utc"] = utc_now()
            write_csv_atomic(paths["summary"], summary_rows)
            checkpoint["summary_sha256"] = sha256_file(paths["summary"])
            # This atomic checkpoint is the sole commit point for both method arms.
            atomic_json(paths["checkpoint"], checkpoint)

            cell_key = measurement_cell_key(target, client_count, filter_name)
            cell_pairs = [
                measurement_pair_key(target, client_count, filter_name, repeat)
                for repeat in range(args.measurement_repeats)
            ]
            if (
                cell_key not in completed_cells
                and all(repeat_pair in completed_pairs for repeat_pair in cell_pairs)
            ):
                repeat_summaries = [
                    row for row in summary_rows
                    if row.get("summary_type") == "repeat"
                    and float(row["target_recall"]) == target
                    and int(row["clients"]) == client_count
                    and row["filter_name"] == filter_name
                ]
                raw_rows = load_measurement_cell_rows(
                    paths["raw"], target, client_count, filter_name
                )
                cell_coverage = validate_measurement_cell_rows(
                    raw_rows,
                    workload,
                    target,
                    client_count,
                    filter_name,
                    args.measurement_repeats,
                )
                for repeat_method in METHODS:
                    aggregate = aggregate_measurement_cell(
                        [row for row in repeat_summaries if row["method"] == repeat_method],
                        [row for row in raw_rows if row["method"] == repeat_method],
                        args.bootstrap_samples,
                        args.bootstrap_seed,
                    )
                    summary_rows.append(aggregate)
                completed_cells.add(cell_key)
                checkpoint.setdefault("cell_coverage_evidence", {})[cell_key] = cell_coverage
                checkpoint["completed_measurement_cells"] = sorted(completed_cells)
                checkpoint["summary_rows"] = len(summary_rows)
                checkpoint["updated_at_utc"] = utc_now()
                write_csv_atomic(paths["summary"], summary_rows)
                checkpoint["summary_sha256"] = sha256_file(paths["summary"])
                atomic_json(paths["checkpoint"], checkpoint)
            manifest["progress"] = {
                "phase": "measurement",
                "completed_pairs": len(completed_pairs),
                "completed_arms": len(completed_pairs) * len(METHODS),
                "total_arms": len(schedule),
                "measurement_rows": checkpoint["measurement_rows"],
            }
            manifest["measurement_evidence"] = checkpoint["arm_evidence"]
            manifest["runtime_identity_evidence"] = args.runtime_sqlens_identity_evidence
            manifest["gates"]["binary"]["observed_runtime_exact_matches"] = (
                args.runtime_sqlens_identity_evidence
            )
            atomic_json(paths["manifest"], manifest)
            for summary in pair_summaries:
                print(
                    f"target={target:.2f} clients={client_count} filter={filter_name} "
                    f"method={summary['method']} "
                    f"qps={float(summary['throughput_qps']):.3f} "
                    f"p95={float(summary['latency_p95_ms']):.3f} "
                    f"p99={float(summary['latency_p99_ms']):.3f}",
                    flush=True,
                )

        # A crash after a repeat commits but before its cell aggregate must remain resumable.
        for target in targets:
            for client_count in clients:
                for filter_spec in filters:
                    cell_key = measurement_cell_key(target, client_count, filter_spec.name)
                    cell_pairs = [
                        measurement_pair_key(
                            target, client_count, filter_spec.name, repeat
                        )
                        for repeat in range(args.measurement_repeats)
                    ]
                    if cell_key in completed_cells or not all(
                        key in completed_pairs for key in cell_pairs
                    ):
                        continue
                    repeat_summaries = [
                        row for row in summary_rows
                        if row.get("summary_type") == "repeat"
                        and float(row["target_recall"]) == target
                        and int(row["clients"]) == client_count
                        and row["filter_name"] == filter_spec.name
                    ]
                    raw_rows = load_measurement_cell_rows(
                        paths["raw"], target, client_count, filter_spec.name
                    )
                    cell_coverage = validate_measurement_cell_rows(
                        raw_rows,
                        workload,
                        target,
                        client_count,
                        filter_spec.name,
                        args.measurement_repeats,
                    )
                    for method in METHODS:
                        summary_rows.append(
                            aggregate_measurement_cell(
                                [row for row in repeat_summaries if row["method"] == method],
                                [row for row in raw_rows if row["method"] == method],
                                args.bootstrap_samples,
                                args.bootstrap_seed,
                            )
                        )
                    completed_cells.add(cell_key)
                    checkpoint.setdefault("cell_coverage_evidence", {})[
                        cell_key
                    ] = cell_coverage
        checkpoint["completed_measurement_cells"] = sorted(completed_cells)
        checkpoint["summary_rows"] = len(summary_rows)
        checkpoint["updated_at_utc"] = utc_now()
        write_csv_atomic(paths["summary"], summary_rows)
        checkpoint["summary_sha256"] = sha256_file(paths["summary"])
        atomic_json(paths["checkpoint"], checkpoint)

        aggregate_rows = [row for row in summary_rows if row.get("summary_type") == "aggregate"]
        invalid = [row for row in aggregate_rows if row.get("status") != "valid"]
        expected_rows = len(schedule) * len(workload.requests)
        expected_cells = len(targets) * len(clients) * len(filters)
        completion_coverage = validate_completion_coverage(
            summary_rows,
            schedule,
            checkpoint["arm_evidence"],
            targets,
            clients,
            filters,
            args.measurement_repeats,
            workload,
            checkpoint.get("cell_coverage_evidence", {}),
        )
        diagnostic_complete = (
            not invalid
            and len(aggregate_rows) == expected_cells * len(METHODS)
            and int(checkpoint["measurement_rows"]) == expected_rows
            and len(completed_pairs) * len(METHODS) == len(schedule)
            and len(completed_cells) == expected_cells
        )
        artifact_flags = artifact_validity_flags(
            args,
            diagnostic_complete=diagnostic_complete,
            completion_coverage=completion_coverage,
        )
        checkpoint["status"] = (
            "complete" if artifact_flags["paper_eligible"] else
            ("diagnostic_complete" if artifact_flags["diagnostic_valid"] else "invalid")
        )
        checkpoint.update(artifact_flags)
        checkpoint["updated_at_utc"] = utc_now()
        atomic_json(paths["checkpoint"], checkpoint)
        manifest.update(
            {
                "status": checkpoint["status"],
                **artifact_flags,
                "finished_at_utc": utc_now(),
                "row_counts": {
                    "matched_recall_configs": len(selection_evidence),
                    "measurement_raw": checkpoint["measurement_rows"],
                    "summary": len(summary_rows),
                    "summary_repeat": len(schedule),
                    "summary_aggregate": len(aggregate_rows),
                },
                "invalid_summary_rows": len(invalid),
                "measurement_evidence": checkpoint["arm_evidence"],
                "telemetry": {
                    "device_resolution": args.telemetry_device_resolution,
                    "devices": list(args.telemetry_devices_resolved),
                    "arm_evidence_contains_full_deltas": True,
                    "summary_contains_flat_deltas_and_full_json": True,
                    "scopes": {
                        "host_cpu": "whole_host",
                        "host_disk": "selected_block_devices",
                        "pg_stat_io": "postgresql_cluster_wide",
                        "pg_stat_database": "current_database_cluster_wide",
                        "backend_cpu": "tracked_postgresql_client_backend_processes",
                        "relation_stats": "target_table_and_hnsw_index",
                    },
                },
                "measurement_protocol": measurement_protocol(args),
                "runtime_identity_evidence": args.runtime_sqlens_identity_evidence,
                "outputs": {
                    key: {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for key, path in paths.items()
                    if key != "manifest" and path.exists()
                },
            }
        )
        manifest["gates"]["measurement_coverage"] = completion_coverage
        atomic_json(paths["manifest"], manifest)
        return 0 if artifact_flags["diagnostic_valid"] else 2
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "diagnostic_valid": False,
                "formal_protocol_complete": False,
                "paper_eligible": False,
                "artifact_valid": False,
                "finished_at_utc": utc_now(),
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
            }
        )
        atomic_json(paths["manifest"], manifest)
        raise
    finally:
        if guard_connection is not None:
            guard_connection.rollback()
            guard_connection.close()


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formal Amazon-10M Stock vs SQLens-D1 throughput/tail-latency benchmark"
    )
    parser.add_argument(
        "--matched-recall-manifest",
        type=Path,
        help=(
            "completed matched-recall manifest; its selected configs are independently "
            "audited and are the only formal configuration source"
        ),
    )
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument(
        "--evaluation-scope",
        choices=EVALUATION_SCOPES,
        default="representative_filters",
        help=(
            "formal reporting scope: a fixed 4-filter stratified slice for the "
            "throughput/tail figure, or the complete 14-filter matrix"
        ),
    )
    parser.add_argument(
        "--calibration-truth-csv", "--truth-csv", dest="calibration_truth_csv",
        type=Path, default=DEFAULT_CALIBRATION_TRUTH,
        help="q200 exact truth used only for matched-recall calibration/audit",
    )
    parser.add_argument(
        "--calibration-truth-manifest", "--truth-manifest", dest="calibration_truth_manifest",
        type=Path, default=DEFAULT_CALIBRATION_TRUTH_MANIFEST,
        help="manifest corresponding to --calibration-truth-csv",
    )
    parser.add_argument(
        "--measurement-query-file", "--query-file", dest="measurement_query_file",
        type=Path, default=None,
        help="complete q10200 cohort; formal measurement uses q200..q10199",
    )
    parser.add_argument(
        "--measurement-query-manifest", type=Path, default=None,
        help="manifest corresponding to the q10200 measurement cohort",
    )
    parser.add_argument(
        "--measurement-truth-csv", type=Path, default=DEFAULT_MEASUREMENT_TRUTH,
        help="exact truth generated from the q10200 cohort",
    )
    parser.add_argument(
        "--measurement-truth-manifest", type=Path, default=DEFAULT_MEASUREMENT_TRUTH_MANIFEST,
        help="manifest corresponding to --measurement-truth-csv",
    )
    parser.add_argument("--query-search-dir", type=Path, default=RESULTS)
    parser.add_argument("--insertion-table", default=DEFAULT_TABLE)
    parser.add_argument("--insertion-index", default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--bfs-table", default=DEFAULT_TABLE)
    parser.add_argument("--bfs-index", default=DEFAULT_BFS_INDEX)
    parser.add_argument("--query-table")
    parser.add_argument("--query-id-column", default="id")
    parser.add_argument("--query-vector-column", default="embedding")
    parser.add_argument(
        "--candidate-validity-predicate", default=DEFAULT_CANDIDATE_VALIDITY_PREDICATE
    )
    parser.add_argument("--expected-candidate-rows", type=positive_int, default=EXPECTED_CANDIDATE_ROWS)
    parser.add_argument("--target-recalls", default="0.90,0.95,0.99")
    parser.add_argument("--clients", default="1,4,8,16,32,64")
    parser.add_argument("--requests", type=positive_int, default=FORMAL_REQUESTS)
    parser.add_argument("--measurement-repeats", type=positive_int, default=FORMAL_MEASUREMENT_REPEATS)
    parser.add_argument("--session-warmup-requests", type=nonnegative_int, default=100)
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--max-scan-tuples", type=positive_int, default=5_000_000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=32.0)
    parser.add_argument("--iterative-scan", choices=("off",), default="off")
    parser.add_argument(
        "--guidance-filter-strategy",
        choices=SUPPORTED_FORMAL_D1_STRATEGIES,
        default=FORMAL_D1_GUIDANCE_STRATEGY,
        help=(
            "must exactly match the independently audited matched-recall manifest; "
            "safe_guided is the default candidate-admission D1 protocol"
        ),
    )
    parser.add_argument("--guidance-selectivity-max-pct", type=float, default=100.0)
    parser.add_argument("--guidance-max-atoms", type=positive_int, default=64)
    parser.add_argument(
        "--traversal-guided-prioritization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--traversal-guided-burst", type=positive_int, default=8)
    parser.add_argument("--d1-cache-mb", type=positive_int, default=1024)
    parser.add_argument("--d3-cache-mb", type=positive_int, default=1024)
    parser.add_argument("--d2-page-access", choices=("off", "prefetch", "reorder"), default="reorder")
    parser.add_argument("--d2-index-page-access", choices=("off", "prefetch"), default="prefetch")
    parser.add_argument("--d2-page-window", type=positive_int, default=128)
    parser.add_argument("--d2-page-prefetch-min-items", type=positive_int, default=2)
    parser.add_argument("--d2-page-disable-after-no-merge", type=positive_int, default=2)
    parser.add_argument("--preferred-index-guc", default="hnsw.preferred_index")
    parser.add_argument("--require-preferred-index-guc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--statement-timeout-ms", type=nonnegative_int, default=300_000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-cache-per-query", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pg-prewarm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend-cpu-list", type=normalize_cpu_list)
    parser.add_argument("--client-cpu-list", type=normalize_cpu_list)
    parser.add_argument(
        "--backend-proc-root",
        type=Path,
        default=PROC_ROOT_PATH,
        help=(
            "procfs root for pg_backend_pid() values; use /proc/<container-host-pid>/root/proc "
            "when PostgreSQL runs in a distinct PID namespace"
        ),
    )
    parser.add_argument(
        "--telemetry-devices",
        default="",
        help=(
            "comma-separated Linux block-device names (for example sda4,nvme0n1); "
            "combined with devices auto-resolved from telemetry paths"
        ),
    )
    parser.add_argument(
        "--telemetry-path",
        type=Path,
        action="append",
        default=[],
        help=(
            "host path whose st_dev should be included in /proc/diskstats telemetry; "
            "repeat for database and index mount points"
        ),
    )
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument("--bootstrap-samples", type=positive_int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--expected-sqlens-build-id")
    parser.add_argument("--expected-vector-so-sha256")
    parser.add_argument("--start-barrier-timeout-seconds", type=positive_int, default=120)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS / "amazon10m_pgvector_formal_throughput_safe_guided_raw.csv",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-nonformal-debug", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if args.dry_run or not args.execute:
        print(json.dumps(dry_run_payload(args), sort_keys=True))
        return 0
    return execute_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
