from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .pgvector_target_recall_selectivity_runner import (
        bootstrap_mean_bounds,
        bootstrap_mean_ci,
        percentile,
    )
except ImportError:  # Direct script execution puts this directory on sys.path.
    from pgvector_target_recall_selectivity_runner import (  # type: ignore[no-redef]
        bootstrap_mean_bounds,
        bootstrap_mean_ci,
        percentile,
    )


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_valid_embeddings_filters.csv"
DEFAULT_TRUTH = ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal.csv"
DEFAULT_TRUTH_MANIFEST = (
    ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal_manifest.json"
)
DEFAULT_CALIBRATION_WORKLOAD = (
    ROOT / "results/hybrid_vector_db/figure5_r35_amazon_calibration.csv"
)
DEFAULT_MEASUREMENT_WORKLOAD = (
    ROOT / "results/hybrid_vector_db/figure5_r35_amazon_measurement.csv"
)
DEFAULT_FBIN = ROOT / "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"
DEFAULT_FAISS_INDEX = ROOT / "data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m32_efc200_seed57_t16.index"
DEFAULT_FAISS_INDEX_MANIFEST = Path(str(DEFAULT_FAISS_INDEX) + ".manifest.json")
FORMAL_HNSW_M = 32
FORMAL_HNSW_EF_CONSTRUCTION = 200
DEFAULT_RESULTS = ROOT / "results/hybrid_vector_db"
DEFAULT_TABLE = "amazon_grocery_reviews_10m_pgvector"
DEFAULT_CANDIDATE_VALIDITY_PREDICATE = "embedding_valid"
EXPECTED_VALID_ROWS = 9_979_556
FORMAL_FILTERS_SHA256 = "ae07c4d94450958f2071bf54f5db48d26c55328538087629cb1375c09bd4bcec"
FORMAL_TRUTH_SHA256 = "62e7f280f953828b680b2ae069de221bd6d593e42b241cd3d699ea870a1bfb5b"
FORMAL_TRUTH_MANIFEST_SHA256 = "0a6ab22579a8cf01eaa29889bf6ee2e822336d6d1c580b15697b7148a149bff2"
FORMAL_QUERY_COHORT_SHA256 = "c25e942bda9f45e435f000eeb938eaecce8e9fc562291bf6f56a57e0ced6a73f"
FORMAL_QUERY_COHORT_MANIFEST_SHA256 = "bdcfc34d46eddffa70e24cea7cd197df851274eba68ebf6291b1fc407569a8fc"
FORMAL_CALIBRATION_WORKLOAD_SHA256 = "54ca60d63c7c68391005e663bf20bf4b0e5b8d749973cf259a55e984a74b79e2"
FORMAL_MEASUREMENT_WORKLOAD_SHA256 = "983622346f87d5c084be24e5784da8cc9063e44302732e9b05f37cc360e719a8"
FORMAL_FBIN_SHA256 = "2a646da3c2925ff8e26b079a0f245badaedde02ef5aabcd4ac2cfd8f15653a6f"
FORMAL_FAISS_INDEX_SHA256 = "7f9db711c2328cbd9a6b4c26e5f48a1e777cb2ed003a3933f129ac58bc7a8ed5"
FORMAL_FAISS_INDEX_MANIFEST_SHA256 = "1de169917f43e45120ab1881971c0d4009f2cb140406f2966a94265bcfa8a832"
DEFAULT_EF_SEARCH = (
    20, 40, 60, 80, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 3000,
    4000, 5000, 7000, 8500, 10000, 20000, 50000, 100000,
)
DEFAULT_TARGETS = (0.90, 0.95, 0.99)
DEFAULT_CALIBRATION_QUERY_OFFSET = 20
DEFAULT_CALIBRATION_QUERIES = 80
DEFAULT_FINAL_QUERY_OFFSET = 100
DEFAULT_FINAL_QUERIES = 100
FORMAL_CALIBRATION_REPEATS = 2
FORMAL_FINAL_REPEATS = 5
CURRENT_CALIBRATION_REQUESTS = 200
CURRENT_MEASUREMENT_REQUESTS = 10_000
CURRENT_MEASUREMENT_REPEATS = 3
FORMAL_K = 10
FORMAL_ROWS = 10_000_000
CURRENT_PROTOCOL = "current-q10k-r3"
LEGACY_PROTOCOL = "legacy-q100-r5"
PROTOCOLS = (CURRENT_PROTOCOL, LEGACY_PROTOCOL)
TARGET_SELECTION_RULES = {
    "mean_latency": (
        "lowest mean latency among complete configurations with query-level "
        "mean Recall@10 >= target; bootstrap LCB95 is report-only"
    ),
    "lcb_then_max_recall": (
        "lowest mean latency among complete configurations whose query-level "
        "bootstrap Recall@10 LCB95 is at least the target; no mean-recall fallback "
        "is selected or publishable"
    ),
}
NA = "N/A"
FINALIZER_VERSION = "amazon10m-matched-recall-finalizer-v2"
CHECKPOINT_PROTOCOL_VERSION = "amazon10m-baselines-current-q200r2-q10kr3-v3"
LEGACY_CHECKPOINT_PROTOCOL_VERSION = "amazon10m-baselines-lcb95-q80r2-q100r5-v2"
FORMAL_FILTER_NAMES = (
    "popular_ge1000", "popular_ge1340", "popular_ge1780", "popular_ge2428",
    "popular_ge3284", "popular_ge4559", "price_10_to_20", "popular_ge10066",
    "rating5_price_le10", "long_review_ge500", "grocery_rating5",
    "grocery_helpful", "helpful_ge20", "grocery_long500",
)
SQL_FIRST_EXACT_LATENCY = "postgres_execute_fetchall_e2e"
FAISS_ALLOWLIST_LATENCY = "faiss_hnsw_cached_allowlist_search_only"
SQL_FIRST_CONTROL_METHOD = "sql_first_exact"
SQL_FIRST_PLANNER_METHOD = "sql_first_planner_chosen_exact"
SQL_FIRST_FORCED_METHOD = "sql_first_forced_indexed_exact"
FAISS_METHOD = "faiss_allowlist"
SQL_FIRST_METHODS = (
    SQL_FIRST_CONTROL_METHOD,
    SQL_FIRST_PLANNER_METHOD,
    SQL_FIRST_FORCED_METHOD,
)
FORMAL_METHODS = (*SQL_FIRST_METHODS, FAISS_METHOD)
METHOD_DESCRIPTIONS = {
    SQL_FIRST_CONTROL_METHOD: "materialized CTE exact SQL control",
    SQL_FIRST_PLANNER_METHOD: "planner-chosen direct exact SQL without HNSW",
    SQL_FIRST_FORCED_METHOD: "forced scalar-index direct exact SQL without HNSW",
    FAISS_METHOD: "Faiss HNSW with a complete PostgreSQL-derived allow-list",
}


@dataclass(frozen=True)
class FilterSpec:
    name: str
    target_rate: str
    predicate: str
    expected_rows: int
    actual_pct: float


@dataclass(frozen=True)
class TruthEntry:
    query_no: int
    query_id: int
    filter_name: str
    split: str
    ids: tuple[int, ...]
    candidate_rows: int
    kth_distance_sq: float
    tie_tolerance: float
    self_excluded: bool


@dataclass
class AllowList:
    selector: Any | None
    bitmap: Any | None
    rows: int
    build_ms: float
    bitmap_bytes: int
    valid: bool
    error: str = ""
    server_execution_ms: float = 0.0
    row_transfer_ms: float = 0.0
    bitmap_construction_ms: float = 0.0
    selector_construction_ms: float = 0.0
    full_setup_ms: float = 0.0


@dataclass(frozen=True)
class WorkloadRequest:
    request_no: int
    query_no: int
    query_id: int
    filter_name: str
    trace_cycle: int
    split: str


def parse_int_csv(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("integer list values must be greater than zero")
    return list(dict.fromkeys(parsed))


def parse_targets(value: str) -> list[float]:
    try:
        parsed = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated recall list") from exc
    if not parsed or any(target <= 0.0 or target > 1.0 for target in parsed):
        raise argparse.ArgumentTypeError("recall targets must be in (0, 1]")
    return parsed


def parse_methods(value: str) -> tuple[str, ...]:
    requested = [part.strip() for part in value.split(",") if part.strip()]
    if not requested:
        raise argparse.ArgumentTypeError("--methods must select at least one method")
    unknown = sorted(set(requested) - set(FORMAL_METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown methods {unknown}; choose from {list(FORMAL_METHODS)}"
        )
    requested_set = set(requested)
    return tuple(method for method in FORMAL_METHODS if method in requested_set)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def validate_table_name(value: str) -> str:
    parts = value.split(".")
    if len(parts) not in (1, 2):
        raise argparse.ArgumentTypeError("table must be table or schema.table")
    for part in parts:
        if not part or not (part[0].isalpha() or part[0] == "_"):
            raise argparse.ArgumentTypeError("table must contain unquoted SQL identifiers")
        if any(not (char.isalnum() or char in "_$") for char in part):
            raise argparse.ArgumentTypeError("table must contain unquoted SQL identifiers")
    return value


def validate_candidate_validity_predicate(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if normalized != DEFAULT_CANDIDATE_VALIDITY_PREDICATE:
        raise argparse.ArgumentTypeError(
            "formal Amazon-10M baselines require candidate universe embedding_valid"
        )
    return normalized


def effective_predicate(predicate: str, candidate_validity_predicate: str) -> str:
    validity = validate_candidate_validity_predicate(candidate_validity_predicate)
    return f"({predicate}) AND ({validity})"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as target:
        if not fields:
            return
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _atomic_write_outputs({path: ("csv", rows)})


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_outputs({path: ("json", value)})


def _atomic_write_outputs(
    outputs: dict[Path, tuple[str, Any]],
) -> None:
    """Write a small derived artifact set without exposing a partial result."""
    destinations = list(outputs)
    if not destinations:
        return
    parent = destinations[0].parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".amazon10m-finalize-", dir=str(parent)))
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for destination, (kind, value) in outputs.items():
            staged_path = stage / destination.name
            if kind == "json":
                write_json(staged_path, value)
            else:
                write_csv(staged_path, value)
            staged[destination] = staged_path
        for destination in destinations:
            if destination.exists():
                backup = stage / f"{destination.name}.backup"
                shutil.copy2(destination, backup)
                backups[destination] = backup
        for destination in destinations:
            os.replace(staged[destination], destination)
            replaced.append(destination)
    except Exception:
        for destination in reversed(replaced):
            backup = backups.get(destination)
            if backup is not None and backup.exists():
                os.replace(backup, destination)
            elif destination.exists():
                destination.unlink()
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_input_hash_errors(
    observed: Mapping[str, str],
    methods: Sequence[str],
    protocol: str = LEGACY_PROTOCOL,
) -> dict[str, dict[str, str | None]]:
    expected = {
        "filters": FORMAL_FILTERS_SHA256,
        "truth": FORMAL_TRUTH_SHA256,
        "truth_manifest": FORMAL_TRUTH_MANIFEST_SHA256,
        "query_cohort_csv": FORMAL_QUERY_COHORT_SHA256,
        "query_cohort_manifest": FORMAL_QUERY_COHORT_MANIFEST_SHA256,
        "fbin": FORMAL_FBIN_SHA256,
    }
    if protocol == CURRENT_PROTOCOL:
        expected.update(
            {
                "calibration_workload": FORMAL_CALIBRATION_WORKLOAD_SHA256,
                "measurement_workload": FORMAL_MEASUREMENT_WORKLOAD_SHA256,
            }
        )
    if FAISS_METHOD in methods:
        expected.update(
            {
                "faiss_index": FORMAL_FAISS_INDEX_SHA256,
                "faiss_index_manifest": FORMAL_FAISS_INDEX_MANIFEST_SHA256,
            }
        )
    return {
        name: {"actual": observed.get(name), "expected": digest}
        for name, digest in expected.items()
        if observed.get(name) != digest
    }


def load_workload(
    path: Path,
    *,
    expected_rows: int,
    expected_split: str,
    filter_names: set[str],
) -> list[WorkloadRequest]:
    rows = read_csv(path)
    required = {
        "request_no",
        "query_no",
        "query_id",
        "filter_name",
        "trace_cycle",
        "split",
    }
    if not rows:
        raise ValueError(f"workload is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"workload is missing columns {sorted(missing)}: {path}")
    if len(rows) != expected_rows:
        raise ValueError(
            f"workload row count mismatch: path={path} expected={expected_rows} "
            f"actual={len(rows)}"
        )
    requests: list[WorkloadRequest] = []
    for position, row in enumerate(rows):
        request = WorkloadRequest(
            request_no=int(row["request_no"]),
            query_no=int(row["query_no"]),
            query_id=int(row["query_id"]),
            filter_name=str(row["filter_name"]),
            trace_cycle=int(row["trace_cycle"]),
            split=str(row["split"]),
        )
        if request.request_no != position:
            raise ValueError(
                f"workload request_no must be contiguous 0..N-1: "
                f"position={position} request_no={request.request_no}"
            )
        if request.split != expected_split:
            raise ValueError(
                f"workload split mismatch at request {position}: "
                f"expected={expected_split} actual={request.split}"
            )
        if request.filter_name not in filter_names:
            raise ValueError(
                f"workload uses unknown filter {request.filter_name!r}"
            )
        if request.trace_cycle < 0:
            raise ValueError("workload trace_cycle must be nonnegative")
        requests.append(request)
    query_nos = [request.query_no for request in requests]
    query_ids = [request.query_id for request in requests]
    if len(set(query_nos)) != len(query_nos):
        raise ValueError(f"workload query_no values are not unique: {path}")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError(f"workload query_id values are not unique: {path}")
    return requests


def validate_workload_pair(
    calibration: Sequence[WorkloadRequest],
    measurement: Sequence[WorkloadRequest],
) -> None:
    calibration_query_nos = {request.query_no for request in calibration}
    measurement_query_nos = {request.query_no for request in measurement}
    calibration_query_ids = {request.query_id for request in calibration}
    measurement_query_ids = {request.query_id for request in measurement}
    if calibration_query_nos & measurement_query_nos:
        raise ValueError("calibration and measurement workload query_no values overlap")
    if calibration_query_ids & measurement_query_ids:
        raise ValueError("calibration and measurement workload query_id values overlap")


def workload_query_nos_by_filter(
    requests: Sequence[WorkloadRequest],
    filter_specs: Sequence[FilterSpec],
) -> dict[str, list[int]]:
    output = {spec.name: [] for spec in filter_specs}
    for request in requests:
        output[request.filter_name].append(request.query_no)
    if any(not query_nos for query_nos in output.values()):
        missing = sorted(name for name, query_nos in output.items() if not query_nos)
        raise ValueError(f"workload does not cover filters: {missing}")
    return output


def file_identity(path: Path, *, hash_contents: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_contents:
        result["sha256"] = sha256_file(path)
    return result


def _identity_matches(record: Any, observed: Mapping[str, Any]) -> bool:
    if not isinstance(record, Mapping) or record.get("sha256") != observed.get("sha256"):
        return False
    declared_size = record.get("bytes", record.get("size_bytes"))
    return declared_size is None or int(declared_size) == int(observed.get("bytes", -1))


def _read_bound_identity(record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping) or not record.get("path"):
        raise ValueError(f"truth manifest is missing {label} identity")
    path = Path(str(record["path"]))
    if not path.is_file():
        raise ValueError(f"truth manifest {label} path is unavailable: {path}")
    identity = file_identity(path, hash_contents=True)
    if not _identity_matches(record, identity):
        raise ValueError(f"truth manifest {label} identity does not match the file")
    return identity


def verify_truth_manifest(
    manifest_path: Path,
    truth_identity: Mapping[str, Any],
    fbin_identity: Mapping[str, Any],
    protocol: str = LEGACY_PROTOCOL,
) -> dict[str, Any]:
    """Bind exact GT to its unique query cohort and source vector collection."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read exact-truth manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("exact-truth manifest must contain a JSON object")
    outputs = payload.get("outputs")
    inputs = payload.get("inputs")
    query_source = payload.get("query_source")
    truth_output = outputs.get("truth_csv") if isinstance(outputs, Mapping) else None
    source_fbin = inputs.get("fbin") if isinstance(inputs, Mapping) else None
    if not (
        payload.get("artifact_valid") is True
        and payload.get("method") == "exact_filtered_l2_tie_aware"
        and int(payload.get("k", 0)) == FORMAL_K
        and int(payload.get("rows", 0)) == FORMAL_ROWS
        and int(payload.get("filters", 0)) == len(FORMAL_FILTER_NAMES)
        and payload.get("query_ids_disjoint") is True
        and _identity_matches(truth_output, truth_identity)
        and _identity_matches(source_fbin, fbin_identity)
        and isinstance(query_source, Mapping)
    ):
        raise ValueError("exact-truth manifest contract failed")

    calibration = payload.get("calibration")
    final = payload.get("final")
    required_calibration = (
        100 if protocol == CURRENT_PROTOCOL else
        DEFAULT_CALIBRATION_QUERY_OFFSET + DEFAULT_CALIBRATION_QUERIES
    )
    required_final = (
        10_100 if protocol == CURRENT_PROTOCOL else DEFAULT_FINAL_QUERIES
    )
    if not (
        isinstance(calibration, Mapping)
        and isinstance(final, Mapping)
        and int(calibration.get("queries", 0)) >= required_calibration
        and int(final.get("queries", 0)) >= required_final
    ):
        raise ValueError(
            "exact-truth manifest does not cover the requested formal protocol: "
            f"protocol={protocol} calibration>={required_calibration} "
            f"final>={required_final}"
        )

    cohort_csv = _read_bound_identity(query_source.get("cohort_csv"), "query cohort CSV")
    cohort_manifest_record = query_source.get("manifest")
    cohort_manifest = _read_bound_identity(cohort_manifest_record, "query cohort manifest")
    try:
        cohort_payload = json.loads(
            Path(str(cohort_manifest_record["path"])).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read bound query cohort manifest: {exc}") from exc
    cohort_output = (
        cohort_payload.get("outputs", {}).get("cohort_csv")
        if isinstance(cohort_payload, Mapping)
        else None
    )
    if not (
        isinstance(cohort_payload, Mapping)
        and cohort_payload.get("artifact_valid") is True
        and cohort_payload.get("selection", {}).get("disjoint") is True
        and _identity_matches(cohort_output, cohort_csv)
    ):
        raise ValueError("bound query cohort manifest contract failed")

    postgres = inputs.get("postgres") if isinstance(inputs, Mapping) else None
    if not isinstance(postgres, Mapping):
        raise ValueError("exact-truth manifest is missing PostgreSQL relation provenance")
    required_relation = ("table", "table_oid", "table_relfilenode", "rows")
    if any(postgres.get(key) in (None, "") for key in required_relation):
        raise ValueError("exact-truth manifest PostgreSQL relation provenance is incomplete")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "artifact_valid": True,
        "truth_csv": dict(truth_output),
        "fbin": dict(source_fbin),
        "query_cohort_csv": cohort_csv,
        "query_cohort_manifest": cohort_manifest,
        "postgres_relation": {
            key: postgres[key] for key in required_relation
        },
    }


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def load_filter_specs(path: Path, selected: set[str] | None = None) -> list[FilterSpec]:
    specs: list[FilterSpec] = []
    seen: set[str] = set()
    for row in read_csv(path):
        name = row["filter_name"]
        if selected and name not in selected:
            continue
        if name in seen:
            raise ValueError(f"duplicate filter_name in {path}: {name}")
        seen.add(name)
        specs.append(
            FilterSpec(
                name=name,
                target_rate=row["target_rate"],
                predicate=row["predicate"],
                expected_rows=int(row["count"]),
                actual_pct=float(row["actual_pct"]),
            )
        )
    if selected:
        missing = selected - seen
        if missing:
            raise ValueError(f"missing filter specs: {sorted(missing)}")
    if not specs:
        raise ValueError(f"no filter specs in {path}")
    return specs


def _parse_ids(value: str, k: int) -> tuple[int, ...]:
    ids = tuple(int(part) for part in value.split(",") if part.strip())
    if len(ids) != k or len(set(ids)) != k:
        raise ValueError(f"truth top-k must contain {k} distinct IDs, got {len(ids)}")
    return ids


def load_truth(
    path: Path,
    filter_specs: Sequence[FilterSpec],
    calibration_query_nos: Sequence[int],
    final_query_nos: Sequence[int],
    k: int,
    *,
    enforce_requested_split: bool = True,
) -> tuple[dict[tuple[str, int], TruthEntry], dict[int, int]]:
    calibration_set = set(calibration_query_nos)
    final_set = set(final_query_nos)
    if calibration_set & final_set:
        raise ValueError("calibration and final query_no sets overlap")
    expected_query_nos = calibration_set | final_set
    specs_by_name = {spec.name: spec for spec in filter_specs}
    filter_names = set(specs_by_name)
    truth: dict[tuple[str, int], TruthEntry] = {}
    query_ids: dict[int, int] = {}

    rows = read_csv(path)
    required_fields = {
        "exact_filtered_topk_ids",
        "filtered_rows",
        "k",
        "kth_distance_sq",
        "tie_tolerance",
        "self_excluded",
        "query_split",
        "candidate_validity_predicate",
        "query_validity_predicate",
        "candidate_rows",
    }
    if not rows or not required_fields.issubset(rows[0]):
        missing = sorted(required_fields - (set(rows[0]) if rows else set()))
        raise ValueError(f"truth artifact uses the retired schema; missing fields: {missing}")

    for row in rows:
        if row.get("method") != "pre_filter_exact":
            continue
        if (
            row.get("candidate_validity_predicate")
            != DEFAULT_CANDIDATE_VALIDITY_PREDICATE
            or row.get("query_validity_predicate")
            != DEFAULT_CANDIDATE_VALIDITY_PREDICATE
        ):
            raise ValueError("truth row candidate/query universe is not embedding_valid")
        filter_name = row["filter_name"]
        query_no = int(row["query_no"])
        if filter_name not in filter_names or query_no not in expected_query_nos:
            continue
        truth_predicate = " ".join(str(row.get("predicate", "")).split())
        configured_predicate = " ".join(specs_by_name[filter_name].predicate.split())
        if truth_predicate != configured_predicate:
            raise ValueError(
                f"truth pair {(filter_name, query_no)} predicate does not match the active filter"
            )
        query_id = int(row["query_id"])
        previous_query_id = query_ids.setdefault(query_no, query_id)
        if previous_query_id != query_id:
            raise ValueError(f"query_no={query_no} maps to multiple query IDs")
        requested_split = "calibration" if query_no in calibration_set else "final"
        split = row.get("query_split", requested_split)
        if split not in {"calibration", "final"}:
            raise ValueError(f"query_no={query_no} has invalid source split={split!r}")
        if enforce_requested_split and split != requested_split:
            raise ValueError(
                f"query_no={query_no} has split={split!r}, expected {requested_split!r}"
            )
        key = (filter_name, query_no)
        if key in truth:
            raise ValueError(f"duplicate truth pair: {key}")
        ids = _parse_ids(row["exact_filtered_topk_ids"], k)
        if int(row["k"]) != k:
            raise ValueError(f"truth pair {key} k mismatch: expected={k} actual={row['k']}")
        self_excluded = str(row["self_excluded"]).strip().lower() == "true"
        if not self_excluded:
            raise ValueError(f"truth pair {key} did not exclude the query row")
        if query_id in ids:
            raise ValueError(f"truth pair {key} contains its own query ID")
        if row.get("recall_at_10_exact_filtered") not in (None, "", "1", "1.0"):
            raise ValueError(f"truth pair {key} is not marked exact")
        candidate_rows = int(row["candidate_rows"])
        filtered_rows = int(float(row["filtered_rows"]))
        if candidate_rows != filtered_rows:
            raise ValueError(f"truth pair {key} candidate_rows/filtered_rows mismatch")
        kth_distance_sq = float(row["kth_distance_sq"])
        tie_tolerance = float(row["tie_tolerance"])
        if not math.isfinite(kth_distance_sq) or kth_distance_sq < 0:
            raise ValueError(f"truth pair {key} has invalid kth_distance_sq")
        if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
            raise ValueError(f"truth pair {key} has invalid tie_tolerance")
        truth[key] = TruthEntry(
            query_no=query_no,
            query_id=query_id,
            filter_name=filter_name,
            split=split,
            ids=ids,
            candidate_rows=candidate_rows,
            kth_distance_sq=kth_distance_sq,
            tie_tolerance=tie_tolerance,
            self_excluded=self_excluded,
        )

    expected_pairs = {
        (filter_name, query_no)
        for filter_name in filter_names
        for query_no in expected_query_nos
    }
    missing = expected_pairs - set(truth)
    extra_query_ids = expected_query_nos - set(query_ids)
    if missing or extra_query_ids:
        preview = sorted(missing)[:5]
        raise ValueError(
            f"truth grid incomplete: missing_pairs={len(missing)} preview={preview} "
            f"missing_query_ids={sorted(extra_query_ids)}"
        )
    calibration_ids = {query_ids[query_no] for query_no in calibration_set}
    final_ids = {query_ids[query_no] for query_no in final_set}
    if len(calibration_ids) != len(calibration_set) or len(final_ids) != len(final_set):
        raise ValueError("query IDs must be unique within each query split")
    if calibration_ids & final_ids:
        raise ValueError("calibration and final query IDs overlap")
    for spec in filter_specs:
        candidate_counts = {
            truth[(spec.name, query_no)].candidate_rows for query_no in expected_query_nos
        }
        if candidate_counts != {spec.expected_rows}:
            raise ValueError(
                f"filter={spec.name} candidate count mismatch: "
                f"config={spec.expected_rows} truth={sorted(candidate_counts)}"
            )
    return truth, query_ids


def read_fbin_memmap(path: Path, limit: int | None = None) -> tuple[Any, int, int]:
    import numpy as np

    with path.open("rb") as source:
        header = source.read(8)
    if len(header) != 8:
        raise ValueError(f"invalid fbin header: {path}")
    stored_rows, dimensions = struct.unpack("ii", header)
    rows = min(stored_rows, limit) if limit is not None else stored_rows
    vectors = np.memmap(
        path,
        dtype="float32",
        mode="r",
        offset=8,
        shape=(stored_rows, dimensions),
    )
    return vectors[:rows], rows, dimensions


def exact_sql(
    table: str,
    predicate: str,
    k: int,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> str:
    """Direct exact SQL shape.

    Adding floating-point zero preserves exact ranking while preventing the
    ORDER BY expression from matching pgvector's HNSW order-by operator path.
    EXPLAIN still proves HNSW absence. Unlike the historical control below,
    this shape does not force CTE materialization.
    """
    predicate = effective_predicate(predicate, candidate_validity_predicate)
    return f"""
SELECT id
FROM {table}
WHERE ({predicate}) AND id <> %s
ORDER BY (embedding <-> %s::vector) + 0.0::double precision, id
LIMIT {int(k)}
""".strip()


def materialized_exact_sql(
    table: str,
    predicate: str,
    k: int,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> str:
    """Historical SQL-first control with an explicit materialization fence."""
    predicate = effective_predicate(predicate, candidate_validity_predicate)
    return f"""
WITH filtered AS MATERIALIZED (
    SELECT id, embedding
    FROM {table}
    WHERE ({predicate}) AND id <> %s
)
SELECT id
FROM filtered
ORDER BY embedding <-> %s::vector, id
LIMIT {int(k)}
""".strip()


def exact_sql_for_method(
    method: str,
    table: str,
    predicate: str,
    k: int,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> str:
    if method == SQL_FIRST_CONTROL_METHOD:
        return materialized_exact_sql(
            table, predicate, k, candidate_validity_predicate
        )
    if method in {SQL_FIRST_PLANNER_METHOD, SQL_FIRST_FORCED_METHOD}:
        return exact_sql(table, predicate, k, candidate_validity_predicate)
    raise ValueError(f"method is not a SQL-first exact arm: {method}")


def plan_index_names(plan: Any) -> set[str]:
    names: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("Index Name"):
                names.add(str(node["Index Name"]))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(plan)
    return names


def assert_no_hnsw_index(plan: Any, hnsw_indexes: Iterable[str]) -> set[str]:
    used = plan_index_names(plan)
    forbidden = {name for item in hnsw_indexes for name in (item, item.rsplit(".", 1)[-1])}
    offenders = sorted(
        name for name in used if name in forbidden or "hnsw" in name.lower()
    )
    if offenders:
        raise RuntimeError(f"sql_first_exact EXPLAIN used HNSW index(es): {offenders}")
    return used


def plan_node_types(plan: Any) -> set[str]:
    names: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("Node Type"):
                names.add(str(node["Node Type"]))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(plan)
    return names


def assert_scalar_index_plan(
    plan: Any, hnsw_indexes: Iterable[str], scalar_indexes: Iterable[str]
) -> set[str]:
    used = assert_no_hnsw_index(plan, hnsw_indexes)
    allowed = {
        name.rsplit(".", 1)[-1]
        for name in scalar_indexes
    }
    matched = {name for name in used if name.rsplit(".", 1)[-1] in allowed}
    if not matched:
        raise RuntimeError(
            "sql_first_forced_indexed_exact EXPLAIN did not use a registered "
            f"scalar index: used={sorted(used)} allowed={sorted(allowed)}"
        )
    return used


def decode_explain(value: Any) -> Any:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, list) and value and isinstance(value[0], dict) and "Plan" in value[0]:
        return value[0]["Plan"]
    if isinstance(value, dict) and "Plan" in value:
        return value["Plan"]
    raise ValueError("unexpected EXPLAIN (FORMAT JSON) result")


def balanced_order(values: Sequence[Any], block_no: int, seed: int) -> list[Any]:
    base = list(values)
    random.Random(seed).shuffle(base)
    if not base:
        return []
    offset = block_no % len(base)
    return base[offset:] + base[:offset]


def set_bitmap_ids(bitmap: Any, ids: Sequence[int] | Any, total_rows: int) -> int:
    import numpy as np

    values = np.asarray(ids, dtype=np.int64)
    if values.size == 0:
        return 0
    if int(values.min()) < 0 or int(values.max()) >= total_rows:
        raise ValueError("allow-list contains an ID outside the Faiss row range")
    byte_positions = values >> 3
    masks = np.left_shift(np.uint8(1), (values & 7).astype(np.uint8))
    np.bitwise_or.at(bitmap, byte_positions, masks)
    return int(values.size)


def bitmap_contains(bitmap: Any, row_id: int) -> bool:
    return bool(int(bitmap[row_id >> 3]) & (1 << (row_id & 7)))


def result_membership_errors(bitmap: Any, result_ids: Sequence[int]) -> list[int]:
    return [int(row_id) for row_id in result_ids if not bitmap_contains(bitmap, int(row_id))]


def allowlist_id_sql(
    table: str,
    predicate: str,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> str:
    """Return the complete predicate ID stream used to build one allow-list.

    There is deliberately no ORDER BY, LIMIT, sampling, or candidate cap. The
    selector must describe every SQL-valid Faiss ID before HNSW starts.
    """
    return (
        f"SELECT id FROM {table} WHERE "
        + effective_predicate(predicate, candidate_validity_predicate)
    )


def build_allow_list(
    conn: Any,
    faiss_module: Any,
    table: str,
    spec: FilterSpec,
    total_rows: int,
    fetch_rows: int,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> AllowList:
    import numpy as np

    started = time.perf_counter()
    bitmap = np.zeros((total_rows + 7) // 8, dtype=np.uint8)
    streamed_rows = 0
    server_execution_ms = 0.0
    row_transfer_ms = 0.0
    bitmap_construction_ms = 0.0
    selector_construction_ms = 0.0
    try:
        suffix = hashlib.sha1(spec.name.encode()).hexdigest()[:12]
        cursor_name = f"allowlist_{suffix}"
        temporary_table = f"allowlist_materialized_{suffix}"
        chunks: list[Any] = []
        with conn.transaction():
            with conn.cursor() as control:
                server_started = time.perf_counter()
                control.execute(
                    f"CREATE TEMP TABLE {temporary_table} ON COMMIT DROP AS "
                    + allowlist_id_sql(
                        table, spec.predicate, candidate_validity_predicate
                    )
                )
                materialized_rows = int(getattr(control, "rowcount", -1))
                if materialized_rows < 0:
                    control.execute(f"SELECT count(*) FROM {temporary_table}")
                    materialized_rows = int(control.fetchone()[0])
                server_execution_ms = (time.perf_counter() - server_started) * 1000.0
            with conn.cursor(name=cursor_name) as cursor:
                transfer_started = time.perf_counter()
                cursor.execute(f"SELECT id FROM {temporary_table}")
                while True:
                    batch = cursor.fetchmany(fetch_rows)
                    if not batch:
                        break
                    values = np.fromiter(
                        (int(row[0]) for row in batch),
                        dtype=np.int64,
                        count=len(batch),
                    )
                    streamed_rows += int(values.size)
                    chunks.append(values)
                row_transfer_ms = (time.perf_counter() - transfer_started) * 1000.0
        if materialized_rows != streamed_rows:
            raise RuntimeError(
                "materialized/streamed allow-list row mismatch: "
                f"materialized={materialized_rows} streamed={streamed_rows}"
            )
        bitmap_started = time.perf_counter()
        for values in chunks:
            set_bitmap_ids(bitmap, values, total_rows)
        bitmap_construction_ms = (time.perf_counter() - bitmap_started) * 1000.0
        # Faiss expects the number of addressable IDs (bits), not the backing
        # array's byte count. Passing bitmap.size would silently exclude IDs
        # above total_rows / 8 and collapse recall on a 10M collection.
        selector_started = time.perf_counter()
        selector = faiss_module.IDSelectorBitmap(total_rows, faiss_module.swig_ptr(bitmap))
        selector_construction_ms = (time.perf_counter() - selector_started) * 1000.0
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if streamed_rows != spec.expected_rows:
            return AllowList(
                selector=selector,
                bitmap=bitmap,
                rows=streamed_rows,
                build_ms=elapsed_ms,
                bitmap_bytes=int(bitmap.nbytes),
                valid=False,
                error=f"row_count_mismatch: expected={spec.expected_rows} actual={streamed_rows}",
                server_execution_ms=server_execution_ms,
                row_transfer_ms=row_transfer_ms,
                bitmap_construction_ms=bitmap_construction_ms,
                selector_construction_ms=selector_construction_ms,
                full_setup_ms=elapsed_ms,
            )
        return AllowList(
            selector=selector,
            bitmap=bitmap,
            rows=streamed_rows,
            build_ms=elapsed_ms,
            bitmap_bytes=int(bitmap.nbytes),
            valid=True,
            server_execution_ms=server_execution_ms,
            row_transfer_ms=row_transfer_ms,
            bitmap_construction_ms=bitmap_construction_ms,
            selector_construction_ms=selector_construction_ms,
            full_setup_ms=elapsed_ms,
        )
    except Exception as exc:  # Keep other filters measurable and make this one explicitly invalid.
        return AllowList(
            selector=None,
            bitmap=bitmap,
            rows=streamed_rows,
            build_ms=(time.perf_counter() - started) * 1000.0,
            bitmap_bytes=int(bitmap.nbytes),
            valid=False,
            error=f"{exc.__class__.__name__}: {exc}",
            server_execution_ms=server_execution_ms,
            row_transfer_ms=row_transfer_ms,
            bitmap_construction_ms=bitmap_construction_ms,
            selector_construction_ms=selector_construction_ms,
            full_setup_ms=(time.perf_counter() - started) * 1000.0,
        )


def recall_at_k(result_ids: Sequence[int], truth_ids: Sequence[int], k: int) -> float:
    denominator = min(k, len(truth_ids))
    if denominator == 0:
        return 0.0
    return len(set(result_ids[:k]) & set(truth_ids[:k])) / denominator


def tie_aware_recall_at_k(
    result_ids: Sequence[int],
    query_id: int,
    vectors: Any,
    truth: TruthEntry,
    k: int,
) -> float:
    import numpy as np

    unique_ids: list[int] = []
    seen: set[int] = set()
    for value in result_ids:
        row_id = int(value)
        if row_id == query_id or row_id in seen:
            continue
        if row_id < 0 or row_id >= len(vectors):
            raise ValueError(f"result ID outside vector row range: {row_id}")
        seen.add(row_id)
        unique_ids.append(row_id)
        if len(unique_ids) == k:
            break
    if not unique_ids:
        return 0.0
    query = np.asarray(vectors[query_id], dtype=np.float32)
    candidates = np.asarray(vectors[np.asarray(unique_ids, dtype=np.int64)], dtype=np.float32)
    distances = np.einsum("ij,ij->i", candidates - query, candidates - query)
    threshold = truth.kth_distance_sq + truth.tie_tolerance
    qualifying = int(np.count_nonzero(distances <= threshold))
    return min(k, qualifying) / k


def search_faiss(
    index: Any,
    faiss_module: Any,
    query: Any,
    selector: Any,
    ef_search: int,
    k: int,
    query_id: int | None = None,
) -> tuple[list[int], float]:
    import numpy as np

    query_batch = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
    params = faiss_module.SearchParametersHNSW()
    params.efSearch = int(ef_search)
    params.sel = selector
    started = time.perf_counter()
    request_k = k + 1 if query_id is not None else k
    _, labels = index.search(query_batch, request_k, params=params)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ids = [
        int(value)
        for value in labels[0]
        if int(value) >= 0 and (query_id is None or int(value) != query_id)
    ]
    return ids[:k], elapsed_ms


def search_sql_exact(
    cursor: Any,
    sql_text: str,
    query_id: int,
    query_vector: str,
    planner_mode: str = "auto",
) -> tuple[list[int], float]:
    if planner_mode not in {"auto", "forced_indexed"}:
        raise ValueError(f"unknown SQL-first planner mode: {planner_mode}")
    cursor.execute(
        "RESET enable_seqscan"
        if planner_mode == "auto"
        else "SET enable_seqscan = off"
    )
    started = time.perf_counter()
    try:
        cursor.execute(sql_text, (query_id, query_vector))
        rows = cursor.fetchall()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return [int(row[0]) for row in rows], elapsed_ms
    finally:
        if planner_mode == "forced_indexed":
            cursor.execute("RESET enable_seqscan")


def pair_key(filter_name: str, query_no: int, repeat: int) -> str:
    return f"{filter_name}|q{query_no}|r{repeat}"


def measurement_row(
    *,
    phase: str,
    method: str,
    spec: FilterSpec,
    query_no: int,
    query_id: int,
    repeat: int,
    schedule_position: int,
    block_no: int,
    ef_search: int | str,
    result_ids: Sequence[int] | None,
    truth_ids: Sequence[int],
    latency_ms: float | str,
    truth_entry: TruthEntry | None = None,
    vectors: Any | None = None,
    error: str = "",
    matched_target_recalls: Sequence[float] = (),
    filter_membership_valid: bool = True,
    request_no: int | str = NA,
    trace_cycle: int | str = NA,
    target_recall: float | str = NA,
) -> dict[str, Any]:
    valid = not error and filter_membership_valid
    if method in SQL_FIRST_METHODS:
        latency_definition = SQL_FIRST_EXACT_LATENCY
        sql_shape = (
            "materialized_cte_control"
            if method == SQL_FIRST_CONTROL_METHOD
            else "direct_exact_sql"
        )
        latency_includes = (
            "PostgreSQL predicate evaluation, exact L2 ranking, server execution, "
            "protocol transfer, and top-k fetchall"
            + (
                "; the historical control also includes MATERIALIZED CTE creation/scan"
                if method == SQL_FIRST_CONTROL_METHOD
                else "; no explicit MATERIALIZED CTE"
            )
        )
        latency_excludes = "query-vector prefetch, EXPLAIN, warmup, ground-truth generation, output I/O"
    elif method == FAISS_METHOD:
        sql_shape = NA
        latency_definition = FAISS_ALLOWLIST_LATENCY
        latency_includes = "Faiss IndexHNSWFlat.search with IDSelectorBitmap allow-list admission"
        latency_excludes = "SQL ID stream, bitmap/selector construction, query-vector materialization, output I/O"
    else:
        raise ValueError(f"unknown measurement method: {method}")
    return {
        "phase": phase,
        "method": method,
        "sql_first_planner_mode": (
            "materialized_control"
            if method == SQL_FIRST_CONTROL_METHOD
            else "planner_chosen"
            if method == SQL_FIRST_PLANNER_METHOD
            else "forced_indexed"
            if method == SQL_FIRST_FORCED_METHOD
            else NA
        ),
        "sql_shape": sql_shape,
        "filter_name": spec.name,
        "target_rate": spec.target_rate,
        "predicate": spec.predicate,
        "actual_selectivity": spec.actual_pct / 100.0,
        "ef_search": ef_search,
        "pair_key": pair_key(spec.name, query_no, repeat),
        "request_no": request_no,
        "trace_cycle": trace_cycle,
        "block_no": block_no,
        "schedule_position": schedule_position,
        "query_no": query_no,
        "query_id": query_id,
        "repeat": repeat,
        "target_recall": target_recall,
        "matched_target_recalls": ",".join(f"{target:.2f}" for target in matched_target_recalls),
        "latency_definition": latency_definition,
        "latency_includes": latency_includes,
        "latency_excludes": latency_excludes,
        "search_latency_ms": latency_ms if valid else NA,
        "cached_allowlist_search_ms": (
            latency_ms if valid and method == FAISS_METHOD else NA
        ),
        "recall_at_10": (
            tie_aware_recall_at_k(
                result_ids or [], query_id, vectors, truth_entry, len(truth_ids)
            )
            if valid and truth_entry is not None and vectors is not None
            else recall_at_k(result_ids or [], truth_ids, len(truth_ids))
            if valid
            else NA
        ),
        "recall_contract": (
            "distance_threshold_tie_aware" if truth_entry is not None else "id_intersection_test_only"
        ),
        "filter_membership_valid": filter_membership_valid if not error else False,
        "returned": len(result_ids or []) if valid else NA,
        "result_ids": ",".join(str(value) for value in (result_ids or [])) if valid else NA,
        "valid": valid,
        "error": error,
    }


def setup_row(spec: FilterSpec, allow_list: AllowList) -> dict[str, Any]:
    return {
        "phase": "setup",
        "method": FAISS_METHOD,
        "filter_name": spec.name,
        "target_rate": spec.target_rate,
        "predicate": spec.predicate,
        "actual_selectivity": spec.actual_pct / 100.0,
        "ef_search": NA,
        "pair_key": NA,
        "query_no": NA,
        "query_id": NA,
        "repeat": NA,
        "latency_definition": "one_time_allowlist_build",
        "latency_includes": "complete PostgreSQL ID stream, bitmap population, and Faiss IDSelectorBitmap construction",
        "latency_excludes": "HNSW search, query-vector materialization, warmup, output I/O",
        "allowlist_sql_contract": "complete SQL-valid ID stream; no ORDER BY, LIMIT, sampling, or candidate cap",
        "search_latency_ms": NA,
        "recall_at_10": NA,
        "returned": NA,
        "result_ids": NA,
        "allowlist_build_rows": allow_list.rows,
        "allowlist_build_ms": allow_list.build_ms,
        "allowlist_sql_materialization_ms": allow_list.server_execution_ms,
        "allowlist_server_execution_ms": allow_list.server_execution_ms,
        "allowlist_row_transfer_ms": allow_list.row_transfer_ms,
        "allowlist_bitmap_build_ms": allow_list.bitmap_construction_ms,
        "allowlist_bitmap_construction_ms": allow_list.bitmap_construction_ms,
        "allowlist_selector_construction_ms": allow_list.selector_construction_ms,
        "allowlist_full_setup_ms": allow_list.full_setup_ms,
        "allowlist_bitmap_bytes": allow_list.bitmap_bytes,
        "valid": allow_list.valid,
        "error": allow_list.error,
    }


def full_setup_search_row(
    *,
    conn: Any,
    faiss_module: Any,
    index: Any,
    table: str,
    spec: FilterSpec,
    total_rows: int,
    fetch_rows: int,
    query: Any,
    query_no: int,
    query_id: int,
    ef_search: int,
    k: int,
    repeat: int = 0,
    request_no: int | str = NA,
    trace_cycle: int | str = NA,
    truth_entry: TruthEntry | None = None,
    vectors: Any | None = None,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[str, Any]:
    started = time.perf_counter()
    allow_list = build_allow_list(
        conn,
        faiss_module,
        table,
        spec,
        total_rows,
        fetch_rows,
        candidate_validity_predicate,
    )
    error = allow_list.error
    ids: list[int] = []
    search_ms: float | str = NA
    if allow_list.valid and allow_list.selector is not None:
        try:
            ids, measured_search_ms = search_faiss(
                index,
                faiss_module,
                query,
                allow_list.selector,
                ef_search,
                k,
                query_id=query_id,
            )
            search_ms = measured_search_ms
            invalid_ids = result_membership_errors(allow_list.bitmap, ids)
            if invalid_ids:
                raise RuntimeError(
                    f"full-e2e Faiss returned IDs outside allow-list: {invalid_ids[:5]}"
                )
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
    e2e_ms = (time.perf_counter() - started) * 1000.0
    return {
        "phase": "setup_search_e2e",
        "method": FAISS_METHOD,
        "filter_name": spec.name,
        "target_rate": spec.target_rate,
        "predicate": spec.predicate,
        "actual_selectivity": spec.actual_pct / 100.0,
        "ef_search": ef_search,
        "query_no": query_no,
        "query_id": query_id,
        "repeat": repeat,
        "request_no": request_no,
        "trace_cycle": trace_cycle,
        "pair_key": f"{spec.name}|q{query_no}|r{repeat}|setup_search_ef{ef_search}",
        "latency_definition": "continuous_full_allowlist_setup_plus_ann_search_e2e",
        "allowlist_sql_contract": "complete SQL-valid ID stream; no ORDER BY, LIMIT, sampling, or candidate cap",
        "allowlist_build_rows": allow_list.rows,
        "allowlist_sql_materialization_ms": allow_list.server_execution_ms,
        "allowlist_server_execution_ms": allow_list.server_execution_ms,
        "allowlist_row_transfer_ms": allow_list.row_transfer_ms,
        "allowlist_bitmap_build_ms": allow_list.bitmap_construction_ms,
        "allowlist_bitmap_construction_ms": allow_list.bitmap_construction_ms,
        "allowlist_selector_construction_ms": allow_list.selector_construction_ms,
        "allowlist_full_setup_ms": allow_list.full_setup_ms,
        "cached_ann_search_ms": search_ms,
        "cached_allowlist_search_ms": search_ms,
        "full_setup_plus_search_e2e_ms": e2e_ms,
        "continuous_full_e2e_ms": e2e_ms,
        "continuous_recall_at_10": (
            tie_aware_recall_at_k(ids, query_id, vectors, truth_entry, k)
            if not error and truth_entry is not None and vectors is not None
            else NA
        ),
        "result_ids": ",".join(str(value) for value in ids) if not error else NA,
        "returned": len(ids) if not error else NA,
        "valid": not error,
        "error": error,
    }


def _row_ok(row: dict[str, Any]) -> bool:
    value = row.get("valid", False)
    if isinstance(value, str):
        value = value.lower() in {"1", "true", "yes"}
    return bool(value) and not row.get("error")


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def aggregate_measurements(
    rows: Sequence[dict[str, Any]],
    *,
    phase: str,
    method: str,
    filter_name: str,
    ef_search: int | None,
    query_nos: Sequence[int],
    repeats: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    target_recall: float | None = None,
) -> dict[str, Any]:
    items = [
        row
        for row in rows
        if row.get("phase") == phase
        and row.get("method") == method
        and row.get("filter_name") == filter_name
        and (ef_search is None or int(row.get("ef_search", -1)) == ef_search)
        and (
            target_recall is None
            or row.get("target_recall") in (None, "", NA)
            or math.isclose(float(row.get("target_recall", -1.0)), target_recall)
        )
    ]
    expected_pairs = {(int(query_no), repeat) for query_no in query_nos for repeat in range(repeats)}
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates = 0
    for row in items:
        key = (int(row["query_no"]), int(row["repeat"]))
        if key in by_pair:
            duplicates += 1
        by_pair[key] = row
    observed_pairs = set(by_pair)
    def metrics_ok(row: dict[str, Any]) -> bool:
        try:
            latency = float(row["search_latency_ms"])
            recall = float(row["recall_at_10"])
        except (KeyError, TypeError, ValueError):
            return False
        return math.isfinite(latency) and latency > 0.0 and 0.0 <= recall <= 1.0

    error_rows = sum(not _row_ok(row) for row in items)
    invalid_metric_rows = sum(_row_ok(row) and not metrics_ok(row) for row in items)
    complete = observed_pairs == expected_pairs and duplicates == 0 and error_rows == 0
    complete = complete and invalid_metric_rows == 0
    base = {
        "queries": len({query_no for query_no, _ in observed_pairs}),
        "samples": len(items),
        "expected_queries": len(query_nos),
        "expected_repeats": repeats,
        "expected_samples": len(expected_pairs),
        "missing_pairs": len(expected_pairs - observed_pairs),
        "extra_pairs": len(observed_pairs - expected_pairs),
        "duplicate_pairs": duplicates,
        "errors": error_rows,
        "invalid_metric_rows": invalid_metric_rows,
        "rows_complete": complete,
        "status": "valid" if complete else "invalid",
    }
    if not complete:
        return {
            **base,
            "recall_mean": NA,
            "recall_lcb95": NA,
            "recall_ci95_low": NA,
            "recall_ci95_high": NA,
            "latency_mean_ms": NA,
            "latency_p50_ms": NA,
            "latency_p95_ms": NA,
            "latency_p99_ms": NA,
            "latency_query_mean_ci95_low_ms": NA,
            "latency_query_mean_ci95_high_ms": NA,
        }

    query_recalls: list[float] = []
    query_latencies: list[float] = []
    sample_latencies: list[float] = []
    for query_no in query_nos:
        query_items = [by_pair[(int(query_no), repeat)] for repeat in range(repeats)]
        query_recalls.append(statistics.fmean(float(row["recall_at_10"]) for row in query_items))
        query_latencies.append(
            statistics.fmean(float(row["search_latency_ms"]) for row in query_items)
        )
        sample_latencies.extend(float(row["search_latency_ms"]) for row in query_items)
    recall_lcb, recall_ci_low, recall_ci_high = bootstrap_mean_bounds(
        query_recalls, bootstrap_samples, bootstrap_seed
    )
    latency_ci_low, latency_ci_high = bootstrap_mean_ci(
        query_latencies, bootstrap_samples, bootstrap_seed + 1
    )
    return {
        **base,
        "recall_mean": statistics.fmean(query_recalls),
        "recall_lcb95": recall_lcb,
        "recall_ci95_low": recall_ci_low,
        "recall_ci95_high": recall_ci_high,
        "latency_mean_ms": statistics.fmean(query_latencies),
        "latency_p50_ms": statistics.median(sample_latencies),
        "latency_p95_ms": percentile(sample_latencies, 0.95),
        "latency_p99_ms": percentile(sample_latencies, 0.99),
        "latency_query_mean_ci95_low_ms": latency_ci_low,
        "latency_query_mean_ci95_high_ms": latency_ci_high,
    }


def calibration_table(
    raw_rows: Sequence[dict[str, Any]],
    filter_specs: Sequence[FilterSpec],
    ef_values: Sequence[int],
    targets: Sequence[float],
    query_nos: Sequence[int] | Mapping[str, Sequence[int]],
    repeats: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    allow_lists: dict[str, AllowList] | None = None,
    selection_policy: str = "mean_latency",
) -> tuple[list[dict[str, Any]], dict[tuple[str, float], int]]:
    if selection_policy not in TARGET_SELECTION_RULES:
        raise ValueError(f"unknown calibration selection policy: {selection_policy}")
    rows: list[dict[str, Any]] = []
    for filter_no, spec in enumerate(filter_specs):
        spec_query_nos = (
            list(query_nos[spec.name])
            if isinstance(query_nos, Mapping)
            else list(query_nos)
        )
        for ef_search in ef_values:
            stats = aggregate_measurements(
                raw_rows,
                phase="calibration",
                method=FAISS_METHOD,
                filter_name=spec.name,
                ef_search=ef_search,
                query_nos=spec_query_nos,
                repeats=repeats,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + filter_no * 1009 + ef_search,
            )
            allow_list = (allow_lists or {}).get(spec.name)
            for target in targets:
                mean_eligible = bool(
                    stats["status"] == "valid"
                    and float(stats["recall_mean"]) >= target
                )
                lcb95_eligible = bool(
                    mean_eligible and float(stats["recall_lcb95"]) >= target
                )
                rows.append(
                    {
                        "filter_name": spec.name,
                        "target_rate": spec.target_rate,
                        "predicate": spec.predicate,
                        "actual_selectivity": spec.actual_pct / 100.0,
                        "method": FAISS_METHOD,
                        "target_recall": target,
                        "ef_search": ef_search,
                        **stats,
                        "eligible": mean_eligible,
                        "mean_eligible": mean_eligible,
                        "lcb95_eligible": lcb95_eligible,
                        "calibration_selection_policy": selection_policy,
                        "selected": False,
                        "allowlist_build_rows": allow_list.rows if allow_list else NA,
                        "allowlist_build_ms": allow_list.build_ms if allow_list else NA,
                        "allowlist_bitmap_bytes": allow_list.bitmap_bytes if allow_list else NA,
                    }
                )

    selected: dict[tuple[str, float], int] = {}
    for spec in filter_specs:
        for target in targets:
            ladder_rows = [
                row
                for row in rows
                if row["filter_name"] == spec.name
                and float(row["target_recall"]) == target
            ]
            ladder_complete = len(ladder_rows) == len(ef_values) and all(
                row["status"] == "valid" for row in ladder_rows
            )
            max_row = max(ladder_rows, key=lambda row: int(row["ef_search"])) if ladder_rows else None
            observed_metrics = [row for row in ladder_rows if row["status"] == "valid"]
            ladder_proof = {
                "configured_ef_search": list(ef_values),
                "observed_ef_search": [int(row["ef_search"]) for row in ladder_rows],
                "all_configs_complete": ladder_complete,
                "all_mean_below_target": bool(
                    ladder_complete
                    and all(float(row["recall_mean"]) < target for row in ladder_rows)
                ),
            }
            mean_candidates = [row for row in ladder_rows if row["mean_eligible"]]
            if selection_policy == "lcb_then_max_recall":
                candidates = [row for row in ladder_rows if row["lcb95_eligible"]]
                selection_fallback = (
                    "none" if candidates else "no_lcb95_qualified_config"
                )
            else:
                candidates = mean_candidates
                selection_fallback = "none" if candidates else "no_mean_qualified_config"
            if candidates:
                selection_pool = candidates
                winner = min(
                    selection_pool,
                    key=lambda row: (float(row["latency_mean_ms"]), int(row["ef_search"])),
                )
                winner["selected"] = True
                selected[(spec.name, target)] = int(winner["ef_search"])
            for row in ladder_rows:
                if not candidates and ladder_complete and selection_policy == "lcb_then_max_recall":
                    row["outcome"] = "lcb95_unattained_on_grid"
                    row["selection_status"] = "no_config_meets_lcb95"
                elif not candidates and ladder_complete:
                    row["outcome"] = "unattainable_on_grid"
                    row["selection_status"] = "unattainable_on_grid"
                elif not candidates:
                    row["outcome"] = "calibration_invalid"
                    row["selection_status"] = "no_config_meets_mean"
                elif row["selected"]:
                    row["outcome"] = "selected_pending_final"
                    row["selection_status"] = "selected"
                elif row["eligible"]:
                    row["outcome"] = "selected_pending_final"
                    row["selection_status"] = "eligible_not_selected"
                else:
                    row["outcome"] = "selected_pending_final"
                    row["selection_status"] = "ineligible"
                row["selected_ef_search"] = selected.get((spec.name, target), NA)
                row["selection_fallback"] = selection_fallback
                row["calibration_ladder_complete"] = ladder_complete
                row["max_ef_search"] = max_row["ef_search"] if max_row else NA
                row["max_observed_recall_mean"] = (
                    max(float(item["recall_mean"]) for item in observed_metrics)
                    if observed_metrics else NA
                )
                row["max_observed_recall_lcb95"] = (
                    max(float(item["recall_lcb95"]) for item in observed_metrics)
                    if observed_metrics else NA
                )
                row["full_ladder_proof"] = json.dumps(ladder_proof, sort_keys=True)
    return rows, selected


def paired_speedup_bounds(
    sql_rows: Sequence[dict[str, Any]],
    faiss_rows: Sequence[dict[str, Any]],
    query_nos: Sequence[int],
    repeats: int,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    sql_by_pair = {
        (int(row["query_no"]), int(row["repeat"])): float(row["search_latency_ms"])
        for row in sql_rows
    }
    faiss_by_pair = {
        (int(row["query_no"]), int(row["repeat"])): float(row["search_latency_ms"])
        for row in faiss_rows
    }
    sql_query = {
        int(query_no): statistics.fmean(sql_by_pair[(int(query_no), repeat)] for repeat in range(repeats))
        for query_no in query_nos
    }
    faiss_query = {
        int(query_no): statistics.fmean(
            faiss_by_pair[(int(query_no), repeat)] for repeat in range(repeats)
        )
        for query_no in query_nos
    }
    point = statistics.fmean(sql_query.values()) / statistics.fmean(faiss_query.values())
    rng = random.Random(seed)
    query_list = [int(query_no) for query_no in query_nos]
    values: list[float] = []
    for _ in range(max(1, samples)):
        chosen = rng.choices(query_list, k=len(query_list)) if len(query_list) > 1 else query_list
        sql_mean = statistics.fmean(sql_query[query_no] for query_no in chosen)
        faiss_mean = statistics.fmean(faiss_query[query_no] for query_no in chosen)
        values.append(sql_mean / faiss_mean)
    return point, percentile(values, 0.025), percentile(values, 0.975)


def final_summary_table(
    final_rows: Sequence[dict[str, Any]],
    filter_specs: Sequence[FilterSpec],
    targets: Sequence[float],
    selected: dict[tuple[str, float], int],
    query_nos: Sequence[int] | Mapping[str, Sequence[int]],
    repeats: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    allow_lists: dict[str, AllowList] | None = None,
    calibration_outcomes: dict[tuple[str, float], str] | None = None,
    methods: Sequence[str] | None = None,
    setup_search_rows: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Summarize each requested arm without conflating timing boundaries."""
    output: list[dict[str, Any]] = []
    if methods is None:
        observed = {str(row.get("method")) for row in final_rows}
        requested_methods = tuple(
            method for method in FORMAL_METHODS if method in observed
        )
        if (
            FAISS_METHOD not in requested_methods
            and (selected or calibration_outcomes)
        ):
            requested_methods = (*requested_methods, FAISS_METHOD)
    else:
        requested_methods = tuple(methods)
    exact_methods = tuple(
        method for method in requested_methods if method in SQL_FIRST_METHODS
    )
    faiss_requested = FAISS_METHOD in requested_methods
    reference_method = (
        SQL_FIRST_PLANNER_METHOD
        if SQL_FIRST_PLANNER_METHOD in exact_methods
        else SQL_FIRST_CONTROL_METHOD
        if SQL_FIRST_CONTROL_METHOD in exact_methods
        else SQL_FIRST_FORCED_METHOD
        if SQL_FIRST_FORCED_METHOD in exact_methods
        else None
    )

    def unavailable_faiss_stats(outcome: str) -> dict[str, Any]:
        exhausted = outcome in {
            "unattainable_on_grid",
            "lcb95_unattained_on_grid",
        }
        return {
            "status": "valid" if exhausted else "invalid",
            "rows_complete": exhausted,
            "samples": 0,
            "expected_samples": 0,
            "missing_pairs": 0 if exhausted else sum(
                len(values) for values in query_nos.values()
            ) * repeats
            if isinstance(query_nos, Mapping)
            else len(query_nos) * repeats,
            "errors": 0,
            "recall_mean": NA,
            "recall_lcb95": NA,
            "latency_mean_ms": NA,
            "latency_p50_ms": NA,
            "latency_p95_ms": NA,
            "latency_p99_ms": NA,
            "latency_query_mean_ci95_low_ms": NA,
            "latency_query_mean_ci95_high_ms": NA,
        }

    for filter_no, spec in enumerate(filter_specs):
        spec_query_nos = (
            list(query_nos[spec.name])
            if isinstance(query_nos, Mapping)
            else list(query_nos)
        )
        sql_stats_by_method = {
            method: aggregate_measurements(
                final_rows,
                phase="final",
                method=method,
                filter_name=spec.name,
                ef_search=None,
                query_nos=spec_query_nos,
                repeats=repeats,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed
                + filter_no * 1009
                + method_no * 131,
            )
            for method_no, method in enumerate(exact_methods)
        }
        for target_no, target in enumerate(targets):
            ef_search = selected.get((spec.name, target))
            calibration_outcome = (calibration_outcomes or {}).get(
                (spec.name, target),
                "selected_pending_final" if ef_search is not None else "calibration_invalid",
            )
            faiss_stats: dict[str, Any] | None = None
            if faiss_requested and ef_search is None:
                faiss_stats = unavailable_faiss_stats(calibration_outcome)
            elif faiss_requested:
                faiss_stats = aggregate_measurements(
                    final_rows,
                    phase="final",
                    method=FAISS_METHOD,
                    filter_name=spec.name,
                    ef_search=ef_search,
                    query_nos=spec_query_nos,
                    repeats=repeats,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed + filter_no * 1009 + target_no * 104729,
                    target_recall=target,
                )
            reference_stats = (
                sql_stats_by_method.get(reference_method)
                if reference_method is not None
                else None
            )
            matched_pairs = bool(
                faiss_requested
                and reference_stats is not None
                and faiss_stats is not None
                and ef_search is not None
                and reference_stats["status"] == "valid"
                and faiss_stats["status"] == "valid"
            )
            matched_comparison_confirmed = bool(
                matched_pairs
                and float(reference_stats["recall_lcb95"]) >= target
                and float(faiss_stats["recall_lcb95"]) >= target
            )
            allow_list = (allow_lists or {}).get(spec.name)
            full_e2e_values = [
                float(row["continuous_full_e2e_ms"])
                for row in (*setup_search_rows, *final_rows)
                if row.get("filter_name") == spec.name
                and row.get("method", FAISS_METHOD) == FAISS_METHOD
                and int(row.get("ef_search", -1)) == ef_search
                and (
                    row.get("target_recall") in (None, "", NA)
                    or math.isclose(float(row["target_recall"]), target)
                )
                and _row_ok(row)
                and (
                    row.get("continuous_full_e2e_valid") in (None, "", NA)
                    or _bool_value(row.get("continuous_full_e2e_valid"))
                )
                and row.get("continuous_full_e2e_ms") not in (None, "", NA)
            ] if ef_search is not None else []
            full_e2e_expected = len(spec_query_nos) * repeats
            full_e2e_rows = [
                row
                for row in final_rows
                if row.get("method") == FAISS_METHOD
                and row.get("filter_name") == spec.name
                and ef_search is not None
                and int(row.get("ef_search", -1)) == ef_search
                and row.get("target_recall") not in (None, "", NA)
                and math.isclose(float(row["target_recall"]), target)
            ]
            full_e2e_errors = (
                sum(
                    1
                    for row in full_e2e_rows
                    if (
                        not _bool_value(row.get("continuous_full_e2e_valid"))
                        or row.get("continuous_full_e2e_ms") in (None, "", NA)
                    )
                )
                if ef_search is not None
                else 0
            )
            full_e2e_pairs = {
                (int(row["query_no"]), int(row["repeat"])): row
                for row in full_e2e_rows
            }
            expected_full_e2e_pairs = {
                (int(query_no), repeat)
                for query_no in spec_query_nos
                for repeat in range(repeats)
            }
            full_e2e_complete = bool(
                len(full_e2e_rows) == full_e2e_expected
                and len(full_e2e_pairs) == full_e2e_expected
                and set(full_e2e_pairs) == expected_full_e2e_pairs
                and full_e2e_errors == 0
            )
            if full_e2e_complete:
                continuous_query_recalls = [
                    statistics.fmean(
                        float(
                            full_e2e_pairs[(int(query_no), repeat)][
                                "continuous_recall_at_10"
                            ]
                        )
                        for repeat in range(repeats)
                    )
                    for query_no in spec_query_nos
                ]
                (
                    continuous_recall_lcb95,
                    _continuous_recall_ci_low,
                    _continuous_recall_ci_high,
                ) = bootstrap_mean_bounds(
                    continuous_query_recalls,
                    bootstrap_samples,
                    bootstrap_seed
                    + filter_no * 1009
                    + target_no * 104729
                    + 17,
                )
                continuous_recall_mean = statistics.fmean(
                    continuous_query_recalls
                )
            else:
                continuous_recall_mean = NA
                continuous_recall_lcb95 = NA

            def full_e2e_stage_mean(field: str) -> float | str:
                values = [
                    float(row[field])
                    for row in full_e2e_rows
                    if _bool_value(row.get("continuous_full_e2e_valid"))
                    and row.get(field) not in (None, "", NA)
                ]
                return statistics.fmean(values) if values else NA

            common = {
                "filter_name": spec.name,
                "target_rate": spec.target_rate,
                "predicate": spec.predicate,
                "actual_selectivity": spec.actual_pct / 100.0,
                "target_recall": target,
                "selected_faiss_ef_search": ef_search if ef_search is not None else NA,
                "matched_pairs_valid": matched_pairs,
                "matched_recall_comparison_valid": matched_comparison_confirmed,
                "comparison_status": (
                    "valid"
                    if matched_comparison_confirmed
                    else "unattainable_on_grid"
                    if calibration_outcome == "unattainable_on_grid"
                    else "lcb95_unattained_on_grid"
                    if calibration_outcome == "lcb95_unattained_on_grid"
                    else "target_unconfirmed"
                    if matched_pairs
                    else "not_a_cross_method_comparison"
                ),
            }
            method_stats = [
                (method, sql_stats_by_method[method]) for method in exact_methods
            ]
            if faiss_requested and faiss_stats is not None:
                method_stats.append((FAISS_METHOD, faiss_stats))
            for method, stats in method_stats:
                exhausted = bool(
                    method == FAISS_METHOD
                    and calibration_outcome
                    in {"unattainable_on_grid", "lcb95_unattained_on_grid"}
                )
                artifact_row_valid = stats["status"] == "valid"
                metrics_valid = artifact_row_valid and not exhausted
                method_target_confirmed = bool(
                    metrics_valid and float(stats["recall_lcb95"]) >= target
                )
                method_outcome = (
                    calibration_outcome
                    if exhausted
                    else "selected_and_confirmed"
                    if method_target_confirmed
                    else "selected_but_final_unconfirmed"
                    if artifact_row_valid
                    else "invalid"
                )
                speedup: float | str = NA
                speedup_low: float | str = NA
                speedup_high: float | str = NA
                if (
                    SQL_FIRST_PLANNER_METHOD in exact_methods
                    and reference_stats is not None
                    and reference_stats["status"] == "valid"
                    and float(reference_stats["recall_lcb95"]) >= target
                    and method_target_confirmed
                ):
                    if method == SQL_FIRST_PLANNER_METHOD:
                        speedup = speedup_low = speedup_high = 1.0
                    else:
                        reference_items = [
                            row
                            for row in final_rows
                            if row.get("phase") == "final"
                            and row.get("method") == SQL_FIRST_PLANNER_METHOD
                            and row.get("filter_name") == spec.name
                        ]
                        method_items = [
                            row
                            for row in final_rows
                            if row.get("phase") == "final"
                            and row.get("method") == method
                            and row.get("filter_name") == spec.name
                            and (
                                method != FAISS_METHOD
                                or int(row["ef_search"]) == ef_search
                            )
                            and (
                                method != FAISS_METHOD
                                or row.get("target_recall") in (None, "", NA)
                                or math.isclose(
                                    float(row["target_recall"]), target
                                )
                            )
                        ]
                        speedup, speedup_low, speedup_high = paired_speedup_bounds(
                            reference_items,
                            method_items,
                            spec_query_nos,
                            repeats,
                            bootstrap_samples,
                            bootstrap_seed
                            + filter_no * 1009
                            + target_no * 104729
                            + FORMAL_METHODS.index(method),
                        )
                output.append(
                    {
                        **common,
                        "method": method,
                        "method_description": METHOD_DESCRIPTIONS[method],
                        "outcome": method_outcome,
                        "status": "valid" if artifact_row_valid else "invalid",
                        "queries": (
                            0 if exhausted
                            else len(spec_query_nos) if artifact_row_valid else NA
                        ),
                        "samples": (
                            0 if exhausted
                            else len(spec_query_nos) * repeats if artifact_row_valid else NA
                        ),
                        "expected_samples": (
                            0 if exhausted
                            else len(spec_query_nos) * repeats if artifact_row_valid else NA
                        ),
                        "recall_mean": stats["recall_mean"] if metrics_valid else NA,
                        "recall_lcb95": stats["recall_lcb95"] if metrics_valid else NA,
                        "search_latency_mean_ms": stats["latency_mean_ms"] if metrics_valid else NA,
                        "search_latency_p50_ms": stats.get("latency_p50_ms", NA) if metrics_valid else NA,
                        "search_latency_p95_ms": stats.get("latency_p95_ms", NA) if metrics_valid else NA,
                        "search_latency_p99_ms": stats.get("latency_p99_ms", NA) if metrics_valid else NA,
                        "search_latency_mean_ci95_low_ms": (
                            stats.get("latency_query_mean_ci95_low_ms", NA) if metrics_valid else NA
                        ),
                        "search_latency_mean_ci95_high_ms": (
                            stats.get("latency_query_mean_ci95_high_ms", NA) if metrics_valid else NA
                        ),
                        "cached_allowlist_search_mean_ms": (
                            stats["latency_mean_ms"]
                            if metrics_valid and method == FAISS_METHOD
                            else NA
                        ),
                        "continuous_full_e2e_mean_ms": (
                            statistics.fmean(full_e2e_values)
                            if method == FAISS_METHOD and full_e2e_values
                            else NA
                        ),
                        "continuous_full_e2e_samples": (
                            len(full_e2e_values) if method == FAISS_METHOD else NA
                        ),
                        "continuous_full_e2e_expected_samples": (
                            full_e2e_expected if method == FAISS_METHOD and not exhausted else 0
                            if method == FAISS_METHOD
                            else NA
                        ),
                        "continuous_full_e2e_errors": (
                            full_e2e_errors if method == FAISS_METHOD else NA
                        ),
                        "continuous_full_e2e_complete": (
                            full_e2e_complete
                            if method == FAISS_METHOD and not exhausted
                            else False
                            if method == FAISS_METHOD
                            else NA
                        ),
                        "continuous_recall_mean": (
                            continuous_recall_mean
                            if method == FAISS_METHOD
                            else NA
                        ),
                        "continuous_recall_lcb95": (
                            continuous_recall_lcb95
                            if method == FAISS_METHOD
                            else NA
                        ),
                        "continuous_full_e2e_p95_ms": (
                            percentile(full_e2e_values, 0.95)
                            if method == FAISS_METHOD and full_e2e_values
                            else NA
                        ),
                        "continuous_full_e2e_p99_ms": (
                            percentile(full_e2e_values, 0.99)
                            if method == FAISS_METHOD and full_e2e_values
                            else NA
                        ),
                        "per_request_allowlist_sql_materialization_mean_ms": (
                            full_e2e_stage_mean(
                                "per_request_allowlist_sql_materialization_ms"
                            )
                            if method == FAISS_METHOD
                            else NA
                        ),
                        "per_request_allowlist_row_transfer_mean_ms": (
                            full_e2e_stage_mean(
                                "per_request_allowlist_row_transfer_ms"
                            )
                            if method == FAISS_METHOD
                            else NA
                        ),
                        "per_request_allowlist_bitmap_build_mean_ms": (
                            full_e2e_stage_mean(
                                "per_request_allowlist_bitmap_build_ms"
                            )
                            if method == FAISS_METHOD
                            else NA
                        ),
                        "per_request_allowlist_selector_construction_mean_ms": (
                            full_e2e_stage_mean(
                                "per_request_allowlist_selector_construction_ms"
                            )
                            if method == FAISS_METHOD
                            else NA
                        ),
                        "per_request_allowlist_full_setup_mean_ms": (
                            full_e2e_stage_mean(
                                "per_request_allowlist_full_setup_ms"
                            )
                            if method == FAISS_METHOD
                            else NA
                        ),
                        "per_request_full_path_search_mean_ms": (
                            full_e2e_stage_mean(
                                "per_request_full_path_search_ms"
                            )
                            if method == FAISS_METHOD
                            else NA
                        ),
                        "target_confirmed_in_final": (
                            method_target_confirmed
                            if metrics_valid
                            else False
                            if exhausted
                            else NA
                        ),
                        "speedup_vs_sql_first_planner_chosen_exact": speedup,
                        "speedup_vs_sql_first_exact": speedup,
                        "speedup_ci95_low": speedup_low,
                        "speedup_ci95_high": speedup_high,
                        "allowlist_build_rows": (
                            allow_list.rows if method == FAISS_METHOD and allow_list else NA
                        ),
                        "allowlist_build_ms_one_time": (
                            allow_list.build_ms if method == FAISS_METHOD and allow_list else NA
                        ),
                        "allowlist_sql_materialization_ms_one_time": (
                            allow_list.server_execution_ms
                            if method == FAISS_METHOD and allow_list
                            else NA
                        ),
                        "allowlist_server_execution_ms_one_time": (
                            allow_list.server_execution_ms
                            if method == FAISS_METHOD and allow_list
                            else NA
                        ),
                        "allowlist_row_transfer_ms_one_time": (
                            allow_list.row_transfer_ms
                            if method == FAISS_METHOD and allow_list
                            else NA
                        ),
                        "allowlist_bitmap_build_ms_one_time": (
                            allow_list.bitmap_construction_ms
                            if method == FAISS_METHOD and allow_list
                            else NA
                        ),
                        "allowlist_bitmap_construction_ms_one_time": (
                            allow_list.bitmap_construction_ms
                            if method == FAISS_METHOD and allow_list
                            else NA
                        ),
                        "allowlist_selector_construction_ms_one_time": (
                            allow_list.selector_construction_ms
                            if method == FAISS_METHOD and allow_list
                            else NA
                        ),
                        "allowlist_full_setup_ms_one_time": (
                            allow_list.full_setup_ms
                            if method == FAISS_METHOD and allow_list
                            else NA
                        ),
                        "allowlist_bitmap_bytes": (
                            allow_list.bitmap_bytes if method == FAISS_METHOD and allow_list else NA
                        ),
                        "missing_pairs": stats.get("missing_pairs", NA),
                        "errors": stats.get("errors", NA),
                    }
                )
    return output


def calibration_outcomes_from_rows(
    calibration_rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, float], str]:
    outcomes: dict[tuple[str, float], str] = {}
    for row in calibration_rows:
        key = (str(row["filter_name"]), float(row["target_recall"]))
        outcome = str(row.get("outcome", ""))
        if not outcome:
            raise ValueError(f"calibration row has no outcome: {key}")
        previous = outcomes.setdefault(key, outcome)
        if previous != outcome:
            raise ValueError(f"inconsistent calibration outcome for {key}")
    return outcomes


def artifact_validation_errors(
    calibration_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    filter_specs: Sequence[FilterSpec],
    ef_values: Sequence[int],
    targets: Sequence[float],
    methods: Sequence[str] | None = None,
) -> list[str]:
    """Validate integrity; an exhausted LCB grid is reportable but never publishable."""
    errors: list[str] = []
    observed_methods = {str(row.get("method")) for row in summary_rows}
    requested_methods = tuple(methods) if methods is not None else tuple(
        method for method in FORMAL_METHODS if method in observed_methods
    )
    expected_calibration = {
        (spec.name, float(target), int(ef))
        for spec in filter_specs
        for target in targets
        for ef in ef_values
    } if FAISS_METHOD in requested_methods else set()
    observed_calibration: set[tuple[str, float, int]] = set()
    for row in calibration_rows:
        try:
            key = (str(row["filter_name"]), float(row["target_recall"]), int(row["ef_search"]))
        except (KeyError, TypeError, ValueError):
            errors.append("calibration row has an invalid key")
            continue
        if key in observed_calibration:
            errors.append(f"duplicate calibration key: {key}")
        observed_calibration.add(key)
        if row.get("status") != "valid":
            errors.append(f"invalid calibration grid cell: {key}")
    if observed_calibration != expected_calibration:
        errors.append("calibration key coverage is incomplete or contains extras")

    expected_summary = {
        (spec.name, float(target), method)
        for spec in filter_specs
        for target in targets
        for method in requested_methods
    }
    observed_summary: set[tuple[str, float, str]] = set()
    accepted_outcomes = {
        "selected_and_confirmed",
        "selected_but_final_unconfirmed",
        "unattainable_on_grid",
        "lcb95_unattained_on_grid",
    }
    for row in summary_rows:
        try:
            key = (str(row["filter_name"]), float(row["target_recall"]), str(row["method"]))
        except (KeyError, TypeError, ValueError):
            errors.append("summary row has an invalid key")
            continue
        if key in observed_summary:
            errors.append(f"duplicate summary key: {key}")
        observed_summary.add(key)
        if row.get("status") != "valid":
            errors.append(f"invalid selected-config final: {key}")
        if row.get("outcome") not in accepted_outcomes:
            errors.append(f"invalid outcome: {key}")
    if observed_summary != expected_summary:
        errors.append("summary key coverage is incomplete or contains extras")
    return errors


def completion_gate(
    calibration_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    filter_specs: Sequence[FilterSpec],
    ef_values: Sequence[int],
    targets: Sequence[float],
    methods: Sequence[str] | None = None,
    protocol: str = LEGACY_PROTOCOL,
) -> dict[str, Any]:
    """Separate a finished requested slice from a publishable all-target release."""
    observed_methods = {str(row.get("method")) for row in summary_rows}
    requested_methods = tuple(methods) if methods is not None else tuple(
        method for method in FORMAL_METHODS if method in observed_methods
    )
    expected_calibration_cells = (
        len(filter_specs) * len(ef_values) * len(targets)
        if FAISS_METHOD in requested_methods
        else 0
    )
    expected_summary_cells = (
        len(filter_specs) * len(targets) * len(requested_methods)
    )
    calibration_complete = (
        len(calibration_rows) == expected_calibration_cells
        and all(row.get("status") == "valid" for row in calibration_rows)
    )
    final_slice_complete = (
        len(summary_rows) == expected_summary_cells
        and all(row.get("status") == "valid" for row in summary_rows)
    )
    requested_slice_complete = calibration_complete and final_slice_complete
    matched_rows = [
        row for row in summary_rows if row.get("method") == FAISS_METHOD
    ]
    full_release_complete = bool(
        requested_slice_complete
        and SQL_FIRST_PLANNER_METHOD in requested_methods
        and FAISS_METHOD in requested_methods
        and len(filter_specs) == len(FORMAL_FILTER_NAMES)
        and tuple(spec.name for spec in filter_specs) == FORMAL_FILTER_NAMES
        and tuple(ef_values) == DEFAULT_EF_SEARCH
        and tuple(float(target) for target in targets) == DEFAULT_TARGETS
        and len(matched_rows) == len(filter_specs) * len(targets)
        and all(
            row.get("outcome") == "selected_and_confirmed"
            and row.get("matched_recall_comparison_valid") is True
            and row.get("target_confirmed_in_final") is True
            and (
                protocol != CURRENT_PROTOCOL
                or int(row.get("continuous_full_e2e_samples", -1))
                == int(row.get("continuous_full_e2e_expected_samples", -2))
            )
            and (
                protocol != CURRENT_PROTOCOL
                or int(row.get("continuous_full_e2e_errors", -1)) == 0
            )
            and (
                protocol != CURRENT_PROTOCOL
                or row.get("continuous_full_e2e_complete") is True
            )
            and (
                protocol != CURRENT_PROTOCOL
                or float(row.get("continuous_recall_lcb95", -1.0))
                >= float(row["target_recall"])
            )
            for row in matched_rows
        )
    )
    return {
        "checkpoint_protocol_version": (
            CHECKPOINT_PROTOCOL_VERSION
            if protocol == CURRENT_PROTOCOL
            else LEGACY_CHECKPOINT_PROTOCOL_VERSION
        ),
        "protocol": protocol,
        "expected_calibration_cells": expected_calibration_cells,
        "observed_calibration_cells": len(calibration_rows),
        "expected_summary_cells": expected_summary_cells,
        "observed_summary_cells": len(summary_rows),
        "requested_methods": list(requested_methods),
        "calibration_complete": calibration_complete,
        "requested_slice_complete": requested_slice_complete,
        "full_release_complete": full_release_complete,
        "publishable_matched_recall": full_release_complete,
        "status": "complete" if requested_slice_complete else "incomplete",
    }


def artifact_validity_flags(
    validation_errors: Sequence[str],
    completion: Mapping[str, Any],
    *,
    formal_provenance_valid: bool = True,
) -> dict[str, bool]:
    """Keep a completed diagnostic slice distinct from a paper-ready release."""
    diagnostic_valid = not validation_errors
    artifact_valid = bool(
        diagnostic_valid and completion.get("requested_slice_complete") is True
    )
    return {
        "diagnostic_valid": diagnostic_valid,
        "artifact_valid": artifact_valid,
        "paper_eligible": bool(
            artifact_valid
            and formal_provenance_valid
            and completion.get("publishable_matched_recall") is True
        ),
    }


def prefetch_sql_query_vectors(
    cursor: Any,
    table: str,
    query_ids: Iterable[int],
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[int, str]:
    wanted = sorted(set(int(query_id) for query_id in query_ids))
    cursor.execute(
        f"SELECT id, embedding::text FROM {table} WHERE id = ANY(%s::bigint[]) "
        f"AND ({validate_candidate_validity_predicate(candidate_validity_predicate)})",
        (wanted,),
    )
    vectors = {int(row[0]): str(row[1]) for row in cursor.fetchall()}
    missing = set(wanted) - set(vectors)
    if missing:
        raise ValueError(f"query vector prefetch missing IDs: {sorted(missing)}")
    return vectors


def postgres_software_identity(cursor: Any) -> dict[str, str]:
    """Fail closed when the serving pgvector binary cannot be identified."""
    try:
        cursor.execute(
            "WITH lib AS ("
            "SELECT setting || '/vector.so' AS path "
            "FROM pg_config WHERE name = 'PKGLIBDIR'"
            ") SELECT path, encode(sha256(pg_read_binary_file(path)), 'hex') FROM lib"
        )
        row = cursor.fetchone()
    except Exception as exc:
        raise RuntimeError("could not bind the serving vector.so identity") from exc
    if not row:
        raise RuntimeError("pg_config did not expose the serving vector.so path")
    path, digest = str(row[0]), str(row[1])
    if not path.endswith("/vector.so") or len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest.lower()
    ):
        raise RuntimeError("serving vector.so identity is invalid")
    return {"vector_so_path": path, "vector_so_sha256": digest}


def explain_exact_plan(
    cursor: Any,
    table: str,
    spec: FilterSpec,
    query_vector: str,
    k: int,
    hnsw_indexes: Sequence[str],
    *,
    method: str = SQL_FIRST_PLANNER_METHOD,
    scalar_indexes: Sequence[str] = (),
) -> dict[str, Any]:
    if method not in SQL_FIRST_METHODS:
        raise ValueError(f"unknown SQL-first method: {method}")
    planner_mode = (
        "forced_indexed" if method == SQL_FIRST_FORCED_METHOD else "auto"
    )
    cursor.execute(
        "RESET enable_seqscan"
        if planner_mode == "auto"
        else "SET enable_seqscan = off"
    )
    try:
        cursor.execute(
            "EXPLAIN (FORMAT JSON, COSTS OFF) "
            + exact_sql_for_method(method, table, spec.predicate, k),
            (-1, query_vector),
        )
        plan = decode_explain(cursor.fetchone()[0])
        used_indexes = (
            assert_no_hnsw_index(plan, hnsw_indexes)
            if planner_mode == "auto"
            else assert_scalar_index_plan(plan, hnsw_indexes, scalar_indexes)
        )
    finally:
        if planner_mode == "forced_indexed":
            cursor.execute("RESET enable_seqscan")
    return {
        "method": method,
        "planner_mode": planner_mode,
        "sql_shape": (
            "materialized_cte_control"
            if method == SQL_FIRST_CONTROL_METHOD
            else "direct_exact_sql"
        ),
        "used_indexes": sorted(used_indexes),
        "node_types": sorted(plan_node_types(plan)),
        "hnsw_indexes": sorted(hnsw_indexes),
        "registered_scalar_indexes": sorted(scalar_indexes),
        "scalar_index_proven": bool(
            planner_mode == "forced_indexed"
            and {
                name.rsplit(".", 1)[-1] for name in used_indexes
            }
            & {
                name.rsplit(".", 1)[-1] for name in scalar_indexes
            }
        ),
        "hnsw_absent": True,
        "plan": plan,
    }


def faiss_index_metadata(index: Any, faiss_module: Any, rows: int, dimensions: int) -> dict[str, Any]:
    if not hasattr(index, "hnsw"):
        raise ValueError(f"Faiss index is not an ordinary HNSW index: {type(index).__name__}")
    if int(index.ntotal) != rows or int(index.d) != dimensions:
        raise ValueError(
            f"Faiss/fbin mismatch: index=({index.ntotal}, {index.d}) fbin=({rows}, {dimensions})"
        )
    if int(index.metric_type) != int(faiss_module.METRIC_L2):
        raise ValueError("Faiss HNSW index must use L2 distance")
    level0_neighbors = int(index.hnsw.nb_neighbors(0))
    level1_neighbors = int(index.hnsw.nb_neighbors(1))
    ef_construction = int(index.hnsw.efConstruction)
    if (level0_neighbors, level1_neighbors) != (64, 32):
        raise ValueError(
            "Faiss HNSW index is not formal M32: "
            f"level0_neighbors={level0_neighbors} level1_neighbors={level1_neighbors}"
        )
    if ef_construction != FORMAL_HNSW_EF_CONSTRUCTION:
        raise ValueError(
            "Faiss HNSW index is not formal efConstruction=200: "
            f"actual={ef_construction}"
        )
    return {
        "type": type(index).__name__,
        "ntotal": int(index.ntotal),
        "dimensions": int(index.d),
        "metric": "L2",
        "m": FORMAL_HNSW_M,
        "ef_construction": ef_construction,
        "level0_neighbors": level0_neighbors,
        "level1_neighbors": level1_neighbors,
    }


def verify_faiss_build_manifest(
    manifest_path: Path,
    index_identity: Mapping[str, Any],
    fbin_identity: Mapping[str, Any],
    rows: int,
    dimensions: int,
) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Faiss build manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Faiss build manifest must contain a JSON object")
    configuration = payload.get("configuration")
    contract = payload.get("index_contract")
    output = payload.get("output_identity")
    inputs = payload.get("inputs")
    fbin = inputs.get("fbin") if isinstance(inputs, Mapping) else None
    passed = (
        payload.get("artifact") == "faiss_hnsw_index_build"
        and payload.get("status") == "complete"
        and payload.get("artifact_valid") is True
        and isinstance(configuration, Mapping)
        and isinstance(contract, Mapping)
        and isinstance(output, Mapping)
        and isinstance(fbin, Mapping)
        and int(configuration.get("m", 0)) == FORMAL_HNSW_M
        and int(configuration.get("ef_construction", 0)) == FORMAL_HNSW_EF_CONSTRUCTION
        and int(configuration.get("rows", 0)) == rows
        and int(configuration.get("dimensions", 0)) == dimensions
        and contract.get("type") == "IndexHNSWFlat"
        and contract.get("metric") == "l2"
        and int(contract.get("m", 0)) == FORMAL_HNSW_M
        and int(contract.get("ef_construction", 0)) == FORMAL_HNSW_EF_CONSTRUCTION
        and int(contract.get("rows", 0)) == rows
        and int(contract.get("dimensions", 0)) == dimensions
        and output.get("sha256") == index_identity.get("sha256")
        and int(output.get("size_bytes", -1)) == int(index_identity.get("bytes", -2))
        and fbin.get("sha256") == fbin_identity.get("sha256")
        and int(fbin.get("size_bytes", -1)) == int(fbin_identity.get("bytes", -2))
    )
    if not passed:
        raise ValueError("Faiss M32/efConstruction=200 build manifest contract failed")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "artifact_valid": True,
        "configuration": dict(configuration),
        "index_contract": dict(contract),
        "output_identity": dict(output),
        "fbin_identity": dict(fbin),
    }


def _run_faiss_measurement(
    raw_rows: list[dict[str, Any]],
    *,
    phase: str,
    spec: FilterSpec,
    query_no: int,
    query_id: int,
    query: Any,
    repeat: int,
    schedule_position: int,
    block_no: int,
    ef_search: int,
    truth_entry: TruthEntry,
    vectors: Any,
    allow_list: AllowList,
    index: Any,
    faiss_module: Any,
    k: int,
    matched_target_recalls: Sequence[float] = (),
) -> None:
    try:
        if not allow_list.valid or allow_list.selector is None:
            raise RuntimeError(allow_list.error or "allow-list setup is invalid")
        ids, latency_ms = search_faiss(
            index,
            faiss_module,
            query,
            allow_list.selector,
            ef_search,
            k,
            query_id=query_id,
        )
        if allow_list.bitmap is None:
            raise RuntimeError("allow-list bitmap is unavailable for membership validation")
        invalid_ids = result_membership_errors(allow_list.bitmap, ids)
        if invalid_ids:
            raise RuntimeError(f"faiss returned IDs outside allow-list: {invalid_ids[:5]}")
        raw_rows.append(
            measurement_row(
                phase=phase,
                method=FAISS_METHOD,
                spec=spec,
                query_no=query_no,
                query_id=query_id,
                repeat=repeat,
                schedule_position=schedule_position,
                block_no=block_no,
                ef_search=ef_search,
                result_ids=ids,
                truth_ids=truth_entry.ids,
                truth_entry=truth_entry,
                vectors=vectors,
                latency_ms=latency_ms,
                matched_target_recalls=matched_target_recalls,
            )
        )
    except Exception as exc:
        raw_rows.append(
            measurement_row(
                phase=phase,
                method=FAISS_METHOD,
                spec=spec,
                query_no=query_no,
                query_id=query_id,
                repeat=repeat,
                schedule_position=schedule_position,
                block_no=block_no,
                ef_search=ef_search,
                result_ids=None,
                truth_ids=truth_entry.ids,
                latency_ms=NA,
                error=f"{exc.__class__.__name__}: {exc}",
                matched_target_recalls=matched_target_recalls,
            )
        )


def run_calibration(
    raw_rows: list[dict[str, Any]],
    *,
    filter_specs: Sequence[FilterSpec],
    query_nos: Sequence[int],
    repeats: int,
    ef_values: Sequence[int],
    query_ids: dict[int, int],
    query_vectors: dict[int, Any],
    truth: dict[tuple[str, int], TruthEntry],
    allow_lists: dict[str, AllowList],
    index: Any,
    faiss_module: Any,
    vectors: Any,
    k: int,
    schedule_seed: int,
    progress_queries: int,
    checkpoint_path: Path | None = None,
) -> None:
    block_no = 0
    for filter_no, spec in enumerate(filter_specs):
        allow_list = allow_lists[spec.name]
        completed = 0
        for repeat in range(repeats):
            ordered_queries = list(query_nos)
            random.Random(schedule_seed + filter_no * 1009 + repeat * 104729).shuffle(ordered_queries)
            for query_no in ordered_queries:
                order = balanced_order(ef_values, block_no, schedule_seed)
                query_id = query_ids[int(query_no)]
                for position, ef_search in enumerate(order, start=1):
                    _run_faiss_measurement(
                        raw_rows,
                        phase="calibration",
                        spec=spec,
                        query_no=int(query_no),
                        query_id=query_id,
                        query=query_vectors[int(query_no)],
                        repeat=repeat,
                        schedule_position=position,
                        block_no=block_no,
                        ef_search=int(ef_search),
                        truth_entry=truth[(spec.name, int(query_no))],
                        vectors=vectors,
                        allow_list=allow_list,
                        index=index,
                        faiss_module=faiss_module,
                        k=k,
                    )
                block_no += 1
                completed += 1
                if progress_queries and completed % progress_queries == 0:
                    print(
                        f"calibration filter={spec.name} queries={completed}/{len(query_nos) * repeats}",
                        flush=True,
                    )
        if checkpoint_path is not None:
            write_csv(checkpoint_path, raw_rows)


def run_final(
    raw_rows: list[dict[str, Any]],
    *,
    table: str,
    filter_specs: Sequence[FilterSpec],
    methods: Sequence[str],
    query_nos: Sequence[int],
    repeats: int,
    selected: dict[tuple[str, float], int],
    targets: Sequence[float],
    query_ids: dict[int, int],
    faiss_query_vectors: dict[int, Any],
    sql_query_vectors: dict[int, str],
    truth: dict[tuple[str, int], TruthEntry],
    allow_lists: dict[str, AllowList],
    exact_plan_valid: dict[str, bool],
    cursor: Any,
    index: Any,
    faiss_module: Any,
    vectors: Any,
    k: int,
    schedule_seed: int,
    progress_queries: int,
    checkpoint_path: Path | None = None,
) -> None:
    block_no = 0
    for filter_no, spec in enumerate(filter_specs):
        selected_efs = sorted(
            {
                selected[(spec.name, target)]
                for target in targets
                if (spec.name, target) in selected
            }
        )
        tasks: list[tuple[str, int | None]] = [
            (method, None) for method in methods if method in SQL_FIRST_METHODS
        ]
        if FAISS_METHOD in methods:
            tasks.extend((FAISS_METHOD, ef_search) for ef_search in selected_efs)
        completed = 0
        for repeat in range(repeats):
            ordered_queries = list(query_nos)
            random.Random(schedule_seed + 1_000_003 + filter_no * 1009 + repeat * 104729).shuffle(
                ordered_queries
            )
            for query_no in ordered_queries:
                query_no = int(query_no)
                query_id = query_ids[query_no]
                truth_entry = truth[(spec.name, query_no)]
                truth_ids = truth_entry.ids
                for position, (method, ef_search) in enumerate(
                    balanced_order(tasks, block_no, schedule_seed + 1), start=1
                ):
                    if method == FAISS_METHOD:
                        if ef_search is None:
                            raise RuntimeError(f"{FAISS_METHOD} final task is missing ef_search")
                        matched_targets = [
                            target
                            for target in targets
                            if selected.get((spec.name, target)) == ef_search
                        ]
                        _run_faiss_measurement(
                            raw_rows,
                            phase="final",
                            spec=spec,
                            query_no=query_no,
                            query_id=query_id,
                            query=faiss_query_vectors[query_no],
                            repeat=repeat,
                            schedule_position=position,
                            block_no=block_no,
                            ef_search=ef_search,
                            truth_entry=truth_entry,
                            vectors=vectors,
                            allow_list=allow_lists[spec.name],
                            index=index,
                            faiss_module=faiss_module,
                            k=k,
                            matched_target_recalls=matched_targets,
                        )
                        continue
                    try:
                        planner_mode = (
                            "forced_indexed"
                            if method == SQL_FIRST_FORCED_METHOD
                            else "auto"
                        )
                        plan_key = (spec.name, method)
                        if not exact_plan_valid.get(plan_key, False):
                            raise RuntimeError(
                                f"{method} EXPLAIN validation failed"
                            )
                        ids, latency_ms = search_sql_exact(
                            cursor,
                            exact_sql_for_method(method, table, spec.predicate, k),
                            query_id,
                            sql_query_vectors[query_id],
                            planner_mode,
                        )
                        allow_list = allow_lists.get(spec.name)
                        if allow_list is not None and allow_list.bitmap is not None:
                            invalid_ids = result_membership_errors(
                                allow_list.bitmap, ids
                            )
                            if invalid_ids:
                                raise RuntimeError(
                                    f"{method} returned IDs outside allow-list: {invalid_ids[:5]}"
                                )
                        recall = tie_aware_recall_at_k(ids, query_id, vectors, truth_entry, k)
                        error = "" if recall == 1.0 and len(ids) == k else (
                            f"exact_result_mismatch: recall={recall} returned={len(ids)}"
                        )
                        raw_rows.append(
                            measurement_row(
                                phase="final",
                                method=method,
                                spec=spec,
                                query_no=query_no,
                                query_id=query_id,
                                repeat=repeat,
                                schedule_position=position,
                                block_no=block_no,
                                ef_search=NA,
                                result_ids=ids,
                                truth_ids=truth_ids,
                                truth_entry=truth_entry,
                                vectors=vectors,
                                latency_ms=latency_ms,
                                error=error,
                                matched_target_recalls=targets,
                            )
                        )
                    except Exception as exc:
                        raw_rows.append(
                            measurement_row(
                                phase="final",
                                method=method,
                                spec=spec,
                                query_no=query_no,
                                query_id=query_id,
                                repeat=repeat,
                                schedule_position=position,
                                block_no=block_no,
                                ef_search=NA,
                                result_ids=None,
                                truth_ids=truth_ids,
                                latency_ms=NA,
                                error=f"{exc.__class__.__name__}: {exc}",
                                matched_target_recalls=targets,
                            )
                        )
                block_no += 1
                completed += 1
                if progress_queries and completed % progress_queries == 0:
                    print(
                        f"final filter={spec.name} queries={completed}/{len(query_nos) * repeats}",
                        flush=True,
                    )
        if checkpoint_path is not None:
            write_csv(checkpoint_path, raw_rows)


def output_paths(out_dir: Path, tag: str) -> dict[str, Path]:
    prefix = f"amazon10m_matched_recall_baselines_{tag}"
    return {
        "raw": out_dir / f"{prefix}_raw.csv",
        "calibration": out_dir / f"{prefix}_calibration.csv",
        "final": out_dir / f"{prefix}_final.csv",
        "summary": out_dir / f"{prefix}_summary.csv",
        "manifest": out_dir / f"{prefix}_manifest.json",
    }


def normalized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
    }


def checkpoint_cell_name(
    *,
    phase: str,
    repeat: int,
    method: str,
    ef_search: int | None = None,
    target_recall: float | None = None,
) -> str:
    parts = [phase, method]
    if ef_search is not None:
        parts.append(f"ef{ef_search}")
    if target_recall is not None:
        parts.append(f"target{target_recall:.2f}".replace(".", "p"))
    parts.append(f"r{repeat}")
    return "__".join(parts)


def validate_checkpoint_prefix(
    rows: Sequence[Mapping[str, Any]],
    requests: Sequence[WorkloadRequest],
    *,
    cell_id: str,
    repeat: int,
) -> None:
    if len(rows) > len(requests):
        raise ValueError(f"checkpoint {cell_id} contains too many rows")
    for position, row in enumerate(rows):
        request = requests[position]
        if (
            int(row.get("request_no", -1)) != request.request_no
            or int(row.get("query_no", -1)) != request.query_no
            or int(row.get("query_id", -1)) != request.query_id
            or str(row.get("filter_name", "")) != request.filter_name
            or int(row.get("repeat", -1)) != repeat
            or str(row.get("checkpoint_cell", "")) != cell_id
        ):
            raise ValueError(
                f"checkpoint {cell_id} is not a valid contiguous workload prefix "
                f"at position {position}"
            )


def execute_checkpointed_cell(
    *,
    checkpoint_dir: Path,
    cell_id: str,
    requests: Sequence[WorkloadRequest],
    repeat: int,
    checkpoint_every: int,
    resume: bool,
    execute_request: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = checkpoint_dir / f"{cell_id}.csv"
    rows: list[dict[str, Any]] = []
    resumed_rows = 0
    if path.exists():
        if not resume:
            raise FileExistsError(
                f"checkpoint exists for {cell_id}; use --resume or --overwrite"
            )
        rows = [dict(row) for row in read_csv(path)]
        validate_checkpoint_prefix(rows, requests, cell_id=cell_id, repeat=repeat)
        resumed_rows = len(rows)
    for position in range(len(rows), len(requests)):
        request = requests[position]
        row = dict(execute_request(request, position))
        row["checkpoint_cell"] = cell_id
        rows.append(row)
        if checkpoint_every and (
            len(rows) % checkpoint_every == 0 or len(rows) == len(requests)
        ):
            atomic_write_csv(path, rows)
    if not path.exists():
        atomic_write_csv(path, rows)
    validate_checkpoint_prefix(rows, requests, cell_id=cell_id, repeat=repeat)
    identity = file_identity(path, hash_contents=True)
    return rows, {
        "cell_id": cell_id,
        "path": str(path.resolve()),
        "sha256": identity["sha256"],
        "rows": len(rows),
        "expected_rows": len(requests),
        "repeat": repeat,
        "resumed_rows": resumed_rows,
        "complete": len(rows) == len(requests),
    }


def checkpoint_contract_payload(
    args: argparse.Namespace,
    observed_hashes: Mapping[str, str],
    methods: Sequence[str],
) -> dict[str, Any]:
    run_args = normalized_args(args)
    for ignored in (
        "out_dir",
        "tag",
        "overwrite",
        "resume",
        "progress_queries",
        "checkpoint_every",
        "dry_run",
    ):
        run_args.pop(ignored, None)
    return {
        "protocol_version": CHECKPOINT_PROTOCOL_VERSION,
        "protocol": CURRENT_PROTOCOL,
        "run_args": run_args,
        "methods": list(methods),
        "input_hashes": dict(sorted(observed_hashes.items())),
    }


def prepare_checkpoint_directory(
    checkpoint_dir: Path,
    contract: Mapping[str, Any],
    *,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    contract_path = checkpoint_dir / "contract.json"
    if overwrite and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    if checkpoint_dir.exists() and not resume:
        raise FileExistsError(
            f"checkpoint directory exists; use --resume or --overwrite: {checkpoint_dir}"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if contract_path.exists():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != contract:
            raise ValueError("checkpoint contract does not match this formal run")
    else:
        atomic_write_json(contract_path, dict(contract))
    return file_identity(contract_path, hash_contents=True)


def _run_legacy(args: argparse.Namespace) -> dict[str, Path]:
    import numpy as np
    import psycopg

    try:
        from .common_pg import pg_config_from_env
    except ImportError:
        from common_pg import pg_config_from_env

    methods = parse_methods(args.methods)
    faiss_enabled = FAISS_METHOD in methods
    sql_methods = tuple(method for method in methods if method in SQL_FIRST_METHODS)
    faiss: Any | None = None
    if faiss_enabled:
        import faiss as loaded_faiss

        faiss = loaded_faiss

    protocol_errors = formal_protocol_errors(args)
    if protocol_errors:
        raise ValueError("non-formal matched-recall protocol: " + "; ".join(protocol_errors))

    paths = output_paths(args.out_dir, args.tag)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "output/checkpoint exists and protocol-v2 never resumes historical raw rows; "
            f"pass --overwrite for a fresh run: {existing[0]}"
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    calibration_query_nos = list(
        range(args.calibration_query_offset, args.calibration_query_offset + args.calibration_queries)
    )
    final_query_nos = list(range(args.final_query_offset, args.final_query_offset + args.final_queries))
    expected_calibration = list(range(
        DEFAULT_CALIBRATION_QUERY_OFFSET,
        DEFAULT_CALIBRATION_QUERY_OFFSET + DEFAULT_CALIBRATION_QUERIES,
    ))
    expected_final = list(range(
        DEFAULT_FINAL_QUERY_OFFSET,
        DEFAULT_FINAL_QUERY_OFFSET + DEFAULT_FINAL_QUERIES,
    ))
    if calibration_query_nos != expected_calibration or final_query_nos != expected_final:
        raise ValueError("formal matched-recall split requires calibration q20..q99 and held-out final q100..q199")
    targets = parse_targets(args.target_recalls)
    ef_values = parse_int_csv(args.ef_search_values)
    specs = load_filter_specs(args.filters_csv, set(args.filter_names) or None)
    if tuple(spec.name for spec in specs) != FORMAL_FILTER_NAMES:
        raise ValueError(
            "formal Amazon-10M baseline requires the ordered 14-filter workload; "
            f"got {[spec.name for spec in specs]}"
        )
    truth, query_ids = load_truth(
        args.truth_csv, specs, calibration_query_nos, final_query_nos, args.k
    )
    vectors, vector_rows, dimensions = read_fbin_memmap(args.fbin, args.rows)
    if vector_rows != args.rows:
        raise ValueError(f"fbin rows={vector_rows}, expected --rows={args.rows}")
    if any(query_id < 0 or query_id >= vector_rows for query_id in query_ids.values()):
        raise ValueError("truth query ID is outside the fbin row range")
    index: Any | None = None
    index_meta: dict[str, Any] | None = None
    if faiss_enabled:
        assert faiss is not None
        index = faiss.read_index(str(args.faiss_index))
        index_meta = faiss_index_metadata(index, faiss, vector_rows, dimensions)
        faiss.omp_set_num_threads(args.faiss_threads)
    faiss_query_vectors = {
        query_no: np.ascontiguousarray(vectors[query_id], dtype=np.float32)
        for query_no, query_id in query_ids.items()
    }
    filters_identity = file_identity(args.filters_csv, hash_contents=True)
    truth_identity = file_identity(args.truth_csv, hash_contents=True)
    fbin_identity = file_identity(args.fbin, hash_contents=True)
    faiss_identity = (
        file_identity(args.faiss_index, hash_contents=True)
        if faiss_enabled
        else None
    )
    truth_manifest = verify_truth_manifest(
        args.truth_manifest, truth_identity, fbin_identity
    )
    observed_hashes = {
        "filters": filters_identity["sha256"],
        "truth": truth_identity["sha256"],
        "truth_manifest": truth_manifest["sha256"],
        "query_cohort_csv": truth_manifest["query_cohort_csv"]["sha256"],
        "query_cohort_manifest": truth_manifest["query_cohort_manifest"]["sha256"],
        "fbin": fbin_identity["sha256"],
    }
    if faiss_enabled:
        assert faiss_identity is not None
        observed_hashes.update(
            {
                "faiss_index": faiss_identity["sha256"],
                "faiss_index_manifest": sha256_file(args.faiss_index_manifest),
            }
        )
    mismatched_hashes = formal_input_hash_errors(observed_hashes, methods)
    if mismatched_hashes:
        raise ValueError(
            "formal current-input hash mismatch (legacy GT/M16 artifacts are ineligible): "
            + json.dumps(mismatched_hashes, sort_keys=True)
        )
    faiss_build = (
        verify_faiss_build_manifest(
            args.faiss_index_manifest,
            faiss_identity,
            fbin_identity,
            vector_rows,
            dimensions,
        )
        if faiss_enabled and faiss_identity is not None
        else None
    )
    runner_identity = file_identity(Path(__file__), hash_contents=True)

    manifest: dict[str, Any] = {
        "artifact": "amazon10m_matched_recall_baselines",
        "diagnostic_valid": False,
        "artifact_valid": False,
        "paper_eligible": False,
        "status": "running",
        "validation_errors": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": normalized_args(args),
        "inputs": {
            "filters": filters_identity,
            "truth": truth_identity,
            "truth_manifest": truth_manifest,
            "fbin": fbin_identity,
            "faiss_index": faiss_identity if faiss_enabled else NA,
            "faiss_index_build_manifest": faiss_build if faiss_enabled else NA,
            "runner": runner_identity,
            "postgres_table": args.table,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "filter_names": [spec.name for spec in specs],
        "run_contract": {
            key: value
            for key, value in normalized_args(args).items()
            if key not in {"filter_names", "tag", "out_dir", "overwrite", "progress_queries"}
        },
        "checkpoint": {
            "protocol_version": CHECKPOINT_PROTOCOL_VERSION,
            "resumable": False,
            "reuse_policy": (
                "raw checkpoints are diagnostic only and are never resumed; an existing "
                "artifact is rejected unless --overwrite starts a fresh protocol-v2 run"
            ),
        },
        "source_db": {"table": args.table},
        "baseline_scope": {
            "requested_methods": list(methods),
            "sql_first_materialized_control": SQL_FIRST_CONTROL_METHOD in methods,
            "sql_first_planner_chosen_exact": SQL_FIRST_PLANNER_METHOD in methods,
            "sql_first_forced_indexed_exact": SQL_FIRST_FORCED_METHOD in methods,
            "faiss_allowlist_hnsw": faiss_enabled,
            "independent_upstream_pgvector": False,
            "sqlens_disabled_stock_control": False,
            "note": (
                "This artifact is a SQL-first/FAISS baseline only. It must not be "
                "reported as an official upstream pgvector or SQLens-disabled control."
            ),
        },
        "source_hashes": {
            **observed_hashes,
            "formal_current_inputs_match": True,
        },
        "query_splits": {
            "calibration_query_nos": calibration_query_nos,
            "final_query_nos": final_query_nos,
            "reserved_query_nos": list(range(20)),
            "query_no_overlap": False,
            "query_id_overlap": False,
        },
        "repeats": {"calibration": args.calibration_repeats, "final": args.final_repeats},
        "target_recalls": targets,
        "ef_ladder": ef_values,
        "faiss_index": index_meta if faiss_enabled else NA,
        "environment": {
            "git_revision": git_revision(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "not_selected"),
            "psycopg": getattr(psycopg, "__version__", "unknown"),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "not_selected"),
            "psycopg": getattr(psycopg, "__version__", "unknown"),
            "measurement_runner_sha256": runner_identity["sha256"],
        },
        "execution": {
            "parallel_claim": False,
            "faiss_openmp_threads": args.faiss_threads,
            "postgres_max_parallel_workers_per_gather": 0,
            "config_order": "deterministic_balanced_interleaved_rotation",
            "schedule_seed": args.schedule_seed,
            "sql_first_exact_latency": (
                "end-to-end PostgreSQL execute+fetchall(top-10); the control uses a "
                "MATERIALIZED CTE while planner-chosen and forced-indexed use direct "
                "exact SQL whose EXPLAIN is required to contain no HNSW index"
            ),
            "faiss_allowlist_latency": (
                "Faiss HNSW search only after a complete prebuilt IDSelectorBitmap; setup is reported separately"
            ),
            "excluded_from_timed_requests": "query-vector prefetch, EXPLAIN, warmup, ground-truth generation, output I/O",
            "ground_truth_latency_used_as_baseline": False,
            "allowlist_cost": "one real SQL stream and one bitmap construction per predicate",
            "qps_reported": False,
            "latency_reciprocal_used_as_qps": False,
        },
        "bootstrap": {
            "unit": "query cluster after averaging repeats",
            "target_selection": TARGET_SELECTION_RULES[
                args.calibration_selection_policy
            ],
            "ci_lcb": "required for formal selection and held-out publication",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
    }
    write_json(paths["manifest"], manifest)

    raw_rows: list[dict[str, Any]] = []
    allow_lists: dict[str, AllowList] = {}
    exact_plan_valid: dict[Any, bool] = {}
    explain_audit: dict[str, Any] = {}
    warmup_errors: list[str] = []
    try:
        with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
            cursor = conn.cursor()
            cursor.execute("SET max_parallel_workers_per_gather = 0")
            cursor.execute("SET jit = off")
            cursor.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
            cursor.execute(f"SELECT count(*), min(id), max(id) FROM {args.table}")
            table_rows, min_id, max_id = (int(value) for value in cursor.fetchone())
            if (table_rows, min_id, max_id) != (vector_rows, 0, vector_rows - 1):
                raise ValueError(
                    "PostgreSQL/Faiss ID-space mismatch: "
                    f"table=({table_rows}, {min_id}, {max_id}) faiss=(0, {vector_rows - 1})"
                )
            cursor.execute(
                "SELECT indexrelid::regclass::text "
                "FROM pg_index JOIN pg_class ON pg_class.oid=indexrelid "
                "JOIN pg_am ON pg_am.oid=pg_class.relam "
                "WHERE indrelid=%s::regclass AND pg_am.amname='hnsw'",
                (args.table,),
            )
            hnsw_indexes = [str(row[0]) for row in cursor.fetchall()]
            cursor.execute(
                "SELECT indexrelid::regclass::text "
                "FROM pg_index JOIN pg_class ON pg_class.oid=indexrelid "
                "JOIN pg_am ON pg_am.oid=pg_class.relam "
                "WHERE indrelid=%s::regclass AND pg_am.amname<>'hnsw' "
                "AND NOT pg_index.indisprimary",
                (args.table,),
            )
            scalar_indexes = [str(row[0]) for row in cursor.fetchall()]
            if SQL_FIRST_FORCED_METHOD in methods and not scalar_indexes:
                raise RuntimeError(
                    "forced-indexed SQL-first arm requires registered non-primary scalar indexes"
                )
            cursor.execute(
                "SELECT current_setting('server_version'), "
                "COALESCE((SELECT extversion FROM pg_extension WHERE extname='vector'), ''), "
                "c.oid::bigint, c.relfilenode::bigint "
                "FROM pg_class AS c WHERE c.oid=%s::regclass",
                (args.table,),
            )
            postgres_version, vector_version, table_oid, table_relfilenode = cursor.fetchone()
            software_identity = postgres_software_identity(cursor)
            truth_relation = truth_manifest["postgres_relation"]
            expected_truth_table = str(truth_relation["table"]).rsplit(".", 1)[-1]
            if (
                args.table.rsplit(".", 1)[-1] != expected_truth_table
                or int(truth_relation["table_oid"]) != int(table_oid)
                or int(truth_relation["table_relfilenode"]) != int(table_relfilenode)
                or int(truth_relation["rows"]) != int(table_rows)
            ):
                raise ValueError(
                    "active PostgreSQL relation does not match exact-GT provenance: "
                    f"truth={dict(truth_relation)} active={{'table': args.table, "
                    f"'oid': {table_oid}, 'relfilenode': {table_relfilenode}, 'rows': {table_rows}}}"
                )
            cursor.execute(
                f"SELECT count(*) FROM {args.table} "
                f"WHERE ({args.candidate_validity_predicate})"
            )
            valid_rows = int(cursor.fetchone()[0])
            if valid_rows != EXPECTED_VALID_ROWS:
                raise ValueError(
                    f"candidate universe row mismatch: expected={EXPECTED_VALID_ROWS} "
                    f"actual={valid_rows}"
                )
            sql_query_vectors = (
                prefetch_sql_query_vectors(
                    cursor,
                    args.table,
                    query_ids.values(),
                    args.candidate_validity_predicate,
                )
                if sql_methods
                else {}
            )
            manifest["postgres"] = {
                "server_version": postgres_version,
                "vector_extension_version": vector_version,
                "table_oid": int(table_oid),
                "table_relfilenode": int(table_relfilenode),
                "rows": table_rows,
                "min_id": min_id,
                "max_id": max_id,
                "hnsw_indexes": hnsw_indexes,
                "scalar_indexes": scalar_indexes,
                "candidate_universe": {
                    "predicate": args.candidate_validity_predicate,
                    "rows": valid_rows,
                },
                "software_identity": software_identity,
            }

            explain_query_id = query_ids[calibration_query_nos[0]]
            for spec in specs:
                explain_audit[spec.name] = {}
                for method in sql_methods:
                    try:
                        explain_audit[spec.name][method] = explain_exact_plan(
                            cursor,
                            args.table,
                            spec,
                            sql_query_vectors[explain_query_id],
                            args.k,
                            hnsw_indexes,
                            method=method,
                            scalar_indexes=scalar_indexes,
                        )
                        exact_plan_valid[(spec.name, method)] = True
                    except Exception as exc:
                        exact_plan_valid[(spec.name, method)] = False
                        explain_audit[spec.name][method] = {
                            "method": method,
                            "error": f"{exc.__class__.__name__}: {exc}",
                            "hnsw_indexes": hnsw_indexes,
                            "scalar_indexes": scalar_indexes,
                        }

            if faiss_enabled:
                assert faiss is not None
                for position, spec in enumerate(specs, start=1):
                    allow_list = build_allow_list(
                        conn,
                        faiss,
                        args.table,
                        spec,
                        vector_rows,
                        args.allowlist_fetch_rows,
                        args.candidate_validity_predicate,
                    )
                    allow_lists[spec.name] = allow_list
                    raw_rows.append(setup_row(spec, allow_list))
                    print(
                        f"allow-list {position}/{len(specs)} filter={spec.name} rows={allow_list.rows} "
                        f"ms={allow_list.build_ms:.2f} bytes={allow_list.bitmap_bytes} "
                        f"valid={allow_list.valid}",
                        flush=True,
                    )
                    write_csv(paths["raw"], raw_rows)

            if faiss_enabled and args.warmup_queries:
                assert faiss is not None and index is not None
                warm_query_nos = calibration_query_nos[: args.warmup_queries]
                for spec in specs:
                    allow_list = allow_lists[spec.name]
                    if not allow_list.valid:
                        continue
                    for query_no in warm_query_nos:
                        for ef_search in ef_values:
                            try:
                                search_faiss(
                                    index,
                                    faiss,
                                    faiss_query_vectors[query_no],
                                    allow_list.selector,
                                    ef_search,
                                    args.k,
                                    query_id=query_ids[query_no],
                                )
                            except Exception as exc:
                                warmup_errors.append(
                                    f"calibration|{spec.name}|q{query_no}|ef{ef_search}|"
                                    f"{exc.__class__.__name__}: {exc}"
                                )

            if faiss_enabled:
                assert faiss is not None and index is not None
                run_calibration(
                    raw_rows,
                    filter_specs=specs,
                    query_nos=calibration_query_nos,
                    repeats=args.calibration_repeats,
                    ef_values=ef_values,
                    query_ids=query_ids,
                    query_vectors=faiss_query_vectors,
                    truth=truth,
                    allow_lists=allow_lists,
                    index=index,
                    faiss_module=faiss,
                    vectors=vectors,
                    k=args.k,
                    schedule_seed=args.schedule_seed,
                    progress_queries=args.progress_queries,
                    checkpoint_path=paths["raw"],
                )
            write_csv(paths["raw"], raw_rows)
            if faiss_enabled:
                calibration_rows, selected = calibration_table(
                    raw_rows,
                    specs,
                    ef_values,
                    targets,
                    calibration_query_nos,
                    args.calibration_repeats,
                    args.bootstrap_samples,
                    args.bootstrap_seed,
                    allow_lists,
                    args.calibration_selection_policy,
                )
            else:
                calibration_rows, selected = [], {}
            write_csv(paths["calibration"], calibration_rows)
            calibration_outcomes = calibration_outcomes_from_rows(calibration_rows)

            if faiss_enabled:
                assert faiss is not None and index is not None
                representative_query_no = final_query_nos[0]
                for spec in specs:
                    for ef_search in sorted(
                        {
                            selected[(spec.name, target)]
                            for target in targets
                            if (spec.name, target) in selected
                        }
                    ):
                        raw_rows.append(
                            full_setup_search_row(
                                conn=conn,
                                faiss_module=faiss,
                                index=index,
                                table=args.table,
                                spec=spec,
                                total_rows=vector_rows,
                                fetch_rows=args.allowlist_fetch_rows,
                                query=faiss_query_vectors[representative_query_no],
                                query_no=representative_query_no,
                                query_id=query_ids[representative_query_no],
                                ef_search=ef_search,
                                k=args.k,
                                candidate_validity_predicate=args.candidate_validity_predicate,
                            )
                        )
                        write_csv(paths["raw"], raw_rows)

            if args.warmup_queries:
                for spec in specs:
                    for query_no in final_query_nos[: args.warmup_queries]:
                        for method in sql_methods:
                            planner_mode = (
                                "forced_indexed"
                                if method == SQL_FIRST_FORCED_METHOD
                                else "auto"
                            )
                            if exact_plan_valid.get((spec.name, method), False):
                                try:
                                    search_sql_exact(
                                        cursor,
                                        exact_sql_for_method(
                                            method,
                                            args.table,
                                            spec.predicate,
                                            args.k,
                                        ),
                                        query_ids[query_no],
                                        sql_query_vectors[query_ids[query_no]],
                                        planner_mode,
                                    )
                                except Exception as exc:
                                    warmup_errors.append(
                                        f"final|{method}|{spec.name}|q{query_no}|"
                                        f"{exc.__class__.__name__}: {exc}"
                                    )
                        if faiss_enabled:
                            assert faiss is not None and index is not None
                            for ef_search in sorted(
                                {
                                    selected[(spec.name, target)]
                                    for target in targets
                                    if (spec.name, target) in selected
                                }
                            ):
                                try:
                                    search_faiss(
                                        index,
                                        faiss,
                                        faiss_query_vectors[query_no],
                                        allow_lists[spec.name].selector,
                                        ef_search,
                                        args.k,
                                        query_id=query_ids[query_no],
                                    )
                                except Exception as exc:
                                    warmup_errors.append(
                                        f"final|{FAISS_METHOD}|{spec.name}|q{query_no}|ef{ef_search}|"
                                        f"{exc.__class__.__name__}: {exc}"
                                    )

            run_final(
                raw_rows,
                table=args.table,
                filter_specs=specs,
                methods=methods,
                query_nos=final_query_nos,
                repeats=args.final_repeats,
                selected=selected,
                targets=targets,
                query_ids=query_ids,
                faiss_query_vectors=faiss_query_vectors,
                sql_query_vectors=sql_query_vectors,
                truth=truth,
                allow_lists=allow_lists,
                exact_plan_valid=exact_plan_valid,
                cursor=cursor,
                index=index,
                faiss_module=faiss,
                vectors=vectors,
                k=args.k,
                schedule_seed=args.schedule_seed,
                progress_queries=args.progress_queries,
                checkpoint_path=paths["raw"],
            )
            write_csv(paths["raw"], raw_rows)
            final_rows = [row for row in raw_rows if row.get("phase") == "final"]
            write_csv(paths["final"], final_rows)
            summary_rows = final_summary_table(
                final_rows,
                specs,
                targets,
                selected,
                final_query_nos,
                args.final_repeats,
                args.bootstrap_samples,
                args.bootstrap_seed,
                allow_lists,
                calibration_outcomes,
                methods,
                [
                    row
                    for row in raw_rows
                    if row.get("phase") == "setup_search_e2e"
                ],
            )
            write_csv(paths["summary"], summary_rows)

        validation_errors = artifact_validation_errors(
            calibration_rows, summary_rows, specs, ef_values, targets, methods
        )
        invalid_setup_rows = [
            str(row.get("pair_key", row.get("filter_name", "unknown")))
            for row in raw_rows
            if row.get("phase") in {"setup", "setup_search_e2e"}
            and not _row_ok(row)
        ]
        if invalid_setup_rows:
            validation_errors.append(
                "allow-list setup/full-e2e rows are invalid: "
                + ",".join(invalid_setup_rows[:10])
            )
        completion = completion_gate(
            calibration_rows, summary_rows, specs, ef_values, targets, methods
        )
        # A finished narrow/debug slice can still be diagnostically useful, but
        # cannot be promoted as a formal baseline without its registered grid.
        validity = artifact_validity_flags(
            validation_errors,
            completion,
            formal_provenance_valid=not mismatched_hashes,
        )
        diagnostic_valid = validity["diagnostic_valid"]
        artifact_valid = validity["artifact_valid"]
        paper_eligible = validity["paper_eligible"]
        manifest.update(validity)
        manifest["status"] = (
            "paper_eligible"
            if paper_eligible
            else completion["status"]
            if artifact_valid
            else "invalid"
        )
        manifest["validation_errors"] = validation_errors
        manifest["completion"] = completion
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["sql_first_exact_explain"] = explain_audit
        manifest["allowlists"] = {
            name: {
                "rows": value.rows,
                "build_ms": value.build_ms,
                "server_execution_ms": value.server_execution_ms,
                "row_transfer_ms": value.row_transfer_ms,
                "bitmap_construction_ms": value.bitmap_construction_ms,
                "selector_construction_ms": value.selector_construction_ms,
                "full_setup_ms": value.full_setup_ms,
                "bitmap_bytes": value.bitmap_bytes,
                "valid": value.valid,
                "error": value.error,
            }
            for name, value in allow_lists.items()
        }
        manifest["latency_reporting"] = {
            "allowlist_sql_materialization": "allowlist_sql_materialization_ms",
            "allowlist_row_transfer": "allowlist_row_transfer_ms",
            "allowlist_bitmap_build": "allowlist_bitmap_build_ms",
            "cached_list_faiss_search": "cached_allowlist_search_ms",
            "continuous_full_e2e": "continuous_full_e2e_ms",
            "continuous_full_e2e_is_measured": True,
            "continuous_full_e2e_is_sum_of_stage_means": False,
            "qps_reported": False,
            "latency_reciprocal_used_as_qps": False,
        }
        manifest["selected_faiss_ef_search"] = {
            f"{filter_name}|{target:.2f}": ef_search
            for (filter_name, target), ef_search in selected.items()
        }
        manifest["summary_valid_rows"] = sum(row["status"] == "valid" for row in summary_rows)
        manifest["matched_recall_valid_rows"] = sum(
            bool(row["matched_recall_comparison_valid"]) for row in summary_rows
        )
        manifest["summary_rows"] = len(summary_rows)
        manifest["row_counts"] = {
            "raw": len(raw_rows),
            "calibration": len(calibration_rows),
            "final": len(final_rows),
            "summary": len(summary_rows),
        }
        manifest["outputs"] = {
            **{name: str(path) for name, path in paths.items()},
            "raw": {
                "path": str(paths["raw"]), "sha256": sha256_file(paths["raw"]), "rows": len(raw_rows)
            },
            "calibration": {
                "path": str(paths["calibration"]), "sha256": sha256_file(paths["calibration"]), "rows": len(calibration_rows)
            },
            "final": {
                "path": str(paths["final"]), "sha256": sha256_file(paths["final"]), "rows": len(final_rows)
            },
            "summary": {
                "path": str(paths["summary"]), "sha256": sha256_file(paths["summary"]), "rows": len(summary_rows)
            },
            "manifest": str(paths["manifest"]),
        }
        manifest["warmup_errors"] = warmup_errors
        write_json(paths["manifest"], manifest)
        return paths
    except Exception as exc:
        manifest["diagnostic_valid"] = False
        manifest["artifact_valid"] = False
        manifest["paper_eligible"] = False
        manifest["status"] = "invalid"
        manifest["validation_errors"] = [f"fatal_error: {exc.__class__.__name__}: {exc}"]
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["fatal_error"] = f"{exc.__class__.__name__}: {exc}"
        manifest["sql_first_exact_explain"] = explain_audit
        manifest["warmup_errors"] = warmup_errors
        if raw_rows:
            write_csv(paths["raw"], raw_rows)
        write_json(paths["manifest"], manifest)
        raise


def _run_current(args: argparse.Namespace) -> dict[str, Path]:
    import numpy as np
    import psycopg

    try:
        from .common_pg import pg_config_from_env
    except ImportError:
        from common_pg import pg_config_from_env

    methods = parse_methods(args.methods)
    faiss_enabled = FAISS_METHOD in methods
    sql_methods = tuple(method for method in methods if method in SQL_FIRST_METHODS)
    faiss: Any | None = None
    if faiss_enabled:
        import faiss as loaded_faiss

        faiss = loaded_faiss

    protocol_errors = formal_protocol_errors(args)
    if protocol_errors:
        raise ValueError(
            "non-formal current matched-recall protocol: "
            + "; ".join(protocol_errors)
        )

    paths = output_paths(args.out_dir, args.tag)
    prefix = f"amazon10m_matched_recall_baselines_{args.tag}"
    checkpoint_dir = args.out_dir / f"{prefix}_checkpoints"
    output_files = list(paths.values())
    if args.overwrite:
        for path in output_files:
            if path.exists():
                path.unlink()
    elif not args.resume:
        existing = [path for path in output_files if path.exists()]
        if existing:
            raise FileExistsError(
                f"formal output exists; use --resume or --overwrite: {existing[0]}"
            )
    elif any(path.exists() for path in output_files) and not checkpoint_dir.exists():
        raise FileNotFoundError(
            "cannot resume formal outputs without their checkpoint directory: "
            f"{checkpoint_dir}"
        )

    specs = load_filter_specs(args.filters_csv, set(args.filter_names) or None)
    if tuple(spec.name for spec in specs) != FORMAL_FILTER_NAMES:
        raise ValueError(
            "formal Amazon-10M baseline requires the ordered 14-filter workload"
        )
    filter_names = {spec.name for spec in specs}
    calibration_requests = load_workload(
        args.calibration_workload_csv,
        expected_rows=CURRENT_CALIBRATION_REQUESTS,
        expected_split="calibration",
        filter_names=filter_names,
    )
    measurement_requests = load_workload(
        args.measurement_workload_csv,
        expected_rows=CURRENT_MEASUREMENT_REQUESTS,
        expected_split="measurement",
        filter_names=filter_names,
    )
    validate_workload_pair(calibration_requests, measurement_requests)
    calibration_query_nos = [request.query_no for request in calibration_requests]
    measurement_query_nos = [request.query_no for request in measurement_requests]
    truth, query_ids = load_truth(
        args.truth_csv,
        specs,
        calibration_query_nos,
        measurement_query_nos,
        args.k,
        enforce_requested_split=False,
    )
    for request in (*calibration_requests, *measurement_requests):
        if query_ids[request.query_no] != request.query_id:
            raise ValueError(
                f"workload query identity mismatch at request={request.request_no}: "
                f"query_no={request.query_no} workload_id={request.query_id} "
                f"truth_id={query_ids[request.query_no]}"
            )

    vectors, vector_rows, dimensions = read_fbin_memmap(args.fbin, args.rows)
    if vector_rows != args.rows:
        raise ValueError(f"fbin rows={vector_rows}, expected --rows={args.rows}")
    index: Any | None = None
    index_meta: dict[str, Any] | None = None
    if faiss_enabled:
        assert faiss is not None
        index = faiss.read_index(str(args.faiss_index))
        index_meta = faiss_index_metadata(index, faiss, vector_rows, dimensions)
        faiss.omp_set_num_threads(args.faiss_threads)
    faiss_query_vectors = {
        query_no: np.ascontiguousarray(vectors[query_id], dtype=np.float32)
        for query_no, query_id in query_ids.items()
    }

    filters_identity = file_identity(args.filters_csv, hash_contents=True)
    truth_identity = file_identity(args.truth_csv, hash_contents=True)
    fbin_identity = file_identity(args.fbin, hash_contents=True)
    calibration_workload_identity = file_identity(
        args.calibration_workload_csv, hash_contents=True
    )
    measurement_workload_identity = file_identity(
        args.measurement_workload_csv, hash_contents=True
    )
    faiss_identity = (
        file_identity(args.faiss_index, hash_contents=True)
        if faiss_enabled
        else None
    )
    truth_manifest = verify_truth_manifest(
        args.truth_manifest, truth_identity, fbin_identity, CURRENT_PROTOCOL
    )
    observed_hashes = {
        "filters": filters_identity["sha256"],
        "truth": truth_identity["sha256"],
        "truth_manifest": truth_manifest["sha256"],
        "query_cohort_csv": truth_manifest["query_cohort_csv"]["sha256"],
        "query_cohort_manifest": truth_manifest["query_cohort_manifest"]["sha256"],
        "calibration_workload": calibration_workload_identity["sha256"],
        "measurement_workload": measurement_workload_identity["sha256"],
        "fbin": fbin_identity["sha256"],
    }
    if faiss_enabled:
        assert faiss_identity is not None
        observed_hashes.update(
            {
                "faiss_index": faiss_identity["sha256"],
                "faiss_index_manifest": sha256_file(args.faiss_index_manifest),
            }
        )
    mismatched_hashes = formal_input_hash_errors(
        observed_hashes, methods, CURRENT_PROTOCOL
    )
    if mismatched_hashes:
        raise ValueError(
            "formal current-input hash mismatch: "
            + json.dumps(mismatched_hashes, sort_keys=True)
        )
    faiss_build = (
        verify_faiss_build_manifest(
            args.faiss_index_manifest,
            faiss_identity,
            fbin_identity,
            vector_rows,
            dimensions,
        )
        if faiss_enabled and faiss_identity is not None
        else None
    )
    checkpoint_contract = checkpoint_contract_payload(
        args, observed_hashes, methods
    )
    checkpoint_contract_identity = prepare_checkpoint_directory(
        checkpoint_dir,
        checkpoint_contract,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    runner_identity = file_identity(Path(__file__), hash_contents=True)
    targets = parse_targets(args.target_recalls)
    ef_values = parse_int_csv(args.ef_search_values)
    calibration_by_filter = workload_query_nos_by_filter(
        calibration_requests, specs
    )
    measurement_by_filter = workload_query_nos_by_filter(
        measurement_requests, specs
    )
    specs_by_name = {spec.name: spec for spec in specs}
    manifest: dict[str, Any] = {
        "artifact": "amazon10m_matched_recall_baselines",
        "protocol": CURRENT_PROTOCOL,
        "diagnostic_valid": False,
        "artifact_valid": False,
        "paper_eligible": False,
        "status": "running",
        "validation_errors": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": normalized_args(args),
        "inputs": {
            "filters": filters_identity,
            "truth": truth_identity,
            "truth_manifest": truth_manifest,
            "calibration_workload": calibration_workload_identity,
            "measurement_workload": measurement_workload_identity,
            "fbin": fbin_identity,
            "faiss_index": faiss_identity if faiss_enabled else NA,
            "faiss_index_build_manifest": faiss_build if faiss_enabled else NA,
            "runner": runner_identity,
            "postgres_table": args.table,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "filter_names": [spec.name for spec in specs],
        "run_contract": {
            key: value
            for key, value in normalized_args(args).items()
            if key
            not in {
                "filter_names",
                "tag",
                "out_dir",
                "overwrite",
                "resume",
                "progress_queries",
                "checkpoint_every",
            }
        },
        "checkpoint": {
            "protocol_version": CHECKPOINT_PROTOCOL_VERSION,
            "directory": str(checkpoint_dir.resolve()),
            "contract": checkpoint_contract_identity,
            "resumable": True,
            "granularity": (
                "one complete request_no-ordered mixed-trace arm/config repeat"
            ),
            "partial_cell_policy": (
                "each cell CSV is an atomically written contiguous trace prefix; "
                "resume validates every request/query/filter/repeat key before continuing"
            ),
        },
        "baseline_scope": {
            "requested_methods": list(methods),
            "sql_first_exact_has_no_ann_matched_config": True,
            "faiss_independently_tuned": faiss_enabled,
            "faiss_targets": targets if faiss_enabled else [],
            "faiss_cached_allowlist_reported_separately": faiss_enabled,
            "faiss_continuous_full_e2e_per_request": faiss_enabled,
        },
        "source_hashes": {
            **observed_hashes,
            "formal_current_inputs_match": True,
        },
        "query_splits": {
            "calibration_workload": {
                "requests": len(calibration_requests),
                "query_nos": len(set(calibration_query_nos)),
                "request_no_range": [0, len(calibration_requests) - 1],
            },
            "measurement_workload": {
                "requests": len(measurement_requests),
                "query_nos": len(set(measurement_query_nos)),
                "request_no_range": [0, len(measurement_requests) - 1],
            },
            "query_no_overlap": False,
            "query_id_overlap": False,
        },
        "repeats": {
            "calibration": args.calibration_repeats,
            "measurement": args.measurement_repeats,
        },
        "target_recalls": targets,
        "ef_ladder": ef_values,
        "faiss_index": index_meta if faiss_enabled else NA,
        "environment": {
            "git_revision": git_revision(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "not_selected"),
            "psycopg": getattr(psycopg, "__version__", "unknown"),
        },
        "execution": {
            "parallel_claim": False,
            "faiss_openmp_threads": args.faiss_threads,
            "postgres_max_parallel_workers_per_gather": 0,
            "trace_order": "strict ascending request_no within every cell",
            "cell_order": "deterministic balanced rotation by repeat",
            "sql_first_latency": "continuous PostgreSQL execute through fetchall(top-10)",
            "faiss_cached_latency": "cached IDSelectorBitmap index.search only",
            "faiss_full_e2e_latency": (
                "a separate continuous per-request interval covering untruncated "
                "PostgreSQL allow-list materialization, complete row transfer, bitmap "
                "and selector construction, and Faiss HNSW search"
            ),
            "continuous_full_e2e_is_sum_of_stage_means": False,
            "qps_reported": False,
        },
        "bootstrap": {
            "unit": "query request after averaging repeats",
            "target_selection": TARGET_SELECTION_RULES[
                args.calibration_selection_policy
            ],
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
    }
    atomic_write_json(paths["manifest"], manifest)

    setup_rows: list[dict[str, Any]] = []
    calibration_cells: list[dict[str, Any]] = []
    final_cells: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    allow_lists: dict[str, AllowList] = {}
    exact_plan_valid: dict[Any, bool] = {}
    explain_audit: dict[str, Any] = {}
    warmup_errors: list[str] = []
    calibration_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    try:
        with psycopg.connect(
            pg_config_from_env().conninfo, autocommit=True
        ) as conn:
            cursor = conn.cursor()
            cursor.execute("SET max_parallel_workers_per_gather = 0")
            cursor.execute("SET jit = off")
            cursor.execute(
                f"SET statement_timeout = {int(args.statement_timeout_ms)}"
            )
            cursor.execute(
                f"SELECT count(*), min(id), max(id) FROM {args.table}"
            )
            table_rows, min_id, max_id = (int(value) for value in cursor.fetchone())
            if (table_rows, min_id, max_id) != (
                vector_rows,
                0,
                vector_rows - 1,
            ):
                raise ValueError(
                    "PostgreSQL/Faiss ID-space mismatch: "
                    f"table=({table_rows}, {min_id}, {max_id})"
                )
            cursor.execute(
                "SELECT indexrelid::regclass::text "
                "FROM pg_index JOIN pg_class ON pg_class.oid=indexrelid "
                "JOIN pg_am ON pg_am.oid=pg_class.relam "
                "WHERE indrelid=%s::regclass AND pg_am.amname='hnsw'",
                (args.table,),
            )
            hnsw_indexes = [str(row[0]) for row in cursor.fetchall()]
            cursor.execute(
                "SELECT indexrelid::regclass::text "
                "FROM pg_index JOIN pg_class ON pg_class.oid=indexrelid "
                "JOIN pg_am ON pg_am.oid=pg_class.relam "
                "WHERE indrelid=%s::regclass AND pg_am.amname<>'hnsw' "
                "AND NOT pg_index.indisprimary",
                (args.table,),
            )
            scalar_indexes = [str(row[0]) for row in cursor.fetchall()]
            if SQL_FIRST_FORCED_METHOD in methods and not scalar_indexes:
                raise RuntimeError(
                    "forced-indexed SQL-first arm requires non-primary scalar indexes"
                )
            cursor.execute(
                "SELECT current_setting('server_version'), "
                "COALESCE((SELECT extversion FROM pg_extension WHERE extname='vector'), ''), "
                "c.oid::bigint, c.relfilenode::bigint "
                "FROM pg_class AS c WHERE c.oid=%s::regclass",
                (args.table,),
            )
            postgres_version, vector_version, table_oid, table_relfilenode = (
                cursor.fetchone()
            )
            truth_relation = truth_manifest["postgres_relation"]
            if (
                args.table.rsplit(".", 1)[-1]
                != str(truth_relation["table"]).rsplit(".", 1)[-1]
                or int(truth_relation["table_oid"]) != int(table_oid)
                or int(truth_relation["table_relfilenode"])
                != int(table_relfilenode)
                or int(truth_relation["rows"]) != table_rows
            ):
                raise ValueError(
                    "active PostgreSQL relation does not match exact-GT provenance"
                )
            cursor.execute(
                f"SELECT count(*) FROM {args.table} "
                f"WHERE ({args.candidate_validity_predicate})"
            )
            valid_rows = int(cursor.fetchone()[0])
            if valid_rows != EXPECTED_VALID_ROWS:
                raise ValueError(
                    f"candidate universe row mismatch: expected={EXPECTED_VALID_ROWS} "
                    f"actual={valid_rows}"
                )
            sql_query_vectors = (
                prefetch_sql_query_vectors(
                    cursor,
                    args.table,
                    query_ids.values(),
                    args.candidate_validity_predicate,
                )
                if sql_methods
                else {}
            )
            manifest["postgres"] = {
                "server_version": postgres_version,
                "vector_extension_version": vector_version,
                "table_oid": int(table_oid),
                "table_relfilenode": int(table_relfilenode),
                "rows": table_rows,
                "min_id": min_id,
                "max_id": max_id,
                "hnsw_indexes": hnsw_indexes,
                "scalar_indexes": scalar_indexes,
                "candidate_universe_rows": valid_rows,
                "software_identity": postgres_software_identity(cursor),
            }

            explain_query_id = calibration_requests[0].query_id
            for spec in specs:
                explain_audit[spec.name] = {}
                for method in sql_methods:
                    try:
                        explain_audit[spec.name][method] = explain_exact_plan(
                            cursor,
                            args.table,
                            spec,
                            sql_query_vectors[explain_query_id],
                            args.k,
                            hnsw_indexes,
                            method=method,
                            scalar_indexes=scalar_indexes,
                        )
                        exact_plan_valid[(spec.name, method)] = True
                    except Exception as exc:
                        exact_plan_valid[(spec.name, method)] = False
                        explain_audit[spec.name][method] = {
                            "error": f"{exc.__class__.__name__}: {exc}"
                        }

            if faiss_enabled:
                assert faiss is not None
                for position, spec in enumerate(specs, start=1):
                    allow_list = build_allow_list(
                        conn,
                        faiss,
                        args.table,
                        spec,
                        vector_rows,
                        args.allowlist_fetch_rows,
                        args.candidate_validity_predicate,
                    )
                    allow_lists[spec.name] = allow_list
                    setup_rows.append(setup_row(spec, allow_list))
                    print(
                        f"cached allow-list {position}/{len(specs)} "
                        f"filter={spec.name} rows={allow_list.rows} "
                        f"ms={allow_list.build_ms:.2f} valid={allow_list.valid}",
                        flush=True,
                    )

            if faiss_enabled and args.warmup_queries:
                assert faiss is not None and index is not None
                for request in calibration_requests[: args.warmup_queries]:
                    allow_list = allow_lists[request.filter_name]
                    for ef_search in ef_values:
                        try:
                            search_faiss(
                                index,
                                faiss,
                                faiss_query_vectors[request.query_no],
                                allow_list.selector,
                                ef_search,
                                args.k,
                                query_id=request.query_id,
                            )
                        except Exception as exc:
                            warmup_errors.append(
                                f"calibration|q{request.query_no}|ef{ef_search}|"
                                f"{exc.__class__.__name__}: {exc}"
                            )

            if faiss_enabled:
                assert faiss is not None and index is not None
                calibration_cell_specs = [
                    (ef_search, repeat)
                    for repeat in range(args.calibration_repeats)
                    for ef_search in balanced_order(
                        ef_values, repeat, args.schedule_seed
                    )
                ]
                for ef_search, repeat in calibration_cell_specs:
                    cell_id = checkpoint_cell_name(
                        phase="calibration",
                        repeat=repeat,
                        method=FAISS_METHOD,
                        ef_search=ef_search,
                    )

                    def execute_calibration(
                        request: WorkloadRequest,
                        position: int,
                        *,
                        selected_ef: int = ef_search,
                        selected_repeat: int = repeat,
                    ) -> dict[str, Any]:
                        spec = specs_by_name[request.filter_name]
                        allow_list = allow_lists[spec.name]
                        truth_entry = truth[(spec.name, request.query_no)]
                        try:
                            if not allow_list.valid:
                                raise RuntimeError(allow_list.error)
                            ids, latency_ms = search_faiss(
                                index,
                                faiss,
                                faiss_query_vectors[request.query_no],
                                allow_list.selector,
                                selected_ef,
                                args.k,
                                query_id=request.query_id,
                            )
                            invalid_ids = result_membership_errors(
                                allow_list.bitmap, ids
                            )
                            if invalid_ids:
                                raise RuntimeError(
                                    f"Faiss result outside allow-list: {invalid_ids[:5]}"
                                )
                            return measurement_row(
                                phase="calibration",
                                method=FAISS_METHOD,
                                spec=spec,
                                query_no=request.query_no,
                                query_id=request.query_id,
                                repeat=selected_repeat,
                                schedule_position=position,
                                block_no=request.request_no,
                                ef_search=selected_ef,
                                result_ids=ids,
                                truth_ids=truth_entry.ids,
                                latency_ms=latency_ms,
                                truth_entry=truth_entry,
                                vectors=vectors,
                                request_no=request.request_no,
                                trace_cycle=request.trace_cycle,
                            )
                        except Exception as exc:
                            return measurement_row(
                                phase="calibration",
                                method=FAISS_METHOD,
                                spec=spec,
                                query_no=request.query_no,
                                query_id=request.query_id,
                                repeat=selected_repeat,
                                schedule_position=position,
                                block_no=request.request_no,
                                ef_search=selected_ef,
                                result_ids=None,
                                truth_ids=truth_entry.ids,
                                latency_ms=NA,
                                error=f"{exc.__class__.__name__}: {exc}",
                                request_no=request.request_no,
                                trace_cycle=request.trace_cycle,
                            )

                    cell_rows, cell_record = execute_checkpointed_cell(
                        checkpoint_dir=checkpoint_dir,
                        cell_id=cell_id,
                        requests=calibration_requests,
                        repeat=repeat,
                        checkpoint_every=args.checkpoint_every,
                        resume=args.resume,
                        execute_request=execute_calibration,
                    )
                    calibration_cells.extend(cell_rows)
                    checkpoint_records.append(cell_record)
                    print(
                        f"complete {cell_id} rows={len(cell_rows)} "
                        f"resumed={cell_record['resumed_rows']}",
                        flush=True,
                    )

                calibration_rows, selected = calibration_table(
                    calibration_cells,
                    specs,
                    ef_values,
                    targets,
                    calibration_by_filter,
                    args.calibration_repeats,
                    args.bootstrap_samples,
                    args.bootstrap_seed,
                    allow_lists,
                    args.calibration_selection_policy,
                )
                calibration_outcomes = calibration_outcomes_from_rows(
                    calibration_rows
                )
            else:
                calibration_rows, selected, calibration_outcomes = [], {}, {}
            atomic_write_csv(paths["calibration"], calibration_rows)

            final_cell_specs: list[tuple[str, float | None]] = [
                (method, None) for method in sql_methods
            ]
            if faiss_enabled:
                final_cell_specs.extend((FAISS_METHOD, target) for target in targets)

            for repeat in range(args.measurement_repeats):
                ordered_cells = balanced_order(
                    final_cell_specs, repeat, args.schedule_seed + 104729
                )
                for method, target in ordered_cells:
                    cell_requests = (
                        [
                            request
                            for request in measurement_requests
                            if (request.filter_name, float(target)) in selected
                        ]
                        if method == FAISS_METHOD and target is not None
                        else list(measurement_requests)
                    )
                    cell_id = checkpoint_cell_name(
                        phase="final",
                        repeat=repeat,
                        method=method,
                        target_recall=target,
                    )

                    def execute_final(
                        request: WorkloadRequest,
                        position: int,
                        *,
                        selected_method: str = method,
                        selected_target: float | None = target,
                        selected_repeat: int = repeat,
                    ) -> dict[str, Any]:
                        spec = specs_by_name[request.filter_name]
                        truth_entry = truth[(spec.name, request.query_no)]
                        if selected_method in SQL_FIRST_METHODS:
                            planner_mode = (
                                "forced_indexed"
                                if selected_method == SQL_FIRST_FORCED_METHOD
                                else "auto"
                            )
                            try:
                                if not exact_plan_valid.get(
                                    (spec.name, selected_method), False
                                ):
                                    raise RuntimeError(
                                        "exact SQL EXPLAIN gate failed"
                                    )
                                ids, latency_ms = search_sql_exact(
                                    cursor,
                                    exact_sql_for_method(
                                        selected_method,
                                        args.table,
                                        spec.predicate,
                                        args.k,
                                    ),
                                    request.query_id,
                                    sql_query_vectors[request.query_id],
                                    planner_mode,
                                )
                                return measurement_row(
                                    phase="final",
                                    method=selected_method,
                                    spec=spec,
                                    query_no=request.query_no,
                                    query_id=request.query_id,
                                    repeat=selected_repeat,
                                    schedule_position=position,
                                    block_no=request.request_no,
                                    ef_search=NA,
                                    result_ids=ids,
                                    truth_ids=truth_entry.ids,
                                    latency_ms=latency_ms,
                                    truth_entry=truth_entry,
                                    vectors=vectors,
                                    request_no=request.request_no,
                                    trace_cycle=request.trace_cycle,
                                )
                            except Exception as exc:
                                return measurement_row(
                                    phase="final",
                                    method=selected_method,
                                    spec=spec,
                                    query_no=request.query_no,
                                    query_id=request.query_id,
                                    repeat=selected_repeat,
                                    schedule_position=position,
                                    block_no=request.request_no,
                                    ef_search=NA,
                                    result_ids=None,
                                    truth_ids=truth_entry.ids,
                                    latency_ms=NA,
                                    error=f"{exc.__class__.__name__}: {exc}",
                                    request_no=request.request_no,
                                    trace_cycle=request.trace_cycle,
                                )

                        assert (
                            selected_target is not None
                            and faiss is not None
                            and index is not None
                        )
                        selected_ef = selected[
                            (spec.name, float(selected_target))
                        ]
                        allow_list = allow_lists[spec.name]
                        cached_error = ""
                        cached_ids: list[int] = []
                        cached_ms: float | str = NA
                        try:
                            if not allow_list.valid:
                                raise RuntimeError(allow_list.error)
                            cached_ids, measured_ms = search_faiss(
                                index,
                                faiss,
                                faiss_query_vectors[request.query_no],
                                allow_list.selector,
                                selected_ef,
                                args.k,
                                query_id=request.query_id,
                            )
                            cached_ms = measured_ms
                            invalid_ids = result_membership_errors(
                                allow_list.bitmap, cached_ids
                            )
                            if invalid_ids:
                                raise RuntimeError(
                                    f"cached Faiss result outside allow-list: "
                                    f"{invalid_ids[:5]}"
                                )
                        except Exception as exc:
                            cached_error = f"{exc.__class__.__name__}: {exc}"
                        row = measurement_row(
                            phase="final",
                            method=FAISS_METHOD,
                            spec=spec,
                            query_no=request.query_no,
                            query_id=request.query_id,
                            repeat=selected_repeat,
                            schedule_position=position,
                            block_no=request.request_no,
                            ef_search=selected_ef,
                            result_ids=cached_ids,
                            truth_ids=truth_entry.ids,
                            latency_ms=cached_ms,
                            truth_entry=truth_entry,
                            vectors=vectors,
                            error=cached_error,
                            matched_target_recalls=(float(selected_target),),
                            request_no=request.request_no,
                            trace_cycle=request.trace_cycle,
                            target_recall=float(selected_target),
                        )
                        full = full_setup_search_row(
                            conn=conn,
                            faiss_module=faiss,
                            index=index,
                            table=args.table,
                            spec=spec,
                            total_rows=vector_rows,
                            fetch_rows=args.allowlist_fetch_rows,
                            query=faiss_query_vectors[request.query_no],
                            query_no=request.query_no,
                            query_id=request.query_id,
                            ef_search=selected_ef,
                            k=args.k,
                            repeat=selected_repeat,
                            request_no=request.request_no,
                            trace_cycle=request.trace_cycle,
                            truth_entry=truth_entry,
                            vectors=vectors,
                            candidate_validity_predicate=(
                                args.candidate_validity_predicate
                            ),
                        )
                        row.update(
                            {
                                "continuous_full_e2e_ms": full[
                                    "continuous_full_e2e_ms"
                                ],
                                "continuous_full_e2e_valid": _row_ok(full),
                                "continuous_full_e2e_error": full["error"],
                                "continuous_recall_at_10": full[
                                    "continuous_recall_at_10"
                                ],
                                "continuous_result_ids": full["result_ids"],
                                "continuous_returned": full["returned"],
                                "per_request_allowlist_rows": full[
                                    "allowlist_build_rows"
                                ],
                                "per_request_allowlist_sql_materialization_ms": full[
                                    "allowlist_sql_materialization_ms"
                                ],
                                "per_request_allowlist_row_transfer_ms": full[
                                    "allowlist_row_transfer_ms"
                                ],
                                "per_request_allowlist_bitmap_build_ms": full[
                                    "allowlist_bitmap_build_ms"
                                ],
                                "per_request_allowlist_selector_construction_ms": full[
                                    "allowlist_selector_construction_ms"
                                ],
                                "per_request_allowlist_full_setup_ms": full[
                                    "allowlist_full_setup_ms"
                                ],
                                "per_request_full_path_search_ms": full[
                                    "cached_ann_search_ms"
                                ],
                            }
                        )
                        return row

                    cell_rows, cell_record = execute_checkpointed_cell(
                        checkpoint_dir=checkpoint_dir,
                        cell_id=cell_id,
                        requests=cell_requests,
                        repeat=repeat,
                        checkpoint_every=args.checkpoint_every,
                        resume=args.resume,
                        execute_request=execute_final,
                    )
                    final_cells.extend(cell_rows)
                    checkpoint_records.append(cell_record)
                    print(
                        f"complete {cell_id} rows={len(cell_rows)} "
                        f"resumed={cell_record['resumed_rows']}",
                        flush=True,
                    )

            final_rows = final_cells
            summary_rows = final_summary_table(
                final_rows,
                specs,
                targets,
                selected,
                measurement_by_filter,
                args.measurement_repeats,
                args.bootstrap_samples,
                args.bootstrap_seed,
                allow_lists,
                calibration_outcomes,
                methods,
            )
            atomic_write_csv(paths["final"], final_rows)
            atomic_write_csv(paths["summary"], summary_rows)

        raw_rows = [*setup_rows, *calibration_cells, *final_rows]
        atomic_write_csv(paths["raw"], raw_rows)
        validation_errors = artifact_validation_errors(
            calibration_rows,
            summary_rows,
            specs,
            ef_values,
            targets,
            methods,
        )
        invalid_raw = [
            str(row.get("checkpoint_cell", row.get("filter_name", "unknown")))
            for row in (*calibration_cells, *final_rows)
            if not _row_ok(row)
        ]
        continuous_errors = [
            str(row.get("checkpoint_cell", "unknown"))
            for row in final_rows
            if row.get("method") == FAISS_METHOD
            and not _bool_value(row.get("continuous_full_e2e_valid"))
        ]
        if invalid_raw:
            validation_errors.append(
                f"raw query errors={len(invalid_raw)} preview={invalid_raw[:5]}"
            )
        if continuous_errors:
            validation_errors.append(
                "continuous full-E2E errors="
                f"{len(continuous_errors)} preview={continuous_errors[:5]}"
            )
        incomplete_checkpoints = [
            record["cell_id"]
            for record in checkpoint_records
            if not record["complete"]
        ]
        expected_checkpoint_cells = (
            (len(ef_values) * args.calibration_repeats if faiss_enabled else 0)
            + len(sql_methods) * args.measurement_repeats
            + (
                len(targets) * args.measurement_repeats
                if faiss_enabled
                else 0
            )
        )
        observed_checkpoint_ids = {
            str(record["cell_id"]) for record in checkpoint_records
        }
        if incomplete_checkpoints:
            validation_errors.append(
                f"incomplete checkpoint cells: {incomplete_checkpoints[:5]}"
            )
        if (
            len(checkpoint_records) != expected_checkpoint_cells
            or len(observed_checkpoint_ids) != expected_checkpoint_cells
        ):
            validation_errors.append(
                "checkpoint cell coverage mismatch: "
                f"expected={expected_checkpoint_cells} "
                f"records={len(checkpoint_records)} "
                f"unique={len(observed_checkpoint_ids)}"
            )
        completion = completion_gate(
            calibration_rows,
            summary_rows,
            specs,
            ef_values,
            targets,
            methods,
            CURRENT_PROTOCOL,
        )
        validity = artifact_validity_flags(
            validation_errors,
            completion,
            formal_provenance_valid=not mismatched_hashes,
        )
        checkpoint_index = {
            "protocol_version": CHECKPOINT_PROTOCOL_VERSION,
            "status": (
                "complete"
                if checkpoint_records
                and all(record["complete"] for record in checkpoint_records)
                else "incomplete"
            ),
            "cells": checkpoint_records,
            "cells_complete": sum(
                bool(record["complete"]) for record in checkpoint_records
            ),
            "cells_total": len(checkpoint_records),
            "expected_cells": expected_checkpoint_cells,
        }
        checkpoint_index_path = checkpoint_dir / "index.json"
        atomic_write_json(checkpoint_index_path, checkpoint_index)
        manifest.update(validity)
        manifest["status"] = (
            "paper_eligible"
            if validity["paper_eligible"]
            else completion["status"]
            if validity["artifact_valid"]
            else "invalid"
        )
        manifest["validation_errors"] = validation_errors
        manifest["completion"] = completion
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["sql_first_exact_explain"] = explain_audit
        manifest["warmup_errors"] = warmup_errors
        manifest["selected_faiss_ef_search"] = {
            f"{filter_name}|{target:.2f}": ef_search
            for (filter_name, target), ef_search in selected.items()
        }
        manifest["checkpoints"] = {
            "index": file_identity(checkpoint_index_path, hash_contents=True),
            "cells_complete": checkpoint_index["cells_complete"],
            "cells_total": checkpoint_index["cells_total"],
            "expected_cells": expected_checkpoint_cells,
            "resumed_rows": sum(
                int(record["resumed_rows"]) for record in checkpoint_records
            ),
            "cell_raw_sha256": {
                record["cell_id"]: record["sha256"]
                for record in checkpoint_records
            },
        }
        manifest["row_counts"] = {
            "raw": len(raw_rows),
            "setup": len(setup_rows),
            "calibration_raw": len(calibration_cells),
            "calibration_summary": len(calibration_rows),
            "final_raw": len(final_rows),
            "summary": len(summary_rows),
            "raw_errors": len(invalid_raw),
            "continuous_full_e2e_errors": len(continuous_errors),
        }
        manifest["outputs"] = {
            "raw": {
                **file_identity(paths["raw"], hash_contents=True),
                "rows": len(raw_rows),
            },
            "calibration": {
                **file_identity(paths["calibration"], hash_contents=True),
                "rows": len(calibration_rows),
            },
            "final": {
                **file_identity(paths["final"], hash_contents=True),
                "rows": len(final_rows),
            },
            "summary": {
                **file_identity(paths["summary"], hash_contents=True),
                "rows": len(summary_rows),
            },
            "manifest": str(paths["manifest"].resolve()),
        }
        atomic_write_json(paths["manifest"], manifest)
        return paths
    except Exception as exc:
        manifest["diagnostic_valid"] = False
        manifest["artifact_valid"] = False
        manifest["paper_eligible"] = False
        manifest["status"] = "invalid"
        manifest["validation_errors"] = [
            f"fatal_error: {exc.__class__.__name__}: {exc}"
        ]
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["fatal_error"] = f"{exc.__class__.__name__}: {exc}"
        manifest["sql_first_exact_explain"] = explain_audit
        manifest["warmup_errors"] = warmup_errors
        atomic_write_json(paths["manifest"], manifest)
        raise


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.protocol == CURRENT_PROTOCOL:
        return _run_current(args)
    if args.protocol == LEGACY_PROTOCOL:
        return _run_legacy(args)
    raise ValueError(f"unknown protocol: {args.protocol}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched-recall Amazon-10M sql-first exact and Faiss HNSW allow-list baselines."
        )
    )
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default=CURRENT_PROTOCOL,
        help=(
            "current-q10k-r3 is the paper protocol; legacy-q100-r5 preserves "
            "the historical q80/r2 calibration and q100/r5 final path"
        ),
    )
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--truth-manifest", type=Path, default=DEFAULT_TRUTH_MANIFEST)
    parser.add_argument(
        "--calibration-workload-csv",
        type=Path,
        default=DEFAULT_CALIBRATION_WORKLOAD,
    )
    parser.add_argument(
        "--measurement-workload-csv",
        type=Path,
        default=DEFAULT_MEASUREMENT_WORKLOAD,
    )
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--faiss-index", type=Path, default=DEFAULT_FAISS_INDEX)
    parser.add_argument(
        "--faiss-index-manifest",
        type=Path,
        default=DEFAULT_FAISS_INDEX_MANIFEST,
    )
    parser.add_argument("--table", type=validate_table_name, default=DEFAULT_TABLE)
    parser.add_argument(
        "--candidate-validity-predicate",
        type=validate_candidate_validity_predicate,
        default=DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tag", default="20260718")
    parser.add_argument(
        "--methods",
        default=",".join(FORMAL_METHODS),
        help=(
            "comma-separated independent arms: "
            + ",".join(FORMAL_METHODS)
        ),
    )
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument("--rows", type=positive_int, default=FORMAL_ROWS)
    parser.add_argument("--k", type=positive_int, default=FORMAL_K)
    parser.add_argument("--ef-search-values", default=",".join(str(value) for value in DEFAULT_EF_SEARCH))
    parser.add_argument("--target-recalls", default=",".join(str(value) for value in DEFAULT_TARGETS))
    parser.add_argument(
        "--calibration-selection-policy",
        choices=tuple(TARGET_SELECTION_RULES),
        default="lcb_then_max_recall",
    )
    parser.add_argument("--calibration-query-offset", type=nonnegative_int, default=DEFAULT_CALIBRATION_QUERY_OFFSET)
    parser.add_argument("--calibration-queries", type=positive_int, default=DEFAULT_CALIBRATION_QUERIES)
    parser.add_argument("--calibration-repeats", type=positive_int, default=FORMAL_CALIBRATION_REPEATS)
    parser.add_argument("--final-query-offset", type=nonnegative_int, default=DEFAULT_FINAL_QUERY_OFFSET)
    parser.add_argument("--final-queries", type=positive_int, default=DEFAULT_FINAL_QUERIES)
    parser.add_argument("--final-repeats", type=positive_int, default=FORMAL_FINAL_REPEATS)
    parser.add_argument(
        "--measurement-repeats",
        type=positive_int,
        default=CURRENT_MEASUREMENT_REPEATS,
    )
    parser.add_argument("--bootstrap-samples", type=positive_int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument("--faiss-threads", type=positive_int, default=1)
    parser.add_argument("--allowlist-fetch-rows", type=positive_int, default=100_000)
    parser.add_argument("--warmup-queries", type=nonnegative_int, default=1)
    parser.add_argument("--progress-queries", type=nonnegative_int, default=25)
    parser.add_argument(
        "--checkpoint-every",
        type=positive_int,
        default=100,
        help="atomically persist each in-progress formal cell after this many requests",
    )
    parser.add_argument("--statement-timeout-ms", type=nonnegative_int, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the formal protocol and command without reading files or contacting PostgreSQL",
    )
    return parser


def formal_protocol_errors(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    try:
        parse_methods(args.methods)
    except argparse.ArgumentTypeError as exc:
        errors.append(f"methods: {exc}")
    shared_expected = {
        "rows": FORMAL_ROWS,
        "k": FORMAL_K,
        "calibration_repeats": FORMAL_CALIBRATION_REPEATS,
        "faiss_threads": 1,
        "calibration_selection_policy": "lcb_then_max_recall",
    }
    expected = dict(shared_expected)
    if args.protocol == CURRENT_PROTOCOL:
        expected.update(
            {
                "calibration_query_offset": DEFAULT_CALIBRATION_QUERY_OFFSET,
                "calibration_queries": DEFAULT_CALIBRATION_QUERIES,
                "final_query_offset": DEFAULT_FINAL_QUERY_OFFSET,
                "final_queries": DEFAULT_FINAL_QUERIES,
                "final_repeats": FORMAL_FINAL_REPEATS,
                "measurement_repeats": CURRENT_MEASUREMENT_REPEATS,
            }
        )
    elif args.protocol == LEGACY_PROTOCOL:
        expected.update(
            {
                "calibration_query_offset": DEFAULT_CALIBRATION_QUERY_OFFSET,
                "calibration_queries": DEFAULT_CALIBRATION_QUERIES,
                "final_query_offset": DEFAULT_FINAL_QUERY_OFFSET,
                "final_queries": DEFAULT_FINAL_QUERIES,
                "final_repeats": FORMAL_FINAL_REPEATS,
            }
        )
    else:
        errors.append(f"unknown protocol={args.protocol!r}")
    for name, wanted in expected.items():
        if getattr(args, name) != wanted:
            errors.append(f"{name}={getattr(args, name)!r}, expected {wanted!r}")
    if args.filter_names:
        errors.append("filter_names must be empty; formal baselines always use all 14 Amazon filters")
    if args.overwrite and args.resume:
        errors.append("--overwrite and --resume are mutually exclusive")
    try:
        ef_values = tuple(parse_int_csv(args.ef_search_values))
    except argparse.ArgumentTypeError as exc:
        errors.append(f"ef_search_values: {exc}")
    else:
        if ef_values != DEFAULT_EF_SEARCH:
            errors.append("ef_search_values must be the complete formal ef=20..100000 grid")
    try:
        targets = tuple(parse_targets(args.target_recalls))
    except argparse.ArgumentTypeError as exc:
        errors.append(f"target_recalls: {exc}")
    else:
        if targets != DEFAULT_TARGETS:
            errors.append("target_recalls must be exactly 0.90,0.95,0.99")
    return errors


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--protocol", args.protocol,
        "--filters-csv", str(args.filters_csv),
        "--truth-csv", str(args.truth_csv),
        "--truth-manifest", str(args.truth_manifest),
        "--calibration-workload-csv", str(args.calibration_workload_csv),
        "--measurement-workload-csv", str(args.measurement_workload_csv),
        "--fbin", str(args.fbin),
        "--faiss-index", str(args.faiss_index),
        "--faiss-index-manifest", str(args.faiss_index_manifest),
        "--table", args.table,
        "--out-dir", str(args.out_dir),
        "--tag", args.tag,
        "--methods", args.methods,
        "--calibration-selection-policy", args.calibration_selection_policy,
    ]
    if args.filter_names:
        command.extend(["--filter-names", *args.filter_names])
    errors = formal_protocol_errors(args)
    try:
        requested_methods = list(parse_methods(args.methods))
    except argparse.ArgumentTypeError:
        requested_methods = []
    return {
        "dry_run": True,
        "protocol": args.protocol,
        "side_effects": {"files_read": False, "files_written": False, "database": False, "faiss_index_loaded": False},
        "formal_protocol_valid": not errors,
        "formal_protocol_errors": errors,
        "command": command,
        "command_shell": shlex.join(command),
        "calibration_selection_policy": args.calibration_selection_policy,
        "target_selection": TARGET_SELECTION_RULES[
            args.calibration_selection_policy
        ],
        "requested_methods": requested_methods,
        "methods": {
            SQL_FIRST_CONTROL_METHOD: {
                "semantics": "historical control: MATERIALIZED predicate-valid rows followed by exact L2 ranking; HNSW use is rejected by EXPLAIN",
                "timing": "end-to-end PostgreSQL execute through fetchall(top-10)",
            },
            SQL_FIRST_PLANNER_METHOD: {
                "semantics": "PostgreSQL planner defaults choose the direct exact SQL path without a MATERIALIZED CTE; HNSW use is rejected by EXPLAIN",
                "timing": "end-to-end cursor.execute through fetchall(top-10), including PostgreSQL predicate/filter/exact ranking and client protocol; query-vector prefetch and EXPLAIN excluded",
            },
            SQL_FIRST_FORCED_METHOD: {
                "semantics": "enable_seqscan=off for this arm only; every predicate EXPLAIN must use a registered scalar index and no HNSW index",
                "timing": "same end-to-end exact-query boundary as planner-chosen SQL-first; planner SET/RESET and EXPLAIN are excluded",
            },
            FAISS_METHOD: {
                "semantics": "one complete, untruncated PostgreSQL ID stream builds a bitmap selector; ordinary Faiss IndexHNSWFlat traverses the full graph and admits only selector members",
                "setup_timing": {
                    "server_execution": "predicate result materialization in a transaction-local PostgreSQL table",
                    "row_transfer": "complete untruncated materialized ID stream to the client",
                    "bitmap_construction": "client bitmap population after transfer",
                    "selector_construction": "Faiss IDSelectorBitmap construction",
                    "full_setup": "wall time across all setup stages",
                    "full_setup_plus_search_e2e": (
                        "one continuous wall interval per final request from untruncated "
                        "SQL allow-list setup through the selected-ef ANN result"
                        if args.protocol == CURRENT_PROTOCOL
                        else "historical representative continuous setup-plus-first-search sample"
                    ),
                },
                "search_timing": "cached-list index.search only; allow-list construction and query-vector materialization excluded",
                "index": {"m": FORMAL_HNSW_M, "ef_construction": FORMAL_HNSW_EF_CONSTRUCTION, "manifest_required": True},
            },
        },
        "throughput": {
            "qps_reported": False,
            "latency_reciprocal_used_as_qps": False,
        },
        "calibration": (
            {
                "workload_csv": str(args.calibration_workload_csv),
                "requests": CURRENT_CALIBRATION_REQUESTS,
                "repeats": FORMAL_CALIBRATION_REPEATS,
                "selection": "independent FAISS efSearch LCB95 per filter and target",
            }
            if args.protocol == CURRENT_PROTOCOL
            else {"query_nos": [20, 99], "queries": 80, "repeats": 2}
        ),
        "final": (
            {
                "workload_csv": str(args.measurement_workload_csv),
                "requests": CURRENT_MEASUREMENT_REQUESTS,
                "repeats": CURRENT_MEASUREMENT_REPEATS,
                "order": "complete mixed trace in ascending request_no per arm/config repeat",
                "sql_first": "exact q10k/r3; no ANN matched configuration",
                "faiss": (
                    "independently selected target configuration; cached allow-list "
                    "search and continuous per-request full E2E are both measured"
                ),
            }
            if args.protocol == CURRENT_PROTOCOL
            else {"query_nos": [100, 199], "queries": 100, "repeats": 5}
        ),
        "checkpoint": (
            {
                "resumable": True,
                "granularity": "arm/config repeat",
                "partial_cell": "atomically persisted contiguous request_no prefix",
            }
            if args.protocol == CURRENT_PROTOCOL
            else {"resumable": False}
        ),
        "targets": list(DEFAULT_TARGETS),
        "ef_search": list(DEFAULT_EF_SEARCH),
        "ground_truth": "self-excluded exact SQL-valid top-10 with tie-aware squared-L2 threshold",
        "failure_gates": [
            *(
                ["complete FAISS calibration grid"]
                if FAISS_METHOD in requested_methods
                else []
            ),
            "complete held-out query/repeat pairs",
            "current q10200 GT/query cohort and filter content hashes",
            *(
                [
                    "frozen q200 calibration workload content hash",
                    "frozen q10k measurement workload content hash",
                    "per-cell checkpoint completeness and raw SHA-256",
                    "zero cached-search and continuous full-E2E request errors",
                ]
                if args.protocol == CURRENT_PROTOCOL
                else []
            ),
            *(
                [
                    "M32/efConstruction200 index and build-manifest content hashes",
                    "filter-membership validation",
                ]
                if FAISS_METHOD in requested_methods
                else []
            ),
            "exact-GT/query-cohort/database-relation/software identity binding",
            "finite latency/recall metrics", "LCB95 held-out target confirmation before publication",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_payload(args), indent=2, sort_keys=True))
        return 0
    paths = run(args)
    for name, path in paths.items():
        print(f"wrote {name}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
