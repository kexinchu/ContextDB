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
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Iterable, Mapping, Sequence

try:
    from .common_pg import pg_config_from_env, require_psycopg
    from . import amazon10m_sql_native_exact_truth as exact_truth_contract
except ImportError:  # Direct script execution puts this directory on sys.path.
    from common_pg import pg_config_from_env, require_psycopg  # type: ignore[no-redef]
    import amazon10m_sql_native_exact_truth as exact_truth_contract  # type: ignore[no-redef]

# Tie-metadata re-validation must match the producer's float32 faiss precision,
# so it uses the same numpy the exact-truth contract computes with.
_np = exact_truth_contract.require_numpy()


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILTERS = exact_truth_contract.DEFAULT_FILTERS
DEFAULT_SCHEMA = ROOT / "experiments/hybrid_vector_db/sql/amazon10m_sql_native_schema.sql"
DEFAULT_QUERY_IDS = exact_truth_contract.DEFAULT_QUERY_IDS
DEFAULT_QUERY_COHORT_MANIFEST = exact_truth_contract.DEFAULT_QUERY_COHORT_MANIFEST
DEFAULT_RESULTS = ROOT / "results/hybrid_vector_db"
DEFAULT_FBIN = ROOT / "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"
DEFAULT_EXACT_TRUTH_DIR = (
    DEFAULT_RESULTS / "amazon10m_sql_native_exact_truth_valid_embeddings"
)
DEFAULT_EXACT_TRUTH_CSV = DEFAULT_EXACT_TRUTH_DIR / "amazon10m_sql_native_exact_truth_q200.csv"
DEFAULT_EXACT_TRUTH_MANIFEST = DEFAULT_EXACT_TRUTH_DIR / "amazon10m_sql_native_exact_truth_manifest.json"
DEFAULT_VECTOR_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
DEFAULT_PRINCIPAL = "amazon10m_sql_native_benchmark"
DEFAULT_SOURCE_INDEX = "public.amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx"
DEFAULT_CLONE_INDEX = "public.amazon10m_hnsw_m32ef200_dupbridge_r29_bfs_idx"
# Compatibility for callers that imported the old constant. Formal artifacts use
# source_index and clone_index explicitly.
DEFAULT_VECTOR_INDEX = DEFAULT_SOURCE_INDEX
DEFAULT_K = 10
DEFAULT_CANDIDATE_VALIDITY_PREDICATE = (
    exact_truth_contract.DEFAULT_CANDIDATE_VALIDITY_PREDICATE
)
DEFAULT_CALIBRATION_QUERY_OFFSET = 20
DEFAULT_CALIBRATION_QUERIES = 80
DEFAULT_FINAL_QUERIES = 100
DEFAULT_CALIBRATION_REPEATS = 2
DEFAULT_FINAL_REPEATS = 5
P0_PROTOCOL = exact_truth_contract.PROTOCOL_Q10200
P0_CALIBRATION_QUERY_OFFSET = 20
P0_CALIBRATION_QUERIES = 80
P0_CALIBRATION_REPEATS = 2
P0_MEASUREMENT_QUERY_OFFSET = 200
P0_MEASUREMENT_QUERIES = 10_000
P0_MEASUREMENT_REPEATS = 3
P0_CONFIRMATION_QUERIES = 2_000
P0_CONFIRMATION_REPEATS = 1
P0_CONFIRMATION_SQL_FIRST_WORKERS = 8
P0_SCREENING_QUERIES = 1_000
P0_SCREENING_REPEATS = 1
P0_TARGET_RECALLS = (0.90,)
# Figure 5 is SQL-native hybrid search: one vector ORDER BY plus an explicit
# relational JOIN. The three families increase join depth: review facts,
# product catalog, then principal ACL. Filters vary JOIN-side selectivity.
P0_WORKLOAD_NAMES = (
    "join_facts",
    "join_catalog",
    "join_acl",
)
P0_FILTER_NAMES = (
    "grocery_helpful",
    "helpful_ge20",
    "grocery_long500",
)
P0_MIN_EF_SEARCH = 100
# Only heap predicates that appear in the executed SQL may be bound. The
# product-dimension EXISTS is correlated on amazon_product_dim, so
# item_rating_number is not implied on the review heap and must not be
# activated as a VisGuide atom (planner proof: predicate_not_implied).
WORKLOAD_GUIDANCE_ATOMS: dict[str, tuple[str, ...]] = {}
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_EF_SEARCH_VALUES = (
    20,
    40,
    60,
    80,
    100,
    150,
    200,
    250,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
)
DEFAULT_D3_PROBE_REQUESTS = 2
DEFAULT_D3_MIN_BENEFIT_PER_BYTE = 0.0
DEFAULT_D3_MAX_FRAGMENT_MB = 16
DEFAULT_D3_PAGE_MIN_SKIP_RATE = 0.05
TARGET_RECALLS = (0.90, 0.95, 0.99)
MODES = ("stock", "d1", "d1_d2", "d1_d2_d3")
SQLENS_MODES = MODES[1:]
SQL_FIRST_MODE = "sql_first_forced_indexed_exact"
P0_MODES = ("stock", SQL_FIRST_MODE, "d1_d2_d3")
P0_TUNABLE_MODES = ("stock", "d1_d2_d3")
ALL_MODES = MODES + (SQL_FIRST_MODE,)
NA = "N/A"
SQLENS_R43_BUILD_ID = (
    "sqlens-v17-predistance-promotion-20260806-r43"
)
SQLENS_R43_VECTOR_SO_SHA256 = (
    "2056a67b9b0012c401c6684d49915cbc31bc8fa770946dbfaddda9d779eecbf2"
)
# Compatibility aliases for direct callers. Formal validation is exact equality.
SQLENS_BUILD_PREFIX = SQLENS_R43_BUILD_ID
SQLENS_BUILD_PREFIXES = (SQLENS_R43_BUILD_ID,)
SQLENS_PROFILE_SEMANTICS = 9.0
BINARY_IDENTITY_REQUIRED_STAGES = (
    "experiment_start",
    "pre_exact_truth",
    "pre_calibration",
    "pre_final",
    "manifest_finalization",
)
CHECKPOINT_VERSION = 5
EXACT_TRUTH_ARTIFACT_VERSION = exact_truth_contract.CHECKPOINT_VERSION
EXACT_TRUTH_COMPATIBLE_VERSIONS = (4, EXACT_TRUTH_ARTIFACT_VERSION)
SQLENS_PROFILE_FIELDS = (
    "traversal_result_target",
    "traversal_guided_result_count",
    "traversal_max_scan_reached",
    "graph_elements_visited",
    "raw_index_tids_returned",
    "hnsw_am_callback_ms",
    "executor_residual_ms",
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
    "priority_reorders",
)
SQLENS_PROFILE_EXPORT_FIELDS = SQLENS_PROFILE_FIELDS + (
    "visited_tuples",
    "returned_tuples",
    "distance_compute_count",
    "idx_blks_hit",
    "idx_blks_read",
    "heap_blks_hit",
    "heap_blks_read",
)
TIMING_DEFINITION = (
    "activation_ms and query_ms are diagnostic sub-intervals. e2e_ms is one continuous "
    "client wall-clock interval from per-request as_of/guidance setup through the single "
    "PostgreSQL hybrid SELECT and result transfer; it is not reconstructed by addition. "
    "Approximate arms return an in-executor guidance proof column; the forced-indexed "
    "SQL-first arm is instead verified by a non-HNSW scalar-index EXPLAIN gate. Connection "
    "setup, EXPLAIN and exact-GT generation are outside e2e_ms. Primary selection and "
    "summaries use e2e_ms."
)

FILTER_COLUMNS = (
    "rating",
    "verified_purchase",
    "helpful_vote",
    "review_text_len",
    "store",
    "main_category",
    "category_id",
    "price",
    "has_price",
    "item_avg_rating",
    "item_rating_number",
)


@dataclass(frozen=True)
class FilterSpec:
    name: str
    target_rate: str
    predicate: str
    atoms: tuple[str, ...]
    expected_rows: int
    actual_pct: float


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    description: str
    bucket_pct: float
    temporal: bool
    width: str = "base"
    boolean_predicate: str = ""
    temporal_kind: str = "none"
    join_kind: str = "acl"


def binding_atoms_for(workload: WorkloadSpec, spec: FilterSpec) -> tuple[str, ...]:
    """Filter atoms plus any heap-local atoms implied by the SQL operator."""
    ordered = list(spec.atoms)
    seen = set(spec.atoms)
    for atom in WORKLOAD_GUIDANCE_ATOMS.get(workload.name, ()):
        if atom not in seen:
            ordered.append(atom)
            seen.add(atom)
    return tuple(ordered)


@dataclass(frozen=True)
class Config:
    ef_search: int
    max_scan_tuples: int
    scan_mem_multiplier: float
    iterative_scan: str
    guided_collect_target: int

    @property
    def label(self) -> str:
        if self.iterative_scan == "forced_indexed_exact":
            return SQL_FIRST_MODE
        mem = str(self.scan_mem_multiplier).replace(".", "p")
        return (
            f"ef{self.ef_search}_max{self.max_scan_tuples}_mem{mem}_"
            f"{self.iterative_scan}_target{self.guided_collect_target}"
        )


SQL_FIRST_CONFIG = Config(0, 0, 0.0, "forced_indexed_exact", 0)


@dataclass(frozen=True)
class ExactTruth:
    ids: tuple[int, ...]
    kth_distance: float
    tie_tolerance: float
    boundary_tied: bool


@dataclass(frozen=True)
class ModeSpec:
    index_role: str
    filter_strategy: str
    guidance_kind: str | None
    adaptive: bool
    guidance_semantics: str


MODE_SPECS = {
    "stock": ModeSpec("source", "off", None, False, "stock_pgvector"),
    "d1": ModeSpec(
        "source",
        "safe_guided",
        "bloom",
        False,
        "candidate_admission_and_validation_guidance",
    ),
    "d1_d2": ModeSpec(
        "clone",
        "safe_guided",
        "bloom",
        False,
        "candidate_admission_and_validation_guidance_on_same_graph_bfs_clone",
    ),
    "d1_d2_d3": ModeSpec(
        "clone",
        "safe_guided",
        "adaptive",
        True,
        "workload_driven_adaptive_candidate_admission_and_validation_guidance",
    ),
    SQL_FIRST_MODE: ModeSpec(
        "source",
        "forced_indexed_exact",
        None,
        False,
        "exact_sql_first_with_registered_scalar_index_and_no_hnsw",
    ),
}


def mode_index(mode: str, source_index: str, clone_index: str) -> str:
    try:
        role = MODE_SPECS[mode].index_role
    except KeyError as exc:
        raise ValueError(f"unknown benchmark mode: {mode}") from exc
    return source_index if role == "source" else clone_index


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


WORKLOADS = tuple(
    WorkloadSpec(
        workload.name,
        workload.description,
        workload.bucket_pct,
        workload.temporal_kind != "none",
        workload.width,
        workload.boolean_predicate,
        workload.temporal_kind,
        workload.join_kind,
    )
    for workload in exact_truth_contract.WORKLOADS
)


def select_workloads(
    names: Sequence[str], protocol: str = exact_truth_contract.PROTOCOL_Q200
) -> list[WorkloadSpec]:
    by_name = {workload.name: workload for workload in WORKLOADS}
    if names:
        order = list(names)
    elif protocol == P0_PROTOCOL:
        order = list(P0_WORKLOAD_NAMES)
    else:
        order = [workload.name for workload in WORKLOADS]
    missing = {name for name in order if name not in by_name}
    if missing:
        raise ValueError(f"unknown workloads: {sorted(missing)}")
    return [by_name[name] for name in order]


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


def parse_float_list(value: str, *, minimum: float = 0.0, maximum: float | None = None) -> list[float]:
    try:
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated number list") from exc
    if not parsed or any(item <= minimum or (maximum is not None and item > maximum) for item in parsed):
        raise argparse.ArgumentTypeError("number list contains an out-of-range value")
    return list(dict.fromkeys(parsed))


def parse_int_list(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("integer list values must be greater than zero")
    return list(dict.fromkeys(parsed))


def parse_word_list(value: str, allowed: set[str] | None = None) -> list[str]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    if not parsed or (allowed is not None and any(item not in allowed for item in parsed)):
        allowed_text = f"; allowed={sorted(allowed)}" if allowed is not None else ""
        raise argparse.ArgumentTypeError(f"expected a non-empty word list{allowed_text}")
    return list(dict.fromkeys(parsed))


def parse_guided_targets(value: str) -> list[str]:
    parsed = parse_word_list(value)
    if any(item != "ef" and (not item.isdigit() or int(item) <= 0) for item in parsed):
        raise argparse.ArgumentTypeError("guided collect targets must be positive integers or ef")
    return parsed


def parse_qualified_name(value: str) -> tuple[str, ...]:
    parts = tuple(value.split("."))
    if len(parts) not in (1, 2) or any(
        not part or not (part[0].isalpha() or part[0] == "_")
        or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$" for char in part)
        for part in parts
    ):
        raise argparse.ArgumentTypeError("table names must be unquoted table or schema.table identifiers")
    return tuple(part.lower() for part in parts)


def qualified_name_arg(value: str) -> str:
    return ".".join(parse_qualified_name(value))


def parse_role_name(value: str) -> str:
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        raise argparse.ArgumentTypeError("principal must be a lowercase PostgreSQL role identifier")
    return value


def expected_sha256_arg(value: str) -> str:
    normalized = str(value or "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise argparse.ArgumentTypeError(
            "expected a lowercase 64-character SHA256"
        )
    return normalized


def expected_sqlens_build_id_arg(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized != value or normalized != SQLENS_R43_BUILD_ID:
        raise argparse.ArgumentTypeError(
            f"expected the frozen r43 SQLens build ID {SQLENS_R43_BUILD_ID!r}"
        )
    return normalized


def require_execution_binary_identity(args: argparse.Namespace) -> tuple[str, str]:
    build_id = getattr(args, "expected_sqlens_build_id", None)
    vector_sha256 = getattr(args, "expected_vector_so_sha256", None)
    if not build_id or not vector_sha256:
        raise RuntimeError(
            "formal --execute requires --expected-sqlens-build-id and "
            "--expected-vector-so-sha256"
        )
    try:
        normalized = (
            expected_sqlens_build_id_arg(str(build_id)),
            expected_sha256_arg(str(vector_sha256)),
        )
        if normalized[1] != SQLENS_R43_VECTOR_SO_SHA256:
            raise argparse.ArgumentTypeError(
                "expected the frozen r43 vector.so SHA256 "
                + SQLENS_R43_VECTOR_SO_SHA256
            )
        return normalized
    except argparse.ArgumentTypeError as exc:
        raise RuntimeError(f"formal binary identity expectation is invalid: {exc}") from exc


validate_candidate_validity_predicate = (
    exact_truth_contract.validate_candidate_validity_predicate
)
candidate_universe_predicate_sha256 = (
    exact_truth_contract.candidate_universe_predicate_sha256
)
workload_scalar_predicate_sha256 = (
    exact_truth_contract.workload_scalar_predicate_sha256
)
query_cohort_sha256 = exact_truth_contract.query_cohort_sha256
relation_epoch_contract = exact_truth_contract.relation_epoch_contract


def parse_targets(value: str) -> list[float]:
    parsed = sorted(set(parse_float_list(value, minimum=0.0, maximum=1.0)))
    if any(target <= 0.0 or target > 1.0 for target in parsed):
        raise argparse.ArgumentTypeError("recall targets must be in (0, 1]")
    return parsed


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def bootstrap_bounds(values: Sequence[float], samples: int, seed: int) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1 or samples <= 0:
        return values[0], values[0], values[0]
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(list(values), k=len(values))) for _ in range(samples)]
    return percentile(means, 0.05), percentile(means, 0.025), percentile(means, 0.975)


def bootstrap_ratio_bounds(
    stock: dict[int, float], method: dict[int, float], samples: int, seed: int
) -> tuple[float, float, float]:
    keys = sorted(set(stock) & set(method))
    if not keys:
        return 0.0, 0.0, 0.0
    if len(keys) == 1 or samples <= 0:
        ratio = stock[keys[0]] / method[keys[0]] if method[keys[0]] > 0 else 0.0
        return ratio, ratio, ratio
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(samples):
        sampled = rng.choices(keys, k=len(keys))
        stock_mean = statistics.fmean(stock[key] for key in sampled)
        method_mean = statistics.fmean(method[key] for key in sampled)
        ratios.append(stock_mean / method_mean if method_mean > 0 else 0.0)
    return percentile(ratios, 0.05), percentile(ratios, 0.025), percentile(ratios, 0.975)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise RuntimeError(f"exact-truth artifact has invalid {label} SHA256")
    return normalized


def _csv_ints(value: Any, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise RuntimeError(f"exact-truth CSV has invalid {label}") from exc
    if not parsed:
        raise RuntimeError(f"exact-truth CSV has empty {label}")
    return parsed


def _csv_floats(value: Any, label: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise RuntimeError(f"exact-truth CSV has invalid {label}") from exc
    if not parsed or any(not math.isfinite(item) or item < 0.0 for item in parsed):
        raise RuntimeError(f"exact-truth CSV has invalid {label}")
    return parsed


def _artifact_error(message: str) -> RuntimeError:
    return RuntimeError(f"exact-truth artifact rejected: {message}")


def _artifact_float_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-9)


def _artifact_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise _artifact_error("truth CSV has invalid boolean metadata")


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def read_filters(path: Path, selected: set[str] | None = None) -> list[FilterSpec]:
    specs: list[FilterSpec] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            name = row["filter_name"]
            if selected and name not in selected:
                continue
            if name in seen:
                raise ValueError(f"duplicate filter_name: {name}")
            atoms = tuple(part.strip() for part in row["atoms"].split("||") if part.strip())
            if not atoms:
                raise ValueError(f"filter has no SQL atoms: {name}")
            specs.append(
                FilterSpec(
                    name=name,
                    target_rate=row["target_rate"],
                    predicate=row["predicate"].strip(),
                    atoms=atoms,
                    expected_rows=int(row["count"]),
                    actual_pct=float(row["actual_pct"]),
                )
            )
            seen.add(name)
    if selected and selected - seen:
        raise ValueError(f"missing filters: {sorted(selected - seen)}")
    if not specs:
        raise ValueError(f"no filters loaded from {path}")
    return specs


def load_query_ids(
    path: Path,
    offset: int,
    count: int,
    *,
    expected_split: str | None = None,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[int, int]:
    candidate_validity_predicate = validate_candidate_validity_predicate(
        candidate_validity_predicate
    )
    wanted = set(range(offset, offset + count))
    found: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "query_no", "query_id", "query_split", "self_excluded", "kth_distance_sq",
            "candidate_validity_predicate", "query_validity_predicate",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"query source uses the retired non-formal truth schema: missing={sorted(missing)}"
            )
        for row in reader:
            if row.get("method") not in (None, "", "pre_filter_exact"):
                continue
            if str(row["self_excluded"]).strip().lower() != "true":
                raise ValueError("query source did not exclude query rows")
            if (
                row.get("candidate_validity_predicate") != candidate_validity_predicate
                or row.get("query_validity_predicate") != candidate_validity_predicate
            ):
                raise ValueError("query source validity universe does not match the benchmark")
            query_no = int(row["query_no"])
            if query_no not in wanted:
                continue
            if expected_split is not None and row.get("query_split") != expected_split:
                raise ValueError(
                    f"query_no={query_no} has query_split={row.get('query_split')!r}; "
                    f"expected {expected_split!r}"
                )
            query_id = int(row["query_id"])
            previous = found.setdefault(query_no, query_id)
            if previous != query_id:
                raise ValueError(f"query_no={query_no} maps to multiple query IDs")
    if set(found) != wanted:
        raise ValueError(f"query split is incomplete: missing={sorted(wanted - set(found))}")
    if len(set(found.values())) != count:
        raise ValueError("query IDs must be unique within a split")
    return dict(sorted(found.items()))


def validate_query_splits(calibration: dict[int, int], final: dict[int, int]) -> None:
    if set(calibration) & set(final):
        raise ValueError("calibration and final query_no sets overlap")
    if set(calibration.values()) & set(final.values()):
        raise ValueError("calibration and final query IDs overlap")


def qualify_predicate(predicate: str, alias: str = "v") -> str:
    result = predicate
    for column in sorted(FILTER_COLUMNS, key=len, reverse=True):
        result = re.sub(rf"(?<![A-Za-z0-9_$.]){re.escape(column)}\b", f"{alias}.{column}", result)
    return result


def build_hybrid_sql(
    table: str,
    predicate: str,
    *,
    workload: WorkloadSpec | str | None = None,
    exact: bool = False,
    official_compatible: bool = False,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> str:
    """One SQL statement: vector ORDER BY plus real dimension, ACL and fact joins."""
    table = ".".join(parse_qualified_name(table))
    workload_name = workload.name if isinstance(workload, WorkloadSpec) else workload
    temporal_kind = (
        workload.temporal_kind
        if isinstance(workload, WorkloadSpec)
        else "none"
        if workload_name == "acl_only"
        else "grant"
        if workload_name == "grant_temporal_selectivity"
        else "fact"
    )
    if temporal_kind == "none":
        temporal_predicate = ""
    elif temporal_kind == "grant":
        temporal_predicate = """
  AND grant_row.valid_from <= %(as_of)s
  AND (grant_row.valid_to IS NULL OR grant_row.valid_to > %(as_of)s)"""
    elif temporal_kind == "fact":
        temporal_predicate = """
  AND fact.valid_from <= %(as_of)s
  AND (fact.valid_to IS NULL OR fact.valid_to > %(as_of)s)"""
    else:
        raise ValueError(f"unknown temporal workload kind: {temporal_kind}")
    boolean_sql = (
        exact_truth_contract.boolean_predicate(workload)
        if isinstance(workload, WorkloadSpec)
        else ""
    )
    binding_predicate = "" if exact or official_compatible else """
  AND (SELECT vector_hnsw_guidance_bind(
           %(vector_index)s::regclass,
           %(binding_atoms)s::text[],
           %(binding_kind)s
       ) OFFSET 0)"""
    candidate_validity = exact_truth_contract.qualify_candidate_validity_predicate(
        candidate_validity_predicate
    )
    query_id_predicate = (
        "v.id <> query_vector.query_id"
        if exact
        else "v.id <> %(query_id)s"
    )
    query_distance = (
        "v.embedding <-> query_vector.embedding"
        if exact
        else "v.embedding <-> %(query_embedding)s::vector"
    )
    grant_sql = (
        exact_truth_contract.grant_visibility_sql(workload)
        if isinstance(workload, WorkloadSpec)
        else """
  AND grant_row.principal_name = CURRENT_USER::text
  AND grant_row.can_read"""
    )
    joins = (
        exact_truth_contract.relation_join_sql(workload)
        if isinstance(workload, WorkloadSpec)
        else (
            "JOIN public.amazon_review_facts AS fact\n"
            "  ON fact.review_id = v.id\n"
            "JOIN public.amazon_product_dim AS product\n"
            "  ON product.parent_asin = fact.parent_asin\n"
            "JOIN public.amazon_principal_tenant_grants AS grant_row\n"
            "  ON grant_row.tenant_id = product.tenant_id"
        )
    )
    validity = f"""
  WHERE ({qualify_predicate(predicate)})
  AND ({candidate_validity})
  AND {query_id_predicate}
{binding_predicate}
{grant_sql}
{temporal_predicate}"""
    validity += boolean_sql
    if not exact:
        profile_column = (
            ""
            if official_compatible
            else ",\n       vector_hnsw_guidance_profile() AS execution_guidance_profile"
        )
        return f"""
SELECT v.id,
       {query_distance} AS distance{profile_column}
FROM {table} AS v
{joins}
{validity}
ORDER BY {query_distance}
LIMIT %(k)s
""".strip()
    return f"""
WITH query_vector AS (
    SELECT id AS query_id, embedding
    FROM {table} AS query_row
    WHERE query_row.id = %(query_id)s
      AND ({exact_truth_contract.qualify_candidate_validity_predicate(candidate_validity_predicate, 'query_row')})
), valid AS MATERIALIZED (
    SELECT v.id, v.embedding
    FROM {table} AS v
    {joins}
    CROSS JOIN query_vector
    {validity}
)
SELECT valid.id,
       valid.embedding <-> query_vector.embedding AS distance
FROM valid
CROSS JOIN query_vector
ORDER BY distance, valid.id
LIMIT %(k)s
""".strip()


def load_query_embeddings(
    cur: Any,
    table: str,
    query_ids: Sequence[int],
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[int, str]:
    """Load query vectors as postgres so approximate SQL can bind PARAM_EXTERN.

    r43 refuses planner-proof attachment on PARAM_EXEC (InitPlan). A bound
    ``vector`` parameter is folded to Const for safe_guided validation.
    """
    wanted = [int(query_id) for query_id in dict.fromkeys(query_ids)]
    if not wanted:
        return {}
    table = ".".join(parse_qualified_name(table))
    validity = exact_truth_contract.qualify_candidate_validity_predicate(
        candidate_validity_predicate, "query_row"
    )
    cur.execute(
        f"SELECT query_row.id, query_row.embedding::text "
        f"FROM {table} AS query_row "
        f"WHERE query_row.id = ANY(%s) AND ({validity})",
        (wanted,),
    )
    loaded = {int(row[0]): str(row[1]) for row in cur.fetchall() if row and row[1]}
    missing = [query_id for query_id in wanted if query_id not in loaded]
    if missing:
        raise RuntimeError(
            f"missing query embeddings for ids={missing[:8]} count={len(missing)}"
        )
    return loaded


def bind_query_embedding(
    params: dict[str, Any],
    query_id: int,
    query_embeddings: Mapping[int, str],
) -> dict[str, Any]:
    embedding = query_embeddings.get(int(query_id))
    if not embedding:
        raise RuntimeError(f"missing query embedding for query_id={query_id}")
    bound = dict(params)
    bound["query_embedding"] = embedding
    return bound


def validate_exact_sql_text(sql_text: str) -> None:
    normalized = sql_text.lower()
    forbidden = [token for token in ("vector_hnsw", "guidance_bind", "hnsw.") if token in normalized]
    if forbidden:
        raise RuntimeError(f"exact SQL contains approximate guidance/HNSW marker(s): {forbidden}")


def build_config_grid(args: argparse.Namespace, mode: str) -> list[Config]:
    configs: list[Config] = []
    targets = args.guided_collect_target_values
    for ef in args.ef_search_values:
        for max_scan in args.max_scan_tuples_values:
            for multiplier in args.scan_mem_multiplier_values:
                for iterative in args.iterative_scan_values:
                    for guided in targets:
                        target = ef if guided == "ef" else int(guided)
                        configs.append(Config(ef, max_scan, multiplier, iterative, target))
    if mode == "stock":
        unique: dict[tuple[int, int, float, str], Config] = {}
        for config in configs:
            unique.setdefault(
                (config.ef_search, config.max_scan_tuples, config.scan_mem_multiplier, config.iterative_scan),
                config,
            )
        configs = list(unique.values())
    unique_labels = {config.label: config for config in configs}
    return sorted(unique_labels.values(), key=lambda config: (config.ef_search, config.label))


def config_groups(configs: Sequence[Config]) -> list[tuple[int, list[Config]]]:
    grouped: dict[int, list[Config]] = {}
    for config in sorted(configs, key=lambda item: (item.ef_search, item.label)):
        grouped.setdefault(config.ef_search, []).append(config)
    return list(grouped.items())


def interleaved_schedule(
    keys: Sequence[tuple[Any, ...]], modes: Sequence[str], seed: int
) -> list[tuple[tuple[Any, ...], str]]:
    rng = random.Random(seed)
    schedule: list[tuple[tuple[Any, ...], str]] = []
    for key in keys:
        order = list(modes)
        rng.shuffle(order)
        schedule.extend((key, mode) for mode in order)
    return schedule


def partition_measurement_schedule(
    schedule: Sequence[tuple[tuple[Any, ...], str]],
    parallel_mode: str,
) -> tuple[
    list[tuple[int, tuple[Any, ...], str]],
    list[tuple[int, tuple[Any, ...], str]],
]:
    sequential: list[tuple[int, tuple[Any, ...], str]] = []
    parallel: list[tuple[int, tuple[Any, ...], str]] = []
    for position, (key, mode) in enumerate(schedule):
        item = (position, key, mode)
        if mode == parallel_mode:
            parallel.append(item)
        else:
            sequential.append(item)
    return sequential, parallel


def prepare_sql_first_session(
    cur: Any,
    principal: str,
    vector_table: str,
    source_index: str,
    clone_index: str,
) -> None:
    cur.execute("SET hnsw.guidance_require_epoch = on")
    ensure_sqlens_fragment_catalog(cur, principal, vector_table)
    cur.execute(f'SET ROLE "{principal}"')
    set_preferred_index(cur, mode_index(SQL_FIRST_MODE, source_index, clone_index))
    set_mode(cur, SQL_FIRST_MODE, SQL_FIRST_CONFIG, source_index)


def build_balanced_mixed_trace(
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    query_ids: Mapping[int, int],
    seed: int,
) -> list[dict[str, Any]]:
    cells = [
        (workload.name, spec.name)
        for workload in workloads
        for spec in filters
    ]
    if not cells or not query_ids:
        raise ValueError("balanced mixed trace requires non-empty cells and queries")
    ordered_queries = sorted((int(no), int(row_id)) for no, row_id in query_ids.items())
    rng = random.Random(seed)
    rng.shuffle(ordered_queries)
    cell_order = list(cells)
    rng.shuffle(cell_order)
    trace = [
        {
            "request_no": request_no,
            "workload": cell_order[request_no % len(cell_order)][0],
            "filter_name": cell_order[request_no % len(cell_order)][1],
            "query_no": query_no,
            "query_id": query_id,
        }
        for request_no, (query_no, query_id) in enumerate(ordered_queries)
    ]
    counts = {
        cell: sum(
            row["workload"] == cell[0] and row["filter_name"] == cell[1]
            for row in trace
        )
        for cell in cells
    }
    if max(counts.values()) - min(counts.values()) > 1:
        raise RuntimeError("balanced mixed trace cell counts differ by more than one")
    if len({int(row["query_no"]) for row in trace}) != len(trace):
        raise RuntimeError("balanced mixed trace reused a query vector")
    return trace


def mixed_trace_sha256(trace: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        [
            [
                int(row["request_no"]),
                str(row["workload"]),
                str(row["filter_name"]),
                int(row["query_no"]),
                int(row["query_id"]),
            ]
            for row in trace
        ]
    )


def trace_request_keys(
    trace: Sequence[Mapping[str, Any]], repeat: int
) -> list[tuple[str, str, int, int]]:
    return [
        (
            str(row["workload"]),
            str(row["filter_name"]),
            int(row["query_no"]),
            int(repeat),
        )
        for row in trace
    ]


def recall_at_k(ids: Sequence[int], truth: Sequence[int], k: int) -> float:
    expected = set(truth[:k])
    return len(expected & set(ids[:k])) / len(expected) if expected else 0.0


def distance_tolerance(distance: float) -> float:
    return max(1e-9, abs(distance) * 1e-6)


def tie_aware_recall_at_k(
    results: Sequence[tuple[int, float]], truth: ExactTruth, query_id: int, k: int
) -> float:
    seen: set[int] = set()
    qualifying = 0
    threshold = truth.kth_distance + truth.tie_tolerance
    for row_id, distance in results:
        row_id = int(row_id)
        if row_id == query_id or row_id in seen:
            continue
        seen.add(row_id)
        if float(distance) <= threshold:
            qualifying += 1
        if len(seen) == k:
            break
    return min(k, qualifying) / k


def _expected_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str, int, int]]:
    return {
        (str(row["workload"]), str(row["filter_name"]), int(row["query_no"]), int(row["repeat"]))
        for row in rows
    }


def adaptive_transition_evidence(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    adaptive_rows = [row for row in rows if row.get("mode") == "d1_d2_d3"]
    probes = sum(bool(row.get("adaptive_probe_observed")) for row in adaptive_rows)
    materializations = sum(
        bool(row.get("adaptive_materialized")) for row in adaptive_rows
    )
    active = sum(bool(row.get("adaptive_active")) for row in adaptive_rows)
    admissions = sum(bool(row.get("adaptive_admission_observed")) for row in adaptive_rows)
    hidden_reuse = sum(
        bool(row.get("hidden_prebuilt_fragment_reused")) for row in adaptive_rows
    )
    return {
        "required": bool(adaptive_rows),
        "rows": len(adaptive_rows),
        "probe_transitions": probes,
        "materialize_transitions": materializations,
        "admission_transitions": admissions,
        "active_requests": active,
        "hidden_prebuilt_fragment_reuse_requests": hidden_reuse,
        "valid": bool(
            adaptive_rows
            and probes > 0
            and materializations > 0
            and admissions > 0
            and active > 0
            and hidden_reuse == 0
        ),
    }


def grouped_adaptive_transition_evidence(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("mode") != "d1_d2_d3":
            continue
        key = (
            row.get("phase"),
            row.get("workload"),
            row.get("filter_name"),
            row.get("config"),
            row.get("target_recall"),
        )
        groups.setdefault(key, []).append(row)
    return [
        {
            "phase": key[0],
            "workload": key[1],
            "filter_name": key[2],
            "config": key[3],
            "target_recall": key[4],
            **adaptive_transition_evidence(group),
        }
        for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0]))
    ]


def summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_keys: set[tuple[str, str, int, int]],
    target_recall: float | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 20260718,
    timing_field: str = "e2e_ms",
    require_adaptive_evidence: bool = True,
) -> dict[str, Any]:
    ok = [row for row in rows if not row.get("error")]
    observed = _expected_keys(ok)
    complete = observed == expected_keys and len(ok) == len(rows) == len(expected_keys)
    adaptive_evidence = adaptive_transition_evidence(rows)
    if adaptive_evidence["required"] and require_adaptive_evidence:
        complete = complete and bool(adaptive_evidence["valid"])
    group = {(str(row["workload"]), str(row["filter_name"])) for row in rows}
    if len(group) != 1:
        raise ValueError("summarize_rows expects one workload/filter group")
    workload, filter_name = next(iter(group))
    by_query: dict[int, list[dict[str, Any]]] = {}
    for row in ok:
        by_query.setdefault(int(row["query_no"]), []).append(row)
    def timing_value(row: dict[str, Any]) -> float:
        value = row.get(timing_field, row.get("query_ms"))
        return float(value)

    latency_query = {
        query_no: statistics.fmean(timing_value(row) for row in items)
        for query_no, items in by_query.items()
    }
    recall_query = {
        query_no: statistics.fmean(float(row["recall"]) for row in items)
        for query_no, items in by_query.items()
    }
    recalls = list(recall_query.values())
    latencies = [timing_value(row) for row in ok]
    activation_values = [float(row.get("activation_ms", 0.0)) for row in ok]
    query_values = [float(row.get("query_ms", 0.0)) for row in ok]
    recall_lcb, recall_ci_low, recall_ci_high = bootstrap_bounds(
        recalls, bootstrap_samples, seed + 1
    )
    latency_ci_low, latency_ci_high = (0.0, 0.0)
    if latency_query:
        _, latency_ci_low, latency_ci_high = bootstrap_bounds(
            list(latency_query.values()), bootstrap_samples, seed + 2
        )
    recall_mean = statistics.fmean(recalls) if recalls else 0.0
    target_met = bool(
        complete
        and (
            target_recall is None
            or (recalls and float(recall_lcb) >= float(target_recall))
        )
    )
    numeric = complete and (target_recall is None or target_met)
    result: dict[str, Any] = {
        "workload": workload,
        "filter_name": filter_name,
        "rows": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "queries": len(latency_query),
        "complete": complete,
        "target_recall": target_recall if target_recall is not None else "",
        "target_met": target_met,
        "recall_mean": recall_mean if recalls else NA,
        "recall_lcb95": recall_lcb if recalls else NA,
        "recall_ci95_low": recall_ci_low if recalls else NA,
        "recall_ci95_high": recall_ci_high if recalls else NA,
        "activation_mean_ms": statistics.fmean(activation_values) if numeric and activation_values else NA,
        "query_mean_ms": statistics.fmean(query_values) if numeric and query_values else NA,
        "primary_timing_field": timing_field,
        "latency_mean_ms": statistics.fmean(latencies) if numeric and latencies else NA,
        "latency_p50_ms": statistics.median(latencies) if numeric and latencies else NA,
        "latency_p95_ms": percentile(latencies, 0.95) if numeric and latencies else NA,
        "latency_p99_ms": percentile(latencies, 0.99) if numeric and latencies else NA,
        "latency_ci95_low_ms": latency_ci_low if numeric and latencies else NA,
        "latency_ci95_high_ms": latency_ci_high if numeric and latencies else NA,
        "status": "complete" if numeric else NA,
        "query_latency_definition": TIMING_DEFINITION,
        "adaptive_transition_evidence": adaptive_evidence,
        "adaptive_mode_active": (
            bool(adaptive_evidence["valid"]) if adaptive_evidence["required"] else NA
        ),
    }
    return result


def select_config(summaries: Sequence[dict[str, Any]], target: float) -> dict[str, Any] | None:
    eligible = [
        row
        for row in summaries
        if bool(row.get("complete"))
        and bool(row.get("target_met"))
        and row.get("latency_mean_ms") not in (None, NA)
        and float(row.get("recall_lcb95", -1.0)) >= target
    ]
    return min(eligible, key=lambda row: (float(row["latency_mean_ms"]), str(row["config"]))) if eligible else None


def calibration_outcome(
    summaries: Sequence[dict[str, Any]],
    configs: Sequence[Config],
    executed_labels: Sequence[str],
    targets: Sequence[float],
) -> dict[str, Any]:
    planned_labels = [config.label for config in configs]
    if list(executed_labels) != planned_labels[: len(executed_labels)]:
        raise ValueError("calibration blocks are not a prefix of the ef-ordered grid")
    by_target = {
        float(target): select_config(
            [row for row in summaries if float(row["target_recall"]) == float(target)],
            float(target),
        )
        for target in targets
    }
    highest_target = max(float(target) for target in targets)
    stopped = by_target[highest_target] is not None
    grid_exhausted = len(executed_labels) == len(planned_labels)
    error_free = all(bool(row.get("complete")) and int(row.get("errors", 0)) == 0 for row in summaries)
    unattainable = [
        target
        for target, choice in by_target.items()
        if choice is None and grid_exhausted and error_free
    ]
    return {
        "planned_blocks": len(planned_labels),
        "executed_blocks": len(executed_labels),
        "executed_labels": list(executed_labels),
        "stopped": stopped,
        "stop_reason": "highest_target_attained" if stopped else "grid_exhausted" if grid_exhausted else "in_progress",
        "grid_exhausted": grid_exhausted,
        "error_free_grid_exhaustion": bool(grid_exhausted and error_free),
        "attainable_targets": [target for target, choice in by_target.items() if choice is not None],
        "unattainable_on_grid": unattainable,
        "indeterminate_targets": [
            target
            for target, choice in by_target.items()
            if choice is None and target not in unattainable
        ],
        "selected": by_target,
    }


def lcb_calibration_outcome(
    summaries: Sequence[dict[str, Any]],
    configs: Sequence[Config],
    executed_labels: Sequence[str],
    targets: Sequence[float],
) -> dict[str, Any]:
    """Calibration proof for a baseline selected by query-level LCB95."""
    outcome = calibration_outcome(summaries, configs, executed_labels, targets)
    selected: dict[float, dict[str, Any] | None] = {}
    for target in targets:
        eligible = [
            row
            for row in summaries
            if bool(row.get("complete"))
            and int(row.get("errors", 0) or 0) == 0
            and float(row.get("recall_lcb95", -1.0)) >= float(target)
            and row.get("latency_mean_ms") not in (None, NA)
        ]
        selected[float(target)] = (
            min(
                eligible,
                key=lambda row: (float(row["latency_mean_ms"]), str(row["config"])),
            )
            if eligible
            else None
        )
    exhausted = bool(outcome["grid_exhausted"] and outcome["error_free_grid_exhaustion"])
    outcome["selected"] = selected
    outcome["attainable_targets"] = [target for target, row in selected.items() if row]
    outcome["unattainable_on_grid"] = [
        target for target, row in selected.items() if row is None and exhausted
    ]
    outcome["indeterminate_targets"] = [
        target for target, row in selected.items() if row is None and not exhausted
    ]
    outcome["selection_rule"] = "lowest_mean_latency_among_lcb95_qualified_configs"
    return outcome


def common_attainable_targets(
    outcomes: Sequence[dict[str, Any]], targets: Sequence[float]
) -> list[float]:
    return [
        float(target)
        for target in targets
        if outcomes
        and all(outcome["selected"].get(float(target)) is not None for outcome in outcomes)
    ]


def preregister_formal_matrix(
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    targets: Sequence[float],
    modes: Sequence[str] = MODES,
) -> list[dict[str, Any]]:
    return [
        {
            "workload": workload.name,
            "filter_name": spec.name,
            "target_recall": float(target),
            "status": "preregistered",
            "reason": "awaiting_independent_calibration",
            "selected_configs": {mode: None for mode in modes},
        }
        for workload in workloads
        for spec in filters
        for target in targets
    ]


def preregister_official_formal_matrix(
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    targets: Sequence[float],
) -> list[dict[str, Any]]:
    """Register every upstream-pgvector cell before calibration starts."""
    return [
        {
            "workload": workload.name,
            "filter_name": spec.name,
            "target_recall": float(target),
            "status": "preregistered",
            "reason": "awaiting_complete_calibration_grid",
            "selected_config": None,
            "calibration_proof": {},
            "final_proof": {},
        }
        for workload in workloads
        for spec in filters
        for target in targets
    ]


def finalize_official_formal_matrix(
    preregistered: Sequence[dict[str, Any]],
    outcomes: Mapping[tuple[str, str], Mapping[str, Any]],
    final_summaries: Mapping[tuple[str, str, float], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Make every registered official cell either held-out complete or proven NA.

    A missing selected configuration is not enough to call a target unattainable:
    the complete ordered grid must have executed without errors.  This is kept
    separate from paper eligibility because a correctly measured NA is a valid
    diagnostic artifact but cannot support a full target-matched comparison.
    """
    matrix: list[dict[str, Any]] = []
    for source in preregistered:
        cell = dict(source)
        workload = str(cell["workload"])
        filter_name = str(cell["filter_name"])
        target = float(cell["target_recall"])
        outcome = outcomes.get((workload, filter_name))
        if outcome is None:
            cell.update(status="invalid", reason="missing_calibration_outcome")
            matrix.append(cell)
            continue
        selected = outcome.get("selected", {}).get(target)
        calibration_proof = {
            "planned_blocks": outcome.get("planned_blocks"),
            "executed_blocks": outcome.get("executed_blocks"),
            "grid_exhausted": outcome.get("grid_exhausted"),
            "error_free_grid_exhaustion": outcome.get("error_free_grid_exhaustion"),
        }
        cell["calibration_proof"] = calibration_proof
        if isinstance(selected, Mapping):
            cell["selected_config"] = selected.get("config")
            final = final_summaries.get((workload, filter_name, target))
            if final is None:
                cell.update(status="invalid", reason="attainable_final_missing")
            else:
                final_proof = {
                    "complete": bool(final.get("complete")),
                    "errors": int(final.get("errors", 0) or 0),
                    "target_met": bool(final.get("target_met")),
                    "recall_mean": final.get("recall_mean"),
                    "recall_lcb95": final.get("recall_lcb95"),
                    "config": final.get("config"),
                }
                cell["final_proof"] = final_proof
                if (
                    final_proof["complete"]
                    and final_proof["errors"] == 0
                    and final_proof["target_met"]
                ):
                    cell.update(status="complete", reason="held_out_target_met")
                else:
                    cell.update(status="invalid", reason="held_out_target_not_met")
        elif target in outcome.get("unattainable_on_grid", []):
            if outcome.get("grid_exhausted") and outcome.get(
                "error_free_grid_exhaustion"
            ):
                cell.update(status=NA, reason="unattainable_on_complete_grid")
            else:
                cell.update(status="invalid", reason="unattainable_without_complete_grid_proof")
        else:
            cell.update(status="invalid", reason="calibration_indeterminate")
        matrix.append(cell)
    return matrix


def formal_matrix_coverage(
    matrix: Sequence[Mapping[str, Any]],
    expected_cells: int,
) -> dict[str, Any]:
    statuses = [str(cell.get("status", "")) for cell in matrix]
    complete = sum(status == "complete" for status in statuses)
    unattainable = sum(status == NA for status in statuses)
    invalid = sum(status not in {"complete", NA} for status in statuses)
    return {
        "expected_cells": expected_cells,
        "observed_cells": len(matrix),
        "complete_cells": complete,
        "unattainable_cells": unattainable,
        "invalid_or_missing_cells": invalid + max(0, expected_cells - len(matrix)),
        "coverage_complete": len(matrix) == expected_cells and invalid == 0,
        "all_targets_attained": len(matrix) == expected_cells and complete == expected_cells,
        "matrix_sha256": canonical_sha256(list(matrix)),
    }


def finalize_formal_matrix(
    preregistered: Sequence[dict[str, Any]],
    outcomes: dict[tuple[str, str, str], dict[str, Any]],
    completed_final: set[tuple[str, str, float]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in preregistered:
        cell = dict(source)
        workload = str(cell["workload"])
        filter_name = str(cell["filter_name"])
        target = float(cell["target_recall"])
        mode_outcomes = [outcomes.get((workload, filter_name, mode)) for mode in MODES]
        if any(outcome is None for outcome in mode_outcomes):
            cell.update(status="invalid", reason="missing_calibration_outcome")
            result.append(cell)
            continue
        selected = {
            mode: outcome["selected"].get(target)  # type: ignore[index]
            for mode, outcome in zip(MODES, mode_outcomes)
        }
        cell["selected_configs"] = {
            mode: choice.get("config") if isinstance(choice, dict) else None
            for mode, choice in selected.items()
        }
        if all(choice is not None for choice in selected.values()):
            if (workload, filter_name, target) in completed_final:
                cell.update(status="complete", reason="held_out_final_complete")
            else:
                cell.update(status="invalid", reason="attainable_final_missing")
        elif all(
            target in outcome.get("unattainable_on_grid", [])  # type: ignore[union-attr]
            for outcome in mode_outcomes
        ):
            cell.update(status=NA, reason="unattainable_on_grid")
        elif any(
            target in outcome.get("indeterminate_targets", [])  # type: ignore[union-attr]
            for outcome in mode_outcomes
        ):
            cell.update(status="invalid", reason="calibration_indeterminate")
        else:
            cell.update(status=NA, reason="not_jointly_attainable")
        result.append(cell)
    return result


def artifact_validation_errors(
    expected_final_blocks: int,
    summaries: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]] = (),
    plans: Sequence[dict[str, Any]] = (),
    formal_matrix: Sequence[dict[str, Any]] = (),
) -> list[str]:
    errors: list[str] = []
    if formal_matrix and any(
        cell.get("status") not in {"complete", NA} for cell in formal_matrix
    ):
        errors.append("pre-registered workload/filter/target matrix is unresolved")
    invalid_final = [
        f"{row['workload']}|{row['filter_name']}|target={row['target_recall']}"
        for row in summaries
        if row.get("phase") == "final"
        and str(row.get("mode", "")).startswith("paired_")
        and row.get("status") != "complete"
    ]
    if invalid_final:
        errors.append(
            "held-out matched-recall validation failed: " + ",".join(invalid_final)
        )
    paired_final = [
        row
        for row in summaries
        if row.get("phase") == "final"
        and str(row.get("mode", "")).startswith("paired_")
    ]
    paired_groups: dict[tuple[Any, Any, Any], set[str]] = {}
    for row in paired_final:
        key = (row.get("workload"), row.get("filter_name"), row.get("target_recall"))
        paired_groups.setdefault(key, set()).add(str(row.get("mode"))[len("paired_") :])
    if (
        len(paired_final) != expected_final_blocks * len(SQLENS_MODES)
        or len(paired_groups) != expected_final_blocks
        or any(modes != set(SQLENS_MODES) for modes in paired_groups.values())
    ):
        errors.append("held-out comparison matrix is incomplete for D1/D2/D3")
    unsafe_rows = [
        str(row.get("pair_key", "unknown"))
        for row in rows
        if row.get("mode") in SQLENS_MODES
        and (
            row.get("filter_strategy") != "safe_guided"
            or row.get("guidance_semantics")
            != MODE_SPECS[str(row.get("mode"))].guidance_semantics
            or bool(row.get("hard_traversal_used"))
        )
    ]
    if unsafe_rows:
        errors.append(
            "join/RLS workload used unsafe or mislabeled guidance: "
            + ",".join(unsafe_rows[:10])
        )
    unproven_rows = [
        str(row.get("pair_key", "unknown"))
        for row in rows
        if not recorded_guidance_proof_is_valid(row)
    ]
    if unproven_rows:
        errors.append(
            "per-row guidance binding/effective/scan/final-path proof failed: "
            + ",".join(unproven_rows[:10])
        )
    d3_store_mismatches = [
        str(row.get("pair_key", "unknown"))
        for row in rows
        if row.get("mode") == "d1_d2_d3"
        and (
            row.get("persistent_fragment_reset_proof", {}).get("valid") is not True
            or int(row.get("prebuilt_fragments", -1)) != 0
        )
    ]
    if d3_store_mismatches:
        errors.append(
            "D3 block did not start from an audited empty persistent store: "
            + ",".join(d3_store_mismatches[:10])
        )
    context_mismatches = [
        str(row.get("pair_key", "unknown"))
        for row in rows
        if not row.get("principal")
        or str(row.get("snapshot_as_of", "")) != str(row.get("as_of", ""))
        or row.get("preferred_index_current_setting")
        != row.get("selected_vector_index")
        or row.get("page_access_current_setting") != "off"
        or row.get("index_page_access_current_setting") != "off"
    ]
    principals = {str(row.get("principal")) for row in rows if row.get("principal")}
    if context_mismatches or len(principals) > 1:
        errors.append(
            "principal/snapshot/preferred-index/prefetch runtime context mismatch: "
            + ",".join(context_mismatches[:10])
        )
    invalid_plans = [
        f"{plan.get('phase')}|{plan.get('workload')}|{plan.get('filter_name')}|{plan.get('mode')}"
        for plan in plans
        if plan.get("mode") in MODES
        and (
            not bool(plan.get("explain_gate", {}).get("valid"))
            or plan.get("selected_vector_index")
            != plan.get("explain_gate", {}).get("expected_index_qualified")
            or plan.get("preferred_index_current_setting")
            != plan.get("selected_vector_index")
            or plan.get("page_access_current_setting") != "off"
            or plan.get("index_page_access_current_setting") != "off"
            or plan.get("explain_order") != "after_all_timed_requests_in_block"
        )
    ]
    if invalid_plans:
        errors.append("wrong or unproven vector index: " + ",".join(invalid_plans[:10]))
    return errors


def database_contract_errors(database: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    benchmark_modes = tuple(database.get("benchmark_modes") or MODES)
    identity = database.get("binary_identity_gate")
    evidence = identity.get("gate_evidence") if isinstance(identity, dict) else None
    recomputed_identity: dict[str, Any] | None = None
    if isinstance(identity, dict) and isinstance(evidence, list):
        try:
            recomputed_identity = binary_identity_gate_summary(
                str(identity.get("expected_sqlens_build_id", "")),
                str(identity.get("expected_vector_so_sha256", "")),
                evidence,
                benchmark_modes,
            )
        except (TypeError, ValueError):
            recomputed_identity = None
    if (
        not isinstance(identity, dict)
        or identity.get("valid") is not True
        or identity.get("all_exact_match") is not True
        or not isinstance(evidence, list)
        or not evidence
        or len(evidence) != identity.get("evidence_count")
        or recomputed_identity != identity
        or identity.get("missing_required_stages") != []
        or identity.get("missing_required_connections") != []
        or str(identity.get("expected_sqlens_build_id", ""))
        != SQLENS_R43_BUILD_ID
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(identity.get("expected_vector_so_sha256", ""))
        )
        or identity.get("observed_vector_so_sha256")
        != identity.get("expected_vector_so_sha256")
        or identity.get("observed_sqlens_build_id")
        != identity.get("expected_sqlens_build_id")
    ):
        errors.append(
            "exact SQLens build ID/server vector.so binary identity gate is missing or invalid"
        )
    proof = database.get("d2_graph_proof")
    comparison = proof.get("comparison") if isinstance(proof, dict) else None
    if (
        not isinstance(proof, dict)
        or proof.get("valid") is not True
        or not isinstance(comparison, dict)
        or any(
            comparison.get(field) is not True
            for field in (
                "same_heap",
                "logical_equal",
                "entry_equal",
                "tuple_coverage_equal",
                "definition_equal",
            )
        )
        or comparison.get("physical_equal") is not False
    ):
        errors.append("D2 same-heap/same-logical-graph BFS proof is missing or invalid")
    end_proof = database.get("d2_graph_proof_end")
    if end_proof != proof:
        errors.append("D2 graph proof changed or was not revalidated at formal-run end")
    start_relations = database.get("relations", {})
    end_indexes = database.get("d2_index_fingerprints_end", {})
    start_indexes = {
        index: start_relations.get(index)
        for index in database.get("d2_index_names", [])
    }
    if not start_indexes or end_indexes != start_indexes:
        errors.append("source/clone index fingerprints changed during the formal run")
    settings = database.get("preferred_index_current_settings")
    mode_indexes = database.get("mode_indexes")
    if (
        not isinstance(settings, dict)
        or not isinstance(mode_indexes, dict)
        or set(settings) != set(benchmark_modes)
        or set(mode_indexes) != set(benchmark_modes)
        or any(
            settings.get(mode) != mode_indexes.get(mode)
            for mode in benchmark_modes
        )
    ):
        errors.append("per-mode hnsw.preferred_index current_setting proof is invalid")
    reset = database.get("d3_startup_reset_evidence")
    try:
        prebuilt_fragments = int(reset.get("prebuilt_fragments", -1))
    except (AttributeError, TypeError, ValueError):
        prebuilt_fragments = -1
    if (
        not isinstance(reset, dict)
        or reset.get("after_reset_empty") is not True
        or prebuilt_fragments != 0
    ):
        errors.append("D3 did not start from an audited empty workload-driven cache")
    persistent_reset = database.get("d3_persistent_fragment_reset")
    if (
        not isinstance(persistent_reset, dict)
        or persistent_reset.get("valid") is not True
        or int(persistent_reset.get("prebuilt_fragments", -1)) != 0
        or not isinstance(database.get("d3_fragment_store_end"), dict)
    ):
        errors.append("D3 persistent fragment store reset/count/hash audit is invalid")
    data_guard = database.get("formal_data_version_proof")
    if (
        not isinstance(data_guard, dict)
        or data_guard.get("valid") is not True
        or data_guard.get("start_hash") != data_guard.get("end_hash")
    ):
        errors.append("formal GT/mode execution data-version guard is missing or invalid")
    security = database.get("rls_security_proofs")
    if not isinstance(security, dict) or set(security) != set(benchmark_modes):
        errors.append("per-session RLS principal/policy/probe proof is incomplete")
    else:
        for mode in benchmark_modes:
            try:
                validate_rls_security_proof(security[mode], str(database.get("principal", "")))
            except RuntimeError:
                errors.append(f"RLS security proof is invalid for mode={mode}")
    return errors


def validate_paired_query_contract(
    stock_rows: Sequence[dict[str, Any]], method_rows: Sequence[dict[str, Any]]
) -> None:
    def contract(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[Any, ...]]:
        return {
            str(row["pair_key"]): (
                int(row["query_id"]),
                str(row["predicate"]),
                str(row["query_sql_sha256"]),
                str(row["exact_gt_ids"]),
                str(row.get("exact_gt_kth_distance", "")),
                str(row.get("exact_gt_tie_tolerance", "")),
                str(row.get("exact_gt_boundary_tied", "")),
                str(row.get("principal", "")),
                str(row.get("as_of", "")),
                str(row.get("snapshot_as_of", "")),
            )
            for row in rows
        }

    stock_contract = contract(stock_rows)
    method_contract = contract(method_rows)
    for pair_key in set(stock_contract) & set(method_contract):
        if stock_contract[pair_key] != method_contract[pair_key]:
            raise RuntimeError(f"paired SQL/GT contract mismatch for {pair_key}")


def paired_summary(
    stock_rows: Sequence[dict[str, Any]],
    method_rows: Sequence[dict[str, Any]],
    *,
    expected_keys: set[tuple[str, str, int, int]],
    target_recall: float,
    bootstrap_samples: int,
    seed: int,
    method_mode: str = "d1",
    require_adaptive_evidence: bool = True,
) -> dict[str, Any]:
    validate_paired_query_contract(stock_rows, method_rows)
    stock = summarize_rows(
        stock_rows, expected_keys=expected_keys, target_recall=target_recall,
        bootstrap_samples=bootstrap_samples, seed=seed,
        require_adaptive_evidence=require_adaptive_evidence,
    )
    method = summarize_rows(
        method_rows, expected_keys=expected_keys, target_recall=target_recall,
        bootstrap_samples=bootstrap_samples, seed=seed + 10,
        require_adaptive_evidence=require_adaptive_evidence,
    )
    stock_by_query: dict[int, list[float]] = {}
    method_by_query: dict[int, list[float]] = {}
    for row in stock_rows:
        if not row.get("error"):
            stock_by_query.setdefault(int(row["query_no"]), []).append(float(row.get("e2e_ms", row["query_ms"])))
    for row in method_rows:
        if not row.get("error"):
            method_by_query.setdefault(int(row["query_no"]), []).append(float(row.get("e2e_ms", row["query_ms"])))
    stock_q = {query: statistics.fmean(values) for query, values in stock_by_query.items()}
    method_q = {query: statistics.fmean(values) for query, values in method_by_query.items()}
    paired = set(stock_q) & set(method_q)
    valid = (
        stock["status"] != NA
        and method["status"] != NA
        and paired == {key[2] for key in expected_keys}
    )
    if valid:
        speed_lcb, speed_low, speed_high = bootstrap_ratio_bounds(
            stock_q, method_q, bootstrap_samples, seed + 20
        )
        deltas = [stock_q[query] - method_q[query] for query in sorted(paired)]
        _, delta_low, delta_high = bootstrap_bounds(deltas, bootstrap_samples, seed + 21)
        paired_values: dict[str, Any] = {
            "paired_queries": len(paired),
            "speedup_vs_stock": statistics.fmean(stock_q.values()) / statistics.fmean(method_q.values()),
            "speedup_lcb95": speed_lcb,
            "speedup_ci95_low": speed_low,
            "speedup_ci95_high": speed_high,
            "paired_latency_saving_mean_ms": statistics.fmean(deltas),
            "paired_latency_saving_ci95_low_ms": delta_low,
            "paired_latency_saving_ci95_high_ms": delta_high,
        }
    else:
        paired_values = {
            "paired_queries": len(paired),
            "speedup_vs_stock": NA,
            "speedup_lcb95": NA,
            "speedup_ci95_low": NA,
            "speedup_ci95_high": NA,
            "paired_latency_saving_mean_ms": NA,
            "paired_latency_saving_ci95_low_ms": NA,
            "paired_latency_saving_ci95_high_ms": NA,
        }
    return {
        "stock": stock,
        method_mode: method,
        **paired_values,
        "status": "complete" if valid else NA,
    }


def configure_hnsw_driven_planner(cur: Any) -> None:
    """Force HNSW as the driving access path on join/RLS SQL.

    ``hnsw.preferred_index`` only prices competing HNSW indexes to infinity.
    With ACL/grant joins, PostgreSQL otherwise starts from the grant/product
    side and uses the reviews primary key, which fails the approximate EXPLAIN
    gate. Pin the written FROM order (vector heap first) and disable seq/bitmap
    scans so ``ORDER BY embedding <-> query`` selects the preferred HNSW index.
    EXISTS/NOT EXISTS still keep heap btree and primary-key scans attractive;
    those indexes are hidden for the approximate arm by
    ``set_heap_competing_indexes_valid``.
    """
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET enable_bitmapscan = off")
    cur.execute("SET enable_indexscan = on")
    cur.execute("SET join_collapse_limit = 1")
    cur.execute("SET from_collapse_limit = 1")


def set_heap_competing_indexes_valid(
    cur: Any, vector_table: str, *, valid: bool
) -> list[str]:
    """Hide or restore every non-HNSW heap index, including the primary key.

    EXISTS SQL otherwise starts from ``helpful_vote`` or the heap PK after the
    cheaper btree is hidden. Catalog-global, so SQL-first must run after the
    approximate arm and this must be restored in ``finally``. Join indexes on
    facts/products/grants stay valid. HNSW indexes stay valid.
    """
    heap_name = parse_qualified_name(vector_table)[-1]
    restored_role = _restore_session_user(cur)
    try:
        cur.execute(
            """
            UPDATE pg_index AS idx
            SET indisvalid = %s
            FROM pg_class AS index_rel,
                 pg_class AS heap_rel,
                 pg_namespace AS ns,
                 pg_am AS am
            WHERE idx.indexrelid = index_rel.oid
              AND idx.indrelid = heap_rel.oid
              AND heap_rel.relnamespace = ns.oid
              AND index_rel.relam = am.oid
              AND ns.nspname = 'public'
              AND heap_rel.relname = %s
              AND am.amname <> 'hnsw'
            RETURNING index_rel.relname
            """,
            (valid, heap_name),
        )
        names = [str(row[0]) for row in (cur.fetchall() or [])]
    finally:
        if restored_role is not None:
            cur.execute(f'SET ROLE "{restored_role}"')
    return names


def _restore_session_user(cur: Any) -> str | None:
    """RESET ROLE when needed and return the role that must be restored."""
    cur.execute("SELECT current_user, session_user")
    row = cur.fetchone()
    current_user = str(row[0]) if row else ""
    session_user = str(row[1]) if row and len(row) > 1 else ""
    if current_user and session_user and current_user != session_user:
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", current_user):
            raise RuntimeError(f"refusing to restore unsafe role {current_user!r}")
        cur.execute("RESET ROLE")
        return current_user
    return None


def ensure_sqlens_fragment_catalog(
    cur: Any,
    principal: str,
    vector_table: str,
    *,
    enable_tracking: bool = False,
) -> dict[str, Any]:
    """Create the r43 FragReuse catalog as postgres and grant the SQL-native role.

    Frozen r43 ``vector.so`` issues ``CREATE TABLE IF NOT EXISTS`` /
    ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` from the current user on the
    first FragReuse use, and again after an abort resets the backend-local
    ready flag. The SQL-native principal is a non-owner without BYPASSRLS, so
    those SPI statements fail unless the catalog exists and the role can DML
    it. This does not grant BYPASSRLS or ownership of the reviews heap.
    """
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", principal):
        raise RuntimeError(f"refusing unsafe principal {principal!r}")
    restored_role = _restore_session_user(cur)
    try:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS public.pgvector_hnsw_fragment_store ("
            "heap_oid oid NOT NULL,"
            "filter_name text NOT NULL,"
            "kind text NOT NULL,"
            "rows bigint NOT NULL,"
            "pages bigint NOT NULL,"
            "bloom_bit_count bigint NOT NULL,"
            "payload bytea NOT NULL,"
            "format_version integer NOT NULL DEFAULT 3,"
            "built_at timestamptz NOT NULL DEFAULT pg_catalog.now(),"
            "build_epoch bigint NOT NULL DEFAULT 0,"
            "relfilenode oid NOT NULL DEFAULT 0,"
            "PRIMARY KEY (heap_oid, filter_name, kind))"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS public.pgvector_hnsw_fragment_epoch ("
            "heap_oid oid PRIMARY KEY,"
            "epoch bigint NOT NULL DEFAULT 0,"
            "updated_at timestamptz NOT NULL DEFAULT pg_catalog.now())"
        )
        cur.execute(
            "ALTER TABLE public.pgvector_hnsw_fragment_store "
            "ADD COLUMN IF NOT EXISTS build_epoch bigint NOT NULL DEFAULT 0, "
            "ADD COLUMN IF NOT EXISTS relfilenode oid NOT NULL DEFAULT 0, "
            "ADD COLUMN IF NOT EXISTS format_version integer NOT NULL DEFAULT 1"
        )
        cur.execute(
            f'GRANT SELECT, INSERT, UPDATE, DELETE '
            f'ON public.pgvector_hnsw_fragment_store TO "{principal}"'
        )
        cur.execute(
            f'GRANT SELECT, INSERT, UPDATE, DELETE '
            f'ON public.pgvector_hnsw_fragment_epoch TO "{principal}"'
        )
        # r43 SPI always issues CREATE TABLE IF NOT EXISTS as the current user.
        cur.execute(f'GRANT CREATE ON SCHEMA public TO "{principal}"')
        cur.execute(
            f'ALTER TABLE public.pgvector_hnsw_fragment_store OWNER TO "{principal}"'
        )
        cur.execute(
            f'ALTER TABLE public.pgvector_hnsw_fragment_epoch OWNER TO "{principal}"'
        )
        if enable_tracking:
            # SHARE ROW EXCLUSIVE on the reviews heap. Must run before the
            # formal data guard holds its long-lived transaction lock.
            cur.execute(
                "SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)",
                (vector_table,),
            )
    finally:
        if restored_role is not None:
            cur.execute(f'SET ROLE "{restored_role}"')
    return {
        "valid": True,
        "principal": principal,
        "vector_table": vector_table,
    }


def set_preferred_index(cur: Any, vector_index: str) -> str:
    """Bind hnsw.preferred_index from a privileged session, then restore ROLE.

    The assign hook requires index ownership. The SQL-native principal is a
    non-owner without BYPASSRLS, so this GUC must be set as session_user
    (postgres) before or after SET ROLE. Session-level set_config survives
    the subsequent SET ROLE used for query execution.
    """
    restored_role = _restore_session_user(cur)
    try:
        cur.execute(
            "SELECT set_config('hnsw.preferred_index', %s, false)",
            (vector_index,),
        )
    finally:
        if restored_role is not None:
            cur.execute(f'SET ROLE "{restored_role}"')
    cur.execute("SELECT current_setting('hnsw.preferred_index')")
    row = cur.fetchone()
    current = str(row[0]) if row else ""
    if current != vector_index:
        raise RuntimeError(
            f"preferred-index gate failed: requested={vector_index!r} current={current!r}"
        )
    return current


def set_mode(
    cur: Any,
    mode: str,
    config: Config,
    vector_index: str,
    d3_settings: dict[str, Any] | None = None,
    *,
    reset_cache: bool = True,
) -> dict[str, str]:
    if mode not in MODE_SPECS:
        raise ValueError(f"unknown benchmark mode: {mode}")
    spec = MODE_SPECS[mode]
    if mode == SQL_FIRST_MODE:
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET enable_indexscan = on")
        cur.execute("SET enable_bitmapscan = on")
        cur.execute("SET join_collapse_limit = 8")
        cur.execute("SET from_collapse_limit = 8")
        cur.execute("SET hnsw.page_access = off")
        cur.execute("SET hnsw.index_page_access = off")
        preferred = set_preferred_index(cur, vector_index)
        return {
            "filter_strategy": spec.filter_strategy,
            "preferred_index": preferred,
        }
    set_search_config(cur, config)
    configure_hnsw_driven_planner(cur)
    # D2 is the physical BFS clone only. Keep heap/index prefetch out of this
    # factorial comparison and prove the settings again immediately per query.
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    settings = d3_settings or {
        "probe_requests": DEFAULT_D3_PROBE_REQUESTS,
        "min_benefit_per_byte": DEFAULT_D3_MIN_BENEFIT_PER_BYTE,
        "max_fragment_mb": DEFAULT_D3_MAX_FRAGMENT_MB,
        "page_min_skip_rate": DEFAULT_D3_PAGE_MIN_SKIP_RATE,
    }
    # Prefer database/session default; only set when still privileged.
    cur.execute(
        "SELECT current_setting('is_superuser', true), "
        "current_setting('hnsw.guidance_require_epoch', true)"
    )
    row = cur.fetchone()
    is_superuser = str(row[0]).lower() if row else ""
    guidance_require_epoch = str(row[1]).lower() if row and len(row) > 1 else ""
    if guidance_require_epoch != "on":
        if is_superuser not in {"on", "true", "1"}:
            # Unit-test cursors may not implement current_setting; attempt SET
            # and let a real privilege error surface on production connections.
            try:
                cur.execute("SET hnsw.guidance_require_epoch = on")
            except Exception as exc:
                raise RuntimeError(
                    "hnsw.guidance_require_epoch must be on before SET ROLE; "
                    "set it as postgres or via ALTER DATABASE ... SET"
                ) from exc
        else:
            cur.execute("SET hnsw.guidance_require_epoch = on")
    cur.execute(f"SET hnsw.d3_probe_requests = {int(settings['probe_requests'])}")
    cur.execute(
        f"SET hnsw.d3_min_benefit_per_byte = {float(settings['min_benefit_per_byte'])}"
    )
    cur.execute(f"SET hnsw.d3_max_fragment_mb = {int(settings['max_fragment_mb'])}")
    cur.execute(
        f"SET hnsw.d3_page_min_skip_rate = {float(settings['page_min_skip_rate'])}"
    )
    cur.execute(f"SET hnsw.filter_strategy = {spec.filter_strategy}")
    preferred = set_preferred_index(cur, vector_index)
    if reset_cache:
        cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    return {
        "filter_strategy": spec.filter_strategy,
        "preferred_index": preferred,
    }


def set_search_config(cur: Any, config: Config) -> None:
    """Switch search budgets without resetting D3's trace-scoped cache."""
    cur.execute(f"SET hnsw.ef_search = {int(config.ef_search)}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(config.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(config.scan_mem_multiplier)}")
    cur.execute(f"SET hnsw.iterative_scan = {config.iterative_scan}")
    cur.execute(f"SET hnsw.guided_collect_target = {int(config.guided_collect_target)}")


def set_as_of(cur: Any, as_of: int) -> None:
    # SET does not accept a bind parameter; set_config keeps this value safely
    # parameterized and session-scoped for the RLS policy.
    cur.execute("SELECT set_config('app.as_of', %s, false)", (str(int(as_of)),))


def fetch_json_object(cur: Any, sql_text: str) -> dict[str, Any]:
    cur.execute(sql_text)
    row = cur.fetchone()
    value = row[0] if row else None
    try:
        parsed = json.loads(value) if isinstance(value, str) else dict(value or {})
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SQLens profile is not a JSON object: {value!r}") from exc
    return parsed


def _profile_counter(profile: dict[str, Any], name: str) -> int:
    try:
        return int(profile.get(name, 0) or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"SQLens profile counter {name!r} is invalid") from exc


def scan_profile_export(profile: dict[str, Any]) -> dict[str, Any]:
    return {field: profile.get(field, NA) for field in SQLENS_PROFILE_EXPORT_FIELDS}


def configure_guidance(
    cur: Any, mode: str, vector_index: str, atoms: Sequence[str]
) -> dict[str, Any]:
    if mode == SQL_FIRST_MODE:
        return {
            "sql_first_forced_indexed_exact": True,
            "guidance_enabled": False,
            "guidance_route": SQL_FIRST_MODE,
            "activation_atom_count": 0,
            "activation_ms": 0.0,
            "before": {},
            "after_activation": {},
        }
    mode_spec = MODE_SPECS[mode]
    before = fetch_json_object(cur, "SELECT vector_hnsw_guidance_profile()")
    started = time.perf_counter()
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    if mode_spec.guidance_kind is None:
        activation_ms = (time.perf_counter() - started) * 1000.0
        return {
            "guidance_enabled": False,
            "guidance_route": "stock",
            "activation_atom_count": 0,
            "before": before,
            "after_activation": fetch_json_object(
                cur, "SELECT vector_hnsw_guidance_profile()"
            ),
            "activation_ms": activation_ms,
        }
    cur.execute(f"SET hnsw.filter_strategy = {mode_spec.filter_strategy}")
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
        (vector_index, list(atoms), mode_spec.guidance_kind),
    )
    row = cur.fetchone()
    activated_atoms = int(row[0]) if row and row[0] is not None else 0
    activation_ms = (time.perf_counter() - started) * 1000.0
    after = fetch_json_object(cur, "SELECT vector_hnsw_guidance_profile()")
    enabled = bool(after.get("active")) and activated_atoms > 0
    if mode_spec.adaptive and not enabled:
        # Probe/rejection requests execute stock HNSW while the extension records
        # workload evidence. The mode is not reported as active on these requests.
        route_started = time.perf_counter()
        cur.execute("SET hnsw.filter_strategy = off")
        activation_ms += (time.perf_counter() - route_started) * 1000.0
    elif not mode_spec.adaptive and not enabled:
        raise RuntimeError(f"{mode} guidance activation did not become active")
    return {
        "guidance_enabled": enabled,
        "guidance_route": (
            f"d3_{after.get('adaptive_state', 'unknown')}" if mode_spec.adaptive
            else "safe_guided_candidate_validation"
        ),
        "activation_atom_count": activated_atoms,
        "before": before,
        "after_activation": after,
        "activation_ms": activation_ms,
    }


def adaptive_transition_for_request(
    activation: dict[str, Any], post_query: dict[str, Any]
) -> dict[str, Any]:
    before = dict(activation.get("before") or {})
    after = dict(activation.get("after_activation") or {})
    probe = _profile_counter(post_query, "adaptive_probes") > _profile_counter(
        before, "adaptive_probes"
    )
    admissions = _profile_counter(post_query, "adaptive_admissions") - _profile_counter(
        before, "adaptive_admissions"
    )
    builds = sum(
        _profile_counter(post_query, field) - _profile_counter(before, field)
        for field in ("adaptive_page_builds", "adaptive_bloom_builds")
    )
    fragment_store_hits = _profile_counter(
        post_query, "fragment_store_hits"
    ) - _profile_counter(before, "fragment_store_hits")
    active = bool(activation.get("guidance_enabled")) and bool(after.get("active"))
    return {
        "adaptive_state_before": str(before.get("adaptive_state", "missing")),
        "adaptive_state_after_activation": str(after.get("adaptive_state", "missing")),
        "adaptive_state_after_query": str(post_query.get("adaptive_state", "missing")),
        "adaptive_probe_observed": probe,
        "adaptive_admission_observed": admissions > 0,
        "adaptive_materialized": builds > 0,
        "adaptive_active": active,
        "hidden_prebuilt_fragment_reused": fragment_store_hits > 0,
        "fragment_store_hit_delta": fragment_store_hits,
        "adaptive_transition": (
            f"{before.get('adaptive_state', 'missing')}->"
            f"{after.get('adaptive_state', 'missing')}->"
            f"{post_query.get('adaptive_state', 'missing')}"
        ),
    }


def guidance_execution_proof(
    mode: str,
    activation: dict[str, Any],
    execution_profile: dict[str, Any],
    scan_profile: dict[str, Any],
    post_guidance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in MODE_SPECS:
        raise ValueError(f"unknown benchmark mode: {mode}")
    if mode == SQL_FIRST_MODE:
        valid = bool(
            activation.get("sql_first_forced_indexed_exact")
            and not execution_profile
            and not scan_profile
        )
        return {
            "valid": valid,
            "execution_profile_complete": True,
            "binding_attempted": False,
            "binding_matched": False,
            "effective_active": False,
            "statement_bound": False,
            "binding_attempts_delta": 0,
            "binding_matches_delta": 0,
            "binding_scan_checks_delta": 0,
            "binding_scan_matches_delta": 0,
            "binding_scan_bypasses_delta": 0,
            "guidance_checks": 0,
            "final_path": SQL_FIRST_MODE,
            "reported_active": False,
            "d3_probe_exception": False,
            "exception_reason": "",
        }
    required_execution_fields = {
        "active",
        "effective_active",
        "statement_bound",
        "binding_attempts",
        "binding_matches",
        "binding_scan_checks",
        "binding_scan_matches",
        "binding_scan_bypasses",
    }
    execution_profile_complete = required_execution_fields.issubset(execution_profile)
    before = dict(activation.get("before") or {})
    attempts_delta = _profile_counter(
        execution_profile, "binding_attempts"
    ) - _profile_counter(before, "binding_attempts")
    matches_delta = _profile_counter(
        execution_profile, "binding_matches"
    ) - _profile_counter(before, "binding_matches")
    scan_checks_delta = _profile_counter(
        execution_profile, "binding_scan_checks"
    ) - _profile_counter(before, "binding_scan_checks")
    scan_matches_delta = _profile_counter(
        execution_profile, "binding_scan_matches"
    ) - _profile_counter(before, "binding_scan_matches")
    scan_bypasses_delta = _profile_counter(
        execution_profile, "binding_scan_bypasses"
    ) - _profile_counter(before, "binding_scan_bypasses")
    guidance_checks = _profile_counter(scan_profile, "guidance_checks")
    final_path = str(scan_profile.get("final_path", "missing"))
    effective_active = bool(execution_profile.get("effective_active"))
    statement_bound = bool(execution_profile.get("statement_bound"))
    scan_valid = bool(scan_profile.get("valid"))

    if mode == "stock":
        valid = (
            execution_profile_complete
            and scan_valid
            and attempts_delta > 0
            and matches_delta == 0
            and not effective_active
            and guidance_checks == 0
            and final_path in {"stock", "stock_bypass"}
        )
        return {
            "valid": valid,
            "execution_profile_complete": execution_profile_complete,
            "binding_attempted": attempts_delta > 0,
            "binding_matched": matches_delta > 0,
            "effective_active": False,
            "statement_bound": statement_bound,
            "binding_attempts_delta": attempts_delta,
            "binding_matches_delta": matches_delta,
            "binding_scan_checks_delta": scan_checks_delta,
            "binding_scan_matches_delta": scan_matches_delta,
            "binding_scan_bypasses_delta": scan_bypasses_delta,
            "guidance_checks": guidance_checks,
            "final_path": final_path,
            "reported_active": False,
            "d3_probe_exception": False,
            "exception_reason": "",
        }

    # r43 increments adaptive_probes on the session profile after the scan,
    # not on the bind-time execution snapshot attached to the result row.
    probe_profile = (
        post_guidance
        if isinstance(post_guidance, dict) and post_guidance
        else execution_profile
    )
    d3_probe = (
        mode == "d1_d2_d3"
        and execution_profile_complete
        and activation.get("guidance_route") == "d3_probing"
        and not bool(activation.get("guidance_enabled"))
        and not effective_active
        and attempts_delta > 0
        and matches_delta == 0
        and str(execution_profile.get("adaptive_state", "")) == "probing"
        and _profile_counter(probe_profile, "adaptive_probes")
        > _profile_counter(before, "adaptive_probes")
        and scan_valid
        and guidance_checks == 0
        and final_path in {"stock", "stock_bypass"}
    )
    active_valid = (
        execution_profile_complete
        and bool(activation.get("guidance_enabled"))
        and bool(execution_profile.get("active"))
        and effective_active
        and statement_bound
        and attempts_delta > 0
        and matches_delta > 0
        and scan_checks_delta > 0
        and scan_matches_delta > 0
        and scan_bypasses_delta == 0
        and scan_valid
        and guidance_checks > 0
        and final_path == "validation_only"
    )
    return {
        "valid": bool(active_valid or d3_probe),
        "execution_profile_complete": execution_profile_complete,
        "binding_attempted": attempts_delta > 0,
        "binding_matched": matches_delta > 0,
        "effective_active": effective_active,
        "statement_bound": statement_bound,
        "binding_attempts_delta": attempts_delta,
        "binding_matches_delta": matches_delta,
        "binding_scan_checks_delta": scan_checks_delta,
        "binding_scan_matches_delta": scan_matches_delta,
        "binding_scan_bypasses_delta": scan_bypasses_delta,
        "guidance_checks": guidance_checks,
        "final_path": final_path,
        "reported_active": bool(active_valid),
        "d3_probe_exception": bool(d3_probe),
        "exception_reason": "workload_driven_probe_stock_route" if d3_probe else "",
    }


def recorded_guidance_proof_is_valid(row: dict[str, Any]) -> bool:
    activation = row.get("guidance_activation_profile")
    execution = row.get("execution_guidance_profile")
    scan = row.get("scan_profile")
    stored = row.get("guidance_execution_proof")
    if not all(isinstance(value, dict) for value in (activation, execution, scan, stored)):
        return False
    try:
        post_guidance = row.get("post_query_guidance_profile")
        recomputed = guidance_execution_proof(
            str(row.get("mode")),
            activation,
            execution,
            scan,
            post_guidance if isinstance(post_guidance, dict) else None,
        )
    except (RuntimeError, TypeError, ValueError):
        return False
    return bool(
        recomputed.get("valid") is True
        and stored == recomputed
        and row.get("guidance_binding_matched")
        is recomputed.get("binding_matched")
        and row.get("guidance_effective_active")
        is recomputed.get("effective_active")
        and row.get("guidance_checks") == recomputed.get("guidance_checks")
        and row.get("guidance_final_path") == recomputed.get("final_path")
        and (
            recomputed.get("d3_probe_exception") is not True
            or recomputed.get("reported_active") is False
        )
    )


def validate_fragment_store_reset(
    before: dict[str, Any], deleted_count: int, after: dict[str, Any]
) -> dict[str, Any]:
    before_count = int(before.get("count", -1))
    after_count = int(after.get("count", -1))
    if deleted_count != before_count or after_count != 0:
        raise RuntimeError(
            "persistent fragment store reset failed: "
            f"before={before_count} deleted={deleted_count} after={after_count}"
        )
    return {
        "valid": True,
        "before": before,
        "deleted_count": deleted_count,
        "after": after,
        "prebuilt_fragments": after_count,
    }


def audit_fragment_store(cur: Any, vector_table: str) -> dict[str, Any]:
    cur.execute("SELECT to_regclass('public.pgvector_hnsw_fragment_store')")
    row = cur.fetchone()
    if row is None or row[0] is None:
        records: list[str] = []
        return {
            "exists": False,
            "count": 0,
            "content_sha256": canonical_sha256(records),
        }
    cur.execute(
        """
        SELECT row_to_json(store_row)::text
        FROM public.pgvector_hnsw_fragment_store AS store_row
        WHERE store_row.heap_oid = to_regclass(%s)
        ORDER BY row_to_json(store_row)::text
        """,
        (vector_table,),
    )
    records = [str(value[0]) for value in cur.fetchall()]
    return {
        "exists": True,
        "count": len(records),
        "content_sha256": canonical_sha256(records),
    }


def clear_fragment_store(cur: Any, vector_table: str) -> dict[str, Any]:
    before = audit_fragment_store(cur, vector_table)
    deleted = 0
    if before["exists"]:
        cur.execute(
            "DELETE FROM public.pgvector_hnsw_fragment_store "
            "WHERE heap_oid = to_regclass(%s)",
            (vector_table,),
        )
        deleted = int(cur.rowcount)
    after = audit_fragment_store(cur, vector_table)
    return validate_fragment_store_reset(before, deleted, after)


def cache_profile_is_empty(profile: dict[str, Any]) -> bool:
    fields = (
        "entries",
        "resident_entries",
        "resident_bytes",
        "composed_guide_entries",
        "composed_exact_entries",
        "adaptive_cache_entries",
        "adaptive_bytes",
    )
    return all(_profile_counter(profile, field) == 0 for field in fields)


def reset_adaptive_state(
    cur: Any, persistent_reset: dict[str, Any] | None = None
) -> dict[str, Any]:
    before = fetch_json_object(cur, "SELECT vector_hnsw_metadata_cache_profile()")
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    after = fetch_json_object(cur, "SELECT vector_hnsw_metadata_cache_profile()")
    if not cache_profile_is_empty(after):
        raise RuntimeError("D3 cold-start gate failed: metadata cache is not empty after reset")
    prebuilt_fragments: Any = NA
    if persistent_reset is not None:
        if persistent_reset.get("valid") is not True:
            raise RuntimeError("D3 persistent fragment reset proof is invalid")
        prebuilt_fragments = int(persistent_reset.get("prebuilt_fragments", -1))
    return {
        "prebuilt_fragments": prebuilt_fragments,
        "before_reset": before,
        "after_reset": after,
        "after_reset_empty": True,
        "persistent_reset": persistent_reset or {"status": "not_supplied"},
    }


def plan_index_names(plan: Any) -> list[str]:
    names: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            value = node.get("Index Name")
            if value:
                names.append(str(value))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(plan)
    return names


def validate_explain_gate(plan: Any, vector_index: str, *, require_hnsw: bool) -> dict[str, Any]:
    expected_index = parse_qualified_name(vector_index)[-1].lower()
    index_names = plan_index_names(plan)
    hnsw_names = [name for name in index_names if "hnsw" in name.lower()]
    uses_expected = any(name.lower() == expected_index for name in index_names)
    vector_hnsw_names = list(hnsw_names)
    valid = (
        uses_expected
        and all(name.lower() == expected_index for name in vector_hnsw_names)
        if require_hnsw
        else not hnsw_names
    )
    if not valid:
        mode = "approximate HNSW" if require_hnsw else "exact non-HNSW"
        raise RuntimeError(
            f"EXPLAIN gate failed for {mode}: expected_index={expected_index!r} "
            f"index_names={index_names!r}"
        )
    return {
        "valid": True,
        "require_hnsw": require_hnsw,
        "expected_index": expected_index,
        "expected_index_qualified": vector_index,
        "index_names": index_names,
        "vector_hnsw_index_names": vector_hnsw_names,
    }


def collect_registered_scalar_indexes(cur: Any) -> list[str]:
    cur.execute(
        """
        SELECT index_rel.relname
        FROM pg_index AS idx
        JOIN pg_class AS index_rel ON index_rel.oid = idx.indexrelid
        JOIN pg_class AS heap_rel ON heap_rel.oid = idx.indrelid
        JOIN pg_namespace AS ns ON ns.oid = heap_rel.relnamespace
        JOIN pg_am AS am ON am.oid = index_rel.relam
        WHERE ns.nspname = 'public'
          AND heap_rel.relname = ANY(%s::text[])
          AND NOT idx.indisprimary
          AND am.amname IN ('btree', 'hash', 'brin', 'gin', 'gist')
        ORDER BY index_rel.relname
        """,
        (
            [
                "amazon_grocery_reviews_10m_pgvector",
                "amazon_review_facts",
                "amazon_product_dim",
                "amazon_principal_tenant_grants",
            ],
        ),
    )
    indexes = [str(row[0]) for row in cur.fetchall()]
    if not indexes:
        raise RuntimeError("no registered non-primary scalar indexes are available")
    return indexes


def validate_sql_first_explain_gate(
    plan: Any, registered_scalar_indexes: Sequence[str]
) -> dict[str, Any]:
    index_names = plan_index_names(plan)
    hnsw_names = [name for name in index_names if "hnsw" in name.lower()]
    allowed = {
        str(name).rsplit(".", 1)[-1].lower()
        for name in registered_scalar_indexes
    }
    matched = sorted(
        name for name in index_names if name.rsplit(".", 1)[-1].lower() in allowed
    )
    if hnsw_names or not matched:
        raise RuntimeError(
            "forced-indexed SQL-first EXPLAIN gate failed: "
            f"hnsw={hnsw_names!r} matched_scalar={matched!r} "
            f"registered={sorted(allowed)!r}"
        )
    return {
        "valid": True,
        "require_hnsw": False,
        "forced_indexed": True,
        "enable_seqscan": "off",
        "index_names": index_names,
        "registered_scalar_indexes": sorted(allowed),
        "matched_scalar_indexes": matched,
        "vector_hnsw_index_names": [],
    }


def validate_graph_compare(
    proof: Any, source_index: str, clone_index: str
) -> dict[str, Any]:
    try:
        normalized = json.loads(proof) if isinstance(proof, str) else dict(proof)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("same-graph BFS proof is not a JSON object") from exc
    required_true = (
        "same_heap",
        "logical_equal",
        "entry_equal",
        "tuple_coverage_equal",
        "definition_equal",
    )
    valid = (
        source_index != clone_index
        and all(normalized.get(field) is True for field in required_true)
        and normalized.get("physical_equal") is False
    )
    if not valid:
        raise RuntimeError(
            "same-heap/same-logical-graph BFS proof failed: "
            + json.dumps(
                {
                    "source_index": source_index,
                    "clone_index": clone_index,
                    "comparison": normalized,
                },
                sort_keys=True,
            )
        )
    return {
        "valid": True,
        "source_index": source_index,
        "clone_index": clone_index,
        "required": {
            "same_heap": True,
            "logical_equal": True,
            "entry_equal": True,
            "tuple_coverage_equal": True,
            "definition_equal": True,
            "physical_equal": False,
        },
        "comparison": normalized,
    }


# Amazon-10M HNSW canonical compare does not fit in the clone-server
# 8GB default; keep the raised limit on the guard session only.
D2_GRAPH_PROOF_MAINTENANCE_WORK_MEM = "64GB"


def graph_clone_proof(cur: Any, source_index: str, clone_index: str) -> dict[str, Any]:
    cur.execute(
        "SELECT set_config('maintenance_work_mem', %s, false)",
        (D2_GRAPH_PROOF_MAINTENANCE_WORK_MEM,),
    )
    cur.execute(
        "SELECT vector_hnsw_graph_compare(%s::regclass, %s::regclass)",
        (source_index, clone_index),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("same-graph BFS proof query returned no row")
    return validate_graph_compare(row[0], source_index, clone_index)


def runtime_sql_context(cur: Any, principal: str, as_of: int) -> dict[str, Any]:
    cur.execute(
        "SELECT current_user::text, current_setting('app.as_of', true), "
        "current_setting('hnsw.preferred_index', true), "
        "current_setting('hnsw.filter_strategy', true), "
        "current_setting('hnsw.page_access', true), "
        "current_setting('hnsw.index_page_access', true)"
    )
    row = cur.fetchone()
    context = {
        "current_user": str(row[0]) if row else "",
        "app_as_of": str(row[1]) if row and row[1] is not None else "",
        "preferred_index": str(row[2]) if row and row[2] is not None else "",
        "filter_strategy": str(row[3]) if row and row[3] is not None else "",
        "page_access": str(row[4]) if row and row[4] is not None else "",
        "index_page_access": str(row[5]) if row and row[5] is not None else "",
    }
    if (
        context["current_user"] != principal
        or context["app_as_of"] != str(int(as_of))
        or context["page_access"] != "off"
        or context["index_page_access"] != "off"
    ):
        raise RuntimeError(
            "principal/snapshot/prefetch gate failed: "
            f"expected=({principal!r},{int(as_of)!r}) observed={context!r}"
        )
    return context


def query_results(
    cur: Any, sql_text: str, params: dict[str, Any], *, exact: bool = False
) -> list[tuple[int, float]]:
    if exact:
        cur.execute("BEGIN")
        try:
            cur.execute("SET LOCAL enable_indexscan = on")
            cur.execute("SET LOCAL enable_bitmapscan = on")
            cur.execute("SET LOCAL enable_seqscan = on")
            cur.execute(sql_text, params)
            rows = [(int(row[0]), float(row[1])) for row in cur.fetchall()]
            cur.execute("COMMIT")
            return rows
        except Exception:
            cur.execute("ROLLBACK")
            raise
    cur.execute(sql_text, params)
    return [(int(row[0]), float(row[1])) for row in cur.fetchall()]


def query_rows(cur: Any, sql_text: str, params: dict[str, Any], *, exact: bool = False) -> list[int]:
    return [row_id for row_id, _ in query_results(cur, sql_text, params, exact=exact)]


def explain(
    cur: Any,
    sql_text: str,
    params: dict[str, Any],
    *,
    vector_index: str,
    require_hnsw: bool,
) -> tuple[Any, dict[str, Any]]:
    cur.execute("EXPLAIN (FORMAT JSON) " + sql_text, params)
    row = cur.fetchone()
    plan = row[0] if row else []
    return plan, validate_explain_gate(plan, vector_index, require_hnsw=require_hnsw)


def prepare_explain_without_runtime_state(cur: Any) -> dict[str, Any]:
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    guidance = fetch_json_object(cur, "SELECT vector_hnsw_guidance_profile()")
    cache = fetch_json_object(cur, "SELECT vector_hnsw_metadata_cache_profile()")
    if bool(guidance.get("active")) or not cache_profile_is_empty(cache):
        raise RuntimeError("EXPLAIN pre-state is not cold and inactive")
    return {"guidance": guidance, "cache": cache}


def finish_explain_without_runtime_state(
    cur: Any, before: dict[str, Any]
) -> dict[str, Any]:
    guidance = fetch_json_object(cur, "SELECT vector_hnsw_guidance_profile()")
    cache = fetch_json_object(cur, "SELECT vector_hnsw_metadata_cache_profile()")
    counter_fields = (
        "binding_attempts",
        "binding_matches",
        "binding_scan_checks",
        "adaptive_probes",
        "adaptive_admissions",
        "fragment_builds",
    )
    unchanged = all(
        _profile_counter(guidance, field)
        == _profile_counter(before["guidance"], field)
        for field in counter_fields
    )
    valid = (
        not bool(guidance.get("active"))
        and cache_profile_is_empty(cache)
        and unchanged
    )
    if not valid:
        raise RuntimeError("EXPLAIN mutated measured guidance/adaptive state")
    return {
        "valid": True,
        "execution": "EXPLAIN_without_ANALYZE",
        "guidance_before": before["guidance"],
        "guidance_after": guidance,
        "cache_before": before["cache"],
        "cache_after": cache,
        "counters_unchanged": True,
    }


relation_fingerprint = exact_truth_contract.relation_fingerprint
validate_rls_security_proof = exact_truth_contract.validate_rls_security_proof


def validate_sqlens_provenance(build_id: Any, profile: Any) -> tuple[str, dict[str, Any]]:
    normalized_build_id = str(build_id or "")
    if normalized_build_id != SQLENS_R43_BUILD_ID:
        raise RuntimeError(
            "SQLens build gate failed: "
            f"vector_sqlens_build_id() returned {normalized_build_id!r}; "
            f"expected exact r43 build {SQLENS_R43_BUILD_ID!r}. "
            "Rebuild/reload vector.so and reconnect."
        )
    try:
        normalized_profile = json.loads(profile) if isinstance(profile, str) else dict(profile)
        semantics = float(normalized_profile["profile_semantics_version"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "SQLens build gate failed: vector_hnsw_last_scan_profile() has no valid "
            "profile_semantics_version. Rebuild/reload vector.so and reconnect."
        ) from exc
    missing = [field for field in SQLENS_PROFILE_FIELDS if field not in normalized_profile]
    if not math.isfinite(semantics) or semantics < SQLENS_PROFILE_SEMANTICS or missing:
        raise RuntimeError(
            "SQLens build gate failed: incompatible scan profile "
            f"semantics={normalized_profile.get('profile_semantics_version')!r} "
            f"minimum={SQLENS_PROFILE_SEMANTICS:g} "
            f"missing={missing!r}. Rebuild/reload vector.so and reconnect."
        )
    return normalized_build_id, normalized_profile


def observe_serving_binary_identity(
    cur: Any,
    expected_sqlens_build_id: str,
    expected_vector_so_sha256: str,
    *,
    stage: str,
    connection: str,
) -> dict[str, Any]:
    try:
        cur.execute(
            "WITH lib AS ("
            "SELECT setting || '/vector.so' AS path "
            "FROM pg_config WHERE name = 'PKGLIBDIR'"
            ") SELECT vector_sqlens_build_id(), path, "
            "encode(sha256(pg_read_binary_file(path)), 'hex'), "
            "pg_backend_pid(), current_database() FROM lib"
        )
        row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 - this formal gate must fail closed.
        raise RuntimeError(
            f"binary identity gate unavailable at stage={stage} connection={connection}: "
            "could not read the serving PostgreSQL vector.so"
        ) from exc
    if row is None:
        raise RuntimeError(
            f"binary identity gate returned no row at stage={stage} connection={connection}"
        )
    observed_build_id = str(row[0] or "")
    observed_path = str(row[1] or "")
    observed_sha256 = str(row[2] or "")
    evidence = {
        "sequence": 0,
        "stage": stage,
        "connection": connection,
        "backend_pid": int(row[3]),
        "database": str(row[4] or ""),
        "expected_sqlens_build_id": expected_sqlens_build_id,
        "observed_sqlens_build_id": observed_build_id,
        "expected_vector_so_sha256": expected_vector_so_sha256,
        "observed_vector_so_sha256": observed_sha256,
        "observed_vector_so_path": observed_path,
        "build_id_exact_match": observed_build_id == expected_sqlens_build_id,
        "vector_so_sha256_exact_match": (
            observed_sha256 == expected_vector_so_sha256
        ),
        "path_valid": observed_path.endswith("/vector.so"),
        "observed_sha256_valid": bool(
            re.fullmatch(r"[0-9a-f]{64}", observed_sha256)
        ),
    }
    evidence["exact_match"] = all(
        evidence[field]
        for field in (
            "build_id_exact_match",
            "vector_so_sha256_exact_match",
            "path_valid",
            "observed_sha256_valid",
        )
    )
    return evidence


def require_exact_binary_identity(evidence: dict[str, Any]) -> dict[str, Any]:
    if evidence.get("exact_match") is not True:
        raise RuntimeError(
            "binary identity gate failed: "
            f"stage={evidence.get('stage')} connection={evidence.get('connection')} "
            f"expected_build_id={evidence.get('expected_sqlens_build_id')!r} "
            f"observed_build_id={evidence.get('observed_sqlens_build_id')!r} "
            f"expected_vector_so_sha256={evidence.get('expected_vector_so_sha256')!r} "
            f"observed_vector_so_sha256={evidence.get('observed_vector_so_sha256')!r} "
            f"observed_vector_so_path={evidence.get('observed_vector_so_path')!r}"
        )
    return evidence


def verify_serving_binary_identity(
    cur: Any,
    expected_sqlens_build_id: str,
    expected_vector_so_sha256: str,
    *,
    stage: str,
    connection: str,
) -> dict[str, Any]:
    return require_exact_binary_identity(
        observe_serving_binary_identity(
            cur,
            expected_sqlens_build_id,
            expected_vector_so_sha256,
            stage=stage,
            connection=connection,
        )
    )


def binary_identity_gate_summary(
    expected_sqlens_build_id: str,
    expected_vector_so_sha256: str,
    evidence: Sequence[dict[str, Any]],
    required_mode_connections: Sequence[str] = MODES,
) -> dict[str, Any]:
    gate_evidence = [dict(item) for item in evidence]
    observed_build_ids = sorted(
        {str(item.get("observed_sqlens_build_id", "")) for item in gate_evidence}
    )
    observed_sha256s = sorted(
        {str(item.get("observed_vector_so_sha256", "")) for item in gate_evidence}
    )
    observed_paths = sorted(
        {str(item.get("observed_vector_so_path", "")) for item in gate_evidence}
    )
    observed_stages = sorted({str(item.get("stage", "")) for item in gate_evidence})
    observed_connections = sorted(
        {str(item.get("connection", "")) for item in gate_evidence}
    )
    missing_stages = [
        stage for stage in BINARY_IDENTITY_REQUIRED_STAGES if stage not in observed_stages
    ]
    required_connections = (
        "fragment_store",
        "data_guard",
        *required_mode_connections,
    )
    missing_connections = [
        connection
        for connection in required_connections
        if connection not in observed_connections
    ]
    all_exact_match = bool(gate_evidence) and all(
        item.get("exact_match") is True
        and item.get("expected_sqlens_build_id") == expected_sqlens_build_id
        and item.get("observed_sqlens_build_id") == expected_sqlens_build_id
        and item.get("expected_vector_so_sha256") == expected_vector_so_sha256
        and item.get("observed_vector_so_sha256") == expected_vector_so_sha256
        and str(item.get("observed_vector_so_path", "")).endswith("/vector.so")
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(item.get("observed_vector_so_sha256", "")),
            )
        )
        for item in gate_evidence
    )
    return {
        "required_for_formal_execute": True,
        "expected_sqlens_build_id": expected_sqlens_build_id,
        "expected_vector_so_sha256": expected_vector_so_sha256,
        "observed_sqlens_build_id": (
            observed_build_ids[0] if len(observed_build_ids) == 1 else None
        ),
        "observed_vector_so_sha256": (
            observed_sha256s[0] if len(observed_sha256s) == 1 else None
        ),
        "observed_vector_so_paths": observed_paths,
        "observed_sqlens_build_ids": observed_build_ids,
        "observed_vector_so_sha256s": observed_sha256s,
        "observed_stages": observed_stages,
        "observed_connections": observed_connections,
        "missing_required_stages": missing_stages,
        "missing_required_connections": missing_connections,
        "evidence_count": len(gate_evidence),
        "all_exact_match": all_exact_match,
        "valid": all_exact_match and not missing_stages and not missing_connections,
        "gate_evidence": gate_evidence,
    }


def database_fingerprint(cur: Any, relations: Sequence[str]) -> dict[str, Any]:
    cur.execute(
        "SELECT current_database(), current_setting('server_version'), "
        "coalesce((SELECT extversion FROM pg_extension WHERE extname = 'vector'), ''), "
        "vector_sqlens_build_id()"
    )
    database, postgres, vector_version, build_id = cur.fetchone()
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    profile_row = cur.fetchone()
    build_id, scan_profile = validate_sqlens_provenance(
        build_id, profile_row[0] if profile_row else None
    )
    relation_data = {relation: relation_fingerprint(cur, relation) for relation in dict.fromkeys(relations)}
    fact_relation = relation_data.get("public.amazon_review_facts")
    if fact_relation is None or not fact_relation["rls"]:
        raise RuntimeError("artifact invalid: amazon_review_facts must have RLS enabled")
    return {
        "database": database,
        "postgres_version": postgres,
        "vector_extension_version": vector_version,
        "sqlens_build_id": build_id,
        "profile_semantics_version": scan_profile["profile_semantics_version"],
        "required_profile_fields": list(SQLENS_PROFILE_FIELDS),
        "loaded_profile": scan_profile,
        "relations": relation_data,
    }


def loaded_session_context(cur: Any) -> dict[str, str]:
    cur.execute(
        "SELECT current_user::text, session_user::text, "
        "txid_current_snapshot()::text, current_database()::text, "
        "current_setting('hnsw.preferred_index', true), "
        "current_setting('hnsw.filter_strategy', true), "
        "current_setting('hnsw.guidance_require_epoch', true)"
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("could not capture loaded role/snapshot context")
    return {
        "current_user": str(row[0]),
        "session_user": str(row[1]),
        "transaction_snapshot": str(row[2]),
        "database": str(row[3]),
        "preferred_index_current_setting": str(row[4] or ""),
        "filter_strategy_current_setting": str(row[5] or ""),
        "guidance_require_epoch_current_setting": str(row[6] or ""),
    }


def expected_keys_for(
    workloads: Sequence[WorkloadSpec], filters: Sequence[FilterSpec], query_ids: dict[int, int], repeats: int
) -> set[tuple[str, str, int, int]]:
    return {
        (workload.name, spec.name, query_no, repeat)
        for workload in workloads
        for spec in filters
        for query_no in query_ids
        for repeat in range(repeats)
    }


def sql_contract_hashes(
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    table: str,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[str, dict[str, dict[str, str]]]:
    validity_hash = candidate_universe_predicate_sha256(
        candidate_validity_predicate
    )
    result: dict[str, dict[str, dict[str, str]]] = {}
    for workload in workloads:
        result[workload.name] = {}
        for spec in filters:
            exact_sql = build_hybrid_sql(
                table,
                spec.predicate,
                workload=workload,
                exact=True,
                candidate_validity_predicate=candidate_validity_predicate,
            )
            validate_exact_sql_text(exact_sql)
            approx_sql = build_hybrid_sql(
                table,
                spec.predicate,
                workload=workload,
                candidate_validity_predicate=candidate_validity_predicate,
            )
            result[workload.name][spec.name] = {
                "exact_sha256": hashlib.sha256(exact_sql.encode()).hexdigest(),
                "approx_sha256": hashlib.sha256(approx_sql.encode()).hexdigest(),
                "workload_scalar_predicate": spec.predicate,
                "workload_scalar_predicate_sha256": workload_scalar_predicate_sha256(
                    spec.predicate
                ),
                "candidate_universe_predicate": candidate_validity_predicate,
                "candidate_universe_predicate_sha256": validity_hash,
            }
    return result


def _artifact_plan_is_non_hnsw(gate: Any, label: str) -> None:
    if not isinstance(gate, dict) or gate.get("valid") is not True:
        raise _artifact_error(f"{label} is missing a successful non-HNSW EXPLAIN gate")
    names = gate.get("index_names")
    if not isinstance(names, list) or any("hnsw" in str(name).lower() for name in names):
        raise _artifact_error(f"{label} EXPLAIN gate used HNSW or has invalid index provenance")


def _artifact_filters_match(source: Any, filters: Sequence[FilterSpec]) -> bool:
    if not isinstance(source, list) or len(source) != len(filters):
        return False
    try:
        return all(
            isinstance(row, dict)
            and row.get("name") == spec.name
            and row.get("target_rate") == spec.target_rate
            and row.get("predicate") == spec.predicate
            and int(row.get("expected_rows", -1)) == spec.expected_rows
            and float(row.get("actual_pct", math.nan)) == spec.actual_pct
            for row, spec in zip(source, filters)
        )
    except (TypeError, ValueError):
        return False


def _artifact_workloads_match(source: Any, workloads: Sequence[WorkloadSpec]) -> bool:
    if not isinstance(source, list) or len(source) != len(workloads):
        return False
    try:
        for row, workload in zip(source, workloads):
            if not isinstance(row, dict):
                return False
            temporal = row.get("temporal")
            if temporal is None:
                temporal = row.get("temporal_kind") != "none"
            if (
                row.get("name") != workload.name
                or float(row.get("bucket_pct", math.nan)) != workload.bucket_pct
                or bool(temporal) != workload.temporal
                or str(row.get("width", "base")) != workload.width
                or str(row.get("boolean_predicate", ""))
                != workload.boolean_predicate
            ):
                return False
    except (TypeError, ValueError):
        return False
    return True


def _artifact_query_ids(source: Any) -> dict[int, int]:
    if not isinstance(source, dict):
        raise _artifact_error("run-spec query_ids is malformed")
    try:
        parsed = {int(query_no): int(query_id) for query_no, query_id in source.items()}
    except (TypeError, ValueError) as exc:
        raise _artifact_error("run-spec query_ids is malformed") from exc
    if len(parsed) != len(source) or len(set(parsed.values())) != len(parsed):
        raise _artifact_error("run-spec query IDs are duplicate or malformed")
    return parsed


def _artifact_query_splits(source: Any) -> dict[int, str]:
    if not isinstance(source, dict):
        raise _artifact_error("run-spec query split contract is malformed")
    try:
        parsed = {int(query_no): str(split) for query_no, split in source.items()}
    except (TypeError, ValueError) as exc:
        raise _artifact_error("run-spec query split contract is malformed") from exc
    if len(parsed) != len(source):
        raise _artifact_error("run-spec query splits are duplicate or malformed")
    return parsed


def _run_spec_for_integrity_hash(run_spec: dict[str, Any]) -> dict[str, Any]:
    """Restore producer key types before hashing a JSON-loaded run_spec.

    json.dumps stringifies integer object keys. sort_keys then orders them
    lexicographically ("10" before "2"), so a loaded spec would not match the
    hash computed from the in-memory int-key dict used by the GT producer.
    """
    normalized = dict(run_spec)
    if "query_ids" in normalized:
        normalized["query_ids"] = _artifact_query_ids(normalized.get("query_ids"))
    if "query_splits" in normalized:
        normalized["query_splits"] = _artifact_query_splits(
            normalized.get("query_splits")
        )
    return normalized


def _artifact_pair_map(
    manifest: dict[str, Any],
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    table: str,
    candidate_validity_predicate: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    expected = {(workload.name, spec.name) for workload in workloads for spec in filters}
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        raise _artifact_error("manifest pairs are missing")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise _artifact_error("manifest pair is malformed")
        filter_data = pair.get("filter")
        key = (str(pair.get("workload")), str(filter_data.get("name") if isinstance(filter_data, dict) else ""))
        if key not in expected or key in result:
            raise _artifact_error(f"manifest has unexpected/duplicate pair: {key}")
        spec = next(item for item in filters if item.name == key[1])
        exact_workload = next(
            item for item in exact_truth_contract.WORKLOADS if item.name == key[0]
        )
        validity_hash = candidate_universe_predicate_sha256(
            candidate_validity_predicate
        )
        scalar_hash = workload_scalar_predicate_sha256(spec.predicate)
        if (
            pair.get("workload_scalar_predicate") != spec.predicate
            or pair.get("workload_scalar_predicate_sha256") != scalar_hash
            or pair.get("candidate_universe_predicate")
            != candidate_validity_predicate
            or pair.get("candidate_universe_predicate_sha256") != validity_hash
        ):
            raise _artifact_error(f"pair scalar/candidate-universe provenance is stale: {key}")
        candidate = pair.get("candidate")
        if not isinstance(candidate, dict):
            raise _artifact_error(f"candidate provenance is missing: {key}")
        candidate_path = Path(str(candidate.get("path", "")))
        candidate_hash = _require_sha256(candidate.get("sha256"), f"candidate {key}")
        if (
            int(candidate.get("count", 0)) <= 0
            or int(candidate.get("min_id", -1)) < 0
            or int(candidate.get("max_id", -1)) < int(candidate.get("min_id", -1))
            or not candidate_path.is_file()
            or sha256_file(candidate_path) != candidate_hash
        ):
            raise _artifact_error(f"candidate provenance is stale or incomplete: {key}")
        candidate_sql = pair.get("candidate_sql")
        if not isinstance(candidate_sql, str) or pair.get("candidate_sql_sha256") != _sha256_text(candidate_sql):
            raise _artifact_error(f"candidate SQL provenance is stale: {key}")
        expected_candidate_sql = exact_truth_contract.build_candidate_sql(
            table, spec.predicate, exact_workload, candidate_validity_predicate
        )
        if candidate_sql != expected_candidate_sql:
            raise _artifact_error(f"candidate SQL scalar/validity contract is stale: {key}")
        validate_exact_sql_text(candidate_sql)
        normalized_candidate = " ".join(candidate_sql.lower().split())
        required_candidate_tokens = exact_truth_contract.required_join_tokens(
            exact_workload
        ) + ("order by v.id",)
        if " limit " in f" {normalized_candidate} " or any(token not in normalized_candidate for token in required_candidate_tokens):
            raise _artifact_error(f"candidate SQL is not the required unbounded relational export: {key}")
        _artifact_plan_is_non_hnsw(pair.get("candidate_explain_gate"), f"candidate {key}")
        spot_sql = pair.get("spot_check_sql")
        if not isinstance(spot_sql, str) or pair.get("spot_check_sql_sha256") != _sha256_text(spot_sql):
            raise _artifact_error(f"spot-check SQL provenance is stale: {key}")
        expected_spot_sql = exact_truth_contract.build_spot_check_sql(
            table, spec.predicate, exact_workload, candidate_validity_predicate
        )
        if spot_sql != expected_spot_sql:
            raise _artifact_error(f"spot-check SQL scalar/validity contract is stale: {key}")
        validate_exact_sql_text(spot_sql)
        _artifact_plan_is_non_hnsw(pair.get("spot_check_explain_gate"), f"spot check {key}")
        checks = pair.get("spot_checks")
        if not isinstance(checks, list) or not checks:
            raise _artifact_error(f"spot checks are missing: {key}")
        seen_checks: set[int] = set()
        for check in checks:
            if not isinstance(check, dict) or check.get("valid") is not True:
                raise _artifact_error(f"spot check is invalid: {key}")
            query_no = int(check.get("query_no", -1))
            sql_ids = check.get("sql_ids")
            sql_distances = check.get("sql_distances")
            if (
                query_no in seen_checks
                or query_no < 0
                or int(check.get("limit", -1)) <= 0
                or not isinstance(sql_ids, list)
                or not isinstance(sql_distances, list)
                or len(sql_ids) != int(check["limit"])
                or len(sql_distances) != int(check["limit"])
                or len({int(value) for value in sql_ids}) != len(sql_ids)
                or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in sql_distances)
            ):
                raise _artifact_error(f"spot check is malformed: {key}")
            seen_checks.add(query_no)
        result[key] = pair
    if set(result) != expected:
        raise _artifact_error("manifest pair keyspace is incomplete")
    return result


def _validate_artifact_spot_check(
    check: dict[str, Any],
    query_id: int,
    expected_ids: Sequence[int],
    expected_distances_sq: Sequence[float],
) -> None:
    observed_ids = [int(value) for value in check["sql_ids"]]
    observed_distances = [float(value) for value in check["sql_distances"]]
    if query_id in observed_ids:
        raise _artifact_error("spot check includes its query row")
    for position, (expected_id, expected_distance, observed_id, observed_distance) in enumerate(
        zip(expected_ids, expected_distances_sq, observed_ids, observed_distances)
    ):
        tolerance = distance_tolerance(expected_distance)
        tied = (
            position > 0 and abs(expected_distance - expected_distances_sq[position - 1]) <= tolerance
        ) or (
            position + 1 < len(expected_distances_sq)
            and abs(expected_distance - expected_distances_sq[position + 1]) <= tolerance
        )
        if tied:
            if abs(observed_distance - expected_distance) > tolerance:
                raise _artifact_error("spot check tied result has the wrong distance")
        elif (
            observed_id != expected_id
            or abs(observed_distance - expected_distance) > max(1e-7, abs(expected_distance) * 5e-5)
        ):
            raise _artifact_error("spot check result does not match the truth record")


def load_external_exact_truth(
    truth_csv: Path,
    manifest_path: Path,
    fbin: Path,
    filters_csv: Path,
    query_ids_csv: Path,
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    query_ids: dict[int, int],
    query_splits: dict[int, str],
    as_of_by_workload: dict[str, int],
    table: str,
    principal: str,
    k: int,
    database_relations: dict[str, Any],
    *,
    require_formal_keyspace: bool = True,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
    query_cohort_manifest: Path | None = None,
) -> tuple[dict[tuple[str, str, int], ExactTruth], dict[str, str]]:
    """Load a producer artifact only after its immutable provenance is fully verified."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _artifact_error(f"manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("artifact_valid") is not True:
        raise _artifact_error("manifest does not declare artifact_valid=true")
    if manifest.get("artifact") != "amazon10m_sql_native_exact_truth":
        raise _artifact_error("manifest has an incompatible artifact type")
    if manifest.get("version") not in EXACT_TRUTH_COMPATIBLE_VERSIONS:
        raise _artifact_error(
            "legacy/incompatible exact-truth artifact version rejected: "
            f"observed={manifest.get('version')!r} "
            f"expected_one_of={EXACT_TRUTH_COMPATIBLE_VERSIONS}"
        )
    candidate_validity_predicate = validate_candidate_validity_predicate(
        candidate_validity_predicate
    )
    expected_validity_hash = candidate_universe_predicate_sha256(
        candidate_validity_predicate
    )
    requested_cohort_hash = query_cohort_sha256(query_ids, query_splits)
    data_version = manifest.get("data_version_proof")
    expected_data_relations = {
        relation: database_relations.get(relation)
        for relation in exact_truth_contract.formal_data_relations(table)
    }
    if (
        not isinstance(data_version, dict)
        or data_version.get("valid") is not True
        or data_version.get("start_hash") != data_version.get("end_hash")
        or data_version.get("start_relations") != expected_data_relations
        or data_version.get("end_relations") != expected_data_relations
        or data_version.get("start_hash") != canonical_sha256(expected_data_relations)
    ):
        raise _artifact_error("GT data-version/epoch proof does not match the formal run")
    if any(
        not isinstance(fingerprint, dict)
        or not any(
            exact_truth_contract.valid_epoch_trigger(trigger)
            for trigger in fingerprint.get("triggers", [])
        )
        for fingerprint in expected_data_relations.values()
    ):
        raise _artifact_error("formal data-version epoch trigger proof is incomplete")
    try:
        expected_relation_epoch = relation_epoch_contract(expected_data_relations)
    except RuntimeError as exc:
        raise _artifact_error("current formal relation epoch contract is incomplete") from exc
    if manifest.get("relation_epoch") != expected_relation_epoch:
        raise _artifact_error("GT relation epoch contract does not match the formal run")
    try:
        gt_security = exact_truth_contract.validate_rls_security_proof(
            dict(manifest.get("rls_security_proof") or {}), principal
        )
    except (TypeError, RuntimeError) as exc:
        raise _artifact_error("GT RLS principal/policy/probe proof is invalid") from exc
    expected_policy_hash = canonical_sha256(
        database_relations.get("public.amazon_review_facts", {}).get("policies", [])
    )
    if gt_security.get("policy_hash") != expected_policy_hash:
        raise _artifact_error("GT RLS policy hash does not match the formal run")
    run_spec = manifest.get("run_spec")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(run_spec, dict) or not isinstance(source_hashes, dict):
        raise _artifact_error("manifest run-spec/source hashes are missing")
    if len(query_ids) > 200 and (
        manifest.get("version") != EXACT_TRUTH_ARTIFACT_VERSION
        or run_spec.get("protocol") != P0_PROTOCOL
        or manifest.get("paper_eligible") is not True
        or not isinstance(manifest.get("requested_slice_completion"), dict)
        or manifest["requested_slice_completion"].get("complete") is not True
    ):
        raise _artifact_error(
            "q10200 exact truth requires the current protocol/version contract"
        )
    artifact_query_ids = _artifact_query_ids(run_spec.get("query_ids"))
    try:
        artifact_query_splits = {
            int(query_no): str(split)
            for query_no, split in run_spec.get("query_splits", {}).items()
        }
    except (TypeError, ValueError) as exc:
        raise _artifact_error("run-spec query split contract is malformed") from exc
    if (
        any(artifact_query_ids.get(query_no) != query_id for query_no, query_id in query_ids.items())
        or any(artifact_query_splits.get(query_no) != split for query_no, split in query_splits.items())
    ):
        raise _artifact_error(
            "query cohort mismatch: requested slice is not a subset of the exact-truth cohort"
        )
    expected_cohort_hash = query_cohort_sha256(
        artifact_query_ids, artifact_query_splits
    )
    if (
        manifest.get("run_spec_hash") != canonical_sha256(_run_spec_for_integrity_hash(run_spec))
        or run_spec.get("source_hashes") != source_hashes
    ):
        raise _artifact_error("manifest run-spec/source hash mismatch")
    candidate_universe = run_spec.get("candidate_universe")
    cohort = run_spec.get("query_cohort")
    try:
        run_splits = artifact_query_splits
    except (TypeError, ValueError) as exc:
        raise _artifact_error("run-spec query split contract is malformed") from exc
    if (
        not isinstance(candidate_universe, dict)
        or candidate_universe.get("predicate") != candidate_validity_predicate
        or candidate_universe.get("predicate_sha256") != expected_validity_hash
        or manifest.get("candidate_universe") != candidate_universe
        or manifest.get("candidate_universe_predicate_sha256")
        != expected_validity_hash
        or not isinstance(cohort, dict)
        or cohort.get("query_cohort_sha256") != expected_cohort_hash
        or cohort.get("source_csv_sha256") != source_hashes.get("query_ids_csv")
        or not isinstance(cohort.get("source_manifest"), dict)
        or cohort["source_manifest"].get("sha256")
        != source_hashes.get("query_cohort_manifest")
        or run_spec.get("query_cohort_sha256") != expected_cohort_hash
        or manifest.get("query_cohort") != cohort
        or manifest.get("query_cohort_sha256") != expected_cohort_hash
        or run_splits != artifact_query_splits
    ):
        raise _artifact_error("manifest query cohort/candidate universe contract mismatch")
    for name in (
        "script", "filters_csv", "query_ids_csv", "query_cohort_manifest", "fbin"
    ):
        _require_sha256(source_hashes.get(name), f"source {name}")
    if query_cohort_manifest is None:
        raise _artifact_error("v4 artifact loading requires the query cohort provenance manifest")
    local_sources = {
        "filters_csv": sha256_file(filters_csv),
        "query_ids_csv": sha256_file(query_ids_csv),
        "query_cohort_manifest": sha256_file(query_cohort_manifest),
        "fbin": sha256_file(fbin),
    }
    if any(source_hashes[name] != digest for name, digest in local_sources.items()):
        raise _artifact_error("source hashes do not match the current fbin/query/filter inputs")
    fbin_info = manifest.get("fbin")
    if not isinstance(fbin_info, dict) or Path(str(fbin_info.get("path", ""))).resolve() != fbin.resolve():
        raise _artifact_error("manifest fbin path is incompatible")
    # The producer records the full GT cohort split counts. The benchmark may
    # request a reserved subset (q0..q19 and q100..q199 are held out), so the
    # run_spec counts are validated against the artifact's own splits; the
    # requested queries are separately proven to be a subset above.
    gt_calibration_queries = sum(
        split == "calibration" for split in artifact_query_splits.values()
    )
    gt_final_queries = sum(
        split == "final" for split in artifact_query_splits.values()
    )
    requested_calibration_queries = sum(
        split == "calibration" for split in query_splits.values()
    )
    requested_final_queries = sum(
        split == "final" for split in query_splits.values()
    )
    try:
        compatible_run = (
            run_spec.get("vector_table") == table
            and run_spec.get("principal") == principal
            and int(run_spec.get("k", -1)) == k
            and int(run_spec.get("calibration_queries", -1)) == gt_calibration_queries
            and int(run_spec.get("final_queries", -1)) == gt_final_queries
            and requested_calibration_queries <= gt_calibration_queries
            and requested_final_queries <= gt_final_queries
        )
    except (TypeError, ValueError):
        compatible_run = False
    if not compatible_run:
        raise _artifact_error("table/principal/k/calibration/final compatibility check failed")
    backend = manifest.get("backend")
    run_backend = run_spec.get("backend")
    if (
        not isinstance(backend, dict)
        or backend != run_backend
        or backend.get("backend") != "faiss"
        or backend.get("class") != "IndexFlatL2"
        or backend.get("exact") is not True
        or backend.get("formal_default") is not True
        or int(backend.get("threads", 0)) <= 0
    ):
        raise _artifact_error("formal exact backend provenance is incompatible")
    mapping = manifest.get("base_table_mapping")
    if not isinstance(mapping, dict) or mapping != run_spec.get("base_table_mapping"):
        raise _artifact_error("base-table/fbin mapping provenance is missing or stale")
    try:
        base_sample_ids = [int(value) for value in mapping["base_sample_ids"]]
        checked_ids = [int(value) for value in mapping["checked_ids"]]
        included_query_ids = [int(value) for value in mapping["query_ids_included"]]
        mapping_valid = (
            base_sample_ids == sorted(set(base_sample_ids))
            and checked_ids == sorted(set(checked_ids))
            and included_query_ids == sorted(artifact_query_ids.values())
            and set(included_query_ids) <= set(checked_ids)
            and int(mapping["checked_rows"]) == len(checked_ids)
            and int(mapping["base_sample_size_requested"]) > 0
            and mapping["base_sample_ids_sha256"] == canonical_sha256(base_sample_ids)
            and mapping["checked_ids_sha256"] == canonical_sha256(checked_ids)
            and mapping["comparison"] == "float32_allclose"
            and math.isfinite(float(mapping["max_abs_error"]))
            and float(mapping["max_abs_error"]) >= 0.0
        )
    except (KeyError, TypeError, ValueError):
        mapping_valid = False
    if not mapping_valid:
        raise _artifact_error("base-table/fbin mapping audit is malformed")
    if not _artifact_filters_match(run_spec.get("filters"), filters) or not _artifact_workloads_match(run_spec.get("workloads"), workloads):
        raise _artifact_error("filter/workload compatibility check failed")
    expected_rows = len(workloads) * len(filters) * len(artifact_query_ids)
    if require_formal_keyspace and expected_rows != 3 * 14 * 200:
        raise _artifact_error("formal execution requires the complete 3*14*q200 keyspace")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise _artifact_error("manifest outputs are missing")
    expected_truth_hash = _require_sha256(outputs.get("truth_csv_sha256"), "truth CSV")
    if not truth_csv.is_file() or sha256_file(truth_csv) != expected_truth_hash:
        raise _artifact_error("truth CSV SHA256 does not match the manifest")
    try:
        pair_map = _artifact_pair_map(
            manifest, workloads, filters, table, candidate_validity_predicate
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _artifact_error("manifest candidate/plan provenance is malformed") from exc
    artifact_expected_keys = {
        (workload.name, spec.name, query_no)
        for workload in workloads
        for spec in filters
        for query_no in artifact_query_ids
    }
    requested_expected_keys = {
        (workload.name, spec.name, query_no)
        for workload in workloads
        for spec in filters
        for query_no in query_ids
    }
    truth: dict[tuple[str, str, int], ExactTruth] = {}
    seen_artifact_keys: set[tuple[str, str, int]] = set()
    required_columns = {
        "workload", "filter_name", "predicate", "workload_scalar_predicate_sha256",
        "candidate_universe_predicate", "candidate_universe_predicate_sha256",
        "query_no", "query_id", "query_split", "k", "as_of", "self_excluded",
        "candidate_count", "candidate_min_id", "candidate_max_id", "candidate_ids_sha256", "exact_topk_ids", "exact_topk_distances_sq",
        "exact_topk_plus_one_ids", "exact_topk_plus_one_distances_sq", "kth_distance_sq", "tie_tolerance", "strict_closer_count", "boundary_tied",
    }
    try:
        with truth_csv.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            missing = required_columns - set(reader.fieldnames or ())
            if missing:
                raise _artifact_error(f"truth CSV has wrong schema: missing={sorted(missing)}")
            for row in reader:
                key = (str(row.get("workload")), str(row.get("filter_name")), int(row.get("query_no", -1)))
                if key not in artifact_expected_keys or key in seen_artifact_keys:
                    raise _artifact_error(f"truth CSV has unexpected/duplicate key: {key}")
                seen_artifact_keys.add(key)
                workload_name, filter_name, query_no = key
                pair = pair_map[(workload_name, filter_name)]
                candidate = pair["candidate"]
                exact_backend = pair.get("exact_backend")
                try:
                    backend_valid = (
                        isinstance(exact_backend, dict)
                        and exact_backend.get("backend") == "faiss"
                        and exact_backend.get("class") == "IndexFlatL2"
                        and exact_backend.get("exact") is True
                        and exact_backend.get("local_positions_mapped_to_global_ids") is True
                        and exact_backend.get("order") == "squared_l2_then_global_id"
                        and int(exact_backend.get("index_ntotal", -1)) == int(candidate["count"])
                        and int(exact_backend.get("threads", 0)) == int(backend["threads"])
                        and int(exact_backend.get("search_calls", 0)) > 0
                        and all(
                            math.isfinite(float(exact_backend.get(field, math.nan)))
                            and float(exact_backend[field]) >= 0.0
                            for field in ("index_add_ms", "search_ms", "elapsed_ms")
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    backend_valid = False
                if not backend_valid:
                    raise _artifact_error(f"pair exact backend provenance is malformed: {key}")
                query_id = artifact_query_ids[query_no]
                ids = _csv_ints(row.get("exact_topk_ids"), "exact_topk_ids")
                distances_sq = _csv_floats(row.get("exact_topk_distances_sq"), "exact_topk_distances_sq")
                plus_one_ids = _csv_ints(row.get("exact_topk_plus_one_ids"), "exact_topk_plus_one_ids")
                plus_one_distances_sq = _csv_floats(row.get("exact_topk_plus_one_distances_sq"), "exact_topk_plus_one_distances_sq")
                kth_sq = float(row.get("kth_distance_sq", math.nan))
                source_tolerance = float(row.get("tie_tolerance", math.nan))
                if (
                    row.get("predicate") != next(spec.predicate for spec in filters if spec.name == filter_name)
                    or row.get("workload_scalar_predicate_sha256")
                    != workload_scalar_predicate_sha256(
                        next(spec.predicate for spec in filters if spec.name == filter_name)
                    )
                    or row.get("candidate_universe_predicate")
                    != candidate_validity_predicate
                    or row.get("candidate_universe_predicate_sha256")
                    != expected_validity_hash
                    or int(row.get("query_id", -1)) != query_id
                    or row.get("query_split") != artifact_query_splits[query_no]
                    or int(row.get("k", -1)) != k
                    or int(row.get("as_of", -1)) != as_of_by_workload[workload_name]
                    or str(row.get("self_excluded")).lower() != "true"
                    or len(ids) != k or len(distances_sq) != k
                    or len(plus_one_ids) != k + 1 or len(plus_one_distances_sq) != k + 1
                    or ids != plus_one_ids[:k] or distances_sq != plus_one_distances_sq[:k]
                    or len(set(plus_one_ids)) != len(plus_one_ids)
                    or query_id in plus_one_ids or any(value < 0 for value in plus_one_ids)
                    or any(right < left for left, right in zip(plus_one_distances_sq, plus_one_distances_sq[1:]))
                    or not math.isfinite(kth_sq) or kth_sq < 0.0
                    or not _artifact_float_equal(kth_sq, distances_sq[-1])
                    or not math.isfinite(source_tolerance) or not _artifact_float_equal(source_tolerance, distance_tolerance(kth_sq))
                    or int(row.get("candidate_count", -1)) != int(candidate["count"])
                    or int(row.get("candidate_min_id", -1)) != int(candidate["min_id"])
                    or int(row.get("candidate_max_id", -1)) != int(candidate["max_id"])
                    or row.get("candidate_ids_sha256") != candidate["sha256"]
                ):
                    raise _artifact_error(f"truth CSV record is stale or malformed: {key}")
                # Re-derive tie metadata with the producer's own routine so the
                # float32/float64 casting semantics match bit-for-bit. faiss
                # distances are float32; numpy's value-based casting compares
                # them against the float64 tolerance scalar in float32, which
                # flips ties that a naive float64 recompute would miscount.
                recomputed_ties = exact_truth_contract.truth_metadata(
                    _np.asarray(plus_one_distances_sq, dtype=_np.float32), k
                )
                boundary_tied = recomputed_ties["boundary_tied"]
                if (
                    _artifact_bool(row.get("boundary_tied")) != boundary_tied
                    or int(row.get("strict_closer_count", -1))
                    != recomputed_ties["strict_closer_count"]
                ):
                    raise _artifact_error(f"truth CSV tie metadata is invalid: {key}")
                for check in pair["spot_checks"]:
                    if int(check["query_no"]) == query_no:
                        _validate_artifact_spot_check(check, query_id, plus_one_ids, plus_one_distances_sq)
                kth_l2 = math.sqrt(kth_sq)
                if key in requested_expected_keys:
                    if key in truth:
                        raise _artifact_error(f"truth CSV has a duplicate requested key: {key}")
                    truth[key] = ExactTruth(
                        ids,
                        kth_l2,
                        distance_tolerance(kth_l2),
                        boundary_tied,
                    )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _artifact_error(f"truth CSV is unreadable: {truth_csv}") from exc
    if seen_artifact_keys != artifact_expected_keys:
        raise _artifact_error(
            "truth CSV artifact keyspace is incomplete: "
            f"rows={len(seen_artifact_keys)} expected={expected_rows}"
        )
    if set(truth) != requested_expected_keys:
        raise _artifact_error(
            "truth CSV requested keyspace is incomplete: "
            f"rows={len(truth)} expected={len(requested_expected_keys)}"
        )
    for pair_key, pair in pair_map.items():
        if int(pair.get("as_of", -1)) != as_of_by_workload[pair_key[0]]:
            raise _artifact_error(f"manifest as_of is incompatible: {pair_key}")
        session = pair.get("session")
        if not isinstance(session, dict) or session.get("current_user") != principal:
            raise _artifact_error(f"manifest principal/session is incompatible: {pair_key}")
        relations = pair.get("relations")
        expected_relations = {relation: database_relations.get(relation) for relation in relations} if isinstance(relations, dict) else {}
        if (
            not isinstance(relations, dict)
            or set(relations) != {
                table,
                "public.amazon_review_facts",
                "public.amazon_product_dim",
                "public.amazon_principal_tenant_grants",
                "public.amazon_sql_native_buckets",
            }
            or relations != expected_relations
        ):
            raise _artifact_error(f"manifest relation/table fingerprint is incompatible: {pair_key}")
        for check in pair["spot_checks"]:
            query_no = int(check["query_no"])
            if query_no not in artifact_query_ids or int(check.get("query_id", -1)) != artifact_query_ids[query_no] or int(check["limit"]) != k + 1:
                raise _artifact_error(f"spot check is incompatible: {pair_key}")
    records_sha256 = canonical_sha256([
        [workload, filter_name, query_no, list(entry.ids), entry.kth_distance, entry.tie_tolerance, entry.boundary_tied]
        for (workload, filter_name, query_no), entry in sorted(truth.items())
    ])
    return truth, {
        "truth_csv_sha256": expected_truth_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "records_sha256": records_sha256,
        "run_spec_hash": str(manifest["run_spec_hash"]),
        "query_cohort_sha256": expected_cohort_hash,
        "requested_query_slice_sha256": requested_cohort_hash,
        "candidate_universe_predicate_sha256": expected_validity_hash,
        "relation_epoch_sha256": str(expected_relation_epoch["sha256"]),
    }


def build_run_spec(
    args: argparse.Namespace,
    filters: Sequence[FilterSpec],
    workloads: Sequence[WorkloadSpec],
    calibration: dict[int, int],
    final: dict[int, int],
    database: dict[str, Any],
    as_of_by_workload: dict[str, int],
    external_truth_provenance: dict[str, str] | None = None,
    query_cohort_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_modes = P0_MODES if args.protocol == P0_PROTOCOL else MODES
    tunable_modes = (
        P0_TUNABLE_MODES if args.protocol == P0_PROTOCOL else MODES
    )
    stable_database = {
        key: database[key]
        for key in (
            "database",
            "postgres_version",
            "vector_extension_version",
            "sqlens_build_id",
            "binary_identity_contract",
            "profile_semantics_version",
            "required_profile_fields",
            "relations",
            "d2_graph_proof",
            "d2_index_names",
            "preferred_index_current_settings",
            "mode_indexes",
            "principal",
            "rls_security_proofs",
            "query_candidate_universe_proof",
        )
        if key in database
    }
    reset = database.get("d3_persistent_fragment_reset")
    if isinstance(reset, dict):
        reset_after = reset.get("after") if isinstance(reset.get("after"), dict) else {}
        stable_database["d3_persistent_fragment_empty_start"] = {
            "valid": reset.get("valid"),
            "count": reset_after.get("count"),
            "content_sha256": reset_after.get("content_sha256"),
            "prebuilt_fragments": reset.get("prebuilt_fragments"),
        }
    guard = database.get("formal_data_guard_start")
    if isinstance(guard, dict):
        stable_database["formal_data_version"] = {
            "lock_mode": guard.get("lock_mode"),
            "relations": guard.get("relations"),
            "start_relations": guard.get("start_relations"),
            "start_hash": guard.get("start_hash"),
        }
    combined_queries = {**calibration, **final}
    query_splits = {
        **{query_no: "calibration" for query_no in calibration},
        **{query_no: "final" for query_no in final},
    }
    cohort_hash = query_cohort_sha256(combined_queries, query_splits)
    validity_predicate = validate_candidate_validity_predicate(
        args.candidate_validity_predicate
    )
    data_relations = {
        relation: database.get("relations", {}).get(relation)
        for relation in exact_truth_contract.formal_data_relations(args.vector_table)
        if database.get("relations", {}).get(relation) is not None
    }
    result = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "protocol": args.protocol,
        "runner_sha256": sha256_file(Path(__file__)),
        "schema_sql_sha256": sha256_file(args.schema_sql),
        "filters_csv_sha256": sha256_file(args.filters_csv),
        "query_ids_csv_sha256": sha256_file(args.query_ids_csv),
        "query_cohort_manifest_sha256": sha256_file(args.query_cohort_manifest),
        "database": stable_database,
        "vector_table": args.vector_table,
        "source_index": args.source_index,
        "clone_index": args.clone_index,
        "mode_semantics": {
            mode: asdict(MODE_SPECS[mode]) for mode in benchmark_modes
        },
        "benchmark_modes": list(benchmark_modes),
        "principal": args.principal,
        "k": args.k,
        "targets": list(args.targets),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "schedule_seed": args.schedule_seed,
        "workloads": [asdict(item) for item in workloads],
        "filters": [asdict(item) for item in filters],
        "workload_scalar_predicates": [
            {
                "filter_name": spec.name,
                "predicate": spec.predicate,
                "predicate_sha256": workload_scalar_predicate_sha256(spec.predicate),
            }
            for spec in filters
        ],
        "candidate_universe": {
            "predicate": validity_predicate,
            "predicate_sha256": candidate_universe_predicate_sha256(
                validity_predicate
            ),
            "sql_role": "candidate_relation_only; separate from workload scalar predicate",
        },
        "query_cohort_sha256": cohort_hash,
        "query_cohort_hash_contract": exact_truth_contract.QUERY_COHORT_HASH_CONTRACT,
        "query_cohort": {
            "query_count": len(combined_queries),
            "query_cohort_sha256": cohort_hash,
            "query_cohort_hash_contract": exact_truth_contract.QUERY_COHORT_HASH_CONTRACT,
            "source_query_cohort": query_cohort_provenance,
        },
        "relation_epoch": (
            relation_epoch_contract(data_relations) if data_relations else None
        ),
        "calibration": {
            "query_ids": [[query_no, query_id] for query_no, query_id in calibration.items()],
            "repeats": args.calibration_repeats,
        },
        "final": {
            "query_ids": [[query_no, query_id] for query_no, query_id in final.items()],
            "repeats": args.final_repeats,
        },
        "confirmation": bool(getattr(args, "confirmation", False)),
        "sql_first_workers": int(getattr(args, "sql_first_workers", 1)),
        "config_grids": {
            mode: [asdict(config) for config in build_config_grid(args, mode)]
            for mode in tunable_modes
        },
        "as_of_by_workload": as_of_by_workload,
        "d3": {
            "initialization": "workload_driven_empty_cache_no_prebuilt_fragments",
            "probe_requests": args.d3_probe_requests,
            "min_benefit_per_byte": args.d3_min_benefit_per_byte,
            "max_fragment_mb": args.d3_max_fragment_mb,
            "page_min_skip_rate": args.d3_page_min_skip_rate,
        },
        "sql_hashes": sql_contract_hashes(
            workloads,
            filters,
            args.vector_table,
            args.candidate_validity_predicate,
        ),
    }
    if external_truth_provenance is not None:
        result["external_exact_truth"] = external_truth_provenance
    return result


def _json_ready(value: Any) -> Any:
    """Normalize tuples so JSON-reloaded checkpoints compare equal to live specs."""
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def _run_spec_without_runner(run_spec: dict[str, Any]) -> dict[str, Any]:
    filtered = dict(run_spec)
    filtered.pop("runner_sha256", None)
    return _json_ready(filtered)


def validate_checkpoint_run_spec(checkpoint: dict[str, Any], run_spec: dict[str, Any]) -> None:
    if int(checkpoint.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise RuntimeError("checkpoint version mismatch")
    stored = checkpoint.get("run_spec")
    if not isinstance(stored, dict):
        raise RuntimeError("checkpoint run-spec mismatch; refusing stale resume")
    if checkpoint.get("run_spec_sha256") != canonical_sha256(stored):
        raise RuntimeError("checkpoint run-spec hash is corrupt")
    # Runner bugfixes must not discard valid measurement shards. Every other
    # contract field (SQL, GT, indexes, binary, queries) still has to match.
    stored_contract = _run_spec_without_runner(stored)
    live_contract = _run_spec_without_runner(run_spec)
    if stored_contract != live_contract:
        diffs: list[str] = []
        for key in sorted(set(stored_contract) | set(live_contract)):
            if stored_contract.get(key) == live_contract.get(key):
                continue
            left = stored_contract.get(key)
            right = live_contract.get(key)
            if isinstance(left, dict) and isinstance(right, dict):
                nested = [
                    nested_key
                    for nested_key in sorted(set(left) | set(right))
                    if left.get(nested_key) != right.get(nested_key)
                ]
                diffs.append(f"{key}:{nested[:12]}")
            else:
                diffs.append(key)
        raise RuntimeError(
            "checkpoint run-spec mismatch; refusing stale resume; "
            f"fields={diffs[:20]}"
        )


def checkpoint_entry_path(path: Path, section: str, key: str) -> Path:
    return path / section / f"{canonical_sha256(key)}.json"


def initialize_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        atomic_write_json(temporary / "run_spec.json", checkpoint["run_spec"])
        persist_checkpoint_meta(temporary, checkpoint)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def persist_checkpoint_meta(path: Path, checkpoint: dict[str, Any]) -> None:
    atomic_write_json(
        path / "meta.json",
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "run_spec_sha256": canonical_sha256(checkpoint["run_spec"]),
            "loaded_sessions": checkpoint["loaded_sessions"],
        },
    )


def persist_checkpoint_entry(path: Path, section: str, key: str, value: dict[str, Any]) -> None:
    entry_path = checkpoint_entry_path(path, section, key)
    if entry_path.exists():
        try:
            existing = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"checkpoint shard is unreadable/incomplete: {entry_path}") from exc
        if existing != value:
            raise RuntimeError(f"checkpoint shard changed for immutable key: {key}")
        return
    atomic_write_json(entry_path, value)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        run_spec = json.loads((path / "run_spec.json").read_text(encoding="utf-8"))
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"checkpoint is unreadable/incomplete: {path}") from exc

    def load_section(section: str) -> list[dict[str, Any]]:
        directory = path / section
        if not directory.exists():
            return []
        entries: list[dict[str, Any]] = []
        for entry_path in sorted(directory.glob("*.json")):
            try:
                entry = json.loads(entry_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"checkpoint shard is unreadable/incomplete: {entry_path}"
                ) from exc
            if not isinstance(entry, dict):
                raise RuntimeError(f"checkpoint shard is not an object: {entry_path}")
            entries.append(entry)
        return entries

    return {
        "checkpoint_version": meta.get("checkpoint_version"),
        "run_spec": run_spec,
        "run_spec_sha256": meta.get("run_spec_sha256"),
        "loaded_sessions": meta.get("loaded_sessions"),
        "exact_truth": load_section("exact_truth"),
        "exact_plans": load_section("exact_plans"),
        "calibration_blocks": load_section("calibration_blocks"),
        "final_blocks": load_section("final_blocks"),
        "invalid_blocks": load_section("invalid_blocks"),
    }


def remove_checkpoint(path: Path) -> None:
    tombstone = path.with_name(f".{path.name}.completed-{os.getpid()}")
    os.replace(path, tombstone)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    shutil.rmtree(tombstone)


def exact_truth_record(
    workload: WorkloadSpec,
    spec: FilterSpec,
    query_no: int,
    query_id: int,
    query_split: str,
    as_of: int,
    table_fingerprint_sha256: str,
    sql_text: str,
    truth: ExactTruth,
) -> dict[str, Any]:
    return {
        "workload": workload.name,
        "filter_name": spec.name,
        "query_no": query_no,
        "query_id": query_id,
        "query_split": query_split,
        "as_of": as_of,
        "vector_table_fingerprint_sha256": table_fingerprint_sha256,
        "exact_sql": sql_text,
        "exact_sql_sha256": hashlib.sha256(sql_text.encode()).hexdigest(),
        "ids": list(truth.ids),
        "kth_distance": truth.kth_distance,
        "tie_tolerance": truth.tie_tolerance,
        "boundary_tied": truth.boundary_tied,
    }


def restore_exact_truth(
    records: Sequence[dict[str, Any]],
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    query_ids: dict[int, int],
    query_splits: dict[int, str],
    as_of_by_workload: dict[str, int],
    table: str,
    table_fingerprint_sha256: str,
    k: int,
    *,
    require_complete: bool = False,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[tuple[str, str, int], ExactTruth]:
    workload_by_name = {item.name: item for item in workloads}
    filter_by_name = {item.name: item for item in filters}
    expected_keys = {
        (workload.name, spec.name, query_no)
        for workload in workloads
        for spec in filters
        for query_no in query_ids
    }
    restored: dict[tuple[str, str, int], ExactTruth] = {}
    for record in records:
        key = (str(record.get("workload")), str(record.get("filter_name")), int(record.get("query_no", -1)))
        if key not in expected_keys or key in restored:
            raise RuntimeError(f"checkpoint exact GT has unexpected/duplicate key: {key}")
        workload = workload_by_name[key[0]]
        spec = filter_by_name[key[1]]
        query_no = key[2]
        sql_text = build_hybrid_sql(
            table,
            spec.predicate,
            workload=workload,
            exact=True,
            candidate_validity_predicate=candidate_validity_predicate,
        )
        validate_exact_sql_text(sql_text)
        expected_sql_hash = hashlib.sha256(sql_text.encode()).hexdigest()
        ids = tuple(int(value) for value in record.get("ids", []))
        kth_distance = float(record.get("kth_distance", math.nan))
        tie_tolerance = float(record.get("tie_tolerance", math.nan))
        valid = (
            int(record.get("query_id", -1)) == query_ids[query_no]
            and record.get("query_split") == query_splits[query_no]
            and int(record.get("as_of", -1)) == as_of_by_workload[workload.name]
            and record.get("vector_table_fingerprint_sha256") == table_fingerprint_sha256
            and record.get("exact_sql") == sql_text
            and record.get("exact_sql_sha256") == expected_sql_hash
            and len(ids) == k
            and len(set(ids)) == k
            and all(value != query_ids[query_no] for value in ids)
            and math.isfinite(kth_distance)
            and kth_distance >= 0.0
            and math.isfinite(tie_tolerance)
            and tie_tolerance == distance_tolerance(kth_distance)
            and isinstance(record.get("boundary_tied"), bool)
        )
        if not valid:
            raise RuntimeError(f"checkpoint exact GT is incomplete or stale: {key}")
        restored[key] = ExactTruth(ids, kth_distance, tie_tolerance, record["boundary_tied"])
    if require_complete and set(restored) != expected_keys:
        missing = sorted(expected_keys - set(restored))
        raise RuntimeError(f"checkpoint exact GT is incomplete: missing={missing[:5]} count={len(missing)}")
    return restored


def calibration_block_id(workload: str, filter_name: str, mode: str, config: Config) -> str:
    return f"calibration|{workload}|{filter_name}|{mode}|{config.label}"


def final_block_id(workload: str, filter_name: str, target: float) -> str:
    return f"final|{workload}|{filter_name}|target{target:.12g}"


def validate_measurement_block(
    block: dict[str, Any],
    *,
    phase: str,
    workload: WorkloadSpec,
    spec: FilterSpec,
    query_ids: dict[int, int],
    repeats: int,
    modes: Sequence[str],
    configs: dict[str, Config],
    target_recall: float | None,
    truth: dict[tuple[str, str, int], ExactTruth],
    table: str,
    principal: str,
    source_index: str,
    clone_index: str,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> None:
    expected = {
        (mode, query_no, repeat)
        for mode in modes
        for query_no in query_ids
        for repeat in range(repeats)
    }
    rows = block.get("rows")
    plans = block.get("plans")
    if not isinstance(rows, list) or not isinstance(plans, list):
        raise RuntimeError("checkpoint measurement block is incomplete")
    observed: set[tuple[str, int, int]] = set()
    sql_text = build_hybrid_sql(
        table,
        spec.predicate,
        workload=workload,
        candidate_validity_predicate=candidate_validity_predicate,
    )
    sql_hash = hashlib.sha256(sql_text.encode()).hexdigest()
    for row in rows:
        key = (str(row.get("mode")), int(row.get("query_no", -1)), int(row.get("repeat", -1)))
        if key in observed:
            raise RuntimeError(f"checkpoint measurement block has duplicate row: {key}")
        observed.add(key)
        mode, query_no, repeat = key
        truth_entry = truth.get((workload.name, spec.name, query_no))
        if mode not in configs or truth_entry is None:
            raise RuntimeError(f"checkpoint measurement block has unexpected row: {key}")
        expected_target: Any = "" if target_recall is None else target_recall
        expected_index = mode_index(mode, source_index, clone_index)
        expected_strategy = MODE_SPECS[mode].filter_strategy
        valid = (
            row.get("phase") == phase
            and row.get("workload") == workload.name
            and row.get("filter_name") == spec.name
            and row.get("predicate") == spec.predicate
            and row.get("workload_scalar_predicate_sha256")
            == workload_scalar_predicate_sha256(spec.predicate)
            and row.get("candidate_universe_predicate")
            == candidate_validity_predicate
            and row.get("candidate_universe_predicate_sha256")
            == candidate_universe_predicate_sha256(candidate_validity_predicate)
            and row.get("config") == configs[mode].label
            and int(row.get("query_id", -1)) == query_ids[query_no]
            and row.get("target_recall") == expected_target
            and row.get("pair_key") == f"{workload.name}|{spec.name}|q{query_no}|r{repeat}"
            and row.get("query_sql") == sql_text
            and row.get("query_sql_sha256") == sql_hash
            and row.get("exact_gt_ids") == ",".join(str(value) for value in truth_entry.ids)
            and float(row.get("exact_gt_kth_distance", math.nan)) == truth_entry.kth_distance
            and float(row.get("exact_gt_tie_tolerance", math.nan)) == truth_entry.tie_tolerance
            and row.get("exact_gt_boundary_tied") is truth_entry.boundary_tied
            and row.get("selected_vector_index") == expected_index
            and row.get("preferred_index_current_setting") == expected_index
            and row.get("principal") == principal
            and row.get("snapshot_as_of") == as_of_value(workload.name, row)
            and row.get("filter_strategy") == expected_strategy
            and row.get("guidance_semantics") == MODE_SPECS[mode].guidance_semantics
            and not bool(row.get("hard_traversal_used"))
            and recorded_guidance_proof_is_valid(row)
        )
        if not valid:
            raise RuntimeError(f"checkpoint measurement row is incomplete or stale: {key}")
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError("checkpoint measurement block has wrong row count/query IDs/repeats")
    if len(plans) != len(modes):
        raise RuntimeError("checkpoint measurement block has wrong EXPLAIN plan count")
    for plan in plans:
        mode = str(plan.get("mode"))
        gate = plan.get("explain_gate", {})
        expected_index = mode_index(mode, source_index, clone_index) if mode in MODE_SPECS else ""
        if (
            mode not in configs
            or plan.get("phase") != phase
            or plan.get("workload") != workload.name
            or plan.get("filter_name") != spec.name
            or plan.get("config") != configs[mode].label
            or plan.get("sql_sha256") != sql_hash
            or not gate.get("valid")
            or gate.get("require_hnsw") is not True
            or gate.get("expected_index_qualified") != expected_index
            or plan.get("selected_vector_index") != expected_index
            or plan.get("preferred_index_current_setting") != expected_index
            or plan.get("principal") != principal
            or plan.get("snapshot_as_of") != int(plan.get("as_of", -1))
            or plan.get("filter_strategy") != MODE_SPECS[mode].filter_strategy
            or bool(plan.get("hard_traversal_used"))
            or plan.get("plan_state_proof", {}).get("valid") is not True
            or plan.get("explain_order") != "after_all_timed_requests_in_block"
        ):
            raise RuntimeError("checkpoint measurement EXPLAIN plan is incomplete or stale")


def as_of_value(workload_name: str, row: dict[str, Any]) -> int:
    try:
        return int(row["as_of"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"measurement row has invalid as_of for {workload_name}") from exc


def validate_exact_plans(
    plans: Sequence[dict[str, Any]],
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    table: str,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> None:
    expected = {(workload.name, spec.name) for workload in workloads for spec in filters}
    observed: set[tuple[str, str]] = set()
    workload_by_name = {item.name: item for item in workloads}
    filter_by_name = {item.name: item for item in filters}
    for plan in plans:
        key = (str(plan.get("workload")), str(plan.get("filter_name")))
        if key not in expected or key in observed:
            raise RuntimeError(f"checkpoint exact EXPLAIN plan has unexpected/duplicate key: {key}")
        observed.add(key)
        sql_text = build_hybrid_sql(
            table,
            filter_by_name[key[1]].predicate,
            workload=workload_by_name[key[0]],
            exact=True,
            candidate_validity_predicate=candidate_validity_predicate,
        )
        validate_exact_sql_text(sql_text)
        gate = plan.get("explain_gate", {})
        if (
            plan.get("phase") != "exact_gt"
            or plan.get("mode") != "exact_gt"
            or plan.get("sql_sha256") != hashlib.sha256(sql_text.encode()).hexdigest()
            or not gate.get("valid")
            or gate.get("require_hnsw") is not False
        ):
            raise RuntimeError(f"checkpoint exact EXPLAIN plan is stale: {key}")


def run_exact_truth(
    cur: Any,
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    query_ids: dict[int, int],
    as_of_by_workload: dict[str, int],
    table: str,
    vector_index: str,
    k: int,
    plans: list[dict[str, Any]],
    existing: dict[tuple[str, str, int], ExactTruth] | None = None,
    on_truth: Any | None = None,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[tuple[str, str, int], ExactTruth]:
    truth: dict[tuple[str, str, int], ExactTruth] = dict(existing or {})
    for workload in workloads:
        for spec in filters:
            sql_text = build_hybrid_sql(
                table,
                spec.predicate,
                workload=workload,
                exact=True,
                candidate_validity_predicate=candidate_validity_predicate,
            )
            validate_exact_sql_text(sql_text)
            first_query_id = next(iter(query_ids.values()))
            set_as_of(cur, as_of_by_workload[workload.name])
            exact_plan, gate = explain(
                cur,
                sql_text,
                {
                    "query_id": first_query_id,
                    "as_of": as_of_by_workload[workload.name],
                    "k": k,
                },
                vector_index=vector_index,
                require_hnsw=False,
            )
            prior_plan = next(
                (
                    plan
                    for plan in plans
                    if plan.get("phase") == "exact_gt"
                    and plan.get("workload") == workload.name
                    and plan.get("filter_name") == spec.name
                ),
                None,
            )
            if prior_plan is None:
                plans.append(
                    {
                        "phase": "exact_gt",
                        "workload": workload.name,
                        "filter_name": spec.name,
                        "mode": "exact_gt",
                        "sql_sha256": hashlib.sha256(sql_text.encode()).hexdigest(),
                        "plan": exact_plan,
                        "explain_gate": gate,
                    }
                )
            for query_no, query_id in query_ids.items():
                if (workload.name, spec.name, query_no) in truth:
                    continue
                set_as_of(cur, as_of_by_workload[workload.name])
                results = query_results(
                    cur,
                    sql_text,
                    {
                        "query_id": query_id,
                        "as_of": as_of_by_workload[workload.name],
                        "k": k + 1,
                    },
                    exact=True,
                )
                ids = [row_id for row_id, _ in results[:k]]
                if len(ids) != k or len(set(ids)) != k:
                    raise RuntimeError(
                        f"exact SQL GT incomplete for {workload.name}/{spec.name}/q{query_no}: {len(ids)} rows"
                    )
                kth_distance = float(results[k - 1][1])
                tolerance = distance_tolerance(kth_distance)
                truth_entry = ExactTruth(
                    ids=tuple(ids),
                    kth_distance=kth_distance,
                    tie_tolerance=tolerance,
                    boundary_tied=(
                        len(results) > k and float(results[k][1]) <= kth_distance + tolerance
                    ),
                )
                truth[(workload.name, spec.name, query_no)] = truth_entry
                if on_truth is not None:
                    on_truth(workload, spec, query_no, query_id, sql_text, truth_entry)
    return truth


def run_measurements(
    connections: dict[str, Any],
    configs: dict[str, Config],
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    query_ids: dict[int, int],
    truth: dict[tuple[str, str, int], ExactTruth | tuple[int, ...]],
    as_of_by_workload: dict[str, int],
    table: str,
    source_index: str,
    clone_index: str,
    principal: str,
    k: int,
    repeats: int,
    phase: str,
    target_recall: float | None,
    schedule_seed: int,
    selected_modes: Sequence[str] | None = None,
    d3_settings: dict[str, Any] | None = None,
    fragment_store_reset: dict[str, Any] | None = None,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
    request_keys: Sequence[tuple[str, str, int, int]] | None = None,
    registered_scalar_indexes: Sequence[str] = (),
    configs_by_cell: Mapping[tuple[str, str, str], Config] | None = None,
    query_embeddings: Mapping[int, str] | None = None,
    sql_first_workers: int = 1,
    sql_first_connections: Sequence[Any] | None = None,
    progress_every: int = 0,
    sql_first_after_sequential: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active_modes = tuple(selected_modes or MODES)
    if not active_modes or any(mode not in ALL_MODES for mode in active_modes):
        raise ValueError(f"selected_modes must be a non-empty subset of {ALL_MODES}")
    if any(mode not in configs for mode in active_modes):
        raise ValueError(f"missing config for selected mode: {active_modes}")
    keys = list(request_keys) if request_keys is not None else [
        (workload.name, spec.name, query_no, repeat)
        for workload in workloads
        for spec in filters
        for query_no in query_ids
        for repeat in range(repeats)
    ]
    valid_workloads = {workload.name for workload in workloads}
    valid_filters = {spec.name for spec in filters}
    if (
        len(keys) != len(set(keys))
        or any(
            workload not in valid_workloads
            or filter_name not in valid_filters
            or query_no not in query_ids
            or repeat < 0
            for workload, filter_name, query_no, repeat in keys
        )
    ):
        raise ValueError("request_keys contains duplicate or out-of-slice requests")
    if SQL_FIRST_MODE in active_modes and not registered_scalar_indexes:
        raise ValueError("SQL-first measurement requires registered scalar indexes")
    if SQL_FIRST_MODE in active_modes:
        sql_first_after_sequential = True
    rows: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    cursors = {mode: connections[mode].cursor() for mode in active_modes}
    selected_indexes = {
        mode: mode_index(mode, source_index, clone_index) for mode in active_modes
    }
    active_config_labels: dict[str, str] = {}

    def config_for(workload_name: str, filter_name: str, mode: str) -> Config:
        if configs_by_cell is not None:
            selected = configs_by_cell.get((workload_name, filter_name, mode))
            if selected is None:
                raise ValueError(
                    "missing per-cell config for "
                    f"{workload_name}/{filter_name}/{mode}"
                )
            return selected
        return configs[mode]

    approx_modes = [mode for mode in active_modes if mode != SQL_FIRST_MODE]
    control_cur = cursors[approx_modes[0]] if approx_modes else next(iter(cursors.values()))

    def hide_heap_competing_indexes() -> None:
        if approx_modes:
            set_heap_competing_indexes_valid(control_cur, table, valid=False)

    def restore_heap_competing_indexes() -> None:
        if approx_modes:
            set_heap_competing_indexes_valid(control_cur, table, valid=True)

    try:
        hide_heap_competing_indexes()
        for mode in active_modes:
            set_mode(
                cursors[mode], mode, configs[mode], selected_indexes[mode], d3_settings
            )
            active_config_labels[mode] = configs[mode].label
        schedule = interleaved_schedule(keys, active_modes, schedule_seed)

        def run_one(
            position: int,
            key: tuple[Any, ...],
            mode: str,
            cur: Any,
            labels: dict[str, str],
        ) -> dict[str, Any]:
            workload_name, filter_name, query_no, repeat = key
            workload = next(item for item in workloads if item.name == workload_name)
            spec = next(item for item in filters if item.name == filter_name)
            request_config = config_for(workload_name, filter_name, mode)
            query_id = query_ids[query_no]
            sql_text = build_hybrid_sql(
                table,
                spec.predicate,
                workload=workload,
                exact=mode == SQL_FIRST_MODE,
                candidate_validity_predicate=candidate_validity_predicate,
            )
            params = {
                "query_id": query_id,
                "as_of": as_of_by_workload[workload.name],
                "k": k,
                "vector_index": selected_indexes[mode],
                "binding_atoms": list(binding_atoms_for(workload, spec)),
                "binding_kind": MODE_SPECS[mode].guidance_kind or "bloom",
            }
            if mode != SQL_FIRST_MODE:
                params = bind_query_embedding(
                    params, query_id, query_embeddings or {}
                )
            if cur is None:
                raise RuntimeError(f"measurement cursor is missing for {mode}")
            error = ""
            ids: list[int] = []
            result_pairs: list[tuple[int, float]] = []
            activation_ms = 0.0
            query_ms = 0.0
            elapsed_ms = 0.0
            activation: dict[str, Any] = {}
            post_guidance: dict[str, Any] = {}
            execution_guidance: dict[str, Any] = {}
            scan_profile: dict[str, Any] = {}
            context: dict[str, Any] = {}
            try:
                if (
                    mode != SQL_FIRST_MODE
                    and labels.get(mode) != request_config.label
                ):
                    set_search_config(cur, request_config)
                    labels[mode] = request_config.label
                if mode != SQL_FIRST_MODE:
                    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
                e2e_started = time.perf_counter()
                set_as_of(cur, as_of_by_workload[workload.name])
                activation = configure_guidance(
                    cur, mode, selected_indexes[mode],
                    binding_atoms_for(workload, spec),
                )
                activation_ms = float(activation["activation_ms"])
                query_started = time.perf_counter()
                cur.execute(sql_text, params)
                query_rows = cur.fetchall()
                result_pairs = [(int(row[0]), float(row[1])) for row in query_rows]
                ids = [row_id for row_id, _ in result_pairs]
                query_ms = (time.perf_counter() - query_started) * 1000.0
                elapsed_ms = (time.perf_counter() - e2e_started) * 1000.0
                context = runtime_sql_context(
                    cur, principal, as_of_by_workload[workload.name]
                )
                if context["preferred_index"] != selected_indexes[mode]:
                    raise RuntimeError(
                        f"preferred index changed during query for {mode}: {context}"
                    )
                if query_rows and len(query_rows[-1]) > 2:
                    value = query_rows[-1][2]
                    execution_guidance = (
                        json.loads(value) if isinstance(value, str) else dict(value or {})
                    )
                if mode != SQL_FIRST_MODE:
                    post_guidance = fetch_json_object(
                        cur, "SELECT vector_hnsw_guidance_profile()"
                    )
                    scan_profile = fetch_json_object(
                        cur, "SELECT vector_hnsw_last_scan_profile()"
                    )
            except Exception as exc:  # noqa: BLE001 - retain failed pair in the artifact.
                error = f"{exc.__class__.__name__}: {exc}"
                try:
                    cur.execute("ROLLBACK")
                    # Abort resets r43's backend-local fragment-store ready flag.
                    ensure_sqlens_fragment_catalog(cur, principal, table)
                    set_mode(
                        cur,
                        mode,
                        request_config,
                        selected_indexes[mode],
                        d3_settings,
                    )
                except Exception:
                    pass
            if error:
                activation_ms = 0.0
                query_ms = 0.0
                elapsed_ms = 0.0
            truth_entry = truth[(workload.name, spec.name, query_no)]
            truth_ids = truth_entry.ids if isinstance(truth_entry, ExactTruth) else truth_entry
            recall = (
                tie_aware_recall_at_k(result_pairs, truth_entry, query_id, k)
                if isinstance(truth_entry, ExactTruth)
                else recall_at_k(ids, truth_ids, k)
            )
            adaptive = (
                adaptive_transition_for_request(activation, post_guidance)
                if MODE_SPECS[mode].adaptive and not error
                else {
                    "adaptive_state_before": "not_adaptive",
                    "adaptive_state_after_activation": "not_adaptive",
                    "adaptive_state_after_query": "not_adaptive",
                    "adaptive_probe_observed": False,
                    "adaptive_admission_observed": False,
                    "adaptive_materialized": False,
                    "adaptive_active": False,
                    "hidden_prebuilt_fragment_reused": False,
                    "fragment_store_hit_delta": 0,
                    "adaptive_transition": "not_adaptive",
                }
            )
            final_path = str(scan_profile.get("final_path", ""))
            execution_proof = guidance_execution_proof(
                mode, activation, execution_guidance, scan_profile, post_guidance
            )
            if not error and not execution_proof["valid"]:
                error = (
                    "RuntimeError: per-row guidance execution proof failed: "
                    + json.dumps(execution_proof, sort_keys=True)
                )
            hard_traversal_used = bool(
                mode in SQLENS_MODES
                and (
                    not bool(execution_proof.get("valid"))
                    or context.get("filter_strategy") not in {"off", "safe_guided"}
                    or final_path in {"guided", "legacy_guided", "unknown", ""}
                )
            )
            return {
                    "phase": phase,
                    "target_recall": target_recall if target_recall is not None else "",
                    "workload": workload.name,
                    "filter_name": filter_name,
                    "predicate": spec.predicate,
                    "workload_scalar_predicate_sha256": workload_scalar_predicate_sha256(
                        spec.predicate
                    ),
                    "candidate_universe_predicate": candidate_validity_predicate,
                    "candidate_universe_predicate_sha256": candidate_universe_predicate_sha256(
                        candidate_validity_predicate
                    ),
                    "as_of": as_of_by_workload[workload.name],
                    "principal": context.get("current_user", ""),
                    "snapshot_as_of": (
                        int(context["app_as_of"]) if context.get("app_as_of") else -1
                    ),
                    "mode": mode,
                    "selected_vector_index": selected_indexes[mode],
                    "preferred_index_current_setting": context.get(
                        "preferred_index", ""
                    ),
                    "filter_strategy": MODE_SPECS[mode].filter_strategy,
                    "filter_strategy_current_setting": context.get(
                        "filter_strategy", ""
                    ),
                    "page_access_current_setting": context.get("page_access", ""),
                    "index_page_access_current_setting": context.get(
                        "index_page_access", ""
                    ),
                    "guidance_kind": MODE_SPECS[mode].guidance_kind or "none",
                    "guidance_semantics": MODE_SPECS[mode].guidance_semantics,
                    "hard_traversal_used": hard_traversal_used,
                    "traversal_final_path": final_path,
                    "config": request_config.label,
                    "query_no": query_no,
                    "query_id": query_id,
                    "repeat": repeat,
                    "pair_key": f"{workload.name}|{filter_name}|q{query_no}|r{repeat}",
                    "schedule_position": position,
                    "execution_order": "interleaved",
                    "query_sql": sql_text,
                    "query_sql_sha256": hashlib.sha256(sql_text.encode()).hexdigest(),
                    "exact_gt_ids": ",".join(str(value) for value in truth_ids),
                    "exact_gt_kth_distance": (
                        truth_entry.kth_distance if isinstance(truth_entry, ExactTruth) else NA
                    ),
                    "exact_gt_tie_tolerance": (
                        truth_entry.tie_tolerance if isinstance(truth_entry, ExactTruth) else NA
                    ),
                    "exact_gt_boundary_tied": (
                        truth_entry.boundary_tied if isinstance(truth_entry, ExactTruth) else NA
                    ),
                    "returned_ids": ",".join(str(value) for value in ids),
                    "returned_distances": ",".join(f"{distance:.17g}" for _, distance in result_pairs),
                    "returned": len(ids),
                    "recall": recall if not error else NA,
                    "activation_ms": activation_ms if not error else NA,
                    "query_ms": query_ms if not error else NA,
                    "e2e_ms": elapsed_ms if not error else NA,
                    "guidance_enabled": bool(
                        activation.get("guidance_enabled", False)
                    ),
                    "guidance_route": activation.get("guidance_route", ""),
                    "activation_atom_count": activation.get(
                        "activation_atom_count", 0
                    ),
                    "adaptive_initialization": (
                        "workload_driven_empty_cache_no_prebuilt_fragments"
                        if MODE_SPECS[mode].adaptive
                        else "not_adaptive"
                    ),
                    "prebuilt_fragments": (
                        int(fragment_store_reset.get("prebuilt_fragments", -1))
                        if fragment_store_reset is not None
                        else NA
                    ),
                    "persistent_fragment_reset_proof": (
                        fragment_store_reset or {"status": "not_supplied"}
                    ),
                    "guidance_activation_profile": activation,
                    "execution_guidance_profile": execution_guidance,
                    "post_query_guidance_profile": post_guidance,
                    "guidance_execution_proof": execution_proof,
                    "guidance_binding_matched": execution_proof["binding_matched"],
                    "guidance_effective_active": execution_proof["effective_active"],
                    "guidance_checks": execution_proof["guidance_checks"],
                    "guidance_final_path": execution_proof["final_path"],
                    **adaptive,
                    **scan_profile_export(scan_profile),
                    "scan_profile": scan_profile,
                    "error": error,
                }

        sequential_items, parallel_items = partition_measurement_schedule(
            schedule, SQL_FIRST_MODE
        )
        worker_conns = list(sql_first_connections or ())
        use_pool = (
            SQL_FIRST_MODE in active_modes
            and int(sql_first_workers) > 1
            and bool(parallel_items)
            and len(worker_conns) >= 1
        )
        collected: list[dict[str, Any] | None] = [None] * len(schedule)
        progress_lock = threading.Lock()
        progress_done = {"sequential": 0, "sql_first": 0}

        def emit_progress(group: str, total: int) -> None:
            if progress_every <= 0:
                return
            with progress_lock:
                progress_done[group] += 1
                done = progress_done[group]
            if done == total or done % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "progress": "measurement_tick",
                            "group": group,
                            "completed": done,
                            "total": total,
                        }
                    ),
                    flush=True,
                )

        def run_sequential() -> None:
            labels = dict(active_config_labels)
            for position, key, mode in sequential_items:
                collected[position] = run_one(
                    position, key, mode, cursors[mode], labels
                )
                emit_progress("sequential", len(sequential_items))

        if use_pool:
            pool: Queue[Any] = Queue()
            for conn in worker_conns:
                pool.put(conn)

            def run_sql_first_item(
                item: tuple[int, tuple[Any, ...], str]
            ) -> tuple[int, dict[str, Any]]:
                position, key, mode = item
                conn = pool.get()
                try:
                    cur = conn.cursor()
                    try:
                        row = run_one(position, key, mode, cur, {})
                    finally:
                        cur.close()
                    return position, row
                finally:
                    pool.put(conn)

            worker_count = min(int(sql_first_workers), len(worker_conns), len(parallel_items))
            with ThreadPoolExecutor(max_workers=max(worker_count, 1) + 1) as pool_exec:
                if sql_first_after_sequential:
                    run_sequential()
                    restore_heap_competing_indexes()
                    parallel_futures = [
                        pool_exec.submit(run_sql_first_item, item)
                        for item in parallel_items
                    ]
                    for future in as_completed(parallel_futures):
                        position, row = future.result()
                        collected[position] = row
                        emit_progress("sql_first", len(parallel_items))
                else:
                    sequential_future = pool_exec.submit(run_sequential)
                    parallel_futures = [
                        pool_exec.submit(run_sql_first_item, item)
                        for item in parallel_items
                    ]
                    pending = {sequential_future, *parallel_futures}
                    for future in as_completed(pending):
                        if future is sequential_future:
                            future.result()
                            continue
                        position, row = future.result()
                        collected[position] = row
                        emit_progress("sql_first", len(parallel_items))
        elif SQL_FIRST_MODE in active_modes and sql_first_after_sequential:
            run_sequential()
            restore_heap_competing_indexes()
            labels = dict(active_config_labels)
            for position, key, mode in parallel_items:
                collected[position] = run_one(
                    position, key, mode, cursors[mode], labels
                )
                emit_progress("sql_first", len(parallel_items))
        else:
            labels = dict(active_config_labels)
            for position, (key, mode) in enumerate(schedule):
                collected[position] = run_one(
                    position, key, mode, cursors[mode], labels
                )
                emit_progress("sequential", len(schedule))
        rows.extend(row for row in collected if row is not None)
        if any(row is None for row in collected):
            raise RuntimeError("measurement schedule produced an incomplete row set")
        for workload in workloads:
            for spec in filters:
                params = {
                    "query_id": next(iter(query_ids.values())),
                    "as_of": as_of_by_workload[workload.name],
                    "k": k,
                    "vector_index": "",
                    "binding_atoms": list(binding_atoms_for(workload, spec)),
                    "binding_kind": "bloom",
                }
                if any(mode != SQL_FIRST_MODE for mode in active_modes):
                    params = bind_query_embedding(
                        params, int(params["query_id"]), query_embeddings or {}
                    )
                for mode in active_modes:
                    if mode == SQL_FIRST_MODE:
                        restore_heap_competing_indexes()
                    else:
                        hide_heap_competing_indexes()
                    request_config = config_for(
                        workload.name, spec.name, mode
                    )
                    sql_text = build_hybrid_sql(
                        table,
                        spec.predicate,
                        workload=workload,
                        exact=mode == SQL_FIRST_MODE,
                        candidate_validity_predicate=candidate_validity_predicate,
                    )
                    vector_index = selected_indexes[mode]
                    params["vector_index"] = vector_index
                    params["binding_kind"] = MODE_SPECS[mode].guidance_kind or "bloom"
                    set_mode(
                        cursors[mode],
                        mode,
                        request_config,
                        vector_index,
                        d3_settings,
                    )
                    set_as_of(cursors[mode], as_of_by_workload[workload.name])
                    plan_state_before = prepare_explain_without_runtime_state(
                        cursors[mode]
                    )
                    context = runtime_sql_context(
                        cursors[mode], principal, as_of_by_workload[workload.name]
                    )
                    if context["preferred_index"] != vector_index:
                        raise RuntimeError(
                            f"preferred index changed before EXPLAIN for {mode}: {context}"
                        )
                    plan, gate = explain(
                        cursors[mode],
                        sql_text,
                        params,
                        vector_index=vector_index,
                        require_hnsw=mode != SQL_FIRST_MODE,
                    )
                    if mode == SQL_FIRST_MODE:
                        gate = validate_sql_first_explain_gate(
                            plan, registered_scalar_indexes
                        )
                    plan_state_proof = finish_explain_without_runtime_state(
                        cursors[mode], plan_state_before
                    )
                    plans.append(
                        {
                            "phase": phase,
                            "target_recall": target_recall,
                            "workload": workload.name,
                            "filter_name": spec.name,
                            "mode": mode,
                            "config": request_config.label,
                            "sql_sha256": hashlib.sha256(sql_text.encode()).hexdigest(),
                            "as_of": as_of_by_workload[workload.name],
                            "principal": context["current_user"],
                            "snapshot_as_of": int(context["app_as_of"]),
                            "selected_vector_index": vector_index,
                            "preferred_index_current_setting": context["preferred_index"],
                            "filter_strategy": MODE_SPECS[mode].filter_strategy,
                            "filter_strategy_current_setting": context["filter_strategy"],
                            "page_access_current_setting": context["page_access"],
                            "index_page_access_current_setting": context[
                                "index_page_access"
                            ],
                            "guidance_semantics": MODE_SPECS[mode].guidance_semantics,
                            "hard_traversal_used": False,
                            "explain_order": "after_all_timed_requests_in_block",
                            "plan_state_proof": plan_state_proof,
                            "plan": plan,
                            "explain_gate": gate,
                        }
                    )
    finally:
        restore_heap_competing_indexes()
        for cur in cursors.values():
            cur.close()
    return rows, plans


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    target = io.StringIO(newline="")
    if fields:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, target.getvalue())


def validate_p0_mixed_block(
    block: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    repeat: int,
    modes: Sequence[str],
    configs_by_cell: Mapping[tuple[str, str, str], Config],
) -> None:
    if (
        block.get("phase") != "measurement"
        or int(block.get("repeat", -1)) != repeat
        or block.get("trace_sha256") != mixed_trace_sha256(trace)
    ):
        raise RuntimeError("P0 mixed block metadata is stale")
    rows = block.get("rows")
    plans = block.get("plans")
    if not isinstance(rows, list) or not isinstance(plans, list):
        raise RuntimeError("P0 mixed block rows/plans are missing")
    expected = {
        (
            str(request["workload"]),
            str(request["filter_name"]),
            int(request["query_no"]),
            repeat,
            mode,
        )
        for request in trace
        for mode in modes
    }
    observed = {
        (
            str(row.get("workload")),
            str(row.get("filter_name")),
            int(row.get("query_no", -1)),
            int(row.get("repeat", -1)),
            str(row.get("mode")),
        )
        for row in rows
    }
    if len(rows) != len(expected) or observed != expected:
        raise RuntimeError("P0 mixed block request/mode coverage is incomplete")
    for row in rows:
        key = (
            str(row["workload"]),
            str(row["filter_name"]),
            str(row["mode"]),
        )
        config = configs_by_cell.get(key)
        if config is None:
            raise RuntimeError(f"P0 mixed row is invalid: {key} missing per-cell config")
        if row.get("config") != config.label:
            raise RuntimeError(
                f"P0 mixed row is invalid: {key} config "
                f"{row.get('config')!r} != {config.label!r}"
            )
        if row.get("error"):
            raise RuntimeError(
                f"P0 mixed row is invalid: {key} error={row.get('error')!r}"
            )
        if row.get("e2e_ms") in (None, NA):
            raise RuntimeError(f"P0 mixed row is invalid: {key} missing e2e_ms")
        if not recorded_guidance_proof_is_valid(row):
            raise RuntimeError(
                f"P0 mixed row is invalid: {key} guidance execution proof failed"
            )
    expected_plan_keys = {
        (
            str(request["workload"]),
            str(request["filter_name"]),
            mode,
        )
        for request in trace
        for mode in modes
    }
    observed_plan_keys = {
        (
            str(plan.get("workload")),
            str(plan.get("filter_name")),
            str(plan.get("mode")),
        )
        for plan in plans
    }
    if observed_plan_keys != expected_plan_keys or len(plans) != len(expected_plan_keys):
        raise RuntimeError("P0 mixed block EXPLAIN coverage is incomplete")
    for plan in plans:
        gate = plan.get("explain_gate")
        if (
            not isinstance(gate, dict)
            or gate.get("valid") is not True
            or (
                plan.get("mode") == SQL_FIRST_MODE
                and (
                    gate.get("forced_indexed") is not True
                    or gate.get("vector_hnsw_index_names") != []
                    or not gate.get("matched_scalar_indexes")
                )
            )
        ):
            raise RuntimeError("P0 mixed block contains an invalid EXPLAIN gate")


def p0_requested_slice_completion(
    rows: Sequence[Mapping[str, Any]],
    trace: Sequence[Mapping[str, Any]],
    modes: Sequence[str],
    repeats: int,
) -> dict[str, Any]:
    measurement = [row for row in rows if row.get("phase") == "measurement"]
    expected_rows = len(trace) * len(modes) * repeats
    observed_keys = {
        (
            str(row.get("mode")),
            int(row.get("repeat", -1)),
            str(row.get("pair_key")),
        )
        for row in measurement
    }
    expected_keys = {
        (
            mode,
            repeat,
            (
                f"{request['workload']}|{request['filter_name']}|"
                f"q{int(request['query_no'])}|r{repeat}"
            ),
        )
        for request in trace
        for mode in modes
        for repeat in range(repeats)
    }
    errors = sum(bool(row.get("error")) for row in measurement)
    return {
        "protocol": P0_PROTOCOL,
        "requested_queries": len(trace),
        "requested_repeats": repeats,
        "requested_modes": list(modes),
        "expected_measurement_rows": expected_rows,
        "observed_measurement_rows": len(measurement),
        "expected_request_mode_repeat_keys": len(expected_keys),
        "observed_request_mode_repeat_keys": len(observed_keys),
        "errors": errors,
        "trace_sha256": mixed_trace_sha256(trace),
        "complete": bool(
            len(measurement) == expected_rows
            and observed_keys == expected_keys
            and errors == 0
        ),
    }


def publish_benchmark_artifacts(
    args: argparse.Namespace,
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    plans: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    checkpoint_path: Path,
) -> int:
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    plans_path = args.plans or args.out.with_suffix(".plans.json")
    valid = manifest.get("artifact_valid") is True
    if not valid:
        manifest["paper_eligible"] = False
        manifest["formal_outputs_published"] = False
        manifest["checkpoint_preserved"] = str(checkpoint_path)
        atomic_write_json(plans_path, plans)
        manifest["outputs"] = {
            "plans": {
                "path": str(plans_path.resolve()),
                "sha256": sha256_file(plans_path),
                "rows": len(plans),
            }
        }
        atomic_write_json(manifest_path, manifest)
        return 2
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    write_csv(args.out, rows)
    write_csv(summary_path, summaries)
    atomic_write_json(plans_path, plans)
    manifest["outputs"] = {
        "raw": {
            "path": str(args.out.resolve()),
            "sha256": sha256_file(args.out),
            "rows": len(rows),
        },
        "summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256_file(summary_path),
            "rows": len(summaries),
        },
        "plans": {
            "path": str(plans_path.resolve()),
            "sha256": sha256_file(plans_path),
            "rows": len(plans),
        },
    }
    requested = manifest.get("requested_slice_completion")
    requested_complete = (
        requested.get("complete") is True
        if isinstance(requested, dict)
        else True
    )
    if getattr(args, "screening", False):
        manifest["paper_eligible"] = False
        manifest["confirmation_only"] = True
        manifest["screening_only"] = True
        manifest["confirmation_reason"] = (
            "q1k/r1 screening slice; not the paper q10k/r3 measurement"
        )
    elif getattr(args, "confirmation", False):
        manifest["paper_eligible"] = False
        manifest["confirmation_only"] = True
        manifest["confirmation_reason"] = (
            "q2k/r1 confirmation slice; not the paper q10k/r3 measurement"
        )
    else:
        manifest["paper_eligible"] = bool(valid and requested_complete)
    manifest["formal_outputs_published"] = True
    atomic_write_json(manifest_path, manifest)
    remove_checkpoint(checkpoint_path)
    return 0


def container_name_arg(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("invalid Docker container name or ID")
    return value


def image_digest_arg(value: str) -> str:
    normalized = value.lower()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized):
        raise argparse.ArgumentTypeError("expected a sha256:<64-hex> image digest")
    return normalized


def git_commit_arg(value: str) -> str:
    normalized = value.lower()
    if not re.fullmatch(r"[0-9a-f]{40}", normalized):
        raise argparse.ArgumentTypeError("expected a 40-character pgvector git commit")
    return normalized


def dsn_fingerprint(conninfo: str) -> str:
    """Bind a target without persisting credentials in a release manifest."""
    if not isinstance(conninfo, str) or not conninfo.strip():
        raise ValueError("an explicit non-empty DSN is required")
    return hashlib.sha256(conninfo.encode("utf-8")).hexdigest()


def inspect_container_image(container: str, expected_digest: str) -> dict[str, str]:
    """Verify the exact running image used by the independently addressed DSN."""
    result = subprocess.run(
        ["docker", "inspect", "--format={{.Image}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect official upstream container {container!r}")
    actual = result.stdout.strip().lower()
    if actual != expected_digest:
        raise RuntimeError(
            "official upstream container image digest mismatch: "
            f"expected {expected_digest}, got {actual or '<empty>'}"
        )
    return {
        "container": container,
        "image_digest": actual,
        "image_digest_matches_expected": True,
    }


def official_runtime_identity(cur: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Read the live binary identity from the independent upstream server."""
    cur.execute(
        "WITH lib AS ("
        "SELECT setting || '/vector.so' AS path FROM pg_config WHERE name='PKGLIBDIR'"
        ") SELECT current_database(), current_setting('server_version'), "
        "coalesce((SELECT extversion FROM pg_extension WHERE extname='vector'), ''), "
        "(SELECT path FROM lib), "
        "(SELECT encode(sha256(pg_read_binary_file(path)), 'hex') FROM lib)"
    )
    row = cur.fetchone()
    if not row:
        raise RuntimeError("official upstream server did not return binary identity")
    database_name, server_version, extension_version, vector_path, vector_sha = row
    observed = str(vector_sha).lower()
    if observed != args.official_vector_so_sha256:
        raise RuntimeError(
            "official upstream vector.so SHA-256 mismatch: "
            f"expected {args.official_vector_so_sha256}, got {observed}"
        )
    if str(extension_version) != "0.8.2":
        raise RuntimeError(
            f"official upstream requires pgvector 0.8.2, got {extension_version!r}"
        )
    return {
        "database": str(database_name),
        "postgres_version": str(server_version),
        "vector_extension_version": str(extension_version),
        "vector_so_path": str(vector_path),
        "vector_so_sha256": observed,
        "expected_vector_so_sha256": args.official_vector_so_sha256,
        "vector_so_sha256_matches_expected": True,
        "pgvector_commit": args.official_pgvector_commit,
        "dsn_sha256": dsn_fingerprint(args.official_dsn),
        "identity_method": "pg_read_binary_file(PKGLIBDIR/vector.so)",
    }


def validate_official_contract_args(args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in (
            ("--official-dsn", getattr(args, "official_dsn", "")),
            ("--official-server-container", getattr(args, "official_server_container", "")),
            ("--official-image-digest", getattr(args, "official_image_digest", "")),
            ("--official-pgvector-commit", getattr(args, "official_pgvector_commit", "")),
            ("--official-vector-so-sha256", getattr(args, "official_vector_so_sha256", "")),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "official upstream execution requires an independent DSN/container contract: "
            + ", ".join(missing)
        )
    dsn_fingerprint(args.official_dsn)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Amazon-10M SQL-native PostgreSQL+pgvector hybrid benchmark."
    )
    parser.add_argument(
        "--protocol",
        choices=(exact_truth_contract.PROTOCOL_Q200, P0_PROTOCOL),
        default=exact_truth_contract.PROTOCOL_Q200,
        help="q10200 enables the P0 SQL-native q80/r2 + balanced q10k/r3 protocol",
    )
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument(
        "--execution-engine",
        choices=("sqlens", "official"),
        default="sqlens",
        help="official uses only upstream pgvector SQL/GUCs and never calls SQLens profile functions",
    )
    parser.add_argument(
        "--official-planner-mode",
        choices=("auto", "forced_hnsw"),
        default="auto",
        help="planner policy for --execution-engine official",
    )
    parser.add_argument(
        "--official-dsn",
        default="",
        help="required explicit DSN for the separate upstream pgvector server; never written verbatim",
    )
    parser.add_argument(
        "--official-server-container",
        type=container_name_arg,
        help="Docker container serving --official-dsn; checked against --official-image-digest",
    )
    parser.add_argument(
        "--official-image-digest",
        type=image_digest_arg,
        help="exact Docker image ID/digest of --official-server-container",
    )
    parser.add_argument(
        "--official-pgvector-commit",
        type=git_commit_arg,
        help="exact upstream pgvector source commit used to build the official server",
    )
    parser.add_argument(
        "--official-vector-so-sha256",
        type=expected_sha256_arg,
        help="exact live PKGLIBDIR/vector.so SHA-256 expected on the upstream server",
    )
    parser.add_argument("--schema-sql", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--query-ids-csv", type=Path, default=DEFAULT_QUERY_IDS)
    parser.add_argument(
        "--query-cohort-manifest",
        type=Path,
        default=DEFAULT_QUERY_COHORT_MANIFEST,
        help="provenance manifest for the truth-format formal q200 cohort CSV",
    )
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--exact-truth-csv", type=Path, default=DEFAULT_EXACT_TRUTH_CSV)
    parser.add_argument("--exact-truth-manifest", type=Path, default=DEFAULT_EXACT_TRUTH_MANIFEST)
    parser.add_argument("--vector-table", type=qualified_name_arg, default=DEFAULT_VECTOR_TABLE)
    parser.add_argument(
        "--source-index",
        "--vector-index",
        dest="source_index",
        type=qualified_name_arg,
        default=DEFAULT_SOURCE_INDEX,
    )
    parser.add_argument("--clone-index", type=qualified_name_arg, default=DEFAULT_CLONE_INDEX)
    parser.add_argument("--principal", type=parse_role_name, default=DEFAULT_PRINCIPAL)
    parser.add_argument(
        "--candidate-validity-predicate",
        type=validate_candidate_validity_predicate,
        default=DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
        help="global candidate-universe SQL predicate; formal value is embedding_valid",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS / "amazon10m_sql_native_benchmark.csv")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--plans", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--k", type=positive_int, default=DEFAULT_K)
    parser.add_argument("--calibration-query-offset", type=nonnegative_int, default=DEFAULT_CALIBRATION_QUERY_OFFSET)
    parser.add_argument("--calibration-queries", type=positive_int, default=DEFAULT_CALIBRATION_QUERIES)
    parser.add_argument("--calibration-repeats", type=positive_int, default=DEFAULT_CALIBRATION_REPEATS)
    parser.add_argument("--final-query-offset", type=nonnegative_int, default=100)
    parser.add_argument("--final-queries", type=positive_int, default=DEFAULT_FINAL_QUERIES)
    parser.add_argument("--final-repeats", type=positive_int, default=DEFAULT_FINAL_REPEATS)
    parser.add_argument("--targets", type=parse_targets, default=list(TARGET_RECALLS))
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument("--workload-names", nargs="*", default=[])
    parser.add_argument(
        "--ef-search-values",
        type=parse_int_list,
        default=list(DEFAULT_EF_SEARCH_VALUES),
    )
    parser.add_argument("--max-scan-tuples-values", type=parse_int_list, default=[5_000_000])
    parser.add_argument("--scan-mem-multiplier-values", type=lambda value: parse_float_list(value), default=[32.0])
    parser.add_argument(
        "--iterative-scan-values",
        type=lambda value: parse_word_list(value, {"off", "strict_order", "relaxed_order"}),
        default=["strict_order", "relaxed_order"],
    )
    parser.add_argument("--guided-collect-target-values", type=parse_guided_targets, default=["ef"])
    parser.add_argument(
        "--d3-probe-requests", type=positive_int, default=DEFAULT_D3_PROBE_REQUESTS
    )
    parser.add_argument(
        "--d3-min-benefit-per-byte",
        type=float,
        default=DEFAULT_D3_MIN_BENEFIT_PER_BYTE,
    )
    parser.add_argument(
        "--d3-max-fragment-mb", type=positive_int, default=DEFAULT_D3_MAX_FRAGMENT_MB
    )
    parser.add_argument(
        "--d3-page-min-skip-rate",
        type=float,
        default=DEFAULT_D3_PAGE_MIN_SKIP_RATE,
    )
    parser.add_argument("--bootstrap-samples", type=positive_int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument(
        "--expected-sqlens-build-id",
        type=expected_sqlens_build_id_arg,
        help="exact vector_sqlens_build_id() required by formal --execute",
    )
    parser.add_argument(
        "--expected-vector-so-sha256",
        type=expected_sha256_arg,
        help="exact SHA256 of the serving PostgreSQL PKGLIBDIR/vector.so required by formal --execute",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the run contract without reading files or opening PostgreSQL")
    parser.add_argument("--execute", action="store_true", help="open PostgreSQL and run the benchmark")
    parser.add_argument(
        "--debug-compute-exact-truth",
        action="store_true",
        help="DEBUG ONLY: compute PostgreSQL exact GT instead of requiring the precomputed artifact",
    )
    parser.add_argument("--resume", action="store_true", help="strictly resume a matching atomic checkpoint")
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="q2k/r1 confirmation slice with parallel SQL-first; not paper-eligible",
    )
    parser.add_argument(
        "--screening",
        action="store_true",
        help="q1k/r1 screening slice for retune loops; not paper-eligible",
    )
    parser.add_argument(
        "--sql-first-workers",
        type=positive_int,
        default=1,
        help="parallel connections for SQL-first exact only; stock/SQLens stay sequential",
    )
    parser.add_argument(
        "--reuse-calibration-from",
        type=Path,
        default=None,
        help="copy completed calibration_blocks from another checkpoint into a new run",
    )
    return parser


def resolve_protocol_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.protocol != P0_PROTOCOL:
        return args
    if args.query_ids_csv == DEFAULT_QUERY_IDS:
        args.query_ids_csv = exact_truth_contract.DEFAULT_Q10200_QUERY_IDS
    if args.query_cohort_manifest == DEFAULT_QUERY_COHORT_MANIFEST:
        args.query_cohort_manifest = (
            exact_truth_contract.DEFAULT_Q10200_QUERY_COHORT_MANIFEST
        )
    if args.calibration_query_offset == DEFAULT_CALIBRATION_QUERY_OFFSET:
        args.calibration_query_offset = P0_CALIBRATION_QUERY_OFFSET
    if args.calibration_queries == DEFAULT_CALIBRATION_QUERIES:
        args.calibration_queries = P0_CALIBRATION_QUERIES
    if args.calibration_repeats == DEFAULT_CALIBRATION_REPEATS:
        args.calibration_repeats = P0_CALIBRATION_REPEATS
    if args.final_query_offset == 100:
        args.final_query_offset = P0_MEASUREMENT_QUERY_OFFSET
    if args.final_queries == DEFAULT_FINAL_QUERIES:
        args.final_queries = P0_MEASUREMENT_QUERIES
    if args.final_repeats == DEFAULT_FINAL_REPEATS:
        args.final_repeats = P0_MEASUREMENT_REPEATS
    if [float(value) for value in args.targets] == list(TARGET_RECALLS):
        args.targets = list(P0_TARGET_RECALLS)
    if not args.filter_names:
        args.filter_names = list(P0_FILTER_NAMES)
    if not args.workload_names:
        args.workload_names = list(P0_WORKLOAD_NAMES)
    if args.exact_truth_csv == DEFAULT_EXACT_TRUTH_CSV:
        args.exact_truth_csv = (
            DEFAULT_RESULTS
            / "amazon10m_sql_native_q10200_r43_sqlops_join"
            / "amazon10m_sql_native_exact_truth_q10200.csv"
        )
    if args.exact_truth_manifest == DEFAULT_EXACT_TRUTH_MANIFEST:
        args.exact_truth_manifest = (
            DEFAULT_RESULTS
            / "amazon10m_sql_native_q10200_r43_sqlops_join"
            / "amazon10m_sql_native_exact_truth_manifest.json"
        )
    if args.out == DEFAULT_RESULTS / "amazon10m_sql_native_benchmark.csv":
        args.out = (
            DEFAULT_RESULTS / "amazon10m_sql_native_p0_r43_q10k_r3_sqlops_join.csv"
        )
    if list(args.ef_search_values) == list(DEFAULT_EF_SEARCH_VALUES):
        args.ef_search_values = [
            value for value in args.ef_search_values if value >= P0_MIN_EF_SEARCH
        ]
    if args.expected_sqlens_build_id is None:
        args.expected_sqlens_build_id = SQLENS_R43_BUILD_ID
    if args.expected_vector_so_sha256 is None:
        args.expected_vector_so_sha256 = SQLENS_R43_VECTOR_SO_SHA256
    if args.screening and args.confirmation:
        raise RuntimeError("--screening and --confirmation cannot be combined")
    if args.screening:
        if args.final_queries in {DEFAULT_FINAL_QUERIES, P0_MEASUREMENT_QUERIES}:
            args.final_queries = P0_SCREENING_QUERIES
        if args.final_repeats in {DEFAULT_FINAL_REPEATS, P0_MEASUREMENT_REPEATS}:
            args.final_repeats = P0_SCREENING_REPEATS
        if args.sql_first_workers == 1:
            args.sql_first_workers = P0_CONFIRMATION_SQL_FIRST_WORKERS
        if args.out in {
            DEFAULT_RESULTS / "amazon10m_sql_native_benchmark.csv",
            DEFAULT_RESULTS / "amazon10m_sql_native_p0_r43_q10k_r3.csv",
            DEFAULT_RESULTS / "amazon10m_sql_native_p0_r43_q10k_r3_sqlops.csv",
            DEFAULT_RESULTS / "amazon10m_sql_native_p0_r43_q10k_r3_sqlops_v2.csv",
            DEFAULT_RESULTS / "amazon10m_sql_native_p0_r43_q10k_r3_sqlops_join.csv",
            DEFAULT_RESULTS / "amazon10m_sql_native_p0_r43_q2k_r1_sqlops_v2_confirm.csv",
        }:
            args.out = (
                DEFAULT_RESULTS
                / "amazon10m_sql_native_p0_r43_q1k_r1_sqlops_join_screen.csv"
            )
    elif args.confirmation:
        if args.final_queries in {DEFAULT_FINAL_QUERIES, P0_MEASUREMENT_QUERIES}:
            args.final_queries = P0_CONFIRMATION_QUERIES
        if args.final_repeats in {DEFAULT_FINAL_REPEATS, P0_MEASUREMENT_REPEATS}:
            args.final_repeats = P0_CONFIRMATION_REPEATS
        if args.sql_first_workers == 1:
            args.sql_first_workers = P0_CONFIRMATION_SQL_FIRST_WORKERS
        if args.out in {
            DEFAULT_RESULTS / "amazon10m_sql_native_benchmark.csv",
            DEFAULT_RESULTS / "amazon10m_sql_native_p0_r43_q10k_r3.csv",
            DEFAULT_RESULTS / "amazon10m_sql_native_p0_r43_q10k_r3_sqlops.csv",
            DEFAULT_RESULTS / "amazon10m_sql_native_p0_r43_q10k_r3_sqlops_v2.csv",
            DEFAULT_RESULTS / "amazon10m_sql_native_p0_r43_q10k_r3_sqlops_join.csv",
        }:
            args.out = (
                DEFAULT_RESULTS
                / "amazon10m_sql_native_p0_r43_q2k_r1_sqlops_join_confirm.csv"
            )
    return args


def validate_formal_dimensions(
    args: argparse.Namespace,
    filters: Sequence[FilterSpec],
    workloads: Sequence[WorkloadSpec] | None = None,
    source_filters: Sequence[FilterSpec] | None = None,
) -> None:
    problems: list[str] = []
    workloads = workloads or list(WORKLOADS)
    source_filters = source_filters or filters
    if len(source_filters) != 14 or len({spec.name for spec in source_filters}) != 14:
        problems.append("the source cohort must retain all 14 registered filters")
    if args.protocol == P0_PROTOCOL:
        if (
            args.calibration_query_offset != P0_CALIBRATION_QUERY_OFFSET
            or args.calibration_queries != P0_CALIBRATION_QUERIES
            or args.calibration_repeats != P0_CALIBRATION_REPEATS
        ):
            problems.append("q10200 calibration must be q80/r2 over q20..q99")
        if args.screening:
            if (
                args.final_query_offset != P0_MEASUREMENT_QUERY_OFFSET
                or args.final_queries != P0_SCREENING_QUERIES
                or args.final_repeats != P0_SCREENING_REPEATS
            ):
                problems.append(
                    "q10200 screening must be balanced q1k/r1 over q200..q1199"
                )
        elif args.confirmation:
            if (
                args.final_query_offset != P0_MEASUREMENT_QUERY_OFFSET
                or args.final_queries != P0_CONFIRMATION_QUERIES
                or args.final_repeats != P0_CONFIRMATION_REPEATS
            ):
                problems.append(
                    "q10200 confirmation must be balanced q2k/r1 over q200..q2199"
                )
        elif (
            args.final_query_offset != P0_MEASUREMENT_QUERY_OFFSET
            or args.final_queries != P0_MEASUREMENT_QUERIES
            or args.final_repeats != P0_MEASUREMENT_REPEATS
        ):
            problems.append("q10200 measurement must be balanced q10k/r3 over q200..q10199")
        if tuple(item.name for item in workloads) != P0_WORKLOAD_NAMES:
            problems.append(
                "q10200 mainline requires facts-join, catalog-join, and ACL-join workloads"
            )
        if tuple(item.name for item in filters) != P0_FILTER_NAMES:
            problems.append(
                "q10200 mainline requires the three selective filters "
                "grocery_helpful, helpful_ge20, grocery_long500"
            )
        if min(int(value) for value in args.ef_search_values) < P0_MIN_EF_SEARCH:
            problems.append(
                f"q10200 approximate grid must start at ef>={P0_MIN_EF_SEARCH}"
            )
        if [float(value) for value in args.targets] != list(P0_TARGET_RECALLS):
            problems.append("q10200 mainline target must be Recall@10 0.90")
    else:
        if len(filters) != 14 or len(workloads) != len(WORKLOADS):
            problems.append("q200 compatibility protocol requires the complete matrix")
        if args.calibration_query_offset != DEFAULT_CALIBRATION_QUERY_OFFSET or args.calibration_queries != DEFAULT_CALIBRATION_QUERIES:
            problems.append("calibration must be q80 over q20..q99, reserving q0..q19 for screening")
        if args.final_query_offset != 100 or args.final_queries != 100:
            problems.append("final must be disjoint q100 at offset 100")
        if args.calibration_repeats != 2 or args.final_repeats != 5:
            problems.append("repeats must be calibration r2 and final r5")
        if [float(value) for value in args.targets] != list(TARGET_RECALLS):
            problems.append("matched-recall targets must be 0.90,0.95,0.99")
    if problems:
        raise RuntimeError("formal experiment dimensions are invalid: " + "; ".join(problems))


def print_dry_run(args: argparse.Namespace) -> None:
    workloads = select_workloads(args.workload_names, args.protocol)
    benchmark_modes = P0_MODES if args.protocol == P0_PROTOCOL else MODES
    print("mode=dry-run")
    print(f"protocol={args.protocol}")
    print(f"execution_engine={args.execution_engine}")
    print(f"official_planner_mode={args.official_planner_mode}")
    print("database=not_opened")
    print("execution=single PostgreSQL SELECT with pgvector ORDER BY plus JOIN/ACL/RLS/temporal predicates")
    if args.execution_engine == "official":
        print("modes=official_stock")
        print("sqlens_profile_dependency=false")
        print("guidance_claim=none")
        print("official_upstream_contract=independent_dsn_container_binary")
        print(f"official_dsn_sha256={dsn_fingerprint(args.official_dsn) if args.official_dsn else NA}")
        print(f"official_server_container={args.official_server_container or NA}")
        print(f"official_image_digest={args.official_image_digest or NA}")
        print(f"official_pgvector_commit={args.official_pgvector_commit or NA}")
        print(f"official_vector_so_sha256={args.official_vector_so_sha256 or NA}")
    else:
        print("modes=" + ",".join(benchmark_modes))
        print("guidance_claim=candidate-admission/validation; hard-traversal-pruning=false")
    print("workloads=" + ",".join(item.name for item in workloads))
    if args.protocol == P0_PROTOCOL:
        print(
            "calibration=q80/r2 (q20..q99); "
            "measurement=balanced q10k/r3 (q200..q10199); "
            "q0..q19_and_q100..q199_reserved=true"
        )
    else:
        print("calibration=q80/r2 (q20..q99); final=q100/r5 (q100..q199); q0..q19_reserved=true")
    print("targets=" + ",".join(f"{target:.2f}" for target in args.targets))
    print("bootstrap_samples=" + str(args.bootstrap_samples))
    print(
        "config_grid="
        f"ef{args.ef_search_values};max_scan{args.max_scan_tuples_values};"
        f"mem{args.scan_mem_multiplier_values}"
    )
    print(
        "timing=single PostgreSQL execute+fetchall wall interval; "
        "EXPLAIN, GT, SET/RESET, and output I/O excluded"
        if args.execution_engine == "official"
        else "timing=" + TIMING_DEFINITION
    )
    print(f"filters_csv={args.filters_csv}")
    print(f"schema_sql={args.schema_sql}")
    print(f"query_ids_csv={args.query_ids_csv}")
    print(f"query_cohort_manifest={args.query_cohort_manifest}")
    print(f"candidate_validity_predicate={args.candidate_validity_predicate}")
    print(
        "candidate_universe_predicate_sha256="
        + candidate_universe_predicate_sha256(args.candidate_validity_predicate)
    )
    print(f"exact_truth_csv={args.exact_truth_csv}")
    print(f"exact_truth_manifest={args.exact_truth_manifest}")
    print(f"vector_table={args.vector_table}")
    print(f"source_index={args.source_index}")
    print(f"clone_index={args.clone_index}")
    print(f"expected_sqlens_build_id={args.expected_sqlens_build_id or NA}")
    print(f"expected_vector_so_sha256={args.expected_vector_so_sha256 or NA}")
    print(f"out={args.out}")


def _manifest(
    args: argparse.Namespace,
    filters: Sequence[FilterSpec],
    calibration: dict[int, int],
    final: dict[int, int],
    database: dict[str, Any],
    as_of: dict[str, int],
    query_cohort_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workloads = select_workloads(args.workload_names, args.protocol)
    benchmark_modes = P0_MODES if args.protocol == P0_PROTOCOL else MODES
    combined_queries = {**calibration, **final}
    query_splits = {
        **{query_no: "calibration" for query_no in calibration},
        **{query_no: "final" for query_no in final},
    }
    cohort_hash = query_cohort_sha256(combined_queries, query_splits)
    validity_predicate = validate_candidate_validity_predicate(
        args.candidate_validity_predicate
    )
    data_version = database.get("formal_data_version_proof", {})
    epoch = relation_epoch_contract(data_version.get("start_relations", {}))
    return {
        "artifact_valid": True,
        "protocol": args.protocol,
        "benchmark": "amazon10m_sql_native_hybrid",
        "dataset": "Amazon-10M real SQL-derived Grocery workload",
        "git_revision": git_revision(),
        "runner_sha256": sha256_file(Path(__file__)),
        "schema_sql_sha256": sha256_file(args.schema_sql),
        "filters_csv_sha256": sha256_file(args.filters_csv),
        "query_ids_csv_sha256": sha256_file(args.query_ids_csv),
        "query_cohort_manifest_sha256": sha256_file(args.query_cohort_manifest),
        "database": database,
        "binary_identity_gate": database.get("binary_identity_gate", {}),
        "vector_table": args.vector_table,
        "source_index": args.source_index,
        "clone_index": args.clone_index,
        "principal": args.principal,
        "modes": list(benchmark_modes),
        "mode_semantics": {
            mode: asdict(MODE_SPECS[mode]) for mode in benchmark_modes
        },
        "sqlens_filter_strategy": "safe_guided candidate-admission/validation only",
        "workloads": [asdict(item) for item in workloads],
        "filters": [asdict(item) for item in filters],
        "workload_scalar_predicates": [
            {
                "filter_name": spec.name,
                "predicate": spec.predicate,
                "predicate_sha256": workload_scalar_predicate_sha256(spec.predicate),
            }
            for spec in filters
        ],
        "candidate_universe": {
            "predicate": validity_predicate,
            "predicate_sha256": candidate_universe_predicate_sha256(
                validity_predicate
            ),
            "sql_role": "candidate_relation_only; separate from workload scalar predicate",
        },
        "candidate_universe_predicate_sha256": candidate_universe_predicate_sha256(
            validity_predicate
        ),
        "query_cohort": {
            "query_count": len(combined_queries),
            "query_cohort_sha256": cohort_hash,
            "query_cohort_hash_contract": exact_truth_contract.QUERY_COHORT_HASH_CONTRACT,
            "source_query_cohort": query_cohort_provenance,
        },
        "query_cohort_sha256": cohort_hash,
        "relation_epoch": epoch,
        "calibration": {"queries": len(calibration), "repeats": args.calibration_repeats, "query_nos": list(calibration), "query_ids": list(calibration.values())},
        "final": {"queries": len(final), "repeats": args.final_repeats, "query_nos": list(final), "query_ids": list(final.values())},
        "as_of_by_workload": as_of,
        "sql_hashes": sql_contract_hashes(
            workloads,
            filters,
            args.vector_table,
            args.candidate_validity_predicate,
        ),
        "target_recalls": args.targets,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed, "recall_lcb": "5th percentile of query-level bootstrap means"},
        "execution_order": "interleaved paired by workload/filter/query/repeat",
        "timing_definition": TIMING_DEFINITION,
        "sql_contract": {
            "single_select": True,
            "all_modes_same_sql_text_and_relational_semantics": True,
            "statement_binding": "PARAM_EXTERN query vector; planner proof folds it to Const for safe_guided",
            "marker_semantics": "the approximate SQL binding marker is always true; per-row executor binding and scan proof controls guidance claims",
            "approx_order_by": "vector distance only; no secondary v.id key so HNSW can satisfy the order",
            "exact_gt": "valid AS MATERIALIZED isolates exact vector sorting; B-tree JOIN indexes remain available; ORDER BY distance, id",
            "recall": "distance-threshold tie-aware using PostgreSQL exact kth distance; query row excluded",
        },
        "rls_and_guidance_contract": {
            "facts_policy_always_enforced": True,
            "rls_table": "public.amazon_review_facts",
            "rls_policy": "amazon_review_facts_acl_select",
            "rls_scope": "ACL only; grant/fact temporal predicates remain explicit workload SQL",
            "guidance_scope": "safe candidate-admission/validation superset for row-local predicates on the non-RLS vector heap only",
            "hard_traversal_pruning": False,
            "hard_traversal_equivalence": (
                "intentionally ineligible: normalized JOIN, RLS-derived ACL, and temporal "
                "residuals are executor semantics that the row-local guide cannot prove equivalent"
            ),
            "executor_recheck": ["JOIN", "ACL", "temporal", "RLS"],
            "guidance_never_replaces": ["JOIN", "ACL", "temporal", "RLS"],
            "rls_relation_fingerprints": {
                relation: fingerprint
                for relation, fingerprint in database.get("relations", {}).items()
                if fingerprint.get("rls")
            },
        },
        "d2_graph_contract": database.get("d2_graph_proof", {}),
        "preferred_index_current_settings": database.get(
            "preferred_index_current_settings", {}
        ),
        "d3_contract": {
            "initialization": "workload_driven_empty_cache_no_prebuilt_fragments",
            "prebuilt_fragments": database.get(
                "d3_persistent_fragment_reset", {}
            ).get("prebuilt_fragments", NA),
            "probe_requests": args.d3_probe_requests,
            "min_benefit_per_byte": args.d3_min_benefit_per_byte,
            "max_fragment_mb": args.d3_max_fragment_mb,
            "page_min_skip_rate": args.d3_page_min_skip_rate,
            "startup_reset_evidence": database.get("d3_startup_reset_evidence", {}),
            "persistent_fragment_reset": database.get(
                "d3_persistent_fragment_reset", {}
            ),
            "persistent_fragment_store_end": database.get(
                "d3_fragment_store_end", {}
            ),
            "formal_active_requires": ["probe", "materialize", "admission", "active"],
        },
        "checkpoint_contract": {
            "version": CHECKPOINT_VERSION,
            "exact_granularity": "workload/filter/query",
            "measurement_granularity": "complete q/repeat block",
            "atomic_replace": True,
            "strict_run_spec": True,
        },
    }


def configure_official_pgvector(
    cur: Any, config: Config, planner_mode: str
) -> dict[str, Any]:
    if planner_mode not in {"auto", "forced_hnsw"}:
        raise ValueError(f"unknown official planner mode: {planner_mode}")
    cur.execute(f"SET hnsw.ef_search = {int(config.ef_search)}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(config.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(config.scan_mem_multiplier)}")
    cur.execute(f"SET hnsw.iterative_scan = {config.iterative_scan}")
    cur.execute(
        "RESET enable_seqscan"
        if planner_mode == "auto"
        else "SET enable_seqscan = off"
    )
    settings: dict[str, Any] = {"planner_mode": planner_mode}
    for name in (
        "enable_seqscan",
        "hnsw.ef_search",
        "hnsw.max_scan_tuples",
        "hnsw.scan_mem_multiplier",
        "hnsw.iterative_scan",
    ):
        cur.execute(f"SHOW {name}")
        row = cur.fetchone()
        settings[name] = str(row[0]) if row else ""
    return settings


def official_explain_route(
    cur: Any,
    sql_text: str,
    params: dict[str, Any],
    source_index: str,
    planner_mode: str,
) -> dict[str, Any]:
    cur.execute("EXPLAIN (FORMAT JSON, SETTINGS) " + sql_text, params)
    row = cur.fetchone()
    plan = row[0] if row else []
    indexes = plan_index_names(plan)
    expected = source_index.rsplit(".", 1)[-1]
    uses_hnsw = any(name.lower() == expected.lower() for name in indexes)
    if planner_mode == "forced_hnsw" and not uses_hnsw:
        raise RuntimeError(
            f"forced official HNSW route missing {source_index}: indexes={indexes}"
        )
    return {
        "valid": True,
        "planner_mode": planner_mode,
        "route": "hnsw" if uses_hnsw else "planner_exact_or_other",
        "expected_index": source_index,
        "index_names": indexes,
        "plan": plan,
    }


def run_official_query_block(
    cur: Any,
    *,
    config: Config,
    planner_mode: str,
    workload: WorkloadSpec,
    spec: FilterSpec,
    query_ids: dict[int, int],
    truth: dict[tuple[str, str, int], ExactTruth],
    as_of: int,
    table: str,
    source_index: str,
    principal: str,
    k: int,
    repeats: int,
    phase: str,
    target_recall: float | None,
    schedule_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = configure_official_pgvector(cur, config, planner_mode)
    sql_text = build_hybrid_sql(
        table,
        spec.predicate,
        workload=workload,
        official_compatible=True,
    )
    if any(
        marker in sql_text.lower()
        for marker in (
            "vector_hnsw_guidance",
            "vector_sqlens",
            "guidance_profile",
        )
    ):
        raise RuntimeError("official-compatible SQL contains a SQLens-only symbol")
    ordered_queries = list(query_ids)
    random.Random(schedule_seed).shuffle(ordered_queries)
    rows: list[dict[str, Any]] = []
    params = {
        "query_id": query_ids[ordered_queries[0]],
        "as_of": as_of,
        "k": k,
    }
    set_as_of(cur, as_of)
    plan = official_explain_route(
        cur, sql_text, params, source_index, planner_mode
    )
    for repeat in range(repeats):
        for query_no in ordered_queries:
            query_id = query_ids[query_no]
            result_pairs: list[tuple[int, float]] = []
            error = ""
            elapsed_ms = 0.0
            try:
                configure_official_pgvector(cur, config, planner_mode)
                set_as_of(cur, as_of)
                started = time.perf_counter()
                cur.execute(
                    sql_text,
                    {"query_id": query_id, "as_of": as_of, "k": k},
                )
                result_pairs = [
                    (int(row[0]), float(row[1])) for row in cur.fetchall()
                ]
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
                try:
                    cur.execute("ROLLBACK")
                except Exception:
                    pass
            truth_entry = truth[(workload.name, spec.name, query_no)]
            rows.append(
                {
                    "phase": phase,
                    "target_recall": target_recall if target_recall is not None else "",
                    "workload": workload.name,
                    "workload_width": workload.width,
                    "boolean_predicate": workload.boolean_predicate,
                    "filter_name": spec.name,
                    "predicate": spec.predicate,
                    "mode": "official_stock",
                    "execution_engine": "official",
                    "planner_mode": planner_mode,
                    "config": config.label,
                    "query_no": query_no,
                    "query_id": query_id,
                    "repeat": repeat,
                    "pair_key": f"{workload.name}|{spec.name}|q{query_no}|r{repeat}",
                    "query_sql": sql_text,
                    "query_sql_sha256": hashlib.sha256(sql_text.encode()).hexdigest(),
                    "returned_ids": ",".join(
                        str(row_id) for row_id, _ in result_pairs
                    ),
                    "returned": len(result_pairs),
                    "recall": (
                        tie_aware_recall_at_k(
                            result_pairs, truth_entry, query_id, k
                        )
                        if not error
                        else NA
                    ),
                    "activation_ms": 0.0 if not error else NA,
                    "query_ms": elapsed_ms if not error else NA,
                    "e2e_ms": elapsed_ms if not error else NA,
                    "principal": principal,
                    "as_of": as_of,
                    "error": error,
                }
            )
    return rows, {
        "phase": phase,
        "target_recall": target_recall,
        "workload": workload.name,
        "filter_name": spec.name,
        "mode": "official_stock",
        "config": config.label,
        "settings": settings,
        **plan,
    }


def run_independent_official_upstream_benchmark(args: argparse.Namespace) -> int:
    if not args.execute:
        raise RuntimeError("refusing to open PostgreSQL without --execute")
    validate_official_contract_args(args)
    exact_truth_contract.require_formal_input_hashes(
        args.filters_csv, args.query_ids_csv, args.query_cohort_manifest
    )
    require_psycopg()
    import psycopg

    filters = read_filters(args.filters_csv, set(args.filter_names) or None)
    validate_formal_dimensions(args, filters)
    expected_splits = {
        **{
            query_no: "calibration"
            for query_no in range(
                args.calibration_query_offset,
                args.calibration_query_offset + args.calibration_queries,
            )
        },
        **{
            query_no: "final"
            for query_no in range(
                args.final_query_offset,
                args.final_query_offset + args.final_queries,
            )
        },
    }
    cohort = exact_truth_contract.load_query_cohort(
        args.query_ids_csv,
        expected_splits,
        args.candidate_validity_predicate,
        source_manifest_path=args.query_cohort_manifest,
        expected_filters=filters,
    )
    calibration = {
        query_no: query_id
        for query_no, query_id in cohort.query_ids.items()
        if expected_splits[query_no] == "calibration"
    }
    final = {
        query_no: query_id
        for query_no, query_id in cohort.query_ids.items()
        if expected_splits[query_no] == "final"
    }
    workloads = list(WORKLOADS)
    # This branch must never inherit the SQLens test DSN.  Its DSN, container
    # image and live shared library are independently bound below.
    conninfo = args.official_dsn
    container_identity = inspect_container_image(
        args.official_server_container, args.official_image_digest
    )
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    preregistered_matrix = preregister_official_formal_matrix(
        workloads, filters, args.targets
    )
    calibration_outcomes: dict[tuple[str, str], dict[str, Any]] = {}
    final_summaries: dict[tuple[str, str, float], dict[str, Any]] = {}
    with (
        psycopg.connect(conninfo, autocommit=True) as guard_conn,
        psycopg.connect(conninfo, autocommit=True) as conn,
    ):
        guard_cur = guard_conn.cursor()
        guard = exact_truth_contract.acquire_formal_data_guard(
            guard_cur, args.vector_table
        )
        try:
            cur = conn.cursor()
            upstream_identity = official_runtime_identity(cur, args)
            cur.execute(f'SET ROLE "{args.principal}"')
            relations = {
                relation: exact_truth_contract.relation_fingerprint(
                    guard_cur, relation
                )
                for relation in exact_truth_contract.formal_data_relations(
                    args.vector_table
                )
            }
            guard_cur.execute(
                "SELECT current_database(), current_setting('server_version'), "
                "coalesce((SELECT extversion FROM pg_extension WHERE extname='vector'), '')"
            )
            database_name, server_version, vector_version = guard_cur.fetchone()
            as_of_by_workload: dict[str, int] = {}
            for workload in workloads:
                guard_cur.execute(
                    "SELECT as_of FROM public.amazon_sql_native_buckets "
                    "WHERE principal_name=%s AND target_pct=%s::numeric",
                    (args.principal, str(workload.bucket_pct)),
                )
                row = guard_cur.fetchone()
                if row is None:
                    raise RuntimeError(
                        f"missing SQL-native bucket for {workload.name}"
                    )
                as_of_by_workload[workload.name] = int(row[0])
            truth, truth_provenance = load_external_exact_truth(
                args.exact_truth_csv,
                args.exact_truth_manifest,
                args.fbin,
                args.filters_csv,
                args.query_ids_csv,
                workloads,
                filters,
                {**calibration, **final},
                expected_splits,
                as_of_by_workload,
                args.vector_table,
                args.principal,
                args.k,
                relations,
                candidate_validity_predicate=args.candidate_validity_predicate,
                query_cohort_manifest=args.query_cohort_manifest,
            )
            configs = build_config_grid(args, "stock")
            selected: dict[tuple[str, str, float], Config] = {}
            for workload in workloads:
                for filter_no, spec in enumerate(filters):
                    config_summaries: list[dict[str, Any]] = []
                    for config_no, config in enumerate(configs):
                        rows, plan = run_official_query_block(
                            cur,
                            config=config,
                            planner_mode=args.official_planner_mode,
                            workload=workload,
                            spec=spec,
                            query_ids=calibration,
                            truth=truth,
                            as_of=as_of_by_workload[workload.name],
                            table=args.vector_table,
                            source_index=args.source_index,
                            principal=args.principal,
                            k=args.k,
                            repeats=args.calibration_repeats,
                            phase="calibration",
                            target_recall=None,
                            schedule_seed=args.schedule_seed
                            + filter_no * 1009
                            + config_no,
                        )
                        all_rows.extend(rows)
                        plans.append(plan)
                        stats = summarize_rows(
                            rows,
                            expected_keys=expected_keys_for(
                                [workload],
                                [spec],
                                calibration,
                                args.calibration_repeats,
                            ),
                            bootstrap_samples=args.bootstrap_samples,
                            seed=args.bootstrap_seed + config_no,
                        )
                        stats.update(
                            phase="calibration",
                            mode="official_stock",
                            config=config.label,
                            **asdict(config),
                        )
                        summaries.append(stats)
                        config_summaries.append(stats)
                    outcome = lcb_calibration_outcome(
                        config_summaries,
                        configs,
                        [config.label for config in configs],
                        args.targets,
                    )
                    calibration_outcomes[(workload.name, spec.name)] = outcome
                    for target, winner in outcome["selected"].items():
                        if winner is not None:
                            selected[(workload.name, spec.name, float(target))] = next(
                                config
                                for config in configs
                                if config.label == winner["config"]
                            )
            for (workload_name, filter_name, target), config in selected.items():
                workload = next(
                    item for item in workloads if item.name == workload_name
                )
                spec = next(item for item in filters if item.name == filter_name)
                rows, plan = run_official_query_block(
                    cur,
                    config=config,
                    planner_mode=args.official_planner_mode,
                    workload=workload,
                    spec=spec,
                    query_ids=final,
                    truth=truth,
                    as_of=as_of_by_workload[workload.name],
                    table=args.vector_table,
                    source_index=args.source_index,
                    principal=args.principal,
                    k=args.k,
                    repeats=args.final_repeats,
                    phase="final",
                    target_recall=target,
                    schedule_seed=args.schedule_seed + int(target * 1000),
                )
                all_rows.extend(rows)
                plans.append(plan)
                stats = summarize_rows(
                    rows,
                    expected_keys=expected_keys_for(
                        [workload], [spec], final, args.final_repeats
                    ),
                    target_recall=target,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + int(target * 1000),
                )
                stats.update(
                    phase="final",
                    mode="official_stock",
                    config=config.label,
                    **asdict(config),
                )
                summaries.append(stats)
                final_summaries[(workload_name, filter_name, float(target))] = stats
            data_version = exact_truth_contract.release_formal_data_guard(
                guard_cur, args.vector_table, guard
            )
            guard = None
        finally:
            if guard is not None:
                guard_cur.execute("ROLLBACK")
    formal_matrix = finalize_official_formal_matrix(
        preregistered_matrix, calibration_outcomes, final_summaries
    )
    coverage = formal_matrix_coverage(
        formal_matrix, len(preregistered_matrix)
    )
    plans_cover_predicates = {
        (str(plan.get("workload")), str(plan.get("filter_name")))
        for plan in plans
        if plan.get("mode") == "official_stock" and bool(plan.get("valid"))
    }
    expected_predicates = {
        (workload.name, spec.name) for workload in workloads for spec in filters
    }
    plan_coverage = {
        "expected_predicates": len(expected_predicates),
        "observed_predicates": len(plans_cover_predicates),
        "coverage_complete": plans_cover_predicates == expected_predicates,
        "plans_valid": all(bool(plan.get("valid")) for plan in plans),
        "predicate_keys_sha256": canonical_sha256(sorted(plans_cover_predicates)),
    }
    diagnostic_valid = all(not row.get("error") for row in all_rows)
    artifact_valid = bool(
        diagnostic_valid
        and coverage["coverage_complete"]
        and plan_coverage["coverage_complete"]
        and plan_coverage["plans_valid"]
    )
    paper_eligible = bool(artifact_valid and coverage["all_targets_attained"])
    write_csv(args.out, all_rows)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    write_csv(summary_path, summaries)
    plans_path = args.plans or args.out.with_name(args.out.stem + "_plans.json")
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    atomic_write_json(plans_path, plans)
    manifest = {
        "diagnostic_valid": diagnostic_valid,
        "artifact_valid": artifact_valid,
        "paper_eligible": paper_eligible,
        "status": (
            "paper_eligible"
            if paper_eligible
            else "complete_with_unattainable_targets"
            if artifact_valid
            else "invalid"
        ),
        "benchmark": "amazon10m_sql_native_official_pgvector",
        "execution_engine": "official",
        "comparison_role": "independent_upstream_pgvector",
        "official_compatible": False,
        "sqlens_profile_dependency": False,
        "planner_mode": args.official_planner_mode,
        "filters_csv_sha256": sha256_file(args.filters_csv),
        "query_ids_csv_sha256": sha256_file(args.query_ids_csv),
        "query_cohort_manifest_sha256": sha256_file(
            args.query_cohort_manifest
        ),
        "workloads": [asdict(workload) for workload in workloads],
        "ef_search_values": list(args.ef_search_values),
        "source_index": args.source_index,
        "upstream_contract": {
            "independent_dsn": True,
            "container": container_identity,
            "runtime": upstream_identity,
            "pgvector_commit": args.official_pgvector_commit,
        },
        "database": {
            "database": database_name,
            "postgres_version": server_version,
            "vector_extension_version": vector_version,
            "relations": relations,
            "data_version_proof": data_version,
        },
        "exact_truth": truth_provenance,
        "selected_configs": {
            f"{workload}|{filter_name}|{target:g}": config.label
            for (workload, filter_name, target), config in selected.items()
        },
        "pre_registered_formal_matrix": formal_matrix,
        "formal_matrix_coverage": coverage,
        "predicate_explain_coverage": plan_coverage,
        "outputs": {
            "raw": {"path": str(args.out), "sha256": sha256_file(args.out), "rows": len(all_rows)},
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path), "rows": len(summaries)},
            "plans": {"path": str(plans_path), "sha256": sha256_file(plans_path), "rows": len(plans)},
        },
    }
    atomic_write_json(manifest_path, manifest)
    return 0 if manifest["artifact_valid"] else 2


def run_benchmark(args: argparse.Namespace) -> int:
    if not args.execute:
        raise RuntimeError("refusing to open PostgreSQL without --execute")
    exact_truth_contract.require_formal_input_hashes(
        args.filters_csv,
        args.query_ids_csv,
        args.query_cohort_manifest,
        args.protocol,
    )
    expected_sqlens_build_id, expected_vector_so_sha256 = (
        require_execution_binary_identity(args)
    )
    require_psycopg()
    import psycopg

    if args.d3_min_benefit_per_byte < 0:
        raise ValueError("--d3-min-benefit-per-byte must be nonnegative")
    if not 0.0 <= args.d3_page_min_skip_rate <= 1.0:
        raise ValueError("--d3-page-min-skip-rate must be in [0, 1]")

    source_filters = read_filters(args.filters_csv)
    filters = read_filters(args.filters_csv, set(args.filter_names) or None)
    workloads = select_workloads(args.workload_names, args.protocol)
    validate_formal_dimensions(args, filters, workloads, source_filters)
    benchmark_modes = P0_MODES if args.protocol == P0_PROTOCOL else MODES
    tunable_modes = (
        P0_TUNABLE_MODES if args.protocol == P0_PROTOCOL else MODES
    )
    expected_query_splits = {
        **{
            query_no: "calibration"
            for query_no in range(
                args.calibration_query_offset,
                args.calibration_query_offset + args.calibration_queries,
            )
        },
        **{
            query_no: "final"
            for query_no in range(
                args.final_query_offset,
                args.final_query_offset + args.final_queries,
            )
        },
    }
    source_query_splits = (
        {
            query_no: (
                "calibration" if query_no < 100 else "final"
            )
            for query_no in range(10_200)
        }
        if args.protocol == P0_PROTOCOL
        else expected_query_splits
    )
    query_cohort = exact_truth_contract.load_query_cohort(
        args.query_ids_csv,
        source_query_splits,
        args.candidate_validity_predicate,
        source_manifest_path=args.query_cohort_manifest,
        expected_filters=source_filters,
    )
    calibration = {
        query_no: query_id
        for query_no, query_id in query_cohort.query_ids.items()
        if query_no
        in range(
            args.calibration_query_offset,
            args.calibration_query_offset + args.calibration_queries,
        )
    }
    final = {
        query_no: query_id
        for query_no, query_id in query_cohort.query_ids.items()
        if query_no
        in range(
            args.final_query_offset,
            args.final_query_offset + args.final_queries,
        )
    }
    validate_query_splits(calibration, final)
    preregistered_matrix = preregister_formal_matrix(
        workloads, filters, args.targets, benchmark_modes
    )
    conninfo = pg_config_from_env().conninfo
    checkpoint_path = args.checkpoint or args.out.with_suffix(".checkpoint")
    connections: dict[str, Any] = {}
    sql_first_worker_conns: list[Any] = []
    database: dict[str, Any] = {}
    as_of_by_workload: dict[str, int] = {}
    truth: dict[tuple[str, str, int], ExactTruth] = {}
    summaries: list[dict[str, Any]] = []
    checkpoint: dict[str, Any] | None = None
    fingerprint_cur: Any | None = None
    guard_conn: Any | None = None
    guard_cur: Any | None = None
    formal_guard: dict[str, Any] | None = None
    fragment_conn: Any | None = None
    fragment_cur: Any | None = None
    identity_evidence: list[dict[str, Any]] = []
    new_checkpoint = False
    d3_settings = {
        "probe_requests": args.d3_probe_requests,
        "min_benefit_per_byte": args.d3_min_benefit_per_byte,
        "max_fragment_mb": args.d3_max_fragment_mb,
        "page_min_skip_rate": args.d3_page_min_skip_rate,
    }

    def record_binary_identity(
        cur: Any, stage: str, connection: str
    ) -> dict[str, Any]:
        evidence = observe_serving_binary_identity(
            cur,
            expected_sqlens_build_id,
            expected_vector_so_sha256,
            stage=stage,
            connection=connection,
        )
        evidence["sequence"] = len(identity_evidence) + 1
        identity_evidence.append(evidence)
        if checkpoint is not None and checkpoint_path.is_dir():
            persist_checkpoint_meta(checkpoint_path, checkpoint)
        return require_exact_binary_identity(evidence)

    try:
        fragment_conn = psycopg.connect(conninfo, autocommit=True)
        fragment_cur = fragment_conn.cursor()
        record_binary_identity(fragment_cur, "experiment_start", "fragment_store")
        ensure_sqlens_fragment_catalog(
            fragment_cur,
            args.principal,
            args.vector_table,
            enable_tracking=True,
        )
        query_embeddings = load_query_embeddings(
            fragment_cur,
            args.vector_table,
            list({**calibration, **final}.values()),
            args.candidate_validity_predicate,
        )
        persistent_fragment_reset = clear_fragment_store(
            fragment_cur, args.vector_table
        )
        guard_conn = psycopg.connect(conninfo, autocommit=True)
        guard_cur = guard_conn.cursor()
        record_binary_identity(guard_cur, "connection_open", "data_guard")
        formal_guard = exact_truth_contract.acquire_formal_data_guard(
            guard_cur, args.vector_table
        )
        probe_ids = exact_truth_contract.select_rls_probe_ids(
            guard_cur, args.principal
        )
        session_contexts: dict[str, dict[str, str]] = {}
        security_proofs: dict[str, dict[str, Any]] = {}
        for mode in benchmark_modes:
            conn = psycopg.connect(conninfo, autocommit=True)
            connections[mode] = conn
            cur = conn.cursor()
            record_binary_identity(cur, "connection_open", mode)
            # Superuser-context GUCs must be set before SET ROLE drops privileges.
            cur.execute("SET hnsw.guidance_require_epoch = on")
            ensure_sqlens_fragment_catalog(cur, args.principal, args.vector_table)
            set_heap_competing_indexes_valid(cur, args.vector_table, valid=True)
            cur.execute(f'SET ROLE "{args.principal}"')
            set_preferred_index(
                cur, mode_index(mode, args.source_index, args.clone_index)
            )
            session_contexts[mode] = loaded_session_context(cur)
            if session_contexts[mode]["current_user"] != args.principal:
                raise RuntimeError(f"loaded role mismatch for {mode}: {session_contexts[mode]}")
            security = exact_truth_contract.collect_rls_security_metadata(cur)
            security.update(
                exact_truth_contract.run_rls_visibility_probes(cur, probe_ids)
            )
            security["controlled_probe_ids"] = probe_ids
            security_proofs[mode] = validate_rls_security_proof(
                security, args.principal
            )
            cur.close()
        fingerprint_cur = connections["stock"].cursor()
        relations = [
            args.vector_table,
            args.source_index,
            args.clone_index,
            "public.amazon_review_facts",
            "public.amazon_product_dim",
            "public.amazon_principal_tenant_grants",
            "public.amazon_sql_native_buckets",
        ]
        database = database_fingerprint(fingerprint_cur, relations)
        initial_identity = binary_identity_gate_summary(
            expected_sqlens_build_id,
            expected_vector_so_sha256,
            identity_evidence,
            benchmark_modes,
        )
        database["binary_identity_contract"] = {
            "expected_sqlens_build_id": expected_sqlens_build_id,
            "expected_vector_so_sha256": expected_vector_so_sha256,
            "observed_sqlens_build_id": initial_identity[
                "observed_sqlens_build_id"
            ],
            "observed_vector_so_sha256": initial_identity[
                "observed_vector_so_sha256"
            ],
            "observed_vector_so_paths": initial_identity[
                "observed_vector_so_paths"
            ],
            "all_exact_match": initial_identity["all_exact_match"],
        }
        database["d2_graph_proof"] = graph_clone_proof(
            guard_cur, args.source_index, args.clone_index
        )
        database["d2_index_names"] = [args.source_index, args.clone_index]
        database["benchmark_modes"] = list(benchmark_modes)
        database["principal"] = args.principal
        database["rls_security_proofs"] = security_proofs
        database["formal_data_guard_start"] = formal_guard
        database["query_candidate_universe_proof"] = (
            exact_truth_contract.verify_query_candidate_universe(
                guard_cur,
                args.vector_table,
                {**calibration, **final},
                args.candidate_validity_predicate,
            )
        )
        database["d3_persistent_fragment_reset"] = persistent_fragment_reset
        database["preferred_index_current_settings"] = {
            mode: context["preferred_index_current_setting"]
            for mode, context in session_contexts.items()
        }
        database["mode_indexes"] = {
            mode: mode_index(mode, args.source_index, args.clone_index)
            for mode in benchmark_modes
        }
        database["registered_scalar_indexes"] = collect_registered_scalar_indexes(
            guard_cur
        )
        d3_startup_cur = connections["d1_d2_d3"].cursor()
        try:
            database["d3_startup_reset_evidence"] = reset_adaptive_state(
                d3_startup_cur, persistent_fragment_reset
            )
        finally:
            d3_startup_cur.close()
        fingerprint_cur.execute(
            "SELECT target_pct, as_of FROM public.amazon_sql_native_buckets WHERE principal_name = %s AND target_pct = ANY(%s::numeric[])",
            (args.principal, [item.bucket_pct for item in workloads]),
        )
        as_of = {str(float(row[0])): int(row[1]) for row in fingerprint_cur.fetchall()}
        for workload in workloads:
            key = str(float(workload.bucket_pct))
            if key not in as_of:
                raise RuntimeError(f"missing prepared as_of bucket for {workload.name}: {workload.bucket_pct}")
            as_of_by_workload[workload.name] = as_of[key]

        combined_queries = {**calibration, **final}
        query_splits = {
            **{query_no: "calibration" for query_no in calibration},
            **{query_no: "final" for query_no in final},
        }
        external_truth_provenance: dict[str, str] | None = None
        if not args.debug_compute_exact_truth:
            truth, external_truth_provenance = load_external_exact_truth(
                args.exact_truth_csv,
                args.exact_truth_manifest,
                args.fbin,
                args.filters_csv,
                args.query_ids_csv,
                workloads,
                filters,
                combined_queries,
                query_splits,
                as_of_by_workload,
                args.vector_table,
                args.principal,
                args.k,
                database["relations"],
                candidate_validity_predicate=args.candidate_validity_predicate,
                query_cohort_manifest=args.query_cohort_manifest,
                require_formal_keyspace=args.protocol != P0_PROTOCOL,
            )
        run_spec = build_run_spec(
            args, filters, workloads, calibration, final, database, as_of_by_workload,
            external_truth_provenance,
            query_cohort.provenance,
        )
        if args.resume:
            if not checkpoint_path.is_dir():
                raise RuntimeError(f"resume checkpoint does not exist: {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path)
            validate_checkpoint_run_spec(checkpoint, run_spec)
        else:
            if checkpoint_path.exists():
                raise RuntimeError(
                    f"checkpoint already exists: {checkpoint_path}; use --resume or move it aside"
                )
            checkpoint = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "run_spec": run_spec,
                "run_spec_sha256": canonical_sha256(run_spec),
                "loaded_sessions": [],
                "exact_truth": [],
                "exact_plans": [],
                "calibration_blocks": [],
                "final_blocks": [],
                "invalid_blocks": [],
            }
            new_checkpoint = True
        for field in ("loaded_sessions", "exact_truth", "exact_plans", "calibration_blocks", "final_blocks", "invalid_blocks"):
            if not isinstance(checkpoint.get(field), list):
                raise RuntimeError(f"checkpoint field is incomplete: {field}")
        for loaded_session in checkpoint["loaded_sessions"]:
            if (
                not isinstance(loaded_session, dict)
                or not isinstance(
                    loaded_session.get("binary_identity_gate_evidence"), list
                )
            ):
                raise RuntimeError(
                    "checkpoint loaded session lacks binary identity gate evidence"
                )
        if checkpoint["invalid_blocks"]:
            raise RuntimeError(
                "checkpoint contains a failed formal block; inspect invalid_blocks and restart from a clean checkpoint"
            )
        checkpoint["loaded_sessions"].append(
            {
                "resume": bool(args.resume),
                "connections": session_contexts,
                "binary_identity_gate_evidence": identity_evidence,
            }
        )
        if new_checkpoint:
            initialize_checkpoint(checkpoint_path, checkpoint)
        else:
            persist_checkpoint_meta(checkpoint_path, checkpoint)
        if args.reuse_calibration_from is not None:
            if args.resume:
                raise RuntimeError(
                    "--reuse-calibration-from cannot be combined with --resume"
                )
            source_checkpoint = load_checkpoint(args.reuse_calibration_from)
            reused = 0
            for block in source_checkpoint.get("calibration_blocks", []):
                if not isinstance(block, dict) or "block_id" not in block:
                    raise RuntimeError("reuse calibration block is missing block_id")
                persist_checkpoint_entry(
                    checkpoint_path,
                    "calibration_blocks",
                    str(block["block_id"]),
                    block,
                )
                checkpoint["calibration_blocks"].append(block)
                reused += 1
            if reused == 0:
                raise RuntimeError(
                    f"reuse calibration source has no blocks: {args.reuse_calibration_from}"
                )
            print(
                json.dumps(
                    {
                        "progress": "reuse_calibration",
                        "source": str(args.reuse_calibration_from),
                        "blocks": reused,
                    }
                ),
                flush=True,
            )

        record_binary_identity(guard_cur, "pre_exact_truth", "data_guard")

        table_fingerprint_sha256 = canonical_sha256(database["relations"][args.vector_table])
        if args.debug_compute_exact_truth:
            validate_exact_plans(
                checkpoint["exact_plans"],
                workloads,
                filters,
                args.vector_table,
                args.candidate_validity_predicate,
            )
            truth = restore_exact_truth(
                checkpoint["exact_truth"],
                workloads,
                filters,
                combined_queries,
                query_splits,
                as_of_by_workload,
                args.vector_table,
                table_fingerprint_sha256,
                args.k,
                candidate_validity_predicate=args.candidate_validity_predicate,
            )
        elif checkpoint["exact_truth"] or checkpoint["exact_plans"]:
            raise RuntimeError("checkpoint exact-GT records are incompatible with external exact truth")
        exact_plan_pairs = {
            (plan["workload"], plan["filter_name"]) for plan in checkpoint["exact_plans"]
        }
        if args.debug_compute_exact_truth and any((key[0], key[1]) not in exact_plan_pairs for key in truth):
            raise RuntimeError("checkpoint exact GT is missing its successful non-HNSW EXPLAIN gate")
        if (checkpoint["calibration_blocks"] or checkpoint["final_blocks"]) and len(truth) != len(
            workloads
        ) * len(filters) * len(combined_queries):
            raise RuntimeError("checkpoint has measurement blocks before exact GT is complete")

        exact_plans = checkpoint["exact_plans"] if args.debug_compute_exact_truth else []

        def checkpoint_truth(
            workload: WorkloadSpec,
            spec: FilterSpec,
            query_no: int,
            query_id: int,
            sql_text: str,
            truth_entry: ExactTruth,
        ) -> None:
            record = exact_truth_record(
                workload,
                spec,
                query_no,
                query_id,
                query_splits[query_no],
                as_of_by_workload[workload.name],
                table_fingerprint_sha256,
                sql_text,
                truth_entry,
            )
            plan = next(
                plan
                for plan in exact_plans
                if plan["workload"] == workload.name and plan["filter_name"] == spec.name
            )
            persist_checkpoint_entry(
                checkpoint_path,
                "exact_plans",
                f"{workload.name}|{spec.name}",
                plan,
            )
            persist_checkpoint_entry(
                checkpoint_path,
                "exact_truth",
                f"{workload.name}|{spec.name}|q{query_no}",
                record,
            )
            checkpoint["exact_truth"].append(record)
            checkpoint["exact_plans"] = exact_plans
            print(
                json.dumps(
                    {
                        "progress": "exact_gt",
                        "completed": len(checkpoint["exact_truth"]),
                        "planned": len(workloads) * len(filters) * len(combined_queries),
                        "block": f"{workload.name}/{spec.name}/q{query_no}",
                    }
                ),
                flush=True,
            )

        if args.debug_compute_exact_truth:
            truth = run_exact_truth(
                fingerprint_cur,
                workloads,
                filters,
                combined_queries,
                as_of_by_workload,
                args.vector_table,
                args.source_index,
                args.k,
                exact_plans,
                existing=truth,
                on_truth=checkpoint_truth,
                candidate_validity_predicate=args.candidate_validity_predicate,
            )
            checkpoint["exact_plans"] = exact_plans
            for plan in exact_plans:
                persist_checkpoint_entry(
                    checkpoint_path,
                    "exact_plans",
                    f"{plan['workload']}|{plan['filter_name']}",
                    plan,
                )
            restore_exact_truth(
                checkpoint["exact_truth"],
                workloads,
                filters,
                combined_queries,
                query_splits,
                as_of_by_workload,
                args.vector_table,
                table_fingerprint_sha256,
                args.k,
                require_complete=True,
                candidate_validity_predicate=args.candidate_validity_predicate,
            )

        record_binary_identity(guard_cur, "pre_calibration", "data_guard")
        workload_by_name = {item.name: item for item in workloads}
        filter_by_name = {item.name: item for item in filters}
        grids = {
            mode: build_config_grid(args, mode) for mode in tunable_modes
        }
        config_by_label = {
            mode: {config.label: config for config in configs} for mode, configs in grids.items()
        }
        calibration_blocks: dict[str, dict[str, Any]] = {}
        for block in checkpoint["calibration_blocks"]:
            workload = workload_by_name.get(str(block.get("workload")))
            spec = filter_by_name.get(str(block.get("filter_name")))
            mode = str(block.get("mode"))
            config = config_by_label.get(mode, {}).get(str(block.get("config")))
            if workload is None or spec is None or config is None:
                raise RuntimeError("checkpoint calibration block is stale")
            block_id = calibration_block_id(workload.name, spec.name, mode, config)
            if block.get("block_id") != block_id or block_id in calibration_blocks:
                raise RuntimeError("checkpoint calibration block ID is unexpected/duplicate")
            validate_measurement_block(
                block,
                phase="calibration",
                workload=workload,
                spec=spec,
                query_ids=calibration,
                repeats=args.calibration_repeats,
                modes=(mode,),
                configs={mode: config},
                target_recall=None,
                truth=truth,
                table=args.vector_table,
                principal=args.principal,
                source_index=args.source_index,
                clone_index=args.clone_index,
                candidate_validity_predicate=args.candidate_validity_predicate,
            )
            calibration_blocks[block_id] = block

        def calibration_summaries(
            workload: WorkloadSpec, spec: FilterSpec, mode: str, configs: Sequence[Config]
        ) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            expected = expected_keys_for(
                [workload], [spec], calibration, args.calibration_repeats
            )
            for config in configs:
                block = calibration_blocks.get(
                    calibration_block_id(workload.name, spec.name, mode, config)
                )
                if block is None:
                    continue
                for target in args.targets:
                    summary = summarize_rows(
                        block["rows"],
                        expected_keys=expected,
                        target_recall=target,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.bootstrap_seed,
                    )
                    summary.update(
                        {
                            "phase": "calibration",
                            "mode": mode,
                            "config": config.label,
                            **asdict(config),
                        }
                    )
                    result.append(summary)
            return result

        for workload in workloads:
            for spec in filters:
                for mode in tunable_modes:
                    labels = [config.label for config in grids[mode]]
                    present = [
                        calibration_block_id(workload.name, spec.name, mode, config)
                        in calibration_blocks
                        for config in grids[mode]
                    ]
                    seen_gap = False
                    for exists in present:
                        seen_gap = seen_gap or not exists
                        if exists and seen_gap:
                            raise RuntimeError(
                                "checkpoint calibration blocks are not an ef-ordered prefix"
                            )
                    executed_count = sum(present)
                    for _, group in config_groups(grids[mode]):
                        group_end = labels.index(group[-1].label) + 1
                        if group_end > executed_count:
                            break
                        prefix = grids[mode][:group_end]
                        outcome = calibration_outcome(
                            calibration_summaries(workload, spec, mode, prefix),
                            grids[mode],
                            [config.label for config in prefix],
                            args.targets,
                        )
                        if outcome["stopped"] and executed_count > group_end:
                            raise RuntimeError(
                                "checkpoint calibration continued after highest target was attained"
                            )

        outcomes: dict[tuple[str, str, str], dict[str, Any]] = {}
        for workload in workloads:
            for spec in filters:
                for mode in tunable_modes:
                    executed_configs: list[Config] = []
                    for ef_search, group in config_groups(grids[mode]):
                        for config in group:
                            block_id = calibration_block_id(
                                workload.name, spec.name, mode, config
                            )
                            if block_id not in calibration_blocks:
                                record_binary_identity(
                                    guard_cur,
                                    f"calibration_block:{block_id}",
                                    "data_guard",
                                )
                                block_fragment_reset = (
                                    clear_fragment_store(fragment_cur, args.vector_table)
                                    if mode == "d1_d2_d3"
                                    else persistent_fragment_reset
                                )
                                try:
                                    rows, plans = run_measurements(
                                        connections,
                                        {mode: config},
                                        [workload],
                                        [spec],
                                        calibration,
                                        truth,
                                        as_of_by_workload,
                                        args.vector_table,
                                        args.source_index,
                                        args.clone_index,
                                        args.principal,
                                        args.k,
                                        args.calibration_repeats,
                                        "calibration",
                                        None,
                                        args.schedule_seed,
                                        selected_modes=(mode,),
                                        d3_settings=d3_settings,
                                        fragment_store_reset=block_fragment_reset,
                                        candidate_validity_predicate=args.candidate_validity_predicate,
                                        query_embeddings=query_embeddings,
                                        progress_every=50,
                                    )
                                except BaseException as exc:
                                    persist_checkpoint_entry(
                                        checkpoint_path,
                                        "invalid_blocks",
                                        block_id,
                                        {
                                            "block_id": block_id,
                                            "execution_error": f"{exc.__class__.__name__}: {exc}",
                                        },
                                    )
                                    raise
                                block = {
                                    "block_id": block_id,
                                    "phase": "calibration",
                                    "workload": workload.name,
                                    "filter_name": spec.name,
                                    "mode": mode,
                                    "config": config.label,
                                    "rows": rows,
                                    "plans": plans,
                                }
                                try:
                                    validate_measurement_block(
                                        block,
                                        phase="calibration",
                                        workload=workload,
                                        spec=spec,
                                        query_ids=calibration,
                                        repeats=args.calibration_repeats,
                                        modes=(mode,),
                                        configs={mode: config},
                                        target_recall=None,
                                        truth=truth,
                                        table=args.vector_table,
                                        principal=args.principal,
                                        source_index=args.source_index,
                                        clone_index=args.clone_index,
                                        candidate_validity_predicate=args.candidate_validity_predicate,
                                    )
                                except BaseException as exc:
                                    persist_checkpoint_entry(
                                        checkpoint_path,
                                        "invalid_blocks",
                                        block_id,
                                        {
                                            "block_id": block_id,
                                            "validation_error": f"{exc.__class__.__name__}: {exc}",
                                            "block": block,
                                        },
                                    )
                                    raise
                                persist_checkpoint_entry(
                                    checkpoint_path,
                                    "calibration_blocks",
                                    block_id,
                                    block,
                                )
                                checkpoint["calibration_blocks"].append(block)
                                calibration_blocks[block_id] = block
                                highest_summary = calibration_summaries(
                                    workload, spec, mode, [config]
                                )[-1]
                                print(
                                    json.dumps(
                                        {
                                            "progress": "calibration",
                                            "block": block_id,
                                            "ef_search": ef_search,
                                            "rows": len(rows),
                                            "errors": highest_summary["errors"],
                                            "highest_target_lcb": highest_summary["recall_lcb95"],
                                        }
                                    ),
                                    flush=True,
                                )
                            executed_configs.append(config)
                        pair_summaries = calibration_summaries(
                            workload, spec, mode, executed_configs
                        )
                        outcome = calibration_outcome(
                            pair_summaries,
                            grids[mode],
                            [config.label for config in executed_configs],
                            args.targets,
                        )
                        if outcome["stopped"]:
                            break
                    pair_summaries = calibration_summaries(
                        workload, spec, mode, executed_configs
                    )
                    summaries.extend(pair_summaries)
                    outcomes[(workload.name, spec.name, mode)] = calibration_outcome(
                        pair_summaries,
                        grids[mode],
                        [config.label for config in executed_configs],
                        args.targets,
                    )

        if args.protocol == P0_PROTOCOL:
            target = float(args.targets[0])
            configs_by_cell: dict[tuple[str, str, str], Config] = {}
            for workload in workloads:
                for spec in filters:
                    for mode in tunable_modes:
                        choice = outcomes[
                            (workload.name, spec.name, mode)
                        ]["selected"].get(target)
                        if not isinstance(choice, dict):
                            raise RuntimeError(
                                "P0 matched-recall target is unattainable after the "
                                "complete necessary low-budget grid: "
                                f"{workload.name}/{spec.name}/{mode}/target={target:g}"
                            )
                        configs_by_cell[(workload.name, spec.name, mode)] = (
                            config_by_label[mode][str(choice["config"])]
                        )
                    configs_by_cell[
                        (workload.name, spec.name, SQL_FIRST_MODE)
                    ] = SQL_FIRST_CONFIG

            trace = build_balanced_mixed_trace(
                workloads, filters, final, args.schedule_seed
            )
            trace_hash = mixed_trace_sha256(trace)
            base_configs = {
                mode: next(
                    config
                    for (workload_name, filter_name, config_mode), config
                    in configs_by_cell.items()
                    if config_mode == mode
                )
                for mode in P0_MODES
            }
            p0_blocks: dict[int, dict[str, Any]] = {}
            for block in checkpoint["final_blocks"]:
                if block.get("block_id") != (
                    f"p0_measurement|target{target:.12g}|"
                    f"repeat{int(block.get('repeat', -1))}"
                ):
                    raise RuntimeError("checkpoint P0 measurement block is stale")
                repeat = int(block["repeat"])
                if repeat in p0_blocks or repeat not in range(args.final_repeats):
                    raise RuntimeError(
                        "checkpoint P0 measurement repeat is duplicate/out of range"
                    )
                validate_p0_mixed_block(
                    block, trace, repeat, P0_MODES, configs_by_cell
                )
                p0_blocks[repeat] = block

            record_binary_identity(guard_cur, "pre_final", "data_guard")
            if args.sql_first_workers > 1:
                for worker_no in range(args.sql_first_workers):
                    worker_conn = psycopg.connect(conninfo, autocommit=True)
                    worker_cur = worker_conn.cursor()
                    try:
                        record_binary_identity(
                            worker_cur,
                            "connection_open",
                            f"{SQL_FIRST_MODE}_worker{worker_no}",
                        )
                        prepare_sql_first_session(
                            worker_cur,
                            args.principal,
                            args.vector_table,
                            args.source_index,
                            args.clone_index,
                        )
                    finally:
                        worker_cur.close()
                    sql_first_worker_conns.append(worker_conn)
                print(
                    json.dumps(
                        {
                            "progress": "sql_first_workers",
                            "workers": len(sql_first_worker_conns),
                        }
                    ),
                    flush=True,
                )
            for repeat in range(args.final_repeats):
                if repeat in p0_blocks:
                    continue
                block_id = (
                    f"p0_measurement|target{target:.12g}|repeat{repeat}"
                )
                record_binary_identity(
                    guard_cur, f"measurement_block:{block_id}", "data_guard"
                )
                fragment_reset = clear_fragment_store(
                    fragment_cur, args.vector_table
                )
                try:
                    rows, plans = run_measurements(
                        connections,
                        base_configs,
                        workloads,
                        filters,
                        final,
                        truth,
                        as_of_by_workload,
                        args.vector_table,
                        args.source_index,
                        args.clone_index,
                        args.principal,
                        args.k,
                        1,
                        "measurement",
                        target,
                        args.schedule_seed + repeat,
                        selected_modes=P0_MODES,
                        d3_settings=d3_settings,
                        fragment_store_reset=fragment_reset,
                        candidate_validity_predicate=(
                            args.candidate_validity_predicate
                        ),
                        request_keys=trace_request_keys(trace, repeat),
                        registered_scalar_indexes=database[
                            "registered_scalar_indexes"
                        ],
                        configs_by_cell=configs_by_cell,
                        query_embeddings=query_embeddings,
                        sql_first_workers=args.sql_first_workers,
                        sql_first_connections=sql_first_worker_conns,
                        sql_first_after_sequential=True,
                        progress_every=50,
                    )
                except BaseException as exc:
                    persist_checkpoint_entry(
                        checkpoint_path,
                        "invalid_blocks",
                        block_id,
                        {
                            "block_id": block_id,
                            "execution_error": (
                                f"{exc.__class__.__name__}: {exc}"
                            ),
                        },
                    )
                    raise
                block = {
                    "block_id": block_id,
                    "phase": "measurement",
                    "target_recall": target,
                    "repeat": repeat,
                    "trace_sha256": trace_hash,
                    "fragment_store_reset": fragment_reset,
                    "rows": rows,
                    "plans": plans,
                }
                try:
                    validate_p0_mixed_block(
                        block, trace, repeat, P0_MODES, configs_by_cell
                    )
                except BaseException as exc:
                    persist_checkpoint_entry(
                        checkpoint_path,
                        "invalid_blocks",
                        block_id,
                        {
                            "block_id": block_id,
                            "validation_error": (
                                f"{exc.__class__.__name__}: {exc}"
                            ),
                            "block": block,
                        },
                    )
                    raise
                persist_checkpoint_entry(
                    checkpoint_path, "final_blocks", block_id, block
                )
                checkpoint["final_blocks"].append(block)
                p0_blocks[repeat] = block
                print(
                    json.dumps(
                        {
                            "progress": "p0_measurement",
                            "repeat": repeat,
                            "rows": len(rows),
                            "trace_sha256": trace_hash,
                            "d3_reset_once_for_trace_repeat": True,
                        }
                    ),
                    flush=True,
                )

            measurement_rows = [
                row
                for repeat in range(args.final_repeats)
                for row in p0_blocks[repeat]["rows"]
            ]
            measurement_plans = [
                plan
                for repeat in range(args.final_repeats)
                for plan in p0_blocks[repeat]["plans"]
            ]
            for workload in workloads:
                for spec in filters:
                    query_nos = {
                        int(request["query_no"])
                        for request in trace
                        if request["workload"] == workload.name
                        and request["filter_name"] == spec.name
                    }
                    expected = {
                        (workload.name, spec.name, query_no, repeat)
                        for query_no in query_nos
                        for repeat in range(args.final_repeats)
                    }
                    cell_rows = [
                        row
                        for row in measurement_rows
                        if row["workload"] == workload.name
                        and row["filter_name"] == spec.name
                    ]
                    for mode in P0_MODES:
                        summary = summarize_rows(
                            [row for row in cell_rows if row["mode"] == mode],
                            expected_keys=expected,
                            target_recall=target,
                            bootstrap_samples=args.bootstrap_samples,
                            seed=(
                                args.bootstrap_seed
                                + int(target * 1000)
                                + len(summaries)
                            ),
                            require_adaptive_evidence=False,
                        )
                        config = configs_by_cell[
                            (workload.name, spec.name, mode)
                        ]
                        summary.update(
                            {
                                "phase": "measurement",
                                "mode": mode,
                                "config": config.label,
                                **asdict(config),
                            }
                        )
                        summaries.append(summary)
                    paired = paired_summary(
                        [
                            row
                            for row in cell_rows
                            if row["mode"] == "stock"
                        ],
                        [
                            row
                            for row in cell_rows
                            if row["mode"] == "d1_d2_d3"
                        ],
                        expected_keys=expected,
                        target_recall=target,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.bootstrap_seed + int(target * 1000),
                        method_mode="d1_d2_d3",
                        require_adaptive_evidence=False,
                    )
                    summaries.append(
                        {
                            "phase": "measurement",
                            "mode": "paired_d1_d2_d3",
                            "workload": workload.name,
                            "filter_name": spec.name,
                            "target_recall": target,
                            "status": paired["status"],
                            "paired_queries": paired["paired_queries"],
                            "speedup_vs_stock": paired["speedup_vs_stock"],
                            "speedup_lcb95": paired["speedup_lcb95"],
                            "speedup_ci95_low": paired["speedup_ci95_low"],
                            "speedup_ci95_high": paired["speedup_ci95_high"],
                            "query_latency_definition": TIMING_DEFINITION,
                        }
                    )

            database["d2_graph_proof_end"] = graph_clone_proof(
                guard_cur, args.source_index, args.clone_index
            )
            database["d2_index_fingerprints_end"] = {
                index: relation_fingerprint(guard_cur, index)
                for index in (args.source_index, args.clone_index)
            }
            if database["d2_graph_proof_end"] != database["d2_graph_proof"]:
                raise RuntimeError("D2 graph proof changed during the formal run")
            database["formal_data_version_proof"] = (
                exact_truth_contract.release_formal_data_guard(
                    guard_cur, args.vector_table, formal_guard
                )
            )
            formal_guard = None
            database["d3_fragment_store_end"] = audit_fragment_store(
                fragment_cur, args.vector_table
            )
            record_binary_identity(
                guard_cur, "manifest_finalization", "data_guard"
            )
            retained_identity_evidence = [
                item
                for loaded_session in checkpoint["loaded_sessions"]
                for item in loaded_session["binary_identity_gate_evidence"]
                if isinstance(item, dict)
            ]
            database["binary_identity_gate"] = binary_identity_gate_summary(
                expected_sqlens_build_id,
                expected_vector_so_sha256,
                retained_identity_evidence,
                benchmark_modes,
            )
            database["loaded_sessions"] = checkpoint["loaded_sessions"]
            all_rows = [
                row
                for block in checkpoint["calibration_blocks"]
                + checkpoint["final_blocks"]
                for row in block["rows"]
            ]
            all_plans = list(exact_plans) + [
                plan
                for block in checkpoint["calibration_blocks"]
                + checkpoint["final_blocks"]
                for plan in block["plans"]
            ]
            completion = p0_requested_slice_completion(
                all_rows, trace, P0_MODES, args.final_repeats
            )
            adaptive_by_repeat = [
                {
                    "repeat": repeat,
                    **adaptive_transition_evidence(
                        [
                            row
                            for row in p0_blocks[repeat]["rows"]
                            if row["mode"] == "d1_d2_d3"
                        ]
                    ),
                }
                for repeat in range(args.final_repeats)
            ]
            measurement_summaries = [
                row
                for row in summaries
                if row.get("phase") == "measurement"
            ]
            artifact_errors = database_contract_errors(database)
            if not completion["complete"]:
                artifact_errors.append(
                    "requested "
                    f"q{completion['requested_queries']}/r{args.final_repeats} "
                    "mixed measurement slice is incomplete"
                )
            if any(
                row.get("status") != "complete"
                for row in measurement_summaries
            ):
                artifact_errors.append(
                    "held-out P0 matched-recall measurement failed"
                )
            if any(item.get("valid") is not True for item in adaptive_by_repeat):
                artifact_errors.append(
                    "D3 trace-repeat adaptation/materialization/reuse proof failed"
                )
            manifest = _manifest(
                args,
                filters,
                calibration,
                final,
                database,
                as_of_by_workload,
                query_cohort.provenance,
            )
            manifest.update(
                {
                    "artifact_valid": not artifact_errors,
                    "artifact_errors": artifact_errors,
                    "external_exact_truth": external_truth_provenance,
                    "exact_truth_mode": (
                        "debug_in_database"
                        if args.debug_compute_exact_truth
                        else "external_precomputed"
                    ),
                    "selected_calibration_and_measurement_summaries": summaries,
                    "measurement_trace": {
                        "kind": "balanced_mixed_without_replacement",
                        "queries": len(trace),
                        "cells": len(workloads) * len(filters),
                        "cell_counts": {
                            f"{workload.name}|{spec.name}": sum(
                                request["workload"] == workload.name
                                and request["filter_name"] == spec.name
                                for request in trace
                            )
                            for workload in workloads
                            for spec in filters
                        },
                        "sha256": trace_hash,
                        "requests": trace,
                    },
                    "requested_slice_completion": completion,
                    "d3_trace_repeat_reset_proofs": {
                        str(repeat): p0_blocks[repeat][
                            "fragment_store_reset"
                        ]
                        for repeat in range(args.final_repeats)
                    },
                    "d3_trace_repeat_transition_evidence": adaptive_by_repeat,
                    "calibration_selection_rule": (
                        "lowest mean e2e latency among query-bootstrap "
                        "Recall@10 LCB95-qualified configs; execute every config "
                        "in the qualifying ef_search group before stopping"
                    ),
                    "calibration_execution": {
                        "blocks": len(checkpoint["calibration_blocks"]),
                        "modes": list(tunable_modes),
                    },
                    "measurement_execution": {
                        "blocks": len(p0_blocks),
                        "query_repeat_interleaved": True,
                        "arms": list(P0_MODES),
                        "d3_reset_scope": "once_per_complete_trace_repeat",
                        "sql_first_workers": int(args.sql_first_workers),
                        "confirmation": bool(args.confirmation),
                    },
                }
            )
            status = publish_benchmark_artifacts(
                args,
                all_rows,
                summaries,
                all_plans,
                manifest,
                checkpoint_path,
            )
            print(
                json.dumps(
                    {
                        "rows": len(all_rows),
                        "measurement_rows": len(measurement_rows),
                        "summaries": len(summaries),
                        "artifact_valid": manifest["artifact_valid"],
                        "paper_eligible": manifest.get("paper_eligible", False),
                    },
                    indent=2,
                )
            )
            return status

        common_by_pair: dict[tuple[str, str], list[float]] = {}
        expected_final: dict[str, tuple[WorkloadSpec, FilterSpec, float, dict[str, Config]]] = {}
        for workload in workloads:
            for spec in filters:
                mode_outcomes = [
                    outcomes[(workload.name, spec.name, mode)] for mode in MODES
                ]
                common = common_attainable_targets(mode_outcomes, args.targets)
                common_by_pair[(workload.name, spec.name)] = common
                for target in common:
                    config_map = {
                        mode: config_by_label[mode][outcomes[(workload.name, spec.name, mode)]["selected"][target]["config"]]
                        for mode in MODES
                    }
                    expected_final[final_block_id(workload.name, spec.name, target)] = (
                        workload,
                        spec,
                        target,
                        config_map,
                    )

        record_binary_identity(guard_cur, "pre_final", "data_guard")
        final_blocks: dict[str, dict[str, Any]] = {}
        for block in checkpoint["final_blocks"]:
            block_id = str(block.get("block_id"))
            expected_block = expected_final.get(block_id)
            if expected_block is None or block_id in final_blocks:
                raise RuntimeError("checkpoint final block is stale/unexpected/duplicate")
            workload, spec, target, config_map = expected_block
            if block.get("configs") != {
                mode: config.label for mode, config in config_map.items()
            }:
                raise RuntimeError("checkpoint final block selected configs changed")
            validate_measurement_block(
                block,
                phase="final",
                workload=workload,
                spec=spec,
                query_ids=final,
                repeats=args.final_repeats,
                modes=MODES,
                configs=config_map,
                target_recall=target,
                truth=truth,
                table=args.vector_table,
                principal=args.principal,
                source_index=args.source_index,
                clone_index=args.clone_index,
                candidate_validity_predicate=args.candidate_validity_predicate,
            )
            final_blocks[block_id] = block

        for block_id, (workload, spec, target, config_map) in expected_final.items():
            if block_id not in final_blocks:
                record_binary_identity(
                    guard_cur,
                    f"final_block:{block_id}",
                    "data_guard",
                )
                block_fragment_reset = clear_fragment_store(
                    fragment_cur, args.vector_table
                )
                try:
                    rows, plans = run_measurements(
                        connections,
                        config_map,
                        [workload],
                        [spec],
                        final,
                        truth,
                        as_of_by_workload,
                        args.vector_table,
                        args.source_index,
                        args.clone_index,
                        args.principal,
                        args.k,
                        args.final_repeats,
                        "final",
                        target,
                        args.schedule_seed + int(target * 1000),
                        d3_settings=d3_settings,
                        fragment_store_reset=block_fragment_reset,
                        candidate_validity_predicate=args.candidate_validity_predicate,
                        query_embeddings=query_embeddings,
                    )
                except BaseException as exc:
                    persist_checkpoint_entry(
                        checkpoint_path,
                        "invalid_blocks",
                        block_id,
                        {
                            "block_id": block_id,
                            "execution_error": f"{exc.__class__.__name__}: {exc}",
                        },
                    )
                    raise
                block = {
                    "block_id": block_id,
                    "phase": "final",
                    "workload": workload.name,
                    "filter_name": spec.name,
                    "target_recall": target,
                    "configs": {mode: config.label for mode, config in config_map.items()},
                    "rows": rows,
                    "plans": plans,
                }
                try:
                    validate_measurement_block(
                        block,
                        phase="final",
                        workload=workload,
                        spec=spec,
                        query_ids=final,
                        repeats=args.final_repeats,
                        modes=MODES,
                        configs=config_map,
                        target_recall=target,
                        truth=truth,
                        table=args.vector_table,
                        principal=args.principal,
                        source_index=args.source_index,
                        clone_index=args.clone_index,
                        candidate_validity_predicate=args.candidate_validity_predicate,
                    )
                except BaseException as exc:
                    persist_checkpoint_entry(
                        checkpoint_path,
                        "invalid_blocks",
                        block_id,
                        {
                            "block_id": block_id,
                            "validation_error": f"{exc.__class__.__name__}: {exc}",
                            "block": block,
                        },
                    )
                    raise
                persist_checkpoint_entry(
                    checkpoint_path,
                    "final_blocks",
                    block_id,
                    block,
                )
                checkpoint["final_blocks"].append(block)
                final_blocks[block_id] = block
                print(
                    json.dumps(
                        {
                            "progress": "final",
                            "block": block_id,
                            "rows": len(rows),
                            "execution_order": "query/repeat interleaved across stock and SQLens",
                        }
                    ),
                    flush=True,
                )

        for block_id, (workload, spec, target, config_map) in expected_final.items():
            rows = final_blocks[block_id]["rows"]
            expected = expected_keys_for([workload], [spec], final, args.final_repeats)
            for mode in MODES:
                mode_rows = [row for row in rows if row["mode"] == mode]
                summary = summarize_rows(
                    mode_rows,
                    expected_keys=expected,
                    target_recall=target,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + int(target * 1000),
                )
                summary.update(
                    {
                        "phase": "final",
                        "mode": mode,
                        "config": config_map[mode].label,
                        **asdict(config_map[mode]),
                    }
                )
                summaries.append(summary)
            for method_mode in SQLENS_MODES:
                paired = paired_summary(
                    [row for row in rows if row["mode"] == "stock"],
                    [row for row in rows if row["mode"] == method_mode],
                    expected_keys=expected,
                    target_recall=target,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + int(target * 1000),
                    method_mode=method_mode,
                )
                summaries.append(
                    {
                        "phase": "final",
                        "mode": f"paired_{method_mode}",
                        "workload": workload.name,
                        "filter_name": spec.name,
                        "target_recall": target,
                        "config": (
                            f"stock={config_map['stock'].label};"
                            f"{method_mode}={config_map[method_mode].label}"
                        ),
                        "status": paired["status"],
                        "paired_queries": paired["paired_queries"],
                        "speedup_vs_stock": paired["speedup_vs_stock"],
                        "speedup_lcb95": paired["speedup_lcb95"],
                        "speedup_ci95_low": paired["speedup_ci95_low"],
                        "speedup_ci95_high": paired["speedup_ci95_high"],
                        "paired_latency_saving_mean_ms": paired[
                            "paired_latency_saving_mean_ms"
                        ],
                        "paired_latency_saving_ci95_low_ms": paired[
                            "paired_latency_saving_ci95_low_ms"
                        ],
                        "paired_latency_saving_ci95_high_ms": paired[
                            "paired_latency_saving_ci95_high_ms"
                        ],
                        "stock_recall_lcb95": paired["stock"]["recall_lcb95"],
                        "method_recall_lcb95": paired[method_mode]["recall_lcb95"],
                        "query_latency_definition": TIMING_DEFINITION,
                    }
                )

        pair_execution: list[dict[str, Any]] = []
        for workload in workloads:
            for spec in filters:
                for mode in MODES:
                    outcome = outcomes[(workload.name, spec.name, mode)]
                    pair_execution.append(
                        {
                            "workload": workload.name,
                            "filter_name": spec.name,
                            "mode": mode,
                            **{key: value for key, value in outcome.items() if key != "selected"},
                            "selected_configs": {
                                str(target): choice["config"] if choice else None
                                for target, choice in outcome["selected"].items()
                            },
                        }
                    )
        completed_final = {
            (workload.name, spec.name, float(target))
            for workload, spec, target, _ in expected_final.values()
        }
        formal_matrix = finalize_formal_matrix(
            preregistered_matrix, outcomes, completed_final
        )
        database["d2_graph_proof_end"] = graph_clone_proof(
            guard_cur, args.source_index, args.clone_index
        )
        database["d2_index_fingerprints_end"] = {
            index: relation_fingerprint(guard_cur, index)
            for index in (args.source_index, args.clone_index)
        }
        if database["d2_graph_proof_end"] != database["d2_graph_proof"]:
            raise RuntimeError("D2 graph proof changed during the formal run")
        database["formal_data_version_proof"] = (
            exact_truth_contract.release_formal_data_guard(
                guard_cur, args.vector_table, formal_guard
            )
        )
        formal_guard = None
        database["d3_fragment_store_end"] = audit_fragment_store(
            fragment_cur, args.vector_table
        )
        record_binary_identity(
            guard_cur, "manifest_finalization", "data_guard"
        )
        retained_identity_evidence = [
            item
            for loaded_session in checkpoint["loaded_sessions"]
            for item in loaded_session["binary_identity_gate_evidence"]
            if isinstance(item, dict)
        ]
        database["binary_identity_gate"] = binary_identity_gate_summary(
            expected_sqlens_build_id,
            expected_vector_so_sha256,
            retained_identity_evidence,
            benchmark_modes,
        )
        database["loaded_sessions"] = checkpoint["loaded_sessions"]
        all_rows = [
            row
            for block in checkpoint["calibration_blocks"] + checkpoint["final_blocks"]
            for row in block["rows"]
        ]
        all_plans = list(exact_plans) + [
            plan
            for block in checkpoint["calibration_blocks"] + checkpoint["final_blocks"]
            for plan in block["plans"]
        ]
        manifest = _manifest(
            args,
            filters,
            calibration,
            final,
            database,
            as_of_by_workload,
            query_cohort.provenance,
        )
        artifact_errors = artifact_validation_errors(
            len(expected_final), summaries, all_rows, all_plans, formal_matrix
        )
        artifact_errors.extend(database_contract_errors(database))
        manifest["artifact_valid"] = not artifact_errors
        manifest["artifact_errors"] = artifact_errors
        manifest.update(
            {
                "external_exact_truth": external_truth_provenance,
                "exact_truth_mode": "debug_in_database" if args.debug_compute_exact_truth else "external_precomputed",
                "selected_calibration_and_final_summaries": summaries,
                "d3_measured_transition_evidence": grouped_adaptive_transition_evidence(
                    all_rows
                ),
                "d3_block_fragment_reset_proofs": {
                    canonical_sha256(row["persistent_fragment_reset_proof"]): row[
                        "persistent_fragment_reset_proof"
                    ]
                    for row in all_rows
                    if row.get("mode") == "d1_d2_d3"
                    and isinstance(row.get("persistent_fragment_reset_proof"), dict)
                },
                "truth_pairs": len(truth),
                "calibration_execution": {
                    "planned_blocks": sum(item["planned_blocks"] for item in pair_execution),
                    "executed_blocks": len(checkpoint["calibration_blocks"]),
                    "stopped_pairs": sum(bool(item["stopped"]) for item in pair_execution),
                    "grid_exhausted_pairs": sum(
                        bool(item["grid_exhausted"]) for item in pair_execution
                    ),
                    "pairs": pair_execution,
                },
                "final_execution": {
                    "planned_blocks": len(expected_final),
                    "executed_blocks": len(checkpoint["final_blocks"]),
                    "query_repeat_interleaved": True,
                    "paired_ci": True,
                },
                "common_attainable": [
                    {
                        "workload": workload,
                        "filter_name": filter_name,
                        "targets": targets,
                    }
                    for (workload, filter_name), targets in common_by_pair.items()
                ],
                "pre_registered_formal_matrix": formal_matrix,
            }
        )
        manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
        plans_path = args.plans or args.out.with_suffix(".plans.json")
        status = publish_benchmark_artifacts(
            args, all_rows, summaries, all_plans, manifest, checkpoint_path
        )
        print(
            json.dumps(
                {
                    "rows": len(all_rows),
                    "summaries": len(summaries),
                    "manifest": str(manifest_path),
                    "plans": str(plans_path),
                    "checkpoint_removed": status == 0,
                    "artifact_valid": manifest["artifact_valid"],
                },
                indent=2,
            )
        )
        return status
    finally:
        if formal_guard is not None and guard_cur is not None:
            try:
                guard_cur.execute("ROLLBACK")
            except Exception:
                pass
        if fingerprint_cur is not None:
            fingerprint_cur.close()
        for conn in connections.values():
            conn.close()
        for conn in sql_first_worker_conns:
            try:
                conn.close()
            except Exception:
                pass
        if guard_cur is not None:
            guard_cur.close()
        if guard_conn is not None:
            guard_conn.close()
        if fragment_cur is not None:
            fragment_cur.close()
        if fragment_conn is not None:
            fragment_conn.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = resolve_protocol_args(create_argument_parser().parse_args(argv))
    if args.dry_run or not args.execute:
        print_dry_run(args)
        return 0
    if args.execution_engine == "official":
        return run_independent_official_upstream_benchmark(args)
    require_execution_binary_identity(args)
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
