from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

try:
    from .common_pg import pg_config_from_env
    from .pgvector_target_recall_selectivity_runner import (
        DEFAULT_P0_RELEASE_CONTRACT,
        load_p0_release_contract,
    )
except ImportError:
    from common_pg import pg_config_from_env
    from pgvector_target_recall_selectivity_runner import (
        DEFAULT_P0_RELEASE_CONTRACT,
        load_p0_release_contract,
    )


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "hybrid_vector_db"
TARGET_RUNNER = (
    ROOT
    / "experiments"
    / "hybrid_vector_db"
    / "scripts"
    / "pgvector_target_recall_selectivity_runner.py"
)
INDEPENDENT_AUDITOR = (
    ROOT
    / "experiments"
    / "hybrid_vector_db"
    / "scripts"
    / "audit_sigmod_matched_recall_artifact.py"
)
OOD_ANNS_DATA = Path(
    os.environ.get("OOD_ANNS_DATA", ROOT / "data" / "OOD-ANNS")
)
TARGET_RECALLS = (0.90, 0.95, 0.99)
EF_SEARCH_GRID = (
    20,
    40,
    60,
    80,
    100,
    150,
    200,
    250,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    4000,
    5000,
    7000,
    8500,
    10000,
    20000,
    50000,
    100000,
)
CALIBRATION_QUERIES = 80
CALIBRATION_REPEATS = 2
FINAL_QUERIES = 100
FINAL_REPEATS = 5
FINAL_QUERY_OFFSET = CALIBRATION_QUERIES
MODES = ("original", "design1_bloom")
ITERATIVE_SCAN_VALUES = "off,strict_order"
CALIBRATION_GRID_POLICY = "complete_base_grid_target_gated_high_recall_extension"
BASE_EF_SEARCH_GRID = tuple(value for value in EF_SEARCH_GRID if value <= 10_000)
HIGH_EF_SEARCH_EXTENSION = tuple(value for value in EF_SEARCH_GRID if value > 10_000)
EXPECTED_FILTERS = 14
EXPECTED_CELLS = EXPECTED_FILTERS * len(TARGET_RECALLS) * len(MODES)
FORMAL_SQLENS_BUILD_PREFIX = (
    "sqlens-v16-d3-representation-preserving-exact-d2-edge-trace-"
)
FORMAL_SQLENS_BUILD_PREFIXES = (
    FORMAL_SQLENS_BUILD_PREFIX,
    "sqlens-v16-d3-full-materialization-persisted-reuse-",
)
TRUTH_REQUIRED_FIELDS = {
    "query_no",
    "query_id",
    "query_split",
    "filter_name",
    "predicate",
    "candidate_validity_predicate",
    "method",
    "filtered_rows",
    "kth_distance_sq",
    "tie_tolerance",
    "self_excluded",
}


def formal_sqlens_build_compatible(build_id: object) -> bool:
    return str(build_id or "").startswith(FORMAL_SQLENS_BUILD_PREFIXES)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    table: str
    query_table: str
    index: str
    guidance_meta_table: str
    query_id_column: str
    query_vector_column: str
    filter_names: tuple[str, ...]
    default_filters_csv: Path
    default_truth_csv: Path
    truth_builder_command: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def load_and_audit_filters(path: Path) -> tuple[list[str], dict[str, dict[str, str]], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return [], {}, [f"missing filters CSV: {path}"]
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "filter_name",
            "target_rate",
            "expected_rows",
            "predicate",
            "atoms",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            return [], {}, [f"filters CSV missing fields: {sorted(missing)}"]
        rows = list(reader)
    names: list[str] = []
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row.get("filter_name") or "").strip()
        predicate = str(row.get("predicate") or "").strip()
        atoms = [part.strip() for part in str(row.get("atoms") or "").split("||") if part.strip()]
        if not name:
            errors.append("filters CSV contains an empty filter_name")
            continue
        if name in by_name:
            errors.append(f"duplicate filter_name: {name}")
            continue
        if not predicate or any(token in predicate for token in (";", "--", "/*", "*/")):
            errors.append(f"invalid predicate for {name}")
        if not atoms:
            errors.append(f"empty atoms for {name}")
        elif "&&" in predicate and len(atoms) > 1:
            expected_or_positions = [
                token.upper() == "OR"
                for token in atoms
            ]
            if any(
                is_or != (position % 2 == 1)
                for position, is_or in enumerate(expected_or_positions)
            ):
                errors.append(
                    f"overlap predicate {name} must encode singleton atoms as atom||OR||atom"
                )
        try:
            float(str(row.get("target_rate") or "").replace("%", ""))
        except ValueError:
            errors.append(f"invalid target_rate for {name}")
        names.append(name)
        by_name[name] = row
    if len(names) != EXPECTED_FILTERS:
        errors.append(f"expected {EXPECTED_FILTERS} filters, found {len(names)}")
    return names, by_name, errors


def audit_truth(
    path: Path,
    filter_names: list[str],
    filters_by_name: dict[str, dict[str, str]],
) -> dict[str, Any]:
    errors: list[str] = []
    if not path.is_file():
        return {"path": str(path), "ready": False, "errors": [f"missing truth CSV: {path}"]}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = TRUTH_REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing:
            return {
                "path": str(path),
                "ready": False,
                "errors": [f"truth CSV missing fields: {sorted(missing)}"],
            }
        rows = list(reader)

    expected_query_nos = set(range(CALIBRATION_QUERIES + FINAL_QUERIES))
    expected_keys = {(name, query_no) for name in filter_names for query_no in expected_query_nos}
    seen_keys: set[tuple[str, int]] = set()
    query_ids: dict[int, int] = {}
    calibration_rows = 0
    final_rows = 0
    for row_no, row in enumerate(rows, start=2):
        try:
            query_no = int(row["query_no"])
            query_id = int(row["query_id"])
            filtered_rows = int(row["filtered_rows"])
            kth_distance_sq = float(row["kth_distance_sq"])
            tie_tolerance = float(row["tie_tolerance"])
        except (TypeError, ValueError) as exc:
            errors.append(f"truth row {row_no} has invalid numeric fields: {exc}")
            continue
        name = row["filter_name"]
        key = (name, query_no)
        if key in seen_keys:
            errors.append(f"duplicate truth key: {key}")
        seen_keys.add(key)
        previous = query_ids.setdefault(query_no, query_id)
        if previous != query_id:
            errors.append(f"query_no={query_no} maps to multiple query IDs")
        expected_split = "calibration" if query_no < FINAL_QUERY_OFFSET else "final"
        if row["query_split"] != expected_split:
            errors.append(f"truth key {key} has query_split={row['query_split']!r}, expected {expected_split!r}")
        calibration_rows += int(expected_split == "calibration")
        final_rows += int(expected_split == "final")
        if row["method"] != "pre_filter_exact":
            errors.append(f"truth key {key} is not pre_filter_exact")
        if parse_bool(row["self_excluded"]):
            errors.append(f"truth key {key} incorrectly excludes the external query ID")
        if str(row["candidate_validity_predicate"]).strip().upper() != "TRUE":
            errors.append(f"truth key {key} does not use candidate_validity_predicate=TRUE")
        if filtered_rows < 10 or kth_distance_sq < 0 or tie_tolerance < 0:
            errors.append(f"truth key {key} has invalid tie-aware boundary fields")
        filter_row = filters_by_name.get(name)
        if filter_row is None:
            errors.append(f"truth key {key} references unknown filter")
        else:
            if str(row["predicate"]).strip() != str(filter_row["predicate"]).strip():
                errors.append(f"truth key {key} predicate differs from filters CSV")
            try:
                expected_rows = int(filter_row["expected_rows"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"filter {name} lacks a valid expected_rows value")
            else:
                if filtered_rows != expected_rows:
                    errors.append(
                        f"truth key {key} filtered_rows={filtered_rows}, expected {expected_rows}"
                    )

    missing_keys = expected_keys - seen_keys
    extra_keys = seen_keys - expected_keys
    if missing_keys:
        errors.append(f"truth matrix is missing {len(missing_keys)} filter/query cells")
    if extra_keys:
        errors.append(f"truth matrix has {len(extra_keys)} unexpected filter/query cells")
    if set(query_ids) != expected_query_nos:
        errors.append("truth query_no domain is not exactly 0..179")
    expected_calibration_rows = EXPECTED_FILTERS * CALIBRATION_QUERIES
    expected_final_rows = EXPECTED_FILTERS * FINAL_QUERIES
    if calibration_rows != expected_calibration_rows or final_rows != expected_final_rows:
        errors.append(
            "truth split cardinality mismatch: "
            f"calibration={calibration_rows}/{expected_calibration_rows}, "
            f"final={final_rows}/{expected_final_rows}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "query_count": len(query_ids),
        "calibration_queries": CALIBRATION_QUERIES,
        "final_queries": FINAL_QUERIES,
        "ready": not errors,
        "errors": errors[:50],
    }


def database_readiness(spec: DatasetSpec, filters_by_name: dict[str, dict[str, str]]) -> dict[str, Any]:
    checks: dict[str, Any] = {"ready": False, "errors": []}
    try:
        with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            relations: dict[str, dict[str, Any]] = {}
            for relation in (spec.table, spec.query_table, spec.guidance_meta_table):
                cur.execute(
                    "SELECT c.oid, c.reltuples::bigint FROM pg_class c WHERE c.oid = %s::regclass",
                    (relation,),
                )
                row = cur.fetchone()
                if row is None:
                    checks["errors"].append(f"missing relation: {relation}")
                else:
                    relations[relation] = {"oid": int(row[0]), "estimated_rows": int(row[1])}
            cur.execute(
                "SELECT i.indexdef FROM pg_indexes i WHERE schemaname = split_part(%s, '.', 1) "
                "AND indexname = split_part(%s, '.', 2)",
                (spec.index, spec.index),
            )
            index_row = cur.fetchone()
            if index_row is None:
                checks["errors"].append(f"missing HNSW index: {spec.index}")
                index_definition = None
            else:
                index_definition = str(index_row[0])
                if " USING hnsw " not in index_definition or " WHERE " in index_definition:
                    checks["errors"].append(f"expected a full HNSW index: {spec.index}")
            cur.execute(f"SELECT count(*) FROM {spec.query_table}")
            query_rows = int(cur.fetchone()[0])
            if query_rows < CALIBRATION_QUERIES + FINAL_QUERIES:
                checks["errors"].append(
                    f"query table has {query_rows} rows; at least 180 are required"
                )
            for name, row in filters_by_name.items():
                try:
                    cur.execute(f"EXPLAIN SELECT 1 FROM {spec.table} WHERE ({row['predicate']}) LIMIT 0")
                    cur.fetchall()
                    cur.execute(f"SELECT count(*) FROM {spec.table} WHERE ({row['predicate']})")
                    observed_rows = int(cur.fetchone()[0])
                    expected_rows = int(row["expected_rows"])
                    if observed_rows != expected_rows:
                        checks["errors"].append(
                            f"filter {name} row count differs: PostgreSQL={observed_rows}, "
                            f"filters CSV={expected_rows}"
                        )
                except Exception as exc:
                    checks["errors"].append(f"predicate/count audit failed for {name}: {exc}")
                    conn.rollback()
            try:
                cur.execute(
                    "WITH lib AS ("
                    "SELECT setting || '/vector.so' AS path "
                    "FROM pg_config WHERE name = 'PKGLIBDIR'"
                    ") SELECT vector_sqlens_build_id(), path, "
                    "encode(sha256(pg_read_binary_file(path)), 'hex') FROM lib"
                )
                runtime_row = cur.fetchone()
                build_id = str(runtime_row[0] if runtime_row else "")
                vector_so_path = str(runtime_row[1] if runtime_row else "")
                vector_so_sha256 = str(runtime_row[2] if runtime_row else "")
                if not formal_sqlens_build_compatible(build_id):
                    checks["errors"].append(
                        "SQLens runtime build is not the formal release family: "
                        f"observed={build_id!r} "
                        f"required_prefixes={FORMAL_SQLENS_BUILD_PREFIXES!r}"
                    )
                if (
                    not vector_so_path.endswith("/vector.so")
                    or len(vector_so_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in vector_so_sha256
                    )
                ):
                    checks["errors"].append(
                        "loaded vector.so path/SHA256 is unavailable or malformed"
                    )
            except Exception as exc:
                build_id = None
                vector_so_path = None
                vector_so_sha256 = None
                checks["errors"].append(f"SQLens runtime identity unavailable: {exc}")
            checks.update(
                {
                    "relations": relations,
                    "query_rows": query_rows,
                    "index": spec.index,
                    "index_definition": index_definition,
                    "sqlens_build_id": build_id,
                    "vector_so_path": vector_so_path,
                    "vector_so_sha256": vector_so_sha256,
                    "required_sqlens_build_prefix": FORMAL_SQLENS_BUILD_PREFIX,
                    "required_sqlens_build_prefixes": list(
                        FORMAL_SQLENS_BUILD_PREFIXES
                    ),
                }
            )
    except Exception as exc:
        checks["errors"].append(f"database readiness failed: {exc}")
    checks["ready"] = not checks["errors"]
    return checks


def build_target_command(
    spec: DatasetSpec,
    args: argparse.Namespace,
    filter_names: list[str],
) -> list[str]:
    expected_build_id = str(
        getattr(args, "expected_sqlens_build_id", "") or ""
    )
    expected_vector_sha = str(
        getattr(args, "expected_vector_so_sha256", "") or ""
    )
    contract = load_p0_release_contract(
        Path(getattr(args, "release_contract", DEFAULT_P0_RELEASE_CONTRACT))
    )
    if not expected_build_id or len(expected_vector_sha) != 64:
        raise ValueError(
            "external formal command requires an exact SQLens build ID/vector.so SHA256 binding"
        )
    if (
        expected_build_id != contract["expected_sqlens_build_id"]
        or expected_vector_sha != contract["expected_vector_so_sha256"]
    ):
        raise ValueError("external formal command binding differs from the immutable P0 release contract")
    full_tag = f"{spec.key}_{args.tag}"
    cmd = [
        sys.executable,
        str(TARGET_RUNNER),
        "--tag",
        full_tag,
        "--target-recalls",
        ",".join(f"{value:.2f}" for value in TARGET_RECALLS),
        "--calibration-recall-margin",
        str(args.calibration_recall_margin),
        "--calibration-selection-policy",
        getattr(args, "calibration_selection_policy", "lcb_then_max_recall"),
        "--filters",
        *filter_names,
        "--modes",
        *MODES,
        "--calibration-queries",
        str(CALIBRATION_QUERIES),
        "--calibration-repeats",
        str(CALIBRATION_REPEATS),
        "--calibration-query-offset",
        "0",
        "--final-queries",
        str(FINAL_QUERIES),
        "--final-repeats",
        str(FINAL_REPEATS),
        "--final-query-offset",
        str(FINAL_QUERY_OFFSET),
        "--final-execution-order",
        "interleaved",
        "--schedule-seed",
        str(args.schedule_seed),
        "--ef-search-values",
        ",".join(str(value) for value in EF_SEARCH_GRID),
        "--guided-collect-target-values",
        "1",
        "--traversal-guided-target-values",
        "11",
        "--max-scan-tuples-values",
        "5000000",
        "--scan-mem-multiplier-values",
        "32",
        "--iterative-scan-values",
        ITERATIVE_SCAN_VALUES,
        "--stock-iterative-scan-values",
        ITERATIVE_SCAN_VALUES,
        "--filters-csv",
        str(args.filters_csv),
        "--truth-csv",
        str(args.truth_csv),
        "--insertion-table",
        spec.table,
        "--insertion-index",
        spec.index,
        "--bfs-table",
        spec.table,
        "--bfs-index",
        spec.index,
        "--query-table",
        spec.query_table,
        "--query-id-column",
        spec.query_id_column,
        "--query-vector-column",
        spec.query_vector_column,
        "--candidate-validity-predicate",
        "TRUE",
        "--no-expected-truth-self-excluded",
        "--guidance-filter-strategy",
        "safe_guided",
        "--no-traversal-guided-prioritization",
        "--guidance-selectivity-max-pct",
        "100",
        "--guidance-max-atoms",
        str(args.guidance_max_atoms),
        "--index-health-ef-search",
        "10000",
        "--expected-sqlens-build-id",
        expected_build_id,
        "--expected-vector-so-sha256",
        expected_vector_sha,
        "--release-contract",
        str(getattr(args, "release_contract", DEFAULT_P0_RELEASE_CONTRACT)),
        "--statement-timeout-ms",
        str(args.statement_timeout_ms),
        "--progress-queries",
        str(args.progress_queries),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--warmup-all-queries",
        "--force-hnsw",
        "--require-preferred-index-guc",
        (
            "--prewarm-index-health"
            if args.prewarm_index_health
            else "--no-prewarm-index-health"
        ),
        "--resume" if args.resume else "--no-resume",
    ]
    if args.backend_cpu_list:
        cmd.extend(["--backend-cpu-list", args.backend_cpu_list])
    if getattr(args, "reuse_calibration_manifest", None):
        cmd.extend(
            [
                "--reuse-calibration-manifest",
                str(args.reuse_calibration_manifest),
            ]
        )
    return cmd


def truth_builder_command(spec: DatasetSpec, args: argparse.Namespace) -> list[str]:
    replacements = {
        "{python}": sys.executable,
        "{filters_csv}": str(args.filters_csv),
        "{truth_csv}": str(args.truth_csv),
        "{ood_anns_data}": str(OOD_ANNS_DATA),
    }
    command: list[str] = []
    for part in spec.truth_builder_command:
        rendered = part
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)
        command.append(rendered)
    return command


def audit_generic_manifest(path: Path, spec: DatasetSpec, args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    contract = load_p0_release_contract(
        Path(getattr(args, "release_contract", DEFAULT_P0_RELEASE_CONTRACT))
    )
    selection_policy = getattr(
        args,
        "calibration_selection_policy",
        "lcb_then_max_recall",
    )
    run_args = (payload.get("run_spec") or {}).get("args") or {}
    bound_build_id = str(
        getattr(args, "expected_sqlens_build_id", "")
        or run_args.get("expected_sqlens_build_id")
        or ""
    )
    bound_vector_sha = str(
        getattr(args, "expected_vector_so_sha256", "")
        or run_args.get("expected_vector_so_sha256")
        or ""
    )
    if (
        bound_build_id != contract["expected_sqlens_build_id"]
        or bound_vector_sha != contract["expected_vector_so_sha256"]
    ):
        errors.append("formal manifest binding differs from the immutable P0 release contract")
    expected = {
        "calibration_queries": CALIBRATION_QUERIES,
        "calibration_repeats": CALIBRATION_REPEATS,
        "calibration_query_offset": 0,
        "final_queries": FINAL_QUERIES,
        "final_repeats": FINAL_REPEATS,
        "final_query_offset": FINAL_QUERY_OFFSET,
        "final_execution_order": "interleaved",
        "calibration_selection_policy": selection_policy,
        "candidate_validity_predicate": "TRUE",
        "expected_truth_self_excluded": False,
        "insertion_table": spec.table,
        "insertion_index": spec.index,
        "query_table": spec.query_table,
        "filters_csv": str(args.filters_csv),
        "truth_csv": str(args.truth_csv),
        "iterative_scan_values": ITERATIVE_SCAN_VALUES,
        "stock_iterative_scan_values": ITERATIVE_SCAN_VALUES,
        "ef_search_values": ",".join(str(value) for value in EF_SEARCH_GRID),
        "target_recalls": ",".join(f"{value:.2f}" for value in TARGET_RECALLS),
        "traversal_guided_prioritization": False,
        "prewarm_index_health": True,
        "expected_sqlens_build_id": bound_build_id,
        "expected_vector_so_sha256": bound_vector_sha,
    }
    for key, value in expected.items():
        if run_args.get(key) != value:
            errors.append(f"run_spec.args.{key}={run_args.get(key)!r}, expected {value!r}")
    calibration_policy = payload.get("calibration_policy") or {}
    if not isinstance(calibration_policy, dict):
        errors.append("manifest calibration_policy is missing or malformed")
    elif selection_policy == "lcb_then_max_recall":
        if calibration_policy.get("calibration_selection_policy") != "lcb_then_max_recall":
            errors.append("manifest calibration policy does not match lcb_then_max_recall")
        if calibration_policy.get("stop_metric") != "recall_lcb95":
            errors.append("manifest calibration qualification metric is not recall_lcb95")
        if calibration_policy.get("grid_policy") != CALIBRATION_GRID_POLICY:
            errors.append("manifest calibration does not use the staged formal grid")
        if (
            calibration_policy.get("base_grid_max_ef") != max(BASE_EF_SEARCH_GRID)
            or calibration_policy.get("base_grid_complete_required") is not True
            or calibration_policy.get("extension_ef_search_values")
            != list(HIGH_EF_SEARCH_EXTENSION)
            or calibration_policy.get("extension_trigger")
            != "max_target_lcb95_unmet_after_complete_base_grid"
            or calibration_policy.get("extension_complete_required_when_triggered")
            is not True
            or calibration_policy.get("early_stop_allowed") is not False
            or calibration_policy.get("grid_exhaustion_semantics")
            != "all_policy_required_configs_executed"
        ):
            errors.append("manifest calibration staged-grid semantics are incomplete")
        stop_condition = str(calibration_policy.get("stop_condition") or "")
        if (
            "20--10000" not in stop_condition
            or "20000--100000" not in stop_condition
            or "maximum target" not in stop_condition
        ):
            errors.append("manifest calibration stop condition does not bind the staged grid")
    if payload.get("targets") != list(TARGET_RECALLS):
        errors.append("manifest target recalls differ from 0.90/0.95/0.99")
    if payload.get("modes") != list(MODES):
        errors.append("manifest modes differ from symmetric Stock/D1 protocol")
    if int(payload.get("expected_cells") or -1) != EXPECTED_CELLS:
        errors.append(f"manifest expected_cells is not {EXPECTED_CELLS}")
    run_spec = payload.get("run_spec") or {}
    if run_spec.get("filters_sha256") != sha256_file(args.filters_csv):
        errors.append("manifest filters hash differs from the audited input")
    if run_spec.get("truth_sha256") != sha256_file(args.truth_csv):
        errors.append("manifest truth hash differs from the audited input")
    runtime = run_spec.get("sqlens_runtime_provenance") or {}
    binding = run_spec.get("runtime_identity_binding") or {}
    manifest_contract = run_spec.get("p0_release_contract") or {}
    if (
        runtime.get("loaded_vector_sqlens_build_id")
        != bound_build_id
        or runtime.get("loaded_vector_so_sha256")
        != bound_vector_sha
        or binding.get("expected_build_id") != bound_build_id
        or binding.get("expected_vector_so_sha256")
        != bound_vector_sha
        or binding.get("exact_match") is not True
    ):
        errors.append("manifest does not preserve the wrapper's exact SQLens runtime binding")
    for field in (
        "contract_id",
        "sha256",
        "expected_sqlens_build_id",
        "expected_vector_so_sha256",
    ):
        if manifest_contract.get(field) != contract.get(field):
            errors.append(f"manifest P0 release contract mismatch: {field}")
    health = run_spec.get("index_query_health") or {}
    health_indexes = health.get("indexes") if isinstance(health, dict) else None
    if not isinstance(health_indexes, list) or not health_indexes:
        errors.append("manifest omits index-health prewarm evidence")
    else:
        for item in health_indexes:
            prewarm = item.get("prewarm") if isinstance(item, dict) else None
            if (
                not isinstance(prewarm, dict)
                or prewarm.get("enabled") is not True
                or not isinstance(prewarm.get("blocks"), int)
                or int(prewarm["blocks"]) < 0
                or not isinstance(prewarm.get("elapsed_ms"), (int, float))
                or float(prewarm["elapsed_ms"]) < 0
            ):
                errors.append("manifest contains incomplete index-health prewarm evidence")
                break
    expected_grid = {
        (ef_search, iterative_scan)
        for ef_search in EF_SEARCH_GRID
        for iterative_scan in ITERATIVE_SCAN_VALUES.split(",")
    }
    for mode in MODES:
        observed_grid = {
            (int(row["ef_search"]), str(row["iterative_scan"]))
            for row in (payload.get("mode_grids") or {}).get(mode, [])
        }
        if observed_grid != expected_grid:
            errors.append(f"manifest {mode} calibration grid is not the symmetric formal grid")
    expected_pairs = {
        (filter_name, mode)
        for filter_name in spec.filter_names
        for mode in MODES
    }
    calibration_pairs = payload.get("calibration_pairs") or []
    observed_pairs = {
        (str(pair.get("filter_name") or ""), str(pair.get("mode") or ""))
        for pair in calibration_pairs
    }
    if len(calibration_pairs) != len(expected_pairs) or observed_pairs != expected_pairs:
        errors.append("manifest calibration evidence is not the complete filter x method matrix")
    for pair in calibration_pairs:
        pair_name = f"{pair.get('filter_name')}/{pair.get('mode')}"
        if pair.get("calibration_grid_policy") != CALIBRATION_GRID_POLICY:
            errors.append(f"calibration pair {pair_name} does not use the staged formal grid")
            continue
        if pair.get("grid_exhausted") is not True or pair.get("stopped_early") is not False:
            errors.append(f"calibration pair {pair_name} did not exhaust its required staged grid")
        families = pair.get("families") or {}
        if set(families) != set(ITERATIVE_SCAN_VALUES.split(",")):
            errors.append(f"calibration pair {pair_name} omits an iterative-scan family")
            continue
        for family, evidence in families.items():
            family_name = f"{pair_name}/{family}"
            required = evidence.get("high_extension_required") is True
            expected_configs = len(EF_SEARCH_GRID) if required else len(BASE_EF_SEARCH_GRID)
            if int(evidence.get("configs_planned") or -1) != expected_configs:
                errors.append(f"calibration family {family_name} has an invalid planned grid")
            if int(evidence.get("configs_executed") or -1) != expected_configs:
                errors.append(f"calibration family {family_name} has an incomplete executed grid")
            if evidence.get("grid_exhausted") is not True:
                errors.append(f"calibration family {family_name} is not exhausted")
            if required:
                if evidence.get("high_extension_executed") is not True:
                    errors.append(f"calibration family {family_name} skipped a required extension")
                if int(evidence.get("max_ef_evaluated") or 0) != max(EF_SEARCH_GRID):
                    errors.append(f"calibration family {family_name} did not reach max extension ef")
            else:
                if evidence.get("high_extension_executed") is not False:
                    errors.append(f"calibration family {family_name} ran an unnecessary extension")
                if evidence.get("high_extension_skip_reason") != (
                    "max_target_lcb_met_on_complete_base_grid"
                ):
                    errors.append(f"calibration family {family_name} lacks a valid skip reason")
                if int(evidence.get("max_ef_evaluated") or 0) != max(BASE_EF_SEARCH_GRID):
                    errors.append(f"calibration family {family_name} did not complete the base grid")
    outputs = payload.get("outputs") or {}
    for required_output in ("calibration", "selected", "final"):
        if required_output not in outputs:
            errors.append(f"manifest is missing required output: {required_output}")
    for name, artifact in outputs.items():
        artifact_path = Path(str((artifact or {}).get("path") or ""))
        if not artifact_path.is_file():
            errors.append(f"missing manifest output {name}: {artifact_path}")
        elif (artifact or {}).get("sha256") != sha256_file(artifact_path):
            errors.append(f"manifest output hash mismatch: {name}")
    expected_matrix = {
        (filter_name, target, mode)
        for filter_name in spec.filter_names
        for target in TARGET_RECALLS
        for mode in MODES
    }
    for artifact_name in ("selected", "final"):
        artifact = outputs.get(artifact_name) or {}
        artifact_path = Path(str(artifact.get("path") or ""))
        if not artifact_path.is_file():
            continue
        with artifact_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        keys = [
            (str(row["filter_name"]), float(row["target_recall"]), str(row["mode"]))
            for row in rows
        ]
        if len(keys) != EXPECTED_CELLS or set(keys) != expected_matrix:
            errors.append(
                f"manifest {artifact_name} output is not the complete independent "
                f"filter x target x method matrix"
            )
        if len(keys) != len(set(keys)):
            errors.append(f"manifest {artifact_name} output contains duplicate cells")
    protocol_complete = bool(
        payload.get("status") == "complete"
        and payload.get("requested_slice_complete") is True
        and payload.get("matrix_complete") is True
        and payload.get("measurement_complete") is True
        and payload.get("comparison_valid") is True
        and not errors
    )
    diagnostic_valid = bool(protocol_complete)
    artifact_valid = bool(
        protocol_complete
        and payload.get("formal_release_complete") is True
        and payload.get("artifact_valid") is True
        and payload.get("paper_eligible") is True
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "protocol_complete": protocol_complete,
        "diagnostic_valid": diagnostic_valid,
        "artifact_valid": artifact_valid,
        "paper_eligible": artifact_valid,
        "release_contract": contract,
        "generic_formal_release_complete": payload.get("formal_release_complete"),
        "generic_formal_release_note": (
            "Expected false for this Stock/D1-only dataset slice; protocol_complete audits the requested 84 cells."
        ),
        "errors": errors,
    }


def locate_generic_manifest(full_tag: str) -> Path:
    matches = sorted(RESULTS.glob(f"sigmod_matched_recall_manifest_*_{full_tag}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one generic manifest for tag {full_tag!r}, found {len(matches)}")
    return matches[0]


def record_launch_failure(
    wrapper_payload: dict[str, Any], exc: BaseException
) -> dict[str, Any]:
    wrapper_payload["completed_at"] = utc_now()
    wrapper_payload["target_runner_returncode"] = (
        130 if isinstance(exc, KeyboardInterrupt) else None
    )
    wrapper_payload["status"] = (
        "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
    )
    wrapper_payload["error"] = {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }
    return wrapper_payload


def run_independent_raw_audit(
    manifest_path: Path,
    args: argparse.Namespace,
    *,
    audit_path: Path,
) -> dict[str, Any]:
    """Execute the standalone raw-level auditor and retain its immutable output hash."""
    command = [
        sys.executable,
        str(INDEPENDENT_AUDITOR),
        "--manifest",
        str(manifest_path),
        "--truth-csv",
        str(args.truth_csv),
        "--filters-csv",
        str(args.filters_csv),
        "--release-contract",
        str(args.release_contract),
        "--json",
        str(audit_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    result: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "path": str(audit_path),
        "sha256": None,
        "overall_valid": False,
    }
    if not audit_path.is_file():
        result["error"] = "independent raw-level auditor did not write JSON output"
        return result
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"cannot read independent audit output: {exc}"
        return result
    result["sha256"] = sha256_file(audit_path)
    result["overall_valid"] = bool(
        completed.returncode == 0 and payload.get("overall_valid") is True
    )
    result["audit"] = payload
    return result


def parser_for(spec: DatasetSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Run {spec.display_name} with the Amazon matched-recall calibration/held-out protocol."
        )
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--filters-csv", type=Path, default=spec.default_filters_csv)
    parser.add_argument("--truth-csv", type=Path, default=spec.default_truth_csv)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-database", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibration-recall-margin", type=float, default=0.0)
    parser.add_argument(
        "--calibration-selection-policy",
        choices=["mean_latency", "lcb_then_max_recall"],
        default="lcb_then_max_recall",
    )
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument("--statement-timeout-ms", type=int, default=300000)
    parser.add_argument("--progress-queries", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--guidance-max-atoms", type=int, default=128)
    parser.add_argument("--backend-cpu-list")
    parser.add_argument(
        "--release-contract",
        type=Path,
        default=DEFAULT_P0_RELEASE_CONTRACT,
        help="Immutable P0 r33 build-id/vector.so SHA contract.",
    )
    parser.add_argument(
        "--prewarm-index-health",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--expected-sqlens-build-id")
    parser.add_argument("--expected-vector-so-sha256")
    parser.add_argument("--reuse-calibration-manifest", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def bind_release_contract(args: argparse.Namespace) -> dict[str, object]:
    contract = load_p0_release_contract(Path(args.release_contract))
    build_id = str(contract["expected_sqlens_build_id"])
    vector_sha = str(contract["expected_vector_so_sha256"])
    if args.expected_sqlens_build_id and args.expected_sqlens_build_id != build_id:
        raise SystemExit("--expected-sqlens-build-id differs from the immutable P0 release contract")
    if args.expected_vector_so_sha256 and args.expected_vector_so_sha256 != vector_sha:
        raise SystemExit("--expected-vector-so-sha256 differs from the immutable P0 release contract")
    args.expected_sqlens_build_id = build_id
    args.expected_vector_so_sha256 = vector_sha
    args.release_contract_provenance = contract
    return contract


def run_dataset(spec: DatasetSpec, argv: list[str] | None = None) -> int:
    args = parser_for(spec).parse_args(argv)
    release_contract = bind_release_contract(args)
    if args.calibration_recall_margin < 0:
        raise SystemExit("--calibration-recall-margin must be non-negative")
    observed_filter_names, filters_by_name, filter_errors = load_and_audit_filters(args.filters_csv)
    filter_names = list(spec.filter_names)
    if not filter_errors and observed_filter_names != filter_names:
        filter_errors.append(
            "filters CSV names/order differ from the preregistered dataset workload"
        )
    truth = audit_truth(args.truth_csv, filter_names, filters_by_name)
    database = (
        database_readiness(spec, filters_by_name)
        if args.check_database
        else {"ready": True, "skipped": True, "errors": []}
    )
    if args.check_database and database.get("ready") is True:
        observed_build = str(database.get("sqlens_build_id") or "")
        observed_sha = str(database.get("vector_so_sha256") or "")
        if args.expected_sqlens_build_id and args.expected_sqlens_build_id != observed_build:
            database["errors"].append(
                "requested SQLens build ID differs from the loaded runtime"
            )
        if (
            args.expected_vector_so_sha256
            and args.expected_vector_so_sha256 != observed_sha
        ):
            database["errors"].append(
                "requested vector.so SHA256 differs from the loaded runtime"
            )
        database["ready"] = not database["errors"]
    elif not args.check_database and (
        not args.expected_sqlens_build_id or not args.expected_vector_so_sha256
    ):
        database["errors"].append(
            "database checks may be skipped only with an exact build ID/vector.so SHA256 binding"
        )
        database["ready"] = False
    try:
        command = build_target_command(spec, args, filter_names)
    except ValueError as exc:
        database["errors"].append(str(exc))
        database["ready"] = False
        command = []
    dataset_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(spec).items()
    }
    readiness = {
        "dataset": dataset_payload,
        "protocol": {
            "targets": list(TARGET_RECALLS),
            "ef_search_grid": list(EF_SEARCH_GRID),
            "calibration": {"queries": 80, "repeats": 2, "offset": 0},
            "final": {"queries": 100, "repeats": 5, "offset": 80, "execution_order": "interleaved"},
            "modes": list(MODES),
            "iterative_scan_grid_by_mode": {mode: ITERATIVE_SCAN_VALUES.split(",") for mode in MODES},
            "candidate_validity_predicate": "TRUE",
            "truth_self_excluded": False,
            "recall_truth": "exact SQL-valid tie-aware top-10",
        },
        "filters": {
            "path": str(args.filters_csv),
            "count": len(observed_filter_names),
            "sha256": sha256_file(args.filters_csv) if args.filters_csv.is_file() else None,
            "errors": filter_errors,
        },
        "truth": truth,
        "database": database,
        "release_contract": release_contract,
        "truth_builder_command": truth_builder_command(spec, args),
        "target_runner_command": command,
        "target_runner_command_shell": shlex.join(command),
    }
    readiness["ready"] = not filter_errors and truth.get("ready") is True and database.get("ready") is True
    if args.dry_run:
        print(json.dumps(readiness, indent=2, default=str), flush=True)
        return 0 if readiness["ready"] else 2
    if not readiness["ready"]:
        print(json.dumps(readiness, indent=2, default=str), file=sys.stderr, flush=True)
        raise SystemExit("dataset protocol inputs are not ready; run the reported truth_builder_command first")

    full_tag = f"{spec.key}_{args.tag}"
    launch_manifest = RESULTS / f"{spec.key}_matched_recall_launch_{args.tag}.json"
    wrapper_payload: dict[str, Any] = {
        **readiness,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "generic_manifest": None,
        "independent_raw_audit": None,
        "diagnostic_valid": False,
        "artifact_valid": False,
        "paper_eligible": False,
        "formal_complete": False,
    }
    write_json_atomic(launch_manifest, wrapper_payload)
    try:
        result = subprocess.run(command, cwd=ROOT, check=False)
    except BaseException as exc:
        record_launch_failure(wrapper_payload, exc)
        try:
            generic_manifest = locate_generic_manifest(full_tag)
            wrapper_payload["generic_manifest"] = audit_generic_manifest(
                generic_manifest, spec, args
            )
        except Exception as audit_exc:
            wrapper_payload["manifest_error"] = (
                f"{audit_exc.__class__.__name__}: {audit_exc}"
            )
        write_json_atomic(launch_manifest, wrapper_payload)
        raise
    wrapper_payload["completed_at"] = utc_now()
    wrapper_payload["target_runner_returncode"] = result.returncode
    try:
        generic_manifest = locate_generic_manifest(full_tag)
        wrapper_payload["generic_manifest"] = audit_generic_manifest(generic_manifest, spec, args)
        raw_audit_path = RESULTS / f"{spec.key}_matched_recall_raw_audit_{args.tag}.json"
        wrapper_payload["independent_raw_audit"] = run_independent_raw_audit(
            generic_manifest,
            args,
            audit_path=raw_audit_path,
        )
        raw_valid = bool(wrapper_payload["independent_raw_audit"].get("overall_valid"))
        generic = wrapper_payload["generic_manifest"]
        wrapper_payload["diagnostic_valid"] = bool(
            generic.get("diagnostic_valid") and raw_valid
        )
        wrapper_payload["artifact_valid"] = bool(
            generic.get("artifact_valid") and raw_valid
        )
        wrapper_payload["paper_eligible"] = bool(
            generic.get("paper_eligible") and raw_valid
        )
        wrapper_payload["formal_complete"] = bool(
            result.returncode == 0
            and generic.get("protocol_complete")
            and raw_valid
        )
        wrapper_payload["status"] = (
            "complete"
            if wrapper_payload["formal_complete"]
            else "incomplete"
        )
    except Exception as exc:
        wrapper_payload["status"] = "failed"
        wrapper_payload["manifest_error"] = f"{exc.__class__.__name__}: {exc}"
    write_json_atomic(launch_manifest, wrapper_payload)
    print(f"wrote {launch_manifest}", flush=True)
    return 0 if wrapper_payload["status"] == "complete" else 2
