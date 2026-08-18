"""Create an independently audited exact-truth provenance manifest.

The auditor is intentionally read-only with respect to PostgreSQL.  It binds an
external 14-filter truth matrix to the current source table, HNSW index, query
cohort, and loaded SQLens binary.  A historical launch manifest may contribute
source provenance even when its benchmark was interrupted, but only when its
completed truth/filter/database sections still match the files and live OIDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import psycopg
from psycopg import sql

try:
    from .common_pg import pg_config_from_env
except ImportError:
    from common_pg import pg_config_from_env


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_FILTER_COUNT = 14
EXACT_K = 10
RECALL_CONTRACT = "distance_squared_threshold_tie_aware_v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_SQL_TOKENS = (";", "--", "/*", "*/")


class AuditError(RuntimeError):
    """An exact-truth provenance contract could not be established."""


@dataclass(frozen=True)
class FilterSpec:
    name: str
    predicate: str
    expected_rows: int
    row: Mapping[str, str]


@dataclass(frozen=True)
class CohortSpec:
    calibration_offset: int
    calibration_queries: int
    final_offset: int
    final_queries: int

    def validate(self) -> None:
        if min(
            self.calibration_offset,
            self.calibration_queries,
            self.final_offset,
            self.final_queries,
        ) < 0:
            raise AuditError("query offsets/counts must be non-negative")
        if self.calibration_queries == 0 or self.final_queries == 0:
            raise AuditError("calibration and final cohorts must both be non-empty")
        if self.calibration_query_nos & self.final_query_nos:
            raise AuditError("calibration and final query cohorts overlap")

    @property
    def calibration_query_nos(self) -> set[int]:
        return set(
            range(
                self.calibration_offset,
                self.calibration_offset + self.calibration_queries,
            )
        )

    @property
    def final_query_nos(self) -> set[int]:
        return set(range(self.final_offset, self.final_offset + self.final_queries))

    @property
    def all_query_nos(self) -> set[int]:
        return self.calibration_query_nos | self.final_query_nos

    def split_for(self, query_no: int) -> str:
        if query_no in self.calibration_query_nos:
            return "calibration"
        if query_no in self.final_query_nos:
            return "final"
        raise AuditError(f"query_no={query_no} is outside the requested cohort")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predicate_sha256(predicate: str) -> str:
    return hashlib.sha256(predicate.encode("utf-8")).hexdigest()


def canonical_relation(value: object) -> str:
    name = str(value or "").strip()
    if not name:
        raise AuditError("relation name is empty")
    parts = name.split(".")
    if len(parts) == 1:
        parts.insert(0, "public")
    if len(parts) != 2 or any(not part or "\x00" in part for part in parts):
        raise AuditError(f"invalid relation name: {value!r}")
    return ".".join(parts)


def relation_identifier(value: str) -> sql.Identifier:
    return sql.Identifier(*canonical_relation(value).split("."))


def column_identifier(value: object) -> sql.Identifier:
    name = str(value or "").strip()
    if not name or "." in name or "\x00" in name:
        raise AuditError(f"invalid column name: {value!r}")
    return sql.Identifier(name)


def normalize_sql(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def validate_predicate(value: object, label: str) -> str:
    predicate = normalize_sql(value)
    if not predicate or any(token in predicate for token in FORBIDDEN_SQL_TOKENS):
        raise AuditError(f"{label} is empty or contains a forbidden SQL token")
    return predicate


def parse_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise AuditError(f"{label} is not a boolean: {value!r}")


def parse_int(value: object, label: str, *, minimum: int | None = None) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} is not an integer: {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise AuditError(f"{label} must be >= {minimum}, observed={parsed}")
    return parsed


def parse_float(value: object, label: str, *, minimum: float | None = None) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise AuditError(f"{label} is not finite")
    if minimum is not None and parsed < minimum:
        raise AuditError(f"{label} must be >= {minimum}, observed={parsed}")
    return parsed


def read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise AuditError(f"{label} does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if not fields:
        raise AuditError(f"{label} has no header")
    if any(None in row for row in rows):
        raise AuditError(f"{label} contains a row wider than its header")
    return fields, rows


def require_fields(fields: Sequence[str], required: set[str], label: str) -> None:
    missing = sorted(required - set(fields))
    if missing:
        raise AuditError(f"{label} is missing fields: {missing}")


def audit_filters(path: Path) -> tuple[list[FilterSpec], dict[str, Any]]:
    fields, rows = read_csv(path, "filters CSV")
    require_fields(fields, {"filter_name", "predicate", "expected_rows"}, "filters CSV")
    if len(rows) != EXPECTED_FILTER_COUNT:
        raise AuditError(
            f"filters CSV must contain exactly {EXPECTED_FILTER_COUNT} rows, observed={len(rows)}"
        )
    specs: list[FilterSpec] = []
    seen: set[str] = set()
    for row_no, row in enumerate(rows, start=2):
        name = str(row.get("filter_name") or "").strip()
        if not name:
            raise AuditError(f"filters CSV row {row_no} has an empty filter_name")
        if name in seen:
            raise AuditError(f"filters CSV contains duplicate filter_name={name!r}")
        seen.add(name)
        predicate = validate_predicate(row.get("predicate"), f"filter {name} predicate")
        expected_rows = parse_int(
            row.get("expected_rows"), f"filter {name} expected_rows", minimum=EXACT_K
        )
        specs.append(FilterSpec(name, predicate, expected_rows, row))
    return specs, {
        "valid": True,
        "row_count": len(specs),
        "filter_names": [item.name for item in specs],
        "expected_rows": {item.name: item.expected_rows for item in specs},
        "sha256": sha256_file(path),
    }


def parse_csv_ints(value: object, label: str) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    return [parse_int(part, label) for part in text.split(",")]


def parse_csv_floats(value: object, label: str) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return []
    return [parse_float(part, label, minimum=0.0) for part in text.split(",")]


def audit_truth(
    path: Path,
    filters: Sequence[FilterSpec],
    cohort: CohortSpec,
    *,
    candidate_validity_predicate: str,
    self_excluded: bool,
) -> tuple[dict[int, int], dict[str, Any]]:
    fields, rows = read_csv(path, "truth CSV")
    required = {
        "query_no",
        "query_id",
        "query_split",
        "filter_name",
        "predicate",
        "candidate_validity_predicate",
        "method",
        "k",
        "recall_at_10_exact_filtered",
        "returned",
        "filtered_rows",
        "search_candidate_rows",
        "result_ids",
        "exact_filtered_topk_ids",
        "exact_filtered_topk_distances_sq",
        "kth_distance_sq",
        "tie_tolerance",
        "strict_closer_count",
        "boundary_tied",
        "self_excluded",
        "self_excluded_rows",
    }
    require_fields(fields, required, "truth CSV")
    by_filter = {item.name: item for item in filters}
    expected_cells = {
        (item.name, query_no)
        for item in filters
        for query_no in cohort.all_query_nos
    }
    if len(rows) != len(expected_cells):
        raise AuditError(
            "truth CSV row count does not equal the full 14-filter cohort: "
            f"expected={len(expected_cells)}, observed={len(rows)}"
        )
    seen_cells: set[tuple[str, int]] = set()
    query_ids: dict[int, int] = {}
    id_to_query_no: dict[int, int] = {}
    boundary_tied_cells = 0
    for row_no, row in enumerate(rows, start=2):
        query_no = parse_int(row.get("query_no"), f"truth row {row_no} query_no")
        query_id = parse_int(row.get("query_id"), f"truth row {row_no} query_id")
        split = cohort.split_for(query_no)
        if str(row.get("query_split") or "").strip() != split:
            raise AuditError(
                f"truth row {row_no} query_split does not match query_no={query_no}"
            )
        name = str(row.get("filter_name") or "").strip()
        spec = by_filter.get(name)
        if spec is None:
            raise AuditError(f"truth row {row_no} references unknown filter={name!r}")
        cell = (name, query_no)
        if cell in seen_cells:
            raise AuditError(f"truth CSV contains duplicate cell={cell!r}")
        seen_cells.add(cell)
        previous_id = query_ids.setdefault(query_no, query_id)
        if previous_id != query_id:
            raise AuditError(f"query_no={query_no} maps to multiple query IDs")
        previous_query_no = id_to_query_no.setdefault(query_id, query_no)
        if previous_query_no != query_no:
            raise AuditError(f"query_id={query_id} maps to multiple query numbers")

        observed_predicate = validate_predicate(
            row.get("predicate"), f"truth row {row_no} predicate"
        )
        if observed_predicate != spec.predicate:
            raise AuditError(f"truth cell={cell!r} predicate differs from filters CSV")
        if normalize_sql(row.get("candidate_validity_predicate")) != (
            candidate_validity_predicate
        ):
            raise AuditError(
                f"truth cell={cell!r} candidate-validity predicate differs from CLI"
            )
        if parse_bool(row.get("self_excluded"), f"truth cell={cell!r} self_excluded") != (
            self_excluded
        ):
            raise AuditError(f"truth cell={cell!r} self-exclusion contract differs from CLI")
        if str(row.get("method") or "").strip() != "pre_filter_exact":
            raise AuditError(f"truth cell={cell!r} is not pre_filter_exact")
        k = parse_int(row.get("k"), f"truth cell={cell!r} k", minimum=1)
        if k != EXACT_K:
            raise AuditError(f"truth cell={cell!r} uses k={k}, expected={EXACT_K}")
        recall = parse_float(
            row.get("recall_at_10_exact_filtered"),
            f"truth cell={cell!r} exact recall",
            minimum=0.0,
        )
        if recall != 1.0:
            raise AuditError(f"truth cell={cell!r} exact recall is not 1.0")
        returned = parse_int(row.get("returned"), f"truth cell={cell!r} returned")
        if returned != EXACT_K:
            raise AuditError(f"truth cell={cell!r} returned={returned}, expected={EXACT_K}")
        filtered_rows = parse_int(
            row.get("filtered_rows"), f"truth cell={cell!r} filtered_rows"
        )
        if filtered_rows != spec.expected_rows:
            raise AuditError(
                f"truth cell={cell!r} filtered_rows={filtered_rows}, "
                f"expected_rows={spec.expected_rows}"
            )
        excluded_rows = parse_int(
            row.get("self_excluded_rows"),
            f"truth cell={cell!r} self_excluded_rows",
            minimum=0,
        )
        if excluded_rows not in ({0, 1} if self_excluded else {0}):
            raise AuditError(f"truth cell={cell!r} has invalid self_excluded_rows")
        search_rows = parse_int(
            row.get("search_candidate_rows"),
            f"truth cell={cell!r} search_candidate_rows",
        )
        if search_rows != filtered_rows - excluded_rows:
            raise AuditError(f"truth cell={cell!r} search candidate count is inconsistent")
        for optional_count in ("candidates", "candidate_rows"):
            if optional_count in fields and str(row.get(optional_count) or "").strip():
                observed_count = parse_int(
                    row.get(optional_count), f"truth cell={cell!r} {optional_count}"
                )
                if observed_count != filtered_rows:
                    raise AuditError(f"truth cell={cell!r} {optional_count} is inconsistent")

        result_ids = parse_csv_ints(row.get("result_ids"), f"truth cell={cell!r} result_ids")
        exact_ids = parse_csv_ints(
            row.get("exact_filtered_topk_ids"), f"truth cell={cell!r} exact IDs"
        )
        distances = parse_csv_floats(
            row.get("exact_filtered_topk_distances_sq"),
            f"truth cell={cell!r} exact distances",
        )
        if result_ids != exact_ids or len(exact_ids) != EXACT_K or len(distances) != EXACT_K:
            raise AuditError(f"truth cell={cell!r} exact top-k payload is inconsistent")
        if any(right < left for left, right in zip(distances, distances[1:])):
            raise AuditError(f"truth cell={cell!r} exact distances are not ordered")
        kth = parse_float(
            row.get("kth_distance_sq"), f"truth cell={cell!r} kth_distance_sq", minimum=0.0
        )
        tolerance = parse_float(
            row.get("tie_tolerance"), f"truth cell={cell!r} tie_tolerance", minimum=0.0
        )
        expected_tolerance = max(1e-9, abs(kth) * 1e-6)
        if not math.isclose(distances[-1], kth, rel_tol=1e-8, abs_tol=1e-12):
            raise AuditError(f"truth cell={cell!r} kth distance differs from top-k payload")
        if not math.isclose(tolerance, expected_tolerance, rel_tol=1e-7, abs_tol=1e-12):
            raise AuditError(f"truth cell={cell!r} tie tolerance violates the contract")
        strict = parse_int(
            row.get("strict_closer_count"),
            f"truth cell={cell!r} strict_closer_count",
            minimum=0,
        )
        expected_strict = sum(value < kth - tolerance for value in distances)
        if strict != expected_strict or strict >= EXACT_K:
            raise AuditError(f"truth cell={cell!r} strict-closer count is inconsistent")
        if parse_bool(row.get("boundary_tied"), f"truth cell={cell!r} boundary_tied"):
            boundary_tied_cells += 1

    if seen_cells != expected_cells:
        missing = len(expected_cells - seen_cells)
        extra = len(seen_cells - expected_cells)
        raise AuditError(f"truth matrix is incomplete: missing={missing}, extra={extra}")
    if set(query_ids) != cohort.all_query_nos:
        raise AuditError("truth query_no domain differs from the requested full cohort")
    return query_ids, {
        "valid": True,
        "row_count": len(rows),
        "cell_count": len(seen_cells),
        "filter_count": len(filters),
        "query_count": len(query_ids),
        "query_no_min": min(query_ids),
        "query_no_max": max(query_ids),
        "unique_query_id_count": len(set(query_ids.values())),
        "boundary_tied_cells": boundary_tied_cells,
        "recall_contract": RECALL_CONTRACT,
        "self_excluded": self_excluded,
        "sha256": sha256_file(path),
    }


def fetch_relation(cur: Any, relation: str) -> dict[str, Any]:
    cur.execute(
        "SELECT c.oid::bigint, c.oid::regclass::text, "
        "pg_relation_filenode(c.oid)::bigint, c.relkind, c.reltuples::bigint, "
        "pg_relation_size(c.oid)::bigint "
        "FROM pg_class c WHERE c.oid=to_regclass(%s)",
        (relation,),
    )
    row = cur.fetchone()
    if row is None:
        raise AuditError(f"database relation does not exist: {relation}")
    if str(row[3]) not in {"r", "p", "m"}:
        raise AuditError(f"database relation has unsupported relkind={row[3]!r}: {relation}")
    return {
        "name": canonical_relation(row[1]),
        "oid": int(row[0]),
        "relfilenode": int(row[2]),
        "relkind": str(row[3]),
        "estimated_rows": int(row[4]),
        "bytes": int(row[5]),
    }


def fetch_columns(cur: Any, relation_oid: int, expected: Sequence[str]) -> dict[str, Any]:
    cur.execute(
        "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull "
        "FROM pg_attribute a WHERE a.attrelid=%s AND a.attnum > 0 AND NOT a.attisdropped",
        (relation_oid,),
    )
    observed = {
        str(row[0]): {"type": str(row[1]), "not_null": bool(row[2])}
        for row in cur.fetchall()
    }
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise AuditError(f"database relation oid={relation_oid} is missing columns: {missing}")
    return {name: observed[name] for name in expected}


def fetch_source_index(cur: Any, index: str, table_oid: int, vector_column: str) -> dict[str, Any]:
    cur.execute(
        "SELECT idx.oid::bigint, idx.oid::regclass::text, "
        "pg_relation_filenode(idx.oid)::bigint, i.indrelid::bigint, "
        "tbl.oid::regclass::text, am.amname, i.indisvalid, i.indisready, i.indislive, "
        "pg_get_indexdef(idx.oid), pg_get_expr(i.indpred, i.indrelid), i.indnkeyatts, "
        "CASE WHEN i.indexprs IS NULL THEN a.attname ELSE NULL END "
        "FROM pg_class idx JOIN pg_index i ON i.indexrelid=idx.oid "
        "JOIN pg_class tbl ON tbl.oid=i.indrelid "
        "JOIN pg_am am ON am.oid=idx.relam "
        "LEFT JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=i.indkey[0] "
        "WHERE idx.oid=to_regclass(%s)",
        (index,),
    )
    row = cur.fetchone()
    if row is None:
        raise AuditError(f"source index does not exist: {index}")
    evidence = {
        "name": canonical_relation(row[1]),
        "oid": int(row[0]),
        "relfilenode": int(row[2]),
        "heap_oid": int(row[3]),
        "heap_name": canonical_relation(row[4]),
        "access_method": str(row[5]),
        "valid": bool(row[6]),
        "ready": bool(row[7]),
        "live": bool(row[8]),
        "definition": str(row[9]),
        "predicate": None if row[10] is None else str(row[10]),
        "key_columns": int(row[11]),
        "indexed_column": None if row[12] is None else str(row[12]),
    }
    if evidence["heap_oid"] != table_oid:
        raise AuditError("source HNSW index is not attached to the requested source table")
    if evidence["access_method"] != "hnsw":
        raise AuditError("source index does not use the HNSW access method")
    if not all(evidence[field] for field in ("valid", "ready", "live")):
        raise AuditError("source HNSW index is not valid, ready, and live")
    if evidence["key_columns"] != 1 or evidence["indexed_column"] != vector_column:
        raise AuditError("source HNSW index does not index the requested vector column")
    return evidence


def audit_database(
    args: argparse.Namespace,
    filters: Sequence[FilterSpec],
    query_ids: Mapping[int, int],
    *,
    connect_factory: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    connect = connect_factory or psycopg.connect
    table = canonical_relation(args.table)
    query_table = canonical_relation(args.query_table)
    source_index = canonical_relation(args.source_index)
    candidate = validate_predicate(
        args.candidate_validity_predicate, "candidate-validity predicate"
    )
    expected_sha = str(args.expected_vector_so_sha256).lower()
    if not SHA256_RE.fullmatch(expected_sha):
        raise AuditError("expected vector.so SHA256 must be 64 lowercase hexadecimal characters")
    if not str(args.expected_sqlens_build_id or "").strip():
        raise AuditError("expected SQLens build ID is empty")

    with connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(
            "WITH lib AS (SELECT setting || '/vector.so' AS path "
            "FROM pg_config WHERE name='PKGLIBDIR') "
            "SELECT current_setting('server_version'), "
            "COALESCE((SELECT extversion FROM pg_extension WHERE extname='vector'), ''), "
            "vector_sqlens_build_id(), path, "
            "encode(sha256(pg_read_binary_file(path)), 'hex') FROM lib"
        )
        runtime_row = cur.fetchone()
        if runtime_row is None:
            raise AuditError("SQLens runtime identity query returned no row")
        runtime = {
            "postgres_version": str(runtime_row[0]),
            "vector_extension_version": str(runtime_row[1]),
            "expected_sqlens_build_id": str(args.expected_sqlens_build_id),
            "observed_sqlens_build_id": str(runtime_row[2]),
            "loaded_vector_so_path": str(runtime_row[3]),
            "expected_vector_so_sha256": expected_sha,
            "observed_vector_so_sha256": str(runtime_row[4]).lower(),
        }
        runtime["build_id_exact_match"] = (
            runtime["observed_sqlens_build_id"] == runtime["expected_sqlens_build_id"]
        )
        runtime["vector_so_sha256_exact_match"] = (
            runtime["observed_vector_so_sha256"] == expected_sha
        )
        if not runtime["build_id_exact_match"] or not runtime["vector_so_sha256_exact_match"]:
            raise AuditError("loaded SQLens build/vector.so does not match the requested runtime")
        if not str(runtime["loaded_vector_so_path"]).endswith("/vector.so"):
            raise AuditError("loaded vector.so path is invalid")

        table_evidence = fetch_relation(cur, table)
        query_evidence = fetch_relation(cur, query_table)
        table_columns = fetch_columns(
            cur, table_evidence["oid"], (args.table_id_column, args.table_vector_column)
        )
        query_columns = fetch_columns(
            cur,
            query_evidence["oid"],
            (args.query_id_column, args.query_vector_column),
        )
        index_evidence = fetch_source_index(
            cur, source_index, table_evidence["oid"], args.table_vector_column
        )

        requested_ids = sorted(set(query_ids.values()))
        query = sql.SQL(
            "SELECT {column}, count(*)::bigint FROM {table} "
            "WHERE {column} = ANY(%s) GROUP BY {column}"
        ).format(
            column=column_identifier(args.query_id_column),
            table=relation_identifier(query_table),
        )
        cur.execute(query, (requested_ids,))
        found_counts = {int(row[0]): int(row[1]) for row in cur.fetchall()}
        missing_ids = sorted(set(requested_ids) - set(found_counts))
        duplicate_ids = sorted(key for key, count in found_counts.items() if count != 1)
        if missing_ids or duplicate_ids:
            raise AuditError(
                "query table does not contain exactly one row per truth query ID: "
                f"missing={missing_ids[:10]}, non_unique={duplicate_ids[:10]}"
            )

        filter_counts: dict[str, Any] = {}
        for spec in filters:
            started = time.perf_counter()
            query = sql.SQL(
                "SELECT count(*)::bigint FROM {table} "
                "WHERE ({candidate}) AND ({predicate})"
            ).format(
                table=relation_identifier(table),
                candidate=sql.SQL(candidate),
                predicate=sql.SQL(spec.predicate),
            )
            cur.execute(query)
            row = cur.fetchone()
            observed_rows = int(row[0]) if row is not None else -1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            filter_counts[spec.name] = {
                "predicate": spec.predicate,
                "expected_rows": spec.expected_rows,
                "observed_rows": observed_rows,
                "exact_count_ms": elapsed_ms,
                "matches": observed_rows == spec.expected_rows,
            }
            if observed_rows != spec.expected_rows:
                raise AuditError(
                    f"database COUNT for filter={spec.name!r} is {observed_rows}, "
                    f"expected={spec.expected_rows}"
                )

    database = {
        "valid": True,
        "relations": {table: table_evidence, query_table: query_evidence},
        "columns": {table: table_columns, query_table: query_columns},
        "source_index": index_evidence,
        "query_ids": {
            "requested": len(requested_ids),
            "found": len(found_counts),
            "all_present_once": True,
            "sha256": hashlib.sha256(
                ",".join(str(value) for value in requested_ids).encode("ascii")
            ).hexdigest(),
        },
        "filter_counts": filter_counts,
        "candidate_validity_predicate": candidate,
        "candidate_validity_predicate_sha256": predicate_sha256(candidate),
    }
    runtime["valid"] = True
    return database, runtime


def resolve_launch_artifact(path_value: object, manifest_path: Path) -> Path:
    path = Path(str(path_value or ""))
    if path.is_absolute():
        return path.resolve()
    root_candidate = (ROOT / path).resolve()
    if root_candidate.exists():
        return root_candidate
    return (manifest_path.parent / path).resolve()


def require_launch_artifact(
    section: Mapping[str, Any], expected_path: Path, manifest_path: Path, label: str
) -> None:
    observed_path = resolve_launch_artifact(section.get("path"), manifest_path)
    if observed_path != expected_path.resolve():
        raise AuditError(
            f"old launch {label} path mismatch: expected={expected_path.resolve()}, "
            f"observed={observed_path}"
        )
    expected_sha = sha256_file(expected_path)
    if str(section.get("sha256") or "").lower() != expected_sha:
        raise AuditError(f"old launch {label} SHA256 does not match the audited file")


def require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise AuditError(f"{label} mismatch: expected={expected!r}, observed={observed!r}")


def audit_old_launch(
    path: Path,
    args: argparse.Namespace,
    filters_path: Path,
    truth_path: Path,
    filters: Sequence[FilterSpec],
    truth_audit: Mapping[str, Any],
    database: Mapping[str, Any],
    cohort: CohortSpec,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read old launch manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditError("old launch manifest is not a JSON object")
    status = payload.get("status")
    if status not in {"complete", "interrupted"}:
        raise AuditError(f"old launch status is not reusable: {status!r}")
    if payload.get("ready") is not True:
        raise AuditError("old launch global readiness gate did not pass")
    sections: dict[str, Mapping[str, Any]] = {}
    for name in ("dataset", "database", "truth", "filters", "protocol"):
        section = payload.get(name)
        if not isinstance(section, dict):
            raise AuditError(f"old launch has no {name} section")
        sections[name] = section
    for name in ("database", "truth", "filters"):
        section = sections[name]
        if section.get("ready") is not True or section.get("errors") not in (None, []):
            raise AuditError(f"old launch {name} section is not independently ready")
    require_launch_artifact(sections["truth"], truth_path, path, "truth")
    require_launch_artifact(sections["filters"], filters_path, path, "filters")

    dataset = sections["dataset"]
    for field, expected in (
        ("table", canonical_relation(args.table)),
        ("index", canonical_relation(args.source_index)),
        ("query_table", canonical_relation(args.query_table)),
    ):
        require_equal(canonical_relation(dataset.get(field)), expected, f"old launch {field}")
    for field in ("query_id_column", "query_vector_column"):
        require_equal(dataset.get(field), getattr(args, field), f"old launch {field}")
    require_equal(
        list(dataset.get("filter_names") or []),
        [item.name for item in filters],
        "old launch filter order",
    )

    old_database = sections["database"]
    require_equal(
        canonical_relation(old_database.get("index")),
        canonical_relation(args.source_index),
        "old launch database index",
    )
    old_relations = old_database.get("relations")
    if not isinstance(old_relations, dict):
        raise AuditError("old launch database section has no relation inventory")
    for relation in (canonical_relation(args.table), canonical_relation(args.query_table)):
        old_relation = old_relations.get(relation)
        live_relation = database["relations"].get(relation)
        if not isinstance(old_relation, dict) or not isinstance(live_relation, dict):
            raise AuditError(f"old/live relation provenance is missing for {relation}")
        require_equal(
            parse_int(old_relation.get("oid"), f"old launch {relation} OID"),
            int(live_relation["oid"]),
            f"old launch {relation} OID",
        )
    old_index_relation = old_relations.get(canonical_relation(args.source_index))
    if isinstance(old_index_relation, dict) and old_index_relation.get("oid") is not None:
        require_equal(
            parse_int(old_index_relation.get("oid"), "old launch source-index OID"),
            int(database["source_index"]["oid"]),
            "old launch source-index OID",
        )

    truth = sections["truth"]
    require_equal(
        parse_int(truth.get("row_count"), "old launch truth row_count"),
        int(truth_audit["row_count"]),
        "old launch truth row_count",
    )
    require_equal(
        parse_int(truth.get("query_count"), "old launch truth query_count"),
        len(cohort.all_query_nos),
        "old launch truth query_count",
    )
    require_equal(
        parse_int(sections["filters"].get("count"), "old launch filter count"),
        EXPECTED_FILTER_COUNT,
        "old launch filter count",
    )
    protocol = sections["protocol"]
    require_equal(
        normalize_sql(protocol.get("candidate_validity_predicate")),
        normalize_sql(args.candidate_validity_predicate),
        "old launch candidate-validity predicate",
    )
    require_equal(
        protocol.get("truth_self_excluded"),
        args.self_excluded,
        "old launch self-exclusion contract",
    )
    for split, offset, queries in (
        ("calibration", cohort.calibration_offset, cohort.calibration_queries),
        ("final", cohort.final_offset, cohort.final_queries),
    ):
        value = protocol.get(split)
        if not isinstance(value, dict):
            raise AuditError(f"old launch protocol has no {split} split")
        require_equal(
            parse_int(value.get("offset"), f"old launch {split} offset"),
            offset,
            f"old launch {split} offset",
        )
        require_equal(
            parse_int(value.get("queries"), f"old launch {split} queries"),
            queries,
            f"old launch {split} queries",
        )
    return {
        "kind": "external_launch_manifest",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "launch_status": status,
        "interrupted_benchmark_accepted": status == "interrupted",
        "truth_filter_database_sections_independently_ready": True,
        "artifact_sha256_matches": True,
        "relation_oid_matches": True,
    }


def atomic_write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise AuditError(f"refusing to overwrite existing output: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(payload, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise AuditError(f"refusing to overwrite existing output: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_manifest(
    args: argparse.Namespace,
    *,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    cohort = CohortSpec(
        args.calibration_offset,
        args.calibration_queries,
        args.final_offset,
        args.final_queries,
    )
    cohort.validate()
    filters_path = Path(args.filters_csv).resolve()
    truth_path = Path(args.truth_csv).resolve()
    filters, filters_audit = audit_filters(filters_path)
    candidate = validate_predicate(
        args.candidate_validity_predicate, "candidate-validity predicate"
    )
    query_ids, truth_audit = audit_truth(
        truth_path,
        filters,
        cohort,
        candidate_validity_predicate=candidate,
        self_excluded=args.self_excluded,
    )
    database, runtime = audit_database(
        args, filters, query_ids, connect_factory=connect_factory
    )
    if args.old_launch_manifest is None:
        source_provenance = {
            "kind": "independent_file_and_live_database_audit",
            "old_launch_manifest_supplied": False,
        }
    else:
        source_provenance = audit_old_launch(
            Path(args.old_launch_manifest).resolve(),
            args,
            filters_path,
            truth_path,
            filters,
            truth_audit,
            database,
            cohort,
        )
    table = canonical_relation(args.table)
    query_table = canonical_relation(args.query_table)
    source_index = canonical_relation(args.source_index)
    return {
        "schema_version": 1,
        "artifact_type": "external_exact_truth_provenance",
        "artifact_valid": True,
        "created_at": utc_now(),
        "recall_contract": RECALL_CONTRACT,
        "self_excluded": args.self_excluded,
        "outputs": {
            "truth_csv": {
                "path": str(truth_path),
                "sha256": truth_audit["sha256"],
                "row_count": truth_audit["row_count"],
            }
        },
        "inputs": {
            "filters_csv": {
                "path": str(filters_path),
                "sha256": filters_audit["sha256"],
                "row_count": filters_audit["row_count"],
            },
            "postgres": {
                "table": table,
                "source_index": source_index,
                "query_table": query_table,
                "columns": {
                    "table_id": args.table_id_column,
                    "table_vector": args.table_vector_column,
                    "query_id": args.query_id_column,
                    "query_vector": args.query_vector_column,
                },
                "query_population": {
                    "candidate_validity_predicate": candidate,
                    "candidate_validity_predicate_sha256": predicate_sha256(candidate),
                    "calibration": {
                        "offset": cohort.calibration_offset,
                        "queries": cohort.calibration_queries,
                    },
                    "final": {
                        "offset": cohort.final_offset,
                        "queries": cohort.final_queries,
                    },
                },
            },
            "old_launch_manifest": source_provenance,
        },
        "protocol": {
            "filter_count": EXPECTED_FILTER_COUNT,
            "query_count": len(cohort.all_query_nos),
            "expected_cells": EXPECTED_FILTER_COUNT * len(cohort.all_query_nos),
            "k": EXACT_K,
            "calibration": {
                "offset": cohort.calibration_offset,
                "queries": cohort.calibration_queries,
            },
            "final": {
                "offset": cohort.final_offset,
                "queries": cohort.final_queries,
            },
        },
        "audits": {
            "filters": filters_audit,
            "truth": truth_audit,
            "source_provenance": source_provenance,
        },
        "database": database,
        "runtime": runtime,
    }


def sha256_argument(value: str) -> str:
    normalized = value.strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected 64 hexadecimal SHA256 characters")
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filters-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--query-table", required=True)
    parser.add_argument("--table-id-column", default="id")
    parser.add_argument("--table-vector-column", default="embedding")
    parser.add_argument("--query-id-column", default="qid")
    parser.add_argument("--query-vector-column", default="embedding")
    parser.add_argument("--candidate-validity-predicate", required=True)
    exclusion = parser.add_mutually_exclusive_group(required=True)
    exclusion.add_argument("--self-excluded", dest="self_excluded", action="store_true")
    exclusion.add_argument(
        "--no-self-excluded", dest="self_excluded", action="store_false"
    )
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--calibration-queries", type=int, default=80)
    parser.add_argument("--final-offset", type=int, default=80)
    parser.add_argument("--final-queries", type=int, default=100)
    parser.add_argument("--expected-sqlens-build-id", required=True)
    parser.add_argument("--expected-vector-so-sha256", type=sha256_argument, required=True)
    parser.add_argument("--old-launch-manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if os.path.lexists(args.out):
            raise AuditError(f"refusing to overwrite existing output: {args.out}")
        manifest = build_manifest(args)
        atomic_write_json_exclusive(args.out, manifest)
    except (AuditError, OSError, psycopg.Error) as exc:
        print(f"audit_external_exact_truth: {exc}", file=sys.stderr)
        return 1
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
