"""Audit HNSW query health before admitting an index to formal experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import psycopg

try:
    from .common_pg import pg_config_from_env
except ImportError:
    from common_pg import pg_config_from_env


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
DEFAULT_COHORT = ROOT / "results/hybrid_vector_db/amazon10m_unique_embedding_query_cohort_q10200.csv"
DEFAULT_COHORT_MANIFEST = ROOT / "results/hybrid_vector_db/amazon10m_unique_embedding_query_cohort_q10200_manifest.json"
DEFAULT_OUT = ROOT / "results/hybrid_vector_db/amazon10m_hnsw_query_health.csv"
DEFAULT_MANIFEST = ROOT / "results/hybrid_vector_db/amazon10m_hnsw_query_health_manifest.json"
DEFAULT_INDEXES = (
    "public.amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx",
)
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


class HealthAuditError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualified_identifier(value: str) -> str:
    parts = str(value).split(".")
    if len(parts) not in (1, 2) or any(not IDENTIFIER_RE.fullmatch(part) for part in parts):
        raise argparse.ArgumentTypeError("expected an unquoted relation or schema.relation")
    return ".".join(part.lower() for part in parts)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def declared_output_hash(manifest: Mapping[str, Any], filename: str) -> str | None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return None
    for value in outputs.values():
        if not isinstance(value, Mapping):
            continue
        if Path(str(value.get("path", ""))).name == filename:
            sha = value.get("sha256")
            return str(sha) if sha else None
    return None


def load_cohort(
    cohort_path: Path,
    manifest_path: Path,
    split: str,
    expected_queries: int,
    query_no_start: int | None = None,
    query_no_end_exclusive: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        query_no_start is not None
        and query_no_end_exclusive is not None
        and query_no_start >= query_no_end_exclusive
    ):
        raise HealthAuditError("query-number slice must have a positive width")
    if not cohort_path.is_file() or not manifest_path.is_file():
        raise HealthAuditError("query cohort and cohort manifest must both exist")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HealthAuditError("query cohort manifest is unreadable") from exc
    if not isinstance(manifest, Mapping) or manifest.get("artifact_valid") is not True:
        raise HealthAuditError("query cohort manifest must declare artifact_valid=true")
    observed_hash = sha256_file(cohort_path)
    if declared_output_hash(manifest, cohort_path.name) != observed_hash:
        raise HealthAuditError("query cohort SHA256 does not match its manifest")
    with cohort_path.open(newline="", encoding="utf-8") as source:
        raw_rows = list(csv.DictReader(source))
    required = {
        "query_no",
        "query_id",
        "query_split",
        "candidate_validity_predicate",
        "query_validity_predicate",
    }
    if not raw_rows or not required.issubset(raw_rows[0]):
        raise HealthAuditError("query cohort schema is incomplete")
    selected: list[dict[str, Any]] = []
    for row in raw_rows:
        if split != "all" and row["query_split"] != split:
            continue
        query_no = int(row["query_no"])
        if query_no_start is not None and query_no < query_no_start:
            continue
        if query_no_end_exclusive is not None and query_no >= query_no_end_exclusive:
            continue
        if row["candidate_validity_predicate"] != "embedding_valid":
            raise HealthAuditError("query cohort candidate universe is not embedding_valid")
        if row["query_validity_predicate"] != "embedding_valid":
            raise HealthAuditError("query cohort query universe is not embedding_valid")
        selected.append(
            {
                "query_no": query_no,
                "query_id": int(row["query_id"]),
                "query_split": row["query_split"],
            }
        )
    if len(selected) != expected_queries:
        raise HealthAuditError(
            f"expected {expected_queries} {split} queries, observed {len(selected)}"
        )
    query_nos = [row["query_no"] for row in selected]
    query_ids = [row["query_id"] for row in selected]
    if len(set(query_nos)) != len(query_nos) or len(set(query_ids)) != len(query_ids):
        raise HealthAuditError("query cohort contains duplicate query numbers or IDs")
    if query_no_start is not None and query_no_end_exclusive is not None:
        if query_nos != list(range(query_no_start, query_no_end_exclusive)):
            raise HealthAuditError("query cohort does not exactly cover the requested query-number slice")
    return selected, {
        "cohort_csv": str(cohort_path.resolve()),
        "cohort_csv_sha256": observed_hash,
        "cohort_manifest": str(manifest_path.resolve()),
        "cohort_manifest_sha256": sha256_file(manifest_path),
        "split": split,
        "queries": len(selected),
        "query_no_start": query_no_start,
        "query_no_end_exclusive": query_no_end_exclusive,
    }


def decode_json(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise HealthAuditError("server profile did not return a JSON object")
    return dict(value)


def plan_index_names(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.lstrip()
        if not stripped.startswith(("[", "{")):
            return []
        value = json.loads(value)
    names: list[str] = []
    if isinstance(value, Mapping):
        if value.get("Index Name"):
            names.append(str(value["Index Name"]))
        for child in value.values():
            names.extend(plan_index_names(child))
    elif isinstance(value, list):
        for child in value:
            names.extend(plan_index_names(child))
    return names


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_rows(rows: Sequence[Mapping[str, Any]], indexes: Sequence[str], k: int) -> dict[str, Any]:
    by_index: dict[str, Any] = {}
    all_valid = bool(rows)
    for index in indexes:
        selected = [row for row in rows if row.get("index") == index]
        latencies = [float(row["latency_ms"]) for row in selected]
        exhausted = [int(row["query_no"]) for row in selected if int(row.get("exhausted_terminations", 0)) > 0]
        incomplete = [int(row["query_no"]) for row in selected if int(row.get("returned", -1)) != k]
        errors = [int(row["query_no"]) for row in selected if row.get("error")]
        plan_failures = [int(row["query_no"]) for row in selected if row.get("plan_index_verified") is not True]
        semantic_failures = [
            int(row["query_no"])
            for row in selected
            if row.get("ids_unique") is not True
            or row.get("self_excluded") is not True
            or row.get("profile_valid") is not True
            or row.get("profile_final_path") != "stock"
        ]
        valid = (
            bool(selected)
            and not exhausted
            and not incomplete
            and not errors
            and not plan_failures
            and not semantic_failures
        )
        all_valid = all_valid and valid
        by_index[index] = {
            "queries": len(selected),
            "valid": valid,
            "exhausted_queries": exhausted,
            "incomplete_topk_queries": incomplete,
            "error_queries": errors,
            "plan_failure_queries": plan_failures,
            "semantic_failure_queries": semantic_failures,
            "visited_min": min((int(row.get("visited_tuples", 0)) for row in selected), default=0),
            "visited_mean": statistics.fmean(
                int(row.get("visited_tuples", 0)) for row in selected
            ) if selected else 0.0,
            "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
            "latency_p95_ms": percentile(latencies, 0.95),
            "latency_p99_ms": percentile(latencies, 0.99),
        }
    expected = len(rows) // max(len(indexes), 1)
    if any(item["queries"] != expected for item in by_index.values()):
        all_valid = False
    return {"artifact_valid": all_valid, "queries_per_index": expected, "indexes": by_index}


def render_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def relation_identity(cur: psycopg.Cursor, name: str, table: str) -> dict[str, Any]:
    cur.execute(
        "SELECT c.oid::bigint, c.relfilenode::bigint, pg_relation_size(c.oid)::bigint, "
        "pg_get_indexdef(c.oid), obj_description(c.oid, 'pg_class'), am.amname, "
        "i.indisvalid, i.indisready, i.indrelid::regclass::text "
        "FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid "
        "JOIN pg_am am ON am.oid=c.relam "
        "WHERE c.oid=to_regclass(%s) AND i.indrelid=to_regclass(%s)",
        (name, table),
    )
    row = cur.fetchone()
    if row is None:
        raise HealthAuditError(f"HNSW index does not exist on {table}: {name}")
    identity = {
        "name": name,
        "oid": int(row[0]),
        "relfilenode": int(row[1]),
        "bytes": int(row[2]),
        "definition": str(row[3]),
        "comment": None if row[4] is None else str(row[4]),
        "access_method": str(row[5]),
        "valid": bool(row[6]),
        "ready": bool(row[7]),
        "heap": str(row[8]),
    }
    if identity["access_method"] != "hnsw" or not identity["valid"] or not identity["ready"]:
        raise HealthAuditError(f"index is not a valid ready HNSW index: {name}")
    return identity


def runtime_identity(cur: psycopg.Cursor, table: str) -> dict[str, Any]:
    cur.execute(
        "WITH lib AS (SELECT setting || '/vector.so' AS path FROM pg_config WHERE name='PKGLIBDIR') "
        "SELECT vector_sqlens_build_id(), lib.path, "
        "encode(sha256(pg_read_binary_file(lib.path)), 'hex'), "
        "t.oid::bigint, t.relfilenode::bigint FROM lib, pg_class t WHERE t.oid=to_regclass(%s)",
        (table,),
    )
    row = cur.fetchone()
    if row is None:
        raise HealthAuditError("could not bind SQLens binary and table identity")
    return {
        "sqlens_build_id": str(row[0]),
        "vector_so_path": str(row[1]),
        "vector_so_sha256": str(row[2]),
        "table": table,
        "table_oid": int(row[3]),
        "table_relfilenode": int(row[4]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cohort, cohort_identity = load_cohort(
        args.cohort,
        args.cohort_manifest,
        args.query_split,
        args.expected_queries,
        args.query_no_start,
        args.query_no_end_exclusive,
    )
    indexes = tuple(args.index)
    query_sql = (
        f"SELECT id FROM {args.table} WHERE embedding_valid AND id <> %s "
        f"ORDER BY embedding <-> (SELECT embedding FROM {args.table} WHERE id = %s) "
        f"LIMIT {args.k}"
    )
    explain_sql = "EXPLAIN (FORMAT JSON, COSTS OFF) " + query_sql
    rows: list[dict[str, Any]] = []
    started_at = utc_now()
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        runtime = runtime_identity(cur, args.table)
        index_identities = {
            index: relation_identity(cur, index, args.table) for index in indexes
        }
        cur.execute("SET jit=off")
        cur.execute("SET enable_sort=off")
        cur.execute("SET hnsw.filter_strategy=off")
        cur.execute("SET hnsw.iterative_scan=off")
        cur.execute(
            "SELECT set_config('hnsw.ef_search', %s, false)",
            (str(args.ef_search),),
        )
        for index in indexes:
            cur.execute("SELECT set_config('hnsw.preferred_index', %s, false)", (index,))
            for query in cohort:
                query_id = int(query["query_id"])
                cur.execute(explain_sql, (query_id, query_id))
                plan_value = cur.fetchone()[0]
                observed_plan_indexes = plan_index_names(plan_value)
                expected_short_name = index.rsplit(".", 1)[-1]
                plan_verified = expected_short_name in observed_plan_indexes
                cur.execute("SELECT vector_hnsw_reset_scan_profile()")
                began = time.perf_counter()
                error = ""
                ids: list[int] = []
                try:
                    cur.execute(query_sql, (query_id, query_id))
                    ids = [int(row[0]) for row in cur.fetchall()]
                except Exception as exc:  # The row remains in the fail-closed artifact.
                    error = exc.__class__.__name__
                latency_ms = (time.perf_counter() - began) * 1000.0
                cur.execute("SELECT vector_hnsw_last_scan_profile()")
                profile = decode_json(cur.fetchone()[0])
                ids_unique = len(ids) == len(set(ids))
                self_excluded = query_id not in ids
                rows.append(
                    {
                        "index": index,
                        "index_oid": index_identities[index]["oid"],
                        "query_no": query["query_no"],
                        "query_id": query_id,
                        "query_split": query["query_split"],
                        "ef_search": args.ef_search,
                        "returned": len(ids),
                        "returned_ids": ",".join(str(value) for value in ids),
                        "ids_unique": ids_unique,
                        "self_excluded": self_excluded,
                        "latency_ms": latency_ms,
                        "visited_tuples": int(profile.get("visited_tuples", 0)),
                        "expanded_nodes": int(profile.get("expanded_nodes", 0)),
                        "distance_computations": int(profile.get("distance_computations", 0)),
                        "exhausted_terminations": int(profile.get("traversal_exhausted_terminations", 0)),
                        "max_scan_terminations": int(profile.get("traversal_max_scan_terminations", 0)),
                        "profile_valid": bool(profile.get("valid", False)),
                        "profile_final_path": str(profile.get("final_path", "")),
                        "plan_index_verified": plan_verified,
                        "plan_indexes": ",".join(observed_plan_indexes),
                        "error": error,
                    }
                )
        cur.close()
    summary = summarize_rows(rows, indexes, args.k)
    summary["artifact_valid"] = bool(
        summary["artifact_valid"]
        and all(item["queries"] == len(cohort) for item in summary["indexes"].values())
        and all(row["profile_valid"] for row in rows)
    )
    atomic_write(args.out, render_csv(rows))
    payload = {
        "artifact_contract": "sqlens_hnsw_query_health_v1",
        "artifact_valid": summary["artifact_valid"],
        "started_at": started_at,
        "completed_at": utc_now(),
        "inputs": cohort_identity,
        "runtime": runtime,
        "index_identities": index_identities,
        "settings": {
            "table": args.table,
            "indexes": list(indexes),
            "query_split": args.query_split,
            "query_no_start": args.query_no_start,
            "query_no_end_exclusive": args.query_no_end_exclusive,
            "k": args.k,
            "ef_search": args.ef_search,
            "iterative_scan": "off",
            "filter_strategy": "off",
            "candidate_validity_predicate": "embedding_valid",
            "query_shape": "unfiltered_partial_index_self_excluded_topk",
        },
        "summary": summary,
        "outputs": {
            "csv": {
                "path": str(args.out.resolve()),
                "rows": len(rows),
                "sha256": sha256_file(args.out),
            }
        },
    }
    atomic_write(args.manifest, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=qualified_identifier, default=DEFAULT_TABLE)
    parser.add_argument("--index", type=qualified_identifier, action="append")
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    parser.add_argument("--query-split", choices=("calibration", "final", "all"), default="final")
    parser.add_argument("--expected-queries", type=positive_int, default=10_000)
    parser.add_argument("--query-no-start", type=nonnegative_int, default=200)
    parser.add_argument("--query-no-end-exclusive", type=positive_int, default=10_200)
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--ef-search", type=positive_int, default=1_000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.index:
        args.index = list(DEFAULT_INDEXES)
    if len(set(args.index)) != len(args.index):
        raise SystemExit("--index values must be distinct")
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["artifact_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
