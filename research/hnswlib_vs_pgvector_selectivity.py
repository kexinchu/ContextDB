from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hnswlib
import numpy as np
import psycopg
from psycopg import errors

from pg_conn import pg_conninfo


DEFAULT_TABLE = "amazon_grocery_reviews_10m_pgvector_id_order_200k"
DEFAULT_INDEX = "amazon_grocery_reviews_10m_pgvector_id_order_200k_embedding_hns"
DEFAULT_SELECTIVITIES = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def flatten_profile(prefix: str, text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        profile = json.loads(text)
    except json.JSONDecodeError:
        return {f"{prefix}_raw": text}
    return {f"{prefix}_{key}": value for key, value in profile.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_vec(text: str) -> list[float]:
    return [float(x) for x in text.strip("[]").split(",")]


def load_query_ids(path: Path, queries: int) -> list[int]:
    ids: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["query_id"]))
            if len(ids) >= queries:
                break
    return ids


def ensure_query_csv(cur, table: str, path: Path, queries: int) -> None:
    if path.exists():
        existing = load_query_ids(path, queries)
        if len(existing) >= queries:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    cur.execute(f"SELECT id FROM {table} ORDER BY md5(id::text) LIMIT %s", (queries,))
    ids = [int(r[0]) for r in cur.fetchall()]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id"])
        writer.writerows([[x] for x in ids])


def load_table_vectors(cur, table: str) -> tuple[np.ndarray, np.ndarray]:
    cur.execute(f"SELECT id, embedding::text FROM {table} ORDER BY id")
    ids: list[int] = []
    vectors: list[list[float]] = []
    for row_id, vec in cur:
        ids.append(int(row_id))
        vectors.append(parse_vec(str(vec)))
    return np.asarray(ids, dtype=np.int64), np.asarray(vectors, dtype=np.float32)


def load_query_vectors(cur, table: str, query_ids: list[int]) -> dict[int, str]:
    cur.execute(
        f"""
        SELECT id, embedding::text
        FROM {table}
        WHERE id = ANY(%s::bigint[])
        """,
        (query_ids,),
    )
    out = {int(row[0]): str(row[1]) for row in cur.fetchall()}
    missing = [qid for qid in query_ids if qid not in out]
    if missing:
        raise RuntimeError(f"missing query vectors from {table}: {missing[:5]}")
    return out


def build_or_load_hnsw(index_path: Path, ids: np.ndarray, vectors: np.ndarray, m: int, ef_construction: int, ef_search: int) -> hnswlib.Index:
    dim = int(vectors.shape[1])
    index = hnswlib.Index(space="l2", dim=dim)
    if index_path.exists():
        index.load_index(str(index_path), max_elements=len(ids))
    else:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index.init_index(max_elements=len(ids), ef_construction=ef_construction, M=m)
        index.add_items(vectors, ids)
        index.save_index(str(index_path))
    index.set_ef(ef_search)
    return index


def filter_sql(selectivity: int) -> str:
    if selectivity >= 100:
        return "TRUE"
    return f"(id %% 100) < {int(selectivity)}"


def pg_configure(
    cur,
    ef_search: int,
    max_scan_tuples: int,
    statement_timeout_ms: int,
    iterative_scan: str,
    page_access: str,
    page_window: int,
    index_page_access: str,
) -> None:
    cur.execute("SET jit = off")
    cur.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(max_scan_tuples)}")
    cur.execute("SET hnsw.scan_mem_multiplier = 8")
    cur.execute("SET enable_sort = off")
    cur.execute("SET enable_bitmapscan = off")
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute(f"SET hnsw.page_access = {page_access}")
    cur.execute(f"SET hnsw.page_window = {int(page_window)}")
    cur.execute(f"SET hnsw.index_page_access = {index_page_access}")


def ensure_guidance_functions(cur) -> None:
    function_sql = [
        "CREATE OR REPLACE FUNCTION vector_hnsw_last_scan_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_reset_scan_profile() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_activate(regclass, text[], text) "
        "RETURNS int4 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
    ]
    for sql in function_sql:
        cur.execute(sql)


def ensure_guidance_meta(cur, table: str) -> None:
    meta = f"{table}_guidance_meta"
    cur.execute("SELECT to_regclass(%s)", (meta,))
    exists = cur.fetchone()[0] is not None
    rebuild = not exists
    if exists:
        cur.execute(f"SELECT count(*) FROM {meta}")
        rebuild = int(cur.fetchone()[0]) == 0
    if rebuild:
        if exists:
            cur.execute(f"DROP TABLE {meta}")
        cur.execute(
            f"""
            CREATE UNLOGGED TABLE {meta} AS
            SELECT ctid AS heap_tid, id
            FROM {table}
            """
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS hr_gm_id_idx ON {meta} (id)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS hr_gm_tid_idx ON {meta} (heap_tid)")
    cur.execute(f"ANALYZE {meta}")


def pg_iterative(cur, table: str, qvec: str, selectivity: int, k: int, timeout_ms: int) -> tuple[list[int], float, str]:
    pred = filter_sql(selectivity)

    def run():
        cur.execute(
            f"""
            SELECT id
            FROM {table}
            WHERE {pred}
            ORDER BY embedding <-> %s::vector
            LIMIT {int(k)}
            """,
            (qvec,),
        )
        return [int(r[0]) for r in cur.fetchall()]

    try:
        ids, elapsed = timed_ms(run)
        return ids, elapsed, ""
    except errors.QueryCanceled as exc:
        cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
        return [], float(timeout_ms), exc.__class__.__name__


def activate_pg_acorn1(cur, index_name: str, selectivity: int, guidance_kind: str) -> None:
    guidance_pred = "TRUE" if selectivity >= 100 else f"(id % 100) < {int(selectivity)}"
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
        (index_name, [f"sql:{guidance_pred}"], guidance_kind),
    )
    cur.execute("SET hnsw.filter_strategy = acorn1")


def deactivate_pg_acorn1(cur) -> None:
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute("SELECT vector_hnsw_guidance_reset()")


def fetch_one_text(cur, sql: str) -> str:
    cur.execute(sql)
    value = cur.fetchone()[0]
    return str(value)


def pg_acorn1(
    cur,
    table: str,
    index_name: str,
    qvec: str,
    selectivity: int,
    k: int,
    timeout_ms: int,
    collect_profile: bool = False,
    guidance_kind: str = "exact",
) -> tuple[list[int], float, str, dict[str, Any]]:
    pred = filter_sql(selectivity)

    def run():
        cur.execute(
            f"""
            SELECT id
            FROM {table}
            WHERE {pred}
            ORDER BY embedding <-> %s::vector
            LIMIT {int(k)}
            """,
            (qvec,),
        )
        return [int(r[0]) for r in cur.fetchall()]

    profile: dict[str, Any] = {}
    try:
        activate_pg_acorn1(cur, index_name, selectivity, guidance_kind)
        if collect_profile:
            cur.execute("SELECT vector_hnsw_reset_scan_profile()")
            profile.update(flatten_profile("guidance", fetch_one_text(cur, "SELECT vector_hnsw_guidance_profile()")))
        ids, elapsed = timed_ms(run)
        if collect_profile:
            profile.update(flatten_profile("scan", fetch_one_text(cur, "SELECT vector_hnsw_last_scan_profile()")))
        deactivate_pg_acorn1(cur)
        return ids, elapsed, "", profile
    except errors.QueryCanceled as exc:
        cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
        if collect_profile:
            try:
                profile.update(flatten_profile("scan", fetch_one_text(cur, "SELECT vector_hnsw_last_scan_profile()")))
            except Exception:
                pass
        deactivate_pg_acorn1(cur)
        return [], float(timeout_ms), exc.__class__.__name__, profile


def pg_sweeping(cur, table: str, qvec: str, selectivity: int, k: int, overfetch: int, timeout_ms: int) -> tuple[list[int], float, str]:
    pred = filter_sql(selectivity)

    def run():
        cur.execute(
            f"""
            WITH candidates AS MATERIALIZED (
              SELECT id, embedding <-> %s::vector AS dist
              FROM {table}
              ORDER BY embedding <-> %s::vector
              LIMIT {int(overfetch)}
            )
            SELECT id
            FROM candidates
            WHERE {pred}
            ORDER BY dist
            LIMIT {int(k)}
            """,
            (qvec, qvec),
        )
        return [int(r[0]) for r in cur.fetchall()]

    try:
        ids, elapsed = timed_ms(run)
        return ids, elapsed, ""
    except errors.QueryCanceled as exc:
        cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
        return [], float(timeout_ms), exc.__class__.__name__


def hnswlib_filtered(index: hnswlib.Index, query: np.ndarray, selectivity: int, k: int) -> tuple[list[int], float, str]:
    if selectivity >= 100:
        filt = None
    else:
        filt = lambda label: (int(label) % 100) < int(selectivity)
    try:
        (labels, _), elapsed = timed_ms(lambda: index.knn_query(query.reshape(1, -1), k=k, filter=filt, num_threads=1))
        return [int(x) for x in labels[0] if x >= 0], elapsed, ""
    except RuntimeError as exc:
        return [], 0.0, exc.__class__.__name__


def hnswlib_sweeping(index: hnswlib.Index, query: np.ndarray, selectivity: int, k: int, overfetch: int) -> tuple[list[int], float, str]:
    try:
        (labels, _), elapsed = timed_ms(lambda: index.knn_query(query.reshape(1, -1), k=overfetch, num_threads=1))
        out: list[int] = []
        for label in labels[0]:
            value = int(label)
            if selectivity >= 100 or (value % 100) < selectivity:
                out.append(value)
                if len(out) >= k:
                    break
        return out, elapsed, ""
    except RuntimeError as exc:
        return [], 0.0, exc.__class__.__name__


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["system"]), int(row["selectivity_pct"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (system, sel), items in sorted(groups.items()):
        ok = [r for r in items if not r["error"]]
        vals = [float(r["latency_ms"]) for r in ok]
        returned = [float(r["returned"]) for r in ok]
        numeric_keys = [
            key
            for key in ok[0]
            if key.startswith(("scan_", "guidance_")) and isinstance(ok[0][key], (int, float, bool))
        ]
        profile_means = {
            f"{key}_mean": statistics.mean(float(r[key]) for r in ok if key in r and r[key] not in ("", None))
            for key in numeric_keys
            if any(key in r and r[key] not in ("", None) for r in ok)
        }
        if not vals:
            continue
        row = {
            "system": system,
            "selectivity_pct": sel,
            "repeats": len({int(r.get("repeat", 0)) for r in ok}),
            "queries": len(ok),
            "attempts": len(items),
            "errors": len(items) - len(ok),
            "latency_ms_mean": statistics.mean(vals),
            "latency_ms_p50": statistics.median(vals),
            "latency_ms_p95": sorted(vals)[max(0, int(0.95 * len(vals)) - 1)],
            "returned_mean": statistics.mean(returned),
            "full_k_rate": sum(1 for x in returned if x >= float(items[0]["k"])) / len(ok),
        }
        row.update(profile_means)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conninfo", default=None)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--pg-index", default=DEFAULT_INDEX)
    parser.add_argument("--query-id-csv", type=Path, default=Path("research/results/amazon_200k_query_ids_100.csv"))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--selectivities", default=",".join(str(x) for x in DEFAULT_SELECTIVITIES))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--pg-max-scan-tuples", type=int, default=50000)
    parser.add_argument("--pg-iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--statement-timeout-ms", type=int, default=5000)
    parser.add_argument("--pg-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--pg-page-window", type=int, default=128)
    parser.add_argument("--pg-index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--pg-guidance-kind", default="exact", choices=["exact", "page", "bloom"])
    parser.add_argument("--overfetch-multiplier", type=float, default=4.0)
    parser.add_argument("--index-path", type=Path, default=Path("research/results/hnswlib_amazon_200k_m16_ef64.bin"))
    parser.add_argument("--systems", default="HNSWlib-filtered,HNSWlib-sweeping,PGVector-ACORN1,PGVector-iterative,PGVector-sweeping")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--collect-pg-profile", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("research/results/hnswlib_vs_pgvector_200k_q100.csv"))
    args = parser.parse_args()
    if not args.conninfo:
        args.conninfo = pg_conninfo("55432")

    selectivities = [int(x) for x in args.selectivities.split(",") if x]
    requested_systems = {x.strip() for x in args.systems.split(",") if x.strip()}
    rows: list[dict[str, Any]] = []
    with psycopg.connect(args.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        ensure_query_csv(cur, args.table, args.query_id_csv, args.queries)
        query_ids = load_query_ids(args.query_id_csv, args.queries)
        pg_configure(
            cur,
            args.ef_search,
            args.pg_max_scan_tuples,
            args.statement_timeout_ms,
            args.pg_iterative_scan,
            args.pg_page_access,
            args.pg_page_window,
            args.pg_index_page_access,
        )
        ensure_guidance_functions(cur)
        ensure_guidance_meta(cur, args.table)
        query_texts = load_query_vectors(cur, args.table, query_ids)
        needs_hnswlib = any(system.startswith("HNSWlib") for system in requested_systems)
        ids: np.ndarray | None = None
        vectors: np.ndarray | None = None
        index: hnswlib.Index | None = None
        id_to_pos: dict[int, int] = {}
        if needs_hnswlib:
            ids, vectors = load_table_vectors(cur, args.table)
            index = build_or_load_hnsw(args.index_path, ids, vectors, args.m, args.ef_construction, args.ef_search)
            id_to_pos = {int(row_id): i for i, row_id in enumerate(ids)}

        for sel in selectivities:
            table_size = len(ids) if ids is not None else 10_000_000
            overfetch = min(table_size, max(args.k, int(np.ceil(args.k * args.overfetch_multiplier * 100.0 / max(sel, 1)))))
            for qno, qid in enumerate(query_ids):
                qvec_text = query_texts[qid]
                qvec = vectors[id_to_pos[qid]] if vectors is not None else None
                all_runs = {
                    "HNSWlib-filtered": lambda: hnswlib_filtered(index, qvec, sel, args.k),
                    "HNSWlib-sweeping": lambda: hnswlib_sweeping(index, qvec, sel, args.k, overfetch),
                    "PGVector-ACORN1": lambda: pg_acorn1(
                        cur,
                        args.table,
                        args.pg_index,
                        qvec_text,
                        sel,
                        args.k,
                        args.statement_timeout_ms,
                        args.collect_pg_profile,
                        args.pg_guidance_kind,
                    ),
                    "PGVector-iterative": lambda: pg_iterative(cur, args.table, qvec_text, sel, args.k, args.statement_timeout_ms),
                    "PGVector-sweeping": lambda: pg_sweeping(cur, args.table, qvec_text, sel, args.k, overfetch, args.statement_timeout_ms),
                }
                runs = [(system, all_runs[system]) for system in all_runs if system in requested_systems]
                for repeat in range(args.repeats):
                    for system, fn in runs:
                        result = fn()
                        if len(result) == 4:
                            result_ids, elapsed, error, profile = result
                        else:
                            result_ids, elapsed, error = result
                            profile = {}
                        row = {
                            "system": system,
                            "repeat": repeat,
                            "query_no": qno,
                            "query_id": qid,
                            "selectivity_pct": sel,
                            "k": args.k,
                            "overfetch": overfetch,
                            "latency_ms": elapsed,
                            "returned": len(result_ids),
                            "error": error,
                            "ids": ",".join(map(str, result_ids)),
                        }
                        row.update(profile)
                        rows.append(row)
                        print(
                            f"{system} rep={repeat} sel={sel} q={qno} ms={elapsed:.2f} "
                            f"ret={len(result_ids)} overfetch={overfetch} err={error}",
                            flush=True,
                        )

    write_csv(args.out, rows)
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summarize(rows))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
