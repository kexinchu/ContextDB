"""Figure 1 four-curve recall–latency frontier + SQL-join comparison.

(a) Mixed-selectivity 5k queries on the 200k Amazon subset.
    Systems: HNSWlib-ACORN (FAISS ACORN-1), HNSWlib-sweeping,
             PGVector-ACORN, PGVector-sweeping.
    Auto-selects the cheapest ef that meets each recall target.

(b) JOIN implementation: PostgreSQL runs the join in SQL;
    HNSWlib can only search after a SQL-materialized allow-list.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hnswlib
import numpy as np
import psycopg

from hnswlib_vs_pgvector_selectivity import (
    deactivate_pg_acorn1,
    ensure_guidance_functions,
    ensure_guidance_meta,
    pg_configure,
    timed_ms,
)
from pg_conn import pg_conninfo


def ensure_sqlens_wrappers(cur) -> None:
    ensure_guidance_functions(cur)
    cur.execute(
        "CREATE OR REPLACE FUNCTION vector_hnsw_fragment_tracking_enable(regclass) "
        "RETURNS int8 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE"
    )
    cur.execute(
        "CREATE OR REPLACE FUNCTION vector_hnsw_fragment_epoch_bump_trigger() "
        "RETURNS trigger AS 'vector' LANGUAGE C SECURITY DEFINER "
        "SET search_path = pg_catalog, pg_temp"
    )
    for fn in (
        "vector_hnsw_fragment_epoch_bump_trigger()",
        "vector_hnsw_fragment_tracking_enable(regclass)",
    ):
        try:
            cur.execute(f"ALTER EXTENSION vector ADD FUNCTION {fn}")
        except Exception:
            pass
    cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", ("amazon_fig1_200k",))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FBIN = ROOT / "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"
DEFAULT_TABLE = "amazon_fig1_200k"
DEFAULT_INDEX = "amazon_fig1_200k_hnsw"
DEFAULT_PRODUCT = "amazon_fig1_200k_product"
DEFAULT_SELECTIVITIES = [1, 2, 5, 10, 20, 50, 100]
DEFAULT_EFS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_TARGETS = [0.75, 0.80, 0.85, 0.90, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
DEFAULT_M = 32
DEFAULT_EFC = 200
SYSTEMS = ["HNSWlib-ACORN", "PGVector-ACORN", "HNSWlib-sweeping", "PGVector-sweeping"]
COLORS = {
    "HNSWlib-ACORN": "#F58518",
    "HNSWlib-sweeping": "#4C78A8",
    "PGVector-ACORN": "#F58518",
    "PGVector-sweeping": "#4C78A8",
}
MARKERS = {
    "HNSWlib-ACORN": "o",
    "HNSWlib-sweeping": "o",
    "PGVector-ACORN": "s",
    "PGVector-sweeping": "^",
}
LINESTYLES = {
    "HNSWlib-ACORN": "-",
    "HNSWlib-sweeping": "-",
    "PGVector-ACORN": "--",
    "PGVector-sweeping": "--",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_ids(text: str) -> list[int]:
    text = (text or "").strip()
    if not text:
        return []
    sep = ";" if ";" in text else ","
    return [int(x) for x in text.split(sep) if x != ""]


def read_fbin(path: Path, rows: int) -> np.ndarray:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    n = min(n, rows)
    arr = np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, d))
    return np.ascontiguousarray(arr[:n])


def vec_text(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8g}" for x in vec) + "]"


def row_pred(selectivity: int) -> str:
    if selectivity >= 100:
        return "TRUE"
    return f"bucket < {int(selectivity)}"


def activate_bucket_acorn(cur, index_name: str, selectivity: int, guidance_kind: str) -> None:
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
        (index_name, [f"sql:{row_pred(selectivity)}"], guidance_kind),
    )
    cur.execute("SET hnsw.filter_strategy = acorn1")


def overfetch_k(k: int, selectivity: int, n: int, multiplier: float = 4.0) -> int:
    return min(n, max(k, int(math.ceil(k * multiplier * 100.0 / max(selectivity, 1)))))


def knn_query_safe(index: hnswlib.Index, query: np.ndarray, k: int):
    requested = int(k)
    while requested >= 1:
        try:
            return index.knn_query(query.reshape(1, -1), k=requested, num_threads=1)
        except RuntimeError:
            if requested <= 1:
                raise
            requested = max(1, requested // 2)
    raise RuntimeError("knn_query_safe exhausted")


def knn_query_safe_filter(index: hnswlib.Index, query: np.ndarray, k: int, filt):
    requested = int(k)
    while requested >= 1:
        try:
            return index.knn_query(query.reshape(1, -1), k=requested, num_threads=1, filter=filt)
        except RuntimeError:
            if requested <= 1:
                raise
            requested = max(1, requested // 2)
    raise RuntimeError("knn_query_safe_filter exhausted")


def build_workload(n: int, queries: int, selectivities: list[int], seed: int) -> list[dict[str, int]]:
    rng = np.random.default_rng(seed)
    qids = rng.choice(n, size=queries, replace=False)
    sels = np.array([selectivities[i % len(selectivities)] for i in range(queries)])
    rng.shuffle(sels)
    return [
        {"query_no": i, "query_id": int(qid), "selectivity_pct": int(sel)}
        for i, (qid, sel) in enumerate(zip(qids, sels))
    ]


def build_ground_truth(
    vectors: np.ndarray,
    workload: list[dict[str, int]],
    k: int,
) -> dict[int, set[int]]:
    n = len(vectors)
    ids = np.arange(n, dtype=np.int64)
    out: dict[int, set[int]] = {}
    for req in workload:
        sel = int(req["selectivity_pct"])
        qid = int(req["query_id"])
        mask = np.ones(n, dtype=bool) if sel >= 100 else (ids % 100) < sel
        filtered_ids = ids[mask]
        filtered = vectors[mask]
        q = vectors[qid]
        dist = np.sum((filtered - q) ** 2, axis=1)
        order = np.argsort(dist, kind="stable")[:k]
        out[int(req["query_no"])] = {int(x) for x in filtered_ids[order]}
    return out


def setup_postgres(
    conninfo: str,
    table: str,
    index_name: str,
    product: str,
    vectors: np.ndarray,
    m: int = DEFAULT_M,
    efc: int = DEFAULT_EFC,
    reload: bool = False,
    acorn_build: bool = False,
) -> None:
    n, dim = vectors.shape
    with psycopg.connect(conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"SELECT to_regclass(%s)", (table,))
        exists = cur.fetchone()[0] is not None
        if exists and not reload:
            print(f"rebuilding HNSW {index_name} m={m} efc={efc} acorn={acorn_build} (table kept)", flush=True)
            cur.execute(f"DROP INDEX IF EXISTS {index_name}")
            acorn_opt = ", acorn = true" if acorn_build else ""
            cur.execute(
                f"""
                CREATE INDEX {index_name}
                ON {table} USING hnsw (embedding vector_l2_ops)
                WITH (m = {int(m)}, ef_construction = {int(efc)}{acorn_opt})
                """
            )
            cur.execute(f"ANALYZE {table}")
            ensure_sqlens_wrappers(cur)
            ensure_guidance_meta(cur, table)
            cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", (table,))
            print(f"pg index rebuilt m={m} efc={efc}", flush=True)
            return
        cur.execute(f"DROP TABLE IF EXISTS {product} CASCADE")
        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        cur.execute(f"CREATE TABLE {table} (id bigint PRIMARY KEY, embedding vector({dim}))")
        print(f"copying {n} rows into {table}", flush=True)
        with cur.copy(f"COPY {table} (id, embedding) FROM STDIN") as copy:
            for i in range(n):
                copy.write_row((i, vec_text(vectors[i])))
        cur.execute(f"ALTER TABLE {table} ADD COLUMN bucket int")
        cur.execute(f"UPDATE {table} SET bucket = (id % 100)")
        cur.execute(f"CREATE INDEX {table}_bucket_idx ON {table} (bucket)")
        print(f"creating HNSW index m={m} efc={efc} acorn={acorn_build}", flush=True)
        acorn_opt = ", acorn = true" if acorn_build else ""
        cur.execute(
            f"""
            CREATE INDEX {index_name}
            ON {table} USING hnsw (embedding vector_l2_ops)
            WITH (m = {int(m)}, ef_construction = {int(efc)}{acorn_opt})
            """
        )
        cur.execute(
            f"""
            CREATE TABLE {product} AS
            SELECT id, (id % 20) AS category
            FROM {table}
            """
        )
        cur.execute(f"ALTER TABLE {product} ADD PRIMARY KEY (id)")
        cur.execute(f"CREATE INDEX {product}_cat_idx ON {product} (category)")
        cur.execute(f"ANALYZE {table}")
        cur.execute(f"ANALYZE {product}")
        ensure_sqlens_wrappers(cur)
        ensure_guidance_meta(cur, table)
        cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", (table,))
        cur.execute(f"SELECT count(*) FROM {table}")
        print(f"pg rows={cur.fetchone()[0]} index={index_name}", flush=True)


def hnswlib_sweeping_run(
    index: hnswlib.Index,
    vectors: np.ndarray,
    workload: list[dict[str, int]],
    efs: list[int],
    k: int,
    overfetches: list[int] | None = None,
) -> list[dict[str, Any]]:
    n = len(vectors)
    aligned = bool(overfetches)
    knobs = list(overfetches) if aligned else list(efs)
    rows: list[dict[str, Any]] = []
    for knob in knobs:
        for req in workload:
            sel = int(req["selectivity_pct"])
            q = vectors[int(req["query_id"])]
            if aligned:
                overfetch = int(knob)
                ef = int(knob)
            else:
                overfetch = overfetch_k(k, sel, n)
                ef = max(int(knob), overfetch)
            index.set_ef(ef)
            (labels, _), elapsed = timed_ms(lambda: knn_query_safe(index, q, overfetch))
            out: list[int] = []
            for label in labels[0]:
                value = int(label)
                if sel >= 100 or (value % 100) < sel:
                    out.append(value)
                    if len(out) >= k:
                        break
            rows.append(
                {
                    "system": "HNSWlib-sweeping",
                    "ef_search": ef,
                    "query_no": req["query_no"],
                    "query_id": req["query_id"],
                    "selectivity_pct": sel,
                    "k": k,
                    "overfetch": overfetch,
                    "latency_ms": elapsed,
                    "returned": len(out),
                    "ids": ";".join(map(str, out)),
                }
            )
        print(f"HNSWlib-sweeping finished {'overfetch' if aligned else 'ef'}={knob}", flush=True)
    return rows


def pg_acorn_run(
    cur,
    table: str,
    index_name: str,
    vectors: np.ndarray,
    workload: list[dict[str, int]],
    efs: list[int],
    k: int,
    timeout_ms: int,
    guidance_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ef in efs:
        cur.execute(f"SET hnsw.ef_search = {int(ef)}")
        for req in workload:
            sel = int(req["selectivity_pct"])
            qvec = vec_text(vectors[int(req["query_id"])])
            pred = row_pred(sel)

            def run(pred=pred, qvec=qvec):
                cur.execute(
                    f"""
                    SELECT id FROM {table}
                    WHERE {pred}
                    ORDER BY embedding <-> %s::vector
                    LIMIT {int(k)}
                    """,
                    (qvec,),
                )
                return [int(r[0]) for r in cur.fetchall()]

            activate_bucket_acorn(cur, index_name, sel, guidance_kind)
            try:
                ids, elapsed = timed_ms(run)
                error = ""
            except Exception as exc:
                cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
                ids, elapsed, error = [], float(timeout_ms), type(exc).__name__
            deactivate_pg_acorn1(cur)
            rows.append(
                {
                    "system": "PGVector-ACORN",
                    "ef_search": ef,
                    "query_no": req["query_no"],
                    "query_id": req["query_id"],
                    "selectivity_pct": sel,
                    "k": k,
                    "latency_ms": elapsed,
                    "returned": len(ids),
                    "error": error,
                    "ids": ";".join(map(str, ids)),
                }
            )
        print(f"PGVector-ACORN finished ef={ef}", flush=True)
    return rows


def pg_sweeping_run(
    cur,
    table: str,
    vectors: np.ndarray,
    workload: list[dict[str, int]],
    efs: list[int],
    k: int,
    n: int,
    timeout_ms: int,
    overfetches: list[int] | None = None,
) -> list[dict[str, Any]]:
    aligned = bool(overfetches)
    knobs = list(overfetches) if aligned else list(efs)
    rows: list[dict[str, Any]] = []
    cur.execute("SET hnsw.filter_strategy = off")
    for knob in knobs:
        for req in workload:
            sel = int(req["selectivity_pct"])
            qvec = vec_text(vectors[int(req["query_id"])])
            pred = row_pred(sel)
            if aligned:
                overfetch = int(knob)
                ef = int(knob)
            else:
                overfetch = overfetch_k(k, sel, n)
                ef = int(knob)
            cur.execute(f"SET hnsw.ef_search = {int(ef)}")

            def run(pred=pred, qvec=qvec, overfetch=overfetch):
                cur.execute(
                    f"""
                    WITH candidates AS MATERIALIZED (
                      SELECT id, bucket, embedding <-> %s::vector AS dist
                      FROM {table}
                      ORDER BY embedding <-> %s::vector
                      LIMIT {int(overfetch)}
                    )
                    SELECT id FROM candidates
                    WHERE {pred}
                    ORDER BY dist
                    LIMIT {int(k)}
                    """,
                    (qvec, qvec),
                )
                return [int(r[0]) for r in cur.fetchall()]

            try:
                ids, elapsed = timed_ms(run)
                error = ""
            except Exception as exc:
                cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
                ids, elapsed, error = [], float(timeout_ms), type(exc).__name__
            rows.append(
                {
                    "system": "PGVector-sweeping",
                    "ef_search": ef,
                    "query_no": req["query_no"],
                    "query_id": req["query_id"],
                    "selectivity_pct": sel,
                    "k": k,
                    "overfetch": overfetch,
                    "latency_ms": elapsed,
                    "error": error,
                    "returned": len(ids),
                    "ids": ";".join(map(str, ids)),
                }
            )
        print(f"PGVector-sweeping finished {'overfetch' if aligned else 'ef'}={knob}", flush=True)
    return rows


def compile_acorn(root: Path) -> Path:
    binary = root / "research/acorn_faiss_mixed_frontier"
    src = root / "research/acorn_faiss_mixed_frontier.cpp"
    lib = root / "external/ACORN/build/faiss/libfaiss.a"
    if binary.exists() and binary.stat().st_mtime >= src.stat().st_mtime:
        return binary
    cmd = [
        "g++", "-O3", "-std=c++17", "-fopenmp",
        f"-I{root / 'external/ACORN'}",
        str(src),
        str(lib),
        "-o", str(binary),
        "-L/usr/lib/x86_64-linux-gnu",
        "-lmkl_intel_lp64", "-lmkl_sequential", "-lmkl_core",
        "-lz", "-ldl", "-lpthread", "-lm",
    ]
    print("compiling", binary, flush=True)
    subprocess.run(cmd, check=True)
    return binary


def run_acorn_cpp(
    root: Path,
    fbin: Path,
    workload_csv: Path,
    out_csv: Path,
    rows: int,
    k: int,
    efs: list[int],
    m: int = DEFAULT_M,
    efc: int = DEFAULT_EFC,
    gamma: int = 1,
) -> list[dict[str, Any]]:
    binary = compile_acorn(root)
    subprocess.run(
        [
            str(binary),
            "--fbin", str(fbin),
            "--workload-csv", str(workload_csv),
            "--out", str(out_csv),
            "--rows", str(rows),
            "--k", str(k),
            "--m", str(m),
            "--ef-construction", str(efc),
            "--gamma", str(gamma),
            "--ef-search-list", ",".join(map(str, efs)),
        ],
        check=True,
    )
    with out_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def attach_recall(rows: list[dict[str, Any]], gt: dict[int, set[int]], k: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("error"):
            continue
        truth = gt[int(row["query_no"])]
        ids = parse_ids(str(row.get("ids", "")))
        recall = len(set(ids[:k]) & truth) / max(len(truth), 1)
        item = dict(row)
        item["recall_at_k"] = recall
        item["ef_search"] = int(row["ef_search"])
        item["latency_ms"] = float(row["latency_ms"])
        out.append(item)
    return out


def summarize_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["system"]), int(row["ef_search"])), []).append(row)
    out = []
    for (system, ef), items in sorted(groups.items()):
        lat = [float(r["latency_ms"]) for r in items]
        rec = [float(r["recall_at_k"]) for r in items]
        out.append(
            {
                "system": system,
                "ef_search": ef,
                "queries": len(items),
                "latency_ms_mean": statistics.mean(lat),
                "latency_ms_p50": statistics.median(lat),
                "recall_at_10_mean": statistics.mean(rec),
            }
        )
    return out


def select_iso_recall(summary: list[dict[str, Any]], targets: list[float]) -> list[dict[str, Any]]:
    by_system: dict[str, list[dict[str, Any]]] = {}
    for row in summary:
        by_system.setdefault(str(row["system"]), []).append(row)
    selected = []
    for system, items in by_system.items():
        ordered = sorted(items, key=lambda r: (float(r["latency_ms_mean"]), -float(r["recall_at_10_mean"])))
        for target in targets:
            hits = [r for r in ordered if float(r["recall_at_10_mean"]) + 1e-12 >= target]
            if not hits:
                selected.append(
                    {
                        "system": system,
                        "target_recall": target,
                        "status": "unreached",
                        "ef_search": "",
                        "recall_at_10_mean": max(float(r["recall_at_10_mean"]) for r in items),
                        "latency_ms_mean": "",
                    }
                )
                continue
            best = min(hits, key=lambda r: float(r["latency_ms_mean"]))
            selected.append(
                {
                    "system": system,
                    "target_recall": target,
                    "status": "ok",
                    "ef_search": best["ef_search"],
                    "recall_at_10_mean": best["recall_at_10_mean"],
                    "latency_ms_mean": best["latency_ms_mean"],
                }
            )
    return selected


def plot_frontier(summary: list[dict[str, Any]], selected: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for system in SYSTEMS:
        items = sorted(
            [r for r in summary if r["system"] == system],
            key=lambda r: float(r["latency_ms_mean"]),
        )
        if not items:
            continue
        ax.plot(
            [float(r["latency_ms_mean"]) for r in items],
            [float(r["recall_at_10_mean"]) for r in items],
            label=system,
            color=COLORS[system],
            marker=MARKERS[system],
            linestyle=LINESTYLES[system],
            linewidth=1.4,
            markersize=4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.15, 1.02)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def run_join_compare(
    conninfo: str,
    table: str,
    index_name: str,
    product: str,
    vectors: np.ndarray,
    index: hnswlib.Index,
    query_ids: list[int],
    categories: list[int],
    efs: list[int],
    k: int,
    timeout_ms: int,
) -> list[dict[str, Any]]:
    """Compare native SQL JOIN vs HNSWlib allow-list for the same eligibility."""
    n = len(vectors)
    rows: list[dict[str, Any]] = []
    with psycopg.connect(conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        pg_configure(cur, efs[0], 200000, timeout_ms, "off", "off", 128, "off")
        ensure_sqlens_wrappers(cur)
        cur.execute("SET hnsw.guidance_require_epoch = off")
        join_pred = f"sql:id IN (SELECT id FROM {product} WHERE category < {int(categories[0])})"
        try:
            cur.execute(
                "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
                (index_name, [join_pred], "page"),
            )
            acorn_join = "accepted"
            deactivate_pg_acorn1(cur)
        except Exception as exc:
            acorn_join = f"rejected: {exc}"
        print(f"HNSWlib has no JOIN operator; ACORN guidance on JOIN predicate: {acorn_join}", flush=True)
        for cat in categories:
            pred_sql = f"p.category < {int(cat)}"
            allow_sql = f"SELECT v.id FROM {table} v JOIN {product} p ON p.id = v.id WHERE {pred_sql}"
            cur.execute(allow_sql)
            allow = {int(r[0]) for r in cur.fetchall()}
            sel_est = 100.0 * len(allow) / max(n, 1)
            print(f"join category<{cat} allow={len(allow)} sel≈{sel_est:.2f}%", flush=True)
            for ef in efs:
                cur.execute(f"SET hnsw.ef_search = {int(ef)}")
                index.set_ef(ef)
                for qno, qid in enumerate(query_ids):
                    q = vectors[qid]
                    qvec = vec_text(q)

                    def pg_join():
                        cur.execute(
                            f"""
                            SELECT v.id
                            FROM {table} v
                            JOIN {product} p ON p.id = v.id
                            WHERE {pred_sql}
                            ORDER BY v.embedding <-> %s::vector
                            LIMIT {int(k)}
                            """,
                            (qvec,),
                        )
                        return [int(r[0]) for r in cur.fetchall()]

                    cur.execute("SET hnsw.filter_strategy = off")
                    try:
                        pg_ids, pg_ms = timed_ms(pg_join)
                        pg_err = ""
                    except Exception as exc:
                        cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
                        pg_ids, pg_ms, pg_err = [], float(timeout_ms), type(exc).__name__
                    rows.append(
                        {
                            "system": "PGVector-JOIN-native",
                            "ef_search": ef,
                            "query_no": qno,
                            "query_id": qid,
                            "join_category_lt": cat,
                            "allow_list_size": len(allow),
                            "latency_ms": pg_ms,
                            "allow_ms": 0.0,
                            "search_ms": pg_ms,
                            "returned": len(pg_ids),
                            "error": pg_err,
                            "ids": ";".join(map(str, pg_ids)),
                        }
                    )

                    overfetch = overfetch_k(k, max(int(sel_est), 1), n)

                    def pg_sweep():
                        cur.execute(
                            f"""
                            WITH candidates AS MATERIALIZED (
                              SELECT v.id, v.embedding <-> %s::vector AS dist
                              FROM {table} v
                              ORDER BY v.embedding <-> %s::vector
                              LIMIT {int(overfetch)}
                            )
                            SELECT c.id
                            FROM candidates c
                            JOIN {product} p ON p.id = c.id
                            WHERE {pred_sql}
                            ORDER BY dist
                            LIMIT {int(k)}
                            """,
                            (qvec, qvec),
                        )
                        return [int(r[0]) for r in cur.fetchall()]

                    cur.execute("SET hnsw.filter_strategy = off")
                    try:
                        sw_ids, sw_ms = timed_ms(pg_sweep)
                        sw_err = ""
                    except Exception as exc:
                        cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
                        sw_ids, sw_ms, sw_err = [], float(timeout_ms), type(exc).__name__
                    rows.append(
                        {
                            "system": "PGVector-JOIN-sweeping",
                            "ef_search": ef,
                            "query_no": qno,
                            "query_id": qid,
                            "join_category_lt": cat,
                            "allow_list_size": len(allow),
                            "latency_ms": sw_ms,
                            "allow_ms": 0.0,
                            "search_ms": sw_ms,
                            "returned": len(sw_ids),
                            "error": sw_err,
                            "ids": ";".join(map(str, sw_ids)),
                        }
                    )

                    compiled = f"sql:id IN (SELECT id FROM {product} WHERE category < {int(cat)})"
                    try:
                        cur.execute(
                            "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
                            (index_name, [compiled], "page"),
                        )
                        cur.execute("SET hnsw.filter_strategy = acorn1")
                        ac_ids, ac_ms = timed_ms(pg_join)
                        ac_err = ""
                    except Exception as exc:
                        cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
                        ac_ids, ac_ms, ac_err = [], float(timeout_ms), type(exc).__name__
                    deactivate_pg_acorn1(cur)
                    rows.append(
                        {
                            "system": "PGVector-JOIN-ACORN",
                            "ef_search": ef,
                            "query_no": qno,
                            "query_id": qid,
                            "join_category_lt": cat,
                            "allow_list_size": len(allow),
                            "latency_ms": ac_ms,
                            "allow_ms": 0.0,
                            "search_ms": ac_ms,
                            "returned": len(ac_ids),
                            "error": ac_err,
                            "ids": ";".join(map(str, ac_ids)),
                        }
                    )

                    def allow_then_filter(ids_set):
                        filt = lambda label, ids_set=ids_set: int(label) in ids_set
                        (labels, _), search_ms = timed_ms(
                            lambda: knn_query_safe_filter(index, q, k, filt)
                        )
                        out_ids = [int(x) for x in labels[0] if x >= 0]
                        return out_ids, search_ms

                    t0 = time.perf_counter()
                    cur.execute(allow_sql)
                    ids_set = {int(r[0]) for r in cur.fetchall()}
                    allow_ms = (time.perf_counter() - t0) * 1000.0
                    try:
                        h_ids, search_ms = allow_then_filter(ids_set)
                        h_err = ""
                    except Exception as exc:
                        h_ids, search_ms, h_err = [], 0.0, type(exc).__name__
                    rows.append(
                        {
                            "system": "HNSWlib-JOIN-allowlist",
                            "ef_search": ef,
                            "query_no": qno,
                            "query_id": qid,
                            "join_category_lt": cat,
                            "allow_list_size": len(allow),
                            "latency_ms": allow_ms + search_ms,
                            "allow_ms": allow_ms,
                            "search_ms": search_ms,
                            "returned": len(h_ids),
                            "error": h_err,
                            "ids": ";".join(map(str, h_ids)),
                        }
                    )
                    rows.append(
                        {
                            "system": "HNSWlib-JOIN-allowlist-cached",
                            "ef_search": ef,
                            "query_no": qno,
                            "query_id": qid,
                            "join_category_lt": cat,
                            "allow_list_size": len(allow),
                            "latency_ms": search_ms,
                            "allow_ms": 0.0,
                            "search_ms": search_ms,
                            "returned": len(h_ids),
                            "error": h_err,
                            "ids": ";".join(map(str, h_ids)),
                        }
                    )
                print(f"join cat<{cat} ef={ef} done", flush=True)
    return rows


def cmd_setup(args: argparse.Namespace) -> None:
    vectors = read_fbin(args.fbin, args.rows)
    setup_postgres(
        args.conninfo, args.table, args.pg_index, args.product, vectors,
        m=args.m, efc=args.ef_construction, reload=args.reload,
        acorn_build=args.acorn_build,
    )


def cmd_run_a(args: argparse.Namespace) -> None:
    vectors = read_fbin(args.fbin, args.rows)
    n = len(vectors)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reused = args.reuse_workload
    if reused and reused.exists():
        with reused.open(newline="", encoding="utf-8") as f:
            workload = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]
        if args.queries and args.queries < len(workload):
            workload = workload[: args.queries]
        print(f"reused workload {reused} queries={len(workload)}", flush=True)
    else:
        workload = build_workload(n, args.queries, args.selectivities, args.seed)
    write_csv(out_dir / "workload.csv", workload)
    print(f"workload queries={len(workload)} sels={sorted({r['selectivity_pct'] for r in workload})}", flush=True)

    print("building exact ground truth", flush=True)
    t0 = time.perf_counter()
    gt = build_ground_truth(vectors, workload, args.k)
    print(f"ground truth elapsed_s={time.perf_counter() - t0:.1f}", flush=True)

    index_path = out_dir / f"hnswlib_fig1_200k_m{args.m}_efc{args.ef_construction}.bin"
    index = hnswlib.Index(space="l2", dim=int(vectors.shape[1]))
    if index_path.exists() and not args.reload:
        index.load_index(str(index_path), max_elements=n)
    else:
        index.init_index(max_elements=n, ef_construction=args.ef_construction, M=args.m)
        index.add_items(vectors, np.arange(n, dtype=np.int64))
        index.save_index(str(index_path))

    all_rows: list[dict[str, Any]] = []
    sweep_overfetches = args.sweep_overfetches or None
    if sweep_overfetches:
        print(f"aligned sweeping overfetches={sweep_overfetches} (ef=overfetch on both sides)", flush=True)
    if "HNSWlib-sweeping" in args.systems:
        all_rows.extend(
            hnswlib_sweeping_run(
                index, vectors, workload, args.efs, args.k,
                overfetches=sweep_overfetches,
            )
        )
    if "HNSWlib-ACORN" in args.systems:
        acorn_rows = run_acorn_cpp(
            ROOT, args.fbin, out_dir / "workload.csv", out_dir / "hnswlib_acorn_raw.csv",
            args.rows, args.k, args.efs, m=args.m, efc=args.ef_construction, gamma=args.gamma,
        )
        all_rows.extend(acorn_rows)
    if any(s.startswith("PGVector") for s in args.systems):
        with psycopg.connect(args.conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            pg_configure(cur, args.efs[0], 200000, args.timeout_ms, "off", "off", 128, "off")
            ensure_sqlens_wrappers(cur)
            ensure_guidance_meta(cur, args.table)
            cur.execute("SET hnsw.guidance_require_epoch = off")
            try:
                cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", (args.table,))
            except Exception as exc:
                print(f"fragment tracking enable skipped: {exc}", flush=True)
            if "PGVector-ACORN" in args.systems:
                all_rows.extend(
                    pg_acorn_run(
                        cur, args.table, args.pg_index, vectors, workload,
                        args.efs, args.k, args.timeout_ms, args.guidance_kind,
                    )
                )
            if "PGVector-sweeping" in args.systems:
                all_rows.extend(
                    pg_sweeping_run(
                        cur, args.table, vectors, workload, args.efs, args.k, n, args.timeout_ms,
                        overfetches=sweep_overfetches,
                    )
                )

    scored = attach_recall(all_rows, gt, args.k)
    write_csv(out_dir / "detail.csv", scored)
    summary = summarize_frontier(scored)
    write_csv(out_dir / "frontier_summary.csv", summary)
    selected = select_iso_recall(summary, args.targets)
    write_csv(out_dir / "iso_recall.csv", selected)
    plot_frontier(summary, selected, args.figure)
    print("iso-recall points:", flush=True)
    for row in selected:
        print(
            f"  {row['system']:22s} t={row['target_recall']:.2f} "
            f"status={row['status']} ef={row['ef_search']} "
            f"recall={row['recall_at_10_mean']} lat={row['latency_ms_mean']}",
            flush=True,
        )


def cmd_run_b(args: argparse.Namespace) -> None:
    vectors = read_fbin(args.fbin, args.rows)
    n = len(vectors)
    rng = np.random.default_rng(args.seed)
    query_ids = [int(x) for x in rng.choice(n, size=args.queries, replace=False)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / f"hnswlib_fig1_200k_m{args.m}_efc{args.ef_construction}.bin"
    index = hnswlib.Index(space="l2", dim=int(vectors.shape[1]))
    if index_path.exists() and not args.reload:
        index.load_index(str(index_path), max_elements=n)
    else:
        index.init_index(max_elements=n, ef_construction=args.ef_construction, M=args.m)
        index.add_items(vectors, np.arange(n, dtype=np.int64))
        index.save_index(str(index_path))
    print("building join ground truth", flush=True)
    ids = np.arange(n, dtype=np.int64)
    category = ids % 20
    gt: dict[tuple[int, int], set[int]] = {}
    for cat in args.join_categories:
        mask = category < int(cat)
        filtered_ids = ids[mask]
        filtered = vectors[mask]
        for qid in query_ids:
            dist = np.sum((filtered - vectors[qid]) ** 2, axis=1)
            order = np.argsort(dist, kind="stable")[: args.k]
            gt[(int(qid), int(cat))] = {int(x) for x in filtered_ids[order]}
    rows = run_join_compare(
        args.conninfo, args.table, args.pg_index, args.product,
        vectors, index, query_ids, args.join_categories, args.efs, args.k, args.timeout_ms,
    )
    for row in rows:
        truth = gt.get((int(row["query_id"]), int(row["join_category_lt"])), set())
        got = set(parse_ids(str(row.get("ids", ""))))
        row["recall_at_k"] = (len(got & truth) / max(len(truth), 1)) if not row.get("error") else ""
    write_csv(args.out_dir / "join_detail.csv", rows)
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("error"):
            continue
        key = (str(row["system"]), int(row["ef_search"]), int(row["join_category_lt"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (system, ef, cat), items in sorted(groups.items()):
        recs = [float(r["recall_at_k"]) for r in items if r.get("recall_at_k") != ""]
        summary.append(
            {
                "system": system,
                "ef_search": ef,
                "join_category_lt": cat,
                "queries": len(items),
                "allow_list_size": items[0]["allow_list_size"],
                "latency_ms_mean": statistics.mean(float(r["latency_ms"]) for r in items),
                "allow_ms_mean": statistics.mean(float(r["allow_ms"]) for r in items),
                "search_ms_mean": statistics.mean(float(r["search_ms"]) for r in items),
                "returned_mean": statistics.mean(float(r["returned"]) for r in items),
                "recall_at_10_mean": statistics.mean(recs) if recs else "",
            }
        )
    write_csv(args.out_dir / "join_summary.csv", summary)
    write_csv(
        args.out_dir / "join_impl.csv",
        [
            {
                "system": "HNSWlib",
                "join_support": "none",
                "how": "materialize JOIN allow-list in SQL, then hnswlib filter callback",
            },
            {
                "system": "HNSWlib-ACORN/FAISS-ACORN",
                "join_support": "none",
                "how": "labels/metadata only; JOIN must be compiled to a bit vector first",
            },
            {
                "system": "PGVector-native",
                "join_support": "SQL JOIN in one statement",
                "how": "SELECT v.id FROM reviews v JOIN product p ... ORDER BY embedding LIMIT k",
            },
            {
                "system": "PGVector-sweeping",
                "join_support": "post-filter JOIN",
                "how": "HNSW overfetch, then JOIN-filter the candidate CTE",
            },
            {
                "system": "PGVector-ACORN",
                "join_support": "compiled JOIN only",
                "how": "guidance_activate(sql:id IN (SELECT ...)) then same ORDER BY LIMIT as native JOIN",
            },
        ],
    )
    print("join summary:", flush=True)
    for row in summary:
        print(
            f"  {row['system']:24s} cat<{row['join_category_lt']} ef={row['ef_search']} "
            f"e2e={row['latency_ms_mean']:.2f} allow={row['allow_ms_mean']:.2f} "
            f"search={row['search_ms_mean']:.2f} recall={row['recall_at_10_mean']}",
            flush=True,
        )


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["setup", "run-a", "run-b"])
    parser.add_argument("--conninfo", default=None)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--pg-index", default=DEFAULT_INDEX)
    parser.add_argument("--product", default=DEFAULT_PRODUCT)
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--rows", type=int, default=200000)
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--selectivities", default=",".join(map(str, DEFAULT_SELECTIVITIES)))
    parser.add_argument("--efs", default=",".join(map(str, DEFAULT_EFS)))
    parser.add_argument(
        "--sweep-overfetches",
        default="",
        help="If set, sweeping uses this overfetch grid and sets ef=overfetch on both HNSWlib and PG. "
        "ACORN still uses --efs. Empty keeps the legacy mixed-selectivity knobs.",
    )
    parser.add_argument("--targets", default=",".join(map(str, DEFAULT_TARGETS)))
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--guidance-kind", default="page")
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--ef-construction", type=int, default=DEFAULT_EFC)
    parser.add_argument("--gamma", type=int, default=1)
    parser.add_argument("--acorn-build", action="store_true", help="Build PG HNSW index with acorn=true (auto-enabled for PGVector-ACORN in setup)")
    parser.add_argument("--reuse-workload", type=Path, default=None)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--join-categories", default="1,2,4,10")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "research/results/fig1_four_curve_m32")
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "paper/figures/fig_intro_recall_latency_frontier.pdf",
    )
    args = parser.parse_args()
    if not args.conninfo:
        args.conninfo = pg_conninfo("55438")
    args.selectivities = parse_ints(args.selectivities)
    args.efs = parse_ints(args.efs)
    args.sweep_overfetches = parse_ints(args.sweep_overfetches) if args.sweep_overfetches else []
    args.targets = parse_floats(args.targets)
    args.systems = [x.strip() for x in args.systems.split(",") if x.strip()]
    args.join_categories = parse_ints(args.join_categories)
    if args.command == "setup" and "PGVector-ACORN" in args.systems:
        args.acorn_build = True
    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "run-a":
        cmd_run_a(args)
    else:
        cmd_run_b(args)


if __name__ == "__main__":
    main()
