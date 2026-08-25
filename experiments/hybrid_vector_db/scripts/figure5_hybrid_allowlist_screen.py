#!/usr/bin/env python3
"""q1K screen: SQL hybrid search vs stock / SQLens / FAISS allow-list.

Not paper-eligible. Four SQL shapes share grocery_helpful (~1%) and the
frozen q200..q1199 cohort. FAISS e2e includes allow-list SQL + bitmap +
HNSW search. Panel (a) is grouped stacked bars; (b) amortizes Catalog JOIN
allow-list over N shared queries.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import amazon10m_sql_native_benchmark as bench
import amazon10m_sql_native_exact_truth as truth
from amazon10m_matched_recall_baselines import (
    read_fbin_memmap,
    search_faiss,
    set_bitmap_ids,
)
from common_pg import pg_config_from_env, require_psycopg

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/hybrid_vector_db"
FILTER_NAME = "grocery_helpful"
RETUNE_FILTERS = ("grocery_helpful", "grocery_long500")
QUERY_OFFSET = 200
QUERY_COUNT = 1_000
K = 10
EF_PG = 100
# Stay at or above the paper's P0 min ef. Cheaper efs make stock already
# hit 0.90 while SQLens still pays the D3/activation tax, so speedup falls.
PG_EF_GRID = (100, 150, 200, 250, 400)
CALIB_QUERY_NOS = tuple(range(20, 100))
WARMUP_QUERY_NOS = tuple(range(20, 70))
FAISS_EF_GRID = (2000, 4000, 8000, 16000, 32000, 50000, 100000)
AMORTIZE_NS = (1, 10, 100, 1000)
TABLE = bench.DEFAULT_VECTOR_TABLE
SOURCE_INDEX = bench.DEFAULT_SOURCE_INDEX
CLONE_INDEX = bench.DEFAULT_CLONE_INDEX
PRINCIPAL = bench.DEFAULT_PRINCIPAL
FAISS_INDEX = ROOT / "data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m32_efc200_seed57_t16.index"
FBIN = ROOT / "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"
JOIN_GT = (
    RESULTS
    / "amazon10m_sql_native_q10200_r43_sqlops_join"
    / "amazon10m_sql_native_exact_truth_q10200.csv"
)
ATTR_GT = RESULTS / "amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal.csv"
OUT_DIR = RESULTS / "figure5_hybrid_allowlist_q1k_screen"

SHAPES = (
    ("attributes", "Attributes", "none"),
    ("join_facts", "Facts JOIN", "facts"),
    ("join_catalog", "Catalog JOIN", "catalog"),
    ("join_acl", "ACL JOIN", "acl"),
)


def _workload(name: str, join_kind: str) -> bench.WorkloadSpec:
    return bench.WorkloadSpec(
        name,
        f"hybrid search shape {name}",
        50.0,
        False,
        "base",
        "",
        "none",
        join_kind,
    )


def _parse_ids(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(text).split(",") if part.strip())


def load_truth(
    filter_name: str = FILTER_NAME,
) -> tuple[dict[int, int], dict[str, dict[int, tuple[int, ...]]], int]:
    query_ids: dict[int, int] = {}
    by_shape: dict[str, dict[int, tuple[int, ...]]] = {name: {} for name, _, _ in SHAPES}
    as_of = 0
    with JOIN_GT.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["filter_name"] != filter_name:
                continue
            query_no = int(row["query_no"])
            query_id = int(row["query_id"])
            query_ids[query_no] = query_id
            if as_of == 0:
                as_of = int(row["as_of"])
            workload = row["workload"]
            if workload in by_shape:
                by_shape[workload][query_no] = _parse_ids(row["exact_topk_ids"])
    with ATTR_GT.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["filter_name"] != filter_name:
                continue
            query_no = int(row["query_no"])
            query_id = int(row["query_id"])
            if query_ids.setdefault(query_no, query_id) != query_id:
                raise RuntimeError(f"query_id mismatch at q{query_no}")
            by_shape["attributes"][query_no] = _parse_ids(row["exact_filtered_topk_ids"])
    wanted = list(CALIB_QUERY_NOS) + list(
        range(QUERY_OFFSET, QUERY_OFFSET + QUERY_COUNT)
    )
    missing = [
        (name, query_no)
        for name, _, _ in SHAPES
        for query_no in wanted
        if query_no not in by_shape[name]
    ]
    if missing:
        raise RuntimeError(f"missing GT cells: {missing[:8]} count={len(missing)}")
    return query_ids, by_shape, as_of


def allowlist_sql(spec: bench.FilterSpec, workload: bench.WorkloadSpec) -> str:
    return truth.build_candidate_sql(TABLE, spec.predicate, workload)


def build_allow_list(
    conn: Any,
    faiss_module: Any,
    sql_text: str,
    total_rows: int,
) -> dict[str, Any]:
    import numpy as np

    started = time.perf_counter()
    bitmap = np.zeros((total_rows + 7) // 8, dtype=np.uint8)
    chunks: list[Any] = []
    streamed = 0
    with conn.transaction():
        with conn.cursor() as control:
            server_started = time.perf_counter()
            control.execute(
                "CREATE TEMP TABLE allowlist_fig5 ON COMMIT DROP AS " + sql_text
            )
            materialized = int(getattr(control, "rowcount", -1))
            if materialized < 0:
                control.execute("SELECT count(*) FROM allowlist_fig5")
                materialized = int(control.fetchone()[0])
            server_ms = (time.perf_counter() - server_started) * 1000.0
        with conn.cursor(name="allowlist_fig5_cur") as cursor:
            transfer_started = time.perf_counter()
            cursor.execute("SELECT id FROM allowlist_fig5")
            while True:
                batch = cursor.fetchmany(100_000)
                if not batch:
                    break
                values = np.fromiter(
                    (int(row[0]) for row in batch),
                    dtype=np.int64,
                    count=len(batch),
                )
                streamed += int(values.size)
                chunks.append(values)
            transfer_ms = (time.perf_counter() - transfer_started) * 1000.0
    if materialized != streamed:
        raise RuntimeError(
            f"allow-list row mismatch materialized={materialized} streamed={streamed}"
        )
    bitmap_started = time.perf_counter()
    for values in chunks:
        set_bitmap_ids(bitmap, values, total_rows)
    selector = faiss_module.IDSelectorBitmap(total_rows, faiss_module.swig_ptr(bitmap))
    build_ms = (time.perf_counter() - started) * 1000.0
    return {
        "selector": selector,
        "bitmap": bitmap,
        "rows": streamed,
        "build_ms": build_ms,
        "server_ms": server_ms,
        "transfer_ms": transfer_ms,
        "bitmap_ms": (time.perf_counter() - bitmap_started) * 1000.0,
    }


def recall_at_k(got: list[int], truth_ids: tuple[int, ...], k: int) -> float:
    denom = min(k, len(truth_ids))
    if denom == 0:
        return 0.0
    return len(set(got[:k]) & set(truth_ids[:k])) / denom


def lcb95(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean
    se = statistics.pstdev(values) / math.sqrt(len(values))
    return mean - 1.96 * se


def set_cohort(offset: int, count: int) -> None:
    global QUERY_OFFSET, QUERY_COUNT
    QUERY_OFFSET = int(offset)
    QUERY_COUNT = int(count)


def fragment_memory(cur: Any) -> dict[str, Any]:
    audit = bench.audit_fragment_store(cur, TABLE)
    store_bytes = 0
    try:
        cur.execute(
            """
            SELECT COALESCE(SUM(pg_column_size(store_row)), 0)
            FROM public.pgvector_hnsw_fragment_store AS store_row
            WHERE store_row.heap_oid = to_regclass(%s)
            """,
            (TABLE,),
        )
        store_bytes = int(cur.fetchone()[0])
    except Exception as exc:  # noqa: BLE001
        store_bytes = -1
        audit = {**audit, "size_error": str(exc)}
    cache: dict[str, Any] = {}
    try:
        cache = bench.fetch_json_object(cur, "SELECT vector_hnsw_metadata_cache_profile()")
    except Exception:
        cache = {}
    return {
        "store_count": audit.get("count", 0),
        "store_bytes": store_bytes,
        "store_mib": round(store_bytes / (1024 * 1024), 3) if store_bytes >= 0 else None,
        "cache": cache,
    }


def prepare_pg(cur: Any) -> None:
    bench.ensure_sqlens_fragment_catalog(cur, PRINCIPAL, TABLE)
    cur.execute("SET hnsw.guidance_require_epoch = on")
    cur.execute(f'SET ROLE "{PRINCIPAL}"')


def run_pg_shape(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    mode: str,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth_ids: dict[int, tuple[int, ...]],
    as_of: int,
    ef_search: int,
    query_nos: list[int] | None = None,
    phase: str = "measurement",
    reuse_guidance: bool = False,
    warmup_nos: list[int] | None = None,
) -> list[dict[str, Any]]:
    vector_index = bench.mode_index(mode, SOURCE_INDEX, CLONE_INDEX)
    config = bench.Config(ef_search, 5_000_000, 32.0, "relaxed_order", ef_search)
    bench.set_mode(cur, mode, config, vector_index)
    sql_text = bench.build_hybrid_sql(TABLE, spec.predicate, workload=workload)
    atoms = bench.binding_atoms_for(workload, spec)
    rows: list[dict[str, Any]] = []

    def _one_query(query_no: int, *, timed: bool) -> dict[str, Any]:
        query_id = query_ids[query_no]
        params = bench.bind_query_embedding(
            {
                "query_id": query_id,
                "as_of": as_of,
                "k": K,
                "vector_index": vector_index,
                "binding_atoms": list(atoms),
                "binding_kind": bench.MODE_SPECS[mode].guidance_kind or "bloom",
            },
            query_id,
            embeddings,
        )
        error = ""
        ids: list[int] = []
        activation_ms = 0.0
        e2e_ms = 0.0
        try:
            if timed and reuse_guidance:
                cur.execute("SELECT vector_hnsw_reset_scan_profile()")
                started = time.perf_counter()
                cur.execute(sql_text, params)
                fetched = cur.fetchall()
                e2e_ms = (time.perf_counter() - started) * 1000.0
                ids = [int(row[0]) for row in fetched]
            else:
                cur.execute("SELECT vector_hnsw_reset_scan_profile()")
                started = time.perf_counter()
                bench.set_as_of(cur, as_of)
                activation = bench.configure_guidance(cur, mode, vector_index, atoms)
                activation_ms = float(activation["activation_ms"])
                cur.execute(sql_text, params)
                fetched = cur.fetchall()
                ids = [int(row[0]) for row in fetched]
                e2e_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            error = f"{exc.__class__.__name__}: {exc}"
            try:
                cur.execute("ROLLBACK")
                bench.ensure_sqlens_fragment_catalog(cur, PRINCIPAL, TABLE)
                bench.set_mode(cur, mode, config, vector_index)
            except Exception:
                pass
        return {
            "phase": "warmup" if not timed else phase,
            "shape": workload.name,
            "filter_name": spec.name,
            "mode": mode,
            "query_no": query_no,
            "query_id": query_id,
            "ef_search": ef_search,
            "e2e_ms": e2e_ms if not error else "",
            "allow_ms": 0.0,
            "search_ms": e2e_ms if not error else "",
            "activation_ms": activation_ms if not error else "",
            "recall": (
                recall_at_k(ids, truth_ids[query_no], K) if not error else ""
            ),
            "error": error,
        }

    if warmup_nos:
        for query_no in warmup_nos:
            rows.append(_one_query(query_no, timed=False))
        print(
            json.dumps(
                {
                    "progress": "warmup_done",
                    "shape": workload.name,
                    "filter_name": spec.name,
                    "mode": mode,
                    "n": len(warmup_nos),
                }
            ),
            flush=True,
        )
    if reuse_guidance:
        bench.set_as_of(cur, as_of)
        bench.configure_guidance(cur, mode, vector_index, atoms)

    wanted = query_nos or list(range(QUERY_OFFSET, QUERY_OFFSET + QUERY_COUNT))
    for i, query_no in enumerate(wanted, start=1):
        rows.append(_one_query(query_no, timed=True))
        if i % 50 == 0:
            print(
                json.dumps(
                    {
                        "progress": phase,
                        "shape": workload.name,
                        "filter_name": spec.name,
                        "mode": mode,
                        "ef_search": ef_search,
                        "completed": i,
                        "total": len(wanted),
                    }
                ),
                flush=True,
            )
    return rows


def _calib_stats(rows: list[dict[str, Any]]) -> tuple[float, float]:
    ok = [row for row in rows if not row.get("error") and row.get("recall") != ""]
    recalls = [float(row["recall"]) for row in ok]
    latencies = [float(row["e2e_ms"]) for row in ok if row.get("e2e_ms") != ""]
    return (
        lcb95(recalls),
        statistics.fmean(latencies) if latencies else float("inf"),
    )


def choose_pg_ef(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    mode: str,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth_ids: dict[int, tuple[int, ...]],
    as_of: int,
) -> tuple[int, bool, float, float]:
    best = (PG_EF_GRID[-1], False, 0.0, float("inf"))
    for ef in PG_EF_GRID:
        rows = run_pg_shape(
            cur,
            workload,
            spec,
            mode,
            query_ids,
            embeddings,
            truth_ids,
            as_of,
            ef,
            list(CALIB_QUERY_NOS),
            phase="calibration",
        )
        bound, mean_ms = _calib_stats(rows)
        print(
            json.dumps(
                {
                    "progress": "pg_calib",
                    "shape": workload.name,
                    "filter_name": spec.name,
                    "mode": mode,
                    "ef": ef,
                    "n": len(CALIB_QUERY_NOS),
                    "lcb95": bound,
                    "mean_e2e_ms": mean_ms,
                }
            ),
            flush=True,
        )
        if bound >= 0.90 and mean_ms < best[3]:
            return ef, True, bound, mean_ms
        if bound > best[2]:
            best = (ef, False, bound, mean_ms)
    return best


def choose_matched_speedup(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth_ids: dict[int, tuple[int, ...]],
    as_of: int,
    sqlens_modes: tuple[str, ...],
) -> dict[str, Any]:
    """Pick the shared ef / SQLens mode with the largest stock/SQLens speedup.

    Both arms must hit Recall@10 LCB95 >= 0.90 at that same ef. This avoids
    the cheap-ef trap where stock already qualifies and SQLens only pays tax.
    """
    candidates: list[dict[str, Any]] = []
    fallback: dict[str, Any] | None = None
    for ef in PG_EF_GRID:
        stock_rows = run_pg_shape(
            cur,
            workload,
            spec,
            "stock",
            query_ids,
            embeddings,
            truth_ids,
            as_of,
            ef,
            list(CALIB_QUERY_NOS),
            phase="calibration",
        )
        stock_lcb, stock_ms = _calib_stats(stock_rows)
        print(
            json.dumps(
                {
                    "progress": "pg_calib",
                    "shape": workload.name,
                    "filter_name": spec.name,
                    "mode": "stock",
                    "ef": ef,
                    "lcb95": stock_lcb,
                    "mean_e2e_ms": stock_ms,
                }
            ),
            flush=True,
        )
        for mode in sqlens_modes:
            bench.clear_fragment_store(cur, TABLE)
            sqlens_rows = run_pg_shape(
                cur,
                workload,
                spec,
                mode,
                query_ids,
                embeddings,
                truth_ids,
                as_of,
                ef,
                list(CALIB_QUERY_NOS),
                phase="calibration",
            )
            sqlens_lcb, sqlens_ms = _calib_stats(sqlens_rows)
            speedup = stock_ms / sqlens_ms if stock_ms and sqlens_ms else 0.0
            row = {
                "ef": ef,
                "mode": mode,
                "stock_lcb95": stock_lcb,
                "sqlens_lcb95": sqlens_lcb,
                "stock_ms": stock_ms,
                "sqlens_ms": sqlens_ms,
                "speedup": speedup,
                "attained": stock_lcb >= 0.90 and sqlens_lcb >= 0.90,
            }
            print(json.dumps({"progress": "pg_calib_pair", **row, "shape": workload.name, "filter_name": spec.name}), flush=True)
            if row["attained"]:
                candidates.append(row)
            if fallback is None or sqlens_lcb > fallback["sqlens_lcb95"]:
                fallback = row
    if candidates:
        return max(candidates, key=lambda item: (item["speedup"], -item["sqlens_ms"]))
    if fallback is None:
        raise RuntimeError(f"no PG calib rows for {spec.name}/{workload.name}")
    return fallback


def choose_faiss_ef(
    index: Any,
    faiss_module: Any,
    vectors: Any,
    selector: Any,
    query_ids: dict[int, int],
    truth_ids: dict[int, tuple[int, ...]],
) -> tuple[int, bool, float]:
    calib = list(range(20, 100))
    best = (FAISS_EF_GRID[0], 0.0, 0.0)
    for ef in FAISS_EF_GRID:
        recalls: list[float] = []
        for query_no in calib:
            query_id = query_ids[query_no]
            ids, _ = search_faiss(
                index,
                faiss_module,
                vectors[query_id],
                selector,
                ef,
                K,
                query_id,
            )
            recalls.append(recall_at_k(ids, truth_ids[query_no], K))
        mean = statistics.fmean(recalls)
        bound = lcb95(recalls)
        print(
            json.dumps(
                {
                    "progress": "faiss_calib",
                    "ef": ef,
                    "mean": mean,
                    "lcb95": bound,
                }
            ),
            flush=True,
        )
        if bound > best[1]:
            best = (ef, bound, mean)
        if bound >= 0.90:
            return ef, True, bound
    return best[0], False, best[1]


def run_faiss_shape(
    index: Any,
    faiss_module: Any,
    vectors: Any,
    selector: Any,
    allow_ms: float,
    ef: int,
    shape: str,
    query_ids: dict[int, int],
    truth_ids: dict[int, tuple[int, ...]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wanted = list(range(QUERY_OFFSET, QUERY_OFFSET + QUERY_COUNT))
    for i, query_no in enumerate(wanted, start=1):
        query_id = query_ids[query_no]
        error = ""
        ids: list[int] = []
        search_ms = 0.0
        try:
            ids, search_ms = search_faiss(
                index,
                faiss_module,
                vectors[query_id],
                selector,
                ef,
                K,
                query_id,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{exc.__class__.__name__}: {exc}"
        rows.append(
            {
                "phase": "measurement",
                "shape": shape,
                "filter_name": FILTER_NAME,
                "mode": "faiss_allowlist",
                "query_no": query_no,
                "query_id": query_id,
                "ef_search": ef,
                "e2e_ms": (allow_ms + search_ms) if not error else "",
                "allow_ms": allow_ms,
                "search_ms": search_ms if not error else "",
                "activation_ms": 0.0,
                "recall": (
                    recall_at_k(ids, truth_ids[query_no], K) if not error else ""
                ),
                "error": error,
            }
        )
        if i % 50 == 0:
            print(
                json.dumps(
                    {
                        "progress": "faiss",
                        "shape": shape,
                        "completed": i,
                        "total": QUERY_COUNT,
                    }
                ),
                flush=True,
            )
    return rows


def summarize(
    rows: list[dict[str, Any]],
    allow_by_shape: dict[str, dict[str, Any]],
    filter_name: str = FILTER_NAME,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("phase", "measurement")) not in {"measurement", "final", ""}:
            continue
        if filter_name and row.get("filter_name") not in {None, "", filter_name}:
            continue
        if row.get("error") or row.get("e2e_ms") in {"", None}:
            continue
        grouped[(row["shape"], row["mode"])].append(row)
    cells = []
    for shape, title, _ in SHAPES:
        stock = grouped.get((shape, "stock"), [])
        sqlens = grouped.get((shape, "d1_d2_d3"), []) or grouped.get((shape, "d1"), [])
        faiss = grouped.get((shape, "faiss_allowlist"), [])
        stock_ms = statistics.fmean(float(r["e2e_ms"]) for r in stock) if stock else None
        sqlens_ms = statistics.fmean(float(r["e2e_ms"]) for r in sqlens) if sqlens else None
        search_ms = (
            statistics.fmean(float(r["search_ms"]) for r in faiss) if faiss else None
        )
        allow_ms = float(allow_by_shape.get(shape, {}).get("build_ms") or 0.0)
        cells.append(
            {
                "shape": shape,
                "panel": title,
                "stock_ms": stock_ms,
                "sqlens_ms": sqlens_ms,
                "speedup_vs_stock": (
                    stock_ms / sqlens_ms if stock_ms and sqlens_ms else None
                ),
                "faiss_allow_ms": allow_ms,
                "faiss_search_ms": search_ms,
                "faiss_e2e_cold_ms": (
                    allow_ms + search_ms if search_ms is not None else None
                ),
                "stock_recall": (
                    statistics.fmean(float(r["recall"]) for r in stock) if stock else None
                ),
                "sqlens_recall": (
                    statistics.fmean(float(r["recall"]) for r in sqlens)
                    if sqlens
                    else None
                ),
                "faiss_recall": (
                    statistics.fmean(float(r["recall"]) for r in faiss) if faiss else None
                ),
                "stock_n": len(stock),
                "sqlens_n": len(sqlens),
                "faiss_n": len(faiss),
                "beats_stock": bool(
                    stock_ms and sqlens_ms and sqlens_ms < stock_ms
                ),
            }
        )
    amort_cell = next(cell for cell in cells if cell["shape"] == "join_acl")
    amortize = []
    for count in AMORTIZE_NS:
        allow = amort_cell["faiss_allow_ms"] or 0.0
        search = amort_cell["faiss_search_ms"] or 0.0
        amortize.append(
            {
                "n": count,
                "shape": amort_cell["shape"],
                "stock_ms": amort_cell["stock_ms"],
                "sqlens_ms": amort_cell["sqlens_ms"],
                "faiss_ms": allow / count + search,
            }
        )
    return {
        "paper_eligible": bool(
            QUERY_COUNT >= 10_000
            and all(cell.get("stock_n") == QUERY_COUNT for cell in cells)
            and all(cell.get("sqlens_n") == QUERY_COUNT for cell in cells)
            and all(cell.get("faiss_n") == QUERY_COUNT for cell in cells)
            and all(cell.get("beats_stock") for cell in cells)
        ),
        "filter_name": filter_name,
        "queries": QUERY_COUNT,
        "cells": cells,
        "amortize": amortize,
        "all_sqlens_beat_stock": all(cell["beats_stock"] for cell in cells),
    }


def plot_preview(score: dict[str, Any], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    cells = score["cells"]
    labels = [cell["panel"] for cell in cells]
    xpos = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    stock = [cell["stock_ms"] or 0.0 for cell in cells]
    sqlens = [cell["sqlens_ms"] or 0.0 for cell in cells]
    allow = [cell["faiss_allow_ms"] or 0.0 for cell in cells]
    search = [cell["faiss_search_ms"] or 0.0 for cell in cells]
    ax.bar(xpos - width, stock, width, label="stock pgvector", color="#4C78A8")
    ax.bar(xpos, sqlens, width, label="SQLens", color="#B279A2")
    ax.bar(xpos + width, allow, width, label="FAISS allow-list", color="#F58518")
    ax.bar(
        xpos + width,
        search,
        width,
        bottom=allow,
        label="FAISS HNSW search",
        color="#F2CF5B",
    )
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean e2e ms")
    ax.set_title(f"{score.get('filter_name', FILTER_NAME)}  (q1K screen, not paper)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    bars = out_dir / "preview_a_stacked_bars.pdf"
    fig.savefig(bars)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = [row["n"] for row in score["amortize"]]
    ax.plot(
        xs,
        [row["stock_ms"] for row in score["amortize"]],
        "s--",
        color="#4C78A8",
        label="stock pgvector",
    )
    ax.plot(
        xs,
        [row["sqlens_ms"] for row in score["amortize"]],
        "D-",
        color="#B279A2",
        label="SQLens",
    )
    ax.plot(
        xs,
        [row["faiss_ms"] for row in score["amortize"]],
        "o-",
        color="#F58518",
        label="FAISS  T_allow/N + T_search",
    )
    ax.set_xscale("log")
    ax.set_xlabel("queries sharing the same ACL JOIN SQL (N)")
    ax.set_ylabel("mean e2e ms / query")
    ax.set_title("Allow-list amortization  (q1K screen, not paper)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    amort = out_dir / "preview_b_amortize.pdf"
    fig.savefig(amort)
    plt.close(fig)
    (out_dir / "score.json").write_text(
        json.dumps(score, indent=2, default=str), encoding="utf-8"
    )
    return [str(bars), str(amort)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--retune", action="store_true")
    parser.add_argument("--hot-guidance", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--query-offset", type=int, default=QUERY_OFFSET)
    parser.add_argument("--query-count", type=int, default=0)
    parser.add_argument("--reuse-faiss-from", type=Path)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    count = args.query_count
    if count <= 0:
        count = 10_000 if args.formal else QUERY_COUNT
    set_cohort(args.query_offset, count)
    if args.formal:
        args.hot_guidance = False
        args.retune = False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "screen.csv"
    if args.plot_only:
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        allow_path = args.out_dir / "allowlist.json"
        allow_by_shape = (
            json.loads(allow_path.read_text(encoding="utf-8")) if allow_path.is_file() else {}
        )
        score = summarize(rows, allow_by_shape)
        print(json.dumps({"plots": plot_preview(score, args.out_dir), "score": score}, indent=2))
        return 0
    if not args.execute:
        print("dry-run: pass --execute to measure")
        return 0

    filter_names = list(
        RETUNE_FILTERS if args.retune and not args.hot_guidance else (FILTER_NAME,)
    )
    spec = bench.read_filters(bench.DEFAULT_FILTERS, {filter_names[0]})[0]
    query_ids, truth_by_shape, as_of = load_truth(filter_names[0])
    print(
        json.dumps(
            {
                "progress": "gt_loaded",
                "queries": QUERY_COUNT,
                "as_of": as_of,
                "filter": FILTER_NAME,
            }
        ),
        flush=True,
    )

    require_psycopg()
    import psycopg

    cfg = pg_config_from_env()
    conninfo = cfg.conninfo
    allow_by_shape: dict[str, dict[str, Any]] = {}
    faiss_efs: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    selected_efs: dict[str, Any] = {}
    memory_log: dict[str, Any] = {}
    reuse_ef_only = bool(args.formal and args.reuse_faiss_from is not None)
    if args.reuse_faiss_from is not None and not reuse_ef_only:
        frozen_csv = args.reuse_faiss_from / "screen.csv"
        allow_path = args.reuse_faiss_from / "allowlist.json"
        ef_path = args.reuse_faiss_from / "faiss_ef.json"
        reused = [
            row
            for row in csv.DictReader(frozen_csv.open(encoding="utf-8"))
            if row.get("mode") == "faiss_allowlist"
        ]
        all_rows.extend(reused)
        if allow_path.is_file():
            allow_by_shape = json.loads(allow_path.read_text(encoding="utf-8"))
        if ef_path.is_file():
            faiss_efs = json.loads(ef_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "progress": "reuse_faiss",
                    "source": str(args.reuse_faiss_from),
                    "rows": len(reused),
                }
            ),
            flush=True,
        )
    if args.reuse_faiss_from is not None and reuse_ef_only:
        ef_path = args.reuse_faiss_from / "faiss_ef.json"
        if ef_path.is_file():
            faiss_efs = json.loads(ef_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "progress": "reuse_faiss_ef_only",
                    "source": str(args.reuse_faiss_from),
                    "efs": {name: row.get("ef") for name, row in faiss_efs.items()},
                }
            ),
            flush=True,
        )
    if args.reuse_faiss_from is None or reuse_ef_only:
        import faiss

        vectors, vector_rows, _ = read_fbin_memmap(FBIN)
        index = faiss.read_index(str(FAISS_INDEX))
        print(
            json.dumps(
                {
                    "progress": "faiss_loaded",
                    "ntotal": int(index.ntotal),
                    "fbin_rows": int(vector_rows),
                }
            ),
            flush=True,
        )
        with psycopg.connect(conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            prepare_pg(cur)
            bench.set_heap_competing_indexes_valid(cur, TABLE, valid=True)
            for name, title, join_kind in SHAPES:
                workload = _workload(name, join_kind)
                sql_text = allowlist_sql(spec, workload)
                print(
                    json.dumps(
                        {"progress": "allowlist_start", "shape": name, "title": title}
                    ),
                    flush=True,
                )
                allow = build_allow_list(conn, faiss, sql_text, int(index.ntotal))
                allow_by_shape[name] = {
                    key: allow[key]
                    for key in ("rows", "build_ms", "server_ms", "transfer_ms")
                }
                print(
                    json.dumps(
                        {
                            "progress": "allowlist_done",
                            "shape": name,
                            **allow_by_shape[name],
                        }
                    ),
                    flush=True,
                )
                frozen = faiss_efs.get(name) if reuse_ef_only else None
                if frozen and frozen.get("ef"):
                    ef = int(frozen["ef"])
                    attained = bool(frozen.get("lcb_attained", True))
                    bound = float(frozen.get("calib_lcb95") or 0.0)
                else:
                    ef, attained, bound = choose_faiss_ef(
                        index,
                        faiss,
                        vectors,
                        allow["selector"],
                        query_ids,
                        truth_by_shape[name],
                    )
                faiss_efs[name] = {
                    "ef": ef,
                    "lcb_attained": attained,
                    "calib_lcb95": bound,
                    "reused_ef": bool(frozen),
                }
                all_rows.extend(
                    run_faiss_shape(
                        index,
                        faiss,
                        vectors,
                        allow["selector"],
                        float(allow["build_ms"]),
                        ef,
                        name,
                        query_ids,
                        truth_by_shape[name],
                    )
                )
            cur.execute("RESET ROLE")

    sqlens_modes = ("d1", "d1_d2_d3") if args.retune else ("d1_d2_d3",)
    with psycopg.connect(conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        prepare_pg(cur)
        try:
            bench.set_heap_competing_indexes_valid(cur, TABLE, valid=False)
            for filter_name in filter_names:
                spec = bench.read_filters(bench.DEFAULT_FILTERS, {filter_name})[0]
                query_ids, truth_by_shape, as_of = load_truth(filter_name)
                embed_ids = [
                    query_ids[query_no]
                    for query_no in list(CALIB_QUERY_NOS)
                    + list(range(QUERY_OFFSET, QUERY_OFFSET + QUERY_COUNT))
                ]
                embeddings = bench.load_query_embeddings(cur, TABLE, embed_ids)
                print(
                    json.dumps(
                        {
                            "progress": "gt_loaded",
                            "queries": QUERY_COUNT,
                            "as_of": as_of,
                            "filter": filter_name,
                        }
                    ),
                    flush=True,
                )
                for name, title, join_kind in SHAPES:
                    workload = _workload(name, join_kind)
                    if args.formal:
                        bench.clear_fragment_store(cur, TABLE)
                        for mode in ("stock", "d1_d2_d3"):
                            if mode != "stock":
                                bench.clear_fragment_store(cur, TABLE)
                            print(
                                json.dumps(
                                    {
                                        "progress": "pg_start",
                                        "filter": filter_name,
                                        "shape": name,
                                        "mode": mode,
                                        "ef": EF_PG,
                                        "formal": True,
                                    }
                                ),
                                flush=True,
                            )
                            all_rows.extend(
                                run_pg_shape(
                                    cur,
                                    workload,
                                    spec,
                                    mode,
                                    query_ids,
                                    embeddings,
                                    truth_by_shape[name],
                                    as_of,
                                    EF_PG,
                                )
                            )
                        memory_log[f"{filter_name}|{name}"] = fragment_memory(cur)
                        selected_efs[f"{filter_name}|{name}"] = {
                            "ef": EF_PG,
                            "mode": "d1_d2_d3",
                            "formal": True,
                        }
                        continue
                    if args.hot_guidance:
                        if name == "attributes":
                            bench.clear_fragment_store(cur, TABLE)
                        for mode in ("stock", "d1_d2_d3"):
                            print(
                                json.dumps(
                                    {
                                        "progress": "pg_start",
                                        "filter": filter_name,
                                        "shape": name,
                                        "mode": mode,
                                        "ef": EF_PG,
                                        "hot_guidance": True,
                                    }
                                ),
                                flush=True,
                            )
                            all_rows.extend(
                                run_pg_shape(
                                    cur,
                                    workload,
                                    spec,
                                    mode,
                                    query_ids,
                                    embeddings,
                                    truth_by_shape[name],
                                    as_of,
                                    EF_PG,
                                    reuse_guidance=True,
                                    warmup_nos=list(WARMUP_QUERY_NOS),
                                )
                            )
                            selected_efs[f"{filter_name}|{name}|{mode}"] = {
                                "ef": EF_PG,
                                "hot_guidance": True,
                                "warmup": len(WARMUP_QUERY_NOS),
                            }
                        continue
                    bench.clear_fragment_store(cur, TABLE)
                    picked = choose_matched_speedup(
                        cur,
                        workload,
                        spec,
                        query_ids,
                        embeddings,
                        truth_by_shape[name],
                        as_of,
                        sqlens_modes,
                    )
                    if not picked.get("attained"):
                        raise RuntimeError(
                            f"no matched ef reached Recall@10 0.90 for {filter_name}/{name}: {picked}"
                        )
                    selected_efs[f"{filter_name}|{name}"] = picked
                    print(
                        json.dumps(
                            {
                                "progress": "pg_selected",
                                "filter": filter_name,
                                "shape": name,
                                **picked,
                            }
                        ),
                        flush=True,
                    )
                    for mode, ef in (
                        ("stock", int(picked["ef"])),
                        (str(picked["mode"]), int(picked["ef"])),
                    ):
                        if mode != "stock":
                            bench.clear_fragment_store(cur, TABLE)
                        print(
                            json.dumps(
                                {
                                    "progress": "pg_start",
                                    "filter": filter_name,
                                    "shape": name,
                                    "mode": mode,
                                    "ef": ef,
                                }
                            ),
                            flush=True,
                        )
                        all_rows.extend(
                            run_pg_shape(
                                cur,
                                workload,
                                spec,
                                mode,
                                query_ids,
                                embeddings,
                                truth_by_shape[name],
                                as_of,
                                ef,
                            )
                        )
        finally:
            try:
                bench.set_heap_competing_indexes_valid(cur, TABLE, valid=True)
            except Exception:
                pass
            try:
                cur.execute("RESET ROLE")
            except Exception:
                pass

    fieldnames = [
        "phase",
        "shape",
        "filter_name",
        "mode",
        "query_no",
        "query_id",
        "ef_search",
        "e2e_ms",
        "allow_ms",
        "search_ms",
        "activation_ms",
        "recall",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    (args.out_dir / "allowlist.json").write_text(
        json.dumps(allow_by_shape, indent=2), encoding="utf-8"
    )
    (args.out_dir / "faiss_ef.json").write_text(
        json.dumps(faiss_efs, indent=2), encoding="utf-8"
    )
    (args.out_dir / "selected_efs.json").write_text(
        json.dumps(selected_efs, indent=2), encoding="utf-8"
    )
    (args.out_dir / "fragment_memory.json").write_text(
        json.dumps(memory_log, indent=2, default=str), encoding="utf-8"
    )
    payload: dict[str, Any] = {}
    for filter_name in filter_names:
        score = summarize(all_rows, allow_by_shape, filter_name)
        plots = plot_preview(score, args.out_dir / filter_name)
        payload[filter_name] = {"score": score, "plots": plots}
    (args.out_dir / "score.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
