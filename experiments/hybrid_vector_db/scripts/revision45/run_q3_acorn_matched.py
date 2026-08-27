#!/usr/bin/env python3
"""Q3: independently tune stock, VisGuide, and acorn1 to Recall@10 LCB >= 0.90.

New table only. Does not rewrite the 2.63× / 171→304 cells.
Uses the Figure 5 / B1 SQL-native attributes path so stock and VisGuide
are not the iterative_scan=off post-filter collapse from the fixed-ef C1
screen.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import amazon10m_sql_native_benchmark as bench
import figure5_hybrid_allowlist_screen as fig5
from common_pg import pg_config_from_env, require_psycopg

ROOT = Path(__file__).resolve().parents[4]
FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
OUT_DIR = ROOT / "results/hybrid_vector_db/revision45/q3_acorn_matched"
QUERY_OFFSET = 200
QUERY_COUNT = 50
K = 10
TARGET_LCB = 0.90
FILTER_NAMES = (
    "long_review_ge500",
    "grocery_helpful",
    "helpful_ge20",
    "grocery_long500",
)
STOCK_GUIDE_EFS = (100, 250, 500, 750, 1000)
ACORN_EFS = (100, 200, 400, 800, 1600)


def _workload() -> bench.WorkloadSpec:
    return fig5._workload("attributes", "none")


def _summarize_rows(rows: list[dict]) -> dict[str, Any] | None:
    ok = [
        row
        for row in rows
        if row.get("phase") == "measurement"
        and not row.get("error")
        and row.get("e2e_ms") != ""
        and row.get("recall") != ""
    ]
    if not ok:
        return None
    rec = [float(row["recall"]) for row in ok]
    lat = [float(row["e2e_ms"]) for row in ok]
    return {
        "n": len(ok),
        "errors": sum(1 for row in rows if row.get("error")),
        "mean_ms": statistics.fmean(lat),
        "recall": statistics.fmean(rec),
        "recall_lcb95": fig5.lcb95(rec),
        "ef_search": int(ok[0]["ef_search"]),
    }


def run_acorn1(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth: dict[int, tuple[int, ...]],
    as_of: int,
    query_nos: list[int],
    ef_search: int,
) -> list[dict]:
    """Predicate-aware acorn1 on the same hybrid SQL; no VisGuide activation."""
    config = bench.Config(ef_search, 5_000_000, 32.0, "off", ef_search)
    cur.execute("RESET ROLE")
    bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=False)
    bench.set_search_config(cur, config)
    bench.configure_hnsw_driven_planner(cur)
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    cur.execute("SET hnsw.filter_strategy = acorn1")
    bench.set_preferred_index(cur, fig5.SOURCE_INDEX)
    cur.execute(f'SET ROLE "{fig5.PRINCIPAL}"')
    sql_text = bench.build_hybrid_sql(
        fig5.TABLE,
        spec.predicate,
        workload=workload,
        official_compatible=True,
    )
    rows: list[dict] = []
    printed_error = False
    for query_no in query_nos:
        query_id = query_ids[query_no]
        params = bench.bind_query_embedding(
            {
                "query_id": query_id,
                "as_of": as_of,
                "k": K,
                "vector_index": fig5.SOURCE_INDEX,
            },
            query_id,
            embeddings,
        )
        error = ""
        ids: list[int] = []
        e2e_ms = 0.0
        try:
            bench.set_as_of(cur, as_of)
            cur.execute("SELECT vector_hnsw_reset_scan_profile()")
            started = time.perf_counter()
            cur.execute(sql_text, params)
            fetched = cur.fetchall()
            e2e_ms = (time.perf_counter() - started) * 1000.0
            ids = [int(row[0]) for row in fetched]
        except Exception as exc:  # noqa: BLE001
            error = f"{exc.__class__.__name__}: {exc}"
            if not printed_error:
                print(
                    json.dumps(
                        {
                            "progress": "acorn1_error",
                            "filter": spec.name,
                            "ef": ef_search,
                            "error": error,
                        }
                    ),
                    flush=True,
                )
                printed_error = True
            try:
                cur.execute("ROLLBACK")
                cur.execute("RESET ROLE")
                cur.execute("SET hnsw.filter_strategy = acorn1")
                bench.set_preferred_index(cur, fig5.SOURCE_INDEX)
                cur.execute(f'SET ROLE "{fig5.PRINCIPAL}"')
            except Exception:
                pass
        rows.append(
            {
                "phase": "measurement",
                "shape": workload.name,
                "filter_name": spec.name,
                "mode": "acorn1",
                "query_no": query_no,
                "query_id": query_id,
                "ef_search": ef_search,
                "e2e_ms": e2e_ms if not error else "",
                "recall": fig5.recall_at_k(ids, truth[query_no], K) if not error else "",
                "error": error,
            }
        )
    return rows


def calibrate_mode(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    mode: str,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth: dict[int, tuple[int, ...]],
    as_of: int,
    query_nos: list[int],
    efs: tuple[int, ...],
) -> tuple[list[dict], dict[str, Any] | None, list[dict[str, Any]]]:
    sweep: list[dict[str, Any]] = []
    chosen_rows: list[dict] = []
    chosen: dict[str, Any] | None = None
    for ef in efs:
        print(
            json.dumps(
                {"progress": "try_ef", "filter": spec.name, "mode": mode, "ef": ef}
            ),
            flush=True,
        )
        if mode == "acorn1":
            rows = run_acorn1(
                cur, workload, spec, query_ids, embeddings, truth, as_of, query_nos, ef
            )
        else:
            cur.execute("RESET ROLE")
            bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=False)
            fig5.prepare_pg(cur)
            rows = fig5.run_pg_shape(
                cur,
                workload,
                spec,
                mode,
                query_ids,
                embeddings,
                truth,
                as_of,
                ef,
                query_nos=query_nos,
                phase="measurement",
                reuse_guidance=(mode == "d1"),
            )
        stats = _summarize_rows(rows)
        if stats:
            sweep.append({"mode": mode, "filter_name": spec.name, **stats})
            print(json.dumps({"progress": "ef_done", **sweep[-1]}), flush=True)
            if stats["recall_lcb95"] >= TARGET_LCB and stats["n"] == len(query_nos):
                chosen_rows = rows
                chosen = stats
                break
        else:
            sweep.append(
                {
                    "mode": mode,
                    "filter_name": spec.name,
                    "ef_search": ef,
                    "n": 0,
                    "errors": sum(1 for row in rows if row.get("error")),
                }
            )
    if chosen is None and sweep:
        # Keep the highest-LCB point so the table can report a miss honestly.
        eligible = [item for item in sweep if item.get("recall_lcb95") is not None]
        if eligible:
            best = max(eligible, key=lambda item: item["recall_lcb95"])
            chosen = {**best, "met_target": False}
    if chosen is not None and "met_target" not in chosen:
        chosen["met_target"] = chosen.get("recall_lcb95", 0.0) >= TARGET_LCB
    return chosen_rows, chosen, sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--filter-names", nargs="*", default=list(FILTER_NAMES))
    parser.add_argument("--query-count", type=int, default=QUERY_COUNT)
    args = parser.parse_args()
    names = args.filter_names
    contract = {
        "paper_eligible": False,
        "plan_item": "Q3",
        "filters": names,
        "query_offset": QUERY_OFFSET,
        "query_count": int(args.query_count),
        "target_recall_lcb95": TARGET_LCB,
        "stock_guide_efs": list(STOCK_GUIDE_EFS),
        "acorn_efs": list(ACORN_EFS),
        "out_dir": str(args.out_dir),
    }
    if not args.execute:
        print(json.dumps({"dry_run": True, **contract}, indent=2))
        return 0

    fig5.set_cohort(QUERY_OFFSET, int(args.query_count))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    require_psycopg()
    import psycopg

    workload = _workload()
    specs = {spec.name: spec for spec in bench.read_filters(bench.DEFAULT_FILTERS, set(names))}
    cfg = pg_config_from_env()
    all_rows: list[dict] = []
    sweep: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    query_nos = list(range(QUERY_OFFSET, QUERY_OFFSET + int(args.query_count)))
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        from rowlocal_faiss14_screen import load_attr_truth

        for name in names:
            spec = specs[name]
            query_ids, truth, as_of = load_attr_truth(name)
            embed_ids = [query_ids[query_no] for query_no in query_nos if query_no in query_ids]
            embeddings = bench.load_query_embeddings(cur, fig5.TABLE, embed_ids)
            print(json.dumps({"progress": "filter", "filter": name}), flush=True)
            cell: dict[str, Any] = {"filter_name": name}
            for mode, efs in (
                ("stock", STOCK_GUIDE_EFS),
                ("d1", STOCK_GUIDE_EFS),
                ("acorn1", ACORN_EFS),
            ):
                rows, chosen, mode_sweep = calibrate_mode(
                    cur,
                    workload,
                    spec,
                    mode,
                    query_ids,
                    embeddings,
                    truth,
                    as_of,
                    query_nos,
                    efs,
                )
                all_rows.extend(rows)
                sweep.extend(mode_sweep)
                cell[mode] = chosen
            if cell.get("stock") and cell.get("d1") and cell.get("acorn1"):
                stock_ms = cell["stock"]["mean_ms"]
                guide_ms = cell["d1"]["mean_ms"]
                acorn_ms = cell["acorn1"]["mean_ms"]
                cell["guide_vs_stock"] = stock_ms / guide_ms if guide_ms else None
                cell["acorn_vs_guide"] = acorn_ms / guide_ms if guide_ms else None
                cell["all_met_target"] = all(
                    bool(cell[mode].get("met_target")) for mode in ("stock", "d1", "acorn1")
                )
            cells.append(cell)

    csv_path = args.out_dir / "screen.csv"
    if all_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    score = {
        **contract,
        "cells": cells,
        "sweep": sweep,
        "rows": len(all_rows),
    }
    (args.out_dir / "score.json").write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "manifest.json").write_text(
        json.dumps({**contract, "rows": len(all_rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": str(args.out_dir), "rows": len(all_rows), "paper_eligible": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
