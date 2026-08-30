#!/usr/bin/env python3
"""Full SQLens (d1_d2_d3 on BFS) vs acorn1 on the same clone.

New directory only. Does not rewrite eval_acorn_matched.tex or Q3 score.json.
Default instance is the r44 Amazon replica (PGPORT=55440), not 55437.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

import amazon10m_sql_native_benchmark as bench
import figure5_hybrid_allowlist_screen as fig5
import run_q3_acorn_matched as q3
from common_pg import pg_config_from_env, require_psycopg

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "results/hybrid_vector_db/revision45/q3_full_vs_acorn"
QUERY_OFFSET = 200
QUERY_COUNT = 50
MODES = ("stock", "d1_d2_d3", "acorn1")


def run_acorn1_on_index(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth: dict[int, tuple[int, ...]],
    as_of: int,
    query_nos: list[int],
    ef_search: int,
    vector_index: str,
) -> list[dict]:
    config = bench.Config(ef_search, 5_000_000, 32.0, "off", ef_search)
    cur.execute("RESET ROLE")
    bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=False)
    bench.set_search_config(cur, config)
    bench.configure_hnsw_driven_planner(cur)
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    cur.execute("SET hnsw.filter_strategy = acorn1")
    bench.set_preferred_index(cur, vector_index)
    cur.execute(f'SET ROLE "{fig5.PRINCIPAL}"')
    sql_text = bench.build_hybrid_sql(
        fig5.TABLE,
        spec.predicate,
        workload=workload,
        official_compatible=True,
    )
    rows: list[dict] = []
    for query_no in query_nos:
        query_id = query_ids[query_no]
        params = bench.bind_query_embedding(
            {
                "query_id": query_id,
                "as_of": as_of,
                "k": q3.K,
                "vector_index": vector_index,
            },
            query_id,
            embeddings,
        )
        error = ""
        ids: list[int] = []
        e2e_ms = 0.0
        try:
            import time

            bench.set_as_of(cur, as_of)
            cur.execute("SELECT vector_hnsw_reset_scan_profile()")
            started = time.perf_counter()
            cur.execute(sql_text, params)
            fetched = cur.fetchall()
            e2e_ms = (time.perf_counter() - started) * 1000.0
            ids = [int(row[0]) for row in fetched]
        except Exception as exc:  # noqa: BLE001
            error = f"{exc.__class__.__name__}: {exc}"
            try:
                cur.execute("ROLLBACK")
                cur.execute("RESET ROLE")
                cur.execute("SET hnsw.filter_strategy = acorn1")
                bench.set_preferred_index(cur, vector_index)
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
                "recall": fig5.recall_at_k(ids, truth[query_no], q3.K) if not error else "",
                "error": error,
                "vector_index": vector_index,
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
            rows = run_acorn1_on_index(
                cur,
                workload,
                spec,
                query_ids,
                embeddings,
                truth,
                as_of,
                query_nos,
                ef,
                fig5.CLONE_INDEX,
            )
        else:
            cur.execute("RESET ROLE")
            bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=False)
            fig5.prepare_pg(cur)
            if mode == "d1_d2_d3":
                bench.clear_fragment_store(cur, fig5.TABLE)
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
                reuse_guidance=(mode == "d1_d2_d3"),
            )
        stats = q3._summarize_rows(rows)
        if stats:
            sweep.append({"mode": mode, "filter_name": spec.name, **stats})
            print(json.dumps({"progress": "ef_done", **sweep[-1]}), flush=True)
            if stats["recall_lcb95"] >= q3.TARGET_LCB and stats["n"] == len(query_nos):
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
        eligible = [item for item in sweep if item.get("recall_lcb95") is not None]
        if eligible:
            best = max(eligible, key=lambda item: item["recall_lcb95"])
            chosen = {**best, "met_target": False}
    if chosen is not None and "met_target" not in chosen:
        chosen["met_target"] = chosen.get("recall_lcb95", 0.0) >= q3.TARGET_LCB
    return chosen_rows, chosen, sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--filter-names", nargs="*", default=list(q3.FILTER_NAMES))
    parser.add_argument("--query-count", type=int, default=QUERY_COUNT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    names = args.filter_names or list(q3.FILTER_NAMES)
    contract = {
        "paper_eligible": False,
        "plan_item": "Q3_FULL_VS_ACORN",
        "comparison": "d1_d2_d3 on BFS vs acorn1 on the same BFS clone",
        "filters": names,
        "query_offset": QUERY_OFFSET,
        "query_count": int(args.query_count),
        "target_recall_lcb95": q3.TARGET_LCB,
        "stock_full_efs": list(q3.STOCK_GUIDE_EFS),
        "acorn_efs": list(q3.ACORN_EFS),
        "full_mode": "d1_d2_d3",
        "acorn_index": fig5.CLONE_INDEX,
        "out_dir": str(args.out_dir),
        "rewrites_eval_acorn_matched": False,
    }
    if not args.execute:
        print(json.dumps({"dry_run": True, **contract}, indent=2))
        return 0

    fig5.set_cohort(QUERY_OFFSET, int(args.query_count))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    require_psycopg()
    import psycopg

    workload = q3._workload()
    specs = {
        spec.name: spec for spec in bench.read_filters(bench.DEFAULT_FILTERS, set(names))
    }
    cfg = pg_config_from_env()
    score_path = args.out_dir / "score.json"
    csv_path = args.out_dir / "screen.csv"
    prior = (
        json.loads(score_path.read_text(encoding="utf-8"))
        if args.resume and score_path.exists()
        else {}
    )
    cells = list(prior.get("cells") or [])
    sweep = list(prior.get("sweep") or [])
    done = {
        str(cell["filter_name"])
        for cell in cells
        if cell.get("stock") and cell.get("d1_d2_d3") and cell.get("acorn1")
    }
    all_rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            all_rows.extend(csv.DictReader(handle))
    query_nos = list(range(QUERY_OFFSET, QUERY_OFFSET + int(args.query_count)))

    def _checkpoint() -> None:
        if all_rows:
            fields: list[str] = []
            seen: set[str] = set()
            for row in all_rows:
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        fields.append(key)
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_rows)
        payload = {**contract, "cells": cells, "sweep": sweep, "rows": len(all_rows)}
        score_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        from rowlocal_faiss14_screen import load_attr_truth

        for name in names:
            if name in done:
                print(json.dumps({"progress": "resume_skip", "filter": name}), flush=True)
                continue
            spec = specs[name]
            query_ids, truth, as_of = load_attr_truth(name)
            embed_ids = [
                query_ids[query_no] for query_no in query_nos if query_no in query_ids
            ]
            embeddings = bench.load_query_embeddings(cur, fig5.TABLE, embed_ids)
            print(json.dumps({"progress": "filter", "filter": name}), flush=True)
            cell: dict[str, Any] = {"filter_name": name}
            for mode, efs in (
                ("stock", q3.STOCK_GUIDE_EFS),
                ("d1_d2_d3", q3.STOCK_GUIDE_EFS),
                ("acorn1", q3.ACORN_EFS),
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
            if cell.get("stock") and cell.get("d1_d2_d3") and cell.get("acorn1"):
                stock_ms = cell["stock"]["mean_ms"]
                full_ms = cell["d1_d2_d3"]["mean_ms"]
                acorn_ms = cell["acorn1"]["mean_ms"]
                cell["full_vs_stock"] = stock_ms / full_ms if full_ms else None
                cell["acorn_vs_full"] = acorn_ms / full_ms if full_ms else None
                cell["all_met_target"] = all(
                    bool(cell[mode].get("met_target"))
                    for mode in ("stock", "d1_d2_d3", "acorn1")
                )
            cells = [item for item in cells if item.get("filter_name") != name]
            cells.append(cell)
            _checkpoint()
            print(json.dumps({"progress": "filter_done", "filter": name, **{
                k: cell.get(k) for k in ("full_vs_stock", "acorn_vs_full", "all_met_target")
            }}), flush=True)

    _checkpoint()
    (args.out_dir / "manifest.json").write_text(
        json.dumps({**contract, "rows": len(all_rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "wrote": str(args.out_dir),
                "rows": len(all_rows),
                "paper_eligible": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
