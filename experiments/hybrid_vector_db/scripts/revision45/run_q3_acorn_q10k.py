#!/usr/bin/env python3
"""Layer 1: ACORN vs stock vs VisGuide at the Table 5 cohort.

Calibrate on q20--q99 (80 queries). Measure q200--q10199 (10,000).
Writes a NEW directory. Does not rewrite Table 5 or the q50 ACORN table.
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
OUT_DIR = ROOT / "results/hybrid_vector_db/revision45/q3_acorn_q10k"
CALIB_NOS = list(range(20, 100))
MEASURE_OFFSET = 200
MEASURE_COUNT = 10_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--filter-names", nargs="*", default=list(q3.FILTER_NAMES))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    names = args.filter_names or list(q3.FILTER_NAMES)
    measure_nos = list(range(MEASURE_OFFSET, MEASURE_OFFSET + MEASURE_COUNT))
    contract = {
        "paper_eligible": False,
        "plan_item": "L1_ACORN_Q10K",
        "adds_results": True,
        "rewrites_table5": False,
        "rewrites_q50_acorn": False,
        "filters": names,
        "calib_nos": [CALIB_NOS[0], CALIB_NOS[-1]],
        "query_offset": MEASURE_OFFSET,
        "query_count": MEASURE_COUNT,
        "target_recall_lcb95": q3.TARGET_LCB,
        "out_dir": str(args.out_dir),
    }
    if not args.execute:
        print(json.dumps({"dry_run": True, **contract}, indent=2))
        return 0

    fig5.set_cohort(MEASURE_OFFSET, MEASURE_COUNT)
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
        if cell.get("stock") and cell.get("d1") and cell.get("acorn1")
        and cell.get("measured")
    }
    all_rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            all_rows.extend(csv.DictReader(handle))

    def _checkpoint() -> None:
        if all_rows:
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
        payload = {**contract, "cells": cells, "sweep": sweep, "rows": len(all_rows)}
        score_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        from rowlocal_faiss14_screen import load_attr_truth

        try:
            for name in names:
                if name in done:
                    print(
                        json.dumps({"progress": "resume_skip", "filter": name}),
                        flush=True,
                    )
                    continue
                spec = specs[name]
                query_ids, truth, as_of = load_attr_truth(name)
                wanted = list(CALIB_NOS) + measure_nos
                embed_ids = [
                    query_ids[query_no] for query_no in wanted if query_no in query_ids
                ]
                embeddings = bench.load_query_embeddings(cur, fig5.TABLE, embed_ids)
                print(json.dumps({"progress": "filter", "filter": name}), flush=True)
                cell: dict[str, Any] = {"filter_name": name, "measured": False}
                for mode, efs in (
                    ("stock", q3.STOCK_GUIDE_EFS),
                    ("d1", q3.STOCK_GUIDE_EFS),
                    ("acorn1", q3.ACORN_EFS),
                ):
                    if mode == "d1":
                        cur.execute("RESET ROLE")
                        bench.clear_fragment_store(cur, fig5.TABLE)
                    _calib_rows, chosen, mode_sweep = q3.calibrate_mode(
                        cur,
                        workload,
                        spec,
                        mode,
                        query_ids,
                        embeddings,
                        truth,
                        as_of,
                        CALIB_NOS,
                        efs,
                    )
                    sweep.extend(mode_sweep)
                    cell[f"{mode}_calib"] = chosen
                    if not chosen or not chosen.get("met_target"):
                        cell[mode] = chosen
                        continue
                    ef = int(chosen["ef_search"])
                    print(
                        json.dumps(
                            {
                                "progress": "measure",
                                "filter": name,
                                "mode": mode,
                                "ef": ef,
                                "n": MEASURE_COUNT,
                            }
                        ),
                        flush=True,
                    )
                    if mode == "d1":
                        cur.execute("RESET ROLE")
                        bench.clear_fragment_store(cur, fig5.TABLE)
                    if mode == "acorn1":
                        rows = q3.run_acorn1(
                            cur,
                            workload,
                            spec,
                            query_ids,
                            embeddings,
                            truth,
                            as_of,
                            measure_nos,
                            ef,
                        )
                    else:
                        cur.execute("RESET ROLE")
                        bench.set_heap_competing_indexes_valid(
                            cur, fig5.TABLE, valid=False
                        )
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
                            query_nos=measure_nos,
                            phase="measurement",
                        )
                    stats = q3._summarize_rows(rows)
                    all_rows.extend(rows)
                    if stats:
                        stats["met_target"] = stats["recall_lcb95"] >= q3.TARGET_LCB
                        stats["n_measure"] = stats["n"]
                        cell[mode] = stats
                    else:
                        cell[mode] = chosen
                if cell.get("stock") and cell.get("d1") and cell.get("acorn1"):
                    stock_ms = cell["stock"]["mean_ms"]
                    guide_ms = cell["d1"]["mean_ms"]
                    acorn_ms = cell["acorn1"]["mean_ms"]
                    cell["guide_vs_stock"] = stock_ms / guide_ms if guide_ms else None
                    cell["acorn_vs_guide"] = acorn_ms / guide_ms if guide_ms else None
                    cell["all_met_target"] = all(
                        bool(cell[mode].get("met_target"))
                        for mode in ("stock", "d1", "acorn1")
                    )
                    cell["measured"] = True
                cells = [item for item in cells if item.get("filter_name") != name]
                cells.append(cell)
                _checkpoint()
                print(
                    json.dumps({"progress": "filter_done", "filter": name}),
                    flush=True,
                )
        finally:
            try:
                bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=True)
            except Exception:
                pass
            try:
                cur.execute("RESET ROLE")
            except Exception:
                pass

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
                "rewrites_published": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
