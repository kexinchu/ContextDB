#!/usr/bin/env python3
"""B1 screen: SQL-first vs VisGuide vs stock on 14 Amazon atoms (q1K).

Not paper-eligible. Reuses the Figure 5 / SQL-native helpers. Results stay
under results/hybrid_vector_db/revision45/b1_sql_first_q1k/.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import amazon10m_sql_native_benchmark as bench
import figure5_hybrid_allowlist_screen as fig5
from common_pg import pg_config_from_env, require_psycopg

ROOT = Path(__file__).resolve().parents[4]
FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
OUT_DIR = ROOT / "results/hybrid_vector_db/revision45/b1_sql_first_q1k"
MODES = ("stock", "d1", bench.SQL_FIRST_MODE)
QUERY_OFFSET = 200
QUERY_COUNT = 1000
EF_SEARCH = 100
K = 10


def _workload() -> bench.WorkloadSpec:
    return fig5._workload("attributes", "none")


def run_sql_first(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    query_ids: dict[int, int],
    truth: dict[int, tuple[int, ...]],
    as_of: int,
    query_nos: list[int],
) -> list[dict]:
    """Materialize-then-rank SQL-first: exact SQL, no HNSW."""
    import time

    cur.execute("RESET ROLE")
    bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=True)
    bench.prepare_sql_first_session(
        cur, fig5.PRINCIPAL, fig5.TABLE, fig5.SOURCE_INDEX, fig5.CLONE_INDEX
    )
    sql_text = bench.build_hybrid_sql(
        fig5.TABLE, spec.predicate, workload=workload, exact=True
    )
    scalars = bench.collect_registered_scalar_indexes(cur)
    first_id = query_ids[query_nos[0]]
    params = {"query_id": first_id, "as_of": as_of, "k": K}
    bench.set_as_of(cur, as_of)
    plan, _ = bench.explain(
        cur, sql_text, params, vector_index=fig5.SOURCE_INDEX, require_hnsw=False
    )
    bench.validate_sql_first_explain_gate(plan, scalars)
    rows: list[dict] = []
    for query_no in query_nos:
        query_id = query_ids[query_no]
        params = {"query_id": query_id, "as_of": as_of, "k": K}
        error = ""
        ids: list[int] = []
        e2e_ms = 0.0
        try:
            started = time.perf_counter()
            fetched = bench.query_results(cur, sql_text, params, exact=True)
            e2e_ms = (time.perf_counter() - started) * 1000.0
            ids = [row_id for row_id, _ in fetched]
        except Exception as exc:  # noqa: BLE001
            error = f"{exc.__class__.__name__}: {exc}"
        rows.append(
            {
                "phase": "measurement",
                "shape": workload.name,
                "filter_name": spec.name,
                "mode": bench.SQL_FIRST_MODE,
                "query_no": query_no,
                "query_id": query_id,
                "ef_search": 0,
                "e2e_ms": e2e_ms if not error else "",
                "recall": fig5.recall_at_k(ids, truth[query_no], K) if not error else "",
                "error": error,
            }
        )
    return rows


def _summarize(rows: list[dict], query_count: int) -> dict:
    summary = []
    for name in {str(row["filter_name"]) for row in rows}:
        cell = {"filter_name": name}
        for mode in MODES:
            subset = [
                row
                for row in rows
                if row["filter_name"] == name
                and row["mode"] == mode
                and row.get("phase") == "measurement"
                and not row.get("error")
            ]
            if not subset:
                cell[mode] = None
                continue
            lat = [float(row["e2e_ms"]) for row in subset]
            rec = [float(row["recall"]) for row in subset if row.get("recall") != ""]
            cell[mode] = {
                "n": len(subset),
                "mean_ms": statistics.fmean(lat),
                "recall": statistics.fmean(rec) if rec else None,
            }
        if cell.get("stock") and cell.get("d1"):
            cell["guide_vs_stock"] = cell["stock"]["mean_ms"] / cell["d1"]["mean_ms"]
        if cell.get("stock") and cell.get(bench.SQL_FIRST_MODE):
            cell["sql_first_vs_guide"] = (
                cell[bench.SQL_FIRST_MODE]["mean_ms"] / cell["d1"]["mean_ms"]
                if cell.get("d1")
                else None
            )
        summary.append(cell)
    return {
        "paper_eligible": False,
        "plan_item": "B1",
        "queries": query_count,
        "query_offset": QUERY_OFFSET,
        "ef_search": EF_SEARCH,
        "k": K,
        "modes": list(MODES),
        "cells": sorted(summary, key=lambda row: row["filter_name"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument("--query-count", type=int, default=QUERY_COUNT)
    args = parser.parse_args()
    names = args.filter_names or [
        row["filter_name"] for row in csv.DictReader(FILTERS.open(encoding="utf-8"))
    ]
    contract = {
        "paper_eligible": False,
        "plan_item": "B1",
        "filters": names,
        "modes": list(MODES),
        "query_offset": QUERY_OFFSET,
        "query_count": int(args.query_count),
        "ef_search": EF_SEARCH,
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
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        fig5.prepare_pg(cur)
        from rowlocal_faiss14_screen import load_attr_truth

        query_nos = list(range(QUERY_OFFSET, QUERY_OFFSET + int(args.query_count)))
        for name in names:
            spec = specs[name]
            query_ids, truth, as_of = load_attr_truth(name)
            embed_ids = [query_ids[query_no] for query_no in query_nos if query_no in query_ids]
            embeddings = bench.load_query_embeddings(cur, fig5.TABLE, embed_ids)
            print(json.dumps({"progress": "filter", "filter": name}), flush=True)
            for mode in ("stock", "d1"):
                bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=False)
                rows = fig5.run_pg_shape(
                    cur,
                    workload,
                    spec,
                    mode,
                    query_ids,
                    embeddings,
                    truth,
                    as_of,
                    EF_SEARCH,
                    query_nos=query_nos,
                    phase="measurement",
                    reuse_guidance=(mode == "d1"),
                )
                all_rows.extend(rows)
                print(
                    json.dumps({"progress": "mode_done", "filter": name, "mode": mode, "n": len(rows)}),
                    flush=True,
                )
            rows = run_sql_first(cur, workload, spec, query_ids, truth, as_of, query_nos)
            all_rows.extend(rows)
            print(
                json.dumps(
                    {
                        "progress": "mode_done",
                        "filter": name,
                        "mode": bench.SQL_FIRST_MODE,
                        "n": len(rows),
                    }
                ),
                flush=True,
            )

    csv_path = args.out_dir / "screen.csv"
    if all_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    score = _summarize(all_rows, int(args.query_count))
    (args.out_dir / "score.json").write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "manifest.json").write_text(
        json.dumps({**contract, "paper_eligible": False, "rows": len(all_rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": str(args.out_dir), "rows": len(all_rows), "paper_eligible": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
