#!/usr/bin/env python3
"""2x2: {stock, SQLens} x {sweeping, acorn1} at matched Recall@10 LCB >= 0.90.

Sweeping is iterative_scan=relaxed_order. ACORN is filter_strategy=acorn1
with iterative_scan=off. SQLens arms use d1_d2_d3 on the BFS clone and
activate fragments; the ACORN SQLens arm then switches the GUC to acorn1
so hybrid L0 can see a live guide.

Writes only under revision45/q3_sweep_acorn_2x2. Does not touch 55437 or
paper tables. Default PGPORT is 55440.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
OUT_DIR = ROOT / "results/hybrid_vector_db/revision45/q3_sweep_acorn_2x2"
QUERY_OFFSET = 200
QUERY_COUNT = 50
ARMS = ("stock_sweep", "sqlens_sweep", "stock_acorn", "sqlens_acorn")


def _sql_and_params(
    spec: bench.FilterSpec,
    workload: bench.WorkloadSpec,
    query_id: int,
    as_of: int,
    embeddings: dict[int, str],
    vector_index: str,
    atoms: tuple[str, ...] | None = None,
) -> tuple[str, dict[str, Any]]:
    sql_text = bench.build_hybrid_sql(
        fig5.TABLE,
        spec.predicate,
        workload=workload,
        official_compatible=True,
    )
    payload: dict[str, Any] = {
        "query_id": query_id,
        "as_of": as_of,
        "k": q3.K,
        "vector_index": vector_index,
    }
    if atoms:
        payload["binding_atoms"] = list(atoms)
        payload["binding_kind"] = "adaptive"
    return sql_text, bench.bind_query_embedding(payload, query_id, embeddings)


def _timed_sql(
    cur: Any,
    sql_text: str,
    params: dict[str, Any],
    as_of: int,
) -> tuple[list[int], float, str]:
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
        try:
            cur.execute("ROLLBACK")
        except Exception:
            pass
    return ids, e2e_ms, error


def run_stock_acorn(
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
    config = bench.Config(ef_search, 5_000_000, 32.0, "off", ef_search)
    cur.execute("RESET ROLE")
    bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=False)
    bench.set_search_config(cur, config)
    bench.configure_hnsw_driven_planner(cur)
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    cur.execute("SET hnsw.filter_strategy = acorn1")
    bench.set_preferred_index(cur, fig5.CLONE_INDEX)
    cur.execute(f'SET ROLE "{fig5.PRINCIPAL}"')
    rows: list[dict] = []
    for query_no in query_nos:
        sql_text, params = _sql_and_params(
            spec, workload, query_ids[query_no], as_of, embeddings, fig5.CLONE_INDEX
        )
        ids, e2e_ms, error = _timed_sql(cur, sql_text, params, as_of)
        if error:
            try:
                cur.execute("RESET ROLE")
                cur.execute("SET hnsw.filter_strategy = acorn1")
                bench.set_preferred_index(cur, fig5.CLONE_INDEX)
                cur.execute(f'SET ROLE "{fig5.PRINCIPAL}"')
            except Exception:
                pass
        rows.append(
            {
                "phase": "measurement",
                "filter_name": spec.name,
                "mode": "stock_acorn",
                "query_no": query_no,
                "ef_search": ef_search,
                "e2e_ms": e2e_ms if not error else "",
                "recall": fig5.recall_at_k(ids, truth[query_no], q3.K) if not error else "",
                "error": error,
            }
        )
    return rows


def run_sqlens_acorn(
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
    config = bench.Config(ef_search, 5_000_000, 32.0, "off", ef_search)
    cur.execute("RESET ROLE")
    bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=False)
    fig5.prepare_pg(cur)
    bench.clear_fragment_store(cur, fig5.TABLE)
    bench.set_mode(cur, "d1_d2_d3", config, fig5.CLONE_INDEX)
    cur.execute("SET hnsw.filter_strategy = acorn1")
    cur.execute("SET hnsw.iterative_scan = off")
    atoms = bench.binding_atoms_for(workload, spec)
    bench.set_as_of(cur, as_of)
    bench.configure_guidance(cur, "d1_d2_d3", fig5.CLONE_INDEX, atoms)
    rows: list[dict] = []
    for query_no in query_nos:
        sql_text, params = _sql_and_params(
            spec,
            workload,
            query_ids[query_no],
            as_of,
            embeddings,
            fig5.CLONE_INDEX,
            atoms,
        )
        ids, e2e_ms, error = _timed_sql(cur, sql_text, params, as_of)
        if error:
            try:
                cur.execute("ROLLBACK")
                fig5.prepare_pg(cur)
                bench.set_mode(cur, "d1_d2_d3", config, fig5.CLONE_INDEX)
                cur.execute("SET hnsw.filter_strategy = acorn1")
                cur.execute("SET hnsw.iterative_scan = off")
                bench.configure_guidance(cur, "d1_d2_d3", fig5.CLONE_INDEX, atoms)
            except Exception:
                pass
        rows.append(
            {
                "phase": "measurement",
                "filter_name": spec.name,
                "mode": "sqlens_acorn",
                "query_no": query_no,
                "ef_search": ef_search,
                "e2e_ms": e2e_ms if not error else "",
                "recall": fig5.recall_at_k(ids, truth[query_no], q3.K) if not error else "",
                "error": error,
            }
        )
    return rows


def run_arm(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    arm: str,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth: dict[int, tuple[int, ...]],
    as_of: int,
    query_nos: list[int],
    ef_search: int,
) -> list[dict]:
    if arm == "stock_sweep":
        cur.execute("RESET ROLE")
        bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=False)
        fig5.prepare_pg(cur)
        return fig5.run_pg_shape(
            cur,
            workload,
            spec,
            "stock",
            query_ids,
            embeddings,
            truth,
            as_of,
            ef_search,
            query_nos=query_nos,
            phase="measurement",
        )
    if arm == "sqlens_sweep":
        cur.execute("RESET ROLE")
        bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=False)
        fig5.prepare_pg(cur)
        bench.clear_fragment_store(cur, fig5.TABLE)
        return fig5.run_pg_shape(
            cur,
            workload,
            spec,
            "d1_d2_d3",
            query_ids,
            embeddings,
            truth,
            as_of,
            ef_search,
            query_nos=query_nos,
            phase="measurement",
            reuse_guidance=True,
        )
    if arm == "stock_acorn":
        return run_stock_acorn(
            cur, workload, spec, query_ids, embeddings, truth, as_of, query_nos, ef_search
        )
    if arm == "sqlens_acorn":
        return run_sqlens_acorn(
            cur, workload, spec, query_ids, embeddings, truth, as_of, query_nos, ef_search
        )
    raise ValueError(arm)


def calibrate_arm(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    arm: str,
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
        print(json.dumps({"progress": "try_ef", "filter": spec.name, "arm": arm, "ef": ef}), flush=True)
        rows = run_arm(
            cur, workload, spec, arm, query_ids, embeddings, truth, as_of, query_nos, ef
        )
        stats = q3._summarize_rows(rows)
        if stats:
            sweep.append({"arm": arm, "filter_name": spec.name, **stats})
            print(json.dumps({"progress": "ef_done", **sweep[-1]}), flush=True)
            if stats["recall_lcb95"] >= q3.TARGET_LCB and stats["n"] == len(query_nos):
                chosen_rows = rows
                chosen = stats
                break
        else:
            sweep.append(
                {
                    "arm": arm,
                    "filter_name": spec.name,
                    "ef_search": ef,
                    "n": 0,
                    "errors": sum(1 for row in rows if row.get("error")),
                }
            )
    if chosen is None and sweep:
        eligible = [item for item in sweep if item.get("recall_lcb95") is not None]
        if eligible:
            chosen = {**max(eligible, key=lambda item: item["recall_lcb95"]), "met_target": False}
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
        "plan_item": "Q3_SWEEP_ACORN_2X2",
        "arms": list(ARMS),
        "filters": names,
        "query_offset": QUERY_OFFSET,
        "query_count": int(args.query_count),
        "target_recall_lcb95": q3.TARGET_LCB,
        "sweep_efs": list(q3.STOCK_GUIDE_EFS),
        "acorn_efs": list(q3.ACORN_EFS),
        "out_dir": str(args.out_dir),
    }
    if not args.execute:
        print(json.dumps({"dry_run": True, **contract}, indent=2))
        return 0

    fig5.set_cohort(QUERY_OFFSET, int(args.query_count))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    require_psycopg()
    import psycopg

    workload = q3._workload()
    specs = {spec.name: spec for spec in bench.read_filters(bench.DEFAULT_FILTERS, set(names))}
    cfg = pg_config_from_env()
    score_path = args.out_dir / "score.json"
    prior = (
        json.loads(score_path.read_text(encoding="utf-8"))
        if args.resume and score_path.exists()
        else {}
    )
    cells = list(prior.get("cells") or [])
    sweep = list(prior.get("sweep") or [])
    done = {str(cell["filter_name"]) for cell in cells if all(cell.get(arm) for arm in ARMS)}
    query_nos = list(range(QUERY_OFFSET, QUERY_OFFSET + int(args.query_count)))

    def _checkpoint() -> None:
        payload = {**contract, "cells": cells, "sweep": sweep}
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
            embed_ids = [query_ids[q] for q in query_nos if q in query_ids]
            embeddings = bench.load_query_embeddings(cur, fig5.TABLE, embed_ids)
            print(json.dumps({"progress": "filter", "filter": name}), flush=True)
            cell: dict[str, Any] = {"filter_name": name}
            for arm, efs in (
                ("stock_sweep", q3.STOCK_GUIDE_EFS),
                ("sqlens_sweep", q3.STOCK_GUIDE_EFS),
                ("stock_acorn", q3.ACORN_EFS),
                ("sqlens_acorn", q3.ACORN_EFS),
            ):
                _rows, chosen, mode_sweep = calibrate_arm(
                    cur, workload, spec, arm, query_ids, embeddings, truth, as_of, query_nos, efs
                )
                sweep.extend(mode_sweep)
                cell[arm] = chosen
            if all(cell.get(arm) for arm in ARMS):
                ss = cell["stock_sweep"]["mean_ms"]
                qs = cell["sqlens_sweep"]["mean_ms"]
                sa = cell["stock_acorn"]["mean_ms"]
                qa = cell["sqlens_acorn"]["mean_ms"]
                cell["sqlens_sweep_vs_stock_sweep"] = ss / qs if qs else None
                cell["sqlens_acorn_vs_stock_acorn"] = sa / qa if qa else None
                cell["all_met_target"] = all(bool(cell[arm].get("met_target")) for arm in ARMS)
            cells = [item for item in cells if item.get("filter_name") != name]
            cells.append(cell)
            _checkpoint()
            print(
                json.dumps(
                    {
                        "progress": "filter_done",
                        "filter": name,
                        "sqlens_sweep_vs_stock_sweep": cell.get("sqlens_sweep_vs_stock_sweep"),
                        "sqlens_acorn_vs_stock_acorn": cell.get("sqlens_acorn_vs_stock_acorn"),
                        "all_met_target": cell.get("all_met_target"),
                    }
                ),
                flush=True,
            )

    _checkpoint()
    (args.out_dir / "manifest.json").write_text(
        json.dumps({**contract, "cells": len(cells)}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"wrote": str(args.out_dir), "paper_eligible": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
