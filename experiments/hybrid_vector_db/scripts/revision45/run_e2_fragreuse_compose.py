#!/usr/bin/env python3
"""E2: forced-kind composition on three Amazon conjunctions.

q200--q1199, ef=100, VisGuide on the source index. New directory.
Does not rewrite Table 5, Figure 6, or the published FragReuse replay.
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
sys.path.insert(0, str(SCRIPT_DIR))

import amazon10m_sql_native_benchmark as bench
import figure5_hybrid_allowlist_screen as fig5
import run_q3_acorn_matched as q3
from common_pg import pg_config_from_env, require_psycopg

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "results/hybrid_vector_db/revision45/e2_fragreuse_compose"
MEASURE_NOS = list(range(200, 1200))
EF = 100
K = 10

GROCERY = "sql:main_category = 'Grocery'"
HELPFUL = "sql:helpful_vote >= 1"
RATING5 = "sql:rating = 5"
LONG500 = "sql:review_text_len >= 500"

CASES = {
    "grocery_helpful": {
        "other": HELPFUL,
        "combo": "sql:main_category = 'Grocery' AND helpful_vote >= 1",
    },
    "grocery_rating5": {
        "other": RATING5,
        "combo": "sql:main_category = 'Grocery' AND rating = 5",
    },
    "grocery_long500": {
        "other": LONG500,
        "combo": "sql:main_category = 'Grocery' AND review_text_len >= 500",
    },
}

# compose_complete: None leaves the GUC alone. False forces the requested
# kind (needed for page∘page). True is the r44 default that upgrades
# multi-atom page to Bloom.
ARMS = (
    ("stock", None, (), None),
    ("grocery_exact", "exact", (GROCERY,), False),
    ("other_exact", "exact", "OTHER", False),
    ("exact_compose", "exact", "BOTH", False),
    ("page_compose", "page", "BOTH", False),
    ("r44_upgrade", "page", "BOTH", True),
    ("and_exact", "exact", "COMBO", False),
)


def _atoms(name: str, spec_atoms: tuple[str, ...] | str) -> tuple[str, ...]:
    case = CASES[name]
    if spec_atoms == "OTHER":
        return (case["other"],)
    if spec_atoms == "BOTH":
        return (GROCERY, case["other"])
    if spec_atoms == "COMBO":
        return (case["combo"],)
    return spec_atoms


def _summarize(rows: list[dict]) -> dict[str, Any] | None:
    ok = [
        row
        for row in rows
        if row.get("phase") == "measurement"
        and not row.get("error")
        and row.get("e2e_ms") not in ("", None)
        and row.get("recall") not in ("", None)
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
        "ef_search": EF,
    }


def _set_compose_complete(cur: Any, compose_complete: bool | None) -> None:
    if compose_complete is None:
        return
    value = "on" if compose_complete else "off"
    cur.execute(f"SET hnsw.d3_compose_complete_only = {value}")


def run_arm(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth: dict[int, tuple[int, ...]],
    as_of: int,
    arm: str,
    kind: str | None,
    atoms: tuple[str, ...],
    compose_complete: bool | None,
) -> list[dict]:
    vector_index = bench.DEFAULT_SOURCE_INDEX
    mode = "stock" if arm == "stock" else "d1"
    config = bench.Config(EF, 5_000_000, 32.0, "relaxed_order", EF)
    bench.set_mode(cur, mode, config, vector_index)
    _set_compose_complete(cur, compose_complete)
    sql_text = bench.build_hybrid_sql(fig5.TABLE, spec.predicate, workload=workload)
    bind_atoms = bench.binding_atoms_for(workload, spec)
    rows: list[dict[str, Any]] = []
    for i, query_no in enumerate(MEASURE_NOS):
        query_id = query_ids[query_no]
        params = bench.bind_query_embedding(
            {
                "query_id": query_id,
                "as_of": as_of,
                "k": K,
                "vector_index": vector_index,
                "binding_atoms": list(bind_atoms),
                "binding_kind": kind or "bloom",
            },
            query_id,
            embeddings,
        )
        error = ""
        ids: list[int] = []
        e2e_ms = 0.0
        try:
            cur.execute("SELECT vector_hnsw_reset_scan_profile()")
            started = time.perf_counter()
            bench.set_as_of(cur, as_of)
            if mode != "stock":
                bench.configure_guidance(
                    cur, mode, vector_index, atoms, guidance_kind=kind
                )
            cur.execute(sql_text, params)
            fetched = cur.fetchall()
            ids = [int(row[0]) for row in fetched]
            e2e_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            error = f"{exc.__class__.__name__}: {exc}"
            try:
                cur.execute("ROLLBACK")
            except Exception:
                pass
        truth_ids = truth.get(query_no, ())
        recall = (
            len(set(ids) & set(truth_ids)) / float(K)
            if truth_ids and not error
            else ""
        )
        rows.append(
            {
                "phase": "measurement",
                "filter_name": spec.name,
                "arm": arm,
                "kind": kind or "none",
                "compose_complete": (
                    "" if compose_complete is None else str(compose_complete)
                ),
                "atoms": "|".join(atoms),
                "query_no": query_no,
                "query_id": query_id,
                "ef_search": EF,
                "e2e_ms": "" if error else f"{e2e_ms:.6f}",
                "recall": "" if recall == "" else f"{recall:.6f}",
                "error": error,
            }
        )
        if (i + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "progress": "measurement",
                        "filter": spec.name,
                        "arm": arm,
                        "completed": i + 1,
                        "total": len(MEASURE_NOS),
                    }
                ),
                flush=True,
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--filters", nargs="*", default=list(CASES))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    contract = {
        "paper_eligible": False,
        "plan_item": "E2_FRAGREUSE_COMPOSE",
        "adds_results": True,
        "rewrites_table5": False,
        "filters": args.filters,
        "query_offset": 200,
        "query_count": 1000,
        "ef_search": EF,
        "out_dir": str(args.out_dir),
    }
    if not args.execute:
        print(json.dumps({"dry_run": True, **contract}, indent=2))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    require_psycopg()
    import psycopg

    fig5.set_cohort(200, 1000)
    workload = q3._workload()
    specs = {
        spec.name: spec
        for spec in bench.read_filters(bench.DEFAULT_FILTERS, set(args.filters))
    }
    score_path = args.out_dir / "score.json"
    csv_path = args.out_dir / "screen.csv"
    prior = (
        json.loads(score_path.read_text(encoding="utf-8"))
        if args.resume and score_path.exists()
        else {}
    )
    cells = [cell for cell in (prior.get("cells") or []) if cell.get("measured")]
    done = {(cell["filter_name"], cell["arm"]) for cell in cells}
    all_rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            all_rows.extend(
                row
                for row in csv.DictReader(handle)
                if (row.get("filter_name"), row.get("arm")) in done
            )

    def _checkpoint() -> None:
        if all_rows:
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
        payload = {**contract, "cells": cells, "rows": len(all_rows)}
        score_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    cfg = pg_config_from_env()
    from rowlocal_faiss14_screen import load_attr_truth

    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        try:
            for name in args.filters:
                spec = specs[name]
                query_ids, truth, as_of = load_attr_truth(name)
                embed_ids = [
                    query_ids[query_no]
                    for query_no in MEASURE_NOS
                    if query_no in query_ids
                ]
                embeddings = bench.load_query_embeddings(cur, fig5.TABLE, embed_ids)
                for arm, kind, spec_atoms, compose_complete in ARMS:
                    if (name, arm) in done:
                        print(
                            json.dumps(
                                {"progress": "resume_skip", "filter": name, "arm": arm}
                            ),
                            flush=True,
                        )
                        continue
                    atoms = _atoms(name, spec_atoms)
                    print(
                        json.dumps(
                            {
                                "progress": "arm",
                                "filter": name,
                                "arm": arm,
                                "kind": kind,
                                "compose_complete": compose_complete,
                                "atoms": list(atoms),
                            }
                        ),
                        flush=True,
                    )
                    cur.execute("RESET ROLE")
                    bench.clear_fragment_store(cur, fig5.TABLE)
                    bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=False)
                    fig5.prepare_pg(cur)
                    _set_compose_complete(cur, compose_complete)
                    rows = run_arm(
                        cur,
                        workload,
                        spec,
                        query_ids,
                        embeddings,
                        truth,
                        as_of,
                        arm,
                        kind,
                        atoms,
                        compose_complete,
                    )
                    all_rows.extend(rows)
                    summary = _summarize(rows)
                    cells.append(
                        {
                            "filter_name": name,
                            "arm": arm,
                            "kind": kind,
                            "compose_complete": compose_complete,
                            "atoms": list(atoms),
                            "measured": summary is not None,
                            **(summary or {}),
                        }
                    )
                    _checkpoint()
        finally:
            try:
                cur.execute("RESET ROLE")
                bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=True)
            except Exception:
                pass
    print(json.dumps({"wrote": str(args.out_dir), "rewrites_published": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
