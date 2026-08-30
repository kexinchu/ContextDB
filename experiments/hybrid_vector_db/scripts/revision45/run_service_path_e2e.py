#!/usr/bin/env python3
"""Service-path matched-recall on Amazon-14: stock vs d1_d3 (source, no BFS).

Calibrate q20--q99. Measure q200--q10199. New directory only.
Does not rewrite Table 5, Figure 6, or the published FragReuse replay.
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
OUT_DIR = ROOT / "results/hybrid_vector_db/revision45/service_path_e2e"
CALIB_NOS = list(range(20, 100))
MEASURE_OFFSET = 200
MEASURE_COUNT = 10_000
TARGETS = (0.90, 0.95, 0.99)
MODES = ("stock", "d1_d3")
EFS = q3.STOCK_GUIDE_EFS


def _pick_efs(sweep: list[dict[str, Any]], targets: tuple[float, ...]) -> dict[str, Any]:
    chosen: dict[str, Any] = {}
    ordered = sorted(
        [row for row in sweep if row.get("recall_lcb95") is not None],
        key=lambda row: int(row["ef_search"]),
    )
    for target in targets:
        hit = next(
            (
                row
                for row in ordered
                if float(row["recall_lcb95"]) >= target and int(row.get("n") or 0) > 0
            ),
            None,
        )
        key = f"{target:.2f}"
        chosen[key] = (
            {**hit, "met_target": True, "target": target}
            if hit
            else {"met_target": False, "target": target}
        )
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--filter-names", nargs="*", default=list(q3.FILTER_NAMES))
    parser.add_argument("--targets", nargs="*", type=float, default=list(TARGETS))
    parser.add_argument("--query-count", type=int, default=MEASURE_COUNT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    names = args.filter_names or list(q3.FILTER_NAMES)
    targets = tuple(float(value) for value in args.targets)
    measure_nos = list(range(MEASURE_OFFSET, MEASURE_OFFSET + int(args.query_count)))
    contract = {
        "paper_eligible": False,
        "plan_item": "SERVICE_PATH_E2E",
        "adds_results": True,
        "rewrites_table5": False,
        "modes": list(MODES),
        "filters": names,
        "targets": list(targets),
        "calib_nos": [CALIB_NOS[0], CALIB_NOS[-1]],
        "query_offset": MEASURE_OFFSET,
        "query_count": int(args.query_count),
        "efs": list(EFS),
        "vector_index": bench.DEFAULT_SOURCE_INDEX,
        "out_dir": str(args.out_dir),
    }
    if not args.execute:
        print(json.dumps({"dry_run": True, **contract}, indent=2))
        return 0

    fig5.set_cohort(MEASURE_OFFSET, int(args.query_count))
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
    cells = [cell for cell in (prior.get("cells") or []) if cell.get("measured")]
    sweep = list(prior.get("sweep") or [])
    done = {(str(cell["filter_name"]), str(cell["mode"])) for cell in cells}
    all_rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            all_rows.extend(
                row
                for row in csv.DictReader(handle)
                if (row.get("filter_name"), row.get("mode")) in done
            )

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

        cur.execute("SET work_mem = '4MB'")
        cur.execute("SHOW shared_buffers")
        contract["shared_buffers_live"] = str(cur.fetchone()[0])
        cur.execute("SELECT vector_sqlens_build_id()")
        contract["build_id"] = str(cur.fetchone()[0])
        try:
            for name in names:
                spec = specs[name]
                query_ids, truth, as_of = load_attr_truth(name)
                wanted = list(CALIB_NOS) + measure_nos
                embed_ids = [
                    query_ids[query_no]
                    for query_no in wanted
                    if query_no in query_ids
                ]
                embeddings = bench.load_query_embeddings(cur, fig5.TABLE, embed_ids)
                for mode in MODES:
                    if (name, mode) in done:
                        print(
                            json.dumps(
                                {
                                    "progress": "resume_skip",
                                    "filter": name,
                                    "mode": mode,
                                }
                            ),
                            flush=True,
                        )
                        continue
                    print(
                        json.dumps(
                            {"progress": "filter_mode", "filter": name, "mode": mode}
                        ),
                        flush=True,
                    )
                    cur.execute("RESET ROLE")
                    bench.clear_fragment_store(cur, fig5.TABLE)
                    _calib_rows, _chosen_090, mode_sweep = q3.calibrate_mode(
                        cur,
                        workload,
                        spec,
                        mode,
                        query_ids,
                        embeddings,
                        truth,
                        as_of,
                        CALIB_NOS,
                        EFS,
                    )
                    # Re-sweep remaining efs if 0.90 was met early and a
                    # higher target still needs a larger ef.
                    have = {int(row["ef_search"]) for row in mode_sweep if "ef_search" in row}
                    for ef in EFS:
                        if ef in have:
                            continue
                        if all(
                            item.get("met_target")
                            for item in _pick_efs(mode_sweep, targets).values()
                        ):
                            break
                        more_rows, more_chosen, more_sweep = q3.calibrate_mode(
                            cur,
                            workload,
                            spec,
                            mode,
                            query_ids,
                            embeddings,
                            truth,
                            as_of,
                            CALIB_NOS,
                            (ef,),
                        )
                        mode_sweep.extend(more_sweep)
                    sweep.extend(mode_sweep)
                    picked = _pick_efs(mode_sweep, targets)
                    measured: dict[str, Any] = {}
                    for target in targets:
                        key = f"{target:.2f}"
                        pick = picked[key]
                        if not pick.get("met_target"):
                            measured[key] = pick
                            continue
                        ef = int(pick["ef_search"])
                        if any(
                            measured.get(other, {}).get("ef_search") == ef
                            and measured[other].get("n")
                            for other in measured
                        ):
                            prior_key = next(
                                other
                                for other in measured
                                if measured[other].get("ef_search") == ef
                            )
                            measured[key] = {
                                **measured[prior_key],
                                "target": target,
                                "shared_with": prior_key,
                            }
                            continue
                        print(
                            json.dumps(
                                {
                                    "progress": "measure",
                                    "filter": name,
                                    "mode": mode,
                                    "target": target,
                                    "ef": ef,
                                    "n": len(measure_nos),
                                }
                            ),
                            flush=True,
                        )
                        cur.execute("RESET ROLE")
                        bench.clear_fragment_store(cur, fig5.TABLE)
                        bench.set_heap_competing_indexes_valid(
                            cur, fig5.TABLE, valid=False
                        )
                        fig5.prepare_pg(cur)
                        cur.execute("SET work_mem = '4MB'")
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
                            stats["met_target"] = stats["recall_lcb95"] >= target
                            stats["target"] = target
                            measured[key] = stats
                        else:
                            measured[key] = {**pick, "measured": False}
                    cells.append(
                        {
                            "filter_name": name,
                            "mode": mode,
                            "measured": True,
                            "calib": picked,
                            "targets": measured,
                        }
                    )
                    done.add((name, mode))
                    _checkpoint()
        finally:
            try:
                cur.execute("RESET ROLE")
                bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=True)
            except Exception:
                pass

    _checkpoint()
    print(json.dumps({"wrote": str(args.out_dir), "rewrites_published": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
