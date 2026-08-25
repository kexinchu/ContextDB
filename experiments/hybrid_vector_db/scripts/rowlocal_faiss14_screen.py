#!/usr/bin/env python3
"""P2: FAISS allow-list on the 14 row-local Amazon predicates (q1K).

Attribute SQL only — no JOIN compile. Stock/SQLens numbers stay in Table 6.
Not a substitute for a later q10K library confirm.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import amazon10m_sql_native_benchmark as bench
import figure5_hybrid_allowlist_screen as fig5
from amazon10m_matched_recall_baselines import read_fbin_memmap
from common_pg import pg_config_from_env, require_psycopg

ROOT = Path(__file__).resolve().parents[3]
FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
ATTR_GT = fig5.ATTR_GT
OUT_DIR = ROOT / "results/hybrid_vector_db/rowlocal_faiss14_q1k_screen"


def load_attr_truth(filter_name: str) -> tuple[dict[int, int], dict[int, tuple[int, ...]], int]:
    query_ids: dict[int, int] = {}
    truth: dict[int, tuple[int, ...]] = {}
    as_of = 0
    with ATTR_GT.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["filter_name"] != filter_name:
                continue
            query_no = int(row["query_no"])
            query_ids[query_no] = int(row["query_id"])
            if as_of == 0:
                as_of = int(row.get("as_of") or 0)
            truth[query_no] = fig5._parse_ids(row["exact_filtered_topk_ids"])
    wanted = list(fig5.CALIB_QUERY_NOS) + list(
        range(fig5.QUERY_OFFSET, fig5.QUERY_OFFSET + fig5.QUERY_COUNT)
    )
    missing = [query_no for query_no in wanted if query_no not in truth]
    if missing:
        raise RuntimeError(f"{filter_name} missing GT {missing[:6]} count={len(missing)}")
    return query_ids, truth, as_of


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if not args.execute:
        print("dry-run: pass --execute")
        return 0
    fig5.set_cohort(200, 1000)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = [row["filter_name"] for row in csv.DictReader(FILTERS.open(encoding="utf-8"))]
    require_psycopg()
    import faiss
    import psycopg

    cfg = pg_config_from_env()
    vectors, _, _ = read_fbin_memmap(fig5.FBIN)
    index = faiss.read_index(str(fig5.FAISS_INDEX))
    workload = fig5._workload("attributes", "none")
    rows: list[dict] = []
    summary: list[dict] = []
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        fig5.prepare_pg(cur)
        bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=True)
        for name in names:
            spec = bench.read_filters(bench.DEFAULT_FILTERS, {name})[0]
            query_ids, truth, _as_of = load_attr_truth(name)
            sql_text = fig5.allowlist_sql(spec, workload)
            print(json.dumps({"progress": "allowlist_start", "filter": name}), flush=True)
            allow = fig5.build_allow_list(conn, faiss, sql_text, int(index.ntotal))
            ef, attained, bound = fig5.choose_faiss_ef(
                index, faiss, vectors, allow["selector"], query_ids, truth
            )
            print(
                json.dumps(
                    {
                        "progress": "faiss_selected",
                        "filter": name,
                        "ef": ef,
                        "lcb": bound,
                        "attained": attained,
                        "allow_ms": allow["build_ms"],
                        "rows": allow["rows"],
                    }
                ),
                flush=True,
            )
            measured = fig5.run_faiss_shape(
                index,
                faiss,
                vectors,
                allow["selector"],
                float(allow["build_ms"]),
                ef,
                "attributes",
                query_ids,
                truth,
            )
            for row in measured:
                row["filter_name"] = name
            rows.extend(measured)
            ok = [row for row in measured if not row.get("error") and row.get("e2e_ms") != ""]
            search = [float(row["search_ms"]) for row in ok]
            recalls = [float(row["recall"]) for row in ok]
            summary.append(
                {
                    "filter_name": name,
                    "ef": ef,
                    "lcb_attained": attained,
                    "calib_lcb95": bound,
                    "allow_ms": allow["build_ms"],
                    "allow_rows": allow["rows"],
                    "search_ms": sum(search) / len(search) if search else None,
                    "e2e_cold_ms": (
                        float(allow["build_ms"]) + sum(search) / len(search) if search else None
                    ),
                    "recall": sum(recalls) / len(recalls) if recalls else None,
                    "n": len(ok),
                }
            )
        cur.execute("RESET ROLE")
    with (args.out_dir / "faiss14.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "score.json").write_text(
        json.dumps({"paper_eligible": False, "queries": 1000, "cells": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out_dir), "cells": summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
