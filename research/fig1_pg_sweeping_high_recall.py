"""Extra PG-sweeping points on the frozen 5% workload.

Reuses research/results/fig1_four_curve_m32_5pct/workload.csv.
Does not overwrite the mixed freeze or the original 5% frontier_summary.

Two protocols, same overfetch=ef grid:
  off          — same as the 5% aligned run (single HNSW scan)
  strict_order — iterative_scan fills LIMIT so overfetch is actually returned
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import psycopg

from fig1_four_curve_frontier import (
    DEFAULT_FBIN,
    DEFAULT_TABLE,
    attach_recall,
    build_ground_truth,
    pg_sweeping_run,
    plot_frontier,
    read_fbin,
    select_iso_recall,
    summarize_frontier,
    write_csv,
)
from pg_conn import pg_conninfo
from hnswlib_vs_pgvector_selectivity import pg_configure

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "research/results/fig1_four_curve_m32_5pct"
DEFAULT_OVERFETCHES = [2400, 3200, 4800, 8000, 10000]
DEFAULT_MODES = ["off", "strict_order"]
TARGETS = [0.75, 0.80, 0.85, 0.90, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.997]


def load_workload(path: Path) -> list[dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load_existing_summary(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overfetches", default=",".join(map(str, DEFAULT_OVERFETCHES)))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--timeout-ms", type=int, default=60000)
    args = parser.parse_args()
    out_dir = args.out_dir
    overfetches = [int(x) for x in args.overfetches.split(",") if x]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    workload = load_workload(out_dir / "workload.csv")
    vectors = read_fbin(DEFAULT_FBIN, 200000)
    n = len(vectors)
    print(f"reused workload {out_dir / 'workload.csv'} queries={len(workload)}", flush=True)
    print("building exact ground truth", flush=True)
    gt = build_ground_truth(vectors, workload, 10)
    print(f"overfetches={overfetches} modes={modes}", flush=True)

    extra_rows: list[dict] = []
    with psycopg.connect(pg_conninfo("55438"), autocommit=True) as conn:
        cur = conn.cursor()
        for mode in modes:
            pg_configure(cur, overfetches[0], 200000, args.timeout_ms, mode, "off", 128, "off")
            cur.execute(f"SET hnsw.iterative_scan = {mode}")
            cur.execute("SET hnsw.filter_strategy = off")
            cur.execute("SHOW hnsw.iterative_scan")
            print(f"iterative_scan={cur.fetchone()[0]}", flush=True)
            scored = attach_recall(
                pg_sweeping_run(
                    cur, DEFAULT_TABLE, vectors, workload, overfetches, 10, n, args.timeout_ms,
                    overfetches=overfetches,
                ),
                gt,
                10,
            )
            for row in scored:
                row["iterative_scan"] = mode
                if mode != "off":
                    row["system"] = f"PGVector-sweeping-{mode}"
            extra_rows.extend(scored)
            write_csv(out_dir / f"pg_sweeping_high_recall_{mode}.csv", scored)
            summary = summarize_frontier(scored)
            print(f"=== PG-sweeping iterative={mode} ===", flush=True)
            for row in summary:
                print(
                    f"  of={row['ef_search']:<5} recall={float(row['recall_at_10_mean']):.4f} "
                    f"lat={float(row['latency_ms_mean']):.2f}ms n={row['queries']}",
                    flush=True,
                )

    write_csv(out_dir / "pg_sweeping_high_recall_detail.csv", extra_rows)
    extra_summary = summarize_frontier(extra_rows)
    write_csv(out_dir / "pg_sweeping_high_recall_summary.csv", extra_summary)

    base = load_existing_summary(out_dir / "frontier_summary.csv")
    merged = list(base) + extra_summary
    write_csv(out_dir / "frontier_summary_with_high_pg.csv", merged)
    selected = select_iso_recall(extra_summary, TARGETS)
    write_csv(out_dir / "pg_sweeping_high_recall_iso.csv", selected)
    plot_frontier(
        extra_summary,
        selected,
        out_dir / "fig_pg_sweeping_high_recall.pdf",
    )
    print("iso-recall on extra PG-sweeping points:", flush=True)
    for row in selected:
        print(
            f"  {row['system']:28s} t={row['target_recall']} "
            f"status={row['status']} of={row['ef_search']} "
            f"recall={row['recall_at_10_mean']} lat={row['latency_ms_mean']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
