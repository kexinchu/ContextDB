#!/usr/bin/env python3
"""Export YFCC matched-iso warm rows into Figure 6 long latency CSV.

Reads ``yfcc10m_matched_iso_warm_summary.csv`` (+ optional per-filter detail)
and writes an Amazon-compatible long table for plotting / coverage audits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY = (
    ROOT
    / "results/hybrid_vector_db/yfcc10m_v3_matched_iso_warm/"
    / "yfcc10m_matched_iso_warm_summary.csv"
)
DEFAULT_OUT = (
    ROOT
    / "results/hybrid_vector_db/yfcc10m_v3_iso_recall_plot/"
    / "yfcc10m_iso_recall_pairs_long.csv"
)
DEFAULT_TARGETS = (0.75, 0.8, 0.85, 0.9, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--targets",
        default=",".join(str(x) for x in DEFAULT_TARGETS),
    )
    args = parser.parse_args()
    targets = {float(x) for x in args.targets.split(",") if x.strip()}
    rows = list(csv.DictReader(args.summary.open(newline="", encoding="utf-8")))
    long_rows: list[dict[str, object]] = []
    for row in rows:
        target = float(row["target_recall"])
        if targets and target not in targets:
            continue
        matched = int(float(row["n_filters_matched"]))
        total = int(float(row["n_filters_total"]))
        for arm, recall_key, lat_key in (
            ("stock", "stock_recall_mean", "stock_latency_mean_ms"),
            ("sqlens", "sqlens_recall_mean", "sqlens_latency_mean_ms"),
        ):
            recall = float(row[recall_key])
            long_rows.append(
                {
                    "dataset": "yfcc10m",
                    "target_recall": target,
                    "arm": arm,
                    "recall": recall,
                    "latency_mean_ms": float(row[lat_key]),
                    "family": "matched_iso_warm",
                    "ef_search": "",
                    "scan_cap": "",
                    "abs_err_vs_target": abs(recall - target),
                    "n_filters_matched": matched,
                    "n_filters_total": total,
                    "metric_source": str(args.summary),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)
    print(f"wrote {args.out} rows={len(long_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
