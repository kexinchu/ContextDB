#!/usr/bin/env python3
"""Summarize a paired YFCC iso@0.991 Stock/SQLens run and gate recall match."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


STOCK = "original"
SQLENS = "design1_bloom_bfs_layout_d3"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    idx = (len(ys) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ys) - 1)
    frac = idx - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def geomean(xs: list[float]) -> float:
    vals = [x for x in xs if x > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(x) for x in vals) / len(vals))


def summarize_mode(rows: list[dict[str, str]], mode: str) -> dict[str, float]:
    recalls: list[float] = []
    lats: list[float] = []
    for row in rows:
        if row["mode"] != mode:
            continue
        recalls.append(float(row["recall"]))
        lats.append(float(row["end_to_end_ms"]))
    return {
        "n": float(len(recalls)),
        "recall": mean(recalls),
        "mean_ms": mean(lats),
        "p95_ms": percentile(lats, 95.0),
        "p99_ms": percentile(lats, 99.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--target-center", type=float, default=0.991)
    parser.add_argument(
        "--max-center-dev",
        type=float,
        default=0.004,
        help="Each arm must land within this of --target-center.",
    )
    parser.add_argument(
        "--max-recall-gap",
        type=float,
        default=0.003,
        help="Pass if abs(stock_recall - sqlens_recall) <= this.",
    )
    args = parser.parse_args()

    with args.paired_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    stock = summarize_mode(rows, STOCK)
    sqlens = summarize_mode(rows, SQLENS)
    if stock["n"] < 1 or sqlens["n"] < 1:
        raise SystemExit("missing stock/sqlens rows")

    gap = abs(stock["recall"] - sqlens["recall"])
    stock_dev = abs(stock["recall"] - args.target_center)
    sqlens_dev = abs(sqlens["recall"] - args.target_center)
    speedup_mean = (
        stock["mean_ms"] / sqlens["mean_ms"] if sqlens["mean_ms"] > 0 else float("nan")
    )

    by_filter: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"sr": [], "qr": [], "sl": [], "ql": []}
    )
    for row in rows:
        fn = row["filter_name"]
        r = float(row["recall"])
        lat = float(row["end_to_end_ms"])
        if row["mode"] == STOCK:
            by_filter[fn]["sr"].append(r)
            by_filter[fn]["sl"].append(lat)
        elif row["mode"] == SQLENS:
            by_filter[fn]["qr"].append(r)
            by_filter[fn]["ql"].append(lat)

    filters = []
    speedups = []
    for fn in sorted(by_filter):
        b = by_filter[fn]
        sm = mean(b["sl"])
        qm = mean(b["ql"])
        spd = (sm / qm) if qm > 0 else float("nan")
        if spd == spd:
            speedups.append(spd)
        filters.append(
            {
                "filter_name": fn,
                "stock_recall": mean(b["sr"]),
                "sqlens_recall": mean(b["qr"]),
                "stock_mean_ms": sm,
                "sqlens_mean_ms": qm,
                "speedup": spd,
            }
        )

    center_pass = stock_dev <= args.max_center_dev and sqlens_dev <= args.max_center_dev
    gap_pass = gap <= args.max_recall_gap
    passed = center_pass and gap_pass

    summary = {
        "artifact_type": "table6_yfcc_target099_iso991_summary_v1",
        "source": str(args.paired_csv),
        "target_center": args.target_center,
        "max_center_dev": args.max_center_dev,
        "max_recall_gap": args.max_recall_gap,
        "stock_recall": stock["recall"],
        "sqlens_recall": sqlens["recall"],
        "recall_gap_abs": gap,
        "stock_center_dev": stock_dev,
        "sqlens_center_dev": sqlens_dev,
        "stock_mean_ms": stock["mean_ms"],
        "sqlens_mean_ms": sqlens["mean_ms"],
        "stock_p95_ms": stock["p95_ms"],
        "sqlens_p95_ms": sqlens["p95_ms"],
        "stock_p99_ms": stock["p99_ms"],
        "sqlens_p99_ms": sqlens["p99_ms"],
        "speedup_mean": speedup_mean,
        "speedup_geomean": geomean(speedups),
        "filters_sqlens_faster": sum(1 for s in speedups if s > 1.0),
        "filters": filters,
        "iso_recall_pass": passed,
        "center_pass": center_pass,
        "gap_pass": gap_pass,
        "stock_n": stock["n"],
        "sqlens_n": sqlens["n"],
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "filters"}, indent=2))
    print(
        f"[gate] pass={passed} R={stock['recall']:.5f}/{sqlens['recall']:.5f} "
        f"gap={gap:.5f} center_dev={stock_dev:.5f}/{sqlens_dev:.5f} "
        f"spd_mean={speedup_mean:.3f}x geo={summary['speedup_geomean']:.3f}x"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
