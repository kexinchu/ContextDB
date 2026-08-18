#!/usr/bin/env python3
"""Stitch frozen Stock + remasured SQLens for YFCC target-0.99 iso-recall."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


STOCK_MODE = "original"
SQLENS_MODE = "design1_bloom_bfs_layout_d3"
PAIR_KEY = ("query_no", "query_id", "filter_name")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def key_of(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[k] for k in PAIR_KEY)  # type: ignore[return-value]


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


def summarize(rows: list[dict[str, str]], mode: str) -> dict[str, float]:
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


def per_filter(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"stock_r": [], "sqlens_r": [], "stock_l": [], "sqlens_l": []}
    )
    for row in rows:
        fn = row["filter_name"]
        r = float(row["recall"])
        lat = float(row["end_to_end_ms"])
        if row["mode"] == STOCK_MODE:
            buckets[fn]["stock_r"].append(r)
            buckets[fn]["stock_l"].append(lat)
        elif row["mode"] == SQLENS_MODE:
            buckets[fn]["sqlens_r"].append(r)
            buckets[fn]["sqlens_l"].append(lat)
    out: list[dict[str, object]] = []
    for fn in sorted(buckets):
        b = buckets[fn]
        sm = mean(b["stock_l"])
        qm = mean(b["sqlens_l"])
        out.append(
            {
                "filter_name": fn,
                "queries": len(b["stock_r"]),
                "stock_recall": mean(b["stock_r"]),
                "sqlens_recall": mean(b["sqlens_r"]),
                "recall_gap": mean(b["stock_r"]) - mean(b["sqlens_r"]),
                "stock_mean_ms": sm,
                "sqlens_mean_ms": qm,
                "speedup": (sm / qm) if qm > 0 else float("nan"),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stock-source",
        type=Path,
        default=Path(
            "results/hybrid_vector_db/table6_r41_yfcc_target099/"
            "yfcc_target099_paired_q10k.csv"
        ),
    )
    parser.add_argument("--sqlens-source", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument(
        "--max-recall-gap",
        type=float,
        default=0.00015,
        help="Pass if stock_recall - sqlens_recall <= this (SQLens may slightly exceed).",
    )
    args = parser.parse_args()

    stock_rows = [
        r for r in read_rows(args.stock_source) if r["mode"] == STOCK_MODE
    ]
    sqlens_rows = [
        r for r in read_rows(args.sqlens_source) if r["mode"] == SQLENS_MODE
    ]
    if len(stock_rows) != 10000:
        raise SystemExit(f"expected 10000 stock rows, got {len(stock_rows)}")
    if len(sqlens_rows) != 10000:
        raise SystemExit(f"expected 10000 sqlens rows, got {len(sqlens_rows)}")

    stock_by = {key_of(r): r for r in stock_rows}
    sqlens_by = {key_of(r): r for r in sqlens_rows}
    missing = [k for k in stock_by if k not in sqlens_by]
    extra = [k for k in sqlens_by if k not in stock_by]
    if missing or extra:
        raise SystemExit(
            f"pairing mismatch: missing_sqlens={len(missing)} extra_sqlens={len(extra)}"
        )

    # Restore stock request_no onto sqlens rows for frozen-trace pairing.
    paired: list[dict[str, str]] = []
    fieldnames = list(stock_rows[0].keys())
    for key, stock in stock_by.items():
        sqlens = dict(sqlens_by[key])
        if "request_no" in stock and "request_no" in sqlens:
            sqlens["request_no"] = stock["request_no"]
        paired.append(stock)
        paired.append(sqlens)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(paired)

    stock_s = summarize(paired, STOCK_MODE)
    sqlens_s = summarize(paired, SQLENS_MODE)
    gap = stock_s["recall"] - sqlens_s["recall"]
    speedup = (
        stock_s["mean_ms"] / sqlens_s["mean_ms"]
        if sqlens_s["mean_ms"] > 0
        else float("nan")
    )
    filters = per_filter(paired)
    # Pass if SQLens reaches stock within tolerance, or slightly exceeds stock.
    passed = gap <= args.max_recall_gap
    summary = {
        "artifact_type": "table6_yfcc_target099_isomatch_summary_v1",
        "stock_source": str(args.stock_source),
        "stock_source_sha256": sha256_file(args.stock_source),
        "sqlens_source": str(args.sqlens_source),
        "sqlens_source_sha256": sha256_file(args.sqlens_source),
        "output": str(args.out_csv),
        "output_sha256": sha256_file(args.out_csv),
        "target_recall": 0.99,
        "stock_recall": stock_s["recall"],
        "sqlens_recall": sqlens_s["recall"],
        "recall_gap_stock_minus_sqlens": gap,
        "max_recall_gap": args.max_recall_gap,
        "iso_recall_pass": passed,
        "stock_mean_ms": stock_s["mean_ms"],
        "sqlens_mean_ms": sqlens_s["mean_ms"],
        "stock_p95_ms": stock_s["p95_ms"],
        "sqlens_p95_ms": sqlens_s["p95_ms"],
        "stock_p99_ms": stock_s["p99_ms"],
        "sqlens_p99_ms": sqlens_s["p99_ms"],
        "speedup_mean": speedup,
        "filters": filters,
        "filters_sqlens_below_stock": sum(
            1 for f in filters if float(f["recall_gap"]) > 1e-12
        ),
    }
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "filters"}, indent=2))
    print(
        f"[gate] iso_recall_pass={passed} "
        f"R={stock_s['recall']:.5f}/{sqlens_s['recall']:.5f} gap={gap:.5f} "
        f"speedup={speedup:.3f}x"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
