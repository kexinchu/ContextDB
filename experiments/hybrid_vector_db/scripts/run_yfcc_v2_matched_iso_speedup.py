#!/usr/bin/env python3
"""YFCC v3 filters: bounded per-filter matched iso-recall Stock vs SQLens.

For each target recall T and each predicate, independently pick the cheapest
Stock config and cheapest SQLens config inside [T-0.01, min(1, T+0.05)], then
compare latencies.  Aggregate speedup is geomean(stock_e2e / sqlens_e2e).

Gate: speedup >= 1.5x on every requested target (when both arms cover all filters).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/kec23008/Hybrid-Retrieval")
DEFAULT_CFG = ROOT / "experiments/hybrid_vector_db/configs/figure5_r41_yfcc_primary_v3filters.json"
DEFAULT_OUT = ROOT / "results/hybrid_vector_db/figure5_r41_yfcc_v3_matched_iso"
DEFAULT_PLOT = ROOT / "results/hybrid_vector_db/yfcc10m_v3_matched_iso"
LOCK = "results/hybrid_vector_db/.figure5_yfcc_primary_db.lock"
CPU = "48-63"
# Mutable by main() so the same runner can target v3/v4 filter configs.
CFG = DEFAULT_CFG
OUT = DEFAULT_OUT
PLOT = DEFAULT_PLOT
# Figure 6 iso-recall ladder (0.05 in the request was treated as 0.95).
TARGETS = [0.75, 0.80, 0.85, 0.90, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
MIN_SPEEDUP = 1.5
LOWER_RECALL_MARGIN = 0.01
UPPER_RECALL_MARGIN = 0.05
MODE_STOCK = "original"
MODE_SQLENS = "design1_bloom_bfs_layout_d3"
PAT = re.compile(r"calibration_(.+)_ef(\d+)(?:_cap(\d+))?_profile_summary\.csv$")

# Search grids (Stock-only where possible).
# Sparse grids: fill coverage for high-R OR/AND without a dense 15-point ladder.
STOCK_STRICT_EFS = [20, 40, 60, 80, 100, 150, 200, 500, 1000, 2000, 5000]
STOCK_BOTH_OFF_EFS = [200, 500, 1000, 2000, 5000, 10000]
SQLENS_CAPS = [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 100000]
# ef<=20 already measured (legacy); start above for high-sel OR R>=0.85+.
SQLENS_BOTH_OFF_EFS = [25, 30, 40, 50, 70, 90, 100, 150, 200, 250, 500, 750, 1000]


def geomean(xs: list[float]) -> float | None:
    vals = [x for x in xs if x > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(x) for x in vals) / len(vals))


def run_frontier(*, family: str, ef: int, out_dir: Path, caps: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "experiments/hybrid_vector_db/scripts/run_figure5_frontier.py"),
        "--config",
        str(CFG),
        "--phase",
        "calibration",
        "--datasets",
        "yfcc",
        "--grid",
        "base",
        "--scan-families",
        family,
        "--ef-search-values",
        str(ef),
        "--backend-cpu-list",
        CPU,
        "--out-dir",
        str(out_dir),
        "--global-db-lock-path",
        LOCK,
        "--require-global-db-lock",
        "--resume",
        "--overwrite",
        "--execute",
    ]
    if caps is not None:
        cmd.extend(["--sqlens-scan-cap-values", caps])
    if family == "sqlens_cap" or (
        family == "both_off"
        and ef
        not in {20, 40, 60, 80, 100, 150, 200, 250, 500, 750, 1000}
    ):
        cmd.append("--allow-expensive-sqlens-calibration")
    print(f"[run] {family} ef={ef} caps={caps}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"[done] {family} ef={ef} caps={caps} ({time.time()-t0:.0f}s)", flush=True)


def profile_exists(out_dir: Path, family: str, ef: int, cap: int | None) -> bool:
    stem = f"figure5_r35_yfcc_calibration_{family}_ef{ef}"
    if cap is not None:
        stem += f"_cap{cap}"
    return (out_dir / f"{stem}_profile_summary.csv").is_file()


def load_per_filter_points(dirs: list[Path]) -> list[dict]:
    """One row per (arm, family, ef, cap, filter)."""
    points: list[dict] = []
    for directory in dirs:
        if not directory.is_dir():
            continue
        for profile in directory.glob("*_profile_summary.csv"):
            match = PAT.search(profile.name)
            if match is None:
                continue
            family, ef, cap_s = match.group(1), int(match.group(2)), match.group(3)
            cap = int(cap_s) if cap_s else None
            with profile.open() as handle:
                for row in csv.DictReader(handle):
                    mode = row["mode"]
                    if mode == MODE_STOCK:
                        arm = "stock"
                    elif mode == MODE_SQLENS:
                        arm = "sqlens"
                    else:
                        continue
                    if family == "sqlens_cap" and arm == "stock":
                        continue
                    if family == "stock_strict" and arm == "sqlens":
                        continue
                    points.append(
                        {
                            "arm": arm,
                            "family": family,
                            "ef_search": ef,
                            "sqlens_scan_cap": cap if cap is not None else "",
                            "filter_name": row["filter_name"],
                            "recall": float(row["recall_mean"]),
                            "latency_e2e_ms": float(row["end_to_end_mean_ms"]),
                            "latency_query_ms": float(row["query_latency_mean_ms"]),
                        }
                    )
    return points


def pick_filter_config(
    points: list[dict], *, arm: str, filter_name: str, target: float
) -> dict | None:
    lower = max(0.0, target - LOWER_RECALL_MARGIN)
    upper = min(1.0, target + UPPER_RECALL_MARGIN)
    elig = [
        p
        for p in points
        if p["arm"] == arm
        and p["filter_name"] == filter_name
        and p["recall"] + 1e-12 >= lower
        and p["recall"] <= upper + 1e-12
    ]
    if not elig:
        return None
    return min(
        elig,
        key=lambda p: (p["latency_e2e_ms"], abs(p["recall"] - target), p["ef_search"]),
    )


def evaluate(points: list[dict]) -> list[dict]:
    filters = sorted({p["filter_name"] for p in points})
    rows = []
    for target in TARGETS:
        per_filter = []
        missing_stock = []
        missing_sqlens = []
        for fname in filters:
            s = pick_filter_config(points, arm="stock", filter_name=fname, target=target)
            q = pick_filter_config(points, arm="sqlens", filter_name=fname, target=target)
            if s is None:
                missing_stock.append(fname)
            if q is None:
                missing_sqlens.append(fname)
            if s is None or q is None:
                continue
            per_filter.append(
                {
                    "filter_name": fname,
                    "stock_recall": s["recall"],
                    "stock_latency_e2e_ms": s["latency_e2e_ms"],
                    "stock_family": s["family"],
                    "stock_ef_search": s["ef_search"],
                    "sqlens_recall": q["recall"],
                    "sqlens_latency_e2e_ms": q["latency_e2e_ms"],
                    "sqlens_family": q["family"],
                    "sqlens_ef_search": q["ef_search"],
                    "sqlens_cap": q["sqlens_scan_cap"],
                    "speedup": s["latency_e2e_ms"] / q["latency_e2e_ms"]
                    if q["latency_e2e_ms"] > 0
                    else None,
                }
            )
        speedups = [float(x["speedup"]) for x in per_filter if x["speedup"]]
        stock_lats = [x["stock_latency_e2e_ms"] for x in per_filter]
        sqlens_lats = [x["sqlens_latency_e2e_ms"] for x in per_filter]
        g = geomean(speedups) if speedups else None
        rows.append(
            {
                "target_recall": target,
                "n_filters_total": len(filters),
                "n_filters_matched": len(per_filter),
                "missing_stock": ",".join(missing_stock),
                "missing_sqlens": ",".join(missing_sqlens),
                "stock_latency_geomean_ms": geomean(stock_lats),
                "sqlens_latency_geomean_ms": geomean(sqlens_lats),
                "stock_latency_mean_ms": (sum(stock_lats) / len(stock_lats))
                if stock_lats
                else None,
                "sqlens_latency_mean_ms": (sum(sqlens_lats) / len(sqlens_lats))
                if sqlens_lats
                else None,
                "speedup_geomean": g,
                "speedup_mean": (sum(speedups) / len(speedups)) if speedups else None,
                "stock_recall_mean": (
                    sum(x["stock_recall"] for x in per_filter) / len(per_filter)
                    if per_filter
                    else None
                ),
                "sqlens_recall_mean": (
                    sum(x["sqlens_recall"] for x in per_filter) / len(per_filter)
                    if per_filter
                    else None
                ),
                "filters_sqlens_faster": sum(1 for x in speedups if x > 1.0),
                "filters_speedup_ge_1_5": sum(1 for x in speedups if x >= MIN_SPEEDUP),
                "gate_all_filters_matched": len(per_filter) == len(filters) and len(filters) > 0,
                "gate_speedup_ge_1_5": bool(g is not None and g >= MIN_SPEEDUP),
                "pass": bool(
                    len(per_filter) == len(filters)
                    and len(filters) > 0
                    and g is not None
                    and g >= MIN_SPEEDUP
                ),
                "per_filter": per_filter,
            }
        )
    return rows


def print_summary(rows: list[dict]) -> None:
    print("\n=== Bounded per-filter matched iso-recall (v3 filters) ===", flush=True)
    print(
        f"{'T':>5} | {'cov':>6} | {'Stock geo ms':>12} | {'SQLens geo ms':>13} | "
        f"{'spd geo':>7} | {'>=1.5x':>6} | pass",
        flush=True,
    )
    print("-" * 80, flush=True)
    for row in rows:
        cov = f"{row['n_filters_matched']}/{row['n_filters_total']}"
        sg = row["stock_latency_geomean_ms"]
        qg = row["sqlens_latency_geomean_ms"]
        sp = row["speedup_geomean"]
        print(
            f"{row['target_recall']:5.2f} | {cov:>6} | "
            f"{(f'{sg:.1f}' if sg else 'n/a'):>12} | "
            f"{(f'{qg:.1f}' if qg else 'n/a'):>13} | "
            f"{(f'{sp:.2f}x' if sp else 'n/a'):>7} | "
            f"{row['filters_speedup_ge_1_5']:>3}/{row['n_filters_matched']:<2} | "
            f"{'PASS' if row['pass'] else 'FAIL'}",
            flush=True,
        )


def write_outputs(rows: list[dict], points: list[dict]) -> None:
    PLOT.mkdir(parents=True, exist_ok=True)
    summary_path = PLOT / "yfcc10m_v2_matched_iso_summary.csv"
    detail_path = PLOT / "yfcc10m_v2_matched_iso_per_filter.csv"
    with summary_path.open("w", newline="") as handle:
        fields = [
            "target_recall",
            "n_filters_total",
            "n_filters_matched",
            "stock_latency_geomean_ms",
            "sqlens_latency_geomean_ms",
            "stock_latency_mean_ms",
            "sqlens_latency_mean_ms",
            "stock_recall_mean",
            "sqlens_recall_mean",
            "speedup_geomean",
            "speedup_mean",
            "filters_sqlens_faster",
            "filters_speedup_ge_1_5",
            "gate_all_filters_matched",
            "gate_speedup_ge_1_5",
            "pass",
            "missing_stock",
            "missing_sqlens",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    detail_rows = []
    for row in rows:
        for item in row["per_filter"]:
            detail_rows.append({"target_recall": row["target_recall"], **item})
    if detail_rows:
        with detail_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)
    (PLOT / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "per_filter_min_latency_inside_target_band",
                "recall_band": {
                    "lower_margin": LOWER_RECALL_MARGIN,
                    "upper_margin": UPPER_RECALL_MARGIN,
                },
                "aggregate": "geomean_latency_and_geomean_speedup",
                "min_speedup_gate": MIN_SPEEDUP,
                "targets": TARGETS,
                "n_points": len(points),
                "all_targets_pass": all(r["pass"] for r in rows),
                "summary_csv": str(summary_path),
                "detail_csv": str(detail_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {detail_path}", flush=True)


def main() -> int:
    global CFG, OUT, PLOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args()
    CFG = args.config.resolve()
    OUT = args.out_dir.resolve()
    PLOT = args.plot_dir.resolve()
    if not CFG.is_file():
        raise SystemExit(f"missing {CFG}")

    stock_dir = OUT / "stock"
    sqlens_dir = OUT / "sqlens"
    # Do not reuse v2 artifacts: v3 has a different frozen filter set.
    search_dirs = [stock_dir, sqlens_dir]

    def need_sqlens_both_off(rows: list[dict]) -> bool:
        return any(r["missing_sqlens"] for r in rows)

    def need_stock_more(rows: list[dict]) -> bool:
        return any(r["missing_stock"] for r in rows)

    def maybe_finish(rows: list[dict]) -> bool:
        print_summary(rows)
        if all(r["pass"] for r in rows):
            print("[gate] all targets PASS early", flush=True)
            write_outputs(rows, load_per_filter_points(search_dirs))
            return True
        return False

    if not args.select_only:
        # 1) SQLens cap ladder.
        for cap in SQLENS_CAPS:
            if profile_exists(sqlens_dir, "sqlens_cap", 11, cap):
                continue
            run_frontier(family="sqlens_cap", ef=11, out_dir=sqlens_dir, caps=str(cap))

        # 2) Prefer Stock strict_order coverage first (blocks more targets at
        #    R>=0.85), then SQLens both_off for high-sel OR filters.
        for ef in STOCK_STRICT_EFS:
            if profile_exists(stock_dir, "stock_strict", ef, None):
                continue
            run_frontier(family="stock_strict", ef=ef, out_dir=stock_dir)
            rows = evaluate(load_per_filter_points(search_dirs))
            if maybe_finish(rows):
                return 0

        # 3) SQLens both_off fill for high-R OR / residual gaps.
        for sef in SQLENS_BOTH_OFF_EFS:
            rows = evaluate(load_per_filter_points(search_dirs))
            if not need_sqlens_both_off(rows):
                break
            if profile_exists(sqlens_dir, "both_off", sef, None):
                continue
            run_frontier(family="both_off", ef=sef, out_dir=sqlens_dir)
            rows = evaluate(load_per_filter_points(search_dirs))
            if maybe_finish(rows):
                return 0
            print_summary(rows)

        # 4) Stock both_off only if strict_order still cannot cover a target.
        for ef in STOCK_BOTH_OFF_EFS:
            points = load_per_filter_points(search_dirs)
            rows = evaluate(points)
            if not need_stock_more(rows):
                break
            if profile_exists(stock_dir, "both_off", ef, None):
                continue
            run_frontier(family="both_off", ef=ef, out_dir=stock_dir)

    points = load_per_filter_points(search_dirs)
    rows = evaluate(points)
    print_summary(rows)
    write_outputs(rows, points)
    return 0 if all(r["pass"] for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
