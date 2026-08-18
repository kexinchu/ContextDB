#!/usr/bin/env python3
"""YFCC v2 target-band iso-recall: land BOTH arms inside each recall band.

User requirement: for target T (e.g. 0.70), measured recall for Stock and
SQLens must both fall near T (example band: 0.69–0.75), not R=0.06 vs R=0.96
at the same ef.

Protocol:
  band(T) = [T - 0.01, min(1.0, T + 0.05)]
  Search Stock and SQLens configs independently until each target has an
  in-band point (prefer lowest e2e latency). Emit a pairs table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/kec23008/Hybrid-Retrieval")
CFG = ROOT / "experiments/hybrid_vector_db/configs/figure5_r41_yfcc_primary_v2filters.json"
OUT = ROOT / "results/hybrid_vector_db/figure5_r41_yfcc_v2_target_band"
PLOT = ROOT / "results/hybrid_vector_db/yfcc10m_v2_target_band_iso"
LOCK = "results/hybrid_vector_db/.figure5_yfcc_primary_db.lock"
CPU = "48-63"
TARGETS = [0.70, 0.80, 0.85, 0.90, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
MODE_STOCK = "original"
MODE_SQLENS = "design1_bloom_bfs_layout_d3"
PAT = re.compile(r"calibration_(.+)_ef(\d+)(?:_cap(\d+))?_profile_summary\.csv$")


def band(target: float) -> tuple[float, float]:
    return (target - 0.01, min(1.0, target + 0.05))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def aggregate_profile(path: Path) -> dict[str, dict[str, float]]:
    by: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            by[row["mode"]].append(
                (
                    float(row["recall_mean"]),
                    float(row["end_to_end_mean_ms"]),
                    float(row["query_latency_mean_ms"]),
                )
            )
    out: dict[str, dict[str, float]] = {}
    for mode, rows in by.items():
        out[mode] = {
            "recall": mean([r[0] for r in rows]),
            "e2e": mean([r[1] for r in rows]),
            "query": mean([r[2] for r in rows]),
            "n_filters": float(len(rows)),
        }
    return out


def collect_points(out_dirs: list[Path]) -> list[dict]:
    points: list[dict] = []
    for directory in out_dirs:
        if not directory.is_dir():
            continue
        for profile in directory.glob("*_profile_summary.csv"):
            match = PAT.search(profile.name)
            if match is None:
                continue
            family, ef, cap = match.group(1), int(match.group(2)), match.group(3)
            agg = aggregate_profile(profile)
            for mode, arm in ((MODE_STOCK, "stock"), (MODE_SQLENS, "sqlens")):
                if mode not in agg:
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
                        "sqlens_scan_cap": int(cap) if cap else "",
                        "recall": round(agg[mode]["recall"], 6),
                        "latency_e2e_ms": round(agg[mode]["e2e"], 3),
                        "latency_query_ms": round(agg[mode]["query"], 3),
                        "profile": str(profile.relative_to(ROOT)),
                    }
                )
    uniq: dict[tuple, dict] = {}
    for point in points:
        key = (
            point["arm"],
            point["family"],
            point["ef_search"],
            str(point["sqlens_scan_cap"]),
        )
        prev = uniq.get(key)
        if prev is None or point["latency_e2e_ms"] < prev["latency_e2e_ms"]:
            uniq[key] = point
    return list(uniq.values())


def in_band(recall: float, target: float) -> bool:
    lo, hi = band(target)
    return lo - 1e-12 <= recall <= hi + 1e-12


def pick_in_band(points: list[dict], arm: str, target: float) -> dict | None:
    elig = [p for p in points if p["arm"] == arm and in_band(p["recall"], target)]
    if not elig:
        return None
    return min(elig, key=lambda p: (p["latency_e2e_ms"], abs(p["recall"] - target)))


def run_frontier(
    *,
    family: str,
    ef: int,
    out_dir: Path,
    caps: str | None = None,
    allow_expensive: bool = False,
) -> None:
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
    if allow_expensive or ef > 1000 or (caps is not None):
        # high-ef / nonstandard efs / caps need the expensive allow flag for SQLens paths
        if family in {"both_off", "sqlens_cap"} and (allow_expensive or ef not in {
            20, 40, 60, 80, 100, 150, 200, 250, 500, 750, 1000
        } or caps is not None):
            if "--allow-expensive-sqlens-calibration" not in cmd:
                cmd.append("--allow-expensive-sqlens-calibration")
    # Always allow for sqlens_cap and dense-like efs
    if family == "sqlens_cap" and "--allow-expensive-sqlens-calibration" not in cmd:
        cmd.append("--allow-expensive-sqlens-calibration")
    if family == "both_off" and ef not in {
        20, 40, 60, 80, 100, 150, 200, 250, 500, 750, 1000
    }:
        if "--allow-expensive-sqlens-calibration" not in cmd:
            cmd.append("--allow-expensive-sqlens-calibration")

    print(f"[run] family={family} ef={ef} caps={caps} out={out_dir.name}", flush=True)
    started = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"[done] family={family} ef={ef} caps={caps} in {time.time()-started:.0f}s", flush=True)


def coverage(points: list[dict]) -> dict[str, dict[float, bool]]:
    out = {"stock": {}, "sqlens": {}}
    for target in TARGETS:
        out["stock"][target] = pick_in_band(points, "stock", target) is not None
        out["sqlens"][target] = pick_in_band(points, "sqlens", target) is not None
    return out


def print_coverage(points: list[dict]) -> None:
    cov = coverage(points)
    print("\nBand coverage (need True/True for each target):", flush=True)
    for target in TARGETS:
        lo, hi = band(target)
        s = pick_in_band(points, "stock", target)
        q = pick_in_band(points, "sqlens", target)
        def fmt(p):
            if p is None:
                return "MISSING"
            cap = f"+cap{p['sqlens_scan_cap']}" if p["sqlens_scan_cap"] != "" else ""
            return (
                f"{p['family']}/ef{p['ef_search']}{cap} "
                f"R={p['recall']:.4f} e2e={p['latency_e2e_ms']:.1f}"
            )
        print(
            f"  T={target:.2f} band=[{lo:.2f},{hi:.2f}] "
            f"Stock[{fmt(s)}] SQLens[{fmt(q)}] "
            f"ok={cov['stock'][target] and cov['sqlens'][target]}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    if not CFG.is_file():
        raise SystemExit(f"missing config {CFG}")

    stock_dir = OUT / "stock"
    sqlens_dir = OUT / "sqlens"
    stock_dir.mkdir(parents=True, exist_ok=True)
    sqlens_dir.mkdir(parents=True, exist_ok=True)
    PLOT.mkdir(parents=True, exist_ok=True)

    # Stock candidates: both_off high-ef (Stock recall was ~0.06 at ef20 on v2)
    # plus stock_strict for high-R bands. Run Stock families that don't force
    # a useless SQLens twin when possible (stock_strict = Stock-only).
    stock_plan: list[tuple[str, int, str | None]] = [
        ("stock_strict", ef, None)
        for ef in (20, 40, 60, 80, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 3000, 5000)
    ] + [
        # both_off also measures SQLens at same ef — expensive; use sparse ladder
        # only to fill mid-R bands stock_strict cannot hit.
        ("both_off", ef, None)
        for ef in (500, 750, 1000, 1500, 2000, 3000, 5000, 10000)
    ]

    # SQLens candidates: sqlens_cap first (lands mid recall without R≈0.96 overshoot),
    # then very low both_off efs only if needed.
    sqlens_plan: list[tuple[str, int, str | None]] = [
        ("sqlens_cap", 11, str(cap))
        for cap in (500, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 100000)
    ] + [
        ("both_off", ef, None)
        for ef in (11, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 70, 90, 100, 150, 200, 250, 500)
    ]

    if not args.select_only:
        # Phase A: sqlens_cap ladder (SQLens-only cells) — highest priority for mid bands
        for family, ef, caps in sqlens_plan:
            points = collect_points([stock_dir, sqlens_dir, ROOT / "results/hybrid_vector_db/figure5_r41_yfcc_v2_iso_calib"])
            cov = coverage(points)
            if all(cov["sqlens"][t] for t in TARGETS) and family != "sqlens_cap":
                # still may need stock; skip further SQLens both_off if SQLens covered
                if family == "both_off":
                    continue
            if family == "sqlens_cap":
                assert caps is not None
                # skip if this cap already present
                if any(
                    p["arm"] == "sqlens"
                    and p["family"] == "sqlens_cap"
                    and p["sqlens_scan_cap"] == int(caps)
                    for p in points
                ):
                    continue
                run_frontier(family=family, ef=ef, out_dir=sqlens_dir, caps=caps)
            else:
                # both_off SQLens fill — only if some target still missing for SQLens
                if all(cov["sqlens"][t] for t in TARGETS):
                    continue
                if any(
                    p["arm"] == "sqlens" and p["family"] == "both_off" and p["ef_search"] == ef
                    for p in points
                ):
                    continue
                run_frontier(
                    family=family,
                    ef=ef,
                    out_dir=sqlens_dir,
                    allow_expensive=True,
                )
            points = collect_points([stock_dir, sqlens_dir, ROOT / "results/hybrid_vector_db/figure5_r41_yfcc_v2_iso_calib"])
            print_coverage(points)
            if all(coverage(points)["sqlens"][t] for t in TARGETS):
                print("[gate] all SQLens target bands covered", flush=True)
                break

        # Phase B: Stock ladder
        for family, ef, caps in stock_plan:
            points = collect_points([stock_dir, sqlens_dir, ROOT / "results/hybrid_vector_db/figure5_r41_yfcc_v2_iso_calib"])
            if all(coverage(points)["stock"][t] for t in TARGETS):
                print("[gate] all Stock target bands covered", flush=True)
                break
            if any(
                p["arm"] == "stock"
                and p["family"] == family
                and p["ef_search"] == ef
                for p in points
            ):
                continue
            # Skip both_off if stock_strict already covers mid/high and both_off would
            # drag expensive SQLens twin — only run both_off for uncovered low/mid targets.
            if family == "both_off":
                uncovered = [t for t in TARGETS if not coverage(points)["stock"][t]]
                if not uncovered:
                    continue
                # both_off needed primarily when stock_strict overshoots low targets
            run_frontier(
                family=family,
                ef=ef,
                out_dir=stock_dir,
                allow_expensive=(family == "both_off"),
            )
            points = collect_points([stock_dir, sqlens_dir, ROOT / "results/hybrid_vector_db/figure5_r41_yfcc_v2_iso_calib"])
            print_coverage(points)

    points = collect_points(
        [
            stock_dir,
            sqlens_dir,
            ROOT / "results/hybrid_vector_db/figure5_r41_yfcc_v2_iso_calib",
        ]
    )
    print_coverage(points)

    wide = []
    for target in TARGETS:
        lo, hi = band(target)
        s = pick_in_band(points, "stock", target)
        q = pick_in_band(points, "sqlens", target)
        # fallback: closest recall (marked) if band empty — should not happen if plan worked
        def closest(arm: str):
            arm_pts = [p for p in points if p["arm"] == arm]
            if not arm_pts:
                return None, "missing"
            best = min(arm_pts, key=lambda p: (abs(p["recall"] - target), p["latency_e2e_ms"]))
            return best, ("ok" if in_band(best["recall"], target) else "out_of_band")

        if s is None:
            s, sst = closest("stock")
        else:
            sst = "ok"
        if q is None:
            q, qst = closest("sqlens")
        else:
            qst = "ok"
        if s is None or q is None:
            raise SystemExit(f"no points at all for target {target}")
        wide.append(
            {
                "dataset": "yfcc10m_v2_target_band",
                "target_recall": target,
                "band_lo": lo,
                "band_hi": hi,
                "stock_status": sst,
                "stock_family": s["family"],
                "stock_ef_search": s["ef_search"],
                "stock_recall": s["recall"],
                "stock_latency_e2e_ms": s["latency_e2e_ms"],
                "stock_latency_query_ms": s["latency_query_ms"],
                "sqlens_status": qst,
                "sqlens_family": q["family"],
                "sqlens_ef_search": q["ef_search"],
                "sqlens_cap": q["sqlens_scan_cap"],
                "sqlens_recall": q["recall"],
                "sqlens_latency_e2e_ms": q["latency_e2e_ms"],
                "sqlens_latency_query_ms": q["latency_query_ms"],
                "e2e_speedup_vs_stock": round(
                    s["latency_e2e_ms"] / q["latency_e2e_ms"], 4
                )
                if q["latency_e2e_ms"]
                else None,
            }
        )

    pairs_path = PLOT / "yfcc10m_v2_target_band_pairs.csv"
    with pairs_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(wide[0].keys()))
        writer.writeheader()
        writer.writerows(wide)
    ops_path = PLOT / "yfcc10m_v2_target_band_operating_points.csv"
    with ops_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(points, key=lambda p: (p["arm"], p["family"], p["ef_search"], str(p["sqlens_scan_cap"]))))
    (PLOT / "manifest.json").write_text(
        json.dumps(
            {
                "targets": TARGETS,
                "band_rule": "[target-0.01, min(1, target+0.05)]",
                "pairs_csv": str(pairs_path),
                "n_operating_points": len(points),
                "all_in_band": all(
                    row["stock_status"] == "ok" and row["sqlens_status"] == "ok"
                    for row in wide
                ),
            },
            indent=2,
        )
        + "\n"
    )

    print("\n=== Target-band iso-recall pairs ===", flush=True)
    print(
        f"{'T':>5} {'band':>11} | {'Stock':>28} {'R':>7} {'e2e':>8} | "
        f"{'SQLens':>28} {'R':>7} {'e2e':>8} | spd",
        flush=True,
    )
    for row in wide:
        sf = f"{row['stock_family']}/ef{row['stock_ef_search']}"
        qf = f"{row['sqlens_family']}/ef{row['sqlens_ef_search']}"
        if row["sqlens_cap"] != "":
            qf += f"+cap{row['sqlens_cap']}"
        mark_s = "" if row["stock_status"] == "ok" else "!"
        mark_q = "" if row["sqlens_status"] == "ok" else "!"
        print(
            f"{row['target_recall']:5.2f} [{row['band_lo']:.2f},{row['band_hi']:.2f}] | "
            f"{sf:>28}{mark_s} {row['stock_recall']:7.4f} {row['stock_latency_e2e_ms']:8.1f} | "
            f"{qf:>28}{mark_q} {row['sqlens_recall']:7.4f} {row['sqlens_latency_e2e_ms']:8.1f} | "
            f"{row['e2e_speedup_vs_stock']}",
            flush=True,
        )
    print(f"wrote {pairs_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
