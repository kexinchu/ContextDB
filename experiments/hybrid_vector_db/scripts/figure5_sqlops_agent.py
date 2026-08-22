#!/usr/bin/env python3
"""Figure 5 JOIN-hybrid status, screening scores, and preview plots.

Screening/confirmation artifacts are never paper-eligible. Official figure
PDFs are written only from a later q10K/r3 artifact that passes the 0.90 LCB
gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/hybrid_vector_db"
LOG = RESULTS / "amazon10m_sql_native_p0_r43_q1k_r1_sqlops_join_screen.log"
CHECKPOINT = RESULTS / "amazon10m_sql_native_p0_r43_q1k_r1_sqlops_join_screen.checkpoint"
CONFIRM_CSV = RESULTS / "amazon10m_sql_native_p0_r43_q1k_r1_sqlops_join_screen.csv"
CONFIRM_MANIFEST = RESULTS / "amazon10m_sql_native_p0_r43_q1k_r1_sqlops_join_screen.manifest.json"
Q10K_LOG = RESULTS / "amazon10m_sql_native_p0_r43_q10k_r3_sqlops_join.log"
PREVIEW_DIR = RESULTS / "figure5_sqlops_join_preview"
TARGET_SPEEDUP = 1.5
SQLENS_MODE = "d1_d2_d3"
STOCK_MODE = "stock"
SQL_FIRST_MODE = "sql_first_forced_indexed_exact"
WORKLOADS = ("join_facts", "join_catalog", "join_acl")
FILTERS = ("grocery_helpful", "helpful_ge20", "grocery_long500")
PANEL = {
    "join_facts": "Facts JOIN",
    "join_catalog": "Catalog JOIN",
    "join_acl": "ACL JOIN",
}


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def live_status() -> dict[str, Any]:
    lines = _json_lines(LOG)
    calibrations = [row for row in lines if row.get("progress") == "calibration"]
    ticks = [row for row in lines if row.get("progress") == "measurement_tick"]
    last_tick = ticks[-1] if ticks else None
    log_tail = ""
    if LOG.is_file():
        text = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = text[-1] if text else ""
    failed = "Traceback" in log_tail or (
        log_tail.startswith("P0_EXIT:") and not log_tail.startswith("P0_EXIT:0")
    )
    if LOG.is_file() and "P0_EXIT:0" in LOG.read_text(encoding="utf-8", errors="replace"):
        phase = "screen_complete"
    elif last_tick and last_tick.get("group") == "sql_first":
        phase = "screen_sql_first"
    elif last_tick and last_tick.get("group") == "sequential":
        phase = "screen_measurement"
    elif calibrations:
        phase = "screen_calibration"
    else:
        phase = "starting_or_idle"
    return {
        "phase": phase,
        "failed": failed,
        "log_tail": log_tail,
        "calibration_blocks": len(calibrations),
        "calibration_errors": sum(int(row.get("errors") or 0) for row in calibrations),
        "min_calib_lcb": min((float(row["highest_target_lcb"]) for row in calibrations), default=None),
        "last_tick": last_tick,
        "confirm_csv": CONFIRM_CSV.is_file(),
        "q10k_log": Q10K_LOG.is_file(),
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def score_csv(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    recalls: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    errors = 0
    for row in rows:
        if str(row.get("phase", "")) not in {"measurement", "final", ""}:
            continue
        if row.get("error"):
            errors += 1
            continue
        key = (row["workload"], row["filter_name"], row["mode"])
        grouped[key].append(float(row["e2e_ms"]))
        if row.get("recall") not in {None, "", "NA"}:
            recalls[key].append(float(row["recall"]))
    cells = []
    for workload in WORKLOADS:
        for filt in FILTERS:
            stock = _mean(grouped[(workload, filt, STOCK_MODE)])
            sqlens = _mean(grouped[(workload, filt, SQLENS_MODE)])
            sql_first = _mean(grouped[(workload, filt, SQL_FIRST_MODE)])
            speedup = (stock / sqlens) if stock and sqlens else None
            cells.append(
                {
                    "workload": workload,
                    "panel": PANEL[workload],
                    "filter_name": filt,
                    "stock_ms": stock,
                    "sqlens_ms": sqlens,
                    "sql_first_ms": sql_first,
                    "speedup": speedup,
                    "stock_recall": _mean(recalls[(workload, filt, STOCK_MODE)]),
                    "sqlens_recall": _mean(recalls[(workload, filt, SQLENS_MODE)]),
                    "beats_stock": bool(speedup and speedup > 1.0),
                    "meets_1p5": bool(speedup and speedup >= TARGET_SPEEDUP),
                }
            )
    qualified = [cell for cell in cells if cell["speedup"] is not None]
    geo = (
        math.exp(statistics.fmean(math.log(cell["speedup"]) for cell in qualified))
        if qualified
        else None
    )
    return {
        "paper_eligible": False,
        "rows": len(rows),
        "errors": errors,
        "cells": cells,
        "wins": sum(1 for cell in cells if cell["beats_stock"]),
        "wins_1p5": sum(1 for cell in cells if cell["meets_1p5"]),
        "geo_speedup": geo,
        "all_beat_stock": all(cell["beats_stock"] for cell in cells) if cells else False,
        "all_meet_1p5": all(cell["meets_1p5"] for cell in cells) if cells else False,
    }


def plot_preview(score: dict[str, Any], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for workload, title in PANEL.items():
        subset = [cell for cell in score["cells"] if cell["workload"] == workload]
        labels = [cell["filter_name"] for cell in subset]
        stock = [cell["stock_ms"] or 0.0 for cell in subset]
        sqlens = [cell["sqlens_ms"] or 0.0 for cell in subset]
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        xpos = list(range(len(labels)))
        width = 0.36
        ax.bar([x - width / 2 for x in xpos], stock, width, label="stock pgvector", color="#4C78A8")
        ax.bar([x + width / 2 for x in xpos], sqlens, width, label="SQLens", color="#B279A2")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("mean e2e ms")
        ax.set_title(f"{title} (q1K screen, not paper)")
        ax.legend(frameon=False)
        fig.tight_layout()
        target = out_dir / f"preview_{workload}.pdf"
        fig.savefig(target)
        plt.close(fig)
        written.append(str(target))
    (out_dir / "score.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-csv", type=Path, default=CONFIRM_CSV)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    status = live_status()
    payload: dict[str, Any] = {"status": status}
    if args.score_csv.is_file():
        payload["score"] = score_csv(args.score_csv)
        if args.plot:
            payload["plots"] = plot_preview(payload["score"], PREVIEW_DIR)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
