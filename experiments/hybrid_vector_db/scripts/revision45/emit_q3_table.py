#!/usr/bin/env python3
"""Emit a NEW matched-recall ACORN table. Does not edit 2.63× / 171→304 cells."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
LABELS = {
    "popular_ge1000": r"popular$\ge$1000",
    "popular_ge1340": r"popular$\ge$1340",
    "popular_ge1780": r"popular$\ge$1780",
    "popular_ge2428": r"popular$\ge$2428",
    "popular_ge3284": r"popular$\ge$3284",
    "popular_ge4559": r"popular$\ge$4559",
    "price_10_to_20": r"price 10--20",
    "popular_ge10066": r"popular$\ge$10066",
    "rating5_price_le10": r"rating5 $\land$ price$\le$10",
    "long_review_ge500": r"len$\ge$500",
    "grocery_rating5": r"Grocery $\land$ rating=5",
    "grocery_helpful": r"Grocery $\land$ helpful$\ge$1",
    "helpful_ge20": r"helpful$\ge$20",
    "grocery_long500": r"Grocery $\land$ len$\ge$500",
}


def _geomean(values: list[float]) -> float:
    logs = [math.log(v) for v in values if v > 0]
    return math.exp(sum(logs) / len(logs)) if logs else 0.0


def _cell(cell: dict, mode: str) -> tuple[float | None, float | None, int | None, bool]:
    payload = cell.get(mode) or {}
    if not payload or payload.get("mean_ms") is None:
        return None, None, None, False
    return (
        float(payload["mean_ms"]),
        float(payload["recall"]),
        int(payload["ef_search"]),
        bool(payload.get("met_target", False)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper/tables/eval_acorn_matched.tex",
    )
    args = parser.parse_args()
    score = json.loads(args.score.read_text(encoding="utf-8"))
    sel = {
        row["filter_name"]: row["actual_pct"]
        for row in csv.DictReader(FILTERS.open(encoding="utf-8"))
    }
    cells = {cell["filter_name"]: cell for cell in score["cells"]}
    order = [name for name in sel if name in cells]
    rows: list[tuple] = []
    for name in order:
        stock_ms, stock_r, stock_ef, stock_ok = _cell(cells[name], "stock")
        guide_ms, _guide_r, guide_ef, guide_ok = _cell(cells[name], "d1")
        acorn_ms, acorn_r, acorn_ef, acorn_ok = _cell(cells[name], "acorn1")
        if stock_ms is None or guide_ms is None or acorn_ms is None:
            continue
        rows.append(
            (
                name,
                float(sel.get(name, 0)),
                stock_ms,
                guide_ms,
                acorn_ms,
                stock_r,
                acorn_r,
                stock_ef,
                guide_ef,
                acorn_ef,
                stock_ok and guide_ok and acorn_ok,
            )
        )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stock, VisGuide, and in-engine \texttt{acorn1} after",
        r"independent Recall@10 LCB95 $\ge 0.90$ calibration",
        r"($k{=}10$, attributes SQL, q50). Latencies are mean",
        r"milliseconds. The 10{,}000-request headline is stock versus",
        r"full \system.}",
        r"\label{tab:eval-acorn-matched}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Predicate & Sel. & Stock & Guide & ACORN & ACORN/guide & ACORN R \\",
        r"\midrule",
    ]
    ratios: list[float] = []
    for name, pct, stock_ms, guide_ms, acorn_ms, _stock_r, acorn_r, *_efs, ok in rows:
        ratio = acorn_ms / guide_ms if guide_ms else 0.0
        ratios.append(ratio)
        mark = r"$^\dagger$" if not ok else ""
        lines.append(
            f"{LABELS.get(name, name.replace('_', r'_'))} & "
            f"{pct:.1f}\\% & "
            f"{stock_ms:.1f} & {guide_ms:.1f} & {acorn_ms:.1f}{mark} & "
            f"{ratio:.2f}$\\times$ & {acorn_r:.2f} \\\\"
        )
    if rows:
        g_stock = _geomean([row[2] for row in rows])
        g_guide = _geomean([row[3] for row in rows])
        g_acorn = _geomean([row[4] for row in rows])
        lines.append(r"\midrule")
        lines.append(
            f"geomean & & {g_stock:.1f} & {g_guide:.1f} & {g_acorn:.1f} & "
            f"{_geomean(ratios):.2f}$\\times$ & \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
