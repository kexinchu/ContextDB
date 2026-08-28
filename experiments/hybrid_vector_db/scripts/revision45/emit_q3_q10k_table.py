#!/usr/bin/env python3
"""Emit paper/tables/eval_acorn_q10k.tex from the new q10K score.

Never writes eval_acorn_matched.tex (the published q50 table).
"""
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score",
        type=Path,
        default=ROOT / "results/hybrid_vector_db/revision45/q3_acorn_q10k/score.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper/tables/eval_acorn_q10k.tex",
    )
    args = parser.parse_args()
    if args.out.resolve() == (ROOT / "paper/tables/eval_acorn_matched.tex").resolve():
        raise SystemExit("refusing to overwrite the published q50 ACORN table")
    score = json.loads(args.score.read_text(encoding="utf-8"))
    sel = {
        row["filter_name"]: row["actual_pct"]
        for row in csv.DictReader(FILTERS.open(encoding="utf-8"))
    }
    cells = {cell["filter_name"]: cell for cell in score.get("cells") or []}
    ratios: list[float] = []
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Added q10K screen: stock, VisGuide, and in-engine",
        r"\texttt{acorn1} after independent Recall@10 LCB95 $\ge 0.90$",
        r"calibration on q20--q99 and measurement on q200--q10199",
        r"($k{=}10$, attributes SQL). This table does not replace",
        r"Table~\ref{tab:eval-matched-recall} or",
        r"Table~\ref{tab:eval-acorn-matched}.}",
        r"\label{tab:eval-acorn-q10k}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Predicate & Sel. & Stock & Guide & ACORN & ACORN/guide & ACORN R \\",
        r"\midrule",
    ]
    for name, pct in sel.items():
        cell = cells.get(name)
        if not cell:
            continue
        stock = cell.get("stock") or {}
        guide = cell.get("d1") or {}
        acorn = cell.get("acorn1") or {}
        if not (stock.get("mean_ms") and guide.get("mean_ms") and acorn.get("mean_ms")):
            continue
        ratio = float(acorn["mean_ms"]) / float(guide["mean_ms"])
        ratios.append(ratio)
        lines.append(
            f"{LABELS.get(name, name)} & {float(pct):.1f}\\% & "
            f"{float(stock['mean_ms']):.1f} & {float(guide['mean_ms']):.1f} & "
            f"{float(acorn['mean_ms']):.1f} & {ratio:.2f}$\\times$ & "
            f"{float(acorn.get('recall') or 0):.2f} \\\\"
        )
    if ratios:
        lines.append(r"\midrule")
        lines.append(
            f"geomean & & & & & {_geomean(ratios):.2f}$\\times$ & \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "n": len(ratios)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
