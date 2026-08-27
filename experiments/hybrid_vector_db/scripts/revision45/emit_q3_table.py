#!/usr/bin/env python3
"""Emit a NEW matched-recall ACORN table. Does not edit 2.63× / 171→304 cells."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
LABELS = {
    "long_review_ge500": r"len$\ge$500",
    "grocery_helpful": r"Grocery $\land$ helpful$\ge$1",
    "helpful_ge20": r"helpful$\ge$20",
    "grocery_long500": r"Grocery $\land$ len$\ge$500",
}
ORDER = (
    "long_review_ge500",
    "grocery_helpful",
    "helpful_ge20",
    "grocery_long500",
)


def _cell(cell: dict, mode: str) -> tuple[str, str, str]:
    payload = cell.get(mode) or {}
    if not payload or payload.get("mean_ms") is None:
        return "---", "---", "---"
    ms = f"{float(payload['mean_ms']):.1f}"
    rec = f"{float(payload['recall']):.2f}"
    ef = str(int(payload["ef_search"]))
    if payload.get("met_target") is False:
        ms = ms + r"$^\dagger$"
    return ms, rec, ef


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
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{In-engine \pgvector \texttt{acorn1} versus stock and",
        r"VisGuide after independent Recall@10 LCB95 $\ge 0.90$ calibration",
        r"($k{=}10$, attributes SQL, q50).",
        r"Each arm uses the cheapest measured $ef$ that meets the gate.",
        r"This screen does not replace the 10{,}000-request 2.63$\times$",
        r"geomean or the $171\to 304$ QPS frontier.",
        r"The fixed-$ef{=}100$ diagnostic is Table~\ref{tab:eval-acorn-amazon4}.}",
        r"\label{tab:eval-acorn-matched}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
        r"\toprule",
        r"Predicate & Sel. & Stock $ef$ & Guide $ef$ & ACORN $ef$ & Stock & Guide & ACORN & Stock R & ACORN R \\",
        r"\midrule",
    ]
    for name in ORDER:
        cell = cells.get(name)
        if not cell:
            continue
        stock_ms, stock_r, stock_ef = _cell(cell, "stock")
        guide_ms, _guide_r, guide_ef = _cell(cell, "d1")
        acorn_ms, acorn_r, acorn_ef = _cell(cell, "acorn1")
        lines.append(
            f"{LABELS.get(name, name.replace('_', r'_'))} & "
            f"{float(sel.get(name, 0)):.1f}\\% & "
            f"{stock_ef} & {guide_ef} & {acorn_ef} & "
            f"{stock_ms} & {guide_ms} & {acorn_ms} & "
            f"{stock_r} & {acorn_r} \\\\"
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
