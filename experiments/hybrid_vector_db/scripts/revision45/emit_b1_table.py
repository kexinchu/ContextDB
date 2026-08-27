#!/usr/bin/env python3
"""Emit a NEW paper table from a B1 score.json. Does not edit existing tables."""
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
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper/tables/eval_sql_first_14.tex",
    )
    args = parser.parse_args()
    score = json.loads(args.score.read_text(encoding="utf-8"))
    sel = {
        row["filter_name"]: row["actual_pct"]
        for row in csv.DictReader(FILTERS.open(encoding="utf-8"))
    }
    cells = {cell["filter_name"]: cell for cell in score["cells"]}
    order = list(sel)
    rows: list[tuple[str, float, float, float, float, float]] = []
    for name in order:
        cell = cells.get(name)
        if not cell or not cell.get("d1") or not cell.get("sql_first_forced_indexed_exact"):
            continue
        stock = cell["stock"]["mean_ms"]
        guide = cell["d1"]["mean_ms"]
        first = cell["sql_first_forced_indexed_exact"]["mean_ms"]
        ratio = first / guide if guide else 0.0
        rows.append((name, float(sel[name]), stock, guide, first, ratio))
    qn = score.get("queries", "")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{SQL-first versus VisGuide on Amazon row-local atoms",
        r"($k{=}10$, $\mathrm{ef}{=}100$, attributes SQL, q"
        + str(qn)
        + r").",
        r"All fourteen atoms; SQL-first is exact with registered scalar",
        r"indexes and no \hnsw. The four-atom q100 screen in",
        r"Table~\ref{tab:eval-sql-first-q100} is unchanged.}",
        r"\label{tab:eval-sql-first-14}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Predicate & Sel. & Stock & Guide & SQL-first & SQL/guide \\",
        r"\midrule",
    ]
    for name, pct, stock, guide, first, ratio in rows:
        lines.append(
            f"{LABELS.get(name, name.replace('_', r'_'))} & {pct:.1f}\\% & "
            f"{stock:.1f} & {guide:.1f} & {first:.1f} & {ratio:.2f}$\\times$ \\\\"
        )
    if rows:
        g_stock = _geomean([r[2] for r in rows])
        g_guide = _geomean([r[3] for r in rows])
        g_first = _geomean([r[4] for r in rows])
        g_ratio = _geomean([r[5] for r in rows])
        lines.append(r"\midrule")
        lines.append(
            f"geomean & & {g_stock:.1f} & {g_guide:.1f} & "
            f"{g_first:.1f} & {g_ratio:.2f}$\\times$ \\\\"
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
