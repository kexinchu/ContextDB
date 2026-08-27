#!/usr/bin/env python3
"""Emit a NEW ACORN comparison table. Does not edit matched-recall cells."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
LABELS = {
    "grocery_long500": r"Grocery $\land$ len$\ge$500",
    "helpful_ge20": r"helpful$\ge$20",
    "grocery_helpful": r"Grocery $\land$ helpful$\ge$1",
    "long_review_ge500": r"len$\ge$500",
    "popular_ge1000": r"popular$\ge$1000",
}
ORDER = (
    "long_review_ge500",
    "grocery_helpful",
    "helpful_ge20",
    "grocery_long500",
)


def _summary_from_profile(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    out: dict[tuple[str, str], tuple[float, float]] = {}
    if not path.exists():
        return out
    for row in csv.DictReader(path.open(encoding="utf-8")):
        out[(row["filter_name"], row["mode"])] = (
            float(row["end_to_end_mean_ms"]),
            float(row["recall_mean"]),
        )
    return out


def _summary_from_failed(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    by: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row.get("error"):
            continue
        by[(row["filter_name"], row["mode"])].append(
            (float(row["end_to_end_ms"]), float(row["recall"]))
        )
    return {
        key: (statistics.fmean(ms for ms, _ in items), statistics.fmean(rec for _, rec in items))
        for key, items in by.items()
        if items
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-guided", type=Path, required=True)
    parser.add_argument("--acorn1", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper/tables/eval_acorn_amazon4.tex",
    )
    args = parser.parse_args()
    safe_csv = args.safe_guided.with_name(args.safe_guided.stem + "_profile_summary.csv")
    safe = _summary_from_profile(safe_csv)
    acorn_failed = args.acorn1.with_suffix(args.acorn1.suffix + ".failed.csv")
    acorn = _summary_from_failed(acorn_failed) if acorn_failed.exists() else {}
    if args.acorn1.exists() and args.acorn1.suffix == ".json" and not acorn:
        payload = json.loads(args.acorn1.read_text(encoding="utf-8"))
        acorn = _summary_from_profile(
            args.acorn1.with_name(args.acorn1.stem + "_profile_summary.csv")
        ) or acorn
        del payload
    sel = {
        row["filter_name"]: row["actual_pct"]
        for row in csv.DictReader(FILTERS.open(encoding="utf-8"))
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{In-engine \pgvector \texttt{acorn1} versus stock and",
        r"VisGuide at the same $\mathrm{ef}{=}100$ ($k{=}10$, q50).",
        r"ACORN preserves Recall@10; stock and VisGuide post-filter a short",
        r"list. This screen does not replace the matched-recall 2.63$\times$",
        r"geomean or the $171\to 304$ QPS frontier.}",
        r"\label{tab:eval-acorn-amazon4}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Predicate & Sel. & Stock & Guide & ACORN & Stock R & ACORN R \\",
        r"\midrule",
    ]
    for name in ORDER:
        stock = safe.get((name, "original"))
        guide = safe.get((name, "design1_bloom"))
        ac = acorn.get((name, "design1_bloom"))
        if not stock or not guide or not ac:
            continue
        lines.append(
            f"{LABELS.get(name, name.replace('_', r'_'))} & "
            f"{float(sel.get(name, 0)):.1f}\\% & "
            f"{stock[0]:.1f} & {guide[0]:.1f} & {ac[0]:.1f} & "
            f"{stock[1]:.2f} & {ac[1]:.2f} \\\\"
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
