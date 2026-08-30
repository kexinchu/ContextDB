#!/usr/bin/env python3
"""Emit a NEW shared_buffers table. Does not write Table 5."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
LABELS = {
    "popular_ge1000": r"popular$\ge$1000",
    "long_review_ge500": r"len$\ge$500",
    "grocery_helpful": r"Grocery $\land$ helpful$\ge$1",
    "helpful_ge20": r"helpful$\ge$20",
    "grocery_long500": r"Grocery $\land$ len$\ge$500",
}
SIZES = ("128MB", "8GB", "64GB")


def _ms(cell: dict, mode: str) -> float | None:
    payload = (cell or {}).get(mode) or {}
    if payload.get("mean_ms") is None:
        return None
    return float(payload["mean_ms"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results/hybrid_vector_db/revision45/l1_shared_buffers",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper/tables/eval_shared_buffers.tex",
    )
    args = parser.parse_args()
    sel = {
        row["filter_name"]: row["actual_pct"]
        for row in csv.DictReader(FILTERS.open(encoding="utf-8"))
    }
    by_size: dict[str, dict[str, dict]] = {}
    for size in SIZES:
        path = args.root / size / "score.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_size[size] = {cell["filter_name"]: cell for cell in payload.get("cells") or []}
    if not by_size:
        raise SystemExit(f"no score.json under {args.root}")
    names = [name for name in sel if any(name in by_size[size] for size in by_size)]
    compact = [name for name in LABELS if name in names] or names
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Added shared\_buffers screen on Amazon attributes SQL",
        r"(q200--q1199, $k{=}10$). Stock / VisGuide mean milliseconds.",
        r"Does not replace Table~\ref{tab:eval-matched-recall}.}",
        r"\label{tab:eval-shared-buffers}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Predicate & 128MB S/G & 8GB S/G & 64GB S/G \\",
        r"\midrule",
    ]
    for name in compact:
        cells = []
        for size in SIZES:
            cell = by_size.get(size, {}).get(name, {})
            stock = _ms(cell, "stock")
            guide = _ms(cell, "d1")
            if stock is None or guide is None:
                cells.append("---")
            else:
                cells.append(f"{stock:.1f}/{guide:.1f}")
        lines.append(f"{LABELS.get(name, name)} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"wrote": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
