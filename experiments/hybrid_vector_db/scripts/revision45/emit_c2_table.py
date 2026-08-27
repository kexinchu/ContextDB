#!/usr/bin/env python3
"""Emit a NEW fail-open write-sweep table from cell summaries."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LABELS = {
    "popular_ge1000": r"popular$\ge$1000",
    "long_review_ge500": r"len$\ge$500",
    "helpful_ge20": r"helpful$\ge$20",
}


def _read_cells(cells_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(cells_dir.glob("cell_*.summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(payload)
        elif isinstance(payload, dict):
            rows.append(payload)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper/tables/eval_failopen_write_sweep.tex",
    )
    args = parser.parse_args()
    grouped: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for row in _read_cells(args.cells):
        if row.get("kind") != "read":
            continue
        grouped[
            (
                str(row.get("filter_name")),
                float(row.get("update_rate_tps") or 0.0),
                str(row.get("method")),
            )
        ].append(row)
    keys = sorted(grouped, key=lambda item: (item[0], item[1], item[2]))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fail-open VisGuide on the source index, 16 readers.",
        r"Delivery is committed writer TPS over the requested rate;",
        r"supported only at $\ge$90\%. Panel~B 246/180 QPS is unchanged.}",
        r"\label{tab:eval-failopen-write-sweep}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Predicate & Rate & Stock QPS & Guide QPS & Delivery \\",
        r"\midrule",
    ]
    stock_by = {
        (filt, rate): statistics.fmean(float(r["qps"]) for r in rows)
        for (filt, rate, method), rows in grouped.items()
        if method == "stock" and rows
    }
    seen_pair: set[tuple[str, float]] = set()
    for filt, rate, method in keys:
        if method != "sqlens_full" or (filt, rate) in seen_pair:
            continue
        seen_pair.add((filt, rate))
        guide_rows = grouped[(filt, rate, "sqlens_full")]
        guide = statistics.fmean(float(r["qps"]) for r in guide_rows)
        stock = stock_by.get((filt, rate))
        deliveries = [
            float(r["update_delivery_ratio"])
            for r in guide_rows
            if r.get("update_delivery_ratio") is not None
        ]
        if rate == 0.0:
            deliv = "n/a"
        elif deliveries:
            mean_d = statistics.fmean(deliveries)
            deliv = f"{100.0 * mean_d:.0f}\\%" if mean_d >= 0.90 else "overload"
        else:
            deliv = "overload"
        stock_s = f"{stock:.0f}" if stock is not None else "---"
        lines.append(
            f"{LABELS.get(filt, filt.replace('_', r'_'))} & {rate:.0f} & "
            f"{stock_s} & {guide:.0f} & {deliv} \\\\"
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
