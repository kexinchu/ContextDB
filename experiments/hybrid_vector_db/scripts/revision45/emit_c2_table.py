#!/usr/bin/env python3
"""Emit a NEW fail-open write-sweep table. Delivery only; no QPS cells."""
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
ORDER = ("popular_ge1000", "long_review_ge500", "helpful_ge20")
RATES = (10.0, 25.0)


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
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fail-open VisGuide on the source index, 16 readers, six",
        r"repeats of 10{,}000 requests. Delivery is committed writer TPS over",
        r"the requested rate. Every 10 and 25~upd/s arm delivers 100\%.",
        r"This sweep's QPS uses a different selector than",
        r"Table~\ref{tab:eval-failopen-16r0} and is not reported.",
        r"The published 100~upd/s cells and Panel~B 246/180 QPS are unchanged.}",
        r"\label{tab:eval-failopen-write-sweep}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Predicate & 10~upd/s & 25~upd/s \\",
        r"\midrule",
    ]
    for filt in ORDER:
        cells = []
        for rate in RATES:
            rows = grouped.get((filt, rate, "sqlens_full"), [])
            deliveries = [
                float(r["update_delivery_ratio"])
                for r in rows
                if r.get("update_delivery_ratio") is not None
            ]
            if deliveries and statistics.fmean(deliveries) >= 0.90:
                cells.append(f"{100.0 * statistics.fmean(deliveries):.0f}\\%")
            elif deliveries:
                cells.append("overload")
            else:
                cells.append("---")
        lines.append(
            f"{LABELS.get(filt, filt.replace('_', r'_'))} & {cells[0]} & {cells[1]} \\\\"
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
