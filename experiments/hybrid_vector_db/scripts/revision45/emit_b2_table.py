#!/usr/bin/env python3
"""Emit a NEW warm-fragment table. Does not edit empty-start Figure 5 cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SHAPE_ORDER = ("attributes", "join_facts", "join_catalog", "join_acl")
PANELS = {
    "attributes": "Attributes",
    "join_facts": "Facts JOIN",
    "join_catalog": "Catalog JOIN",
    "join_acl": "ACL JOIN",
}


def _cells(score: dict) -> dict[str, dict]:
    if "cells" in score:
        payload = score
    elif "grocery_helpful" in score:
        block = score["grocery_helpful"]
        payload = block["score"] if "score" in block else block
    else:
        payload = score.get("score", score)
    return {cell["shape"]: cell for cell in payload["cells"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--empty", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper/tables/eval_join_warm.tex",
    )
    args = parser.parse_args()
    warm = json.loads(args.warm.read_text(encoding="utf-8"))
    empty = json.loads(args.empty.read_text(encoding="utf-8"))
    wcells = _cells(warm)
    ecells = _cells(empty)
    qn = warm.get("queries") or warm.get("grocery_helpful", {}).get("score", {}).get(
        "queries", ""
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Guide/stock speedup on the same four SQL shapes with a",
        rf"resident \texttt{{grocery\_helpful}} fragment versus empty-start q{qn}.",
        r"Empty-start cells are the frozen q1K screen, not the 10{,}000-request",
        r"Figure~\ref{fig:eval-sql-native} annotations (1.44$\times$/1.17$\times$/0.92$\times$/1.00$\times$).}",
        r"\label{tab:eval-join-warm}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Shape & Empty-start & Resident fragment \\",
        r"\midrule",
    ]
    for name in SHAPE_ORDER:
        e = ecells[name]
        w = wcells[name]
        e_ratio = e["stock_ms"] / e["sqlens_ms"] if e["sqlens_ms"] else 0.0
        w_ratio = w["stock_ms"] / w["sqlens_ms"] if w["sqlens_ms"] else 0.0
        lines.append(
            f"{PANELS[name]} & {e_ratio:.2f}$\\times$ & {w_ratio:.2f}$\\times$ \\\\"
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
