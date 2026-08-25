#!/usr/bin/env python3
"""After P1–P4 land: plot Figure 5 and emit paper-facing numbers.

Does not overwrite frozen q1K dirs. Does not edit TeX.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/hybrid_vector_db"
P1 = RESULTS / "figure5_hybrid_allowlist_q10k_formal"
P2 = RESULTS / "rowlocal_faiss14_q1k_screen"
P3 = RESULTS / "figure5_qps16_readonly"
PLOT = ROOT / "paper/scripts/plot_figure5_sql_native.py"
FIG = ROOT / "paper/figures"
PYTHON = sys.executable


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _p1_score(payload: dict) -> dict:
    block = payload.get("grocery_helpful", payload)
    return block["score"] if "score" in block else block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("p1", "all"), default="all")
    args = parser.parse_args()
    out: dict = {"stage": args.stage, "p1": None, "p2": None, "p3": None, "plots": []}

    p1_score_path = P1 / "score.json"
    if not p1_score_path.is_file():
        print(json.dumps({"error": "P1 score.json missing", "path": str(p1_score_path)}))
        return 2
    p1 = _load(p1_score_path)
    score = _p1_score(p1)
    cells = {cell["shape"]: cell for cell in score["cells"]}
    out["p1"] = {
        "paper_eligible": score.get("paper_eligible"),
        "queries": score.get("queries"),
        "all_sqlens_beat_stock": score.get("all_sqlens_beat_stock"),
        "cells": score["cells"],
        "amortize": score.get("amortize"),
        "fragment_memory": (
            _load(P1 / "fragment_memory.json")
            if (P1 / "fragment_memory.json").is_file()
            else None
        ),
        "geomean": math.exp(
            sum(math.log(float(cell["speedup_vs_stock"])) for cell in score["cells"])
            / len(score["cells"])
        )
        if all(cell.get("speedup_vs_stock") for cell in score["cells"])
        else None,
    }
    plotted = subprocess.run(
        [
            PYTHON,
            str(PLOT),
            "--score",
            str(p1_score_path),
            "--out-dir",
            str(FIG),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    out["plots"] = {
        "returncode": plotted.returncode,
        "stdout": plotted.stdout.strip(),
        "stderr": plotted.stderr.strip(),
    }
    if args.stage == "all" or (P2 / "score.json").is_file():
        if (P2 / "score.json").is_file():
            out["p2"] = _load(P2 / "score.json")
    if args.stage == "all" or (P3 / "score.json").is_file():
        if (P3 / "score.json").is_file():
            out["p3"] = _load(P3 / "score.json")

    numbers_path = RESULTS / "overnight_p1_p4_paper_numbers.json"
    numbers_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    return 0 if plotted.returncode == 0 else plotted.returncode


if __name__ == "__main__":
    raise SystemExit(main())
