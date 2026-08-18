#!/usr/bin/env python3
"""Audit Figure 6 iso-recall latency/throughput coverage for fixed targets.

Reads ``configs/figure6_iso_recall_targets.json`` and existing Amazon/YFCC/LAION
bundles, then writes a coverage matrix plus a fill queue of missing cells.

Examples::

    python3 experiments/hybrid_vector_db/scripts/audit_figure6_iso_recall_coverage.py
    python3 experiments/hybrid_vector_db/scripts/audit_figure6_iso_recall_coverage.py \\
      --out-dir results/hybrid_vector_db/figure6_iso_recall_fill
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT / "experiments/hybrid_vector_db/configs/figure6_iso_recall_targets.json"
)
DEFAULT_OUT = ROOT / "results/hybrid_vector_db/figure6_iso_recall_fill"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nearest(
    rows: Sequence[Mapping[str, str]],
    *,
    target: float,
    arm: str | None = None,
    target_key: str = "target_recall",
    arm_key: str = "arm",
) -> dict[str, str] | None:
    candidates = []
    for row in rows:
        try:
            row_target = float(row[target_key])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(row_target - target) > 1e-9:
            continue
        if arm is not None and str(row.get(arm_key) or "").lower() != arm:
            continue
        candidates.append(row)
    return candidates[0] if candidates else None


def amazon_latency_status(
    row: Mapping[str, str] | None,
    target: float,
    max_err: float,
) -> dict[str, Any]:
    if row is None:
        return {
            "present": False,
            "ok": False,
            "reason": "missing_row",
            "recall": "",
            "value": "",
            "family": "",
        }
    recall = float(row["recall"])
    err = abs(recall - target)
    ok = err <= max_err
    return {
        "present": True,
        "ok": ok,
        "reason": "ok" if ok else f"recall_err={err:.4f}>{max_err}",
        "recall": recall,
        "value": float(row["latency_mean_ms"]),
        "family": row.get("family", ""),
    }


def amazon_throughput_status(
    row: Mapping[str, str] | None,
    target: float,
    max_err: float,
) -> dict[str, Any]:
    if row is None:
        return {
            "present": False,
            "ok": False,
            "reason": "missing_row",
            "recall": "",
            "value": "",
            "family": "",
        }
    recall = float(row["recall"])
    err = abs(recall - target)
    ok = err <= max_err
    return {
        "present": True,
        "ok": ok,
        "reason": "ok" if ok else f"recall_err={err:.4f}>{max_err}",
        "recall": recall,
        "value": float(row["throughput_qps"]),
        "family": row.get("family", ""),
    }


def yfcc_latency_status(
    row: Mapping[str, str] | None,
    target: float,
    max_err: float,
) -> dict[str, Any]:
    """YFCC warm summary is aggregate (both arms in one row)."""
    if row is None:
        return {
            "present": False,
            "ok": False,
            "reason": "missing_row",
            "stock": None,
            "sqlens": None,
        }
    matched = int(float(row.get("n_filters_matched") or 0))
    total = int(float(row.get("n_filters_total") or 14))
    stock_r = float(row["stock_recall_mean"])
    sqlens_r = float(row["sqlens_recall_mean"])
    stock_ok = abs(stock_r - target) <= max_err
    sqlens_ok = abs(sqlens_r - target) <= max_err
    coverage_ok = matched == total
    return {
        "present": True,
        "ok": coverage_ok and stock_ok and sqlens_ok,
        "reason": (
            "ok"
            if coverage_ok and stock_ok and sqlens_ok
            else (
                f"matched={matched}/{total}; "
                f"stock_err={abs(stock_r - target):.4f}; "
                f"sqlens_err={abs(sqlens_r - target):.4f}"
            )
        ),
        "stock": {
            "ok": coverage_ok and stock_ok,
            "recall": stock_r,
            "value": float(row["stock_latency_mean_ms"]),
        },
        "sqlens": {
            "ok": coverage_ok and sqlens_ok,
            "recall": sqlens_r,
            "value": float(row["sqlens_latency_mean_ms"]),
        },
    }


def audit(config: Mapping[str, Any]) -> dict[str, Any]:
    targets = [float(x) for x in config["targets"]]
    max_err = float(config["gates"]["max_abs_recall_error"])
    matrix: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []

    # Amazon
    amazon = config["datasets"]["amazon"]
    amazon_lat = read_csv(ROOT / amazon["latency_bundle"])
    amazon_thr = read_csv(ROOT / amazon["throughput_bundle"])
    for target in targets:
        for arm in ("stock", "sqlens"):
            lat_row = nearest(amazon_lat, target=target, arm=arm)
            thr_row = nearest(amazon_thr, target=target, arm=arm)
            lat = amazon_latency_status(lat_row, target, max_err)
            thr = amazon_throughput_status(thr_row, target, max_err)
            for metric, status in (("latency", lat), ("throughput", thr)):
                cell = {
                    "dataset": "amazon",
                    "target_recall": target,
                    "arm": arm,
                    "metric": metric,
                    "present": status["present"],
                    "ok": status["ok"],
                    "reason": status["reason"],
                    "recall": status["recall"],
                    "value": status["value"],
                    "family": status["family"],
                }
                matrix.append(cell)
                if not status["ok"]:
                    queue.append(
                        {
                            **cell,
                            "priority": (
                                10
                                if not status["present"]
                                else 20
                            ),
                            "action": (
                                "remeasure_or_reselect"
                                if status["present"]
                                else "measure"
                            ),
                            "pg_port": amazon["pg_port"],
                        }
                    )

    # YFCC latency (aggregate rows); throughput may be absent
    yfcc = config["datasets"]["yfcc"]
    yfcc_lat_path = ROOT / yfcc["latency_bundle"]
    yfcc_lat = read_csv(yfcc_lat_path)
    # Use v3 warm only when the configured v4 bundle path does not exist yet.
    if not yfcc_lat_path.exists():
        interim = (
            ROOT
            / "results/hybrid_vector_db/yfcc10m_v3_matched_iso_warm/"
            "yfcc10m_matched_iso_warm_summary.csv"
        )
        yfcc_lat = read_csv(interim)
    yfcc_thr = read_csv(ROOT / yfcc["throughput_bundle"])
    for target in targets:
        lat_row = nearest(yfcc_lat, target=target, arm=None)
        lat = yfcc_latency_status(lat_row, target, max_err)
        for arm in ("stock", "sqlens"):
            arm_status = (lat.get(arm) if lat["present"] else None) or {
                "ok": False,
                "recall": "",
                "value": "",
            }
            cell = {
                "dataset": "yfcc",
                "target_recall": target,
                "arm": arm,
                "metric": "latency",
                "present": lat["present"],
                "ok": bool(arm_status.get("ok")),
                "reason": lat["reason"] if lat["present"] else "missing_row",
                "recall": arm_status.get("recall", ""),
                "value": arm_status.get("value", ""),
                "family": "",
            }
            matrix.append(cell)
            if not cell["ok"]:
                queue.append(
                    {
                        **cell,
                        "priority": 10 if not lat["present"] else 15,
                        "action": "run_or_finish_matched_iso_calibration",
                        "pg_port": yfcc["pg_port"],
                    }
                )
            thr_row = nearest(yfcc_thr, target=target, arm=arm)
            thr = amazon_throughput_status(thr_row, target, max_err)
            cell_t = {
                "dataset": "yfcc",
                "target_recall": target,
                "arm": arm,
                "metric": "throughput",
                "present": thr["present"],
                "ok": thr["ok"],
                "reason": thr["reason"],
                "recall": thr["recall"],
                "value": thr["value"],
                "family": thr["family"],
            }
            matrix.append(cell_t)
            if not thr["ok"]:
                queue.append(
                    {
                        **cell_t,
                        "priority": 30,
                        "action": "measure_c16_throughput_after_latency_select",
                        "pg_port": yfcc["pg_port"],
                    }
                )

    # LAION
    laion = config["datasets"]["laion"]
    laion_lat = read_csv(ROOT / laion["latency_bundle"])
    laion_thr = read_csv(ROOT / laion["throughput_bundle"])
    for target in targets:
        for arm in ("stock", "sqlens"):
            for metric, rows, status_fn in (
                ("latency", laion_lat, amazon_latency_status),
                ("throughput", laion_thr, amazon_throughput_status),
            ):
                row = nearest(rows, target=target, arm=arm)
                status = status_fn(row, target, max_err)
                cell = {
                    "dataset": "laion",
                    "target_recall": target,
                    "arm": arm,
                    "metric": metric,
                    "present": status["present"],
                    "ok": status["ok"],
                    "reason": status["reason"],
                    "recall": status["recall"],
                    "value": status["value"],
                    "family": status["family"],
                }
                matrix.append(cell)
                if not status["ok"]:
                    queue.append(
                        {
                            **cell,
                            "priority": 5,
                            "action": "build_laion_iso_recall_bundle",
                            "pg_port": laion["pg_port"],
                        }
                    )

    queue.sort(
        key=lambda row: (
            int(row["priority"]),
            row["dataset"],
            float(row["target_recall"]),
            row["metric"],
            row["arm"],
        )
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "targets": targets,
        "n_cells": len(matrix),
        "n_ok": sum(1 for row in matrix if row["ok"]),
        "n_missing_or_bad": sum(1 for row in matrix if not row["ok"]),
        "by_dataset": {},
    }
    for dataset in ("amazon", "yfcc", "laion"):
        cells = [row for row in matrix if row["dataset"] == dataset]
        summary["by_dataset"][dataset] = {
            "n_cells": len(cells),
            "n_ok": sum(1 for row in cells if row["ok"]),
            "n_missing_or_bad": sum(1 for row in cells if not row["ok"]),
        }
    return {"summary": summary, "matrix": matrix, "queue": queue}


def render_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Figure 6 iso-recall coverage",
        "",
        f"- created: `{summary['created_at']}`",
        f"- targets: {summary['targets']}",
        f"- cells ok: **{summary['n_ok']}/{summary['n_cells']}**",
        "",
        "| dataset | ok | missing/bad |",
        "|---|---:|---:|",
    ]
    for dataset, stats in summary["by_dataset"].items():
        lines.append(
            f"| {dataset} | {stats['n_ok']} | {stats['n_missing_or_bad']} |"
        )
    lines.extend(["", "## Fill queue (first 40)", ""])
    lines.append(
        "| pri | dataset | T | metric | arm | reason | action |"
    )
    lines.append("|---:|---|---:|---|---|---|---|")
    for row in payload["queue"][:40]:
        lines.append(
            f"| {row['priority']} | {row['dataset']} | {row['target_recall']} | "
            f"{row['metric']} | {row['arm']} | {row['reason']} | {row['action']} |"
        )
    if len(payload["queue"]) > 40:
        lines.append(f"\n… {len(payload['queue']) - 40} more rows in CSV.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    payload = audit(config)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage_summary.json").write_text(
        json.dumps(payload["summary"], indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(out / "coverage_matrix.csv", payload["matrix"])
    write_csv(out / "fill_queue.csv", payload["queue"])
    (out / "coverage.md").write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    print(f"wrote {out}")
    return 0 if payload["summary"]["n_missing_or_bad"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
