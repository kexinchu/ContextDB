#!/usr/bin/env python3
"""Run a bounded q200 pilot for SQLens traversal scan-cap calibration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import run_figure5_frontier as frontier
except ImportError:
    import run_figure5_frontier as frontier


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = (
    ROOT / "results/hybrid_vector_db/figure5_r36_scan_cap_pilot"
)
DEFAULT_CAPS = (500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000)
SQLENS_MODE = "design1_bloom_bfs_layout_d3"
FIXED_EVENT_FIELDS = (
    "d3_adaptive_page_builds_delta",
    "d3_adaptive_bloom_builds_delta",
    "d3_adaptive_exact_builds_delta",
    "d3_adaptive_refinements_delta",
    "d3_adaptive_rejections_delta",
    "d3_fragment_builds_delta",
)


class ScanCapPilotError(RuntimeError):
    """The traversal-cap pilot cannot satisfy its bounded protocol."""


def parse_caps(value: str) -> list[int]:
    try:
        caps = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "scan caps must be comma-separated integers"
        ) from exc
    if not caps or any(cap <= 0 for cap in caps):
        raise argparse.ArgumentTypeError("scan caps must be positive")
    return list(dict.fromkeys(caps))


def replace_argument(command: list[str], option: str, value: str) -> None:
    try:
        position = command.index(option)
    except ValueError as exc:
        raise ScanCapPilotError(
            f"generated command lacks {option}"
        ) from exc
    command[position + 1] = value


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        return int(float(row.get(field) or 0))
    except ValueError:
        return 0


def projected_q10k_latency(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    """Project only D3 one-time cost; final claims still use real q10k E2E."""
    if not rows:
        raise ScanCapPilotError("cannot summarize an empty SQLens cell")
    activations = [float(row["activation_ms"]) for row in rows]
    query_latencies = [float(row["query_latency_ms"]) for row in rows]
    recurring = [
        float(row["activation_ms"])
        for row in rows
        if not any(_integer(row, field) > 0 for field in FIXED_EVENT_FIELDS)
    ]
    recurring_activation = (
        statistics.median(recurring)
        if recurring
        else statistics.median(activations)
    )
    fixed_excess = sum(
        max(float(row["activation_ms"]) - recurring_activation, 0.0)
        for row in rows
        if any(_integer(row, field) > 0 for field in FIXED_EVENT_FIELDS)
    )
    return {
        "recurring_activation_ms": recurring_activation,
        "fixed_activation_excess_ms": fixed_excess,
        "projected_q10k_e2e_ms": (
            statistics.fmean(query_latencies)
            + recurring_activation
            + fixed_excess / 10_000.0
        ),
    }


def summarize(raw: Path, cap: int) -> dict[str, object]:
    with raw.open(newline="", encoding="utf-8") as source:
        rows = [
            row
            for row in csv.DictReader(source)
            if row.get("mode") == SQLENS_MODE
        ]
    if len(rows) != 200:
        raise ScanCapPilotError(
            f"{raw} has {len(rows)} SQLens rows, expected 200"
        )
    projection = projected_q10k_latency(rows)
    return {
        "sqlens_max_scan_tuples": cap,
        "requests": len(rows),
        "recall_mean": statistics.fmean(float(row["recall"]) for row in rows),
        "query_latency_mean_ms": statistics.fmean(
            float(row["query_latency_ms"]) for row in rows
        ),
        "activation_mean_ms": statistics.fmean(
            float(row["activation_ms"]) for row in rows
        ),
        "end_to_end_mean_ms": statistics.fmean(
            float(row["end_to_end_ms"]) for row in rows
        ),
        "visited_tuples_mean": statistics.fmean(
            float(row["visited_tuples"]) for row in rows
        ),
        "max_scan_reached_rate": statistics.fmean(
            str(row["traversal_max_scan_reached"]).lower() == "true"
            for row in rows
        ),
        **projection,
        "raw": str(raw),
        "raw_sha256": frontier.sha256_file(raw),
    }


def write_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = frontier.load_config(config_path)
    if args.dataset not in config["datasets"]:
        raise ScanCapPilotError(f"unknown dataset: {args.dataset}")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "scan_cap_pilot_manifest.json"
    schedule: list[dict[str, Any]] = []
    for cap in args.scan_caps:
        raw = out_dir / (
            f"{args.dataset}_ef{args.ef_search}_sqlenscap{cap}.csv"
        )
        command, provenance = frontier.build_cell_command(
            config,
            args.dataset,
            "calibration",
            "both_off",
            args.ef_search,
            raw,
            args.backend_cpu_list,
        )
        mode_configs = dict(provenance["mode_configs"])
        sqlens_config = dict(mode_configs[SQLENS_MODE])
        sqlens_config["max_scan_tuples"] = cap
        mode_configs[SQLENS_MODE] = sqlens_config
        namespace = (
            f"fig5-r36-cap-pilot-{args.dataset}-ef{args.ef_search}-cap{cap}"
        )
        replace_argument(
            command,
            "--mode-configs-json",
            json.dumps(mode_configs, separators=(",", ":"), sort_keys=True),
        )
        replace_argument(
            command, "--d3-fragment-store-namespace", namespace
        )
        replace_argument(
            command, "--orchestrator-source", str(Path(__file__).resolve())
        )
        provenance["mode_configs"] = mode_configs
        provenance["d3_fragment_store_namespace"] = namespace
        provenance["execution_sources"]["orchestrator"] = {
            "path": str(Path(__file__).resolve()),
            "sha256": frontier.sha256_file(Path(__file__).resolve()),
        }
        plan = raw.with_suffix(raw.suffix + ".plan.json")
        schedule.append(
            {
                "cap": cap,
                "raw": str(raw),
                "plan": str(plan),
                "command": command,
                "provenance": provenance,
                "status": (
                    "complete"
                    if frontier.cell_complete(raw, plan, 400, provenance)
                    else "pending"
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "sqlens_figure5_scan_cap_pilot",
        "status": "planned",
        "paper_eligible": False,
        "purpose": (
            "calibration-grid design only; projected latency is never a paper result"
        ),
        "config": {
            "path": str(config_path),
            "sha256": frontier.sha256_file(config_path),
        },
        "release_contract": {
            "path": config["release_contract_path"],
            "sha256": config["release_contract_sha256"],
            **config["release_identity"],
        },
        "dataset": args.dataset,
        "ef_search": args.ef_search,
        "scan_caps": args.scan_caps,
        "schedule": schedule,
        "cells_total": len(schedule),
        "cells_complete": sum(cell["status"] == "complete" for cell in schedule),
    }
    frontier.atomic_json(manifest_path, manifest)
    if not args.execute:
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "cells_total": len(schedule),
                    "cells_complete": manifest["cells_complete"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    manifest["status"] = "running"
    frontier.atomic_json(manifest_path, manifest)
    for cell in schedule:
        raw = Path(cell["raw"])
        plan = Path(cell["plan"])
        provenance = cell["provenance"]
        if cell["status"] == "complete" and args.resume:
            continue
        if (raw.exists() or plan.exists()) and not args.overwrite:
            raise ScanCapPilotError(
                f"incomplete pilot output exists: {raw}"
            )
        for path in (raw, plan):
            if args.overwrite and path.exists():
                path.unlink()
        frontier.clear_fragment_store_namespace(
            str(provenance["d3_fragment_store_table"]),
            str(provenance["d3_fragment_store_namespace"]),
        )
        log = raw.with_suffix(raw.suffix + ".log")
        with log.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                cell["command"],
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        cell["returncode"] = completed.returncode
        if completed.returncode != 0 or not frontier.cell_complete(
            raw, plan, 400, provenance
        ):
            cell["status"] = "failed"
            manifest["status"] = "failed"
            frontier.atomic_json(manifest_path, manifest)
            raise ScanCapPilotError(
                f"scan-cap cell failed cap={cell['cap']}; see {log}"
            )
        cell["status"] = "complete"
        cell["summary"] = summarize(raw, int(cell["cap"]))
        manifest["cells_complete"] = sum(
            item["status"] == "complete" for item in schedule
        )
        frontier.atomic_json(manifest_path, manifest)

    summaries = [
        cell.get("summary") or summarize(Path(cell["raw"]), int(cell["cap"]))
        for cell in schedule
    ]
    summary_path = out_dir / "scan_cap_pilot_summary.csv"
    write_summary(summary_path, summaries)
    manifest["status"] = "complete"
    manifest["summary"] = {
        "path": str(summary_path),
        "sha256": frontier.sha256_file(summary_path),
        "rows": len(summaries),
    }
    manifest["paper_eligible"] = False
    frontier.atomic_json(manifest_path, manifest)
    print(f"wrote {summary_path}", flush=True)
    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=frontier.DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset",
        choices=("amazon", "yfcc", "laion"),
        default="amazon",
    )
    parser.add_argument("--ef-search", type=int, default=11)
    parser.add_argument(
        "--scan-caps",
        type=parse_caps,
        default=list(DEFAULT_CAPS),
    )
    parser.add_argument("--backend-cpu-list", default="48-63")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(create_parser().parse_args(argv))
    except (
        ScanCapPilotError,
        frontier.Figure5ContractError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
