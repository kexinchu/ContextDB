#!/usr/bin/env python3
"""Run the Stock strict-order scan-cap extension for matched-recall tuning."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from . import run_figure5_frontier as frontier
except ImportError:
    import run_figure5_frontier as frontier


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = ROOT / "results/hybrid_vector_db/figure5_r36_formal"
DEFAULT_CAPS = (
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    2_000_000,
)
STOCK_MODE = "original"


class StockCapExtensionError(RuntimeError):
    pass


def parse_caps(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "scan caps must be comma-separated integers"
        ) from exc
    if not result or any(cap <= 0 for cap in result):
        raise argparse.ArgumentTypeError("scan caps must be positive")
    return list(dict.fromkeys(result))


def replace_argument(command: list[str], option: str, value: str) -> None:
    try:
        position = command.index(option)
    except ValueError as exc:
        raise StockCapExtensionError(
            f"generated command lacks {option}"
        ) from exc
    command[position + 1] = value


def rewrite_stock_cap_cell(
    command: list[str],
    provenance: dict[str, Any],
    *,
    dataset: str,
    ef_search: int,
    cap: int,
    orchestrator: Path,
    release_prefix: str,
) -> tuple[list[str], dict[str, Any]]:
    command = list(command)
    provenance = dict(provenance)
    mode_configs = {
        mode: dict(config)
        for mode, config in provenance["mode_configs"].items()
    }
    stock_config = dict(mode_configs[STOCK_MODE])
    stock_config["iterative_scan"] = "strict_order"
    stock_config["max_scan_tuples"] = cap
    mode_configs[STOCK_MODE] = stock_config
    namespace = (
        f"{release_prefix}-{dataset}-calibration-stock_cap-"
        f"ef{ef_search}-cap{cap}"
    )
    replace_argument(
        command,
        "--mode-configs-json",
        json.dumps(mode_configs, separators=(",", ":"), sort_keys=True),
    )
    replace_argument(command, "--max-scan-tuples", str(cap))
    replace_argument(command, "--iterative-scan", "strict_order")
    replace_argument(command, "--d3-fragment-store-namespace", namespace)
    replace_argument(command, "--orchestrator-source", str(orchestrator))
    provenance.update(
        {
            "scan_family": "stock_cap",
            "stock_scan_cap": cap,
            "mode_configs": mode_configs,
            "d3_fragment_store_namespace": namespace,
            "execution_sources": {
                **provenance["execution_sources"],
                "orchestrator": {
                    "path": str(orchestrator),
                    "sha256": frontier.sha256_file(orchestrator),
                },
            },
        }
    )
    return command, provenance


def build_schedule(
    config: dict[str, Any],
    datasets: Sequence[str],
    caps: Sequence[int],
    ef_search: int,
    backend_cpu_list: str,
    out_dir: Path,
) -> list[dict[str, Any]]:
    orchestrator = Path(__file__).resolve()
    release_prefix = frontier.release_namespace_prefix(
        config["release_identity"]
    )
    schedule: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset not in config["datasets"]:
            raise StockCapExtensionError(f"unknown dataset: {dataset}")
        for cap in caps:
            raw = out_dir / (
                f"figure5_r35_{dataset}_calibration_stock_cap_"
                f"ef{ef_search}_cap{cap}.csv"
            )
            command, provenance = frontier.build_cell_command(
                config,
                dataset,
                "calibration",
                "stock_strict",
                ef_search,
                raw,
                backend_cpu_list,
            )
            command, provenance = rewrite_stock_cap_cell(
                command,
                provenance,
                dataset=dataset,
                ef_search=ef_search,
                cap=cap,
                orchestrator=orchestrator,
                release_prefix=release_prefix,
            )
            plan = raw.with_suffix(raw.suffix + ".plan.json")
            schedule.append(
                {
                    **provenance,
                    "raw": str(raw),
                    "plan": str(plan),
                    "command": command,
                    "status": (
                        "complete"
                        if frontier.cell_complete(
                            raw,
                            plan,
                            int(provenance["expected_rows"]),
                            provenance,
                        )
                        else "pending"
                    ),
                }
            )
    return schedule


def run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = frontier.load_config(config_path)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "figure5_r36_stock_cap_extension_manifest.json"
    lock = frontier.acquire_lock(manifest_path.with_suffix(".lock"))
    try:
        schedule = build_schedule(
            config,
            args.datasets,
            args.scan_caps,
            args.ef_search,
            args.backend_cpu_list,
            out_dir,
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "sqlens_figure5_stock_scan_cap_extension",
            "status": "planned",
            "paper_eligible": False,
            "purpose": (
                "Stock baseline calibration extension; final latency comes "
                "from the disjoint q10k measurement trace"
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
            "datasets": list(args.datasets),
            "ef_search": args.ef_search,
            "scan_caps": list(args.scan_caps),
            "schedule": schedule,
            "cells_total": len(schedule),
            "cells_complete": sum(
                cell["status"] == "complete" for cell in schedule
            ),
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
            if cell["status"] == "complete" and args.resume:
                continue
            raw = Path(cell["raw"])
            plan = Path(cell["plan"])
            if (raw.exists() or plan.exists()) and not args.overwrite:
                raise StockCapExtensionError(
                    f"incomplete Stock-cap output exists: {raw}"
                )
            if args.overwrite:
                for path in (raw, plan):
                    if path.exists():
                        path.unlink()
            cell["status"] = "running"
            cell["started_at"] = frontier.utc_now()
            frontier.atomic_json(manifest_path, manifest)
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
            cell["completed_at"] = frontier.utc_now()
            if completed.returncode != 0 or not frontier.cell_complete(
                raw,
                plan,
                int(cell["expected_rows"]),
                cell,
            ):
                cell["status"] = "failed"
                cell["log"] = str(log)
                manifest["status"] = "failed"
                frontier.atomic_json(manifest_path, manifest)
                raise StockCapExtensionError(
                    f"Stock-cap cell failed: dataset={cell['dataset']}, "
                    f"cap={cell['stock_scan_cap']}; see {log}"
                )
            cell["status"] = "complete"
            cell["raw_sha256"] = frontier.sha256_file(raw)
            manifest["cells_complete"] = sum(
                item["status"] == "complete" for item in schedule
            )
            frontier.atomic_json(manifest_path, manifest)

        manifest["status"] = "complete"
        manifest["completed_at"] = frontier.utc_now()
        manifest["requested_slice_complete"] = True
        frontier.atomic_json(manifest_path, manifest)
        print(f"wrote {manifest_path}", flush=True)
        return 0
    finally:
        lock.close()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=frontier.DEFAULT_CONFIG)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("amazon", "yfcc", "laion"),
        default=["amazon", "yfcc", "laion"],
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
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(create_parser().parse_args(argv))
    except (
        StockCapExtensionError,
        frontier.Figure5ContractError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
