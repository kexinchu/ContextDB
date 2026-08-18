#!/usr/bin/env python3
"""Run SQLens calibration with independently tuned traversal result targets."""

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
DEFAULT_SETTINGS = (
    (20, 11),
    (40, 11),
    (40, 20),
    (80, 11),
    (80, 20),
    (80, 40),
    (150, 11),
    (150, 20),
    (150, 40),
    (250, 11),
    (250, 20),
    (250, 40),
    (500, 11),
    (500, 20),
    (500, 40),
    (1000, 11),
    (1000, 20),
    (1000, 40),
)
SQLENS_MODE = "design1_bloom_bfs_layout_d3"


class SQLensTargetExtensionError(RuntimeError):
    pass


def parse_settings(value: str) -> list[tuple[int, int]]:
    settings: list[tuple[int, int]] = []
    try:
        for item in value.split(","):
            if not item.strip():
                continue
            ef_text, target_text = item.split(":", 1)
            setting = (int(ef_text), int(target_text))
            if setting not in settings:
                settings.append(setting)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "settings must be comma-separated ef:target pairs"
        ) from exc
    if not settings:
        raise argparse.ArgumentTypeError("at least one ef:target setting is required")
    for ef_search, target in settings:
        if ef_search <= 0 or target < 11 or target > ef_search:
            raise argparse.ArgumentTypeError(
                "each setting must satisfy ef > 0 and 11 <= target <= ef"
            )
    return settings


def replace_argument(command: list[str], option: str, value: str) -> None:
    try:
        position = command.index(option)
    except ValueError as exc:
        raise SQLensTargetExtensionError(
            f"generated command lacks {option}"
        ) from exc
    command[position + 1] = value


def rewrite_sqlens_target_cell(
    command: list[str],
    provenance: dict[str, Any],
    *,
    dataset: str,
    ef_search: int,
    target: int,
    orchestrator: Path,
    release_prefix: str,
) -> tuple[list[str], dict[str, Any]]:
    command = list(command)
    provenance = dict(provenance)
    mode_configs = {
        mode: dict(config)
        for mode, config in provenance["mode_configs"].items()
    }
    sqlens_config = dict(mode_configs[SQLENS_MODE])
    sqlens_config["guided_collect_target"] = ef_search
    sqlens_config["traversal_guided_target"] = target
    mode_configs[SQLENS_MODE] = sqlens_config
    namespace = (
        f"{release_prefix}-{dataset}-calibration-sqlens_target-"
        f"ef{ef_search}-target{target}"
    )
    replace_argument(
        command,
        "--mode-configs-json",
        json.dumps(mode_configs, separators=(",", ":"), sort_keys=True),
    )
    replace_argument(command, "--guided-collect-target", str(ef_search))
    replace_argument(command, "--traversal-guided-target", str(target))
    replace_argument(command, "--d3-fragment-store-namespace", namespace)
    replace_argument(command, "--orchestrator-source", str(orchestrator))
    provenance.update(
        {
            "scan_family": "sqlens_target",
            "sqlens_scan_cap": None,
            "sqlens_traversal_target": target,
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
    settings: Sequence[tuple[int, int]],
    backend_cpu_list: str,
    out_dir: Path,
) -> list[dict[str, Any]]:
    orchestrator = Path(__file__).resolve()
    release_prefix = frontier.release_namespace_prefix(
        config["release_identity"]
    )
    max_scan_tuples = int(config["search_grid"]["max_scan_tuples"])
    schedule: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset not in config["datasets"]:
            raise SQLensTargetExtensionError(f"unknown dataset: {dataset}")
        for ef_search, target in settings:
            raw = out_dir / (
                f"figure5_r35_{dataset}_calibration_sqlens_target_"
                f"ef{ef_search}_target{target}.csv"
            )
            command, provenance = frontier.build_cell_command(
                config,
                dataset,
                "calibration",
                frontier.SQLENS_CAP_FAMILY,
                ef_search,
                raw,
                backend_cpu_list,
                sqlens_scan_cap=max_scan_tuples,
            )
            command, provenance = rewrite_sqlens_target_cell(
                command,
                provenance,
                dataset=dataset,
                ef_search=ef_search,
                target=target,
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
    db_lock_path = frontier.global_db_lock_path(args)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / args.manifest_name
    lock = frontier.acquire_lock(manifest_path.with_suffix(".lock"))
    try:
        prior_manifest = (
            frontier.read_json(manifest_path)
            if args.require_global_db_lock and manifest_path.is_file()
            else None
        )
        schedule = build_schedule(
            config,
            args.datasets,
            args.settings,
            args.backend_cpu_list,
            out_dir,
        )
        for cell in schedule:
            prior_isolation = frontier.prior_completed_cell_isolation(
                prior_manifest,
                Path(cell["raw"]),
                db_lock_path,
            )
            if prior_isolation is not None:
                cell["database_isolation"] = prior_isolation
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "sqlens_figure5_sqlens_target_extension",
            "status": "planned",
            "paper_eligible": False,
            "purpose": (
                "SQLens calibration extension that decouples the HNSW beam "
                "budget from the number of predicate-valid traversal results"
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
            "settings": [
                {"ef_search": ef_search, "traversal_guided_target": target}
                for ef_search, target in args.settings
            ],
            "schedule": schedule,
            "cells_total": len(schedule),
            "cells_complete": sum(
                cell["status"] == "complete" for cell in schedule
            ),
            "requested_slice_complete": False,
        }
        if args.require_global_db_lock:
            manifest["database_isolation"] = (
                frontier.planned_global_db_isolation(db_lock_path)
            )
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

        global_lock: frontier.GlobalDBLock | None = None
        try:
            if args.require_global_db_lock:
                global_lock = frontier.acquire_global_db_lock(
                    db_lock_path,
                    "run_figure5_sqlens_target_extension",
                )
                manifest["database_isolation"] = global_lock.evidence(
                    held_through_completion=False
                )
                for cell in schedule:
                    if (
                        cell["status"] == "complete"
                        and args.resume
                        and not frontier.completed_isolation_evidence_valid(
                            cell.get("database_isolation"),
                            db_lock_path,
                        )
                    ):
                        raise SQLensTargetExtensionError(
                            "cannot resume a completed cell without valid global "
                            "DB isolation evidence; use --overwrite --no-resume"
                        )

            manifest["status"] = "running"
            frontier.atomic_json(manifest_path, manifest)
            for cell in schedule:
                if cell["status"] == "complete" and args.resume:
                    continue
                raw = Path(cell["raw"])
                plan = Path(cell["plan"])
                log = raw.with_suffix(raw.suffix + ".log")
                if (raw.exists() or plan.exists()) and not args.overwrite:
                    raise SQLensTargetExtensionError(
                        f"incomplete SQLens-target output exists: {raw}"
                    )
                if args.overwrite:
                    for path in (raw, plan, log):
                        if path.exists():
                            path.unlink()
                    reset_records = [
                        {
                            "namespace": namespace,
                            "rows_deleted": frontier.clear_fragment_store_namespace(
                                str(cell["d3_fragment_store_table"]),
                                namespace,
                            ),
                        }
                        for namespace in frontier.fragment_store_namespaces(cell)
                    ]
                    cell["d3_namespace_reset_evidence"] = reset_records
                    cell["d3_namespace_rows_deleted"] = sum(
                        int(record["rows_deleted"])
                        for record in reset_records
                    )
                cell["status"] = "running"
                cell["started_at"] = frontier.utc_now()
                manifest["updated_at"] = frontier.utc_now()
                frontier.atomic_json(manifest_path, manifest)
                log.parent.mkdir(parents=True, exist_ok=True)
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
                    raise SQLensTargetExtensionError(
                        f"SQLens-target cell failed: dataset={cell['dataset']}, "
                        f"ef={cell['ef_search']}, "
                        f"target={cell['sqlens_traversal_target']}; see {log}"
                    )
                cell["status"] = "complete"
                cell["raw_sha256"] = frontier.sha256_file(raw)
                if global_lock is not None:
                    cell["database_isolation"] = global_lock.evidence(
                        held_through_completion=True
                    )
                manifest["cells_complete"] = sum(
                    item["status"] == "complete" for item in schedule
                )
                manifest["updated_at"] = frontier.utc_now()
                frontier.atomic_json(manifest_path, manifest)

            manifest["status"] = "complete"
            manifest["completed_at"] = frontier.utc_now()
            manifest["requested_slice_complete"] = True
            if global_lock is not None:
                if not all(
                    frontier.completed_isolation_evidence_valid(
                        cell.get("database_isolation"),
                        db_lock_path,
                    )
                    for cell in schedule
                    if cell["status"] == "complete"
                ):
                    raise SQLensTargetExtensionError(
                        "completed cell is missing global DB isolation evidence"
                    )
                manifest["database_isolation"] = global_lock.evidence(
                    held_through_completion=True
                )
            frontier.atomic_json(manifest_path, manifest)
            print(f"wrote {manifest_path}", flush=True)
            return 0
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            manifest["updated_at"] = frontier.utc_now()
            if global_lock is not None:
                manifest["database_isolation"] = global_lock.evidence(
                    held_through_completion=True
                )
            frontier.atomic_json(manifest_path, manifest)
            raise
        finally:
            if global_lock is not None:
                global_lock.close()
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
    parser.add_argument(
        "--settings",
        type=parse_settings,
        default=list(DEFAULT_SETTINGS),
        help="Comma-separated ef_search:traversal_guided_target pairs.",
    )
    parser.add_argument("--backend-cpu-list", default="48-63")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--manifest-name",
        default="figure5_r36_sqlens_target_extension_manifest.json",
    )
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
    parser.add_argument(
        "--require-global-db-lock",
        action="store_true",
        help=(
            "Require the shared formal calibration DB lock before any "
            "namespace reset or database experiment."
        ),
    )
    parser.add_argument(
        "--global-db-lock-path",
        type=Path,
        help=(
            "Shared lock file used by all formal Figure 5 calibration "
            "orchestrators; requires --require-global-db-lock."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(create_parser().parse_args(argv))
    except (
        SQLensTargetExtensionError,
        frontier.Figure5ContractError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
