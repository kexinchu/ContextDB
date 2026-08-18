#!/usr/bin/env python3
"""Translate q5K per-filter latency settings into throughput-runner overrides."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

try:
    from .run_target090_q5k_per_filter import SQLENS_MODE, mode_config
except ImportError:
    from run_target090_q5k_per_filter import SQLENS_MODE, mode_config


class ConfigError(RuntimeError):
    pass


def read_dataset(path: Path, dataset: str) -> dict[str, dict[str, object]]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    value = root.get(dataset) if isinstance(root, dict) else None
    if not isinstance(value, dict):
        raise ConfigError(f"{path} does not define dataset {dataset!r}")
    if any(not isinstance(pair, dict) for pair in value.values()):
        raise ConfigError(f"{path} contains a non-object filter config")
    return {str(name): dict(pair) for name, pair in value.items()}


def build(
    base_path: Path,
    dataset: str,
    overlays: list[Path],
) -> dict[str, dict[str, dict[str, object]]]:
    pairs = read_dataset(base_path, dataset)
    for path in overlays:
        overlay = read_dataset(path, dataset)
        unknown = set(overlay) - set(pairs)
        if unknown:
            raise ConfigError(f"{path} adds unknown filters {sorted(unknown)}")
        pairs.update(overlay)
    result = {"original": {}, SQLENS_MODE: {}}
    for filter_name, pair in pairs.items():
        stock = pair.get("stock")
        sqlens = pair.get("sqlens")
        if not isinstance(stock, dict) or not isinstance(sqlens, dict):
            raise ConfigError(f"{filter_name} is missing stock/sqlens settings")
        result["original"][filter_name] = mode_config(stock, sqlens=False)
        result[SQLENS_MODE][filter_name] = mode_config(sqlens, sqlens=True)
    return result


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("amazon", "yfcc"), required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(
        args.out,
        build(args.base_config, args.dataset, list(args.overlay)),
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
