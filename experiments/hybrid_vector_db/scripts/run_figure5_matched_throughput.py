#!/usr/bin/env python3
"""Run formal r36 Figure 5 matched-recall service cells.

This wrapper is the serial database sidecar for the r36 calibration and
matched-latency pipeline. It consumes the selector CSV/plan/manifest audited
by ``run_figure5_matched_latency.py``, publishes one normalized immutable
measurement plan, and invokes ``pgvector_figure5_throughput.py`` for every
selected pair and client count in one preregistered protocol slice.

The core runner owns request execution, fresh D3 namespaces, and wall-clock
throughput measurement. This wrapper never derives QPS from latency. A cell
is complete only after all q10k request/repeat rows, output SHA bindings,
paired traces, release identity, and completed/barrier-wall formula have been
re-audited.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

try:
    from . import pgvector_figure5_throughput as throughput
    from . import run_figure5_matched_latency as matched
except ImportError:
    import pgvector_figure5_throughput as throughput
    import run_figure5_matched_latency as matched


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = matched.DEFAULT_CONFIG
DISTINCT_C16_PROTOCOL = "distinct-c16-q10k-r3"
FIXED_R090_PROTOCOL = "fixed-r090-service-q10k-r6"
FIXED_TARGETS_C16_PROTOCOL = "fixed-targets-c16-q10k-r3"
DEFAULT_PROTOCOL_SLICE = DISTINCT_C16_PROTOCOL
R36_FORMAL_DIR = ROOT / "results/hybrid_vector_db/figure5_r36_formal"
DEFAULT_SELECTION_CSV_BY_PROTOCOL = {
    DISTINCT_C16_PROTOCOL: R36_FORMAL_DIR / "figure5_r36_matched_configs.csv",
    FIXED_R090_PROTOCOL: (
        R36_FORMAL_DIR / "figure5_r36_fixed_target_configs.csv"
    ),
    FIXED_TARGETS_C16_PROTOCOL: (
        R36_FORMAL_DIR / "figure5_r36_fixed_target_configs.csv"
    ),
}
DEFAULT_OUT_DIR_BY_PROTOCOL = {
    DISTINCT_C16_PROTOCOL: (
        R36_FORMAL_DIR / "service_distinct_c16_q10k_r3"
    ),
    FIXED_R090_PROTOCOL: (
        R36_FORMAL_DIR / "service_fixed_r090_q10k_r6"
    ),
    FIXED_TARGETS_C16_PROTOCOL: (
        R36_FORMAL_DIR / "service_fixed_targets_c16_q10k_r3"
    ),
}
DEFAULT_WORKLOAD_MANIFESTS = {
    dataset: (
        ROOT
        / f"results/hybrid_vector_db/figure5_r35_{dataset}_manifest.json"
    )
    for dataset in ("amazon", "yfcc", "laion")
}
RUNNER_VERSION = "sqlens-figure5-matched-throughput-orchestrator-r36-v4"
EXPECTED_REQUESTS = 10_000
# Compatibility constants for direct helper callers. Formal runs take their
# repeat count and row counts from the selected ProtocolSlice.
EXPECTED_REPEATS = 6
EXPECTED_REQUEST_ROWS = EXPECTED_REQUESTS * EXPECTED_REPEATS * 2
EXPECTED_REPEAT_ROWS = EXPECTED_REPEATS * 2
SERVICE_CURVE_CLIENTS = (1, 4, 8, 16, 32, 64)
DEFAULT_CLIENTS = SERVICE_CURVE_CLIENTS
DEFAULT_CLIENT_CPU_LIST = "0-31"
DEFAULT_BACKEND_CPU_LIST = "32-63"
DEFAULT_TELEMETRY_DEVICES = "sda"
TRAVERSAL_GUIDED_BURST = 8
SERVICE_CURVE_MAX_CLIENTS = 64
CORE_FORMAL_MIN_REPEATS = throughput.MIN_REPEATS
FORMAL_FIXED_TARGETS = (0.90, 0.95, 0.99)
EXPECTED_FORMAL_PREDICATES = 14
MODES_BY_ARM = {
    "stock_pgvector": "original",
    "sqlens_full": "design1_bloom_bfs_layout_d3",
}
SERIAL_DB_REASON = (
    "Cells share one PostgreSQL instance, buffer pool, persistent-fragment "
    "store, storage device, and host telemetry. Concurrent DB cells would "
    "contaminate cache state and resource/QPS attribution; client concurrency "
    "is varied only inside one cell."
)
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")


class MatchedThroughputError(RuntimeError):
    """The formal matched-throughput protocol cannot be satisfied."""


@dataclass(frozen=True)
class WorkloadBinding:
    path: Path
    sha256: str
    measurement_path: Path
    measurement_sha256: str


@dataclass(frozen=True)
class CellPaths:
    prefix: Path
    requests: Path
    repeats: Path
    manifest: Path
    log: Path


@dataclass(frozen=True)
class ProtocolSlice:
    name: str
    selector_policy: str
    clients: tuple[int, ...]
    repeats: int
    expected_pairs: int | None
    fixed_target_recall: float | None
    selection_csv: Path
    out_dir: Path
    fixed_targets: tuple[float, ...] = ()
    client_cpu_list: str = DEFAULT_CLIENT_CPU_LIST
    backend_cpu_list: str = DEFAULT_BACKEND_CPU_LIST

    @property
    def expected_cells(self) -> int | None:
        if self.expected_pairs is None:
            return None
        return self.expected_pairs * len(self.clients)

    @property
    def expected_request_rows_per_cell(self) -> int:
        return EXPECTED_REQUESTS * self.repeats * len(MODES_BY_ARM)

    @property
    def expected_repeat_rows_per_cell(self) -> int:
        return self.repeats * len(MODES_BY_ARM)


PROTOCOL_SLICES = {
    DISTINCT_C16_PROTOCOL: ProtocolSlice(
        name=DISTINCT_C16_PROTOCOL,
        selector_policy="distinct_pairs",
        clients=(16,),
        repeats=3,
        expected_pairs=32,
        fixed_target_recall=None,
        selection_csv=DEFAULT_SELECTION_CSV_BY_PROTOCOL[
            DISTINCT_C16_PROTOCOL
        ],
        out_dir=DEFAULT_OUT_DIR_BY_PROTOCOL[DISTINCT_C16_PROTOCOL],
    ),
    FIXED_R090_PROTOCOL: ProtocolSlice(
        name=FIXED_R090_PROTOCOL,
        selector_policy="fixed",
        clients=SERVICE_CURVE_CLIENTS,
        repeats=6,
        expected_pairs=3,
        fixed_target_recall=0.90,
        selection_csv=DEFAULT_SELECTION_CSV_BY_PROTOCOL[
            FIXED_R090_PROTOCOL
        ],
        out_dir=DEFAULT_OUT_DIR_BY_PROTOCOL[FIXED_R090_PROTOCOL],
    ),
    FIXED_TARGETS_C16_PROTOCOL: ProtocolSlice(
        name=FIXED_TARGETS_C16_PROTOCOL,
        selector_policy="fixed",
        clients=(16,),
        repeats=3,
        expected_pairs=None,
        fixed_target_recall=None,
        selection_csv=DEFAULT_SELECTION_CSV_BY_PROTOCOL[
            FIXED_TARGETS_C16_PROTOCOL
        ],
        out_dir=DEFAULT_OUT_DIR_BY_PROTOCOL[FIXED_TARGETS_C16_PROTOCOL],
        fixed_targets=FORMAL_FIXED_TARGETS,
        backend_cpu_list="48-63",
    ),
}


def utc_now() -> str:
    return matched.utc_now()


def sha256_file(path: Path) -> str:
    return matched.sha256_file(path)


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    matched.atomic_json(path, dict(payload))


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return matched.read_json(path, label)
    except matched.MatchedLatencyError as exc:
        raise MatchedThroughputError(str(exc)) from exc


def require_int(value: object, label: str, *, lower: int = 0) -> int:
    try:
        return matched.require_int(value, label, lower=lower)
    except matched.MatchedLatencyError as exc:
        raise MatchedThroughputError(str(exc)) from exc


def require_float(value: object, label: str, *, lower: float = 0.0) -> float:
    try:
        number = matched.require_float(value, label, lower=lower)
    except matched.MatchedLatencyError as exc:
        raise MatchedThroughputError(str(exc)) from exc
    if not math.isfinite(number):
        raise MatchedThroughputError(f"{label} is not finite")
    return number


def require_sha(value: object, label: str) -> str:
    try:
        return matched.require_sha(value, label)
    except matched.MatchedLatencyError as exc:
        raise MatchedThroughputError(str(exc)) from exc


def resolve_path(value: object) -> Path:
    return matched.resolve_path(value)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def acquire_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise MatchedThroughputError(
            f"another matched-throughput runner owns {path}"
        ) from exc
    return handle


def parse_client_grid(values: Iterable[int]) -> tuple[int, ...]:
    clients = tuple(values)
    if not clients:
        raise MatchedThroughputError("client grid is empty")
    if len(set(clients)) != len(clients):
        raise MatchedThroughputError("client grid contains duplicates")
    if tuple(sorted(clients)) != clients:
        raise MatchedThroughputError("client grid must be strictly increasing")
    if clients[0] < 1 or clients[-1] > SERVICE_CURVE_MAX_CLIENTS:
        raise MatchedThroughputError(
            f"client grid must be within [1, {SERVICE_CURVE_MAX_CLIENTS}]"
        )
    return clients


def selected_protocol_slice(name: str) -> ProtocolSlice:
    try:
        return PROTOCOL_SLICES[name]
    except KeyError as exc:
        raise MatchedThroughputError(
            f"unknown service protocol slice {name!r}"
        ) from exc


def protocol_client_grid(
    values: Iterable[int],
    protocol_slice: ProtocolSlice,
) -> tuple[int, ...]:
    clients = parse_client_grid(values)
    if clients != protocol_slice.clients:
        raise MatchedThroughputError(
            f"{protocol_slice.name} requires clients="
            f"{list(protocol_slice.clients)}, observed={list(clients)}"
        )
    return clients


def protocol_pairs(
    pairs: Sequence[matched.SelectedPair],
    protocol_slice: ProtocolSlice,
) -> list[matched.SelectedPair]:
    if protocol_slice.fixed_targets:
        selected = [
            pair
            for pair in pairs
            if any(
                _close(pair.target_recall, target)
                for target in protocol_slice.fixed_targets
            )
        ]
    elif protocol_slice.fixed_target_recall is None:
        selected = list(pairs)
    else:
        selected = [
            pair
            for pair in pairs
            if _close(
                pair.target_recall,
                protocol_slice.fixed_target_recall,
            )
        ]
    if not selected:
        raise MatchedThroughputError(
            f"selector has no pairs for {protocol_slice.name}"
        )
    return selected


def requested_pairs(
    pairs: Sequence[matched.SelectedPair],
    *,
    datasets: Sequence[str],
    pair_ids: Sequence[str],
) -> list[matched.SelectedPair]:
    dataset_filter = set(datasets)
    pair_filter = set(pair_ids)
    selected = [
        pair
        for pair in pairs
        if (not dataset_filter or pair.dataset in dataset_filter)
        and (not pair_filter or pair.pair_id in pair_filter)
    ]
    observed_ids = {pair.pair_id for pair in selected}
    if pair_filter - observed_ids:
        raise MatchedThroughputError(
            "requested pair IDs are outside the selected protocol slice: "
            + ", ".join(sorted(pair_filter - observed_ids))
        )
    if not selected:
        raise MatchedThroughputError(
            "no protocol pairs match the requested dataset/pair filters"
        )
    return selected


def service_curve_client_cpu_assignment(
    client_cpu_list: str | None, clients: int
) -> tuple[int, ...]:
    if not 1 <= clients <= SERVICE_CURVE_MAX_CLIENTS:
        raise throughput.Figure5ThroughputError(
            f"clients must be in [1, {SERVICE_CURVE_MAX_CLIENTS}], "
            f"observed={clients}"
        )
    cpus = tuple(throughput.telemetry.parse_cpu_set(client_cpu_list))
    if not cpus:
        return ()
    return tuple(cpus[client_id % len(cpus)] for client_id in range(clients))


def validate_cpu_lists(
    client_cpu_list: str, backend_cpu_list: str, clients: Sequence[int]
) -> dict[str, object]:
    maximum = max(clients)
    try:
        client_cpus = service_curve_client_cpu_assignment(
            client_cpu_list, maximum
        )
        backend_cpus = tuple(
            throughput.telemetry.parse_cpu_set(backend_cpu_list)
        )
    except (
        ValueError,
        argparse.ArgumentTypeError,
        throughput.Figure5ThroughputError,
    ) as exc:
        raise MatchedThroughputError(f"invalid CPU binding: {exc}") from exc
    if not backend_cpus:
        raise MatchedThroughputError(
            "backend CPU list is empty"
        )
    overlap = sorted(set(client_cpus) & set(backend_cpus))
    if overlap:
        raise MatchedThroughputError(
            f"client and PostgreSQL backend CPU lists overlap: {overlap}"
        )
    return {
        "client_cpu_list": client_cpu_list,
        "client_cpus_for_max_clients": list(client_cpus),
        "client_cpu_assignment_policy": (
            "round_robin_within_partition_by_client_id"
        ),
        "client_cpu_partition_size": len(set(client_cpus)),
        "backend_cpu_list": backend_cpu_list,
        "backend_cpus": list(backend_cpus),
        "backend_cpu_policy": (
            "all independent PostgreSQL backends confined to this partition"
        ),
    }


def full_release_scope(
    args: argparse.Namespace,
    protocol_slice: ProtocolSlice,
    clients: Sequence[int],
    pairs: Sequence[matched.SelectedPair],
    all_pairs: Sequence[matched.SelectedPair],
    *,
    selection_bindings: Mapping[str, object] | None = None,
    enforce_frozen_selector: bool = False,
) -> dict[str, object]:
    bindings = selection_bindings or {}
    expected_pair_count = protocol_slice.expected_pairs
    if expected_pair_count is None and protocol_slice.fixed_targets:
        expected_pair_count = int(bindings.get("selected_pairs") or 0)
    selected_datasets = {pair.dataset for pair in all_pairs}
    checks = {
        "all_datasets_requested": (
            not args.datasets
            or set(args.datasets) == set(matched.FROZEN_DATASETS)
        ),
        "all_selected_pairs_requested": (
            not args.pair_ids
            and {pair.pair_id for pair in pairs}
            == {pair.pair_id for pair in all_pairs}
        ),
        "protocol_client_grid": tuple(clients) == protocol_slice.clients,
        "protocol_pair_count": (
            expected_pair_count is not None
            and len(all_pairs) == expected_pair_count
        ),
        "protocol_dataset_coverage": (
            selected_datasets == set(matched.FROZEN_DATASETS)
            or bool(protocol_slice.fixed_targets)
        ),
        "default_client_cpu_partition": (
            args.client_cpu_list == protocol_slice.client_cpu_list
        ),
        "default_backend_cpu_partition": (
            args.backend_cpu_list == protocol_slice.backend_cpu_list
        ),
        "warm_relation_protocol": args.pg_prewarm is True,
    }
    if enforce_frozen_selector:
        datasets = selected_datasets
        per_dataset = {
            dataset: [pair for pair in all_pairs if pair.dataset == dataset]
            for dataset in matched.FROZEN_DATASETS
        }
        checks["selector_covers_frozen_datasets"] = (
            datasets == set(matched.FROZEN_DATASETS)
        )
        checks["selector_policy_matches_protocol"] = (
            bindings.get("target_policy")
            == protocol_slice.selector_policy
        )
        if protocol_slice.fixed_targets:
            targets_by_dataset = bindings.get("targets_by_dataset")
            if not isinstance(targets_by_dataset, Mapping):
                targets_by_dataset = {}
            target_rows = int(bindings.get("target_rows") or 0)
            selected_pairs = int(bindings.get("selected_pairs") or 0)
            unattainable_pairs = int(
                bindings.get("unattainable_pairs") or 0
            )
            expected_targets = list(protocol_slice.fixed_targets)
            selected_keys = {
                (
                    pair.dataset,
                    next(
                        (
                            target
                            for target in expected_targets
                            if _close(pair.target_recall, target)
                        ),
                        None,
                    ),
                )
                for pair in all_pairs
            }
            checks.update({
                "selector_uses_formal_predicate_qualification": (
                    bindings.get("qualification_scope")
                    == matched.QUALIFICATION_SCOPE_FORMAL
                ),
                "selector_covers_frozen_datasets": (
                    set(targets_by_dataset) == set(matched.FROZEN_DATASETS)
                ),
                "selector_uses_formal_fixed_targets": all(
                    list(targets_by_dataset.get(dataset, ()))
                    == expected_targets
                    for dataset in matched.FROZEN_DATASETS
                ),
                "selector_resolves_every_fixed_target": (
                    target_rows
                    == len(matched.FROZEN_DATASETS) * len(expected_targets)
                    and selected_pairs + unattainable_pairs == target_rows
                    and selected_pairs == len(all_pairs)
                ),
                "selected_pairs_are_unique_fixed_targets": (
                    len(selected_keys) == len(all_pairs)
                    and all(target is not None for _, target in selected_keys)
                ),
            })
        elif protocol_slice.fixed_target_recall is None:
            checks.update({
                "selector_declares_minimum_point_gate": (
                    int(
                        bindings.get(
                            "min_distinct_pairs_per_dataset"
                        )
                        or 0
                    )
                    >= matched.MIN_FORMAL_POINTS_PER_ARM_DATASET
                ),
                "selector_uses_formal_target_floor": (
                    float(bindings.get("target_floor") or -1.0) >= 0.70
                ),
                "selector_has_minimum_distinct_stock_points": all(
                    len({
                        pair.stock.get("config_sha256")
                        for pair in dataset_pairs
                    })
                    >= matched.MIN_FORMAL_POINTS_PER_ARM_DATASET
                    for dataset_pairs in per_dataset.values()
                ),
                "selector_has_minimum_distinct_sqlens_points": all(
                    len({
                        pair.sqlens.get("config_sha256")
                        for pair in dataset_pairs
                    })
                    >= matched.MIN_FORMAL_POINTS_PER_ARM_DATASET
                    for dataset_pairs in per_dataset.values()
                ),
            })
        else:
            checks.update({
                "fixed_target_is_recall_090": _close(
                    protocol_slice.fixed_target_recall, 0.90
                ),
                "one_fixed_target_pair_per_dataset": all(
                    len(dataset_pairs) == 1
                    and _close(
                        dataset_pairs[0].target_recall,
                        protocol_slice.fixed_target_recall,
                    )
                    for dataset_pairs in per_dataset.values()
                ),
                "selector_resolves_formal_fixed_targets": (
                    int(bindings.get("target_rows") or 0) == 9
                    and int(bindings.get("selected_pairs") or 0)
                    + int(bindings.get("unattainable_pairs") or 0)
                    == 9
                ),
            })
    return {
        "requested": all(checks.values()),
        "kind": protocol_slice.name,
        "checks": checks,
        "required_pairs": sorted(pair.pair_id for pair in all_pairs),
        "required_pair_cells": sorted(
            (
                {
                    "dataset": pair.dataset,
                    "pair_id": pair.pair_id,
                    "target_recall": pair.target_recall,
                }
                for pair in all_pairs
            ),
            key=lambda item: (
                str(item["dataset"]),
                str(item["pair_id"]),
            ),
        ),
        "requested_pairs": sorted(pair.pair_id for pair in pairs),
        "required_clients": list(protocol_slice.clients),
        "requested_clients": list(clients),
        "required_repeats": protocol_slice.repeats,
    }


def workload_manifest_overrides(
    values: Sequence[str],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        dataset, separator, raw_path = value.partition("=")
        if (
            not separator
            or dataset not in DEFAULT_WORKLOAD_MANIFESTS
            or not raw_path.strip()
        ):
            raise MatchedThroughputError(
                "--workload-manifest expects "
                "amazon|yfcc|laion=/path/to/manifest.json"
            )
        if dataset in result:
            raise MatchedThroughputError(
                f"duplicate workload manifest override for {dataset}"
            )
        path = Path(raw_path.strip())
        result[dataset] = (
            path if path.is_absolute() else (ROOT / path)
        ).resolve()
    return result


def validate_workload_manifest(
    dataset: Mapping[str, object], path: Path
) -> WorkloadBinding:
    if not path.is_file():
        raise MatchedThroughputError(f"workload manifest is missing: {path}")
    payload = read_json(path, "workload manifest")
    if (
        payload.get("artifact_type") != "figure5_frontier_workload"
        or payload.get("artifact_valid") is not True
    ):
        raise MatchedThroughputError(
            f"workload manifest is not a valid Figure 5 workload: {path}"
        )
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        raise MatchedThroughputError("workload manifest has no output bindings")
    measurement_output = outputs.get("measurement_workload_csv")
    if not isinstance(measurement_output, Mapping):
        raise MatchedThroughputError(
            "workload manifest does not bind measurement_workload_csv"
        )
    measurement = resolve_path(dataset["measurement_workload_csv"]).resolve()
    manifest_measurement = resolve_path(measurement_output.get("path")).resolve()
    if manifest_measurement != measurement:
        raise MatchedThroughputError(
            "workload manifest measurement path differs from frozen config"
        )
    if not measurement.is_file():
        raise MatchedThroughputError(
            f"measurement workload is missing: {measurement}"
        )
    observed_sha = sha256_file(measurement)
    if (
        require_sha(
            measurement_output.get("sha256"), "measurement workload SHA"
        )
        != observed_sha
        or require_int(
            measurement_output.get("rows"),
            "measurement workload rows",
            lower=1,
        )
        != EXPECTED_REQUESTS
        or matched.frontier.count_csv_rows(measurement) != EXPECTED_REQUESTS
    ):
        raise MatchedThroughputError(
            f"measurement workload binding failed: {measurement}"
        )
    return WorkloadBinding(
        path=path.resolve(),
        sha256=sha256_file(path),
        measurement_path=measurement,
        measurement_sha256=observed_sha,
    )


def normalized_pair(pair: matched.SelectedPair) -> dict[str, object]:
    fields = (
        "ef_search",
        "iterative_scan",
        "max_scan_tuples",
        "scan_mem_multiplier",
        "guided_collect_target",
        "traversal_guided_target",
    )
    return {
        "pair_id": pair.pair_id,
        "config_id": pair.pair_id,
        "dataset": pair.dataset,
        "target_recall": pair.target_recall,
        "selection_status": "selected",
        "stock": {
            **{field: pair.stock[field] for field in fields},
            "traversal_guided_burst": TRAVERSAL_GUIDED_BURST,
            "d2_page_access": pair.stock["d2_page_access"],
            "d2_index_page_access": pair.stock["d2_index_page_access"],
        },
        "sqlens": {
            **{field: pair.sqlens[field] for field in fields},
            "traversal_guided_burst": TRAVERSAL_GUIDED_BURST,
            "d2_page_access": pair.sqlens["d2_page_access"],
            "d2_index_page_access": pair.sqlens["d2_index_page_access"],
        },
    }


def normalized_measurement_plan(
    pairs: Sequence[matched.SelectedPair],
    *,
    config_path: Path,
    config_sha256: str,
    selection_csv: Path,
    selection_plan: Path,
    selection_manifest: Path,
    selection_bindings: Mapping[str, str],
    release: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "sqlens_figure5_r36_normalized_throughput_plan",
        "runner_version": RUNNER_VERSION,
        "artifact_valid": True,
        "normalization": {
            "traversal_guided_burst": TRAVERSAL_GUIDED_BURST,
            "reason": (
                "The selector freezes the tuned fields; r36 matched-latency "
                "and throughput both use traversal_guided_burst=8."
            ),
        },
        "frontier_config": {
            "path": str(config_path),
            "sha256": config_sha256,
        },
        "release_contract": dict(release),
        "selector": {
            "csv": str(selection_csv),
            "plan": str(selection_plan),
            "manifest": str(selection_manifest),
            **dict(selection_bindings),
        },
        "required_grid_contract": dict(
            selection_bindings["required_grid_contract"]
        ),
        "pairs": [normalized_pair(pair) for pair in pairs],
    }


def publish_normalized_plan(
    path: Path, payload: Mapping[str, object], *, overwrite: bool
) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    expected_sha = hashlib.sha256(encoded).hexdigest()
    if path.exists():
        if sha256_file(path) == expected_sha:
            return expected_sha
        if not overwrite:
            raise MatchedThroughputError(
                f"normalized measurement plan differs: {path}; use --overwrite"
            )
    atomic_bytes(path, encoded)
    if sha256_file(path) != expected_sha:
        raise MatchedThroughputError(
            f"normalized measurement plan failed post-publish SHA check: {path}"
        )
    return expected_sha


def safe_pair_component(pair: matched.SelectedPair) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", pair.pair_id).strip(".-")
    if not SAFE_COMPONENT_RE.fullmatch(component):
        raise MatchedThroughputError(
            f"unsafe pair ID for output path: {pair.pair_id!r}"
        )
    return component


def cell_paths(
    out_dir: Path, pair: matched.SelectedPair, clients: int
) -> CellPaths:
    prefix = out_dir / (
        f"figure5_r36_{pair.dataset}_matched_throughput_"
        f"{safe_pair_component(pair)}_c{clients}"
    )
    outputs = throughput.output_paths(prefix)
    return CellPaths(
        prefix=prefix,
        requests=outputs["requests"],
        repeats=outputs["repeats"],
        manifest=outputs["manifest"],
        log=Path(str(prefix) + ".log"),
    )


def expected_search_settings(
    pair: matched.SelectedPair,
) -> throughput.SearchSettings:
    def arm(value: Mapping[str, object]) -> throughput.ArmSearchSettings:
        return throughput.ArmSearchSettings(
            ef_search=int(value["ef_search"]),
            iterative_scan=str(value["iterative_scan"]),
            max_scan_tuples=int(value["max_scan_tuples"]),
            scan_mem_multiplier=float(value["scan_mem_multiplier"]),
            guided_collect_target=int(value["guided_collect_target"]),
            traversal_guided_target=int(value["traversal_guided_target"]),
            traversal_guided_burst=TRAVERSAL_GUIDED_BURST,
        )

    return throughput.SearchSettings(
        config_id=pair.pair_id,
        pair_id=pair.pair_id,
        target_recall=pair.target_recall,
        stock=arm(pair.stock),
        sqlens=arm(pair.sqlens),
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise MatchedThroughputError(f"CSV has no header: {path}")
        return list(reader)


def _true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _identity_value(value: object) -> str:
    return str(value if value is not None else "").strip()


def frozen_workload_identity(path: Path) -> dict[int, tuple[str, str, str]]:
    """Load the immutable request identity used by every throughput arm."""
    required = {"request_no", "query_id", "query_no", "filter_name"}
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or ())
            missing = sorted(required - fields)
            if missing:
                raise MatchedThroughputError(
                    f"frozen workload is missing fields: {missing}"
                )
            identity: dict[int, tuple[str, str, str]] = {}
            for row in reader:
                request_no = require_int(
                    row.get("request_no"), "frozen request_no", lower=0
                )
                if request_no in identity:
                    raise MatchedThroughputError(
                        f"frozen workload repeats request_no={request_no}"
                    )
                values = tuple(
                    _identity_value(row.get(field))
                    for field in ("query_id", "query_no", "filter_name")
                )
                if any(not value for value in values):
                    raise MatchedThroughputError(
                        f"frozen workload has an empty identity at request_no={request_no}"
                    )
                identity[request_no] = values
    except OSError as exc:
        raise MatchedThroughputError(
            f"cannot read frozen measurement workload: {path}"
        ) from exc
    expected = set(range(EXPECTED_REQUESTS))
    if set(identity) != expected:
        raise MatchedThroughputError(
            "frozen workload request_no coverage is not exactly 0..9999"
        )
    return identity


def matched_recall_gate(
    rows: Sequence[Mapping[str, str]],
    pair: matched.SelectedPair,
    *,
    repeats: int | None = None,
) -> dict[str, object]:
    """Apply the formal aggregate and per-predicate matched-recall contract."""
    try:
        from .figure5_latency_repeats import query_cluster_bootstrap_recall
    except ImportError:
        from figure5_latency_repeats import query_cluster_bootstrap_recall

    repeat_count = EXPECTED_REPEATS if repeats is None else repeats
    expected = {
        (arm, repeat)
        for arm in MODES_BY_ARM
        for repeat in range(repeat_count)
    }
    grouped: dict[tuple[str, int], list[Mapping[str, str]]] = {
        key: [] for key in expected
    }
    for row in rows:
        try:
            arm = str(row.get("arm_id") or "")
            repeat = require_int(row.get("repeat_id"), "recall repeat", lower=0)
            value = float(_identity_value(row.get("recall_at_10")))
        except (TypeError, ValueError, MatchedThroughputError):
            return {"passed": False, "reason": "invalid recall row", "arms": {}}
        filter_name = str(row.get("filter_name") or "").strip()
        query_id = str(row.get("query_id") or "").strip()
        key = (arm, repeat)
        if (
            key not in grouped
            or not filter_name
            or not query_id
            or not 0.0 <= value <= 1.0
        ):
            return {
                "passed": False,
                "reason": "recall row is outside the expected domain",
                "aggregate": {},
                "per_predicate": {},
            }
        grouped[key].append(row)

    filter_names = sorted(
        {
            str(row.get("filter_name") or "").strip()
            for group in grouped.values()
            for row in group
        }
    )
    if len(filter_names) != EXPECTED_FORMAL_PREDICATES:
        return {
            "qualification_scope": matched.QUALIFICATION_SCOPE_FORMAL,
            "formal_predicate_sample_floor": (
                matched.MIN_FORMAL_PREDICATE_SAMPLES
            ),
            "expected_predicate_count": EXPECTED_FORMAL_PREDICATES,
            "observed_predicate_count": len(filter_names),
            "filter_names": filter_names,
            "passed": False,
            "paper_eligible": False,
            "reason": (
                "formal recall coverage does not contain exactly "
                f"{EXPECTED_FORMAL_PREDICATES} predicates"
            ),
            "aggregate": {},
            "per_predicate": {},
        }

    aggregate: dict[str, dict[str, object]] = {}
    per_predicate: dict[str, dict[str, dict[str, object]]] = {}
    worst_by_arm: dict[str, dict[str, object]] = {}
    aggregate_passed = True
    predicate_passed = True
    coverage_complete = True
    for arm in MODES_BY_ARM:
        for repeat in range(repeat_count):
            group = grouped[(arm, repeat)]
            if len(group) != EXPECTED_REQUESTS:
                return {
                    "passed": False,
                    "paper_eligible": False,
                    "reason": (
                        f"{arm}/repeat={repeat} aggregate recall coverage "
                        "is incomplete"
                    ),
                    "aggregate": aggregate,
                    "per_predicate": per_predicate,
                }
            arm_key = f"{arm}/repeat={repeat}"
            try:
                bootstrap = query_cluster_bootstrap_recall(
                    group,
                    value_field="recall_at_10",
                    seed_label=f"{pair.pair_id}:{arm_key}:throughput",
                )
            except Exception as exc:
                return {
                    "passed": False,
                    "paper_eligible": False,
                    "reason": f"{arm_key} cluster bootstrap failed: {exc}",
                    "aggregate": aggregate,
                    "per_predicate": per_predicate,
                }
            mean = float(bootstrap["mean"])
            lower = float(bootstrap["lower"])
            upper = float(bootstrap["upper"])
            aggregate[arm_key] = {
                "sample_count": len(group),
                "query_cluster_count": bootstrap["query_cluster_count"],
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "target": pair.target_recall,
                "passed": lower >= pair.target_recall,
                "ci_method": bootstrap["method"],
                "bootstrap_samples": bootstrap["samples"],
                "bootstrap_seed": bootstrap["seed"],
                "bootstrap_seed_label": bootstrap["seed_label"],
            }
            aggregate_passed = (
                aggregate_passed and lower >= pair.target_recall
            )

            per_predicate[arm_key] = {}
            for filter_name in filter_names:
                bootstrap_stats = bootstrap["per_predicate"].get(filter_name)
                if not isinstance(bootstrap_stats, Mapping):
                    bootstrap_stats = {}
                sample_count = int(bootstrap_stats.get("sample_count") or 0)
                sufficient = (
                    sample_count >= matched.MIN_FORMAL_PREDICATE_SAMPLES
                )
                filter_mean = bootstrap_stats.get("mean")
                filter_lower = bootstrap_stats.get("lower")
                filter_upper = bootstrap_stats.get("upper")
                filter_passed = bool(
                    sufficient
                    and filter_lower is not None
                    and float(filter_lower) >= pair.target_recall
                )
                stats: dict[str, object] = {
                    "sample_count": sample_count,
                    "query_cluster_count": int(
                        bootstrap_stats.get("query_cluster_count") or 0
                    ),
                    "sample_count_sufficient": sufficient,
                    "mean": filter_mean,
                    "lower": filter_lower,
                    "upper": filter_upper,
                    "target": pair.target_recall,
                    "passed": filter_passed,
                    "ci_method": bootstrap["method"],
                }
                per_predicate[arm_key][filter_name] = stats
                coverage_complete = coverage_complete and sample_count > 0
                predicate_passed = predicate_passed and filter_passed

            worst_by_arm[arm_key] = min(
                (
                    {"filter_name": filter_name, **stats}
                    for filter_name, stats in per_predicate[arm_key].items()
                ),
                key=lambda item: (
                    float(item["lower"])
                    if item["lower"] is not None
                    else float("-inf"),
                    float(item["mean"])
                    if item["mean"] is not None
                    else float("-inf"),
                    str(item["filter_name"]),
                ),
            )

    worst_predicate = min(
        (
            {"arm_repeat": arm_key, **stats}
            for arm_key, stats in worst_by_arm.items()
        ),
        key=lambda item: (
            float(item["lower"])
            if item["lower"] is not None
            else float("-inf"),
            float(item["mean"])
            if item["mean"] is not None
            else float("-inf"),
            str(item["arm_repeat"]),
            str(item["filter_name"]),
        ),
    )
    passed = aggregate_passed and coverage_complete and predicate_passed
    if not coverage_complete:
        reason = "per-predicate recall coverage is incomplete"
    elif not predicate_passed:
        reason = "per-predicate recall LCB or sample-count gate misses target"
    elif not aggregate_passed:
        reason = "aggregate recall LCB misses target"
    else:
        reason = "ok"
    return {
        "qualification_scope": matched.QUALIFICATION_SCOPE_FORMAL,
        "formal_predicate_sample_floor": (
            matched.MIN_FORMAL_PREDICATE_SAMPLES
        ),
        "expected_predicate_count": EXPECTED_FORMAL_PREDICATES,
        "observed_predicate_count": len(filter_names),
        "filter_names": filter_names,
        "recall_ci_method": (
            "query_id_cluster_stratified_predicate_percentile_bootstrap_95"
        ),
        "passed": passed,
        "paper_eligible": passed,
        "reason": reason,
        "aggregate": aggregate,
        "per_predicate": per_predicate,
        "worst_predicate_by_arm": worst_by_arm,
        "worst_predicate": worst_predicate,
    }


def expected_request_dispatch(
    *,
    schedule_seed: int,
    dataset_id: str,
    config_id: str,
    clients: int,
    repeat_id: int,
    requests: int = EXPECTED_REQUESTS,
) -> tuple[int, str, dict[int, int]]:
    """Reconstruct the core runner's deterministic request permutation."""
    trace_seed = throughput.telemetry.stable_seed(
        schedule_seed,
        "figure5_mixed_q10k",
        dataset_id,
        config_id,
        clients,
        repeat_id,
    )
    request_nos = list(range(requests))
    random.Random(trace_seed).shuffle(request_nos)
    order_sha = throughput.canonical_sha256(request_nos)
    positions = {
        request_no: dispatch_position
        for dispatch_position, request_no in enumerate(request_nos)
    }
    return trace_seed, order_sha, positions


def effective_mode_config_gate(
    manifest: Mapping[str, object],
    pair: matched.SelectedPair,
    expected_arm_shas: Mapping[str, str],
) -> list[str]:
    """Require the core's effective mode settings to equal the frozen pair."""
    settings = expected_search_settings(pair)
    expected = {
        "stock_pgvector": {
            "mode_id": "original",
            "search": settings.stock.mode_config(guidance_enabled=False),
            "config_sha256": expected_arm_shas["stock_pgvector"],
        },
        "sqlens_full": {
            "mode_id": "design1_bloom_bfs_layout_d3",
            "search": settings.sqlens.mode_config(guidance_enabled=True),
            "config_sha256": expected_arm_shas["sqlens_full"],
            "d3_measurement_policy": "workload_driven_adaptive",
            "unmeasured_query_count": 0,
        },
    }
    methods = manifest.get("methods")
    if not isinstance(methods, Mapping):
        return ["core manifest lacks effective mode methods"]
    reasons: list[str] = []
    for arm, expected_method in expected.items():
        actual = methods.get(arm)
        if not isinstance(actual, Mapping):
            reasons.append(f"core manifest lacks effective method {arm}")
            continue
        for field, value in expected_method.items():
            if actual.get(field) != value:
                reasons.append(f"effective {arm} {field} differs from frozen pair")
    return reasons


def provenance_gate(
    manifest: Mapping[str, object],
    *,
    pair: matched.SelectedPair,
    config_path: Path,
    normalized_plan: Path,
    workload: WorkloadBinding,
) -> list[str]:
    """Bind code, plan, config, and frozen workload provenance at cell scope."""
    reasons: list[str] = []
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        return ["core manifest lacks provenance inputs"]
    expected_sources = {
        "throughput_core": {
            "path": str(Path(throughput.__file__).resolve()),
            "sha256": sha256_file(Path(throughput.__file__).resolve()),
        },
        "orchestrator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    if inputs.get("execution_sources") != expected_sources:
        reasons.append("effective execution-source provenance is invalid")
    try:
        plan_payload = read_json(normalized_plan, "normalized measurement plan")
        plan_rows = plan_payload.get("pairs", plan_payload.get("rows"))
        selected_rows = [
            row
            for row in (plan_rows if isinstance(plan_rows, list) else [])
            if isinstance(row, Mapping) and row.get("pair_id") == pair.pair_id
        ]
        if len(selected_rows) != 1 or selected_rows[0] != normalized_pair(pair):
            reasons.append("normalized measurement-plan content is invalid")
    except (OSError, ValueError, KeyError, MatchedThroughputError):
        reasons.append("normalized measurement-plan content is unreadable")
    pair_input = inputs.get("measurement_pair")
    if not isinstance(pair_input, Mapping) or (
        pair_input.get("source") != "measurement_plan"
        or Path(str(pair_input.get("path") or "")).resolve()
        != normalized_plan.resolve()
        or pair_input.get("sha256") != sha256_file(normalized_plan)
        or pair_input.get("pair_id") != pair.pair_id
        or not _close(float(pair_input.get("target_recall")), pair.target_recall)
    ):
        reasons.append("effective measurement-plan provenance is invalid")
    config_input = inputs.get("frontier_config")
    if not isinstance(config_input, Mapping) or (
        Path(str(config_input.get("path") or "")).resolve() != config_path.resolve()
        or config_input.get("sha256") != sha256_file(config_path)
    ):
        reasons.append("effective frontier-config provenance is invalid")
    workload_input = inputs.get("workload_manifest")
    if not isinstance(workload_input, Mapping) or (
        Path(str(workload_input.get("path") or "")).resolve()
        != workload.path.resolve()
        or workload_input.get("sha256") != workload.sha256
    ):
        reasons.append("effective workload-manifest provenance is invalid")
    return reasons


def complete_throughput_prewarm(
    value: object, dataset: Mapping[str, object]
) -> bool:
    expected_relations = list(
        dict.fromkeys(
            str(dataset[field])
            for field in (
                "table",
                "source_index",
                "bfs_index",
                "query_table",
            )
        )
    )
    if (
        not isinstance(value, Mapping)
        or value.get("enabled") is not True
        or value.get("complete") is not True
        or value.get("method") != "pg_prewarm(regclass,'read','main')"
    ):
        return False
    records = value.get("records")
    if not isinstance(records, list) or len(records) != len(
        expected_relations
    ):
        return False
    try:
        return {
            str(record.get("relation"))
            for record in records
            if isinstance(record, Mapping)
            and require_int(
                record.get("expected_blocks"),
                "prewarm expected blocks",
            )
            == require_int(
                record.get("warmed_blocks"),
                "prewarm warmed blocks",
            )
        } == set(expected_relations)
    except MatchedThroughputError:
        return False


def audit_request_rows(
    rows: Sequence[Mapping[str, str]],
    pair: matched.SelectedPair,
    clients: int,
    *,
    workload_identity: Mapping[int, tuple[str, str, str]],
    run_id: str,
    release_sha: str,
    expected_arm_shas: Mapping[str, str],
    schedule_seed: int,
    dataset_id: str,
    config_id: str,
    repeats: int | None = None,
) -> list[str]:
    reasons: list[str] = []
    repeat_count = EXPECTED_REPEATS if repeats is None else repeats
    expected_rows = EXPECTED_REQUESTS * repeat_count * len(MODES_BY_ARM)
    if len(rows) != expected_rows:
        return [
            f"request row count {len(rows)} != {expected_rows}"
        ]
    groups: dict[tuple[str, int], list[Mapping[str, str]]] = {}
    for row in rows:
        arm = str(row.get("arm_id") or "")
        try:
            repeat = require_int(row.get("repeat_id"), "request repeat_id")
        except MatchedThroughputError as exc:
            reasons.append(str(exc))
            continue
        groups.setdefault((arm, repeat), []).append(row)
        expected_mode = MODES_BY_ARM.get(arm)
        checks = (
            (row.get("runner_version"), throughput.RUNNER_VERSION),
            (row.get("run_id"), run_id),
            (row.get("dataset"), throughput.DATASET_IDS[pair.dataset]),
            (row.get("pair_id"), pair.pair_id),
            (row.get("mode_id"), expected_mode),
            (row.get("clients"), str(clients)),
            (row.get("release_identity_sha256"), release_sha),
            (row.get("arm_config_sha256"), expected_arm_shas.get(arm)),
        )
        if any(str(actual) != str(expected) for actual, expected in checks):
            reasons.append("request row identity/config mismatch")
        if str(row.get("error_type") or "").strip() or str(
            row.get("error") or ""
        ).strip():
            reasons.append("request row contains an error")
    expected_keys = {
        (arm, repeat)
        for arm in MODES_BY_ARM
        for repeat in range(repeat_count)
    }
    if set(groups) != expected_keys:
        reasons.append("request arm/repeat groups are incomplete")
        return sorted(set(reasons))
    signatures: dict[
        tuple[str, int], set[tuple[int, str, str, str, str, str]]
    ] = {}
    for key, group in groups.items():
        arm, repeat = key
        if len(group) != EXPECTED_REQUESTS:
            reasons.append(f"request group {key} has {len(group)} rows")
            continue
        trace_seed, expected_trace_sha, expected_positions = (
            expected_request_dispatch(
                schedule_seed=schedule_seed,
                dataset_id=dataset_id,
                config_id=config_id,
                clients=clients,
                repeat_id=repeat,
                requests=EXPECTED_REQUESTS,
            )
        )
        request_nos = {
            require_int(row.get("request_no"), "request_no") for row in group
        }
        dispatch_positions = {
            require_int(row.get("dispatch_position"), "dispatch_position")
            for row in group
        }
        trace_shas = {str(row.get("trace_order_sha256") or "") for row in group}
        if (
            request_nos != set(range(EXPECTED_REQUESTS))
            or dispatch_positions != set(range(EXPECTED_REQUESTS))
            or len(trace_shas) != 1
            or not all(throughput.SHA256_RE.fullmatch(value) for value in trace_shas)
            or trace_shas != {expected_trace_sha}
        ):
            reasons.append(f"request group {key} trace coverage is invalid")
        client_counts: Counter[int] = Counter()
        client_positions: dict[int, set[int]] = {}
        for row in group:
            request_no = require_int(row.get("request_no"), "request_no", lower=0)
            dispatch_position = require_int(
                row.get("dispatch_position"), "dispatch_position", lower=0
            )
            client_id = require_int(row.get("client_id"), "client_id", lower=0)
            observed_trace_seed = require_int(
                row.get("trace_permutation_seed"),
                "trace_permutation_seed",
                lower=0,
            )
            if (
                client_id >= clients
                or client_id != dispatch_position % clients
                or observed_trace_seed != trace_seed
                or expected_positions.get(request_no) != dispatch_position
            ):
                reasons.append(f"request group {key} has invalid client dispatch")
            client_counts[client_id] += 1
            client_positions.setdefault(client_id, set()).add(dispatch_position)
            expected_identity = workload_identity.get(request_no)
            observed_identity = (
                _identity_value(row.get("query_id")),
                _identity_value(row.get("query_no")),
                _identity_value(row.get("filter_name")),
            )
            if expected_identity != observed_identity:
                reasons.append(
                    f"request group {key} differs from frozen workload at request_no={request_no}"
                )
        expected_counts = Counter(
            position % clients for position in range(EXPECTED_REQUESTS)
        )
        if client_counts != expected_counts:
            reasons.append(f"request group {key} client dispatch coverage is invalid")
        for client_id in range(clients):
            expected_positions = set(
                range(client_id, EXPECTED_REQUESTS, clients)
            )
            if client_positions.get(client_id, set()) != expected_positions:
                reasons.append(
                    f"request group {key} client {client_id} dispatch positions are invalid"
                )
        signatures[key] = {
            (
                require_int(row.get("dispatch_position"), "dispatch_position"),
                str(row.get("request_no") or ""),
                str(row.get("query_id") or ""),
                str(row.get("query_no") or ""),
                str(row.get("filter_name") or ""),
                str(row.get("client_id") or ""),
            )
            for row in group
        }
    for repeat in range(repeat_count):
        if signatures.get(("stock_pgvector", repeat)) != signatures.get(
            ("sqlens_full", repeat)
        ):
            reasons.append(f"paired request trace differs in repeat {repeat}")
    return sorted(set(reasons))


def audit_repeat_rows(
    rows: Sequence[Mapping[str, str]],
    pair: matched.SelectedPair,
    clients: int,
    *,
    run_id: str,
    release_sha: str,
    expected_arm_shas: Mapping[str, str],
    schedule_seed: int,
    dataset_id: str,
    config_id: str,
    repeats: int | None = None,
    backend_proc_root: Path | None = None,
) -> list[str]:
    reasons: list[str] = []
    repeat_count = EXPECTED_REPEATS if repeats is None else repeats
    expected_rows = repeat_count * len(MODES_BY_ARM)
    if len(rows) != expected_rows:
        return [f"repeat row count {len(rows)} != {expected_rows}"]
    groups: dict[tuple[str, int], Mapping[str, str]] = {}
    for row in rows:
        arm = str(row.get("arm_id") or "")
        try:
            repeat = require_int(row.get("repeat_id"), "repeat_id")
            completed = require_int(
                row.get("completed_queries"), "completed_queries"
            )
            requests = require_int(row.get("requests"), "requests")
            unique = require_int(row.get("unique_queries"), "unique_queries")
            errors = require_int(row.get("error_count"), "error_count")
            wall = require_float(
                row.get("wall_clock_seconds"),
                "wall_clock_seconds",
                lower=0.0,
            )
            qps = require_float(
                row.get("throughput_qps"), "throughput_qps", lower=0.0
            )
        except MatchedThroughputError as exc:
            reasons.append(str(exc))
            continue
        key = (arm, repeat)
        if key in groups:
            reasons.append(f"duplicate repeat row {key}")
        groups[key] = row
        expected_mode = MODES_BY_ARM.get(arm)
        expected_order = throughput.balanced_arm_order(repeat, schedule_seed)
        trace_seed, trace_order_sha, _ = expected_request_dispatch(
            schedule_seed=schedule_seed,
            dataset_id=dataset_id,
            config_id=config_id,
            clients=clients,
            repeat_id=repeat,
            requests=EXPECTED_REQUESTS,
        )
        checks = (
            (row.get("runner_version"), throughput.RUNNER_VERSION),
            (row.get("run_id"), run_id),
            (row.get("dataset"), throughput.DATASET_IDS[pair.dataset]),
            (row.get("pair_id"), pair.pair_id),
            (row.get("mode_id"), expected_mode),
            (row.get("clients"), str(clients)),
            (row.get("release_identity_sha256"), release_sha),
            (row.get("arm_config_sha256"), expected_arm_shas.get(arm)),
            (row.get("status"), "valid"),
            (row.get("throughput_source"), throughput.THROUGHPUT_SOURCE),
            (
                row.get("arm_order"),
                repeat * len(MODES_BY_ARM) + expected_order.index(expected_mode),
            ),
            (row.get("trace_permutation_seed"), trace_seed),
            (row.get("trace_order_sha256"), trace_order_sha),
        )
        if any(str(actual) != str(expected) for actual, expected in checks):
            reasons.append("repeat row identity/config/status mismatch")
        if (
            requests != EXPECTED_REQUESTS
            or unique != EXPECTED_REQUESTS
            or completed != EXPECTED_REQUESTS
            or errors != 0
            or wall <= 0.0
            or qps <= 0.0
            or not _close(qps, completed / wall)
        ):
            reasons.append(
                "repeat QPS is not completed_queries/barrier_wall_clock_seconds"
            )
        if not _true(row.get("telemetry_collected")):
            reasons.append("repeat telemetry is incomplete")
        if backend_proc_root is not None:
            try:
                telemetry_payload = json.loads(
                    str(row.get("telemetry_json") or "")
                )
            except json.JSONDecodeError:
                telemetry_payload = {}
            if (
                not isinstance(telemetry_payload, Mapping)
                or Path(
                    str(telemetry_payload.get("backend_proc_root") or "")
                ).resolve()
                != backend_proc_root.resolve()
            ):
                reasons.append(
                    "repeat backend_proc_root telemetry binding is invalid"
                )
        if clients > 1 and not _true(row.get("true_concurrency_observed")):
            reasons.append("repeat did not observe true client concurrency")
        if arm == "sqlens_full" and (
            row.get("d3_measurement_policy") != "workload_driven_adaptive"
            or require_int(
                row.get("d3_namespace_rows_before"),
                "d3_namespace_rows_before",
            )
            != 0
            or not _true(row.get("d3_online_cost_charged"))
        ):
            reasons.append("SQLens repeat lacks fresh online D3 evidence")
    expected_keys = {
        (arm, repeat)
        for arm in MODES_BY_ARM
        for repeat in range(repeat_count)
    }
    if set(groups) != expected_keys:
        reasons.append("repeat arm/repeat groups are incomplete")
    for repeat in range(repeat_count):
        stock = groups.get(("stock_pgvector", repeat))
        sqlens = groups.get(("sqlens_full", repeat))
        if stock and sqlens and (
            stock.get("trace_order_sha256") != sqlens.get("trace_order_sha256")
            or stock.get("request_trace_sha256")
            != sqlens.get("request_trace_sha256")
        ):
            reasons.append(f"paired repeat trace differs in repeat {repeat}")
    return sorted(set(reasons))


def cell_completion_evidence(
    paths: CellPaths,
    pair: matched.SelectedPair,
    clients: int,
    *,
    config: Mapping[str, object],
    config_path: Path,
    normalized_plan: Path,
    normalized_plan_sha: str,
    workload: WorkloadBinding,
    repeats: int | None = None,
    backend_proc_root: Path | None = None,
    pg_prewarm: bool = True,
    client_cpu_list: str = DEFAULT_CLIENT_CPU_LIST,
    backend_cpu_list: str = DEFAULT_BACKEND_CPU_LIST,
) -> dict[str, object]:
    reasons: list[str] = []
    recall_gate: dict[str, object] | None = None
    repeat_count = EXPECTED_REPEATS if repeats is None else repeats
    expected_request_rows = (
        EXPECTED_REQUESTS * repeat_count * len(MODES_BY_ARM)
    )
    expected_repeat_rows = repeat_count * len(MODES_BY_ARM)
    required_paths = (paths.requests, paths.repeats, paths.manifest)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        return {"complete": False, "reasons": ["missing outputs: " + ", ".join(missing)]}
    try:
        manifest = read_json(paths.manifest, "throughput cell manifest")
        request_rows = _rows(paths.requests)
        repeat_rows = _rows(paths.repeats)
        release = config["release_identity"]
        settings = expected_search_settings(pair)
        schedule_seed = int(config["protocol"]["schedule_seed"])
        expected_arm_shas = {
            arm: throughput.arm_config_sha256(settings, arm)
            for arm in MODES_BY_ARM
        }
        release_sha = str(config["release_contract_sha256"])
        run_id = str(manifest.get("run_id") or "")
        dataset = manifest.get("dataset")
        configuration = manifest.get("configuration")
        protocol = manifest.get("protocol")
        inputs = manifest.get("inputs")
        outputs = manifest.get("outputs")
        gates = manifest.get("gates")
        evidence = manifest.get("evidence")
        if (
            manifest.get("artifact_type")
            != "sqlens_figure5_mixed_q10k_throughput_cell"
            or manifest.get("runner_version") != throughput.RUNNER_VERSION
            or manifest.get("artifact_valid") is not True
            or manifest.get("paper_eligible") is not True
            or not run_id
        ):
            reasons.append("core manifest identity/status is invalid")
        expected_dataset_id = throughput.DATASET_IDS[pair.dataset]
        if not isinstance(dataset, Mapping) or (
            dataset.get("key") != pair.dataset
            or dataset.get("dataset_id") != expected_dataset_id
        ):
            reasons.append("core manifest dataset binding is invalid")
        if not isinstance(configuration, Mapping) or (
            configuration.get("pair_id") != pair.pair_id
            or not _close(
                require_float(
                    configuration.get("target_recall"), "target_recall"
                ),
                pair.target_recall,
            )
            or configuration.get("stock_config_sha256")
            != expected_arm_shas["stock_pgvector"]
            or configuration.get("sqlens_config_sha256")
            != expected_arm_shas["sqlens_full"]
        ):
            reasons.append("core manifest independently tuned pair is invalid")
        reasons.extend(
            effective_mode_config_gate(manifest, pair, expected_arm_shas)
        )
        reasons.extend(
            provenance_gate(
                manifest,
                pair=pair,
                config_path=config_path,
                normalized_plan=normalized_plan,
                workload=workload,
            )
        )
        release_manifest = manifest.get("release_contract")
        if not isinstance(release_manifest, Mapping) or (
            release_manifest.get("sha256") != release_sha
            or release_manifest.get("expected_sqlens_build_id")
            != release["expected_sqlens_build_id"]
            or release_manifest.get("expected_vector_so_sha256")
            != release["expected_vector_so_sha256"]
        ):
            reasons.append("core manifest release binding is invalid")
        if not isinstance(protocol, Mapping) or (
            require_int(protocol.get("requests_per_arm_repeat"), "protocol requests")
            != EXPECTED_REQUESTS
            or require_int(protocol.get("unique_queries_per_arm_repeat"), "protocol unique")
            != EXPECTED_REQUESTS
            or require_int(protocol.get("filters"), "protocol filters") != 14
            or require_int(protocol.get("repeats"), "protocol repeats")
            != repeat_count
            or require_int(protocol.get("clients"), "protocol clients") != clients
            or require_int(protocol.get("schedule_seed"), "protocol schedule seed")
            != schedule_seed
            or protocol.get("throughput_source") != throughput.THROUGHPUT_SOURCE
            or protocol.get("throughput_formula")
            != "completed_queries / barrier_wall_clock_seconds"
            or protocol.get("independently_tuned_arms") is not True
            or protocol.get("independent_connection_per_client") is not True
            or protocol.get("pg_prewarm") is not pg_prewarm
            or protocol.get("client_cpu_list") != client_cpu_list
            or protocol.get("backend_cpu_list") != backend_cpu_list
            or protocol.get("client_cpu_assignment")
            != list(
                service_curve_client_cpu_assignment(
                    client_cpu_list, clients
                )
            )
        ):
            reasons.append("core throughput protocol is invalid")
        if not isinstance(gates, Mapping) or not gates or any(
            value is not True for value in gates.values()
        ):
            reasons.append("one or more core completion gates are false")
        if not isinstance(inputs, Mapping):
            reasons.append("core manifest lacks inputs")
        else:
            expected_sources = {
                "orchestrator": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "throughput_core": {
                    "path": str(Path(throughput.__file__).resolve()),
                    "sha256": sha256_file(Path(throughput.__file__).resolve()),
                },
            }
            if inputs.get("execution_sources") != expected_sources:
                reasons.append("throughput execution-source binding is invalid")
            pair_input = inputs.get("measurement_pair")
            config_input = inputs.get("frontier_config")
            workload_input = inputs.get("workload_manifest")
            if not isinstance(pair_input, Mapping) or (
                pair_input.get("source") != "measurement_plan"
                or Path(str(pair_input.get("path") or "")).resolve()
                != normalized_plan.resolve()
                or pair_input.get("sha256") != normalized_plan_sha
                or pair_input.get("pair_id") != pair.pair_id
            ):
                reasons.append("normalized measurement plan binding is invalid")
            if not isinstance(config_input, Mapping) or (
                Path(str(config_input.get("path") or "")).resolve()
                != config_path.resolve()
                or config_input.get("sha256") != sha256_file(config_path)
            ):
                reasons.append("frontier config input binding is invalid")
            if not isinstance(workload_input, Mapping) or (
                Path(str(workload_input.get("path") or "")).resolve()
                != workload.path.resolve()
                or workload_input.get("sha256") != workload.sha256
            ):
                reasons.append("workload manifest input binding is invalid")
        output_specs = {
            "requests": (paths.requests, expected_request_rows),
            "repeats": (paths.repeats, expected_repeat_rows),
        }
        if not isinstance(outputs, Mapping):
            reasons.append("core manifest lacks output SHA bindings")
        else:
            for name, (path, expected_rows) in output_specs.items():
                output = outputs.get(name)
                if not isinstance(output, Mapping) or (
                    Path(str(output.get("path") or "")).resolve() != path.resolve()
                    or require_int(output.get("rows"), f"{name} output rows")
                    != expected_rows
                    or output.get("sha256") != sha256_file(path)
                ):
                    reasons.append(f"{name} output SHA/row binding is invalid")
        if not isinstance(evidence, Mapping) or (
            not matched.identity_matches(
                evidence.get("runtime_binary_identity_start"), release
            )
            or not matched.identity_matches(
                evidence.get("runtime_binary_identity_end"), release
            )
        ):
            reasons.append("runtime r36 binary identity evidence is invalid")
        elif pg_prewarm and not complete_throughput_prewarm(
            evidence.get("prewarm"),
            config["datasets"][pair.dataset],
        ):
            reasons.append("relation pg_prewarm evidence is incomplete")
        elif not pg_prewarm:
            prewarm = evidence.get("prewarm")
            if not isinstance(prewarm, Mapping) or (
                prewarm.get("enabled") is not False
                or prewarm.get("complete") is not True
            ):
                reasons.append("disabled prewarm evidence is invalid")
        if sha256_file(workload.measurement_path) != workload.measurement_sha256:
            reasons.append("frozen measurement workload SHA changed")
        workload_identity = frozen_workload_identity(workload.measurement_path)
        reasons.extend(
            audit_request_rows(
                request_rows,
                pair,
                clients,
                workload_identity=workload_identity,
                run_id=run_id,
                release_sha=release_sha,
                expected_arm_shas=expected_arm_shas,
                schedule_seed=schedule_seed,
                dataset_id=expected_dataset_id,
                config_id=settings.config_id,
                repeats=repeat_count,
            )
        )
        recall_gate = matched_recall_gate(
            request_rows, pair, repeats=repeat_count
        )
        if recall_gate.get("passed") is not True:
            reasons.append(
                "request-level global_min_predicate_lcb recall gate failed: "
                + str(recall_gate.get("reason") or "unknown reason")
            )
        reasons.extend(
            audit_repeat_rows(
                repeat_rows,
                pair,
                clients,
                run_id=run_id,
                release_sha=release_sha,
                expected_arm_shas=expected_arm_shas,
                schedule_seed=schedule_seed,
                dataset_id=expected_dataset_id,
                config_id=settings.config_id,
                repeats=repeat_count,
                backend_proc_root=backend_proc_root,
            )
        )
    except (
        OSError,
        ValueError,
        KeyError,
        csv.Error,
        MatchedThroughputError,
    ) as exc:
        reasons.append(f"completion audit failed: {exc}")
    reasons = sorted(set(reasons))
    result: dict[str, object] = {
        "complete": not reasons,
        "reasons": reasons,
    }
    if recall_gate is not None:
        result["recall_gate"] = recall_gate
    if not reasons:
        result["outputs"] = {
            "requests": {
                "path": str(paths.requests),
                "rows": expected_request_rows,
                "sha256": sha256_file(paths.requests),
            },
            "repeats": {
                "path": str(paths.repeats),
                "rows": expected_repeat_rows,
                "sha256": sha256_file(paths.repeats),
            },
            "manifest": {
                "path": str(paths.manifest),
                "sha256": sha256_file(paths.manifest),
            },
        }
    return result


def build_cell_command(
    *,
    config_path: Path,
    config: Mapping[str, object],
    pair: matched.SelectedPair,
    clients: int,
    workload: WorkloadBinding,
    normalized_plan: Path,
    paths: CellPaths,
    run_id: str,
    repeats: int,
    client_cpu_list: str,
    backend_cpu_list: str,
    backend_proc_root: Path,
    telemetry_devices: str | None,
    telemetry_paths: Sequence[Path],
    pg_prewarm: bool,
    overwrite: bool,
    execute: bool,
) -> list[str]:
    protocol = config["protocol"]
    core_arguments = [
        "--frontier-config",
        str(config_path),
        "--orchestrator-source",
        str(Path(__file__).resolve()),
        "--release-contract",
        str(config["release_contract_path"]),
        "--dataset",
        pair.dataset,
        "--workload-manifest",
        str(workload.path),
        "--measurement-plan",
        str(normalized_plan),
        "--pair-id",
        pair.pair_id,
        "--clients",
        str(clients),
        "--repeats",
        str(repeats),
        "--schedule-seed",
        str(int(protocol["schedule_seed"])),
        "--client-cpu-list",
        client_cpu_list,
        "--backend-cpu-list",
        backend_cpu_list,
        "--backend-proc-root",
        str(backend_proc_root.resolve()),
        "--guidance-max-atoms",
        str(int(protocol["guidance_max_atoms"])),
        "--d2-page-access",
        str(pair.sqlens["d2_page_access"]),
        "--d2-index-page-access",
        str(pair.sqlens["d2_index_page_access"]),
        "--run-id",
        run_id,
        "--out-prefix",
        str(paths.prefix),
        "--pg-prewarm" if pg_prewarm else "--no-pg-prewarm",
    ]
    if (
        clients <= throughput.MAX_CLIENTS
        and repeats >= CORE_FORMAL_MIN_REPEATS
    ):
        command = [
            sys.executable,
            str(Path(throughput.__file__).resolve()),
            *core_arguments,
        ]
    else:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--delegate-core",
            *core_arguments,
        ]
    if telemetry_devices:
        command.extend(["--telemetry-devices", telemetry_devices])
    for path in telemetry_paths:
        command.extend(["--telemetry-path", str(path.resolve())])
    if overwrite:
        command.append("--overwrite")
    if execute:
        command.append("--execute")
    return command


def output_presence(paths: CellPaths) -> list[Path]:
    return [
        path
        for path in (paths.requests, paths.repeats, paths.manifest)
        if path.exists()
    ]


def protocol_fingerprint(
    *,
    protocol_slice: ProtocolSlice,
    config_sha: str,
    selection_bindings: Mapping[str, str],
    normalized_plan_sha: str,
    clients: Sequence[int],
    pairs: Sequence[matched.SelectedPair],
    workloads: Mapping[str, WorkloadBinding],
    cpu: Mapping[str, object],
    backend_proc_root: Path,
    telemetry_devices: str | None,
    telemetry_paths: Sequence[Path],
    pg_prewarm: bool,
) -> str:
    return sha256_json(
        {
            "runner_version": RUNNER_VERSION,
            "protocol_slice": protocol_slice.name,
            "execution_sources": {
                "orchestrator": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "throughput_core": {
                    "path": str(Path(throughput.__file__).resolve()),
                    "sha256": sha256_file(Path(throughput.__file__).resolve()),
                },
            },
            "config_sha256": config_sha,
            "selector": dict(selection_bindings),
            "normalized_plan_sha256": normalized_plan_sha,
            "clients": list(clients),
            "pairs": [
                {
                    "pair_id": pair.pair_id,
                    "dataset": pair.dataset,
                    "target_recall": pair.target_recall,
                    "stock": pair.stock,
                    "sqlens": pair.sqlens,
                }
                for pair in pairs
            ],
            "workloads": {
                dataset: binding.sha256
                for dataset, binding in sorted(workloads.items())
            },
            "cpu": dict(cpu),
            "backend_proc_root": str(backend_proc_root.resolve()),
            "telemetry_devices": telemetry_devices,
            "telemetry_paths": [
                str(path.resolve()) for path in telemetry_paths
            ],
            "pg_prewarm": pg_prewarm,
            "requests": EXPECTED_REQUESTS,
            "repeats": protocol_slice.repeats,
            "throughput_source": throughput.THROUGHPUT_SOURCE,
            "serial_db_cells": True,
        }
    )


def validate_existing_run_manifest(
    path: Path, fingerprint: str, *, resume: bool, overwrite: bool
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    existing = read_json(path, "matched-throughput run manifest")
    if (
        existing.get("status") == "complete"
        or existing.get("paper_eligible") is True
        or existing.get("full_release_complete") is True
    ):
        raise MatchedThroughputError(
            "completed or paper-eligible throughput manifest is immutable; "
            "use a new --out-dir"
        )
    compatible = (
        existing.get("artifact_type")
        == "sqlens_figure5_matched_throughput_run"
        and existing.get("runner_version") == RUNNER_VERSION
        and existing.get("protocol_fingerprint_sha256") == fingerprint
    )
    if not compatible and not overwrite:
        raise MatchedThroughputError(
            f"existing run manifest is incompatible: {path}; use --overwrite"
        )
    if compatible and not resume and not overwrite:
        raise MatchedThroughputError(
            f"run manifest already exists: {path}; use --resume or --overwrite"
        )
    return existing if compatible and not overwrite else None


def _attempt_run_id(pair: matched.SelectedPair, clients: int) -> str:
    return (
        f"f5r36t_{pair.dataset}_{safe_pair_component(pair)}_c{clients}_"
        f"{uuid.uuid4().hex[:12]}"
    )


def run(args: argparse.Namespace) -> int:
    protocol_slice = selected_protocol_slice(args.protocol_slice)
    try:
        config = matched.load_config(args.config.resolve())
    except matched.MatchedLatencyError as exc:
        raise MatchedThroughputError(str(exc)) from exc
    config_path = args.config.resolve()
    config_sha = sha256_file(config_path)
    if args.required_grid_contract is None:
        raise MatchedThroughputError(
            "--required-grid-contract is mandatory for final matched throughput"
        )
    required_grid_contract = args.required_grid_contract.resolve()
    selection_csv = (
        args.selection_csv or protocol_slice.selection_csv
    ).resolve()
    inferred_plan, inferred_manifest = matched.inferred_selection_paths(
        selection_csv
    )
    selection_plan = (args.selection_plan or inferred_plan).resolve()
    selection_manifest = (
        args.selection_manifest or inferred_manifest
    ).resolve()
    try:
        selection_bindings = matched.validate_selection_artifacts(
            selection_csv,
            selection_plan,
            selection_manifest,
            config,
            config_path=config_path,
            required_grid_contract=required_grid_contract,
        )
        selector_pairs = matched.load_selected_pairs(
            selection_csv,
            config,
            datasets=(),
            pair_ids=(),
        )
    except matched.MatchedLatencyError as exc:
        raise MatchedThroughputError(str(exc)) from exc
    all_pairs = protocol_pairs(selector_pairs, protocol_slice)
    pairs = requested_pairs(
        all_pairs,
        datasets=args.datasets,
        pair_ids=args.pair_ids,
    )
    clients = protocol_client_grid(
        args.clients if args.clients is not None else protocol_slice.clients,
        protocol_slice,
    )
    release_scope = full_release_scope(
        args,
        protocol_slice,
        clients,
        pairs,
        all_pairs,
        selection_bindings=selection_bindings,
        enforce_frozen_selector=True,
    )
    cpu = validate_cpu_lists(
        args.client_cpu_list, args.backend_cpu_list, clients
    )
    backend_proc_root = args.backend_proc_root.resolve()
    if not backend_proc_root.is_dir():
        raise MatchedThroughputError(
            f"backend proc root is not a directory: {backend_proc_root}"
        )
    if not args.telemetry_devices and not args.telemetry_path:
        raise MatchedThroughputError(
            "formal throughput requires --telemetry-devices or --telemetry-path"
        )
    stems = [safe_pair_component(pair) for pair in pairs]
    if len(stems) != len(set((pair.dataset, stem) for pair, stem in zip(pairs, stems))):
        raise MatchedThroughputError("selected pair IDs collide after path normalization")

    workload_overrides = workload_manifest_overrides(
        args.workload_manifest
    )
    workloads = {
        dataset: validate_workload_manifest(
            config["datasets"][dataset],
            workload_overrides.get(
                dataset, DEFAULT_WORKLOAD_MANIFESTS[dataset]
            ),
        )
        for dataset in sorted({pair.dataset for pair in pairs})
    }
    out_dir = (args.out_dir or protocol_slice.out_dir).resolve()
    run_manifest_path = (
        out_dir
        / f"figure5_r36_{protocol_slice.name}_throughput_run_manifest.json"
    )
    normalized_plan_path = (
        out_dir
        / f"figure5_r36_{protocol_slice.name}_measurement_plan.json"
    )
    lock = acquire_lock(run_manifest_path.with_suffix(".lock"))
    try:
        # Fail before publishing a normalized plan too: a completed release is
        # an immutable artifact, including its sibling provenance files.
        if run_manifest_path.is_file():
            existing_manifest = read_json(
                run_manifest_path, "matched-throughput run manifest"
            )
            if (
                existing_manifest.get("status") == "complete"
                or existing_manifest.get("paper_eligible") is True
                or existing_manifest.get("full_release_complete") is True
            ):
                raise MatchedThroughputError(
                    "completed or paper-eligible throughput manifest is immutable; "
                    "use a new --out-dir"
                )
        normalized_payload = normalized_measurement_plan(
            pairs,
            config_path=config_path,
            config_sha256=config_sha,
            selection_csv=selection_csv,
            selection_plan=selection_plan,
            selection_manifest=selection_manifest,
            selection_bindings=selection_bindings,
            release={
                "path": config["release_contract_path"],
                "sha256": config["release_contract_sha256"],
                **config["release_identity"],
            },
        )
        normalized_sha = publish_normalized_plan(
            normalized_plan_path,
            normalized_payload,
            overwrite=args.overwrite,
        )
        fingerprint = protocol_fingerprint(
            protocol_slice=protocol_slice,
            config_sha=config_sha,
            selection_bindings=selection_bindings,
            normalized_plan_sha=normalized_sha,
            clients=clients,
            pairs=pairs,
            workloads=workloads,
            cpu=cpu,
            backend_proc_root=backend_proc_root,
            telemetry_devices=args.telemetry_devices,
            telemetry_paths=args.telemetry_path,
            pg_prewarm=args.pg_prewarm,
        )
        existing = validate_existing_run_manifest(
            run_manifest_path,
            fingerprint,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        schedule: list[dict[str, Any]] = []
        for pair in pairs:
            for client_count in clients:
                paths = cell_paths(out_dir, pair, client_count)
                audit = cell_completion_evidence(
                    paths,
                    pair,
                    client_count,
                    config=config,
                    config_path=config_path,
                    normalized_plan=normalized_plan_path,
                    normalized_plan_sha=normalized_sha,
                    workload=workloads[pair.dataset],
                    repeats=protocol_slice.repeats,
                    backend_proc_root=backend_proc_root,
                    pg_prewarm=args.pg_prewarm,
                    client_cpu_list=args.client_cpu_list,
                    backend_cpu_list=args.backend_cpu_list,
                )
                schedule.append(
                    {
                        "cell_id": (
                            f"{pair.dataset}:{pair.pair_id}:clients={client_count}"
                        ),
                        "dataset": pair.dataset,
                        "pair_id": pair.pair_id,
                        "target_recall": pair.target_recall,
                        "clients": client_count,
                        "status": "complete" if audit["complete"] else "pending",
                        "completion_audit": audit,
                        "paths": {
                            "prefix": str(paths.prefix),
                            "requests": str(paths.requests),
                            "repeats": str(paths.repeats),
                            "manifest": str(paths.manifest),
                            "log": str(paths.log),
                        },
                        "inputs": {
                            "workload_manifest": {
                                "path": str(workloads[pair.dataset].path),
                                "sha256": workloads[pair.dataset].sha256,
                            },
                            "measurement_workload": {
                                "path": str(
                                    workloads[pair.dataset].measurement_path
                                ),
                                "sha256": workloads[
                                    pair.dataset
                                ].measurement_sha256,
                                "rows": EXPECTED_REQUESTS,
                            },
                            "normalized_measurement_plan": {
                                "path": str(normalized_plan_path),
                                "sha256": normalized_sha,
                            },
                        },
                    }
                )
        created_at = (
            str(existing.get("created_at"))
            if existing and existing.get("created_at")
            else utc_now()
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "sqlens_figure5_matched_throughput_run",
            "runner_version": RUNNER_VERSION,
            "protocol_slice": protocol_slice.name,
            "status": "planned",
            "artifact_valid": False,
            "paper_eligible": False,
            "requested_slice_complete": False,
            "full_release_complete": False,
            "created_at": created_at,
            "updated_at": utc_now(),
            "protocol_fingerprint_sha256": fingerprint,
            "full_release_scope": release_scope,
            "execution": {
                "database_cells_parallel": False,
                "database_cell_order": "serial",
                "why_not_parallel": SERIAL_DB_REASON,
                "requests_per_arm_repeat": EXPECTED_REQUESTS,
                "repeats": protocol_slice.repeats,
                "expected_request_rows_per_cell": (
                    protocol_slice.expected_request_rows_per_cell
                ),
                "expected_repeat_rows_per_cell": (
                    protocol_slice.expected_repeat_rows_per_cell
                ),
                "expected_predicate_count": EXPECTED_FORMAL_PREDICATES,
                "recall_ci_method": (
                    "query_id_cluster_stratified_predicate_"
                    "percentile_bootstrap_95"
                ),
                "client_grid": list(clients),
                "protocol_client_grid": list(protocol_slice.clients),
                "client_grid_matches_protocol": (
                    clients == protocol_slice.clients
                ),
                "throughput_source": throughput.THROUGHPUT_SOURCE,
                "throughput_formula": (
                    "completed_queries / barrier_wall_clock_seconds"
                ),
                "qps_from_latency_forbidden": True,
                "cpu": cpu,
                "backend_proc_root": str(backend_proc_root),
                "client_cpu_sharing_at_64_clients": (
                    "two pinned client threads per client-partition CPU"
                ),
                "telemetry_devices": args.telemetry_devices,
                "telemetry_paths": [
                    str(path.resolve()) for path in args.telemetry_path
                ],
                "pg_prewarm": args.pg_prewarm,
            },
            "frontier_config": {
                "path": str(config_path),
                "sha256": config_sha,
            },
            "required_grid_contract": dict(
                selection_bindings["required_grid_contract"]
            ),
            "release_contract": {
                "path": config["release_contract_path"],
                "sha256": config["release_contract_sha256"],
                **config["release_identity"],
            },
            "selector": {
                "csv": str(selection_csv),
                "plan": str(selection_plan),
                "manifest": str(selection_manifest),
                **selection_bindings,
            },
            "normalized_measurement_plan": {
                "path": str(normalized_plan_path),
                "sha256": normalized_sha,
                "traversal_guided_burst": TRAVERSAL_GUIDED_BURST,
            },
            "datasets": {
                dataset: {
                    "workload_manifest": str(binding.path),
                    "workload_manifest_sha256": binding.sha256,
                    "workload_manifest_source": (
                        "cli_override"
                        if dataset in workload_overrides
                        else "frozen_q10k_dataset_default"
                    ),
                    "measurement_workload": str(binding.measurement_path),
                    "measurement_workload_sha256": binding.measurement_sha256,
                }
                for dataset, binding in workloads.items()
            },
            "filters": {
                "datasets": list(args.datasets),
                "pair_ids": list(args.pair_ids),
            },
            "schedule": schedule,
            "cells_total": len(schedule),
            "cells_complete": sum(
                cell["status"] == "complete" for cell in schedule
            ),
        }
        atomic_json(run_manifest_path, manifest)
        if not args.execute:
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0

        manifest["status"] = "running"
        atomic_json(run_manifest_path, manifest)
        pair_lookup = {pair.pair_id: pair for pair in pairs}
        for cell in schedule:
            pair = pair_lookup[str(cell["pair_id"])]
            client_count = int(cell["clients"])
            paths = cell_paths(out_dir, pair, client_count)
            if cell["status"] == "complete" and args.resume:
                print(
                    f"resume: complete pair={pair.pair_id} clients={client_count}",
                    flush=True,
                )
                continue
            existing_outputs = output_presence(paths)
            if existing_outputs and not args.overwrite:
                raise MatchedThroughputError(
                    "incomplete throughput output exists; use --overwrite: "
                    + ", ".join(str(path) for path in existing_outputs)
                )
            run_id = _attempt_run_id(pair, client_count)
            command = build_cell_command(
                config_path=config_path,
                config=config,
                pair=pair,
                clients=client_count,
                workload=workloads[pair.dataset],
                normalized_plan=normalized_plan_path,
                paths=paths,
                run_id=run_id,
                repeats=protocol_slice.repeats,
                client_cpu_list=args.client_cpu_list,
                backend_cpu_list=args.backend_cpu_list,
                backend_proc_root=backend_proc_root,
                telemetry_devices=args.telemetry_devices,
                telemetry_paths=args.telemetry_path,
                pg_prewarm=args.pg_prewarm,
                overwrite=args.overwrite,
                execute=True,
            )
            cell.update(
                {
                    "status": "running",
                    "attempt_run_id": run_id,
                    "command": command,
                    "started_at": utc_now(),
                }
            )
            manifest["updated_at"] = utc_now()
            atomic_json(run_manifest_path, manifest)
            paths.log.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"running pair={pair.pair_id} clients={client_count}",
                flush=True,
            )
            with paths.log.open("w", encoding="utf-8") as output:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=os.environ.copy(),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            cell["returncode"] = completed.returncode
            cell["completed_at"] = utc_now()
            audit = cell_completion_evidence(
                paths,
                pair,
                client_count,
                config=config,
                config_path=config_path,
                normalized_plan=normalized_plan_path,
                normalized_plan_sha=normalized_sha,
                workload=workloads[pair.dataset],
                repeats=protocol_slice.repeats,
                backend_proc_root=backend_proc_root,
                pg_prewarm=args.pg_prewarm,
                client_cpu_list=args.client_cpu_list,
                backend_cpu_list=args.backend_cpu_list,
            )
            cell["completion_audit"] = audit
            if completed.returncode != 0 or not audit["complete"]:
                cell["status"] = "failed"
                cell["log_sha256"] = (
                    sha256_file(paths.log) if paths.log.is_file() else None
                )
                manifest["status"] = "failed"
                manifest["updated_at"] = utc_now()
                atomic_json(run_manifest_path, manifest)
                raise MatchedThroughputError(
                    f"throughput cell failed: pair={pair.pair_id} "
                    f"clients={client_count}; see {paths.log}"
                )
            cell["status"] = "complete"
            cell["log_sha256"] = sha256_file(paths.log)
            manifest["cells_complete"] = sum(
                item["status"] == "complete" for item in schedule
            )
            manifest["updated_at"] = utc_now()
            atomic_json(run_manifest_path, manifest)
        manifest["status"] = "complete"
        manifest["artifact_valid"] = True
        manifest["requested_slice_complete"] = True
        manifest["full_release_complete"] = bool(release_scope["requested"])
        manifest["paper_eligible"] = manifest["full_release_complete"]
        if not manifest["paper_eligible"]:
            manifest["paper_eligible_reason"] = (
                "requested slice is complete, but the run does not cover the "
                "full frozen pair/client/CPU/prewarm release protocol"
            )
        manifest["completed_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_json(run_manifest_path, manifest)
        print(f"wrote {run_manifest_path}", flush=True)
        return 0
    finally:
        lock.close()


def positive_client(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if not 1 <= parsed <= SERVICE_CURVE_MAX_CLIENTS:
        raise argparse.ArgumentTypeError(
            f"expected an integer in [1, {SERVICE_CURVE_MAX_CLIENTS}]"
        )
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-slice",
        choices=tuple(PROTOCOL_SLICES),
        default=DEFAULT_PROTOCOL_SLICE,
        help=(
            "Preregistered service slice: all 32 distinct pairs at c16/r3, "
            "the three Recall=0.90 pairs over c1..64/r6, or every selected "
            "Recall=0.90/0.95/0.99 fixed target at c16/r3."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--selection-csv",
        type=Path,
        help="Defaults to the r36 selector for --protocol-slice.",
    )
    parser.add_argument("--selection-plan", type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument(
        "--required-grid-contract",
        type=Path,
        help=(
            "Explicit complete required-grid contract. Final runs fail closed "
            "when it is absent or disagrees with the selector or --config."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=("amazon", "yfcc", "laion"),
        default=[],
    )
    parser.add_argument("--pair-ids", nargs="*", default=[])
    parser.add_argument(
        "--workload-manifest",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help=(
            "Override one explicit frozen q10k workload manifest; may be "
            "repeated for amazon, yfcc, and laion."
        ),
    )
    parser.add_argument(
        "--clients",
        nargs="+",
        type=positive_client,
        help="Must exactly match the preregistered protocol slice.",
    )
    parser.add_argument(
        "--client-cpu-list", default=DEFAULT_CLIENT_CPU_LIST
    )
    parser.add_argument(
        "--backend-cpu-list", default=DEFAULT_BACKEND_CPU_LIST
    )
    parser.add_argument(
        "--backend-proc-root",
        type=Path,
        default=Path("/proc"),
        help=(
            "Procfs root used by the core to measure PostgreSQL backend CPU; "
            "for Docker pass /proc/<container-host-pid>/root/proc."
        ),
    )
    parser.add_argument(
        "--telemetry-devices", default=DEFAULT_TELEMETRY_DEVICES
    )
    parser.add_argument(
        "--telemetry-path", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--pg-prewarm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Defaults to the r36 output directory for --protocol-slice.",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_parser().parse_args(argv))


def delegate_core(argv: Sequence[str]) -> int:
    """Apply only the preregistered wrapper extensions to the unchanged core."""
    try:
        repeats = int(argv[argv.index("--repeats") + 1])
        clients = int(argv[argv.index("--clients") + 1])
    except (ValueError, IndexError):
        print(
            "matched-throughput error: delegated core lacks valid "
            "--repeats/--clients",
            file=sys.stderr,
        )
        return 2
    distinct = PROTOCOL_SLICES[DISTINCT_C16_PROTOCOL]
    if repeats == distinct.repeats and (clients,) == distinct.clients:
        throughput.MIN_REPEATS = repeats
    elif repeats < CORE_FORMAL_MIN_REPEATS:
        print(
            "matched-throughput error: unsupported delegated repeat count "
            f"{repeats}",
            file=sys.stderr,
        )
        return 2
    throughput.MAX_CLIENTS = SERVICE_CURVE_MAX_CLIENTS
    throughput.client_cpu_assignment = service_curve_client_cpu_assignment
    return throughput.main(argv)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--delegate-core":
        raise SystemExit(delegate_core(sys.argv[2:]))
    try:
        raise SystemExit(main())
    except MatchedThroughputError as exc:
        print(f"matched-throughput error: {exc}", file=sys.stderr)
        raise SystemExit(2)
