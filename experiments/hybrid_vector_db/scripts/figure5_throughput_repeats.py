#!/usr/bin/env python3
"""Aggregate audited Figure 5 throughput cells into one converter artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import figure5_converter_binding as converter_binding
    from . import figure5_frontier_artifact as artifact
    from . import pgvector_figure5_throughput as throughput
    from . import run_figure5_matched_throughput as matched_throughput
except ImportError:
    import figure5_converter_binding as converter_binding
    import figure5_frontier_artifact as artifact
    import pgvector_figure5_throughput as throughput
    import run_figure5_matched_throughput as matched_throughput


class ThroughputRepeatError(RuntimeError):
    """A throughput run cannot be converted into formal repeat evidence."""


QPS_BOOTSTRAP_SAMPLES = 2_000
QPS_BOOTSTRAP_SEED = 20_260_728
QPS_BOOTSTRAP_METHOD = "repeat_bootstrap_pooled_qps_percentile_95"


CPU_WEIGHTED_FIELDS = (
    "host_cpu_utilization_pct",
    "host_cpu_user_pct",
    "host_cpu_system_pct",
    "host_cpu_iowait_pct",
)
COUNTER_FIELDS = (
    "host_disk_reads_completed",
    "host_disk_read_bytes",
    "host_disk_read_time_ms",
    "host_disk_writes_completed",
    "host_disk_write_bytes",
    "host_disk_write_time_ms",
    "host_disk_io_time_ms",
    "host_disk_weighted_io_time_ms",
    "pg_database_blks_read",
    "pg_database_blks_hit",
    "pg_database_temp_files",
    "pg_database_temp_bytes",
    "pg_database_blk_read_time_ms",
    "pg_database_blk_write_time_ms",
    "pg_io_reads",
    "pg_io_read_bytes",
    "pg_io_read_time_ms",
    "pg_io_writes",
    "pg_io_write_bytes",
    "pg_io_write_time_ms",
    "pg_io_hits",
    "pg_io_evictions",
    "pg_target_table_heap_blks_read",
    "pg_target_table_heap_blks_hit",
    "pg_target_table_idx_blks_read",
    "pg_target_table_idx_blks_hit",
    "pg_target_index_blks_read",
    "pg_target_index_blks_hit",
    "pg_backend_cpu_user_ms",
    "pg_backend_cpu_system_ms",
    "pg_backend_cpu_total_ms",
)
SERVICE_SUMMARY_FIELDS = (
    "schema_version",
    "protocol_slice",
    "dataset",
    "pair_id",
    "target_recall",
    "arm_id",
    "mode_id",
    "config_sha256",
    "arm_config_sha256",
    "stock_config_sha256",
    "sqlens_config_sha256",
    "clients",
    "repeats",
    "requests_per_repeat",
    "total_requests",
    "completed_queries",
    "error_count",
    "timeout_count",
    "total_barrier_wall_clock_seconds",
    "throughput_qps",
    "throughput_ci95_low",
    "throughput_ci95_high",
    "throughput_ci_method",
    "throughput_source",
    "latency_p95_ms",
    "latency_p95_ci95_low_ms",
    "latency_p95_ci95_high_ms",
    "latency_p99_ms",
    "latency_p99_ci95_low_ms",
    "latency_p99_ci95_high_ms",
    "tail_latency_ci_method",
    "recall_mean",
    "recall_lcb95",
    "recall_ci95_high",
    "recall_min_repeat_lcb95",
    "recall_ci_method",
    "recall_qualification_scope",
    "recall_formal_predicate_sample_floor",
    "recall_predicate_count",
    "recall_worst_predicate_filter",
    "recall_worst_predicate_repeat",
    "recall_gate_sha256",
    "target_lcb95_met",
    "pg_backend_cpu_processes",
    *CPU_WEIGHTED_FIELDS,
    *COUNTER_FIELDS,
    "backend_proc_root",
    "frontier_config_sha256",
    "selection_csv_sha256",
    "selection_plan_sha256",
    "selection_manifest_sha256",
    "normalized_measurement_plan_sha256",
    "protocol_fingerprint_sha256",
    "release_contract_sha256",
    "source_manifest_sha256",
)


def _bound_output(
    cell: Mapping[str, Any],
    *,
    expected_rows: int,
) -> tuple[Path, str, int]:
    completion = cell.get("completion_audit")
    outputs = (
        completion.get("outputs")
        if isinstance(completion, Mapping)
        else None
    )
    repeats = outputs.get("repeats") if isinstance(outputs, Mapping) else None
    if (
        cell.get("status") != "complete"
        or not isinstance(completion, Mapping)
        or completion.get("complete") is not True
        or not isinstance(repeats, Mapping)
    ):
        raise ThroughputRepeatError(
            f"incomplete throughput cell: {cell.get('cell_id')!r}"
        )
    path = Path(str(repeats.get("path") or "")).resolve()
    sha = str(repeats.get("sha256") or "")
    try:
        rows = int(repeats.get("rows"))
    except (TypeError, ValueError) as exc:
        raise ThroughputRepeatError(
            "throughput completion audit has an invalid repeat row count"
        ) from exc
    if (
        not path.is_file()
        or converter_binding.sha256_file(path) != sha
        or rows != expected_rows
    ):
        raise ThroughputRepeatError(
            f"throughput repeat output binding is invalid: {path}"
        )
    cell_paths = cell.get("paths")
    if (
        not isinstance(cell_paths, Mapping)
        or Path(str(cell_paths.get("repeats") or "")).resolve() != path
    ):
        raise ThroughputRepeatError(
            "throughput schedule path differs from completion audit"
        )
    return path, sha, rows


def _raw_repeat_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        required = set(throughput.REPEAT_FIELDS)
        missing = sorted(required - fields)
        if missing:
            raise ThroughputRepeatError(
                f"throughput repeat CSV lacks service evidence fields: {missing}"
            )
        return list(reader)


def _protocol_gate(
    manifest: Mapping[str, Any],
) -> tuple[matched_throughput.ProtocolSlice, int]:
    name = str(manifest.get("protocol_slice") or "")
    try:
        protocol_slice = matched_throughput.PROTOCOL_SLICES[name]
    except KeyError as exc:
        raise ThroughputRepeatError(
            f"unknown formal service protocol slice {name!r}"
        ) from exc
    execution = manifest.get("execution")
    expected_repeat_rows = protocol_slice.expected_repeat_rows_per_cell
    if not isinstance(execution, Mapping) or (
        int(execution.get("requests_per_arm_repeat", -1))
        != matched_throughput.EXPECTED_REQUESTS
        or int(execution.get("repeats", -1)) != protocol_slice.repeats
        or int(execution.get("expected_repeat_rows_per_cell", -1))
        != expected_repeat_rows
        or list(execution.get("client_grid") or [])
        != list(protocol_slice.clients)
        or execution.get("throughput_source")
        != throughput.THROUGHPUT_SOURCE
        or execution.get("throughput_formula")
        != "completed_queries / barrier_wall_clock_seconds"
        or execution.get("qps_from_latency_forbidden") is not True
    ):
        raise ThroughputRepeatError(
            "throughput run manifest does not bind its formal q10k service slice"
        )
    scope = manifest.get("full_release_scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("kind") != name
        or scope.get("requested") is not True
        or list(scope.get("required_clients") or [])
        != list(protocol_slice.clients)
        or int(scope.get("required_repeats", -1))
        != protocol_slice.repeats
    ):
        raise ThroughputRepeatError(
            "throughput run manifest has an invalid full-release scope"
        )
    checks = scope.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ThroughputRepeatError(
            "throughput run manifest did not satisfy every full-release check"
        )
    return protocol_slice, expected_repeat_rows


def _integer(value: object, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ThroughputRepeatError(f"{label} is not an integer") from exc


def _expected_schedule_cells(
    manifest: Mapping[str, Any],
    protocol_slice: matched_throughput.ProtocolSlice,
) -> dict[tuple[str, str, int], float]:
    scope = manifest.get("full_release_scope")
    selector = manifest.get("selector")
    if not isinstance(scope, Mapping) or not isinstance(selector, Mapping):
        raise ThroughputRepeatError(
            "throughput run manifest lacks selector coverage evidence"
        )
    required = scope.get("required_pair_cells")
    required_pair_ids = scope.get("required_pairs")
    requested_pair_ids = scope.get("requested_pairs")
    if not isinstance(required, list) or not required:
        raise ThroughputRepeatError(
            "throughput run manifest lacks required selected-pair identities"
        )
    pair_cells: dict[tuple[str, str], float] = {}
    for item in required:
        if not isinstance(item, Mapping):
            raise ThroughputRepeatError(
                "required selected-pair identity is malformed"
            )
        dataset = str(item.get("dataset") or "")
        pair_id = str(item.get("pair_id") or "")
        try:
            target = float(item.get("target_recall"))
        except (TypeError, ValueError) as exc:
            raise ThroughputRepeatError(
                "required selected-pair target is invalid"
            ) from exc
        if (
            dataset not in throughput.DATASET_IDS
            or not pair_id
            or not math.isfinite(target)
            or not 0.0 <= target <= 1.0
            or (dataset, pair_id) in pair_cells
        ):
            raise ThroughputRepeatError(
                "required selected-pair coverage is invalid"
            )
        pair_cells[(dataset, pair_id)] = target
    pair_ids = sorted(pair_id for _, pair_id in pair_cells)
    if (
        not isinstance(required_pair_ids, list)
        or sorted(str(value) for value in required_pair_ids) != pair_ids
        or not isinstance(requested_pair_ids, list)
        or sorted(str(value) for value in requested_pair_ids) != pair_ids
    ):
        raise ThroughputRepeatError(
            "full-release pair IDs differ from selected-pair identities"
        )

    declared_pairs = selector.get("selected_pairs")
    if declared_pairs is None and protocol_slice.expected_pairs is not None:
        declared_pair_count = protocol_slice.expected_pairs
    else:
        try:
            declared_pair_count = int(declared_pairs)
        except (TypeError, ValueError) as exc:
            raise ThroughputRepeatError(
                "selector selected_pairs is invalid"
            ) from exc
    if declared_pair_count != len(pair_cells):
        raise ThroughputRepeatError(
            "selector selected_pairs differs from required pair coverage"
        )
    if (
        protocol_slice.expected_pairs is not None
        and declared_pair_count != protocol_slice.expected_pairs
    ):
        raise ThroughputRepeatError(
            "selector selected_pairs differs from the static protocol"
        )
    if protocol_slice.fixed_targets:
        try:
            target_rows = int(selector.get("target_rows"))
            unattainable = int(selector.get("unattainable_pairs"))
        except (TypeError, ValueError) as exc:
            raise ThroughputRepeatError(
                "fixed-target selector coverage counts are invalid"
            ) from exc
        targets_by_dataset = selector.get("targets_by_dataset")
        expected_targets = list(protocol_slice.fixed_targets)
        selected_dataset_targets = {
            (dataset, target) for (dataset, _), target in pair_cells.items()
        }
        if (
            target_rows
            != len(matched_throughput.matched.FROZEN_DATASETS)
            * len(protocol_slice.fixed_targets)
            or declared_pair_count + unattainable != target_rows
            or selector.get("qualification_scope")
            != matched_throughput.matched.QUALIFICATION_SCOPE_FORMAL
            or not isinstance(targets_by_dataset, Mapping)
            or set(targets_by_dataset)
            != set(matched_throughput.matched.FROZEN_DATASETS)
            or any(
                list(targets_by_dataset.get(dataset) or [])
                != expected_targets
                for dataset in matched_throughput.matched.FROZEN_DATASETS
            )
            or len(selected_dataset_targets) != len(pair_cells)
            or any(
                not any(
                    matched_throughput._close(target, expected)
                    for expected in expected_targets
                )
                for _, target in selected_dataset_targets
            )
        ):
            raise ThroughputRepeatError(
                "fixed-target selector coverage is incomplete"
            )
    return {
        (dataset, pair_id, clients): target
        for (dataset, pair_id), target in pair_cells.items()
        for clients in protocol_slice.clients
    }


def _formal_recall_evidence(
    cell: Mapping[str, Any],
    *,
    target: float,
    repeats: int,
) -> dict[str, dict[str, Any]]:
    completion = cell.get("completion_audit")
    gate = (
        completion.get("recall_gate")
        if isinstance(completion, Mapping)
        else None
    )
    if not isinstance(gate, Mapping):
        raise ThroughputRepeatError(
            "throughput cell lacks request-level formal recall evidence"
        )
    sample_floor = matched_throughput.matched.MIN_FORMAL_PREDICATE_SAMPLES
    if (
        gate.get("qualification_scope")
        != matched_throughput.matched.QUALIFICATION_SCOPE_FORMAL
        or gate.get("passed") is not True
        or gate.get("paper_eligible") is not True
        or _integer(
            gate.get("formal_predicate_sample_floor"),
            "formal predicate sample floor",
        )
        != sample_floor
        or _integer(
            gate.get("expected_predicate_count"),
            "expected predicate count",
        )
        != matched_throughput.EXPECTED_FORMAL_PREDICATES
        or _integer(
            gate.get("observed_predicate_count"),
            "observed predicate count",
        )
        != matched_throughput.EXPECTED_FORMAL_PREDICATES
    ):
        raise ThroughputRepeatError(
            "throughput cell formal recall gate is invalid"
        )
    filter_names = gate.get("filter_names")
    aggregate = gate.get("aggregate")
    per_predicate = gate.get("per_predicate")
    if (
        not isinstance(filter_names, list)
        or len(filter_names) != matched_throughput.EXPECTED_FORMAL_PREDICATES
        or len(set(str(value) for value in filter_names))
        != matched_throughput.EXPECTED_FORMAL_PREDICATES
        or not isinstance(aggregate, Mapping)
        or not isinstance(per_predicate, Mapping)
    ):
        raise ThroughputRepeatError(
            "throughput cell predicate recall coverage is invalid"
        )

    result: dict[str, dict[str, Any]] = {}
    for arm in matched_throughput.MODES_BY_ARM:
        worst: dict[str, Any] | None = None
        repeat_aggregate: dict[int, dict[str, float]] = {}
        for repeat in range(repeats):
            arm_key = f"{arm}/repeat={repeat}"
            aggregate_stats = aggregate.get(arm_key)
            predicate_stats = per_predicate.get(arm_key)
            if (
                not isinstance(aggregate_stats, Mapping)
                or _integer(
                    aggregate_stats.get("sample_count"),
                    "aggregate recall sample count",
                )
                != matched_throughput.EXPECTED_REQUESTS
                or aggregate_stats.get("passed") is not True
                or _number(aggregate_stats.get("lower"), "aggregate recall LCB")
                < target
                or not isinstance(predicate_stats, Mapping)
                or set(str(key) for key in predicate_stats)
                != set(str(value) for value in filter_names)
            ):
                raise ThroughputRepeatError(
                    f"throughput cell aggregate/predicate gate is invalid for {arm_key}"
                )
            repeat_aggregate[repeat] = {
                "mean": _number(
                    aggregate_stats.get("mean"),
                    "aggregate recall mean",
                ),
                "lower": _number(
                    aggregate_stats.get("lower"),
                    "aggregate recall LCB",
                ),
                "upper": _number(
                    aggregate_stats.get("upper"),
                    "aggregate recall UCB",
                ),
            }
            for filter_name in filter_names:
                stats = predicate_stats.get(filter_name)
                if not isinstance(stats, Mapping):
                    raise ThroughputRepeatError(
                        f"throughput predicate evidence is missing for {arm_key}"
                    )
                lower = _number(stats.get("lower"), "predicate recall LCB")
                count = _integer(
                    stats.get("sample_count"),
                    "predicate recall sample count",
                )
                if (
                    count < sample_floor
                    or stats.get("sample_count_sufficient") is not True
                    or stats.get("passed") is not True
                    or lower < target
                ):
                    raise ThroughputRepeatError(
                        f"throughput predicate recall gate misses target for {arm_key}"
                    )
                candidate = {
                    "filter_name": str(filter_name),
                    "repeat": repeat,
                    "lower": lower,
                }
                if worst is None or (
                    candidate["lower"],
                    candidate["repeat"],
                    candidate["filter_name"],
                ) < (
                    worst["lower"],
                    worst["repeat"],
                    worst["filter_name"],
                ):
                    worst = candidate
            if (
                sum(
                    _integer(
                        stats["sample_count"],
                        "predicate recall sample count",
                    )
                    for stats in predicate_stats.values()
                    if isinstance(stats, Mapping)
                )
                != matched_throughput.EXPECTED_REQUESTS
            ):
                raise ThroughputRepeatError(
                    f"throughput predicate sample counts differ from {arm_key} coverage"
                )
        if worst is None:
            raise ThroughputRepeatError(
                f"throughput cell has no predicate evidence for {arm}"
            )
        result[arm] = {
            "qualification_scope": (
                matched_throughput.matched.QUALIFICATION_SCOPE_FORMAL
            ),
            "sample_floor": sample_floor,
            "predicate_count": len(filter_names),
            "worst_filter_name": worst["filter_name"],
            "worst_repeat": worst["repeat"],
            "min_predicate_lcb95": worst["lower"],
            "gate_sha256": matched_throughput.sha256_json(gate),
            "repeat_aggregate": repeat_aggregate,
        }
    return result


def convert_manifest(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    manifest, release, source_sha = converter_binding.audited_run_manifest(
        manifest_path,
        expected_artifact_type="sqlens_figure5_matched_throughput_run",
    )
    protocol_slice, expected_repeat_rows = _protocol_gate(manifest)
    expected_cells = _expected_schedule_cells(manifest, protocol_slice)
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ThroughputRepeatError("throughput run manifest has no schedule")
    if (
        int(manifest.get("cells_total", -1)) != len(schedule)
        or int(manifest.get("cells_complete", -1)) != len(schedule)
        or len(schedule) != len(expected_cells)
    ):
        raise ThroughputRepeatError(
            "throughput run manifest cell coverage is incomplete"
        )

    rows: list[dict[str, Any]] = []
    seen_inputs: set[Path] = set()
    seen_cells: set[tuple[str, str, int]] = set()
    for cell in schedule:
        if not isinstance(cell, Mapping):
            raise ThroughputRepeatError(
                "throughput run manifest contains a malformed cell"
            )
        dataset_key = str(cell.get("dataset") or "")
        pair_id = str(cell.get("pair_id") or "")
        try:
            clients = int(cell.get("clients"))
        except (TypeError, ValueError) as exc:
            raise ThroughputRepeatError(
                "throughput cell has invalid clients"
            ) from exc
        cell_key = (dataset_key, pair_id, clients)
        if cell_key in seen_cells:
            raise ThroughputRepeatError(
                f"throughput schedule repeats cell {cell_key!r}"
            )
        seen_cells.add(cell_key)
        if dataset_key not in throughput.DATASET_IDS:
            raise ThroughputRepeatError(
                f"unknown throughput dataset {dataset_key!r}"
            )
        path, _, expected_rows = _bound_output(
            cell, expected_rows=expected_repeat_rows
        )
        if path in seen_inputs:
            raise ThroughputRepeatError(
                f"throughput repeat input is reused: {path}"
            )
        seen_inputs.add(path)
        input_rows, _ = artifact.read_repeat_csv(path, "throughput")
        raw_rows = _raw_repeat_rows(path)
        if len(input_rows) != expected_rows:
            raise ThroughputRepeatError(
                f"throughput repeat row count drifted: {path}"
            )
        expected_dataset = throughput.DATASET_IDS[dataset_key]
        target = _number(cell.get("target_recall"), "cell target_recall")
        if (
            cell_key not in expected_cells
            or not matched_throughput._close(
                target, expected_cells[cell_key]
            )
        ):
            raise ThroughputRepeatError(
                f"throughput cell differs from selected target: {cell_key!r}"
            )
        recall_evidence = _formal_recall_evidence(
            cell,
            target=target,
            repeats=protocol_slice.repeats,
        )
        if len(raw_rows) != len(input_rows):
            raise ThroughputRepeatError(
                f"raw/canonical repeat row count differs: {path}"
            )
        for row, raw in zip(input_rows, raw_rows):
            if (
                row["dataset"] != expected_dataset
                or row["experiment_kind"] != "throughput"
                or row["config_id"] != pair_id
                or int(row["clients"]) != clients
                or row["release_identity_sha256"] != release["sha256"]
            ):
                raise ThroughputRepeatError(
                    f"throughput repeat identity differs from {cell_key!r}"
                )
            row["pair_id"] = pair_id
            row["target_recall"] = target
            arm_evidence = recall_evidence[str(row["arm_id"])]
            repeat_stats = arm_evidence["repeat_aggregate"][
                int(row["repeat_id"])
            ]
            row["recall_mean"] = repeat_stats["mean"]
            row["recall_ci95_low"] = repeat_stats["lower"]
            row["recall_ci95_high"] = repeat_stats["upper"]
            row["formal_recall_qualification_scope"] = arm_evidence[
                "qualification_scope"
            ]
            row["formal_recall_sample_floor"] = arm_evidence["sample_floor"]
            row["formal_recall_predicate_count"] = arm_evidence[
                "predicate_count"
            ]
            row["formal_recall_worst_filter"] = arm_evidence[
                "worst_filter_name"
            ]
            row["formal_recall_worst_repeat"] = arm_evidence["worst_repeat"]
            row["formal_recall_min_predicate_lcb95"] = arm_evidence[
                "min_predicate_lcb95"
            ]
            row["formal_recall_gate_sha256"] = arm_evidence["gate_sha256"]
            for field in throughput.REPEAT_EVIDENCE_FIELDS:
                row[field] = raw.get(field, "")
            for field in (
                "arm_config_sha256",
                "stock_config_sha256",
                "sqlens_config_sha256",
            ):
                row[field] = raw.get(field, "")
        rows.extend(input_rows)

    expected_cell_keys = set(expected_cells)
    if seen_cells != expected_cell_keys:
        missing = sorted(expected_cell_keys - seen_cells)
        extra = sorted(seen_cells - expected_cell_keys)
        raise ThroughputRepeatError(
            "throughput schedule differs from selected-pair coverage: "
            f"missing={missing}, extra={extra}"
        )

    unique_rows = {
        (
            row["dataset"],
            row["arm_id"],
            row["config_id"],
            int(row["clients"]),
            int(row["repeat_id"]),
        )
        for row in rows
    }
    if len(unique_rows) != len(rows):
        raise ThroughputRepeatError(
            "throughput aggregate contains duplicate repeat rows"
        )
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["config_id"],
            int(row["clients"]),
            int(row["repeat_id"]),
            row["arm_id"],
        )
    )
    return rows, release, source_sha


def _number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ThroughputRepeatError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ThroughputRepeatError(f"{label} is not finite")
    return number


def _single(values: Sequence[object], label: str) -> object:
    unique = {str(value) for value in values}
    if len(unique) != 1:
        raise ThroughputRepeatError(f"{label} is not constant: {unique}")
    return values[0]


def _t_critical_95(count: int) -> float:
    values = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
    }
    return values.get(count, 1.96)


def _mean_ci95(
    values: Sequence[float],
    *,
    center: float | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float, float]:
    if not values:
        raise ThroughputRepeatError("cannot summarize an empty metric")
    observed = statistics.fmean(values) if center is None else center
    if len(values) == 1:
        low = high = observed
    else:
        half_width = (
            _t_critical_95(len(values))
            * statistics.stdev(values)
            / math.sqrt(len(values))
        )
        low, high = observed - half_width, observed + half_width
    if lower_bound is not None:
        low = max(lower_bound, low)
    if upper_bound is not None:
        high = min(upper_bound, high)
    return observed, low, high


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ThroughputRepeatError("cannot summarize an empty bootstrap")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def pooled_qps_bootstrap(
    completed_queries: Sequence[int],
    wall_clock_seconds: Sequence[float],
    *,
    samples: int = QPS_BOOTSTRAP_SAMPLES,
    seed: int = QPS_BOOTSTRAP_SEED,
    seed_label: str = "",
) -> tuple[float, float, float]:
    """Bootstrap repeat clusters using the pooled-QPS statistic throughout."""
    if (
        len(completed_queries) != len(wall_clock_seconds)
        or not completed_queries
        or samples < 100
    ):
        raise ThroughputRepeatError("pooled-QPS bootstrap inputs are invalid")
    if any(value < 0 for value in completed_queries) or any(
        not math.isfinite(value) or value <= 0.0
        for value in wall_clock_seconds
    ):
        raise ThroughputRepeatError(
            "pooled-QPS bootstrap requires nonnegative counts and positive walls"
        )
    center = sum(completed_queries) / sum(wall_clock_seconds)
    if len(completed_queries) == 1:
        return center, center, center
    digest = hashlib.sha256(seed_label.encode("utf-8")).digest()
    rng = random.Random(seed ^ int.from_bytes(digest[:8], "big"))
    count = len(completed_queries)
    draws: list[float] = []
    for _ in range(samples):
        indexes = [rng.randrange(count) for _ in range(count)]
        draws.append(
            sum(completed_queries[index] for index in indexes)
            / sum(wall_clock_seconds[index] for index in indexes)
        )
    return center, _percentile(draws, 0.025), _percentile(draws, 0.975)


def _configuration_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    selector = manifest.get("selector")
    frontier = manifest.get("frontier_config")
    normalized = manifest.get("normalized_measurement_plan")
    if not all(
        isinstance(value, Mapping)
        for value in (selector, frontier, normalized)
    ):
        raise ThroughputRepeatError(
            "service manifest lacks configuration hash bindings"
        )
    hashes = {
        "frontier_config_sha256": str(frontier.get("sha256") or ""),
        "selection_csv_sha256": str(
            selector.get("selection_csv_sha256") or ""
        ),
        "selection_plan_sha256": str(
            selector.get("selection_plan_sha256") or ""
        ),
        "selection_manifest_sha256": str(
            selector.get("selection_manifest_sha256") or ""
        ),
        "normalized_measurement_plan_sha256": str(
            normalized.get("sha256") or ""
        ),
        "protocol_fingerprint_sha256": str(
            manifest.get("protocol_fingerprint_sha256") or ""
        ),
        "release_contract_sha256": str(
            (manifest.get("release_contract") or {}).get("sha256") or ""
        ),
    }
    invalid = [
        name
        for name, value in hashes.items()
        if not throughput.SHA256_RE.fullmatch(value)
    ]
    if invalid:
        raise ThroughputRepeatError(
            f"service manifest has invalid configuration hashes: {invalid}"
        )
    return hashes


def aggregate_service_rows(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
) -> list[dict[str, Any]]:
    protocol_slice, _ = _protocol_gate(manifest)
    hashes = _configuration_hashes(manifest)
    execution = manifest["execution"]
    backend_proc_root = str(execution.get("backend_proc_root") or "")
    if not backend_proc_root:
        raise ThroughputRepeatError(
            "service manifest does not bind backend_proc_root"
        )
    groups: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        groups[
            (
                str(row["dataset"]),
                str(row["pair_id"]),
                str(row["arm_id"]),
                int(row["clients"]),
            )
        ].append(row)

    schedule = manifest.get("schedule")
    if not isinstance(schedule, list):
        raise ThroughputRepeatError("service manifest has no schedule")
    expected_groups = len(schedule) * len(
        matched_throughput.MODES_BY_ARM
    )
    if len(groups) != expected_groups:
        raise ThroughputRepeatError(
            f"service aggregate has {len(groups)} arm/cell groups, "
            f"expected {expected_groups}"
        )

    result: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        dataset, pair_id, arm_id, clients = key
        group = sorted(group, key=lambda row: int(row["repeat_id"]))
        if (
            len(group) != protocol_slice.repeats
            or {int(row["repeat_id"]) for row in group}
            != set(range(protocol_slice.repeats))
        ):
            raise ThroughputRepeatError(
                f"service repeat coverage is incomplete for {key!r}"
            )
        if any(
            str(row.get("telemetry_collected") or "").lower()
            not in {"true", "1", "t", "yes"}
            for row in group
        ):
            raise ThroughputRepeatError(
                f"service telemetry is incomplete for {key!r}"
            )
        for row in group:
            try:
                telemetry_payload = json.loads(
                    str(row.get("telemetry_json") or "")
                )
            except json.JSONDecodeError as exc:
                raise ThroughputRepeatError(
                    f"service telemetry JSON is invalid for {key!r}"
                ) from exc
            if (
                not isinstance(telemetry_payload, Mapping)
                or str(telemetry_payload.get("backend_proc_root") or "")
                != backend_proc_root
            ):
                raise ThroughputRepeatError(
                    f"service backend_proc_root evidence differs for {key!r}"
                )
        walls = [
            _number(row["wall_clock_seconds"], "wall_clock_seconds")
            for row in group
        ]
        completed_by_repeat = [
            int(row["completed_queries"]) for row in group
        ]
        completed = sum(completed_by_repeat)
        total_wall = sum(walls)
        pooled_qps, qps_low, qps_high = pooled_qps_bootstrap(
            completed_by_repeat,
            walls,
            seed_label=f"{dataset}:{pair_id}:{arm_id}:clients={clients}",
        )
        p95, p95_low, p95_high = _mean_ci95(
            [
                _number(row["latency_p95_ms"], "latency_p95_ms")
                for row in group
            ],
            lower_bound=0.0,
        )
        p99, p99_low, p99_high = _mean_ci95(
            [
                _number(row["latency_p99_ms"], "latency_p99_ms")
                for row in group
            ],
            lower_bound=0.0,
        )
        recall, _, _ = _mean_ci95(
            [
                _number(row["recall_mean"], "recall_mean")
                for row in group
            ],
            lower_bound=0.0,
            upper_bound=1.0,
        )
        target = _number(
            _single(
                [row["target_recall"] for row in group],
                "target_recall",
            ),
            "target_recall",
        )
        repeat_lcb = _number(
            _single(
                [
                    row["formal_recall_min_predicate_lcb95"]
                    for row in group
                ],
                "formal_recall_min_predicate_lcb95",
            ),
            "formal_recall_min_predicate_lcb95",
        )
        repeat_ucb = max(
            _number(row["recall_ci95_high"], "recall_ci95_high")
            for row in group
        )
        errors = sum(int(row["error_count"]) for row in group)
        if errors:
            raise ThroughputRepeatError(
                f"formal service group contains errors: {key!r}"
            )
        cell_hashes = {
            field: str(_single(
                [row[field] for row in group], field
            ))
            for field in (
                "config_sha256",
                "arm_config_sha256",
                "stock_config_sha256",
                "sqlens_config_sha256",
            )
        }
        invalid_hashes = [
            field
            for field, value in cell_hashes.items()
            if not throughput.SHA256_RE.fullmatch(value)
        ]
        if invalid_hashes:
            raise ThroughputRepeatError(
                f"service group has invalid config hashes: {invalid_hashes}"
            )
        summary: dict[str, Any] = {
            "schema_version": 1,
            "protocol_slice": protocol_slice.name,
            "dataset": dataset,
            "pair_id": pair_id,
            "target_recall": target,
            "arm_id": arm_id,
            "mode_id": _single(
                [row["mode_id"] for row in group], "mode_id"
            ),
            **cell_hashes,
            "clients": clients,
            "repeats": protocol_slice.repeats,
            "requests_per_repeat": matched_throughput.EXPECTED_REQUESTS,
            "total_requests": sum(int(row["requests"]) for row in group),
            "completed_queries": completed,
            "error_count": errors,
            "timeout_count": 0,
            "total_barrier_wall_clock_seconds": total_wall,
            "throughput_qps": pooled_qps,
            "throughput_ci95_low": qps_low,
            "throughput_ci95_high": qps_high,
            "throughput_ci_method": QPS_BOOTSTRAP_METHOD,
            "throughput_source": throughput.THROUGHPUT_SOURCE,
            "latency_p95_ms": p95,
            "latency_p95_ci95_low_ms": p95_low,
            "latency_p95_ci95_high_ms": p95_high,
            "latency_p99_ms": p99,
            "latency_p99_ci95_low_ms": p99_low,
            "latency_p99_ci95_high_ms": p99_high,
            "tail_latency_ci_method": "t95_over_repeat_percentiles",
            "recall_mean": recall,
            "recall_lcb95": repeat_lcb,
            "recall_ci95_high": repeat_ucb,
            "recall_min_repeat_lcb95": repeat_lcb,
            "recall_ci_method": (
                "global_min_predicate_lcb"
            ),
            "recall_qualification_scope": _single(
                [
                    row["formal_recall_qualification_scope"]
                    for row in group
                ],
                "formal_recall_qualification_scope",
            ),
            "recall_formal_predicate_sample_floor": int(_single(
                [row["formal_recall_sample_floor"] for row in group],
                "formal_recall_sample_floor",
            )),
            "recall_predicate_count": int(_single(
                [row["formal_recall_predicate_count"] for row in group],
                "formal_recall_predicate_count",
            )),
            "recall_worst_predicate_filter": _single(
                [row["formal_recall_worst_filter"] for row in group],
                "formal_recall_worst_filter",
            ),
            "recall_worst_predicate_repeat": int(_single(
                [row["formal_recall_worst_repeat"] for row in group],
                "formal_recall_worst_repeat",
            )),
            "recall_gate_sha256": _single(
                [row["formal_recall_gate_sha256"] for row in group],
                "formal_recall_gate_sha256",
            ),
            "target_lcb95_met": repeat_lcb >= target,
            "pg_backend_cpu_processes": max(
                int(_number(
                    row["pg_backend_cpu_processes"],
                    "pg_backend_cpu_processes",
                ))
                for row in group
            ),
            "backend_proc_root": backend_proc_root,
            **hashes,
            "source_manifest_sha256": source_manifest_sha256,
        }
        for field in CPU_WEIGHTED_FIELDS:
            summary[field] = sum(
                _number(row[field], field) * wall
                for row, wall in zip(group, walls)
            ) / total_wall
        for field in COUNTER_FIELDS:
            summary[field] = sum(
                _number(row[field], field) for row in group
            )
        if summary["target_lcb95_met"] is not True:
            raise ThroughputRepeatError(
                f"service aggregate misses matched recall target: {key!r}"
            )
        result.append(summary)
    return result


def write_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                *artifact.REPEAT_FIELDS,
                *sorted(artifact.OPTIONAL_PROVENANCE_FIELDS),
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_service_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=SERVICE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def bound_source_payload(
    path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    if converter_binding.sha256_file(path) != expected_sha256:
        raise ThroughputRepeatError(
            "audited service manifest changed during aggregation"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ThroughputRepeatError(
            "audited service manifest must be a JSON object"
        )
    return payload


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--service-summary",
        type=Path,
        help=(
            "Formal service aggregate; defaults to "
            "<out-without-suffix>.service.csv."
        ),
    )
    parser.add_argument(
        "--binding-manifest",
        type=Path,
        help="Converter sidecar path; defaults to <out>.manifest.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        source_manifest = args.run_manifest.resolve()
        output = args.out.resolve()
        rows, release, source_sha = convert_manifest(source_manifest)
        source_payload = bound_source_payload(
            source_manifest, source_sha
        )
        service_rows = aggregate_service_rows(
            rows,
            source_payload,
            source_manifest_sha256=source_sha,
        )
        provenance = converter_binding.row_provenance(
            release, source_manifest, source_sha
        )
        for row in rows:
            row.update(provenance)
        write_rows(output, rows)
        service_output = (
            args.service_summary.resolve()
            if args.service_summary
            else output.with_name(output.stem + ".service.csv")
        )
        write_service_rows(service_output, service_rows)
        binding_path = (
            args.binding_manifest.resolve()
            if args.binding_manifest
            else output.with_suffix(output.suffix + ".manifest.json")
        )
        binding = converter_binding.publish_converter_binding(
            kind="throughput",
            source_manifest=source_manifest,
            source_sha256=source_sha,
            release=release,
            output=output,
            rows=len(rows),
            converter_source=Path(__file__),
            binding_path=binding_path,
        )
        bound_source_payload(source_manifest, source_sha)
        binding.update(
            {
                "protocol_slice": source_payload["protocol_slice"],
                "configuration_hashes": _configuration_hashes(
                    source_payload
                ),
                "service_aggregate": {
                    "path": str(service_output),
                    "sha256": converter_binding.sha256_file(service_output),
                    "rows": len(service_rows),
                    "qps_source": throughput.THROUGHPUT_SOURCE,
                    "qps_from_latency_forbidden": True,
                    "fields": list(SERVICE_SUMMARY_FIELDS),
                },
            }
        )
        converter_binding.atomic_json(binding_path, binding)
    except (
        converter_binding.ConverterBindingError,
        ThroughputRepeatError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        f"wrote {output} rows={len(rows)} "
        f"service={service_output} service_rows={len(service_rows)} "
        f"binding={binding_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
