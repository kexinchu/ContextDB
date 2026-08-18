#!/usr/bin/env python3
"""Audit and summarize expedited per-filter matched-recall measurements.

This utility is intentionally separate from the formal Table 6 builder.  The
current expedited campaign uses one balanced q5K pass, while the registered
paper protocol requires q10K/r3 latency and independently measured service
throughput.  The output is therefore useful for an honest interim table, but
is always marked ``paper_eligible=false``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
STOCK_MODE = "original"
SQLENS_MODE = "design1_bloom_bfs_layout_d3"
MODES = (STOCK_MODE, SQLENS_MODE)
EXPECTED_FILTERS = 14
EXPECTED_REQUESTS = 5_000
SCHEMA_VERSION = 1


class AuditError(RuntimeError):
    """The expedited artifact cannot be summarized without ambiguity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AuditError("cannot summarize an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc
    if not fields:
        raise AuditError(f"CSV has no header: {path}")
    if any(None in row for row in rows):
        raise AuditError(f"CSV has a row wider than its header: {path}")
    return fields, rows


def require_fields(fields: Sequence[str], required: Sequence[str], path: Path) -> None:
    missing = sorted(set(required) - set(fields))
    if missing:
        raise AuditError(f"{path} is missing fields {missing}")


def canonical_request_trace(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_requests: int,
    label: str,
) -> tuple[str, tuple[tuple[object, ...], ...]]:
    """Hash logical requests independent of CSV line endings or execution order."""
    identities: dict[tuple[int, int], tuple[object, ...]] = {}
    for row in rows:
        key = (int(row["request_no"]), int(row["trace_cycle"]))
        identity = (
            key[0], key[1], row["filter_name"], int(row["query_no"]),
            int(row["query_id"]),
        )
        previous = identities.setdefault(key, identity)
        if previous != identity:
            raise AuditError(f"{label} disagrees on logical request {key}")
    if len(identities) != expected_requests:
        raise AuditError(
            f"{label} covers {len(identities)} logical requests, expected "
            f"{expected_requests}"
        )
    ordered = tuple(identities[key] for key in sorted(identities))
    return sha256_json(ordered), ordered


def atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise AuditError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: Path, dataset: str) -> dict[str, object]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(root, dict) or not isinstance(root.get(dataset), dict):
        raise AuditError(f"config does not define dataset {dataset!r}")
    value = dict(root[dataset])
    filters = value.get("filters")
    if not isinstance(filters, list) or len(filters) != EXPECTED_FILTERS:
        raise AuditError(f"{dataset} must declare exactly {EXPECTED_FILTERS} filters")
    if len(set(str(item) for item in filters)) != EXPECTED_FILTERS:
        raise AuditError(f"{dataset} filter list contains duplicates")
    replacements = value.get("replacements", {})
    if not isinstance(replacements, dict):
        raise AuditError(f"{dataset} replacements must be an object")
    unknown = set(replacements) - set(filters)
    if unknown:
        raise AuditError(f"{dataset} replacements contain unknown filters {sorted(unknown)}")
    return value


def resolve(path: object) -> Path:
    value = Path(str(path))
    return value if value.is_absolute() else ROOT / value


def source_for(config: Mapping[str, object], filter_name: str) -> Path:
    replacements = config.get("replacements", {})
    assert isinstance(replacements, Mapping)
    if filter_name in replacements:
        return resolve(replacements[filter_name])
    return resolve(config["base_dir"]) / f"{config['file_prefix']}{filter_name}.csv"


def audit_filter(
    path: Path,
    filter_name: str,
    *,
    expected_build_id: str,
    expected_vector_sha: str,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    fields, rows = read_csv(path)
    require_fields(
        fields,
        (
            "filter_name", "mode", "query_no", "query_id", "repeat",
            "pair_key", "recall", "end_to_end_ms", "effective_ef_search",
            "iterative_scan", "effective_iterative_scan",
            "backend_cpu_exact_match", "error",
        ),
        path,
    )
    if not rows:
        raise AuditError(f"empty measurement shard: {path}")
    if {row["filter_name"] for row in rows} != {filter_name}:
        raise AuditError(f"{path} contains a different filter")
    if {row["mode"] for row in rows} != set(MODES):
        raise AuditError(f"{path} does not contain exactly Stock and SQLens")
    if any(row.get("error", "").strip() for row in rows):
        raise AuditError(f"{path} contains query errors")
    if any(row.get("backend_cpu_exact_match", "").lower() != "true" for row in rows):
        raise AuditError(f"{path} contains a backend outside its CPU partition")

    plan_path = Path(str(path) + ".plan.json")
    if not plan_path.exists():
        raise AuditError(f"missing runner plan: {plan_path}")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read runner plan {plan_path}: {exc}") from exc
    encoded = json.dumps(plan, sort_keys=True)
    if expected_build_id not in encoded or expected_vector_sha not in encoded:
        raise AuditError(f"{plan_path} is not bound to the expected r41 binary")

    pairs: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["query_no"], row["repeat"])
        mode = row["mode"]
        if mode in pairs[key]:
            raise AuditError(f"duplicate {filter_name}/{key}/{mode} in {path}")
        latency = float(row["end_to_end_ms"])
        recall = float(row["recall"])
        if not math.isfinite(latency) or latency <= 0.0:
            raise AuditError(f"invalid latency in {path}")
        if not math.isfinite(recall) or not 0.0 <= recall <= 1.0:
            raise AuditError(f"invalid recall in {path}")
        pairs[key][mode] = row
    if any(set(arms) != set(MODES) for arms in pairs.values()):
        raise AuditError(f"{path} is not strictly paired")
    repeats = {repeat for _query, repeat in pairs}
    if repeats != {"0"}:
        raise AuditError(f"{path} is not the registered expedited r1 protocol")

    settings: dict[str, dict[str, object]] = {}
    for mode in MODES:
        arm_rows = [row for row in rows if row["mode"] == mode]
        ef_values = {row["effective_ef_search"] for row in arm_rows}
        scan_values = {row["iterative_scan"] for row in arm_rows}
        if len(ef_values) != 1 or len(scan_values) != 1:
            raise AuditError(f"{path} changes effective settings within {mode}")
        effective_scan_counts: dict[str, int] = defaultdict(int)
        for row in arm_rows:
            effective_scan_counts[row["effective_iterative_scan"]] += 1
        settings[mode] = {
            "ef_search": int(next(iter(ef_values))),
            "iterative_scan": next(iter(scan_values)),
            "effective_iterative_scan_counts": dict(
                sorted(effective_scan_counts.items())
            ),
        }
    return rows, {
        "raw_path": str(path),
        "raw_sha256": sha256_file(path),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "paired_queries": len(pairs),
        "settings": settings,
    }


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{label} root is not an object: {path}")
    return value


def audit_combined_raw(
    path: Path,
    *,
    expected_requests: int = EXPECTED_REQUESTS,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Validate one paired/interleaved q5K latency artifact and its plan."""
    fields, rows = read_csv(path)
    require_fields(
        fields,
        (
            "selectivity", "filter_name", "mode", "query_no", "query_id",
            "repeat", "pair_key", "request_no", "trace_cycle", "recall",
            "end_to_end_ms", "effective_ef_search", "max_scan_tuples",
            "scan_mem_multiplier", "iterative_scan", "guided_collect_target",
            "traversal_guided_target",
            "effective_iterative_scan", "backend_cpu_exact_match", "error",
        ),
        path,
    )
    if len(rows) != expected_requests * len(MODES):
        raise AuditError(
            f"{path} has {len(rows)} rows, expected "
            f"{expected_requests * len(MODES)}"
        )
    if any(row.get("error", "").strip() for row in rows):
        raise AuditError(f"{path} contains query errors")
    if any(
        row.get("backend_cpu_exact_match", "").lower() != "true"
        for row in rows
    ):
        raise AuditError(f"{path} contains a backend outside its CPU partition")
    if {row["mode"] for row in rows} != set(MODES):
        raise AuditError(f"{path} does not contain exactly Stock and SQLens")
    filters = sorted({row["filter_name"] for row in rows})
    if len(filters) != EXPECTED_FILTERS:
        raise AuditError(
            f"{path} covers {len(filters)} filters, expected {EXPECTED_FILTERS}"
        )

    pairs: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    arm_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        key = (row["filter_name"], row["query_no"], row["repeat"])
        mode = row["mode"]
        if mode in pairs[key]:
            raise AuditError(f"duplicate latency row for {key}/{mode}")
        latency = float(row["end_to_end_ms"])
        recall = float(row["recall"])
        if not math.isfinite(latency) or latency <= 0.0:
            raise AuditError(f"invalid latency in {path}")
        if not math.isfinite(recall) or not 0.0 <= recall <= 1.0:
            raise AuditError(f"invalid recall in {path}")
        pairs[key][mode] = row
        arm_counts[mode] += 1
    if set(arm_counts.values()) != {expected_requests}:
        raise AuditError(f"unbalanced arm coverage in {path}: {dict(arm_counts)}")
    if any(set(arms) != set(MODES) for arms in pairs.values()):
        raise AuditError(f"{path} is not strictly paired")
    if {repeat for _filter, _query, repeat in pairs} != {"0"}:
        raise AuditError(f"{path} is not the expedited r1 protocol")
    for key, arms in pairs.items():
        pair_keys = {row["pair_key"] for row in arms.values()}
        if len(pair_keys) != 1:
            raise AuditError(f"paired arms disagree on pair_key for {key}")
    trace_sha, _trace = canonical_request_trace(
        rows, expected_requests=expected_requests, label="latency trace"
    )
    search_identity: dict[str, dict[str, object]] = {}
    for filter_name in filters:
        search_identity[filter_name] = {}
        for mode in MODES:
            settings = {
                (
                    int(row["effective_ef_search"]),
                    int(row["max_scan_tuples"]),
                    float(row["scan_mem_multiplier"]),
                    row["iterative_scan"],
                    int(row["guided_collect_target"]),
                    int(row["traversal_guided_target"]),
                )
                for row in rows
                if row["filter_name"] == filter_name and row["mode"] == mode
            }
            if len(settings) != 1:
                raise AuditError(
                    f"latency settings change within {filter_name}/{mode}"
                )
            setting = next(iter(settings))
            search_identity[filter_name][mode] = {
                "ef_search": setting[0],
                "max_scan_tuples": setting[1],
                "scan_mem_multiplier": setting[2],
                "iterative_scan": setting[3],
                "guided_collect_target": setting[4],
                "traversal_guided_target": setting[5],
            }

    plan_path = Path(str(path) + ".plan.json")
    plan = read_json_object(plan_path, "latency plan")
    raw_sha = sha256_file(path)
    query_errors = plan.get("query_error_summary")
    query_contract = plan.get("query_contract")
    if (
        plan.get("status") != "complete"
        or int(plan.get("output_rows", -1)) != len(rows)
        or plan.get("output_sha256") != raw_sha
        or not isinstance(query_errors, Mapping)
        or int(query_errors.get("error_rows", -1)) != 0
        or not isinstance(query_contract, Mapping)
        or int(query_contract.get("workload_requests", -1)) != expected_requests
        or int(query_contract.get("workload_unique_queries", -1))
        != expected_requests
    ):
        raise AuditError(f"latency plan does not prove complete q5K coverage: {plan_path}")
    startup = plan.get("sqlens_runtime_identity_startup")
    final = plan.get("sqlens_runtime_identity_final")
    if (
        not isinstance(startup, Mapping)
        or startup.get("exact_match") is not True
        or not isinstance(final, Mapping)
        or final.get("exact_match") is not True
        or startup.get("observed_build_id") != final.get("observed_build_id")
        or startup.get("observed_vector_so_sha256")
        != final.get("observed_vector_so_sha256")
    ):
        raise AuditError(f"latency plan lacks stable SQLens runtime identity: {plan_path}")
    return rows, {
        "raw_path": str(path),
        "raw_sha256": raw_sha,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "filters": filters,
        "requests_per_arm": expected_requests,
        "request_trace_identity_sha256": trace_sha,
        "workload_sha256": str(query_contract.get("workload_sha256", "")),
        "filters_sha256": str(query_contract.get("filters_sha256", "")),
        "truth_sha256": str(query_contract.get("truth_sha256", "")),
        "search_identity": search_identity,
        "sqlens_build_id": str(startup.get("observed_build_id", "")),
        "vector_so_sha256": str(startup.get("observed_vector_so_sha256", "")),
    }


def audit_throughput_repeats(
    path: Path,
    *,
    target_recall: float,
    expected_requests: int = EXPECTED_REQUESTS,
    target_tolerance: float = 0.005,
) -> tuple[dict[str, float], dict[str, object]]:
    """Validate independently measured c16 QPS without weakening release gates."""
    fields, rows = read_csv(path)
    require_fields(
        fields,
        (
            "arm_id", "clients", "repeat_id", "requests", "unique_queries",
            "completed_queries", "error_count", "wall_clock_seconds",
            "recall_mean", "recall_ci95_low", "recall_ci95_high",
            "throughput_qps", "throughput_source", "status",
        ),
        path,
    )
    if len(rows) != 2 or {row["arm_id"] for row in rows} != {
        "stock_pgvector", "sqlens_full"
    }:
        raise AuditError(f"{path} must contain exactly two throughput arms")
    qps: dict[str, float] = {}
    arm_evidence: dict[str, object] = {}
    for row in rows:
        arm = row["arm_id"]
        completed = int(row["completed_queries"])
        wall_seconds = float(row["wall_clock_seconds"])
        observed_qps = float(row["throughput_qps"])
        recall = float(row["recall_mean"])
        if (
            int(row["clients"]) != 16
            or row["repeat_id"] != "0"
            or int(row["requests"]) != expected_requests
            or int(row["unique_queries"]) != expected_requests
            or completed != expected_requests
            or int(row["error_count"]) != 0
            or wall_seconds <= 0.0
            or row["throughput_source"]
            != "measured_completed_over_barrier_wall_clock"
            or not math.isclose(
                observed_qps, completed / wall_seconds, rel_tol=1e-12
            )
            or recall < target_recall - target_tolerance
        ):
            raise AuditError(f"invalid throughput evidence for {arm} in {path}")
        qps[arm] = observed_qps
        arm_evidence[arm] = {
            "recall_mean": recall,
            "recall_ci95_low": float(row["recall_ci95_low"]),
            "recall_ci95_high": float(row["recall_ci95_high"]),
            "wall_clock_seconds": wall_seconds,
            "status": row["status"],
        }

    suffix = ".repeats.csv"
    if not path.name.endswith(suffix):
        raise AuditError(f"throughput filename must end in {suffix}: {path}")
    manifest_path = path.with_name(path.name[: -len(suffix)] + ".manifest.json")
    manifest = read_json_object(manifest_path, "throughput manifest")
    output = manifest.get("outputs", {}).get("repeats", {})
    requests_output = manifest.get("outputs", {}).get("requests", {})
    protocol = manifest.get("protocol")
    gates = manifest.get("gates")
    if (
        not isinstance(output, Mapping)
        or output.get("sha256") != sha256_file(path)
        or int(output.get("rows", -1)) != len(rows)
        or not isinstance(protocol, Mapping)
        or int(protocol.get("clients", -1)) != 16
        or int(protocol.get("requests_per_arm_repeat", -1)) != expected_requests
        or protocol.get("throughput_formula")
        != "completed_queries / barrier_wall_clock_seconds"
        or not isinstance(gates, Mapping)
        or gates.get("barrier_wall_clock_qps") is not True
        or gates.get("independent_client_backends") is not True
        or gates.get("telemetry_complete") is not True
    ):
        raise AuditError(f"throughput manifest is not bound to {path}")
    if not isinstance(requests_output, Mapping):
        raise AuditError(f"throughput manifest has no requests output: {manifest_path}")
    requests_path = Path(str(requests_output.get("path", "")))
    if not requests_path.is_absolute():
        requests_path = ROOT / requests_path
    request_fields, request_rows = read_csv(requests_path)
    require_fields(
        request_fields,
        (
            "arm_id", "repeat_id", "request_no", "trace_cycle",
            "filter_name", "query_no", "query_id", "recall_at_10",
            "error_type", "error",
        ),
        requests_path,
    )
    if (
        len(request_rows) != expected_requests * 2
        or int(requests_output.get("rows", -1)) != len(request_rows)
        or requests_output.get("sha256") != sha256_file(requests_path)
        or any(row["error_type"].strip() or row["error"].strip() for row in request_rows)
    ):
        raise AuditError(f"invalid throughput request evidence: {requests_path}")
    arm_traces: dict[str, str] = {}
    filters: set[str] = set()
    for arm in ("stock_pgvector", "sqlens_full"):
        arm_rows = [row for row in request_rows if row["arm_id"] == arm]
        if len(arm_rows) != expected_requests or {row["repeat_id"] for row in arm_rows} != {"0"}:
            raise AuditError(f"incomplete throughput request trace for {arm}")
        arm_traces[arm], _trace = canonical_request_trace(
            arm_rows,
            expected_requests=expected_requests,
            label=f"throughput trace {arm}",
        )
        filters.update(row["filter_name"] for row in arm_rows)
    if len(set(arm_traces.values())) != 1 or len(filters) != EXPECTED_FILTERS:
        raise AuditError("throughput arms do not share one complete 14-filter trace")
    per_filter_recall: dict[str, dict[str, object]] = {}
    all_filter_targets_met = True
    for filter_name in sorted(filters):
        per_filter_recall[filter_name] = {}
        for arm in ("stock_pgvector", "sqlens_full"):
            recalls = [
                float(row["recall_at_10"])
                for row in request_rows
                if row["arm_id"] == arm and row["filter_name"] == filter_name
            ]
            if not recalls or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in recalls
            ):
                raise AuditError(
                    f"invalid throughput recall for {filter_name}/{arm}"
                )
            recall_mean = statistics.fmean(recalls)
            target_met = recall_mean >= target_recall - target_tolerance
            all_filter_targets_met = all_filter_targets_met and target_met
            per_filter_recall[filter_name][arm] = {
                "requests": len(recalls),
                "recall_mean": recall_mean,
                "target_met": target_met,
            }

    configuration = manifest.get("configuration")
    inputs = manifest.get("inputs")
    runtime_binary = manifest.get("runtime_binary")
    if (
        not isinstance(configuration, Mapping)
        or not isinstance(configuration.get("value"), Mapping)
        or not isinstance(inputs, Mapping)
        or not isinstance(runtime_binary, Mapping)
    ):
        raise AuditError(f"throughput manifest lacks identity evidence: {manifest_path}")
    config_value = configuration["value"]
    assert isinstance(config_value, Mapping)
    base_search = config_value.get("search")
    per_filter_search = config_value.get("per_filter_search")
    if not isinstance(base_search, Mapping) or not isinstance(per_filter_search, Mapping):
        raise AuditError(f"throughput manifest lacks search configuration: {manifest_path}")
    search_identity: dict[str, dict[str, object]] = {}
    arm_to_mode = {
        "stock_pgvector": STOCK_MODE,
        "sqlens_full": SQLENS_MODE,
    }
    for filter_name in sorted(filters):
        search_identity[filter_name] = {}
        for arm, mode in arm_to_mode.items():
            base = base_search.get(mode)
            overrides = per_filter_search.get(arm)
            if not isinstance(base, Mapping) or not isinstance(overrides, Mapping):
                raise AuditError(f"missing throughput search settings for {arm}")
            ef_overrides = overrides.get("ef_search", {})
            target_overrides = overrides.get("traversal_guided_target", {})
            if not isinstance(ef_overrides, Mapping) or not isinstance(target_overrides, Mapping):
                raise AuditError(f"invalid per-filter search settings for {arm}")
            search_identity[filter_name][mode] = {
                "ef_search": int(ef_overrides.get(filter_name, base["ef_search"])),
                "max_scan_tuples": int(base["max_scan_tuples"]),
                "scan_mem_multiplier": float(base["scan_mem_multiplier"]),
                "iterative_scan": str(base["iterative_scan"]),
                "guided_collect_target": int(base["guided_collect_target"]),
                "traversal_guided_target": int(
                    target_overrides.get(
                        filter_name, base["traversal_guided_target"]
                    )
                ),
            }
    filters_input = inputs.get("filters_csv")
    truth_input = inputs.get("truth_csv")
    if not isinstance(filters_input, Mapping) or not isinstance(truth_input, Mapping):
        raise AuditError(f"throughput manifest lacks truth/filter bindings: {manifest_path}")
    return qps, {
        "repeats_path": str(path),
        "repeats_sha256": sha256_file(path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_valid": manifest.get("artifact_valid") is True,
        "paper_eligible": manifest.get("paper_eligible") is True,
        "requests_path": str(requests_path),
        "requests_sha256": sha256_file(requests_path),
        "request_trace_identity_sha256": next(iter(arm_traces.values())),
        "filters_sha256": str(filters_input.get("sha256", "")),
        "truth_sha256": str(truth_input.get("sha256", "")),
        "search_identity": search_identity,
        "per_filter_recall": per_filter_recall,
        "all_filter_targets_met_with_0p005_tolerance": (
            all_filter_targets_met
        ),
        "sqlens_build_id": str(runtime_binary.get("expected_build_id", "")),
        "vector_so_sha256": str(
            runtime_binary.get("expected_vector_so_sha256", "")
        ),
        "protocol": {
            "clients": protocol.get("clients"),
            "repeats": protocol.get("repeats"),
            "client_cpu_list": protocol.get("client_cpu_list"),
            "backend_cpu_list": protocol.get("backend_cpu_list"),
        },
        "gates": dict(gates),
        "arms": arm_evidence,
    }


def stratified_speedup(
    by_filter: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float, int]:
    if samples < 100:
        raise AuditError("bootstrap requires at least 100 samples")
    rng = random.Random(seed)
    observed_logs: list[float] = []
    bootstrap_logs = [0.0] * samples
    wins = 0
    for filter_name in sorted(by_filter):
        stock = list(by_filter[filter_name][STOCK_MODE])
        sqlens = list(by_filter[filter_name][SQLENS_MODE])
        if len(stock) != len(sqlens) or not stock:
            raise AuditError(f"unpaired latency vectors for {filter_name}")
        stock_mean = statistics.fmean(stock)
        sqlens_mean = statistics.fmean(sqlens)
        observed_logs.append(math.log(stock_mean / sqlens_mean))
        wins += int(sqlens_mean < stock_mean)
        for sample in range(samples):
            indexes = [rng.randrange(len(stock)) for _ in stock]
            stock_sample = statistics.fmean(stock[index] for index in indexes)
            sqlens_sample = statistics.fmean(sqlens[index] for index in indexes)
            bootstrap_logs[sample] += math.log(stock_sample / sqlens_sample)
    denominator = len(by_filter)
    distribution = [math.exp(value / denominator) for value in bootstrap_logs]
    return (
        math.exp(statistics.fmean(observed_logs)),
        percentile(distribution, 0.025),
        percentile(distribution, 0.975),
        wins,
    )


def summarize_combined(
    raw_csv: Path,
    throughput_repeats_csv: Path,
    dataset: str,
    target_recall: float,
    out_prefix: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    expected_requests: int = EXPECTED_REQUESTS,
) -> dict[str, Path]:
    """Summarize a monolithic q5K latency run plus independent c16 QPS."""
    rows, latency_evidence = audit_combined_raw(
        raw_csv, expected_requests=expected_requests
    )
    qps, throughput_evidence = audit_throughput_repeats(
        throughput_repeats_csv,
        target_recall=target_recall,
        expected_requests=expected_requests,
    )
    identity_fields = (
        "request_trace_identity_sha256",
        "filters_sha256",
        "truth_sha256",
        "sqlens_build_id",
        "vector_so_sha256",
    )
    mismatches = [
        field for field in identity_fields
        if latency_evidence.get(field) != throughput_evidence.get(field)
    ]
    if latency_evidence.get("search_identity") != throughput_evidence.get(
        "search_identity"
    ):
        mismatches.append("search_identity")
    if mismatches:
        raise AuditError(
            "latency and throughput evidence are not the same experiment: "
            + ", ".join(mismatches)
        )
    filters = list(latency_evidence["filters"])
    latency_by_filter: dict[str, dict[str, list[float]]] = {
        filter_name: {mode: [] for mode in MODES} for filter_name in filters
    }
    recall_by_filter: dict[str, dict[str, list[float]]] = {
        filter_name: {mode: [] for mode in MODES} for filter_name in filters
    }
    latency_by_mode: dict[str, list[float]] = {mode: [] for mode in MODES}
    recall_by_mode: dict[str, list[float]] = {mode: [] for mode in MODES}
    selectivity_by_filter: dict[str, str] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            item["filter_name"], int(item["query_no"]),
            int(item["repeat"]), MODES.index(item["mode"]),
        ),
    ):
        filter_name = row["filter_name"]
        mode = row["mode"]
        latency = float(row["end_to_end_ms"])
        recall = float(row["recall"])
        selectivity_by_filter.setdefault(filter_name, row["selectivity"])
        latency_by_filter[filter_name][mode].append(latency)
        recall_by_filter[filter_name][mode].append(recall)
        latency_by_mode[mode].append(latency)
        recall_by_mode[mode].append(recall)

    speedup, low, high, wins = stratified_speedup(
        latency_by_filter, samples=bootstrap_samples, seed=bootstrap_seed
    )
    per_filter: list[dict[str, object]] = []
    all_targets_met = True
    for filter_name in filters:
        stock_latency = latency_by_filter[filter_name][STOCK_MODE]
        sqlens_latency = latency_by_filter[filter_name][SQLENS_MODE]
        stock_recall = statistics.fmean(
            recall_by_filter[filter_name][STOCK_MODE]
        )
        sqlens_recall = statistics.fmean(
            recall_by_filter[filter_name][SQLENS_MODE]
        )
        stock_mean = statistics.fmean(stock_latency)
        sqlens_mean = statistics.fmean(sqlens_latency)
        stock_met = stock_recall >= target_recall - 0.005
        sqlens_met = sqlens_recall >= target_recall - 0.005
        all_targets_met = all_targets_met and stock_met and sqlens_met
        per_filter.append({
            "dataset": dataset,
            "target_recall": target_recall,
            "filter_name": filter_name,
            "selectivity_pct": selectivity_by_filter[filter_name],
            "requests_per_arm": len(stock_latency),
            "stock_recall": stock_recall,
            "sqlens_recall": sqlens_recall,
            "stock_mean_ms": stock_mean,
            "sqlens_mean_ms": sqlens_mean,
            "stock_p95_ms": percentile(stock_latency, 0.95),
            "sqlens_p95_ms": percentile(sqlens_latency, 0.95),
            "stock_p99_ms": percentile(stock_latency, 0.99),
            "sqlens_p99_ms": percentile(sqlens_latency, 0.99),
            "speedup": stock_mean / sqlens_mean,
            "stock_target_met": stock_met,
            "sqlens_target_met": sqlens_met,
        })

    stock_latency = latency_by_mode[STOCK_MODE]
    sqlens_latency = latency_by_mode[SQLENS_MODE]
    throughput_targets_met = bool(
        throughput_evidence[
            "all_filter_targets_met_with_0p005_tolerance"
        ]
    )
    combined_targets_met = all_targets_met and throughput_targets_met
    summary = [{
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "target_recall": target_recall,
        "stock_recall": statistics.fmean(recall_by_mode[STOCK_MODE]),
        "sqlens_recall": statistics.fmean(recall_by_mode[SQLENS_MODE]),
        "stock_mean_latency_ms": statistics.fmean(stock_latency),
        "sqlens_mean_latency_ms": statistics.fmean(sqlens_latency),
        "stock_latency_p95_ms": percentile(stock_latency, 0.95),
        "sqlens_latency_p95_ms": percentile(sqlens_latency, 0.95),
        "stock_latency_p99_ms": percentile(stock_latency, 0.99),
        "sqlens_latency_p99_ms": percentile(sqlens_latency, 0.99),
        "stock_qps": qps["stock_pgvector"],
        "sqlens_qps": qps["sqlens_full"],
        "speedup_geomean": speedup,
        "speedup_ci95_low": low,
        "speedup_ci95_high": high,
        "wins": wins,
        "wins_denominator": EXPECTED_FILTERS,
        "requests_per_arm": expected_requests,
        "repeats": 1,
        "latency_all_filter_targets_met_with_0p005_tolerance": all_targets_met,
        "throughput_all_filter_targets_met_with_0p005_tolerance": (
            throughput_targets_met
        ),
        "all_filter_targets_met_with_0p005_tolerance": combined_targets_met,
        "classification": "expedited_q5k_r1_with_independent_c16_qps",
        "paper_eligible": False,
    }]
    paths = {
        "per_filter": out_prefix.with_name(out_prefix.name + "_per_filter.csv"),
        "summary": out_prefix.with_name(out_prefix.name + "_summary.csv"),
        "manifest": out_prefix.with_name(out_prefix.name + "_manifest.json"),
    }
    atomic_csv(paths["per_filter"], per_filter)
    atomic_csv(paths["summary"], summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "sqlens_expedited_q5k_matched_recall_summary",
        "status": "complete",
        "artifact_valid": combined_targets_met,
        "paper_eligible": False,
        "classification": "expedited_q5k_r1_with_independent_c16_qps",
        "dataset": dataset,
        "target_recall": target_recall,
        "filters": filters,
        "requests_per_arm": expected_requests,
        "repeats": 1,
        "target_gate": {
            "definition": "per-filter mean Recall@10 >= target - 0.005",
            "latency_all_filters_met": all_targets_met,
            "throughput_all_filters_met": throughput_targets_met,
            "all_filters_met": combined_targets_met,
        },
        "bootstrap": {
            "method": "paired_query_cluster_within_filter_then_14_filter_log_ratio_geomean",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "latency_evidence": latency_evidence,
        "throughput_evidence": throughput_evidence,
        "cross_artifact_identity": {
            "passed": True,
            "request_trace_identity_sha256": latency_evidence[
                "request_trace_identity_sha256"
            ],
            "filters_sha256": latency_evidence["filters_sha256"],
            "truth_sha256": latency_evidence["truth_sha256"],
            "sqlens_build_id": latency_evidence["sqlens_build_id"],
            "vector_so_sha256": latency_evidence["vector_so_sha256"],
            "search_identity_sha256": sha256_json(
                latency_evidence["search_identity"]
            ),
        },
        "release_note": (
            "This q5K/r1 artifact is intentionally not promoted to the "
            "registered q10K/r3 latency and r6 throughput release protocol."
        ),
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items() if name != "manifest"
        },
    }
    atomic_json(paths["manifest"], manifest)
    return paths


def summarize(
    config_path: Path,
    dataset: str,
    out_prefix: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Path]:
    config = load_config(config_path, dataset)
    filters = [str(item) for item in config["filters"]]
    target = float(config.get("target_recall", 0.90))
    build_id = str(config["expected_build_id"])
    vector_sha = str(config["expected_vector_so_sha256"])
    all_rows: list[dict[str, str]] = []
    source_records: dict[str, object] = {}
    per_filter: list[dict[str, object]] = []
    latency_by_filter: dict[str, dict[str, list[float]]] = {}
    latency_by_mode: dict[str, list[float]] = {mode: [] for mode in MODES}
    recall_by_mode: dict[str, list[float]] = {mode: [] for mode in MODES}

    for filter_name in filters:
        path = source_for(config, filter_name)
        rows, source_record = audit_filter(
            path,
            filter_name,
            expected_build_id=build_id,
            expected_vector_sha=vector_sha,
        )
        source_records[filter_name] = source_record
        all_rows.extend(rows)
        latency_by_filter[filter_name] = {}
        arm_stats: dict[str, dict[str, object]] = {}
        by_mode = {mode: [row for row in rows if row["mode"] == mode] for mode in MODES}
        for mode in MODES:
            ordered = sorted(by_mode[mode], key=lambda row: (int(row["query_no"]), int(row["repeat"])))
            latencies = [float(row["end_to_end_ms"]) for row in ordered]
            recalls = [float(row["recall"]) for row in ordered]
            latency_by_filter[filter_name][mode] = latencies
            latency_by_mode[mode].extend(latencies)
            recall_by_mode[mode].extend(recalls)
            arm_stats[mode] = {
                "recall": statistics.fmean(recalls),
                "mean_ms": statistics.fmean(latencies),
                "p95_ms": percentile(latencies, 0.95),
                "p99_ms": percentile(latencies, 0.99),
            }
        per_filter.append({
            "dataset": dataset,
            "target_recall": target,
            "filter_name": filter_name,
            "selectivity_pct": rows[0]["selectivity"],
            "requests_per_arm": len(by_mode[STOCK_MODE]),
            "stock_recall": arm_stats[STOCK_MODE]["recall"],
            "sqlens_recall": arm_stats[SQLENS_MODE]["recall"],
            "stock_mean_ms": arm_stats[STOCK_MODE]["mean_ms"],
            "sqlens_mean_ms": arm_stats[SQLENS_MODE]["mean_ms"],
            "stock_p95_ms": arm_stats[STOCK_MODE]["p95_ms"],
            "sqlens_p95_ms": arm_stats[SQLENS_MODE]["p95_ms"],
            "stock_p99_ms": arm_stats[STOCK_MODE]["p99_ms"],
            "sqlens_p99_ms": arm_stats[SQLENS_MODE]["p99_ms"],
            "speedup": arm_stats[STOCK_MODE]["mean_ms"] / arm_stats[SQLENS_MODE]["mean_ms"],
            "stock_target_met": arm_stats[STOCK_MODE]["recall"] >= target - 0.005,
            "sqlens_target_met": arm_stats[SQLENS_MODE]["recall"] >= target - 0.005,
            "source_path": source_record["raw_path"],
            "source_sha256": source_record["raw_sha256"],
        })

    request_counts = {mode: len(values) for mode, values in latency_by_mode.items()}
    if set(request_counts.values()) != {EXPECTED_REQUESTS}:
        raise AuditError(
            f"{dataset} request coverage is {request_counts}, expected {EXPECTED_REQUESTS} per arm"
        )
    speedup, low, high, wins = stratified_speedup(
        latency_by_filter, samples=bootstrap_samples, seed=bootstrap_seed
    )
    target_met = all(
        bool(row["stock_target_met"]) and bool(row["sqlens_target_met"])
        for row in per_filter
    )
    summary = [{
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "target_recall": target,
        "stock_recall": statistics.fmean(recall_by_mode[STOCK_MODE]),
        "sqlens_recall": statistics.fmean(recall_by_mode[SQLENS_MODE]),
        "stock_mean_latency_ms": statistics.fmean(latency_by_mode[STOCK_MODE]),
        "sqlens_mean_latency_ms": statistics.fmean(latency_by_mode[SQLENS_MODE]),
        "stock_latency_p95_ms": percentile(latency_by_mode[STOCK_MODE], 0.95),
        "sqlens_latency_p95_ms": percentile(latency_by_mode[SQLENS_MODE], 0.95),
        "stock_latency_p99_ms": percentile(latency_by_mode[STOCK_MODE], 0.99),
        "sqlens_latency_p99_ms": percentile(latency_by_mode[SQLENS_MODE], 0.99),
        "stock_qps": "",
        "sqlens_qps": "",
        "speedup_geomean": speedup,
        "speedup_ci95_low": low,
        "speedup_ci95_high": high,
        "wins": wins,
        "wins_denominator": EXPECTED_FILTERS,
        "requests_per_arm": EXPECTED_REQUESTS,
        "repeats": 1,
        "all_filter_targets_met_with_0p005_tolerance": target_met,
        "classification": "expedited_q5k_r1",
        "paper_eligible": False,
    }]

    paths = {
        "raw": out_prefix.with_name(out_prefix.name + "_raw.csv"),
        "per_filter": out_prefix.with_name(out_prefix.name + "_per_filter.csv"),
        "summary": out_prefix.with_name(out_prefix.name + "_summary.csv"),
        "manifest": out_prefix.with_name(out_prefix.name + "_manifest.json"),
    }
    all_rows.sort(key=lambda row: (filters.index(row["filter_name"]), int(row["query_no"]), int(row["repeat"]), MODES.index(row["mode"])))
    atomic_csv(paths["raw"], all_rows)
    atomic_csv(paths["per_filter"], per_filter)
    atomic_csv(paths["summary"], summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "sqlens_expedited_q5k_matched_recall_summary",
        "status": "complete",
        "artifact_valid": True,
        "paper_eligible": False,
        "classification": "expedited_q5k_r1",
        "dataset": dataset,
        "target_recall": target,
        "filters": filters,
        "requests_per_arm": EXPECTED_REQUESTS,
        "repeats": 1,
        "expected_build_id": build_id,
        "expected_vector_so_sha256": vector_sha,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "sources": source_records,
        "source_set_sha256": sha256_json(source_records),
        "bootstrap": {
            "method": "paired_query_cluster_within_filter_then_14_filter_log_ratio_geomean",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "target_gate": {
            "tolerance": 0.005,
            "all_filters_met": target_met,
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items() if name != "manifest"
        },
    }
    atomic_json(paths["manifest"], manifest)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--throughput-repeats", type=Path)
    parser.add_argument("--target-recall", type=float)
    parser.add_argument(
        "--dataset", choices=("amazon", "yfcc", "laion"), required=True
    )
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    args = parser.parse_args()
    if args.raw_csv is not None:
        if args.config is not None:
            parser.error("--raw-csv and --config are mutually exclusive")
        if args.throughput_repeats is None or args.target_recall is None:
            parser.error(
                "--raw-csv requires --throughput-repeats and --target-recall"
            )
        paths = summarize_combined(
            args.raw_csv.resolve(),
            args.throughput_repeats.resolve(),
            args.dataset,
            args.target_recall,
            args.out_prefix.resolve(),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    else:
        if args.config is None:
            parser.error("one of --config or --raw-csv is required")
        paths = summarize(
            args.config.resolve(),
            args.dataset,
            args.out_prefix.resolve(),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    print(paths["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
