#!/usr/bin/env python3
"""Convert audited Figure 5 query rows into repeat-level latency evidence."""

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
    from . import run_figure5_matched_latency as matched_latency
except ImportError:
    import figure5_converter_binding as converter_binding
    import figure5_frontier_artifact as artifact
    import pgvector_figure5_throughput as throughput
    import run_figure5_matched_latency as matched_latency


DATASET_IDS = {
    "amazon": "amazon10m",
    "yfcc": "yfcc10m",
    "laion": "laion25m",
}
MODE_ARMS = {
    "original": "stock_pgvector",
    "design1_bloom_bfs_layout_d3": "sqlens_full",
}
CLUSTER_BOOTSTRAP_SAMPLES = 2_000
CLUSTER_BOOTSTRAP_SEED = 20_260_728
CLUSTER_BOOTSTRAP_METHOD = (
    "query_id_cluster_stratified_predicate_percentile_bootstrap_95"
)


class LatencyRepeatError(RuntimeError):
    """Raw query evidence violates the Figure 5 latency contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LatencyRepeatError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LatencyRepeatError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise LatencyRepeatError(f"cannot read CSV {path}: {exc}") from exc
    if not fields or not rows:
        raise LatencyRepeatError(f"CSV is empty: {path}")
    if any(None in row for row in rows):
        raise LatencyRepeatError(f"CSV contains a row wider than its header: {path}")
    return fields, rows


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise LatencyRepeatError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def mean_ci95(
    values: Sequence[float],
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[float, float, float]:
    if not values:
        raise LatencyRepeatError("cannot compute a confidence interval without values")
    mean = statistics.fmean(values)
    if len(values) == 1:
        low = high = mean
    else:
        half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
        low, high = mean - half, mean + half
    if lower is not None:
        low = max(lower, low)
    if upper is not None:
        high = min(upper, high)
    return mean, low, high


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def query_cluster_bootstrap_recall(
    rows: Sequence[Mapping[str, object]],
    *,
    value_field: str,
    query_field: str = "query_id",
    filter_field: str = "filter_name",
    samples: int = CLUSTER_BOOTSTRAP_SAMPLES,
    seed: int = CLUSTER_BOOTSTRAP_SEED,
    seed_label: str = "",
) -> dict[str, object]:
    """Return a predicate-stratified Recall@10 CI over query clusters.

    A query may occur in multiple repeats, so all observations for one
    ``query_id`` are resampled together. Predicate strata are preserved in
    every draw, matching the selector's global-min-predicate target semantics.
    """
    if samples < 100:
        raise LatencyRepeatError("cluster bootstrap requires at least 100 samples")
    clusters: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    query_predicates: dict[str, str] = {}
    observed_values: list[float] = []
    for row in rows:
        query_id = str(row.get(query_field) or "").strip()
        filter_name = str(row.get(filter_field) or "").strip()
        try:
            raw_value = row.get(value_field)
            value = float(str(raw_value).strip())
        except (TypeError, ValueError) as exc:
            raise LatencyRepeatError(
                f"{value_field} contains a non-numeric value"
            ) from exc
        if (
            not query_id
            or not filter_name
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise LatencyRepeatError(
                "cluster-bootstrap recall row is outside the expected domain"
            )
        previous = query_predicates.setdefault(query_id, filter_name)
        if previous != filter_name:
            raise LatencyRepeatError(
                f"query_id {query_id!r} is assigned to multiple predicates"
            )
        clusters[filter_name][query_id].append(value)
        observed_values.append(value)
    if not observed_values:
        raise LatencyRepeatError("cluster bootstrap has no recall rows")

    cluster_means: dict[str, list[float]] = {}
    cluster_sizes: set[int] = set()
    for filter_name, query_groups in sorted(clusters.items()):
        means: list[float] = []
        for values in query_groups.values():
            cluster_sizes.add(len(values))
            means.append(statistics.fmean(values))
        cluster_means[filter_name] = means
    if len(cluster_sizes) != 1:
        raise LatencyRepeatError(
            "query clusters do not contain equal repeat coverage"
        )

    filter_names = sorted(cluster_means)
    filter_weights = {
        filter_name: len(cluster_means[filter_name])
        for filter_name in filter_names
    }
    total_clusters = sum(filter_weights.values())
    rng = random.Random(_derived_seed(seed, seed_label))
    aggregate_samples: list[float] = []
    predicate_samples: dict[str, list[float]] = {
        filter_name: [] for filter_name in filter_names
    }
    for _ in range(samples):
        weighted_sum = 0.0
        for filter_name in filter_names:
            values = cluster_means[filter_name]
            sampled_mean = statistics.fmean(
                values[rng.randrange(len(values))] for _ in values
            )
            predicate_samples[filter_name].append(sampled_mean)
            weighted_sum += sampled_mean * filter_weights[filter_name]
        aggregate_samples.append(weighted_sum / total_clusters)

    per_predicate: dict[str, dict[str, object]] = {}
    for filter_name in filter_names:
        values = [
            value
            for cluster in clusters[filter_name].values()
            for value in cluster
        ]
        sampled = predicate_samples[filter_name]
        per_predicate[filter_name] = {
            "sample_count": len(values),
            "query_cluster_count": filter_weights[filter_name],
            "mean": statistics.fmean(values),
            "lower": percentile(sampled, 0.025),
            "upper": percentile(sampled, 0.975),
        }
    return {
        "method": CLUSTER_BOOTSTRAP_METHOD,
        "samples": samples,
        "seed": seed,
        "seed_label": seed_label,
        "query_cluster_field": query_field,
        "predicate_field": filter_field,
        "sample_count": len(observed_values),
        "query_cluster_count": total_clusters,
        "predicate_count": len(filter_names),
        "filter_names": filter_names,
        "mean": statistics.fmean(observed_values),
        "lower": percentile(aggregate_samples, 0.025),
        "upper": percentile(aggregate_samples, 0.975),
        "per_predicate": per_predicate,
        "min_predicate_lcb95": min(
            float(stats["lower"]) for stats in per_predicate.values()
        ),
    }


def _require_fields(fields: Sequence[str], required: set[str], path: Path) -> None:
    missing = sorted(required - set(fields))
    if missing:
        raise LatencyRepeatError(f"{path} is missing fields {missing}")


def _prewarm_valid(plan: Mapping[str, Any]) -> bool:
    prewarm = plan.get("relation_prewarm")
    if not isinstance(prewarm, Mapping):
        return False
    records = prewarm.get("records")
    return (
        prewarm.get("enabled") is True
        and prewarm.get("complete") is True
        and isinstance(records, list)
        and len(records) == 3
        and all(
            isinstance(item, Mapping)
            and int(item.get("warmed_blocks", -1))
            == int(item.get("expected_blocks", -2))
            and int(item.get("warmed_blocks", 0)) > 0
            for item in records
        )
    )


def _config_binding(
    cell: Mapping[str, Any],
    mode: str,
    release_sha256: str,
) -> tuple[str, str]:
    family = str(cell["scan_family"])
    ef_search = int(cell["ef_search"])
    config_id = f"{family}_ef{ef_search}"
    mode_configs = cell.get("mode_configs")
    if not isinstance(mode_configs, Mapping) or not isinstance(
        mode_configs.get(mode), Mapping
    ):
        raise LatencyRepeatError(
            f"cell {config_id} has no effective config for mode={mode}"
        )
    binding = {
        "artifact": "sqlens_figure5_latency_config_v1",
        "dataset": str(cell["dataset"]),
        "config_id": config_id,
        "mode": mode,
        "effective_mode_config": mode_configs[mode],
        "inputs": cell.get("inputs"),
        "cache_protocol": cell.get("cache_protocol"),
        "release_identity_sha256": release_sha256,
    }
    return config_id, sha256_json(binding)


def _matched_path(value: object, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise LatencyRepeatError(f"matched manifest has no {label} path")
    path = Path(text)
    if not path.is_absolute():
        path = matched_latency.ROOT / path
    if not path.is_file():
        raise LatencyRepeatError(f"matched {label} is missing: {path}")
    return path.resolve()


def _require_sha256(value: object, label: str) -> str:
    text = str(value or "").strip().lower()
    if not throughput.SHA256_RE.fullmatch(text):
        raise LatencyRepeatError(f"{label} is not a SHA-256 value")
    return text


def _audit_bound_file(
    binding: Mapping[str, Any],
    label: str,
    *,
    path_key: str = "path",
    sha_key: str = "sha256",
) -> tuple[Path, str]:
    path = _matched_path(binding.get(path_key), label)
    expected_sha = _require_sha256(binding.get(sha_key), f"{label} SHA")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise LatencyRepeatError(
            f"matched {label} SHA drifted: expected={expected_sha}, "
            f"observed={observed_sha}"
        )
    return path, observed_sha


def _matched_release(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    release = manifest.get("release_contract")
    if not isinstance(release, Mapping):
        raise LatencyRepeatError("matched run manifest has no release contract")
    _, release_sha256 = _audit_bound_file(release, "release contract")
    contract_id = str(release.get("contract_id") or "")
    build_id = str(release.get("expected_sqlens_build_id") or "")
    try:
        matched_latency.require_matching_release_tag(contract_id, build_id)
    except matched_latency.MatchedLatencyError as exc:
        raise LatencyRepeatError(str(exc)) from exc
    _require_sha256(
        release.get("expected_vector_so_sha256"),
        "matched expected vector.so SHA",
    )
    return release, release_sha256


def _matched_selector_pair(
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    release: Mapping[str, Any],
    release_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    selector = manifest.get("selector")
    if not isinstance(selector, Mapping):
        raise LatencyRepeatError("matched run manifest has no selector binding")
    csv_path, csv_sha = _audit_bound_file(
        selector,
        "selector CSV",
        path_key="csv",
        sha_key="selection_csv_sha256",
    )
    plan_path, plan_sha = _audit_bound_file(
        selector,
        "selector plan",
        path_key="plan",
        sha_key="selection_plan_sha256",
    )
    manifest_path, manifest_sha = _audit_bound_file(
        selector,
        "selector manifest",
        path_key="manifest",
        sha_key="selection_manifest_sha256",
    )
    frontier_binding = manifest.get("frontier_config")
    required_grid_binding = manifest.get("required_grid_contract")
    if not isinstance(frontier_binding, Mapping) or not isinstance(
        required_grid_binding, Mapping
    ):
        raise LatencyRepeatError(
            "matched run manifest lacks config/required-grid bindings"
        )
    config_path, _ = _audit_bound_file(frontier_binding, "frontier config")
    required_grid_path, _ = _audit_bound_file(
        required_grid_binding, "required-grid contract"
    )
    config = {
        "release_identity": {
            "expected_sqlens_build_id": release["expected_sqlens_build_id"],
            "expected_vector_so_sha256": release["expected_vector_so_sha256"],
        },
        "release_contract_sha256": release_sha256,
    }
    try:
        audited = matched_latency.validate_selection_artifacts(
            csv_path,
            plan_path,
            manifest_path,
            config,
            config_path=config_path,
            required_grid_contract=required_grid_path,
        )
    except matched_latency.MatchedLatencyError as exc:
        raise LatencyRepeatError(f"matched selector audit failed: {exc}") from exc
    expected_bindings = {
        "selection_csv_sha256": csv_sha,
        "selection_plan_sha256": plan_sha,
        "selection_manifest_sha256": manifest_sha,
    }
    if any(
        audited.get(key) != value
        for key, value in expected_bindings.items()
    ):
        raise LatencyRepeatError("matched selector SHA bindings are inconsistent")

    pair_id = str(cell.get("pair_id") or "").strip()
    if not pair_id:
        raise LatencyRepeatError("matched cell has no pair_id")
    fields, rows = read_csv(csv_path)
    _require_fields(
        fields,
        {"pair_id", "dataset", "target_recall", "selection_status"},
        csv_path,
    )
    selected = [
        row
        for row in rows
        if row["pair_id"] == pair_id and row["selection_status"] == "selected"
    ]
    if len(selected) != 1:
        raise LatencyRepeatError(
            f"matched pair {pair_id!r} is not uniquely selected by the selector"
        )
    selected_row = selected[0]
    try:
        selected_target = float(selected_row["target_recall"])
        cell_target = float(cell.get("target_recall"))
        stock = matched_latency.arm_config(selected_row, "stock")
        sqlens = matched_latency.arm_config(selected_row, "sqlens")
    except (TypeError, ValueError, matched_latency.MatchedLatencyError) as exc:
        raise LatencyRepeatError(
            f"matched selector pair {pair_id!r} is malformed: {exc}"
        ) from exc
    if (
        not math.isfinite(selected_target)
        or not 0.0 < selected_target <= 1.0
        or not math.isclose(cell_target, selected_target, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise LatencyRepeatError(
            f"matched pair target drifted for {pair_id!r}: "
            f"cell={cell_target!r}, selector={selected_target!r}"
        )
    if selected_row["dataset"] != str(cell.get("dataset") or ""):
        raise LatencyRepeatError(f"matched pair dataset drifted for {pair_id!r}")
    if cell.get("stock_config") != stock or cell.get("sqlens_config") != sqlens:
        raise LatencyRepeatError(
            f"matched pair configs drifted from selector for {pair_id!r}"
        )
    return stock, sqlens


def _matched_search_settings(
    cell: Mapping[str, Any],
    stock: Mapping[str, object],
    sqlens: Mapping[str, object],
) -> throughput.SearchSettings:
    pair_id = str(cell["pair_id"])
    target_recall = float(cell["target_recall"])
    mode_configs = cell.get("mode_configs")
    if not isinstance(mode_configs, Mapping):
        raise LatencyRepeatError(f"matched pair {pair_id!r} has no mode configs")

    def arm_settings(
        selected: Mapping[str, object],
        mode: str,
        *,
        guidance_enabled: bool,
    ) -> throughput.ArmSearchSettings:
        mode_config = mode_configs.get(mode)
        if not isinstance(mode_config, Mapping):
            raise LatencyRepeatError(
                f"matched pair {pair_id!r} has no mode config for {mode}"
            )
        selected_fields = (
            "ef_search",
            "iterative_scan",
            "max_scan_tuples",
            "scan_mem_multiplier",
            "guided_collect_target",
            "traversal_guided_target",
            "d2_page_access",
            "d2_index_page_access",
        )
        for field in selected_fields:
            if mode_config.get(field) != selected.get(field):
                raise LatencyRepeatError(
                    f"matched pair {pair_id!r} {mode}.{field} drifted "
                    "from the selector"
                )
        if mode_config.get("traversal_guided_prioritization") is not guidance_enabled:
            raise LatencyRepeatError(
                f"matched pair {pair_id!r} has invalid guidance mode for {mode}"
            )
        try:
            settings = throughput.ArmSearchSettings(
                ef_search=int(selected["ef_search"]),
                iterative_scan=str(selected["iterative_scan"]),
                max_scan_tuples=int(selected["max_scan_tuples"]),
                scan_mem_multiplier=float(selected["scan_mem_multiplier"]),
                guided_collect_target=int(selected["guided_collect_target"]),
                traversal_guided_target=int(selected["traversal_guided_target"]),
                traversal_guided_burst=int(mode_config["traversal_guided_burst"]),
                traversal_guided_early_stop=bool(
                    mode_config.get("traversal_guided_early_stop", False)
                ),
                traversal_guided_early_stop_distance_ratio=float(
                    mode_config.get(
                        "traversal_guided_early_stop_distance_ratio", 0.0
                    )
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LatencyRepeatError(
                f"matched pair {pair_id!r} has an invalid search config"
            ) from exc
        canonical = settings.mode_config(guidance_enabled=guidance_enabled)
        normalized_mode_config = dict(mode_config)
        normalized_mode_config.setdefault("traversal_guided_early_stop", False)
        normalized_mode_config.setdefault(
            "traversal_guided_early_stop_distance_ratio", 0.0
        )
        expected_keys = set(canonical) | {
            "d2_page_access",
            "d2_index_page_access",
        }
        if set(normalized_mode_config) != expected_keys or any(
            normalized_mode_config.get(key) != value
            for key, value in canonical.items()
        ):
            raise LatencyRepeatError(
                f"matched pair {pair_id!r} {mode} search config is not canonical"
            )
        return settings

    return throughput.SearchSettings(
        config_id=pair_id,
        pair_id=pair_id,
        target_recall=target_recall,
        stock=arm_settings(stock, "original", guidance_enabled=False),
        sqlens=arm_settings(
            sqlens,
            "design1_bloom_bfs_layout_d3",
            guidance_enabled=True,
        ),
    )


def convert_cell(
    cell: Mapping[str, Any],
    *,
    release_sha256: str,
    release: Mapping[str, Any],
) -> list[dict[str, object]]:
    dataset_name = str(cell.get("dataset") or "")
    dataset = DATASET_IDS.get(dataset_name)
    if dataset is None:
        raise LatencyRepeatError(f"unknown dataset in cell: {dataset_name!r}")
    if str(cell.get("phase")) != "measurement":
        raise LatencyRepeatError("latency repeat conversion accepts measurement cells only")
    raw = Path(str(cell.get("raw") or ""))
    plan_path = Path(str(cell.get("plan") or ""))
    if not raw.is_file() or not plan_path.is_file():
        raise LatencyRepeatError(f"cell inputs are missing: raw={raw}, plan={plan_path}")
    plan = read_json(plan_path)
    if plan.get("status") != "complete":
        raise LatencyRepeatError(f"cell plan is not complete: {plan_path}")
    if plan.get("output_sha256") != sha256_file(raw):
        raise LatencyRepeatError(f"cell raw SHA does not match plan: {raw}")
    if not _prewarm_valid(plan):
        raise LatencyRepeatError(f"cell lacks complete warm-cache evidence: {plan_path}")

    fields, rows = read_csv(raw)
    required_fields = {
        "mode",
        "repeat",
        "request_no",
        "query_id",
        "filter_name",
        "recall",
        "end_to_end_ms",
        "error",
        "sqlens_build_id",
        "vector_so_sha256",
    }
    _require_fields(fields, required_fields, raw)
    expected_rows = int(cell["expected_rows"])
    if len(rows) != expected_rows or int(plan.get("output_rows", -1)) != expected_rows:
        raise LatencyRepeatError(
            f"cell row count mismatch: expected={expected_rows}, observed={len(rows)}"
        )
    observed_modes = {row["mode"] for row in rows}
    if observed_modes != set(MODE_ARMS):
        raise LatencyRepeatError(
            f"cell must contain Stock and full SQLens only: {sorted(observed_modes)}"
        )
    expected_build = str(release["expected_sqlens_build_id"])
    expected_so = str(release["expected_vector_so_sha256"])
    if any(
        row["sqlens_build_id"] != expected_build
        or row["vector_so_sha256"] != expected_so
        for row in rows
    ):
        raise LatencyRepeatError(f"cell runtime release identity drifted: {raw}")

    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        try:
            grouped[(int(row["repeat"]), row["mode"])].append(row)
        except ValueError as exc:
            raise LatencyRepeatError(f"cell contains a non-integer repeat: {raw}") from exc
    repeats = int(cell["repeats"])
    requests = int(cell["requests"])
    if requests != artifact.EXPECTED_REQUESTS:
        raise LatencyRepeatError(
            f"formal latency cell requires q{artifact.EXPECTED_REQUESTS}, "
            f"observed={requests}"
        )
    if repeats < artifact.MIN_REPEATS["latency"]:
        raise LatencyRepeatError(
            "formal latency cell requires at least "
            f"{artifact.MIN_REPEATS['latency']} repeats, observed={repeats}"
        )
    expected_groups = {
        (repeat, mode)
        for repeat in range(repeats)
        for mode in MODE_ARMS
    }
    if set(grouped) != expected_groups:
        raise LatencyRepeatError(
            f"cell repeat/mode coverage differs from {sorted(expected_groups)}"
        )

    for repeat in range(repeats):
        signatures: dict[str, set[tuple[str, str, str]]] = {}
        for mode in MODE_ARMS:
            group = grouped[(repeat, mode)]
            signatures[mode] = {
                (row["request_no"], row["query_id"], row["filter_name"])
                for row in group
            }
        if signatures["original"] != signatures["design1_bloom_bfs_layout_d3"]:
            raise LatencyRepeatError(
                f"paired trace mismatch in repeat={repeat}: {raw}"
            )

    trace = cell.get("inputs", {}).get("workload", {})
    trace_sha256 = str(trace.get("sha256") or "")
    if len(trace_sha256) != 64:
        raise LatencyRepeatError(f"cell has no bound request trace SHA: {raw}")
    run_id = (
        f"figure5-r35-{dataset}-latency-"
        f"{sha256_file(raw)[:12]}-{str(cell['scan_family'])}-ef{int(cell['ef_search'])}"
    )
    output: list[dict[str, object]] = []
    for repeat in range(repeats):
        for mode, arm in MODE_ARMS.items():
            group = grouped[(repeat, mode)]
            request_numbers = [int(row["request_no"]) for row in group]
            query_ids = {row["query_id"] for row in group}
            errors = [row for row in group if row["error"].strip()]
            if len(group) != requests or len(set(request_numbers)) != requests:
                raise LatencyRepeatError(
                    f"{dataset}/{mode}/repeat={repeat} does not contain {requests} requests"
                )
            if len(query_ids) != requests:
                raise LatencyRepeatError(
                    f"{dataset}/{mode}/repeat={repeat} does not contain unique q10K"
                )
            if errors:
                raise LatencyRepeatError(
                    f"{dataset}/{mode}/repeat={repeat} has {len(errors)} errors"
                )
            recalls = [float(row["recall"]) for row in group]
            latencies = [float(row["end_to_end_ms"]) for row in group]
            if any(not math.isfinite(value) or value <= 0 for value in latencies):
                raise LatencyRepeatError("latency rows must be finite and positive")
            recall, recall_low, recall_high = mean_ci95(
                recalls, lower=0.0, upper=1.0
            )
            config_id, config_sha256 = _config_binding(
                cell, mode, release_sha256
            )
            output.append(
                {
                    "schema_version": artifact.SCHEMA_VERSION,
                    "run_id": run_id,
                    "dataset": dataset,
                    "experiment_kind": "latency",
                    "arm_id": arm,
                    "mode_id": mode,
                    "config_id": config_id,
                    "config_sha256": config_sha256,
                    "release_identity_sha256": release_sha256,
                    "clients": 1,
                    "repeat_id": repeat,
                    "request_trace_sha256": trace_sha256,
                    "requests": requests,
                    "unique_queries": len(query_ids),
                    "completed_queries": requests,
                    "error_count": 0,
                    "wall_clock_seconds": sum(latencies) / 1000.0,
                    "recall_mean": recall,
                    "recall_ci95_low": recall_low,
                    "recall_ci95_high": recall_high,
                    "latency_mean_ms": statistics.fmean(latencies),
                    "latency_p95_ms": percentile(latencies, 0.95),
                    "latency_p99_ms": percentile(latencies, 0.99),
                    "throughput_qps": "",
                    "throughput_ci95_low": "",
                    "throughput_ci95_high": "",
                    "throughput_source": "",
                    "status": "valid",
                }
            )
    return output


def convert_matched_cell(
    cell: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    release_sha256: str,
    release: Mapping[str, Any],
) -> list[dict[str, object]]:
    pair_id = str(cell.get("pair_id") or "").strip()
    dataset_name = str(cell.get("dataset") or "")
    dataset = DATASET_IDS.get(dataset_name)
    if not pair_id:
        raise LatencyRepeatError("matched cell has no pair_id")
    if dataset is None:
        raise LatencyRepeatError(f"unknown dataset in matched cell: {dataset_name!r}")
    if cell.get("status") != "complete":
        raise LatencyRepeatError(f"matched cell {pair_id!r} is incomplete")
    if (
        int(cell.get("expected_requests", -1)) != matched_latency.EXPECTED_REQUESTS
        or int(cell.get("expected_repeats", -1)) != matched_latency.EXPECTED_REPEATS
        or int(cell.get("expected_rows", -1)) != matched_latency.EXPECTED_ROWS
    ):
        raise LatencyRepeatError(
            f"matched cell {pair_id!r} does not bind formal q10k/r3"
        )

    raw = _matched_path(cell.get("raw"), f"{pair_id} raw CSV")
    plan_path = _matched_path(cell.get("plan"), f"{pair_id} plan")
    raw_sha256 = _require_sha256(
        cell.get("raw_sha256"), f"matched cell {pair_id} raw SHA"
    )
    plan_sha256 = _require_sha256(
        cell.get("plan_sha256"), f"matched cell {pair_id} plan SHA"
    )
    if sha256_file(raw) != raw_sha256:
        raise LatencyRepeatError(f"matched cell {pair_id!r} raw SHA drifted")
    if sha256_file(plan_path) != plan_sha256:
        raise LatencyRepeatError(f"matched cell {pair_id!r} plan SHA drifted")

    stock, sqlens = _matched_selector_pair(
        manifest, cell, release, release_sha256
    )
    settings = _matched_search_settings(cell, stock, sqlens)
    pair = matched_latency.SelectedPair(
        pair_id=pair_id,
        dataset=dataset_name,
        target_recall=float(cell["target_recall"]),
        stock=dict(stock),
        sqlens=dict(sqlens),
    )
    completion_config = {
        "release_identity": {
            "expected_sqlens_build_id": release["expected_sqlens_build_id"],
            "expected_vector_so_sha256": release["expected_vector_so_sha256"],
        }
    }
    if not matched_latency.cell_complete(
        raw, plan_path, pair, completion_config, cell
    ):
        raise LatencyRepeatError(
            f"matched cell {pair_id!r} fails the matched-latency completion gates"
        )
    predicate_completion = cell.get("predicate_completion")
    if (
        not isinstance(predicate_completion, Mapping)
        or int(predicate_completion.get("expected_predicate_count", -1))
        != matched_latency.EXPECTED_FORMAL_PREDICATES
        or int(predicate_completion.get("observed_predicate_count", -1))
        != matched_latency.EXPECTED_FORMAL_PREDICATES
        or predicate_completion.get("exact_coverage") is not True
        or not isinstance(predicate_completion.get("predicate_names"), list)
        or len(set(predicate_completion["predicate_names"]))
        != matched_latency.EXPECTED_FORMAL_PREDICATES
    ):
        raise LatencyRepeatError(
            f"matched cell {pair_id!r} lacks exact predicate completion evidence"
        )

    plan = read_json(plan_path)
    if (
        plan.get("status") != "complete"
        or int(plan.get("output_rows", -1)) != matched_latency.EXPECTED_ROWS
        or plan.get("output_sha256") != raw_sha256
    ):
        raise LatencyRepeatError(f"matched cell {pair_id!r} plan is incomplete")
    errors = plan.get("query_error_summary")
    if not isinstance(errors, Mapping) or int(errors.get("error_rows", -1)) != 0:
        raise LatencyRepeatError(
            f"matched cell {pair_id!r} plan reports query errors"
        )

    inputs = cell.get("input_bindings")
    workload_binding = (
        inputs.get("measurement_workload_csv")
        if isinstance(inputs, Mapping)
        else None
    )
    if not isinstance(workload_binding, Mapping):
        raise LatencyRepeatError(
            f"matched cell {pair_id!r} has no measurement workload binding"
        )
    workload_path, trace_sha256 = _audit_bound_file(
        workload_binding, f"{pair_id} measurement workload"
    )
    if (
        matched_latency.frontier.count_csv_rows(workload_path)
        != matched_latency.EXPECTED_REQUESTS
    ):
        raise LatencyRepeatError(
            f"matched cell {pair_id!r} workload is not q10k"
        )

    fields, rows = read_csv(raw)
    _require_fields(
        fields,
        {
            "mode",
            "repeat",
            "request_no",
            "query_id",
            "filter_name",
            "recall",
            "end_to_end_ms",
            "error",
            "sqlens_build_id",
            "vector_so_sha256",
        },
        raw,
    )
    if len(rows) != matched_latency.EXPECTED_ROWS:
        raise LatencyRepeatError(
            f"matched cell {pair_id!r} row count is not "
            f"{matched_latency.EXPECTED_ROWS}"
        )
    if {row["mode"] for row in rows} != set(MODE_ARMS):
        raise LatencyRepeatError(
            f"matched cell {pair_id!r} must contain Stock and full SQLens"
        )
    expected_build = str(release["expected_sqlens_build_id"])
    expected_so = str(release["expected_vector_so_sha256"])
    if any(
        row["sqlens_build_id"] != expected_build
        or row["vector_so_sha256"] != expected_so
        for row in rows
    ):
        raise LatencyRepeatError(
            f"matched cell {pair_id!r} runtime release identity drifted"
        )

    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    try:
        for row in rows:
            grouped[(int(row["repeat"]), row["mode"])].append(row)
    except ValueError as exc:
        raise LatencyRepeatError(
            f"matched cell {pair_id!r} has a non-integer repeat"
        ) from exc
    expected_groups = {
        (repeat, mode)
        for repeat in range(matched_latency.EXPECTED_REPEATS)
        for mode in MODE_ARMS
    }
    if set(grouped) != expected_groups:
        raise LatencyRepeatError(
            f"matched cell {pair_id!r} repeat/mode coverage is incomplete"
        )
    for mode in MODE_ARMS:
        combined_rows = [
            row
            for repeat in range(matched_latency.EXPECTED_REPEATS)
            for row in grouped[(repeat, mode)]
        ]
        combined_recall = query_cluster_bootstrap_recall(
            combined_rows,
            value_field="recall",
            seed_label=f"{pair_id}:{mode}:all-repeats",
        )
        if float(combined_recall["min_predicate_lcb95"]) < pair.target_recall:
            raise LatencyRepeatError(
                "matched q10k minimum-predicate cluster-bootstrap recall LCB "
                f"misses target for pair={pair_id!r}, mode={mode}: "
                f"lcb={float(combined_recall['min_predicate_lcb95']):.6f}, "
                f"target={pair.target_recall:.6f}"
            )

    run_id = (
        f"figure5-r35-{dataset}-latency-{raw_sha256[:12]}-"
        f"{sha256_json(pair_id)[:12]}"
    )
    output: list[dict[str, object]] = []
    for repeat in range(matched_latency.EXPECTED_REPEATS):
        signatures: dict[str, set[tuple[str, str, str]]] = {}
        for mode in MODE_ARMS:
            group = grouped[(repeat, mode)]
            signatures[mode] = {
                (row["request_no"], row["query_id"], row["filter_name"])
                for row in group
            }
        if signatures["original"] != signatures["design1_bloom_bfs_layout_d3"]:
            raise LatencyRepeatError(
                f"paired trace mismatch in matched pair={pair_id!r}, repeat={repeat}"
            )

        for mode, arm in MODE_ARMS.items():
            group = grouped[(repeat, mode)]
            request_numbers = [int(row["request_no"]) for row in group]
            query_ids = {row["query_id"] for row in group}
            if (
                len(group) != matched_latency.EXPECTED_REQUESTS
                or len(set(request_numbers)) != matched_latency.EXPECTED_REQUESTS
                or len(query_ids) != matched_latency.EXPECTED_REQUESTS
            ):
                raise LatencyRepeatError(
                    f"{pair_id}/{mode}/repeat={repeat} is not complete q10k"
                )
            if any(row["error"].strip() for row in group):
                raise LatencyRepeatError(
                    f"{pair_id}/{mode}/repeat={repeat} contains query errors"
                )
            try:
                latencies = [float(row["end_to_end_ms"]) for row in group]
            except ValueError as exc:
                raise LatencyRepeatError(
                    f"{pair_id}/{mode}/repeat={repeat} has non-numeric metrics"
                ) from exc
            if any(
                not math.isfinite(value) or value <= 0 for value in latencies
            ):
                raise LatencyRepeatError(
                    "matched latency rows must be finite and positive"
                )
            recall_stats = query_cluster_bootstrap_recall(
                group,
                value_field="recall",
                seed_label=f"{pair_id}:{mode}:repeat={repeat}",
            )
            output.append(
                {
                    "schema_version": artifact.SCHEMA_VERSION,
                    "run_id": run_id,
                    "dataset": dataset,
                    "experiment_kind": "latency",
                    "arm_id": arm,
                    "mode_id": mode,
                    "config_id": pair_id,
                    # This is intentionally the same per-arm search-config
                    # binding used by the formal throughput runner.
                    "config_sha256": throughput.arm_config_sha256(settings, arm),
                    "release_identity_sha256": release_sha256,
                    "clients": 1,
                    "repeat_id": repeat,
                    "request_trace_sha256": trace_sha256,
                    "requests": matched_latency.EXPECTED_REQUESTS,
                    "unique_queries": len(query_ids),
                    "completed_queries": matched_latency.EXPECTED_REQUESTS,
                    "error_count": 0,
                    "wall_clock_seconds": sum(latencies) / 1000.0,
                    "recall_mean": recall_stats["mean"],
                    "recall_ci95_low": recall_stats["lower"],
                    "recall_ci95_high": recall_stats["upper"],
                    "latency_mean_ms": statistics.fmean(latencies),
                    "latency_p95_ms": percentile(latencies, 0.95),
                    "latency_p99_ms": percentile(latencies, 0.99),
                    "throughput_qps": "",
                    "throughput_ci95_low": "",
                    "throughput_ci95_high": "",
                    "throughput_source": "",
                    "status": "valid",
                }
            )
    return output


def convert_matched_manifest(
    manifest: Mapping[str, Any],
) -> list[dict[str, object]]:
    if (
        manifest.get("status") != "complete"
        or manifest.get("artifact_valid") is not True
        or manifest.get("requested_slice_complete") is not True
        or manifest.get("full_release_complete") is not True
        or manifest.get("paper_eligible") is not True
    ):
        raise LatencyRepeatError(
            "matched run manifest is incomplete or not a full release"
        )
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping) or (
        int(execution.get("requests", -1)) != matched_latency.EXPECTED_REQUESTS
        or int(execution.get("repeats", -1)) != matched_latency.EXPECTED_REPEATS
        or int(execution.get("expected_rows_per_pair", -1))
        != matched_latency.EXPECTED_ROWS
        or int(execution.get("expected_predicate_count", -1))
        != matched_latency.EXPECTED_FORMAL_PREDICATES
        or execution.get("execution_order") != "paired_interleaved"
    ):
        raise LatencyRepeatError("matched run manifest does not bind q10k/r3")

    frontier_config = manifest.get("frontier_config")
    if not isinstance(frontier_config, Mapping):
        raise LatencyRepeatError("matched run manifest has no frontier config")
    _audit_bound_file(frontier_config, "frontier config")
    release, release_sha256 = _matched_release(manifest)
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise LatencyRepeatError("matched run manifest has no scheduled pairs")
    if (
        int(manifest.get("pairs_total", -1)) != len(schedule)
        or int(manifest.get("pairs_complete", -1)) != len(schedule)
    ):
        raise LatencyRepeatError("matched run manifest pair counts are incomplete")

    rows: list[dict[str, object]] = []
    pair_ids: set[str] = set()
    for cell in schedule:
        if not isinstance(cell, Mapping):
            raise LatencyRepeatError("matched run manifest has a malformed pair")
        pair_id = str(cell.get("pair_id") or "")
        if pair_id in pair_ids:
            raise LatencyRepeatError(
                f"matched run manifest repeats pair_id={pair_id!r}"
            )
        pair_ids.add(pair_id)
        rows.extend(
            convert_matched_cell(
                cell,
                manifest=manifest,
                release_sha256=release_sha256,
                release=release,
            )
        )
    return rows


def convert_manifest(manifest_path: Path) -> list[dict[str, object]]:
    manifest = read_json(manifest_path)
    artifact_type = manifest.get("artifact_type")
    if artifact_type == "sqlens_figure5_matched_latency_run":
        return convert_matched_manifest(manifest)
    if artifact_type != "sqlens_figure5_frontier_run":
        raise LatencyRepeatError("input is not a Figure 5 run manifest")
    if manifest.get("phase") != "measurement":
        raise LatencyRepeatError("input run manifest is not a measurement run")
    release = manifest.get("release_contract")
    if not isinstance(release, Mapping):
        raise LatencyRepeatError("run manifest has no release contract")
    release_sha256 = str(release.get("sha256") or "")
    if len(release_sha256) != 64:
        raise LatencyRepeatError("run manifest has no release contract SHA")
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise LatencyRepeatError("run manifest has no scheduled cells")
    rows: list[dict[str, object]] = []
    for cell in schedule:
        if not isinstance(cell, Mapping):
            raise LatencyRepeatError("run manifest contains a malformed cell")
        if cell.get("status") != "complete":
            raise LatencyRepeatError(
                f"run manifest contains incomplete cell: {cell.get('raw')}"
            )
        rows.extend(
            convert_cell(cell, release_sha256=release_sha256, release=release)
        )
    return rows


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                *artifact.REPEAT_FIELDS,
                *sorted(artifact.OPTIONAL_PROVENANCE_FIELDS),
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build formal Figure 5 repeat-level latency evidence."
    )
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--binding-manifest",
        type=Path,
        help=(
            "Converter sidecar path; defaults to "
            "<out>.manifest.json."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        source_manifest = args.run_manifest.resolve()
        output = args.out.resolve()
        _, release, source_sha = converter_binding.audited_run_manifest(
            source_manifest,
            expected_artifact_type="sqlens_figure5_matched_latency_run",
        )
        rows = convert_manifest(source_manifest)
        provenance = converter_binding.row_provenance(
            release, source_manifest, source_sha
        )
        if any(
            str(row.get("release_identity_sha256") or "")
            != release["sha256"]
            for row in rows
        ):
            raise LatencyRepeatError(
                "converted rows do not match the audited release contract"
            )
        for row in rows:
            row.update(provenance)
        write_rows(output, rows)
        binding_path = (
            args.binding_manifest.resolve()
            if args.binding_manifest
            else output.with_suffix(output.suffix + ".manifest.json")
        )
        converter_binding.publish_converter_binding(
            kind="latency",
            source_manifest=source_manifest,
            source_sha256=source_sha,
            release=release,
            output=output,
            rows=len(rows),
            converter_source=Path(__file__),
            binding_path=binding_path,
        )
    except (
        converter_binding.ConverterBindingError,
        LatencyRepeatError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        f"wrote {output} rows={len(rows)} binding={binding_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
