#!/usr/bin/env python3
"""Build a deterministic inventory of r41 matched-recall evidence artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PLAN_SUFFIX = ".csv.plan.json"
MANIFEST_SUFFIX = ".manifest.json"
COMPLETE_STATUSES = {"complete", "completed", "success", "valid"}
SETTING_FIELDS = (
    "effective_ef_search",
    "ef_search",
    "max_scan_tuples",
    "scan_mem_multiplier",
    "iterative_scan",
    "guided_collect_target",
    "traversal_guided_target",
    "traversal_guided_prioritization",
    "traversal_guided_burst",
)
CSV_FIELDS = (
    "artifact_path",
    "artifact_kind",
    "artifact_sha256",
    "status",
    "classification",
    "reasons",
    "release_id",
    "release_generation",
    "build_id",
    "vector_so_sha256",
    "git_sha",
    "workload_requests",
    "workload_unique_queries",
    "workload_tier",
    "modes",
    "mode_count",
    "error_count",
    "settings_scope",
    "settings_source",
    "settings_json",
    "output_path",
    "output_rows",
    "output_sha256",
    "artifact_valid",
    "paper_eligible",
)


class InventoryError(ValueError):
    """Raised when an input cannot be safely interpreted."""


@dataclass(frozen=True)
class InventoryRecord:
    artifact_path: str
    artifact_kind: str
    artifact_sha256: str
    status: str
    classification: str
    reasons: tuple[str, ...]
    release_id: str | None
    release_generation: str | None
    build_id: str | None
    vector_so_sha256: str | None
    git_sha: str | None
    workload_requests: int | None
    workload_unique_queries: int | None
    workload_tier: str
    modes: tuple[str, ...]
    mode_count: int
    error_count: int | None
    settings_scope: str
    settings_source: str | None
    settings: Mapping[str, Any]
    output_path: str | None
    output_rows: int | None
    output_sha256: str | None
    artifact_valid: bool | None
    paper_eligible: bool | None

    def json_value(self) -> dict[str, Any]:
        return asdict(self)

    def csv_value(self) -> dict[str, Any]:
        value = self.json_value()
        value["reasons"] = "; ".join(self.reasons)
        value["modes"] = ";".join(self.modes)
        value["settings_json"] = canonical_json(self.settings)
        del value["settings"]
        for key in ("artifact_valid", "paper_eligible"):
            if value[key] is None:
                value[key] = ""
            else:
                value[key] = str(value[key]).lower()
        for key, item in tuple(value.items()):
            if item is None:
                value[key] = ""
        return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise InventoryError(f"{label} must be a JSON object")
    return value


def optional_bool(payload: Mapping[str, Any], key: str, label: str) -> bool | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, bool):
        raise InventoryError(f"{label}.{key} must be a boolean or null")
    return value


def optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise InventoryError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise InventoryError(f"{label} must be a non-negative integer") from exc
    if result < 0 or (isinstance(value, float) and value != result):
        raise InventoryError(f"{label} must be a non-negative integer")
    return result


def nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def release_generation(*values: str | None) -> str | None:
    generations: set[str] = set()
    for value in values:
        if not value:
            continue
        generations.update(
            f"r{match.group(1)}"
            for match in re.finditer(r"(?:^|[-_])r(\d+)(?=$|[-_])", value, re.IGNORECASE)
        )
    return sorted(generations)[-1] if generations else None


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryError(f"malformed JSON in {path}: {exc}") from exc
    return require_mapping(value, str(path))


def is_throughput_manifest(payload: Mapping[str, Any]) -> bool:
    artifact_type = str(payload.get("artifact_type", "")).lower()
    protocol = payload.get("protocol")
    outputs = payload.get("outputs")
    return (
        "throughput" in artifact_type
        or (
            isinstance(protocol, dict)
            and any(key in protocol for key in ("throughput_formula", "throughput_source"))
        )
        or (
            isinstance(outputs, dict)
            and "requests" in outputs
            and "repeats" in outputs
            and isinstance(payload.get("methods"), dict)
        )
    )


def discover_artifacts(inputs: Sequence[Path]) -> list[tuple[Path, str]]:
    found: dict[Path, str] = {}
    for supplied in inputs:
        path = supplied.expanduser()
        if not path.exists():
            raise InventoryError(f"artifact path does not exist: {path}")
        if path.is_file():
            if path.name.endswith(PLAN_SUFFIX):
                found[path.resolve()] = "benchmark_plan"
            elif path.name.endswith(MANIFEST_SUFFIX):
                payload = load_json(path)
                if not is_throughput_manifest(payload):
                    raise InventoryError(f"not a throughput manifest: {path}")
                found[path.resolve()] = "throughput_manifest"
            else:
                raise InventoryError(
                    f"unsupported artifact file {path}; expected *{PLAN_SUFFIX} or *{MANIFEST_SUFFIX}"
                )
            continue
        if not path.is_dir():
            raise InventoryError(f"artifact path is neither a file nor directory: {path}")
        for candidate in sorted(path.rglob(f"*{PLAN_SUFFIX}")):
            found[candidate.resolve()] = "benchmark_plan"
        for candidate in sorted(path.rglob(f"*{MANIFEST_SUFFIX}")):
            payload = load_json(candidate)
            if is_throughput_manifest(payload):
                found[candidate.resolve()] = "throughput_manifest"
    if not found:
        raise InventoryError("no benchmark plans or throughput manifests found")
    return sorted(found.items(), key=lambda item: str(item[0]))


def resolve_reference(reference: str, owner: Path) -> Path:
    path = Path(reference).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, owner.parent / path]
    candidates.extend(parent / path for parent in owner.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (owner.parent / path).resolve()


def parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"[-+]?\d+", stripped):
        return int(stripped)
    try:
        return float(stripped)
    except ValueError:
        return stripped


def normalize_config(row: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if row.get("effective_ef_search") not in (None, ""):
        config["ef_search"] = parse_scalar(str(row["effective_ef_search"]))
    elif row.get("ef_search") not in (None, ""):
        config["ef_search"] = parse_scalar(str(row["ef_search"]))
    for field in SETTING_FIELDS[2:]:
        if row.get(field) not in (None, ""):
            config[field] = parse_scalar(str(row[field]))
    return config


def settings_from_groups(
    grouped: Mapping[str, Mapping[str, set[str]]],
) -> tuple[str, dict[str, Any], list[str]]:
    if not grouped:
        return "unknown", {}, ["search settings could not be inferred"]
    inconsistent = [
        f"{mode}/{filter_name}"
        for mode, filters in grouped.items()
        for filter_name, configs in filters.items()
        if len(configs) != 1
    ]
    if inconsistent:
        return (
            "unknown",
            {},
            ["settings vary within mode/filter cells: " + ", ".join(sorted(inconsistent))],
        )
    decoded = {
        mode: {
            filter_name: json.loads(next(iter(configs)))
            for filter_name, configs in sorted(filters.items())
        }
        for mode, filters in sorted(grouped.items())
    }
    filter_names = {filter_name for filters in decoded.values() for filter_name in filters}
    if len(filter_names) < 2:
        return "unknown", decoded, ["only one filter is present; setting scope is not inferable"]
    per_filter = any(
        len({canonical_json(config) for config in filters.values()}) > 1
        for filters in decoded.values()
    )
    if per_filter:
        return "per_filter", decoded, []
    return (
        "global",
        {mode: next(iter(filters.values())) for mode, filters in decoded.items()},
        [],
    )


def inspect_benchmark_csv(
    path: Path,
) -> tuple[int, tuple[str, ...], int, str, dict[str, Any], list[str]]:
    grouped: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    modes: set[str] = set()
    rows = 0
    errors = 0
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "mode" not in reader.fieldnames:
                raise InventoryError(f"benchmark CSV lacks a mode column: {path}")
            for row in reader:
                rows += 1
                mode = str(row.get("mode", "")).strip()
                if not mode:
                    raise InventoryError(f"benchmark CSV has an empty mode at row {rows + 1}: {path}")
                modes.add(mode)
                if str(row.get("error", "")).strip():
                    errors += 1
                config = normalize_config(row)
                if config:
                    filter_name = str(row.get("filter_name", "")).strip() or "<unspecified>"
                    grouped[mode][filter_name].add(canonical_json(config))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InventoryError(f"malformed benchmark CSV {path}: {exc}") from exc
    scope, settings, issues = settings_from_groups(grouped)
    return rows, tuple(sorted(modes)), errors, scope, settings, issues


def settings_from_checks(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, ...], str, dict[str, Any], list[str]]:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return (), "unknown", {}, ["no benchmark CSV or checks available for modes/settings"]
    grouped: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    modes: set[str] = set()
    for index, item in enumerate(checks):
        if not isinstance(item, dict):
            raise InventoryError(f"checks[{index}] must be an object")
        mode = str(item.get("mode", "")).strip()
        if not mode:
            continue
        modes.add(mode)
        config_value = item.get("config")
        if isinstance(config_value, dict) and config_value:
            filter_name = str(item.get("filter_name", "")).strip() or "<unspecified>"
            grouped[mode][filter_name].add(canonical_json(normalize_config(config_value)))
    scope, settings, issues = settings_from_groups(grouped)
    return tuple(sorted(modes)), scope, settings, issues


def configured_search_settings(
    payload: Mapping[str, Any],
) -> tuple[str, dict[str, Any], list[str]] | None:
    value = payload.get("search_configuration")
    if value is None:
        return None
    config = require_mapping(value, "search_configuration")
    scope = str(config.get("configured_scope") or "").strip()
    if scope not in {"global_policy", "per_filter"}:
        raise InventoryError(
            "search_configuration.configured_scope must be global_policy or per_filter"
        )
    mode_defaults = config.get("mode_defaults")
    ef_overrides = config.get("filter_ef_search_overrides")
    target_overrides = config.get("filter_traversal_target_overrides")
    bypass = config.get("guidance_bypass_policy")
    for label, item in (
        ("mode_defaults", mode_defaults),
        ("filter_ef_search_overrides", ef_overrides),
        ("filter_traversal_target_overrides", target_overrides),
        ("guidance_bypass_policy", bypass),
    ):
        if not isinstance(item, dict):
            raise InventoryError(f"search_configuration.{label} must be an object")
    has_overrides = bool(ef_overrides or target_overrides)
    if (scope == "per_filter") != has_overrides:
        raise InventoryError(
            "search_configuration scope disagrees with its per-filter overrides"
        )
    settings = {
        "mode_defaults": mode_defaults,
        "filter_ef_search_overrides": ef_overrides,
        "filter_traversal_target_overrides": target_overrides,
        "guidance_bypass_policy": bypass,
    }
    return scope, settings, []


def identity_fields(payload: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind == "benchmark_plan":
        final = nested(payload, "sqlens_runtime_identity_final") or {}
        startup = nested(payload, "sqlens_runtime_identity_startup") or {}
        contract = nested(payload, "release_contract") or {}
    else:
        final = nested(payload, "evidence", "runtime_binary_identity_end") or {}
        startup = nested(payload, "evidence", "runtime_binary_identity_start") or {}
        contract = nested(payload, "release_contract") or {}
    for label, value in (("runtime final identity", final), ("runtime startup identity", startup), ("release contract", contract)):
        if not isinstance(value, dict):
            raise InventoryError(f"{label} must be an object")
    expected_build = first_string(
        final.get("expected_build_id"), startup.get("expected_build_id"),
        contract.get("expected_sqlens_build_id"),
    )
    observed_build = first_string(
        final.get("observed_build_id"), startup.get("observed_build_id"),
        nested(payload, "evidence", "database_end", "sqlens_build_id"),
    )
    expected_sha = first_string(
        final.get("expected_vector_so_sha256"), startup.get("expected_vector_so_sha256"),
        contract.get("expected_vector_so_sha256"),
    )
    observed_sha = first_string(
        final.get("observed_vector_so_sha256"), startup.get("observed_vector_so_sha256"),
    )
    mismatches: list[str] = []
    if expected_build and observed_build and expected_build != observed_build:
        mismatches.append("expected and observed build IDs differ")
    if expected_sha and observed_sha and expected_sha != observed_sha:
        mismatches.append("expected and observed vector.so SHA-256 values differ")
    for label, value in (("startup", startup), ("final", final)):
        if value.get("exact_match") is False:
            mismatches.append(f"{label} runtime identity reports exact_match=false")
    release_id = first_string(contract.get("contract_id"), payload.get("release_id"))
    build_id = observed_build or expected_build
    vector_sha = observed_sha or expected_sha
    generation = release_generation(build_id, release_id)
    git_sha = first_string(
        payload.get("git_sha"), payload.get("commit_sha"),
        contract.get("git_sha"), contract.get("commit_sha"),
    )
    return {
        "release_id": release_id,
        "release_generation": generation,
        "build_id": build_id,
        "vector_so_sha256": vector_sha,
        "git_sha": git_sha,
        "issues": mismatches,
    }


def workload_tier(requests: int | None) -> str:
    if requests is None:
        return "unknown"
    if requests >= 10000:
        return "q10k"
    if requests >= 5000:
        return "q5k"
    return "sub_q5k"


def classify(
    *,
    status: str,
    artifact_valid: bool | None,
    paper_eligible: bool | None,
    workload: int | None,
    modes: Sequence[str],
    errors: int | None,
    settings_scope: str,
    identity: Mapping[str, Any],
    integrity_issues: Sequence[str],
    inference_issues: Sequence[str],
    expedited_reasons: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    blockers: list[str] = []
    if status.lower() not in COMPLETE_STATUSES:
        blockers.append(f"artifact status is {status}, not complete")
    if artifact_valid is False:
        blockers.append("artifact_valid=false")
    if paper_eligible is False:
        blockers.append("paper_eligible=false")
    if errors is None:
        blockers.append("query error count is unavailable")
    elif errors:
        blockers.append(f"contains {errors} query errors")
    if workload is None:
        blockers.append("workload size is unavailable")
    elif workload < 5000:
        blockers.append(f"workload has only {workload} requests per arm (<5000)")
    if len(modes) < 2:
        blockers.append(f"only {len(modes)} mode(s) are evidenced")
    if not identity.get("build_id"):
        blockers.append("runtime build ID is unavailable")
    if not identity.get("vector_so_sha256"):
        blockers.append("vector.so SHA-256 is unavailable")
    if identity.get("release_generation") != "r41":
        observed = identity.get("release_generation") or "unknown"
        blockers.append(f"release generation is {observed}, not r41")
    if settings_scope == "unknown":
        blockers.extend(inference_issues or ["search setting scope is unknown"])
    blockers.extend(identity.get("issues", ()))
    blockers.extend(integrity_issues)
    if blockers:
        return "diagnostic", tuple(dict.fromkeys(blockers))
    if workload_tier(workload) == "q5k":
        return (
            "expedited",
            tuple(dict.fromkeys(["q5k workload is expedited evidence (<q10k)", *expedited_reasons])),
        )
    if expedited_reasons:
        return "expedited", tuple(dict.fromkeys(expedited_reasons))
    return (
        "formal_candidate",
        (
            "complete q10k-or-larger artifact with zero query errors",
            "r41 runtime build and vector.so SHA-256 are recorded",
            f"paired modes with {settings_scope} search settings are evidenced",
        ),
    )


def inspect_output_binding(
    owner: Path, binding: Mapping[str, Any], label: str
) -> tuple[Path | None, int | None, list[str]]:
    reference = binding.get("path")
    if not isinstance(reference, str) or not reference:
        return None, None, [f"{label} output path is unavailable"]
    path = resolve_reference(reference, owner)
    if not path.is_file():
        return path, None, [f"{label} output is missing: {path}"]
    issues: list[str] = []
    expected_sha = binding.get("sha256")
    if isinstance(expected_sha, str) and expected_sha and sha256_file(path) != expected_sha:
        issues.append(f"{label} output SHA-256 mismatch")
    rows = 0
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.reader(source)
            next(reader, None)
            rows = sum(1 for _ in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InventoryError(f"malformed {label} CSV {path}: {exc}") from exc
    expected_rows = optional_nonnegative_int(binding.get("rows"), f"{label}.rows")
    if expected_rows is not None and expected_rows != rows:
        issues.append(f"{label} output row count mismatch ({rows} != {expected_rows})")
    return path, rows, issues


def benchmark_record(path: Path, payload: Mapping[str, Any]) -> InventoryRecord:
    status_value = payload.get("status")
    if not isinstance(status_value, str) or not status_value.strip():
        raise InventoryError(f"benchmark plan status must be a non-empty string: {path}")
    status = status_value.strip().lower()
    for key in ("query_contract", "checks", "execution_sources"):
        value = payload.get(key)
        if value is not None and not isinstance(value, (dict, list)):
            raise InventoryError(f"{path}: {key} has an invalid type")
    if not any(key in payload for key in ("query_contract", "checks", "execution_sources")):
        raise InventoryError(f"not a benchmark plan (missing structural evidence): {path}")
    contract_value = payload.get("query_contract") or {}
    contract = require_mapping(contract_value, f"{path}.query_contract")
    workload = optional_nonnegative_int(
        contract.get("workload_requests", contract.get("expected_workload_requests")),
        f"{path}.query_contract.workload_requests",
    )
    unique = optional_nonnegative_int(
        contract.get("workload_unique_queries"),
        f"{path}.query_contract.workload_unique_queries",
    )
    summary_value = payload.get("query_error_summary") or {}
    summary = require_mapping(summary_value, f"{path}.query_error_summary")
    declared_errors = optional_nonnegative_int(
        summary.get("error_rows"), f"{path}.query_error_summary.error_rows"
    )
    output_reference = payload.get("output")
    output_path: Path | None = None
    output_rows: int | None = None
    output_sha = first_string(payload.get("output_sha256"))
    modes: tuple[str, ...] = ()
    settings_scope = "unknown"
    settings: dict[str, Any] = {}
    settings_source: str | None = None
    inference_issues: list[str] = []
    integrity_issues: list[str] = []
    csv_errors: int | None = None
    if output_reference is not None and not isinstance(output_reference, str):
        raise InventoryError(f"{path}.output must be a string or null")
    if isinstance(output_reference, str) and output_reference:
        output_path = resolve_reference(output_reference, path)
        if output_path.is_file():
            output_rows, modes, csv_errors, settings_scope, settings, inference_issues = (
                inspect_benchmark_csv(output_path)
            )
            settings_source = "benchmark_csv"
            declared_rows = optional_nonnegative_int(payload.get("output_rows"), f"{path}.output_rows")
            if declared_rows is not None and declared_rows != output_rows:
                integrity_issues.append(
                    f"benchmark output row count mismatch ({output_rows} != {declared_rows})"
                )
            if output_sha and sha256_file(output_path) != output_sha:
                integrity_issues.append("benchmark output SHA-256 mismatch")
        elif status in COMPLETE_STATUSES:
            integrity_issues.append(f"benchmark output is missing: {output_path}")
    elif status in COMPLETE_STATUSES:
        integrity_issues.append("complete benchmark plan has no output binding")
    if not modes:
        modes, settings_scope, settings, inference_issues = settings_from_checks(payload)
        settings_source = "plan_checks" if modes else None
    configured = configured_search_settings(payload)
    if configured is not None:
        settings_scope, settings, inference_issues = configured
        settings_source = "plan_search_configuration"
    elif settings_scope == "per_filter":
        settings_scope = "unknown"
        inference_issues = [
            "effective settings vary by filter, but configured tuning scope is not persisted"
        ]
    errors = declared_errors if declared_errors is not None else csv_errors
    if declared_errors is not None and csv_errors is not None and declared_errors != csv_errors:
        integrity_issues.append(
            f"query error count mismatch ({csv_errors} CSV != {declared_errors} plan)"
        )
    checks = payload.get("checks")
    if isinstance(checks, list):
        failed = sum(isinstance(item, dict) and item.get("passed") is False for item in checks)
        if failed:
            integrity_issues.append(f"{failed} benchmark planner/provenance checks failed")
    identity = identity_fields(payload, "benchmark_plan")
    classification, reasons = classify(
        status=status,
        artifact_valid=None,
        paper_eligible=None,
        workload=workload,
        modes=modes,
        errors=errors,
        settings_scope=settings_scope,
        identity=identity,
        integrity_issues=integrity_issues,
        inference_issues=inference_issues,
        expedited_reasons=(),
    )
    return InventoryRecord(
        artifact_path=str(path), artifact_kind="benchmark_plan",
        artifact_sha256=sha256_file(path), status=status, classification=classification,
        reasons=reasons, release_id=identity["release_id"],
        release_generation=identity["release_generation"], build_id=identity["build_id"],
        vector_so_sha256=identity["vector_so_sha256"], git_sha=identity["git_sha"],
        workload_requests=workload, workload_unique_queries=unique,
        workload_tier=workload_tier(workload), modes=modes, mode_count=len(modes),
        error_count=errors, settings_scope=settings_scope, settings_source=settings_source,
        settings=settings, output_path=str(output_path) if output_path else None,
        output_rows=output_rows, output_sha256=output_sha,
        artifact_valid=None, paper_eligible=None,
    )


def throughput_settings(
    payload: Mapping[str, Any]
) -> tuple[tuple[str, ...], str, dict[str, Any], list[str]]:
    value = nested(payload, "configuration", "value")
    value = value if isinstance(value, dict) else {}
    modes_value = value.get("modes")
    modes: set[str] = {
        str(item) for item in modes_value if isinstance(item, str) and item
    } if isinstance(modes_value, list) else set()
    search = value.get("search")
    settings: dict[str, Any] = {}
    if isinstance(search, dict) and search:
        for mode, config in search.items():
            if isinstance(config, dict):
                modes.add(str(mode))
                settings[str(mode)] = dict(sorted(config.items()))
    methods = payload.get("methods")
    if isinstance(methods, dict):
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue
            mode = first_string(details.get("mode_id"), method)
            if mode:
                modes.add(mode)
                method_search = details.get("search")
                if isinstance(method_search, dict) and mode not in settings:
                    settings[mode] = dict(sorted(method_search.items()))
    if not settings:
        return tuple(sorted(modes)), "unknown", {}, ["throughput search settings are unavailable"]
    per_filter = any(
        isinstance(config, dict)
        and any("filter" in key.lower() and isinstance(item, dict) for key, item in config.items())
        for config in settings.values()
    )
    return tuple(sorted(modes)), "per_filter" if per_filter else "global", settings, []


def throughput_record(path: Path, payload: Mapping[str, Any]) -> InventoryRecord:
    if not is_throughput_manifest(payload):
        raise InventoryError(f"not a throughput manifest: {path}")
    artifact_valid = optional_bool(payload, "artifact_valid", str(path))
    paper_eligible = optional_bool(payload, "paper_eligible", str(path))
    status_value = payload.get("status")
    if status_value is not None and not isinstance(status_value, str):
        raise InventoryError(f"{path}.status must be a string or null")
    status = (
        status_value.strip().lower()
        if isinstance(status_value, str) and status_value.strip()
        else "complete" if artifact_valid is True else "incomplete"
    )
    protocol_value = payload.get("protocol") or {}
    protocol = require_mapping(protocol_value, f"{path}.protocol")
    workload = optional_nonnegative_int(
        protocol.get(
            "unique_queries_per_arm_repeat",
            protocol.get("requests_per_arm_repeat", payload.get("requests_per_arm")),
        ),
        f"{path}.protocol.requests_per_arm_repeat",
    )
    modes, settings_scope, settings, inference_issues = throughput_settings(payload)
    identity = identity_fields(payload, "throughput_manifest")
    errors = optional_nonnegative_int(
        payload.get("query_errors", payload.get("error_count")), f"{path}.query_errors"
    )
    integrity_issues: list[str] = []
    output_path: Path | None = None
    output_rows: int | None = None
    output_sha: str | None = None
    outputs_value = payload.get("outputs") or {}
    outputs = require_mapping(outputs_value, f"{path}.outputs")
    repeats_binding = outputs.get("repeats")
    if isinstance(repeats_binding, dict):
        output_path, output_rows, issues = inspect_output_binding(path, repeats_binding, "repeats")
        integrity_issues.extend(issues)
        output_sha = first_string(repeats_binding.get("sha256"))
        if output_path and output_path.is_file():
            total_errors = 0
            found_error_column = False
            try:
                with output_path.open(newline="", encoding="utf-8") as source:
                    for row in csv.DictReader(source):
                        if "error_count" in row and row["error_count"] not in (None, ""):
                            found_error_column = True
                            total_errors += optional_nonnegative_int(
                                row["error_count"], f"{output_path}.error_count"
                            ) or 0
            except (OSError, UnicodeError, csv.Error) as exc:
                raise InventoryError(f"malformed repeats CSV {output_path}: {exc}") from exc
            if found_error_column:
                if errors is not None and errors != total_errors:
                    integrity_issues.append(
                        f"query error count mismatch ({total_errors} repeats CSV != {errors} manifest)"
                    )
                errors = total_errors if errors is None else errors
    elif repeats_binding is not None:
        raise InventoryError(f"{path}.outputs.repeats must be an object")
    elif status in COMPLETE_STATUSES:
        integrity_issues.append("complete throughput manifest has no repeats output binding")
    gates = payload.get("gates")
    expedited_reasons: list[str] = []
    if isinstance(gates, dict):
        failed = sorted(key for key, value in gates.items() if value is False)
        for key in failed:
            if key == "minimum_six_repeats" and gates.get("single_pass_override") is True:
                expedited_reasons.append("throughput uses a single-pass override (<6 repeats)")
            else:
                integrity_issues.append(f"throughput gate {key}=false")
    elif gates is not None:
        raise InventoryError(f"{path}.gates must be an object")
    repeats = optional_nonnegative_int(protocol.get("repeats"), f"{path}.protocol.repeats")
    if repeats is not None and repeats < 3 and not expedited_reasons:
        expedited_reasons.append(f"throughput has only {repeats} repeat(s)")
    classification, reasons = classify(
        status=status,
        artifact_valid=artifact_valid,
        paper_eligible=paper_eligible,
        workload=workload,
        modes=modes,
        errors=errors,
        settings_scope=settings_scope,
        identity=identity,
        integrity_issues=integrity_issues,
        inference_issues=inference_issues,
        expedited_reasons=expedited_reasons,
    )
    return InventoryRecord(
        artifact_path=str(path), artifact_kind="throughput_manifest",
        artifact_sha256=sha256_file(path), status=status, classification=classification,
        reasons=reasons, release_id=identity["release_id"],
        release_generation=identity["release_generation"], build_id=identity["build_id"],
        vector_so_sha256=identity["vector_so_sha256"], git_sha=identity["git_sha"],
        workload_requests=workload, workload_unique_queries=workload,
        workload_tier=workload_tier(workload), modes=modes, mode_count=len(modes),
        error_count=errors, settings_scope=settings_scope,
        settings_source="throughput_manifest", settings=settings,
        output_path=str(output_path) if output_path else None, output_rows=output_rows,
        output_sha256=output_sha, artifact_valid=artifact_valid,
        paper_eligible=paper_eligible,
    )


def build_inventory(artifacts: Sequence[Path]) -> list[InventoryRecord]:
    records: list[InventoryRecord] = []
    for path, kind in discover_artifacts(artifacts):
        payload = load_json(path)
        if kind == "benchmark_plan":
            records.append(benchmark_record(path, payload))
        else:
            records.append(throughput_record(path, payload))
    return records


def output_paths(prefix: Path) -> tuple[Path, Path]:
    return Path(str(prefix) + ".csv"), Path(str(prefix) + ".json")


def write_inventory(records: Sequence[InventoryRecord], prefix: Path) -> tuple[Path, Path]:
    csv_path, json_path = output_paths(prefix)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(record.csv_value() for record in records)
    counts = Counter(record.classification for record in records)
    payload = {
        "schema_version": 1,
        "generator": "build_matched_recall_evidence_inventory.py",
        "release_scope": "r41",
        "artifact_count": len(records),
        "classification_counts": dict(sorted(counts.items())),
        "artifacts": [record.json_value() for record in records],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", action="append", required=True, type=Path,
        help="Artifact sidecar or directory to scan; repeat for multiple roots.",
    )
    parser.add_argument(
        "--out-prefix", required=True, type=Path,
        help="Output prefix; writes <prefix>.csv and <prefix>.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records = build_inventory(args.artifact)
        csv_path, json_path = write_inventory(records, args.out_prefix)
    except InventoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(records)} artifacts to {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
