#!/usr/bin/env python3
"""Fail-closed audit for SQLens matched-recall experiment artifacts.

The formal runner already validates PostgreSQL plans, runtime identity, the SQL
query contract, and the D2 same-graph proof while it runs.  This tool verifies
that the persisted manifest still describes those exact files and that every
final method has the requested query/repeat coverage at its recall target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .pgvector_design1_design2_design3_selectivity_benchmark import (
        stable_d2_graph_proof,
        validate_d2_graph_proof,
    )
    from .pgvector_target_recall_selectivity_runner import (
        DEFAULT_P0_RELEASE_CONTRACT,
        FORMAL_BASE_GRID_MAX_EF,
        FORMAL_CALIBRATION_GRID_POLICY,
        formal_completion_gate,
        load_p0_release_contract,
        require_plan_evidence,
    )
except ImportError:
    from pgvector_design1_design2_design3_selectivity_benchmark import (
        stable_d2_graph_proof,
        validate_d2_graph_proof,
    )
    from pgvector_target_recall_selectivity_runner import (
        DEFAULT_P0_RELEASE_CONTRACT,
        FORMAL_BASE_GRID_MAX_EF,
        FORMAL_CALIBRATION_GRID_POLICY,
        formal_completion_gate,
        load_p0_release_contract,
        require_plan_evidence,
    )


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRUTH_CSV = (
    ROOT
    / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_unique_embeddings_formal.csv"
)
DEFAULT_FILTERS_CSV = (
    ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
)
D2_MODES = {"design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3"}


class AuditError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as source:
        return sum(1 for _ in csv.DictReader(source))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise AuditError(f"CSV has no header: {path}")
        return list(reader)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok", "complete"}


def _release_contract_errors(
    manifest: Mapping[str, Any],
    run_spec: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    allow_legacy_source: bool,
) -> list[str]:
    """Verify the immutable r33 identity without relying on a live database."""
    errors: list[str] = []
    runtime = run_spec.get("sqlens_runtime_provenance")
    binding = run_spec.get("runtime_identity_binding")
    expected_build = str(contract["expected_sqlens_build_id"])
    expected_sha = str(contract["expected_vector_so_sha256"])
    if not isinstance(runtime, Mapping):
        return ["run_spec.sqlens_runtime_provenance is missing for release contract audit"]
    if runtime.get("loaded_vector_sqlens_build_id") != expected_build:
        errors.append("loaded SQLens build ID differs from the P0 release contract")
    if runtime.get("loaded_vector_so_sha256") != expected_sha:
        errors.append("loaded vector.so SHA256 differs from the P0 release contract")
    if not isinstance(binding, Mapping):
        errors.append("run_spec.runtime_identity_binding is missing for release contract audit")
    elif (
        binding.get("expected_build_id") != expected_build
        or binding.get("expected_vector_so_sha256") != expected_sha
        or binding.get("exact_match") is not True
    ):
        errors.append("runtime identity binding differs from the P0 release contract")

    observed = run_spec.get("p0_release_contract")
    if not isinstance(observed, Mapping):
        if not allow_legacy_source:
            errors.append("run_spec.p0_release_contract is missing")
        return errors
    required = {
        "contract_id": contract["contract_id"],
        "sha256": contract["sha256"],
        "expected_sqlens_build_id": expected_build,
        "expected_vector_so_sha256": expected_sha,
    }
    for field, value in required.items():
        if observed.get(field) != value:
            errors.append(f"run_spec.p0_release_contract.{field} differs from the immutable contract")
    return errors


def _release_completion(
    manifest: Mapping[str, Any],
    raw_valid: bool,
) -> dict[str, bool]:
    requested_complete = manifest.get("requested_slice_complete") is True
    full_complete = manifest.get("formal_release_complete") is True
    diagnostic_valid = bool(raw_valid and requested_complete)
    artifact_valid = bool(raw_valid and full_complete)
    return {
        "diagnostic_valid": diagnostic_valid,
        "artifact_valid": artifact_valid,
        "paper_eligible": artifact_valid,
    }


def _completion_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    """Convert persisted CSV strings to the runner's typed completion contract."""
    boolean_fields = (
        "rows_complete",
        "target_confirmed_in_calibration",
        "target_confirmed_in_final",
        "matched_recall_comparison_valid",
    )
    normalized: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = dict(row)
        for field in boolean_fields:
            item[field] = is_true(row.get(field))
        item["errors"] = integer(row.get("errors") or 0, "final.errors")
        normalized.append(item)
    return normalized


def write_completion_reaudited_manifest(
    manifest_path: Path,
    *,
    truth_csv: Path,
    filters_csv: Path,
    output_path: Path | None = None,
    recall_tolerance: float = 0.0,
    release_contract: Path = DEFAULT_P0_RELEASE_CONTRACT,
) -> Path:
    """Write a release-audited sibling without mutating a historical source manifest.

    The source manifest remains immutable.  The new manifest binds the source
    SHA-256, the immutable r33 contract, and every replaced completion field.
    """
    source_sha256 = sha256_file(manifest_path)
    source = read_json(manifest_path)
    contract = load_p0_release_contract(release_contract)
    audit = audit_manifest(
        manifest_path,
        truth_csv=truth_csv,
        filters_csv=filters_csv,
        recall_tolerance=recall_tolerance,
        require_complete=False,
        release_contract=release_contract,
        allow_legacy_contract=True,
    )
    if not audit["valid"]:
        raise AuditError(
            "source artifact failed re-audit: " + "; ".join(audit["errors"])
        )

    outputs = source.get("outputs")
    if not isinstance(outputs, Mapping):
        raise AuditError("manifest.outputs is missing or is not an object")
    selected = read_csv(_artifact_path(outputs.get("selected")))
    final_rows = read_csv(_artifact_path(outputs.get("final")))
    source_run_spec = source.get("run_spec")
    source_args = source_run_spec.get("args") if isinstance(source_run_spec, Mapping) else None
    protocol_args: argparse.Namespace | None = None
    if isinstance(source_args, Mapping):
        protocol_args = argparse.Namespace(**dict(source_args))
        protocol_args.filters_csv = Path(str(source_args.get("filters_csv") or filters_csv))
        protocol_args.truth_csv = Path(str(source_args.get("truth_csv") or truth_csv))
        protocol_args.release_contract_provenance = contract
    completion_args = protocol_args
    try:
        completion = formal_completion_gate(
            [str(value) for value in source.get("filters", [])],
            [str(value) for value in source.get("modes", [])],
            [float(value) for value in source.get("targets", [])],
            _completion_rows(selected),
            _completion_rows(final_rows),
            False,
            completion_args,
        )
    except AttributeError:
        # Legacy manifests may not preserve the complete parser namespace.  They
        # can be diagnostic siblings, never full P0 releases, until rerun.
        completion = formal_completion_gate(
            [str(value) for value in source.get("filters", [])],
            [str(value) for value in source.get("modes", [])],
            [float(value) for value in source.get("targets", [])],
            _completion_rows(selected),
            _completion_rows(final_rows),
            False,
            None,
        )
    if completion["status"] != "complete":
        raise AuditError(
            "requested slice is not complete after recomputation: "
            + json.dumps(completion, sort_keys=True)
        )

    completion_fields = tuple(completion)
    previous = {field: source.get(field) for field in completion_fields}
    amended = dict(source)
    amended.update(completion)
    amended["diagnostic_valid"] = True
    amended["artifact_valid"] = bool(completion["formal_release_complete"])
    amended["paper_eligible"] = bool(completion["formal_release_complete"])
    amended_run_spec = dict(source_run_spec) if isinstance(source_run_spec, Mapping) else {}
    amended_run_spec["p0_release_contract"] = contract
    uses_d2 = any(str(mode) in D2_MODES for mode in amended.get("modes", []))
    amended_run_spec["run_spec_hash"] = _run_spec_hash(amended_run_spec, uses_d2)
    amended["run_spec"] = amended_run_spec
    amended["run_spec_hash"] = amended_run_spec["run_spec_hash"]
    amended["completion_reaudit"] = {
        "contract": "p0_release_audited_sibling_v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": source_sha256,
        "source_completion": previous,
        "artifact_audit_valid": True,
        "artifact_audit_errors": [],
        "artifact_audit_warnings": audit["warnings"],
        "release_contract": contract,
        "release_audited_completion": {
            "diagnostic_valid": amended["diagnostic_valid"],
            "artifact_valid": amended["artifact_valid"],
            "paper_eligible": amended["paper_eligible"],
        },
    }
    destination = output_path or manifest_path.with_name(
        f"{manifest_path.stem}.release-audited{manifest_path.suffix}"
    )
    if destination.resolve() == manifest_path.resolve():
        raise AuditError("re-audited manifest must not overwrite the source manifest")
    write_json_atomic(destination, amended)

    verification = audit_manifest(
        destination,
        truth_csv=truth_csv,
        filters_csv=filters_csv,
        recall_tolerance=recall_tolerance,
        require_complete=True,
        release_contract=release_contract,
    )
    if not verification["valid"]:
        destination.unlink(missing_ok=True)
        raise AuditError(
            "re-audited manifest failed verification: "
            + "; ".join(verification["errors"])
        )
    return destination


def _raw_bool(value: Any, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise AuditError(f"{field} must be an explicit boolean")


def _d3_phase_errors(row: Mapping[str, Any], context: str) -> list[str]:
    phase = str(row.get("d3_phase") or "")
    route = str(row.get("guidance_route") or "")
    if phase not in {"probe", "admission", "warm", "bypass"}:
        return [f"{context} has invalid D3 phase: {phase!r}"]
    try:
        enabled = _raw_bool(row.get("guidance_enabled"), f"{context}.guidance_enabled")
        active_after = _raw_bool(row.get("d3_active_after"), f"{context}.d3_active_after")
        admitted_after = _raw_bool(
            row.get("d3_admitted_after"), f"{context}.d3_admitted_after"
        )
        if phase == "probe":
            if route != "d3_stock_probe" or enabled or active_after or admitted_after:
                return [f"{context} lacks an inactive D3 stock-probe proof"]
        elif phase == "bypass":
            if route == "d3_stock_probe" or enabled or active_after:
                return [f"{context} lacks an inactive D3 policy-bypass proof"]
        elif phase == "admission":
            if (
                route != "enabled"
                or not enabled
                or not admitted_after
                or integer(
                    row.get("d3_adaptive_admissions_delta"),
                    f"{context}.d3_adaptive_admissions_delta",
                )
                <= 0
            ):
                return [f"{context} lacks a D3 admission proof"]
        else:
            same_predicate = _raw_bool(
                row.get("d3_same_predicate_before"),
                f"{context}.d3_same_predicate_before",
            )
            admitted_before = _raw_bool(
                row.get("d3_admitted_before"), f"{context}.d3_admitted_before"
            )
            reused = _raw_bool(
                row.get("d3_active_guidance_reused"),
                f"{context}.d3_active_guidance_reused",
            )
            if (
                route != "enabled"
                or not enabled
                or not same_predicate
                or not admitted_before
                or not admitted_after
                or not reused
            ):
                return [f"{context} lacks predicate-scoped D3 warm-reuse proof"]
    except AuditError as exc:
        return [str(exc)]
    return []


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise AuditError(f"{label} is not finite: {value!r}")
    return result


def integer(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} is not an integer: {value!r}") from exc


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered)) - 1
    return ordered[min(len(ordered) - 1, max(0, rank))]


def _artifact_path(value: Any) -> Path:
    if isinstance(value, Mapping):
        value = value.get("path")
    return Path(str(value or ""))


def _canonical(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _artifact_errors(artifact: Any, label: str, *, csv_file: bool) -> list[str]:
    if not isinstance(artifact, Mapping):
        return [f"missing {label} artifact metadata"]
    path = _artifact_path(artifact)
    if not path.is_file():
        return [f"missing {label} artifact file: {path}"]
    errors: list[str] = []
    expected_sha = str(artifact.get("sha256") or "")
    actual_sha = sha256_file(path)
    if expected_sha != actual_sha:
        errors.append(f"{label} sha256 mismatch: {path}")
    if "bytes" in artifact and integer(artifact.get("bytes"), f"{label}.bytes") != path.stat().st_size:
        errors.append(f"{label} byte count mismatch: {path}")
    if csv_file and "row_count" in artifact:
        expected_rows = integer(artifact.get("row_count"), f"{label}.row_count")
        if expected_rows != csv_row_count(path):
            errors.append(f"{label} row count mismatch: {path}")
    return errors


def _run_spec_hash(run_spec: Mapping[str, Any], uses_d2: bool = False) -> str:
    payload = dict(run_spec)
    payload.pop("run_spec_hash", None)
    if uses_d2:
        proof = payload.get("d2_graph_proof")
        if not isinstance(proof, dict):
            raise AuditError("D2 run spec is missing d2_graph_proof")
        payload["d2_graph_proof"] = stable_d2_graph_proof(proof)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_value(manifest: Mapping[str, Any], run_spec: Mapping[str, Any], name: str) -> int:
    value = manifest.get(name)
    if value is None and isinstance(run_spec.get("args"), Mapping):
        value = run_spec["args"].get(name)  # type: ignore[index]
    result = integer(value, name)
    if result < 0 or (not name.endswith("offset") and result <= 0):
        raise AuditError(f"invalid {name}: {result}")
    return result


def _calibration_policy_errors(
    manifest: Mapping[str, Any], run_spec: Mapping[str, Any]
) -> list[str]:
    policy = manifest.get("calibration_policy")
    if not isinstance(policy, Mapping):
        return []
    args = run_spec.get("args")
    run_policy = args.get("calibration_selection_policy") if isinstance(args, Mapping) else None
    manifest_policy = policy.get("calibration_selection_policy")
    if manifest_policy and run_policy and manifest_policy != run_policy:
        return ["calibration selection policy differs between manifest and run_spec.args"]
    effective = str(manifest_policy or run_policy or "")
    description = str(policy.get("selection") or "").lower().replace("_", " ")
    stop_metric = str(policy.get("stop_metric") or "")
    stop_condition = str(policy.get("stop_condition") or "").lower()
    grid_policy = str(policy.get("grid_policy") or "")
    report_only = "report-only" in description or "report only" in description
    errors: list[str] = []
    if effective == "lcb_then_max_recall":
        if report_only or "lcb" not in description:
            errors.append(
                "calibration policy is lcb_then_max_recall but its selection description "
                "does not make LCB part of selection"
            )
        if grid_policy:
            if grid_policy != FORMAL_CALIBRATION_GRID_POLICY:
                errors.append("formal calibration grid policy is unknown or incomplete")
            if (
                policy.get("base_grid_max_ef") != FORMAL_BASE_GRID_MAX_EF
                or policy.get("base_grid_complete_required") is not True
                or policy.get("extension_ef_search_values")
                != [20_000, 50_000, 100_000]
                or policy.get("extension_trigger")
                != "max_target_lcb95_unmet_after_complete_base_grid"
                or policy.get("extension_complete_required_when_triggered")
                is not True
                or policy.get("early_stop_allowed") is not False
                or policy.get("grid_exhaustion_semantics")
                != "all_policy_required_configs_executed"
            ):
                errors.append(
                    "formal calibration policy does not bind complete base/conditional "
                    "extension grids and forbid early stop"
                )
            if stop_metric != "recall_lcb95":
                errors.append(
                    "formal calibration qualification metric is not recall_lcb95"
                )
        elif stop_metric != "recall_lcb95":
            errors.append(
                "calibration policy is lcb_then_max_recall but its qualification "
                "metric is not recall_lcb95"
            )
    elif effective == "mean_latency" and "lcb" in description and not report_only:
        errors.append(
            "calibration policy is mean_latency but its selection description makes LCB operative"
        )
    elif effective == "mean_latency" and stop_metric not in {"", "recall_mean"}:
        errors.append(
            "calibration policy is mean_latency but calibration early-stop is not bound to recall_mean"
        )
    return errors


def _staged_grid_errors(
    manifest: Mapping[str, Any],
    filters: Sequence[object],
    modes: Sequence[object],
) -> list[str]:
    errors: list[str] = []
    mode_grids = manifest.get("mode_grids")
    pairs = manifest.get("calibration_pairs")
    if not isinstance(mode_grids, Mapping) or not isinstance(pairs, list):
        return ["formal staged-grid evidence is missing"]
    expected_pairs = {
        (str(filter_name), str(mode))
        for filter_name in filters
        for mode in modes
    }
    observed_pairs = {
        (str(pair.get("filter_name") or ""), str(pair.get("mode") or ""))
        for pair in pairs
        if isinstance(pair, Mapping)
    }
    if observed_pairs != expected_pairs or len(pairs) != len(expected_pairs):
        errors.append("calibration_pairs does not cover the complete filter x mode matrix")
    for pair in pairs:
        if not isinstance(pair, Mapping):
            errors.append("calibration_pairs contains a malformed entry")
            continue
        pair_name = f"{pair.get('filter_name')}/{pair.get('mode')}"
        mode = str(pair.get("mode") or "")
        if pair.get("calibration_grid_policy") != FORMAL_CALIBRATION_GRID_POLICY:
            errors.append(f"{pair_name}: legacy or unknown calibration grid policy")
        if pair.get("grid_exhausted") is not True:
            errors.append(f"{pair_name}: required staged grid is not exhausted")
        if pair.get("stopped_early") is not False:
            errors.append(f"{pair_name}: forbidden calibration early stop")
        grid = mode_grids.get(mode)
        if not isinstance(grid, list):
            errors.append(f"{pair_name}: mode grid is missing")
            continue
        families = pair.get("families")
        if not isinstance(families, Mapping):
            errors.append(f"{pair_name}: iterative-family evidence is missing")
            continue
        expected_families = {
            str(config.get("iterative_scan") or "")
            for config in grid
            if isinstance(config, Mapping)
        }
        if set(families) != expected_families:
            errors.append(f"{pair_name}: iterative-family coverage differs from mode grid")
            continue
        for family, evidence in families.items():
            family_name = f"{pair_name}/{family}"
            if not isinstance(evidence, Mapping):
                errors.append(f"{family_name}: malformed family evidence")
                continue
            family_grid = [
                config
                for config in grid
                if isinstance(config, Mapping)
                and str(config.get("iterative_scan") or "") == family
            ]
            base = [
                config
                for config in family_grid
                if int(config.get("ef_search") or 0) <= FORMAL_BASE_GRID_MAX_EF
            ]
            extension = [
                config
                for config in family_grid
                if int(config.get("ef_search") or 0) > FORMAL_BASE_GRID_MAX_EF
            ]
            extension_required = evidence.get("high_extension_required") is True
            required_count = len(base) + (len(extension) if extension_required else 0)
            if int(evidence.get("configs_planned") or -1) != required_count:
                errors.append(f"{family_name}: planned config count violates staged grid")
            if int(evidence.get("configs_executed") or -1) != required_count:
                errors.append(f"{family_name}: executed config count violates staged grid")
            if evidence.get("grid_exhausted") is not True:
                errors.append(f"{family_name}: required staged grid is not exhausted")
            if evidence.get("stopped_early") is True:
                errors.append(f"{family_name}: forbidden first-crossing early stop")
            if evidence.get("stopped_by_cross_family_target") is True:
                errors.append(f"{family_name}: forbidden latency-dominance early stop")
            if extension_required:
                if evidence.get("high_extension_executed") is not True:
                    errors.append(f"{family_name}: required extension was not completed")
                expected_max = max(
                    int(config.get("ef_search") or 0) for config in family_grid
                )
            else:
                if evidence.get("high_extension_executed") is not False:
                    errors.append(f"{family_name}: unnecessary extension was executed")
                if extension and evidence.get("high_extension_skip_reason") != (
                    "max_target_lcb_met_on_complete_base_grid"
                ):
                    errors.append(f"{family_name}: extension skip lacks base-grid LCB proof")
                expected_max = max(
                    int(config.get("ef_search") or 0) for config in base
                )
            if int(evidence.get("max_ef_evaluated") or 0) != expected_max:
                errors.append(f"{family_name}: max evaluated ef violates staged grid")
    return errors


def _query_ids(run_spec: Mapping[str, Any], name: str) -> list[int]:
    raw = run_spec.get(name)
    if not isinstance(raw, list):
        raise AuditError(f"run_spec.{name} is missing or is not a list")
    values = [integer(value, f"run_spec.{name}") for value in raw]
    if len(values) != len(set(values)):
        raise AuditError(f"run_spec.{name} has non-unique ids")
    return values


def _validate_d2_parent(
    run_spec: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[str]:
    proof = run_spec.get("d2_graph_proof")
    final_proof = manifest.get("d2_graph_proof_final")
    if not isinstance(proof, dict) or not isinstance(final_proof, dict):
        return ["D2 run is missing startup or final same-graph proof"]
    try:
        startup = validate_d2_graph_proof(
            proof,
            str(proof.get("source_index") or ""),
            str(proof.get("clone_index") or ""),
        )
        final = validate_d2_graph_proof(
            final_proof,
            str(final_proof.get("source_index") or ""),
            str(final_proof.get("clone_index") or ""),
        )
    except Exception as exc:
        return [f"invalid D2 same-graph proof: {exc}"]
    if startup.get("stable_fingerprint_sha256") != final.get("stable_fingerprint_sha256"):
        return ["stale D2 same-graph proof: startup/final fingerprints differ"]
    return []


def _plan_entries(
    manifest: Mapping[str, Any], loaded_build: str, loaded_sha: str
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    raw_entries = manifest.get("plan_evidence")
    if not isinstance(raw_entries, list) or not raw_entries:
        return {}, ["manifest has no plan_evidence entries"]
    entries: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for number, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping):
            errors.append(f"plan_evidence[{number}] is not an object")
            continue
        errors.extend(_artifact_errors(entry, f"plan_evidence[{number}]", csv_file=False))
        raw_artifact = entry.get("raw_output")
        errors.extend(
            _artifact_errors(raw_artifact, f"plan_evidence[{number}].raw_output", csv_file=True)
        )
        raw_path = _artifact_path(raw_artifact)
        key = _canonical(raw_path)
        if key in entries:
            errors.append(f"duplicate plan evidence for raw output: {raw_path}")
            continue
        entries[key] = entry
        if not raw_path.is_file():
            continue
        expected_plan_path = raw_path.with_suffix(raw_path.suffix + ".plan.json")
        if _canonical(_artifact_path(entry)) != _canonical(expected_plan_path):
            errors.append(f"plan evidence path does not match raw output: {raw_path}")
            continue
        try:
            payload = require_plan_evidence(raw_path)
        except Exception as exc:
            errors.append(f"plan evidence gate failed for {raw_path}: {exc}")
            continue
        for phase in ("sqlens_runtime_identity_startup", "sqlens_runtime_identity_final"):
            identity = payload.get(phase)
            if not isinstance(identity, Mapping):
                errors.append(f"{phase} missing for {raw_path}")
                continue
            if (
                identity.get("expected_build_id") != loaded_build
                or identity.get("observed_build_id") != loaded_build
                or identity.get("expected_vector_so_sha256") != loaded_sha
                or identity.get("observed_vector_so_sha256") != loaded_sha
                or identity.get("exact_match") is not True
            ):
                errors.append(f"runtime identity mismatch in {phase} for {raw_path}")
    return entries, errors


def _coverage_errors(
    raw_path: Path,
    filter_name: str,
    mode: str,
    query_offset: int,
    expected_query_ids: Sequence[int],
    repeats: int,
) -> list[str]:
    rows = read_csv(raw_path)
    relevant = [
        row
        for row in rows
        if str(row.get("filter_name") or "") == filter_name
        and str(row.get("mode") or "") == mode
    ]
    expected_mapping = {
        query_offset + position: query_id
        for position, query_id in enumerate(expected_query_ids)
    }
    expected_keys = {
        (query_no, expected_mapping[query_no], repeat)
        for query_no in expected_mapping
        for repeat in range(repeats)
    }
    observed: set[tuple[int, int, int]] = set()
    errors: list[str] = []
    for row_number, row in enumerate(relevant):
        if str(row.get("error") or "").strip():
            errors.append(
                f"raw row reports error for {filter_name}/{mode}: "
                f"q={row.get('query_no')} r={row.get('repeat')}"
            )
            continue
        if mode == "design1_bloom_bfs_layout_d3":
            errors.extend(
                _d3_phase_errors(
                    row,
                    f"raw[{row_number}] {filter_name}/{mode}",
                )
            )
        try:
            key = (
                integer(row.get("query_no"), "raw.query_no"),
                integer(row.get("query_id"), "raw.query_id"),
                integer(row.get("repeat"), "raw.repeat"),
            )
        except AuditError as exc:
            errors.append(str(exc))
            continue
        if key in observed:
            errors.append(
                f"duplicate raw mode/query/repeat row for {filter_name}/{mode}: {key}"
            )
        observed.add(key)
    if observed != expected_keys:
        missing = len(expected_keys - observed)
        extra = len(observed - expected_keys)
        errors.append(
            f"raw coverage mismatch for {filter_name}/{mode}: "
            f"expected={len(expected_keys)} observed={len(observed)} missing={missing} extra={extra}"
        )
    return errors


def _interleaving_errors(
    raw_path: Path,
    filter_name: str,
    modes: Sequence[str],
    query_offset: int,
    expected_query_ids: Sequence[int],
    repeats: int,
) -> list[str]:
    """Verify that a persisted final raw file contains a balanced interleave."""
    rows = read_csv(raw_path)
    context = f"{filter_name} in {raw_path}"
    if not rows:
        return [f"interleaved final raw is empty for {context}"]

    required_fields = {
        "filter_name",
        "mode",
        "pair_key",
        "query_no",
        "repeat",
        "schedule_position",
    }
    missing_fields = sorted(required_fields - set(rows[0]))
    if missing_fields:
        return [
            f"interleaved final raw missing required fields for {context}: "
            + ", ".join(missing_fields)
        ]

    requested_modes = [str(mode) for mode in modes]
    if len(requested_modes) != len(set(requested_modes)):
        return [f"requested interleaved modes are not unique for {context}"]
    mode_count = len(requested_modes)
    expected_positions = set(range(1, mode_count + 1))
    expected_pair_keys = {
        f"{filter_name}|q{query_offset + query_position}|r{repeat}"
        for query_position, _query_id in enumerate(expected_query_ids)
        for repeat in range(repeats)
    }
    relevant = [row for row in rows if str(row.get("filter_name") or "") == filter_name]
    if not relevant:
        return [f"interleaved final raw has no rows for {context}"]

    errors: list[str] = []
    grouped: dict[str, list[tuple[str, int]]] = {}
    position_counts = {
        mode: {position: 0 for position in expected_positions}
        for mode in requested_modes
    }
    for row_number, row in enumerate(relevant, start=2):
        pair = str(row.get("pair_key") or "")
        mode = str(row.get("mode") or "")
        try:
            query_no = integer(row.get("query_no"), f"raw[{row_number}].query_no")
            repeat = integer(row.get("repeat"), f"raw[{row_number}].repeat")
            position = integer(
                row.get("schedule_position"),
                f"raw[{row_number}].schedule_position",
            )
        except AuditError as exc:
            errors.append(str(exc))
            continue
        expected_pair = f"{filter_name}|q{query_no}|r{repeat}"
        if not pair:
            errors.append(f"raw[{row_number}] has an empty pair_key for {context}")
            continue
        if pair != expected_pair:
            errors.append(
                f"raw[{row_number}] pair_key mismatch for {context}: "
                f"observed={pair!r} expected={expected_pair!r}"
            )
        grouped.setdefault(pair, []).append((mode, position))
        if mode in position_counts and position in expected_positions:
            position_counts[mode][position] += 1

    observed_pair_keys = set(grouped)
    if observed_pair_keys != expected_pair_keys:
        errors.append(
            f"interleaved pair coverage mismatch for {context}: "
            f"expected={len(expected_pair_keys)} observed={len(observed_pair_keys)} "
            f"missing={len(expected_pair_keys - observed_pair_keys)} "
            f"extra={len(observed_pair_keys - expected_pair_keys)}"
        )

    requested_mode_set = set(requested_modes)
    for pair in sorted(expected_pair_keys | observed_pair_keys):
        pair_rows = grouped.get(pair, [])
        observed_modes = [mode for mode, _position in pair_rows]
        observed_positions = [position for _mode, position in pair_rows]
        if len(observed_modes) != mode_count or set(observed_modes) != requested_mode_set:
            errors.append(
                f"interleaved pair modes mismatch for {context}, pair={pair!r}: "
                f"expected={sorted(requested_mode_set)} observed={sorted(observed_modes)}"
            )
        if len(observed_positions) != mode_count or set(observed_positions) != expected_positions:
            errors.append(
                f"interleaved schedule positions mismatch for {context}, pair={pair!r}: "
                f"expected={sorted(expected_positions)} observed={sorted(observed_positions)}"
            )

    for mode, counts_by_position in position_counts.items():
        counts = [counts_by_position[position] for position in sorted(expected_positions)]
        if max(counts) - min(counts) > 1:
            errors.append(
                f"unbalanced interleaved schedule positions for {context}, mode={mode}: "
                f"counts={counts_by_position}"
            )
    return errors


def _summary(final_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    stock = {
        (str(row.get("filter_name")), finite_float(row.get("target_recall"), "target_recall")):
        finite_float(row.get("latency_mean_ms"), "latency_mean_ms")
        for row in final_rows
        if row.get("mode") == "original"
    }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in final_rows:
        grouped.setdefault(str(row.get("mode") or ""), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for mode, rows in grouped.items():
        latencies = [finite_float(row.get("latency_mean_ms"), "latency_mean_ms") for row in rows]
        speedups: list[float] = []
        if mode != "original":
            for row, latency in zip(rows, latencies):
                key = (
                    str(row.get("filter_name")),
                    finite_float(row.get("target_recall"), "target_recall"),
                )
                if key in stock and latency > 0:
                    speedups.append(stock[key] / latency)
        out[mode] = {
            "rows": len(rows),
            "mean_latency_ms": statistics.fmean(latencies),
            "p95_latency_ms": percentile(latencies, 0.95),
            "p99_latency_ms": percentile(latencies, 0.99),
            "mean_speedup_vs_stock": statistics.fmean(speedups) if speedups else None,
        }
    return out


def audit_manifest(
    manifest_path: Path,
    *,
    truth_csv: Path,
    filters_csv: Path,
    recall_tolerance: float = 0.0,
    require_complete: bool = True,
    release_contract: Path = DEFAULT_P0_RELEASE_CONTRACT,
    allow_legacy_contract: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = read_json(manifest_path)
        run_spec = manifest.get("run_spec")
        if not isinstance(run_spec, Mapping):
            raise AuditError("manifest.run_spec is missing or is not an object")
        contract = load_p0_release_contract(release_contract)
        errors.extend(
            _release_contract_errors(
                manifest,
                run_spec,
                contract,
                allow_legacy_source=allow_legacy_contract,
            )
        )
        filters = manifest.get("filters")
        modes = manifest.get("modes")
        targets = manifest.get("targets")
        if not isinstance(filters, list) or not filters:
            raise AuditError("manifest.filters is missing or empty")
        if not isinstance(modes, list) or not modes:
            raise AuditError("manifest.modes is missing or empty")
        if not isinstance(targets, list) or not targets:
            raise AuditError("manifest.targets is missing or empty")
        errors.extend(_staged_grid_errors(manifest, filters, modes))

        if require_complete:
            if manifest.get("status") != "complete":
                errors.append(f"manifest status is {manifest.get('status')!r}, expected 'complete'")
            for flag in ("matrix_complete", "measurement_complete", "comparison_valid"):
                if manifest.get(flag) is not True:
                    errors.append(f"manifest completion flag is false: {flag}")
            if manifest.get("requested_slice_complete") is not True:
                errors.append("manifest requested_slice_complete is false")

        uses_d2 = any(str(mode) in D2_MODES for mode in modes)
        expected_run_hash = _run_spec_hash(run_spec, uses_d2)
        if manifest.get("run_spec_hash") != expected_run_hash:
            errors.append("manifest run_spec_hash mismatch")
        if run_spec.get("run_spec_hash") != expected_run_hash:
            errors.append("run_spec embedded run_spec_hash mismatch")
        errors.extend(_calibration_policy_errors(manifest, run_spec))

        if sha256_file(truth_csv) != run_spec.get("truth_sha256"):
            errors.append("truth_sha256 mismatch against provided truth CSV")
        if sha256_file(filters_csv) != run_spec.get("filters_sha256"):
            errors.append("filters_sha256 mismatch against provided filters CSV")

        calibration_queries = _split_value(manifest, run_spec, "calibration_queries")
        final_queries = _split_value(manifest, run_spec, "final_queries")
        final_repeats = _split_value(manifest, run_spec, "final_repeats")
        final_offset = _split_value(manifest, run_spec, "final_query_offset")
        run_args = run_spec.get("args")
        if not isinstance(run_args, Mapping):
            raise AuditError("run_spec.args is missing or is not an object")
        manifest_final_order = str(manifest.get("final_execution_order") or "")
        run_spec_final_order = str(run_args.get("final_execution_order") or "")
        if manifest_final_order != "interleaved":
            errors.append(
                "formal final execution order must be interleaved in the manifest"
            )
        if run_spec_final_order != "interleaved":
            errors.append(
                "formal final execution order must be interleaved in run_spec.args"
            )
        calibration_ids = _query_ids(run_spec, "calibration_query_ids")
        final_ids = _query_ids(run_spec, "final_query_ids")
        if len(calibration_ids) != calibration_queries:
            errors.append("calibration_query_ids length mismatch")
        if len(final_ids) != final_queries:
            errors.append("final_query_ids length mismatch")
        if set(calibration_ids).intersection(final_ids):
            errors.append("calibration/final query_id overlap detected")

        outputs = manifest.get("outputs")
        if not isinstance(outputs, Mapping):
            raise AuditError("manifest.outputs is missing or is not an object")
        for name in ("calibration", "selected", "final"):
            errors.extend(_artifact_errors(outputs.get(name), f"outputs.{name}", csv_file=True))
        selected_path = _artifact_path(outputs.get("selected"))
        selected_rows = read_csv(selected_path) if selected_path.is_file() else []
        final_path = _artifact_path(outputs.get("final"))
        final_rows = read_csv(final_path) if final_path.is_file() else []
        calibration_policy = manifest.get("calibration_policy")
        if (
            isinstance(calibration_policy, Mapping)
            and calibration_policy.get("grid_policy")
            == FORMAL_CALIBRATION_GRID_POLICY
        ):
            for row_no, row in enumerate(selected_rows):
                if not is_true(row.get("grid_exhausted")):
                    errors.append(f"selected[{row_no}] does not exhaust its calibration grid")
                if is_true(row.get("stopped_early")):
                    errors.append(f"selected[{row_no}] reports forbidden calibration early-stop")
                if row.get("calibration_grid_policy") != FORMAL_CALIBRATION_GRID_POLICY:
                    errors.append(f"selected[{row_no}] has an incompatible calibration grid policy")
                if row.get("selection_status") != "selected":
                    errors.append(f"selected[{row_no}] is a non-publishable diagnostic fallback")
                if not is_true(row.get("target_lcb95_met_in_calibration")):
                    errors.append(f"selected[{row_no}] does not meet the calibration LCB target")

        runtime = run_spec.get("sqlens_runtime_provenance")
        if not isinstance(runtime, Mapping):
            raise AuditError("run_spec.sqlens_runtime_provenance is missing")
        loaded_build = str(runtime.get("loaded_vector_sqlens_build_id") or "")
        loaded_sha = str(runtime.get("loaded_vector_so_sha256") or "")
        if not loaded_build or len(loaded_sha) != 64:
            errors.append("invalid SQLens runtime provenance in run_spec")
        expected_build = str(run_args.get("expected_sqlens_build_id") or "")
        expected_sha = str(run_args.get("expected_vector_so_sha256") or "")
        if expected_build or expected_sha:
            binding = run_spec.get("runtime_identity_binding")
            if (
                not isinstance(binding, Mapping)
                or not expected_build
                or len(expected_sha) != 64
                or loaded_build != expected_build
                or loaded_sha != expected_sha
                or binding.get("expected_build_id") != expected_build
                or binding.get("expected_vector_so_sha256") != expected_sha
                or binding.get("exact_match") is not True
            ):
                errors.append("exact wrapper/parent SQLens runtime binding is invalid")
        if run_args.get("prewarm_index_health") is True:
            health = run_spec.get("index_query_health")
            indexes = health.get("indexes") if isinstance(health, Mapping) else None
            if not isinstance(indexes, list) or not indexes:
                errors.append("index-health prewarm evidence is missing")
            else:
                for item in indexes:
                    prewarm = item.get("prewarm") if isinstance(item, Mapping) else None
                    if (
                        not isinstance(prewarm, Mapping)
                        or prewarm.get("enabled") is not True
                        or integer(prewarm.get("blocks"), "prewarm.blocks") < 0
                        or finite_float(
                            prewarm.get("elapsed_ms"), "prewarm.elapsed_ms"
                        )
                        < 0
                    ):
                        errors.append("index-health prewarm evidence is incomplete")
                        break
        plan_entries, plan_errors = _plan_entries(manifest, loaded_build, loaded_sha)
        errors.extend(plan_errors)
        if uses_d2:
            errors.extend(_validate_d2_parent(run_spec, manifest))

        expected_cells = {
            (str(filter_name), float(target), str(mode))
            for filter_name in filters
            for target in targets
            for mode in modes
        }
        observed_cells: set[tuple[str, float, str]] = set()
        coverage_cache: set[tuple[str, str, str]] = set()
        for row_no, row in enumerate(final_rows):
            filter_name = str(row.get("filter_name") or "")
            mode = str(row.get("mode") or "")
            target = finite_float(row.get("target_recall"), f"final[{row_no}].target_recall")
            cell = (filter_name, target, mode)
            if cell in observed_cells:
                errors.append(f"duplicate final summary cell: {cell}")
            observed_cells.add(cell)
            recall = finite_float(row.get("recall_mean"), f"final[{row_no}].recall_mean")
            if recall + recall_tolerance < target:
                errors.append(f"final recall below target for {cell}: {recall:.6f} < {target:.6f}")
            lcb = row.get("recall_lcb95")
            if str(lcb or "").strip() and finite_float(lcb, "recall_lcb95") < target:
                warnings.append(f"recall LCB95 below target for {cell}; mean-recall contract still passes")
            for flag in (
                "target_confirmed_in_calibration",
                "target_met_in_final",
                "target_confirmed_in_final",
                "rows_complete",
            ):
                if not is_true(row.get(flag)):
                    errors.append(f"final row has false {flag} for {cell}")
            if str(row.get("final_status") or "") != "complete":
                errors.append(f"final row is incomplete for {cell}")
            if integer(row.get("errors") or 0, "final.errors") != 0:
                errors.append(f"final row reports errors for {cell}")
            if not is_true(row.get("matched_recall_comparison_valid")):
                errors.append(f"matched-recall comparison is invalid for {cell}")
            if integer(row.get("expected_queries"), "expected_queries") != final_queries:
                errors.append(f"expected_queries mismatch for {cell}")
            if integer(row.get("expected_repeats"), "expected_repeats") != final_repeats:
                errors.append(f"expected_repeats mismatch for {cell}")
            if mode != "original":
                if integer(row.get("paired_queries"), "paired_queries") != final_queries:
                    errors.append(f"paired query count mismatch for {cell}")
                if integer(row.get("paired_repeats"), "paired_repeats") != final_repeats:
                    errors.append(f"paired repeat count mismatch for {cell}")
                if integer(row.get("paired_samples"), "paired_samples") != final_queries * final_repeats:
                    errors.append(f"paired sample count mismatch for {cell}")

            raw_path = Path(str(row.get("final_raw") or ""))
            if not raw_path.is_file():
                errors.append(f"missing final raw file for {cell}: {raw_path}")
                continue
            if str(row.get("final_raw_sha256") or "") != sha256_file(raw_path):
                errors.append(f"final_raw_sha256 mismatch for {cell}")
            if integer(row.get("final_raw_rows"), "final_raw_rows") != csv_row_count(raw_path):
                errors.append(f"final_raw_rows mismatch for {cell}")
            if _canonical(raw_path) not in plan_entries:
                errors.append(f"no plan evidence for final raw: {raw_path}")
            coverage_key = (_canonical(raw_path), filter_name, mode)
            if coverage_key not in coverage_cache:
                errors.extend(
                    _coverage_errors(
                        raw_path,
                        filter_name,
                        mode,
                        final_offset,
                        final_ids,
                        final_repeats,
                    )
                )
                coverage_cache.add(coverage_key)

        if observed_cells != expected_cells:
            errors.append(
                "final matrix mismatch: "
                f"expected={len(expected_cells)} observed={len(observed_cells)} "
                f"missing={len(expected_cells - observed_cells)} extra={len(observed_cells - expected_cells)}"
            )

        interleaving_cache: set[tuple[str, str, tuple[str, ...]]] = set()
        for filter_name in filters:
            for target in targets:
                group = [
                    row
                    for row in final_rows
                    if row.get("filter_name") == filter_name
                    and finite_float(row.get("target_recall"), "target_recall") == float(target)
                ]
                if not group:
                    continue
                orders = {str(row.get("final_execution_order") or "") for row in group}
                if len(orders) != 1:
                    errors.append(f"mixed final execution orders for {filter_name}/{target}")
                    continue
                if orders == {"interleaved"}:
                    raw_paths = {_canonical(Path(str(row.get("final_raw") or ""))) for row in group}
                    schedule_ids = {str(row.get("final_schedule_id") or "") for row in group}
                    if len(raw_paths) != 1 or len(schedule_ids) != 1 or "" in schedule_ids:
                        errors.append(f"invalid interleaved pairing for {filter_name}/{target}")
                        continue
                    raw_path = Path(str(group[0].get("final_raw") or ""))
                    cache_key = (
                        _canonical(raw_path),
                        str(filter_name),
                        tuple(str(mode) for mode in modes),
                    )
                    if cache_key not in interleaving_cache:
                        errors.extend(
                            _interleaving_errors(
                                raw_path,
                                str(filter_name),
                                [str(mode) for mode in modes],
                                final_offset,
                                final_ids,
                                final_repeats,
                            )
                        )
                        interleaving_cache.add(cache_key)
                else:
                    errors.append(
                        f"formal final is not interleaved for {filter_name}/{target}"
                    )

        methods = _summary(final_rows) if final_rows else {}
    except (AuditError, FileNotFoundError, OSError, RuntimeError, csv.Error, json.JSONDecodeError) as exc:
        errors.append(str(exc))
        methods = {}
        final_rows = []

    raw_valid = not errors
    completion = _release_completion(manifest if "manifest" in locals() else {}, raw_valid)
    if isinstance(manifest if "manifest" in locals() else None, Mapping):
        audited_run_spec = run_spec if isinstance(run_spec, Mapping) else {}
        has_contract = isinstance(audited_run_spec.get("p0_release_contract"), Mapping)
        if has_contract:
            for field, value in completion.items():
                if manifest.get(field) is not value:
                    errors.append(f"manifest {field} does not match independently audited completion")
            raw_valid = not errors
            completion = _release_completion(manifest, raw_valid)
    return {
        "manifest": str(manifest_path),
        "valid": not errors,
        "release_contract": contract if "contract" in locals() else None,
        "requested_slice_complete": bool(
            (manifest if "manifest" in locals() else {}).get("requested_slice_complete") is True
        ),
        "formal_release_complete": bool(
            (manifest if "manifest" in locals() else {}).get("formal_release_complete") is True
        ),
        **completion,
        "errors": errors,
        "warnings": warnings,
        "final_rows": len(final_rows),
        "methods": methods,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH_CSV)
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS_CSV)
    parser.add_argument("--release-contract", type=Path, default=DEFAULT_P0_RELEASE_CONTRACT)
    parser.add_argument("--recall-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--repair-legacy-completion",
        action="store_true",
        help=(
            "write a sibling .release-audited.json manifest after recomputing "
            "completion and binding the immutable P0 contract; never overwrites the source"
        ),
    )
    parser.add_argument(
        "--write-release-audited-sibling",
        action="store_true",
        help="alias for --repair-legacy-completion",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    manifest_paths = list(args.manifest)
    if args.repair_legacy_completion or args.write_release_audited_sibling:
        repaired: list[Path] = []
        for path in manifest_paths:
            repaired.append(
                write_completion_reaudited_manifest(
                    path,
                    truth_csv=args.truth_csv,
                    filters_csv=args.filters_csv,
                    recall_tolerance=args.recall_tolerance,
                    release_contract=args.release_contract,
                )
            )
        manifest_paths = repaired

    reports = [
        audit_manifest(
            path,
            truth_csv=args.truth_csv,
            filters_csv=args.filters_csv,
            recall_tolerance=args.recall_tolerance,
            release_contract=args.release_contract,
        )
        for path in manifest_paths
    ]
    payload = {
        "artifact": "sqlens_matched_recall_audit_v2",
        "overall_valid": all(report["valid"] for report in reports),
        "audits": reports,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if payload["overall_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
