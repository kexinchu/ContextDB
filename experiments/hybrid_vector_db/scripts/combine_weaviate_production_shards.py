"""Fail-closed combiner for formal Weaviate production result shards.

Shard manifests are attestations, not hints.  Every input and every declared
output is validated before a combined bundle is staged.  The combined
manifest is committed last so readers never treat partial data as complete.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_SHARDS = 4
EXPECTED_FILTERS = 14
EXPECTED_ROWS = 10_000_000
EXPECTED_CANDIDATE_ROWS = 9_979_556
DATASET = "amazon10m"
CANDIDATE_VALIDITY_PREDICATE = "embedding_valid"
EXPECTED_TARGETS = (0.90, 0.95, 0.99)
EXPECTED_CALIBRATION_QUERY_NOS = tuple(range(20, 100))
EXPECTED_FINAL_QUERY_NOS = tuple(range(100, 200))
FORMAL_CALIBRATION_QUERY_NOS = tuple(range(200))
FORMAL_FINAL_QUERY_NOS = tuple(range(200, 10_200))
FORMAL_CALIBRATION_REQUEST_NOS = tuple(range(200))
FORMAL_FINAL_REQUEST_NOS = tuple(range(10_000))
LEGACY_EXECUTION_PROTOCOL = "legacy-q100-r5"
FORMAL_EXECUTION_PROTOCOL = "current-q10k-r3"
FORMAL_CALIBRATION_REPEATS = 2
FORMAL_FINAL_REPEATS = 3
EXPECTED_K = 10
MANIFEST_PATTERN = "weaviate_production_matched_recall_*_manifest.json"
DATA_OUTPUTS = ("raw_csv", "summary_csv", "schema_json", "config_json")
ALL_OUTPUTS = (*DATA_OUTPUTS, "manifest_json")
OUTPUT_SUFFIXES = {
    "raw_csv": "_raw.csv",
    "summary_csv": "_summary.csv",
    "config_json": "_config.json",
    "schema_json": "_schema.json",
    "manifest_json": "_manifest.json",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$")
LATENCY_DEFINITION = "end_to_end_http_json_parse_row_id_transfer"
LATENCY_SCOPE = "end_to_end"
QPS_DEFINITION = "not_measured_by_single_client_latency_run"
RECALL_CONTRACT = "distance_threshold_tie_aware_v1"
MEASUREMENT_MODE = "single_client_sequential"
CALIBRATION_SELECTION_POLICY = "calibration_lcb95_then_max_mean_recall_v1"
CALIBRATION_SELECTION_RULE = (
    "calibration-only: select the lowest mean-latency complete HNSW configuration "
    "with bootstrap recall LCB95 >= target; only when no LCB-qualified HNSW "
    "configuration exists and both strategy EF grids are fully exhausted, select "
    "the highest calibration mean recall with latency as the tie-break; held-out "
    "final evidence cannot reselect"
)
CALIBRATION_STOP_METRIC = "recall_lcb95"
CALIBRATION_FALLBACK = "max_calibration_mean_recall_after_full_hnsw_grid"
LCB_SELECTION_MODE = "calibration_lcb95_qualified_fastest"
FALLBACK_SELECTION_MODE = "calibration_max_mean_recall_fallback"
ALLOWED_TARGET_STATUSES = {"selected"}
ALLOWED_TARGET_OUTCOMES = {"selected_and_confirmed"}
REQUIRED_SOURCE_HASHES = {
    "runner", "baseline_runner", "filters_csv", "truth_csv", "truth_manifest",
    "fbin", "import_manifest", "query_cohort_csv", "query_cohort_manifest",
    "import_attributes_csv",
}
FORMAL_REQUIRED_SOURCE_HASHES = {
    *REQUIRED_SOURCE_HASHES,
    "calibration_workload_csv",
    "measurement_workload_csv",
}
CURRENT_PROTOCOL = "amazon10m_current_corpus_q10200_truth_weaviate_v2"
LEGACY_CURRENT_PROTOCOL = "amazon10m_current_corpus_q10200_truth_weaviate_v1"
TIMED_RESPONSE_FIELDS = ["row_id", "_additional.distance"]


class ValidationFailure(ValueError):
    """Raised when a formal combined artifact cannot be proven valid."""


@dataclass
class Shard:
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    paths: dict[str, Path]
    hashes: dict[str, str]
    raw_fields: list[str]
    raw_rows: list[dict[str, str]]
    summary_fields: list[str]
    summary_rows: list[dict[str, str]]
    config: dict[str, Any]
    schema: dict[str, Any]
    filters: list[str]
    targets: list[dict[str, Any]]
    contract: dict[str, Any]
    target_outcomes: dict[str, int]
    original_schema_sha256: str
    query_ids: dict[int, int]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationFailure(f"{label} is not an object: {path}")
    return value


def _read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise ValidationFailure(f"cannot read {label} {path}: {exc}") from exc
    if not fields or any(not field for field in fields) or len(fields) != len(set(fields)):
        raise ValidationFailure(f"{label} has an invalid or duplicate CSV header: {path}")
    if any(None in row for row in rows):
        raise ValidationFailure(f"{label} has a row wider than its header: {path}")
    return fields, rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationFailure(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValidationFailure(f"{label} must be an array")
    return list(value)


def _exact_int(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool):
        raise ValidationFailure(f"{label} must be {expected}, got {value!r}")
    try:
        actual = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"{label} must be {expected}, got {value!r}") from exc
    if actual != expected:
        raise ValidationFailure(f"{label} must be {expected}, got {actual}")


def _positive_finite(value: Any, label: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValidationFailure(f"{label} is not finite and positive: {value!r}")
    return result


def _csv_true(value: Any, label: str) -> None:
    if str(value).strip().lower() != "true":
        raise ValidationFailure(f"{label} must be true, got {value!r}")


def _csv_bool_value(value: Any, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValidationFailure(f"{label} must be true or false, got {value!r}")
    return normalized == "true"


def _csv_zero(value: Any, label: str) -> None:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"{label} must be zero, got {value!r}") from exc
    if result != 0:
        raise ValidationFailure(f"{label} must be zero, got {result}")


def _target(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"{label} is not a target recall: {value!r}") from exc
    for expected in EXPECTED_TARGETS:
        if math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
            return expected
    raise ValidationFailure(f"{label} must be one of {list(EXPECTED_TARGETS)}, got {value!r}")


def _resolve_recorded_path(manifest_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise ValidationFailure(f"{manifest_path}: {label} is not a path")
    recorded = Path(value)
    if recorded.is_absolute():
        return recorded.resolve()
    candidates = [Path.cwd() / recorded, manifest_path.parent / recorded]
    candidates.extend(parent / recorded for parent in manifest_path.parents)
    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if len(existing) > 1:
        raise ValidationFailure(
            f"{manifest_path}: {label} resolves to multiple existing files: "
            f"{sorted(str(path) for path in existing)}"
        )
    if existing:
        return next(iter(existing))
    return (manifest_path.parent / recorded).resolve()


def _expected_filters(path: Path) -> tuple[list[str], str]:
    path = path.resolve()
    _, rows = _read_csv(path, "expected filters CSV")
    names: list[str] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=2):
        name = row.get("filter_name", "").strip()
        if not name:
            raise ValidationFailure(f"{path}:{number}: missing filter_name")
        if name in seen:
            raise ValidationFailure(f"{path}:{number}: duplicate expected filter {name!r}")
        names.append(name)
        seen.add(name)
    if len(names) != EXPECTED_FILTERS:
        raise ValidationFailure(
            f"expected filters CSV must contain exactly {EXPECTED_FILTERS} unique filters, "
            f"got {len(names)}"
        )
    return names, sha256_file(path)


def _validate_hashes(value: Any, label: str) -> dict[str, str]:
    hashes = dict(_mapping(value, label))
    keys = frozenset(hashes)
    if keys not in {
        frozenset(REQUIRED_SOURCE_HASHES),
        frozenset(FORMAL_REQUIRED_SOURCE_HASHES),
    }:
        expected = (
            FORMAL_REQUIRED_SOURCE_HASHES
            if {"calibration_workload_csv", "measurement_workload_csv"} & set(hashes)
            else REQUIRED_SOURCE_HASHES
        )
        raise ValidationFailure(
            f"{label} keys mismatch: missing={sorted(expected - set(hashes))} "
            f"extra={sorted(set(hashes) - expected)}"
        )
    for name, digest in hashes.items():
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValidationFailure(f"{label}.{name} is not a SHA256 digest")
    return hashes


def _output_paths_and_hashes(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Path], dict[str, str]]:
    outputs = _mapping(manifest.get("outputs"), f"{manifest_path}: outputs")
    if set(outputs) != set(ALL_OUTPUTS):
        raise ValidationFailure(
            f"{manifest_path}: outputs keys mismatch: "
            f"missing={sorted(set(ALL_OUTPUTS) - set(outputs))} "
            f"extra={sorted(set(outputs) - set(ALL_OUTPUTS))}"
        )
    paths = {
        name: _resolve_recorded_path(manifest_path, outputs[name], f"outputs.{name}")
        for name in ALL_OUTPUTS
    }
    if paths["manifest_json"] != manifest_path:
        raise ValidationFailure(f"{manifest_path}: outputs.manifest_json does not name this manifest")
    declared = dict(_mapping(manifest.get("output_sha256"), f"{manifest_path}: output_sha256"))
    if set(declared) != set(DATA_OUTPUTS):
        raise ValidationFailure(
            f"{manifest_path}: output_sha256 keys must be exactly {list(DATA_OUTPUTS)}"
        )
    actual: dict[str, str] = {}
    for name in DATA_OUTPUTS:
        digest = declared[name]
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValidationFailure(f"{manifest_path}: output_sha256.{name} is invalid")
        if not paths[name].is_file():
            raise ValidationFailure(f"{manifest_path}: missing declared output {name}: {paths[name]}")
        actual[name] = sha256_file(paths[name])
        if actual[name] != digest:
            raise ValidationFailure(f"{manifest_path}: {name} SHA256 does not match manifest")
    return paths, actual


def _service_identity(
    manifest_path: Path, manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, str]:
    service = _mapping(manifest.get("service"), f"{manifest_path}: service")
    version = service.get("version")
    expected_version = service.get("expected_version")
    digest = service.get("image_digest")
    if not isinstance(version, str) or not version or version != expected_version:
        raise ValidationFailure(f"{manifest_path}: immutable Weaviate version gate is invalid")
    if service.get("version_gate_passed") is not True:
        raise ValidationFailure(f"{manifest_path}: service.version_gate_passed is not true")
    if not isinstance(digest, str) or not IMAGE_DIGEST_RE.fullmatch(digest):
        raise ValidationFailure(f"{manifest_path}: service.image_digest is not immutable")
    _exact_int(service.get("count"), EXPECTED_ROWS, f"{manifest_path}: service.count")
    if service.get("measurement_mode") != MEASUREMENT_MODE or service.get("concurrency") != 1:
        raise ValidationFailure(f"{manifest_path}: service is not single-client sequential")
    errors = service.get("errors")
    if not isinstance(errors, list) or errors:
        raise ValidationFailure(f"{manifest_path}: service.errors is not an empty array")

    config_identity = _mapping(
        config.get("service_identity"), f"{manifest_path}: config.service_identity"
    )
    expected_identity = {
        "actual_version": version,
        "expected_version": expected_version,
        "service_image_digest": digest,
    }
    if dict(config_identity) != expected_identity:
        raise ValidationFailure(f"{manifest_path}: config service identity does not match manifest")
    return expected_identity


def _normal_int_list(value: Any, label: str, *, allow_zero: bool = False) -> list[int]:
    items = _sequence(value, label)
    try:
        result = [int(item) for item in items]
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"{label} contains a non-integer") from exc
    if not result or result != sorted(set(result)):
        raise ValidationFailure(f"{label} must be sorted, unique, and non-empty")
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in result):
        raise ValidationFailure(f"{label} contains a value below {minimum}")
    return result


def _candidate_universe(
    manifest_path: Path, manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_value = manifest.get("candidate_universe")
    service = _mapping(manifest.get("service"), f"{manifest_path}: service")
    service_value = service.get("candidate_universe")
    config_value = config.get("candidate_universe")
    attestations = [
        ("manifest.candidate_universe", manifest_value),
        ("service.candidate_universe", service_value),
        ("config.candidate_universe", config_value),
    ]
    mappings: list[tuple[str, Mapping[str, Any]]] = []
    for label, value in attestations:
        if value is not None:
            mappings.append((label, _mapping(value, f"{manifest_path}: {label}")))
    if not mappings:
        raise ValidationFailure(f"{manifest_path}: candidate universe attestation is missing")

    observed = False
    for label, value in mappings:
        if "predicate" in value:
            predicate = " ".join(str(value["predicate"]).strip().split())
            if predicate != CANDIDATE_VALIDITY_PREDICATE:
                raise ValidationFailure(
                    f"{manifest_path}: {label}.predicate must be "
                    f"{CANDIDATE_VALIDITY_PREDICATE!r}"
                )
        for row_field in ("expected_rows", "observed_rows", "rows", "count"):
            if row_field not in value:
                continue
            _exact_int(
                value[row_field],
                EXPECTED_CANDIDATE_ROWS,
                f"{manifest_path}: {label}.{row_field}",
            )
            if row_field in {"observed_rows", "rows", "count"}:
                observed = True

    predicates = {
        " ".join(str(value.get("predicate", "")).strip().split())
        for _, value in mappings
        if value.get("predicate") is not None
    }
    if predicates != {CANDIDATE_VALIDITY_PREDICATE}:
        raise ValidationFailure(
            f"{manifest_path}: every candidate universe attestation must identify "
            f"{CANDIDATE_VALIDITY_PREDICATE!r}"
        )
    return {
        "predicate": CANDIDATE_VALIDITY_PREDICATE,
        "expected_rows": EXPECTED_CANDIDATE_ROWS,
        "observed_rows": EXPECTED_CANDIDATE_ROWS if observed else None,
    }


def _query_cohorts(
    manifest_path: Path, config: Mapping[str, Any], execution_protocol: str
) -> dict[str, list[int] | bool]:
    calibration = _mapping(config.get("calibration"), f"{manifest_path}: config.calibration")
    final = _mapping(config.get("final"), f"{manifest_path}: config.final")
    calibration_query_nos = _normal_int_list(
        calibration.get("query_nos"),
        f"{manifest_path}: config.calibration.query_nos",
        allow_zero=True,
    )
    final_query_nos = _normal_int_list(
        final.get("query_nos"),
        f"{manifest_path}: config.final.query_nos",
        allow_zero=True,
    )
    if execution_protocol == FORMAL_EXECUTION_PROTOCOL:
        expected_calibration = list(FORMAL_CALIBRATION_QUERY_NOS)
        expected_final = list(FORMAL_FINAL_QUERY_NOS)
        expected_calibration_requests = list(FORMAL_CALIBRATION_REQUEST_NOS)
        expected_final_requests = list(FORMAL_FINAL_REQUEST_NOS)
        calibration_request_nos = _normal_int_list(
            calibration.get("request_nos"),
            f"{manifest_path}: config.calibration.request_nos",
            allow_zero=True,
        )
        final_request_nos = _normal_int_list(
            final.get("request_nos"),
            f"{manifest_path}: config.final.request_nos",
            allow_zero=True,
        )
        if calibration_request_nos != expected_calibration_requests:
            raise ValidationFailure(
                f"{manifest_path}: formal calibration request_no coverage is not q200"
            )
        if final_request_nos != expected_final_requests:
            raise ValidationFailure(
                f"{manifest_path}: formal final request_no coverage is not q10k"
            )
    else:
        expected_calibration = list(EXPECTED_CALIBRATION_QUERY_NOS)
        expected_final = list(EXPECTED_FINAL_QUERY_NOS)
        calibration_request_nos = expected_calibration
        final_request_nos = expected_final
    if calibration_query_nos != expected_calibration:
        raise ValidationFailure(f"{manifest_path}: calibration query cohort is incompatible")
    if final_query_nos != expected_final:
        raise ValidationFailure(f"{manifest_path}: final query cohort is incompatible")
    return {
        "reserved_query_nos": (
            [] if execution_protocol == FORMAL_EXECUTION_PROTOCOL else list(range(20))
        ),
        "calibration_request_nos": calibration_request_nos,
        "final_request_nos": final_request_nos,
        "calibration_query_nos": calibration_query_nos,
        "final_query_nos": final_query_nos,
        "query_no_overlap": not bool(
            set(calibration_query_nos) & set(final_query_nos)
        ),
    }


def _frozen_workload_contract(
    manifest_path: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    expected_split: str,
    expected_requests: int,
    expected_query_nos: Sequence[int],
    expected_sha256: str,
) -> dict[str, Any]:
    path = _resolve_recorded_path(manifest_path, record.get("path"), label)
    if record.get("sha256") != expected_sha256 or sha256_file(path) != expected_sha256:
        raise ValidationFailure(f"{manifest_path}: {label} SHA256 mismatch")
    fields, rows = _read_csv(path, label)
    required = {"request_no", "query_no", "query_id", "filter_name", "split"}
    if not required <= set(fields):
        raise ValidationFailure(
            f"{manifest_path}: {label} lacks fields {sorted(required - set(fields))}"
        )
    if len(rows) != expected_requests:
        raise ValidationFailure(
            f"{manifest_path}: {label} expected {expected_requests} rows, got {len(rows)}"
        )
    requests: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=2):
        try:
            request = {
                "request_no": int(row["request_no"]),
                "query_no": int(row["query_no"]),
                "query_id": int(row["query_id"]),
                "filter_name": str(row["filter_name"]),
            }
        except (TypeError, ValueError) as exc:
            raise ValidationFailure(
                f"{manifest_path}: malformed {label} row {number}"
            ) from exc
        if row["split"] != expected_split or not request["filter_name"]:
            raise ValidationFailure(
                f"{manifest_path}: invalid split/filter in {label} row {number}"
            )
        requests.append(request)
    expected_request_nos = set(range(expected_requests))
    expected_queries = {int(value) for value in expected_query_nos}
    observed_request_nos = {request["request_no"] for request in requests}
    observed_query_nos = {request["query_no"] for request in requests}
    if (
        observed_request_nos != expected_request_nos
        or len(observed_request_nos) != len(requests)
        or observed_query_nos != expected_queries
        or len(observed_query_nos) != len(requests)
    ):
        raise ValidationFailure(
            f"{manifest_path}: {label} request/query coverage is incomplete"
        )
    filters = {request["filter_name"] for request in requests}
    if len(filters) != EXPECTED_FILTERS:
        raise ValidationFailure(
            f"{manifest_path}: {label} must cover {EXPECTED_FILTERS} filters"
        )
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "split": expected_split,
        "requests": requests,
        "request_counts_by_filter": {
            name: sum(request["filter_name"] == name for request in requests)
            for name in sorted(filters)
        },
    }


def _run_contract(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    identity: Mapping[str, str],
) -> tuple[list[str], dict[str, Any]]:
    filters = [str(item) for item in _sequence(
        config.get("filter_names"), f"{manifest_path}: config.filter_names"
    )]
    if not filters or any(not name for name in filters) or len(filters) != len(set(filters)):
        raise ValidationFailure(f"{manifest_path}: config.filter_names is empty or duplicated")
    if config.get("source_hashes") != source_hashes:
        raise ValidationFailure(f"{manifest_path}: config source_hashes do not match manifest")
    current_protocol = manifest.get("current_protocol")
    execution_protocol = (
        manifest.get("execution_protocol")
        or config.get("execution_protocol")
        or (
            LEGACY_EXECUTION_PROTOCOL
            if current_protocol == LEGACY_CURRENT_PROTOCOL else None
        )
    )
    if execution_protocol not in {
        FORMAL_EXECUTION_PROTOCOL, LEGACY_EXECUTION_PROTOCOL
    }:
        raise ValidationFailure(f"{manifest_path}: execution protocol is missing")
    expected_current_protocol = (
        CURRENT_PROTOCOL
        if execution_protocol == FORMAL_EXECUTION_PROTOCOL
        else LEGACY_CURRENT_PROTOCOL
    )
    if (
        current_protocol != expected_current_protocol
        or config.get("current_protocol") != expected_current_protocol
    ):
        raise ValidationFailure(f"{manifest_path}: current protocol identity is missing")
    if (
        execution_protocol == FORMAL_EXECUTION_PROTOCOL
        and (
            manifest.get("execution_protocol") != execution_protocol
            or config.get("execution_protocol") != execution_protocol
        )
    ):
        raise ValidationFailure(f"{manifest_path}: formal execution protocol is not explicit")
    required_hashes = (
        FORMAL_REQUIRED_SOURCE_HASHES
        if execution_protocol == FORMAL_EXECUTION_PROTOCOL
        else REQUIRED_SOURCE_HASHES
    )
    if set(source_hashes) != required_hashes:
        raise ValidationFailure(
            f"{manifest_path}: source hashes do not match execution protocol"
        )
    formal = execution_protocol == FORMAL_EXECUTION_PROTOCOL
    manifest_inputs = _mapping(
        manifest.get("input_provenance"), f"{manifest_path}: manifest.input_provenance"
    )
    config_inputs = _mapping(
        config.get("input_provenance"), f"{manifest_path}: config.input_provenance"
    )
    if _canonical(manifest_inputs) != _canonical(config_inputs):
        raise ValidationFailure(f"{manifest_path}: input provenance differs across outputs")
    if manifest_inputs.get("protocol") != expected_current_protocol:
        raise ValidationFailure(f"{manifest_path}: input provenance protocol is incompatible")
    provenance_hash_fields = {
        "truth_manifest": "truth_manifest",
        "import_manifest": "import_manifest",
        "truth_csv": "truth_csv",
        "filters_csv": "filters_csv",
        "vectors_fbin": "fbin",
        "query_cohort_csv": "query_cohort_csv",
        "query_cohort_manifest": "query_cohort_manifest",
        "import_attributes_csv": "import_attributes_csv",
    }
    if execution_protocol == FORMAL_EXECUTION_PROTOCOL:
        provenance_hash_fields.update({
            "calibration_workload_csv": "calibration_workload_csv",
            "measurement_workload_csv": "measurement_workload_csv",
        })
    for record_name, hash_name in provenance_hash_fields.items():
        record = _mapping(
            manifest_inputs.get(record_name),
            f"{manifest_path}: input_provenance.{record_name}",
        )
        digest = record.get("sha256")
        if digest != source_hashes[hash_name]:
            raise ValidationFailure(
                f"{manifest_path}: {record_name} hash is not bound to source_hashes"
            )
        if not isinstance(record.get("path"), str) or not record["path"]:
            raise ValidationFailure(
                f"{manifest_path}: input_provenance.{record_name}.path is missing"
            )
    workloads: dict[str, Any] | None = None
    if formal:
        workloads = {
            "calibration": _frozen_workload_contract(
                manifest_path,
                _mapping(
                    manifest_inputs.get("calibration_workload_csv"),
                    f"{manifest_path}: calibration workload provenance",
                ),
                label="calibration workload",
                expected_split="calibration",
                expected_requests=len(FORMAL_CALIBRATION_REQUEST_NOS),
                expected_query_nos=FORMAL_CALIBRATION_QUERY_NOS,
                expected_sha256=source_hashes["calibration_workload_csv"],
            ),
            "measurement": _frozen_workload_contract(
                manifest_path,
                _mapping(
                    manifest_inputs.get("measurement_workload_csv"),
                    f"{manifest_path}: measurement workload provenance",
                ),
                label="measurement workload",
                expected_split="measurement",
                expected_requests=len(FORMAL_FINAL_REQUEST_NOS),
                expected_query_nos=FORMAL_FINAL_QUERY_NOS,
                expected_sha256=source_hashes["measurement_workload_csv"],
            ),
        }
    corpus_gates = _mapping(
        manifest_inputs.get("corpus_gates"),
        f"{manifest_path}: input_provenance.corpus_gates",
    )
    for field, expected in (
        ("rows", EXPECTED_ROWS),
        ("candidate_rows", EXPECTED_CANDIDATE_ROWS),
        ("filter_count_coverage", EXPECTED_FILTERS),
        ("complete_filter_matrix", EXPECTED_FILTERS),
    ):
        _exact_int(
            corpus_gates.get(field), expected,
            f"{manifest_path}: input_provenance.corpus_gates.{field}",
        )
    timed_response = _mapping(
        config.get("timed_graphql_response"),
        f"{manifest_path}: config.timed_graphql_response",
    )
    if (
        timed_response.get("fields") != TIMED_RESPONSE_FIELDS
        or timed_response.get("all_scalar_properties_requested") is not False
    ):
        raise ValidationFailure(f"{manifest_path}: timed GraphQL payload is not minimal")
    throughput = _mapping(
        config.get("throughput"), f"{manifest_path}: config.throughput"
    )
    if (
        throughput.get("qps_measured") is not False
        or throughput.get("definition") != QPS_DEFINITION
        or throughput.get("latency_reciprocal_is_not_measured_qps") is not True
    ):
        raise ValidationFailure(f"{manifest_path}: QPS reporting contract is invalid")
    run_spec_hash = manifest.get("run_spec_hash")
    if not isinstance(run_spec_hash, str) or not SHA256_RE.fullmatch(run_spec_hash):
        raise ValidationFailure(f"{manifest_path}: run_spec_hash is invalid")
    if config.get("run_spec_hash") != run_spec_hash:
        raise ValidationFailure(f"{manifest_path}: config run_spec_hash does not match manifest")
    if config.get("measurement_mode") != MEASUREMENT_MODE:
        raise ValidationFailure(f"{manifest_path}: config measurement mode is incompatible")
    _exact_int(config.get("vector_rows"), EXPECTED_ROWS, f"{manifest_path}: config.vector_rows")
    _exact_int(config.get("k"), EXPECTED_K, f"{manifest_path}: config.k")
    dimensions = int(_positive_finite(
        config.get("dimensions"), f"{manifest_path}: config.dimensions"
    ))
    strategies = [str(item) for item in _sequence(
        config.get("configured_filter_strategies"),
        f"{manifest_path}: config.configured_filter_strategies",
    )]
    if strategies != ["acorn", "sweeping"]:
        raise ValidationFailure(f"{manifest_path}: production filter strategies are incompatible")
    ef_values = _normal_int_list(config.get("ef_values"), f"{manifest_path}: config.ef_values")
    cutoffs = _normal_int_list(
        config.get("flat_search_cutoffs"),
        f"{manifest_path}: config.flat_search_cutoffs",
        allow_zero=True,
    )
    if cutoffs[0] != 0:
        raise ValidationFailure(f"{manifest_path}: flat_search_cutoffs must start at zero")
    targets = [_target(item, f"{manifest_path}: config.targets") for item in _sequence(
        config.get("targets"), f"{manifest_path}: config.targets"
    )]
    if targets != list(EXPECTED_TARGETS):
        raise ValidationFailure(f"{manifest_path}: config targets must be exactly {list(EXPECTED_TARGETS)}")
    dominance_guard = _positive_finite(
        config.get("hnsw_dominance_guard"),
        f"{manifest_path}: config.hnsw_dominance_guard",
    )
    if dominance_guard < 1.0:
        raise ValidationFailure(f"{manifest_path}: hnsw dominance guard is below one")

    calibration = _mapping(config.get("calibration"), f"{manifest_path}: config.calibration")
    final = _mapping(config.get("final"), f"{manifest_path}: config.final")
    checkpoint = _mapping(config.get("checkpoint"), f"{manifest_path}: config.checkpoint")
    calibration_queries = (
        len(FORMAL_CALIBRATION_QUERY_NOS)
        if formal else len(EXPECTED_CALIBRATION_QUERY_NOS)
    )
    final_queries = (
        len(FORMAL_FINAL_QUERY_NOS) if formal else len(EXPECTED_FINAL_QUERY_NOS)
    )
    calibration_repeats = FORMAL_CALIBRATION_REPEATS if formal else 2
    final_repeats = FORMAL_FINAL_REPEATS if formal else 5
    _exact_int(
        calibration.get("queries"),
        calibration_queries,
        f"{manifest_path}: calibration.queries",
    )
    _exact_int(
        calibration.get("repeats"), calibration_repeats,
        f"{manifest_path}: calibration.repeats",
    )
    _exact_int(final.get("queries"), final_queries, f"{manifest_path}: final.queries")
    _exact_int(final.get("repeats"), final_repeats, f"{manifest_path}: final.repeats")
    required_calibration_contract = {
        "selection_rule": CALIBRATION_SELECTION_RULE,
        "selection_policy": CALIBRATION_SELECTION_POLICY,
        "qualification_metric": CALIBRATION_STOP_METRIC,
        "qualification_operator": ">= target",
        "bootstrap_ci_lcb": "qualification_and_early_stop",
        "fallback": CALIBRATION_FALLBACK,
        "fallback_requires_full_hnsw_grid": True,
        "early_stop_metric": CALIBRATION_STOP_METRIC,
        "early_stop_target": EXPECTED_TARGETS[-1],
    }
    for field, expected in required_calibration_contract.items():
        actual = calibration.get(field)
        if actual != expected:
            raise ValidationFailure(
                f"{manifest_path}: calibration.{field} does not prove the formal "
                f"LCB95 selection protocol: expected={expected!r} actual={actual!r}"
            )
    for field, expected in (
        ("runs_selected_system_configs_plus_flat_exactness_controls", True),
        ("reuses_one_exact_measurement_for_multiple_targets", not formal),
        ("retunes_after_held_out_measurement", False),
    ):
        if final.get(field) is not expected:
            raise ValidationFailure(f"{manifest_path}: final.{field} is incompatible")
    dedupe = final.get("deduplication_key")
    expected_dedupe = [
        "configured_filter_strategy", "filter_name", "flat_search_cutoff", "ef",
    ] + (["target_recall"] if formal else [])
    if dedupe != expected_dedupe:
        raise ValidationFailure(f"{manifest_path}: final deduplication contract is incompatible")
    if formal and final.get("target_specific_measurement") is not True:
        raise ValidationFailure(
            f"{manifest_path}: formal final must measure every target independently"
        )
    for field in ("schedule_order", "selection_rule"):
        if not isinstance(calibration.get(field), str) or not calibration[field]:
            raise ValidationFailure(f"{manifest_path}: calibration.{field} is missing")
    checkpoint_contract = {
        field: checkpoint.get(field)
        for field in ("persistence", "storage", "complete_block_boundary")
    }
    if checkpoint_contract != {
        "persistence": "atomic complete-block snapshot",
        "storage": "single JSON snapshot",
        "complete_block_boundary": True,
    }:
        raise ValidationFailure(f"{manifest_path}: checkpoint contract is incompatible")
    effective = _mapping(
        config.get("effective_cutoffs_by_filter"),
        f"{manifest_path}: config.effective_cutoffs_by_filter",
    )
    if set(effective) != set(filters):
        raise ValidationFailure(f"{manifest_path}: effective cutoff filters do not match shard")
    for name in filters:
        representatives = _normal_int_list(
            effective[name], f"{manifest_path}: effective cutoff for {name}", allow_zero=True
        )
        if len(representatives) != 2 or representatives[0] != 0:
            raise ValidationFailure(f"{manifest_path}: effective cutoff proof is incomplete for {name}")

    contract = {
        "class": config.get("class"),
        "git_revision": config.get("git_revision"),
        "source_hashes": dict(source_hashes),
        "current_protocol": expected_current_protocol,
        "execution_protocol": execution_protocol,
        "input_provenance": dict(manifest_inputs),
        "vector_rows": EXPECTED_ROWS,
        "dimensions": dimensions,
        "k": EXPECTED_K,
        "configured_filter_strategies": strategies,
        "ef_values": ef_values,
        "flat_search_cutoffs": cutoffs,
        "targets": targets,
        "hnsw_dominance_guard": dominance_guard,
        "service_identity": dict(identity),
        "measurement_mode": MEASUREMENT_MODE,
        "timed_graphql_response": dict(timed_response),
        "throughput": dict(throughput),
        "calibration": {
            "queries": calibration_queries,
            "repeats": calibration_repeats,
            "schedule_order": calibration["schedule_order"],
            "selection_rule": calibration["selection_rule"],
            "selection_policy": calibration["selection_policy"],
            "qualification_metric": calibration["qualification_metric"],
            "qualification_operator": calibration["qualification_operator"],
            "bootstrap_ci_lcb": calibration["bootstrap_ci_lcb"],
            "fallback": calibration["fallback"],
            "fallback_requires_full_hnsw_grid": True,
            "early_stop_metric": calibration["early_stop_metric"],
            "early_stop_target": float(calibration["early_stop_target"]),
        },
        "final": dict(final),
        "checkpoint": checkpoint_contract,
        "workloads": workloads,
    }
    if not isinstance(contract["class"], str) or not contract["class"]:
        raise ValidationFailure(f"{manifest_path}: config.class is missing")
    if not isinstance(contract["git_revision"], str) or not contract["git_revision"]:
        raise ValidationFailure(f"{manifest_path}: config.git_revision is missing")
    if manifest.get("git_revision") != contract["git_revision"]:
        raise ValidationFailure(f"{manifest_path}: git_revision does not match config")
    contract["dataset"] = DATASET
    contract["candidate_universe"] = _candidate_universe(manifest_path, manifest, config)
    contract["query_splits"] = _query_cohorts(
        manifest_path, config, execution_protocol
    )
    contract["latency_scope"] = LATENCY_SCOPE
    contract["latency_definition"] = LATENCY_DEFINITION
    contract["runner_provenance"] = {
        "runner_sha256": source_hashes["runner"],
        "baseline_runner_sha256": source_hashes["baseline_runner"],
        "git_revision": contract["git_revision"],
    }
    contract["service_provenance"] = {
        "version": identity["actual_version"],
        "expected_version": identity["expected_version"],
        "service_image_digest": identity["service_image_digest"],
    }
    return filters, contract


def _validate_raw_rows(
    manifest_path: Path,
    rows: Sequence[Mapping[str, str]],
    filters: set[str],
    contract: Mapping[str, Any],
) -> tuple[set[tuple[str, str, str, int, int, str]], dict[int, int]]:
    formal = contract.get("execution_protocol") == FORMAL_EXECUTION_PROTOCOL
    required = {
        "phase", "configured_filter_strategy", "filter_name", "ef", "flat_search_cutoff",
        "query_no", "query_id", "repeat", "end_to_end_ms", "latency_definition",
        "recall_at_10", "recall_contract", "retry_count", "order_error", "valid", "error",
    }
    if formal:
        required.update({"request_no", "target_recall"})
    if not rows:
        raise ValidationFailure(f"{manifest_path}: raw CSV is empty")
    seen: set[tuple[str, ...]] = set()
    final_blocks: set[tuple[str, str, str, int, int, str]] = set()
    block_pairs: dict[
        tuple[str, str, str, int, int, str], set[tuple[int, int, int]]
    ] = {}
    query_ids: dict[int, int] = {}
    workloads = contract.get("workloads")
    expected_by_phase_filter: dict[
        tuple[str, str], set[tuple[int, int]]
    ] = {}
    if formal:
        if not isinstance(workloads, Mapping):
            raise ValidationFailure(f"{manifest_path}: formal workload contract is missing")
        for phase, workload_name in (
            ("calibration", "calibration"),
            ("final", "measurement"),
        ):
            workload = _mapping(
                workloads.get(workload_name),
                f"{manifest_path}: {workload_name} workload",
            )
            for request in _sequence(
                workload.get("requests"),
                f"{manifest_path}: {workload_name} requests",
            ):
                item = _mapping(request, f"{manifest_path}: workload request")
                name = str(item["filter_name"])
                if name in filters:
                    expected_by_phase_filter.setdefault((phase, name), set()).add(
                        (int(item["request_no"]), int(item["query_no"]))
                    )
    for number, row in enumerate(rows, start=2):
        absent = sorted(field for field in required if field not in row)
        if absent:
            raise ValidationFailure(f"{manifest_path}: raw row {number} lacks fields {absent}")
        missing = sorted(
            field for field in required - {"order_error", "error"} if row[field] == ""
        )
        if missing:
            raise ValidationFailure(f"{manifest_path}: raw row {number} missing fields {missing}")
        phase = row["phase"]
        if phase not in {"warmup", "final_warmup", "calibration", "final"}:
            raise ValidationFailure(f"{manifest_path}: raw row {number} has invalid phase {phase!r}")
        if row["filter_name"] not in filters:
            raise ValidationFailure(f"{manifest_path}: raw row {number} has foreign filter")
        if row["latency_definition"] != LATENCY_DEFINITION:
            raise ValidationFailure(f"{manifest_path}: raw row {number} timing contract changed")
        if row["recall_contract"] != RECALL_CONTRACT:
            raise ValidationFailure(f"{manifest_path}: raw row {number} recall contract changed")
        _positive_finite(row["end_to_end_ms"], f"{manifest_path}: raw row {number} latency")
        if phase in {"calibration", "final"}:
            recall = _positive_finite(
                row["recall_at_10"],
                f"{manifest_path}: raw row {number} recall",
                allow_zero=True,
            )
            if recall > 1.0:
                raise ValidationFailure(f"{manifest_path}: raw row {number} recall exceeds one")
            _csv_true(row["valid"], f"{manifest_path}: raw row {number} valid")
            _csv_zero(row["retry_count"], f"{manifest_path}: raw row {number} retry_count")
            if row.get("error", "").strip() or row.get("order_error", "").strip():
                raise ValidationFailure(f"{manifest_path}: raw row {number} records an error")
        query_no = int(row["query_no"])
        query_id = int(row["query_id"])
        request_no = int(row.get("request_no", query_no))
        repeat = int(row["repeat"])
        previous_query_id = query_ids.setdefault(query_no, query_id)
        if previous_query_id != query_id:
            raise ValidationFailure(
                f"{manifest_path}: query_no {query_no} maps to multiple query_id values"
            )
        target = str(row.get("target_recall", "N/A"))
        if formal:
            if phase == "calibration" and target != "N/A":
                raise ValidationFailure(
                    f"{manifest_path}: calibration row has a target"
                )
            if phase == "final" and target != "N/A":
                _target(target, f"{manifest_path}: raw final target")
        block = (
            phase, row["configured_filter_strategy"], row["filter_name"],
            int(row["flat_search_cutoff"]), int(row["ef"]),
            target if formal else "N/A",
        )
        key = (
            phase, row["configured_filter_strategy"], row["filter_name"],
            row["flat_search_cutoff"], row["ef"], target if formal else "N/A",
            str(request_no), row["query_no"], row["repeat"],
        )
        if key in seen:
            raise ValidationFailure(f"{manifest_path}: duplicate raw measurement pair {key!r}")
        seen.add(key)
        if phase in {"calibration", "final"}:
            block_pairs.setdefault(block, set()).add((request_no, query_no, repeat))
        if phase == "final":
            final_blocks.add(block)
    for block, pairs in block_pairs.items():
        phase, _, filter_name, _, _, _ = block
        expected_repeats = (
            int(contract["calibration"]["repeats"])
            if phase == "calibration"
            else int(contract["final"]["repeats"])
        )
        if formal:
            expected_requests = expected_by_phase_filter.get((phase, filter_name), set())
        else:
            expected_query_nos = (
                EXPECTED_CALIBRATION_QUERY_NOS
                if phase == "calibration" else EXPECTED_FINAL_QUERY_NOS
            )
            expected_requests = {
                (query_no, query_no) for query_no in expected_query_nos
            }
        expected_pairs = {
            (request_no, query_no, repeat)
            for request_no, query_no in expected_requests
            for repeat in range(expected_repeats)
        }
        if pairs != expected_pairs or len(pairs) != len(expected_pairs):
            raise ValidationFailure(
                f"{manifest_path}: missing/duplicate raw pairs for block {block!r}: "
                f"expected={len(expected_pairs)} observed={len(pairs)}"
            )
    return final_blocks, query_ids


def _validate_target_records(
    manifest_path: Path, manifest: Mapping[str, Any], filters: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[tuple[str, float], dict[str, Any]]]:
    selection = _mapping(
        manifest.get("calibration_selection"), f"{manifest_path}: calibration_selection"
    )
    required_selection_evidence = {
        "rule": CALIBRATION_SELECTION_RULE,
        "selection_policy": CALIBRATION_SELECTION_POLICY,
        "qualification_metric": CALIBRATION_STOP_METRIC,
        "qualification_operator": ">= target",
        "bootstrap_ci_lcb": "qualification_and_early_stop",
        "lcb_is_report_only": False,
        "fallback": CALIBRATION_FALLBACK,
        "fallback_requires_full_hnsw_grid": True,
        "early_stop_metric": CALIBRATION_STOP_METRIC,
        "early_stop_target": EXPECTED_TARGETS[-1],
    }
    for field, expected in required_selection_evidence.items():
        if selection.get(field) != expected:
            raise ValidationFailure(
                f"{manifest_path}: calibration_selection.{field} does not prove "
                f"the formal LCB95 protocol"
            )
    records = _sequence(selection.get("targets"), f"{manifest_path}: calibration_selection.targets")
    target_map: dict[tuple[str, float], dict[str, Any]] = {}
    normalized: list[dict[str, Any]] = []
    for number, value in enumerate(records):
        record = dict(_mapping(value, f"{manifest_path}: target record {number}"))
        name = record.get("filter_name")
        if name not in filters:
            raise ValidationFailure(f"{manifest_path}: target record has foreign filter {name!r}")
        target = _target(record.get("target_recall"), f"{manifest_path}: target record")
        key = (str(name), target)
        if key in target_map:
            raise ValidationFailure(f"{manifest_path}: duplicate target pair {key!r}")
        status = record.get("status")
        if status not in ALLOWED_TARGET_STATUSES:
            raise ValidationFailure(f"{manifest_path}: invalid target status for {key!r}: {status!r}")
        for field in (
            "selected_filter_strategy", "selected_ef", "selected_flat_search_cutoff",
            "selection_mode", "calibration_selection_policy",
            "calibration_recall_mean", "calibration_recall_lcb95",
            "calibration_lcb95_qualified", "fallback_used",
        ):
            if record.get(field) in (None, "", "N/A"):
                raise ValidationFailure(f"{manifest_path}: selected target {key!r} lacks {field}")
        if record["calibration_selection_policy"] != CALIBRATION_SELECTION_POLICY:
            raise ValidationFailure(
                f"{manifest_path}: selected target {key!r} has the wrong LCB policy"
            )
        mean = _positive_finite(
            record["calibration_recall_mean"],
            f"{manifest_path}: selected target {key!r} calibration_recall_mean",
            allow_zero=True,
        )
        lcb = _positive_finite(
            record["calibration_recall_lcb95"],
            f"{manifest_path}: selected target {key!r} calibration_recall_lcb95",
            allow_zero=True,
        )
        if mean > 1.0 or lcb > mean:
            raise ValidationFailure(
                f"{manifest_path}: selected target {key!r} has invalid recall evidence"
            )
        mode = record["selection_mode"]
        if mode == LCB_SELECTION_MODE:
            if (record["calibration_lcb95_qualified"] is not True
                    or record["fallback_used"] is not False or lcb < target):
                raise ValidationFailure(
                    f"{manifest_path}: selected target {key!r} lacks LCB95 qualification"
                )
        elif mode == FALLBACK_SELECTION_MODE:
            if (record["calibration_lcb95_qualified"] is not False
                    or record["fallback_used"] is not True or lcb >= target):
                raise ValidationFailure(
                    f"{manifest_path}: selected target {key!r} has an invalid fallback claim"
                )
        else:
            raise ValidationFailure(
                f"{manifest_path}: selected target {key!r} has unknown selection mode {mode!r}"
            )
        normalized_record = {**record, "filter_name": str(name), "target_recall": target}
        target_map[key] = normalized_record
        normalized.append(normalized_record)
    expected = {(name, target) for name in filters for target in EXPECTED_TARGETS}
    if set(target_map) != expected:
        raise ValidationFailure(
            f"{manifest_path}: target pair coverage mismatch: "
            f"missing={sorted(expected - set(target_map))} extra={sorted(set(target_map) - expected)}"
        )
    return normalized, target_map


def _validate_summary_rows(
    manifest_path: Path,
    rows: Sequence[Mapping[str, str]],
    filters: set[str],
    target_map: Mapping[tuple[str, float], Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, int]:
    formal = contract.get("execution_protocol") == FORMAL_EXECUTION_PROTOCOL
    workload_counts: dict[tuple[str, str], int] = {}
    if formal:
        workloads = _mapping(contract.get("workloads"), "formal workload contract")
        for phase, name in (("calibration", "calibration"), ("final", "measurement")):
            workload = _mapping(workloads.get(name), f"{name} workload")
            counts = _mapping(
                workload.get("request_counts_by_filter"),
                f"{name} request counts",
            )
            workload_counts.update(
                {(phase, str(filter_name)): int(count) for filter_name, count in counts.items()}
            )
    required = {
        "phase", "configured_filter_strategy", "filter_name", "ef", "flat_search_cutoff",
        "expected_queries", "expected_repeats", "expected_samples", "observed_samples",
        "error_count", "duplicate_pairs", "missing_pairs", "complete", "latency_definition",
        "service_qps_definition", "recall_mean", "recall_lcb95", "recall_ci95_low",
        "recall_ci95_high", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
        "latency_p99_ms", "latency_ci95_low_ms", "latency_ci95_high_ms",
        "single_client_service_qps", "target_recall", "target_status", "target_outcome",
        "comparison_status", "selected_ef", "selected_flat_search_cutoff",
        "calibration_recall_mean", "calibration_recall_lcb95",
        "calibration_selection_policy", "selection_mode",
        "calibration_lcb95_qualified", "fallback_used", "qps_measured",
    }
    if not rows:
        raise ValidationFailure(f"{manifest_path}: summary CSV is empty")
    calibration_blocks: set[tuple[str, str, int, int]] = set()
    calibration_rows: dict[str, list[Mapping[str, str]]] = {
        name: [] for name in filters
    }
    final_pairs: dict[tuple[str, float], Mapping[str, str]] = {}
    outcomes = {
        "selected_and_confirmed": 0,
        "selected_but_final_unconfirmed": 0,
        "unattainable_on_grid": sum(
            record["status"] == "unattainable_on_grid" for record in target_map.values()
        ),
    }
    for number, row in enumerate(rows, start=2):
        missing = sorted(field for field in required if field not in row)
        if missing:
            raise ValidationFailure(f"{manifest_path}: summary header lacks fields {missing}")
        phase = row["phase"]
        if phase not in {"calibration", "final"}:
            raise ValidationFailure(f"{manifest_path}: summary row {number} has invalid phase")
        if row["filter_name"] not in filters:
            raise ValidationFailure(f"{manifest_path}: summary row {number} has foreign filter")
        for field in ("error_count", "duplicate_pairs", "missing_pairs"):
            _csv_zero(row[field], f"{manifest_path}: summary row {number} {field}")
        _csv_true(row["complete"], f"{manifest_path}: summary row {number} complete")
        if row["latency_definition"] != LATENCY_DEFINITION:
            raise ValidationFailure(f"{manifest_path}: summary row {number} timing contract changed")
        if row["service_qps_definition"] != QPS_DEFINITION:
            raise ValidationFailure(f"{manifest_path}: summary row {number} QPS contract changed")
        if _csv_bool_value(
            row["qps_measured"], f"{manifest_path}: summary row {number} qps_measured"
        ):
            raise ValidationFailure(f"{manifest_path}: summary row {number} claims measured QPS")
        if row["single_client_service_qps"] not in {"", "N/A"}:
            raise ValidationFailure(
                f"{manifest_path}: summary row {number} derives QPS from latency"
            )
        expected_queries = int(row["expected_queries"])
        expected_repeats = int(row["expected_repeats"])
        expected_samples = int(row["expected_samples"])
        observed_samples = int(row["observed_samples"])
        if expected_samples != expected_queries * expected_repeats or observed_samples != expected_samples:
            raise ValidationFailure(f"{manifest_path}: summary row {number} sample pairs are incomplete")
        expected_contract = (
            (
                workload_counts[(phase, row["filter_name"])],
                FORMAL_CALIBRATION_REPEATS
                if phase == "calibration" else FORMAL_FINAL_REPEATS,
            )
            if formal else
            (
                (len(EXPECTED_CALIBRATION_QUERY_NOS), 2)
                if phase == "calibration"
                else (len(EXPECTED_FINAL_QUERY_NOS), 5)
            )
        )
        if (expected_queries, expected_repeats) != expected_contract:
            raise ValidationFailure(
                f"{manifest_path}: summary row {number} query/repeat contract changed"
            )
        for field in (
            "recall_mean", "recall_lcb95", "recall_ci95_low", "recall_ci95_high",
        ):
            metric = _positive_finite(
                row[field], f"{manifest_path}: summary row {number} {field}", allow_zero=True
            )
            if metric > 1.0:
                raise ValidationFailure(f"{manifest_path}: summary row {number} {field} exceeds one")
        if float(row["recall_lcb95"]) > float(row["recall_mean"]):
            raise ValidationFailure(
                f"{manifest_path}: summary row {number} recall_lcb95 exceeds recall_mean"
            )
        for field in (
            "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
            "latency_ci95_low_ms", "latency_ci95_high_ms",
        ):
            _positive_finite(row[field], f"{manifest_path}: summary row {number} {field}")

        if phase == "calibration":
            key = (
                row["configured_filter_strategy"], row["filter_name"],
                int(row["flat_search_cutoff"]), int(row["ef"]),
            )
            if key in calibration_blocks:
                raise ValidationFailure(f"{manifest_path}: duplicate calibration summary block {key!r}")
            calibration_blocks.add(key)
            calibration_rows[row["filter_name"]].append(row)
            if row["target_recall"] not in {"", "N/A"}:
                raise ValidationFailure(f"{manifest_path}: calibration summary has a target pair")
            continue

        target = _target(row["target_recall"], f"{manifest_path}: final summary row {number}")
        pair = (row["filter_name"], target)
        if pair in final_pairs:
            raise ValidationFailure(f"{manifest_path}: duplicate final target pair {pair!r}")
        record = target_map.get(pair)
        if record is None or record["status"] != "selected":
            raise ValidationFailure(f"{manifest_path}: unexpected final target pair {pair!r}")
        if row["target_status"] != "selected":
            raise ValidationFailure(f"{manifest_path}: final target {pair!r} is not selected")
        outcome = row["target_outcome"]
        if outcome not in ALLOWED_TARGET_OUTCOMES:
            raise ValidationFailure(f"{manifest_path}: final target {pair!r} has invalid outcome")
        expected_comparison = (
            "confirmed" if outcome == "selected_and_confirmed" else "unconfirmed"
        )
        if row["comparison_status"] != expected_comparison:
            raise ValidationFailure(f"{manifest_path}: final target {pair!r} comparison is invalid")
        if (
            row["configured_filter_strategy"] != str(record["selected_filter_strategy"])
            or int(row["selected_ef"]) != int(record["selected_ef"])
            or int(row["selected_flat_search_cutoff"])
            != int(record["selected_flat_search_cutoff"])
        ):
            raise ValidationFailure(f"{manifest_path}: final target {pair!r} selection changed")
        if row["selection_mode"] != record["selection_mode"]:
            raise ValidationFailure(f"{manifest_path}: final target {pair!r} selection mode changed")
        if row["calibration_selection_policy"] != CALIBRATION_SELECTION_POLICY:
            raise ValidationFailure(f"{manifest_path}: final target {pair!r} LCB policy changed")
        if _csv_bool_value(
            row["calibration_lcb95_qualified"],
            f"{manifest_path}: final target {pair!r} calibration_lcb95_qualified",
        ) is not record["calibration_lcb95_qualified"]:
            raise ValidationFailure(f"{manifest_path}: final target {pair!r} LCB status changed")
        if _csv_bool_value(
            row["fallback_used"], f"{manifest_path}: final target {pair!r} fallback_used",
        ) is not record["fallback_used"]:
            raise ValidationFailure(f"{manifest_path}: final target {pair!r} fallback status changed")
        for field in ("calibration_recall_mean", "calibration_recall_lcb95"):
            if not math.isclose(
                float(row[field]), float(record[field]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValidationFailure(
                    f"{manifest_path}: final target {pair!r} {field} changed"
                )
        final_pairs[pair] = row
        outcomes[outcome] += 1

    expected_efs = [int(value) for value in contract["ef_values"]]
    strategies = [str(value) for value in contract["configured_filter_strategies"]]
    highest_target = max(EXPECTED_TARGETS)
    for filter_name, candidates in calibration_rows.items():
        hnsw = [row for row in candidates if int(row["flat_search_cutoff"]) == 0]
        route_full: dict[str, bool] = {}
        for strategy in strategies:
            route = sorted(
                (row for row in hnsw if row["configured_filter_strategy"] == strategy),
                key=lambda row: int(row["ef"]),
            )
            measured = [int(row["ef"]) for row in route]
            if not route or measured != expected_efs[:len(measured)]:
                raise ValidationFailure(
                    f"{manifest_path}: {filter_name}/{strategy} is not an ascending EF prefix"
                )
            qualifying = [
                index for index, row in enumerate(route)
                if float(row["recall_lcb95"]) >= highest_target
            ]
            if qualifying and qualifying[0] != len(route) - 1:
                raise ValidationFailure(
                    f"{manifest_path}: {filter_name}/{strategy} continued after the "
                    "highest-target LCB95 early stop"
                )
            route_full[strategy] = measured == expected_efs

        flat = [row for row in candidates if int(row["flat_search_cutoff"]) != 0]
        if (len(flat) != 1 or float(flat[0]["recall_mean"]) != 1.0
                or float(flat[0]["recall_lcb95"]) != 1.0):
            raise ValidationFailure(
                f"{manifest_path}: {filter_name} lacks one exact flat calibration control"
            )

        for target in EXPECTED_TARGETS:
            record = target_map[(filter_name, target)]
            qualified = [
                row for row in hnsw if float(row["recall_lcb95"]) >= target
            ]
            if qualified:
                expected = min(
                    qualified,
                    key=lambda row: (
                        float(row["latency_mean_ms"]), int(row["flat_search_cutoff"]),
                        int(row["ef"]), str(row["configured_filter_strategy"]),
                    ),
                )
                expected_mode = LCB_SELECTION_MODE
            else:
                if not all(route_full.values()):
                    raise ValidationFailure(
                        f"{manifest_path}: {filter_name}/{target} used max-recall fallback "
                        "before both HNSW EF grids were exhausted"
                    )
                expected = min(
                    hnsw,
                    key=lambda row: (
                        -float(row["recall_mean"]), float(row["latency_mean_ms"]),
                        int(row["ef"]), str(row["configured_filter_strategy"]),
                    ),
                )
                expected_mode = FALLBACK_SELECTION_MODE
            expected_identity = (
                expected["configured_filter_strategy"], int(expected["ef"]),
                int(expected["flat_search_cutoff"]),
            )
            recorded_identity = (
                record["selected_filter_strategy"], int(record["selected_ef"]),
                int(record["selected_flat_search_cutoff"]),
            )
            if recorded_identity != expected_identity or record["selection_mode"] != expected_mode:
                raise ValidationFailure(
                    f"{manifest_path}: {filter_name}/{target} does not match the "
                    "recomputed LCB95-then-max-recall selection"
                )
            if not math.isclose(
                float(record["calibration_recall_mean"]), float(expected["recall_mean"]),
                rel_tol=0.0, abs_tol=1e-12,
            ) or not math.isclose(
                float(record["calibration_recall_lcb95"]), float(expected["recall_lcb95"]),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValidationFailure(
                    f"{manifest_path}: {filter_name}/{target} calibration evidence "
                    "does not match the selected block"
                )
    selected_pairs = set(target_map)
    if set(final_pairs) != selected_pairs:
        raise ValidationFailure(
            f"{manifest_path}: final target pair coverage mismatch: "
            f"missing={sorted(selected_pairs - set(final_pairs))}"
        )
    return outcomes


def _validate_schema_restore(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> str:
    evidence = _mapping(manifest.get("schema"), f"{manifest_path}: schema evidence")
    if evidence.get("original_definition_restored") is not True:
        raise ValidationFailure(f"{manifest_path}: original schema restore is not attested")
    original_hash = evidence.get("original_schema_sha256")
    if not isinstance(original_hash, str) or not SHA256_RE.fullmatch(original_hash):
        raise ValidationFailure(f"{manifest_path}: original_schema_sha256 is invalid")
    checkpoint = _mapping(manifest.get("checkpoint"), f"{manifest_path}: checkpoint evidence")
    if checkpoint.get("original_schema_persisted_before_schema_put") is not True:
        raise ValidationFailure(f"{manifest_path}: pre-mutation schema persistence is not attested")
    config_checkpoint = _mapping(config.get("checkpoint"), f"{manifest_path}: config.checkpoint")
    if config_checkpoint.get("original_schema_sha256") != original_hash:
        raise ValidationFailure(f"{manifest_path}: config original schema hash does not match")
    if schema.get("class") != config.get("class") or schema.get("source_hashes") != source_hashes:
        raise ValidationFailure(f"{manifest_path}: schema artifact provenance does not match")
    records = _sequence(schema.get("records"), f"{manifest_path}: schema.records")
    if not records:
        raise ValidationFailure(f"{manifest_path}: schema.records is empty")
    first = _mapping(records[0], f"{manifest_path}: first schema record")
    last = _mapping(records[-1], f"{manifest_path}: last schema record")
    if first.get("phase") != "original_schema_snapshot" or last.get("phase") != "restore":
        raise ValidationFailure(f"{manifest_path}: schema snapshot/restore boundary is missing")
    original = _mapping(first.get("schema"), f"{manifest_path}: original schema")
    restored = _mapping(last.get("schema"), f"{manifest_path}: restored schema")
    if _sha256_json(original) != original_hash:
        raise ValidationFailure(f"{manifest_path}: original schema hash does not match evidence")
    if _canonical(original) != _canonical(restored):
        raise ValidationFailure(f"{manifest_path}: restored schema differs from original")
    return original_hash


def _validate_flat_gate(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    filters: set[str],
    final_blocks: set[tuple[str, str, str, int, int, str]],
) -> None:
    gate = _mapping(
        manifest.get("flat_held_out_exactness_gate"),
        f"{manifest_path}: flat held-out exactness gate",
    )
    if gate.get("required_recall_mean") != 1.0 or gate.get("required_recall_lcb95") != 1.0:
        raise ValidationFailure(f"{manifest_path}: flat exactness requirements changed")
    records = _sequence(gate.get("records"), f"{manifest_path}: flat exactness records")
    seen: set[str] = set()
    for value in records:
        record = _mapping(value, f"{manifest_path}: flat exactness record")
        name = record.get("filter_name")
        if name not in filters or name in seen:
            raise ValidationFailure(f"{manifest_path}: flat exactness filter coverage is invalid")
        seen.add(str(name))
        if record.get("held_out_recall_mean") != 1.0 or record.get("held_out_recall_lcb95") != 1.0:
            raise ValidationFailure(f"{manifest_path}: flat exactness gate failed for {name}")
        representative = _mapping(
            record.get("representative"), f"{manifest_path}: flat representative"
        )
        block = (
            "final", str(representative.get("configured_filter_strategy")), str(name),
            int(representative.get("flat_search_cutoff")), int(representative.get("ef")),
            "N/A",
        )
        if block not in final_blocks:
            raise ValidationFailure(f"{manifest_path}: flat exactness block is missing for {name}")
    if seen != filters:
        raise ValidationFailure(f"{manifest_path}: flat exactness records do not cover every filter")


def _load_shard(manifest_path: Path, expected_filters_sha256: str) -> Shard:
    manifest_path = manifest_path.resolve()
    if not fnmatch.fnmatch(manifest_path.name, MANIFEST_PATTERN):
        raise ValidationFailure(f"input is not a production matched-recall manifest: {manifest_path}")
    manifest = _read_json(manifest_path, "manifest")
    if manifest.get("artifact_valid") is not True or manifest.get("status") != "complete":
        raise ValidationFailure(
            f"{manifest_path}: artifact_valid/status must be true/'complete'"
        )
    if manifest.get("paper_eligible") is not True:
        raise ValidationFailure(f"{manifest_path}: shard is not paper_eligible")
    if manifest.get("publication_errors") != []:
        raise ValidationFailure(f"{manifest_path}: shard has publication errors")
    if manifest.get("manifest_commit") != "atomic_last":
        raise ValidationFailure(f"{manifest_path}: manifest was not committed atomic-last")
    source_hashes = _validate_hashes(
        manifest.get("source_hashes"), f"{manifest_path}: source_hashes"
    )
    if source_hashes["filters_csv"] != expected_filters_sha256:
        raise ValidationFailure(f"{manifest_path}: expected filters CSV hash is incompatible")
    paths, hashes = _output_paths_and_hashes(manifest_path, manifest)
    config = _read_json(paths["config_json"], "config JSON")
    schema = _read_json(paths["schema_json"], "schema JSON")
    raw_fields, raw_rows = _read_csv(paths["raw_csv"], "raw CSV")
    summary_fields, summary_rows = _read_csv(paths["summary_csv"], "summary CSV")
    _exact_int(manifest.get("raw_rows"), len(raw_rows), f"{manifest_path}: raw_rows")
    _exact_int(
        manifest.get("summary_rows"), len(summary_rows), f"{manifest_path}: summary_rows"
    )
    identity = _service_identity(manifest_path, manifest, config)
    filters, contract = _run_contract(
        manifest_path, manifest, config, source_hashes, identity
    )
    service = _mapping(manifest["service"], f"{manifest_path}: service")
    filter_counts = _mapping(service.get("filter_counts"), f"{manifest_path}: service.filter_counts")
    if set(filter_counts) != set(filters):
        raise ValidationFailure(f"{manifest_path}: service filter counts do not match shard filters")
    for name, count in filter_counts.items():
        number = int(count)
        if number <= 0 or number > EXPECTED_ROWS:
            raise ValidationFailure(f"{manifest_path}: invalid service filter count for {name}")
    targets, target_map = _validate_target_records(manifest_path, manifest, filters)
    final_blocks, query_ids = _validate_raw_rows(
        manifest_path, raw_rows, set(filters), contract
    )
    outcomes = _validate_summary_rows(
        manifest_path, summary_rows, set(filters), target_map, contract
    )
    if contract["execution_protocol"] == FORMAL_EXECUTION_PROTOCOL:
        release = _mapping(
            manifest.get("release_scope"), f"{manifest_path}: release_scope"
        )
        if (
            release.get("kind") != "formal_filter_target_shard"
            or release.get("requested_slice_complete") is not True
            or int(release.get("expected_filter_target_cells", -1))
            != len(filters) * len(EXPECTED_TARGETS)
            or int(release.get("observed_filter_target_cells", -1))
            != len(filters) * len(EXPECTED_TARGETS)
        ):
            raise ValidationFailure(
                f"{manifest_path}: formal requested-slice release gate is incomplete"
            )
    declared_outcomes = _mapping(
        manifest.get("target_outcomes"), f"{manifest_path}: target_outcomes"
    )
    if dict(declared_outcomes) != outcomes:
        raise ValidationFailure(f"{manifest_path}: target outcome counts do not match rows")
    if outcomes["selected_but_final_unconfirmed"] != 0:
        raise ValidationFailure(
            f"{manifest_path}: selected_but_final_unconfirmed cannot be published"
        )
    _validate_flat_gate(manifest_path, manifest, set(filters), final_blocks)
    original_hash = _validate_schema_restore(
        manifest_path, manifest, config, schema, source_hashes
    )
    return Shard(
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        paths=paths,
        hashes=hashes,
        raw_fields=raw_fields,
        raw_rows=raw_rows,
        summary_fields=summary_fields,
        summary_rows=summary_rows,
        config=config,
        schema=schema,
        filters=filters,
        targets=targets,
        contract=contract,
        target_outcomes=outcomes,
        original_schema_sha256=original_hash,
        query_ids=query_ids,
    )


def resolve_manifests(values: Iterable[str | os.PathLike[str]]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        text = os.fspath(value)
        matches = glob.glob(text, recursive=True)
        if not matches and Path(text).is_file():
            matches = [text]
        paths.update(Path(match).resolve() for match in matches if Path(match).is_file())
    result = sorted(paths)
    if len(result) != EXPECTED_SHARDS:
        raise ValidationFailure(
            f"exactly {EXPECTED_SHARDS} distinct input manifests are required, got {len(result)}"
        )
    return result


def _sort_atom(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        try:
            return (1, float(value))
        except (TypeError, ValueError):
            return (2, str(value))


def _raw_sort_key(row: Mapping[str, str], order: Mapping[str, int]) -> tuple[Any, ...]:
    phase_order = {"warmup": 0, "calibration": 1, "final": 2}
    fields = (
        "schedule_index", "configured_filter_strategy", "flat_search_cutoff", "ef",
        "target_recall", "request_no", "query_no", "repeat", "query_id",
    )
    return (
        order[row["filter_name"]], phase_order[row["phase"]],
        *(_sort_atom(row.get(field, "")) for field in fields), _canonical(dict(row)),
    )


def _summary_sort_key(row: Mapping[str, str], order: Mapping[str, int]) -> tuple[Any, ...]:
    phase_order = {"calibration": 0, "final": 1}
    fields = (
        "target_recall", "schedule_index", "configured_filter_strategy",
        "flat_search_cutoff", "ef",
    )
    return (
        order[row["filter_name"]], phase_order[row["phase"]],
        *(_sort_atom(row.get(field, "")) for field in fields), _canonical(dict(row)),
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        target.flush()
        os.fsync(target.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as target:
        json.dump(value, target, indent=2, sort_keys=True, default=str)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def _commit_bundle(staged: Mapping[str, Path], destinations: Mapping[str, Path]) -> None:
    """Commit data first and manifest last, restoring prior files on failure."""
    token = uuid.uuid4().hex
    backups: dict[str, Path] = {}
    replaced: list[str] = []
    try:
        for name in ALL_OUTPUTS:
            destination = destinations[name]
            if destination.exists():
                backup = staged[name].parent / f".{destination.name}.{token}.backup"
                os.replace(destination, backup)
                backups[name] = backup
        for name in ALL_OUTPUTS:
            os.replace(staged[name], destinations[name])
            replaced.append(name)
    except BaseException:
        for name in reversed(replaced):
            destinations[name].unlink(missing_ok=True)
        for name, backup in backups.items():
            if backup.exists():
                os.replace(backup, destinations[name])
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def combine(
    input_manifests: Sequence[Path], expected_filters_csv: Path, out_prefix: Path
) -> dict[str, Path]:
    manifest_paths = sorted({Path(path).resolve() for path in input_manifests})
    if len(manifest_paths) != EXPECTED_SHARDS:
        raise ValidationFailure(
            f"exactly {EXPECTED_SHARDS} distinct input manifests are required, "
            f"got {len(manifest_paths)}"
        )
    expected_filters, expected_filters_sha256 = _expected_filters(expected_filters_csv)
    shards = [_load_shard(path, expected_filters_sha256) for path in manifest_paths]

    reference_contract = shards[0].contract
    for shard in shards[1:]:
        if _canonical(shard.contract) != _canonical(reference_contract):
            raise ValidationFailure(
                f"{shard.manifest_path}: source hashes/run contract are incompatible"
            )
    original_hashes = {shard.original_schema_sha256 for shard in shards}
    if len(original_hashes) != 1:
        raise ValidationFailure("shards do not attest the same original Weaviate schema")
    if len({shard.manifest.get("run_spec_hash") for shard in shards}) != EXPECTED_SHARDS:
        raise ValidationFailure("shard run_spec_hash values are not distinct")
    formal = reference_contract["execution_protocol"] == FORMAL_EXECUTION_PROTOCOL
    if not formal:
        for shard in shards[1:]:
            if shard.query_ids != shards[0].query_ids:
                raise ValidationFailure(
                    f"{shard.manifest_path}: query_id mapping/run contract is incompatible"
                )
    else:
        combined_query_ids: dict[int, int] = {}
        for shard in shards:
            for query_no, query_id in shard.query_ids.items():
                previous = combined_query_ids.setdefault(query_no, query_id)
                if previous != query_id:
                    raise ValidationFailure(
                        f"{shard.manifest_path}: query_id mapping conflicts across shards"
                    )
        expected_query_ids = {
            int(request["query_no"]): int(request["query_id"])
            for workload in reference_contract["workloads"].values()
            for request in workload["requests"]
        }
        if combined_query_ids != expected_query_ids:
            raise ValidationFailure(
                "combined formal query mapping does not cover the frozen q200+q10k workloads"
            )

    all_filters = [name for shard in shards for name in shard.filters]
    duplicates = sorted({name for name in all_filters if all_filters.count(name) > 1})
    if duplicates:
        raise ValidationFailure(f"duplicate filters across shards: {duplicates}")
    if len(all_filters) != EXPECTED_FILTERS or set(all_filters) != set(expected_filters):
        raise ValidationFailure(
            f"filter coverage mismatch: missing={sorted(set(expected_filters) - set(all_filters))} "
            f"extra={sorted(set(all_filters) - set(expected_filters))}"
        )

    if any(shard.raw_fields != shards[0].raw_fields for shard in shards[1:]):
        raise ValidationFailure("raw CSV headers differ between shards")
    if any(shard.summary_fields != shards[0].summary_fields for shard in shards[1:]):
        raise ValidationFailure("summary CSV headers differ between shards")
    filter_order = {name: index for index, name in enumerate(expected_filters)}
    raw_rows = [row for shard in shards for row in shard.raw_rows]
    summary_rows = [row for shard in shards for row in shard.summary_rows]
    raw_rows.sort(key=lambda row: _raw_sort_key(row, filter_order))
    summary_rows.sort(key=lambda row: _summary_sort_key(row, filter_order))
    target_records = [record for shard in shards for record in shard.targets]
    target_records.sort(key=lambda row: (filter_order[row["filter_name"]], row["target_recall"]))
    outcomes = {
        name: sum(shard.target_outcomes[name] for shard in shards)
        for name in (
            "selected_and_confirmed", "selected_but_final_unconfirmed", "unattainable_on_grid"
        )
    }
    raw_identity_fields = (
        "phase", "configured_filter_strategy", "filter_name",
        "flat_search_cutoff", "ef",
        *(("target_recall", "request_no") if formal else ()),
        "query_no", "repeat",
    )
    raw_identities = [
        tuple(str(row.get(field, "")) for field in raw_identity_fields)
        for row in raw_rows
    ]
    raw_duplicate_count = len(raw_identities) - len(set(raw_identities))
    summary_identity_fields = (
        "phase", "configured_filter_strategy", "filter_name",
        "flat_search_cutoff", "ef", "target_recall",
    )
    summary_identities = [
        tuple(str(row.get(field, "")) for field in summary_identity_fields)
        for row in summary_rows
    ]
    summary_duplicate_count = len(summary_identities) - len(set(summary_identities))
    target_identities = [
        (str(record["filter_name"]), float(record["target_recall"]))
        for record in target_records
    ]
    target_duplicate_count = len(target_identities) - len(set(target_identities))
    expected_targets = {
        (name, target) for name in expected_filters for target in EXPECTED_TARGETS
    }
    missing_targets = sorted(expected_targets - set(target_identities))
    extra_targets = sorted(set(target_identities) - expected_targets)
    formal_target_coverage: dict[str, Any] = {}
    formal_coverage_errors: list[str] = []
    if formal:
        measurement_requests = {
            int(request["request_no"]): request
            for request in reference_contract["workloads"]["measurement"]["requests"]
        }
        expected_pairs = {
            (request_no, repeat)
            for request_no in FORMAL_FINAL_REQUEST_NOS
            for repeat in range(FORMAL_FINAL_REPEATS)
        }
        for target in EXPECTED_TARGETS:
            rows = [
                row for row in raw_rows
                if row["phase"] == "final"
                and math.isclose(
                    float(row.get("target_recall", "nan")),
                    target,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            observed: set[tuple[int, int]] = set()
            mismatched = 0
            for row in rows:
                pair = int(row["request_no"]), int(row["repeat"])
                request = measurement_requests.get(pair[0])
                if (
                    request is None
                    or int(row["query_no"]) != int(request["query_no"])
                    or int(row["query_id"]) != int(request["query_id"])
                    or row["filter_name"] != request["filter_name"]
                ):
                    mismatched += 1
                observed.add(pair)
            missing = len(expected_pairs - observed)
            extra = len(observed - expected_pairs)
            duplicates_for_target = len(rows) - len(observed)
            complete = not (
                missing or extra or duplicates_for_target or mismatched
            )
            formal_target_coverage[str(target)] = {
                "expected_samples": len(expected_pairs),
                "observed_samples": len(rows),
                "unique_request_repeat_pairs": len(observed),
                "missing_pairs": missing,
                "extra_pairs": extra,
                "duplicate_pairs": duplicates_for_target,
                "workload_mismatches": mismatched,
                "complete": complete,
            }
            if not complete:
                formal_coverage_errors.append(
                    f"target={target} missing={missing} extra={extra} "
                    f"duplicates={duplicates_for_target} mismatched={mismatched}"
                )
    if (
        raw_duplicate_count
        or summary_duplicate_count
        or target_duplicate_count
        or missing_targets
        or extra_targets
        or outcomes["selected_but_final_unconfirmed"]
        or formal_coverage_errors
    ):
        raise ValidationFailure(
            "combined coverage is incomplete: "
            f"raw_duplicates={raw_duplicate_count} "
            f"summary_duplicates={summary_duplicate_count} "
            f"target_duplicates={target_duplicate_count} "
            f"missing_targets={missing_targets} extra_targets={extra_targets} "
            f"unconfirmed={outcomes['selected_but_final_unconfirmed']} "
            f"formal_coverage_errors={formal_coverage_errors}"
        )
    combined_coverage = {
        "complete": True,
        "expected_shards": EXPECTED_SHARDS,
        "observed_shards": len(shards),
        "expected_filters": EXPECTED_FILTERS,
        "observed_filters": len(all_filters),
        "missing_filters": [],
        "duplicate_filters": [],
        "expected_target_pairs": len(expected_targets),
        "observed_target_pairs": len(target_identities),
        "missing_target_pairs": [],
        "extra_target_pairs": [],
        "duplicate_target_pairs": 0,
        "raw_rows": len(raw_rows),
        "duplicate_raw_measurement_keys": 0,
        "summary_rows": len(summary_rows),
        "duplicate_summary_keys": 0,
        "required_source_hashes": sorted(
            FORMAL_REQUIRED_SOURCE_HASHES if formal else REQUIRED_SOURCE_HASHES
        ),
        "source_hash_coverage": {
            name: all(name in shard.manifest["source_hashes"] for shard in shards)
            for name in sorted(
                FORMAL_REQUIRED_SOURCE_HASHES if formal else REQUIRED_SOURCE_HASHES
            )
        },
        "formal_filter_target_cells": (
            EXPECTED_FILTERS * len(EXPECTED_TARGETS) if formal else None
        ),
        "formal_target_trace_coverage": formal_target_coverage,
        "samples_per_target": (
            len(FORMAL_FINAL_REQUEST_NOS) * FORMAL_FINAL_REPEATS if formal else None
        ),
        "zero_missing": True,
        "zero_duplicate": True,
    }

    out_prefix = Path(out_prefix).resolve()
    destinations = {
        name: out_prefix.with_name(out_prefix.name + suffix)
        for name, suffix in OUTPUT_SUFFIXES.items()
    }
    input_paths = {
        shard.manifest_path for shard in shards
    } | {
        path for shard in shards for path in shard.paths.values()
    } | {Path(expected_filters_csv).resolve()}
    collisions = sorted(str(path) for path in destinations.values() if path in input_paths)
    if collisions:
        raise ValidationFailure(f"combined outputs would overwrite inputs: {collisions}")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    shard_evidence = [
        {
            "manifest": str(shard.manifest_path),
            "manifest_sha256": shard.manifest_sha256,
            "run_spec_hash": shard.manifest["run_spec_hash"],
            "git_revision": shard.manifest.get("git_revision"),
            "source_hashes": shard.manifest.get("source_hashes"),
            "filters": shard.filters,
            "calibration_selection_policy": shard.config.get("calibration", {}).get(
                "selection_policy"
            ),
            "calibration_qualification_metric": shard.config.get("calibration", {}).get(
                "qualification_metric"
            ),
            "calibration_fallback": shard.config.get("calibration", {}).get("fallback"),
            "outputs": {
                name: {
                    "path": str(shard.paths[name]),
                    "sha256": shard.hashes[name],
                    **(
                        {"rows": len(shard.raw_rows)} if name == "raw_csv" else
                        {"rows": len(shard.summary_rows)} if name == "summary_csv" else {}
                    ),
                }
                for name in DATA_OUTPUTS
            },
        }
        for shard in shards
    ]
    combined_config = {
        "artifact": "weaviate_production_matched_recall_combined",
        "dataset": DATASET,
        "source_hashes": reference_contract["source_hashes"],
        "current_protocol": reference_contract["current_protocol"],
        "execution_protocol": reference_contract["execution_protocol"],
        "input_provenance": reference_contract["input_provenance"],
        "run_contract": reference_contract,
        "filter_names": expected_filters,
        "targets": list(EXPECTED_TARGETS),
        "candidate_universe": reference_contract["candidate_universe"],
        "query_splits": reference_contract["query_splits"],
        "latency_scope": LATENCY_SCOPE,
        "latency_definition": LATENCY_DEFINITION,
        "runner_provenance": reference_contract["runner_provenance"],
        "service_provenance": reference_contract["service_provenance"],
        "target_records": target_records,
        "input_shards": [
            {
                "manifest": str(shard.manifest_path),
                "manifest_sha256": shard.manifest_sha256,
                "config": str(shard.paths["config_json"]),
                "config_sha256": shard.hashes["config_json"],
                "run_spec_hash": shard.manifest["run_spec_hash"],
                "filters": shard.filters,
                "calibration_selection_policy": shard.config.get("calibration", {}).get(
                    "selection_policy"
                ),
                "calibration_qualification_metric": shard.config.get("calibration", {}).get(
                    "qualification_metric"
                ),
            }
            for shard in shards
        ],
        "measurement_mode": MEASUREMENT_MODE,
        "timed_graphql_response": reference_contract["timed_graphql_response"],
        "throughput": reference_contract["throughput"],
        "coverage": combined_coverage,
    }
    original_schema = _mapping(
        _sequence(shards[0].schema["records"], "schema records")[0], "original record"
    )["schema"]
    combined_schema = {
        "artifact": "weaviate_production_matched_recall_combined_schema_evidence",
        "class": reference_contract["class"],
        "source_hashes": reference_contract["source_hashes"],
        "original_schema_sha256": shards[0].original_schema_sha256,
        "original_schema": original_schema,
        "original_definition_restored": True,
        "shards": [
            {
                "manifest": str(shard.manifest_path),
                "manifest_sha256": shard.manifest_sha256,
                "schema_json": str(shard.paths["schema_json"]),
                "schema_sha256": shard.hashes["schema_json"],
                "filters": shard.filters,
                "records": shard.schema["records"],
            }
            for shard in shards
        ],
    }

    stage_dir = Path(tempfile.mkdtemp(prefix=f".{out_prefix.name}.", dir=out_prefix.parent))
    try:
        staged = {name: stage_dir / Path(path).name for name, path in destinations.items()}
        _write_csv(staged["raw_csv"], shards[0].raw_fields, raw_rows)
        _write_csv(staged["summary_csv"], shards[0].summary_fields, summary_rows)
        _write_json(staged["config_json"], combined_config)
        _write_json(staged["schema_json"], combined_schema)
        output_hashes = {name: sha256_file(staged[name]) for name in DATA_OUTPUTS}
        combined_manifest = {
            "artifact": "weaviate_production_matched_recall_combined",
            "artifact_valid": True,
            "paper_eligible": True,
            "requested_slice_complete": True,
            "full_release_complete": True,
            "status": "complete",
            "manifest_commit": "atomic_last",
            "validation_errors": [],
            "publication_errors": [],
            "dataset": DATASET,
            "source_hashes": reference_contract["source_hashes"],
            "current_protocol": reference_contract["current_protocol"],
            "execution_protocol": reference_contract["execution_protocol"],
            "input_provenance": reference_contract["input_provenance"],
            "measurement_runner_sha256": reference_contract["source_hashes"]["runner"],
            "expected_filters_csv": {
                "path": str(Path(expected_filters_csv).resolve()),
                "sha256": expected_filters_sha256,
            },
            "run_contract": reference_contract,
            "candidate_validity_predicate": CANDIDATE_VALIDITY_PREDICATE,
            "candidate_universe": reference_contract["candidate_universe"],
            "query_splits": reference_contract["query_splits"],
            "query_cohort": {
                "calibration_query_nos": reference_contract["query_splits"]["calibration_query_nos"],
                "query_nos": reference_contract["query_splits"]["final_query_nos"],
                "gt_hash": reference_contract["source_hashes"]["truth_csv"],
            },
            "latency_scope": LATENCY_SCOPE,
            "latency_definition": LATENCY_DEFINITION,
            "timed_graphql_response": reference_contract["timed_graphql_response"],
            "throughput": reference_contract["throughput"],
            "runner_provenance": reference_contract["runner_provenance"],
            "service_provenance": reference_contract["service_provenance"],
            "software_versions": {
                "measurement_runner_sha256": reference_contract["source_hashes"]["runner"],
                "baseline_runner_sha256": reference_contract["source_hashes"]["baseline_runner"],
            },
            "service": {
                "version": reference_contract["service_identity"]["actual_version"],
                "expected_version": reference_contract["service_identity"]["expected_version"],
                "version_gate_passed": True,
                "image_digest": reference_contract["service_identity"]["service_image_digest"],
                "count": EXPECTED_ROWS,
                "candidate_universe": reference_contract["candidate_universe"],
                "measurement_mode": MEASUREMENT_MODE,
                "concurrency": 1,
                "errors": [],
            },
            "filter_names": expected_filters,
            "target_recalls": list(EXPECTED_TARGETS),
            "calibration_selection": {
                "targets": target_records,
                "scope": "exactly one explicit status per filter/target",
                "policy": CALIBRATION_SELECTION_POLICY,
                "qualification_metric": CALIBRATION_STOP_METRIC,
                "qualification_operator": ">= target",
                "lcb_is_report_only": False,
                "fallback": CALIBRATION_FALLBACK,
                "fallback_requires_full_hnsw_grid": True,
                "early_stop_metric": CALIBRATION_STOP_METRIC,
            },
            "target_outcomes": outcomes,
            "schema": {
                "original_schema_sha256": shards[0].original_schema_sha256,
                "original_definition_restored": True,
                "all_shards_restored": True,
            },
            "input_shards": shard_evidence,
            "coverage": combined_coverage,
            "release_scope": {
                "kind": (
                    "formal_14_filter_3_target_mixed_trace"
                    if formal else "legacy_14_filter_3_target_matrix"
                ),
                "requested_slice_complete": True,
                "full_release_complete": True,
                "filter_target_cells": len(expected_targets),
                "samples_per_target": (
                    len(FORMAL_FINAL_REQUEST_NOS) * FORMAL_FINAL_REPEATS
                    if formal else None
                ),
            },
            "raw_rows": len(raw_rows),
            "summary_rows": len(summary_rows),
            "outputs": {name: str(path) for name, path in destinations.items()},
            "output_sha256": output_hashes,
        }
        _write_json(staged["manifest_json"], combined_manifest)
        _commit_bundle(staged, destinations)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    return destinations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed combiner for four Weaviate production matched-recall shards."
    )
    parser.add_argument("--input-manifests", nargs="+", action="append", required=True)
    parser.add_argument("--expected-filters-csv", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = [item for group in args.input_manifests for item in group]
    try:
        manifests = resolve_manifests(values)
        outputs = combine(manifests, args.expected_filters_csv, args.out_prefix)
    except (OSError, ValidationFailure, ValueError) as exc:
        print(f"artifact_valid=false: {exc}", file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
