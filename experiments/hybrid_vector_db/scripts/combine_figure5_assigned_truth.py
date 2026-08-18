#!/usr/bin/env python3
"""Fail-closed merger for a formal Figure 5 assigned exact-truth workload.

The formal workload has 2,800 Cartesian calibration pairs (14 predicates by
200 queries) followed by 10,000 measurement pairs.  This tool takes newly
computed calibration truth and reuses only the exact matching measurement rows
from a previously audited q10,200 truth artifact.  It never guesses mappings,
repairs malformed truth, or writes a partial artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_FILTERS = 14
CALIBRATION_ROWS = 2_800
CALIBRATION_PER_FILTER = 200
MEASUREMENT_ROWS = 10_000
K = 10
TRUTH_FIELDS = (
    "query_no",
    "query_id",
    "filter_name",
    "predicate",
    "actual_selectivity",
    "candidate_validity_predicate",
    "candidate_validity_provenance",
    "query_validity_predicate",
    "query_validity_provenance",
    "method",
    "k",
    "latency_ms",
    "recall_at_10_exact_filtered",
    "returned",
    "candidates",
    "filtered_rows",
    "search_candidate_rows",
    "result_ids",
    "exact_filtered_topk_ids",
    "exact_filtered_topk_distances_sq",
    "kth_distance_sq",
    "tie_tolerance",
    "strict_closer_count",
    "boundary_tied",
    "self_excluded",
    "candidate_rows",
    "self_excluded_rows",
)
WORKLOAD_FIELDS = (
    "request_no",
    "query_no",
    "query_id",
    "filter_name",
    "trace_cycle",
    "split",
)
TIE_AWARE_FIELDS = (
    "exact_filtered_topk_ids",
    "exact_filtered_topk_distances_sq",
    "kth_distance_sq",
    "tie_tolerance",
    "strict_closer_count",
    "boundary_tied",
)
SEMANTIC_FIELDS = tuple(field for field in TRUTH_FIELDS if field != "latency_ms")


class TruthMergeError(RuntimeError):
    """An input does not prove a complete, compatible exact-truth artifact."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TruthMergeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TruthMergeError(f"JSON object required: {path}")
    return value


def read_csv(path: Path, expected_fields: Sequence[str], label: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = tuple(reader.fieldnames or ())
            if fields != tuple(expected_fields):
                raise TruthMergeError(
                    f"{label} schema drift: expected={list(expected_fields)!r}, observed={list(fields)!r}"
                )
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise TruthMergeError(f"cannot read {label} {path}: {exc}") from exc
    if not rows:
        raise TruthMergeError(f"{label} is empty: {path}")
    if any(None in row for row in rows):
        raise TruthMergeError(f"{label} contains an over-wide row: {path}")
    return rows


def parse_int(value: Any, label: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise TruthMergeError(f"{label} is not an integer: {value!r}") from exc


def parse_float(value: Any, label: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise TruthMergeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise TruthMergeError(f"{label} is not finite: {value!r}")
    return result


def parse_bool(value: Any, label: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise TruthMergeError(f"{label} is not boolean: {value!r}")


def parse_encoded_list(value: Any, label: str) -> list[str]:
    """Read the generator's comma encoding and legacy JSON-list fixtures."""
    text = str(value).strip()
    if not text:
        raise TruthMergeError(f"{label} is empty")
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TruthMergeError(f"{label} is malformed JSON: {value!r}") from exc
        if not isinstance(parsed, list):
            raise TruthMergeError(f"{label} must be a list")
        return [str(item) for item in parsed]
    return [item.strip() for item in text.split(",") if item.strip()]


def key(row: Mapping[str, str]) -> tuple[int, int, str]:
    return (
        parse_int(row.get("query_no"), "query_no"),
        parse_int(row.get("query_id"), "query_id"),
        str(row.get("filter_name", "")).strip(),
    )


def pair_set_sha(keys: Sequence[tuple[int, int, str]]) -> str:
    payload = "".join(f"{query_no}\t{query_id}\t{filter_name}\n" for query_no, query_id, filter_name in sorted(keys))
    return sha256_bytes(payload.encode("utf-8"))


def filter_predicates(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or ())
            if {"filter_name", "predicate"} - fields:
                raise TruthMergeError(f"filters CSV missing filter_name/predicate: {path}")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise TruthMergeError(f"cannot read filters CSV {path}: {exc}") from exc
    result: dict[str, str] = {}
    for row in rows:
        name = str(row.get("filter_name", "")).strip()
        predicate = str(row.get("predicate", "")).strip()
        if not name or not predicate or name in result:
            raise TruthMergeError(f"filters CSV has an empty/duplicate filter: {name!r}")
        result[name] = predicate
    if len(result) != EXPECTED_FILTERS:
        raise TruthMergeError(f"formal Figure 5 requires {EXPECTED_FILTERS} filters, observed={len(result)}")
    return result


def validate_truth_row(row: Mapping[str, str], predicates: Mapping[str, str], source: str) -> None:
    row_key = key(row)
    query_no, _, filter_name = row_key
    if query_no < 0 or not filter_name:
        raise TruthMergeError(f"{source}: invalid truth key={row_key!r}")
    expected_predicate = predicates.get(filter_name)
    if expected_predicate is None:
        raise TruthMergeError(f"{source}: unknown filter {filter_name!r}")
    if str(row.get("predicate", "")).strip() != expected_predicate:
        raise TruthMergeError(f"{source}: predicate/filter mismatch for {row_key!r}")
    if parse_int(row.get("k"), f"{source}.k") != K:
        raise TruthMergeError(f"{source}: truth k must equal {K}")
    if str(row.get("method", "")).strip() != "pre_filter_exact":
        raise TruthMergeError(f"{source}: row method is not pre_filter_exact")
    if parse_float(row.get("recall_at_10_exact_filtered"), f"{source}.recall") != 1.0:
        raise TruthMergeError(f"{source}: row is not exact recall=1 truth")
    if str(row.get("latency_ms", "")).strip():
        parse_float(row.get("latency_ms"), f"{source}.latency_ms")
    for field in TIE_AWARE_FIELDS:
        if not str(row.get(field, "")).strip():
            raise TruthMergeError(f"{source}: missing tie-aware field {field}")
    ids = parse_encoded_list(
        row["exact_filtered_topk_ids"], f"{source}.exact_filtered_topk_ids"
    )
    distances = parse_encoded_list(
        row["exact_filtered_topk_distances_sq"],
        f"{source}.exact_filtered_topk_distances_sq",
    )
    if len(ids) != K or len(distances) != K:
        raise TruthMergeError(f"{source}: invalid exact top-k cardinality")
    if any(parse_int(item, f"{source}.exact id") < 0 for item in ids):
        raise TruthMergeError(f"{source}: invalid exact id")
    parsed_distances = [
        parse_float(value, f"{source}.exact distance") for value in distances
    ]
    if any(right < left for left, right in zip(parsed_distances, parsed_distances[1:])):
        raise TruthMergeError(f"{source}: exact distances are not ordered")
    kth = parse_float(row["kth_distance_sq"], f"{source}.kth_distance_sq")
    if kth < 0:
        raise TruthMergeError(f"{source}: negative kth distance")
    if not math.isclose(kth, parsed_distances[-1], rel_tol=1e-8, abs_tol=1e-12):
        raise TruthMergeError(f"{source}: kth distance differs from exact payload")
    tie_tolerance = parse_float(
        row["tie_tolerance"], f"{source}.tie_tolerance"
    )
    if tie_tolerance < 0:
        raise TruthMergeError(f"{source}: negative tie tolerance")
    strict_closer = parse_int(row["strict_closer_count"], f"{source}.strict_closer_count")
    if strict_closer < 0 or strict_closer >= K:
        raise TruthMergeError(f"{source}: invalid strict closer count")
    if strict_closer != sum(
        value < kth - tie_tolerance for value in parsed_distances
    ):
        raise TruthMergeError(f"{source}: strict closer count is inconsistent")
    parse_bool(row["boundary_tied"], f"{source}.boundary_tied")
    if parse_bool(row.get("self_excluded"), f"{source}.self_excluded"):
        raise TruthMergeError(f"{source}: self-excluded truth is not accepted")


def truth_index(rows: Sequence[dict[str, str]], predicates: Mapping[str, str], source: str) -> dict[tuple[int, int, str], dict[str, str]]:
    result: dict[tuple[int, int, str], dict[str, str]] = {}
    for row in rows:
        validate_truth_row(row, predicates, source)
        row_key = key(row)
        if row_key in result:
            raise TruthMergeError(f"{source}: duplicate truth key={row_key!r}")
        result[row_key] = row
    return result


def validate_truth_manifest(path: Path, truth_path: Path, expected_rows: int, source: str) -> dict[str, Any]:
    manifest = read_json(path)
    output = manifest.get("output")
    coverage = manifest.get("exact_coverage")
    if manifest.get("generator") != "figure5_external_exact_truth.py":
        raise TruthMergeError(f"{source}: unexpected truth generator")
    if not isinstance(output, Mapping) or not isinstance(coverage, Mapping):
        raise TruthMergeError(f"{source}: missing output/exact_coverage manifest blocks")
    if str(output.get("sha256", "")) != sha256_file(truth_path):
        raise TruthMergeError(f"{source}: truth CSV digest does not match its manifest")
    if parse_int(output.get("rows"), f"{source}.output.rows") != expected_rows:
        raise TruthMergeError(f"{source}: manifest output row count mismatch")
    if not bool(coverage.get("complete")):
        raise TruthMergeError(f"{source}: exact coverage is not complete")
    if parse_int(coverage.get("emitted_rows"), f"{source}.emitted_rows") != expected_rows:
        raise TruthMergeError(f"{source}: emitted row count mismatch")
    if parse_int(manifest.get("k"), f"{source}.k") != K:
        raise TruthMergeError(f"{source}: manifest k must equal {K}")
    method = str(coverage.get("method", ""))
    if not (method.startswith("full_base_scan_plus_") and method.endswith("_float32_gemm_topk")):
        raise TruthMergeError(f"{source}: manifest is not exact float32 full-scan truth")
    if bool(coverage.get("self_excluded")):
        raise TruthMergeError(f"{source}: self-excluded manifest truth is not accepted")
    return manifest


def validate_workload_manifest(path: Path, assigned_path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    outputs = manifest.get("outputs")
    formal = manifest.get("formal_paper_calibration")
    construction = manifest.get("construction")
    if not isinstance(outputs, Mapping):
        raise TruthMergeError("assigned workload manifest is missing outputs")
    # The workload builder deliberately publishes a frozen trace before its
    # assigned exact truth exists.  That state is admissible here only when it
    # is explicitly the formal pending-truth contract; this merger supplies the
    # missing proof rather than treating a generic invalid workload as valid.
    if not bool(manifest.get("artifact_valid")):
        truth = manifest.get("truth")
        if not (
            manifest.get("stage") == "trace_pending_truth"
            and isinstance(truth, Mapping)
            and truth.get("contract") == "pending_exact_truth_for_frozen_assigned_pairs_v1"
        ):
            raise TruthMergeError("assigned workload manifest is not an audited or frozen pending-truth artifact")
    assigned = outputs.get("assigned_workload_csv")
    if not isinstance(assigned, Mapping) or str(assigned.get("sha256", "")) != sha256_file(assigned_path):
        raise TruthMergeError("assigned workload digest does not match its manifest")
    if parse_int(assigned.get("rows"), "assigned workload rows") != CALIBRATION_ROWS + MEASUREMENT_ROWS:
        raise TruthMergeError("assigned workload manifest row count mismatch")
    if not isinstance(formal, Mapping) or not bool(formal.get("passed")):
        raise TruthMergeError("formal per-predicate calibration gate did not pass")
    if not isinstance(construction, Mapping) or not isinstance(construction.get("calibration"), Mapping):
        raise TruthMergeError("assigned workload is missing construction.calibration")
    if construction["calibration"].get("protocol") != "formal_per_predicate_cartesian_v1":
        raise TruthMergeError("assigned workload uses a non-formal calibration protocol")
    return manifest


def workload_rows(path: Path, predicates: Mapping[str, str]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rows = read_csv(path, WORKLOAD_FIELDS, "assigned workload")
    if len(rows) != CALIBRATION_ROWS + MEASUREMENT_ROWS:
        raise TruthMergeError(f"assigned workload must contain 12800 rows, observed={len(rows)}")
    by_request: dict[int, dict[str, str]] = {}
    keys: set[tuple[int, int, str]] = set()
    calibration: list[dict[str, str]] = []
    measurement: list[dict[str, str]] = []
    for row in rows:
        request_no = parse_int(row.get("request_no"), "request_no")
        row_key = key(row)
        query_no, _, filter_name = row_key
        if request_no in by_request or row_key in keys:
            raise TruthMergeError(f"assigned workload contains duplicate request/key: {row_key!r}")
        if filter_name not in predicates:
            raise TruthMergeError(f"assigned workload references unknown filter {filter_name!r}")
        split = str(row.get("split", "")).strip()
        expected_split = "calibration" if query_no < 200 else "measurement"
        if split != expected_split:
            raise TruthMergeError(f"assigned workload split/query_no mismatch: {row_key!r}")
        by_request[request_no] = row
        keys.add(row_key)
        (calibration if query_no < 200 else measurement).append(row)
    if set(by_request) != set(range(CALIBRATION_ROWS + MEASUREMENT_ROWS)):
        raise TruthMergeError("assigned workload request_no must be exactly 0..12799")
    if len(calibration) != CALIBRATION_ROWS or len(measurement) != MEASUREMENT_ROWS:
        raise TruthMergeError("assigned workload split cardinalities are invalid")
    if {parse_int(row["query_no"], "query_no") for row in calibration} != set(range(200)):
        raise TruthMergeError("calibration workload query numbers must be exactly 0..199")
    counts = Counter(row["filter_name"] for row in calibration)
    if set(counts) != set(predicates) or any(value != CALIBRATION_PER_FILTER for value in counts.values()):
        raise TruthMergeError("calibration workload is not 14 x 200")
    measurement_qnos = [parse_int(row["query_no"], "query_no") for row in measurement]
    if set(measurement_qnos) != set(range(200, 10_200)) or len(set(measurement_qnos)) != MEASUREMENT_ROWS:
        raise TruthMergeError("measurement workload query numbers must be unique 200..10199")
    return rows, calibration, measurement


def truth_equivalent(left: Mapping[str, str], right: Mapping[str, str]) -> bool:
    """Accept CPU/GPU float32 and tied-ID ordering differences only."""
    exact_fields = (
        "query_no",
        "query_id",
        "filter_name",
        "predicate",
        "candidate_validity_predicate",
        "candidate_validity_provenance",
        "query_validity_predicate",
        "query_validity_provenance",
        "method",
        "k",
        "recall_at_10_exact_filtered",
        "returned",
        "candidates",
        "filtered_rows",
        "search_candidate_rows",
        "strict_closer_count",
        "boundary_tied",
        "self_excluded",
        "candidate_rows",
        "self_excluded_rows",
    )
    if any(str(left[field]) != str(right[field]) for field in exact_fields):
        return False
    if not math.isclose(
        parse_float(left["actual_selectivity"], "left.actual_selectivity"),
        parse_float(right["actual_selectivity"], "right.actual_selectivity"),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        return False
    left_ids = parse_encoded_list(
        left["exact_filtered_topk_ids"], "left.exact_filtered_topk_ids"
    )
    right_ids = parse_encoded_list(
        right["exact_filtered_topk_ids"], "right.exact_filtered_topk_ids"
    )
    if len(left_ids) != len(set(left_ids)) or len(right_ids) != len(set(right_ids)):
        return False
    if (
        set(parse_encoded_list(left["result_ids"], "left.result_ids"))
        != set(left_ids)
        or set(parse_encoded_list(right["result_ids"], "right.result_ids"))
        != set(right_ids)
    ):
        return False
    left_distances = [
        parse_float(value, "left.exact distance")
        for value in parse_encoded_list(
            left["exact_filtered_topk_distances_sq"],
            "left.exact_filtered_topk_distances_sq",
        )
    ]
    right_distances = [
        parse_float(value, "right.exact distance")
        for value in parse_encoded_list(
            right["exact_filtered_topk_distances_sq"],
            "right.exact_filtered_topk_distances_sq",
        )
    ]
    left_by_id = dict(zip(left_ids, left_distances))
    right_by_id = dict(zip(right_ids, right_distances))
    tolerance = max(
        parse_float(left["tie_tolerance"], "left.tie_tolerance"),
        parse_float(right["tie_tolerance"], "right.tie_tolerance"),
        1e-9,
    )
    if any(
        abs(left_by_id[item_id] - right_by_id[item_id]) > tolerance
        for item_id in set(left_by_id) & set(right_by_id)
    ):
        return False
    left_kth = parse_float(left["kth_distance_sq"], "left.kth_distance_sq")
    right_kth = parse_float(right["kth_distance_sq"], "right.kth_distance_sq")
    if abs(left_kth - right_kth) > tolerance:
        return False
    if set(left_ids) == set(right_ids):
        return True
    if not (
        parse_bool(left["boundary_tied"], "left.boundary_tied")
        and parse_bool(right["boundary_tied"], "right.boundary_tied")
    ):
        return False
    left_strict = {
        item_id
        for item_id, distance in left_by_id.items()
        if distance < left_kth
        - parse_float(left["tie_tolerance"], "left.tie_tolerance")
    }
    right_strict = {
        item_id
        for item_id, distance in right_by_id.items()
        if distance < right_kth
        - parse_float(right["tie_tolerance"], "right.tie_tolerance")
    }
    if left_strict != right_strict:
        return False
    boundary_low = min(left_kth, right_kth) - tolerance
    boundary_high = max(left_kth, right_kth) + tolerance
    return all(
        boundary_low <= distance <= boundary_high
        for item_id, distance in (*left_by_id.items(), *right_by_id.items())
        if item_id not in left_strict
    )


def csv_payload(rows: Sequence[Mapping[str, str]]) -> bytes:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=TRUTH_FIELDS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def atomic_publish(payloads: Mapping[Path, bytes], *, overwrite: bool) -> None:
    paths = list(payloads)
    if any(path.exists() for path in paths) and not overwrite:
        raise TruthMergeError("output exists; pass --overwrite")
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for path, payload in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            with os.fdopen(descriptor, "wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            staged[path] = Path(name)
        for path in paths:
            if path.exists():
                descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".backup", dir=path.parent)
                os.close(descriptor)
                backup = Path(name)
                os.replace(path, backup)
                backups[path] = backup
        for path in paths:
            os.replace(staged[path], path)
            published.append(path)
        for parent in {path.parent for path in paths}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        for path, backup in backups.items():
            if backup.exists():
                os.replace(backup, path)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def merge(
    *,
    assigned_workload_csv: Path,
    workload_manifest: Path,
    filters_csv: Path,
    calibration_truth_csv: Path,
    calibration_truth_manifest: Path,
    legacy_truth_csv: Path,
    legacy_truth_manifest: Path,
    output_truth_csv: Path,
    output_manifest: Path,
    execute: bool,
    overwrite: bool,
) -> dict[str, Any]:
    predicates = filter_predicates(filters_csv)
    workload_manifest_value = validate_workload_manifest(workload_manifest, assigned_workload_csv)
    assigned, calibration_workload, measurement_workload = workload_rows(assigned_workload_csv, predicates)
    calibration_manifest = validate_truth_manifest(
        calibration_truth_manifest, calibration_truth_csv, CALIBRATION_ROWS, "calibration truth"
    )
    legacy_manifest = validate_truth_manifest(
        legacy_truth_manifest, legacy_truth_csv, 10_200, "legacy truth"
    )
    calibration_index = truth_index(
        read_csv(calibration_truth_csv, TRUTH_FIELDS, "calibration truth"), predicates, "calibration truth"
    )
    legacy_index = truth_index(
        read_csv(legacy_truth_csv, TRUTH_FIELDS, "legacy truth"), predicates, "legacy truth"
    )
    calibration_keys = {key(row) for row in calibration_workload}
    measurement_keys = {key(row) for row in measurement_workload}
    if set(calibration_index) != calibration_keys:
        missing = sorted(calibration_keys - set(calibration_index))
        extra = sorted(set(calibration_index) - calibration_keys)
        raise TruthMergeError(f"calibration truth keys do not exactly match workload: missing={missing[:3]}, extra={extra[:3]}")
    if len(legacy_index) != 10_200:
        raise TruthMergeError(f"legacy q10200 truth must contain 10200 unique rows, observed={len(legacy_index)}")
    legacy_qnos = {row_key[0] for row_key in legacy_index}
    if legacy_qnos != set(range(10_200)):
        raise TruthMergeError("legacy truth query_no domain must be exactly 0..10199")
    legacy_measurement = {row_key: row for row_key, row in legacy_index.items() if row_key[0] >= 200}
    if set(legacy_measurement) != measurement_keys:
        missing = sorted(measurement_keys - set(legacy_measurement))
        extra = sorted(set(legacy_measurement) - measurement_keys)
        raise TruthMergeError(f"legacy measurement keys do not exactly match workload: missing={missing[:3]}, extra={extra[:3]}")
    overlap = set(calibration_index) & set(legacy_index)
    for row_key in overlap:
        if not truth_equivalent(
            calibration_index[row_key], legacy_index[row_key]
        ):
            raise TruthMergeError(f"cross-source truth conflict for key={row_key!r}")
    # request_no is the frozen publication order; new formal calibration wins on overlap.
    output_rows = [
        calibration_index[row_key] if row_key in calibration_keys else legacy_measurement[row_key]
        for row_key in (key(row) for row in assigned)
    ]
    if len(output_rows) != CALIBRATION_ROWS + MEASUREMENT_ROWS:
        raise TruthMergeError("merged output row count is invalid")
    output_keys = [key(row) for row in output_rows]
    if len(set(output_keys)) != len(output_keys):
        raise TruthMergeError("merged output contains duplicate keys")
    payload = csv_payload(output_rows)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "figure5_assigned_truth_merge",
        "artifact_valid": True,
        "contract": {
            "key": ["query_no", "query_id", "filter_name"],
            "calibration": "formal_per_predicate_cartesian_v1: 14 filters x 200 queries",
            "measurement": "reuse only exact q10k workload key matches from q10200 truth",
            "truth": "exact full-base float32 top-k with tie-aware boundary fields",
            "priority": "new_calibration_truth_over_legacy_truth",
        },
        "workload": {
            "assigned_csv": {"path": str(assigned_workload_csv.resolve()), "sha256": sha256_file(assigned_workload_csv), "rows": len(assigned)},
            "manifest": {"path": str(workload_manifest.resolve()), "sha256": sha256_file(workload_manifest), "artifact_content_sha256": workload_manifest_value.get("outputs", {}).get("manifest_json", {}).get("content_sha256")},
            "pair_set_sha256": pair_set_sha(output_keys),
        },
        "sources": {
            "calibration": {
                "truth_csv": {"path": str(calibration_truth_csv.resolve()), "sha256": sha256_file(calibration_truth_csv), "rows": len(calibration_index)},
                "manifest": {"path": str(calibration_truth_manifest.resolve()), "sha256": sha256_file(calibration_truth_manifest), "output_sha256": calibration_manifest["output"]["sha256"]},
                "computed_rows": CALIBRATION_ROWS,
            },
            "legacy": {
                "truth_csv": {"path": str(legacy_truth_csv.resolve()), "sha256": sha256_file(legacy_truth_csv), "rows": len(legacy_index)},
                "manifest": {"path": str(legacy_truth_manifest.resolve()), "sha256": sha256_file(legacy_truth_manifest), "output_sha256": legacy_manifest["output"]["sha256"]},
                "reused_rows": MEASUREMENT_ROWS,
            },
        },
        "counts": {
            "output_rows": len(output_rows),
            "computed_rows": CALIBRATION_ROWS,
            "reused_rows": MEASUREMENT_ROWS,
            "cross_source_identical_overlaps": len(overlap),
        },
        "output": {"path": str(output_truth_csv.resolve()), "sha256": sha256_bytes(payload), "rows": len(output_rows), "fields": list(TRUTH_FIELDS)},
        "execution": {"execute": bool(execute), "dry_run": not execute},
    }
    if execute:
        atomic_publish(
            {output_truth_csv: payload, output_manifest: canonical_json_bytes(manifest)},
            overwrite=overwrite,
        )
        if sha256_file(output_truth_csv) != manifest["output"]["sha256"]:
            raise TruthMergeError("published truth digest mismatch")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assigned-workload-csv", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--filters-csv", type=Path, required=True)
    parser.add_argument("--calibration-truth-csv", type=Path, required=True)
    parser.add_argument("--calibration-truth-manifest", type=Path, required=True)
    parser.add_argument("--legacy-truth-csv", type=Path, required=True)
    parser.add_argument("--legacy-truth-manifest", type=Path, required=True)
    parser.add_argument("--output-truth-csv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Atomically publish the merged CSV and manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs only with --execute.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.overwrite and not args.execute:
        raise TruthMergeError("--overwrite requires --execute")
    manifest = merge(
        assigned_workload_csv=args.assigned_workload_csv,
        workload_manifest=args.workload_manifest,
        filters_csv=args.filters_csv,
        calibration_truth_csv=args.calibration_truth_csv,
        calibration_truth_manifest=args.calibration_truth_manifest,
        legacy_truth_csv=args.legacy_truth_csv,
        legacy_truth_manifest=args.legacy_truth_manifest,
        output_truth_csv=args.output_truth_csv,
        output_manifest=args.output_manifest,
        execute=args.execute,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
