#!/usr/bin/env python3
"""Build the audited Figure 5 recall/latency and throughput/recall artifact.

The input files are repeat-level summaries produced by measurement runners.
This finalizer does not run PostgreSQL and does not infer method identity from
legacy names.  Every row must explicitly identify either stock pgvector or the
complete D1+D2+D3 SQLens arm.  Throughput is accepted only when it was measured
over a barrier wall-clock interval and is recomputed from completed requests.

The manifest is the commit marker: repeat and point CSVs are published first,
then a manifest binding their SHA-256 digests is atomically installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_VERSION = "sqlens-figure5-frontier-artifact-v1"
SCHEMA_VERSION = "1"
EXPECTED_DATASETS = ("amazon10m", "yfcc10m", "laion25m")
EXPERIMENT_KINDS = ("latency", "throughput")
ARM_MODES = {
    "stock_pgvector": "original",
    "sqlens_full": "design1_bloom_bfs_layout_d3",
}
EXPECTED_REQUESTS = 10_000
MIN_REPEATS = {"latency": 3, "throughput": 6}
MEASURED_THROUGHPUT_SOURCE = "measured_completed_over_barrier_wall_clock"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PARETO_TOLERANCE = 1e-12

REPEAT_FIELDS = (
    "schema_version",
    "run_id",
    "dataset",
    "experiment_kind",
    "arm_id",
    "mode_id",
    "config_id",
    "config_sha256",
    "release_identity_sha256",
    "clients",
    "repeat_id",
    "request_trace_sha256",
    "requests",
    "unique_queries",
    "completed_queries",
    "error_count",
    "wall_clock_seconds",
    "recall_mean",
    "recall_ci95_low",
    "recall_ci95_high",
    "latency_mean_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_qps",
    "throughput_ci95_low",
    "throughput_ci95_high",
    "throughput_source",
    "status",
)

# These fields are optional in legacy converter CSVs.  When a converter emits
# them, the finalizer treats them as assertions rather than decorative data.
OPTIONAL_PROVENANCE_FIELDS = {
    "release_contract_path",
    "release_contract_sha256",
    "release_contract_id",
    "release_build_id",
    "release_vector_so_sha256",
    "source_manifest_path",
    "source_manifest_sha256",
}

POINT_FIELDS = (
    "schema_version",
    "point_id",
    "run_id",
    "dataset",
    "experiment_kind",
    "arm_id",
    "mode_id",
    "config_id",
    "config_sha256",
    "release_identity_sha256",
    "clients",
    "repeats",
    "requests_per_repeat",
    "recall_mean",
    "recall_ci95_low",
    "recall_ci95_high",
    "latency_mean_ms",
    "latency_ci95_low_ms",
    "latency_ci95_high_ms",
    "throughput_qps",
    "throughput_ci95_low",
    "throughput_ci95_high",
    "pareto",
    "is_plot_eligible",
)

REQUIRED_INPUT_FIELDS = set(REPEAT_FIELDS) - {
    "throughput_ci95_low",
    "throughput_ci95_high",
}
FORBIDDEN_FIELD_FRAGMENTS = (
    "single_client_throughput",
    "derived_throughput",
    "qps_from_latency",
)
FORBIDDEN_METHOD_VALUES = {
    "d1",
    "d1+d2",
    "design1",
    "design1_bloom",
    "design1_bloom_bfs_layout",
    "sqlens_d1",
    "sqlens_d1_d2",
}


class Figure5ArtifactError(ValueError):
    """An input cannot be admitted to the formal Figure 5 artifact."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: object, label: str) -> str:
    text = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(text):
        raise Figure5ArtifactError(f"{label} is not a SHA-256 value")
    return text


def _text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise Figure5ArtifactError(f"{label} is empty")
    return text


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    text = str(value or "").strip()
    try:
        result = int(text)
    except (TypeError, ValueError) as exc:
        raise Figure5ArtifactError(f"{label} is not an integer: {value!r}") from exc
    if str(result) != text and text not in {f"+{result}", f"-{abs(result)}"}:
        raise Figure5ArtifactError(f"{label} is not a canonical integer: {value!r}")
    if result < minimum:
        raise Figure5ArtifactError(f"{label} must be >= {minimum}, observed={result}")
    return result


def _number(value: object, label: str, *, minimum: float | None = None) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise Figure5ArtifactError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise Figure5ArtifactError(f"{label} is not finite")
    if minimum is not None and result < minimum:
        raise Figure5ArtifactError(f"{label} must be >= {minimum}, observed={result}")
    return result


def _recall(value: object, label: str) -> float:
    result = _number(value, label, minimum=0.0)
    if result > 1.0:
        raise Figure5ArtifactError(f"{label} must be <= 1, observed={result}")
    return result


def _blank(value: object) -> bool:
    return str(value or "").strip() == ""


def _optional_text(raw: Mapping[str, str], field: str, label: str) -> str | None:
    value = raw.get(field)
    if value is None or _blank(value):
        return None
    return _text(value, f"{label} {field}")


def _method_token(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("-", "_")


def _is_forbidden_method_value(value: object) -> bool:
    token = _method_token(value)
    return token in FORBIDDEN_METHOD_VALUES or token in {
        "d1_d2",
        "design1+design2",
        "design1_design2",
    }


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return format(value, ".17g")


def _format_int(value: int) -> str:
    return str(value)


def read_repeat_csv(path: Path, expected_kind: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read and validate one input CSV without applying cross-file gates."""
    if expected_kind not in EXPERIMENT_KINDS:
        raise Figure5ArtifactError(f"unknown expected experiment kind: {expected_kind}")
    if not path.is_file():
        raise Figure5ArtifactError(f"input CSV does not exist: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = tuple(reader.fieldnames or ())
            raw_rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise Figure5ArtifactError(f"cannot read input CSV {path}: {exc}") from exc
    if not fields:
        raise Figure5ArtifactError(f"input CSV has no header: {path}")
    if len(fields) != len(set(fields)):
        raise Figure5ArtifactError(f"input CSV has duplicate header fields: {path}")
    lowered_fields = tuple(field.strip().lower() for field in fields)
    for field in lowered_fields:
        if (
            any(fragment in field for fragment in FORBIDDEN_FIELD_FRAGMENTS)
            or ("qps" in field and field != "throughput_qps")
        ):
            raise Figure5ArtifactError(
                f"input CSV contains forbidden derived-throughput field {field!r}: {path}"
            )
    missing = sorted(REQUIRED_INPUT_FIELDS - set(fields))
    if missing:
        raise Figure5ArtifactError(f"input CSV is missing fields {missing}: {path}")
    if not raw_rows:
        raise Figure5ArtifactError(f"input CSV is empty: {path}")

    rows: list[dict[str, Any]] = []
    for row_number, raw in enumerate(raw_rows, start=2):
        if None in raw:
            raise Figure5ArtifactError(f"{path}:{row_number} is wider than its header")
        label = f"{path}:{row_number}"
        row = _normalize_row(raw, expected_kind, label)
        rows.append(row)
    return rows, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "experiment_kind": expected_kind,
        "rows": len(rows),
    }


def _normalize_row(raw: Mapping[str, str], expected_kind: str, label: str) -> dict[str, Any]:
    for field, value in raw.items():
        if field not in {"arm_id", "mode_id"} and _is_forbidden_method_value(value):
            raise Figure5ArtifactError(
                f"{label} field {field!r} identifies a forbidden partial SQLens method"
            )
    schema_version = _text(raw.get("schema_version"), f"{label} schema_version")
    if schema_version != SCHEMA_VERSION:
        raise Figure5ArtifactError(
            f"{label} schema_version must be {SCHEMA_VERSION}, observed={schema_version!r}"
        )
    kind = _text(raw.get("experiment_kind"), f"{label} experiment_kind")
    if kind != expected_kind:
        raise Figure5ArtifactError(
            f"{label} experiment_kind={kind!r} does not match {expected_kind!r} input"
        )
    dataset = _text(raw.get("dataset"), f"{label} dataset")
    if dataset not in EXPECTED_DATASETS:
        raise Figure5ArtifactError(f"{label} has unsupported dataset={dataset!r}")
    arm = _text(raw.get("arm_id"), f"{label} arm_id")
    mode = _text(raw.get("mode_id"), f"{label} mode_id")
    if _is_forbidden_method_value(arm) or _is_forbidden_method_value(mode):
        raise Figure5ArtifactError(f"{label} contains forbidden partial SQLens method")
    if arm not in ARM_MODES:
        raise Figure5ArtifactError(f"{label} has unsupported arm_id={arm!r}")
    if mode != ARM_MODES[arm]:
        raise Figure5ArtifactError(
            f"{label} arm_id={arm!r} requires mode_id={ARM_MODES[arm]!r}, "
            f"observed={mode!r}"
        )
    clients = _integer(raw.get("clients"), f"{label} clients", minimum=1)
    if kind == "latency" and clients != 1:
        raise Figure5ArtifactError(f"{label} latency measurement must use clients=1")

    requests = _integer(raw.get("requests"), f"{label} requests")
    unique_queries = _integer(raw.get("unique_queries"), f"{label} unique_queries")
    completed = _integer(raw.get("completed_queries"), f"{label} completed_queries")
    errors = _integer(raw.get("error_count"), f"{label} error_count")
    if (requests, unique_queries, completed) != (
        EXPECTED_REQUESTS,
        EXPECTED_REQUESTS,
        EXPECTED_REQUESTS,
    ):
        raise Figure5ArtifactError(
            f"{label} fails q10k gate: requests={requests}, "
            f"unique_queries={unique_queries}, completed_queries={completed}"
        )
    if errors != 0:
        raise Figure5ArtifactError(f"{label} has error_count={errors}, expected=0")
    status = _text(raw.get("status"), f"{label} status")
    if status != "valid":
        raise Figure5ArtifactError(f"{label} status must be 'valid', observed={status!r}")

    recall_mean = _recall(raw.get("recall_mean"), f"{label} recall_mean")
    recall_low = _recall(raw.get("recall_ci95_low"), f"{label} recall_ci95_low")
    recall_high = _recall(raw.get("recall_ci95_high"), f"{label} recall_ci95_high")
    if not recall_low <= recall_mean <= recall_high:
        raise Figure5ArtifactError(f"{label} recall mean is outside its CI")
    wall = _number(raw.get("wall_clock_seconds"), f"{label} wall_clock_seconds", minimum=0.0)
    if wall <= 0.0:
        raise Figure5ArtifactError(f"{label} wall_clock_seconds must be positive")
    latency_mean = _number(raw.get("latency_mean_ms"), f"{label} latency_mean_ms", minimum=0.0)
    latency_p95 = _number(raw.get("latency_p95_ms"), f"{label} latency_p95_ms", minimum=0.0)
    latency_p99 = _number(raw.get("latency_p99_ms"), f"{label} latency_p99_ms", minimum=0.0)
    if (
        latency_mean <= 0.0
        or latency_p95 <= 0.0
        or latency_p99 <= 0.0
        or latency_p99 < latency_p95
    ):
        raise Figure5ArtifactError(
            f"{label} requires positive latency metrics and "
            "latency_p95_ms <= latency_p99_ms"
        )

    throughput_qps: float | None = None
    throughput_source = str(raw.get("throughput_source") or "").strip()
    if kind == "latency":
        if not _blank(raw.get("throughput_qps")) or throughput_source:
            raise Figure5ArtifactError(
                f"{label} latency rows must not contain throughput values"
            )
    else:
        throughput_qps = _number(
            raw.get("throughput_qps"), f"{label} throughput_qps", minimum=0.0
        )
        if throughput_source != MEASURED_THROUGHPUT_SOURCE:
            raise Figure5ArtifactError(
                f"{label} throughput_source must be {MEASURED_THROUGHPUT_SOURCE!r}; "
                "derived QPS is forbidden"
            )
        recomputed = completed / wall
        tolerance = max(1e-9, recomputed * 1e-6)
        if abs(throughput_qps - recomputed) > tolerance:
            raise Figure5ArtifactError(
                f"{label} throughput_qps mismatch: input={throughput_qps}, "
                f"completed/wall={recomputed}"
            )
        throughput_qps = recomputed

    optional_provenance: dict[str, str | None] = {}
    for field in OPTIONAL_PROVENANCE_FIELDS:
        value = _optional_text(raw, field, label)
        if value is not None:
            optional_provenance[field] = value

    return {
        "schema_version": schema_version,
        "run_id": _text(raw.get("run_id"), f"{label} run_id"),
        "dataset": dataset,
        "experiment_kind": kind,
        "arm_id": arm,
        "mode_id": mode,
        "config_id": _text(raw.get("config_id"), f"{label} config_id"),
        "config_sha256": _sha(raw.get("config_sha256"), f"{label} config_sha256"),
        "release_identity_sha256": _sha(
            raw.get("release_identity_sha256"), f"{label} release_identity_sha256"
        ),
        "clients": clients,
        "repeat_id": _integer(raw.get("repeat_id"), f"{label} repeat_id"),
        "request_trace_sha256": _sha(
            raw.get("request_trace_sha256"), f"{label} request_trace_sha256"
        ),
        "requests": requests,
        "unique_queries": unique_queries,
        "completed_queries": completed,
        "error_count": errors,
        "wall_clock_seconds": wall,
        "recall_mean": recall_mean,
        "recall_ci95_low": recall_low,
        "recall_ci95_high": recall_high,
        "latency_mean_ms": latency_mean,
        "latency_p95_ms": latency_p95,
        "latency_p99_ms": latency_p99,
        "throughput_qps": throughput_qps,
        "throughput_ci95_low": None,
        "throughput_ci95_high": None,
        "throughput_source": throughput_source,
        "status": status,
        **optional_provenance,
    }


def _single(values: Iterable[Any], label: str) -> Any:
    unique = set(values)
    if len(unique) != 1:
        raise Figure5ArtifactError(f"{label} is inconsistent: {sorted(unique)!r}")
    return next(iter(unique))


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Figure5ArtifactError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Figure5ArtifactError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Figure5ArtifactError(f"{label} must contain a JSON object: {path}")
    return value


def _contract_value(
    payload: Mapping[str, Any],
    primary: str,
    aliases: Sequence[str],
    label: str,
) -> str:
    observed = []
    for field in (primary, *aliases):
        if field in payload and not _blank(payload.get(field)):
            observed.append((field, _text(payload[field], f"{label} {field}")))
    if not observed:
        raise Figure5ArtifactError(f"{label} has no {primary}")
    values = {value for _, value in observed}
    if len(values) != 1:
        raise Figure5ArtifactError(f"{label} has conflicting {primary} aliases")
    return observed[0][1]


def _load_release_contract(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload = _read_json_object(resolved, "release contract")
    observed_sha = sha256_file(resolved)
    contract_id = _contract_value(payload, "contract_id", (), "release contract")
    build_id = _contract_value(
        payload,
        "expected_sqlens_build_id",
        ("build_id",),
        "release contract",
    )
    vector_sha = _contract_value(
        payload,
        "expected_vector_so_sha256",
        ("vector_so_sha256",),
        "release contract",
    )
    vector_sha = _sha(vector_sha, "release contract vector.so SHA")
    return {
        "path": str(resolved),
        "sha256": observed_sha,
        "contract_id": contract_id,
        "build_id": build_id,
        "vector_so_sha256": vector_sha,
        "schema_version": payload.get("schema_version"),
    }


def _validate_contract_binding(
    binding: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    path = Path(_text(binding.get("path"), f"{label} path")).resolve()
    if str(path) != expected["path"]:
        raise Figure5ArtifactError(
            f"{label} path mismatch: expected={expected['path']}, observed={path}"
        )
    if not path.is_file():
        raise Figure5ArtifactError(f"{label} does not exist: {path}")
    bound_sha = _sha(binding.get("sha256"), f"{label} SHA")
    if bound_sha != expected["sha256"] or sha256_file(path) != bound_sha:
        raise Figure5ArtifactError(f"{label} SHA binding is invalid")
    contract_id = _text(binding.get("contract_id"), f"{label} contract_id")
    build_id = _contract_value(
        binding,
        "expected_sqlens_build_id",
        ("build_id",),
        label,
    )
    vector_sha = _sha(
        _contract_value(
            binding,
            "expected_vector_so_sha256",
            ("vector_so_sha256",),
            label,
        ),
        f"{label} vector.so SHA",
    )
    if (
        contract_id != expected["contract_id"]
        or build_id != expected["build_id"]
        or vector_sha != expected["vector_so_sha256"]
    ):
        raise Figure5ArtifactError(f"{label} does not match the release contract")


def _validate_audited_source_manifest(
    path: Path,
    expected_kind: str,
    input_path: Path,
    release: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a converter binding and the run manifest it audits.

    A raw repeat CSV is not provenance.  The binding must name the exact CSV,
    its SHA, the audited run manifest, and the same immutable release contract.
    This accepts either a converter-binding sidecar (preferred) or a run
    manifest with an explicit ``converter_binding``/``outputs`` entry.
    """
    binding_path = path.resolve()
    payload = _read_json_object(binding_path, f"{expected_kind} source manifest")
    for field in ("artifact_valid", "full_release_complete", "paper_eligible"):
        if payload.get(field) is not True:
            raise Figure5ArtifactError(
                f"{expected_kind} source manifest is not paper-eligible: {field}"
            )
    if payload.get("status") not in {"complete", "valid"}:
        raise Figure5ArtifactError(
            f"{expected_kind} source manifest status is not complete"
        )
    contract_binding = payload.get("release_contract")
    if not isinstance(contract_binding, Mapping):
        raise Figure5ArtifactError(
            f"{expected_kind} source manifest has no release contract binding"
        )
    _validate_contract_binding(contract_binding, release, f"{expected_kind} release contract")

    converter = payload.get("converter_binding")
    if converter is None:
        converter = payload
    if not isinstance(converter, Mapping):
        raise Figure5ArtifactError(
            f"{expected_kind} source manifest has no converter binding"
        )
    output = converter.get("output")
    if output is None:
        outputs = converter.get("outputs")
        if isinstance(outputs, Mapping):
            for candidate in (
                "repeat_csv",
                "converted_repeat_csv",
                "measurement_repeats",
                "repeats",
            ):
                if isinstance(outputs.get(candidate), Mapping):
                    output = outputs[candidate]
                    break
    if not isinstance(output, Mapping):
        raise Figure5ArtifactError(
            f"{expected_kind} source manifest has no converter output binding"
        )
    output_path = Path(_text(output.get("path"), f"{expected_kind} output path")).resolve()
    output_sha = _sha(output.get("sha256"), f"{expected_kind} output SHA")
    if output_path != input_path.resolve() or sha256_file(output_path) != output_sha:
        raise Figure5ArtifactError(
            f"{expected_kind} input CSV is not the SHA-bound converter output"
        )
    output_kind = _text(
        output.get("experiment_kind") or output.get("kind"),
        f"{expected_kind} output experiment_kind",
    )
    if output_kind != expected_kind:
        raise Figure5ArtifactError(
            f"converter output kind={output_kind!r} does not match {expected_kind!r}"
        )

    source_ref = converter.get("source_manifest")
    if source_ref is None:
        source_ref = converter.get("run_manifest")
    if source_ref is None:
        raise Figure5ArtifactError(
            f"{expected_kind} converter binding has no audited run manifest"
        )
    if not isinstance(source_ref, Mapping):
        raise Figure5ArtifactError(
            f"{expected_kind} source manifest has malformed source_manifest binding"
        )
    source_path = Path(
        _text(source_ref.get("path"), f"{expected_kind} source manifest path")
    ).resolve()
    source_sha = _sha(
        source_ref.get("sha256"), f"{expected_kind} source manifest SHA"
    )
    if not source_path.is_file() or sha256_file(source_path) != source_sha:
        raise Figure5ArtifactError(
            f"{expected_kind} source manifest path/SHA binding is invalid"
        )
    audited = _read_json_object(source_path, f"{expected_kind} audited run manifest")
    for field in ("artifact_valid", "full_release_complete", "paper_eligible"):
        if audited.get(field) is not True:
            raise Figure5ArtifactError(
                f"{expected_kind} audited run manifest is not paper-eligible: {field}"
            )
    if audited.get("status") not in {"complete", "valid"}:
        raise Figure5ArtifactError(
            f"{expected_kind} audited run manifest status is not complete"
        )
    audited_release = audited.get("release_contract")
    if not isinstance(audited_release, Mapping):
        raise Figure5ArtifactError(
            f"{expected_kind} audited run manifest has no release contract"
        )
    _validate_contract_binding(
        audited_release, release, f"{expected_kind} audited release contract"
    )
    return {
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": source_sha,
        "output_path": str(output_path),
        "output_sha256": output_sha,
        "experiment_kind": expected_kind,
    }


def _validate_row_release_fields(
    rows: Sequence[Mapping[str, Any]],
    release: Mapping[str, Any],
    label: str,
    source_binding: Mapping[str, Any] | None = None,
) -> None:
    for index, row in enumerate(rows):
        row_label = f"{label} row {index + 1}"
        optional = {
            field: row.get(field)
            for field in OPTIONAL_PROVENANCE_FIELDS
            if row.get(field) is not None
        }
        if not optional:
            continue
        expected = {
            "release_contract_path": release["path"],
            "release_contract_sha256": release["sha256"],
            "release_contract_id": release["contract_id"],
            "release_build_id": release["build_id"],
            "release_vector_so_sha256": release["vector_so_sha256"],
        }
        if source_binding is not None:
            expected.update(
                {
                    "source_manifest_path": source_binding["source_manifest_path"],
                    "source_manifest_sha256": source_binding[
                        "source_manifest_sha256"
                    ],
                }
            )
        for field, value in optional.items():
            if field in expected:
                observed = str(value).strip()
                if field.endswith("sha256") or field == "release_vector_so_sha256":
                    observed = _sha(observed, f"{row_label} {field}")
                if observed != expected[field]:
                    raise Figure5ArtifactError(
                        f"{row_label} {field} disagrees with release contract"
                    )


def audit_rows(
    rows: Sequence[dict[str, Any]], release: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply artifact-wide release, pairing, trace, config, and repeat gates."""
    if not rows:
        raise Figure5ArtifactError("no repeat rows were supplied")
    release_sha = _single(
        (row["release_identity_sha256"] for row in rows),
        "release_identity_sha256",
    )
    if release_sha != release["sha256"]:
        raise Figure5ArtifactError(
            "repeat rows carry a release_identity_sha256 that does not match "
            "the supplied release contract"
        )
    observed_datasets = set(row["dataset"] for row in rows)
    if observed_datasets != set(EXPECTED_DATASETS):
        raise Figure5ArtifactError(
            "three-dataset release gate failed: "
            f"expected={list(EXPECTED_DATASETS)!r}, observed={sorted(observed_datasets)!r}"
        )

    for dataset in EXPECTED_DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        _single(
            (row["request_trace_sha256"] for row in dataset_rows),
            f"{dataset} request_trace_sha256",
        )
        observed_kinds = set(row["experiment_kind"] for row in dataset_rows)
        if observed_kinds != set(EXPERIMENT_KINDS):
            raise Figure5ArtifactError(
                f"{dataset} must contain latency and throughput, observed={sorted(observed_kinds)!r}"
            )

    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["experiment_kind"],
            row["arm_id"],
            row["mode_id"],
            row["config_id"],
            row["clients"],
        )
        cells[key].append(row)

    for key, cell_rows in cells.items():
        dataset, kind, arm, mode, config_id, clients = key
        required_repeats = MIN_REPEATS[kind]
        if len(cell_rows) < required_repeats:
            raise Figure5ArtifactError(
                f"{key!r} requires at least {required_repeats} repeats, "
                f"observed={len(cell_rows)}"
            )
        repeat_ids = [row["repeat_id"] for row in cell_rows]
        if len(repeat_ids) != len(set(repeat_ids)):
            raise Figure5ArtifactError(f"{key!r} contains duplicate repeat_id")
        if sorted(repeat_ids) != list(range(len(cell_rows))):
            raise Figure5ArtifactError(
                f"{key!r} repeat_id values must be contiguous from zero, "
                f"observed={sorted(repeat_ids)!r}"
            )
        _single((row["run_id"] for row in cell_rows), f"{key!r} run_id")
        _single((row["config_sha256"] for row in cell_rows), f"{key!r} config_sha256")
        _single(
            (row["request_trace_sha256"] for row in cell_rows),
            f"{key!r} request_trace_sha256",
        )
        if mode != ARM_MODES[arm]:
            raise Figure5ArtifactError(f"{key!r} violates arm/mode identity")

    config_bindings: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        binding_key = (
            row["dataset"],
            row["experiment_kind"],
            row["arm_id"],
            row["mode_id"],
            row["config_id"],
        )
        config_bindings[binding_key].add(row["config_sha256"])
    for key, hashes in config_bindings.items():
        if len(hashes) != 1:
            raise Figure5ArtifactError(
                f"{key!r} maps one config_id to multiple config_sha256 values: "
                f"{sorted(hashes)!r}"
            )

    # Latency and throughput must traverse the same configuration points.  A
    # figure that silently compares different search grids is not a matched
    # frontier, even when every individual cell is internally valid.
    for dataset in EXPECTED_DATASETS:
        for arm in ARM_MODES:
            grids = {}
            for kind in EXPERIMENT_KINDS:
                grids[kind] = {
                    (row["config_id"], row["config_sha256"])
                    for row in rows
                    if row["dataset"] == dataset
                    and row["arm_id"] == arm
                    and row["experiment_kind"] == kind
                }
            if grids["latency"] != grids["throughput"]:
                raise Figure5ArtifactError(
                    f"{dataset}/{arm} latency and throughput config sets differ: "
                    f"latency_only={sorted(grids['latency'] - grids['throughput'])!r}, "
                    f"throughput_only={sorted(grids['throughput'] - grids['latency'])!r}"
                )

    paired: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_key = (
            row["dataset"],
            row["experiment_kind"],
            row["config_id"],
            row["clients"],
            row["repeat_id"],
        )
        paired[pair_key].append(row)
    for key, pair_rows in paired.items():
        arms = [row["arm_id"] for row in pair_rows]
        if sorted(arms) != sorted(ARM_MODES):
            raise Figure5ArtifactError(
                f"{key!r} must contain exactly one stock and one full SQLens row, "
                f"observed={arms!r}"
            )
        _single((row["run_id"] for row in pair_rows), f"{key!r} paired run_id")
        _single(
            (row["request_trace_sha256"] for row in pair_rows),
            f"{key!r} paired request trace",
        )
        _single((row["requests"] for row in pair_rows), f"{key!r} paired requests")
        _single(
            (row["unique_queries"] for row in pair_rows),
            f"{key!r} paired unique_queries",
        )

    return {
        "release_identity_sha256": release_sha,
        "release_contract": dict(release),
        "config_sets_match": True,
        "datasets": {
            dataset: {
                "request_trace_sha256": _single(
                    (
                        row["request_trace_sha256"]
                        for row in rows
                        if row["dataset"] == dataset
                    ),
                    f"{dataset} request trace",
                ),
                "repeat_rows": sum(row["dataset"] == dataset for row in rows),
            }
            for dataset in EXPECTED_DATASETS
        },
        "repeat_rows": len(rows),
        "cells": len(cells),
        "paired_repeat_cells": len(paired),
    }


def _t_critical_95(sample_size: int) -> float:
    values = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    return values.get(sample_size, 1.96)


def _mean_ci(
    values: Sequence[float],
    *,
    center: float | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float, float]:
    if not values:
        raise Figure5ArtifactError("cannot aggregate an empty sample")
    observed_center = statistics.fmean(values) if center is None else center
    if len(values) == 1:
        low = high = observed_center
    else:
        half_width = (
            _t_critical_95(len(values))
            * statistics.stdev(values)
            / math.sqrt(len(values))
        )
        low, high = observed_center - half_width, observed_center + half_width
    if lower_bound is not None:
        low = max(lower_bound, low)
    if upper_bound is not None:
        high = min(upper_bound, high)
    return observed_center, low, high


def aggregate_points(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["experiment_kind"],
            row["arm_id"],
            row["mode_id"],
            row["config_id"],
            row["clients"],
        )
        groups[key].append(row)

    points: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        dataset, kind, arm, mode, config_id, clients = key
        group_rows = sorted(group_rows, key=lambda row: row["repeat_id"])
        recall, recall_low, recall_high = _mean_ci(
            [row["recall_mean"] for row in group_rows],
            lower_bound=0.0,
            upper_bound=1.0,
        )
        latency: float | None = None
        latency_low: float | None = None
        latency_high: float | None = None
        throughput: float | None = None
        throughput_low: float | None = None
        throughput_high: float | None = None
        if kind == "latency":
            latency, latency_low, latency_high = _mean_ci(
                [row["latency_mean_ms"] for row in group_rows],
                lower_bound=0.0,
            )
        else:
            total_completed = sum(row["completed_queries"] for row in group_rows)
            total_wall = sum(row["wall_clock_seconds"] for row in group_rows)
            throughput = total_completed / total_wall
            _, throughput_low, throughput_high = _mean_ci(
                [row["throughput_qps"] for row in group_rows],
                center=throughput,
                lower_bound=0.0,
            )
        point_identity = "|".join(
            (dataset, kind, arm, mode, config_id, str(clients))
        )
        points.append(
            {
                "schema_version": SCHEMA_VERSION,
                "point_id": hashlib.sha256(point_identity.encode("utf-8")).hexdigest()[:20],
                "run_id": _single(
                    (row["run_id"] for row in group_rows), f"{key!r} run_id"
                ),
                "dataset": dataset,
                "experiment_kind": kind,
                "arm_id": arm,
                "mode_id": mode,
                "config_id": config_id,
                "config_sha256": _single(
                    (row["config_sha256"] for row in group_rows),
                    f"{key!r} config_sha256",
                ),
                "release_identity_sha256": _single(
                    (row["release_identity_sha256"] for row in group_rows),
                    f"{key!r} release_identity_sha256",
                ),
                "clients": clients,
                "repeats": len(group_rows),
                "requests_per_repeat": EXPECTED_REQUESTS,
                "recall_mean": recall,
                "recall_ci95_low": recall_low,
                "recall_ci95_high": recall_high,
                "latency_mean_ms": latency,
                "latency_ci95_low_ms": latency_low,
                "latency_ci95_high_ms": latency_high,
                "throughput_qps": throughput,
                "throughput_ci95_low": throughput_low,
                "throughput_ci95_high": throughput_high,
                "pareto": False,
                "is_plot_eligible": True,
            }
        )
    return mark_pareto(points)


def _same_coordinate(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if abs(left["recall_mean"] - right["recall_mean"]) > PARETO_TOLERANCE:
        return False
    kind = left["experiment_kind"]
    metric = "latency_mean_ms" if kind == "latency" else "throughput_qps"
    return abs(left[metric] - right[metric]) <= PARETO_TOLERANCE


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    recall_better = left["recall_mean"] >= right["recall_mean"] - PARETO_TOLERANCE
    recall_strict = left["recall_mean"] > right["recall_mean"] + PARETO_TOLERANCE
    if left["experiment_kind"] == "latency":
        metric_better = (
            left["latency_mean_ms"]
            <= right["latency_mean_ms"] + PARETO_TOLERANCE
        )
        metric_strict = (
            left["latency_mean_ms"]
            < right["latency_mean_ms"] - PARETO_TOLERANCE
        )
    else:
        metric_better = (
            left["throughput_qps"]
            >= right["throughput_qps"] - PARETO_TOLERANCE
        )
        metric_strict = (
            left["throughput_qps"]
            > right["throughput_qps"] + PARETO_TOLERANCE
        )
    return recall_better and metric_better and (recall_strict or metric_strict)


def mark_pareto(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark Pareto points independently within dataset/kind/arm/mode."""
    result = [dict(point, pareto=False) for point in points]
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, point in enumerate(result):
        key = (
            point["dataset"],
            point["experiment_kind"],
            point["arm_id"],
            point["mode_id"],
        )
        groups[key].append(index)

    for indices in groups.values():
        ordered = sorted(
            indices,
            key=lambda index: (
                str(result[index]["config_id"]),
                int(result[index]["clients"]),
                str(result[index]["point_id"]),
            ),
        )
        representatives: list[int] = []
        for index in ordered:
            if not any(
                _same_coordinate(result[index], result[existing])
                for existing in representatives
            ):
                representatives.append(index)
        for index in representatives:
            if not any(
                other != index and _dominates(result[other], result[index])
                for other in representatives
            ):
                result[index]["pareto"] = True
    return result


def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(
        target,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        serialized: dict[str, Any] = {}
        for field in fields:
            value = row.get(field)
            if isinstance(value, bool):
                serialized[field] = "true" if value else "false"
            elif isinstance(value, float):
                serialized[field] = _format_float(value)
            elif value is None:
                serialized[field] = ""
            else:
                serialized[field] = value
        writer.writerow(serialized)
    return target.getvalue().encode("utf-8")


def _repeat_output_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            EXPECTED_DATASETS.index(row["dataset"]),
            EXPERIMENT_KINDS.index(row["experiment_kind"]),
            row["config_id"],
            row["clients"],
            row["repeat_id"],
            row["arm_id"],
        ),
    )
    result: list[dict[str, Any]] = []
    for row in ordered:
        output = dict(row)
        if row["experiment_kind"] == "throughput":
            output["throughput_ci95_low"] = ""
            output["throughput_ci95_high"] = ""
        result.append(output)
    return result


def _point_output_rows(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        points,
        key=lambda row: (
            EXPECTED_DATASETS.index(row["dataset"]),
            EXPERIMENT_KINDS.index(row["experiment_kind"]),
            row["arm_id"],
            row["clients"],
            row["config_id"],
        ),
    )


def build_artifact(
    latency_inputs: Sequence[Path],
    throughput_inputs: Sequence[Path],
    *,
    release_contract_path: Path,
    latency_run_manifest: Path,
    throughput_run_manifest: Path,
) -> tuple[bytes, bytes, dict[str, Any]]:
    if not latency_inputs:
        raise Figure5ArtifactError("at least one --latency-input is required")
    if not throughput_inputs:
        raise Figure5ArtifactError("at least one --throughput-input is required")
    release = _load_release_contract(release_contract_path)
    if len(latency_inputs) != 1 or len(throughput_inputs) != 1:
        raise Figure5ArtifactError(
            "formal Figure 5 finalization requires exactly one converter CSV "
            "for each experiment kind"
        )
    rows: list[dict[str, Any]] = []
    input_identity: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for kind, paths in (
        ("latency", latency_inputs),
        ("throughput", throughput_inputs),
    ):
        for path in paths:
            resolved = path.resolve()
            if resolved in seen_paths:
                raise Figure5ArtifactError(f"input CSV supplied more than once: {resolved}")
            seen_paths.add(resolved)
            input_rows, identity = read_repeat_csv(path, kind)
            rows.extend(input_rows)
            input_identity.append(identity)

    source_bindings = {
        "latency": _validate_audited_source_manifest(
            latency_run_manifest,
            "latency",
            latency_inputs[0],
            release,
        ),
        "throughput": _validate_audited_source_manifest(
            throughput_run_manifest,
            "throughput",
            throughput_inputs[0],
            release,
        ),
    }
    _validate_row_release_fields(
        [row for row in rows if row["experiment_kind"] == "latency"],
        release,
        "latency",
        source_bindings["latency"],
    )
    _validate_row_release_fields(
        [row for row in rows if row["experiment_kind"] == "throughput"],
        release,
        "throughput",
        source_bindings["throughput"],
    )
    audit = audit_rows(rows, release)
    points = aggregate_points(rows)
    repeat_bytes = _csv_bytes(REPEAT_FIELDS, _repeat_output_rows(rows))
    point_bytes = _csv_bytes(POINT_FIELDS, _point_output_rows(points))
    tool_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "artifact_type": "sqlens_figure5_frontier",
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "artifact_valid": True,
        "paper_eligible": True,
        "release_identity": {
            "sha256": audit["release_identity_sha256"],
        },
        "release_contract": dict(release),
        "source_bindings": source_bindings,
        "datasets": audit["datasets"],
        "methods": {
            arm: {
                "arm_id": arm,
                "mode_id": mode,
            }
            for arm, mode in ARM_MODES.items()
        },
        "search_grid": {
            "config_ids": sorted(set(row["config_id"] for row in rows)),
            "config_sha256": {
                "|".join(
                    (
                        row["dataset"],
                        row["experiment_kind"],
                        row["arm_id"],
                        row["config_id"],
                    )
                ): row["config_sha256"]
                for row in rows
            },
        },
        "protocol": {
            "requests_per_repeat": EXPECTED_REQUESTS,
            "unique_queries_per_repeat": EXPECTED_REQUESTS,
            "minimum_repeats": MIN_REPEATS,
            "latency_clients": 1,
            "throughput_source": MEASURED_THROUGHPUT_SOURCE,
            "throughput_aggregation": "sum_completed_queries_over_sum_barrier_wall_clock",
            "pareto_tolerance": PARETO_TOLERANCE,
            "pareto_scope": "dataset_experiment_kind_arm_mode",
        },
        "inputs": sorted(
            input_identity,
            key=lambda item: (item["experiment_kind"], item["path"]),
        ),
        "outputs": {
            "repeats": {
                "path": "figure5_measurement_repeats.csv",
                "rows": len(rows),
                "sha256": sha256_bytes(repeat_bytes),
            },
            "points": {
                "path": "figure5_points.csv",
                "rows": len(points),
                "pareto_rows": sum(bool(point["pareto"]) for point in points),
                "sha256": sha256_bytes(point_bytes),
            },
        },
        "gates": {
            "three_datasets": True,
            "q10k_complete": True,
            "paired_trace": True,
            "release_identity": True,
            "release_contract": True,
            "audited_source_manifests": True,
            "latency_throughput_config_sets_match": audit["config_sets_match"],
            "config_identity": True,
            "full_sqlens_only": True,
            "minimum_repeats": True,
            "measured_throughput_only": True,
            "throughput_recomputed": True,
            "pareto_scoped_per_arm": True,
        },
        "audit": {
            "tool": str(tool_path),
            "tool_sha256": sha256_file(tool_path),
            "repeat_rows": audit["repeat_rows"],
            "cells": audit["cells"],
            "paired_repeat_cells": audit["paired_repeat_cells"],
        },
    }
    return repeat_bytes, point_bytes, manifest


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_artifact(
    output_dir: Path,
    repeat_bytes: bytes,
    point_bytes: bytes,
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    """Transactionally publish data files followed by the manifest commit marker."""
    expected_repeat_sha = str(
        manifest.get("outputs", {}).get("repeats", {}).get("sha256", "")
    )
    expected_point_sha = str(
        manifest.get("outputs", {}).get("points", {}).get("sha256", "")
    )
    if sha256_bytes(repeat_bytes) != expected_repeat_sha:
        raise Figure5ArtifactError("repeat CSV payload does not match manifest SHA-256")
    if sha256_bytes(point_bytes) != expected_point_sha:
        raise Figure5ArtifactError("points CSV payload does not match manifest SHA-256")
    if manifest.get("artifact_valid") is not True or manifest.get("paper_eligible") is not True:
        raise Figure5ArtifactError("cannot publish an invalid or paper-ineligible manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "repeats": output_dir / "figure5_measurement_repeats.csv",
        "points": output_dir / "figure5_points.csv",
        "manifest": output_dir / "figure5_manifest.json",
    }
    payloads = {
        "repeats": repeat_bytes,
        "points": point_bytes,
        "manifest": _json_bytes(manifest),
    }
    stage_dir = Path(tempfile.mkdtemp(prefix=".figure5-stage-", dir=output_dir))
    backup_dir = stage_dir / "backups"
    backup_dir.mkdir()
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    installed: list[str] = []
    try:
        for name, payload in payloads.items():
            path = stage_dir / destinations[name].name
            with path.open("wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            staged[name] = path
        _fsync_directory(stage_dir)
        for name in ("repeats", "points", "manifest"):
            destination = destinations[name]
            if destination.exists():
                backup = backup_dir / destination.name
                os.replace(destination, backup)
                backups[name] = backup
            os.replace(staged[name], destination)
            installed.append(name)
            _fsync_directory(output_dir)
    except Exception:
        for name in reversed(installed):
            destination = destinations[name]
            if destination.exists():
                destination.unlink()
            if name in backups and backups[name].exists():
                os.replace(backups[name], destination)
        for name, backup in backups.items():
            if name not in installed and backup.exists():
                os.replace(backup, destinations[name])
        _fsync_directory(output_dir)
        raise
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    return destinations


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latency-input",
        action="append",
        default=[],
        type=Path,
        help="Repeat-level latency summary CSV; may be supplied multiple times.",
    )
    parser.add_argument(
        "--throughput-input",
        action="append",
        default=[],
        type=Path,
        help="Repeat-level throughput summary CSV; may be supplied multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for figure5_measurement_repeats.csv, points CSV, and manifest.",
    )
    parser.add_argument(
        "--release-contract",
        type=Path,
        required=True,
        help="Immutable release contract JSON; path and SHA are bound into the artifact.",
    )
    parser.add_argument(
        "--latency-run-manifest",
        type=Path,
        required=True,
        help="Audited latency converter/run manifest binding the latency CSV.",
    )
    parser.add_argument(
        "--throughput-run-manifest",
        type=Path,
        required=True,
        help="Audited throughput converter/run manifest binding the throughput CSV.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and aggregate inputs, print the prospective manifest, write nothing.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Audit, aggregate, and atomically publish the formal artifact.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    try:
        repeat_bytes, point_bytes, manifest = build_artifact(
            args.latency_input,
            args.throughput_input,
            release_contract_path=args.release_contract,
            latency_run_manifest=args.latency_run_manifest,
            throughput_run_manifest=args.throughput_run_manifest,
        )
        if args.dry_run:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            destinations = publish_artifact(
                args.output_dir,
                repeat_bytes,
                point_bytes,
                manifest,
            )
            print(
                json.dumps(
                    {name: str(path) for name, path in destinations.items()},
                    indent=2,
                    sort_keys=True,
                )
            )
    except Figure5ArtifactError as exc:
        print(f"figure5 artifact rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
