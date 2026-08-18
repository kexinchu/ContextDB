#!/usr/bin/env python3
"""Build deterministic, audited Figure 5 calibration and measurement traces."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import sys
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
EXPECTED_FILTERS = 14
DEFAULT_SEED = 20260728
LEGACY_CALIBRATION_PROTOCOL = "legacy_balanced_v1"
FORMAL_CALIBRATION_PROTOCOL = "formal_per_predicate_cartesian_v1"
FORMAL_CALIBRATION_QUERY_COUNT = 200
FORMAL_CALIBRATION_REQUESTS = FORMAL_CALIBRATION_QUERY_COUNT * EXPECTED_FILTERS
CALIBRATION_PROTOCOLS = {
    LEGACY_CALIBRATION_PROTOCOL,
    FORMAL_CALIBRATION_PROTOCOL,
}
OUTPUT_FIELDS = (
    "request_no",
    "query_no",
    "query_id",
    "filter_name",
    "trace_cycle",
    "split",
)
TIE_AWARE_FIELDS = {
    "exact_filtered_topk_ids",
    "exact_filtered_topk_distances_sq",
    "kth_distance_sq",
    "tie_tolerance",
    "strict_closer_count",
    "boundary_tied",
}


class WorkloadError(RuntimeError):
    """An input or output failed the Figure 5 workload contract."""


@dataclass(frozen=True)
class Query:
    query_no: int
    query_id: str


@dataclass(frozen=True)
class Filter:
    name: str
    predicate: str
    atoms: str
    selectivity_field: str
    selectivity_raw: str
    selectivity_pct: float


@dataclass(frozen=True)
class BuildConfig:
    query_cohort_csv: Path
    filters_csv: Path
    truth_csv: Path | None
    out_prefix: Path
    query_no_column: str = "query_no"
    query_id_column: str = "query_id"
    truth_query_no_column: str = "query_no"
    truth_query_id_column: str = "query_id"
    truth_filter_column: str = "filter_name"
    seed: int = DEFAULT_SEED
    calibration_query_start: int = 0
    calibration_query_count: int = 200
    calibration_requests: int = 2_000
    calibration_protocol: str = LEGACY_CALIBRATION_PROTOCOL
    require_formal_paper_calibration: bool = False
    measurement_query_start: int = 200
    measurement_query_count: int = 10_000
    k: int = 10
    truth_coverage: str = "full"
    trace_only: bool = False

    def validate(self) -> None:
        integer_values = {
            "calibration_query_start": self.calibration_query_start,
            "calibration_query_count": self.calibration_query_count,
            "calibration_requests": self.calibration_requests,
            "measurement_query_start": self.measurement_query_start,
            "measurement_query_count": self.measurement_query_count,
            "k": self.k,
        }
        for name, value in integer_values.items():
            if value < 0:
                raise WorkloadError(f"{name} must be non-negative")
        if self.calibration_query_count == 0:
            raise WorkloadError("calibration_query_count must be positive")
        if self.measurement_query_count == 0:
            raise WorkloadError("measurement_query_count must be positive")
        if self.calibration_requests < EXPECTED_FILTERS:
            raise WorkloadError(
                f"calibration_requests must be at least {EXPECTED_FILTERS}"
            )
        if self.calibration_protocol not in CALIBRATION_PROTOCOLS:
            raise WorkloadError(
                "calibration_protocol must be one of "
                f"{sorted(CALIBRATION_PROTOCOLS)}, observed={self.calibration_protocol!r}"
            )
        if self.calibration_protocol == FORMAL_CALIBRATION_PROTOCOL:
            expected = self.calibration_query_count * EXPECTED_FILTERS
            if self.calibration_requests != expected:
                raise WorkloadError(
                    "formal_per_predicate_cartesian_v1 requires one request for every "
                    "calibration query/filter pair: "
                    f"expected={expected}, observed={self.calibration_requests}"
                )
        if self.require_formal_paper_calibration:
            if self.calibration_protocol != FORMAL_CALIBRATION_PROTOCOL:
                raise WorkloadError(
                    "--require-formal-paper-calibration requires "
                    "--calibration-protocol formal_per_predicate_cartesian_v1"
                )
            if self.calibration_query_count != FORMAL_CALIBRATION_QUERY_COUNT:
                raise WorkloadError(
                    "formal paper calibration requires exactly "
                    f"{FORMAL_CALIBRATION_QUERY_COUNT} calibration queries, "
                    f"observed={self.calibration_query_count}"
                )
            if self.calibration_requests != FORMAL_CALIBRATION_REQUESTS:
                raise WorkloadError(
                    "formal paper calibration requires exactly "
                    f"{FORMAL_CALIBRATION_REQUESTS} requests, "
                    f"observed={self.calibration_requests}"
                )
        if self.measurement_query_count < EXPECTED_FILTERS:
            raise WorkloadError(
                f"measurement_query_count must be at least {EXPECTED_FILTERS}"
            )
        if self.k <= 0:
            raise WorkloadError("k must be positive")
        if self.truth_coverage not in {"full", "assigned"}:
            raise WorkloadError(
                f"truth_coverage must be full or assigned, observed={self.truth_coverage!r}"
            )
        if not self.trace_only and self.truth_csv is None:
            raise WorkloadError("truth_csv is required unless trace_only is enabled")
        calibration = set(
            range(
                self.calibration_query_start,
                self.calibration_query_start + self.calibration_query_count,
            )
        )
        measurement = set(
            range(
                self.measurement_query_start,
                self.measurement_query_start + self.measurement_query_count,
            )
        )
        if calibration & measurement:
            raise WorkloadError("calibration and measurement query ranges overlap")
        columns = {
            self.query_no_column,
            self.query_id_column,
            self.truth_query_no_column,
            self.truth_query_id_column,
            self.truth_filter_column,
        }
        if any(not value.strip() for value in columns):
            raise WorkloadError("configured CSV column names must not be empty")

    @property
    def calibration_query_nos(self) -> range:
        return range(
            self.calibration_query_start,
            self.calibration_query_start + self.calibration_query_count,
        )

    @property
    def measurement_query_nos(self) -> range:
        return range(
            self.measurement_query_start,
            self.measurement_query_start + self.measurement_query_count,
        )

    @property
    def output_paths(self) -> dict[str, Path]:
        parent = self.out_prefix.parent
        stem = self.out_prefix.name
        return {
            "calibration_workload_csv": parent / f"{stem}_calibration.csv",
            "measurement_workload_csv": parent / f"{stem}_measurement.csv",
            "assigned_workload_csv": parent / f"{stem}_assigned.csv",
            "manifest_json": parent / f"{stem}_manifest.json",
        }

    @property
    def journal_path(self) -> Path:
        return self.out_prefix.parent / f".{self.out_prefix.name}.publish.journal.json"

    @property
    def lock_path(self) -> Path:
        return self.out_prefix.parent / f".{self.out_prefix.name}.publish.lock"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_ascii(value: str, label: str) -> str:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise WorkloadError(f"{label} must contain only ASCII characters") from exc
    return value


def _parse_int(value: object, label: str, *, minimum: int | None = None) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise WorkloadError(f"{label} is not an integer: {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise WorkloadError(f"{label} must be >= {minimum}, observed={parsed}")
    return parsed


def _parse_float(
    value: object, label: str, *, minimum: float | None = None
) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise WorkloadError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise WorkloadError(f"{label} must be finite")
    if minimum is not None and parsed < minimum:
        raise WorkloadError(f"{label} must be >= {minimum}, observed={parsed}")
    return parsed


def _parse_bool(value: object, label: str) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise WorkloadError(f"{label} is not a boolean: {value!r}")


def _normalize_sql(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _parse_csv_ints(value: object, label: str) -> list[int]:
    text = str(value or "").strip()
    if not text:
        raise WorkloadError(f"{label} is empty")
    return [_parse_int(part, label) for part in text.split(",")]


def _parse_csv_floats(value: object, label: str) -> list[float]:
    text = str(value or "").strip()
    if not text:
        raise WorkloadError(f"{label} is empty")
    return [_parse_float(part, label, minimum=0.0) for part in text.split(",")]


def _read_csv_bytes(path: Path, label: str) -> tuple[bytes, list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise WorkloadError(f"{label} does not exist: {path}")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorkloadError(f"{label} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = list(reader.fieldnames or ())
    if not fields:
        raise WorkloadError(f"{label} has no header")
    if len(fields) != len(set(fields)):
        raise WorkloadError(f"{label} contains duplicate header fields")
    rows = list(reader)
    if any(None in row for row in rows):
        raise WorkloadError(f"{label} contains a row wider than its header")
    return payload, fields, rows


def _require_fields(fields: Sequence[str], required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(fields))
    if missing:
        raise WorkloadError(f"{label} is missing fields: {missing}")


def load_queries(
    payload: bytes,
    *,
    query_no_column: str,
    query_id_column: str,
) -> tuple[dict[int, Query], int]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    fields = list(reader.fieldnames or ())
    _require_fields(fields, {query_no_column, query_id_column}, "query cohort CSV")
    queries: dict[int, Query] = {}
    id_to_no: dict[str, int] = {}
    row_count = 0
    for row_no, row in enumerate(reader, start=2):
        row_count += 1
        if None in row:
            raise WorkloadError("query cohort CSV contains a row wider than its header")
        query_no = _parse_int(
            row.get(query_no_column), f"query cohort row {row_no} query_no", minimum=0
        )
        query_id = _require_ascii(
            str(row.get(query_id_column) or "").strip(),
            f"query cohort row {row_no} query_id",
        )
        if not query_id:
            raise WorkloadError(f"query cohort row {row_no} has an empty query_id")
        if query_no in queries:
            raise WorkloadError(
                f"query cohort CSV contains duplicate query_no={query_no}"
            )
        if query_id in id_to_no:
            raise WorkloadError(
                f"query cohort CSV maps query_id={query_id!r} to multiple query numbers"
            )
        queries[query_no] = Query(query_no, query_id)
        id_to_no[query_id] = query_no
    return queries, row_count


def _selectivity_pct(field: str, raw: str, label: str) -> float:
    value = raw.strip()
    if not value:
        raise WorkloadError(f"{label} is empty")
    if value.endswith("%"):
        parsed = _parse_float(value[:-1], label, minimum=0.0)
    else:
        parsed = _parse_float(value, label, minimum=0.0)
        if field == "target_rate" and parsed <= 1.0:
            parsed *= 100.0
    if not 0.0 < parsed <= 100.0:
        raise WorkloadError(f"{label} must represent a percentage in (0, 100]")
    return parsed


def load_filters(
    payload: bytes,
) -> tuple[list[Filter], int]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    fields = list(reader.fieldnames or ())
    _require_fields(fields, {"filter_name", "predicate", "atoms"}, "filters CSV")
    if not {"actual_pct", "target_rate"} & set(fields):
        raise WorkloadError(
            "filters CSV must contain actual_pct or target_rate"
        )
    filters: list[Filter] = []
    seen: set[str] = set()
    for row_no, row in enumerate(reader, start=2):
        if None in row:
            raise WorkloadError("filters CSV contains a row wider than its header")
        name = _require_ascii(
            str(row.get("filter_name") or "").strip(),
            f"filters row {row_no} filter_name",
        )
        if not name:
            raise WorkloadError(f"filters row {row_no} has an empty filter_name")
        if name in seen:
            raise WorkloadError(f"filters CSV contains duplicate filter_name={name!r}")
        seen.add(name)
        predicate = _require_ascii(
            _normalize_sql(row.get("predicate")),
            f"filter {name} predicate",
        )
        atoms = _require_ascii(
            str(row.get("atoms") or "").strip(),
            f"filter {name} atoms",
        )
        if not predicate or not atoms:
            raise WorkloadError(f"filter {name} must have non-empty predicate and atoms")
        selectivity_field = (
            "actual_pct"
            if str(row.get("actual_pct") or "").strip()
            else "target_rate"
        )
        selectivity_raw = str(row.get(selectivity_field) or "").strip()
        filters.append(
            Filter(
                name=name,
                predicate=predicate,
                atoms=atoms,
                selectivity_field=selectivity_field,
                selectivity_raw=selectivity_raw,
                selectivity_pct=_selectivity_pct(
                    selectivity_field,
                    selectivity_raw,
                    f"filter {name} {selectivity_field}",
                ),
            )
        )
    if len(filters) != EXPECTED_FILTERS:
        raise WorkloadError(
            f"filters CSV must contain exactly {EXPECTED_FILTERS} rows, "
            f"observed={len(filters)}"
        )
    return filters, len(filters)


def audit_truth(
    payload: bytes,
    *,
    filters: Sequence[Filter],
    queries: Mapping[int, Query],
    required_query_nos: set[int],
    query_no_column: str,
    query_id_column: str,
    filter_column: str,
    k: int,
    expected_pairs: set[tuple[int, str]] | None = None,
) -> tuple[set[tuple[int, str]], dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    fields = list(reader.fieldnames or ())
    required = {
        query_no_column,
        query_id_column,
        filter_column,
        *TIE_AWARE_FIELDS,
    }
    _require_fields(fields, required, "truth CSV")
    by_filter = {item.name: item for item in filters}
    if expected_pairs is None:
        expected_pairs = {
            (query_no, filter_name)
            for query_no in required_query_nos
            for filter_name in by_filter
        }
    else:
        expected_pairs = set(expected_pairs)
        invalid_pairs = {
            pair
            for pair in expected_pairs
            if pair[0] not in required_query_nos or pair[1] not in by_filter
        }
        if invalid_pairs:
            raise WorkloadError(
                f"assigned truth contains invalid workload pairs: {sorted(invalid_pairs)[:3]}"
            )
    pairs: set[tuple[int, str]] = set()
    boundary_tied_pairs = 0
    audited_rows = 0
    total_rows = 0
    for row_no, row in enumerate(reader, start=2):
        total_rows += 1
        if None in row:
            raise WorkloadError("truth CSV contains a row wider than its header")
        query_no = _parse_int(
            row.get(query_no_column), f"truth row {row_no} query_no", minimum=0
        )
        if query_no not in required_query_nos:
            continue
        audited_rows += 1
        query = queries[query_no]
        query_id = _require_ascii(
            str(row.get(query_id_column) or "").strip(),
            f"truth row {row_no} query_id",
        )
        if query_id != query.query_id:
            raise WorkloadError(
                f"truth row {row_no} query_id differs from query cohort for "
                f"query_no={query_no}"
            )
        filter_name = _require_ascii(
            str(row.get(filter_column) or "").strip(),
            f"truth row {row_no} filter_name",
        )
        spec = by_filter.get(filter_name)
        if spec is None:
            raise WorkloadError(
                f"truth row {row_no} references unknown filter={filter_name!r}"
            )
        pair = (query_no, filter_name)
        if pair in pairs:
            raise WorkloadError(f"truth CSV contains duplicate pair={pair!r}")
        pairs.add(pair)

        if "predicate" in fields:
            predicate = _normalize_sql(row.get("predicate"))
            if predicate != spec.predicate:
                raise WorkloadError(
                    f"truth pair={pair!r} predicate differs from filters CSV"
                )
        if "k" in fields and str(row.get("k") or "").strip():
            observed_k = _parse_int(row.get("k"), f"truth pair={pair!r} k")
            if observed_k != k:
                raise WorkloadError(
                    f"truth pair={pair!r} uses k={observed_k}, expected={k}"
                )
        if "method" in fields and str(row.get("method") or "").strip():
            if str(row.get("method") or "").strip() != "pre_filter_exact":
                raise WorkloadError(
                    f"truth pair={pair!r} is not produced by pre_filter_exact"
                )

        exact_ids = _parse_csv_ints(
            row.get("exact_filtered_topk_ids"), f"truth pair={pair!r} exact IDs"
        )
        distances = _parse_csv_floats(
            row.get("exact_filtered_topk_distances_sq"),
            f"truth pair={pair!r} exact distances",
        )
        if len(exact_ids) != k or len(set(exact_ids)) != k:
            raise WorkloadError(
                f"truth pair={pair!r} must contain {k} unique exact IDs"
            )
        if len(distances) != k:
            raise WorkloadError(
                f"truth pair={pair!r} must contain {k} exact distances"
            )
        if any(right < left for left, right in zip(distances, distances[1:])):
            raise WorkloadError(
                f"truth pair={pair!r} exact distances are not ordered"
            )
        kth = _parse_float(
            row.get("kth_distance_sq"),
            f"truth pair={pair!r} kth_distance_sq",
            minimum=0.0,
        )
        if not math.isclose(distances[-1], kth, rel_tol=1e-8, abs_tol=1e-12):
            raise WorkloadError(
                f"truth pair={pair!r} kth distance differs from exact payload"
            )
        tolerance = _parse_float(
            row.get("tie_tolerance"),
            f"truth pair={pair!r} tie_tolerance",
            minimum=0.0,
        )
        if tolerance <= 0.0:
            raise WorkloadError(
                f"truth pair={pair!r} tie_tolerance must be positive"
            )
        strict = _parse_int(
            row.get("strict_closer_count"),
            f"truth pair={pair!r} strict_closer_count",
            minimum=0,
        )
        expected_strict = sum(value < kth - tolerance for value in distances)
        if strict != expected_strict or strict >= k:
            raise WorkloadError(
                f"truth pair={pair!r} strict_closer_count is inconsistent"
            )
        if _parse_bool(
            row.get("boundary_tied"), f"truth pair={pair!r} boundary_tied"
        ):
            boundary_tied_pairs += 1

        if "result_ids" in fields and str(row.get("result_ids") or "").strip():
            result_ids = _parse_csv_ints(
                row.get("result_ids"), f"truth pair={pair!r} result_ids"
            )
            if result_ids != exact_ids:
                raise WorkloadError(
                    f"truth pair={pair!r} result_ids differ from exact IDs"
                )

    missing = expected_pairs - pairs
    extra = pairs - expected_pairs
    if missing or extra:
        example = sorted(missing)[:3]
        raise WorkloadError(
            "truth CSV does not contain the complete required query/filter matrix: "
            f"missing={len(missing)}, extra={len(extra)}, examples={example}"
        )
    return pairs, {
        "valid": True,
        "contract": "complete_tie_aware_exact_sql_valid_topk_v1",
        "k": k,
        "input_rows": total_rows,
        "audited_rows": audited_rows,
        "required_pairs": len(expected_pairs),
        "matched_pairs": len(pairs),
        "boundary_tied_pairs": boundary_tied_pairs,
        "required_fields": sorted(required),
    }


def _seeded_permutation(
    values: Sequence[Any], *, seed: int, domain: str
) -> list[Any]:
    decorated = []
    for ordinal, value in enumerate(values):
        key = hashlib.sha256(
            f"figure5-v1\0{seed}\0{domain}\0{ordinal}".encode("ascii")
        ).digest()
        decorated.append((key, ordinal, value))
    return [value for _, _, value in sorted(decorated)]


def _balanced_filter_slots(
    filters: Sequence[Filter], count: int, *, seed: int, domain: str
) -> list[str]:
    extra_order = _seeded_permutation(
        [item.name for item in filters], seed=seed, domain=f"{domain}-extras"
    )
    base, remainder = divmod(count, len(filters))
    counts = {item.name: base for item in filters}
    for name in extra_order[:remainder]:
        counts[name] += 1
    slots = [
        item.name
        for item in filters
        for _ in range(counts[item.name])
    ]
    return _seeded_permutation(slots, seed=seed, domain=f"{domain}-slots")


def build_measurement_rows(
    queries: Mapping[int, Query],
    filters: Sequence[Filter],
    *,
    query_nos: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    selected = [queries[query_no] for query_no in query_nos]
    shuffled_queries = _seeded_permutation(
        selected, seed=seed, domain="measurement-queries"
    )
    slots = _balanced_filter_slots(
        filters, len(shuffled_queries), seed=seed, domain="measurement-filters"
    )
    return [
        {
            "request_no": request_no,
            "query_no": query.query_no,
            "query_id": query.query_id,
            "filter_name": filter_name,
            "trace_cycle": 0,
            "split": "measurement",
        }
        for request_no, (query, filter_name) in enumerate(
            zip(shuffled_queries, slots, strict=True)
        )
    ]


def build_calibration_rows(
    queries: Mapping[int, Query],
    filters: Sequence[Filter],
    *,
    query_nos: Sequence[int],
    requests: int,
    seed: int,
    protocol: str = LEGACY_CALIBRATION_PROTOCOL,
) -> list[dict[str, Any]]:
    if protocol == FORMAL_CALIBRATION_PROTOCOL:
        return build_formal_calibration_rows(
            queries,
            filters,
            query_nos=query_nos,
            seed=seed,
        )
    if protocol != LEGACY_CALIBRATION_PROTOCOL:
        raise WorkloadError(f"unknown calibration protocol: {protocol!r}")
    cohort = [queries[query_no] for query_no in query_nos]
    scheduled: list[tuple[Query, int]] = []
    cycle = 0
    while len(scheduled) < requests:
        shuffled = _seeded_permutation(
            cohort, seed=seed, domain=f"calibration-query-cycle-{cycle}"
        )
        remaining = requests - len(scheduled)
        scheduled.extend((query, cycle) for query in shuffled[:remaining])
        cycle += 1
    slots = _balanced_filter_slots(
        filters, requests, seed=seed, domain="calibration-filters"
    )
    return [
        {
            "request_no": request_no,
            "query_no": query.query_no,
            "query_id": query.query_id,
            "filter_name": filter_name,
            "trace_cycle": trace_cycle,
            "split": "calibration",
        }
        for request_no, ((query, trace_cycle), filter_name) in enumerate(
            zip(scheduled, slots, strict=True)
        )
    ]


def build_formal_calibration_rows(
    queries: Mapping[int, Query],
    filters: Sequence[Filter],
    *,
    query_nos: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    """Build one request for every calibration query/filter pair.

    A trace cycle contains every query once. Filters rotate over a seeded base
    query order, so every cycle is interleaved and all cycles form a complete
    Cartesian product without duplicate pairs.
    """
    base_queries = _seeded_permutation(
        [queries[query_no] for query_no in query_nos],
        seed=seed,
        domain="formal-calibration-base-queries",
    )
    ordered_filters = _seeded_permutation(
        list(filters),
        seed=seed,
        domain="formal-calibration-filters",
    )
    base_positions = {
        query.query_no: position for position, query in enumerate(base_queries)
    }
    rows: list[dict[str, Any]] = []
    for trace_cycle in range(len(ordered_filters)):
        cycle_queries = _seeded_permutation(
            base_queries,
            seed=seed,
            domain=f"formal-calibration-query-cycle-{trace_cycle}",
        )
        for query in cycle_queries:
            filter_index = (base_positions[query.query_no] + trace_cycle) % len(
                ordered_filters
            )
            rows.append(
                {
                    "request_no": len(rows),
                    "query_no": query.query_no,
                    "query_id": query.query_id,
                    "filter_name": ordered_filters[filter_index].name,
                    "trace_cycle": trace_cycle,
                    "split": "calibration",
                }
            )
    return rows


def _distribution(
    rows: Sequence[Mapping[str, Any]], filter_names: Sequence[str]
) -> dict[str, Any]:
    counts = Counter(str(row["filter_name"]) for row in rows)
    values = [counts[name] for name in filter_names]
    query_nos = [int(row["query_no"]) for row in rows]
    query_ids = [str(row["query_id"]) for row in rows]
    cycles = [int(row["trace_cycle"]) for row in rows]
    return {
        "requests": len(rows),
        "unique_queries": len(set(query_nos)),
        "unique_query_ids": len(set(query_ids)),
        "query_no_min": min(query_nos),
        "query_no_max": max(query_nos),
        "trace_cycles": max(cycles) + 1,
        "filter_counts": {name: counts[name] for name in filter_names},
        "filter_count_min": min(values),
        "filter_count_max": max(values),
        "filter_count_spread": max(values) - min(values),
    }


def _cartesian_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    query_nos: Sequence[int],
    filter_names: Sequence[str],
) -> dict[str, Any]:
    expected_pairs = {
        (query_no, filter_name)
        for query_no in query_nos
        for filter_name in filter_names
    }
    observed_pairs = [
        (int(row["query_no"]), str(row["filter_name"])) for row in rows
    ]
    observed_set = set(observed_pairs)
    canonical_pairs = "".join(
        f"{query_no}\t{filter_name}\n"
        for query_no, filter_name in sorted(observed_set)
    ).encode("ascii")
    return {
        "expected_pairs": len(expected_pairs),
        "observed_rows": len(observed_pairs),
        "observed_unique_pairs": len(observed_set),
        "missing_pairs": len(expected_pairs - observed_set),
        "duplicate_pairs": len(observed_pairs) - len(observed_set),
        "complete": observed_set == expected_pairs
        and len(observed_pairs) == len(expected_pairs),
        "canonical_pair_sha256": sha256_bytes(canonical_pairs),
    }


def validate_workload_rows(
    calibration: Sequence[Mapping[str, Any]],
    measurement: Sequence[Mapping[str, Any]],
    *,
    config: BuildConfig,
    queries: Mapping[int, Query],
    filters: Sequence[Filter],
    truth_pairs: set[tuple[int, str]],
) -> dict[str, bool]:
    filter_names = [item.name for item in filters]
    filter_set = set(filter_names)
    calibration_domain = set(config.calibration_query_nos)
    measurement_domain = set(config.measurement_query_nos)

    def validate_common(
        rows: Sequence[Mapping[str, Any]],
        split: str,
        domain: set[int],
        expected_rows: int,
    ) -> None:
        if len(rows) != expected_rows:
            raise WorkloadError(
                f"{split} row count mismatch: expected={expected_rows}, "
                f"observed={len(rows)}"
            )
        if [int(row["request_no"]) for row in rows] != list(range(expected_rows)):
            raise WorkloadError(f"{split} request_no values are not contiguous")
        if {str(row["split"]) for row in rows} != {split}:
            raise WorkloadError(f"{split} rows contain an incorrect split value")
        if {str(row["filter_name"]) for row in rows} != filter_set:
            raise WorkloadError(f"{split} does not cover all 14 filters")
        counts = Counter(str(row["filter_name"]) for row in rows)
        if max(counts.values()) - min(counts.values()) > 1:
            raise WorkloadError(f"{split} filter counts differ by more than one")
        for row in rows:
            query_no = int(row["query_no"])
            filter_name = str(row["filter_name"])
            if query_no not in domain:
                raise WorkloadError(
                    f"{split} contains query_no={query_no} outside its cohort"
                )
            if str(row["query_id"]) != queries[query_no].query_id:
                raise WorkloadError(
                    f"{split} query_id differs from the cohort for query_no={query_no}"
                )
            if (query_no, filter_name) not in truth_pairs:
                raise WorkloadError(
                    f"{split} request has no truth pair={(query_no, filter_name)!r}"
                )

    validate_common(
        calibration,
        "calibration",
        calibration_domain,
        config.calibration_requests,
    )
    validate_common(
        measurement,
        "measurement",
        measurement_domain,
        config.measurement_query_count,
    )
    measurement_query_nos = [int(row["query_no"]) for row in measurement]
    measurement_query_ids = [str(row["query_id"]) for row in measurement]
    if len(set(measurement_query_nos)) != config.measurement_query_count:
        raise WorkloadError("measurement query_no values are not unique")
    if len(set(measurement_query_ids)) != config.measurement_query_count:
        raise WorkloadError("measurement query vectors are not unique")
    if set(measurement_query_nos) != measurement_domain:
        raise WorkloadError("measurement query_no coverage is incomplete")

    seen_cycle_queries: set[tuple[int, int]] = set()
    for row in calibration:
        key = (int(row["trace_cycle"]), int(row["query_no"]))
        if key in seen_cycle_queries:
            raise WorkloadError(
                f"calibration repeats query_no={key[1]} within trace_cycle={key[0]}"
            )
        seen_cycle_queries.add(key)
    expected_unique_calibration = min(
        config.calibration_requests, config.calibration_query_count
    )
    if len({int(row["query_no"]) for row in calibration}) != expected_unique_calibration:
        raise WorkloadError("calibration unique-query count is inconsistent")

    formal_cartesian_coverage = _cartesian_coverage(
        calibration,
        query_nos=list(config.calibration_query_nos),
        filter_names=filter_names,
    )
    if config.calibration_protocol == FORMAL_CALIBRATION_PROTOCOL:
        if not formal_cartesian_coverage["complete"]:
            raise WorkloadError(
                "formal calibration does not cover every query/filter pair exactly once"
            )
        if Counter(str(row["filter_name"]) for row in calibration) != Counter(
            {name: config.calibration_query_count for name in filter_names}
        ):
            raise WorkloadError("formal calibration does not have exact per-filter counts")
        if len({int(row["trace_cycle"]) for row in calibration}) != EXPECTED_FILTERS:
            raise WorkloadError("formal calibration must contain exactly 14 trace cycles")

    return {
        "exactly_14_filters": len(filters) == EXPECTED_FILTERS,
        "truth_pair_coverage": True,
        "calibration_request_count": True,
        "calibration_filter_coverage": True,
        "calibration_filter_balance": True,
        "calibration_cycle_uniqueness": True,
        "calibration_cartesian_coverage": (
            formal_cartesian_coverage["complete"]
            if config.calibration_protocol == FORMAL_CALIBRATION_PROTOCOL
            else True
        ),
        "calibration_exact_per_filter_count": (
            all(
                count == config.calibration_query_count
                for count in Counter(
                    str(row["filter_name"]) for row in calibration
                ).values()
            )
            if config.calibration_protocol == FORMAL_CALIBRATION_PROTOCOL
            else True
        ),
        "measurement_request_count": True,
        "measurement_query_no_uniqueness": True,
        "measurement_query_vector_uniqueness": True,
        "measurement_filter_coverage": True,
        "measurement_filter_balance": True,
        "split_disjointness": not bool(calibration_domain & measurement_domain),
    }


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(
        target,
        fieldnames=list(OUTPUT_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    value = target.getvalue()
    _require_ascii(value, "workload CSV")
    return value.encode("ascii")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("ascii")


def manifest_content_sha256(manifest: Mapping[str, Any]) -> str:
    copy = json.loads(json.dumps(manifest))
    copy["outputs"]["manifest_json"].pop("content_sha256", None)
    return sha256_bytes(canonical_json_bytes(copy))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_file_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_file_fsynced(temporary, canonical_json_bytes(payload))
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def recover_atomic_bundle(journal_path: Path) -> bool:
    """Recover an interrupted publish, returning whether a journal was consumed."""
    if not journal_path.exists():
        return False
    try:
        journal = json.loads(journal_path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkloadError(f"cannot read recovery journal: {journal_path}") from exc
    entries = journal.get("entries")
    if not isinstance(entries, list) or not entries:
        raise WorkloadError(f"recovery journal has no entries: {journal_path}")
    committed = journal.get("phase") == "committed"
    parent = journal_path.parent.resolve()
    for entry in reversed(entries):
        destination = Path(str(entry["destination"]))
        staged = Path(str(entry["staged"]))
        backup = Path(str(entry["backup"]))
        if any(path.parent.resolve() != parent for path in (destination, staged, backup)):
            raise WorkloadError("recovery journal references a different directory")
        if committed:
            if not destination.exists():
                raise WorkloadError(
                    f"committed bundle is missing destination: {destination}"
                )
        elif backup.exists():
            os.replace(backup, destination)
        elif not bool(entry["old_existed"]):
            destination.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)
        if committed:
            backup.unlink(missing_ok=True)
    _fsync_directory(parent)
    journal_path.unlink()
    _fsync_directory(parent)
    return True


@contextmanager
def _publish_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkloadError(f"another workload publisher holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _install_staged_file(staged: Path, destination: Path) -> None:
    os.replace(staged, destination)


def recoverable_atomic_write_bundle(
    payloads: Mapping[Path, bytes],
    *,
    journal_path: Path,
    lock_path: Path,
) -> None:
    """Publish all files with rollback-on-restart recovery and manifest-last order."""
    if not payloads:
        raise WorkloadError("output bundle is empty")
    destinations = list(payloads)
    parent = destinations[0].parent.resolve()
    if any(path.parent.resolve() != parent for path in destinations):
        raise WorkloadError("all output bundle files must share one directory")
    parent.mkdir(parents=True, exist_ok=True)
    with _publish_lock(lock_path):
        recover_atomic_bundle(journal_path)
        transaction = uuid.uuid4().hex
        entries: list[dict[str, Any]] = []
        for destination in destinations:
            entries.append(
                {
                    "destination": str(destination.resolve()),
                    "staged": str(
                        destination.with_name(
                            f".{destination.name}.{transaction}.staged"
                        ).resolve()
                    ),
                    "backup": str(
                        destination.with_name(
                            f".{destination.name}.{transaction}.backup"
                        ).resolve()
                    ),
                    "old_existed": destination.exists(),
                }
            )
        journal: dict[str, Any] = {
            "schema_version": 1,
            "transaction": transaction,
            "phase": "prepared",
            "entries": entries,
        }
        _write_json_atomic(journal_path, journal)
        try:
            for destination, entry in zip(destinations, entries, strict=True):
                _write_file_fsynced(Path(entry["staged"]), payloads[destination])
            journal["phase"] = "staged"
            _write_json_atomic(journal_path, journal)
            for entry in entries:
                destination = Path(entry["destination"])
                if destination.exists():
                    os.replace(destination, Path(entry["backup"]))
            _fsync_directory(parent)
            for entry in entries:
                _install_staged_file(
                    Path(entry["staged"]), Path(entry["destination"])
                )
            _fsync_directory(parent)
            journal["phase"] = "committed"
            _write_json_atomic(journal_path, journal)
            recover_atomic_bundle(journal_path)
        except BaseException:
            recover_atomic_bundle(journal_path)
            raise


def _require_queries(
    queries: Mapping[int, Query], required_query_nos: Iterable[int]
) -> None:
    missing = sorted(set(required_query_nos) - set(queries))
    if missing:
        raise WorkloadError(
            f"query cohort is missing {len(missing)} required query numbers; "
            f"examples={missing[:5]}"
        )


def build(config: BuildConfig) -> dict[str, Any]:
    config.validate()
    input_specs: list[tuple[str, Path, str]] = [
        ("query_cohort_csv", config.query_cohort_csv, "query cohort CSV"),
        ("filters_csv", config.filters_csv, "filters CSV"),
    ]
    if config.truth_csv is not None:
        input_specs.append(("truth_csv", config.truth_csv, "truth CSV"))
    input_bytes: dict[str, bytes] = {}
    input_fields: dict[str, list[str]] = {}
    input_rows: dict[str, int] = {}
    for name, path, label in input_specs:
        payload, fields, rows = _read_csv_bytes(path, label)
        input_bytes[name] = payload
        input_fields[name] = fields
        input_rows[name] = len(rows)

    queries, query_rows = load_queries(
        input_bytes["query_cohort_csv"],
        query_no_column=config.query_no_column,
        query_id_column=config.query_id_column,
    )
    filters, filter_rows = load_filters(input_bytes["filters_csv"])
    required_query_nos = set(config.calibration_query_nos) | set(
        config.measurement_query_nos
    )
    _require_queries(queries, required_query_nos)
    calibration = build_calibration_rows(
        queries,
        filters,
        query_nos=list(config.calibration_query_nos),
        requests=config.calibration_requests,
        seed=config.seed,
        protocol=config.calibration_protocol,
    )
    measurement = build_measurement_rows(
        queries,
        filters,
        query_nos=list(config.measurement_query_nos),
        seed=config.seed,
    )
    assigned_pairs = {
        (int(row["query_no"]), str(row["filter_name"]))
        for row in (*calibration, *measurement)
    }
    if config.trace_only:
        truth_pairs = assigned_pairs
        truth_audit: dict[str, Any] = {
            "valid": False,
            "contract": "pending_exact_truth_for_frozen_assigned_pairs_v1",
            "k": config.k,
            "required_pairs": len(assigned_pairs),
            "matched_pairs": 0,
            "coverage": config.truth_coverage,
        }
    else:
        truth_pairs, truth_audit = audit_truth(
            input_bytes["truth_csv"],
            filters=filters,
            queries=queries,
            required_query_nos=required_query_nos,
            query_no_column=config.truth_query_no_column,
            query_id_column=config.truth_query_id_column,
            filter_column=config.truth_filter_column,
            k=config.k,
            expected_pairs=(
                assigned_pairs if config.truth_coverage == "assigned" else None
            ),
        )
        truth_audit["coverage"] = config.truth_coverage
    gates = validate_workload_rows(
        calibration,
        measurement,
        config=config,
        queries=queries,
        filters=filters,
        truth_pairs=truth_pairs,
    )
    gates.update(
        {
            "input_sha256_bound": True,
            "truth_tie_aware": bool(truth_audit["valid"]),
            "ascii_outputs": True,
            "output_sha256_verified": True,
        }
    )
    calibration_payload = csv_bytes(calibration)
    measurement_payload = csv_bytes(measurement)
    assigned = [
        {**row, "request_no": request_no}
        for request_no, row in enumerate((*calibration, *measurement))
    ]
    assigned_payload = csv_bytes(assigned)
    output_paths = config.output_paths
    filter_names = [item.name for item in filters]
    calibration_distribution = _distribution(calibration, filter_names)
    calibration_cartesian_coverage = _cartesian_coverage(
        calibration,
        query_nos=list(config.calibration_query_nos),
        filter_names=filter_names,
    )
    formal_paper_calibration_passed = (
        config.calibration_protocol == FORMAL_CALIBRATION_PROTOCOL
        and config.calibration_query_count == FORMAL_CALIBRATION_QUERY_COUNT
        and config.calibration_requests == FORMAL_CALIBRATION_REQUESTS
        and calibration_cartesian_coverage["complete"]
        and all(
            count == FORMAL_CALIBRATION_QUERY_COUNT
            for count in calibration_distribution["filter_counts"].values()
        )
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "figure5_frontier_workload",
        "artifact_valid": not config.trace_only and all(gates.values()),
        "stage": "trace_pending_truth" if config.trace_only else "audited",
        "gates": gates,
        "construction": {
            "algorithm": "sha256_seeded_permutation_v1",
            "seed": config.seed,
            "filter_assignment": (
                "complete_query_filter_cartesian_product"
                if config.calibration_protocol == FORMAL_CALIBRATION_PROTOCOL
                else "balanced_counts_spread_at_most_one"
            ),
            "truth_coverage": config.truth_coverage,
            "calibration": {
                "protocol": config.calibration_protocol,
                "query_start": config.calibration_query_start,
                "query_count": config.calibration_query_count,
                "requests": config.calibration_requests,
                "trace_cycle_repetition": (
                    config.calibration_protocol == LEGACY_CALIBRATION_PROTOCOL
                ),
                "per_predicate_cartesian": (
                    config.calibration_protocol == FORMAL_CALIBRATION_PROTOCOL
                ),
            },
            "measurement": {
                "query_start": config.measurement_query_start,
                "query_count": config.measurement_query_count,
                "requests": config.measurement_query_count,
                "queries_used_once": True,
            },
        },
        "columns": {
            "query_cohort": {
                "query_no": config.query_no_column,
                "query_id": config.query_id_column,
            },
            "truth": {
                "query_no": config.truth_query_no_column,
                "query_id": config.truth_query_id_column,
                "filter_name": config.truth_filter_column,
            },
            "workload": list(OUTPUT_FIELDS),
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_bytes(input_bytes[name]),
                "rows": (
                    query_rows
                    if name == "query_cohort_csv"
                    else filter_rows
                    if name == "filters_csv"
                    else input_rows[name]
                ),
                "fields": input_fields[name],
            }
            for name, path, _ in input_specs
        },
        "filters": [
            {
                "filter_name": item.name,
                "selectivity_field": item.selectivity_field,
                "selectivity_raw": item.selectivity_raw,
                "selectivity_pct": item.selectivity_pct,
                "predicate_sha256": sha256_bytes(item.predicate.encode("ascii")),
                "atoms_sha256": sha256_bytes(item.atoms.encode("ascii")),
            }
            for item in filters
        ],
        "truth": truth_audit,
        "distribution": {
            "calibration": {
                **calibration_distribution,
                "cartesian_coverage": calibration_cartesian_coverage,
            },
            "measurement": _distribution(measurement, filter_names),
        },
        "formal_paper_calibration": {
            "required": config.require_formal_paper_calibration,
            "passed": formal_paper_calibration_passed,
            "contract": {
                "protocol": FORMAL_CALIBRATION_PROTOCOL,
                "calibration_query_count": FORMAL_CALIBRATION_QUERY_COUNT,
                "calibration_requests": FORMAL_CALIBRATION_REQUESTS,
                "per_filter_requests": FORMAL_CALIBRATION_QUERY_COUNT,
                "cartesian_coverage": "every query/filter pair exactly once",
            },
        },
        "outputs": {
            "calibration_workload_csv": {
                "path": str(output_paths["calibration_workload_csv"].resolve()),
                "sha256": sha256_bytes(calibration_payload),
                "rows": len(calibration),
            },
            "measurement_workload_csv": {
                "path": str(output_paths["measurement_workload_csv"].resolve()),
                "sha256": sha256_bytes(measurement_payload),
                "rows": len(measurement),
            },
            "assigned_workload_csv": {
                "path": str(output_paths["assigned_workload_csv"].resolve()),
                "sha256": sha256_bytes(assigned_payload),
                "rows": len(assigned),
            },
            "manifest_json": {
                "path": str(output_paths["manifest_json"].resolve()),
                "content_sha256_contract": (
                    "sha256 of canonical manifest after removing "
                    "outputs.manifest_json.content_sha256"
                ),
            },
        },
        "publication": {
            "protocol": "recoverable_atomic_bundle_manifest_last_v1",
            "journal": str(config.journal_path.resolve()),
            "lock": str(config.lock_path.resolve()),
        },
    }
    if (
        not config.trace_only
        and config.require_formal_paper_calibration
        and not formal_paper_calibration_passed
    ):
        raise WorkloadError("formal paper calibration gate did not pass")
    if not config.trace_only and not manifest["artifact_valid"]:
        raise WorkloadError("workload artifact gates did not all pass")
    manifest["outputs"]["manifest_json"]["content_sha256"] = (
        manifest_content_sha256(manifest)
    )
    manifest_payload = canonical_json_bytes(manifest)
    payloads = {
        output_paths["calibration_workload_csv"]: calibration_payload,
        output_paths["measurement_workload_csv"]: measurement_payload,
        output_paths["assigned_workload_csv"]: assigned_payload,
        output_paths["manifest_json"]: manifest_payload,
    }
    recoverable_atomic_write_bundle(
        payloads,
        journal_path=config.journal_path,
        lock_path=config.lock_path,
    )
    if sha256_file(output_paths["calibration_workload_csv"]) != (
        manifest["outputs"]["calibration_workload_csv"]["sha256"]
    ):
        raise WorkloadError("published calibration workload SHA256 mismatch")
    if sha256_file(output_paths["measurement_workload_csv"]) != (
        manifest["outputs"]["measurement_workload_csv"]["sha256"]
    ):
        raise WorkloadError("published measurement workload SHA256 mismatch")
    if sha256_file(output_paths["assigned_workload_csv"]) != (
        manifest["outputs"]["assigned_workload_csv"]["sha256"]
    ):
        raise WorkloadError("published assigned workload SHA256 mismatch")
    return manifest


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic Figure 5 calibration and q10K measurement "
            "workloads from a query cohort, 14 filters, and complete tie-aware truth."
        )
    )
    parser.add_argument("--query-cohort-csv", type=Path, required=True)
    parser.add_argument("--filters-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--query-no-column", default="query_no")
    parser.add_argument("--query-id-column", default="query_id")
    parser.add_argument("--truth-query-no-column", default="query_no")
    parser.add_argument("--truth-query-id-column", default="query_id")
    parser.add_argument("--truth-filter-column", default="filter_name")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--calibration-query-start", type=int, default=0)
    parser.add_argument("--calibration-query-count", type=int, default=200)
    parser.add_argument("--calibration-requests", type=int, default=2_000)
    parser.add_argument(
        "--calibration-protocol",
        choices=tuple(sorted(CALIBRATION_PROTOCOLS)),
        default=LEGACY_CALIBRATION_PROTOCOL,
        help=(
            "legacy_balanced_v1 preserves historical traces; "
            "formal_per_predicate_cartesian_v1 emits every calibration "
            "query/filter pair exactly once."
        ),
    )
    parser.add_argument(
        "--require-formal-paper-calibration",
        action="store_true",
        help=(
            "Require the formal q2800 calibration contract: 200 queries x "
            "14 filters, with exactly 200 requests per filter."
        ),
    )
    parser.add_argument("--measurement-query-start", type=int, default=200)
    parser.add_argument("--measurement-query-count", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--truth-coverage",
        choices=("full", "assigned"),
        default="full",
        help=(
            "full requires every selected query/filter combination; assigned requires "
            "exact truth only for pairs present in the frozen traces."
        ),
    )
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help=(
            "Publish deterministic traces with an explicitly non-valid pending manifest; "
            "rerun with exact truth to create the audited artifact."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    config = BuildConfig(
        query_cohort_csv=args.query_cohort_csv,
        filters_csv=args.filters_csv,
        truth_csv=args.truth_csv,
        out_prefix=args.out_prefix,
        query_no_column=args.query_no_column,
        query_id_column=args.query_id_column,
        truth_query_no_column=args.truth_query_no_column,
        truth_query_id_column=args.truth_query_id_column,
        truth_filter_column=args.truth_filter_column,
        seed=args.seed,
        calibration_query_start=args.calibration_query_start,
        calibration_query_count=args.calibration_query_count,
        calibration_requests=args.calibration_requests,
        calibration_protocol=args.calibration_protocol,
        require_formal_paper_calibration=args.require_formal_paper_calibration,
        measurement_query_start=args.measurement_query_start,
        measurement_query_count=args.measurement_query_count,
        k=args.k,
        truth_coverage=args.truth_coverage,
        trace_only=args.trace_only,
    )
    try:
        manifest = build(config)
    except (OSError, WorkloadError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        json.dumps(
            {
                "artifact_valid": manifest["artifact_valid"],
                "calibration": manifest["distribution"]["calibration"],
                "measurement": manifest["distribution"]["measurement"],
                "manifest": manifest["outputs"]["manifest_json"]["path"],
            },
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
