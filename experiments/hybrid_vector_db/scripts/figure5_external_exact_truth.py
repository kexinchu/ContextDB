#!/usr/bin/env python3
"""Generate exact SQL-valid truth for assigned Figure 5 external pairs.

The generator validates the frozen assignment and uses either the legacy
by-predicate scan or, only for a proven complete query/filter Cartesian product,
one shared-query GEMM per base chunk.  Both paths use float32 distances and the
same exact top-(k+1) merge; the manifest records which audited path ran.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - production environments have torch
    torch = None  # type: ignore[assignment]


SCHEMA_VERSION = 3
K_DEFAULT = 10
EXPECTED_FILTERS = 14
OUTPUT_FIELDS = (
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


class ExactTruthError(RuntimeError):
    """Raised for an invalid or incomplete exact-truth input/output."""


@dataclass(frozen=True)
class FilterSpec:
    name: str
    predicate: str
    labels: tuple[int, ...]
    actual_selectivity: float | None
    expected_rows: int | None
    field: str
    # "or" = tags && ARRAY[...] (overlap); "and" = tags @> ARRAY[...] (containment).
    match_mode: str = "or"


@dataclass(frozen=True)
class AssignedPair:
    request_no: int
    query_no: int
    query_id: int
    filter_name: str


@dataclass(frozen=True)
class InputBundle:
    dataset: str
    workload: Path
    filters: Path
    query_vectors: Path
    base_vectors: tuple[Path, ...]
    metadata: Path | None
    label_offsets: Path | None
    flat_labels: Path | None
    base_row_limit: int | None

    def input_paths(self) -> tuple[Path, ...]:
        paths = [self.workload, self.filters, self.query_vectors, *self.base_vectors]
        for path in (self.metadata, self.label_offsets, self.flat_labels):
            if path is not None:
                paths.append(path)
        return tuple(paths)


_PREDICATE_RE = re.compile(
    r"^(?P<field>tags|labels)\s*(?P<op>&&|@>)\s*ARRAY\s*\[\s*(?P<labels>[0-9]+(?:\s*,\s*[0-9]+)*)\s*\]"
    r"\s*::\s*int\s*\[\s*\s*\]\s*$",
    re.IGNORECASE,
)
_ATOM_RE = re.compile(
    r"^sql:(?P<field>tags|labels)\s*@>\s*ARRAY\s*\[\s*(?P<label>[0-9]+)\s*\]"
    r"\s*::\s*int\s*\[\s*\s*\]\s*$",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ExactTruthError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ExactTruthError(f"{label} is not finite: {value!r}")
    return result


def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ExactTruthError(f"invalid boolean: {value!r}")


def parse_labels_from_predicate(
    predicate: str, dataset: str
) -> tuple[str, tuple[int, ...], str]:
    match = _PREDICATE_RE.fullmatch(" ".join(predicate.strip().split()))
    if match is None:
        raise ExactTruthError(f"unsupported predicate: {predicate!r}")
    field = match.group("field").lower()
    expected = "tags" if dataset == "yfcc" else "labels"
    if field != expected:
        raise ExactTruthError(
            f"{dataset} predicate uses {field!r}; expected {expected!r}: {predicate!r}"
        )
    labels = tuple(int(item.strip()) for item in match.group("labels").split(","))
    if not labels or len(set(labels)) != len(labels):
        raise ExactTruthError(f"predicate labels must be non-empty and unique: {predicate!r}")
    op = match.group("op")
    match_mode = "and" if op == "@>" else "or"
    return field, labels, match_mode


def parse_filter_atoms(
    value: Any, dataset: str, *, match_mode: str
) -> tuple[str, tuple[int, ...]]:
    text = str(value or "").strip()
    if not text:
        raise ExactTruthError("filter atoms are empty")
    if match_mode == "or":
        # Historical overlap encoding: atom||OR||atom
        if "||OR||" in text:
            parts = [part.strip() for part in text.split("||OR||") if part.strip()]
        else:
            parts = [text]
    else:
        # AND encoding: atom||atom (no OR separator tokens)
        parts = [part.strip() for part in text.split("||") if part.strip()]
        if any(part.upper() == "OR" for part in parts):
            raise ExactTruthError(
                "AND filter atoms must not contain OR separators; use atom||atom"
            )
    if not parts:
        raise ExactTruthError("filter atoms are empty")
    expected = "tags" if dataset == "yfcc" else "labels"
    parsed: list[int] = []
    for atom in parts:
        match = _ATOM_RE.fullmatch(" ".join(atom.split()))
        if match is None or match.group("field").lower() != expected:
            raise ExactTruthError(f"unsupported filter atom: {atom!r}")
        parsed.append(int(match.group("label")))
    labels = tuple(parsed)
    if len(set(labels)) != len(labels):
        raise ExactTruthError("filter atoms contain duplicate labels")
    return expected, labels


def load_filters(path: Path, dataset: str) -> dict[str, FilterSpec]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ExactTruthError(f"empty filter CSV: {path}")
    required = {"filter_name", "predicate", "atoms"}
    missing = required - set(rows[0])
    if missing:
        raise ExactTruthError(f"filter CSV missing fields: {sorted(missing)}")
    specs: dict[str, FilterSpec] = {}
    for row in rows:
        name = str(row.get("filter_name", "")).strip()
        if not name or name in specs:
            raise ExactTruthError(f"duplicate/empty filter name: {name!r}")
        predicate = str(row.get("predicate", "")).strip()
        field, predicate_labels, match_mode = parse_labels_from_predicate(
            predicate, dataset
        )
        atom_field, atom_labels = parse_filter_atoms(
            row.get("atoms"), dataset, match_mode=match_mode
        )
        if field != atom_field or set(predicate_labels) != set(atom_labels):
            raise ExactTruthError(
                f"predicate/atoms mismatch for {name}: predicate={predicate_labels}, atoms={atom_labels}"
            )
        if match_mode == "and" and len(predicate_labels) >= 1 and "||OR||" in str(
            row.get("atoms") or ""
        ):
            raise ExactTruthError(
                f"AND predicate {name} must not encode atoms with ||OR||"
            )
        actual = None
        for key in ("actual_selectivity", "actual_pct"):
            if str(row.get(key, "")).strip():
                actual = finite_float(row[key], f"{name}.{key}")
                if key == "actual_pct":
                    actual /= 100.0
                if actual < 0 or actual > 1:
                    raise ExactTruthError(f"{name} selectivity is outside [0,1]: {actual}")
                break
        expected = None
        if str(row.get("expected_rows", "")).strip():
            try:
                expected = int(row["expected_rows"])
            except ValueError as exc:
                raise ExactTruthError(f"invalid expected_rows for {name}") from exc
            if expected < 0:
                raise ExactTruthError(f"negative expected_rows for {name}")
        specs[name] = FilterSpec(
            name, predicate, predicate_labels, actual, expected, field, match_mode
        )
    if len(specs) != EXPECTED_FILTERS:
        raise ExactTruthError(f"formal Figure 5 filters require 14 rows, found {len(specs)}")
    return specs


def load_workload(path: Path, filters: Mapping[str, FilterSpec]) -> list[AssignedPair]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    required = {"request_no", "query_no", "query_id", "filter_name"}
    if not rows:
        raise ExactTruthError(f"empty workload CSV: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ExactTruthError(f"workload CSV missing fields: {sorted(missing)}")
    result: list[AssignedPair] = []
    request_numbers: set[int] = set()
    pairs: set[tuple[int, str]] = set()
    query_no_to_id: dict[int, int] = {}
    for row in rows:
        try:
            request_no = int(row["request_no"])
            query_no = int(row["query_no"])
            query_id = int(row["query_id"])
        except (TypeError, ValueError) as exc:
            raise ExactTruthError(f"non-integer workload row: {row}") from exc
        name = str(row["filter_name"]).strip()
        if request_no in request_numbers:
            raise ExactTruthError(f"duplicate request_no={request_no}")
        if (query_id, name) in pairs:
            raise ExactTruthError(f"duplicate assigned pair query_id={query_id}, filter={name}")
        if name not in filters:
            raise ExactTruthError(f"workload references unknown filter {name!r}")
        if query_no in query_no_to_id and query_no_to_id[query_no] != query_id:
            raise ExactTruthError(f"query_no={query_no} maps to multiple query IDs")
        request_numbers.add(request_no)
        pairs.add((query_id, name))
        query_no_to_id[query_no] = query_id
        result.append(AssignedPair(request_no, query_no, query_id, name))
    expected = list(range(len(result)))
    if sorted(request_numbers) != expected:
        raise ExactTruthError("request_no must be contiguous from zero")
    return sorted(result, key=lambda item: item.request_no)


def cartesian_workload_proof(
    pairs: Sequence[AssignedPair],
    filters: Mapping[str, FilterSpec],
) -> dict[str, Any]:
    """Prove whether a workload is the complete shared-query filter Cartesian product.

    This is deliberately stricter than merely observing a balanced filter count.  The
    fast path is valid only when every formal filter is paired exactly once with every
    query ID in one common query set.  The canonical pair hash makes that proof part of
    the truth artifact rather than an inferred runtime optimization.
    """
    expected_filters = tuple(sorted(filters))
    observed_filters = tuple(sorted({pair.filter_name for pair in pairs}))
    query_ids = tuple(sorted({pair.query_id for pair in pairs}))
    canonical = tuple(
        (pair.query_id, pair.filter_name)
        for pair in sorted(pairs, key=lambda item: (item.query_id, item.filter_name, item.request_no))
    )
    canonical_text = "".join(f"{query_id}\t{filter_name}\n" for query_id, filter_name in canonical)
    proof: dict[str, Any] = {
        "checked": True,
        "formal_filter_count": len(expected_filters),
        "observed_filter_count": len(observed_filters),
        "shared_query_count": len(query_ids),
        "expected_pair_count": len(expected_filters) * len(query_ids),
        "observed_pair_count": len(pairs),
        "canonical_pair_sha256": hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
        "eligible": False,
        "reason": "",
    }
    if observed_filters != expected_filters:
        proof["reason"] = "workload does not contain exactly the formal filter set"
        return proof
    if not query_ids:
        proof["reason"] = "workload has no query IDs"
        return proof
    if len(canonical) != len(set(canonical)):
        proof["reason"] = "duplicate (query_id, filter_name) pair"
        return proof
    expected_pairs = {(query_id, filter_name) for query_id in query_ids for filter_name in expected_filters}
    observed_pairs = set(canonical)
    if observed_pairs != expected_pairs:
        proof["reason"] = "incomplete shared query/filter Cartesian product"
        return proof
    proof["eligible"] = True
    proof["reason"] = "complete shared query/filter Cartesian product"
    return proof


def read_xbin(path: Path, dtype: np.dtype[Any]) -> np.memmap:
    header = np.fromfile(path, dtype="<u4", count=2)
    if len(header) != 2:
        raise ExactTruthError(f"invalid xbin header: {path}")
    rows, dim = int(header[0]), int(header[1])
    expected = 8 + rows * dim * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ExactTruthError(f"xbin size mismatch for {path}: {path.stat().st_size} != {expected}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(rows, dim))


def read_fbin(path: Path) -> np.memmap:
    header = np.fromfile(path, dtype="<i4", count=2)
    if len(header) != 2:
        raise ExactTruthError(f"invalid fbin header: {path}")
    rows, dim = int(header[0]), int(header[1])
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ExactTruthError(f"fbin size mismatch for {path}: {path.stat().st_size} != {expected}")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))


def read_spmat(path: Path) -> tuple[int, np.memmap, np.memmap]:
    header = np.fromfile(path, dtype="<i8", count=3)
    if len(header) != 3:
        raise ExactTruthError(f"invalid spmat header: {path}")
    rows, _cols, nnz = (int(value) for value in header)
    indptr_offset = 3 * 8
    indptr = np.memmap(path, dtype="<i8", mode="r", offset=indptr_offset, shape=(rows + 1,))
    indices_offset = indptr_offset + (rows + 1) * 8
    indices = np.memmap(path, dtype="<i4", mode="r", offset=indices_offset, shape=(nnz,))
    if int(indptr[0]) != 0 or int(indptr[-1]) != nnz or np.any(np.diff(indptr) < 0):
        raise ExactTruthError(f"invalid CSR pointers in {path}")
    return rows, indptr, indices


def membership_mask_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    start: int,
    end: int,
    labels: Sequence[int],
    *,
    match_mode: str = "or",
) -> np.ndarray:
    """Return rows in [start,end) matching labels (OR overlap or AND containment)."""
    n = end - start
    mask = np.zeros(n, dtype=bool)
    if not labels or start == end:
        return mask
    lo, hi = int(indptr[start]), int(indptr[end])
    segment = np.asarray(indices[lo:hi])
    if segment.size == 0:
        return mask
    ptr = indptr[start : end + 1]
    label_arr = np.asarray(tuple(labels), dtype=segment.dtype)
    if match_mode == "or":
        hit = np.flatnonzero(np.isin(segment, label_arr))
        if hit.size:
            local_positions = (
                np.searchsorted(ptr, hit.astype(np.int64) + lo, side="right") - 1
            )
            mask[np.unique(local_positions)] = True
        return mask
    if match_mode != "and":
        raise ExactTruthError(f"unsupported match_mode: {match_mode!r}")
    # AND: row must contain every requested label.
    combined = np.ones(n, dtype=bool)
    for label in label_arr:
        part = np.zeros(n, dtype=bool)
        hit = np.flatnonzero(segment == label)
        if hit.size:
            local_positions = (
                np.searchsorted(ptr, hit.astype(np.int64) + lo, side="right") - 1
            )
            part[np.unique(local_positions)] = True
        combined &= part
        if not combined.any():
            break
    return combined


def membership_mask_offsets(
    offsets: np.ndarray,
    labels_flat: np.ndarray,
    start: int,
    end: int,
    labels: Sequence[int],
    *,
    match_mode: str = "or",
) -> np.ndarray:
    return membership_mask_csr(
        offsets, labels_flat, start, end, labels, match_mode=match_mode
    )


def merge_topk(
    current_distances: Any,
    current_ids: Any,
    candidate_distances: Any,
    candidate_ids: Any,
    width: int,
) -> tuple[Any, Any]:
    """Merge tensors shaped [queries, width] and [queries, candidates]."""
    if candidate_distances.shape[1] == 0:
        return current_distances, current_ids
    take = min(width, int(candidate_distances.shape[1]))
    local_distances, local_positions = torch.topk(
        candidate_distances, k=take, dim=1, largest=False, sorted=True
    )
    local_ids = torch.gather(candidate_ids.expand(candidate_distances.shape[0], -1), 1, local_positions)
    merged_distances = torch.cat((current_distances, local_distances), dim=1)
    merged_ids = torch.cat((current_ids, local_ids), dim=1)
    distances, positions = torch.topk(merged_distances, k=width, dim=1, largest=False, sorted=True)
    return distances, torch.gather(merged_ids, 1, positions)


def tie_fields(distances: Sequence[float], k: int) -> dict[str, Any]:
    if len(distances) < k or any(not math.isfinite(float(value)) for value in distances[:k]):
        raise ExactTruthError(f"incomplete/non-finite top-k: {distances}")
    kth = float(distances[k - 1])
    tolerance = max(1e-9, abs(kth) * 1e-6)
    strict = sum(float(value) < kth - tolerance for value in distances[:k])
    boundary = len(distances) > k and float(distances[k]) <= kth + tolerance
    return {
        "kth_distance_sq": f"{kth:.9g}",
        "tie_tolerance": f"{tolerance:.9g}",
        "strict_closer_count": strict,
        "boundary_tied": boundary,
    }


def cuda_device(value: str) -> Any:
    if torch is None:
        raise ExactTruthError("PyTorch is unavailable")
    if str(value).strip().lower() == "cpu":
        raise ExactTruthError("CUDA device is required; CPU is test-only")
    if not torch.cuda.is_available():
        raise ExactTruthError("CUDA is unavailable")
    try:
        device = torch.device(value if str(value).startswith("cuda:") else f"cuda:{value}")
        index = 0 if device.index is None else int(device.index)
    except (RuntimeError, ValueError) as exc:
        raise ExactTruthError(f"invalid CUDA device {value!r}") from exc
    if index < 0 or index >= torch.cuda.device_count():
        raise ExactTruthError(f"CUDA device {index} is unavailable")
    return device


def cpu_device(cpu_threads: int | None) -> tuple[Any, dict[str, Any]]:
    """Configure an explicitly requested CPU exact-truth execution path."""
    if torch is None:
        raise ExactTruthError("PyTorch is unavailable")
    if cpu_threads is not None:
        if cpu_threads <= 0:
            raise ExactTruthError("--cpu-threads must be positive")
        try:
            torch.set_num_threads(cpu_threads)
        except RuntimeError as exc:
            raise ExactTruthError(f"cannot set Torch CPU threads to {cpu_threads}") from exc
    return torch.device("cpu"), {
        "requested_device": "cpu",
        "device_type": "cpu",
        "resolved_device": "cpu",
        "cpu_threads_requested": cpu_threads,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def resolve_device(args: argparse.Namespace, *, device: Any | None = None) -> tuple[Any, dict[str, Any]]:
    """Resolve an explicitly requested execution device without CPU fallback."""
    requested = getattr(args, "device", None)
    legacy_cuda = getattr(args, "cuda_device", None)
    cpu_threads = getattr(args, "cpu_threads", None)
    if requested is not None and legacy_cuda is not None:
        raise ExactTruthError("use either --device or --cuda-device, not both")
    if requested is None and legacy_cuda is None:
        raise ExactTruthError("an explicit --device cpu/cuda[:N] or --cuda-device is required")
    if requested is None:
        requested = str(legacy_cuda)
        resolved = cuda_device(requested)
        index = 0 if resolved.index is None else int(resolved.index)
        return resolved, cuda_provenance(resolved, f"cuda:{index}")

    requested_text = str(requested).strip().lower()
    if requested_text == "cpu":
        if device is not None and str(device) != "cpu":
            raise ExactTruthError("injected device does not match --device cpu")
        return cpu_device(cpu_threads)
    if cpu_threads is not None:
        raise ExactTruthError("--cpu-threads is only valid with --device cpu")
    if requested_text == "cuda":
        requested_text = "cuda:0"
    if not requested_text.startswith("cuda:"):
        raise ExactTruthError("--device must be cpu, cuda, or cuda:N")
    if device is not None and str(device) != requested_text:
        raise ExactTruthError(f"injected device {device!s} does not match --device {requested_text}")
    resolved = cuda_device(requested_text)
    return resolved, cuda_provenance(resolved, requested_text)


def cuda_provenance(device: Any, requested_device: str) -> dict[str, Any]:
    if torch is None:
        raise ExactTruthError("PyTorch is unavailable")
    index = 0 if device.index is None else int(device.index)
    return {
        "requested_device": requested_device,
        "device_type": "cuda",
        "resolved_device": str(device),
        "device_index": index,
        "device_name": torch.cuda.get_device_name(index),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "tf32_disabled": True,
    }


def torch_backend_provenance(device_provenance: Mapping[str, Any]) -> dict[str, Any]:
    if torch is None:
        raise ExactTruthError("PyTorch is unavailable")
    result: dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": dict(device_provenance),
    }
    if str(device_provenance["device_type"]) == "cpu":
        result["cpu_backend"] = {
            "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
            "mkl_available": bool(torch.backends.mkl.is_available()),
            "openmp_available": bool(torch.backends.openmp.is_available()),
        }
    return result


def _vector_source(path: Path, dataset: str) -> np.memmap | np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2:
            raise ExactTruthError(f"vector source is not 2-D: {path}")
        if dataset == "laion" and np.dtype(array.dtype) not in {
            np.dtype("<f2"),
            np.dtype("<f4"),
        }:
            raise ExactTruthError(
                f"LAION vector source must be float16/float32: {path} has {array.dtype}"
            )
        return array
    if dataset == "yfcc" and suffix == ".u8bin":
        return read_xbin(path, np.dtype("u1"))
    if dataset == "laion" and suffix == ".fbin":
        return read_fbin(path)
    raise ExactTruthError(f"unsupported vector source for {dataset}: {path}")


def iter_base_chunks(
    paths: Sequence[Path],
    dataset: str,
    chunk_rows: int,
    row_limit: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    offset = 0
    remaining = row_limit
    for path in paths:
        if remaining is not None and remaining <= 0:
            break
        array = _vector_source(path, dataset)
        take = int(array.shape[0])
        if remaining is not None:
            take = min(take, remaining)
        for start in range(0, take, chunk_rows):
            end = min(start + chunk_rows, take)
            yield offset + start, array[start:end]
        offset += take
        if remaining is not None:
            remaining -= take
    if remaining is not None and remaining > 0:
        raise ExactTruthError(
            f"base vector sources are {remaining} rows short of row_limit={row_limit}"
        )


def metadata_sources(bundle: InputBundle, dataset: str) -> tuple[int, Any, Any]:
    if dataset == "yfcc":
        if bundle.metadata is None:
            raise ExactTruthError("YFCC requires --base-metadata-source")
        return read_spmat(bundle.metadata)
    if bundle.label_offsets is None or bundle.flat_labels is None:
        raise ExactTruthError("LAION requires --label-offsets-source and --flat-labels-source")
    if bundle.label_offsets.stat().st_size % 8 or bundle.flat_labels.stat().st_size % 4:
        raise ExactTruthError("LAION label source has invalid byte size")
    rows = bundle.label_offsets.stat().st_size // 8 - 1
    offsets = np.memmap(bundle.label_offsets, dtype="<i8", mode="r", shape=(rows + 1,))
    labels = np.memmap(bundle.flat_labels, dtype="<i4", mode="r", shape=(bundle.flat_labels.stat().st_size // 4,))
    if int(offsets[0]) != 0 or int(offsets[-1]) > len(labels) or np.any(np.diff(offsets) < 0):
        raise ExactTruthError("invalid LAION label offsets")
    return int(rows), offsets, labels


def _filter_mask(
    dataset: str,
    metadata: tuple[int, Any, Any],
    start: int,
    end: int,
    labels: Sequence[int],
    *,
    match_mode: str = "or",
) -> np.ndarray:
    rows, first, second = metadata
    if end > rows:
        raise ExactTruthError(f"metadata rows do not cover base row {end}")
    if dataset == "yfcc":
        return membership_mask_csr(
            first, second, start, end, labels, match_mode=match_mode
        )
    return membership_mask_offsets(
        first, second, start, end, labels, match_mode=match_mode
    )


def _prepare_exact_context(
    bundle: InputBundle,
    pairs: Sequence[AssignedPair],
) -> tuple[np.ndarray, tuple[int, Any, Any], int]:
    query_dtype = np.dtype("u1") if bundle.dataset == "yfcc" else np.dtype("<f4")
    queries = _vector_source(bundle.query_vectors, bundle.dataset)
    if queries.dtype != query_dtype:
        queries = np.asarray(queries, dtype=query_dtype)
    metadata = metadata_sources(bundle, bundle.dataset)
    metadata_rows = metadata[0]
    source_base_rows = sum(int(_vector_source(path, bundle.dataset).shape[0]) for path in bundle.base_vectors)
    base_rows = bundle.base_row_limit if bundle.base_row_limit is not None else source_base_rows
    if base_rows <= 0 or base_rows > source_base_rows:
        raise ExactTruthError(f"invalid base row limit: limit={base_rows}, available={source_base_rows}")
    if metadata_rows != base_rows:
        raise ExactTruthError(f"base vectors/metadata row mismatch: {base_rows} != {metadata_rows}")
    if any(pair.query_id < 0 or pair.query_id >= int(queries.shape[0]) for pair in pairs):
        raise ExactTruthError("workload references a query outside the query vector source")
    if int(queries.shape[1]) == 0:
        raise ExactTruthError("query vector dimension is zero")
    return queries, metadata, base_rows


def _result_row(
    pair: AssignedPair,
    spec: FilterSpec,
    distances: Sequence[float],
    ids: Sequence[int],
    *,
    matched_count: int,
    base_rows: int,
    k: int,
) -> dict[str, Any]:
    width = k + 1
    if len(ids) < width or any(value < 0 for value in ids[:k]):
        raise ExactTruthError(f"incomplete top-k for pair {(pair.query_id, spec.name)}")
    if any(not math.isfinite(value) for value in distances[:k]):
        raise ExactTruthError(f"non-finite top-k for pair {(pair.query_id, spec.name)}")
    ties = tie_fields(distances, k)
    result_ids = list(ids[:k])
    result_distances = list(distances[:k])
    return {
        "query_no": pair.query_no,
        "query_id": pair.query_id,
        "filter_name": spec.name,
        "predicate": spec.predicate,
        "actual_selectivity": f"{matched_count / base_rows:.12g}",
        "candidate_validity_predicate": "TRUE",
        "candidate_validity_provenance": "full_base_scan_and_source_metadata",
        "query_validity_predicate": "TRUE",
        "query_validity_provenance": "external_query_vector_source",
        "method": "pre_filter_exact",
        "k": k,
        "latency_ms": "",
        "recall_at_10_exact_filtered": "1.0",
        "returned": k,
        "candidates": matched_count,
        "filtered_rows": matched_count,
        "search_candidate_rows": matched_count,
        "result_ids": ",".join(str(value) for value in result_ids),
        "exact_filtered_topk_ids": ",".join(str(value) for value in result_ids),
        "exact_filtered_topk_distances_sq": ",".join(f"{value:.9g}" for value in result_distances),
        **ties,
        "self_excluded": "false",
        "candidate_rows": matched_count,
        "self_excluded_rows": 0,
    }


def exact_assigned_pairs_by_filter(
    bundle: InputBundle,
    pairs: Sequence[AssignedPair],
    filters: Mapping[str, FilterSpec],
    *,
    device: Any,
    query_batch: int = 32,
    chunk_rows: int = 100_000,
    k: int = K_DEFAULT,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    if torch is None:
        raise ExactTruthError("PyTorch is unavailable")
    if query_batch <= 0 or chunk_rows <= 0 or k <= 0:
        raise ExactTruthError("query_batch, chunk_rows, and k must be positive")
    if str(device).startswith("cuda"):
        # Exact truth must not silently use Ampere TF32 products.
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
    queries, metadata, base_rows = _prepare_exact_context(bundle, pairs)
    by_filter: dict[str, list[AssignedPair]] = {}
    for pair in pairs:
        by_filter.setdefault(pair.filter_name, []).append(pair)
    output_by_pair: dict[tuple[int, str], dict[str, Any]] = {}
    candidate_counts: dict[str, int] = {}
    width = k + 1
    gemm_passes = 0
    base_chunks_scanned = 0
    for filter_name, filter_pairs in by_filter.items():
        spec = filters[filter_name]
        query_ids = [pair.query_id for pair in filter_pairs]
        q_count = len(query_ids)
        filter_started = time.perf_counter()
        print(
            f"[{bundle.dataset}] exact filter={filter_name} "
            f"queries={q_count} labels={len(spec.labels)}",
            flush=True,
        )
        top_distances = torch.full((q_count, width), float("inf"), dtype=torch.float32, device=device)
        top_ids = torch.full((q_count, width), -1, dtype=torch.int64, device=device)
        q_norm: Any | None = None
        matched_count = 0
        chunks_scanned = 0
        for global_start, base_chunk in iter_base_chunks(
            bundle.base_vectors,
            bundle.dataset,
            chunk_rows,
            bundle.base_row_limit,
        ):
            chunks_scanned += 1
            base_chunks_scanned += 1
            global_end = global_start + int(base_chunk.shape[0])
            mask = _filter_mask(
                bundle.dataset,
                metadata,
                global_start,
                global_end,
                spec.labels,
                match_mode=spec.match_mode,
            )
            local_count = int(np.count_nonzero(mask))
            matched_count += local_count
            if local_count == 0:
                continue
            selected = np.ascontiguousarray(np.asarray(base_chunk[mask], dtype=np.float32))
            if not np.isfinite(selected).all():
                raise ExactTruthError(f"non-finite base vector in {filter_name}")
            candidate_id_cpu = np.arange(global_start, global_end, dtype=np.int64)[mask]
            candidate_ids = torch.from_numpy(candidate_id_cpu).to(device=device, dtype=torch.int64)
            base_gpu = torch.from_numpy(selected).to(device=device, dtype=torch.float32)
            base_norm = (base_gpu * base_gpu).sum(dim=1)
            for q_start in range(0, q_count, query_batch):
                q_end = min(q_start + query_batch, q_count)
                q_cpu = np.asarray(queries[query_ids[q_start:q_end]], dtype=np.float32)
                if not np.isfinite(q_cpu).all():
                    raise ExactTruthError(f"non-finite query vector in {filter_name}")
                q_gpu = torch.from_numpy(np.ascontiguousarray(q_cpu)).to(device=device, dtype=torch.float32)
                q_norm_batch = (q_gpu * q_gpu).sum(dim=1)
                distances = (
                    base_norm[:, None]
                    + q_norm_batch[None, :]
                    - 2.0 * torch.matmul(base_gpu, q_gpu.transpose(0, 1))
                ).transpose(0, 1)
                distances = torch.clamp_min(distances, 0.0)
                gemm_passes += 1
                current_d = top_distances[q_start:q_end]
                current_i = top_ids[q_start:q_end]
                top_distances[q_start:q_end], top_ids[q_start:q_end] = merge_topk(
                    current_d,
                    current_i,
                    distances,
                    candidate_ids,
                    width,
                )
                del q_gpu, distances
            del base_gpu, candidate_ids
            if q_norm is not None:
                del q_norm
            if chunks_scanned % 25 == 0:
                print(
                    f"[{bundle.dataset}] filter={filter_name} "
                    f"rows={global_end}/{base_rows} matched={matched_count}",
                    flush=True,
                )
        if matched_count < k:
            raise ExactTruthError(f"filter {filter_name} has only {matched_count} candidates; need {k}")
        candidate_counts[filter_name] = matched_count
        if spec.expected_rows is not None and spec.expected_rows != matched_count:
            raise ExactTruthError(
                f"filter {filter_name} expected_rows={spec.expected_rows}, observed={matched_count}"
            )
        for index, pair in enumerate(filter_pairs):
            distances = [float(value) for value in top_distances[index].detach().cpu().tolist()]
            ids = [int(value) for value in top_ids[index].detach().cpu().tolist()]
            output_by_pair[(pair.query_id, filter_name)] = _result_row(
                pair,
                spec,
                distances,
                ids,
                matched_count=matched_count,
                base_rows=base_rows,
                k=k,
            )
        print(
            f"[{bundle.dataset}] exact filter={filter_name} complete "
            f"candidates={matched_count} elapsed_s={time.perf_counter() - filter_started:.1f}",
            flush=True,
        )
    return (
        [output_by_pair[(pair.query_id, pair.filter_name)] for pair in sorted(pairs, key=lambda item: item.request_no)],
        candidate_counts,
        {
            "execution_path": "legacy_by_filter",
            "base_scan_passes": len(by_filter),
            "base_chunks_scanned": base_chunks_scanned,
            "gemm_passes": gemm_passes,
        },
    )


def exact_assigned_pairs_cartesian(
    bundle: InputBundle,
    pairs: Sequence[AssignedPair],
    filters: Mapping[str, FilterSpec],
    *,
    device: Any,
    query_batch: int = 32,
    chunk_rows: int = 100_000,
    k: int = K_DEFAULT,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    """Exact truth for a proven shared-query Cartesian workload.

    Every base chunk is transferred and multiplied with each shared query batch once.
    The resulting exact float32 distances are then masked independently for every
    predicate before the unchanged top-(k+1) merge.  No candidate, distance, or tie
    decision is borrowed from another predicate.
    """
    if torch is None:
        raise ExactTruthError("PyTorch is unavailable")
    if query_batch <= 0 or chunk_rows <= 0 or k <= 0:
        raise ExactTruthError("query_batch, chunk_rows, and k must be positive")
    proof = cartesian_workload_proof(pairs, filters)
    if not proof["eligible"]:
        raise ExactTruthError(f"Cartesian fast path requires complete proof: {proof['reason']}")
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")

    queries, metadata, base_rows = _prepare_exact_context(bundle, pairs)
    query_ids = sorted({pair.query_id for pair in pairs})
    q_count = len(query_ids)
    width = k + 1
    top_distances = {
        name: torch.full((q_count, width), float("inf"), dtype=torch.float32, device=device)
        for name in filters
    }
    top_ids = {
        name: torch.full((q_count, width), -1, dtype=torch.int64, device=device)
        for name in filters
    }
    candidate_counts = {name: 0 for name in filters}
    gemm_passes = 0
    base_chunks_scanned = 0

    for global_start, base_chunk in iter_base_chunks(
        bundle.base_vectors,
        bundle.dataset,
        chunk_rows,
        bundle.base_row_limit,
    ):
        base_chunks_scanned += 1
        global_end = global_start + int(base_chunk.shape[0])
        masks = {
            name: _filter_mask(
                bundle.dataset,
                metadata,
                global_start,
                global_end,
                spec.labels,
                match_mode=spec.match_mode,
            )
            for name, spec in filters.items()
        }
        for name, mask in masks.items():
            candidate_counts[name] += int(np.count_nonzero(mask))
        base_cpu = np.ascontiguousarray(np.asarray(base_chunk, dtype=np.float32))
        if not np.isfinite(base_cpu).all():
            raise ExactTruthError("non-finite base vector in Cartesian scan")
        base_device = torch.from_numpy(base_cpu).to(device=device, dtype=torch.float32)
        base_norm = (base_device * base_device).sum(dim=1)
        all_ids = torch.arange(global_start, global_end, device=device, dtype=torch.int64)
        for q_start in range(0, q_count, query_batch):
            q_end = min(q_start + query_batch, q_count)
            q_cpu = np.asarray(queries[query_ids[q_start:q_end]], dtype=np.float32)
            if not np.isfinite(q_cpu).all():
                raise ExactTruthError("non-finite query vector in Cartesian scan")
            q_device = torch.from_numpy(np.ascontiguousarray(q_cpu)).to(device=device, dtype=torch.float32)
            q_norm = (q_device * q_device).sum(dim=1)
            distances = (
                base_norm[:, None]
                + q_norm[None, :]
                - 2.0 * torch.matmul(base_device, q_device.transpose(0, 1))
            ).transpose(0, 1)
            distances = torch.clamp_min(distances, 0.0)
            gemm_passes += 1
            for name, mask in masks.items():
                if not mask.any():
                    continue
                positions = torch.from_numpy(np.flatnonzero(mask).astype(np.int64, copy=False)).to(device=device)
                candidate_distances = torch.index_select(distances, 1, positions)
                candidate_ids = torch.index_select(all_ids, 0, positions)
                current_d = top_distances[name][q_start:q_end]
                current_i = top_ids[name][q_start:q_end]
                top_distances[name][q_start:q_end], top_ids[name][q_start:q_end] = merge_topk(
                    current_d,
                    current_i,
                    candidate_distances,
                    candidate_ids,
                    width,
                )
            del q_device, distances
        del base_device, all_ids
        if base_chunks_scanned % 25 == 0:
            print(f"[{bundle.dataset}] Cartesian exact rows={global_end}/{base_rows}", flush=True)

    query_position = {query_id: index for index, query_id in enumerate(query_ids)}
    rows_by_pair: dict[tuple[int, str], dict[str, Any]] = {}
    for name, spec in filters.items():
        matched_count = candidate_counts[name]
        if matched_count < k:
            raise ExactTruthError(f"filter {name} has only {matched_count} candidates; need {k}")
        if spec.expected_rows is not None and spec.expected_rows != matched_count:
            raise ExactTruthError(
                f"filter {name} expected_rows={spec.expected_rows}, observed={matched_count}"
            )
    for pair in pairs:
        name = pair.filter_name
        index = query_position[pair.query_id]
        distances = [float(value) for value in top_distances[name][index].detach().cpu().tolist()]
        ids = [int(value) for value in top_ids[name][index].detach().cpu().tolist()]
        rows_by_pair[(pair.query_id, name)] = _result_row(
            pair,
            filters[name],
            distances,
            ids,
            matched_count=candidate_counts[name],
            base_rows=base_rows,
            k=k,
        )
    return (
        [rows_by_pair[(pair.query_id, pair.filter_name)] for pair in sorted(pairs, key=lambda item: item.request_no)],
        candidate_counts,
        {
            "execution_path": "audited_cartesian_shared_query_gemm",
            "base_scan_passes": 1,
            "base_chunks_scanned": base_chunks_scanned,
            "gemm_passes": gemm_passes,
            "shared_query_batches": math.ceil(q_count / query_batch),
        },
    )


def exact_assigned_pairs(
    bundle: InputBundle,
    pairs: Sequence[AssignedPair],
    filters: Mapping[str, FilterSpec],
    *,
    device: Any,
    query_batch: int = 32,
    chunk_rows: int = 100_000,
    k: int = K_DEFAULT,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any], dict[str, Any]]:
    proof = cartesian_workload_proof(pairs, filters)
    if proof["eligible"]:
        rows, counts, execution = exact_assigned_pairs_cartesian(
            bundle, pairs, filters, device=device, query_batch=query_batch, chunk_rows=chunk_rows, k=k
        )
    else:
        rows, counts, execution = exact_assigned_pairs_by_filter(
            bundle, pairs, filters, device=device, query_batch=query_batch, chunk_rows=chunk_rows, k=k
        )
    return rows, counts, execution, proof


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            os.fsync(target.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def build_manifest(
    bundle: InputBundle,
    pairs: Sequence[AssignedPair],
    filters: Mapping[str, FilterSpec],
    candidate_counts: Mapping[str, int],
    output: Path,
    device: Any,
    device_provenance: Mapping[str, Any],
    k: int,
    execution: Mapping[str, Any],
    cartesian_proof: Mapping[str, Any],
) -> dict[str, Any]:
    if torch is None:
        raise ExactTruthError("PyTorch is unavailable")
    method = f"full_base_scan_plus_{device_provenance['device_type']}_float32_gemm_topk"
    input_hashes = {str(path): sha256_file(path) for path in bundle.input_paths()}
    source_base_rows = sum(
        int(_vector_source(path, bundle.dataset).shape[0])
        for path in bundle.base_vectors
    )
    base_rows = (
        bundle.base_row_limit
        if bundle.base_row_limit is not None
        else source_base_rows
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "figure5_external_exact_truth.py",
        "created_at": utc_now(),
        "dataset": bundle.dataset,
        "device": dict(device_provenance),
        "torch_backend": torch_backend_provenance(device_provenance),
        "k": k,
        "workload": {
            "path": str(bundle.workload),
            "requests": len(pairs),
            "unique_pairs": len(pairs),
            "unique_queries": len({pair.query_id for pair in pairs}),
            "unique_filters": len({pair.filter_name for pair in pairs}),
        },
        "filters": {
            "path": str(bundle.filters),
            "count": len(filters),
            "names": list(filters),
            "candidate_counts": dict(candidate_counts),
        },
        "exact_coverage": {
            "assigned_pairs": len(pairs),
            "emitted_rows": len(pairs),
            "complete": True,
            "method": method,
            "self_excluded": False,
            "candidate_transfer": (
                "all_base_vectors_for_shared_query_gemm"
                if execution["execution_path"] == "audited_cartesian_shared_query_gemm"
                else "predicate_matches_only"
            ),
            "base_rows": base_rows,
            "source_base_rows": source_base_rows,
            "base_row_limit_applied": bundle.base_row_limit is not None,
        },
        "execution": {
            **dict(execution),
            "cartesian_proof": dict(cartesian_proof),
        },
        "inputs": input_hashes,
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": len(pairs)},
    }


def publish_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(manifest, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def run_generation(args: argparse.Namespace, *, device: Any | None = None) -> dict[str, Any]:
    dataset = str(args.dataset).lower()
    if dataset not in {"yfcc", "laion"}:
        raise ExactTruthError(f"unsupported dataset {dataset!r}")
    output = Path(args.output_truth_csv)
    manifest_path = Path(args.output_manifest)
    if (output.exists() or manifest_path.exists()) and not args.overwrite:
        raise ExactTruthError("output exists; pass --overwrite")
    bundle = InputBundle(
        dataset,
        Path(args.workload_csv),
        Path(args.filters_csv),
        Path(args.query_vector_source),
        tuple(Path(value) for value in args.base_vector_source),
        Path(args.base_metadata_source) if args.base_metadata_source else None,
        Path(args.label_offsets_source) if args.label_offsets_source else None,
        Path(args.flat_labels_source) if args.flat_labels_source else None,
        (
            int(args.base_row_limit)
            if getattr(args, "base_row_limit", None) is not None
            else None
        ),
    )
    for path in bundle.input_paths():
        if not path.is_file():
            raise ExactTruthError(f"input does not exist: {path}")
    filters = load_filters(bundle.filters, dataset)
    pairs = load_workload(bundle.workload, filters)
    actual_device, device_provenance = resolve_device(args, device=device)
    rows, candidate_counts, execution, cartesian_proof = exact_assigned_pairs(
        bundle,
        pairs,
        filters,
        device=actual_device,
        query_batch=int(args.query_batch),
        chunk_rows=int(args.chunk_rows),
        k=int(args.k),
    )
    _atomic_write_csv(output, rows)
    manifest = build_manifest(
        bundle,
        pairs,
        filters,
        candidate_counts,
        output,
        actual_device,
        device_provenance,
        int(args.k),
        execution,
        cartesian_proof,
    )
    publish_manifest(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("yfcc", "laion"), required=True)
    parser.add_argument("--workload-csv", required=True)
    parser.add_argument("--filters-csv", required=True)
    parser.add_argument("--query-vector-source", "--query-source", dest="query_vector_source", required=True)
    parser.add_argument("--base-vector-source", dest="base_vector_source", nargs="+", required=True)
    parser.add_argument("--base-metadata-source", "--metadata-source", dest="base_metadata_source")
    parser.add_argument("--label-offsets-source")
    parser.add_argument("--flat-labels-source")
    parser.add_argument(
        "--base-row-limit",
        type=int,
        help="Use exactly this many rows across the ordered base vector sources.",
    )
    parser.add_argument("--output-truth-csv", "--truth-out", dest="output_truth_csv", required=True)
    parser.add_argument("--output-manifest", "--manifest-out", dest="output_manifest", required=True)
    parser.add_argument(
        "--device",
        help="Explicit exact-truth device: cpu, cuda, or cuda:N. CPU is never selected implicitly.",
    )
    parser.add_argument("--cuda-device", help="Legacy CUDA ordinal or cuda:N; incompatible with --device")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        help="Torch intra-op threads for an explicit --device cpu run.",
    )
    parser.add_argument("--query-batch", type=int, default=32)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_generation(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
