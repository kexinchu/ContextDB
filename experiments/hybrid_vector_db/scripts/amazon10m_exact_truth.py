from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psycopg

from common_pg import pg_config_from_env
from faiss_hnsw_sql_attribute_filter_10m import read_fbin_memmap


DEFAULT_FBIN = Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin")
DEFAULT_FILTERS = Path("experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv")
DEFAULT_OUT = Path("results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_formal.csv")
DEFAULT_VALIDITY_PREDICATE = "embedding_valid"
CHECKPOINT_SCHEMA_VERSION = 2
EXTERNAL_QUERY_COHORT_SCHEMA_VERSION = 1
EXTERNAL_QUERY_COHORT_METHOD = "deterministic_unique_projection_fingerprint_cohort_v1"
STRICT_PREFIX_EXTENSION_VERSION = 1


def safe_sql_predicate(value: str) -> str:
    predicate = value.strip()
    if not predicate:
        raise argparse.ArgumentTypeError("predicate must not be empty")
    if "\x00" in predicate or any(marker in predicate for marker in (";", "--", "/*", "*/")):
        raise argparse.ArgumentTypeError("predicate must not contain semicolons or SQL comments")
    return predicate


def predicate_declares_embedding_valid(predicate: str) -> bool:
    return re.search(r"\bembedding_valid\b", predicate, flags=re.IGNORECASE) is not None


def resolve_candidate_validity_predicate(candidate_predicate: str | None) -> tuple[str, str]:
    if candidate_predicate is None:
        return DEFAULT_VALIDITY_PREDICATE, "formal_default_embedding_valid"
    return candidate_predicate, "explicit_cli_predicate"


def resolve_query_validity_predicate(candidate_predicate: str, query_predicate: str | None) -> str:
    return candidate_predicate if query_predicate is None else query_predicate


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_filters(path: Path, selected: set[str]) -> list[dict[str, str]]:
    rows = read_csv(path)
    if selected:
        rows = [row for row in rows if row["filter_name"] in selected]
        missing = selected - {row["filter_name"] for row in rows}
        if missing:
            raise SystemExit(f"missing filter specs: {sorted(missing)}")
    if not rows:
        raise SystemExit(f"no filters in {path}")
    return rows


def sample_disjoint_query_ids(
    rows: int,
    excluded: set[int],
    count: int,
    seed: int,
) -> np.ndarray:
    if count > rows - len(excluded):
        raise ValueError("not enough rows for a disjoint final query set")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    chosen_set: set[int] = set()
    batch = max(1024, count * 4)
    while len(chosen) < count:
        for query_id in rng.integers(0, rows, size=batch, endpoint=False):
            value = int(query_id)
            if value in excluded or value in chosen_set:
                continue
            chosen.append(value)
            chosen_set.add(value)
            if len(chosen) == count:
                break
    return np.asarray(chosen, dtype=np.int64)


def sample_disjoint_eligible_query_ids(
    eligible_ids: np.ndarray,
    excluded: set[int],
    count: int,
    seed: int,
) -> np.ndarray:
    ids = np.asarray(eligible_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("eligible query IDs must be one-dimensional")
    if ids.size and np.any(ids[1:] <= ids[:-1]):
        raise ValueError("eligible query IDs must be sorted and unique")
    excluded_eligible = sum(
        int(position < ids.size and int(ids[position]) == value)
        for value in excluded
        for position in [int(np.searchsorted(ids, value))]
    )
    if count < 0 or count > ids.size - excluded_eligible:
        raise ValueError("not enough eligible rows for a disjoint query set")
    if count == 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    chosen_set: set[int] = set()
    batch = max(1024, count * 4)
    while len(chosen) < count:
        for position in rng.integers(0, ids.size, size=batch, endpoint=False):
            value = int(ids[int(position)])
            if value in excluded or value in chosen_set:
                continue
            chosen.append(value)
            chosen_set.add(value)
            if len(chosen) == count:
                break
    return np.asarray(chosen, dtype=np.int64)


def sorted_ids_sha256(ids: np.ndarray) -> str:
    normalized = np.ascontiguousarray(np.asarray(ids, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(b"sqlens-sorted-postgres-id-population-v1\0")
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def ordered_ids_sha256(ids: np.ndarray) -> str:
    normalized = np.ascontiguousarray(np.asarray(ids, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(b"sqlens-ordered-id-population-v1\0")
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".checkpoint.json")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as destination:
        destination.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_external_query_cohort(
    cohort_path: Path,
    manifest_path: Path,
    *,
    calibration_queries: int,
    calibration_seed: int,
    final_queries: int,
    final_seed: int,
    candidate_validity_predicate: str,
    query_validity_predicate: str,
    fbin_contract: dict[str, str],
    vector_rows: int,
    table_identity: dict[str, Any],
) -> dict[str, Any]:
    required_fields = [
        "query_no",
        "query_id",
        "query_split",
        "candidate_validity_predicate",
        "query_validity_predicate",
    ]
    with cohort_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if list(reader.fieldnames or ()) != required_fields:
            raise SystemExit("external query cohort does not use the dedicated strict schema")
        rows = list(reader)
    expected_count = calibration_queries + final_queries
    if len(rows) != expected_count:
        raise SystemExit(
            f"external query cohort row mismatch: expected={expected_count} observed={len(rows)}"
        )
    query_ids: list[int] = []
    for query_no, row in enumerate(rows):
        expected_split = "calibration" if query_no < calibration_queries else "final"
        try:
            observed_query_no = int(row["query_no"])
            query_id = int(row["query_id"])
        except (TypeError, ValueError) as exc:
            raise SystemExit("external query cohort contains a non-integer mapping") from exc
        if observed_query_no != query_no or row["query_split"] != expected_split:
            raise SystemExit("external query cohort numbering/split contract is invalid")
        if (
            row["candidate_validity_predicate"] != candidate_validity_predicate
            or row["query_validity_predicate"] != query_validity_predicate
        ):
            raise SystemExit("external query cohort validity predicate is incompatible")
        if query_id < 0 or query_id >= vector_rows:
            raise SystemExit(f"external query cohort ID is outside fbin: {query_id}")
        query_ids.append(query_id)
    if len(set(query_ids)) != len(query_ids):
        raise SystemExit("external query cohort IDs are not unique")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("external query cohort manifest is unreadable") from exc
    outputs = manifest.get("outputs")
    cohort_output = outputs.get("cohort_csv") if isinstance(outputs, dict) else None
    inputs = manifest.get("inputs")
    manifest_fbin = inputs.get("fbin") if isinstance(inputs, dict) else None
    manifest_postgres = inputs.get("postgres") if isinstance(inputs, dict) else None
    selection = manifest.get("selection")
    calibration = selection.get("calibration") if isinstance(selection, dict) else None
    final = selection.get("final") if isinstance(selection, dict) else None
    eligible = manifest.get("eligible_query_population")
    uniqueness = manifest.get("uniqueness_contract")
    ids_array = np.asarray(query_ids, dtype=np.int64)
    calibration_ids = ids_array[:calibration_queries]
    final_ids = ids_array[calibration_queries:]
    failures = []
    if manifest.get("schema_version") != EXTERNAL_QUERY_COHORT_SCHEMA_VERSION:
        failures.append("schema_version")
    if manifest.get("artifact_valid") is not True:
        failures.append("artifact_valid")
    if manifest.get("method") != EXTERNAL_QUERY_COHORT_METHOD:
        failures.append("method")
    if manifest.get("candidate_validity_predicate") != candidate_validity_predicate:
        failures.append("candidate_validity_predicate")
    if manifest.get("query_validity_predicate") != query_validity_predicate:
        failures.append("query_validity_predicate")
    if not isinstance(cohort_output, dict) or (
        cohort_output.get("sha256") != sha256_file(cohort_path)
        or int(cohort_output.get("rows", -1)) != expected_count
    ):
        failures.append("outputs.cohort_csv")
    if not isinstance(manifest_fbin, dict) or (
        manifest_fbin.get("sha256") != fbin_contract["sha256"]
        or int(manifest_fbin.get("rows", -1)) != vector_rows
    ):
        failures.append("inputs.fbin")
    table_fields = {
        "rows": "rows",
        "min_id": "min_id",
        "max_id": "max_id",
        "table_oid": "oid",
        "table_relfilenode": "relfilenode",
    }
    if not isinstance(manifest_postgres, dict) or any(
        int(manifest_postgres.get(manifest_field, -1))
        != int(table_identity[identity_field])
        for manifest_field, identity_field in table_fields.items()
    ):
        failures.append("inputs.postgres")
    if not isinstance(calibration, dict) or (
        int(calibration.get("queries", -1)) != calibration_queries
        or int(calibration.get("seed", -1)) != calibration_seed
        or calibration.get("ordered_ids_sha256") != ordered_ids_sha256(calibration_ids)
    ):
        failures.append("selection.calibration")
    if not isinstance(final, dict) or (
        int(final.get("queries", -1)) != final_queries
        or int(final.get("seed", -1)) != final_seed
        or final.get("ordered_ids_sha256") != ordered_ids_sha256(final_ids)
    ):
        failures.append("selection.final")
    if not isinstance(selection, dict) or (
        selection.get("disjoint") is not True
        or selection.get("query_ids_sha256") != ordered_ids_sha256(ids_array)
    ):
        failures.append("selection")
    if not isinstance(eligible, dict) or (
        isinstance(eligible.get("rows"), bool)
        or not isinstance(eligible.get("rows"), int)
        or int(eligible["rows"]) < expected_count
        or isinstance(eligible.get("embedding_valid_rows"), bool)
        or not isinstance(eligible.get("embedding_valid_rows"), int)
        or int(eligible["embedding_valid_rows"]) < int(eligible["rows"])
        or not isinstance(eligible.get("ids_sha256"), str)
        or len(eligible["ids_sha256"]) != 64
        or not isinstance(eligible.get("provenance"), str)
    ):
        failures.append("eligible_query_population")
    if not isinstance(uniqueness, dict) or (
        uniqueness.get("all_rows_fingerprinted") is not True
        or uniqueness.get("duplicate_admission_false_negative_only") is not True
        or int(uniqueness.get("hashes", -1)) != 2
    ):
        failures.append("uniqueness_contract")
    if failures:
        raise SystemExit(
            "external query cohort manifest contract failed: " + ", ".join(failures)
        )
    return {
        "query_ids": ids_array,
        "calibration_query_ids": calibration_ids,
        "final_query_ids": final_ids,
        "eligible_query_population": {
            "rows": int(eligible["rows"]),
            "ids_sha256": str(eligible["ids_sha256"]),
            "provenance": str(eligible["provenance"]),
        },
        "candidate_universe_rows": int(eligible["embedding_valid_rows"]),
        "source_contract": {
            "kind": "external_unique_vector_query_cohort",
            "method": EXTERNAL_QUERY_COHORT_METHOD,
            "cohort_csv": {
                "path": str(cohort_path.resolve()),
                "sha256": sha256_file(cohort_path),
            },
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "uniqueness_contract": uniqueness,
        },
    }


def update_topk(
    top_dist: np.ndarray,
    top_ids: np.ndarray,
    distances: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
) -> None:
    if distances.shape[0] == 0:
        return
    take = min(k, distances.shape[0])
    for query_pos in range(distances.shape[1]):
        column = distances[:, query_pos]
        threshold = np.partition(column, take - 1)[take - 1]
        local_pos = np.flatnonzero(column <= threshold)
        local_order = np.lexsort((candidate_ids[local_pos], column[local_pos]))[:take]
        selected = local_pos[local_order]
        merged_dist = np.concatenate((top_dist[query_pos], column[selected]))
        merged_ids = np.concatenate((top_ids[query_pos], candidate_ids[selected]))
        finite = np.isfinite(merged_dist) & (merged_ids >= 0)
        merged_dist = merged_dist[finite]
        merged_ids = merged_ids[finite]
        order = np.lexsort((merged_ids, merged_dist))[:k]
        top_dist[query_pos].fill(np.inf)
        top_ids[query_pos].fill(-1)
        top_dist[query_pos, : len(order)] = merged_dist[order]
        top_ids[query_pos, : len(order)] = merged_ids[order]


def merge_retained_topk(
    top_dist: np.ndarray,
    top_ids: np.ndarray,
    local_dist: np.ndarray,
    local_ids: np.ndarray,
) -> None:
    """Merge one chunk's sorted retained candidates deterministically."""
    if top_dist.shape != top_ids.shape or local_dist.shape != local_ids.shape:
        raise ValueError("distance and ID arrays must have matching shapes")
    if top_dist.shape != local_dist.shape:
        raise ValueError("global and local retained arrays must have matching shapes")
    retained = top_dist.shape[1]
    for query_pos in range(top_dist.shape[0]):
        distances = np.concatenate((top_dist[query_pos], local_dist[query_pos]))
        ids = np.concatenate((top_ids[query_pos], local_ids[query_pos]))
        finite = np.isfinite(distances) & (ids >= 0)
        distances = distances[finite]
        ids = ids[finite]
        order = np.lexsort((ids, distances))[:retained]
        top_dist[query_pos].fill(np.inf)
        top_ids[query_pos].fill(-1)
        top_dist[query_pos, : len(order)] = distances[order]
        top_ids[query_pos, : len(order)] = ids[order]


def exact_topk_batch(
    xb: np.memmap,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
    chunk_rows: int,
    progress_chunks: int,
    filter_name: str,
    distance_workers: int = 1,
) -> tuple[list[list[int]], list[list[float]], float]:
    try:
        from scipy.spatial.distance import cdist
    except ImportError as exc:
        raise SystemExit(
            "exact_topk_batch requires scipy; install scipy before generating exact truth"
        ) from exc
    if k <= 0:
        raise ValueError("k must be positive")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if distance_workers <= 0:
        raise ValueError("distance_workers must be positive")
    query_ids = np.asarray(query_ids, dtype=np.int64)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    queries = np.asarray(xb[query_ids], dtype=np.float64)
    # k+1 is enough to expose a boundary tie while retaining the lowest IDs
    # deterministically when more than k+1 candidates share that distance.
    retained = k + 1
    top_dist = np.full((len(query_ids), retained), np.inf, dtype=np.float64)
    top_ids = np.full((len(query_ids), retained), -1, dtype=np.int64)
    started = time.perf_counter()

    def compute_chunk(start: int) -> tuple[int, int, np.ndarray, np.ndarray]:
        ids = candidate_ids[start : start + chunk_rows]
        vectors = np.asarray(xb[ids], dtype=np.float64)
        distances = cdist(vectors, queries, metric="sqeuclidean")
        for query_pos, query_id in enumerate(query_ids):
            self_positions = np.flatnonzero(ids == query_id)
            if self_positions.size:
                distances[self_positions, query_pos] = np.inf
        local_dist = np.full((len(query_ids), retained), np.inf, dtype=np.float64)
        local_ids = np.full((len(query_ids), retained), -1, dtype=np.int64)
        update_topk(local_dist, local_ids, distances, ids, retained)
        return start, min(start + chunk_rows, len(candidate_ids)), local_dist, local_ids

    starts = list(range(0, len(candidate_ids), chunk_rows))
    if distance_workers == 1:
        computed: Iterable[tuple[int, int, np.ndarray, np.ndarray]] = (
            compute_chunk(start) for start in starts
        )
        executor = None
    else:
        executor = ThreadPoolExecutor(
            max_workers=distance_workers,
            thread_name_prefix="exact-l2",
        )
        futures: list[Future[tuple[int, int, np.ndarray, np.ndarray]]] = [
            executor.submit(compute_chunk, start) for start in starts[:distance_workers]
        ]

        def ordered_results() -> Iterable[tuple[int, int, np.ndarray, np.ndarray]]:
            next_start = distance_workers
            for future in futures:
                yield future.result()
                if next_start < len(starts):
                    futures.append(executor.submit(compute_chunk, starts[next_start]))
                    next_start += 1

        computed = ordered_results()

    try:
        for chunk_no, (start, done, local_dist, local_ids) in enumerate(computed, start=1):
            merge_retained_topk(top_dist, top_ids, local_dist, local_ids)
            del local_dist, local_ids
            if progress_chunks > 0 and chunk_no % progress_chunks == 0:
                elapsed = (time.perf_counter() - started) / 60.0
                print(
                    f"filter={filter_name} exact rows={done}/{len(candidate_ids)} "
                    f"workers={distance_workers} elapsed={elapsed:.1f} min",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ids_out: list[list[int]] = []
    distances_out: list[list[float]] = []
    for query_pos, query_id in enumerate(query_ids):
        retained_ids = top_ids[query_pos][top_ids[query_pos] >= 0]
        if retained_ids.size < k:
            raise ValueError(
                f"filter={filter_name} query_id={int(query_id)} has only "
                f"{retained_ids.size} search candidates after self-exclusion; need k={k}"
            )
        retained_distances = top_dist[query_pos][top_ids[query_pos] >= 0]
        order = np.lexsort((retained_ids, retained_distances))
        ids_out.append([int(value) for value in retained_ids[order]])
        distances_out.append([float(value) for value in retained_distances[order]])
    return ids_out, distances_out, elapsed_ms


def fetch_eligible_query_ids(
    cur: psycopg.Cursor,
    table: str,
    validity_predicate: str,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    cur.execute(f"SELECT id FROM {table} WHERE ({validity_predicate}) ORDER BY id")
    ids = np.fromiter((int(row[0]) for row in cur), dtype=np.int64)
    return ids, (time.perf_counter() - started) * 1000.0


def fetch_candidate_ids(
    cur: psycopg.Cursor,
    table: str,
    predicate: str,
    validity_predicate: str = DEFAULT_VALIDITY_PREDICATE,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    cur.execute(f"SELECT id FROM {table} WHERE ({predicate}) AND ({validity_predicate})")
    ids = np.fromiter((int(row[0]) for row in cur), dtype=np.int64)
    return ids, (time.perf_counter() - started) * 1000.0


def distance_tolerance(distance_sq: float) -> float:
    return max(1e-9, abs(distance_sq) * 1e-6)


def parse_vector_text(value: str) -> np.ndarray:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("unexpected PostgreSQL vector text")
    return np.fromstring(text[1:-1], sep=",", dtype=np.float32)


def verify_query_vector_mapping(
    cur: psycopg.Cursor,
    table: str,
    xb: np.memmap,
    query_ids: np.ndarray,
    query_validity_predicate: str = DEFAULT_VALIDITY_PREDICATE,
) -> dict[str, Any]:
    absolute_tolerance = 1e-7
    relative_tolerance = 1e-6
    wanted = [int(value) for value in query_ids]
    cur.execute(
        f"SELECT id, embedding::text FROM {table} "
        f"WHERE id = ANY(%s::bigint[]) AND ({query_validity_predicate})",
        (wanted,),
    )
    observed = {int(row[0]): parse_vector_text(str(row[1])) for row in cur.fetchall()}
    missing = sorted(set(wanted) - set(observed))
    if missing:
        raise SystemExit(
            "PostgreSQL/fbin mapping or query-validity check is missing query IDs: "
            f"{missing[:10]}"
        )
    max_abs_error = 0.0
    minimum_fbin_norm_sq = float("inf")
    for query_id in wanted:
        database_vector = observed[query_id]
        file_vector = np.asarray(xb[query_id], dtype=np.float32)
        if database_vector.shape != file_vector.shape:
            raise SystemExit(
                f"PostgreSQL/fbin dimension mismatch at id={query_id}: "
                f"database={database_vector.shape} fbin={file_vector.shape}"
            )
        error = float(np.max(np.abs(database_vector - file_vector)))
        max_abs_error = max(max_abs_error, error)
        if not np.allclose(
            database_vector,
            file_vector,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ):
            raise SystemExit(
                f"PostgreSQL/fbin vector mismatch at id={query_id}: max_abs_error={error}"
            )
        norm_sq = float(np.dot(file_vector, file_vector))
        if not np.isfinite(norm_sq) or norm_sq <= 0.0:
            raise SystemExit(f"query vector must be finite and nonzero at id={query_id}: norm_sq={norm_sq}")
        minimum_fbin_norm_sq = min(minimum_fbin_norm_sq, norm_sq)
    return {
        "checked_query_rows": len(wanted),
        "comparison": "float32_allclose",
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "max_abs_error": max_abs_error,
        "all_fbin_query_vectors_nonzero": True,
        "minimum_fbin_norm_sq": minimum_fbin_norm_sq if wanted else None,
        "embedding_valid_claimed": predicate_declares_embedding_valid(query_validity_predicate),
        "query_validity_predicate": query_validity_predicate,
    }


def truth_boundary(distances: list[float], k: int) -> tuple[float, float, int, bool]:
    if len(distances) < k:
        raise ValueError(f"exact result has {len(distances)} distances, expected at least {k}")
    kth = float(distances[k - 1])
    tolerance = distance_tolerance(kth)
    strict = sum(value < kth - tolerance for value in distances[:k])
    boundary_tied = len(distances) > k and distances[k] <= kth + tolerance
    return kth, tolerance, strict, boundary_tied


def truth_row(
    query_no: int,
    query_id: int,
    filter_spec: dict[str, str],
    actual_selectivity: float,
    candidate_count: int,
    exact_ids: list[int],
    exact_distances: list[float],
    amortized_ms: float,
    seed: int,
    split: str,
    self_in_filter: bool,
    k: int,
    candidate_validity_predicate: str,
    query_validity_predicate: str,
    eligible_query_population: int,
    eligible_query_ids_sha256: str,
    eligible_query_population_provenance: str,
    candidate_validity_provenance: str = "unspecified",
    query_validity_provenance: str = "unspecified",
    candidate_ids_sha256: str = "",
) -> dict[str, Any]:
    top_ids = exact_ids[:k]
    top_distances = exact_distances[:k]
    ids = ",".join(str(value) for value in top_ids)
    kth, tolerance, strict, boundary_tied = truth_boundary(exact_distances, k)
    return {
        "query_no": query_no,
        "query_id": query_id,
        "filter_name": filter_spec["filter_name"],
        "target_rate": filter_spec["target_rate"],
        "predicate": filter_spec["predicate"],
        "candidate_validity_predicate": candidate_validity_predicate,
        "candidate_validity_provenance": candidate_validity_provenance,
        "query_validity_predicate": query_validity_predicate,
        "query_validity_provenance": query_validity_provenance,
        "eligible_query_population": eligible_query_population,
        "eligible_query_ids_sha256": eligible_query_ids_sha256,
        "eligible_query_population_provenance": eligible_query_population_provenance,
        "actual_selectivity": actual_selectivity,
        "method": "pre_filter_exact",
        "k": k,
        "post_overfetch": "",
        "post_ef_search": "",
        "in_ef_search": "",
        "latency_ms": amortized_ms,
        "recall_at_10_exact_filtered": 1.0,
        "returned": len(top_ids),
        "candidates": candidate_count,
        "filtered_rows": candidate_count,
        "search_candidate_rows": candidate_count - int(self_in_filter),
        "result_ids": ids,
        "exact_filtered_topk_ids": ids,
        "exact_filtered_topk_distances_sq": ",".join(f"{value:.9g}" for value in top_distances),
        "kth_distance_sq": f"{kth:.9g}",
        "tie_tolerance": f"{tolerance:.9g}",
        "strict_closer_count": strict,
        "boundary_tied": boundary_tied,
        "self_excluded": True,
        "candidate_rows": candidate_count,
        "candidate_ids_sha256": candidate_ids_sha256,
        "self_excluded_rows": int(self_in_filter),
        "query_split": split,
        "query_seed": seed,
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["filter_name"]), str(row.get("query_split", "calibration"))), []).append(row)
    summary: list[dict[str, Any]] = []
    for (filter_name, split), items in grouped.items():
        summary.append(
            {
                "filter_name": filter_name,
                "query_split": split,
                "queries": len(items),
                "target_rate": items[0]["target_rate"],
                "predicate": items[0]["predicate"],
                "actual_selectivity": statistics.median(float(row["actual_selectivity"]) for row in items),
                "candidate_count": int(float(items[0]["candidates"])),
                "boundary_tied_queries": sum(str(row["boundary_tied"]).lower() == "true" for row in items),
                "zero_kth_distance_queries": sum(float(row["kth_distance_sq"]) == 0.0 for row in items),
                "exact_latency_amortized_mean_ms": statistics.fmean(float(row["latency_ms"]) for row in items),
            }
        )
    write_csv(path.with_name(path.stem + "_summary.csv"), summary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def filter_contract(filters: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "filter_name": row["filter_name"],
            "target_rate": row.get("target_rate", ""),
            "predicate": row["predicate"],
        }
        for row in filters
    ]


def truth_manifest_path_for(path: Path) -> Path:
    return path.with_name(path.stem + "_manifest.json")


def _require_sha256_reference(reference: Any, path: Path, label: str) -> None:
    if not isinstance(reference, dict):
        raise SystemExit(f"{label} reference is missing or malformed")
    observed_path = Path(str(reference.get("path", ""))).expanduser()
    if observed_path.resolve() != path.resolve():
        raise SystemExit(
            f"{label} path mismatch: expected={path.resolve()} observed={observed_path}"
        )
    if reference.get("sha256") != sha256_file(path):
        raise SystemExit(f"{label} sha256 mismatch: {path}")


def _strict_truth_schema_fields() -> set[str]:
    return {
        "query_no",
        "query_id",
        "filter_name",
        "target_rate",
        "predicate",
        "candidate_validity_predicate",
        "query_validity_predicate",
        "k",
        "filtered_rows",
        "search_candidate_rows",
        "candidate_rows",
        "candidate_ids_sha256",
        "result_ids",
        "exact_filtered_topk_ids",
        "exact_filtered_topk_distances_sq",
        "kth_distance_sq",
        "tie_tolerance",
        "strict_closer_count",
        "boundary_tied",
        "self_excluded",
        "self_excluded_rows",
    }


def validate_strict_truth_rows(
    rows: list[dict[str, str]],
    query_ids: np.ndarray,
    filters: list[dict[str, str]],
    *,
    k: int,
    candidate_validity_predicate: str,
    query_validity_predicate: str,
    expected_candidate_populations: dict[str, dict[str, Any]],
    allow_partial_filters: set[str] | None = None,
    prefix_query_count: int | None = None,
) -> None:
    """Validate identity, candidate binding, and tie-aware truth fields.

    Extension checkpoints may contain the copied prefix for filters whose new
    rows have not been computed yet. Such filters are explicitly listed in
    ``allow_partial_filters``; ordinary truth artifacts never use this escape.
    """
    expected_ids = {position: int(query_id) for position, query_id in enumerate(query_ids)}
    expected_filters = {row["filter_name"] for row in filters}
    observed_filters = {row.get("filter_name", "") for row in rows}
    unexpected = observed_filters - expected_filters
    if unexpected:
        raise SystemExit(f"truth contains unexpected filters: {sorted(unexpected)}")
    partial = allow_partial_filters or set()
    if not partial.issubset(expected_filters):
        raise SystemExit("partial truth filter set is incompatible with filter contract")
    for filter_spec in filters:
        filter_name = filter_spec["filter_name"]
        items = [row for row in rows if row.get("filter_name") == filter_name]
        expected_count = len(query_ids)
        if filter_name in partial:
            if prefix_query_count is None:
                raise SystemExit("partial truth validation requires a prefix query count")
            expected_count = prefix_query_count
        if len(items) != expected_count:
            raise SystemExit(
                f"truth filter={filter_name} row count mismatch: "
                f"expected={expected_count} observed={len(items)}"
            )
        seen_query_nos: set[int] = set()
        observed_candidate: set[tuple[int, str]] = set()
        expected_population = expected_candidate_populations.get(filter_name)
        if expected_population is None:
            raise SystemExit(f"truth candidate population is missing for filter={filter_name}")
        for row in items:
            missing = _strict_truth_schema_fields() - set(row)
            if missing:
                raise SystemExit(
                    f"truth filter={filter_name} is missing tie-aware fields: {sorted(missing)}"
                )
            try:
                query_no = int(row["query_no"])
                query_id = int(row["query_id"])
                observed_k = int(row["k"])
                candidate_rows = int(row["candidate_rows"])
                filtered_rows = int(row["filtered_rows"])
                search_rows = int(row["search_candidate_rows"])
                self_rows = int(row["self_excluded_rows"])
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"truth filter={filter_name} has invalid typed fields") from exc
            if query_no in seen_query_nos or query_no not in expected_ids:
                raise SystemExit(f"truth filter={filter_name} query_no values are invalid")
            if expected_ids[query_no] != query_id:
                raise SystemExit(f"truth query mapping mismatch for filter={filter_name} q={query_no}")
            if observed_k != k:
                raise SystemExit(f"truth k mismatch for filter={filter_name} q={query_no}")
            if (
                row["candidate_validity_predicate"] != candidate_validity_predicate
                or row["query_validity_predicate"] != query_validity_predicate
            ):
                raise SystemExit(f"truth validity contract mismatch for filter={filter_name} q={query_no}")
            if (
                filtered_rows != candidate_rows
                or search_rows != filtered_rows - self_rows
                or str(row["self_excluded"]).lower() != "true"
                or self_rows not in (0, 1)
            ):
                raise SystemExit(f"truth candidate/self-exclusion contract mismatch for filter={filter_name}")
            candidate_hash = str(row["candidate_ids_sha256"])
            if candidate_rows != int(expected_population["candidate_rows"]):
                raise SystemExit(f"truth candidate row count mismatch for filter={filter_name}")
            if candidate_hash != str(expected_population["candidate_ids_sha256"]):
                raise SystemExit(f"truth candidate hash mismatch for filter={filter_name}")
            seen_query_nos.add(query_no)
            observed_candidate.add((candidate_rows, candidate_hash))
        expected_query_nos = set(range(expected_count))
        if seen_query_nos != expected_query_nos:
            raise SystemExit(f"truth filter={filter_name} query_no values are incomplete")
        if len(observed_candidate) != 1:
            raise SystemExit(f"truth filter={filter_name} candidate population is inconsistent")


def validate_candidate_population_bindings(
    rows: list[dict[str, str]],
    completed_filters: dict[str, Any],
    filters: list[dict[str, str]],
) -> None:
    for filter_spec in filters:
        name = filter_spec["filter_name"]
        record = completed_filters.get(name)
        if not isinstance(record, dict):
            raise SystemExit(f"checkpoint candidate population is missing for filter={name}")
        items = [row for row in rows if row.get("filter_name") == name]
        pairs = {
            (int(row["candidate_rows"]), str(row["candidate_ids_sha256"]))
            for row in items
        }
        expected = (int(record.get("candidate_rows", -1)), str(record.get("candidate_ids_sha256", "")))
        if pairs != {expected}:
            raise SystemExit(f"checkpoint/truth candidate hash binding failed for filter={name}")


def validate_copied_prefix_rows(
    base_rows: list[dict[str, str]],
    extended_rows: list[dict[str, str]],
    base_query_count: int,
) -> None:
    base_by_key = {
        (str(row.get("filter_name")), int(row["query_no"])): row
        for row in base_rows
    }
    extended_by_key = {
        (str(row.get("filter_name")), int(row["query_no"])): row
        for row in extended_rows
        if int(row["query_no"]) < base_query_count
    }
    if set(extended_by_key) != set(base_by_key):
        raise SystemExit("strict prefix output is missing or adding copied prefix rows")
    for key, row in base_by_key.items():
        if extended_by_key[key] != row:
            raise SystemExit(f"strict prefix copied row changed for filter={key[0]} q={key[1]}")


def validate_prefix_query_cohort(
    prior_query_ids: np.ndarray,
    extended_query_ids: np.ndarray,
    *,
    required_added_queries: int = 100,
) -> np.ndarray:
    prior = np.asarray(prior_query_ids, dtype=np.int64)
    extended = np.asarray(extended_query_ids, dtype=np.int64)
    if extended.size != prior.size + required_added_queries:
        raise SystemExit(
            "strict prefix extension requires exactly "
            f"{required_added_queries} new query rows: old={prior.size} new={extended.size}"
        )
    if not np.array_equal(extended[: prior.size], prior):
        raise SystemExit("external query cohort is not a strict query-ID prefix extension")
    added = extended[prior.size :]
    if len(set(int(value) for value in added)) != required_added_queries:
        raise SystemExit("strict prefix extension query IDs are not unique")
    if set(int(value) for value in added) & set(int(value) for value in prior):
        raise SystemExit("strict prefix extension introduces a query-ID overlap")
    return added


def validate_stable_cohort_contract(
    prior: dict[str, Any],
    extended: dict[str, Any],
) -> None:
    stable_fields = (
        "eligible_query_population",
        "candidate_universe_rows",
    )
    for field in stable_fields:
        if prior.get(field) != extended.get(field):
            raise SystemExit(f"query cohort contract changed: {field}")
    prior_source = prior.get("source_contract", {})
    extended_source = extended.get("source_contract", {})
    if prior_source.get("method") != extended_source.get("method"):
        raise SystemExit("query cohort method changed during strict prefix extension")
    if prior_source.get("uniqueness_contract") != extended_source.get("uniqueness_contract"):
        raise SystemExit("query cohort uniqueness contract changed during strict prefix extension")


def truth_manifest_eligible_population(manifest: dict[str, Any]) -> dict[str, Any] | None:
    population = manifest.get("eligible_query_population")
    if not isinstance(population, dict):
        return None
    return {
        "rows": population.get("eligible_rows"),
        "ids_sha256": population.get("eligible_ids_sha256"),
        "provenance": population.get("provenance"),
    }


def truth_manifest_postgres_identity(manifest: dict[str, Any]) -> dict[str, Any] | None:
    inputs = manifest.get("inputs")
    postgres = inputs.get("postgres") if isinstance(inputs, dict) else None
    if not isinstance(postgres, dict):
        return None
    fields = ("table", "rows", "min_id", "max_id", "table_oid", "table_relfilenode")
    return {field: postgres.get(field) for field in fields}


def validate_prefix_truth_contract(
    prior_rows: list[dict[str, str]],
    prior_checkpoint: dict[str, Any],
    prior_manifest: dict[str, Any],
    prior_query_ids: np.ndarray,
    extended_query_ids: np.ndarray,
    filters: list[dict[str, str]],
    *,
    k: int,
    fbin: dict[str, str],
    filters_csv: dict[str, str],
    table_identity: dict[str, Any],
    eligible_population: dict[str, Any],
    candidate_validity_predicate: str,
    query_validity_predicate: str,
    candidate_validity_provenance: str,
    query_validity_provenance: str,
    prior_query_source: dict[str, Any],
    extended_query_source: dict[str, Any],
    check_prefix: bool = True,
) -> np.ndarray:
    """Validate an immutable completed artifact before any extension work."""
    if truth_manifest_eligible_population(prior_manifest) != eligible_population:
        raise SystemExit("extend-from truth eligible query population changed")
    if prior_manifest.get("artifact_valid") is not True:
        raise SystemExit("extend-from truth artifact is not marked artifact_valid")
    if prior_manifest.get("method") != "exact_filtered_l2_tie_aware":
        raise SystemExit("extend-from truth uses an unsupported exact-truth method")
    if prior_manifest.get("rows") != int(table_identity["rows"]):
        raise SystemExit("extend-from truth table/vector row contract changed")
    if prior_manifest.get("k") != k:
        raise SystemExit("extend-from truth k contract changed")
    if prior_manifest.get("filters") != len(filters) or prior_manifest.get("truth_rows") != len(prior_rows):
        raise SystemExit("extend-from truth filter/truth row contract is incomplete")
    validity = prior_manifest.get("validity_contract")
    if not isinstance(validity, dict) or (
        validity.get("candidate_validity_predicate") != candidate_validity_predicate
        or validity.get("query_validity_predicate") != query_validity_predicate
        or validity.get("candidate_validity_provenance") != candidate_validity_provenance
        or validity.get("query_validity_provenance") != query_validity_provenance
    ):
        raise SystemExit("extend-from truth validity contract changed")
    inputs = prior_manifest.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("fbin") != fbin:
        raise SystemExit("extend-from truth fbin contract changed")
    expected_postgres = {
        "table": table_identity["name"],
        "rows": table_identity["rows"],
        "min_id": table_identity["min_id"],
        "max_id": table_identity["max_id"],
        "table_oid": table_identity["oid"],
        "table_relfilenode": table_identity["relfilenode"],
    }
    if truth_manifest_postgres_identity(prior_manifest) != expected_postgres:
        raise SystemExit("extend-from truth PostgreSQL table contract changed")
    if prior_checkpoint.get("filters") != filter_contract(filters):
        raise SystemExit("extend-from checkpoint filter contract changed")
    prior_population = prior_checkpoint.get("eligible_query_population")
    if prior_population != eligible_population:
        raise SystemExit("extend-from checkpoint eligible query population changed")
    validate_resume_checkpoint(
        prior_checkpoint,
        prior_rows,
        prior_query_ids,
        filters,
        k=k,
        fbin=fbin,
        filters_csv=filters_csv,
        table_identity=table_identity,
        eligible_population=eligible_population,
        candidate_validity_predicate=candidate_validity_predicate,
        query_validity_predicate=query_validity_predicate,
        candidate_validity_provenance=candidate_validity_provenance,
        query_validity_provenance=query_validity_provenance,
        query_source=prior_query_source,
    )
    completed = prior_checkpoint.get("completed_filters")
    if not isinstance(completed, dict) or set(completed) != {row["filter_name"] for row in filters}:
        raise SystemExit("extend-from checkpoint has partial or unexpected filters")
    validate_strict_truth_rows(
        prior_rows,
        prior_query_ids,
        filters,
        k=k,
        candidate_validity_predicate=candidate_validity_predicate,
        query_validity_predicate=query_validity_predicate,
        expected_candidate_populations=completed,
    )
    validate_candidate_population_bindings(prior_rows, completed, filters)
    expected_manifest_source = prior_manifest.get("query_source")
    if expected_manifest_source != prior_query_source:
        raise SystemExit("extend-from truth query cohort provenance changed")
    if prior_manifest.get("query_ids_disjoint") is not True:
        raise SystemExit("extend-from truth is not disjoint/tie-aware")
    if check_prefix:
        return validate_prefix_query_cohort(prior_query_ids, extended_query_ids)
    return np.empty(0, dtype=np.int64)


def load_extension_base(
    truth_path: Path,
    checkpoint_path: Path,
    *,
    current_fbin: dict[str, str],
    current_filters_csv: dict[str, str],
    table_identity: dict[str, Any],
    filters: list[dict[str, str]],
    candidate_validity_predicate: str,
    query_validity_predicate: str,
    candidate_validity_provenance: str,
    query_validity_provenance: str,
    k: int,
) -> dict[str, Any]:
    if not truth_path.exists() or not checkpoint_path.exists():
        raise SystemExit("strict prefix extension requires an existing truth and checkpoint")
    manifest_path = truth_manifest_path_for(truth_path)
    if not manifest_path.exists():
        raise SystemExit(f"extend-from truth manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"extend-from artifact is unreadable: {exc}") from exc
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise SystemExit("extend-from truth outputs are missing")
    _require_sha256_reference(outputs.get("truth_csv"), truth_path, "extend-from truth")
    _require_sha256_reference(outputs.get("checkpoint"), checkpoint_path, "extend-from checkpoint")
    source = manifest.get("query_source")
    if not isinstance(source, dict):
        raise SystemExit("extend-from truth query cohort provenance is missing")
    cohort_ref = source.get("cohort_csv")
    manifest_ref = source.get("manifest")
    if not isinstance(cohort_ref, dict) or not isinstance(manifest_ref, dict):
        raise SystemExit("extend-from truth query cohort references are missing")
    old_cohort_path = Path(str(cohort_ref.get("path", ""))).expanduser()
    old_cohort_manifest_path = Path(str(manifest_ref.get("path", ""))).expanduser()
    if not old_cohort_path.exists() or not old_cohort_manifest_path.exists():
        raise SystemExit("extend-from truth query cohort files are missing")
    if cohort_ref.get("sha256") != sha256_file(old_cohort_path):
        raise SystemExit("extend-from truth cohort sha256 mismatch")
    if manifest_ref.get("sha256") != sha256_file(old_cohort_manifest_path):
        raise SystemExit("extend-from truth cohort manifest sha256 mismatch")
    calibration = manifest.get("calibration")
    final = manifest.get("final")
    if not isinstance(calibration, dict) or not isinstance(final, dict):
        raise SystemExit("extend-from truth calibration/final selection is missing")
    old_calibration = int(calibration.get("queries", -1))
    old_final = int(final.get("queries", -1))
    old_query_count = old_calibration + old_final
    if old_calibration < 0 or old_final < 0 or old_query_count <= 0:
        raise SystemExit(
            "strict prefix extension requires a positive, complete base query cohort"
        )
    old_cohort = load_external_query_cohort(
        old_cohort_path,
        old_cohort_manifest_path,
        calibration_queries=old_calibration,
        calibration_seed=int(calibration.get("seed", -1)),
        final_queries=old_final,
        final_seed=int(final.get("seed", -1)),
        candidate_validity_predicate=candidate_validity_predicate,
        query_validity_predicate=query_validity_predicate,
        fbin_contract=current_fbin,
        vector_rows=int(table_identity["rows"]),
        table_identity=table_identity,
    )
    prior_rows = read_csv(truth_path)
    validate_prefix_truth_contract(
        prior_rows,
        checkpoint,
        manifest,
        old_cohort["query_ids"],
        old_cohort["query_ids"],
        filters,
        k=k,
        fbin=current_fbin,
        filters_csv=current_filters_csv,
        table_identity=table_identity,
        eligible_population=old_cohort["eligible_query_population"],
        candidate_validity_predicate=candidate_validity_predicate,
        query_validity_predicate=query_validity_predicate,
        candidate_validity_provenance=candidate_validity_provenance,
        query_validity_provenance=query_validity_provenance,
        prior_query_source=old_cohort["source_contract"],
        extended_query_source=old_cohort["source_contract"],
        check_prefix=False,
    )
    return {
        "manifest": manifest,
        "checkpoint": checkpoint,
        "rows": prior_rows,
        "query_ids": old_cohort["query_ids"],
        "cohort": old_cohort,
        "source": old_cohort["source_contract"],
        "old_cohort_path": old_cohort_path,
        "old_cohort_manifest_path": old_cohort_manifest_path,
        "old_query_count": old_query_count,
    }


def extension_checkpoint_payload(
    *,
    k: int,
    query_ids: np.ndarray,
    candidate_validity_predicate: str,
    candidate_validity_provenance: str,
    query_validity_predicate: str,
    query_validity_provenance: str,
    fbin: dict[str, str],
    filters_csv: dict[str, str],
    filters: list[dict[str, str]],
    table: dict[str, Any],
    eligible_population: dict[str, Any],
    query_source: dict[str, Any],
    candidate_populations: dict[str, Any],
    completed_filters: set[str],
    base_truth: Path,
    base_checkpoint: Path,
    base_query_ids: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "operation": "strict_prefix_extension",
        "k": k,
        "query_count": len(query_ids),
        "query_ids_sha256": ordered_ids_sha256(query_ids),
        "candidate_validity_predicate": candidate_validity_predicate,
        "candidate_validity_provenance": candidate_validity_provenance,
        "query_validity_predicate": query_validity_predicate,
        "query_validity_provenance": query_validity_provenance,
        "fbin": fbin,
        "filters_csv": filters_csv,
        "filters": filter_contract(filters),
        "table": table,
        "eligible_query_population": eligible_population,
        "query_source": query_source,
        "candidate_populations": candidate_populations,
        "completed_filters": {
            name: candidate_populations[name]
            for name in sorted(completed_filters)
        },
        "extension": {
            "extension_version": STRICT_PREFIX_EXTENSION_VERSION,
            "base_truth": {
                "path": str(base_truth.resolve()),
                "sha256": sha256_file(base_truth),
            },
            "base_checkpoint": {
                "path": str(base_checkpoint.resolve()),
                "sha256": sha256_file(base_checkpoint),
            },
            "base_query_count": len(base_query_ids),
            "base_query_ids_sha256": ordered_ids_sha256(base_query_ids),
            "added_query_count": len(query_ids) - len(base_query_ids),
            "added_query_ids_sha256": ordered_ids_sha256(query_ids[len(base_query_ids) :]),
        },
    }


def validate_extension_checkpoint(
    checkpoint: dict[str, Any],
    rows: list[dict[str, str]],
    query_ids: np.ndarray,
    filters: list[dict[str, str]],
    *,
    base_query_count: int,
    k: int,
    fbin: dict[str, str],
    filters_csv: dict[str, str],
    table_identity: dict[str, Any],
    eligible_population: dict[str, Any],
    candidate_validity_predicate: str,
    query_validity_predicate: str,
    candidate_validity_provenance: str,
    query_validity_provenance: str,
    query_source: dict[str, Any],
    base_truth: Path,
    base_checkpoint: Path,
    base_query_ids: np.ndarray,
    base_rows: list[dict[str, str]],
) -> set[str]:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit("extension checkpoint schema is missing or unsupported")
    if checkpoint.get("operation") != "strict_prefix_extension":
        raise SystemExit("checkpoint is not a strict prefix extension checkpoint")
    expected = {
        "k": k,
        "query_count": len(query_ids),
        "query_ids_sha256": ordered_ids_sha256(query_ids),
        "candidate_validity_predicate": candidate_validity_predicate,
        "candidate_validity_provenance": candidate_validity_provenance,
        "query_validity_predicate": query_validity_predicate,
        "query_validity_provenance": query_validity_provenance,
        "fbin": fbin,
        "filters_csv": filters_csv,
        "filters": filter_contract(filters),
        "table": table_identity,
        "eligible_query_population": eligible_population,
        "query_source": query_source,
    }
    mismatched = {
        field: {"expected": value, "observed": checkpoint.get(field)}
        for field, value in expected.items()
        if checkpoint.get(field) != value
    }
    extension = checkpoint.get("extension")
    if not isinstance(extension, dict) or extension.get("extension_version") != STRICT_PREFIX_EXTENSION_VERSION:
        mismatched["extension"] = {"expected": "strict_prefix_extension", "observed": extension}
    else:
        expected_extension = {
            "base_query_count": base_query_count,
            "base_query_ids_sha256": ordered_ids_sha256(base_query_ids),
            "added_query_count": len(query_ids) - base_query_count,
            "added_query_ids_sha256": ordered_ids_sha256(query_ids[base_query_count:]),
        }
        for field, value in expected_extension.items():
            if extension.get(field) != value:
                mismatched[f"extension.{field}"] = {"expected": value, "observed": extension.get(field)}
        for field, path in (("base_truth", base_truth), ("base_checkpoint", base_checkpoint)):
            _require_sha256_reference(extension.get(field), path, f"extension.{field}")
    populations = checkpoint.get("candidate_populations")
    if not isinstance(populations, dict):
        raise SystemExit("extension checkpoint candidate populations are missing")
    expected_names = {row["filter_name"] for row in filters}
    if set(populations) != expected_names:
        raise SystemExit("extension checkpoint candidate filter set is incomplete")
    if any(
        not isinstance(record, dict)
        or set(record) != {"candidate_rows", "candidate_ids_sha256"}
        or not isinstance(record["candidate_rows"], int)
        or not isinstance(record["candidate_ids_sha256"], str)
        for record in populations.values()
    ):
        raise SystemExit("extension checkpoint candidate population record is malformed")
    completed = checkpoint.get("completed_filters")
    if not isinstance(completed, dict) or not set(completed).issubset(expected_names):
        raise SystemExit("extension checkpoint completed filter set is invalid")
    if any(completed[name] != populations[name] for name in completed):
        raise SystemExit("extension checkpoint completed candidate binding changed")
    validate_strict_truth_rows(
        rows,
        query_ids,
        filters,
        k=k,
        candidate_validity_predicate=candidate_validity_predicate,
        query_validity_predicate=query_validity_predicate,
        expected_candidate_populations=populations,
        allow_partial_filters=expected_names - set(completed),
        prefix_query_count=base_query_count,
    )
    validate_copied_prefix_rows(base_rows, rows, base_query_count)
    return set(completed)


def validate_resumed_rows(
    rows: list[dict[str, str]],
    query_ids: np.ndarray,
    filters: list[dict[str, str]],
    candidate_validity_predicate: str = DEFAULT_VALIDITY_PREDICATE,
    query_validity_predicate: str = DEFAULT_VALIDITY_PREDICATE,
    eligible_query_population: int | None = None,
    eligible_query_ids_sha256: str | None = None,
    eligible_query_population_provenance: str | None = None,
    candidate_validity_provenance: str | None = None,
    query_validity_provenance: str | None = None,
    expected_k: int | None = None,
) -> set[str]:
    expected_queries = {position: int(query_id) for position, query_id in enumerate(query_ids)}
    expected_filters = {row["filter_name"] for row in filters}
    observed_filters = {row.get("filter_name", "") for row in rows}
    unexpected_filters = observed_filters - expected_filters
    if unexpected_filters:
        raise SystemExit(f"resume contains unexpected filters: {sorted(unexpected_filters)}")
    completed: set[str] = set()
    for filter_name in expected_filters:
        items = [row for row in rows if row.get("filter_name") == filter_name]
        if not items:
            continue
        if len(items) != len(query_ids):
            raise SystemExit(f"resume filter={filter_name} is partial: rows={len(items)}")
        query_nos: list[int] = []
        for row in items:
            try:
                query_no = int(row["query_no"])
                query_id = int(row["query_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"resume filter={filter_name} has invalid query identity") from exc
            query_nos.append(query_no)
        if len(set(query_nos)) != len(query_nos) or set(query_nos) != set(expected_queries):
            raise SystemExit(f"resume filter={filter_name} query_no values are not unique and complete")
        for row in items:
            query_no = int(row["query_no"])
            if expected_queries.get(query_no) != int(row["query_id"]):
                raise SystemExit(f"resume query mapping mismatch for filter={filter_name} q={query_no}")
        filter_spec = next(row for row in filters if row["filter_name"] == filter_name)
        for row in items:
            if "predicate" in filter_spec and row.get("predicate") != filter_spec["predicate"]:
                raise SystemExit(f"resume predicate mismatch for filter={filter_name}")
            if "target_rate" in filter_spec and row.get("target_rate") != filter_spec["target_rate"]:
                raise SystemExit(f"resume filter specification mismatch for filter={filter_name}")
            required = {"filtered_rows", "kth_distance_sq", "tie_tolerance", "self_excluded"}
            if not required.issubset(row) or str(row["self_excluded"]).lower() != "true":
                raise SystemExit("resume artifact uses the retired non-tie-aware truth schema")
            if expected_k is not None:
                try:
                    observed_k = int(row["k"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise SystemExit(f"resume k is missing or invalid for filter={filter_name}") from exc
                if observed_k != expected_k:
                    raise SystemExit(f"resume k mismatch for filter={filter_name}")
            expected_contract = {
                "candidate_validity_predicate": candidate_validity_predicate,
                "query_validity_predicate": query_validity_predicate,
            }
            if candidate_validity_provenance is not None:
                expected_contract["candidate_validity_provenance"] = candidate_validity_provenance
            if query_validity_provenance is not None:
                expected_contract["query_validity_provenance"] = query_validity_provenance
            if eligible_query_population is not None:
                expected_contract["eligible_query_population"] = str(eligible_query_population)
            if eligible_query_ids_sha256 is not None:
                expected_contract["eligible_query_ids_sha256"] = eligible_query_ids_sha256
            if eligible_query_population_provenance is not None:
                expected_contract["eligible_query_population_provenance"] = (
                    eligible_query_population_provenance
                )
            mismatched = {
                field: {"expected": expected, "observed": row.get(field)}
                for field, expected in expected_contract.items()
                if str(row.get(field)) != str(expected)
            }
            if mismatched:
                raise SystemExit(
                    f"resume validity contract mismatch for filter={filter_name} q={query_no}: "
                    f"{json.dumps(mismatched, sort_keys=True)}"
                )
            if "search_candidate_rows" in row and "filtered_rows" in row:
                try:
                    expected_search_rows = int(row["filtered_rows"]) - int(row.get("self_excluded_rows", "0"))
                    observed_search_rows = int(row["search_candidate_rows"])
                except (TypeError, ValueError) as exc:
                    raise SystemExit(f"resume candidate/search population is invalid for filter={filter_name}") from exc
                if observed_search_rows != expected_search_rows:
                    raise SystemExit(f"resume candidate/search population mismatch for filter={filter_name}")
        completed.add(filter_name)
    return completed


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    rows: list[dict[str, str]],
    query_ids: np.ndarray,
    filters: list[dict[str, str]],
    *,
    k: int,
    fbin: dict[str, str],
    filters_csv: dict[str, str],
    table_identity: dict[str, Any],
    eligible_population: dict[str, Any],
    candidate_validity_predicate: str,
    query_validity_predicate: str,
    candidate_validity_provenance: str,
    query_validity_provenance: str,
    query_source: dict[str, Any] | None = None,
) -> set[str]:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit("resume checkpoint schema is missing or unsupported")
    checkpoint_eligible = checkpoint.get("eligible_query_population")
    if not isinstance(checkpoint_eligible, dict) or not {"rows", "ids_sha256", "provenance"}.issubset(
        checkpoint_eligible
    ):
        raise SystemExit("resume checkpoint eligible population is missing or malformed")
    completed = validate_resumed_rows(
        rows,
        query_ids,
        filters,
        candidate_validity_predicate,
        query_validity_predicate,
        int(eligible_population["rows"]),
        eligible_population["ids_sha256"],
        eligible_population["provenance"],
        candidate_validity_provenance,
        query_validity_provenance,
        k,
    )
    expected_scalars = {
        "k": k,
        "query_count": len(query_ids),
        "query_ids_sha256": ordered_ids_sha256(query_ids),
        "candidate_validity_predicate": candidate_validity_predicate,
        "query_validity_predicate": query_validity_predicate,
        "candidate_validity_provenance": candidate_validity_provenance,
        "query_validity_provenance": query_validity_provenance,
    }
    mismatched = {
        field: {"expected": expected, "observed": checkpoint.get(field)}
        for field, expected in expected_scalars.items()
        if checkpoint.get(field) != expected
    }
    expected_filters = filter_contract(filters)
    if checkpoint.get("filters") != expected_filters:
        mismatched["filters"] = {"expected": expected_filters, "observed": checkpoint.get("filters")}
    for field, expected in (
        ("fbin", fbin),
        ("filters_csv", filters_csv),
        ("table", table_identity),
        ("eligible_query_population", eligible_population),
    ):
        if checkpoint.get(field) != expected:
            mismatched[field] = {"expected": expected, "observed": checkpoint.get(field)}
    if query_source is not None and checkpoint.get("query_source") != query_source:
        mismatched["query_source"] = {
            "expected": query_source,
            "observed": checkpoint.get("query_source"),
        }
    checkpoint_completed = checkpoint.get("completed_filters")
    if not isinstance(checkpoint_completed, dict) or set(checkpoint_completed) != completed:
        mismatched["completed_filters"] = {
            "expected": sorted(completed),
            "observed": sorted(checkpoint_completed) if isinstance(checkpoint_completed, dict) else checkpoint_completed,
        }
    elif any(
        not isinstance(record, dict)
        or set(record) != {"candidate_rows", "candidate_ids_sha256"}
        or not isinstance(record["candidate_rows"], int)
        for record in checkpoint_completed.values()
    ):
        mismatched["completed_filters"] = {"expected": "candidate count/hash records", "observed": checkpoint_completed}
    if mismatched:
        raise SystemExit(f"resume checkpoint contract mismatch: {json.dumps(mismatched, sort_keys=True)}")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tie-aware exact filtered L2 truth for Amazon-10M.")
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--table", default="amazon_grocery_reviews_10m_pgvector")
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--calibration-queries", type=int, default=100)
    parser.add_argument("--calibration-seed", type=int, default=57)
    parser.add_argument("--final-queries", type=int, default=100)
    parser.add_argument("--final-seed", type=int, default=58)
    parser.add_argument(
        "--query-cohort-csv",
        type=Path,
        help="Optional preregistered dedicated query cohort; disables in-script query sampling.",
    )
    parser.add_argument(
        "--query-cohort-manifest",
        type=Path,
        help="Required provenance manifest for --query-cohort-csv.",
    )
    parser.add_argument(
        "--extend-from-truth",
        type=Path,
        help=(
            "Create a larger truth artifact by copying the immutable complete prefix "
            "from this truth CSV and computing only the newly appended query suffix."
        ),
    )
    parser.add_argument(
        "--extend-from-checkpoint",
        type=Path,
        help="Checkpoint paired with --extend-from-truth; it is never modified.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=50_000)
    parser.add_argument(
        "--distance-workers", type=int, default=1,
        help="bounded candidate-chunk workers; results are merged in candidate-ID order",
    )
    parser.add_argument("--progress-chunks", type=int, default=20)
    parser.add_argument(
        "--candidate-validity-predicate",
        type=safe_sql_predicate,
        default=None,
        help="Global SQL predicate; formal default is embedding_valid. TRUE is allowed only as an explicit override.",
    )
    parser.add_argument(
        "--query-validity-predicate",
        type=safe_sql_predicate,
        default=None,
        help="Eligible-query predicate; defaults to --candidate-validity-predicate.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be positive")
    if args.chunk_rows <= 0 or args.distance_workers <= 0:
        parser.error("--chunk-rows and --distance-workers must be positive")
    if args.calibration_queries < 0 or args.final_queries < 0:
        parser.error("query counts must not be negative")
    if args.calibration_queries + args.final_queries == 0:
        parser.error("at least one query is required")
    if (args.query_cohort_csv is None) != (args.query_cohort_manifest is None):
        parser.error("--query-cohort-csv and --query-cohort-manifest must be provided together")
    extension_requested = args.extend_from_truth is not None or args.extend_from_checkpoint is not None
    if extension_requested and (
        args.extend_from_truth is None or args.extend_from_checkpoint is None
    ):
        parser.error("--extend-from-truth and --extend-from-checkpoint must be provided together")
    if extension_requested and (args.query_cohort_csv is None or args.query_cohort_manifest is None):
        parser.error("strict prefix extension requires the new external query cohort and manifest")
    if extension_requested and args.filter_names:
        parser.error("strict prefix extension does not accept --filter-names; all filters are required")
    if extension_requested and args.out.resolve() == args.extend_from_truth.resolve():
        parser.error("strict prefix extension output must be different from --extend-from-truth")
    if extension_requested and checkpoint_path_for(args.out).resolve() == args.extend_from_checkpoint.resolve():
        parser.error("strict prefix extension checkpoint output must be different from --extend-from-checkpoint")
    xb, vector_rows, _ = read_fbin_memmap(args.fbin, args.rows)
    filters = load_filters(args.filters_csv, set(args.filter_names))
    candidate_validity_predicate, candidate_validity_provenance = resolve_candidate_validity_predicate(
        args.candidate_validity_predicate
    )
    query_validity_predicate = resolve_query_validity_predicate(
        candidate_validity_predicate,
        args.query_validity_predicate,
    )
    query_validity_provenance = (
        "inherits_candidate_validity"
        if args.query_validity_predicate is None
        else "explicit_cli_predicate"
    )
    fbin_contract = {"path": str(args.fbin.resolve()), "sha256": sha256_file(args.fbin)}
    filters_csv_contract = {
        "path": str(args.filters_csv.resolve()),
        "sha256": sha256_file(args.filters_csv),
    }
    checkpoint_path = checkpoint_path_for(args.out)
    rows_out: list[dict[str, Any]] = []
    completed: set[str] = set()
    checkpoint_completed_records: dict[str, Any] = {}
    database_source: dict[str, Any] = {}
    query_population: dict[str, Any] = {}
    extension_base: dict[str, Any] | None = None
    extension_added_query_ids = np.empty(0, dtype=np.int64)
    extension_start = 0
    checkpoint_candidate_records: dict[str, Any] = {}
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT count(*), min(id), max(id), %s::regclass::oid::bigint, "
            f"pg_relation_filenode(%s::regclass)::bigint FROM {args.table}",
            (args.table, args.table),
        )
        table_rows, min_id, max_id, table_oid, table_relfilenode = (
            int(value) for value in cur.fetchone()
        )
        if table_rows != vector_rows:
            raise SystemExit(f"table/vector row mismatch: table={table_rows}, vectors={vector_rows}")
        if (min_id, max_id) != (0, vector_rows - 1):
            raise SystemExit(
                f"table/vector ID-space mismatch: table=({min_id}, {max_id}) "
                f"fbin=(0, {vector_rows - 1})"
            )
        table_identity = {
            "name": args.table,
            "rows": table_rows,
            "min_id": min_id,
            "max_id": max_id,
            "oid": table_oid,
            "relfilenode": table_relfilenode,
        }
        if args.query_cohort_csv is not None:
            external_cohort = load_external_query_cohort(
                args.query_cohort_csv,
                args.query_cohort_manifest,
                calibration_queries=args.calibration_queries,
                calibration_seed=args.calibration_seed,
                final_queries=args.final_queries,
                final_seed=args.final_seed,
                candidate_validity_predicate=candidate_validity_predicate,
                query_validity_predicate=query_validity_predicate,
                fbin_contract=fbin_contract,
                vector_rows=vector_rows,
                table_identity=table_identity,
            )
            query_ids = external_cohort["query_ids"]
            calibration_query_ids = external_cohort["calibration_query_ids"]
            final_query_ids = external_cohort["final_query_ids"]
            eligible_population_contract = external_cohort["eligible_query_population"]
            query_source_contract = external_cohort["source_contract"]
            eligible_sql_ms = 0.0
        else:
            eligible_query_ids, eligible_sql_ms = fetch_eligible_query_ids(
                cur,
                args.table,
                query_validity_predicate,
            )
            if eligible_query_ids.size and (
                int(eligible_query_ids.min()) < 0
                or int(eligible_query_ids.max()) >= vector_rows
            ):
                raise SystemExit("eligible query population has id outside vector row range")
            if eligible_query_ids.size and np.any(
                eligible_query_ids[1:] <= eligible_query_ids[:-1]
            ):
                raise SystemExit("eligible query IDs returned by PostgreSQL are not sorted and unique")
            eligible_query_ids_hash = sorted_ids_sha256(eligible_query_ids)
            eligible_query_population_provenance = (
                f"postgres_ordered_id_scan_v1:{args.table}:oid={table_oid}:"
                f"relfilenode={table_relfilenode}"
            )
            eligible_population_contract = {
                "rows": len(eligible_query_ids),
                "ids_sha256": eligible_query_ids_hash,
                "provenance": eligible_query_population_provenance,
            }
            calibration_query_ids = sample_disjoint_eligible_query_ids(
                eligible_query_ids,
                set(),
                args.calibration_queries,
                args.calibration_seed,
            )
            final_query_ids = sample_disjoint_eligible_query_ids(
                eligible_query_ids,
                set(int(value) for value in calibration_query_ids),
                args.final_queries,
                args.final_seed,
            )
            query_ids = np.concatenate((calibration_query_ids, final_query_ids))
            query_source_contract = {
                "kind": "postgres_validity_population_seeded_sample",
                "eligible_query_population": eligible_population_contract,
            }
        if extension_requested:
            extension_base = load_extension_base(
                args.extend_from_truth,
                args.extend_from_checkpoint,
                current_fbin=fbin_contract,
                current_filters_csv=filters_csv_contract,
                table_identity=table_identity,
                filters=filters,
                candidate_validity_predicate=candidate_validity_predicate,
                query_validity_predicate=query_validity_predicate,
                candidate_validity_provenance=candidate_validity_provenance,
                query_validity_provenance=query_validity_provenance,
                k=args.k,
            )
            validate_stable_cohort_contract(extension_base["cohort"], external_cohort)
            extension_added_query_ids = validate_prefix_truth_contract(
                extension_base["rows"],
                extension_base["checkpoint"],
                extension_base["manifest"],
                extension_base["query_ids"],
                query_ids,
                filters,
                k=args.k,
                fbin=fbin_contract,
                filters_csv=filters_csv_contract,
                table_identity=table_identity,
                eligible_population=eligible_population_contract,
                candidate_validity_predicate=candidate_validity_predicate,
                query_validity_predicate=query_validity_predicate,
                candidate_validity_provenance=candidate_validity_provenance,
                query_validity_provenance=query_validity_provenance,
                prior_query_source=extension_base["source"],
                extended_query_source=query_source_contract,
            )
            extension_start = extension_base["old_query_count"]
            if args.out.exists() != checkpoint_path.exists():
                raise SystemExit("strict prefix extension destination requires both output and checkpoint, or neither")
            if args.out.exists() and not args.resume:
                raise SystemExit("strict prefix extension destination exists; pass --resume to continue it")
            if truth_manifest_path_for(args.out).exists() and args.out.exists():
                raise SystemExit("strict prefix extension destination is already finalized; choose a new --out")
            rows_out = [dict(row) for row in extension_base["rows"]]
            checkpoint_candidate_records = {
                name: dict(record)
                for name, record in extension_base["checkpoint"]["completed_filters"].items()
            }
            if args.out.exists():
                try:
                    rows_out = read_csv(args.out)
                    extension_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise SystemExit(f"strict prefix extension destination is unreadable: {exc}") from exc
                completed = validate_extension_checkpoint(
                    extension_checkpoint,
                    rows_out,
                    query_ids,
                    filters,
                    base_query_count=extension_start,
                    k=args.k,
                    fbin=fbin_contract,
                    filters_csv=filters_csv_contract,
                    table_identity=table_identity,
                    eligible_population=eligible_population_contract,
                    candidate_validity_predicate=candidate_validity_predicate,
                    query_validity_predicate=query_validity_predicate,
                    candidate_validity_provenance=candidate_validity_provenance,
                    query_validity_provenance=query_validity_provenance,
                    query_source=query_source_contract,
                    base_truth=args.extend_from_truth,
                    base_checkpoint=args.extend_from_checkpoint,
                    base_query_ids=extension_base["query_ids"],
                    base_rows=extension_base["rows"],
                )
                checkpoint_candidate_records = extension_checkpoint["candidate_populations"]
        elif args.resume and (args.out.exists() or checkpoint_path.exists()):
            if not args.out.exists() or not checkpoint_path.exists():
                raise SystemExit("resume requires both truth CSV and atomic checkpoint sidecar")
            try:
                resumed = read_csv(args.out)
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"resume artifact/checkpoint cannot be read: {exc}") from exc
            rows_out = resumed
            completed = validate_resume_checkpoint(
                checkpoint,
                rows_out,
                query_ids,
                filters,
                k=args.k,
                fbin=fbin_contract,
                filters_csv=filters_csv_contract,
                table_identity=table_identity,
                eligible_population=eligible_population_contract,
                candidate_validity_predicate=candidate_validity_predicate,
                query_validity_predicate=query_validity_predicate,
                candidate_validity_provenance=candidate_validity_provenance,
                query_validity_provenance=query_validity_provenance,
                query_source=query_source_contract,
            )
            checkpoint_completed_records = checkpoint["completed_filters"]
        eligible_query_count = int(eligible_population_contract["rows"])
        eligible_query_ids_hash = str(eligible_population_contract["ids_sha256"])
        eligible_query_population_provenance = str(
            eligible_population_contract["provenance"]
        )
        split_by_query_no = {
            query_no: ("calibration" if query_no < len(calibration_query_ids) else "final")
            for query_no in range(len(query_ids))
        }
        seed_by_query_no = {
            query_no: (
                args.calibration_seed
                if query_no < len(calibration_query_ids)
                else args.final_seed
            )
            for query_no in range(len(query_ids))
        }

        print(
            json.dumps(
                {
                    "vectors": vector_rows,
                    "filters": len(filters),
                    "candidate_validity_predicate": candidate_validity_predicate,
                    "candidate_validity_provenance": candidate_validity_provenance,
                    "query_validity_predicate": query_validity_predicate,
                    "query_validity_provenance": query_validity_provenance,
                    "eligible_query_population": eligible_query_count,
                    "eligible_query_ids_sha256": eligible_query_ids_hash,
                    "calibration_queries": len(calibration_query_ids),
                    "calibration_seed": args.calibration_seed,
                    "final_queries": len(final_query_ids),
                    "final_seed": args.final_seed,
                    "disjoint": not bool(set(final_query_ids) & set(calibration_query_ids)),
                    "self_excluded": True,
                    "tie_aware": True,
                    "completed_filters": sorted(completed),
                },
                indent=2,
            ),
            flush=True,
        )

        query_vector_mapping = verify_query_vector_mapping(
            cur,
            args.table,
            xb,
            query_ids,
            query_validity_predicate,
        )
        query_population = {
            "candidate_validity_predicate": candidate_validity_predicate,
            "candidate_validity_provenance": candidate_validity_provenance,
            "query_validity_predicate": query_validity_predicate,
            "query_validity_provenance": query_validity_provenance,
            "eligible_rows": eligible_query_count,
            "eligible_ids_sha256": eligible_query_ids_hash,
            "eligible_ids_hash_contract": "sha256(little-endian int64 sorted PostgreSQL IDs), domain-separated v1",
            "provenance": eligible_query_population_provenance,
            "fetch_sql_contract": "SELECT id WHERE (query_validity_predicate) ORDER BY id",
            "fetch_latency_ms": eligible_sql_ms,
            "query_source": query_source_contract,
            "selection": {
                "algorithm": "numpy.default_rng rejection sampling over sorted eligible IDs without replacement",
                "numpy_version": np.__version__,
                "calibration_ids_sha256": sorted_ids_sha256(np.sort(calibration_query_ids)),
                "final_ids_sha256": sorted_ids_sha256(np.sort(final_query_ids)),
                "calibration_final_disjoint": not bool(
                    set(final_query_ids) & set(calibration_query_ids)
                ),
            },
        }
        database_source = {
            "table": args.table,
            "rows": table_rows,
            "min_id": min_id,
            "max_id": max_id,
            "table_oid": table_oid,
            "table_relfilenode": table_relfilenode,
            "query_vector_mapping": query_vector_mapping,
            "query_population": query_population,
        }

        for position, filter_spec in enumerate(filters, start=1):
            filter_name = filter_spec["filter_name"]
            candidate_ids, sql_ms = fetch_candidate_ids(
                cur,
                args.table,
                filter_spec["predicate"],
                candidate_validity_predicate,
            )
            if candidate_ids.size < args.k:
                raise SystemExit(f"filter={filter_name} only has {candidate_ids.size} candidates")
            if int(candidate_ids.min()) < 0 or int(candidate_ids.max()) >= vector_rows:
                raise SystemExit(f"filter={filter_name} has id outside vector row range")
            candidate_ids.sort()
            if candidate_ids.size > 1 and np.any(candidate_ids[1:] == candidate_ids[:-1]):
                raise SystemExit(f"filter={filter_name} candidate IDs are not unique")
            candidate_ids_hash = sorted_ids_sha256(candidate_ids)
            if extension_requested:
                record = checkpoint_candidate_records.get(filter_name)
                expected_record = {
                    "candidate_rows": len(candidate_ids),
                    "candidate_ids_sha256": candidate_ids_hash,
                }
                if record != expected_record:
                    raise SystemExit(
                        f"strict prefix extension candidate population mismatch for filter={filter_name}: "
                        f"expected={json.dumps(record, sort_keys=True)} "
                        f"observed={json.dumps(expected_record, sort_keys=True)}"
                    )
                if filter_name in completed:
                    print(f"filter={filter_name} already extended; candidate population revalidated", flush=True)
                    continue
            elif filter_name in completed:
                record = checkpoint_completed_records.get(filter_name)
                expected_record = {
                    "candidate_rows": len(candidate_ids),
                    "candidate_ids_sha256": candidate_ids_hash,
                }
                if record != expected_record:
                    raise SystemExit(
                        f"resume candidate population mismatch for filter={filter_name}: "
                        f"expected={json.dumps(record, sort_keys=True)} "
                        f"observed={json.dumps(expected_record, sort_keys=True)}"
                    )
                for row in rows_out:
                    if row.get("filter_name") == filter_name and row.get("candidate_ids_sha256") != candidate_ids_hash:
                        raise SystemExit(f"resume row candidate hash mismatch for filter={filter_name}")
                print(f"filter={filter_name} already complete; candidate population revalidated", flush=True)
                continue
            search_query_ids = (
                extension_added_query_ids if extension_requested else query_ids
            )
            query_in_filter = np.isin(search_query_ids, candidate_ids, assume_unique=True)
            print(
                f"filter={filter_name} ({position}/{len(filters)}) candidates={len(candidate_ids)} "
                f"selectivity={len(candidate_ids) / table_rows:.6f} sql_ms={sql_ms:.1f} "
                f"queries={len(search_query_ids)}",
                flush=True,
            )
            try:
                exact_ids, exact_distances, exact_ms = exact_topk_batch(
                    xb,
                    search_query_ids,
                    candidate_ids,
                    args.k,
                    args.chunk_rows,
                    args.progress_chunks,
                    filter_name,
                    args.distance_workers,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            amortized_ms = exact_ms / len(search_query_ids)
            actual_selectivity = len(candidate_ids) / table_rows
            rows_out.extend(
                truth_row(
                    (extension_start + query_pos if extension_requested else query_pos),
                    int(query_id),
                    filter_spec,
                    actual_selectivity,
                    len(candidate_ids),
                    exact_ids[query_pos],
                    exact_distances[query_pos],
                    amortized_ms,
                    seed_by_query_no[extension_start + query_pos if extension_requested else query_pos],
                    split_by_query_no[extension_start + query_pos if extension_requested else query_pos],
                    bool(query_in_filter[query_pos]),
                    args.k,
                    candidate_validity_predicate,
                    query_validity_predicate,
                    eligible_query_count,
                    eligible_query_ids_hash,
                    eligible_query_population_provenance,
                    candidate_validity_provenance,
                    query_validity_provenance,
                    candidate_ids_hash,
                )
                for query_pos, query_id in enumerate(search_query_ids)
            )
            rows_out.sort(key=lambda row: (str(row["filter_name"]), int(row["query_no"])))
            write_csv(args.out, rows_out)
            write_summary(args.out, rows_out)
            completed.add(filter_name)
            record = {
                "candidate_rows": len(candidate_ids),
                "candidate_ids_sha256": candidate_ids_hash,
            }
            if extension_requested:
                checkpoint_candidate_records[filter_name] = record
            else:
                checkpoint_completed_records[filter_name] = record
            checkpoint_payload = (
                extension_checkpoint_payload(
                    k=args.k,
                    query_ids=query_ids,
                    candidate_validity_predicate=candidate_validity_predicate,
                    candidate_validity_provenance=candidate_validity_provenance,
                    query_validity_predicate=query_validity_predicate,
                    query_validity_provenance=query_validity_provenance,
                    fbin=fbin_contract,
                    filters_csv=filters_csv_contract,
                    filters=filters,
                    table=table_identity,
                    eligible_population=eligible_population_contract,
                    query_source=query_source_contract,
                    candidate_populations=checkpoint_candidate_records,
                    completed_filters=completed,
                    base_truth=args.extend_from_truth,
                    base_checkpoint=args.extend_from_checkpoint,
                    base_query_ids=extension_base["query_ids"],
                )
                if extension_requested
                else {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "k": args.k,
                    "query_count": len(query_ids),
                    "query_ids_sha256": ordered_ids_sha256(query_ids),
                    "candidate_validity_predicate": candidate_validity_predicate,
                    "candidate_validity_provenance": candidate_validity_provenance,
                    "query_validity_predicate": query_validity_predicate,
                    "query_validity_provenance": query_validity_provenance,
                    "fbin": fbin_contract,
                    "filters_csv": filters_csv_contract,
                    "filters": filter_contract(filters),
                    "table": table_identity,
                    "eligible_query_population": eligible_population_contract,
                    "query_source": query_source_contract,
                    "completed_filters": checkpoint_completed_records,
                }
            )
            atomic_write_json(
                checkpoint_path,
                checkpoint_payload,
            )
            print(f"filter={filter_name} exact_ms={exact_ms:.1f}; checkpointed {args.out}", flush=True)

    write_summary(args.out, rows_out)
    manifest_path = truth_manifest_path_for(args.out)
    manifest = {
        "artifact_valid": len(rows_out) == len(filters) * len(query_ids),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "method": "exact_filtered_l2_tie_aware",
        "exact_distance_contract": {
            "metric": "squared_l2",
            "implementation": "scipy.spatial.distance.cdist",
            "dtype": "float64",
            "chunking": "one candidate chunk per cdist call",
            "candidate_chunk_rows": args.chunk_rows,
            "distance_workers": args.distance_workers,
            "parallel_merge_order": "ascending candidate chunk start",
            "retained_boundary": "k+1, tie-broken by (distance,id)",
        },
        "self_excluded": True,
        "recall_contract": "returned SQL-valid IDs with squared L2 <= kth_distance_sq + tie_tolerance, capped at k",
        "rows": vector_rows,
        "k": args.k,
        "calibration": {"queries": len(calibration_query_ids), "seed": args.calibration_seed},
        "final": {"queries": len(final_query_ids), "seed": args.final_seed},
        "query_ids_disjoint": not bool(set(final_query_ids) & set(calibration_query_ids)),
        "validity_contract": {
            "candidate_validity_predicate": candidate_validity_predicate,
            "candidate_validity_provenance": candidate_validity_provenance,
            "query_validity_predicate": query_validity_predicate,
            "query_validity_provenance": query_validity_provenance,
            "query_inherits_candidate": args.query_validity_predicate is None,
            "candidate_fetch_sql_contract": "WHERE (filter_predicate) AND (candidate_validity_predicate)",
        },
        "eligible_query_population": query_population,
        "candidate_universe": {
            "candidate_validity_predicate": candidate_validity_predicate,
            "rows": (
                int(external_cohort["candidate_universe_rows"])
                if args.query_cohort_csv is not None
                else int(eligible_query_count)
            ),
            "provenance": (
                "bound_external_query_cohort_manifest"
                if args.query_cohort_csv is not None
                else "postgres_candidate_validity_scan"
            ),
        },
        "query_source": query_source_contract,
        "filters": len(filters),
        "truth_rows": len(rows_out),
        "inputs": {
            "fbin": fbin_contract,
            "filters_csv": filters_csv_contract,
            "postgres": database_source,
        },
        "outputs": {
            "truth_csv": {"path": str(args.out.resolve()), "sha256": sha256_file(args.out)},
            "summary_csv": {
                "path": str(args.out.with_name(args.out.stem + "_summary.csv").resolve()),
                "sha256": sha256_file(args.out.with_name(args.out.stem + "_summary.csv")),
            },
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
            },
        },
    }
    if extension_requested:
        manifest["extension"] = {
            "version": STRICT_PREFIX_EXTENSION_VERSION,
            "operation": "strict_prefix_extension",
            "base_truth": {
                "path": str(args.extend_from_truth.resolve()),
                "sha256": sha256_file(args.extend_from_truth),
            },
            "base_checkpoint": {
                "path": str(args.extend_from_checkpoint.resolve()),
                "sha256": sha256_file(args.extend_from_checkpoint),
            },
            "base_query_count": extension_start,
            "base_query_ids_sha256": ordered_ids_sha256(extension_base["query_ids"]),
            "added_query_count": len(extension_added_query_ids),
            "added_query_ids_sha256": ordered_ids_sha256(extension_added_query_ids),
            "computed_query_range": [extension_start, len(query_ids) - 1],
            "exact_recomputation": "new_query_suffix_only",
        }
    atomic_write_json(manifest_path, manifest)
    if not manifest["artifact_valid"]:
        raise SystemExit("truth artifact is incomplete")
    print(f"wrote {args.out} rows={len(rows_out)}", flush=True)


if __name__ == "__main__":
    main()
