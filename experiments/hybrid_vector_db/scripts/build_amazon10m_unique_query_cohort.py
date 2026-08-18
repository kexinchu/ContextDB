from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import psycopg

from amazon10m_exact_truth import (
    DEFAULT_FBIN,
    DEFAULT_VALIDITY_PREDICATE,
    atomic_write_json,
    fetch_eligible_query_ids,
    ordered_ids_sha256,
    sample_disjoint_eligible_query_ids,
    sha256_file,
    sorted_ids_sha256,
    verify_query_vector_mapping,
    write_csv,
)
from common_pg import pg_config_from_env
from faiss_hnsw_sql_attribute_filter_10m import read_fbin_memmap


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = (
    ROOT
    / "results/hybrid_vector_db/amazon10m_unique_embedding_query_cohort_q200.csv"
)
SCHEMA_VERSION = 1
METHOD = "deterministic_unique_projection_fingerprint_cohort_v1"
FINGERPRINT_CONTRACT = (
    "two deterministic uint64 hashes over fixed raw-vector word projections; "
    "an exact duplicate always shares a fingerprint, so accepting only singleton "
    "fingerprints cannot admit an exact duplicate; hash/projection collisions only "
    "reject otherwise eligible rows"
)


def fingerprint_word_positions(dimensions: int, requested: int) -> tuple[int, ...]:
    if dimensions <= 0 or dimensions % 2:
        raise ValueError("raw uint64 fingerprinting requires a positive even dimension")
    word_count = dimensions // 2
    if requested <= 0:
        raise ValueError("fingerprint word count must be positive")
    count = min(requested, word_count)
    positions = (np.arange(count, dtype=np.int64) * word_count) // count
    result = tuple(int(value) for value in positions)
    if len(set(result)) != count:
        raise AssertionError("fingerprint word selection is not unique")
    return result


def _splitmix64(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=False)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def projected_fingerprints(
    vectors: np.ndarray,
    positions: Sequence[int],
    *,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if vectors.ndim != 2 or vectors.shape[1] % 2:
        raise ValueError("vectors must be a two-dimensional even-width float32 matrix")
    if vectors.dtype != np.dtype(np.float32):
        raise ValueError("fingerprint input must be float32")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    word_count = vectors.shape[1] // 2
    selected = tuple(int(value) for value in positions)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("fingerprint positions must be nonempty and unique")
    if min(selected) < 0 or max(selected) >= word_count:
        raise ValueError("fingerprint position is outside the raw-vector word range")

    rows = vectors.shape[0]
    first = np.empty(rows, dtype=np.uint64)
    second = np.empty(rows, dtype=np.uint64)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        raw = np.ascontiguousarray(vectors[start:end]).view(np.uint64).reshape(end - start, word_count)
        h1 = np.full(end - start, np.uint64(0x243F6A8885A308D3), dtype=np.uint64)
        h2 = np.full(end - start, np.uint64(0x13198A2E03707344), dtype=np.uint64)
        for ordinal, position in enumerate(selected):
            word = raw[:, position]
            salt1 = np.uint64((0x9E3779B97F4A7C15 * (ordinal + 1)) & ((1 << 64) - 1))
            salt2 = np.uint64((0xD1B54A32D192ED03 * (ordinal + 1)) & ((1 << 64) - 1))
            h1 ^= _splitmix64(word ^ salt1)
            h2 = _splitmix64(h2 ^ word ^ salt2)
        first[start:end] = h1
        second[start:end] = h2
    return first, second


def singleton_fingerprint_ids(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("fingerprint columns must be equal-length vectors")
    pairs = np.empty(first.size, dtype=[("first", "<u8"), ("second", "<u8")])
    pairs["first"] = first
    pairs["second"] = second
    _, first_positions, counts = np.unique(
        pairs, return_index=True, return_counts=True
    )
    result = np.sort(first_positions[counts == 1].astype(np.int64, copy=False))
    if result.size > 1 and np.any(result[1:] <= result[:-1]):
        raise AssertionError("singleton fingerprint IDs are not sorted and unique")
    return result


def write_cohort_csv(
    path: Path,
    query_ids: np.ndarray,
    calibration_queries: int,
    candidate_validity_predicate: str,
) -> None:
    write_csv(
        path,
        [
            {
                "query_no": query_no,
                "query_id": int(query_id),
                "query_split": (
                    "calibration" if query_no < calibration_queries else "final"
                ),
                "candidate_validity_predicate": candidate_validity_predicate,
                "query_validity_predicate": candidate_validity_predicate,
            }
            for query_no, query_id in enumerate(query_ids)
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a preregistered Amazon-10M query cohort without exact-vector duplicates."
    )
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--table", default="public.amazon_grocery_reviews_10m_pgvector")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--calibration-queries", type=int, default=100)
    parser.add_argument("--calibration-seed", type=int, default=57)
    parser.add_argument("--final-queries", type=int, default=100)
    parser.add_argument("--final-seed", type=int, default=58)
    parser.add_argument("--fingerprint-words", type=int, default=16)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    args = parser.parse_args()

    if args.calibration_queries < 0 or args.final_queries < 0:
        parser.error("query counts must not be negative")
    if args.calibration_queries + args.final_queries <= 0:
        parser.error("at least one query is required")

    started = time.perf_counter()
    vectors, rows, dimensions = read_fbin_memmap(args.fbin, args.rows)
    if rows != args.rows:
        raise SystemExit(f"fbin row mismatch: expected={args.rows} observed={rows}")
    positions = fingerprint_word_positions(dimensions, args.fingerprint_words)
    first, second = projected_fingerprints(
        vectors, positions, chunk_rows=args.chunk_rows
    )
    singleton_ids = singleton_fingerprint_ids(first, second)
    del first, second
    fingerprint_ms = (time.perf_counter() - started) * 1000.0

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
        if (table_rows, min_id, max_id) != (rows, 0, rows - 1):
            raise SystemExit(
                "PostgreSQL/fbin ID-space mismatch: "
                f"table={(table_rows, min_id, max_id)} fbin={(rows, 0, rows - 1)}"
            )
        valid_ids, valid_fetch_ms = fetch_eligible_query_ids(
            cur, args.table, DEFAULT_VALIDITY_PREDICATE
        )
        eligible_ids = np.intersect1d(
            singleton_ids, valid_ids, assume_unique=True
        ).astype(np.int64, copy=False)
        if eligible_ids.size < args.calibration_queries + args.final_queries:
            raise SystemExit("not enough singleton, embedding-valid query vectors")
        calibration = sample_disjoint_eligible_query_ids(
            eligible_ids, set(), args.calibration_queries, args.calibration_seed
        )
        final = sample_disjoint_eligible_query_ids(
            eligible_ids,
            {int(value) for value in calibration},
            args.final_queries,
            args.final_seed,
        )
        query_ids = np.concatenate((calibration, final))
        mapping = verify_query_vector_mapping(
            cur,
            args.table,
            vectors,
            query_ids,
            DEFAULT_VALIDITY_PREDICATE,
        )

    write_cohort_csv(
        args.out, query_ids, args.calibration_queries, DEFAULT_VALIDITY_PREDICATE
    )
    fbin_sha256 = sha256_file(args.fbin)
    manifest_path = args.out.with_name(args.out.stem + "_manifest.json")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": METHOD,
        "candidate_validity_predicate": DEFAULT_VALIDITY_PREDICATE,
        "query_validity_predicate": DEFAULT_VALIDITY_PREDICATE,
        "uniqueness_contract": {
            "contract": FINGERPRINT_CONTRACT,
            "raw_dtype": "little-endian float32 viewed as uint64 word pairs",
            "hashes": 2,
            "selected_word_positions": list(positions),
            "selected_word_count": len(positions),
            "all_rows_fingerprinted": True,
            "singleton_fingerprints": int(singleton_ids.size),
            "duplicate_admission_false_negative_only": True,
        },
        "eligible_query_population": {
            "rows": int(eligible_ids.size),
            "ids_sha256": sorted_ids_sha256(eligible_ids),
            "provenance": (
                f"{METHOD}:{args.table}:oid={table_oid}:relfilenode={table_relfilenode}"
            ),
            "embedding_valid_rows": int(valid_ids.size),
            "singleton_fingerprint_rows": int(singleton_ids.size),
        },
        "selection": {
            "algorithm": "numpy.default_rng rejection sampling over sorted eligible IDs",
            "numpy_version": np.__version__,
            "calibration": {
                "queries": int(calibration.size),
                "seed": args.calibration_seed,
                "ordered_ids_sha256": ordered_ids_sha256(calibration),
            },
            "final": {
                "queries": int(final.size),
                "seed": args.final_seed,
                "ordered_ids_sha256": ordered_ids_sha256(final),
            },
            "disjoint": not bool(set(calibration) & set(final)),
            "query_ids_sha256": ordered_ids_sha256(query_ids),
        },
        "inputs": {
            "fbin": {
                "path": str(args.fbin.resolve()),
                "sha256": fbin_sha256,
                "rows": rows,
                "dimensions": dimensions,
            },
            "postgres": {
                "table": args.table,
                "rows": table_rows,
                "min_id": min_id,
                "max_id": max_id,
                "table_oid": table_oid,
                "table_relfilenode": table_relfilenode,
                "valid_fetch_latency_ms": valid_fetch_ms,
                "query_vector_mapping": mapping,
            },
        },
        "timing": {"fingerprint_and_singleton_detection_ms": fingerprint_ms},
        "outputs": {
            "cohort_csv": {
                "path": str(args.out.resolve()),
                "sha256": sha256_file(args.out),
                "rows": int(query_ids.size),
            }
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "cohort_csv": str(args.out),
                "manifest": str(manifest_path),
                "embedding_valid_rows": int(valid_ids.size),
                "singleton_fingerprint_rows": int(singleton_ids.size),
                "eligible_rows": int(eligible_ids.size),
                "queries": int(query_ids.size),
                "elapsed_s": (time.perf_counter() - started),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
