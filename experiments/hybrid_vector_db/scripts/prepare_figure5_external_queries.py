#!/usr/bin/env python3
"""Prepare disjoint q200 calibration and q10K measurement query cohorts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import psycopg

try:
    from .common_pg import pg_config_from_env
except ImportError:
    from common_pg import pg_config_from_env


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path(os.environ.get("OOD_ANNS_DATA", ROOT / "data" / "OOD-ANNS"))
CALIBRATION_QUERIES = 200
MEASUREMENT_QUERIES = 10_000
TOTAL_QUERIES = CALIBRATION_QUERIES + MEASUREMENT_QUERIES


class QueryPreparationError(RuntimeError):
    """External query preparation failed a reproducibility gate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cohort_rows() -> list[dict[str, object]]:
    rows = [
        {
            "query_no": query_no,
            "query_id": 10_000 + query_no,
            "query_split": "calibration",
        }
        for query_no in range(CALIBRATION_QUERIES)
    ]
    rows.extend(
        {
            "query_no": CALIBRATION_QUERIES + query_id,
            "query_id": query_id,
            "query_split": "measurement",
        }
        for query_id in range(MEASUREMENT_QUERIES)
    )
    return rows


def write_csv_atomic(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=["query_no", "query_id", "query_split"]
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_fbin_atomic(path: Path, vectors: np.ndarray) -> None:
    if vectors.ndim != 2 or vectors.dtype != np.float32:
        raise QueryPreparationError("fbin vectors must be a float32 matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as target:
        target.write(struct.pack("<ii", vectors.shape[0], vectors.shape[1]))
        target.write(np.ascontiguousarray(vectors, dtype="<f4").tobytes())
    temporary.replace(path)


def vector_text(vector: np.ndarray) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"


def ensure_query_ids(cur: Any, table: str, required: int) -> dict[str, int]:
    cur.execute(
        f"SELECT count(*), min(qid), max(qid), "
        f"count(DISTINCT qid) FROM {table} WHERE qid >= 0 AND qid < %s",
        (required,),
    )
    count, minimum, maximum, distinct = cur.fetchone()
    observed = tuple(int(value) for value in (count, minimum, maximum, distinct))
    expected = (required, 0, required - 1, required)
    if observed != expected:
        raise QueryPreparationError(
            f"{table} query ID coverage mismatch: expected={expected}, observed={observed}"
        )
    return {
        "required_rows": required,
        "observed_rows": observed[0],
        "min_qid": observed[1],
        "max_qid": observed[2],
        "distinct_qids": observed[3],
    }


def prepare_laion(
    *,
    source_npy: Path,
    query_table: str,
    query_fbin: Path,
    execute_db: bool,
) -> dict[str, Any]:
    if not source_npy.is_file():
        raise QueryPreparationError(f"LAION text query source is missing: {source_npy}")
    source = np.load(source_npy, mmap_mode="r")
    if source.ndim != 2 or source.shape[0] < TOTAL_QUERIES or source.shape[1] != 512:
        raise QueryPreparationError(
            f"LAION query source must contain at least {TOTAL_QUERIES}x512 vectors"
        )
    vectors = np.asarray(source[:TOTAL_QUERIES], dtype=np.float32)
    write_fbin_atomic(query_fbin, vectors)

    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        if execute_db:
            for qid in range(MEASUREMENT_QUERIES, TOTAL_QUERIES):
                cur.execute(
                    f"INSERT INTO {query_table} "
                    "(qid, embedding, width, labels, label_count) "
                    "VALUES (%s, %s::vector, 0, ARRAY[]::int[], 0) "
                    "ON CONFLICT (qid) DO UPDATE SET embedding = EXCLUDED.embedding",
                    (qid, vector_text(vectors[qid])),
                )
        coverage = ensure_query_ids(cur, query_table, TOTAL_QUERIES)
    return {
        "dataset": "laion",
        "query_table": query_table,
        "source": {
            "path": str(source_npy.resolve()),
            "sha256": sha256_file(source_npy),
            "rows": int(source.shape[0]),
            "dimensions": int(source.shape[1]),
        },
        "query_fbin": {
            "path": str(query_fbin.resolve()),
            "sha256": sha256_file(query_fbin),
            "rows": TOTAL_QUERIES,
            "dimensions": 512,
        },
        "postgres": coverage,
        "db_extension_executed": execute_db,
    }


def prepare_yfcc(*, query_u8bin: Path, query_table: str) -> dict[str, Any]:
    if not query_u8bin.is_file():
        raise QueryPreparationError(f"YFCC query source is missing: {query_u8bin}")
    with query_u8bin.open("rb") as source:
        header = np.fromfile(source, dtype="<u4", count=2)
    if len(header) != 2:
        raise QueryPreparationError(f"bad YFCC u8bin header: {query_u8bin}")
    rows, dimensions = map(int, header)
    if rows < TOTAL_QUERIES or dimensions != 192:
        raise QueryPreparationError(
            f"YFCC query source must contain at least {TOTAL_QUERIES}x192 vectors"
        )
    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        coverage = ensure_query_ids(conn.cursor(), query_table, TOTAL_QUERIES)
    return {
        "dataset": "yfcc",
        "query_table": query_table,
        "source": {
            "path": str(query_u8bin.resolve()),
            "sha256": sha256_file(query_u8bin),
            "rows": rows,
            "dimensions": dimensions,
        },
        "postgres": coverage,
        "db_extension_executed": False,
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Figure 5 external query cohorts and LAION q10200 source."
    )
    parser.add_argument("--dataset", choices=("yfcc", "laion"), required=True)
    parser.add_argument("--out-cohort", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--execute-db", action="store_true")
    parser.add_argument(
        "--yfcc-query-u8bin",
        type=Path,
        default=DEFAULT_DATA_ROOT / "YFCC10M/query.public.100K.u8bin",
    )
    parser.add_argument("--yfcc-query-table", default="public.yfcc10m_queries")
    parser.add_argument(
        "--laion-source-npy",
        type=Path,
        default=DEFAULT_DATA_ROOT / "LAION25M/text_emb/text_emb_26.npy",
    )
    parser.add_argument("--laion-query-table", default="public.laion25m_queries")
    parser.add_argument(
        "--laion-query-fbin",
        type=Path,
        default=ROOT / "results/hybrid_vector_db/figure5_r35_laion_query_q10200.fbin",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        details = (
            prepare_yfcc(
                query_u8bin=args.yfcc_query_u8bin.resolve(),
                query_table=args.yfcc_query_table,
            )
            if args.dataset == "yfcc"
            else prepare_laion(
                source_npy=args.laion_source_npy.resolve(),
                query_table=args.laion_query_table,
                query_fbin=args.laion_query_fbin.resolve(),
                execute_db=args.execute_db,
            )
        )
        rows = cohort_rows()
        write_csv_atomic(args.out_cohort.resolve(), rows)
        manifest = {
            "schema_version": 1,
            "artifact_type": "sqlens_figure5_external_query_cohort",
            "artifact_valid": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selection": {
                "calibration_query_ids": "10000..10199",
                "measurement_query_ids": "0..9999",
                "calibration_measurement_disjoint": True,
                "query_no_domain": "0..10199",
            },
            "cohort": {
                "path": str(args.out_cohort.resolve()),
                "sha256": sha256_file(args.out_cohort.resolve()),
                "rows": len(rows),
            },
            **details,
        }
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.manifest_out.with_suffix(args.manifest_out.suffix + ".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.manifest_out)
    except (OSError, ValueError, psycopg.Error, QueryPreparationError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        f"wrote {args.out_cohort.resolve()} and {args.manifest_out.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
