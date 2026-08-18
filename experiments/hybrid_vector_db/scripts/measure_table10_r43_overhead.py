#!/usr/bin/env python3
"""Measure Table 10 Panel C overhead cells under the r43 release contract.

Supports:
- memory: resident guidance/fragment-cache profile (+ optional RSS delta)
- storage: Stock vs BFS relation/index sizes
- build: wall time / peak RSS from a provided BFS rewrite proof JSON
- maintenance: p50/p95 invalidation/rebuild/reactivation from lifecycle CSV

Use --dry-run to emit a Pending skeleton without touching PostgreSQL.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any


ARTIFACT_TYPE = "sqlens_table10_overhead"
SCHEMA_VERSION = 1


def pending_rows() -> list[dict[str, str]]:
    return [
        {
            "cost": "resident_guidance_reuse_memory",
            "stock": "---",
            "sqlens": "Pending",
            "delta": "Pending",
        },
        {
            "cost": "persistent_db_storage",
            "stock": "Pending",
            "sqlens": "Pending",
            "delta": "Pending",
        },
        {
            "cost": "hnsw_build_bfs_rewrite_time",
            "stock": "Pending",
            "sqlens": "Pending",
            "delta": "Pending",
        },
        {
            "cost": "maintenance_under_writes_p95",
            "stock": "---",
            "sqlens": "Pending",
            "delta": "Pending",
        },
    ]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def measure_memory(args: argparse.Namespace) -> dict[str, str]:
    import psycopg

    conninfo = (
        f"host={args.pghost} port={args.pgport} dbname={args.pgdatabase} "
        f"user={args.pguser} password={args.pgpassword}"
    )
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT vector_sqlens_build_id()")
            build_id = cur.fetchone()[0]
            if args.expected_build_id and build_id != args.expected_build_id:
                raise SystemExit(
                    f"build id mismatch: {build_id} != {args.expected_build_id}"
                )
            cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
            profile = json.loads(cur.fetchone()[0])
    resident_bytes = float(
        profile.get("resident_bytes")
        or profile.get("cache_resident_bytes")
        or profile.get("total_bytes")
        or 0.0
    )
    resident_mib = resident_bytes / (1024.0 * 1024.0)
    meta = float(profile.get("metadata_bytes") or 0.0) / (1024.0 * 1024.0)
    cache = float(profile.get("fragment_cache_bytes") or profile.get("cache_bytes") or 0.0) / (
        1024.0 * 1024.0
    )
    return {
        "cost": "resident_guidance_reuse_memory",
        "stock": "---",
        "sqlens": f"{resident_mib:.2f} MiB",
        "delta": f"meta={meta:.2f} MiB; cache={cache:.2f} MiB",
    }


def measure_storage(args: argparse.Namespace) -> dict[str, str]:
    import psycopg

    conninfo = (
        f"host={args.pghost} port={args.pgport} dbname={args.pgdatabase} "
        f"user={args.pguser} password={args.pgpassword}"
    )
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  pg_total_relation_size(%s::regclass) AS stock_bytes,
                  pg_total_relation_size(%s::regclass) AS bfs_bytes
                """,
                (args.stock_relation, args.bfs_relation),
            )
            stock_bytes, bfs_bytes = cur.fetchone()
    stock_gib = float(stock_bytes) / (1024.0**3)
    bfs_gib = float(bfs_bytes) / (1024.0**3)
    delta_pct = ((bfs_gib / stock_gib) - 1.0) * 100.0 if stock_gib > 0 else float("nan")
    return {
        "cost": "persistent_db_storage",
        "stock": f"{stock_gib:.2f} GiB",
        "sqlens": f"{bfs_gib:.2f} GiB",
        "delta": f"+{delta_pct:.2f}%",
    }


def measure_build(proof_json: Path) -> dict[str, str]:
    payload = json.loads(proof_json.read_text(encoding="utf-8"))
    stock = payload.get("source_build") or payload.get("stock_build") or {}
    rewrite = payload.get("bfs_rewrite") or payload.get("rewrite") or {}
    stock_s = stock.get("wall_seconds") or stock.get("wall_time_s")
    rewrite_s = rewrite.get("wall_seconds") or rewrite.get("wall_time_s")
    stock_rss = stock.get("peak_rss_mb") or stock.get("peak_rss_mib")
    rewrite_rss = rewrite.get("peak_rss_mb") or rewrite.get("peak_rss_mib")
    ratio = payload.get("storage_ratio") or rewrite.get("storage_ratio")
    return {
        "cost": "hnsw_build_bfs_rewrite_time",
        "stock": (
            f"{float(stock_s):.1f}s / {float(stock_rss):.0f} MiB"
            if stock_s is not None and stock_rss is not None
            else "Pending"
        ),
        "sqlens": (
            f"{float(rewrite_s):.1f}s / {float(rewrite_rss):.0f} MiB"
            if rewrite_s is not None and rewrite_rss is not None
            else "Pending"
        ),
        "delta": f"storage_ratio={float(ratio):.4f}" if ratio is not None else "Pending",
    }


def measure_maintenance(lifecycle_csv: Path) -> dict[str, str]:
    with lifecycle_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lat_keys = [
        "invalidation_ms",
        "rebuild_ms",
        "reactivation_ms",
        "maintenance_ms",
        "event_latency_ms",
    ]
    values: list[float] = []
    counts = {"invalidation": 0, "rebuild": 0, "reactivation": 0, "stale_bypass": 0}
    for row in rows:
        event = str(row.get("event") or row.get("kind") or "").lower()
        for key, token in (
            ("invalidation", "invalid"),
            ("rebuild", "rebuild"),
            ("reactivation", "reactivat"),
            ("stale_bypass", "stale"),
        ):
            if token in event:
                counts[key] += 1
        for key in lat_keys:
            raw = row.get(key)
            if raw in (None, ""):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
    if not values:
        return {
            "cost": "maintenance_under_writes_p95",
            "stock": "---",
            "sqlens": "Pending",
            "delta": "Pending",
        }
    p50 = percentile(values, 50.0)
    p95 = percentile(values, 95.0)
    return {
        "cost": "maintenance_under_writes_p95",
        "stock": "---",
        "sqlens": f"{p95:.2f} ms",
        "delta": (
            f"p50={p50:.2f}; inv={counts['invalidation']}; "
            f"rebuild={counts['rebuild']}; react={counts['reactivation']}; "
            f"stale={counts['stale_bypass']}"
        ),
    }


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    rows = pending_rows()
    by_cost = {row["cost"]: row for row in rows}
    details: dict[str, Any] = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if args.dry_run:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "paper_eligible": False,
            "artifact_valid": True,
            "status": "dry_run_pending_skeleton",
            "rows": rows,
            "details": details,
        }

    if args.measure_memory:
        by_cost["resident_guidance_reuse_memory"] = measure_memory(args)
    if args.measure_storage:
        by_cost["persistent_db_storage"] = measure_storage(args)
    if args.build_proof_json is not None:
        by_cost["hnsw_build_bfs_rewrite_time"] = measure_build(args.build_proof_json)
        details["build_proof_json"] = str(args.build_proof_json)
    if args.lifecycle_csv is not None:
        by_cost["maintenance_under_writes_p95"] = measure_maintenance(args.lifecycle_csv)
        details["lifecycle_csv"] = str(args.lifecycle_csv)

    ordered = [by_cost[key] for key in (
        "resident_guidance_reuse_memory",
        "persistent_db_storage",
        "hnsw_build_bfs_rewrite_time",
        "maintenance_under_writes_p95",
    )]
    complete = all(row["sqlens"] != "Pending" and row["delta"] != "Pending" for row in ordered)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "paper_eligible": bool(args.mark_paper_eligible and complete),
        "artifact_valid": True,
        "status": "complete" if complete else "partial",
        "rows": ordered,
        "details": details,
        "release_contract": str(args.release_contract) if args.release_contract else "",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument("--measure-storage", action="store_true")
    parser.add_argument("--build-proof-json", type=Path, default=None)
    parser.add_argument("--lifecycle-csv", type=Path, default=None)
    parser.add_argument("--mark-paper-eligible", action="store_true")
    parser.add_argument("--release-contract", type=Path, default=None)
    parser.add_argument("--expected-build-id", default="")
    parser.add_argument("--pghost", default=os.environ.get("PGHOST", "127.0.0.1"))
    parser.add_argument("--pgport", type=int, default=int(os.environ.get("PGPORT", "55437")))
    parser.add_argument("--pgdatabase", default=os.environ.get("PGDATABASE", "hybrid_vector"))
    parser.add_argument("--pguser", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--pgpassword", default=os.environ.get("PGPASSWORD", "postgres"))
    parser.add_argument(
        "--stock-relation",
        default="public.amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx",
    )
    parser.add_argument(
        "--bfs-relation",
        default="public.amazon10m_hnsw_m32ef200_dupbridge_r29_bfs_idx",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/hybrid_vector_db/table10_r43/overhead/table10_r43_overhead.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = build_artifact(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: artifact[k] for k in ("status", "paper_eligible", "artifact_valid")}, indent=2))
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
