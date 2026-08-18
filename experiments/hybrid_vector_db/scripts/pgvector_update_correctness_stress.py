#!/usr/bin/env python3
"""Fail-closed MVCC correctness stress for SQLens and pgvector.

This is deliberately a *correctness* harness, not a latency benchmark.  It
creates a dedicated scratch relation from a deterministic Amazon subset,
builds a dedicated HNSW index, and then overlaps committed predicate/vector
mutations with reader transactions.  Every reader derives its exact SQL-valid
top-k inside the very same REPEATABLE READ snapshot used for the normal
pgvector hybrid statement.  Consequently the comparison never depends on
offline ground truth or on a stale view of the writer.

The script is opt-in: without ``--execute`` it only emits a protocol preview
and does not connect to PostgreSQL.  The scratch relation is dropped at the
end unless ``--keep-scratch`` is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import psycopg
from psycopg import sql

try:
    from .common_pg import pg_config_from_env
except ImportError:  # pragma: no cover - direct execution
    from common_pg import pg_config_from_env


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "hybrid_vector_db"
RUNNER_VERSION = "sqlens-mvcc-update-correctness-stress-v3-r36-identity"
ARTIFACT_SCHEMA_VERSION = "pgvector_update_correctness_stress.v3"
CORRECTNESS_CONTRACT = "paired_stock_safe_guided_exact_same_snapshot_v3"
R36_BUILD_ID = (
    "sqlens-v16-d3-sticky-rejection-mixed-predicate-reuse-d2-edge-trace-"
    "readbuffer-profile-orderchangefix-ef500k-20260729-r36"
)
R36_VECTOR_SO_SHA256 = (
    "5ab03631a5167dd56c1c74638475fec9282508c87f26218d44440b23f98f1679"
)
TIE_TOLERANCE = 1e-10
METHODS = ("stock", "sqlens_guided")
MUTATIONS = ("predicate", "vector", "insert", "delete")
GUIDED_PATHS = {"validation_only"}
STOCK_PATHS = {"stock", "stock_bypass", "fresh_stock_fallback"}
STALE_REASONS = {"stale_relation", "stale_epoch", "stale_guide"}
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class StressContractError(RuntimeError):
    """The requested run cannot truthfully be represented as a valid artifact."""


@dataclass(frozen=True)
class MutationEvent:
    sequence: int
    mutation: str
    target_id: int
    donor_id: int | None


class CommitTracker:
    """Thread-safe committed-event ledger shared by readers and writers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []

    def record(self, event: MutationEvent, writer_id: int) -> dict[str, Any]:
        with self._lock:
            item = {
                "commit": len(self._events) + 1,
                "writer_id": writer_id,
                "event_sequence": event.sequence,
                "mutation": event.mutation,
            }
            self._events.append(item)
            return dict(item)

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def evidence(self, first_exclusive: int, last_inclusive: int | None = None) -> dict[str, Any]:
        with self._lock:
            end = len(self._events) if last_inclusive is None else min(last_inclusive, len(self._events))
            selected = [dict(item) for item in self._events[first_exclusive:end]]
        counts = Counter(str(item["mutation"]) for item in selected)
        return {
            "commit_before": first_exclusive,
            "commit_after": end,
            "overlap_count": len(selected),
            "first_commit": selected[0] if selected else None,
            "last_commit": selected[-1] if selected else None,
            "mutation_counts": {name: int(counts.get(name, 0)) for name in MUTATIONS},
        }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_identity(path: Path = Path(__file__)) -> dict[str, Any]:
    """Bind the exact checked-out source revision and this runner's bytes."""
    relative = path.resolve().relative_to(ROOT)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        revision = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--", str(relative))
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise StressContractError(f"cannot establish runner Git identity: {exc}") from exc
    return {
        "git_revision": revision,
        "runner_path": str(relative),
        "runner_sha256": sha256_file(path),
        "runner_git_status": status,
        "runner_tracked_clean": status == "",
    }


def source_identity_gate(args: argparse.Namespace) -> dict[str, Any]:
    identity = git_identity()
    expected_runner = str(args.expected_runner_sha256 or "").lower()
    expected_git = str(args.expected_git_revision or "")
    runner_matches = bool(expected_runner) and identity["runner_sha256"] == expected_runner
    git_matches = bool(expected_git) and identity["git_revision"] == expected_git
    return {
        **identity,
        "expected_runner_sha256": expected_runner,
        "expected_git_revision": expected_git,
        "runner_sha256_matches_expected": runner_matches,
        "git_revision_matches_expected": git_matches,
        "passed": runner_matches and git_matches,
    }


def checked_identifier(value: str, label: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise StressContractError(f"{label} must be a simple PostgreSQL identifier: {value!r}")
    return value


def qualified_identifier(name: str) -> sql.Identifier:
    parts = name.split(".")
    if len(parts) not in {1, 2} or any(not IDENTIFIER.fullmatch(part) for part in parts):
        raise StressContractError(f"invalid relation name: {name!r}")
    return sql.Identifier(*parts)


def parse_mutation_mix(value: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in value.split(","):
        name, separator, raw_weight = item.strip().partition(":")
        if not separator or name not in MUTATIONS:
            raise argparse.ArgumentTypeError(
                "--mutation-mix must name predicate,vector,insert,delete with weights"
            )
        try:
            weight = int(raw_weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("mutation weights must be integers") from exc
        if weight < 0:
            raise argparse.ArgumentTypeError("mutation weights must be non-negative")
        out[name] = weight
    if set(out) != set(MUTATIONS) or sum(out.values()) <= 0:
        raise argparse.ArgumentTypeError("mutation mix must include all four operations with positive total")
    if out["predicate"] <= 0 or out["vector"] <= 0:
        raise argparse.ArgumentTypeError("mutation mix must include actual predicate and vector changes")
    return out


def build_mutation_schedule(
    target_ids: Sequence[int], donor_ids: Sequence[int], count: int, seed: int,
    mix: Mapping[str, int],
) -> list[MutationEvent]:
    """Build a deterministic committed-write schedule with no query-pool IDs."""
    if count <= 0 or not target_ids or not donor_ids:
        raise StressContractError("mutation schedule needs positive count and nonempty disjoint pools")
    choices = [name for name in MUTATIONS for _ in range(int(mix[name]))]
    if not choices:
        raise StressContractError("mutation schedule has no enabled operations")
    rng = random.Random(seed)
    rng.shuffle(choices)
    if "insert" in choices:
        # A lifecycle delete must remove a prior committed lifecycle insert.
        choices.remove("insert")
        choices.insert(0, "insert")
    schedule: list[MutationEvent] = []
    lifecycle_ids: list[int] = []
    for sequence in range(count):
        mutation = choices[sequence % len(choices)]
        if mutation == "delete" and not lifecycle_ids:
            mutation = "insert"
        target = int(target_ids[(sequence * 7919 + seed) % len(target_ids)])
        donor = int(donor_ids[(sequence * 104729 + seed) % len(donor_ids)])
        if mutation == "insert":
            target = -(sequence + 1)
            lifecycle_ids.append(target)
        elif mutation == "delete":
            target = lifecycle_ids.pop(0)
        schedule.append(MutationEvent(sequence, mutation, target, donor))
    if not any(event.mutation == "predicate" for event in schedule):
        schedule[0] = MutationEvent(0, "predicate", schedule[0].target_id, schedule[0].donor_id)
    if not any(event.mutation == "vector" for event in schedule):
        schedule[-1] = MutationEvent(schedule[-1].sequence, "vector", schedule[-1].target_id, schedule[-1].donor_id)
    return schedule


def partition_mutation_schedule(
    schedule: Sequence[MutationEvent], writer_clients: int,
) -> list[list[MutationEvent]]:
    """Keep lifecycle causality in one lane while predicate/vector writes overlap."""
    if writer_clients <= 0:
        raise StressContractError("writer clients must be positive")
    lanes: list[list[MutationEvent]] = [[] for _ in range(writer_clients)]
    next_lane = 0
    for event in schedule:
        if event.mutation in {"insert", "delete"}:
            lanes[0].append(event)
        else:
            lanes[next_lane % writer_clients].append(event)
            next_lane += 1
    return lanes


def exact_tie_aware_match(
    approximate: Sequence[tuple[int, float]], exact: Sequence[tuple[int, float]], k: int,
    tolerance: float = TIE_TOLERANCE,
) -> dict[str, Any]:
    """Validate complete strict-prefix retrieval while permitting boundary ties."""
    if k <= 0:
        raise StressContractError("k must be positive")
    if len(approximate) != min(k, len(exact)):
        return {"passed": False, "reason": "result_cardinality", "recall_at_k": 0.0}
    if not exact:
        return {"passed": len(approximate) == 0, "reason": "empty", "recall_at_k": 1.0}
    boundary = float(exact[min(k, len(exact)) - 1][1])
    strict_ids = {int(item_id) for item_id, distance in exact if float(distance) < boundary - tolerance}
    approximate_ids = {int(item_id) for item_id, _ in approximate}
    invalid_ids = [
        int(item_id) for item_id, distance in approximate
        if float(distance) > boundary + tolerance
    ]
    missing_strict = sorted(strict_ids - approximate_ids)
    denominator = min(k, len(exact))
    credited = sum(float(distance) <= boundary + tolerance for _, distance in approximate)
    return {
        "passed": not invalid_ids and not missing_strict and len(approximate_ids) == len(approximate),
        "reason": "" if not invalid_ids and not missing_strict else "tie_aware_result_mismatch",
        "recall_at_k": credited / denominator,
        "boundary_distance": boundary,
        "strict_required": len(strict_ids),
        "missing_strict_ids": missing_strict,
        "outside_boundary_ids": invalid_ids,
    }


def same_snapshot_contract(snapshot_before: str | None, snapshot_after: str | None) -> bool:
    """A reader may compare only results obtained from one PostgreSQL snapshot."""
    return bool(snapshot_before) and snapshot_before == snapshot_after


def classify_profile(
    method: str, guidance: Mapping[str, Any], scan: Mapping[str, Any],
    snapshot_epoch: int, relation_relfilenode: int,
) -> dict[str, Any]:
    """Fail closed unless SQLens describes a known, snapshot-safe route."""
    if method not in METHODS:
        raise StressContractError(f"unknown method: {method}")
    final_path = scan.get("final_path")
    profile_version = scan.get("profile_semantics_version")
    if (
        not isinstance(final_path, str)
        or not isinstance(profile_version, int)
        or profile_version < 10
        or scan.get("valid") is not True
    ):
        return {"passed": False, "classification": "unprofiled", "reason": "missing final_path/profile version"}
    stale_reason = str(scan.get("stock_bypass_reason") or scan.get("fallback_reason") or "")
    scan_stale = stale_reason in STALE_REASONS
    if method == "stock":
        passed = final_path in STOCK_PATHS and not bool(guidance.get("effective_active", False))
        return {
            "passed": passed, "classification": "stock", "final_path": final_path,
            "stale_fallback": scan_stale, "reason": "" if passed else "stock path/profile mismatch",
        }

    required = ("epoch_tracked", "relation_epoch", "relation_relfilenode", "guide_generation")
    if any(field not in guidance for field in required):
        return {"passed": False, "classification": "unprofiled", "reason": "incomplete guidance profile"}
    epoch_matches = int(guidance["relation_epoch"]) == int(snapshot_epoch)
    relfilenode_matches = int(guidance["relation_relfilenode"]) == int(relation_relfilenode)
    stale_guide = not epoch_matches or not relfilenode_matches or scan_stale
    is_safe_fallback = final_path in STOCK_PATHS and scan_stale
    is_guided = final_path in GUIDED_PATHS
    stale_admitted = stale_guide and is_guided
    proof_generation = scan.get("planner_proof_guide_generation")
    proof_matches = (
        scan.get("planner_proof_succeeded") is True
        and isinstance(proof_generation, int)
        and proof_generation == int(guidance["guide_generation"])
    )
    passed = (
        bool(guidance.get("epoch_tracked"))
        and not stale_admitted
        and (is_safe_fallback or (is_guided and proof_matches))
    )
    return {
        "passed": passed,
        "classification": "stale_fallback" if is_safe_fallback and stale_guide else "guided" if is_guided else "invalid",
        "final_path": final_path,
        "stale_fallback": is_safe_fallback and stale_guide,
        "stale_guide_admitted": stale_admitted,
        "guide_epoch": int(guidance["relation_epoch"]),
        "snapshot_epoch": int(snapshot_epoch),
        "guide_generation": int(guidance["guide_generation"]),
        "planner_proof_guide_generation": proof_generation,
        "planner_proof_matches_guide": proof_matches,
        "filter_strategy": "safe_guided",
        "reason": "" if passed else "stale guide admitted, planner proof mismatch, or non-safe-guided path",
    }


def ordered_ann_equivalence(
    stock: Sequence[tuple[int, float]], guided: Sequence[tuple[int, float]],
    tolerance: float = TIE_TOLERANCE,
) -> dict[str, Any]:
    if len(stock) != len(guided):
        return {"passed": False, "reason": "cardinality", "stock_count": len(stock), "guided_count": len(guided)}
    mismatches: list[dict[str, Any]] = []
    for rank, (stock_row, guided_row) in enumerate(zip(stock, guided), start=1):
        stock_id, stock_distance = stock_row
        guided_id, guided_distance = guided_row
        if stock_id != guided_id or not math.isclose(
            float(stock_distance), float(guided_distance), rel_tol=0.0, abs_tol=tolerance
        ):
            mismatches.append({
                "rank": rank, "stock_id": stock_id, "guided_id": guided_id,
                "stock_distance": stock_distance, "guided_distance": guided_distance,
            })
    return {"passed": not mismatches, "reason": "" if not mismatches else "ordered_ids_or_distances", "mismatches": mismatches[:10]}


def partition_query_lanes(query_ids: Sequence[int], reader_clients: int) -> list[list[int]]:
    if reader_clients <= 0 or len(query_ids) < reader_clients:
        raise StressContractError("each persistent reader backend needs a nonempty deterministic lane")
    return [list(query_ids[lane::reader_clients]) for lane in range(reader_clients)]


def classify_post_update_lifecycle(
    warm_epoch: int, snapshot_epoch: int, activation: Mapping[str, Any],
    profile_classification: Mapping[str, Any],
) -> dict[str, Any]:
    epoch_advanced = snapshot_epoch > warm_epoch
    fresh_epoch = int(activation.get("relation_epoch", -1)) == snapshot_epoch
    refresh_signal = any(int(activation.get(name, 0) or 0) > 0 for name in (
        "fragment_builds", "fragment_cache_misses", "fragment_store_hits"
    )) or float(activation.get("last_cache_build_ms", 0.0) or 0.0) > 0.0
    safe_fallback = bool(profile_classification.get("stale_fallback"))
    passed = epoch_advanced and (safe_fallback or (fresh_epoch and refresh_signal))
    return {
        "passed": passed,
        "mode": "safe_stale_fallback" if safe_fallback else "epoch_refresh" if fresh_epoch and refresh_signal else "missing",
        "warm_epoch": warm_epoch,
        "snapshot_epoch": snapshot_epoch,
        "epoch_advanced": epoch_advanced,
        "activation_relation_epoch": activation.get("relation_epoch"),
        "fresh_epoch": fresh_epoch,
        "refresh_signal": refresh_signal,
        "safe_stale_fallback": safe_fallback,
    }


def formal_protocol_status(args: argparse.Namespace) -> dict[str, Any]:
    checks = {
        "execute": bool(args.execute),
        "subset_rows_ge_250k": args.subset_rows >= 250_000,
        "queries_ge_1000": args.queries >= 1_000,
        "multiple_queries_per_reader": args.queries >= args.reader_clients * 2,
        "writer_clients_ge_2": args.writer_clients >= 2,
        "reader_clients_ge_4": args.reader_clients >= 4,
        "commits_ge_2000": args.writer_commits >= 2_000,
        "meaningful_overlap_gate": args.min_overlap_queries >= max(20, args.reader_clients),
        "preferred_index_guc": args.preferred_index_guc == "hnsw.preferred_index",
        "all_mutations": all(args.mutation_mix[name] > 0 for name in MUTATIONS),
        "exact_r36_build_id": args.expected_sqlens_build_id == R36_BUILD_ID,
        "exact_r36_vector_sha": args.expected_vector_so_sha256 == R36_VECTOR_SO_SHA256,
        "expected_runner_sha": bool(
            re.fullmatch(r"[0-9a-f]{64}", str(args.expected_runner_sha256 or ""))
        ),
        "expected_git_revision": bool(
            re.fullmatch(r"[0-9a-f]{40,64}", str(args.expected_git_revision or ""))
        ),
        "server_loaded_vector_identity_required": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"formal": not failed, "failed_checks": failed, "checks": checks}


def paper_eligibility(
    diagnostic_valid: bool,
    protocol: Mapping[str, Any],
    source: Mapping[str, Any],
) -> bool:
    return bool(
        diagnostic_valid
        and protocol.get("formal") is True
        and source.get("passed") is True
    )


def validate_args(args: argparse.Namespace) -> None:
    checked_identifier(args.scratch_schema, "scratch schema")
    checked_identifier(args.scratch_name, "scratch table")
    checked_identifier(args.scratch_index_name, "scratch index")
    if args.k > 100:
        raise StressContractError("k is bounded at 100 for same-snapshot exact scans")
    if args.writer_commits < args.writer_clients:
        raise StressContractError("writer commits must cover every writer client")
    if args.subset_rows < args.queries * 4:
        raise StressContractError("subset must leave disjoint query/donor/mutation pools")
    if args.queries < args.reader_clients * 2:
        raise StressContractError("each persistent reader backend must process at least two query transactions")
    if args.min_overlap_queries > args.queries:
        raise StressContractError("--min-overlap-queries cannot exceed --queries")
    if args.preferred_index_guc != "hnsw.preferred_index":
        raise StressContractError("correctness stress requires hnsw.preferred_index")
    if args.execute and args.dry_run:
        raise StressContractError("--execute and --dry-run are mutually exclusive")
    if args.execute and args.vector_so is not None and not args.vector_so.is_file():
        raise StressContractError(f"optional client vector.so mirror does not exist: {args.vector_so}")
    if args.expected_vector_so_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.expected_vector_so_sha256):
        raise StressContractError("--expected-vector-so-sha256 must be a lowercase SHA-256 digest")
    if args.expected_runner_sha256 and not re.fullmatch(
        r"[0-9a-f]{64}", args.expected_runner_sha256
    ):
        raise StressContractError("--expected-runner-sha256 must be a lowercase SHA-256 digest")
    if args.expected_git_revision and not re.fullmatch(
        r"[0-9a-f]{40,64}", args.expected_git_revision
    ):
        raise StressContractError("--expected-git-revision must be a full Git object ID")
    if args.formal and not formal_protocol_status(args)["formal"]:
        raise StressContractError(
            "--formal requires execute, 250k rows, 1000 queries, persistent readers, two writers, 2000 commits, "
            "all mutation types, exact r36 build/vector identity, and expected runner/Git identity"
        )


def relation_identity(cur: psycopg.Cursor, relation: str) -> dict[str, Any]:
    cur.execute(
        "SELECT c.oid::bigint, c.relfilenode::bigint, c.reltuples::bigint, pg_relation_size(c.oid) "
        "FROM pg_class c WHERE c.oid = %s::regclass",
        (relation,),
    )
    row = cur.fetchone()
    if row is None:
        raise StressContractError(f"relation does not exist: {relation}")
    return {"name": relation, "oid": int(row[0]), "relfilenode": int(row[1]), "reltuples": int(row[2]), "bytes": int(row[3])}


def runtime_identity(
    cur: psycopg.Cursor,
    expected_build_id: str,
    expected_vector_sha256: str,
    client_vector_so: Path | None = None,
) -> dict[str, Any]:
    cur.execute(
        "WITH lib AS ("
        "SELECT setting || '/vector.so' AS path FROM pg_config WHERE name = 'PKGLIBDIR'"
        ") SELECT vector_sqlens_build_id(), current_setting('server_version'), "
        "(SELECT extversion FROM pg_extension WHERE extname = 'vector'), "
        "lib.path, encode(sha256(pg_read_binary_file(lib.path)), 'hex') FROM lib"
    )
    row = cur.fetchone()
    if row is None or not row[0] or not row[3]:
        raise StressContractError("server-loaded SQLens/vector.so identity is unavailable")
    build_id = str(row[0])
    if expected_build_id and build_id != expected_build_id:
        raise StressContractError(f"SQLens build ID mismatch: expected {expected_build_id!r}, got {build_id!r}")
    loaded_path = str(row[3])
    loaded_sha = str(row[4] or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", loaded_sha):
        raise StressContractError("server-loaded vector.so SHA256 is unavailable or malformed")
    if expected_vector_sha256 and loaded_sha != expected_vector_sha256:
        raise StressContractError(
            "server-loaded vector.so SHA256 does not match --expected-vector-so-sha256"
        )
    client_path = str(client_vector_so.resolve()) if client_vector_so is not None else None
    client_sha = sha256_file(client_vector_so) if client_vector_so is not None else None
    client_matches = client_sha == loaded_sha if client_sha is not None else None
    if client_matches is False:
        raise StressContractError(
            "optional client vector.so mirror does not match the server-loaded vector.so"
        )
    return {
        "sqlens_build_id": build_id,
        "postgres_version": str(row[1]),
        "vector_extension_version": str(row[2] or ""),
        "vector_so": loaded_path,
        "vector_so_sha256": loaded_sha,
        "vector_so_identity_source": "server_pg_config_pkglibdir_pg_read_binary_file",
        "client_vector_so": client_path,
        "client_vector_so_sha256": client_sha,
        "client_vector_so_matches_loaded": client_matches,
    }


def create_scratch(cur: psycopg.Cursor, args: argparse.Namespace) -> dict[str, Any]:
    source = qualified_identifier(args.source_table)
    scratch = sql.Identifier(args.scratch_schema, args.scratch_name)
    index = sql.Identifier(args.scratch_index_name)
    source_identity = relation_identity(cur, args.source_table)
    cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(args.scratch_schema)))
    cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(scratch))
    # TABLESAMPLE is not deterministic across PostgreSQL versions.  A stable id hash
    # gives a reproducible subset while preserving a genuinely sampled Amazon slice.
    cur.execute(
        sql.SQL(
            "CREATE TABLE {} AS "
            "SELECT id::bigint, embedding, embedding_valid, has_price, price, "
            "rating, main_category, helpful_vote, review_text_len, false::boolean AS lifecycle_row "
            "FROM {} WHERE embedding_valid "
            "ORDER BY hashint8extended(id::bigint, %s) LIMIT %s"
        ).format(scratch, source),
        (args.seed, args.subset_rows),
    )
    cur.execute(sql.SQL("ALTER TABLE {} ADD PRIMARY KEY (id)").format(scratch))
    # CREATE INDEX infers the schema from the target table; PostgreSQL does not
    # accept a schema-qualified index name in this position.
    cur.execute(sql.SQL("CREATE INDEX {} ON {} USING hnsw (embedding vector_l2_ops) WHERE embedding_valid").format(index, scratch))
    cur.execute(sql.SQL("ANALYZE {}").format(scratch))
    cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", (f"{args.scratch_schema}.{args.scratch_name}",))
    cur.fetchone()
    return {
        "source": source_identity,
        "table": relation_identity(cur, f"{args.scratch_schema}.{args.scratch_name}"),
        "index": relation_identity(cur, f"{args.scratch_schema}.{args.scratch_index_name}"),
    }


def drop_scratch(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(args.scratch_schema, args.scratch_name)))


def load_pools(cur: psycopg.Cursor, args: argparse.Namespace) -> tuple[list[int], list[int], list[int]]:
    table = qualified_identifier(f"{args.scratch_schema}.{args.scratch_name}")
    cur.execute(sql.SQL("SELECT id FROM {} WHERE embedding_valid ORDER BY hashint8extended(id, %s)").format(table), (args.seed + 1,))
    ids = [int(row[0]) for row in cur.fetchall()]
    needed = args.queries * 3 + args.writer_clients * 4
    if len(ids) < needed:
        raise StressContractError(f"scratch table has {len(ids)} valid vectors, needs at least {needed}")
    queries = ids[:args.queries]
    donors = ids[args.queries: args.queries * 2]
    targets = ids[args.queries * 2:]
    if set(queries) & set(donors) or set(queries) & set(targets) or set(donors) & set(targets):
        raise StressContractError("query, donor, and mutation pools must be disjoint")
    return queries, donors, targets


GUIDANCE_ATOMS = ["sql:has_price AND price <= 20"]
REUSE_COUNTERS = (
    "fast_reactivation_hits",
    "fragment_cache_hits",
    "fragment_store_hits",
    "composed_guide_hits",
    "composed_exact_hit",
)


def configure_common_session(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    cur.execute("SET jit = off")
    cur.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(args.ef_search),))
    cur.execute("SET hnsw.iterative_scan = strict_order")
    cur.execute("SELECT set_config('hnsw.max_scan_tuples', %s, false)", (str(args.max_scan_tuples),))
    cur.execute("SELECT set_config('hnsw.scan_mem_multiplier', %s, false)", (str(args.scan_mem_multiplier),))
    cur.execute("SELECT set_config('statement_timeout', %s, false)", (str(args.statement_timeout_ms),))


def configure_reader_session(cur: psycopg.Cursor, args: argparse.Namespace) -> dict[str, Any]:
    configure_common_session(cur, args)
    index_name = f"{args.scratch_schema}.{args.scratch_index_name}"
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET enable_bitmapscan = off")
    cur.execute("SELECT set_config(%s, %s, false)", (args.preferred_index_guc, index_name))
    cur.execute(
        "SELECT pg_backend_pid(), current_setting(%s, true), "
        "current_setting(%s, true)::regclass = %s::regclass",
        (args.preferred_index_guc, args.preferred_index_guc, index_name),
    )
    row = cur.fetchone()
    if row is None or row[1] is None or row[2] is not True:
        raise StressContractError("hnsw.preferred_index is unavailable or did not bind the scratch HNSW index")
    return {"backend_pid": int(row[0]), "preferred_index_current_setting": str(row[1]), "preferred_index_matches": True}


def read_json_function(cur: psycopg.Cursor, name: str) -> dict[str, Any]:
    cur.execute(f"SELECT {name}()")
    row = cur.fetchone()
    raw = row[0] if row else None
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise StressContractError(f"{name} returned no JSON profile")
    return value


def epoch_in_snapshot(cur: psycopg.Cursor, table: str) -> tuple[int, int]:
    cur.execute(
        "SELECT e.epoch, c.relfilenode::bigint FROM public.pgvector_hnsw_fragment_epoch e "
        "JOIN pg_class c ON c.oid=e.heap_oid WHERE e.heap_oid=%s::regclass",
        (table,),
    )
    row = cur.fetchone()
    if row is None:
        raise StressContractError("scratch relation has no SQLens fragment epoch")
    return int(row[0]), int(row[1])


def hybrid_statement(table: sql.Identifier, guided: bool) -> sql.Composed:
    binding = (
        sql.SQL("(SELECT vector_hnsw_guidance_bind(%s::regclass, %s::text[], 'exact') OFFSET 0) AND ")
        if guided else sql.SQL("")
    )
    return sql.SQL(
        "SELECT c.id, c.embedding <-> (SELECT q.embedding FROM {table} AS q WHERE q.id = %s) AS distance "
        "FROM {table} AS c WHERE {binding}embedding_valid AND has_price AND price <= 20 AND c.id <> %s "
        "ORDER BY distance LIMIT %s"
    ).format(table=table, binding=binding)


def exact_statement(table: sql.Identifier) -> sql.Composed:
    return sql.SQL(
        "SELECT c.id, c.embedding <-> (SELECT q.embedding FROM {table} AS q WHERE q.id = %s) AS distance "
        "FROM {table} AS c WHERE embedding_valid AND has_price AND price <= 20 AND c.id <> %s "
        "ORDER BY distance, c.id LIMIT %s"
    ).format(table=table)


def hnsw_params(query_id: int, k: int, index_name: str, guided: bool) -> tuple[Any, ...]:
    if guided:
        return (query_id, index_name, GUIDANCE_ATOMS, query_id, k)
    return (query_id, query_id, k)


def plan_index_names(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("Index Name"), str):
            names.append(str(value["Index Name"]))
        for child in value.values():
            names.extend(plan_index_names(child))
    elif isinstance(value, list):
        for child in value:
            names.extend(plan_index_names(child))
    return names


def explain_index_plan_gate(cur: psycopg.Cursor, args: argparse.Namespace, query_id: int) -> dict[str, Any]:
    table = qualified_identifier(f"{args.scratch_schema}.{args.scratch_name}")
    index_name = f"{args.scratch_schema}.{args.scratch_index_name}"
    expected_leaf = args.scratch_index_name
    gates: dict[str, Any] = {}
    for guided, arm in ((False, "stock"), (True, "sqlens_guided")):
        if guided:
            cur.execute("SET hnsw.filter_strategy = safe_guided")
            cur.execute(
                "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], 'exact')",
                (index_name, GUIDANCE_ATOMS),
            )
            row = cur.fetchone()
            if row is None or int(row[0]) != 1:
                raise StressContractError("plan gate could not activate the exact SQL atom")
        else:
            cur.execute("SELECT vector_hnsw_guidance_reset()")
            cur.execute("SET hnsw.filter_strategy = off")
        cur.execute(
            sql.SQL("EXPLAIN (FORMAT JSON, VERBOSE) ") + hybrid_statement(table, guided),
            hnsw_params(query_id, args.k, index_name, guided),
        )
        plan = cur.fetchone()[0]
        if isinstance(plan, str):
            plan = json.loads(plan)
        names = plan_index_names(plan)
        passed = expected_leaf in names
        gates[arm] = {"passed": passed, "expected_index": index_name, "observed_index_names": names, "plan": plan}
        if not passed:
            raise StressContractError(f"{arm} EXPLAIN did not use {index_name}")
    return gates


def execute_hnsw(cur: psycopg.Cursor, args: argparse.Namespace, query_id: int, guided: bool) -> tuple[list[tuple[int, float]], dict[str, Any]]:
    table = qualified_identifier(f"{args.scratch_schema}.{args.scratch_name}")
    index_name = f"{args.scratch_schema}.{args.scratch_index_name}"
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(hybrid_statement(table, guided), hnsw_params(query_id, args.k, index_name, guided))
    rows = [(int(row[0]), float(row[1])) for row in cur.fetchall()]
    return rows, read_json_function(cur, "vector_hnsw_last_scan_profile")


def activate_exact_guide(cur: psycopg.Cursor, args: argparse.Namespace) -> dict[str, Any]:
    index_name = f"{args.scratch_schema}.{args.scratch_index_name}"
    cur.execute("SET LOCAL hnsw.filter_strategy = safe_guided")
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], 'exact')",
        (index_name, GUIDANCE_ATOMS),
    )
    row = cur.fetchone()
    if row is None or int(row[0]) != 1:
        raise StressContractError("SQLens guidance activation did not materialize exactly one SQL atom")
    return read_json_function(cur, "vector_hnsw_guidance_profile")


def pre_update_warmup(cur: psycopg.Cursor, args: argparse.Namespace, query_id: int) -> dict[str, Any]:
    table_name = f"{args.scratch_schema}.{args.scratch_name}"
    attempts: list[dict[str, Any]] = []
    for iteration in range(2):
        cur.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        cur.execute("SELECT txid_current_snapshot()")
        snapshot = str(cur.fetchone()[0])
        epoch, relfilenode = epoch_in_snapshot(cur, table_name)
        activation = activate_exact_guide(cur, args)
        rows, scan = execute_hnsw(cur, args, query_id, True)
        guidance_after = read_json_function(cur, "vector_hnsw_guidance_profile")
        classification = classify_profile(
            "sqlens_guided", guidance_after, scan, epoch, relfilenode
        )
        cur.execute("COMMIT")
        attempts.append({
            "iteration": iteration, "snapshot": snapshot, "epoch": epoch,
            "relation_relfilenode": relfilenode, "activation_profile": activation,
            "guidance_profile": guidance_after, "scan_profile": scan,
            "profile_classification": classification, "returned": len(rows),
        })
    first = attempts[0]["guidance_profile"]
    second = attempts[1]["guidance_profile"]
    deltas = {
        name: int(second.get(name, 0) or 0) - int(first.get(name, 0) or 0)
        for name in REUSE_COUNTERS if not isinstance(second.get(name), bool)
    }
    boolean_reuse = bool(second.get("composed_exact_hit", False) or second.get("composed_guide_hit", False))
    reused = (
        (boolean_reuse or any(value > 0 for value in deltas.values()))
        and all(bool(attempt["profile_classification"].get("passed")) for attempt in attempts)
    )
    return {
        "attempts": attempts, "reuse_deltas": deltas, "pre_update_guided_reuse": reused,
        "warm_epoch": int(attempts[-1]["epoch"]),
        "warm_guide_epoch": int(second.get("relation_epoch", -1)),
    }


def sql_valid_rows(cur: psycopg.Cursor, args: argparse.Namespace, rows: Sequence[tuple[int, float]]) -> bool:
    if len({item_id for item_id, _ in rows}) != len(rows):
        return False
    if not rows:
        return True
    table = qualified_identifier(f"{args.scratch_schema}.{args.scratch_name}")
    cur.execute(
        sql.SQL("SELECT count(*) FROM {} WHERE id = ANY(%s) AND embedding_valid AND has_price AND price <= 20").format(table),
        ([item_id for item_id, _ in rows],),
    )
    return int(cur.fetchone()[0]) == len(rows)


def run_paired_snapshot_request(
    conn: psycopg.Connection, args: argparse.Namespace, query_id: int,
    backend_id: int, warm_epoch: int, commits: CommitTracker,
) -> dict[str, Any]:
    table_name = f"{args.scratch_schema}.{args.scratch_name}"
    table = qualified_identifier(table_name)
    cur = conn.cursor()
    started = time.perf_counter()
    commit_before = commits.count()
    try:
        cur.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        cur.execute("SELECT txid_current_snapshot()")
        snapshot_before = str(cur.fetchone()[0])
        commit_before = commits.count()
        epoch, relfilenode = epoch_in_snapshot(cur, table_name)
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        cur.execute("SET LOCAL hnsw.filter_strategy = off")
        stock, stock_scan = execute_hnsw(cur, args, query_id, False)
        stock_classification = classify_profile("stock", {"effective_active": False}, stock_scan, epoch, relfilenode)

        activation = activate_exact_guide(cur, args)
        guided, guided_scan = execute_hnsw(cur, args, query_id, True)
        guidance_after = read_json_function(cur, "vector_hnsw_guidance_profile")
        guided_classification = classify_profile("sqlens_guided", activation, guided_scan, epoch, relfilenode)

        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_indexonlyscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        cur.execute("SET LOCAL enable_seqscan = on")
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        cur.execute(
            "SELECT current_setting('enable_indexscan'), current_setting('enable_indexonlyscan'), "
            "current_setting('enable_bitmapscan'), current_setting('enable_seqscan')"
        )
        exact_scan_gucs = tuple(str(value) for value in cur.fetchone())
        if exact_scan_gucs != ("off", "off", "off", "on"):
            raise StressContractError("exact branch could not enforce a sequential-scan planner contract")
        cur.execute(exact_statement(table), (query_id, query_id, args.k))
        exact = [(int(row[0]), float(row[1])) for row in cur.fetchall()]
        stock_sql_valid = sql_valid_rows(cur, args, stock)
        guided_sql_valid = sql_valid_rows(cur, args, guided)
        cur.execute("SELECT txid_current_snapshot()")
        snapshot_after = str(cur.fetchone()[0])
        stock_diagnostic = exact_tie_aware_match(stock, exact, args.k)
        guided_diagnostic = exact_tie_aware_match(guided, exact, args.k)
        equivalence = ordered_ann_equivalence(stock, guided)
        recall_equal = math.isclose(
            float(stock_diagnostic["recall_at_k"]), float(guided_diagnostic["recall_at_k"]),
            rel_tol=0.0, abs_tol=1e-15,
        )
        snapshot_matches = same_snapshot_contract(snapshot_before, snapshot_after)
        cur.execute("ROLLBACK")
        commit_after = commits.count()
        overlap = commits.evidence(commit_before, commit_after)
        post_update_lifecycle = classify_post_update_lifecycle(
            warm_epoch, epoch, activation, guided_classification
        )
        passed = (
            bool(stock_classification["passed"])
            and bool(guided_classification["passed"])
            and bool(equivalence["passed"])
            and recall_equal
            and stock_sql_valid
            and guided_sql_valid
            and snapshot_matches
        )
        return {
            "backend_id": backend_id, "query_id": query_id, "snapshot_before": snapshot_before,
            "snapshot_after": snapshot_after, "snapshot_matches": snapshot_matches,
            "snapshot_epoch": epoch, "relation_relfilenode": relfilenode,
            "stock": stock, "guided": guided, "exact": exact,
            "exact_scan_gucs": exact_scan_gucs,
            "stock_scan_profile": stock_scan, "guided_scan_profile": guided_scan,
            "guidance_activation_profile": activation, "guidance_after_profile": guidance_after,
            "stock_profile_classification": stock_classification,
            "guided_profile_classification": guided_classification,
            "ordered_stock_guided_equivalence": equivalence,
            "stock_exact_recall": stock_diagnostic, "guided_exact_recall": guided_diagnostic,
            "same_tie_aware_recall": recall_equal,
            "stock_sql_valid": stock_sql_valid, "guided_sql_valid": guided_sql_valid,
            "search_budget": {"ef_search": args.ef_search, "max_scan_tuples": args.max_scan_tuples, "iterative_scan": "strict_order"},
            "filter_strategy": "safe_guided", "guidance_atoms": GUIDANCE_ATOMS, "guidance_kind": "exact",
            "warm_epoch": warm_epoch,
            "epoch_advanced_since_warm": post_update_lifecycle["epoch_advanced"],
            "post_update_lifecycle": post_update_lifecycle,
            "post_update_refresh_or_safe_fallback": post_update_lifecycle["passed"],
            "commit_overlap": overlap, "passed": passed,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
    except BaseException as exc:
        try:
            cur.execute("ROLLBACK")
        except Exception:
            pass
        return {
            "backend_id": backend_id, "query_id": query_id, "passed": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "commit_overlap": commits.evidence(commit_before, commits.count()),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
    finally:
        cur.close()


def apply_mutation(cur: psycopg.Cursor, args: argparse.Namespace, event: MutationEvent) -> int:
    table = qualified_identifier(f"{args.scratch_schema}.{args.scratch_name}")
    if event.mutation == "predicate":
        cur.execute(
            sql.SQL("UPDATE {} SET has_price = NOT has_price, price = CASE WHEN has_price THEN NULL ELSE 9.99 END WHERE id = %s").format(table),
            (event.target_id,),
        )
    elif event.mutation == "vector":
        cur.execute(
            sql.SQL("UPDATE {} AS target SET embedding = donor.embedding FROM {} AS donor WHERE target.id = %s AND donor.id = %s AND target.embedding <> donor.embedding").format(table, table),
            (event.target_id, event.donor_id),
        )
    elif event.mutation == "insert":
        cur.execute(
            sql.SQL("INSERT INTO {} (id, embedding, embedding_valid, has_price, price, rating, main_category, helpful_vote, review_text_len, lifecycle_row) "
                    "SELECT -(%s::bigint + 1), embedding, embedding_valid, has_price, price, rating, main_category, helpful_vote, review_text_len, true FROM {} WHERE id = %s "
                    "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, has_price = EXCLUDED.has_price, price = EXCLUDED.price, lifecycle_row = true").format(table, table),
            (event.sequence, event.donor_id),
        )
    elif event.mutation == "delete":
        cur.execute(sql.SQL("DELETE FROM {} WHERE id = %s AND lifecycle_row").format(table), (event.target_id,))
    else:  # pragma: no cover - guarded by schedule construction
        raise StressContractError(f"unsupported mutation: {event.mutation}")
    return int(cur.rowcount)


def wait_barrier(barrier: threading.Barrier) -> None:
    try:
        barrier.wait(timeout=60)
    except threading.BrokenBarrierError as exc:
        raise StressContractError("persistent reader/writer start barrier broke") from exc


def writer_worker(
    writer_id: int, args: argparse.Namespace, events: Iterable[MutationEvent],
    barrier: threading.Barrier, commits: CommitTracker, errors: queue.Queue[str],
) -> None:
    try:
        with psycopg.connect(pg_config_from_env().conninfo, autocommit=False) as conn:
            cur = conn.cursor()
            configure_common_session(cur, args)
            wait_barrier(barrier)
            for event in events:
                affected = apply_mutation(cur, args, event)
                if affected != 1:
                    raise StressContractError(
                        f"{event.mutation} event {event.sequence} did not change exactly one scratch row"
                    )
                conn.commit()
                commits.record(event, writer_id)
                if args.writer_delay_ms > 0:
                    time.sleep(args.writer_delay_ms / 1000.0)
            cur.close()
    except BaseException as exc:
        errors.put(f"writer-{writer_id}: {exc.__class__.__name__}: {exc}")


def persistent_reader_worker(
    backend_id: int, args: argparse.Namespace, query_ids: Sequence[int],
    barrier: threading.Barrier, commits: CommitTracker,
) -> dict[str, Any]:
    backend: dict[str, Any] = {"backend_id": backend_id, "query_ids": list(query_ids), "records": []}
    barrier_crossed = False
    try:
        with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            backend.update(configure_reader_session(cur, args))
            backend["plan_gates"] = explain_index_plan_gate(cur, args, int(query_ids[0]))
            backend["pre_update_lifecycle"] = pre_update_warmup(cur, args, int(query_ids[0]))
            wait_barrier(barrier)
            barrier_crossed = True
            backend["reader_interval_start_commit"] = commits.count()
            for lane_query_index, query_id in enumerate(query_ids):
                record = run_paired_snapshot_request(
                    conn, args, int(query_id), backend_id,
                    int(backend["pre_update_lifecycle"]["warm_epoch"]), commits,
                )
                record["backend_pid"] = backend["backend_pid"]
                record["lane_query_index"] = lane_query_index
                backend["records"].append(record)
            backend["reader_interval_end_commit"] = commits.count()
            backend["reader_interval"] = commits.evidence(
                int(backend["reader_interval_start_commit"]),
                int(backend["reader_interval_end_commit"]),
            )
            backend["post_update_lifecycle_evidence"] = any(
                bool(record.get("post_update_refresh_or_safe_fallback"))
                for record in backend["records"]
            )
            backend["post_update_lifecycle"] = [
                {
                    "query_id": record.get("query_id"),
                    "warm_epoch": record.get("warm_epoch"),
                    "snapshot_epoch": record.get("snapshot_epoch"),
                    "epoch_advanced_since_warm": record.get("epoch_advanced_since_warm"),
                    "lifecycle_classification": record.get("post_update_lifecycle"),
                    "activation_relation_epoch": record.get("guidance_activation_profile", {}).get("relation_epoch"),
                    "fragment_builds": record.get("guidance_activation_profile", {}).get("fragment_builds"),
                    "fragment_store_hits": record.get("guidance_activation_profile", {}).get("fragment_store_hits"),
                    "fragment_cache_misses": record.get("guidance_activation_profile", {}).get("fragment_cache_misses"),
                    "profile_classification": record.get("guided_profile_classification"),
                    "commit_overlap": record.get("commit_overlap"),
                }
                for record in backend["records"]
                if record.get("post_update_refresh_or_safe_fallback")
            ]
            backend["passed"] = (
                bool(backend["pre_update_lifecycle"].get("pre_update_guided_reuse"))
                and bool(backend["post_update_lifecycle_evidence"])
                and all(bool(gate.get("passed")) for gate in backend["plan_gates"].values())
                and all(bool(record.get("passed")) for record in backend["records"])
            )
            cur.close()
            return backend
    except BaseException as exc:
        if not barrier_crossed:
            try:
                wait_barrier(barrier)
            except Exception:
                pass
        backend.update({"passed": False, "error": f"reader-{backend_id}: {exc.__class__.__name__}: {exc}"})
        return backend


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def artifact_validity(
    records: Sequence[Mapping[str, Any]], backends: Sequence[Mapping[str, Any]],
    writer_errors: Sequence[str], committed: Mapping[str, int], requested_commits: int,
    reader_interval: Mapping[str, Any], expected_backends: int, expected_queries: int,
    min_overlap_queries: int,
) -> dict[str, Any]:
    failures: list[str] = []
    if writer_errors:
        failures.append("writer_error")
    if sum(int(value) for value in committed.values()) != requested_commits:
        failures.append("commit_count_mismatch")
    if any(int(committed.get(mutation, 0)) <= 0 for mutation in MUTATIONS):
        failures.append("mutation_mix_not_realized")
    if not records:
        failures.append("no_reader_records")
    if len(records) != expected_queries:
        failures.append("query_count_mismatch")
    backend_pids = [int(backend.get("backend_pid", 0)) for backend in backends]
    if len(set(backend_pids)) != expected_backends:
        failures.append("persistent_backend_pid_mismatch")
    if len(backends) != expected_backends or any(
        not backend.get("passed")
        or backend.get("post_update_lifecycle_evidence") is not True
        or not isinstance(backend.get("pre_update_lifecycle"), Mapping)
        or backend.get("pre_update_lifecycle", {}).get("pre_update_guided_reuse") is not True
        or not isinstance(backend.get("post_update_lifecycle"), list)
        or not backend.get("post_update_lifecycle")
        or int(backend.get("backend_pid", 0)) <= 0
        for backend in backends
    ):
        failures.append("persistent_backend_lifecycle")
    if any(int(reader_interval.get("mutation_counts", {}).get(mutation, 0)) <= 0 for mutation in MUTATIONS):
        failures.append("reader_interval_missing_mutation_type")
    overlap_queries = sum(
        int(record.get("commit_overlap", {}).get("overlap_count", 0)) > 0
        for record in records
    )
    if overlap_queries < min_overlap_queries:
        failures.append("insufficient_query_commit_overlap")
    for record in records:
        if not record.get("passed"):
            failures.append("reader_correctness_failure")
            break
        if record.get("ordered_stock_guided_equivalence", {}).get("passed") is not True:
            failures.append("stock_guided_ordered_mismatch")
            break
        if record.get("same_tie_aware_recall") is not True:
            failures.append("stock_guided_recall_mismatch")
            break
        if record.get("stock_sql_valid") is not True or record.get("guided_sql_valid") is not True:
            failures.append("sql_invalid_result")
            break
        if tuple(record.get("exact_scan_gucs", ())) != ("off", "off", "off", "on"):
            failures.append("exact_seq_scan_contract")
            break
        if int(record.get("backend_pid", 0)) not in set(backend_pids):
            failures.append("query_backend_pid_mismatch")
            break
        classification = record.get("guided_profile_classification", {})
        if classification.get("stale_guide_admitted"):
            failures.append("stale_guide_admitted")
            break
        if not record.get("snapshot_matches"):
            failures.append("snapshot_mismatch")
            break
    return {
        "artifact_valid": not failures,
        "failed_gates": sorted(set(failures)),
        "overlap_queries": overlap_queries,
        "minimum_overlap_queries": min_overlap_queries,
        "reader_interval_mutation_counts": dict(reader_interval.get("mutation_counts", {})),
    }


def correctness_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stock_recalls = [float(record.get("stock_exact_recall", {}).get("recall_at_k", 0.0)) for record in records]
    guided_recalls = [float(record.get("guided_exact_recall", {}).get("recall_at_k", 0.0)) for record in records]
    return {
        "paired_requests": len(records),
        "ordered_equivalent": sum(
            record.get("ordered_stock_guided_equivalence", {}).get("passed") is True
            for record in records
        ),
        "same_tie_aware_recall": sum(record.get("same_tie_aware_recall") is True for record in records),
        "stock_sql_valid": sum(record.get("stock_sql_valid") is True for record in records),
        "guided_sql_valid": sum(record.get("guided_sql_valid") is True for record in records),
        "stock_recall_mean": sum(stock_recalls) / len(stock_recalls) if stock_recalls else 0.0,
        "guided_recall_mean": sum(guided_recalls) / len(guided_recalls) if guided_recalls else 0.0,
        "max_recall_delta": max(
            (abs(stock - guided) for stock, guided in zip(stock_recalls, guided_recalls)),
            default=0.0,
        ),
        "recall_one_required": False,
    }


def manifest_query_diagnostics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for record in records:
        diagnostics.append({
            "backend_id": record.get("backend_id"),
            "backend_pid": record.get("backend_pid"),
            "query_id": record.get("query_id"),
            "snapshot": record.get("snapshot_before"),
            "snapshot_matches": record.get("snapshot_matches"),
            "stock_ordered_sha256": canonical_sha256(record.get("stock", [])),
            "guided_ordered_sha256": canonical_sha256(record.get("guided", [])),
            "ordered_equivalence": record.get("ordered_stock_guided_equivalence"),
            "stock_exact_recall": record.get("stock_exact_recall"),
            "guided_exact_recall": record.get("guided_exact_recall"),
            "same_tie_aware_recall": record.get("same_tie_aware_recall"),
            "stock_sql_valid": record.get("stock_sql_valid"),
            "guided_sql_valid": record.get("guided_sql_valid"),
            "exact_scan_gucs": record.get("exact_scan_gucs"),
            "commit_overlap": record.get("commit_overlap"),
        })
    return diagnostics


def validate_manifest_payload(manifest: Mapping[str, Any]) -> list[str]:
    """Validate the immutable evidence needed to interpret an artifact later."""
    errors: list[str] = []
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        errors.append("schema_version")
    runtime = manifest.get("runtime")
    runtime_fields = runtime if isinstance(runtime, Mapping) else {}
    if not isinstance(runtime, Mapping) or not runtime.get("sqlens_build_id"):
        errors.append("runtime_build_id")
    elif not re.fullmatch(r"[0-9a-f]{64}", str(runtime.get("vector_so_sha256", ""))):
        errors.append("runtime_vector_so_sha256")
    elif runtime.get("vector_so_identity_source") != (
        "server_pg_config_pkglibdir_pg_read_binary_file"
    ):
        errors.append("runtime_vector_so_identity_source")
    binding = manifest.get("runtime_identity_binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("loaded_sqlens_build_id") != runtime_fields.get("sqlens_build_id")
        or binding.get("loaded_vector_so_sha256") != runtime_fields.get("vector_so_sha256")
        or binding.get("build_id_matches_expected") is not True
        or binding.get("vector_so_sha256_matches_expected") is not True
    ):
        errors.append("runtime_identity_binding")
    source = manifest.get("source_identity")
    if (
        not isinstance(source, Mapping)
        or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("runner_sha256", "")))
        or not re.fullmatch(r"[0-9a-f]{40,64}", str(source.get("git_revision", "")))
        or source.get("runner_sha256_matches_expected") is not True
        or source.get("git_revision_matches_expected") is not True
    ):
        errors.append("source_identity")
    scratch = manifest.get("scratch")
    if not isinstance(scratch, Mapping):
        errors.append("scratch_identity")
    else:
        for role in ("source", "table", "index"):
            identity = scratch.get(role)
            if not isinstance(identity, Mapping) or int(identity.get("oid", 0)) <= 0 or int(identity.get("relfilenode", 0)) <= 0:
                errors.append(f"scratch_{role}_identity")
    raw = manifest.get("raw_output")
    if not isinstance(raw, Mapping) or not re.fullmatch(r"[0-9a-f]{64}", str(raw.get("sha256", ""))):
        errors.append("raw_output_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("records_sha256", ""))):
        errors.append("records_sha256")
    gates = manifest.get("artifact_gates")
    if not isinstance(gates, Mapping) or gates.get("artifact_valid") is not True or manifest.get("artifact_valid") is not True:
        errors.append("artifact_valid_gate")
    backends = manifest.get("backend_lifecycles")
    backend_invalid = not isinstance(backends, list) or not backends
    if isinstance(backends, list):
        for item in backends:
            pre = item.get("pre_update_lifecycle") if isinstance(item, Mapping) else None
            if (
                not isinstance(item, Mapping)
                or int(item.get("backend_pid", 0)) <= 0
                or not isinstance(pre, Mapping)
                or pre.get("pre_update_guided_reuse") is not True
                or item.get("post_update_lifecycle_evidence") is not True
                or not isinstance(item.get("post_update_lifecycle"), list)
                or not item.get("post_update_lifecycle")
            ):
                backend_invalid = True
                break
    if backend_invalid:
        errors.append("backend_lifecycle_evidence")
    overlap = manifest.get("reader_interval_overlap")
    if not isinstance(overlap, Mapping) or int(overlap.get("overlap_count", 0)) <= 0:
        errors.append("reader_interval_overlap")
    summary = manifest.get("correctness_summary")
    if (
        not isinstance(summary, Mapping)
        or int(summary.get("paired_requests", 0)) <= 0
        or summary.get("ordered_equivalent") != summary.get("paired_requests")
        or summary.get("same_tie_aware_recall") != summary.get("paired_requests")
        or summary.get("recall_one_required") is not False
    ):
        errors.append("paired_correctness_summary")
    diagnostics = manifest.get("query_diagnostics")
    if (
        not isinstance(diagnostics, list)
        or not isinstance(summary, Mapping)
        or len(diagnostics) != int(summary.get("paired_requests", -1))
        or any(
            not isinstance(item, Mapping)
            or int(item.get("backend_pid", 0)) <= 0
            or item.get("snapshot_matches") is not True
            or item.get("ordered_equivalence", {}).get("passed") is not True
            or item.get("same_tie_aware_recall") is not True
            or item.get("stock_sql_valid") is not True
            or item.get("guided_sql_valid") is not True
            or tuple(item.get("exact_scan_gucs", ())) != ("off", "off", "off", "on")
            for item in diagnostics
        )
    ):
        errors.append("query_diagnostics")
    return errors


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    source_identity = source_identity_gate(args)
    if args.dry_run or not args.execute:
        return {
            "artifact": "pgvector_update_correctness_stress_preview",
            "database_connected": False, "runner_version": RUNNER_VERSION,
            "formal_protocol": formal_protocol_status(args),
            "source_identity": source_identity,
            "contract": CORRECTNESS_CONTRACT, "mutation_mix": args.mutation_mix,
            "scratch": f"{args.scratch_schema}.{args.scratch_name}",
        }
    if args.formal and not source_identity["passed"]:
        raise StressContractError(
            "formal run source identity does not match runner/Git expectations"
        )
    table_name = f"{args.scratch_schema}.{args.scratch_name}"
    index_name = f"{args.scratch_schema}.{args.scratch_index_name}"
    setup: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    backend_lifecycles: list[dict[str, Any]] = []
    commits = CommitTracker()
    errors: queue.Queue[str] = queue.Queue()
    try:
        with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as admin:
            cur = admin.cursor()
            runtime = runtime_identity(
                cur,
                args.expected_sqlens_build_id,
                args.expected_vector_so_sha256,
                args.vector_so,
            )
            runtime_identity_binding = {
                "expected_sqlens_build_id": args.expected_sqlens_build_id,
                "expected_vector_so_sha256": args.expected_vector_so_sha256,
                "loaded_sqlens_build_id": runtime["sqlens_build_id"],
                "loaded_vector_so_sha256": runtime["vector_so_sha256"],
                "build_id_matches_expected": (
                    not args.expected_sqlens_build_id
                    or runtime["sqlens_build_id"] == args.expected_sqlens_build_id
                ),
                "vector_so_sha256_matches_expected": (
                    not args.expected_vector_so_sha256
                    or runtime["vector_so_sha256"] == args.expected_vector_so_sha256
                ),
            }
            setup = create_scratch(cur, args)
            query_ids, donor_ids, target_ids = load_pools(cur, args)
            cur.close()
        schedule = build_mutation_schedule(target_ids, donor_ids, args.writer_commits, args.seed + 17, args.mutation_mix)
        partitions = partition_mutation_schedule(schedule, args.writer_clients)
        query_lanes = partition_query_lanes(query_ids, args.reader_clients)
        barrier = threading.Barrier(args.writer_clients + args.reader_clients)
        reader_interval_start = commits.count()
        with ThreadPoolExecutor(max_workers=args.writer_clients + args.reader_clients) as executor:
            writers = [
                executor.submit(writer_worker, writer_id, args, partition, barrier, commits, errors)
                for writer_id, partition in enumerate(partitions)
            ]
            readers = [
                executor.submit(persistent_reader_worker, backend_id, args, lane, barrier, commits)
                for backend_id, lane in enumerate(query_lanes)
            ]
            for future in readers:
                backend = future.result()
                records.extend(backend.get("records", []))
                backend_lifecycles.append({key: value for key, value in backend.items() if key != "records"})
            reader_interval_end = commits.count()
            for writer in writers:
                writer.result()
        reader_interval = commits.evidence(reader_interval_start, reader_interval_end)
        all_commits = commits.evidence(0, commits.count())
        committed = all_commits["mutation_counts"]
        writer_errors: list[str] = []
        while not errors.empty():
            writer_errors.append(errors.get_nowait())
        validity = artifact_validity(
            records, backend_lifecycles, writer_errors, committed, args.writer_commits,
            reader_interval, args.reader_clients, args.queries, args.min_overlap_queries,
        )
        paired_summary = correctness_summary(records)
        query_diagnostics = manifest_query_diagnostics(records)
        protocol = formal_protocol_status(args)
        paper_eligible = paper_eligibility(
            validity["artifact_valid"], protocol, source_identity
        )
        output = {
            "artifact": "pgvector_update_correctness_stress",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "runner_version": RUNNER_VERSION,
            "correctness_contract": CORRECTNESS_CONTRACT,
            "created_at": utc_now(),
            "source_identity": source_identity,
            "runtime": runtime,
            "runtime_identity_binding": runtime_identity_binding,
            "scratch": {"name": table_name, "index": index_name, **setup},
            "args": {name: (str(value) if isinstance(value, Path) else value) for name, value in vars(args).items()},
            "mutation_mix": args.mutation_mix,
            "schedule_sha256": canonical_sha256([asdict(event) for event in schedule]),
            "committed_mutations": dict(committed), "writer_errors": writer_errors,
            "reader_interval_overlap": reader_interval,
            "backend_lifecycles": backend_lifecycles,
            "correctness_summary": paired_summary,
            "query_diagnostics": query_diagnostics,
            "records": sorted(records, key=lambda item: (int(item.get("backend_id", -1)), int(item.get("query_id", -1)))),
            "artifact_gates": validity,
            "diagnostic_valid": validity["artifact_valid"],
            "artifact_valid": validity["artifact_valid"],
            "paper_eligible": paper_eligible,
        }
        write_json(args.out, output)
        manifest = {
            "artifact": "pgvector_update_correctness_stress_manifest",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "runner_version": RUNNER_VERSION, "created_at": utc_now(),
            "source_identity": source_identity,
            "runtime": runtime,
            "runtime_identity_binding": runtime_identity_binding,
            "scratch": output["scratch"], "seeds": {"subset": args.seed, "schedule": args.seed + 17},
            "query_count": len(records), "requested_commits": args.writer_commits,
            "committed_mutations": dict(committed), "mutation_mix": args.mutation_mix,
            "reader_interval_overlap": reader_interval,
            "backend_lifecycles": backend_lifecycles,
            "correctness_summary": paired_summary,
            "query_diagnostics": query_diagnostics,
            "paired_query_contract": {
                "transaction": "stock_hnsw_then_safe_guided_hnsw_then_exact_seq_scan_in_one_repeatable_read_snapshot",
                "ann_equivalence": "ordered_ids_and_distances",
                "approximation_gate": "stock_and_guided_tie_aware_recall_equal; recall need not be 1.0",
                "filter_strategy": "safe_guided", "atoms": GUIDANCE_ATOMS, "kind": "exact",
            },
            "raw_output": {"path": str(args.out.resolve()), "sha256": sha256_file(args.out)},
            "output_sha256": sha256_file(args.out),
            "records_sha256": canonical_sha256(output["records"]),
            "diagnostic_valid": validity["artifact_valid"],
            "artifact_valid": validity["artifact_valid"],
            "paper_eligible": paper_eligible,
            "artifact_gates": validity,
            "formal_protocol": protocol,
        }
        manifest_errors = validate_manifest_payload(manifest)
        manifest["manifest_validation"] = {"passed": not manifest_errors, "errors": manifest_errors}
        if manifest_errors:
            manifest["artifact_valid"] = False
            manifest["paper_eligible"] = False
            manifest["artifact_gates"] = {
                "artifact_valid": False,
                "failed_gates": sorted(set([*validity["failed_gates"], "manifest_validation"])),
            }
        write_json(args.manifest, manifest)
        final_validity = bool(manifest["artifact_valid"])
        if not final_validity:
            failed = manifest["artifact_gates"]["failed_gates"]
            raise StressContractError("correctness stress failed closed: " + ", ".join(failed))
        return {"output": str(args.out), "manifest": str(args.manifest), **validity}
    finally:
        if args.execute and not args.keep_scratch:
            try:
                with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as admin:
                    drop_scratch(admin.cursor(), args)
            except Exception as exc:
                print(f"warning: could not drop scratch relation {table_name}: {exc}", file=sys.stderr)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Same-snapshot SQLens/pgvector concurrent-update correctness stress")
    parser.add_argument("--execute", action="store_true", help="perform opt-in database work")
    parser.add_argument("--dry-run", action="store_true", help="validate/print the protocol without connecting to PostgreSQL")
    parser.add_argument("--formal", action="store_true", help="enforce the substantial formal protocol gates")
    parser.add_argument("--source-table", default="public.amazon_grocery_reviews_10m_pgvector")
    parser.add_argument("--scratch-schema", default="sqlens_stress")
    parser.add_argument("--scratch-name", default="amazon_mvcc_stress")
    parser.add_argument("--scratch-index-name", default="amazon_mvcc_stress_hnsw_idx")
    parser.add_argument("--subset-rows", type=positive_int, default=250_000)
    parser.add_argument("--queries", type=positive_int, default=1_000)
    parser.add_argument("--reader-clients", type=positive_int, default=4)
    parser.add_argument("--writer-clients", type=positive_int, default=2)
    parser.add_argument("--writer-commits", type=positive_int, default=2_000)
    parser.add_argument("--writer-delay-ms", type=nonnegative_int, default=5)
    parser.add_argument("--min-overlap-queries", type=positive_int, default=20)
    parser.add_argument("--mutation-mix", type=parse_mutation_mix, default=parse_mutation_mix("predicate:4,vector:4,insert:1,delete:1"))
    parser.add_argument("--seed", type=nonnegative_int, default=20260722)
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--ef-search", type=positive_int, default=10_000)
    parser.add_argument("--max-scan-tuples", type=positive_int, default=500_000)
    parser.add_argument("--scan-mem-multiplier", type=positive_int, default=8)
    parser.add_argument("--statement-timeout-ms", type=positive_int, default=300_000)
    parser.add_argument("--preferred-index-guc", default="hnsw.preferred_index")
    parser.add_argument(
        "--expected-sqlens-build-id",
        default=os.environ.get("SQLENS_BUILD_ID", R36_BUILD_ID),
    )
    parser.add_argument(
        "--expected-vector-so-sha256",
        default=os.environ.get("SQLENS_VECTOR_SO_SHA256", R36_VECTOR_SO_SHA256),
    )
    parser.add_argument(
        "--expected-runner-sha256",
        default=os.environ.get("SQLENS_CORRECTNESS_RUNNER_SHA256", ""),
    )
    parser.add_argument(
        "--expected-git-revision",
        default=os.environ.get("SQLENS_GIT_REVISION", ""),
    )
    parser.add_argument(
        "--vector-so",
        type=Path,
        default=None,
        help="optional client-side mirror; identity authority is server PKGLIBDIR/vector.so",
    )
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULTS / "pgvector_update_correctness_stress.json")
    parser.add_argument("--manifest", type=Path, default=RESULTS / "pgvector_update_correctness_stress_manifest.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    try:
        result = run_experiment(args)
    except StressContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
