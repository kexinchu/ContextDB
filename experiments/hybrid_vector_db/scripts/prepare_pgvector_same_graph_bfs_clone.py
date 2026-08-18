"""Build and audit a same-graph, BFS-page-ordered SQLens HNSW clone.

The source and clone index the same heap.  SQLens loads the source graph and
writes the identical logical graph in BFS page order, so D2 changes physical
index layout without changing graph construction or tuple identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import psycopg
from psycopg import sql

try:
    from .common_pg import pg_config_from_env
    from .pgvector_design1_design2_design3_selectivity_benchmark import (
        require_d2_graph_proof,
    )
except ImportError:
    from common_pg import pg_config_from_env
    from pgvector_design1_design2_design3_selectivity_benchmark import (
        require_d2_graph_proof,
    )


SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PreparationError(RuntimeError):
    """The clone cannot be built under the requested formal contract."""


@dataclass(frozen=True)
class IndexState:
    name: str
    oid: int
    relfilenode: int
    heap_oid: int
    heap_name: str
    valid: bool
    ready: bool
    live: bool
    access_method: str
    column: str
    opclass: str
    predicate: str | None
    reloptions: tuple[str, ...]
    blocks: int
    bytes: int
    definition: str
    comment: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def measured_elapsed_ms(start_ns: int, finish_ns: int, phase: str) -> float:
    if start_ns < 0 or finish_ns < 0 or finish_ns <= start_ns:
        raise PreparationError(
            f"{phase} monotonic timing is invalid: start={start_ns}, finish={finish_ns}"
        )
    return (finish_ns - start_ns) / 1_000_000.0


def timing_measurements(
    *,
    stage: str,
    created: bool,
    total_start_ns: int,
    total_finish_ns: int,
    proof_start_ns: int,
    proof_finish_ns: int,
    creation_start_ns: int | None,
    creation_finish_ns: int | None,
) -> dict[str, Any]:
    if created:
        if creation_start_ns is None or creation_finish_ns is None:
            raise PreparationError(
                "created clone is missing its CREATE INDEX transaction timing"
            )
        creation = {
            "status": "measured",
            "elapsed_ms": measured_elapsed_ms(
                creation_start_ns,
                creation_finish_ns,
                "CREATE INDEX transaction",
            ),
            "reason": None,
        }
    else:
        if creation_start_ns is not None or creation_finish_ns is not None:
            raise PreparationError(
                "reused clone must not carry a CREATE INDEX transaction timing"
            )
        creation = {
            "status": "not_measured",
            "elapsed_ms": None,
            "reason": (
                "proof_only_existing_clone"
                if stage == "proof"
                else "existing_valid_clone_reused"
            ),
        }

    return {
        "schema": "sqlens-bfs-rewrite-overhead-v1",
        "clock": "time.monotonic_ns",
        "unit": "milliseconds",
        "clone_creation_transaction": {
            **creation,
            "semantics": (
                "Elapsed wall-clock time from immediately before entering the "
                "transaction through its successful commit. Includes transaction-local "
                "GUC setup, CREATE INDEX, COMMENT ON INDEX, and commit; excludes clone "
                "catalog re-read and graph proof."
            ),
        },
        "graph_proof": {
            "status": "measured",
            "elapsed_ms": measured_elapsed_ms(
                proof_start_ns, proof_finish_ns, "graph proof"
            ),
            "semantics": (
                "Elapsed wall-clock time spent in require_d2_graph_proof only."
            ),
        },
        "total_prepare": {
            "status": "measured",
            "elapsed_ms": measured_elapsed_ms(
                total_start_ns, total_finish_ns, "total prepare"
            ),
            "semantics": (
                "Elapsed wall-clock time from immediately before opening the database "
                "connection through binary/source validation, lock acquisition, optional "
                "invalid-clone replacement and clone creation, graph proof, and final "
                "catalog size capture. Excludes artifact serialization/write and CLI "
                "printing."
            ),
        },
    }


def storage_measurements(
    source: IndexState, clone: IndexState
) -> dict[str, Any]:
    if source.bytes <= 0 or source.blocks <= 0:
        raise PreparationError("cannot measure storage ratio for an empty source index")
    if clone.bytes <= 0 or clone.blocks <= 0:
        raise PreparationError("cannot measure storage ratio for an empty clone index")
    return {
        "schema": "sqlens-bfs-rewrite-storage-v1",
        "source": {"bytes": source.bytes, "blocks": source.blocks},
        "clone": {"bytes": clone.bytes, "blocks": clone.blocks},
        "clone_to_source_storage_ratio": clone.bytes / source.bytes,
        "clone_to_source_bytes_ratio": clone.bytes / source.bytes,
        "clone_to_source_blocks_ratio": clone.blocks / source.blocks,
        "semantics": {
            "bytes": (
                "Main-fork bytes from pg_relation_size(index_oid), captured after the "
                "graph proof."
            ),
            "blocks": (
                "Main-fork blocks computed as pg_relation_size(index_oid) divided by "
                "PostgreSQL block_size, captured after the graph proof."
            ),
            "clone_to_source_storage_ratio": "clone bytes divided by source bytes",
        },
    }


def split_name(value: str) -> tuple[str, str]:
    parts = value.split(".")
    if len(parts) == 1:
        return "public", parts[0]
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1]
    raise PreparationError(f"expected [schema.]relation, got {value!r}")


def qualified_name(value: str) -> str:
    schema, relation = split_name(value)
    return f"{schema}.{relation}"


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def exact_binary_identity(
    cur: psycopg.Cursor, expected_build_id: str, expected_sha256: str
) -> dict[str, Any]:
    cur.execute(
        "WITH lib AS (SELECT setting || '/vector.so' AS path "
        "FROM pg_config WHERE name='PKGLIBDIR') "
        "SELECT vector_sqlens_build_id(), path, "
        "encode(sha256(pg_read_binary_file(path)), 'hex') FROM lib"
    )
    row = cur.fetchone()
    if row is None:
        raise PreparationError("SQLens binary identity query returned no row")
    build_id, path, observed_sha = map(str, row)
    if build_id != expected_build_id or observed_sha != expected_sha256:
        raise PreparationError(
            "serving SQLens binary differs from the requested build: "
            f"build={build_id!r}, sha256={observed_sha}"
        )
    return {
        "build_id": build_id,
        "vector_so_path": path,
        "vector_so_sha256": observed_sha,
        "checked_at": utc_now(),
    }


def index_state(cur: psycopg.Cursor, name: str) -> IndexState | None:
    cur.execute(
        "SELECT index_class.oid::bigint, index_class.relfilenode::bigint, "
        "idx.indrelid::bigint, idx.indrelid::regclass::text, idx.indisvalid, "
        "idx.indisready, idx.indislive, am.amname, attribute.attname, "
        "opclass.opcname, pg_get_expr(idx.indpred, idx.indrelid), "
        "coalesce(index_class.reloptions, ARRAY[]::text[]), "
        "pg_relation_size(index_class.oid)::bigint / "
        "current_setting('block_size')::bigint, "
        "pg_relation_size(index_class.oid)::bigint, "
        "pg_get_indexdef(index_class.oid), "
        "obj_description(index_class.oid, 'pg_class') "
        "FROM pg_class index_class "
        "JOIN pg_index idx ON idx.indexrelid=index_class.oid "
        "JOIN pg_am am ON am.oid=index_class.relam "
        "JOIN pg_opclass opclass ON opclass.oid=idx.indclass[0] "
        "JOIN pg_attribute attribute ON attribute.attrelid=idx.indrelid "
        "AND attribute.attnum=idx.indkey[0] "
        "WHERE index_class.oid=to_regclass(%s)",
        (qualified_name(name),),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return IndexState(
        name=qualified_name(name),
        oid=int(row[0]),
        relfilenode=int(row[1]),
        heap_oid=int(row[2]),
        heap_name=qualified_name(str(row[3])),
        valid=bool(row[4]),
        ready=bool(row[5]),
        live=bool(row[6]),
        access_method=str(row[7]),
        column=str(row[8]),
        opclass=str(row[9]),
        predicate=str(row[10]) if row[10] is not None else None,
        reloptions=tuple(str(item) for item in row[11]),
        blocks=int(row[12]),
        bytes=int(row[13]),
        definition=str(row[14]),
        comment=str(row[15]) if row[15] is not None else None,
    )


def source_contract(state: IndexState, table: str) -> None:
    if state.heap_name != qualified_name(table):
        raise PreparationError(
            f"source index belongs to {state.heap_name}, not {qualified_name(table)}"
        )
    if not (state.valid and state.ready and state.live):
        raise PreparationError("source index is not valid, ready, and live")
    if state.access_method != "hnsw":
        raise PreparationError(f"source access method is {state.access_method!r}, not hnsw")
    if state.blocks <= 0 or state.bytes <= 0:
        raise PreparationError("source index is empty")


def semantic_contract(source: IndexState, clone: IndexState) -> None:
    fields = (
        "heap_oid",
        "heap_name",
        "access_method",
        "column",
        "opclass",
        "predicate",
        "reloptions",
    )
    mismatches = {
        field: {"source": getattr(source, field), "clone": getattr(clone, field)}
        for field in fields
        if getattr(source, field) != getattr(clone, field)
    }
    if mismatches:
        raise PreparationError(
            "clone index definition differs from source: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if not (clone.valid and clone.ready and clone.live):
        raise PreparationError("clone index is not valid, ready, and live")
    if clone.oid == source.oid or clone.relfilenode == source.relfilenode:
        raise PreparationError("source and clone do not have distinct physical storage")


def relation_comment(args: argparse.Namespace, source: IndexState) -> str:
    payload = {
        "artifact": "sqlens-same-graph-bfs-clone-v1",
        "source_index": source.name,
        "source_oid": source.oid,
        "source_relfilenode": source.relfilenode,
        "build_page_order": "bfs",
        "clone_source": source.name,
        "require_full_memory_build": True,
        "maintenance_work_mem": args.maintenance_work_mem,
        "sqlens_build_id": args.expected_sqlens_build_id,
        "vector_so_sha256": args.expected_vector_so_sha256,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def create_index_sql(source: IndexState, clone_name: str) -> sql.Composed:
    source_schema, _ = split_name(source.heap_name)
    clone_schema, clone_relation = split_name(clone_name)
    if clone_schema != source_schema:
        raise PreparationError("clone index must use the heap table's schema")
    statement = sql.SQL("CREATE INDEX {} ON {} USING hnsw ({} {})").format(
        sql.Identifier(clone_relation),
        sql.Identifier(*split_name(source.heap_name)),
        sql.Identifier(source.column),
        sql.Identifier(source.opclass),
    )
    options: dict[str, str] = {}
    for item in source.reloptions:
        key, separator, value = item.partition("=")
        if separator and key in {"m", "ef_construction"}:
            options[key] = value
    if set(options) != {"m", "ef_construction"}:
        raise PreparationError(
            f"source HNSW reloptions are incomplete: {source.reloptions!r}"
        )
    statement += sql.SQL(" WITH (m = {}, ef_construction = {})").format(
        sql.Literal(int(options["m"])), sql.Literal(int(options["ef_construction"]))
    )
    if source.predicate:
        statement += sql.SQL(" WHERE ") + sql.SQL(source.predicate)
    return statement


def acquire_lock(cur: psycopg.Cursor, table: str) -> None:
    cur.execute(
        "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
        (f"sqlens-same-graph-bfs-clone-v1:{qualified_name(table)}",),
    )
    if cur.fetchone()[0] is not True:
        raise PreparationError("another clone preparation owns the table lock")


def release_lock(cur: psycopg.Cursor, table: str) -> None:
    cur.execute(
        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
        (f"sqlens-same-graph-bfs-clone-v1:{qualified_name(table)}",),
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    total_start_ns = time.monotonic_ns()
    creation_start_ns: int | None = None
    creation_finish_ns: int | None = None
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        # Exact r43 identity is the serving contract. The shared
        # ensure_functions() prefix gate still stops at v16 and would reject
        # this binary before clone_source can run.
        binary = exact_binary_identity(
            cur, args.expected_sqlens_build_id, args.expected_vector_so_sha256
        )
        acquire_lock(cur, args.table)
        try:
            source = index_state(cur, args.source_index)
            if source is None:
                raise PreparationError(f"source index is missing: {args.source_index}")
            source_contract(source, args.table)
            clone = index_state(cur, args.clone_index)
            replaced_invalid = False
            if clone is not None and not (clone.valid and clone.ready and clone.live):
                if not args.replace_invalid_clone:
                    raise PreparationError(
                        "clone exists but is invalid; pass --replace-invalid-clone to replace it"
                    )
                with conn.transaction():
                    cur.execute("SET LOCAL statement_timeout=0")
                    cur.execute(
                        sql.SQL("DROP INDEX {}").format(
                            sql.Identifier(*split_name(args.clone_index))
                        )
                    )
                clone = None
                replaced_invalid = True

            created = False
            if clone is None:
                if args.stage == "proof":
                    raise PreparationError("proof-only stage requires an existing valid clone")
                creation_start_ns = time.monotonic_ns()
                with conn.transaction():
                    cur.execute("SET LOCAL statement_timeout=0")
                    # Clone pages are reconstructible from the source graph.  Compress
                    # their large WAL records and defer the final commit flush without
                    # weakening the durability of the source index or heap.
                    cur.execute("SET LOCAL wal_compression=on")
                    cur.execute("SET LOCAL synchronous_commit=off")
                    cur.execute(
                        "SELECT set_config('maintenance_work_mem', %s, true)",
                        (args.maintenance_work_mem,),
                    )
                    cur.execute("SET LOCAL max_parallel_maintenance_workers=0")
                    cur.execute(
                        "SELECT set_config('hnsw.require_full_memory_build', 'on', true)"
                    )
                    cur.execute("SELECT set_config('hnsw.build_page_order', 'bfs', true)")
                    cur.execute(
                        "SELECT set_config('hnsw.clone_source', %s, true)",
                        (source.name,),
                    )
                    cur.execute(create_index_sql(source, args.clone_index))
                    cur.execute(
                        sql.SQL("COMMENT ON INDEX {} IS {}").format(
                            sql.Identifier(*split_name(args.clone_index)),
                            sql.Literal(relation_comment(args, source)),
                        )
                    )
                creation_finish_ns = time.monotonic_ns()
                clone = index_state(cur, args.clone_index)
                created = True
            if clone is None:
                raise PreparationError("clone index was not created")
            semantic_contract(source, clone)

            cur.execute(
                "SELECT set_config('maintenance_work_mem', %s, false)",
                (args.maintenance_work_mem,),
            )
            proof_start_ns = time.monotonic_ns()
            proof = require_d2_graph_proof(cur, source.name, clone.name)
            proof_finish_ns = time.monotonic_ns()
            proof["artifact_valid"] = True

            final_source = index_state(cur, args.source_index)
            final_clone = index_state(cur, args.clone_index)
            if final_source is None or final_clone is None:
                raise PreparationError(
                    "source or clone disappeared before final storage measurement"
                )
            source_contract(final_source, args.table)
            semantic_contract(final_source, final_clone)
            total_finish_ns = time.monotonic_ns()

            payload = {
                **proof,
                "preparation": {
                    "artifact": "sqlens-same-graph-bfs-clone-v1",
                    "completed_at": utc_now(),
                    "binary": binary,
                    "source": asdict(final_source),
                    "clone": asdict(final_clone),
                    "created": created,
                    "replaced_invalid_clone": replaced_invalid,
                    "maintenance_work_mem": args.maintenance_work_mem,
                    "max_parallel_maintenance_workers": 0,
                    "wal_compression": "on",
                    "synchronous_commit": "off",
                    "timing": timing_measurements(
                        stage=args.stage,
                        created=created,
                        total_start_ns=total_start_ns,
                        total_finish_ns=total_finish_ns,
                        proof_start_ns=proof_start_ns,
                        proof_finish_ns=proof_finish_ns,
                        creation_start_ns=creation_start_ns,
                        creation_finish_ns=creation_finish_ns,
                    ),
                    "storage": storage_measurements(final_source, final_clone),
                },
            }
            atomic_write_json(args.out, payload)
            return payload
        finally:
            release_lock(cur, args.table)


def sha256_arg(value: str) -> str:
    normalized = value.lower()
    if not SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256 digest")
    return normalized


def memory_arg(value: str) -> str:
    if not re.fullmatch(r"[1-9][0-9]*(?:kB|MB|GB|TB)", value):
        raise argparse.ArgumentTypeError("use a PostgreSQL memory value such as 256GB")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and prove a same-graph BFS-page-ordered SQLens HNSW clone."
    )
    parser.add_argument("--table", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--clone-index", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maintenance-work-mem", type=memory_arg, default="256GB")
    parser.add_argument("--stage", choices=("all", "clone", "proof"), default="all")
    parser.add_argument(
        "--replace-invalid-clone",
        action="store_true",
        help="Drop and replace only an existing clone whose catalog state is invalid.",
    )
    parser.add_argument("--expected-sqlens-build-id", required=True)
    parser.add_argument("--expected-vector-so-sha256", type=sha256_arg, required=True)
    args = parser.parse_args(argv)
    if qualified_name(args.source_index) == qualified_name(args.clone_index):
        parser.error("source and clone indexes must be distinct")
    if args.stage == "clone":
        parser.error("--stage=clone is unsupported: a formal clone must include its proof")
    if args.out.exists():
        parser.error(f"refusing to overwrite proof artifact: {args.out}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = prepare(args)
    print(
        json.dumps(
            {
                "proof": str(args.out),
                "stable_fingerprint_sha256": payload["stable_fingerprint_sha256"],
                "source_index": payload["source_index"],
                "clone_index": payload["clone_index"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
