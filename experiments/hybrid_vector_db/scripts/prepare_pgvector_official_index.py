"""Build and attest the dedicated upstream-pgvector Amazon-10M HNSW index.

This tool intentionally does not install or switch ``vector.so``.  A controller
must first load the pinned official binary, then invoke this script.  The script
fails closed unless the live server binary, extension version, source checkout,
frozen table, workload inputs, and resulting index all match the requested
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import psycopg
from psycopg import sql

try:
    from .common_pg import pg_config_from_env
except ImportError:  # Direct execution places this directory on sys.path.
    from common_pg import pg_config_from_env


ARTIFACT_CONTRACT = "sqlens_pgvector_official_source_index_v1"
COMMENT_PREFIX = "sqlens-official-index-v1:"
DEFAULT_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
DEFAULT_INDEX = "public.amazon10m_official_hnsw_m32ef200_source_idx"
DEFAULT_MANIFEST = Path(
    "results/hybrid_vector_db/amazon10m_pgvector_official_hnsw_m32ef200_index.json"
)
DEFAULT_VECTOR_SO_SHA256 = (
    "a97f730478cd3628820fb072273f8185c94ffa76a9ee802a006db7028e7b8d87"
)
DEFAULT_SOURCE_TAG = "v0.8.2"
DEFAULT_EXTENSION_VERSION = "0.8.2"
DEFAULT_TABLE_ROWS = 10_000_000
DEFAULT_PREDICATE_ROWS = 9_979_556
DEFAULT_MAINTENANCE_WORK_MEM = "128GB"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
PREDICATE = "embedding_valid"
MAX_PARALLEL_MAINTENANCE_WORKERS = 0

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
MEMORY_RE = re.compile(r"^[1-9][0-9]*(?:kB|MB|GB|TB)$")


class PreparationError(RuntimeError):
    """An official-index provenance or preparation invariant failed."""


@dataclass(frozen=True)
class TableState:
    name: str
    oid: int
    relfilenode: int
    physical_relfilenode: int
    relation_filepath: str
    relation_size_bytes: int
    total_rows: int
    predicate_rows: int


@dataclass(frozen=True)
class IndexState:
    name: str
    oid: int
    relfilenode: int
    physical_relfilenode: int
    relation_filepath: str
    relation_size_bytes: int
    heap_oid: int
    heap_relfilenode: int
    valid: bool
    ready: bool
    live: bool
    access_method: str
    unique: bool
    primary: bool
    key_attributes: int
    total_attributes: int
    indexed_column: str | None
    opclass: str | None
    predicate: str | None
    reloptions: tuple[str, ...]
    comment: str | None
    definition: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def sha256_value(value: str) -> str:
    normalized = value.strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA256")
    return normalized


def git_commit(value: str) -> str:
    normalized = value.strip().lower()
    if not GIT_COMMIT_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a full 40-64 character Git commit")
    return normalized


def memory_setting(value: str) -> str:
    if not MEMORY_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "memory setting must use a positive PostgreSQL unit such as 128GB"
        )
    return value


def parse_qualified_name(value: str) -> tuple[str, str]:
    parts = value.split(".")
    if len(parts) != 2 or any(not IDENTIFIER_RE.fullmatch(part) for part in parts):
        raise argparse.ArgumentTypeError(
            "relation names must use unquoted schema.relation identifiers"
        )
    return parts[0].lower(), parts[1].lower()


def qualified_name(value: str) -> str:
    schema, relation = parse_qualified_name(value)
    return f"{schema}.{relation}"


def quote_identifier(value: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"invalid PostgreSQL identifier: {value!r}")
    return f'"{value}"'


def quote_qualified_name(value: str) -> str:
    schema, relation = parse_qualified_name(value)
    return f"{quote_identifier(schema)}.{quote_identifier(relation)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def create_index_sql(table: str = DEFAULT_TABLE, index: str = DEFAULT_INDEX) -> str:
    table_schema, _ = parse_qualified_name(table)
    index_schema, index_relation = parse_qualified_name(index)
    if index_schema != table_schema:
        raise PreparationError("official index must be in the table's schema")
    return (
        f"CREATE INDEX {quote_identifier(index_relation)} "
        f"ON {quote_qualified_name(table)} "
        f"USING hnsw ({quote_identifier('embedding')} vector_l2_ops) "
        f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION}) "
        f"WHERE {PREDICATE}"
    )


def comment_sql(index: str, comment: str) -> sql.Composed:
    schema, relation = parse_qualified_name(index)
    return sql.SQL("COMMENT ON INDEX {} IS {}").format(
        sql.Identifier(schema, relation), sql.Literal(comment)
    )


def normalize_predicate(value: str | None) -> str:
    text = "" if value is None else value.lower()
    return re.sub(r"[\s()]", "", text)


def parse_reloptions(values: Sequence[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values or ():
        key, separator, value = str(item).partition("=")
        if not separator or not key or key in result:
            raise PreparationError(f"invalid or duplicate index reloption: {item!r}")
        result[key] = value
    return result


def run_git(
    argv: Sequence[str], *, cwd: Path, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def source_checkout_identity(
    repo: Path,
    tag: str,
    commit: str,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_git,
) -> dict[str, Any]:
    resolved = repo.expanduser().resolve()
    if not resolved.is_dir():
        raise PreparationError(f"official source repository does not exist: {resolved}")
    try:
        head = command_runner(
            ("git", "rev-parse", "HEAD^{commit}"), cwd=resolved
        ).stdout.strip().lower()
        tag_commit = command_runner(
            ("git", "rev-parse", f"refs/tags/{tag}^{{commit}}"), cwd=resolved
        ).stdout.strip().lower()
        dirty = command_runner(
            ("git", "status", "--porcelain", "--untracked-files=no"), cwd=resolved
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreparationError(
            f"could not verify official source checkout {resolved}"
        ) from exc
    if head != commit:
        raise PreparationError(
            f"official source HEAD mismatch: expected={commit}, observed={head}"
        )
    if tag_commit != commit:
        raise PreparationError(
            f"official source tag mismatch: tag={tag!r}, expected={commit}, "
            f"observed={tag_commit}"
        )
    if dirty:
        raise PreparationError("official source checkout has tracked modifications")
    return {
        "repo": str(resolved),
        "source_tag": tag,
        "source_commit": commit,
        "head_commit": head,
        "tag_commit": tag_commit,
        "tracked_tree_clean": True,
        "verified": True,
    }


def input_artifact_identity(
    paths: Mapping[str, Path],
    *,
    file_hasher: Callable[[Path], str] = sha256_file,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise PreparationError(f"required input {name} does not exist: {resolved}")
        before = resolved.stat()
        digest = file_hasher(resolved)
        after = resolved.stat()
        if (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise PreparationError(f"required input {name} changed while hashing")
        if not SHA256_RE.fullmatch(digest):
            raise PreparationError(f"invalid SHA256 produced for input {name}")
        result[name] = {
            "path": str(resolved),
            "sha256": digest,
            "size_bytes": after.st_size,
        }
    return result


def server_identity(
    cur: psycopg.Cursor,
    *,
    expected_extension_version: str,
    expected_vector_so_sha256: str,
) -> dict[str, Any]:
    try:
        cur.execute(
            "WITH lib AS ("
            "SELECT setting || '/vector.so' AS path "
            "FROM pg_config WHERE name = 'PKGLIBDIR'"
            ") "
            "SELECT current_setting('server_version'), "
            "current_setting('server_version_num')::integer, "
            "COALESCE((SELECT extversion FROM pg_extension "
            "WHERE extname = 'vector'), ''), "
            "lib.path, encode(sha256(pg_read_binary_file(lib.path)), 'hex'), "
            "to_regprocedure('vector_sqlens_build_id()')::text "
            "FROM lib"
        )
        row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 - provenance must fail closed.
        raise PreparationError(
            "server vector.so SHA/version provenance query is unavailable"
        ) from exc
    if row is None:
        raise PreparationError("server provenance query returned no row")
    (
        postgres_version,
        postgres_version_num,
        extension_version,
        vector_so_path,
        vector_so_sha256,
        build_id_function,
    ) = row
    observed_sha = str(vector_so_sha256).lower()
    if str(extension_version) != expected_extension_version:
        raise PreparationError(
            "vector extension version mismatch: "
            f"expected={expected_extension_version!r}, "
            f"observed={str(extension_version)!r}"
        )
    if observed_sha != expected_vector_so_sha256:
        raise PreparationError(
            "server vector.so SHA256 mismatch: "
            f"expected={expected_vector_so_sha256}, observed={observed_sha}"
        )
    if not str(vector_so_path).endswith("/vector.so"):
        raise PreparationError(f"unexpected server vector.so path: {vector_so_path!r}")
    return {
        "postgresql_version": str(postgres_version),
        "postgresql_version_num": int(postgres_version_num),
        "vector_extension_version": str(extension_version),
        "vector_so_path": str(vector_so_path),
        "vector_so_sha256": observed_sha,
        "vector_build_id_function": (
            None if build_id_function is None else str(build_id_function)
        ),
        "binary_identity_method": "server_file_sha256",
        "controller_evidence_required": build_id_function is None,
        "exact_match": True,
        "checked_at": utc_now(),
    }


def table_state(cur: psycopg.Cursor, table: str) -> TableState:
    cur.execute(
        "SELECT c.oid::bigint, c.relfilenode::bigint, "
        "pg_relation_filenode(c.oid)::bigint, pg_relation_filepath(c.oid), "
        "pg_relation_size(c.oid)::bigint "
        "FROM pg_class c WHERE c.oid = to_regclass(%s)",
        (table,),
    )
    relation = cur.fetchone()
    if relation is None:
        raise PreparationError(f"frozen Amazon table does not exist: {table}")
    cur.execute(
        f"SELECT count(*)::bigint, "
        f"count(*) FILTER (WHERE {PREDICATE})::bigint "
        f"FROM {quote_qualified_name(table)}"
    )
    counts = cur.fetchone()
    if counts is None:
        raise PreparationError("could not count the frozen Amazon table")
    state = TableState(
        name=table,
        oid=int(relation[0]),
        relfilenode=int(relation[1]),
        physical_relfilenode=int(relation[2]),
        relation_filepath=str(relation[3]),
        relation_size_bytes=int(relation[4]),
        total_rows=int(counts[0]),
        predicate_rows=int(counts[1]),
    )
    if min(
        state.oid,
        state.relfilenode,
        state.physical_relfilenode,
        state.relation_size_bytes,
    ) <= 0:
        raise PreparationError(f"invalid frozen table physical identity: {state}")
    return state


def validate_frozen_table(
    state: TableState, *, expected_rows: int, expected_predicate_rows: int
) -> None:
    if state.total_rows != expected_rows:
        raise PreparationError(
            f"frozen table row count mismatch: expected={expected_rows}, "
            f"observed={state.total_rows}"
        )
    if state.predicate_rows != expected_predicate_rows:
        raise PreparationError(
            f"embedding_valid row count mismatch: expected={expected_predicate_rows}, "
            f"observed={state.predicate_rows}"
        )


def index_state(cur: psycopg.Cursor, index: str) -> IndexState | None:
    cur.execute(
        "SELECT idx.oid::bigint, idx.relfilenode::bigint, "
        "pg_relation_filenode(idx.oid)::bigint, pg_relation_filepath(idx.oid), "
        "pg_relation_size(idx.oid)::bigint, ix.indrelid::bigint, "
        "heap.relfilenode::bigint, ix.indisvalid, ix.indisready, ix.indislive, "
        "am.amname, ix.indisunique, ix.indisprimary, ix.indnkeyatts, ix.indnatts, "
        "att.attname, opc.opcname, pg_get_expr(ix.indpred, ix.indrelid, true), "
        "idx.reloptions, obj_description(idx.oid, 'pg_class'), "
        "pg_get_indexdef(idx.oid) "
        "FROM pg_class idx "
        "JOIN pg_index ix ON ix.indexrelid = idx.oid "
        "JOIN pg_class heap ON heap.oid = ix.indrelid "
        "JOIN pg_am am ON am.oid = idx.relam "
        "LEFT JOIN pg_attribute att ON att.attrelid = ix.indrelid "
        "AND att.attnum = ix.indkey[0] "
        "LEFT JOIN pg_opclass opc ON opc.oid = ix.indclass[0] "
        "WHERE idx.oid = to_regclass(%s)",
        (index,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return IndexState(
        name=index,
        oid=int(row[0]),
        relfilenode=int(row[1]),
        physical_relfilenode=int(row[2]),
        relation_filepath=str(row[3]),
        relation_size_bytes=int(row[4]),
        heap_oid=int(row[5]),
        heap_relfilenode=int(row[6]),
        valid=bool(row[7]),
        ready=bool(row[8]),
        live=bool(row[9]),
        access_method=str(row[10]),
        unique=bool(row[11]),
        primary=bool(row[12]),
        key_attributes=int(row[13]),
        total_attributes=int(row[14]),
        indexed_column=None if row[15] is None else str(row[15]),
        opclass=None if row[16] is None else str(row[16]),
        predicate=None if row[17] is None else str(row[17]),
        reloptions=tuple(row[18] or ()),
        comment=None if row[19] is None else str(row[19]),
        definition=str(row[20]),
    )


def structural_index_diff(
    state: IndexState, table: TableState
) -> dict[str, dict[str, Any]]:
    expected: dict[str, Any] = {
        "heap_oid": table.oid,
        "heap_relfilenode": table.relfilenode,
        "valid": True,
        "ready": True,
        "live": True,
        "access_method": "hnsw",
        "unique": False,
        "primary": False,
        "key_attributes": 1,
        "total_attributes": 1,
        "indexed_column": "embedding",
        "opclass": "vector_l2_ops",
        "predicate": PREDICATE,
        "reloptions": {
            "m": str(HNSW_M),
            "ef_construction": str(HNSW_EF_CONSTRUCTION),
        },
    }
    observed: dict[str, Any] = {
        "heap_oid": state.heap_oid,
        "heap_relfilenode": state.heap_relfilenode,
        "valid": state.valid,
        "ready": state.ready,
        "live": state.live,
        "access_method": state.access_method,
        "unique": state.unique,
        "primary": state.primary,
        "key_attributes": state.key_attributes,
        "total_attributes": state.total_attributes,
        "indexed_column": state.indexed_column,
        "opclass": state.opclass,
        "predicate": normalize_predicate(state.predicate),
        "reloptions": parse_reloptions(state.reloptions),
    }
    return {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }


def provenance_comment(
    request_contract: Mapping[str, Any],
    *,
    build_started_at: str,
    build_completed_at: str,
    build_wall_seconds: float,
) -> str:
    payload = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "request_contract": dict(request_contract),
        "request_contract_sha256": sha256_json(request_contract),
        "build": {
            "started_at": build_started_at,
            "completed_at": build_completed_at,
            "wall_seconds": build_wall_seconds,
        },
    }
    return COMMENT_PREFIX + json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def parse_provenance_comment(value: str | None) -> dict[str, Any]:
    if value is None or not value.startswith(COMMENT_PREFIX):
        raise PreparationError("existing official index lacks its provenance comment")
    try:
        payload = json.loads(value[len(COMMENT_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise PreparationError("existing official index provenance comment is invalid") from exc
    if not isinstance(payload, dict):
        raise PreparationError("existing official index provenance is not an object")
    contract = payload.get("request_contract")
    if (
        payload.get("artifact_contract") != ARTIFACT_CONTRACT
        or not isinstance(contract, dict)
        or payload.get("request_contract_sha256") != sha256_json(contract)
    ):
        raise PreparationError("existing official index provenance is incomplete")
    build = payload.get("build")
    if (
        not isinstance(build, dict)
        or not isinstance(build.get("wall_seconds"), (int, float))
        or float(build["wall_seconds"]) < 0
    ):
        raise PreparationError("existing official index build timing is invalid")
    return payload


def validate_existing_index(
    state: IndexState, table: TableState, request_contract: Mapping[str, Any]
) -> dict[str, Any]:
    differences = structural_index_diff(state, table)
    if differences:
        raise PreparationError(
            "existing official index definition mismatch; refusing to modify or drop it: "
            + json.dumps(differences, sort_keys=True)
        )
    provenance = parse_provenance_comment(state.comment)
    if provenance["request_contract"] != dict(request_contract):
        raise PreparationError(
            "existing official index provenance does not match this request; "
            "refusing to modify or drop it"
        )
    return provenance


def request_contract(
    args: argparse.Namespace,
    *,
    source: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
    server: Mapping[str, Any],
    table: TableState,
) -> dict[str, Any]:
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "data_epoch": args.data_epoch,
        "table": qualified_name(args.table),
        "table_oid": table.oid,
        "table_relfilenode": table.relfilenode,
        "table_physical_relfilenode": table.physical_relfilenode,
        "expected_table_rows": args.expected_table_rows,
        "expected_predicate_rows": args.expected_predicate_rows,
        "index": qualified_name(args.index),
        "create_sql": create_index_sql(args.table, args.index),
        "index_layout": "upstream_source_insertion_order",
        "column": "embedding",
        "opclass": "vector_l2_ops",
        "predicate": PREDICATE,
        "m": HNSW_M,
        "ef_construction": HNSW_EF_CONSTRUCTION,
        "maintenance_work_mem": args.maintenance_work_mem,
        "max_parallel_maintenance_workers": MAX_PARALLEL_MAINTENANCE_WORKERS,
        "builder": {
            "implementation": "pgvector_official",
            "vector_extension_version": server["vector_extension_version"],
            "vector_so_sha256": server["vector_so_sha256"],
            "source_repo": source["repo"],
            "source_tag": source["source_tag"],
            "source_commit": source["source_commit"],
        },
        "input_sha256": {
            name: identity["sha256"] for name, identity in sorted(inputs.items())
        },
    }


def acquire_lock(cur: psycopg.Cursor, table: str, index: str) -> None:
    key = f"{ARTIFACT_CONTRACT}:{table}:{index}"
    cur.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (key,))
    row = cur.fetchone()
    if row is None or row[0] is not True:
        raise PreparationError("another official-index preparation owns the DB lock")


def release_lock(cur: psycopg.Cursor, table: str, index: str) -> None:
    key = f"{ARTIFACT_CONTRACT}:{table}:{index}"
    cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,))


def prepare_index(
    conn: psycopg.Connection,
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[TableState, IndexState, dict[str, Any], bool]:
    with conn.transaction():
        cur.execute(f"LOCK TABLE {quote_qualified_name(args.table)} IN SHARE MODE")
        before = table_state(cur, args.table)
        validate_frozen_table(
            before,
            expected_rows=args.expected_table_rows,
            expected_predicate_rows=args.expected_predicate_rows,
        )
        existing = index_state(cur, args.index)
        if existing is not None:
            provenance = validate_existing_index(existing, before, contract)
            after = table_state(cur, args.table)
            if after != before:
                raise PreparationError(
                    "frozen Amazon table changed while validating the official index"
                )
            return after, existing, provenance, False

        cur.execute("SET LOCAL statement_timeout = 0")
        cur.execute(
            "SELECT set_config('maintenance_work_mem', %s, true)",
            (args.maintenance_work_mem,),
        )
        cur.execute(
            f"SET LOCAL max_parallel_maintenance_workers = "
            f"{MAX_PARALLEL_MAINTENANCE_WORKERS}"
        )
        started_at = utc_now()
        started = monotonic()
        cur.execute(create_index_sql(args.table, args.index))
        wall_seconds = max(0.0, monotonic() - started)
        completed_at = utc_now()
        created = index_state(cur, args.index)
        if created is None:
            raise PreparationError("CREATE INDEX completed but the index is absent")
        differences = structural_index_diff(created, before)
        if differences:
            raise PreparationError(
                "new official index failed structural validation: "
                + json.dumps(differences, sort_keys=True)
            )
        comment = provenance_comment(
            contract,
            build_started_at=started_at,
            build_completed_at=completed_at,
            build_wall_seconds=wall_seconds,
        )
        cur.execute(comment_sql(args.index, comment))
        final = index_state(cur, args.index)
        if final is None:
            raise PreparationError("official index disappeared after COMMENT")
        provenance = validate_existing_index(final, before, contract)
        after = table_state(cur, args.table)
        if after != before:
            raise PreparationError(
                "frozen Amazon table changed during the official HNSW build"
            )
        return after, final, provenance, True


def manifest_payload(
    args: argparse.Namespace,
    *,
    source: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
    server: Mapping[str, Any],
    table: TableState,
    index: IndexState,
    provenance: Mapping[str, Any],
    created: bool,
) -> dict[str, Any]:
    contract = provenance["request_contract"]
    build = provenance["build"]
    indexdef_sha256 = hashlib.sha256(index.definition.encode("utf-8")).hexdigest()
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "artifact_valid": True,
        "status": "complete",
        "completed_at": utc_now(),
        "table": qualified_name(args.table),
        "index": qualified_name(args.index),
        "data_epoch": args.data_epoch,
        "create_sql": contract["create_sql"],
        "created_in_this_run": created,
        "resumed_existing_index": not created,
        "drop_policy": "never_drop_fail_closed_on_any_mismatch",
        "request_contract": contract,
        "request_contract_sha256": provenance["request_contract_sha256"],
        "builder": {
            "implementation": "pgvector_official",
            "vector_extension_version": server["vector_extension_version"],
            "vector_so_sha256": server["vector_so_sha256"],
            "vector_so_path": server["vector_so_path"],
            "binary_identity_method": server["binary_identity_method"],
            "vector_build_id_function": server["vector_build_id_function"],
            "controller_evidence_required": server["controller_evidence_required"],
            "source_repo": source["repo"],
            "source_tag": source["source_tag"],
            "source_commit": source["source_commit"],
            "source_tag_commit": source["tag_commit"],
            "source_tree_clean": source["tracked_tree_clean"],
            "postgresql_version": server["postgresql_version"],
            "postgresql_version_num": server["postgresql_version_num"],
            "m": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
            "layout": "upstream_source_insertion_order",
            "maintenance_work_mem": args.maintenance_work_mem,
            "max_parallel_maintenance_workers": (
                MAX_PARALLEL_MAINTENANCE_WORKERS
            ),
            "build_started_at": build["started_at"],
            "build_completed_at": build["completed_at"],
            "build_wall_seconds": float(build["wall_seconds"]),
            "binary_source_binding": (
                "server_file_sha256_plus_verified_source_checkout"
            ),
        },
        "input_artifacts": dict(inputs),
        "table_fingerprint": asdict(table),
        "index_fingerprint": {
            "index": index.name,
            "index_oid": index.oid,
            "index_relfilenode": index.relfilenode,
            "index_physical_relfilenode": index.physical_relfilenode,
            "relation_filepath": index.relation_filepath,
            "relation_size_bytes": index.relation_size_bytes,
            "heap_oid": index.heap_oid,
            "heap_relfilenode": index.heap_relfilenode,
            "indexdef": index.definition,
            "indexdef_sha256": indexdef_sha256,
            "predicate": index.predicate,
            "predicate_rows": table.predicate_rows,
            "indisvalid": index.valid,
            "indisready": index.ready,
            "indislive": index.live,
            "access_method": index.access_method,
            "indexed_column": index.indexed_column,
            "opclass": index.opclass,
            "reloptions": list(index.reloptions),
        },
    }


def validate_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    table: TableState,
    index: IndexState,
) -> None:
    if (
        manifest.get("artifact_contract") != ARTIFACT_CONTRACT
        or manifest.get("artifact_valid") is not True
        or manifest.get("status") != "complete"
    ):
        raise PreparationError("resume manifest is not a valid completed artifact")
    if manifest.get("request_contract") != dict(request):
        raise PreparationError("resume manifest belongs to a different request")
    if manifest.get("request_contract_sha256") != sha256_json(request):
        raise PreparationError("resume manifest request hash is stale")
    table_manifest = manifest.get("table_fingerprint")
    if table_manifest != asdict(table):
        raise PreparationError("live frozen table differs from the resume manifest")
    fingerprint = manifest.get("index_fingerprint")
    if not isinstance(fingerprint, dict):
        raise PreparationError("resume manifest index fingerprint is missing")
    live = {
        "index_oid": index.oid,
        "index_relfilenode": index.relfilenode,
        "index_physical_relfilenode": index.physical_relfilenode,
        "relation_filepath": index.relation_filepath,
        "relation_size_bytes": index.relation_size_bytes,
        "indexdef_sha256": hashlib.sha256(
            index.definition.encode("utf-8")
        ).hexdigest(),
    }
    for key, value in live.items():
        if fingerprint.get(key) != value:
            raise PreparationError(
                f"live official index {key} differs from the resume manifest"
            )


def dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "dry_run": True,
        "database_connected": False,
        "filesystem_inputs_read": False,
        "binary_switched": False,
        "table": qualified_name(args.table),
        "index": qualified_name(args.index),
        "create_sql": create_index_sql(args.table, args.index),
        "predicate": PREDICATE,
        "expected_table_rows": args.expected_table_rows,
        "expected_predicate_rows": args.expected_predicate_rows,
        "builder": {
            "vector_extension_version": args.expected_vector_extension_version,
            "vector_so_sha256": args.expected_vector_so_sha256,
            "source_repo": str(args.source_repo),
            "source_tag": args.source_tag,
            "source_commit": args.source_commit,
            "m": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
            "layout": "upstream_source_insertion_order",
            "max_parallel_maintenance_workers": (
                MAX_PARALLEL_MAINTENANCE_WORKERS
            ),
        },
        "inputs": {
            "filters_csv": str(args.filters_csv),
            "calibration_workload_csv": str(args.calibration_workload_csv),
            "measurement_workload_csv": str(args.measurement_workload_csv),
            "truth_csv": str(args.truth_csv),
        },
        "manifest": str(args.manifest),
        "resume": args.resume,
        "drop_policy": "never_drop_fail_closed_on_any_mismatch",
    }


def run(
    args: argparse.Namespace,
    *,
    connect: Callable[..., psycopg.Connection] = psycopg.connect,
    source_probe: Callable[..., dict[str, Any]] = source_checkout_identity,
    file_hasher: Callable[[Path], str] = sha256_file,
) -> dict[str, Any]:
    validate_args(args)
    if args.dry_run:
        return dry_run_plan(args)
    if args.manifest.exists() and not args.resume:
        raise PreparationError(
            f"manifest already exists; pass --resume to validate it: {args.manifest}"
        )

    source = source_probe(args.source_repo, args.source_tag, args.source_commit)
    inputs = input_artifact_identity(
        {
            "filters_csv": args.filters_csv,
            "calibration_workload_csv": args.calibration_workload_csv,
            "measurement_workload_csv": args.measurement_workload_csv,
            "truth_csv": args.truth_csv,
        },
        file_hasher=file_hasher,
    )
    try:
        existing_manifest = (
            json.loads(args.manifest.read_text(encoding="utf-8"))
            if args.manifest.exists()
            else None
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"could not load resume manifest: {args.manifest}") from exc
    conninfo = args.dsn or pg_config_from_env().conninfo
    with connect(conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        server = server_identity(
            cur,
            expected_extension_version=args.expected_vector_extension_version,
            expected_vector_so_sha256=args.expected_vector_so_sha256,
        )
        acquire_lock(cur, args.table, args.index)
        try:
            initial_table = table_state(cur, args.table)
            validate_frozen_table(
                initial_table,
                expected_rows=args.expected_table_rows,
                expected_predicate_rows=args.expected_predicate_rows,
            )
            contract = request_contract(
                args,
                source=source,
                inputs=inputs,
                server=server,
                table=initial_table,
            )
            table, index, provenance, created = prepare_index(
                conn, cur, args, contract
            )
            source_after = source_probe(
                args.source_repo, args.source_tag, args.source_commit
            )
            if source_after != source:
                raise PreparationError(
                    "official source checkout changed during index preparation"
                )
            inputs_after = input_artifact_identity(
                {
                    "filters_csv": args.filters_csv,
                    "calibration_workload_csv": args.calibration_workload_csv,
                    "measurement_workload_csv": args.measurement_workload_csv,
                    "truth_csv": args.truth_csv,
                },
                file_hasher=file_hasher,
            )
            if inputs_after != inputs:
                raise PreparationError(
                    "a workload input changed during index preparation"
                )
            server_after = server_identity(
                cur,
                expected_extension_version=args.expected_vector_extension_version,
                expected_vector_so_sha256=args.expected_vector_so_sha256,
            )
            stable_server_fields = (
                "postgresql_version",
                "postgresql_version_num",
                "vector_extension_version",
                "vector_so_path",
                "vector_so_sha256",
                "vector_build_id_function",
                "binary_identity_method",
                "controller_evidence_required",
            )
            if any(server_after[field] != server[field] for field in stable_server_fields):
                raise PreparationError(
                    "server PostgreSQL/vector identity changed during index preparation"
                )
            final_table = table_state(cur, args.table)
            if final_table != table:
                raise PreparationError(
                    "frozen Amazon table changed before manifest publication"
                )
            if existing_manifest is not None:
                validate_resume_manifest(
                    existing_manifest,
                    request=contract,
                    table=final_table,
                    index=index,
                )
                return dict(existing_manifest)
            if args.manifest.exists():
                raise PreparationError(
                    "manifest appeared during preparation; refusing to overwrite it"
                )
            payload = manifest_payload(
                args,
                source=source,
                inputs=inputs,
                server=server,
                table=final_table,
                index=index,
                provenance=provenance,
                created=created,
            )
            atomic_write_json(args.manifest, payload)
            return payload
        finally:
            release_lock(cur, args.table, args.index)
            cur.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and attest a dedicated official-pgvector Amazon-10M source-order "
            "HNSW index; this tool never switches vector.so"
        )
    )
    parser.add_argument("--dsn", default="")
    parser.add_argument("--table", type=qualified_name, default=DEFAULT_TABLE)
    parser.add_argument("--index", type=qualified_name, default=DEFAULT_INDEX)
    parser.add_argument("--data-epoch", required=True)
    parser.add_argument(
        "--expected-vector-so-sha256",
        type=sha256_value,
        default=DEFAULT_VECTOR_SO_SHA256,
    )
    parser.add_argument(
        "--expected-vector-extension-version",
        default=DEFAULT_EXTENSION_VERSION,
    )
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--source-tag", default=DEFAULT_SOURCE_TAG)
    parser.add_argument("--source-commit", type=git_commit, required=True)
    parser.add_argument("--filters-csv", type=Path, required=True)
    parser.add_argument("--calibration-workload-csv", type=Path, required=True)
    parser.add_argument("--measurement-workload-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument(
        "--expected-table-rows", type=positive_int, default=DEFAULT_TABLE_ROWS
    )
    parser.add_argument(
        "--expected-predicate-rows",
        type=positive_int,
        default=DEFAULT_PREDICATE_ROWS,
    )
    parser.add_argument(
        "--maintenance-work-mem",
        type=memory_setting,
        default=DEFAULT_MAINTENANCE_WORK_MEM,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    table_schema, _ = parse_qualified_name(args.table)
    index_schema, index_relation = parse_qualified_name(args.index)
    if table_schema != index_schema:
        raise PreparationError("official index must be in the table's schema")
    if len(index_relation.encode("utf-8")) > 63:
        raise PreparationError("official index name exceeds PostgreSQL's 63-byte limit")
    if not args.data_epoch.strip():
        raise PreparationError("data epoch must not be empty")
    if not args.expected_vector_extension_version.strip():
        raise PreparationError("expected vector extension version must not be empty")
    if not args.source_tag.strip():
        raise PreparationError("official source tag must not be empty")
    if args.expected_predicate_rows > args.expected_table_rows:
        raise PreparationError("predicate rows cannot exceed table rows")
    if args.manifest.is_dir():
        raise PreparationError("manifest path names a directory")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = run(args)
    except PreparationError as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
