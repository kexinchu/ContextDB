from __future__ import annotations

import argparse

import pytest
from psycopg import sql

from experiments.hybrid_vector_db.scripts import prepare_pgvector_same_graph_bfs_clone as prep


def state(**overrides):
    values = {
        "name": "public.items_hnsw",
        "oid": 10,
        "relfilenode": 11,
        "heap_oid": 12,
        "heap_name": "public.items",
        "valid": True,
        "ready": True,
        "live": True,
        "access_method": "hnsw",
        "column": "embedding",
        "opclass": "vector_l2_ops",
        "predicate": None,
        "reloptions": ("m=16", "ef_construction=100"),
        "blocks": 100,
        "bytes": 819200,
        "definition": "CREATE INDEX ...",
        "comment": None,
    }
    values.update(overrides)
    return prep.IndexState(**values)


def test_semantic_contract_accepts_distinct_same_heap_clone():
    source = state()
    clone = state(name="public.items_hnsw_bfs", oid=20, relfilenode=21)
    prep.semantic_contract(source, clone)


def test_semantic_contract_rejects_different_heap():
    source = state()
    clone = state(
        name="public.items_hnsw_bfs",
        oid=20,
        relfilenode=21,
        heap_oid=99,
        heap_name="public.items_bfs",
    )
    with pytest.raises(prep.PreparationError, match="differs from source"):
        prep.semantic_contract(source, clone)


def test_source_contract_rejects_invalid_source():
    with pytest.raises(prep.PreparationError, match="valid, ready, and live"):
        prep.source_contract(state(valid=False), "public.items")


def test_create_index_sql_preserves_predicate_and_hnsw_options():
    statement = prep.create_index_sql(
        state(predicate="embedding_valid"), "public.items_hnsw_bfs"
    )
    rendered = statement.as_string(None)
    assert 'CREATE INDEX "items_hnsw_bfs" ON "public"."items"' in rendered
    assert '"embedding" "vector_l2_ops"' in rendered
    assert "m = 16" in rendered
    assert "ef_construction = 100" in rendered
    assert "WHERE embedding_valid" in rendered


def test_parse_args_rejects_same_index(tmp_path):
    with pytest.raises(SystemExit):
        prep.parse_args(
            [
                "--table",
                "public.items",
                "--source-index",
                "public.idx",
                "--clone-index",
                "idx",
                "--out",
                str(tmp_path / "proof.json"),
                "--expected-sqlens-build-id",
                "sqlens-test",
                "--expected-vector-so-sha256",
                "a" * 64,
            ]
        )


def test_relation_comment_binds_binary_and_source():
    args = argparse.Namespace(
        maintenance_work_mem="128GB",
        expected_sqlens_build_id="sqlens-v16-test",
        expected_vector_so_sha256="b" * 64,
    )
    comment = prep.relation_comment(args, state())
    assert '"build_page_order":"bfs"' in comment
    assert '"source_index":"public.items_hnsw"' in comment
    assert '"vector_so_sha256":"' + "b" * 64 + '"' in comment


def test_timing_measurements_records_actual_clone_transaction():
    measured = prep.timing_measurements(
        stage="all",
        created=True,
        total_start_ns=1_000_000,
        total_finish_ns=21_000_000,
        proof_start_ns=15_000_000,
        proof_finish_ns=19_000_000,
        creation_start_ns=3_000_000,
        creation_finish_ns=13_000_000,
    )

    assert measured["clock"] == "time.monotonic_ns"
    assert measured["clone_creation_transaction"] == {
        "status": "measured",
        "elapsed_ms": 10.0,
        "reason": None,
        "semantics": measured["clone_creation_transaction"]["semantics"],
    }
    assert measured["graph_proof"]["elapsed_ms"] == 4.0
    assert measured["total_prepare"]["elapsed_ms"] == 20.0


@pytest.mark.parametrize(
    ("stage", "expected_reason"),
    [
        ("all", "existing_valid_clone_reused"),
        ("proof", "proof_only_existing_clone"),
    ],
)
def test_timing_measurements_never_fabricates_reused_clone_time(
    stage, expected_reason
):
    measured = prep.timing_measurements(
        stage=stage,
        created=False,
        total_start_ns=1,
        total_finish_ns=101,
        proof_start_ns=20,
        proof_finish_ns=80,
        creation_start_ns=None,
        creation_finish_ns=None,
    )

    creation = measured["clone_creation_transaction"]
    assert creation["status"] == "not_measured"
    assert creation["elapsed_ms"] is None
    assert creation["reason"] == expected_reason


def test_timing_measurements_fails_closed_on_missing_or_stale_measurement():
    with pytest.raises(prep.PreparationError, match="missing its CREATE INDEX"):
        prep.timing_measurements(
            stage="all",
            created=True,
            total_start_ns=1,
            total_finish_ns=100,
            proof_start_ns=20,
            proof_finish_ns=80,
            creation_start_ns=None,
            creation_finish_ns=None,
        )

    with pytest.raises(prep.PreparationError, match="monotonic timing is invalid"):
        prep.timing_measurements(
            stage="all",
            created=False,
            total_start_ns=100,
            total_finish_ns=100,
            proof_start_ns=20,
            proof_finish_ns=80,
            creation_start_ns=None,
            creation_finish_ns=None,
        )


def test_storage_measurements_reports_bytes_blocks_and_ratios():
    measured = prep.storage_measurements(
        state(blocks=100, bytes=819_200),
        state(
            name="public.items_hnsw_bfs",
            oid=20,
            relfilenode=21,
            blocks=125,
            bytes=1_024_000,
        ),
    )

    assert measured["source"] == {"bytes": 819_200, "blocks": 100}
    assert measured["clone"] == {"bytes": 1_024_000, "blocks": 125}
    assert measured["clone_to_source_storage_ratio"] == pytest.approx(1.25)
    assert measured["clone_to_source_bytes_ratio"] == pytest.approx(1.25)
    assert measured["clone_to_source_blocks_ratio"] == pytest.approx(1.25)


def test_storage_measurements_rejects_empty_relation():
    with pytest.raises(prep.PreparationError, match="empty clone"):
        prep.storage_measurements(
            state(),
            state(name="public.items_hnsw_bfs", oid=20, relfilenode=21, blocks=0, bytes=0),
        )
