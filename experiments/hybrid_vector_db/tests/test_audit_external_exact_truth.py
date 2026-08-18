from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.hybrid_vector_db.scripts import audit_external_exact_truth as audit
from experiments.hybrid_vector_db.scripts import pgvector_d2_cache_isolation_control as d2


BUILD = "sqlens-v16-external-truth-test"
SHA = "a" * 64
TABLE = "public.items"
INDEX = "public.items_embedding_hnsw"
QUERY_TABLE = "public.queries"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    filters_path = tmp_path / "filters.csv"
    truth_path = tmp_path / "truth.csv"
    filters = [
        {
            "filter_name": f"f{index}",
            "target_rate": 50 - index,
            "actual_pct": 50 - index,
            "expected_rows": 100 + index,
            "predicate": f"tag = {index}",
            "atoms": f"sql:tag = {index}",
        }
        for index in range(14)
    ]
    write_csv(filters_path, filters)
    distances = [index / 10 for index in range(1, 11)]
    truth: list[dict[str, object]] = []
    for query_no in range(4):
        for filter_no, filter_row in enumerate(filters):
            ids = [query_no * 100 + filter_no * 10 + index for index in range(10)]
            truth.append(
                {
                    "query_no": query_no,
                    "query_id": 1000 + query_no,
                    "query_split": "calibration" if query_no < 2 else "final",
                    "filter_name": filter_row["filter_name"],
                    "target_rate": filter_row["target_rate"],
                    "predicate": filter_row["predicate"],
                    "candidate_validity_predicate": "TRUE",
                    "method": "pre_filter_exact",
                    "k": 10,
                    "recall_at_10_exact_filtered": 1.0,
                    "returned": 10,
                    "candidates": filter_row["expected_rows"],
                    "filtered_rows": filter_row["expected_rows"],
                    "search_candidate_rows": filter_row["expected_rows"],
                    "result_ids": ",".join(str(value) for value in ids),
                    "exact_filtered_topk_ids": ",".join(str(value) for value in ids),
                    "exact_filtered_topk_distances_sq": ",".join(
                        str(value) for value in distances
                    ),
                    "kth_distance_sq": 1.0,
                    "tie_tolerance": 0.000001,
                    "strict_closer_count": 9,
                    "boundary_tied": False,
                    "self_excluded": False,
                    "candidate_rows": filter_row["expected_rows"],
                    "self_excluded_rows": 0,
                }
            )
    write_csv(truth_path, truth)
    return filters_path, truth_path


def make_args(tmp_path: Path, filters: Path, truth: Path) -> argparse.Namespace:
    return argparse.Namespace(
        filters_csv=filters,
        truth_csv=truth,
        out=tmp_path / "truth.manifest.json",
        table=TABLE,
        source_index=INDEX,
        query_table=QUERY_TABLE,
        table_id_column="id",
        table_vector_column="embedding",
        query_id_column="qid",
        query_vector_column="embedding",
        candidate_validity_predicate="TRUE",
        self_excluded=False,
        calibration_offset=0,
        calibration_queries=2,
        final_offset=2,
        final_queries=2,
        expected_sqlens_build_id=BUILD,
        expected_vector_so_sha256=SHA,
        old_launch_manifest=None,
    )


class FakeCursor:
    def __init__(self, responses: list[list[tuple[Any, ...]]]) -> None:
        self.responses = list(responses)
        self.current: list[tuple[Any, ...]] = []

    def execute(self, _query: object, _params: object = None) -> None:
        if not self.responses:
            raise AssertionError("unexpected database query")
        self.current = self.responses.pop(0)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.current[0] if self.current else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self.current)


class FakeConnection:
    def __init__(self, responses: list[list[tuple[Any, ...]]]) -> None:
        self.fake_cursor = FakeCursor(responses)

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, exc_type: object, *_args: object) -> None:
        if exc_type is None and self.fake_cursor.responses:
            raise AssertionError(
                f"database audit left {len(self.fake_cursor.responses)} scripted responses"
            )

    def cursor(self) -> FakeCursor:
        return self.fake_cursor


def database_responses(
    expected_rows: list[int],
    *,
    build: str = BUILD,
    sha: str = SHA,
    table_oid: int = 101,
) -> list[list[tuple[Any, ...]]]:
    return [
        [["16.4", "0.8.0", build, "/usr/lib/postgresql/vector.so", sha]],
        [[table_oid, TABLE, 1001, "r", 1000, 8192]],
        [[102, QUERY_TABLE, 1002, "r", 4, 8192]],
        [["id", "bigint", True], ["embedding", "vector(128)", False]],
        [["qid", "bigint", True], ["embedding", "vector(128)", False]],
        [
            [
                201,
                INDEX,
                2001,
                table_oid,
                TABLE,
                "hnsw",
                True,
                True,
                True,
                f"CREATE INDEX items_embedding_hnsw ON {TABLE} USING hnsw (embedding)",
                None,
                1,
                "embedding",
            ]
        ],
        [[1000, 1], [1001, 1], [1002, 1], [1003, 1]],
        *[[[value]] for value in expected_rows],
    ]


def fake_connect(responses: list[list[tuple[Any, ...]]]):
    def connect(_conninfo: str, *, autocommit: bool) -> FakeConnection:
        assert autocommit is True
        return FakeConnection(responses)

    return connect


def mutate_csv(path: Path, mutate) -> None:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    mutate(rows)
    write_csv(path, rows)


def old_launch(
    path: Path,
    filters: Path,
    truth: Path,
    *,
    status: str = "interrupted",
    table_oid: int = 101,
) -> None:
    payload = {
        "status": status,
        "ready": True,
        "dataset": {
            "table": TABLE,
            "index": INDEX,
            "query_table": QUERY_TABLE,
            "query_id_column": "qid",
            "query_vector_column": "embedding",
            "filter_names": [f"f{index}" for index in range(14)],
        },
        "database": {
            "ready": True,
            "errors": [],
            "index": INDEX,
            "relations": {
                TABLE: {"oid": table_oid},
                QUERY_TABLE: {"oid": 102},
                INDEX: {"oid": 201},
            },
        },
        "truth": {
            "ready": True,
            "errors": [],
            "path": str(truth),
            "sha256": sha256(truth),
            "row_count": 56,
            "query_count": 4,
        },
        "filters": {
            "ready": True,
            "errors": [],
            "path": str(filters),
            "sha256": sha256(filters),
            "count": 14,
        },
        "protocol": {
            "candidate_validity_predicate": "TRUE",
            "truth_self_excluded": False,
            "calibration": {"offset": 0, "queries": 2},
            "final": {"offset": 2, "queries": 2},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_valid_manifest(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, Any]]:
    filters, truth = fixture_files(tmp_path)
    args = make_args(tmp_path, filters, truth)
    responses = database_responses([100 + index for index in range(14)])
    manifest = audit.build_manifest(args, connect_factory=fake_connect(responses))
    return args, manifest


def test_builds_d2_compatible_manifest_with_live_database_and_runtime_audit(
    tmp_path: Path,
) -> None:
    args, manifest = build_valid_manifest(tmp_path)
    assert manifest["artifact_valid"] is True
    assert manifest["recall_contract"] == audit.RECALL_CONTRACT
    assert manifest["outputs"]["truth_csv"]["row_count"] == 56
    assert manifest["inputs"]["filters_csv"]["row_count"] == 14
    assert (
        manifest["inputs"]["postgres"]["query_population"][
            "candidate_validity_predicate"
        ]
        == "TRUE"
    )
    assert manifest["database"]["source_index"]["access_method"] == "hnsw"
    assert manifest["database"]["source_index"]["valid"] is True
    assert manifest["database"]["query_ids"]["all_present_once"] is True
    assert manifest["runtime"]["build_id_exact_match"] is True

    audit.atomic_write_json_exclusive(args.out, manifest)
    consumed = d2.audit_exact_truth_manifest(
        args.out,
        args.truth_csv,
        args.filters_csv,
        expected_table=TABLE,
        expected_index=INDEX,
        expected_query_table=QUERY_TABLE,
        expected_query_id_column="qid",
        expected_query_vector_column="embedding",
        expected_candidate_validity_predicate="TRUE",
        expected_self_excluded=False,
        query_offset=2,
        queries=2,
        expected_filter_names=[f"f{index}" for index in range(14)],
    )
    assert consumed["artifact_valid"] is True
    assert consumed["provenance_kind"] == "exact_truth_manifest"


def test_rejects_incomplete_truth_matrix(tmp_path: Path) -> None:
    filters_path, truth_path = fixture_files(tmp_path)
    mutate_csv(truth_path, lambda rows: rows.pop())
    filters, _ = audit.audit_filters(filters_path)
    with pytest.raises(audit.AuditError, match="full 14-filter cohort"):
        audit.audit_truth(
            truth_path,
            filters,
            audit.CohortSpec(0, 2, 2, 2),
            candidate_validity_predicate="TRUE",
            self_excluded=False,
        )


def test_rejects_truth_predicate_drift(tmp_path: Path) -> None:
    filters_path, truth_path = fixture_files(tmp_path)
    mutate_csv(truth_path, lambda rows: rows[0].__setitem__("predicate", "tag = 99"))
    filters, _ = audit.audit_filters(filters_path)
    with pytest.raises(audit.AuditError, match="predicate differs"):
        audit.audit_truth(
            truth_path,
            filters,
            audit.CohortSpec(0, 2, 2, 2),
            candidate_validity_predicate="TRUE",
            self_excluded=False,
        )


def test_rejects_tie_contract_drift(tmp_path: Path) -> None:
    filters_path, truth_path = fixture_files(tmp_path)
    mutate_csv(truth_path, lambda rows: rows[0].__setitem__("tie_tolerance", "0.1"))
    filters, _ = audit.audit_filters(filters_path)
    with pytest.raises(audit.AuditError, match="tie tolerance"):
        audit.audit_truth(
            truth_path,
            filters,
            audit.CohortSpec(0, 2, 2, 2),
            candidate_validity_predicate="TRUE",
            self_excluded=False,
        )


def test_rejects_loaded_binary_mismatch(tmp_path: Path) -> None:
    filters_path, truth_path = fixture_files(tmp_path)
    args = make_args(tmp_path, filters_path, truth_path)
    filters, _ = audit.audit_filters(filters_path)
    query_ids, _ = audit.audit_truth(
        truth_path,
        filters,
        audit.CohortSpec(0, 2, 2, 2),
        candidate_validity_predicate="TRUE",
        self_excluded=False,
    )
    responses = database_responses(
        [100 + index for index in range(14)], build="wrong-build"
    )
    with pytest.raises(audit.AuditError, match="loaded SQLens"):
        audit.audit_database(
            args, filters, query_ids, connect_factory=fake_connect(responses)
        )


def test_rejects_real_database_count_mismatch(tmp_path: Path) -> None:
    filters_path, truth_path = fixture_files(tmp_path)
    args = make_args(tmp_path, filters_path, truth_path)
    filters, _ = audit.audit_filters(filters_path)
    query_ids, _ = audit.audit_truth(
        truth_path,
        filters,
        audit.CohortSpec(0, 2, 2, 2),
        candidate_validity_predicate="TRUE",
        self_excluded=False,
    )
    counts = [100 + index for index in range(14)]
    counts[6] += 1
    responses = database_responses(counts)
    with pytest.raises(audit.AuditError, match="database COUNT"):
        audit.audit_database(
            args, filters, query_ids, connect_factory=fake_connect(responses)
        )


def test_interrupted_launch_is_accepted_when_ready_sha_and_oids_match(
    tmp_path: Path,
) -> None:
    filters_path, truth_path = fixture_files(tmp_path)
    launch_path = tmp_path / "launch.json"
    old_launch(launch_path, filters_path, truth_path)
    args = make_args(tmp_path, filters_path, truth_path)
    args.old_launch_manifest = launch_path
    responses = database_responses([100 + index for index in range(14)])
    manifest = audit.build_manifest(args, connect_factory=fake_connect(responses))
    provenance = manifest["audits"]["source_provenance"]
    assert provenance["launch_status"] == "interrupted"
    assert provenance["interrupted_benchmark_accepted"] is True
    assert provenance["artifact_sha256_matches"] is True
    assert provenance["relation_oid_matches"] is True


def test_interrupted_launch_rejects_stale_relation_oid(tmp_path: Path) -> None:
    filters_path, truth_path = fixture_files(tmp_path)
    launch_path = tmp_path / "launch.json"
    old_launch(launch_path, filters_path, truth_path, table_oid=999)
    args = make_args(tmp_path, filters_path, truth_path)
    args.old_launch_manifest = launch_path
    responses = database_responses([100 + index for index in range(14)])
    with pytest.raises(audit.AuditError, match="OID mismatch"):
        audit.build_manifest(args, connect_factory=fake_connect(responses))


def test_interrupted_launch_rejects_unready_truth_section(tmp_path: Path) -> None:
    filters_path, truth_path = fixture_files(tmp_path)
    launch_path = tmp_path / "launch.json"
    old_launch(launch_path, filters_path, truth_path)
    payload = json.loads(launch_path.read_text(encoding="utf-8"))
    payload["truth"]["ready"] = False
    launch_path.write_text(json.dumps(payload), encoding="utf-8")
    args = make_args(tmp_path, filters_path, truth_path)
    args.old_launch_manifest = launch_path
    responses = database_responses([100 + index for index in range(14)])
    with pytest.raises(audit.AuditError, match="truth section"):
        audit.build_manifest(args, connect_factory=fake_connect(responses))


def test_atomic_writer_never_overwrites_existing_manifest(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    output.write_text("original\n", encoding="utf-8")
    with pytest.raises(audit.AuditError, match="refusing to overwrite"):
        audit.atomic_write_json_exclusive(output, {"artifact_valid": True})
    assert output.read_text(encoding="utf-8") == "original\n"
