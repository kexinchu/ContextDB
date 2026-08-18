from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.hybrid_vector_db.scripts import prepare_pgvector_official_index as prepare
from experiments.hybrid_vector_db.scripts import pgvector_upstream_overhead_control


SHA256 = "a" * 64
COMMIT = "b" * 40


def args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "dsn": "",
        "table": prepare.DEFAULT_TABLE,
        "index": prepare.DEFAULT_INDEX,
        "data_epoch": "amazon10m-frozen-r36",
        "expected_vector_so_sha256": SHA256,
        "expected_vector_extension_version": "0.8.2",
        "source_repo": Path("/tmp/pgvector-upstream-v0.8.2"),
        "source_tag": "v0.8.2",
        "source_commit": COMMIT,
        "filters_csv": Path("filters.csv"),
        "calibration_workload_csv": Path("calibration.csv"),
        "measurement_workload_csv": Path("measurement.csv"),
        "truth_csv": Path("truth.csv"),
        "expected_table_rows": 10_000_000,
        "expected_predicate_rows": 9_979_556,
        "maintenance_work_mem": "128GB",
        "manifest": Path("official-index.json"),
        "resume": False,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def table() -> prepare.TableState:
    return prepare.TableState(
        name=prepare.DEFAULT_TABLE,
        oid=10,
        relfilenode=20,
        physical_relfilenode=20,
        relation_filepath="base/1/20",
        relation_size_bytes=123_456,
        total_rows=10_000_000,
        predicate_rows=9_979_556,
    )


def request() -> dict[str, object]:
    return {
        "artifact_contract": prepare.ARTIFACT_CONTRACT,
        "table": prepare.DEFAULT_TABLE,
        "create_sql": prepare.create_index_sql(),
    }


def index(
    *,
    comment: str | None = None,
    predicate: str = "embedding_valid",
    reloptions: tuple[str, ...] = ("m=32", "ef_construction=200"),
) -> prepare.IndexState:
    if comment is None:
        comment = prepare.provenance_comment(
            request(),
            build_started_at="2026-07-30T00:00:00+00:00",
            build_completed_at="2026-07-30T00:01:00+00:00",
            build_wall_seconds=60.0,
        )
    return prepare.IndexState(
        name=prepare.DEFAULT_INDEX,
        oid=30,
        relfilenode=40,
        physical_relfilenode=40,
        relation_filepath="base/1/40",
        relation_size_bytes=654_321,
        heap_oid=10,
        heap_relfilenode=20,
        valid=True,
        ready=True,
        live=True,
        access_method="hnsw",
        unique=False,
        primary=False,
        key_attributes=1,
        total_attributes=1,
        indexed_column="embedding",
        opclass="vector_l2_ops",
        predicate=predicate,
        reloptions=reloptions,
        comment=comment,
        definition=(
            "CREATE INDEX amazon10m_official_hnsw_m32ef200_source_idx "
            "ON public.amazon_grocery_reviews_10m_pgvector USING hnsw "
            "(embedding vector_l2_ops) WITH (m='32', ef_construction='200') "
            "WHERE embedding_valid"
        ),
    )


class FakeTransaction:
    def __enter__(self) -> "FakeTransaction":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


class FakeCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.closed = False

    def execute(self, statement: object, params: object = None) -> None:
        rendered = (
            statement.as_string(None)
            if hasattr(statement, "as_string")
            else str(statement)
        )
        self.statements.append(" ".join(rendered.split()))

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def transaction(self) -> FakeTransaction:
        return FakeTransaction()


class ContextConnection(FakeConnection):
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> "ContextConnection":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def cursor(self) -> FakeCursor:
        return self._cursor


class SqlAndArgumentsTests(unittest.TestCase):
    def test_create_sql_is_dedicated_partial_source_order_contract(self) -> None:
        statement = prepare.create_index_sql()
        self.assertIn(
            'USING hnsw ("embedding" vector_l2_ops)', statement
        )
        self.assertIn("WITH (m = 32, ef_construction = 200)", statement)
        self.assertTrue(statement.endswith("WHERE embedding_valid"))
        self.assertNotIn("CONCURRENTLY", statement)
        self.assertNotIn("IF NOT EXISTS", statement)
        self.assertNotIn("DROP", statement.upper())

    def test_parser_pins_official_defaults_and_requires_provenance_inputs(self) -> None:
        parsed = prepare.build_parser().parse_args(
            [
                "--data-epoch",
                "frozen",
                "--source-repo",
                "/tmp/source",
                "--source-commit",
                COMMIT,
                "--filters-csv",
                "filters.csv",
                "--calibration-workload-csv",
                "cal.csv",
                "--measurement-workload-csv",
                "measure.csv",
                "--truth-csv",
                "truth.csv",
            ]
        )
        self.assertEqual(parsed.expected_vector_so_sha256, prepare.DEFAULT_VECTOR_SO_SHA256)
        self.assertEqual(parsed.source_tag, "v0.8.2")
        self.assertEqual(parsed.expected_vector_extension_version, "0.8.2")
        self.assertEqual(parsed.expected_predicate_rows, 9_979_556)

    def test_validate_args_rejects_cross_schema_and_invalid_population(self) -> None:
        with self.assertRaisesRegex(prepare.PreparationError, "table's schema"):
            prepare.validate_args(args(index="other.official_idx"))
        with self.assertRaisesRegex(prepare.PreparationError, "cannot exceed"):
            prepare.validate_args(
                args(expected_table_rows=10, expected_predicate_rows=11)
            )

    def test_dry_run_reads_nothing_and_never_connects_or_writes(self) -> None:
        parsed = args(dry_run=True)
        with (
            mock.patch.object(
                prepare, "source_checkout_identity", side_effect=AssertionError
            ),
            mock.patch.object(
                prepare, "input_artifact_identity", side_effect=AssertionError
            ),
            mock.patch.object(
                prepare, "atomic_write_json", side_effect=AssertionError
            ),
            mock.patch.object(
                prepare, "pg_config_from_env", side_effect=AssertionError
            ),
        ):
            payload = prepare.run(
                parsed,
                connect=mock.Mock(side_effect=AssertionError),
                source_probe=mock.Mock(side_effect=AssertionError),
                file_hasher=mock.Mock(side_effect=AssertionError),
            )
        self.assertTrue(payload["dry_run"])
        self.assertFalse(payload["database_connected"])
        self.assertFalse(payload["binary_switched"])
        self.assertIn("CREATE INDEX", payload["create_sql"])


class ProvenanceTests(unittest.TestCase):
    def test_source_checkout_requires_head_tag_commit_and_clean_tree(self) -> None:
        outputs = iter((COMMIT, COMMIT, ""))

        def runner(*_argv: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, next(outputs), "")

        with tempfile.TemporaryDirectory() as temporary:
            identity = prepare.source_checkout_identity(
                Path(temporary), "v0.8.2", COMMIT, command_runner=runner
            )
        self.assertTrue(identity["verified"])
        self.assertEqual(identity["tag_commit"], COMMIT)
        self.assertTrue(identity["tracked_tree_clean"])

    def test_source_checkout_rejects_dirty_or_wrong_tag(self) -> None:
        def dirty(*_argv: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            dirty.calls += 1
            outputs = (COMMIT, COMMIT, " M src/hnsw.c")
            return subprocess.CompletedProcess([], 0, outputs[dirty.calls - 1], "")

        dirty.calls = 0
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(prepare.PreparationError, "tracked modifications"):
                prepare.source_checkout_identity(
                    Path(temporary), "v0.8.2", COMMIT, command_runner=dirty
                )

    def test_server_identity_accepts_sha_fallback_without_build_id(self) -> None:
        cursor = mock.Mock()
        cursor.fetchone.return_value = (
            "16.3",
            160003,
            "0.8.2",
            "/usr/lib/postgresql/16/lib/vector.so",
            SHA256,
            None,
        )
        identity = prepare.server_identity(
            cursor,
            expected_extension_version="0.8.2",
            expected_vector_so_sha256=SHA256,
        )
        self.assertEqual(identity["binary_identity_method"], "server_file_sha256")
        self.assertTrue(identity["controller_evidence_required"])

    def test_server_identity_rejects_extension_or_binary_mismatch(self) -> None:
        cursor = mock.Mock()
        cursor.fetchone.return_value = (
            "16.3", 160003, "0.8.1", "/x/vector.so", SHA256, None
        )
        with self.assertRaisesRegex(prepare.PreparationError, "extension version"):
            prepare.server_identity(
                cursor,
                expected_extension_version="0.8.2",
                expected_vector_so_sha256=SHA256,
            )
        cursor.fetchone.return_value = (
            "16.3", 160003, "0.8.2", "/x/vector.so", "c" * 64, None
        )
        with self.assertRaisesRegex(prepare.PreparationError, "SHA256 mismatch"):
            prepare.server_identity(
                cursor,
                expected_extension_version="0.8.2",
                expected_vector_so_sha256=SHA256,
            )


class ExistingIndexTests(unittest.TestCase):
    def test_matching_existing_index_is_idempotent(self) -> None:
        provenance = prepare.validate_existing_index(index(), table(), request())
        self.assertEqual(provenance["request_contract"], request())
        self.assertEqual(provenance["build"]["wall_seconds"], 60.0)

    def test_mismatched_existing_index_fails_closed(self) -> None:
        with self.assertRaisesRegex(prepare.PreparationError, "refusing to modify or drop"):
            prepare.validate_existing_index(
                index(predicate="NOT embedding_valid"), table(), request()
            )
        with self.assertRaisesRegex(prepare.PreparationError, "lacks its provenance"):
            prepare.validate_existing_index(
                index(comment="built elsewhere"), table(), request()
            )

    def test_prepare_existing_emits_no_create_or_drop(self) -> None:
        cursor = FakeCursor()
        existing = index()
        with (
            mock.patch.object(
                prepare, "table_state", side_effect=[table(), table()]
            ),
            mock.patch.object(prepare, "index_state", return_value=existing),
        ):
            _, observed, _, created = prepare.prepare_index(
                FakeConnection(), cursor, args(), request()
            )
        self.assertEqual(observed, existing)
        self.assertFalse(created)
        statements = " ".join(cursor.statements).upper()
        self.assertNotIn("CREATE INDEX", statements)
        self.assertNotIn("DROP", statements)

    def test_new_build_records_timing_and_comment_in_one_transaction(self) -> None:
        cursor = FakeCursor()
        no_comment = index(comment=None)
        # Explicitly remove the helper-generated comment for the post-CREATE read.
        no_comment = prepare.IndexState(**(asdict_index(no_comment) | {"comment": None}))
        final = index()
        times = iter((10.0, 12.5))
        with (
            mock.patch.object(
                prepare, "table_state", side_effect=[table(), table()]
            ),
            mock.patch.object(
                prepare, "index_state", side_effect=[None, no_comment, final]
            ),
            mock.patch.object(
                prepare,
                "provenance_comment",
                wraps=prepare.provenance_comment,
            ) as comment_builder,
        ):
            _, _, provenance, created = prepare.prepare_index(
                FakeConnection(),
                cursor,
                args(),
                request(),
                monotonic=lambda: next(times),
            )
        self.assertTrue(created)
        self.assertEqual(provenance["build"]["wall_seconds"], 60.0)
        self.assertEqual(
            comment_builder.call_args.kwargs["build_wall_seconds"], 2.5
        )
        statements = " ".join(cursor.statements)
        self.assertIn("CREATE INDEX", statements)
        self.assertIn("COMMENT ON INDEX", statements)
        self.assertNotIn("DROP", statements.upper())


def asdict_index(value: prepare.IndexState) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in value.__dataclass_fields__
    }


class ManifestTests(unittest.TestCase):
    def test_manifest_is_compatible_with_existing_p0_2_loader(self) -> None:
        source = {
            "repo": "/tmp/source",
            "source_tag": "v0.8.2",
            "source_commit": COMMIT,
            "tag_commit": COMMIT,
            "tracked_tree_clean": True,
        }
        server = {
            "vector_extension_version": "0.8.2",
            "vector_so_sha256": SHA256,
            "vector_so_path": "/x/vector.so",
            "binary_identity_method": "server_file_sha256",
            "vector_build_id_function": None,
            "controller_evidence_required": True,
            "postgresql_version": "16.3",
            "postgresql_version_num": 160003,
        }
        provenance = prepare.parse_provenance_comment(index().comment)
        payload = prepare.manifest_payload(
            args(),
            source=source,
            inputs={"filters_csv": {"sha256": SHA256}},
            server=server,
            table=table(),
            index=index(),
            provenance=provenance,
            created=True,
        )
        self.assertTrue(payload["artifact_valid"])
        self.assertEqual(payload["builder"]["source_tag"], "v0.8.2")
        self.assertEqual(payload["builder"]["vector_so_sha256"], SHA256)
        self.assertEqual(payload["index_fingerprint"]["index_oid"], 30)
        self.assertRegex(
            payload["index_fingerprint"]["indexdef_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(payload["index_fingerprint"]["predicate_rows"], 9_979_556)

    def test_manifest_loads_through_existing_p0_2_contract(self) -> None:
        source = {
            "repo": "/tmp/source",
            "source_tag": "v0.8.2",
            "source_commit": COMMIT,
            "tag_commit": COMMIT,
            "tracked_tree_clean": True,
        }
        server = {
            "vector_extension_version": "0.8.2",
            "vector_so_sha256": prepare.DEFAULT_VECTOR_SO_SHA256,
            "vector_so_path": "/x/vector.so",
            "binary_identity_method": "server_file_sha256",
            "vector_build_id_function": None,
            "controller_evidence_required": True,
            "postgresql_version": "16.3",
            "postgresql_version_num": 160003,
        }
        provenance = prepare.parse_provenance_comment(index().comment)
        payload = prepare.manifest_payload(
            args(expected_vector_so_sha256=prepare.DEFAULT_VECTOR_SO_SHA256),
            source=source,
            inputs={"filters_csv": {"sha256": SHA256}},
            server=server,
            table=table(),
            index=index(),
            provenance=provenance,
            created=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            identity = (
                pgvector_upstream_overhead_control.load_official_index_build_identity(
                    path,
                    table=prepare.DEFAULT_TABLE,
                    index=prepare.DEFAULT_INDEX,
                    official_source_commit=COMMIT,
                )
            )
        self.assertTrue(identity["artifact_valid"])
        self.assertEqual(identity["index_fingerprint"]["index_oid"], 30)

    def test_atomic_writer_replaces_complete_json_and_leaves_no_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            prepare.atomic_write_json(path, {"artifact_valid": True})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"artifact_valid": True},
            )
            self.assertEqual(list(Path(temporary).glob(".*.tmp")), [])

    def test_existing_manifest_requires_resume_before_any_connection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(prepare.PreparationError, "--resume"):
                prepare.run(
                    args(manifest=path),
                    connect=mock.Mock(side_effect=AssertionError),
                    source_probe=mock.Mock(side_effect=AssertionError),
                )

    def test_execute_then_resume_composes_all_gates_and_preserves_manifest(self) -> None:
        source_identity: dict[str, object]
        server = {
            "postgresql_version": "16.3",
            "postgresql_version_num": 160003,
            "vector_extension_version": "0.8.2",
            "vector_so_path": "/x/vector.so",
            "vector_so_sha256": SHA256,
            "vector_build_id_function": None,
            "binary_identity_method": "server_file_sha256",
            "controller_evidence_required": True,
            "exact_match": True,
            "checked_at": "ignored",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_repo = root / "source"
            source_repo.mkdir()
            source_identity = {
                "repo": str(source_repo.resolve()),
                "source_tag": "v0.8.2",
                "source_commit": COMMIT,
                "head_commit": COMMIT,
                "tag_commit": COMMIT,
                "tracked_tree_clean": True,
                "verified": True,
            }
            inputs = {}
            for name in ("filters", "calibration", "measurement", "truth"):
                path = root / f"{name}.csv"
                path.write_text(f"{name}\n", encoding="utf-8")
                inputs[name] = path
            manifest = root / "manifest.json"
            parsed = args(
                dsn="postgresql://example",
                source_repo=source_repo,
                filters_csv=inputs["filters"],
                calibration_workload_csv=inputs["calibration"],
                measurement_workload_csv=inputs["measurement"],
                truth_csv=inputs["truth"],
                manifest=manifest,
            )
            cursor = FakeCursor()
            connection = ContextConnection(cursor)

            def prepared(
                _conn: object,
                _cur: object,
                _args: object,
                contract: dict[str, object],
            ) -> tuple[
                prepare.TableState,
                prepare.IndexState,
                dict[str, object],
                bool,
            ]:
                provenance = prepare.parse_provenance_comment(
                    prepare.provenance_comment(
                        contract,
                        build_started_at="start",
                        build_completed_at="complete",
                        build_wall_seconds=42.0,
                    )
                )
                return table(), index(), provenance, True

            with (
                mock.patch.object(prepare, "server_identity", return_value=server),
                mock.patch.object(prepare, "acquire_lock"),
                mock.patch.object(prepare, "release_lock"),
                mock.patch.object(
                    prepare, "table_state", side_effect=[table(), table()]
                ),
                mock.patch.object(prepare, "prepare_index", side_effect=prepared),
            ):
                payload = prepare.run(
                    parsed,
                    connect=mock.Mock(return_value=connection),
                    source_probe=mock.Mock(return_value=source_identity),
                    file_hasher=mock.Mock(return_value=SHA256),
                )
            self.assertTrue(payload["artifact_valid"])
            self.assertTrue(manifest.is_file())

            parsed.resume = True
            with (
                mock.patch.object(prepare, "server_identity", return_value=server),
                mock.patch.object(prepare, "acquire_lock"),
                mock.patch.object(prepare, "release_lock"),
                mock.patch.object(
                    prepare, "table_state", side_effect=[table(), table()]
                ),
                mock.patch.object(prepare, "prepare_index", side_effect=prepared),
            ):
                resumed = prepare.run(
                    parsed,
                    connect=mock.Mock(return_value=connection),
                    source_probe=mock.Mock(return_value=source_identity),
                    file_hasher=mock.Mock(return_value=SHA256),
                )
            self.assertEqual(resumed, payload)


if __name__ == "__main__":
    unittest.main()
