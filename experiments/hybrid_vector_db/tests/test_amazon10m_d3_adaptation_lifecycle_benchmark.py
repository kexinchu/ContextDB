from __future__ import annotations

import json
import random
import sys
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import amazon10m_d3_adaptation_lifecycle_benchmark as runner  # noqa: E402


RUNTIME_BUILD_ID = "sqlens-test-build"
VECTOR_SO_PATH = "/usr/lib/postgresql/vector.so"
VECTOR_SO_SHA256 = "a" * 64
DATABASE_INDEX_FINGERPRINT = "b" * 64
SOURCE_FILE_SHA256 = {"hnsw.c": "c" * 64, "hnsw.h": "d" * 64}
SOURCE_AGGREGATE_SHA256 = runner.aggregate_source_file_sha256(SOURCE_FILE_SHA256)


def database_identity() -> dict[str, object]:
    return {
        "loaded_vector_sqlens_build_id": RUNTIME_BUILD_ID,
        "loaded_vector_so_path": VECTOR_SO_PATH,
        "loaded_vector_so_sha256": VECTOR_SO_SHA256,
        "database_index_fingerprint": DATABASE_INDEX_FINGERPRINT,
    }


def source_identity() -> dict[str, object]:
    database = database_contract_identity()
    return {
        "sqlens_source": {
            "source_root": "third_party/pgvector-sqlens/src",
            "file_globs": ["*.c", "*.h"],
            "file_count": len(SOURCE_FILE_SHA256),
            "file_sha256": dict(SOURCE_FILE_SHA256),
            "aggregate_sha256": SOURCE_AGGREGATE_SHA256,
            "declared_build_id": RUNTIME_BUILD_ID,
            "latest_source_mtime_ns": 1,
        },
        "local_vector_so": {
            "path": "third_party/pgvector-sqlens/vector.so",
            "sha256": VECTOR_SO_SHA256,
            "mtime_ns": 2,
            "built_after_source_tree": True,
        },
        "index_query_health": {
            "artifact_valid": True,
            "runtime": {
                "sqlens_build_id": RUNTIME_BUILD_ID,
                "vector_so_sha256": VECTOR_SO_SHA256,
            },
            "index_identity": {
                "oid": database["index_oid"],
                "relfilenode": database["index_relfilenode"],
                "definition": database["indexdef"],
            },
        },
    }


class IndexHealthBinaryBindingTest(unittest.TestCase):
    def test_requires_same_build_id_and_vector_binary(self) -> None:
        source = source_identity()
        health = source["index_query_health"]
        runner.validate_index_health_binary_binding(health, source)

        changed = json.loads(json.dumps(health))
        changed["runtime"]["vector_so_sha256"] = "f" * 64
        with self.assertRaisesRegex(runner.BenchmarkContractError, "different SQLens binary"):
            runner.validate_index_health_binary_binding(changed, source)

        changed = json.loads(json.dumps(health))
        changed["runtime"]["sqlens_build_id"] = "sqlens-other-build"
        with self.assertRaisesRegex(runner.BenchmarkContractError, "different SQLens binary"):
            runner.validate_index_health_binary_binding(changed, source)


def row_identity() -> dict[str, object]:
    return {
        "runtime_build_id": RUNTIME_BUILD_ID,
        "loaded_vector_so_path": VECTOR_SO_PATH,
        "loaded_vector_so_sha256": VECTOR_SO_SHA256,
        "database_index_fingerprint": DATABASE_INDEX_FINGERPRINT,
        "sqlens_source_aggregate_sha256": SOURCE_AGGREGATE_SHA256,
        "profile_reported_build_id": "unreported",
    }


def request_provenance() -> dict[str, object]:
    return {**database_identity(), "sqlens_source_aggregate_sha256": SOURCE_AGGREGATE_SHA256}


def persisted_reuse_evidence() -> dict[str, object]:
    return {
        "artifact_valid": True,
        "fresh_backend_distinct": True,
        "cache_started_empty": True,
        "all_materialized_filters_reloaded": True,
        "store_unchanged": True,
        "materialized_filter_count": 1,
    }


def database_contract_identity() -> dict[str, object]:
    identity = {
        "server_version": "17.5",
        "vector_extension": "0.8.2",
        "table": runner.DEFAULT_TABLE,
        "table_oid": 41,
        "table_relfilenode": 42,
        "index": runner.DEFAULT_INDEX,
        "index_oid": 43,
        "index_relfilenode": 44,
        "indexdef": "CREATE INDEX amazon10m_embedding_valid_source_idx ON public.amazon_grocery_reviews_10m_pgvector USING hnsw (embedding)",
        "index_predicate": "embedding_valid",
    }
    return {
        **identity,
        **database_identity(),
        "database_index_fingerprint": runner.canonical_sha256(identity),
    }


def filters() -> list[runner.FilterSpec]:
    return [runner.FilterSpec(f"f{number}", f"rating = {number}", (f"sql:rating = {number}",), 1, 1.0) for number in range(14)]


def truth(items: list[runner.FilterSpec], query_count: int = 200) -> dict[tuple[str, int], runner.TruthEntry]:
    return {(item.name, query_no): runner.TruthEntry(
                item.name, query_no, query_no, tuple(range(10)), 1.0, 0.0, 9, False,
                "calibration" if query_count == runner.FORMAL_TRUTH_QUERY_COUNT
                and query_no < runner.FORMAL_CALIBRATION_QUERY_COUNT else "final",
            )
            for item in items for query_no in range(query_count)}


class FakeSession:
    def __init__(self, profiles: list[dict[str, object]]) -> None:
        self.profiles = iter(profiles)
        self.calls: list[str] = []
        self.value: object = None

    def execute(self, sql: str, params=None) -> None:  # type: ignore[no-untyped-def]
        self.calls.append(sql)
        if "metadata_cache_profile" in sql:
            self.value = json.dumps(next(self.profiles))

    def one(self):  # type: ignore[no-untyped-def]
        return self.value

    def row(self):  # type: ignore[no-untyped-def]
        return self.value

    def all(self):  # type: ignore[no-untyped-def]
        return []


class FragmentStoreSession:
    def __init__(self, audits: list[tuple[object, ...]], records: list[list[tuple[str]]], deleted: int) -> None:
        self.audits = iter(audits)
        self.records = iter(records)
        self.deleted = deleted
        self.calls: list[tuple[str, object]] = []
        self.last_sql = ""

    def execute(self, sql: str, params=None) -> None:  # type: ignore[no-untyped-def]
        self.last_sql = sql
        self.calls.append((sql, params))

    def one(self):  # type: ignore[no-untyped-def]
        raise AssertionError("fragment-store audit uses row(), not one()")

    def row(self):  # type: ignore[no-untyped-def]
        return next(self.audits)

    def all(self):  # type: ignore[no-untyped-def]
        if "DELETE FROM" in self.last_sql:
            return [(1,)] * self.deleted
        return next(self.records)


class ReloadCursor:
    def __init__(self) -> None:
        self.value: tuple[object, ...] = (None,)

    def execute(self, sql: str, params=None) -> None:  # type: ignore[no-untyped-def]
        if "pg_backend_pid" in sql:
            self.value = (999,)
        elif "SELECT count(*)" in sql:
            self.value = (10,)

    def fetchone(self) -> tuple[object, ...]:
        return self.value

    def fetchall(self) -> list[tuple[object, ...]]:
        return []


class ReloadConnection:
    def __init__(self) -> None:
        self.cursor_value = ReloadCursor()
        self.closed = False

    def cursor(self) -> ReloadCursor:
        return self.cursor_value

    def close(self) -> None:
        self.closed = True


class Amazon10MD3AdaptationLifecycleTests(unittest.TestCase):
    def test_fresh_backend_reloads_online_fragment_without_rebuilding(self) -> None:
        args = runner.create_argument_parser().parse_args(["--requests", "1", "--truth-query-count", "1"])
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        exact = runner.TruthEntry("f", 0, 99, tuple(range(10)), 1.0, 0.0, 9, False, "final")
        adaptive_rows = [{
            "materialization_observed": True,
            "filter_name": "f",
            "request_no": 7,
            "query_no": 0,
            "query_id": 99,
            "guidance_profile_after": {"kind": "page"},
            "returned_ids": list(range(10)),
            "returned_distances_sq": [0.1] * 10,
            "recall_at_10": 1.0,
        }]
        store = {
            "exists": True, "count": 1, "content_sha256": "a" * 64,
            "epoch": 3, "relfilenode": 4, "epoch_proof": {"valid": True},
        }
        connection = ReloadConnection()
        psycopg = mock.Mock()
        psycopg.connect.return_value = connection
        empty_cache = {"entries": 0, "resident_entries": 0, "resident_bytes": 0}
        loaded_cache = {"entries": 1, "resident_entries": 1, "resident_bytes": 1024}
        expected_database = database_contract_identity()
        with mock.patch.object(runner, "database_provenance", return_value=expected_database), \
             mock.patch.object(runner, "configure"), \
             mock.patch.object(runner, "reset_guidance"), \
             mock.patch.object(runner, "json_profile", side_effect=[empty_cache, empty_cache, loaded_cache]), \
             mock.patch.object(runner, "audit_fragment_store", side_effect=[store, dict(store)]), \
             mock.patch.object(
                 runner, "activate",
                 return_value=({
                     "active": True, "kind": "page", "fragment_store_hits": 1,
                     "fragment_builds": 0,
                 }, 1.0),
             ), \
             mock.patch.object(
                 runner, "run_search",
                 return_value=(
                     list(range(10)), [0.1] * 10,
                     {"valid": True, "planner_proof_succeeded": True}, "", 2.0,
                 ),
             ):
            evidence = runner.audit_persisted_fragment_reload(
                psycopg, "dbname=test", args,
                adaptive_rows=adaptive_rows,
                filters_by_name={"f": spec}, truth={("f", 0): exact},
                existing_backend_pids=[10, 11, 12], expected_database=expected_database,
            )
        self.assertTrue(evidence["artifact_valid"])
        self.assertTrue(evidence["cache_started_empty"])
        self.assertTrue(evidence["fresh_backend_distinct"])
        self.assertTrue(evidence["store_unchanged"])
        self.assertEqual(evidence["per_filter"][0]["fragment_store_hits"], 1)
        self.assertEqual(evidence["per_filter"][0]["fragment_builds"], 0)
        self.assertTrue(connection.closed)

    def test_common_warmup_executes_on_every_persistent_mode_backend(self) -> None:
        args = runner.create_argument_parser().parse_args([
            "--requests", "2", "--window-size", "1", "--truth-query-count", "2",
        ])
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        exact = {
            ("f", query_no): runner.TruthEntry(
                "f", query_no, 100 + query_no, tuple(range(10)),
                1.0, 0.0, 9, False, "calibration",
            )
            for query_no in range(2)
        }
        backends: dict[str, runner.ModeBackend] = {}
        for number, mode in enumerate(runner.MODES):
            session = mock.Mock()
            session.one.return_value = 1
            backends[mode] = runner.ModeBackend(mode, mock.Mock(), session, 40 + number, {})
        empty = {"entries": 0, "resident_entries": 0, "resident_bytes": 0}
        with mock.patch.object(runner, "reset_guidance"), \
             mock.patch.object(runner, "configure") as configure, \
             mock.patch.object(runner, "json_profile", side_effect=[dict(empty) for _ in range(6)]), \
             mock.patch.object(
                 runner, "run_search",
                 return_value=(
                     list(range(10)), [0.1] * 10,
                     {"valid": True}, "", 1.0,
                 ),
             ) as run_search:
            evidence = runner.common_post_setup_warmup(backends, args, [spec], exact)
        self.assertTrue(evidence["all_mode_backends_executed"])
        self.assertTrue(evidence["all_backend_pids_distinct"])
        self.assertTrue(evidence["all_backend_d3_state_untouched"])
        self.assertEqual(evidence["calibration_queries_per_backend"], 2)
        self.assertEqual(run_search.call_count, 6)
        self.assertEqual(
            {call.args[2] for call in configure.call_args_list}, set(runner.MODES)
        )

    def test_extension_exports_monotonic_d3_event_and_build_timers(self) -> None:
        source = (
            Path(__file__).resolve().parents[3]
            / "third_party/pgvector-sqlens/src/vector.c"
        ).read_text(encoding="utf-8")
        self.assertIn('"\\\"adaptive_event_sequence\\\":" INT64_FORMAT', source)
        self.assertIn('"\\\"adaptive_fast_reactivation_hits\\\":" INT64_FORMAT', source)
        self.assertIn('"\\\"adaptive_fragment_build_ms\\\":%.6f,"', source)
        self.assertIn("hnsw_adaptive_profile.fragmentBuildMs +=", source)
        self.assertIn("hnsw_adaptive_profile.fastReactivationHits++;", source)
        self.assertIn("static double\nHnswMetadataSaveFragmentStore", source)
        self.assertIn(
            "cache->pageBuildMs += HnswMetadataSaveFragmentStore(", source
        )
        self.assertIn(
            "cache->bloomBuildMs += HnswMetadataSaveFragmentStore(", source
        )
        save_start = source.index("HnswMetadataSaveFragmentStore(Oid heapOid")
        save_end = source.index("static HnswMetadataCacheEntry *\nGetHnswMetadataPageCache", save_start)
        save_body = source[save_start:save_end]
        self.assertLess(
            save_body.index("INSTR_TIME_SET_CURRENT(start)"),
            save_body.index("HnswMetadataEnsureFragmentStore()"),
        )
        self.assertLess(
            save_body.index("SPI_execute_with_args("),
            save_body.index("INSTR_TIME_SET_CURRENT(elapsed)"),
        )
        adaptive_start = source.index("HnswGuidanceActivateAdaptive(")
        adaptive_end = source.index("bool\nHnswGuidanceIsActive", adaptive_start)
        adaptive_body = source[adaptive_start:adaptive_end]
        self.assertLess(
            adaptive_body.rindex("hnsw_active_guidance = nextGuidance;"),
            adaptive_body.rindex("hnsw_adaptive_profile.refinements++;"),
        )

    def test_real_fourteen_amazon_predicates_and_atoms_are_loaded_without_synthesis(self) -> None:
        specs = runner.load_filters(Path(__file__).resolve().parents[1] / "configs" / "amazon10m_selectivity14_filters.csv")
        self.assertEqual(len(specs), 14)
        self.assertEqual(specs[0].predicate, "item_rating_number >= 1000")
        self.assertEqual(specs[-1].atoms, ("sql:main_category = 'Grocery'", "sql:review_text_len >= 500"))
        self.assertFalse(any("%" in spec.predicate for spec in specs))

    def test_trace_is_deterministic_unique_vector_and_phase_shifted_hot_cold(self) -> None:
        specs = filters()
        exact = truth(specs)
        one = runner.build_trace(specs, exact, requests=200, window_size=20, seed=41)
        two = runner.build_trace(specs, exact, requests=200, window_size=20, seed=41)
        self.assertEqual(one, two)
        self.assertEqual({request.query_no for request in one}, set(range(200)))
        self.assertEqual(len({request.query_id for request in one}), 200)
        self.assertTrue(any(request.reuse_distance is not None for request in one))
        first = Counter(request.filter_name for request in one[:100])
        second = Counter(request.filter_name for request in one[100:])
        ranked = [spec.name for spec in specs]
        random.Random(41).shuffle(ranked)
        phase_two_hot = set(ranked[4:8])
        self.assertTrue(phase_two_hot.isdisjoint(first))
        self.assertTrue(phase_two_hot <= set(second))
        self.assertNotEqual(first.most_common(1), second.most_common(1))
        self.assertEqual(one[99].phase, "steady_hot")
        self.assertEqual(one[100].phase, "phase_shift_hot")

    def test_formal_trace_excludes_selection_and_confirmation_prefix(self) -> None:
        specs = filters()
        exact = truth(specs, query_count=runner.FORMAL_TRUTH_QUERY_COUNT)

        trace = runner.build_trace(specs, exact)

        self.assertEqual(
            {request.query_no for request in trace},
            set(range(runner.FORMAL_MEASUREMENT_QUERY_OFFSET, runner.FORMAL_TRUTH_QUERY_COUNT)),
        )
        self.assertEqual(runner.formal_trace_contract_errors(trace), [])

    def test_adaptive_cache_gate_detects_preexisting_and_resets_empty(self) -> None:
        session = FakeSession([
            {"entries": 1, "resident_entries": 1, "resident_bytes": 8},
            {"entries": 0, "resident_entries": 0, "resident_bytes": 0},
        ])
        empty, evidence = runner.adaptive_cache_empty_gate(session)
        self.assertTrue(empty)
        self.assertFalse(evidence["before_reset_empty"])
        self.assertTrue(evidence["after_reset_empty"])
        self.assertEqual(evidence["after_reset"]["entries"], 0)
        self.assertIn("SELECT vector_hnsw_metadata_cache_reset()", session.calls)

    def test_fragment_store_reset_is_targeted_and_epoch_proven(self) -> None:
        before = {"heap_oid": 41, "count": 2, "epoch": 7, "relfilenode": 91,
                  "epoch_proof": {"valid": True, "rows_checked": 2}}
        after = {"heap_oid": 41, "count": 0, "epoch": 7, "relfilenode": 91,
                 "epoch_proof": {"valid": True, "rows_checked": 0, "epoch": 7}}
        proof = runner.validate_fragment_store_reset(before, 2, after)
        self.assertTrue(proof["valid"])
        self.assertEqual(proof["deleted"], 2)
        self.assertEqual(proof["heap_oid"], 41)
        self.assertEqual(proof["epoch_proof"]["epoch"], 7)
        with self.assertRaisesRegex(runner.BenchmarkContractError, "persistent fragment store"):
            runner.validate_fragment_store_reset(before, 1, after)
        with self.assertRaisesRegex(runner.BenchmarkContractError, "epoch proof"):
            runner.validate_fragment_store_reset(before, 2, {**after, "epoch_proof": {"valid": False, "rows_checked": 0}})

    def test_fragment_store_audit_and_clear_use_target_heap_not_global_delete(self) -> None:
        row = json.dumps({"heap_oid": 41, "build_epoch": 7, "relfilenode": 91})
        session = FragmentStoreSession(
            [("pgvector_hnsw_fragment_store", 41, 91, 7, True), ("pgvector_hnsw_fragment_store", 41, 91, 7, True)],
            [[(row,), (row,)], []],
            2,
        )
        proof = runner.clear_fragment_store(session, "public.reviews")
        self.assertEqual(proof["prebuilt_fragments"], 0)
        delete_sql = next(sql for sql, _ in session.calls if "DELETE FROM" in sql)
        self.assertIn("WHERE heap_oid = %s::regclass::oid", delete_sql)
        self.assertIn("RETURNING heap_oid", delete_sql)
        self.assertEqual(proof["heap_oid"], 41)

    def test_fragment_store_clear_is_scoped_to_one_mode_namespace(self) -> None:
        row = json.dumps({"heap_oid": 41, "build_epoch": 7, "relfilenode": 91})
        session = FragmentStoreSession(
            [("pgvector_hnsw_fragment_store", 41, 91, 7, True)] * 2,
            [[(row,)], []],
            1,
        )
        proof = runner.clear_fragment_store(session, "public.reviews", "run_adaptive")
        self.assertTrue(proof["valid"])
        self.assertEqual(proof["fragment_store_namespace"], "run_adaptive")
        scoped = [
            (sql, params) for sql, params in session.calls
            if "row_to_json" in sql or "DELETE FROM" in sql
        ]
        self.assertTrue(all("chr(31)" in sql for sql, _ in scoped))
        self.assertTrue(all("left(" in sql and "LIKE" not in sql for sql, _ in scoped))
        self.assertTrue(all(params == ("public.reviews", "run_adaptive", "run_adaptive") for _, params in scoped))

    def test_lifecycle_counter_regression_fails_closed(self) -> None:
        with self.assertRaisesRegex(runner.BenchmarkContractError, "non-monotonic lifecycle counter"):
            runner._counter_delta({"adaptive_probes": 2}, {"adaptive_probes": 1}, "adaptive_probes")

    def test_old_checkpoint_schema_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "old_checkpoint.json"
            path.write_text(json.dumps({"checkpoint_schema_version": 2}), encoding="utf-8")
            with self.assertRaisesRegex(runner.BenchmarkContractError, "checkpoint schema"):
                runner.load_checkpoint(path, "unused")

    def test_lifecycle_classifies_creation_reuse_eviction_and_reason(self) -> None:
        created = runner.lifecycle_classification({"entries": 0, "evictions": 0}, {"entries": 1, "evictions": 0},
                                                 {"fragment_builds": 1, "active": True}, admitted=True, reason="admit")
        self.assertTrue(created["fragment_created"])
        reused = runner.lifecycle_classification({"entries": 1, "evictions": 0}, {"entries": 1, "evictions": 1},
                                                {"fragment_store_hits": 1, "active": True}, admitted=True, reason="reuse")
        self.assertTrue(reused["fragment_reused"])
        self.assertTrue(reused["fragment_evicted"])
        self.assertEqual(reused["admission_reason"], "reuse")

    def test_checkpoint_is_atomic_only_for_complete_cross_mode_paired_windows(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_checkpoint.json"
            rows = {mode: [{"window": 0, "request_no": n} for n in range(3)] for mode in runner.MODES}
            runner.write_checkpoint(path, "same", rows, [0], 3)
            restored = runner.load_checkpoint(path, "same")
            self.assertEqual(restored["completed_paired_windows"], [0])
            self.assertEqual(restored["resume_contract"]["cross_process_resume"], "forbidden")
            with self.assertRaisesRegex(runner.BenchmarkContractError, "run-spec"):
                runner.load_checkpoint(path, "different")
            checkpoint_before = path.read_text(encoding="utf-8")
            partial = {mode: list(block) for mode, block in rows.items()}
            partial["adaptive"].pop()
            with self.assertRaisesRegex(runner.BenchmarkContractError, "partial paired window"):
                runner.write_checkpoint(path, "same", partial, [0], 3)
            self.assertEqual(path.read_text(encoding="utf-8"), checkpoint_before)
            manifest = json.loads(checkpoint_before)
            shard_path = runner.checkpoint_shard_directory(path) / manifest["shards"]["0"]["path"]
            tampered_shard = json.loads(shard_path.read_text(encoding="utf-8"))
            tampered_shard["rows_by_mode"]["stock"][0]["request_no"] = 99
            shard_path.write_text(json.dumps(tampered_shard), encoding="utf-8")
            with self.assertRaisesRegex(runner.BenchmarkContractError, "SHA mismatch"):
                runner.load_checkpoint(path, "same")

    def test_checkpoint_appends_one_new_shard_without_rewriting_old_windows(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_checkpoint.json"
            rows = {mode: [] for mode in runner.MODES}
            for window in (0, 1):
                for mode in runner.MODES:
                    rows[mode].extend(
                        {"window": window, "request_no": window * 2 + offset}
                        for offset in range(2)
                    )
                with mock.patch.object(runner, "atomic_json", wraps=runner.atomic_json) as atomic:
                    runner.write_checkpoint(path, "same", rows, list(range(window + 1)), 2)
                paths = [call.args[0] for call in atomic.call_args_list]
                self.assertEqual(len(paths), 2)
                self.assertIn(runner.checkpoint_shard_directory(path) / f"window_{window:06d}.json", paths)
                self.assertIn(path, paths)
                if window == 0:
                    first_shard_bytes = (
                        runner.checkpoint_shard_directory(path) / "window_000000.json"
                    ).read_bytes()
                else:
                    self.assertEqual(
                        first_shard_bytes,
                        (runner.checkpoint_shard_directory(path) / "window_000000.json").read_bytes(),
                    )
            restored = runner.load_checkpoint(path, "same")
            self.assertEqual(restored["completed_paired_windows"], [0, 1])
            self.assertEqual(len(restored["rows_by_mode"]["stock"]), 4)

    def test_checkpoint_manifest_tamper_fails_closed_during_load(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_checkpoint.json"
            rows = {mode: [{"window": 0, "request_no": n} for n in range(2)] for mode in runner.MODES}
            runner.write_checkpoint(path, "same", rows, [0], 2)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["shards"]["0"]["row_counts"]["stock"] = 1
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(runner.BenchmarkContractError, "mode row count mismatch"):
                runner.load_checkpoint(path, "same")

    def test_checkpoint_cleanup_removes_manifest_and_shard_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_checkpoint.json"
            rows = {mode: [{"window": 0, "request_no": n} for n in range(2)] for mode in runner.MODES}
            runner.write_checkpoint(path, "same", rows, [0], 2)
            shard_directory = runner.checkpoint_shard_directory(path)
            self.assertTrue(path.exists())
            self.assertTrue(shard_directory.exists())
            runner.cleanup_checkpoint(path)
            self.assertFalse(path.exists())
            self.assertFalse(shard_directory.exists())
            self.assertFalse(runner.checkpoint_exists(path))

    def test_open_mode_backends_uses_three_distinct_persistent_connections(self) -> None:
        psycopg = mock.Mock()
        connections = [mock.Mock(name=f"connection_{mode}") for mode in runner.MODES]
        sessions = [mock.Mock(name=f"session_{mode}") for mode in runner.MODES]
        for pid, session in enumerate(sessions, start=101):
            session.one.return_value = pid
        psycopg.connect.side_effect = connections
        with mock.patch.object(runner, "CursorSession", side_effect=sessions), \
             mock.patch.object(runner, "database_provenance", return_value=database_identity()):
            backends = runner.open_mode_backends(psycopg, "postgresql://test", table="reviews", index="reviews_idx")
        self.assertEqual(psycopg.connect.call_count, len(runner.MODES))
        self.assertEqual([backends[mode].session for mode in runner.MODES], sessions)
        self.assertEqual([backends[mode].backend_pid for mode in runner.MODES], [101, 102, 103])
        self.assertEqual(len({id(backends[mode].session) for mode in runner.MODES}), len(runner.MODES))
        runner.close_mode_backends(backends)
        self.assertTrue(all(connection.close.called for connection in connections))

    def test_database_provenance_reads_loaded_runtime_and_server_binary_sha(self) -> None:
        session = mock.Mock()
        session.row.side_effect = [
            (RUNTIME_BUILD_ID, VECTOR_SO_PATH, VECTOR_SO_SHA256),
            (
                "17.5", "0.8.0", 101, 201, 102, 202,
                "CREATE INDEX reviews_idx ON public.reviews USING hnsw (embedding)",
                "embedding_valid",
            ),
        ]
        observed = runner.database_provenance(session, "public.reviews", "public.reviews_idx")
        runtime_sql = session.execute.call_args_list[0].args[0]
        self.assertIn("vector_sqlens_build_id()", runtime_sql)
        self.assertIn("pg_config WHERE name = 'PKGLIBDIR'", runtime_sql)
        self.assertIn("sha256(pg_read_binary_file(path))", runtime_sql)
        self.assertEqual(observed["loaded_vector_sqlens_build_id"], RUNTIME_BUILD_ID)
        self.assertEqual(observed["loaded_vector_so_path"], VECTOR_SO_PATH)
        self.assertEqual(observed["loaded_vector_so_sha256"], VECTOR_SO_SHA256)
        self.assertTrue(runner.valid_sha256(observed["database_index_fingerprint"]))
        self.assertNotIn("database_build_id", observed)

    def test_database_provenance_fails_closed_on_invalid_server_runtime_identity(self) -> None:
        session = mock.Mock()
        session.row.return_value = (RUNTIME_BUILD_ID, "relative/vector.so", "not-a-sha")
        with self.assertRaisesRegex(runner.BenchmarkContractError, "runtime build ID/vector.so"):
            runner.database_provenance(session, "public.reviews", "public.reviews_idx")
        self.assertEqual(session.execute.call_count, 1)

    def test_validate_database_contract_requires_three_consistent_hnsw_backends(self) -> None:
        args = runner.create_argument_parser().parse_args([])
        database = database_contract_identity()
        backends = {
            mode: runner.ModeBackend(
                mode, mock.Mock(), mock.Mock(), pid, dict(database),
            )
            for mode, pid in zip(runner.MODES, (101, 102, 103))
        }
        observed = runner.validate_database_contract(backends, args, source_identity())
        self.assertTrue(observed["three_independent_backend_identities"])
        self.assertTrue(observed["partial_index_predicate_matches_candidate_universe"])
        self.assertEqual(set(observed["backend_sessions"]), set(runner.MODES))
        bad = {mode: value for mode, value in backends.items()}
        bad["adaptive"] = runner.ModeBackend(
            "adaptive", mock.Mock(), mock.Mock(), 104,
            {**database, "loaded_vector_so_sha256": "e" * 64},
        )
        with self.assertRaisesRegex(runner.BenchmarkContractError, "same loaded SQLens runtime"):
            runner.validate_database_contract(bad, args, source_identity())

    def test_preflight_returns_ready_json_and_does_not_run_or_mutate(self) -> None:
        args = runner.create_argument_parser().parse_args(["--preflight"])
        specs = filters()
        exact = truth(specs, query_count=runner.FORMAL_TRUTH_QUERY_COUNT)
        database = database_contract_identity()
        backends = {
            mode: runner.ModeBackend(
                mode, mock.Mock(), mock.Mock(), pid, dict(database),
            )
            for mode, pid in zip(runner.MODES, (201, 202, 203))
        }
        with mock.patch.object(runner, "load_filters", return_value=specs), \
             mock.patch.object(runner, "load_truth", return_value=exact), \
             mock.patch.object(runner, "source_provenance", return_value=source_identity()), \
             mock.patch.object(runner, "open_mode_backends", return_value=backends), \
             mock.patch.object(runner, "validate_database_contract", return_value={
                 "database": database,
                 "backend_sessions": {mode: {"backend_pid": pid} for mode, pid in zip(runner.MODES, (201, 202, 203))},
                 "three_independent_backend_identities": True,
             }), \
             mock.patch.object(runner, "configure", side_effect=AssertionError("configure called")), \
             mock.patch.object(runner, "reset_guidance", side_effect=AssertionError("reset called")), \
             mock.patch.object(runner, "clear_fragment_store", side_effect=AssertionError("store mutated")), \
             mock.patch.object(runner, "run_search", side_effect=AssertionError("timed search called")):
            payload = runner.preflight_experiment(args, psycopg_module=mock.Mock(), conninfo="postgresql://test")
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["errors"], [])
        self.assertTrue(payload["database_connected"])
        self.assertFalse(payload["timed_requests_executed"])
        self.assertFalse(payload["cache_or_fragment_store_modified"])
        self.assertFalse(payload["files_written"])
        self.assertTrue(payload["checks"]["trace_contract"]["ready"])
        self.assertTrue(payload["checks"]["source_provenance"]["ready"])
        self.assertTrue(payload["checks"]["database"]["ready"])
        self.assertTrue(all(backend.connection.close.called for backend in backends.values()))

    def test_preflight_fails_closed_with_structured_errors_and_no_output_write(self) -> None:
        args = runner.create_argument_parser().parse_args(["--preflight"])
        missing = Path("/definitely/missing-q10k-truth.csv")
        args.truth = missing
        with mock.patch.object(runner, "load_filters", return_value=filters()), \
             mock.patch.object(runner, "open_mode_backends", side_effect=AssertionError("must not connect after invalid inputs")):
            payload = runner.preflight_experiment(args, psycopg_module=mock.Mock(), conninfo="postgresql://test")
        self.assertFalse(payload["ready"])
        self.assertTrue(payload["errors"])
        self.assertEqual(payload["errors"][0]["check"], "exact_truth")
        self.assertFalse(payload["files_written"])
        self.assertFalse(payload["timed_requests_executed"])

    def test_main_preflight_prints_json_and_returns_readiness_status(self) -> None:
        payload = {"preflight": True, "ready": True, "errors": []}
        with mock.patch.object(runner, "preflight_experiment", return_value=payload) as preflight, \
             mock.patch("builtins.print") as output:
            self.assertEqual(runner.main(["--preflight"]), 0)
        preflight.assert_called_once()
        output.assert_called_once_with(json.dumps(payload, sort_keys=True))

    def test_profile_reported_build_id_remains_diagnostic_and_does_not_use_epochs(self) -> None:
        self.assertEqual(
            runner.reported_profile_build_id(({}, {"build_id": "profile-build-7"})),
            "profile-build-7",
        )
        self.assertEqual(
            runner.reported_profile_build_id(({"guide_generation": 12, "fragment_epoch": 3},)),
            "unreported",
        )

    def test_source_tree_provenance_hashes_each_c_and_h_file_and_aggregate(self) -> None:
        with TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "src"
            source_dir.mkdir()
            (source_dir / "vector.c").write_text("one\n", encoding="utf-8")
            (source_dir / "hnsw.h").write_text(
                f'#define SQLENS_BUILD_ID "{RUNTIME_BUILD_ID}"\ntwo\n', encoding="utf-8"
            )
            (source_dir / "README.txt").write_text("ignored\n", encoding="utf-8")
            observed = runner.sqlens_source_tree_provenance(source_dir)
            self.assertEqual(set(observed["file_sha256"]), {"vector.c", "hnsw.h"})
            self.assertEqual(observed["file_count"], 2)
            self.assertEqual(observed["declared_build_id"], RUNTIME_BUILD_ID)
            self.assertEqual(
                observed["aggregate_sha256"],
                runner.aggregate_source_file_sha256(observed["file_sha256"]),
            )
            old_aggregate = observed["aggregate_sha256"]
            (source_dir / "vector.c").write_text("changed\n", encoding="utf-8")
            self.assertNotEqual(
                runner.sqlens_source_tree_provenance(source_dir)["aggregate_sha256"],
                old_aggregate,
            )

    def test_each_paired_request_rotates_modes_and_retains_each_mode_session_cache(self) -> None:
        args = runner.create_argument_parser().parse_args(["--requests", "6", "--window-size", "3"])
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        exact = runner.TruthEntry("f", 0, 99, tuple(range(10)), 1.0, 0.0)
        trace = [runner.Request(number, "steady_hot", number // 3, "f", 0, 99, None) for number in range(6)]
        sessions = {mode: mock.Mock(name=f"{mode}_session") for mode in runner.MODES}
        backends = {
            mode: runner.ModeBackend(mode, mock.Mock(), sessions[mode], index + 1, database_identity())
            for index, mode in enumerate(runner.MODES)
        }
        calls: list[tuple[str, int, object]] = []
        cache_entries = {mode: 0 for mode in runner.MODES}

        def fake_run_request(session, unused_args, mode, request, unused_filter, unused_truth, unused_provenance, *, adaptive_started_empty, online_materializations_before, online_materializations_before_for_filter, previous_filter_name):  # type: ignore[no-untyped-def]
            self.assertIs(session, sessions[mode])
            self.assertTrue(adaptive_started_empty)
            self.assertEqual(previous_filter_name, None if request.request_no == 0 else "f")
            if mode == "adaptive":
                self.assertEqual(online_materializations_before, int(request.request_no > 0))
                self.assertEqual(online_materializations_before_for_filter, int(request.request_no > 0))
            cache_entries[mode] += 1
            calls.append((mode, request.request_no, session))
            return {"mode": mode, "request_no": request.request_no, "window": request.window,
                    "fragment_builds_delta": int(mode == "adaptive" and request.request_no == 0),
                    "materialization_observed": bool(mode == "adaptive" and request.request_no == 0)}

        lifecycle_state = {"online_materializations": 0, "online_materializations_by_filter": {}}
        with mock.patch.object(runner, "run_request", side_effect=fake_run_request):
            first = runner.run_paired_window(
                backends, args, trace, {"f": spec}, {("f", 0): exact}, request_provenance(),
                window=0, adaptive_started_empty=True, adaptive_lifecycle_state=lifecycle_state,
            )
            second = runner.run_paired_window(
                backends, args, trace, {"f": spec}, {("f", 0): exact}, request_provenance(),
                window=1, adaptive_started_empty=True, adaptive_lifecycle_state=lifecycle_state,
            )

        self.assertEqual(runner.paired_request_mode_order(0), runner.MODES)
        self.assertEqual(runner.paired_request_mode_order(1), ("adaptive", "eager_prebuilt", "stock"))
        self.assertEqual(runner.paired_request_mode_order(2), ("eager_prebuilt", "stock", "adaptive"))
        self.assertEqual(
            [mode for mode, _, _ in calls],
            [
                "stock", "adaptive", "eager_prebuilt",
                "adaptive", "eager_prebuilt", "stock",
                "eager_prebuilt", "stock", "adaptive",
            ] * 2,
        )
        self.assertEqual(cache_entries, {mode: 6 for mode in runner.MODES})
        self.assertEqual(first["stock"][0]["backend_pid"], 1)
        self.assertEqual(second["stock"][0]["paired_request_mode_rank"], 0)
        self.assertEqual(second["adaptive"][0]["paired_request_mode_rank"], 1)
        self.assertEqual(lifecycle_state["online_materializations"], 1)

    def test_paired_execution_rejects_shared_mode_session(self) -> None:
        shared_session = mock.Mock()
        backends = {
            mode: runner.ModeBackend(mode, mock.Mock(), shared_session, index + 1, database_identity())
            for index, mode in enumerate(runner.MODES)
        }
        with self.assertRaisesRegex(runner.BenchmarkContractError, "independent persistent session/cache"):
            runner.validate_independent_mode_sessions(backends)

    def test_cross_process_resume_fails_closed_before_inputs_or_database(self) -> None:
        args = runner.create_argument_parser().parse_args(["--resume"])
        with mock.patch.object(runner, "load_filters") as load_filters:
            with self.assertRaisesRegex(runner.BenchmarkContractError, "cross-process --resume is disabled"):
                runner.execute_experiment(args)
        load_filters.assert_not_called()
        contract = runner.checkpoint_resume_contract()
        self.assertEqual(contract["cross_process_resume"], "forbidden")
        self.assertEqual(contract["cache_lifecycle_fingerprints"], "audit_only_not_replayable")

    def test_break_even_and_percentiles_follow_formal_rank_rule(self) -> None:
        adaptive = [{"request_no": 0, "e2e_ms": 12.0}, {"request_no": 1, "e2e_ms": 7.0}]
        stock = {0: {"e2e_ms": 10.0}, 1: {"e2e_ms": 10.0}}
        self.assertEqual(runner.break_even_request(adaptive, stock), 1)
        noisy = [{"request_no": 0, "e2e_ms": 9.0}, {"request_no": 1, "e2e_ms": 12.0},
                 {"request_no": 2, "e2e_ms": 8.0}]
        noisy_stock = {request_no: {"e2e_ms": 10.0} for request_no in range(3)}
        self.assertEqual(runner.break_even_request(noisy, noisy_stock), 2)
        self.assertEqual(runner.percentile([1, 2, 3, 4, 5], .95), 5)
        self.assertEqual(runner.percentile([1, 2, 3, 4, 5], .99), 5)
        summary = runner.summary_for_window([
            {"e2e_ms": value, "query_ms": value, "recall_at_10": 1.0, "fragment_reused": False,
             "guidance_checks": 1, "guidance_skips": 0, "cache_resident_bytes_after": 1, "error": ""}
            for value in (1, 2, 3, 4, 5)
        ], bootstrap_samples=20, bootstrap_seed=7)
        self.assertEqual(summary["e2e_p95_ms"], 5.0)
        self.assertEqual(summary["e2e_p99_ms"], 5.0)

    def test_summary_proves_lifecycle_deltas_and_percentile_contract(self) -> None:
        rows = [
            {"e2e_ms": 4.0, "query_ms": 3.0, "recall_at_10": 1.0, "fragment_reused": False,
             "fragment_store_hit_delta": 0, "probe_observed": True, "materialization_observed": True,
             "reuse_observed": False, "refine_observed": False, "evict_observed": False,
             "hidden_prebuilt_fragment_reused": False, "lifecycle_path": "probe->materialize",
             "guidance_checks": 1, "guidance_skips": 0, "cache_resident_bytes_after": 2, "error": ""},
            {"e2e_ms": 2.0, "query_ms": 1.0, "recall_at_10": 1.0, "fragment_reused": True,
             "fragment_store_hit_delta": 1, "probe_observed": False, "materialization_observed": False,
             "reuse_observed": True, "refine_observed": True, "evict_observed": True,
             "hidden_prebuilt_fragment_reused": False, "lifecycle_path": "reuse->refine->evict",
             "guidance_checks": 1, "guidance_skips": 1, "cache_resident_bytes_after": 1, "error": ""},
        ]
        summary = runner.summary_for_window(rows, bootstrap_samples=20, bootstrap_seed=7)
        self.assertEqual(summary["e2e_p50_ms"], 2.0)
        self.assertEqual(summary["e2e_p95_ms"], 4.0)
        self.assertEqual(summary["e2e_p99_ms"], 4.0)
        self.assertEqual(summary["fragment_store_hit_delta"], 1)
        self.assertEqual(summary["lifecycle_event_counts"]["materialize"], 1)
        self.assertEqual(summary["hidden_prebuilt_reuse_count"], 0)

    def test_run_spec_names_q10k_unique_trace_and_rejects_cracking_claim(self) -> None:
        args = runner.create_argument_parser().parse_args([])
        trace = [
            runner.Request(number, "steady_hot", number // 100, "f", number, number, None)
            for number in range(runner.FORMAL_REQUESTS)
        ]
        spec = runner.make_run_spec(args, {}, {}, trace)
        self.assertEqual(spec["workload_manifest_name"], runner.FORMAL_WORKLOAD_MANIFEST_NAME)
        self.assertEqual(spec["unique_query_vectors"], runner.FORMAL_REQUESTS)
        self.assertTrue(spec["one_request_per_query_vector"])
        self.assertFalse(spec["database_cracking"])
        self.assertIn("deterministic 10,000-request PostgreSQL online replay", spec["trace_contract"])
        self.assertIn("every vector is used exactly once", spec["trace_contract"])
        self.assertEqual(spec["trace_kind"], runner.TRACE_KIND)
        self.assertFalse(spec["production_trace"])
        self.assertTrue(spec["postgresql_online_execution"])
        self.assertEqual(spec["fixed_exact_truth_query_count"], 10_200)
        self.assertEqual(spec["candidate_validity_predicate"], "embedding_valid")
        self.assertEqual(
            spec["search_configuration"],
            {
                "k": 10,
                "ef_search": 10_000,
                "iterative_scan": "strict_order",
                "max_scan_tuples": 5_000_000,
                "scan_mem_multiplier": 32.0,
                "force_hnsw": True,
                "statement_timeout_ms": 120_000,
            },
        )

    def test_formal_defaults_bind_unique_embedding_truth_and_partial_index(self) -> None:
        args = runner.create_argument_parser().parse_args([])
        self.assertIn("q10200_unique_embeddings_formal", args.truth.name)
        self.assertIn("q10200_unique_embeddings_formal", args.truth_manifest.name)
        self.assertIn("query_cohort_q10200", args.query_cohort.name)
        self.assertEqual(args.truth_query_count, 10_200)
        self.assertIn("valid_embeddings_filters", args.filters_csv.name)
        self.assertEqual(args.candidate_validity_predicate, "embedding_valid")
        self.assertEqual(
            args.index, "amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx"
        )
        self.assertTrue(runner.formal_protocol(args))

    def test_formal_protocol_rejects_quality_and_admission_overrides(self) -> None:
        for override in (
            ["--k", "1"],
            ["--ef-search", "9999"],
            ["--max-scan-tuples", "500000"],
            ["--scan-mem-multiplier", "8"],
            ["--d3-min-benefit-per-byte", "0"],
            ["--absolute-recall-target", "0.1"],
        ):
            with self.subTest(override=override):
                self.assertFalse(
                    runner.formal_protocol(
                        runner.create_argument_parser().parse_args(override)
                    )
                )

    def test_tie_aware_quality_accepts_boundary_substitution_and_audits_strict_closer(self) -> None:
        exact = runner.TruthEntry(
            "f", 0, 99, (1, 2), 1.0, 1e-6, 1, True, "final"
        )
        accepted = runner.tie_aware_result_quality(
            [1, 3], [0.25, 1.0000005], exact, k=2
        )
        self.assertEqual(accepted["recall"], 1.0)
        self.assertTrue(accepted["all_strict_closer_returned"])
        rejected = runner.tie_aware_result_quality(
            [3, 4], [1.0, 1.0], exact, k=2
        )
        self.assertEqual(rejected["recall"], 0.5)
        self.assertFalse(rejected["all_strict_closer_returned"])
        self.assertEqual(rejected["strict_closer_missing_ids"], [1])

    def test_query_cohort_truth_and_both_manifests_are_hash_bound(self) -> None:
        specs = filters()
        exact = truth(specs, query_count=2)
        exact = {
            key: runner.TruthEntry(
                entry.filter_name, entry.query_no, entry.query_id, entry.ids,
                entry.kth_distance_sq, entry.tie_tolerance, entry.strict_closer_count,
                entry.boundary_tied, "calibration" if entry.query_no == 0 else "final",
            )
            for key, entry in exact.items()
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort = root / "cohort.csv"
            cohort.write_text(
                "query_no,query_id,query_split,candidate_validity_predicate,query_validity_predicate\n"
                "0,0,calibration,embedding_valid,embedding_valid\n"
                "1,1,final,embedding_valid,embedding_valid\n",
                encoding="utf-8",
            )
            cohort_manifest = root / "cohort_manifest.json"
            cohort_manifest.write_text(json.dumps({
                "schema_version": 1,
                "artifact_valid": True,
                "method": "deterministic_unique_projection_fingerprint_cohort_v1",
                "candidate_validity_predicate": "embedding_valid",
                "query_validity_predicate": "embedding_valid",
                "selection": {
                    "disjoint": True,
                    "query_ids_sha256": runner.ordered_query_ids_sha256([0, 1]),
                    "calibration": {"queries": 1},
                    "final": {"queries": 1},
                },
                "uniqueness_contract": {
                    "all_rows_fingerprinted": True,
                    "duplicate_admission_false_negative_only": True,
                    "hashes": 2,
                },
                "eligible_query_population": {"singleton_fingerprint_rows": 2},
                "outputs": {"cohort_csv": {
                    "sha256": runner.sha256_file(cohort), "rows": 2,
                }},
            }), encoding="utf-8")
            exact_manifest = {"query_source": {
                "kind": "external_unique_vector_query_cohort",
                "cohort_csv": {"sha256": runner.sha256_file(cohort)},
                "manifest": {"sha256": runner.sha256_file(cohort_manifest)},
            }}
            args = runner.create_argument_parser().parse_args([
                "--query-cohort", str(cohort),
                "--query-cohort-manifest", str(cohort_manifest),
                "--truth-query-count", "2",
                "--requests", "2",
                "--window-size", "1",
            ])
            proof = runner.load_query_cohort_provenance(args, exact_manifest, exact, specs)
            self.assertTrue(proof["exact_truth_binding_verified"])
            self.assertEqual(proof["unique_query_ids"], 2)

            original_manifest = cohort_manifest.read_text(encoding="utf-8")
            invalid_manifest = json.loads(original_manifest)
            invalid_manifest["uniqueness_contract"]["all_rows_fingerprinted"] = False
            cohort_manifest.write_text(json.dumps(invalid_manifest), encoding="utf-8")
            with self.assertRaisesRegex(runner.BenchmarkContractError, "unique-vector"):
                runner.load_query_cohort_provenance(args, exact_manifest, exact, specs)
            cohort_manifest.write_text(original_manifest, encoding="utf-8")

            corrupted = dict(exact)
            corrupted[(specs[0].name, 1)] = runner.TruthEntry(
                specs[0].name, 1, 99, tuple(range(10)), 1.0, 0.0
            )
            with self.assertRaisesRegex(runner.BenchmarkContractError, "filter-dependent"):
                runner.load_query_cohort_provenance(
                    args, exact_manifest, corrupted, specs
                )

    def test_truth_loader_rejects_silent_query_grid_slicing(self) -> None:
        specs = filters()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "truth.csv"
            path.write_text(
                "filter_name,query_no,query_id,exact_filtered_topk_ids,kth_distance_sq,tie_tolerance,"
                "strict_closer_count,boundary_tied,self_excluded,query_split\n"
                f"{specs[0].name},2,2,0;1;2;3;4;5;6;7;8;9,1.0,0.0,9,false,true,final\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(runner.BenchmarkContractError, "outside requested q2"):
                runner.load_truth(path, specs, expected_query_count=2)

    def test_post_replay_correctness_audit_rechecks_predicate_without_timing_it(self) -> None:
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        rows = {
            mode: [{
                "filter_name": "f", "query_no": 0, "query_id": 99,
                "returned": 2, "returned_ids": [1, 2],
                "returned_distances_sq": [0.25, 1.0], "recall_at_10": 1.0,
                "error": "",
            }]
            for mode in runner.MODES
        }
        session = mock.Mock()
        session.all.return_value = [(1,), (2,)]
        audit = runner.audit_result_correctness(
            session, rows, filters_by_name={"f": spec}, table="reviews",
            candidate_validity_predicate="embedding_valid", k=2,
            truth={
                ("f", 0): runner.TruthEntry(
                    "f", 0, 99, (1, 2), 1.0, 0.0, 1, False, "final"
                )
            },
        )
        sql = session.execute.call_args.args[0]
        self.assertIn("id = ANY(%s::bigint[])", sql)
        self.assertIn("rating = 1", sql)
        self.assertFalse(audit["included_in_online_latency"])
        self.assertTrue(audit["all_rows_correct"])
        self.assertTrue(all(rows[mode][0]["result_correct"] for mode in runner.MODES))

    def test_post_replay_semantic_audit_does_not_require_exact_ann_topk(self) -> None:
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        rows = {
            mode: [{
                "filter_name": "f", "query_no": 0, "query_id": 99,
                "returned": 2, "returned_ids": [3, 4],
                "returned_distances_sq": [1.5, 2.0], "recall_at_10": 0.0,
                "error": "",
            }]
            for mode in runner.MODES
        }
        session = mock.Mock()
        session.all.return_value = [(3,), (4,)]
        audit = runner.audit_result_correctness(
            session, rows, filters_by_name={"f": spec}, table="reviews",
            candidate_validity_predicate="embedding_valid", k=2,
            truth={
                ("f", 0): runner.TruthEntry(
                    "f", 0, 99, (1, 2), 1.0, 0.0, 1, False, "final"
                )
            },
        )
        self.assertTrue(audit["all_rows_correct"])
        for mode in runner.MODES:
            self.assertTrue(rows[mode][0]["result_semantically_correct"])
            self.assertFalse(rows[mode][0]["result_exact_topk"])

    def test_validation_rejects_cross_mode_result_mismatch(self) -> None:
        trace = [runner.Request(0, "steady_hot", 0, "f", 0, 100, None)]
        base = {
            "request_no": 0, "filter_name": "f", "e2e_ms": 10.0,
            "recall_at_10": 1.0, "result_correct": True, "returned": 2,
            "returned_ids": [1, 2], "returned_distances_sq": [0.1, 0.2],
            "hnsw_scan_profile_required": True, "hnsw_scan_profile_valid": True,
            **row_identity(), "error": "",
        }
        rows = {
            "stock": [{**base, "fragment_store_namespace": "run_stock"}],
            "adaptive": [{
                **base, "returned_ids": [1, 3],
                "fragment_store_namespace": "run_adaptive",
                "adaptive_cache_started_empty": True,
                "persistent_fragment_reset_proof": {"valid": True, "prebuilt_fragments": 0},
                "online_arm": True,
            }],
            "eager_prebuilt": [{
                **base, "fragment_store_namespace": "run_eager",
                "explicit_eager_control": True,
            }],
        }
        errors = runner.validate_artifact(
            rows, trace, recall_delta=.01, provenance=database_identity(),
            source=source_identity(), formal=False,
        )
        self.assertIn("result_equivalence_failure:adaptive", errors)

    def test_formal_gate_requires_per_filter_quality_lifecycle_and_amortization(self) -> None:
        trace = [runner.Request(number, "steady_hot", 0, "f", number, number + 100, None) for number in range(3)]
        base = {
            "filter_name": "f", **row_identity(),
            "recall_at_10": 1.0, "result_correct": True, "returned": 10,
            "returned_ids": list(range(10)), "error": "", "e2e_ms": 10.0,
            "hnsw_scan_profile_required": True, "hnsw_scan_profile_valid": True,
            "truth_query_split": "final",
        }
        stock = [
            {**base, "request_no": number, "fragment_store_namespace": "run_stock"}
            for number in range(3)
        ]
        adaptive = [
            {**base, "request_no": 0, "e2e_ms": 12.0, "adaptive_cache_started_empty": True,
             "fragment_store_namespace": "run_adaptive",
             "persistent_fragment_reset_proof": {"valid": True, "prebuilt_fragments": 0},
             "online_arm": True, "probe_observed": True, "materialization_observed": True,
             "admission_observed": True,
             "activation_materialization_observed": True, "query_materialization_observed": False,
             "reuse_observed": False, "online_materializations_before": 0,
             "online_materializations_before_for_filter": 0, "materialization_ms": 2.0,
             "adaptive_fragment_build_ms_delta": 2.0},
            {**base, "request_no": 1, "e2e_ms": 5.0, "adaptive_cache_started_empty": True,
             "fragment_store_namespace": "run_adaptive",
             "online_arm": True, "probe_observed": False, "materialization_observed": False,
             "reuse_observed": True, "direct_reuse_signal": True, "online_materializations_before": 1,
             "online_materializations_before_for_filter": 1, "materialization_ms": 0.0,
             "adaptive_fast_reactivation_hits_delta": 1},
            {**base, "request_no": 2, "e2e_ms": 5.0, "adaptive_cache_started_empty": True,
             "fragment_store_namespace": "run_adaptive",
             "online_arm": True, "probe_observed": False, "materialization_observed": False,
             "reuse_observed": True, "direct_reuse_signal": True, "online_materializations_before": 1,
             "online_materializations_before_for_filter": 1, "materialization_ms": 0.0,
             "adaptive_fast_reactivation_hits_delta": 1},
        ]
        eager = [
            {
                **base, "request_no": number, "explicit_eager_control": True,
                "fragment_store_namespace": "run_eager",
                **({"eager_prebuild_evidence": {
                    "setup_outside_timed_requests": True, "all_filters_prebuilt": True,
                    "common_post_setup_warmup": {
                        "executed_after_eager_prebuild": True,
                        "all_mode_backends_executed": True,
                        "all_backend_pids_distinct": True,
                        "all_backend_d3_state_untouched": True,
                        "calibration_queries_per_backend": runner.FORMAL_CALIBRATION_QUERY_COUNT,
                    },
                }} if number == 0 else {}),
            }
            for number in range(3)
        ]
        rows = {"stock": stock, "adaptive": adaptive, "eager_prebuilt": eager}
        errors = runner.validate_artifact(
            rows, trace, recall_delta=.01, provenance=database_identity(), source=source_identity(),
            absolute_recall_target=.90, k=10, formal=True,
            persisted_reuse_evidence=persisted_reuse_evidence(),
        )
        self.assertEqual(errors, [])
        lifecycle = runner.adaptive_lifecycle_summary(adaptive)
        self.assertTrue(lifecycle["sequence_complete"])
        amortization = runner.amortization_summary(adaptive, {row["request_no"]: row for row in stock})
        self.assertEqual(amortization["materialization_cost_ms"], 2.0)
        self.assertEqual(amortization["reuse_savings_vs_stock_ms"], 10.0)
        self.assertEqual(amortization["cumulative_break_even_request"], 1)

    def test_formal_gate_fails_closed_on_filter_recall_correctness_or_missing_reuse(self) -> None:
        trace = [runner.Request(0, "steady_hot", 0, "f", 0, 100, None)]
        base = {"request_no": 0, "filter_name": "f", "e2e_ms": 10.0, "recall_at_10": .8,
                "result_correct": False, **row_identity(), "error": ""}
        rows = {
            "stock": [base],
            "adaptive": [{**base, "adaptive_cache_started_empty": True,
                          "persistent_fragment_reset_proof": {"valid": True, "prebuilt_fragments": 0},
                          "online_arm": True, "probe_observed": True, "materialization_observed": True,
                          "reuse_observed": False, "materialization_ms": 0.0}],
            "eager_prebuilt": [{**base, "explicit_eager_control": True}],
        }
        errors = runner.validate_artifact(
            rows, trace, recall_delta=.01, provenance=database_identity(), source=source_identity(),
            absolute_recall_target=.90, formal=True,
        )
        self.assertIn("per_filter_recall_target_not_met:adaptive:f", errors)
        self.assertIn("per_filter_correctness_failure:adaptive:f", errors)
        self.assertIn("adaptive_lifecycle_incomplete:probe_materialize_reuse", errors)
        self.assertIn("adaptive_materialization_cost_missing", errors)
        self.assertIn("adaptive_cumulative_break_even_not_reached", errors)

    def test_invalidation_catches_missing_rows_recall_planner_and_build_mismatch(self) -> None:
        trace = [runner.Request(0, "steady_hot", 0, "f", 0, 0, None)]
        base = {"request_no": 0, "e2e_ms": 1.0, "recall_at_10": 1.0, **row_identity(), "error": ""}
        rows = {"stock": [base], "adaptive": [{**base, "recall_at_10": .5, "activation_attempted": True,
                                                  "planner_proof_required": True,
                                                  "planner_proof_verified": False, "adaptive_cache_started_empty": False,
                                                  "database_index_fingerprint": "e" * 64}], "eager_prebuilt": []}
        errors = runner.validate_artifact(
            rows, trace, recall_delta=.01, provenance=database_identity(), source=source_identity()
        )
        self.assertIn("runtime_identity_mismatch:adaptive", errors)
        self.assertIn("planner_proof_failure:adaptive", errors)
        self.assertIn("recall_regression:adaptive", errors)
        self.assertIn("preexisting_adaptive_cache", errors)
        self.assertIn("missing_or_duplicate_windows:eager_prebuilt", errors)

    def test_formal_gate_checks_runtime_and_source_binding_for_every_mode(self) -> None:
        trace = [runner.Request(0, "steady_hot", 0, "f", 0, 100, None)]
        base = {
            "request_no": 0, "filter_name": "f", "e2e_ms": 1.0, "recall_at_10": 1.0,
            "result_correct": True, "returned": 10, "returned_ids": list(range(10)),
            **row_identity(), "error": "",
        }
        for mode in runner.MODES:
            with self.subTest(identity_mode=mode):
                rows = {candidate: [{**base}] for candidate in runner.MODES}
                rows["adaptive"][0].update({
                    "adaptive_cache_started_empty": True,
                    "persistent_fragment_reset_proof": {"valid": True, "prebuilt_fragments": 0},
                    "online_arm": True,
                })
                rows["eager_prebuilt"][0]["explicit_eager_control"] = True
                rows[mode][0]["runtime_build_id"] = "different-runtime"
                errors = runner.validate_artifact(
                    rows, trace, recall_delta=.01, provenance=database_identity(),
                    source=source_identity(), formal=True,
                )
                self.assertIn(f"runtime_identity_mismatch:{mode}", errors)
            with self.subTest(source_mode=mode):
                rows = {candidate: [{**base}] for candidate in runner.MODES}
                rows["adaptive"][0].update({
                    "adaptive_cache_started_empty": True,
                    "persistent_fragment_reset_proof": {"valid": True, "prebuilt_fragments": 0},
                    "online_arm": True,
                })
                rows["eager_prebuilt"][0]["explicit_eager_control"] = True
                rows[mode][0]["sqlens_source_aggregate_sha256"] = "e" * 64
                errors = runner.validate_artifact(
                    rows, trace, recall_delta=.01, provenance=database_identity(),
                    source=source_identity(), formal=True,
                )
                self.assertIn(f"source_binding_mismatch:{mode}", errors)

    def test_formal_gate_rejects_incomplete_runtime_and_self_inconsistent_source_manifest(self) -> None:
        trace = [runner.Request(0, "steady_hot", 0, "f", 0, 100, None)]
        rows = {
            mode: [{"request_no": 0, "e2e_ms": 1.0, "recall_at_10": 1.0, **row_identity(), "error": ""}]
            for mode in runner.MODES
        }
        malformed_source = source_identity()
        malformed_source["sqlens_source"]["aggregate_sha256"] = "f" * 64
        errors = runner.validate_artifact(
            rows, trace, recall_delta=.01,
            provenance={**database_identity(), "loaded_vector_so_path": "relative/vector.so"},
            source=malformed_source, formal=True,
        )
        self.assertIn("runtime_provenance_incomplete", errors)
        self.assertIn("source_provenance_invalid", errors)

    def test_formal_gate_binds_current_source_to_loaded_binary(self) -> None:
        trace = [runner.Request(0, "steady_hot", 0, "f", 0, 100, None)]
        rows = {
            mode: [{"request_no": 0, "e2e_ms": 1.0, "recall_at_10": 1.0, **row_identity(), "error": ""}]
            for mode in runner.MODES
        }
        for mutation in (
            lambda source: source["local_vector_so"].update({"sha256": "f" * 64}),
            lambda source: source["sqlens_source"].update({"declared_build_id": "other-build"}),
            lambda source: source["local_vector_so"].update({"built_after_source_tree": False}),
        ):
            with self.subTest(mutation=mutation):
                source = source_identity()
                mutation(source)
                errors = runner.validate_artifact(
                    rows, trace, recall_delta=.01, provenance=database_identity(),
                    source=source, formal=True,
                )
                self.assertIn("source_runtime_binary_binding_invalid", errors)

    def test_adaptive_request_always_enters_extension_state_machine(self) -> None:
        args = runner.create_argument_parser().parse_args([])
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        request = runner.Request(0, "steady_hot", 0, "f", 0, 99, None)
        exact = runner.TruthEntry("f", 0, 99, tuple(range(10)), 1.0, 0.0)
        session = mock.Mock()
        cache = {"entries": 0, "resident_entries": 0, "resident_bytes": 0}
        with mock.patch.object(runner, "json_profile", side_effect=[cache, cache, cache, cache, cache]), \
             mock.patch.object(runner, "activate", return_value=({"active": False, "adaptive_state": "probing"}, 2.5)) as activate, \
             mock.patch.object(runner, "run_search", return_value=(list(range(10)), [0.1] * 10, {}, "", 7.5)):
            row = runner.run_request(
                session, args, "adaptive", request, spec, exact,
                request_provenance(), adaptive_started_empty=True,
            )
        self.assertEqual(activate.call_args.args[3], "adaptive")
        self.assertTrue(row["activation_attempted"])
        self.assertFalse(row["guidance_active"])
        self.assertFalse(row["planner_proof_required"])
        self.assertEqual(row["adaptive_state"], "probing")
        self.assertEqual(row["activation_ms"], 2.5)
        self.assertEqual(row["query_ms"], 7.5)
        self.assertEqual(row["e2e_ms"], 10.0)
        self.assertEqual(row["runtime_build_id"], RUNTIME_BUILD_ID)
        self.assertEqual(row["loaded_vector_so_sha256"], VECTOR_SO_SHA256)
        self.assertEqual(row["database_index_fingerprint"], DATABASE_INDEX_FINGERPRINT)
        self.assertEqual(row["sqlens_source_aggregate_sha256"], SOURCE_AGGREGATE_SHA256)
        self.assertEqual(row["profile_reported_build_id"], "unreported")
        self.assertNotIn("profile_build_id", row)
        self.assertNotIn("database_build_id", row)

    def test_materialization_cost_is_not_recharged_from_a_stale_profile_timer(self) -> None:
        args = runner.create_argument_parser().parse_args([])
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        exact = runner.TruthEntry("f", 0, 99, tuple(range(10)), 1.0, 0.0)
        cache_before = {
            "entries": 1, "resident_entries": 1, "resident_bytes": 8,
            "adaptive_fragment_builds": 0, "adaptive_fragment_build_ms": 0.0,
        }
        cache_after = {
            **cache_before, "adaptive_fragment_builds": 1,
            "adaptive_page_builds": 1, "adaptive_fragment_build_ms": 3.25,
        }
        before_build = {"fragment_builds": 0}
        after_build = {"fragment_builds": 1, "last_cache_build_ms": 3.25, "active": True}
        session = mock.Mock()
        with mock.patch.object(
            runner, "json_profile", side_effect=[cache_before, before_build, cache_after, cache_after, after_build]
        ), mock.patch.object(
            runner, "activate", return_value=(after_build, 4.0)
        ), mock.patch.object(
            runner, "run_search",
            return_value=(list(range(10)), [0.1] * 10, {"valid": True, "planner_proof_succeeded": True}, "", 2.0),
        ):
            materialized = runner.run_request(
                session, args, "adaptive", runner.Request(0, "steady_hot", 0, "f", 0, 99, None),
                spec, exact, request_provenance(), adaptive_started_empty=True,
            )
        self.assertTrue(materialized["materialization_observed"])
        self.assertEqual(materialized["materialization_ms"], 3.25)
        self.assertEqual(materialized["payload_build_ms"], 3.25)

        with mock.patch.object(
            runner, "json_profile", side_effect=[cache_after, after_build, cache_after, cache_after, after_build]
        ), mock.patch.object(
            runner, "activate", return_value=(after_build, 1.0)
        ), mock.patch.object(
            runner, "run_search",
            return_value=(list(range(10)), [0.1] * 10, {"valid": True, "planner_proof_succeeded": True}, "", 2.0),
        ):
            reused = runner.run_request(
                session, args, "adaptive", runner.Request(1, "steady_hot", 0, "f", 0, 99, 1),
                spec, exact, request_provenance(), adaptive_started_empty=True,
                online_materializations_before=1,
            )
        self.assertFalse(reused["materialization_observed"])
        self.assertFalse(reused["reuse_observed"])
        self.assertEqual(reused["materialization_ms"], 0.0)

    def test_query_phase_materialization_is_charged_to_the_timed_query(self) -> None:
        args = runner.create_argument_parser().parse_args([])
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        exact = runner.TruthEntry("f", 0, 99, tuple(range(10)), 1.0, 0.0)
        before = {"entries": 0, "resident_entries": 0, "resident_bytes": 0,
                  "adaptive_fragment_builds": 0, "adaptive_fragment_build_ms": 0.0}
        after_activation = dict(before)
        after_query = {**before, "adaptive_fragment_builds": 1, "adaptive_bloom_builds": 1,
                       "adaptive_fragment_build_ms": 3.5}
        guidance_before = {"fragment_builds": 0}
        guidance_after = {"fragment_builds": 1, "last_cache_build_ms": 3.5, "active": True}
        with mock.patch.object(
            runner, "json_profile",
            side_effect=[before, guidance_before, after_activation, after_query, guidance_after],
        ), mock.patch.object(
            runner, "activate", return_value=(guidance_after, 4.0)
        ), mock.patch.object(
            runner, "run_search",
            return_value=(list(range(10)), [0.1] * 10, {"valid": True, "planner_proof_succeeded": True}, "", 7.0),
        ):
            row = runner.run_request(
                mock.Mock(), args, "adaptive", runner.Request(0, "steady_hot", 0, "f", 0, 99, None),
                spec, exact, request_provenance(), adaptive_started_empty=True,
            )
        self.assertTrue(row["materialization_observed"])
        self.assertFalse(row["activation_materialization_observed"])
        self.assertTrue(row["query_materialization_observed"])
        self.assertEqual(row["materialization_ms"], 3.5)
        self.assertEqual(row["e2e_ms"], 11.0)

    def test_adaptive_reuse_requires_direct_same_predicate_materialization_evidence(self) -> None:
        args = runner.create_argument_parser().parse_args([])
        spec = runner.FilterSpec("new_filter", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        exact = runner.TruthEntry("new_filter", 0, 99, tuple(range(10)), 1.0, 0.0)
        cache = {"entries": 1, "resident_entries": 1, "resident_bytes": 8, "adaptive_fragment_builds": 4}
        guidance = {"active": True, "fragment_cache_hits": 0, "fragment_store_hits": 0}
        with mock.patch.object(runner, "json_profile", side_effect=[cache, guidance, cache, cache, guidance]), \
             mock.patch.object(runner, "activate", return_value=(guidance, 1.0)), \
             mock.patch.object(runner, "run_search", return_value=(list(range(10)), [0.1] * 10, {"valid": True, "planner_proof_succeeded": True}, "", 2.0)):
            row = runner.run_request(
                mock.Mock(), args, "adaptive", runner.Request(2, "steady_hot", 0, "new_filter", 0, 99, 1),
                spec, exact, request_provenance(), adaptive_started_empty=True,
                online_materializations_before=4, online_materializations_before_for_filter=0,
                previous_filter_name="new_filter",
            )
        self.assertFalse(row["direct_reuse_signal"])
        self.assertFalse(row["reuse_observed"])
        self.assertFalse(row["fragment_reused"])

    def test_formal_trace_and_paired_schedule_checks_fail_closed(self) -> None:
        trace = [
            runner.Request(number, "steady_hot" if number < 5_000 else "phase_shift_hot", number // 100,
                           f"f{number % 14}", number, number, None)
            for number in range(runner.FORMAL_REQUESTS)
        ]
        self.assertIn("trace_hot_set_shift_not_observed", runner.formal_trace_contract_errors(trace))
        rows = {
            mode: [{
                "request_no": request.request_no, "backend_pid": index + 10,
                "backend_mode": mode, "paired_request_mode_order": list(runner.paired_request_mode_order(request.request_no)),
                "paired_request_mode_rank": runner.paired_request_mode_order(request.request_no).index(mode),
            } for request in trace]
            for index, mode in enumerate(runner.MODES)
        }
        self.assertEqual(runner.paired_execution_errors(rows, trace), [])
        rows["adaptive"][2]["paired_request_mode_rank"] = 0
        self.assertIn("paired_rank_mismatch:adaptive", runner.paired_execution_errors(rows, trace))

    def test_search_binds_active_guidance_inside_the_hybrid_query(self) -> None:
        session = mock.Mock()
        session.all.return_value = [(7, 0.5)]
        with mock.patch.object(
            runner, "json_profile", return_value={"planner_proof_succeeded": True}
        ):
            ids, distances_sq, profile, error, _ = runner.run_search(
                session,
                "reviews",
                "rating = 5",
                "embedding_valid",
                42,
                10,
                guidance_binding=("reviews_hnsw", ("sql:rating = 5",), "adaptive"),
            )
        search_call = session.execute.call_args_list[1]
        self.assertIn("vector_hnsw_guidance_bind", search_call.args[0])
        self.assertIn("rating = 5", search_call.args[0])
        self.assertEqual(
            search_call.args[1],
            (42, "reviews_hnsw", ["sql:rating = 5"], "adaptive", 42),
        )
        self.assertEqual(ids, [7])
        self.assertEqual(distances_sq, [0.25])
        self.assertTrue(profile["planner_proof_succeeded"])
        self.assertEqual(error, "")

    def test_eager_control_uses_explicit_nonadaptive_fragment_kind(self) -> None:
        args = runner.create_argument_parser().parse_args(["--eager-kind", "page"])
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        request = runner.Request(0, "steady_hot", 0, "f", 0, 99, None)
        exact = runner.TruthEntry("f", 0, 99, tuple(range(10)), 1.0, 0.0)
        session = mock.Mock()
        cache = {"entries": 1, "resident_entries": 1, "resident_bytes": 8}
        with mock.patch.object(runner, "json_profile", side_effect=[cache, cache, cache, cache, cache]), \
             mock.patch.object(runner, "activate", return_value=({"active": True}, 1.0)) as activate, \
             mock.patch.object(runner, "run_search", return_value=(list(range(10)), [0.1] * 10, {"planner_proof_succeeded": True}, "", 2.0)):
            row = runner.run_request(
                session, args, "eager_prebuilt", request, spec, exact,
                request_provenance(), adaptive_started_empty=True,
            )
        self.assertEqual(activate.call_args.args[3], "page")
        self.assertTrue(row["guidance_active"])
        self.assertTrue(row["planner_proof_required"])

    def test_runtime_guc_and_timer_contract_use_real_extension_names(self) -> None:
        args = runner.create_argument_parser().parse_args(["--d3-page-min-skip-rate", "0.25"])
        session = mock.Mock()
        with mock.patch.object(runner, "json_profile", return_value={}):
            runner.configure(session, args, "adaptive")
        statements = [call.args[0] for call in session.execute.call_args_list]
        self.assertIn("SELECT set_config('hnsw.preferred_index', %s, false)", statements)
        preferred_call = next(
            call for call in session.execute.call_args_list
            if call.args[0] == "SELECT set_config('hnsw.preferred_index', %s, false)"
        )
        self.assertEqual(preferred_call.args[1], (args.index,))
        self.assertIn("SET hnsw.d3_page_min_skip_rate = 0.25", statements)
        namespace_call = next(
            call for call in session.execute.call_args_list
            if call.args[0] == "SELECT set_config('hnsw.fragment_store_namespace', %s, false)"
        )
        self.assertEqual(namespace_call.args[1], (runner.fragment_store_namespace(args, "adaptive"),))
        self.assertFalse(any("d3_refine_skip_rate" in statement for statement in statements))

    def test_dry_run_reads_no_inputs_or_database_and_debug_override_is_labeled(self) -> None:
        missing = Path("/definitely/missing")
        with mock.patch.object(runner, "execute_experiment") as execute:
            self.assertEqual(runner.main(["--dry-run", "--filters-csv", str(missing), "--truth", str(missing),
                                          "--requests", "200", "--window-size", "20"]), 0)
        execute.assert_not_called()
        payload = runner.dry_run_payload(runner.create_argument_parser().parse_args(["--requests", "200", "--window-size", "20"]))
        self.assertTrue(payload["debug_override_labeled_non_formal"])
        self.assertEqual(payload["trace_kind"], "deterministic_non_formal_postgresql_online_replay")
        self.assertIn("200-request non-formal", payload["trace_contract"])
        self.assertEqual(payload["search_configuration"]["ef_search"], 10_000)
        self.assertEqual(payload["search_configuration"]["iterative_scan"], "strict_order")


if __name__ == "__main__":
    unittest.main()
