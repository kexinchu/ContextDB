from __future__ import annotations

from unittest import mock

import pytest
from psycopg import sql

from experiments.hybrid_vector_db.scripts import pgvector_update_correctness_stress as stress


def test_schedule_and_persistent_lanes_are_deterministic() -> None:
    mix = stress.parse_mutation_mix("predicate:4,vector:4,insert:1,delete:1")
    first = stress.build_mutation_schedule([10, 11, 12], [20, 21, 22], 20, 7, mix)
    second = stress.build_mutation_schedule([10, 11, 12], [20, 21, 22], 20, 7, mix)
    assert first == second
    assert {event.mutation for event in first} == set(stress.MUTATIONS)
    inserted = {event.target_id for event in first if event.mutation == "insert"}
    deleted = [event.target_id for event in first if event.mutation == "delete"]
    assert deleted and set(deleted).issubset(inserted)

    lanes = stress.partition_query_lanes(list(range(20)), 4)
    assert lanes == [list(range(offset, 20, 4)) for offset in range(4)]
    assert all(len(lane) > 1 for lane in lanes)


def test_lifecycle_mutations_stay_ordered_in_one_writer_lane() -> None:
    mix = stress.parse_mutation_mix("predicate:4,vector:4,insert:1,delete:1")
    schedule = stress.build_mutation_schedule([10, 11, 12], [20, 21, 22], 30, 9, mix)
    lanes = stress.partition_mutation_schedule(schedule, 3)
    assert all(event.mutation not in {"insert", "delete"} for lane in lanes[1:] for event in lane)
    lifecycle = [event for event in lanes[0] if event.mutation in {"insert", "delete"}]
    for event in lifecycle:
        if event.mutation == "delete":
            prior_inserts = {
                prior.target_id for prior in lifecycle
                if prior.sequence < event.sequence and prior.mutation == "insert"
            }
            assert event.target_id in prior_inserts


def test_commit_tracker_records_first_last_and_mutation_mix() -> None:
    tracker = stress.CommitTracker()
    events = [
        stress.MutationEvent(0, "predicate", 1, 2),
        stress.MutationEvent(1, "vector", 3, 4),
        stress.MutationEvent(2, "insert", -3, 4),
    ]
    for event in events:
        tracker.record(event, writer_id=0)
    evidence = tracker.evidence(0, 3)
    assert evidence["overlap_count"] == 3
    assert evidence["first_commit"]["commit"] == 1
    assert evidence["last_commit"]["commit"] == 3
    assert evidence["mutation_counts"]["predicate"] == 1
    assert evidence["mutation_counts"]["delete"] == 0


def test_same_snapshot_and_tie_diagnostics_do_not_require_recall_one() -> None:
    assert stress.same_snapshot_contract("100:100:", "100:100:") is True
    assert stress.same_snapshot_contract("100:100:", "101:101:") is False
    exact = [(1, 0.1), (2, 0.2)]
    stock = [(1, 0.1), (9, 0.9)]
    guided = [(1, 0.1), (9, 0.9)]
    stock_recall = stress.exact_tie_aware_match(stock, exact, 2)
    guided_recall = stress.exact_tie_aware_match(guided, exact, 2)
    assert stock_recall["recall_at_k"] == guided_recall["recall_at_k"] == 0.5
    assert stock_recall["passed"] is False
    assert stress.ordered_ann_equivalence(stock, guided)["passed"] is True


def test_ordered_equivalence_rejects_id_or_distance_changes() -> None:
    stock = [(1, 0.1), (2, 0.2)]
    assert stress.ordered_ann_equivalence(stock, [(1, 0.1), (2, 0.2)])["passed"] is True
    assert stress.ordered_ann_equivalence(stock, [(2, 0.1), (1, 0.2)])["passed"] is False
    assert stress.ordered_ann_equivalence(stock, [(1, 0.1), (2, 0.3)])["passed"] is False


def _guided_scan(final_path: str = "validation_only", **extra: object) -> dict[str, object]:
    return {
        "profile_semantics_version": 10,
        "valid": True,
        "final_path": final_path,
        "planner_proof_succeeded": True,
        "planner_proof_guide_generation": 4,
        **extra,
    }


def test_safe_guided_profile_rejects_stale_admission_but_allows_safe_fallback() -> None:
    guidance = {
        "epoch_tracked": True, "relation_epoch": 3, "relation_relfilenode": 9,
        "guide_generation": 4, "effective_active": True,
    }
    stale = stress.classify_profile(
        "sqlens_guided", guidance, _guided_scan(), snapshot_epoch=4, relation_relfilenode=9,
    )
    fallback = stress.classify_profile(
        "sqlens_guided", guidance,
        _guided_scan("stock_bypass", stock_bypass_reason="stale_relation", planner_proof_succeeded=False),
        snapshot_epoch=4, relation_relfilenode=9,
    )
    assert stale["passed"] is False and stale["stale_guide_admitted"] is True
    assert fallback["passed"] is True and fallback["classification"] == "stale_fallback"
    assert fallback["filter_strategy"] == "safe_guided"


def test_post_update_lifecycle_requires_epoch_advance_and_refresh_or_fallback() -> None:
    refresh = stress.classify_post_update_lifecycle(
        1, 3, {"relation_epoch": 3, "fragment_builds": 1}, {"stale_fallback": False},
    )
    fallback = stress.classify_post_update_lifecycle(
        1, 3, {"relation_epoch": 1}, {"stale_fallback": True},
    )
    unchanged = stress.classify_post_update_lifecycle(
        3, 3, {"relation_epoch": 3, "fragment_builds": 1}, {"stale_fallback": False},
    )
    assert refresh["passed"] is True and refresh["mode"] == "epoch_refresh"
    assert fallback["passed"] is True and fallback["mode"] == "safe_stale_fallback"
    assert unchanged["passed"] is False


def test_guidance_signature_is_exact_kind_without_legacy_prefix() -> None:
    assert stress.GUIDANCE_ATOMS == ["sql:has_price AND price <= 20"]
    statement = stress.hybrid_statement(sql.Identifier("fixture"), True).as_string(None)
    assert "vector_hnsw_guidance_bind" in statement
    assert "'exact'" in statement
    assert "exact:sql:" not in statement


def test_scratch_hnsw_uses_unqualified_index_name(monkeypatch: pytest.MonkeyPatch) -> None:
    args = stress.create_argument_parser().parse_args([])
    cursor = mock.MagicMock()
    monkeypatch.setattr(
        stress,
        "relation_identity",
        lambda _cursor, relation: {
            "name": relation,
            "oid": 1,
            "relfilenode": 2,
            "reltuples": 3,
            "bytes": 4,
        },
    )

    stress.create_scratch(cursor, args)

    statements = [
        call.args[0].as_string(None)
        if isinstance(call.args[0], (sql.SQL, sql.Composed))
        else str(call.args[0])
        for call in cursor.execute.call_args_list
    ]
    create_index = next(statement for statement in statements if statement.startswith("CREATE INDEX"))
    assert f'CREATE INDEX "{args.scratch_index_name}" ON' in create_index
    assert f'CREATE INDEX "{args.scratch_schema}"."{args.scratch_index_name}"' not in create_index


def test_explain_plan_gate_finds_expected_hnsw_for_both_arms() -> None:
    args = stress.create_argument_parser().parse_args([])
    cursor = mock.MagicMock()
    cursor.fetchone.side_effect = [
        ([{"Plan": {"Index Name": args.scratch_index_name}}],),
        (1,),
        ([{"Plan": {"Index Name": args.scratch_index_name}}],),
    ]
    gates = stress.explain_index_plan_gate(cursor, args, query_id=7)
    assert gates["stock"]["passed"] is True
    assert gates["sqlens_guided"]["passed"] is True
    executed = [str(call.args[0]) for call in cursor.execute.call_args_list]
    assert any("safe_guided" in statement for statement in executed)


def test_reader_session_binds_preferred_index_and_records_backend_pid() -> None:
    args = stress.create_argument_parser().parse_args([])
    cursor = mock.MagicMock()
    expected = f"{args.scratch_schema}.{args.scratch_index_name}"
    cursor.fetchone.return_value = (4321, expected, True)
    evidence = stress.configure_reader_session(cursor, args)
    assert evidence == {
        "backend_pid": 4321,
        "preferred_index_current_setting": expected,
        "preferred_index_matches": True,
    }
    calls = [call.args for call in cursor.execute.call_args_list]
    assert any(args.preferred_index_guc in str(arguments) for arguments in calls)


def test_formal_gate_requires_identity_and_substantial_overlap(tmp_path) -> None:
    args = stress.create_argument_parser().parse_args([])
    assert stress.formal_protocol_status(args)["formal"] is False
    args.execute = True
    args.expected_sqlens_build_id = stress.R36_BUILD_ID
    args.expected_vector_so_sha256 = stress.R36_VECTOR_SO_SHA256
    args.expected_runner_sha256 = "e" * 64
    args.expected_git_revision = "d" * 40
    assert stress.formal_protocol_status(args)["formal"] is True


def test_source_identity_and_paper_eligibility_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = stress.create_argument_parser().parse_args([])
    args.expected_runner_sha256 = "a" * 64
    args.expected_git_revision = "b" * 40
    monkeypatch.setattr(
        stress,
        "git_identity",
        lambda: {
            "runner_sha256": "a" * 64,
            "git_revision": "b" * 40,
            "runner_path": "runner.py",
            "runner_git_status": "",
            "runner_tracked_clean": True,
        },
    )
    source = stress.source_identity_gate(args)
    assert source["passed"] is True
    assert stress.paper_eligibility(
        True, {"formal": True}, source
    ) is True
    assert stress.paper_eligibility(
        True, {"formal": False}, source
    ) is False
    assert stress.paper_eligibility(
        True, {"formal": True}, source | {"passed": False}
    ) is False
    assert stress.paper_eligibility(
        False, {"formal": True}, source
    ) is False


def test_runtime_identity_hashes_server_loaded_vector_and_binds_optional_mirror(tmp_path) -> None:
    mirror = tmp_path / "vector.so"
    mirror.write_bytes(b"server binary")
    expected_sha = stress.sha256_file(mirror)
    cursor = mock.MagicMock()
    cursor.fetchone.return_value = (
        "sqlens-v16-release",
        "17.5",
        "0.8.0",
        "/usr/lib/postgresql/17/lib/vector.so",
        expected_sha,
    )

    identity = stress.runtime_identity(
        cursor, "sqlens-v16-release", expected_sha, mirror
    )

    assert identity["vector_so"] == "/usr/lib/postgresql/17/lib/vector.so"
    assert identity["vector_so_sha256"] == expected_sha
    assert identity["vector_so_identity_source"].startswith("server_pg_config")
    assert identity["client_vector_so_matches_loaded"] is True
    statement = cursor.execute.call_args.args[0]
    assert "pg_read_binary_file" in statement
    assert "PKGLIBDIR" in statement


def test_runtime_identity_rejects_server_sha_mismatch() -> None:
    cursor = mock.MagicMock()
    cursor.fetchone.return_value = (
        "sqlens-v16-release",
        "17.5",
        "0.8.0",
        "/server/vector.so",
        "a" * 64,
    )
    with pytest.raises(stress.StressContractError, match="server-loaded vector.so"):
        stress.runtime_identity(cursor, "sqlens-v16-release", "b" * 64)


def test_dry_run_never_connects_to_database(monkeypatch: pytest.MonkeyPatch) -> None:
    args = stress.create_argument_parser().parse_args(["--dry-run"])
    monkeypatch.setattr(
        stress.psycopg, "connect",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("DB touched")),
    )
    result = stress.run_experiment(args)
    assert result["database_connected"] is False
    assert result["contract"] == stress.CORRECTNESS_CONTRACT


def _valid_record(overlap: int = 1) -> dict[str, object]:
    return {
        "passed": True,
        "backend_pid": 123,
        "snapshot_matches": True,
        "guided_profile_classification": {"stale_guide_admitted": False},
        "ordered_stock_guided_equivalence": {"passed": True},
        "same_tie_aware_recall": True,
        "stock_sql_valid": True,
        "guided_sql_valid": True,
        "exact_scan_gucs": ("off", "off", "off", "on"),
        "stock_exact_recall": {"recall_at_k": 0.8},
        "guided_exact_recall": {"recall_at_k": 0.8},
        "commit_overlap": {"overlap_count": overlap},
    }


def _valid_backend() -> dict[str, object]:
    return {
        "passed": True,
        "backend_pid": 123,
        "pre_update_lifecycle": {"pre_update_guided_reuse": True},
        "post_update_lifecycle_evidence": True,
        "post_update_lifecycle": [{"query_id": 7, "snapshot_epoch": 2}],
    }


def _interval() -> dict[str, object]:
    return {
        "overlap_count": 4,
        "first_commit": {"commit": 1},
        "last_commit": {"commit": 4},
        "mutation_counts": {name: 1 for name in stress.MUTATIONS},
    }


def test_artifact_gate_requires_backend_lifecycle_and_real_overlap() -> None:
    committed = {name: 1 for name in stress.MUTATIONS}
    valid = stress.artifact_validity(
        [_valid_record()], [_valid_backend()], [], committed, 4, _interval(), 1, 1, 1,
    )
    assert valid["artifact_valid"] is True

    invalid = stress.artifact_validity(
        [_valid_record(overlap=0)], [_valid_backend() | {"post_update_lifecycle_evidence": False}],
        [], committed, 4, _interval(), 1, 1, 1,
    )
    assert invalid["artifact_valid"] is False
    assert "persistent_backend_lifecycle" in invalid["failed_gates"]
    assert "insufficient_query_commit_overlap" in invalid["failed_gates"]


def test_manifest_validator_preserves_paired_and_backend_evidence() -> None:
    identity = {"oid": 10, "relfilenode": 20}
    summary = stress.correctness_summary([_valid_record()])
    runtime = {
        "sqlens_build_id": "sqlens-test",
        "vector_so_sha256": "a" * 64,
        "vector_so_identity_source": "server_pg_config_pkglibdir_pg_read_binary_file",
    }
    manifest = {
        "artifact_schema_version": stress.ARTIFACT_SCHEMA_VERSION,
        "runtime": runtime,
        "runtime_identity_binding": {
            "loaded_sqlens_build_id": runtime["sqlens_build_id"],
            "loaded_vector_so_sha256": runtime["vector_so_sha256"],
            "build_id_matches_expected": True,
            "vector_so_sha256_matches_expected": True,
        },
        "source_identity": {
            "runner_sha256": "d" * 64,
            "git_revision": "e" * 40,
            "runner_sha256_matches_expected": True,
            "git_revision_matches_expected": True,
        },
        "scratch": {"source": identity, "table": identity, "index": identity},
        "raw_output": {"sha256": "b" * 64}, "records_sha256": "c" * 64,
        "artifact_valid": True, "artifact_gates": {"artifact_valid": True},
        "backend_lifecycles": [_valid_backend()],
        "reader_interval_overlap": _interval(),
        "correctness_summary": summary,
        "query_diagnostics": stress.manifest_query_diagnostics([_valid_record()]),
    }
    assert stress.validate_manifest_payload(manifest) == []
    broken = manifest | {"correctness_summary": summary | {"ordered_equivalent": 0}}
    assert "paired_correctness_summary" in stress.validate_manifest_payload(broken)


def test_parser_rejects_nonpersistent_or_nonreal_protocols() -> None:
    with pytest.raises(SystemExit):
        stress.create_argument_parser().parse_args([
            "--mutation-mix", "predicate:1,vector:0,insert:1,delete:1",
        ])
    args = stress.create_argument_parser().parse_args(["--queries", "4", "--reader-clients", "4"])
    with pytest.raises(stress.StressContractError, match="at least two"):
        stress.validate_args(args)
