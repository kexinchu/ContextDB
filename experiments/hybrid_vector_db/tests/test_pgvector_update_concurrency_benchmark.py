from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from experiments.hybrid_vector_db.scripts import pgvector_update_concurrency_benchmark as runner


def test_default_is_dry_run_and_declares_real_concurrent_contract(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.main([]) == 0
    payload = capsys.readouterr().out
    assert '"database_connected": false' in payload
    assert '"measurement_repeats": 6' in payload
    assert '"readers": [1, 4, 8, 16, 32, 64]' in payload
    assert '"independent_exact_sql_valid_spot_audit"' in payload
    assert '"physical_mvcc_tuple_rewrite": true' in payload


def test_execute_requires_audited_manifest_before_database_work() -> None:
    with pytest.raises(runner.BenchmarkContractError, match="matched-recall-manifest"):
        runner.run_experiment(runner.create_argument_parser().parse_args(["--execute"]))


def test_schedule_arguments_require_six_repeats_and_positive_measurement() -> None:
    args = runner.create_argument_parser().parse_args(["--measurement-repeats", "4"])
    with pytest.raises(runner.BenchmarkContractError, match="at least six"):
        runner.validate_args(args)

    with pytest.raises(SystemExit):
        runner.create_argument_parser().parse_args(["--measure-seconds", "0"])


def test_audit_selection_is_deterministic_unique_and_bounded() -> None:
    query_ids = {number: 1000 + number for number in range(20)}
    first = runner.select_audit_requests(query_ids, "filter", 2, 5, 77)
    second = runner.select_audit_requests(query_ids, "filter", 2, 5, 77)
    assert first == second
    assert len(first) == 5
    assert len({query_no for query_no, _ in first}) == 5
    with pytest.raises(runner.BenchmarkContractError, match="fewer query"):
        runner.select_audit_requests(query_ids, "filter", 2, 21, 77)


def test_repeat_summary_reports_errors_timeouts_and_real_percentiles() -> None:
    rows = [
        {"kind": "read", "measurement_repeat": 0, "method": "stock", "filter_name": "f", "readers": 1,
         "writer_clients": 1, "update_rate_tps": 0.5, "latency_ms": 3.0, "error": "", "timeout": False,
         "recall_at_10": 0.9, "query_no": 0, "profile_complete": True, "path_class": "stock"},
        {"kind": "read", "measurement_repeat": 0, "method": "stock", "filter_name": "f", "readers": 1,
         "writer_clients": 1, "update_rate_tps": 0.5, "latency_ms": 7.0, "error": "timeout", "timeout": True,
         "recall_at_10": 0.0, "query_no": 1, "profile_complete": False, "path_class": "stock"},
        {"kind": "write", "measurement_repeat": 0, "method": "stock", "filter_name": "f", "readers": 1,
         "writer_clients": 1, "update_rate_tps": 0.5, "latency_ms": 2.0, "error": "", "timeout": False,
         "schedule_lag_ms": 4.0},
    ]
    read, write = runner.summarize_repeat(rows, 2.0)
    assert read["completed"] == 1
    assert read["errors"] == 1
    assert read["timeouts"] == 1
    assert read["qps"] == 0.5
    assert read["status"] == "invalid"
    assert write["p50_ms"] == 2.0
    assert write["status"] == "valid"
    assert write["achieved_update_tps"] == 0.5
    assert write["writer_schedule_lag_p95_ms"] == 4.0


def test_zero_update_rate_is_explicitly_not_applicable_not_a_fake_zero_latency() -> None:
    rows = [{"kind": "read", "measurement_repeat": 0, "method": "stock", "filter_name": "f", "readers": 1,
             "writer_clients": 1, "update_rate_tps": 0.0, "latency_ms": 1.0, "error": "", "timeout": False,
             "recall_at_10": 1.0, "query_no": 0, "profile_complete": True, "path_class": "stock"}]
    read, write = runner.summarize_repeat(rows, 1.0)
    assert read["status"] == "valid"
    assert write["status"] == "not_applicable"
    aggregate = runner.aggregate_summaries([write] * 5)[0]
    assert aggregate["status"] == "not_applicable"


def test_aggregate_does_not_mix_recall_targets_or_configs() -> None:
    rows = []
    for target, config, latency in ((0.90, "ef100", 1.0), (0.95, "ef500", 5.0)):
        for repeat in range(5):
            rows.append({
                "summary_type": "repeat", "kind": "read", "measurement_repeat": repeat,
                "method": "stock", "filter_name": "f", "target_recall": target,
                "config": config, "ef_search": int(config[2:]), "readers": 1,
                "writer_clients": 1, "update_rate_tps": 0.0, "wall_seconds": 1.0,
                "completed": 1, "qps": 1.0, "p50_ms": latency, "p95_ms": latency,
                "p99_ms": latency, "errors": 0, "timeouts": 0, "status": "valid",
            })
    aggregate = runner.aggregate_summaries(rows)
    assert len(aggregate) == 2
    assert {(row["target_recall"], row["config"]) for row in aggregate} == {
        (0.90, "ef100"), (0.95, "ef500")
    }


def test_aggregate_tail_pools_all_raw_requests_and_keeps_repeat_cluster_ci() -> None:
    repeat_rows = []
    raw_rows = []
    latencies = [1.0, 2.0, 3.0, 4.0, 100.0, 101.0]
    for repeat, latency in enumerate(latencies):
        repeat_rows.append({
            "summary_type": "repeat", "kind": "read", "measurement_repeat": repeat,
            "method": "stock", "filter_name": "f", "target_recall": 0.9,
            "config": "ef100", "ef_search": 100, "readers": 4,
            "writer_clients": 1, "update_rate_tps": 0.0, "wall_seconds": 1.0,
            "attempts": 1, "completed": 1, "p50_ms": 999.0, "p95_ms": 999.0,
            "p99_ms": 999.0, "errors": 0, "timeouts": 0, "status": "valid",
            "minimum_update_delivery_ratio": 0.9,
        })
        raw_rows.append({
            "kind": "read", "measurement_repeat": repeat, "method": "stock",
            "filter_name": "f", "target_recall": 0.9, "config": "ef100",
            "ef_search": 100, "readers": 4, "writer_clients": 1,
            "update_rate_tps": 0.0, "query_no": repeat, "recall_at_10": 1.0,
            "latency_ms": latency, "error": "", "profile_complete": True,
            "path_class": "stock",
        })

    aggregate = runner.aggregate_summaries(
        repeat_rows, raw_rows, bootstrap_samples=100, bootstrap_seed=19
    )[0]

    assert aggregate["p50_ms"] == 3.0
    assert aggregate["p95_ms"] == 101.0
    assert aggregate["tail_point_estimate_source"] == "all_successful_raw_requests_pooled"
    assert aggregate["tail_raw_request_pool_complete"] is True
    assert aggregate["p99_repeat_cluster_ci95_low_ms"] <= aggregate["p99_ms"]
    assert aggregate["p99_repeat_cluster_ci95_high_ms"] >= aggregate["p99_ms"]


def test_update_delivery_gate_records_tps_and_lag_and_fails_below_ninety_percent() -> None:
    base = {
        "measurement_repeat": 0, "method": "stock", "filter_name": "f",
        "readers": 1, "writer_clients": 1, "update_rate_tps": 10.0,
        "error": "", "timeout": False,
    }
    rows = [{
        **base, "kind": "read", "latency_ms": 1.0, "recall_at_10": 1.0,
        "query_no": 0, "profile_complete": True, "path_class": "stock",
    }]
    rows.extend({
        **base, "kind": "write", "latency_ms": 0.5, "schedule_lag_ms": float(index),
    } for index in range(8))

    read, write = runner.summarize_repeat(rows, 1.0)

    assert read["requested_update_tps"] == 10.0
    assert read["achieved_update_tps"] == 8.0
    assert read["update_delivery_ratio"] == 0.8
    assert read["minimum_update_delivery_ratio"] == 0.9
    assert read["update_delivery_gate_passed"] is False
    assert read["status"] == write["status"] == "overload"
    assert write["writer_schedule_lag_p99_ms"] == 7.0

    passing = runner.writer_delivery_metrics(
        [{**row} for row in rows[1:]] + [{
            **base, "kind": "write", "latency_ms": 0.5, "schedule_lag_ms": 8.0,
        }],
        1.0,
        10.0,
    )
    assert passing["update_delivery_ratio"] == 0.9
    assert passing["update_delivery_gate_passed"] is True


def test_exact_audit_count_is_independent_of_arms_and_client_grid() -> None:
    assert runner.expected_exact_audit_rows(
        (0.90, 0.95, 0.99), [object()] * 14, 6, 5
    ) == 3 * 14 * 6 * 5 * 2


def test_update_pool_is_relation_sampled_and_exact_size() -> None:
    args = runner.create_argument_parser().parse_args(["--update-id-pool-size", "3"])
    cursor = mock.MagicMock()
    cursor.fetchall.return_value = [(41,), (77,), (99,)]
    connection = mock.MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value = cursor
    relation = mock.MagicMock()
    relation.as_string.return_value = '"public"."amazon_grocery_reviews_10m_pgvector"'
    with mock.patch.object(runner.psycopg, "connect", return_value=connection), mock.patch.object(
        runner, "relation_identifier", return_value=relation
    ):
        assert runner.load_update_id_pool(args) == [41, 77, 99]
    sql = cursor.execute.call_args.args[0]
    assert "TABLESAMPLE SYSTEM (10)" in sql
    assert "REPEATABLE" in sql


def test_update_pool_excludes_query_rows_and_formal_writer_uses_active_arm() -> None:
    args = runner.create_argument_parser().parse_args(["--update-id-pool-size", "2"])
    cursor = mock.MagicMock()
    cursor.fetchall.return_value = [(41,), (77,)]
    connection = mock.MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value = cursor
    relation = mock.MagicMock()
    relation.as_string.return_value = '"public"."amazon"'
    with mock.patch.object(
        runner.psycopg, "connect", return_value=connection,
    ), mock.patch.object(
        runner, "relation_identifier", return_value=relation,
    ):
        assert runner.load_update_id_pool(args, [10, 11]) == [41, 77]
    statement, params = cursor.execute.call_args.args
    assert "WHERE NOT (id = ANY(%s))" in statement
    assert params == ([10, 11], 2)

    args.insertion_table = "public.source"
    assert runner.writer_table_for_protocol(args, "public.bfs") == "public.source"
    args.protocol = runner.FORMAL_PROTOCOL
    assert runner.writer_table_for_protocol(args, "public.bfs") == "public.bfs"


def test_update_pool_offset_changes_rows_without_changing_rate_schedule() -> None:
    update_ids = [10, 20, 30, 40]
    first = runner.update_batch_ids(update_ids, pool_offset=0, sequence=0, batch_size=2)
    shifted = runner.update_batch_ids(update_ids, pool_offset=2, sequence=0, batch_size=2)
    assert first == [10, 20]
    assert shifted == [30, 40]


def test_main_does_not_execute_without_explicit_execute_flag() -> None:
    with mock.patch.object(runner, "run_experiment") as execute:
        assert runner.main(["--measurement-repeats", "6"]) == 0
    execute.assert_not_called()


def test_parse_rejects_unknown_method_and_negative_rate() -> None:
    with pytest.raises(SystemExit):
        runner.create_argument_parser().parse_args(["--methods", "stock,nope"])
    with pytest.raises(Exception):
        runner.parse_nonnegative_rate_list("-1")


def test_truth_parser_names_match_execute_path() -> None:
    args = runner.create_argument_parser().parse_args([])
    assert hasattr(args, "calibration_truth_csv")
    assert hasattr(args, "calibration_truth_manifest")
    assert hasattr(args, "measurement_truth_csv")
    assert hasattr(args, "measurement_truth_manifest")
    assert not hasattr(args, "truth_csv")
    assert not hasattr(args, "truth_manifest")


def test_query_id_split_gate_is_three_way_and_fail_closed() -> None:
    manifest = {
        "run_spec": {"args": {
            "calibration_query_offset": 0,
            "calibration_queries": 2,
            "final_query_offset": 2,
            "final_queries": 2,
        }}
    }
    calibration_ids = {0: 10, 1: 11, 2: 12, 3: 13}
    measurement = [SimpleNamespace(query_id=20), SimpleNamespace(query_id=21)]
    evidence = runner.query_id_disjoint_gate(manifest, calibration_ids, measurement)
    assert evidence["passed"] is True
    assert evidence["calibration_queries"] == 2
    assert evidence["confirmation_queries"] == 2
    assert evidence["measurement_queries"] == 2

    with pytest.raises(runner.BenchmarkContractError, match="overlap"):
        runner.query_id_disjoint_gate(
            manifest, calibration_ids, [SimpleNamespace(query_id=12)]
        )


def test_requests_select_unique_measurement_prefix_and_reject_replay() -> None:
    workload = SimpleNamespace(
        requests=tuple(
            SimpleNamespace(request_no=index, query_no=100 + index, query_id=1000 + index)
            for index in range(4)
        )
    )
    selected = runner.select_measurement_requests(workload, 3)
    assert [item.request_no for item in selected] == [0, 1, 2]
    with pytest.raises(runner.BenchmarkContractError, match="between 1"):
        runner.select_measurement_requests(workload, 5)

    duplicate = SimpleNamespace(
        requests=(
            SimpleNamespace(request_no=0, query_no=100, query_id=9),
            SimpleNamespace(request_no=1, query_no=101, query_id=9),
        )
    )
    with pytest.raises(runner.BenchmarkContractError, match="unique"):
        runner.select_measurement_requests(duplicate, 2)


def test_formal_overrides_downgrade_instead_of_mislabeling() -> None:
    args = runner.create_argument_parser().parse_args([])
    args.protocol = runner.FORMAL_PROTOCOL
    args.filter_names = list(runner.FORMAL_FILTERS)
    status = runner.formal_protocol_status(
        args,
        runner.FORMAL_TARGETS,
        runner.FORMAL_READERS,
        runner.FORMAL_UPDATE_RATES,
        runner.METHODS,
        filter_count=3,
        filter_names=runner.FORMAL_FILTERS,
        split_gate_passed=True,
        source_identity_passed=True,
        selector_bound=True,
    )
    assert status["formal"] is True
    assert status["label"] == "formal"

    args.requests = 9999
    downgraded = runner.formal_protocol_status(
        args,
        runner.FORMAL_TARGETS,
        runner.FORMAL_READERS,
        runner.FORMAL_UPDATE_RATES,
        runner.METHODS,
        filter_count=3,
        filter_names=runner.FORMAL_FILTERS,
        split_gate_passed=True,
        source_identity_passed=True,
        selector_bound=True,
    )
    assert downgraded["formal"] is False
    assert downgraded["label"] == "nonformal_debug"
    assert "requests_q10k" in downgraded["failed_checks"]


def test_formal_validation_installs_sensitivity_slice_and_full_sqlens() -> None:
    args = runner.create_argument_parser().parse_args([
        "--protocol", runner.FORMAL_PROTOCOL,
    ])
    targets, readers, rates, methods = runner.validate_args(args)
    assert tuple(targets) == runner.FORMAL_TARGETS
    assert tuple(readers) == runner.FORMAL_READERS
    assert tuple(rates) == runner.FORMAL_UPDATE_RATES
    assert tuple(methods) == ("stock", "sqlens_full")
    assert tuple(args.filter_names) == runner.FORMAL_FILTERS

    args.methods = ["stock", "sqlens_d1"]
    with pytest.raises(runner.BenchmarkContractError, match="stock,sqlens_full"):
        runner.validate_args(args)


def test_source_identity_requires_exact_runner_and_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = runner.create_argument_parser().parse_args([])
    args.expected_runner_sha256 = runner.sha256_file(
        runner.Path(runner.__file__).resolve()
    )
    args.expected_git_revision = "b" * 40
    calls = iter(("b" * 40 + "\n", ""))

    def completed(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=next(calls))

    monkeypatch.setattr(runner.subprocess, "run", completed)
    identity = runner.source_identity(args)
    assert identity["runner_sha256_matches_expected"] is True
    assert identity["git_revision_matches_expected"] is True
    assert identity["runner_tracked_clean"] is True


def test_fixed_selector_binds_r36_and_current_input_hashes(tmp_path) -> None:
    filters_csv = tmp_path / "filters.csv"
    truth_csv = tmp_path / "truth.csv"
    workload_csv = tmp_path / "workload.csv"
    filters_csv.write_text("filters\n", encoding="utf-8")
    truth_csv.write_text("truth\n", encoding="utf-8")
    workload_csv.write_text("workload\n", encoding="utf-8")
    query_contract = {
        "filters_sha256": runner.sha256_file(filters_csv),
        "truth_sha256": runner.sha256_file(truth_csv),
        "workload_sha256": runner.sha256_file(workload_csv),
    }
    bindings = []
    for config_id in ("stock_cfg", "sqlens_cfg"):
        plan = tmp_path / f"{config_id}.json"
        plan.write_text(
            json.dumps({"query_contract": query_contract}) + "\n",
            encoding="utf-8",
        )
        bindings.append({
            "dataset": "amazon",
            "config_id": config_id,
            "input_plan": str(plan),
            "input_plan_sha256": runner.sha256_file(plan),
        })
    selector = tmp_path / "selector.csv"
    runner.write_csv_atomic(selector, [{
        "dataset": "amazon",
        "target_recall": 0.90,
        "selection_status": "selected",
        "pair_id": "amazon-r36-r090",
        "stock_config_id": "stock_cfg",
        "sqlens_config_id": "sqlens_cfg",
        "stock_config_sha256": "1" * 64,
        "sqlens_config_sha256": "2" * 64,
        "stock_ef_search": 100,
        "stock_max_scan_tuples": 200_000,
        "stock_scan_mem_multiplier": 8,
        "stock_iterative_scan": "strict_order",
        "stock_guided_collect_target": 11,
        "stock_traversal_guided_target": 11,
        "sqlens_ef_search": 80,
        "sqlens_max_scan_tuples": 200_000,
        "sqlens_scan_mem_multiplier": 8,
        "sqlens_iterative_scan": "strict_order",
        "sqlens_guided_collect_target": 11,
        "sqlens_traversal_guided_target": 11,
        "stock_table": "public.amazon_source",
        "stock_index": "amazon_source_hnsw",
        "sqlens_table": "public.amazon_bfs",
        "sqlens_index": "amazon_bfs_hnsw",
        "sqlens_d2_page_access": "bfs",
        "sqlens_d2_index_page_access": "bfs",
    }])
    manifest = tmp_path / "selector.manifest.json"
    manifest.write_text(json.dumps({
        "artifact_valid": True,
        "release_contract": {
            "expected_sqlens_build_id": runner.R36_BUILD_ID,
            "expected_vector_so_sha256": runner.R36_VECTOR_SO_SHA256,
        },
        "outputs": {
            "measurement_plan_csv": {
                "sha256": runner.sha256_file(selector),
            },
        },
        "input_bindings": bindings,
    }) + "\n", encoding="utf-8")
    args = runner.create_argument_parser().parse_args([])
    args.fixed_recall_selector_csv = selector
    args.fixed_recall_selector_manifest = manifest
    args.filters_csv = filters_csv
    args.measurement_truth_csv = truth_csv
    args.fixed_selector_workload_csv = workload_csv
    spec = runner.throughput.FilterSpec(
        "popular_ge1000", "helpful_votes >= 1000",
        ("helpful_votes >= 1000",), 5_000_000, 50.0,
    )

    matched = runner.load_fixed_recall_selector(args, [spec])

    assert set(matched.configs) == {
        ("popular_ge1000", "stock", 0.90),
        ("popular_ge1000", "sqlens_full", 0.90),
    }
    assert matched.configs[("popular_ge1000", "stock", 0.90)].ef_search == 100
    assert matched.configs[("popular_ge1000", "sqlens_full", 0.90)].ef_search == 80
    assert matched.provenance["filters_sha256"] == query_contract["filters_sha256"]
    assert matched.provenance["measurement_truth_sha256"] == query_contract["truth_sha256"]
    assert matched.provenance["selector_workload_sha256"] == query_contract["workload_sha256"]
    assert args.insertion_table == "public.amazon_source"
    assert args.bfs_table == "public.amazon_bfs"

    truth_csv.write_text("changed truth\n", encoding="utf-8")
    with pytest.raises(runner.BenchmarkContractError, match="truth_sha256"):
        runner.load_fixed_recall_selector(args, [spec])


def test_per_cell_checkpoint_roundtrip_and_contract_mismatch(tmp_path) -> None:
    key = runner.cell_key(0.90, "helpful_ge20", 16, 100.0, 2, "sqlens_full")
    checkpoint = runner.persist_cell(
        tmp_path,
        key,
        "c" * 64,
        [{"kind": "read", "query_no": 7}],
        [{"kind": "read", "status": "valid"}],
        {"workers": []},
        [{"kind": "profile", "profile_complete": True}],
        {"passed": True},
    )
    assert checkpoint["status"] == "complete"
    loaded = runner.load_cell(tmp_path, key, "c" * 64)
    assert loaded is not None
    assert loaded["raw"][0]["query_no"] == 7
    assert loaded["profiles"][0]["profile_complete"] is True
    with pytest.raises(runner.BenchmarkContractError, match="identity mismatch"):
        runner.load_cell(tmp_path, key, "d" * 64)


def test_real_mutation_schedule_and_lifecycle_gate() -> None:
    mix = runner.parse_mutation_mix("predicate:4,vector:4,insert:1,delete:1")
    observed = {
        runner.mutation_choice(sequence, mix, 23)
        for sequence in range(sum(mix.values()))
    }
    assert observed == set(runner.MUTATIONS)

    cursor = mock.MagicMock()
    cursor.rowcount = 1
    statements = {name: object() for name in (
        "snapshot", "predicate", "vector", "insert", "delete",
    )}
    statements["predicate_threshold"] = 20
    lifecycle_ids: list[int] = []
    mutation, affected, target = runner.execute_mutation(
        cursor, statements, "predicate", [10], 20, lifecycle_ids, -1,
    )
    assert (mutation, affected, target) == ("predicate", 1, 10)
    assert cursor.execute.call_args_list[-1].args == (
        statements["predicate"], (20, 20, [10]),
    )
    mutation, _, target = runner.execute_mutation(
        cursor, statements, "vector", [11], 20, lifecycle_ids, -2,
    )
    assert (mutation, target) == ("vector", 11)
    mutation, _, target = runner.execute_mutation(
        cursor, statements, "insert", [12], 20, lifecycle_ids, -3,
    )
    assert (mutation, target, lifecycle_ids) == ("insert", -3, [-3])
    mutation, _, target = runner.execute_mutation(
        cursor, statements, "delete", [12], 20, lifecycle_ids, -4,
    )
    assert (mutation, target, lifecycle_ids) == ("delete", -3, [])

    rows = [
        {"kind": "write", "mutation": name, "error": ""}
        for name in runner.MUTATIONS
    ]
    evidence = {
        "relation_epoch_before": 10,
        "relation_epoch_after": 14,
        "relation_epoch_delta": 4,
    }
    profiles = [{
        "profile_complete": True,
        "fragment_builds_delta": 1,
        "fast_reactivation_hits_delta": 0,
        "fragment_store_hits_delta": 0,
    }]
    gate = runner.lifecycle_gate(
        "sqlens_full", 100.0, rows, evidence, profiles,
    )
    assert gate["passed"] is True
    assert gate["all_mutation_types_observed"] is True
    assert gate["invalidations_cover_commits"] is True
    assert gate["invalidation_events_observed"] == 4
    assert gate["fragment_rebuilds_observed"] == 1
    assert gate["fast_reactivations_observed"] == 0
    failed = runner.lifecycle_gate(
        "sqlens_full", 100.0, rows,
        evidence | {"relation_epoch_after": 13, "relation_epoch_delta": 3},
        profiles,
    )
    assert failed["passed"] is False
    assert failed["invalidations_cover_commits"] is False


def test_formal_predicate_mutation_targets_the_active_filter_column() -> None:
    cursor = mock.MagicMock()
    cursor.fetchall.return_value = [
        ("id",), ("embedding",), ("item_rating_number",),
        ("review_text_len",), ("helpful_vote",),
    ]
    statements = runner.mutation_sql(
        cursor, "public.amazon_items", "helpful_ge20"
    )
    assert statements["predicate_column"] == "helpful_vote"
    assert statements["predicate_threshold"] == 20
    assert '"helpful_vote"' in statements["predicate"].as_string(None)
    assert '"helpful_vote"' in statements["restore"].as_string(None)
    with pytest.raises(runner.BenchmarkContractError, match="no real predicate"):
        runner.mutation_sql(cursor, "public.amazon_items", "unknown_filter")


def _formal_read_aggregate(*, completed: int = 60_000, repeats: int = 6) -> dict[str, object]:
    return {
        "completed": completed,
        "repeats": repeats,
        "tail_raw_request_pool_complete": True,
        "status": "valid",
        "target_recall_lcb95_met": True,
        "profile_complete": True,
        "update_delivery_gate_passed": True,
    }


def test_nonformal_diagnostic_slice_cannot_be_artifact_or_paper_eligible() -> None:
    args = runner.create_argument_parser().parse_args([])
    args.requests = 2
    protocol = runner.formal_protocol_status(
        args,
        runner.FORMAL_TARGETS,
        runner.FORMAL_READERS,
        runner.FORMAL_UPDATE_RATES,
        runner.METHODS,
        filter_count=14,
        split_gate_passed=True,
    )
    result = runner.artifact_eligibility(
        diagnostic_valid=True,
        protocol=protocol,
        read_aggregates=[_formal_read_aggregate()],
        expected_read_cells=1,
    )
    assert result["diagnostic_valid"] is True
    assert result["artifact_valid"] is False
    assert result["paper_eligible"] is False
    assert result["formal_checks"]["formal_protocol_requested"] is False


def test_formal_artifact_requires_every_q10k_cell_and_runtime_gate() -> None:
    args = runner.create_argument_parser().parse_args([])
    args.protocol = runner.FORMAL_PROTOCOL
    args.filter_names = list(runner.FORMAL_FILTERS)
    protocol = runner.formal_protocol_status(
        args,
        runner.FORMAL_TARGETS,
        runner.FORMAL_READERS,
        runner.FORMAL_UPDATE_RATES,
        runner.METHODS,
        filter_count=3,
        filter_names=runner.FORMAL_FILTERS,
        split_gate_passed=True,
        source_identity_passed=True,
        selector_bound=True,
    )
    incomplete = runner.artifact_eligibility(
        diagnostic_valid=True,
        protocol=protocol,
        read_aggregates=[_formal_read_aggregate(completed=59_999)],
        expected_read_cells=1,
    )
    assert incomplete["diagnostic_valid"] is True
    assert incomplete["artifact_valid"] is False
    assert incomplete["paper_eligible"] is False
    assert incomplete["formal_checks"]["q10k_reads_per_repeat_per_cell"] is False

    eligible = runner.artifact_eligibility(
        diagnostic_valid=True,
        protocol=protocol,
        read_aggregates=[_formal_read_aggregate()],
        expected_read_cells=1,
        sampled_profiles=[{"profile_complete": True, "error": ""}],
        lifecycle_gates=[{"passed": True}],
    )
    assert eligible["diagnostic_valid"] is True
    assert eligible["artifact_valid"] is True
    assert eligible["paper_eligible"] is True


def test_pooled_recall_lcb_invalidates_cell_below_target() -> None:
    repeat_rows = []
    raw_rows = []
    for repeat in range(5):
        repeat_rows.append({
            "summary_type": "repeat", "kind": "read", "measurement_repeat": repeat,
            "method": "stock", "filter_name": "f", "target_recall": 0.9,
            "config": "ef100", "ef_search": 100, "readers": 1,
            "writer_clients": 1, "update_rate_tps": 10.0, "wall_seconds": 1.0,
            "completed": 2, "p50_ms": 1.0, "p95_ms": 1.0, "p99_ms": 1.0,
            "errors": 0, "timeouts": 0, "status": "valid",
        })
        for query_no in range(2):
            raw_rows.append({
                "kind": "read", "measurement_repeat": repeat, "method": "stock",
                "filter_name": "f", "target_recall": 0.9, "config": "ef100",
                "ef_search": 100, "readers": 1, "writer_clients": 1,
                "update_rate_tps": 10.0, "query_no": query_no,
                "recall_at_10": 0.7, "latency_ms": 1.0, "error": "",
                "profile_complete": True, "path_class": "stock",
            })
    aggregate = runner.aggregate_summaries(
        repeat_rows, raw_rows, bootstrap_samples=100, bootstrap_seed=7
    )[0]
    assert aggregate["pooled_recall_lcb95"] == pytest.approx(0.7)
    assert aggregate["target_recall_lcb95_met"] is False
    assert aggregate["status"] == "invalid"


def test_profile_path_classification_covers_guided_stale_stock_and_unknown() -> None:
    assert runner.classify_profile_path(
        "sqlens_d1", {"final_path": "candidate_admission_validation_only"}
    ) == ("guided", False)
    assert runner.classify_profile_path(
        "sqlens_d1", {
            "final_path": "stock_bypass",
            "planner_proof_bypass_reason": "stale_relation",
        },
    ) == ("stale_fallback", True)
    assert runner.classify_profile_path(
        "sqlens_d1", {"final_path": "stock_bypass"}
    ) == ("stock", False)
    assert runner.classify_profile_path(
        "sqlens_d1", {"final_path": "future_path"}
    ) == ("unknown", False)


def test_stock_is_initialized_once_and_request_does_not_activate() -> None:
    cursor = mock.MagicMock()
    runtime = SimpleNamespace(cur=cursor, mode="original")
    args = SimpleNamespace(
        k=10,
        guidance_filter_strategy="safe_guided",
        candidate_validity_predicate="embedding_valid",
        query_id_column="id",
        query_vector_column="embedding",
    )
    filter_spec = SimpleNamespace(name="f", predicate="price > 0")
    runner.initialize_search_runtime(args, runtime, "stock")
    assert cursor.execute.call_count == 2
    cursor.reset_mock()

    with (
        mock.patch.object(runner, "activate") as activate,
        mock.patch.object(runner, "mode_table_index", return_value=("public.items", "public.idx")),
        mock.patch.object(runner, "candidate_self_exclusion", return_value=True),
        mock.patch.object(runner, "query_table_for_candidate", return_value="public.items"),
        mock.patch.object(runner, "run_query", return_value=([1], [0.1], {})),
        mock.patch.object(runner, "read_scan_profile", return_value={"final_path": "stock"}),
        mock.patch.object(runner, "read_guidance_profile", return_value={}),
        mock.patch.object(runner, "read_relation_epoch", return_value=4),
        mock.patch.object(runner, "tie_aware_recall", return_value=1.0),
    ):
        result = runner.execute_profiled_search(
            args, runtime, "stock", filter_spec, 100, 77, object(), {}
        )
    activate.assert_not_called()
    assert result["path_class"] == "stock"
    assert result["profile_complete"] is True
    assert result["relation_epoch_after_scan"] == 4


def test_run_overlap_dispatches_each_requested_query_exactly_once() -> None:
    requests = tuple(
        SimpleNamespace(request_no=index, query_no=100 + index, query_id=1000 + index)
        for index in range(5)
    )
    truth = {("f", request.query_no): object() for request in requests}
    runtimes = [
        SimpleNamespace(cur=mock.MagicMock(), mode="original"),
        SimpleNamespace(cur=mock.MagicMock(), mode="original"),
    ]
    args = SimpleNamespace(
        requests=5, writer_clients=1, start_barrier_timeout_seconds=2,
        schedule_seed=7, session_warmup_requests=0, warmup_seconds=0.0,
        measure_seconds=5, write_statement_timeout_ms=1000,
        insertion_table="public.items", update_column="review_text",
        update_batch_size=1,
    )
    filter_spec = SimpleNamespace(name="f", actual_pct=1.0, predicate="price > 0")

    def searched(_args, _runtime, _method, _filter, query_no, query_id, _truth, _state):
        return {
            "query_no": query_no, "query_id": query_id, "latency_ms": 1.0,
            "error": "", "error_type": "", "timeout": False, "returned": 10,
            "recall_at_10": 1.0, "profile_complete": True,
            "path_class": "stock", "final_path": "stock", "stale_relation": False,
        }

    writer_connection = mock.MagicMock()
    with (
        mock.patch.object(runner.throughput, "configure_args_for_runtime", return_value="original"),
        mock.patch.object(runner, "open_mode_runtime", side_effect=runtimes),
        mock.patch.object(runner, "initialize_search_runtime"),
        mock.patch.object(runner, "execute_profiled_search", side_effect=searched),
        mock.patch.object(runner, "writer_sql", return_value="UPDATE items SET x=x"),
        mock.patch.object(runner.psycopg, "connect", return_value=writer_connection),
        mock.patch.object(runner, "close_mode_runtime"),
    ):
        rows, _wall, evidence = runner.run_overlap(
            args, "stock", object(), filter_spec, truth, requests,
            2, 0.0, 0, [1], 0,
        )
    reads = [row for row in rows if row["kind"] == "read"]
    assert len(reads) == 5
    assert {row["request_no"] for row in reads} == set(range(5))
    assert {row["query_id"] for row in reads} == {1000, 1001, 1002, 1003, 1004}
    assert evidence["measurement_watchdog_timeout"] is False


def test_successful_mock_execute_path_uses_measurement_truth_and_exact_request_count(tmp_path) -> None:
    paths = {
        name: tmp_path / filename
        for name, filename in {
            "matched": "matched.json",
            "filters": "filters.csv",
            "cal_truth": "cal.csv",
            "cal_manifest": "cal_manifest.json",
            "query": "q10200.csv",
            "query_manifest": "q10200_manifest.json",
            "measure_truth": "measurement.csv",
            "measure_manifest": "measurement_manifest.json",
        }.items()
    }
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    out = tmp_path / "result.json"
    args = runner.create_argument_parser().parse_args([
        "--execute", "--matched-recall-manifest", str(paths["matched"]),
        "--filters-csv", str(paths["filters"]),
        "--calibration-truth-csv", str(paths["cal_truth"]),
        "--calibration-truth-manifest", str(paths["cal_manifest"]),
        "--measurement-query-file", str(paths["query"]),
        "--measurement-query-manifest", str(paths["query_manifest"]),
        "--measurement-truth-csv", str(paths["measure_truth"]),
        "--measurement-truth-manifest", str(paths["measure_manifest"]),
        "--target-recalls", "0.90", "--readers", "1", "--update-rates", "0",
        "--methods", "stock", "--requests", "2", "--measurement-repeats", "6",
        "--audit-spots", "1", "--update-id-pool-size", "2", "--out", str(out),
    ])
    filter_spec = runner.throughput.FilterSpec("f", "price > 0", ("price > 0",), 10, 1.0)
    config = runner.throughput.SearchConfig(100, 1000, 2.0, "off", 11)
    matched = runner.throughput.MatchedRecallBundle(
        {("f", "stock", 0.9): config},
        (),
        {},
        {"run_spec": {"args": {
            "calibration_query_offset": 0, "calibration_queries": 1,
            "final_query_offset": 1, "final_queries": 1,
        }}},
        "safe_guided",
    )
    calibration_ids = {0: 1, 1: 2}
    measurement_ids = {100: 10, 101: 11}
    workload = runner.throughput.Workload(
        (
            runner.throughput.WorkloadRequest(0, 100, 10, 0),
            runner.throughput.WorkloadRequest(1, 101, 11, 0),
        ),
        "q10200_cohort_measurement_split", str(paths["query"]), "a" * 64,
        "q200..q10199", False, 2,
    )

    def prepare(runtime_args, _filters):
        runtime_args.runtime_sqlens_identity_evidence = []
        runtime_args.backend_cpu_evidence = []
        runtime_args.fragment_tracking_evidence = {"prepared": True}

    def overlap(runtime_args, method, _config, spec, _truth, requests, readers, rate, repeat, *_rest):
        rows = [{
            "kind": "read", "measurement_repeat": repeat, "method": method,
            "filter_name": spec.name, "readers": readers,
            "writer_clients": runtime_args.writer_clients, "update_rate_tps": rate,
            "client_id": 0, "request_no": request.request_no,
            "dispatch_position": position, "query_no": request.query_no,
            "query_id": request.query_id, "latency_ms": 1.0,
            "profile_collection_ms": 0.1, "error": "", "error_type": "",
            "timeout": False, "returned": 10, "recall_at_10": 1.0,
            "profile_complete": True, "path_class": "stock",
            "final_path": "stock", "stale_relation": False,
        } for position, request in enumerate(requests)]
        return rows, 1.0, {"workers": [], "measurement_watchdog_timeout": False}

    audit = [{"passed": True}]
    with (
        mock.patch.object(runner.throughput, "load_filters", return_value=[filter_spec]),
        mock.patch.object(runner.throughput, "load_audited_matched_recall_configs", return_value=matched),
        mock.patch.object(runner.throughput, "bind_matched_recall_provenance"),
        mock.patch.object(runner.throughput, "verify_truth_manifest", return_value={"artifact_valid": True}),
        mock.patch.object(runner.throughput, "verify_measurement_query_manifest", return_value={"artifact_valid": True}),
        mock.patch.object(runner.throughput, "verify_measurement_truth_manifest", return_value={"artifact_valid": True}),
        mock.patch.object(runner.throughput, "load_truth", side_effect=[({}, calibration_ids), ({}, measurement_ids)]),
        mock.patch.object(runner.throughput, "load_true_query_workload", return_value=workload),
        mock.patch.object(runner.throughput, "validate_workload_query_mapping"),
        mock.patch.object(runner.throughput, "validate_truth_coverage"),
        mock.patch.object(runner, "prepare_runtime_args", side_effect=prepare),
        mock.patch.object(runner, "validate_update_column", return_value={"mode": "rewrite_same_value"}),
        mock.patch.object(runner, "live_identity_gate", return_value={"passed": True}),
        mock.patch.object(runner, "load_update_id_pool", return_value=[41, 42]),
        mock.patch.object(runner.throughput, "warm_database_cache", return_value={"passed": True}),
        mock.patch.object(runner, "exact_sql_valid_spot_audit", return_value=audit),
        mock.patch.object(runner, "run_overlap", side_effect=overlap),
    ):
        assert runner.run_experiment(args) == 0

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["diagnostic_valid"] is True
    assert manifest["artifact_valid"] is False
    assert manifest["paper_eligible"] is False
    assert manifest["formal_artifact_valid"] is False
    assert manifest["protocol"]["label"] == "nonformal_debug"
    assert manifest["status"] == "diagnostic_complete"
    assert manifest["query_id_split_gate"]["passed"] is True
    assert manifest["query_cohort"]["queries"] == 2
    assert manifest["completion"]["observed_read_rows"] == 12
