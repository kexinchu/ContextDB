from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import run_figure5_matched_latency as runner


BUILD = "sqlens-v16-test-r35"
VECTOR_SHA = "a" * 64


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_file(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def pair_row(status: str = "selected") -> dict[str, str]:
    row = {
        "pair_id": "amazon:recall_0.900", "dataset": "amazon", "target_recall": "0.9",
        "selection_status": status,
        "qualification_scope": runner.QUALIFICATION_SCOPE_FORMAL,
    }
    values = {
        "stock": ("stock-ef100", "b" * 64, "100", "strict_order", "5000000", "32", "100", "40", "off", "off", "public.items", "public.source_idx"),
        "sqlens": ("sqlens-ef250", "c" * 64, "250", "off", "200000", "16", "250", "80", "off", "off", "public.items", "public.bfs_idx"),
    }
    fields = ("config_id", "config_sha256", "ef_search", "iterative_scan", "max_scan_tuples", "scan_mem_multiplier", "guided_collect_target", "traversal_guided_target", "d2_page_access", "d2_index_page_access", "table", "index")
    for arm, values_for_arm in values.items():
        row.update({f"{arm}_{field}": value for field, value in zip(fields, values_for_arm)})
    return row


def config(root: Path) -> dict[str, object]:
    files = {
        "truth_csv": write_file(root / "truth.csv", "truth\n"),
        "measurement_workload_csv": root / "workload.csv",
        "filters_csv": write_file(root / "filters.csv", "filter\n"),
        "d2_graph_proof_json": write_file(root / "proof.json", "{}\n"),
    }
    with files["measurement_workload_csv"].open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["request_no", "query_no", "query_id", "filter_name", "trace_cycle", "split"],
        )
        writer.writeheader()
        for request_no in range(runner.EXPECTED_REQUESTS):
            writer.writerow(
                {
                    "request_no": request_no,
                    "query_no": request_no,
                    "query_id": 100_000 + request_no,
                    "filter_name": "filter_a" if request_no % 2 == 0 else "filter_b",
                    "trace_cycle": request_no,
                    "split": "measurement",
                }
            )
    return {
        "protocol": {"schedule_seed": 7, "guidance_filter_strategy": "traversal_guided", "d3_measurement_policy": "workload_driven_adaptive", "guidance_max_atoms": 160},
        "release_identity": {"expected_sqlens_build_id": BUILD, "expected_vector_so_sha256": VECTOR_SHA},
        "release_contract_sha256": "d" * 64,
        "datasets": {
            "amazon": {
                "label": "Amazon", "table": "public.items", "query_table": "public.items",
                "query_id_column": "id", "query_vector_column": "embedding",
                "source_index": "public.source_idx", "bfs_index": "public.bfs_idx",
                "candidate_validity_predicate": "embedding_valid", "truth_self_excluded": True,
                **{name: str(path) for name, path in files.items()},
            }
        },
    }


def selection_artifacts(root: Path, row: dict[str, str], cfg: dict[str, object]) -> tuple[Path, Path, Path]:
    config_path = root / "frontier-config.json"
    config_path.write_text(json.dumps(cfg, sort_keys=True), encoding="utf-8")
    required_grid = root / "required-grid.json"
    required_grid.write_text(
        json.dumps(
            {
                "contract_type": runner.REQUIRED_GRID_CONTRACT_TYPE,
                "grid_complete": True,
                "qualification_scope": row["qualification_scope"],
                "dataset_config": {
                    "path": str(config_path),
                    "sha256": sha(config_path),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    grid_binding = {
        "path": str(required_grid),
        "sha256": sha(required_grid),
    }
    csv_path = root / "selection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    plan_path = root / "selection.json"
    plan = {
        "artifact_valid": True,
        "qualification_scope": row["qualification_scope"],
        "execution_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha(Path(__file__).resolve()),
        },
        "measurement_plan_csv": {"sha256": sha(csv_path)},
        "release_contract": {"sha256": "d" * 64, **cfg["release_identity"]},
        "required_grid_contract": grid_binding,
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    manifest_path = root / "selection.manifest.json"
    manifest = {
        "artifact_valid": True,
        "qualification_scope": row["qualification_scope"],
        "release_contract": {"sha256": "d" * 64, **cfg["release_identity"]},
        "required_grid_contract": grid_binding,
        "outputs": {
            "measurement_plan_csv": {"sha256": sha(csv_path)},
            "measurement_plan_json": {"sha256": sha(plan_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return csv_path, plan_path, manifest_path


def tiny_provenance(tmp_path: Path, pair: runner.SelectedPair) -> dict[str, object]:
    workload = tmp_path / "tiny-workload.csv"
    with workload.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["request_no", "query_no", "query_id", "filter_name"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"request_no": 0, "query_no": 10, "query_id": 1010, "filter_name": "filter_a"},
                {"request_no": 1, "query_no": 11, "query_id": 1011, "filter_name": "filter_b"},
            ]
        )
    mode_configs = {
        "original": {
            "ef_search": 100,
            "max_scan_tuples": 5_000_000,
            "scan_mem_multiplier": 32.0,
            "iterative_scan": "strict_order",
            "guided_collect_target": 100,
            "traversal_guided_target": 100,
            "traversal_guided_prioritization": False,
            "traversal_guided_burst": 8,
        },
        "design1_bloom_bfs_layout_d3": {
            "ef_search": 250,
            "max_scan_tuples": 200_000,
            "scan_mem_multiplier": 16.0,
            "iterative_scan": "off",
            "guided_collect_target": 250,
            "traversal_guided_target": 250,
            "traversal_guided_prioritization": True,
            "traversal_guided_burst": 8,
        },
    }
    return {
        "schedule_seed": 7,
        "input_bindings": {
            "measurement_workload_csv": {
                "path": str(workload),
                "sha256": sha(workload),
            }
        },
        "mode_configs": mode_configs,
        "d3_fragment_store_namespace": runner.pair_namespace(pair),
        "d3_repeat_namespaces": runner.pair_repeat_namespaces(pair),
    }


def tiny_raw_rows(
    pair: runner.SelectedPair,
    provenance: dict[str, object],
    *,
    requests: int = 2,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    identities = {
        0: ("1010", "10", "filter_a"),
        1: ("1011", "11", "filter_b"),
    }
    for repeat in range(runner.EXPECTED_REPEATS):
        query_positions = runner.expected_query_positions(
            repeat, 7, requests=requests
        )
        for request_no in range(requests):
            query_position = query_positions[request_no]
            block_no = repeat * requests + query_position - 1
            for mode, arm in (("original", pair.stock), ("design1_bloom_bfs_layout_d3", pair.sqlens)):
                row = {
                    "mode": mode,
                    "repeat": str(repeat),
                    "request_no": str(request_no),
                    "query_id": identities[request_no][0],
                    "query_no": identities[request_no][1],
                    "filter_name": identities[request_no][2],
                    "block_no": str(block_no),
                    "query_order_position": str(query_position),
                    "schedule_position": str(runner.expected_schedule_position(block_no, 7, mode)),
                    "execution_order": "interleaved",
                    "schedule_seed": "7",
                    "traversal_prioritization_burst": "0" if mode == "original" else "8",
                    "error": "",
                    "recall": "0.95",
                    "d3_fragment_store_namespace": runner.pair_repeat_namespaces(pair)[repeat],
                }
                for field in (
                    "ef_search", "iterative_scan", "max_scan_tuples", "scan_mem_multiplier",
                    "guided_collect_target", "traversal_guided_target", "d2_page_access", "d2_index_page_access",
                ):
                    row[field] = str(arm[field])
                rows.append(row)
    return rows


def test_builds_independent_stock_and_sqlens_search_configs(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    pair = runner.load_selected_pairs(
        selection_artifacts(tmp_path, pair_row(), cfg)[0], cfg, datasets=(), pair_ids=()
    )[0]
    command, provenance = runner.build_pair_command(cfg, pair, tmp_path / "raw.csv", "48-63")
    mode_configs = json.loads(command[command.index("--mode-configs-json") + 1])
    assert mode_configs["original"]["ef_search"] == 100
    assert mode_configs["original"]["iterative_scan"] == "strict_order"
    assert mode_configs["design1_bloom_bfs_layout_d3"]["ef_search"] == 250
    assert mode_configs["design1_bloom_bfs_layout_d3"]["iterative_scan"] == "off"
    assert "d2_page_access" not in mode_configs["original"]
    assert "d2_index_page_access" not in mode_configs["original"]
    assert "d2_page_access" not in mode_configs["design1_bloom_bfs_layout_d3"]
    assert "d2_index_page_access" not in mode_configs["design1_bloom_bfs_layout_d3"]
    assert provenance["expected_rows"] == 60_000


def test_selection_validation_binds_formal_qualification_scope(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    csv_path, plan_path, manifest_path = selection_artifacts(tmp_path, pair_row(), cfg)

    bindings = runner.validate_selection_artifacts(
        csv_path,
        plan_path,
        manifest_path,
        cfg,
        config_path=tmp_path / "frontier-config.json",
        required_grid_contract=tmp_path / "required-grid.json",
    )

    assert bindings["qualification_scope"] == runner.QUALIFICATION_SCOPE_FORMAL
    assert (
        bindings["required_grid_contract"]["dataset_config_sha256"]
        == sha(tmp_path / "frontier-config.json")
    )


def test_selection_validation_rejects_required_grid_or_config_sha_drift(
    tmp_path: Path,
) -> None:
    cfg = config(tmp_path)
    csv_path, plan_path, manifest_path = selection_artifacts(
        tmp_path, pair_row(), cfg
    )
    config_path = tmp_path / "frontier-config.json"
    required_grid = tmp_path / "required-grid.json"
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        runner.MatchedLatencyError,
        match="required-grid, and active --config SHA",
    ):
        runner.validate_selection_artifacts(
            csv_path,
            plan_path,
            manifest_path,
            cfg,
            config_path=config_path,
            required_grid_contract=required_grid,
        )


def test_final_latency_run_requires_explicit_required_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "frontier.json"
    config_path.write_text("{}\n", encoding="utf-8")
    cfg = config(tmp_path)
    monkeypatch.setattr(runner, "load_config", lambda path: cfg)
    args = runner.create_parser().parse_args(["--config", str(config_path)])
    assert args.required_grid_contract is None
    with pytest.raises(
        runner.MatchedLatencyError,
        match="--required-grid-contract is mandatory",
    ):
        runner.run(args)


def test_unattainable_rows_are_skipped_and_requested_pair_fails_closed(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    selected = pair_row()
    unavailable = pair_row("unattainable_on_calibration_grid")
    unavailable["pair_id"] = "amazon:recall_0.990"
    csv_path = tmp_path / "selection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(selected))
        writer.writeheader()
        writer.writerows([selected, unavailable])
    pairs = runner.load_selected_pairs(csv_path, cfg, datasets=(), pair_ids=())
    assert [pair.pair_id for pair in pairs] == ["amazon:recall_0.900"]
    with pytest.raises(runner.MatchedLatencyError, match="absent or not selected"):
        runner.load_selected_pairs(csv_path, cfg, datasets=(), pair_ids=("amazon:recall_0.990",))


def test_full_release_scope_distinguishes_requested_slice() -> None:
    selected = [
        runner.SelectedPair("amazon:recall_0.900", "amazon", 0.9, {}, {}),
        runner.SelectedPair("yfcc:recall_0.900", "yfcc", 0.9, {}, {}),
    ]
    full_args = runner.create_parser().parse_args([])
    scope = runner.full_release_scope(full_args, selected, selected)
    assert scope["requested"] is True
    assert all(scope["checks"].values())

    slice_args = runner.create_parser().parse_args(
        ["--datasets", "amazon", "--backend-cpu-list", "32-47"]
    )
    scope = runner.full_release_scope(slice_args, selected[:1], selected)
    assert scope["requested"] is False
    assert scope["checks"]["all_datasets_requested"] is False
    assert scope["checks"]["all_selected_pairs_requested"] is False
    assert scope["checks"]["default_backend_cpu_partition"] is False


def test_full_release_scope_rejects_partial_or_sparse_selector() -> None:
    args = runner.create_parser().parse_args([])
    sparse = [
        runner.SelectedPair(
            f"amazon:recall_{index}",
            "amazon",
            0.9,
            {"config_sha256": f"s{index}"},
            {"config_sha256": f"q{index}"},
        )
        for index in range(10)
    ]
    scope = runner.full_release_scope(
        args,
        sparse,
        sparse,
        selection_bindings={
            "target_policy": "distinct_pairs",
            "min_distinct_pairs_per_dataset": 10,
            "target_floor": 0.70,
            "qualification_scope": runner.QUALIFICATION_SCOPE_FORMAL,
        },
        enforce_frozen_selector=True,
    )

    assert scope["requested"] is False
    assert scope["checks"]["selector_covers_frozen_datasets"] is False


def test_full_release_scope_accepts_dense_three_dataset_selector() -> None:
    args = runner.create_parser().parse_args([])
    pairs = [
        runner.SelectedPair(
            f"{dataset}:recall_{index}",
            dataset,
            0.8 + index / 100.0,
            {"config_sha256": f"{dataset}-stock-{index}"},
            {"config_sha256": f"{dataset}-sqlens-{index}"},
        )
        for dataset in runner.FROZEN_DATASETS
        for index in range(runner.MIN_FORMAL_POINTS_PER_ARM_DATASET)
    ]
    scope = runner.full_release_scope(
        args,
        pairs,
        pairs,
        selection_bindings={
            "target_policy": "distinct_pairs",
            "min_distinct_pairs_per_dataset": 10,
            "target_floor": 0.70,
            "qualification_scope": runner.QUALIFICATION_SCOPE_FORMAL,
        },
        enforce_frozen_selector=True,
    )

    assert scope["requested"] is True
    assert all(scope["checks"].values())


def test_full_release_scope_accepts_fixed_targets_with_unattainable_cells() -> None:
    args = runner.create_parser().parse_args([])
    pairs = [
        runner.SelectedPair(
            f"{dataset}:recall_{target}",
            dataset,
            target,
            {"config_sha256": f"{dataset}-stock-{target}"},
            {"config_sha256": f"{dataset}-sqlens-{target}"},
        )
        for dataset in runner.FROZEN_DATASETS
        for target in (0.90, 0.95, 0.99)
        if not (dataset == "laion" and target in (0.95, 0.99))
    ]
    scope = runner.full_release_scope(
        args,
        pairs,
        pairs,
        selection_bindings={
            "target_policy": "fixed",
            "qualification_scope": runner.QUALIFICATION_SCOPE_FORMAL,
            "targets_by_dataset": {
                dataset: [0.90, 0.95, 0.99]
                for dataset in runner.FROZEN_DATASETS
            },
            "target_rows": 9,
            "selected_pairs": 7,
            "unattainable_pairs": 2,
        },
        enforce_frozen_selector=True,
    )

    assert scope["kind"] == "matched_targets"
    assert scope["requested"] is True
    assert all(scope["checks"].values())


def test_full_release_scope_rejects_incomplete_fixed_target_contract() -> None:
    args = runner.create_parser().parse_args([])
    pairs = [
        runner.SelectedPair(
            f"{dataset}:recall_0.9",
            dataset,
            0.9,
            {"config_sha256": f"{dataset}-stock"},
            {"config_sha256": f"{dataset}-sqlens"},
        )
        for dataset in runner.FROZEN_DATASETS
    ]
    scope = runner.full_release_scope(
        args,
        pairs,
        pairs,
        selection_bindings={
            "target_policy": "fixed",
            "qualification_scope": runner.QUALIFICATION_SCOPE_FORMAL,
            "targets_by_dataset": {
                dataset: [0.90]
                for dataset in runner.FROZEN_DATASETS
            },
            "target_rows": 3,
            "selected_pairs": 3,
            "unattainable_pairs": 0,
        },
        enforce_frozen_selector=True,
    )

    assert scope["kind"] == "matched_targets"
    assert scope["requested"] is False
    assert scope["checks"]["selector_uses_formal_fixed_targets"] is False


def test_error_rows_and_nonempty_namespace_fail_the_completion_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config(tmp_path)
    pair = runner.load_selected_pairs(
        selection_artifacts(tmp_path, pair_row(), cfg)[0], cfg, datasets=(), pair_ids=()
    )[0]
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["mode", "error"])
        writer.writeheader()
        writer.writerow({"mode": "original", "error": "RuntimeError"})
    plan = raw.with_suffix(".csv.plan.json")
    plan.write_text(json.dumps({"status": "complete", "output_rows": 60_000, "output_sha256": sha(raw)}), encoding="utf-8")
    assert not runner.cell_complete(raw, plan, pair, cfg, {"input_bindings": {}})

    monkeypatch.setattr(runner, "namespace_rows", lambda table, namespace: 3)
    with pytest.raises(runner.MatchedLatencyError, match="not fresh"):
        runner.ensure_fresh_namespace("public.items", "freshness-test", overwrite=False)


def test_raw_rows_validate_every_repeat_workload_identity_and_interleaving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 2)
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 12)
    cfg = config(tmp_path)
    pair = runner.load_selected_pairs(
        selection_artifacts(tmp_path, pair_row(), cfg)[0], cfg, datasets=(), pair_ids=()
    )[0]
    provenance = tiny_provenance(tmp_path, pair)
    rows = tiny_raw_rows(pair, provenance)
    assert runner.raw_rows_match_pair(rows, pair, provenance)

    identity_drift = [dict(row) for row in rows]
    identity_drift[0]["query_id"] = "9999"
    assert not runner.raw_rows_match_pair(identity_drift, pair, provenance)

    repeat_schedule_drift = [dict(row) for row in rows]
    repeat_schedule_drift[2]["schedule_position"] = "2" if repeat_schedule_drift[2]["schedule_position"] == "1" else "1"
    assert not runner.raw_rows_match_pair(repeat_schedule_drift, pair, provenance)

    repeat_query_order_drift = [dict(row) for row in rows]
    repeat_query_order_drift[4]["query_order_position"] = (
        "2"
        if repeat_query_order_drift[4]["query_order_position"] == "1"
        else "1"
    )
    assert not runner.raw_rows_match_pair(
        repeat_query_order_drift, pair, provenance
    )

    repeat_request_gap = [row for row in rows if not (row["repeat"] == "1" and row["request_no"] == "1")]
    assert not runner.raw_rows_match_pair(repeat_request_gap, pair, provenance)


def test_matched_recall_gate_is_per_repeat_and_per_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 2)
    monkeypatch.setattr(runner, "MIN_FORMAL_PREDICATE_SAMPLES", 1)
    monkeypatch.setattr(runner, "EXPECTED_FORMAL_PREDICATES", 2)
    cfg = config(tmp_path)
    pair = runner.load_selected_pairs(
        selection_artifacts(tmp_path, pair_row(), cfg)[0], cfg, datasets=(), pair_ids=()
    )[0]
    provenance = tiny_provenance(tmp_path, pair)
    rows = tiny_raw_rows(pair, provenance)
    assert runner.matched_recall_gate(
        rows, pair, runner.QUALIFICATION_SCOPE_FORMAL
    )["passed"] is True

    failing = [dict(row) for row in rows]
    for row in failing:
        if row["mode"] == "design1_bloom_bfs_layout_d3" and row["repeat"] == "2":
            row["recall"] = "0.50"
    gate = runner.matched_recall_gate(
        failing, pair, runner.QUALIFICATION_SCOPE_FORMAL
    )
    assert gate["passed"] is False
    assert gate["aggregate"]["design1_bloom_bfs_layout_d3/repeat=2"]["passed"] is False


def test_latency_completion_gate_requires_exactly_fourteen_predicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 2)
    monkeypatch.setattr(runner, "MIN_FORMAL_PREDICATE_SAMPLES", 1)
    cfg = config(tmp_path)
    pair = runner.load_selected_pairs(
        selection_artifacts(tmp_path, pair_row(), cfg)[0],
        cfg,
        datasets=(),
        pair_ids=(),
    )[0]

    gate = runner.matched_recall_gate(
        tiny_raw_rows(pair, tiny_provenance(tmp_path, pair)),
        pair,
        runner.QUALIFICATION_SCOPE_FORMAL,
    )

    assert gate["passed"] is False
    assert gate["expected_predicate_count"] == 14
    assert gate["observed_predicate_count"] == 2
    assert gate["filter_names"] == ["filter_a", "filter_b"]
    assert runner.predicate_completion_contract(gate) == {
        "expected_predicate_count": 14,
        "observed_predicate_count": 2,
        "predicate_names": ["filter_a", "filter_b"],
        "exact_coverage": False,
    }


def test_formal_quality_gate_rejects_one_predicate_even_when_aggregate_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 20)
    monkeypatch.setattr(runner, "MIN_FORMAL_PREDICATE_SAMPLES", 1)
    monkeypatch.setattr(runner, "EXPECTED_FORMAL_PREDICATES", 2)
    cfg = config(tmp_path)
    pair = runner.load_selected_pairs(
        selection_artifacts(tmp_path, pair_row(), cfg)[0], cfg, datasets=(), pair_ids=()
    )[0]
    pair = runner.SelectedPair(
        pair.pair_id, pair.dataset, 0.10, pair.stock, pair.sqlens
    )
    rows = tiny_raw_rows(pair, tiny_provenance(tmp_path, pair), requests=2)
    extras: list[dict[str, str]] = []
    for row in rows:
        if row["filter_name"] != "filter_a":
            continue
        for request_no in range(2, 20):
            extra = dict(row)
            extra["request_no"] = str(request_no)
            extra["query_id"] = f"extra-{request_no}"
            extras.append(extra)
    rows.extend(extras)
    for row in rows:
        if row["mode"] == "design1_bloom_bfs_layout_d3" and row["filter_name"] == "filter_b":
            row["recall"] = "0.00"

    gate = runner.matched_recall_gate(
        rows, pair, runner.QUALIFICATION_SCOPE_FORMAL
    )

    assert all(item["passed"] for item in gate["aggregate"].values())
    assert gate["passed"] is False
    assert gate["per_predicate"]["design1_bloom_bfs_layout_d3/repeat=0"]["filter_b"]["passed"] is False
    assert gate["worst_predicate"]["filter_name"] == "filter_b"


def test_formal_quality_gate_fails_closed_on_predicate_sample_shortage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 2)
    monkeypatch.setattr(runner, "EXPECTED_FORMAL_PREDICATES", 2)
    cfg = config(tmp_path)
    pair = runner.load_selected_pairs(
        selection_artifacts(tmp_path, pair_row(), cfg)[0], cfg, datasets=(), pair_ids=()
    )[0]
    rows = tiny_raw_rows(pair, tiny_provenance(tmp_path, pair))

    gate = runner.matched_recall_gate(
        rows, pair, runner.QUALIFICATION_SCOPE_FORMAL
    )

    assert gate["passed"] is False
    assert "sample-count" in str(gate["reason"])
    assert not gate["per_predicate"]["original/repeat=0"]["filter_a"]["sample_count_sufficient"]


def test_aggregate_scope_is_audit_only_even_when_aggregate_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 2)
    monkeypatch.setattr(runner, "EXPECTED_FORMAL_PREDICATES", 2)
    cfg = config(tmp_path)
    row = pair_row()
    row["qualification_scope"] = runner.QUALIFICATION_SCOPE_AGGREGATE
    pair = runner.load_selected_pairs(
        selection_artifacts(tmp_path, row, cfg)[0], cfg, datasets=(), pair_ids=()
    )[0]
    gate = runner.matched_recall_gate(
        tiny_raw_rows(pair, tiny_provenance(tmp_path, pair)),
        pair,
        runner.QUALIFICATION_SCOPE_AGGREGATE,
    )

    assert gate["passed"] is True
    assert gate["paper_eligible"] is False

    legacy_scope = runner.full_release_scope(
        runner.create_parser().parse_args([]),
        [pair],
        [pair],
        selection_bindings={
            "target_policy": "fixed",
            "qualification_scope": runner.QUALIFICATION_SCOPE_AGGREGATE,
            "targets_by_dataset": {
                dataset: [0.90, 0.95, 0.99]
                for dataset in runner.FROZEN_DATASETS
            },
            "target_rows": 9,
            "selected_pairs": 1,
            "unattainable_pairs": 8,
        },
        enforce_frozen_selector=True,
    )
    assert legacy_scope["checks"]["selector_uses_formal_predicate_qualification"] is False
    assert legacy_scope["requested"] is False


def test_explicit_target_override_uses_aggregate_lcb_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 20)
    monkeypatch.setattr(runner, "MIN_FORMAL_PREDICATE_SAMPLES", 1)
    monkeypatch.setattr(runner, "EXPECTED_FORMAL_PREDICATES", 2)
    cfg = config(tmp_path)
    pair = runner.load_selected_pairs(
        selection_artifacts(tmp_path, pair_row(), cfg)[0],
        cfg,
        datasets=(),
        pair_ids=(),
    )[0]
    pair = runner.SelectedPair(
        pair.pair_id,
        pair.dataset,
        0.10,
        pair.stock,
        pair.sqlens,
        "aggregate_lcb",
    )
    rows = tiny_raw_rows(pair, tiny_provenance(tmp_path, pair), requests=2)
    for row in list(rows):
        if row["filter_name"] != "filter_a":
            continue
        for request_no in range(2, 20):
            extra = dict(row)
            extra["request_no"] = str(request_no)
            extra["query_id"] = f"override-extra-{request_no}"
            rows.append(extra)
    for row in rows:
        if (
            row["mode"] == "design1_bloom_bfs_layout_d3"
            and row["filter_name"] == "filter_b"
        ):
            row["recall"] = "0.00"

    gate = runner.matched_recall_gate(
        rows, pair, runner.QUALIFICATION_SCOPE_FORMAL
    )

    assert all(item["passed"] for item in gate["aggregate"].values())
    assert gate["passed"] is True
    assert gate["quality_gate_override"] == "aggregate_lcb"
    assert gate["paper_eligible"] is False


def test_quality_gate_plan_binding_rejects_stale_or_missing_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 2)
    monkeypatch.setattr(runner, "MIN_FORMAL_PREDICATE_SAMPLES", 1)
    monkeypatch.setattr(runner, "EXPECTED_FORMAL_PREDICATES", 2)
    cfg = config(tmp_path)
    pair = runner.load_selected_pairs(
        selection_artifacts(tmp_path, pair_row(), cfg)[0], cfg, datasets=(), pair_ids=()
    )[0]
    gate = runner.matched_recall_gate(
        tiny_raw_rows(pair, tiny_provenance(tmp_path, pair)),
        pair,
        runner.QUALIFICATION_SCOPE_FORMAL,
    )
    plan_path = tmp_path / "cell.plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    assert not runner.quality_gate_matches_plan({}, gate)
    runner.write_quality_gate_to_plan(plan_path, gate)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert runner.quality_gate_matches_plan(plan, gate)
    assert plan["matched_latency_predicate_completion"] == {
        "expected_predicate_count": 2,
        "observed_predicate_count": 2,
        "predicate_names": ["filter_a", "filter_b"],
        "exact_coverage": True,
    }
    plan["matched_latency_quality_gate"]["passed"] = False
    assert not runner.quality_gate_matches_plan(plan, gate)


def test_effective_mode_config_comparison_rejects_traversal_field_drift(
    tmp_path: Path,
) -> None:
    cfg = config(tmp_path)
    pair = runner.load_selected_pairs(
        selection_artifacts(tmp_path, pair_row(), cfg)[0], cfg, datasets=(), pair_ids=()
    )[0]
    provenance = tiny_provenance(tmp_path, pair)
    checks = [
        {"mode": mode, "config": config_value}
        for mode, config_value in provenance["mode_configs"].items()
    ]
    assert runner.effective_mode_config_complete({"checks": checks}, provenance)
    drifted = json.loads(json.dumps(checks))
    drifted[1]["config"]["traversal_guided_prioritization"] = False
    assert not runner.effective_mode_config_complete({"checks": drifted}, provenance)


def test_subset_cannot_overwrite_full_release_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "sqlens_figure5_matched_latency_run",
                "runner_version": runner.RUNNER_VERSION,
                "protocol_fingerprint_sha256": "a" * 64,
                "full_release_complete": True,
                "paper_eligible": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(runner.MatchedLatencyError, match="subset run cannot overwrite"):
        runner.validate_existing_run_manifest(
            manifest,
            "b" * 64,
            {"requested": False},
            resume=True,
            overwrite=True,
        )


def test_execution_source_compatibility_is_limited_to_validator_only_sha() -> None:
    current = {
        "core_runner": {"path": "/core.py", "sha256": "1" * 64},
        "orchestrator": {"path": "/runner.py", "sha256": "2" * 64},
    }
    assert runner.execution_sources_compatible(current, current)

    validator_only = json.loads(json.dumps(current))
    validator_only["orchestrator"]["sha256"] = next(
        iter(runner.VALIDATOR_ONLY_COMPATIBLE_ORCHESTRATOR_SHA256)
    )
    assert runner.execution_sources_compatible(validator_only, current)

    wrong_core = json.loads(json.dumps(validator_only))
    wrong_core["core_runner"]["sha256"] = "3" * 64
    assert not runner.execution_sources_compatible(wrong_core, current)

    unknown_orchestrator = json.loads(json.dumps(current))
    unknown_orchestrator["orchestrator"]["sha256"] = "4" * 64
    assert not runner.execution_sources_compatible(unknown_orchestrator, current)
