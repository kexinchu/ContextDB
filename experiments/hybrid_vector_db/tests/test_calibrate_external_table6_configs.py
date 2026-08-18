from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pytest

from experiments.hybrid_vector_db.scripts import calibrate_external_table6_configs as calibrator
from experiments.hybrid_vector_db.scripts import pgvector_d1_stock_increment_control as d1_control
from experiments.hybrid_vector_db.scripts import pgvector_d2_cache_isolation_control as d2_control


BUILD_ID = "sqlens-v16-test-build"
VECTOR_SHA = "a" * 64


def candidate(
    config: calibrator.SearchConfig,
    *,
    filter_name: str = "f",
    mode: str = "original",
    recall_lcb95: float = 0.91,
    recall_mean: float | None = None,
    latency_ms: float = 10.0,
    reused: bool = False,
) -> calibrator.CandidateResult:
    return calibrator.CandidateResult(
        filter_name=filter_name,
        mode=mode,
        config=config,
        recall_mean=(
            max(recall_lcb95, 0.92) if recall_mean is None else recall_mean
        ),
        recall_lcb95=recall_lcb95,
        recall_ci95_low=recall_lcb95 - 0.01,
        recall_ci95_high=min(1.0, recall_lcb95 + 0.01),
        latency_mean_ms=latency_ms,
        latency_p50_ms=latency_ms,
        queries=2,
        samples=2,
        raw_path="child.csv",
        raw_sha256="1" * 64,
        plan_path="child.csv.plan.json",
        plan_sha256="2" * 64,
        table_summary_path="child_table.csv",
        table_summary_sha256="3" * 64,
        profile_summary_path="child_profile_summary.csv",
        profile_summary_sha256="4" * 64,
        command_sha256="5" * 64,
        child_reused=reused,
        relation_provenance={
            "expected_table": "public.items",
            "expected_table_oid": 10,
            "expected_index": "public.items_hnsw",
            "expected_index_oid": 11,
        },
        binary_provenance={
            "observed_build_id": BUILD_ID,
            "observed_vector_so_sha256": VECTOR_SHA,
            "exact_match": True,
        },
    )


def config(ef_search: int, iterative_scan: str = "off") -> calibrator.SearchConfig:
    return calibrator.SearchConfig(
        ef_search=ef_search,
        iterative_scan=iterative_scan,
        max_scan_tuples=5_000_000,
        scan_mem_multiplier=32.0,
        guided_collect_target=ef_search,
        guided_collect_target_tracks_ef=True,
        traversal_guided_target=min(40, ef_search),
        traversal_guided_prioritization=False,
        traversal_guided_burst=8,
    )


def write_filters(path: Path, atom_count: int = 1) -> None:
    atoms = "||OR||".join(f"sql:labels @> ARRAY[{number}]::int[]" for number in range(atom_count))
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["filter_name", "target_rate", "actual_pct", "predicate", "atoms"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filter_name": "f",
                "target_rate": "50",
                "actual_pct": "49.7",
                "predicate": "labels && ARRAY[1]::int[]",
                "atoms": atoms,
            }
        )


def write_truth(path: Path, query_count: int = 2, calibration_count: int | None = None) -> None:
    calibration_count = query_count if calibration_count is None else calibration_count
    fields = [
        "filter_name",
        "query_no",
        "query_id",
        "query_split",
        "method",
        "self_excluded",
        "candidate_validity_predicate",
    ]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for query_no in range(query_count):
            writer.writerow(
                {
                    "filter_name": "f",
                    "query_no": query_no,
                    "query_id": 100 + query_no,
                    "query_split": "calibration" if query_no < calibration_count else "final",
                    "method": "pre_filter_exact",
                    "self_excluded": "False",
                    "candidate_validity_predicate": "TRUE",
                }
            )


def parser_args(tmp_path: Path, filters: Path, truth: Path, *extra: str) -> Namespace:
    truth_manifest = tmp_path / "truth.manifest.json"
    if not truth_manifest.exists():
        truth_manifest.write_text('{"status":"complete"}\n', encoding="utf-8")
    return calibrator.create_argument_parser().parse_args(
        [
            "--out",
            str(tmp_path / "external_calibration_configs.csv"),
            "--filters-csv",
            str(filters),
            "--truth-csv",
            str(truth),
            "--insertion-table",
            "public.items",
            "--insertion-index",
            "public.items_hnsw",
            "--bfs-index",
            "public.items_hnsw_bfs",
            "--query-table",
            "public.queries",
            "--query-id-column",
            "qid",
            "--query-vector-column",
            "embedding",
            "--expected-sqlens-build-id",
            BUILD_ID,
            "--expected-vector-so-sha256",
            VECTOR_SHA,
            "--backend-cpu-list",
            "48-63",
            "--truth-provenance-manifest",
            str(truth_manifest),
            *extra,
        ]
    )


def test_laion_70_leaf_or_requires_runtime_token_ceiling_256(tmp_path: Path) -> None:
    filters = tmp_path / "filters.csv"
    truth = tmp_path / "truth.csv"
    write_filters(filters, atom_count=70)
    write_truth(truth)
    default_args = parser_args(tmp_path, filters, truth)
    assert default_args.selection_coupling == "independent"
    assert default_args.shared_latency_objective == "max"
    with pytest.raises(ValueError, match="exceeding --guidance-max-atoms=128"):
        calibrator.load_filters(filters, None, default_args.guidance_max_atoms)

    args = parser_args(tmp_path, filters, truth, "--guidance-max-atoms", "256")

    loaded = calibrator.load_filters(filters, None, args.guidance_max_atoms)
    query_nos, query_ids = calibrator.load_calibration_split(
        truth, loaded, 0, 2, args.expected_truth_self_excluded, "TRUE"
    )

    assert args.guidance_max_atoms == 256
    assert args.expected_truth_self_excluded is False
    assert loaded[0].atom_count == 139
    assert query_nos == [0, 1]
    assert query_ids == [100, 101]


def test_filter_atom_limit_fails_closed(tmp_path: Path) -> None:
    filters = tmp_path / "filters.csv"
    write_filters(filters, atom_count=65)
    with pytest.raises(ValueError, match="exceeding --guidance-max-atoms=128"):
        calibrator.load_filters(filters, None, 128)


def test_seed_exponential_bracket_then_local_lower_ef() -> None:
    measured: list[int] = []
    recalls = {80: 0.85, 120: 0.91, 160: 0.93, 320: 0.95}

    def evaluate(item: calibrator.SearchConfig) -> calibrator.CandidateResult:
        measured.append(item.ef_search)
        return candidate(
            item,
            recall_lcb95=recalls[item.ef_search],
            latency_ms=float(item.ef_search),
        )

    results = calibrator.calibrate_family(config(320), [20, 40, 80, 120, 160, 320], evaluate)

    assert measured == [320, 160, 80, 120]
    assert [item.config.ef_search for item in results] == [80, 120, 160, 320]
    assert calibrator.select_fastest_qualified(results).config.ef_search == 120


def test_unmet_seed_expands_up_before_local_search() -> None:
    measured: list[int] = []
    recalls = {40: 0.70, 80: 0.85, 120: 0.91, 160: 0.94}

    def evaluate(item: calibrator.SearchConfig) -> calibrator.CandidateResult:
        measured.append(item.ef_search)
        return candidate(item, recall_lcb95=recalls[item.ef_search])

    calibrator.calibrate_family(config(40), [20, 40, 80, 120, 160], evaluate)
    assert measured == [40, 80, 160, 120]


def test_selection_uses_lowest_latency_not_lowest_ef() -> None:
    slower_low_ef = candidate(config(120), recall_lcb95=0.91, latency_ms=12.0)
    faster_high_ef = candidate(config(320), recall_lcb95=0.95, latency_ms=10.0)
    unqualified = candidate(config(80), recall_lcb95=0.89, latency_ms=5.0)
    assert calibrator.select_fastest_qualified(
        [slower_low_ef, faster_high_ef, unqualified]
    ) is faster_high_ef


def test_mean_confirmed_fallback_is_explicit_and_grid_ceiling_only() -> None:
    lower = candidate(
        config(50_000), recall_lcb95=0.89, recall_mean=0.94, latency_ms=5.0
    )
    faster_ceiling = candidate(
        config(100_000), recall_lcb95=0.89, recall_mean=0.91, latency_ms=8.0
    )
    safer_ceiling = candidate(
        config(100_000, "strict_order"),
        recall_lcb95=0.895,
        recall_mean=0.925,
        latency_ms=10.0,
    )
    with pytest.raises(RuntimeError, match="LCB95"):
        calibrator.select_fastest_qualified(
            [lower, faster_ceiling, safer_ceiling]
        )
    selected = calibrator.select_fastest_qualified(
        [lower, faster_ceiling, safer_ceiling],
        allow_mean_at_grid_ceiling=True,
    )
    assert selected is safer_ceiling
    assert selected.qualification == "mean_confirmed"


def test_mean_confirmed_config_requires_explicit_d1_admission(tmp_path: Path) -> None:
    path = tmp_path / "configs.csv"
    selected = [
        candidate(
            config(100_000, "strict_order"),
            mode=mode,
            recall_lcb95=0.895,
            recall_mean=0.925,
        )
        for mode in calibrator.MODES
    ]
    calibrator.write_configs(path, selected)
    with pytest.raises(d1_control.ControlError, match="out-of-range"):
        d1_control.load_configs(path)
    configs, order = d1_control.load_configs(path, allow_mean_qualified=True)
    assert order == ["f"]
    assert {configs["f"][mode].qualification for mode in calibrator.MODES} == {
        "mean_confirmed"
    }


def test_shared_mean_fallback_prefers_recall_before_latency() -> None:
    fast_off = config(100_000, "off")
    safer_strict = config(100_000, "strict_order")
    results = [
        candidate(
            item,
            mode=mode,
            recall_lcb95=0.89,
            recall_mean=0.91 if item is fast_off else 0.925,
            latency_ms=8.0 if item is fast_off else 10.0,
        )
        for item in (fast_off, safer_strict)
        for mode in calibrator.MODES
    ]
    selected = calibrator.select_shared_qualified(
        results,
        "max",
        allow_mean_at_grid_ceiling=True,
    )
    assert {item.config for item in selected} == {safer_strict}
    assert {item.qualification for item in selected} == {"mean_confirmed"}


def test_shared_selection_uses_common_qualified_config_and_objective() -> None:
    common_mean = config(120)
    common_max = config(160, "strict_order")
    original_only = config(320)
    results = [
        candidate(
            common_mean,
            mode="original",
            recall_lcb95=0.92,
            latency_ms=4.0,
        ),
        candidate(
            common_mean,
            mode="design1_bloom",
            recall_lcb95=0.91,
            latency_ms=10.0,
        ),
        candidate(
            common_max,
            mode="original",
            recall_lcb95=0.93,
            latency_ms=8.0,
        ),
        candidate(
            common_max,
            mode="design1_bloom",
            recall_lcb95=0.94,
            latency_ms=8.0,
        ),
        candidate(
            original_only,
            mode="original",
            recall_lcb95=0.99,
            latency_ms=1.0,
        ),
    ]

    selected_mean = calibrator.select_shared_qualified(results, "mean")
    selected_max = calibrator.select_shared_qualified(results, "max")

    assert [item.mode for item in selected_mean] == list(calibrator.MODES)
    assert {item.config for item in selected_mean} == {common_mean}
    assert {item.config for item in selected_max} == {common_max}


def test_shared_selection_requires_both_arms_to_qualify() -> None:
    shared = config(120)
    results = [
        candidate(shared, mode="original", recall_lcb95=0.91),
        candidate(shared, mode="design1_bloom", recall_lcb95=0.89),
    ]

    with pytest.raises(RuntimeError, match="both original and design1_bloom"):
        calibrator.select_shared_qualified(results, "max")


def test_child_command_transmits_atom_limit_cpu_and_external_contract(tmp_path: Path) -> None:
    filters = tmp_path / "filters.csv"
    truth = tmp_path / "truth.csv"
    write_filters(filters)
    write_truth(truth)
    args = parser_args(tmp_path, filters, truth)
    command = calibrator.build_child_command(
        args,
        calibrator.FilterSpec("f", "49.7", "labels && ARRAY[1]::int[]", 1),
        "design1_bloom",
        config(80, "strict_order"),
        tmp_path / "child.csv",
    )

    assert command[command.index("--guidance-max-atoms") + 1] == "128"
    assert command[command.index("--backend-cpu-list") + 1] == "48-63"
    assert command[command.index("--iterative-scan") + 1] == "strict_order"
    assert "--no-expected-truth-self-excluded" in command
    assert "--no-traversal-guided-prioritization" in command
    assert "--query-table" in command


def write_child_artifacts(
    raw_path: Path,
    args: Namespace,
    item: calibrator.SearchConfig,
) -> Path:
    fields = [
        "filter_name",
        "mode",
        "table",
        "index",
        "query_table",
        "query_id_column",
        "query_vector_column",
        "candidate_validity_predicate",
        "sqlens_build_id",
        "vector_so_sha256",
        "ef_search",
        "iterative_scan",
        "max_scan_tuples",
        "scan_mem_multiplier",
        "guided_collect_target",
        "traversal_guided_target",
        "truth_self_excluded",
        "planner_proof_verified",
        "guidance_enabled",
        "query_no",
        "query_id",
        "repeat",
        "recall",
        "end_to_end_ms",
        "error",
        "error_detail",
    ]
    with raw_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for query_no in (0, 1):
            for repeat in (0, 1):
                writer.writerow(
                    {
                        "filter_name": "f",
                        "mode": "design1_bloom",
                        "table": args.insertion_table,
                        "index": args.insertion_index,
                        "query_table": args.query_table,
                        "query_id_column": args.query_id_column,
                        "query_vector_column": args.query_vector_column,
                        "candidate_validity_predicate": "TRUE",
                        "sqlens_build_id": BUILD_ID,
                        "vector_so_sha256": VECTOR_SHA,
                        "ef_search": item.ef_search,
                        "iterative_scan": item.iterative_scan,
                        "max_scan_tuples": item.max_scan_tuples,
                        "scan_mem_multiplier": item.scan_mem_multiplier,
                        "guided_collect_target": item.guided_collect_target,
                        "traversal_guided_target": item.traversal_guided_target,
                        "truth_self_excluded": "False",
                        "planner_proof_verified": "True",
                        "guidance_enabled": "True",
                        "query_no": query_no,
                        "query_id": 100 + query_no,
                        "repeat": repeat,
                        "recall": "0.95",
                        "end_to_end_ms": str(10 + query_no + repeat),
                        "error": "",
                        "error_detail": "",
                    }
                )
    raw_path.with_name(raw_path.stem + "_table.csv").write_text("x\n1\n", encoding="utf-8")
    raw_path.with_name(raw_path.stem + "_profile_summary.csv").write_text("x\n1\n", encoding="utf-8")
    identity = {
        "observed_build_id": BUILD_ID,
        "observed_vector_so_sha256": VECTOR_SHA,
        "exact_match": True,
    }
    plan_path = raw_path.with_suffix(raw_path.suffix + ".plan.json")
    plan_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "error": None,
                "output": str(raw_path),
                "output_rows": 4,
                "output_sha256": calibrator.sha256_file(raw_path),
                "checks": [
                    {
                        "passed": True,
                        "expected_table": args.insertion_table,
                        "expected_table_oid": 10,
                        "expected_table_identity": args.insertion_table,
                        "expected_index": args.insertion_index,
                        "expected_index_oid": 11,
                        "expected_index_identity": args.insertion_index,
                        "expected_index_access_method": "hnsw",
                        "catalog_index_predicate": None,
                        "catalog_index_predicate_sha256": "c" * 64,
                        "query_table": args.query_table,
                        "query_id_column": args.query_id_column,
                        "query_vector_column": args.query_vector_column,
                        "query_id": 100,
                        "self_excluded": False,
                        "candidate_validity_predicate": "TRUE",
                    }
                ],
                "query_contract": {
                    "query_table": args.query_table,
                    "query_id_column": args.query_id_column,
                    "query_vector_column": args.query_vector_column,
                    "self_excluded": False,
                    "candidate_validity_predicate": "TRUE",
                },
                "sqlens_runtime_identity_startup": identity,
                "sqlens_runtime_identity_final": identity,
                "runtime_sqlens_identity_evidence": [identity],
                "backend_cpu_evidence": [
                    {
                        "exact_match": True,
                        "requested_cpu_list": "48-63",
                        "observed_cpu_list": "48-63",
                    }
                ],
                "execution_lifecycle": {
                    "backend_cpu_provenance_complete": True,
                    "runtime_sqlens_identity_complete": True,
                    "warmup_complete": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return plan_path


def test_child_plan_raw_binary_relation_and_query_provenance_are_gated(tmp_path: Path) -> None:
    filters = tmp_path / "filters.csv"
    truth = tmp_path / "truth.csv"
    write_filters(filters)
    write_truth(truth)
    args = parser_args(tmp_path, filters, truth, "--repeats", "2", "--bootstrap-samples", "100")
    args.calibration_query_id_by_no = {0: 100, 1: 101}
    item = config(80)
    raw_path = tmp_path / "child.raw.csv"
    plan_path = write_child_artifacts(raw_path, args, item)

    result = calibrator.summarize_and_validate_child(
        raw_path,
        plan_path,
        calibrator.FilterSpec("f", "49.7", "labels && ARRAY[1]::int[]", 1),
        "design1_bloom",
        item,
        [0, 1],
        args,
        ["python", "child.py"],
        child_reused=False,
    )

    assert result.recall_mean == pytest.approx(0.95)
    assert result.recall_lcb95 == pytest.approx(0.95)
    assert result.queries == 2
    assert result.samples == 4

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["sqlens_runtime_identity_final"]["observed_vector_so_sha256"] = "b" * 64
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="binary provenance"):
        calibrator.summarize_and_validate_child(
            raw_path,
            plan_path,
            calibrator.FilterSpec("f", "49.7", "p", 1),
            "design1_bloom",
            item,
            [0, 1],
            args,
            [],
            child_reused=True,
        )


def test_output_guard_never_overwrites_foreign_artifact(tmp_path: Path) -> None:
    out = tmp_path / "formal_results.csv"
    manifest = tmp_path / "formal_results.csv.manifest.json"
    children = tmp_path / "children"
    out.write_text("formal\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="without a calibration manifest"):
        calibrator.guard_output_paths(out, manifest, children, "spec", resume=True)


def test_run_writes_calibration_only_manifest_and_resumes_without_children(tmp_path: Path) -> None:
    filters = tmp_path / "filters.csv"
    truth = tmp_path / "truth.csv"
    write_filters(filters)
    write_truth(truth, query_count=4, calibration_count=2)
    args = parser_args(
        tmp_path,
        filters,
        truth,
        "--query-count",
        "2",
        "--repeats",
        "1",
        "--workers",
        "2",
        "--ef-grid",
        "20",
        "--default-seed-ef",
        "20",
        "--final-query-offset",
        "2",
        "--final-queries",
        "2",
        "--selection-coupling",
        "shared-search-config",
        "--shared-latency-objective",
        "max",
    )

    def fake_child(
        unused_args: Namespace,
        unused_children: Path,
        filter_spec: calibrator.FilterSpec,
        mode: str,
        item: calibrator.SearchConfig,
        unused_query_nos: list[int],
    ) -> calibrator.CandidateResult:
        return candidate(item, filter_name=filter_spec.name, mode=mode)

    with mock.patch.object(calibrator, "run_or_resume_child", side_effect=fake_child) as child:
        out, manifest_path = calibrator.run(args)
        assert child.call_count == 4

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["artifact_scope"] == "calibration_only"
    assert manifest["target_recall"] == 0.90
    assert manifest["query_split"]["held_out_final_executed"] is False
    assert manifest["held_out_final"]["executed"] is False
    assert manifest["protocol"]["calibration"]["query_offset"] == 0
    assert manifest["protocol"]["calibration"]["queries"] == 2
    assert manifest["protocol"]["query_offset"] == 2
    assert manifest["protocol"]["queries"] == 2
    assert manifest["protocol"]["repeats"] == 5
    assert manifest["protocol"]["workers"] == 2
    assert manifest["protocol"]["guidance_max_atoms"] == 128
    assert manifest["selection"]["coupling"] == "shared-search-config"
    assert manifest["selection"]["shared_search_config"] is True
    assert manifest["selection"]["latency_objective"] == "max"
    assert manifest["protocol"]["selection_coupling"] == "shared-search-config"
    assert manifest["protocol"]["selection_policy"] == calibrator.SHARED_SELECTION_POLICY
    assert manifest["output"]["configs_rows"] == 2
    assert calibrator.sha256_file(out) == manifest["output"]["configs_sha256"]

    with out.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2
    shared_fields = set(calibrator.SHARED_CONFIG_FIELDS) - {
        "guided_collect_target_tracks_ef"
    }
    assert all(rows[0][field] == rows[1][field] for field in shared_fields)

    d2_args = Namespace(
        expected_sqlens_build_id=BUILD_ID,
        expected_vector_so_sha256=VECTOR_SHA,
        table="public.items",
        source_index="public.items_hnsw",
        bfs_index="public.items_hnsw_bfs",
        query_table="public.queries",
        query_id_column="qid",
        query_vector_column="embedding",
        candidate_validity_predicate="TRUE",
        expected_truth_self_excluded=False,
        guidance_max_atoms=128,
        query_offset=2,
        queries=2,
        repeats=5,
        truth_manifest=tmp_path / "truth.manifest.json",
        filter_names=["f"],
        matched_target_recall=0.90,
        allow_mean_qualified_matched_config=False,
        config_csv=out,
        config_manifest=manifest_path,
        filters_csv=filters,
        truth_csv=truth,
    )
    audited = d2_control.audit_matched_configs_csv(
        out, manifest_path, d2_args, filters, truth
    )
    assert audited["f"].qualification == "lcb95"
    configs, order = d1_control.load_configs(out)
    d1_audit = d1_control.audit_config_provenance(
        d2_args, order, configs
    )
    assert d1_audit["artifact_valid"] is True

    with mock.patch.object(calibrator, "run_or_resume_child") as child:
        calibrator.run(args)
        child.assert_not_called()
