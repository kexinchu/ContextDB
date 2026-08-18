from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.hybrid_vector_db.scripts import audit_sigmod_matched_recall_artifact as audit
from experiments.hybrid_vector_db.scripts.pgvector_design1_design2_design3_selectivity_benchmark import (
    d2_stable_fingerprint,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": audit.sha256_file(path),
        "bytes": path.stat().st_size,
        "row_count": audit.csv_row_count(path),
    }


def d2_proof() -> dict[str, object]:
    equal = "sha256:" + "1" * 64
    proof: dict[str, object] = {
        "proof_contract": "sqlens_same_heap_same_logical_graph_physical_layout_v2",
        "source_index": "public.source_idx",
        "clone_index": "public.clone_idx",
        "relations": {
            "source": {"name": "public.source_idx", "oid": 10, "relfilenode": 20, "heap_oid": 5},
            "clone": {"name": "public.clone_idx", "oid": 11, "relfilenode": 21, "heap_oid": 5},
        },
        "comparison": {
            "format": "sqlens-hnsw-compare-v2",
            "same_heap": True,
            "logical_equal": True,
            "physical_equal": False,
            "entry_equal": True,
            "definition_equal": True,
            "tuple_coverage_equal": True,
            "left_definition_digest": equal,
            "right_definition_digest": equal,
            "left_tuple_coverage_digest": equal,
            "right_tuple_coverage_digest": equal,
            "left_logical_digest": equal,
            "right_logical_digest": equal,
            "left_physical_digest": "sha256:" + "2" * 64,
            "right_physical_digest": "sha256:" + "3" * 64,
        },
    }
    proof["stable_fingerprint_sha256"] = d2_stable_fingerprint(proof)
    return proof


def write_plan(
    raw: Path,
    modes: list[str],
    *,
    build_id: str,
    vector_sha: str,
    proof: dict[str, object] | None,
) -> Path:
    identity = {
        "expected_build_id": build_id,
        "observed_build_id": build_id,
        "expected_vector_so_sha256": vector_sha,
        "observed_vector_so_sha256": vector_sha,
        "exact_match": True,
    }
    payload: dict[str, object] = {
        "status": "complete",
        "checks": [{"mode": mode, "passed": True} for mode in modes],
        "execution_lifecycle": {
            "warmup_complete": True,
            "d3_lifecycle_complete": True,
            "backend_cpu_provenance_complete": True,
            "runtime_sqlens_identity_complete": True,
        },
        "backend_cpu_evidence": [
            {
                "backend_pid": 123,
                "observed_cpu_list": "0",
                "requested_cpu_list": None,
                "pinning_attempted_by_runner": False,
            }
        ],
        "runtime_sqlens_identity_evidence": [{"exact_match": True}],
        "sqlens_runtime_identity_startup": identity,
        "sqlens_runtime_identity_final": identity,
        "output_sha256": audit.sha256_file(raw),
        "output_rows": audit.csv_row_count(raw),
    }
    if proof is not None:
        payload["d2_graph_proof"] = proof
        payload["d2_graph_proof_final"] = proof
    path = raw.with_suffix(raw.suffix + ".plan.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def build_manifest(
    tmp_path: Path,
    *,
    method: str = "design1_bloom_bfs_layout_d3",
    final_ids: list[int] | None = None,
    calibration_ids: list[int] | None = None,
    final_repeats: int = 2,
    interleaved: bool = True,
    omit_key: tuple[str, int, int] | None = None,
    duplicate_key: tuple[str, int, int] | None = None,
    wrong_query_id: tuple[str, int, int] | None = None,
    row_overrides: dict[tuple[str, int, int], dict[str, object]] | None = None,
    drop_raw_fields: dict[tuple[str, int, int], set[str]] | None = None,
) -> tuple[Path, Path, Path]:
    final_ids = final_ids or [1001, 1002, 1003]
    calibration_ids = calibration_ids or [1, 2]
    final_offset = 100
    modes = ["original", method]
    filters = ["f1"]
    targets = [0.9]
    uses_d2 = method in audit.D2_MODES
    proof = d2_proof() if uses_d2 else None

    truth = tmp_path / "truth.csv"
    filters_csv = tmp_path / "filters.csv"
    write_csv(truth, [{"query_no": 0, "query_id": 1}])
    write_csv(filters_csv, [{"filter_name": "f1", "sql": "price < 10"}])

    row_overrides = row_overrides or {}
    drop_raw_fields = drop_raw_fields or {}
    raw_by_mode: dict[str, Path] = {}
    shared_raw = tmp_path / "final.interleaved.csv"
    all_rows: list[dict[str, object]] = []
    mode_rows = {mode: [] for mode in modes}
    mode_row_counts = {mode: 0 for mode in modes}
    pair_number = 0
    for query_position, query_id in enumerate(final_ids):
        query_no = final_offset + query_position
        for repeat in range(final_repeats):
            mode_order = modes[pair_number % len(modes) :] + modes[: pair_number % len(modes)]
            for schedule_position, mode in enumerate(mode_order, start=1):
                key = (mode, query_no, repeat)
                if key == omit_key:
                    continue
                row = {
                    "filter_name": "f1",
                    "mode": mode,
                    "query_no": query_no,
                    "query_id": query_id + (999 if key == wrong_query_id else 0),
                    "repeat": repeat,
                    "error": "",
                }
                if interleaved:
                    row.update(
                        {
                            "pair_key": f"f1|q{query_no}|r{repeat}",
                            "schedule_position": schedule_position,
                        }
                    )
                if mode == "design1_bloom_bfs_layout_d3":
                    d3_position = mode_row_counts[mode]
                    if d3_position < 2:
                        row.update(
                            {
                                "d3_phase": "probe",
                                "guidance_route": "d3_stock_probe",
                                "guidance_enabled": False,
                                "d3_active_after": False,
                                "d3_admitted_after": False,
                            }
                        )
                    elif d3_position == 2:
                        row.update(
                            {
                                "d3_phase": "admission",
                                "guidance_route": "enabled",
                                "guidance_enabled": True,
                                "d3_active_after": True,
                                "d3_admitted_after": True,
                                "d3_adaptive_admissions_delta": 1,
                            }
                        )
                    else:
                        row.update(
                            {
                                "d3_phase": "warm",
                                "guidance_route": "enabled",
                                "guidance_enabled": True,
                                "d3_active_after": True,
                                "d3_admitted_after": True,
                                "d3_same_predicate_before": True,
                                "d3_admitted_before": True,
                                "d3_active_guidance_reused": True,
                            }
                        )
                row.update(row_overrides.get(key, {}))
                for field in drop_raw_fields.get(key, set()):
                    row.pop(field, None)
                mode_rows[mode].append(row)
                mode_row_counts[mode] += 1
                if interleaved:
                    all_rows.append(row)
                if key == duplicate_key:
                    mode_rows[mode].append(dict(row))
                    if interleaved:
                        all_rows.append(dict(row))
            pair_number += 1
    for mode in modes:
        if interleaved:
            raw_by_mode[mode] = shared_raw
        else:
            path = tmp_path / f"final.{mode}.csv"
            write_csv(path, mode_rows[mode])
            raw_by_mode[mode] = path
    if interleaved:
        write_csv(shared_raw, all_rows)

    contract = audit.load_p0_release_contract(audit.DEFAULT_P0_RELEASE_CONTRACT)
    build_id = str(contract["expected_sqlens_build_id"])
    vector_sha = str(contract["expected_vector_so_sha256"])
    plan_entries = []
    for raw in sorted(set(raw_by_mode.values())):
        raw_modes = modes if interleaved else [mode for mode, path in raw_by_mode.items() if path == raw]
        plan = write_plan(
            raw,
            raw_modes,
            build_id=build_id,
            vector_sha=vector_sha,
            proof=proof if any(mode in audit.D2_MODES for mode in raw_modes) else None,
        )
        plan_entries.append({**artifact(plan), "raw_output": artifact(raw)})

    selected = tmp_path / "selected.csv"
    write_csv(
        selected,
        [
            {
                "target_recall": 0.9,
                "filter_name": "f1",
                "mode": mode,
                "grid_exhausted": True,
                "stopped_early": False,
                "calibration_grid_policy": audit.FORMAL_CALIBRATION_GRID_POLICY,
                "selection_status": "selected",
                "target_lcb95_met_in_calibration": True,
            }
            for mode in modes
        ],
    )
    calibration = tmp_path / "calibration.csv"
    write_csv(
        calibration,
        [{"target_recall": 0.9, "filter_name": "f1", "mode": mode} for mode in modes],
    )
    final = tmp_path / "final.csv"
    final_rows = []
    for mode in modes:
        raw = raw_by_mode[mode]
        final_rows.append(
            {
                "target_recall": 0.9,
                "filter_name": "f1",
                "mode": mode,
                "recall_mean": 0.92,
                "recall_lcb95": 0.90,
                "target_met_in_final": True,
                "target_confirmed_in_calibration": True,
                "target_confirmed_in_final": True,
                "rows_complete": True,
                "final_status": "complete",
                "errors": 0,
                "matched_recall_comparison_valid": True,
                "expected_queries": len(final_ids),
                "expected_repeats": final_repeats,
                "paired_queries": len(final_ids),
                "paired_repeats": final_repeats,
                "paired_samples": len(final_ids) * final_repeats,
                "latency_mean_ms": 10.0 if mode == "original" else 5.0,
                "final_raw": str(raw),
                "final_raw_sha256": audit.sha256_file(raw),
                "final_raw_rows": audit.csv_row_count(raw),
                "final_execution_order": "interleaved" if interleaved else "mode_major",
                "final_schedule_id": "schedule-1" if interleaved else f"schedule-{mode}",
            }
        )
    write_csv(final, final_rows)

    run_spec: dict[str, object] = {
        "args": {
            "calibration_queries": len(calibration_ids),
            "final_queries": len(final_ids),
            "final_repeats": final_repeats,
            "final_query_offset": final_offset,
            "final_execution_order": "interleaved" if interleaved else "mode_major",
            "calibration_selection_policy": "lcb_then_max_recall",
            "prewarm_index_health": False,
        },
        "truth_sha256": audit.sha256_file(truth),
        "filters_sha256": audit.sha256_file(filters_csv),
        "calibration_query_ids": calibration_ids,
        "final_query_ids": final_ids,
        "sqlens_runtime_provenance": {
            "loaded_vector_sqlens_build_id": build_id,
            "loaded_vector_so_sha256": vector_sha,
        },
        "runtime_identity_binding": {
            "expected_build_id": build_id,
            "expected_vector_so_sha256": vector_sha,
            "exact_match": True,
        },
        "p0_release_contract": contract,
        "d2_graph_proof": proof,
    }
    run_spec["run_spec_hash"] = audit._run_spec_hash(run_spec, uses_d2)
    manifest = {
        "status": "complete",
        "matrix_complete": True,
        "measurement_complete": True,
        "comparison_valid": True,
        "requested_slice_complete": True,
        "formal_release_complete": False,
        "diagnostic_valid": True,
        "artifact_valid": False,
        "paper_eligible": False,
        "run_spec_hash": run_spec["run_spec_hash"],
        "run_spec": run_spec,
        "filters": filters,
        "modes": modes,
        "targets": targets,
        "calibration_queries": len(calibration_ids),
        "final_queries": len(final_ids),
        "final_repeats": final_repeats,
        "final_query_offset": final_offset,
        "final_execution_order": "interleaved" if interleaved else "mode_major",
        "calibration_policy": {
            "calibration_selection_policy": "lcb_then_max_recall",
            "selection": (
                "lowest mean latency among all complete configurations whose LCB95 "
                "reaches the target"
            ),
            "stop_metric": "recall_lcb95",
            "grid_policy": audit.FORMAL_CALIBRATION_GRID_POLICY,
            "base_grid_max_ef": audit.FORMAL_BASE_GRID_MAX_EF,
            "base_grid_complete_required": True,
            "extension_ef_search_values": [20_000, 50_000, 100_000],
            "extension_trigger": (
                "max_target_lcb95_unmet_after_complete_base_grid"
            ),
            "extension_complete_required_when_triggered": True,
            "early_stop_allowed": False,
            "grid_exhaustion_semantics": "all_policy_required_configs_executed",
            "stop_condition": (
                "complete every 20--10000 base configuration and run the complete "
                "20000--100000 extension only when the maximum target remains unmet; "
                "early stop are forbidden"
            ),
        },
        "mode_grids": {
            mode: [
                {"ef_search": 100, "iterative_scan": "strict_order"},
                {"ef_search": 20_000, "iterative_scan": "strict_order"},
            ]
            for mode in modes
        },
        "calibration_pairs": [
            {
                "filter_name": "f1",
                "mode": mode,
                "calibration_grid_policy": audit.FORMAL_CALIBRATION_GRID_POLICY,
                "grid_exhausted": True,
                "stopped_early": False,
                "families": {
                    "strict_order": {
                        "configs_planned": 1,
                        "configs_executed": 1,
                        "grid_exhausted": True,
                        "stopped_early": False,
                        "stopped_by_cross_family_target": False,
                        "high_extension_required": False,
                        "high_extension_executed": False,
                        "high_extension_skip_reason": (
                            "max_target_lcb_met_on_complete_base_grid"
                        ),
                        "max_ef_evaluated": 100,
                    }
                },
            }
            for mode in modes
        ],
        "outputs": {
            "calibration": artifact(calibration),
            "selected": artifact(selected),
            "final": artifact(final),
        },
        "plan_evidence": plan_entries,
        "d2_graph_proof_final": proof,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, truth, filters_csv


def run_audit(paths: tuple[Path, Path, Path]) -> dict[str, object]:
    manifest, truth, filters = paths
    return audit.audit_manifest(manifest, truth_csv=truth, filters_csv=filters)


def test_accepts_real_interleaved_shape_and_d2_proof(tmp_path: Path) -> None:
    manifest, truth, filters = build_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    calibration_rows = read_csv_rows(Path(payload["outputs"]["calibration"]["path"]))

    assert "pair_key" not in calibration_rows[0]
    assert "schedule_position" not in calibration_rows[0]

    report = audit.audit_manifest(manifest, truth_csv=truth, filters_csv=filters)
    assert report["valid"] is True, report["errors"]
    assert report["methods"]["design1_bloom_bfs_layout_d3"]["mean_speedup_vs_stock"] == 2.0


def test_d3_phase_audit_rejects_cross_predicate_warm_claim() -> None:
    row = {
        "d3_phase": "warm",
        "guidance_route": "enabled",
        "guidance_enabled": True,
        "d3_active_after": True,
        "d3_admitted_after": True,
        "d3_same_predicate_before": False,
        "d3_admitted_before": True,
        "d3_active_guidance_reused": True,
    }

    errors = audit._d3_phase_errors(row, "raw[0]")

    assert errors
    assert "predicate-scoped" in errors[0]


def test_rejects_calibration_final_overlap(tmp_path: Path) -> None:
    report = run_audit(build_manifest(tmp_path, calibration_ids=[1, 1002]))
    assert report["valid"] is False
    assert any("overlap" in error for error in report["errors"])


def test_rejects_lcb_policy_with_report_only_manifest_description(tmp_path: Path) -> None:
    manifest, truth, filters = build_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["run_spec"]["args"]["calibration_selection_policy"] = "lcb_then_max_recall"
    payload["calibration_policy"] = {
        "calibration_selection_policy": "lcb_then_max_recall",
        "selection": "lowest mean latency; bootstrap CI/LCB is report-only",
    }
    uses_d2 = any(mode in audit.D2_MODES for mode in payload["modes"])
    run_hash = audit._run_spec_hash(payload["run_spec"], uses_d2)
    payload["run_spec"]["run_spec_hash"] = run_hash
    payload["run_spec_hash"] = run_hash
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = audit.audit_manifest(manifest, truth_csv=truth, filters_csv=filters)

    assert report["valid"] is False
    assert any("does not make LCB part of selection" in error for error in report["errors"])


def test_rejects_lcb_policy_with_mean_based_early_stop(tmp_path: Path) -> None:
    manifest, truth, filters = build_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["run_spec"]["args"]["calibration_selection_policy"] = "lcb_then_max_recall"
    payload["calibration_policy"] = {
        "calibration_selection_policy": "lcb_then_max_recall",
        "selection": "lowest latency among configurations whose LCB95 reaches the target",
        "stop_metric": "recall_mean",
        "stop_condition": "stop when recall_mean reaches the highest target",
    }
    uses_d2 = any(mode in audit.D2_MODES for mode in payload["modes"])
    run_hash = audit._run_spec_hash(payload["run_spec"], uses_d2)
    payload["run_spec"]["run_spec_hash"] = run_hash
    payload["run_spec_hash"] = run_hash
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = audit.audit_manifest(manifest, truth_csv=truth, filters_csv=filters)

    assert report["valid"] is False
    assert any(
        "qualification metric" in error or "formal calibration policy" in error
        for error in report["errors"]
    )


def test_rejects_legacy_first_crossing_grid_policy(tmp_path: Path) -> None:
    manifest, truth, filters = build_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["calibration_policy"]["grid_policy"] = (
        "ascending_prefix_first_max_target_lcb_or_latency_dominated"
    )
    payload["calibration_policy"]["stop_condition"] = (
        "stop at first max-target LCB crossing or latency dominance"
    )
    uses_d2 = any(mode in audit.D2_MODES for mode in payload["modes"])
    run_hash = audit._run_spec_hash(payload["run_spec"], uses_d2)
    payload["run_spec"]["run_spec_hash"] = run_hash
    payload["run_spec_hash"] = run_hash
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = audit.audit_manifest(manifest, truth_csv=truth, filters_csv=filters)

    assert report["valid"] is False
    assert any("unknown or incomplete" in error for error in report["errors"])


def test_rejects_non_unique_final_query_ids(tmp_path: Path) -> None:
    report = run_audit(build_manifest(tmp_path, final_ids=[1001, 1001, 1002]))
    assert report["valid"] is False
    assert any("non-unique" in error for error in report["errors"])


def test_rejects_truth_hash_mismatch(tmp_path: Path) -> None:
    manifest, truth, filters = build_manifest(tmp_path)
    truth.write_text("tampered\n", encoding="utf-8")
    report = audit.audit_manifest(manifest, truth_csv=truth, filters_csv=filters)
    assert report["valid"] is False
    assert any("truth_sha256" in error for error in report["errors"])


def test_rejects_persisted_artifact_tampering(tmp_path: Path) -> None:
    manifest, truth, filters = build_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    final = Path(payload["outputs"]["final"]["path"])
    final.write_text(final.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    report = audit.audit_manifest(manifest, truth_csv=truth, filters_csv=filters)
    assert report["valid"] is False
    assert any("outputs.final sha256 mismatch" in error for error in report["errors"])


def test_rejects_stale_d2_proof(tmp_path: Path) -> None:
    manifest, truth, filters = build_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["d2_graph_proof_final"]["stable_fingerprint_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    report = audit.audit_manifest(manifest, truth_csv=truth, filters_csv=filters)
    assert report["valid"] is False
    assert any("D2" in error for error in report["errors"])


def test_rejects_missing_repeat(tmp_path: Path) -> None:
    report = run_audit(
        build_manifest(tmp_path, omit_key=("design1_bloom_bfs_layout_d3", 101, 1))
    )
    assert report["valid"] is False
    assert any("coverage mismatch" in error for error in report["errors"])


def test_rejects_duplicate_interleaved_row(tmp_path: Path) -> None:
    report = run_audit(build_manifest(tmp_path, duplicate_key=("original", 100, 0)))
    assert report["valid"] is False
    assert any("duplicate raw" in error for error in report["errors"])


def test_rejects_interleaved_row_missing_pair_key(tmp_path: Path) -> None:
    report = run_audit(
        build_manifest(
            tmp_path,
            drop_raw_fields={("original", 100, 0): {"pair_key"}},
        )
    )

    assert report["valid"] is False
    assert any("empty pair_key" in error for error in report["errors"])


def test_rejects_interleaved_row_missing_schedule_position(tmp_path: Path) -> None:
    report = run_audit(
        build_manifest(
            tmp_path,
            drop_raw_fields={("original", 100, 0): {"schedule_position"}},
        )
    )

    assert report["valid"] is False
    assert any("schedule_position is not an integer" in error for error in report["errors"])


def test_rejects_interleaved_pair_missing_requested_mode(tmp_path: Path) -> None:
    report = run_audit(
        build_manifest(
            tmp_path,
            omit_key=("design1_bloom_bfs_layout_d3", 100, 0),
        )
    )

    assert report["valid"] is False
    assert any("interleaved pair modes mismatch" in error for error in report["errors"])


def test_rejects_duplicate_schedule_position_within_pair(tmp_path: Path) -> None:
    report = run_audit(
        build_manifest(
            tmp_path,
            row_overrides={
                ("design1_bloom_bfs_layout_d3", 100, 0): {"schedule_position": 1}
            },
        )
    )

    assert report["valid"] is False
    assert any("schedule positions mismatch" in error for error in report["errors"])


def test_rejects_globally_unbalanced_mode_positions(tmp_path: Path) -> None:
    method = "design1_bloom_bfs_layout_d3"
    overrides = {
        (mode, query_no, repeat): {
            "schedule_position": 1 if mode == "original" else 2
        }
        for mode in ("original", method)
        for query_no in (100, 101, 102)
        for repeat in range(2)
    }
    report = run_audit(build_manifest(tmp_path, row_overrides=overrides))

    assert report["valid"] is False
    assert any(
        "unbalanced interleaved schedule positions" in error
        for error in report["errors"]
    )


def test_rejects_pair_key_that_does_not_match_request_identity(tmp_path: Path) -> None:
    report = run_audit(
        build_manifest(
            tmp_path,
            row_overrides={("original", 100, 0): {"pair_key": "f1|q999|r0"}},
        )
    )

    assert report["valid"] is False
    assert any("pair_key mismatch" in error for error in report["errors"])


def test_rejects_wrong_query_id_mapping(tmp_path: Path) -> None:
    report = run_audit(build_manifest(tmp_path, wrong_query_id=("original", 102, 1)))
    assert report["valid"] is False
    assert any("coverage mismatch" in error for error in report["errors"])


def test_rejects_fake_interleave_across_raw_files(tmp_path: Path) -> None:
    manifest, truth, filters = build_manifest(tmp_path, interleaved=False)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    final_path = Path(payload["outputs"]["final"]["path"])
    rows = read_csv_rows(final_path)
    for row in rows:
        row["final_execution_order"] = "interleaved"
        row["final_schedule_id"] = "same-schedule"
    write_csv(final_path, rows)
    payload["outputs"]["final"] = artifact(final_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    report = audit.audit_manifest(manifest, truth_csv=truth, filters_csv=filters)
    assert report["valid"] is False
    assert any("invalid interleaved pairing" in error for error in report["errors"])


def test_rejects_mode_major_formal_final(tmp_path: Path) -> None:
    report = run_audit(build_manifest(tmp_path, interleaved=False))

    assert report["valid"] is False
    assert any(
        "formal final execution order must be interleaved" in error
        for error in report["errors"]
    )


def test_reaudits_legacy_requested_slice_without_mutating_source(tmp_path: Path) -> None:
    manifest, truth, filters = build_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "incomplete",
            "matrix_complete": False,
            "measurement_complete": False,
            "comparison_valid": False,
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    source_sha256 = audit.sha256_file(manifest)

    amended = audit.write_completion_reaudited_manifest(
        manifest, truth_csv=truth, filters_csv=filters
    )

    assert audit.sha256_file(manifest) == source_sha256
    amended_payload = json.loads(amended.read_text(encoding="utf-8"))
    assert amended.name.endswith(".release-audited.json")
    assert amended_payload["status"] == "complete"
    assert amended_payload["matrix_complete"] is True
    assert amended_payload["measurement_complete"] is True
    assert amended_payload["comparison_valid"] is True
    assert amended_payload["formal_release_complete"] is False
    assert amended_payload["diagnostic_valid"] is True
    assert amended_payload["artifact_valid"] is False
    assert amended_payload["paper_eligible"] is False
    assert (
        amended_payload["run_spec"]["p0_release_contract"]["sha256"]
        == audit.load_p0_release_contract(audit.DEFAULT_P0_RELEASE_CONTRACT)["sha256"]
    )
    assert amended_payload["completion_reaudit"]["source_manifest_sha256"] == source_sha256
    report = audit.audit_manifest(amended, truth_csv=truth, filters_csv=filters)
    assert report["valid"] is True, report["errors"]


def test_reaudit_refuses_tampered_artifact(tmp_path: Path) -> None:
    manifest, truth, filters = build_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    final_path = Path(payload["outputs"]["final"]["path"])
    final_path.write_text(final_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    try:
        audit.write_completion_reaudited_manifest(
            manifest, truth_csv=truth, filters_csv=filters
        )
    except audit.AuditError as exc:
        assert "failed re-audit" in str(exc)
    else:
        raise AssertionError("tampered artifact unexpectedly produced a repaired manifest")


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))
