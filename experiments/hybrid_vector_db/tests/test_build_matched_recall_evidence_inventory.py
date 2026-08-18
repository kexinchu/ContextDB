from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.hybrid_vector_db.scripts import (
    build_matched_recall_evidence_inventory as inventory,
)


VECTOR_SHA = "a" * 64
OUTPUT_SHA = "b" * 64
BUILD_ID = "sqlens-test-build-20260801-r41"
MODES = ("original", "design1_bloom_bfs_layout_d3")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity() -> dict[str, object]:
    return {
        "expected_build_id": BUILD_ID,
        "observed_build_id": BUILD_ID,
        "expected_vector_so_sha256": VECTOR_SHA,
        "observed_vector_so_sha256": VECTOR_SHA,
        "exact_match": True,
    }


def benchmark_rows(*, per_filter: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in MODES:
        for filter_index, filter_name in enumerate(("filter_a", "filter_b")):
            rows.append(
                {
                    "mode": mode,
                    "filter_name": filter_name,
                    "ef_search": 1000,
                    "effective_ef_search": 1000 + 1000 * filter_index if per_filter else 1000,
                    "max_scan_tuples": 5000000,
                    "iterative_scan": "strict_order" if mode == "original" else "off",
                    "traversal_guided_target": 80,
                    "error": "",
                }
            )
    return rows


def write_plan(
    raw: Path,
    *,
    workload_requests: int,
    status: str = "complete",
    per_filter: bool = False,
) -> Path:
    write_csv(raw, benchmark_rows(per_filter=per_filter))
    plan = raw.with_name(raw.name + ".plan.json")
    payload = {
        "status": status,
        "output": str(raw),
        "output_rows": 4,
        "output_sha256": inventory.sha256_file(raw),
        "query_contract": {
            "workload_requests": workload_requests,
            "workload_unique_queries": workload_requests,
        },
        "query_error_summary": {"rows": 4, "error_rows": 0},
        "search_configuration": {
            "schema_version": 1,
            "configured_scope": "per_filter" if per_filter else "global_policy",
            "mode_defaults": {
                mode: {"ef_search": 1000} for mode in MODES
            },
            "filter_ef_search_overrides": (
                {
                    mode: {"filter_a": 1000, "filter_b": 2000}
                    for mode in MODES
                }
                if per_filter
                else {}
            ),
            "filter_traversal_target_overrides": {},
            "guidance_bypass_policy": {},
        },
        "checks": [
            {"mode": mode, "filter_name": filter_name, "passed": True}
            for mode in MODES
            for filter_name in ("filter_a", "filter_b")
        ],
        "sqlens_runtime_identity_startup": identity(),
        "sqlens_runtime_identity_final": identity(),
    }
    plan.write_text(json.dumps(payload), encoding="utf-8")
    return plan


def write_throughput_manifest(
    root: Path,
    *,
    repeats: int = 6,
    single_pass_override: bool = False,
) -> Path:
    repeats_csv = root / "throughput.repeats.csv"
    write_csv(
        repeats_csv,
        [
            {"arm_id": "stock", "requests": 10000, "error_count": 0},
            {"arm_id": "sqlens", "requests": 10000, "error_count": 0},
        ],
    )
    manifest = root / "throughput.manifest.json"
    gates = {"release_contract": True, "minimum_six_repeats": repeats >= 6}
    if single_pass_override:
        gates["single_pass_override"] = True
    payload = {
        "artifact_type": "sqlens_matched_recall_throughput_cell",
        "artifact_valid": True,
        "paper_eligible": True,
        "release_contract": {
            "contract_id": "sigmod-p0-r41-test",
            "expected_sqlens_build_id": BUILD_ID,
            "expected_vector_so_sha256": VECTOR_SHA,
        },
        "protocol": {
            "throughput_formula": "completed_queries / wall_clock_seconds",
            "unique_queries_per_arm_repeat": 10000,
            "repeats": repeats,
        },
        "configuration": {
            "value": {
                "modes": list(MODES),
                "search": {
                    "original": {"ef_search": 1000, "iterative_scan": "strict_order"},
                    "design1_bloom_bfs_layout_d3": {
                        "ef_search": 800,
                        "iterative_scan": "off",
                    },
                },
            }
        },
        "methods": {
            "stock": {"mode_id": "original"},
            "sqlens": {"mode_id": "design1_bloom_bfs_layout_d3"},
        },
        "evidence": {
            "runtime_binary_identity_start": identity(),
            "runtime_binary_identity_end": identity(),
        },
        "gates": gates,
        "outputs": {
            "repeats": {
                "path": str(repeats_csv),
                "rows": 2,
                "sha256": inventory.sha256_file(repeats_csv),
            }
        },
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_q10k_benchmark_is_formal_candidate_and_uses_csv_content(tmp_path: Path) -> None:
    plan = write_plan(tmp_path / "opaque.csv", workload_requests=10000)

    [record] = inventory.build_inventory([plan])

    assert record.classification == "formal_candidate"
    assert record.workload_tier == "q10k"
    assert record.modes == tuple(sorted(MODES))
    assert record.error_count == 0
    assert record.settings_scope == "global_policy"
    assert record.settings_source == "plan_search_configuration"
    assert record.build_id == BUILD_ID
    assert record.release_generation == "r41"
    assert record.vector_so_sha256 == VECTOR_SHA
    assert "complete q10k-or-larger artifact" in record.reasons[0]


def test_q5k_per_filter_benchmark_is_expedited(tmp_path: Path) -> None:
    plan = write_plan(
        tmp_path / "not_named_q5k.csv",
        workload_requests=5000,
        per_filter=True,
    )

    [record] = inventory.build_inventory([plan])

    assert record.classification == "expedited"
    assert record.workload_tier == "q5k"
    assert record.settings_scope == "per_filter"
    assert record.settings["filter_ef_search_overrides"]["original"]["filter_a"] == 1000
    assert record.settings["filter_ef_search_overrides"]["original"]["filter_b"] == 2000
    assert "q5k workload is expedited evidence" in record.reasons[0]


def test_well_formed_incomplete_plan_is_diagnostic(tmp_path: Path) -> None:
    plan = tmp_path / "in_progress.csv.plan.json"
    plan.write_text(
        json.dumps(
            {
                "status": "running",
                "query_contract": {
                    "workload_requests": 10000,
                    "workload_unique_queries": 10000,
                },
                "query_error_summary": {"error_rows": 0},
                "checks": [
                    {
                        "mode": mode,
                        "filter_name": filter_name,
                        "passed": True,
                        "config": {"ef_search": 1000, "iterative_scan": "strict_order"},
                    }
                    for mode in MODES
                    for filter_name in ("filter_a", "filter_b")
                ],
                "sqlens_runtime_identity_startup": identity(),
            }
        ),
        encoding="utf-8",
    )

    [record] = inventory.build_inventory([plan])

    assert record.classification == "diagnostic"
    assert record.status == "running"
    assert record.settings_source == "plan_checks"
    assert record.settings_scope == "global"
    assert "artifact status is running, not complete" in record.reasons


def test_legacy_effective_per_filter_settings_do_not_prove_tuning_scope(
    tmp_path: Path,
) -> None:
    plan = write_plan(
        tmp_path / "legacy.csv", workload_requests=10000, per_filter=True
    )
    payload = json.loads(plan.read_text(encoding="utf-8"))
    del payload["search_configuration"]
    plan.write_text(json.dumps(payload), encoding="utf-8")

    [record] = inventory.build_inventory([plan])

    assert record.classification == "diagnostic"
    assert record.settings_scope == "unknown"
    assert record.settings_source == "benchmark_csv"
    assert (
        "effective settings vary by filter, but configured tuning scope is not persisted"
        in record.reasons
    )


def test_throughput_manifest_extracts_protocol_and_repeats_errors(tmp_path: Path) -> None:
    manifest = write_throughput_manifest(tmp_path)

    [record] = inventory.build_inventory([manifest])

    assert record.classification == "formal_candidate"
    assert record.artifact_kind == "throughput_manifest"
    assert record.workload_requests == 10000
    assert record.error_count == 0
    assert record.settings_scope == "global"
    assert record.output_rows == 2
    assert record.release_id == "sigmod-p0-r41-test"


def test_single_pass_throughput_is_expedited(tmp_path: Path) -> None:
    manifest = write_throughput_manifest(
        tmp_path,
        repeats=1,
        single_pass_override=True,
    )

    [record] = inventory.build_inventory([manifest])

    assert record.classification == "expedited"
    assert "throughput uses a single-pass override (<6 repeats)" in record.reasons


def test_directory_scan_skips_well_formed_non_throughput_manifest(tmp_path: Path) -> None:
    plan = write_plan(tmp_path / "nested" / "measurement.csv", workload_requests=10000)
    unrelated = tmp_path / "truth.manifest.json"
    unrelated.write_text(
        json.dumps({"artifact_type": "exact_truth", "rows": 10000}),
        encoding="utf-8",
    )

    records = inventory.build_inventory([tmp_path])

    assert [record.artifact_path for record in records] == [str(plan.resolve())]


def test_cli_emits_stable_csv_and_json(tmp_path: Path) -> None:
    plan = write_plan(tmp_path / "input" / "measurement.csv", workload_requests=10000)
    first = tmp_path / "out" / "first"
    second = tmp_path / "out" / "second"

    assert inventory.main(["--artifact", str(plan), "--out-prefix", str(first)]) == 0
    assert inventory.main(["--artifact", str(plan), "--out-prefix", str(second)]) == 0

    first_csv, first_json = inventory.output_paths(first)
    second_csv, second_json = inventory.output_paths(second)
    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert payload["classification_counts"] == {"formal_candidate": 1}
    with first_csv.open(newline="", encoding="utf-8") as source:
        [row] = list(csv.DictReader(source))
    assert row["classification"] == "formal_candidate"
    assert (
        json.loads(row["settings_json"])["mode_defaults"]["original"]["ef_search"]
        == 1000
    )


def test_malformed_input_fails_closed_without_outputs(tmp_path: Path) -> None:
    malformed = tmp_path / "broken.csv.plan.json"
    malformed.write_text("{not json", encoding="utf-8")
    prefix = tmp_path / "inventory"

    assert inventory.main(
        ["--artifact", str(malformed), "--out-prefix", str(prefix)]
    ) == 2

    csv_path, json_path = inventory.output_paths(prefix)
    assert not csv_path.exists()
    assert not json_path.exists()
