from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import (
    figure5_throughput_repeats as repeats,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dynamic_pairs: bool = False,
) -> tuple[Path, Path]:
    monkeypatch.setattr(repeats.matched_throughput, "EXPECTED_REQUESTS", 3)
    monkeypatch.setattr(
        repeats.matched_throughput, "EXPECTED_FORMAL_PREDICATES", 2
    )
    monkeypatch.setattr(
        repeats.matched_throughput.matched,
        "MIN_FORMAL_PREDICATE_SAMPLES",
        1,
    )
    monkeypatch.setattr(repeats.artifact, "EXPECTED_REQUESTS", 3)
    monkeypatch.setitem(repeats.artifact.MIN_REPEATS, "throughput", 2)
    protocol_name = (
        "test-fixed-targets-c1-q3-r2"
        if dynamic_pairs
        else "test-c1-q3-r2"
    )
    protocol = repeats.matched_throughput.ProtocolSlice(
        name=protocol_name,
        selector_policy="fixed" if dynamic_pairs else "distinct_pairs",
        clients=(1,),
        repeats=2,
        expected_pairs=None if dynamic_pairs else 1,
        fixed_target_recall=None,
        selection_csv=root / "selection.csv",
        out_dir=root / "out",
        fixed_targets=(0.90, 0.95, 0.99) if dynamic_pairs else (),
    )
    monkeypatch.setitem(
        repeats.matched_throughput.PROTOCOL_SLICES,
        protocol_name,
        protocol,
    )

    contract = root / "release.json"
    contract.write_text(
        json.dumps(
            {
                "contract_id": "test-r36",
                "expected_sqlens_build_id": "sqlens-test-r36",
                "expected_vector_so_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    release_sha = sha(contract)
    repeat_csv = root / "cell.repeats.csv"
    rows: list[dict[str, object]] = []
    for arm, mode, config_sha in (
        ("stock_pgvector", "original", "b" * 64),
        (
            "sqlens_full",
            "design1_bloom_bfs_layout_d3",
            "c" * 64,
        ),
    ):
        for repeat_id in range(2):
            rows.append(
                row := {
                    "schema_version": 1,
                    "run_id": "test-run",
                    "dataset": "amazon10m",
                    "experiment_kind": "throughput",
                    "arm_id": arm,
                    "mode_id": mode,
                    "config_id": "amazon:recall_0.900",
                    "config_sha256": config_sha,
                    "stock_config_sha256": "b" * 64,
                    "sqlens_config_sha256": "c" * 64,
                    "arm_config_sha256": config_sha,
                    "release_identity_sha256": release_sha,
                    "clients": 1,
                    "repeat_id": repeat_id,
                    "request_trace_sha256": "d" * 64,
                    "requests": 3,
                    "unique_queries": 3,
                    "completed_queries": 3,
                    "error_count": 0,
                    "wall_clock_seconds": 1.0,
                    "recall_mean": 0.95,
                    "recall_ci95_low": 0.94,
                    "recall_ci95_high": 0.96,
                    "latency_mean_ms": 10.0,
                    "latency_p95_ms": 12.0,
                    "latency_p99_ms": 13.0,
                    "throughput_qps": 3.0,
                    "throughput_ci95_low": 2.8,
                    "throughput_ci95_high": 3.2,
                    "throughput_source": (
                        repeats.throughput.THROUGHPUT_SOURCE
                    ),
                    "status": "valid",
                }
            )
            for field in repeats.throughput.REPEAT_FIELDS:
                row.setdefault(field, "")
            row.update(
                {
                    "pair_id": "amazon:recall_0.900",
                    "target_recall": 0.90,
                    "telemetry_collected": "true",
                    "telemetry_json": json.dumps(
                        {"backend_proc_root": "/proc/123/root/proc"}
                    ),
                    "pg_backend_cpu_processes": 1,
                    **{
                        field: 10.0
                        for field in repeats.CPU_WEIGHTED_FIELDS
                    },
                    **{
                        field: 1.0
                        for field in repeats.COUNTER_FIELDS
                    },
                }
            )
    write_csv(repeat_csv, rows)
    filter_names = ["f0", "f1"]
    aggregate = {}
    per_predicate = {}
    worst_by_arm = {}
    for arm in ("stock_pgvector", "sqlens_full"):
        for repeat_id in range(2):
            arm_key = f"{arm}/repeat={repeat_id}"
            aggregate[arm_key] = {
                "sample_count": 3,
                "mean": 0.95,
                "lower": 0.94,
                "upper": 0.96,
                "target": 0.90,
                "passed": True,
            }
            per_predicate[arm_key] = {
                filter_name: {
                    "sample_count": 1 if filter_name == "f0" else 2,
                    "sample_count_sufficient": True,
                    "mean": 0.95,
                    "lower": 0.94,
                    "upper": 0.96,
                    "target": 0.90,
                    "passed": True,
                }
                for filter_name in filter_names
            }
            worst_by_arm[arm_key] = {
                "filter_name": "f0",
                **per_predicate[arm_key]["f0"],
            }
    recall_gate = {
        "qualification_scope": (
            repeats.matched_throughput.matched.QUALIFICATION_SCOPE_FORMAL
        ),
        "formal_predicate_sample_floor": 1,
        "expected_predicate_count": 2,
        "observed_predicate_count": 2,
        "filter_names": filter_names,
        "passed": True,
        "paper_eligible": True,
        "reason": "ok",
        "aggregate": aggregate,
        "per_predicate": per_predicate,
        "worst_predicate_by_arm": worst_by_arm,
        "worst_predicate": {
            "arm_repeat": "sqlens_full/repeat=0",
            **worst_by_arm["sqlens_full/repeat=0"],
        },
    }
    manifest = root / "throughput-run.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": (
                    "sqlens_figure5_matched_throughput_run"
                ),
                "status": "complete",
                "artifact_valid": True,
                "requested_slice_complete": True,
                "full_release_complete": True,
                "paper_eligible": True,
                "protocol_slice": protocol_name,
                "protocol_fingerprint_sha256": "e" * 64,
                "release_contract": {
                    "path": str(contract),
                    "sha256": release_sha,
                    "contract_id": "test-r36",
                    "expected_sqlens_build_id": "sqlens-test-r36",
                    "expected_vector_so_sha256": "a" * 64,
                },
                "execution": {
                    "requests_per_arm_repeat": 3,
                    "repeats": 2,
                    "expected_repeat_rows_per_cell": 4,
                    "client_grid": [1],
                    "backend_proc_root": "/proc/123/root/proc",
                    "throughput_source": (
                        repeats.throughput.THROUGHPUT_SOURCE
                    ),
                    "throughput_formula": (
                        "completed_queries / barrier_wall_clock_seconds"
                    ),
                    "qps_from_latency_forbidden": True,
                },
                "full_release_scope": {
                    "kind": protocol_name,
                    "requested": True,
                    "checks": {"fixture_complete": True},
                    "required_pairs": ["amazon:recall_0.900"],
                    "required_pair_cells": [
                        {
                            "dataset": "amazon",
                            "pair_id": "amazon:recall_0.900",
                            "target_recall": 0.90,
                        }
                    ],
                    "requested_pairs": ["amazon:recall_0.900"],
                    "required_clients": [1],
                    "requested_clients": [1],
                    "required_repeats": 2,
                },
                "frontier_config": {"sha256": "f" * 64},
                "selector": {
                    "selection_csv_sha256": "1" * 64,
                    "selection_plan_sha256": "2" * 64,
                    "selection_manifest_sha256": "3" * 64,
                    "qualification_scope": (
                        repeats.matched_throughput.matched
                        .QUALIFICATION_SCOPE_FORMAL
                    ),
                    "selected_pairs": 1,
                    "target_rows": 9 if dynamic_pairs else 1,
                    "unattainable_pairs": 8 if dynamic_pairs else 0,
                    "targets_by_dataset": {
                        dataset: [0.90, 0.95, 0.99]
                        for dataset in ("amazon", "yfcc", "laion")
                    } if dynamic_pairs else {},
                },
                "normalized_measurement_plan": {"sha256": "4" * 64},
                "cells_total": 1,
                "cells_complete": 1,
                "schedule": [
                    {
                        "cell_id": "amazon:test:c1",
                        "dataset": "amazon",
                        "pair_id": "amazon:recall_0.900",
                        "target_recall": 0.90,
                        "clients": 1,
                        "status": "complete",
                        "paths": {"repeats": str(repeat_csv)},
                        "completion_audit": {
                            "complete": True,
                            "recall_gate": recall_gate,
                            "outputs": {
                                "repeats": {
                                    "path": str(repeat_csv),
                                    "sha256": sha(repeat_csv),
                                    "rows": 4,
                                }
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, repeat_csv


def test_converter_publishes_one_audited_throughput_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "throughput-repeats.csv"

    assert repeats.main(
        [
            "--run-manifest",
            str(manifest),
            "--out",
            str(output),
        ]
    ) == 0

    binding_path = output.with_suffix(output.suffix + ".manifest.json")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["paper_eligible"] is True
    assert binding["converter_binding"]["output"]["rows"] == 4
    assert binding["converter_binding"]["output"]["sha256"] == sha(output)
    assert binding["service_aggregate"]["rows"] == 2
    assert binding["service_aggregate"]["qps_from_latency_forbidden"] is True
    with output.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4
    assert {row["source_manifest_sha256"] for row in rows} == {
        sha(manifest)
    }
    service = output.with_name(output.stem + ".service.csv")
    with service.open(newline="", encoding="utf-8") as source:
        service_rows = list(csv.DictReader(source))
    assert len(service_rows) == 2
    assert {row["throughput_qps"] for row in service_rows} == {"3.0"}
    assert {
        row["throughput_ci_method"] for row in service_rows
    } == {repeats.QPS_BOOTSTRAP_METHOD}
    assert {
        row["recall_qualification_scope"] for row in service_rows
    } == {"global_min_predicate_lcb"}
    assert {row["recall_predicate_count"] for row in service_rows} == {"2"}
    assert {
        row["recall_worst_predicate_filter"] for row in service_rows
    } == {"f0"}
    assert all(
        float(row["recall_lcb95"]) == 0.94 for row in service_rows
    )
    assert all(float(row["latency_p95_ci95_low_ms"]) == 12.0 for row in service_rows)
    assert all(float(row["latency_p99_ci95_high_ms"]) == 13.0 for row in service_rows)
    assert all(float(row["host_disk_read_bytes"]) == 2.0 for row in service_rows)
    assert all(float(row["pg_backend_cpu_total_ms"]) == 2.0 for row in service_rows)


def test_pooled_qps_ci_bootstraps_the_same_ratio_as_its_center() -> None:
    center, lower, upper = repeats.pooled_qps_bootstrap(
        [100, 10],
        [10.0, 2.0],
        samples=1_000,
        seed=11,
        seed_label="amazon:pair:stock",
    )

    assert center == pytest.approx(110.0 / 12.0)
    assert lower == pytest.approx(5.0)
    assert upper == pytest.approx(10.0)
    assert lower <= center <= upper


def test_converter_rejects_repeat_sha_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, repeat_csv = fixture(tmp_path, monkeypatch)
    repeat_csv.write_text(
        repeat_csv.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        repeats.ThroughputRepeatError,
        match="binding is invalid",
    ):
        repeats.convert_manifest(manifest)


def test_fixed_target_converter_accepts_dynamic_selected_pair_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = fixture(tmp_path, monkeypatch, dynamic_pairs=True)

    output = tmp_path / "dynamic-throughput-repeats.csv"

    assert repeats.main(
        ["--run-manifest", str(manifest), "--out", str(output)]
    ) == 0
    with output.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    service = output.with_name(output.stem + ".service.csv")
    with service.open(newline="", encoding="utf-8") as source:
        service_rows = list(csv.DictReader(source))

    assert len(rows) == 4
    assert len(service_rows) == 2
    assert {row["recall_predicate_count"] for row in service_rows} == {"2"}


def test_fixed_target_converter_rejects_selector_schedule_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = fixture(tmp_path, monkeypatch, dynamic_pairs=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["selector"]["selected_pairs"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        repeats.ThroughputRepeatError,
        match="selected_pairs differs",
    ):
        repeats.convert_manifest(manifest)


def test_converter_rejects_aggregate_only_recall_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = fixture(tmp_path, monkeypatch, dynamic_pairs=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["schedule"][0]["completion_audit"]["recall_gate"][
        "per_predicate"
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        repeats.ThroughputRepeatError,
        match="predicate recall coverage is invalid",
    ):
        repeats.convert_manifest(manifest)
