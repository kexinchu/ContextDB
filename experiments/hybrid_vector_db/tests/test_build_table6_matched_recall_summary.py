from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import (
    build_table6_matched_recall_summary as summary,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
PAIR_ID = "amazon:recall_0.900000000:stock:sqlens"


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fields: list[str] | None = None,
) -> None:
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def release_contract(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(
            {
                "contract_id": "table6-test-r36",
                "expected_sqlens_build_id": "sqlens-table6-test-r36",
                "expected_vector_so_sha256": SHA_A,
            }
        ),
        encoding="utf-8",
    )
    return path, {
        "path": str(path),
        "sha256": summary.sha256_file(path),
        "contract_id": "table6-test-r36",
        "expected_sqlens_build_id": "sqlens-table6-test-r36",
        "expected_vector_so_sha256": SHA_A,
    }


def selection_evidence(
    tmp_path: Path,
    *,
    include_unattainable: bool = False,
) -> summary.SelectionEvidence:
    _, release = release_contract(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    pairs = [
        summary.SelectionPair(
            dataset="amazon",
            dataset_id="amazon10m",
            target=0.90,
            pair_id=PAIR_ID,
            status=summary.SELECTED,
            stock=None,
            sqlens=None,
            stock_selection_sha=SHA_B,
            sqlens_selection_sha=SHA_C,
            stock_arm_sha=SHA_D,
            sqlens_arm_sha=SHA_E,
        )
    ]
    if include_unattainable:
        pairs.append(
            summary.SelectionPair(
                dataset="laion",
                dataset_id="laion25m",
                target=0.99,
                pair_id="laion:recall_0.990000000:unattainable",
                status=summary.UNATTAINABLE,
                stock=None,
                sqlens=None,
                stock_selection_sha="",
                sqlens_selection_sha="",
                stock_arm_sha="",
                sqlens_arm_sha="",
                stock_status=summary.UNATTAINABLE,
                sqlens_status=summary.UNATTAINABLE,
            )
        )
    return summary.SelectionEvidence(
        pairs=tuple(pairs),
        release=release,
        qualification_scope="global_min_predicate_lcb",
        bindings={
            "selection_csv_sha256": "1" * 64,
            "selection_plan_sha256": "2" * 64,
            "selection_manifest_sha256": "3" * 64,
        },
        config={
            "path": str(config),
            "sha256": summary.sha256_file(config),
        },
        required_grid={
            "path": str(tmp_path / "required-grid.json"),
            "sha256": "4" * 64,
            "cell_keys_sha256": "5" * 64,
        },
    )


def latency_evidence() -> summary.LatencyPairEvidence:
    by_filter = {
        f"filter_{index:02d}": {
            (request_no, str(100 + request_no)): {
                "stock_pgvector": [stock] * 3,
                "sqlens_full": [sqlens] * 3,
            }
            for request_no, (stock, sqlens) in enumerate(
                ((10.0, 5.0), (12.0, 6.0), (14.0, 7.0))
            )
        }
        for index in range(summary.EXPECTED_FILTERS)
    }
    stock = [
        value
        for clusters in by_filter.values()
        for arms in clusters.values()
        for value in arms["stock_pgvector"]
    ]
    sqlens = [
        value
        for clusters in by_filter.values()
        for arms in clusters.values()
        for value in arms["sqlens_full"]
    ]
    return summary.LatencyPairEvidence(
        dataset="amazon",
        target=0.90,
        pair_id=PAIR_ID,
        by_filter=by_filter,
        recall_by_arm={
            "stock_pgvector": [0.91] * len(stock),
            "sqlens_full": [0.92] * len(sqlens),
        },
        latency_by_arm={
            "stock_pgvector": stock,
            "sqlens_full": sqlens,
        },
        workload_sha256=SHA_A,
        filters_sha256=SHA_B,
        source_manifest_sha256=SHA_C,
    )


def throughput_evidence(
    *, workload_sha: str = SHA_A
) -> summary.ThroughputPairEvidence:
    return summary.ThroughputPairEvidence(
        dataset="amazon",
        target=0.90,
        pair_id=PAIR_ID,
        qps_by_arm={
            "stock_pgvector": 100.0,
            "sqlens_full": 180.0,
        },
        workload_sha256=workload_sha,
        source_manifest_sha256=SHA_D,
        protocol_fingerprint_sha256=SHA_E,
    )


def test_paired_stratified_bootstrap_is_deterministic_and_filter_weighted() -> None:
    evidence = latency_evidence()

    first = summary.paired_stratified_speedup(
        evidence.by_filter, samples=250, seed=17
    )
    second = summary.paired_stratified_speedup(
        evidence.by_filter, samples=250, seed=17
    )

    assert first == second
    center, low, high, wins = first
    assert center == pytest.approx(2.0)
    assert low == pytest.approx(2.0)
    assert high == pytest.approx(2.0)
    assert wins == 14


def test_paired_stratified_bootstrap_rejects_unpaired_filter_arrays() -> None:
    evidence = latency_evidence()
    evidence.by_filter["filter_03"][(0, "100")]["sqlens_full"] = [1.0, 2.0]

    with pytest.raises(summary.Table6SummaryError, match="strictly paired"):
        summary.paired_stratified_speedup(
            evidence.by_filter, samples=100, seed=5
        )


def test_query_cluster_bootstrap_does_not_treat_repeats_as_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    single_repeat = {
        f"filter_{filter_no:02d}": {
            (request_no, str(100 + request_no)): {
                "stock_pgvector": [stock],
                "sqlens_full": [sqlens],
            }
            for request_no, (stock, sqlens) in enumerate(
                ((8.0, 8.0), (24.0, 6.0), (18.0, 12.0))
            )
        }
        for filter_no in range(summary.EXPECTED_FILTERS)
    }
    repeated_three_times = {
        filter_name: {
            cluster_key: {
                arm: values * 3
                for arm, values in arms.items()
            }
            for cluster_key, arms in clusters.items()
        }
        for filter_name, clusters in single_repeat.items()
    }

    monkeypatch.setattr(summary, "EXPECTED_LATENCY_REPEATS", 1)
    single = summary.paired_stratified_speedup(
        single_repeat, samples=2000, seed=23
    )
    monkeypatch.setattr(summary, "EXPECTED_LATENCY_REPEATS", 3)
    repeated = summary.paired_stratified_speedup(
        repeated_three_times, samples=2000, seed=23
    )

    assert repeated == single
    assert repeated[0] == pytest.approx(
        ((8.0 + 24.0 + 18.0) / 3.0)
        / ((8.0 + 6.0 + 12.0) / 3.0)
    )
    assert repeated[1] < repeated[0] < repeated[2]
    assert repeated[3] == summary.EXPECTED_FILTERS


def test_query_cluster_bootstrap_requires_exact_repeat_coverage() -> None:
    evidence = latency_evidence()
    evidence.by_filter["filter_03"][(0, "100")]["stock_pgvector"].append(10.0)
    evidence.by_filter["filter_03"][(0, "100")]["sqlens_full"].append(5.0)

    with pytest.raises(summary.Table6SummaryError, match="repeats, expected"):
        summary.paired_stratified_speedup(
            evidence.by_filter, samples=100, seed=5
        )


def test_paired_stratified_bootstrap_rejects_unpaired_cluster_arms() -> None:
    evidence = latency_evidence()
    del evidence.by_filter["filter_03"][(0, "100")]["sqlens_full"]

    with pytest.raises(summary.Table6SummaryError, match="strictly paired"):
        summary.paired_stratified_speedup(
            evidence.by_filter, samples=100, seed=5
        )


def test_summary_emits_unattainable_status_without_fabricated_values(
    tmp_path: Path,
) -> None:
    selection = selection_evidence(tmp_path, include_unattainable=True)

    rows = summary.summarize(
        selection,
        {PAIR_ID: latency_evidence()},
        {PAIR_ID: throughput_evidence()},
        bootstrap_samples=100,
        bootstrap_seed=9,
    )

    selected, unattainable = rows
    assert selected["stock_recall"] == pytest.approx(0.91)
    assert selected["sqlens_recall"] == pytest.approx(0.92)
    assert selected["speedup_geomean"] == pytest.approx(2.0)
    assert selected["wins"] == 14
    assert selected["stock_qps"] == pytest.approx(100.0)
    assert selected["sqlens_qps"] == pytest.approx(180.0)
    assert unattainable["status"] == summary.UNATTAINABLE
    assert unattainable["status_detail"] == summary.UNATTAINABLE
    for field in (
        "stock_recall",
        "sqlens_recall",
        "stock_mean_latency_ms",
        "sqlens_mean_latency_ms",
        "stock_qps",
        "sqlens_qps",
        "speedup_geomean",
        "speedup_ci95_low",
        "speedup_ci95_high",
        "wins",
    ):
        assert unattainable[field] == ""


def test_summary_rejects_latency_throughput_workload_sha_drift(
    tmp_path: Path,
) -> None:
    with pytest.raises(summary.Table6SummaryError, match="workload SHA differs"):
        summary.summarize(
            selection_evidence(tmp_path),
            {PAIR_ID: latency_evidence()},
            {PAIR_ID: throughput_evidence(workload_sha=SHA_E)},
            bootstrap_samples=100,
            bootstrap_seed=9,
        )


def latency_loader_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mismatched_key: bool = False,
    arm_sha_drift: bool = False,
) -> tuple[
    Path,
    str,
    list[dict[str, object]],
    summary.SelectionEvidence,
]:
    monkeypatch.setattr(summary, "EXPECTED_REQUESTS", 2)
    monkeypatch.setattr(summary, "EXPECTED_LATENCY_REPEATS", 2)
    monkeypatch.setattr(summary, "EXPECTED_FILTERS", 2)
    selection = selection_evidence(tmp_path)

    workload = tmp_path / "workload.csv"
    write_csv(
        workload,
        [
            {"request_no": 0, "query_id": 100},
            {"request_no": 1, "query_id": 101},
        ],
    )
    filters = tmp_path / "filters.csv"
    write_csv(filters, [{"filter_name": "f0"}, {"filter_name": "f1"}])
    raw = tmp_path / "latency.csv"
    raw_rows: list[dict[str, object]] = []
    for repeat in range(2):
        for request_no in range(2):
            for mode in summary.MODE_TO_ARM:
                query_id = 100 + request_no
                if (
                    mismatched_key
                    and repeat == 1
                    and request_no == 1
                    and mode == "design1_bloom_bfs_layout_d3"
                ):
                    query_id = 999
                raw_rows.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "request_no": request_no,
                        "query_id": query_id,
                        "filter_name": f"f{request_no}",
                        "recall": 0.95,
                        "end_to_end_ms": (
                            10.0 if mode == "original" else 5.0
                        ),
                        "error": "",
                    }
                )
    write_csv(raw, raw_rows)

    manifest_path = tmp_path / "latency-run.json"
    manifest = {
        "artifact_type": "sqlens_figure5_matched_latency_run",
        "selector": dict(selection.bindings),
        "release_contract": dict(selection.release),
        "frontier_config": dict(selection.config),
        "execution": {
            "requests": 2,
            "repeats": 2,
            "expected_rows_per_pair": 8,
            "execution_order": "paired_interleaved",
        },
        "schedule": [
            {
                "pair_id": PAIR_ID,
                "dataset": "amazon",
                "target_recall": 0.90,
                "stock_config": {"config_sha256": SHA_B},
                "sqlens_config": {"config_sha256": SHA_C},
                "input_bindings": {
                    "measurement_workload_csv": {
                        "path": str(workload),
                        "sha256": summary.sha256_file(workload),
                    },
                    "filters_csv": {
                        "path": str(filters),
                        "sha256": summary.sha256_file(filters),
                    },
                },
                "raw": str(raw),
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    converted: list[dict[str, object]] = []
    for arm, arm_sha in (
        ("stock_pgvector", SHA_D),
        ("sqlens_full", SHA_E),
    ):
        for repeat in range(2):
            converted.append(
                {
                    "config_id": PAIR_ID,
                    "arm_id": arm,
                    "repeat_id": repeat,
                    "config_sha256": (
                        SHA_A
                        if arm_sha_drift
                        and arm == "sqlens_full"
                        and repeat == 1
                        else arm_sha
                    ),
                    "dataset": "amazon10m",
                }
            )
    return (
        manifest_path,
        summary.sha256_file(manifest_path),
        converted,
        selection,
    )


def test_latency_loader_rejects_nonidentical_paired_request_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source_sha, converted, selection = latency_loader_fixture(
        tmp_path, monkeypatch, mismatched_key=True
    )

    with pytest.raises(summary.Table6SummaryError, match="strictly paired"):
        summary.load_latency(manifest, source_sha, converted, selection)


def test_latency_loader_rejects_arm_config_sha_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source_sha, converted, selection = latency_loader_fixture(
        tmp_path, monkeypatch, arm_sha_drift=True
    )

    with pytest.raises(summary.Table6SummaryError, match="arm SHA differs"):
        summary.load_latency(manifest, source_sha, converted, selection)


def converter_binding_fixture(
    tmp_path: Path,
    *,
    experiment_kind: str,
    service: bool = False,
) -> tuple[Path, Path, Path, Path | None]:
    output = tmp_path / f"{experiment_kind}.csv"
    output.write_text("field\nvalue\n", encoding="utf-8")
    source = tmp_path / f"{experiment_kind}-source.json"
    source.write_text("{}\n", encoding="utf-8")
    service_path = tmp_path / "service.csv" if service else None
    if service_path is not None:
        service_path.write_text("field\nvalue\n", encoding="utf-8")
    binding_path = output.with_suffix(output.suffix + ".manifest.json")
    payload: dict[str, object] = {
        "artifact_type": "sqlens_figure5_converter_binding",
        "experiment_kind": experiment_kind,
        "status": "complete",
        "artifact_valid": True,
        "requested_slice_complete": True,
        "full_release_complete": True,
        "paper_eligible": True,
        "converter_binding": {
            "output": {
                "path": str(output),
                "sha256": summary.sha256_file(output),
            },
            "source_manifest": {
                "path": str(source),
                "sha256": summary.sha256_file(source),
            },
        },
    }
    if service_path is not None:
        payload["service_aggregate"] = {
            "path": str(service_path),
            "sha256": summary.sha256_file(service_path),
            "qps_source": summary.throughput.THROUGHPUT_SOURCE,
            "qps_from_latency_forbidden": True,
        }
    binding_path.write_text(json.dumps(payload), encoding="utf-8")
    return output, source, binding_path, service_path


def test_converter_binding_rejects_output_sha_drift(tmp_path: Path) -> None:
    output, _, binding, _ = converter_binding_fixture(
        tmp_path, experiment_kind="latency"
    )
    output.write_text("field\nchanged\n", encoding="utf-8")

    with pytest.raises(summary.Table6SummaryError, match="SHA drifted"):
        summary.audit_converter_binding(
            output, binding, experiment_kind="latency"
        )


def test_throughput_loader_rejects_non_c16_protocol(tmp_path: Path) -> None:
    repeat_csv, source_path, binding_path, service_csv = (
        converter_binding_fixture(
            tmp_path, experiment_kind="throughput", service=True
        )
    )
    assert service_csv is not None
    selection = selection_evidence(tmp_path)
    source = {
        "artifact_type": "sqlens_figure5_matched_throughput_run",
        "status": "complete",
        "artifact_valid": True,
        "requested_slice_complete": True,
        "full_release_complete": True,
        "paper_eligible": True,
        "protocol_slice": "fixed-target-c16-q10k-r3",
        "protocol_fingerprint_sha256": SHA_A,
        "release_contract": dict(selection.release),
        "selector": dict(selection.bindings),
        "execution": {
            "requests_per_arm_repeat": 10_000,
            "repeats": 3,
            "expected_repeat_rows_per_cell": 6,
            "client_grid": [1],
            "throughput_source": summary.throughput.THROUGHPUT_SOURCE,
            "throughput_formula": (
                "completed_queries / barrier_wall_clock_seconds"
            ),
            "qps_from_latency_forbidden": True,
        },
        "full_release_scope": {
            "requested": True,
            "required_clients": [16],
            "required_repeats": 3,
        },
    }
    source_path.write_text(json.dumps(source), encoding="utf-8")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["protocol_slice"] = source["protocol_slice"]
    binding["converter_binding"]["source_manifest"]["sha256"] = (
        summary.sha256_file(source_path)
    )
    binding_path.write_text(json.dumps(binding), encoding="utf-8")

    with pytest.raises(
        summary.Table6SummaryError, match="fixed-target c16/q10k/r3"
    ):
        summary.load_throughput(
            repeat_csv,
            service_csv,
            binding_path,
            selection,
        )


def test_manifest_selector_join_rejects_sha_drift(tmp_path: Path) -> None:
    selection = selection_evidence(tmp_path)
    manifest = {
        "selector": {
            **selection.bindings,
            "selection_plan_sha256": SHA_A,
        }
    }

    with pytest.raises(
        summary.Table6SummaryError, match="does not match the supplied selector"
    ):
        summary._manifest_selector_gate(
            manifest, selection, label="test manifest"
        )


def test_manifest_config_gate_binds_required_grid_config(
    tmp_path: Path,
) -> None:
    selection = selection_evidence(tmp_path)
    assert selection.config is not None

    path, observed_sha = summary._manifest_config_gate(
        dict(selection.config),
        selection,
        base=tmp_path,
        label="test manifest",
    )

    assert path == Path(selection.config["path"]).resolve()
    assert observed_sha == selection.config["sha256"]

    other = tmp_path / "other-config.json"
    other.write_text('{"different": true}\n', encoding="utf-8")
    with pytest.raises(
        summary.Table6SummaryError,
        match="differs from required grid",
    ):
        summary._manifest_config_gate(
            {
                "path": str(other),
                "sha256": summary.sha256_file(other),
            },
            selection,
            base=tmp_path,
            label="test manifest",
        )
