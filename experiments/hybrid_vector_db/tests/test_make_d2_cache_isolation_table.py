from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "paper/scripts/make_d2_cache_isolation_table.py"
SPEC = importlib.util.spec_from_file_location("make_d2_cache_isolation_table", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_row(
    index: int, regime: str, *, include_r33_components: bool = True
) -> dict[str, object]:
    selectivities = (50, 45, 40, 35, 30, 25, 20, 15, 10, 5.88, 2, 1, 0.61, 0.21)
    selectivity = selectivities[index]
    source_e2e = 20.0 + index
    bfs_e2e = 15.0 + index
    read_source = 4.0
    read_bfs = 2.0
    distance_source = distance_bfs = 3.0
    hnsw_source = 5.0
    hnsw_bfs = 4.0
    source_loads = 1000.0 + index
    bfs_loads = source_loads
    source_reads = 800.0 + index
    bfs_reads = 500.0 + index
    row: dict[str, object] = {
        "filter_name": f"filter_{index}_with_underscore",
        "selectivity": str(selectivity),
        "cache_regime": regime,
        "queries": 100 if regime == "warm_resident" else 1,
        "repeats": 5,
        "paired_cluster_unit": "query" if regime == "warm_resident" else "cold_eviction_block",
        "paired_clusters": 100 if regime == "warm_resident" else 5,
        "d1_bfs_minus_source_query_cluster_mean_ms": bfs_e2e - source_e2e,
        "d1_bfs_minus_source_ci95_low_ms": -6.0,
        "d1_bfs_minus_source_ci95_high_ms": -4.0,
        "d1_bfs_speedup_over_source": source_e2e / bfs_e2e,
    }
    for arm, e2e, read_total, distance, hnsw, idx_reads in (
        ("d1_source", source_e2e, read_source, distance_source, hnsw_source, source_reads),
        ("d1_bfs", bfs_e2e, read_bfs, distance_bfs, hnsw_bfs, bfs_reads),
    ):
        prefix = arm + "_"
        row.update(
            {
                prefix + "end_to_end_ms_mean": e2e,
                prefix + "activation_ms_mean": 0.5,
                prefix + "query_latency_ms_mean": e2e - 0.5,
                prefix + "vector_search_ms_mean": e2e - 1.0,
                prefix + "recall_mean": 0.95,
                prefix + "idx_blks_read_mean": idx_reads,
                prefix + "index_page_loads_mean": source_loads,
                prefix + "distance_compute_count_mean": 400.0,
                prefix + "index_readbuffer_calls_mean": source_loads,
                prefix + "index_readbuffer_shared_read_calls_mean": 100.0,
                prefix + "index_readbuffer_shared_hit_calls_mean": source_loads - 100.0,
                prefix + "index_readbuffer_unclassified_calls_mean": 0.0,
                prefix + "distance_compute_timed_calls_mean": 400.0,
                prefix + "index_readbuffer_ms_mean": read_total,
                prefix + "index_readbuffer_shared_read_ms_mean": read_total / 2.0,
                prefix + "index_readbuffer_shared_hit_ms_mean": read_total / 2.0,
                prefix + "index_readbuffer_unclassified_ms_mean": 0.0,
                prefix + "distance_compute_ms_mean": distance,
                prefix + "hnsw_remaining_ms_mean": hnsw,
                prefix + "hnsw_remaining_ms_is_residual_rate": 1.0,
                prefix + "index_readbuffer_timing_scope": "hnsw scan ReadBuffer wall time",
                prefix + "index_readbuffer_classification_scope": "pgBufferUsage delta",
                prefix + "distance_compute_timing_scope": "distance callback wall time",
                prefix + "hnsw_remaining_scope": "callback residual",
                prefix + "profile_timer_overhead_scope": "included",
            }
        )
    if not include_r33_components:
        r33_fragments = (
            "index_readbuffer",
            "distance_compute_timed_calls",
            "distance_compute_ms",
            "hnsw_remaining",
            "profile_timer_overhead_scope",
        )
        for key in tuple(row):
            if any(fragment in key for fragment in r33_fragments):
                del row[key]
    return row


def write_summary(
    path: Path, regime: str, *, include_r33_components: bool = True
) -> list[dict[str, object]]:
    rows = [
        summary_row(index, regime, include_r33_components=include_r33_components)
        for index in range(14)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def manifest_payload(
    summary: Path, regime: str, *, legacy_warm: bool = False
) -> dict[str, object]:
    measurement: dict[str, object] = {
        "queries": 100 if regime == "warm_resident" else 1,
        "repeats": 5,
    }
    if regime == "cold_io":
        measurement.update(
            {
                "cold_protocol_semantics_version": 4,
                "query_slice_policy": "distinct_contiguous_per_eviction_block",
                "total_distinct_queries": 5,
                "block_query_slices": [
                    {"control_repeat": repeat, "query_offset": 100 + repeat, "queries": 1}
                    for repeat in range(5)
                ],
            }
        )
    protocol: dict[str, object] = {
        "name": "sqlens-d2-cache-isolation-v3" if legacy_warm else "sqlens-d2-cache-isolation-v5",
        "version": 3 if legacy_warm else 5,
        "cache_regime": regime,
        "measurement": measurement,
    }
    if not legacy_warm:
        protocol["profile_contract"] = {"required_profile_semantics_min": 12}
    payload: dict[str, object] = {
        "status": "complete",
        "artifact_valid": True,
        "paired_gate_passed": True,
        "protocol": protocol,
        "outputs": {
            "summary": {
                "path": str(summary),
                "rows": 14,
                "sha256": sha256(summary),
            }
        },
    }
    if regime == "cold_io":
        protocol["arms"] = [
            {"index_role": "source", "expected_index": "public.source_idx"},
            {"index_role": "bfs", "expected_index": "public.bfs_idx"},
        ]
        payload["index_identities_start"] = {
            "public.source_idx": {"name": "public.source_idx", "size_bytes": 1000},
            "public.bfs_idx": {"name": "public.bfs_idx", "size_bytes": 1010},
        }
    return payload


def overhead_payload() -> dict[str, object]:
    return {
        "artifact_valid": True,
        "preparation": {
            "artifact": "sqlens-same-graph-bfs-clone-v1",
            "timing": {
                "schema": "sqlens-bfs-rewrite-overhead-v1",
                "clone_creation_transaction": {"status": "measured", "elapsed_ms": 12000.0},
                "graph_proof": {"status": "measured", "elapsed_ms": 500.0},
            },
            "storage": {
                "schema": "sqlens-bfs-rewrite-storage-v1",
                "source": {"bytes": 1000, "blocks": 1},
                "clone": {"bytes": 1010, "blocks": 1},
                "clone_to_source_storage_ratio": 1.01,
            },
        },
    }


def fixture_artifacts(
    tmp_path: Path, *, legacy_warm: bool = False
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "warm_summary": tmp_path / "warm.csv",
        "cold_summary": tmp_path / "cold.csv",
        "warm_manifest": tmp_path / "warm.json",
        "cold_manifest": tmp_path / "cold.json",
        "overhead": tmp_path / "overhead.json",
        "table": tmp_path / "table.tex",
        "breakdown": tmp_path / "breakdown.tex",
    }
    write_summary(
        paths["warm_summary"],
        "warm_resident",
        include_r33_components=not legacy_warm,
    )
    write_summary(paths["cold_summary"], "cold_io")
    paths["warm_manifest"].write_text(
        json.dumps(
            manifest_payload(
                paths["warm_summary"], "warm_resident", legacy_warm=legacy_warm
            )
        )
    )
    paths["cold_manifest"].write_text(
        json.dumps(manifest_payload(paths["cold_summary"], "cold_io"))
    )
    paths["overhead"].write_text(json.dumps(overhead_payload()))
    return paths


def rewrite_manifest(path: Path, mutate) -> None:
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


def rewrite_summary(summary: Path, manifest: Path, mutate) -> None:
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mutate(rows)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rewrite_manifest(
        manifest,
        lambda value: value["outputs"]["summary"].update({"sha256": sha256(summary)}),
    )


def run_generator(
    paths: dict[str, Path], *, separate: bool = True, include_overhead: bool = True
) -> None:
    arguments = [
        "--warm-manifest",
        str(paths["warm_manifest"]),
        "--cold-manifest",
        str(paths["cold_manifest"]),
        "--out-table",
        str(paths["table"]),
    ]
    if include_overhead:
        arguments[4:4] = ["--bfs-overhead-json", str(paths["overhead"])]
    if separate:
        arguments += ["--out-breakdown", str(paths["breakdown"])]
    MODULE.run(MODULE.parse_args(arguments))


def test_success_generates_complete_main_and_breakdown_tables(tmp_path: Path) -> None:
    paths = fixture_artifacts(tmp_path)
    run_generator(paths)
    main = paths["table"].read_text()
    breakdown = paths["breakdown"].read_text()
    assert main.count(r"filter\_") == 14
    assert "Warm resident" in main
    assert "Cold I/O" in main
    assert r"\begin{tabular}{rlrrrrrrrrrrr}" in main
    assert "95\\% paired" in main
    assert "12.0~s" in main
    assert "BFS/source index-storage ratio of 1.010" in main
    assert "Buffer-read events" in main
    assert "Physical-read RB (s)" in main
    assert "universal significance claim" in main
    assert breakdown.count("Warm &") == 4
    assert breakdown.count("Cold &") == 4
    forbidden = ("staged", "incomplete", "cherry-picked")
    assert not any(word in (main + breakdown).lower() for word in forbidden)


def test_complete_legacy_warm_uses_only_aggregate_fields(tmp_path: Path) -> None:
    paths = fixture_artifacts(tmp_path, legacy_warm=True)
    run_generator(paths)
    main = paths["table"].read_text(encoding="utf-8")
    breakdown = paths["breakdown"].read_text(encoding="utf-8")
    assert main.count(r"filter\_") == 14
    assert breakdown.count("Cold &") == 4
    assert "Warm &" not in breakdown
    assert "legacy v3" in breakdown


def test_legacy_warm_without_numeric_protocol_version_is_accepted(
    tmp_path: Path,
) -> None:
    paths = fixture_artifacts(tmp_path, legacy_warm=True)
    rewrite_manifest(paths["warm_manifest"], lambda value: value["protocol"].pop("version"))
    run_generator(paths)


def test_overhead_is_optional_and_storage_uses_cold_start_identities(
    tmp_path: Path,
) -> None:
    paths = fixture_artifacts(tmp_path, legacy_warm=True)
    run_generator(paths, include_overhead=False)
    main = paths["table"].read_text(encoding="utf-8")
    assert "BFS rewrite took" not in main
    assert "BFS/source index-storage ratio of 1.010" in main


def test_breakdown_is_appended_when_no_separate_path(tmp_path: Path) -> None:
    paths = fixture_artifacts(tmp_path)
    run_generator(paths, separate=False)
    output = paths["table"].read_text()
    assert output.count(r"\begin{table*}") == 2


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("status", "failed"),
        ("artifact_valid", False),
        ("paired_gate_passed", False),
    ),
)
def test_manifest_completion_gates_fail_closed(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    paths = fixture_artifacts(tmp_path)
    rewrite_manifest(paths["warm_manifest"], lambda value: value.update({field: bad_value}))
    with pytest.raises(MODULE.TableError):
        run_generator(paths)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    (
        (("protocol", "name"), "legacy"),
        (("protocol", "version"), 4),
        (("protocol", "profile_contract", "required_profile_semantics_min"), 11),
    ),
)
def test_protocol_gates_fail_closed(
    tmp_path: Path, path: tuple[str, ...], bad_value: object
) -> None:
    paths = fixture_artifacts(tmp_path)

    def mutate(value):
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad_value

    rewrite_manifest(paths["warm_manifest"], mutate)
    with pytest.raises(MODULE.TableError):
        run_generator(paths)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    (
        (("protocol", "name"), "sqlens-d2-cache-isolation-v4"),
        (("protocol", "version"), 4),
    ),
)
def test_legacy_warm_protocol_is_narrowly_allowlisted(
    tmp_path: Path, path: tuple[str, ...], bad_value: object
) -> None:
    paths = fixture_artifacts(tmp_path, legacy_warm=True)

    def mutate(value):
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad_value

    rewrite_manifest(paths["warm_manifest"], mutate)
    with pytest.raises(MODULE.TableError, match="protocol"):
        run_generator(paths)


def test_summary_hash_and_row_count_are_bound(tmp_path: Path) -> None:
    paths = fixture_artifacts(tmp_path)
    paths["warm_summary"].write_text(paths["warm_summary"].read_text() + "\n")
    with pytest.raises(MODULE.TableError, match="sha256 mismatch"):
        run_generator(paths)
    paths = fixture_artifacts(tmp_path / "second")
    rewrite_manifest(
        paths["warm_manifest"],
        lambda value: value["outputs"]["summary"].update({"rows": 13}),
    )
    with pytest.raises(MODULE.TableError, match="14"):
        run_generator(paths)


def test_cold_requires_distinct_five_cluster_protocol(tmp_path: Path) -> None:
    paths = fixture_artifacts(tmp_path)
    rewrite_manifest(
        paths["cold_manifest"],
        lambda value: value["protocol"]["measurement"].update(
            {"query_slice_policy": "repeated_slice_per_eviction_block"}
        ),
    )
    with pytest.raises(MODULE.TableError, match="distinct"):
        run_generator(paths)


def test_warm_and_cold_filter_contract_must_match(tmp_path: Path) -> None:
    paths = fixture_artifacts(tmp_path)
    rewrite_summary(
        paths["cold_summary"],
        paths["cold_manifest"],
        lambda rows: rows[0].update({"filter_name": "different_filter"}),
    )
    with pytest.raises(MODULE.TableError, match="filter sets differ"):
        run_generator(paths)


def test_warm_and_cold_selectivity_must_match_but_recall_cohorts_may_differ(
    tmp_path: Path,
) -> None:
    paths = fixture_artifacts(tmp_path)
    rewrite_summary(
        paths["cold_summary"],
        paths["cold_manifest"],
        lambda rows: rows[0].update({"selectivity": "49.9"}),
    )
    with pytest.raises(MODULE.TableError, match="selectivity differs"):
        run_generator(paths)
    paths = fixture_artifacts(tmp_path / "recall")
    rewrite_summary(
        paths["cold_summary"],
        paths["cold_manifest"],
        lambda rows: rows[0].update(
            {"d1_source_recall_mean": "0.94", "d1_bfs_recall_mean": "0.94"}
        ),
    )
    run_generator(paths)
    rendered = paths["table"].read_text(encoding="utf-8")
    assert "0.950/0.940" in rendered


def test_source_and_bfs_recall_and_ci_are_fail_closed(tmp_path: Path) -> None:
    paths = fixture_artifacts(tmp_path)
    rewrite_summary(
        paths["warm_summary"],
        paths["warm_manifest"],
        lambda rows: rows[0].update({"d1_bfs_recall_mean": "0.94"}),
    )
    with pytest.raises(MODULE.TableError, match="recall differs"):
        run_generator(paths)
    paths = fixture_artifacts(tmp_path / "ci")
    rewrite_summary(
        paths["warm_summary"],
        paths["warm_manifest"],
        lambda rows: rows[0].update(
            {
                "d1_bfs_minus_source_ci95_low_ms": "1",
                "d1_bfs_minus_source_ci95_high_ms": "-1",
            }
        ),
    )
    with pytest.raises(MODULE.TableError, match="CI is inverted"):
        run_generator(paths)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("d1_source_index_readbuffer_shared_hit_ms_mean", "99", "timing classes"),
        ("d1_source_distance_compute_timed_calls_mean", "399", "distance calls"),
        ("d1_source_hnsw_remaining_ms_is_residual_rate", "0", "non-residual"),
        ("d1_source_distance_compute_ms_mean", "", "missing"),
    ),
)
def test_r33_component_gates_fail_closed(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    paths = fixture_artifacts(tmp_path)
    rewrite_summary(
        paths["cold_summary"],
        paths["cold_manifest"],
        lambda rows: rows[0].update({field: value}),
    )
    with pytest.raises(MODULE.TableError, match=message):
        run_generator(paths)


def test_negative_executor_residual_fails_closed(tmp_path: Path) -> None:
    paths = fixture_artifacts(tmp_path)
    rewrite_summary(
        paths["warm_summary"],
        paths["warm_manifest"],
        lambda rows: rows[0].update(
            {
                "d1_source_end_to_end_ms_mean": "10",
                "d1_bfs_speedup_over_source": str(10 / 15),
                "d1_bfs_minus_source_query_cluster_mean_ms": "5",
            }
        ),
    )
    with pytest.raises(MODULE.TableError, match="negative executor"):
        run_generator(paths)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("preparation", "timing", "schema"), "old-schema", "timing schema"),
        (
            ("preparation", "timing", "clone_creation_transaction", "status"),
            "not_measured",
            "must be measured",
        ),
        (
            ("preparation", "timing", "graph_proof", "elapsed_ms"),
            0,
            "must be positive",
        ),
        (
            ("preparation", "storage", "clone_to_source_storage_ratio"),
            1.2,
            "does not match",
        ),
    ),
)
def test_bfs_overhead_gates_fail_closed(
    tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    paths = fixture_artifacts(tmp_path)
    payload = json.loads(paths["overhead"].read_text())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    paths["overhead"].write_text(json.dumps(payload))
    with pytest.raises(MODULE.TableError, match=message):
        run_generator(paths)
