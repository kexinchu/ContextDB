from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import figure5_frontier_artifact as artifact


SHA_RELEASE = "a" * 64
TRACE_SHA = {
    "amazon10m": "b" * 64,
    "yfcc10m": "c" * 64,
    "laion25m": "d" * 64,
}
CONFIG_SHA = {
    ("latency", "stock_pgvector"): "1" * 64,
    ("latency", "sqlens_full"): "2" * 64,
    ("throughput", "stock_pgvector"): "3" * 64,
    ("throughput", "sqlens_full"): "4" * 64,
}


def row(
    dataset: str,
    kind: str,
    arm: str,
    repeat_id: int,
    *,
    config_id: str = "ef100",
    clients: int | None = None,
    recall: float | None = None,
    latency_ms: float | None = None,
    wall_seconds: float | None = None,
) -> dict[str, object]:
    if clients is None:
        clients = 1 if kind == "latency" else 8
    if recall is None:
        recall = 0.80 if config_id == "ef100" else 0.90
    if latency_ms is None:
        latency_ms = 100.0 if config_id == "ef100" else 200.0
    if arm == "sqlens_full":
        latency_ms *= 0.8
    if wall_seconds is None:
        wall_seconds = 20.0 if arm == "stock_pgvector" else 10.0
    mode = artifact.ARM_MODES[arm]
    throughput = (
        artifact.EXPECTED_REQUESTS / wall_seconds if kind == "throughput" else ""
    )
    return {
        "schema_version": artifact.SCHEMA_VERSION,
        "run_id": f"{dataset}-{kind}-run",
        "dataset": dataset,
        "experiment_kind": kind,
        "arm_id": arm,
        "mode_id": mode,
        "config_id": config_id,
        "config_sha256": CONFIG_SHA[(kind, arm)],
        "release_identity_sha256": SHA_RELEASE,
        "clients": clients,
        "repeat_id": repeat_id,
        "request_trace_sha256": TRACE_SHA[dataset],
        "requests": artifact.EXPECTED_REQUESTS,
        "unique_queries": artifact.EXPECTED_REQUESTS,
        "completed_queries": artifact.EXPECTED_REQUESTS,
        "error_count": 0,
        "wall_clock_seconds": wall_seconds,
        "recall_mean": recall,
        "recall_ci95_low": max(0.0, recall - 0.01),
        "recall_ci95_high": min(1.0, recall + 0.01),
        "latency_mean_ms": latency_ms,
        "latency_p95_ms": latency_ms * 1.2,
        "latency_p99_ms": latency_ms * 1.4,
        "throughput_qps": throughput,
        "throughput_ci95_low": "",
        "throughput_ci95_high": "",
        "throughput_source": (
            artifact.MEASURED_THROUGHPUT_SOURCE if kind == "throughput" else ""
        ),
        "status": "valid",
    }


def complete_rows(*, configs: tuple[str, ...] = ("ef100",)) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in artifact.EXPECTED_DATASETS:
        for kind in artifact.EXPERIMENT_KINDS:
            repeats = artifact.MIN_REPEATS[kind]
            for config_id in configs:
                for repeat_id in range(repeats):
                    for arm in artifact.ARM_MODES:
                        rows.append(row(dataset, kind, arm, repeat_id, config_id=config_id))
    return rows


def write_inputs(
    root: Path,
    rows: list[dict[str, object]],
    *,
    extra_field: str | None = None,
) -> tuple[Path, Path]:
    paths = {
        "latency": root / "latency.csv",
        "throughput": root / "throughput.csv",
    }
    fields = list(artifact.REPEAT_FIELDS)
    if extra_field is not None:
        fields.append(extra_field)
    for kind, path in paths.items():
        selected = [item for item in rows if item["experiment_kind"] == kind]
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for item in selected:
                output = dict(item)
                if extra_field is not None:
                    output[extra_field] = ""
                writer.writerow(output)
    return paths["latency"], paths["throughput"]


def prepare_formal_inputs(
    root: Path,
    rows: list[dict[str, object]],
    *,
    extra_field: str | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Create the same release/converter bindings required by the finalizer."""
    latency, throughput = write_inputs(root, rows, extra_field=extra_field)
    contract = root / "release-contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract_id": "sigmod-test-r1",
                "expected_sqlens_build_id": "sqlens-test-build",
                "expected_vector_so_sha256": "9" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()

    # Converter output rows bind the release contract SHA, not a free-standing
    # opaque identity.  Rewrite only this fixture field before publishing the
    # output binding sidecars.
    for path in (latency, throughput):
        with path.open(newline="", encoding="utf-8") as source:
            fieldnames = list(csv.DictReader(source).fieldnames or [])
            source.seek(0)
            reader = csv.DictReader(source)
            output_rows = []
            for item in reader:
                if item["release_identity_sha256"] == SHA_RELEASE:
                    item["release_identity_sha256"] = contract_sha
                output_rows.append(item)
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    release = {
        "path": str(contract.resolve()),
        "sha256": contract_sha,
        "contract_id": "sigmod-test-r1",
        "expected_sqlens_build_id": "sqlens-test-build",
        "expected_vector_so_sha256": "9" * 64,
    }
    bindings: list[Path] = []
    for kind, output in (("latency", latency), ("throughput", throughput)):
        audited_run = root / f"{kind}-audited-run.json"
        audited_run.write_text(
            json.dumps(
                {
                    "artifact_type": "sqlens_figure5_matched_run",
                    "status": "complete",
                    "artifact_valid": True,
                    "full_release_complete": True,
                    "paper_eligible": True,
                    "release_contract": release,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        binding = root / f"{kind}-converter-binding.json"
        binding.write_text(
            json.dumps(
                {
                    "artifact_type": "sqlens_figure5_converter_binding",
                    "status": "complete",
                    "artifact_valid": True,
                    "full_release_complete": True,
                    "paper_eligible": True,
                    "release_contract": release,
                    "converter_binding": {
                        "source_manifest": {
                            "path": str(audited_run.resolve()),
                            "sha256": hashlib.sha256(audited_run.read_bytes()).hexdigest(),
                        },
                        "output": {
                            "path": str(output.resolve()),
                            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                            "experiment_kind": kind,
                        },
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        bindings.append(binding)
    # The formal frontier uses the same config grid in both experiments.  The
    # fixture's original hashes intentionally distinguish the input families;
    # normalize them here to model a shared frozen config binding.
    latency_config_sha = {}
    for item in rows:
        if item["experiment_kind"] == "latency":
            latency_config_sha[(item["arm_id"], item["config_id"])] = item[
                "config_sha256"
            ]
    for path in (throughput,):
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fieldnames = list(reader.fieldnames or [])
            output_rows = []
            for item in reader:
                key = (item["arm_id"], item["config_id"])
                expected_fixture_hash = CONFIG_SHA.get(
                    ("throughput", item["arm_id"])
                )
                if key in latency_config_sha and item["config_sha256"] == expected_fixture_hash:
                    item["config_sha256"] = latency_config_sha[key]
                output_rows.append(item)
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
    # The throughput binding was created before the config normalization.
    throughput_binding = bindings[1]
    binding_payload = json.loads(throughput_binding.read_text(encoding="utf-8"))
    binding_payload["converter_binding"]["output"]["sha256"] = hashlib.sha256(
        throughput.read_bytes()
    ).hexdigest()
    throughput_binding.write_text(
        json.dumps(binding_payload, sort_keys=True), encoding="utf-8"
    )
    return latency, throughput, contract, bindings[0], bindings[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def build(tmp_path: Path, rows: list[dict[str, object]]):
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, rows)
    )
    return artifact.build_artifact(
        [latency],
        [throughput],
        release_contract_path=contract,
        latency_run_manifest=latency_binding,
        throughput_run_manifest=throughput_binding,
    )


def build_with_extra(
    tmp_path: Path, rows: list[dict[str, object]], extra_field: str
):
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, rows, extra_field=extra_field)
    )
    return artifact.build_artifact(
        [latency],
        [throughput],
        release_contract_path=contract,
        latency_run_manifest=latency_binding,
        throughput_run_manifest=throughput_binding,
    )


def test_execute_publishes_sha_bound_three_dataset_artifact(tmp_path: Path) -> None:
    repeat_bytes, point_bytes, manifest = build(tmp_path, complete_rows())
    outputs = artifact.publish_artifact(tmp_path / "out", repeat_bytes, point_bytes, manifest)

    assert manifest["artifact_valid"] is True
    assert manifest["paper_eligible"] is True
    assert set(manifest["datasets"]) == set(artifact.EXPECTED_DATASETS)
    assert manifest["methods"] == {
        "stock_pgvector": {
            "arm_id": "stock_pgvector",
            "mode_id": "original",
        },
        "sqlens_full": {
            "arm_id": "sqlens_full",
            "mode_id": "design1_bloom_bfs_layout_d3",
        },
    }
    assert hashlib.sha256(outputs["repeats"].read_bytes()).hexdigest() == (
        manifest["outputs"]["repeats"]["sha256"]
    )
    assert hashlib.sha256(outputs["points"].read_bytes()).hexdigest() == (
        manifest["outputs"]["points"]["sha256"]
    )
    on_disk = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert len(read_csv(outputs["repeats"])) == len(complete_rows())
    assert len(read_csv(outputs["points"])) == 3 * 2 * 2


def test_publish_rejects_unbound_payload_and_restores_old_artifact_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeat_bytes, point_bytes, manifest = build(tmp_path, complete_rows())
    output_dir = tmp_path / "out"
    outputs = artifact.publish_artifact(output_dir, repeat_bytes, point_bytes, manifest)
    previous = {name: path.read_bytes() for name, path in outputs.items()}

    with pytest.raises(artifact.Figure5ArtifactError, match="does not match"):
        artifact.publish_artifact(output_dir, repeat_bytes + b"\n", point_bytes, manifest)
    assert {name: path.read_bytes() for name, path in outputs.items()} == previous

    real_replace = artifact.os.replace
    failed = False

    def fail_once(source: object, destination: object) -> None:
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not failed
            and source_path.name == "figure5_points.csv"
            and destination_path.parent == output_dir
        ):
            failed = True
            raise OSError("injected publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(artifact.os, "replace", fail_once)
    with pytest.raises(OSError, match="injected"):
        artifact.publish_artifact(output_dir, repeat_bytes, point_bytes, manifest)
    assert {name: path.read_bytes() for name, path in outputs.items()} == previous


@pytest.mark.parametrize(
    ("arm_id", "mode_id"),
    [
        ("sqlens_d1", "design1_bloom"),
        ("sqlens_full", "design1_bloom"),
        ("sqlens_full", "design1_bloom_bfs_layout"),
    ],
)
def test_rejects_partial_or_wrong_sqlens_mode(
    tmp_path: Path,
    arm_id: str,
    mode_id: str,
) -> None:
    rows = complete_rows()
    target = next(item for item in rows if item["arm_id"] == "sqlens_full")
    target["arm_id"] = arm_id
    target["mode_id"] = mode_id
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, rows)
    )
    with pytest.raises(artifact.Figure5ArtifactError, match="partial|requires|unsupported"):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )


def test_rejects_derived_qps_field_and_source(tmp_path: Path) -> None:
    rows = complete_rows()
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(
            tmp_path,
            rows,
            extra_field="single_client_throughput_qps",
        )
    )
    with pytest.raises(artifact.Figure5ArtifactError, match="forbidden"):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )

    target = next(item for item in rows if item["experiment_kind"] == "throughput")
    target["throughput_source"] = "derived_from_latency"
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, rows)
    )
    with pytest.raises(artifact.Figure5ArtifactError, match="derived QPS is forbidden"):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )


def test_rejects_legacy_method_field_that_contradicts_full_sqlens(tmp_path: Path) -> None:
    rows = complete_rows()
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, rows, extra_field="legacy_method")
    )
    with throughput.open(newline="", encoding="utf-8") as source:
        source_rows = list(csv.DictReader(source))
        fields = list(source_rows[0])
    source_rows[0]["legacy_method"] = "D1+D2"
    with throughput.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_rows)
    with pytest.raises(artifact.Figure5ArtifactError, match="partial SQLens method"):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )


def test_recomputes_each_repeat_and_pools_completed_over_wall(tmp_path: Path) -> None:
    rows = complete_rows()
    for item in rows:
        if (
            item["dataset"] == "amazon10m"
            and item["experiment_kind"] == "throughput"
            and item["arm_id"] == "stock_pgvector"
        ):
            walls = [10.0, 20.0, 40.0, 10.0, 20.0, 40.0]
            wall = walls[int(item["repeat_id"])]
            item["wall_clock_seconds"] = wall
            item["throughput_qps"] = artifact.EXPECTED_REQUESTS / wall
    _, point_bytes, _ = build(tmp_path, rows)
    points_path = tmp_path / "points.csv"
    points_path.write_bytes(point_bytes)
    points = read_csv(points_path)
    target = next(
        item
        for item in points
        if item["dataset"] == "amazon10m"
        and item["experiment_kind"] == "throughput"
        and item["arm_id"] == "stock_pgvector"
    )
    assert float(target["throughput_qps"]) == pytest.approx(60_000 / 140)

    rows[0]["experiment_kind"] = "latency"
    throughput_target = next(
        item for item in rows if item["experiment_kind"] == "throughput"
    )
    throughput_target["throughput_qps"] = float(throughput_target["throughput_qps"]) + 10
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, rows)
    )
    with pytest.raises(artifact.Figure5ArtifactError, match="throughput_qps mismatch"):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )


def test_pareto_uses_opposite_latency_and_throughput_directions() -> None:
    base = {
        "dataset": "amazon10m",
        "arm_id": "stock_pgvector",
        "mode_id": "original",
        "clients": 1,
        "is_plot_eligible": True,
    }
    latency_points = [
        dict(
            base,
            point_id="l1",
            config_id="a",
            experiment_kind="latency",
            recall_mean=0.80,
            latency_mean_ms=100.0,
            throughput_qps=None,
        ),
        dict(
            base,
            point_id="l2",
            config_id="b",
            experiment_kind="latency",
            recall_mean=0.80,
            latency_mean_ms=120.0,
            throughput_qps=None,
        ),
        dict(
            base,
            point_id="l3",
            config_id="c",
            experiment_kind="latency",
            recall_mean=0.90,
            latency_mean_ms=150.0,
            throughput_qps=None,
        ),
    ]
    throughput_points = [
        dict(
            base,
            point_id="t1",
            config_id="a",
            experiment_kind="throughput",
            recall_mean=0.80,
            latency_mean_ms=None,
            throughput_qps=500.0,
        ),
        dict(
            base,
            point_id="t2",
            config_id="b",
            experiment_kind="throughput",
            recall_mean=0.80,
            latency_mean_ms=None,
            throughput_qps=400.0,
        ),
        dict(
            base,
            point_id="t3",
            config_id="c",
            experiment_kind="throughput",
            recall_mean=0.90,
            latency_mean_ms=None,
            throughput_qps=300.0,
        ),
    ]
    result = artifact.mark_pareto(latency_points + throughput_points)
    pareto_ids = {item["point_id"] for item in result if item["pareto"]}
    assert pareto_ids == {"l1", "l3", "t1", "t3"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_latency_repeat", "requires at least 3 repeats"),
        ("q10k", "fails q10k gate"),
        ("missing_dataset", "three-dataset release gate failed"),
        ("release", "release_identity_sha256 is inconsistent"),
    ],
)
def test_formal_repeat_q10k_dataset_and_release_gates(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    rows = complete_rows()
    if mutation == "missing_latency_repeat":
        rows = [
            item
            for item in rows
            if not (
                item["dataset"] == "amazon10m"
                and item["experiment_kind"] == "latency"
                and item["arm_id"] == "stock_pgvector"
                and item["repeat_id"] == 2
            )
        ]
    elif mutation == "q10k":
        rows[0]["requests"] = 9_999
    elif mutation == "missing_dataset":
        rows = [item for item in rows if item["dataset"] != "laion25m"]
    elif mutation == "release":
        rows[0]["release_identity_sha256"] = "e" * 64
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, rows)
    )
    with pytest.raises(artifact.Figure5ArtifactError, match=message):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )


def test_config_id_is_bound_across_clients_and_repeats(tmp_path: Path) -> None:
    rows = complete_rows()
    target = next(
        item
        for item in rows
        if item["experiment_kind"] == "throughput"
        and item["arm_id"] == "sqlens_full"
        and item["repeat_id"] == 5
    )
    target["config_sha256"] = "f" * 64
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, rows)
    )
    with pytest.raises(artifact.Figure5ArtifactError, match="config_sha256"):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )


def test_dry_run_writes_nothing_and_execute_cli_commits_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, complete_rows())
    )
    output = tmp_path / "artifact"
    common = [
        "--latency-input", str(latency),
        "--throughput-input", str(throughput),
        "--release-contract", str(contract),
        "--latency-run-manifest", str(latency_binding),
        "--throughput-run-manifest", str(throughput_binding),
        "--output-dir", str(output),
    ]
    assert artifact.main([*common, "--dry-run"]) == 0
    assert not output.exists()
    preview = json.loads(capsys.readouterr().out)
    assert preview["paper_eligible"] is True

    assert artifact.main([*common, "--execute"]) == 0
    assert (output / "figure5_manifest.json").is_file()


def test_rejects_latency_throughput_config_set_mismatch(tmp_path: Path) -> None:
    rows = complete_rows()
    for item in rows:
        if (
            item["experiment_kind"] == "throughput"
            and item["dataset"] == "amazon10m"
            and item["arm_id"] == "stock_pgvector"
        ):
            item["config_id"] = "ef200"
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, rows)
    )
    with pytest.raises(artifact.Figure5ArtifactError, match="config sets differ"):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )


def test_rejects_csv_with_only_opaque_release_identity(tmp_path: Path) -> None:
    rows = complete_rows()
    for item in rows:
        item["release_identity_sha256"] = "e" * 64
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, rows)
    )
    with pytest.raises(artifact.Figure5ArtifactError, match="does not match"):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )


def test_rejects_source_manifest_without_all_paper_gates(tmp_path: Path) -> None:
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, complete_rows())
    )
    payload = json.loads(latency_binding.read_text(encoding="utf-8"))
    payload["full_release_complete"] = False
    latency_binding.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(artifact.Figure5ArtifactError, match="full_release_complete"):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )


def test_rejects_converter_without_explicit_audited_run_manifest(
    tmp_path: Path,
) -> None:
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, complete_rows())
    )
    payload = json.loads(latency_binding.read_text(encoding="utf-8"))
    del payload["converter_binding"]["source_manifest"]
    latency_binding.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(artifact.Figure5ArtifactError, match="audited run manifest"):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )


def test_rejects_release_contract_sha_or_identity_drift(tmp_path: Path) -> None:
    latency, throughput, contract, latency_binding, throughput_binding = (
        prepare_formal_inputs(tmp_path, complete_rows())
    )
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["contract_id"] = "different-contract"
    contract.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(artifact.Figure5ArtifactError, match="SHA|release contract"):
        artifact.build_artifact(
            [latency],
            [throughput],
            release_contract_path=contract,
            latency_run_manifest=latency_binding,
            throughput_run_manifest=throughput_binding,
        )
