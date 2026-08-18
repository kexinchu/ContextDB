from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from paper.scripts import plot_figure5_frontier as plotter


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifact(root: Path) -> None:
    fields = (
        "dataset",
        "experiment_kind",
        "arm_id",
        "config_id",
        "recall_mean",
        "latency_mean_ms",
        "throughput_qps",
        "pareto",
        "is_plot_eligible",
    )
    points = root / "figure5_points.csv"
    root.mkdir(parents=True)
    rows = []
    for dataset in plotter.EXPECTED_DATASETS:
        for kind in plotter.EXPECTED_KINDS:
            for arm in plotter.EXPECTED_ARMS:
                for index in range(plotter.MIN_CONFIGS_PER_CURVE):
                    rows.append(
                        {
                            "dataset": dataset,
                            "experiment_kind": kind,
                            "arm_id": arm,
                            "config_id": f"{arm}-{index}",
                            "recall_mean": 0.70 + index * 0.025,
                            "latency_mean_ms": (
                                10 + index * 3 if kind == "latency" else ""
                            ),
                            "throughput_qps": (
                                100 - index * 4 if kind == "throughput" else ""
                            ),
                            "pareto": "true",
                            "is_plot_eligible": "true",
                        }
                    )
    with points.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "artifact_type": "sqlens_figure5_frontier",
        "artifact_valid": True,
        "paper_eligible": True,
        "gates": {"audited": True, "measured_throughput_only": True},
        "outputs": {
            "points": {
                "path": points.name,
                "rows": len(rows),
                "sha256": sha256(points),
            }
        },
    }
    (root / "figure5_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_plotter_requires_audited_points(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    write_artifact(artifact)

    rows = plotter.load_points(artifact)
    assert len(rows) == 120

    manifest_path = artifact / "figure5_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["paper_eligible"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        plotter.load_points(artifact)
    except plotter.PlotArtifactError:
        pass
    else:
        raise AssertionError("plotter accepted a non-paper-eligible artifact")


def test_plotter_writes_all_panels(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    artifact = tmp_path / "artifact"
    output = tmp_path / "figures"
    write_artifact(artifact)

    paths = plotter.run(artifact, output)

    assert len(paths) == 7
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
