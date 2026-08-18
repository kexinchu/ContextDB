#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/hybrid_vector_db/scripts"))

import audit_figure6_iso_recall_coverage as audit  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_targets_match_figure6_contract(tmp_path: Path) -> None:
    rel = Path("results/hybrid_vector_db/_figure6_audit_fixture")
    fixture = audit.ROOT / rel
    fixture.mkdir(parents=True, exist_ok=True)
    try:
        _write_csv(
            fixture / "amazon_lat.csv",
            [
                {
                    "target_recall": "0.75",
                    "arm": "stock",
                    "recall": "0.75",
                    "latency_mean_ms": "10",
                    "family": "both_off",
                },
                {
                    "target_recall": "0.75",
                    "arm": "sqlens",
                    "recall": "0.76",
                    "latency_mean_ms": "8",
                    "family": "sqlens_cap",
                },
            ],
        )
        _write_csv(
            fixture / "amazon_thr.csv",
            [
                {
                    "target_recall": "0.75",
                    "arm": "stock",
                    "recall": "0.75",
                    "throughput_qps": "100",
                    "family": "both_off",
                },
                {
                    "target_recall": "0.75",
                    "arm": "sqlens",
                    "recall": "0.76",
                    "throughput_qps": "120",
                    "family": "sqlens_cap",
                },
            ],
        )
        for name in (
            "yfcc_lat.csv",
            "yfcc_thr.csv",
            "laion_lat.csv",
            "laion_thr.csv",
        ):
            _write_csv(fixture / name, [])

        config = {
            "targets": [0.75, 0.8, 0.85, 0.9, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
            "gates": {"max_abs_recall_error": 0.03},
            "datasets": {
                "amazon": {
                    "pg_port": 55433,
                    "latency_bundle": str(rel / "amazon_lat.csv"),
                    "throughput_bundle": str(rel / "amazon_thr.csv"),
                },
                "yfcc": {
                    "pg_port": 55432,
                    "latency_bundle": str(rel / "yfcc_lat.csv"),
                    "throughput_bundle": str(rel / "yfcc_thr.csv"),
                },
                "laion": {
                    "pg_port": 55434,
                    "latency_bundle": str(rel / "laion_lat.csv"),
                    "throughput_bundle": str(rel / "laion_thr.csv"),
                },
            },
        }
        payload = audit.audit(config)
        assert payload["summary"]["n_cells"] == 120
        assert payload["summary"]["n_ok"] == 4
        assert any(
            row["dataset"] == "laion" and not row["ok"] for row in payload["queue"]
        )
    finally:
        for path in fixture.glob("*"):
            path.unlink()
        if fixture.exists():
            fixture.rmdir()


def test_config_targets_are_figure6_list() -> None:
    raw = json.loads(
        (
            audit.ROOT
            / "experiments/hybrid_vector_db/configs/figure6_iso_recall_targets.json"
        ).read_text(encoding="utf-8")
    )
    assert raw["targets"] == [
        0.75,
        0.8,
        0.85,
        0.9,
        0.94,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
    ]
