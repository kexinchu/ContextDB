from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.hybrid_vector_db.scripts import measure_table10_r43_overhead as overhead


def test_dry_run_emits_pending_skeleton(tmp_path: Path) -> None:
    out = tmp_path / "overhead.json"
    rc = overhead.main(["--dry-run", "--out-json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["paper_eligible"] is False
    assert payload["status"] == "dry_run_pending_skeleton"
    assert len(payload["rows"]) == 4
    assert all(row["sqlens"] == "Pending" for row in payload["rows"])


def test_maintenance_from_lifecycle_csv(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["event", "invalidation_ms", "rebuild_ms"]
        )
        writer.writeheader()
        writer.writerow({"event": "invalidation", "invalidation_ms": "10", "rebuild_ms": ""})
        writer.writerow({"event": "rebuild", "invalidation_ms": "", "rebuild_ms": "40"})
        writer.writerow({"event": "reactivation", "invalidation_ms": "20", "rebuild_ms": ""})
    row = overhead.measure_maintenance(path)
    assert row["cost"] == "maintenance_under_writes_p95"
    assert row["sqlens"].endswith("ms")
    assert "inv=1" in row["delta"]
    assert "rebuild=1" in row["delta"]


def test_build_proof_parsing(tmp_path: Path) -> None:
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            {
                "source_build": {"wall_seconds": 100.0, "peak_rss_mb": 2048},
                "bfs_rewrite": {"wall_seconds": 30.0, "peak_rss_mb": 1800},
                "storage_ratio": 1.005,
            }
        ),
        encoding="utf-8",
    )
    row = overhead.measure_build(proof)
    assert "100.0s" in row["stock"]
    assert "30.0s" in row["sqlens"]
    assert "1.0050" in row["delta"]
