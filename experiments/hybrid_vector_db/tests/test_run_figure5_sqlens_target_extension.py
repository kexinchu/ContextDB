from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import (
    run_figure5_sqlens_target_extension as extension,
)


def test_rewrites_sqlens_traversal_target_independently() -> None:
    orchestrator = Path(extension.__file__).resolve()
    command = [
        "python",
        "core.py",
        "--mode-configs-json",
        "{}",
        "--guided-collect-target",
        "80",
        "--traversal-guided-target",
        "80",
        "--d3-fragment-store-namespace",
        "",
        "--orchestrator-source",
        "old.py",
    ]
    provenance = {
        "scan_family": "sqlens_cap",
        "sqlens_scan_cap": 5_000_000,
        "modes": [extension.SQLENS_MODE],
        "mode_configs": {
            "original": {
                "ef_search": 80,
                "guided_collect_target": 80,
                "traversal_guided_target": 80,
            },
            extension.SQLENS_MODE: {
                "ef_search": 80,
                "guided_collect_target": 80,
                "traversal_guided_target": 80,
            },
        },
        "execution_sources": {
            "core_runner": {"path": "core.py", "sha256": "a" * 64},
            "orchestrator": {"path": "old.py", "sha256": "b" * 64},
        },
    }

    rewritten, evidence = extension.rewrite_sqlens_target_cell(
        command,
        provenance,
        dataset="laion",
        ef_search=80,
        target=20,
        orchestrator=orchestrator,
        release_prefix="fig5-r36",
    )

    configs = json.loads(
        rewritten[rewritten.index("--mode-configs-json") + 1]
    )
    assert configs[extension.SQLENS_MODE]["guided_collect_target"] == 80
    assert configs[extension.SQLENS_MODE]["traversal_guided_target"] == 20
    assert rewritten[
        rewritten.index("--traversal-guided-target") + 1
    ] == "20"
    assert evidence["scan_family"] == "sqlens_target"
    assert evidence["sqlens_scan_cap"] is None
    assert evidence["sqlens_traversal_target"] == 20
    assert evidence["d3_fragment_store_namespace"] == (
        "fig5-r36-laion-calibration-sqlens_target-ef80-target20"
    )
    assert evidence["execution_sources"]["orchestrator"]["path"] == str(
        orchestrator
    )


def test_parse_settings_rejects_invalid_values() -> None:
    assert extension.parse_settings("80:11,80:20,80:11") == [
        (80, 11),
        (80, 20),
    ]
    with pytest.raises(Exception, match="11 <= target <= ef"):
        extension.parse_settings("80:10")
    with pytest.raises(Exception, match="ef:target"):
        extension.parse_settings("80")


def test_frontier_and_target_extension_share_global_lock_protocol(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "shared.lock"
    frontier_args = extension.frontier.create_parser().parse_args(
        [
            "--require-global-db-lock",
            "--global-db-lock-path",
            str(lock_path),
        ]
    )
    extension_args = extension.create_parser().parse_args(
        [
            "--require-global-db-lock",
            "--global-db-lock-path",
            str(lock_path),
        ]
    )
    assert extension.frontier.global_db_lock_path(frontier_args) == (
        extension.frontier.global_db_lock_path(extension_args)
    )

    frontier_lock = extension.frontier.acquire_global_db_lock(
        lock_path,
        "run_figure5_frontier",
    )
    try:
        with pytest.raises(
            extension.frontier.Figure5ContractError,
            match="already owned",
        ):
            extension.frontier.acquire_global_db_lock(
                lock_path,
                "run_figure5_sqlens_target_extension",
            )
    finally:
        frontier_lock.close()

    extension_lock = extension.frontier.acquire_global_db_lock(
        lock_path,
        "run_figure5_sqlens_target_extension",
    )
    extension_lock.close()


def test_target_extension_records_shared_lock_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    config = {
        "release_contract_path": str(tmp_path / "release.json"),
        "release_contract_sha256": "a" * 64,
        "release_identity": {"contract_id": "test-r36"},
    }
    raw = tmp_path / "out/cell.csv"
    plan = tmp_path / "out/cell.csv.plan.json"
    cell = {
        "dataset": "amazon",
        "ef_search": 200,
        "sqlens_traversal_target": 200,
        "raw": str(raw),
        "plan": str(plan),
        "command": ["fake-db-cell", str(raw)],
        "status": "pending",
        "expected_rows": 1,
        "d3_fragment_store_table": "public.fragments",
        "d3_fragment_store_namespace": "test-r36-target-cell",
        "isolated_repeat_runtimes": False,
    }
    monkeypatch.setattr(
        extension.frontier,
        "load_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        extension,
        "build_schedule",
        lambda config, datasets, settings, backend_cpu_list, out_dir: [cell],
    )
    monkeypatch.setattr(
        extension.frontier,
        "cell_complete",
        lambda raw, plan, expected_rows, provenance=None: raw.is_file(),
    )
    lock_path = tmp_path / "shared.lock"
    reset_observed_lock = False
    cell_observed_lock = False
    final_manifest_observed_lock = False

    def assert_lock_owned() -> None:
        with pytest.raises(
            extension.frontier.Figure5ContractError,
            match="already owned",
        ):
            extension.frontier.acquire_global_db_lock(
                lock_path,
                "conflicting-frontier",
            )

    original_atomic_json = extension.frontier.atomic_json

    def audited_atomic_json(
        path: Path,
        payload: dict[str, object],
    ) -> None:
        nonlocal final_manifest_observed_lock
        isolation = payload.get("database_isolation")
        if (
            payload.get("status") == "complete"
            and isinstance(isolation, dict)
            and isolation.get("held_through_completion") is True
        ):
            assert_lock_owned()
            final_manifest_observed_lock = True
        original_atomic_json(path, payload)

    def fake_reset(table: str, namespace: str) -> int:
        nonlocal reset_observed_lock
        assert table == "public.fragments"
        assert namespace == "test-r36-target-cell"
        assert_lock_owned()
        reset_observed_lock = True
        return 0

    def fake_subprocess(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal cell_observed_lock
        del kwargs
        assert_lock_owned()
        cell_observed_lock = True
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[-1]).write_text("result\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        extension.frontier,
        "clear_fragment_store_namespace",
        fake_reset,
    )
    monkeypatch.setattr(
        extension.frontier,
        "atomic_json",
        audited_atomic_json,
    )
    monkeypatch.setattr(extension.subprocess, "run", fake_subprocess)
    args = extension.create_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--datasets",
            "amazon",
            "--settings",
            "200:200",
            "--out-dir",
            str(tmp_path / "out"),
            "--overwrite",
            "--no-resume",
            "--require-global-db-lock",
            "--global-db-lock-path",
            str(lock_path),
            "--execute",
        ]
    )

    assert extension.run(args) == 0
    assert reset_observed_lock is True
    assert cell_observed_lock is True
    assert final_manifest_observed_lock is True
    manifest = json.loads(
        (
            tmp_path
            / "out/figure5_r36_sqlens_target_extension_manifest.json"
        ).read_text(encoding="utf-8")
    )
    top = manifest["database_isolation"]
    completed = manifest["schedule"][0]["database_isolation"]
    assert top["parallel_db_cells"] is False
    assert top["held_through_completion"] is True
    assert top["lock_path"] == str(lock_path.resolve())
    assert completed["held_through_completion"] is True
    assert completed["lock_owner_token"] == top["lock_owner_token"]

    released = extension.frontier.acquire_global_db_lock(
        lock_path,
        "after-extension",
    )
    released.close()
