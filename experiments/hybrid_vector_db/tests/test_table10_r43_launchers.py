from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "experiments/hybrid_vector_db/scripts"


def test_concurrency_launcher_requests_frozen_matrix() -> None:
    source = (SCRIPTS / "run_table10_r43_amazon_concurrency.sh").read_text(encoding="utf-8")
    assert "--protocol p0_6_full" in source
    assert "--readers" in source
    assert "--update-rates" in source
    assert "--methods stock,sqlens_full" in source
    assert "--requests 10000" in source
    assert "--measurement-repeats 6" in source
    assert "--resume" in source
    assert "--execute" in source
    assert "p0_release_contract_r43.json" in source
    assert ".pg55437_experiment.lock" in source


def test_overhead_launcher_defaults_to_dry_run() -> None:
    source = (SCRIPTS / "run_table10_r43_overhead.sh").read_text(encoding="utf-8")
    assert "--dry-run" in source
    assert "measure_table10_r43_overhead.py" in source
    assert "EXECUTE" in source
