from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import (
    run_figure5_stock_scan_cap_extension as extension,
)


def test_rewrites_stock_only_strict_order_scan_cap() -> None:
    orchestrator = Path(extension.__file__).resolve()
    command = [
        "python",
        "core.py",
        "--mode-configs-json",
        "{}",
        "--max-scan-tuples",
        "5000000",
        "--iterative-scan",
        "strict_order",
        "--d3-fragment-store-namespace",
        "",
        "--orchestrator-source",
        "old.py",
    ]
    provenance = {
        "scan_family": "stock_strict",
        "modes": ["original"],
        "mode_configs": {
            "original": {
                "ef_search": 11,
                "iterative_scan": "strict_order",
                "max_scan_tuples": 5_000_000,
            },
            "design1_bloom_bfs_layout_d3": {
                "ef_search": 11,
                "iterative_scan": "off",
                "max_scan_tuples": 5_000_000,
            },
        },
        "execution_sources": {
            "core_runner": {"path": "core.py", "sha256": "a" * 64},
            "orchestrator": {"path": "old.py", "sha256": "b" * 64},
        },
    }

    rewritten, evidence = extension.rewrite_stock_cap_cell(
        command,
        provenance,
        dataset="amazon",
        ef_search=11,
        cap=20_000,
        orchestrator=orchestrator,
        release_prefix="fig5-r36",
    )

    configs = json.loads(
        rewritten[rewritten.index("--mode-configs-json") + 1]
    )
    assert configs["original"]["max_scan_tuples"] == 20_000
    assert configs["original"]["iterative_scan"] == "strict_order"
    assert rewritten[rewritten.index("--max-scan-tuples") + 1] == "20000"
    assert evidence["scan_family"] == "stock_cap"
    assert evidence["stock_scan_cap"] == 20_000
    assert evidence["d3_fragment_store_namespace"] == (
        "fig5-r36-amazon-calibration-stock_cap-ef11-cap20000"
    )
    assert evidence["execution_sources"]["orchestrator"]["path"] == str(
        orchestrator
    )


def test_parse_caps_rejects_invalid_values() -> None:
    assert extension.parse_caps("500,1000,500") == [500, 1000]
    with pytest.raises(Exception, match="positive"):
        extension.parse_caps("0,100")
