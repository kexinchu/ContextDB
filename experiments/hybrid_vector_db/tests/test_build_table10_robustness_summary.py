from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.hybrid_vector_db.scripts import build_table10_robustness_summary as summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def adversarial_payload(*, failures: int = 0, bypasses: int = 108) -> dict:
    records = []
    for i in range(1134):
        records.append(
            {
                "ordered_mismatch": i < failures,
                "false_negative": False,
                "error": "",
                "stale_bypass": i < bypasses,
            }
        )
    return {
        "artifact_valid": True,
        "paper_eligible": True,
        "summary": {"records": 1134, "strict_failures": failures},
        "records": records,
    }


def nonowner_payload() -> dict:
    return {
        "artifact_valid": True,
        "paper_eligible": True,
        "checks": {
            "failed": 0,
            "exact_stock_equal": 12,
            "exact_sqlens_equal": 12,
            "visible_row_violations": 0,
            "rls_stock_bypasses": 12,
        },
    }


def stress_payload() -> dict:
    records = []
    for i in range(1000):
        records.append(
            {
                "post_update_refresh_or_safe_fallback": i < 31,
                "guided_profile_classification": {"stale_fallback": False},
            }
        )
    return {
        "artifact_valid": True,
        "paper_eligible": True,
        "correctness_summary": {
            "paired_requests": 1000,
            "ordered_equivalent": 1000,
            "guided_sql_valid": 1000,
        },
        "committed_mutations": {"insert": 200, "delete": 200, "predicate": 800, "vector": 800},
        "records": records,
    }


def test_panel_a_loads_and_marks_complete(tmp_path: Path) -> None:
    adv = tmp_path / "adv.json"
    non = tmp_path / "non.json"
    stress = tmp_path / "stress.json"
    man = tmp_path / "stress.manifest.json"
    write_json(adv, adversarial_payload())
    write_json(non, nonowner_payload())
    write_json(stress, stress_payload())
    write_json(
        man,
        {
            "artifact_valid": True,
            "paper_eligible": True,
            "artifact_gates": {"overlap_queries": 31},
        },
    )
    args = summary.parse_args(
        [
            "--adversarial-json",
            str(adv),
            "--nonowner-json",
            str(non),
            "--stress-json",
            str(stress),
            "--stress-manifest",
            str(man),
            "--out-json",
            str(tmp_path / "out.json"),
        ]
    )
    built = summary.build_summary(args)
    assert built["panel_a_complete"] is True
    assert built["panel_b_filled_cells"] == 0
    assert built["panel_c_filled_rows"] == 0
    assert built["paper_table_complete"] is False
    assert built["panel_a_correctness"][0]["bypass"] == 108
    assert built["panel_a_correctness"][2]["epoch_refreshes"] == 31
    assert built["panel_a_correctness"][2]["overlap_queries"] == 31


def test_rejects_non_eligible_adversarial(tmp_path: Path) -> None:
    adv = tmp_path / "adv.json"
    payload = adversarial_payload()
    payload["paper_eligible"] = False
    write_json(adv, payload)
    with pytest.raises(summary.Table10SummaryError, match="paper_eligible"):
        summary.load_panel_a_adversarial(adv)


def test_panel_b_pending_without_csv() -> None:
    cells = summary.load_panel_b_concurrency(None)
    assert len(cells) == 4
    assert all(cell["status"] == "Pending" for cell in cells)
    assert cells[0]["delivered_tps_ratio"] == "---"
    assert cells[1]["delivered_tps_ratio"] == "Pending"


def test_render_tex_contains_three_panels(tmp_path: Path) -> None:
    adv = tmp_path / "adv.json"
    non = tmp_path / "non.json"
    stress = tmp_path / "stress.json"
    write_json(adv, adversarial_payload())
    write_json(non, nonowner_payload())
    write_json(stress, stress_payload())
    args = summary.parse_args(
        [
            "--adversarial-json",
            str(adv),
            "--nonowner-json",
            str(non),
            "--stress-json",
            str(stress),
            "--out-json",
            str(tmp_path / "out.json"),
            "--out-tex",
            str(tmp_path / "out.tex"),
            "--write-tex",
        ]
    )
    built = summary.build_summary(args)
    tex = summary.render_tex(
        built["panel_a_correctness"],
        built["panel_b_concurrency"],
        built["panel_c_overhead"],
    )
    assert "(A) Correctness" in tex
    assert "(B) Concurrency" in tex
    assert "(C) Overhead" in tex
    assert "Pending" in tex
    assert "1{,}134" in tex or "1134" in tex
