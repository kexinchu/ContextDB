from __future__ import annotations

import argparse

import pytest

from experiments.hybrid_vector_db.scripts import (
    run_figure5_sqlens_scan_cap_pilot as pilot,
)


def row(
    activation_ms: float,
    query_latency_ms: float,
    *,
    build: int = 0,
) -> dict[str, str]:
    value = {
        "activation_ms": str(activation_ms),
        "query_latency_ms": str(query_latency_ms),
    }
    value.update({field: "0" for field in pilot.FIXED_EVENT_FIELDS})
    value["d3_adaptive_page_builds_delta"] = str(build)
    return value


def test_projected_q10k_latency_amortizes_only_fixed_d3_cost() -> None:
    projection = pilot.projected_q10k_latency(
        [
            row(2.0, 10.0),
            row(2.0, 10.0),
            row(102.0, 10.0, build=1),
        ]
    )

    assert projection["recurring_activation_ms"] == pytest.approx(2.0)
    assert projection["fixed_activation_excess_ms"] == pytest.approx(100.0)
    assert projection["projected_q10k_e2e_ms"] == pytest.approx(12.01)


def test_parse_caps_is_positive_and_deduplicated() -> None:
    assert pilot.parse_caps("500,1000,500") == [500, 1000]
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        pilot.parse_caps("0,100")
