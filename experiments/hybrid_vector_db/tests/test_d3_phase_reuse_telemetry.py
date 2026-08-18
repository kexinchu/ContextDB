from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import pgvector_design1_design2_design3_selectivity_benchmark as benchmark  # noqa: E402


def _profile(**overrides: object) -> dict[str, object]:
    profile: dict[str, object] = {
        "active": True,
        "adaptive_state": "page",
        "adaptive_admissions": 1,
        "adaptive_refinements": 0,
        "adaptive_page_builds": 0,
        "adaptive_bloom_builds": 0,
        "adaptive_exact_builds": 0,
        "adaptive_fragment_builds": 0,
        "fragment_cache_hits": 0,
        "fragment_store_hits": 0,
        "fragment_builds": 0,
        "fast_reactivation_hits": 0,
        "composed_guide_hits": 0,
    }
    profile.update(overrides)
    return profile


def _cache(**overrides: object) -> dict[str, object]:
    profile: dict[str, object] = {
        "resident_entries": 1,
        "resident_bytes": 4096,
        "composed_guide_hits": 0,
    }
    profile.update(overrides)
    return profile


def test_page_to_bloom_build_is_refinement_not_warm_reuse() -> None:
    evidence = benchmark.d3_phase_evidence(
        _profile(),
        _profile(
            adaptive_state="bloom",
            adaptive_bloom_builds=1,
            adaptive_refinements=1,
            adaptive_fragment_builds=1,
            fragment_builds=1,
        ),
        _cache(),
        _cache(),
        {
            "guidance_enabled": True,
            "guidance_route": "d3_adaptive",
            "activation_atom_count": 1,
        },
        same_predicate_before=True,
    )

    assert evidence["d3_phase_detail"] == "refinement"
    assert evidence["d3_phase"] == "refinement"
    assert evidence["d3_build_observed"] is True
    assert evidence["d3_refinement_observed"] is True
    assert evidence["d3_active_guidance_reused"] is False
    assert evidence["d3_reuse_event"] == ""
    assert "refinement" in benchmark.D3_PHASES


def test_positive_fast_cache_event_is_the_only_warm_reuse_proof() -> None:
    evidence = benchmark.d3_phase_evidence(
        _profile(),
        _profile(
            fragment_cache_hits=1,
            fast_reactivation_hits=1,
            composed_guide_hits=1,
        ),
        _cache(),
        _cache(composed_guide_hits=1),
        {
            "guidance_enabled": True,
            "guidance_route": "d3_adaptive",
            "activation_atom_count": 1,
        },
        same_predicate_before=True,
    )

    assert evidence["d3_phase_detail"] == "fragment_cache_reactivation"
    assert evidence["d3_phase"] == "warm"
    assert evidence["d3_reuse_event"] == "fragment_cache"
    assert evidence["d3_reuse_event_trusted"] is True
    assert evidence["d3_active_guidance_reused"] is True
    assert evidence["d3_fragment_cache_hits_delta"] == 1
    assert evidence["d3_fragment_cache_hits_delta_invalid"] is False


def test_negative_delta_fails_closed_without_fabricating_a_hit() -> None:
    evidence = benchmark.d3_phase_evidence(
        _profile(fragment_cache_hits=4, fast_reactivation_hits=2),
        _profile(fragment_cache_hits=0, fast_reactivation_hits=0),
        _cache(),
        _cache(),
        {
            "guidance_enabled": True,
            "guidance_route": "d3_adaptive",
            "activation_atom_count": 1,
        },
        same_predicate_before=True,
    )

    assert evidence["d3_counter_reset_observed"] is True
    assert "fragment_cache_hits" in json.loads(evidence["d3_counter_reset_fields"])
    assert evidence["d3_fragment_cache_hits_delta"] == 0
    assert evidence["d3_fragment_cache_hits_delta_invalid"] is True
    assert evidence["d3_reuse_event_trusted"] is False
    assert evidence["d3_active_guidance_reused"] is False
    assert evidence["d3_cache_reuse_observed"] is False
