#!/usr/bin/env python3
"""Aligned ACORN-1: same hybrid L0 on the BFS clone, different oracles.

Stock activates an exact TID set for the full SQL predicate, then acorn1.
SQLens activates d1_d2_d3 fragments, then acorn1. Both keep
iterative_scan=off. Smoke fails unless every measured scan reports
final_path=hybrid_l0.

Writes only under revision45/q3_acorn_aligned. Does not touch 55437 or
paper tables. Default PGPORT is 55440.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

import amazon10m_sql_native_benchmark as bench
import figure5_hybrid_allowlist_screen as fig5
import run_q3_acorn_matched as q3
from common_pg import pg_config_from_env, require_psycopg

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "results/hybrid_vector_db/revision45/q3_acorn_aligned"
QUERY_OFFSET = 200
QUERY_COUNT = 50
SCREEN_FILTERS = (
    "popular_ge1000",
    "popular_ge3284",
    "popular_ge10066",
    "grocery_helpful",
    "grocery_long500",
)
SMOKE_FILTERS = ("grocery_helpful", "popular_ge1340")
SMOKE_QUERY_COUNT = 5
SMOKE_EFS = (100,)
ARMS = ("stock_acorn", "sqlens_acorn")
REQUIRED_PATH = "hybrid_l0"
SQLENS_KIND = "bloom"


def exact_atoms(spec: bench.FilterSpec) -> tuple[str, ...]:
    """One exact atom for the full heap-local predicate."""
    return (f"sql:{spec.predicate}",)


def _sql_and_params(
    spec: bench.FilterSpec,
    workload: bench.WorkloadSpec,
    query_id: int,
    as_of: int,
    embeddings: dict[int, str],
    atoms: tuple[str, ...],
    binding_kind: str,
) -> tuple[str, dict[str, Any]]:
    sql_text = bench.build_hybrid_sql(
        fig5.TABLE,
        spec.predicate,
        workload=workload,
        official_compatible=False,
    )
    payload: dict[str, Any] = {
        "query_id": query_id,
        "as_of": as_of,
        "k": q3.K,
        "vector_index": fig5.CLONE_INDEX,
        "binding_atoms": list(atoms),
        "binding_kind": binding_kind,
    }
    return sql_text, bench.bind_query_embedding(payload, query_id, embeddings)


def _profile_slice(profile: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "final_path",
        "filter_strategy",
        "iterative_scan",
        "effective_iterative_scan",
        "stock_bypass_reason",
        "traversal_guidance_scope",
        "planner_proof_succeeded",
        "planner_proof_bypass_reason",
    )
    return {key: profile.get(key) for key in keys}


def _prepare_clone_session(cur: Any, ef_search: int) -> bench.Config:
    config = bench.Config(ef_search, 5_000_000, 32.0, "off", ef_search)
    cur.execute("RESET ROLE")
    bench.set_heap_competing_indexes_valid(cur, fig5.TABLE, valid=False)
    fig5.prepare_pg(cur)
    cur.execute("RESET ROLE")
    bench.set_search_config(cur, config)
    bench.configure_hnsw_driven_planner(cur)
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    bench.set_preferred_index(cur, fig5.CLONE_INDEX)
    cur.execute("SET hnsw.filter_strategy = acorn1")
    cur.execute("SET hnsw.iterative_scan = off")
    return config


def _activate(
    cur: Any,
    atoms: tuple[str, ...],
    kind: str,
    *,
    allow_rebuild: bool = True,
) -> dict[str, Any]:
    cur.execute("RESET ROLE")
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SET statement_timeout = '30min'")
    started = time.perf_counter()
    activated = 0
    activation: dict[str, Any] = {}
    attempts = 3 if kind == "adaptive" else 1
    for attempt in range(attempts):
        cur.execute(
            "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
            (fig5.CLONE_INDEX, list(atoms), kind),
        )
        activated = int(cur.fetchone()[0] or 0)
        activation = bench.fetch_json_object(cur, "SELECT vector_hnsw_guidance_profile()")
        if bool(activation.get("active")) and activated > 0:
            break
        if kind != "adaptive":
            break
    if (not bool(activation.get("active")) or activated < 1) and allow_rebuild:
        try:
            cur.execute(
                "SELECT vector_hnsw_guidance_rebuild(%s::regclass, %s::text[], %s::text)",
                (fig5.CLONE_INDEX, list(atoms), kind),
            )
            activated = int(cur.fetchone()[0] or 0)
            activation = bench.fetch_json_object(cur, "SELECT vector_hnsw_guidance_profile()")
        except Exception:
            pass
    activation_ms = (time.perf_counter() - started) * 1000.0
    cur.execute("SET statement_timeout = 0")
    cur.execute("SET hnsw.filter_strategy = acorn1")
    cur.execute("SET hnsw.iterative_scan = off")
    cur.execute(f'SET ROLE "{fig5.PRINCIPAL}"')
    enabled = bool(activation.get("active")) and activated > 0
    if not enabled:
        raise RuntimeError(
            f"{kind} guidance did not become active: atoms={activated} "
            f"profile={json.dumps(activation, sort_keys=True)[:800]}"
        )
    return {
        "guidance_enabled": True,
        "activation_atom_count": activated,
        "activation_ms": activation_ms,
        "guidance_kind": str(activation.get("kind", kind)),
        "composed_exact_active": bool(activation.get("composed_exact_active")),
        "composed_exact_rows": activation.get("composed_exact_rows"),
        "adaptive_state": activation.get("adaptive_state"),
        "after_activation": activation,
    }


def _run_queries(
    cur: Any,
    spec: bench.FilterSpec,
    workload: bench.WorkloadSpec,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth: dict[int, tuple[int, ...]],
    as_of: int,
    query_nos: list[int],
    ef_search: int,
    arm: str,
    atoms: tuple[str, ...],
    binding_kind: str,
    activation: dict[str, Any],
) -> list[dict]:
    rows: list[dict] = []
    for query_no in query_nos:
        sql_text, params = _sql_and_params(
            spec,
            workload,
            query_ids[query_no],
            as_of,
            embeddings,
            atoms,
            binding_kind,
        )
        error = ""
        ids: list[int] = []
        e2e_ms = 0.0
        scan: dict[str, Any] = {}
        try:
            bench.set_as_of(cur, as_of)
            cur.execute("SELECT vector_hnsw_reset_scan_profile()")
            started = time.perf_counter()
            cur.execute(sql_text, params)
            fetched = cur.fetchall()
            e2e_ms = (time.perf_counter() - started) * 1000.0
            ids = [int(row[0]) for row in fetched]
            scan = bench.fetch_json_object(cur, "SELECT vector_hnsw_last_scan_profile()")
        except Exception as exc:  # noqa: BLE001
            error = f"{exc.__class__.__name__}: {exc}"
            try:
                cur.execute("ROLLBACK")
            except Exception:
                pass
        path = str(scan.get("final_path", ""))
        rows.append(
            {
                "phase": "measurement",
                "filter_name": spec.name,
                "mode": arm,
                "query_no": query_no,
                "ef_search": ef_search,
                "e2e_ms": e2e_ms if not error else "",
                "recall": fig5.recall_at_k(ids, truth[query_no], q3.K) if not error else "",
                "activation_ms": activation["activation_ms"],
                "final_path": path,
                "aligned_ann": path == REQUIRED_PATH,
                "error": error,
                **_profile_slice(scan),
            }
        )
    return rows


def run_stock_acorn(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth: dict[int, tuple[int, ...]],
    as_of: int,
    query_nos: list[int],
    ef_search: int,
) -> tuple[list[dict], dict[str, Any]]:
    atoms = exact_atoms(spec)
    _prepare_clone_session(cur, ef_search)
    activation = _activate(cur, atoms, "exact")
    rows = _run_queries(
        cur,
        spec,
        workload,
        query_ids,
        embeddings,
        truth,
        as_of,
        query_nos,
        ef_search,
        "stock_acorn",
        atoms,
        "exact",
        activation,
    )
    return rows, activation


def run_sqlens_acorn(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth: dict[int, tuple[int, ...]],
    as_of: int,
    query_nos: list[int],
    ef_search: int,
) -> tuple[list[dict], dict[str, Any]]:
    atoms = bench.binding_atoms_for(workload, spec)
    _prepare_clone_session(cur, ef_search)
    # Bloom for every filter so PAGE cannot force effective iterative_scan=strict.
    activation = _activate(cur, atoms, SQLENS_KIND)
    rows = _run_queries(
        cur,
        spec,
        workload,
        query_ids,
        embeddings,
        truth,
        as_of,
        query_nos,
        ef_search,
        "sqlens_acorn",
        atoms,
        SQLENS_KIND,
        activation,
    )
    return rows, activation


def _summarize(rows: list[dict]) -> dict[str, Any] | None:
    stats = q3._summarize_rows(rows)
    if stats is None:
        return None
    paths = [str(row.get("final_path") or "") for row in rows]
    stats["final_paths"] = sorted(set(paths))
    stats["aligned_ann"] = bool(paths) and all(path == REQUIRED_PATH for path in paths)
    stats["effective_iterative_scan"] = rows[0].get("effective_iterative_scan")
    stats["stock_bypass_reason"] = rows[0].get("stock_bypass_reason")
    return stats


def run_arm(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    arm: str,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth: dict[int, tuple[int, ...]],
    as_of: int,
    query_nos: list[int],
    ef_search: int,
) -> tuple[list[dict], dict[str, Any]]:
    if arm == "stock_acorn":
        return run_stock_acorn(
            cur, workload, spec, query_ids, embeddings, truth, as_of, query_nos, ef_search
        )
    if arm == "sqlens_acorn":
        return run_sqlens_acorn(
            cur, workload, spec, query_ids, embeddings, truth, as_of, query_nos, ef_search
        )
    raise ValueError(arm)


def calibrate_arm(
    cur: Any,
    workload: bench.WorkloadSpec,
    spec: bench.FilterSpec,
    arm: str,
    query_ids: dict[int, int],
    embeddings: dict[int, str],
    truth: dict[int, tuple[int, ...]],
    as_of: int,
    query_nos: list[int],
    efs: tuple[int, ...],
    *,
    require_aligned: bool,
) -> tuple[list[dict], dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    sweep: list[dict[str, Any]] = []
    chosen_rows: list[dict] = []
    chosen: dict[str, Any] | None = None
    last_activation: dict[str, Any] = {}
    for ef in efs:
        print(
            json.dumps({"progress": "try_ef", "filter": spec.name, "arm": arm, "ef": ef}),
            flush=True,
        )
        rows, activation = run_arm(
            cur, workload, spec, arm, query_ids, embeddings, truth, as_of, query_nos, ef
        )
        last_activation = activation
        stats = _summarize(rows)
        if stats:
            sweep.append({"arm": arm, "filter_name": spec.name, **stats})
            print(json.dumps({"progress": "ef_done", **sweep[-1]}, default=str), flush=True)
            if require_aligned and not stats["aligned_ann"]:
                sample = next((row for row in rows if row.get("final_path") != REQUIRED_PATH), {})
                raise RuntimeError(
                    f"{arm} on {spec.name} at ef={ef} did not stay on {REQUIRED_PATH}: "
                    f"{stats['final_paths']} proof={sample.get('planner_proof_bypass_reason')} "
                    f"iter={sample.get('effective_iterative_scan')}"
                )
            if stats["recall_lcb95"] >= q3.TARGET_LCB and stats["n"] == len(query_nos):
                chosen_rows = rows
                chosen = stats
                break
        else:
            errors = [row.get("error") for row in rows if row.get("error")]
            sweep.append(
                {
                    "arm": arm,
                    "filter_name": spec.name,
                    "ef_search": ef,
                    "n": 0,
                    "errors": len(errors),
                    "error_sample": errors[:1],
                }
            )
            if errors:
                raise RuntimeError(f"{arm} on {spec.name} failed: {errors[0]}")
    if chosen is None and sweep:
        eligible = [item for item in sweep if item.get("recall_lcb95") is not None]
        if eligible:
            chosen = {**max(eligible, key=lambda item: item["recall_lcb95"]), "met_target": False}
    if chosen is not None and "met_target" not in chosen:
        chosen["met_target"] = chosen.get("recall_lcb95", 0.0) >= q3.TARGET_LCB
    if chosen is not None:
        chosen["activation_ms"] = last_activation.get("activation_ms")
        chosen["guidance_kind"] = last_activation.get("guidance_kind")
        chosen["composed_exact_active"] = last_activation.get("composed_exact_active")
        chosen["composed_exact_rows"] = last_activation.get("composed_exact_rows")
        chosen["adaptive_state"] = last_activation.get("adaptive_state")
    return chosen_rows, chosen, sweep, last_activation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--filter-names", nargs="*")
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--efs", nargs="+", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    smoke = bool(args.smoke)
    names = list(args.filter_names or (SMOKE_FILTERS if smoke else SCREEN_FILTERS))
    query_count = int(args.query_count or (SMOKE_QUERY_COUNT if smoke else QUERY_COUNT))
    if args.efs:
        efs = tuple(int(v) for v in args.efs)
    elif smoke:
        efs = SMOKE_EFS
    else:
        efs = q3.ACORN_EFS
    high_ef = bool(args.efs)
    contract = {
        "paper_eligible": False,
        "plan_item": (
            "Q3_ACORN_ALIGNED_SMOKE" if smoke
            else "Q3_ACORN_ALIGNED_HIGH_EF" if high_ef
            else "Q3_ACORN_ALIGNED"
        ),
        "protocol": {
            "index": fig5.CLONE_INDEX,
            "ann": "HnswSearchHybridL0",
            "filter_strategy": "acorn1",
            "iterative_scan": "off",
            "stock_oracle": "exact_tid_set",
            "sqlens_oracle": "bloom",
            "required_final_path": REQUIRED_PATH,
        },
        "arms": list(ARMS),
        "filters": names,
        "query_offset": QUERY_OFFSET,
        "query_count": query_count,
        "target_recall_lcb95": q3.TARGET_LCB,
        "acorn_efs": list(efs),
        "smoke": smoke,
        "out_dir": str(args.out_dir),
    }
    if not args.execute:
        print(json.dumps({"dry_run": True, **contract}, indent=2))
        return 0

    fig5.set_cohort(QUERY_OFFSET, query_count)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    require_psycopg()
    import psycopg

    workload = q3._workload()
    specs = {spec.name: spec for spec in bench.read_filters(bench.DEFAULT_FILTERS, set(names))}
    cfg = pg_config_from_env()
    if int(cfg.port) == 55437:
        raise SystemExit("refusing to run the aligned ACORN screen on 55437")
    score_path = args.out_dir / (
        "smoke.json" if smoke else "score_high_ef.json" if high_ef else "score.json"
    )
    prior = (
        json.loads(score_path.read_text(encoding="utf-8"))
        if args.resume and score_path.exists()
        else {}
    )
    cells = list(prior.get("cells") or [])
    sweep = list(prior.get("sweep") or [])
    done = {str(cell["filter_name"]) for cell in cells if all(cell.get(arm) for arm in ARMS)}
    query_nos = list(range(QUERY_OFFSET, QUERY_OFFSET + query_count))

    def _checkpoint() -> None:
        payload = {**contract, "cells": cells, "sweep": sweep}
        score_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        from rowlocal_faiss14_screen import load_attr_truth

        for name in names:
            if name in done:
                print(json.dumps({"progress": "resume_skip", "filter": name}), flush=True)
                continue
            spec = specs[name]
            query_ids, truth, as_of = load_attr_truth(name)
            embed_ids = [query_ids[q] for q in query_nos if q in query_ids]
            embeddings = bench.load_query_embeddings(cur, fig5.TABLE, embed_ids)
            print(json.dumps({"progress": "filter", "filter": name}), flush=True)
            cell: dict[str, Any] = {"filter_name": name}
            for arm in ARMS:
                _rows, chosen, mode_sweep, activation = calibrate_arm(
                    cur,
                    workload,
                    spec,
                    arm,
                    query_ids,
                    embeddings,
                    truth,
                    as_of,
                    query_nos,
                    efs,
                    require_aligned=True,
                )
                sweep.extend(mode_sweep)
                cell[arm] = chosen
                cell[f"{arm}_activation"] = {
                    key: activation.get(key)
                    for key in (
                        "activation_ms",
                        "guidance_kind",
                        "composed_exact_active",
                        "composed_exact_rows",
                        "adaptive_state",
                        "activation_atom_count",
                    )
                }
            if all(cell.get(arm) for arm in ARMS):
                stock_ms = cell["stock_acorn"]["mean_ms"]
                sqlens_ms = cell["sqlens_acorn"]["mean_ms"]
                cell["sqlens_vs_stock"] = stock_ms / sqlens_ms if sqlens_ms else None
                cell["all_met_target"] = all(
                    bool(cell[arm].get("met_target")) for arm in ARMS
                )
                cell["all_aligned_ann"] = all(
                    bool(cell[arm].get("aligned_ann")) for arm in ARMS
                )
            cells = [item for item in cells if item.get("filter_name") != name]
            cells.append(cell)
            _checkpoint()
            print(
                json.dumps(
                    {
                        "progress": "filter_done",
                        "filter": name,
                        "sqlens_vs_stock": cell.get("sqlens_vs_stock"),
                        "all_aligned_ann": cell.get("all_aligned_ann"),
                        "all_met_target": cell.get("all_met_target"),
                    }
                ),
                flush=True,
            )

    _checkpoint()
    (args.out_dir / "manifest.json").write_text(
        json.dumps({**contract, "cells": len(cells)}, indent=2) + "\n", encoding="utf-8"
    )
    aligned = all(bool(cell.get("all_aligned_ann")) for cell in cells) and len(cells) == len(names)
    print(
        json.dumps(
            {
                "wrote": str(args.out_dir),
                "paper_eligible": False,
                "smoke": smoke,
                "aligned_ann": aligned,
                "filters": len(cells),
            }
        )
    )
    return 0 if aligned or not smoke else 2


if __name__ == "__main__":
    raise SystemExit(main())
