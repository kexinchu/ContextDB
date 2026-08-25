#!/usr/bin/env python3
"""P3: 16-client read-only QPS on grocery_helpful attributes (same SQL as Fig 5)."""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from pathlib import Path

_SETUP_LOCK = threading.Lock()

import amazon10m_sql_native_benchmark as bench
import figure5_hybrid_allowlist_screen as fig5
from common_pg import pg_config_from_env, require_psycopg

OUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "results/hybrid_vector_db/figure5_qps16_readonly"
)


def _worker(
    conninfo: str,
    mode: str,
    sql_text: str,
    jobs: list[dict],
    stop_at: float,
    counts: list[int],
    slot: int,
) -> None:
    import psycopg

    completed = 0
    try:
        with psycopg.connect(conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            config = bench.Config(fig5.EF_PG, 5_000_000, 32.0, "relaxed_order", fig5.EF_PG)
            index = bench.mode_index(mode, fig5.SOURCE_INDEX, fig5.CLONE_INDEX)
            with _SETUP_LOCK:
                fig5.prepare_pg(cur)
                # First heap touch as the principal; otherwise later vector_hnsw_*
                # calls fail with "must be owner of index" on a fresh session.
                cur.execute(f"SELECT id FROM {fig5.TABLE} LIMIT 1")
                cur.fetchone()
                bench.set_mode(cur, mode, config, index, reset_cache=True)
            i = 0
            try:
                while time.perf_counter() < stop_at:
                    job = jobs[i % len(jobs)]
                    step = "set_as_of"
                    bench.set_as_of(cur, job["as_of"])
                    step = "configure_guidance"
                    bench.configure_guidance(cur, mode, index, job["atoms"])
                    step = "hybrid_sql"
                    cur.execute(sql_text, job["params"])
                    cur.fetchall()
                    completed += 1
                    i += 1
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"{step}: {exc.__class__.__name__}: {exc}") from exc
            finally:
                try:
                    cur.execute("RESET ROLE")
                except Exception:
                    pass
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "progress": "qps_worker_error",
                    "mode": mode,
                    "slot": slot,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "where": traceback.format_exc().splitlines()[-2] if traceback.format_exc() else "",
                }
            ),
            flush=True,
        )
    counts[slot] = completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--clients", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if not args.execute:
        print("dry-run")
        return 0
    fig5.set_cohort(200, 1000)
    require_psycopg()
    import psycopg

    cfg = pg_config_from_env()
    spec = bench.read_filters(bench.DEFAULT_FILTERS, {fig5.FILTER_NAME})[0]
    query_ids, truth_by_shape, as_of = fig5.load_truth(fig5.FILTER_NAME)
    workload = fig5._workload("attributes", "none")
    sql_text = bench.build_hybrid_sql(fig5.TABLE, spec.predicate, workload=workload)
    atoms = bench.binding_atoms_for(workload, spec)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"clients": args.clients, "seconds": args.seconds, "arms": {}}
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        fig5.prepare_pg(cur)
        embed_ids = [
            query_ids[query_no]
            for query_no in range(fig5.QUERY_OFFSET, fig5.QUERY_OFFSET + fig5.QUERY_COUNT)
        ]
        embeddings = bench.load_query_embeddings(cur, fig5.TABLE, embed_ids)
        cur.execute("RESET ROLE")
    jobs = []
    for query_no in range(fig5.QUERY_OFFSET, fig5.QUERY_OFFSET + fig5.QUERY_COUNT):
        query_id = query_ids[query_no]
        jobs.append(
            {
                "as_of": as_of,
                "atoms": atoms,
                "params": bench.bind_query_embedding(
                    {
                        "query_id": query_id,
                        "as_of": as_of,
                        "k": fig5.K,
                        "vector_index": "",
                        "binding_atoms": list(atoms),
                        "binding_kind": "adaptive",
                    },
                    query_id,
                    embeddings,
                ),
            }
        )
    with psycopg.connect(cfg.conninfo, autocommit=True) as control:
        control_cur = control.cursor()
        fig5.prepare_pg(control_cur)
        bench.set_heap_competing_indexes_valid(control_cur, fig5.TABLE, valid=False)
        control_cur.execute("RESET ROLE")
    try:
      for mode in ("stock", "d1_d2_d3"):
        if mode == "d1_d2_d3":
            with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
                cur = conn.cursor()
                fig5.prepare_pg(cur)
                bench.clear_fragment_store(cur, fig5.TABLE)
                cur.execute("RESET ROLE")
        # patch vector_index per mode
        index = bench.mode_index(mode, fig5.SOURCE_INDEX, fig5.CLONE_INDEX)
        config = bench.Config(fig5.EF_PG, 5_000_000, 32.0, "relaxed_order", fig5.EF_PG)
        with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            bench.set_mode(cur, mode, config, index, reset_cache=True)
        for job in jobs:
            job["params"]["vector_index"] = index
            job["params"]["binding_kind"] = bench.MODE_SPECS[mode].guidance_kind or "bloom"
        counts = [0] * args.clients
        stop_at = time.perf_counter() + args.seconds
        threads = [
            threading.Thread(
                target=_worker,
                args=(cfg.conninfo, mode, sql_text, jobs, stop_at, counts, slot),
            )
            for slot in range(args.clients)
        ]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - started
        total = sum(counts)
        payload["arms"][mode] = {
            "completed": total,
            "elapsed_s": elapsed,
            "qps": total / elapsed if elapsed else 0.0,
            "per_client": counts,
        }
        print(json.dumps({"progress": "qps_done", "mode": mode, **payload["arms"][mode]}), flush=True)
    finally:
        with psycopg.connect(cfg.conninfo, autocommit=True) as control:
            control_cur = control.cursor()
            fig5.prepare_pg(control_cur)
            bench.set_heap_competing_indexes_valid(control_cur, fig5.TABLE, valid=True)
            control_cur.execute("RESET ROLE")
    if payload["arms"]["stock"]["qps"]:
        payload["speedup"] = payload["arms"]["d1_d2_d3"]["qps"] / payload["arms"]["stock"]["qps"]
    (args.out_dir / "score.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
