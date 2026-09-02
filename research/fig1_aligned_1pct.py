"""Figure 1 aligned 1% recall–latency retest.

Fixed selectivity = 1%. One quality knob per family. M=32 / efc=200.

  FAISS-ACORN / PG-ACORN: vary ef; iterative_scan=off on PG.
  HNSWlib-sweeping / PG-sweeping: vary overfetch; ef_search = overfetch
  (no max(ef, overfetch) boost).

Queries: 1% slice of the frozen 5k mixed workload (715 queries).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fig1_four_curve_frontier import (
    activate_bucket_acorn,
    attach_recall,
    build_ground_truth,
    read_fbin,
    row_pred,
    run_acorn_cpp,
    vec_text,
    write_csv,
)
from hnswlib_vs_pgvector_selectivity import (
    deactivate_pg_acorn1,
    ensure_guidance_functions,
    ensure_guidance_meta,
    pg_configure,
    timed_ms,
)
from pg_conn import pg_conninfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FBIN = ROOT / "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"
DEFAULT_TABLE = "amazon_fig1_200k"
DEFAULT_INDEX = "amazon_fig1_200k_hnsw"
DEFAULT_WORKLOAD = ROOT / "research/results/fig1_four_curve_m32/workload.csv"
DEFAULT_HNSWLIB = ROOT / "research/results/fig1_four_curve_m32/hnswlib_fig1_200k_m32_efc200.bin"
DEFAULT_OUT = ROOT / "research/results/fig1_aligned_1pct"
ACORN_EFS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
SWEEP_OVERFETCH = [20, 50, 100, 200, 400, 800, 2000, 4000]
SYSTEMS = ["FAISS-ACORN", "PGVector-ACORN", "HNSWlib-sweeping", "PGVector-sweeping"]
COLORS = {
    "FAISS-ACORN": "#F58518",
    "HNSWlib-sweeping": "#4C78A8",
    "PGVector-ACORN": "#F58518",
    "PGVector-sweeping": "#4C78A8",
}
MARKERS = {
    "FAISS-ACORN": "o",
    "HNSWlib-sweeping": "o",
    "PGVector-ACORN": "s",
    "PGVector-sweeping": "^",
}
LINESTYLES = {
    "FAISS-ACORN": "-",
    "HNSWlib-sweeping": "-",
    "PGVector-ACORN": "--",
    "PGVector-sweeping": "--",
}


def load_one_pct(path: Path) -> list[dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]
    out = [r for r in rows if r["selectivity_pct"] == 1]
    if not out:
        raise RuntimeError(f"no 1% queries in {path}")
    return out


def hnswlib_sweeping(
    index: hnswlib.Index,
    vectors: np.ndarray,
    workload: list[dict[str, int]],
    overfetches: list[int],
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for overfetch in overfetches:
        index.set_ef(int(overfetch))
        for req in workload:
            q = vectors[int(req["query_id"])]

            def run(q=q, overfetch=overfetch):
                labels, _ = index.knn_query(q.reshape(1, -1), k=int(overfetch), num_threads=1)
                out: list[int] = []
                for label in labels[0]:
                    value = int(label)
                    if (value % 100) < 1:
                        out.append(value)
                        if len(out) >= k:
                            break
                return out

            ids, elapsed = timed_ms(run)
            rows.append(
                {
                    "system": "HNSWlib-sweeping",
                    "ef_search": overfetch,
                    "query_no": req["query_no"],
                    "query_id": req["query_id"],
                    "selectivity_pct": 1,
                    "k": k,
                    "overfetch": overfetch,
                    "latency_ms": elapsed,
                    "returned": len(ids),
                    "ids": ";".join(map(str, ids)),
                }
            )
        print(f"HNSWlib-sweeping finished overfetch={overfetch}", flush=True)
    return rows


def pg_sweeping(
    cur,
    table: str,
    vectors: np.ndarray,
    workload: list[dict[str, int]],
    overfetches: list[int],
    k: int,
    timeout_ms: int,
) -> list[dict[str, Any]]:
    pred = row_pred(1)
    rows: list[dict[str, Any]] = []
    for overfetch in overfetches:
        cur.execute(f"SET hnsw.ef_search = {int(overfetch)}")
        cur.execute("SET hnsw.iterative_scan = off")
        cur.execute("SET hnsw.filter_strategy = off")
        for req in workload:
            qvec = vec_text(vectors[int(req["query_id"])])

            def run(qvec=qvec, overfetch=overfetch):
                cur.execute(
                    f"""
                    WITH candidates AS MATERIALIZED (
                      SELECT id, bucket, embedding <-> %s::vector AS dist
                      FROM {table}
                      ORDER BY embedding <-> %s::vector
                      LIMIT {int(overfetch)}
                    )
                    SELECT id FROM candidates
                    WHERE {pred}
                    ORDER BY dist
                    LIMIT {int(k)}
                    """,
                    (qvec, qvec),
                )
                return [int(r[0]) for r in cur.fetchall()]

            try:
                ids, elapsed = timed_ms(run)
                error = ""
            except Exception as exc:
                cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
                ids, elapsed, error = [], float(timeout_ms), type(exc).__name__
            rows.append(
                {
                    "system": "PGVector-sweeping",
                    "ef_search": overfetch,
                    "query_no": req["query_no"],
                    "query_id": req["query_id"],
                    "selectivity_pct": 1,
                    "k": k,
                    "overfetch": overfetch,
                    "latency_ms": elapsed,
                    "returned": len(ids),
                    "error": error,
                    "ids": ";".join(map(str, ids)),
                }
            )
        print(f"PGVector-sweeping finished overfetch={overfetch}", flush=True)
    return rows


def pg_acorn(
    cur,
    table: str,
    index_name: str,
    vectors: np.ndarray,
    workload: list[dict[str, int]],
    efs: list[int],
    k: int,
    timeout_ms: int,
    guidance_kind: str,
) -> list[dict[str, Any]]:
    pred = row_pred(1)
    rows: list[dict[str, Any]] = []
    for ef in efs:
        cur.execute(f"SET hnsw.ef_search = {int(ef)}")
        cur.execute("SET hnsw.iterative_scan = off")
        activate_bucket_acorn(cur, index_name, 1, guidance_kind)
        try:
            for req in workload:
                qvec = vec_text(vectors[int(req["query_id"])])

                def run(qvec=qvec):
                    cur.execute(
                        f"""
                        SELECT id FROM {table}
                        WHERE {pred}
                        ORDER BY embedding <-> %s::vector
                        LIMIT {int(k)}
                        """,
                        (qvec,),
                    )
                    return [int(r[0]) for r in cur.fetchall()]

                try:
                    ids, elapsed = timed_ms(run)
                    error = ""
                except Exception as exc:
                    cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
                    ids, elapsed, error = [], float(timeout_ms), type(exc).__name__
                rows.append(
                    {
                        "system": "PGVector-ACORN",
                        "ef_search": ef,
                        "query_no": req["query_no"],
                        "query_id": req["query_id"],
                        "selectivity_pct": 1,
                        "k": k,
                        "overfetch": "",
                        "latency_ms": elapsed,
                        "returned": len(ids),
                        "error": error,
                        "ids": ";".join(map(str, ids)),
                    }
                )
        finally:
            deactivate_pg_acorn1(cur)
        print(f"PGVector-ACORN finished ef={ef}", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["system"]), int(row["ef_search"])), []).append(row)
    out = []
    for (system, knob), items in sorted(groups.items()):
        lat = [float(r["latency_ms"]) for r in items]
        rec = [float(r["recall_at_k"]) for r in items]
        over = items[0].get("overfetch", "")
        out.append(
            {
                "system": system,
                "ef_search": knob,
                "overfetch": over,
                "queries": len(items),
                "latency_ms_mean": statistics.mean(lat),
                "latency_ms_p50": statistics.median(lat),
                "recall_at_10_mean": statistics.mean(rec),
            }
        )
    return out


def plot_frontier(summary: list[dict[str, Any]], out_paths: list[Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for system in SYSTEMS:
        items = sorted(
            [r for r in summary if r["system"] == system],
            key=lambda r: (float(r["latency_ms_mean"]), -float(r["recall_at_10_mean"])),
        )
        if not items:
            continue
        kept, best = [], -1.0
        for row in items:
            rec = float(row["recall_at_10_mean"])
            if rec > best + 1e-6:
                kept.append(row)
                best = rec
        ax.plot(
            [float(r["latency_ms_mean"]) for r in kept],
            [float(r["recall_at_10_mean"]) for r in kept],
            label=system,
            color=COLORS[system],
            marker=MARKERS[system],
            linestyle=LINESTYLES[system],
            linewidth=1.6,
            markersize=5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, which="both", linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for dest in out_paths:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, bbox_inches="tight", dpi=300)
        print(f"wrote {dest}", flush=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conninfo", default=None)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--pg-index", default=DEFAULT_INDEX)
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--rows", type=int, default=200000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--guidance-kind", default="page")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--gamma", type=int, default=1)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--hnswlib-index", type=Path, default=DEFAULT_HNSWLIB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-queries", type=int, default=0, help="0 = all 1% queries")
    parser.add_argument(
        "--systems",
        default=",".join(SYSTEMS),
        help="comma-separated subset of " + ",".join(SYSTEMS),
    )
    args = parser.parse_args()
    if not args.conninfo:
        args.conninfo = pg_conninfo("55438")
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    workload = load_one_pct(args.workload)
    if args.max_queries > 0:
        workload = workload[: args.max_queries]
    write_csv(out_dir / "workload.csv", workload)
    print(f"1% queries={len(workload)} from {args.workload}", flush=True)

    vectors = read_fbin(args.fbin, args.rows)
    n = len(vectors)
    print("building exact ground truth for 1% slice", flush=True)
    t0 = time.perf_counter()
    gt = build_ground_truth(vectors, workload, args.k)
    print(f"ground truth elapsed_s={time.perf_counter() - t0:.1f}", flush=True)

    all_rows: list[dict[str, Any]] = []
    if "HNSWlib-sweeping" in systems:
        index = hnswlib.Index(space="l2", dim=int(vectors.shape[1]))
        index.load_index(str(args.hnswlib_index), max_elements=n)
        all_rows.extend(hnswlib_sweeping(index, vectors, workload, SWEEP_OVERFETCH, args.k))

    if "FAISS-ACORN" in systems:
        raw = out_dir / "faiss_acorn_raw.csv"
        acorn_rows = run_acorn_cpp(
            ROOT,
            args.fbin,
            out_dir / "workload.csv",
            raw,
            args.rows,
            args.k,
            ACORN_EFS,
            m=args.m,
            efc=args.ef_construction,
            gamma=args.gamma,
        )
        for row in acorn_rows:
            row["system"] = "FAISS-ACORN"
            row["overfetch"] = ""
        all_rows.extend(acorn_rows)

    if any(s.startswith("PGVector") for s in systems):
        with psycopg.connect(args.conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            pg_configure(cur, ACORN_EFS[0], 200000, args.timeout_ms, "off", "off", 128, "off")
            ensure_guidance_functions(cur)
            ensure_guidance_meta(cur, args.table)
            cur.execute("SET hnsw.guidance_require_epoch = off")
            try:
                cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", (args.table,))
            except Exception as exc:
                print(f"fragment tracking enable skipped: {exc}", flush=True)
            cur.execute(f"SELECT count(*) FROM {args.table}")
            print(f"pg rows={cur.fetchone()[0]} index={args.pg_index}", flush=True)
            if "PGVector-ACORN" in systems:
                all_rows.extend(
                    pg_acorn(
                        cur,
                        args.table,
                        args.pg_index,
                        vectors,
                        workload,
                        ACORN_EFS,
                        args.k,
                        args.timeout_ms,
                        args.guidance_kind,
                    )
                )
            if "PGVector-sweeping" in systems:
                all_rows.extend(
                    pg_sweeping(
                        cur,
                        args.table,
                        vectors,
                        workload,
                        SWEEP_OVERFETCH,
                        args.k,
                        args.timeout_ms,
                    )
                )

    scored = attach_recall(all_rows, gt, args.k)
    write_csv(out_dir / "detail.csv", scored)
    summary = summarize(scored)
    write_csv(out_dir / "frontier_summary.csv", summary)
    print("frontier summary:", flush=True)
    for row in summary:
        print(
            f"  {row['system']:20s} knob={row['ef_search']:5}  "
            f"recall={row['recall_at_10_mean']:.4f}  "
            f"lat={row['latency_ms_mean']:.3f}ms",
            flush=True,
        )
    plot_frontier(
        summary,
        [
            out_dir / "fig_intro_recall_latency_frontier_1pct.pdf",
            ROOT / "paper/figures/fig_intro_recall_latency_frontier_1pct.pdf",
        ],
    )


if __name__ == "__main__":
    main()
