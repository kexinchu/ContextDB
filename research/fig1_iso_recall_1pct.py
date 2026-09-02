"""Retest Figure 1 at the old 13x / 6x operating point.

Protocol matches paper/scripts/plot_hnswlib_pgvector_comparison.py:
  200k Amazon subset, M=16, efc=64, 100 query IDs, 1% selectivity,
  10 latency repeats.

Unlike the mixed-5k iso-recall run:
  * HNSWlib-sweeping uses requested ef, not max(ef, overfetch).
  * Sweeping also varies overfetch so recall (and latency) can move.
  * PG-ACORN uses page guidance + iterative_scan=off, as in the old figure.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hnswlib
import numpy as np
import psycopg

from fig1_four_curve_frontier import (
    COLORS,
    LINESTYLES,
    MARKERS,
    activate_bucket_acorn,
    parse_ids,
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
DEFAULT_QUERY_CSV = ROOT / "research/results/amazon_200k_query_ids_100.csv"
DEFAULT_HNSWLIB = ROOT / "research/results/hnswlib_amazon_200k_m16_ef64.bin"
DEFAULT_TARGETS = [0.75, 0.80, 0.85, 0.90, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
ACORN_EFS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
SWEEP_OVERFETCH = [100, 250, 500, 1000, 2000, 4000]
SWEEP_EFS = [16, 32, 64, 128, 256]
SYSTEMS = ["HNSWlib-ACORN", "PGVector-ACORN", "HNSWlib-sweeping", "PGVector-sweeping"]


def load_query_ids(path: Path, n: int) -> list[int]:
    ids: list[int] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["query_id"]))
            if len(ids) >= n:
                break
    if len(ids) < n:
        raise RuntimeError(f"only {len(ids)} query ids in {path}")
    return ids


def read_fbin(path: Path, rows: int) -> np.ndarray:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
        n = min(n, rows)
        return np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)


def recall_at_k(pred: list[int], truth: list[int], k: int) -> float:
    if not truth:
        return 0.0
    return len(set(pred[:k]) & set(truth[:k])) / float(min(k, len(truth)))


def brute_gt(vectors: np.ndarray, query_ids: list[int], sel: int, k: int) -> dict[int, list[int]]:
    n = len(vectors)
    mask = np.array([(i % 100) < sel for i in range(n)], dtype=bool) if sel < 100 else np.ones(n, dtype=bool)
    allowed = np.nonzero(mask)[0]
    subset = vectors[allowed]
    out: dict[int, list[int]] = {}
    t0 = time.time()
    for qid in query_ids:
        dist = np.linalg.norm(subset - vectors[qid], axis=1)
        top = allowed[np.argpartition(dist, kth=min(k, len(dist) - 1))[:k]]
        top = top[np.argsort(np.linalg.norm(vectors[top] - vectors[qid], axis=1))]
        out[qid] = [int(x) for x in top[:k]]
    print(f"ground truth 1% done in {time.time() - t0:.1f}s", flush=True)
    return out


def summarize(
    rows: list[dict[str, Any]],
    gt: dict[int, list[int]],
    k: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["system"], int(row["ef_search"]), int(row.get("overfetch", 0)))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (system, ef, overfetch), items in sorted(groups.items()):
        ok = [r for r in items if not r.get("error")]
        if not ok:
            continue
        first = [r for r in ok if int(r.get("repeat", 0)) == 0] or ok
        recalls = [recall_at_k(parse_ids(str(r.get("ids", ""))), gt[int(r["query_id"])], k) for r in first]
        lats = [float(r["latency_ms"]) for r in ok]
        out.append(
            {
                "system": system,
                "ef_search": ef,
                "overfetch": overfetch,
                "n_latency": len(lats),
                "n_recall": len(recalls),
                "recall_mean": statistics.mean(recalls),
                "latency_ms_mean": statistics.mean(lats),
                "latency_ms_p50": statistics.median(lats),
            }
        )
    return out


def iso_recall(summary: list[dict[str, Any]], targets: list[float]) -> list[dict[str, Any]]:
    by_sys: dict[str, list[dict[str, Any]]] = {}
    for row in summary:
        by_sys.setdefault(str(row["system"]), []).append(row)
    out: list[dict[str, Any]] = []
    for system, items in by_sys.items():
        items = sorted(items, key=lambda r: (float(r["latency_ms_mean"]), -float(r["recall_mean"])))
        for target in targets:
            chosen = next((r for r in items if float(r["recall_mean"]) + 1e-12 >= target), None)
            out.append(
                {
                    "system": system,
                    "target_recall": target,
                    "reached": chosen is not None,
                    "ef_search": chosen["ef_search"] if chosen else "",
                    "overfetch": chosen["overfetch"] if chosen else "",
                    "achieved_recall": chosen["recall_mean"] if chosen else "",
                    "latency_ms": chosen["latency_ms_mean"] if chosen else "",
                }
            )
    return out


def plot_frontier(summary: list[dict[str, Any]], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for system in SYSTEMS:
        pts = sorted(
            (float(r["latency_ms_mean"]), float(r["recall_mean"]))
            for r in summary
            if r["system"] == system
        )
        if not pts:
            continue
        # Pareto: keep points that improve recall as latency grows.
        xs, ys, best = [], [], -1.0
        for x, y in pts:
            if y > best + 1e-6:
                xs.append(x)
                ys.append(y)
                best = y
        ax.plot(
            xs,
            ys,
            color=COLORS[system],
            linestyle=LINESTYLES[system],
            marker=MARKERS[system],
            label=system,
            linewidth=1.8,
            markersize=6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Achieved Recall@10")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    plt.close(fig)


def print_old_protocol(summary: list[dict[str, Any]]) -> None:
    def find(system: str, ef: int, overfetch: int) -> dict[str, Any] | None:
        hits = [
            r for r in summary
            if r["system"] == system and int(r["ef_search"]) == ef and int(r["overfetch"]) == overfetch
        ]
        return hits[0] if hits else None

    fa = find("HNSWlib-ACORN", 128, 0)
    pa = find("PGVector-ACORN", 256, 0)
    hs = find("HNSWlib-sweeping", 128, 4000)
    ps = find("PGVector-sweeping", 128, 4000)
    print("\n=== old-figure operating point (1% sel, M=16 efc=64) ===", flush=True)
    for name, row in (
        ("FAISS-ACORN ef=128", fa),
        ("PG-ACORN ef=256 page", pa),
        ("HNSWlib-sweeping of=4000 ef=128", hs),
        ("PG-sweeping of=4000 ef=128", ps),
    ):
        if row:
            print(
                f"  {name:34s} recall={float(row['recall_mean']):.3f}  "
                f"lat={float(row['latency_ms_mean']):.3f}ms",
                flush=True,
            )
        else:
            print(f"  {name:34s} missing", flush=True)
    if fa and pa:
        print(f"  ACORN gap   PG/FAISS = {float(pa['latency_ms_mean'])/float(fa['latency_ms_mean']):.2f}x", flush=True)
    if hs and ps:
        print(f"  sweep gap   PG/HNSW  = {float(ps['latency_ms_mean'])/float(hs['latency_ms_mean']):.2f}x", flush=True)


def plot_iso(iso: list[dict[str, Any]], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for system in SYSTEMS:
        xs, ys = [], []
        seen: set[tuple[float, float]] = set()
        for row in iso:
            if row["system"] != system or not row["reached"]:
                continue
            x, y = float(row["latency_ms"]), float(row["target_recall"])
            key = (round(x, 4), round(y, 4))
            if key in seen:
                continue
            seen.add(key)
            xs.append(x)
            ys.append(y)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            color=COLORS[system],
            linestyle=LINESTYLES[system],
            marker=MARKERS[system],
            label=system,
            linewidth=1.8,
            markersize=6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Recall@10 target")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    plt.close(fig)


def print_ratios(iso: list[dict[str, Any]]) -> None:
    by: dict[tuple[str, float], dict[str, Any]] = {
        (str(r["system"]), float(r["target_recall"])): r for r in iso if r["reached"]
    }
    print("\n=== iso-recall at 1% selectivity ===", flush=True)
    print(f"{'T':>6}  {'FAISS-ACORN':>12}  {'PG-ACORN':>10}  {'ACORN x':>8}  "
          f"{'HNSW-sw':>10}  {'PG-sw':>10}  {'sweep x':>8}", flush=True)
    for target in DEFAULT_TARGETS:
        fa = by.get(("HNSWlib-ACORN", target))
        pa = by.get(("PGVector-ACORN", target))
        hs = by.get(("HNSWlib-sweeping", target))
        ps = by.get(("PGVector-sweeping", target))
        def fmt(row: dict[str, Any] | None) -> str:
            return f"{float(row['latency_ms']):10.3f}" if row else f"{'—':>10}"
        acorn_x = (float(pa["latency_ms"]) / float(fa["latency_ms"])) if fa and pa else float("nan")
        sweep_x = (float(ps["latency_ms"]) / float(hs["latency_ms"])) if hs and ps else float("nan")
        print(
            f"{target:6.2f}  {fmt(fa):>12}  {fmt(pa):>10}  {acorn_x:8.2f}  "
            f"{fmt(hs):>10}  {fmt(ps):>10}  {sweep_x:8.2f}",
            flush=True,
        )


def run_hnswlib_sweeping(
    index: hnswlib.Index,
    vectors: np.ndarray,
    query_ids: list[int],
    sel: int,
    k: int,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for overfetch in SWEEP_OVERFETCH:
        for ef in SWEEP_EFS:
            if ef > overfetch:
                continue
            index.set_ef(max(ef, overfetch))
            for qid in query_ids:
                q = vectors[qid]
                ids: list[int] = []
                for rep in range(repeats):
                    (labels, _), elapsed = timed_ms(
                        lambda q=q, overfetch=overfetch: index.knn_query(q.reshape(1, -1), k=overfetch, num_threads=1)
                    )
                    if rep == 0:
                        ids = []
                        for label in labels[0]:
                            value = int(label)
                            if (value % 100) < sel:
                                ids.append(value)
                                if len(ids) >= k:
                                    break
                    rows.append(
                        {
                            "system": "HNSWlib-sweeping",
                            "ef_search": ef,
                            "overfetch": overfetch,
                            "query_id": qid,
                            "repeat": rep,
                            "latency_ms": elapsed,
                            "returned": len(ids),
                            "error": "",
                            "ids": ";".join(map(str, ids)) if rep == 0 else "",
                        }
                    )
            print(f"HNSWlib-sweeping ef={ef} overfetch={overfetch}", flush=True)
    return rows


def run_pg_acorn(
    cur,
    table: str,
    index_name: str,
    vectors: np.ndarray,
    query_ids: list[int],
    sel: int,
    k: int,
    repeats: int,
    timeout_ms: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ef in ACORN_EFS:
        cur.execute(f"SET hnsw.ef_search = {int(ef)}")
        for qid in query_ids:
            qvec = vec_text(vectors[qid])

            pred = row_pred(sel)

            def run(qvec=qvec, pred=pred):
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

            ids: list[int] = []
            for rep in range(repeats):
                activate_bucket_acorn(cur, index_name, sel, "page")
                try:
                    got, elapsed = timed_ms(run)
                    error = ""
                except Exception as exc:
                    cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
                    got, elapsed, error = [], float(timeout_ms), type(exc).__name__
                deactivate_pg_acorn1(cur)
                if rep == 0:
                    ids = got
                rows.append(
                    {
                        "system": "PGVector-ACORN",
                        "ef_search": ef,
                        "overfetch": 0,
                        "query_id": qid,
                        "repeat": rep,
                        "latency_ms": elapsed,
                        "returned": len(ids),
                        "error": error,
                        "ids": ";".join(map(str, ids)) if rep == 0 else "",
                    }
                )
        print(f"PGVector-ACORN finished ef={ef}", flush=True)
    return rows


def run_pg_sweeping(
    cur,
    table: str,
    vectors: np.ndarray,
    query_ids: list[int],
    sel: int,
    k: int,
    repeats: int,
    timeout_ms: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute("SET hnsw.iterative_scan = strict_order")
    cur.execute("SET hnsw.max_scan_tuples = 200000")
    for overfetch in SWEEP_OVERFETCH:
        for ef in SWEEP_EFS:
            if ef > overfetch:
                continue
            cur.execute(f"SET hnsw.ef_search = {int(ef)}")
            for qid in query_ids:
                qvec = vec_text(vectors[qid])

                pred = row_pred(sel)

                def run(qvec=qvec, overfetch=overfetch, pred=pred):
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

                ids: list[int] = []
                for rep in range(repeats):
                    try:
                        got, elapsed = timed_ms(run)
                        error = ""
                    except Exception as exc:
                        cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
                        got, elapsed, error = [], float(timeout_ms), type(exc).__name__
                    if rep == 0:
                        ids = got
                    rows.append(
                        {
                            "system": "PGVector-sweeping",
                            "ef_search": ef,
                            "overfetch": overfetch,
                            "query_id": qid,
                            "repeat": rep,
                            "latency_ms": elapsed,
                            "returned": len(ids),
                            "error": error,
                            "ids": ";".join(map(str, ids)) if rep == 0 else "",
                        }
                    )
            print(f"PGVector-sweeping ef={ef} overfetch={overfetch}", flush=True)
    return rows


def run_faiss_acorn(
    root: Path,
    fbin: Path,
    workload_csv: Path,
    out_csv: Path,
    query_ids: list[int],
    sel: int,
    k: int,
) -> list[dict[str, Any]]:
    write_csv(
        workload_csv,
        [{"query_no": i, "query_id": qid, "selectivity_pct": sel} for i, qid in enumerate(query_ids)],
    )
    raw = run_acorn_cpp(
        root, fbin, workload_csv, out_csv, 200000, k, ACORN_EFS, m=16, efc=64, gamma=1,
    )
    rows: list[dict[str, Any]] = []
    for row in raw:
        rows.append(
            {
                "system": "HNSWlib-ACORN",
                "ef_search": int(row["ef_search"]),
                "overfetch": 0,
                "query_id": int(row["query_id"]),
                "repeat": 0,
                "latency_ms": float(row["latency_ms"]),
                "returned": int(row.get("returned") or 0),
                "error": "",
                "ids": row.get("ids", ""),
            }
        )
    print(f"FAISS ACORN-1 finished n={len(rows)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "research/results/fig1_iso_recall_1pct")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--selectivity", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=20000)
    parser.add_argument("--conninfo", default=None)
    parser.add_argument(
        "--systems",
        default="HNSWlib-ACORN,HNSWlib-sweeping,PGVector-ACORN,PGVector-sweeping",
    )
    args = parser.parse_args()
    conninfo = args.conninfo or pg_conninfo("55438")
    wanted = {x.strip() for x in args.systems.split(",") if x.strip()}

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = read_fbin(DEFAULT_FBIN, 200000)
    query_ids = load_query_ids(DEFAULT_QUERY_CSV, args.queries)
    print(f"loaded vectors {vectors.shape} queries={len(query_ids)}", flush=True)
    gt = brute_gt(vectors, query_ids, args.selectivity, args.k)

    def load_detail_system(system: str) -> list[dict[str, Any]]:
        path = out_dir / "detail.csv"
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return [r for r in csv.DictReader(f) if r["system"] == system and not r.get("error")]

    all_rows: list[dict[str, Any]] = []
    faiss_csv = out_dir / "faiss_acorn.csv"
    if "HNSWlib-ACORN" in wanted:
        if faiss_csv.exists() and faiss_csv.stat().st_size > 0:
            print(f"reusing {faiss_csv}", flush=True)
            with faiss_csv.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    all_rows.append(
                        {
                            "system": "HNSWlib-ACORN",
                            "ef_search": int(row["ef_search"]),
                            "overfetch": 0,
                            "query_id": int(row["query_id"]),
                            "repeat": 0,
                            "latency_ms": float(row["latency_ms"]),
                            "returned": int(row.get("returned") or 0),
                            "error": "",
                            "ids": row.get("ids", ""),
                        }
                    )
        else:
            all_rows.extend(
                run_faiss_acorn(
                    ROOT, DEFAULT_FBIN, out_dir / "workload_1pct.csv", faiss_csv,
                    query_ids, args.selectivity, args.k,
                )
            )

    if "HNSWlib-sweeping" in wanted:
        hnsw_csv = out_dir / "hnswlib_sweeping.csv"
        if hnsw_csv.exists() and hnsw_csv.stat().st_size > 0:
            print(f"reusing {hnsw_csv}", flush=True)
            with hnsw_csv.open(encoding="utf-8") as f:
                all_rows.extend(list(csv.DictReader(f)))
        else:
            index = hnswlib.Index(space="l2", dim=vectors.shape[1])
            index.load_index(str(DEFAULT_HNSWLIB))
            index.set_num_threads(1)
            hnsw_rows = run_hnswlib_sweeping(index, vectors, query_ids, args.selectivity, args.k, args.repeats)
            write_csv(hnsw_csv, hnsw_rows)
            all_rows.extend(hnsw_rows)

    need_pg = bool(wanted & {"PGVector-ACORN", "PGVector-sweeping"})
    if need_pg:
        with psycopg.connect(conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            pg_configure(cur, ACORN_EFS[0], 200000, args.timeout_ms, "off", "off", 128, "off")
            ensure_guidance_functions(cur)
            ensure_guidance_meta(cur, DEFAULT_TABLE)
            if "PGVector-ACORN" in wanted:
                cached = load_detail_system("PGVector-ACORN")
                if cached:
                    print(f"reusing {len(cached)} PGVector-ACORN rows", flush=True)
                    all_rows.extend(cached)
                else:
                    all_rows.extend(
                        run_pg_acorn(
                            cur, DEFAULT_TABLE, DEFAULT_INDEX, vectors, query_ids,
                            args.selectivity, args.k, args.repeats, args.timeout_ms,
                        )
                    )
            if "PGVector-sweeping" in wanted:
                all_rows.extend(
                    run_pg_sweeping(
                        cur, DEFAULT_TABLE, vectors, query_ids,
                        args.selectivity, args.k, args.repeats, args.timeout_ms,
                    )
                )

    write_csv(out_dir / "detail.csv", all_rows)
    summary = summarize(all_rows, gt, args.k)
    write_csv(out_dir / "frontier_summary.csv", summary)
    iso = iso_recall(summary, DEFAULT_TARGETS)
    write_csv(out_dir / "iso_recall.csv", iso)
    plot_frontier(
        summary,
        out_dir / "fig_intro_recall_latency_1pct.pdf",
        "1% selectivity, 200k, M=16, efc=64, q100 × r10",
    )
    plot_iso(
        iso,
        out_dir / "fig_intro_recall_latency_1pct_iso.pdf",
        "Iso-recall @ 1% selectivity",
    )
    print_old_protocol(summary)
    print_ratios(iso)
    print("summary rows:", len(summary), "detail:", len(all_rows), flush=True)
    for row in summary:
        print(
            f"  {row['system']:20s} ef={row['ef_search']:4} of={row['overfetch']:5} "
            f"recall={row['recall_mean']:.4f} lat={row['latency_ms_mean']:.3f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
