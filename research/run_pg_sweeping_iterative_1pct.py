"""PG-sweeping at 1% with iterative_scan=strict_order (fills overfetch)."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import psycopg

from fig1_four_curve_frontier import row_pred, write_csv
from fig1_iso_recall_1pct import (
    DEFAULT_FBIN,
    DEFAULT_QUERY_CSV,
    DEFAULT_TABLE,
    brute_gt,
    load_query_ids,
    read_fbin,
    recall_at_k,
    vec_text,
)
from pg_conn import pg_conninfo
from hnswlib_vs_pgvector_selectivity import pg_configure, timed_ms

OUT = Path(__file__).resolve().parents[1] / "research/results/fig1_iso_recall_1pct/pg_sweeping_iterative.csv"
OVERFETCH = [100, 250, 500, 1000, 2000, 4000]
EF = 128
REPEATS = 5
K = 10
SEL = 1


def main() -> None:
    vectors = read_fbin(DEFAULT_FBIN, 200000)
    query_ids = load_query_ids(DEFAULT_QUERY_CSV, 100)
    gt = brute_gt(vectors, query_ids, SEL, K)
    rows = []
    with psycopg.connect(pg_conninfo("55438"), autocommit=True) as conn:
        cur = conn.cursor()
        pg_configure(cur, EF, 200000, 30000, "strict_order", "off", 128, "off")
        cur.execute("SET hnsw.iterative_scan = strict_order")
        cur.execute("SET hnsw.filter_strategy = off")
        cur.execute("SHOW hnsw.iterative_scan")
        print("iterative", cur.fetchone()[0], flush=True)
        for overfetch in OVERFETCH:
            cur.execute(f"SET hnsw.ef_search = {EF}")
            lats: list[float] = []
            recs: list[float] = []
            rets: list[int] = []
            for qid in query_ids:
                qvec = vec_text(vectors[qid])

                def run(qvec=qvec, overfetch=overfetch):
                    cur.execute(
                        f"""
                        WITH candidates AS MATERIALIZED (
                          SELECT id, bucket, embedding <-> %s::vector AS dist
                          FROM {DEFAULT_TABLE}
                          ORDER BY embedding <-> %s::vector
                          LIMIT {int(overfetch)}
                        )
                        SELECT id FROM candidates
                        WHERE {row_pred(SEL)}
                        ORDER BY dist
                        LIMIT {K}
                        """,
                        (qvec, qvec),
                    )
                    return [int(r[0]) for r in cur.fetchall()]

                ids: list[int] = []
                for rep in range(REPEATS):
                    got, ms = timed_ms(run)
                    lats.append(ms)
                    if rep == 0:
                        ids = got
                recs.append(recall_at_k(ids, gt[qid], K))
                rets.append(len(ids))
            row = {
                "system": "PGVector-sweeping",
                "ef_search": EF,
                "overfetch": overfetch,
                "recall_mean": statistics.mean(recs),
                "latency_ms_mean": statistics.mean(lats),
                "latency_ms_p50": statistics.median(lats),
                "returned_mean": statistics.mean(rets),
                "n_latency": len(lats),
                "n_recall": len(recs),
            }
            rows.append(row)
            print(
                f"of={overfetch:4d} recall={row['recall_mean']:.3f} "
                f"lat={row['latency_ms_mean']:.2f}ms ret={row['returned_mean']:.2f}",
                flush=True,
            )
    write_csv(OUT, rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
