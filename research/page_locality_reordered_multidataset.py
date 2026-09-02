from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import psycopg

from pg_conn import pg_conninfo


DATASETS = {
    "amazon": {
        "label": "Amazon-10M",
        "source": "amazon_grocery_reviews_10m_pgvector",
        "reordered": "amazon_grocery_reviews_10m_pgvector_vector_clustered_10m",
        "id": "id",
    },
    "msmarco": {
        "label": "MS MARCO-1M",
        "source": "msmarco_kill_passages",
        "reordered": "msmarco_kill_passages_vector_clustered",
        "id": "pid",
    },
    "enron": {
        "label": "Enron-50K",
        "source": "enron_messages",
        "reordered": "enron_messages_vector_clustered",
        "id": "message_id",
    },
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_or_create_query_ids(cur, table: str, id_col: str, path: Path, queries: int) -> list[int]:
    if path.exists():
        ids: list[int] = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ids.append(int(row["query_id"]))
                if len(ids) >= queries:
                    return ids
    path.parent.mkdir(parents=True, exist_ok=True)
    cur.execute(f"SELECT {id_col} FROM {table} ORDER BY md5({id_col}::text) LIMIT %s", (queries,))
    ids = [int(r[0]) for r in cur.fetchall()]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id"])
        writer.writerows([[x] for x in ids])
    return ids


def block_from_ctid(ctid: str) -> int:
    return int(str(ctid).strip("()").split(",", 1)[0])


def count_runs(blocks: list[int]) -> int:
    if not blocks:
        return 0
    runs = 1
    prev = blocks[0]
    for block in blocks[1:]:
        if block != prev:
            runs += 1
            prev = block
    return runs


def p95(values: list[float]) -> float:
    vals = sorted(values)
    idx = max(0, int(0.95 * len(vals)) - 1)
    return vals[idx] if vals else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for dataset in sorted({str(r["dataset"]) for r in rows}):
        items = [r for r in rows if r["dataset"] == dataset]

        def vals(key: str) -> list[float]:
            return [float(r[key]) for r in items]

        out.append(
            {
                "dataset": dataset,
                "label": items[0]["label"],
                "queries": len(items),
                "candidate_limit": items[0]["candidate_limit"],
                "candidate_count_mean": statistics.mean(vals("candidate_count")),
                "search_ms_mean": statistics.mean(vals("search_ms")),
                "search_ms_p95": p95(vals("search_ms")),
                "source_distance_page_runs_mean": statistics.mean(vals("source_distance_page_runs")),
                "source_distinct_pages_mean": statistics.mean(vals("source_distinct_pages")),
                "reordered_distinct_pages_mean": statistics.mean(vals("reordered_distinct_pages")),
                "page_footprint_reduction_mean": statistics.mean(vals("page_footprint_reduction")),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conninfo", default=None)
    parser.add_argument("--datasets", default="amazon,msmarco,enron")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=1000)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("research/results/page_locality_reordered_multidataset_q100_c1000.csv"))
    args = parser.parse_args()
    if not args.conninfo:
        args.conninfo = pg_conninfo("55432")

    rows: list[dict[str, Any]] = []
    with psycopg.connect(args.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("SET jit = off")
        cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
        cur.execute("SET hnsw.iterative_scan = off")
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET statement_timeout = 30000")

        for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
            cfg = DATASETS[name]
            cur.execute("SELECT to_regclass(%s)", (cfg["reordered"],))
            if cur.fetchone()[0] is None:
                raise RuntimeError(f"missing reordered table {cfg['reordered']}")
            qpath = Path(f"research/results/{name}_page_locality_query_ids_{args.queries}.csv")
            query_ids = load_or_create_query_ids(cur, cfg["source"], cfg["id"], qpath, args.queries)
            for qno, qid in enumerate(query_ids):
                cur.execute(f"SELECT embedding::text FROM {cfg['source']} WHERE {cfg['id']} = %s", (qid,))
                qvec = cur.fetchone()[0]
                start = time.perf_counter()
                cur.execute(
                    f"""
                    WITH candidates AS MATERIALIZED (
                        SELECT row_number() OVER () AS ord, {cfg['id']} AS row_id, ctid::text AS source_ctid
                        FROM (
                            SELECT {cfg['id']}, ctid
                            FROM {cfg['source']}
                            ORDER BY embedding <-> %s::vector
                            LIMIT %s
                        ) s
                    )
                    SELECT c.ord, c.source_ctid, r.ctid::text AS reordered_ctid
                    FROM candidates c
                    JOIN {cfg['reordered']} r ON r.{cfg['id']} = c.row_id
                    ORDER BY c.ord
                    """,
                    (qvec, args.candidate_limit),
                )
                result = cur.fetchall()
                search_ms = (time.perf_counter() - start) * 1000.0
                source_blocks = [block_from_ctid(row[1]) for row in result]
                reordered_blocks = [block_from_ctid(row[2]) for row in result]
                source_runs = count_runs(source_blocks)
                reordered_pages = len(set(reordered_blocks))
                row = {
                    "dataset": name,
                    "label": cfg["label"],
                    "query_no": qno,
                    "query_id": qid,
                    "candidate_limit": args.candidate_limit,
                    "candidate_count": len(result),
                    "search_ms": search_ms,
                    "source_distance_page_runs": source_runs,
                    "source_distinct_pages": len(set(source_blocks)),
                    "reordered_distinct_pages": reordered_pages,
                    "page_footprint_reduction": source_runs / max(reordered_pages, 1),
                }
                rows.append(row)
                print(
                    f"{name} q={qno} cand={len(result)} runs={source_runs} reordered_pages={reordered_pages}",
                    flush=True,
                )

    write_csv(args.out, rows)
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summarize(rows))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
