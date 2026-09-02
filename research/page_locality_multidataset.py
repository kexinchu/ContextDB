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
        "table": "amazon_grocery_reviews_10m_pgvector",
        "id": "id",
        "index": "amazon_grocery_reviews_10m_pgvector_embedding_hnsw_idx",
    },
    "msmarco": {
        "label": "MS MARCO-1M",
        "table": "msmarco_kill_passages",
        "id": "pid",
        "index": "msmarco_kill_passages_embedding_hnsw",
    },
    "enron": {
        "label": "Enron-50K",
        "table": "enron_messages",
        "id": "message_id",
        "index": "enron_messages_embedding_hnsw",
    },
}


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


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
    if not values:
        return 0.0
    vals = sorted(values)
    idx = max(0, int(0.95 * len(vals)) - 1)
    return vals[idx]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["dataset"]), []).append(row)
    out: list[dict[str, Any]] = []
    for dataset, items in sorted(groups.items()):
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
                "search_ms_p50": statistics.median(vals("search_ms")),
                "search_ms_p95": p95(vals("search_ms")),
                "distinct_heap_pages_mean": statistics.mean(vals("distinct_heap_pages")),
                "distance_order_page_runs_mean": statistics.mean(vals("distance_order_page_runs")),
                "page_sorted_runs_mean": statistics.mean(vals("page_sorted_runs")),
                "runs_per_candidate_mean": statistics.mean(vals("runs_per_candidate")),
                "runs_per_page_mean": statistics.mean(vals("runs_per_page")),
                "page_sort_run_reduction_mean": statistics.mean(vals("page_sort_run_reduction")),
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
    parser.add_argument("--out", type=Path, default=Path("research/results/page_locality_multidataset_q100_c1000.csv"))
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
            qpath = Path(f"research/results/{name}_page_locality_query_ids_{args.queries}.csv")
            query_ids = load_or_create_query_ids(cur, cfg["table"], cfg["id"], qpath, args.queries)
            for qno, qid in enumerate(query_ids):
                cur.execute(
                    f"SELECT embedding::text FROM {cfg['table']} WHERE {cfg['id']} = %s",
                    (qid,),
                )
                qvec = cur.fetchone()[0]

                def run():
                    cur.execute(
                        f"""
                        SELECT {cfg['id']} AS row_id, ctid::text
                        FROM {cfg['table']}
                        ORDER BY embedding <-> %s::vector
                        LIMIT %s
                        """,
                        (qvec, args.candidate_limit),
                    )
                    return cur.fetchall()

                candidates, search_ms = timed_ms(run)
                blocks = [block_from_ctid(str(ctid)) for _, ctid in candidates]
                distinct_pages = len(set(blocks))
                distance_runs = count_runs(blocks)
                page_sorted_runs = count_runs(sorted(blocks))
                row = {
                    "dataset": name,
                    "label": cfg["label"],
                    "query_no": qno,
                    "query_id": qid,
                    "candidate_limit": args.candidate_limit,
                    "candidate_count": len(candidates),
                    "search_ms": search_ms,
                    "distinct_heap_pages": distinct_pages,
                    "distance_order_page_runs": distance_runs,
                    "page_sorted_runs": page_sorted_runs,
                    "runs_per_candidate": distance_runs / max(len(candidates), 1),
                    "runs_per_page": distance_runs / max(distinct_pages, 1),
                    "page_sort_run_reduction": distance_runs / max(page_sorted_runs, 1),
                }
                rows.append(row)
                print(
                    f"{name} q={qno} ms={search_ms:.2f} cand={len(candidates)} "
                    f"pages={distinct_pages} runs={distance_runs}",
                    flush=True,
                )

    write_csv(args.out, rows)
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summarize(rows))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
