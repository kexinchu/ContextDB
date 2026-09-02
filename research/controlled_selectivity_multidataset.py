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
from psycopg import errors

from pg_conn import pg_conninfo


DEFAULT_SELECTIVITIES = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

DATASETS = {
    "amazon": {
        "label": "Amazon-10M",
        "table": "amazon_grocery_reviews_10m_pgvector",
        "id": "id",
    },
    "msmarco": {
        "label": "MS MARCO-1M",
        "table": "msmarco_kill_passages",
        "id": "pid",
    },
    "enron": {
        "label": "Enron-50K",
        "table": "enron_messages",
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


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def filter_sql(id_col: str, selectivity: int) -> str:
    if selectivity >= 100:
        return "TRUE"
    return f"({id_col} %% 100) < {int(selectivity)}"


def load_or_create_query_ids(cur, dataset: str, table: str, id_col: str, path: Path, queries: int) -> list[int]:
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


def parse_profile(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        import json

        return json.loads(str(value))
    except Exception:
        return {}


def fetch_profile(cur) -> dict[str, Any]:
    try:
        cur.execute("SELECT vector_hnsw_last_scan_profile()")
        return parse_profile(cur.fetchone()[0])
    except Exception:
        return {}


def reset_profile(cur) -> None:
    try:
        cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    except Exception:
        pass


def run_query(cur, table: str, id_col: str, qvec: str, selectivity: int, k: int, timeout_ms: int):
    reset_profile(cur)
    pred = filter_sql(id_col, selectivity)

    def run():
        cur.execute(
            f"""
            SELECT {id_col}
            FROM {table}
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
    except errors.QueryCanceled as exc:
        ids, elapsed, error = [], float(timeout_ms), exc.__class__.__name__
        cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
    return ids, elapsed, fetch_profile(cur), error


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    return vals[max(0, int(0.95 * len(vals)) - 1)]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), int(row["selectivity_pct"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (dataset, sel), items in sorted(groups.items()):
        ok = [r for r in items if not r.get("error")]
        if not ok:
            continue

        def vals(key: str) -> list[float]:
            return [float(r.get(key, 0) or 0) for r in ok]

        waste = [float(r["returned_tuples"]) / max(float(r["final_rows"]), 1.0) for r in ok]
        out.append(
            {
                "dataset": dataset,
                "label": ok[0]["label"],
                "selectivity_pct": sel,
                "queries": len(ok),
                "latency_ms_mean": statistics.mean(vals("latency_ms")),
                "latency_ms_p50": statistics.median(vals("latency_ms")),
                "latency_ms_p95": p95(vals("latency_ms")),
                "returned_tuples_mean": statistics.mean(vals("returned_tuples")),
                "final_rows_mean": statistics.mean(vals("final_rows")),
                "candidate_per_valid_mean": statistics.mean(waste),
                "executor_reject_ratio_mean": statistics.mean(vals("executor_reject_ratio")),
                "visited_tuples_mean": statistics.mean(vals("visited_tuples")),
                "index_element_pages_mean": statistics.mean(vals("index_page_element_distinct_pages")),
                "index_element_runs_mean": statistics.mean(vals("index_page_element_runs")),
                "index_neighbor_pages_mean": statistics.mean(vals("index_page_neighbor_distinct_pages")),
                "index_neighbor_runs_mean": statistics.mean(vals("index_page_neighbor_runs")),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conninfo", default=None)
    parser.add_argument("--datasets", default="amazon,msmarco,enron")
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--selectivities", default=",".join(str(x) for x in DEFAULT_SELECTIVITIES))
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=500000)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--out", type=Path, default=Path("research/results/controlled_selectivity_multidataset_q30.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.conninfo:
        args.conninfo = pg_conninfo("55432")

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    selectivities = [int(x) for x in args.selectivities.split(",") if x.strip()]
    rows: list[dict[str, Any]] = list(read_existing_rows(args.out)) if args.resume else []
    done = {(r["dataset"], int(r["selectivity_pct"]), int(r["query_no"])) for r in rows}
    if args.out.exists() and not args.resume:
        args.out.unlink()

    with psycopg.connect(args.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("SET client_min_messages = warning")
        cur.execute("SET jit = off")
        cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
        cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
        cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
        cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
        cur.execute("SET hnsw.scan_mem_multiplier = 8")
        cur.execute("SET enable_sort = off")
        cur.execute("SET enable_bitmapscan = off")
        for dataset in datasets:
            cfg = DATASETS[dataset]
            qpath = Path(f"research/results/{dataset}_controlled_query_ids_{args.queries}.csv")
            query_ids = load_or_create_query_ids(cur, dataset, cfg["table"], cfg["id"], qpath, args.queries)
            cur.execute(
                f"SELECT {cfg['id']}, embedding::text FROM {cfg['table']} WHERE {cfg['id']} = ANY(%s::bigint[])",
                (query_ids,),
            )
            qvecs = {int(row[0]): str(row[1]) for row in cur.fetchall()}
            for sel in selectivities:
                for qno, qid in enumerate(query_ids):
                    key = (dataset, sel, qno)
                    if key in done:
                        continue
                    ids, elapsed, profile, error = run_query(
                        cur,
                        cfg["table"],
                        cfg["id"],
                        qvecs[qid],
                        sel,
                        args.k,
                        args.statement_timeout_ms,
                    )
                    returned_tuples = float(profile.get("returned_tuples", 0) or 0)
                    final_rows = len(ids)
                    reject = (returned_tuples - final_rows) / returned_tuples if returned_tuples else 0.0
                    row = {
                        "dataset": dataset,
                        "label": cfg["label"],
                        "table": cfg["table"],
                        "query_no": qno,
                        "query_id": qid,
                        "selectivity_pct": sel,
                        "k": args.k,
                        "latency_ms": elapsed,
                        "error": error,
                        "final_rows": final_rows,
                        "returned_tuples": returned_tuples,
                        "executor_reject_ratio": max(0.0, reject),
                        "visited_tuples": profile.get("visited_tuples", 0),
                        "index_page_element_distinct_pages": profile.get("index_page_element_distinct_pages", 0),
                        "index_page_element_runs": profile.get("index_page_element_runs", 0),
                        "index_page_neighbor_distinct_pages": profile.get("index_page_neighbor_distinct_pages", 0),
                        "index_page_neighbor_runs": profile.get("index_page_neighbor_runs", 0),
                    }
                    rows.append(row)
                    append_csv_row(args.out, row)
                    print(
                        f"{dataset} sel={sel} q={qno} ms={elapsed:.1f} "
                        f"final={final_rows} returned_tuples={returned_tuples:.0f} err={error}",
                        flush=True,
                    )
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summarize(rows))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
