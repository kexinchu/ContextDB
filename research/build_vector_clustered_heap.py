from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import psycopg

from pg_conn import pg_conninfo


DATASETS = {
    "msmarco": {
        "source": "msmarco_kill_passages",
        "target": "msmarco_kill_passages_vector_clustered",
        "id": "pid",
    },
    "enron": {
        "source": "enron_messages",
        "target": "enron_messages_vector_clustered",
        "id": "message_id",
    },
}


def timed(label: str, fn):
    start = time.perf_counter()
    value = fn()
    print(f"{label} elapsed_s={time.perf_counter() - start:.2f}", flush=True)
    return value


def parse_vector(text: str) -> np.ndarray:
    return np.fromstring(text.strip("[]"), sep=",", dtype=np.float32)


def load_vectors(conninfo: str, table: str, id_col: str) -> tuple[np.ndarray, np.ndarray]:
    ids: list[int] = []
    vectors: list[np.ndarray] = []
    with psycopg.connect(conninfo) as conn:
        with conn.cursor(name=f"{table}_vector_export") as cur:
            cur.itersize = 5000
            cur.execute(f"SELECT {id_col}, embedding::text FROM {table} ORDER BY {id_col}")
            for row_id, emb in cur:
                ids.append(int(row_id))
                vectors.append(parse_vector(str(emb)))
                if len(ids) % 100000 == 0:
                    print(f"loaded rows={len(ids)}", flush=True)
    return np.asarray(ids, dtype=np.int64), np.vstack(vectors).astype(np.float32)


def cluster_order(vectors: np.ndarray, clusters: int, train_sample: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import faiss

    rng = np.random.default_rng(seed)
    sample_size = min(train_sample, len(vectors))
    sample = np.ascontiguousarray(vectors[rng.choice(len(vectors), size=sample_size, replace=False)])
    kmeans = faiss.Kmeans(vectors.shape[1], clusters, niter=20, nredo=1, seed=seed, verbose=True, gpu=False)
    timed(f"train kmeans sample={sample_size} clusters={clusters}", lambda: kmeans.train(sample))

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(kmeans.centroids)
    cluster_ids = np.empty(len(vectors), dtype=np.int32)
    distances = np.empty(len(vectors), dtype=np.float32)
    chunk = 100000
    for start in range(0, len(vectors), chunk):
        stop = min(start + chunk, len(vectors))
        dists, cids = index.search(np.ascontiguousarray(vectors[start:stop]), 1)
        cluster_ids[start:stop] = cids[:, 0]
        distances[start:stop] = dists[:, 0]
        print(f"assigned rows={stop}/{len(vectors)}", flush=True)
    order = np.lexsort((distances, cluster_ids))
    return order, cluster_ids


def create_clustered_table(conninfo: str, source: str, target: str, id_col: str, ids: np.ndarray, order: np.ndarray, clusters: np.ndarray) -> None:
    with psycopg.connect(conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '2GB'")
            cur.execute("SET work_mem = '2GB'")
            cur.execute(f"DROP TABLE IF EXISTS {target}")
            cur.execute("DROP TABLE IF EXISTS vector_heap_order")
            cur.execute(
                """
                CREATE TEMP TABLE vector_heap_order (
                    ord integer NOT NULL,
                    row_id bigint NOT NULL,
                    vector_cluster_id integer NOT NULL
                ) ON COMMIT PRESERVE ROWS
                """
            )
            with cur.copy("COPY vector_heap_order (ord, row_id, vector_cluster_id) FROM STDIN") as copy:
                for ord_no, pos in enumerate(order):
                    copy.write(f"{ord_no}\t{int(ids[pos])}\t{int(clusters[pos])}\n".encode("utf-8"))
            cur.execute("ANALYZE vector_heap_order")
            cur.execute(
                f"""
                CREATE TABLE {target} AS
                SELECT s.*, o.vector_cluster_id
                FROM vector_heap_order o
                JOIN {source} s ON s.{id_col} = o.row_id
                ORDER BY o.ord
                """
            )
            cur.execute(f"ALTER TABLE {target} ADD PRIMARY KEY ({id_col})")
            cur.execute(f"ANALYZE {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conninfo", default=None)
    parser.add_argument("--datasets", default="msmarco,enron")
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--train-sample", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--manifest", type=Path, default=Path("research/results/vector_clustered_heap_manifest.csv"))
    args = parser.parse_args()
    if not args.conninfo:
        args.conninfo = pg_conninfo("55432")

    rows = []
    for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        cfg = DATASETS[name]
        ids, vectors = timed(name + " load vectors", lambda: load_vectors(args.conninfo, cfg["source"], cfg["id"]))
        order, clusters = timed(
            name + " cluster order",
            lambda: cluster_order(vectors, min(args.clusters, max(2, len(vectors) // 100)), args.train_sample, args.seed),
        )
        timed(
            name + " create clustered heap",
            lambda: create_clustered_table(args.conninfo, cfg["source"], cfg["target"], cfg["id"], ids, order, clusters),
        )
        rows.append({"dataset": name, **cfg, "rows": len(ids), "clusters": int(clusters.max()) + 1})

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
