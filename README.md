# SQLens

Prototype code for SQL-native filtered vector search in PostgreSQL with
pgvector. SQLens adds conservative predicate guidance at the HNSW candidate
boundary, reusable heap-local fragments, and same-graph BFS packing. PostgreSQL
still performs MVCC, residual SQL, and final validation.

This repository is the source snapshot referenced by the paper. It does not
include raw datasets, generated HNSW indexes, PostgreSQL data directories, or
the numeric artifacts behind the paper tables. Those must be rebuilt locally.

The evaluated paper binary is the r44 fork.

## Layout

| Path | Contents |
|---|---|
| `third_party/pgvector-sqlens-r44/` | **Default / paper binary.** pgvector 0.8.2 with SQLens (r44) |
| `third_party/pgvector-sqlens/` | Earlier SQLens tree kept for comparison |
| `third_party/hnswlib/` | In-memory HNSW / ACORN diagnostic sources |
| `patches/` | Audit patches against upstream pgvector |
| `experiments/hybrid_vector_db/` | Benchmark scripts, SQL smoke tests, and pytest suite |
| `docs/` | Artifact notes |

## Dependencies

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

PostgreSQL 16.14 is the evaluated server. Connection settings use the usual
environment variables (the local Docker default password is `postgres`):

```bash
export PGHOST=127.0.0.1
export PGPORT=55432
export PGDATABASE=hybrid_vector
export PGUSER=postgres
export PGPASSWORD=postgres
```

Dataset roots, if you rebuild the paper workloads:

```bash
export OOD_ANNS_DATA=/path/to/ood-anns/data
export LAION25M_DATA_DIR=/path/to/LAION25M
export YFCC10M_DATA_DIR=/path/to/YFCC10M
```

## Build the paper binary (r44)

```bash
cd third_party/pgvector-sqlens-r44
make
make install
```

Restart PostgreSQL after replacing `vector.so`, then check the build id:

```sql
SELECT vector_sqlens_build_id();
```

The audit patch in `patches/pgvector-sqlens.patch` is against upstream
pgvector commit `cab9da72c04353f143bb06b42ab70a403daac64a`.

## Tests that run without the paper datasets

```bash
.venv/bin/python -m pytest -q experiments/hybrid_vector_db/tests
```

SQL smoke tests under `experiments/hybrid_vector_db/sql/` cover candidate
admission, MVCC/epoch invalidation, HOT updates, same-graph layout, and
fragment reuse.

## What this snapshot is not

It is not a one-command reproduction of the paper's 10M/25M tables. Amazon
Reviews, YFCC, and LAION embeddings, index builds, and overnight benchmark
scripts are environment-specific. Use the pytest suite and SQL smoke tests to
inspect safety and the access-method contract.
