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
| `experiments/hybrid_vector_db/` | Paper-facing benchmark scripts, SQL smoke tests, and pytest suite |
| `docs/` | Artifact notes and [paper table → script map](docs/paper-script-map.md) |

## Dependencies

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

PostgreSQL 16.14 is the evaluated server. Connection settings use the usual
environment variables. `PGPASSWORD` is required and must not be committed:

```bash
export PGHOST=127.0.0.1
export PGPORT=55432
export PGDATABASE=hybrid_vector
export PGUSER=postgres
export PGPASSWORD
```

Dataset roots, if you rebuild the paper workloads:

```bash
export OOD_ANNS_DATA=/path/to/ood-anns/data
export LAION25M_DATA_DIR=/path/to/LAION25M
export YFCC10M_DATA_DIR=/path/to/YFCC10M
export TABLE10_PGDATA=/path/to/pgdata
export R44_PGDATA=/path/to/pgdata-r44
export WORKDIR=/path/to/scratch
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

## Paper tables and figures

Internal script names (`figure5`, `table6`, `table10`) do not match the
camera-ready numbering. The keep set is listed in
`docs/paper-script-map.md`.

| Paper | Scripts |
|---|---|
| Table 5 / Fig 4 | `run_figure5_matched_latency.py`, `run_figure5_matched_throughput.py`, `run_figure5_frontier.py` |
| Fig 5 | `amazon10m_sql_native_benchmark.py`, `revision45/run_b2_join_warm.sh` |
| Table 6 | `revision45/run_b1_sql_first_q1k.py` |
| Table 7 | `revision45/run_q3_acorn_matched.py`, `revision45/run_q3_acorn_aligned.py` |
| Tables 8–11 | `pgvector_design1_design2_design3_selectivity_benchmark.py`, `pgvector_d2_cache_isolation_control.py`, `amazon10m_d3_adaptation_lifecycle_benchmark.py` |
| Tables 12–13 | `pgvector_update_correctness_stress.py`, `pgvector_update_concurrency_benchmark.py`, `revision45/run_c2_failopen_write_sweep.sh` |

Figure 1 (Amazon-200K HNSWlib vs pgvector) is not runnable from this
snapshot. `third_party/hnswlib/` remains for the diagnostic sources.

## Tests that run without the paper datasets

```bash
export PGPASSWORD   # required even for unit tests that only build conninfo
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
