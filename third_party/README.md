# Vendored sources

| Tree | Role |
|---|---|
| `pgvector-sqlens-r44/` | Default SQLens build. Paper measurements use this tree (r44). |
| `pgvector-sqlens/` | Earlier SQLens snapshot. Keep it for script and test paths that still name this directory. |
| `hnswlib/` | In-memory HNSW / ACORN sources used by diagnostic comparisons. |

Build and install r44 unless a script explicitly points at the earlier tree.
