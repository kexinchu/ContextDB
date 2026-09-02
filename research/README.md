# Motivation experiments

Source for paper Figure 1, Figure 2, and Table 1. Result CSVs, compiled
binaries, and `late_bound_visibility/` stay local and are gitignored.

`PGPASSWORD` is required. Do not commit a password.

## Figure 1 (Amazon-200K engine gap)

HNSWlib sweeping / ACORN versus stock pgvector on a 200K Amazon subset.

```bash
export PGPASSWORD
# optional: clone official ACORN if you rebuild the FAISS-ACORN arm
git clone https://github.com/stanford-futuredata/ACORN.git external/ACORN
bash research/setup_fig1_200k_pg.sh
bash research/run_hnsw_engine_comparison.sh
```

| Script | Role |
|---|---|
| `setup_fig1_200k_pg.sh` | Dedicated PostgreSQL on port 55438 |
| `fig1_four_curve_frontier.py` | Mixed-selectivity four-curve frontier |
| `fig1_aligned_1pct.py` / `fig1_iso_recall_1pct.py` | 1% aligned / iso-recall |
| `hnswlib_vs_pgvector_selectivity.py` | HNSWlib vs pgvector selectivity sweep |
| `acorn_faiss_mixed_frontier.cpp` / `acorn_faiss_selectivity.cpp` | Official ACORN (FAISS) arm |
| `acorn_hnswlib_*.cpp` / `hnswlib_acorn_*.hpp` | HNSWlib ACORN diagnostic |

The FAISS-ACORN binary is built against a local clone at `external/ACORN`.
That checkout is not in this repository. HNSWlib headers live in
`third_party/hnswlib/`.

## Figure 2 (candidate waste and heap locality)

Amazon-10M, MS MARCO-1M, and Enron-50K.

| Script | Paper |
|---|---|
| `controlled_selectivity_multidataset.py` | Fig 2(a) candidates per SQL-valid row |
| `page_locality_multidataset.py` | Fig 2(b) distance-order page runs |
| `page_locality_reordered_multidataset.py` | Fig 2(b) vector-local lower bound |
| `build_vector_clustered_heap.py` | Builds the reordered heap used above |

## Table 1 (C4 atom reuse)

`filter_reuse_benchmark.py` summarizes reuse on a local C4 trace CSV. The
trace itself is not in the repository.
