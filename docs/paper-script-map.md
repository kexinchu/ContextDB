# Paper figures and tables → scripts

Internal names (`figure5`, `table6`, `table10`) do not match the
camera-ready numbering. This map uses the paper's numbers.

Scripts listed here are the keep set. Everything else under
`experiments/hybrid_vector_db/scripts` and most candidate configs are
eligible for deletion.

pytest and SQL smoke tests stay. They do not produce paper numbers, but
they are the reviewer-facing contract.

Motivation runners live under `research/`. Result CSVs, compiled
binaries, and `research/late_bound_visibility/` stay local. Official
ACORN is cloned into `external/ACORN` and is not tracked.

| Paper | What it reports | Keep |
|---|---|---|
| Fig 1 | Amazon-200K HNSWlib / ACORN vs pgvector | `research/fig1_four_curve_frontier.py`, `research/fig1_aligned_1pct.py`, `research/fig1_iso_recall_1pct.py`, `research/hnswlib_vs_pgvector_selectivity.py`, `research/run_hnsw_engine_comparison.sh`, `research/setup_fig1_200k_pg.sh` |
| Fig 2(a) | Candidates per SQL-valid row | `research/controlled_selectivity_multidataset.py`, `sigmod_candidate_waste_summary.py` |
| Fig 2(b) / page-load note | Heap page runs | `research/page_locality_multidataset.py`, `research/page_locality_reordered_multidataset.py`, `research/build_vector_clustered_heap.py`, `sigmod_page_access_summary.py`, `fixed_candidate_page_verify.py`, `pgvector_page_cluster_verify.py`, `pgvector_hnsw_materialize_benchmark.py` |
| Table 1 | C4 atom-reuse rates | `research/filter_reuse_benchmark.py`, `analyze_amazon_c4_trace_cache.py`, `select_amazon_c4_pgvector_queries.py` |
| Table 4 | Dataset footprints | `prepare_amazon10m_sql_native.py`, `prepare_amazon10m_valid_embedding_indexes.py`, `prepare_yfcc_pgvector.py`, `prepare_laion25m_pgvector.py`, `prepare_laion_pgvector.py`, `prepare_pgvector_official_index.py`, `prepare_figure5_external_queries.py`, `build_amazon10m_unique_query_cohort.py` |
| Table 5 | Calibrated held-out latency / QPS | `select_figure5_matched_configs.py`, `run_figure5_matched_latency.py`, `run_figure5_matched_throughput.py`, `pgvector_figure5_throughput.py`, `calibrate_external_table6_configs.py`, `external_dataset_matched_recall_runner.py`, `pgvector_target_recall_selectivity_runner.py`, `yfcc_pgvector_target_recall_runner.py`, `laion_pgvector_target_recall_runner.py` |
| Fig 4 | Recall–cost frontiers | `run_figure5_frontier.py`, `figure5_frontier_artifact.py`, `build_figure5_frontier_workload.py`, `plot_amazon10m_iso_recall.py`, `audit_figure6_iso_recall_coverage.py` |
| Fig 5 | SQL-native shapes + FAISS | `amazon10m_sql_native_benchmark.py`, `figure5_hybrid_allowlist_screen.py`, `rowlocal_faiss14_screen.py`, `faiss_hnsw_sql_attribute_filter_10m.py`, `build_faiss_hnsw_from_fbin.py`, `revision45/run_b2_join_warm.sh`, `revision45/emit_b2_table.py` |
| Table 6 | SQL-first vs VisGuide | `revision45/run_b1_sql_first_q1k.py`, `revision45/run_b1_sql_first_q1k.sh`, `revision45/emit_b1_table.py` |
| Table 7 | Sweeping / aligned ACORN q50 | `revision45/run_q3_acorn_matched.py`, `revision45/run_q3_acorn_aligned.py`, `revision45/run_q3_acorn_matched.sh`, `revision45/run_q3_acorn_aligned.sh`, `revision45/emit_q3_table.py`, `revision45/run_c1_acorn_amazon14.sh`, `revision45/emit_c1_table.py` |
| Table 8 | VisGuide / Locality increments | `pgvector_design1_design2_design3_selectivity_benchmark.py`, `pgvector_d1_stock_increment_control.py`, `pgvector_four_arm_table7_control.py` |
| Table 9 | Locality warm / cold | `pgvector_d2_cache_isolation_control.py` |
| Table 10 | Empty-start FragReuse replay | `amazon10m_d3_adaptation_lifecycle_benchmark.py` |
| Table 11 | Exact / page skip occupancy | `pgvector_page_cluster_verify.py`, `fixed_candidate_page_verify.py` |
| Table 12 | Correctness and overhead | `pgvector_update_correctness_stress.py`, `pgvector_update_concurrency_correctness.py`, `measure_table10_r43_overhead.py`, `build_table10_robustness_summary.py` |
| Table 13 | Read/write concurrency | `pgvector_update_concurrency_benchmark.py`, `revision45/run_c2_failopen_write_sweep.sh`, `revision45/emit_c2_table.py` |

Shared support that those runners import:

- `common_pg.py`
- exact-truth builders: `amazon10m_exact_truth.py`, `amazon10m_sql_native_exact_truth.py`, `figure5_external_exact_truth.py`, `laion25m_exact_truth.py`, `yfcc10m_overlap_workload_truth.py`, `audit_external_exact_truth.py`, `combine_figure5_assigned_truth.py`
- layout: `prepare_pgvector_same_graph_bfs_clone.py`
- launch: `start_pgvector_docker.sh`, `start_amazon_table10_r43.sh`, `clone_laion25m_instance.sh`, `revision45/build_r44.sh`, `revision45/start_r44_amazon.sh`, `revision45/clone_r44_pgdata.py`

Final dataset configs that stay (not band/pilot/candidate drafts):

- `configs/figure5_r41_formal_datasets.json`
- `configs/figure5_r41_amazon_secondary.json`
- `configs/figure5_r41_yfcc_primary.json`
- `configs/figure5_r41_yfcc_primary_v4filters.json`
- `configs/figure5_r41_laion_primary.json`
- `configs/figure5_frontier_datasets.json`
- `configs/figure6_iso_recall_targets.json`
- `configs/p0_release_contract.json`
- `configs/p0_release_contract_r41.json`
- `configs/p0_release_contract_r43.json`
- `configs/p0_release_contract_r43_laion.json`
- `configs/amazon10m_selectivity14_filters.csv`
- `configs/amazon10m_selectivity14_valid_embeddings_filters.csv`
- `configs/target090_q5k_filter_seeds.json`
- `configs/target095_q5k_filter_seeds.json`
- `configs/target099_q5k_filter_seeds.json`
- `configs/amazon_target095_q5k_final.json`
- `configs/amazon_target099_q5k_final.json`
- `configs/table6_r41_laion_target090_per_filter_{modes,ef,targets}.json`
- `configs/table6_r41_laion_target095_per_filter_{modes,ef}.json`
- `configs/table6_r41_laion_target099_per_filter_{modes,ef,targets}.json`
- `configs/table6_r41_{yfcc,laion}_*_paired.json` finals only
- `configs/yfcc10m_r31_table6_calibration_seeds.csv`
- `configs/laion25m_r32_table6_shared_calibration_seeds.csv`
- `configs/pgvector_v082_ef10000_formal_ladder.csv`
- `configs/pgvector_v082_ef100000_formal_ladder.csv`
