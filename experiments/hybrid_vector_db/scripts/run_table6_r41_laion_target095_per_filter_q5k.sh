#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON:-python3}"

out_dir="results/hybrid_vector_db/table6_r41_laion_target095_per_filter_q5k"
out_csv="${out_dir}/laion_target095_per_filter_paired_q5k.csv"
mkdir -p "${out_dir}"

export PGHOST=127.0.0.1
export PGPORT=55432
export PGDATABASE=hybrid_vector
export PGUSER=postgres
: "${PGPASSWORD:?set PGPASSWORD}"
export PYTHONUNBUFFERED=1

"${python_bin}" \
  experiments/hybrid_vector_db/scripts/pgvector_design1_design2_design3_selectivity_benchmark.py \
  --insertion-table public.laion25m_pgvector \
  --insertion-index public.laion25m_pgvector_embedding_hnsw \
  --bfs-table public.laion25m_pgvector \
  --bfs-index public.laion25m_pgvector_embedding_hnsw_bfs_r32 \
  --query-table public.laion25m_queries \
  --query-id-column qid \
  --query-vector-column embedding \
  --candidate-validity-predicate TRUE \
  --no-expected-truth-self-excluded \
  --truth-csv results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_laion_exact_truth_q12800.csv \
  --workload-csv results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r41_laion_measurement_q5000.csv \
  --expected-workload-requests 5000 \
  --require-unique-workload-queries \
  --filters-csv results/hybrid_vector_db/laion25m_matched_recall_filters_q180_orfreq.csv \
  --modes original design1_bloom_bfs_layout_d3 \
  --execution-order interleaved \
  --schedule-seed 20260804 \
  --mode-configs-json experiments/hybrid_vector_db/configs/table6_r41_laion_target095_per_filter_modes.json \
  --filter-ef-search-json experiments/hybrid_vector_db/configs/table6_r41_laion_target095_per_filter_ef.json \
  --prewarm-relation public.laion25m_pgvector \
  --prewarm-relation public.laion25m_pgvector_embedding_hnsw \
  --prewarm-relation public.laion25m_pgvector_embedding_hnsw_bfs_r32 \
  --repeats 1 \
  --isolate-repeat-runtimes \
  --warmup-queries 0 \
  --no-warmup-all-queries \
  --k 10 \
  --guidance-filter-strategy traversal_guided \
  --guidance-bypass-iterative-scan strict_order \
  --guidance-bypass-ef-search 0 \
  --guidance-low-selectivity-bypass-ef-search 0 \
  --d2-source-on-guidance-bypass \
  --d1-exact-max-selectivity-pct 6 \
  --collapse-exact-and-guidance \
  --d3-cache-mb 1024 \
  --d3-measurement-policy workload_driven_adaptive \
  --d3-fragment-store-namespace table6-r41-laion-t095-per-filter-q5k \
  --d3-probe-requests 2 \
  --d3-min-benefit-per-byte 0 \
  --d3-max-fragment-mb 16 \
  --d3-page-min-skip-rate 0.05 \
  --guidance-selectivity-min-pct 0.6 \
  --guidance-selectivity-max-pct 6 \
  --guidance-composite-max-selectivity-pct 6 \
  --guidance-max-atoms 8 \
  --d2-page-access off \
  --d2-index-page-access off \
  --statement-timeout-ms 300000 \
  --force-hnsw \
  --progress-queries 250 \
  --d2-graph-proof-json results/hybrid_vector_db/laion25m_r32_table6_shared_d2_warm_q100r5_20260723.csv.d2_graph_proof.json \
  --expected-sqlens-build-id sqlens-v16-distance-aware-route-budget-ef500k-20260801-r41 \
  --expected-vector-so-sha256 8f53226d35cae28d4e1b6926b13b01fa01fd1f6720c5f57c96c7886905f5eaf0 \
  --backend-cpu-list 48-63 \
  --out "${out_csv}"
