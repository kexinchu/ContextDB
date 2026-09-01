#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON:-python3}"

pause_file="${SQLENS_EXPERIMENT_PAUSE_FILE:-results/hybrid_vector_db/.pause_laion_target099_pilot}"
if [[ -e "${pause_file}" ]]; then
  echo "paused by ${pause_file}"
  exit 0
fi

out_dir="results/hybrid_vector_db/table6_r41_laion_target099_per_filter_q5k"
mkdir -p "${out_dir}"
mode_config="${MODE_CONFIG:-experiments/hybrid_vector_db/configs/table6_r41_laion_target099_pilot_modes.json}"
guidance_min_pct="${GUIDANCE_MIN_PCT:-0.6}"
output_suffix="${OUTPUT_SUFFIX:-}"
queries="${QUERIES:-20}"
filter_ef_json="${FILTER_EF_JSON:-}"
filter_target_json="${FILTER_TARGET_JSON:-}"
prewarm_each_filter="${PREWARM_EACH_FILTER:-1}"
modes_text="${MODES:-original design1_bloom_bfs_layout_d3}"
read -r -a modes <<< "${modes_text}"

extra_config_args=()
if [[ -n "${filter_ef_json}" ]]; then
  extra_config_args+=(--filter-ef-search-json "${filter_ef_json}")
fi
if [[ -n "${filter_target_json}" ]]; then
  extra_config_args+=(--filter-traversal-target-json "${filter_target_json}")
fi

prewarm_args=()
if [[ "${prewarm_each_filter}" == "1" ]]; then
  prewarm_args+=(
    --prewarm-relation public.laion25m_pgvector
    --prewarm-relation public.laion25m_pgvector_embedding_hnsw
    --prewarm-relation public.laion25m_pgvector_embedding_hnsw_bfs_r32
  )
fi

default_filters=(
  labelor_top70 labelor_top55 labelor_top40 labelor_top30 labelor_top20
  labelor_top14 labelor_top9 labelor_top6 labelor_top3 label_175 label_79
  label_2039 label_1432 label_281
)
if (($#)); then
  filters=("$@")
else
  filters=("${default_filters[@]}")
fi

export PGHOST=127.0.0.1
export PGPORT=55432
export PGDATABASE=hybrid_vector
export PGUSER=postgres
: "${PGPASSWORD:?set PGPASSWORD}"
export PYTHONUNBUFFERED=1

for filter_name in "${filters[@]}"; do
  out_csv="${out_dir}/laion_target099_pilot_${filter_name}${output_suffix}_q${queries}.csv"
  if [[ -s "${out_csv}" && -s "${out_csv}.plan.json" ]]; then
    echo "skip complete ${filter_name}"
    continue
  fi
  echo "run ${filter_name}"

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
  --filters-csv results/hybrid_vector_db/laion25m_matched_recall_filters_q180_orfreq.csv \
  --filter-names "${filter_name}" \
  --modes "${modes[@]}" \
  --execution-order interleaved \
  --schedule-seed 20260806 \
  --mode-configs-json "${mode_config}" \
  "${extra_config_args[@]}" \
  "${prewarm_args[@]}" \
  --queries "${queries}" \
  --query-offset 0 \
  --repeats 1 \
  --no-isolate-repeat-runtimes \
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
  --d3-fragment-store-namespace "table6-r41-laion-t099-pilot-${filter_name}${output_suffix}" \
  --d3-probe-requests 2 \
  --d3-min-benefit-per-byte 0 \
  --d3-max-fragment-mb 16 \
  --d3-page-min-skip-rate 0.05 \
  --guidance-selectivity-min-pct "${guidance_min_pct}" \
  --guidance-selectivity-max-pct 6 \
  --guidance-composite-max-selectivity-pct 6 \
  --guidance-max-atoms 8 \
  --d2-page-access off \
  --d2-index-page-access off \
  --statement-timeout-ms 300000 \
  --force-hnsw \
  --progress-queries 20 \
  --d2-graph-proof-json results/hybrid_vector_db/laion25m_r32_table6_shared_d2_warm_q100r5_20260723.csv.d2_graph_proof.json \
  --expected-sqlens-build-id sqlens-v16-distance-aware-route-budget-ef500k-20260801-r41 \
  --expected-vector-so-sha256 8f53226d35cae28d4e1b6926b13b01fa01fd1f6720c5f57c96c7886905f5eaf0 \
  --backend-cpu-list 48-63 \
  --out "${out_csv}"
done
