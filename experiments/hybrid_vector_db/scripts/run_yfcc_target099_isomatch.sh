#!/usr/bin/env bash
# YFCC-10M Table-6 target=0.99 iso-recall retest:
# raise SQLens until aggregate Recall@10 matches frozen Stock, then measure QPS.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON:-python3}"
out_dir="${OUT_DIR:-results/hybrid_vector_db/table6_r41_yfcc_target099_isomatch}"
bypass_ef="${BYPASS_EF:-10000}"
run_tag="${RUN_TAG:-bypass${bypass_ef}_ef1000_t200}"
log_file="${out_dir}/pipeline_${run_tag}.log"
mkdir -p "${out_dir}"

export PGHOST=127.0.0.1
export PGPORT=55432
export PGDATABASE=hybrid_vector
export PGUSER=postgres
: "${PGPASSWORD:?set PGPASSWORD}"
export PYTHONUNBUFFERED=1

stock_source="results/hybrid_vector_db/table6_r41_yfcc_target099/yfcc_target099_paired_q10k.csv"
sqlens_csv="${out_dir}/yfcc_target099_sqlens_${run_tag}_q10k.csv"
paired_csv="${out_dir}/yfcc_target099_isomatch_${run_tag}_q10k.csv"
summary_json="${out_dir}/yfcc_target099_isomatch_${run_tag}_summary.json"
qps_prefix="${out_dir}/yfcc_target099_isomatch_${run_tag}_c16_qps"

mode_configs="experiments/hybrid_vector_db/configs/table6_r41_yfcc_target099_sqlens_isomatch.json"
d2_proof="results/hybrid_vector_db/yfcc10m_r31_table6_shared_d2_warm_q100r5_20260723.csv.d2_graph_proof.json"
truth="results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_yfcc_exact_truth_q12800.csv"
workload="results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_yfcc_measurement.csv"
filters="results/hybrid_vector_db/yfcc10m_matched_recall_filters_q180.csv"
build_id="sqlens-v16-distance-aware-route-budget-ef500k-20260801-r41"
vector_sha="8f53226d35cae28d4e1b6926b13b01fa01fd1f6720c5f57c96c7886905f5eaf0"

exec > >(tee -a "${log_file}") 2>&1

echo "[isomatch] start $(date -Is) bypass_ef=${bypass_ef} tag=${run_tag}"

if [[ ! -f "${sqlens_csv}" ]]; then
  echo "[isomatch] measuring SQLens q10k ..."
  "${python_bin}" \
    experiments/hybrid_vector_db/scripts/pgvector_design1_design2_design3_selectivity_benchmark.py \
    --insertion-table public.yfcc10m_pgvector \
    --insertion-index public.yfcc10m_pgvector_embedding_hnsw \
    --bfs-table public.yfcc10m_pgvector \
    --bfs-index public.yfcc10m_pgvector_embedding_hnsw_bfs_r31 \
    --query-table public.yfcc10m_queries \
    --query-id-column qid \
    --query-vector-column embedding \
    --candidate-validity-predicate TRUE \
    --no-expected-truth-self-excluded \
    --truth-csv "${truth}" \
    --workload-csv "${workload}" \
    --expected-workload-requests 10000 \
    --require-unique-workload-queries \
    --filters-csv "${filters}" \
    --modes design1_bloom_bfs_layout_d3 \
    --execution-order interleaved \
    --schedule-seed 20260801 \
    --mode-configs-json "${mode_configs}" \
    --prewarm-relation public.yfcc10m_pgvector \
    --prewarm-relation public.yfcc10m_pgvector_embedding_hnsw \
    --prewarm-relation public.yfcc10m_pgvector_embedding_hnsw_bfs_r31 \
    --repeats 1 \
    --isolate-repeat-runtimes \
    --warmup-queries 1 \
    --no-warmup-all-queries \
    --k 10 \
    --guidance-filter-strategy traversal_guided \
    --guidance-bypass-iterative-scan strict_order \
    --guidance-bypass-ef-search "${bypass_ef}" \
    --guidance-low-selectivity-bypass-ef-search 50000 \
    --d2-source-on-guidance-bypass \
    --d1-exact-max-selectivity-pct 6 \
    --collapse-exact-and-guidance \
    --d3-cache-mb 1024 \
    --d3-measurement-policy workload_driven_adaptive \
    --d3-fragment-store-namespace "table6-r41-yfcc-t099-isomatch-${run_tag}" \
    --d3-probe-requests 2 \
    --d3-min-benefit-per-byte 0 \
    --d3-max-fragment-mb 16 \
    --d3-page-min-skip-rate 0.05 \
    --guidance-selectivity-min-pct 0.5 \
    --guidance-selectivity-max-pct 6 \
    --guidance-composite-max-selectivity-pct 6 \
    --guidance-max-atoms 8 \
    --d2-page-access off \
    --d2-index-page-access off \
    --statement-timeout-ms 7200000 \
    --force-hnsw \
    --require-preferred-index-guc \
    --progress-queries 250 \
    --d2-graph-proof-json "${d2_proof}" \
    --expected-sqlens-build-id "${build_id}" \
    --expected-vector-so-sha256 "${vector_sha}" \
    --backend-cpu-list 48-63 \
    --out "${sqlens_csv}"
else
  echo "[isomatch] reuse existing ${sqlens_csv}"
fi

echo "[isomatch] stitch + iso-recall gate ..."
set +e
"${python_bin}" experiments/hybrid_vector_db/scripts/stitch_yfcc_target099_isomatch.py \
  --stock-source "${stock_source}" \
  --sqlens-source "${sqlens_csv}" \
  --out-csv "${paired_csv}" \
  --out-summary "${summary_json}" \
  --max-recall-gap 0.00015
stitch_rc=$?
set -e

if [[ "${stitch_rc}" -ne 0 ]]; then
  echo "[isomatch] GATE FAIL (rc=${stitch_rc}). Not launching QPS."
  echo "[isomatch] Re-run with BYPASS_EF=20000 or stronger guided ef."
  exit "${stitch_rc}"
fi

echo "[isomatch] iso-recall gate passed; launching c16 QPS ..."
container_pid="$(docker inspect --format '{{.State.Pid}}' hybrid-pgvector)"

# Keep Stock at ef=50000; SQLens uses the same elevated matched config.
"${python_bin}" \
  experiments/hybrid_vector_db/scripts/pgvector_figure5_throughput.py \
  --frontier-config experiments/hybrid_vector_db/configs/figure5_r41_formal_datasets.json \
  --dataset yfcc \
  --workload-manifest results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_yfcc_manifest.json \
  --workload-request-limit 10000 \
  --config-id "yfcc-r41-target099-isomatch-${run_tag}" \
  --pair-id "yfcc-r41-target099-isomatch-${run_tag}" \
  --target-recall 0.99 \
  --stock-ef-search 50000 \
  --sqlens-ef-search 1000 \
  --stock-iterative-scan strict_order \
  --sqlens-iterative-scan off \
  --stock-max-scan-tuples 5000000 \
  --sqlens-max-scan-tuples 5000000 \
  --stock-scan-mem-multiplier 32 \
  --sqlens-scan-mem-multiplier 32 \
  --stock-guided-collect-target 50000 \
  --sqlens-guided-collect-target 1000 \
  --stock-traversal-guided-target 10 \
  --sqlens-traversal-guided-target 200 \
  --stock-traversal-guided-burst 1 \
  --sqlens-traversal-guided-burst 8 \
  --no-stock-traversal-guided-early-stop \
  --no-sqlens-traversal-guided-early-stop \
  --stock-traversal-guided-early-stop-distance-ratio 0 \
  --sqlens-traversal-guided-early-stop-distance-ratio 0 \
  --d1-exact-max-selectivity-pct 6 \
  --collapse-exact-and-guidance \
  --guidance-selectivity-min-pct 0.5 \
  --guidance-selectivity-max-pct 6 \
  --guidance-composite-max-selectivity-pct 6 \
  --guidance-max-atoms 8 \
  --d2-source-on-guidance-bypass \
  --guidance-bypass-iterative-scan strict_order \
  --guidance-bypass-ef-search "${bypass_ef}" \
  --guidance-low-selectivity-bypass-ef-search 50000 \
  --d2-page-access off \
  --d2-index-page-access off \
  --d3-cache-mb 1024 \
  --d3-probe-requests 2 \
  --d3-min-benefit-per-byte 0 \
  --d3-max-fragment-mb 16 \
  --d3-page-min-skip-rate 0.05 \
  --clients 16 \
  --repeats 1 \
  --allow-single-pass \
  --schedule-seed 20260801 \
  --backend-cpu-list 48-63 \
  --client-cpu-list 0-31 \
  --backend-proc-root "/proc/${container_pid}/root/proc" \
  --pg-prewarm \
  --statement-timeout-ms 7200000 \
  --bootstrap-samples 2000 \
  --bootstrap-seed 20260801 \
  --run-id "yfcc-t099-isomatch-${run_tag}-c16-r1" \
  --out-prefix "${qps_prefix}" \
  --overwrite \
  --execute

echo "[isomatch] done $(date -Is)"
echo "[isomatch] summary=${summary_json}"
echo "[isomatch] qps_prefix=${qps_prefix}"
