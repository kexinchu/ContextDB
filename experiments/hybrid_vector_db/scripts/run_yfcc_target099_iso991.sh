#!/usr/bin/env bash
# YFCC-10M Table-6 target≈0.99 iso-recall retest at ~0.991 for BOTH arms.
# Stock: strict_order ef=1500 (calib R≈0.9906)
# SQLens: ef=500 + early-stop, tuned so aggregate recall matches Stock.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON:-/home/kec23008/miniconda3/bin/python}"
out_dir="${OUT_DIR:-results/hybrid_vector_db/table6_r41_yfcc_target099_iso991}"
run_tag="${RUN_TAG:-stock1500_sqlens_ef750_t40_r099}"
sqlens_ef="${SQLENS_EF:-750}"
sqlens_t="${SQLENS_T:-40}"
sqlens_ratio="${SQLENS_RATIO:-0.99}"
bypass_ef="${BYPASS_EF:-${sqlens_ef}}"
low_bypass_ef="${LOW_BYPASS_EF:-1500}"
phase="${PHASE:-all}"  # pilot|paired|qps|all
mkdir -p "${out_dir}"
log_file="${out_dir}/pipeline_${run_tag}.log"

export PGHOST=127.0.0.1
export PGPORT=55432
export PGDATABASE=hybrid_vector
export PGUSER=postgres
export PGPASSWORD=postgres
export PYTHONUNBUFFERED=1

truth="results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_yfcc_exact_truth_q12800.csv"
calib_workload="results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_yfcc_calibration.csv"
meas_workload="results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_yfcc_measurement.csv"
filters="results/hybrid_vector_db/yfcc10m_matched_recall_filters_q180.csv"
d2_proof="results/hybrid_vector_db/yfcc10m_r31_table6_shared_d2_warm_q100r5_20260723.csv.d2_graph_proof.json"
build_id="sqlens-v16-distance-aware-route-budget-ef500k-20260801-r41"
vector_sha="8f53226d35cae28d4e1b6926b13b01fa01fd1f6720c5f57c96c7886905f5eaf0"

mode_configs="${out_dir}/mode_configs_${run_tag}.json"
paired_csv="${out_dir}/yfcc_iso991_${run_tag}_q10k.csv"
summary_json="${out_dir}/yfcc_iso991_${run_tag}_summary.json"
qps_prefix="${out_dir}/yfcc_iso991_${run_tag}_c16_qps"
pilot_csv="${out_dir}/yfcc_iso991_${run_tag}_pilot_q2800.csv"

exec > >(tee -a "${log_file}") 2>&1
echo "[iso991] start $(date -Is) tag=${run_tag} phase=${phase} ef=${sqlens_ef} t=${sqlens_t} ratio=${sqlens_ratio}"

cat > "${mode_configs}" <<EOF
{
  "original": {
    "ef_search": 1500,
    "max_scan_tuples": 5000000,
    "scan_mem_multiplier": 32,
    "iterative_scan": "strict_order",
    "guided_collect_target": 1500,
    "traversal_guided_target": 10,
    "traversal_guided_prioritization": false,
    "traversal_guided_burst": 1,
    "traversal_guided_early_stop": false,
    "traversal_guided_early_stop_distance_ratio": 0
  },
  "design1_bloom_bfs_layout_d3": {
    "ef_search": ${sqlens_ef},
    "max_scan_tuples": 5000000,
    "scan_mem_multiplier": 32,
    "iterative_scan": "off",
    "guided_collect_target": ${sqlens_ef},
    "traversal_guided_target": ${sqlens_t},
    "traversal_guided_prioritization": true,
    "traversal_guided_burst": 1,
    "traversal_guided_early_stop": true,
    "traversal_guided_early_stop_distance_ratio": ${sqlens_ratio}
  }
}
EOF

run_bench() {
  local modes="$1"
  local workload="$2"
  local expected="$3"
  local out="$4"
  local unique_flag="$5"
  local namespace="$6"
  local cfg="$7"

  local unique_args=()
  if [[ "${unique_flag}" == "unique" ]]; then
    unique_args+=(--require-unique-workload-queries)
  else
    unique_args+=(--no-require-unique-workload-queries)
  fi

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
    --expected-workload-requests "${expected}" \
    "${unique_args[@]}" \
    --filters-csv "${filters}" \
    --modes ${modes} \
    --execution-order interleaved \
    --schedule-seed 20260801 \
    --mode-configs-json "${cfg}" \
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
    --guidance-low-selectivity-bypass-ef-search "${low_bypass_ef}" \
    --d2-source-on-guidance-bypass \
    --d1-exact-max-selectivity-pct 6 \
    --collapse-exact-and-guidance \
    --d3-cache-mb 1024 \
    --d3-measurement-policy workload_driven_adaptive \
    --d3-fragment-store-namespace "${namespace}" \
    --d3-probe-requests 2 \
    --d3-min-benefit-per-byte 0 \
    --d3-max-fragment-mb 16 \
    --d3-page-min-skip-rate 0.05 \
    --guidance-selectivity-min-pct 1.0 \
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
    --out "${out}"
}

if [[ "${phase}" == "pilot" || "${phase}" == "all" ]]; then
  if [[ ! -f "${pilot_csv}" ]]; then
    echo "[iso991] pilot q2800 paired ..."
    run_bench \
      "original design1_bloom_bfs_layout_d3" \
      "${calib_workload}" \
      2800 \
      "${pilot_csv}" \
      "nounique" \
      "table6-r41-yfcc-iso991-pilot-${run_tag}" \
      "${mode_configs}"
  else
    echo "[iso991] reuse pilot ${pilot_csv}"
  fi
  "${python_bin}" experiments/hybrid_vector_db/scripts/summarize_yfcc_iso991.py \
    --paired-csv "${pilot_csv}" \
    --out-summary "${out_dir}/yfcc_iso991_${run_tag}_pilot_summary.json" \
    --target-center 0.991 \
    --max-center-dev 0.006 \
    --max-recall-gap 0.004 \
    || echo "[iso991] pilot gate soft-fail (continuing; inspect summary)"
fi

if [[ "${phase}" == "paired" || "${phase}" == "all" ]]; then
  if [[ ! -f "${paired_csv}" ]]; then
    echo "[iso991] formal paired q10k ..."
    run_bench \
      "original design1_bloom_bfs_layout_d3" \
      "${meas_workload}" \
      10000 \
      "${paired_csv}" \
      "unique" \
      "table6-r41-yfcc-iso991-q10k-${run_tag}" \
      "${mode_configs}"
  else
    echo "[iso991] reuse paired ${paired_csv}"
  fi

  set +e
  "${python_bin}" experiments/hybrid_vector_db/scripts/summarize_yfcc_iso991.py \
    --paired-csv "${paired_csv}" \
    --out-summary "${summary_json}" \
    --target-center 0.991 \
    --max-center-dev 0.004 \
    --max-recall-gap 0.002
  gate_rc=$?
  set -e
  if [[ "${gate_rc}" -ne 0 ]]; then
    echo "[iso991] GATE FAIL rc=${gate_rc}. Adjust SQLENS_EF / SQLENS_T / SQLENS_RATIO and re-run."
    exit "${gate_rc}"
  fi
fi

if [[ "${phase}" == "qps" || "${phase}" == "all" ]]; then
  if [[ ! -f "${summary_json}" ]]; then
    echo "[iso991] missing ${summary_json}; run paired first" >&2
    exit 2
  fi
  echo "[iso991] c16 QPS ..."
  container_pid="$(docker inspect --format '{{.State.Pid}}' hybrid-pgvector)"
  sudo -n --preserve-env=HOME,PGHOST,PGPORT,PGDATABASE,PGUSER,PGPASSWORD,PYTHONUNBUFFERED \
    "${python_bin}" \
    experiments/hybrid_vector_db/scripts/pgvector_figure5_throughput.py \
    --frontier-config experiments/hybrid_vector_db/configs/figure5_r41_formal_datasets.json \
    --dataset yfcc \
    --workload-manifest results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_yfcc_manifest.json \
    --workload-request-limit 10000 \
    --config-id "yfcc-r41-target099-iso991-${run_tag}" \
    --pair-id "yfcc-r41-target099-iso991-${run_tag}" \
    --target-recall 0.99 \
    --stock-ef-search 1500 \
    --sqlens-ef-search "${sqlens_ef}" \
    --stock-iterative-scan strict_order \
    --sqlens-iterative-scan off \
    --stock-max-scan-tuples 5000000 \
    --sqlens-max-scan-tuples 5000000 \
    --stock-scan-mem-multiplier 32 \
    --sqlens-scan-mem-multiplier 32 \
    --stock-guided-collect-target 1500 \
    --sqlens-guided-collect-target "${sqlens_ef}" \
    --stock-traversal-guided-target 10 \
    --sqlens-traversal-guided-target "${sqlens_t}" \
    --stock-traversal-guided-burst 1 \
    --sqlens-traversal-guided-burst 1 \
    --no-stock-traversal-guided-early-stop \
    --sqlens-traversal-guided-early-stop \
    --stock-traversal-guided-early-stop-distance-ratio 0 \
    --sqlens-traversal-guided-early-stop-distance-ratio "${sqlens_ratio}" \
    --d1-exact-max-selectivity-pct 6 \
    --collapse-exact-and-guidance \
    --guidance-selectivity-min-pct 1.0 \
    --guidance-selectivity-max-pct 6 \
    --guidance-composite-max-selectivity-pct 6 \
    --guidance-max-atoms 8 \
    --d2-source-on-guidance-bypass \
    --guidance-bypass-iterative-scan strict_order \
    --guidance-bypass-ef-search "${bypass_ef}" \
    --guidance-low-selectivity-bypass-ef-search "${low_bypass_ef}" \
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
    --telemetry-devices nvme0n1 \
    --pg-prewarm \
    --statement-timeout-ms 7200000 \
    --bootstrap-samples 2000 \
    --bootstrap-seed 20260801 \
    --run-id "yfcc-iso991-${run_tag}-c16-r1" \
    --out-prefix "${qps_prefix}" \
    --overwrite \
    --execute
fi

echo "[iso991] done $(date -Is)"
echo "[iso991] summary=${summary_json}"
echo "[iso991] qps_prefix=${qps_prefix}"
