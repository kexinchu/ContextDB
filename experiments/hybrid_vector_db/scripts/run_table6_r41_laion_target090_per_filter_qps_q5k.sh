#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON:-python3}"
telemetry_path="${SQLENS_TELEMETRY_PATH:-${repo_root}/.pgvector-data}"

export PGHOST=127.0.0.1
export PGPORT=55432
export PGDATABASE=hybrid_vector
export PGUSER=postgres
export PGPASSWORD=postgres
export PYTHONUNBUFFERED=1

out_dir="results/hybrid_vector_db/table6_r41_laion_target090_per_filter_qps_q5k"
mkdir -p "${out_dir}"

exec "${python_bin}" \
  experiments/hybrid_vector_db/scripts/pgvector_figure5_throughput.py \
  --frontier-config experiments/hybrid_vector_db/configs/figure5_r41_formal_datasets.json \
  --dataset laion \
  --workload-manifest results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_laion_manifest.json \
  --workload-request-limit 5000 \
  --config-id table6-r41-laion-t090-per-filter-q5k-c16 \
  --pair-id table6-r41-laion-t090-per-filter-q5k-c16 \
  --target-recall 0.90 \
  --stock-ef-search 1000 \
  --sqlens-ef-search 1000 \
  --stock-iterative-scan strict_order \
  --sqlens-iterative-scan off \
  --stock-max-scan-tuples 5000000 \
  --sqlens-max-scan-tuples 5000000 \
  --stock-scan-mem-multiplier 32 \
  --sqlens-scan-mem-multiplier 32 \
  --stock-guided-collect-target 1000 \
  --sqlens-guided-collect-target 1000 \
  --stock-traversal-guided-target 10 \
  --sqlens-traversal-guided-target 80 \
  --stock-traversal-guided-burst 1 \
  --sqlens-traversal-guided-burst 8 \
  --filter-ef-search-json experiments/hybrid_vector_db/configs/table6_r41_laion_target090_per_filter_ef.json \
  --filter-traversal-target-json experiments/hybrid_vector_db/configs/table6_r41_laion_target090_per_filter_targets.json \
  --d1-exact-max-selectivity-pct 6 \
  --collapse-exact-and-guidance \
  --guidance-selectivity-min-pct 0.6 \
  --guidance-selectivity-max-pct 6 \
  --guidance-composite-max-selectivity-pct 6 \
  --guidance-max-atoms 8 \
  --d2-source-on-guidance-bypass \
  --guidance-bypass-ef-search 0 \
  --guidance-low-selectivity-bypass-ef-search 0 \
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
  --schedule-seed 20260805 \
  --client-cpu-list 0-31 \
  --backend-cpu-list 48-63 \
  --backend-proc-root /proc/3960809/root/proc \
  --telemetry-path "${telemetry_path}" \
  --pg-prewarm \
  --statement-timeout-ms 7200000 \
  --bootstrap-samples 2000 \
  --bootstrap-seed 20260805 \
  --run-id table6-r41-laion-t090-per-filter-q5k-c16-r1 \
  --out-prefix "${out_dir}/laion_target090_per_filter_q5k_c16_r1" \
  --overwrite \
  --execute
