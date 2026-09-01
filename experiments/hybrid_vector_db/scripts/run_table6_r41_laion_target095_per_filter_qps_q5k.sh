#!/usr/bin/env bash
set -euo pipefail

# LAION-25M target-0.95 16-client QPS, matched to the paper latency row:
# results/.../laion_target095_per_filter_paired_q5k_final.csv
# (modes.json defaults + per-filter ef overrides; q5000 measurement prefix)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON:-python3}"
telemetry_path="${SQLENS_TELEMETRY_PATH:-${repo_root}/.pgvector-data}"

export PGHOST="${PGHOST:-127.0.0.1}"
export PGPORT="${PGPORT:-55432}"
export PGDATABASE="${PGDATABASE:-hybrid_vector}"
export PGUSER="${PGUSER:-postgres}"
: "${PGPASSWORD:?set PGPASSWORD}"
export PYTHONUNBUFFERED=1

out_dir="${OUT_DIR:-results/hybrid_vector_db/table6_r41_laion_target095_per_filter_qps_q5k}"
client_cpu_list="${CLIENT_CPU_LIST:-0-31}"
backend_cpu_list="${BACKEND_CPU_LIST:-48-63}"
run_suffix="${RUN_SUFFIX:-r1}"
pg_container="${PG_CONTAINER:-hybrid-pgvector}"
mkdir -p "${out_dir}"
container_pid="$(docker inspect --format '{{.State.Pid}}' "${pg_container}")"

exec sudo -n env \
  "HOME=${HOME}" \
  "PATH=${PATH}" \
  "PYTHONPATH=${PYTHONPATH}" \
  "PGHOST=${PGHOST}" \
  "PGPORT=${PGPORT}" \
  "PGDATABASE=${PGDATABASE}" \
  "PGUSER=${PGUSER}" \
  "PGPASSWORD=${PGPASSWORD}" \
  "PYTHONUNBUFFERED=1" \
  "${python_bin}" \
  experiments/hybrid_vector_db/scripts/pgvector_figure5_throughput.py \
  --frontier-config experiments/hybrid_vector_db/configs/figure5_r41_formal_datasets.json \
  --dataset laion \
  --workload-manifest results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_laion_manifest.json \
  --workload-request-limit 5000 \
  --config-id table6-r41-laion-t095-per-filter-q5k-c16 \
  --pair-id table6-r41-laion-t095-per-filter-q5k-c16 \
  --target-recall 0.95 \
  --stock-ef-search 2000 \
  --sqlens-ef-search 2000 \
  --stock-iterative-scan strict_order \
  --sqlens-iterative-scan off \
  --stock-max-scan-tuples 5000000 \
  --sqlens-max-scan-tuples 5000000 \
  --stock-scan-mem-multiplier 32 \
  --sqlens-scan-mem-multiplier 32 \
  --stock-guided-collect-target 2000 \
  --sqlens-guided-collect-target 2000 \
  --stock-traversal-guided-target 10 \
  --sqlens-traversal-guided-target 160 \
  --stock-traversal-guided-burst 1 \
  --sqlens-traversal-guided-burst 8 \
  --filter-ef-search-json experiments/hybrid_vector_db/configs/table6_r41_laion_target095_per_filter_ef.json \
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
  --schedule-seed 20260808 \
  --client-cpu-list "${client_cpu_list}" \
  --backend-cpu-list "${backend_cpu_list}" \
  --backend-proc-root "/proc/${container_pid}/root/proc" \
  --telemetry-path "${telemetry_path}" \
  --pg-prewarm \
  --statement-timeout-ms 7200000 \
  --bootstrap-samples 2000 \
  --bootstrap-seed 20260808 \
  --run-id "table6-r41-laion-t095-per-filter-q5k-c16-${run_suffix}" \
  --out-prefix "${out_dir}/laion_target095_per_filter_q5k_c16_${run_suffix}" \
  --overwrite \
  --execute
