#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON:-python3}"

dataset=${1:?usage: run_target090_q5k_matched_qps.sh amazon|yfcc}
case "$dataset" in
  amazon)
    export PGHOST=127.0.0.1 PGPORT=55433 PGDATABASE=hybrid_vector
    : "${PGUSER:?set PGUSER}" "${PGPASSWORD:?set PGPASSWORD}"
    workload_manifest=results/hybrid_vector_db/table6_r41_target090_q5k_workloads/amazon_target090_q5k_manifest.json
    workload_limit=0
    backend_cpus=0-31
    client_cpus=32-47
    container=hybrid-pgvector-amazon-frontier
    ;;
  yfcc)
    export PGHOST=127.0.0.1 PGPORT=55432 PGDATABASE=hybrid_vector
    : "${PGUSER:?set PGUSER}" "${PGPASSWORD:?set PGPASSWORD}"
    workload_manifest=results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_yfcc_manifest.json
    workload_limit=5000
    backend_cpus=48-63
    client_cpus=32-47
    container=hybrid-pgvector
    ;;
  *)
    echo "unknown dataset: $dataset" >&2
    exit 2
    ;;
esac

container_pid=$(docker inspect -f '{{.State.Pid}}' "$container")
out_dir=results/hybrid_vector_db/target090_q5k_r41/audited_qps
mkdir -p "$out_dir"

exec "${python_bin}" \
  experiments/hybrid_vector_db/scripts/pgvector_figure5_throughput.py \
  --frontier-config experiments/hybrid_vector_db/configs/target090_q5k_throughput_datasets.json \
  --dataset "$dataset" \
  --workload-manifest "$workload_manifest" \
  --workload-request-limit "$workload_limit" \
  --release-contract experiments/hybrid_vector_db/configs/target090_q5k_r41_release_contract.json \
  --pair-id "$dataset-target090-q5k-per-filter-r41" \
  --config-id "$dataset-target090-q5k-per-filter-r41" \
  --target-recall 0.90 \
  --stock-ef-search 100 \
  --sqlens-ef-search 100 \
  --stock-iterative-scan off \
  --sqlens-iterative-scan off \
  --stock-max-scan-tuples 5000000 \
  --sqlens-max-scan-tuples 5000000 \
  --scan-mem-multiplier 32 \
  --stock-guided-collect-target 100 \
  --sqlens-guided-collect-target 100 \
  --stock-traversal-guided-target 100 \
  --sqlens-traversal-guided-target 100 \
  --stock-traversal-guided-burst 1 \
  --sqlens-traversal-guided-burst 8 \
  --no-stock-traversal-guided-early-stop \
  --sqlens-traversal-guided-early-stop \
  --stock-traversal-guided-early-stop-distance-ratio 0 \
  --sqlens-traversal-guided-early-stop-distance-ratio 0.95 \
  --filter-mode-configs-json "results/hybrid_vector_db/target090_q5k_r41/${dataset}_target090_q5k_filter_mode_configs.json" \
  --d1-exact-max-selectivity-pct 6 \
  --collapse-exact-and-guidance \
  --guidance-selectivity-min-pct 0 \
  --guidance-selectivity-max-pct 6 \
  --guidance-composite-max-selectivity-pct 100 \
  --guidance-max-atoms 8 \
  --no-d2-source-on-guidance-bypass \
  --guidance-bypass-iterative-scan off \
  --guidance-bypass-ef-search 0 \
  --guidance-low-selectivity-bypass-ef-search 0 \
  --d3-cache-mb 1024 \
  --d3-probe-requests 2 \
  --d3-min-benefit-per-byte 0 \
  --d3-max-fragment-mb 16 \
  --d3-page-min-skip-rate 0.05 \
  --clients 16 \
  --repeats 1 \
  --allow-single-pass \
  --schedule-seed 20260803 \
  --backend-cpu-list "$backend_cpus" \
  --client-cpu-list "$client_cpus" \
  --backend-proc-root "/proc/$container_pid/root/proc" \
  --telemetry-devices nvme0n1 \
  --run-id "$dataset-target090-q5k-c16-r1" \
  --out-prefix "$out_dir/$dataset-target090-q5k-c16-r1" \
  --execute
