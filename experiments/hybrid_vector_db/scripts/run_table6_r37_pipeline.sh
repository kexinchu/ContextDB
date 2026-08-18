#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="experiments/hybrid_vector_db/configs/figure5_r37_formal_datasets.json"
GRID_PLAN="experiments/hybrid_vector_db/configs/table6_r37_isolated_grid_plan.json"
RELEASE="experiments/hybrid_vector_db/configs/p0_release_contract.json"
CAL_ROOT="results/hybrid_vector_db/figure5_r37_formal_calibration_isolated"
OUT="results/hybrid_vector_db/figure5_r37_table6"
GRID="$OUT/figure5_r37_required_grid_contract.json"
PREFIX="$OUT/figure5_r37_fixed_target_configs"
LAT="$OUT/matched_latency"
THR="$OUT/matched_throughput_fixed_targets_c16_q10k_r3"
LOG="$OUT/table6_r37_pipeline.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
trap 'status=$?; printf "[%s] FAILED line=%s status=%s\n" "$(date --iso-8601=seconds)" "$LINENO" "$status"; exit "$status"' ERR

run() {
  printf '\n[%s] RUN' "$(date --iso-8601=seconds)"
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

printf '[%s] Table 6 r37 pipeline start\n' "$(date --iso-8601=seconds)"

# Calibration cells share one PostgreSQL instance and are intentionally
# serial.  --resume preserves completed, individually lock-audited cells.
run "$PYTHON" experiments/hybrid_vector_db/scripts/run_figure5_sqlens_target_extension.py \
  --config "$CONFIG" --datasets amazon \
  --settings 200:200,500:500,5000:1000,10000:2000 \
  --backend-cpu-list 48-63 \
  --out-dir "$CAL_ROOT/amazon_sqlens" --manifest-name isolated_manifest.json \
  --require-global-db-lock --resume --overwrite --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/run_figure5_frontier.py \
  --config "$CONFIG" --phase calibration --datasets yfcc \
  --ef-search-values 250,2000,20000,50000 \
  --scan-families stock_strict --calibration-repeats 1 \
  --backend-cpu-list 48-63 --out-dir "$CAL_ROOT/yfcc_stock" \
  --require-global-db-lock --resume --overwrite --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/run_figure5_sqlens_target_extension.py \
  --config "$CONFIG" --datasets yfcc \
  --settings 100:20,200:80,5000:1000,10000:2000 \
  --backend-cpu-list 48-63 \
  --out-dir "$CAL_ROOT/yfcc_sqlens" --manifest-name isolated_manifest.json \
  --require-global-db-lock --resume --overwrite --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/run_figure5_frontier.py \
  --config "$CONFIG" --phase calibration --datasets laion \
  --ef-search-values 20000,50000,100000 \
  --scan-families stock_strict --calibration-repeats 1 \
  --backend-cpu-list 48-63 --out-dir "$CAL_ROOT/laion_stock" \
  --require-global-db-lock --resume --overwrite --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/run_figure5_sqlens_target_extension.py \
  --config "$CONFIG" --datasets laion \
  --settings 3000:250,5000:500,10000:1000 \
  --backend-cpu-list 48-63 \
  --out-dir "$CAL_ROOT/laion_sqlens" --manifest-name isolated_manifest.json \
  --require-global-db-lock --resume --overwrite --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/build_figure5_required_grid_contract.py \
  --grid-plan "$GRID_PLAN" --output "$GRID"

run "$PYTHON" experiments/hybrid_vector_db/scripts/select_figure5_matched_configs.py \
  --input-dir "$CAL_ROOT/amazon_stock" \
  --extra-input-dir "$CAL_ROOT/amazon_sqlens" \
  --extra-input-dir "$CAL_ROOT/yfcc_stock" \
  --extra-input-dir "$CAL_ROOT/yfcc_sqlens" \
  --extra-input-dir "$CAL_ROOT/laion_stock" \
  --extra-input-dir "$CAL_ROOT/laion_sqlens" \
  --contract "$RELEASE" --required-grid-contract "$GRID" \
  --out-prefix "$PREFIX" --targets 0.90,0.95,0.99 \
  --target-policy fixed --qualification-scope global_min_predicate_lcb \
  --bootstrap-samples 2000 --bootstrap-seed 20260728 --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/run_figure5_matched_latency.py \
  --config "$CONFIG" --selection-csv "$PREFIX.csv" \
  --selection-plan "$PREFIX.json" \
  --selection-manifest "$PREFIX.manifest.json" \
  --required-grid-contract "$GRID" --backend-cpu-list 48-63 \
  --out-dir "$LAT" --resume --no-overwrite --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/figure5_latency_repeats.py \
  --run-manifest "$LAT/figure5_r35_matched_latency_run_manifest.json" \
  --out "$OUT/figure5_r37_matched_latency_repeats.csv" \
  --binding-manifest "$OUT/figure5_r37_matched_latency_repeats.manifest.json"

PG_CONTAINER_PID="$(docker inspect -f '{{.State.Pid}}' hybrid-pgvector)"
run "$PYTHON" experiments/hybrid_vector_db/scripts/run_figure5_matched_throughput.py \
  --protocol-slice fixed-targets-c16-q10k-r3 --config "$CONFIG" \
  --selection-csv "$PREFIX.csv" --selection-plan "$PREFIX.json" \
  --selection-manifest "$PREFIX.manifest.json" \
  --required-grid-contract "$GRID" \
  --workload-manifest amazon=results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_amazon_manifest.json \
  --workload-manifest yfcc=results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_yfcc_manifest.json \
  --workload-manifest laion=results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_laion_manifest.json \
  --clients 16 --client-cpu-list 0-31 --backend-cpu-list 48-63 \
  --backend-proc-root "/proc/${PG_CONTAINER_PID}/root/proc" \
  --telemetry-devices sda --pg-prewarm --out-dir "$THR" \
  --resume --no-overwrite --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/figure5_throughput_repeats.py \
  --run-manifest "$THR/figure5_r36_fixed-targets-c16-q10k-r3_throughput_run_manifest.json" \
  --out "$OUT/figure5_r37_matched_throughput_repeats.csv" \
  --service-summary "$OUT/figure5_r37_matched_throughput_service.csv" \
  --binding-manifest "$OUT/figure5_r37_matched_throughput_repeats.manifest.json"

run "$PYTHON" experiments/hybrid_vector_db/scripts/build_table6_matched_recall_summary.py \
  --selection-csv "$PREFIX.csv" --selection-plan "$PREFIX.json" \
  --selection-manifest "$PREFIX.manifest.json" \
  --required-grid-contract "$GRID" \
  --latency-run-manifest "$LAT/figure5_r35_matched_latency_run_manifest.json" \
  --throughput-repeat-csv "$OUT/figure5_r37_matched_throughput_repeats.csv" \
  --throughput-service-csv "$OUT/figure5_r37_matched_throughput_service.csv" \
  --throughput-binding-manifest "$OUT/figure5_r37_matched_throughput_repeats.manifest.json" \
  --out-csv "$OUT/table6_matched_recall_summary.csv" \
  --out-json "$OUT/table6_matched_recall_summary.json" \
  --bootstrap-samples 2000 --bootstrap-seed 20260728

printf '[%s] Table 6 r37 pipeline complete\n' "$(date --iso-8601=seconds)"
touch "$OUT/PIPELINE_COMPLETE"
