#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
CONFIG="experiments/hybrid_vector_db/configs/figure5_r37_formal_datasets.json"
RELEASE="experiments/hybrid_vector_db/configs/p0_release_contract.json"
GRID_PLAN="experiments/hybrid_vector_db/configs/table6_r37_amazon_grid_plan.json"
CAL_ROOT="results/hybrid_vector_db/figure5_r37_formal_calibration_isolated"
OUT="results/hybrid_vector_db/figure5_r37_table6_amazon_priority"
GRID="$OUT/figure5_r37_amazon_required_grid_contract.json"
PREFIX="$OUT/figure5_r37_amazon_fixed_target_configs"
LAT="$OUT/matched_latency"
THR="$OUT/matched_throughput_fixed_targets_c16_q10k_r3"
REPAIR_MANIFEST="$CAL_ROOT/amazon_sqlens/repair_ef200_manifest.json"
LOG="$OUT/pipeline.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
trap 'status=$?; printf "[%s] FAILED line=%s status=%s\n" "$(date --iso-8601=seconds)" "$LINENO" "$status"; exit "$status"' ERR

run() {
  printf '\n[%s] RUN' "$(date --iso-8601=seconds)"
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

printf '[%s] Amazon Table 6 priority pipeline start\n' "$(date --iso-8601=seconds)"

while ! jq -e '.status == "complete" and .requested_slice_complete == true' \
  "$REPAIR_MANIFEST" >/dev/null 2>&1; do
  printf '[%s] waiting for Amazon ef200 calibration repair\n' "$(date --iso-8601=seconds)"
  sleep 30
done

# The repair manifest is committed before its process releases the flock.
sleep 5

# Re-audit all four existing Amazon SQLens calibration cells without replacing
# their raw outputs. The grid contract records that the old manifest's lock
# evidence was lost; no replacement lock token is invented here.
run "$PYTHON" experiments/hybrid_vector_db/scripts/run_figure5_sqlens_target_extension.py \
  --config "$CONFIG" --datasets amazon \
  --settings 200:200,500:500,5000:1000,10000:2000 \
  --backend-cpu-list 48-63 --out-dir "$CAL_ROOT/amazon_sqlens" \
  --manifest-name isolated_manifest.json \
  --resume --no-overwrite --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/build_figure5_required_grid_contract.py \
  --grid-plan "$GRID_PLAN" --output "$GRID"

run "$PYTHON" experiments/hybrid_vector_db/scripts/select_figure5_matched_configs.py \
  --input-dir "$CAL_ROOT/amazon_stock" \
  --extra-input-dir "$CAL_ROOT/amazon_sqlens" \
  --contract "$RELEASE" --required-grid-contract "$GRID" \
  --out-prefix "$PREFIX" --targets 0.90,0.95,0.99 \
  --aggregate-lcb-override-targets 0.99 \
  --target-policy fixed --qualification-scope global_min_predicate_lcb \
  --bootstrap-samples 2000 --bootstrap-seed 20260728 --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/run_figure5_matched_latency.py \
  --config "$CONFIG" --selection-csv "$PREFIX.csv" \
  --selection-plan "$PREFIX.json" \
  --selection-manifest "$PREFIX.manifest.json" \
  --required-grid-contract "$GRID" --datasets amazon \
  --backend-cpu-list 48-63 --out-dir "$LAT" \
  --resume --no-overwrite --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/figure5_latency_repeats.py \
  --run-manifest "$LAT/figure5_r35_matched_latency_run_manifest.json" \
  --out "$OUT/figure5_r37_amazon_matched_latency_repeats.csv" \
  --binding-manifest "$OUT/figure5_r37_amazon_matched_latency_repeats.manifest.json"

PG_CONTAINER_PID="$(docker inspect -f '{{.State.Pid}}' hybrid-pgvector)"
run "$PYTHON" experiments/hybrid_vector_db/scripts/run_figure5_matched_throughput.py \
  --protocol-slice fixed-targets-c16-q10k-r3 --config "$CONFIG" \
  --selection-csv "$PREFIX.csv" --selection-plan "$PREFIX.json" \
  --selection-manifest "$PREFIX.manifest.json" \
  --required-grid-contract "$GRID" --datasets amazon \
  --workload-manifest amazon=results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_amazon_manifest.json \
  --clients 16 --client-cpu-list 0-31 --backend-cpu-list 48-63 \
  --backend-proc-root "/proc/${PG_CONTAINER_PID}/root/proc" \
  --telemetry-devices sda --pg-prewarm --out-dir "$THR" \
  --resume --no-overwrite --execute

run "$PYTHON" experiments/hybrid_vector_db/scripts/figure5_throughput_repeats.py \
  --run-manifest "$THR/figure5_r36_fixed-targets-c16-q10k-r3_throughput_run_manifest.json" \
  --out "$OUT/figure5_r37_amazon_matched_throughput_repeats.csv" \
  --service-summary "$OUT/figure5_r37_amazon_matched_throughput_service.csv" \
  --binding-manifest "$OUT/figure5_r37_amazon_matched_throughput_repeats.manifest.json"

printf '[%s] Amazon Table 6 measurements complete\n' "$(date --iso-8601=seconds)"
touch "$OUT/MEASUREMENTS_COMPLETE"
