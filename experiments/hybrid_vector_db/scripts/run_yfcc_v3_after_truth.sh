#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WDIR="$ROOT/results/hybrid_vector_db/figure5_r41_yfcc_v3filters_workloads"
TRUTH="$WDIR/figure5_r41_yfcc_v3_exact_truth_q12800.csv"
TRUTH_MANIFEST="$WDIR/figure5_r41_yfcc_v3_exact_truth_q12800_manifest.json"
TRUTH_PID="${YFCC_V3_TRUTH_PID:?set YFCC_V3_TRUTH_PID}"

while [[ ! -s "$TRUTH_MANIFEST" ]]; do
  if ! kill -0 "$TRUTH_PID" 2>/dev/null; then
    echo "[error] exact-truth process exited without a manifest" >&2
    exit 1
  fi
  sleep 60
done

cd "$ROOT"
python3 experiments/hybrid_vector_db/scripts/build_figure5_frontier_workload.py \
  --query-cohort-csv results/hybrid_vector_db/figure5_r35_yfcc_query_cohort_q10200.csv \
  --filters-csv results/hybrid_vector_db/yfcc10m_figure5_v3_filters_uniform_0p2_14pct.csv \
  --truth-csv "$TRUTH" \
  --out-prefix "$WDIR/figure5_r41_yfcc_v3" \
  --calibration-protocol formal_per_predicate_cartesian_v1 \
  --calibration-requests 2800 \
  --calibration-query-count 200 \
  --require-formal-paper-calibration \
  --truth-coverage assigned

python3 -u experiments/hybrid_vector_db/scripts/run_yfcc_v3_matched_iso_speedup.py
