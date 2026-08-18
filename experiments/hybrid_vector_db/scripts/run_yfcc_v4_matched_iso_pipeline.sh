#!/usr/bin/env bash
# YFCC v4 (0.5–8% filters): wait for exact truth + primary DB, then matched-iso.
set -euo pipefail

ROOT=/home/kec23008/Hybrid-Retrieval
WDIR="$ROOT/results/hybrid_vector_db/figure5_r41_yfcc_v4filters_workloads"
TRUTH="$WDIR/figure5_r41_yfcc_v4_exact_truth_q12800.csv"
TRUTH_MANIFEST="$WDIR/figure5_r41_yfcc_v4_exact_truth_q12800_manifest.json"
CFG="$ROOT/experiments/hybrid_vector_db/configs/figure5_r41_yfcc_primary_v4filters.json"
OUT="$ROOT/results/hybrid_vector_db/figure5_r41_yfcc_v4_matched_iso"
PLOT="$ROOT/results/hybrid_vector_db/yfcc10m_v4_matched_iso"
WARM="$ROOT/results/hybrid_vector_db/yfcc10m_v4_matched_iso_warm"
LOGDIR=/mnt/nvme-pg/home/kec23008/pg-amazon-frontier/logs
STATUS="$OUT/pipeline.status.json"
mkdir -p "$OUT" "$PLOT" "$WARM" "$LOGDIR"

log() { echo "[$(date -Is)] $*"; }
write_status() {
  python3 - "$STATUS" "$@" <<'PY'
import json, sys, time
path = sys.argv[1]
state = sys.argv[2]
extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
payload = {"state": state, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
open(path, "w").write(json.dumps(payload, indent=2) + "\n")
print(payload)
PY
}

write_status waiting_truth '{}'
log "waiting for exact truth manifest: $TRUTH_MANIFEST"
while [[ ! -s "$TRUTH_MANIFEST" ]]; do
  sleep 30
done
log "truth ready; rebuilding audited workload"
cd "$ROOT"
python3 experiments/hybrid_vector_db/scripts/build_figure5_frontier_workload.py \
  --query-cohort-csv results/hybrid_vector_db/figure5_r35_yfcc_query_cohort_q10200.csv \
  --filters-csv results/hybrid_vector_db/yfcc10m_figure5_v4_filters_and_0p5_8pct.csv \
  --truth-csv "$TRUTH" \
  --out-prefix "$WDIR/figure5_r41_yfcc_v4" \
  --calibration-protocol formal_per_predicate_cartesian_v1 \
  --calibration-requests 2800 \
  --calibration-query-count 200 \
  --require-formal-paper-calibration \
  --truth-coverage assigned

write_status waiting_db '{}'
log "waiting for primary YFCC DB (no table6/figure5 yfcc benchmark on CPUs 48-63)"
while pgrep -af 'pgvector_design1_design2_design3_selectivity_benchmark.py' \
  | rg -q 'yfcc10m_pgvector|figure5_r41_yfcc|run_figure5_frontier'; do
  sleep 60
done
# Also wait until iso991 shell exits if still holding the slot.
while pgrep -af 'run_yfcc_target099_iso991.sh' >/dev/null; do
  sleep 60
done

write_status running_matched_iso '{}'
log "starting v4 matched-iso Stock vs SQLens"
python3 -u experiments/hybrid_vector_db/scripts/run_yfcc_v2_matched_iso_speedup.py \
  --config "$CFG" \
  --out-dir "$OUT" \
  --plot-dir "$PLOT" \
  2>&1 | tee "$LOGDIR/yfcc_v4_matched_iso_$(date +%Y%m%d_%H%M%S).log"

write_status running_warm_eval '{}'
log "warm-latency evaluation (paper-facing mean)"
python3 experiments/hybrid_vector_db/scripts/eval_yfcc_matched_iso_warm.py \
  --out-dir "$WARM" \
  --search-dir "$OUT/stock" \
  --search-dir "$OUT/sqlens" \
  2>&1 | tee "$LOGDIR/yfcc_v4_matched_iso_warm_$(date +%Y%m%d_%H%M%S).log"

write_status done "{\"warm_summary\":\"$WARM/yfcc10m_matched_iso_warm_summary.csv\"}"
log "pipeline complete"
cat "$WARM/yfcc10m_matched_iso_warm_summary.csv"
