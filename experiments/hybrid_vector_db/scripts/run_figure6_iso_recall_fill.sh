#!/usr/bin/env bash
# Figure 6 iso-recall fill orchestrator.
#
# Targets (0.05 treated as 0.95):
#   0.75 0.80 0.85 0.90 0.94 0.95 0.96 0.97 0.98 0.99
#
# Phases:
#   audit       — coverage matrix + fill queue (no DB)
#   amazon-refresh — rebuild Amazon latency pairs from existing calib (no DB)
#   yfcc-latency   — continue/finish YFCC v4 matched-iso on :55432
#   yfcc-throughput — c16 QPS after latency configs exist
#   laion-latency   — LAION matched-iso calibration on :55434
#   laion-throughput
#   all-offline     — audit + amazon-refresh only
set -euo pipefail

ROOT=/home/kec23008/Hybrid-Retrieval
cd "$ROOT"
PY="${PYTHON:-/home/kec23008/miniconda3/bin/python}"
CFG=experiments/hybrid_vector_db/configs/figure6_iso_recall_targets.json
OUT=results/hybrid_vector_db/figure6_iso_recall_fill
TARGETS=0.75,0.80,0.85,0.90,0.94,0.95,0.96,0.97,0.98,0.99
PHASE="${1:-all-offline}"

mkdir -p "$OUT" "$OUT/logs"

audit() {
  echo "[figure6] audit coverage"
  "$PY" experiments/hybrid_vector_db/scripts/audit_figure6_iso_recall_coverage.py \
    --config "$CFG" \
    --out-dir "$OUT"
}

amazon_refresh() {
  echo "[figure6] refresh Amazon latency pairs (both_off primary + attainable)"
  "$PY" experiments/hybrid_vector_db/scripts/plot_amazon10m_iso_recall.py \
    --refresh \
    --targets "$TARGETS"
  cp -f results/hybrid_vector_db/amazon10m_iso_recall_plot/figures/amazon10m_iso_recall_latency.pdf \
    paper/figures/amazon10m_iso_recall_latency.pdf 2>/dev/null || true
  cp -f results/hybrid_vector_db/amazon10m_iso_recall_plot/figures/amazon10m_throughput_vs_recall.pdf \
    paper/figures/ 2>/dev/null || true
}

yfcc_latency() {
  echo "[figure6] YFCC v4 matched-iso latency (uses :55432 lock)"
  bash experiments/hybrid_vector_db/scripts/run_yfcc_v4_matched_iso_pipeline.sh
}

yfcc_throughput() {
  echo "[figure6] YFCC throughput fill requires selected iso pairs first"
  local pairs="$OUT/yfcc_iso_recall_selected_pairs.json"
  if [[ ! -f "$pairs" ]]; then
    echo "missing $pairs — run latency select and export pairs first" >&2
    exit 1
  fi
  echo "TODO: wire pgvector_figure5_throughput.py from selected pairs → $OUT/yfcc"
  exit 1
}

laion_latency() {
  echo "[figure6] LAION iso-recall latency not yet wired as a first-class pipeline"
  echo "Use figure5 formal calibration on :55434 after r43 releases the DB."
  echo "Config: experiments/hybrid_vector_db/configs/figure5_r41_formal_datasets.json"
  exit 1
}

case "$PHASE" in
  audit) audit ;;
  amazon-refresh) amazon_refresh; audit ;;
  yfcc-latency) yfcc_latency; audit ;;
  yfcc-throughput) yfcc_throughput ;;
  laion-latency) laion_latency ;;
  all-offline) amazon_refresh; audit ;;
  *)
    echo "usage: $0 {audit|amazon-refresh|yfcc-latency|yfcc-throughput|laion-latency|all-offline}" >&2
    exit 2
    ;;
esac

echo "[figure6] done phase=$PHASE"
cat "$OUT/coverage.md"
