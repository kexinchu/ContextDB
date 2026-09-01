#!/usr/bin/env bash
# q1K/r1 screening for Figure 5 JOIN hybrid search. Not paper-eligible.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-python3}
lock_path="$results/.pg55437_experiment.lock"
runner="$repo_root/experiments/hybrid_vector_db/scripts/amazon10m_sql_native_benchmark.py"
start_script="$script_dir/start_amazon_table10_r43.sh"
gt_dir="$results/amazon10m_sql_native_q10200_r43_sqlops_join"
gt_manifest="$gt_dir/amazon10m_sql_native_exact_truth_manifest.json"
log="$results/amazon10m_sql_native_p0_r43_q1k_r1_sqlops_join_screen.log"
checkpoint="$results/amazon10m_sql_native_p0_r43_q1k_r1_sqlops_join_screen.checkpoint"

if [[ ! -f $gt_manifest ]]; then
  echo "join exact truth is missing; run run_amazon10m_sql_native_figure5_sqlops_gt.sh first" >&2
  exit 2
fi

mkdir -p "$results"
"$start_script"

if ! flock -n "$lock_path" -c true; then
  echo "Amazon Table-10 lock is already held: $lock_path" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PGHOST=127.0.0.1
export PGPORT=55437
export PGDATABASE=hybrid_vector
export PGUSER=postgres
: "${PGPASSWORD:?set PGPASSWORD}"
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

resume_args=()
if [[ -d $checkpoint ]]; then
  resume_args=(--resume)
fi

exec 9>>"$lock_path"
flock 9
{
  echo "P0_JOIN_SCREEN_EXECUTE:$(date -Is) resume=${#resume_args[@]}"
  "$python" "$runner" \
    --protocol q10200 \
    --screening \
    --execute \
    "${resume_args[@]}" \
    --exact-truth-csv "$gt_dir/amazon10m_sql_native_exact_truth_q10200.csv" \
    --exact-truth-manifest "$gt_manifest" \
    --expected-sqlens-build-id sqlens-v17-predistance-promotion-20260806-r43 \
    --expected-vector-so-sha256 2056a67b9b0012c401c6684d49915cbc31bc8fa770946dbfaddda9d779eecbf2
  echo "P0_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$log"
