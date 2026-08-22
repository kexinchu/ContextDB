#!/usr/bin/env bash
# Fast confirmation: reuse q80 calibration, measure q2k/r1 with parallel SQL-first.
# Not paper-eligible. The parked q10k/r3 checkpoint remains the formal run.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-/home/kec23008/Hybrid-Retrieval}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-/home/kec23008/miniconda3/bin/python3}
lock_path="$results/.pg55437_experiment.lock"
runner="$repo_root/experiments/hybrid_vector_db/scripts/amazon10m_sql_native_benchmark.py"
start_script="$script_dir/start_amazon_table10_r43.sh"
log="$results/amazon10m_sql_native_p0_r43_q2k_r1_confirm.log"
checkpoint="$results/amazon10m_sql_native_p0_r43_q2k_r1_confirm.checkpoint"
calib_source="${1:-$results/amazon10m_sql_native_p0_r43_q10k_r3.checkpoint}"

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
export PGPASSWORD=postgres
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

reuse_args=()
if [[ -d $calib_source ]]; then
  reuse_args=(--reuse-calibration-from "$calib_source")
fi
resume_args=()
if [[ -d $checkpoint ]]; then
  resume_args=(--resume)
  reuse_args=()
fi

exec 9>>"$lock_path"
flock 9
{
  echo "P0_CONFIRM_EXECUTE:$(date -Is) resume=${#resume_args[@]} reuse=${#reuse_args[@]}"
  "$python" "$runner" \
    --protocol q10200 \
    --confirmation \
    --execute \
    "${reuse_args[@]}" \
    "${resume_args[@]}" \
    --expected-sqlens-build-id sqlens-v17-predistance-promotion-20260806-r43 \
    --expected-vector-so-sha256 2056a67b9b0012c401c6684d49915cbc31bc8fa770946dbfaddda9d779eecbf2
  echo "P0_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$log"
