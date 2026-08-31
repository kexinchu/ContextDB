#!/usr/bin/env bash
# Aligned ACORN-1 on the r44 Amazon replica (55440).
# Does not touch 55437 or rewrite paper tables.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-$repo_root}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-/home/kec23008/miniconda3/bin/python3}
lock_path="$results/.pg55440_experiment.lock"
out_dir="$results/revision45/q3_acorn_aligned"

if [[ ${1:-} != --execute ]]; then
  echo '{"dry_run": true, "plan_item": "Q3_ACORN_ALIGNED", "paper_eligible": false}'
  "$python" "$script_dir/run_q3_acorn_aligned.py" "$@"
  exit 0
fi
shift

mkdir -p "$out_dir"
export PYTHONUNBUFFERED=1
export PGHOST=127.0.0.1
export PGPORT=55440
export PGDATABASE=hybrid_vector
export PGUSER=postgres
export PGPASSWORD=postgres
export PYTHONPATH="$(dirname -- "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"

exec 9>>"$lock_path"
flock 9
{
  echo "REV45_Q3_ACORN_ALIGNED_EXECUTE:$(date -Is)"
  "$python" "$script_dir/run_q3_acorn_aligned.py" --execute --out-dir "$out_dir" "$@"
  echo "REV45_Q3_ACORN_ALIGNED_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$out_dir/run.log"
