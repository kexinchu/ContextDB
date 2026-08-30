#!/usr/bin/env bash
# Full SQLens (d1_d2_d3 on BFS) vs acorn1 on the r44 Amazon replica.
# Does not touch 55437 or rewrite paper/tables/eval_acorn_matched.tex.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-$repo_root}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-/home/kec23008/miniconda3/bin/python3}
lock_path="$results/.pg55440_experiment.lock"
out_dir="$results/revision45/q3_full_vs_acorn"

if [[ ${1:-} != --execute ]]; then
  echo '{"dry_run": true, "plan_item": "Q3_FULL_VS_ACORN", "paper_eligible": false}'
  "$python" "$script_dir/run_q3_full_vs_acorn.py"
  exit 0
fi

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
  echo "REV45_Q3_FULL_VS_ACORN_EXECUTE:$(date -Is)"
  "$python" "$script_dir/run_q3_full_vs_acorn.py" --execute --out-dir "$out_dir" "$@"
  echo "REV45_Q3_FULL_VS_ACORN_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$out_dir/run.log"
