#!/usr/bin/env bash
# Layer 1: ACORN q10K. New artifact dir. Does not touch q50 or Table 5.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-$repo_root}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-/home/kec23008/miniconda3/bin/python3}
lock_path="$results/.pg55437_experiment.lock"
start_script="$script_dir/../start_amazon_table10_r43.sh"
out_dir="$results/revision45/q3_acorn_q10k"

if [[ ${1:-} != --execute ]]; then
  echo '{"dry_run": true, "plan_item": "L1_ACORN_Q10K", "rewrites_published": false}'
  "$python" "$script_dir/run_q3_acorn_q10k.py"
  exit 0
fi

mkdir -p "$out_dir"
"$start_script"
export PYTHONUNBUFFERED=1
export PGHOST=127.0.0.1
export PGPORT=55437
export PGDATABASE=hybrid_vector
export PGUSER=postgres
export PGPASSWORD=postgres
export PYTHONPATH="$(dirname -- "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"

exec 9>>"$lock_path"
flock 9
{
  echo "L1_ACORN_Q10K_START:$(date -Is)"
  "$python" "$script_dir/run_q3_acorn_q10k.py" --execute --out-dir "$out_dir"
  echo "L1_ACORN_Q10K_EXIT:$? $(date -Is)"
} 2>&1 | stdbuf -oL -eL tee -a "$out_dir/run.log"
