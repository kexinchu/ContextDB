#!/usr/bin/env bash
# Q3: matched-recall acorn1 vs stock vs VisGuide on four Amazon atoms.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-$repo_root}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-python3}
lock_path="$results/.pg55437_experiment.lock"
start_script="$script_dir/../start_amazon_table10_r43.sh"
out_dir="$results/revision45/q3_acorn_matched"

if [[ ${1:-} != --execute ]]; then
  echo '{"dry_run": true, "plan_item": "Q3", "paper_eligible": false}'
  "$python" "$script_dir/run_q3_acorn_matched.py"
  exit 0
fi

mkdir -p "$out_dir"
"$start_script"
export PYTHONUNBUFFERED=1
export PGHOST=127.0.0.1
export PGPORT=55437
export PGDATABASE=hybrid_vector
export PGUSER=postgres
: "${PGPASSWORD:?set PGPASSWORD}"
export PYTHONPATH="$(dirname -- "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"

exec 9>>"$lock_path"
flock 9
{
  echo "REV45_Q3_EXECUTE:$(date -Is)"
  "$python" "$script_dir/run_q3_acorn_matched.py" --execute --out-dir "$out_dir"
  echo "REV45_Q3_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$out_dir/run.log"
