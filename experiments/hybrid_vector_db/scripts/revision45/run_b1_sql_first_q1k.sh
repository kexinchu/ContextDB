#!/usr/bin/env bash
# B1 screen: 14 Amazon atoms, stock vs VisGuide vs SQL-first, q1K.
# Not paper-eligible. Default is dry-run.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-$repo_root}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-/home/kec23008/miniconda3/bin/python3}
lock_path="$results/.pg55437_experiment.lock"
runner="$script_dir/run_b1_sql_first_q1k.py"
out_dir="$results/revision45/b1_sql_first_q1k"
start_script="$script_dir/../start_amazon_table10_r43.sh"

execute=0
extra=()
for arg in "$@"; do
  if [[ $arg == --execute ]]; then
    execute=1
  else
    extra+=("$arg")
  fi
done

if [[ $execute -eq 0 ]]; then
  "$python" "$runner" "${extra[@]}"
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
export PYTHONPATH="$script_dir/..${PYTHONPATH:+:$PYTHONPATH}"

exec 9>>"$lock_path"
flock 9
{
  echo "REV45_B1_EXECUTE:$(date -Is)"
  "$python" "$runner" --execute --out-dir "$out_dir" "${extra[@]}"
  echo "REV45_B1_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$out_dir/run.log"
