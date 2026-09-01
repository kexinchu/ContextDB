#!/usr/bin/env bash
# E2 FragReuse compose on the r44 Amazon replica. Port 55440. Does not touch 55437.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
results="$repo_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-python3}
lock_path="$results/.pg55440_experiment.lock"
runner="$script_dir/run_e2_fragreuse_compose.py"
out_dir="$results/revision45/e2_fragreuse_compose_r44"
start_script="$script_dir/start_r44_amazon.sh"

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
export PGPORT=55440
export PGDATABASE=hybrid_vector
export PGUSER=postgres
: "${PGPASSWORD:?set PGPASSWORD}"
export PYTHONPATH="$script_dir/..${PYTHONPATH:+:$PYTHONPATH}"

if ! flock -n "$lock_path" -c true; then
  echo "r44 Amazon lock is already held: $lock_path" >&2
  exit 2
fi

exec 9>>"$lock_path"
flock 9
{
  echo "REV45_E2_R44_EXECUTE:$(date -Is)"
  "$python" "$runner" --execute --out-dir "$out_dir" "${extra[@]}"
  echo "REV45_E2_R44_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$out_dir/run.log"
