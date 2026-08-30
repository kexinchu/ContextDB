#!/usr/bin/env bash
# Service-path e2e on the r44 Amazon replica. Port 55440. Does not touch 55437.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
results="$repo_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-/home/kec23008/miniconda3/bin/python3}
lock_path="$results/.pg55440_experiment.lock"
runner="$script_dir/run_service_path_e2e.py"
out_dir="$results/revision45/service_path_e2e"
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
export R44_SHARED_BUFFERS=${R44_SHARED_BUFFERS:-128MB}
export R44_EFFECTIVE_CACHE=${R44_EFFECTIVE_CACHE:-4GB}
live_sb=""
if docker inspect sqlens-r44-amazon >/dev/null 2>&1 && docker exec sqlens-r44-amazon pg_isready -U postgres >/dev/null 2>&1; then
  live_sb=$(docker exec sqlens-r44-amazon psql -U postgres -d hybrid_vector -Atc "SHOW shared_buffers")
fi
if [[ $live_sb != "$R44_SHARED_BUFFERS" ]]; then
  export R44_RECREATE=1
fi
"$start_script"
export PYTHONUNBUFFERED=1
export PGHOST=127.0.0.1
export PGPORT=55440
export PGDATABASE=hybrid_vector
export PGUSER=postgres
export PGPASSWORD=postgres
export PYTHONPATH="$script_dir/..${PYTHONPATH:+:$PYTHONPATH}"

if ! flock -n "$lock_path" -c true; then
  echo "r44 Amazon lock is already held: $lock_path" >&2
  exit 2
fi

exec 9>>"$lock_path"
flock 9
{
  echo "REV45_SERVICE_PATH_E2E:$(date -Is)"
  "$python" "$runner" --execute --out-dir "$out_dir" "${extra[@]}"
  echo "REV45_SERVICE_PATH_E2E_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$out_dir/run.log"
