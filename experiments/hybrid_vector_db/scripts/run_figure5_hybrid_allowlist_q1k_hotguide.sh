#!/usr/bin/env bash
# Same Figure 5 story (grocery_helpful, 4 SQL shapes, ef=100).
# SQLens: warmup + keep guidance hot; reuse D3 fragments across JOIN shapes.
# Does not overwrite frozen q1K or retune directories.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shared_root="${TABLE10_SHARED_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-python3}
lock_path="$results/.pg55437_experiment.lock"
runner="$script_dir/figure5_hybrid_allowlist_screen.py"
start_script="$script_dir/start_amazon_table10_r43.sh"
frozen="$results/figure5_hybrid_allowlist_q1k_screen_frozen_20260820"
out_dir="$results/figure5_hybrid_allowlist_q1k_hotguide"
log="$out_dir/screen.log"

if [[ ! -f $frozen/score.json ]]; then
  echo "frozen q1K screen is missing: $frozen" >&2
  exit 2
fi

mkdir -p "$out_dir"
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

exec 9>>"$lock_path"
flock 9
{
  echo "FIG5_ALLOWLIST_HOTGUIDE_EXECUTE:$(date -Is)"
  "$python" "$runner" \
    --execute \
    --hot-guidance \
    --reuse-faiss-from "$frozen" \
    --out-dir "$out_dir"
  echo "FIG5_ALLOWLIST_HOTGUIDE_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$log"
