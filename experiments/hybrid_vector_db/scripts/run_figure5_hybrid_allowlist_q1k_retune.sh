#!/usr/bin/env bash
# Overnight retune: independent ef for stock vs SQLens (d1 and d1_d2_d3),
# grocery_helpful + grocery_long500. Reuses frozen FAISS allow-list numbers.
# Does not overwrite figure5_hybrid_allowlist_q1k_screen_frozen_20260820.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shared_root="${TABLE10_SHARED_ROOT:-/home/kec23008/Hybrid-Retrieval}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-/home/kec23008/miniconda3/bin/python3}
lock_path="$results/.pg55437_experiment.lock"
runner="$script_dir/figure5_hybrid_allowlist_screen.py"
start_script="$script_dir/start_amazon_table10_r43.sh"
frozen="$results/figure5_hybrid_allowlist_q1k_screen_frozen_20260820"
out_dir="$results/figure5_hybrid_allowlist_q1k_retune"
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
export PGPASSWORD=postgres
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

exec 9>>"$lock_path"
flock 9
{
  echo "FIG5_ALLOWLIST_RETUNE_EXECUTE:$(date -Is)"
  "$python" "$runner" \
    --execute \
    --retune \
    --reuse-faiss-from "$frozen" \
    --out-dir "$out_dir"
  echo "FIG5_ALLOWLIST_RETUNE_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$log"
