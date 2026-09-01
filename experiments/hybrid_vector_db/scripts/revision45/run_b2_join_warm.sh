#!/usr/bin/env bash
# B2 screen: Figure 5 four SQL shapes with a resident grocery_helpful fragment.
# Not paper-eligible. Does not overwrite the frozen empty-start cells.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-$repo_root}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-python3}
lock_path="$results/.pg55437_experiment.lock"
runner="$script_dir/../figure5_hybrid_allowlist_screen.py"
start_script="$script_dir/../start_amazon_table10_r43.sh"
frozen="$results/figure5_hybrid_allowlist_q1k_screen_frozen_20260820"
out_dir="$results/revision45/b2_join_warm"

if [[ ${1:-} != --execute ]]; then
  echo "{\"dry_run\": true, \"plan_item\": \"B2\", \"paper_eligible\": false, \"out_dir\": \"$out_dir\", \"hot_guidance\": true}"
  exit 0
fi

if [[ ! -f $frozen/score.json ]]; then
  echo "frozen q1K screen is missing: $frozen" >&2
  exit 2
fi

mkdir -p "$out_dir"
"$start_script"
export PYTHONUNBUFFERED=1
export PGHOST=127.0.0.1
export PGPORT=55437
export PGDATABASE=hybrid_vector
export PGUSER=postgres
: "${PGPASSWORD:?set PGPASSWORD}"
export PYTHONPATH="$(dirname -- "$runner")${PYTHONPATH:+:$PYTHONPATH}"

echo '{"paper_eligible": false, "plan_item": "B2"}' > "$out_dir/manifest.json"

exec 9>>"$lock_path"
flock 9
{
  echo "REV45_B2_EXECUTE:$(date -Is)"
  "$python" "$runner" \
    --execute \
    --hot-guidance \
    --reuse-faiss-from "$frozen" \
    --out-dir "$out_dir"
  echo "REV45_B2_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$out_dir/run.log"
