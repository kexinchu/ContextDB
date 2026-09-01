#!/usr/bin/env bash
# Sequential overnight: P1 Figure 5 q10K, P2 FAISS-14 q1K, P3 QPS-16, P4 memory already in P1.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../../.." && pwd)
results="$root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-python3}
lock_path="$results/.pg55437_experiment.lock"
start_script="$script_dir/start_amazon_table10_r43.sh"
frozen="$results/figure5_hybrid_allowlist_q1k_screen_frozen_20260820"
p1_dir="$results/figure5_hybrid_allowlist_q10k_formal"
p2_dir="$results/rowlocal_faiss14_q1k_screen"
p3_dir="$results/figure5_qps16_readonly"
log="$results/overnight_p1_p4.log"

mkdir -p "$p1_dir" "$p2_dir" "$p3_dir"
"$start_script"
if ! flock -n "$lock_path" -c true; then
  echo "lock held: $lock_path" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1 PGHOST=127.0.0.1 PGPORT=55437 PGDATABASE=hybrid_vector
: "${PGUSER:?set PGUSER}" "${PGPASSWORD:?set PGPASSWORD}"
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

exec 9>>"$lock_path"
flock 9
{
  echo "OVERNIGHT_P1P4_START:$(date -Is)"
  echo "=== P1 formal q10K ==="
  "$python" "$script_dir/figure5_hybrid_allowlist_screen.py" \
    --execute --formal --query-offset 200 --query-count 10000 \
    --reuse-faiss-from "$frozen" \
    --out-dir "$p1_dir"
  echo "P1_EXIT:$? $(date -Is)"
  echo "=== P2 FAISS 14-filter q1K ==="
  "$python" "$script_dir/rowlocal_faiss14_screen.py" --execute --out-dir "$p2_dir"
  echo "P2_EXIT:$? $(date -Is)"
  echo "=== P3 16-client read-only QPS ==="
  "$python" "$script_dir/figure5_qps16_readonly.py" --execute --clients 16 --seconds 90 --out-dir "$p3_dir"
  echo "P3_EXIT:$? $(date -Is)"
  echo "OVERNIGHT_P1P4_DONE:$(date -Is)"
} 2>&1 | stdbuf -oL -eL tee -a "$log"
