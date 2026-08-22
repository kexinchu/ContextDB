#!/usr/bin/env bash
# Exact GT for Figure 5 JOIN hybrid search: facts / catalog / ACL
# crossed with three join-side selectivities. Same q10200 query cohort.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shared_root="${TABLE10_SHARED_ROOT:-/home/kec23008/Hybrid-Retrieval}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-/home/kec23008/miniconda3/bin/python3}
lock_path="$results/.pg55437_experiment.lock"
producer="$script_dir/amazon10m_sql_native_exact_truth.py"
start_script="$script_dir/start_amazon_table10_r43.sh"
artifact="$results/amazon10m_sql_native_q10200_r43_sqlops_join"
log="$results/amazon10m_sql_native_q10200_r43_sqlops_join_gt.log"

mkdir -p "$artifact"
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

resume_args=()
if [[ -d $artifact/checkpoints ]]; then
  resume_args=(--resume)
fi

exec 9>>"$lock_path"
flock 9
{
  echo "FIG5_JOIN_GT_EXECUTE:$(date -Is) resume=${#resume_args[@]}"
  "$python" "$producer" \
    --protocol q10200 \
    --execute \
    "${resume_args[@]}" \
    --artifact-dir "$artifact" \
    --workload-names \
      join_facts \
      join_catalog \
      join_acl \
    --filter-names \
      grocery_helpful \
      helpful_ge20 \
      grocery_long500 \
    --faiss-threads 14
  echo "FIG5_JOIN_GT_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$log"
