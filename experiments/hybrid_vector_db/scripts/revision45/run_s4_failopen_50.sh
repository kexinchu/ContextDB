#!/usr/bin/env bash
# S4: fail-open delivery at 50 upd/s only. Does not rewrite 10/25 cells.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export C2_UPDATE_RATES="${C2_UPDATE_RATES:-50}"
export C2_READERS="${C2_READERS:-16}"
export C2_REQUESTS="${C2_REQUESTS:-10000}"
export C2_REPEATS="${C2_REPEATS:-6}"

# Reuse the C2 runner with a disjoint artifact directory.
results="${TABLE10_SHARED_ROOT:-$(cd -- "$script_dir/../../../.." && pwd)}/results/hybrid_vector_db"
# The shared script hard-codes c2_failopen_write_sweep/. Patch via OUT override
# by invoking the same body with a copied out path.
# We call the python runner through the existing script after swapping out via
# a one-line wrapper below.

repo_root=$(cd -- "$script_dir/../../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-$repo_root}"
python=${TABLE10_PYTHON:-python3}
lock_path="$results/.pg55437_experiment.lock"
runner="$script_dir/../pgvector_update_concurrency_benchmark.py"
start_script="$script_dir/../start_amazon_table10_r43.sh"
selector_root="$results/table10_r43/selector_formal"
out="$results/revision45/s4_failopen_50/concurrency.json"
container=${TABLE10_AMAZON_CONTAINER:-hybrid-pgvector-amazon-table10-r43}

expected_build=sqlens-v17-predistance-promotion-20260806-r43
expected_sha=2056a67b9b0012c401c6684d49915cbc31bc8fa770946dbfaddda9d779eecbf2

common=(
  --protocol p0_6_full
  --paper-table-slice
  --fail-open-stale
  --expected-sqlens-build-id "$expected_build"
  --expected-vector-so-sha256 "$expected_sha"
  --fixed-recall-selector-csv "$selector_root/table10_r43_fixed_target_configs.csv"
  --fixed-recall-selector-manifest "$selector_root/table10_r43_fixed_target_configs.manifest.json"
  --fixed-selector-workload-csv "$results/figure5_r37_formal_workloads/figure5_r37_amazon_calibration.csv"
  --filters-csv "$repo_root/experiments/hybrid_vector_db/configs/amazon10m_selectivity14_valid_embeddings_filters.csv"
  --calibration-truth-csv "$results/amazon_selectivity14_exact_truth_q200_unique_embeddings_formal.csv"
  --calibration-truth-manifest "$results/amazon_selectivity14_exact_truth_q200_unique_embeddings_formal_manifest.json"
  --measurement-query-file "$results/amazon10m_unique_embedding_query_cohort_q10200.csv"
  --measurement-query-manifest "$results/amazon10m_unique_embedding_query_cohort_q10200_manifest.json"
  --measurement-truth-csv "$results/amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal.csv"
  --measurement-truth-manifest "$results/amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal_manifest.json"
  --insertion-table public.amazon_grocery_reviews_10m_pgvector
  --insertion-index public.amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx
  --bfs-table public.amazon_grocery_reviews_10m_pgvector
  --bfs-index public.amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx
  --candidate-validity-predicate embedding_valid
  --expected-candidate-rows 9979556
  --target-recalls 0.90
  --readers "$C2_READERS"
  --update-rates "$C2_UPDATE_RATES"
  --methods stock,sqlens_full
  --requests "$C2_REQUESTS"
  --measurement-repeats "$C2_REPEATS"
  --writer-clients 1
  --update-batch-size 1
  --update-id-pool-size 100000
  --mutation-mix predicate:4,vector:4,insert:1,delete:1
  --min-update-delivery-ratio 0.90
)

if [[ ${1:-} != --execute ]]; then
  echo "{\"dry_run\": true, \"plan_item\": \"S4\", \"update_rates\": \"$C2_UPDATE_RATES\", \"out\": \"$out\"}"
  exit 0
fi

mkdir -p "$(dirname -- "$out")"
"$start_script"
echo '{"paper_eligible": false, "plan_item": "S4", "update_rates": [50]}' > "$(dirname -- "$out")/manifest.json"

container_pid=$(docker inspect -f '{{.State.Pid}}' "$container")
backend_proc_root="/proc/$container_pid/root/proc"
runner_sha=$(sha256sum "$runner" | cut -d' ' -f1)
git_revision=$(git -C "$repo_root" rev-parse HEAD)

export PYTHONUNBUFFERED=1
export PGHOST=127.0.0.1
export PGPORT=55437
export PGDATABASE=hybrid_vector
export PGUSER=postgres
: "${PGPASSWORD:?set PGPASSWORD}"

exec 9>>"$lock_path"
flock 9
{
  echo "REV45_S4_EXECUTE:$(date -Is)"
  "$python" "$runner" "${common[@]}" \
    --backend-cpu-list "${TABLE10_BACKEND_CPU:-0-15}" \
    --client-cpu-list "${TABLE10_CLIENT_CPU:-16-47}" \
    --start-barrier-timeout-seconds "${TABLE10_BARRIER_TIMEOUT:-600}" \
    --backend-proc-root "$backend_proc_root" \
    --telemetry-path "${TABLE10_PGDATA:?set TABLE10_PGDATA}" \
    --expected-runner-sha256 "$runner_sha" \
    --expected-git-revision "$git_revision" \
    --out "$out" \
    --resume \
    --execute
  echo "REV45_S4_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$(dirname -- "$out")/run.log"
