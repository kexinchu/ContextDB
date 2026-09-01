#!/usr/bin/env bash
# C1 screen: stock vs safe_guided vs pgvector acorn1 on Amazon-14, q1K, ef=100.
# Not paper-eligible. Uses the existing selectivity runner; two strategy passes.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-$repo_root}"
results="$shared_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-python3}
lock_path="$results/.pg55437_experiment.lock"
runner="$script_dir/../pgvector_design1_design2_design3_selectivity_benchmark.py"
start_script="$script_dir/../start_amazon_table10_r43.sh"
out_dir="$results/revision45/c1_acorn_amazon4"
filters="$repo_root/experiments/hybrid_vector_db/configs/amazon10m_selectivity14_valid_embeddings_filters.csv"
truth="$results/amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal.csv"
expected_build=${TABLE10_EXPECTED_BUILD:-sqlens-v17-predistance-promotion-20260806-r43}
expected_sha=${TABLE10_EXPECTED_SHA:-2056a67b9b0012c401c6684d49915cbc31bc8fa770946dbfaddda9d779eecbf2}
queries=${C1_QUERIES:-50}
filter_names=${C1_FILTERS:-grocery_long500,helpful_ge20,grocery_helpful,long_review_ge500}

common=(
  --filters-csv "$filters"
  --truth-csv "$truth"
  --modes original design1_bloom
  --filter-names ${filter_names//,/ }
  --queries "$queries"
  --query-offset 200
  --repeats 1
  --k 10
  --ef-search 100
  --insertion-table public.amazon_grocery_reviews_10m_pgvector
  --insertion-index public.amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx
  --bfs-table public.amazon_grocery_reviews_10m_pgvector
  --bfs-index public.amazon10m_hnsw_m32ef200_dupbridge_r29_bfs_idx
  --candidate-validity-predicate embedding_valid
  --expected-sqlens-build-id "$expected_build"
  --expected-vector-so-sha256 "$expected_sha"
  --no-database-experiment-lock
)

if [[ ${1:-} != --execute ]]; then
  echo "{\"dry_run\": true, \"plan_item\": \"C1\", \"paper_eligible\": false, \"out_dir\": \"$out_dir\", \"passes\": [\"safe_guided\", \"acorn1\"], \"queries\": $queries, \"filters\": \"$filter_names\", \"ef_search\": 100}"
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
export PYTHONPATH="$(dirname -- "$runner")${PYTHONPATH:+:$PYTHONPATH}"

echo '{"paper_eligible": false, "plan_item": "C1"}' > "$out_dir/manifest.json"

exec 9>>"$lock_path"
flock 9
{
  echo "REV45_C1_EXECUTE:$(date -Is)"
  "$python" "$runner" "${common[@]}" \
    --guidance-filter-strategy safe_guided \
    --out "$out_dir/safe_guided.json"
  "$python" "$runner" "${common[@]}" \
    --guidance-filter-strategy acorn1 \
    --out "$out_dir/acorn1.json"
  echo "REV45_C1_EXIT:$? $(date -Is)"
} 2>&1 | tee -a "$out_dir/run.log"
