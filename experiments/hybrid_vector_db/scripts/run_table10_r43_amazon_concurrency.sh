#!/usr/bin/env bash
# Table 10 Panel B: Amazon-10M read/write concurrency matrix under r43.
# Resumes the frozen p0_6_full grid; paper cells are 16r/{0,100,1000} and 64r/100.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
results="$shared_root/results/hybrid_vector_db"
container=${TABLE10_AMAZON_CONTAINER:-hybrid-pgvector-amazon-table10-r43}
python=${TABLE10_PYTHON:-python3}
lock_path="$results/.pg55437_experiment.lock"
runner="$repo_root/experiments/hybrid_vector_db/scripts/pgvector_update_concurrency_benchmark.py"
selector_root="$results/table10_r43/selector_formal"
out="$results/table10_r43/concurrency/table10_r43_amazon_concurrency.json"
contract="$repo_root/experiments/hybrid_vector_db/configs/p0_release_contract_r43.json"

expected_build=sqlens-v17-predistance-promotion-20260806-r43
expected_sha=2056a67b9b0012c401c6684d49915cbc31bc8fa770946dbfaddda9d779eecbf2

for artifact in \
    "$selector_root/table10_r43_fixed_target_configs.csv" \
    "$selector_root/table10_r43_fixed_target_configs.manifest.json" \
    "$contract" \
    "$runner"; do
    if [[ ! -f $artifact ]]; then
        echo "Table 10 prerequisite missing: $artifact" >&2
        exit 2
    fi
done
if ! docker inspect "$container" >/dev/null 2>&1 \
    || [[ $(docker inspect -f '{{.State.Running}}' "$container") != true ]]; then
    echo "Table 10 Amazon container is unavailable or stopped: $container" >&2
    exit 2
fi
if ! flock -n "$lock_path" -c true; then
    echo "Table 10 Amazon database lock is already held: $lock_path" >&2
    exit 2
fi

admission=$(docker exec "$container" psql -U postgres -d hybrid_vector -At -c \
    "SELECT count(*) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND backend_type = 'client backend' AND state <> 'idle'; SELECT vector_sqlens_build_id();")
active_clients=$(sed -n '1p' <<<"$admission")
observed_build=$(sed -n '2p' <<<"$admission")
if [[ $active_clients != 0 || $observed_build != "$expected_build" ]]; then
    echo "Table 10 admission failed: active=$active_clients build=$observed_build" >&2
    exit 2
fi
observed_sha=$(docker exec "$container" sh -lc \
    'sha256sum "$(pg_config --pkglibdir)/vector.so" | cut -d" " -f1')
if [[ $observed_sha != "$expected_sha" ]]; then
    echo "Table 10 Amazon vector.so mismatch: $observed_sha" >&2
    exit 2
fi

container_pid=$(docker inspect -f '{{.State.Pid}}' "$container")
backend_proc_root="/proc/$container_pid/root/proc"
runner_sha=$(sha256sum "$runner" | cut -d' ' -f1)
git_revision=$(git -C "$repo_root" rev-parse HEAD)
mkdir -p "$(dirname -- "$out")"

# Optional paper-only subset for faster targeted fills.
readers=${TABLE10_READERS:-1,4,8,16,32,64}
update_rates=${TABLE10_UPDATE_RATES:-0,10,100,1000}

flock "$lock_path" env \
    PYTHONUNBUFFERED=1 \
    PGHOST=127.0.0.1 \
    PGPORT=55437 \
    PGDATABASE=hybrid_vector \
    PGUSER=postgres \
    PGPASSWORD="${PGPASSWORD:?set PGPASSWORD}" \
    "$python" "$runner" \
    --protocol p0_6_full \
    --release-contract "$contract" \
    --fixed-recall-selector-csv "$selector_root/table10_r43_fixed_target_configs.csv" \
    --fixed-recall-selector-manifest "$selector_root/table10_r43_fixed_target_configs.manifest.json" \
    --fixed-selector-workload-csv "$results/figure5_r37_formal_workloads/figure5_r37_amazon_calibration.csv" \
    --filters-csv "$repo_root/experiments/hybrid_vector_db/configs/amazon10m_selectivity14_valid_embeddings_filters.csv" \
    --calibration-truth-csv "$results/amazon_selectivity14_exact_truth_q200_unique_embeddings_formal.csv" \
    --calibration-truth-manifest "$results/amazon_selectivity14_exact_truth_q200_unique_embeddings_formal_manifest.json" \
    --measurement-query-file "$results/amazon10m_unique_embedding_query_cohort_q10200.csv" \
    --measurement-query-manifest "$results/amazon10m_unique_embedding_query_cohort_q10200_manifest.json" \
    --measurement-truth-csv "$results/amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal.csv" \
    --measurement-truth-manifest "$results/amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal_manifest.json" \
    --insertion-table public.amazon_grocery_reviews_10m_pgvector \
    --insertion-index public.amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx \
    --bfs-table public.amazon_grocery_reviews_10m_pgvector \
    --bfs-index public.amazon10m_hnsw_m32ef200_dupbridge_r29_bfs_idx \
    --candidate-validity-predicate embedding_valid \
    --expected-candidate-rows 9979556 \
    --target-recalls 0.90 \
    --readers "$readers" \
    --update-rates "$update_rates" \
    --methods stock,sqlens_full \
    --requests 10000 \
    --measurement-repeats 6 \
    --writer-clients 1 \
    --update-batch-size 1 \
    --update-id-pool-size 100000 \
    --mutation-mix predicate:4,vector:4,insert:1,delete:1 \
    --backend-cpu-list 0-31 \
    --client-cpu-list 32-63 \
    --backend-proc-root "$backend_proc_root" \
    --telemetry-path "${TABLE10_PGDATA:?set TABLE10_PGDATA}" \
    --expected-runner-sha256 "$runner_sha" \
    --expected-git-revision "$git_revision" \
    --out "$out" \
    --resume \
    --execute
