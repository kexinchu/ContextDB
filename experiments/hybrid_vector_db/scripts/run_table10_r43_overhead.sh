#!/usr/bin/env bash
# Table 10 Panel C overhead measurement under r43.
# Dry-run by default; pass EXECUTE=1 to touch the Amazon Table-10 container.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
shared_root="${TABLE10_SHARED_ROOT:-/home/kec23008/Hybrid-Retrieval}"
results="$shared_root/results/hybrid_vector_db/table10_r43"
python=${TABLE10_PYTHON:-/home/kec23008/miniconda3/bin/python3}
out_json="$results/overhead/table10_r43_overhead.json"
lifecycle_csv="${TABLE10_LIFECYCLE_CSV:-$results/concurrency/table10_r43_amazon_concurrency_lifecycle.csv}"
build_proof="${TABLE10_BUILD_PROOF_JSON:-}"
contract="$repo_root/experiments/hybrid_vector_db/configs/p0_release_contract_r43.json"
execute=${EXECUTE:-0}

mkdir -p "$(dirname -- "$out_json")"

args=(
  "$python"
  "$repo_root/experiments/hybrid_vector_db/scripts/measure_table10_r43_overhead.py"
  --release-contract "$contract"
  --expected-build-id sqlens-v17-predistance-promotion-20260806-r43
  --out-json "$out_json"
)

if [[ "$execute" != "1" ]]; then
  args+=(--dry-run)
  echo "[table10-overhead] dry-run skeleton -> $out_json"
  "${args[@]}"
  exit 0
fi

args+=(--measure-memory --measure-storage)
if [[ -n "$build_proof" && -f "$build_proof" ]]; then
  args+=(--build-proof-json "$build_proof")
fi
if [[ -f "$lifecycle_csv" ]]; then
  args+=(--lifecycle-csv "$lifecycle_csv")
fi
if [[ "${MARK_PAPER_ELIGIBLE:-0}" == "1" ]]; then
  args+=(--mark-paper-eligible)
fi

export PGHOST=127.0.0.1 PGPORT=55437 PGDATABASE=hybrid_vector PGUSER=postgres PGPASSWORD=postgres
echo "[table10-overhead] execute -> $out_json"
"${args[@]}"
