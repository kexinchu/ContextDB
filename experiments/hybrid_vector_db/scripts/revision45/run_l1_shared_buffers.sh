#!/usr/bin/env bash
# Layer 1 cache panel: 64GB (live) then 128MB then 8GB, restore 64GB.
# Recreates the r43 container; PGDATA and vector.so stay put.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
results="$repo_root/results/hybrid_vector_db"
python=${TABLE10_PYTHON:-python3}
lock_path="$results/.pg55437_experiment.lock"
container=${TABLE10_AMAZON_CONTAINER:-hybrid-pgvector-amazon-table10-r43}
image=${TABLE10_IMAGE:-pgvector/pgvector:pg16}
pgdata=${TABLE10_PGDATA:?set TABLE10_PGDATA}
vector_so=${TABLE10_VECTOR_SO:-$results/release_binaries/r43/vector.so}
expected_sha=2056a67b9b0012c401c6684d49915cbc31bc8fa770946dbfaddda9d779eecbf2
expected_build=sqlens-v17-predistance-promotion-20260806-r43
out_root="$results/revision45/l1_shared_buffers"
log="$out_root/run.log"

if [[ ${1:-} != --execute ]]; then
  echo '{"dry_run": true, "plan_item": "L1_SHARED_BUFFERS", "rewrites_published": false}'
  "$python" "$script_dir/run_l1_shared_buffers.py" --shared-buffers 64GB
  exit 0
fi

mkdir -p "$out_root"
export PYTHONUNBUFFERED=1 PGHOST=127.0.0.1 PGPORT=55437 PGDATABASE=hybrid_vector
: "${PGUSER:?set PGUSER}" "${PGPASSWORD:?set PGPASSWORD}"
export PYTHONPATH="$(dirname -- "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"

wait_ready() {
  local i
  for i in $(seq 1 120); do
    if docker exec "$container" pg_isready -U postgres >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "PostgreSQL did not become ready after recreate" >&2
  docker logs --tail 40 "$container" >&2 || true
  return 1
}

check_identity() {
  local live_sha live_build
  live_sha=$(docker exec "$container" sh -lc \
    'sha256sum "$(pg_config --pkglibdir)/vector.so" | cut -d" " -f1')
  if [[ $live_sha != "$expected_sha" ]]; then
    echo "vector.so SHA mismatch: $live_sha" >&2
    return 1
  fi
  live_build=$(docker exec "$container" psql -U postgres -d hybrid_vector -Atc \
    "SELECT vector_sqlens_build_id()")
  if [[ $live_build != "$expected_build" ]]; then
    echo "build id mismatch: $live_build" >&2
    return 1
  fi
}

recreate() {
  local sb=$1 ecs=$2
  echo "RECREATE shared_buffers=$sb effective_cache_size=$ecs"
  docker stop "$container" >/dev/null
  docker rm "$container" >/dev/null
  docker run -d \
    --name "$container" \
    --cpuset-cpus 0-31 \
    --shm-size 320g \
    -p 55437:5432 \
    -v "$pgdata:/var/lib/postgresql/data" \
    -v "$vector_so:/usr/lib/postgresql/16/lib/vector.so:ro" \
    -e POSTGRES_PASSWORD="${PGPASSWORD:?set PGPASSWORD}" \
    -e POSTGRES_DB=hybrid_vector \
    "$image" \
    -c max_worker_processes=32 \
    -c max_parallel_workers=16 \
    -c max_parallel_maintenance_workers=15 \
    -c "shared_buffers=$sb" \
    -c "effective_cache_size=$ecs" \
    -c maintenance_work_mem=16GB \
    -c max_wal_size=8GB \
    -c checkpoint_timeout=15min \
    -c wal_compression=on >/dev/null
  wait_ready
  check_identity
  docker exec "$container" psql -U postgres -d hybrid_vector -c "SHOW shared_buffers;"
}

restore_64g() {
  echo "RESTORE shared_buffers=64GB"
  recreate 64GB 192GB || true
}

run_one() {
  local sb=$1
  echo "=== MEASURE $sb ==="
  "$python" "$script_dir/run_l1_shared_buffers.py" \
    --execute --shared-buffers "$sb" \
    --out-dir "$out_root/$sb"
}

exec 9>>"$lock_path"
flock 9
trap restore_64g EXIT
{
  echo "L1_SHARED_BUFFERS_START:$(date -Is)"
  echo "=== MEASURE live 64GB (no recreate) ==="
  run_one 64GB
  recreate 128MB 4GB
  run_one 128MB
  recreate 8GB 24GB
  run_one 8GB
  echo "L1_SHARED_BUFFERS_DONE:$(date -Is)"
} 2>&1 | stdbuf -oL -eL tee -a "$log"
