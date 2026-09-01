#!/usr/bin/env bash
# Restore the Amazon Table-10 r43 PostgreSQL instance from existing PGDATA
# plus the archived r43 vector.so. Official pgvector:pg16 supplies PG 16;
# we overlay the frozen SQLens binary (SHA 2056a67b...).
set -euo pipefail

container=${TABLE10_AMAZON_CONTAINER:-hybrid-pgvector-amazon-table10-r43}
image=${TABLE10_IMAGE:-pgvector/pgvector:pg16}
pgdata=${TABLE10_PGDATA:?set TABLE10_PGDATA}
vector_so=${TABLE10_VECTOR_SO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/results/hybrid_vector_db/release_binaries/r43/vector.so}
host_port=${TABLE10_HOST_PORT:-55437}
cpuset=${TABLE10_CPUSET:-0-31}
shm_size=${TABLE10_SHM_SIZE:-320g}
expected_sha=${TABLE10_EXPECTED_SHA:-2056a67b9b0012c401c6684d49915cbc31bc8fa770946dbfaddda9d779eecbf2}
expected_build=${TABLE10_EXPECTED_BUILD:-sqlens-v17-predistance-promotion-20260806-r43}

if [[ ! -f $vector_so ]]; then
  echo "r43 vector.so missing: $vector_so" >&2
  exit 2
fi
observed_sha=$(sha256sum "$vector_so" | awk '{print $1}')
if [[ $observed_sha != "$expected_sha" ]]; then
  echo "r43 vector.so SHA mismatch: $observed_sha != $expected_sha" >&2
  exit 2
fi
if ! sudo -n test -f "$pgdata/PG_VERSION"; then
  echo "PGDATA is missing or unreadable: $pgdata" >&2
  exit 2
fi
pg_major=$(sudo -n cat "$pgdata/PG_VERSION")
if [[ $pg_major != 16 ]]; then
  echo "PGDATA major $pg_major != 16" >&2
  exit 2
fi

if docker inspect "$container" >/dev/null 2>&1; then
  if [[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]]; then
    echo "container already running: $container"
  else
    docker start "$container" >/dev/null
    echo "started existing container: $container"
  fi
else
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    docker pull "$image"
  fi
  if sudo -n test -f "$pgdata/postmaster.pid"; then
    echo "removing stale postmaster.pid"
    sudo -n rm -f "$pgdata/postmaster.pid"
  fi
  docker run -d \
    --name "$container" \
    --cpuset-cpus "$cpuset" \
    --shm-size "$shm_size" \
    -p "$host_port:5432" \
    -v "$pgdata:/var/lib/postgresql/data" \
    -v "$vector_so:/usr/lib/postgresql/16/lib/vector.so:ro" \
    -e POSTGRES_PASSWORD="${PGPASSWORD:?set PGPASSWORD}" \
    -e POSTGRES_DB=hybrid_vector \
    "$image" \
    -c max_worker_processes=32 \
    -c max_parallel_workers=16 \
    -c max_parallel_maintenance_workers=15 \
    -c shared_buffers=64GB \
    -c effective_cache_size=192GB \
    -c maintenance_work_mem=16GB \
    -c max_wal_size=8GB \
    -c checkpoint_timeout=15min \
    -c wal_compression=on
fi

echo "waiting for PostgreSQL on port $host_port"
for _ in $(seq 1 120); do
  if docker exec "$container" pg_isready -U postgres >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! docker exec "$container" pg_isready -U postgres >/dev/null 2>&1; then
  echo "PostgreSQL did not become ready" >&2
  docker logs --tail 80 "$container" >&2 || true
  exit 1
fi

live_sha=$(docker exec "$container" sh -lc \
  'sha256sum "$(pg_config --pkglibdir)/vector.so" | cut -d" " -f1')
if [[ $live_sha != "$expected_sha" ]]; then
  echo "live vector.so SHA mismatch: $live_sha != $expected_sha" >&2
  exit 1
fi
live_build=$(docker exec "$container" psql -U postgres -d hybrid_vector -Atc \
  "SELECT vector_sqlens_build_id()")
if [[ $live_build != "$expected_build" ]]; then
  echo "live build id mismatch: $live_build != $expected_build" >&2
  exit 1
fi
echo "ready:$container:$live_build"
