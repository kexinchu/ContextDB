#!/usr/bin/env bash
# Amazon-10M replica for FragReuse r44. Port 55440. Does not touch 55437.
set -euo pipefail

repo=/home/kec23008/Hybrid-Retrieval
container=${R44_AMAZON_CONTAINER:-sqlens-r44-amazon}
image=${TABLE10_IMAGE:-pgvector/pgvector:pg16}
pgdata=${R44_PGDATA:-/mnt/nvme-pg/home/kec23008/pgdata-amazon-table10-r44}
so=${R44_VECTOR_SO:-$repo/results/hybrid_vector_db/release_binaries/r44/vector.so}
src=$repo/third_party/pgvector-sqlens-r44
host_port=${R44_HOST_PORT:-55440}
cpuset=${R44_CPUSET:-16-47}
expected_sha=${R44_EXPECTED_SHA:-513c5ea4f0279feb45ef4af1931185b97fbb90c48c87383620dd4eb417f854b4}
expected_build=${R44_EXPECTED_BUILD:-sqlens-v19-fragreuse-admit-20260830}
shared_buffers=${R44_SHARED_BUFFERS:-64GB}
effective_cache=${R44_EFFECTIVE_CACHE:-192GB}
recreate=${R44_RECREATE:-0}

test -f "$so"
observed_sha=$(sha256sum "$so" | awk '{print $1}')
if [[ $observed_sha != "$expected_sha" ]]; then
  echo "r44 vector.so SHA mismatch: $observed_sha != $expected_sha" >&2
  exit 2
fi
if ! sudo -n test -f "$pgdata/PG_VERSION"; then
  echo "r44 PGDATA is missing: $pgdata" >&2
  exit 2
fi
pg_major=$(sudo -n cat "$pgdata/PG_VERSION")
if [[ $pg_major != 16 ]]; then
  echo "PGDATA major $pg_major != 16" >&2
  exit 2
fi

if [[ $recreate == 1 ]] && docker inspect "$container" >/dev/null 2>&1; then
  echo "recreating $container shared_buffers=$shared_buffers"
  docker stop "$container" >/dev/null
  docker rm "$container" >/dev/null
fi

if docker inspect "$container" >/dev/null 2>&1; then
  if [[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]]; then
    echo "container already running: $container"
  else
    docker start "$container" >/dev/null
    echo "started existing container: $container"
  fi
else
  if sudo -n test -f "$pgdata/postmaster.pid"; then
    echo "removing leftover postmaster.pid from the replica"
    sudo -n rm -f "$pgdata/postmaster.pid"
  fi
  docker run -d \
    --name "$container" \
    --cpuset-cpus "$cpuset" \
    --shm-size 320g \
    -p "$host_port:5432" \
    -v "$pgdata:/var/lib/postgresql/data" \
    -v "$so:/usr/lib/postgresql/16/lib/vector.so:ro" \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=hybrid_vector \
    "$image" \
    -c max_worker_processes=32 \
    -c max_parallel_workers=16 \
    -c max_parallel_maintenance_workers=15 \
    -c "shared_buffers=$shared_buffers" \
    -c "effective_cache_size=$effective_cache" \
    -c work_mem=4MB \
    -c maintenance_work_mem=16GB \
    -c max_wal_size=8GB \
    -c checkpoint_timeout=15min \
    -c wal_compression=on
fi

echo "waiting for PostgreSQL on port $host_port"
for _ in $(seq 1 180); do
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

docker cp "$src/vector.control" "$container:/usr/share/postgresql/16/extension/vector.control"
docker cp "$src/sql/vector--0.8.2.sql" "$container:/usr/share/postgresql/16/extension/vector--0.8.2.sql"

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
echo "ready:$container:$live_build:$host_port"
