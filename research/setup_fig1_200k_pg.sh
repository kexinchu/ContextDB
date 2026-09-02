#!/usr/bin/env bash
# Dedicated PostgreSQL for Figure-1 200k engine-gap experiments.
# Does not touch :55437 (Amazon SQL-native Table-10).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container=${FIG1_CONTAINER:-hybrid-pgvector-fig1-200k}
image=${FIG1_IMAGE:-pgvector/pgvector:pg16}
vector_so=${FIG1_VECTOR_SO:-$ROOT/results/hybrid_vector_db/release_binaries/r43/vector.so}
host_port=${FIG1_HOST_PORT:-55438}
pgdata=${FIG1_PGDATA:-$ROOT/.pgdata-fig1-200k}

if [[ ! -f $vector_so ]]; then
  echo "missing sqlens vector.so: $vector_so" >&2
  exit 2
fi

mkdir -p "$pgdata"

if docker inspect "$container" >/dev/null 2>&1; then
  if [[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]]; then
    echo "container already running: $container"
  else
    docker start "$container" >/dev/null
    echo "started existing container: $container"
  fi
else
  docker run -d \
    --name "$container" \
    --shm-size 16g \
    -p "$host_port:5432" \
    -v "$pgdata:/var/lib/postgresql/data" \
    -v "$vector_so:/usr/lib/postgresql/16/lib/vector.so:ro" \
    -e POSTGRES_PASSWORD="${PGPASSWORD:?set PGPASSWORD}" \
    -e POSTGRES_DB=hybrid_vector \
    "$image" \
    -c shared_buffers=4GB \
    -c effective_cache_size=12GB \
    -c maintenance_work_mem=2GB \
    -c max_wal_size=4GB
fi

echo "waiting for PostgreSQL on port $host_port"
for _ in $(seq 1 90); do
  if docker exec "$container" pg_isready -U postgres >/dev/null 2>&1; then
    echo "PostgreSQL is ready on $host_port"
    exit 0
  fi
  sleep 2
done
echo "PostgreSQL did not become ready" >&2
exit 1
