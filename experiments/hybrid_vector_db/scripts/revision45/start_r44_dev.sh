#!/usr/bin/env bash
# Side copy for FragReuse r44. Port 55439. Does not touch 55437.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
so=$repo/results/hybrid_vector_db/release_binaries/r44/vector.so
src=$repo/third_party/pgvector-sqlens-r44
container=sqlens-r44-dev
image=${TABLE10_IMAGE:-pgvector/pgvector:pg16}

test -f "$so"
docker rm -f "$container" >/dev/null 2>&1 || true
docker run -d \
  --name "$container" \
  -p 55439:5432 \
  -v "$so:/usr/lib/postgresql/16/lib/vector.so:ro" \
  -e POSTGRES_PASSWORD="${PGPASSWORD:?set PGPASSWORD}" \
  -e POSTGRES_DB=hybrid_vector \
  "$image" >/dev/null
for _ in $(seq 1 60); do
  docker exec "$container" pg_isready -U postgres >/dev/null 2>&1 && break
  sleep 1
done
docker cp "$src/vector.control" "$container:/usr/share/postgresql/16/extension/vector.control"
docker cp "$src/sql/vector--0.8.2.sql" "$container:/usr/share/postgresql/16/extension/vector--0.8.2.sql"
docker exec "$container" psql -U postgres -d hybrid_vector -v ON_ERROR_STOP=1 -c \
  "CREATE EXTENSION IF NOT EXISTS vector VERSION '0.8.2'; SELECT vector_sqlens_build_id();"
echo "r44-dev ready on 127.0.0.1:55439"
