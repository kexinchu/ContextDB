#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${PYTHON:-python3}"
storage_root="${SQLENS_CLONE_STORAGE_ROOT:-${repo_root}/.pgvector-data/laion25m-clone}"

SOURCE_CONTAINER=${SOURCE_CONTAINER:-hybrid-pgvector}
CLONE_CONTAINER=${CLONE_CONTAINER:-hybrid-pgvector-laion25m}
IMAGE=${IMAGE:-hybrid-pgvector-sqlens:pg16}
PGDATA_DIR=${PGDATA_DIR:-${storage_root}/pgdata}
HOST_PORT=${HOST_PORT:-55434}
CPUSET=${CPUSET:-32-47}
MAX_PARALLEL_MAINTENANCE_WORKERS=${MAX_PARALLEL_MAINTENANCE_WORKERS:-15}
SHM_SIZE=${SHM_SIZE:-320g}
DATABASE=${DATABASE:-hybrid_vector}
LOG_DIR=${LOG_DIR:-${storage_root}/logs}
PROOF=${PROOF:-${repo_root}/results/hybrid_vector_db/laion25m_clone_55434_source_bfs_graph_proof.json}
MIN_FREE_GIB=${MIN_FREE_GIB:-220}
RESUME_EXISTING=${RESUME_EXISTING:-0}
REUSE_PGDATA=${REUSE_PGDATA:-0}
EXPECTED_BUILD_ID=${EXPECTED_BUILD_ID:-sqlens-v16-distance-aware-route-budget-ef500k-20260801-r41}
EXPECTED_VECTOR_SHA=${EXPECTED_VECTOR_SHA:-8f53226d35cae28d4e1b6926b13b01fa01fd1f6720c5f57c96c7886905f5eaf0}

mkdir -p "$LOG_DIR" "$(dirname "$PROOF")"
STATUS="$LOG_DIR/status"
LOG="$LOG_DIR/clone.log"
exec > >(tee -a "$LOG") 2>&1

fail() {
  printf 'failed:%s\n' "$(date -Is)" >"$STATUS"
}
trap fail ERR

container_exists=0
if docker container inspect "$CLONE_CONTAINER" >/dev/null 2>&1; then
  container_exists=1
fi
if (( container_exists == 1 )) && [[ "$RESUME_EXISTING" != 1 ]]; then
  echo "clone container already exists: $CLONE_CONTAINER" >&2
  exit 2
fi
if (( container_exists == 0 )) && [[ "$REUSE_PGDATA" != 1 ]] && [[ -e "$PGDATA_DIR" ]] && [[ -n "$(sudo -n find "$PGDATA_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "clone PGDATA is not empty: $PGDATA_DIR" >&2
  exit 2
fi

free_kib=$(df --output=avail -k "$PGDATA_DIR" 2>/dev/null | tail -1 || df --output=avail -k "$(dirname "$PGDATA_DIR")" | tail -1)
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if (( free_kib < required_kib )); then
  echo "insufficient free space: need ${MIN_FREE_GIB}GiB, have $((free_kib / 1024 / 1024))GiB" >&2
  exit 2
fi

echo "starting:$CLONE_CONTAINER:$(date -Is)" >"$STATUS"
if (( container_exists == 0 )); then
  sudo -n mkdir -p "$PGDATA_DIR"
  docker run -d \
    --name "$CLONE_CONTAINER" \
    --cpuset-cpus "$CPUSET" \
    --shm-size "$SHM_SIZE" \
    -p "$HOST_PORT:5432" \
    -v "$PGDATA_DIR:/var/lib/postgresql/data" \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB="$DATABASE" \
    "$IMAGE" \
    -c max_worker_processes=32 \
    -c max_parallel_workers=16 \
    -c max_parallel_maintenance_workers=15 \
    -c shared_buffers=64GB \
    -c effective_cache_size=192GB \
    -c maintenance_work_mem=256GB \
    -c max_wal_size=8GB \
    -c checkpoint_timeout=15min \
    -c wal_compression=on
else
  docker start "$CLONE_CONTAINER" >/dev/null
fi

for _ in $(seq 1 120); do
  if docker exec "$CLONE_CONTAINER" pg_isready -U postgres -d "$DATABASE" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker exec "$CLONE_CONTAINER" pg_isready -U postgres -d "$DATABASE"

# The runtime image may predate the SQLens build installed in the source
# container. Keep both instances on the exact same extension binary.
source_vector_sha=$(docker exec "$SOURCE_CONTAINER" sha256sum /usr/lib/postgresql/16/lib/vector.so | awk '{print $1}')
clone_vector_sha=$(docker exec "$CLONE_CONTAINER" sha256sum /usr/lib/postgresql/16/lib/vector.so | awk '{print $1}')
if [[ "$source_vector_sha" != "$clone_vector_sha" ]]; then
  vector_tmp="/tmp/${CLONE_CONTAINER}-vector.so"
  docker cp "$SOURCE_CONTAINER:/usr/lib/postgresql/16/lib/vector.so" "$vector_tmp" >/dev/null
  docker cp "$vector_tmp" "$CLONE_CONTAINER:/usr/lib/postgresql/16/lib/vector.so" >/dev/null
  docker restart "$CLONE_CONTAINER" >/dev/null
  for _ in $(seq 1 120); do
    if docker exec "$CLONE_CONTAINER" pg_isready -U postgres -d "$DATABASE" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi
clone_vector_sha=$(docker exec "$CLONE_CONTAINER" sha256sum /usr/lib/postgresql/16/lib/vector.so | awk '{print $1}')
if [[ "$clone_vector_sha" != "$EXPECTED_VECTOR_SHA" ]]; then
  echo "unexpected clone vector.so SHA: $clone_vector_sha" >&2
  exit 2
fi

echo "loading_extensions:$(date -Is)" >"$STATUS"
docker exec -i "$CLONE_CONTAINER" psql -v ON_ERROR_STOP=1 -U postgres -d "$DATABASE" <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_prewarm;
CREATE EXTENSION IF NOT EXISTS hybrid_qual_profile;
SQL

echo "streaming_tables_and_building_source_indexes:$(date -Is)" >"$STATUS"
{
  printf '%s\n' 'BEGIN;' "SET LOCAL synchronous_commit=off;" "SET LOCAL maintenance_work_mem='256GB';" "SET LOCAL max_parallel_maintenance_workers='$MAX_PARALLEL_MAINTENANCE_WORKERS';"
  docker exec "$SOURCE_CONTAINER" pg_dump -U postgres -d "$DATABASE" \
    --section=pre-data --no-owner --no-privileges \
    --table=public.laion25m_pgvector \
    --table=public.laion25m_queries \
    --table=public.laion25m_pgvector_guidance_meta
  docker exec "$SOURCE_CONTAINER" pg_dump -U postgres -d "$DATABASE" \
    --section=data --no-owner --no-privileges \
    --table=public.laion25m_pgvector \
    --table=public.laion25m_queries \
    --table=public.laion25m_pgvector_guidance_meta
  cat <<'SQL'
ALTER TABLE ONLY public.laion25m_pgvector
  ADD CONSTRAINT laion25m_pgvector_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.laion25m_queries
  ADD CONSTRAINT laion25m_queries_pkey PRIMARY KEY (qid);
CREATE INDEX laion25m_pgvector_guidance_meta_id_idx
  ON public.laion25m_pgvector_guidance_meta USING btree (id);
CREATE INDEX laion25m_pgvector_guidance_meta_labels_gin
  ON public.laion25m_pgvector_guidance_meta USING gin (labels);
CREATE INDEX laion25m_pgvector_guidance_meta_width_idx
  ON public.laion25m_pgvector_guidance_meta USING btree (width);
CREATE INDEX laion25m_pgvector_label_count_idx
  ON public.laion25m_pgvector USING btree (label_count);
CREATE INDEX laion25m_pgvector_labels_gin
  ON public.laion25m_pgvector USING gin (labels);
CREATE INDEX laion25m_pgvector_width_idx
  ON public.laion25m_pgvector USING btree (width);
SELECT '[0]'::public.vector;
SELECT set_config('hnsw.require_full_memory_build', 'on', true);
SELECT set_config('hnsw.build_page_order', 'insertion', true);
CREATE INDEX laion25m_pgvector_embedding_hnsw
  ON public.laion25m_pgvector USING hnsw (embedding public.vector_l2_ops)
  WITH (m=16, ef_construction=100);
CREATE TRIGGER pgvector_hnsw_fragment_epoch
  AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE
  ON public.laion25m_pgvector FOR EACH STATEMENT
  EXECUTE FUNCTION public.vector_hnsw_fragment_epoch_bump_trigger();
COMMIT;
ANALYZE public.laion25m_pgvector;
ANALYZE public.laion25m_queries;
ANALYZE public.laion25m_pgvector_guidance_meta;
CHECKPOINT;
SQL
} | docker exec -i "$CLONE_CONTAINER" psql -v ON_ERROR_STOP=1 -U postgres -d "$DATABASE"

echo "building_same_graph_bfs_clone:$(date -Is)" >"$STATUS"
rm -f "$PROOF"
PGHOST=127.0.0.1 PGPORT="$HOST_PORT" PGDATABASE="$DATABASE" \
PGUSER=postgres PGPASSWORD=postgres \
  "${python_bin}" \
  "${repo_root}/experiments/hybrid_vector_db/scripts/prepare_pgvector_same_graph_bfs_clone.py" \
  --table public.laion25m_pgvector \
  --source-index public.laion25m_pgvector_embedding_hnsw \
  --clone-index public.laion25m_pgvector_embedding_hnsw_bfs_r32 \
  --maintenance-work-mem 256GB \
  --expected-sqlens-build-id "$EXPECTED_BUILD_ID" \
  --expected-vector-so-sha256 "$EXPECTED_VECTOR_SHA" \
  --out "$PROOF"

docker exec -i "$CLONE_CONTAINER" psql -v ON_ERROR_STOP=1 -U postgres -d "$DATABASE" <<'SQL'
CHECKPOINT;
SELECT relname, pg_size_pretty(pg_total_relation_size(oid)) AS total_size
FROM pg_class
WHERE relname LIKE 'laion25m%'
ORDER BY pg_total_relation_size(oid) DESC;
SQL

printf 'complete:%s:%s\n' "$CLONE_CONTAINER" "$(date -Is)" >"$STATUS"
echo "LAION-25M clone ready on 127.0.0.1:$HOST_PORT"
