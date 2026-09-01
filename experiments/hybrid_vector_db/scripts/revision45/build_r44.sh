#!/usr/bin/env bash
# Build the r44 FragReuse copy. Does not touch the 55437 table10 instance.
set -euo pipefail
src=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/third_party/pgvector-sqlens-r44
out=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/results/hybrid_vector_db/release_binaries/r44
image=${TABLE10_IMAGE:-pgvector/pgvector:pg16}

mkdir -p "$out"
docker run --rm \
  --user 0 \
  -v "$src:/src" \
  -w /src \
  "$image" \
  bash -lc '
    set -euo pipefail
    if ! command -v gcc >/dev/null; then
      apt-get update -qq
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc make postgresql-server-dev-16 >/dev/null
    fi
    make clean
    make OPTFLAGS="" -j"$(nproc)"
    test -f vector.so
  '
cp -a "$src/vector.so" "$out/vector.so"
sha256sum "$out/vector.so"
strings "$out/vector.so" | grep -F 'sqlens-v19-fragreuse-admit-20260830'
echo "wrote $out/vector.so"
