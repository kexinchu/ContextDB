#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACORN_SRC="${ACORN_SRC:-external/ACORN}"
ACORN_LIB="$ACORN_SRC/build/faiss/libfaiss.a"
BIN="research/acorn_faiss_selectivity"
: "${PGPASSWORD:?set PGPASSWORD}"

if [[ ! -d "$ACORN_SRC" ]]; then
  echo "missing $ACORN_SRC; git clone https://github.com/stanford-futuredata/ACORN.git external/ACORN" >&2
  exit 1
fi

if [[ ! -f "$ACORN_LIB" ]]; then
  cmake -S "$ACORN_SRC" -B "$ACORN_SRC/build" \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$ACORN_SRC/build" --target faiss -j4
fi

g++ -O3 -std=c++17 -fopenmp \
  -I"$ACORN_SRC" \
  -I"$ACORN_SRC/build/_deps/nlohmann_json-src/include" \
  research/acorn_faiss_selectivity.cpp \
  "$ACORN_LIB" \
  -o "$BIN" \
  -L/usr/lib/x86_64-linux-gnu \
  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
  -lz -ldl -lpthread -lm

"$BIN" \
  --rows 200000 \
  --queries 100 \
  --k 10 \
  --m 16 \
  --ef-construction 64 \
  --ef-search 128 \
  --gamma 12 \
  --selectivities 1,2,5,10,20,30,40,50,60,70,80,90,100 \
  --systems ACORN-faiss-1 \
  --repeats 10 \
  --out research/results/acorn_faiss_200k_q100_r10.csv

export PGPORT="${PGPORT:-55438}"
. .venv/bin/activate
python research/hnswlib_vs_pgvector_selectivity.py \
  --queries 100 \
  --selectivities 1,2,5,10,20,30,40,50,60,70,80,90,100 \
  --systems HNSWlib-sweeping,PGVector-ACORN1,PGVector-sweeping \
  --repeats 10 \
  --out research/results/hnswlib_pgvector_4curves_200k_q100_r10.csv \
  --statement-timeout-ms 10000

python research/hnswlib_vs_pgvector_selectivity.py \
  --queries 100 \
  --selectivities 1,2,5,10,20,30,40,50,60,70,80,90,100 \
  --systems PGVector-ACORN1 \
  --repeats 10 \
  --collect-pg-profile \
  --pg-guidance-kind exact \
  --ef-search 16 \
  --out research/results/pgvector_acorn1_exact_pageguard_ef16_200k_q100_r10.csv \
  --statement-timeout-ms 10000

# Plot from the generated CSVs with research/plot_fig1_recall_latency_1pct.py.
