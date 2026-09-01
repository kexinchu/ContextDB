#!/usr/bin/env bash
# LAION-25M Figure-6 iso-recall calibration on dedicated clone :55434.
# CPU 32-47 (primary YFCC uses 48-63; Amazon secondary uses 0-31).
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WORKDIR="${WORKDIR:?set WORKDIR}"
OUT=$ROOT/results/hybrid_vector_db/figure5_r41_laion_iso_calib
STATUS=$ROOT/results/hybrid_vector_db/figure6_iso_recall_fill/laion_pipeline.status
TS=$(date +%Y%m%d_%H%M%S)
LOG=$WORKDIR/logs/laion_iso_calib_55434_${TS}.log
CFG=experiments/hybrid_vector_db/configs/figure5_r41_laion_primary.json
LOCK=results/hybrid_vector_db/.figure5_laion_55434_db.lock
CPU=32-47
CONTAINER=hybrid-pgvector-laion25m

mkdir -p "$OUT" "${OUT}_sqlens_cap" "${OUT}_sqlens_dense" "${OUT}_stock_iso_fill" \
  "$(dirname "$LOG")" "$(dirname "$STATUS")"

export PGHOST=127.0.0.1
export PGPORT=55434
export PGDATABASE=hybrid_vector
export PGUSER=postgres
: "${PGPASSWORD:?set PGPASSWORD}"
export PYTHONUNBUFFERED=1

exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"

echo "[$(date -Is)] LAION-25M iso-calib start (clone :55434, cpu ${CPU})"
echo "running_laion_iso_calib" >"$STATUS"
printf 'owner=laion_iso_calib_55434 pid=%s started=%s\n' "$$" "$(date -Is)" >"$LOCK"

docker update --cpuset-cpus "$CPU" "$CONTAINER" >/dev/null
observed=$(docker inspect "$CONTAINER" --format '{{.HostConfig.CpusetCpus}}')
if [[ "$observed" != "$CPU" ]]; then
  echo "cpuset mismatch requested=$CPU observed=$observed" >&2
  exit 1
fi

# Fail closed if vector.so drifted away from the r41 contract.
expected_sha=8f53226d35cae28d4e1b6926b13b01fa01fd1f6720c5f57c96c7886905f5eaf0
observed_sha=$(docker exec "$CONTAINER" sha256sum /usr/lib/postgresql/16/lib/vector.so | awk '{print $1}')
if [[ "$observed_sha" != "$expected_sha" ]]; then
  echo "LAION vector.so sha mismatch expected=$expected_sha observed=$observed_sha" >&2
  exit 1
fi
observed_build=$(docker exec "$CONTAINER" bash -lc "psql -U postgres -d hybrid_vector -tAc \"SELECT vector_sqlens_build_id();\"" | tr -d '[:space:]')
expected_build=sqlens-v16-distance-aware-route-budget-ef500k-20260801-r41
if [[ "$observed_build" != "$expected_build" ]]; then
  echo "LAION build_id mismatch expected=$expected_build observed=$observed_build" >&2
  exit 1
fi
# Formal workloads need qid 0..10199.
n_queries=$(docker exec "$CONTAINER" bash -lc "psql -U postgres -d hybrid_vector -tAc \"SELECT count(*) FROM laion25m_queries;\"" | tr -d '[:space:]')
if [[ "$n_queries" -lt 10200 ]]; then
  echo "LAION query table incomplete: count=$n_queries need >=10200" >&2
  exit 1
fi

run_cell() {
  local label=$1; shift
  echo "[$(date -Is)] >>> $label"
  python3 experiments/hybrid_vector_db/scripts/run_figure5_frontier.py "$@"
  echo "[$(date -Is)] <<< $label OK"
}

# Prefer stock_strict early (covers high recall), then sqlens_cap / both_off fills.
run_cell "stock_strict ladder" \
  --config "$CFG" --phase calibration --datasets laion --grid base \
  --scan-families stock_strict \
  --ef-search-values 20,40,60,80,100,150,200,250,500,750,1000,1500,2000,3000,5000,10000 \
  --backend-cpu-list "$CPU" --out-dir "$OUT" \
  --global-db-lock-path "$LOCK" --require-global-db-lock --resume --overwrite --execute

run_cell "sqlens_cap ladder" \
  --config "$CFG" --phase calibration --datasets laion --grid base \
  --scan-families sqlens_cap --ef-search-values 11 \
  --sqlens-scan-cap-values 500,1000,1500,2000,3000,4000,5000,7500,10000,15000,20000,25000,30000,50000,100000 \
  --backend-cpu-list "$CPU" --out-dir "${OUT}_sqlens_cap" \
  --global-db-lock-path "$LOCK" --require-global-db-lock --resume --overwrite --execute

run_cell "both_off ef<=1000" \
  --config "$CFG" --phase calibration --datasets laion --grid base \
  --scan-families both_off \
  --ef-search-values 20,40,60,80,100,150,200,250,500,750,1000 \
  --backend-cpu-list "$CPU" --out-dir "$OUT" \
  --global-db-lock-path "$LOCK" --require-global-db-lock --resume --overwrite --execute

run_cell "both_off dense fill" \
  --config "$CFG" --phase calibration --datasets laion --grid base \
  --scan-families both_off \
  --ef-search-values 12,14,16,18,25,30,35,45,50,70,90 \
  --allow-expensive-sqlens-calibration \
  --backend-cpu-list "$CPU" --out-dir "${OUT}_sqlens_dense" \
  --global-db-lock-path "$LOCK" --require-global-db-lock --resume --overwrite --execute

run_cell "both_off high-ef extension" \
  --config "$CFG" --phase calibration --datasets laion --grid base \
  --scan-families both_off \
  --ef-search-values 1500,2000,3000,5000,10000 \
  --allow-expensive-sqlens-calibration \
  --backend-cpu-list "$CPU" --out-dir "${OUT}_stock_iso_fill" \
  --global-db-lock-path "$LOCK" --require-global-db-lock --resume --overwrite --execute

echo "complete_laion_iso_calib" >"$STATUS"
echo "[$(date -Is)] LAION-25M ISO-CALIB COMPLETE on :55434"

# Build provisional iso-recall latency pairs for Figure 6 targets.
python3 experiments/hybrid_vector_db/scripts/plot_amazon10m_iso_recall.py --help >/dev/null 2>&1 || true
python3 - <<'PY'
import csv, re, math
from pathlib import Path
from collections import defaultdict

ROOT = Path(".")
targets = [0.75, 0.80, 0.85, 0.90, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
dirs = sorted(ROOT.glob("results/hybrid_vector_db/figure5_r41_laion_iso_calib*"))
MODE_S, MODE_Q = "original", "design1_bloom_bfs_layout_d3"
pat = re.compile(r"calibration_(.+)_ef(\d+)(?:_cap(\d+))?_profile_summary\.csv$")

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def geomean(xs):
    vals = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in vals) / len(vals)) if vals else None

pts_s, pts_q = [], []
for d in dirs:
    if not d.is_dir():
        continue
    for p in d.glob("*_profile_summary.csv"):
        m = pat.search(p.name)
        if not m:
            continue
        fam, ef = m.group(1), int(m.group(2))
        cap = int(m.group(3)) if m.group(3) else None
        by = defaultdict(list)
        with p.open() as f:
            for r in csv.DictReader(f):
                by[r["mode"]].append(
                    (float(r["recall_mean"]), float(r["end_to_end_mean_ms"]))
                )
        if MODE_S in by:
            rs = [x[0] for x in by[MODE_S]]
            ls = [x[1] for x in by[MODE_S]]
            pts_s.append({"family": fam, "ef": ef, "cap": cap, "recall": mean(rs), "lat": mean(ls), "lat_geo": geomean(ls)})
        if MODE_Q in by:
            rs = [x[0] for x in by[MODE_Q]]
            ls = [x[1] for x in by[MODE_Q]]
            pts_q.append({"family": fam, "ef": ef, "cap": cap, "recall": mean(rs), "lat": mean(ls), "lat_geo": geomean(ls)})

def pick(pts, t):
    if not pts:
        return None
    return min(pts, key=lambda p: (0 if p["recall"] >= t - 1e-9 else 1, abs(p["recall"] - t), p["lat"]))

out_dir = ROOT / "results/hybrid_vector_db/laion25m_iso_recall_plot"
out_dir.mkdir(parents=True, exist_ok=True)
long_rows = []
wide_rows = []
for t in targets:
    s, q = pick(pts_s, t), pick(pts_q, t)
    if s is None or q is None:
        print(f"[laion-pairs] incomplete at {t}: stock={s is not None} sqlens={q is not None}")
        continue
    wide_rows.append({
        "dataset": "laion25m",
        "target_recall": t,
        "stock_recall": round(s["recall"], 6),
        "stock_latency_mean_ms": round(s["lat"], 3),
        "stock_family": s["family"],
        "stock_ef_search": s["ef"],
        "sqlens_recall": round(q["recall"], 6),
        "sqlens_latency_mean_ms": round(q["lat"], 3),
        "sqlens_family": q["family"],
        "sqlens_ef_search": q["ef"],
        "sqlens_cap": q["cap"] or "",
        "speedup_vs_stock": round(s["lat"] / q["lat"], 4) if q["lat"] else "",
    })
    for arm, p in (("stock", s), ("sqlens", q)):
        long_rows.append({
            "dataset": "laion25m",
            "target_recall": t,
            "arm": arm,
            "recall": round(p["recall"], 6),
            "latency_mean_ms": round(p["lat"], 3),
            "family": p["family"],
            "ef_search": p["ef"],
            "scan_cap": p["cap"] or "",
            "abs_err_vs_target": round(abs(p["recall"] - t), 6),
            "metric_source": "laion_iso_calib_q2800",
        })

if wide_rows:
    with (out_dir / "laion25m_iso_recall_pairs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wide_rows[0].keys()))
        w.writeheader(); w.writerows(wide_rows)
    with (out_dir / "laion25m_iso_recall_pairs_long.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader(); w.writerows(long_rows)
    print(f"wrote {out_dir} pairs={len(wide_rows)} points_s={len(pts_s)} points_q={len(pts_q)}")
else:
    print("no laion pairs yet")
PY
