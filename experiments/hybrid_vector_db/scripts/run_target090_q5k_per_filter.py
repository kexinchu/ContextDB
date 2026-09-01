#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "experiments/hybrid_vector_db/scripts/pgvector_design1_design2_design3_selectivity_benchmark.py"
DEFAULT_CONFIG = ROOT / "experiments/hybrid_vector_db/configs/target090_q5k_filter_seeds.json"
BUILD_ID = "sqlens-v16-distance-aware-route-budget-ef500k-20260801-r41"
VECTOR_SHA = "8f53226d35cae28d4e1b6926b13b01fa01fd1f6720c5f57c96c7886905f5eaf0"
SQLENS_MODE = "design1_bloom_bfs_layout_d3"


DATASETS = {
    "amazon": {
        "port": "55433",
        "table": "public.amazon_grocery_reviews_10m_pgvector",
        "source_index": "public.amazon10m_hnsw_m32ef200_dupbridge_r29_source_idx",
        "bfs_index": "public.amazon10m_hnsw_m32ef200_dupbridge_r29_bfs_idx",
        "query_table": "public.amazon_grocery_reviews_10m_pgvector",
        "query_id_column": "id",
        "query_vector_column": "embedding",
        "candidate_validity": "embedding_valid",
        "truth": "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal.csv",
        "filters": "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_valid_embeddings_filters.csv",
        "calibration_workload": None,
        "measurement_workload": "results/hybrid_vector_db/table6_r41_target090_q5k_workloads/amazon_target090_q5k_measurement.csv",
        "measurement_source_requests": 5000,
        "measurement_limit": 0,
        "self_excluded": True,
        "graph_proof": "results/hybrid_vector_db/amazon10m_r41_secondary_source_bfs_graph_proof.json",
        "backend_cpus": "0-31",
    },
    "yfcc": {
        "port": "55432",
        "table": "public.yfcc10m_pgvector",
        "source_index": "public.yfcc10m_pgvector_embedding_hnsw",
        "bfs_index": "public.yfcc10m_pgvector_embedding_hnsw_bfs_r31",
        "query_table": "public.yfcc10m_queries",
        "query_id_column": "qid",
        "query_vector_column": "embedding",
        "candidate_validity": "TRUE",
        "truth": "results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_yfcc_exact_truth_q12800.csv",
        "filters": "results/hybrid_vector_db/yfcc10m_matched_recall_filters_q180.csv",
        "calibration_workload": None,
        "measurement_workload": "results/hybrid_vector_db/figure5_r37_formal_workloads/figure5_r37_yfcc_measurement.csv",
        "measurement_source_requests": 10000,
        "measurement_limit": 5000,
        "self_excluded": False,
        "graph_proof": "results/hybrid_vector_db/yfcc10m_r31_table6_shared_d2_warm_q100r5_20260723.csv.d2_graph_proof.json",
        "backend_cpus": "48-63",
    },
}


def mode_config(config: dict[str, object], *, sqlens: bool) -> dict[str, object]:
    defaults: dict[str, object] = {
        "ef_search": 100,
        "max_scan_tuples": 5_000_000,
        "scan_mem_multiplier": 32.0,
        "iterative_scan": "off",
        "guided_collect_target": 100,
        "traversal_guided_target": 10,
        "traversal_guided_prioritization": sqlens,
        "traversal_guided_burst": 8 if sqlens else 1,
        "traversal_guided_early_stop": sqlens,
        "traversal_guided_early_stop_distance_ratio": 0.95 if sqlens else 0.0,
    }
    defaults.update(config)
    defaults["guided_collect_target"] = int(defaults.get("guided_collect_target", defaults["ef_search"]))
    if not sqlens:
        defaults["traversal_guided_prioritization"] = False
        defaults["traversal_guided_early_stop"] = False
        defaults["traversal_guided_early_stop_distance_ratio"] = 0.0
    return defaults


def command_for(
    dataset: str,
    phase: str,
    target_recall: float,
    filter_name: str,
    pair: dict[str, object],
    out: Path,
) -> list[str]:
    meta = DATASETS[dataset]
    stock = mode_config(dict(pair["stock"]), sqlens=False)
    sqlens = mode_config(dict(pair["sqlens"]), sqlens=True)
    if dataset == "amazon":
        sqlens["traversal_guided_target"] = max(
            11, int(sqlens["traversal_guided_target"])
        )
    guidance = bool(pair.get("guidance", True))
    namespace_tag = hashlib.sha256(str(out.parent).encode("utf-8")).hexdigest()[:8]
    target_tag = f"t{round(target_recall * 100):02d}"
    configs = json.dumps({"original": stock, SQLENS_MODE: sqlens}, separators=(",", ":"))
    cmd = [
        sys.executable,
        str(RUNNER),
        "--insertion-table", meta["table"],
        "--insertion-index", meta["source_index"],
        "--bfs-table", meta["table"],
        "--bfs-index", meta["bfs_index"],
        "--query-table", meta["query_table"],
        "--query-id-column", meta["query_id_column"],
        "--query-vector-column", meta["query_vector_column"],
        "--candidate-validity-predicate", meta["candidate_validity"],
        "--truth-csv", str(ROOT / meta["truth"]),
        "--filters-csv", str(ROOT / meta["filters"]),
        "--filter-names", filter_name,
        "--modes", "original", SQLENS_MODE,
        "--execution-order", "interleaved",
        "--schedule-seed", "20260803",
        "--mode-configs-json", configs,
        "--repeats", "1",
        "--no-isolate-repeat-runtimes",
        "--warmup-queries", "0",
        "--no-warmup-all-queries",
        "--k", "10",
        "--guidance-filter-strategy", "traversal_guided",
        "--guidance-bypass-iterative-scan", str(sqlens["iterative_scan"]),
        "--guidance-bypass-ef-search", "0",
        "--guidance-low-selectivity-bypass-ef-search", "0",
        "--d1-exact-max-selectivity-pct", "6",
        "--collapse-exact-and-guidance",
        "--d3-cache-mb", "1024",
        "--d3-measurement-policy", "workload_driven_adaptive",
        "--d3-fragment-store-namespace", f"{target_tag}-{namespace_tag}-{dataset}-{phase[:3]}-{filter_name}"[:64],
        "--d3-probe-requests", "2",
        "--d3-min-benefit-per-byte", "0",
        "--d3-max-fragment-mb", "16",
        "--d3-page-min-skip-rate", "0.05",
        "--guidance-selectivity-min-pct", "0",
        "--guidance-selectivity-max-pct", "100" if guidance else "0",
        "--guidance-composite-max-selectivity-pct", "100" if guidance else "0",
        "--guidance-max-atoms", "8",
        "--d2-page-access", "off",
        "--d2-index-page-access", "off",
        "--statement-timeout-ms", "300000",
        "--force-hnsw",
        "--progress-queries", "0",
        "--d2-graph-proof-json", str(ROOT / meta["graph_proof"]),
        "--expected-sqlens-build-id", BUILD_ID,
        "--expected-vector-so-sha256", VECTOR_SHA,
        "--backend-cpu-list", meta["backend_cpus"],
        "--out", str(out),
    ]
    cmd.append("--expected-truth-self-excluded" if meta["self_excluded"] else "--no-expected-truth-self-excluded")
    if phase == "calibration":
        cmd.extend(["--queries", "200", "--query-offset", "0"])
    else:
        cmd.extend([
            "--workload-csv", str(ROOT / meta["measurement_workload"]),
            "--expected-workload-requests", str(meta["measurement_source_requests"]),
            "--workload-request-limit", str(meta["measurement_limit"]),
            "--require-unique-workload-queries",
        ])
    return cmd


def summarize(outputs: list[Path], out: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in outputs:
        by_mode: dict[str, list[dict[str, str]]] = {}
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row.get("error"):
                    raise RuntimeError(f"query error in {path}: {row['error']}")
                by_mode.setdefault(row["mode"], []).append(row)
        for mode, values in by_mode.items():
            latencies = [float(value["end_to_end_ms"]) for value in values]
            recalls = [float(value["recall"]) for value in values]
            ordered = sorted(latencies)
            p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
            rows.append({
                "filter_name": values[0]["filter_name"],
                "selectivity": values[0]["selectivity"],
                "mode": mode,
                "requests": len(values),
                "recall_mean": statistics.fmean(recalls),
                "end_to_end_mean_ms": statistics.fmean(latencies),
                "end_to_end_p95_ms": p95,
                "ef_search": values[0]["effective_ef_search"],
                "iterative_scan": values[0]["effective_iterative_scan"],
            })
    rows.sort(key=lambda row: (-float(row["selectivity"]), str(row["mode"])))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    for row in rows:
        print(
            f"{float(row['selectivity']):7.3f}% {row['filter_name']:24s} "
            f"{row['mode']:31s} n={row['requests']:>3} "
            f"recall={float(row['recall_mean']):.4f} "
            f"mean={float(row['end_to_end_mean_ms']):.2f}ms"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--phase", choices=["calibration", "measurement"], required=True)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--filters", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.target_recall <= 1.0:
        parser.error("--target-recall must be in (0, 1]")
    config = json.loads(args.config.read_text(encoding="utf-8"))[args.dataset]
    if args.filters:
        missing = set(args.filters) - set(config)
        if missing:
            parser.error(f"unknown filters: {sorted(missing)}")
        config = {name: config[name] for name in args.filters}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PGHOST": "127.0.0.1",
        "PGPORT": DATASETS[args.dataset]["port"],
        "PGDATABASE": "hybrid_vector",
        "PGUSER": "postgres",
        "PGPASSWORD": os.environ["PGPASSWORD"],
        "PYTHONUNBUFFERED": "1",
    })

    def run_one(item: tuple[str, dict[str, object]]) -> Path:
        filter_name, pair = item
        out = args.out_dir / f"{args.dataset}_{args.phase}_{filter_name}.csv"
        if out.exists():
            if not args.overwrite:
                return out
            out.unlink()
        completed = subprocess.run(
            command_for(
                args.dataset,
                args.phase,
                args.target_recall,
                filter_name,
                pair,
                out,
            ),
            cwd=ROOT,
            env=env,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"{filter_name} failed with exit code {completed.returncode}")
        return out

    outputs: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {executor.submit(run_one, item): item[0] for item in config.items()}
        for future in as_completed(futures):
            outputs.append(future.result())
            print(f"complete {args.dataset}/{args.phase}/{futures[future]}", flush=True)
    summarize(outputs, args.out_dir / f"{args.dataset}_{args.phase}_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
