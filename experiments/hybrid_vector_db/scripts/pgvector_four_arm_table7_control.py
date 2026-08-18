"""Formal four-arm matched-recall Table-7 controller.

For each predicate, the controller restarts PostgreSQL once, prewarms the
source and same-graph BFS HNSW indexes once each, and delegates one balanced
request-level interleaved run containing Stock, D1, D1+D2, and D1+D2+D3.
The D3 arm completes admission during an untimed warmup and must reuse the
admitted representation on every measured request.  Formal Bloom runs force
the adaptive page probe to refine to Bloom before measurement so D3 remains a
representation-preserving reuse optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import statistics
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import pgvector_d2_cache_isolation_control as d2
    from . import pgvector_d3_table6_increment_control as d3
except ImportError:
    import pgvector_d2_cache_isolation_control as d2
    import pgvector_d3_table6_increment_control as d3


RUNNER = Path(__file__).with_name(
    "pgvector_design1_design2_design3_selectivity_benchmark.py"
)
MODES = (
    "original",
    "design1_bloom",
    "design1_bloom_bfs_layout",
    "design1_bloom_bfs_layout_d3",
)
MODE_LABELS = {
    "original": "Stock",
    "design1_bloom": "D1",
    "design1_bloom_bfs_layout": "D1+D2",
    "design1_bloom_bfs_layout_d3": "D1+D2+D3",
}
INCREMENTS = (
    ("d1", "original", "design1_bloom"),
    ("d2", "design1_bloom", "design1_bloom_bfs_layout"),
    ("d3", "design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3"),
)


class ControlError(RuntimeError):
    """The four-arm Table-7 contract was not established."""


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": d2.sha256_file(resolved),
    }


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mode_config(config: d2.MatchedConfig) -> dict[str, object]:
    return {
        "ef_search": config.ef_search,
        "max_scan_tuples": config.max_scan_tuples,
        "scan_mem_multiplier": config.scan_mem_multiplier,
        "iterative_scan": config.iterative_scan,
        "guided_collect_target": 1,
        "traversal_guided_prioritization": False,
    }


def namespace_for(run_id: str, filter_name: str) -> str:
    namespace = f"t7_{run_id}_{filter_name}"
    if len(namespace) > 64:
        raise ControlError(f"D3 namespace is too long: {namespace}")
    return namespace


def child_path(
    out: Path, position: int, filter_name: str, attempt: int
) -> Path:
    return out.with_name(
        f"{out.stem}.f{position:02d}.{filter_name}.a{attempt:03d}{out.suffix}"
    )


def expected_index(args: argparse.Namespace, mode: str) -> str:
    if mode in {"design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3"}:
        return args.bfs_index
    return args.source_index


def build_runner_command(
    args: argparse.Namespace,
    child_out: Path,
    filter_name: str,
    namespace: str,
    config: d2.MatchedConfig,
) -> list[str]:
    configs = {mode: mode_config(config) for mode in MODES}
    command = [
        str(args.python),
        str(RUNNER),
        "--out",
        str(child_out),
        "--filters-csv",
        str(args.filters_csv),
        "--truth-csv",
        str(args.truth_csv),
        "--insertion-table",
        args.table,
        "--insertion-index",
        args.source_index,
        "--bfs-table",
        args.table,
        "--bfs-index",
        args.bfs_index,
        "--candidate-validity-predicate",
        args.candidate_validity_predicate,
        "--modes",
        *MODES,
        "--execution-order",
        "interleaved",
        "--schedule-seed",
        str(args.schedule_seed),
        "--mode-configs-json",
        json.dumps(configs, sort_keys=True),
        "--filter-names",
        filter_name,
        "--queries",
        str(args.queries),
        "--query-offset",
        str(args.query_offset),
        "--repeats",
        str(args.repeats),
        "--k",
        str(args.k),
        "--guidance-filter-strategy",
        "safe_guided",
        "--guidance-max-atoms",
        str(args.guidance_max_atoms),
        "--d1-guidance-kind",
        args.d1_guidance_kind,
        "--d1-cache-mb",
        str(args.d1_cache_mb),
        "--d3-cache-mb",
        str(args.d3_cache_mb),
        "--d3-measurement-policy",
        "admitted_warm_reuse",
        "--d3-fragment-store-namespace",
        namespace,
        "--d3-probe-requests",
        str(args.d3_probe_requests),
        "--d3-min-benefit-per-byte",
        str(args.d3_min_benefit_per_byte),
        "--d3-max-fragment-mb",
        str(args.d3_max_fragment_mb),
        "--d3-page-min-skip-rate",
        str(args.d3_page_min_skip_rate),
        "--d2-page-access",
        "off",
        "--d2-index-page-access",
        "off",
        "--d2-graph-proof-json",
        str(args.d2_graph_proof_json),
        "--warmup-all-queries",
        "--fragment-tracking-prepared",
        "--statement-timeout-ms",
        str(args.statement_timeout_ms),
        "--progress-queries",
        str(args.progress_queries),
        "--expected-sqlens-build-id",
        args.expected_sqlens_build_id,
        "--expected-vector-so-sha256",
        args.expected_vector_so_sha256,
        "--backend-cpu-list",
        str(args.backend_cpu),
    ]
    command.append(
        "--expected-truth-self-excluded"
        if args.expected_truth_self_excluded
        else "--no-expected-truth-self-excluded"
    )
    if args.query_table:
        command.extend(["--query-table", args.query_table])
    command.extend(["--query-id-column", args.query_id_column])
    command.extend(["--query-vector-column", args.query_vector_column])
    return command


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def validate_child(
    rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
    filter_name: str,
    namespace: str,
    config: d2.MatchedConfig,
) -> list[dict[str, Any]]:
    expected_rows = args.queries * args.repeats * len(MODES)
    if len(rows) != expected_rows:
        raise ControlError(
            f"{filter_name}: expected {expected_rows} rows, observed {len(rows)}"
        )

    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    requests_by_mode: dict[str, set[tuple[int, int]]] = {
        mode: set() for mode in MODES
    }
    for row in rows:
        mode = str(row.get("mode") or "")
        if row.get("filter_name") != filter_name or mode not in MODES:
            raise ControlError(f"{filter_name}: unexpected filter or arm")
        if row.get("error"):
            raise ControlError(
                f"{filter_name}/{mode}: query failed: {row.get('error_detail')}"
            )
        if row.get("index") != expected_index(args, mode):
            raise ControlError(f"{filter_name}/{mode}: wrong HNSW index")
        if not _parse_bool(row.get("planner_proof_verified")):
            raise ControlError(f"{filter_name}/{mode}: planner proof failed")
        if not _parse_bool(row.get("backend_cpu_exact_match")):
            raise ControlError(f"{filter_name}/{mode}: backend CPU affinity drifted")
        if row.get("sqlens_build_id") != args.expected_sqlens_build_id:
            raise ControlError(f"{filter_name}/{mode}: SQLens build ID drifted")
        if row.get("vector_so_sha256") != args.expected_vector_so_sha256:
            raise ControlError(f"{filter_name}/{mode}: vector.so hash drifted")
        if int(row["ef_search"]) != config.ef_search:
            raise ControlError(f"{filter_name}/{mode}: ef_search drifted")
        if int(row["max_scan_tuples"]) != config.max_scan_tuples:
            raise ControlError(f"{filter_name}/{mode}: max_scan_tuples drifted")
        if row["iterative_scan"] != config.iterative_scan:
            raise ControlError(f"{filter_name}/{mode}: iterative_scan drifted")
        if int(row["guided_collect_target"]) != 1:
            raise ControlError(f"{filter_name}/{mode}: guided_collect_target drifted")
        request = (int(row["query_no"]), int(row["repeat"]))
        if request[0] not in range(
            args.query_offset, args.query_offset + args.queries
        ) or request[1] not in range(args.repeats):
            raise ControlError(f"{filter_name}: request is outside the held-out split")
        requests_by_mode[mode].add(request)
        grouped[str(row["pair_key"])].append(row)

    expected_requests = {
        (query_no, repeat)
        for query_no in range(args.query_offset, args.query_offset + args.queries)
        for repeat in range(args.repeats)
    }
    if any(requests != expected_requests for requests in requests_by_mode.values()):
        raise ControlError(f"{filter_name}: per-arm request coverage is incomplete")
    if len(grouped) != args.queries * args.repeats:
        raise ControlError(f"{filter_name}: pair-key coverage is incomplete")

    enriched: list[dict[str, Any]] = []
    for pair_key, group in grouped.items():
        by_mode = {str(row["mode"]): row for row in group}
        if len(group) != len(MODES) or set(by_mode) != set(MODES):
            raise ControlError(f"{filter_name}: incomplete four-arm pair {pair_key}")
        stock = by_mode["original"]
        for mode in MODES[1:]:
            row = by_mode[mode]
            for field in ("ids", "result_distances", "recall", "returned"):
                if row[field] != stock[field]:
                    raise ControlError(
                        f"{filter_name}: {mode} changed {field} for {pair_key}"
                    )

        d123 = by_mode["design1_bloom_bfs_layout_d3"]
        if d123.get("d3_initialization") != "admitted_warm_reuse":
            raise ControlError(f"{filter_name}: D3 policy drifted")
        if d123.get("d3_fragment_store_namespace") != namespace:
            raise ControlError(f"{filter_name}: D3 namespace drifted")
        if d123.get("d3_phase") != "warm":
            raise ControlError(f"{filter_name}: measured D3 request is not warm")
        representation = d3.expected_d3_representation(
            args, float(d123["selectivity"])
        )
        state = str(d123.get("d3_state_after") or "")
        if representation == "exact" and state != "exact":
            raise ControlError(
                f"{filter_name}: exact D3 representation became {state!r}"
            )
        if representation == "bloom" and state != "bloom":
            raise ControlError(
                f"{filter_name}: Bloom D3 representation became {state!r}"
            )
        if representation == "bloom_or_page" and state not in {"page", "bloom"}:
            raise ControlError(
                f"{filter_name}: invalid adaptive D3 representation {state!r}"
            )
        if not _parse_bool(d123.get("d3_active_guidance_reused")):
            raise ControlError(f"{filter_name}: measured D3 guide was not reused")
        if int(d123.get("d3_fragment_builds_delta") or 0) != 0:
            raise ControlError(f"{filter_name}: measured D3 request rebuilt guidance")
        if int(d123.get("d3_composed_guide_hits_delta") or 0) <= 0:
            raise ControlError(f"{filter_name}: measured D3 request missed reuse")

        for mode in MODES:
            enriched.append(
                {
                    **by_mode[mode],
                    "table7_config_qualification": config.qualification,
                    "table7_config_target_recall": config.target_recall,
                    "table7_d3_namespace": namespace,
                }
            )
    return enriched


def bootstrap_query_mean_ci(
    values: Mapping[int, Sequence[float]], seed: int, samples: int = 10_000
) -> tuple[float, float]:
    query_ids = sorted(values)
    cluster_means = [statistics.fmean(values[q]) for q in query_ids]
    rng = random.Random(seed)
    draws = [
        statistics.fmean(rng.choice(cluster_means) for _ in cluster_means)
        for _ in range(samples)
    ]
    draws.sort()
    return draws[int(0.025 * samples)], draws[int(0.975 * samples) - 1]


def summarize(
    rows: Sequence[Mapping[str, Any]],
    filter_order: Sequence[str],
    seed: int,
    target_recall: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for filter_position, filter_name in enumerate(filter_order):
        selected = [row for row in rows if row["filter_name"] == filter_name]
        by_mode = {
            mode: [row for row in selected if row["mode"] == mode]
            for mode in MODES
        }
        if any(not arm for arm in by_mode.values()):
            raise ControlError(f"{filter_name}: summary arm is empty")

        means = {
            mode: statistics.fmean(
                float(row["end_to_end_ms"]) for row in by_mode[mode]
            )
            for mode in MODES
        }
        recalls = {
            mode: statistics.fmean(float(row["recall"]) for row in by_mode[mode])
            for mode in MODES
        }
        if len({round(value, 12) for value in recalls.values()}) != 1:
            raise ControlError(f"{filter_name}: four-arm mean recall differs")
        if any(value < target_recall for value in recalls.values()):
            raise ControlError(
                f"{filter_name}: held-out recall is below {target_recall:.2f}"
            )

        item: dict[str, Any] = {
            "filter_name": filter_name,
            "selectivity": float(selected[0]["selectivity"]),
            "ef_search": int(selected[0]["ef_search"]),
            "max_scan_tuples": int(selected[0]["max_scan_tuples"]),
            "scan_mem_multiplier": float(selected[0]["scan_mem_multiplier"]),
            "iterative_scan": selected[0]["iterative_scan"],
            "queries": len(
                {int(row["query_no"]) for row in by_mode["original"]}
            ),
            "repeats": len(by_mode["original"])
            // len({int(row["query_no"]) for row in by_mode["original"]}),
            "stock_ms": means["original"],
            "d1_ms": means["design1_bloom"],
            "d12_ms": means["design1_bloom_bfs_layout"],
            "d123_ms": means["design1_bloom_bfs_layout_d3"],
            "stock_recall": recalls["original"],
            "d1_recall": recalls["design1_bloom"],
            "d12_recall": recalls["design1_bloom_bfs_layout"],
            "d123_recall": recalls["design1_bloom_bfs_layout_d3"],
            "d1_speedup_over_stock": means["original"]
            / means["design1_bloom"],
            "d12_speedup_over_stock": means["original"]
            / means["design1_bloom_bfs_layout"],
            "d123_speedup_over_stock": means["original"]
            / means["design1_bloom_bfs_layout_d3"],
            "d2_increment_speedup": means["design1_bloom"]
            / means["design1_bloom_bfs_layout"],
            "d3_increment_speedup": means["design1_bloom_bfs_layout"]
            / means["design1_bloom_bfs_layout_d3"],
            "best_sqlens_ms": min(means[mode] for mode in MODES[1:]),
            "best_sqlens_speedup": means["original"]
            / min(means[mode] for mode in MODES[1:]),
            "all_results_identical": True,
            "all_recall_targets_met": True,
        }
        for increment_position, (name, base_mode, new_mode) in enumerate(
            INCREMENTS
        ):
            base_by_pair = {
                str(row["pair_key"]): row for row in by_mode[base_mode]
            }
            deltas: dict[int, list[float]] = defaultdict(list)
            for row in by_mode[new_mode]:
                base = base_by_pair[str(row["pair_key"])]
                deltas[int(row["query_no"])].append(
                    float(row["end_to_end_ms"])
                    - float(base["end_to_end_ms"])
                )
            ci_low, ci_high = bootstrap_query_mean_ci(
                deltas,
                seed + 101 * filter_position + 17 * increment_position,
            )
            item[f"{name}_minus_base_ms"] = statistics.fmean(
                statistics.fmean(values) for values in deltas.values()
            )
            item[f"{name}_minus_base_ci95_low_ms"] = ci_low
            item[f"{name}_minus_base_ci95_high_ms"] = ci_high
            item[f"{name}_statistically_positive"] = ci_high < 0
        output.append(item)
    return output


def plan_evidence(
    path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not path.is_file():
        raise ControlError(f"missing child plan evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ControlError(f"child plan evidence is not complete: {path}")
    checks = payload.get("checks")
    if not isinstance(checks, list) or len(checks) != len(MODES):
        raise ControlError(f"expected four planner checks in {path}")
    by_mode = {str(check.get("mode")): check for check in checks}
    if set(by_mode) != set(MODES):
        raise ControlError(f"planner checks do not cover all four arms in {path}")
    for mode, check in by_mode.items():
        if check.get("passed") is not True:
            raise ControlError(f"{mode}: planner check did not pass")
        if check.get("expected_index") != expected_index(args, mode):
            raise ControlError(f"{mode}: planner expected-index drifted")
        if check.get("planner_proof_verified") is not True:
            raise ControlError(f"{mode}: planner proof was not verified")

    lifecycle = payload.get("execution_lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise ControlError(f"missing child lifecycle evidence: {path}")
    if lifecycle.get("warmup_policy") != "admitted_warm_reuse":
        raise ControlError(f"child D3 warmup policy drifted: {path}")
    if lifecycle.get("warmup_complete") is not True:
        raise ControlError(f"child warmup did not complete: {path}")
    if lifecycle.get("d3_lifecycle_complete") is not True:
        raise ControlError(f"child D3 lifecycle did not complete: {path}")
    return {
        "path": str(path),
        "sha256": d2.sha256_file(path),
        "planner_checks": len(checks),
        "warmup_observed": lifecycle.get("warmup_observed"),
        "d3_phase_counts": lifecycle.get("d3_phase_counts"),
        "d3_warmup_phase_evidence_count": lifecycle.get(
            "d3_warmup_phase_evidence_count"
        ),
    }


def clear_fragment_namespace(args: argparse.Namespace, namespace: str) -> int:
    runner = d2.load_runner()
    connection = runner.psycopg.connect(
        runner.pg_config_from_env().conninfo, autocommit=True
    )
    try:
        cur = connection.cursor()
        cur.execute(
            "DELETE FROM public.pgvector_hnsw_fragment_store "
            "WHERE heap_oid=%s::regclass::oid "
            "AND left(filter_name, length(%s) + 1) = %s || chr(31)",
            (args.table, namespace, namespace),
        )
        return int(cur.rowcount)
    finally:
        connection.close()


def prewarm_both_indexes(
    args: argparse.Namespace,
    identities: Mapping[str, Mapping[str, Any]],
    position: int,
) -> list[dict[str, Any]]:
    source = d2.Arm("table7_source", "original", "source", args.source_index)
    bfs = d2.Arm(
        "table7_bfs", "design1_bloom_bfs_layout", "bfs", args.bfs_index
    )
    arms = [source, bfs] if position % 2 else [bfs, source]
    return [
        d2.prewarm_for_arm(
            args,
            arm,
            identities,
            int(identities[arm.expected_index]["blocks"]),
        )
        for arm in arms
    ]


def run_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "sqlens-four-arm-table7-v1",
        "out": str(args.out.resolve()),
        "filters_csv": file_identity(args.filters_csv),
        "truth_csv": file_identity(args.truth_csv),
        "truth_manifest": file_identity(args.truth_manifest),
        "matched_configs_csv": file_identity(args.matched_configs_csv),
        "matched_configs_manifest": file_identity(
            args.matched_configs_manifest
        ),
        "d2_graph_proof_json": file_identity(args.d2_graph_proof_json),
        "filter_names": list(args.filter_names),
        "table": args.table,
        "source_index": args.source_index,
        "bfs_index": args.bfs_index,
        "query_table": args.query_table,
        "query_id_column": args.query_id_column,
        "query_vector_column": args.query_vector_column,
        "candidate_validity_predicate": args.candidate_validity_predicate,
        "expected_truth_self_excluded": args.expected_truth_self_excluded,
        "backend_cpu": args.backend_cpu,
        "query_offset": args.query_offset,
        "queries": args.queries,
        "repeats": args.repeats,
        "k": args.k,
        "schedule_seed": args.schedule_seed,
        "guidance_max_atoms": args.guidance_max_atoms,
        "d1_guidance_kind": args.d1_guidance_kind,
        "d1_cache_mb": args.d1_cache_mb,
        "d3_cache_mb": args.d3_cache_mb,
        "d3_probe_requests": args.d3_probe_requests,
        "d3_min_benefit_per_byte": args.d3_min_benefit_per_byte,
        "d3_max_fragment_mb": args.d3_max_fragment_mb,
        "d3_page_min_skip_rate": args.d3_page_min_skip_rate,
        "matched_target_recall": args.matched_target_recall,
        "expected_sqlens_build_id": args.expected_sqlens_build_id,
        "expected_vector_so_sha256": args.expected_vector_so_sha256,
        "expected_candidate_rows": args.expected_candidate_rows,
    }


def protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": "sqlens-four-arm-table7-v1",
        "arms": [
            {
                "mode": mode,
                "label": MODE_LABELS[mode],
                "index": expected_index(args, mode),
            }
            for mode in MODES
        ],
        "configuration": {
            "source": str(args.matched_configs_csv),
            "shared_full_search_config_across_four_arms": True,
            "target_recall": args.matched_target_recall,
        },
        "measurement": {
            "query_offset": args.query_offset,
            "queries": args.queries,
            "repeats": args.repeats,
            "execution_order": "balanced seeded request-level interleaving",
            "latency": "activation plus SQL query end-to-end milliseconds",
            "truth_self_excluded": args.expected_truth_self_excluded,
        },
        "cache": {
            "postgres_restarts_per_filter": 1,
            "source_index_full_prewarm_per_filter": 1,
            "bfs_index_full_prewarm_per_filter": 1,
            "prewarm_order_balanced_across_filters": True,
            "untimed_full_query_pass_per_arm": args.queries,
        },
        "d3": {
            "measurement_policy": "admitted_warm_reuse",
            "namespace_per_filter": True,
            "measured_fragment_builds_allowed": 0,
            "online_adaptation_cost_reported_separately": True,
            "representation_preserving": args.d1_guidance_kind == "bloom",
            "page_probe_must_refine_to_bloom": (
                args.d1_guidance_kind == "bloom"
            ),
            "page_min_skip_rate": args.d3_page_min_skip_rate,
        },
        "resume": {
            "completed_children_reaudited": True,
            "incomplete_filter_gets_new_attempt": True,
        },
    }


def _load_completed_rows(
    manifest: Mapping[str, Any],
    args: argparse.Namespace,
    configs: Mapping[str, d2.MatchedConfig],
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    for record in manifest.get("invocations", []):
        if record.get("status") != "complete":
            continue
        filter_name = str(record["filter_name"])
        if filter_name in completed:
            raise ControlError(f"duplicate completed filter in checkpoint: {filter_name}")
        artifact = record.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ControlError(f"{filter_name}: checkpoint artifact is missing")
        path = Path(str(artifact["path"]))
        if not path.is_file() or d2.sha256_file(path) != artifact.get("sha256"):
            raise ControlError(f"{filter_name}: checkpoint artifact identity drifted")
        namespace = str(record["namespace"])
        child_rows = d2.read_csv(path)
        rows.extend(
            validate_child(
                child_rows, args, filter_name, namespace, configs[filter_name]
            )
        )
        plan_evidence(
            path.with_suffix(path.suffix + ".plan.json"),
            args,
        )
        completed.add(filter_name)
    return rows, completed


def execute(args: argparse.Namespace) -> None:
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    spec = run_spec(args)
    spec_sha = sha256_json(spec)
    configs = d2.load_matched_configs(args, args.filters_csv, args.truth_csv)
    truth = d2.audit_truth_for_args(args)

    if args.resume:
        if not manifest_path.is_file():
            raise ControlError("--resume requires an existing manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_spec_sha256") != spec_sha:
            raise ControlError("resume run specification differs from checkpoint")
        run_id = str(manifest["run_id"])
        scheduled = list(manifest["scheduled_filter_order"])
        combined, completed_filters = _load_completed_rows(
            manifest, args, configs
        )
        manifest["status"] = "running"
        manifest["artifact_valid"] = False
        manifest["resumed_at"] = d2.utc_now()
        manifest.pop("error", None)
    else:
        if args.out.exists() or summary_path.exists() or manifest_path.exists():
            raise ControlError("refusing to overwrite an existing Table-7 artifact")
        run_id = uuid.uuid4().hex[:12]
        scheduled = list(args.filter_names)
        random.Random(args.schedule_seed).shuffle(scheduled)
        combined = []
        completed_filters: set[str] = set()
        manifest: dict[str, Any] = {
            "status": "running",
            "artifact_valid": False,
            "started_at": d2.utc_now(),
            "argv": sys.argv,
            "controller_sha256": d2.sha256_file(Path(__file__)),
            "runner_sha256": d2.sha256_file(RUNNER),
            "run_id": run_id,
            "run_spec": spec,
            "run_spec_sha256": spec_sha,
            "protocol": protocol(args),
            "filter_order": list(args.filter_names),
            "scheduled_filter_order": scheduled,
            "exact_truth_audit": truth,
            "matched_config_source": d2.matched_config_source_evidence(
                args, configs
            ),
            "matched_configs": {
                name: value.as_dict() for name, value in configs.items()
            },
            "invocations": [],
        }
    d2.atomic_write_json(manifest_path, manifest)

    original_cpuset = d2.inspect_cpuset(args.container)
    manifest["original_container_cpuset"] = original_cpuset
    d2.atomic_write_json(manifest_path, manifest)
    try:
        manifest["dedicated_server_gate"] = d2.require_dedicated_server()
        d2.set_container_cpu(args.container, args.backend_cpu)
        manifest["startup_restart"] = d2.restart_postgres(args)
        proof, identities = d2.prepare_database_contract(args)
        manifest["d2_graph_proof"] = proof
        manifest["index_identities_start"] = identities
        d2.atomic_write_json(manifest_path, manifest)

        for position, filter_name in enumerate(scheduled, start=1):
            if filter_name in completed_filters:
                continue
            prior_attempts = [
                int(record.get("attempt", 0))
                for record in manifest["invocations"]
                if record.get("filter_name") == filter_name
            ]
            attempt = max(prior_attempts, default=0) + 1
            namespace = namespace_for(run_id, filter_name)
            child_out = child_path(
                args.out, position, filter_name, attempt
            )
            record: dict[str, Any] = {
                "position": position,
                "filter_name": filter_name,
                "attempt": attempt,
                "namespace": namespace,
                "matched_config": configs[filter_name].as_dict(),
                "status": "running",
            }
            manifest["invocations"].append(record)
            d2.atomic_write_json(manifest_path, manifest)

            record["restart"] = d2.restart_postgres(args)
            record["cleared_fragment_rows"] = clear_fragment_namespace(
                args, namespace
            )
            record["prewarm_order"] = (
                ["source", "bfs"] if position % 2 else ["bfs", "source"]
            )
            record["prewarm"] = prewarm_both_indexes(
                args, identities, position
            )
            if d3.fragment_store_count(args, namespace) != 0:
                raise ControlError(
                    f"{filter_name}: D3 namespace did not start empty"
                )

            command = build_runner_command(
                args,
                child_out,
                filter_name,
                namespace,
                configs[filter_name],
            )
            record["runner_argv"] = command
            record["runner_shell"] = shlex.join(command)
            completed = d2.run_command(command)
            child_rows = d2.read_csv(child_out)
            validated = validate_child(
                child_rows,
                args,
                filter_name,
                namespace,
                configs[filter_name],
            )
            combined.extend(validated)
            plan = plan_evidence(
                child_out.with_suffix(child_out.suffix + ".plan.json"),
                args,
            )
            d3_rows = [
                row
                for row in validated
                if row["mode"] == "design1_bloom_bfs_layout_d3"
            ]
            record.update(
                {
                    "status": "complete",
                    "completed_at": d2.utc_now(),
                    "d3_states": sorted(
                        {str(row["d3_state_after"]) for row in d3_rows}
                    ),
                    "fragment_store_rows_after": d3.fragment_store_count(
                        args, namespace
                    ),
                    "plan_evidence": plan,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "artifact": {
                        "path": str(child_out),
                        "rows": len(child_rows),
                        "sha256": d2.sha256_file(child_out),
                    },
                }
            )
            completed_filters.add(filter_name)
            d2.atomic_write_json(manifest_path, manifest)

        if completed_filters != set(args.filter_names):
            raise ControlError("not all requested filters completed")
        summaries = summarize(
            combined,
            args.filter_names,
            args.schedule_seed,
            args.matched_target_recall,
        )
        d2.atomic_write_csv(args.out, combined)
        d2.atomic_write_csv(summary_path, summaries)

        proof_final, identities_final = d2.prepare_database_contract(args)
        if identities_final != identities:
            raise ControlError("index identity changed during Table-7")
        if (
            proof_final["stable_fingerprint_sha256"]
            != proof["stable_fingerprint_sha256"]
        ):
            raise ControlError("same-graph proof changed during Table-7")

        manifest.update(
            {
                "status": "complete",
                "artifact_valid": True,
                "completed_at": d2.utc_now(),
                "all_results_identical": True,
                "all_recall_targets_met": True,
                "completed_filter_count": len(completed_filters),
                "geomean_speedups": {
                    key: math.exp(
                        statistics.fmean(
                            math.log(float(row[key])) for row in summaries
                        )
                    )
                    for key in (
                        "d1_speedup_over_stock",
                        "d12_speedup_over_stock",
                        "d123_speedup_over_stock",
                        "d2_increment_speedup",
                        "d3_increment_speedup",
                    )
                },
                "positive_points": {
                    name: sum(
                        bool(row[f"{name}_statistically_positive"])
                        for row in summaries
                    )
                    for name, _, _ in INCREMENTS
                },
                "index_identities_final": identities_final,
                "d2_graph_proof_final": proof_final,
                "outputs": {
                    "raw": {
                        "path": str(args.out),
                        "rows": len(combined),
                        "sha256": d2.sha256_file(args.out),
                    },
                    "summary": {
                        "path": str(summary_path),
                        "rows": len(summaries),
                        "sha256": d2.sha256_file(summary_path),
                    },
                },
            }
        )
        d2.atomic_write_json(manifest_path, manifest)
        print(
            json.dumps(
                {"manifest": str(manifest_path), "summary": summaries},
                indent=2,
            )
        )
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "artifact_valid": False,
                "completed_at": d2.utc_now(),
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
            }
        )
        d2.atomic_write_json(manifest_path, manifest)
        raise
    finally:
        if d2.inspect_cpuset(args.container) != original_cpuset:
            d2.run_command(
                [
                    "docker",
                    "update",
                    f"--cpuset-cpus={original_cpuset}",
                    args.container,
                ]
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one restart/prewarm lifecycle per filter and compare Stock, "
            "D1, D1+D2, and D1+D2+D3 in one interleaved benchmark."
        )
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filters-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument(
        "--truth-provenance-manifest",
        "--truth-manifest",
        dest="truth_manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--matched-configs-csv", type=Path, required=True)
    parser.add_argument("--matched-configs-manifest", type=Path, required=True)
    parser.add_argument("--d2-graph-proof-json", type=Path, required=True)
    parser.add_argument("--filter-names", nargs="+", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--bfs-index", required=True)
    parser.add_argument("--query-table")
    parser.add_argument("--query-id-column", default="id")
    parser.add_argument("--query-vector-column", default="embedding")
    parser.add_argument("--candidate-validity-predicate", default="TRUE")
    parser.add_argument(
        "--expected-truth-self-excluded",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--container", default="hybrid-pgvector")
    parser.add_argument("--backend-cpu", type=int, required=True)
    parser.add_argument("--query-offset", type=int, default=80)
    parser.add_argument("--queries", type=d2.positive_int, default=100)
    parser.add_argument("--repeats", type=d2.positive_int, default=5)
    parser.add_argument("--k", type=d2.positive_int, default=10)
    parser.add_argument("--schedule-seed", type=int, default=20260723)
    parser.add_argument("--guidance-max-atoms", type=d2.positive_int, default=128)
    parser.add_argument(
        "--d1-guidance-kind",
        choices=("auto", "exact", "bloom"),
        default="auto",
    )
    parser.add_argument("--d1-cache-mb", type=d2.positive_int, default=1024)
    parser.add_argument("--d3-cache-mb", type=d2.positive_int, default=1024)
    parser.add_argument("--d3-probe-requests", type=d2.positive_int, default=2)
    parser.add_argument("--d3-min-benefit-per-byte", type=float, default=0.0)
    parser.add_argument("--d3-max-fragment-mb", type=d2.positive_int, default=256)
    parser.add_argument("--d3-page-min-skip-rate", type=float, default=1.0)
    parser.add_argument(
        "--matched-target-recall", type=float, default=0.90
    )
    parser.add_argument(
        "--statement-timeout-ms", type=d2.positive_int, default=600_000
    )
    parser.add_argument("--progress-queries", type=int, default=25)
    parser.add_argument("--readiness-timeout-s", type=float, default=60.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--expected-sqlens-build-id", required=True)
    parser.add_argument(
        "--expected-vector-so-sha256", type=d2.sha256_arg, required=True
    )
    parser.add_argument(
        "--expected-candidate-rows", type=d2.positive_int, required=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args(argv)

    args.matched_mode = "design1_bloom"
    args.matched_config_index_policy = "exact"
    args.live_graph_proof_policy = "delegated_immutable"
    args.cache_regime = "warm_resident"
    args.prewarm_index_blocks = None
    args.prewarm_common_relation = []
    args.allow_mean_qualified_matched_config = False
    args.matched_recall_manifest = None

    if args.backend_cpu < 0:
        parser.error("--backend-cpu must be nonnegative")
    if args.query_offset < 0:
        parser.error("--query-offset must be nonnegative")
    if len(set(args.filter_names)) != len(args.filter_names):
        parser.error("--filter-names contains duplicates")
    if not 0 < args.matched_target_recall <= 1:
        parser.error("--matched-target-recall must be within (0,1]")
    if not math.isfinite(args.d3_min_benefit_per_byte) or (
        args.d3_min_benefit_per_byte < 0
    ):
        parser.error("--d3-min-benefit-per-byte must be finite and nonnegative")
    if not 0 <= args.d3_page_min_skip_rate <= 1:
        parser.error("--d3-page-min-skip-rate must be within [0,1]")
    if (
        args.d1_guidance_kind == "bloom"
        and args.d3_page_min_skip_rate != 1.0
    ):
        parser.error(
            "formal Bloom four-arm runs require "
            "--d3-page-min-skip-rate=1.0 so D3 cannot retain a page "
            "representation"
        )
    for path in (
        args.filters_csv,
        args.truth_csv,
        args.truth_manifest,
        args.matched_configs_csv,
        args.matched_configs_manifest,
        args.d2_graph_proof_json,
        RUNNER,
    ):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    execute(parse_args(argv))


if __name__ == "__main__":
    main()
