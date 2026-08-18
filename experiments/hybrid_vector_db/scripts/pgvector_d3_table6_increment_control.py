"""Audited Table-6 D1+D2 versus D1+D2+D3 steady-state control.

The controller transfers per-predicate D1 search settings selected by an
audited matched-recall run to the same-graph BFS index.  For each predicate it
restarts PostgreSQL, fully prewarms that index, and delegates one interleaved
run containing exactly two arms.  D3 starts in an empty, run-specific
persistent-store namespace, completes probe/admission/materialization during
an untimed pass over the requested split, and must reuse admitted guidance on
every measured request.  Online D3 build cost belongs to the separate q10K
lifecycle experiment, not this steady-state incremental ablation.
"""

from __future__ import annotations

import argparse
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
except ImportError:
    import pgvector_d2_cache_isolation_control as d2


ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name(
    "pgvector_design1_design2_design3_selectivity_benchmark.py"
)
MODES = ("design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3")
DEFAULT_MATCHED_MANIFEST = ROOT / (
    "results/hybrid_vector_db/"
    "sigmod_matched_recall_manifest_5d701893fef3_r22_"
    "amazon14_safe_d12_t90_q80r2_q100r5.reaudited.json"
)
DEFAULT_GRAPH_PROOF = ROOT / (
    "results/hybrid_vector_db/amazon10m_r30_source_bfs_graph_proof_20260722.json"
)
DEFAULT_FILTERS = ROOT / (
    "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
)
DEFAULT_FILTER_NAMES = (
    "popular_ge1000",
    "popular_ge1340",
    "popular_ge1780",
    "popular_ge2428",
    "popular_ge3284",
    "popular_ge4559",
    "price_10_to_20",
    "popular_ge10066",
    "rating5_price_le10",
    "long_review_ge500",
    "grocery_rating5",
    "grocery_helpful",
    "helpful_ge20",
    "grocery_long500",
)


class ControlError(RuntimeError):
    """The Table-6 incremental-ablation contract was not established."""


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
    namespace = f"t6d3_{run_id}_{filter_name}"
    if len(namespace) > 64:
        raise ControlError(f"D3 namespace is too long: {namespace}")
    return namespace


def expected_d3_representation(args: argparse.Namespace, selectivity: float) -> str:
    if args.d1_guidance_kind != "auto":
        return args.d1_guidance_kind
    return "exact" if selectivity <= 2.5 else "bloom_or_page"


def fragment_store_count(args: argparse.Namespace, namespace: str) -> int:
    runner = d2.load_runner()
    connection = runner.psycopg.connect(
        runner.pg_config_from_env().conninfo, autocommit=True
    )
    try:
        cur = connection.cursor()
        cur.execute(
            "SELECT count(*)::bigint FROM public.pgvector_hnsw_fragment_store "
            "WHERE heap_oid=%s::regclass::oid "
            "AND left(filter_name, length(%s) + 1) = %s || chr(31)",
            (args.table, namespace, namespace),
        )
        return int(cur.fetchone()[0])
    finally:
        connection.close()


def child_path(out: Path, position: int, filter_name: str) -> Path:
    return out.with_name(f"{out.stem}.f{position:02d}.{filter_name}{out.suffix}")


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


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def validate_child(
    rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
    filter_name: str,
    namespace: str,
    config: d2.MatchedConfig,
) -> list[dict[str, Any]]:
    expected = args.queries * args.repeats * len(MODES)
    if len(rows) != expected:
        raise ControlError(
            f"{filter_name}: expected {expected} child rows, observed {len(rows)}"
        )
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    observed_requests: set[tuple[int, int]] = set()
    for row in rows:
        if row.get("filter_name") != filter_name or row.get("mode") not in MODES:
            raise ControlError(f"{filter_name}: child contains an unexpected arm/filter")
        if row.get("error"):
            raise ControlError(f"{filter_name}: measured query failed: {row.get('error_detail')}")
        if row.get("index") != args.bfs_index:
            raise ControlError(f"{filter_name}: measured arm did not use the BFS index")
        if not parse_bool(row.get("planner_proof_verified")):
            raise ControlError(f"{filter_name}: planner proof is not verified")
        if not parse_bool(row.get("backend_cpu_exact_match")):
            raise ControlError(f"{filter_name}: backend CPU affinity did not match")
        if row.get("sqlens_build_id") != args.expected_sqlens_build_id:
            raise ControlError(f"{filter_name}: SQLens build ID drifted")
        if row.get("vector_so_sha256") != args.expected_vector_so_sha256:
            raise ControlError(f"{filter_name}: vector.so hash drifted")
        if int(row["ef_search"]) != config.ef_search:
            raise ControlError(f"{filter_name}: ef_search differs from Table 6")
        if int(row["max_scan_tuples"]) != config.max_scan_tuples:
            raise ControlError(f"{filter_name}: max_scan_tuples differs from Table 6")
        if row["iterative_scan"] != config.iterative_scan:
            raise ControlError(f"{filter_name}: iterative_scan differs from Table 6")
        try:
            request = (int(row["query_no"]), int(row["repeat"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlError(f"{filter_name}: child has an invalid query/repeat") from exc
        if request[0] not in range(
            args.query_offset, args.query_offset + args.queries
        ) or request[1] not in range(args.repeats):
            raise ControlError(f"{filter_name}: child query/repeat is outside the CLI split")
        observed_requests.add(request)
        grouped[str(row["pair_key"])].append(row)

    expected_requests = {
        (query_no, repeat)
        for query_no in range(args.query_offset, args.query_offset + args.queries)
        for repeat in range(args.repeats)
    }
    if observed_requests != expected_requests:
        raise ControlError(f"{filter_name}: CLI query/repeat coverage is incomplete")
    if len(grouped) != args.queries * args.repeats:
        raise ControlError(f"{filter_name}: paired-key coverage is incomplete")
    enriched: list[dict[str, Any]] = []
    for pair_key, pair in grouped.items():
        if len(pair) != 2 or {row["mode"] for row in pair} != set(MODES):
            raise ControlError(f"{filter_name}: incomplete pair {pair_key}")
        by_mode = {row["mode"]: row for row in pair}
        d12 = by_mode[MODES[0]]
        d123 = by_mode[MODES[1]]
        for field in ("ids", "result_distances", "recall"):
            if d12[field] != d123[field]:
                raise ControlError(
                    f"{filter_name}: D3 changed {field} for pair {pair_key}"
                )
        if d123.get("d3_initialization") != "admitted_warm_reuse":
            raise ControlError(f"{filter_name}: D3 initialization policy drifted")
        if d123.get("d3_fragment_store_namespace") != namespace:
            raise ControlError(f"{filter_name}: D3 namespace drifted")
        if d123.get("d3_phase") != "warm":
            raise ControlError(f"{filter_name}: measured D3 phase is not warm")
        representation = expected_d3_representation(
            args, float(d123["selectivity"])
        )
        state_after = str(d123.get("d3_state_after") or "")
        if representation == "exact" and state_after != "exact":
            raise ControlError(
                f"{filter_name}: D3 replaced D1 exact guidance with {state_after!r}"
            )
        if representation == "bloom" and state_after != "bloom":
            raise ControlError(
                f"{filter_name}: D3 did not preserve requested Bloom guidance"
            )
        if representation == "bloom_or_page" and state_after not in {"page", "bloom"}:
            raise ControlError(
                f"{filter_name}: unexpected adaptive representation {state_after!r}"
            )
        if not parse_bool(d123.get("d3_active_guidance_reused")):
            raise ControlError(f"{filter_name}: measured D3 guide was not reused")
        if int(d123.get("d3_fragment_builds_delta") or 0) != 0:
            raise ControlError(f"{filter_name}: measured D3 request rebuilt a fragment")
        if int(d123.get("d3_composed_guide_hits_delta") or 0) <= 0:
            raise ControlError(f"{filter_name}: measured D3 request missed composed guidance")
        for row in pair:
            enriched.append(
                {
                    **row,
                    "table6_config_qualification": config.qualification,
                    "table6_config_target_recall": config.target_recall,
                    "d3_namespace": namespace,
                }
            )
    return enriched


def bootstrap_query_mean_ci(
    values: Mapping[int, Sequence[float]], seed: int, samples: int = 5000
) -> tuple[float, float]:
    query_ids = sorted(values)
    cluster_means = [statistics.fmean(values[q]) for q in query_ids]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(
            statistics.fmean(rng.choice(cluster_means) for _ in cluster_means)
        )
    draws.sort()
    return draws[int(0.025 * samples)], draws[int(0.975 * samples) - 1]


def summarize(
    rows: Sequence[Mapping[str, Any]], filter_order: Sequence[str], seed: int
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for position, filter_name in enumerate(filter_order):
        selected = [row for row in rows if row["filter_name"] == filter_name]
        by_mode = {
            mode: [row for row in selected if row["mode"] == mode] for mode in MODES
        }
        if any(not arm for arm in by_mode.values()):
            raise ControlError(f"{filter_name}: summary arm is empty")
        deltas: dict[int, list[float]] = defaultdict(list)
        keyed = {str(row["pair_key"]): row for row in by_mode[MODES[0]]}
        for row in by_mode[MODES[1]]:
            base = keyed[str(row["pair_key"])]
            deltas[int(row["query_no"])].append(
                float(row["end_to_end_ms"]) - float(base["end_to_end_ms"])
            )
        ci_low, ci_high = bootstrap_query_mean_ci(deltas, seed + position)
        d12_mean = statistics.fmean(
            float(row["end_to_end_ms"]) for row in by_mode[MODES[0]]
        )
        d123_mean = statistics.fmean(
            float(row["end_to_end_ms"]) for row in by_mode[MODES[1]]
        )
        d12_query = statistics.fmean(
            float(row["query_latency_ms"]) for row in by_mode[MODES[0]]
        )
        d123_query = statistics.fmean(
            float(row["query_latency_ms"]) for row in by_mode[MODES[1]]
        )
        summaries.append(
            {
                "filter_name": filter_name,
                "selectivity": float(selected[0]["selectivity"]),
                "ef_search": int(selected[0]["ef_search"]),
                "max_scan_tuples": int(selected[0]["max_scan_tuples"]),
                "iterative_scan": selected[0]["iterative_scan"],
                "queries": len(deltas),
                "repeats": len(by_mode[MODES[0]]) // len(deltas),
                "d12_end_to_end_ms_mean": d12_mean,
                "d123_end_to_end_ms_mean": d123_mean,
                "d123_speedup_over_d12": d12_mean / d123_mean,
                "d123_minus_d12_query_cluster_mean_ms": statistics.fmean(
                    statistics.fmean(value) for value in deltas.values()
                ),
                "d123_minus_d12_ci95_low_ms": ci_low,
                "d123_minus_d12_ci95_high_ms": ci_high,
                "d12_activation_ms_mean": statistics.fmean(
                    float(row["activation_ms"]) for row in by_mode[MODES[0]]
                ),
                "d123_activation_ms_mean": statistics.fmean(
                    float(row["activation_ms"]) for row in by_mode[MODES[1]]
                ),
                "d12_query_latency_ms_mean": d12_query,
                "d123_query_latency_ms_mean": d123_query,
                "d123_query_speedup_over_d12": d12_query / d123_query,
                "d12_recall_mean": statistics.fmean(
                    float(row["recall"]) for row in by_mode[MODES[0]]
                ),
                "d123_recall_mean": statistics.fmean(
                    float(row["recall"]) for row in by_mode[MODES[1]]
                ),
                "d12_returned_tuples_mean": statistics.fmean(
                    float(row["returned_tuples"]) for row in by_mode[MODES[0]]
                ),
                "d123_returned_tuples_mean": statistics.fmean(
                    float(row["returned_tuples"]) for row in by_mode[MODES[1]]
                ),
                "all_measured_d3_warm": all(
                    row["d3_phase"] == "warm" for row in by_mode[MODES[1]]
                ),
                "all_results_identical": True,
                "statistically_positive": ci_high < 0,
            }
        )
    return summaries


def plan_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ControlError(f"missing child plan evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ControlError(f"child plan evidence is not complete: {path}")
    lifecycle = payload.get("execution_lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise ControlError(f"child lifecycle evidence is missing: {path}")
    if lifecycle.get("warmup_policy") != "admitted_warm_reuse":
        raise ControlError(f"child D3 lifecycle policy drifted: {path}")
    if lifecycle.get("d3_lifecycle_complete") is not True:
        raise ControlError(f"child D3 lifecycle did not complete: {path}")
    return {
        "path": str(path),
        "sha256": d2.sha256_file(path),
        "plan_checks": len(payload.get("checks") or []),
        "warmup_observed": lifecycle.get("warmup_observed"),
        "d3_phase_counts": lifecycle.get("d3_phase_counts"),
        "d3_warmup_phase_evidence_count": lifecycle.get(
            "d3_warmup_phase_evidence_count"
        ),
    }


def prewarm_bfs(
    args: argparse.Namespace, identities: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    arm = d2.Arm("d12_d123", MODES[0], "bfs", args.bfs_index)
    blocks = int(identities[args.bfs_index]["blocks"])
    return d2.prewarm_for_arm(args, arm, identities, blocks)


def protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": "sqlens-table6-d3-steady-increment-v1",
        "configuration_source": (
            str(args.matched_configs_csv)
            if args.matched_configs_csv is not None
            else str(args.matched_recall_manifest)
        ),
        "configuration_transfer": (
            "Table-6 D1 per-filter settings transferred to current same-graph BFS index; "
            "this is an incremental ablation, not independent retuning"
        ),
        "arms": list(MODES),
        "measurement": {
            "query_offset": args.query_offset,
            "queries": args.queries,
            "repeats": args.repeats,
            "execution_order": "seeded request-level interleaving",
            "latency": "activation plus SQL query end-to-end milliseconds",
            "guidance_max_atoms": args.guidance_max_atoms,
            "truth_self_excluded": args.expected_truth_self_excluded,
        },
        "cache": {
            "postgres_restart_before_each_filter": True,
            "full_bfs_index_prewarm_before_each_filter": True,
            "untimed_q_pass_per_arm": args.queries,
        },
        "d3": {
            "measurement_policy": "admitted_warm_reuse",
            "namespace_per_filter": True,
            "persistent_store_must_start_empty": True,
            "probe_requests": args.d3_probe_requests,
            "min_benefit_per_byte": args.d3_min_benefit_per_byte,
            "max_fragment_mb": args.d3_max_fragment_mb,
            "page_min_skip_rate": args.d3_page_min_skip_rate,
            "measured_builds_allowed": 0,
            "online_cost_reported_separately": True,
        },
    }


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ControlError(f"refusing to overwrite raw artifact: {args.out}")
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    if summary_path.exists() or manifest_path.exists():
        raise ControlError("refusing to overwrite summary or manifest artifact")

    truth = d2.audit_truth_for_args(args)
    configs = d2.load_matched_configs(args, args.filters_csv, args.truth_csv)
    run_id = uuid.uuid4().hex[:12]
    filter_order = list(args.filter_names)
    scheduled = list(filter_order)
    random.Random(args.schedule_seed).shuffle(scheduled)
    original_cpuset = d2.inspect_cpuset(args.container)
    manifest: dict[str, Any] = {
        "status": "running",
        "artifact_valid": False,
        "started_at": d2.utc_now(),
        "argv": sys.argv,
        "controller_sha256": d2.sha256_file(Path(__file__)),
        "runner_sha256": d2.sha256_file(RUNNER),
        "protocol": protocol(args),
        "run_id": run_id,
        "filter_order": filter_order,
        "scheduled_filter_order": scheduled,
        "exact_truth_audit": truth,
        "matched_config_source": d2.matched_config_source_evidence(args, configs),
        "matched_configs": {name: value.as_dict() for name, value in configs.items()},
        "original_container_cpuset": original_cpuset,
        "invocations": [],
    }
    d2.atomic_write_json(manifest_path, manifest)
    combined: list[dict[str, Any]] = []
    try:
        manifest["dedicated_server_gate"] = d2.require_dedicated_server()
        d2.set_container_cpu(args.container, args.backend_cpu)
        manifest["startup_restart"] = d2.restart_postgres(args)
        proof, identities = d2.prepare_database_contract(args)
        manifest["d2_graph_proof"] = proof
        manifest["index_identities_start"] = identities
        d2.atomic_write_json(manifest_path, manifest)

        for position, filter_name in enumerate(scheduled, start=1):
            namespace = namespace_for(run_id, filter_name)
            record: dict[str, Any] = {
                "position": position,
                "filter_name": filter_name,
                "namespace": namespace,
                "matched_config": configs[filter_name].as_dict(),
                "status": "running",
            }
            manifest["invocations"].append(record)
            d2.atomic_write_json(manifest_path, manifest)
            record["restart"] = d2.restart_postgres(args)
            record["prewarm"] = prewarm_bfs(args, identities)
            before_count = fragment_store_count(args, namespace)
            if before_count != 0:
                raise ControlError(
                    f"{filter_name}: D3 namespace did not start empty ({before_count} rows)"
                )
            record["fragment_store_rows_before"] = before_count
            child_out = child_path(args.out, position, filter_name)
            if child_out.exists():
                raise ControlError(f"refusing to overwrite child artifact: {child_out}")
            command = build_runner_command(
                args, child_out, filter_name, namespace, configs[filter_name]
            )
            record["runner_argv"] = command
            record["runner_shell"] = shlex.join(command)
            completed = d2.run_command(command)
            rows = d2.read_csv(child_out)
            validated = validate_child(
                rows, args, filter_name, namespace, configs[filter_name]
            )
            combined.extend(validated)
            after_count = fragment_store_count(args, namespace)
            d3_rows = [row for row in validated if row["mode"] == MODES[1]]
            d3_states = {str(row["d3_state_after"]) for row in d3_rows}
            if d3_states == {"exact"}:
                if after_count != 0:
                    raise ControlError(
                        f"{filter_name}: backend-resident exact admission unexpectedly persisted rows"
                    )
                persistence = "backend_resident_exact"
            else:
                if after_count <= 0:
                    raise ControlError(f"{filter_name}: D3 admission persisted no fragment")
                persistence = "persistent_page_or_bloom"
            plan = plan_evidence(child_out.with_suffix(child_out.suffix + ".plan.json"))
            record.update(
                {
                    "status": "complete",
                    "completed_at": d2.utc_now(),
                    "fragment_store_rows_after": after_count,
                    "d3_representation": sorted(d3_states),
                    "d3_persistence": persistence,
                    "plan_evidence": plan,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "artifact": {
                        "path": str(child_out),
                        "rows": len(rows),
                        "sha256": d2.sha256_file(child_out),
                    },
                }
            )
            d2.atomic_write_json(manifest_path, manifest)

        summaries = summarize(combined, filter_order, args.schedule_seed)
        d2.atomic_write_csv(args.out, combined)
        d2.atomic_write_csv(summary_path, summaries)
        proof_final, identities_final = d2.prepare_database_contract(args)
        if identities_final != identities:
            raise ControlError("index identity changed during the experiment")
        if proof_final["stable_fingerprint_sha256"] != proof["stable_fingerprint_sha256"]:
            raise ControlError("same-graph D2 proof changed during the experiment")
        speedups = [float(row["d123_speedup_over_d12"]) for row in summaries]
        manifest.update(
            {
                "status": "complete",
                "artifact_valid": True,
                "completed_at": d2.utc_now(),
                "index_identities_final": identities_final,
                "d2_graph_proof_final": proof_final,
                "all_results_identical": True,
                "all_measured_d3_warm": True,
                "positive_points": sum(
                    bool(row["statistically_positive"]) for row in summaries
                ),
                "geomean_d123_speedup_over_d12": math.exp(
                    statistics.fmean(math.log(value) for value in speedups)
                ),
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
        print(json.dumps({"manifest": str(manifest_path), "summary": summaries}, indent=2))
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "artifact_valid": False,
                "completed_at": d2.utc_now(),
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
            }
        )
        d2.atomic_write_json(manifest_path, manifest)
        raise
    finally:
        if d2.inspect_cpuset(args.container) != original_cpuset:
            d2.run_command(
                ["docker", "update", f"--cpuset-cpus={original_cpuset}", args.container]
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Table-6 D1+D2 versus admitted-warm D1+D2+D3 control."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--truth-csv", type=Path, default=d2.DEFAULT_TRUTH)
    parser.add_argument(
        "--truth-provenance-manifest",
        "--truth-manifest",
        dest="truth_manifest",
        type=Path,
        default=d2.DEFAULT_TRUTH_MANIFEST,
        help=(
            "Audited exact-truth manifest or an external-dataset launch manifest; "
            "--truth-manifest remains a compatibility alias."
        ),
    )
    parser.add_argument(
        "--matched-recall-manifest", type=Path, default=DEFAULT_MATCHED_MANIFEST
    )
    parser.add_argument(
        "--matched-configs-csv",
        type=Path,
        help="Current-build per-filter matched configuration CSV.",
    )
    parser.add_argument(
        "--matched-configs-manifest",
        type=Path,
        help="Fail-closed provenance manifest required with --matched-configs-csv.",
    )
    parser.add_argument("--d2-graph-proof-json", type=Path, default=DEFAULT_GRAPH_PROOF)
    parser.add_argument("--filter-names", nargs="+", default=list(DEFAULT_FILTER_NAMES))
    parser.add_argument("--table", default=d2.DEFAULT_TABLE)
    parser.add_argument("--source-index", default=d2.DEFAULT_SOURCE_INDEX)
    parser.add_argument("--bfs-index", default=d2.DEFAULT_BFS_INDEX)
    parser.add_argument("--query-table")
    parser.add_argument("--query-id-column", default="id")
    parser.add_argument("--query-vector-column", default="embedding")
    parser.add_argument("--candidate-validity-predicate", default="embedding_valid")
    parser.add_argument(
        "--expected-truth-self-excluded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require this exact self_excluded value in the truth provenance and CSV.",
    )
    parser.add_argument("--container", default="hybrid-pgvector")
    parser.add_argument("--backend-cpu", type=int, required=True)
    parser.add_argument("--queries", type=d2.positive_int, default=100)
    parser.add_argument("--query-offset", type=int, default=100)
    parser.add_argument("--repeats", type=d2.positive_int, default=5)
    parser.add_argument("--k", type=d2.positive_int, default=10)
    parser.add_argument("--schedule-seed", type=int, default=20260722)
    parser.add_argument("--d1-cache-mb", type=d2.positive_int, default=1024)
    parser.add_argument("--d3-cache-mb", type=d2.positive_int, default=1024)
    parser.add_argument(
        "--d1-guidance-kind", choices=("auto", "exact", "bloom"), default="auto"
    )
    parser.add_argument("--guidance-max-atoms", type=d2.positive_int, default=64)
    parser.add_argument("--d3-probe-requests", type=d2.positive_int, default=2)
    parser.add_argument("--d3-min-benefit-per-byte", type=float, default=0.0)
    parser.add_argument("--d3-max-fragment-mb", type=d2.positive_int, default=256)
    parser.add_argument("--d3-page-min-skip-rate", type=float, default=0.80)
    parser.add_argument("--statement-timeout-ms", type=d2.positive_int, default=300_000)
    parser.add_argument("--progress-queries", type=int, default=0)
    parser.add_argument("--readiness-timeout-s", type=float, default=60.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--expected-sqlens-build-id", required=True)
    parser.add_argument("--expected-vector-so-sha256", type=d2.sha256_arg, required=True)
    parser.add_argument("--expected-candidate-rows", type=d2.positive_int, default=9_979_556)
    parser.add_argument(
        "--allow-mean-qualified-matched-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Accept Table-6 settings that met and confirmed the mean-recall target "
            "without an LCB95 crossing; qualification is retained in every output row."
        ),
    )
    args = parser.parse_args(argv)
    args.matched_mode = "design1_bloom"
    args.matched_target_recall = 0.90
    args.matched_config_index_policy = "same_table_borrowed"
    args.live_graph_proof_policy = "delegated_immutable"
    args.cache_regime = "warm_resident"
    args.prewarm_index_blocks = None
    args.prewarm_common_relation = []
    if args.backend_cpu < 0:
        parser.error("--backend-cpu must be nonnegative")
    if args.query_offset < 0:
        parser.error("--query-offset must be nonnegative")
    if not math.isfinite(args.d3_min_benefit_per_byte) or args.d3_min_benefit_per_byte < 0:
        parser.error("--d3-min-benefit-per-byte must be finite and nonnegative")
    if not 0 <= args.d3_page_min_skip_rate <= 1:
        parser.error("--d3-page-min-skip-rate must be within [0,1]")
    if len(set(args.filter_names)) != len(args.filter_names):
        parser.error("--filter-names contains duplicates")
    if (args.matched_configs_csv is None) != (args.matched_configs_manifest is None):
        parser.error(
            "--matched-configs-csv and --matched-configs-manifest must be supplied together"
        )
    required_paths = [
        args.filters_csv,
        args.truth_csv,
        args.truth_manifest,
        args.d2_graph_proof_json,
        RUNNER,
    ]
    if args.matched_configs_csv is not None:
        required_paths.extend([args.matched_configs_csv, args.matched_configs_manifest])
    else:
        required_paths.append(args.matched_recall_manifest)
    for path in required_paths:
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
