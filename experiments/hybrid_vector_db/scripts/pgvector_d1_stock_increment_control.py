"""Formal, dataset-independent Stock-versus-D1 paired control.

Each filter is measured after a PostgreSQL restart and a complete source-HNSW
prewarm.  The delegated benchmark opens one backend per arm and executes the
configured Stock and D1 requests in a balanced, request-level interleaving.
The parent revalidates exact-truth coverage, plans, relation identity, binary
identity, CPU affinity, search settings, recall targets, and paired coverage
before publishing raw, summary, and manifest artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import pgvector_d2_cache_isolation_control as d2
except ImportError:
    import pgvector_d2_cache_isolation_control as d2


RUNNER = Path(__file__).with_name(
    "pgvector_design1_design2_design3_selectivity_benchmark.py"
)
MODES = ("original", "design1_bloom")
ITERATIVE_SCAN_VALUES = {"off", "strict_order", "relaxed_order"}


class ControlError(RuntimeError):
    """A formal Stock-versus-D1 experiment contract was not established."""


@dataclass(frozen=True)
class SearchConfig:
    filter_name: str
    mode: str
    target_recall: float
    ef_search: int
    max_scan_tuples: int
    scan_mem_multiplier: float
    iterative_scan: str
    qualification: str = "lcb95"
    calibration_recall_mean: float | None = None
    calibration_recall_lcb95: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def runner_dict(self) -> dict[str, object]:
        return {
            "ef_search": self.ef_search,
            "max_scan_tuples": self.max_scan_tuples,
            "scan_mem_multiplier": self.scan_mem_multiplier,
            "iterative_scan": self.iterative_scan,
            "guided_collect_target": 1,
            # This control measures safe candidate validation.  Do not inherit
            # the generic runner's traversal-prioritization default.
            "traversal_guided_prioritization": False,
        }


def canonical_relation(value: object) -> str:
    return d2.canonical_public_relation(value)


def parse_bool(value: object) -> bool:
    try:
        return d2.parse_bool(value)
    except d2.ControlError as exc:
        raise ControlError(str(exc)) from exc


def effective_candidate_validity_predicate(value: object = "") -> str:
    """Mirror the delegated runner's empty-expression normalization."""
    predicate = str(value or "").strip()
    forbidden = (";", "--", "/*", "*/", "\x00")
    token = next((item for item in forbidden if item in predicate), None)
    if token is not None:
        raise ControlError(
            "candidate validity predicate must be one comment-free expression; "
            f"found {token!r}"
        )
    return predicate or "TRUE"


def _finite_float(row: Mapping[str, object], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ControlError(f"invalid {field}: {row.get(field)!r}") from exc
    if not math.isfinite(value):
        raise ControlError(f"invalid {field}: {value!r}")
    return value


def load_configs(
    path: Path,
    allow_mean_qualified: bool = False,
) -> tuple[dict[str, dict[str, SearchConfig]], list[str]]:
    required = {
        "filter_name",
        "mode",
        "target_recall",
        "ef_search",
        "max_scan_tuples",
        "scan_mem_multiplier",
        "iterative_scan",
        "qualification",
        "calibration_recall_mean",
        "calibration_recall_lcb95",
    }
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ControlError(f"config CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ControlError("config CSV is empty")

    configs: dict[str, dict[str, SearchConfig]] = defaultdict(dict)
    order: list[str] = []
    for row in rows:
        name = str(row.get("filter_name") or "").strip()
        mode = str(row.get("mode") or "").strip()
        if not name or mode not in MODES:
            raise ControlError(f"invalid filter/mode in config row: {name!r}/{mode!r}")
        if name not in configs:
            order.append(name)
        if mode in configs[name]:
            raise ControlError(f"duplicate config for filter={name} mode={mode}")
        try:
            config = SearchConfig(
                filter_name=name,
                mode=mode,
                target_recall=float(row["target_recall"]),
                ef_search=int(row["ef_search"]),
                max_scan_tuples=int(row["max_scan_tuples"]),
                scan_mem_multiplier=float(row["scan_mem_multiplier"]),
                iterative_scan=str(row["iterative_scan"]),
                qualification=str(row["qualification"]),
                calibration_recall_mean=float(row["calibration_recall_mean"]),
                calibration_recall_lcb95=float(row["calibration_recall_lcb95"]),
            )
        except (TypeError, ValueError) as exc:
            raise ControlError(f"invalid search config for {name}/{mode}") from exc
        lcb_qualified = (
            config.qualification == "lcb95"
            and config.calibration_recall_lcb95 is not None
            and math.isfinite(config.calibration_recall_lcb95)
            and config.calibration_recall_lcb95 >= config.target_recall
        )
        mean_qualified = (
            allow_mean_qualified
            and config.qualification == "mean_confirmed"
            and config.calibration_recall_mean is not None
            and math.isfinite(config.calibration_recall_mean)
            and config.calibration_recall_mean >= config.target_recall
        )
        if (
            not math.isfinite(config.target_recall)
            or not 0.0 <= config.target_recall <= 1.0
            or config.ef_search <= 0
            or config.max_scan_tuples <= 0
            or not math.isfinite(config.scan_mem_multiplier)
            or config.scan_mem_multiplier <= 0
            or config.iterative_scan not in ITERATIVE_SCAN_VALUES
            or config.calibration_recall_mean is None
            or not math.isfinite(config.calibration_recall_mean)
            or config.calibration_recall_lcb95 is None
            or not math.isfinite(config.calibration_recall_lcb95)
            or not (lcb_qualified or mean_qualified)
        ):
            raise ControlError(f"out-of-range search config for {name}/{mode}")
        configs[name][mode] = config

    for name, by_mode in configs.items():
        if set(by_mode) != set(MODES):
            raise ControlError(
                f"filter={name} must contain exactly {list(MODES)}, got {sorted(by_mode)}"
            )
        targets = {by_mode[mode].target_recall for mode in MODES}
        if len(targets) != 1:
            raise ControlError(f"filter={name} arms do not use one matched recall target")
    return dict(configs), order


def select_filters(
    filters_csv: Path,
    configs: Mapping[str, Mapping[str, SearchConfig]],
    requested: Sequence[str] | None,
) -> tuple[list[str], dict[str, float]]:
    runner = d2.load_runner()
    specs, _ = runner.load_filter_specs(filters_csv)
    filter_rates: dict[str, float] = {}
    source_order: list[str] = []
    for name, rate, _ in specs:
        if name in filter_rates:
            raise ControlError(f"filters CSV has duplicate filter={name}")
        filter_rates[name] = runner.parse_pct(rate)
        source_order.append(name)
    wanted = list(requested) if requested else [name for name in source_order if name in configs]
    if not wanted:
        raise ControlError("no filters selected")
    if len(set(wanted)) != len(wanted):
        raise ControlError("filter selection contains duplicates")
    missing_specs = sorted(set(wanted) - set(filter_rates))
    missing_configs = sorted(set(wanted) - set(configs))
    if missing_specs:
        raise ControlError(f"filters CSV lacks selected filters: {missing_specs}")
    if missing_configs:
        raise ControlError(f"config CSV lacks selected filters: {missing_configs}")
    return wanted, {name: filter_rates[name] for name in wanted}


def audit_truth(
    args: argparse.Namespace, filter_order: Sequence[str]
) -> tuple[dict[str, Any], set[int]]:
    try:
        provenance = d2.audit_exact_truth_manifest(
            args.truth_manifest,
            args.truth_csv,
            args.filters_csv,
            expected_table=args.table,
            expected_index=args.source_index,
            expected_query_table=args.query_table,
            expected_query_id_column=args.query_id_column,
            expected_query_vector_column=args.query_vector_column,
            expected_candidate_validity_predicate=args.candidate_validity_predicate,
            expected_self_excluded=args.expected_truth_self_excluded,
            query_offset=args.query_offset,
            queries=args.queries,
            expected_filter_names=filter_order,
        )
    except d2.ControlError as exc:
        raise ControlError(f"exact-truth provenance audit failed: {exc}") from exc
    runner = d2.load_runner()
    try:
        truth, query_by_no = runner.load_tie_aware_truth(
            args.truth_csv,
            expected_self_excluded=args.expected_truth_self_excluded,
            expected_candidate_validity_predicate=args.candidate_validity_predicate,
        )
    except Exception as exc:  # noqa: BLE001 - normalize delegated audit failures
        raise ControlError(f"exact-truth audit failed: {exc}") from exc
    all_query_nos = sorted(query_by_no)
    selected = all_query_nos[args.query_offset : args.query_offset + args.queries]
    if len(selected) != args.queries:
        raise ControlError(
            f"truth query slice offset={args.query_offset} count={args.queries} "
            f"contains only {len(selected)} queries"
        )
    expected_cells = {
        (filter_name, query_no)
        for filter_name in filter_order
        for query_no in selected
    }
    observed_cells = set(truth).intersection(expected_cells)
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)[:5]
        raise ControlError(f"truth does not cover every selected filter/query cell: {missing}")
    return (
        {
            "path": str(args.truth_csv.resolve()),
            "sha256": d2.sha256_file(args.truth_csv),
            "recall_contract": "distance_squared_threshold_tie_aware_v1",
            "self_excluded": args.expected_truth_self_excluded,
            "candidate_validity_predicate": runner.effective_candidate_validity_predicate(
                args.candidate_validity_predicate
            ),
            "query_offset_semantics": "ordinal slice over sorted truth query_no values",
            "query_offset": args.query_offset,
            "queries": args.queries,
            "query_nos": selected,
            "query_ids": [query_by_no[query_no] for query_no in selected],
            "filter_count": len(filter_order),
            "covered_cells": len(expected_cells),
            "provenance_manifest": provenance,
        },
        set(selected),
    )


def audit_config_provenance(
    args: argparse.Namespace,
    filter_order: Sequence[str],
    configs: Mapping[str, Mapping[str, SearchConfig]],
) -> dict[str, Any]:
    """Bind both D1 arms to the current-build LCB-qualified tuner output."""
    args.filter_names = list(filter_order)
    try:
        audited_d1 = d2.audit_matched_configs_csv(
            args.config_csv,
            args.config_manifest,
            args,
            args.filters_csv,
            args.truth_csv,
        )
    except d2.ControlError as exc:
        raise ControlError(f"matched-config provenance audit failed: {exc}") from exc
    for name in filter_order:
        expected = configs[name]["design1_bloom"]
        observed = audited_d1[name]
        for field in (
            "target_recall",
            "ef_search",
            "max_scan_tuples",
            "scan_mem_multiplier",
            "iterative_scan",
            "qualification",
            "calibration_recall_mean",
            "calibration_recall_lcb95",
        ):
            if getattr(observed, field) != getattr(expected, field):
                raise ControlError(
                    f"audited D1 config mismatch for filter={name} field={field}"
                )
        for mode in MODES:
            config = configs[name][mode]
            if not math.isclose(
                config.target_recall,
                args.matched_target_recall,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ControlError(
                    f"filter={name} mode={mode} target differs from "
                    f"--matched-target-recall={args.matched_target_recall}"
                )
    return {
        "path": str(args.config_manifest.resolve()),
        "sha256": d2.sha256_file(args.config_manifest),
        "artifact_valid": True,
        "qualification": (
            "LCB95, with explicitly admitted grid-ceiling mean-confirmed rows"
            if getattr(args, "allow_mean_qualified_matched_config", False)
            else "one-sided bootstrap Recall@10 LCB95"
        ),
        "target_recall": args.matched_target_recall,
        "filter_count": len(filter_order),
        "modes": list(MODES),
    }


def inspect_source_index(cur: Any, args: argparse.Namespace) -> dict[str, Any]:
    runner = d2.load_runner()
    cur.execute(
        "SELECT c.oid::bigint, n.nspname || '.' || c.relname, "
        "c.relfilenode::bigint, i.indrelid::bigint, "
        "tn.nspname || '.' || t.relname, pg_relation_size(c.oid)::bigint, "
        "current_setting('block_size')::bigint, am.amname, "
        "i.indisvalid, i.indisready, i.indislive, "
        "pg_get_expr(i.indpred, i.indrelid) "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_index i ON i.indexrelid=c.oid "
        "JOIN pg_class t ON t.oid=i.indrelid "
        "JOIN pg_namespace tn ON tn.oid=t.relnamespace "
        "JOIN pg_am am ON am.oid=c.relam WHERE c.oid=%s::regclass",
        (args.source_index,),
    )
    row = cur.fetchone()
    if not row:
        raise ControlError(f"source index does not exist: {args.source_index}")
    size = int(row[5])
    block_size = int(row[6])
    identity = {
        "oid": int(row[0]),
        "name": str(row[1]),
        "relfilenode": int(row[2]),
        "heap_oid": int(row[3]),
        "table": str(row[4]),
        "size_bytes": size,
        "block_size": block_size,
        "blocks": (size + block_size - 1) // block_size,
        "access_method": str(row[7]),
        "indisvalid": bool(row[8]),
        "indisready": bool(row[9]),
        "indislive": bool(row[10]),
        "predicate": row[11],
    }
    if identity["name"] != args.source_index or identity["table"] != args.table:
        raise ControlError(f"source index relation binding mismatch: {identity}")
    if (
        identity["access_method"] != "hnsw"
        or identity["blocks"] <= 0
        or not all(identity[field] for field in ("indisvalid", "indisready", "indislive"))
    ):
        raise ControlError(f"source index is not a live non-empty HNSW index: {identity}")
    if not runner.candidate_validity_index_predicate_matches(
        identity["predicate"], args.candidate_validity_predicate
    ):
        raise ControlError(
            "source index predicate does not match --candidate-validity-predicate: "
            f"catalog={identity['predicate']!r} expected={args.candidate_validity_predicate!r}"
        )
    return identity


def prepare_source_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    runner = d2.load_runner()
    connection = runner.psycopg.connect(
        runner.pg_config_from_env().conninfo, autocommit=True
    )
    try:
        cur = connection.cursor()
        runtime = runner.require_exact_sqlens_identity(
            cur, args.expected_sqlens_build_id, args.expected_vector_so_sha256
        )
        runner.ensure_functions(cur)
        runner.ensure_tracking(cur, args.table)
        return inspect_source_index(cur, args), runtime
    finally:
        connection.close()


def prewarm_source(
    args: argparse.Namespace, expected_identity: Mapping[str, Any]
) -> dict[str, Any]:
    runner = d2.load_runner()
    connection = runner.psycopg.connect(
        runner.pg_config_from_env().conninfo, autocommit=True
    )
    try:
        cur = connection.cursor()
        runtime = runner.require_exact_sqlens_identity(
            cur, args.expected_sqlens_build_id, args.expected_vector_so_sha256
        )
        observed = inspect_source_index(cur, args)
        if observed != dict(expected_identity):
            raise ControlError("source index identity or size changed before prewarm")
        cur.execute(
            "SELECT pg_prewarm(%s::regclass, 'read', 'main')::bigint",
            (args.source_index,),
        )
        warmed = int(cur.fetchone()[0])
        expected_blocks = int(expected_identity["blocks"])
        if warmed != expected_blocks:
            raise ControlError(
                f"full source-index prewarm mismatch: expected={expected_blocks}, got={warmed}"
            )
        return {
            "completed_at": d2.utc_now(),
            "target_index": args.source_index,
            "mode": "read",
            "fork": "main",
            "first_block": 0,
            "last_block": expected_blocks - 1,
            "blocks": warmed,
            "coverage_ratio": 1.0,
            "target_index_prewarmed": True,
            "backend": "dedicated prewarm connection closed before measured backends",
            "sqlens_runtime_identity": runtime,
        }
    finally:
        connection.close()


def child_path(out: Path, position: int, filter_name: str) -> Path:
    safe = "".join(char if char.isalnum() or char in "_.-" else "_" for char in filter_name)
    if not safe:
        raise ControlError(f"filter name cannot form an artifact path: {filter_name!r}")
    return out.with_name(f"{out.stem}.f{position:02d}.{safe[:80]}{out.suffix}")


def child_artifacts(path: Path) -> tuple[Path, ...]:
    return (
        path,
        path.with_suffix(path.suffix + ".plan.json"),
        path.with_name(path.stem + "_table.csv"),
        path.with_name(path.stem + "_profile.csv"),
    )


def build_runner_command(
    args: argparse.Namespace,
    child_out: Path,
    filter_name: str,
    configs: Mapping[str, SearchConfig],
) -> list[str]:
    mode_configs = {mode: configs[mode].runner_dict() for mode in MODES}
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
        args.source_index,
        "--candidate-validity-predicate",
        args.candidate_validity_predicate,
        "--modes",
        *MODES,
        "--execution-order",
        "interleaved",
        "--schedule-seed",
        str(args.schedule_seed),
        "--mode-configs-json",
        json.dumps(mode_configs, sort_keys=True),
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
        "--d1-guidance-kind",
        args.d1_guidance_kind,
        "--d1-exact-max-selectivity-pct",
        str(args.d1_exact_max_selectivity_pct),
        "--d1-cache-mb",
        str(args.d1_cache_mb),
        "--guidance-selectivity-max-pct",
        "100",
        "--guidance-max-atoms",
        str(args.guidance_max_atoms),
        "--d2-page-access",
        "off",
        "--d2-index-page-access",
        "off",
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
        "--query-id-column",
        args.query_id_column,
        "--query-vector-column",
        args.query_vector_column,
    ]
    command.append(
        "--expected-truth-self-excluded"
        if args.expected_truth_self_excluded
        else "--no-expected-truth-self-excluded"
    )
    if args.query_table:
        command.extend(["--query-table", args.query_table])
    return command


def _validate_row_config(row: Mapping[str, str], config: SearchConfig) -> None:
    for field in ("ef_search", "max_scan_tuples"):
        if int(row[field]) != getattr(config, field):
            raise ControlError(
                f"{config.filter_name}/{config.mode} used unexpected {field}"
            )
    if not math.isclose(
        float(row["scan_mem_multiplier"]), config.scan_mem_multiplier, abs_tol=1e-12
    ):
        raise ControlError(
            f"{config.filter_name}/{config.mode} used unexpected scan_mem_multiplier"
        )
    if row["iterative_scan"] != config.iterative_scan:
        raise ControlError(
            f"{config.filter_name}/{config.mode} used unexpected iterative_scan"
        )


def validate_child(
    rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
    filter_name: str,
    configs: Mapping[str, SearchConfig],
    expected_query_nos: set[int],
) -> list[dict[str, Any]]:
    expected_rows = args.queries * args.repeats * len(MODES)
    if len(rows) != expected_rows:
        raise ControlError(
            f"{filter_name}: expected {expected_rows} rows, observed {len(rows)}"
        )
    expected_predicate = effective_candidate_validity_predicate(
        args.candidate_validity_predicate
    )
    expected_query_table = args.query_table or args.table
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    per_mode: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    coverage: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        mode = str(row.get("mode") or "")
        if row.get("filter_name") != filter_name or mode not in MODES:
            raise ControlError(f"{filter_name}: child contains an unexpected arm/filter")
        if row.get("error"):
            raise ControlError(
                f"{filter_name}/{mode}: measured request failed: {row.get('error_detail')}"
            )
        if canonical_relation(row.get("table")) != args.table:
            raise ControlError(f"{filter_name}/{mode}: candidate table drifted")
        if canonical_relation(row.get("index")) != args.source_index:
            raise ControlError(f"{filter_name}/{mode}: source index was not used")
        if canonical_relation(row.get("query_table")) != expected_query_table:
            raise ControlError(f"{filter_name}/{mode}: query table drifted")
        if row.get("query_id_column") != args.query_id_column or row.get(
            "query_vector_column"
        ) != args.query_vector_column:
            raise ControlError(f"{filter_name}/{mode}: query columns drifted")
        if effective_candidate_validity_predicate(
            row.get("candidate_validity_predicate", "")
        ) != expected_predicate:
            raise ControlError(f"{filter_name}/{mode}: candidate-validity predicate drifted")
        for field in (
            "planner_proof_verified",
            "backend_cpu_exact_match",
            "guidance_scan_verified",
            "guidance_binding_verified",
        ):
            if not parse_bool(row.get(field)):
                raise ControlError(f"{filter_name}/{mode}: {field} is false")
        if row.get("backend_cpu_requested") != str(args.backend_cpu) or row.get(
            "backend_cpu_observed"
        ) != str(args.backend_cpu):
            raise ControlError(f"{filter_name}/{mode}: backend CPU affinity drifted")
        if row.get("sqlens_build_id") != args.expected_sqlens_build_id:
            raise ControlError(f"{filter_name}/{mode}: SQLens build ID drifted")
        if row.get("vector_so_sha256") != args.expected_vector_so_sha256:
            raise ControlError(f"{filter_name}/{mode}: vector.so SHA256 drifted")
        if row.get("guidance_filter_strategy") != "safe_guided":
            raise ControlError(f"{filter_name}/{mode}: strategy is not safe_guided")
        guidance_enabled = parse_bool(row.get("guidance_enabled"))
        expected_path = "stock" if mode == "original" else "validation_only"
        if guidance_enabled != (mode == "design1_bloom"):
            raise ControlError(f"{filter_name}/{mode}: guidance activation drifted")
        if row.get("final_path") != expected_path:
            raise ControlError(
                f"{filter_name}/{mode}: expected final_path={expected_path!r}, "
                f"observed={row.get('final_path')!r}"
            )
        if not parse_bool(row.get("warmup_all_queries")):
            raise ControlError(f"{filter_name}/{mode}: full-query warmup was not enabled")
        if parse_bool(row.get("truth_self_excluded")) != args.expected_truth_self_excluded:
            raise ControlError(f"{filter_name}/{mode}: truth self-exclusion drifted")
        recall = _finite_float(row, "recall")
        if not 0.0 <= recall <= 1.0:
            raise ControlError(f"{filter_name}/{mode}: recall is outside [0,1]")
        latency = _finite_float(row, "end_to_end_ms")
        if latency <= 0:
            raise ControlError(f"{filter_name}/{mode}: non-positive end-to-end latency")
        _validate_row_config(row, configs[mode])
        query_no = int(row["query_no"])
        repeat = int(row["repeat"])
        if query_no not in expected_query_nos or not 0 <= repeat < args.repeats:
            raise ControlError(f"{filter_name}/{mode}: query/repeat coverage drifted")
        coverage[(mode, repeat)].add(query_no)
        grouped[str(row["pair_key"])].append(row)
        per_mode[mode].append(row)

    for mode in MODES:
        for repeat in range(args.repeats):
            if coverage[(mode, repeat)] != expected_query_nos:
                raise ControlError(f"{filter_name}/{mode}: incomplete q/r coverage")
    if len(grouped) != args.queries * args.repeats:
        raise ControlError(f"{filter_name}: paired-key coverage is incomplete")
    for pair_key, pair in grouped.items():
        if len(pair) != 2 or {row["mode"] for row in pair} != set(MODES):
            raise ControlError(f"{filter_name}: incomplete pair {pair_key}")
        values = {row["mode"]: row for row in pair}
        for field in (
            "query_no",
            "query_id",
            "repeat",
            "truth_filtered_rows",
            "truth_kth_distance_sq",
            "truth_tie_tolerance",
            "truth_self_excluded",
        ):
            if values[MODES[0]].get(field) != values[MODES[1]].get(field):
                raise ControlError(f"{filter_name}: pair {pair_key} changed {field}")

    for mode in MODES:
        mean_recall = statistics.fmean(float(row["recall"]) for row in per_mode[mode])
        if mean_recall + args.recall_tolerance < configs[mode].target_recall:
            raise ControlError(
                f"{filter_name}/{mode}: mean recall {mean_recall:.6f} is below "
                f"target {configs[mode].target_recall:.6f}"
            )
    return [
        {
            **row,
            "configured_target_recall": configs[str(row["mode"])].target_recall,
            "config_source": str(args.config_csv),
        }
        for row in rows
    ]


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(probability * (len(ordered) - 1))))
    return ordered[index]


def paired_bootstrap(
    pairs: Mapping[int, Sequence[tuple[float, float, float, float]]],
    seed: int,
    samples: int,
) -> dict[str, float]:
    clusters = []
    for query_no in sorted(pairs):
        values = pairs[query_no]
        clusters.append(
            tuple(
                statistics.fmean(item[position] for item in values)
                for position in range(4)
            )
        )
    if not clusters:
        raise ControlError("paired bootstrap has no query clusters")
    rng = random.Random(seed)
    latency_deltas: list[float] = []
    speedups: list[float] = []
    recall_deltas: list[float] = []
    for _ in range(samples):
        draw = [rng.choice(clusters) for _ in clusters]
        stock_latency = statistics.fmean(item[0] for item in draw)
        d1_latency = statistics.fmean(item[1] for item in draw)
        latency_deltas.append(d1_latency - stock_latency)
        speedups.append(stock_latency / d1_latency)
        recall_deltas.append(
            statistics.fmean(item[3] - item[2] for item in draw)
        )
    return {
        "d1_minus_stock_latency_ci95_low_ms": _percentile(latency_deltas, 0.025),
        "d1_minus_stock_latency_ci95_high_ms": _percentile(latency_deltas, 0.975),
        "d1_speedup_ci95_low": _percentile(speedups, 0.025),
        "d1_speedup_ci95_high": _percentile(speedups, 0.975),
        "d1_minus_stock_recall_ci95_low": _percentile(recall_deltas, 0.025),
        "d1_minus_stock_recall_ci95_high": _percentile(recall_deltas, 0.975),
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    filter_order: Sequence[str],
    configs: Mapping[str, Mapping[str, SearchConfig]],
    filter_rates: Mapping[str, float],
    seed: int,
    bootstrap_samples: int,
    recall_tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for position, filter_name in enumerate(filter_order):
        selected = [row for row in rows if row["filter_name"] == filter_name]
        by_mode = {
            mode: [row for row in selected if row["mode"] == mode] for mode in MODES
        }
        if any(not values for values in by_mode.values()):
            raise ControlError(f"{filter_name}: summary arm is empty")
        stock_by_key = {str(row["pair_key"]): row for row in by_mode["original"]}
        pairs: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        for d1_row in by_mode["design1_bloom"]:
            stock_row = stock_by_key[str(d1_row["pair_key"])]
            pairs[int(d1_row["query_no"])].append(
                (
                    float(stock_row["end_to_end_ms"]),
                    float(d1_row["end_to_end_ms"]),
                    float(stock_row["recall"]),
                    float(d1_row["recall"]),
                )
            )
        stock_latency = statistics.fmean(
            float(row["end_to_end_ms"]) for row in by_mode["original"]
        )
        d1_latency = statistics.fmean(
            float(row["end_to_end_ms"]) for row in by_mode["design1_bloom"]
        )
        stock_recall = statistics.fmean(
            float(row["recall"]) for row in by_mode["original"]
        )
        d1_recall = statistics.fmean(
            float(row["recall"]) for row in by_mode["design1_bloom"]
        )
        bootstrap = paired_bootstrap(
            pairs, seed + position, bootstrap_samples
        )
        stock_config = configs[filter_name]["original"]
        d1_config = configs[filter_name]["design1_bloom"]
        summaries.append(
            {
                "filter_name": filter_name,
                "selectivity": filter_rates[filter_name],
                "target_recall": stock_config.target_recall,
                "queries": len(pairs),
                "repeats": len(by_mode["original"]) // len(pairs),
                "stock_ef_search": stock_config.ef_search,
                "stock_max_scan_tuples": stock_config.max_scan_tuples,
                "stock_scan_mem_multiplier": stock_config.scan_mem_multiplier,
                "stock_iterative_scan": stock_config.iterative_scan,
                "d1_ef_search": d1_config.ef_search,
                "d1_max_scan_tuples": d1_config.max_scan_tuples,
                "d1_scan_mem_multiplier": d1_config.scan_mem_multiplier,
                "d1_iterative_scan": d1_config.iterative_scan,
                "stock_end_to_end_ms_mean": stock_latency,
                "d1_end_to_end_ms_mean": d1_latency,
                "d1_minus_stock_latency_query_cluster_mean_ms": statistics.fmean(
                    statistics.fmean(item[1] - item[0] for item in values)
                    for values in pairs.values()
                ),
                "d1_speedup_over_stock": stock_latency / d1_latency,
                "stock_query_latency_ms_mean": statistics.fmean(
                    float(row["query_latency_ms"]) for row in by_mode["original"]
                ),
                "d1_query_latency_ms_mean": statistics.fmean(
                    float(row["query_latency_ms"]) for row in by_mode["design1_bloom"]
                ),
                "stock_activation_ms_mean": statistics.fmean(
                    float(row["activation_ms"]) for row in by_mode["original"]
                ),
                "d1_activation_ms_mean": statistics.fmean(
                    float(row["activation_ms"]) for row in by_mode["design1_bloom"]
                ),
                "stock_recall_mean": stock_recall,
                "d1_recall_mean": d1_recall,
                "d1_minus_stock_recall_mean": d1_recall - stock_recall,
                "stock_recall_target_met": (
                    stock_recall + recall_tolerance >= stock_config.target_recall
                ),
                "d1_recall_target_met": (
                    d1_recall + recall_tolerance >= d1_config.target_recall
                ),
                "statistically_positive": bootstrap[
                    "d1_minus_stock_latency_ci95_high_ms"
                ] < 0,
                **bootstrap,
            }
        )
    return summaries


def validate_plan_evidence(
    path: Path,
    child_out: Path,
    args: argparse.Namespace,
    filter_name: str,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise ControlError(f"missing child plan evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ControlError(f"child plan evidence is not complete: {path}")
    if Path(str(payload.get("output"))).resolve() != child_out.resolve():
        raise ControlError(f"child plan evidence output path drifted: {path}")
    expected_rows = args.queries * args.repeats * len(MODES)
    if int(payload.get("output_rows", -1)) != expected_rows or payload.get(
        "output_sha256"
    ) != d2.sha256_file(child_out):
        raise ControlError(f"child plan evidence output identity drifted: {path}")
    query_contract = payload.get("query_contract")
    if not isinstance(query_contract, Mapping):
        raise ControlError(f"child query contract is missing: {path}")
    expected_query_table = args.query_table or "candidate_table_per_mode"
    expected_predicate = effective_candidate_validity_predicate(
        args.candidate_validity_predicate
    )
    if (
        query_contract.get("query_table") != expected_query_table
        or query_contract.get("query_id_column") != args.query_id_column
        or query_contract.get("query_vector_column") != args.query_vector_column
        or query_contract.get("self_excluded") is not args.expected_truth_self_excluded
        or effective_candidate_validity_predicate(
            query_contract.get("candidate_validity_predicate", "")
        )
        != expected_predicate
    ):
        raise ControlError(f"child query contract drifted: {path}")
    checks = payload.get("checks")
    if not isinstance(checks, list) or len(checks) != len(MODES):
        raise ControlError(f"child plan checks are incomplete: {path}")
    by_mode = {str(check.get("mode")): check for check in checks}
    if set(by_mode) != set(MODES):
        raise ControlError(f"child plan modes drifted: {path}")
    for mode, check in by_mode.items():
        cpu = check.get("backend_cpu_provenance")
        runtime = check.get("sqlens_runtime_identity")
        if (
            check.get("passed") is not True
            or check.get("filter_name") != filter_name
            or check.get("expected_table_identity") != args.table
            or check.get("expected_index_identity") != args.source_index
            or int(check.get("expected_index_oid", 0)) != int(identity["oid"])
            or int(check.get("catalog_index_oid", 0)) != int(identity["oid"])
            or check.get("preferred_index_current_setting") != args.source_index
            or check.get("catalog_index_predicate_matches") is not True
            or effective_candidate_validity_predicate(
                check.get("candidate_validity_predicate", "")
            )
            != expected_predicate
            or not isinstance(cpu, Mapping)
            or cpu.get("exact_match") is not True
            or cpu.get("requested_cpu_list") != str(args.backend_cpu)
            or cpu.get("observed_cpu_list") != str(args.backend_cpu)
            or not isinstance(runtime, Mapping)
            or runtime.get("exact_match") is not True
            or runtime.get("expected_build_id") != args.expected_sqlens_build_id
            or runtime.get("expected_vector_so_sha256")
            != args.expected_vector_so_sha256
        ):
            raise ControlError(f"child plan/build/index/CPU gate failed for {mode}: {path}")
    lifecycle = payload.get("execution_lifecycle")
    if (
        not isinstance(lifecycle, Mapping)
        or lifecycle.get("warmup_complete") is not True
        or lifecycle.get("backend_cpu_provenance_complete") is not True
        or lifecycle.get("runtime_sqlens_identity_complete") is not True
        or int(lifecycle.get("warmup_observed", -1)) != args.queries * len(MODES)
    ):
        raise ControlError(f"child execution lifecycle is incomplete: {path}")
    return {
        "path": str(path),
        "sha256": d2.sha256_file(path),
        "status": "complete",
        "checks": len(checks),
        "modes": list(MODES),
        "index_oid": int(identity["oid"]),
        "warmup_observed": int(lifecycle["warmup_observed"]),
    }


def protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": "sqlens-stock-d1-paired-interleaved-v1",
        "arms": list(MODES),
        "configuration": {
            "source": str(args.config_csv),
            "manifest": str(args.config_manifest),
            "per_filter_per_mode": True,
            "matched_recall_target_required": True,
            "qualification": (
                "LCB95, with explicitly admitted grid-ceiling mean-confirmed rows"
                if getattr(args, "allow_mean_qualified_matched_config", False)
                else "one-sided bootstrap Recall@10 LCB95"
            ),
        },
        "measurement": {
            "query_offset": args.query_offset,
            "queries": args.queries,
            "repeats": args.repeats,
            "execution_order": "balanced seeded request-level interleaving",
            "latency": "activation plus SQL query end-to-end milliseconds",
            "paired_bootstrap_samples": args.bootstrap_samples,
        },
        "cache": {
            "postgres_restart_before_each_filter": True,
            "complete_source_index_prewarm_before_each_filter": True,
            "untimed_full_query_pass_per_arm": args.queries,
        },
        "d1": {
            "guidance_filter_strategy": "safe_guided",
            "guidance_kind": args.d1_guidance_kind,
            "exact_max_selectivity_pct": args.d1_exact_max_selectivity_pct,
            "guidance_selectivity_max_pct": 100.0,
            "guidance_max_atoms": args.guidance_max_atoms,
            "traversal_guided_prioritization": False,
        },
        "query_contract": {
            "query_table": args.query_table or "candidate table",
            "query_id_column": args.query_id_column,
            "query_vector_column": args.query_vector_column,
            "truth_self_excluded": args.expected_truth_self_excluded,
        },
    }


def run(args: argparse.Namespace) -> None:
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    for path in (args.out, summary_path, manifest_path):
        if path.exists():
            raise ControlError(f"refusing to overwrite artifact: {path}")

    configs, config_order = load_configs(
        args.config_csv,
        args.allow_mean_qualified_matched_config,
    )
    requested = args.filter_names if args.filter_names else config_order
    filter_order, filter_rates = select_filters(
        args.filters_csv, configs, requested
    )
    truth_audit, query_nos = audit_truth(args, filter_order)
    config_audit = audit_config_provenance(args, filter_order, configs)
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
        "inputs": {
            "config_csv": {
                "path": str(args.config_csv.resolve()),
                "sha256": d2.sha256_file(args.config_csv),
            },
            "config_manifest": config_audit,
            "filters_csv": {
                "path": str(args.filters_csv.resolve()),
                "sha256": d2.sha256_file(args.filters_csv),
            },
            "truth_csv": truth_audit,
        },
        "table": args.table,
        "source_index": args.source_index,
        "expected_sqlens_build_id": args.expected_sqlens_build_id,
        "expected_vector_so_sha256": args.expected_vector_so_sha256,
        "guidance_max_atoms": args.guidance_max_atoms,
        "filter_order": filter_order,
        "scheduled_filter_order": scheduled,
        "filter_selectivities": filter_rates,
        "configs": {
            name: {mode: configs[name][mode].as_dict() for mode in MODES}
            for name in filter_order
        },
        "original_container_cpuset": original_cpuset,
        "invocations": [],
    }
    d2.atomic_write_json(manifest_path, manifest)
    combined: list[dict[str, Any]] = []
    try:
        manifest["dedicated_server_gate"] = d2.require_dedicated_server()
        d2.set_container_cpu(args.container, args.backend_cpu)
        identity, runtime_identity = prepare_source_contract(args)
        manifest["source_index_identity_start"] = identity
        manifest["sqlens_runtime_identity_start"] = runtime_identity
        d2.atomic_write_json(manifest_path, manifest)

        for position, filter_name in enumerate(scheduled, start=1):
            record: dict[str, Any] = {
                "position": position,
                "filter_name": filter_name,
                "configs": {
                    mode: configs[filter_name][mode].as_dict() for mode in MODES
                },
                "guidance_max_atoms": args.guidance_max_atoms,
                "status": "running",
            }
            manifest["invocations"].append(record)
            d2.atomic_write_json(manifest_path, manifest)
            record["restart"] = d2.restart_postgres(args)
            record["prewarm"] = prewarm_source(args, identity)
            child_out = child_path(args.out, position, filter_name)
            conflicts = [path for path in child_artifacts(child_out) if path.exists()]
            if conflicts:
                raise ControlError(f"refusing to overwrite child artifacts: {conflicts}")
            command = build_runner_command(
                args, child_out, filter_name, configs[filter_name]
            )
            record["runner_argv"] = command
            record["runner_shell"] = shlex.join(command)
            completed = d2.run_command(command)
            rows = d2.read_csv(child_out)
            validated = validate_child(
                rows, args, filter_name, configs[filter_name], query_nos
            )
            plan = validate_plan_evidence(
                child_out.with_suffix(child_out.suffix + ".plan.json"),
                child_out,
                args,
                filter_name,
                identity,
            )
            combined.extend(validated)
            record.update(
                {
                    "status": "complete",
                    "completed_at": d2.utc_now(),
                    "plan_evidence": plan,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "raw_artifact": {
                        "path": str(child_out),
                        "rows": len(rows),
                        "sha256": d2.sha256_file(child_out),
                    },
                }
            )
            d2.atomic_write_json(manifest_path, manifest)

        summaries = summarize(
            combined,
            filter_order,
            configs,
            filter_rates,
            args.schedule_seed,
            args.bootstrap_samples,
            args.recall_tolerance,
        )
        d2.atomic_write_csv(args.out, combined)
        d2.atomic_write_csv(summary_path, summaries)
        final_identity, final_runtime = prepare_source_contract(args)
        if final_identity != identity:
            raise ControlError("source index identity changed during the experiment")
        speedups = [float(row["d1_speedup_over_stock"]) for row in summaries]
        manifest.update(
            {
                "status": "complete",
                "artifact_valid": True,
                "completed_at": d2.utc_now(),
                "source_index_identity_final": final_identity,
                "sqlens_runtime_identity_final": final_runtime,
                "all_recall_targets_met": all(
                    bool(row["stock_recall_target_met"])
                    and bool(row["d1_recall_target_met"])
                    for row in summaries
                ),
                "statistically_positive_points": sum(
                    bool(row["statistically_positive"]) for row in summaries
                ),
                "geomean_d1_speedup_over_stock": math.exp(
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


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a formal paired/interleaved Stock-versus-D1 control."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-csv", type=Path, required=True)
    parser.add_argument("--config-manifest", type=Path, required=True)
    parser.add_argument("--filters-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument(
        "--truth-provenance-manifest",
        "--truth-manifest",
        dest="truth_manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--filter-names", nargs="+")
    parser.add_argument("--table", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument(
        "--bfs-index",
        required=True,
        help="BFS index bound by the shared matched-recall calibration manifest.",
    )
    parser.add_argument("--query-table")
    parser.add_argument("--query-id-column", default="id")
    parser.add_argument("--query-vector-column", default="embedding")
    parser.add_argument("--candidate-validity-predicate", default="TRUE")
    parser.add_argument(
        "--expected-truth-self-excluded",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Truth/query self-exclusion contract. By default it is inferred as false "
            "for an external query table and true for candidate-table queries."
        ),
    )
    parser.add_argument("--container", default="hybrid-pgvector")
    parser.add_argument("--backend-cpu", type=int, required=True)
    parser.add_argument("--query-offset", type=nonnegative_int, required=True)
    parser.add_argument("--queries", type=d2.positive_int, default=100)
    parser.add_argument("--repeats", type=d2.positive_int, default=5)
    parser.add_argument("--k", type=d2.positive_int, default=10)
    parser.add_argument("--schedule-seed", type=int, default=20260722)
    parser.add_argument("--bootstrap-samples", type=d2.positive_int, default=5000)
    parser.add_argument("--recall-tolerance", type=float, default=1e-12)
    parser.add_argument("--matched-target-recall", type=float, default=0.90)
    parser.add_argument(
        "--allow-mean-qualified-matched-config",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Accept explicitly marked mean_confirmed calibration rows. The "
            "held-out q100/r5 run still fails unless both arms reach the target."
        ),
    )
    parser.add_argument("--d1-cache-mb", type=d2.positive_int, default=1024)
    parser.add_argument(
        "--d1-guidance-kind", choices=("auto", "exact", "bloom"), default="auto"
    )
    parser.add_argument("--d1-exact-max-selectivity-pct", type=float, default=2.5)
    parser.add_argument(
        "--guidance-max-atoms",
        type=d2.positive_int,
        default=128,
        help=(
            "Maximum predicate atoms admitted by D1. The default covers LAION's "
            "70-atom 50%% OR predicate and is passed verbatim to the delegated runner."
        ),
    )
    parser.add_argument("--statement-timeout-ms", type=d2.positive_int, default=300_000)
    parser.add_argument("--progress-queries", type=nonnegative_int, default=0)
    parser.add_argument("--readiness-timeout-s", type=float, default=60.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--expected-sqlens-build-id", required=True)
    parser.add_argument(
        "--expected-vector-so-sha256", type=d2.sha256_arg, required=True
    )
    args = parser.parse_args(argv)
    args.table = canonical_relation(args.table)
    args.source_index = canonical_relation(args.source_index)
    args.bfs_index = canonical_relation(args.bfs_index)
    args.query_table = canonical_relation(args.query_table) if args.query_table else None
    if args.expected_truth_self_excluded is None:
        args.expected_truth_self_excluded = not (
            args.query_table is not None and args.query_table != args.table
        )
    if args.backend_cpu < 0:
        parser.error("--backend-cpu must be non-negative")
    if not math.isfinite(args.recall_tolerance) or args.recall_tolerance < 0:
        parser.error("--recall-tolerance must be finite and non-negative")
    if not math.isfinite(args.matched_target_recall) or not 0 < args.matched_target_recall <= 1:
        parser.error("--matched-target-recall must be finite and within (0, 1]")
    for path in (
        args.config_csv,
        args.config_manifest,
        args.filters_csv,
        args.truth_csv,
        args.truth_manifest,
        args.python,
    ):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
