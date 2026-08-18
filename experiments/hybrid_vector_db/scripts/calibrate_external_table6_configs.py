from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK = (
    ROOT
    / "experiments/hybrid_vector_db/scripts/"
    "pgvector_design1_design2_design3_selectivity_benchmark.py"
)
MODES = ("original", "design1_bloom")
ITERATIVE_SCANS = ("off", "strict_order")
TARGET_RECALL = 0.90
MANIFEST_TYPE = "external_table6_matched_recall_calibration"
MANIFEST_SCHEMA_VERSION = 1
INDEPENDENT_SELECTION_POLICY = (
    "lowest mean end-to-end latency among complete calibration configurations "
    "whose query-level bootstrap Recall@10 LCB95 is at least 0.90"
)
SHARED_SELECTION_POLICY = (
    "among complete search configurations measured by both original and "
    "design1_bloom whose per-arm query-level bootstrap Recall@10 LCB95 is at "
    "least 0.90, minimize the configured shared latency objective"
)
SHARED_CONFIG_FIELDS = (
    "ef_search",
    "iterative_scan",
    "max_scan_tuples",
    "scan_mem_multiplier",
    "guided_collect_target",
    "guided_collect_target_tracks_ef",
    "traversal_guided_target",
    "traversal_guided_prioritization",
    "traversal_guided_burst",
)
BRACKET_POLICY = (
    "seed each mode/filter/iterative family; exponentially expand ef_search until "
    "the target is bracketed; then evaluate all lower ef_search values inside the bracket"
)
DEFAULT_EF_GRID = "20,40,80,120,160,240,320,480,640,800,1000,1280,1920,2560,3840,5120,7680,10000"


@dataclass(frozen=True)
class FilterSpec:
    name: str
    selectivity: str
    predicate: str
    atom_count: int


@dataclass(frozen=True)
class SearchConfig:
    ef_search: int
    iterative_scan: str
    max_scan_tuples: int
    scan_mem_multiplier: float
    guided_collect_target: int
    guided_collect_target_tracks_ef: bool
    traversal_guided_target: int
    traversal_guided_prioritization: bool
    traversal_guided_burst: int

    def with_ef(self, ef_search: int) -> "SearchConfig":
        collect_target = ef_search if self.guided_collect_target_tracks_ef else self.guided_collect_target
        traversal_target = min(self.traversal_guided_target, ef_search)
        return replace(
            self,
            ef_search=ef_search,
            guided_collect_target=collect_target,
            traversal_guided_target=traversal_target,
        )


@dataclass(frozen=True)
class CandidateResult:
    filter_name: str
    mode: str
    config: SearchConfig
    recall_mean: float
    recall_lcb95: float
    recall_ci95_low: float
    recall_ci95_high: float
    latency_mean_ms: float
    latency_p50_ms: float
    queries: int
    samples: int
    raw_path: str
    raw_sha256: str
    plan_path: str
    plan_sha256: str
    table_summary_path: str
    table_summary_sha256: str
    profile_summary_path: str
    profile_summary_sha256: str
    command_sha256: str
    child_reused: bool
    relation_provenance: dict[str, object]
    binary_provenance: dict[str, object]

    @property
    def qualified(self) -> bool:
        return self.recall_lcb95 >= TARGET_RECALL

    @property
    def qualification(self) -> str:
        if self.qualified:
            return "lcb95"
        if self.recall_mean >= TARGET_RECALL:
            return "mean_confirmed"
        return "unqualified"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_positive_ints(value: str) -> list[int]:
    try:
        parsed = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer grid: {value!r}") from exc
    if not parsed or parsed[0] <= 0:
        raise argparse.ArgumentTypeError("integer grid must contain positive values")
    return parsed


def parse_iterative_scans(value: str) -> list[str]:
    scans = list(dict.fromkeys(token.strip() for token in value.split(",") if token.strip()))
    invalid = sorted(set(scans) - set(ITERATIVE_SCANS))
    if not scans or invalid:
        raise argparse.ArgumentTypeError(
            f"iterative scans must be a non-empty subset of {ITERATIVE_SCANS}; invalid={invalid}"
        )
    return scans


def normalize_cpu_list(value: str) -> str:
    cpus: set[int] = set()
    try:
        for token in value.split(","):
            token = token.strip()
            if not token:
                raise ValueError("empty CPU range")
            first_text, separator, last_text = token.partition("-")
            first = int(first_text)
            last = int(last_text) if separator else first
            if first < 0 or last < first:
                raise ValueError(token)
            cpus.update(range(first, last + 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid CPU list {value!r}") from exc
    if not cpus:
        raise argparse.ArgumentTypeError("CPU list must not be empty")
    ordered = sorted(cpus)
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _required_columns(reader: csv.DictReader, required: set[str], path: Path) -> None:
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def load_filters(
    path: Path,
    selected_names: Sequence[str] | None,
    guidance_max_atoms: int,
) -> list[FilterSpec]:
    selected = set(selected_names or ())
    filters: list[FilterSpec] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        _required_columns(reader, {"filter_name", "predicate", "atoms"}, path)
        for row in reader:
            name = row["filter_name"].strip()
            if selected and name not in selected:
                continue
            if not name or name in seen:
                raise ValueError(f"duplicate or empty filter_name in {path}: {name!r}")
            atoms = [part.strip() for part in row["atoms"].split("||") if part.strip()]
            # SQLens receives the complete composition token stream. OR tokens
            # consume admission slots too (N OR leaves become 2N-1 tokens), so
            # preflight must match should_enable_guidance() in the runner.
            atom_count = len(atoms)
            if atom_count <= 0:
                raise ValueError(f"filter {name!r} has no predicate atoms")
            if atom_count > guidance_max_atoms:
                raise ValueError(
                    f"filter {name!r} has {atom_count} atoms, exceeding "
                    f"--guidance-max-atoms={guidance_max_atoms}"
                )
            filters.append(
                FilterSpec(
                    name=name,
                    selectivity=(row.get("actual_pct") or row.get("target_rate") or "").strip(),
                    predicate=row["predicate"].strip(),
                    atom_count=atom_count,
                )
            )
            seen.add(name)
    missing_selected = sorted(selected - seen)
    if missing_selected:
        raise ValueError(f"selected filters are absent from {path}: {missing_selected}")
    if not filters:
        raise ValueError(f"no filters selected from {path}")
    return filters


def load_calibration_split(
    path: Path,
    filters: Sequence[FilterSpec],
    query_offset: int,
    query_count: int,
    expected_self_excluded: bool,
    candidate_validity_predicate: str,
    expected_query_split: str = "calibration",
) -> tuple[list[int], list[int]]:
    filter_names = {item.name for item in filters}
    by_filter: dict[str, dict[int, int]] = {name: {} for name in filter_names}
    self_exclusion_values: set[bool] = set()
    validity_values: set[str] = set()
    split_by_filter: dict[str, dict[int, str]] = {name: {} for name in filter_names}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        _required_columns(
            reader,
            {"filter_name", "query_no", "query_id", "method", "self_excluded"},
            path,
        )
        for row in reader:
            if row.get("method") != "pre_filter_exact" or row.get("filter_name") not in filter_names:
                continue
            filter_name = row["filter_name"]
            query_no = int(row["query_no"])
            query_id = int(row["query_id"])
            previous = by_filter[filter_name].setdefault(query_no, query_id)
            if previous != query_id:
                raise ValueError(f"query_no={query_no} maps to multiple query IDs")
            self_exclusion_values.add(parse_bool(row["self_excluded"]))
            if "query_split" in (reader.fieldnames or ()):
                split_by_filter[filter_name][query_no] = (row.get("query_split") or "").strip()
            if "candidate_validity_predicate" in (reader.fieldnames or ()):
                validity_values.add((row.get("candidate_validity_predicate") or "TRUE").strip())
    if self_exclusion_values != {expected_self_excluded}:
        raise ValueError(
            "truth self_excluded contract mismatch: "
            f"expected {expected_self_excluded}, observed {sorted(self_exclusion_values)}"
        )
    expected_validity = candidate_validity_predicate.strip() or "TRUE"
    if validity_values and validity_values != {expected_validity}:
        raise ValueError(
            "truth candidate_validity_predicate mismatch: "
            f"expected {expected_validity!r}, observed {sorted(validity_values)!r}"
        )
    reference_name = filters[0].name
    all_query_nos = sorted(by_filter[reference_name])
    query_nos = all_query_nos[query_offset : query_offset + query_count]
    if len(query_nos) != query_count:
        raise ValueError(
            f"calibration split requested {query_count} queries at offset {query_offset}, "
            f"but truth provides {len(query_nos)}"
        )
    query_ids = [by_filter[reference_name][query_no] for query_no in query_nos]
    for filter_name, mapping in sorted(by_filter.items()):
        observed = [mapping.get(query_no) for query_no in query_nos]
        if observed != query_ids:
            raise ValueError(f"truth calibration cohort is incomplete or inconsistent for {filter_name}")
        observed_splits = {
            split_by_filter[filter_name].get(query_no, "") for query_no in query_nos
        }
        if any(split_by_filter[filter_name].values()) and observed_splits != {expected_query_split}:
            raise ValueError(
                f"truth rows selected for {filter_name} are not exclusively "
                f"{expected_query_split} rows: "
                f"{sorted(observed_splits)}"
            )
    return query_nos, query_ids


def _optional_int(row: dict[str, str], name: str, default: int) -> int:
    text = (row.get(name) or "").strip()
    return int(text) if text else default


def _optional_float(row: dict[str, str], name: str, default: float) -> float:
    text = (row.get(name) or "").strip()
    return float(text) if text else default


def default_config(args: argparse.Namespace, iterative_scan: str, ef_search: int) -> SearchConfig:
    tracks_ef = str(args.guided_collect_target).strip().lower() == "ef"
    collect_target = ef_search if tracks_ef else int(args.guided_collect_target)
    return SearchConfig(
        ef_search=ef_search,
        iterative_scan=iterative_scan,
        max_scan_tuples=args.max_scan_tuples,
        scan_mem_multiplier=args.scan_mem_multiplier,
        guided_collect_target=collect_target,
        guided_collect_target_tracks_ef=tracks_ef,
        traversal_guided_target=min(args.traversal_guided_target, ef_search),
        traversal_guided_prioritization=args.traversal_guided_prioritization,
        traversal_guided_burst=args.traversal_guided_burst,
    )


def load_seed_configs(
    path: Path | None,
    args: argparse.Namespace,
) -> dict[tuple[str, str, str], SearchConfig]:
    if path is None:
        return {}
    seeds: dict[tuple[str, str, str], SearchConfig] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        _required_columns(reader, {"ef_search"}, path)
        for row in reader:
            filter_name = (row.get("filter_name") or "*").strip() or "*"
            mode = (row.get("mode") or "*").strip() or "*"
            iterative_scan = (row.get("iterative_scan") or "*").strip() or "*"
            if mode not in {*MODES, "*"}:
                raise ValueError(f"unsupported seed mode: {mode!r}")
            if iterative_scan not in {*ITERATIVE_SCANS, "*"}:
                raise ValueError(f"unsupported seed iterative_scan: {iterative_scan!r}")
            ef_search = int(row["ef_search"])
            if ef_search <= 0:
                raise ValueError("seed ef_search must be positive")
            tracks_ef = (row.get("guided_collect_target_policy") or "").strip() == "ef"
            raw_collect = (row.get("guided_collect_target") or "").strip()
            if raw_collect.lower() == "ef" or not raw_collect:
                tracks_ef = True
                collect_target = ef_search
            else:
                collect_target = int(raw_collect)
            key = (filter_name, mode, iterative_scan)
            if key in seeds:
                raise ValueError(f"duplicate seed config for {key}")
            seeds[key] = SearchConfig(
                ef_search=ef_search,
                iterative_scan=iterative_scan,
                max_scan_tuples=_optional_int(row, "max_scan_tuples", args.max_scan_tuples),
                scan_mem_multiplier=_optional_float(
                    row, "scan_mem_multiplier", args.scan_mem_multiplier
                ),
                guided_collect_target=collect_target,
                guided_collect_target_tracks_ef=tracks_ef,
                traversal_guided_target=min(
                    _optional_int(
                        row, "traversal_guided_target", args.traversal_guided_target
                    ),
                    ef_search,
                ),
                traversal_guided_prioritization=(
                    parse_bool(row["traversal_guided_prioritization"])
                    if (row.get("traversal_guided_prioritization") or "").strip()
                    else args.traversal_guided_prioritization
                ),
                traversal_guided_burst=_optional_int(
                    row, "traversal_guided_burst", args.traversal_guided_burst
                ),
            )
    return seeds


def seed_for_family(
    seeds: dict[tuple[str, str, str], SearchConfig],
    filter_name: str,
    mode: str,
    iterative_scan: str,
    args: argparse.Namespace,
) -> SearchConfig:
    keys = (
        (filter_name, mode, iterative_scan),
        (filter_name, mode, "*"),
        (filter_name, "*", iterative_scan),
        ("*", mode, iterative_scan),
        (filter_name, "*", "*"),
        ("*", mode, "*"),
        ("*", "*", iterative_scan),
        ("*", "*", "*"),
    )
    for key in keys:
        if key in seeds:
            return replace(seeds[key], iterative_scan=iterative_scan)
    return default_config(args, iterative_scan, args.default_seed_ef)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def bootstrap_mean_bounds(
    values: Sequence[float], samples: int, seed: int
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1 or samples <= 0:
        return values[0], values[0], values[0]
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return percentile(means, 0.05), percentile(means, 0.025), percentile(means, 0.975)


def _bool_csv(value: object) -> bool:
    return parse_bool(value)


def _canonical_path(value: object) -> Path:
    return Path(str(value)).expanduser().resolve()


def summarize_and_validate_child(
    raw_path: Path,
    plan_path: Path,
    filter_spec: FilterSpec,
    mode: str,
    config: SearchConfig,
    query_nos: Sequence[int],
    args: argparse.Namespace,
    command: Sequence[str],
    *,
    child_reused: bool,
) -> CandidateResult:
    expected_rows = len(query_nos) * args.repeats
    if not raw_path.is_file() or not plan_path.is_file():
        raise ValueError("child raw output or plan evidence is missing")
    raw_sha = sha256_file(raw_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "complete" or plan.get("error") is not None:
        raise ValueError("child plan did not complete successfully")
    if _canonical_path(plan.get("output")) != raw_path.resolve():
        raise ValueError("child plan output path does not bind the raw artifact")
    if plan.get("output_sha256") != raw_sha or int(plan.get("output_rows") or -1) != expected_rows:
        raise ValueError("child plan output hash or row count does not match raw output")
    checks = plan.get("checks")
    if not isinstance(checks, list) or len(checks) != 1 or checks[0].get("passed") is not True:
        raise ValueError("child HNSW plan proof is missing or failed")
    check = checks[0]
    expected_query_table = args.query_table or args.insertion_table
    expected_contract = {
        "expected_table": args.insertion_table,
        "expected_index": args.insertion_index,
        "query_table": expected_query_table,
        "query_id_column": args.query_id_column,
        "query_vector_column": args.query_vector_column,
        "self_excluded": args.expected_truth_self_excluded,
    }
    for field, expected in expected_contract.items():
        if check.get(field) != expected:
            raise ValueError(f"child plan provenance mismatch for {field}: {check.get(field)!r}")
    if check.get("candidate_validity_predicate") != args.candidate_validity_predicate:
        raise ValueError("child plan candidate-validity provenance mismatch")
    expected_query_ids = getattr(args, "calibration_query_id_by_no", {})
    if expected_query_ids and check.get("query_id") != expected_query_ids[query_nos[0]]:
        raise ValueError("child plan was proved with the wrong calibration query")
    query_contract = plan.get("query_contract") or {}
    for field, expected in (
        ("query_table", expected_query_table),
        ("query_id_column", args.query_id_column),
        ("query_vector_column", args.query_vector_column),
        ("self_excluded", args.expected_truth_self_excluded),
        ("candidate_validity_predicate", args.candidate_validity_predicate),
    ):
        if query_contract.get(field) != expected:
            raise ValueError(f"child query contract mismatch for {field}")
    for identity_field in ("sqlens_runtime_identity_startup", "sqlens_runtime_identity_final"):
        identity = plan.get(identity_field) or {}
        if (
            identity.get("exact_match") is not True
            or identity.get("observed_build_id") != args.expected_sqlens_build_id
            or identity.get("observed_vector_so_sha256") != args.expected_vector_so_sha256
        ):
            raise ValueError(f"child binary provenance failed in {identity_field}")
    runtime_identities = plan.get("runtime_sqlens_identity_evidence") or []
    backend_evidence = plan.get("backend_cpu_evidence") or []
    if len(runtime_identities) != 1 or runtime_identities[0].get("exact_match") is not True:
        raise ValueError("child runtime identity evidence is incomplete")
    if (
        len(backend_evidence) != 1
        or backend_evidence[0].get("exact_match") is not True
        or backend_evidence[0].get("requested_cpu_list") != args.backend_cpu_list
        or backend_evidence[0].get("observed_cpu_list") != args.backend_cpu_list
    ):
        raise ValueError("child backend CPU provenance is incomplete or mismatched")
    lifecycle = plan.get("execution_lifecycle") or {}
    if not all(
        lifecycle.get(field) is True
        for field in (
            "backend_cpu_provenance_complete",
            "runtime_sqlens_identity_complete",
            "warmup_complete",
        )
    ):
        raise ValueError("child execution lifecycle is incomplete")

    with raw_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != expected_rows:
        raise ValueError(f"child raw row count is {len(rows)}, expected {expected_rows}")
    by_query: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("error"):
            raise ValueError(f"child row contains error: {row.get('error_detail') or row['error']}")
        if row.get("filter_name") != filter_spec.name or row.get("mode") != mode:
            raise ValueError("child row filter or mode does not match requested candidate")
        if row.get("table") != args.insertion_table or row.get("index") != args.insertion_index:
            raise ValueError("child row table/index provenance mismatch")
        if row.get("query_table") != expected_query_table:
            raise ValueError("child row query table provenance mismatch")
        if row.get("query_id_column") != args.query_id_column:
            raise ValueError("child row query ID column provenance mismatch")
        if row.get("query_vector_column") != args.query_vector_column:
            raise ValueError("child row query vector column provenance mismatch")
        if row.get("candidate_validity_predicate") != args.candidate_validity_predicate:
            raise ValueError("child row candidate-validity provenance mismatch")
        if row.get("sqlens_build_id") != args.expected_sqlens_build_id:
            raise ValueError("child row build ID mismatch")
        if row.get("vector_so_sha256") != args.expected_vector_so_sha256:
            raise ValueError("child row vector.so SHA mismatch")
        if int(row["ef_search"]) != config.ef_search:
            raise ValueError("child row ef_search mismatch")
        if row.get("iterative_scan") != config.iterative_scan:
            raise ValueError("child row iterative_scan mismatch")
        if int(row["max_scan_tuples"]) != config.max_scan_tuples:
            raise ValueError("child row max_scan_tuples mismatch")
        if not math.isclose(
            float(row["scan_mem_multiplier"]), config.scan_mem_multiplier, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("child row scan_mem_multiplier mismatch")
        if int(row["guided_collect_target"]) != config.guided_collect_target:
            raise ValueError("child row guided_collect_target mismatch")
        if int(row["traversal_guided_target"]) != config.traversal_guided_target:
            raise ValueError("child row traversal_guided_target mismatch")
        if _bool_csv(row["truth_self_excluded"]) != args.expected_truth_self_excluded:
            raise ValueError("child row truth self-exclusion mismatch")
        if _bool_csv(row["planner_proof_verified"]) is not True:
            raise ValueError("child row planner proof was not verified")
        recall = float(row["recall"])
        latency = float(row["end_to_end_ms"])
        if not math.isfinite(recall) or recall < 0.0 or recall > 1.0:
            raise ValueError("child row has invalid recall")
        if not math.isfinite(latency) or latency <= 0.0:
            raise ValueError("child row has invalid latency")
        query_no = int(row["query_no"])
        if expected_query_ids and int(row["query_id"]) != expected_query_ids.get(query_no):
            raise ValueError("child row query ID does not match the calibration split")
        if mode == "design1_bloom" and _bool_csv(row["guidance_enabled"]) is not True:
            raise ValueError("design1_bloom child silently ran without predicate guidance")
        by_query.setdefault(query_no, []).append(row)
    if sorted(by_query) != sorted(query_nos):
        raise ValueError("child rows do not exactly cover the calibration query split")
    if any(len(items) != args.repeats for items in by_query.values()):
        raise ValueError("child rows do not exactly cover calibration repeats")
    if any(
        sorted(int(row["repeat"]) for row in items) != list(range(args.repeats))
        for items in by_query.values()
    ):
        raise ValueError("child rows contain duplicate or missing repeat numbers")

    recall_query_means = [
        statistics.fmean(float(row["recall"]) for row in by_query[query_no])
        for query_no in sorted(by_query)
    ]
    latencies = [float(row["end_to_end_ms"]) for row in rows]
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(
            f"{args.bootstrap_seed}\0{filter_spec.name}\0{mode}\0{config.ef_search}\0{config.iterative_scan}".encode()
        ).digest()[:8],
        "big",
    )
    recall_lcb, recall_ci_low, recall_ci_high = bootstrap_mean_bounds(
        recall_query_means, args.bootstrap_samples, bootstrap_seed
    )
    table_summary = raw_path.with_name(raw_path.stem + "_table.csv")
    profile_summary = raw_path.with_name(raw_path.stem + "_profile_summary.csv")
    if not table_summary.is_file() or not profile_summary.is_file():
        raise ValueError("child summary artifacts are missing")
    return CandidateResult(
        filter_name=filter_spec.name,
        mode=mode,
        config=config,
        recall_mean=statistics.fmean(recall_query_means),
        recall_lcb95=recall_lcb,
        recall_ci95_low=recall_ci_low,
        recall_ci95_high=recall_ci_high,
        latency_mean_ms=statistics.fmean(latencies),
        latency_p50_ms=statistics.median(latencies),
        queries=len(by_query),
        samples=len(rows),
        raw_path=str(raw_path),
        raw_sha256=raw_sha,
        plan_path=str(plan_path),
        plan_sha256=sha256_file(plan_path),
        table_summary_path=str(table_summary),
        table_summary_sha256=sha256_file(table_summary),
        profile_summary_path=str(profile_summary),
        profile_summary_sha256=sha256_file(profile_summary),
        command_sha256=stable_sha256(list(command)),
        child_reused=child_reused,
        relation_provenance={
            key: check.get(key)
            for key in (
                "expected_table",
                "expected_table_oid",
                "expected_table_identity",
                "expected_index",
                "expected_index_oid",
                "expected_index_identity",
                "expected_index_access_method",
                "catalog_index_predicate",
                "catalog_index_predicate_sha256",
            )
        },
        binary_provenance=plan["sqlens_runtime_identity_final"],
    )


def build_child_command(
    args: argparse.Namespace,
    filter_spec: FilterSpec,
    mode: str,
    config: SearchConfig,
    raw_path: Path,
) -> list[str]:
    command = [
        args.python,
        str(args.benchmark_script),
        "--insertion-table",
        args.insertion_table,
        "--insertion-index",
        args.insertion_index,
        "--bfs-table",
        args.insertion_table,
        "--bfs-index",
        args.insertion_index,
        "--query-id-column",
        args.query_id_column,
        "--query-vector-column",
        args.query_vector_column,
        "--candidate-validity-predicate",
        args.candidate_validity_predicate,
        "--truth-csv",
        str(args.truth_csv.resolve()),
        "--out",
        str(raw_path.resolve()),
        "--filters-csv",
        str(args.filters_csv.resolve()),
        "--modes",
        mode,
        "--execution-order",
        "mode_major",
        "--schedule-seed",
        str(args.schedule_seed),
        "--filter-names",
        filter_spec.name,
        "--queries",
        str(args.query_count),
        "--query-offset",
        str(args.query_offset),
        "--repeats",
        str(args.repeats),
        "--warmup-queries",
        str(min(args.warmup_queries, args.query_count)),
        "--k",
        "10",
        "--ef-search",
        str(config.ef_search),
        "--guided-collect-target",
        str(config.guided_collect_target),
        "--traversal-guided-target",
        str(config.traversal_guided_target),
        "--traversal-guided-burst",
        str(config.traversal_guided_burst),
        "--guidance-filter-strategy",
        args.guidance_filter_strategy,
        "--iterative-scan",
        config.iterative_scan,
        "--max-scan-tuples",
        str(config.max_scan_tuples),
        "--scan-mem-multiplier",
        str(config.scan_mem_multiplier),
        "--d1-guidance-kind",
        args.d1_guidance_kind,
        "--d1-exact-max-selectivity-pct",
        str(args.d1_exact_max_selectivity_pct),
        "--d1-cache-mb",
        str(args.d1_cache_mb),
        "--guidance-selectivity-max-pct",
        str(args.guidance_selectivity_max_pct),
        "--guidance-max-atoms",
        str(args.guidance_max_atoms),
        "--statement-timeout-ms",
        str(args.statement_timeout_ms),
        "--progress-queries",
        "0",
        "--expected-sqlens-build-id",
        args.expected_sqlens_build_id,
        "--expected-vector-so-sha256",
        args.expected_vector_so_sha256,
        "--backend-cpu-list",
        args.backend_cpu_list,
    ]
    if args.query_table:
        command.extend(["--query-table", args.query_table])
    command.append(
        "--expected-truth-self-excluded"
        if args.expected_truth_self_excluded
        else "--no-expected-truth-self-excluded"
    )
    command.append("--warmup-all-queries" if args.warmup_all_queries else "--no-warmup-all-queries")
    command.append(
        "--traversal-guided-prioritization"
        if config.traversal_guided_prioritization
        else "--no-traversal-guided-prioritization"
    )
    command.append("--force-hnsw" if args.force_hnsw else "--no-force-hnsw")
    command.append(
        "--require-preferred-index-guc"
        if args.require_preferred_index_guc
        else "--no-require-preferred-index-guc"
    )
    return command


def _candidate_token(filter_name: str, mode: str, config: SearchConfig) -> str:
    digest = stable_sha256({"filter": filter_name, "mode": mode, **asdict(config)})[:16]
    safe_filter = "".join(char if char.isalnum() or char in "_.-" else "_" for char in filter_name)
    return f"{safe_filter}.{mode}.{config.iterative_scan}.ef{config.ef_search}.{digest}"


def run_or_resume_child(
    args: argparse.Namespace,
    children_dir: Path,
    filter_spec: FilterSpec,
    mode: str,
    config: SearchConfig,
    query_nos: Sequence[int],
) -> CandidateResult:
    token = _candidate_token(filter_spec.name, mode, config)
    attempts = sorted(children_dir.glob(f"{token}.a*.raw.csv"))
    if args.resume:
        for raw_path in reversed(attempts):
            command = build_child_command(args, filter_spec, mode, config, raw_path)
            try:
                return summarize_and_validate_child(
                    raw_path,
                    raw_path.with_suffix(raw_path.suffix + ".plan.json"),
                    filter_spec,
                    mode,
                    config,
                    query_nos,
                    args,
                    command,
                    child_reused=True,
                )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, csv.Error):
                continue
    elif attempts:
        raise FileExistsError(f"child calibration artifact already exists: {attempts[-1]}")

    raw_path = children_dir / f"{token}.a{len(attempts) + 1:03d}.raw.csv"
    command = build_child_command(args, filter_spec, mode, config, raw_path)
    log_path = raw_path.with_suffix(raw_path.suffix + ".log")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"child benchmark failed with exit code {result.returncode}; see {log_path}"
        )
    return summarize_and_validate_child(
        raw_path,
        raw_path.with_suffix(raw_path.suffix + ".plan.json"),
        filter_spec,
        mode,
        config,
        query_nos,
        args,
        command,
        child_reused=False,
    )


def _next_exponential_lower(values: Sequence[int], current: int) -> int | None:
    lower = [value for value in values if value < current]
    if not lower:
        return None
    candidates = [value for value in lower if value <= current / 2]
    return max(candidates) if candidates else max(lower)


def _next_exponential_upper(values: Sequence[int], current: int) -> int | None:
    upper = [value for value in values if value > current]
    if not upper:
        return None
    candidates = [value for value in upper if value >= current * 2]
    return min(candidates) if candidates else min(upper)


def calibrate_family(
    seed: SearchConfig,
    ef_grid: Sequence[int],
    evaluate: Callable[[SearchConfig], CandidateResult],
) -> list[CandidateResult]:
    values = sorted({*ef_grid, seed.ef_search})
    results: dict[int, CandidateResult] = {}

    def measure(ef_search: int) -> CandidateResult:
        if ef_search not in results:
            results[ef_search] = evaluate(seed.with_ef(ef_search))
        return results[ef_search]

    seed_result = measure(seed.ef_search)
    if seed_result.qualified:
        known_pass = seed.ef_search
        current = seed.ef_search
        while True:
            lower = _next_exponential_lower(values, current)
            if lower is None:
                break
            lower_result = measure(lower)
            if not lower_result.qualified:
                for ef_search in sorted(
                    value for value in values if lower < value < known_pass
                ):
                    measure(ef_search)
                break
            known_pass = lower
            current = lower
    else:
        known_fail = seed.ef_search
        current = seed.ef_search
        while True:
            upper = _next_exponential_upper(values, current)
            if upper is None:
                break
            upper_result = measure(upper)
            if upper_result.qualified:
                for ef_search in sorted(
                    value for value in values if known_fail < value < upper
                ):
                    measure(ef_search)
                break
            known_fail = upper
            current = upper
    return [results[value] for value in sorted(results)]


def select_fastest_qualified(
    results: Iterable[CandidateResult],
    allow_mean_at_grid_ceiling: bool = False,
) -> CandidateResult:
    candidates = list(results)
    qualified = [result for result in candidates if result.qualified]
    if not qualified:
        if allow_mean_at_grid_ceiling and candidates:
            ceiling = max(result.config.ef_search for result in candidates)
            mean_confirmed = [
                result
                for result in candidates
                if result.config.ef_search == ceiling
                and result.recall_mean >= TARGET_RECALL
            ]
            if mean_confirmed:
                return min(
                    mean_confirmed,
                    key=lambda result: (
                        -result.recall_mean,
                        result.latency_mean_ms,
                        result.config.iterative_scan != "strict_order",
                    ),
                )
        raise RuntimeError(
            "no complete calibration configuration reaches Recall@10 LCB95 >= 0.90"
        )
    return min(
        qualified,
        key=lambda result: (
            result.latency_mean_ms,
            result.config.ef_search,
            result.config.iterative_scan != "strict_order",
        ),
    )


def _search_config_sort_key(config: SearchConfig) -> tuple[object, ...]:
    return tuple(getattr(config, field) for field in SHARED_CONFIG_FIELDS)


def select_shared_qualified(
    results: Iterable[CandidateResult],
    latency_objective: str,
    allow_mean_at_grid_ceiling: bool = False,
) -> list[CandidateResult]:
    if latency_objective not in {"mean", "max"}:
        raise ValueError(f"unsupported shared latency objective: {latency_objective!r}")
    by_mode: dict[str, dict[SearchConfig, CandidateResult]] = {
        mode: {} for mode in MODES
    }
    filter_names: set[str] = set()
    for result in results:
        if result.mode not in by_mode:
            raise ValueError(f"unsupported calibration mode: {result.mode!r}")
        filter_names.add(result.filter_name)
        if result.config in by_mode[result.mode]:
            raise ValueError(
                f"duplicate measured configuration for filter={result.filter_name} "
                f"mode={result.mode}"
            )
        by_mode[result.mode][result.config] = result
    if len(filter_names) != 1:
        raise ValueError(
            "shared-search-config selection requires results for exactly one filter"
        )

    common_configs = set.intersection(
        *(set(by_mode[mode]) for mode in MODES)
    )
    qualified_configs = [
        config
        for config in common_configs
        if all(by_mode[mode][config].qualified for mode in MODES)
    ]
    if not qualified_configs and allow_mean_at_grid_ceiling and common_configs:
        ceiling = max(config.ef_search for config in common_configs)
        mean_confirmed_configs = [
            config
            for config in common_configs
            if config.ef_search == ceiling
            and all(
                by_mode[mode][config].recall_mean >= TARGET_RECALL
                for mode in MODES
            )
        ]
        if mean_confirmed_configs:
            selected_config = min(
                mean_confirmed_configs,
                key=lambda config: (
                    -min(by_mode[mode][config].recall_mean for mode in MODES),
                    max(by_mode[mode][config].latency_mean_ms for mode in MODES),
                    _search_config_sort_key(config),
                ),
            )
            return [by_mode[mode][selected_config] for mode in MODES]
    if not qualified_configs:
        filter_name = next(iter(filter_names), "<none>")
        raise RuntimeError(
            f"filter={filter_name} has no shared measured search configuration "
            "for which both original and design1_bloom reach Recall@10 LCB95 >= 0.90"
        )

    def selection_key(config: SearchConfig) -> tuple[object, ...]:
        latencies = tuple(
            by_mode[mode][config].latency_mean_ms for mode in MODES
        )
        mean_latency = statistics.fmean(latencies)
        max_latency = max(latencies)
        objective = mean_latency if latency_objective == "mean" else max_latency
        return (
            objective,
            max_latency,
            mean_latency,
            _search_config_sort_key(config),
        )

    selected_config = min(qualified_configs, key=selection_key)
    return [by_mode[mode][selected_config] for mode in MODES]


def selection_contract(args: argparse.Namespace) -> dict[str, object]:
    shared = args.selection_coupling == "shared-search-config"
    return {
        "coupling": args.selection_coupling,
        "shared_search_config": shared,
        "latency_objective": (
            args.shared_latency_objective if shared else "per-arm-mean"
        ),
        "policy": (
            SHARED_SELECTION_POLICY
            if shared
            else INDEPENDENT_SELECTION_POLICY
        ),
        "qualification": (
            "each selected arm must independently satisfy query-level bootstrap "
            "Recall@10 LCB95 >= 0.90; when explicitly enabled, an otherwise "
            "unqualified family may use its grid-ceiling configuration only if "
            "calibration mean Recall@10 >= 0.90, subject to held-out final gating"
        ),
        "allow_mean_qualified_at_grid_ceiling": (
            args.allow_mean_qualified_at_grid_ceiling
        ),
        "shared_config_identity_fields": (
            list(SHARED_CONFIG_FIELDS) if shared else []
        ),
        "deterministic_tie_break": (
            ["max_arm_latency", "mean_arm_latency", *SHARED_CONFIG_FIELDS]
            if shared
            else ["ef_search", "strict_order_before_off"]
        ),
    }


def _output_manifest_path(args: argparse.Namespace) -> Path:
    return args.manifest_out or args.out.with_suffix(args.out.suffix + ".manifest.json")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def guard_output_paths(
    out: Path,
    manifest_path: Path,
    children_dir: Path,
    run_spec_hash: str,
    resume: bool,
) -> bool:
    if out.resolve() == manifest_path.resolve():
        raise ValueError("configs CSV and manifest paths must differ")
    if out.exists() or manifest_path.exists():
        if not resume:
            raise FileExistsError("calibration output already exists and --no-resume was requested")
        if not manifest_path.is_file():
            raise FileExistsError("refusing to overwrite an artifact without a calibration manifest")
        manifest = _read_json(manifest_path)
        if (
            manifest.get("artifact_type") != MANIFEST_TYPE
            or manifest.get("artifact_scope") != "calibration_only"
            or manifest.get("run_spec_hash") != run_spec_hash
        ):
            raise FileExistsError("refusing to overwrite a foreign or formal artifact")
        if manifest.get("status") == "complete":
            output = manifest.get("output") or {}
            if (
                out.is_file()
                and output.get("configs_sha256") == sha256_file(out)
                and output.get("manifest_path") == str(manifest_path)
            ):
                return True
            raise ValueError("completed calibration output failed its manifest hash gate")
    if children_dir.exists() and not resume and any(children_dir.iterdir()):
        raise FileExistsError("calibration child directory is non-empty under --no-resume")
    return False


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_configs(path: Path, selected: Sequence[CandidateResult]) -> None:
    fields = [
        "filter_name",
        "mode",
        "target_recall",
        "selection_status",
        "qualification",
        "ef_search",
        "iterative_scan",
        "max_scan_tuples",
        "scan_mem_multiplier",
        "guided_collect_target",
        "guided_collect_target_policy",
        "traversal_guided_target",
        "traversal_guided_prioritization",
        "traversal_guided_burst",
        "recall_mean",
        "recall_lcb95",
        "calibration_recall_mean",
        "calibration_recall_lcb95",
        "recall_ci95_low",
        "recall_ci95_high",
        "latency_mean_ms",
        "latency_p50_ms",
        "calibration_queries",
        "calibration_samples",
        "child_raw",
        "child_raw_sha256",
        "child_plan",
        "child_plan_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for result in sorted(selected, key=lambda item: (item.filter_name, MODES.index(item.mode))):
            writer.writerow(
                {
                    "filter_name": result.filter_name,
                    "mode": result.mode,
                    "target_recall": f"{TARGET_RECALL:.2f}",
                    "selection_status": f"{result.qualification}_qualified",
                    "qualification": result.qualification,
                    "ef_search": result.config.ef_search,
                    "iterative_scan": result.config.iterative_scan,
                    "max_scan_tuples": result.config.max_scan_tuples,
                    "scan_mem_multiplier": result.config.scan_mem_multiplier,
                    "guided_collect_target": result.config.guided_collect_target,
                    "guided_collect_target_policy": (
                        "ef" if result.config.guided_collect_target_tracks_ef else "fixed"
                    ),
                    "traversal_guided_target": result.config.traversal_guided_target,
                    "traversal_guided_prioritization": result.config.traversal_guided_prioritization,
                    "traversal_guided_burst": result.config.traversal_guided_burst,
                    "recall_mean": f"{result.recall_mean:.12g}",
                    "recall_lcb95": f"{result.recall_lcb95:.12g}",
                    "calibration_recall_mean": f"{result.recall_mean:.12g}",
                    "calibration_recall_lcb95": f"{result.recall_lcb95:.12g}",
                    "recall_ci95_low": f"{result.recall_ci95_low:.12g}",
                    "recall_ci95_high": f"{result.recall_ci95_high:.12g}",
                    "latency_mean_ms": f"{result.latency_mean_ms:.12g}",
                    "latency_p50_ms": f"{result.latency_p50_ms:.12g}",
                    "calibration_queries": result.queries,
                    "calibration_samples": result.samples,
                    "child_raw": result.raw_path,
                    "child_raw_sha256": result.raw_sha256,
                    "child_plan": result.plan_path,
                    "child_plan_sha256": result.plan_sha256,
                }
            )
    temporary.replace(path)


def candidate_manifest_entry(result: CandidateResult) -> dict[str, object]:
    payload = asdict(result)
    payload["qualified"] = result.qualified
    return payload


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibration-only matched-recall tuner for external Table-6 datasets. "
            "It never runs a held-out final cohort."
        )
    )
    parser.add_argument("--out", type=Path, required=True, help="Selected calibration configs CSV")
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--children-dir", type=Path)
    parser.add_argument("--filters-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--seed-configs", "--seed-configs-csv", dest="seed_configs", type=Path)
    parser.add_argument("--filter-names", nargs="*")
    parser.add_argument("--insertion-table", "--table", dest="insertion_table", required=True)
    parser.add_argument("--insertion-index", "--source-index", dest="insertion_index", required=True)
    parser.add_argument(
        "--bfs-index",
        required=True,
        help="Same-graph BFS index authorized for the downstream D2 measurement contract.",
    )
    parser.add_argument("--query-table")
    parser.add_argument("--query-id-column", default="id")
    parser.add_argument("--query-vector-column", default="embedding")
    parser.add_argument("--candidate-validity-predicate", default="TRUE")
    parser.add_argument(
        "--expected-truth-self-excluded",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="External query tables normally require false; both contracts are supported.",
    )
    parser.add_argument("--expected-sqlens-build-id", required=True)
    parser.add_argument("--expected-vector-so-sha256", required=True)
    parser.add_argument("--backend-cpu-list", type=normalize_cpu_list, required=True)
    parser.add_argument("--query-offset", "--calibration-query-offset", dest="query_offset", type=int, default=0)
    parser.add_argument("--query-count", "--calibration-queries", dest="query_count", type=int, default=80)
    parser.add_argument("--repeats", "--calibration-repeats", dest="repeats", type=int, default=2)
    parser.add_argument("--final-query-offset", type=int, default=80)
    parser.add_argument("--final-queries", type=int, default=100)
    parser.add_argument("--final-repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ef-grid", "--ef-search-values", dest="ef_grid", type=parse_positive_ints, default=parse_positive_ints(DEFAULT_EF_GRID))
    parser.add_argument("--iterative-scans", "--iterative-scan-values", dest="iterative_scans", type=parse_iterative_scans, default=list(ITERATIVE_SCANS))
    parser.add_argument("--default-seed-ef", type=int, default=1000)
    parser.add_argument("--max-scan-tuples", type=int, default=5_000_000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=32.0)
    parser.add_argument("--guided-collect-target", default="ef")
    parser.add_argument("--traversal-guided-target", type=int, default=40)
    parser.add_argument("--traversal-guided-prioritization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--traversal-guided-burst", type=int, default=8)
    parser.add_argument("--guidance-filter-strategy", choices=["safe_guided"], default="safe_guided")
    parser.add_argument("--guidance-max-atoms", type=int, default=128)
    parser.add_argument("--guidance-selectivity-max-pct", type=float, default=100.0)
    parser.add_argument("--d1-guidance-kind", choices=["auto", "exact", "bloom"], default="auto")
    parser.add_argument("--d1-exact-max-selectivity-pct", type=float, default=2.5)
    parser.add_argument("--d1-cache-mb", type=int, default=1024)
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument("--warmup-all-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-preferred-index-guc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--statement-timeout-ms", type=int, default=300_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument(
        "--selection-coupling",
        choices=["independent", "shared-search-config"],
        default="independent",
        help=(
            "Use independent per-arm configs (default), or require Stock and D1 "
            "to use one identical measured search config per filter."
        ),
    )
    parser.add_argument(
        "--shared-latency-objective",
        choices=["mean", "max"],
        default="max",
        help=(
            "In shared-search-config mode, minimize either the two-arm mean or "
            "the slower arm's latency; ties are deterministic."
        ),
    )
    parser.add_argument(
        "--allow-mean-qualified-at-grid-ceiling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail-closed by default. When enabled, a mode/filter family with no "
            "LCB95-qualified configuration may select only a grid-ceiling "
            "configuration whose calibration mean Recall@10 reaches 0.90; the "
            "held-out final controller must still confirm mean recall >= 0.90."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--benchmark-script", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--truth-provenance-manifest",
        "--truth-manifest",
        dest="truth_provenance_manifest",
        type=Path,
        required=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.candidate_validity_predicate = args.candidate_validity_predicate.strip() or "TRUE"
    if args.query_offset < 0 or args.query_count <= 0 or args.repeats <= 0:
        raise ValueError("query offset must be non-negative and count/repeats must be positive")
    if args.final_query_offset < 0 or args.final_queries <= 0 or args.final_repeats <= 0:
        raise ValueError("final query offset must be non-negative and count/repeats must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.default_seed_ef <= 0 or args.max_scan_tuples <= 0:
        raise ValueError("seed ef_search and max_scan_tuples must be positive")
    if not math.isfinite(args.scan_mem_multiplier) or args.scan_mem_multiplier <= 0:
        raise ValueError("--scan-mem-multiplier must be finite and positive")
    if args.guidance_max_atoms <= 0:
        raise ValueError("--guidance-max-atoms must be positive")
    if len(args.expected_vector_so_sha256) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in args.expected_vector_so_sha256
    ):
        raise ValueError("--expected-vector-so-sha256 must be a 64-character hex digest")
    if not args.expected_sqlens_build_id:
        raise ValueError("--expected-sqlens-build-id must not be empty")
    if not args.filters_csv.is_file() or not args.truth_csv.is_file():
        raise FileNotFoundError("filters and truth CSV inputs must exist")
    if args.seed_configs is not None and not args.seed_configs.is_file():
        raise FileNotFoundError(args.seed_configs)
    if not args.truth_provenance_manifest.is_file():
        raise FileNotFoundError(args.truth_provenance_manifest)
    if not args.benchmark_script.is_file():
        raise FileNotFoundError(args.benchmark_script)
    if args.guidance_filter_strategy != "safe_guided":
        raise ValueError("external Table-6 matched configs require safe_guided")
    if args.traversal_guided_prioritization:
        raise ValueError("safe_guided requires --no-traversal-guided-prioritization")
    if args.guided_collect_target != "ef":
        try:
            if int(args.guided_collect_target) <= 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError("--guided-collect-target must be 'ef' or a positive integer") from exc


def run_spec_payload(
    args: argparse.Namespace,
    query_nos: Sequence[int],
    query_ids: Sequence[int],
    final_query_nos: Sequence[int],
    final_query_ids: Sequence[int],
) -> dict[str, object]:
    selection = selection_contract(args)
    return {
        "artifact_type": MANIFEST_TYPE,
        "artifact_scope": "calibration_only",
        "target_recall": TARGET_RECALL,
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "child_runner": {"path": str(args.benchmark_script.resolve()), "sha256": sha256_file(args.benchmark_script)},
        "inputs": {
            "filters_csv": {"path": str(args.filters_csv.resolve()), "sha256": sha256_file(args.filters_csv)},
            "truth_csv": {"path": str(args.truth_csv.resolve()), "sha256": sha256_file(args.truth_csv)},
            "truth_provenance_manifest": {
                "path": str(args.truth_provenance_manifest.resolve()),
                "sha256": sha256_file(args.truth_provenance_manifest),
            },
            "seed_configs": (
                {"path": str(args.seed_configs.resolve()), "sha256": sha256_file(args.seed_configs)}
                if args.seed_configs is not None
                else None
            ),
        },
        "binary": {
            "expected_sqlens_build_id": args.expected_sqlens_build_id,
            "expected_vector_so_sha256": args.expected_vector_so_sha256.lower(),
        },
        "relations": {
            "candidate_table": args.insertion_table,
            "hnsw_index": args.insertion_index,
            "query_table": args.query_table or args.insertion_table,
            "query_id_column": args.query_id_column,
            "query_vector_column": args.query_vector_column,
            "candidate_validity_predicate": args.candidate_validity_predicate,
        },
        "query_split": {
            "role": "calibration",
            "offset": args.query_offset,
            "count": args.query_count,
            "query_nos": list(query_nos),
            "query_ids": list(query_ids),
            "self_excluded": args.expected_truth_self_excluded,
            "held_out_final_executed": False,
        },
        "selection": selection,
        "protocol": {
            "table": args.insertion_table,
            "source_index": args.insertion_index,
            "bfs_index": args.bfs_index,
            "query_table": args.query_table,
            "query_id_column": args.query_id_column,
            "query_vector_column": args.query_vector_column,
            "candidate_validity_predicate": args.candidate_validity_predicate,
            "expected_truth_self_excluded": args.expected_truth_self_excluded,
            "guidance_filter_strategy": "safe_guided",
            "guidance_max_atoms": args.guidance_max_atoms,
            "modes": list(MODES),
            "query_offset": args.final_query_offset,
            "queries": args.final_queries,
            "repeats": args.final_repeats,
            "final_query_offset": args.final_query_offset,
            "final_queries": args.final_queries,
            "final_repeats": args.final_repeats,
            "measurement": {
                "query_offset": args.final_query_offset,
                "queries": args.final_queries,
                "repeats": args.final_repeats,
                "query_nos": list(final_query_nos),
                "query_ids": list(final_query_ids),
                "executed_by_calibrator": False,
            },
            "calibration": {
                "query_offset": args.query_offset,
                "queries": args.query_count,
                "repeats": args.repeats,
                "query_nos": list(query_nos),
                "query_ids": list(query_ids),
                "executed_by_calibrator": True,
            },
            "iterative_scans": list(args.iterative_scans),
            "ef_grid": list(args.ef_grid),
            "default_seed_ef": args.default_seed_ef,
            "workers": args.workers,
            "backend_cpu_list_for_every_child": args.backend_cpu_list,
            "selection_coupling": selection["coupling"],
            "selection_policy": selection["policy"],
            "shared_latency_objective": selection["latency_objective"],
            "selection": selection,
            "search_policy": BRACKET_POLICY,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "schedule_seed": args.schedule_seed,
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    validate_args(args)
    args.out = args.out.resolve()
    if args.manifest_out is not None:
        args.manifest_out = args.manifest_out.resolve()
    if args.children_dir is not None:
        args.children_dir = args.children_dir.resolve()
    args.filters_csv = args.filters_csv.resolve()
    args.truth_csv = args.truth_csv.resolve()
    args.truth_provenance_manifest = args.truth_provenance_manifest.resolve()
    args.benchmark_script = args.benchmark_script.resolve()
    args.expected_vector_so_sha256 = args.expected_vector_so_sha256.lower()
    filters = load_filters(args.filters_csv, args.filter_names, args.guidance_max_atoms)
    query_nos, query_ids = load_calibration_split(
        args.truth_csv,
        filters,
        args.query_offset,
        args.query_count,
        args.expected_truth_self_excluded,
        args.candidate_validity_predicate,
    )
    final_query_nos, final_query_ids = load_calibration_split(
        args.truth_csv,
        filters,
        args.final_query_offset,
        args.final_queries,
        args.expected_truth_self_excluded,
        args.candidate_validity_predicate,
        expected_query_split="final",
    )
    if set(query_nos) & set(final_query_nos):
        raise ValueError("calibration and formal measurement query splits overlap")
    args.calibration_query_id_by_no = dict(zip(query_nos, query_ids))
    seeds = load_seed_configs(args.seed_configs, args)
    spec = run_spec_payload(args, query_nos, query_ids, final_query_nos, final_query_ids)
    run_spec_hash = stable_sha256(spec)
    manifest_path = _output_manifest_path(args)
    children_dir = args.children_dir or (
        args.out.parent / f"{args.out.stem}.calibration_children" / run_spec_hash[:16]
    )
    if guard_output_paths(
        args.out, manifest_path, children_dir, run_spec_hash, args.resume
    ):
        print(f"reusing completed calibration {args.out}", flush=True)
        return args.out, manifest_path
    children_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    family_jobs = [
        (filter_spec, mode, iterative_scan)
        for filter_spec in filters
        for mode in MODES
        for iterative_scan in args.iterative_scans
    ]

    def execute_family(job: tuple[FilterSpec, str, str]) -> list[CandidateResult]:
        filter_spec, mode, iterative_scan = job
        seed = seed_for_family(seeds, filter_spec.name, mode, iterative_scan, args)
        return calibrate_family(
            seed,
            args.ef_grid,
            lambda config: run_or_resume_child(
                args, children_dir, filter_spec, mode, config, query_nos
            ),
        )

    all_results: list[CandidateResult] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(execute_family, job): job for job in family_jobs}
            for future in as_completed(futures):
                job = futures[future]
                family_results = future.result()
                all_results.extend(family_results)
                print(
                    f"calibrated filter={job[0].name} mode={job[1]} "
                    f"iterative_scan={job[2]} configs={len(family_results)}",
                    flush=True,
                )
        selected: list[CandidateResult] = []
        for filter_spec in filters:
            filter_results = [
                result
                for result in all_results
                if result.filter_name == filter_spec.name
            ]
            if args.selection_coupling == "shared-search-config":
                selected.extend(
                    select_shared_qualified(
                        filter_results,
                        args.shared_latency_objective,
                        args.allow_mean_qualified_at_grid_ceiling,
                    )
                )
            else:
                selected.extend(
                    select_fastest_qualified(
                        (
                            result
                            for result in filter_results
                            if result.mode == mode
                        ),
                        args.allow_mean_qualified_at_grid_ceiling,
                    )
                    for mode in MODES
                )
        relation_identities = {
            stable_sha256(result.relation_provenance) for result in all_results
        }
        if len(relation_identities) != 1:
            raise RuntimeError("child relation OID/catalog provenance changed during calibration")
        write_configs(args.out, selected)
        manifest = {
            **spec,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "artifact_valid": True,
            "started_at": started_at,
            "completed_at": utc_now(),
            "run_spec_hash": run_spec_hash,
            "filter_names": [item.name for item in filters],
            "filter_atom_counts": {item.name: item.atom_count for item in filters},
            "children": [
                candidate_manifest_entry(result)
                for result in sorted(
                    all_results,
                    key=lambda item: (
                        item.filter_name,
                        item.mode,
                        item.config.iterative_scan,
                        item.config.ef_search,
                    ),
                )
            ],
            "selected": [candidate_manifest_entry(result) for result in selected],
            "child_counts": {
                "evaluated": len(all_results),
                "reused": sum(result.child_reused for result in all_results),
                "launched": sum(not result.child_reused for result in all_results),
                "all_plan_and_provenance_gates_passed": True,
            },
            "runtime": {
                "sqlens_build_id": args.expected_sqlens_build_id,
                "vector_so_sha256": args.expected_vector_so_sha256,
                "all_children_exact_match": True,
            },
            "outputs": {
                "matched_configs_csv": {
                    "path": str(args.out),
                    "sha256": sha256_file(args.out),
                    "rows": len(selected),
                }
            },
            "output": {
                "configs_path": str(args.out),
                "configs_sha256": sha256_file(args.out),
                "configs_rows": len(selected),
                "manifest_path": str(manifest_path),
            },
            "held_out_final": {"executed": False, "artifacts": []},
            "error": None,
        }
        _atomic_write_json(manifest_path, manifest)
    except BaseException as exc:
        failure_manifest = {
            **spec,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "failed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "run_spec_hash": run_spec_hash,
            "children": [candidate_manifest_entry(result) for result in all_results],
            "held_out_final": {"executed": False, "artifacts": []},
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
        }
        _atomic_write_json(manifest_path, failure_manifest)
        raise
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {manifest_path}", flush=True)
    return args.out, manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
