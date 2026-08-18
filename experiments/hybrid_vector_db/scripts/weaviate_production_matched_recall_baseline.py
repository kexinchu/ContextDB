"""Production matched-recall runner for Weaviate HNSW filter routing.

This runner reuses the formal Amazon-10M data, truth, split, query, and
statistics contracts from ``weaviate_matched_recall_baseline``.  It adds a
three-dimensional calibration grid over filter strategy, ``ef``, and
``flatSearchCutoff``.  Timed work is one sequential client only.
"""

import argparse
import copy
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import weaviate_matched_recall_baseline as baseline


ROOT = baseline.ROOT
CLASS_NAME = baseline.CLASS_NAME
K = baseline.K
NA = baseline.NA
CALIBRATION_QUERY_NOS = baseline.CALIBRATION_QUERY_NOS
FINAL_QUERY_NOS = baseline.FINAL_QUERY_NOS
CALIBRATION_REPEATS = baseline.CALIBRATION_REPEATS
FINAL_REPEATS = baseline.FINAL_REPEATS
DEFAULT_EF_VALUES = baseline.DEFAULT_EF_VALUES
DEFAULT_TARGETS = baseline.DEFAULT_TARGETS
DEFAULT_FILTER_STRATEGIES = ("acorn", "sweeping")
FLAT_STRATEGY_REPRESENTATIVE = "sweeping"
# Retained as a CLI compatibility field.  The formal threshold is exactly the
# requested target, so non-zero values are rejected.
DEFAULT_CALIBRATION_LCB_MARGIN = 0.0
CALIBRATION_SELECTION_POLICY = "calibration_lcb95_then_max_mean_recall_v1"
CALIBRATION_SELECTION_RULE = (
    "calibration-only: select the lowest mean-latency complete HNSW configuration "
    "with bootstrap recall LCB95 >= target; only when no LCB-qualified HNSW "
    "configuration exists and both strategy EF grids are fully exhausted, select "
    "the highest calibration mean recall with latency as the tie-break; held-out "
    "final evidence cannot reselect"
)
CALIBRATION_STOP_METRIC = "recall_lcb95"
CALIBRATION_FALLBACK = "max_calibration_mean_recall_after_full_hnsw_grid"
# Covers every published Amazon-10M filter cardinality, including the 5.03M
# largest allow-list, while retaining a no-flat baseline at zero.
DEFAULT_FLAT_SEARCH_CUTOFFS = (0, 25_000, 100_000, 250_000, 600_000, 1_000_000,
                               1_500_000, 2_500_000, 3_500_000, 5_100_000)
DEFAULT_OUT = ROOT / "results/hybrid_vector_db/weaviate_production_matched_recall_baseline.csv"
DEFAULT_TRUTH = (
    ROOT
    / "results/hybrid_vector_db/"
    "amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal.csv"
)
DEFAULT_TRUTH_MANIFEST = DEFAULT_TRUTH.with_name(
    "amazon_selectivity14_exact_truth_q10200_unique_embeddings_formal_manifest.json"
)
DEFAULT_IMPORT_MANIFEST = (
    ROOT / "results/hybrid_vector_db/weaviate_amazon10m_formal_import_manifest.json"
)
CHECKPOINT_VERSION = 8
LEGACY_EXECUTION_PROTOCOL = "legacy-q100-r5"
FORMAL_EXECUTION_PROTOCOL = "current-q10k-r3"
CURRENT_PROTOCOL = "amazon10m_current_corpus_q10200_truth_weaviate_v2"
LEGACY_CURRENT_PROTOCOL = "amazon10m_current_corpus_q10200_truth_weaviate_v1"
FORMAL_CALIBRATION_REPEATS = 2
FORMAL_FINAL_REPEATS = 3
FORMAL_CALIBRATION_REQUESTS = 200
FORMAL_FINAL_REQUESTS = 10_000
DEFAULT_CALIBRATION_WORKLOAD = (
    ROOT / "results/hybrid_vector_db/figure5_r35_amazon_calibration.csv"
)
DEFAULT_MEASUREMENT_WORKLOAD = (
    ROOT / "results/hybrid_vector_db/figure5_r35_amazon_measurement.csv"
)
ROUTE_REPORTING_CONTRACT = (
    "configured filterStrategy only; Weaviate may apply an automatic internal "
    "fallback; the effective internal route is not observed"
)
TIMED_RESPONSE_FIELDS = ("row_id", "_additional.distance")
QPS_DEFINITION = "not_measured_by_single_client_latency_run"
HNSW_EF_SOURCE_URL = (
    "https://github.com/weaviate/weaviate/blob/v1.38.0/"
    "adapters/repos/db/vector/hnsw/search.go"
)
HNSW_EF_BUDGET_PROOF = {
    "version": "v1.38.0",
    "source": HNSW_EF_SOURCE_URL,
    "search_time_ef_lines": "search.go:40-62",
    "hnsw_dispatch_lines": "search.go:93-111",
    "queue_budget_lines": "search.go:348-349",
    "claim": (
        "for this runner's positive ascending ef values and fixed k, searchTimeEF returns "
        "max(configured ef, k); knnSearchByVector receives that value and HNSW candidate/result "
        "queues are acquired with ef capacity, so increasing configured ef cannot reduce the "
        "algorithmic HNSW search budget"
    ),
    "internal_route_observed": False,
}


FilterSpec = baseline.FilterSpec
TruthEntry = baseline.TruthEntry
QueryResult = baseline.QueryResult


@dataclass(frozen=True)
class WorkloadRequest:
    request_no: int
    query_no: int
    query_id: int
    filter_name: str


def formal_protocol(args: argparse.Namespace) -> bool:
    return args.protocol == FORMAL_EXECUTION_PROTOCOL


def current_protocol(args: argparse.Namespace) -> str:
    return CURRENT_PROTOCOL if formal_protocol(args) else LEGACY_CURRENT_PROTOCOL


def protocol_query_nos(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if formal_protocol(args):
        return tuple(range(FORMAL_CALIBRATION_REQUESTS)), tuple(
            range(FORMAL_CALIBRATION_REQUESTS, FORMAL_CALIBRATION_REQUESTS + FORMAL_FINAL_REQUESTS)
        )
    return tuple(CALIBRATION_QUERY_NOS), tuple(FINAL_QUERY_NOS)


def protocol_repeats(args: argparse.Namespace) -> tuple[int, int]:
    if formal_protocol(args):
        return FORMAL_CALIBRATION_REPEATS, FORMAL_FINAL_REPEATS
    return CALIBRATION_REPEATS, FINAL_REPEATS


def load_frozen_workload(
    path: Path,
    *,
    split: str,
    expected_requests: int,
    expected_query_nos: Sequence[int],
    query_ids: dict[int, int],
    filter_names: set[str],
) -> tuple[WorkloadRequest, ...]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or ())
            required = {"request_no", "query_no", "query_id", "filter_name", "split"}
            if not required <= fields:
                raise ValueError(
                    f"{path}: workload is missing fields {sorted(required - fields)}"
                )
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise ValueError(f"cannot read frozen {split} workload {path}: {exc}") from exc
    if len(rows) != expected_requests:
        raise ValueError(
            f"{path}: expected {expected_requests} {split} requests, got {len(rows)}"
        )
    requests: list[WorkloadRequest] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            request = WorkloadRequest(
                request_no=int(row["request_no"]),
                query_no=int(row["query_no"]),
                query_id=int(row["query_id"]),
                filter_name=str(row["filter_name"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: malformed workload row {row_number}") from exc
        if row["split"] != split:
            raise ValueError(
                f"{path}: row {row_number} split={row['split']!r}, expected {split!r}"
            )
        if request.filter_name not in filter_names:
            raise ValueError(
                f"{path}: row {row_number} has unknown filter {request.filter_name!r}"
            )
        if query_ids.get(request.query_no) != request.query_id:
            raise ValueError(
                f"{path}: row {row_number} query mapping disagrees with exact GT"
            )
        requests.append(request)
    expected_request_nos = set(range(expected_requests))
    expected_queries = {int(value) for value in expected_query_nos}
    observed_request_nos = {request.request_no for request in requests}
    observed_query_nos = {request.query_no for request in requests}
    if (
        observed_request_nos != expected_request_nos
        or len(observed_request_nos) != len(requests)
    ):
        raise ValueError(f"{path}: request_no coverage is not exactly 0..{expected_requests - 1}")
    if observed_query_nos != expected_queries or len(observed_query_nos) != len(requests):
        raise ValueError(f"{path}: query_no coverage does not match the frozen {split} split")
    observed_filters = {request.filter_name for request in requests}
    if observed_filters != filter_names:
        raise ValueError(
            f"{path}: filter coverage mismatch: "
            f"missing={sorted(filter_names - observed_filters)} "
            f"extra={sorted(observed_filters - filter_names)}"
        )
    return tuple(sorted(requests, key=lambda request: request.request_no))


def legacy_requests(
    query_nos: Sequence[int], query_ids: dict[int, int], filter_name: str,
) -> tuple[WorkloadRequest, ...]:
    return tuple(
        WorkloadRequest(int(query_no), int(query_no), int(query_ids[int(query_no)]), filter_name)
        for query_no in query_nos
    )


def requests_for_filter(
    requests: Sequence[WorkloadRequest], filter_name: str,
) -> tuple[WorkloadRequest, ...]:
    selected = tuple(request for request in requests if request.filter_name == filter_name)
    if not selected:
        raise ValueError(f"frozen workload has no requests for filter {filter_name}")
    return selected


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _manifest_file_record(
    owner: Path, value: Any, label: str, *, verify_file: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing from {owner}")
    path_value = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label}.path is invalid in {owner}")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label}.sha256 is invalid in {owner}")
    path = Path(path_value)
    if not path.is_absolute():
        path = owner.parent / path
    path = path.resolve()
    if verify_file:
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
        actual = baseline.sha256_file(path)
        if actual != digest:
            raise ValueError(
                f"{label} SHA256 mismatch: declared={digest} actual={actual}"
            )
    return {"path": str(path), "sha256": digest}


def _require_same_file(
    actual_path: Path, actual_sha256: str, record: dict[str, Any], label: str,
) -> None:
    if Path(record["path"]).resolve() != actual_path.resolve():
        raise ValueError(
            f"{label} path mismatch: manifest={record['path']} requested={actual_path}"
        )
    if record["sha256"] != actual_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch: manifest={record['sha256']} "
            f"requested={actual_sha256}"
        )


def validate_current_input_manifests(
    args: argparse.Namespace,
    filters: Sequence[FilterSpec],
    source_file_hashes: dict[str, str],
) -> dict[str, Any]:
    truth_path = args.truth_manifest.resolve()
    import_path = args.import_manifest.resolve()
    truth_manifest = _load_json_object(truth_path, "truth manifest")
    import_manifest = _load_json_object(import_path, "import manifest")

    if truth_manifest.get("artifact_valid") is not True:
        raise ValueError("truth manifest is not artifact_valid")
    if int(truth_manifest.get("k", -1)) != K:
        raise ValueError(f"truth manifest does not prove k={K}")
    if int(truth_manifest.get("filters", -1)) != len(baseline.FILTERS):
        raise ValueError("truth manifest does not cover the complete filter matrix")
    truth_outputs = truth_manifest.get("outputs")
    truth_inputs = truth_manifest.get("inputs")
    if not isinstance(truth_outputs, dict) or not isinstance(truth_inputs, dict):
        raise ValueError("truth manifest input/output provenance is incomplete")
    truth_csv = _manifest_file_record(
        truth_path, truth_outputs.get("truth_csv"), "truth outputs.truth_csv",
        verify_file=False,
    )
    truth_filters = _manifest_file_record(
        truth_path, truth_inputs.get("filters_csv"), "truth inputs.filters_csv",
        verify_file=False,
    )
    truth_fbin = _manifest_file_record(
        truth_path, truth_inputs.get("fbin"), "truth inputs.fbin",
        verify_file=False,
    )
    _require_same_file(args.truth_csv, source_file_hashes["truth_csv"], truth_csv, "truth CSV")
    _require_same_file(
        args.filters_csv, source_file_hashes["filters_csv"], truth_filters, "filters CSV"
    )
    _require_same_file(args.fbin, source_file_hashes["fbin"], truth_fbin, "vectors fbin")

    postgres = truth_inputs.get("postgres")
    population = postgres.get("query_population") if isinstance(postgres, dict) else None
    query_source = population.get("query_source") if isinstance(population, dict) else None
    if not isinstance(query_source, dict):
        raise ValueError("truth manifest does not bind the current query cohort")
    cohort_csv = _manifest_file_record(
        truth_path, query_source.get("cohort_csv"), "truth query cohort CSV"
    )
    cohort_manifest = _manifest_file_record(
        truth_path, query_source.get("manifest"), "truth query cohort manifest"
    )
    cohort_payload = _load_json_object(
        Path(cohort_manifest["path"]), "query cohort manifest"
    )
    if cohort_payload.get("artifact_valid") is not True:
        raise ValueError("query cohort manifest is not artifact_valid")

    if (
        import_manifest.get("artifact") != "weaviate_amazon10m_formal_corpus"
        or import_manifest.get("artifact_valid") is not True
        or import_manifest.get("status") != "complete"
    ):
        raise ValueError("import manifest is not a complete formal Weaviate corpus")
    input_files = import_manifest.get("input_files")
    completion = import_manifest.get("completion_gates")
    if not isinstance(input_files, dict) or not isinstance(completion, dict):
        raise ValueError("import manifest corpus provenance is incomplete")
    import_filters = _manifest_file_record(
        import_path, input_files.get("filters_csv"), "import filters CSV",
        verify_file=False,
    )
    import_fbin = _manifest_file_record(
        import_path, input_files.get("vectors_fbin"), "import vectors fbin",
        verify_file=False,
    )
    # The attributes file is already covered by the immutable import-manifest
    # digest. Avoid re-hashing a 1.3 GB file at every shard launch.
    import_attributes = _manifest_file_record(
        import_path,
        input_files.get("attributes_csv"),
        "import attributes CSV",
        verify_file=False,
    )
    _require_same_file(
        args.filters_csv, source_file_hashes["filters_csv"], import_filters,
        "import filters CSV",
    )
    _require_same_file(
        args.fbin, source_file_hashes["fbin"], import_fbin, "import vectors fbin"
    )
    if completion.get("passed") is not True:
        raise ValueError("import manifest completion gate did not pass")
    total_rows = completion.get("total_rows")
    valid_rows = completion.get("embedding_valid_rows")
    if not isinstance(total_rows, dict) or int(total_rows.get("actual", -1)) != baseline.EXPECTED_ROWS:
        raise ValueError("import manifest total-row gate is incompatible")
    if not isinstance(valid_rows, dict) or int(valid_rows.get("actual", -1)) != 9_979_556:
        raise ValueError("import manifest candidate-universe gate is incompatible")
    count_records = completion.get("filter_counts")
    if not isinstance(count_records, dict):
        raise ValueError("import manifest filter-count coverage is missing")
    expected_names = {spec.name for spec in baseline.FILTERS}
    if set(count_records) != expected_names:
        raise ValueError("import manifest does not cover the complete current filter matrix")
    for spec in filters:
        record = count_records.get(spec.name)
        if (
            not isinstance(record, dict)
            or record.get("passed") is not True
            or int(record.get("actual", -1)) != spec.expected_rows
        ):
            raise ValueError(f"import manifest filter count mismatch for {spec.name}")

    return {
        "protocol": current_protocol(args),
        "truth_manifest": {
            "path": str(truth_path), "sha256": baseline.sha256_file(truth_path)
        },
        "import_manifest": {
            "path": str(import_path), "sha256": baseline.sha256_file(import_path)
        },
        "truth_csv": truth_csv,
        "filters_csv": truth_filters,
        "vectors_fbin": truth_fbin,
        "query_cohort_csv": cohort_csv,
        "query_cohort_manifest": cohort_manifest,
        "import_attributes_csv": import_attributes,
        "corpus_gates": {
            "rows": baseline.EXPECTED_ROWS,
            "candidate_rows": 9_979_556,
            "filter_count_coverage": len(count_records),
            "complete_filter_matrix": len(baseline.FILTERS),
        },
    }


def schema_gate(
    schema: dict[str, Any], strategy: str | None = None, ef: int | None = None,
    flat_search_cutoff: int | None = None,
) -> list[str]:
    """Gate all three mutable HNSW settings after every schema PUT/readback."""
    errors = baseline.schema_gate(schema, None, ef)
    config = schema.get("vectorIndexConfig") or {}
    if strategy is not None and str(config.get("filterStrategy", "")).lower() != strategy.lower():
        errors.append(f"filterStrategy read-back mismatch: expected={strategy} actual={config.get('filterStrategy')}")
    if flat_search_cutoff is not None:
        try:
            actual = int(config.get("flatSearchCutoff"))
        except (TypeError, ValueError):
            actual = -1
        if actual != int(flat_search_cutoff):
            errors.append(
                "flatSearchCutoff read-back mismatch: "
                f"expected={flat_search_cutoff} actual={config.get('flatSearchCutoff')}"
            )
    return errors


def verify_schema(
    schema: dict[str, Any], strategy: str | None = None, ef: int | None = None,
    flat_search_cutoff: int | None = None,
) -> dict[str, Any]:
    errors = schema_gate(schema, strategy, ef, flat_search_cutoff)
    if errors:
        raise RuntimeError("schema gate failed: " + "; ".join(errors))
    return schema


def put_schema_definition(
    base_url: str, definition: dict[str, Any], *, timeout: float, retries: int,
    strategy: str | None = None, ef: int | None = None,
    flat_search_cutoff: int | None = None,
) -> tuple[dict[str, Any], int]:
    _, put_retries = baseline.request_json(
        base_url, f"/v1/schema/{CLASS_NAME}", definition, method="PUT",
        timeout=timeout, retries=retries,
    )
    readback, get_retries = baseline.request_json(
        base_url, f"/v1/schema/{CLASS_NAME}", timeout=timeout, retries=retries
    )
    verify_schema(readback, strategy, ef, flat_search_cutoff)
    baseline._verify_full_schema_readback(definition, readback)
    return readback, put_retries + get_retries


def put_hnsw_config(
    base_url: str, strategy: str, ef: int, flat_search_cutoff: int,
    timeout: float, retries: int,
) -> tuple[dict[str, Any], float, int]:
    started = time.perf_counter()
    current, get_retries = baseline.request_json(
        base_url, f"/v1/schema/{CLASS_NAME}", timeout=timeout, retries=retries
    )
    updated = copy.deepcopy(current)
    config = dict(updated.get("vectorIndexConfig") or {})
    config.update({
        "filterStrategy": strategy,
        "ef": int(ef),
        "flatSearchCutoff": int(flat_search_cutoff),
    })
    updated["vectorIndexConfig"] = config
    readback, put_retries = put_schema_definition(
        base_url, updated, timeout=timeout, retries=retries, strategy=strategy,
        ef=ef, flat_search_cutoff=flat_search_cutoff,
    )
    return readback, (time.perf_counter() - started) * 1000.0, get_retries + put_retries


def expected_route(
    spec: FilterSpec, flat_search_cutoff: int, strategy: str | None = None,
) -> dict[str, Any]:
    """Report configuration only; Weaviate does not expose its effective route."""
    return {
        "configured_filter_strategy": strategy or NA,
        "configured_flat_search_cutoff": int(flat_search_cutoff),
        "automatic_fallback": "possible_service_internal",
        "route_observed": False,
        "effective_route": NA,
        "reporting_contract": ROUTE_REPORTING_CONTRACT,
    }


def effective_cutoffs(spec: FilterSpec, grid: Sequence[int]) -> tuple[int, int]:
    """Return one representative for each route semantics present in v1.38.0."""
    values = sorted(set(int(value) for value in grid))
    if 0 not in values:
        raise ValueError("flat search cutoff grid must include 0 as the HNSW representative")
    flat_values = [value for value in values if value > spec.expected_rows]
    if not flat_values:
        raise ValueError(
            f"flat search cutoff grid has no flat-route value for {spec.name}: "
            f"max_cutoff={values[-1]} expected_rows={spec.expected_rows}"
        )
    return 0, flat_values[0]


def cutoff_equivalence_proof(
    spec: FilterSpec,
    grid: Sequence[int],
    ef_values: Sequence[int] = DEFAULT_EF_VALUES,
    strategies: Sequence[str] = DEFAULT_FILTER_STRATEGIES,
) -> dict[str, Any]:
    values = sorted(set(int(value) for value in grid))
    hnsw_representative, flat_representative = effective_cutoffs(spec, values)
    hnsw_values = [value for value in values if value <= spec.expected_rows]
    flat_values = [value for value in values if value > spec.expected_rows]
    representative_for = {
        str(value): hnsw_representative if value <= spec.expected_rows else flat_representative
        for value in values
    }
    return {
        "filter_name": spec.name,
        "allow_list_rows_basis": spec.expected_rows,
        "basis": "predeclared cutoff-grid compression",
        "effective_internal_route_observed": False,
        "effective_cutoffs": [hnsw_representative, flat_representative],
        "configuration_groups": [
            {"group": "cutoff_not_above_filter_cardinality",
             "representative_cutoff": hnsw_representative,
             "equivalent_grid_values": hnsw_values,
             "filter_strategy_and_ef_consulted": True},
            {"group": "cutoff_above_filter_cardinality",
             "representative_cutoff": flat_representative,
             "equivalent_grid_values": flat_values,
             "filter_strategy_and_ef_consulted": False},
        ],
        "flat_configuration_equivalence": {
            "claim_scope": "configured control representative, not an observed route",
            "representative": {
                "configured_filter_strategy": FLAT_STRATEGY_REPRESENTATIVE,
                "ef": int(ef_values[0]),
                "flat_search_cutoff": flat_representative,
            },
            "equivalent_filter_strategies": list(strategies),
            "equivalent_ef_values": [int(value) for value in ef_values],
            "equivalent_cutoff_values": flat_values,
            "equivalent_configuration_count": len(strategies) * len(ef_values) * len(flat_values),
            "timed_representative_count": 1,
            "skipped_equivalent_configuration_count": (
                len(strategies) * len(ef_values) * len(flat_values) - 1
            ),
        },
        "representative_for_grid_value": representative_for,
        "skipped_equivalent_cutoffs": [
            value for value in values if value not in {hnsw_representative, flat_representative}
        ],
        "complete_effective_semantics_coverage": True,
    }


def effective_cutoff_grid(filters: Sequence[FilterSpec], grid: Sequence[int]) -> list[int]:
    return sorted({cutoff for spec in filters for cutoff in effective_cutoffs(spec, grid)})


def calibration_configuration_schedule(
    strategies: Sequence[str], flat_search_cutoffs: Sequence[int], ef_values: Sequence[int]
) -> list[tuple[int, str, int, int]]:
    """Schedule every flat representative before the ascending HNSW grids."""
    schedule: list[tuple[int, str, int, int]] = []
    cutoffs = [int(value) for value in flat_search_cutoffs]
    if not cutoffs or cutoffs[0] != 0:
        raise ValueError("effective cutoff schedule must begin with HNSW representative 0")
    for cutoff in cutoffs[1:]:
        schedule.append(
            (len(schedule), FLAT_STRATEGY_REPRESENTATIVE, cutoff, int(ef_values[0]))
        )
    for ef_index, ef in enumerate(ef_values):
        for strategy in baseline.rotated(strategies, ef_index):
            schedule.append((len(schedule), strategy, 0, int(ef)))
    return schedule


def _route_summaries(
    summaries: Sequence[dict[str, Any]], strategy: str, filter_name: str, cutoff: int
) -> list[dict[str, Any]]:
    return sorted(
        (row for row in summaries
         if row.get("configured_filter_strategy") == strategy
         and row.get("filter_name") == filter_name
         and int(row.get("flat_search_cutoff", -1)) == int(cutoff)),
        key=lambda row: int(row["ef"]),
    )


def _candidate_summaries(
    summaries: Sequence[dict[str, Any]], strategy: str, filter_name: str
) -> list[dict[str, Any]]:
    return [row for row in summaries if row.get("configured_filter_strategy") == strategy and row.get("filter_name") == filter_name]


def required_calibration_lcb(target: float, margin: float = DEFAULT_CALIBRATION_LCB_MARGIN) -> float:
    """Return the predeclared LCB95 threshold for a target."""
    del margin
    return float(target)


def reaches_calibration_lcb(
    summary: dict[str, Any], target: float,
    calibration_lcb_margin: float = DEFAULT_CALIBRATION_LCB_MARGIN,
) -> bool:
    threshold = required_calibration_lcb(target, calibration_lcb_margin)
    return bool(
        summary.get("complete") is True
        and baseline._finite_number(summary.get(CALIBRATION_STOP_METRIC))
        and float(summary[CALIBRATION_STOP_METRIC]) >= threshold
    )


def select_fastest_config(
    summaries: Sequence[dict[str, Any]], target: float,
    calibration_lcb_margin: float = DEFAULT_CALIBRATION_LCB_MARGIN,
) -> dict[str, Any] | None:
    eligible = [
        row for row in summaries
        if reaches_calibration_lcb(row, target, calibration_lcb_margin)
        and baseline._finite_number(row.get("latency_mean_ms"))
    ]
    return min(
        eligible,
        key=lambda row: (
            float(row["latency_mean_ms"]), int(row["flat_search_cutoff"]), int(row["ef"]),
            str(row["configured_filter_strategy"]),
        ),
    ) if eligible else None


def hnsw_grid_fully_exhausted(
    summaries: Sequence[dict[str, Any]], spec: FilterSpec,
    ef_values: Sequence[int],
) -> bool:
    return all(
        route_grid_proof(
            _route_summaries(summaries, strategy, spec.name, 0), ef_values
        )["grid_exhausted_without_errors"]
        for strategy in DEFAULT_FILTER_STRATEGIES
    )


def select_max_recall_config(
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    eligible = [
        row for row in summaries
        if row.get("complete") is True
        and baseline._finite_number(row.get("recall_mean"))
        and baseline._finite_number(row.get("latency_mean_ms"))
    ]
    return min(
        eligible,
        key=lambda row: (
            -float(row["recall_mean"]), float(row["latency_mean_ms"]),
            int(row["ef"]), str(row["configured_filter_strategy"]),
        ),
    ) if eligible else None


def select_conservative_config(
    summaries: Sequence[dict[str, Any]], target: float, spec: FilterSpec,
    flat_search_cutoffs: Sequence[int], ef_values: Sequence[int],
    calibration_lcb_margin: float = DEFAULT_CALIBRATION_LCB_MARGIN,
) -> dict[str, Any] | None:
    """Select by LCB95, falling back only after the complete HNSW grid."""
    del flat_search_cutoffs
    hnsw_candidates = [
        row for row in summaries
        if row.get("filter_name") == spec.name
        and int(row.get("flat_search_cutoff", -1)) == 0
    ]
    selected = select_fastest_config(hnsw_candidates, target, calibration_lcb_margin)
    if selected is not None:
        return {
            **selected,
            "selection_mode": "calibration_lcb95_qualified_fastest",
            "calibration_selection_policy": CALIBRATION_SELECTION_POLICY,
            "calibration_target_recall": float(target),
            "calibration_lcb95_qualified": True,
            "fallback_used": False,
        }
    if not hnsw_grid_fully_exhausted(summaries, spec, ef_values):
        return None
    fallback = select_max_recall_config(hnsw_candidates)
    if fallback is None:
        return None
    return {
        **fallback,
        "selection_mode": "calibration_max_mean_recall_fallback",
        "calibration_selection_policy": CALIBRATION_SELECTION_POLICY,
        "calibration_target_recall": float(target),
        "calibration_lcb95_qualified": False,
        "fallback_used": True,
    }


def route_grid_proof(candidates: Sequence[dict[str, Any]], ef_values: Sequence[int]) -> dict[str, Any]:
    expected = [int(value) for value in ef_values]
    measured = [int(row["ef"]) for row in candidates]
    return {
        "measured_efs": measured,
        "grid_exhausted_without_errors": (
            measured == expected and len(candidates) == len(expected)
            and all(row.get("complete") is True for row in candidates)
        ),
    }


def _complete_finite_latency_ci(summary: dict[str, Any]) -> bool:
    low = summary.get("latency_ci95_low_ms")
    high = summary.get("latency_ci95_high_ms")
    return bool(
        summary.get("complete") is True
        and baseline._finite_number(low)
        and baseline._finite_number(high)
        and float(low) <= float(high)
    )


def hnsw_dominance_proof(
    summaries: Sequence[dict[str, Any]], strategy: str, spec: FilterSpec,
    flat_search_cutoffs: Sequence[int], ef_values: Sequence[int], guard: float,
) -> dict[str, Any]:
    flat = flat_configuration(spec, flat_search_cutoffs, ef_values)
    flat_candidates = [
        row for row in summaries
        if row.get("configured_filter_strategy") == flat["configured_filter_strategy"]
        and row.get("filter_name") == spec.name
        and int(row.get("flat_search_cutoff", -1)) == int(flat["flat_search_cutoff"])
        and int(row.get("ef", -1)) == int(flat["ef"])
    ]
    flat_summary = flat_candidates[0] if len(flat_candidates) == 1 else None
    flat_exact = bool(
        flat_summary is not None
        and flat_summary.get("complete") is True
        and baseline._finite_number(flat_summary.get("recall_mean"))
        and float(flat_summary["recall_mean"]) == 1.0
    )
    flat_finite_ci = bool(flat_summary is not None and _complete_finite_latency_ci(flat_summary))
    hnsw = _route_summaries(summaries, strategy, spec.name, 0)
    measured_efs = [int(row["ef"]) for row in hnsw]
    expected_efs = [int(value) for value in ef_values]
    ascending_grid_prefix = measured_efs == expected_efs[:len(measured_efs)]
    prefix_complete = bool(
        hnsw and ascending_grid_prefix
        and all(row.get("complete") is True for row in hnsw)
    )
    two_consecutive = bool(prefix_complete and len(hnsw) >= 2)
    previous = hnsw[-2] if len(hnsw) >= 2 else None
    latest = hnsw[-1] if hnsw else None
    hnsw_finite = bool(
        previous is not None and latest is not None
        and all(
            baseline._finite_number(row.get(field))
            for row in (previous, latest)
            for field in ("latency_mean_ms", "latency_ci95_low_ms")
        )
    )
    means_nondecreasing = bool(
        hnsw_finite
        and float(latest["latency_mean_ms"]) >= float(previous["latency_mean_ms"])
    )
    ci_lowers_nondecreasing = bool(
        hnsw_finite
        and float(latest["latency_ci95_low_ms"]) >= float(previous["latency_ci95_low_ms"])
    )
    guarded_ci_dominance = bool(
        hnsw_finite and flat_finite_ci
        and float(latest["latency_ci95_low_ms"])
        > float(flat_summary["latency_ci95_high_ms"]) * float(guard)
    )
    gates = {
        "flat_representative_unique": len(flat_candidates) == 1,
        "flat_complete_exact_recall": flat_exact,
        "flat_finite_latency_ci": flat_finite_ci,
        "hnsw_efs_are_an_ascending_grid_prefix": ascending_grid_prefix,
        "hnsw_prefix_complete_without_errors": prefix_complete,
        "two_consecutive_complete_hnsw_blocks": two_consecutive,
        "last_two_hnsw_latency_means_nondecreasing": means_nondecreasing,
        "last_two_hnsw_ci_lowers_nondecreasing": ci_lowers_nondecreasing,
        "latest_hnsw_ci_lower_strictly_above_guarded_flat_ci_upper": guarded_ci_dominance,
    }
    return {
        "dominance_proven": all(gates.values()),
        "guard": float(guard),
        "source_budget_proof": HNSW_EF_BUDGET_PROOF,
        "gates": gates,
        "flat_representative": flat,
        "flat_evidence": {
            "recall_mean": flat_summary.get("recall_mean", NA) if flat_summary else NA,
            "recall_lcb95": flat_summary.get("recall_lcb95", NA) if flat_summary else NA,
            "latency_ci95_low_ms": flat_summary.get("latency_ci95_low_ms", NA) if flat_summary else NA,
            "latency_ci95_high_ms": flat_summary.get("latency_ci95_high_ms", NA) if flat_summary else NA,
        },
        "hnsw_evidence": [
            {
                "ef": row.get("ef"),
                "latency_mean_ms": row.get("latency_mean_ms", NA),
                "latency_ci95_low_ms": row.get("latency_ci95_low_ms", NA),
            }
            for row in hnsw[-2:]
        ],
    }


def hnsw_route_termination(
    summaries: Sequence[dict[str, Any]], strategy: str, spec: FilterSpec,
    flat_search_cutoffs: Sequence[int], ef_values: Sequence[int], highest_target: float,
    dominance_guard: float, calibration_lcb_margin: float = 0.0,
) -> dict[str, Any]:
    route = _route_summaries(summaries, strategy, spec.name, 0)
    reached = [
        index for index, summary in enumerate(route)
        if reaches_calibration_lcb(
            summary, highest_target, calibration_lcb_margin
        )
    ]
    dominance = hnsw_dominance_proof(
        summaries, strategy, spec, flat_search_cutoffs, ef_values, dominance_guard
    )
    if reached:
        reason = "highest_target_lcb95_reached"
        stopped_after_ef = int(route[reached[0]]["ef"])
    elif route_grid_proof(route, ef_values)["grid_exhausted_without_errors"]:
        reason = "full_grid_exhausted"
        stopped_after_ef = int(route[-1]["ef"])
    else:
        reason = "in_progress"
        stopped_after_ef = int(route[-1]["ef"]) if route else NA
    return {
        "termination_reason": reason,
        "stopped_after_ef": stopped_after_ef,
        "measured_efs": [int(row["ef"]) for row in route],
        "stop_metric": CALIBRATION_STOP_METRIC,
        "dominance_is_diagnostic_only": True,
        "dominance_proof": dominance,
    }


def hnsw_route_target_status(
    route: Sequence[dict[str, Any]], target: float, termination: dict[str, Any],
    calibration_lcb_margin: float = 0.0,
) -> str:
    if select_fastest_config(route, target, calibration_lcb_margin) is not None:
        return "attained"
    reason = termination["termination_reason"]
    if reason == "full_grid_exhausted":
        return "not_lcb_qualified_on_hnsw_grid"
    return "incomplete"


def flat_configuration(spec: FilterSpec, flat_search_cutoffs: Sequence[int], ef_values: Sequence[int]) -> dict[str, Any]:
    return {
        "configured_filter_strategy": FLAT_STRATEGY_REPRESENTATIVE,
        "filter_name": spec.name,
        "flat_search_cutoff": effective_cutoffs(spec, flat_search_cutoffs)[1],
        "ef": int(ef_values[0]),
    }


def configuration_grid_proof(
    summaries: Sequence[dict[str, Any]], spec: FilterSpec,
    flat_search_cutoffs: Sequence[int], ef_values: Sequence[int], targets: Sequence[float],
    dominance_guard: float, calibration_lcb_margin: float = 0.0,
) -> dict[str, Any]:
    hnsw_routes: dict[str, dict[str, Any]] = {}
    for strategy in DEFAULT_FILTER_STRATEGIES:
        route = _route_summaries(summaries, strategy, spec.name, 0)
        termination = hnsw_route_termination(
            summaries, strategy, spec, flat_search_cutoffs, ef_values,
            float(targets[-1]), dominance_guard, calibration_lcb_margin,
        )
        hnsw_routes[strategy] = {
            "ef_grid_proof": route_grid_proof(route, ef_values),
            "termination": termination,
            "target_statuses": {
                str(target): hnsw_route_target_status(
                    route, target, termination, calibration_lcb_margin
                )
                for target in targets
            },
        }
    flat = flat_configuration(spec, flat_search_cutoffs, ef_values)
    flat_candidates = [
        row for row in summaries
        if row.get("configured_filter_strategy") == flat["configured_filter_strategy"]
        and row.get("filter_name") == spec.name
        and int(row.get("flat_search_cutoff", -1)) == flat["flat_search_cutoff"]
        and int(row.get("ef", -1)) == flat["ef"]
    ]
    flat_complete = len(flat_candidates) == 1 and flat_candidates[0].get("complete") is True
    terminal_reasons = {
        route["termination"]["termination_reason"] for route in hnsw_routes.values()
    }
    return {
        "hnsw_routes": hnsw_routes,
        "flat_route": {
            "representative": flat,
            "observed_blocks": len(flat_candidates),
            "complete_without_errors": flat_complete,
        },
        "equivalence_proof": cutoff_equivalence_proof(
            spec, flat_search_cutoffs, ef_values
        ),
        "effective_grid_resolved_without_errors": (
            flat_complete
            and all(reason != "in_progress" for reason in terminal_reasons)
        ),
        "all_hnsw_routes_fully_exhausted_without_errors": (
            flat_complete and terminal_reasons == {"full_grid_exhausted"}
        ),
    }


def calibration_target_status(
    summaries: Sequence[dict[str, Any]], spec: FilterSpec, target: float,
    flat_search_cutoffs: Sequence[int], ef_values: Sequence[int], targets: Sequence[float],
    dominance_guard: float, calibration_lcb_margin: float = DEFAULT_CALIBRATION_LCB_MARGIN,
) -> str:
    candidates = [row for row in summaries if row.get("filter_name") == spec.name]
    if select_conservative_config(
        candidates, target, spec, flat_search_cutoffs, ef_values, calibration_lcb_margin
    ):
        return "selected"
    proof = configuration_grid_proof(
        summaries, spec, flat_search_cutoffs, ef_values, targets, dominance_guard,
        calibration_lcb_margin,
    )
    if proof["all_hnsw_routes_fully_exhausted_without_errors"]:
        return "unattainable_on_grid"
    return "incomplete_grid"


def validate_monotone_calibration_state(
    summaries: Sequence[dict[str, Any]], filters: Sequence[FilterSpec],
    flat_search_cutoffs: Sequence[int], ef_values: Sequence[int], highest_target: float,
    dominance_guard: float, calibration_lcb_margin: float = 0.0,
) -> None:
    if any(row.get("complete") is not True for row in summaries):
        raise RuntimeError(
            "checkpoint calibration summary is not a complete error-free block"
        )
    expected_routes = {
        (strategy, spec.name, 0)
        for strategy in DEFAULT_FILTER_STRATEGIES for spec in filters
    } | {
        (FLAT_STRATEGY_REPRESENTATIVE, spec.name, effective_cutoffs(spec, flat_search_cutoffs)[1])
        for spec in filters
    }
    actual_routes = {
        (str(row.get("configured_filter_strategy")), str(row.get("filter_name")), int(row.get("flat_search_cutoff", -1)))
        for row in summaries
    }
    if not actual_routes <= expected_routes:
        raise RuntimeError("checkpoint calibration summary has an unknown strategy/filter/cutoff route")
    expected_efs = [int(value) for value in ef_values]
    specs_by_name = {spec.name: spec for spec in filters}
    if any(int(row.get("flat_search_cutoff", -1)) == 0 for row in summaries):
        for spec in filters:
            flat = flat_configuration(spec, flat_search_cutoffs, ef_values)
            flat_rows = _route_summaries(
                summaries, str(flat["configured_filter_strategy"]), spec.name,
                int(flat["flat_search_cutoff"]),
            )
            if len(flat_rows) != 1 or int(flat_rows[0]["ef"]) != int(flat["ef"]):
                raise RuntimeError(
                    "checkpoint contains HNSW calibration before every flat "
                    f"representative completed: {spec.name}"
                )
    for strategy, filter_name, cutoff in expected_routes:
        route = _route_summaries(summaries, strategy, filter_name, cutoff)
        measured = [int(row["ef"]) for row in route]
        if cutoff != 0:
            if measured not in ([], [expected_efs[0]]) or len(route) != len(measured):
                raise RuntimeError(
                    f"checkpoint flat route is not a single representative: "
                    f"{strategy}/{filter_name}/{cutoff}"
                )
            continue
        if measured != expected_efs[:len(measured)] or len(measured) != len(set(measured)):
            raise RuntimeError(f"checkpoint calibration grid is not an ascending ef prefix: {strategy}/{filter_name}/{cutoff}")
        unrelated = [
            row for row in summaries
            if not (
                row.get("configured_filter_strategy") == strategy
                and row.get("filter_name") == filter_name
                and int(row.get("flat_search_cutoff", -1)) == 0
            )
        ]
        for index in range(len(route)):
            prefix = unrelated + route[:index + 1]
            termination = hnsw_route_termination(
                prefix, strategy, specs_by_name[filter_name], flat_search_cutoffs,
                ef_values, highest_target, dominance_guard,
                calibration_lcb_margin,
            )
            if (termination["termination_reason"] == "highest_target_lcb95_reached"
                    and index != len(route) - 1):
                raise RuntimeError(
                    f"checkpoint continued after recomputable {termination['termination_reason']}: "
                    f"{strategy}/{filter_name}/{route[index]['ef']}"
                )


def select_filter_specs(
    filters: Sequence[FilterSpec], filter_names: Sequence[str] | None
) -> tuple[FilterSpec, ...]:
    if filter_names is None:
        return tuple(filters)
    names = list(filter_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("filter names must be non-empty and unique")
    by_name = {spec.name: spec for spec in filters}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown filter names: {unknown}")
    return tuple(by_name[name] for name in names)


def validate_service_identity(
    service_meta: dict[str, Any], expected_version: str, service_image_digest: str
) -> dict[str, str]:
    digest = service_image_digest.strip()
    if not digest:
        raise ValueError("--service-image-digest is required for formal execution")
    actual_version = service_meta.get("version")
    if actual_version != expected_version:
        raise RuntimeError(
            f"Weaviate service version mismatch: expected={expected_version!r} actual={actual_version!r}"
        )
    return {
        "expected_version": expected_version,
        "actual_version": str(actual_version),
        "service_image_digest": digest,
    }


def raise_after_schema_restore(
    primary_error: BaseException | None, restore_error: BaseException | None
) -> None:
    control_flow = (KeyboardInterrupt, SystemExit)
    if isinstance(primary_error, control_flow):
        if restore_error is not None:
            primary_error.add_note(f"schema restore also failed: {restore_error}")
        raise primary_error
    if isinstance(restore_error, control_flow):
        raise restore_error
    if primary_error is not None and restore_error is not None:
        raise RuntimeError(
            f"primary failure: {primary_error}; schema restore failure: {restore_error}"
        ) from primary_error
    if primary_error is not None:
        raise primary_error
    if restore_error is not None:
        raise restore_error


def group_selected_targets(
    selections: dict[tuple[str, float], dict[str, Any] | None],
) -> dict[tuple[str, str, int, int], list[float]]:
    grouped: dict[tuple[str, str, int, int], list[float]] = {}
    for (filter_name, target), winner in selections.items():
        if winner is not None:
            key = (
                str(winner["configured_filter_strategy"]), filter_name,
                int(winner["flat_search_cutoff"]), int(winner["ef"]),
            )
            grouped.setdefault(key, []).append(target)
    return {key: sorted(values) for key, values in grouped.items()}


def add_flat_exactness_controls(
    groups: dict[tuple[str, str, int, int], list[float]],
    filters: Sequence[FilterSpec], flat_search_cutoffs: Sequence[int], ef_values: Sequence[int],
) -> dict[tuple[str, str, int, int], list[float]]:
    result = {key: list(targets) for key, targets in groups.items()}
    for spec in filters:
        flat = flat_configuration(spec, flat_search_cutoffs, ef_values)
        key = (
            str(flat["configured_filter_strategy"]), spec.name,
            int(flat["flat_search_cutoff"]), int(flat["ef"]),
        )
        result.setdefault(key, [])
    return result


def calibration_query_budget(
    filters: Sequence[FilterSpec],
    ef_values: Sequence[int],
    warmup_queries: int,
    *,
    calibration_requests: int | None = None,
    calibration_repeats: int = CALIBRATION_REPEATS,
) -> dict[str, Any]:
    blocks = len(filters) * (len(DEFAULT_FILTER_STRATEGIES) * len(ef_values) + 1)
    effective_requests = (
        len(filters) * len(CALIBRATION_QUERY_NOS)
        if calibration_requests is None else int(calibration_requests)
    )
    timed_queries = (
        (len(DEFAULT_FILTER_STRATEGIES) * len(ef_values) + 1)
        * effective_requests
        * int(calibration_repeats)
    )
    warmup = blocks * int(warmup_queries)
    return {
        "schedule_order": "all per-filter flat representatives before any HNSW configuration",
        "flat_representative_blocks": len(filters),
        "maximum_hnsw_blocks": len(filters) * len(DEFAULT_FILTER_STRATEGIES) * len(ef_values),
        "maximum_effective_blocks_before_hnsw_early_stop": blocks,
        "maximum_timed_queries_before_hnsw_early_stop": timed_queries,
        "configured_warmup_queries": warmup,
        "maximum_total_service_queries": timed_queries + warmup,
    }


def _target_token(target: float | None) -> str:
    return NA if target is None else f"{float(target):.12g}"


def _block_key(
    phase: str,
    strategy: str,
    filter_name: str,
    cutoff: int,
    ef: int,
    target: float | None = None,
) -> tuple[str, str, str, int, int, str]:
    return phase, strategy, filter_name, int(cutoff), int(ef), _target_token(target)


def _block_record(
    phase: str,
    strategy: str,
    filter_name: str,
    cutoff: int,
    ef: int,
    requests: Sequence[WorkloadRequest] | None = None,
    repeats: int | None = None,
    *,
    target: float | None = None,
) -> dict[str, Any]:
    if requests is None:
        query_nos = (
            CALIBRATION_QUERY_NOS if phase == "calibration" else FINAL_QUERY_NOS
        )
        requests = legacy_requests(query_nos, {query_no: query_no for query_no in query_nos}, filter_name)
    if repeats is None:
        repeats = (
            CALIBRATION_REPEATS if phase == "calibration" else FINAL_REPEATS
        )
    return {
        "phase": phase, "configured_filter_strategy": strategy, "filter_name": filter_name,
        "flat_search_cutoff": int(cutoff), "ef": int(ef),
        "target_recall": _target_token(target),
        "request_nos": [request.request_no for request in requests],
        "query_nos": [request.query_no for request in requests],
        "repeats": int(repeats),
    }


def run_specification(
    args: argparse.Namespace, filters: Sequence[FilterSpec], query_ids: dict[int, int],
    source_hashes: dict[str, str], input_provenance: dict[str, Any],
    calibration_requests: Sequence[WorkloadRequest] = (),
    final_requests: Sequence[WorkloadRequest] = (),
) -> dict[str, Any]:
    calibration_repeats, final_repeats = protocol_repeats(args)
    return {
        "version": CHECKPOINT_VERSION, "class": CLASS_NAME, "source_hashes": source_hashes,
        "current_protocol": current_protocol(args),
        "execution_protocol": args.protocol,
        "input_provenance": input_provenance,
        "endpoint": {"host": args.host, "port": args.port},
        "expected_service_version": args.expected_service_version,
        "service_image_digest": args.service_image_digest.strip(),
        "configured_filter_strategies": list(DEFAULT_FILTER_STRATEGIES),
        "filter_names": [spec.name for spec in filters],
        "filters": [asdict(spec) for spec in filters], "ef_values": [int(value) for value in args.ef_values],
        "flat_search_cutoffs": [int(value) for value in args.flat_search_cutoffs],
        "effective_cutoffs_by_filter": {
            spec.name: list(effective_cutoffs(spec, args.flat_search_cutoffs)) for spec in filters
        },
        "effective_grid": {
            "hnsw": {
                "configured_filter_strategies": list(DEFAULT_FILTER_STRATEGIES),
                "ef_values": [int(value) for value in args.ef_values],
                "flat_search_cutoff": 0,
            },
            "flat_representatives": {
                spec.name: flat_configuration(spec, args.flat_search_cutoffs, args.ef_values)
                for spec in filters
            },
        },
        "hnsw_flat_dominance": {
            "guard": float(args.hnsw_dominance_guard),
            "source_budget_proof": HNSW_EF_BUDGET_PROOF,
            "diagnostic_only": True,
            "empirical_gate": (
                "flat exact complete finite CI; two complete HNSW points; latest HNSW CI lower "
                "> flat CI upper * guard; last two HNSW means and CI lowers nondecreasing"
            ),
        },
        "targets": [float(value) for value in args.targets], "k": K,
        "calibration": {
                        "request_nos": [request.request_no for request in calibration_requests],
                        "query_nos": [request.query_no for request in calibration_requests],
                        "repeats": calibration_repeats,
                        "warmup_queries": args.warmup_queries,
                        "selection_rule": CALIBRATION_SELECTION_RULE,
                        "qualification_metric": CALIBRATION_STOP_METRIC,
                        "qualification_operator": ">= target",
                        "bootstrap_ci_lcb": "qualification_and_early_stop",
                        "selection_policy": CALIBRATION_SELECTION_POLICY,
                        "fallback": CALIBRATION_FALLBACK,
                        "fallback_requires_full_hnsw_grid": True,
                        "early_stop_metric": CALIBRATION_STOP_METRIC,
                        "early_stop_target": float(args.targets[-1])},
        "final": {
                  "request_nos": [request.request_no for request in final_requests],
                  "query_nos": [request.query_no for request in final_requests],
                  "repeats": final_repeats,
                  "final_warmup_phase": "final_warmup",
                  "final_warmup_request_nos": [
                      request.request_no for request in final_requests[: args.warmup_queries]
                  ],
                  "final_warmup_queries_per_selected_configuration": args.warmup_queries,
                  "final_warmup_excluded_from_summaries": True,
                  "target_specific_measurement": formal_protocol(args)},
        "bootstrap_seed": args.bootstrap_seed,
        "workloads": {
            "calibration": [asdict(request) for request in calibration_requests],
            "final": [asdict(request) for request in final_requests],
        },
        "calibration_query_budget": calibration_query_budget(
            filters,
            args.ef_values,
            args.warmup_queries,
            calibration_requests=len(calibration_requests),
            calibration_repeats=calibration_repeats,
        ),
        "query_ids": {str(key): int(value) for key, value in sorted(query_ids.items())},
    }


def run_spec_hash(specification: dict[str, Any]) -> str:
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def schema_snapshot_hash(schema: dict[str, Any]) -> str:
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def schema_config_identity(schema: dict[str, Any]) -> tuple[str, int, int]:
    config = schema.get("vectorIndexConfig")
    if not isinstance(config, dict):
        raise RuntimeError("schema vectorIndexConfig is missing")
    strategy = config.get("filterStrategy")
    if not isinstance(strategy, str) or not strategy:
        raise RuntimeError("schema filterStrategy is missing or invalid")
    try:
        ef = int(config["ef"])
        cutoff = int(config["flatSearchCutoff"])
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("schema ef/flatSearchCutoff is missing or invalid") from None
    if cutoff < 0:
        raise RuntimeError("schema flatSearchCutoff is negative")
    return strategy, ef, cutoff


def validate_checkpoint_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint state is incomplete")
    required_lists = ("completed_blocks", "schema_records", "schema_timings", "node_records")
    if any(not isinstance(state.get(key), list) for key in required_lists):
        raise RuntimeError("checkpoint state is incomplete")
    original_schema = state.get("original_schema")
    if not isinstance(original_schema, dict):
        raise RuntimeError("checkpoint original_schema is missing or invalid")
    verify_schema(original_schema)
    schema_config_identity(original_schema)
    if state.get("original_schema_sha256") != schema_snapshot_hash(original_schema):
        raise RuntimeError("checkpoint original_schema hash mismatch")
    records = state["schema_records"]
    expected_record = {"phase": "original_schema_snapshot", "schema": original_schema}
    if not records or records[0] != expected_record:
        raise RuntimeError("checkpoint original_schema record is missing or inconsistent")
    return state


def _validate_checkpoint_blocks(payload: dict[str, Any], query_ids: dict[int, int]) -> None:
    state = payload.get("state")
    raw_rows = payload.get("raw_rows")
    calibration = payload.get("calibration_summaries")
    final = payload.get("final_results")
    if not all(isinstance(value, list) for value in (raw_rows, calibration, final)):
        raise RuntimeError("checkpoint is missing required state")
    state = validate_checkpoint_state(state)
    blocks = state.get("completed_blocks")
    if not isinstance(blocks, list):
        raise RuntimeError("checkpoint completed_blocks is invalid")
    seen: set[tuple[str, str, str, int, int, str]] = set()
    by_phase = {"calibration": set(), "final": set()}
    for block in blocks:
        if not isinstance(block, dict):
            raise RuntimeError("checkpoint contains a malformed block")
        try:
            phase, strategy, filter_name = block["phase"], block["configured_filter_strategy"], block["filter_name"]
            cutoff, ef = int(block["flat_search_cutoff"]), int(block["ef"])
            target_token = str(block["target_recall"])
            request_nos = [int(value) for value in block["request_nos"]]
            query_nos = [int(value) for value in block["query_nos"]]
            repeats = int(block["repeats"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("checkpoint block identity is invalid") from None
        if (
            phase not in by_phase
            or not isinstance(strategy, str)
            or not isinstance(filter_name, str)
            or repeats <= 0
            or len(request_nos) != len(query_nos)
            or not request_nos
            or len(set(request_nos)) != len(request_nos)
            or len(set(query_nos)) != len(query_nos)
        ):
            raise RuntimeError("checkpoint block identity is invalid")
        target = None if target_token == NA else float(target_token)
        key = _block_key(phase, strategy, filter_name, cutoff, ef, target)
        if key in seen:
            raise RuntimeError(f"checkpoint has duplicate completed block: {key}")
        seen.add(key)
        expected = {
            (request_no, query_no, repeat)
            for request_no, query_no in zip(request_nos, query_nos)
            for repeat in range(repeats)
        }
        observed: set[tuple[int, int, int]] = set()
        matching = [
            row for row in raw_rows if isinstance(row, dict) and row.get("phase") == phase
            and row.get("configured_filter_strategy") == strategy and row.get("filter_name") == filter_name
            and int(row.get("flat_search_cutoff", -1)) == cutoff and int(row.get("ef", -1)) == ef
            and str(row.get("target_recall", NA)) == target_token
        ]
        for row in matching:
            try:
                pair = (
                    int(row.get("request_no", row["query_no"])),
                    int(row["query_no"]),
                    int(row["repeat"]),
                )
                query_id = int(row["query_id"])
            except (KeyError, TypeError, ValueError):
                raise RuntimeError(f"checkpoint row is malformed for block {key}") from None
            if pair in observed or pair not in expected or query_id != query_ids.get(pair[1]):
                raise RuntimeError(f"checkpoint rows are incomplete or inconsistent for block {key}")
            observed.add(pair)
        if observed != expected or len(matching) != len(expected):
            raise RuntimeError(f"checkpoint rows are incomplete for block {key}")
        by_phase[phase].add(key)
    for phase, summaries in (("calibration", calibration), ("final", final)):
        keys = {
            _block_key(phase, str(row.get("configured_filter_strategy")), str(row.get("filter_name")),
                       int(row.get("flat_search_cutoff", -1)), int(row.get("ef", -1)),
                       None if str(row.get("target_recall", NA)) == NA
                       else float(row["target_recall"]))
            for row in summaries if isinstance(row, dict)
        }
        if keys != by_phase[phase] or len(keys) != len(summaries):
            raise RuntimeError(f"checkpoint {phase} summaries do not match completed blocks")


def load_checkpoint(path: Path, specification: dict[str, Any], query_ids: dict[int, int]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read checkpoint {path}: {exc}") from exc
    if (not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION
            or payload.get("run_spec") != specification
            or payload.get("run_spec_hash") != run_spec_hash(specification)):
        raise RuntimeError("checkpoint run-spec/hash mismatch; refusing to reuse measurements")
    _validate_checkpoint_blocks(payload, query_ids)
    return payload


def write_checkpoint(
    path: Path, specification: dict[str, Any], *, raw_rows: Sequence[dict[str, Any]],
    calibration_summaries: Sequence[dict[str, Any]], final_results: Sequence[dict[str, Any]],
    state: dict[str, Any],
) -> None:
    baseline.atomic_write_json(path, {
        "version": CHECKPOINT_VERSION, "run_spec": specification,
        "run_spec_hash": run_spec_hash(specification), "raw_rows": list(raw_rows),
        "calibration_summaries": list(calibration_summaries), "final_results": list(final_results),
        "state": state,
    })


def _run_measurements(
    args: argparse.Namespace, base_url: str, vectors: Any,
    truth: dict[tuple[str, int], TruthEntry], query_ids: dict[int, int], spec: FilterSpec,
    strategy: str, cutoff: int, ef: int, phase: str,
    requests: Sequence[WorkloadRequest], repeats: int,
    *, schedule_rotation: int, schedule_index: int, target: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timed = phase in {"calibration", "final"}
    request_by_query = {request.query_no: request for request in requests}
    if len(request_by_query) != len(requests):
        raise ValueError("measurement block contains duplicate query_no values")
    for query_no, repeat in baseline.measurement_schedule(
        tuple(request_by_query), repeats, schedule_rotation
    ):
        request = request_by_query[query_no]
        query_id = query_ids[query_no]
        if request.query_id != query_id or request.filter_name != spec.name:
            raise ValueError("measurement request disagrees with query/filter contract")
        result = query_once_minimal(
            base_url,
            vectors[query_id],
            spec.where,
            query_id,
            K,
            timeout=args.timeout,
            retries=0 if timed else args.retries,
        )
        try:
            result_distances_sq = (
                baseline.exact_squared_l2(vectors, query_id, result.ids)
                if result.ids
                else ()
            )
        except Exception as exc:
            result = QueryResult(
                result.ids,
                result.latency_ms,
                result.retry_count,
                result.order_error,
                f"{exc.__class__.__name__}: {exc}",
                result.returned_count,
                result.request_limit,
                result.filter_membership_valid,
            )
            result_distances_sq = ()
        row = baseline.measurement_row(
            phase=phase,
            strategy=strategy,
            spec=spec,
            ef=ef,
            query_no=query_no,
            query_id=query_id,
            repeat=repeat,
            result=result,
            truth=truth[(spec.name, query_no)],
            result_distances_sq=result_distances_sq,
        )
        row.update({
            "request_no": request.request_no,
            "target_recall": _target_token(target),
            "schedule_index": schedule_index,
            "schedule_rotation": schedule_rotation,
            "flat_search_cutoff": int(cutoff),
            "timed_response_fields": ",".join(TIMED_RESPONSE_FIELDS),
            "filter_membership_evidence": (
                "server_where_clause_plus_import_manifest_filter_counts_and_exact_gt"
            ),
            **expected_route(spec, cutoff, strategy),
        })
        rows.append(row)
    return rows


def minimal_graphql_query(
    class_name: str, vector: Sequence[float], where: dict[str, Any], limit: int,
) -> str:
    vector_text = baseline.json_to_graphql([float(value) for value in vector])
    return (
        "{ Get { "
        f"{class_name}(nearVector:{{vector:{vector_text}}} "
        f"where:{baseline.json_to_graphql(where)} limit:{int(limit)}) "
        "{ row_id _additional { distance } } } }"
    )


def query_once_minimal(
    base_url: str,
    vector: Sequence[float],
    where: dict[str, Any],
    query_id: int,
    k: int = K,
    *,
    timeout: float = 60.0,
    retries: int = 0,
) -> QueryResult:
    started = time.perf_counter()
    try:
        if k != K:
            raise ValueError(f"formal runner requires k={K}, got {k}")
        request_limit = k + 1
        data, retry_count = baseline.graphql(
            base_url,
            minimal_graphql_query(CLASS_NAME, vector, where, request_limit),
            timeout=timeout,
            retries=retries,
        )
        if data.get("errors"):
            raise RuntimeError(json.dumps(data["errors"], sort_keys=True))
        objects = data["data"]["Get"][CLASS_NAME]
        if any(set(obj) != {"row_id", "_additional"} for obj in objects):
            raise RuntimeError("timed GraphQL response contains undeclared scalar properties")
        raw_ids = tuple(int(obj["row_id"]) for obj in objects)
        service_distances = [
            float(obj["_additional"]["distance"]) for obj in objects
        ]
        if len(raw_ids) > request_limit:
            raise RuntimeError(
                f"oversized result: limit={request_limit} actual={len(raw_ids)}"
            )
        if len(set(raw_ids)) != len(raw_ids):
            raise RuntimeError("duplicate row_id in result")
        order_error = ""
        if any(not baseline.math.isfinite(distance) for distance in service_distances):
            order_error = "non-finite distance"
        elif any(
            left > right + 1e-6
            for left, right in zip(service_distances, service_distances[1:])
        ):
            order_error = "results are not ordered by ascending distance"
        ids = tuple(row_id for row_id in raw_ids if row_id != query_id)[:k]
        return QueryResult(
            ids,
            (time.perf_counter() - started) * 1000.0,
            retry_count,
            order_error,
            "",
            returned_count=len(raw_ids),
            request_limit=request_limit,
            filter_membership_valid=True,
        )
    except Exception as exc:
        return QueryResult(
            (),
            (time.perf_counter() - started) * 1000.0,
            0,
            "",
            f"{exc.__class__.__name__}: {exc}",
            filter_membership_valid=False,
        )


def summarize_configuration(
    rows: Sequence[dict[str, Any]], *, strategy: str, filter_name: str, cutoff: int, ef: int,
    query_nos: Sequence[int], repeats: int, bootstrap_seed: int, phase: str = "calibration",
    target: float | None = None,
) -> dict[str, Any]:
    summary = baseline.summarize_configuration(
        rows, strategy=strategy, filter_name=filter_name, ef=ef, query_nos=query_nos,
        repeats=repeats, bootstrap_seed=bootstrap_seed, phase=phase,
    )
    summary["flat_search_cutoff"] = int(cutoff)
    summary["target_recall"] = _target_token(target)
    summary["single_client_service_qps"] = NA
    summary["service_qps_definition"] = QPS_DEFINITION
    summary["qps_measured"] = False
    return summary


def _summary_row_for_target(
    selected: dict[str, Any], target: float, final: dict[str, Any],
) -> dict[str, Any]:
    result = dict(final)
    target_met = bool(baseline.reaches_target(final, target))
    result.update({
        "phase": "final", "target_recall": target, "selected_ef": selected["ef"],
        "selected_flat_search_cutoff": selected["flat_search_cutoff"], "target_status": "selected",
        "target_met": target_met,
        "target_outcome": "selected_and_confirmed" if target_met else "selected_but_final_unconfirmed",
        "comparison_status": "confirmed" if target_met else "unconfirmed",
        "calibration_recall_mean": selected.get("recall_mean", NA),
        "calibration_recall_lcb95": selected.get("recall_lcb95", NA),
        "calibration_complete": selected["complete"],
        "calibration_target_recall": selected.get("calibration_target_recall", target),
        "calibration_selection_policy": selected.get(
            "calibration_selection_policy", CALIBRATION_SELECTION_POLICY
        ),
        "selection_mode": selected.get("selection_mode", "legacy_calibration_selection"),
        "calibration_lcb95_qualified": selected.get("calibration_lcb95_qualified", False),
        "fallback_used": selected.get("fallback_used", False),
        "target_selection_rule": CALIBRATION_SELECTION_RULE,
    })
    return result


def artifact_gate_errors(
    selected_groups: dict[tuple[str, str, int, int], list[float]],
    final_summaries: Sequence[dict[str, Any]], raw_rows: Sequence[dict[str, Any]],
    final_results: Sequence[dict[str, Any]], filters: Sequence[FilterSpec],
    flat_search_cutoffs: Sequence[int], ef_values: Sequence[int],
    calibration_summaries: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    errors: list[str] = []
    expected = {(strategy, name, target) for (strategy, name, _, _), targets in selected_groups.items() for target in targets}
    found: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in final_summaries:
        found.setdefault((str(row.get("configured_filter_strategy")), str(row.get("filter_name")), float(row.get("target_recall"))), []).append(row)
    for key in expected:
        rows = found.get(key, [])
        if len(rows) != 1:
            errors.append(f"final summary cardinality mismatch for {key}: {len(rows)}")
        elif rows[0].get("target_outcome") not in {
                "selected_and_confirmed", "selected_but_final_unconfirmed"}:
            errors.append(f"selected target outcome is invalid for {key}")
    if set(found) != expected:
        errors.append("final summaries do not match selected system target grid")
    final_by_config = {
        (
            str(row.get("configured_filter_strategy")), str(row.get("filter_name")),
            int(row.get("flat_search_cutoff", -1)), int(row.get("ef", -1)),
        ): row
        for row in final_results
    }
    if set(final_by_config) != set(selected_groups) or len(final_by_config) != len(final_results):
        errors.append("final result blocks do not match selected system configurations and flat controls")
    for spec in filters:
        flat = flat_configuration(spec, flat_search_cutoffs, ef_values)
        key = (
            str(flat["configured_filter_strategy"]), spec.name,
            int(flat["flat_search_cutoff"]), int(flat["ef"]),
        )
        result = final_by_config.get(key)
        if result is None or result.get("complete") is not True:
            errors.append(f"flat held-out exactness control is incomplete: filter={spec.name}")
            continue
        mean = result.get("recall_mean")
        if not baseline._finite_number(mean) or float(mean) != 1.0:
            errors.append(
                f"flat held-out exactness gate failed: filter={spec.name} "
                f"recall_mean={result.get('recall_mean')} recall_lcb95_reported={result.get('recall_lcb95')}"
            )
    for row in raw_rows:
        if row.get("phase") in {"calibration", "final"}:
            if int(row.get("retry_count", -1)) != 0:
                errors.append("timed measurement retried")
            if row.get("valid") is not True or row.get("error") or row.get("order_error"):
                errors.append("invalid timed measurement")
    if calibration_summaries is not None:
        errors.extend(baseline.measurement_block_integrity_errors(
            raw_rows, calibration_summaries, phase="calibration",
            query_nos=CALIBRATION_QUERY_NOS, repeats=CALIBRATION_REPEATS,
            block_fields=("configured_filter_strategy", "filter_name", "flat_search_cutoff", "ef"),
        ))
        errors.extend(baseline.measurement_block_integrity_errors(
            raw_rows, final_results, phase="final",
            query_nos=FINAL_QUERY_NOS, repeats=FINAL_REPEATS,
            block_fields=("configured_filter_strategy", "filter_name", "flat_search_cutoff", "ef"),
        ))
        specs_by_name = {spec.name: spec for spec in filters}
        for row in final_summaries:
            filter_name = str(row.get("filter_name"))
            spec = specs_by_name.get(filter_name)
            if spec is None:
                errors.append(f"selected target has unknown filter: {filter_name}")
                continue
            target = float(row.get("target_recall"))
            candidates = [
                candidate for candidate in calibration_summaries
                if candidate.get("filter_name") == filter_name
            ]
            expected_selected = select_conservative_config(
                candidates, target, spec, flat_search_cutoffs, ef_values,
            )
            if expected_selected is None:
                errors.append(
                    f"selected target has no reproducible calibration selection: "
                    f"filter={filter_name} target={target}"
                )
                continue
            expected_identity = (
                str(expected_selected["configured_filter_strategy"]),
                int(expected_selected["flat_search_cutoff"]),
                int(expected_selected["ef"]),
            )
            observed_identity = (
                str(row.get("configured_filter_strategy")),
                int(row.get("selected_flat_search_cutoff", -1)),
                int(row.get("selected_ef", -1)),
            )
            if observed_identity != expected_identity:
                errors.append(
                    f"selected target does not match recomputed LCB protocol: "
                    f"filter={filter_name} target={target}"
                )
            for field in (
                "selection_mode", "calibration_selection_policy",
                "calibration_lcb95_qualified", "fallback_used",
            ):
                if row.get(field) != expected_selected.get(field):
                    errors.append(
                        f"selected target {field} does not match recomputed LCB protocol: "
                        f"filter={filter_name} target={target}"
                    )
    return errors


def _formal_block_integrity_errors(
    raw_rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    *,
    phase: str,
    requests_by_filter: dict[str, Sequence[WorkloadRequest]],
    repeats: int,
    include_target: bool,
) -> list[str]:
    errors: list[str] = []
    seen_blocks: set[tuple[Any, ...]] = set()
    for summary in summaries:
        target = str(summary.get("target_recall", NA))
        key = (
            str(summary.get("configured_filter_strategy")),
            str(summary.get("filter_name")),
            int(summary.get("flat_search_cutoff", -1)),
            int(summary.get("ef", -1)),
            target if include_target else NA,
        )
        if key in seen_blocks:
            errors.append(f"duplicate {phase} summary block: {key}")
            continue
        seen_blocks.add(key)
        requests = requests_by_filter.get(key[1], ())
        expected = {
            (request.request_no, request.query_no, repeat)
            for request in requests
            for repeat in range(repeats)
        }
        rows = [
            row for row in raw_rows
            if row.get("phase") == phase
            and str(row.get("configured_filter_strategy")) == key[0]
            and str(row.get("filter_name")) == key[1]
            and int(row.get("flat_search_cutoff", -1)) == key[2]
            and int(row.get("ef", -1)) == key[3]
            and (
                not include_target
                or str(row.get("target_recall", NA)) == key[4]
            )
        ]
        observed: set[tuple[int, int, int]] = set()
        for row in rows:
            try:
                pair = (
                    int(row["request_no"]),
                    int(row["query_no"]),
                    int(row["repeat"]),
                )
            except (KeyError, TypeError, ValueError):
                errors.append(f"malformed {phase} raw row in block {key}")
                continue
            if pair in observed:
                errors.append(f"duplicate {phase} request/query/repeat in block {key}: {pair}")
            observed.add(pair)
            if pair not in expected:
                errors.append(f"unexpected {phase} request/query/repeat in block {key}: {pair}")
            if (
                row.get("valid") is not True
                or row.get("error")
                or row.get("order_error")
                or int(row.get("retry_count", -1)) != 0
            ):
                errors.append(f"invalid {phase} raw row in block {key}: {pair}")
        if observed != expected or len(rows) != len(expected):
            errors.append(
                f"{phase} block coverage mismatch for {key}: "
                f"expected={len(expected)} observed={len(rows)} unique={len(observed)}"
            )
        if summary.get("complete") is not True:
            errors.append(f"incomplete {phase} summary block: {key}")
    return errors


def formal_artifact_gate_errors(
    selected_runs: dict[tuple[str, str, int, int, float | None], list[float]],
    final_summaries: Sequence[dict[str, Any]],
    raw_rows: Sequence[dict[str, Any]],
    final_results: Sequence[dict[str, Any]],
    filters: Sequence[FilterSpec],
    flat_search_cutoffs: Sequence[int],
    ef_values: Sequence[int],
    calibration_summaries: Sequence[dict[str, Any]],
    calibration_requests_by_filter: dict[str, Sequence[WorkloadRequest]],
    final_requests_by_filter: dict[str, Sequence[WorkloadRequest]],
    calibration_repeats: int,
    final_repeats: int,
) -> list[str]:
    errors = _formal_block_integrity_errors(
        raw_rows,
        calibration_summaries,
        phase="calibration",
        requests_by_filter=calibration_requests_by_filter,
        repeats=calibration_repeats,
        include_target=False,
    )
    errors.extend(_formal_block_integrity_errors(
        raw_rows,
        final_results,
        phase="final",
        requests_by_filter=final_requests_by_filter,
        repeats=final_repeats,
        include_target=True,
    ))
    expected_target_pairs = {
        (name, float(target))
        for (_, name, _, _, target), targets in selected_runs.items()
        if target is not None
        for target in targets
    }
    observed_target_pairs = {
        (str(row.get("filter_name")), float(row.get("target_recall")))
        for row in final_summaries
    }
    if observed_target_pairs != expected_target_pairs or len(final_summaries) != len(
        expected_target_pairs
    ):
        errors.append("formal final summaries do not cover every selected filter-target pair")
    observed_runs = {
        (
            str(row.get("configured_filter_strategy")),
            str(row.get("filter_name")),
            int(row.get("flat_search_cutoff", -1)),
            int(row.get("ef", -1)),
            None if str(row.get("target_recall", NA)) == NA
            else float(row["target_recall"]),
        )
        for row in final_results
    }
    if observed_runs != set(selected_runs) or len(observed_runs) != len(final_results):
        errors.append("formal final result blocks do not match selected target/control runs")
    specs_by_name = {spec.name: spec for spec in filters}
    for spec in filters:
        flat = flat_configuration(spec, flat_search_cutoffs, ef_values)
        key = (
            str(flat["configured_filter_strategy"]),
            spec.name,
            int(flat["flat_search_cutoff"]),
            int(flat["ef"]),
            None,
        )
        matches = [
            row for row in final_results
            if (
                str(row.get("configured_filter_strategy")),
                str(row.get("filter_name")),
                int(row.get("flat_search_cutoff", -1)),
                int(row.get("ef", -1)),
                None if str(row.get("target_recall", NA)) == NA
                else float(row["target_recall"]),
            ) == key
        ]
        if (
            len(matches) != 1
            or matches[0].get("complete") is not True
            or float(matches[0].get("recall_mean", -1)) != 1.0
        ):
            errors.append(f"formal flat held-out exactness gate failed: filter={spec.name}")
    for row in final_summaries:
        filter_name = str(row.get("filter_name"))
        target = float(row.get("target_recall"))
        spec = specs_by_name.get(filter_name)
        candidates = [
            candidate for candidate in calibration_summaries
            if candidate.get("filter_name") == filter_name
        ]
        expected = (
            select_conservative_config(
                candidates, target, spec, flat_search_cutoffs, ef_values,
            )
            if spec is not None else None
        )
        if expected is None:
            errors.append(
                f"formal target has no reproducible calibration selection: "
                f"filter={filter_name} target={target}"
            )
            continue
        expected_identity = (
            str(expected["configured_filter_strategy"]),
            int(expected["flat_search_cutoff"]),
            int(expected["ef"]),
        )
        observed_identity = (
            str(row.get("configured_filter_strategy")),
            int(row.get("selected_flat_search_cutoff", -1)),
            int(row.get("selected_ef", -1)),
        )
        if observed_identity != expected_identity:
            errors.append(
                f"formal target does not match recomputed LCB95 selection: "
                f"filter={filter_name} target={target}"
            )
    for target in sorted({
        float(target)
        for _, _, _, _, target in selected_runs
        if target is not None
    }):
        expected = {
            (request.request_no, repeat)
            for requests in final_requests_by_filter.values()
            for request in requests
            for repeat in range(final_repeats)
        }
        rows = [
            row for row in raw_rows
            if row.get("phase") == "final"
            and str(row.get("target_recall", NA)) == _target_token(target)
        ]
        observed = {
            (int(row["request_no"]), int(row["repeat"]))
            for row in rows
        }
        if observed != expected or len(rows) != len(expected):
            errors.append(
                f"formal mixed-trace target coverage mismatch: target={target} "
                f"expected={len(expected)} observed={len(rows)} unique={len(observed)}"
            )
    return errors


def publication_gate_errors(
    artifact_errors: Sequence[str], final_summaries: Sequence[dict[str, Any]],
) -> list[str]:
    errors = list(artifact_errors)
    errors.extend(
        "selected target is not confirmed on held-out final data: "
        f"filter={row.get('filter_name')} target={row.get('target_recall')}"
        for row in final_summaries
        if row.get("target_outcome") == "selected_but_final_unconfirmed"
    )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=(FORMAL_EXECUTION_PROTOCOL, LEGACY_EXECUTION_PROTOCOL),
        default=FORMAL_EXECUTION_PROTOCOL,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--filters-csv", type=Path, default=baseline.DEFAULT_FILTERS)
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument(
        "--truth-manifest", type=Path, default=DEFAULT_TRUTH_MANIFEST,
        help="current exact-GT manifest binding filter, query cohort, truth, and fbin hashes",
    )
    parser.add_argument(
        "--import-manifest", type=Path, default=DEFAULT_IMPORT_MANIFEST,
        help="complete Weaviate corpus import manifest",
    )
    parser.add_argument("--fbin", type=Path, default=baseline.DEFAULT_FBIN)
    parser.add_argument(
        "--calibration-workload",
        type=Path,
        default=DEFAULT_CALIBRATION_WORKLOAD,
        help="frozen q200 mixed calibration trace for the current formal protocol",
    )
    parser.add_argument(
        "--measurement-workload",
        type=Path,
        default=DEFAULT_MEASUREMENT_WORKLOAD,
        help="frozen q10k mixed measurement trace for the current formal protocol",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ef-values", type=int, nargs="+", default=list(DEFAULT_EF_VALUES))
    parser.add_argument("--flat-search-cutoffs", type=int, nargs="+", default=list(DEFAULT_FLAT_SEARCH_CUTOFFS))
    parser.add_argument("--filter-names", nargs="+")
    parser.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument(
        "--calibration-lcb-margin", type=float, default=DEFAULT_CALIBRATION_LCB_MARGIN,
        help="deprecated compatibility option; formal LCB95 threshold is exactly the target",
    )
    parser.add_argument("--hnsw-dominance-guard", type=float, default=1.05)
    parser.add_argument("--expected-service-version", default="1.38.0")
    parser.add_argument("--service-image-digest", default="")
    parser.add_argument("--k", type=int, choices=(K,), default=K)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--warmup-queries", type=int, default=1)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--resume", action="store_true", help="strictly resume a matching atomic complete-block checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _dry_run(args: argparse.Namespace) -> int:
    dry_filters: Sequence[Any] = args.filter_names or baseline.FILTERS
    calibration_queries = (
        FORMAL_CALIBRATION_REQUESTS
        if formal_protocol(args)
        else len(CALIBRATION_QUERY_NOS) * len(dry_filters)
    )
    calibration_repeats, final_repeats = protocol_repeats(args)
    calibration_query_nos, final_query_nos = protocol_query_nos(args)
    query_budget = calibration_query_budget(
        dry_filters,
        args.ef_values,
        args.warmup_queries,
        calibration_requests=calibration_queries,
        calibration_repeats=calibration_repeats,
    )
    print(json.dumps({"dry_run": True, "network": False, "files_read": False, "files_written": False,
                      "execution_protocol": args.protocol,
                      "configured_filter_strategies": list(DEFAULT_FILTER_STRATEGIES), "ef_values": args.ef_values,
                      "flat_search_cutoffs": args.flat_search_cutoffs, "targets": args.targets,
                      "filter_names": args.filter_names or "all",
                      "maximum_effective_calibration_timed_queries": query_budget["maximum_timed_queries_before_hnsw_early_stop"],
                      "hnsw_dominance_guard": args.hnsw_dominance_guard,
                      "expected_service_version": args.expected_service_version,
                      "service_image_digest": args.service_image_digest or NA,
                      "truth_manifest": str(args.truth_manifest),
                      "import_manifest": str(args.import_manifest),
                      "calibration_workload": str(args.calibration_workload),
                      "measurement_workload": str(args.measurement_workload),
                      "current_protocol": current_protocol(args),
                      "timed_response_fields": list(TIMED_RESPONSE_FIELDS),
                      "qps_measured": False,
                      "target_selection_rule": CALIBRATION_SELECTION_RULE,
                      "calibration_selection_policy": CALIBRATION_SELECTION_POLICY,
                      "calibration_stop_metric": CALIBRATION_STOP_METRIC,
                      "calibration_queries": len(calibration_query_nos),
                      "calibration_repeats": calibration_repeats,
                      "calibration_query_nos": [
                          calibration_query_nos[0], calibration_query_nos[-1]
                      ],
                      "final_queries": len(final_query_nos),
                      "final_repeats": final_repeats,
                      "final_query_nos": [final_query_nos[0], final_query_nos[-1]],
                      "formal_filter_target_cells": (
                          len(dry_filters) * len(args.targets)
                          if formal_protocol(args) else NA
                      ),
                      "formal_full_release_samples_per_target": (
                          FORMAL_FINAL_REQUESTS * FORMAL_FINAL_REPEATS
                          if formal_protocol(args) else NA
                      )}, sort_keys=True))
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.k != K:
        raise ValueError(f"formal runner requires k={K}")
    if sorted(set(args.targets)) != list(args.targets) or any(value <= 0 or value > 1 for value in args.targets):
        raise ValueError("targets must be sorted, unique, and in (0, 1]")
    if sorted(set(args.ef_values)) != list(args.ef_values) or any(value <= 0 for value in args.ef_values):
        raise ValueError("ef values must be sorted, unique, and positive")
    if sorted(set(args.flat_search_cutoffs)) != list(args.flat_search_cutoffs) or any(value < 0 for value in args.flat_search_cutoffs):
        raise ValueError("flat search cutoffs must be sorted, unique, and non-negative")
    if not args.flat_search_cutoffs or args.flat_search_cutoffs[0] != 0:
        raise ValueError("flat search cutoff grid must include 0")
    if not args.expected_service_version:
        raise ValueError("expected service version must be non-empty")
    if (not baseline._finite_number(args.calibration_lcb_margin)
            or float(args.calibration_lcb_margin) != 0.0):
        raise ValueError(
            "formal calibration LCB95 threshold is exactly the target; "
            "--calibration-lcb-margin must be 0"
        )
    if (not baseline._finite_number(args.hnsw_dominance_guard)
            or float(args.hnsw_dominance_guard) < 1.0):
        raise ValueError("hnsw dominance guard must be finite and >= 1.0")
    if not args.service_image_digest.strip():
        raise ValueError("--service-image-digest is required for formal execution")
    if not args.truth_manifest or not args.import_manifest:
        raise ValueError("--truth-manifest and --import-manifest are required")
    calibration_query_nos, _ = protocol_query_nos(args)
    if (
        args.retries < 0
        or args.warmup_queries < 0
        or args.warmup_queries > len(calibration_query_nos)
    ):
        raise ValueError("invalid setup retries or warmup_queries")
    if formal_protocol(args) and (
        not args.calibration_workload or not args.measurement_workload
    ):
        raise ValueError("formal protocol requires frozen calibration and measurement workloads")


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    base_url = f"http://{args.host}:{args.port}"
    outputs = baseline.sibling_outputs(args.out)
    checkpoint = args.checkpoint or baseline.checkpoint_path(args.out)
    if not args.resume and checkpoint.exists():
        raise RuntimeError(f"checkpoint already exists: {checkpoint}; use --resume or move it aside")
    all_filters = baseline.load_filter_specs(args.filters_csv)
    filters = select_filter_specs(all_filters, args.filter_names)
    for spec in filters:
        effective_cutoffs(spec, args.flat_search_cutoffs)
    vectors, vector_rows, dimensions = baseline.read_fbin_memmap(args.fbin)
    calibration_query_nos, final_query_nos = protocol_query_nos(args)
    calibration_repeats, final_repeats = protocol_repeats(args)
    truth, query_ids = baseline.load_truth(
        args.truth_csv,
        filters,
        calibration_query_nos=calibration_query_nos,
        final_query_nos=final_query_nos,
        k=K,
    )
    if any(query_id < 0 or query_id >= vector_rows for query_id in query_ids.values()):
        raise ValueError("truth query_id is outside the fbin row range")
    if formal_protocol(args):
        all_filter_names = {spec.name for spec in all_filters}
        all_truth, all_query_ids = baseline.load_truth(
            args.truth_csv,
            all_filters,
            calibration_query_nos=calibration_query_nos,
            final_query_nos=final_query_nos,
            k=K,
        )
        del all_truth
        calibration_workload = load_frozen_workload(
            args.calibration_workload,
            split="calibration",
            expected_requests=FORMAL_CALIBRATION_REQUESTS,
            expected_query_nos=calibration_query_nos,
            query_ids=all_query_ids,
            filter_names=all_filter_names,
        )
        final_workload = load_frozen_workload(
            args.measurement_workload,
            split="measurement",
            expected_requests=FORMAL_FINAL_REQUESTS,
            expected_query_nos=final_query_nos,
            query_ids=all_query_ids,
            filter_names=all_filter_names,
        )
    else:
        calibration_workload = legacy_requests(
            calibration_query_nos, query_ids, filters[0].name
        )
        final_workload = legacy_requests(final_query_nos, query_ids, filters[0].name)
    source_file_hashes = {
        "runner": baseline.sha256_file(Path(__file__)),
        "baseline_runner": baseline.sha256_file(Path(baseline.__file__)),
        "filters_csv": baseline.sha256_file(args.filters_csv), "truth_csv": baseline.sha256_file(args.truth_csv),
        "fbin": baseline.sha256_file(args.fbin),
    }
    if formal_protocol(args):
        source_file_hashes.update({
            "calibration_workload_csv": baseline.sha256_file(args.calibration_workload),
            "measurement_workload_csv": baseline.sha256_file(args.measurement_workload),
        })
    input_provenance = validate_current_input_manifests(
        args, filters, source_file_hashes
    )
    if formal_protocol(args):
        input_provenance.update({
            "calibration_workload_csv": {
                "path": str(args.calibration_workload.resolve()),
                "sha256": source_file_hashes["calibration_workload_csv"],
                "requests": FORMAL_CALIBRATION_REQUESTS,
                "repeats": calibration_repeats,
            },
            "measurement_workload_csv": {
                "path": str(args.measurement_workload.resolve()),
                "sha256": source_file_hashes["measurement_workload_csv"],
                "requests": FORMAL_FINAL_REQUESTS,
                "repeats": final_repeats,
            },
        })
    source_hashes = {
        **source_file_hashes,
        "truth_manifest": input_provenance["truth_manifest"]["sha256"],
        "import_manifest": input_provenance["import_manifest"]["sha256"],
        "query_cohort_csv": input_provenance["query_cohort_csv"]["sha256"],
        "query_cohort_manifest": input_provenance["query_cohort_manifest"]["sha256"],
        "import_attributes_csv": input_provenance["import_attributes_csv"]["sha256"],
    }
    specification = run_specification(
        args,
        filters,
        query_ids,
        source_hashes,
        input_provenance,
        calibration_workload,
        final_workload,
    )
    calibration_requests_by_filter = {
        spec.name: (
            requests_for_filter(calibration_workload, spec.name)
            if formal_protocol(args)
            else legacy_requests(calibration_query_nos, query_ids, spec.name)
        )
        for spec in filters
    }
    final_requests_by_filter = {
        spec.name: (
            requests_for_filter(final_workload, spec.name)
            if formal_protocol(args)
            else legacy_requests(final_query_nos, query_ids, spec.name)
        )
        for spec in filters
    }
    quarantined = None if args.resume else baseline.isolate_existing_outputs(outputs)
    raw_rows: list[dict[str, Any]] = []
    calibration_summaries: list[dict[str, Any]] = []
    final_results: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "completed_blocks": [], "schema_records": [], "schema_timings": [], "node_records": []
    }
    if args.resume:
        payload = load_checkpoint(checkpoint, specification, query_ids)
        raw_rows, calibration_summaries, final_results, state = (
            payload["raw_rows"], payload["calibration_summaries"], payload["final_results"], payload["state"]
        )
        validate_checkpoint_state(state)
        validate_monotone_calibration_state(
            calibration_summaries, filters, args.flat_search_cutoffs, args.ef_values,
            args.targets[-1], args.hnsw_dominance_guard, args.calibration_lcb_margin,
        )

    def save_checkpoint() -> None:
        write_checkpoint(checkpoint, specification, raw_rows=raw_rows, calibration_summaries=calibration_summaries,
                         final_results=final_results, state=state)

    original_schema: dict[str, Any] | None = state.get("original_schema") if args.resume else None
    # A resumed service may still carry a crashed run's schema, so restore it even
    # when interruption happens while reading the current live state.
    schema_restore_required = bool(args.resume)
    primary_error: BaseException | None = None
    try:
        live_schema, _ = baseline.request_json(
            base_url, f"/v1/schema/{CLASS_NAME}", timeout=args.timeout, retries=args.retries
        )
        verify_schema(live_schema)
        schema_config_identity(live_schema)
        if args.resume:
            state["schema_records"].append({"phase": "resume_live_schema", "schema": live_schema})
        else:
            original_schema = copy.deepcopy(live_schema)
            state["original_schema"] = original_schema
            state["original_schema_sha256"] = schema_snapshot_hash(original_schema)
            state["schema_records"].append({"phase": "original_schema_snapshot", "schema": original_schema})
        # Persist the immutable restore target and current live state before any schema PUT.
        save_checkpoint()
        completed = {
            _block_key(item["phase"], item["configured_filter_strategy"], item["filter_name"],
                       item["flat_search_cutoff"], item["ef"],
                       None if item.get("target_recall", NA) == NA
                       else float(item["target_recall"]))
            for item in state["completed_blocks"]
        }
        service_meta: dict[str, Any] = {}
        total_count = 0
        filter_counts: dict[str, int] = {}
        scheduled_cutoffs = effective_cutoff_grid(filters, args.flat_search_cutoffs)
        schedule = calibration_configuration_schedule(
            DEFAULT_FILTER_STRATEGIES, scheduled_cutoffs, args.ef_values
        )
        selections: dict[tuple[str, float], dict[str, Any] | None] = {}
        statuses: dict[tuple[str, float], str] = {}
        service_meta, _ = baseline.request_json(base_url, "/v1/meta", timeout=args.timeout, retries=args.retries)
        service_identity = validate_service_identity(
            service_meta, args.expected_service_version, args.service_image_digest
        )
        nodes, retries = baseline.get_ready_nodes(base_url, args.timeout, args.retries)
        state["node_records"].append({"phase": "initial", "retries": retries, "nodes": nodes})
        count_data, _ = baseline.graphql(base_url, f"{{ Aggregate {{ {CLASS_NAME} {{ meta {{ count }} }} }} }}", timeout=args.timeout, retries=args.retries)
        total_count = int(count_data["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"])
        if total_count != baseline.EXPECTED_ROWS:
            raise RuntimeError(f"Weaviate count mismatch: expected={baseline.EXPECTED_ROWS} actual={total_count}")
        for spec in filters:
            data, _ = baseline.graphql(base_url, baseline._count_query(spec), timeout=args.timeout, retries=args.retries)
            filter_counts[spec.name] = int(data["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"])
            if filter_counts[spec.name] != spec.expected_rows:
                raise RuntimeError(f"filter count mismatch for {spec.name}: expected={spec.expected_rows} actual={filter_counts[spec.name]}")

        for schedule_index, strategy, cutoff, ef in schedule:
            pending = [
                spec for spec in baseline.rotated(filters, schedule_index)
                if cutoff in effective_cutoffs(spec, args.flat_search_cutoffs)
                and (
                    cutoff != 0
                    or hnsw_route_termination(
                        calibration_summaries, strategy, spec,
                        args.flat_search_cutoffs, args.ef_values, args.targets[-1],
                        args.hnsw_dominance_guard, args.calibration_lcb_margin,
                    )["termination_reason"] == "in_progress"
                )
                and _block_key("calibration", strategy, spec.name, cutoff, ef) not in completed
            ]
            if not pending:
                continue
            schema_restore_required = True
            schema, update_ms, update_retries = put_hnsw_config(base_url, strategy, ef, cutoff, args.timeout, args.retries)
            state["schema_records"].append({"phase": "calibration", "schedule_index": schedule_index, "configured_filter_strategy": strategy, "ef": ef, "flat_search_cutoff": cutoff, "schema": schema})
            state["schema_timings"].append({"phase": "calibration", "schedule_index": schedule_index, "configured_filter_strategy": strategy, "ef": ef, "flat_search_cutoff": cutoff, "schema_update_ms": update_ms, "schema_retries": update_retries})
            nodes, retries = baseline.get_ready_nodes(base_url, args.timeout, args.retries)
            state["node_records"].append({"phase": "calibration", "schedule_index": schedule_index, "retries": retries, "nodes": nodes})
            for position, spec in enumerate(pending):
                block_requests = calibration_requests_by_filter[spec.name]
                if args.warmup_queries:
                    raw_rows.extend(_run_measurements(
                        args, base_url, vectors, truth, query_ids, spec, strategy,
                        cutoff, ef, "warmup", block_requests[:args.warmup_queries], 1,
                        schedule_rotation=schedule_index, schedule_index=schedule_index,
                    ))
                measured = _run_measurements(
                    args, base_url, vectors, truth, query_ids, spec, strategy,
                    cutoff, ef, "calibration", block_requests, calibration_repeats,
                    schedule_rotation=schedule_index + position,
                    schedule_index=schedule_index,
                )
                raw_rows.extend(measured)
                summary = summarize_configuration(
                    measured, strategy=strategy, filter_name=spec.name, cutoff=cutoff,
                    ef=ef, query_nos=[request.query_no for request in block_requests],
                    repeats=calibration_repeats,
                    bootstrap_seed=args.bootstrap_seed + schedule_index * len(filters) + position,
                )
                summary.update({"schedule_index": schedule_index, "target_recall": NA,
                                "target_lcb95_met": {
                                    str(target): reaches_calibration_lcb(summary, target)
                                    for target in args.targets
                                },
                                **expected_route(spec, cutoff, strategy)})
                calibration_summaries.append(summary)
                state["completed_blocks"].append(_block_record(
                    "calibration", strategy, spec.name, cutoff, ef,
                    block_requests, calibration_repeats,
                ))
                completed.add(_block_key("calibration", strategy, spec.name, cutoff, ef))
                save_checkpoint()

        for spec in filters:
            candidates = [
                row for row in calibration_summaries if row.get("filter_name") == spec.name
            ]
            for target in args.targets:
                key = (spec.name, float(target))
                selections[key] = select_conservative_config(
                    candidates, target, spec, args.flat_search_cutoffs, args.ef_values,
                    args.calibration_lcb_margin,
                )
                statuses[key] = calibration_target_status(
                    calibration_summaries, spec, target,
                    args.flat_search_cutoffs, args.ef_values, args.targets,
                    args.hnsw_dominance_guard, args.calibration_lcb_margin,
                )
        if formal_protocol(args):
            selected_runs: dict[
                tuple[str, str, int, int, float | None], list[float]
            ] = {
                (
                    str(selected["configured_filter_strategy"]),
                    filter_name,
                    int(selected["flat_search_cutoff"]),
                    int(selected["ef"]),
                    float(target),
                ): [float(target)]
                for (filter_name, target), selected in selections.items()
                if selected is not None
            }
            for spec in filters:
                flat = flat_configuration(
                    spec, args.flat_search_cutoffs, args.ef_values
                )
                selected_runs.setdefault(
                    (
                        str(flat["configured_filter_strategy"]),
                        spec.name,
                        int(flat["flat_search_cutoff"]),
                        int(flat["ef"]),
                        None,
                    ),
                    [],
                )
        else:
            legacy_groups = add_flat_exactness_controls(
                group_selected_targets(selections), filters,
                args.flat_search_cutoffs, args.ef_values,
            )
            selected_runs = {
                (strategy, name, cutoff, ef, None): list(targets)
                for (strategy, name, cutoff, ef), targets in legacy_groups.items()
            }
        resumed_final = {
            (
                item["configured_filter_strategy"],
                item["filter_name"],
                int(item["flat_search_cutoff"]),
                int(item["ef"]),
                None if item.get("target_recall", NA) == NA
                else float(item["target_recall"]),
            )
            for item in state["completed_blocks"] if item["phase"] == "final"
        }
        if not resumed_final <= set(selected_runs):
            raise RuntimeError("checkpoint final block is not selected by restored system targets or flat control")
        final_by_run = {
            (
                row["configured_filter_strategy"],
                row["filter_name"],
                int(row["flat_search_cutoff"]),
                int(row["ef"]),
                None if row.get("target_recall", NA) == NA
                else float(row["target_recall"]),
            ): row
            for row in final_results
        }
        specs_by_name = {spec.name: spec for spec in filters}
        for schedule_index, strategy, cutoff, ef in schedule:
            keys = [
                key for key in selected_runs
                if key[0] == strategy and key[2] == cutoff and key[3] == ef
            ]
            pending = [
                key for key in keys
                if _block_key(
                    "final", key[0], key[1], key[2], key[3], key[4]
                ) not in completed
            ]
            if not pending:
                continue
            schema_restore_required = True
            schema, update_ms, update_retries = put_hnsw_config(base_url, strategy, ef, cutoff, args.timeout, args.retries)
            state["schema_records"].append({"phase": "final", "schedule_index": schedule_index, "configured_filter_strategy": strategy, "ef": ef, "flat_search_cutoff": cutoff, "schema": schema})
            state["schema_timings"].append({"phase": "final", "schedule_index": schedule_index, "configured_filter_strategy": strategy, "ef": ef, "flat_search_cutoff": cutoff, "schema_update_ms": update_ms, "schema_retries": update_retries})
            nodes, retries = baseline.get_ready_nodes(base_url, args.timeout, args.retries)
            state["node_records"].append({"phase": "final", "schedule_index": schedule_index, "retries": retries, "nodes": nodes})
            pending = list(baseline.rotated(
                sorted(pending, key=lambda key: (key[1], _target_token(key[4]))),
                schedule_index,
            ))
            for position, key in enumerate(pending):
                spec = specs_by_name[key[1]]
                target = key[4]
                block_requests = final_requests_by_filter[spec.name]
                if args.warmup_queries:
                    final_warmup = _run_measurements(
                        args, base_url, vectors, truth, query_ids, spec, strategy, cutoff, ef,
                        "final_warmup", block_requests[: args.warmup_queries], 1,
                        schedule_rotation=schedule_index + position,
                        schedule_index=schedule_index,
                        target=target,
                    )
                    for row in final_warmup:
                        row["recall_at_10"] = NA
                    raw_rows.extend(final_warmup)
                measured = _run_measurements(
                    args, base_url, vectors, truth, query_ids, spec, strategy,
                    cutoff, ef, "final", block_requests, final_repeats,
                    schedule_rotation=schedule_index + position,
                    schedule_index=schedule_index, target=target,
                )
                reused_targets = selected_runs[key]
                final_config_role = (
                    "target_winner_and_flat_exactness_control"
                    if cutoff != 0 and reused_targets else
                    "flat_exactness_control" if cutoff != 0 else "target_winner"
                )
                for row in measured:
                    row["reused_for_targets"] = ",".join(str(target) for target in reused_targets) or NA
                    row["final_config_role"] = final_config_role
                raw_rows.extend(measured)
                result = summarize_configuration(
                    measured, strategy=strategy, filter_name=spec.name,
                    cutoff=cutoff, ef=ef,
                    query_nos=[request.query_no for request in block_requests],
                    repeats=final_repeats,
                    bootstrap_seed=(
                        args.bootstrap_seed + 100_000
                        + schedule_index * len(filters) + position
                    ),
                    phase="final", target=target,
                )
                result.update(expected_route(spec, cutoff, strategy))
                result["reused_for_targets"] = reused_targets
                result["final_config_role"] = final_config_role
                final_results.append(result)
                final_by_run[key] = result
                state["completed_blocks"].append(_block_record(
                    "final", strategy, spec.name, cutoff, ef,
                    block_requests, final_repeats, target=target,
                ))
                completed.add(_block_key(
                    "final", strategy, spec.name, cutoff, ef, target
                ))
                save_checkpoint()
    except BaseException as exc:
        primary_error = exc
    finally:
        restore_error: BaseException | None = None
        if schema_restore_required:
            try:
                if original_schema is None:
                    raise RuntimeError("original schema unavailable for required restore")
                original_strategy, original_ef, original_cutoff = schema_config_identity(original_schema)
                restored, restore_retries = put_schema_definition(
                    base_url, original_schema, timeout=args.timeout, retries=args.retries,
                    strategy=original_strategy, ef=original_ef,
                    flat_search_cutoff=original_cutoff,
                )
                state["schema_records"].append({"phase": "restore", "schema": restored})
                state["schema_timings"].append({"phase": "restore", "schema_retries": restore_retries})
                nodes, retries = baseline.get_ready_nodes(base_url, args.timeout, args.retries)
                state["node_records"].append({"phase": "restore", "retries": retries, "nodes": nodes})
            except BaseException as exc:
                restore_error = exc
    raise_after_schema_restore(primary_error, restore_error)

    final_summaries = []
    for (filter_name, target), selected in selections.items():
        if selected is None:
            continue
        key = (
            str(selected["configured_filter_strategy"]), filter_name,
            int(selected["flat_search_cutoff"]), int(selected["ef"]),
            float(target) if formal_protocol(args) else None,
        )
        final_summaries.append(_summary_row_for_target(
            selected, target, final_by_run[key]
        ))
    measurement_errors = [
        "invalid timed measurement: "
        f"phase={row.get('phase')} strategy={row.get('configured_filter_strategy')} "
        f"filter={row.get('filter_name')} cutoff={row.get('flat_search_cutoff')} "
        f"ef={row.get('ef')} query_no={row.get('query_no')} error={row.get('error')} "
        f"order_error={row.get('order_error')}"
        for row in raw_rows
        if row.get("phase") in {"calibration", "final"} and row.get("valid") is not True
    ]
    status_errors = [
        f"calibration target grid incomplete: filter={name} target={target}"
        for (name, target), status in statuses.items() if status == "incomplete_grid"
    ]
    if formal_protocol(args):
        gate_errors = formal_artifact_gate_errors(
            selected_runs,
            final_summaries,
            raw_rows,
            final_results,
            filters,
            args.flat_search_cutoffs,
            args.ef_values,
            calibration_summaries,
            calibration_requests_by_filter,
            final_requests_by_filter,
            calibration_repeats,
            final_repeats,
        )
    else:
        selected_groups = {
            (strategy, name, cutoff, ef): targets
            for (strategy, name, cutoff, ef, target), targets in selected_runs.items()
            if target is None
        }
        gate_errors = artifact_gate_errors(
            selected_groups, final_summaries, raw_rows, final_results, filters,
            args.flat_search_cutoffs, args.ef_values, calibration_summaries,
        )
    errors = measurement_errors + status_errors + gate_errors
    revision = baseline.git_revision()
    grid_proofs = {
        spec.name: configuration_grid_proof(
            calibration_summaries, spec, args.flat_search_cutoffs, args.ef_values,
            args.targets, args.hnsw_dominance_guard, args.calibration_lcb_margin,
        )
        for spec in filters
    }
    equivalence_proofs = {
        spec.name: cutoff_equivalence_proof(
            spec, args.flat_search_cutoffs, args.ef_values
        ) for spec in filters
    }
    route_expectations = [
        {"filter_name": spec.name, "ef": int(ef),
         **expected_route(spec, 0, strategy)}
        for spec in filters for strategy in DEFAULT_FILTER_STRATEGIES for ef in args.ef_values
    ] + [
        {**flat_configuration(spec, args.flat_search_cutoffs, args.ef_values),
         **expected_route(
             spec,
             effective_cutoffs(spec, args.flat_search_cutoffs)[1],
             FLAT_STRATEGY_REPRESENTATIVE,
         )}
        for spec in filters
    ]
    query_budget = calibration_query_budget(
        filters,
        args.ef_values,
        args.warmup_queries,
        calibration_requests=(
            len(calibration_workload)
            if formal_protocol(args)
            else len(CALIBRATION_QUERY_NOS) * len(filters)
        ),
        calibration_repeats=calibration_repeats,
    )
    actual_calibration = {
        "completed_blocks": sum(
            block.get("phase") == "calibration" for block in state["completed_blocks"]
        ),
        "timed_queries": sum(row.get("phase") == "calibration" for row in raw_rows),
        "warmup_queries": sum(row.get("phase") == "warmup" for row in raw_rows),
        "final_warmup_queries": sum(row.get("phase") == "final_warmup" for row in raw_rows),
        "termination_reason_counts": {
            reason: sum(
                route["termination"]["termination_reason"] == reason
                for proof in grid_proofs.values()
                for route in proof["hnsw_routes"].values()
            )
            for reason in (
                "highest_target_lcb95_reached", "full_grid_exhausted", "in_progress",
            )
        },
    }
    flat_exactness_records = [
        {
            "filter_name": spec.name,
            "representative": flat_configuration(spec, args.flat_search_cutoffs, args.ef_values),
            "held_out_recall_mean": final_by_run[
                (
                    FLAT_STRATEGY_REPRESENTATIVE, spec.name,
                    effective_cutoffs(spec, args.flat_search_cutoffs)[1], int(args.ef_values[0]),
                    None,
                )
            ].get("recall_mean", NA),
            "held_out_recall_lcb95": final_by_run[
                (
                    FLAT_STRATEGY_REPRESENTATIVE, spec.name,
                    effective_cutoffs(spec, args.flat_search_cutoffs)[1], int(args.ef_values[0]),
                    None,
                )
            ].get("recall_lcb95", NA),
        }
        for spec in filters
    ]
    config = {
        "class": CLASS_NAME, "git_revision": revision, "source_hashes": source_hashes,
        "current_protocol": current_protocol(args),
        "execution_protocol": args.protocol,
        "input_provenance": input_provenance,
        "run_spec_hash": run_spec_hash(specification), "vector_rows": vector_rows, "dimensions": dimensions,
        "k": K, "configured_filter_strategies": list(DEFAULT_FILTER_STRATEGIES),
        "filter_names": [spec.name for spec in filters],
        "ef_values": args.ef_values, "flat_search_cutoffs": args.flat_search_cutoffs, "targets": args.targets,
        "effective_cutoffs_by_filter": {
            spec.name: list(effective_cutoffs(spec, args.flat_search_cutoffs)) for spec in filters
        },
        "effective_calibration_query_budget": query_budget,
        "actual_calibration": actual_calibration,
        "workload_contract": {
            "scope": (
                "frozen_mixed_trace_partitioned_by_filter"
                if formal_protocol(args) else "legacy_cartesian_per_filter"
            ),
            "calibration": {
                "requests_global": len(calibration_workload),
                "requests_in_shard": sum(
                    len(requests)
                    for requests in calibration_requests_by_filter.values()
                ),
                "repeats": calibration_repeats,
                "request_counts_by_filter": {
                    name: len(requests)
                    for name, requests in calibration_requests_by_filter.items()
                },
            },
            "measurement": {
                "requests_global": len(final_workload),
                "requests_in_shard": sum(
                    len(requests) for requests in final_requests_by_filter.values()
                ),
                "repeats": final_repeats,
                "targets": len(args.targets),
                "request_counts_by_filter": {
                    name: len(requests)
                    for name, requests in final_requests_by_filter.items()
                },
                "filter_target_cells_in_shard": len(filters) * len(args.targets),
                "filter_target_cells_full_release": (
                    len(baseline.FILTERS) * len(args.targets)
                ),
                "samples_per_target_full_release": (
                    len(final_workload) * final_repeats
                ),
                "target_specific_measurement": formal_protocol(args),
            },
        },
        "hnsw_dominance_guard": args.hnsw_dominance_guard,
        "service_identity": service_identity,
        "calibration": {
                        "request_nos": [request.request_no for request in calibration_workload],
                        "query_nos": list(calibration_query_nos),
                        "queries": len(calibration_query_nos),
                        "repeats": calibration_repeats, "schedule": schedule,
                        "schedule_order": "all per-filter flat representatives before any HNSW configuration",
                        "selection_rule": CALIBRATION_SELECTION_RULE,
                        "qualification_metric": CALIBRATION_STOP_METRIC,
                        "qualification_operator": ">= target",
                        "bootstrap_ci_lcb": "qualification_and_early_stop",
                        "selection_policy": CALIBRATION_SELECTION_POLICY,
                        "fallback": CALIBRATION_FALLBACK,
                        "fallback_requires_full_hnsw_grid": True,
                        "early_stop_metric": CALIBRATION_STOP_METRIC,
                        "early_stop_target": float(args.targets[-1])},
        "final": {
                  "request_nos": [request.request_no for request in final_workload],
                  "query_nos": list(final_query_nos),
                  "queries": len(final_query_nos), "repeats": final_repeats,
                  "final_warmup_phase": "final_warmup",
                  "final_warmup_request_nos": [
                      request.request_no for request in final_workload[:args.warmup_queries]
                  ],
                  "final_warmup_queries_per_selected_configuration": args.warmup_queries,
                  "final_warmup_excluded_from_summaries": True,
                  "runs_selected_system_configs_plus_flat_exactness_controls": True,
                  "deduplication_key": (
                      ["configured_filter_strategy", "filter_name",
                       "flat_search_cutoff", "ef", "target_recall"]
                      if formal_protocol(args) else
                      ["configured_filter_strategy", "filter_name",
                       "flat_search_cutoff", "ef"]
                  ),
                  "reuses_one_exact_measurement_for_multiple_targets": not formal_protocol(args),
                  "target_specific_measurement": formal_protocol(args),
                  "retunes_after_held_out_measurement": False},
        "checkpoint": {"path": str(checkpoint), "persistence": "atomic complete-block snapshot",
                       "storage": "single JSON snapshot", "complete_block_boundary": True,
                       "original_schema_sha256": state["original_schema_sha256"],
                       "run_spec_hash": run_spec_hash(specification)},
        "measurement_mode": "single_client_sequential",
        "timed_graphql_response": {
            "fields": list(TIMED_RESPONSE_FIELDS),
            "all_scalar_properties_requested": False,
        },
        "throughput": {
            "qps_measured": False,
            "definition": QPS_DEFINITION,
            "latency_reciprocal_is_not_measured_qps": True,
        },
        "schema_timings": state["schema_timings"],
    }
    outcome_counts = baseline.target_outcome_counts(final_summaries, statuses.values())
    outcome_notes = [
        f"held-out target unconfirmed: filter={row.get('filter_name')} target={row.get('target_recall')}"
        for row in final_summaries if row.get("target_outcome") == "selected_but_final_unconfirmed"
    ]
    publication_errors = publication_gate_errors(errors, final_summaries)
    expected_target_cells = len(filters) * len(args.targets)
    observed_target_cells = len(final_summaries)
    expected_final_samples = (
        sum(len(requests) for requests in final_requests_by_filter.values())
        * final_repeats
        * len(args.targets)
        if formal_protocol(args)
        else sum(
            len(final_requests_by_filter[name]) * final_repeats
            for _, name, _, _, _ in selected_runs
        )
    )
    observed_final_samples = sum(
        row.get("phase") == "final"
        and (not formal_protocol(args) or row.get("target_recall", NA) != NA)
        for row in raw_rows
    )
    requested_slice_complete = bool(
        not errors
        and observed_target_cells == expected_target_cells
        and observed_final_samples == expected_final_samples
    )
    full_release_complete = bool(
        requested_slice_complete
        and {spec.name for spec in filters}
        == {spec.name for spec in baseline.FILTERS}
        and observed_target_cells == len(baseline.FILTERS) * len(args.targets)
    )
    if not requested_slice_complete:
        publication_errors.append("requested shard/filter slice is incomplete")
    paper_eligible = not publication_errors
    manifest = {
        "artifact_valid": not errors, "status": "complete" if not errors else "invalid",
        "paper_eligible": paper_eligible,
        "requested_slice_complete": requested_slice_complete,
        "full_release_complete": full_release_complete,
        "publication_errors": publication_errors,
        "git_revision": revision, "source_hashes": source_hashes,
        "current_protocol": current_protocol(args),
        "execution_protocol": args.protocol,
        "input_provenance": input_provenance,
        "run_spec_hash": run_spec_hash(specification), "stale_outputs_quarantined_at": str(quarantined) if quarantined else NA,
        "service": {"meta": service_meta, "version": service_identity["actual_version"],
                    "expected_version": service_identity["expected_version"],
                    "version_gate_passed": True,
                    "image_digest": service_identity["service_image_digest"], "count": total_count,
                    "filter_counts": filter_counts, "measurement_mode": "single_client_sequential", "concurrency": 1,
                    "errors": errors},
        "outcome_notes": outcome_notes, "target_outcomes": outcome_counts,
        "release_scope": {
            "kind": (
                "formal_filter_target_shard"
                if formal_protocol(args) else "legacy_filter_target_shard"
            ),
            "requested_slice_complete": requested_slice_complete,
            "full_release_complete": full_release_complete,
            "expected_filter_target_cells": expected_target_cells,
            "observed_filter_target_cells": observed_target_cells,
            "expected_final_samples": expected_final_samples,
            "observed_final_samples": observed_final_samples,
            "full_release_filter_target_cells": (
                len(baseline.FILTERS) * len(args.targets)
            ),
            "full_release_samples_per_target": len(final_workload) * final_repeats,
        },
        "schema": {"update_method": "GET full definition, PUT full definition, GET readback gate",
                   "gated_fields": ["filterStrategy", "ef", "flatSearchCutoff"],
                   "original_schema_sha256": state["original_schema_sha256"],
                   "resume_restore_source": "checkpoint original_schema",
                   "original_definition_restored": True},
        "route_reporting": {
            "contract": ROUTE_REPORTING_CONTRACT,
            "configured_strategies": list(DEFAULT_FILTER_STRATEGIES),
            "automatic_fallback": "possible_service_internal",
            "effective_route_observed": False,
            "configured_route_records": route_expectations,
            "cutoff_configuration_coverage": equivalence_proofs,
        },
        "timed_graphql_response": {
            "fields": list(TIMED_RESPONSE_FIELDS),
            "all_scalar_properties_requested": False,
            "latency_includes": "HTTP request, response read, JSON parse, row_id and distance extraction",
        },
        "throughput": {
            "qps_measured": False,
            "definition": QPS_DEFINITION,
            "latency_reciprocal_is_not_measured_qps": True,
        },
        "hnsw_flat_dominance": {
            "guard": args.hnsw_dominance_guard,
            "source_budget_proof": HNSW_EF_BUDGET_PROOF,
            "diagnostic_only": True,
            "route_proofs": {
                spec_name: {
                    strategy: route["termination"]
                    for strategy, route in proof["hnsw_routes"].items()
                }
                for spec_name, proof in grid_proofs.items()
            },
        },
        "actual_calibration": actual_calibration,
        "flat_held_out_exactness_gate": {
            "required_recall_mean": 1.0, "required_recall_lcb95": 1.0,
            "bootstrap_ci_lcb": "exactness_gate",
            "records": flat_exactness_records,
        },
        "checkpoint": {"persistence": "atomic complete-block snapshot",
                       "storage": "single JSON snapshot", "complete_block_boundary": True,
                       "original_schema_persisted_before_schema_put": True},
        "calibration_selection": {
            "rule": CALIBRATION_SELECTION_RULE,
            "selection_policy": CALIBRATION_SELECTION_POLICY,
            "qualification_metric": CALIBRATION_STOP_METRIC,
            "qualification_operator": ">= target",
            "bootstrap_ci_lcb": "qualification_and_early_stop",
            "lcb_is_report_only": False,
            "fallback": CALIBRATION_FALLBACK,
            "fallback_requires_full_hnsw_grid": True,
            "early_stop_metric": CALIBRATION_STOP_METRIC,
            "early_stop_target": float(args.targets[-1]),
            "targets": [
            {"filter_name": name, "target_recall": target,
             "status": statuses[(name, target)],
             "selected_filter_strategy": selected["configured_filter_strategy"] if selected else NA,
             "selected_ef": selected["ef"] if selected else NA,
             "selected_flat_search_cutoff": selected["flat_search_cutoff"] if selected else NA,
             "selection_mode": selected["selection_mode"] if selected else NA,
             "calibration_selection_policy": selected.get("calibration_selection_policy", NA) if selected else NA,
             "calibration_target_recall": selected.get("calibration_target_recall", target) if selected else target,
             "calibration_recall_mean": selected.get("recall_mean", NA) if selected else NA,
             "calibration_recall_lcb95": selected.get("recall_lcb95", NA) if selected else NA,
             "calibration_lcb95_qualified": selected.get("calibration_lcb95_qualified", NA) if selected else NA,
             "fallback_used": selected.get("fallback_used", NA) if selected else NA,
             "grid_proof": grid_proofs[name]}
            for (name, target), selected in sorted(selections.items())],
            "scope": "one independently calibrated production route per filter/target"},
        "raw_rows": len(raw_rows), "summary_rows": len(calibration_summaries) + len(final_summaries),
    }
    staged = baseline.staging_outputs(outputs)
    try:
        baseline.write_csv(staged["raw_csv"], raw_rows)
        baseline.write_csv(staged["summary_csv"], calibration_summaries + final_summaries)
        baseline.write_json(staged["schema_json"], {"class": CLASS_NAME, "source_hashes": source_hashes, "records": state["schema_records"]})
        baseline.write_json(staged["config_json"], config)
        manifest = baseline.commit_output_bundle(outputs, staged, manifest)
    finally:
        baseline.cleanup_staging(staged)
    checkpoint.unlink(missing_ok=True)
    return 0 if manifest["paper_eligible"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return _dry_run(args)
    try:
        return run(args)
    except Exception as exc:
        print(f"artifact_valid=false: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
