from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import psycopg

try:
    from .common_pg import pg_config_from_env
except ImportError:
    from common_pg import pg_config_from_env


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT
    / "experiments/hybrid_vector_db/configs/figure5_frontier_datasets.json"
)
CORE_RUNNER = (
    ROOT
    / "experiments/hybrid_vector_db/scripts/"
    "pgvector_design1_design2_design3_selectivity_benchmark.py"
)
RESULTS = ROOT / "results/hybrid_vector_db"
DEFAULT_GLOBAL_DB_LOCK_PATH = (
    RESULTS / ".figure5_formal_calibration_global_db.lock"
)
GLOBAL_DB_LOCK_PROTOCOL = "fcntl_flock_exclusive_nonblocking_v1"
MODES = ("original", "design1_bloom_bfs_layout_d3")
STANDARD_SCAN_FAMILIES = ("both_off", "stock_strict")
SQLENS_CAP_FAMILY = "sqlens_cap"
SCAN_FAMILIES = (*STANDARD_SCAN_FAMILIES, SQLENS_CAP_FAMILY)
MAX_SQLENS_CALIBRATION_EF = 1000
FORMAL_CALIBRATION_DATASETS = ("amazon", "yfcc", "laion")
FORMAL_QUALIFICATION_SCOPE = "global_min_predicate_lcb"
FORMAL_CALIBRATION_PROTOCOL = "formal_per_predicate_cartesian_v1"
FORMAL_CALIBRATION_REQUESTS = 2_800
FORMAL_CALIBRATION_FILTERS = 14
FORMAL_CALIBRATION_PER_FILTER = 200
RELEASE_TAG_RE = re.compile(r"(?:^|-)r([1-9][0-9]*)(?:-|$)")
ARTIFACT_PREFIX_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
FORMAL_LOW_BUDGETS = (11, 12, 14, 16, 18)
FORMAL_BOTH_OFF_BUDGETS = (
    20,
    40,
    60,
    80,
    100,
    150,
    200,
    250,
    500,
    750,
    1000,
)
FORMAL_SQLENS_EXTENSIONS = {"laion": (1500, 2000)}
FORMAL_SQLENS_SCAN_CAPS = (
    500,
    1000,
    2000,
    5000,
    10_000,
    20_000,
    50_000,
    100_000,
)


class Figure5ContractError(RuntimeError):
    """Raised when a Figure 5 run would violate the frozen protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Figure5ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Figure5ContractError(f"JSON root must be an object: {path}")
    return value


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_int_values(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("search budgets must be positive")
    return values


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as source:
        return sum(1 for _ in csv.DictReader(source))


def formal_workload_required(protocol: dict[str, Any]) -> bool:
    """Whether this config claims the formal per-predicate quality protocol."""
    return (
        protocol.get("qualification_scope") == FORMAL_QUALIFICATION_SCOPE
        or protocol.get("calibration_protocol") == FORMAL_CALIBRATION_PROTOCOL
    )


def manifest_content_sha256(manifest: dict[str, Any]) -> str:
    """Verify the workload builder's manifest-last canonical content hash."""
    sanitized = json.loads(json.dumps(manifest))
    try:
        sanitized["outputs"]["manifest_json"].pop("content_sha256", None)
    except KeyError as exc:
        raise Figure5ContractError(
            "workload manifest has no self-referential content SHA"
        ) from exc
    payload = json.dumps(sanitized, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def formal_workload_contract(
    dataset_name: str,
    dataset: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any] | None:
    """Load and fail-close on the formal workload artifact contract.

    Legacy frontier configs intentionally have no workload manifest requirement.
    A config that opts into either the formal qualification scope or formal
    calibration protocol, however, must bind every run cell to the audited
    14 x 200 Cartesian workload bundle that selected its search configuration.
    """
    if not formal_workload_required(protocol):
        return None
    if protocol.get("qualification_scope") != FORMAL_QUALIFICATION_SCOPE:
        raise Figure5ContractError(
            f"{dataset_name}: formal calibration requires qualification_scope="
            f"{FORMAL_QUALIFICATION_SCOPE!r}"
        )
    if protocol.get("calibration_protocol") != FORMAL_CALIBRATION_PROTOCOL:
        raise Figure5ContractError(
            f"{dataset_name}: formal qualification requires calibration_protocol="
            f"{FORMAL_CALIBRATION_PROTOCOL!r}"
        )
    if int(protocol.get("calibration_requests") or -1) != FORMAL_CALIBRATION_REQUESTS:
        raise Figure5ContractError(
            f"{dataset_name}: formal calibration_requests must be "
            f"{FORMAL_CALIBRATION_REQUESTS}"
        )

    manifest_value = dataset.get("workload_manifest_json")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise Figure5ContractError(
            f"{dataset_name}: formal workload requires dataset.workload_manifest_json"
        )
    manifest_path = resolve_path(manifest_value)
    if not manifest_path.is_file():
        raise Figure5ContractError(
            f"{dataset_name}: workload manifest is missing: {manifest_path}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("artifact_valid") is not True:
        raise Figure5ContractError(
            f"{dataset_name}: workload manifest is not artifact_valid"
        )
    formal_gate = manifest.get("formal_paper_calibration")
    if not isinstance(formal_gate, dict) or formal_gate.get("passed") is not True:
        raise Figure5ContractError(
            f"{dataset_name}: workload manifest did not pass formal paper calibration"
        )
    construction = manifest.get("construction")
    distribution = manifest.get("distribution")
    if not isinstance(construction, dict) or not isinstance(distribution, dict):
        raise Figure5ContractError(
            f"{dataset_name}: workload manifest is missing construction/distribution"
        )
    calibration = construction.get("calibration")
    calibration_distribution = distribution.get("calibration")
    if not isinstance(calibration, dict) or not isinstance(calibration_distribution, dict):
        raise Figure5ContractError(
            f"{dataset_name}: workload manifest is missing calibration evidence"
        )
    if (
        calibration.get("protocol") != FORMAL_CALIBRATION_PROTOCOL
        or calibration.get("per_predicate_cartesian") is not True
        or int(calibration.get("requests") or -1) != FORMAL_CALIBRATION_REQUESTS
        or int(calibration.get("query_count") or -1)
        != FORMAL_CALIBRATION_PER_FILTER
    ):
        raise Figure5ContractError(
            f"{dataset_name}: workload manifest is not the formal 14x200 protocol"
        )
    filter_counts = calibration_distribution.get("filter_counts")
    coverage = calibration_distribution.get("cartesian_coverage")
    if (
        not isinstance(filter_counts, dict)
        or len(filter_counts) != FORMAL_CALIBRATION_FILTERS
        or any(int(count) != FORMAL_CALIBRATION_PER_FILTER for count in filter_counts.values())
        or not isinstance(coverage, dict)
        or coverage.get("complete") is not True
        or int(coverage.get("expected_pairs") or -1) != FORMAL_CALIBRATION_REQUESTS
        or int(coverage.get("observed_rows") or -1) != FORMAL_CALIBRATION_REQUESTS
        or int(coverage.get("observed_unique_pairs") or -1)
        != FORMAL_CALIBRATION_REQUESTS
        or coverage.get("missing_pairs") != 0
        or coverage.get("duplicate_pairs") != 0
    ):
        raise Figure5ContractError(
            f"{dataset_name}: formal Cartesian calibration coverage is incomplete"
        )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise Figure5ContractError(f"{dataset_name}: workload manifest has no outputs")
    output_contracts: dict[str, dict[str, Any]] = {}
    for output_name, config_key in (
        ("calibration_workload_csv", "calibration_workload_csv"),
        ("measurement_workload_csv", "measurement_workload_csv"),
    ):
        recorded = outputs.get(output_name)
        configured_value = dataset.get(config_key)
        if not isinstance(recorded, dict) or not isinstance(configured_value, str):
            raise Figure5ContractError(
                f"{dataset_name}: workload output/config is incomplete for {output_name}"
            )
        recorded_value = recorded.get("path")
        recorded_sha = recorded.get("sha256")
        if not isinstance(recorded_value, str) or not isinstance(recorded_sha, str):
            raise Figure5ContractError(
                f"{dataset_name}: workload manifest has no path/SHA for {output_name}"
            )
        configured_path = resolve_path(configured_value).resolve()
        recorded_path = resolve_path(recorded_value).resolve()
        if configured_path != recorded_path:
            raise Figure5ContractError(
                f"{dataset_name}: config {config_key} does not match workload manifest"
            )
        if not configured_path.is_file() or sha256_file(configured_path) != recorded_sha:
            raise Figure5ContractError(
                f"{dataset_name}: {output_name} is missing or SHA-mismatched"
            )
        output_contracts[output_name] = {
            "path": str(configured_path),
            "sha256": recorded_sha,
            "rows": int(recorded.get("rows") or -1),
        }
    if output_contracts["calibration_workload_csv"]["rows"] != FORMAL_CALIBRATION_REQUESTS:
        raise Figure5ContractError(
            f"{dataset_name}: formal calibration output must have "
            f"{FORMAL_CALIBRATION_REQUESTS} rows"
        )
    if count_csv_rows(Path(output_contracts["calibration_workload_csv"]["path"])) != (
        FORMAL_CALIBRATION_REQUESTS
    ):
        raise Figure5ContractError(
            f"{dataset_name}: calibration CSV row count does not match manifest"
        )
    manifest_record = outputs.get("manifest_json")
    if not isinstance(manifest_record, dict) or manifest_record.get("path") is None:
        raise Figure5ContractError(
            f"{dataset_name}: workload manifest is missing self provenance"
        )
    if resolve_path(str(manifest_record["path"])).resolve() != manifest_path.resolve():
        raise Figure5ContractError(
            f"{dataset_name}: workload manifest self path does not match config"
        )
    expected_content_sha = manifest_record.get("content_sha256")
    if not isinstance(expected_content_sha, str) or expected_content_sha != manifest_content_sha256(manifest):
        raise Figure5ContractError(
            f"{dataset_name}: workload manifest content SHA is invalid"
        )
    return {
        "required": True,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": expected_content_sha,
            "artifact_valid": True,
        },
        "protocol": {
            "qualification_scope": FORMAL_QUALIFICATION_SCOPE,
            "calibration_protocol": FORMAL_CALIBRATION_PROTOCOL,
            "calibration_requests": FORMAL_CALIBRATION_REQUESTS,
            "filters": FORMAL_CALIBRATION_FILTERS,
            "per_filter_requests": FORMAL_CALIBRATION_PER_FILTER,
        },
        "formal_paper_calibration": {
            "passed": True,
            "cartesian_complete": True,
            "canonical_pair_sha256": coverage.get("canonical_pair_sha256"),
        },
        "outputs": output_contracts,
    }


def clear_fragment_store_namespace(table: str, namespace: str) -> int:
    if not namespace:
        raise Figure5ContractError("cannot clear an empty D3 fragment-store namespace")
    prefix = namespace + "\x1f"
    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM public.pgvector_hnsw_fragment_store "
            "WHERE heap_oid = %s::regclass "
            "AND left(filter_name, char_length(%s)) = %s",
            (table, prefix, prefix),
        )
        deleted = int(cur.rowcount)
        cur.execute(
            "SELECT count(*) "
            "FROM public.pgvector_hnsw_fragment_store "
            "WHERE heap_oid = %s::regclass "
            "AND left(filter_name, char_length(%s)) = %s",
            (table, prefix, prefix),
        )
        remaining = int(cur.fetchone()[0])
    if remaining != 0:
        raise Figure5ContractError(
            f"failed to clear D3 namespace {namespace!r}: remaining={remaining}"
        )
    return deleted


def fragment_store_namespaces(provenance: dict[str, Any]) -> list[str]:
    base = str(provenance.get("d3_fragment_store_namespace") or "")
    if not base:
        return []
    if bool(provenance.get("isolated_repeat_runtimes")):
        return [
            f"{base}-r{repeat}"
            for repeat in range(int(provenance["repeats"]))
        ]
    return [base]


def namespace_start_matches(
    evidence: object,
    provenance: dict[str, Any],
) -> bool:
    expected = fragment_store_namespaces(provenance)
    if not expected:
        return True
    if not isinstance(evidence, dict):
        return False
    if bool(provenance.get("isolated_repeat_runtimes")):
        records = evidence.get("records")
        return (
            evidence.get("isolated_repeats") is True
            and evidence.get("base_namespace")
            == provenance["d3_fragment_store_namespace"]
            and isinstance(records, list)
            and len(records) == len(expected)
            and all(isinstance(record, dict) for record in records)
            and [record.get("namespace") for record in records] == expected
            and all(
                record.get("empty") is True
                and int(record.get("rows_before", -1)) == 0
                for record in records
            )
        )
    return (
        evidence.get("namespace") == expected[0]
        and evidence.get("empty") is True
        and int(evidence.get("rows_before", -1)) == 0
    )


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != 1:
        raise Figure5ContractError("unsupported Figure 5 config schema")
    artifact_prefix = config.get("artifact_prefix", "figure5_r35")
    if (
        not isinstance(artifact_prefix, str)
        or not ARTIFACT_PREFIX_RE.fullmatch(artifact_prefix)
    ):
        raise Figure5ContractError(
            "artifact_prefix must contain only lowercase letters, digits, and "
            "underscores, and must start with a letter or digit"
        )
    config["artifact_prefix"] = artifact_prefix
    protocol = config.get("protocol")
    datasets = config.get("datasets")
    grid = config.get("search_grid")
    if not isinstance(protocol, dict) or not isinstance(datasets, dict) or not isinstance(grid, dict):
        raise Figure5ContractError("Figure 5 config is missing protocol/datasets/search_grid")
    if tuple(protocol.get("modes", ())) != MODES:
        raise Figure5ContractError(
            "Figure 5 methods must be exactly Stock and full D1+D2+D3"
        )
    if protocol.get("execution_order") != "interleaved":
        raise Figure5ContractError("Figure 5 latency must use paired/interleaved execution")
    if protocol.get("d3_measurement_policy") != "workload_driven_adaptive":
        raise Figure5ContractError("Figure 5 must charge online D3 adaptation")
    release_path = resolve_path(str(config.get("release_contract") or ""))
    if not release_path.is_file():
        raise Figure5ContractError(f"release contract is missing: {release_path}")
    config["release_contract_path"] = str(release_path)
    config["release_contract_sha256"] = sha256_file(release_path)
    config["release_identity"] = read_json(release_path)
    return config


def selected_budgets(
    config: dict[str, Any],
    grid_name: str,
    explicit: Sequence[int],
) -> list[int]:
    if explicit:
        return list(dict.fromkeys(explicit))
    grid = config["search_grid"]
    if grid_name == "base":
        values = grid["ef_search"]
    elif grid_name == "extension":
        values = grid["extension_ef_search"]
    else:
        values = [*grid["ef_search"], *grid["extension_ef_search"]]
    return [int(value) for value in values]


def validate_calibration_budget_policy(
    phase: str,
    families: Sequence[str],
    budgets: Sequence[int],
    *,
    allow_expensive_sqlens_calibration: bool,
) -> None:
    if (
        phase == "calibration"
        and "both_off" in families
        and any(ef_search > MAX_SQLENS_CALIBRATION_EF for ef_search in budgets)
        and not allow_expensive_sqlens_calibration
    ):
        raise Figure5ContractError(
            "formal both_off calibration is capped at ef_search=1000; "
            "use --allow-expensive-sqlens-calibration only for an explicit "
            "dataset-specific frontier extension. Run routine high budgets "
            "with --scan-families stock_strict."
        )


def formal_calibration_cells(
    config: dict[str, Any],
) -> list[tuple[str, str, int, int | None]]:
    """Return the frozen, deduplicated Figure 5 calibration suite."""
    observed = set(config["datasets"])
    expected = set(FORMAL_CALIBRATION_DATASETS)
    if observed != expected:
        raise Figure5ContractError(
            "formal calibration requires exactly the three frozen datasets: "
            f"expected={sorted(expected)}, observed={sorted(observed)}"
        )
    strict_budgets = tuple(int(value) for value in config["search_grid"]["ef_search"])
    if not strict_budgets or any(value <= 0 for value in strict_budgets):
        raise Figure5ContractError("formal stock_strict calibration grid is invalid")

    cells: list[tuple[str, str, int, int | None]] = []
    for dataset in FORMAL_CALIBRATION_DATASETS:
        cells.extend(
            (dataset, family, budget, None)
            for family in STANDARD_SCAN_FAMILIES
            for budget in FORMAL_LOW_BUDGETS
        )
        cells.extend(
            (dataset, "both_off", budget, None)
            for budget in FORMAL_BOTH_OFF_BUDGETS
        )
        cells.extend(
            (dataset, "stock_strict", budget, None)
            for budget in strict_budgets
        )
        cells.extend(
            (dataset, "both_off", budget, None)
            for budget in FORMAL_SQLENS_EXTENSIONS.get(dataset, ())
        )
        cells.extend(
            (dataset, SQLENS_CAP_FAMILY, 11, cap)
            for cap in FORMAL_SQLENS_SCAN_CAPS
        )
    if len(cells) != 146 or len(cells) != len(set(cells)):
        raise Figure5ContractError(
            "formal calibration suite must contain 146 unique cells, "
            f"observed={len(cells)}"
        )
    return cells


def cell_paths(
    out_dir: Path,
    dataset_name: str,
    phase: str,
    family: str,
    ef_search: int,
    sqlens_scan_cap: int | None = None,
    artifact_prefix: str = "figure5_r35",
) -> tuple[Path, Path]:
    if not ARTIFACT_PREFIX_RE.fullmatch(artifact_prefix):
        raise Figure5ContractError(f"invalid artifact prefix: {artifact_prefix!r}")
    stem = f"{artifact_prefix}_{dataset_name}_{phase}_{family}_ef{ef_search}"
    if family == SQLENS_CAP_FAMILY:
        if sqlens_scan_cap is None or sqlens_scan_cap <= 0:
            raise Figure5ContractError(
                "sqlens_cap cells require a positive SQLens scan cap"
            )
        stem += f"_cap{sqlens_scan_cap}"
    elif sqlens_scan_cap is not None:
        raise Figure5ContractError(
            f"{family} cells cannot carry a SQLens-only scan cap"
        )
    raw = out_dir / f"{stem}.csv"
    return raw, raw.with_suffix(raw.suffix + ".plan.json")


def cell_complete(
    raw: Path,
    plan: Path,
    expected_rows: int,
    expected_provenance: dict[str, Any] | None = None,
) -> bool:
    if not raw.is_file() or not plan.is_file():
        return False
    try:
        evidence = read_json(plan)
        prewarm = evidence.get("relation_prewarm")
        prewarm_records = (
            prewarm.get("records", []) if isinstance(prewarm, dict) else []
        )
        with raw.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        config_matches = True
        if expected_provenance is not None:
            expected_modes = set(expected_provenance["modes"])
            mode_configs = expected_provenance["mode_configs"]
            if {row.get("mode") for row in rows} != expected_modes:
                config_matches = False
            else:
                for mode in expected_modes:
                    mode_rows = [row for row in rows if row.get("mode") == mode]
                    expected_mode_rows = (
                        int(expected_provenance["requests"])
                        * int(expected_provenance["repeats"])
                    )
                    if len(mode_rows) != expected_mode_rows:
                        config_matches = False
                        break
                    for field in (
                        "ef_search",
                        "max_scan_tuples",
                        "scan_mem_multiplier",
                        "guided_collect_target",
                        "traversal_guided_target",
                        "iterative_scan",
                    ):
                        expected_text = str(mode_configs[mode][field])
                        if any(str(row.get(field)) != expected_text for row in mode_rows):
                            config_matches = False
                            break
                    if not config_matches:
                        break
            query_contract = evidence.get("query_contract")
            namespace_start = evidence.get("d3_fragment_store_start")
            execution_sources = evidence.get("execution_sources")
            expected_release = expected_provenance["release_identity"]
            expected_build_id = expected_release["expected_sqlens_build_id"]
            expected_vector_sha = expected_release["expected_vector_so_sha256"]
            startup_identity = evidence.get("sqlens_runtime_identity_startup")
            final_identity = evidence.get("sqlens_runtime_identity_final")
            runtime_identities = evidence.get("runtime_sqlens_identity_evidence")

            def identity_matches(record: object) -> bool:
                return (
                    isinstance(record, dict)
                    and record.get("expected_build_id") == expected_build_id
                    and record.get("expected_vector_so_sha256")
                    == expected_vector_sha
                    and record.get("observed_build_id") == expected_build_id
                    and record.get("observed_vector_so_sha256")
                    == expected_vector_sha
                    and record.get("exact_match") is True
                )

            if (
                not isinstance(query_contract, dict)
                or query_contract.get("workload_sha256")
                != expected_provenance["inputs"]["workload"]["sha256"]
                or query_contract.get("truth_sha256")
                != expected_provenance["inputs"]["truth"]["sha256"]
                or query_contract.get("filters_sha256")
                != expected_provenance["inputs"]["filters"]["sha256"]
                or query_contract.get("d2_graph_proof_input_sha256")
                != expected_provenance["inputs"]["d2_graph_proof"][
                    "canonical_json_sha256"
                ]
                or execution_sources
                != expected_provenance["execution_sources"]
                or expected_provenance.get("release_contract_sha256")
                != expected_release.get("contract_sha256")
                or not identity_matches(startup_identity)
                or not identity_matches(final_identity)
                or not isinstance(runtime_identities, list)
                or not runtime_identities
                or not all(identity_matches(item) for item in runtime_identities)
                or any(
                    row.get("sqlens_build_id") != expected_build_id
                    or row.get("vector_so_sha256") != expected_vector_sha
                    for row in rows
                )
                or (
                    "design1_bloom_bfs_layout_d3" in expected_modes
                    and not namespace_start_matches(
                        namespace_start,
                        expected_provenance,
                    )
                )
            ):
                config_matches = False
        return (
            evidence.get("status") == "complete"
            and int(evidence.get("output_rows") or -1) == expected_rows
            and evidence.get("output_sha256") == sha256_file(raw)
            and len(rows) == expected_rows
            and all(not str(row.get("error") or "").strip() for row in rows)
            and isinstance(evidence.get("query_error_summary"), dict)
            and int(evidence["query_error_summary"].get("error_rows", -1)) == 0
            and config_matches
            and isinstance(prewarm, dict)
            and prewarm.get("enabled") is True
            and prewarm.get("complete") is True
            and len(prewarm_records) == 3
            and all(
                isinstance(item, dict)
                and
                int(item.get("warmed_blocks", -1))
                == int(item.get("expected_blocks", -2))
                and int(item.get("warmed_blocks", 0)) > 0
                for item in prewarm_records
            )
        )
    except (Figure5ContractError, OSError, ValueError, KeyError):
        return False


def mode_configs(
    family: str,
    ef_search: int,
    max_scan_tuples: int,
    scan_mem_multiplier: float,
    sqlens_scan_cap: int | None = None,
) -> dict[str, dict[str, object]]:
    if family not in SCAN_FAMILIES:
        raise Figure5ContractError(f"unknown scan family: {family}")
    # Keep the guided result budget coupled to ef_search so the formal
    # frontier actually trades more work for recall. Capping this at 40 made
    # every high-ef SQLens point retain the same number of valid candidates.
    traversal_target = max(11, ef_search)
    stock_iterative = "strict_order" if family == "stock_strict" else "off"
    common = {
        "ef_search": ef_search,
        "max_scan_tuples": max_scan_tuples,
        "scan_mem_multiplier": scan_mem_multiplier,
        "guided_collect_target": ef_search,
        "traversal_guided_target": traversal_target,
        "traversal_guided_burst": 8,
    }
    sqlens_max_scan_tuples = (
        int(sqlens_scan_cap)
        if sqlens_scan_cap is not None
        else max_scan_tuples
    )
    return {
        "original": {
            **common,
            "iterative_scan": stock_iterative,
            "traversal_guided_prioritization": False,
        },
        "design1_bloom_bfs_layout_d3": {
            **common,
            "max_scan_tuples": sqlens_max_scan_tuples,
            "iterative_scan": "off",
            "traversal_guided_prioritization": True,
        },
    }


def release_namespace_prefix(release: dict[str, Any]) -> str:
    contract_id = str(release.get("contract_id") or "")
    match = RELEASE_TAG_RE.search(contract_id)
    if match is None:
        raise Figure5ContractError(
            "release contract_id must contain an explicit rNN tag for "
            f"D3 namespace isolation: {contract_id!r}"
        )
    return f"fig5-r{match.group(1)}"


def build_cell_command(
    config: dict[str, Any],
    dataset_name: str,
    phase: str,
    family: str,
    ef_search: int,
    raw: Path,
    backend_cpu_list: str,
    calibration_repeats: int = 1,
    sqlens_scan_cap: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    dataset = config["datasets"][dataset_name]
    protocol = config["protocol"]
    workload_contract = formal_workload_contract(dataset_name, dataset, protocol)
    grid = config["search_grid"]
    workload_key = f"{phase}_workload_csv"
    workload = resolve_path(str(dataset[workload_key]))
    truth = resolve_path(str(dataset["truth_csv"]))
    filters = resolve_path(str(dataset["filters_csv"]))
    graph_proof = resolve_path(str(dataset["d2_graph_proof_json"]))
    requests = int(protocol[f"{phase}_requests"])
    repeats = (
        calibration_repeats
        if phase == "calibration"
        else int(protocol["latency_repeats"])
    )
    required_paths = {
        "workload": workload,
        "truth": truth,
        "filters": filters,
        "d2_graph_proof": graph_proof,
    }
    missing = [f"{name}={path}" for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise Figure5ContractError(
            f"{dataset_name}/{phase} inputs are incomplete: " + ", ".join(missing)
        )
    if count_csv_rows(workload) != requests:
        raise Figure5ContractError(
            f"{dataset_name}/{phase} workload does not contain {requests} rows"
        )

    release = config["release_identity"]
    namespace = (
        f"{release_namespace_prefix(release)}-{dataset_name}-{phase}-"
        f"{family}-ef{ef_search}"
        + (
            f"-cap{sqlens_scan_cap}"
            if sqlens_scan_cap is not None
            else ""
        )
    )
    if phase == "measurement" or family == "both_off":
        cell_modes = MODES
    elif family == "stock_strict":
        cell_modes = ("original",)
    elif family == SQLENS_CAP_FAMILY:
        cell_modes = ("design1_bloom_bfs_layout_d3",)
    else:
        raise Figure5ContractError(f"unknown scan family: {family}")
    mode_config = mode_configs(
        family,
        ef_search,
        int(grid["max_scan_tuples"]),
        float(grid["scan_mem_multiplier"]),
        sqlens_scan_cap,
    )
    prewarm_relations = [
        str(dataset["table"]),
        str(dataset["source_index"]),
        str(dataset["bfs_index"]),
    ]
    command = [
        sys.executable,
        str(CORE_RUNNER),
        "--insertion-table",
        str(dataset["table"]),
        "--insertion-index",
        str(dataset["source_index"]),
        "--bfs-table",
        str(dataset["table"]),
        "--bfs-index",
        str(dataset["bfs_index"]),
        "--query-table",
        str(dataset["query_table"]),
        "--query-id-column",
        str(dataset["query_id_column"]),
        "--query-vector-column",
        str(dataset["query_vector_column"]),
        "--candidate-validity-predicate",
        str(dataset["candidate_validity_predicate"]),
        (
            "--expected-truth-self-excluded"
            if bool(dataset["truth_self_excluded"])
            else "--no-expected-truth-self-excluded"
        ),
        "--truth-csv",
        str(truth),
        "--workload-csv",
        str(workload),
        "--expected-workload-requests",
        str(requests),
        (
            "--require-unique-workload-queries"
            if phase == "measurement"
            else "--no-require-unique-workload-queries"
        ),
        "--filters-csv",
        str(filters),
        *[
            item
            for relation in prewarm_relations
            for item in ("--prewarm-relation", relation)
        ],
        "--modes",
        *cell_modes,
        "--execution-order",
        "interleaved",
        "--schedule-seed",
        str(int(protocol["schedule_seed"])),
        "--mode-configs-json",
        json.dumps(mode_config, separators=(",", ":"), sort_keys=True),
        "--repeats",
        str(repeats),
        *(
            ["--isolate-repeat-runtimes"]
            if repeats > 1
            else []
        ),
        "--warmup-queries",
        "1",
        "--no-warmup-all-queries",
        "--ef-search",
        str(ef_search),
        "--guided-collect-target",
        str(ef_search),
        "--traversal-guided-target",
        str(max(11, ef_search)),
        "--traversal-guided-prioritization",
        "--guidance-filter-strategy",
        str(protocol["guidance_filter_strategy"]),
        "--iterative-scan",
        "off",
        "--max-scan-tuples",
        str(int(grid["max_scan_tuples"])),
        "--scan-mem-multiplier",
        str(float(grid["scan_mem_multiplier"])),
        "--d2-page-access",
        str(protocol["d2_page_access"]),
        "--d2-index-page-access",
        str(protocol["d2_index_page_access"]),
        "--d1-guidance-kind",
        "auto",
        "--d3-measurement-policy",
        str(protocol["d3_measurement_policy"]),
        "--d3-fragment-store-namespace",
        namespace,
        "--guidance-selectivity-max-pct",
        "100",
        "--guidance-max-atoms",
        str(int(protocol["guidance_max_atoms"])),
        "--statement-timeout-ms",
        "7200000",
        "--force-hnsw",
        "--require-preferred-index-guc",
        "--d2-graph-proof-json",
        str(graph_proof),
        "--expected-sqlens-build-id",
        str(release["expected_sqlens_build_id"]),
        "--expected-vector-so-sha256",
        str(release["expected_vector_so_sha256"]),
        "--backend-cpu-list",
        backend_cpu_list,
        "--progress-queries",
        "25" if phase == "calibration" else "250",
        "--out",
        str(raw),
        "--orchestrator-source",
        str(Path(__file__).resolve()),
    ]
    provenance = {
        "dataset": dataset_name,
        "dataset_label": dataset["label"],
        "phase": phase,
        "scan_family": family,
        "ef_search": ef_search,
        "sqlens_scan_cap": sqlens_scan_cap,
        "requests": requests,
        "repeats": repeats,
        "isolated_repeat_runtimes": repeats > 1,
        "expected_rows": requests * repeats * len(cell_modes),
        "modes": list(cell_modes),
        "d3_fragment_store_namespace": (
            namespace
            if "design1_bloom_bfs_layout_d3" in cell_modes
            else ""
        ),
        "d3_fragment_store_table": str(dataset["table"]),
        "mode_configs": mode_config,
        "cache_protocol": {
            "state": "warm",
            "method": "pg_prewarm(regclass,'read','main')",
            "relations": prewarm_relations,
            "excluded_from_measured_query_latency": True,
            "d3_materialization_excluded": False,
        },
        "workload_contract": workload_contract
        if workload_contract is not None
        else {
            "required": False,
            "protocol": {
                "qualification_scope": protocol.get("qualification_scope"),
                "calibration_protocol": protocol.get("calibration_protocol"),
                "calibration_requests": protocol.get("calibration_requests"),
            },
        },
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in required_paths.items()
        },
        "execution_sources": {
            "core_runner": {
                "path": str(CORE_RUNNER.resolve()),
                "sha256": sha256_file(CORE_RUNNER.resolve()),
            },
            "orchestrator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "release_identity": {
            **release,
            "contract_sha256": config["release_contract_sha256"],
        },
        "release_contract_sha256": config["release_contract_sha256"],
    }
    provenance["inputs"]["d2_graph_proof"]["canonical_json_sha256"] = (
        sha256_json(read_json(graph_proof))
    )
    return command, provenance


def acquire_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise Figure5ContractError(f"another Figure 5 runner owns {path}") from exc
    return handle


class GlobalDBLock:
    def __init__(self, path: Path, handle: Any, owner_runner: str) -> None:
        self.path = path.resolve()
        self.handle = handle
        self.owner_runner = owner_runner
        self.owner_pid = os.getpid()
        self.owner_token = uuid.uuid4().hex
        self.acquired_at = utc_now()
        self._closed = False
        owner = {
            "protocol": GLOBAL_DB_LOCK_PROTOCOL,
            "path": str(self.path),
            "owner_runner": self.owner_runner,
            "owner_pid": self.owner_pid,
            "owner_token": self.owner_token,
            "acquired_at": self.acquired_at,
        }
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(
            json.dumps(owner, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def evidence(self, *, held_through_completion: bool) -> dict[str, Any]:
        if self._closed:
            raise Figure5ContractError(
                "cannot emit global DB isolation evidence after lock release"
            )
        return {
            "parallel_db_cells": False,
            "lock_required": True,
            "lock_path": str(self.path),
            "lock_protocol": GLOBAL_DB_LOCK_PROTOCOL,
            "lock_acquired": True,
            "lock_owner_runner": self.owner_runner,
            "lock_owner_pid": self.owner_pid,
            "lock_owner_token": self.owner_token,
            "lock_acquired_at": self.acquired_at,
            "held_through_completion": held_through_completion,
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self._closed = True


def planned_global_db_isolation(path: Path) -> dict[str, Any]:
    return {
        "parallel_db_cells": None,
        "lock_required": True,
        "lock_path": str(path.resolve()),
        "lock_protocol": GLOBAL_DB_LOCK_PROTOCOL,
        "lock_acquired": False,
        "held_through_completion": False,
    }


def acquire_global_db_lock(path: Path, owner_runner: str) -> GlobalDBLock:
    resolved = path.resolve()
    try:
        handle = acquire_lock(resolved)
    except Figure5ContractError as exc:
        raise Figure5ContractError(
            f"global Figure 5 DB lock is already owned: {resolved}"
        ) from exc
    try:
        return GlobalDBLock(resolved, handle, owner_runner)
    except Exception:
        handle.close()
        raise


def completed_isolation_evidence_valid(
    evidence: object,
    lock_path: Path,
) -> bool:
    return (
        isinstance(evidence, dict)
        and evidence.get("parallel_db_cells") is False
        and evidence.get("lock_required") is True
        and evidence.get("lock_path") == str(lock_path.resolve())
        and evidence.get("lock_protocol") == GLOBAL_DB_LOCK_PROTOCOL
        and evidence.get("lock_acquired") is True
        and evidence.get("held_through_completion") is True
        and isinstance(evidence.get("lock_owner_token"), str)
        and bool(evidence["lock_owner_token"])
    )


def prior_completed_cell_isolation(
    prior_manifest: dict[str, Any] | None,
    raw_path: Path,
    lock_path: Path,
) -> dict[str, Any] | None:
    if prior_manifest is None:
        return None
    for cell in prior_manifest.get("schedule", []):
        if (
            isinstance(cell, dict)
            and cell.get("raw") == str(raw_path)
            and cell.get("status") == "complete"
            and completed_isolation_evidence_valid(
                cell.get("database_isolation"),
                lock_path,
            )
        ):
            return dict(cell["database_isolation"])
    return None


def global_db_lock_path(args: argparse.Namespace) -> Path:
    configured = args.global_db_lock_path
    if configured is not None and not args.require_global_db_lock:
        raise Figure5ContractError(
            "--global-db-lock-path requires --require-global-db-lock"
        )
    return (configured or DEFAULT_GLOBAL_DB_LOCK_PATH).resolve()


def run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_config(config_path)
    db_lock_path = global_db_lock_path(args)
    if args.calibration_repeats < 1:
        raise Figure5ContractError("--calibration-repeats must be positive")
    if args.phase != "calibration" and args.calibration_repeats != 1:
        raise Figure5ContractError(
            "--calibration-repeats applies only to calibration/canary runs"
        )
    if args.formal_calibration_suite:
        if (
            args.phase != "calibration"
            or args.calibration_repeats != 1
            or args.datasets
            or args.ef_search_values
            or args.grid != "base"
            or tuple(args.scan_families) != STANDARD_SCAN_FAMILIES
            or args.sqlens_scan_cap_values
            or args.allow_expensive_sqlens_calibration
        ):
            raise Figure5ContractError(
                "--formal-calibration-suite owns datasets, grid, scan families, "
                "repeat count, and the bounded LAION extension; do not combine "
                "it with slice overrides"
            )
        schedule_cells = formal_calibration_cells(config)
        dataset_names = list(FORMAL_CALIBRATION_DATASETS)
        budgets = sorted({budget for _, _, budget, _ in schedule_cells})
        families = list(SCAN_FAMILIES)
    else:
        dataset_names = args.datasets or list(config["datasets"])
        unknown = sorted(set(dataset_names) - set(config["datasets"]))
        if unknown:
            raise Figure5ContractError(f"unknown datasets: {unknown}")
        budgets = selected_budgets(config, args.grid, args.ef_search_values)
        families = args.scan_families
        validate_calibration_budget_policy(
            args.phase,
            families,
            budgets,
            allow_expensive_sqlens_calibration=args.allow_expensive_sqlens_calibration,
        )
        if SQLENS_CAP_FAMILY in families and not args.sqlens_scan_cap_values:
            raise Figure5ContractError(
                "sqlens_cap slices require --sqlens-scan-cap-values"
            )
        if args.sqlens_scan_cap_values and SQLENS_CAP_FAMILY not in families:
            raise Figure5ContractError(
                "--sqlens-scan-cap-values requires --scan-families sqlens_cap"
            )
        schedule_cells = [
            (dataset_name, family, ef_search, cap)
            for dataset_name in dataset_names
            for family in families
            for ef_search in budgets
            for cap in (
                args.sqlens_scan_cap_values
                if family == SQLENS_CAP_FAMILY
                else (None,)
            )
        ]
    out_dir = args.out_dir.resolve()
    artifact_prefix = config.get("artifact_prefix", "figure5_r35")
    manifest_path = out_dir / f"{artifact_prefix}_{args.phase}_run_manifest.json"
    lock = acquire_lock(manifest_path.with_suffix(".lock"))
    try:
        prior_manifest = (
            read_json(manifest_path)
            if args.require_global_db_lock and manifest_path.is_file()
            else None
        )
        schedule: list[dict[str, Any]] = []
        for dataset_name, family, ef_search, sqlens_scan_cap in schedule_cells:
            raw, plan = cell_paths(
                out_dir,
                dataset_name,
                args.phase,
                family,
                ef_search,
                sqlens_scan_cap,
                artifact_prefix,
            )
            command, provenance = build_cell_command(
                config,
                dataset_name,
                args.phase,
                family,
                ef_search,
                raw,
                args.backend_cpu_list,
                args.calibration_repeats,
                sqlens_scan_cap,
            )
            cell = {
                    **provenance,
                    "expensive_sqlens_calibration_admitted": bool(
                        args.formal_calibration_suite
                        and dataset_name in FORMAL_SQLENS_EXTENSIONS
                        and family == "both_off"
                        and ef_search
                        in FORMAL_SQLENS_EXTENSIONS[dataset_name]
                    ),
                    "raw": str(raw),
                    "plan": str(plan),
                    "command": command,
                    "status": (
                        "complete"
                        if cell_complete(
                            raw,
                            plan,
                            int(provenance["expected_rows"]),
                            provenance,
                        )
                        else "pending"
                    ),
                }
            prior_isolation = prior_completed_cell_isolation(
                prior_manifest,
                raw,
                db_lock_path,
            )
            if prior_isolation is not None:
                cell["database_isolation"] = prior_isolation
            schedule.append(cell)
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "artifact_type": "sqlens_figure5_frontier_run",
            "status": "planned",
            "phase": args.phase,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "release_contract": {
                "path": config["release_contract_path"],
                "sha256": config["release_contract_sha256"],
                **config["release_identity"],
            },
            "protocol": config["protocol"],
            "workload_contracts": {
                dataset_name: next(
                    cell["workload_contract"]
                    for cell in schedule
                    if cell["dataset"] == dataset_name
                )
                for dataset_name in dataset_names
            },
            "search_grid": {
                "budgets": budgets,
                "scan_families": families,
                "formal_calibration_suite": args.formal_calibration_suite,
                "calibration_repeats": args.calibration_repeats,
                "allow_expensive_sqlens_calibration": (
                    args.allow_expensive_sqlens_calibration
                    or args.formal_calibration_suite
                ),
                "bounded_sqlens_extensions": (
                    FORMAL_SQLENS_EXTENSIONS
                    if args.formal_calibration_suite
                    else {}
                ),
                "sqlens_scan_caps": sorted(
                    {
                        cap
                        for _, family, _, cap in schedule_cells
                        if family == SQLENS_CAP_FAMILY and cap is not None
                    }
                ),
            },
            "schedule": schedule,
            "cells_total": len(schedule),
            "cells_complete": sum(cell["status"] == "complete" for cell in schedule),
            "requested_slice_complete": False,
            "full_calibration_suite_complete": False,
            "paper_eligible": False,
        }
        if args.require_global_db_lock:
            manifest["database_isolation"] = planned_global_db_isolation(
                db_lock_path
            )
        atomic_json(manifest_path, manifest)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "manifest": str(manifest_path),
                        "status": manifest["status"],
                        "cells_total": manifest["cells_total"],
                        "cells_complete": manifest["cells_complete"],
                        "formal_calibration_suite": (
                            args.formal_calibration_suite
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        global_lock: GlobalDBLock | None = None
        try:
            if args.require_global_db_lock:
                global_lock = acquire_global_db_lock(
                    db_lock_path,
                    "run_figure5_frontier",
                )
                manifest["database_isolation"] = global_lock.evidence(
                    held_through_completion=False
                )
                for cell in schedule:
                    if (
                        cell["status"] == "complete"
                        and args.resume
                        and not completed_isolation_evidence_valid(
                            cell.get("database_isolation"),
                            db_lock_path,
                        )
                    ):
                        raise Figure5ContractError(
                            "cannot resume a completed cell without valid global "
                            "DB isolation evidence; use --overwrite --no-resume"
                        )

            manifest["status"] = "running"
            atomic_json(manifest_path, manifest)
            for cell in schedule:
                if cell["status"] == "complete" and args.resume:
                    continue
                raw = Path(cell["raw"])
                if raw.exists() and not args.overwrite:
                    raise Figure5ContractError(
                        f"incomplete output exists; use --overwrite or inspect it: {raw}"
                    )
                if args.overwrite:
                    for path in (raw, Path(cell["plan"])):
                        if path.exists():
                            path.unlink()
                    if "design1_bloom_bfs_layout_d3" in cell["modes"]:
                        reset_records = [
                            {
                                "namespace": namespace,
                                "rows_deleted": clear_fragment_store_namespace(
                                    str(cell["d3_fragment_store_table"]),
                                    namespace,
                                ),
                            }
                            for namespace in fragment_store_namespaces(cell)
                        ]
                        cell["d3_namespace_reset_evidence"] = reset_records
                        cell["d3_namespace_rows_deleted"] = sum(
                            int(record["rows_deleted"])
                            for record in reset_records
                        )
                cell["status"] = "running"
                cell["started_at"] = utc_now()
                manifest["updated_at"] = utc_now()
                atomic_json(manifest_path, manifest)
                log = raw.with_suffix(raw.suffix + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w", encoding="utf-8") as output:
                    completed = subprocess.run(
                        cell["command"],
                        cwd=ROOT,
                        env=os.environ.copy(),
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                cell["returncode"] = completed.returncode
                cell["completed_at"] = utc_now()
                if completed.returncode != 0 or not cell_complete(
                    raw,
                    Path(cell["plan"]),
                    int(cell["expected_rows"]),
                    cell,
                ):
                    cell["status"] = "failed"
                    cell["log"] = str(log)
                    raise Figure5ContractError(
                        f"cell failed: {cell['dataset']}/{cell['scan_family']}/"
                        f"ef={cell['ef_search']}; see {log}"
                    )
                cell["status"] = "complete"
                cell["raw_sha256"] = sha256_file(raw)
                if global_lock is not None:
                    cell["database_isolation"] = global_lock.evidence(
                        held_through_completion=True
                    )
                manifest["cells_complete"] = sum(
                    item["status"] == "complete" for item in schedule
                )
                manifest["updated_at"] = utc_now()
                atomic_json(manifest_path, manifest)

            manifest["status"] = "complete"
            manifest["completed_at"] = utc_now()
            manifest["requested_slice_complete"] = True
            manifest["full_calibration_suite_complete"] = bool(
                args.formal_calibration_suite
            )
            manifest["paper_eligible"] = False
            manifest["paper_eligible_reason"] = (
                "raw calibration/latency cells require independent Figure 5 artifact audit"
            )
            if global_lock is not None:
                if not all(
                    completed_isolation_evidence_valid(
                        cell.get("database_isolation"),
                        db_lock_path,
                    )
                    for cell in schedule
                    if cell["status"] == "complete"
                ):
                    raise Figure5ContractError(
                        "completed cell is missing global DB isolation evidence"
                    )
                manifest["database_isolation"] = global_lock.evidence(
                    held_through_completion=True
                )
            atomic_json(manifest_path, manifest)
            print(f"wrote {manifest_path}", flush=True)
            return 0
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            manifest["updated_at"] = utc_now()
            if global_lock is not None:
                manifest["database_isolation"] = global_lock.evidence(
                    held_through_completion=True
                )
            atomic_json(manifest_path, manifest)
            raise
        finally:
            if global_lock is not None:
                global_lock.close()
    finally:
        lock.close()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable Figure 5 Stock-vs-full-SQLens latency frontier cells."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("calibration", "measurement"), default="calibration")
    parser.add_argument("--datasets", nargs="*", choices=("amazon", "yfcc", "laion"))
    parser.add_argument("--grid", choices=("base", "extension", "all"), default="base")
    parser.add_argument(
        "--ef-search-values",
        type=parse_int_values,
        default=[],
        help="Optional comma-separated override.",
    )
    parser.add_argument(
        "--scan-families",
        nargs="+",
        choices=SCAN_FAMILIES,
        default=list(STANDARD_SCAN_FAMILIES),
    )
    parser.add_argument(
        "--sqlens-scan-cap-values",
        type=parse_int_values,
        default=[],
        help=(
            "Comma-separated max_scan_tuples values for sqlens_cap "
            "calibration cells."
        ),
    )
    parser.add_argument("--backend-cpu-list", default="48-63")
    parser.add_argument(
        "--calibration-repeats",
        type=int,
        default=1,
        help=(
            "Calibration repeats per cell; use 3 for a pre-release canary. "
            "The formal 146-cell suite is fixed at 1."
        ),
    )
    parser.add_argument(
        "--formal-calibration-suite",
        action="store_true",
        help=(
            "Run the frozen 146-cell three-dataset calibration in one "
            "SHA-bound resumable manifest."
        ),
    )
    parser.add_argument(
        "--allow-expensive-sqlens-calibration",
        action="store_true",
        help=(
            "Explicitly admit calibration both_off cells above ef_search=1000; "
            "intended only for a bounded dataset-specific frontier extension."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=RESULTS / "figure5_r35")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--require-global-db-lock",
        action="store_true",
        help=(
            "Require the process-wide formal calibration DB lock before any "
            "namespace reset or database experiment."
        ),
    )
    parser.add_argument(
        "--global-db-lock-path",
        type=Path,
        help=(
            "Shared lock file used by all formal Figure 5 calibration "
            "orchestrators; requires --require-global-db-lock."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
