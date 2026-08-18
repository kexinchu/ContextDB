from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from . import external_dataset_matched_recall_runner as shared
except ImportError:
    import external_dataset_matched_recall_runner as shared


DatasetSpec = shared.DatasetSpec
ROOT = shared.ROOT

FORMAL_POLICY = "lcb_then_max_recall"
VECTOR_SO_SHA256 = re.compile(r"^[0-9a-f]{64}$")


SPEC = DatasetSpec(
    key="laion25m",
    display_name="LAION-25M",
    table="public.laion25m_pgvector",
    query_table="public.laion25m_queries",
    index="public.laion25m_pgvector_embedding_hnsw",
    guidance_meta_table="public.laion25m_pgvector_guidance_meta",
    query_id_column="qid",
    query_vector_column="embedding",
    filter_names=(
        "labelor_top70",
        "labelor_top55",
        "labelor_top40",
        "labelor_top30",
        "labelor_top20",
        "labelor_top14",
        "labelor_top9",
        "labelor_top6",
        "labelor_top3",
        "label_175",
        "label_79",
        "label_2039",
        "label_1432",
        "label_281",
    ),
    default_filters_csv=ROOT / "results/hybrid_vector_db/laion25m_matched_recall_filters_q180.csv",
    default_truth_csv=ROOT / "results/hybrid_vector_db/laion25m_matched_recall_exact_truth_q180.csv",
    truth_builder_command=(
        "env",
        "OOD_ANNS_DATA={ood_anns_data}",
        "{python}",
        "experiments/hybrid_vector_db/scripts/laion25m_exact_truth.py",
        "--formal-matched-recall",
        "--selected-queries-in",
        "results/hybrid_vector_db/laion25m_label_or_global_selected_q100_20260716.csv",
        "--formal-query-count",
        "180",
        "--calibration-queries",
        "80",
        "--filters-out",
        "{filters_csv}",
        "--truth-out",
        "{truth_csv}",
    ),
)


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _relation_name_matches(expected: str, observed: object) -> bool:
    observed_name = str(observed or "")
    return observed_name == expected or observed_name == expected.rsplit(".", 1)[-1]


def _audit_reuse_manifest(path: Path) -> list[str]:
    """Reject legacy mean-recall calibration before the generic reuse gate runs."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read reuse calibration manifest: {exc}"]
    run_args = ((payload.get("run_spec") or {}).get("args") or {})
    policy = payload.get("calibration_policy") or {}
    errors: list[str] = []
    if run_args.get("calibration_selection_policy") != FORMAL_POLICY:
        errors.append("reuse manifest was not calibrated with lcb_then_max_recall")
    if policy.get("calibration_selection_policy") != FORMAL_POLICY:
        errors.append("reuse manifest calibration policy is not lcb_then_max_recall")
    if policy.get("stop_metric") != "recall_lcb95":
        errors.append("reuse manifest does not bind LCB95 as its calibration stop metric")
    if policy.get("grid_policy") != shared.CALIBRATION_GRID_POLICY:
        errors.append("reuse manifest does not use the canonical staged grid policy")
    if (
        policy.get("base_grid_max_ef") != max(shared.BASE_EF_SEARCH_GRID)
        or policy.get("base_grid_complete_required") is not True
        or policy.get("extension_ef_search_values")
        != list(shared.HIGH_EF_SEARCH_EXTENSION)
        or policy.get("extension_trigger")
        != "max_target_lcb95_unmet_after_complete_base_grid"
        or policy.get("extension_complete_required_when_triggered") is not True
        or policy.get("early_stop_allowed") is not False
        or policy.get("grid_exhaustion_semantics")
        != "all_policy_required_configs_executed"
    ):
        errors.append("reuse manifest has incomplete canonical staged-grid semantics")
    stop_condition = str(policy.get("stop_condition") or "")
    if (
        "20--10000" not in stop_condition
        or "20000--100000" not in stop_condition
    ):
        errors.append("reuse manifest permits a legacy incomplete-grid stop")
    for pair in payload.get("calibration_pairs") or []:
        if (
            pair.get("calibration_grid_policy")
            != shared.CALIBRATION_GRID_POLICY
            or pair.get("grid_exhausted") is not True
            or pair.get("stopped_early") is not False
        ):
            errors.append("reuse manifest contains incomplete or early-stopped calibration")
            break
    return errors


def validate_formal_args(argv: list[str] | None = None) -> Any:
    """Constrain the generic executor to the preregistered LAION formal protocol."""
    args = shared.parser_for(SPEC).parse_args(argv)
    shared.bind_release_contract(args)
    errors: list[str] = []
    if args.calibration_selection_policy != FORMAL_POLICY:
        errors.append("formal runs require --calibration-selection-policy=lcb_then_max_recall")
    if not args.check_database and not args.dry_run:
        errors.append("formal runs may not disable database provenance/readiness checks")
    if args.reuse_calibration_manifest:
        errors.extend(_audit_reuse_manifest(args.reuse_calibration_manifest))
    if errors:
        raise SystemExit("; ".join(errors))
    return args


def _audit_provenance(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    run_spec = payload.get("run_spec") or {}
    query_contract = run_spec.get("query_contract") or {}
    database = run_spec.get("database") or {}
    runtime = run_spec.get("sqlens_runtime_provenance") or {}
    if len(run_spec.get("calibration_query_ids") or []) != 80:
        errors.append("run_spec does not bind exactly 80 calibration query IDs")
    final_ids = run_spec.get("final_query_ids") or []
    calibration_ids = run_spec.get("calibration_query_ids") or []
    if len(final_ids) != 100 or len(set(final_ids)) != 100:
        errors.append("run_spec does not bind 100 unique final query IDs")
    if set(final_ids).intersection(calibration_ids):
        errors.append("calibration and final query cohorts overlap")
    if query_contract.get("query_table") != SPEC.query_table:
        errors.append("run_spec query table differs from the LAION query relation")
    if query_contract.get("self_excluded") is not False:
        errors.append("run_spec query contract unexpectedly self-excludes external queries")
    if query_contract.get("candidate_validity_predicate") != "TRUE":
        errors.append("run_spec candidate-validity predicate is not TRUE")
    if not VECTOR_SO_SHA256.fullmatch(str(runtime.get("loaded_vector_so_sha256") or "")):
        errors.append("run_spec does not bind a valid loaded vector.so SHA-256")
    build_id = str(runtime.get("loaded_vector_sqlens_build_id") or "")
    if not build_id:
        errors.append("run_spec does not bind the loaded SQLens build ID")
    if database.get("sqlens_build_id") != build_id:
        errors.append("database/build-ID provenance is inconsistent")
    relations = database.get("relations") or {}
    for relation in (SPEC.table, SPEC.index):
        if not isinstance(relations.get(relation), dict):
            errors.append(f"database provenance omits {relation}")
    query_table = database.get("query_table") or {}
    if not _relation_name_matches(SPEC.query_table, query_table.get("name")):
        errors.append("database provenance omits the bound LAION query table")
    return errors


def audit_formal_manifest(path: Path, args: Any) -> dict[str, Any]:
    """Make a generic Stock/D1 slice publishable only when its strict contract holds."""
    base = shared.audit_generic_manifest(path, SPEC, args)
    errors = list(base["errors"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {**base, "formal_complete": False, "errors": errors + [str(exc)]}
    errors.extend(_audit_provenance(payload))
    outputs = payload.get("outputs") or {}
    for artifact_name in ("selected", "final"):
        artifact_path = Path(str((outputs.get(artifact_name) or {}).get("path") or ""))
        if not artifact_path.is_file():
            continue
        with artifact_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            target = float(row.get("target_recall") or 0.0)
            if row.get("calibration_selection_policy") != FORMAL_POLICY:
                errors.append(f"{artifact_name} contains a non-LCB selection row")
                break
            if not _bool(row.get("target_lcb95_met_in_calibration")):
                errors.append(f"{artifact_name} contains a target without calibration LCB95 confirmation")
                break
            if artifact_name == "final" and (
                not _bool(row.get("target_confirmed_in_final"))
                or int(row.get("expected_queries") or 0) < 100
                or int(row.get("expected_repeats") or 0) < 5
                or float(row.get("recall_mean") or 0.0) < target
            ):
                errors.append("final output contains an incomplete or unmatched-recall row")
                break
    return {
        **base,
        "formal_complete": bool(base["protocol_complete"] and not errors),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = validate_formal_args(argv)
    result = shared.run_dataset(SPEC, argv)
    if result != 0 or args.dry_run:
        return result
    launch_path = shared.RESULTS / f"{SPEC.key}_matched_recall_launch_{args.tag}.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    generic = (launch.get("generic_manifest") or {}).get("path")
    if not generic:
        return 2
    audit = audit_formal_manifest(Path(str(generic)), args)
    raw_valid = bool((launch.get("independent_raw_audit") or {}).get("overall_valid"))
    audit["independent_raw_audit_overall_valid"] = raw_valid
    audit["formal_complete"] = bool(audit["formal_complete"] and raw_valid)
    launch["formal_protocol_audit"] = audit
    launch["formal_complete"] = audit["formal_complete"]
    if not audit["formal_complete"]:
        launch["status"] = "incomplete"
    shared.write_json_atomic(launch_path, launch)
    return 0 if audit["formal_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
