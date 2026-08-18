import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.hybrid_vector_db.scripts.external_dataset_matched_recall_runner import (
    BASE_EF_SEARCH_GRID,
    CALIBRATION_GRID_POLICY,
    CALIBRATION_QUERIES,
    CALIBRATION_REPEATS,
    EF_SEARCH_GRID,
    FINAL_QUERIES,
    FINAL_REPEATS,
    FORMAL_SQLENS_BUILD_PREFIX,
    ITERATIVE_SCAN_VALUES,
    MODES,
    TARGET_RECALLS,
    DEFAULT_P0_RELEASE_CONTRACT,
    audit_generic_manifest,
    audit_truth,
    build_target_command,
    formal_sqlens_build_compatible,
    load_and_audit_filters,
    parser_for,
    record_launch_failure,
    run_independent_raw_audit,
    sha256_file,
)
from experiments.hybrid_vector_db.scripts.laion_pgvector_target_recall_runner import (
    SPEC as LAION_SPEC,
)
from experiments.hybrid_vector_db.scripts.yfcc_pgvector_target_recall_runner import (
    SPEC as YFCC_SPEC,
)
from experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner import (
    FORMAL_CALIBRATION_GRID_POLICY as CORE_CALIBRATION_GRID_POLICY,
    load_p0_release_contract,
)


def argument_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def runner_args(filters_csv: Path, truth_csv: Path) -> argparse.Namespace:
    contract = load_p0_release_contract(DEFAULT_P0_RELEASE_CONTRACT)
    return argparse.Namespace(
        tag="unit",
        filters_csv=filters_csv,
        truth_csv=truth_csv,
        calibration_recall_margin=0.0,
        schedule_seed=20260718,
        statement_timeout_ms=300000,
        progress_queries=10,
        bootstrap_samples=10000,
        bootstrap_seed=20260718,
        guidance_max_atoms=128,
        backend_cpu_list=None,
        prewarm_index_health=True,
        expected_sqlens_build_id=contract["expected_sqlens_build_id"],
        expected_vector_so_sha256=contract["expected_vector_so_sha256"],
        release_contract=DEFAULT_P0_RELEASE_CONTRACT,
        release_contract_provenance=contract,
        resume=True,
    )


def write_filters(path: Path) -> list[str]:
    names = [f"filter_{number:02d}" for number in range(14)]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "filter_name",
                "target_rate",
                "actual_pct",
                "expected_rows",
                "predicate",
                "atoms",
            ),
        )
        writer.writeheader()
        for number, name in enumerate(names):
            writer.writerow(
                {
                    "filter_name": name,
                    "target_rate": 50 - number,
                    "actual_pct": 50 - number,
                    "expected_rows": 1000,
                    "predicate": f"tags && ARRAY[{number}]::int[]",
                    "atoms": f"sql:tags @> ARRAY[{number}]::int[]",
                }
            )
    return names


def write_truth(path: Path, names: list[str], *, drop_last: bool = False) -> None:
    rows = []
    for query_no in range(180):
        for name in names:
            rows.append(
                {
                    "query_no": query_no,
                    "query_id": 10000 + query_no,
                    "query_split": "calibration" if query_no < 80 else "final",
                    "filter_name": name,
                    "predicate": f"tags && ARRAY[{int(name[-2:])}]::int[]",
                    "candidate_validity_predicate": "TRUE",
                    "method": "pre_filter_exact",
                    "filtered_rows": 1000,
                    "kth_distance_sq": 1.0,
                    "tie_tolerance": 1e-6,
                    "self_excluded": False,
                }
            )
    if drop_last:
        rows.pop()
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ExternalDatasetMatchedRecallRunnerTests(unittest.TestCase):
    def test_laion_filters_override_is_propagated_losslessly(self):
        override = Path("results/hybrid_vector_db/custom filters/ordered atoms.csv")
        contract = load_p0_release_contract(DEFAULT_P0_RELEASE_CONTRACT)
        args = parser_for(LAION_SPEC).parse_args(
            [
                "--tag",
                "ordered-atoms",
                "--filters-csv",
                str(override),
                "--expected-sqlens-build-id",
                str(contract["expected_sqlens_build_id"]),
                "--expected-vector-so-sha256",
                str(contract["expected_vector_so_sha256"]),
            ]
        )

        command = build_target_command(LAION_SPEC, args, list(LAION_SPEC.filter_names))

        self.assertEqual(args.filters_csv, override)
        self.assertNotEqual(args.filters_csv, LAION_SPEC.default_filters_csv)
        self.assertEqual(argument_value(command, "--filters-csv"), str(override))
        self.assertEqual(command.count("--filters-csv"), 1)
        self.assertIn("--no-traversal-guided-prioritization", command)
        self.assertNotIn("--traversal-guided-prioritization", command)
        self.assertEqual(
            argument_value(command, "--release-contract"),
            str(DEFAULT_P0_RELEASE_CONTRACT),
        )

    def test_independent_raw_audit_is_hashed_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            filters = root / "filters.csv"
            truth = root / "truth.csv"
            filters.write_text("filters\n", encoding="utf-8")
            truth.write_text("truth\n", encoding="utf-8")
            args = runner_args(filters, truth)
            audit_path = root / "raw-audit.json"

            def fake_run(command, **_kwargs):
                Path(command[command.index("--json") + 1]).write_text(
                    json.dumps({"overall_valid": True}), encoding="utf-8"
                )
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with mock.patch(
                "experiments.hybrid_vector_db.scripts.external_dataset_matched_recall_runner.subprocess.run",
                side_effect=fake_run,
            ):
                report = run_independent_raw_audit(manifest, args, audit_path=audit_path)

            self.assertTrue(report["overall_valid"])
            self.assertEqual(report["sha256"], sha256_file(audit_path))
            self.assertEqual(report["audit"]["overall_valid"], True)

    def test_target_command_rejects_a_noncontract_runtime_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = runner_args(root / "filters.csv", root / "truth.csv")
            args.expected_vector_so_sha256 = "a" * 64
            with self.assertRaisesRegex(ValueError, "immutable P0 release contract"):
                build_target_command(YFCC_SPEC, args, list(YFCC_SPEC.filter_names))

    def test_interrupted_launch_is_not_left_running(self):
        payload = {"status": "running", "completed_at": None}

        record_launch_failure(payload, KeyboardInterrupt())

        self.assertEqual(payload["status"], "interrupted")
        self.assertEqual(payload["target_runner_returncode"], 130)
        self.assertEqual(payload["error"]["type"], "KeyboardInterrupt")
        self.assertIsNotNone(payload["completed_at"])

    def test_overlap_filter_requires_explicit_or_atom_composition(self):
        with tempfile.TemporaryDirectory() as temporary:
            filters = Path(temporary) / "filters.csv"
            names = write_filters(filters)
            rows = list(csv.DictReader(filters.open(newline="", encoding="utf-8")))
            rows[0]["predicate"] = "tags && ARRAY[1,2]::int[]"
            rows[0]["atoms"] = (
                "sql:tags @> ARRAY[1]::int[]||sql:tags @> ARRAY[2]::int[]"
            )
            with filters.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            observed_names, _, errors = load_and_audit_filters(filters)
            self.assertEqual(observed_names, names)
            self.assertTrue(any("atom||OR||atom" in error for error in errors))

            rows[0]["atoms"] = (
                "sql:tags @> ARRAY[1]::int[]||OR||sql:tags @> ARRAY[2]::int[]"
            )
            with filters.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            _, _, errors = load_and_audit_filters(filters)
            self.assertEqual(errors, [])

    def test_formal_grid_starts_at_20_and_keeps_original_high_budget_points(self):
        self.assertEqual(CALIBRATION_GRID_POLICY, CORE_CALIBRATION_GRID_POLICY)
        self.assertEqual(
            FORMAL_SQLENS_BUILD_PREFIX,
            "sqlens-v16-d3-representation-preserving-exact-d2-edge-trace-",
        )
        self.assertTrue(
            formal_sqlens_build_compatible(
                FORMAL_SQLENS_BUILD_PREFIX
                + "readbuffer-profile-ef500k-20260727-r33"
            )
        )
        self.assertFalse(
            formal_sqlens_build_compatible("sqlens-v16-d3-event-timers-final-trace-r27")
        )
        self.assertEqual(EF_SEARCH_GRID[:7], (20, 40, 60, 80, 100, 150, 200))
        self.assertEqual(
            EF_SEARCH_GRID[7:],
            (
                250,
                500,
                750,
                1000,
                1500,
                2000,
                3000,
                4000,
                5000,
                7000,
                8500,
                10000,
                20000,
                50000,
                100000,
            ),
        )

    def test_yfcc_and_laion_commands_use_identical_independent_tuning_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            filters = root / "filters.csv"
            truth = root / "truth.csv"
            names = write_filters(filters)
            args = runner_args(filters, truth)
            commands = [
                build_target_command(spec, args, names)
                for spec in (YFCC_SPEC, LAION_SPEC)
            ]
        for command in commands:
            self.assertEqual(argument_value(command, "--target-recalls"), "0.90,0.95,0.99")
            self.assertEqual(
                argument_value(command, "--ef-search-values"),
                ",".join(str(value) for value in EF_SEARCH_GRID),
            )
            self.assertEqual(argument_value(command, "--calibration-queries"), "80")
            self.assertEqual(argument_value(command, "--calibration-repeats"), "2")
            self.assertEqual(argument_value(command, "--calibration-query-offset"), "0")
            self.assertEqual(argument_value(command, "--final-queries"), "100")
            self.assertEqual(argument_value(command, "--final-repeats"), "5")
            self.assertEqual(argument_value(command, "--final-query-offset"), "80")
            self.assertEqual(argument_value(command, "--iterative-scan-values"), ITERATIVE_SCAN_VALUES)
            self.assertEqual(argument_value(command, "--stock-iterative-scan-values"), ITERATIVE_SCAN_VALUES)
            self.assertIn("--no-expected-truth-self-excluded", command)
            self.assertEqual(argument_value(command, "--candidate-validity-predicate"), "TRUE")
            self.assertEqual(
                argument_value(command, "--expected-sqlens-build-id"),
                args.expected_sqlens_build_id,
            )
            self.assertEqual(
                argument_value(command, "--expected-vector-so-sha256"),
                args.expected_vector_so_sha256,
            )
            self.assertIn("--prewarm-index-health", command)
            mode_start = command.index("--modes") + 1
            self.assertEqual(tuple(command[mode_start : mode_start + len(MODES)]), MODES)
        self.assertEqual(TARGET_RECALLS, (0.90, 0.95, 0.99))
        self.assertEqual((CALIBRATION_QUERIES, CALIBRATION_REPEATS), (80, 2))
        self.assertEqual((FINAL_QUERIES, FINAL_REPEATS), (100, 5))

    def test_truth_audit_requires_complete_disjoint_q80_q100_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            filters_path = root / "filters.csv"
            truth_path = root / "truth.csv"
            names = write_filters(filters_path)
            write_truth(truth_path, names)
            loaded_names, filters_by_name, filter_errors = load_and_audit_filters(filters_path)
            self.assertFalse(filter_errors)
            audit = audit_truth(truth_path, loaded_names, filters_by_name)
            self.assertTrue(audit["ready"])
            self.assertEqual(audit["row_count"], 14 * 180)
            self.assertEqual(audit["query_count"], 180)

            write_truth(truth_path, names, drop_last=True)
            incomplete = audit_truth(truth_path, loaded_names, filters_by_name)
            self.assertFalse(incomplete["ready"])
            self.assertTrue(any("missing 1" in error for error in incomplete["errors"]))

    def test_wrapper_manifest_requires_independent_target_method_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            filters = root / "filters.csv"
            truth = root / "truth.csv"
            calibration = root / "calibration.csv"
            selected = root / "selected.csv"
            final = root / "final.csv"
            manifest = root / "manifest.json"
            filters.write_text("formal filters\n", encoding="utf-8")
            truth.write_text("formal truth\n", encoding="utf-8")
            calibration.write_text("calibration\n", encoding="utf-8")
            matrix_rows = [
                {
                    "filter_name": filter_name,
                    "target_recall": target,
                    "mode": mode,
                }
                for filter_name in YFCC_SPEC.filter_names
                for target in TARGET_RECALLS
                for mode in MODES
            ]
            for path in (selected, final):
                with path.open("w", newline="", encoding="utf-8") as target:
                    writer = csv.DictWriter(target, fieldnames=list(matrix_rows[0]))
                    writer.writeheader()
                    writer.writerows(matrix_rows)
            args = runner_args(filters, truth)
            expected_args = {
                "calibration_queries": 80,
                "calibration_repeats": 2,
                "calibration_query_offset": 0,
                "final_queries": 100,
                "final_repeats": 5,
                "final_query_offset": 80,
                "final_execution_order": "interleaved",
                "calibration_selection_policy": "lcb_then_max_recall",
                "candidate_validity_predicate": "TRUE",
                "expected_truth_self_excluded": False,
                "insertion_table": YFCC_SPEC.table,
                "insertion_index": YFCC_SPEC.index,
                "query_table": YFCC_SPEC.query_table,
                "filters_csv": str(filters),
                "truth_csv": str(truth),
                "iterative_scan_values": ITERATIVE_SCAN_VALUES,
                "stock_iterative_scan_values": ITERATIVE_SCAN_VALUES,
                "ef_search_values": ",".join(str(value) for value in EF_SEARCH_GRID),
                "target_recalls": "0.90,0.95,0.99",
                "traversal_guided_prioritization": False,
                "prewarm_index_health": True,
                "expected_sqlens_build_id": args.expected_sqlens_build_id,
                "expected_vector_so_sha256": args.expected_vector_so_sha256,
            }
            mode_grid = [
                {"ef_search": ef_search, "iterative_scan": iterative_scan}
                for ef_search in EF_SEARCH_GRID
                for iterative_scan in ITERATIVE_SCAN_VALUES.split(",")
            ]
            outputs = {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in (
                    ("calibration", calibration),
                    ("selected", selected),
                    ("final", final),
                )
            }
            manifest.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "matrix_complete": True,
                        "measurement_complete": True,
                        "comparison_valid": True,
                        "requested_slice_complete": True,
                        "formal_release_complete": False,
                        "expected_cells": 84,
                        "targets": list(TARGET_RECALLS),
                        "modes": list(MODES),
                        "run_spec": {
                            "args": expected_args,
                            "filters_sha256": sha256_file(filters),
                            "truth_sha256": sha256_file(truth),
                            "sqlens_runtime_provenance": {
                                "loaded_vector_sqlens_build_id": args.expected_sqlens_build_id,
                                "loaded_vector_so_sha256": args.expected_vector_so_sha256,
                            },
                            "runtime_identity_binding": {
                                "expected_build_id": args.expected_sqlens_build_id,
                                "expected_vector_so_sha256": args.expected_vector_so_sha256,
                                "exact_match": True,
                            },
                            "p0_release_contract": args.release_contract_provenance,
                            "index_query_health": {
                                "indexes": [
                                    {
                                        "prewarm": {
                                            "enabled": True,
                                            "blocks": 10,
                                            "elapsed_ms": 1.0,
                                        }
                                    }
                                ]
                            },
                        },
                        "calibration_policy": {
                            "calibration_selection_policy": "lcb_then_max_recall",
                            "selection": "lowest latency among configurations whose LCB95 reaches the target",
                            "stop_metric": "recall_lcb95",
                            "grid_policy": CALIBRATION_GRID_POLICY,
                            "base_grid_max_ef": max(BASE_EF_SEARCH_GRID),
                            "base_grid_complete_required": True,
                            "extension_ef_search_values": list(
                                EF_SEARCH_GRID[len(BASE_EF_SEARCH_GRID) :]
                            ),
                            "extension_trigger": (
                                "max_target_lcb95_unmet_after_complete_base_grid"
                            ),
                            "extension_complete_required_when_triggered": True,
                            "early_stop_allowed": False,
                            "grid_exhaustion_semantics": (
                                "all_policy_required_configs_executed"
                            ),
                            "stop_condition": (
                                "complete 20--10000 base grid; run 20000--100000 only when "
                                "the maximum target LCB remains unmet; early stops are forbidden"
                            ),
                        },
                        "mode_grids": {mode: mode_grid for mode in MODES},
                        "calibration_pairs": [
                            {
                                "filter_name": filter_name,
                                "mode": mode,
                                "calibration_grid_policy": CALIBRATION_GRID_POLICY,
                                "grid_exhausted": True,
                                "stopped_early": False,
                                "families": {
                                    iterative_scan: {
                                        "configs_planned": len(BASE_EF_SEARCH_GRID),
                                        "configs_executed": len(BASE_EF_SEARCH_GRID),
                                        "grid_exhausted": True,
                                        "high_extension_required": False,
                                        "high_extension_executed": False,
                                        "high_extension_skip_reason": (
                                            "max_target_lcb_met_on_complete_base_grid"
                                        ),
                                        "max_ef_evaluated": max(BASE_EF_SEARCH_GRID),
                                    }
                                    for iterative_scan in ITERATIVE_SCAN_VALUES.split(",")
                                },
                            }
                            for filter_name in YFCC_SPEC.filter_names
                            for mode in MODES
                        ],
                        "outputs": outputs,
                    }
                ),
                encoding="utf-8",
            )
            audit = audit_generic_manifest(manifest, YFCC_SPEC, args)
            self.assertTrue(audit["protocol_complete"])
            args.expected_sqlens_build_id = None
            args.expected_vector_so_sha256 = None
            rebound = audit_generic_manifest(manifest, YFCC_SPEC, args)
            self.assertTrue(rebound["protocol_complete"])
            args.expected_sqlens_build_id = expected_args[
                "expected_sqlens_build_id"
            ]
            args.expected_vector_so_sha256 = expected_args[
                "expected_vector_so_sha256"
            ]

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["run_spec"]["args"]["filters_csv"] = "default-filters.csv"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            wrong_filters_path = audit_generic_manifest(manifest, YFCC_SPEC, args)
            self.assertFalse(wrong_filters_path["protocol_complete"])
            self.assertTrue(
                any("filters_csv" in error for error in wrong_filters_path["errors"])
            )

            payload["run_spec"]["args"]["filters_csv"] = str(filters)
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with final.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=list(matrix_rows[0]))
                writer.writeheader()
                writer.writerows(matrix_rows[:-1])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["outputs"]["final"]["sha256"] = sha256_file(final)
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            incomplete = audit_generic_manifest(manifest, YFCC_SPEC, args)
            self.assertFalse(incomplete["protocol_complete"])
            self.assertTrue(any("independent" in error for error in incomplete["errors"]))

    def test_wrapper_manifest_requires_target_gated_extension_when_base_misses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            filters = root / "filters.csv"
            truth = root / "truth.csv"
            calibration = root / "calibration.csv"
            selected = root / "selected.csv"
            final = root / "final.csv"
            manifest = root / "manifest.json"
            for path, contents in (
                (filters, "formal filters\n"),
                (truth, "formal truth\n"),
                (calibration, "calibration\n"),
            ):
                path.write_text(contents, encoding="utf-8")
            matrix_rows = [
                {"filter_name": filter_name, "target_recall": target, "mode": mode}
                for filter_name in YFCC_SPEC.filter_names
                for target in TARGET_RECALLS
                for mode in MODES
            ]
            for path in (selected, final):
                with path.open("w", newline="", encoding="utf-8") as target:
                    writer = csv.DictWriter(target, fieldnames=list(matrix_rows[0]))
                    writer.writeheader()
                    writer.writerows(matrix_rows)
            args = runner_args(filters, truth)
            mode_grid = [
                {"ef_search": ef_search, "iterative_scan": iterative_scan}
                for ef_search in EF_SEARCH_GRID
                for iterative_scan in ITERATIVE_SCAN_VALUES.split(",")
            ]
            expected_args = {
                "calibration_queries": 80,
                "calibration_repeats": 2,
                "calibration_query_offset": 0,
                "final_queries": 100,
                "final_repeats": 5,
                "final_query_offset": 80,
                "final_execution_order": "interleaved",
                "calibration_selection_policy": "lcb_then_max_recall",
                "candidate_validity_predicate": "TRUE",
                "expected_truth_self_excluded": False,
                "insertion_table": YFCC_SPEC.table,
                "insertion_index": YFCC_SPEC.index,
                "query_table": YFCC_SPEC.query_table,
                "filters_csv": str(filters),
                "truth_csv": str(truth),
                "iterative_scan_values": ITERATIVE_SCAN_VALUES,
                "stock_iterative_scan_values": ITERATIVE_SCAN_VALUES,
                "ef_search_values": ",".join(str(value) for value in EF_SEARCH_GRID),
                "target_recalls": "0.90,0.95,0.99",
                "traversal_guided_prioritization": False,
                "prewarm_index_health": True,
                "expected_sqlens_build_id": args.expected_sqlens_build_id,
                "expected_vector_so_sha256": args.expected_vector_so_sha256,
            }
            outputs = {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in (
                    ("calibration", calibration),
                    ("selected", selected),
                    ("final", final),
                )
            }
            extension_families = {
                iterative_scan: {
                    "configs_planned": len(EF_SEARCH_GRID),
                    "configs_executed": len(EF_SEARCH_GRID),
                    "grid_exhausted": True,
                    "high_extension_required": True,
                    "high_extension_executed": True,
                    "high_extension_skip_reason": None,
                    "max_ef_evaluated": max(EF_SEARCH_GRID),
                }
                for iterative_scan in ITERATIVE_SCAN_VALUES.split(",")
            }
            payload = {
                "status": "complete",
                "requested_slice_complete": True,
                "matrix_complete": True,
                "measurement_complete": True,
                "comparison_valid": True,
                "formal_release_complete": False,
                "expected_cells": 84,
                "targets": list(TARGET_RECALLS),
                "modes": list(MODES),
                "run_spec": {
                    "args": expected_args,
                    "filters_sha256": sha256_file(filters),
                    "truth_sha256": sha256_file(truth),
                    "sqlens_runtime_provenance": {
                        "loaded_vector_sqlens_build_id": args.expected_sqlens_build_id,
                        "loaded_vector_so_sha256": args.expected_vector_so_sha256,
                    },
                    "runtime_identity_binding": {
                        "expected_build_id": args.expected_sqlens_build_id,
                        "expected_vector_so_sha256": args.expected_vector_so_sha256,
                        "exact_match": True,
                    },
                    "p0_release_contract": args.release_contract_provenance,
                    "index_query_health": {
                        "indexes": [
                            {
                                "prewarm": {
                                    "enabled": True,
                                    "blocks": 10,
                                    "elapsed_ms": 1.0,
                                }
                            }
                        ]
                    },
                },
                "calibration_policy": {
                    "calibration_selection_policy": "lcb_then_max_recall",
                    "selection": "lowest latency among LCB-qualified configurations",
                    "stop_metric": "recall_lcb95",
                    "grid_policy": CALIBRATION_GRID_POLICY,
                    "base_grid_max_ef": max(BASE_EF_SEARCH_GRID),
                    "base_grid_complete_required": True,
                    "extension_ef_search_values": list(
                        EF_SEARCH_GRID[len(BASE_EF_SEARCH_GRID) :]
                    ),
                    "extension_trigger": (
                        "max_target_lcb95_unmet_after_complete_base_grid"
                    ),
                    "extension_complete_required_when_triggered": True,
                    "early_stop_allowed": False,
                    "grid_exhaustion_semantics": (
                        "all_policy_required_configs_executed"
                    ),
                    "stop_condition": (
                        "complete 20--10000 base grid; run 20000--100000 only when "
                        "the maximum target LCB remains unmet; early stops are forbidden"
                    ),
                },
                "mode_grids": {mode: mode_grid for mode in MODES},
                "calibration_pairs": [
                    {
                        "filter_name": filter_name,
                        "mode": mode,
                        "calibration_grid_policy": CALIBRATION_GRID_POLICY,
                        "grid_exhausted": True,
                        "stopped_early": False,
                        "families": extension_families,
                    }
                    for filter_name in YFCC_SPEC.filter_names
                    for mode in MODES
                ],
                "outputs": outputs,
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(audit_generic_manifest(manifest, YFCC_SPEC, args)["protocol_complete"])

            payload["calibration_pairs"][0]["families"]["off"][
                "high_extension_executed"
            ] = False
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            audit = audit_generic_manifest(manifest, YFCC_SPEC, args)
            self.assertFalse(audit["protocol_complete"])
            self.assertTrue(any("skipped a required extension" in error for error in audit["errors"]))


if __name__ == "__main__":
    unittest.main()
