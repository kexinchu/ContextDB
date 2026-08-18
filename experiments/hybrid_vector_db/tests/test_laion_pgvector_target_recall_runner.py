import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.hybrid_vector_db.scripts import laion_pgvector_target_recall_runner as runner


class LaionFormalRunnerTests(unittest.TestCase):
    CONTRACT = runner.shared.load_p0_release_contract(
        runner.shared.DEFAULT_P0_RELEASE_CONTRACT
    )
    RUNTIME_ARGS = [
        "--expected-sqlens-build-id",
        CONTRACT["expected_sqlens_build_id"],
        "--expected-vector-so-sha256",
        CONTRACT["expected_vector_so_sha256"],
    ]

    def test_protocol_keeps_low_ef_prefix_and_q100_r5_final(self):
        self.assertEqual(runner.shared.EF_SEARCH_GRID[:7], (20, 40, 60, 80, 100, 150, 200))
        self.assertEqual(runner.shared.TARGET_RECALLS, (0.90, 0.95, 0.99))
        self.assertEqual((runner.shared.FINAL_QUERIES, runner.shared.FINAL_REPEATS), (100, 5))
        args = runner.validate_formal_args(
            ["--tag", "unit", "--dry-run", *self.RUNTIME_ARGS]
        )
        command = runner.shared.build_target_command(
            runner.SPEC, args, list(runner.SPEC.filter_names)
        )
        self.assertEqual(
            command[command.index("--ef-search-values") + 1],
            ",".join(str(value) for value in runner.shared.EF_SEARCH_GRID),
        )
        self.assertEqual(
            command[command.index("--calibration-selection-policy") + 1],
            runner.FORMAL_POLICY,
        )
        self.assertIn("--prewarm-index-health", command)

    def test_rejects_mean_latency_policy_and_legacy_reuse(self):
        with self.assertRaises(SystemExit):
            runner.validate_formal_args(
                ["--tag", "unit", "--dry-run", "--calibration-selection-policy", "mean_latency"]
            )
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "legacy.json"
            manifest.write_text(
                json.dumps(
                    {
                        "run_spec": {
                            "args": {
                                "calibration_selection_policy": runner.FORMAL_POLICY
                            }
                        },
                        "calibration_policy": {
                            "calibration_selection_policy": runner.FORMAL_POLICY,
                            "stop_metric": "recall_lcb95",
                            "grid_policy": (
                                "ascending_prefix_first_max_target_lcb_or_latency_dominated"
                            ),
                            "stop_condition": "stop at first max-target LCB crossing",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                runner.validate_formal_args(
                    ["--tag", "unit", "--dry-run", "--reuse-calibration-manifest", str(manifest)]
                )

    def test_manifest_audit_fails_closed_on_missing_runtime_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "selected.csv"
            final = root / "final.csv"
            manifest = root / "manifest.json"
            row = {
                "target_recall": "0.99",
                "calibration_selection_policy": runner.FORMAL_POLICY,
                "target_lcb95_met_in_calibration": "true",
                "target_confirmed_in_final": "true",
                "expected_queries": "100",
                "expected_repeats": "5",
                "recall_mean": "0.99",
            }
            for path in (selected, final):
                with path.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
            payload = self._payload(selected, final)
            payload["run_spec"]["sqlens_runtime_provenance"]["loaded_vector_so_sha256"] = ""
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            args = runner.shared.parser_for(runner.SPEC).parse_args(["--tag", "unit"])
            with mock.patch.object(
                runner.shared,
                "audit_generic_manifest",
                return_value={"errors": [], "protocol_complete": True},
            ):
                audit = runner.audit_formal_manifest(manifest, args)
            self.assertFalse(audit["formal_complete"])
            self.assertTrue(any("vector.so" in error for error in audit["errors"]))

    @staticmethod
    def _payload(selected: Path, final: Path):
        spec = runner.SPEC
        return {
            "run_spec": {
                "calibration_query_ids": list(range(80)),
                "final_query_ids": list(range(80, 180)),
                "query_contract": {
                    "query_table": spec.query_table,
                    "self_excluded": False,
                    "candidate_validity_predicate": "TRUE",
                },
                "sqlens_runtime_provenance": {
                    "loaded_vector_so_sha256": "a" * 64,
                    "loaded_vector_sqlens_build_id": "sqlens-v15-test",
                },
                "database": {
                    "sqlens_build_id": "sqlens-v15-test",
                    "relations": {spec.table: {}, spec.index: {}},
                    "query_table": {"name": spec.query_table.rsplit(".", 1)[-1]},
                },
            },
            "outputs": {"selected": {"path": str(selected)}, "final": {"path": str(final)}},
        }


if __name__ == "__main__":
    unittest.main()
