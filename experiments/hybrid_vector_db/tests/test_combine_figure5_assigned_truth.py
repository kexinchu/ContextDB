import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


combiner = load_module(
    "combine_figure5_assigned_truth",
    ROOT / "experiments/hybrid_vector_db/scripts/combine_figure5_assigned_truth.py",
)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.filters = [f"f{i:02d}" for i in range(14)]
        self.filters_csv = self.root / "filters.csv"
        write_csv(
            self.filters_csv,
            ("filter_name", "predicate"),
            ({"filter_name": name, "predicate": f"labels && ARRAY[{index}]::int[]"} for index, name in enumerate(self.filters)),
        )
        self.assigned = self.root / "assigned.csv"
        calibration = []
        request_no = 0
        for query_no in range(200):
            for filter_name in self.filters:
                calibration.append({
                    "request_no": request_no, "query_no": query_no, "query_id": 10_000 + query_no,
                    "filter_name": filter_name, "trace_cycle": self.filters.index(filter_name), "split": "calibration",
                })
                request_no += 1
        measurement = []
        for query_no in range(200, 10_200):
            filter_name = self.filters[query_no % len(self.filters)]
            measurement.append({
                "request_no": request_no, "query_no": query_no, "query_id": 10_000 + query_no,
                "filter_name": filter_name, "trace_cycle": 0, "split": "measurement",
            })
            request_no += 1
        self.assigned_rows = calibration + measurement
        write_csv(self.assigned, combiner.WORKLOAD_FIELDS, self.assigned_rows)
        self.workload_manifest = self.root / "workload.json"
        write_json(self.workload_manifest, {
            "artifact_valid": False,
            "stage": "trace_pending_truth",
            "truth": {"contract": "pending_exact_truth_for_frozen_assigned_pairs_v1"},
            "construction": {"calibration": {"protocol": "formal_per_predicate_cartesian_v1"}},
            "formal_paper_calibration": {"passed": True},
            "outputs": {"assigned_workload_csv": {"rows": 12800, "sha256": combiner.sha256_file(self.assigned)}},
        })
        self.calibration_truth = self.root / "calibration_truth.csv"
        self.legacy_truth = self.root / "legacy_truth.csv"
        calibration_truth_rows = [self.truth_row(row, latency="") for row in calibration]
        legacy_rows = []
        for query_no in range(10_200):
            if query_no >= 200:
                workload = measurement[query_no - 200]
            else:
                workload = {"query_no": query_no, "query_id": 10_000 + query_no, "filter_name": self.filters[(query_no + 1) % 14]}
            legacy_rows.append(self.truth_row(workload, latency=""))
        write_csv(self.calibration_truth, combiner.TRUTH_FIELDS, calibration_truth_rows)
        write_csv(self.legacy_truth, combiner.TRUTH_FIELDS, legacy_rows)
        self.calibration_manifest = self.root / "calibration_truth.json"
        self.legacy_manifest = self.root / "legacy_truth.json"
        self.write_truth_manifest(self.calibration_manifest, self.calibration_truth, 2800)
        self.write_truth_manifest(self.legacy_manifest, self.legacy_truth, 10200)
        self.output = self.root / "merged.csv"
        self.output_manifest = self.root / "merged.json"

    def truth_row(self, workload, latency):
        index = self.filters.index(workload["filter_name"])
        return {
            "query_no": workload["query_no"], "query_id": workload["query_id"], "filter_name": workload["filter_name"],
            "predicate": f"labels && ARRAY[{index}]::int[]", "actual_selectivity": "0.1",
            "candidate_validity_predicate": "embedding_valid", "candidate_validity_provenance": "fixture",
            "query_validity_predicate": "true", "query_validity_provenance": "fixture",
            "method": "pre_filter_exact", "k": 10, "latency_ms": latency,
            "recall_at_10_exact_filtered": "1.0", "returned": 10, "candidates": 10, "filtered_rows": 10,
            "search_candidate_rows": 10, "result_ids": "1,2,3,4,5,6,7,8,9,10",
            "exact_filtered_topk_ids": "1,2,3,4,5,6,7,8,9,10",
            "exact_filtered_topk_distances_sq": "0,1,2,3,4,5,6,7,8,9",
            "kth_distance_sq": "9", "tie_tolerance": "0.000009",
            "strict_closer_count": 9, "boundary_tied": "false", "self_excluded": "false", "candidate_rows": 10,
            "self_excluded_rows": 0,
        }

    def write_truth_manifest(self, path, truth, rows):
        write_json(path, {
            "generator": "figure5_external_exact_truth.py", "k": 10,
            "output": {"sha256": combiner.sha256_file(truth), "rows": rows},
            "exact_coverage": {"complete": True, "emitted_rows": rows, "self_excluded": False,
                               "method": "full_base_scan_plus_cpu_float32_gemm_topk"},
        })

    def call(self, *, execute=False):
        return combiner.merge(
            assigned_workload_csv=self.assigned, workload_manifest=self.workload_manifest, filters_csv=self.filters_csv,
            calibration_truth_csv=self.calibration_truth, calibration_truth_manifest=self.calibration_manifest,
            legacy_truth_csv=self.legacy_truth, legacy_truth_manifest=self.legacy_manifest,
            output_truth_csv=self.output, output_manifest=self.output_manifest, execute=execute, overwrite=False,
        )


class CombineFigure5AssignedTruthTests(unittest.TestCase):
    def test_dry_run_does_not_write_and_execute_merges_new_calibration_with_reused_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            dry_run = fixture.call()
            self.assertFalse(dry_run["execution"]["execute"])
            self.assertFalse(fixture.output.exists())
            result = fixture.call(execute=True)
            self.assertTrue(fixture.output.exists())
            self.assertEqual(result["counts"], {"output_rows": 12800, "computed_rows": 2800, "reused_rows": 10000, "cross_source_identical_overlaps": 200})
            with fixture.output.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 12800)
            self.assertEqual(rows[0]["method"], "pre_filter_exact")
            self.assertEqual(rows[-1]["method"], "pre_filter_exact")
            published = json.loads(fixture.output_manifest.read_text(encoding="utf-8"))
            self.assertEqual(published["output"]["sha256"], combiner.sha256_file(fixture.output))

    def test_rejects_schema_drift_duplicate_and_non_exact_tie_aware_rows(self):
        mutations = ("schema", "duplicate", "not_tie_aware")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(tmp)
                if mutation == "schema":
                    write_csv(fixture.calibration_truth, combiner.TRUTH_FIELDS[:-1], [])
                else:
                    with fixture.calibration_truth.open(newline="", encoding="utf-8") as source:
                        rows = list(csv.DictReader(source))
                    if mutation == "duplicate":
                        rows.append(dict(rows[0]))
                    else:
                        rows[0]["boundary_tied"] = ""
                    write_csv(fixture.calibration_truth, combiner.TRUTH_FIELDS, rows)
                fixture.write_truth_manifest(fixture.calibration_manifest, fixture.calibration_truth, 2800)
                with self.assertRaises(combiner.TruthMergeError):
                    fixture.call()

    def test_rejects_predicate_mismatch_measurement_missing_extra_and_cross_source_conflict(self):
        cases = ("predicate", "measurement_key", "cross_source")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(tmp)
                with fixture.legacy_truth.open(newline="", encoding="utf-8") as source:
                    rows = list(csv.DictReader(source))
                if case == "predicate":
                    rows[200]["predicate"] = "labels && ARRAY[999]::int[]"
                elif case == "measurement_key":
                    rows[200]["filter_name"] = fixture.filters[(fixture.filters.index(rows[200]["filter_name"]) + 1) % 14]
                    rows[200]["predicate"] = f"labels && ARRAY[{fixture.filters.index(rows[200]['filter_name'])}]::int[]"
                else:
                    # Make an exact old calibration key collide with new truth but change a semantic tie field.
                    rows[0] = fixture.truth_row(fixture.assigned_rows[0], latency="")
                    rows[0]["kth_distance_sq"] = "1.0"
                write_csv(fixture.legacy_truth, combiner.TRUTH_FIELDS, rows)
                fixture.write_truth_manifest(fixture.legacy_manifest, fixture.legacy_truth, 10200)
                with self.assertRaises(combiner.TruthMergeError):
                    fixture.call()

    def test_rejects_unmatched_calibration_extra_and_manifest_digest_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            with fixture.calibration_truth.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows[0]["filter_name"] = fixture.filters[1]
            rows[0]["predicate"] = "labels && ARRAY[1]::int[]"
            write_csv(fixture.calibration_truth, combiner.TRUTH_FIELDS, rows)
            fixture.write_truth_manifest(fixture.calibration_manifest, fixture.calibration_truth, 2800)
            with self.assertRaises(combiner.TruthMergeError):
                fixture.call()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            manifest = json.loads(fixture.legacy_manifest.read_text(encoding="utf-8"))
            manifest["output"]["sha256"] = hashlib.sha256(b"wrong").hexdigest()
            write_json(fixture.legacy_manifest, manifest)
            with self.assertRaises(combiner.TruthMergeError):
                fixture.call()

    def test_accepts_tied_id_order_and_float32_backend_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            with fixture.calibration_truth.open(
                newline="", encoding="utf-8"
            ) as source:
                calibration_rows = list(csv.DictReader(source))
            calibration_rows[0]["exact_filtered_topk_distances_sq"] = (
                "0,1,2,3,4,5,6,7,9,9"
            )
            calibration_rows[0]["tie_tolerance"] = "0.00001"
            calibration_rows[0]["strict_closer_count"] = "8"
            calibration_rows[0]["boundary_tied"] = "true"
            write_csv(
                fixture.calibration_truth,
                combiner.TRUTH_FIELDS,
                calibration_rows,
            )
            fixture.write_truth_manifest(
                fixture.calibration_manifest,
                fixture.calibration_truth,
                2800,
            )
            with fixture.legacy_truth.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows[0] = fixture.truth_row(fixture.assigned_rows[0], latency="")
            rows[0]["result_ids"] = "1,2,3,4,5,6,7,8,10,11"
            rows[0]["exact_filtered_topk_ids"] = "1,2,3,4,5,6,7,8,10,11"
            rows[0]["exact_filtered_topk_distances_sq"] = (
                "0,1,2,3,4,5,6,7,8.999999,9.000001"
            )
            rows[0]["tie_tolerance"] = "0.00001"
            rows[0]["kth_distance_sq"] = "9.000001"
            rows[0]["strict_closer_count"] = "8"
            rows[0]["boundary_tied"] = "true"
            write_csv(fixture.legacy_truth, combiner.TRUTH_FIELDS, rows)
            fixture.write_truth_manifest(
                fixture.legacy_manifest, fixture.legacy_truth, 10200
            )
            result = fixture.call()
            self.assertEqual(result["counts"]["cross_source_identical_overlaps"], 200)


if __name__ == "__main__":
    unittest.main()
