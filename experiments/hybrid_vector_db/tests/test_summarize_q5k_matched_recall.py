from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.hybrid_vector_db.scripts import summarize_q5k_matched_recall as subject


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class SummarizeQ5kMatchedRecallTests(unittest.TestCase):
    def make_shard(self, root: Path, filter_name: str, count: int = 2) -> Path:
        path = root / f"measurement_{filter_name}.csv"
        fields = [
            "selectivity", "filter_name", "mode", "query_no", "query_id",
            "repeat", "pair_key", "request_no", "trace_cycle", "recall",
            "end_to_end_ms", "effective_ef_search", "max_scan_tuples",
            "scan_mem_multiplier", "guided_collect_target",
            "traversal_guided_target", "effective_iterative_scan", "iterative_scan",
            "backend_cpu_exact_match", "error",
        ]
        rows = []
        for query_no in range(count):
            for mode, latency in ((subject.SQLENS_MODE, 5.0), (subject.STOCK_MODE, 10.0)):
                rows.append({
                    "selectivity": "1.0",
                    "filter_name": filter_name,
                    "mode": mode,
                    "query_no": query_no,
                    "query_id": 1000 + query_no,
                    "repeat": 0,
                    "pair_key": f"{filter_name}|q{query_no}|r0",
                    "recall": 0.91,
                    "end_to_end_ms": latency,
                    "effective_ef_search": 100,
                    "iterative_scan": "off",
                    "effective_iterative_scan": "off",
                    "backend_cpu_exact_match": "True",
                    "error": "",
                })
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        plan = {"build": "build-test", "vector": digest("vector")}
        Path(str(path) + ".plan.json").write_text(json.dumps(plan), encoding="utf-8")
        return path

    def test_audit_filter_accepts_strict_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_shard(Path(directory), "f01")
            rows, evidence = subject.audit_filter(
                path,
                "f01",
                expected_build_id="build-test",
                expected_vector_sha=digest("vector"),
            )
            self.assertEqual(len(rows), 4)
            self.assertEqual(evidence["paired_queries"], 2)

    def test_audit_filter_rejects_unpaired_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_shard(Path(directory), "f01")
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(subject.AuditError, "strictly paired"):
                subject.audit_filter(
                    path,
                    "f01",
                    expected_build_id="build-test",
                    expected_vector_sha=digest("vector"),
                )

    def test_stratified_speedup_uses_all_filters(self) -> None:
        values = {
            f"f{number:02d}": {
                subject.STOCK_MODE: [10.0, 12.0],
                subject.SQLENS_MODE: [5.0, 6.0],
            }
            for number in range(14)
        }
        center, low, high, wins = subject.stratified_speedup(
            values, samples=200, seed=17
        )
        self.assertAlmostEqual(center, 2.0)
        self.assertAlmostEqual(low, 2.0)
        self.assertAlmostEqual(high, 2.0)
        self.assertEqual(wins, 14)

    def make_combined(self, root: Path) -> Path:
        path = root / "combined.csv"
        fields = [
            "selectivity", "filter_name", "mode", "query_no", "query_id",
            "repeat", "pair_key", "request_no", "trace_cycle", "recall",
            "end_to_end_ms", "effective_ef_search", "max_scan_tuples",
            "scan_mem_multiplier", "guided_collect_target",
            "traversal_guided_target", "iterative_scan",
            "effective_iterative_scan", "backend_cpu_exact_match", "error",
        ]
        rows = []
        for query_no in range(14):
            filter_name = f"f{query_no:02d}"
            for mode, latency in (
                (subject.STOCK_MODE, 10.0), (subject.SQLENS_MODE, 5.0)
            ):
                rows.append({
                    "selectivity": str(query_no + 1),
                    "filter_name": filter_name,
                    "mode": mode,
                    "query_no": query_no,
                    "query_id": 1000 + query_no,
                    "repeat": 0,
                    "pair_key": f"{filter_name}|q{query_no}|r0",
                    "request_no": query_no,
                    "trace_cycle": 0,
                    "recall": 0.99,
                    "end_to_end_ms": latency,
                    "effective_ef_search": 100,
                    "max_scan_tuples": 5000,
                    "scan_mem_multiplier": 32,
                    "guided_collect_target": 100,
                    "traversal_guided_target": 10,
                    "iterative_scan": "off",
                    "effective_iterative_scan": "off",
                    "backend_cpu_exact_match": "True",
                    "error": "",
                })
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        runtime = {
            "exact_match": True,
            "observed_build_id": "build-test",
            "observed_vector_so_sha256": digest("vector"),
        }
        plan = {
            "status": "complete",
            "output_rows": len(rows),
            "output_sha256": subject.sha256_file(path),
            "query_error_summary": {"error_rows": 0},
            "query_contract": {
                "workload_requests": 14,
                "workload_unique_queries": 14,
                "workload_sha256": digest("workload"),
                "filters_sha256": digest("filters"),
                "truth_sha256": digest("truth"),
            },
            "sqlens_runtime_identity_startup": runtime,
            "sqlens_runtime_identity_final": runtime,
        }
        Path(str(path) + ".plan.json").write_text(
            json.dumps(plan), encoding="utf-8"
        )
        return path

    def make_throughput(self, root: Path) -> Path:
        path = root / "throughput.repeats.csv"
        fields = [
            "arm_id", "clients", "repeat_id", "requests", "unique_queries",
            "completed_queries", "error_count", "wall_clock_seconds",
            "recall_mean", "recall_ci95_low", "recall_ci95_high",
            "throughput_qps", "throughput_source", "status",
        ]
        rows = []
        for arm, seconds, status in (
            ("stock_pgvector", 7.0, "valid"),
            ("sqlens_full", 3.5, "invalid"),
        ):
            rows.append({
                "arm_id": arm,
                "clients": 16,
                "repeat_id": 0,
                "requests": 14,
                "unique_queries": 14,
                "completed_queries": 14,
                "error_count": 0,
                "wall_clock_seconds": seconds,
                "recall_mean": 0.99,
                "recall_ci95_low": 0.989,
                "recall_ci95_high": 0.991,
                "throughput_qps": 14 / seconds,
                "throughput_source": "measured_completed_over_barrier_wall_clock",
                "status": status,
            })
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        requests_path = root / "throughput.requests.csv"
        request_fields = [
            "arm_id", "repeat_id", "request_no", "trace_cycle",
            "filter_name", "query_no", "query_id", "recall_at_10",
            "error_type", "error",
        ]
        request_rows = []
        for arm in ("stock_pgvector", "sqlens_full"):
            for query_no in range(14):
                request_rows.append({
                    "arm_id": arm,
                    "repeat_id": 0,
                    "request_no": query_no,
                    "trace_cycle": 0,
                    "filter_name": f"f{query_no:02d}",
                    "query_no": query_no,
                    "query_id": 1000 + query_no,
                    "recall_at_10": 0.99,
                    "error_type": "",
                    "error": "",
                })
        with requests_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=request_fields)
            writer.writeheader()
            writer.writerows(request_rows)
        base_search = {
            subject.STOCK_MODE: {
                "ef_search": 100,
                "max_scan_tuples": 5000,
                "scan_mem_multiplier": 32,
                "iterative_scan": "off",
                "guided_collect_target": 100,
                "traversal_guided_target": 10,
            },
            subject.SQLENS_MODE: {
                "ef_search": 100,
                "max_scan_tuples": 5000,
                "scan_mem_multiplier": 32,
                "iterative_scan": "off",
                "guided_collect_target": 100,
                "traversal_guided_target": 10,
            },
        }
        manifest = {
            "artifact_valid": False,
            "paper_eligible": False,
            "outputs": {
                "repeats": {
                    "rows": 2,
                    "sha256": subject.sha256_file(path),
                },
                "requests": {
                    "path": str(requests_path),
                    "rows": len(request_rows),
                    "sha256": subject.sha256_file(requests_path),
                },
            },
            "configuration": {
                "value": {
                    "search": base_search,
                    "per_filter_search": {
                        "stock_pgvector": {
                            "ef_search": {},
                            "traversal_guided_target": {},
                        },
                        "sqlens_full": {
                            "ef_search": {},
                            "traversal_guided_target": {},
                        },
                    },
                }
            },
            "inputs": {
                "filters_csv": {"sha256": digest("filters")},
                "truth_csv": {"sha256": digest("truth")},
            },
            "runtime_binary": {
                "expected_build_id": "build-test",
                "expected_vector_so_sha256": digest("vector"),
            },
            "protocol": {
                "clients": 16,
                "repeats": 1,
                "requests_per_arm_repeat": 14,
                "client_cpu_list": "32-47",
                "backend_cpu_list": "48-63",
                "throughput_formula": (
                    "completed_queries / barrier_wall_clock_seconds"
                ),
            },
            "gates": {
                "barrier_wall_clock_qps": True,
                "independent_client_backends": True,
                "telemetry_complete": True,
                "minimum_six_repeats": False,
            },
        }
        path.with_name("throughput.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return path

    def test_combined_summary_binds_latency_and_throughput(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.make_combined(root)
            throughput = self.make_throughput(root)
            paths = subject.summarize_combined(
                raw,
                throughput,
                "laion",
                0.99,
                root / "summary",
                bootstrap_samples=100,
                bootstrap_seed=17,
                expected_requests=14,
            )
            with paths["summary"].open(newline="", encoding="utf-8") as source:
                summary = next(csv.DictReader(source))
            self.assertAlmostEqual(float(summary["stock_qps"]), 2.0)
            self.assertAlmostEqual(float(summary["sqlens_qps"]), 4.0)
            self.assertAlmostEqual(float(summary["speedup_geomean"]), 2.0)
            self.assertEqual(summary["wins"], "14")
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertTrue(manifest["artifact_valid"])
            self.assertFalse(manifest["paper_eligible"])
            self.assertFalse(
                manifest["throughput_evidence"]["artifact_valid"]
            )
            self.assertTrue(manifest["cross_artifact_identity"]["passed"])
            self.assertEqual(
                manifest["latency_evidence"]["request_trace_identity_sha256"],
                manifest["throughput_evidence"][
                    "request_trace_identity_sha256"
                ],
            )

    def test_combined_audit_rejects_plan_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_combined(Path(directory))
            with path.open("a", encoding="utf-8") as target:
                target.write("\n")
            with self.assertRaisesRegex(subject.AuditError, "latency plan"):
                subject.audit_combined_raw(path, expected_requests=14)

    def test_combined_summary_rejects_different_request_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.make_combined(root)
            throughput = self.make_throughput(root)
            requests_path = root / "throughput.requests.csv"
            fields, rows = subject.read_csv(requests_path)
            for row in rows:
                if row["request_no"] == "0":
                    row["query_id"] = "9999"
            with requests_path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = root / "throughput.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"]["requests"]["sha256"] = subject.sha256_file(
                requests_path
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(subject.AuditError, "same experiment"):
                subject.summarize_combined(
                    raw,
                    throughput,
                    "laion",
                    0.99,
                    root / "summary",
                    bootstrap_samples=100,
                    bootstrap_seed=17,
                    expected_requests=14,
                )

    def test_combined_summary_marks_per_filter_throughput_miss_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.make_combined(root)
            throughput = self.make_throughput(root)
            requests_path = root / "throughput.requests.csv"
            fields, rows = subject.read_csv(requests_path)
            for row in rows:
                if (
                    row["arm_id"] == "sqlens_full"
                    and row["filter_name"] == "f00"
                ):
                    row["recall_at_10"] = "0.98"
            with requests_path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = root / "throughput.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"]["requests"]["sha256"] = subject.sha256_file(
                requests_path
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            paths = subject.summarize_combined(
                raw,
                throughput,
                "laion",
                0.99,
                root / "summary",
                bootstrap_samples=100,
                bootstrap_seed=17,
                expected_requests=14,
            )
            result = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertFalse(result["artifact_valid"])
            self.assertTrue(result["target_gate"]["latency_all_filters_met"])
            self.assertFalse(
                result["target_gate"]["throughput_all_filters_met"]
            )


if __name__ == "__main__":
    unittest.main()
