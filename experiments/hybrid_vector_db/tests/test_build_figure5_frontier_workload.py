from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_figure5_frontier_workload as builder  # noqa: E402


FILTER_NAMES = [f"filter_{index:02d}" for index in range(14)]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="ascii", newline="") as source:
        return list(csv.DictReader(source))


class Fixture:
    def __init__(
        self,
        root: Path,
        *,
        calibration_queries: int = 4,
        calibration_requests: int = 18,
        measurement_queries: int = 15,
        custom_query_columns: bool = False,
    ) -> None:
        self.root = root
        self.calibration_queries = calibration_queries
        self.calibration_requests = calibration_requests
        self.measurement_queries = measurement_queries
        self.query_no_column = "ordinal" if custom_query_columns else "query_no"
        self.query_id_column = "vector_id" if custom_query_columns else "query_id"
        self.query_csv = root / "queries.csv"
        self.filters_csv = root / "filters.csv"
        self.truth_csv = root / "truth.csv"
        self.out_prefix = root / "figure5"
        self.write_queries()
        self.write_filters()
        self.write_truth()

    @property
    def all_query_nos(self) -> range:
        return range(self.calibration_queries + self.measurement_queries)

    def write_queries(self) -> None:
        write_csv(
            self.query_csv,
            [self.query_no_column, self.query_id_column],
            [
                {
                    self.query_no_column: query_no,
                    self.query_id_column: 100_000 + query_no,
                }
                for query_no in self.all_query_nos
            ],
        )

    def write_filters(self, count: int = 14) -> None:
        write_csv(
            self.filters_csv,
            [
                "filter_name",
                "actual_pct",
                "target_rate",
                "predicate",
                "atoms",
            ],
            [
                {
                    "filter_name": FILTER_NAMES[index],
                    "actual_pct": f"{50.0 / (index + 1):.6f}" if index % 2 == 0 else "",
                    "target_rate": "" if index % 2 == 0 else f"{40.0 / (index + 1):.6f}%",
                    "predicate": f"tag = {index}",
                    "atoms": f"tag_eq_{index}",
                }
                for index in range(count)
            ],
        )

    def truth_rows(self) -> list[dict[str, object]]:
        ids = ",".join(str(value) for value in range(10, 20))
        distances = ",".join(str(value) for value in range(10))
        rows: list[dict[str, object]] = []
        for query_no in self.all_query_nos:
            for filter_index, filter_name in enumerate(FILTER_NAMES):
                rows.append(
                    {
                        "query_no": query_no,
                        "query_id": 100_000 + query_no,
                        "filter_name": filter_name,
                        "predicate": f"tag = {filter_index}",
                        "method": "pre_filter_exact",
                        "k": 10,
                        "result_ids": ids,
                        "exact_filtered_topk_ids": ids,
                        "exact_filtered_topk_distances_sq": distances,
                        "kth_distance_sq": 9,
                        "tie_tolerance": 0.000009,
                        "strict_closer_count": 9,
                        "boundary_tied": "false",
                    }
                )
        return rows

    def write_truth(self, rows: list[dict[str, object]] | None = None) -> None:
        rows = self.truth_rows() if rows is None else rows
        write_csv(
            self.truth_csv,
            [
                "query_no",
                "query_id",
                "filter_name",
                "predicate",
                "method",
                "k",
                "result_ids",
                "exact_filtered_topk_ids",
                "exact_filtered_topk_distances_sq",
                "kth_distance_sq",
                "tie_tolerance",
                "strict_closer_count",
                "boundary_tied",
            ],
            rows,
        )

    def config(self, *, out_prefix: Path | None = None, seed: int = 17) -> builder.BuildConfig:
        return builder.BuildConfig(
            query_cohort_csv=self.query_csv,
            filters_csv=self.filters_csv,
            truth_csv=self.truth_csv,
            out_prefix=out_prefix or self.out_prefix,
            query_no_column=self.query_no_column,
            query_id_column=self.query_id_column,
            calibration_query_start=0,
            calibration_query_count=self.calibration_queries,
            calibration_requests=self.calibration_requests,
            measurement_query_start=self.calibration_queries,
            measurement_query_count=self.measurement_queries,
            seed=seed,
        )


def test_cli_defaults_define_q0_199_and_exact_q10k_measurement() -> None:
    args = builder.create_argument_parser().parse_args(
        [
            "--query-cohort-csv",
            "queries.csv",
            "--filters-csv",
            "filters.csv",
            "--truth-csv",
            "truth.csv",
            "--out-prefix",
            "figure5",
        ]
    )
    assert args.calibration_query_start == 0
    assert args.calibration_query_count == 200
    assert args.measurement_query_start == 200
    assert args.measurement_query_count == 10_000
    assert args.calibration_requests == 2_000
    assert args.calibration_protocol == builder.LEGACY_CALIBRATION_PROTOCOL
    assert args.require_formal_paper_calibration is False


def test_cli_exposes_formal_per_predicate_q2800_protocol() -> None:
    args = builder.create_argument_parser().parse_args(
        [
            "--query-cohort-csv",
            "queries.csv",
            "--filters-csv",
            "filters.csv",
            "--truth-csv",
            "truth.csv",
            "--out-prefix",
            "figure5",
            "--calibration-protocol",
            builder.FORMAL_CALIBRATION_PROTOCOL,
            "--calibration-requests",
            str(builder.FORMAL_CALIBRATION_REQUESTS),
            "--require-formal-paper-calibration",
        ]
    )
    assert args.calibration_protocol == builder.FORMAL_CALIBRATION_PROTOCOL
    assert args.calibration_requests == 2_800
    assert args.require_formal_paper_calibration is True


def test_default_measurement_constructs_exactly_q200_through_q10199_once() -> None:
    queries = {
        query_no: builder.Query(query_no, str(1_000_000 + query_no))
        for query_no in range(200, 10_200)
    }
    filters = [
        builder.Filter(
            name=name,
            predicate=f"tag = {index}",
            atoms=f"tag_eq_{index}",
            selectivity_field="actual_pct",
            selectivity_raw="1.0",
            selectivity_pct=1.0,
        )
        for index, name in enumerate(FILTER_NAMES)
    ]
    rows = builder.build_measurement_rows(
        queries,
        filters,
        query_nos=list(range(200, 10_200)),
        seed=builder.DEFAULT_SEED,
    )
    counts = {
        name: sum(row["filter_name"] == name for row in rows)
        for name in FILTER_NAMES
    }
    assert len(rows) == 10_000
    assert {row["query_no"] for row in rows} == set(range(200, 10_200))
    assert len({row["query_id"] for row in rows}) == 10_000
    assert set(counts.values()) == {714, 715}


def test_trace_only_then_assigned_truth_closes_the_artifact(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    pending = replace(
        fixture.config(out_prefix=tmp_path / "external"),
        truth_csv=None,
        truth_coverage="assigned",
        trace_only=True,
    )

    pending_manifest = builder.build(pending)

    assert pending_manifest["artifact_valid"] is False
    assert pending_manifest["stage"] == "trace_pending_truth"
    assigned: set[tuple[int, str]] = set()
    for key in ("calibration_workload_csv", "measurement_workload_csv"):
        with Path(pending_manifest["outputs"][key]["path"]).open(
            newline="", encoding="utf-8"
        ) as source:
            assigned.update(
                (int(row["query_no"]), row["filter_name"])
                for row in csv.DictReader(source)
            )
    selected_truth = [
        row
        for row in fixture.truth_rows()
        if (int(row["query_no"]), str(row["filter_name"])) in assigned
    ]
    assigned_truth = tmp_path / "assigned_truth.csv"
    write_csv(
        assigned_truth,
        [
            "query_no",
            "query_id",
            "filter_name",
            "predicate",
            "method",
            "k",
            "result_ids",
            "exact_filtered_topk_ids",
            "exact_filtered_topk_distances_sq",
            "kth_distance_sq",
            "tie_tolerance",
            "strict_closer_count",
            "boundary_tied",
        ],
        selected_truth,
    )
    audited = replace(
        pending,
        truth_csv=assigned_truth,
        trace_only=False,
    )

    manifest = builder.build(audited)

    assert manifest["artifact_valid"] is True
    assert manifest["stage"] == "audited"
    assert manifest["truth"]["coverage"] == "assigned"
    assert manifest["truth"]["required_pairs"] == len(assigned)


def test_builds_balanced_deterministic_workloads_and_valid_manifest(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path, custom_query_columns=True)
    manifest = builder.build(fixture.config())
    paths = fixture.config().output_paths
    calibration = read_csv(paths["calibration_workload_csv"])
    measurement = read_csv(paths["measurement_workload_csv"])

    assert list(measurement[0]) == list(builder.OUTPUT_FIELDS)
    assert len(calibration) == 18
    assert len(measurement) == 15
    assert len({row["query_no"] for row in measurement}) == 15
    assert len({row["query_id"] for row in measurement}) == 15
    assert {row["filter_name"] for row in measurement} == set(FILTER_NAMES)
    measurement_counts = manifest["distribution"]["measurement"]["filter_counts"]
    assert max(measurement_counts.values()) - min(measurement_counts.values()) == 1
    assert manifest["distribution"]["calibration"]["unique_queries"] == 4
    assert manifest["distribution"]["calibration"]["trace_cycles"] == 5
    assert manifest["artifact_valid"] is True
    assert all(manifest["gates"].values())
    assert manifest["truth"]["matched_pairs"] == 19 * 14
    assert manifest["outputs"]["calibration_workload_csv"]["sha256"] == (
        builder.sha256_file(paths["calibration_workload_csv"])
    )
    assert manifest["outputs"]["measurement_workload_csv"]["sha256"] == (
        builder.sha256_file(paths["measurement_workload_csv"])
    )
    disk_manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    assert disk_manifest == manifest
    assert manifest["outputs"]["manifest_json"]["content_sha256"] == (
        builder.manifest_content_sha256(manifest)
    )
    assert not fixture.config().journal_path.exists()


def test_formal_calibration_builds_complete_interleaved_cartesian_trace(
    tmp_path: Path,
) -> None:
    fixture = Fixture(
        tmp_path,
        calibration_queries=4,
        calibration_requests=4 * len(FILTER_NAMES),
        measurement_queries=14,
    )
    config = replace(
        fixture.config(),
        calibration_protocol=builder.FORMAL_CALIBRATION_PROTOCOL,
    )
    manifest = builder.build(config)
    rows = read_csv(config.output_paths["calibration_workload_csv"])
    pairs = {(int(row["query_no"]), row["filter_name"]) for row in rows}

    assert len(rows) == 56
    assert len(pairs) == 56
    assert pairs == {
        (query_no, filter_name)
        for query_no in range(4)
        for filter_name in FILTER_NAMES
    }
    assert {row["trace_cycle"] for row in rows} == {
        str(index) for index in range(14)
    }
    for trace_cycle in range(14):
        cycle_rows = [row for row in rows if row["trace_cycle"] == str(trace_cycle)]
        assert len(cycle_rows) == 4
        assert {row["query_no"] for row in cycle_rows} == {"0", "1", "2", "3"}
    distribution = manifest["distribution"]["calibration"]
    assert distribution["filter_counts"] == {name: 4 for name in FILTER_NAMES}
    assert distribution["cartesian_coverage"] == {
        "expected_pairs": 56,
        "observed_rows": 56,
        "observed_unique_pairs": 56,
        "missing_pairs": 0,
        "duplicate_pairs": 0,
        "complete": True,
        "canonical_pair_sha256": distribution["cartesian_coverage"][
            "canonical_pair_sha256"
        ],
    }
    assert manifest["construction"]["calibration"]["per_predicate_cartesian"] is True
    assert manifest["formal_paper_calibration"]["required"] is False
    assert manifest["formal_paper_calibration"]["passed"] is False
    assert manifest["gates"]["calibration_cartesian_coverage"] is True
    assert manifest["gates"]["calibration_exact_per_filter_count"] is True


def test_formal_calibration_is_byte_deterministic_and_seeded(tmp_path: Path) -> None:
    fixture = Fixture(
        tmp_path,
        calibration_queries=4,
        calibration_requests=56,
        measurement_queries=14,
    )
    first = replace(
        fixture.config(out_prefix=tmp_path / "first", seed=23),
        calibration_protocol=builder.FORMAL_CALIBRATION_PROTOCOL,
    )
    second = replace(
        fixture.config(out_prefix=tmp_path / "second", seed=23),
        calibration_protocol=builder.FORMAL_CALIBRATION_PROTOCOL,
    )
    third = replace(
        fixture.config(out_prefix=tmp_path / "third", seed=24),
        calibration_protocol=builder.FORMAL_CALIBRATION_PROTOCOL,
    )
    builder.build(first)
    builder.build(second)
    builder.build(third)

    assert first.output_paths["calibration_workload_csv"].read_bytes() == (
        second.output_paths["calibration_workload_csv"].read_bytes()
    )
    assert first.output_paths["calibration_workload_csv"].read_bytes() != (
        third.output_paths["calibration_workload_csv"].read_bytes()
    )


def test_formal_paper_contract_is_exactly_q2800_per_predicate() -> None:
    queries = {
        query_no: builder.Query(query_no, str(1_000_000 + query_no))
        for query_no in range(builder.FORMAL_CALIBRATION_QUERY_COUNT)
    }
    filters = [
        builder.Filter(
            name=name,
            predicate=f"tag = {index}",
            atoms=f"tag_eq_{index}",
            selectivity_field="actual_pct",
            selectivity_raw="1.0",
            selectivity_pct=1.0,
        )
        for index, name in enumerate(FILTER_NAMES)
    ]
    rows = builder.build_calibration_rows(
        queries,
        filters,
        query_nos=list(range(builder.FORMAL_CALIBRATION_QUERY_COUNT)),
        requests=builder.FORMAL_CALIBRATION_REQUESTS,
        seed=builder.DEFAULT_SEED,
        protocol=builder.FORMAL_CALIBRATION_PROTOCOL,
    )
    coverage = builder._cartesian_coverage(
        rows,
        query_nos=list(range(builder.FORMAL_CALIBRATION_QUERY_COUNT)),
        filter_names=FILTER_NAMES,
    )

    assert len(rows) == 2_800
    assert Counter(row["filter_name"] for row in rows) == Counter(
        {name: 200 for name in FILTER_NAMES}
    )
    assert coverage["complete"] is True
    assert coverage["observed_unique_pairs"] == 2_800


def test_formal_paper_calibration_gate_accepts_exact_contract(tmp_path: Path) -> None:
    config = builder.BuildConfig(
        query_cohort_csv=tmp_path / "queries.csv",
        filters_csv=tmp_path / "filters.csv",
        truth_csv=tmp_path / "truth.csv",
        out_prefix=tmp_path / "figure5",
        calibration_query_count=builder.FORMAL_CALIBRATION_QUERY_COUNT,
        calibration_requests=builder.FORMAL_CALIBRATION_REQUESTS,
        calibration_protocol=builder.FORMAL_CALIBRATION_PROTOCOL,
        require_formal_paper_calibration=True,
    )

    config.validate()


@pytest.mark.parametrize(
    ("protocol", "query_count", "requests", "message"),
    [
        (
            builder.LEGACY_CALIBRATION_PROTOCOL,
            builder.FORMAL_CALIBRATION_QUERY_COUNT,
            builder.FORMAL_CALIBRATION_REQUESTS,
            "requires --calibration-protocol",
        ),
        (
            builder.FORMAL_CALIBRATION_PROTOCOL,
            199,
            199 * len(FILTER_NAMES),
            "requires exactly 200 calibration queries",
        ),
        (
            builder.FORMAL_CALIBRATION_PROTOCOL,
            builder.FORMAL_CALIBRATION_QUERY_COUNT,
            builder.FORMAL_CALIBRATION_REQUESTS - 1,
            "one request for every calibration query/filter pair",
        ),
    ],
)
def test_formal_paper_calibration_gate_rejects_non_q2800_contract(
    tmp_path: Path,
    protocol: str,
    query_count: int,
    requests: int,
    message: str,
) -> None:
    fixture = Fixture(tmp_path)
    config = replace(
        fixture.config(),
        calibration_query_count=query_count,
        calibration_requests=requests,
        measurement_query_start=query_count,
        calibration_protocol=protocol,
        require_formal_paper_calibration=True,
    )
    with pytest.raises(builder.WorkloadError, match=message):
        config.validate()


def test_same_seed_is_byte_deterministic_and_different_seed_changes_trace(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    first = fixture.config(out_prefix=tmp_path / "first", seed=23)
    second = fixture.config(out_prefix=tmp_path / "second", seed=23)
    third = fixture.config(out_prefix=tmp_path / "third", seed=24)
    builder.build(first)
    builder.build(second)
    builder.build(third)

    assert first.output_paths["calibration_workload_csv"].read_bytes() == (
        second.output_paths["calibration_workload_csv"].read_bytes()
    )
    assert first.output_paths["measurement_workload_csv"].read_bytes() == (
        second.output_paths["measurement_workload_csv"].read_bytes()
    )
    assert first.output_paths["measurement_workload_csv"].read_bytes() != (
        third.output_paths["measurement_workload_csv"].read_bytes()
    )


def test_calibration_repeats_only_across_trace_cycles(tmp_path: Path) -> None:
    fixture = Fixture(
        tmp_path,
        calibration_queries=3,
        calibration_requests=14,
        measurement_queries=14,
    )
    builder.build(fixture.config())
    rows = read_csv(fixture.config().output_paths["calibration_workload_csv"])
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["trace_cycle"], row["query_no"])
        assert key not in seen
        seen.add(key)
    assert {row["trace_cycle"] for row in rows} == {"0", "1", "2", "3", "4"}


def test_rejects_missing_truth_pair_without_publishing(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    rows = fixture.truth_rows()
    rows.pop()
    fixture.write_truth(rows)
    with pytest.raises(builder.WorkloadError, match="complete required"):
        builder.build(fixture.config())
    assert not any(path.exists() for path in fixture.config().output_paths.values())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("exact_filtered_topk_distances_sq", "0,1,2,3,4,5,6,7,9,8", "not ordered"),
        ("kth_distance_sq", 8, "kth distance differs"),
        ("tie_tolerance", 0, "must be positive"),
        ("strict_closer_count", 8, "strict_closer_count is inconsistent"),
        ("method", "approximate", "not produced by pre_filter_exact"),
    ],
)
def test_rejects_malformed_tie_aware_truth(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    fixture = Fixture(tmp_path)
    rows = fixture.truth_rows()
    rows[0][field] = value
    fixture.write_truth(rows)
    with pytest.raises(builder.WorkloadError, match=message):
        builder.build(fixture.config())


def test_rejects_nonunique_measurement_vector_ids(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fields = [fixture.query_no_column, fixture.query_id_column]
    rows = [
        {
            fixture.query_no_column: query_no,
            fixture.query_id_column: 100_000 + query_no,
        }
        for query_no in fixture.all_query_nos
    ]
    rows[-1][fixture.query_id_column] = rows[-2][fixture.query_id_column]
    write_csv(fixture.query_csv, fields, rows)
    with pytest.raises(builder.WorkloadError, match="multiple query numbers"):
        builder.build(fixture.config())


def test_requires_exactly_fourteen_filters(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.write_filters(count=13)
    with pytest.raises(builder.WorkloadError, match="exactly 14"):
        builder.build(fixture.config())


def test_publish_failure_restores_previous_bundle(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    paths = fixture.config().output_paths
    for name, path in paths.items():
        path.write_text(f"old-{name}\n", encoding="ascii")
    original_install = builder._install_staged_file
    calls = 0

    def fail_second_install(staged: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publish failure")
        original_install(staged, destination)

    with mock.patch.object(
        builder, "_install_staged_file", side_effect=fail_second_install
    ):
        with pytest.raises(OSError, match="simulated publish failure"):
            builder.build(fixture.config())
    for name, path in paths.items():
        assert path.read_text(encoding="ascii") == f"old-{name}\n"
    assert not fixture.config().journal_path.exists()


def test_recovery_rolls_back_interrupted_uncommitted_bundle(tmp_path: Path) -> None:
    destination = tmp_path / "result.csv"
    staged = tmp_path / ".result.staged"
    backup = tmp_path / ".result.backup"
    journal = tmp_path / ".result.journal.json"
    destination.write_text("new\n", encoding="ascii")
    backup.write_text("old\n", encoding="ascii")
    staged.write_text("unused\n", encoding="ascii")
    journal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "staged",
                "entries": [
                    {
                        "destination": str(destination.resolve()),
                        "staged": str(staged.resolve()),
                        "backup": str(backup.resolve()),
                        "old_existed": True,
                    }
                ],
            }
        ),
        encoding="ascii",
    )
    assert builder.recover_atomic_bundle(journal) is True
    assert destination.read_text(encoding="ascii") == "old\n"
    assert not staged.exists()
    assert not backup.exists()
    assert not journal.exists()


def test_cli_returns_nonzero_and_preserves_existing_outputs_on_invalid_input(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    outputs = fixture.config().output_paths
    for path in outputs.values():
        path.write_text("sentinel\n", encoding="ascii")
    rows = fixture.truth_rows()
    rows.pop()
    fixture.write_truth(rows)
    result = builder.main(
        [
            "--query-cohort-csv",
            str(fixture.query_csv),
            "--filters-csv",
            str(fixture.filters_csv),
            "--truth-csv",
            str(fixture.truth_csv),
            "--out-prefix",
            str(fixture.out_prefix),
            "--calibration-query-count",
            str(fixture.calibration_queries),
            "--calibration-requests",
            str(fixture.calibration_requests),
            "--measurement-query-start",
            str(fixture.calibration_queries),
            "--measurement-query-count",
            str(fixture.measurement_queries),
        ]
    )
    assert result == 2
    assert all(path.read_text(encoding="ascii") == "sentinel\n" for path in outputs.values())
