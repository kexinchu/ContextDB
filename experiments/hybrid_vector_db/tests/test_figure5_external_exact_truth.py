from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.hybrid_vector_db.scripts import figure5_external_exact_truth as exact


def write_xbin(path: Path, values: np.ndarray, dtype: np.dtype) -> None:
    values = np.asarray(values, dtype=dtype)
    with path.open("wb") as target:
        np.asarray([values.shape[0], values.shape[1]], dtype="<u4").tofile(target)
        values.tofile(target)


def write_spmat(path: Path, rows: list[list[int]]) -> None:
    indptr = [0]
    indices: list[int] = []
    for row in rows:
        indices.extend(row)
        indptr.append(len(indices))
    with path.open("wb") as target:
        np.asarray([len(rows), 100, len(indices)], dtype="<i8").tofile(target)
        np.asarray(indptr, dtype="<i8").tofile(target)
        np.asarray(indices, dtype="<i4").tofile(target)
        np.ones(len(indices), dtype="<f4").tofile(target)


def write_filters(path: Path, field: str, labels: list[int]) -> list[str]:
    names = [f"f{index:02d}" for index in range(14)]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("filter_name", "target_rate", "actual_pct", "expected_rows", "predicate", "atoms"),
        )
        writer.writeheader()
        for index, name in enumerate(names):
            label = labels[index % len(labels)]
            predicate = f"{field} && ARRAY[{label}]::int[]"
            writer.writerow(
                {
                    "filter_name": name,
                    "target_rate": 10,
                    "actual_pct": 50,
                    "expected_rows": "",
                    "predicate": predicate,
                    "atoms": f"sql:{field} @> ARRAY[{label}]::int[]",
                }
            )
    return names


def write_workload(path: Path, filters: list[str], *, duplicate: bool = False) -> None:
    rows = []
    for request_no, name in enumerate(filters):
        rows.append({"request_no": request_no, "query_no": request_no, "query_id": request_no, "filter_name": name})
    if duplicate:
        rows.append({"request_no": len(rows), "query_no": 0, "query_id": 0, "filter_name": filters[0]})
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("request_no", "query_no", "query_id", "filter_name"))
        writer.writeheader()
        writer.writerows(rows)


def write_cartesian_workload(path: Path, filters: list[str], query_ids: list[int]) -> None:
    rows = []
    for query_id in query_ids:
        for name in filters:
            rows.append(
                {
                    "request_no": len(rows),
                    "query_no": query_id,
                    "query_id": query_id,
                    "filter_name": name,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("request_no", "query_no", "query_id", "filter_name"))
        writer.writeheader()
        writer.writerows(rows)


def test_mask_helpers_are_overlap_masks() -> None:
    indptr = np.asarray([0, 2, 2, 3, 5], dtype=np.int64)
    labels = np.asarray([3, 7, 9, 7, 11], dtype=np.int32)
    assert exact.membership_mask_csr(indptr, labels, 0, 4, (7,)).tolist() == [True, False, False, True]
    assert exact.membership_mask_offsets(indptr, labels, 0, 4, (11,)).tolist() == [False, False, False, True]


def test_parser_fails_closed_for_unsupported_logic(tmp_path: Path) -> None:
    path = tmp_path / "filters.csv"
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("filter_name", "predicate", "atoms"))
        writer.writeheader()
        for index in range(14):
            writer.writerow({"filter_name": f"f{index}", "predicate": "tags @> ARRAY[1]::int[]", "atoms": "sql:tags @> ARRAY[1]::int[]"})
    with pytest.raises(exact.ExactTruthError, match="unsupported predicate"):
        exact.load_filters(path, "yfcc")


def test_workload_rejects_duplicate_pair(tmp_path: Path) -> None:
    filters_path = tmp_path / "filters.csv"
    names = write_filters(filters_path, "tags", [1, 2])
    workload_path = tmp_path / "workload.csv"
    write_workload(workload_path, names, duplicate=True)
    filters = exact.load_filters(filters_path, "yfcc")
    with pytest.raises(exact.ExactTruthError, match="duplicate assigned pair"):
        exact.load_workload(workload_path, filters)


def test_tie_fields_are_boundary_aware() -> None:
    fields = exact.tie_fields([1.0, 1.0, 1.0], 2)
    assert fields["strict_closer_count"] == 0
    assert fields["boundary_tied"] is True
    with pytest.raises(exact.ExactTruthError, match="non-finite"):
        exact.tie_fields([1.0, float("inf")], 2)


def test_yfcc_tiny_end_to_end_uses_only_matching_vectors_and_publishes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.u8bin"
    query = tmp_path / "query.u8bin"
    metadata = tmp_path / "base.spmat"
    filters_path = tmp_path / "filters.csv"
    workload = tmp_path / "workload.csv"
    output = tmp_path / "truth.csv"
    manifest = tmp_path / "truth.json"
    write_xbin(base, np.asarray([[0, 0], [10, 0], [0, 10], [1, 1]], dtype=np.uint8), np.dtype("u1"))
    write_xbin(query, np.asarray([[0, 0], [10, 0]], dtype=np.uint8), np.dtype("u1"))
    write_spmat(metadata, [[7], [7], [9], [7, 9]])
    names = write_filters(filters_path, "tags", [7, 9])
    write_workload(workload, names[:2])
    args = SimpleNamespace(
        dataset="yfcc",
        workload_csv=str(workload),
        filters_csv=str(filters_path),
        query_vector_source=str(query),
        base_vector_source=[str(base)],
        base_metadata_source=str(metadata),
        label_offsets_source=None,
        flat_labels_source=None,
        output_truth_csv=str(output),
        output_manifest=str(manifest),
        device="cpu",
        cuda_device=None,
        cpu_threads=1,
        query_batch=2,
        chunk_rows=2,
        k=1,
        overwrite=True,
    )
    result = exact.run_generation(args)
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["method"] == "pre_filter_exact"
    assert rows[0]["candidate_rows"] == "3"
    assert rows[0]["exact_filtered_topk_ids"] == "0"
    assert rows[1]["exact_filtered_topk_ids"] == "3"
    assert result["exact_coverage"]["assigned_pairs"] == 2
    on_disk = json.loads(manifest.read_text(encoding="utf-8"))
    assert on_disk["output"]["sha256"] == result["output"]["sha256"]
    assert on_disk["device"]["device_type"] == "cpu"
    assert on_disk["device"]["torch_num_threads"] == 1
    assert on_disk["exact_coverage"]["method"] == "full_base_scan_plus_cpu_float32_gemm_topk"
    assert on_disk["torch_backend"]["device"]["resolved_device"] == "cpu"
    assert on_disk["execution"]["execution_path"] == "legacy_by_filter"
    assert on_disk["execution"]["cartesian_proof"]["eligible"] is False


def test_laion_tiny_end_to_end_reads_shards_and_offsets(tmp_path: Path) -> None:
    query = tmp_path / "query.fbin"
    shard0 = tmp_path / "img_emb_0.npy"
    shard1 = tmp_path / "img_emb_1.npy"
    offsets_path = tmp_path / "offsets.int64"
    flat_path = tmp_path / "labels.int32"
    filters_path = tmp_path / "filters.csv"
    workload = tmp_path / "workload.csv"
    output = tmp_path / "truth.csv"
    manifest = tmp_path / "truth.json"
    np.save(shard0, np.asarray([[0, 0], [10, 0]], dtype=np.float32))
    np.save(shard1, np.asarray([[0, 10], [1, 1]], dtype=np.float32))
    write_xbin(query, np.asarray([[0, 0]], dtype=np.float32), np.dtype("<f4"))
    np.asarray([0, 1, 2, 3, 5], dtype="<i8").tofile(offsets_path)
    np.asarray([5, 7, 5, 9, 7], dtype="<i4").tofile(flat_path)
    names = write_filters(filters_path, "labels", [5, 7])
    with workload.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("request_no", "query_no", "query_id", "filter_name"))
        writer.writeheader()
        writer.writerows(
            [
                {"request_no": 0, "query_no": 0, "query_id": 0, "filter_name": names[0]},
                {"request_no": 1, "query_no": 1, "query_id": 0, "filter_name": names[1]},
            ]
        )
    args = SimpleNamespace(
        dataset="laion",
        workload_csv=str(workload),
        filters_csv=str(filters_path),
        query_vector_source=str(query),
        base_vector_source=[str(shard0), str(shard1)],
        base_metadata_source=None,
        label_offsets_source=str(offsets_path),
        flat_labels_source=str(flat_path),
        output_truth_csv=str(output),
        output_manifest=str(manifest),
        device="cpu",
        cuda_device=None,
        cpu_threads=1,
        query_batch=1,
        chunk_rows=2,
        k=1,
        overwrite=True,
    )
    exact.run_generation(args, device=torch.device("cpu"))
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert [row["candidate_rows"] for row in rows] == ["2", "2"]
    assert rows[0]["exact_filtered_topk_ids"] == "0"
    assert rows[1]["exact_filtered_topk_ids"] == "3"


def test_cuda_unavailable_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exact.torch.cuda, "is_available", lambda: False)
    with pytest.raises(exact.ExactTruthError, match="CUDA is unavailable"):
        exact.cuda_device("0")


def test_explicit_cpu_device_is_accepted_and_records_actual_thread_setting() -> None:
    args = SimpleNamespace(device="cpu", cuda_device=None, cpu_threads=1)
    device, provenance = exact.resolve_device(args)
    assert str(device) == "cpu"
    assert provenance["device_type"] == "cpu"
    assert provenance["cpu_threads_requested"] == 1
    assert provenance["torch_num_threads"] == 1


def test_parser_accepts_explicit_cpu_and_legacy_cuda_device() -> None:
    required = [
        "--dataset", "yfcc",
        "--workload-csv", "workload.csv",
        "--filters-csv", "filters.csv",
        "--query-vector-source", "queries.u8bin",
        "--base-vector-source", "base.u8bin",
        "--output-truth-csv", "truth.csv",
        "--output-manifest", "truth.json",
    ]
    cpu = exact.build_parser().parse_args([*required, "--device", "cpu", "--cpu-threads", "4"])
    legacy = exact.build_parser().parse_args([*required, "--cuda-device", "0"])
    assert (cpu.device, cpu.cpu_threads, cpu.cuda_device) == ("cpu", 4, None)
    assert (legacy.device, legacy.cuda_device) == (None, "0")


def test_no_explicit_device_never_falls_back_to_cpu() -> None:
    args = SimpleNamespace(device=None, cuda_device=None, cpu_threads=None)
    with pytest.raises(exact.ExactTruthError, match="explicit --device"):
        exact.resolve_device(args)


def test_cpu_threads_are_rejected_for_cuda() -> None:
    args = SimpleNamespace(device="cuda:0", cuda_device=None, cpu_threads=1)
    with pytest.raises(exact.ExactTruthError, match="only valid with --device cpu"):
        exact.resolve_device(args)


def test_cartesian_fast_path_matches_legacy_rows_ids_and_distances(tmp_path: Path) -> None:
    base = tmp_path / "base.u8bin"
    query = tmp_path / "query.u8bin"
    metadata = tmp_path / "base.spmat"
    filters_path = tmp_path / "filters.csv"
    workload_path = tmp_path / "workload.csv"
    write_xbin(base, np.asarray([[0, 0], [10, 0], [0, 10], [1, 1]], dtype=np.uint8), np.dtype("u1"))
    write_xbin(query, np.asarray([[0, 0], [10, 0]], dtype=np.uint8), np.dtype("u1"))
    write_spmat(metadata, [[7], [7], [9], [7, 9]])
    names = write_filters(filters_path, "tags", [7, 9])
    write_cartesian_workload(workload_path, names, [0, 1])
    filters = exact.load_filters(filters_path, "yfcc")
    pairs = exact.load_workload(workload_path, filters)
    bundle = exact.InputBundle(
        "yfcc", workload_path, filters_path, query, (base,), metadata, None, None, None
    )
    proof = exact.cartesian_workload_proof(pairs, filters)
    assert proof["eligible"] is True
    assert proof["observed_pair_count"] == 28

    legacy_rows, legacy_counts, legacy_execution = exact.exact_assigned_pairs_by_filter(
        bundle, pairs, filters, device=torch.device("cpu"), query_batch=2, chunk_rows=2, k=1
    )
    fast_rows, fast_counts, fast_execution = exact.exact_assigned_pairs_cartesian(
        bundle, pairs, filters, device=torch.device("cpu"), query_batch=2, chunk_rows=2, k=1
    )
    assert fast_rows == legacy_rows
    assert fast_counts == legacy_counts
    assert fast_execution["execution_path"] == "audited_cartesian_shared_query_gemm"
    assert fast_execution["base_scan_passes"] == 1
    assert fast_execution["gemm_passes"] == 2
    assert legacy_execution["execution_path"] == "legacy_by_filter"
    assert legacy_execution["base_scan_passes"] == 14

    output = tmp_path / "cartesian_truth.csv"
    manifest_path = tmp_path / "cartesian_truth.json"
    manifest = exact.run_generation(
        SimpleNamespace(
            dataset="yfcc",
            workload_csv=str(workload_path),
            filters_csv=str(filters_path),
            query_vector_source=str(query),
            base_vector_source=[str(base)],
            base_metadata_source=str(metadata),
            label_offsets_source=None,
            flat_labels_source=None,
            output_truth_csv=str(output),
            output_manifest=str(manifest_path),
            device="cpu",
            cuda_device=None,
            cpu_threads=1,
            query_batch=2,
            chunk_rows=2,
            k=1,
            overwrite=True,
        )
    )
    assert manifest["execution"]["execution_path"] == "audited_cartesian_shared_query_gemm"
    assert manifest["execution"]["gemm_passes"] == 2
    assert manifest["execution"]["cartesian_proof"]["eligible"] is True
    assert manifest["execution"]["cartesian_proof"]["canonical_pair_sha256"] == proof["canonical_pair_sha256"]
    assert manifest["exact_coverage"]["candidate_transfer"] == "all_base_vectors_for_shared_query_gemm"


def test_cartesian_proof_rejects_incomplete_and_duplicate_pairs(tmp_path: Path) -> None:
    filters_path = tmp_path / "filters.csv"
    names = write_filters(filters_path, "tags", [1, 2])
    filters = exact.load_filters(filters_path, "yfcc")
    complete = [
        exact.AssignedPair(request_no=index, query_no=query_id, query_id=query_id, filter_name=name)
        for index, (query_id, name) in enumerate((query_id, name) for query_id in (0, 1) for name in names)
    ]
    assert exact.cartesian_workload_proof(complete, filters)["eligible"] is True
    incomplete = complete[:-1]
    incomplete_proof = exact.cartesian_workload_proof(incomplete, filters)
    assert incomplete_proof["eligible"] is False
    assert "Cartesian" in incomplete_proof["reason"]
    duplicate_proof = exact.cartesian_workload_proof([*complete, complete[0]], filters)
    assert duplicate_proof["eligible"] is False
    assert "duplicate" in duplicate_proof["reason"]
