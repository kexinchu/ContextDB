from __future__ import annotations

import struct
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from experiments.hybrid_vector_db.scripts import (
    prepare_figure5_external_queries as prepare,
)


def test_cohort_is_disjoint_q200_plus_q10k() -> None:
    rows = prepare.cohort_rows()

    assert len(rows) == 10_200
    assert rows[0] == {
        "query_no": 0,
        "query_id": 10_000,
        "query_split": "calibration",
    }
    assert rows[199]["query_id"] == 10_199
    assert rows[200]["query_id"] == 0
    assert rows[-1]["query_id"] == 9_999
    calibration = {int(row["query_id"]) for row in rows[:200]}
    measurement = {int(row["query_id"]) for row in rows[200:]}
    assert not calibration & measurement
    assert len(measurement) == 10_000


def test_write_fbin_atomic_preserves_float32_matrix(tmp_path: Path) -> None:
    path = tmp_path / "query.fbin"
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)

    prepare.write_fbin_atomic(path, vectors)

    with path.open("rb") as source:
        rows, dimensions = struct.unpack("<ii", source.read(8))
        observed = np.frombuffer(source.read(), dtype="<f4").reshape(rows, dimensions)
    assert (rows, dimensions) == vectors.shape
    np.testing.assert_array_equal(observed, vectors)
    assert not path.with_suffix(".fbin.tmp").exists()


def test_ensure_query_ids_fails_closed_on_missing_rows() -> None:
    cursor = mock.MagicMock()
    cursor.fetchone.return_value = (10_199, 0, 10_199, 10_199)

    with pytest.raises(prepare.QueryPreparationError, match="coverage mismatch"):
        prepare.ensure_query_ids(cursor, "public.queries", 10_200)
