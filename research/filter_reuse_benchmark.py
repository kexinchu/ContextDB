from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def repeat_stats(keys: Iterable[str]) -> dict[str, float | int]:
    values = [str(k) for k in keys]
    counts = Counter(values)
    n = len(values)
    repeated_requests = sum(c for c in counts.values() if c > 1)
    cache_hits = sum(c - 1 for c in counts.values() if c > 1)
    return {
        "requests": n,
        "unique": len(counts),
        "repeated_request_ratio": repeated_requests / n if n else 0.0,
        "cache_hit_upper_bound_ratio": cache_hits / n if n else 0.0,
        "largest_group": max(counts.values()) if counts else 0,
    }


def workload_row(workload: str, source: str, rows: list[dict[str, str]], filter_keys: list[str], vector_key: str) -> dict[str, object]:
    def joined(row: dict[str, str], keys: list[str]) -> str:
        return "|".join(row.get(k, "") for k in keys)

    filter_stats = repeat_stats(joined(r, filter_keys) for r in rows)
    vector_stats = repeat_stats(r.get(vector_key, "") for r in rows)
    return {
        "workload": workload,
        "source": source,
        "requests": filter_stats["requests"],
        "filter_unique": filter_stats["unique"],
        "filter_repeated_request_ratio": filter_stats["repeated_request_ratio"],
        "filter_cache_hit_upper_bound_ratio": filter_stats["cache_hit_upper_bound_ratio"],
        "filter_largest_group": filter_stats["largest_group"],
        "vector_unique": vector_stats["unique"],
        "vector_repeated_request_ratio": vector_stats["repeated_request_ratio"],
        "vector_cache_hit_upper_bound_ratio": vector_stats["cache_hit_upper_bound_ratio"],
        "vector_largest_group": vector_stats["largest_group"],
    }


def amazon_trace_row() -> dict[str, object]:
    rows = read_csv(ROOT / "results/hybrid_vector_db/amazon_c4_trace_cache_detail.csv")
    return workload_row(
        "Amazon-C4 shopping trace",
        "real trace",
        rows,
        ["sql_fine"],
        "vector_fine",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("research/results/filter_reuse_benchmark_summary.csv"))
    args = parser.parse_args()

    rows = [
        amazon_trace_row(),
        workload_row(
            "MS MARCO ACL replay",
            "benchmark replay",
            read_csv(ROOT / "research/late_bound_visibility/results/msmarco_security_killtest_1m_q100.csv"),
            ["user_id"],
            "qid",
        ),
        workload_row(
            "Enron visibility replay",
            "benchmark replay",
            read_csv(ROOT / "research/late_bound_visibility/results/enron_visibility_benchmark.csv"),
            ["user_id"],
            "query_id",
        ),
        workload_row(
            "10M complex visibility replay",
            "benchmark replay",
            read_csv(ROOT / "research/late_bound_visibility/results/complex_visibility_10m_q20.csv"),
            ["user_label", "user_id"],
            "query_id",
        ),
    ]
    write_csv(ROOT / args.out, rows)
    print(f"wrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
