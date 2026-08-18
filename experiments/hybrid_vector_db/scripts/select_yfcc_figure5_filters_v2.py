#!/usr/bin/env python3
"""Select YFCC Figure-5 filters with diverse selectivity, avoiding hub-tag OR traps.

Design goals (v2):
- Keep exactly 14 filters (Figure 5 formal 14x200 contract).
- Cover selectivity roughly uniformly on a log scale (~0.2% .. ~20%),
  without forcing exact Amazon-style 50/45/... band hits.
- Prefer mid-frequency AND predicates (`tags @> ARRAY[a,b]`) so filters are
  more specific than hub-tag OR unions that scatter across the CLIP graph.
- Exclude ultra-frequent hub tags from OR constructions.

Outputs a filters CSV compatible with build_figure5_frontier_workload.py /
figure5_external_exact_truth.py / run_figure5_frontier.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .prepare_yfcc_pgvector import spmat_fields
except ImportError:
    from prepare_yfcc_pgvector import spmat_fields

DATA_DIR = Path(
    os.environ.get(
        "YFCC10M_DATA_DIR",
        Path(os.environ.get("OOD_ANNS_DATA", "data/ood_anns")) / "YFCC10M",
    )
)
# Soft guidance only; final picks are log-spaced across the achievable range.
DEFAULT_N_FILTERS = 14


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_row_lists(
    indptr: np.memmap,
    indices: np.memmap,
    labels: list[int],
) -> dict[int, np.ndarray]:
    started = time.perf_counter()
    label_arr = np.asarray(labels, dtype=np.int32)
    hit_mask = np.isin(indices, label_arr, assume_unique=False)
    positions = np.flatnonzero(hit_mask)
    hit_labels = np.asarray(indices[positions], dtype=np.int32)
    rows = np.searchsorted(indptr, positions, side="right").astype(np.int64) - 1
    order = np.argsort(hit_labels, kind="stable")
    hit_labels = hit_labels[order]
    rows = rows[order].astype(np.int32, copy=False)
    out: dict[int, np.ndarray] = {}
    starts = np.r_[0, np.flatnonzero(np.diff(hit_labels)) + 1, len(hit_labels)]
    for start, end in zip(starts[:-1], starts[1:]):
        out[int(hit_labels[start])] = np.unique(rows[start:end])
    print(
        f"built {len(out)} row lists from {len(labels)} labels "
        f"in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return out


def _materialize(cand: dict[str, Any], soft_target: float) -> dict[str, Any]:
    labels = cand["labels"]
    if cand["kind"].startswith("and"):
        if len(labels) == 1:
            predicate = f"tags @> ARRAY[{labels[0]}]::int[]"
            atoms = f"sql:tags @> ARRAY[{labels[0]}]::int[]"
            name = f"tagand_{labels[0]}"
        else:
            predicate = (
                "tags @> ARRAY[" + ",".join(str(x) for x in labels) + "]::int[]"
            )
            atoms = "||".join(f"sql:tags @> ARRAY[{x}]::int[]" for x in labels)
            name = "tagand_" + "_".join(str(x) for x in labels)
    else:
        predicate = (
            "tags && ARRAY[" + ",".join(str(x) for x in labels) + "]::int[]"
        )
        atoms = "||OR||".join(f"sql:tags @> ARRAY[{x}]::int[]" for x in labels)
        name = "tagor_" + "_".join(str(x) for x in labels)
    return {
        "filter_name": name,
        "target_rate": f"{soft_target:g}",
        "actual_pct": f"{cand['pct']:.5f}",
        "expected_rows": str(cand["count"]),
        "predicate": predicate,
        "atoms": atoms,
        "kind": cand["kind"],
        "soft_target_pct": soft_target,
    }


def choose_filters_v2(
    nrow: int,
    ncol: int,
    indptr: np.memmap,
    indices: np.memmap,
    n_filters: int,
    *,
    and_hub_pct: float,
    or_hub_pct: float,
    min_selectivity_pct: float,
    max_selectivity_pct: float,
    pool_top: int,
    pair_pool: int,
) -> list[dict[str, Any]]:
    if n_filters != 14:
        raise SystemExit(f"Figure 5 requires 14 filters, got {n_filters}")

    freq = np.bincount(np.asarray(indices, dtype=np.int32), minlength=ncol)
    order = np.argsort(freq)[::-1]
    and_cut = int(nrow * (and_hub_pct / 100.0))
    or_cut = int(nrow * (or_hub_pct / 100.0))
    rare_cut = max(1, int(nrow * 0.0002))

    and_labels = [
        int(t)
        for t in order[:pool_top]
        if rare_cut <= int(freq[t]) < and_cut
    ]
    # OR may use slightly more frequent tags (still below hard hubs) to stretch
    # the high-selectivity end without bringing back top Flickr hubs (~30%+).
    or_labels = [
        int(t)
        for t in order[:pool_top]
        if rare_cut <= int(freq[t]) < or_cut
    ]
    if len(and_labels) < 30 or len(or_labels) < 20:
        raise SystemExit(
            f"tag pools too small and={len(and_labels)} or={len(or_labels)}; "
            f"relax hub thresholds"
        )
    print(
        f"nrow={nrow} and_pool(<{and_hub_pct:g}%)={len(and_labels)} "
        f"or_pool(<{or_hub_pct:g}%)={len(or_labels)} rare_cut={rare_cut}",
        flush=True,
    )

    need_labels = sorted(set(and_labels) | set(or_labels))
    row_lists = build_row_lists(indptr, indices, need_labels)

    candidates: list[dict[str, Any]] = []
    for label in and_labels:
        count = int(freq[label])
        candidates.append(
            {
                "kind": "and1",
                "labels": (label,),
                "count": count,
                "pct": 100.0 * count / nrow,
            }
        )

    pair_labels = and_labels[:pair_pool]
    print(f"enumerating AND pairs among top-{len(pair_labels)} and-pool tags", flush=True)
    for i, a in enumerate(pair_labels):
        a_rows = row_lists[a]
        for b in pair_labels[i + 1 :]:
            inter = np.intersect1d(a_rows, row_lists[b], assume_unique=True)
            count = int(inter.size)
            if count < rare_cut:
                continue
            candidates.append(
                {
                    "kind": "and2",
                    "labels": (a, b) if a < b else (b, a),
                    "count": count,
                    "pct": 100.0 * count / nrow,
                }
            )
        if (i + 1) % 25 == 0:
            print(f"  and-pair progress {i + 1}/{len(pair_labels)}", flush=True)

    print(
        f"enumerating OR pairs among top-{min(50, len(or_labels))} or-pool tags",
        flush=True,
    )
    or_pair_labels = or_labels[: min(50, len(or_labels))]
    for i, a in enumerate(or_pair_labels):
        a_rows = row_lists[a]
        for b in or_pair_labels[i + 1 :]:
            b_rows = row_lists[b]
            inter = np.intersect1d(a_rows, b_rows, assume_unique=True).size
            count = int(len(a_rows) + len(b_rows) - inter)
            candidates.append(
                {
                    "kind": "or2",
                    "labels": (a, b) if a < b else (b, a),
                    "count": count,
                    "pct": 100.0 * count / nrow,
                }
            )

    uniq: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for cand in candidates:
        key = (cand["kind"], cand["labels"])
        prev = uniq.get(key)
        if prev is None or cand["count"] < prev["count"]:
            uniq[key] = cand
    candidates = list(uniq.values())
    print(f"candidate predicates: {len(candidates)}", flush=True)

    achievable_min = min(c["pct"] for c in candidates)
    achievable_max = max(c["pct"] for c in candidates)
    min_pct = max(achievable_min, min_selectivity_pct)
    max_pct = min(achievable_max, max_selectivity_pct)
    if not 0.0 < min_pct < max_pct:
        raise SystemExit(
            "invalid requested selectivity range: "
            f"requested=[{min_selectivity_pct}, {max_selectivity_pct}] "
            f"achievable=[{achievable_min}, {achievable_max}]"
        )
    # Log-uniform coverage across the requested range.  Figure 5 does not
    # require exact selectivity bands, but it does require a representative
    # mixture; bounding the range avoids one pathological 0.05% predicate
    # dominating the dataset-level mean.
    log_targets = [
        float(x) for x in np.geomspace(min_pct, max_pct, n_filters)
    ]
    print(
        f"achievable pct=[{achievable_min:.4f}, {achievable_max:.4f}] "
        f"requested pct=[{min_pct:.4f}, {max_pct:.4f}] "
        f"log_targets={[round(t, 4) for t in log_targets]}",
        flush=True,
    )

    selected: list[dict[str, Any]] = []
    used_labels: set[tuple[int, ...]] = set()
    used_names: set[str] = set()
    for target in reversed(log_targets):  # high -> low
        def rank_candidate(c: dict[str, Any], t: float = target) -> tuple[Any, ...]:
            log_error = abs(
                np.log(max(c["pct"], 1e-9)) - np.log(max(t, 1e-9))
            )
            # Prefer containment predicates whenever they are within 10% of
            # the requested selectivity.  Exact closeness alone previously
            # selected OR pairs throughout the middle of the range, recreating
            # the weak tag/embedding-correlation workload this selector exists
            # to avoid.
            near_target = log_error <= np.log(1.10)
            return (
                0 if near_target else 1,
                0 if near_target and c["kind"].startswith("and") else 1,
                log_error,
                len(c["labels"]),
                c["labels"],
            )

        ordered = sorted(
            candidates,
            key=rank_candidate,
        )
        chosen = None
        for cand in ordered:
            if cand["labels"] in used_labels:
                continue
            row = _materialize(cand, target)
            if row["filter_name"] in used_names:
                continue
            # Keep selected selectivities from collapsing: require >=15% relative gap
            # from already chosen pcts when possible.
            if selected:
                rel = [
                    abs(cand["pct"] - float(s["actual_pct"]))
                    / max(cand["pct"], float(s["actual_pct"]))
                    for s in selected
                ]
                if min(rel) < 0.12 and abs(cand["pct"] - target) > 0.25 * target:
                    continue
            chosen = cand
            break
        if chosen is None:
            # Fallback without diversity gap.
            for cand in ordered:
                if cand["labels"] in used_labels:
                    continue
                row = _materialize(cand, target)
                if row["filter_name"] in used_names:
                    continue
                chosen = cand
                break
        if chosen is None:
            raise SystemExit(f"no unused candidate near log target {target}%")
        used_labels.add(chosen["labels"])
        row = _materialize(chosen, target)
        used_names.add(row["filter_name"])
        selected.append(row)
        print(
            f"log_target={target:.4f}% -> {row['filter_name']} "
            f"kind={chosen['kind']} actual={chosen['pct']:.4f}% "
            f"rows={chosen['count']}",
            flush=True,
        )

    selected = sorted(selected, key=lambda r: float(r["actual_pct"]), reverse=True)
    pcts = [float(r["actual_pct"]) for r in selected]
    print(
        f"selected selectivity span: min={min(pcts):.4f}% "
        f"median={sorted(pcts)[len(pcts)//2]:.4f}% max={max(pcts):.4f}%",
        flush=True,
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--filters-out",
        type=Path,
        default=Path(
            "results/hybrid_vector_db/yfcc10m_figure5_v2_filters_diverse_sel.csv"
        ),
    )
    parser.add_argument("--n-filters", type=int, default=DEFAULT_N_FILTERS)
    parser.add_argument(
        "--and-hub-pct",
        type=float,
        default=3.0,
        help="AND pool excludes tags at/above this frequency (%).",
    )
    parser.add_argument(
        "--or-hub-pct",
        type=float,
        default=8.0,
        help="OR pool excludes tags at/above this frequency (%); higher than AND to stretch coverage.",
    )
    parser.add_argument(
        "--min-selectivity-pct",
        type=float,
        default=0.2,
        help="Lower bound for the log-uniform filter mixture.",
    )
    parser.add_argument(
        "--max-selectivity-pct",
        type=float,
        default=14.0,
        help="Upper bound for the log-uniform filter mixture.",
    )
    parser.add_argument("--pool-top", type=int, default=400)
    parser.add_argument("--pair-pool", type=int, default=80)
    args = parser.parse_args()

    meta = args.data_dir / "base.metadata.10M.spmat"
    nrow, ncol, _, indptr, indices, _ = spmat_fields(meta)
    rows = choose_filters_v2(
        nrow,
        ncol,
        indptr,
        indices,
        int(args.n_filters),
        and_hub_pct=float(args.and_hub_pct),
        or_hub_pct=float(args.or_hub_pct),
        min_selectivity_pct=float(args.min_selectivity_pct),
        max_selectivity_pct=float(args.max_selectivity_pct),
        pool_top=int(args.pool_top),
        pair_pool=int(args.pair_pool),
    )
    # Public filters CSV fields only (Figure 5 contract).
    public = [
        {
            "filter_name": r["filter_name"],
            "target_rate": r["target_rate"],
            "actual_pct": r["actual_pct"],
            "expected_rows": r["expected_rows"],
            "predicate": r["predicate"],
            "atoms": r["atoms"],
        }
        for r in rows
    ]
    write_csv(args.filters_out, public)
    detail = args.filters_out.with_name(args.filters_out.stem + "_detail.csv")
    write_csv(detail, rows)
    print(f"wrote {args.filters_out}", flush=True)
    print(f"wrote {detail}", flush=True)


if __name__ == "__main__":
    main()
