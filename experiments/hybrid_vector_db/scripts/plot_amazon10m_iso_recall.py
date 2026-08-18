#!/usr/bin/env python3
"""Plot Amazon-10M iso-recall Latency and Throughput figures.

Outputs (under ``results/hybrid_vector_db/amazon10m_iso_recall_plot/figures/``):

- ``amazon10m_iso_recall_latency.pdf``
- ``amazon10m_throughput_vs_recall.pdf``

Default inputs:

- ``amazon10m_iso_recall_pairs_long.csv`` — latency iso-recall curve
- ``amazon10m_throughput_vs_recall.csv`` — formal c16 throughput curve

Examples::

    python3 experiments/hybrid_vector_db/scripts/plot_amazon10m_iso_recall.py
    python3 experiments/hybrid_vector_db/scripts/plot_amazon10m_iso_recall.py --refresh
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = ROOT / "results/hybrid_vector_db/amazon10m_iso_recall_plot"
CALIBRATION_GLOB = "figure5_r41_amazon_frontier_secondary*"
MODE_STOCK = "original"
MODE_SQLENS = "design1_bloom_bfs_layout_d3"
DEFAULT_TARGETS = (
    0.75,
    0.80,
    0.85,
    0.90,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
)
# Keep Stock on both_off so the iso-recall latency curve does not jump to
# stock_strict (iterative_scan=strict_order). High targets that both_off cannot
# reach are marked closest_below_or_miss; use --stock-families both_off,stock_strict
# for an attainable-only refresh.
DEFAULT_STOCK_FAMILIES = ("both_off",)
ATTAINABLE_STOCK_FAMILIES = ("both_off", "stock_strict")
ARM_STYLE = {
    "stock": ("Stock pgvector", "#4C78A8", "o"),
    "sqlens": ("SQLens", "#B279A2", "D"),
}
FIGSIZE = (6.4*0.6, 4.8*0.6)
PRIMARY_LATENCY_FIELD = "latency_mean_ms"


class PlotError(RuntimeError):
    pass


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def geomean(values: Sequence[float]) -> float | None:
    positive = [value for value in values if value > 0.0]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def plot_modules() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError as exc:  # pragma: no cover
        raise PlotError("matplotlib is required") from exc
    return plt, mticker


def style() -> None:
    plt, _ = plot_modules()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise PlotError(f"missing data file: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise PlotError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_calibration_dirs(results_root: Path) -> list[Path]:
    return sorted(
        path
        for path in results_root.glob(CALIBRATION_GLOB)
        if path.is_dir() and path.name != "amazon10m_iso_recall_plot"
    )


def collect_operating_points(results_root: Path) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    pattern = re.compile(
        r"calibration_(.+)_ef(\d+)(?:_cap(\d+))?_profile_summary\.csv$"
    )
    for directory in discover_calibration_dirs(results_root):
        for profile in directory.glob("*_profile_summary.csv"):
            match = pattern.search(profile.name)
            if match is None:
                continue
            family = match.group(1)
            ef_search = int(match.group(2))
            cap = int(match.group(3)) if match.group(3) else None
            by_mode: dict[str, list[tuple[float, float]]] = defaultdict(list)
            with profile.open(newline="", encoding="utf-8") as source:
                for row in csv.DictReader(source):
                    by_mode[row["mode"]].append(
                        (
                            float(row["recall_mean"]),
                            float(row["end_to_end_mean_ms"]),
                        )
                    )
            for mode, arm in ((MODE_STOCK, "stock"), (MODE_SQLENS, "sqlens")):
                rows = by_mode.get(mode)
                if not rows:
                    continue
                recalls = [item[0] for item in rows]
                lats = [item[1] for item in rows]
                points.append(
                    {
                        "dataset": "amazon10m",
                        "arm": arm,
                        "mode": mode,
                        "family": family,
                        "ef_search": ef_search,
                        "sqlens_scan_cap": "" if cap is None else cap,
                        "n_filters": len(rows),
                        "recall_mean": round(mean(recalls) or 0.0, 6),
                        "latency_mean_ms": round(mean(lats) or 0.0, 3),
                        "latency_geomean_ms": round(geomean(lats) or 0.0, 3),
                        "batch_dir": directory.name,
                        "source_profile": str(profile.relative_to(results_root)),
                        "metric_source": "calibration_q2800_profile_summary",
                    }
                )
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for point in points:
        key = (
            point["arm"],
            point["family"],
            point["ef_search"],
            point["sqlens_scan_cap"],
        )
        previous = unique.get(key)
        if (
            previous is None
            or point["latency_geomean_ms"] < previous["latency_geomean_ms"]
        ):
            unique[key] = point
    return sorted(
        unique.values(),
        key=lambda item: (
            item["arm"],
            item["family"],
            item["ef_search"],
            str(item["sqlens_scan_cap"]),
        ),
    )


def pick_iso_point(
    points: Sequence[Mapping[str, Any]],
    target: float,
    *,
    families: Sequence[str] | None = None,
    latency_field: str = PRIMARY_LATENCY_FIELD,
) -> Mapping[str, Any] | None:
    eligible = [
        point
        for point in points
        if families is None or str(point["family"]) in set(families)
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda point: (
            0 if float(point["recall_mean"]) >= target - 1e-9 else 1,
            abs(float(point["recall_mean"]) - target),
            float(point[latency_field]),
        ),
    )


def build_iso_pairs(
    points: Sequence[Mapping[str, Any]],
    targets: Sequence[float],
    *,
    stock_families: Sequence[str] = DEFAULT_STOCK_FAMILIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stock = [point for point in points if point["arm"] == "stock"]
    sqlens = [point for point in points if point["arm"] == "sqlens"]
    wide: list[dict[str, Any]] = []
    long: list[dict[str, Any]] = []
    family_tag = "+".join(stock_families)
    metric_source = (
        "calibration_q2800_stock_both_off_only_mean_latency"
        if tuple(stock_families) == DEFAULT_STOCK_FAMILIES
        else f"calibration_q2800_stock_{family_tag}_mean_latency"
    )
    for target in targets:
        stock_point = pick_iso_point(stock, target, families=stock_families)
        sqlens_point = pick_iso_point(sqlens, target)
        if stock_point is None or sqlens_point is None:
            raise PlotError(
                f"missing Stock/SQLens calibration coverage for target {target}"
            )
        stock_lat = float(stock_point[PRIMARY_LATENCY_FIELD])
        sqlens_lat = float(sqlens_point[PRIMARY_LATENCY_FIELD])
        stock_meets = float(stock_point["recall_mean"]) >= target - 1e-9
        wide_row = {
            "dataset": "amazon10m",
            "target_recall": target,
            "stock_status": (
                "selected" if stock_meets else "closest_below_or_miss"
            ),
            "stock_recall": stock_point["recall_mean"],
            "stock_latency_geomean_ms": stock_point["latency_geomean_ms"],
            "stock_latency_mean_ms": stock_point["latency_mean_ms"],
            "stock_family": stock_point["family"],
            "stock_ef_search": stock_point["ef_search"],
            "stock_cap": stock_point["sqlens_scan_cap"],
            "stock_err": round(abs(float(stock_point["recall_mean"]) - target), 6),
            "sqlens_recall": sqlens_point["recall_mean"],
            "sqlens_latency_geomean_ms": sqlens_point["latency_geomean_ms"],
            "sqlens_latency_mean_ms": sqlens_point["latency_mean_ms"],
            "sqlens_family": sqlens_point["family"],
            "sqlens_ef_search": sqlens_point["ef_search"],
            "sqlens_cap": sqlens_point["sqlens_scan_cap"],
            "sqlens_err": round(abs(float(sqlens_point["recall_mean"]) - target), 6),
            "speedup_vs_stock": (
                round(stock_lat / sqlens_lat, 4) if sqlens_lat > 0 else ""
            ),
            "metric_source": metric_source,
            "note": (
                "Same q2800 calibration workload for every cell. "
                f"Stock families={family_tag}. Latency=mean over 14 filters."
            ),
        }
        wide.append(wide_row)
        for arm, prefix in (("stock", "stock"), ("sqlens", "sqlens")):
            long.append(
                {
                    "dataset": "amazon10m",
                    "target_recall": target,
                    "arm": arm,
                    "recall": wide_row[f"{prefix}_recall"],
                    "latency_geomean_ms": wide_row[f"{prefix}_latency_geomean_ms"],
                    "latency_mean_ms": wide_row[f"{prefix}_latency_mean_ms"],
                    "family": wide_row[f"{prefix}_family"],
                    "ef_search": wide_row[f"{prefix}_ef_search"],
                    "scan_cap": wide_row[f"{prefix}_cap"],
                    "abs_err_vs_target": wide_row[f"{prefix}_err"],
                    "metric_source": wide_row["metric_source"],
                    "stock_status": wide_row.get("stock_status", ""),
                }
            )
    return wide, long


def refresh_data_bundle(
    data_dir: Path,
    results_root: Path,
    targets: Sequence[float],
    *,
    stock_families: Sequence[str] = DEFAULT_STOCK_FAMILIES,
) -> dict[str, Any]:
    points = collect_operating_points(results_root)
    if not points:
        raise PlotError(f"no calibration points found under {results_root}")
    wide, long = build_iso_pairs(
        points,
        targets,
        stock_families=stock_families,
    )
    write_csv(data_dir / "amazon10m_calibration_operating_points.csv", points)
    write_csv(data_dir / "amazon10m_iso_recall_pairs.csv", wide)
    write_csv(data_dir / "amazon10m_iso_recall_pairs_long.csv", long)
    # Always also emit an attainable Stock curve (both_off + stock_strict).
    attainable_wide, attainable_long = build_iso_pairs(
        points,
        targets,
        stock_families=ATTAINABLE_STOCK_FAMILIES,
    )
    write_csv(
        data_dir / "amazon10m_iso_recall_pairs_attainable.csv",
        attainable_wide,
    )
    write_csv(
        data_dir / "amazon10m_iso_recall_pairs_attainable_long.csv",
        attainable_long,
    )
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "amazon10m",
        "targets": list(targets),
        "stock_families": list(stock_families),
        "n_operating_points": len(points),
        "n_iso_pairs": len(wide),
        "files": {
            "operating_points": "amazon10m_calibration_operating_points.csv",
            "iso_pairs_wide": "amazon10m_iso_recall_pairs.csv",
            "iso_pairs_long": "amazon10m_iso_recall_pairs_long.csv",
            "iso_pairs_attainable": "amazon10m_iso_recall_pairs_attainable.csv",
            "iso_pairs_attainable_long": (
                "amazon10m_iso_recall_pairs_attainable_long.csv"
            ),
            "throughput": "amazon10m_throughput_vs_recall.csv",
        },
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def save_fig(fig: Any, path: Path) -> None:
    plt, _ = plot_modules()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_iso_latency(long_rows: Sequence[Mapping[str, str]], output_dir: Path) -> Path:
    plt, mticker = plot_modules()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for arm, (label, color, marker) in ARM_STYLE.items():
        rows = sorted(
            (row for row in long_rows if row["arm"] == arm),
            key=lambda row: float(row["target_recall"]),
        )
        xs = [float(row[PRIMARY_LATENCY_FIELD]) for row in rows]
        ys = [float(row["recall"]) for row in rows]
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markersize=6.5,
            linewidth=1.8,
            label=label,
        )
        # for row in rows:
        #     ax.annotate(
        #         f"{float(row['target_recall']):.2f}",
        #         (
        #             float(row[PRIMARY_LATENCY_FIELD]),
        #             float(row["recall"]),
        #         ),
        #         textcoords="offset points",
        #         xytext=(4, 3),
        #         fontsize=8,
        #         color=color,
        #         alpha=0.85,
        #     )
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Recall@10")
    # ax.set_title("Amazon-10M iso-recall (Stock=both_off only)")
    ax.set_ylim(0.68, 1.005)
    ax.set_xlim(left=0.0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.grid(True, linewidth=0.45, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    path = output_dir / "amazon10m_iso_recall_latency.pdf"
    save_fig(fig, path)
    return path


def plot_throughput_vs_recall(
    throughput_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
) -> Path:
    if not throughput_rows:
        raise PlotError("throughput CSV has no rows")
    plt, mticker = plot_modules()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for arm, (label, color, marker) in ARM_STYLE.items():
        rows = sorted(
            (row for row in throughput_rows if row["arm"] == arm),
            key=lambda row: float(row["recall"]),
        )
        if not rows:
            continue
        ax.plot(
            [float(row["recall"]) for row in rows],
            [float(row["throughput_qps"]) for row in rows],
            color=color,
            marker=marker,
            markersize=6.5,
            linewidth=1.8,
            label=label,
        )
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Throughput (QPS)")
    ax.set_title("Amazon-10M Throughput vs Recall")
    ax.set_xlim(0.68, 1.005)
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.grid(True, linewidth=0.45, alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    path = output_dir / "amazon10m_throughput_vs_recall.pdf"
    save_fig(fig, path)
    return path


def parse_targets(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise PlotError("targets must be non-empty")
    if any(not (0.0 < value <= 1.0) for value in values):
        raise PlotError("targets must lie in (0, 1]")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/hybrid_vector_db",
        help="used only with --refresh",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to --data-dir/figures",
    )
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=DEFAULT_TARGETS,
        help="comma-separated recall targets used by --refresh",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rebuild latency CSVs from figure5_r41_amazon_frontier_secondary* profiles",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (data_dir / "figures")
    )
    if args.refresh:
        meta = refresh_data_bundle(data_dir, args.results_root.resolve(), args.targets)
        print(json.dumps({"status": "refreshed", **meta}, sort_keys=True))

    style()
    long_rows = read_csv(data_dir / "amazon10m_iso_recall_pairs_long.csv")
    throughput_rows = read_csv(data_dir / "amazon10m_throughput_vs_recall.csv")
    outputs = [
        plot_iso_latency(long_rows, output_dir),
        plot_throughput_vs_recall(throughput_rows, output_dir),
    ]
    print(
        json.dumps(
            {
                "status": "plotted",
                "data_dir": str(data_dir),
                "output_dir": str(output_dir),
                "figures": [str(path) for path in outputs],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotError as exc:
        raise SystemExit(f"plot error: {exc}") from exc
