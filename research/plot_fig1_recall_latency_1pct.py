"""Figure 1(a): achieved Recall@10 vs latency at 1% selectivity."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research/results/fig1_iso_recall_1pct"
PAPER_DIR = ROOT / "paper/figures"

# 200k Amazon, M=16, efc=64, 100 queries, 1% selectivity, k=10.
# y = measured Recall@10 (not an iso-recall target).
FAISS = [  # HNSWlib-ACORN = FAISS IndexACORNFlat gamma=1
    (0.097, 0.189),
    (0.132, 0.231),
    (0.179, 0.280),
    (0.223, 0.327),
    (0.324, 0.428),
    (0.419, 0.480),
    (1.179, 0.590),
    (1.733, 0.653),
    (2.289, 0.692),
    (3.448, 0.754),
    (4.620, 0.783),
]
HNSW_SWEEP = [  # overfetch 100..4000; 4000 uses 100q×10 snapshot
    (0.056, 0.097),
    (0.233, 0.243),
    (0.300, 0.478),
    (0.577, 0.753),
    (1.161, 0.801),
    (4.464, 0.819),
]
PG_ACORN = [
    (12.398, 0.872),
    (9.102, 0.881),
    (7.010, 0.882),
    (6.108, 0.888),
    (6.819, 0.892),
    (5.988, 0.897),
    (6.232, 0.897),
    (8.376, 0.901),
    (8.399, 0.909),
    (8.239, 0.912),
]


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    best = -1.0
    out: list[tuple[float, float]] = []
    for lat, rec in sorted(points):
        if rec > best + 1e-6:
            out.append((lat, rec))
            best = rec
    return out


def load_pg_sweeping(path: Path) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts.append((float(row["latency_ms_mean"]), float(row["recall_mean"])))
    return pts


def annotate_hgap(ax, x_lo: float, x_hi: float, y: float, text: str) -> None:
    ax.annotate(
        "",
        xy=(x_hi, y),
        xytext=(x_lo, y),
        arrowprops={"arrowstyle": "<->", "color": "#8B0000", "linewidth": 1.4},
    )
    ax.text((x_lo * x_hi) ** 0.5, y + 0.028, text, color="#8B0000", ha="center", va="bottom", fontsize=9)


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(pg_sweep: list[tuple[float, float]], out_pdf: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    series = [
        ("HNSWlib-ACORN", FAISS, "#F58518", "o", "-"),
        ("HNSWlib-sweeping", HNSW_SWEEP, "#4C78A8", "o", "-"),
        ("PGVector-ACORN", PG_ACORN, "#F58518", "s", "--"),
        ("PGVector-sweeping", pg_sweep, "#4C78A8", "^", "--"),
    ]
    fronts: dict[str, list[tuple[float, float]]] = {}
    for label, pts, color, marker, ls in series:
        front = pareto(pts)
        fronts[label] = front
        if label == "PGVector-ACORN" and pts:
            ax.scatter(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=color,
                marker=marker,
                s=28,
                zorder=3,
                label=None,
            )
        if not front:
            continue
        ax.plot(
            [p[0] for p in front],
            [p[1] for p in front],
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=1.8,
            markersize=5.5,
            label=label,
        )

    # Matched-quality sweeping gap near Recall@10 ≈ 0.80.
    hnsw = next((p for p in fronts["HNSWlib-sweeping"] if p[1] >= 0.80), None)
    pg_sw = next((p for p in fronts["PGVector-sweeping"] if p[1] >= 0.80), None)
    if hnsw and pg_sw:
        gap = pg_sw[0] / hnsw[0]
        annotate_hgap(ax, hnsw[0], pg_sw[0], 0.80, f"{gap:.1f}×")

    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.05, 1.02)
    ax.grid(True, which="both", linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    for label, front in fronts.items():
        print(f"  {label}: " + ", ".join(f"{r:.2f}@{l:.2f}ms" for l, r in front))


def main() -> None:
    pg_sweep = load_pg_sweeping(OUT_DIR / "pg_sweeping_iterative.csv")
    plot(pg_sweep, OUT_DIR / "fig_intro_recall_latency_frontier.pdf")
    plot(pg_sweep, PAPER_DIR / "fig_intro_recall_latency_frontier.pdf")
    plot(pg_sweep, ROOT / "research/results/fig1_four_curve_m32/fig_intro_recall_latency_frontier.pdf")


if __name__ == "__main__":
    main()
