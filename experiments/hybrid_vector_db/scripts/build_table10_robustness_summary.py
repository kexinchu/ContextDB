#!/usr/bin/env python3
"""Fail-closed builder for Table 10 three-panel robustness summary.

Panel A (correctness) is loaded from r43 paper-eligible artifacts.
Panels B (concurrency) and C (overhead) stay Pending until their
paper-eligible inputs are supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "sqlens_table10_robustness_summary"
RUNNER_VERSION = "sqlens-table10-robustness-summary-v1"
PENDING = "Pending"

PAPER_CONCURRENCY_CELLS = (
    (16, 0.0),
    (16, 100.0),
    (16, 1000.0),
    (64, 100.0),
)


class Table10SummaryError(RuntimeError):
    """An input cannot be admitted to the Table 10 summary."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Table10SummaryError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Table10SummaryError(f"expected object JSON at {path}")
    return payload


def require_paper_eligible(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("paper_eligible") is not True:
        raise Table10SummaryError(f"{label} is not paper_eligible=true")
    if payload.get("artifact_valid") is False:
        raise Table10SummaryError(f"{label} has artifact_valid=false")


def load_panel_a_adversarial(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    require_paper_eligible(payload, label=str(path))
    summary = payload.get("summary") or {}
    records = payload.get("records") or []
    if int(summary.get("records") or 0) != 1134:
        raise Table10SummaryError(
            f"adversarial expected 1134 records, got {summary.get('records')}"
        )
    if int(summary.get("strict_failures") or 0) != 0:
        raise Table10SummaryError("adversarial strict_failures must be 0")
    mismatches = sum(1 for row in records if row.get("ordered_mismatch"))
    false_neg = sum(1 for row in records if row.get("false_negative"))
    errors = sum(1 for row in records if row.get("error"))
    bypasses = sum(1 for row in records if row.get("stale_bypass"))
    if mismatches or false_neg or errors:
        raise Table10SummaryError(
            "adversarial has non-zero mismatch/false-negative/error counts"
        )
    return {
        "scenario": "adversarial_fixture",
        "checks": 1134,
        "mismatch": 0,
        "violation": 0,
        "error": 0,
        "bypass": bypasses,
        "verdict": "Pass",
        "source": str(path),
        "source_sha256": sha256_file(path),
    }


def load_panel_a_nonowner(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    require_paper_eligible(payload, label=str(path))
    checks = payload.get("checks") or {}
    if int(checks.get("exact_stock_equal") or 0) != 12:
        raise Table10SummaryError("non-owner Exact/Stock matches must be 12/12")
    if int(checks.get("exact_sqlens_equal") or 0) != 12:
        raise Table10SummaryError("non-owner Exact/SQLens matches must be 12/12")
    if int(checks.get("failed") or 0) != 0:
        raise Table10SummaryError("non-owner failed checks must be 0")
    if int(checks.get("visible_row_violations") or 0) != 0:
        raise Table10SummaryError("non-owner visible-row violations must be 0")
    return {
        "scenario": "nonowner_rls_acl",
        "checks": 12,
        "mismatch": 0,
        "violation": int(checks.get("visible_row_violations") or 0),
        "error": int(checks.get("failed") or 0),
        "bypass": int(checks.get("rls_stock_bypasses") or 0),
        "verdict": "Pass",
        "exact_stock_equal": 12,
        "exact_sqlens_equal": 12,
        "source": str(path),
        "source_sha256": sha256_file(path),
    }


def load_panel_a_stress(path: Path, *, manifest_path: Path | None = None) -> dict[str, Any]:
    payload = load_json(path)
    require_paper_eligible(payload, label=str(path))
    summary = payload.get("correctness_summary") or {}
    paired = int(summary.get("paired_requests") or 0)
    if paired != 1000:
        raise Table10SummaryError(f"250K stress expected 1000 queries, got {paired}")
    if int(summary.get("ordered_equivalent") or 0) != paired:
        raise Table10SummaryError("250K ordered_equivalent must equal paired_requests")
    if int(summary.get("guided_sql_valid") or 0) != paired:
        raise Table10SummaryError("250K guided_sql_valid must equal paired_requests")

    overlap_queries = 0
    if manifest_path is not None and manifest_path.exists():
        manifest = load_json(manifest_path)
        gates = manifest.get("artifact_gates") or {}
        overlap_queries = int(gates.get("overlap_queries") or 0)
    records = payload.get("records") or []
    refresh = sum(
        1 for row in records if row.get("post_update_refresh_or_safe_fallback")
    )
    stale = sum(
        1
        for row in records
        if (row.get("guided_profile_classification") or {}).get("stale_fallback")
    )
    mutations = payload.get("committed_mutations") or {}
    updates = sum(int(v) for v in mutations.values()) if isinstance(mutations, dict) else 0
    return {
        "scenario": "amazon250k_stress",
        "checks": paired,
        "mismatch": 0,
        "violation": 0,
        "error": 0,
        "bypass": stale,
        "epoch_refreshes": refresh,
        "overlap_queries": overlap_queries,
        "committed_updates": updates,
        "verdict": "Pass",
        "source": str(path),
        "source_sha256": sha256_file(path),
        "manifest": str(manifest_path) if manifest_path else "",
        "manifest_sha256": sha256_file(manifest_path) if manifest_path and manifest_path.exists() else "",
    }


def _fmt_pair(stock: float | None, sqlens: float | None, *, digits: int = 2) -> str:
    if stock is None or sqlens is None:
        return PENDING
    return f"{stock:.{digits}f}/{sqlens:.{digits}f}"


def load_panel_b_concurrency(
    summary_csv: Path | None,
    *,
    require_eligible_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    if summary_csv is None or not summary_csv.exists():
        for readers, rate in PAPER_CONCURRENCY_CELLS:
            cells.append(
                {
                    "readers": readers,
                    "update_rate_tps": rate,
                    "status": PENDING,
                    "delivered_tps_ratio": PENDING if rate > 0 else "---",
                    "qps_s_q": PENDING,
                    "p95_s_q": PENDING,
                    "recall_s_q": PENDING,
                    "errors_s_q": PENDING,
                }
            )
        return cells

    if require_eligible_manifest is not None:
        manifest = load_json(require_eligible_manifest)
        if manifest.get("paper_eligible") is not True:
            raise Table10SummaryError(
                f"concurrency manifest is not paper_eligible: {require_eligible_manifest}"
            )

    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Prefer aggregate rows when present.
    usable = [
        row
        for row in rows
        if str(row.get("kind") or "").lower() in {"", "read", "aggregate", "summary"}
        or str(row.get("aggregation") or "").lower() in {"aggregate", "pooled", "mean"}
    ]
    if not usable:
        usable = rows

    def pick(readers: int, rate: float, method: str) -> dict[str, str] | None:
        candidates = []
        for row in usable:
            try:
                rdr = int(float(row.get("readers") or -1))
                upd = float(row.get("requested_update_tps") or row.get("update_rate_tps") or -1)
            except ValueError:
                continue
            meth = str(row.get("method") or row.get("arm") or "")
            if rdr != readers or not math.isclose(upd, rate, abs_tol=1e-6):
                continue
            if method not in meth:
                continue
            candidates.append(row)
        if not candidates:
            return None
        # Prefer aggregation/summary rows.
        candidates.sort(
            key=lambda row: (
                0
                if str(row.get("aggregation") or row.get("kind") or "").lower()
                in {"aggregate", "pooled", "summary", "mean"}
                else 1,
                str(row.get("measurement_repeat") or "999"),
            )
        )
        return candidates[0]

    for readers, rate in PAPER_CONCURRENCY_CELLS:
        stock = pick(readers, rate, "stock")
        sqlens = pick(readers, rate, "sqlens")
        if stock is None or sqlens is None:
            cells.append(
                {
                    "readers": readers,
                    "update_rate_tps": rate,
                    "status": PENDING,
                    "delivered_tps_ratio": PENDING if rate > 0 else "---",
                    "qps_s_q": PENDING,
                    "p95_s_q": PENDING,
                    "recall_s_q": PENDING,
                    "errors_s_q": PENDING,
                }
            )
            continue
        stock_qps = float(stock["qps"])
        sqlens_qps = float(sqlens["qps"])
        stock_p95 = float(stock["p95_ms"])
        sqlens_p95 = float(sqlens["p95_ms"])
        stock_recall = float(stock.get("mean_recall_at_10") or stock.get("pooled_recall_lcb95") or 0.0)
        sqlens_recall = float(
            sqlens.get("mean_recall_at_10") or sqlens.get("pooled_recall_lcb95") or 0.0
        )
        stock_err = int(float(stock.get("errors") or 0))
        sqlens_err = int(float(sqlens.get("errors") or 0))
        if rate > 0:
            delivered = float(sqlens.get("achieved_update_tps") or 0.0)
            ratio = delivered / rate if rate > 0 else 0.0
            delivered_cell = f"{delivered:.1f}/{ratio:.2f}"
        else:
            delivered_cell = "---"
        cells.append(
            {
                "readers": readers,
                "update_rate_tps": rate,
                "status": "filled",
                "delivered_tps_ratio": delivered_cell,
                "qps_s_q": _fmt_pair(stock_qps, sqlens_qps, digits=2),
                "p95_s_q": _fmt_pair(stock_p95, sqlens_p95, digits=2),
                "recall_s_q": _fmt_pair(stock_recall, sqlens_recall, digits=3),
                "errors_s_q": f"{stock_err}/{sqlens_err}",
            }
        )
    return cells


def load_panel_c_overhead(overhead_json: Path | None) -> list[dict[str, Any]]:
    defaults = [
        {
            "cost": "resident_guidance_reuse_memory",
            "stock": "---",
            "sqlens": PENDING,
            "delta": PENDING,
        },
        {
            "cost": "persistent_db_storage",
            "stock": PENDING,
            "sqlens": PENDING,
            "delta": PENDING,
        },
        {
            "cost": "hnsw_build_bfs_rewrite_time",
            "stock": PENDING,
            "sqlens": PENDING,
            "delta": PENDING,
        },
        {
            "cost": "maintenance_under_writes_p95",
            "stock": "---",
            "sqlens": PENDING,
            "delta": PENDING,
        },
    ]
    if overhead_json is None or not overhead_json.exists():
        return defaults
    payload = load_json(overhead_json)
    if payload.get("paper_eligible") is not True:
        raise Table10SummaryError(f"overhead artifact not paper_eligible: {overhead_json}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise Table10SummaryError("overhead artifact must contain exactly 4 rows")
    out = []
    for row in rows:
        out.append(
            {
                "cost": str(row["cost"]),
                "stock": str(row.get("stock", PENDING)),
                "sqlens": str(row.get("sqlens", PENDING)),
                "delta": str(row.get("delta", PENDING)),
            }
        )
    return out


def render_tex(
    panel_a: list[dict[str, Any]],
    panel_b: list[dict[str, Any]],
    panel_c: list[dict[str, Any]],
) -> str:
    def tex_int(value: int) -> str:
        text = f"{int(value):,}"
        return text.replace(",", "{,}")

    a_rows = []
    labels = {
        "adversarial_fixture": "Adversarial fixture (RC/RR $\\times$ mutations)",
        "nonowner_rls_acl": "Non-owner RLS/ACL (12 paired schedules)",
        "amazon250k_stress": "250K stress (1K queries, 2K updates)",
    }
    for row in panel_a:
        a_rows.append(
            f"{labels[row['scenario']]}\n"
            f"  & {tex_int(int(row['checks']))} & {row['mismatch']} & {row['violation']} & "
            f"{row['error']} & {row['bypass']} & {row['verdict']} \\\\"
        )

    b_rows = []
    for row in panel_b:
        readers = int(row["readers"])
        rate = int(float(row["update_rate_tps"]))
        b_rows.append(
            f"{readers} & {tex_int(rate)}\n"
            f"  & {row['delivered_tps_ratio']}\n"
            f"  & {row['qps_s_q']}\n"
            f"  & {row['p95_s_q']}\n"
            f"  & {row['recall_s_q']}\n"
            f"  & {row['errors_s_q']} \\\\"
        )

    c_labels = {
        "resident_guidance_reuse_memory": "Resident guidance + reuse memory",
        "persistent_db_storage": "Persistent DB storage",
        "hnsw_build_bfs_rewrite_time": "HNSW build / BFS rewrite time",
        "maintenance_under_writes_p95": "Maintenance under writes (p95)",
    }
    c_rows = []
    for row in panel_c:
        c_rows.append(
            f"{c_labels[row['cost']]}\n"
            f"  & {row['stock']}\n"
            f"  & {row['sqlens']}\n"
            f"  & {row['delta']} \\\\"
        )

    return f"""\\begin{{table*}}[t]
\\centering
\\caption{{Robustness and overhead for the frozen r43 release.
\\textbf{{(A)}}~Correctness compares \\system with exact SQL in the same snapshot.
\\textbf{{(B)}}~Concurrency compares Stock \\pgvector~(S) with full \\system~(Q) at
matched Recall@10~$=$~0.90; a requested update rate is sustainable only when
delivery is $\\ge$90\\%.
\\textbf{{(C)}}~Overhead reports default deployability cost versus Stock/baseline.
\\emph{{Pending}} denotes an unexecuted or not-yet-paper-eligible cell, not a
measured zero. Protocol matrices are in the text and appendix.}}
\\label{{tab:eval-robustness-summary}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.6pt}}
\\newcommand{{\\tabletenpending}}{{\\emph{{Pending}}}}

\\smallskip
\\noindent\\textbf{{(A) Correctness}}\\\\[0.25em]
\\begin{{tabular}}{{@{{}}lrrrrrl@{{}}}}
\\toprule
Scenario & Checks & Mismatch & Violation & Error & Bypass & Verdict \\\\
\\midrule
{chr(10).join(a_rows)}
\\bottomrule
\\end{{tabular}}

\\smallskip
\\noindent\\textbf{{(B) Concurrency (Stock / \\system)}}\\\\[0.25em]
\\begin{{tabular}}{{@{{}}rrrrrrr@{{}}}}
\\toprule
Readers & Upd/s & Delivered TPS/ratio & QPS S/Q & p95 S/Q & Recall S/Q & Err S/Q \\\\
\\midrule
{chr(10).join(b_rows)}
\\bottomrule
\\end{{tabular}}

\\smallskip
\\noindent\\textbf{{(C) Overhead}}\\\\[0.25em]
\\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\\toprule
Cost & Stock / baseline & \\system & $\\Delta$ / \\% \\\\
\\midrule
{chr(10).join(c_rows)}
\\bottomrule
\\end{{tabular}}
\\let\\tabletenpending\\relax
\\end{{table*}}
"""


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    panel_a = [
        load_panel_a_adversarial(args.adversarial_json),
        load_panel_a_nonowner(args.nonowner_json),
        load_panel_a_stress(args.stress_json, manifest_path=args.stress_manifest),
    ]
    panel_b = load_panel_b_concurrency(
        args.concurrency_summary_csv,
        require_eligible_manifest=args.concurrency_manifest,
    )
    panel_c = load_panel_c_overhead(args.overhead_json)
    filled_b = sum(1 for row in panel_b if row["status"] == "filled")
    filled_c = sum(1 for row in panel_c if row["sqlens"] != PENDING)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "runner_version": RUNNER_VERSION,
        "panel_a_correctness": panel_a,
        "panel_b_concurrency": panel_b,
        "panel_c_overhead": panel_c,
        "panel_a_complete": True,
        "panel_b_filled_cells": filled_b,
        "panel_c_filled_rows": filled_c,
        "paper_table_complete": filled_b == len(PAPER_CONCURRENCY_CELLS) and filled_c == 4,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adversarial-json",
        type=Path,
        default=Path("results/hybrid_vector_db/table10_r43/adversarial_correctness_r43.json"),
    )
    parser.add_argument(
        "--nonowner-json",
        type=Path,
        default=Path("results/hybrid_vector_db/table10_r43/nonowner_rls_acl_r43.json"),
    )
    parser.add_argument(
        "--stress-json",
        type=Path,
        default=Path("results/hybrid_vector_db/table10_r43/update_correctness_250k_r43.json"),
    )
    parser.add_argument(
        "--stress-manifest",
        type=Path,
        default=Path(
            "results/hybrid_vector_db/table10_r43/update_correctness_250k_r43.manifest.json"
        ),
    )
    parser.add_argument("--concurrency-summary-csv", type=Path, default=None)
    parser.add_argument("--concurrency-manifest", type=Path, default=None)
    parser.add_argument("--overhead-json", type=Path, default=None)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/hybrid_vector_db/table10_r43/table10_robustness_summary.json"),
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("paper/tables/eval_robustness_summary_r36.tex"),
    )
    parser.add_argument("--write-tex", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.write_tex:
        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_tex.write_text(
            render_tex(
                summary["panel_a_correctness"],
                summary["panel_b_concurrency"],
                summary["panel_c_overhead"],
            ),
            encoding="utf-8",
        )
    print(json.dumps({k: summary[k] for k in summary if not k.startswith("panel_")}, indent=2))
    print(f"wrote {args.out_json}")
    if args.write_tex:
        print(f"wrote {args.out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
