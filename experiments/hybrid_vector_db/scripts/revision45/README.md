# Paper-facing revision screens

These runners produce Table 6, Figure 5, Table 7, and Table 13 cells.
They write under `results/hybrid_vector_db/revision45/`.

| Script | Paper | What it measures |
|---|---|---|
| `run_b1_sql_first_q1k.py` | Table 6 | Stock vs VisGuide vs SQL-first |
| `run_b2_join_warm.sh` | Figure 5 | Four SQL shapes with a resident fragment |
| `run_c1_acorn_amazon14.sh` / `run_q3_acorn_matched.py` | Table 7 | Sweeping ACORN vs VisGuide+BFS |
| `run_q3_acorn_aligned.py` | Table 7 | Aligned ACORN oracles |
| `run_c2_failopen_write_sweep.sh` | Table 13 | Fail-open delivery at 16 readers |

Default is dry-run. Pass `--execute` to open PostgreSQL.
Need `start_r44_amazon.sh` or `start_amazon_table10_r43.sh`, plus `PGPASSWORD`.
