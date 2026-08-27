# Revision-45 screens (not paper-eligible)

These runners implement Phase B and C of the 4.5 plan. They write only
under `results/hybrid_vector_db/revision45/`. Do not copy numbers into
`paper/` until a freeze says so.

| Script | Plan item | What it measures |
|---|---|---|
| `run_b1_sql_first_q1k.py` | B1 | 14 Amazon row-local atoms: stock vs VisGuide vs SQL-first (ran q50) |
| `run_b2_join_warm.sh` | B2 | Figure 5 four SQL shapes with a resident `grocery_helpful` fragment |
| `run_c1_acorn_amazon14.sh` | C1 | pgvector `acorn1` vs stock vs `safe_guided` on four Amazon atoms (q50) |
| `run_c2_failopen_write_sweep.sh` | C2 | Fail-open path at 16 readers × {0,10,25} upd/s, 1000 requests, 6 repeats |

Every script supports a default dry-run. Pass `--execute` to open PostgreSQL.
Each result directory gets `paper_eligible: false` in its manifest.

Prerequisites match the Amazon Table-10 instance: `start_amazon_table10_r43.sh`,
port 55437, lock `.pg55437_experiment.lock`.
