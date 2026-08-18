# SQLens SIGMOD/VLDB Experiment Plan

Last updated: 2026-08-02

This document is the authoritative experiment plan for the SQLens paper. It is
organized around paper claims, not around the chronological order in which
experiments happened. Historical debugging notes remain recoverable from Git
and result manifests; they do not define the current protocol.

## 1. Paper Claim

SQLens optimizes SQL-native filtered vector search inside PostgreSQL and
pgvector without moving SQL, MVCC, access-control, or final validation semantics
into an external vector engine.

The evaluation must support four claims:

1. At matched Recall@10, full SQLens reduces end-to-end latency and improves
   throughput relative to stock PostgreSQL + pgvector.
2. SQLens remains effective when visibility is defined by joins, RLS/ACL, and
   temporal predicates rather than only by row-local payload columns.
3. D1, D2, and D3 address complementary costs: invalid candidate validation,
   physical page locality, and repeated predicate-state construction.
4. SQLens preserves SQL-visible results under updates and concurrency with
   acceptable build, storage, memory, and invalidation overhead.

The headline system is full SQLens (`D1+D2+D3`). D1 alone must not be presented
as the complete system.

## 2. Motivation and Evaluation Boundary

Motivation establishes that the problems exist:

- stock pgvector returns many SQL-invalid TIDs for selective predicates;
- graph traversal and physical index-page placement have poor locality;
- predicate/visibility fragments repeat more often than exact query vectors.

Evaluation measures whether SQLens solves those problems:

- end-to-end latency and measured throughput at matched recall;
- reduction in returned TIDs, heap fetches, SQL validation, and page I/O;
- adaptation, materialization, reuse, and break-even behavior;
- correctness and overhead under SQL-native queries, updates, and concurrency.

Stock-only problem-existence figures belong in Motivation. Treatment-effect
counters that explain a SQLens speedup remain in Evaluation.

## 3. Evaluation Research Questions

### RQ1: End-to-End Performance and Frontiers

At the same Recall@10, how much does full SQLens improve end-to-end latency over
stock PostgreSQL + pgvector, and does it improve the quality-cost frontier
rather than merely selecting a lower-recall operating point?

- Datasets: Amazon-10M, YFCC-10M, LAION-25M.
- Workload: 14 meaningful predicates per dataset, approximately 50% to 0.2%
  selectivity.
- Targets: Recall@10 = 0.90, 0.95, and 0.99.
- Headline comparison: stock pgvector versus full SQLens.
- Metrics: achieved recall, mean, p50, p95, p99 latency, paired 95% confidence
  intervals, win count, and geometric-mean speedup.
- Use at least ten distinct configurations per arm and dataset.
- Latency uses one 10,000-request mixed trace per dataset/configuration, not
  10,000 requests per predicate.
- Throughput uses clients = 1, 4, 8, 16, 32, and 64.
- QPS is `completed requests / barrier wall time`; it is never derived from
  inverse mean latency.
- Report CPU, PostgreSQL backend CPU, error rate, and block-device/relation I/O
  alongside p95 and p99.

### RQ2: SQL-Native Visibility

Does SQLens retain its advantage when the filter is defined by relational and
transactional semantics?

Amazon-10M contains three required workload families:

| Workload | SQL semantics |
|---|---|
| `acl_only` | product-dimension join, principal grant join, and PostgreSQL RLS |
| `grant_temporal_selectivity` | ACL/RLS plus grant validity interval |
| `fact_temporal_selectivity` | fact/product join plus source-row validity interval |

Each request must be one normal PostgreSQL hybrid SQL statement. Python-generated
allow-lists cannot be reported as SQL-native execution. Results use independent,
exact SQL-valid top-10 ground truth.

### RQ3: Component Analysis and Ablation

What is the cumulative contribution of VisGuide, Locality, and FragReuse, and
which measured mechanism explains each increment?

- Main-paper scope: Amazon-10M only.
- Arms: Stock, VisGuide, VisGuide+Locality, and SQLens.
- The existing Amazon-10M ablation is retained; it is not a P0 rerun.
- Before publication, regenerate the paper table from authoritative source CSVs
  and resolve any cross-table numeric mismatch. This is an artifact audit and
  table-generation task, not a new experiment.
- Per-predicate rows remain in the paper or appendix according to space.
- VisGuide: visited nodes, distance computations, returned TIDs, heap fetches, SQL
  qualifier calls, and guidance skips.
- Locality: same-logical-graph proof, page span/runs, ReadBuffer hits/misses and service
  time, warm latency, cold-I/O sensitivity, rewrite time, and storage overhead.
- FragReuse: empty-cache start, probes, online materializations, reuses, evictions,
  phase shift, cumulative break-even, p95/p99, and fresh-backend persistent
  reload.

YFCC and LAION ablation results are supporting/appendix evidence and are not a
submission blocker.

### RQ4: Correctness, Concurrency, and Deployability

Can SQLens preserve PostgreSQL-visible results and remain useful under load?

- Read Committed and Repeatable Read.
- Commit/rollback, insert, delete, predicate-crossing update, vector update,
  truncate, and TID reuse.
- Requested update rates = 0, 10, 100, and 1000 TPS.
- Read clients = 1, 4, 8, 16, 32, and 64.
- Report ordered-ID/distance mismatches, delivered update TPS, delivery ratio,
  read QPS, p95/p99, invalidations, rebuild/reactivation latency, stale bypass,
  metadata size, cache size, index size, and build/rewrite time.

## 4. Main-Paper Figure and Table Layout

| Position | Form | Content |
|---|---|---|
| Table 1 | Setup table | datasets, rows, dimensions, predicates, queries, table/index size, GT |
| Figure 5 | 2 x 3 panels | recall-latency and measured throughput-recall for three datasets |
| Table 2 | Matched-recall summary | dataset x target, stock/full SQLens latency, p95/p99, recall, QPS, speedup |
| Figure 6 | 1 x 3 SQL-native panels | ACL/RLS, grant-time, and fact-time end-to-end latency versus actual combined selectivity |
| Table 3 | Component analysis and ablation | Stock, VisGuide, VisGuide+Locality, SQLens, recall and speedup |
| Table 4 | Robustness/overhead | updates, concurrency, correctness, memory, storage, build cost |

Appendix material:

- all 14 per-predicate matched-recall rows;
- complete baseline parameter grids and unattainable cells;
- complete SQL-native per-cell latency, recall, counters, plans, and six
  boolean-complex workload variants;
- YFCC/LAION ablation;
- D2 cold-I/O and ReadBuffer detail;
- cache-budget and predicate-complexity sensitivity;
- full update/concurrency matrix.

Do not allocate a main-paper figure to calibration scores, fixed-`ef_search`
comparisons, historical q400 results, or derived QPS.

## 5. Unified Measurement Contract

### 5.1 Release identity

The current frozen release is r41. r36 artifacts remain immutable historical
evidence and must not be relabeled as r41:

- build ID:
  `sqlens-v16-distance-aware-route-budget-ef500k-20260801-r41`
- `vector.so` SHA256:
  `8f53226d35cae28d4e1b6926b13b01fa01fd1f6720c5f57c96c7886905f5eaf0`
- release contract:
  `experiments/hybrid_vector_db/configs/p0_release_contract_r41.json`

Paper artifacts must verify the server-side build ID, installed `vector.so`
digest, table/index OID and relfilenode, selected HNSW index, filter/query/GT
hashes, query split, and EXPLAIN plan.

### 5.2 Recall and configuration selection

- `k=10`, exact tie-aware SQL-valid top-10 ground truth.
- Stock, full SQLens, and every approximate baseline tune independently.
- Calibration and final measurement queries are disjoint.
- The main protocol uses one global configuration per
  `(dataset, target, method)`. It does not retune per predicate.
- Formal calibration is the complete Cartesian product of the 200 calibration
  queries and 14 predicates: 2,800 requests and exactly 200 observations per
  predicate. The historical q200 mixed trace is audit-only.
- A configuration qualifies only if both its aggregate Recall@10 LCB95 and
  every predicate's Recall@10 LCB95 meet the target. Among qualifying
  configurations, select the lowest measured calibration cost.
- Predicate-conditioned tuning, if reported, is a separate appendix result.
  It must give Stock and every baseline the same per-predicate tuning budget
  and include selector overhead.
- A method that cannot reach a target after exhausting its required grid is
  marked `unattainable_on_grid`; it is not compared as though it met the target.
- Final paper latency comes only from held-out end-to-end requests. Calibration
  scores are never plotted as final latency.

### 5.3 Final trace

- Use a frozen 10,000-request mixed trace per dataset.
- Queries q0--q199 are reserved for screening/calibration/confirmation;
  q200--q10199 are measurement queries.
- Methods use seeded paired/interleaved scheduling.
- D3 begins each repeat with an empty namespace. Probe, materialization, reuse,
  and activation costs are included.
- Warm-cache is the main result. Cold-I/O is a labeled mechanism stress test.

### 5.4 Expedited per-predicate evidence

The August 2 expedited campaign requested by the author is a distinct result
class. It uses one held-out 5,000-unique-query mixed trace, one paired/interleaved
repeat, and independently tuned per-predicate settings for both arms. It is
useful for filling and debugging target-matched rows quickly, but it is not the
global-configuration q10K/r3 protocol above and is therefore not automatically
`paper_eligible` for the headline table. Promotion requires either completing
the formal protocol or explicitly revising the paper contract and rerunning all
datasets consistently.

The r41 LAION-25M Recall@10 approximately 0.90 artifact is:

`results/hybrid_vector_db/table6_r41_laion_target090_per_filter_q5k/`
`laion_target090_per_filter_paired_q5k.csv`

It contains 5,000 unique queries per arm, all 14 predicates, 10,000 rows, zero
errors, exact runtime identity and CPU-affinity evidence, and a complete online
D3 lifecycle. Stock/SQLens mean recall is 0.9084/0.9092; workload-weighted mean
latency is 335.02/290.65 ms; per-predicate latency geomean is 113.56/108.75 ms
(1.044x); and SQLens wins 11/14 predicates. Per-predicate recall ranges from
0.8957 to 0.9228, with at most 0.0089 difference between the two arms on a
predicate. This is valid expedited evidence, not a strict every-predicate
0.900 result.

The raw rows preserve every effective setting, but this run predates the plan
schema that separately records configured per-filter overrides versus global
adaptive routing. Its current evidence-inventory classification is therefore
`diagnostic` until a SHA-bound provenance supplement or a rerun with the new
`search_configuration` plan block proves the tuning scope. This does not change
the measured values; it prevents them from being promoted under an ambiguous
configuration contract.

### 5.4 Statistical reporting

- Latency: mean, p50, p95, p99, standard deviation, and paired 95% CI.
- Recall: aggregate mean Recall@10, per-predicate mean and LCB95, minimum
  predicate recall, and target delta. Every predicate must pass in every final
  repeat for a fixed-target cell to enter the paper.
- Throughput: measured completed QPS and 95% CI.
- Each plot point binds to raw request rows and a complete manifest.

## 6. Baselines

| Baseline | Paper role | Required treatment |
|---|---|---|
| Official upstream pgvector 0.8.2 | primary DB baseline | auto planner and forced HNSW reported separately |
| SQLens binary, all features disabled | instrumentation overhead control | same table, SQL, index, and query trace |
| Indexed SQL-first exact | exact DB baseline | one PostgreSQL statement; offline GT time is not substituted |
| FAISS HNSW allow-list | standalone FVS baseline | report materialization, transfer, search, and cached-list control |
| Weaviate 1.38 | production payload-filter baseline | independently tune ACORN, sweeping, cutoff, and ef |
| Full SQLens | proposed system | D1+D2+D3 with online D3 costs included |

FAISS and Weaviate are compared only on predicates they can faithfully express.
Join/RLS/temporal semantics are a scope boundary, not a claim that PostgreSQL
universally outperforms vector-native systems.

## 7. Existing Evidence Inventory

### 7.1 Ready for the paper

- D2 Amazon-10M same-graph warm/cold locality:
  `amazon10m_r30_d2_cache_isolated_warm_q100r5_retry3_20260722.csv.manifest.json`
  and
  `amazon10m_r33_d2_cache_isolated_cold14_q1r5_distinct_20260727.csv.manifest.json`.
- D3 Amazon-10M q10K online adaptation:
  `amazon10m_d3_adaptation_lifecycle_r29_formal_q10k_clean_manifest.json`.
- Amazon-10M component ablation is retained for Table 3, subject to
  authoritative-source and table-generation audit.

D2/D3 use older release identities. Before final submission, audit whether the
relevant implementation paths changed. Rerun only the affected mechanism if
behavioral equivalence cannot be established.

### 7.2 Complete foundations, not final performance results

- Three datasets and their 14-filter workload definitions.
- Row-local exact SQL-valid GT and frozen q10K traces.
- Amazon SQL-native q200 GT for 3 workloads x 14 filters:
  `amazon10m_sql_native_exact_truth_valid_embeddings/amazon10m_sql_native_exact_truth_manifest.json`.
- r36 Figure 5 calibration: 146 base cells + 36 stock-cap cells + 33 SQLens
  target-extension cells. These are configuration-selection evidence and are
  explicitly not paper latency.
- The old Weaviate Amazon-10M run has 41 of 42 target cells final-confirmed,
  but its filter and GT hashes predate the current corpus import. It is useful
  only as protocol/debugging evidence; all current-protocol cells must rerun.
- Existing SQL-first/FAISS combined artifacts are protocol-reuse candidates,
  not yet current-P0 paper evidence.
- r41 expedited per-predicate q5K results exist for Amazon, YFCC, and LAION.
  They are useful target-matching and implementation evidence. The LAION 0.90
  run is a single complete paired artifact; the Amazon/YFCC campaigns are
  per-filter shards plus repair cells and require a deterministic merge audit.
  None should be mixed with r36 global q10K rows.
- The current machine-readable r41 inventory is
  `results/hybrid_vector_db/r41_matched_recall_evidence_inventory.{csv,json}`.
  Of the eleven enumerated artifacts, four single-pass throughput artifacts are
  `expedited` and seven latency artifacts are `diagnostic`; none yet passes the
  complete formal publication gate.

### 7.3 Missing paper evidence

1. Current-release three-dataset matched-recall q10K final results.
2. Official upstream pgvector and SQLens-disabled A/B.
3. Current-protocol SQL-first, FAISS, and fully closed Weaviate baselines.
4. Amazon SQL-native q10K GT and formal execution measurements.
5. Formal measured service curves.
6. Current-release update correctness and read/write concurrency.

The old universal-frontier-dominance, full-system `1.95x`, formal-throughput,
and current-release transactional-update claims remain withheld until the
corresponding evidence passes the release gate.

## 8. Authoritative P0 Execution Order

The P0 queue follows the paper dependency graph. A later stage cannot begin
until the previous stage publishes an audited manifest.

### P0-1: Three-dataset current-release matched recall

1. Build the balanced q2800 calibration traces from the existing q200 query
   cohort and exact all-predicate truth, then rerun the r36 candidate grid.
2. Generate the fixed-target selector for Recall@10 = 0.90/0.95/0.99.
3. Run q10K/r3 paired/interleaved latency for the selected fixed-target pairs
   and preserve fully exhausted unattainable targets as explicit results.
4. Generate the Table 2 matched-target summary.
5. Separately generate the distinct-pair frontier selector with at least ten
   configurations per arm and dataset. Reuse any fixed-target raw cells whose
   configuration-pair hashes are identical, then run only the missing q10K/r3
   pairs for the latency half of Figure 5.

Current status (2026-08-02):

- r41 is the active binary. Any remaining r36 references in paper-facing runner
  defaults are release-migration defects; historical manifests retain their
  original r36 identity.
- A complete r41 LAION per-predicate q5K Recall approximately 0.90 run is
  available and classified under Section 5.4. A corresponding Recall
  approximately 0.95 artifact also exists, but contains targeted repair shards
  and must retain its merge provenance.
- r41 YFCC global q10K and measured c16 artifacts exist for several targets;
  their publication status must be recomputed against the current release and
  configuration-selection contract instead of copied from the r36 table.
- An r41 Amazon global-grid calibration is currently running. Do not launch a
  second database benchmark against the same PostgreSQL/storage instance while
  it is active.
- The bullets below describe the earlier r36/r37 campaign and remain as
  historical rationale for the stricter gates.

- The r37 formal workload artifacts now exist for Amazon-10M, YFCC-10M, and
  LAION-25M. Each calibration trace is the complete 200-query x 14-predicate
  Cartesian product (2,800 requests, exactly 200 observations per predicate),
  followed by the frozen disjoint q10K measurement trace.
- Amazon reuses its already complete exact all-predicate truth. YFCC and LAION
  calibration truth was recomputed by an exact float32 full-base scan over 10M
  and 25M rows, respectively. Their q10K measurement truth is reused only for
  exact `(query_no, query_id, filter_name)` matches. The 200 CPU/GPU overlap
  rows pass a fail-closed audit that permits only float32 rounding and tied
  IDs at the exact kth-distance boundary.
- The formal workload manifests and the 12,800-row assigned-truth artifacts are
  valid. Contract tests for workload construction, exact-truth generation and
  merge, selector, and final latency gates pass (100 tests).
- Base, Stock-cap, and SQLens-target q200 calibration completed with zero
  request errors and exact r36 runtime identity, but this trace has only about
  14 observations per predicate and is now classified as audit-only.
- A parallel q200 high-budget screening run is in progress only to reduce the
  expensive formal grid. Screening latency is invalid under resource
  interference and will never be selected or plotted; only its recall signal
  is used to choose formal q2800 cells. Formal calibration and all held-out
  latency measurements remain cache-isolated and sequential.
- The legacy aggregate-only fixed-target selector is
  `figure5_r36_formal/figure5_r36_fixed_target_configs.{csv,json,manifest.json}`.
  It binds 274 calibration artifacts and publishes eight selected pairs:
  Recall@10 0.90/0.95/0.99 on Amazon and YFCC, and 0.90/0.95 on LAION.
- The completed Amazon 0.90 q10K/r3 diagnostic is not paper evidence. Although
  aggregate recall passes 0.90, `long_review_ge500` falls from Stock 0.9342 to
  SQLens 0.4613 while contributing a 4.94x apparent speedup. This is precisely
  the quality-cost substitution that the formal per-predicate gate prevents.
- Under the legacy grid, LAION 0.99 is
  `unattainable_on_calibration_grid`: Stock exhausted
  ef 20K/50K/100K, while SQLens exhausted the registered high-recall beam/target
  extension through ef 10K and target 1K. The formal q2800 calibration must
  re-establish this result.
- The aggregate-only q10K run was stopped before Amazon 0.95 completed. P0-1
  resumes only after the balanced calibration, formal selector, and
  per-predicate final gate are published and tested.

### P0-2: Official upstream and disabled control

1. Rebuild one dedicated Amazon source-order HNSW index under the unmodified
   official pgvector 0.8.2 binary and archive binary/source/index provenance.
2. Upgrade the existing A/B runner from q100/r5 to the frozen q10K/r3 trace,
   pass DSN/planner/workload arguments through the binary controller, and bind
   the exact r36 build ID rather than an obsolete prefix.
3. Run official upstream and SQLens-disabled on that same table, index, SQL,
   trace, GT, and independently selected matched-recall targets.
4. Report auto-planner and forced-HNSW variants. The disabled arm must prove
   `final_path=stock` and zero D1/D2/D3 activity for every measured query.
5. Restore and verify the r36 binary before continuing.

The existing official/disabled artifacts are not reusable: they are q200,
partial, built from older SQLens binaries, or use a patched/non-pinned official
binary. No current artifact has a publishable official-vs-disabled manifest.

Implementation status (2026-07-30): the q10K/r3 A/B controller, disabled-path
proof, and official-index preparation/provenance tool pass local tests and
dry-run. The dedicated official-built index and the real binary-switch A/B
artifact have not yet been executed.

### P0-3: SQL-first, FAISS, and Weaviate

1. Re-audit existing SQL-first/FAISS artifacts against the current query,
   predicate, GT, and target contract.
2. Add independent method selection to the shared Amazon runner and rerun the
   SQL-first and FAISS cells against the current M32 index, filters, and GT.
3. Report planner-chosen and forced-indexed SQL-first separately. For FAISS,
   report SQL materialization, row transfer, bitmap construction, ANN search,
   full end-to-end, and cached-allow-list control as distinct metrics.
4. Rerun the complete Weaviate matrix because the old 41/42 result binds the
   previous filter/GT hashes. Bind the current import and truth manifests and
   return only row ID and distance in the timed GraphQL response.
5. Add `paper_eligible` gates and produce a payload-compatible strong-baseline
   summary. Do not treat latency reciprocals as measured concurrent QPS.

Implementation status (2026-07-30): SQL-first/FAISS and Weaviate now expose
current q200 calibration plus q10K/r3 formal protocols, per-request checkpoints,
current input hashes, and fail-closed publication gates. No current-protocol
PostgreSQL/FAISS or Weaviate final artifact has been executed.

### P0-4: SQL-native workloads

1. Restrict the main experiment to `acl_only`,
   `grant_temporal_selectivity`, and `fact_temporal_selectivity`, using four
   representative predicates spanning roughly 38% to 0.055% combined
   selectivity. Keep the six boolean-complex variants for the appendix.
2. Upgrade the GT producer from the hard-coded q100+q100 protocol and generate
   exact SQL-valid GT for the frozen q10,200 cohort.
3. Upgrade the runner to q80/r2 calibration plus a balanced q10K/r3 mixed
   measurement trace. Fix the continuous end-to-end timer, use Recall LCB95
   for selection, and reset D3 once per trace repeat rather than once per cell.
4. Run stock, forced-indexed SQL-first exact, and full SQLens for the three
   workload families under the r36 release identity.
5. Require complete trace/output hashes, RLS positive/negative probes, relation
   epoch stability, plan proofs, and zero errors before publication.
6. Report recall, mean/p95/p99, returned TIDs, heap/index activity, and plans;
   then generate Figure 6.

Implementation status (2026-07-30): the GT producer and three-arm SQL-native
runner pass local tests and q10K/r3 dry-run. The q10,200 SQL-native GT and the
real formal benchmark remain unexecuted.

### P0-5: Formal service curves

1. Remove all latency-reciprocal throughput numbers from paper-facing
   artifacts. QPS is measured only as completed queries divided by the
   barrier-delimited wall-clock interval.
2. For the throughput-recall half of Figure 5, run the 32 distinct matched
   pairs at one preregistered concurrency (16 clients), q10K/r3. This retains
   at least ten frontier points per dataset without multiplying every point by
   the full client grid.
3. Separately run the fixed Recall@10 0.90 pair for each dataset at
   clients 1/4/8/16/32/64, q10K/r6, to report closed-loop concurrency scaling.
   Describe this as one outstanding request per client, not open-loop SLO
   capacity.
4. Pass the container's real backend `/proc` root through the wrapper, use
   disjoint client/backend CPU partitions, and hold a global experiment lock
   while collecting host/device telemetry.
5. Record measured QPS with CI, per-request p95/p99 with CI, recall LCB95,
   errors/timeouts, backend and host CPU, device I/O, and relation/index
   reads/hits. Preserve those fields in a service-summary artifact rather than
   dropping them in the plotting aggregate.
6. Generate the throughput half of Figure 5 and an appendix
   QPS/p95/p99-versus-clients figure.

This split executes about 4.08M real hybrid queries instead of the redundant
23.04M-query Cartesian product of every frontier pair and every client count.
The current q400 concurrency artifacts and `1000 / mean_latency` summaries are
not reusable as formal service evidence.

Implementation status (2026-07-30): both preregistered service slices, Docker
backend-proc telemetry, and the service-summary gate pass local tests/dry-run.
No formal measured service cell exists yet.

### P0-6: Updates and concurrency

1. Add exact r36 source/binary gates, per-cell atomic checkpoints/resume,
   full-SQLens mode, current matched-recall input bindings, and formal
   `paper_eligible` gates to the two new update runners.
2. Run the 128-row adversarial commit/rollback/update/delete/TRUNCATE harness
   under RC and RR, including non-owner RLS/ACL cases.
3. Run the current-release 250K-row/1K-query correctness stress with real
   predicate/vector/insert/delete changes, four readers, two writers, and 2K
   committed updates.
4. Run Stock and full SQLens at Recall@10 0.90 for requested update rates
   0/10/100/1000 TPS across 1/4/8/16/32/64 readers, q10K/r6. Disable
   per-query profiling in the timed service run and collect a separate sampled
   profile trace.
5. Record overload cells and continue the matrix. Require delivered/requested
   update TPS >= 0.90 only for cells claimed as sustainable.
6. Gate recall/correctness, read p95/p99, update lag p95/p99, zero errors,
   successful commits, relation epoch transitions, invalidations, rebuilds,
   and reactivations.
7. Add a 50%/5%/0.5% selectivity sensitivity at 16 readers and
   0/100/1000 TPS, then generate Table 4 and the appendix service matrix.

The existing r22/r26 correctness and q400 read-only concurrency results remain
historical diagnostics. They are not current-release read/write evidence.

Implementation status (2026-07-30): the current-release correctness and
read/write runners support the formal matrix, atomic per-cell resume, sampled
profiling, real mutations, and lifecycle gates in local tests/dry-run. They
remain unexecuted and therefore provide no paper result yet.

## 9. Execution Efficiency

- Do not run independent database benchmark cells concurrently against the same
  PostgreSQL instance and storage path. They would interfere through buffer
  cache, backend CPU, and block-device I/O.
- Use one paired/interleaved invocation per dataset/configuration to avoid
  repeated restarts and prewarm.
- Parallelize pure CPU exact-GT shards, artifact validation, statistics, and
  plotting on separate CPU/NUMA partitions when they do not read the experiment
  database or shared block device.
- Throughput experiments deliberately use multiple clients; different
  throughput cells remain sequential.
- Run an auditor after every stage and resume only missing cells.

## 10. Artifact Release Gate

Every paper-facing artifact must satisfy:

1. `status=complete`, `artifact_valid=true`, `paper_eligible=true`.
2. Requested slice and full paper-required coverage are both explicit.
3. Runtime binary, source, index, relation, query, filter, workload, and GT
   identities match the frozen contract.
4. Raw row count, query coverage, repeat coverage, plan coverage, and hashes
   close exactly.
5. Errors, timeouts, silent fallbacks, plan drift, and missing cells are zero.
6. Unattainable cells include complete required-grid exhaustion proof.
7. Paper tables and figures are generated automatically from audited summaries.

## 11. Immediate Next Action

1. Let the active r41 Amazon calibration finish without a competing database
   workload, then audit its requested cells and recall gates.
2. Produce one r41 evidence inventory for all three datasets and targets,
   separating global q10K artifacts, expedited per-predicate q5K artifacts,
   calibration-only cells, and measured-QPS artifacts.
3. Migrate paper-facing runner defaults from r36 to the immutable r41 contract
   without rewriting historical manifests, and rerun affected unit tests.
4. Complete only the missing r41 matched-recall cells after the inventory;
   do not rerun cells that already satisfy the selected protocol.
5. Continue P0-2 through P0-6 only after P0-1 has one internally consistent
   protocol and fail-closed summary.
