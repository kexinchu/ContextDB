# Table 6 数据审计

更新时间：2026-07-30

## 1. Table 6 的数据口径

论文中的 Table 6 是三个数据集的 target-matched end-to-end 汇总表，而不是
普通的 selectivity sweep。每一行对应一个 `(dataset, target recall)`，要求：

- Stock PGVector 和完整 SQLens 分别独立选择一个全局配置；
- 同一配置覆盖该数据集的全部 14 个 filter，不能按 filter 单独调参；
- aggregate Recall@10 LCB95 和每个 filter 的 Recall@10 LCB95 均达到目标；
- final latency 来自 disjoint q10K held-out trace，三次 paired/interleaved repeat；
- 报告 mean、p95、p99、recall、paired 95% CI 和 14-filter win count；
- QPS 来自 16-client closed-loop 实测，不能由 latency 倒数推导；
- artifact 必须绑定 r36 build、`vector.so`、relation/index、query/filter/GT 和 plan。

因此，旧的 q100/r5、per-filter tuning 或 calibration 数据不能直接填入 Table 6。
结果目录中大量文件名包含 `table6`，但它们沿用的是早期稿件编号；论文重排
后，当前 Table 6 已改为 target-matched summary。这些旧文件名并不表示数据
已经满足当前 Table 6 的协议。

## 2. 当前正式数据状态

| Dataset | Target | r36 calibration | q10K final latency | 16-client QPS | Table 6 状态 |
|---|---:|---|---|---|---|
| Amazon-10M | 0.90 | 旧 aggregate selector 已选配置；正式 q2800 calibration 待完成 | 有一组 q10K/r3 diagnostic，但 per-filter recall gate 失败 | 缺失 | 不可填 |
| Amazon-10M | 0.95 | 旧 selector 已选配置 | 运行到约 4,750/10,000 后停止，无完整 artifact | 缺失 | 不可填 |
| Amazon-10M | 0.99 | 旧 selector 已选配置 | 缺失 | 缺失 | 不可填 |
| YFCC-10M | 0.90 | 旧 selector 已选配置 | 缺失 | 缺失 | 不可填 |
| YFCC-10M | 0.95 | 旧 selector 已选配置 | 缺失 | 缺失 | 不可填 |
| YFCC-10M | 0.99 | 旧 selector 已选配置 | 缺失 | 缺失 | 不可填 |
| LAION-25M | 0.90 | 旧 selector 已选配置 | 缺失 | 缺失 | 不可填 |
| LAION-25M | 0.95 | 旧 selector 已选配置 | 缺失 | 缺失 | 不可填 |
| LAION-25M | 0.99 | 旧 grid 上不可达到；正式 q2800 calibration 尚未复核 | 不适用 | 不适用 | 暂定，不是最终结论 |

当前 Table 6 因此不是“没有任何实验基础”，但严格按论文当前 caption 和
measurement contract，尚无完整、paper-eligible 的数据行。

## 3. 已有 selectivity 数据

此前确实在三个数据集上测试过 14 个 selectivity。它们主要属于 component
ablation，而不是当前 Table 6 的正式 end-to-end protocol。

| Dataset / artifact | Protocol | Existing result | 可用于 Table 6？ |
|---|---|---|---|
| Amazon Table 7 | q100/r5；per-filter matched settings；部分列来自不同实验 | 14 filters；SQLens geomean speedup 1.95x；14/14 wins | 否：非 q10K、非全局配置、非统一 r36 artifact |
| YFCC r32 four-arm | q100/r5；per-filter matched `ef_search` | S/Q recall 0.9414/0.9414；mean 63.06/38.93 ms；p95 271.40/174.62 ms；p99 867.87/575.53 ms；geomean speedup 1.32x；14/14 wins | 否：r32、q100、per-filter tuning、无实测 QPS |
| LAION r32 four-arm | q100/r5；per-filter matched `ef_search` | S/Q recall 0.9397/0.9397；mean 955.99/832.48 ms；p95 8540.30/7544.70 ms；p99 11315.67/10212.89 ms；geomean speedup 1.23x；14/14 wins | 否：r32、q100、per-filter tuning、无实测 QPS |

这些结果能够证明三个数据集的 selectivity sweep 已经执行，并可作为
ablation/appendix evidence；它们不能伪装成新的 held-out Table 6 数据。

对应文件：

- `paper/tables/eval_amazon_combined_ablation.tex`
- `results/hybrid_vector_db/yfcc10m_r32_four_arm_table7_repr_bloom_q100r5_20260723.csv`
- `results/hybrid_vector_db/yfcc10m_r32_four_arm_table7_repr_bloom_q100r5_20260723.csv.manifest.json`
- `results/hybrid_vector_db/laion25m_r32_four_arm_table7_repr_bloom_q100r5_20260723.csv`
- `results/hybrid_vector_db/laion25m_r32_four_arm_table7_repr_bloom_q100r5_20260723.csv.manifest.json`

## 4. Amazon 0.90 q10K diagnostic

r36 下已经完成一组 60,000-row Amazon q10K/r3 paired artifact：

`results/hybrid_vector_db/figure5_r36_fixed_target_latency/`
`figure5_r35_amazon_matched_latency_amazon-recall_0.900000000-`
`6b89f8e8e7-c934539f47.csv`

其诊断汇总为：

| Metric | Stock PGVector | SQLens |
|---|---:|---:|
| Aggregate Recall@10 | 0.9411 | 0.9175 |
| Mean latency | 92.60 ms | 65.60 ms |
| p95 latency | 505.19 ms | 373.02 ms |
| p99 latency | 776.03 ms | 571.90 ms |
| Minimum per-filter recall | 0.8578 | 0.4613 |

SQLens 在 `long_review_ge500` 上的 recall 仅为 0.4613；Stock 的最差 filter
也未达到 0.90。虽然 aggregate latency 看起来有 1.41x speedup，这个结果发生
了严重的 quality-cost substitution，因此 manifest 为
`paper_eligible=false`，不能填入 Table 6。

## 5. 仍需生成的数据

1. 完成三个数据集的 balanced q2800 formal calibration，按
   `(dataset, target, method)` 选择一个全局配置，并执行 per-filter LCB gate。
2. 对八个可达到的 dataset/target pair 执行 held-out q10K/r3 paired latency。
3. 用相同选中配置执行 16-client q10K/r3 throughput，生成实测 QPS 和 CI。
4. 汇总 Recall、mean、p95、p99、speedup CI 和 win count，填入 Table 6。
5. 用 formal calibration 重新确认 LAION-25M Recall@10 0.99 是否
   `unattainable_on_grid`。

当前名为 `figure5_r37_*` 的目录是下一轮实验 campaign；其 manifest 仍绑定
r36 release contract。现有状态为 planned/screening，不是新的 binary release，
也还不是 Table 6 final result。
