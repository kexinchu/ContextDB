# Agentic Memory DB / ContextDB：蓝图、现状综述与三篇论文路线图

## 1. 最终蓝图：我们要造的系统是什么？

目标系统是一个 **Agentic Memory DB（ContextDB）** —— 面向 LLM Agentic OS 的“外部大脑”，提供统一、可控、高效、可重放的上下文与长期记忆管理。

### 🔥 核心思想（一句话）

> **把“上下文选择 + 长期记忆 + token/延迟预算控制”当成数据库问题来做**，具有自己的数据模型、查询语言、优化器、执行器和物理层设计。

---

## 1.1 系统架构（高层蓝图）

┌───────────────────────────┐
│ Agentic OS │
│ (LangGraph / AutoGen / │
│ OpenAI Agents / ADK ) │
└───────────────┬───────────┘
        查询上下文 / 存取记忆
┌───────────────▼──────────────────┐
│ Agentic Memory DB │
│ (ContextDB) │
│           │
│ ① 统一数据模型 (V+G+R) │
│ - 向量：语义检索 │
│ - 图：关联关系/多跳推理 │
│ - 关系：结构化属性 │
│           │
│ ② ContextQL / 上下文查询 IR │
│ DECLARE context WITH … LIMIT │
│           │
│ ③ 上下文优化器（Context Optimizer）│
│ - token/latency-aware │
│ - cost model │
│ - hardness-aware fast/slow │
│           │
│ ④ 执行引擎 │
│ - 向量检索 + 图搜索 + SQL │
│ - summarization │
│ - 多模态执行管线 │
│           │
│ ⑤ 物理层 (HBM/DRAM/CXL/NVMe) │
│ - placement / compaction │
│ - log-structured memory │
│ - multi-tier caching │
│           │
│ ⑥ 安全 / 多租户 / 重放 │
│ - agent 级权限 / 审计 / replay│
└───────────────────────────────────┘
                │
┌───────────────▼──────────────────┐
│ 存储和数据库基座 │
│ Postgres / TiDB / BigKV / KV+Graph│
└───────────────────────────────────┘


---

## 2. 当前现状：工业界/学术界已有的“拼图碎片”

虽然没有任何系统提供完整的 ContextDB，但多个方向已出现关键组件。

---

### 2.1 Percolate 系列：**最接近 ContextDB 的实战系统**

- *“agentic orchestrator **inside** a relational-vector-graph/KV database”*  
- 统一 Postgres + 向量 + 图 + KV + HTTP  
- 通过 MCP 把 DB 暴露给桌面 agent  
- Blog-style，缺乏 formalization 和系统论证

👉 **价值**：证明“把 agentic memory 放数据库里”是可行且有效的。

---

### 2.2 AI-native Database 运动（Oracle / TiDB / seekdb / MonkDB 等）

代表观点：

- **Oracle 26ai**：AI-native DB = 关系+向量+图，支持 agent workflow  
- **TiDB AI**：Graph RAG、多模态搜索、知识图谱  
- **OceanBase seekdb**：统一 SQL + 全文 + 向量  
- **MonkDB**：统一向量、时序、地理、文档、多模态引擎  

👉 **限制**：它们能做 hybrid search，但没有“上下文选择优化器 + agentic IR”。

---

### 2.3 工业界直接点名「上下文管理是关键瓶颈」

- **AWS**：context management 是 agentic AI 的核心组件  
- **Anthropic**：context engineering pipeline  
- **Google ADK**：把“存储 vs 展示”分开  
- **DataHub**：context management 是最容易导致 agentic 项目失败的部分  
- **Elastic / Letta**：提出 memory blocks、context-bench

👉 **启示**：大家知道这是问题，但没有数据库级解决方案。

---

### 2.4 学术界零散萌芽

- **Manage the Context of LLM Agents like Git**：版本化上下文  
- **Agentic AI Trustworthiness Survey**：跨层 DB/OS 安全  
- **Letta context-bench**：评估 context selection

👉 **缺点**：没有系统化 DB 架构，也没有 cost-based optimization。

---

## 3. 论文路线图：用三篇论文讲完整故事

### 《总线索》
三篇论文从“抽象 → 原型 → 完整系统”依次递进：

1. **第一篇：定义问题（formalization + optimizer 原型）**  
2. **第二篇：构建系统（ContextDB 内核）**  
3. **第三篇：和 Agentic OS 融合（事务、安全、重放、多租户）**

---

## 3.1 第一篇：**Context as a Query Optimization Problem**

**工作标题示例**：  
> *ContextDB-0: Token- and Latency-Aware Context Optimization for Agentic Workloads*

**贡献点：**

- 第一次将“上下文选择”形式化为数据库优化问题  
- 定义 ContextIR / ContextQL  
- 提出 cost model（token + latency + recall proxy）  
- 硬度感知（hardness-aware）fast/slow context retrieval  
- prototype planner：向量 + 图过滤 + token 裁剪  

**实验：**

- 用 code assistant / data copilot / 多 agent task 等 workload  
- 对比：
  - 朴素 RAG  
  - memory blocks heuristics  
  - 实际框架 baseline（Anthropic/AWS/Elastic）  

👉 **目标：确立 research agenda + 提出可运行的 optimizer 原型**。

---

## 3.2 第二篇：**ContextDB——一个完整的 Agentic Memory 数据库**

**工作标题示例：**  
> *ContextDB: An AI-Native Memory Database for LLM Agentic Systems*

**贡献点：**

1. **统一 V+G+R 数据模型与 schema**  
2. **ContextQL query language + logical plan**  
3. **Context optimizer → DB-style optimizer**  
4. **执行引擎**：向量检索 + 图 traversal + summarization operators  
5. **物理设计：**  
   - 多层 DRAM/CXL/NVMe  
   - log-structured memory graph  
   - compaction / pruning / dynamic augmentation（结合你的 OOD 工作）  
6. **与 Agentic Runtime 集成**（LangGraph/OpenAI Agents/ADK）

**系统实验：**

- end-to-end latency、成本、成功率  
- 真实场景：代码助手 / 文档问答 / 多智能体协作  
- 对比 TiDB AI-native、Percolate、向量 DB baseline

👉 **目标：发表在 OSDI/SOSP/VLDB/SIGMOD/MLSys** 的系统论文。

---

## 3.3 第三篇：**AgenticOS-Memory——事务、安全、多租户、可重放**

**工作标题示例**：  
> *AgenticOS-Memory: Transactional, Isolated, and Replayable Context Management for LLM Agents*

**贡献点：**

1. **Agentic transaction model（Saga-like）**  
   - 记录所有 memory reads/writes  
   - 可回放、可审计、可调试  
   - 可复现（deterministic LLM path-aware）

2. **多租户与权限治理**（结合 SafeKV）  
   - row/column masking  
   - agent capability model  
   - jurisdiction-aware memory storage（合规性）

3. **安全 / fail-slow 分析**  
   - prompt injection 检测（跨层数据流）  
   - failure replay 图（完整路径）  
   - graph drift / stale memory 修复

4. **大规模长期运行实验（几天到几周）**  
   - memory growth  
   - compaction 效果  
   - retrieval 延迟稳定性  
   - agent success rate over time

👉 **目标 venue：** NSDI / USENIX Security / NDSS / MLSys / OSDI side track。

---

## 4. 下一步应该立刻做什么？

### Step 1：写出上下文选择的 formalization
- 记忆对象定义  
- 输入（任务/query + budget）→ 输出（context set）  
- cost model（token + latency + quality proxy/hardness）

### Step 2：做一个小 prototype（第一篇核心）
- 用 Milvus/Qdrant + SQLite/Postgres 实现  
- 实现 basic planner：coarse vector → graph → prune → summarize  
- 做简单 agentic workload 测试

### Step 3：画出 ContextDB 正式架构图
- 我可以继续帮你完善到论文级别（双栏样式）

---

# 结语

未来两篇论文将扩展为：

- **第二篇：ContextDB — Agentic Memory Database 内核**  
  - 真正统一 V+G+R  
  - 多模态索引  
  - 动态更新 + 多层存储 + physical operators  

- **第三篇：AgenticOS-Memory — 安全、事务、可重放与多租户**  
  - 事务模型  
  - 内存追踪  
  - 欺骗防御 / prompt injection  
  - 调试与可观测性  


