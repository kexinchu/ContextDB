# ContextDB-0: Token- and Latency-Aware Context Optimization for Agentic Workflows

# ContextDB / Agentic Memory 调研笔记（适用于论文前期准备）

---

# 🎯 最终目标：构建一个 **Agentic Memory DB / ContextDB**

一个作为 Agentic OS “外部大脑” 的数据库系统：

- 支持统一的 **V+G+R（Vector + Graph + Relational）** 数据模型  
- 提供 **上下文查询语言（Context IR / DSL）**  
- 提供 **上下文优化器（Context Optimizer）**  
- 对 token / latency / cost 进行 **优化（cost-based planning）**  
- 在底层实现 **多层存储（HBM/DRAM/CXL/NVMe）、graph-based memory、动态维护**  
- 支持 **可重放（deterministic replay）**、**调试**、**多租户安全控制**  

目标：替代所有 ad-hoc RAG / heuristic context selection，实现可控、可解释、可扩展的 agent 记忆系统。

---

# 🧩 当前现状（拼图碎片）

## 1. Percolate 系列（Agent + DB 强耦合）

- **Percolate：An agentic orchestrator *inside* a relational-vector-graph/key-value database**  
  - 把 agent orchestration 推入数据库内部  
  - 提供多模态索引（向量 + graph + KV + relational）  
  - 强调 memory paths + graph augmentation  
- **Designs for agentic memory**：用 LLM 生成 graph paths，作为 Knowledge Graph 的基础  
- MCP 集成：用 Model Context Protocol 让 DB 暴露工具与 memory 供桌面 agents 使用  

> ⭐ 缺点：工程实践超前，但没有 cost-based 优化器、formal models、系统级 evaluation。

---

## 2. AI-native Databases（行业趋势）

### TiDB（PingCAP）
- 推出 “AI-native DB”  
- Graph RAG + Knowledge Graph  
- 统一 SQL + 向量 + 图  
- 强调 HTAP + 多模态搜索

### OceanBase seekdb
- 推出生代向量/全文/SQL 一体化的 “AI-native hybrid search engine”

### MonkDB
- 统一时序 + 文档 + 向量 + 搜索  
- 强调 AI-native multimodal database

### Yugabyte / Google / AWS 的 AI 数据平台
- 强调必须支持：  
  - 多模态数据  
  - 复杂检索  
  - agent-friendly schemas  
  - 数据治理与 lineage

> ⭐ 但它们**没有**处理“上下文预算（token budget）”“context optimization”这些 agent-specific 问题。

---

## 3. 工业界明确承认“Context Management 是关键问题”的文章

### AWS — *Key components of a data-driven agentic AI application*
- 直接指出：  
  > “Context management component is essential.”  
- 因为：  
  - LLM 无状态  
  - 上下文有限且昂贵  
  - 内存选择必须动态与可控

### Anthropic — *Effective context engineering for AI agents*
- 分解 agentic context pipeline  
- 强调：retrieval + filtering + summarization + budget control

### Google ADK — *Efficient context-aware multi-agent frameworks*
- 引入 Flow + processors  
- 区分“存储（store）”与“展示（context builder）”

### Elastic — Agentic AI and context engineering
- 强调 hybrid search  
- 强调 context relevance != TopK

### DataHub — *Context Management is the Missing Piece*
- 指出：  
  > “大量 agent 项目失败，是因为缺乏 systematic context management。”

### Letta — *Benchmarking LLMs on Agentic Context Engineering (context-bench)*
- 介绍 “Memory Blocks” 概念  
- 给首次上下文选择评测基准

> ⭐ 所有这些工作都在说：**context selection 是核心痛点，但它被严重低估。**

---

## 4. 初步学术方向（还没有系统化 DB 论文）

### *Manage the Context of LLM-based Agents like Git*
- 提出“上下文版本管理与 git 类比”  
- 有 branch / merge / snapshot 思路  
- 实验有限，未触及 cost model 与查询优化

### *Trustworthy agentic AI systems: a cross-layer review*
- 跨层分析安全性  
- 指出 DB/存储/工具层应协同保障 agent 系统可信

> ⭐ 没有论文真正 formalize：  
> **“上下文优化 = 数据库查询优化问题”**  
> **也没有 cost-aware optimizer、上下文 IR。**

---

# 🧱 研究缺口（Your Opportunity）

现有系统没有：

1. **统一的上下文 IR / DSL**  
2. **上下文相关成本模型（token + retrieval latency + LLM latency）**  
3. **Cost-based Context Optimizer**  
4. **Adaptive retrieval（easy vs hard queries）**  
5. **多模态记忆的组合查询（vector + graph + relational）**  
6. **多层存储布局（HBM/DRAM/CXL/NVMe）**  
7. **上下文稳定性 / 重放 / 调试机制**

你可以一次性补齐，开创一个新方向：  
> **Agentic Memory DB / ContextDB**

---

# 🧪 第一篇论文要解决的问题（核心创新）
**把上下文选择定义成数据库优化问题：**

- 输入：用户 query、历史对话、工具结果、memory pool  
- 输出：context subset（token ≤ budget）  
- 目标：maximize relevance, minimize cost  
- 代价包含：  
  - token cost  
  - retrieval latency  
  - summarization cost  
  - LLM forward latency  

> ⭐ 这是首次从 *DB 视角* 形式化 agent memory。

---

# 📊 Motivation Test（必须展示的 5 个痛点）

## Test 1 — *Token cost is unstable (10–100× variance)*
- naive RAG 会随机超预算  
- 真实任务中 token/latency 不可预测  
- → 需要 cost-aware optimization

## Test 2 — *Budget violation leads to severe quality drop*
- 当 context 超/低于 budget 时，任务成功率急崩  
- → 必须有 budget-aware selection

## Test 3 — *TopK relevance ≠ task-level usefulness*
- embedding similarity 不等于任务相关性  
- → 必须用 structured memory（task graph, workflow DAG, dependency）

## Test 4 — *Hard queries require different retrieval plans*
- easy query: fast path（lower cost）  
- hard query: deep search（higher recall）  
- → context optimization must be adaptive

## Test 5 — *RAG is non-deterministic → agents cannot be debugged*
- naive RAG 在多轮运行中 context 会变化  
- agent 输出也变化  
- → 需要 IR + optimizer + trace

---

