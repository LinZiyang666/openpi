<!-- ---
status: Plan (议事完成 2026-05-23 — 9 议题全部决议；待启动 2 个孵化 plan)
level: L1 (议事框架文档；本文件本身不修改 src/，孵化的独立 plan 各自经 §3 G1)
date: 2026-05-23
author: Ziyang Lin (executor)
authority: Execution
parent_audit: logs/server_concurrency_resource_audit.log.md
spawned_plans:
  - L1: 独立 Baseline `_sync()` 移除 plan（A5）— 待启动
  - L3: 联合 Concurrent Serving Optimization plan（A1 + A2 + A6 + A7 + A8）— 待启动
hard_constraints:
  - C1: 保留非 --concurrent 模式作为极限速度基准
  - C2: server runtime 禁止修改数据库内容（read-only + frozen 守护）
note: 本 log 不进入 §4 Code，不需要 G1；它是 §2 Understand 阶段的协调文档，目的是在写任何具体优化 plan 前先把 6 项 + 2 项 latent issue 的方案空间、得失和决策点全部摊在桌面上，让项目所有者按顺序拍板。每议题决议后，若选 "实施"，会按 WA §2.1 升级为独立 L2/L3 plan 文件并启动 §3 G1。
--- -->

# Serving Optimization Council — Agenda & Decision Log

> **背景**：研究报告 [`server_concurrency_resource_audit.log.md`](server_concurrency_resource_audit.log.md) Audit R3 APPROVED，作为静态可行性审计完成。本文件把报告 §12 的 6 个优化机会 + §9.1 / §9.5 浮出的 2 个 latent code issue 拆为 8 个议题，逐一摊开方案空间，等项目所有者按顺序决议。
>
> **本文件不是某个具体优化的 plan**。它是 plan-of-plans：议程 + 每议题的背景 / 候选方案 / 得失 / 决策点 / 决议追踪。
>
> **议事结果是有约束力的**：议题决议为"实施"后，executor 才能按决议结果起草对应的独立 plan（`logs/<feature>_plan.log.md`），并按 WA §2.3 / §2.4 走 G1。

---

## 0. 议事性质与适用规则

### 0.1 本文件与 Working Agreement 的关系

- **本文件是 §2 Understand 阶段的产物**，不是 §2.3 Plan。它不引入 src/ 改动；不需要 G1；目标是产出"议程 + 决议"两份结构化记录。
- **每议题决议后的独立 plan** 才是 §2.3 Plan 的对象。决议为 "实施" 时：executor 起草 `logs/<feature>_plan.log.md` → 用户确认需求 → §3 G1 → §4 Code → §5 G2 → §6 Verify。
- **本文件本身的修改也是 append-only**：议题决议、Discussion Log 追加，旧记录不重写。这是为了保留议事链路的可追溯性，与 WA §10.1 Review Log append-only 同精神。

### 0.2 议事原则（参考 audit R3 reviewer scope 限制）

1. **实测先于决策**：报告 §B.2 列出 10 项 limitations + §B.3 列出 5 项廉价实测。**所有耗时数字、显存数字、SM 占用百分比、batching 收益倍数均为未实测的架构推断**。任何议题进入"实施"决议前，必须先有该议题相关的实测数字做支撑（除非议题本身是修 latent bug，与性能无关）。
2. **每议题独立 plan**：8 议题决议后，可能孵化出 1-8 份独立 plan。每份 plan 必须独立经 G1；不允许"打包 plan"绕过 G1 粒度。
3. **决策权属项目所有者**：本文件的"候选方案 / 得失"由 executor 整理；选哪个由项目所有者拍板；执行者不替项目所有者做决议。
4. **议事顺序可调**：默认按下方议程顺序逐一讨论，但项目所有者可随时跳转、并行讨论或暂停某议题。
5. **议程可扩展**：议事过程中若发现新议题（如某议题决议触发了 derived issue），追加为 A9+ 议题，而非塞回旧议题。

### 0.3 议事 Workflow

```
本 log 创建 (Round 0)
  ↓
按议程逐议题讨论（议题 N Round 1 → Round 2 → ...）
  ↓ 决议为 "实施" / "推迟" / "否决"
若 "实施"：
  起草独立 plan → 用户确认需求 → §3 G1 → §4 Code → §5 G2 → §6 Verify
若 "推迟"：
  填入 "决议追踪表"，标 trigger 条件（如 "等 A0 实测完成后重审"）
若 "否决"：
  填入 "决议追踪表"，标理由
  ↓
所有议题决议完成后，本 log status 转 `Validated`，移入 archive/
```

---

### 0.4 硬约束（项目所有者明确，对所有后续 plan 具有约束力）

> 这些约束在议事过程中由项目所有者明确提出。**所有后续起草的 plan 必须在 risk register 中 explicit 引用并满足这些约束**，G1 reviewer 据此审查。

| ID | 约束 | 提出于 | 适用范围 |
|----|------|--------|---------|
| **C1** | **保留非 `--concurrent` 模式**（single-connection + `interceptor.py:172-178` 的 non-eager 路径 + `torch.compile("max-autotune-no-cudagraphs")` 编译产物）**作为极限速度测试基准**。A2 batching / 多 stream / 协调层等任何优化代码改动**不得破坏 non-concurrent 路径的现有行为**——non-concurrent 模式下 `serve_policy.py` 启动后必须仍走原 Policy.infer → 单 request → torch.compile 编译产物的链路，不引入 barrier / queue / dispatcher | Round 4 (A3) | A1 + A2 + A6 联合 plan；所有 src/ 改动 |
| **C2** | **服务器运行期间禁止修改数据库内容**——backend `_entries` 的 `upsert` / `delete` / `insert` 在 server runtime 期间**完全禁用**。Server 启动时 load 一次 pkl 后转为**read-only**；任何 offline artifact build / enrich / factor backfill 必须在 server stop 后离线做。**Runtime 仅允许 derived state mutation**：`_score_memo`（per-search-session cache）/ `_active_search_sessions`（sid 生命周期管理）/ `search_call_count` 等——这些是 search 路径的派生缓存，不属于"数据库内容" | Round 7 (A6) | A1 + A2 + A6 联合 plan；A7 议题；所有 src/ 改动 |

**实现影响**（给联合 plan 起草者参考）：

C1 (non-concurrent 保留)：
- batching coordinator（barrier / queue / batch window 逻辑）只在 `--concurrent` 模式下激活
- non-concurrent 模式下 `WebsocketPolicyServer._handler` / `Policy.infer` / `InferenceInterceptor.infer` 行为完全不变
- `interceptor.py:172-178` 的 `eager=True` 分支与 `_get_or_compile_stages()` 分支同时保留，CLI / config 决定走哪条
- 任何新加的状态 / 类 / 函数必须可被 non-concurrent 路径 bypass

C2 (runtime write-frozen)：
- `InMemoryBackend.insert / delete` 在 server runtime 期间应主动拒绝（throw `BackendFrozenError` 或类似），不再依赖 `_active_search_sessions` mutation guard 兜底
- **直接将 audit report §9.1 揭示的 RLock 漏洞从 "潜在 bug" 降为 "non-issue"**：runtime 共享 backend `_entries` 是 read-only，GIL 保证 dict lookup 单 op 原子 → 多 connection 并发 read 100% 安全无 race
- **A7 议题（Backend RLock 一致性）几乎自动决议为 B**（改 docstring 明确"runtime write-frozen，不需要 RLock"）
- offline_writers / DumpingJudge 写 jsonl / pkl 是 **on-disk write**，不属于 backend `_entries` mutation → 仍允许（不被 C2 禁止）

---

## 1. 议程总览（议题速查表）

> 议程顺序按 **"决策依赖关系 + 风险-收益比"** 排列。A0 是其他议题的前提；A1 决议会影响 A2 的设计前提；A7 / A8 是 latent bug，独立于性能优化但需要在 A2 落地前澄清并发模型。

| # | 议题 | 类型 | 决议状态 | 前置依赖 |
|---|------|------|---------|---------|
| **A0** | 实测先于决策 — 是否先按 §B.3 完成 5 项实测再启动具体议题？ | Meta | 未决 | — |
| **A1** | 进程模型 — 保留 3 server 多进程 vs 切换单进程 `--concurrent` | 架构 | 未决 | A0 |
| **A2** | Request Batching — 是否做？做哪类（全 batching / micro-batching / 不做） | 架构 / 核心 | 未决 | A0, A1, A7 |
| **A3** | 重 KeyBuilder 异步化 — CLIPKeyBuilder / LLMLayerExtract 是否搬到独立 stream | 实现 | 未决 | A0 |
| **A4** | 多 CUDA stream — per-connection / per-stage 流并行 | 实现 | 未决 | A0, A2 |
| **A5** | Baseline path `_sync()` 移除 — 是否清理 `policy.py:104-129` 强制 sync | 实现 | 未决 | （独立） |
| **A6** | 多 yaml 共存 — `_current_bundle` 从单 latest 改为 dict[bundle_id] | 架构 | 未决 | （独立） |
| **A7** | Backend RLock 一致性 — docstring 声明 vs 代码未实现 | Latent bug | 未决 | （独立） |
| **A8** | offline_writers 属性名不一致 — `_extractors` vs `_factors` | Latent bug | 未决 | A0（先实测确认是否 dead code） |

---

## 2. 议题详细说明

> 每议题块格式：背景 → 候选方案 → 各方案得失 → 决策点（项目所有者要拍板的具体问题） → 决议（默认"未决"，决议后追加 Round 标记）。

---

### A0 — 实测先于决策

**类型**：Meta 议题（决定所有其他议题的进入门槛）

#### 背景

- 研究报告 §B.2 列出 10 项 limitations（全代码静态分析、无 profiler、无 nvidia-smi、KV cache 估算未实测、InMemoryBackend 100K-entry 库延迟未测、17 因子链路未测、CLIPKeyBuilder / LLMLayerExtract 实际 GPU 占用未测、无 load test、未跑过 baseline、JAX 路径未涉及）。
- §B.3 列出 5 项 < 1 小时工作量的廉价实测：
  1. `nvidia-smi dmon -s u`（60 秒，看 SM / mem / PCIe）
  2. 历史 sweep 的 `--timing_csv_dir` CSV 回推 stage1/2/3 比例
  3. `py-spy top --pid <server_pid>` 30 秒（CPU 时间花在哪些 Python frame）
  4. **batched-infer microbench**：手写脚本，`Policy._model.sample_actions(device, batched_obs, noise)` 直接传 batch_size = 1/2/4/8/16，画 throughput vs batch_size 曲线
  5. `nsight-systems` trace（确认 default stream 串行假设）
- Audit R3 reviewer scope 明确："本 approval 仅批准本文件作为静态资源画像 / 可行性审计；不批准任何后续 serving 优化代码方案、实现路径或性能收益数字。"

#### 候选方案

- **方案 A — 串行：先做完 5 项实测，再开 A1+ 议题**
- **方案 B — 并行：A1+ 议题讨论方案设计的同时跑实测，实测结果按需注入各议题决议**
- **方案 C — 按需：议题决议时若该议题对实测有依赖再做对应实测，不依赖的议题（如 A5/A7/A8）直接讨论**
- **方案 D — 跳过：按报告估算直接拍板，实施后再用实测验证**

#### 各方案得失

| 方案 | Pros | Cons |
|------|------|------|
| **A 串行** | 决议可信度最高；A2 batching 的 batch window 等参数能用实测曲线选定；§B.3 时间成本仅 < 5 小时（5 项 × < 1h） | 拖延启动；需要 GPU 占用窗口（实测期间不能跑其他 sweep） |
| **B 并行** | 议事推进 + 实测同时进行，决议时机够准；讨论本身能澄清 "哪个实测最关键" → 实测目标更精准 | 要求 executor / user 协调实测与议事节奏；某些议题（如 A2）讨论到 70% 才发现需要新实测 |
| **C 按需** | 灵活；非性能议题（A5 / A7 / A8）零等待 | 性能议题（A1-A4 / A6）大概率都需要实测，本质上和方案 A 接近，但缺乏统一调度 |
| **D 跳过** | 最快进入 code | 若估算偏差大（例如 batching 收益实际只有 1.5× 而非估算的 2-4×），可能浪费 L2/L3 工时；Audit R3 scope 明确禁止用报告数字做工程决策 |

#### 决策点

1. 选 A / B / C / D 中的哪一种？
2. 若选 A 或 B：5 项实测的执行优先级？建议优先级 = `batched-infer microbench (4) > SystemTimer CSV 回推 (2) > nvidia-smi dmon (1) > py-spy (3) > nsight-systems (5)`，理由是 (4) 直接验证 A2 batching 议题最大变量，(2) 几乎零成本（数据已存在），(1) 抓 baseline 状态，(3)(5) 是 finer-grain 验证。
3. 实测的执行者 — 是 executor 跑还是 user 跑？是否要为此立一个 micro-plan（不需要 G1，因为是 "运行已存在工具 + 读 CSV"）？
4. 实测 artifact 落在哪里？建议 `exp/serving_optimization_baselines/` 新目录，与现有 sweep 实验分开，per-measurement 一个 markdown report。

#### 决议

未决（议事 Round 0 立项）。

---

### A1 — 进程模型：3 server 多进程 vs 单进程 `--concurrent`

**类型**：架构议题（影响 A2 / A4 的实施方式）

#### 背景

- 现状：一台机器跑 3 个独立 `serve_policy.py` 进程，每进程独立加载模型权重 ≈ 4.6 GB（bf16 2.3B 参数）→ 3 × 4.6 GB = **13.8 GB 仅权重**，加上 KV cache + 激活 + CUDA context + caching allocator overhead 估总占用 16-22 GB（未实测）。
- `--concurrent` 模式已存在并经测试：`scripts/serve_policy.py:405-432` 把 base policy 传给 `WebsocketPolicyServer(concurrent=True, connection_policy_factory=...)`，每连接一份 wrapper stack，base policy + shared_storage 跨连接共享。
- **关键事实**：报告 §0 + §12.2 — 单进程多连接**不会自动提升吞吐**，因为还是 default stream FIFO + batch=1。显存节省 ÷ 3 → 给 batching 留余地，但必须**配合 A2 (Request Batching) 才有真实吞吐收益**。
- 当前我们所有 verdict_phase5 sweep 都跑 `--concurrent`，所以 `--concurrent` 模式本身是**生产已验证**的；多 server 进程的用法是为了 "一机一卡跑多 yaml 串行 cell" 的并行加速 — 每 yaml 一个 server，多 server 同时跑不同 yaml 的 cell。

#### 候选方案

- **方案 A — 全切单进程 concurrent（一台机器一个 server，所有连接复用同一份权重）**
  - 单 yaml 模式下：单进程 + N 个 worker 连接，全部走 `--concurrent` 路径
  - 多 yaml 模式下：仍是单进程，靠 A6 议题（多 yaml 共存）或时间切片承担多 yaml
- **方案 B — 折中：减少到 2 server 进程**（一台机器 2 server，每进程 N/2 worker）
- **方案 C — 保持现状 3 server 进程**
- **方案 D — 动态：保留两种模式的代码路径，由 CLI 决定**（现状已经是这样，因为 `--concurrent` 是 flag）

#### 各方案得失

| 方案 | Pros | Cons |
|------|------|------|
| **A 全切单进程** | 显存 ÷ 3 → ~9 GB 释放，可用于增大 batch 或挂额外 yaml；多 yaml 也只需 1 份模型；运维简化（一台机器一个 process） | **单进程吞吐不会自动提升**，必须配合 A2；切换前需 load test 验证 `--concurrent` 在生产 sweep 强度下的稳定性（虽现在 phase5 已经在跑，但 worker 数从 5 提到 15 是否仍稳？）；多 yaml 场景损失 "进程隔离" 故障域 |
| **B 2 进程折中** | 部分省显存（节省 4.6 GB）；故障隔离保留；2 进程比 3 进程少一份 cuda context | 仍未根本节省；与 A 比 PR 工作量相同 |
| **C 现状 3 进程** | 零改动；进程隔离故障域；多 yaml 场景天然并行（每 yaml 一个 server）；调试时一个 process crash 不影响另两个 | 显存占用最高；权重 ×3 是浪费；GPU 仍未跑满（默认 stream FIFO） |
| **D 保留两种模式** | 灵活；不同实验选不同模式 | 维护两条 code path 成本高（现状已是 D，但 wrapper stack / config / serve_policy CLI 都需要兼顾两种模式） |

#### 决策点

1. 选 A / B / C / D？
2. 若选 A：是否要先在 A0 实测中加一项 "单进程 concurrent 在 15 worker / 30 worker 并发强度下的显存 / latency / crash rate"？
3. 单进程模式下，目前 `--concurrent` 路径的 per-connection 状态隔离是否已经经过强度测试？（报告 §9.2 列出 per-conn 独立项，但未验证 multi-worker 同时跑会不会触发 race）
4. 若决议为 A 但 A2 决议为 "不做 batching"，则 A 的吞吐收益变成 0 — 是否仍切换？（显存节省本身就有价值，因为可以挂更多 yaml 或扩大 KV cache window）

#### 决议

未决。

---

### A2 — Request Batching（核心议题）

**类型**：架构议题（最大杠杆 + 最大工程量）

#### 背景

- 报告 §12.1 评级 ★★★。模型 100% 支持 batch>1（已亲验 `pi0_pytorch.py:497, 578, 636, 675, 707, 728` 多处 `bsize=state.shape[0]`，所有 stage forward 广播 batch 维度）。
- **batch=1 注入点**（两条 hot path 都有）：
  - Baseline path：`policy.py:87 [None, ...]` + `policy.py:138, 199 x[0, ...]`
  - **Cache hot path**：`interceptor.py:512-516 [None, ...]` + `interceptor.py:559-560`（FULL_HIT）+ `interceptor.py:668-669`（MISS/WS）`x[0, ...]`
- **工程难点**（来自报告 §12.1 工程难点清单）：
  1. **Transform 不支持 batched**：`transforms.py:24-30` `DataTransformFn` Protocol docstring 明确 "unbatched data elements"；`TokenizePrompt :248` / `Normalize :115` 都单实例
  2. **Cache 状态隐含 batch=1**：所有 KeyBuilder、Orchestrator `_state_history` / `_action_history`、SearchStrategy `_search_session_id` 都假设 single trajectory
  3. **`__hit_meta__` 是 per-request 标注**：`interceptor.py:561, 668` 写到单个 outputs dict，batching 后必须拆分
  4. **CP1/CP3 决策可能分叉**：batch 内不同 worker 可能 FULL_HIT / WARM_START / MISS 三种结果并存，无法走单一分支
  5. **Backend 在 concurrent 下未加锁**（议题 A7 同一根源）：batching 引入新的并发 search 调用前必须先决 A7

#### 候选方案

- **方案 A — 全 batching**：等 N ms 收集未处理 obs → batch=K forward → 拆分发回
  - **子选择 A.1 batch window 策略**：固定 N ms / 动态根据当前队列长度 / 第一个请求到达后立刻凑齐 N 个
  - **子选择 A.2 决策分叉策略**：
    - (a) **worst-case path 强制**：batch 内任一 worker 没 FULL_HIT 就全部走 stage2 + stage3，FULL_HIT 的浪费 stage2/3 算力但实现简单
    - (b) **sub-batch split**：CP1 后按 hit_type 拆 sub-batch，FULL_HIT 直接返回，WARM_START / MISS 继续走 stage2，但代码复杂度高
  - **子选择 A.3 Transform 路径**：
    - (a) **per-request transform → stack**：每请求独立跑 transform → torch.stack → 模型 → unstack → 每请求独立 output_transform。**避开 transform 改造**但损失 transform 阶段本身的 batch 收益（transform 时间 < 推理时间，损失可接受）
    - (b) **系统性改造 batched transform**：每个 transform 类支持 batched 输入，包括 batched tokenize、batched mask 构造。工程量大但全程 batched
- **方案 B — Micro-batching**：只 batch stage 内的 GPU forward，cache 检索仍 per-request
  - 实现：所有 N 个连接独立跑 CP1（per-request KeyBuilder / search / judge），然后**收集 N 个 stage2 输入 → 一次 batch forward → 拆分 → 各连接独立跑 CP3 + stage3**
  - 适合 "只想 GPU 多吃点" 的目标
- **方案 C — 不做 batching，仅做 stream pipeline（议题 A4）**
- **方案 D — 推迟到实测之后**：A0 跑完 batched-infer microbench 后，根据曲线决定值得做哪一种

#### 各方案得失

| 方案 | Pros | Cons |
|------|------|------|
| **A 全 batching** | 最大吞吐杠杆（理论 2-4×，未实测）；GPU 利用率随 batch 上升直到 KV cache / 内存带宽变瓶颈 | 工程量最大（L3 architectural）；需要重新设计 wrapper stack 与状态划分；CP1/CP3 分叉处理是核心难点；与 A7（Backend RLock）耦合 |
| **B Micro-batching** | 工程量中等（L2）；cache 检索路径不变，避开 KeyBuilder / Orchestrator / SearchStrategy 全部状态拆分难题；transform 路径也避开 | 收益小一些（只 GPU 阶段 batched，cache 检索 N 次串行）；如果 CP1 search 耗时显著（报告估 10-50 ms / 检索），N 个串行 search 会成为新瓶颈 |
| **C 不做 batching** | 零工程；A4 stream pipeline 可单独实施 | GPU 利用率仍 batch=1 状态；理论上限低 |
| **D 推迟到实测** | 决策有数；避免做 over-engineering | 拖延整体进度；不过 §B.3 实测 < 1 小时 |

#### 决策点

1. 选 A / B / C / D？
2. 若选 A：A.1 / A.2 / A.3 各取什么？
3. **batch window 上限是多少 ms 是可接受的 latency 增加**？（取决于下游 LIBERO worker 对 action server 的延迟容忍）
4. **batch size 上限怎么定**？（取决于 batched-infer microbench 的拐点 — A0 议题输出）
5. 若决议为 A 或 B：是否要求 A1 (单进程模式) 先决议为 A？（多进程模式下 batching 也能做，但每进程独立 batch 窗口收益不如单进程内合并）
6. 是否要求 A7 (Backend RLock) 先决议？（batching 引入更强的 backend 并发，必须先澄清锁语义）

#### 决议

未决。

---

### A3 — 重 KeyBuilder 异步化（CLIPKeyBuilder + LLMLayerExtract）

**类型**：实现议题（条件性收益 — 仅当这两个 KeyBuilder 实际被使用）

#### 背景

- 报告 §6.2.1：`CLIPKeyBuilder` 独立 open_clip ViT-B-32（88M 参数），device 来自 stage1 GPU（同 stream），每次推理 CP1 一次 `encode_image`，**估计 50-100 ms**（未实测）。
- 报告 §6.2.2：`CP1LLMLayerExtractKeyBuilder` 借用 base policy paligemma layers（共享权重 → 不重复加载），但跑 `extract_layer` 层 forward，**估每层 ≈ stage2/18 ms**（gemma_2b depth=18）。`extract_layer=5` 跑 6 层 ≈ stage2 的 33% 额外开销。
- 两者都和 stage2 同 GPU 同 stream **架构上串行**。
- **使用频率不明**：当前 sweep 实验有多少 cell 用了这两个 KeyBuilder？需要统计。

#### 候选方案

- **方案 A — 异步化（独立 stream）**：把 CLIPKeyBuilder / LLMLayerExtract 移到独立 CUDA stream，与 stage1 后续 CPU work 或主路径其他部分重叠
- **方案 B — 替换为更轻量 KeyBuilder**：用 stage1 输出 + 池化做廉价 key，避开独立 88M 模型
- **方案 C — Skip CP1 / 只在某些 step 用**：condition gate，特定条件下才跑 CLIP / LLMLayerExtract
- **方案 D — 不优化**：仅当后续实测发现这两个 KeyBuilder 占比小 / 用得少
- **方案 E — 推迟**：A0 完成后先看实测占比再决议

#### 各方案得失

| 方案 | Pros | Cons |
|------|------|------|
| **A 异步化** | CLIP 用户能省估 ~50 ms / step；LLMLayerExtract 用户能省 6-33% 的 stage2 等量时间 | CP1 check critical path 需要 KeyBuilder 输出，不能简单 fire-and-forget；需要重新排序 stage1 → keybuilder → CP1，引入 stream sync 点；与 torch.compile / CUDAGraph 兼容性需验证 |
| **B 轻量替换** | 避开 stream 复杂度；KeyBuilder 调用变 < 5 ms | 检索准确性可能下降（CLIP / LLMLayerExtract 的表达力是有原因的）；需要重新跑 baseline 实验确认替换后命中率不下降 |
| **C 条件跳过** | 减少平均 latency；保留语义 | gate 设计复杂；可能误跳关键 step；与 verdict factor judge 17 因子配合需要重新设计 |
| **D 不优化** | 零工程 | 用户继续吃 CLIP 50-100 ms / step 的串行成本（如果他们在用） |
| **E 推迟** | 数据驱动决策 | 拖延 |

#### 决策点

1. **先实测一次 sweep 中 CLIPKeyBuilder / LLMLayerExtract 的使用频率**：grep 现有 yaml 看多少 cell 用了；看实际历史 timing CSV 看占比。这是 A0 实测的子项之一。
2. 选 A / B / C / D / E？
3. 若选 A：与 A2 / A4 配合关系如何？（如果 A4 决议多 stream，A 是 A4 的一个 instance）

#### 决议

未决。

---

### A4 — 多 CUDA stream（per-connection / per-stage）

**类型**：实现议题（条件性收益 — 取决于 A2）

#### 背景

- 报告 §0 / §12.3：所有 PyTorch CUDA op 进同一 default stream（代码全文无 `torch.cuda.stream(...)` 上下文，已亲验），FIFO 串行执行所有连接的 kernel。即使 GIL 释放，多 worker 并发的 kernel 仍是 GPU 端**严格串行**。
- 物理上：stream overlap 可以让 A 连接的 stage2 + B 连接的 stage1 在 GPU 上并行（如果 stage1 是 vision 比较 light，stage2 是 LLM 比较 heavy）。
- **关键约束（来自报告）**：单 worker batch=1 状态下，stream overlap 在小 batch 上效果有限（kernel 太短，launch latency 撑不起 overlap）；**配合 A2 batching 后基本没必要**（因为 batched kernel 时长变长，stream overlap 收益相对变小）。

#### 候选方案

- **方案 A — per-connection 一个 stream**：每个 WebSocket 连接的 wrapper stack 持自己的 stream，所有 forward 进该 stream
- **方案 B — per-stage 一个 stream**：stage1 / stage2 / stage3 三流并行，跨 stage 同步点显式管理
- **方案 C — KeyBuilder 独立 stream + main 流**（A3 议题的 stream 实现版本）
- **方案 D — 不做**：等 A2 batching 落地后再评估
- **方案 E — 推迟**：A0 nsight-systems trace 后看 default stream 是否真的是瓶颈

#### 各方案得失

| 方案 | Pros | Cons |
|------|------|------|
| **A per-connection** | CPU 端 issue kernel 时能并行进流；调试简单（一连接一流）；不影响 cache hot path 状态 | 单 batch=1 kernel 太短，stream launch latency 撑不起 overlap；与 torch.compile / CUDAGraph 兼容性问号；GPU 端仍需要资源（SM / register / shared mem）调度，stream overlap ≠ kernel 真并行 |
| **B per-stage** | 流水线收益直接；不同 stage 用不同算力特征（vision / LLM / action expert）能更好地填满 GPU | 严格依赖管理（stage2 必须等 stage1 + KeyBuilder + CP1 完成）；和 cache 路径状态依赖耦合复杂；若 stage1/2/3 时长差异大，pipeline 效率低 |
| **C KeyBuilder 独立** | 解决 A3 同一根源问题 | 与 A 重叠 |
| **D 不做** | 零工程；等 A2 后或许根本不需要 | 错过 baseline 阶段的小收益（不大） |
| **E 推迟** | 数据驱动 | 拖延 |

#### 决策点

1. 选 A / B / C / D / E？
2. 与 A2 的顺序：先 A2 还是先 A4？（推荐：A2 决议 "做" → 等 A2 实施完成后 A4 自然降级；A2 决议 "不做" → A4 升优先级）
3. 与 torch.compile / CUDAGraph 兼容性如何处理？（当前 concurrent path `eager=True` 不走 compile，所以暂无冲突；但如果未来打开 compile，stream 需要重新设计）

#### 决议

未决。

---

### A5 — Baseline path `_sync()` 移除

**类型**：实现议题（低优先级 — 仅当我们仍用 baseline 路径）

#### 背景

- 报告 §3.3 / §12.4：`policy.py:101-145` 是 Policy.infer 的 staged 分支（`_staged_inference=True` 且**没有 cache 包装**时触发）：
  ```python
  torch.cuda.synchronize()  # line 104-105
  ... stage1 ...; _sync()  # line 112
  ... stage2 ...; _sync()  # line 118
  ... stage3 ...; _sync()  # line 129
  ```
- 每 stage 强制 `torch.cuda.synchronize()` 等待整个 GPU 空 — 为获取 per-stage CPU 时间戳服务。
- **关键事实**：该路径在 `--cache` / `--cache_config` 模式下**完全不走**（`InferenceInterceptor.infer` 是 wrapper stack 顶层入口，不调用 `self._policy.infer`，自己用 CUDAEventBackend 做 per-stage timing）。
- 我们当前所有 verdict_phase5 sweep 都走 cache 路径，**这个 `_sync` 对实验吞吐无影响**。

#### 候选方案

- **方案 A — 删除 baseline `_sync()`**：保留 timing 但改用 CUDA Event（不阻塞 stream）
- **方案 B — 加 flag 控制**：`--no-stage-sync` 或环境变量，默认保留旧行为
- **方案 C — 不动**

#### 各方案得失

| 方案 | Pros | Cons |
|------|------|------|
| **A 删除** | 消除 baseline 路径不必要 sync；和 cache 路径架构一致 | 我们的实验**不走** baseline，收益为零；只对外部用 baseline 的 user 有用；改 `policy.py` 是公共代码 |
| **B 加 flag** | 保留可选 timing | 增加 complexity for zero internal gain；额外维护 |
| **C 不动** | 零工程 | baseline 路径在 multi-worker 下仍受影响（但我们不用） |

#### 决策点

1. **我们未来是否还会用 baseline 路径**？（用来做 ablation 对比 / sanity check？）
2. 若不用：A5 优先级降到最低，决议直接为 "C 不动"。
3. 若用：A 或 B？

#### 决议

未决（推测决议方向：C 不动；待用户确认）。

---

### A6 — 多 yaml 共存（`_current_bundle` → dict[bundle_id]）

**类型**：架构议题（中长期方向）

#### 背景

- 报告 §9.3 / §12.6：当前 `_current_bundle` (`websocket_policy_server.py:91`) 单 latest，无法同时挂 yaml-a 和 yaml-b 的 cache 配置；多 yaml 实际靠：
  1. 多 server 进程（最常见）
  2. 运行时 `load_cache_config` ctrl 消息切换（但只有最新一份，老连接持自己进来时的 snapshot）
  3. 多 sweep cell 串行（一次跑一个 cell 再换）
- "时间叠加" 而非 "空间并存"。
- 改成 `dict[bundle_id, CurrentCacheBundle]` 后 client `__ctrl__` 指定 bundle_id 即可同时挂多 yaml。

#### 候选方案

- **方案 A — 改成 dict[bundle_id]**：架构层面支持多 yaml 同时跑
- **方案 B — 现状不变**：多 yaml 用多 server / 时间切片
- **方案 C — 推迟**：等 A1 (单进程模式) 决议后再决

#### 各方案得失

| 方案 | Pros | Cons |
|------|------|------|
| **A dict[bundle_id]** | 根本上支持多 yaml 同时跑；显存进一步节省（无需为每 yaml 起 server）；多 sweep cell 可并行而非串行 | 架构改动大：`build_shared_storage` 与 yaml 强绑定，需重构；`__ctrl__` 协议改动（client 也要改）；现有 `verdict_phase5` sweep 是串行 cell 设计，未必需要并行 |
| **B 现状** | 零改动；A1 (单进程) 若决议为 A，多 yaml 仍能用 ctrl 消息切换 | 多 yaml 强度上限被 "时间切片" 限制 |
| **C 推迟** | 让 A1 / A2 先决议，再看 A6 是否仍有意义 | 拖延 |

#### 决策点

1. **我们的 sweep 实验是否需要 "同时挂多 yaml"**？（例如：一台机器同时跑 yaml-a 的 cell-1 + yaml-b 的 cell-2，避免串行？）
2. 若不需要：B 不变即可。
3. 若需要：A 是中长期方向，但需要在 A1 / A2 落地后再上。

#### 决议

未决（推测决议方向：B 不变 或 C 推迟；待用户确认 sweep workflow 实际需求）。

---

### A7 — Backend RLock 一致性（latent bug）

**类型**：Latent bug（独立于性能优化但与 A2 强耦合）

#### 背景

- 报告 §9.1 / §12.1 工程难点 #5：`backend_base.py:14-16` 模块 docstring 声称 "Backends are NOT required to be thread-safe. CacheStorage serialises all calls with an RLock."
- **代码实际未实现这个 RLock**（已亲验）：
  - `cache_storage.py:93-112` `CacheStorage.search()` 直接 `return self._backend.search(spec)`，无 lock 上下文、无 `threading` 导入
  - `cache_storage.py` 全文 grep `_lock|RLock|Lock|threading` 0 匹配
  - `InMemoryBackend` 共享 `_entries` / `_active_search_sessions` / `_score_memo` 可变状态，hot path 全部 lock-free
- 当前 hot path 是 lock-free read 居多 + mutation guard（`_active_search_sessions` 检查 + `SearchSessionActiveError` raise），**没有可观察到的 data corruption**。
- 但 **A2 batching / 多 worker 并发场景下**：复合操作（如 `bucket = setdefault(sid, {}); bucket[key] = ...`）不是 atomic，两个连接可能拿到同一个 bucket dict 互相覆盖。

#### 候选方案

- **方案 A — 补 RLock 兑现 docstring**：`CacheStorage` 加 RLock，所有 search / fetch / upsert 在 RLock 内
- **方案 B — 改 docstring**：明确 "不线程安全，concurrent 调用者负责同步"，删除 RLock 承诺
- **方案 C — InMemoryBackend 自持 RLock**：在 backend 内部加锁，CacheStorage 不变
- **方案 D — 改成 per-connection backend instance**：每连接自己的 backend，无共享状态
- **方案 E — 推迟到 A2 决议后**：A2 不做 batching → A7 优先级降；A2 做 batching → A7 升 blocker

#### 各方案得失

| 方案 | Pros | Cons |
|------|------|------|
| **A CacheStorage RLock** | 兑现 docstring 承诺；one centralized lock，调试简单 | hot path 每次 search 加锁可能让 latency 增加（需要 benchmark）；现有 InMemoryBackend 内部已经 lock-free，加 outer RLock 等于全序列化 |
| **B 改 docstring** | 诚实声明；代码零改动 | 未来 concurrent 优化必须自己解决并发；docstring 现状本身就是误导 |
| **C InMemoryBackend RLock** | 锁粒度更细（per-backend）；docstring 含义保留 | 多个 backend 类型（InMemory / Qdrant）各自加锁，一致性需要协议规定；Qdrant 是 HTTP / gRPC，本身已经 thread-safe 不需要锁 |
| **D per-conn backend** | 彻底隔离 | 76 MB pkl × N 连接显存爆炸；与 A1 (单进程多连接) 直接冲突 |
| **E 推迟** | 让 A2 先决议 | 现状下 latent bug 仍存在 |

#### 决策点

1. 选 A / B / C / D / E？
2. A2 决议 "做 batching" → A7 必须先决议（建议 A 或 C）
3. A2 决议 "不做" → A7 优先级降，可以 B（改 docstring）
4. 与 `_active_search_sessions` mutation guard 的关系：现在它是 "唯一并发防御"，新加 RLock 后是保留还是移除？

#### 决议

未决。

---

### A8 — offline_writers 属性名不一致（latent bug 嫌疑）

**类型**：Latent bug（独立于性能优化）

#### 背景

- 报告 §9.5 / 附录 A：`config.py:1768` `_collect_offline_writers_from_judges` 用 `extractors = getattr(judge, "_extractors", ())` 查 `_extractors` 属性。
- `composite_judge.py:137` 实际存 `self._factors: list[Factor] = list(factors)`。
- 字面上看，`getattr(..., "_extractors", ())` 在 CompositeJudge 上**总会回退到空 tuple `()`** → offline_writers 列表也总会是空。
- **三种可能解释**：
  1. **外层 wrapper** 在某条更外层路径上（如 DumpingJudge 内部）暴露 `_extractors`；研究未覆盖该 wrapping 层
  2. **真实代码 bug**：`on_episode_end` offline write 实际从未触发；可以通过 print 一次 sweep cell 即可证伪
  3. **`compute_for_episode` 协议根本未被任何启用因子使用**：整个路径目前是 dead code

#### 候选方案

- **方案 A — 直接修复**：改 `config.py:1768` 查 `_factors`（注意：需要先确认 `_factors` 类型 — 不是所有 factor 都是 OfflineWriter，需要再过滤 `isinstance(f, OfflineWriter)` 或类似）
- **方案 B — 加 `_extractors` 属性**：CompositeJudge 暴露 `_extractors` 作为 `_factors` 的 alias / 过滤视图，向后兼容
- **方案 C — 通过外层 wrapper 暴露**：保留两边不变，在 DumpingJudge 或其他 wrapper 上添加 `_extractors`
- **方案 D — 先实测确认是不是 dead code**：在 `on_episode_end` 打 print，跑一次 sweep cell 看是否被触发
- **方案 E — 删除 offline_writers 路径**：如果实测确认是 dead code

#### 各方案得失

| 方案 | Pros | Cons |
|------|------|------|
| **A 改 config.py** | 直接修复；改动小 | 需要先确认 `_factors` 是否需要过滤；可能引入 regression（如果某些 factor 不该被 offline write） |
| **B CompositeJudge 加属性** | 向后兼容；改动局部 | 属性 alias 显得冗余；属性命名上 `_factors` vs `_extractors` 语义可能不完全等价 |
| **C 外层 wrapper 暴露** | 不动 composite / config | 需要先研究 wrapper 是不是真有这层；可能找不到 |
| **D 先实测** | 数据驱动；可能发现是 dead code 不需要修 | < 30 分钟可验证 |
| **E 删除** | 如果是 dead code，删了减负 | 仅当 D 确认后才能选 |

#### 决策点

1. 优先 D 实测验证 → 根据实测结果决议 A / B / C / E
2. D 实测路径：在 `factors/offline.py` 的 `compute_for_episode` 入口或 `config.py:1768` 收集函数加 print，跑一个 verdict_factor 启用的 cell（如 phase5 任一 cell），看是否触发
3. 此议题与 A0 实测可合并

#### 决议

未决（推测下一步：先做 D 实测）。

---

## 3. 决议追踪表

> 议题决议后，executor 在对应行更新 "决议 / Round / 触发的 plan / 状态"。本表 append-only：决议变更通过新增 Round 实现，不重写旧行。

| 议题 | Round | 日期 | 决议 | 触发的 plan | 后续状态 |
|------|-------|------|------|------------|---------|
| A0 | — | — | 未决 | — | 待议事 Round 1 |
| A0 | 1 | 2026-05-23 | **D 跳过实测** — 项目所有者决议直接按报告估算拍板；后续 plan 阶段若 G1 reviewer 追问数字依据，按情况临时补实测或改写 plan 用 hedge 表述 | — | **关闭** |
| A0 | 1 (clarification) | 2026-05-23 | **D 含义澄清**：不是"永远不做"，而是"实测推迟到 §6 Verify 阶段用真实负载做"；executor 的 risk note 部分撤回（后续 plan 在 G1 前不需要补实测，但 plan 文件需明确"实测在 Verify 阶段执行"） | — | **关闭（含义已澄清）** |
| A1 | — | — | 未决 | — | 待议事 Round 1（依赖 A0） |
| A1 | 2 | 2026-05-23 | **A 全切单进程 concurrent** — 与项目最初目标"用一台设备跑一个 server"对齐；释放 ~9 GB 显存；`--concurrent` 已经是每 server 内部并发模型，切换本质是把 worker 数从 5 提到 15 | 待 A2 / A6 决议后联合起草 plan（A1 切换与多 yaml workflow 强耦合） | **关闭，pending A2 / A6 联合 plan** |
| A2 | — | — | 未决 | — | 待议事 Round 1（依赖 A0, A1, A7） |
| A2 | 3 | 2026-05-23 | **组合 ① 全 batching** — 主方案 A + CPU-1 保持 per-request 多线程 + A.1 动态 window + A.2 sub-batch split + A.3 per-request → stack；CPU 端零改动靠 OS 线程 + BLAS 多核继续天然并行；GPU 端做 stage1/2/3 三道 barrier batched forward | 待 A6 决议后与 A1/A2/A6 联合起草单一 plan（三议题强耦合） | **关闭，pending A1+A2+A6 联合 plan** |
| A3 | — | — | 未决 | — | 待议事 Round 1（依赖 A0） |
| A3 | 4 | 2026-05-23 | **D 不优化** — 项目所有者理由：当前主 workflow (phase5_libero10 / phase5_systematic / phase4 / phase3 / random_periodic_gate / warm_start) 全部用轻量 KeyBuilder (`cp1_spatial_pool_16` / `cp1_max_pool` / `cp1_mean_pool`)，连轻量都没优化，没必要先动重的；同议事 Round 4 项目所有者提出硬约束 **C1 保留非 concurrent 模式**作为极限速度基准 | — | **关闭，不孵化 plan** |
| A4 | — | — | 未决 | — | 待议事 Round 1（依赖 A0, A2） |
| A4 | 5 | 2026-05-23 | **D 不做** — A2 batching 后 kernel 时长拉长 stream overlap 边际收益小；与 `torch.compile` 兼容性（C1 约束的 non-concurrent 路径需要 compile）问号；YAGNI 与 A3 = D 同一逻辑 | — | **关闭，不孵化 plan** |
| A5 | — | — | 未决 | — | 待议事 Round 1 |
| A5 | 6 | 2026-05-23 | **A 删除 `_sync()` 用 CUDA Event 替代** — 与 cache 路径 `CudaEventBackend` 架构对齐；真正实现 C1 极限速度基准（baseline 路径不被 forced sync 拖累）；timing 功能保留；工程量小（~20 行） | 独立 L1 plan（与 A1+A2+A6 联合 plan 完全解耦） | **关闭，待启动独立 L1 plan** |
| A6 | — | — | 未决 | — | 待议事 Round 1 |
| A6 | 7 | 2026-05-23 | **A 改 `_current_bundle: dict[bundle_id, CurrentCacheBundle]`** —— A1 = A 必要配套；client `__ctrl__` 指定 bundle_id；**子优化 (PO 提出)**：按 pkl path 共享 backend 实例池，多个 yaml 用同一 pkl 时不重复加载（76 MB × K → 76 MB × distinct_pkl_count）；**触发新硬约束 C2**：runtime 数据库 read-only，A7 RLock 议题降级为 non-issue | 与 A1+A2+A6 合成**单一联合 plan**（推荐 K_max ≈ 6-8 与当前 phase5 6-server 拓扑对齐） | **关闭，待启动 A1+A2+A6 联合 plan** |
| A7 | — | — | 未决 | — | 待议事 Round 1（与 A2 耦合） |
| A7 | 8 | 2026-05-23 | **B 改 docstring + 加 frozen 守护** —— C2 让 RLock 路径过时；`InMemoryBackend.insert / delete` 加 `BackendFrozenError` 拒绝 runtime mutation；docstring 与代码同步到 C2 现实；hot path 零开销 | 合并进 A1+A2+A6 联合 plan（C2 守护本就是联合 plan 的一部分） | **关闭，合并进联合 plan** |
| A8 | — | — | 未决 | — | 待议事 Round 1 |
| A8 | 9 | 2026-05-23 | **A 修 bug** —— 改 `config.py:1768 _collect_offline_writers_from_judges` 查 `_factors` + `isinstance(f, OfflineWriter)` filter；audit report §9.5 揭示的属性名不一致闭环解决；保留 offline_writers 能力以备未来用 | 合并进 A1+A2+A6 联合 plan（顺手修属性名 bug，与 `build_per_connection_components` 函数边界相邻） | **关闭，合并进联合 plan** |

---

## 4. 后续 Plan 触发表

> 议题决议为 "实施" 后，本表追加对应独立 plan 的文件名 / 预估 level / 启动状态。每份 plan 必须独立经 §3 G1。

| 议题决议 | 触发的 plan 文件（命名待定） | 预估 Level | 启动状态 |
|---------|----------------|----------|---------|
| **A5 = A** 删除 baseline `_sync()` 用 CUDA Event 替代 | `logs/baseline_sync_removal_plan.log.md`（建议名） | **L1** | 待项目所有者确认启动时机；与联合 plan 完全解耦可并行 |
| **A1 = A 全切单进程** + **A2 = 组合 ① 全 batching** (CPU-1 + 动态 window + sub-batch split + per-request transform→stack) + **A6 = A `dict[bundle_id]` + pkl-shared backend pool** + **A7 = B frozen 守护** + **A8 = A 修 `_extractors` → `_factors` filter** | `logs/concurrent_serving_optimization_plan.log.md`（建议名） | **L3** | 待项目所有者确认启动时机；硬约束 C1 + C2 必须在 risk register 显式引用 |

---

## Discussion Log

> append-only。每议题每轮讨论按 `### Council Round N — Topic A<x>` 分块。Round 0 = 议程立项 = 当前状态。

### Council Round 0 — 议程立项 — 2026-05-23 (Executor)

- 议程已立 8 议题（A0 实测前置 / A1 进程模型 / A2 Request Batching 核心 / A3 重 KeyBuilder 异步化 / A4 多 CUDA stream / A5 Baseline `_sync()` 清理 / A6 多 yaml 共存 / A7 Backend RLock 一致性 / A8 offline_writers 属性名不一致）。
- 每议题已给出 候选方案 + 各方案得失 + 决策点。
- 议事顺序建议：从 A0 开始按表格顺序逐一讨论；A5 / A7 / A8 是独立议题，可与主轴并行讨论。
- 等待项目所有者指定 Round 1 的起始议题。

### Council Round 1 — Topic A0（实测先于决策）— 2026-05-23

**Executor opening**：
- Recap 4 候选方案（A 串行 / B 并行 / C 按需 / D 跳过）
- Executor 推荐 **B 并行**，理由：议事 Round 间天然有 wait，实测可填空隙；A2 议事需要 #4 microbench 数据；非性能议题（A5/A7/A8）不依赖实测可并行推进
- 5 项实测优先级建议：P0 SystemTimer CSV 回推（零成本，已有数据）→ P1 batched microbench（关键，驱动 A2）→ P2 nvidia-smi dmon → P3 py-spy → P4 nsight-systems

**Project Owner decision**：**D 跳过**。直接按报告估算拍板，不做前置实测。

**Executor closing & risk note**：
- 接受 D 决议。
- Voice 一个 risk reminder（不阻止决议）：后续 A1-A8 决议为 "实施" 时孵化的独立 plan，G1 reviewer 大概率追问数字依据（Audit R3 scope 明确禁止用估算做工程决策）。两种应对：(a) 后续 plan 写得不依赖具体数字、用 hedge 表述；(b) 接受 G1 因数字未实测 NEEDS REVISION 一轮再补测。
- 本议题关闭。决议追踪表 A0 行追加 Round 1 决议。
- 下一议题：等项目所有者指定（默认按议程顺序进入 A1 — 进程模型）。

### Council Round 1 — Topic A0 — clarification — 2026-05-23

**Project Owner clarification (post-decision)**：A0 = D 的实际含义是 "实测推迟到代码改完跑真实负载阶段做"，**不是 "永远不做"**。

**Executor closing (revised)**：撤回 Round 1 closing 的 risk note 的一部分 —— 后续 plan 在 §3 G1 前**不需要**补前置实测；plan 文件需要明确写出 "实测在 §6 Verify 阶段用真实负载执行"，G1 reviewer 据此理解决议依据即可。本议题最终关闭。

### Council Round 2 — Topic A1（进程模型）— 2026-05-23

**Executor opening**：4 方案速查（A 全切单进程 / B 折中 2 进程 / C 现状 3 进程 / D 保留两种模式）。推荐 **A** —— 与项目最初任务陈述 "用一台设备跑一个 server，充分利用设备资源" 直接对齐；9 GB 显存释放是 immediate benefit（即使 A2 不做 batching 也可挂更多 yaml / 扩 KV cache window）；`--concurrent` 已是每 server 内部并发模型，切换本质是把 worker 数从 5 提到 15；主要 unknown：15 worker 强度下 stability load test 未做（按 A0 决议推迟到 Verify 阶段验证）。

**Project Owner decision**：**A 全切单进程**。

**Executor closing**：
- A1 = A 决议关闭。
- 下游影响：sweep workflow 的多 yaml 并行**不能再靠多 server 进程承担**，必须由 A6（多 yaml 共存 `dict[bundle_id]`）或时间切片承担 → **A6 优先级从原 "中长期方向" 升为 "A1 = A 的必要配套"**。
- A1 的独立 plan 启动时机：待 A2 / A6 决议后联合起草。理由：A1 切换的实际工作量 = 改 `serve_policy.py` 启动模式 + sweep workflow 协议改动，与 A6 plan 高度重叠。
- 下一议题：A2 — Request Batching。

### Council Round 3 — Topic A2（Request Batching）— 2026-05-23

**Executor opening**：4 主方案（A 全 batching / B Micro-batching / C 不做 / D 推迟）+ CPU 端 3 策略（CPU-1 保持 per-request / CPU-2 改 batched API / CPU-3 混合）+ 主方案 A 的 3 子选择（A.1 batch window / A.2 决策分叉 / A.3 transform 路径）。推荐组合 ①：A + CPU-1 + 动态 window + sub-batch split + per-request → stack。

**议事中的项目所有者澄清问题（3 项）**：
1. **"CPU 端怎么做"** — verify 当前 CPU 已天然多核并行：OS 线程级（`asyncio.to_thread` 默认 `ThreadPoolExecutor(max_workers=min(32, cpu_count()+4))`）+ BLAS 多核（OpenBLAS / MKL 受 OMP_NUM_THREADS 控制，自己 spawn 多核）+ GIL 释放叠加。隐性问题：若 `OMP_NUM_THREADS` 未设可能 oversubscribe（N 个 worker × M 核 BLAS > cpu_count），独立 hygiene 项。
2. **"会话隔离怎么做的"** — verify 两层机制：(a) 每连接 fresh wrapper stack — `_connection_policy_factory(shared_base)` → `build_per_connection_components` 全新构造 `SystemTimer / KeyBuilder / Gate / Judge / SearchStrategy / Interceptor / Orchestrator` + per-connection `CacheStorage` facade；(b) 共享 backend 用 UUID4 `_search_session_id` 切分内部状态 — sid 在 `SearchStrategy.on_episode_start` mint (`search_strategy.py:119`)，Orchestrator 通过 `storage.open_search_session(sid)` 注册到 backend (`in_memory_backend.py:113-119`)，`_score_memo` 外层 key 是 sid (`in_memory_backend.py:91`) → 不同连接 sid 不同 → 落到不同 bucket 物理无冲突；GIL 保证 dict/set 单 op 原子。
3. **"per-request transform → stack 是什么"** — Transform 系统功能（input: repack/InjectDefaultPrompt/data_transforms/Normalize/TokenizePrompt；output: 反向链）；batched 路径 (a) 在 batching coordinator 层加 stack/unstack，transforms.py 零改动；vs (b) 系统性 batched transform 工程量大但 transform 阶段本身只占 5-20 ms（推理估 250-650 ms），投产比不划算。

**Project Owner decision**：**组合 ①**。
- 主方案 A（全 batching）
- CPU 端 CPU-1（保持 per-request 多线程，CPU 端零改动）
- A.1 动态 batch window（先到先服务，凑够 K 或 N ms 超时）
- A.2 sub-batch split（CP1 后按 hit_type 拆，FULL_HIT 直接返回）
- A.3 per-request transform → stack（transforms.py 零改动）

**Executor closing**：
- A2 决议关闭。
- 关键事实记录（供联合 plan 调用）：
  - CPU 端检索路径 lock-free read + sid namespace 切分 + GIL atomic 单 op → **天然兼容 batching CPU-1**
  - **A7（Backend RLock）在 CPU-1 路线下不是 blocker**（不引入"同 sid 多线程并发"或"同 entry 复合写"路径）
  - 组合 ① 的物理流：N 个 obs → per-request transform → barrier(stage1 batched) → per-request KeyBuilder collect/build + CP1 search + CP1 judge → barrier(stage2 batched，FULL_HIT 提前返回) → per-request CP3 → barrier(stage3 batched) → per-request output_transform → N 路回复
- **A2 plan 启动时机**：A2 + A1 + A6 三议题强耦合（单进程模式 + batching coordinator + 多 yaml workflow），起草**单一联合 plan**而非三份独立 plan，待 A6 决议后启动。
- 下一议题：A3 — 重 KeyBuilder 异步化（CLIPKeyBuilder + LLMLayerExtract）。

### Council Round 4 — Topic A3（重 KeyBuilder 异步化）— 2026-05-23

**Executor opening**：议题速记 + 当前 sweep 使用情况调查（verify phase5 系列只用 `cp1_spatial_pool_16`，CLIP yaml 集中在 archive 标 superseded 的 phase0/1/2 早期实验）。5 候选方案（A 异步化+batched / B 替换轻量 / C gate / D 不优化 / E 推迟），推荐 D。

**Project Owner decision**：**D 不优化**。理由：当前轻型 keybuilder 也没优化，没必要先动重的。

**Project Owner additional constraint (C1)**：**保留非 `--concurrent` 模式作为极限速度测试基准**，A2 batching / 任何优化改动不得破坏该路径。

**Executor closing**：
- A3 = D 关闭，不孵化 plan。
- **C1 硬约束写入 §0.4**，对 A1+A2+A6 联合 plan 具有约束力 → batching coordinator 必须 opt-in，non-concurrent 路径完全保留 (`Policy.infer` + `interceptor.py` non-eager 路径 + `torch.compile` 编译产物零改动)。
- 下一议题：A4 — 多 CUDA stream。

### Council Round 5 — Topic A4（多 CUDA stream）— 2026-05-23

**Executor opening**：5 方案（A per-connection / B per-stage / C KeyBuilder 独立 / D 不做 / E 推迟）。C 已被 A3 = D 否决不可选。A2 batching 后 GPU 主要矛盾已解决；stream overlap 在 batched kernel 时长拉长后边际效益小；与 `torch.compile`（C1 约束保留的 non-concurrent 路径需要 compile）兼容性需额外验证；推荐 D。

**Project Owner decision**：**D 不做**。

**Executor closing**：
- A4 = D 关闭，不孵化 plan。
- 与 A3 = D 同一逻辑（先做主优化，附属优化等需要再做）。YAGNI。
- 下一议题：A5 — Baseline path `_sync()` 移除。

### Council Round 6 — Topic A5（Baseline path `_sync()` 移除）— 2026-05-23

**Executor opening**：C1 硬约束触发 A5 语境变化——项目所有者要把 non-concurrent 路径作为极限速度测试基准，`policy.py:104-129` `_sync()` 每 stage 强制 GPU flush 是限速点。3 方案（A 删除 + CUDA Event 替代 / B 加 flag / C 不动）。推荐 A：与 cache 路径 `CudaEventBackend` 架构对齐 + 工程量小（~20 行替换）+ 真正实现 C1 极限速度精神。

**Project Owner decision**：**A 删除 `_sync()` 用 CUDA Event 替代**。

**Executor closing**：
- A5 = A 关闭。
- Plan 路径：独立 L1 plan（与 batching / concurrent / A1+A2+A6 联合 plan 完全解耦），可并行推进。Plan 启动时机由项目所有者决定。
- Plan 阶段必须 verify 项：grep `_sync\|cuda.synchronize` 找下游依赖（按 audit report §3.3 应该没有，但要 verify）。
- 下一议题：A6 — 多 yaml 共存（已被 A1 = A 升级为必要配套）。

### Council Round 7 — Topic A6（多 yaml 共存）— 2026-05-23

**Executor opening**：A1 = A 后多 yaml 不能再靠多 server 进程承担，必须由 A6 承担。3 方案：A `dict[bundle_id]` / B 时间切片串行 / C 不动（已被 A1 = A 否决）。推荐 A：A1 必要配套；与 A2 batching 协同（不同 yaml 共享模型权重 → 可混 yaml 同 batch forward）；WarmupPool 已是 per-yaml LRU 算配套基础。

**Project Owner decision**：**A 改 dict[bundle_id]**。
- **子优化 1（PO 提出）**：**按 pkl path 共享 backend 实例池**。当前架构 `build_shared_storage` 每 yaml 一份 backend；许多 yaml 共用同一份 pkl（如 phase5 系列全部 `cp1_spatial_pool_16.pkl`），不应重复加载 76 MB。改为按 pkl path 做 backend 实例池，K 个 yaml 引用 distinct_pkl_count 个 backend instance。
- **子优化 2（PO 提出 → 升级为硬约束 C2）**：**服务器运行期间禁止修改数据库内容** → backend `_entries` runtime read-only → 不需要 RLock → A7 RLock 议题自动降级为 non-issue。

**Executor closing**：
- A6 = A 关闭。
- 子优化 1 (pkl-shared backend pool) 与 A6 主决议合并写入 plan 数据结构层：
  - `_current_bundle: dict[bundle_id, CurrentCacheBundle]`
  - `_backend_pool: dict[pkl_path, Backend]`（新）
  - `CurrentCacheBundle.storage` 引用 `_backend_pool` 中 entry
- 子优化 2 升级为硬约束 **C2**，写入 §0.4。**直接影响 A7**（下一议题）—— C2 让 RLock 漏洞从 "潜在 bug" 降为 "non-issue"，runtime 共享 backend 全 read-only + GIL 原子 dict lookup = 100% 安全。
- **A1 + A2 + A6 联合 plan 启动条件**：A6 已是议程上的最后耦合议题，本轮闭合后**联合 plan 可以启动**（待项目所有者确认时机）。
- 下一议题：A7 — Backend RLock 一致性（C2 触发预决议，议事 Round 应很短）。

### Council Round 8 — Topic A7（Backend RLock 一致性）— 2026-05-23

**Executor opening**：C2 已让 A7 几乎自动决议。原 5 方案被 C2 削平为 2 个：A 补 RLock（不必要 + 性能税）vs B 改 docstring + 加 frozen 守护（与 C2 一致 + hot path 零开销）。推荐 B + 合并进 A1+A2+A6 联合 plan。

**Project Owner decision**：**B 改 docstring + 加 frozen 守护**。合并进联合 plan。

**Executor closing**：
- A7 = B 关闭。
- 解决方式：`InMemoryBackend.insert / delete` 加 `BackendFrozenError` 拒绝 runtime mutation；`backend_base.py:14-16` docstring 改为明确 "runtime write-frozen，不需要 RLock"；与 audit report §9.1 揭示的 "声明 vs 实现不一致" 问题对齐解决。
- 下一议题：A8 — offline_writers `_extractors` vs `_factors` 属性名不一致（议程上最后一个议题）。

### Council Round 9 — Topic A8（offline_writers 属性名不一致）— 2026-05-23

**Executor opening**：A0 = D 把 "D 先实测确认是不是 dead code" 推后到 Verify 阶段；议事阶段在 3 方案（A 修 bug / E 删除 dead code / F 不动）中决议。推荐 E 删除（dead code 旁证：`--factors-yaml` 离线 enrich 是更好替代；现有 sweep 无 yaml 触发 `compute_for_episode` 协议）+ 合并进联合 plan。

**Project Owner decision**：**A 修 bug**。理由：保留 offline_writers 能力以备未来用（不删除）。

**Executor closing**：
- A8 = A 关闭。
- 解决方式：改 `config.py:1768 _collect_offline_writers_from_judges` 查 `getattr(judge, "_factors", [])` + `isinstance(f, OfflineWriter)` filter；audit report §9.5 揭示的属性名不一致闭环。
- Plan 路径：合并进 A1+A2+A6+A7+A8 联合 plan（与 `build_per_connection_components` 函数边界相邻，集中改一次）。
- **A8 是议程上最后一个议题，本轮闭合后 9 议题全部决议**。

---

### Council 议事结束 — 2026-05-23

**议程闭合状态**：A0 (D 跳过实测) / A1 (A 单进程) / A2 (组合 ① batching) / A3 (D 不优化) / A4 (D 不做) / A5 (A 删 _sync()) / A6 (A dict[bundle_id] + pkl 池) / A7 (B frozen 守护) / A8 (A 修属性名 bug) — **全部 9 议题决议完毕**。

**孵化的 plan**（2 份）：
1. **独立 L1** — Baseline `_sync()` 移除（A5）— 与联合 plan 完全解耦，可并行推进
2. **联合 L3** — Concurrent Serving Optimization（A1 + A2 + A6 + A7 + A8）— 单 plan 解决单进程模式 + batching coordinator + 多 yaml dict[bundle_id] + pkl-shared backend pool + runtime frozen 守护 + offline_writers 属性名修复

**硬约束**（联合 plan 必须遵守，见 §0.4）：
- **C1** 保留非 `--concurrent` 模式作为极限速度测试基准（不破坏 non-concurrent 路径）
- **C2** 服务器运行期间禁止修改数据库内容（runtime backend write-frozen + frozen 守护）

**议事 log status 转换**：`In Progress` → `Plan`（议事完成，待启动 2 个孵化 plan）。本 log 保留在 `logs/` 顶层 Active 作为联合 plan 起草时的依据；按 audit report 归档误判教训，**归档动作需要项目所有者明确指令**。

### Plan Kickoff Amendment — 2026-05-23（项目所有者追加 L3 plan scope）

议事结束 + L3 联合 plan 启动指令同时，**项目所有者追加需求**：

> "实验的基础设施要做好，我们可以需要一个自动化的脚本来探索吞吐量和 latency 的极限和关系，比如请求从稀疏到越来越多"

**Scope amendment**：L3 联合 plan 范围**新增 Module M7 — Throughput/Latency Benchmark Tool**。

- 与服务端优化 (M1-M6 = A1/A2/A6/A7/A8 落地) **并列且同等优先**
- 用途：
  - Verify 阶段实测 baseline (C1 non-concurrent) vs concurrent batching 的真实吞吐 / latency 曲线
  - 扫描 worker 数 / 请求频率 / yaml 数 / batch_window 等维度
  - 输出 throughput vs latency Pareto frontier 用于 Plan 数字 hedge 表述的事后验证（对应 A0 = D 的 Verify 阶段实测策略）
- 与 audit report §B.3 实测建议直接对应（特别是第 4 项 batched-infer microbench）
- 文件落点：`exp/serving_benchmark/`（新目录）

本 amendment 直接写入 L3 plan §0 scope，不另立独立 plan（与 M1-M6 同 plan 推进，确保 benchmark 在 verify 时就绪）。
