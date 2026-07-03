# Gate 组件设计方案头脑风暴 — CP1"搜不搜"决策

- **Status**: Design Only
- **Date**: 2026-07-02
- **Updated**: 2026-07-03 — 整合 Opus 独立方案（§4 末"双模型对照"）；按 owner 判断，V1（省延迟）视为有效收益方向、与 V2/V3 并列，原"V1 不成立"论证已移除（延迟量级数据保留为正向参考）；按 owner 更正，明确标注 `deviate_score` 为 GT 事后 oracle 量（线上不可得），仅用于上限 / 离线标签，非 gate 输入（§1.5 / TL;DR #2 醒目声明，oracle 分析保留）；按 owner 决定剔除 B1 视觉变化门 / B2 动作动力学门（信号与 verdict kinematic 因子重合，无独立于 verdict 的增量），Tier B 仅留 B3 / B4
- **Level**: 本文档为 Understand 阶段产物（L0 纯文档）；任何入选方案的实现均为 L2（新 GateFunction 组件），需另走 Plan → G1 → Code → G2 → Verify
- **范围**: 只讨论 gate（搜索前"搜不搜"决策）。search / verdict 内部机制不在本文重设计范围；与它们的协同接口（如 verdict 结果回传）属于本文范围。

---

## 0. TL;DR

1. **gate 有三个真实价值**：**V1 省延迟**（每拦截步省掉 search+judge+fetch）、**V2 保成功率**（在 verdict 之前拦掉难步/危险复用）、**V3 SR–算力预算旋钮**（把 RandomGate/PeriodicGate 的盲 Pareto 曲线往左上推）。三者可由同一决策统一（见 §4 末 Opus 期望值判据）。V1 的省量随部署栈变化：优化栈 ~4 ms/步、stock `InMemoryBackend` ~31–34 ms/步（`exp/cache_latency_bench/analysis/latency_breakdown.md`），且随库规模线性增长、远程 backend 下含网络往返——在高吞吐 serving 与大库场景下按并发放大，是实打实的收益（量级见 §1.4）。
2. **V2 有一个 oracle 上限（非可实现值）**：`trajectory_deviation` step3 的定向拦截用的是 `deviate_score`——**拿数据库 GT 标准答案事后算出的偏差，线上根本不可得**。因此它衡量的是"若难步检测器完美"的天花板，**不是任何真实 gate 能拿到的成绩**。该 oracle 换算到全体口径后相对 random/periodic 盲拦截约 **同预算 +8~12 pp SR / 同 SR 省 ~40% 推理预算**（保守口径下缩到与 periodic 最优点打平；细节见附录 B）。**真实 gate 的收益落在盲基线与该 oracle 之间，取决于搜索前廉价信号对难步的预测力——这正是 G0b 要回答的。**
3. **"难步"高度集中且有结构**：step2 显示大偏离 cycle 只占 6–7%（每 episode 约 1–1.5 个），集中在轨迹中段+末段，task 间差异显著（task_8 最危险）。廉价先验（步位置、任务、库覆盖）+ 元信号（连续命中债务、卡死检测）都有明确的物理依据可挖（视觉变化 / gripper 类步难度信号归 verdict，不作 gate 信号）。
4. **首个行动建议不是写 gate，而是做一个离线信号预测力研究（方案 G0b）**：用已有的 step2 偏差分数 + DumpingJudge JSONL 标签，量化每个候选廉价信号对"难步/白搜步"的预测力（AUC / P-R）。一周量级、零风险、直接决定后续投入方向。与之并行可落地的最小在线方案是 **B3 复用债务门**（零训练、信号独门、periodic>random 的实验证据直接支持其机制）。
5. 所有在线原型都可以先走 **ClientControlledGate + `__gate_decision__` 通道在 exp 层实现**（step3 已趟通此路），零 src 改动；服务器化留到方案定型后。

---

## 1. 现状核实

### 1.1 check 流水线与 gate 的真实位置

源码（`src/openpi/cache/orchestrator.py:414-455`）的实际顺序：

```
collect → gate → build(D2H, 无条件执行) → [gate skip? → record_query_keys → 返回 MISS]
        → search → judge → fetch
```

三个关键事实：

- **gate 在 search 之前，天生看不到搜索结果** ✅（用户信息 1 属实）。
- **gate 读到的是 D2H 之前的 GPU 原始张量**（`key_builder.cached_data`：`state [B,32]` + `prefix_embs [B,~768+,2048]`）✅；**但 gate skip 并不能省掉 build/D2H**——`build()` 无条件执行以保证轨迹历史 gap-free（`orchestrator.py:423-425`）。gate 可省的只有 search + judge + fetch。
- **文档失真**：`docs/architecture/cache_system.md` §5.1 与 `docs/cache/tutorial.md` §3.1 写的顺序是 collect → build → gate，与源码不符（gate 实际在 build 之前）。两处文档需要修正（另行 L0 文档变更）。

### 1.2 gate 现有实现

不止 `AlwaysSearchGate`。`src/openpi/cache/components/gate.py` 现有 5 个实现，但全部是固定策略或外部遥控，**没有任何基于信号的自适应 gate**：

| 实现 | 行为 | 用途 |
|---|---|---|
| `AlwaysSearchGate` | 永远搜 | 默认 |
| `AlwaysSkipGate` | 永远跳 | trajectory_deviation Step 2 背景采样 |
| `RandomGate(p_inference, seed)` | Bernoulli 随机强制推理 | 盲拦截基线实验 |
| `PeriodicGate(cache_len, inference_len)` | 每 k 步缓存后强制 n 步推理 | 盲拦截基线实验 |
| `ClientControlledGate` | 客户端 per-request 遥控（`__gate_decision__`） | step3 oracle 实验 |

配置扩展点齐备：`GateConfig` dataclass + `_GATE_TYPES` + `_build_gate` factory + `validate_cache_config` per-type 参数校验（`config.py:82-89, 440, 1130-1190, 2025-2052`）。新增 gate 类型是模式化的四处小改。

### 1.3 gate 手上有什么 / 没有什么

| 信息 | 现状 | 备注 |
|---|---|---|
| GPU 原始 `cached_data`（state + prefix_embs 全量 tokens） | ✅ 有 | 信息量大于 build 后的 pooled key；含 prompt_emb 段（任务语言指令的嵌入） |
| `request_context` | ✅ 有 | 目前仅 client_controlled 用（`gate_decision`） |
| step 序号 | ⚠️ 自己数 | 签名不含 step；`PeriodicGate` 已有自数先例，`on_episode_start` 重置 |
| 本 episode 动作历史 | ✅ 有 hook | `record_action` 广播已就位（orchestrator.py:354-356） |
| task_key / episode 身份 | ❌ 没有 | orchestrator 只把 `extra_metadata` 广播给 judge（`_broadcast_episode_start`，orchestrator.py:282-286）；gate 的 `on_episode_start` 无参。补上是小改（方案 G0a）。间接替代：从 prompt_emb 自建任务签名 |
| 上一步 verdict 结果（hit_type/score） | ❌ 没有 | gate 收到的 `record_action` 不区分动作来源。B3 类方案需要此回传（方案 G0a） |
| 搜索结果 / 库内容 | ❌ 架构红线 | gate.py 头注释：gate 不与 CacheStorage 交互。库相关信号必须走离线预计算查表 |

### 1.4 延迟账本（gate 每步可省的量级）

`exp/cache_latency_bench`（libero_10 / spatial_pool_16 / ~2640 条库 / CPU 回放）实测中位数：

| 段 | 优化后 | stock src |
|---|---:|---:|
| collect | 0.005 ms | 0.006 ms |
| gate | 0.002 ms | 0.002 ms |
| build（gate 省不掉） | 0.446 ms | 0.948 ms |
| **search（gate 可省）** | **3.539 ms** | **33.921 ms** |
| judge（gate 可省） | 0.009 ms（threshold）/ 0.431 ms（kinematic composite） | 0.013 ms |
| fetch（gate 可省） | 0.004 ms | 0.005 ms |

- gate 每次拦截可省 search+judge+fetch：优化栈 **~3.5–4.2 ms**，stock 栈 **~31–34 ms**（build/D2H 无条件执行，省不掉）。
- brute-force search 随库条数线性增长：库到 ~50k 条时单次 search 推回 ~70 ms 量级；远程 backend 另加网络往返。库规模 / backend 变更时重跑该 bench 即可拿到当档省量。
- 高吞吐 serving（每秒多请求 × 多并发）下，per-step 的毫秒级省量按吞吐放大，是实打实的收益；单条闭环延迟（~1.7 s 量级）视角下占比小，但那不是唯一部署视角。"先量化 search 成本"这个前提已由本 bench 量化，无需重做。

### 1.5 已有实验资产（gate 研究的三大件）

1. **盲拦截 Pareto 基线**（`exp/random_periodic_gate/analysis/aggregate.csv`，3 keybuilder cfg × 26 点 × 500 ep）：SR–inference_ratio 全曲线。两端点：纯缓存 SR 0.674–0.696，纯推理 0.984–0.992。**关键模式：periodic 显著优于 random**（如 clip cfg @34% 预算：periodic k5_n3 SR=0.920 vs random p0.3 SR=0.815）——连续拦截块比独立随机拦截好得多，强烈暗示"错误积累需要连续几步纠正"。
2. **难步 ground truth + oracle 上限**（`exp/trajectory_deviation`）：step2 对 failure subset 每 cycle 算了 `deviate_score = cache_l2 / max(background_l2, 0.1)`（缓存回放动作相对 GT 的偏差，归一化到推理自身波动）；step3 用该分数做 oracle 定向拦截（tau 触发 + 连拦 n 步），通过 `__gate_decision__` 遥控。难步分布结论：≥5 的大偏离 cycle 占 6–7%，每 episode 约 1–1.5 个，集中在轨迹 20–60% 中段与 80–100% 末段，前 20% 最安静；task 差异大于 keybuilder 差异（task_8 最危险，12.9%）。
   > **⚠ `deviate_score` 是 GT 事后 oracle 量，线上不可得。** 它由"缓存回放动作 vs 数据库 GT 标准答案"算出，而拿到 GT 就等于已经跑了那次本想跳过的真实推理。因此 `deviate_score` **绝不是任何真实 gate 的输入信号**，只有两个离线用途：(a) 作 oracle **上限**（本报告 step3 / §2.3 / 附录 B 的全部结论都是天花板，非可实现值）；(b) 作**离线标签**，训练/验证"线上可得的廉价信号"对难步的预测力（G0b / D1）。§4 方案谱系里所有 gate 均只用搜索前可得信号，无一以 `deviate_score` 为输入。
3. **步级标签管线**（DumpingJudge JSONL + `__hit_meta__` schema v2）：每步导出 `hit_type / winner_id / cp1_score / start_t / 17 因子原值 / task_id / step_idx`，客户端可 per-step 记录。另外 weighted_sum trajectory step-weight 系列（twa 主跑 17,100 ep 在产）会带来大规模步级数据。**注意 selection bias：gate skip 的步不产生 judge/dump 行——学习型 gate 的训练数据必须用 always_search 配置采集**。

---

## 2. 前提质疑与价值重估

### 2.1 V1（省延迟）：真实收益，随部署栈放大

见 §1.4。gate 每拦截步省掉 search+judge+fetch（优化栈 ~4 ms、stock ~31–34 ms），省量随库规模线性、随并发吞吐放大、远程 backend 下含网络往返。V1 与 V2 并列为 gate 的立项理由；设计上让"省搜索开销"成为第一类目标（G0b 的"白搜步"预测目标即为此服务），而非事后副产品。约束仅一条：**gate 自身在线成本必须 ≪ 被省的 search 成本**（见 §2.2 边界条件 2 与 C2）——否则本来会 FULL_HIT 的步也白付 gate 成本。

### 2.2 "gate 低风险"论断：成立，但有三个边界条件

主论断成立：gate skip → 完整推理，与 no-cache 同路径，**SR 不会低于 no-cache 基线**；相对 always_search，gate 误拦一个本可安全 FULL_HIT 的步只损失该步的延迟收益，不损失 SR。但：

1. **写路径侧效应**：gate skip 的步仍会 `buffer_for_write`；若 write_policy 开启，gate 会改变入库轨迹的构成（更多真实推理步入库——通常是好事，但改变了库分布，跨实验对比时要注意）。当前主实验多为冻结库（preload artifact + never write），影响可控。
2. **gate 自身成本**：论断的前提是 gate 便宜。若 gate 做重计算（跑网络、强制 GPU 同步点），每步都付出该成本（包括本来会 FULL_HIT 的步），可能吃掉全部 V1 并拖累 hit 路径延迟。约束：gate 在线成本 ≪ 1 ms。
3. **系统层面的道德风险**：若把 gate 当作正确性保险，可能诱使 verdict 阈值调得更激进，整体反而变脆。规则：**gate 与 verdict 独立标定，gate 是第二道保险而非 verdict 放松的理由**。

### 2.3 V2 上限量化：oracle vs 盲拦截（核心证据，含口径警告）

step3 oracle（事后偏差分数选步）与 RPG 盲拦截不在同一坐标系上（step3 在 failure subset 上跑，RPG 在全体 500 ep 上跑），换算细节见附录 B。结论区间：

- **官方保守口径**（y 转全体、x 保持 subset 轴，`analyze_gate_sweep.py` 的 overlay 口径）：oracle 最优点（tau=3,n=2：x≈0.30, SR≈0.90–0.93）与 periodic 最优点（k5_n3：x=0.34, SR≈0.91–0.92）**基本打平**。
- **合理全体口径**（成功 episode 上 oracle 几乎不触发，全体 inference_ratio 折到 ~0.15–0.20）：oracle 同预算比盲拦截 **+8~12 pp SR**，或同 SR **省 ~40% 推理预算**。
- oracle 本身用的是在线不可得的特权信息（GT 对照的事后偏差）。**在线 gate 的可实现收益落在盲基线与 oracle 之间**，位置取决于搜索前信号对难步的预测力——这正是 G0b 要回答的问题。

另一个被两组数据同时支持的结构性结论：**拦截必须成块**。step3 中 n=1 在所有 tau 下都显著差于 n≥2（clip tau=3: n=1 SR=0.61 vs n=2 SR=0.77）；periodic 优于 random 同理。单步纠正拉不回轨迹。

### 2.4 重新表述 gate 的价值主张

- **V1（省延迟）**：每拦截步省掉 search+judge+fetch，随库规模 / 并发吞吐 / 远程 backend 放大（§2.1 / §1.4）。对应的搜索前预测目标是"白搜步"（verdict 会判 MISS 的步）。
- **V2（保成功率）**：在 verdict 之前、不依赖搜索结果地识别"此刻不该信任缓存机制本身"的步，强制连续 n 步真实推理。与 verdict 的分工：verdict 回答"库里最好的候选够不够好"，gate 回答"这一步该不该把命运交给缓存"——后者在轨迹敏感期（接触/抓取/放置）应当为否，*无论库里有多像*。
- **V3（预算旋钮）**：gate 是 SR–算力曲线上的**部署旋钮**。RandomGate/PeriodicGate 已经是这个旋钮的盲版本并有完整 Pareto 数据；智能 gate 的目标是把整条曲线往左上推。验收标准：**同 inference_ratio 下 SR 必须打败 periodic（不是打败 random）**。
- **三者统一**：V1 与 V2 可折进同一个期望值判据（§4 末 Opus 补充），V3 是该判据的阈值 / 预算旋钮化。

---

## 3. 设计约束（公理集）

| # | 约束 | 来源 |
|---|---|---|
| C1 | gate 不读写 CacheStorage；库相关信号一律离线预计算 | gate.py 头注释的架构红线 |
| C2 | 在线成本 ≪ 1 ms，禁止在 gate 内制造 GPU 强制同步点（GPU 上算子异步发射，仅决策时一次标量同步；或干脆用 CPU 侧历史信号） | §2.2 边界条件 2 |
| C3 | 错误代价不对称：误拦 = 损失一次 ~1.7 s 延迟收益；漏拦 = episode 失败概率上升。初版阈值保守（高 precision 拦截），拦截率从低往上调 | §2.2 / §2.3 |
| C4 | 拦截成块：触发后强制连续 n≥2 步推理（接口上 gate 内部自持冷却/滞回状态即可，无需改协议） | step3 n=1 数据 + periodic>random |
| C5 | 学习/标定数据必须来自 always_search 配置（避免 selection bias）；gate 上线后的数据不能直接回流训练 | §1.5 |
| C6 | gate 与 verdict 独立标定；评估必须同时报 SR、inference_ratio、FULL_HIT 率三元组 | §2.2 边界条件 3 |
| C7 | 每 episode 前 K 步（temporal buffer 未满期）默认放行搜索——step2 显示前 20% 轨迹偏差最小，且缓存在此段收益最稳 | step2 位置分布 |

---

## 4. 方案谱系

总览（成本 = 在线每步开销；难度 = 落地工程量含标定）：

| 方案 | 信号（搜索前） | 成本 | 主要收益 | 难度 | 一句话风险 |
|---|---|---|---|---|---|
| G0a hook 补丁 | — | — | 解锁 A2/B3 | 极低（src 小改） | 无 |
| G0b 离线信号研究 | 全部候选信号 | 离线 | 决定投入方向 | 低（纯分析） | 结论可能是"都没预测力" |
| A1 步位置门 | 自数 step + episode 相位 | ~0 | V2 弱 | 极低 | 任务平均相位对个体失准 |
| A2 任务先验门 | task_key / prompt_emb 签名 | ~0 | V2 中（高危任务） | 低 | 粒度粗，误伤简单步 |
| A3 库覆盖门 | 离线密度表 + 当前 key | O(k·d) 微秒 | V1+V2 | 中 | 表签名过粗；库更新需重算 |
| B3 复用债务门 | 连续命中计数 + 累计漂移 | ~0 | V2 强 | 低-中（需 G0a） | 预算参数需扫描 |
| B4 停滞检测门 | 视觉+状态 N 步不变 | ~0 | V2（防灾） | 低 | 静止等待型任务误报 |
| C1 标定组合门 | A/B 全部信号 → 双头查表/LR | 微秒 | V2+V1 | 中 | 标定域漂移 |
| D1 学习难度门 | pooled 嵌入 + 时序特征 → 小 MLP | <1 ms GPU | V2 上限 | 高 | 过拟合 suite、管线重 |

### Tier 0 — 前置工作（先于一切 gate 实现）

**G0a. 信号可见性补丁**（L2 小改，src）
- `_broadcast_episode_start` 给 gate 的 `on_episode_start` 也传 `task_key` / `extra_metadata`（现在只给 judge，orchestrator.py:280-286）；新增 verdict 结果回传 hook（如 `record_check_result(hit_type, score)`，orchestrator 在 `check()` 返回前广播给 gate，与 `record_action` 同款模式）。
- 这是 A2 / B3 的解锁项。广播机制现成，`_safe_call_lifecycle` 的签名过滤保证向后兼容。

**G0b. 离线信号预测力研究**（exp 层纯分析，无 src 改动，最高优先级）
- 数据：step2 `deviate_score_{cfg}.json`（难步回归目标）+ 对应 episode 的观测/动作序列（step1b GT 数据）+ DumpingJudge JSONL（`hit_type`/`cp1_score` → "白搜步"分类目标）。
- 对每个候选廉价信号（步相位、task、连续命中数、库密度分；视觉变化 / gripper / jerk 类步难度信号归 verdict，不作 gate 信号）计算对两个目标的 AUC / precision-recall：
  - 目标一（保命）：`deviate_score ≥ 5`（危险步）；
  - 目标二（省时）：verdict MISS（白搜步，对应 V1 省搜索开销）。
- 产出：信号排行榜 + "可实现 Pareto 预估"（用信号分数做模拟拦截，离线重放出 SR–ratio 曲线的乐观估计）。**它直接回答"gate 到底值不值得做、往哪个信号投入"**，且完全复用现有数据，不跑新 rollout。

### Tier A — 静态先验类（零训练、零状态）

**A1. 步位置门（EpisodePhaseGate）**
- 信号：自数 step + 任务典型长度 → episode 相位；按 step2 的相位风险曲线（中段/末段危险）设置拦截区间。
- 收益：V2 弱——相位只是难步的粗统计。价值主要是作为 C1 的一个特征以及最简单的"结构化拦截"对照（比 periodic 多一点先验的版本）。
- 失败模式：episode 长度方差大时相位错位；单独使用预计只略优于 periodic。
- 落地：极易（PeriodicGate 同级）。**不建议单独立项，并入 C1 特征集。**

**A2. 任务先验门（TaskPriorGate）**
- 信号：task 身份 → 离线统计的 per-task 风险表（step2 已给出：task_8 合并 ≥5 占比 12.9% vs task_2 3.5%）。task 身份来源：G0a 传入 task_key，或零 src 方案——对 `cached_data` 的 prompt_emb 段 mean-pool 后与离线任务签名表做最近邻（prompt 每 episode 不变，第一步算一次即可）。
- 行为：高危任务整体调高拦截率/调低 verdict 交给量（如 task_8 用 periodic k2_n2 档，低危任务用 k10_n1 档）。
- 收益：V2 中等——把预算从简单任务挪给困难任务，是 Pareto 曲线的免费改进（RPG 数据可以直接离线验证这一点：按 task 分层重组预算）。
- 失败模式：任务内部难度不均，粗粒度先验误伤；新任务无先验（回退默认档）。
- 落地：容易。且**可先离线验证**：用 RPG 现有 per-episode 结果按 task 重新分配 periodic 档位，看合成 Pareto 是否左移——零新实验。

**A3. 库覆盖/密度门（LibraryCoverageGate）**
- 信号："库根本给不出好结果"的直接预测版。离线对冻结库拟合密度摘要（k-means centroids / 每 (task, 相位桶) 的历史 top-1 分数分位表），在线用当前 pooled key 对 centroids 算距离（O(k·d)，微秒），低密度区 → skip。
- 收益：V1+V2 双收——低密度区搜了大概率 MISS（省搜索）或勉强命中（危险）。
- 失败模式：密度签名与"能否安全复用"并不等价（密度高的敏感步仍危险——所以 A3 管不了接触敏感期，必须与 B 类互补）；库更新需重算摘要（当前冻结库模式下可接受）。
- 落地：中等。注意 C1 红线——一切库信息离线出，gate 在线不碰 storage。

### Tier B — 时序动态类（零训练、有状态；gate 的独门领地）

这一族用的是 verdict 难以正当使用的信号维度：**episode 内的时间动态与"缓存机制自身的使用历史"**。verdict 每步面对候选打分；"该不该暂时禁用缓存机制"天然是 gate 的定义域。

> **已剔除 B1 视觉变化门 / B2 动作动力学门（owner 决定，2026-07-03）**：这两者用帧间视觉变化 / gripper 翻转 / jerk 去判"这一步难不难"，而"步的物理难度 → 该不该复用"正是 verdict 的活——composite verdict 的 17 因子已含 action/state 通道的 kinematic 描述子，且它在**搜索后**拿着候选判得更准。搜索前用同类信号再判一遍，对 V2 无独立于 verdict 的增量（仅省一次 search），思想与 verdict 重合。gate 真正正交于 verdict 的信号是"缓存机制自身的使用历史"(B3)、"卡死"(B4) 等**元信号**，而非重新推导步难度。故 Tier B 仅保留 B3 / B4。

**B3. 复用债务门（ReuseDebtGate）— 本文最推荐的机制创新**
- 信号：自上次完整推理以来的连续缓存命中步数 + （可选）累计视觉漂移。超过债务预算 → 强制连拦 n 步"清账"，然后重置。
- 机制依据：这是 periodic 的**事件驱动自适应版**。periodic 显著优于 random 的实验事实说明"定期用真实推理清除积累误差"是有效机制；ReuseDebt 把"定期"改成"只在确实连续复用后"，把预算精确花在风险积累处。FULL_HIT 回放期间轨迹被拖着贴近库轨迹，观测相似度虚高（verdict 被骗的机制之一），此时只有"复用了多久"这个 meta 信号是诚实的——**verdict 原理上给不出这个判断，正交性最强**。
- 依赖：G0a 的 verdict 结果回传（gate 需要知道上一步是 FULL_HIT 还是真实推理）。零 src 过渡方案：客户端知道 `__hit_meta__.hit_type`，可在 exp 层用 ClientControlledGate 先原型。
- 参数：债务上限 k、清账长度 n——与 periodic 的 (k,n) 同构，可直接借用 RPG 的网格与评估管线。
- 失败模式：退化边界——若 verdict 命中率本来就低（频繁 MISS 走推理），债务从不积累，gate 无操作（无害）；若命中率极高，行为趋近 periodic（不劣于已知基线）。**下界安全**。
- 落地：低-中。

**B4. 停滞检测门（StuckDetectorGate）**
- 信号：视觉帧向量 + robot_state 连续 N 步几乎不变（机器人卡住/反复无效尝试）→ 强制推理试图脱困。
- 依据：step2 极端 episode（如 task_8/episode_13 的 cycle 20-28 连续大偏离块）呈现卡死形态。
- 收益：V2 防灾——救不回的少数极端 episode。失败模式：合法静止段（等待/精细对齐）误报；LIBERO 中较少见但需白名单机制。
- 落地：容易，gate 自维护一个轻量视觉 + robot_state 帧 buffer 即可（`CP1TemporalPruneKeyBuilder._temporal_prune`（key_builder.py:618-653）已证明该 buffer 族的工程可行性；其 buffer 是 builder 私有，gate 需自建私有实例）。

### Tier C — 标定组合类（轻量学习）

**C1. 双目标标定门（CalibratedComboGate）**
- 结构：A/B 全部廉价信号 → 两个独立的轻量打分头（查表或逻辑回归，在线微秒级）：
  - `P(危险命中 | 信号)` → 保命拦截（V2，阈值保守）；
  - `P(MISS | 信号)` → 白搜拦截（V1：省 search+judge+fetch；库越大 / 栈越 stock 收益越高）。
  两头 OR 合成决策；触发后按 C4 连拦 n 步。
- 数据：G0b 的同一套离线管线直接升级为训练管线（C5 红线：always_search 采集）。可用 conformal / 分位数校准把拦截率钉在目标预算上（把 gate 变成可精确设定 inference_ratio 的旋钮——V3 的完全体）。
- 失败模式：标定域漂移（换 task suite / 换库 / 换 keybuilder 需重标定——与 verdict 阈值同款问题，运维上可接受）；特征间相关导致的过自信（conformal 缓解）。
- 落地：中等。**这是 Tier A/B 信号的自然汇聚点，建议作为第二阶段目标。**

### Tier D — 学习型（小模型）

**D1. 难度预测头（MLPDifficultyGate）**
- 结构：pooled 视觉嵌入 + state + 时序特征（十几到几百维）→ 2-3 层 MLP → 难度分。GPU <1 ms（注意 C2：与决策同步点的交互要设计好，必要时决策延迟一步）。
- 数据：已有 500 ep×3 cfg 的 step2 偏差标签 + twa 主跑 17,100 ep 的步级数据在产——数据量对小 MLP 足够。
- 收益：V2 上限最高（能吃嵌入里查表吃不到的细粒度模式）。
- 失败模式：过拟合 LIBERO / 特定库；训练-部署管线重；可解释性差导致调试困难；分布偏移（C5）。
- 落地：难度最高。**仅当 G0b 显示廉价信号 AUC 明显不足、且 C1 的 Pareto 距 oracle 仍有大缺口时才立项。**

### 双模型对照：Opus 独立提案的映射与补充

本报告由 Fable 产出前，Opus 在仅有系统现状（未见本报告）的条件下独立给过一版 gate 方案（Tier 0–3）。两版高度收敛，收敛项可互为交叉验证；下表为映射，其后是 Opus 版里 Fable 未晶体化的补充。

| Opus 提案 | 对应本报告方案 | 关系 |
|---|---|---|
| 连续-miss 退避（hit-streak backoff）：连命中后按预算强制推理清账 | **B3 复用债务门** | 机制几乎同构（B3 更完整：加了累计漂移 + 债务/清账参数与 periodic 网格对齐） |
| step 相位门 | **A1 步位置门** | 同 |
| 视觉动态门（帧间变化=难） | ~~B1 视觉变化门~~（已剔除） | 两模型都提了视觉/动力学难度信号，但已判定与 verdict kinematic 因子重合而剔除（见 Tier B 剔除说明） |
| 到库 novelty（聚类中心距离）门 | **A3 库覆盖门** | 同（A3 密度摘要更细：per (task, 相位桶) 分位表） |
| 学习分类器：廉价特征 → 「是否有用命中」 | **C1 标定组合门 / D1 学习门** | 同方向；C1 用双头 + conformal 更具体 |
| 成本感知期望值门 | **V3 / C1 conformal** | Opus 给了显式判据（见下补充 1） |

**Opus 补充 1 — 显式期望值判据（统一 V1+V2 的决策上层）**：把"搜不搜"写成一次期望值比较——

```
搜 iff  P(命中|信号)·省下的算力  −  P(坏命中|信号)·SR代价  >  搜索成本
```

它把 V1（省延迟：`省下的算力` 项、`搜索成本` 项）与 V2（保成功率：`P(坏命中)·SR代价` 项）折进同一个标量决策，而非两条独立拦截规则的 OR。这给 C1 的"双头打分"一个原则化落点：两头分数（`P(危险命中)`、`P(MISS)`）连同各自代价 / 收益权重进同一个 EV，用一个阈值输出决策；conformal 校准把该阈值映成可精确设定的 inference_ratio（V3 完全体）。落地上仍建议先做机制驱动的 B3——EV 判据需要 `P` 的可靠估计，依赖 G0b 的信号预测力结论。

**Opus 补充 2 — 低风险论断**：Opus 与本报告 §2.2 主论断一致（gate skip = 完整推理 = 不低于 no-cache）；本报告补的三个边界条件（写路径侧效应 / gate 自身成本 / 道德风险）更完整，以本报告版为准。

### 接口演化备忘（实现阶段再定）

- 拦截块（C4）用 gate 内部状态实现即可，`bool` 返回值协议不用动；若未来想让 gate 输出"强制推理 n 步 + 建议 verdict 收紧"，再考虑扩展返回类型（涉及 orchestrator 协议，L2+）。
- 组合模式：多个廉价子 gate 以 OR 织入一个 `CompositeGate` 容器（与 composite judge 的 composer 思想同构，但先不上框架——两三个子 gate 时硬编码即可，避免过度设计）。
- per-checkpoint：gates 本就是 `dict[CheckpointID, GateFunction]`，CP1 先行，CP3 语义（预测式命中）另议。

---

## 5. 评估框架（全部复用现成资产）

1. **坐标系**：SR vs inference_ratio（RPG 口径），叠加 FULL_HIT 率（C6）。
2. **必须同图的四条参照**：random 曲线、periodic 曲线（**及格线**）、step3 oracle 点（**上限参照，注意附录 B 口径**）、纯缓存/纯推理端点。绘图管线 `analyze_gate_sweep.py` 现成，新 gate 结果按其 JSONL 约定落盘即可直接 overlay。
3. **分层报告**：按 task 分层（task_8 等高危任务单列）+ 按 episode 相位分层——防止"平均好看、高危更糟"。
4. **快速通道**：先用 G0b 的离线重放模拟（信号分数→模拟拦截→离线 SR 估计）粗筛，过线的方案再上 500 ep 真 rollout（评估成本与 RPG 每点相同）。

---

## 6. 优先级排序与建议路线

| 优先级 | 项目 | 理由 |
|---|---|---|
| P0 | **G0b 离线信号预测力研究** | 零风险、全复用现有数据、直接决定"gate 值不值得做+往哪做"；也是 C1/D1 的数据管线雏形 |
| P0.5 | G0a hook 补丁（随下一个 src 改动窗口捎带） | 两个小 hook 解锁 A2/B3 的服务器化；exp 层原型不等它 |
| P1 | **B3 复用债务门**（exp 层 ClientControlledGate 原型） | 机制正交性最强（verdict 看不到"复用了多久"）；零训练；RPG (k,n) 网格与管线直接复用；下界安全 |
| P2 | A2 任务先验（先做零成本离线验证：RPG 数据按 task 重分配预算） | 免费的 Pareto 改进检验；结论并入 C1 |
| P3 | C1 标定组合门 | A/B 信号的汇聚点；conformal 版本给出可控预算旋钮（V3 完全体） |
| P4 | A3 库覆盖门 | 直接预测"库给不出好结果"，V1+V2 双收；库规模增长时优先级上调 |
| P5 | D1 学习门 | 仅当 P1-P3 距 oracle 仍有大缺口时立项 |
| 随手 | 文档修正：cache_system.md §5.1 / tutorial §3.1 的 gate/build 顺序 | L0，随下次文档 commit |

**为什么先 B3 而不是直接上 C1**：C1 需要标定管线成熟；B3 是"机制驱动"而非"标定驱动"——参数空间小（k,n 两参）、有 periodic 的同构网格可借、失败模式下界安全，最快能产出"智能 gate 是否真能打败 periodic"的判决性证据。若 B3 都打不过 periodic，说明搜索前信号的时序维度没有增量，D1 之外的路都可以停。

---

## 7. 开放问题

1. **成功 episode 的偏差分布缺失**：step2 只覆盖 failure subset，oracle 换算的全体口径依赖"成功 episode 少触发"的假设。若要收紧上限估计，需对成功 episode 补一轮 deviate_score（成本：~340 ep × 3 cfg 的 GT 对照回放）。
2. **warm start 交互**：gate 拦截同样保护 WARM_START 步（从缓存 x_t 续跑也有偏差风险），但 warm 步的风险结构与 FULL_HIT 不同（跑了 S2，只省部分 S3），B3 的债务计数是否应给 warm 步打折——留给实现阶段。
3. **跨 suite 泛化**：所有先验/标定基于 libero（spatial/10）。gate 参数是否要做成 per-suite 配置（现有 yaml 体系天然支持，但标定数据要分 suite 采）。
4. **gate 与 verdict 的联合调优**：C6 要求独立标定，但最终部署点是二者的联合 Pareto——是否需要一轮联合网格（gate 预算 × verdict 阈值）留给 C1 之后。

---

## 附录 A：延迟账本口径

- 数据源：`exp/cache_latency_bench/analysis/latency_breakdown.md`（CPU 回放、库 ~2640 条、中位数）。gate 可省部分 = search + judge + fetch；collect + build 无条件执行。参考量级：closed-loop 单请求推理 ~1.7 s（scale-out 闭环测量，含网络与仿真回路；纯 GPU S2+S3 为百 ms 量级）——单请求视角下 gate 省量占比小，但 serving 吞吐视角按并发放大。
- brute-force search 复杂度随库条数线性；50k 条外推 ~67 ms（优化后栈），远程 backend 另加网络往返——大库 / 远程场景 V1 省量显著上升。

## 附录 B：oracle vs 盲拦截换算

- step3 在 per-cfg failure subset（159/154/150 ep）上跑；RPG 在全体 500 ep 上跑。
- y 轴换算（`analyze_gate_sweep.py::load_step3_overlay` 口径）：`full_SR = baseline + (1 - baseline) × subset_SR`，baseline 为该 cfg 纯缓存全体 SR（0.674 / 0.692 / 0.696，`cache_eval_results.json`）。
- x 轴：官方 overlay 不换算（保 subset 轴）——保守口径。全体口径下 `full_ratio = (n_success × r_success + n_fail × r_fail) / 500`，r_success 未测：取 r_success=0 得下界（clip tau3n2 → 0.096），r_success=r_fail 得上界（0.30）。正文引用的 "+8~12 pp / 省 ~40%" 基于中间估计 r_success ≈ 0.5 × r_fail。
- 示例点（clip_w7_d4）：oracle tau3n2 subset (0.301, SR_subset 0.774) → full SR 0.926 @ full ratio 0.10–0.30；对照 periodic k5_n3 (0.340, 0.920)、k10_n5 (0.270, 0.804)、random p0.3 (0.300, 0.815)。
- n=1 全线拉胯（clip tau3: n1 0.61 vs n2 0.77 subset SR）→ 约束 C4 的数据来源。
