# X15 代理监督风险门控 Router（risk_router）— Plan（G1 APPROVED）

> **状态：G1 APPROVED（Round 4，2026-08-22 11:36 CDT）**。六项 owner 裁决已记录（2026-08-22「按你的建议」）。四轮评审共 21 条 blocking/建议全部处理（历史见 git；Review Log 按 §3.1 Post-G1 polish 删除）。**下一阶段：§4 Code，按 §11 触点矩阵 U0→U6 逐单元实现并过 G2。**
> 起草 2026-08-22。整体定级 **L3**（跨 Orchestrator/Interceptor/dump 生命周期 + 新采集协议 + 统计协议）。
> 前作：X14 在线 RL 基线（负结果，结论不受本线影响，`mlp_router` 及其屏蔽测试冻结）。本 plan 不启动任何计算。

---

## 0-pre. 六项 OWNER 裁决 — **已全数裁定（2026-08-22，按执行者建议）**

| # | 决策 | **裁定**（=执行者建议，owner「按你的建议」） |
|---|---|---|
| ① | f12 = t/T_max 是否算 A 档（Markov 状态）内 | **算**：步号是状态的一部分，T_max=520 是冻结常量在线可得；不涉及轨迹公式 |
| ② | B 档（轨迹特征）做不做 ablation | 做，且**不进本篇正文**，标注归 Markov 继承篇 |
| ③ | headline 口径 | **iso-SR@0.80 的 teacher 份额节省**为 headline（C3）；τ\* 定点规则=「满足 SR≥0.80 的最小份额」 |
| ④ | **叙事冲突**：outline Q&A Q1 写死「监督学习语义上不可用/唯一语义正确路线是在线 RL」 | X15 定位为**代理标签监督**：Q1 的三层论证针对的是**反事实成功标签**，X15 用「教师动作偏差」代理标签绕开（ThriftyDAgger 谱系），语义上是 myopic 代理而非价值函数——**不推翻 Q1，但 Q1 需修订**为「反事实标签监督不可用；代理标签监督可用但需付 shadow 成本且不承诺最优性」。X14 结论不动。修订 outline 属 owner 文档。**④已裁：按二分修订 Q1**（反事实标签监督不可用 / 代理标签监督可用但付 shadow 成本且不承诺最优性）；修订文本在 X15 报告阶段起草，X14 结论不动 |
| ⑤ | D1 功效兜底：若 pilot 不一致对率显示 A 池 500 配对不足功效，是否允许 3 seeds/init（1,500/臂）并以 init 级 cluster bootstrap 为 primary | 建议允许，作为**预注册的条件分支**（触碰 A 前定死），否则 D1 维持 500 |
| ⑥ | teacher 驻留 d 是否属 A 档 scope | 建议**属**：驻留是执行器策略状态（与 chunk 化同类），不是风险模型输入 |

另：G1 排期与资源窗口（H200 ~14 h 总量）由 owner 排。

## 0. 目标与可检验主张（按 R1-6 收窄）

**工程目标**：teacher(pi0.5)/cache 二臂，SR → 0.80 一带，teacher 份额显著低于盲混合同 SR 的 ~0.73（P0-b 补点后为内插）。

- **C1（唯一 primary）**：risk_router@τ\* 在**近似匹配份额（|Δshare|≤0.02，p̂ B 侧冻结）**下优于 **global 常数策略**——A 池 500 配对（同 init/seed），init 级 cluster bootstrap 为主推断，exact McNemar 并报。**归因不在 C1 内**（见 D2 族 ablation）。
- **C2 族（探索性，Holm 校正）**：D2 vs 手调 threshold judge（**信息量不同，明示非同信息对照**）；**D2b score-only 消融**（风险模型只喂融合 top-1 分）vs threshold judge = 真·同信息对照（学出来的校准 vs 手调阈值）；**D2c 任务分层常数**（逐任务匹配 router 的任务内份额）——分离「任务间预算分配」与「episode 内择时」两种收益来源。
- **C3（headline 数，③已裁）**：iso-SR@0.80 teacher 份额节省 vs 盲曲线；τ\* 定点规则 =「满足 SR≥0.80 的最小份额」。
- matched-total-compute / yoked per-episode-rate 对照**本轮不做**；若 C1 要进 TIER 论文主张需 owner 追加立项（R1-6 尾款，明示）。

## 1. SOTA 落点（不变，摘要）

SAFE（NeurIPS'25：latent→小网络→functional CP）/ Sentinel（时序动作一致性）/ ThriftyDAgger 谱系（**专家动作偏差作监督标签**）/ Conformalized Interactive IL / UCCI（先校准后取阈）/ RT-Cache-VINN（kNN 距离即置信）/ 语义缓存 verifier 化。公共答案：**低维校准特征 + 稠密代理监督 + 校准阈值**，无人用 65k 维原始输入 + episode 级 REINFORCE。

## 2. 与 X14 的关系及资产（按 R1-7/R1-9 更正）

- **叙事**：见 0-pre ④。X15 不与 X14 争「语义正确性」——它验证的是「语义上 myopic 的代理监督，在同预算下能否比语义正确但样本饥饿的在线 RL 多回收 oracle 空隙（X14 实测只回收 ~3%）」。
- **资产更正**：X14 的 shard 已按设计 reclaim，**b0000–b0283 特征不存在**；现存 b0284–b0287 四批（~21.6k 步）只作 U1 管线冒烟。**fp16 dump 无损记录的是量化后的网络输入；逆 μ/σ 只能近似恢复原始 keys**（fp16 量化不可逆）——离线分数的正确性由 §4 的 parity 门保证，不由「可逆」神话保证。
- 盲混合锚点、`paired_mcnemar`（8 测试）、conductor/sweep 工具链、H200 runbook 照常复用。

## 3. 总体设计（不变）＋ 3.5 特征供给链（R1-1 新增，冻结接口）

```
runtime: query keys ─检索→ StepRetrievalFeatures(新) + PayloadView(现有) ─→ x_t(59) ─→ 风险MLP ─→ r_t ≥ τ ? teacher(驻留d) : cache
离线:   shadow-teacher 逐步标签 u_t ─→ 监督拟合 ─→ isotonic ─→ τ 闭环网格
```

### 3.5 特征供给链（数据生产者→传递路径→文件触点，全部 additive）

| 环节 | 设计 | 文件触点 |
|---|---|---|
| **分数生产（在 backend，不在 strategy）** | 逐字段分与融合实际发生在 `InMemoryBackend._search_weighted_score_sum()`（实证 L711/L933 `per_field_scores`）。**生产者 = backend**：搜索入口先清空、结束时留存 `StepRetrievalFeatures`（新 dataclass：`fused_topk: list[(id,score)]` k 槽、`winner_per_field: dict[field,score]`（**冻结定义：融合 winner 的各字段归一化分**）、`field_own_margin: dict[field, top1−top2]`（**各字段自身排名**）、`fused_margin`、`n_results`）；`search()` 返回类型不变 | `backends/in_memory_backend.py`、`storage_types.py` |
| **传递契约（并发安全，Round 3 修）** | ⚠ `BackendPool` 按 fingerprint **跨连接共享同一 backend 实例**，backend 上的可变 `last_*` 槽会被并发连接覆写。冻结为**原子返回**：backend 新增 additive 方法 `search_with_diagnostics(spec) -> (list[SearchResultLite], StepRetrievalFeatures)`（公开 `search()` 保持原签名=薄包装弃诊断）；**每连接的 `CacheStorage` facade 调用它并持有当次快照**，strategy `last_step_features()` 读自己 facade 的快照——共享层零可变诊断状态；Orchestrator 签名门控注入不变，legacy 路径零触碰。测试：双连接 barrier 强制交错，各自 judge 断言拿到本连接的 59 维输入 | `backends/in_memory_backend.py`、`cache_storage.py`、`search_strategy.py`、`orchestrator.py` |
| **防 stale（冻结语义）** | 诊断在**每次 search 入口清空**；空库/异常/`n<k` 时留存的是**当次真实状态**（短列表+`n_results`），绝不返回上一步残留。测试：连续两查第二次空库 ⇒ features 反映空，不是上次的值 | 同上 |
| **邻居内容** | top-k 邻居动作块、`robot_state` 键、源 `(trajectory_id, step_idx)` **全部经现有 `PayloadView.get_entry()`**（实证已具备，撤回 Round 2 的"storage 新 accessor"规划） | `payload_view.py`（零改动或注释） |
| **f4 取消** | `RetrievalSignals` 仅 `DualRetrievalKnnStrategy` 生产，本臂集用 weighted_score_sum ⇒ 恒 None。**A 档删除 f4**（s_pos≈f1 top-1、delta_pos≈f3 margin，信息已覆盖），不为它接双池 | — |
| **fail-fast** | config 校验：`judge.type==risk_router` ⇒ ① `search.top_k ≥ K_feat(=5)`；② **capability 检查：backend/strategy 路径必须是支持 `search_with_diagnostics` 的 in-memory weighted-score-sum**（其它组合加载即错）；运行时 `n_results<k` ⇒ 0 填充 + f13 覆盖度承接（不 fail episode） | `config.py` |
| **parity** | 在线 dump 落 `step_features` 全文 ⇒ U1 离线重算与在线**逐值比对**（§4 门）；legacy 回归：threshold/mlp_router 决策字节不变（签名未声明→未注入） | tests |

## 4. Phase 0 — 判别力与天花板

- **P0-a 管线冒烟**：现存 4 批上开发 U1（流式，见内存红线）。**离线↔在线 parity 门**：P0-b 新数据上，离线重算 vs dump 的在线 `step_features`——融合分 MAE ≤1e-3 且 **top-1 一致率 ≥99.5%、top-5 集合一致率 ≥99%**；不过门 ⇒ U1 改为 dump 增记**预归一化 raw keys（fp16）**再算（对 zscore 打分而言 fp16 相对误差 ~1e-3 可接受），仍不过 ⇒ G0 判死一票。
- **P0-b 在线扫描（~2,400 ep ≈ 2.5 h，init 池 = **gradient 300 专用**、显式排除 B-cal/B-test；G0 与一切特征选择只读此侧，ledger guard 断言；全开 dump 含 step_features）**：threshold judge 份额 {0.25,0.40,0.55,0.70}×400 + 盲补点 {0.55,0.70}×400，同 slot 池。
- **P0-c 判别力分析**（P0-b 数据 ~130k 步）：episode 级 AUROC（份额分箱控混淆）+ 步级分布分离；诚实边界=关联非反事实。
- **⛔ G0（预注册）**：P0-c AUROC<0.60 **且** P0-b 各匹配份额 vs 盲差 <2 SE ⇒ 止损转投 cache 质量线（ws_search）。
- **⚠ ziyang10 内存红线（全 plan）**：32 GiB cgroup 硬墙、OOM 杀全 pod 含 tether agent。U1 流式 ≤1 批/次、RSS 预算 ≤8 GiB、跑前后 `memory.current` 自检、不与 pi05 server（RSS ~19 GiB）并发；「整块载入」写法 review 直接打回。

## 5. Phase A — shadow-teacher 标定采集（R1-3 重写接线；R1-5 池划分）

### 5.1 接线（缝的位置与时序，冻结）

- **缝在 Interceptor**（不在 Orchestrator——后者 verdict 前后都无执行 stage2/3 的能力）：新 `ShadowTeacherRecorder` 钩子，arm yaml `shadow_teacher.enabled`（默认 false=零操作，legacy 路径零触碰）。
- **顺序冻结**：judge verdict → fetch cache payload（**执行动作永远来自这里**）→ *shadow：同一观测跑一次 teacher stage2/3 前向，结果只记录* → 派发 cache 动作。**每决策恰一次 teacher 前向**，计入 shadow 台账不计臂成本。
- **双向标签覆盖**：cache 执行步 = shadow teacher 前向；**teacher 执行步 = 免费**（teacher chunk 就是执行链路的，cache top-1 chunk 补一次 payload fetch）⇒ 两臂状态上都有 `u_t`。
- **落盘与 join（row union schema，R3-5 冻结）**：sidecar `shadow_rows.jsonl`，行 = `{task_uid, attempt, decision_idx, status: "ok"|"error"|"finalize", teacher_chunk?: fp16(仅 ok), error_type?: str(仅 error), terminal?: bool(仅 finalize), wall_ms?: float}`——error 与 abort/finalize 终态行有一等表达；join 键 **(task_uid, attempt, decision_idx)**；**完整性不变量**：每决策恰一行 ok/error，episode 末恰一行 finalize（缺 ⇒ 剔出标签集并计数，>2% ALERT）。
- **RNG 隔离（R2 提出、R3 补完整契约——没有它逐字节 parity 在设计上不成立）**：pi05 `sample_noise` 用**全局 torch RNG**（`torch.normal` 无 generator，实证 pi0_pytorch.py:311），shadow 前向会推进主路随机序列。冻结：`sample_noise` 加 additive 可选 `generator` 参数（默认 None=全局，主路字节不变，触点 `models_pytorch/pi0_pytorch.py`）；shadow 前向一律使用 Recorder 自有 `torch.Generator`——**seed = 跨进程稳定 digest**：`int.from_bytes(sha256(f"{task_uid}|{attempt}|{decision_idx}").digest()[:8], "little")`（**禁用内建 `hash`**：进程随机化毁掉可复算契约）；**Generator 与 stage3 采样张量同 device 创建**（CUDA 路=CUDA generator）；**direct 与 coordinator 两条执行路都接**。测试：shadow 前向前后全局 RNG state bitwise 不变；同 seed on/off 的 env 动作与 journal 逐字节一致（在 RNG 隔离之上才成立）。
- **异常状态机（冻结）**：shadow stage2/3 失败 ⇒ **cache 动作照常派发（fail-open）**，写 `status="error"` 的 shadow 行；episode/task 异常结束 ⇒ finalize 钩子 flush sidecar 并写终态行；缺行/错行 episode 由完整性不变量剔出标签集并计数（>2% ALERT）。**shadow 任何路径都不得阻断或改变主路动作。**
- **测试**：RNG 隔离两条断言；异常 fail-open；abort finalize；单前向计费；join 完整性；重试语义（attempt 进 key）。

### 5.2 标签与池划分（R1-5，全按 `docs/iclr/tier_experiment_designs.md` 冻结章程）

- `u_t = mean_h ‖a^C − a^T‖₂/σ_a`（norm-stats 逐维归一）；`d_t = 1[u_t>δ]`。
- **B 池四方互斥切分（init 级，Round 2 重划——评测与标定彻底分离）**：
  - **gradient ← B-train 非保护 300**；**δ/模型选择 ← B-train 非保护 50**；**B-cal ← B-train 非保护 50**（isotonic + CP 初始化 + τ 网格/τ\* 选择）——三者合计恰为 B-train 非保护 400；
  - **B-test ← 章程 B-val 50**（最干净切片）：**本线内零拟合零调参**，只承载 D2 族与 C3 的独立测量；
  - **库保护 50**：只作库源，不跑任何 rollout；
  - B-train 侧密度偏置按章程披露；**A 池零触碰直到 D1，每臂一次**。
- **init/seed ledger**：`exp/rl_router/data/x15_init_ledger.json` 记录每 phase 的 init id + seed；发射门禁断言 fit 侧与 A 交集为空。
- **采集覆盖 = gradient 300 + δ 50 + B-cal 50 全部 400 init**（B-cal 的 isotonic/CP 拟合需要该切片自己的 u/d 标签——350 init 的旧口径会让 B-cal 断标签）；规模 **2,000 ep**（400 init × 5 seeds，行为混合：50% 冻结 v288 sample / 30% 常数 p∈{0.15,0.3} / 20% DAgger 式补采）≈ 108k 步标签，~2.5 h。
- **磁盘/内存**：抽完即弃（raw shard → 59 维特征+标签 ≈ 25 MB 全量 → 删 raw），峰值 = 在飞 1–2 批。

## 6. Phase B — 风险模型（R1-4 定标量；R1-2 修特征口径）

### 6.1 部署标量 r_t 与训练流水线（冻结，不留到 Code 临场定）

1. 双头 MLP `x_t(59)→128→128→{u头, d头}`；**主损失 = Huber(u, δ_H=1.0) + 0.5·BCE(d)**。
2. **δ 定于「δ 片 50」**：对 episode 结局的 Youden J；模型选择（早停/超参）同片。
3. **部署标量 = isotonic(u_hat)**，isotonic 在 **B-cal** 上拟合 `u_hat → P(d=1)`（单调映射到校准超越概率）；**τ 阈值打在 isotonic 输出上**。d 头只作辅助正则不进部署。
4. **CP 初始化**：split-CP，nonconformity = 成功 episode 步上的 r 分布，τ₀ = 其 (1−α) 分位；**只作 τ 网格中心，不宣称保证**（干预破坏 exchangeability）。
5. **fail-safe**：特征 NaN/缺字段/空库 ⇒ r=+∞ ⇒ teacher，计数并 ALERT。
6. **artifact schema**：`{W,b, feature_schema_sha, dims, δ, isotonic_knots, cp_tau0, seed, git_sha}`；加载校验 schema sha 与运行时特征构建器版本一致（错配拒载）。确定性：固定种子 + 单线程 eval（沿用 mlp_router 的 pin 机制）。

### 6.2 特征表（A 档 **59** 维 = primary；B 档 +9 = ablation）

| # | 特征 | 维 | 来源（§3.5 供给链） |
|---|---|---|---|
| f1 | 融合 top-k 分数（k=5） | 5 | step_features |
| f2 | 逐字段 top-1（v0/v1/rs） | 3 | step_features |
| f3 | 融合 + 逐字段 margin | 4 | step_features |
| f5 | robot_state 差向量（query − top-1 邻居键） | 32 | query_keys ⊖ `PayloadView.get_entry()` |
| f6 | ‖f5‖₂ | 1 | — |
| f7 | 邻居相位 t′_env/T_max、\|t_env−t′_env\|/T_max | 2 | **双时间轴各自换算到物理环境步**（`exp/markov_sufficiency/_timeaxis.py` 明文两轴不可直比）：query 侧 `t_env = decision_idx × replan_steps`；**库侧 `CacheEntry.step_idx` 是推理周期 0,1,2…**（`_build_entry_chain` enumerate `record.steps`，实证 orchestrator.py:810），故 `t′_env = step_idx × library_replan_steps`。`library_replan_steps` 来源=库 artifact meta，缺失 ⇒ risk_router config 必填，仍缺 ⇒ **加载 fail-fast**；sanity：`max(step_idx)×library_replan_steps ≤ 1.2×T_max`。测试覆盖 query/library 双轴与两侧不同 replan |
| f8 | top-k 动作块方差 | 1 | `PayloadView.get()` |
| f9 | top-k 同源率 | 1 | 邻居元数据 |
| f11 | 任务 embedding（10 任务） | 8 | task id |
| f13 | 检索覆盖 n_results/k | 1 | StepRetrievalFeatures（空库/短列表的显式承接） |
| f12 | t_env/T_max | 1 | 同 f7 单位（`decision_idx × replan_steps / 520`）；**①已裁：属 A 档**。测试：非默认 replan_steps 与末周期截断 |
| — | **A 档合计** | **59** | |
| B 档 | f10 检索 chunk vs 上一执行 chunk 重叠一致性（**从 A 档移入：依赖上一步=history**）；分数 EMA/斜率；距上次 teacher；连续 cache；上一臂；上一 r；累计份额 | +9 | **②已裁：做 ablation，不进正文** |

首步/reset 语义：B 档历史特征在 t=0 置零并带指示位；A 档无历史依赖故无此问题。

## 7. Phase C — 决策规则（微修）

`r_t≥τ`→teacher，**驻留 d∈{1,2}**（**⑥已裁：属执行器策略状态**，非模型输入）；τ 网格 = τ₀ 两侧 4 点 × 200 ep（**B-cal init**），取 SR≥0.80 最小份额为 τ\*；全网格进 frontier。RCPO 在线自适应**不做 primary**（新训练动力学，风险>收益）。

## 8. Phase D — 评测（R1-5/R1-6 重定）

**p̂ 冻结与 C1 估计量（R2-4/R3-4）**：`p̂ :=` risk_router@τ\* 在 **B-cal τ\* 格**（Phase C 的 200 ep）上的实测 teacher 份额，**触碰 A 之前落 ledger**；A 上常数臂 = Bernoulli(p̂)。**匹配容差收紧到 |Δshare| ≤ 0.02**（盲曲线低份额段斜率 ≈0.54 ⇒ 0.02 份额差 ≈0.011 SR 混淆，低于 0.02 检出限；原 0.05 会引入 ~0.027 的 teacher 预算混淆，撑不住 primary）；C1 措辞冻结为「**近似匹配份额（≤0.02）下的策略对比**」。超容差 ⇒ 预注册降级：C1 改报「策略对比＋份额敏感性界」（盲曲线局部斜率 × Δshare 给 SR 差的混淆上界，disclosed secondary），**绝不在看到 A 份额后回改对照**。所有对照臂参数（threshold 阈值、任务分层份额）一律 B 侧冻结后进 B-test/A。

**C3 估计量（预注册，R3-4）**：① router 三 τ 点在 B-cal 上按预测 SR≈{0.75,0.80,0.85} 选定并落 ledger（先于 B-test）；② 曲线 = (share, SR) 相邻点**线性插值**；非单调 ⇒ 三点先 isotonic 回归再插值；③ CI = init 级 cluster bootstrap 每次重采样**重拟合插值**后取 iso-SR 份额的 percentile 区间；④ **bracketing 失败规则**：任一曲线相邻点未夹住 SR=0.80 ⇒ C3 判「**不可估（未 bracket）**」，只报最接近实测点，**禁止外推**。

| 编号 | 比较 | 池/规模（**B-test 各行：50 init × 10 seeds/臂，聚类单位=init；D1 行：A=500 唯一 init**） | 判据 |
|---|---|---|---|
| **D1（唯一 primary）** | risk_router@τ\* vs global 常数@p̂（B 侧冻结） | **A 池 500 配对**（10×50 pruned_init，同 init/seed，每臂一次） | init 级 cluster bootstrap 主推断 + exact McNemar 并报，双侧 α=0.05 |
| D2 | vs threshold judge@p̂（阈值 P0-b 侧标定冻结） | **B-test 500** | Holm 族，init 级 cluster bootstrap |
| D2b | score-only 消融 vs threshold judge（同信息） | B-test 500 | 同上 |
| D2c | 任务分层常数（份额自 B-cal 冻结） | B-test 500 | 同上 |
| D3 | vs 冻结 RL v288@匹配份额 | B-test 500 | 同上 |
| **D4=C3 headline** | **frontier 在 B-test 上独立重测**：router {τ\*, τ±} 3 点 + global 常数 {p̂, 0.40, 0.70} 3 点，各 500 | B-test，共 3,000 ep | iso-SR@0.80 份额节省**只从 B-test 内部两条曲线读出**（Phase C 网格降级为 tuning diagnostics；历史盲混合锚点只作背景不作比较基准，池不同） |

- **功效（预注册决策分支）**：从 τ 网格 episodes 估不一致对率 q̂；若 power(500, q̂, Δ=0.04)<0.8 ⇒ 走 **⑤已裁的预注册分支**（3 seeds/init，cluster bootstrap primary）；分支选择在**触碰 A 之前**落 ledger。
- 严格同 slot、零单边丢弃（`paired_mcnemar` 已有 WARNING 路径升级为 hard fail on D1）、retry 终态去重（现有 `terminal_outcomes`）。

## 9. 预算（微增）

P0 ~3 h / A ~2.5 h / B <0.5 h / C ~1 h / D：A 池 2×500 + **B-test 10 臂×500 = 5,000** ≈ 6,000 ep ~6 h ⇒ **总 ~12,700 ep ≈ 13–14 h H200**。硬约束：ziyang10 32 GiB RAM（§4 红线）。

## 10. 风险与边界（R1 增补后）

1. 代理标签≠反事实成功（双向误差；δ 对结局标定 + 闭环 SR 判决缓解）；
2. 分布漂移（行为混合 + DAgger 轮）；
3. G0 判死可能（判别力不足 ⇒ 修库不修 router，止损即成果）；
4. 论文口径（B 档轨迹特征归继承篇；owner ①②④⑥）；
5. CP 不承诺保证（只作初始化）；
6. 32 GiB RAM 墙（OOM 杀全 pod 含 tether agent）；
7. **fp16 量化误差**可能翻转近邻次序 ⇒ §4 parity 门 + raw-keys fallback；
8. **B-train 密度偏置**（库对 B-train 覆盖稠密 ⇒ 相似度乐观）——按章程披露，δ 在 B-train δ 片、isotonic/CP/τ 全在 **B-cal** 侧缓解（B-test 零拟合）；
9. X14 负结果独立成立；`mlp_router` 及其测试冻结。

## 11. L3 触点矩阵（R1-8 重列；G1 后逐单元过 G2）

| 单元 | 级 | 文件触点 | 测试 | 文档 |
|---|---|---|---|---|
| U1 离线分数管线 | L1 | `analysis/offline_scores.py` | parity 门（在线逐值/top-k 序一致）、流式 RSS 预算（合成 memory 断言）、4 批冒烟 | — |
| **U0 backend 特征侧信道** | **L2** | `backends/in_memory_backend.py`、`cache_storage.py`、`search_strategy.py`、`storage_types.py` | **防 stale**（连续查询第二次空库不得返回残留）、`n<k` 短列表、异常清空、winner_per_field/field_own_margin 定义级单测、**legacy 搜索路径字节回归** | — |
| U2 P0-b yamls + dump 扩 `step_features` | L2 | `emit_router_yamls.py`、`dumping_judge.py` | dump schema roundtrip、legacy dump 不变 | — |
| **U3 shadow-teacher 接线** | **L3** | `interceptor.py`(+Recorder)、**`models_pytorch/pi0_pytorch.py`（sample_noise 加 additive generator 参数）**、`config.py`、shadow sidecar | **RNG 隔离（全局 state 前后 bitwise 不变）**、同 seed on/off 逐字节 env/journal parity、异常 fail-open、abort finalize、单前向计费、join 完整性、重试语义、hot-swap 下 shadow 存活、**direct/coordinator 双路** | `docs/architecture/cache_system.md`（Interceptor 缝）、`docs/cache/tutorial.md` |
| U4 特征构建 + `train_risk_model.py` | L2 | `exp/rl_router/` 新文件 | **四方切分互斥断言（ledger 驱动，含 B-test 零拟合 guard）**、**f7/f12 单位测试（非默认 replan_steps、末周期截断）**、schema sha 校验、确定性重训 bitwise、isotonic/δ 只见许可池的 guard 测试 | — |
| **U5 `risk_router` judge** | **L3** | `mlp_router_judge.py` 旁新文件、`config.py`（`_JUDGE_TYPES` 注册；**不进 `_ROUTING_JUDGE_TYPES`**，R1-10）、`orchestrator.py`（step_features 注入） | 读到 step_features/payload 断言、**A 档不读历史断言**、fail-safe（空库/缺字段→teacher+计数）、top_k fail-fast、**legacy 回归：threshold/mlp_router/composite 决策逐字节不变**、payload 缺失/链 fork 边界 | 同 U3 两处 + `docs/README.md` 若增页 |
| U6 统计驱动 | L2 | `analysis/`（cluster bootstrap + 功效计算） | 合成数据正确性（已知效应恢复）、**重复 init 聚类正确性（50 init×10 seeds 下覆盖率仿真）**、D1 hard-fail on 单边丢弃、**完整池隔离断言（P0/G0/特征选择不读 B-cal/B-test；一切 fit 不读 B-test）**、**p̂/τ± 冻结先于 B-test/A 触碰的时序断言**、**iso-SR 插值/单调化/bracketing 失败规则单测** | — |
| 索引 | — | `logs/README.md`（**已同步**）、结项时 `exp/rl_router/analysis/` 报告 | — | — |
