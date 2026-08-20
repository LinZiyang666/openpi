# TIER：面向机器人策略的经验分层推理 — 论文提纲 v2（评审后修订，中文稿）

> 英文主稿：[`tier_paper_outline.md`](tier_paper_outline.md)。本文为逐节对应的中文稿。
>
> Thesis（一句话）：**经验库对机器人策略 serving 的价值在于它的索引（index），而不是它存的内容（payload）**——对 teacher 自身 rollout 表征的检索相似度是一个免训练的、逐步的路由*信号*；基于它的正确系统形态是三层分派（replay / student / teacher），而不是缓存回放。
>
> v2 已吸收 4 名审稿人（ML 主审 / robotics / 统计 / systems）的对抗评审：32 条 findings，裁决日志见文末。
> 篇幅预算：正文 9 页；正文 float ≤ 6 个 + related work 1 张对比表；appendix ≤ 11 页。
> Scope lock（不变）：检索只用 depth-1 单帧 key；全文无 history/trajectory 项；Markov 继承理论留给第二篇论文；历史增强负结果 = 正文 1 句 + 附录 C 一张表。

---

## §1 引言（约 1.25 页正文；Fig 1）

- **P1**：VLA 逐步推理成本主导 serving 经济学；LLM serving 已把按查询条件路由（cascade/router）做成标准范式；机器人领域只有三类相邻但不同的原语——dual-system VLA 两个模型永远都跑（频率切分）；DeeR-VLA 在单模型内部做深度自适应；interactive IL（SafeDAgger / EnsembleDAgger / ThriftyDAgger）逐步在 policy 与专家间切换，**但目标是省训练期专家查询、用的是学习出来的 gate/ensemble**——不是部署期算力、不是免训练信号。
- **P2**：缺的不是 router 而是路由*信号*：逐步、免训练、闭环下有效的"廉价执行体在此处是否胜任"估计。（"training-free" 全文限定于**信号**层——系统含蒸馏学生与离线阈值标定。）
- **P3**：核心洞察——部署中的 teacher 自己持续生产该信号的原料：它的 rollout。经 success 过滤、以 teacher 自身内部表征编码的经验库索引的是**被证实的胜任**；检索相似度逐步估计廉价执行体是否够用；同一索引在近重复状态还支持直接 replay。
- **P4**：TIER 方法预览（三层执行体共用一个索引；级联 key；SR 约束下的离线标定）。
- **P5**：**三条贡献**（key-space 降为支撑性分析）：
  1. **方法/系统**：TIER——首个跨独立 policy、以免训练*信号*做部署期算力分配、在算力–成功率前沿上评估的逐步路由系统（优先权声明已收窄；interactive IL 谱系已让渡）。
  2. **发现**：受控 factorial 替换研究给出双重解离——payload 可替换（hit 槽换蒸馏学生反而超过最强标定 replay），index 承重（学生在 index 标记的 miss 处崩塌；效应经干净臂 + 交互项分析与 replay 前缀伤害剥离）。修正 retrieval-as-replacement 一线的内容价值假设——**限定在"同一批经验足以蒸馏出学生"的 regime 内**。
  3. **前沿结果**：TIER 在中段算力区间扩张可达的算力–SR 前沿（措辞限定区间，不用全域 dominate），由预注册操作点上的 matched-**compute** 与 yoked matched-rate 双对照确立。
- Hypothesis box：只含前瞻部分（X2 对照、信号身份预测）；X1 作为"已确立的发现、由此提出假设"陈述，不伪装成待检验。
- Fig 1：系统图 + 前沿 teaser。

## §2 相关工作（约 0.75 页正文 + 对比表 float）

五族：
- 机器人检索（VINN、RT-Cache、Behavior Retrieval、DARP）：让渡"库"原语，占据新角色（库=胜任索引而非动作来源）。
- **Interactive IL / 逐步策略切换（SafeDAgger、EnsembleDAgger、ThriftyDAgger）**［评审新增］：让渡"切换"原语；差分在目标函数（部署算力 vs 训练期专家经济）、信号（免训练检索 vs 学习 gate/ensemble）、评价（算力–SR 前沿）。
- 条件算力与路由（FrugalGPT、RouteLLM；DeeR-VLA、MoLe-VLA；dual-system RoboDual/HiRT/GR00T-N1）+ **token 级缓存（VLA-Cache）**［新增］：一句话对比——VLA-Cache 跨帧复用 payload，我们证明 payload 是押错的那一半。
- 失败检测 / UQ（Sentinel/STAC、SAFECAST、VLAConf、ARMADA、kNN-OOD）：信号的最近亲；告警级、路由对象是人；与本方信号的逐项差异在 §6.2 写明（参考集=success 过滤的自身 rollout；key 层；消费方式=分派而非告警）。
- 语义/近似缓存（GPTCache、NIRVANA、**IC-Cache** SOSP'25）：范式源头，开环。IC-Cache 在 LLM serving 域独立记录了 payload 问题（朴素语义缓存回放 win rate 50%→18%），其解法 = ICL 增强 + 逐请求 bandit router（吃用户反馈）——两个机制在闭环控制中都不存在（固定权重 policy 无法 prompt 增强；无免费逐决策反馈）。趋同的动机、互斥的解法空间。

## §3 方法（约 1.5 页正文）

### 3.1 设定
a = g(φ(o))；学生 π_S；库 L 来自 success 过滤的 teacher rollout；depth-1 查询 k(o_t)；两阈值 → 三路分派。success 过滤作为设计选择陈述，**其效应被测量而非断言**（前向指针 X12）。

### 3.2 为什么用 policy 自身的表征（支撑性分析，不占贡献条目）
- 充分统计量视角（`I(a*;o|k)` 残差），一段。
- **Remark 1**（由 Proposition 降级）：Lipschitz 复合直觉一句话 + 附录 sketch；承重证据是经验性的——X7 的 ε→δ 曲线 **+ 从日志里挖出的真实反例对（CLIP 距离近但动作远）**作分离论证。
- 成本诚实：内部 key 每步需部分 teacher 前向；级联 = 信号质量 × 查询成本上的操作点；**级联决断率是被测量的量（X11），savings-ceiling 曲线进 §5.1 正文**。
- **为什么不训一个 router**（预判质问；完整论述见 Q&A Q1，实验 X14）：对逐步闭环路由，监督学习在语义上不可用——标签是反事实的、纯执行体 rollout 的标签在到达分布与后续语义两头错位、且路由目标量 Bellman 耦合到 router 自己的未来决策（按定义是 RL 问题）。唯一语义正确的训练路线是在线 RL——我们真的把它跑成基线（X14：三变体、离线 MC 热启动、慷慨交互预算），用 interaction-efficiency 曲线对比零训练水平线标价。辅助不对称性：fail-closed OOD 方向（新颖→不相似→teacher）vs 学习 router 的不受控外推；乘法维护成本（trained router 绑定 执行体组合×操作点×学生×套件×模型版本；index 由管线 rollout 零标签重建、随 teacher 表征自动同步）；cache 层执行时反正需要检索取 payload（router 只出类别，动作 chunk 在检到的 entry 里）。

### 3.3 同一经验的两种压缩
Replay = 最近邻零阶保持；学生 = 参数化插值。该框架是生成预测的（预测 §6.1 解离与密度行为），**同源 caveat 前置声明**并交由 §6.2 的身份实验（X12）裁决——不当挡箭牌。

### 3.4 标定协议（预注册面）
在专用标定 init 切分上离线进行（init 账本进附录 F：标定 init ∩ 评测 init = ∅，数目写明）；阈值**以及**所有 Table-1/X2 操作点在评测前冻结；replay 基线用**同一**程序标定（对称条款）；确定性 tie-break；无在线自适应。

## §4 实验设置（约 0.4 页正文）

- Teacher = Pi0.5；学生 = ACT + SmolVLA（蒸馏）**+ demo-trained ACT 臂**（X12c）；LIBERO-Spatial + LIBERO-10；X8 第二 benchmark 家族（RoboCasa / SimplerEnv / MimicGen 候选）作**确认集**——全部分析选择在 LIBERO 上冻结后 X8 才跑。
- 角色分工：LIBERO-10 = 主判别战场（teacher 0.868，有区分空间）；Spatial = 天花板区间无损验证，**披露 discordant-pair 计数与统计功效**。
- 算力度量：权重 = 指定硬件 batch=1 实测 per-invocation GPU-time；FLOPs 口径作鲁棒性副本；前沿图一个 panel 直接用裸 GPU-seconds/step（无归一化）。
- 统计：配对 init/seed；McNemar；等价性 TOST **δ=5pp 主判据 / 3pp 描述性报告**（附录 D 给基于 pilot 不一致率的功效分析）；**预先声明的极小 primary 检验族**（每套件 2 操作点 × 2 主对照）走 Holm，其余描述性；全程 episode 级 cluster bootstrap。

## §5 主结果（约 1.2 页正文；Fig 2、Table 1）

### 5.1 前沿 [X4 + X11 + X12 臂]
- SR × 实测算力 Pareto：TIER 扫描 vs 纯 teacher、纯学生、最强标定 replay、**teacher-cheapening 基线**［评审新增］：(a) flow 积分步数 sweep，(b) 拉长 chunk / 降查询率 sweep。
- **Savings-ceiling inset**：可达算力节省作为级联短路率的函数（X11），把 key 成本地板显式画给读者。
- 三区制讨论（replay 仅超廉价端有竞争力——指针到附录 E 密度分析；TIER 中段；teacher 质量顶）。"混合超越 teacher"（student@hit + teacher@miss > teacher 单跑）作为 filtered-distillation 增益公开讨论；分区归因交给 §5.2 对照。
- Table 1：预注册操作点 × {SR, GPU-s/step, 三层占用率, 驻留显存}。

### 5.2 分区 vs 率 vs 算力：对照家族 [X2 — 生死线，已重设计]
- **主对照 A — matched-total-compute random 混合**：把 TIER 的信号开销（key 构建+检索）折算进对照的 teacher 率，总成本持平。
- **主对照 B — yoked 逐-episode 率 random**：对每个配对 init，按 TIER 该 episode 的实现 teacher 率逐步随机分派 → 把 *per-step* 信息从 episode 级预算分配中剥离。
- 次级：global-rate random + 多相位 periodic；含 replay 的操作点做三路匹配混合或明写排除在 primary 之外。
- 协议预注册：名义率/实现率/实现算力与容差入表；claim 措辞：只在预先声明的点上做 Holm 校正的优越性检验；**预写降级叙事**：若仅中段显著，thesis 改述为"分区价值集中于中预算区间"。
- **反转分区臂**（匹配率下的反向路由）作一行式戏剧性检验。

## §6 分析：价值住在哪里？（约 1.5 页正文；Table 2、Fig 3、Fig 4）

### 6.1 双重解离：factorial + 因果解剖 [X1 扩展 + X5 重设计；Table 2]
- 完整 2×2 factorial {hit: replay|student} × {miss: teacher|student} **+ (hit→teacher, miss→student) 干净臂**；配对 episode 上报两主效应 + 交互——禁止只报对角差。头条分解：崩塌中 miss 槽 / replay 前缀 / 交互各占多少。
- 因果解剖并入（原 §6.6）：**state-reset 闭环探针**（仿真器置位到 teacher rollout 采出的 miss/hit 态，按任务进度配平，学生闭环滚到底）+ **接管剂量-响应**（miss-onset 处学生接管 k 步后 teacher 接回，恢复率 vs k）；DAgger 式 teacher relabel 沿学生轨迹量化访问分布混杂；开环探针 → 附录 E 并注明多模态 caveat。

### 6.2 索引到底在测什么？信号身份 + 胜任估计 [X3 + X12；Fig 3]
- 预测学生失败的 AUROC + 校准 vs 免训练基线（kNN-OOD 特征距离、STAC 式动作一致性、likelihood/entropy）+ ensemble 参照 + **trained-router 基线**（同一内部特征上的小 probe，用 held-out 标注学生 rollout 训练；报 **label-efficiency 曲线**——AUROC 随标注 episode 数变化，对比零标签 index 信号的水平线；预注册立场：零标签追平标签饥渴 router 即为胜，若 router 大标签量下反超则交点本身入结论——部署场景在标签稀缺侧）；标签构造显式化；lead-time 分层（失败前 ≥k 步；事故后步剔除/单列）；episode 级 cluster bootstrap；全部基线同标签。
- **身份消融** [X12]：(a) 库 success-filter {开、关、仅失败} 配平密度 → 胜任 vs 到访之辨、与 kNN-OOD 的精确差分；(b) **解耦批次**——学生蒸馏于 rollout 批 A、库建于不相交批 B；(c) **demo-trained ACT**——index 能否路由一个与它零数据共享的学生；(d) teacher 自身失败预测对照——通用难度 vs 学生特异胜任；各分支结论按结果预先限定。

### 6.3 key 空间 [X6 + X7 合并；Fig 4 单 float]
- 按 key 来源（内部 vision-token / LLM-layer vs CLIP / DINOv2 vs 本体感知）的信号 AUROC，**附每变体查询成本列**；各 key 的端到端前沿 → 附录 E。
- ε→δ 曲线用**分层状态对采样**（同 episode 近/远、跨 episode 同 init、跨 init；headline 只取跨 episode 层）与动作分布距离（energy distance，采样协议写明）；展示真实 CLIP-近/动作-远反例对。

## §7 局限与讨论（约 0.3 页）

仅仿真；key-space 主张的单 teacher (n=1) 范围；同任务族部署 regime；信号成本地板（级联缓解、X11 量化）；teacher 更新时的学生维护成本（training-free 只属于信号不属于系统）；replay 层开环机制（报段长分布、中途相似度复查/abort、段长上界）；serving 级吞吐/尾延迟留给系统后续工作（本文主张挂在 per-step compute；若 X13 落地则附录 B 有 mini-bench）；历史增强单句披露。

## §8 结论（约 0.1 页）

---

## Float 台账（正文；最多 6 + §2 一表）

Fig 1 系统+teaser · Fig 2 前沿（双 panel：normalized + 裸 GPU-s）含 ceiling inset · Table 1 操作点（SR/成本/占用率/显存）· Table 2 factorial 解离 · Fig 3 信号 AUROC+校准 · Fig 4 key 空间（AUROC×成本 + ε→δ）· §2 对比表。

## Appendix 计划（≤ 11 页；无正文指针者删）

- A（0.5）：Remark 1 sketch + 假设讨论。
- B（2.5–3）：算力记账——stage 拆分 microbench（FLOPs + wall-clock，batch=1）；每 key 变体构建成本；级联决断率与摊销（X11）；权重表 + 双口径鲁棒性；normalized-vs-实测**对账**逐层归因；server-side vs end-to-end 延迟拆分；驻留显存表；标定 episode 预算；[可选 X13 serving mini-bench：N 并发 steps/s/GPU + 延迟 CDF]。
- C（0.5）：历史增强负结果（一表 + 3 句）。
- D（2）：ε→δ 全图；标定网格；**功效分析**（pilot 不一致率；TOST 可行性）；预注册记录（操作点、检验族）。
- E（2）：分套件/分任务表；SmolVLA 臂；key 来源端到端前沿；replay 密度分析；开环探针。
- F（1.5）：实现；库构建；蒸馏配方；**init 账本**（标定/评测/训练不相交证明）；可复现性。

---

## 实验台账 v2（详细设计卡见 [`tier_experiment_designs.md`](tier_experiment_designs.md)）

| ID | 内容 | 章节 | 状态 | 优先级 |
|----|------|------|------|--------|
| X1 | factorial 替换矩阵 + 干净臂 (T@hit,S@miss) + 反转分区臂 | §6.1 | 部分已有——**新增 2 类臂** | **P0** |
| X2 | 对照家族：matched-**compute** random（主 A）+ yoked 逐-episode 率（主 B）+ global-rate/periodic（次级）；操作点预注册 | §5.2 | 新 | **P0（生死线）** |
| X3 | 信号 AUROC/校准 vs 基线；lead-time 分层；聚类推断；teacher-failure 对照 | §6.2 | 新 | **P0** |
| X4 | 前沿扫描 + teacher-cheapening 基线（积分步数与 chunk 长度 sweep）+ 裸 GPU-s panel | §5.1 | 部分已设计——**新增 2 条基线 sweep** | **P0** |
| X5 | 因果解剖：state-reset 探针 + 接管剂量-响应 + DAgger relabel | §6.1 | 新（重设计） | **P0**（原 P1） |
| X11 | 级联决断率、摊销 key 成本、级联关闭前沿 | §5.1/附录 B | 新 | **P0** |
| X12 | 信号身份：filter {开/关/仅失败}、解耦批次、demo-trained ACT（含 X3d） | §6.2 | 新（a,d 离线廉价；b,c 需训练） | **P0(a,d) / P1(b,c)** |
| X6 | key 来源消融 + 成本列 | §6.3 | 新（基建现成） | P1 |
| X7 | ε→δ 分层曲线 + 反例挖掘 | §6.3 | 新，离线 | P1 |
| X8 | 第二 benchmark 家族（确认集；分析先冻结） | §4 | 新 | **P1（升级）** |
| X9 | replay 误差 vs 库密度 | 附录 E | 新，离线 | P2 |
| X10 | 历史负结果表 | 附录 C | 数据已有 | P2 |
| X13 | serving mini-bench（吞吐/GPU、延迟 CDF） | 附录 B | 可选 | P2 |
| X14 | 在线 RL router 对决：R_ts/R_tc/R_tsc 环内 RL router（只看进库前特征、屏蔽检索分；离线 MC 热启动；batch on-policy + bundle 热切换），interaction-efficiency 曲线 + 冻结权重 A 池对决 | §6.2/Q&A | 新 | **P0** |

---

## Q&A — 预判的审稿问题（rebuttal 弹药库）

**Q1. “你们的贡献本质上就是一个 router（用 index 比对的 router）——为什么不干脆训一个小模型当 router？”**（导师 2026-08-15 提出；正文计划在 §3.2，实证回应在 X14）

1. **监督学习对这个问题不可用——三层结构性论证。** (a) 标签“执行体 E 从状态 s 接管会不会成功”是反事实的：除非真让 E 从 s 跑，自然数据里不发生。(b) 纯执行体 rollout 的标签也错位：把 episode 结局广播给沿途状态，标注的是“E 从自己走到的状态、独跑到底”的成功率——部署时 router 把 E 派到混合前缀到达的状态、且接管后还可能被切走，到达分布与后续语义双重错位。(c) 最深一层：在 s 选 E 的价值取决于 router 在后续状态的决策——标签定义 Bellman 耦合到待学策略本身。“用监督学习训 router”在语义上不成立；可行路线只有：偏置离线 MC、sim 特权 state-reset 标注（一条标签一次 rollout，真机无此机械）、迭代交互式、在线 RL。contextual bandit 路线（IC-Cache, SOSP'25）的前提是免费逐决策反馈——LLM serving 域特权，闭环控制没有（决策逐步、结局是一个延迟 episode 级 bit、每条样本一次物理 rollout）；IC-Cache 自己都在标签最便宜的域以标签成本否了 classifier router。
2. **唯一语义正确的路线是在线 RL——所以基线就真跑它（X14）。** Router = 环内 RL 策略（reward = 成功 − λ·成本），sim 内合法（非 eval）init 池上 batch on-policy 训练，离线偏置 MC 头热启动（对“你把 RL 基线训弱了”的疫苗），三变体对应我们三种模式（{T,S}/{T,C}/{T,S,C}），输入限定为与我们 key 同源的进库前模型内部特征——检索分数与库侧信息一律屏蔽。用 **interaction-efficiency 曲线**标价：冻结策略部署 SR 随累计训练 episode 变化，对比零训练水平线。乘法成本是结构性的：trained router 绑定（执行体组合×λ×学生×套件×模型版本），任一变动即重训；真机上其训练过程 = 未训好的 router 在机队上 explore。
3. **预注册结局——每支可发表。** Router 追不上：免训练信号直接胜。Router 花 N 交互 ep 追平：N 就是价签，且每次配置变动重付一遍，而 index 对这些全部免疫。热启动+慷慨预算下仍不收敛：如实报告——200 步 1-bit 稀疏奖励是结构性原因。不随 AUROC/SR 变的辅助不对称性：fail-closed OOD 方向；标签/交互经济学的 benchmark 硬上限（合法 episode ~500、九成在学生训练分布内）；cache 层执行时取 payload 在任何臂里都需要检索。所引论文赠品：IC-Cache 实测朴素回放 win rate 50%→18%——payload 主张的 LLM 域独立证据；其解法（ICL prepending）在固定权重 policy 上不存在。趋同动机、互斥解法空间。

## 裁决日志（32 findings → 簇；执行方裁决）

- **A. −32.8pp 归因混淆**（Rob-1、AC-6、Stats-1）：**全部接受**——factorial 分析、干净臂、反转分区臂、X5 重设计（state-reset + 剂量-响应 + relabel）、X5→P0、开环探针降附录。
- **B. 库-学生同源 / kNN-OOD 同一性**（Rob-2、AC-2、Stats-2）：**全部接受**——X12 身份套件（filter 消融、解耦批次、demo-trained ACT、teacher-failure 对照）；§3.3 caveat 前置；结论按结果限定。
- **C. X2 对照升级**（Sys-2、Stats-3、AC-1）：**全部接受**——matched-compute 主对照、yoked 逐-episode 率主对照、协议预注册、操作点预先声明、claim 改写、预写降级叙事。
- **D. 成本记账脊柱**（Sys-1,3,4,5,7,8；Rob-5）：**全部接受**——新增 X11；savings-ceiling 进正文；实测 GPU-time 权重 + 裸 GPU-s panel；对账；显存列；附录 B 扩到 2.5–3 页。
- **E. 主张/措辞**（AC-3,4,5；Rob-7,8；Sys-7b、AC-8）：**全部接受**——"first" 收窄 + interactive IL 谱系让渡；"dominates"→"区间内扩张前沿"；Prop→Remark + 真实反例；贡献 4→3；hypothesis box 仅前瞻；"training-free signal"；"corrects"→"revises … in the regime where…"。
- **F. Benchmark/基线**（Rob-3,4,6）：**接受**——X8→P1 确认集；l10 主战场、Spatial 披露功效；teacher-cheapening 基线；replay 基线对称条款；replay 开环机制披露。*部分*：Spatial 仍留在主前沿图（不逐出主结果）。
- **G. 统计严谨性**（Stats-4,5,6,8）：**接受，一处修改**——预注册面进 §3.4/§4/附录 D；δ=5pp 主判据；极小 primary 族；AUROC 标签/聚类/lead-time 协议；ε→δ 分层。**修改/驳回原样采纳**：Stats-4d 的跨套件"Spatial 探索 / l10 冻结确认"协议——l10 的 X1 数据已存在，冻结是事后虚构；确认集角色改由 X8（未触碰的家族）承担。
- **H. 篇幅预算**（AC-7、Sys-8、owner 指令）：**接受**——§6 由 6 小节收编为 3；§6.5→附录 E、§6.6 并入 §6.1；§2 表格化压缩；float 台账封顶 6+1；appendix ≤11 页逐节限额。
- **暂缓**：Sys-6 完整 serving bench → 可选 X13（主张按其方案 (b) 收缩至 per-step compute）。
