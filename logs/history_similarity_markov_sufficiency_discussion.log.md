# History-Frame Similarity 无增量问题 —— 方法学讨论纪要（Markov 充分性假说）

> **状态**: Discussion Record（会议纪要）| **级别**: L0（纯讨论，零代码改动）| **日期**: 2026-08-12
> **参与**: Ziyang（owner，提出核心假说）+ Claude agent（Execution authority，证据调研与论证展开）
> **议题**: trajectory search（把 history frame 的 similarity 纳入当前 frame 的 cache 检索）在 libero_spatial 与 libero_10 上、于最强 z-score-tanh `weighted_score_sum` 配置下基本无增量，d1（只用当前帧）几乎总是最优——成因讨论。

---

## 1. 议题与背景

跨 5 组实验、两 suite 合计约 14 万 episodes 的一致结论：在强检索配置上，`trajectory_depth > 1` 平均为净负收益。团队此前已排除两个候选解释（见 §2.3），目前存活解释为「trajectory 是时间平滑算子」（H-mechanism，`logs/archive/weighted_sum_trajectory_weight_research.log.md`）。本次讨论提出并论证一个更根本的假说：**LIBERO spatial / libero_10 在这条 pipeline 下基本上是 Markov 充分的**，H-mechanism 是它的操作层推论而非竞争假说。

## 2. 证据基线（本次调研汇总）

### 2.1 机制要点

检索打分为三层嵌套加权和（`src/openpi/cache/components/score_normalizers.py:174-178`、`backends/in_memory_backend.py:600-641, 869-896`）：

```
TrajScore(x) = Σ_l w_l · Σ_f w_f · ½(tanh((s_f(anc_l(x), q_{t-l}) − μ_f)/σ_f) + 1)
```

- z-score tanh 是 Layer-1 归一化器（非独立策略）；策略为 `weighted_score_sum_knn`。
- History 机制：query 侧 `TrajectoryMixin` 维护最近 d 帧 key 缓冲；library 侧沿候选 entry 的 `prev_ids` 祖先链回走，第 l 层祖先对齐第 t−l 帧 query 打分，再按 `trajectory_weights`（newest-first）加权。
- 讨论相关的实现细节：①祖先链走法假设 query/library 轨迹 1:1 逐步对齐，无时间弹性；②缺失祖先记 0 分而非跳过（`in_memory_backend.py:889-893`）；③权重前缀截断不重归一化；④ d1 与 d≥2 走不同后端函数（trap T1），**d=2 从未测过**。
- history 粒度 = inference step（`replan_steps=5`，`examples/libero/main.py:56`）⇒ d5 回看 25 个 env steps。

### 2.2 实验结果（`always_hit` 下 SR 直接度量排序质量）

libero_spatial（`exp/weighted_sum/analysis/libero_spatial/`）：

- 18 base × d{3,4,5,6}：top-10 均值 d1 73.0%，最优 depth Δ = **−6.4pp，10/10 全负**；最弱 4 base **+10pp**（rescue-the-weak）。
- wsweep（每 depth 独立重搜 78 组模态权重）：d1=74%，d3=d4=d5=72% 齐平 → 重调权重追不回。
- 171 逐步权重形状筛选：无一追平 d1；{Δ>0 且 McNemar p<0.05} 为空集；形状排序 peak > decreasing > … > uniform（最差）——越摊给旧帧越差。
- threshold-pareto：mean SR 随 depth 严格单调降（93.8→89.9）。

libero_10（`logs/session_handoff.md:98,382-395` + `exp/weighted_sum/analysis/libero_10/`）：

- always_hit：max SR d1 0.52 > d3 0.51 > d5 0.49 > d4 0.46 > d6 0.42（"negative result"）。
- 逐步权重筛选的唯一松动：d3 最优 0.56（+0.04 vs d1 prior），最优形状 trough/increasing（偏旧帧），与 spatial 形状排序**反转**；McNemar p=0.070，Bonferroni 下无一存活。
- threshold judge 下 d3 反而赢（peak SR 0.97 vs d1 0.93，per-x dominance 54%）。

延迟维度已排除：score memo 使 depth 几乎不增加 search 耗时 → history 的失败是纯排序/SR 问题，非成本问题。

### 2.3 团队已排除的解释

- **H-3a（模态权重过拟合 d1 regime）**：wsweep 中最优权重向量确随 depth 漂移，但 SR 纹丝不动 → 否证。
- **「递减逐步权重形状不合理」**（合作者提出）：171 形状扫完，decreasing 本就接近最优，uniform 最差 → 否证（至少 spatial）。
- 相邻线 `exp/temporal_prune`（用 history 改 key 而非改分数）同样负结果且窗口单调恶化——跨机制一致。

## 3. 核心假说：Markov 充分性（三层分解，强度递减）

**第一层（结构性，近乎必然）：检索目标天生 Markov，因为 teacher 无记忆。**
库 entry = (obs_t 的 key, π(obs_t) 的 action chunk)，而 Pi0.5 只吃当前帧 + robot_state + prompt，无历史输入。故给定当前观测，检索目标与历史条件独立：

```
I(a*_t ; h_{t-1} | o_t) = 0   （按构造成立）
```

即使环境存在混叠状态（历史不同、观测相同、需不同动作），base policy 同样分不清——库标签本身不含该信息，检索加历史只是在逼近一个本就不区分历史的函数。**cache 系统的排序天花板被 teacher 的 Markov 性钉死，环境是否 POMDP 反而次要。**（⚠ 修正见 §13.2：此绝对表述仅对未过滤 rollout 严格成立；success 过滤库存在 collider 例外通道，本项目库默认 success 过滤。）

**第二层（经验性，真正的战场）：给定的不是 o_t 而是有损的 key k_t。**
历史唯一可能的贡献是补偿 key 的信息损失：

```
history 有增量 ⟺ I(a*_t ; h | k_t) > 0 ⟺ k_t 对 o_t 不充分
```

假说的 pipeline 层表述：spatial_16/max_pool + robot_state + zscore-tanh 这套 key 已近似充分，该条件互信息 ≈ 0。

**第三层（环境性）：LIBERO 本身接近 obs-MDP。**
准静态操作、物体不自主运动、任务进度基本写在场景里、腕相机+夹爪状态消解「接近 vs 撤退」歧义、robot_state 给本体真值。真正的非 Markov 源仅剩瞬态遮挡与速度（而 teacher 本来就看不到速度，见第一层）。

## 4. 现有证据在假说下的落位

假说的统一预测：**历史收益 = key 不充分度的单调函数**。

| 观察 | 落位 |
|---|---|
| rescue-the-weak：弱 key +10~21pp，强 key −3~−12pp | 无意中跑出的**剂量-反应曲线**：key 信息损失越大历史越有用，正是 I(a*;h\|k) 随 key 退化增大的直接体现（最强单点证据） |
| wsweep 全平（74 vs 72/72/72） | 条件互信息 ≈ 0 时任何模态权重重调都变不出信息 |
| current-dominant 形状最优、uniform 最差、无一形状显著赢 | 历史帧为纯干扰时最优解 = 权重塌缩回当前帧；整个形状空间在优化一个系数为零的项 |
| temporal_prune 窗口单调负效应 | 机制不同的第二条 history 路径同样失败，跨机制一致性 |
| d5 回看 25 env steps 后单调恶化 | Markov 充分下 25 步前的帧是纯 distractor |
| **gating 有效 vs ranking 无效**：prev cp1_score 预测下步 MISS AUC 0.973–0.986（gate_research） | 假说的**双重预测**：历史对「动作目标」（第一层已证 Markov）无信息，对「检索可靠性」（误差过程时间自相关的 epistemic 量）信息量大。分界线恰落在 Markov 论证划出的位置 |

## 5. 假说未覆盖的残余

1. **libero_10 d3-trough/increasing 信号**（+0.04，p=0.070，未裁决）。两种读法：(a) 长时程任务存在少量真混叠（运输阶段腕相机视野、遮挡瞬态），第二层互信息在 libero_10 上不严格为零——与「基本充分」相容；(b) 171 格筛选的 winner's curse。当前证据无法区分。
2. **libero_10 threshold-pareto d3 赢**：可归因于 depth 拉宽分数分布（std 0.023→0.094）改善置信信号可分辨性——**calibration 收益而非信息收益**，与 ranking 层 Markov 充分不矛盾，反而再次呼应「历史对 epistemic 量有用」的分界。

## 6. 可证伪检验方案（全离线，无需 rollout，按性价比排序）

1. **动作预测残差检验（直接估 I(a*;h|k_t)）**：现有 H5 库 LOEO——用 k_t 做 kNN 回归预测 action chunk 首步，再加入 k_{t-1..t-d} 特征重做。强 key 下残差不降 → 假说成立；显著下降 → key 仍有历史可补的损失。CPU 级工作量。（→ §9.3 已升级为 A/B/C 三组特征判别设计）
2. **d1 失败案例尸检（区分混叠错误 vs 覆盖错误）**：用 `__hit_meta__` 的 winner_id/step，统计失败 episode 中 winner 的 task/step 偏差。若失败时 winner 仍同任务、时间对齐（只是动作不够好）→ 库覆盖/密度问题，历史无从修复；若存在大量高分错阶段/错任务 winner → 存在混叠、历史该救未救，需回查对齐脆性（H-B）。
3. **混叠率直接量化**：库内全对扫描「key 相似度极高但 action 距离大」的角落质量，每 suite 各算一个 aliasing rate。预测：spatial ≈ 0，libero_10 略高但仍小；若 libero_10 混叠对集中于特定任务，可顺带裁决 d3-trough 信号真伪。
4. **oracle 步对齐消融（排除竞争假说 H-B 时间对齐脆性）**：最强配置 + `step_filter: window` 强制步对齐重跑 trajectory search。oracle 对齐下历史仍无增量 → 排除对齐脆性，Markov 假说独占解释权；变好 → 问题在对齐机制而非信息本身。

检验 1、2 顺带回答 74% 天花板的残余错误是覆盖问题还是排序问题——对「预算投给加密库还是改检索」是决定性输入。

## 7. 战略含义（若假说成立）

- **历史信息在本系统的正确用途已被 gate 线找到：做 gating/置信信号，不做 ranking**（N4 已在 L=6 上 4/6 pass）。排序层面继续抢救历史的期望收益为零。
- 第一层论证同时给出一个一般性设计原则：**训练自由 cache 的检索 key 的信息上界由 teacher 的输入决定**——teacher 无记忆 ⇒ 任何历史侧检索增强的理论空间为零，只剩 key-denoising 一条路，而该路的收益随 key 质量趋于充分而消失。

## 8. 结论与后续

- 评估：第一层论证基本封死「历史提升检索排序」的理论空间；第二层的剂量-反应证据（rescue-the-weak）+ gating/ranking 分界已相当强。**本假说是当前所有候选中解释力最强、且唯一能同时解释正反两类结果的。**
- 后续（未排期，待 owner 决定）：§6 检验 1、2 优先；libero_10 d3-trough 若要裁决，走此前建议过但从未执行的确认性重跑（500–1000 ep、同批次 d1 anchor、多重比较校正）。

## 9. 增补（2026-08-12）：算子类分析——为什么 171 形状搜索注定为空集

> **触发背景**：owner 导师转来 WCM 论文（*A World Critic Model for Vision-Language-Action Reinforcement Learning*, arXiv 2607.29613）——VLA-RL 工作：critic 吃 K 帧观测历史（K=3 最优）、仅训练期用于 advantage 估计，policy 全程单帧（π0/π0.5/OpenVLA-OFT backbone），benchmark 含 LIBERO-Plus。**经讨论定性：该文为 RL 微调管线，方法与本 training-free cache 系统不可迁移，无零件可搬。** 保留记录仅因两点：
> ① **任务族/模型族证据（与其方法无关）**：同款 backbone + LIBERO 系任务上，SOTA 工作亦未给 policy 加历史、单帧即够，历史仅在估值侧（critic）起效——为 §3 第一/三层与 §4 的 gating/ranking 分界提供一个可引用的独立数据点。
> ② 其 Fig 5 消融（naive 历史拼接即使给 critic 也无效——"scalar value regression provides weak supervision for learning cross-temporal dynamics"；须配 next-latent 预测目标方有效）触发了本节以下分析。**以下分析是关于本系统自身打分公式的数学事实，不依赖该论文成立。**

### 9.1 trajectory search 的两级信息封闭（表达力失败，可证明）

`TrajScore(x) = Σ_l w_l·s_l`，`s_l = Score(anc_l(x), q_{t-l})`：

1. **标量化封闭**：每帧在进入时间组合前先塌缩为边际相似度标量，key 向量层面的时间结构（Δq 与 Δanc 是否同向等 dynamics 匹配特征）在此销毁——此类特征根本不是 {s_l} 的函数，任何后续组合都写不出。
2. **非负加性封闭**：{s_l} 残存的趋势结构（如「匹配质量随时间改善」，最简表达 = 差分核 [1,−1]）被加性（跨时刻交互项 ∂²TrajScore/∂s_i∂s_j ≡ 0）+ **非负约束（`config.py:2067-2069`，负权重过不了 validation）**封死。可配置空间 = 边际相似度的锥组合 = **纯低通平滑核**。

推论一：171 形状筛选是在**与 dynamics-aware 打分交集为空的函数类内部搜索**，调 w 只在选平滑核带宽/滞后——「空集」结果由算子类预先决定。
推论二：本失败是**表达力失败**（公式按构造写不出帧间关系，零学习环节，可证明），比「有信息但没学会」类失败更根本。

### 9.2 双条件框架（升级 §3/§4）

历史有增量 ⟺ **(a) 目标量依赖时间结构 × (b) 算子能表达该依赖**：

| 场景 | (a) | (b) | 结果 |
|---|---|---|---|
| 本线 ranking（trajectory search） | ✗ teacher 无记忆⇒目标 Markov | ✗ 平滑核 | d1 最优，171 形状全灭 |
| 本项目 gate（N1/N4） | ✓ 下步 verdict 时间自相关 | ✓ lag-1 分数足表 persistence | AUC 0.973–0.986 |

(a)(b) 独立可分。平滑核签名的回溯解释（全部本项目数据）：rescue-the-weak = 低通滤波定义性行为（信号噪→降方差，信号净→滞后偏差）；current-dominant 最优/uniform 最差 = 信号干净时最优核退化为 δ 核；wsweep 全平 = 模态权重在帧内、动不了时间算子类。**libero_10 d3-trough 新解读**：加性类唯一未封死的历史通道是**静态混叠消解**（旧帧边际相似度即可区分混叠分支，无需 dynamics）——若该信号为真，只能来自静态混叠（算子类装不下别的），比 §5-1 的二分更尖锐。

### 9.3 检验 1 升级：A/B/C 三组特征判别设计

- **A**：{k_t}（基线）/ **B**：{k_t, k_{t-1}, …}（raw 历史拼接 = 已扫过的信息集）/ **C**：{k_t, Δk_t, Δ²k_t}（差分特征 = 平滑核类取不出的 dynamics 通道）。
- 预测树：B、C 均不降残差 → 历史确实无信息，第一层 Markov 论证独占，ranking 侧结案；B 不降而 C 降 → 算子类是（部分）罪魁，存在从未触及的 dynamics 通道，改 search 才有理论依据。纯离线 CPU，同一套 LOEO 数据。**这是「Markov 充分」vs「算子错误」两解释间的判别实验。**

### 9.4 Judge 侧设计含义（推测，未验证）

judge/gate 功能上即本系统的估值器（判「此 hit 复用会否成功」）。若要在系统内任何位置引入「历史+时间结构」，对应槽位是 judge 而非 search；候选路线 = 「**预测下一步，预测误差做不信任信号**」：`walk_next` 即免训练世界模型——预测 = winner 后继 key，实际 = 下一步 query key，失配增大 = 离开缓存轨迹盆地 → 降信任/强制搜索。旁证：verdict factor judge 的一阶/二阶差分运动学因子（jerk/direction）本就在正确算子类内（差分 = 跨时刻项），或为 judge 线历史有效的深层原因。（对照：「相似度特征 + logistic」路线 TRACER M2 已走且为负结果。）

### 9.5 边界

本节分析独立于 WCM 成立；对该文的引用仅限触发背景 ① 的证据用途（任务族/模型族数据点，类比性，非方法迁移）。§9 不改变 §8 结论，仅收紧其机制解释并新增一项判别性检验（9.3）。

## 10. 讨论总结（2026-08-12 定稿，owner 口径 + 三处修正）

### 10.1 对现状的解释（假说定稿）

- **「history 有用」是人类直觉，对 π0.5 类无记忆 policy 不成立**：policy 输出只依赖当前单帧输入，任务中每一个决策都是无记忆决策，不以直觉为转移。
- **cache 继承 teacher 的无记忆**：基于模型中间结果（vision tower 之后 / backbone 间）构建的 cache，库标签 = π(o_t)、key = f(o_t)，整个系统的信息上界被 teacher 的输入钉死——cache 系统的工作方式本质上是**马尔可夫充分**的（修正一：非「马尔可夫均衡」，均衡为博弈论术语）。
- **trajectory search 的本质意义不在「提高 ceiling」，而在「提高不良搜索方式的成功率」**：keybuilder / 模态融合质量不良时 query key 噪声大，时间平滑降方差 → SR 升；key 已足够好时无方差可降，历史只引入**滞后偏差**（修正二：非「噪声」——平滑永远在降方差，其代价是把排序系统性拖向「过去像」的候选；数据签名为一致性下滑（10/10 全负、−3~−12pp 方向稳定、current-dominant 形状最优）而非不稳定波动，与偏差相符）→ SR 降。**拐杖论**：拐杖能让蹒跚老人走得更快，不可能让本就健壮的运动员拿着提高成绩。
- **开放条款（owner 增补）**：**或许 history frame 真的有用，但方法一定不是简单加权相加**。§9.1 已证该算子类（逐帧标量化 + 非负加性组合 = 纯平滑核）在公式层面只能做「拐杖」；若历史确有可取信息（dynamics、趋势、分支语境），提取它需要不同的算子（差分/预测式，参 §9.3 C 组、§9.4）。此条款使结论保持精确：**已被数据否定的是「加权平均边际相似度」这个方法，不是 history 本身**；history 是否有残余价值由检验 1（A/B/C）裁决——这正是双条件框架（§9.2）中 (a)「目标是否依赖时间结构」与 (b)「算子能否表达」的分离。

### 10.2 佐证实验清单（定稿）

1. **LOEO 动作预测残差，A/B/C 三组特征**（§6-1 + §9.3）：判别「历史无信息（Markov 充分）」vs「有信息但算子取不出」。优先级最高，纯离线。
2. **d1 失败尸检**（§6-2）：失败 episode 中 winner 的 task/step 偏差分布，区分混叠错误 vs 库覆盖错误，顺带裁决 74% 天花板归属。
3. **库内混叠率量化**（§6-3）：「key 极像但 action 差远」配对质量，两 suite 各一，顺带裁决 libero_10 d3-trough 真伪。
4. **oracle 步对齐消融**（§6-4）：`step_filter: window` 强制对齐重跑，排除时间错位竞争解释（H-B）。
5. **记忆假肢实验（新，owner 提出方向经重新框定）**：换非马尔可夫任务而 teacher 不换时，按 §3 第一层论证 history 仍不会帮助「逼近 teacher」（库标签里无历史信息可取）——故该实验测的是**更强命题：cache + history 能否作为记忆假肢超越无记忆 teacher**（混叠态上 teacher 靠运气选分支，成功库存有两分支各自正确的轨迹，history 匹配识别当前 episode 属于哪个分支并回放正确动作——cache 做到 policy 本身做不到的事）。任务挑选须瞄准「观测混叠」而非「更难」：复杂 ≠ 非马尔可夫（libero_10 不简单但动作侧近马尔可夫；WCM 核对显示其 4 benchmark 无一需要动作侧记忆，连 CALVIN 长程串接的记忆也被指令+可见场景状态外化）。理想设计 = 构造性两分支混叠任务（目标由 episode 早期、此后不可见的线索决定），无记忆 teacher ≈ 50%，测 history-retrieval 是否显著 > 50%。bootstrap 约束：teacher 须偶尔成功，否则无库可建。

### 10.3 与前文关系

§10 为 §3–§9 的收敛口径，对 §8 结论无改动；修正仅在表述层（噪声→偏差、均衡→充分）。第 5 项实验把负结果扩展为完整可检验故事：**history 有用 ⟺ (a) 任务需要超出 teacher 的记忆 ∧ (b) 算子能表达时间结构**——现有全部数据落在 (a)(b) 双不满足象限，实验 1 检验 (b)，实验 5 检验 (a)。

## 11. 致导师摘要（一段式，2026-08-12）

> 用途：owner 发送给教授的讨论成果摘要；内容为 §10 定稿口径的一段式润色。（文献定位与论文化评估见 §12。）

关于 history frame 在我们的 training-free cache 系统中几乎无增量（LIBERO spatial / libero_10 上最优配置几乎总是只用当前帧）的现象，我们目前的解释是："历史有用"是人类的直觉，但对 π0.5 这类无记忆 policy 并不成立，因为 policy 的输出只依赖当前单帧观测，任务中的每一个决策都是无记忆决策，不以我们的直觉为转移；因此基于其模型中间结果（vision tower 之后或 backbone 层间）构建的 cache 也继承了这种无记忆性，整个检索系统本质上是马尔可夫充分的。需要强调，这一点与任务难度无关：cache 的无记忆性是由无记忆的 policy 导致的，而不是任务太简单的产物，因此只要 teacher 仍是单帧 policy，换更难的任务同样看不到历史增量；"更难"不等于"需要记忆"，真正能改变结论的变量是任务是否存在观测混叠，而非难度本身。在这个前提下，trajectory search（带历史的检索）的真实作用不是提高性能上限，而是提高不良检索配置的成功率：当 key builder / 模态融合质量不佳、query key 噪声较大时，跨帧平滑压低噪声、显著拉升成功率（弱配置上实测 +10~21pp）；而当 key 质量已经足够好时，系统以低噪声逼近纯净的无记忆推理，历史帧不再有方差可降，只会引入陈旧帧带来的滞后偏差，成功率随之稳定下降（最优配置上 10/10 全负，−3~−12pp）。打个比方，trajectory search 是拐杖：能让蹒跚的老人走得更快，但不可能让健壮的运动员提高成绩。同时我们保留一个开放条款：或许历史帧确实有用，但用法一定不是我们目前"逐帧算相似度再加权相加"的方式，因为这类算子在数学上只能做平滑，表达不了帧与帧之间的动力学关系；被我们的数据否定的是这个方法，而非历史本身。为验证这一解释，我们计划两类实验：其一是离线判别实验，在动作预测任务上对比"仅当前 key / 加历史 key / 加差分特征"三组，区分"历史确实无信息"与"信息存在但被平滑算子丢失"，并对现有失败案例做混叠/覆盖归因；其二是考虑到 LIBERO 这两个 suite 可能本身就是马尔可夫充分的，我们计划构造观测存在混叠、无记忆 teacher 只能靠运气选择的任务场景，测试带历史的检索能否在真正需要记忆的任务上超越无记忆的 teacher 本身，而这也是历史信息在此类系统中理论上唯一能产生真实增量的位置。

## 12. 文献定位与论文化评估（2026-08-12，web 调研）

> 触发：owner 问「有没有论文讨论过这个/类似问题，能否成一篇 ICML」，并随后把命题从 cache 抬升到「模型本身是马尔可夫的」。本节记录调研结果（截至 2026-08-12）与定位结论。

### 12.1 相邻文献地图（五条线）

**线 1 — 历史让 BC policy 变差（训练期机制，必引必划界）**：Causal Confusion in Imitation Learning（NeurIPS 2019, arXiv 1905.11979，"more information can yield worse performance"）；Fighting Copycat Agents（NeurIPS 2020, arXiv 2010.14876）；Codevilla et al. inertia problem（ICCV 2019, arXiv 1904.08980）。机制 = 学习期捷径/伪相关；**本项目机制 = 推理期 Markov 充分 + 平滑偏差，无学习环节**——同表象不同成因，是卖点也是必须划清的界。

**线 2 — VLA 要不要历史（2025–2026 激辩中，题目热但拥挤）**：HAMLET（arXiv 2510.00695，主张单帧是根本局限，学习式 memory 有增益；两个对本项目有利的事实——**朴素拼接几乎无增益**、历史代价大 +4 帧→前向慢 35%/显存 3.6×）；ReMem-VLA（2603.12942）、VPWEM（2603.04910）、Explicit Language Memory（2608.04765）。⭐ **Present but Not Remembered**（arXiv 2607.03372，2026-07，最近邻）：对 frozen VLA 逐层线性探针 + 因果干预，结论「**历史中超出当前帧的独有信息接近零**，历史只是当前帧冗余副本，仅当前帧严重退化时被因果调用，重注入历史也无法消歧」——即本纪要第一层假说的**表征层实证**；与本项目互补（它审计模型内部表征，本项目刻画派生系统的功能后果）。WCM（2607.29613，见 §9 触发背景）：历史只进训练期 critic。

**线 3 — 检索式操控（本项目系统家族，确切命题无人占）**：VINN、Behavior Retrieval、DINOBot、RT-Cache（2505.09040）全部单帧检索；⭐ **ActionCache**（Training-Free Acceleration for VLAs with Action Caching and Refinement, arXiv 2607.06370，2026-07）：外部 cache + 多模态 key + warm-start，**与本系统设计高度同源，必引对照**。VLA-Cache（2502.02175）/C³ache/EfficientVLA：算力复用型，相邻帧相似度是省算力信号非检索信号。**「检索 key 该不该含历史」无人系统研究过。**

**线 4 — 理论与测量方法学**：Belief Representations for IL in POMDPs（1906.09510）、Learning Memory Mechanisms through Demonstrations（2411.07954）给出正方向（专家用记忆⇒学习者需记忆）；**反方向命题（专家无记忆⇒历史对模仿目标信息量为零）作为系统设计定律无先例**。测量工具：λ-discrepancy（Brown）、Markov Violation Score（2503.00206）、POBAX（2508.00046）。

**线 5 — 记忆 benchmark（已存在，实验 5 场地现成）**：**MIKASA-Robo**（arXiv 2502.10550）：32 个记忆密集操控任务（物体/空间/顺序/容量四类）；**RoboMME**（arXiv 2603.04639，2026-03）：16 个记忆任务分类学 + **π0.5 backbone 上 14 种 memory-augmented 变体**系统比较，结论「记忆有效但高度任务依赖」。

### 12.2 地盘判定

- 「模型是马尔可夫的」正面命题**两端已被占**：模型内部审计（PbNR）+ 记忆任务上加记忆有用（RoboMME，且用的就是 π0.5）。
- 本项目确切命题（history in 检索 key）**无人占**，但单独成文偏窄。
- **未被占据的抬升位 = 继承定律（Markovness is Inherited）**：teacher 无记忆 ⇒ 一切派生物（cache、蒸馏 student、检索库、rollout 数据集）的历史信息上界钉死为零。一行条件独立可证，系统后果无人刻画。本项目的独家资产：training-free cache 是该定律的**干净测量仪器**（无训练环节、不受 causal confusion 混淆），~14 万 ep 剂量-反应数据（弱 key +10~21pp / 强 key 全负）直接测出 I(目标;历史|key) 随 key 质量趋零。双条件框架（§9.2）收编全场：PbNR = (a)✗ 表征层证据、RoboMME = 记忆任务上 (a)✓ 故 memory 有用、HAMLET = 拼接 (b)✗ / 结构化 (b)✓、causal confusion = 训练期另一失败模式、本项目 = (a)(b) 双✗ 象限。

### 12.3 论文化路径与差距

**形态**：定律 + 双条件框架 + 三层证据（任务层审计 / 模型层引 PbNR / 派生层本项目数据）+ 判别实验。**锋利的可证伪预测**（在 MIKASA-Robo/RoboMME 上）：真正需要记忆的任务上，无记忆 teacher 的 cache 即使配带历史检索也无法超过 teacher 天花板——除非成功库中混叠分支均有存料且检索靠历史选对分支（§10.2-5 记忆假肢情形）；RoboMME 的 14 个 memory-augmented π0.5 变体可作上界对照。**A/B/C 升格为任务侧马尔可夫审计方法学**（配 λ-discrepancy 类工具，量化旧 benchmark 的记忆需求量——RoboMME 们造新任务证明记忆有用，无人量化旧任务为何不需要）。

**差距（不补难过审）**：① 正面对照（记忆假肢实验，负结果→完整刻画的分水岭；场地用 MIKASA-Robo，免自建）；② A/B/C 落地；③ 泛化性（单 policy 单 benchmark 族会被打；`ablation_study` 线的 SmolVLA/ACT 学生可复用）；④ **时效**：PbNR 2026-07、RoboMME 2026-03，半年内更挤。venue：ICML/NeurIPS 有先例（Causal Confusion 即 NeurIPS），CoRL/RSS 受众更自然。最短路径 = A/B/C（纯离线）+ MIKASA-Robo 上验继承定律预测。

### 12.4 参考文献（arXiv ID 索引）

1905.11979 (Causal Confusion) / 2010.14876 (Copycat) / 1904.08980 (Codevilla limitations) / 2510.00695 (HAMLET) / 2607.03372 (Present but Not Remembered) / 2603.12942 (ReMem-VLA) / 2603.04910 (VPWEM) / 2608.04765 (Explicit Language Memory) / 2607.29613 (WCM, §9) / 2505.09040 (RT-Cache) / 2607.06370 (ActionCache) / 2502.02175 (VLA-Cache) / 2606.08962 (C³ache) / 1906.09510 (Belief Repr. IL POMDP) / 2411.07954 (Memory Mechanisms from Demos) / 2503.00206 (Markov Violation Score) / 2508.00046 (POBAX) / 2502.10550 (MIKASA) / 2603.04639 (RoboMME)

## 13. 继承定律的形式化分析与 ICML 支撑力评估（2026-08-12）

> 触发：owner 问「继承定律有道理吗，能否数学证明」及「够不够支撑一篇 ICML」。结论：可证，但需拆为「两条引理 + 一条带环境假设的定理 + 一个可证明的例外通道」；例外通道击中本项目自身事实（库默认 success 过滤），据此修正 §3 第一层的绝对化表述（已在 §3 加指针）。

### 13.1 形式化与证明

**设定**：episode $(o_1,a_1,o_2,a_2,\dots)$；teacher 无记忆 $a_t \sim \pi(\cdot|o_t)$；历史 $h_t=(o_{<t},a_{<t})$；key $k_t=f(o_t)$；成功 $Y\in\{0,1\}$。

**引理 1（标签条件独立，一行即证）**：rollout 分布下 $a_t \perp h_t \mid o_t$。证：$p(a_t|o_t,h_t)=\pi(a_t|o_t)$ 按定义。∎ 定义级事实，非卖点。

**引理 2（去噪上界，三行可证；rescue-the-weak 的定理化）**：$I(a_t;h_t|k_t) \le I(a_t;o_t|k_t)$。
证：对 $I(a_t;\,o_t,h_t\,|k_t)$ 两次链式展开——一方面 $= I(a_t;o_t|k_t) + I(a_t;h_t|o_t,k_t)$，末项因 $k_t=f(o_t)$ 为条件变量的函数而等于 $I(a_t;h_t|o_t)=0$（引理 1）；另一方面 $\ge I(a_t;h_t|k_t)$。联立即得。∎
含义：**全部历史的边际价值被 key 的信息亏损 $I(a_t;o_t|k_t)$ 上界压死**；key 趋充分 ⇒ 上界趋零。rescue-the-weak 剂量-反应曲线 = 此上界收紧的经验轨迹；实验 1 B 组估计左端。

**定理 3（成功率天花板，需环境假设）**：若环境为 obs-MDP（转移与成功对 $(o_t,a_t)$ 可测），则 (i) success 过滤保持引理 1；(ii) 任何 $(o_t,h_t)$-可测检索策略（库动作的历史依赖混合）存在只依赖 $o_t$ 的策略达到不劣成功率。证明思路：(ii) 为标准 MDP 论证（Markov 最优策略存在 + 混合策略类对按-$o_t$-取边际封闭），约一页细心无障碍。**环境假设不可去**——去掉后 (ii) 为假，即命题 4 的入口。注意事项：需假设成功相关进度可由当前状态承载（或 Y 由终态决定），否则 (i) 中 $P(Y|o,a,h)$ 仍可依赖 h。

**命题 4（例外通道：success 过滤 = collider，重注历史信息）**：
$$p(a_t|o_t,h_t,Y{=}1) \propto \pi(a_t|o_t)\cdot P(Y{=}1|o_t,a_t,h_t)$$
环境非 Markov 时第二因子真依赖 $h_t$——$Y$ 是 collider，条件化产生 explaining-away，**引理 1 在成功条件化分布下失效**。玩具构造（闭式）：t=0 线索 $c\in\{L,R\}$ 此后不可见，t=2 两分支观测混叠，正确终局动作 $=c$；无记忆 teacher 上限 50%；success 过滤库中 $c{=}L$ 的成功轨迹在混叠处全执行 $a{=}L$，带历史检索（历史含 $o_0$）选对分支 → 100%。**继承定律的破口精确开在 collider 上，破口大小 = 历史对成功的超额预测力**。LIBERO 上仍 null 的原因：env ≈ obs-Markov 时 $P(Y|o,a,h)\approx P(Y|o,a)$，过滤几乎不重注信息——本身可测。

### 13.2 对 §3 第一层的修正

原表述「检索目标按构造与历史条件独立」**仅对未过滤 rollout 严格成立**。本项目库默认 success 过滤（`--outcome-filter` 默认 `success`），严格表述应为：「对 success 过滤库，成立当且仅当环境（就成功而言）obs-Markov」。此修正不动摇现有结论（LIBERO ≈ obs-Markov ⇒ 重注 ≈ 0，与全部数据自洽），反而使框架获得自己的例外预测：**实验 5（记忆假肢）设计上必须用 success 过滤库——这是定理机制本身，非实现细节**。

### 13.3 两通道综合与范围限制

**派生系统中历史的全部价值 = 去噪通道（引理 2 的 gap）+ 过滤通道（命题 4 的 collider 项），无第三条路。** 实验 1（A/B/C）测第一项，实验 5（假肢）测第二项——理论骨架与实验计划一一对应。文献钩子：命题 4 = outcome-conditioning 榨出超越行为策略的策略的检索形式，与 Decision Transformer / RvS 概念相邻，可引。范围限制：仅覆盖模仿型派生（cache/检索/BC 蒸馏），RL-from-rollouts 不覆盖；一切为 population 级——causal confusion/copycat 是叠加在「零条件信息」之上的有限样本病理（恰好收编线 1 文献）。数学难度诚实评估：全部初等（链式法则/DPI/collider/标准 MDP），卖点是组织力与实验对应，非技术深度（Causal Confusion NeurIPS 2019 理论量级先例）。

### 13.4 ICML 支撑力评估（分档判定）

- **仅理论骨架**：不够。引理 1 平凡、定理 3 标准，审稿人会问 "so what"。
- **理论 + 现有 14 万 ep 数据**：好 workshop / borderline 主会——单系统（cache）、单 backbone（π0.5）、单 benchmark 族（LIBERO）的负结果分析。
- **理论 + MVP 清单落地**：够格竞争 ICML/NeurIPS 主会。**MVP 必须件**：(a) A/B/C 判别结果（任一方向都是信息）；(b) 假肢正对照（MIKASA-Robo 或自建两分支，证明 collider 通道真实存在且可被检索利用——负结果→完整刻画的分水岭）；(c) 第二 backbone 复现（`ablation_study` 的 SmolVLA/ACT 学生现成，可加 OpenVLA-OFT）；(d) 与 PbNR / RoboMME / causal-confusion 的清晰划界（§12.1）。**加分件**：benchmark 马尔可夫审计方法学（λ-discrepancy 配合）、gate/ranking 分界旁证、ActionCache 系统对照。
- **双赢结构（对投稿有利）**：若 A/B/C 显示 C 组大幅降残差 → 故事转向「正确的历史算子」（(b)✗ 而非 (a)✗），换标题仍可发；若不降 → Markov 充分刻画成立。两个世界都有论文，预注册式写法可提前锁定。
- **时间线**：A/B/C 纯离线（周级）；假肢需 π0.5 在新环境 rollout（GPU 中等）；至 ICML 截稿（历年 ~1 月）约 5 个月，优先级拉满则可行。核心风险仍是时效（PbNR 2026-07 / RoboMME 2026-03，方向正在变挤）。

---

## 14. 判决结果（2026-08-13，五项实验实测；plan §11-3 回写）

> 实验计划与执行记录：[`markov_sufficiency_plan.log.md`](markov_sufficiency_plan.log.md)
> 逐实验报告：`exp/markov_sufficiency/analysis/{e1_residual,e2_forensics,e3_aliasing,e4_index_filter,e5_d3_confirmatory,synthesis}.md`
> 规模：离线 E1/E3 用全部 6 个 key builder 的库；rollout E4/E5 共 **12,400 个配对 episode**（950 ep/臂，已用尽 db_init + 官方 pruned 两池的全部 held-out 容量）。

### 14.1 §8 的"评估"需要三处修正

§8 原文写"第一层论证基本封死『历史提升检索排序』的理论空间"。实测支持这个方向，但**措辞必须收窄**：

| §8 原表述 | 实测后的准确表述 |
|---|---|
| `I(a*; h \| k_t) = 0`（**注意：这不是继承定律，见 §14.1a**） | E1 四个 primary cell 中三个的 Holm 后区间下界 > 0（spatial B-d3 +5.35%、C-g1.0 +8.80%，libero_10 B-d3 +2.86%）⇒ **该项确为正**，但这正是 §13 引理 2 允许并预测的去噪通道，**不构成对继承定律的削弱** |
| 历史无用 | **准确说法**：那点增益在**阶段对齐后消失**（E1-O 的 8 个 cell 无一支持 H-B；spatial 在 ε=0.05 下 Δ% 精确为 0.00%，且对齐后候选池中位仍有 8–20 个、无一退化到 ≤1），并且在生产算子（`top_k=1`）下取不到 |
| 混叠是候选解释之一 | **H-A 被明确削弱**：E3 的 12 个 (builder, suite) 组合**全部**判 `almost_no_aliasing`，ADR 0.24–2.79% vs 随机对 47–48% |

### 14.1a 一处措辞更正（修订 2026-08-14）——继承定律并未受损

> **编辑历史**：§14.1 的表格与 §14.3 初稿（2026-08-13 撰写）把 E1 的结果表述为"强版 Markov 充分性**被削弱**"。经复核，该措辞不准确且会误导，现更正如下；原表述保留在本条中以便追溯。

问题出在**条件在什么上**：

| 命题 | 条件变量 | 本实验的关系 |
|---|---|---|
| **继承定律**（§13 引理 1）`I(a*; h \| o_t) = 0` | **完整观测** `o_t` | **未检验**（需完整 `o_t`，本实验无此数据），**亦无一条结果与之冲突** |
| **引理 2 去噪上界** `I(a*; h \| k_t) ≤ I(a*; o_t \| k_t)` | **有损 key** `k_t` | **E1 直接测的就是左端**；实测为正（+2.9%~+8.8%），**落在引理 2 允许的范围内** |

⇒ **`I(a*; h | k_t) = 0` 从来不是本项目的主张**，引理 2 明确允许它为正。把 E1 的正结果说成"强版本被削弱"，等于把一个**符合理论预测**的观测记成了对理论的反证。

**E1-O 进一步给出了这点增量的归属**：在归一化进度 oracle 对齐下，spatial ε=0.05 的增益从 +5.35% 降到**精确的 0.00%**（8/8 cell 无一支持 H-B，且对齐后候选池中位仍有 8–20 个、无一退化到 ≤1）。也就是说，`I(a*; h | k_t)` 中几乎全部来自**阶段维度**——而阶段正是 `cp1_spatial_pool_16` 这类 pooled 视觉 key 丢掉的东西（key 里没有 cycle 索引）。**历史在这里扮演的是"补 key 之失"的去噪角色，不是新的动力学通道。**

这与纪要 §10 的定性表述"trajectory search = 拐杖（救弱不助强，强 key 下加的是滞后偏差非噪声）"完全一致，本实验的贡献是给它配上了量：拐杖的长度 ≤ key 的有损性，且在本系统的 key 质量下已短到生产算子（`top_k=1`）够不着、rollout 上被滞后偏差淹没（E4 复现 spatial depth 退化 −3.47pp）。

**因此对外表述应为**：不是"历史无信息"，而是"**历史的信息量以 key 的有损性为上界，且已被证实仅来自阶段维度、在现行算子类下不可利用**"。

### 14.2 五项实验判决表

| 实验 | 问题 | 判决 |
|---|---|---|
| **E1** | 历史/差分是否降低 kNN 动作残差？ | 降低，但幅度小（k=1 下 −1.7%～+8.8%），三个 cell 区间下界 > 0；pilot power 三个 < 0.8 ⇒ 按预注册规则降为估计性 |
| **E1-O** | 阶段对齐后历史是否恢复增益？（H-B 的真检验） | **8/8 cell 不支持 H-B**；spatial ε=0.05 下增益归零 |
| **E2** | 失败 episode 的 winner 是否更常来自错阶段？ | libero_10 **`aliasing`**（无条件 estimand +26.9pp，CI [+21.5, +31.9]）；spatial `inconclusive`（+6.9pp，未过 10pp 实用界）。**两源差距本身是结果**：threshold judge 的接受过滤把效应从 +26.9 压到 +16.5pp |
| **E3** | 高相似对的动作是否分歧？（H-A） | **12/12 组合 `almost_no_aliasing`**；ADR 随 \|Δcycle\| 单调上升（l10：1.2% → 6.8% → 16.0%）⇒ 相似度本身已隐含编码阶段 |
| **E4** | 名义 index 过滤能否救回历史？ | **两 suite primary 均无改善**（spatial −2.21pp 触发预注册降级；l10 +0.21pp）；**两个 interaction CI 都含 0** ⇒ 过滤不是 depth 失效的原因。spatial 复现历史 depth 退化（A2−A0 = −3.47pp，CI 排除 0） |
| **E5** | libero_10 的 d3-trough 是真信号还是 winner's curse？ | *待最后一批 rollout 完成后填写（估计性判读，不做 p 值判决）* |

### 14.3 §13"双赢结构"的落位：既非 A 也非 B，而是第三种

§13 预设了二分：*C 组大幅降残差 ⇒ 转向「正确的历史算子」故事；不降 ⇒ Markov 充分刻画成立*。实测落在中间，而且这个中间态比两端都更有信息：

- **C 组确实降残差**（spatial k=1 下 +8.80%，k=5 下 +12.42%）⇒ 差分特征里**有**东西；
- **但阶段对齐后归零**（E1-O）⇒ 那"东西"是阶段错位的替代补偿，不是 Markov 之外的动力学通道；
- **且生产算子取不到**：k=5 下效应普遍远大于 k=1（libero_10 的 C-g0.5 从 −0.20% 跳到 **+10.24%**），而生产是 `top_k=1`；
- **rollout 侧一致**：E4 在两个可配置的时间旋钮（window / exact）下都没能把历史变得有用。

⇒ 准确的论文表述是 **(a) 与 (b) 都不成立**，但不是各自的强版本：**目标对历史的依赖存在但极弱、且经 E1-O 定位仅来自阶段维度**（(a) 弱不成立；该弱依赖本身是 §13 引理 2 预测的去噪通道，见 §14.1a），**而现行算子类连这点弱依赖都表达不了**（(b) 不成立）。§5 的表达力闭包论证由 k=1 vs k=5 的反差获得直接实证支持。

### 14.4 一个"没能执行"的检验，须如实记录

**剂量-反应（key 质量谱系 × 历史增益）没有被真正执行。** 6 个 key builder 的质量代理 `median r_A` 跨度只有 spatial 7.5% / libero_10 2.6%，远小于因变量变异；E3 独立地看到同一件事（6 个 builder 的 ADR 高度同质，0.24–0.71% / 2.03–2.38%）。⇒ **不得**据此宣称"rescue-the-weak 的离线影子未被观察到"。要做这个检验需要刻意构造弱 key（降维 / 加噪编码），本 plan 未采集。§4 的 rescue-the-weak 证据仍只来自已完成的 rollout 实验。

### 14.5 一个方法学副产品

E4 实测发现：**`step_filter` 会制造 MISS**（window 20–26%、exact 24–29%），因为 index 过滤会清空候选集，那些步回退到 teacher 推理。plan §5.2 原本假设"`always_hit` 下每步都有 winner"，该假设仅在无 filter 时成立。

⇒ 任何"过滤提升 SR"的读法都必须先扣掉"少用 cache"这一项：A4（exact）在 spatial 上 SR 最高（0.730）**同时** MISS 率最高（23.9%）。这条对未来任何涉及候选集过滤的实验设计都适用。

### 14.6 后续建议（owner 决定）

1. **记忆假肢正对照**（§13 MVP 的 (b)）：本项目全部为负结果，需要一个构造性的正例证明 collider 通道真实存在且可被检索利用，否则"负结果 vs 完整刻画"的分水岭跨不过去。
2. **真正的弱 key 谱系**：降维 / 加噪编码，把剂量-反应从"测不了"变成"测得了"。
3. **E2 的因果方向**：现设计无法区分"错阶段 winner 导致失败"与"失败查询漂到库覆盖稀疏区"。E3 的 `|Δcycle|` 分层倾向后者，但要判定需要干预式设计。
