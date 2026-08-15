# TIER 实验设计全卡 v1（对应 outline v2 台账 X1–X13）

约定：两套件 = libero_spatial（天花板区间，无损验证）+ libero_10（主判别战场，teacher 0.868）；500 ep/臂 = 10 任务 × 50 官方 pruned_init，全部配对（同 init/seed）；统计 = McNemar + episode 级 cluster bootstrap，primary 族走 Holm；已有基建 = conductor 臂矩阵、hit/miss executor 路由、sidecar（ACT/SmolVLA）、__hit_meta__（cp1_score/hit_type/winner_id/searched）、逐步采集、cache_latency_bench 回放、client-controlled gate。

**Init 池协议（污染红线；2026-08-15 owner 纠正后核实版）**：官方全集 = 100 init/任务 × 10 任务 = 1000/套件，恰分两半，**没有可新铸的第三池**（差集池 = 全集 − pruned_init 的全部，蒸馏时已用尽；见 `logs/ablation_study_plan.log.md` L29/L76 与 tracked `split_<suite>.yaml`）。测试集只许测量、不许拟合："拟合"类活动（库构建/学生蒸馏/router 训练标签/阈值标定/任何超参与信号选择）限制在 B 内；"测量"类活动（各臂 SR、各信号 AUROC 评估）允许在 A 上进行。池结构：**A = 官方 pruned_init 500**（测试集，"冻结后每模型只跑一次"纪律，任何 fit 触碰即作废）；**B = 差集池 500**，内部既有切分 **B-train 45/任务（450，学生梯度数据源；cache 库 50 init 受 `protected_in_train` 锁在此侧）+ B-val 5/任务（50，零梯度且不在 cache 库内）**。TIER 新实验的标定与 router 标签只能从 B 内出：**首选 B-val**（唯一既不在学生训练内也不在库内的切片 = "新 init"条件的最佳合法代理）；不够用补 B-train 并披露分布内偏置（库对 B-train 覆盖稠密 → 相似度与学生成功率双向乐观）。router 协议 = B 内训练、A 池 shadow rollout 上评估。**硬数字论据（入 Q1）**：本 benchmark 上 router 合法可见的标注 episode 被官方 init 预算封顶在 ~500 条（ACT 确定性 → 每 init 基本一条），其中九成在学生训练分布内；最干净的 B-val 只有 50 条、按 ~80% 学生成功率失败标签不足 10 条——标签的数量上限与位置偏置都是 benchmark 结构给定的。

## X1 — factorial 替换矩阵（扩展）【P0】
- 目的：贡献 2 双重解离的干净归因。
- 设计：hit 槽 ∈ {replay, student, teacher} × miss 槽 ∈ {teacher, student} 全交叉 6 格。已有：(R,T)=0.704、(S,T)=0.888/0.830、(R,S)=0.466/0.474、纯 student、teacher 锚点。新增每套件 3 臂：**(T,S)-ACT、(T,S)-SmolVLA（干净归因臂/反对齐臂）、(T,T)（index 纯开销对照）**。实现：routing 增加 hit_executor=teacher 的恒等选项（小 config 扩展）。规模 3 臂 × 2 套件 × 500 ep = 3,000 ep。
- 分析：episode 级配对二元结果上报两主效应 + 交互（cluster bootstrap），禁止只报对角差。分解：replay@hit 伤害 = (T,T)−(R,T)；蒸馏增益 = (S,T)−(T,T)；miss 槽净效应 = (T,T)−(T,S)；交互 = (R,S) 残差。
- 产出：6 格 × 2 学生 × 2 套件 SR 表 + 效应分解表 + 逐步 hit_meta 日志。
- 判读：① (T,S) 同样崩（≈0.47）→ miss 状态对学生内在致命，index 是检测器（最强版主张）；② (T,S) ≈ 纯学生（0.794）→ 崩塌是 replay 前缀制造的，贡献 2 改写为 (S,T) 增益 + X2/X3 信号证据承重（预注册降级）；③ 中间 → 报分解，两种效应都真。

## X2 — 对照家族（生死线，动笔前先出数）【P0】
- 目的：贡献 3 "partition 而非 rate/compute 携带价值"。
- 设计：每套件在标定切分上**预注册 2 个两层操作点**（如 teacher 率 ~40%/~20%，记 OP-A/OP-B）。每 OP 四臂：**TIER**；**主对照 A** matched-total-compute random（用 X11 实测把 TIER 的 key+检索成本折算进对照的 teacher 率 p'>p_TIER，Bernoulli(p') 逐步派发，client-controlled gate 实现）；**主对照 B** yoked 逐-episode 率（TIER 先跑，读每个配对 init 的实现 teacher 率 p_e，对照按 Bernoulli(p_e) 逐步派发）；**次级** global-rate random + 多相位 periodic。含 replay 的操作点不入 primary（明写 scope）。X1 的反对齐臂 (T,S) 在分析中作双符号检验：对齐臂胜其 matched-random 且反对齐臂负于其 matched-random。
- 规模：(4 臂 × 2 OP − 复用 TIER) × 2 套件 ≈ 7,000 ep。协议表：名义率/实现率/实现 compute 及容差全部入表。
- 产出：每 OP 配对 SR + 实现率/实现算力表 + 逐步派发日志。primary 族 = 每套件 2 OP × 2 主对照 = 8 检验，Holm。
- 判读：① TIER 双胜 A、B → per-step 信息成立，thesis 全额；② 胜 A 不胜 B → 信息在 episode 级难度分配，主张降为 "experience-indexed budget allocation"（预写叙事）；③ 不胜 A → 分区价值 ≤ 信号成本，thesis 在该区间失败，论文重心移向解离+信号分析（kill 场景，故先跑）；④ 仅一 OP 胜 → "价值集中于中预算区间"。

## X3 — 信号 AUROC/校准 vs 基线【P0】
- 目的："index 是能力估计器"+ 对抗 kNN-OOD 的新颖性防线。
- 设计：**shadow 模式**跑纯学生闭环（search+judge 照常、派发关闭、恒执行学生）各 500 ep/套件。**数据协议**：AUROC 评估用的 shadow rollout 跑在 A 池（官方 pruned_init，纯测量）；trained-router 的训练标签来自 **B 池内部**（首选 B-val 50 init 切片，补充 B-train 须披露分布内偏置）的独立学生 rollout——训练在 B、评估在 A，任何信号/基线不得在 A 上拟合或调参。逐步记录：本方信号（cp1_score/margin/winner 持久性）+ 基线信号：kNN-OOD 距离（学生特征空间对学生训练集）、STAC 式动作时序一致性（相邻 chunk 重叠段距离）、SmolVLA 采样方差（K=8）/ACT 用 3-seed ensemble 分歧（仅一套件，控成本）+ **离线偏置 MC 头（信号级参考 + X14 热启动初始化，不再是独立系统基线）**——从学生 rollout 的 episode 结局拟合成功概率的小 probe（监督语义的三层缺陷见 outline Q&A Q1：反事实标签/分布错位/Bellman 耦合，故系统级对决由 X14 在线 RL 承担）；其 label-efficiency 曲线（AUROC 随标注 episode 数 {25,50,100,200,~450}，上限由官方 init 预算封顶）保留为信号级参考与热启动质量证据。**标签协议**（预注册）：episode 失败为结局；step 级 AUROC 按 lead-time 分层（距 episode 结束 ≥k 步的窗口单列；失败 episode 末段 X% 剔除或单列为"事故后检测"）；推断单位 = episode（cluster bootstrap）；所有基线共用同一标签同一聚类。**teacher-failure 对照**：同法在 teacher-only rollout 上预测 teacher 失败。
- 产出：AUROC 表（信号 × 分层 × 套件）+ 校准曲线 + CI。
- 判读：① 本方 ≥ 最佳免训练基线 → 信号有竞争力且零边际成本（够用即赢）；② ≈ kNN-OOD 且 X12a 显示 filter 无差 → 诚实收缩为"工程化 kNN 胜任估计器嵌入 serving"，解离仍立；③ 学生失败 AUROC ≫ teacher 失败 AUROC → 学生特异；≈ 则改口"难度感知分配"；④ trained-router 分支（预写立场）：零标签信号追平标签饥渴 router 即为胜（另有 fail-closed OOD 方向 / 免维护 / 每学生免单独 router / 与 replay 层共用基础设施四层论据，写入 §3.2 "why not a learned router" 段）；若 router 大标签量下反超，label-efficiency 交点本身入结论（"标签稀缺侧 index 是正确选择"——部署场景恰在该侧）。

## X4 — 前沿 + teacher-cheapening 基线【P0】
- 目的：贡献 3 前沿；回答"为什么要学生而不是便宜版 teacher"。
- 设计：TIER (τ_replay, τ_student) 网格 5–8 操作点/套件（4b 阈值臂矩阵部分现成）；基线：纯 teacher / 纯蒸馏学生 ×2 / demo-ACT（X12c）/ 最强 replay（**与 TIER 同标定程序**选阈值，对称条款）/ **teacher-cheapening**：(a) flow 积分步数 ∈ {10,5,3,2}（serve 配置）(b) 重规划间隔 ∈ {1×,2×,4×}（chunk 开环拉长）。≈7 基线臂/套件 ≈ 3,500 ep/套件（可减为 3+2 档）。
- 产出：SR × 实测 GPU-s/step 散点 + Pareto（normalized 与裸 GPU-s 双 panel）+ savings-ceiling inset（X11）+ Table 1（预注册 OP × SR/成本/三层占用率/驻留显存）。**层结构消融呈现（owner 2026-08-15 提议，零新增 rollout 家族）**：前沿图上把三种层结构画成独立曲线——teacher+replay（=cache_baseline 家族）/ teacher+student（=(S,T) 臂 + 两层阈值扫描）/ 完整三层——每层边际贡献一图可见（预期：replay 延伸超廉价端、student 填充中段、三层并集 = 完整前沿）。
- 判读：① TIER 在中段高于全部基线 → 贡献 3（区间限定措辞）；② 减步 teacher 在同算力追平 TIER → 预注册的重大修正：TIER 层级并入 cheap-teacher（replay/student/减步 teacher/全 teacher 四层）或主张收缩——这是必须提前知道的结果。

## X5 — miss 状态因果解剖【P0（由 P1 升级）】
- 目的：miss 处的难是被 index **检测**到的（内在）还是被系统**制造**的（闭环/replay 诱导）——§6.1 因果解读的支撑。
- 设计三探针：**(a) state-reset 闭环探针**：从 teacher-only rollout（shadow 打 hit/miss 标）采样 N≈200 miss 态 + 200 hit 态，按任务进度分箱配平；仿真器直接置位（记录 mujoco 状态需在采集 wrapper 加存档），学生从该状态闭环滚到底，测成功/子目标完成率。**(b) 接管剂量-响应**：teacher 主驾 episode 在首个 miss-onset 交学生 k∈{1,2,4,8} 个 chunk 后 teacher 接回，配对 init 扫 k 测最终成功率。**(c) DAgger 式 relabel**：对纯学生闭环轨迹（X3 的日志）离线跑 teacher 前向，算 teacher-学生动作分歧沿**学生访问分布**逐状态曲线，按 hit/miss 标分层，与 teacher 访问分布下对照。
- 产出：(a) {状态类 × 进度箱} 成功率表；(b) 成功率-k 曲线；(c) 分歧分布 × 标签 × 访问源。
- 判读：(a) miss 态起步成功率 ≪ hit 态（配平进度）→ 内在难，index 是检测器；无差 → 崩塌是反馈制造，§6.1 解读改写到交互项；(b) k=1 即崩 = 即时致命 vs 平滑退化 = 复合漂移，机制注脚；(c) 两种访问分布下 miss 态分歧都高 → 表征级检测；仅学生访问下高 → 漂移驱动。

## X11 — 级联决断率 + key 成本记账【P0】
- 目的：成本模型的脊柱；savings ceiling；x 轴无可指摘。
- 设计：**microbench**（扩展 cache_latency_bench，batch=1 实测 GPU-time + 解析 FLOPs 双口径）：SigLIP 编码 / Gemma prefix 全量（Stage 1）/ 各 key 变体构建 / 检索（已知 ~4ms）/ 学生前向含 sidecar IPC / 10 步 action expert 环。**级联埋点**：TIER eval 逐步记录级联路径——外部 key 单独决断占比（判 miss 直通 teacher / 判 hit-tier）、升级到内部 key 占比；摊销 key 成本/步 = P(升级)×c_internal + c_external。**级联关闭消融**：全内部 key 跑 2 个 OP 对比前沿。**对账**：normalized 预测每步时间 vs 实测（server-side 与 end-to-end 分列），逐层归因差值。
- 产出：成本权重表（=X4 x 轴的定义来源）、各 OP 决断率表、savings-ceiling 解析曲线、对账表、驻留显存表。
- 判读：摊销 key 成本 < teacher_full − student 成本 → student 层节省真实；若 ≥ → student 层在算力轴上中性，主张移向延迟/replay-重操作点（预注册风险检查，宁可现在知道）。

## X12 — 信号身份消融套件【(a)(d) P0 离线；(b)(c) P1】
- 目的：把"index=胜任索引"从"index=训练集密度"里剥出来（对抗同义反复攻击）。
- 设计：**(a) filter 消融（离线近零成本）**：库 ∈ {success-only（现行）, unfiltered, failure-only}，条目数配平（下采样同密度），用 X3 的 shadow 日志离线重放算 AUROC。**(b) 解耦批次**：teacher rollout 池二分为不相交 A/B，学生蒸馏于 A、库建于 B（1 学生类型 ACT × 2 套件控成本），重算 AUROC + 1 个路由 OP。**(c) demo-trained ACT**：在 LIBERO 原始人类演示上按标准配方训 ACT（数字社区可查），测 index 对它的失败预测 AUROC + hit→demoACT / miss→demoACT 两臂 × 2 套件（2,000 ep）。**(d) teacher-failure 对照**：并入 X3。
- 产出：(a) 3 库 AUROC 表；(b) 解耦 vs 同源 AUROC/SR 对比；(c) demoACT 的 AUROC + 路由臂 SR。
- 判读：(a) success-filter 提升 AUROC → "competence not visitation"实证成立，与 kNN-OOD 的差分坐实；无差 → 信号=到访密度，措辞收窄、kNN-OOD 差分降为工程；(b) 效应保持 → 非同源伪影；退化 → 定量报告同源贡献占比；(c) index 能路由零数据共享的学生 → 升格为通用胜任信号（全文最便宜的增强）；不能 → 主张限定于 library-distilled 学生。

## X14 — 在线 RL router 对决（trained-router 基线终版）【P0】
- 目的：正面证明「免训练检索-阈值路由 ≥ 训练出的 router」。监督学习经三层论证不可用（反事实标签 / 到达分布+后续语义双错位 / Bellman 耦合，见 outline Q&A Q1），唯一语义正确的训练路线 = 在线 RL → 基线就真跑在线 RL，三变体对应三种模式。
- **Router 规范**：与我们系统除「都在做路由」外零关系的独立网络。输入 = 进库前的模型内部特征 φ(o_t)（与 keybuilder 同源，公平同信息）；**屏蔽**：相似度分数、检索结果、任何库侧量（搜索照常跑——cache 档执行时取 payload 需要——但这些信息对 MLP 不可见）。实现落位：verdict 层新 Judge（MLP 顶替 threshold verdict 的位置，屏蔽契约由构造保证并有测试锁定；工程细节见独立 L3 plan）。三变体：**R_tsc** {teacher, student, cache} / **R_tc** {teacher, cache} / **R_ts** {teacher, student}，各对照同模式 TIER 配置。
- **训练 = batch on-policy RL**（REINFORCE with baseline，可升 PPO）：冻结权重跑一批 episode（B 池 init）→ 用 episode 成败 + 逐步执行成本算 return（reward = 成功 − λ·成本，λ 定操作点）→ 更新权重 → bundle 热切换下发 → 下一批。**离线偏置 MC 头只作热启动初始化**（steelman + 稀疏奖励收敛保险），不是独立基线。训练只能在 B 池（500 init 循环使用；对 B 过拟合、A 池泛化落差本身是测量的一部分）。
- **产出**：interaction-efficiency 曲线——冻结策略在 A 池的部署 SR 随累计训练 episode {500, 1k, 2k, 4k} 变化，对比零训练水平线（**全档曲线仅旗舰 run，其余 run 仅终点**——owner 裁 D5/D8b）；训练终点冻结权重 A 池 500 ep 配对 vs 对应 TIER 模式（3 变体）。
- **scope（owner 已裁 2026-08-15，D1–D8 按建议）**：libero_10 主战场 R_ts@λ₁（双训练种子）+ R_ts@λ₂ + R_tsc@λ₁ + R_tc@λ₁ 共五 run ≈ 20k 训练 ep + 冻结评测；spatial 最优配置 1 确认点。另 owner 裁定：全系统不用 cache warm-start，所有 cache hit = FULL_HIT 直接回放 clean action。
- 判读（预注册，每支可发表）：① router 追不上 → 免训练信号胜，问题关闭；② 花 N 交互 ep 追平/反超 → N 入结论（且换学生/λ/套件/模型版本即重付，index 免疫）；③ 热启动+慷慨预算仍不收敛 → 如实报告，200 步 1-bit 稀疏奖励是结构性原因。
- 真机语义披露：sim 内在线 RL 合法（烧 GPU）；真机上该训练过程 = 未训好的 router 在机队 explore。
- 依赖（owner 裁 D8a：**自含**）：λ 由实现 plan 自带 pilot 协议标定、cost 由自带 microbench 实测、热启动数据自采——X3/X4/X11 均非前置；实现细节与全部冻结参数见 [`logs/rl_router_baseline_plan.log.md`](../../logs/rl_router_baseline_plan.log.md)。
## X6 — key 来源消融【P1】
- 目的：贡献支撑"policy 自身表征是正确 key 空间"。
- 设计：固定路由栈，换 key：内部（vision-token pool / LLM-layer extract）vs 外部（CLIP；可选 DINOv2 新 builder）vs 本体感知（robot_state）vs 组合（weighted_sum 多模态）。每种：同 rollout 重建库 pkl（builder 现成）→ 同程序重标定 → 信号层用 X3 shadow 日志离线重放算 AUROC（主文，附**每变体查询成本列**）；系统层挑 top-2 key 各跑 2 OP 小前沿（附录）。
- 产出：AUROC × 查询成本表；附录前沿。
- 判读：内部 key 在 AUROC-at-cost 上占优 → 主张成立；CLIP 追平 AUROC → key-space 论证退守 X7 ε→δ + 维护成本论证（预写措辞）。

## X7 — ε→δ 动作一致性曲线【P1，离线为主】
- 目的：Remark 1 的经验面；内部 vs 外部 key 的分离证据。
- 设计：从已采 H5（obs+各段 embedding+动作）分层采状态对：同 episode 近程(|Δt|≤5)/同 episode 远程/跨 episode 同 init 任务/跨 init；headline 只用跨 episode 层。动作分歧：teacher 对子集状态（~2k 态）采样 K=8 动作 chunk，用 energy distance（协议写明）；首轮可用已记录单动作做廉价版并注明 caveat。每 key 空间画 ε-δ 分位曲线 + 层内 Spearman + P(δ 小|ε 小)。**反例挖掘**：CLIP 距离小但 δ 大的真实对，展示其内部 key 距离（大则为分离证据），做定性图。
- 产出：曲线族 × key × 层；反例画廊。
- 判读：内部 key 跨 episode 层更紧更单调 → §3.2 成立；否则 key-space 小节降格为"经验相当，选内部出于统一基底与自动同步"。

## X8 — 第二 benchmark 家族（确认集）【P1（升级），最贵项】
- 目的：普适性；LIBERO 上冻结的全部分析的一次性确认。
- 设计：候选 RoboCasa / SimplerEnv / MimicGen（选 teacher 可微调 + 学生可训 + 有 headroom 者）。完整复制管线：Pi0.5 微调 teacher → rollout 采集（库+蒸馏池）→ 库构建 → 学生蒸馏 → **一次性**跑冻结分析：X1 六格核心臂、X2 两主对照 × 2 OP、紧凑前沿（TIER 4 OP + 基线）。分析选择全部在 LIBERO 上注册后不再改。
- 产出：确认表（效应方向+量级 vs LIBERO）。
- 判读：复现 → 普适性；不复现 → 主张 scope 到 LIBERO + 差异讨论。**时间线 fallback**（预声明）：LIBERO-90 中库/学生训练未覆盖的任务子集作弱泛化探针。

## X9 — replay 误差 vs 库密度【P2，离线，附录 E】
- 设计：对 shadow 日志中 hit-eligible 状态，算 replay 动作对 teacher 实际动作的误差随 NN 距离分位的曲线；同状态学生误差（用 X5c relabel 数据）对照，找交叉点。
- 产出：误差-密度双曲线 + 交叉点。
- 判读：replay 仅在近重复区间误差 < 学生 → 解释三区制前沿与 replay 层存在的理由。

## X10 — 历史增强负结果表【P2，附录 C，数据已有】
- 设计：171 config 逐步权重扫描 × 2 套件（各 17,100 ep）+ depth{3..6} 扩展（7,200 ep）压缩为一表：每 depth 最优 config vs d1 的配对 ΔSR + CI。三句话观察式陈述，无命名无机制。
- 判读：固定结论——单帧 key 下无增益；left to future work。

## X13 — serving mini-bench【P2，可选，附录 B】
- 设计：现有 replica/router 基建，N∈{4,8,16} 并发 episode 压单 GPU server：TIER vs teacher-only 的 steps/s/GPU、每步延迟 CDF（p50/95/99）、10Hz 预算 deadline 命中率。
- 判读：per-step 节省是否兑现为 serving 吞吐；不做则正文主张收缩在 per-step compute（已按 Sys-6(b) 预设）。

## 前置基建改动清单（全部小改）
1. routing 增加 hit_executor=teacher 恒等选项（X1）。
2. shadow 模式（search+judge 开、派发关、恒执行指定执行体）+ 逐步信号落盘（X3/X6/X12a 共用；gate_research 采集缝可复用大半）。
3. client gate 的 yoked 逐-episode 率表输入（X2B）。
4. 采集 wrapper 存 mujoco 状态（X5a）+ miss-onset 接管调度（X5b）。
5. cache_latency_bench 扩 microbench 项 + 级联路径埋点（X11）。

## 推荐执行顺序（依赖与风险前置）
- **波 0（离线/轻 GPU，立刻可做）**：X11 microbench → X12a → X7 廉价版 → X10 → X9。X11 先行因为 X2 主对照 A 和 X4 的 x 轴都依赖它的成本权重。
- **波 1（rollout 主战，风险前置）**：标定切分冻结操作点 → X1 新臂 + X2（TIER 先跑、yoked 后跑）→ X4 基线 sweep。X2 出数决定全文叙事版本。
- **波 2**：X3 shadow 采集（同时喂 X6/X12a 复算）→ X5 三探针 → X12b/c → X6 系统层。
- **波 3**：X8 确认集（若时间线允许，否则 fallback 探针）。
