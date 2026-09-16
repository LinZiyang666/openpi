# 跨领域迁移调研：自动驾驶 / 导航中的扩散·流匹配规划与预测

> 调研员：driving_planning lane（2026-09-15）。依据 `logs/cache_transfer/briefing.md`、`docs/iclr/iclr_paper/arxivd.tex` L64–344、`logs/session_handoff.md` §1.2。
> 覆盖：DiffusionDrive / GoalFlow（NAVSIM，E2E 感知前缀 + 扩散/流头）、Diffusion Planner（nuPlan，向量化输入）、MotionDiffuser / TrajFlow（WOMD 预测）、NoMaD / NavDP / RoamFlow（视觉导航）、Bench2Drive 闭环。
> 标注规则：带 URL 的是文献数字；"估计"是我按文献数字推算；"不确定"是查不到。

---

## 0. 一页结论（先看）

- **结构对得上，但成本余量反了**。VLA 里头（flow-matching 动作专家）占 44% 成本、prefix 占 56%；驾驶的两类主流栈两头不讨好：
  - **E2E 感知栈**（DiffusionDrive / GoalFlow on NAVSIM）：前缀（ResNet-34 TransFuser BEV）≈ 14.6 ms，头（2 步截断扩散）只 7.6 ms → c_pre/c_0 ≈ **0.66**（hit 也只省 34%）；且 **1 步 ≈ 2 步**（87.9 vs 88.1 PDMS）。
  - **向量化规划器**（Diffusion Planner on nuPlan）：前缀（MLP-Mixer + 小 transformer）几乎免费，头（DiT 10 步 DPM-Solver++）占绝大多数 → hit 地板很低（估计 <15%），**但 5 步 DDIM 与 10 步同分，2 步 DPM-Solver++（TDDM）持平甚至更好**，减步傻基线的地板已经压到约 20–30%，warm 档没有 counterfactual。
- **局部性是"时间轴"的，不是"库"的**：10 Hz 重规划、8 s 视界，相邻两次规划重叠 ~99% —— 这让"从上一帧自己的输出 warm start"（Diffuser 2022 / CoDiG 2025 已做，3× 加速）成为免费的、必然被审稿人要求的对照；跨 episode 的库命中因为他车状态不重复而不可靠。
- **闭环与 RIT 这两块反而比 VLA 好**：nuPlan / Bench2Drive 是真闭环、有 open-loop→closed-loop 漂移的公开证据；NAVSIM 的 PDM 子指标（NC/DAC/TTC/EP）可离线逐决策算，能当比 L2 偏差更硬的 shadow 信号 D_a。
- **novelty 地形密**：RealDrive（检索增强扩散规划，NVIDIA 2025）、anchor/truncated 扩散（DiffusionDrive/AnchDrive/DriveAnchor：从存好的轨迹先验起步）、词表规划器（Hydra-MDP/VADv2：8192 条存好的轨迹里选）、记忆库预测（MANTRA/MemoNet/Recall-to-Predict）、时序 warm-start（Diffuser/CoDiG）、时序一致性记忆（MomAD）。我们剩下的是"training-free 分档 + RIT 离线标定 + 闭环 (IR,SR) 前沿"这层壳，但壳里能装的收益 ≤ 2 个去噪步。
- **结论：NO**（E2E 栈 NO；nuPlan 向量化栈 MAYBE-弱，需要 2 周傻基线证伪；导航 NO）。

---

## 1. 领域内最贴合的 2–3 个具体系统

| 系统 | 模型 / 开源 | benchmark | 成本拆分（前缀 vs 头） | 头的步数与减步证据 |
|---|---|---|---|---|
| **DiffusionDrive**（CVPR 2025 Highlight） | ResNet-34 TransFuser 感知（BEV 特征 + ego/agent query）+ 截断扩散 transformer decoder；20 个 K-Means anchor，从 anchored Gaussian 起步（训练截断到 50/1000）。代码开源 https://github.com/hustvl/diffusiondrive | NAVSIM navtest（非反应式、开环打分 PDMS；12,146 scenes，见 https://arxiv.org/abs/2406.15349） | RTX 4090：整体 45 FPS（≈22.2 ms），规划头 7.6 ms（3.8 ms/步 × 2 步）→ 前缀 ≈ 14.6 ms，**c_pre/c_0 ≈ 0.66**；TransFuser 裸跑 60 FPS（≈16.7 ms）；vanilla diffusion（TransfuserDP，UNet，20 步）7 FPS、130 ms 规划时间。https://arxiv.org/html/2411.15139 | 1 步 87.9 / 2 步 88.1 PDMS；vanilla 20 步 84.6（且模式多样性只 11%）。**减步基线已经内建在模型设计里。** |
| **GoalFlow**（CVPR 2025） | TransFuser 感知 + goal point 词表打分 + rectified flow 轨迹 decoder，每次采 128 条候选再打分选。https://github.com/YvanYin/GoalFlow（README 说代码 coming soon，2025-03） | NAVSIM navtest | 单样本 decoder 延迟：20 步 177.8 ms、1 步 10.4 ms（≈8.8 ms/步）；**前缀延迟未报**（估计 ≈ TransFuser 的 15 ms 量级，不确定）。https://arxiv.org/html/2503.05689v1 | 20 步 89.9 / 5 步 90.3 / **1 步 88.9**（−1.4 PDMS）。这是本领域"减到 1 步会掉一点"的唯一硬证据，但只掉 1.4 分。 |
| **Diffusion Planner**（ICLR 2025 Oral） | 向量化输入（agent 历史、lane、静态物）→ MLP-Mixer + 小 transformer encoder；DiT decoder（3 块、hidden 192）；DPM-Solver++ 10 步；可选 classifier guidance（推理期要梯度）。代码+ckpt 开源 https://github.com/ZhengYinan-AIR/Diffusion-Planner | nuPlan 闭环（Val14 1118 场景 / Test14-hard 272 场景，15 s @10 Hz；8 s 视界，20 Hz 推理） | 论文：0.04 s/次 @A6000（encoder/decoder 未分拆）；PlannerRFT Table A2：10 步 DPM 86.43 ms、5 步 DDIM 34.27 ms（硬件未注）→ 头 ≈ 7–9 ms/步，**前缀估计 <10 ms（不确定）**。https://arxiv.org/html/2501.15564 、https://arxiv.org/html/2601.12901v1 | 论文称"对步数鲁棒"（Fig. 7，未取到数值）；PlannerRFT 5 步 DDIM 89.96/84.46 vs 10 步 89.87/82.80（Val14 NR/R）；TDDM 用 **2 步** DPM-Solver++：Val14 89.81 vs DP 89.76、Test14-hard 77.95 vs 75.67。https://arxiv.org/html/2603.25462 |

补充参照：
- **闭环 E2E**：BridgeDrive 把 DiffusionDrive 搬到 Bench2Drive：80.79 DS / 58.18% SR；BridgeDrive 20 步、≈0.10 s/帧，87.99 DS / 74.99% SR，无步数 ablation。https://arxiv.org/html/2509.23589v2
- **感知前缀有多贵**（E2E 大栈）：UniAD 555.6 ms（UAD 论文测 465.1 ms，其中 Det&Track 31.2%、Map 19.8%、Motion 10.9%、Occ 9.9%；规划头只 9.7 ms）、VAD 224.3 ms、BridgeAD 157.2 ms、UAD 138.3 ms（规划头 1.5 ms）。https://arxiv.org/pdf/2406.17680 、https://arxiv.org/pdf/2503.14182 。**头在 E2E 栈里是零头。**
- **预测**：MotionDiffuser（WOMD，PCA 压缩轨迹 + 扩散）步数与延迟未能从 PDF 提取（不确定）https://arxiv.org/abs/2306.03083 ；TrajFlow（flow matching 预测）步数 ablation 未能提取 https://arxiv.org/pdf/2506.08541 。预测任务是开环指标（minADE/mAP），无闭环漂移，不是本系统的主战场。
- **导航**：NoMaD 19M 参数、训练 K=10 步 https://arxiv.org/html/2310.07896 ；RoamFlow（一步 MeanFlow）Gibson 延迟 19.6 ms vs NoMaD 49.1 / NavDP 61.1 / NaviDiffusor 89.5 ms，**且 SR +9.4、SPL +14.4 于 NavDP**——一步不但不掉点还涨点。https://arxiv.org/html/2606.29934v1

---

## 2. 映射表（驾驶 → 我们的符号）

| 我们的符号 | E2E 栈（DiffusionDrive/GoalFlow） | 向量化规划器（Diffusion Planner） | 视觉导航（NoMaD 类） |
|---|---|---|---|
| 共享前缀 c_pre | 多相机（+LiDAR）backbone → BEV 特征 + ego/agent/map query（≈14.6 ms，占 66%） | agent/lane/static 的 MLP-Mixer + transformer encoder（小，估计 <15%，不确定） | EfficientNet-B0 × 观测 + goal 图像 + transformer（占比未报，不确定） |
| tap point | BEV 特征 / 感知 query 输出处 | encoder 输出的 scene token | transformer 上下文向量 |
| 头（迭代生成） | 截断扩散 decoder（2 步）/ rectified flow（1–20 步） | DiT，DPM-Solver++ 10 步（可 2–5 步） | 1D UNet 扩散 10 步 |
| 中间状态（warm start 起点） | 第 j 步去噪轨迹（20 条 anchor 各一条）；**注意 DiffusionDrive 的 anchor 本身就是"存好的轨迹 + 小噪声"** | 第 j 步去噪 8 s 轨迹 | 第 j 步去噪 waypoint 序列 |
| tiers | tier 0 全算；warm：从库条目的中间轨迹继续；hit：直接回放库里 4 s（NAVSIM）/8 s（nuPlan）轨迹 | 同 | 同 |
| key 字段 | BEV 池化 token（相机）+ ego 状态（速度/加速度/指令）+ 导航指令 embedding（≈ 我们的 instruction cell） | scene token 池化 + ego 历史 + 路网局部几何 | 观测图像 token + goal token |
| 分数 s_t | 各字段 tanh 标准化后加权和；IVF cell 按导航指令 / 地图位置 | 同 | 同；cell 按拓扑图节点 |
| 偏差 D_a | tier-a 轨迹 vs 从噪声全算轨迹的标准化 L2；**更硬的替代：对 tier-a 轨迹离线算 PDM 子指标（NC/DAC/TTC/EP）差** | 同；nuPlan 也有离线 closed-loop 打分器可逐决策评 | waypoint L2 |
| 成功度量 / ε | NAVSIM PDMS（开环）；Bench2Drive Driving Score / SR（闭环） | nuPlan closed-loop score（NR/R） | SR / SPL |
| 成本模型 | c_a = c_pre + Σ_{j≥a} Δ_j，Δ ≈ 3.8 ms（DD）/ 8.8 ms（GF） | Δ ≈ 7–9 ms | 未报 |
| 闭环性 | NAVSIM **非闭环**（v2 是两阶段伪仿真）；Bench2Drive 真闭环 | nuPlan 真闭环，10 Hz 重规划 | 真闭环（真机/Habitat） |

---

## 3. 打分表 A–G

| 项 | 分 | 依据 |
|---|---|---|
| A 结构匹配 | **2** | 前缀 + 迭代头 + 天然 tap point 全都有（BEV/scene token）；但 E2E 栈头太便宜（7.6/22.2 ms），nuPlan 栈前缀太便宜——"两头都贵"的形态只在 vanilla 20 步扩散上出现，而那已被淘汰。https://arxiv.org/html/2411.15139 |
| B 成本余量 | **1** | E2E：c_pre/c_0 ≈ 0.66，hit 上限省 34%；k=1 基线 87.9 vs 88.1（DD）、88.9 vs 90.3（GoalFlow）。nuPlan：hit 地板低，但 5 步 = 10 步（PlannerRFT）、2 步 ≥ 10 步（TDDM）。导航：一步反而涨点（RoamFlow）。**减步傻基线几乎无损**，warm 档没有 counterfactual。https://arxiv.org/html/2503.05689v1 、https://arxiv.org/html/2601.12901v1 、https://arxiv.org/html/2603.25462 、https://arxiv.org/html/2606.29934v1 |
| C 局部性 | **2** | 时间轴局部性极强：10 Hz 重规划、8 s 视界，相邻规划重叠 ~99%；Diffuser 与 CoDiG 都用"上一帧输出加小噪声"warm start（3×）。https://arxiv.org/html/2505.13131 。**跨 episode 库命中不确定**：NAVSIM navtest / nuPlan Val14 都按场景类型抽样去冗余；他车状态不重复；RealDrive 用学出来的 embedding 做 FAISS top-K 只敢"条件化"不敢"回放"。https://arxiv.org/pdf/2505.24808 。导航拓扑图重访是例外（2–3）。 |
| D 容忍度与闭环 | **3** | nuPlan / Bench2Drive 真闭环；"开环近满分、闭环掉"是公开共识（BridgeDrive；NAVSIM↔Bench2Drive 相关性研究 ρ 非单调、有排名反转）。DS/PDMS 天然给 ε。代价：碰撞罚 0.5–0.6，单次回放错误的容忍度比 LIBERO 低。https://arxiv.org/abs/2605.00066 、https://arxiv.org/html/2509.23589v2 |
| E RIT 可用性 | **3** | shadow 可离线构造，且比 VLA 好：NAVSIM/nuPlan 都有规则打分器，D_a 可以直接是"tier-a 轨迹的 NC/DAC/TTC/EP 相对参考的损失"而非 L2；Hydra-MDP 证明这些子指标可被学出来（对 8192 条候选打分）。命中成串：直行巡航/红灯等待段天然成串。https://arxiv.org/html/2406.06978 |
| F novelty 地形 | **1** | 已有：RealDrive（RAG + 扩散规划，检索条目在去噪过程里插值，WOMD 碰撞率 −40%，开环）；anchor/truncated（DiffusionDrive、DiffVLA、AnchDrive 的"动态上下文 anchor"、DriveAnchor、DiffRefiner）= 从存好的轨迹先验 warm start 已训练进模型；词表规划（VADv2/Hydra-MDP/Hydra-MDP++ 91.0 PDMS）= 在固定库里选、生成头为零；记忆库预测（MANTRA CVPR 2020、MemoNet、Recall to Predict 2026）；时序 warm start（Diffuser 2022、CoDiG）；MomAD/MomADv2 时序记忆；DriveCache（跨步特征缓存，世界模型）。剩给我们的：training-free 分档 + 一个 δ 切所有 cut + 闭环 (IR,SR) 前沿定义。https://arxiv.org/pdf/2505.24808 、https://arxiv.org/abs/2509.20253v1 、https://arxiv.org/html/2406.06978 、https://arxiv.org/html/2608.16354 、https://arxiv.org/html/2503.03125 |
| G 做得动 | **2** | Diffusion Planner 代码+ckpt 开源，1×H100 足够；nuPlan devkit + 数据（数百 GB）与闭环仿真是 CPU 密集，Test14-hard 272 场景一点估计 1–3 h（不确定）；Bench2Drive 需 CARLA；NAVSIM 需 OpenScene 数据但只开环。三周内在 nuPlan 上出一条前沿可行，但每点成本比 LIBERO 高一个量级。https://github.com/ZhengYinan-AIR/Diffusion-Planner |

**合计 14/21，但 B=1 与 F=1 是一票否决项**：没有减步会掉点的证据，warm 档就没有存在理由；没有 warm 档，系统退化成"hit-or-miss 二元 + 一个 RIT 阈值"，而 hit 在 E2E 栈里只省 34%、在 nuPlan 里省的就是 2 个去噪步。

---

## 4. "减迭代次数"傻基线：文献里已经做过，结论一致——几乎无损

| 系统 | 步数 → 指标 | 出处 |
|---|---|---|
| DiffusionDrive（NAVSIM） | 1 步 87.9 / 2 步 88.1 PDMS（vanilla 20 步 84.6） | https://arxiv.org/html/2411.15139 |
| GoalFlow（NAVSIM） | 1 步 88.9 / 5 步 90.3 / 20 步 89.9 | https://arxiv.org/html/2503.05689v1 |
| DiffE2E（NAVSIM） | 1 步最低、2 步峰值、再多缓降（数值未取到） | https://arxiv.org/pdf/2505.19516 |
| PRIX（NAVSIM） | 步数 2→50 越多越差（over-smoothing） | https://arxiv.org/html/2507.17596v2 |
| Diffusion Planner / PlannerRFT（nuPlan 闭环） | 10 步 DPM 89.87/82.80 vs 5 步 DDIM 89.96/84.46（Val14 NR/R） | https://arxiv.org/html/2601.12901v1 |
| TDDM（nuPlan 闭环） | 2 步 DPM-Solver++：Val14 89.81、Test14-hard 77.95，≥ DP 的 10 步 | https://arxiv.org/html/2603.25462 |
| RoamFlow vs NavDP/NoMaD（导航） | 一步 19.6 ms，SR/SPL 反超多步 | https://arxiv.org/html/2606.29934v1 |

**为什么多峰性没有救场**：驾驶输出确实多峰（左转/直行、让/抢），但本领域 2024 年后的主流做法是把多峰塞进 **anchor / goal 词表 / 候选打分**，扩散头只做 anchor 附近的小幅精炼（DiffusionDrive 截断到 50/1000 噪声、GoalFlow 先选 goal 再 flow）。头的任务已经是单峰的，所以 1–2 步足够——这正是 briefing §2 讲的"一步 Euler ≈ 条件均值，单峰无损"在驾驶里的形态。只有 vanilla 从纯噪声起的扩散（TransfuserDP 20 步 84.6、模式多样性 11%）会在 1 步崩，但那类模型已无人用。

与 VLA 对比：VLA 的减步地板受前缀限制（60.6% / 40.7%），cache hit 地板 15%，中间有 25–45 pp 的空间只有 cache 能到；驾驶 E2E 栈减步地板 ≈ (14.6+3.8)/22.2 = **83%**、hit 地板 **66%**，空间 17 pp，且这 17 pp 里 SR 要靠回放 4 s 轨迹换；nuPlan 减步地板（2 步）估计 20–30%、hit 地板估计 5–15%，空间 ≤ 20 pp，绝对值是 ~15 ms/决策。

---

## 5. 预期收益（尽量给数、给不出就说不确定）

- **hit 率**：
  - 时间轴（同 episode 上一决策作为库条目）：几乎 100% 可"warm"，因为相邻规划重叠 99%（10 Hz、8 s 视界）；CoDiG 已用它拿 3× 加速，但代价是"更保守、更粗"，且它是**免费**的，不需要库、不需要 IVF。https://arxiv.org/html/2505.13131
  - 跨 episode 库：**不确定**，文献无人报告"可回放"的命中率。RealDrive 的 top-1 检索在 nuScenes/WOMD 上语义相似（图 3），但它把检索结果按 sigmoid 调度插进去噪过程，随机检索（Setting 5/6）会显著掉点——说明检索质量对结果敏感，直接回放风险更大。https://arxiv.org/pdf/2505.24808
  - 导航拓扑图重访：同环境重复行走时命中率可以很高（ViNT/NoMaD 本来就在先前遍历建的图上跑），但见 §4，一步流匹配已经比多步强，收益只剩 hit 省的头成本。
- **IR 能压到多少**：E2E 栈理论地板 0.66（100% hit）；实际按 VLA 经验 hit 率 30–60% 估，IR ≈ 0.80–0.90，而傻基线 k=1 就到 0.83 且 PDMS −0.2。nuPlan：hit 地板估计 0.05–0.15，2 步基线 ≈ 0.2–0.3；cache 只在"hit 率很高且回放不掉分"时才在 0.15–0.25 这一小段赢。
- **SR 损失**：回放 4–8 s 轨迹在他车变化时的损失无文献数字（不确定）；但 nuPlan Test14-hard 的 NR/R 差 7 分、Bench2Drive 碰撞罚 0.6，提示闭环回放误差的惩罚比 LIBERO 陡。
- **与 VLA 对比**：VLA 的问题是"减步不掉点所以 warm 没戏"；驾驶把这个问题加倍——减步不掉点 **且** hit 本身也省不了多少（E2E）或省的就是两步（nuPlan）。

---

## 6. novelty 地形与最强反驳

已占位的坑（按与我们的重叠度排序）：
1. **RealDrive**（NVIDIA，2505.24808）：任务相关 embedding + FAISS KNN 检索训练集 expert 轨迹，检索条目在去噪各步与当前观测/动作插值（RIM）；nuScenes/WOMD 开环 minADE/CR 改善、WOMD CR −40%。它是"检索 + 扩散规划"的正主，差别是它要联合训练、不做分档、不做闭环 IR。
2. **Anchor/truncated 扩散**（DiffusionDrive、DiffVLA、AnchDrive 的静态词表 + 动态上下文 anchor、DriveAnchor、DiffRefiner）：把"从存好的轨迹起步、只跑少量步"训练进模型。**我们的 warm 档在这些模型上就是"换一个 anchor 来源"**。
3. **词表规划器**（VADv2 / Hydra-MDP / Hydra-MDP++ 91.0 PDMS）：8192 条存好的轨迹 + 学出来的多头打分，生成头为零。**hit 档 = 一个只有 1-NN 打分器的词表规划器**——审稿人一句话就能打掉。
4. **时序 warm start**（Diffuser 2022 附录、CoDiG 2025 3×）与 **MomAD/MomADv2** 时序一致性：驾驶社区默认的"重用上一帧"路线；我们的库检索必须证明比它强，而它免费。
5. **记忆库预测**（MANTRA CVPR 2020、MemoNet CVPR 2022、Recall to Predict 2026）：非参数记忆检索未来轨迹，多模态预测。
6. **DriveCache / TeaCache 类**：跨去噪步特征缓存，目前打在世界模型视频生成上（1.8–2.0×），不在规划器上；说明"cache 加速扩散"的词已被占。

我们还剩：(a) training-free 的三档调度 + 一个 δ 切所有 cut 的 RIT 离线标定；(b) "可恢复冗余 R_C(ε)" 这一闭环量在驾驶规划器上的定义与测量；(c) 用 PDM 子指标当 shadow 风险（比 L2 偏差更贴安全语义）。

**最强反驳**（预演审稿意见）：
- "你的 hit 档就是 Hydra-MDP 去掉打分头；warm 档就是 DiffusionDrive 换 anchor；时序 warm start 免费且已被 CoDiG 做过。你在 DiffusionDrive 上最多省 34%，而它已经 45 FPS 实时；在 Diffusion Planner 上省的是 2 个去噪步。IR 收益不足以支撑一篇方法论文。"
- "NAVSIM 不是闭环，你的漂移故事只能在 nuPlan/Bench2Drive 讲，而这两处的 2 步基线已与 10 步同分。"

---

## 7. 首个实验方案（≤ 2 周，只为证伪 B）

**选型**：Diffusion Planner（nuPlan，开源 ckpt，1×H100 跑推理、CPU 跑仿真）。理由：它是三类里唯一"前缀便宜、头贵"的，是 cache 唯一可能有正收益的形态；E2E 栈（c_pre/c_0=0.66）不必试。

**步骤**（第一步永远是傻基线 + 成本拆分）：
1. 装 nuPlan devkit + Test14-hard（272 场景）数据；复现 DP 官方 Test14-hard NR 75.99。
2. **成本拆分实测**：encoder ms / DiT 单步 ms / guidance 开关，在同一 GPU 上量；得 c_pre、Δ。
3. **傻基线**：DPM-Solver++ k ∈ {1, 2, 3, 5, 10}，Test14-hard NR+R 各一遍（每点估计 1–3 h，不确定）。判据：若 k=2 与 k=10 的差 ≤ 0.5 分（TDDM 已暗示是），warm 档判死。
4. **免费对照**：上一帧输出加噪到 ρ∈{0.3, 0.5} 再跑剩余步（Diffuser/CoDiG 式），同样扫 Test14-hard；这是任何库检索方案必须赢的基线。
5. 仅当 3 和 4 都留出 ≥ 1 分 × ≥ 10 pp IR 的空间时，才建库：用 nuPlan train split 同城场景做 collect → shadow（D_a 用 nuPlan 离线闭环打分器算 tier-a 轨迹的 score 差，而非 L2）→ RIT 三档，出 (IR, score) 前沿。
6. 若 5 不成立，本线关闭，记录数字。

**算力**：单 H100 推理绰绰有余；瓶颈是 nuPlan 仿真 CPU 与磁盘（数据数百 GB）。两周内可完成 1–4。

---

## 8. 结论

**NO**。驾驶规划把 VLA 的死穴放大了：主流扩散/流规划头已经通过 anchor/goal 词表把多峰吞掉，1–2 步与全步同分（DiffusionDrive 87.9/88.1、GoalFlow 88.9/90.3、nuPlan 5 步=10 步、2 步≥10 步），warm 档没有 counterfactual；而 hit 在 E2E 栈里只能省 34%（前缀占 66%）、在向量化栈里省的就是两个去噪步；跨 episode 库命中又被他车状态打散，时间轴局部性早已被免费的"上一帧 warm start"占用。闭环漂移与 RIT 的离线安全打分是这里唯一比 VLA 强的两点，但它们只是壳，壳里装不下收益。唯一值得花两周的是在 Diffusion Planner 上跑 §7 的傻基线把这个 NO 钉死。
