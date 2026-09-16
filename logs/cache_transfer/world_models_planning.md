# 跨领域迁移调研：决策回路里的世界模型（world models for planning / RL / evaluation）

> 调研日期 2026-09-15。依据 `logs/cache_transfer/briefing.md` §4 的 8 段格式。范围限定为**世界模型在闭环里被反复调用**的场景（想象式规划 / MPC、world foundation model 做规划与评估、驾驶与机器人世界模型的闭环评测）；把世界模型当视频生成器的用法不在本文范围。
> 数字来源一律给 URL；文献没有的数字标"估计"或"不确定"，不编。

## 0. 先把领域切成三个子族（决定了后面所有打分）

| 子族 | 代表系统 | 世界模型在哪一步被调用 | 对我们形式的意义 |
|---|---|---|---|
| **(i) 决策时规划**（decision-time planning） | TD-MPC2、V-JEPA 2-AC、Diffuser 类扩散规划器、Cosmos Policy 的 best-of-N、Drive-WM 树状 rollout | **每个控制步**：编码器一次 + 几十到上万次 rollout 头 | 与我们"每决策一次前缀 + 头"完全同构，闭环漂移天然存在 |
| **(ii) 训练时想象**（imagination for training） | Dreamer v3、IRIS、DIAMOND | 只在训练 actor-critic 时 rollout；**部署时策略直接出动作** | 部署时没有可省的头；缓存只能省训练算力，不存在闭环 SR 故事 |
| **(iii) 世界模型当评测器 / 数据引擎** | WorldEval、WorldGym、DreamDojo、Cosmos-Surg-dVRK、NAVSIM v2 pseudo-simulation | 对**每个被评策略**在固定初始状态集上 rollout 世界模型 | 有跨策略的场景重复（局部性强），但 hit = 复用别的策略的想象未来，结构上不成立；warm 可以 |

结论先行：**只有子族 (i) 值得迁**；(ii) 从定义上没有决策时的头可省；(iii) 局部性最好但成功度量不是 SR，且 NAVSIM v2 一类已经把评测算力摊薄了。下文打分以 (i) 为主，(ii)(iii) 只作说明。

- Dreamer v3 的 imagination 只用于训练（"improves its behavior by imagining future scenarios"），部署时 actor 直接作用在 latent 上：https://arxiv.org/abs/2301.04104
- DIAMOND 同理（agent 在扩散世界模型里训练，部署时策略直接出动作）：https://arxiv.org/abs/2405.12399

## 1. 领域内最贴合的 2–3 个具体系统

### 1.1 TD-MPC2（首选实验平台）

- 代码 MIT 开源，含 1M–317M 单任务 / 多任务 checkpoint：https://github.com/nicklashansen/tdmpc2 ；论文 https://arxiv.org/abs/2310.16828
- 结构：encoder（状态输入为 2 层 MLP，像素输入为小 CNN）→ latent z_t（latent_dim 512）→ MPPI 规划：每步 `num_samples 512 × iterations 6 × horizon 3`，elites 64，其中 24 条轨迹来自 policy prior；**规划用上一步解平移一格做 warm start**（Table 8 "shifted by 1"）。配置原文：https://raw.githubusercontent.com/nicklashansen/tdmpc2/main/tdmpc2/config.yaml
- 成本拆分：每步世界模型评估次数 N×I×H = 512×6×3 = **9216**（Dream-MPC 论文 §5.6 的算法对比口径），实测 TD-MPC2 **20.83 ± 0.14 ms/步（RTX 4090，Acrobot Swingup）**：https://arxiv.org/abs/2605.04568 。TD-MPC(2022) 报告默认设置 ≈ 20 ms/步（50 Hz），H=1 时 ≈ 12 ms，"只用联合训练的 policy πθ 比规划快近 6×，但一般更差"：https://arxiv.org/abs/2203.04955
- 编码器占比：FLOPs 上 encoder 只是 9216 次 rollout 头评估里的 1 次，**c_pre/c_0 按 FLOPs ≈ 0**；但 batch=1 的小 MLP 是 kernel-launch-bound，wall-clock 上 policy-only（encoder + 一次 policy MLP）≈ 规划的 1/6 ≈ 3.5 ms（由上面 6× 换算，估计），hit 只跑 encoder + kNN 应低于此。**实际 wall-clock 地板需实测**（Dream-MPC 用 15 次评估替代 9216 次，wall-clock 只从 20.83 降到 18.15 ms，说明这一档规模的延迟几乎全是 launch 开销：https://arxiv.org/abs/2605.04568 ）。
- 成功度量：Meta-World MT50/MT80 有 success rate；DMControl 用归一化 return。
- 规划的贡献：PWM 复现 MT30 时"TD-MPC2 without planning 解不了 MT30"，说明 policy-only（= 头步数砍到 0）在多任务上崩：https://arxiv.org/abs/2407.02466

### 1.2 V-JEPA 2-AC（成本最像我们、但闭环只能真机）

- 代码/权重开源：https://github.com/facebookresearch/vjepa2 ；论文 https://arxiv.org/abs/2506.09985
- 结构：冻结 ViT-g 编码器（~1B）一次编码当前观测 + 目标图；300M predictor 做 CEM 规划，**800 条样本、16 s/action**；对比 Cosmos 做同样规划 80 条样本要 4 min/action。receding horizon：只执行第一个动作再从头规划。任务 SR：reach 100%、grasp cup 65%、grasp box 25%、pick-place cup 80%、box 65%（真机 Franka）。
- c_pre/c_0：ViT-g 一次前向（估计几十到一百多 ms）对 16 s 的规划，**< 1%**，是三者里 hit 省得最多的；但闭环只有真机，两周内我们跑不了。

### 1.3 Cosmos Policy（能直接落在我们 LIBERO harness 上的"世界模型进决策回路"）

- 代码/权重开源（Predict2-2B 微调），LIBERO 98.5% / RoboCasa 67.1%：https://arxiv.org/abs/2601.16163 ；LIBERO 权重 https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B
- 推理延迟（1 张 H100）：1 / 5 / 10 去噪步 = 0.16 / 0.61 / 0.95 s；model-based planning best-of-N=8 用 8 张 H100 并行 4.9 s/chunk（同上论文）。
- 意义：它同时是 VLA 与世界模型，planning 模式 = 采 N 个动作 chunk + 想象未来 + 价值挑最好。库里存"想象未来 + 价值"就是本领域的库条目。缺点：单卡 N=8 要串行 ≈ 5 s/chunk，500 集一个前沿点 ≈ 20 h（估计），且 planning 相对直出策略的增益论文摘要没给数（**不确定**）。

### 1.4 其它（只列位置）

- DIAMOND：4M UNet，默认 3 去噪步，Atari 12.7 ms/帧；CS:GO 版 381M，RTX 3090 上 10 Hz。**属于子族 (ii)**：https://arxiv.org/abs/2405.12399
- 驾驶：Epona 2.5B（关掉视频头只做轨迹规划可到 20 Hz）https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Epona_Autoregressive_Diffusion_World_Model_for_Autonomous_Driving_ICCV_2025_paper.pdf ；Drive-WM 树状 rollout 选轨迹 https://arxiv.org/abs/2311.17918 ；NAVSIM v2 pseudo-simulation 每场景 13 次规划器推理（vs nuPlan 闭环 80 次），与闭环相关 R²=0.8：https://arxiv.org/abs/2506.04218
- 评测器：DreamDojo 与真机 SR 的 Pearson r=0.995 https://arxiv.org/abs/2602.06949 ；WorldGym r=0.78 https://world-model-eval.github.io/ ；Cosmos-Surg-dVRK r=0.718 https://arxiv.org/abs/2510.16240 ；WorldEval https://arxiv.org/abs/2505.19017
- 不开源或不可控：Genie 3（仅产品入口）https://deepmind.google/models/genie/ ；WHAM 1.6B 权重开源但是游戏帧生成、无规划回路 https://huggingface.co/microsoft/wham ；1X 世界模型挑战只放数据与基线 https://github.com/1x-technologies/1xgpt

## 2. 映射表（以 TD-MPC2 为主，括号里给 V-JEPA 2-AC / 扩散规划器的对应）

| 我们的符号 | 本领域对应 |
|---|---|
| 共享前缀 c_pre | encoder：观测 → latent z_t（V-JEPA：冻结 ViT-g 编码当前帧与目标帧；Cosmos Policy：视频 tokenizer + 前缀） |
| tap point | z_t 算完、MPPI/CEM 开始采样之前（扩散规划器：条件编码完、去噪链开始前） |
| 头（迭代生成） | MPPI 的 I 次迭代（每次 N 条样本 × H 步 latent rollout + Q/reward 评估）；扩散规划器的 T 步去噪；Cosmos Policy 的 N 个候选 chunk 各一次想象 + 价值 |
| 中间状态（warm start 起点） | MPPI 第 i 次迭代后的 (μ_{t:t+H}, σ_{t:t+H})；扩散规划器里部分加噪的轨迹（Diffuser/AOR 的"replan with future"）；Cosmos 的部分去噪 latent |
| tier K = hit | 直接执行库里存的动作序列首动作（或整段 plan 开环执行若干步），跳过全部 rollout；成本 = encoder + kNN |
| 中间 tier | 用库里的 (μ, σ) 或部分去噪轨迹初始化，只跑 ρ_a·I 次迭代 / ρ_a·T 步去噪，并可同时缩样本数 N |
| tier 0 = miss | 完整 I 次迭代（注意：TD-MPC2 的 miss 本身已经用**自己上一步的解**平移做 warm start，这是免费的强基线） |
| key 字段 | z_t（latent，替代相机 token map）、本体状态 / 目标 latent（V-JEPA：goal embedding，替代指令 embedding 做 IVF cell）、任务 id |
| 分数 s_t | z 空间 cosine 或负 L2，按库自比留一估 μ/τ 做 tanh 标准化后加权 |
| 偏差 D_a | tier a 产出的动作序列（只取会被执行的部分）与"从零规划"参考序列的标准化距离；或两者在世界模型下的估计回报差（Acting-upon-Imagination 的 BICHO 口径） |
| 成功度量 SR | Meta-World success rate；DMControl 归一化 return（容差 ε 改成 return 相对损失）；真机任务 SR |
| 成本模型 | c_a = c_enc + a 档剩余迭代数 × (N × H × 单步 rollout + Q 评估)；wall-clock 版需实测（launch-bound） |
| 闭环性 | 每步只执行首动作、下一步重规划的 receding-horizon 回路；缓存 plan 开环执行会漂移（AdaReP 的 stale-plan penalty；AOR 的 error accumulation） |
| 状态门 | AdaReP 的"偏差超过自适应容差才重规划" = 我们的 gate；TD-MPC2 的 shift-warm-start = 单步版 stateful 复用 |

## 3. 打分表 A–G

| 项 | 分 | 依据（一句） | URL |
|---|---|---|---|
| A 结构匹配 | **2** | 子族 (i) 满分：encoder 一次 + MPPI 迭代/去噪链是带中间状态的迭代头，tap point 天然；子族 (ii) 决策时无头可省，得 0；综合 2 | https://arxiv.org/abs/2310.16828 ；https://arxiv.org/abs/2301.04104 |
| B 成本余量 | **2** | FLOPs 上 c_pre/c_0 ≈ 1/9216（TD-MPC2）、< 1%（V-JEPA 2-AC 16 s/action），远优于 VLA 的 15%；但 wall-clock 在 4090 上 launch-bound，policy-only 仅比规划快 6×，hit 地板实测前只能估 ≤ 1/6；"减迭代"傻基线在推理侧砍一半迭代不掉点，砍到 H=1 / policy-only 在高维与多任务上掉 | https://arxiv.org/abs/2605.04568 ；https://arxiv.org/abs/2203.04955 ；https://arxiv.org/abs/2407.02466 ；https://arxiv.org/abs/2506.09985 |
| C 局部性 / 命中潜力 | **2** | 时序局部性有硬证据：AdaReP 在 TD-MPC2/DMControl 30 任务砍 54.5% NFE 不掉分、真机 Franka 砍 >80% 查询；Acting-upon-Imagination 只需 10–14% 步重规划；**跨 episode 库的命中率没有文献数字（不确定）** | https://arxiv.org/abs/2606.23079 ；https://arxiv.org/abs/2105.05716 |
| D 容忍度与闭环 | **3** | 有 SR/return 与容差；缓存 plan 开环执行漂移被三篇独立工作当成核心问题；世界模型想象本身长程失真（"kinematic not dynamic"）进一步放大过期风险 | https://arxiv.org/abs/2606.23079 ；https://arxiv.org/abs/2310.09629 ；https://arxiv.org/abs/2607.05966 |
| E RIT 可用性 | **2** | shadow 可离线构造（记录 z_t、完整规划的 plan/价值，事后检索算 D_a）；AdaReP 的 d_t = ‖z_t − ẑ_t‖ 与 BICHO 的回报差就是 D_a 的在线版；分数单调性与"一个 δ 切所有 cut"未验证；gate 有直接对应物 | https://arxiv.org/abs/2606.23079 ；https://arxiv.org/abs/2105.05716 |
| F novelty 地形 | **1** | 拥挤：AdaReP（2026-06，training-free、缓存 rollout + 偏差触发重规划、挂在 TD-MPC2/VP2/Franka 上）几乎就是 hit + gate；AOR（2023）= 扩散规划器 warm-start + 似然触发；DreamLedger（2026-08）= 想象可信度记账门；Lightning/Memory-of-Motion = kNN 轨迹库 warm start；NEC = kNN 价值缓存。剩下的新东西：跨 episode 库 + 多档 + 单 δ 离线标定 + 按预算反解 | https://arxiv.org/abs/2606.23079 ；https://arxiv.org/abs/2310.09629 ；https://arxiv.org/abs/2608.23863 ；https://goldberg.berkeley.edu/pubs/berenson-ICRA2012-final.pdf ；https://arxiv.org/abs/1907.01474 ；https://arxiv.org/abs/1703.01988 |
| G 我们做得动 | **3** | TD-MPC2 开源 + checkpoint + Meta-World SR + 20 ms/步：Meta-World 一集 200 步 ≈ 4 s，500 集一个前沿点 ≈ 35 min（4090 单卡，估计）；V-JEPA 2-AC 需真机（0 分）；Cosmos Policy 可落 LIBERO 但单卡 ≈ 20 h/点 | https://github.com/nicklashansen/tdmpc2 ；https://arxiv.org/abs/2605.04568 |

合计 **15 / 21**。短板在 F（竞品已经把最直观的一档做掉了）和 B 的 wall-clock 不确定性。

## 4. 该领域的"减迭代次数"傻基线

有人做过，而且结果分两段：

1. **免费区**：TD-MPC(2022) Fig. 6 明说"推理时把迭代数砍一半（相对训练）不掉分"（https://arxiv.org/abs/2203.04955 ）；Horizon Imagination 在 Atari-100k/Craftium 上用一半去噪预算保持控制性能（https://arxiv.org/abs/2602.08032 ）；DISK 在驾驶扩散世界模型上轨迹头 2×、视频头 1.6× 跳步，L2/PDMS 不变（https://arxiv.org/abs/2602.00440 ）。这一段和 VLA 上 k=1…7 平台是同一个现象：迭代冗余先被傻基线吃掉。
2. **悬崖区**：H 从 5 砍到 1 在高维动作空间掉分（TD-MPC 2022）；把规划整个砍掉只用 policy prior，TD-MPC2 在 MT30 上失败（PWM，https://arxiv.org/abs/2407.02466 ）；DIAMOND 去噪步 n=1 在 Boxing 这种未来多峰的游戏里预测发糊、agent 掉分（https://arxiv.org/abs/2405.12399 ）。
3. 与 VLA 的关键区别：VLA 的 k=1 = 条件均值，在单峰任务上无损；而**采样式规划的"k=0" = policy prior，是一个明确更差的分布**（TD-MPC 2022："πθ generally performs worse than planning"），所以本领域的傻基线曲线有真实的 SR 落差，warm 档（从库里真实规划过的 (μ,σ) 起、少跑迭代）有 counterfactual。这是 B 能拿 2 分的主要原因。
4. 但要警惕另一个傻基线：**Dream-MPC 用梯度规划 15 次评估追平 9216 次采样规划**（https://arxiv.org/abs/2605.04568 ）。若审稿人拿它当"全算"，我们的 miss 成本就缩了 600×（FLOPs），hit 省的只剩 encoder 以外的一点。wall-clock 上它只快 13%（launch-bound），所以现实成本模型必须按 wall-clock 或按"边缘 CPU 部署"来立。

## 5. 预期收益（与 VLA 对比）

- **可恢复冗余的下界已被竞品给出**：AdaReP 在 TD-MPC2/DMControl 30 任务上砍 54.5% NFE、50.3% wall-clock，"closely matching" 逐步重规划的均分；Franka 真机 36/50 vs 34/50 且查询 −80%（https://arxiv.org/abs/2606.23079 ）。换成我们的口径：仅靠**自身上一条 plan 的复用**（无跨 episode 库、无 warm 档）就有 IR ≈ 0.45–0.5、ε ≈ 0。对比 VLA：π0.5 libero_10 IR 60 时 SR 已掉 9 pp。**这个领域的冗余量级确实比 VLA 大一个档**。
- 跨 episode 库能再压多少：无文献数字，**不确定**。理由上可期待的增量集中在 (a) episode 起点（自身无 plan 可复用）、(b) 自身 plan 过期但库里有邻近状态的成熟 plan、(c) warm 档把"过期一点"的 plan 用 1–2 次迭代修好而不是全算 6 次。若 (b)(c) 各贡献 10–15 个百分点，IR 能到 ~0.3（估计，需实验）。
- hit 成本地板：FLOPs 上 ≈ 0；wall-clock 在 4090 上估计 2–4 ms / 20.83 ms ≈ 10–20%（**实测前不可信**）；在 CPU/边缘部署或 317M 大模型上 FLOPs 主导，地板才会显著低于 VLA 的 15%。
- SR 损失：AdaReP/AuI 显示自身复用可做到 ε≈0；库复用的额外损失来自跨 episode 状态失配，需 RIT 控制，数字未知。

## 6. novelty 地形与最强反驳

已有工作按与我们形式的距离排序：

1. **AdaReP**（2026-06）：training-free、缓存 rollout、偏差 d_t=‖z_t−ẑ_t‖ + 局部敏感度估计 → 自适应容差 → 触发重规划；挂在 TD-MPC2 / VP2 / Franka。= 我们的 hit（自身 plan）+ gate。没有跨 episode 库、没有 warm 档、没有离线标定。https://arxiv.org/abs/2606.23079
2. **Acting upon Imagination**（2021）：四种"何时信任想象轨迹、跳过重规划"判据（FSA/CB/FUT/BICHO），Cartpole 只需 10% 重规划。同样是自身 plan 复用。https://arxiv.org/abs/2105.05716
3. **Adaptive Online Replanning with Diffusion Models**（NeurIPS 2023）：用扩散模型对现有 plan 的似然决定是否重规划，"replan with future"从旧 plan 部分加噪起、N=80 步追平从零 256 步。= warm 档（自身 plan）+ 似然门。https://arxiv.org/abs/2310.09629
4. **DreamLedger**（2026-08）：按条件/区域/horizon 记账想象的历史可信度，信用不足就拒绝想象（缩 horizon 或触发观测）。挂在 DreamerV3/TD-MPC2/V-JEPA 2-AC。与我们 RIT 的"离线拟合风险曲线"最像。https://arxiv.org/abs/2608.23863
5. **Lightning / Thunder / Memory of Motion**：kNN 轨迹库给优化式规划器做 warm start，是经典的"跨 episode 库 + warm"。https://goldberg.berkeley.edu/pubs/berenson-ICRA2012-final.pdf ；https://arxiv.org/abs/1410.1950 ；https://arxiv.org/abs/1907.01474
6. **Episodic control（NEC）/ SoRB / Retrieval-Augmented RL**：kNN 键值缓存回报、在 replay buffer 上做图搜索、检索过去轨迹进策略。是"库"这个概念的出处，但都是学习型而非 training-free 推理加速。https://arxiv.org/abs/1703.01988 ；https://arxiv.org/abs/1906.05253 ；https://arxiv.org/abs/2202.08417
7. 头内加速：Horizon Imagination、DISK、Dream-MPC——直接把头做便宜，和我们正交但会缩小 miss 成本。

**最强反驳**（审稿人会怎么打）：
- "AdaReP + AOR 已经把 hit/warm/gate 三件事在这个领域做完了，你们只是加了个跨 episode 库和一个离线标定；请证明库比自身上一条 plan 多省多少。" 这是必须在第一个实验里正面回答的问题（见 §7 的 kill 条件）。
- "TD-MPC2 的 miss 本来就用上一步解 warm start，你们的 warm 档相对这个免费 warm start 的增量是什么？"
- "20 ms/步的规划器不需要省；真正贵的（V-JEPA 2-AC 16 s、Cosmos 4 min）你们没跑闭环。"
- "Dream-MPC 15 次评估追平 9216 次，你们的成本模型按谁算？"

我们还剩的新东西：跨 episode 库（AdaReP 没有）+ 多档同时存在（他们二元）+ **RIT 的单 δ 离线标定与按预算反解**（他们在线调 ε₀/α 超参、"tuned to match task success rates"）+ 把 R_C(ε)/V_C(B) 的形式化搬过来。这些够写一节，不够独立成篇——除非库带来的增量在 IR 上显著（比如 0.45 → 0.3）。

## 7. 首个实验方案（≤ 2 周，TD-MPC2 × Meta-World，单张 4090）

第一步永远是傻基线 + 成本拆分实测。

**第 1–3 天：成本拆分与傻基线**
- 取 TD-MPC2 官方 5M 单任务 checkpoint，Meta-World 10 个任务（有 SR）+ DMControl 3 个（return），每点 50 集。
- 实测 wall-clock：encoder 单独、kNN 查库、每次 MPPI 迭代、policy-only；同时记 FLOPs 口径。得出 c_pre/c_0 两个版本。
- 傻基线 ladder：iterations 6→3→2→1→0(policy prior)；samples 512→128→32；H 3→1。画 (IR, SR)。
- 自身复用基线（复现 AdaReP 的 MPC^m_k 固定间隔与固定阈值版）：每 m 步重规划、其余开环执行旧 plan，m=1,2,4,8；再加一个 d_t 阈值触发版。这条曲线是我们必须超过的线。

**第 4–7 天：库 + shadow + RIT**
- collect：每任务用完整规划跑 200 集，成功集入库；条目 = (z_t, 动作序列 μ_{t:t+H}, σ, 首动作, 估计价值)。
- shadow：另跑 50 集/任务，记录每步 z_t 与从零规划的参考动作序列；事后在库里检索（z 的 cosine / 负 L2，按任务 id 分 cell），算 D_hit = ‖a_lib − a_ref‖（只取首动作或前 k 步），D_warm(1 iter) / D_warm(2 iter) = 从 (μ_lib, σ_lib) 起跑 1/2 次迭代得到的动作与 a_ref 的距离。
- 检查：q_a(s) 是否随 s 单调、各档曲线是否有序、上一步分数对下一步是否命中的 AUROC（gate 的前提）。任一不成立就记录并停。

**第 8–12 天：闭环前沿**
- K=1,2,3 各扫 δ 6–8 个点，每点 50 集 × 10 任务 ≈ 35 min，总计约 1 天机时。
- 对照：傻基线 ladder、自身复用基线、AdaReP 简化版。
- 报 (IR_wall, IR_flops, SR)、hit/warm/miss 占比、episode 内命中成串统计。

**Kill 条件（任一触发即写 NO）**
- 自身复用基线在同 SR 下的 IR 与库前沿差距 < 5 个百分点（库不值一篇）。
- wall-clock hit 地板 > 30%，且 FLOPs 口径讲不通（无 CPU/大模型部署故事）。
- q_a(s) 不单调（RIT 在此领域不成立）。

**备选平台**：若两周结果为正，第二阶段用 Cosmos Policy（LIBERO，我们已有 harness，单卡 ≈ 20 h/点）验证在"贵的世界模型"上的绝对节省；V-JEPA 2-AC 只作成本论证引用，不跑。

## 8. 结论

**MAYBE（偏 GO，限 TD-MPC2/Meta-World 子族）**：结构与闭环性比 VLA 匹配得更好、可恢复冗余量级更大（竞品已示 IR≈0.5 零损失），但 AdaReP/AOR/DreamLedger 已经占住了 hit+gate+warm 的直观版本，能否成篇取决于两周实验里"跨 episode 库 + RIT 多档"相对"自身上一条 plan 复用"能否再压 ≥ 10–15 个百分点 IR，以及 wall-clock 的 hit 地板是否真的远低于 VLA 的 15%。
