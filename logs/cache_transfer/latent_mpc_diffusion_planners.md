# 跨领域迁移调研：扩散/流匹配轨迹规划器 + MPC、推理时搜索型策略（机器人操作，2025-01 之后）

> 调研日期 2026-09-16。依据 `logs/cache_transfer/briefing.md` §3 打分表 A–G 与 §4 报告格式；世界模型子族（TD-MPC2 / V-JEPA 2-AC / Cosmos Policy / AdaReP / DreamLedger）已在 `world_models_planning.md` 覆盖，本文不重复。
> 范围：(1) Diffuser 后继与扩散 MPC；(2) test-time scaling / best-of-N with verifier / MCTS over action chunks 的 VLA 或策略；(3) DDPM 型 Diffusion Policy 大步数变体的公开权重。筛选偏好：公开权重 + 仿真 benchmark、每次决策 ≥ 0.3 s、成本在可 warm-start 的迭代循环里、文献里有"砍迭代会掉点"的证据。
> 数字一律给 URL；文献没有的标"估计"或"不确定"，不编。

## 0. 一页结论

| 排名 | 候选 | 每决策耗时 | 循环体占比 | 减预算 counterfactual | 权重 | A–G 合计 |
|---|---|---|---|---|---|---|
| **1** | **Diffusion Policy（DDPM-100 / DDIM-10）× robomimic / PushT，Columbia 官方权重** | 0.66–0.82 s（DDPM-100，A100）；DDIM-10 66 ms | ≥ 95%（UNet 去噪链） | DDIM-1 全任务 0%；DDIM-2 均值 0.64、PushT 0.29（满步 0.94） | 有（MIT） | **16 / 21** |
| **2** | **MCTD / Fast-MCTD × OGBench cube-single/double/triple** | MCTD-Replan 9.2 / 38.8 / 102 s；Fast-MCTD 3.0 / 5.9 / 9.1 s（8×4090）每次重规划 | ≈ 100%（树搜索 × 部分去噪） | 不搜索的 Diffuser：cube-double 12%、triple 8%（MCTD 78% / 40%） | 迷宫有、cube **不确定** | **15 / 21** |
| **3** | **VLA test-time verification（RoboMonkey；同族 CoVer-VLA / MG-Select / RoVer）× LIBERO / SIMPLER** | 0.65 s（16 候选，H100） | 60–75%（采样 + 7B verifier） | N=1 → N=16：LIBERO-Long 49.8 → 56.5；RoboCasa N=1 27.6 → N=4 31.0 | 有（OpenVLA + 7B verifier） | **14 / 21** |
| 4 | VLAPS / V-VLAPS（仿真器内 MCTS，Octo-93M × LIBERO） | 600–1800 s / 集（≈ 10–25 s / 决策，估计） | ≈ 100% | 不搜索 60.2% → 搜索 87.4%（600 s）→ 91.6%（1800 s） | 代码 MIT；Octo-LIBERO 权重不确定 | 14 / 21 |

一句话：**这条线的结构匹配比 VLA 好一档（hit 成本地板 0.5–5% vs VLA 15%），"砍迭代会崩"的 counterfactual 在 DP 的 1–2 步端与 MCTD 的搜索端都有硬数字；但 DP 端的 warm-start 已被 RTI-DP / Falcon / STEP / WarmPrior 四篇 2025–2026 工作占住（都是"自身上一条 chunk"的 warm start），我们剩下的新东西仍是跨 episode 库 + 多档 + RIT 单 δ 离线标定 + 状态门，与世界模型线的结论同构。** 三天杀手门必须直接回答"跨 episode 库能否打过自身上一条 chunk 的 warm start"。

## 1. 候选系统卡片

### 1.1 Diffusion Policy（DDPM-100）与其减步/热启动变体 —— robomimic / PushT

| 字段 | 内容 | URL |
|---|---|---|
| 名称 / 日期 / venue | Diffusion Policy（Chi et al.，RSS 2023，IJRR 2024/2025 扩展版）；权重 2023-03 发布，仍是 2025–2026 所有减步工作的公共基线 | https://github.com/real-stanford/diffusion_policy ；https://journals.sagepub.com/doi/10.1177/02783649241273668 |
| 架构 / 参数量 | DiffusionUnetHybridImagePolicy：ResNet-18 视觉编码（84×84 crop，n_obs_steps=2）+ 1D 条件 UNet 去噪头；UNet 约 256M（STEP 文给 256M） | https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml ；https://arxiv.org/html/2602.08245 |
| 推理算法 / 预算 | 官方 config：DDPMScheduler，num_train_timesteps=100，**num_inference_steps=100**，horizon 16，n_action_steps 8（每 8 步决策一次） | 同上 config.yaml |
| 每次决策耗时 | DDPM-100：660 ms（OneDP Table 5）、~650–750 ms（STEP）、~820 ms（RTI-DP，A100-SXM4-40GB）；DDIM-10：66 ms；单步 7 ms | https://arxiv.org/html/2410.21257 ；https://arxiv.org/html/2602.08245 ；https://arxiv.org/html/2508.05396 |
| 循环体占比 | 头 = 100 × UNet 前向 ≈ 6.6 ms/步；前缀 ResNet-18 ×2 相机 batch=1 在 H100 上 **估计 2–5 ms** ⇒ c_pre/c_0 ≈ 0.5–1%（DDPM-100）或 ≈ 5%（DDIM-10）。**需实测** | 由上行延迟换算（估计） |
| Benchmark / SR（含减预算对照） | OneDP Table 1（robomimic image + PushT，DDPM 训练）：DP-100 均值 0.829，DDIM-10 0.836，**DDIM-1 全任务 0.000**；STEP：DDPM-100 robomimic 均值 ≈ 0.94 / PushT 0.94，**DDIM-2 ≈ 0.64 / 0.29**，DDIM-4 ≈ 0.94 / 0.73；Two-Steps GDP（Adroit，DDPM）：100 步 0.68–0.88，5 步 0.85–1.00，**2 步 0.00–0.13** | https://arxiv.org/html/2410.21257 ；https://arxiv.org/html/2602.08245 ；https://arxiv.org/html/2510.21991 |
| 权重 | Columbia 服务器公开全部训练日志与 ckpt：image/{can,lift,square,transport}_{ph,mh}、tool_hang_ph、pusht；low_dim 同结构；例 `low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt` | https://diffusion-policy.cs.columbia.edu/data/experiments/image/ ；README 下载说明 https://github.com/real-stanford/diffusion_policy/blob/main/README.md |
| 其它公开权重 | lerobot/diffusion_pusht（Apache-2.0，500 集 SR 65.4%）；ManiSkill 官方 DP baseline 代码有、结果标 WIP、未见托管权重；**LIBERO 官方 DP 权重未找到（不确定）** | https://huggingface.co/lerobot/diffusion_pusht ；https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/baselines.html ；https://github.com/huggingface/lerobot/issues/2894 |
| 许可证 | 代码 MIT；权重随代码发布，未单独声明（按 MIT 处理，不确定） | https://github.com/real-stanford/diffusion_policy |
| 硬件可行性 | 4090-48G / A5000 均可（UNet 256M，batch=1）；robomimic 仿真 CPU/EGL；50 集一个点 = 分钟级 | — |
| 已有复用 / 缓存 / 热启动 | **拥挤**：RTI-DP（IROS 2025，training-free，上一 chunk 平移做初值，3 步 ≈ 满步，25–145 ms）；Falcon（2025-03，training-free，缓冲区里挑部分去噪的历史动作、Tweedie 一步估计打分，DDPM 100 NFE → 12.9–46.9 NFE，2–7×，含 RDT-1B）；STEP（2026-02，0.98M 预测器给 warm start，2 步 ≈ 0.91）；WarmPrior（2026-05，训练时把源分布锚到上一 chunk，NFE=1 时 Square-MH 65.9 → 77.8，含 GR00T N1.5）；Action-Prior Denoising（2026-05，训练时模拟延迟）；OneDP / Consistency Policy / GDP（蒸馏或群体选择，非复用） | https://arxiv.org/html/2508.05396 ；https://arxiv.org/html/2503.00339 ；https://arxiv.org/html/2602.08245 ；https://arxiv.org/html/2605.13959 ；https://arxiv.org/abs/2605.25537 ；https://arxiv.org/html/2410.21257 |

打分（A–G，依据见 §3）：A 3 / B 2 / C 2 / D 3 / E 2 / F 1 / G 3 = **16**。

### 1.2 MCTD / Fast-MCTD —— Diffuser 后继里唯一把"推理时算力 → 成功率"曲线画出来的规划器

| 字段 | 内容 | URL |
|---|---|---|
| 名称 / 日期 / venue | Monte Carlo Tree Diffusion（Yoon, Cho, Baek, Bengio, Ahn；ICML 2025 Spotlight）；Fast-MCTD（NeurIPS 2025 Spotlight）；Compositional MCTD（2025-10） | https://arxiv.org/abs/2502.07202 ；https://arxiv.org/abs/2506.09498 ；https://arxiv.org/abs/2510.21361 |
| 架构 / 参数量 | Diffusion Forcing 骨干，Transformer：hidden 128、12 层、4 头、FF 512（参数量论文未给，**按配置估计 < 10M**）；输入为状态 + 目标（OGBench 状态版），视觉版用 8 维 latent | https://arxiv.org/html/2502.07202 |
| 推理算法 / 预算 | 去噪链树化：每个节点 = 部分去噪的计划（partial denoising 20 步、jumpy 间隔 10），meta-action = guidance 等级，MCTS 最多 500 次迭代，horizon 500（cube 多物 500、单物 200），**每 50 步重规划**；Fast-MCTD 加并行 200 rollouts + 轨迹稀疏化 H=5 | https://arxiv.org/html/2502.07202 ；https://arxiv.org/html/2506.09498v4 |
| 每次决策耗时（每次重规划） | cube single / double / triple：Diffuser 6.3 / 6.4 / 6.5 s；Diffusion Forcing 2.9 / 15.2 / 15.9 s；**MCTD-Replan 9.2 / 38.8 / 102.0 s**；Fast-MCTD-Replan 3.0 / 5.9 / 9.1 s（8×RTX 4090）；迷宫 giant：MCTD 264 s → Fast 2.4 s | https://arxiv.org/html/2506.09498v4 |
| 循环体占比 | 前缀几乎为零（状态输入）⇒ c_pre/c_0 ≈ 0；成本全在搜索迭代 × 部分去噪 | 同上 |
| Benchmark / SR（含减预算对照） | OGBench cube single / double / triple：Diffuser（无搜索）78 / **12 / 8**；Diffusion Forcing 100 / 18 / 16；MCTD-Replan 100 / **78 / 40**；Fast-MCTD-Replan 100 / 77 / 50。迷宫 pointmaze-large：MCTD 98 vs Diffuser 44；Fig. 6：giant 迷宫上 MCTD 随去噪预算上升趋于满分，Diffuser-Random-Search 几乎不涨 | https://arxiv.org/html/2506.09498v4 ；https://arxiv.org/html/2502.07202 |
| 权重 / 代码 | 官方 ahn-ml/mctd（含 MCTD 与 Fast-MCTD），Google Drive 提供 `planner_trained_models.tar.gz` 与 `dql_trained_models.tar.gz`；README 只列 pointmaze / antmaze，**cube 权重与脚本是否包含：不确定**；Docker + MuJoCo 2.1 + 定制 OGBench（迷宫改为确定性起终点） | https://github.com/ahn-ml/mctd |
| 许可证 | 仓库有 LICENSE 文件，类型未在 README 说明（不确定） | 同上 |
| 硬件可行性 | 模型极小，1×H100 能跑 200 并行 rollouts；但论文时间是 8×4090，单卡 wall-clock **未知**；cube-triple MCTD 102 s/重规划 × 每集 ~10 次重规划（H=500、每 50 步一次，估计）× 50 集 ≈ 14 h/点（估计），Fast-MCTD ≈ 1.3 h/点（估计） | 由上表换算 |
| 已有复用 / 缓存 / 热启动 | Fast-MCTD 的 "redundancy-aware selection" 只在一棵树内去重；C-MCTD 组合子计划；无跨 episode 计划库；Diffusion Forcing 仓库自认 MCTG 未复现 | https://arxiv.org/abs/2506.09498 ；https://github.com/buoyancy99/diffusion-forcing |

打分：A 2 / B 3 / C 2 / D 2 / E 2 / F 2 / G 2 = **15**。

### 1.3 VLA test-time sampling + verifier（RoboMonkey 为主；CoVer-VLA / MG-Select / RoVer / SVA 同族）

| 字段 | RoboMonkey | CoVer-VLA | MG-Select | RoVer | SVA（Look Before You Leap） |
|---|---|---|---|---|---|
| 日期 / venue | 2025-06，CoRL 2025 | 2026-02，ECCV 2026 | 2025-10 | 2025-10 | 2026-07 |
| 架构 | OpenVLA-7B + **7B LLaVA verifier**（ViT-L + reward head） | π0 / π0.5 + **1B 对比 verifier ×3 集成**（SigLIP2 冻结） | π0-FAST / OpenVLA / MiniVLA，**无 verifier**（条件 vs 遮蔽条件分布的 KL） | GR-1 / Dita / MoDE / DP + **0.2B PRM**（可训练 40M） | OpenVLA / π0 / π0.5 + Qwen3.5-0.8B 蒸馏 Q（5 头 MLP，2.6M） |
| 推理预算 | 策略采 N̂=5 + 高斯扰动 K̂=16（仿真 8/16/32） | 8 次改写指令 × 5 动作 = 40 候选 | N ∈ {1,4,8,64} | K = N（1–10）+ M 方向引导扩展 | N=32 候选，MCTS 仅在训练期仿真器内 |
| 每决策耗时 | **16 候选 650 ms（1.5 Hz，1×H100）**；SGLang bs16 0.72 s vs naive 2.4 s | π0.5 445 ms + verifier 8 ms = **453 ms（RTX 5090，bs16）** | 抽取到 N=1 20.2 s、N=4 43.4 s（single-prefill 23.7 s）——**单位疑有误，只取相对倍数**：N=4 vanilla ≈ 2.1×、prefix 共享 ≈ 1.17× | perception cache 后每动作 5.7–6.2 ms（10 动作 0.414 → 0.093 s） | 生成最高 11.2 s @ N=32，评估 ≤ 1.33 s |
| Benchmark / SR（减预算对照） | SIMPLER 38.5 → 46.3；**LIBERO-Long 49.8 → 56.5**；真机 OOD 35 → 60；动作 RMSE 随 N 幂律下降（10k 样本 −59.3%） | SIMPLER ID +22 / OOD +13（相对 policy scaling）；PolaRiS SR +9.3；真机 +45 | RoboCasa（100 demo）**N=1 27.6 → N=4 31.0 → N=64 33.3**；LIBERO 92.0 → 93.1（π0-FAST） | CALVIN ABC→D SR@5：GR-1 43.4 → 48.7，Dita 50.0 → 59.2，MoDE 63.5 → 66.6 | SimplerEnv π0 38.5 → 50.7；RoboTwin π0.5 36.0 → 43.5；**pass@1 33% → pass@32 92%** |
| 权重 / 代码 | 代码 + verifier HF（monkey-verifier-7b）+ 偏好数据集 | 项目页 cover-vla.github.io，代码/权重**不确定** | 无专门仓库（不确定） | "acceptance 后开源"，**未放** | 未放 |
| 许可证 | 论文 CC BY 4.0；代码许可未见 | CC BY 4.0（论文） | — | — | — |
| URL | https://arxiv.org/html/2506.17811v1 ；https://github.com/robomonkey-vla/RoboMonkey ；https://huggingface.co/robomonkey-vla/monkey-verifier-7b | https://arxiv.org/html/2602.12281v1 | https://arxiv.org/html/2510.05681 | https://arxiv.org/html/2510.10975 | https://arxiv.org/html/2607.03751 |

循环体占比：RoboMonkey 里 VLA 采样 5 条 + verifier 打分 16 条，前缀（VLA 视觉 + LLM prefill）**估计 25–35%**，hit 地板不优于 VLA 线；CoVer 更极端：verifier 只占 8/453 ms，成本几乎全是 π0.5 的 40 候选生成，此时"库存候选集 + 只跑 verifier"这一档能省 98%。
已有复用机制：RoVer 的 **perception cache**（候选间共享感知前缀，4.5–7.2×）与 MG-Select 的 **single-prefill**（N 候选共享 prefill）都是"前缀共享"而非跨决策缓存；无跨 episode 库、无 warm 档、无离线标定。
硬件可行性：OpenVLA-7B + 7B verifier bf16 ≈ 30 GB，H100 / 4090-48G 都放得下；LIBERO harness 现成；500 集 × ~300 决策 × 0.65 s ≈ 27 h/点（单流，估计），须并发多 worker 摊薄。

打分：A 2 / B 2 / C 2 / D 3 / E 2 / F 1 / G 2 = **14**。

### 1.4 仿真器 / 世界模型内 MCTS over action chunks（VLAPS、V-VLAPS、VLA-Reasoner、WorldPlanner）

| 字段 | VLAPS | V-VLAPS | VLA-Reasoner | WorldPlanner |
|---|---|---|---|---|
| 日期 | 2025-08 | 2026-01 | 2025-09（ICRA 2026） | 2025-11 |
| 架构 | Octo-base-1.5（93M）微调 LIBERO + **LIBERO 仿真器**作模型 | 同上 + 2.4M 价值 MLP + **2000 medoid 动作 chunk 库** | OpenVLA-7B / Octo-Small / SpatialVLA / π0-FAST + **iVideoGPT 600M 世界模型** | 动作条件视觉世界模型 + 扩散动作采样器 + reward 模型 |
| 预算 | 每节点 300 采样、扩展 10、深度 100、rollout 300 步、**600 s wall-clock / 集** | 600 s 或 1800 s / 集 | top-k 扩展 + UCB，预算未给 | MCTS + 零阶 MPC |
| SR | LIBERO 整体 +42 pts（50k ckpt），Libero-Object 6 → 73 | VLA-only 60.2 → VLAPS 87.4（600 s）→ V-VLAPS 91.6（1800 s） | LIBERO 均值 76.0 → 81.0；SimplerEnv Octo 26.5 → 37.3；真机 π0-FAST 64 → 74 | 3 个真机任务 |
| 耗时 | 每 100 集一个 suite 3.5–4.5 h（H100） | 同左 | 未报 | 未报 |
| 代码 / 权重 | MIT，github.com/cyrusneary/vlaps；Octo-LIBERO ckpt 需自备（**不确定是否发布**） | 未给 | 项目页有，仓库/权重未确认 | 未给 |
| URL | https://arxiv.org/html/2508.12211 ；https://github.com/cyrusneary/vlaps | https://arxiv.org/html/2601.00969 | https://arxiv.org/html/2509.22643 ；https://vla-reasoner.github.io/ | https://arxiv.org/abs/2511.03077 |

打分（VLAPS/V-VLAPS）：A 2 / B 3 / C 2 / D 2 / E 2 / F 2 / G 1 = **14**。G 低因每点 ≥ 17 h 且 JAX+PyTorch 双栈；D 打 2 因"仿真器在回路里"不是可部署策略，审稿人会说这是在加速 MCTS 而非策略推理。

### 1.5 其它已查、不进前四（一行一条，说明为何不选）

| 系统 | 一句话 | 不选理由 | URL |
|---|---|---|---|
| D-MPC（DeepMind，2024-10） | 两个扩散模型（动作提议 + 动力学）做采样式 MPC，D4RL | 2025 前、无代码、locomotion | https://arxiv.org/abs/2410.05364 |
| DiffuserLite（NeurIPS 2024） | 粗到细 planning refinement，122 Hz，D4RL / robomimic | 头已做便宜，是"miss 成本缩水"的反例 | https://arxiv.org/abs/2401.15443 |
| CompDiffuser（NeurIPS 2025 Spotlight） | 分块双向扩散拼接长程轨迹，OGBench 迷宫，代码 + 权重 | 迷宫而非操作；无推理预算曲线 | https://github.com/devinluo27/comp_diffuser_release |
| MPDiffuser（2025-12） | 规划器与动力学扩散交替采样 + 排序 | D4RL/DSRL + 四足，无操作 | https://arxiv.org/abs/2512.08280 |
| Generative Models From and For Sampling-Based MPC（2025-10） | flow-matching 提议分布给 MPPI，四足 loco-manipulation | 无公开代码、非操作臂 | https://arxiv.org/abs/2510.14643 |
| ProxPI（2026-09） | 学习先验失配时的 MPPI 近端注入 | 方法论文，无我们要的成本模型 | https://arxiv.org/abs/2609.00941 |
| Warm-Starting Collision-Free MPC with Object-Centric Diffusion（2026-01） | 扩散轨迹先验 → 约束 MPC 求解器投影 | 是"扩散给求解器 warm start"，与我们方向相反（我们给扩散 warm start） | https://arxiv.org/abs/2601.02873 |
| Warm Starts Accelerate Conditional Diffusion（2025-07） | 确定性模型预测信息先验矩，加速条件扩散 | 通用生成，非机器人闭环 | https://arxiv.org/abs/2507.09212 |
| FlowMP（IROS 2025） | 二阶流匹配运动场，快 100× | 运动规划器非策略，无 SR 型闭环 | https://arxiv.org/abs/2503.06135 ；https://github.com/mkhangg/flow_mp |
| DynaGuide（NeurIPS 2025） | 外部动力学模型对现成 DP 做 guidance，CALVIN 70% steering | 是引导不是省算力；每步多一次动力学梯度 | https://arxiv.org/abs/2506.13922 ；https://github.com/MaxDu17/DynaGuide |
| QPILOTS（2026-06） | 流策略的 test-time Q-steering（投影到干净动作再取梯度） | 引导型，成本增量未报 | https://arxiv.org/abs/2606.14801 |
| General Policy Composition（ICLR 2026） | 多策略分数凸组合 + test-time search，robomimic/PushT/RoboTwin | 成本乘数未报；组合不是复用 | https://arxiv.org/abs/2510.01068 |
| Two-Steps DP via Genetic Denoising（2025-10） | 去噪链内群体选择，2 步达满步；RTX 3080 上 3.8 ms/NFE（状态版） | 无代码（公司政策） | https://arxiv.org/html/2510.21991 |
| Inference-time Scaling through Classical Search（2025-05） | BFS/DFS 树 + 退火 Langevin，PointMaze Pareto、D4RL 86.1 | 迷宫 + 图像，无操作 SR | https://arxiv.org/html/2505.23614v2 |
| E-TTS（ECCV 2026） | 推理-动作联合采样 + 历史缓冲 verifier，仿真 +33.14% | 数字为相对增益，成本未在摘要 | https://arxiv.org/abs/2606.27268 |
| DREAM-Chunk（2026-06） | latent 世界模型 rollout 多候选 chunk、按观测匹配选 | Kinetix 为主，无权重 | https://arxiv.org/abs/2606.18589 |
| ITPO with Differentiable World Models（2026-03） | 通过可微世界模型梯度优化动作序列 | D4RL locomotion/AntMaze | https://arxiv.org/abs/2603.22430 |
| Latent Diffusion Planning（2025-04） | VAE latent 里的扩散规划 + IDM | 摘要无成本拆分；未查到步数与权重 | https://arxiv.org/abs/2504.16925 |
| Diffusion Forcing（NeurIPS 2024） | 逐 token 噪声等级，MCTG；MIT，迷宫 ckpt | 2025 前；MCTG 官方未复现 | https://github.com/buoyancy99/diffusion-forcing |

## 2. 映射表（以 DP-DDPM 为主，括号给 MCTD / verifier-TTS 的对应）

| 我们的符号 | 本领域对应 |
|---|---|
| 共享前缀 c_pre | ResNet-18 ×相机 + 本体状态拼接（MCTD：无，状态直接进；RoboMonkey：VLA 视觉 + LLM prefill，跨候选共享 = MG-Select single-prefill / RoVer perception cache） |
| tap point | obs 特征向量算完、UNet 去噪链开始前（MCTD：根节点部分去噪前；TTS：prefill 完、采样前） |
| 头（迭代生成） | K=100 步 DDPM（或 10 步 DDIM）UNet 去噪（MCTD：≤500 次树迭代 × 20 步部分去噪；TTS：N 候选生成 + N 次 verifier） |
| 中间状态（warm start 起点） | 库条目存的部分去噪 chunk x_t（Falcon 的 latent buffer、RTI-DP 的平移 chunk 是"自身版"）（MCTD：部分去噪的根计划 / 子树；TTS：库存候选集 + verifier 分数） |
| tier K = hit | 直接回放库 chunk（8 个动作），成本 = 前缀 + kNN（MCTD：执行库计划 50 步；TTS：执行库里 verifier 选中的动作） |
| 中间 tier | 从库 x_t 起跑 ρ_a·K 步（Falcon 证明 DDPM 从历史部分去噪起只需 12.9–46.9 NFE）（MCTD：以库计划为根、跑 ρ_a·500 次迭代；TTS：跳过采样只跑 verifier，或缩 N） |
| tier 0 = miss | 从高斯噪声全算 K 步（注意 RTI-DP / Falcon 已把"自身上一 chunk 做初值"当默认，是免费强基线） |
| key 字段 | ResNet 特征（替代相机 token map）、本体状态、任务 id（robomimic 单任务 ⇒ 一个 cell；LIBERO/RoboCasa 用指令） |
| 分数 s_t | 特征 cosine + 本体负 L2，库自比留一估 μ/τ，tanh 标准化后加权 |
| 偏差 D_a | tier a 产出 chunk 与从噪声全算参考 chunk 的标准化距离（只取会执行的 8 步）（TTS：verifier 分数差 = 天然 D_a；MCTD：计划的价值差或首 50 步距离） |
| 成功度量 SR | robomimic / PushT / LIBERO / OGBench 成功率；容差 ε 同 VLA |
| 成本模型 | c_a = c_pre + 剩余步数 × 单步 UNet；DDPM-100：单步 6.6 ms，前缀估计 2–5 ms（TTS：c_pre + N × 生成 + N × verifier） |
| 闭环性 | receding horizon（8 步执行后重算）；RTI-DP 自述 warm start 在"突变 / 模式切换"处失效、Falcon 用最小噪声阈值防动作重复 ⇒ 连续复用会漂移 |
| 状态门 | Falcon 的 buffer 优先队列 + 噪声阈值是"何时信历史"的单步版；MCTD-Replan 每 50 步强制重规划 = 我们的 L 步强制 miss |

## 3. 打分表 A–G（三个主候选并列）

| 项 | DP-DDPM × robomimic | MCTD × OGBench cube | RoboMonkey/CoVer × LIBERO/SIMPLER | 依据 |
|---|---|---|---|---|
| A 结构匹配 | **3** | 2 | 2 | DP 完全同构且中间状态显式；MCTD 无昂贵前缀但树节点即中间状态；TTS 候选集是离散样本，warm 起点不天然（π0/π0.5 版可兼有流匹配中间态） https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml ；https://arxiv.org/html/2502.07202 ；https://arxiv.org/html/2602.12281v1 |
| B 成本余量 | 2 | **3** | 2 | DP：hit 地板 ≈ 1%（DDPM-100）/ 5%（DDIM-10），但 DDIM-4…10 无损、只在 1–2 步崩（DDIM-1 = 0%，DDIM-2 = 0.64 / PushT 0.29）；MCTD：不搜索 12% / 8% vs 78% / 40%，成本 s 级；TTS：N=1 → N=16 只差 5–7 pp，且 hit 地板 25–35% https://arxiv.org/html/2410.21257 ；https://arxiv.org/html/2602.08245 ；https://arxiv.org/html/2506.09498v4 ；https://arxiv.org/html/2506.17811v1 |
| C 局部性 | 2 | 2 | 2 | 时序局部性硬证据：Falcon 2–7×、RTI-DP 3 步；跨 episode 命中率**三者都无文献数字**；OGBench cube 目标集固定 5 个 + 初始位姿随机；LIBERO 命中率我们自己有 https://arxiv.org/html/2503.00339 ；https://arxiv.org/html/2508.05396 |
| D 容忍度与闭环 | **3** | 2 | **3** | DP/TTS 有 SR、receding horizon、漂移证据（RTI-DP 突变失效）；MCTD 每 50 步开环执行，闭环粒度粗 https://arxiv.org/html/2508.05396 ；https://arxiv.org/html/2506.09498v4 |
| E RIT 可用性 | 2 | 2 | 2 | 三者 shadow 都可离线构造；TTS 的 verifier 分数是现成 D_a；单调性与单 δ 未验证 |
| F novelty | **1** | 2 | 1 | DP：RTI-DP / Falcon / STEP / WarmPrior / Action-Prior / BRIDGER / Streaming DP 全是 warm start；TTS：RoVer perception cache、MG-Select single-prefill、verifier-as-gate 显而易见；MCTD 线只有树内去重 https://arxiv.org/html/2508.05396 ；https://arxiv.org/html/2503.00339 ；https://arxiv.org/html/2605.13959 ；https://arxiv.org/html/2510.10975 ；https://arxiv.org/html/2510.05681 |
| G 做得动 | **3** | 2 | 2 | DP：官方 ckpt + 分钟级评测 + 4090 即可；MCTD：ckpt 覆盖 cube 不确定、单卡时间未知、Docker；TTS：权重全有、LIBERO harness 现成，但 27 h/点 https://github.com/real-stanford/diffusion_policy ；https://github.com/ahn-ml/mctd ；https://github.com/robomonkey-vla/RoboMonkey |
| 合计 | **16** | **15** | **14** | |

## 4. 该领域的"减迭代次数"傻基线

有人做过，而且曲线形状与 VLA 不同——**是悬崖不是平台**：

1. **DP（DDPM 训练，robomimic image + PushT）**：100 步 0.829 → DDIM-10 0.836（无损）→ DDIM-4 ≈ 0.94（STEP 口径，无损）→ **DDIM-2 ≈ 0.64、PushT 0.29** → **DDIM-1 全任务 0.000**（OneDP Table 1）。悬崖在 2 步以下。https://arxiv.org/html/2410.21257 ；https://arxiv.org/html/2602.08245
2. **DDPM 采样器直接砍步（不换 DDIM）**：Adroit 上 100 步 0.68–0.88、5 步 0.85–1.00、**2 步 0.00–0.13**（GDP Table 1）；"DDIM 反而是机器人任务里更好的快速采样器"。https://arxiv.org/html/2510.21991
3. **为什么与 VLA 不同**：VLA 的头是 flow-matching x₀-预测，一步 Euler = 条件均值；DP-DDPM 是 ε-预测 + 随机采样，一步 DDIM 从纯噪声出发的 x₀ 估计在多峰动作分布上崩，PushT（对称双峰）掉得最狠（0.94 → 0.29）正是这个机制。这与简报 §2 "输出多峰时均值崩"的判断一致，而且是**公开权重就能复现的**。
4. **但傻基线有第二段**："自身上一 chunk 做初值"：RTI-DP 3 步 ≈ 满步（Push-T 1.00 vs 0.92）、Falcon 训练无关 2–7×、WarmPrior NFE=1 反超满步 12 pp。所以 DP 线真正要打的基线不是"从噪声 k 步"而是"从自身上一 chunk k 步"，我们的 warm 档相对它的增量只在 (a) episode 起点、(b) 模式切换 / 突变处（RTI-DP 自认失效）、(c) 1 步 vs 3 步的 2 步差。https://arxiv.org/html/2508.05396 ；https://arxiv.org/html/2503.00339 ；https://arxiv.org/html/2605.13959
5. **MCTD**：把搜索砍到 0 = Diffuser：cube-double 78 → 12、triple 40 → 8；Fig. 6 Diffuser 加随机搜索样本几乎不涨、MCTD 随预算涨到满分。这是本调研里最硬的"砍预算会崩"证据。https://arxiv.org/html/2506.09498v4 ；https://arxiv.org/html/2502.07202
6. **TTS-VLA**：砍候选数掉得温和：RoboMonkey LIBERO-Long N=16 比 N=1 高 6.7 pp、MG-Select RoboCasa N=4 比 N=1 高 3.4 pp、SVA pass@1 33% vs pass@32 92%（这是上界不是 SR）。https://arxiv.org/html/2506.17811v1 ；https://arxiv.org/html/2510.05681 ；https://arxiv.org/html/2607.03751
7. **VLAPS**：砍到 0 = Octo 自己 60.2 vs 87.4；预算 600 → 1800 s 再 +4.2 pp（带价值）。https://arxiv.org/html/2601.00969

## 5. 预期收益（与 VLA 对比）

- **hit 成本地板**：DP-DDPM-100 ≈ 1%（估计，前缀 2–5 ms / 660 ms）、DDIM-10 ≈ 5%；MCTD ≈ 0%；TTS 25–35%（估计）。VLA 是 15%。前两者比 VLA 好一个数量级。
- **可恢复冗余的下界已被竞品给出（DP 线）**：Falcon 在 DDPM-100 上砍到 12.9–46.9 NFE 不掉点 ⇒ 仅靠自身历史 IR ≈ 0.13–0.47、ε ≈ 0；RTI-DP 3 步 ⇒ IR ≈ 0.03–0.05（相对 DDPM-100）或 ≈ 0.3（相对 DDIM-10）。**这意味着以 DDPM-100 为 miss 的 IR 数字会非常好看但审稿人不认；以 DDIM-10 为 miss，自身复用已到 IR≈0.3，我们要在 IR < 0.3 的区间靠库把 SR 撑住。**
- **跨 episode 库的增量**：无文献数字，**不确定**。可期待的来源同世界模型线：episode 起点（自身无历史）、模式切换处（自身初值错峰）、1 步 vs 3 步。若三者合计能把无损 IR 从 0.3 压到 0.15（相对 DDIM-10），就够一节；能否成篇取决于 §7 杀手门。
- **MCTD 线**：Fast-MCTD 已把 wall-clock 压 100×，但每次重规划仍 3–9 s（8 卡）；库命中直接跳过搜索 ⇒ 单集若 10 次重规划中 5 次 hit，IR ≈ 0.5、绝对省 15–45 s/集。SR 代价未知。
- **TTS 线**：CoVer 结构下"库存候选 + 只跑 verifier"能把 453 ms 压到约 8 ms + 前缀（估计 > 100 ms），IR 地板由 π0.5 前缀决定 ≈ 25–30%，与 VLA 线相当，**收益不比现在大**。

## 6. novelty 地形与最强反驳

按与我们形式的距离排序：

1. **RTI-DP**（IROS 2025）：training-free，上一 chunk 平移做初值，减少去噪步数 = 我们的 warm 档（自身版）。https://arxiv.org/html/2508.05396
2. **Falcon**（2025-03）：training-free，latent buffer 存部分去噪历史，Tweedie 一步估计打分选起点、噪声阈值防重复 = warm 档 + 单步版 gate；还挂到 RDT-1B。https://arxiv.org/html/2503.00339
3. **STEP / WarmPrior / Action-Prior Denoising**（2026）：需训练的 warm start（预测器 / 源分布锚定 / 训练期模拟延迟）。https://arxiv.org/html/2602.08245 ；https://arxiv.org/html/2605.13959 ；https://arxiv.org/abs/2605.25537
4. **RoVer perception cache、MG-Select single-prefill**：候选间共享前缀（同一决策内），不是跨决策缓存。https://arxiv.org/html/2510.10975 ；https://arxiv.org/html/2510.05681
5. **V-VLAPS 的 2000-medoid 动作 chunk 库**：从成功轨迹聚类出的动作库当 MCTS 动作集——"库"概念已出现在 VLA 搜索线，但用途是搜索空间离散化，不是缓存。https://arxiv.org/html/2601.00969
6. **ActionCache**（VLA 语义 cache，hit/warm 两档）：已知竞品。
7. **AOR / AdaReP / Lightning**：见世界模型线 §6。

**最强反驳**：
- "Falcon 已经是 training-free、带 buffer 与阈值的部分去噪复用，且在 48 个环境 + RDT-1B 上验证；你们只是把 buffer 换成跨 episode 库。请证明库在 Falcon 达不到的区间（IR < 0.13 相对 DDPM-100）还能保 SR。"
- "DDIM-10 无损，你们的 miss 成本按 DDPM-100 算是灌水。"——必须两套 IR 都报，主图用 DDIM-10。
- "DP 每决策 66 ms 不需要省。"——只能靠 MCTD（s 级）或 CoVer（0.45 s）补绝对时间故事。
- "OGBench cube 目标只有 5 个、迷宫起终点被改成确定性，库命中是 memorization。"——cube 任务必须报按初始位姿分桶的命中率，并与 Diffuser（无搜索）和 MCTD 低预算作对照。

我们剩下的新东西：跨 episode 库（Falcon 只用自身历史）+ 多档同存 + RIT 单 δ 离线标定与按预算反解 + 状态门 + R_C(ε)/V_C(B) 形式化——与世界模型线结论一致。

## 7. 首个实验方案：三天杀手门（DP-DDPM × robomimic，单张 4090 或 A5000）

选 DP 而非 MCTD 做门的理由：三天内**确定能跑**（官方 ckpt、robomimic 50 集分钟级），且门要回答的问题对三条线通用："跨 episode 库能否打过自身上一 chunk 的 warm start"。MCTD 的 cube 权重是否存在都未确认，不能当三天门。

**Day 1 — 成本拆分 + 两条傻基线（只跑 Square-mh 与 PushT-image，各 50 集）**
- 下载 `image/square_mh/diffusion_policy_cnn` 与 `image/pusht/diffusion_policy_cnn` 官方 ckpt。实测：ResNet 前缀单独、UNet 单步、DDPM-100 / DDIM-10 全程 wall-clock。得 c_pre/c_0 两个口径。
- 傻基线 A（从噪声）：DDIM k = 10, 4, 2, 1。预期复现 0.94 / 0.94 / 0.64 / 0（robomimic 均值口径）与 PushT 0.94 / 0.73 / 0.29 / 0。
- 傻基线 B（自身 warm start，复现 RTI-DP 平移初值）：k = 3, 2, 1。这条是必须超过的线。
- **Kill-1**：若基线 B 在 k=1 已 ≥ 满步 −2 pp，则 DP 线无 warm 档 counterfactual，转 MCTD。

**Day 2 — 库 + shadow + RIT 前提检查**
- collect：每任务用 DDPM-100 跑 200 集，成功集入库；条目 = (ResNet 特征、本体状态、8 步 chunk、x_t 在 t ∈ {2, 5, 10} 的部分去噪状态)。
- shadow：另跑 50 集，记录每决策特征与从噪声全算的参考 chunk；离线检索（单 cell），算 D_hit、D_warm(k=1)、D_warm(k=2)；同时算基线 B 的 D_self(k=1,2)。
- 检查：q_a(s) 随 s 单调？各档有序？上一决策分数对下一次 hit 的 AUROC？**库的 D 在高分区是否低于 D_self**（这是库存在的必要条件）。
- **Kill-2**：q_a(s) 不单调，或高分区 D_hit ≥ D_self(k=1)。

**Day 3 — 闭环前沿（K=2：hit + warm-1 步）**
- δ 扫 4 点 × 50 集 × 2 任务；对照：基线 A、基线 B、Falcon 简化版（buffer + 噪声阈值）。报 (IR_DDPM100, IR_DDIM10, SR)、hit/warm/miss 占比、命中成串统计。
- **Kill-3**：在 SR ≥ 满步 −3 pp 约束下，库前沿的 IR_DDIM10 比基线 B / Falcon 简化版低不到 10 个百分点。

三个 kill 任一触发即写 NO 并把算力转向 MCTD cube（先确认 ahn-ml/mctd 是否含 cube 权重；否则要自训 Diffusion Forcing 规划器，估计 1–2 天）。

**若门通过的第二阶段**：(i) MCTD cube-double/triple 上做"库跳过搜索"的 s 级绝对节省故事；(ii) RoboMonkey 或 CoVer 结构上做"库存候选 + verifier-only"档，复用我们 LIBERO harness，把 verifier 分数当 D_a 直接喂 RIT。

## 8. 结论

**MAYBE（偏 GO，限 DP-DDPM × robomimic/PushT 做门、MCTD × OGBench cube 做绝对时间故事）**：结构匹配与 hit 成本地板（≈ 1–5%）都比 VLA 好一档，"砍步会崩"的 counterfactual 有公开权重可复现（DDIM-1 = 0%、DDIM-2 PushT 0.29），MCTD 线更有 s 级成本与 12% vs 78% 的搜索 counterfactual；但 DP 线的 warm start 已被 Falcon / RTI-DP / STEP / WarmPrior 占满，能否成篇取决于三天门里"跨 episode 库 vs 自身上一 chunk"在 IR < 0.3（相对 DDIM-10）区间能否多压 ≥ 10 个百分点且不掉 SR。TTS-VLA 线 hit 地板与现在的 VLA 线相当，不建议作主线。
