# 跨领域迁移调研（第二轮）：视频/扩散世界模型在推理回路里做规划或验证（2025-01 之后）

> 调研日期 2026-09-16。承接 `logs/cache_transfer/world_models_planning.md`（第一轮：TD-MPC2 / V-JEPA 2-AC / Cosmos Policy / AdaReP 等，本文不重复），按 `briefing.md` §3 打分表 A–G 与 §4 报告格式。
> 范围：**world-model-in-the-loop** —— best-of-N 提案 + 想象 rollout + 验证器/价值打分、MPC over video/latent model、world model as verifier/critic、world-action model（WAM）、导航世界模型规划、UniPi/AVDC 后继、Genie/Oasis 类交互模型。Cosmos Policy 本身按指令**不调研**，但别人挂在它上面的方法（GeoBoN）的数字照引。
> 数字一律给 URL；文献没有的标"估计"或"不确定"，不编。

## 0. 本轮结论先行

1. 2026 年这条线的主流已经从"视频模型当规划器 + CEM"转成 **World Action Model（WAM）**：一个 5–14B 的视频扩散骨干同时去噪未来视频与动作 chunk（LingBot-VA / X-WAM / Motus / DreamZero / Cosmos Policy），**每次决策 1–8 s**，头是 20–75 步的联合去噪，而且 **减步会崩**（DreamZero 4 步 83% → 1 步 52%；Flash-WAM 不蒸馏直接减步"collapses"）。这一点与 VLA 上"k=1 无损"正好相反，是本领域对我们 warm 档最有利的证据。
2. 但有两个新出现的"傻基线"会正面打我们：**Fast-WAM**（训练时联合视频、**测试时跳过视频生成**，LIBERO 97.6 vs 98.5、延迟 190 ms vs 810 ms）和 **best-of-N 只买到 1–2 pp**（GeoBoN：LIBERO-Long 97.2→98.3、RoboCasa 80.8→82.5）。因此"昂贵的头"只有在**想象被用来验证/择优**且**该设定有 SR 余量**时才立得住；LIBERO 天花板 97–99% 几乎没有余量，RoboCasa（79–82%）与 RLBench/World-in-World（24→45）才有。
3. novelty 地形比第一轮更挤：**C³ache**（跨 chunk 残差缓存，LIBERO 2.51×、+0.2 pp）、**FFDC/When-to-Trust-Imagination**（自适应执行长度，前向调用 −69%）、**Gated GeoBoN**（只在 26% 决策点触发 BoN）、**CheckVLA**（离线 shadow 轨迹 + conformal 分位阈值 —— 就是 RIT 的近亲）、**DreamZero DiT cache**（16→4 步）。我们剩下的新东西仍是：**跨 episode 库 + 从检索到的部分去噪 latent 热启动 + 多档单 δ 离线标定 + 按预算反解**。
4. 综合分 **16 / 21**（A3 B2 C2 D3 E2 F1 G3），比第一轮 TD-MPC2 线（15/21）高 1 分，形状不同：B 更好（头对步数敏感）、G 因 LeRobot 已集成 LingBot-VA 而更好、F 同样拥挤。
5. **Top-3**：① LingBot-VA（Apache-2.0，LeRobot 集成，LIBERO-Long ckpt，18–24 GB 显存）+ GeoBoN 式 BoN；② X-WAM（Apache-2.0，RoboCasa ckpt，RTX 3090 上 1.03 s/决策，N=8 时 9.67 s）；③ World-in-World（ICLR'26 Oral，MIT）RLBench 回路里的 post-trained Wan2.2-5B / Cosmos-Predict2-2B（3D-DP 24.0 → 44.7）。三天杀手门见 §7。

## 1. 候选清单

### 1A. 主候选：世界模型在决策回路里（有公开权重 + 仿真 benchmark 优先）

| # | 名称 / 日期 / venue | 架构与参数量 | 推理算法与预算（提案数 × 去噪步 × rollout） | 每次决策耗时（GPU） | 循环体占比 | 闭环？ | URL |
|---|---|---|---|---|---|---|---|
| 1 | **LingBot-VA**（2026-01，RSS 2026） | Wan2.2-5B 双流 MoT（视频流 3072d × 30 层 + 动作流 768d），共 **5.3B**；帧因果自回归 + KV cache | 单提案；LeRobot 默认 **视频 20 步（CFG）+ 动作 50 步**，每 chunk 4 个 latent 帧 × 4 动作 = 16 动作；论文部署版 3 步视频（到 s=0.6）+ 10 步动作 | **6.77 s/chunk（LIBERO，L40S，20v/50a）**、8.1 s（RoboTwin 25v/50a：视频 3550 ms + 动作 4550 ms）；GeoBoN 实测基线 **2.27 s**（LIBERO-Long，H200 级） | 去噪循环 ≈ 全部（3550+4550 ms = 8.1 s 全额），前缀（VAE 编码 + UMT5 + KV prefill）估计 < 5%（**估计**） | 是：每 chunk 重推，观测到的真实关键帧回灌 KV cache | 论文 https://arxiv.org/abs/2601.21998 ；代码 https://github.com/robbyant/lingbot-va ；LeRobot 文档（超参/显存）https://huggingface.co/docs/lerobot/en/lingbot_va ；延迟拆分来自 Flash-WAM https://arxiv.org/html/2606.05254 |
| 2 | **GeoBoN / Gated GeoBoN**（2026-07-20，UMich+NVIDIA） | training-free，挂在 Cosmos Policy / X-WAM / LingBot-VA / Motus 公开 ckpt 上；验证器 = 冻结几何基础模型 VGGT-Ω 的跨视角深度重投影不一致性 | **N ∈ {2,4,8,16}** 提案 × 各自 WAM 的完整去噪 × 一个 chunk 的想象视频；Gated 版先用"动作–光流一致性"门（阈值 τ_gate = −0.2），只在 **26.2%** 决策点触发额外采样 | 基线 / Gated / 固定 N=8（秒）：Cosmos-RoboCasa 0.90/1.29/3.65；X-WAM-RoboCasa 2.68/3.11/9.67；LingBot-LIBERO-Long 2.27/2.83/3.86；Motus-RoboTwin 2.29/2.83/3.86（H200 与 RTX Pro 6000） | 采样 vs 验证器拆分未给（不确定） | 是：逐决策点 | https://arxiv.org/html/2607.17454 |
| 3 | **X-WAM**（2026-04-29，ckpt 2026-06-09） | Wan2.2-TI2V-5B，多视角 RGB-D 视频 + 动作联合；Asynchronous Noise Sampling（动作 T_a=5 步、视频 T_O=25 步） | 单提案；动作在前 5 步即可解码（4.5× 提速），视频 25 步只在需要想象时跑完 | **1033 ms/动作（RTX 3090，仅动作早停）** vs 4665 ms（同步全跑）；GeoBoN 全跑基线 2.68 s | 视频尾部 20 步 ≈ 78% 的全跑时间（由 1033/4665 反推，**估计**） | 是 | https://arxiv.org/html/2604.26694v2 ；https://github.com/sharinka0715/X-WAM |
| 4 | **DreamZero**（2026-02-17，NVIDIA GEAR） | Wan2.1-I2V-14B 自回归视频扩散 + 动作联合流匹配，**14B**（5B 消融版） | 单提案；**16 步**，K=2 latent 帧 = 1.6 s 动作视野；DiT caching 16→4 步；Flash 版 1 步 | naive **5.7 s/chunk**；repo：**H100 ≈ 3 s/次、GB200 ≈ 0.6 s**；Flash ≈ 150 ms；需 **2 GPU**（CFG 并行） | DiT 去噪主导（16 步 + CFG 双份） | 是：异步，真实观测替换 KV cache 里生成帧 | https://arxiv.org/html/2602.15922v1 ；https://github.com/dreamzero0/dreamzero |
| 5 | **Motus / MotuBrain**（2025-12 CVPR 2026 / 2026-04） | Wan2.2 5B + Qwen3-VL-2B + 专家 ≈ **8B**，MoT 三专家，UniDiffuser 调度可切 WM/VLA/IDM 模式 | 单提案；20 步 | GeoBoN 基线 2.29 s（RoboTwin）；显存 24 GB（预编码 T5）/41 GB | 不确定 | 是 | https://arxiv.org/abs/2512.13030 ；https://huggingface.co/motus-robotics/Motus ；MotuBrain https://arxiv.org/abs/2604.27792 |
| 6 | **World-in-World**（2025-10，ICLR'26 Oral） | 统一在线规划 harness；被测 WM：SVD 1.5B、LTX 2B、Hunyuan 13B、Wan2.1 14B、Wan2.2 5B/14B、Cosmos-Predict2 2B、NWM 1B、PathDreamer、SE3DS、Gen4；† = 在 RLBench 动作-观测数据上 post-train | policy-guided beam search：提案策略出 **M** 条动作序列 → WM 想象 → 打分择优；操作任务 M=5，L=5（VLM 策略，≤15 宏步）或 L=50（3D-DP，≤8 宏步） | 未报延迟（不确定）；14B 视频模型每次想象数秒级（**估计**） | — | 是：每宏步重规划 | https://arxiv.org/html/2510.18135 ；https://github.com/World-In-World/world-in-world |
| 7 | **VLA-Reasoner**（2025-09，ICRA 2026） | 世界模型 = **iVideoGPT 600M**（token 自回归，非扩散），在 VLA 自身失败 rollout 上微调；插件式挂 OpenVLA-7B / Octo / SpatialVLA / π0-FAST | **MCTS**：KDE 建离线动作分布 → top-k 扩展 × MaxDepth；具体次数未给 | 未报（真机用 RTX 4090） | — | 是：每步重建树，无树复用 | https://arxiv.org/html/2509.22643 ；iVideoGPT 权重（MIT）https://github.com/thuml/iVideoGPT |
| 8 | **GPC（Generative Predictive Control）**（2025-02，T-RO 2026） | 像素扩散世界模型（EDM UNet，Nd=3 步，H=4）+ 冻结 Diffusion Policy（100 步 DDPM） | GPC-RANK：K=10–100 提案 × WM rollout；GPC-OPT：M=20–30 步梯度精修（从策略样本热启动） | BC 0.5 s → RANK **12 s** → OPT 39 s → RANK+OPT 374 s（每周期） | **WM rollout 占 90–95%** | 是：receding horizon | https://arxiv.org/html/2502.00622 |
| 9 | **EV-WM**（2026-06-15） | DINO-WM 式特征空间动力学 + CEM + 谓词级验证（进度/语义一致/物理可行/不确定性） | CEM（样本数/迭代未给）；带 **retrieval-initialized planning**（最近 latent 轨迹初始化） | 未给 | — | 是：保守混合门逐决策点重评 | https://arxiv.org/html/2606.13053v2 |
| 10 | **QWM（Q-Learning With World Models）**（2026-08，Stanford） | 视觉版：Wan2.2-TI2V-5B 改动作条件，5 帧 128² 视频；状态版 3 层 MLP | 树搜索 **N=8 动作 × K=8 预测 × D=4 深度**，Q/V 聚合 | 未量化（"non-trivial overhead"） | — | 是 | https://arxiv.org/html/2608.17163 |
| 11 | **World Action Planner**（2026-07-30，Yilun Du） | Wan-T2V-1.3B 姿态-图像条件 WM（20 步，21 历史帧→20 未来帧）+ Gemini 3.0 Flash 提案/打分 + 低层 DP | 全局优化（VLM 评想象）→ 局部网格采样 N 候选 → VLM 择优 | 附录 E 有 wall-clock，正文未给（不确定） | — | 迭代式 | https://arxiv.org/html/2607.27599 |
| 12 | **NWM 及后继**（NWM CVPR 2025；NavWM 2026-06；NavWAM 2026-06） | NWM：CDiT-XL **1B**；NavWM：Mamba 骨干 + latent world tokens；NavWAM：去掉 CEM 的一体化 WAM | NWM：CEM **N=120 × 1 迭代 × 8 步（2 s 视野）** | NWM：**30.3 s/rollout（RTX 6000 Ada）**，作者称量化后可到 0.1 s | — | NWM 单次开环规划（非闭环）；NavWM 闭环 | https://arxiv.org/html/2412.03572 ；https://arxiv.org/abs/2606.24101 ；https://arxiv.org/pdf/2606.13494 |
| 13 | **LeWM / Fast-LeWM / 摊销规划**（2026-03 / 06 / 05） | JEPA latent WM **15M**；Fast-LeWM 用 action-prefix 并行预测多视野 latent | CEM（Two-Room/Reacher/PushT/Cube） | Fast-LeWM：动力学模块 31.4 s → 8.0 s，**整段 CEM 54.4 s → 28.3 s**；摊销 GC-IDM 每决策便宜 **100–130×** | — | 是 | https://arxiv.org/abs/2603.19312 ；https://fast-lewm.github.io/ ；https://arxiv.org/abs/2605.08732 |
| 14 | **VLAPS / V-VLAPS**（2025-08 / 2026-01） | 用**真仿真器**当模型（非世界模型），Octo-base 93M；V 版加轻量价值头 | MCTS 300 次模拟/迭代，k=10，深度 ≤100；每集 600–1800 s 时限 | **每 100 集 3.5–4.5 h（H100）** | — | 是：每步重建树 | https://arxiv.org/html/2508.12211 ；https://arxiv.org/html/2601.00969 |
| 15 | **DREAMSTEER**（2026-07，Meta FAIR） | 动作条件 latent WM + 语言条件价值模型 | 采样 VLA chunk + 运动原语 → 想象 → 排序 | 未给 | — | 是 | https://arxiv.org/abs/2607.02865 |
| 16 | **WoW-1 + SOPHIA**（2025-09） | 14B（另有 1.3B 开源）生成式 WM + VLM critic 团队迭代改写 prompt + FM-IDM 出动作 | 预测→批评→精修闭环（轮数未给） | 未给 | — | 是（真机） | https://arxiv.org/abs/2509.22642 ；https://huggingface.co/WoW-world-model/WoW-1-Wan-1.3B-2M |
| 17 | **MinD**（2025-06） | 低频视频生成器 LoDiff + 高频扩散策略 HiDiff + DiffMatcher；关键发现：策略只需**单步去噪的低分辨率 latent** | 单提案 | **11.3 FPS** | — | 是，兼做失败预警（提前识别 74% 失败） | https://arxiv.org/abs/2506.18897 |
| 18 | **GVP-WM**（2026-02） | 视频生成器出 plan + 动作条件 WM 做 video-guided latent collocation 投影到可行流形 | 一次生成 + 优化 | 未给 | — | 不确定 | https://arxiv.org/abs/2602.01960 |

### 1B. 验证器 / 门类（world model as verifier / critic；多数是我们 gate 与 RIT 的直接竞品）

| 名称 / 日期 | 机制 | benchmark 与数字 | 与我们的关系 | URL |
|---|---|---|---|---|
| **When to Trust Imagination（FFDC-WAM）** 2026-05 | 轻量验证器 FFDC 拿 WAM 想象的视频/动作 token（存为 KV cache）对比最新真实观测，决定还执行多少已想象动作；**阈值固定 0.5**，无标定 | RoboTwin 2.0（Motus）：短 chunk 基线 87.66%（5.47 次推理/集，21.3 s）→ 长 chunk-64 88.46%（1.56 次）→ FFDC 88.90%（**1.72 次，−69.1% 前向**，13.5 s）；真机 45→80% | = 我们的"状态门 + 自身 plan 复用"，无跨 episode 库、无离线标定 | https://arxiv.org/html/2605.06222 |
| **CheckVLA** 2026-07 | 冻结 V-JEPA 2-AC 特征 WM（监视器 88.4M）逐步比对预测与观测；**离线 shadow-mode 名义成功轨迹上做 functional conformal 分位标定** δ_t = μ_t + q̂_α σ_t，保证首次误干预率 ≤ α | RoboCasa365：周期重规划 27.6% → **36.1%**；1.18× wall-clock，p95 16.4 ms；调用次数与周期基线持平（≈10/集） | **最像 RIT**（离线 shadow + 分位阈值），但目标是可靠性不是省算力，单档、无 δ 扫 ladder | https://arxiv.org/html/2607.26789 |
| **Gated GeoBoN** 2026-07 | 见 1A#2：动作–光流一致门 → 只在 26.2% 决策点做 BoN | 回收 74.8% 的 always-on 增益 | = 二元"hit-or-miss"版规划预算门，阈值手调 | https://arxiv.org/html/2607.17454 |
| **Pre-VLA** 2026-05 | 执行/想象前的抢先验证头（安全置信 + critic 优势） | LIBERO（RynnVLA-002）30.79 → 37.62%；验证 183.9 ms/chunk | 验证器成本参考 | https://arxiv.org/abs/2605.22446 |
| **World Action Verifier（WAV）** 2026-04 | 前向–逆向不对称自验证，让 WM 识别自身预测错误 | — | 想象可信度 | https://arxiv.org/html/2604.01985v1 |
| **SV-VLA（speculative verification）** 2026-04 | 重 VLA 低频开环宏规划 + 轻验证器闭环监控，必要时才重规划；无世界模型 | 未给数字 | 名字撞"speculative"，机制 = 门 | https://arxiv.org/abs/2604.02965 |
| **A3 Dynamic Execution Commitment** 2026-05 | 组采样共识 + 前缀一致性决定执行多长；无 WM | 未给 | 门 | https://arxiv.org/abs/2605.11567 |

### 1C. 只在训练 / 评测阶段用世界模型（第一轮子族 ii/iii，本轮只登记位置，不打分）

Ctrl-World（ICLR 2026，DROID，HF 权重 https://huggingface.co/yjguo/Ctrl-World ，评测排序 + 合成 SFT +44.7%）https://arxiv.org/abs/2510.10125 ；World4RL（冻结扩散 WM 内 RL 精修）https://arxiv.org/abs/2509.19080 ；WMPO https://arxiv.org/pdf/2511.09515 ；RISE（RSS 2026，组合式 WM + 进度价值模型，想象空间自我改进）https://arxiv.org/abs/2602.11075 ；VLA-RFT（可控 WM 里验证奖励，<400 步微调）https://arxiv.org/abs/2510.00406 ；WorldSample https://arxiv.org/pdf/2607.02431 ；Genie Envisioner GE-Base/GE-Act/GE-Sim（CC BY-NC-SA 4.0，CALVIN 权重）https://github.com/AgibotTech/Genie-Envisioner ；τ0-WM（AgiBot，27.3k h，带"动作评估"接口出进度分）https://arxiv.org/abs/2606.01027 ；WEAVER（多视角 latent 流匹配 WM 预测 latent + reward）https://arxiv.org/abs/2606.13672 。
Genie/Oasis 类交互模型（Genie 3、Matrix-Game 2.0/3.0、Oasis）目前只有游戏/漫游控制，无操作 benchmark，且 Genie 3 不开权重：https://arxiv.org/html/2508.13009v1 ；https://arxiv.org/html/2604.08995v2 。
UniPi/AVDC 后继（视频 plan + IDM，闭环只执行首动作）：Adapting Internet Video Knowledge（Meta-World/LIBERO）https://arxiv.org/pdf/2504.15369 ；TC-IDM https://arxiv.org/pdf/2601.18323 ；Implicit State Estimation via Video Replanning https://arxiv.org/pdf/2510.17315 —— 这一族每步一次视频生成、无 BoN、无中间状态复用故事，不如 WAM 族贴合，不展开。

### 1D. 主候选的 benchmark 与"不用世界模型 / 减预算"对照

| 系统 | benchmark | 有 WM 回路 | 对照（无 WM / 减预算） | 增益 | URL |
|---|---|---|---|---|---|
| GeoBoN on LingBot-VA | LIBERO-Long（50 集/任务） | N=8：**98.3%** | N=1：97.2% | +1.1 pp | https://arxiv.org/html/2607.17454 |
| GeoBoN on Cosmos Policy | LIBERO-Long / RoboCasa（10 集/任务） | 99.3% / 68.4% | 97.5% / 66.3% | +1.8 / +2.1 pp | 同上 |
| GeoBoN on X-WAM | RoboCasa | 82.5% | 80.8% | +1.7 pp | 同上 |
| GeoBoN on Motus | RoboTwin 2.0 | 89.9% | 87.8% | +2.1 pp | 同上 |
| World-in-World（SVD† / Cosmos-P2†） | RLBench 4 任务 × 50 集 | 3D-DP 提案：**44.7% / 38.0%**；VLM 提案：46.5% / 45.0% | 3D-DP 单独 24.0%；VLM 单独 44.5% | **+20.7 pp**（3D-DP）；+2.0 pp（VLM） | https://arxiv.org/html/2510.18135 |
| World-in-World 推理算力扩展 | Active Recognition | 每集平均推理数 3 → 11 | — | SR 53.36 → 60.98% | 同上 |
| VLA-Reasoner | LIBERO（OpenVLA-SFT）/ SimplerEnv | 81.0% / Octo 37.3%、SpatialVLA 41.8% | 76.0% / 26.5%、34.0% | +5 / +10.8、+7.8 pp | https://arxiv.org/html/2509.22643 |
| GPC | Push-T 视觉 | RANK 0.739 / OPT 0.791 / 两者 0.882 | BC 0.642 | +9.7 / +14.9 / +24 pp | https://arxiv.org/html/2502.00622 |
| EV-WM | PointMaze 随机态 | 0.94 | DINO-WM 0.90 | +4 pp | https://arxiv.org/html/2606.13053v2 |
| V-VLAPS（仿真器当模型） | LIBERO-10（Octo） | 600 s：75% / 1800 s：85% | 38% | +37 / +47 pp（**上界参考**，模型无误差） | https://arxiv.org/html/2601.00969 |
| DreamZero 步数阶梯 | 真机 table bussing（task progress） | 4 步 **83%** | 1 步 **52%**；Flash 蒸馏 1 步 74% | 减步 −31 pp | https://arxiv.org/html/2602.15922v1 |
| Flash-WAM（LingBot-VA 蒸馏） | RoboTwin 2.0 / LIBERO / 真机 | 教师 25v/50a：91.25 / 98.6 / 66.7% | 蒸馏 1v/1a：81.41 / 95.1；不蒸馏减到 1v/2a 真机 **40.0%**；naive joint LCM 36.32% | 减步不蒸馏 −27 pp（真机） | https://arxiv.org/html/2606.05254 |
| Fast-WAM（测试时跳过视频） | LIBERO / RoboTwin 2.0 | 联合想象 98.5 / 90.6%；IDM 版 98.0 / 91.3% | **跳过视频 97.6 / 91.8%**；无视频共训 93.5 / 83.8% | 跳视频 −0.9 / +1.2 pp，延迟 810 → **190 ms**（RTX 5090D） | https://arxiv.org/html/2603.16666v1 |
| C³ache（跨 chunk 残差缓存） | LIBERO 4 套 2000 集 / RoboTwin 50 任务 | 2.51× / 1.84× 提速 | 无缓存 | +0.2 / −1.0 pp | https://arxiv.org/html/2606.08962 |

### 1E. 权重 / 许可证 / 我们硬件上的可行性

| 系统 | 权重 | 许可证 | 显存 / 硬件 | 我们能跑？ | 备注 |
|---|---|---|---|---|---|
| LingBot-VA | `robbyant/lingbot-va-base`、`-posttrain-libero-long`、`-posttrain-robotwin`；LeRobot 转换版 `lerobot/lingbot_va_libero_long` | **Apache-2.0** | 5B DiT + 冻结 VAE/UMT5（UMT5 可放 CPU）**18–24 GB**；仅 `--eval.batch_size=1` | **能**：4090 48G 跑 N=1；H100 跑 N=8 批量；LIBERO harness 现成（lerobot-eval + 我们自己的 LIBERO 池） | 微调需 LoRA；RoboTwin 需 SAPIEN+CuRobo 栈 |
| X-WAM | GitHub 指向 HF 的 RoboCasa / RoboTwin 微调 ckpt + Wan2.2-TI2V-5B 底座 | **Apache-2.0** | 3090 可跑（1033 ms） | **能**（需 RoboCasa 24 任务版环境；与我们 RoboCasa365 harness 是否同一 env 版本**不确定**） | https://github.com/sharinka0715/X-WAM |
| DreamZero | `GEAR-Dreams/DreamZero-DROID`、`-AgiBot`（14B，~45 GB） | Apache-2.0 | **最少 2 GPU**（CFG 并行），H100 ≈ 3 s/次 | **勉强/不确定**：H100 + 4090 异构双卡未验证；仅 DROID-sim/RoboArena，PolaRiS/Genie Sim "coming soon"，无 LIBERO/RoboCasa | https://github.com/dreamzero0/dreamzero |
| Motus | `motus-robotics/Motus`（Stage-2 预训练） | Apache-2.0 | 24 GB（预编码 T5）/41 GB | 能跑，但只有 RoboTwin 2.0 评测 | https://huggingface.co/motus-robotics/Motus |
| World-in-World | harness MIT；被测 WM 均为公开视频模型；† post-train 权重是否放出**不确定** | MIT | 14B 视频模型 → H100 | 能（需装 RLBench/CoppeliaSim 与 wow-manip 后端） | https://github.com/World-In-World/world-in-world |
| VLA-Reasoner | 无代码（仅项目页）；iVideoGPT 权重 MIT | — | 600M WM + 7B VLA | 需自己微调 iVideoGPT 并重写 MCTS：中风险 | https://github.com/thuml/iVideoGPT |
| GPC / EV-WM / QWM / FFDC / C³ache / CheckVLA / GeoBoN | 均**未见代码** | — | — | 只能复现思路 | — |
| WoW-1 | 14B / 1.3B HF | 不确定 | — | 无仿真 benchmark，不选 | https://huggingface.co/WoW-world-model/WoW-1-Wan-1.3B-2M |
| Genie Envisioner | GE-Base/GE-Act(CALVIN)/GE-Sim(Cosmos2) | CC BY-NC-SA 4.0（非 Apache 部分） | — | NC 许可 + 真机为主，不选 | https://github.com/AgibotTech/Genie-Envisioner |

### 1F. 各候选自带的"复用 / 缓存 / 热启动"机制（逐条 novelty 风险）

| 系统 | 自带机制 | 跨 episode？ | 多档？ | 离线标定？ | 对我们的风险 |
|---|---|---|---|---|---|
| LingBot-VA | 帧因果 KV cache 跨 chunk 复用上下文；异步推理（执行当前 chunk 时预测下一 chunk） | 否 | 否 | 否 | 低：只是前缀复用，正是我们的 c_pre |
| DreamZero | KV cache + 真实观测替换生成帧；DiT caching（速度方向一致性，16→4 步）；异步执行 | 否 | 否 | 否 | 中：步内缓存把 miss 做便宜 |
| X-WAM | ANS：动作 5 步早停、视频 25 步续跑（两级噪声结构） | 否 | 结构上有两级 | 否 | 中：可被说成"warm 档已内置" |
| Motus / MotuBrain | 推理栈：减步 + CUDA graph + FP8 + DiT cache，>50× | 否 | 否 | 否 | 中：miss 成本口径 |
| GeoBoN / Gated | 动作–光流一致门决定是否 BoN | 否 | 二元 | 否（手调 τ_gate） | **高**：二元 hit/miss 的规划预算门 |
| FFDC | 想象-现实验证器决定继续执行想象动作还是重推；验证器 KV cache | 否 | 否（可变执行长度） | 否（固定 0.5） | **高**：= 状态门 |
| C³ache | 跨 chunk 残差缓存，固定刷新间隔 | 否 | 否 | 否 | **高**：= 周期式 warm（残差级） |
| CheckVLA | 离线 shadow 轨迹 conformal 分位阈值 | 否 | 单档 | **是** | **高**：RIT 标定近亲 |
| EV-WM | retrieval-initialized planning（最近 latent 轨迹初始化 CEM）+ archive validation | 是（库） | 否 | 否 | **高**：库热启动的单档版 |
| GPC | GPC-OPT 从策略样本热启动梯度精修 | 否 | 否 | 否 | 低 |
| World-in-World | 无（每宏步全量重规划） | 否 | 否 | 否 | 低：干净的 harness |
| VLA-Reasoner / QWM / WAP | 每步重建树 / 无复用 | 否 | 否 | 否 | 低 |
| Fast-WAM / Flash-WAM / MinD | 跳视频 / 蒸馏 1 步 / 单步低分辨率 latent | 否 | 否 | 否 | **高**（作为傻基线） |
| ActionCache（VLA） | 跨 episode 语义 cache，hit/warm 两档 | 是 | 两档 | 否 | 高（同第一轮） |

## 2. 映射表（以"WAM + BoN + 验证器"回路为主；括号给 latent-CEM 族对应）

| 我们的符号 | 本领域对应 |
|---|---|
| 共享前缀 c_pre | 当前观测 VAE 编码 + 指令 UMT5 编码（可按任务缓存）+ 历史帧的 KV cache prefill（LingBot-VA / DreamZero 都把真实观测回灌 KV cache）；（DINO-WM/EV-WM：DINOv2 编码当前帧与目标） |
| tap point | KV prefill 完成、开始为 N 个候选采噪声之前；query 特征 = 当前帧 latent token 池化 + 本体状态 + 指令 embedding（与 VLA 里"各相机 token map 池化"同构，白拿） |
| 头（迭代生成） | **N 个候选 × (T_O 步视频去噪 + T_a 步动作去噪) + 验证器**（VGGT-Ω 几何一致 / VLM 打分 / 价值头）；（CEM：I 次迭代 × N 样本 × H 步 latent rollout） |
| 中间状态（warm-start 起点） | 候选（或胜者）的**部分去噪视频-动作 latent** (x_video^σ, x_action^σ)；X-WAM 的 ANS 已经把"动作在第 5 步解码、视频继续到 25 步"做成结构，天然给出两个噪声水平 |
| tier K = hit | 直接回放库里**已验证过的胜者动作 chunk**，跳过全部采样与验证；成本 = c_pre + kNN |
| 中间 tier | 从库里胜者的 σ 水平 latent 出发、在当前观测条件下只跑剩余 ρ_a·T 步、**只跑 1 个候选、不跑验证器**（或缩 N）；再高一档：跑 ρ_a·T 步 + 小 N + 验证器 |
| tier 0 = miss | 完整 N × 全步 + 验证器（注意：DreamZero DiT cache、C³ache 已经让 miss 本身比 naive 便宜 2.5–5×，成本模型要按"已优化的 miss"立） |
| key 字段 | 当前帧 latent 池化（替代相机 token map）、本体状态、指令 embedding（IVF cell）、任务 id |
| 分数 s_t | latent cosine + 本体负欧氏，库自比留一估 μ/τ 做 tanh 标准化后加权（与现行式 norm/fuse 完全一致） |
| 偏差 D_a | tier a 产出的动作 chunk 与"从零 BoN 的胜者 chunk"的标准化距离；或两者在验证器分数上的差（GeoBoN 的几何一致分 / 价值差） |
| 成功度量 SR | LIBERO / RoboCasa / RoboTwin / RLBench success rate；容差 ε 同现行 |
| 成本模型 | c_a = c_pre + n_a · ρ_a · (T_O·t_v + T_a·t_a) + 1[验证]·t_verify；wall-clock 版按 GeoBoN Table 2 实测（LingBot：2.27 → 3.86 s，Cosmos：0.90 → 3.65 s） |
| 闭环性 | 每 chunk 重推、真实观测回灌；开环执行过期 chunk 会漂（FFDC 与 CheckVLA 就是为这个漂移而生） |
| 状态门 | FFDC 的"想象与现实一致就继续执行"= 我们 gate 的单 episode 版；C³ache 的固定刷新间隔 τ = 无门的周期版 |

## 3. 打分表 A–G

| 项 | 分 | 依据 | URL |
|---|---|---|---|
| A 结构匹配 | **3** | 前缀（VAE + 文本 + KV prefill）小、头是 20–75 步联合去噪 + N 倍候选 + 验证器；中间去噪 latent 就是 warm 起点；X-WAM 的 ANS 甚至已把"动作早停/视频续跑"做成两级噪声结构；tap point 在 KV cache 之后天然存在 | https://arxiv.org/html/2604.26694v2 ；https://huggingface.co/docs/lerobot/en/lingbot_va |
| B 成本余量 | **2** | 有利：c_pre/c_0 估计 < 5%（LingBot 8.1 s 全是去噪；GPC 里 WM rollout 占 90–95%），远优于 VLA 的 15%；**减步会崩**（DreamZero 4 步 83 → 1 步 52；Flash-WAM 不蒸馏减步 collapses）。不利：Fast-WAM 证明**测试时可整个跳过视频**（LIBERO 97.6 vs 98.5，190 ms），BoN 在 LIBERO/RoboCasa 只买 1–2 pp —— 所以余量只存在于"想象确实被用来择优且有 SR 余量"的设定（RLBench 3D-DP +20.7 pp、SimplerEnv +8–11 pp） | https://arxiv.org/html/2602.15922v1 ；https://arxiv.org/html/2606.05254 ；https://arxiv.org/html/2603.16666v1 ；https://arxiv.org/html/2607.17454 ；https://arxiv.org/html/2510.18135 |
| C 局部性 / 命中潜力 | **2** | episode 内局部性硬证据：C³ache 跨 chunk 残差强相关（LIBERO 2.51× 零损）、FFDC 前向 −69%、DreamZero DiT 方向一致性 16→4 步；跨 episode：LIBERO 每任务 50 个近重复初始态是我们自己 VLA cache 已证实的命中源，WAM 上无文献数字（**不确定**） | https://arxiv.org/html/2606.08962 ；https://arxiv.org/html/2605.06222 |
| D 容忍度与闭环 | **3** | SR + 容差现成；逐 chunk 闭环；执行过期想象会漂被 FFDC、CheckVLA、DreamZero（真实观测必须替换生成帧否则误差累积）三方独立确认 | https://arxiv.org/html/2607.26789 ；https://arxiv.org/html/2602.15922v1 |
| E RIT 可用性 | **2** | shadow 可离线构造（记 KV/latent 特征 + 完整 BoN 胜者 + 验证分）；**CheckVLA 已证明"离线 shadow 轨迹 + conformal 分位阈值"在 RoboCasa365 上成立**，是 RIT 可行的正面证据也是竞品；分数单调、单 δ 切多档、按预算反解未验证 | https://arxiv.org/html/2607.26789 |
| F novelty 地形 | **1** | 直观档位全被占：Gated GeoBoN（二元规划门）、FFDC（自适应执行 = gate）、C³ache（chunk 间缓存）、DiT cache/TeaCache 类（步内缓存）、CheckVLA（离线分位标定）、EV-WM（检索初始化规划）、ActionCache（VLA 语义 cache）。剩：跨 episode 库 + 从检索 latent 热启动 + 多档单 δ + R_C(ε)/V_C(B) | 见 §6 |
| G 我们做得动 | **3** | LingBot-VA 已进 LeRobot（`lerobot-eval --env.type=libero`，18–24 GB，4090 可跑），Apache-2.0；X-WAM Apache-2.0 + RoboCasa ckpt，3090 级即可；每前沿点成本：LIBERO-10 一集 ≤ 33 个 chunk 决策 × 2.3–6.8 s ≈ 1–4 min，**500 集 ≈ 8–30 h 单卡（估计）**，比 VLA 慢一个量级但两周内出得了 3–4 个点 | https://huggingface.co/docs/lerobot/en/lingbot_va ；https://github.com/sharinka0715/X-WAM |

合计 **16 / 21**。短板仍是 F；B 的"2"是条件分——必须选有 SR 余量的设定。

## 4. 本领域的"减迭代次数"傻基线（四种，都有人做过）

1. **N → 1（不做 BoN）**：GeoBoN 报 N=1 与 N=8 差 1.1–2.1 pp（LIBERO-Long 97.2→98.3；RoboCasa 80.8→82.5；RoboTwin 87.8→89.9），Gated 版只在 26.2% 决策点触发即回收 74.8% 增益 → **在 LIBERO 上"不用世界模型"几乎免费**。World-in-World 的 3D-DP 提案 24.0 → 44.7 是唯一大余量的仿真数字（RLBench，弱基座）。https://arxiv.org/html/2607.17454 ；https://arxiv.org/html/2510.18135
2. **去噪步 → 1（不蒸馏）**：**会崩**。DreamZero 真机 4 步 83% → 1 步 52%（蒸馏 Flash 74%）；Flash-WAM 称"reducing steps without distillation collapses performance"，真机 66.7 → 40.0%（1v/2a），naive joint LCM 36.32%（RoboTwin）。这和 VLA 的 k=1 平台完全相反：WAM 头对步数敏感，warm 档有 counterfactual。https://arxiv.org/html/2602.15922v1 ；https://arxiv.org/html/2606.05254
3. **跳过视频生成（Fast-WAM）**：训练时联合、测试时只跑动作专家 10 步：LIBERO 97.6 vs 98.5（联合）vs 98.0（IDM），RoboTwin 91.8 vs 90.6，延迟 810 → 190 ms。这是最狠的一刀：**如果想象没被用来验证/择优，视频头就是纯浪费**。MinD 也发现策略只需单步低分辨率 latent。https://arxiv.org/html/2603.16666v1 ；https://arxiv.org/abs/2506.18897
4. **时序复用（自身上一 chunk）**：C³ache 跨 chunk 残差缓存 2.51×（LIBERO，+0.2 pp）；FFDC 自适应执行 −69% 前向；DreamZero DiT cache 16→4 步（H100 累计 5.5×）；X-WAM ANS 动作早停 4.5×。**miss 的真实成本要按这些已优化版本算**，否则审稿人说我们省的是 naive 实现的浪费。https://arxiv.org/html/2606.08962 ；https://arxiv.org/html/2605.06222

结论：本领域的傻基线曲线形状是 —— 沿"N"方向几乎免费（LIBERO），沿"步数"方向有悬崖，沿"跳视频"方向免费。**我们的 tiers 必须沿步数方向切、且只在"想象被用于择优"的设定里才有故事**。

## 5. 预期收益（与 VLA 对比）

- **hit 成本地板**：c_pre/c_0 估计 2–5%（LingBot 2.27–8.1 s 的决策里前缀 < 0.2 s，**估计**，需实测），对比 VLA 的 15% 和 TD-MPC2 wall-clock 的 10–20%。这是三条线里 hit 省得最多的。
- **可恢复冗余下界**（竞品给出）：C³ache 自身复用 IR ≈ 0.40（2.51×）零损；FFDC 调用数 5.47 → 1.72（IR ≈ 0.31）SR 反升 1.2 pp；Gated GeoBoN 在 BoN 预算上 IR ≈ 0.35（3.65 → 1.29 s）保 75% 增益。**仅靠 episode 内复用就已到 IR 0.3–0.4**，比 VLA（IR 60 掉 9 pp）好一个档。
- **跨 episode 库能再压多少**：无文献数字（**不确定**）。可预期增量：(a) episode 起点与 chunk 边界处（自身无可复用残差）；(b) 从库里胜者 latent 热启动只跑 1 候选 × ρ 步，把 N × T 的 miss 压到 1 × ρT；(c) 验证器免跑（库条目已验证）。若 miss = 3.86 s（N=8）、warm(ρ=0.25, N=1) ≈ 0.6 s、hit ≈ 0.15 s，命中结构与我们 LIBERO VLA 实验相当（hit 40% / warm 30% / miss 30%）时 **IR ≈ 0.35–0.40（估计）**；相对 Gated GeoBoN 的 0.35 增量不大 —— 故事必须落在"warm 档保住 BoN 增益而 Gated 保不住"上，这只有在有余量的 benchmark 上能测出来。
- **SR 损失**：LIBERO 上 teacher 98.5%，任何 tier 的最坏损失被 BoN 增益（1–2 pp）+ 减步悬崖共同界定；RoboCasa（X-WAM 79–82%）与 RLBench（24–45%）才有可见的 (IR, SR) 曲线。

## 6. novelty 地形与最强反驳

按与我们形式的距离排序：

1. **Gated GeoBoN**（2026-07）：training-free、门 + BoN + 几何验证器，挂在四个公开 WAM 上，26.2% 触发率。= 我们的"hit（N=1 直出）/ miss（N=8）"二元版，阈值手调、无库、无 warm、无离线标定。https://arxiv.org/html/2607.17454
2. **FFDC / When to Trust Imagination**（2026-05）：验证器决定继续执行想象还是重推，−69% 前向，固定阈值 0.5、作者自认阈值未系统研究。= 状态门。https://arxiv.org/html/2605.06222
3. **C³ache**（2026-06）：跨 chunk 残差缓存，固定刷新间隔 τ，无相似度判据。= 无门的周期式 warm。https://arxiv.org/html/2606.08962
4. **CheckVLA**（2026-07）：离线 shadow-mode 轨迹 + functional conformal 分位阈值 + 首次干预错误率保证。= RIT 的单档、可靠性导向版。审稿人会问"你们的 (1−α) 分位曲线与 conformal 保证的区别"。https://arxiv.org/html/2607.26789
5. **EV-WM**：retrieval-initialized planning（最近 latent 轨迹初始化 CEM）。= 库热启动的单档版（DINO-WM 族）。https://arxiv.org/html/2606.13053v2
6. **DreamZero DiT cache / Motus 推理栈 / Flash-WAM / Fast-WAM**：步内缓存、蒸馏、跳视频 —— 与我们正交但把 miss 做便宜。https://arxiv.org/html/2602.15922v1 ；https://arxiv.org/abs/2604.27792 ；https://arxiv.org/html/2606.05254 ；https://arxiv.org/html/2603.16666v1
7. **ActionCache**（VLA 语义 cache，2607.06370）、RT-Cache（检索轨迹）：库的概念出处。https://arxiv.org/abs/2607.06370 ；https://arxiv.org/pdf/2505.09040
8. **MemoryWAM / Mem-World / HiMem-WAM**：episode 内记忆 token，不是跨 episode 库。https://arxiv.org/abs/2606.20562 ；https://arxiv.org/abs/2606.18960

**最强反驳**：
- "Fast-WAM 190 ms 就是 97.6%，你们省的是本来就不该跑的视频头。" → 只能用"想象被用于择优/验证且带来可测 SR 增益"的设定回答：RoboCasa（X-WAM +1.7 pp）、RLBench（+20.7 pp）、SimplerEnv（VLA-Reasoner +8–11 pp）。LIBERO 不行。
- "Gated GeoBoN 已经用一个门把 BoN 预算压到 35% 且保 75% 增益；你们的库和 warm 档相对它多省多少？" → 三天门必须直接测 Gated 基线。
- "CheckVLA 已经在 RoboCasa365 上做了 shadow + conformal 标定。" → 我们的区别是多档 + 单 δ 生成 ladder + 按预算反解 + 服务算力而非可靠性；需在文中正面引用并对比。
- "500 集一个前沿点要 8–30 h，你们能扫几个 δ？" → G 里给出估计，两周 3–4 点，靠 gate 早停缩短。

## 7. Top-3 推荐 + 三天杀手门

### Top-3

| 序 | 推荐 | 为什么 | 风险 |
|---|---|---|---|
| 1 | **LingBot-VA（LeRobot 版）+ 自实现 BoN(N=8) + 廉价验证器**，LIBERO-10 | 唯一"今天就能在我们 harness 上跑"的 WAM：Apache-2.0、18–24 GB、`lerobot-eval --env.type=libero`；每决策 2.3–6.8 s，头 20v+50a 步；GeoBoN 已给 N=1/N=8/Gated 三条基线数字 | LIBERO-Long 98.5% 天花板，BoN 只 +1.1 pp → (IR,SR) 曲线可能平到看不出档位差；验证器需自建（GeoBoN 无代码） |
| 2 | **X-WAM + BoN**，RoboCasa（24 任务） | Apache-2.0、RoboCasa ckpt、3090 级 1.03 s（动作早停）/ 2.68 s（全跑）/ 9.67 s（N=8）；79.2–82.5% 有余量；ANS 的"动作 5 步、视频 25 步"天然两级噪声 | 需 RoboCasa 24 任务版环境（与我们 RoboCasa365 是否同版**不确定**）；GeoBoN 增益也只 +1.7 pp |
| 3 | **World-in-World harness + post-trained Wan2.2-5B/Cosmos-Predict2-2B，RLBench 4 任务，3D-DP 提案 M=5** | 最大的 WM 回路 counterfactual（24.0 → 44.7 pp）；MIT harness 已把"提案–想象–打分"API 做好；推理算力扩展曲线（3→11 次/集：53→61%）本身就是我们的 IR 轴 | 需装 RLBench/CoppeliaSim；† post-train 权重是否放出**不确定**（否则自己 post-train，H100 1–2 天，估计）；无延迟数字 |

备选：DreamZero（14B，Apache-2.0）—— 只有解决"2 GPU CFG 并行"且接受 DROID-sim 才考虑；VLA-Reasoner（iVideoGPT 600M + MCTS）—— 需自训 WM，第二阶段再看。

### 三天杀手门（平台：LingBot-VA × LIBERO-10；4090 跑 N=1，H100 跑 N=8）

**Day 1 —— 成本拆分 + 步数阶梯（回答 B 的"悬崖"是否存在）**
- 在 `LingBotVAPolicy.select_action` 里打点：c_pre（VAE 编码 + 文本编码（按任务缓存）+ KV prefill）、每步视频去噪 t_v、每步动作去噪 t_a、VAE 解码（若验证器需要像素）。得 c_pre/c_0 两个口径（wall-clock、FLOPs）。
- 阶梯：(20v/50a) / (10v/25a) / (3v/10a) / (1v/2a)，libero_10 每配置 100 集（10 任务 × 10）。
- **Kill-1**：若 (3v/10a) ≥ (20v/50a) − 1 pp 且 (1v/2a) ≥ −2 pp → 头在 LIBERO 上步数不敏感（与 VLA 同病），warm 档无 counterfactual → LIBERO 线 NO，转 X-WAM/RoboCasa 或停。
- 预算：400 集 × ~1.5 min ≈ 10 h（4090，估计）。

**Day 2 —— BoN 头有没有买到 SR（回答"Fast-WAM/N=1 就够了"）**
- 实现 BoN N=8（批量采样 8 组噪声）+ 两个验证器：(i) **oracle**：克隆仿真状态、各候选 chunk 真执行、按任务谓词进度择优（= 任何验证器的上界，VLAPS 口径）；(ii) **廉价**：GeoBoN 的动作–光流一致性 + 目标图相似（无需 VGGT）。同时跑 Gated 版（阈值触发）。
- libero_10 每臂 100 集。
- **Kill-2**：oracle-BoN(N=8) − N=1 < 3 pp → 这个设定里昂贵的头根本没有可被 cache 保护的价值，任何 tier 的 SR 差异都在噪声里 → NO（LIBERO）。若 oracle ≥ 3 pp 但廉价验证器 < 1 pp → 验证器是瓶颈，先换验证器再谈 cache。
- 预算：100 集 × ~20 决策 × (8 × 2.3 s + 仿真克隆) ≈ 12 h（H100，估计）。

**Day 3 —— 跨 episode 局部性 + shadow 单调性（回答 C/E）**
- 库：Day 2 的成功集（≈ 150–190 集）入库，条目 = (当前帧 latent 池化 + 本体 + 指令 emb，胜者 chunk，胜者在 σ∈{0.25,0.5} 的部分去噪 latent，验证分)。
- shadow：另跑 50 集完整 BoN 并记录；离线检索（按任务 cell、cosine + 负欧氏、留一 μ/τ），算 D_hit = ‖a_lib − a_ref‖ 标准化、D_warm(ρ=0.25/0.5) = 从库 latent 起跑 1 候选剩余步的 chunk 与 a_ref 的距离；拟合 q_a(s)；算上一步分数对下一步命中的 AUROC。
- **Kill-3**：q_a(s) 不随 s 单调 / 各档不有序 / 顶分区间的 D_hit 与随机检索无差 / 可命中决策点（s 高于留一 τ）< 20% → RIT 在此领域不成立，NO。
- 全程离线，< 2 h。

**三天全过才进两周方案**（K=2,3 各扫 δ 5 点 × 200 集，对照 = N=1、Fast-WAM 式跳视频、C³ache 周期复用、Gated GeoBoN），报 (IR_wall, IR_flops, SR) 与 hit/warm/miss 占比。若 Kill-1/2 在 LIBERO 触发，把同一套门原样搬到 X-WAM × RoboCasa（多 2–3 天装环境）。

## 8. 结论

**MAYBE（偏 GO，条件是选对 benchmark）**：结构匹配（A3）与闭环漂移（D3）比 VLA 好，头对步数敏感（DreamZero 83→52、Flash-WAM 崩）给了 warm 档 VLA 上没有的 counterfactual，且 LingBot-VA/X-WAM 都是 Apache-2.0 + 单卡可跑 + 有 LIBERO/RoboCasa 权重；但 Fast-WAM（跳视频 190 ms 无损）与 GeoBoN（BoN 只 +1–2 pp）说明**在 LIBERO 上昂贵的头没有可被保护的价值**，而 Gated GeoBoN / FFDC / C³ache / CheckVLA 已把门、周期复用、离线分位标定各占一角。能否成篇取决于三天门：步数悬崖是否在仿真里存在、oracle-BoN 是否 ≥ 3 pp、跨 episode 检索是否单调可标定。
