# 跨领域迁移调研：JEPA / latent-predictive 世界模型的推理时 latent-space MPC（2025-01 之后）

> 调研日期 2026-09-16。承接 `logs/cache_transfer/world_models_planning.md`（下称"前文"）的子族 (i)「决策时规划」，只推进其中 **JEPA / latent-predictive 世界模型 + CEM/MPPI/梯度 MPC** 这一支；TD-MPC2、Cosmos Policy、AdaReP/AOR/DreamLedger 等前文已覆盖的内容不再重复，只在需要对照时点名。
> 规则同简报：数字一律给 URL；文献没有的标"估计"或"不确定"，不编。本文只写本文件，不跑实验、不碰 git。

## 0. 一句话结论

这一支的**结构匹配度是所有候选里最高的**（冻结编码器一次前向 = 前缀，CEM 的 (μ,σ) 迭代 = 带中间状态的头，c_pre/c_0 ≈ 0.01–1%），而且 **2026 年已有人做了"检索初始化 CEM"（IMWM）却没有人做"检索 → 少跑迭代 → 离线标定风险"**；短板是 (1) 单次决策成本在可跑仿真的模型上分布极端：LeWM 族 ≈ 1 s 量级（实现相关，最高报到 54 s），DINO-WM 族 47–144 s，V-JEPA 2-AC 16 s 但只有真机与未验证的社区 LIBERO 微调；(2) 傻基线（砍 CEM 迭代）的落差在文献里只有间接证据，且 DA-LeWM 的 CEM 分阶段 Spearman 显示后期迭代接近噪声，必须第一天实测；(3) 摊销式规划器（GC-IDM 1.5M 参数、100–130× 便宜、7/8 设定追平 CEM）是比"减迭代"更狠的傻基线。

**推荐：top-1 DINO-WM 族（DINO-WM 官方 + Meta jepa-wms 的 Metaworld/RoboCasa 版）、top-2 LeWM 族（OGBench-Cube/Push-T，三天杀手门的首选平台）、top-3 V-JEPA 2-AC（只做成本论证与真机故事，不做前沿）。** 详见 §8。

## 1. 候选清单（按"权重 + 仿真 benchmark + 昂贵循环"三条筛）

先给总表，再逐个展开。"循环体占比"= CEM/梯度迭代占单次决策 wall-clock 的比例；"hit 地板"= 编码器一次前向 + kNN 相对全算的比例。

| # | 系统 | 日期 / venue | 编码器 + predictor | 规划器与预算（样本×轮×horizon） | 每决策耗时（GPU） | 循环体占比 / hit 地板 | Benchmark（SR） | 权重 / 许可证 | 我们硬件可行性 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **DINO-WM**（Zhou et al.） | ICML 2025；arXiv 2411.04983 | DINOv2 ViT-S 冻结（196 token）+ 19M ViT predictor（depth 6） | CEM；论文 Table 10 为 100×10，官方 README 默认 `opt_steps=30`，LeWM 系复现用 300×30×H5 | 论文自报 CEM ≈ 53 s/plan（A6000，0.014 s/batch-32 步）；LeWM 报 47 s/plan；Slot-MPC 实测 **144 s/决策**（A6000，Meta-World/robosuite 配置） | ≈ 99.9% / ≈ 0.02%（DINOv2-S 一帧 ≈ 10 ms，估计） | PushT 0.90、PointMaze 0.98、Wall 0.96、Reach 0.92；OGBench-Cube 86（Fast-LeWM 表） | OSF ckpt（PushT/PointMaze/Wall）+ LeWM 作者 Google Drive 基线 ckpt（含 Cube/Reacher/Two-Room）；代码 MIT | 4090 可跑；一个 50 集前沿点 ≈ 1.5–3 h（LeWM 协议每集 2–4 次规划，估计） |
| 2 | **JEPA-WMs**（Meta，Terver et al.） | TMLR 2026-05；arXiv 2512.24497 | DINOv2-S（仿真最佳）或 DINOv3-L / V-JEPA2-L（300M，冻结）+ predictor depth 6–12、dim 384/1024、AdaLN+RoPE | CEM/NeverGrad/Adam/GD；Table 10：Metaworld N=300, H=6, m=3, K_e=10, J=15, W_p=2, f=5；RoboCasa N=300, H=3, m=1, J=15 | **未报告**；按 27,000（MW）/ 13,500（RC）次 predictor 前向估计 5–15 s/plan（ViT-S）、ViT-L 再 ×3–5（估计，不确定） | ≈ 99% / < 1%（估计） | MW-Reach 58.2、MW-Reach-Wall 41.6、RoboCasa-Reach 25.4 / -Place 30.7、DROID 48.2；含 DINO-WM 与 V-JEPA-2-AC(fixed) 基线数字 | HF `facebook/jepa-wms`（Metaworld/PointMaze/Push-T/Wall/DROID/RoboCasa ckpt + DINO-WM 基线 + V-JEPA-2-AC(fixed)）；**CC-BY-NC 4.0** | 可跑；RoboCasa 评测集只有 16 条 teleop 轨迹、Metaworld 只评 2 个任务，SR 低（25–58%）→ ε 容差故事噪声大 |
| 3 | **LeWorldModel (LeWM)** | 2026-03；arXiv 2603.19312 | ViT-Tiny 5M（单 192-d token）+ 10M transformer predictor（AdaLN），总 15–18M | CEM 300×30×H5（PushT）/ 300×10（其它），elites 30，frame-skip 5 → 每次规划 45,000 次 predictor 前向 | 论文："< 1 s"、比 DINO-WM 快 48×；LEAP 复现 1.20 s/trial；**Latent Geometry 复现 10.3–44 s/plan（L4/A100）**；Fast-LeWM 复现 CEM solve 54.4 s（4090，口径未说明）→ 实现相关，必须实测 | ≈ 99% / ≈ 0.5%（ViT-Tiny 一帧 < 5 ms，估计） | Two-Room 87、Reacher 86、PushT 96、OGBench-Cube 74（Fast-LeWM Table 1）；LeWM 自报 PushT 比 PLDM +18% | HF `quentinll/lewm-{pusht,cube,tworooms,reacher}` + 数据集；代码 MIT（依赖 stable-worldmodel） | 最易跑：Hi-LeWM 报 50 集一档 5–58 min（A100/H100） |
| 4 | **Fast-LeWM** | 2026-06；arXiv 2606.26217 | LeWM 编码器 + action-prefix causal transformer（3 层）+ 6 层残差 MLP predictor，17.9M | 同 LeWM CEM 协议；一次前缀编码并行预测 H 个 horizon（model calls 5→1） | Two-Room 4090：dynamics 31.4→8.0 s，CEM solve 54.4→28.3 s | 同上 | Two-Room 98、Reacher 88、PushT 96、Cube 80（avg 85.8→90.5） | GitHub `Yuntian-Gao/Fast-LeWorldModel`；CC-BY 4.0（论文） | 可跑；但它把我们的 miss 成本砍半，属"头内加速"正交项 |
| 5 | **IMWM** | 2026-06；arXiv 2606.01626 | 冻结 LeWM + 单独训练的 intuition model | CEM 300×30，elites 30，**Retrieval Initialization**：3,000 条 demo/任务的库，key = cos([z_t; z_g]) ，用检索到的动作 chunk 做 CEM 初始均值（σ=1）；**不减迭代** | 未报告（"intuition 成本相对 rollout 很小"） | 同 LeWM | Two-Room 87.7→99.2、Reacher 83.2→83.8、Push-T 89.8→92.7、**OGBench-Cube 66.2→94.7** | 承诺匿名 artifact（代码/ckpt/检索库）；CC-BY 4.0 | 可复现（LeWM 上加检索） |
| 6 | **HWM**（Hierarchical Planning with Latent WMs） | 2026-04；arXiv 2604.03208 | DINO-WM 25M（低层）+ 75M（高层） | 两级 CEM 各 1200×20，低层 H=5、每 5 步重规划 | 未报告绝对值；"比 flat 少 3× 规划算力" | 同 DINO-WM | Push-T d=25/50/75：flat 84/55/17 → HWM 89/78/61 | GitHub `kevinghst/HWM_PLDM`（README 只给 PLDM/Diverse Maze 最小实现）；CC-BY 4.0 | DINO-WM 成本 ×2 |
| 7 | **V-JEPA 2-AC** | 2025-06；arXiv 2506.09985 | ViT-g 1B 冻结（16 帧×256²）+ 300M predictor（24 层） | CEM 800×10，**H=1**，L1-ball 0.075 | **16 s/action（RTX 4090）**；Cosmos 对照 4 min | ≈ 99% / ≈ 1%（编码器 16 帧 clip：jepa.cpp 报 27.8 ms/clip on RTX 4500 Ada；估计 4090 上 ≤ 0.2 s） | 仅真机 Franka：reach 100、grasp cup 65、box 25、pick-place cup 80、box 65 | `dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`；代码 MIT | 单卡可跑，但 LIBERO 一集 200–500 步 × 16 s = 1–2 h/集，前沿不可行；只能做小网格 |
| 8 | **Slot-MPC** | 2026-05；arXiv 2605.14937 | SAVi 4 slot×128 + 4 层 transformer predictor | **梯度 MPC 3 次迭代、单候选**，policy 热启动 + 上一步 shifted mean；MPPI 版 64×5 elites 16 | 梯度版 0.42 s（MW）/ 0.48 s（robosuite）；MPPI 版 4.22 / 5.19 s；DINO-WM 对照 144 s（A6000） | 梯度版循环只有 3 步 → ladder 太短；MPPI 版 5 轮可做 | MW Button 0.64 / Lever 0.52；robosuite Stack 0.42 / Square 0.22（50 集）；DINO-WM 在同配置 0.00 | 代码 slot-mpc.github.io；CC-BY-SA 4.0 | 可跑；但 SR 低、循环短 |
| 9 | **Pixels to Proofs / SLS²** | 2026-06；arXiv 2606.15594 | 自研 tiny JEPA（5–48-d latent）+ MLP 动力学 | GPU 并行 SLS robust MPC + conformal 误差管 | 0.36 s（Reacher）/ 0.65 s（Rope）每步（4090） | 循环 = 凸/鲁棒 MPC 求解，热启动天然但非 CEM/去噪 | Reacher 83.5、Cube 91.5、Push-T 51.5、Rope 93.8 | GitHub `trustworthyrobotics/SLS-squared`；CC-BY 4.0 | 可跑；结构与我们"采样式头"不同 |
| 10 | **LEAP** | 2026-09；arXiv 2609.03294 | 冻结 LeWM 官方 ckpt + 3 层 MLP proposal | L-BFGS 10 轮（Push-T 32 候选、Cube 1 候选），H=5 | LEAP 1.28 s/trial vs LeWM+CEM 1.20 s/trial（同 GPU，未指明型号） | — | LeWM+CEM 77.5 → LEAP 94.8 avg（Cube 62→100） | 未给仓库；CC-BY 4.0 | 可复现 |
| 11 | **GC-IDM**（Latent Geometry Beyond Search） | 2026-05；arXiv 2605.08732 | 冻结 LeWM + 1.5M MLP 逆动力学 | **无搜索**，直接出动作 | GC-IDM 86–399 ms/plan vs CEM 10,317–43,969 ms/plan（L4；Cube 用 A100） | — | Two-Room 100 vs 84、Push-T 84.2 vs 82.5、Cube 98.7 vs 67.0、Reacher 99.7 vs 70.3（n=200） | GitHub `hdnndh/Latent-Geometry-Beyond-Search-...`；CC-BY 4.0 | 这是 k=0 傻基线的"学习版"，审稿人的主炮 |
| 12 | **RP1**（Reinforced Planning） | 2026-08；arXiv 2608.18669 | 冻结 LeWM/PLDM + 学习型残差 refiner | 8 轮学习式精炼，99 次 rollout（vs CEM 9,000） | 单规划器 13× 快、50 并发 67× | — | TwoRoom 98–100、Reacher 88.7–98.7、Cube 75–89 | 未给仓库；CC-BY-NC-SA 4.0 | 同上，摊销威胁 |

其它相关但不入选（原因）：
- **Being-H0.7**（arXiv 2605.00078）：VLA 里插 latent query 做"future-aware"推理，部署时 prior 分支直接出动作，**没有推理时搜索循环** → A=0，不属于本线。https://arxiv.org/abs/2605.00078
- **VLA-JEPA / JEPA-VLA / AtomVLA**：JEPA 只用作 VLA 预训练目标，推理是策略直出。https://arxiv.org/abs/2602.10098 ；https://arxiv.org/abs/2602.11832 ；https://arxiv.org/abs/2603.08519
- **Foresight**（arXiv 2606.23085）：把 V-JEPA 2-AC 的 predictor 在 LIBERO-Long / ManiSkill-Long / BEHAVIOR-1K rollout 上从头训（H200），但只做失败检测、无规划、无权重。https://arxiv.org/html/2606.23085
- **RLA-WM**（arXiv 2605.07079）：ManiSkill 上的 flow-matching latent-action 世界模型（30 步 Euler），用作 RL 模拟器而非决策时规划。https://arxiv.org/html/2605.07079
- **LaDi-WM**（arXiv 2505.11528）：LIBERO-LONG/CALVIN 的 latent diffusion 世界模型 + 迭代精炼的扩散策略（LIBERO-LONG +27.9%），是"去噪循环"结构但非 MPC 搜索，权重不明。https://arxiv.org/abs/2505.11528
- **EV-WM**（arXiv 2606.13053）：LIBERO 上采样式候选 + 事件验证器，无代码。https://arxiv.org/abs/2606.13053
- **RC-aux**（arXiv 2605.07278）：LeWM + reachability 辅助目标，**有 LIBERO-Goal 扩展**，代码 `Guang000/RC-aux`，数字未抽到。https://arxiv.org/abs/2605.07278
- **DA-LeWM**（arXiv 2608.18746）、**VLWM**（2606.21775）、**Hi-LeWM**（2607.12547）、**SAGE**（2607.17973）、**Causal-JEPA**（2602.11389，Push-T 88.67 vs DINO-WM 91.33，规划快 8×：673 s vs 5,763 s / 50 条轨迹，L40s）、**Sparse Imagination**（2506.01392）：都是 LeWM/DINO-WM 上的目标/结构改动或头内加速，作为 miss 成本会变小的正交项记录。
- **HaM-World**（2605.05951）：DMC 上 BC 先验热启动 CEM，非操作。https://arxiv.org/abs/2605.05951
- **stable-worldmodel**（2602.08968 / 2605.21800）：统一平台（16 环境含 Push-T、Two-Room、OGBench Cube/Scene、DMC 13 任务；DINO-WM/LeWM/PLDM/TD-MPC2 基线；CEM/iCEM/MPPI/GD/GRASP 求解器），是 LeWM 族所有后续工作的底座。https://arxiv.org/abs/2605.21800 ；https://github.com/galilai-group/stable-worldmodel

### 1.1 特别核查：V-JEPA 2-AC / DINO-WM / LeWM 有没有被迁到标准机器人仿真 benchmark

| 目标 | 结果 | 来源 |
|---|---|---|
| V-JEPA 2-AC → RoboCasa | **官方有**：JEPA-WMs 发布 "V-JEPA-2-AC(fixed)" ckpt，在 RoboCasa（DROID 相机视角、Robotiq 夹爪、16 条自采 teleop 轨迹定义 Reach/Pick/Place 子任务）上 Reach 16.2 / Place 33.1；CC-BY-NC | https://github.com/facebookresearch/jepa-wms ；https://arxiv.org/pdf/2512.24497 |
| V-JEPA 2-AC → LIBERO | **社区有、未验证**：HF `muniker/vjepa2-ac-libero-finetuned`（≈2026-08-27 上传；`latest.pt` 11.8 GB + `params-pretrain.yaml` + `log_r0.csv`，无 model card、无评测数字、下载不计数）。另有 `MilesKimRLWRLD/vjepa2-ac-domino`、`episod/vjepa2-ac-vitg-fpc64-256-droid-tt`（64 帧 DROID 变体）。Foresight 在 LIBERO-Long 上从头训 predictor 但不放权重 | https://huggingface.co/muniker/vjepa2-ac-libero-finetuned/tree/main ；https://huggingface.co/models?search=vjepa2%20ac |
| V-JEPA 2-AC → CPU | jepa.cpp（MIT）GGUF F32/F16/Q8/Q4：Threadripper 7995WX F16 2,228 ms/16 帧 clip；RTX 4500 Ada 27.8 ms/clip；支持一图内评 K 个候选的 CEM；Q4 会让多步 rollout 错排 | https://huggingface.co/jepacpp/vjepa2-ac-vitg-GGUF |
| V-JEPA 2-AC → 其它 | D-JEPA（Apache 2.0）在 V-JEPA 2-AC predictor 上加决策对齐模块（PushT ckpt、RoboTwin）；`RobvanGastel/adapt-vjepa-world-model` 只有 Pendulum 玩具 | https://github.com/NEBULIS-Lab/D-JEPA ；https://github.com/RobvanGastel/adapt-vjepa-world-model |
| DINO-WM → Meta-World / RoboCasa | **官方有**：JEPA-WMs 发布 DINO-WM 基线 ckpt（Metaworld、RoboCasa、Push-T、PointMaze、Wall、DROID）；Slot-MPC 自训 DINO-WM 在 MW Button/Lever、robosuite Stack/Square 全 0.00（144 s/决策） | https://github.com/facebookresearch/jepa-wms ；https://arxiv.org/html/2605.14937 |
| DINO-WM → OGBench-Cube / Reacher / Two-Room | LeWM 作者放了 DINO-WM/PLDM 基线 ckpt（Google Drive）；Fast-LeWM 表：Cube 86 | https://github.com/lucas-maes/le-wm ；https://arxiv.org/pdf/2606.26217 |
| LeWM → LIBERO | 只有 RC-aux 的 "LIBERO-Goal extension"（代码有、数字未抽到）；官方 4 环境里 **OGBench-Cube 本身就是 UR5e + Robotiq 2F-85 的 MuJoCo 抓放**（5-D 动作、28-D 状态、goal image） | https://arxiv.org/abs/2605.07278 ；https://arxiv.org/abs/2410.20092 |
| 社区微调（awesome-jepa 汇总） | 截至 2026-06 未列任何 V-JEPA 2-AC / DINO-WM / LeWM 的社区微调 | https://github.com/AbdelStark/awesome-jepa |

## 2. 映射表（以 DINO-WM / LeWM 的 CEM 为主，括号给 V-JEPA 2-AC / 梯度规划器的对应）

| 我们的符号 | 本子族对应 |
|---|---|
| 共享前缀 c_pre | 冻结视觉编码器一次前向：DINOv2-S 一帧 196 token（DINO-WM）/ ViT-Tiny 单 token（LeWM）/ ViT-g 16 帧（V-JEPA 2-AC）+ 目标图编码（每集一次，可缓存） |
| tap point | z_t（和 z_g）算完、CEM 抽第一批样本之前；query 特征 = z_t 本身（IMWM 就用 cos([z_t; z_g]) 做检索 key） |
| 头（迭代生成） | CEM：J 轮 × N 条 × H 步 predictor rollout + 代价评估（LeWM 45,000 次/plan；JEPA-WMs MW 27,000；V-JEPA 2-AC 8,000）；梯度版：J 步 L-BFGS/Adam（LEAP 10 轮；train-test-gap 论文 100 步/MPC 周期） |
| 中间状态（warm start 起点） | 第 j 轮后的 (μ_{t:t+H}, σ_{t:t+H})（V-JEPA 2-AC 还带动量项）；梯度版 = 当前动作序列变量 |
| tier K = hit | 直接执行库条目的动作 chunk（LeWM 协议是整段 25 步开环执行后才重规划，hit 恰好等于"执行库里的整段"）；成本 = 编码器 + kNN |
| 中间 tier | 用库里的 (μ, σ) 做初始分布，只跑 ρ_a·J 轮（也可同时缩 N）；IMWM 已做"库均值初始化"但 J 不变 |
| tier 0 = miss | 完整 J 轮从 μ_0=0、σ_0=1 起（注意：Slot-MPC / TD-MPC2 的 miss 本身带 shifted-mean 自热启动，LeWM 族默认**没有**） |
| key 字段 | z_t（视觉 latent）、z_g（目标 latent，替代指令 embedding 做 IVF cell）、本体状态（若模型带 proprio） |
| 分数 s_t | z 空间 cosine / 负 L2，库自比留一估 μ/τ 后 tanh 标准化加权 |
| 偏差 D_a | tier a 输出动作序列（前 m 步）与从零 CEM 参考序列的标准化距离；或两者在世界模型下的终端代价差 ‖ẑ_{t+H} − z_g‖ 之差（DA-LeWM 的 plan-real Spearman 提示后者与真实结果相关性只有 0.28–0.41，需要小心） |
| 成功度量 SR | goal-reaching success（Push-T/Cube/Reacher/Two-Room 均为二值 SR）；Metaworld/RoboCasa 子任务 SR |
| 成本模型 | c_a = c_enc + a 档剩余轮数 × N × H × 单步 predictor + 代价评估；wall-clock 版 LeWM 族在 1–50 s 之间实现相关，必须实测 |
| 闭环性 | receding-horizon：LeWM 族执行整段 25 步再重规划、JEPA-WMs 每 m=1–3 个动作重规划、V-JEPA 2-AC 每个动作重规划；开环误差随 horizon 累积（Fast-LeWM §1、Hi-LeWM "staged execution 只在序列仍可靠时有效"） |
| 状态门 | 同集内连续规划的 z_t 连续变化 → 上一次分数预测下一次命中；对应物 = TD-MPC2/Slot-MPC 的 shifted-mean 自热启动（单步版） |

## 3. 打分表 A–G（按 top-1 DINO-WM 族打；括号给 LeWM 族 / V-JEPA 2-AC 的差异）

| 项 | 分 | 依据（一句） | URL |
|---|---|---|---|
| A 结构匹配 | **3** | 冻结编码器一次前向 + CEM (μ,σ) 迭代头，tap point 天然、query 白拿（IMWM 直接用 cos([z_t;z_g]) 检索）；三族全同构 | https://arxiv.org/html/2411.04983 ；https://arxiv.org/html/2606.01626v1 |
| B 成本余量 | **2**（LeWM 2、V-JEPA 2-AC 3） | c_pre/c_0：DINO-WM ≈ 10 ms / 47–144 s，V-JEPA 2-AC ≈ 0.2 s / 16 s，远优于 VLA 15%；但"减迭代掉点"只有间接证据：JEPA-WMs 说 SR 对规划超参"非常敏感"，HWM/Latent Geometry 有 budget-vs-SR Pareto 但没抽出数字，而 DA-LeWM 的 CEM 分阶段 Spearman（random 0.40 → mid 0.23 → elite 0.04）暗示后期迭代接近噪声 → 傻基线可能很强 | https://arxiv.org/pdf/2512.24497 ；https://arxiv.org/html/2604.03208v1 ；https://arxiv.org/html/2605.08732v1 ；https://arxiv.org/html/2608.18746 |
| C 局部性 / 命中潜力 | **2** | 评测 start/goal 对从离线数据集采样（Push-T/RoboCasa/DROID）、同任务反复；IMWM 3,000 条 demo 的检索库把 Cube 从 66.2 拉到 94.7，说明近邻在 z 空间有效；跨 episode 命中率**无文献数字（不确定）** | https://arxiv.org/html/2606.01626v1 ；https://arxiv.org/pdf/2512.24497 |
| D 容忍度与闭环 | **3** | 二值 SR + receding horizon；开环 rollout 误差随 horizon 累积（Fast-LeWM 把 25→50 步漂移当核心问题）；Hi-LeWM 显示预算好的子目标序列过期即失效 | https://arxiv.org/pdf/2606.26217 ；https://arxiv.org/html/2607.12547v1 |
| E RIT 可用性 | **2** | shadow 可离线构造（记录 z_t、参考 (μ,σ)、执行动作；事后检索算 D_a）；单调性与"一个 δ 切所有 cut"未验证；gate 对应物是同集内 shifted-mean（Slot-MPC/TD-MPC2） | https://arxiv.org/html/2605.14937 |
| F novelty 地形 | **2** | 已有：IMWM 检索初始化（不减算力、无风险标定）、Slot-MPC/TD-MPC2 自热启动、LEAP/GC-IDM/RP1 摊销、Fast-LeWM/Causal-JEPA/Sparse Imagination 头内加速、AdaReP（前文）；**没人做"检索 → 少跑迭代 → 离线单 δ 标定 → 按预算反解"**，比 TD-MPC2 那支（前文 F=1）空一些 | https://arxiv.org/html/2606.01626v1 ；https://arxiv.org/html/2609.03294 ；https://arxiv.org/html/2608.18669 |
| G 我们做得动 | **2**（LeWM 3、V-JEPA 2-AC 1） | DINO-WM：MIT 权重 + Push-T/Cube，50 集/点 ≈ 1.5–3 h（估计）；jepa-wms Metaworld/RoboCasa 有 ckpt 但 CC-BY-NC、RoboCasa 评测集 16 条、无耗时数据；LeWM 50 集/档 5–58 min；V-JEPA 2-AC LIBERO 1–2 h/集不可做前沿 | https://github.com/gaoyuezhou/dino_wm ；https://github.com/facebookresearch/jepa-wms ；https://arxiv.org/html/2607.12547v1 |

合计 **16 / 21**（DINO-WM 族）；LeWM 族 16（B、G 互换）；V-JEPA 2-AC 14（G=1）。比前文 TD-MPC2 路线（15）高在 A 与 F，低在 B 的证据完整度。

## 4. "减迭代次数"傻基线：文献里有什么

1. **有 budget 扫描但没抽出可引用的 ladder 数字**：Latent Geometry 附录 E.4 在 4 个环境扫 num_samples ∈ {30,100,300,1000} × n_steps ∈ {2,5,10,30}（60–30,000 rollouts/plan）只给 Pareto 图，结论是"没有任何 CEM 配置同时比 GC-IDM 又快又准"（https://arxiv.org/html/2605.08732v1 ）；HWM 附录 D 扫 samples 150–1500 × iterations 10–40，flat 规划器在 d=50/75 随预算下降（https://arxiv.org/html/2604.03208v1 ）；IMWM 扫 T ∈ {15,30,60}，只说"增益不随 T 关闭"（https://arxiv.org/html/2606.01626v1 ）。
2. **偏"傻基线会很强"的证据**：DA-LeWM 测 CEM 各阶段候选代价与真实结果的 Spearman，random 阶段 +0.40、mid +0.23、elite +0.04（https://arxiv.org/html/2608.18746 ）——即最后几轮 elite 排序几乎是噪声，砍掉可能不掉 SR；JEPA-WMs 在 DROID 上用 15 轮而非 V-JEPA 2-AC 的 10 轮 + 动量，"没发现动量有差别"（https://arxiv.org/pdf/2512.24497 ）。
3. **偏"傻基线会掉"的证据**：V-JEPA 2-AC vs Cosmos 80 样本（不同模型，非同模型减预算，弱证据）；JEPA-WMs 明说"SR 对规划超参非常敏感，需要在 DINO-WM 上仔细网格搜"；Slot-MPC 发现离线覆盖有限时 MPPI 64×5 在 robosuite 上 0.00 而梯度 3 步 0.42（说明搜索方式比轮数更关键）（https://arxiv.org/html/2605.14937 ）。
4. **更狠的傻基线 = 摊销**：GC-IDM（1.5M MLP，20 min 训练）100–130× 便宜且 7/8 设定 ≥ CEM；RP1 99 次 rollout 追平 9,000；LEAP 用 proposal + 10 轮 L-BFGS 把 Cube 从 62 拉到 100。与 VLA 的 k=1 不同，这里"不搜索"不是条件均值退化，而是**换了一个学习型规划器**——审稿人会问"为什么不直接训 GC-IDM"。我们的回答只能是 training-free + 跨任务库 + 风险可标定，且必须在同一 (IR, SR) 图上把 GC-IDM 画进去。
5. 与 VLA 的关键区别：本子族的 miss 是**从 μ_0=0、σ_0=1 的高斯起搜**，减到 1–2 轮 = 几乎随机采样的 elite 均值，理论上比"从检索到的真实 plan 起 1–2 轮"差得多（IMWM 只改初值就 +28.5 pp）——这是 warm 档有 counterfactual 的结构性理由，但 §4.2 的 Spearman 证据要求第一天就实测 ladder。

## 5. 预期收益（尽量给文献数字）

- **hit 地板**：编码器 + kNN ≈ 0.02–1% c_0（DINO-WM/V-JEPA 2-AC），LeWM ≈ 0.5%；比 VLA 的 15% 低 1–3 个量级。这是本线最硬的卖点。
- **warm 档的"零成本上限"**：IMWM 的检索初始化在 J 不变时 Cube +28.5 pp、Two-Room +11.5 pp（https://arxiv.org/html/2606.01626v1 ）——说明库初值能把同预算 SR 抬高；我们要证的是把 J 砍到 ρJ 时 SR 不低于原 miss。
- **自热启动基线**：Slot-MPC 去掉 policy 初始化后 MW 0.64→0.32、robosuite 0.42→0.00（https://arxiv.org/html/2605.14937 ）；TD-MPC2/AdaReP 在前文（IR≈0.45–0.5、ε≈0）。跨 episode 库能再压多少：**无文献数字（不确定）**。
- **IR 可达范围（估计）**：若 hit 率 30–50%（同任务 start/goal 从数据集采样的评测协议下，估计）且 warm 档跑 1/3 迭代，IR ≈ 0.5×(1/3) + 0.2×1 ≈ 0.35–0.45；成本地板远低于 VLA，所以 IR 曲线能一路画到 0.05 而不是 0.15 截断。
- **SR 损失**：库失配风险由 RIT 控；OGBench-Cube 是唯一 SR 处在 66–86 的"有余量"环境（Two-Room/Push-T 已 90+，Metaworld/RoboCasa 25–58 太低）。

## 6. novelty 地形与最强反驳

按距离排序：
1. **IMWM**（2026-06）：检索初始化 + 混合代价 + 可靠性门，挂在 LeWM 上。= 我们的 warm 档初值 + 一个 gate 的雏形；**不减迭代、不标定风险、不做 (IR,SR) 前沿**。https://arxiv.org/html/2606.01626v1
2. **Slot-MPC / TD-MPC2**：policy 先验 + shifted-mean 自热启动。= 单步 stateful 复用。https://arxiv.org/html/2605.14937
3. **LEAP / GC-IDM / RP1**：训练一个 proposal / 逆动力学 / refiner 替代或初始化搜索。= 学习型 k=0/k=1。https://arxiv.org/html/2609.03294 ；https://arxiv.org/html/2605.08732v1 ；https://arxiv.org/html/2608.18669
4. **Fast-LeWM / Causal-JEPA / Sparse Imagination**：把每次 rollout 做便宜（3.9×/8×）。正交，但缩小 miss 成本。https://arxiv.org/pdf/2606.26217 ；https://arxiv.org/html/2602.11389
5. **AdaReP / AOR / DreamLedger**（前文）：自身 plan 复用 + 触发重规划。
6. **DA-LeWM**：诊断 CEM 各阶段与真实的相关性——正好是我们 D_a 可信度的反面证据来源。https://arxiv.org/html/2608.18746

**最强反驳**：
- "GC-IDM 1.5M 参数、20 分钟训练、100× 便宜、还更准；你们的缓存系统在同一张图上在哪？"（必须画进去，并把 GC-IDM 当作另一种 tier 0 候选正面处理）
- "IMWM 已经做了检索初始化；你们只是把它的 J 砍了并加了个分位回归。"（回答：IMWM 没有成本-风险权衡，没有 ladder，没有闭环漂移分析；且我们 training-free、不需要 intuition model）
- "LeWM 一次规划 1 s，省它干嘛？"（回答：按 DINO-WM 47–144 s / V-JEPA 2-AC 16 s 立成本模型；LeWM 只当快速验证平台）
- "DA-LeWM 显示后期 CEM 迭代是噪声，减迭代就够了。"（这就是杀手门第一天要正面回答的）

## 7. 首个实验方案：三天杀手门（LeWM × OGBench-Cube + Push-T，单张 4090；通过后升级 DINO-WM / jepa-wms）

平台选择理由：LeWM 有 MIT 权重 + HF 数据集 + stable-worldmodel 求解器（CEM/iCEM/MPPI/GD），Cube 是真机器人臂抓放，50 集一档 5–58 min；三天内能把三条基线全画出来。DINO-WM（47–144 s/plan）留作"贵的模型"验证。

**Day 1 — 成本拆分 + 减迭代 ladder（傻基线）**
- 实测 4090 wall-clock：编码器一帧、目标图编码、每轮 CEM（300 样本 × H5）、代价评估、kNN；顺手对齐 LeWM "<1 s" vs 复现 10–54 s 的口径（batch 是否并行、是否走了 stable-worldmodel 的慢路径）。
- ladder：n_steps ∈ {30, 15, 8, 4, 2, 1}（N=300），再 N ∈ {300, 100, 30}（n_steps=30）；Cube 与 Push-T 各 50 集（沿用 LeWM 协议：25 步整段执行后重规划）。
- **Kill-1**：若 n_steps=4 在两环境 SR 都在 n_steps=30 的 2 pp 内 → warm 档没有 counterfactual（同 VLA k=1），LeWM 线写 NO，改测 DINO-WM 再判一次。

**Day 2 — 自热启动 + hit 地板**
- 自热启动：用上一次规划的 μ 平移 5 步作初值，跑 ρ·30 轮，ρ ∈ {1/3, 1/6, 1/15}；再加 "0 轮 = 直接执行上一段 plan 的续段"。
- hit 地板：库 = 200 集从零规划成功轨迹的 (z_t, z_g, μ, σ, 执行动作)；评测时每步只做编码 + kNN + 执行库动作段（0 轮）。记 IR_wall、IR_flops、SR。
- **Kill-2**：wall-clock hit 地板 > 30%（说明实现是 launch-bound，故事只能按 FLOPs/DINO-WM 讲）。

**Day 3 — shadow + RIT + 一个闭环点**
- shadow 50 集：记录 z_t、参考 (μ,σ)；事后检索算 D_hit、D_warm(ρ=1/3, 1/6)；检查 q_a(s) 单调、曲线有序、上一步分数预测下一步命中的 AUROC。
- 闭环：K=2（hit + warm 1/6），δ 扫 4 点 × 50 集，与 Day 1/2 的三条基线同图；把 GC-IDM 的 (86–399 ms, SR) 点按文献画进去。
- **Kill-3**：库前沿在同 SR 下的 IR 与自热启动差 < 5 pp；或 q_a(s) 不单调。

通过后第二阶段：DINO-WM（LeWM 作者 Drive 里的 Cube/Push-T 基线 ckpt，47 s/plan）复跑同一套，得到"贵模型"上的绝对节省；再选 jepa-wms Metaworld（Reach/Reach-Wall，SR 58/42）验证第三个环境。V-JEPA 2-AC 仅做成本模型引用（16 s/action、编码器 < 0.2 s），不跑前沿。

## 8. 结论与 top-3

| 排名 | 系统 | 理由 | 风险 |
|---|---|---|---|
| **1** | **DINO-WM 族**（官方 MIT ckpt + LeWM Drive 基线 + Meta jepa-wms 的 Metaworld/RoboCasa 版） | 单次决策 47–144 s、hit 地板 ≈ 0.02%，公开权重覆盖 Push-T/Cube/Metaworld/RoboCasa，闭环 receding horizon | 每前沿点小时级；jepa-wms CC-BY-NC、RoboCasa 评测集 16 条、SR 25–58 |
| **2** | **LeWM 族**（HF `quentinll/lewm-*`，MIT） | 三天能画全三条基线；后继 IMWM/LEAP/GC-IDM/DA-LeWM 全在同平台，对照齐全 | 每决策 ~1 s（实现相关最高 54 s），"省它干嘛"需靠 DINO-WM 背书 |
| **3** | **V-JEPA 2-AC**（官方 ckpt + jepa-wms RoboCasa 版 + 社区 LIBERO 微调） | 16 s/action、ViT-g 编码器占比 ≈ 1%，是本线成本故事的锚 | 仿真版要么 CC-BY-NC 且 SR 16–33，要么社区无卡无数字；LIBERO 前沿 1–2 h/集不可做 |

**MAYBE（偏 GO，前提是 Day 1 的减迭代 ladder 有落差）**：结构匹配与成本余量是所有候选里最好的（16/21），IMWM 已证明检索初值在 Cube 上 +28.5 pp 却没人把它变成成本-风险 ladder；但 DA-LeWM 的 elite-stage Spearman ≈ 0 与 GC-IDM 的 100× 摊销是两把悬在 B 上的刀，必须在三天杀手门里正面回答。
