# benchmark 与现成微调版盘点：推理时做迭代规划/搜索的机器人模型 × 仿真 benchmark × 公开权重

> 调研日期 2026-09-16。承接 `logs/cache_transfer/world_models_planning.md`（子族 (i) 决策时规划）。目标：找到**不用自己训练**就能跑起来的"迭代规划/搜索型"模型 + 配套仿真 benchmark + 评测脚本，并估每次决策成本。
> 硬件前提（briefing §3 G）：1×H100 80G、1×4090、若干 A5000/1080 仿真机。
> 规则：每条结论附 URL；文献没有的数字标"估计"或"不确定"，不编。"三件齐全"= 权重 + benchmark 环境 + 评测/规划脚本都是官方或同一仓库提供。

## 0. 一句话结论

- **最省事的三件齐全组合是 latent-JEPA 一族**：LeWM（HF 权重、MIT、~1 s/plan）、DINO-WM（OSF 权重、MIT、~47 s/plan）、Meta 的 jepa-wms（HF 权重、CC-BY-NC、同时附 DINO-WM 与 V-JEPA-2-AC 的 sim 版基线），全部落在 Push-T / PointMaze / Wall / OGBench-Cube / Metaworld-Reach 这类小环境上，单卡当天可跑通。
- **TD-MPC2**（324 个 ckpt、MIT、20 ms/步）仍是"最便宜且最成熟"的采样式 MPC 平台，但 Meta-World/DMControl 是状态输入（非像素），成本地板要实测。
- **大型视频世界模型（Cosmos-Predict2 系、GE-Sim 2.0、WorldGym、Ctrl-World、NWM）权重都开，但没有一个附带"闭环规划 + sim benchmark"脚本**，只做策略评测/数据引擎；单卡 25–181 s/段，作规划头不可行。
- Dreamer 4 / Genie 3 无官方权重；Oasis 500M / Matrix-Game 2.0 有权重但是 Minecraft/GTA 视频交互，无机器人 benchmark。
- 2026-08 的 ARC-Bench 直接审计了 jepa-wms 官方 ckpt，结论"**降低 replanning 频率成功率崩塌**"——这既是对我们 hit 档的警告，也是 warm 档存在理由的文献证据（见 §4）。

## 1. benchmark 清单（2025–2026 世界模型/规划器常用）

| benchmark | 任务数 | 多峰 / 长程 | 公开的**规划型**（非纯前馈）baseline 带 ckpt | URL |
|---|---|---|---|---|
| **Push-T**（gym-pusht / DINO-WM 版 pusht_noise） | 1 任务（多目标） | 接触多峰（推 T 块可绕左/右）；短程 | Diffusion Policy 官方 ckpt；DINO-WM（OSF）；LeWM（HF）；jepa-wms（HF，含 DINO-WM/V-JEPA-2-AC 基线）；stable-worldmodel `swm/PushT-v1` | https://github.com/real-stanford/diffusion_policy ；https://github.com/gaoyuezhou/dino_wm ；https://huggingface.co/quentinll/lewm-pusht ；https://huggingface.co/facebook/jepa-wms |
| **PointMaze / Wall / Two-Rooms / Diverse Maze**（DINO-WM、PLDM 自制导航环境） | 各 1 环境，随机起终点 | 导航多解（绕墙）；non-greedy 规划 | DINO-WM（PointMaze/Wall）；jepa-wms（pointmaze/wall）；LeWM（tworooms）；PLDM（Two-Rooms/Diverse Maze，代码+数据，ckpt 见 LeWM 的 Google Drive 基线套件）；HWM_PLDM（Diverse Maze） | https://github.com/vladisai/PLDM ；https://github.com/kevinghst/HWM_PLDM ；https://huggingface.co/quentinll/lewm-tworooms |
| **OGBench**（visual-cube/scene/puzzle 等） | 8 类环境、85 数据集，每数据集 5 个评测 goal；有 64×64 像素版 | cube-double/scene/puzzle 长程组合；单 cube 短程 | LeWM `lewm-cube`（OGBench-Cube）；stable-worldmodel `OGBCube`/`OGBScene`；IMWM 在 OGBench-Cube 94.7%（**无代码**） | https://github.com/seohongpark/ogbench ；https://huggingface.co/quentinll/lewm-cube ；https://arxiv.org/abs/2606.01626 |
| **Meta-World**（MT50 / ML45） | 50 任务，稀疏 success | 单峰为主；短程 | TD-MPC2 单任务 5M ckpt（50 任务 × 3 seed，HF）+ MT80/MT30 多任务 ckpt；jepa-wms `jepa_wm_metaworld`（只评 Reach / Reach-Wall 两任务）；iVideoGPT 提供 MBRL 训练脚本（无现成规划 ckpt） | https://huggingface.co/nicklashansen/tdmpc2/tree/main/metaworld ；https://huggingface.co/facebook/jepa-wms ；https://github.com/thuml/iVideoGPT |
| **DMControl**（TD-MPC2 用 39 任务含 11 自定义） | 39 | return 度量、非 SR | TD-MPC2 单任务 ckpt；Dream-MPC 代码（基于 TD-MPC2 ckpt）；AdaReP 在 30 任务上（**无代码**） | https://github.com/nicklashansen/tdmpc2 ；https://github.com/jspieler/dream-mpc ；https://arxiv.org/abs/2606.23079 |
| **ManiSkill2 / ManiSkill3** | MS2：TD-MPC2 用 5 任务（LiftCube/PickCube/StackCube/PickYCB/TurnFaucet）；MS3：数十任务族 + SimplerEnv 已并入 | PickYCB 74 物体多样；短程 | TD-MPC2 ManiSkill2 ckpt（HF `maniskill2/`）；ManiSkill3 自带 TD-MPC2 baseline 代码但无官方预训练规划 ckpt（**不确定**） | https://huggingface.co/nicklashansen/tdmpc2 ；https://github.com/haosulab/ManiSkill |
| **MyoSuite** | 10 任务 | 高维肌肉 | TD-MPC2 ckpt | https://huggingface.co/nicklashansen/tdmpc2 |
| **robomimic**（lift/can/square/transport/tool_hang，ph/mh） | 5 任务 × 2 数据质量 | mh 数据多峰；短程 | Diffusion Policy 官方 image/low_dim ckpt（含 kitchen、block_pushing） | https://diffusion-policy.cs.columbia.edu/data/experiments/image/ ；https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/ |
| **D4RL maze2d / locomotion** | maze2d 3 尺寸；locomotion 3 环境 × 3 数据 | maze2d 多解；长程 | Diffuser 官方预训练（扩散规划器 + value）；replandiffuser（AOR）代码基于 Diffuser | https://github.com/jannerm/diffuser ；https://github.com/rainbow979/replandiffuser |
| **VP2**（robosuite 桌面 + RoboDesk，"world model 专用"控制中心 benchmark） | 11 任务类、310 实例 | 桌面推/RoboDesk 多物体；短程 | 自带 MPPI/CEM 规划实现 + FitVid/SVG'/MCVD/Struct-VRNN 预训练视频预测器；iVideoGPT 提供 VP2 robosuite/robodesk 动作条件 ckpt（HF）+ `vp/` 规划代码 | https://github.com/s-tian/vp2 ；https://github.com/thuml/iVideoGPT |
| **LIBERO**（Spatial/Object/Goal/10 各 10 任务 + LIBERO-90） | 40 (+90) | 单峰为主（我们实测 k=1 无损）；LIBERO-10 长程 | 唯一带权重的规划型 = Cosmos Policy（best-of-N，**本次排除**）；Pre-VLA（world-model rollout 预验证，183.9 ms/chunk，代码**不确定**）；MG-Select（verifier-free best-of-N，套在任意 VLA 上）；"Look Before You Leap"（MCTS 蒸馏 Q，代码**不确定**） | https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B ；https://arxiv.org/html/2605.22446 ；https://arxiv.org/abs/2510.05681 ；https://arxiv.org/abs/2607.03751 |
| **LIBERO-Plus / LIBERO-PRO** | Plus：10,030 扰动评测任务（7 维 21 子维）；PRO：4 维扰动 | 同 LIBERO | 无规划型 baseline（都是 VLA 鲁棒性评测） | https://arxiv.org/abs/2510.13626 ；https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/liberoplus_pro.html |
| **RoboCasa 2024 / RoboCasa365** | 2024：25 atomic + 75 composite；365：365 任务 / 2,500 厨房 / 600 h demo | composite 长程 | jepa-wms 用 DROID 训练的模型在 RoboCasa 自定义 Reach/Place 零样本规划（SR 25.4 / 30.7）；Cosmos Policy RoboCasa（排除）；365 榜单四家 baseline（DP/π0/π0.5/GR00T N1.5）全是前馈 | https://arxiv.org/html/2512.24497 ；https://robocasa.ai/leaderboard.html ；https://arxiv.org/pdf/2603.04356 |
| **CALVIN** | 34 技能，5 步链 | 语言条件长程链 | GE-Act（CALVIN 权重；"single-step visual planner + IDM"，非搜索）；未找到带 ckpt 的 MPC 规划器（**不确定**） | https://github.com/AgibotTech/Genie-Envisioner |
| **SimplerEnv**（Google Robot + WidowX Bridge） | 8 类任务 | 真机→仿真复现，单峰 | 未找到带 ckpt 的规划型 baseline（**不确定**）；已并入 ManiSkill3 | https://github.com/simpler-env/SimplerEnv |
| **RLBench** | 100 任务（PerAct 18） | 多阶段 | 未找到带 ckpt 的世界模型规划器；3D Diffuser Actor 等是扩散策略非搜索 | https://github.com/stepjam/RLBench |
| **world model 专用评测**：WorldModelBench、MiraBench、WorldArena、WorldSimProbe、1X World Model Challenge | — | — | 都是评"世界模型本身"（物理一致 / 动作跟随 / 乐观偏差 / PSNR），**没有闭环策略 benchmark**；1X 只有 Sampling 与 Compression 两赛道 + GENIE 式 baseline，policy-evaluation 赛道只是"planned" | https://worldmodelbench.github.io/ ；https://arxiv.org/abs/2605.29360 ；https://github.com/1x-technologies/1xgpt ；https://arxiv.org/abs/2510.07092 |

补充平台：**stable-worldmodel**（galilai-group，MIT，2026-05）把 LeWM / DINO-WM / PLDM / GCBC / GCIVL / GCIQL 与 CEM / iCEM / MPPI / Predictive Sampling / SGD / Adam / PGD 求解器统一在一个库里，环境含 PushT、TwoRoom、OGBCube、OGBScene、DMC 12 任务、Fetch 4 任务、Craftax、Atari。`pip install stable-worldmodel`，ckpt 放 `$STABLEWM_HOME`。https://github.com/galilai-group/stable-worldmodel ；https://arxiv.org/abs/2605.21800

## 2. 逐模型：有没有 benchmark 微调版权重

### 2.1 V-JEPA 2-AC
- 官方：`vjepa2-ac-vitg.pt`（ViT-g 1B 冻结编码器 + 300M 动作条件 predictor，DROID <62 h 后训练），MIT（部分文件 Apache 2.0），2025-06。**只有真机 Franka 评测**（reach 100% / grasp cup 65% / pick-place cup 80%），CEM 800 样本 **16 s/action**。无 sim benchmark 权重。https://github.com/facebookresearch/vjepa2 ；https://arxiv.org/abs/2506.09985
- **社区/官方 sim 版**：jepa-wms 仓库发布了 "V-JEPA-2-AC(fixed)"（ViT-G/16）在 Push-T / PointMaze / Wall / Metaworld 上重训的基线 ckpt（CC-BY-NC 4.0）。论文只在 RoboCasa/DROID 列了它的数（Robocasa-Reach 16.2、Place 33.1、DROID Action Score 42.9）。https://huggingface.co/facebook/jepa-wms ；https://arxiv.org/html/2512.24497
- 2026-08 ARC-Bench 用 V-JEPA 1/2 编码器替换 DINOv2 做受控实验，结论：冻结 latent 的动作可排序性系统性失败。https://arxiv.org/abs/2609.05461

### 2.2 DINO-WM
- 官方 ckpt：PointMaze / PushT / Wall（OSF 链接在 README），Rope/Granular 只有数据（需 PyFleX）。MIT。规划 CEM `opt_steps=30`, `n_evals=5`, `goal_H=5`。https://github.com/gaoyuezhou/dino_wm ；https://arxiv.org/abs/2411.04983
- 成本：LeWM 项目页实测 DINO-WM 一次 plan ≈ **47 s**（LeWM ≈ 1 s，48×）；Sparse Imagination（ICLR 2026）报 full-patch 规划时间 PointMaze 184 s / Wall 79 s / PushT 173 s / BlockPush 297 s（单位按原文"s/iter"，口径**不确定**），50% patch dropout 平均省 40.5%。https://le-wm.github.io/ ；https://arxiv.org/html/2506.01392v2 ；https://github.com/AlexP210/sparse_imagination
- jepa-wms 另发一套 DINO-WM 基线（DINOv2 ViT-S/14，5 环境），成功率 Maze 81.6 / Wall 64.1 / Push-T 66.0 / MW-Reach 44.8 / MW-Reach-Wall 35.1。https://arxiv.org/html/2512.24497

### 2.3 LeWorldModel（LeWM）
- 权重：HF `quentinll/lewm-{pusht,cube,tworooms,reacher}`（weights.pt + config.json），MIT；Google Drive 另有 PLDM / LeJEPA / IVL / IQL / GCBC 全套基线。~15M（ViT-Tiny 5M + 6 层 predictor 10M，192 维单 token）。2026-03-13（v3 2026-06-03）。https://github.com/lucas-maes/le-wm ；https://huggingface.co/quentinll/lewm-pusht ；https://arxiv.org/abs/2603.19312
- 规划：CEM 300 样本 × 30 iter（PushT；其它环境 10 iter）× top-30 elites，H=5 宏步（frame-skip 5 ⇒ 25 环境步），**执行完整序列再重规划**；一次 plan **< 1 s**。https://arxiv.org/html/2603.19312
- 评测数字：论文只给相对结论（PushT 比 PLDM +18 pp、与 DINO-WM 持平；Cube 略低于 DINO-WM；Two-Room 低于 PLDM/DINO-WM；Reacher 优于 PLDM）。IMWM 论文把 LeWM+CEM 基线量化为 Two-Room 87.7 / OGBench-Cube 66.2 / Push-T 89.8 / Reacher 83.2。https://arxiv.org/html/2606.01626v1
- 评测命令：`python eval.py --config-name=<task>.yaml policy=<ckpt>`；依赖 stable-worldmodel。

### 2.4 PLDM
- 代码 + 自制数据集（Two-Rooms、Diverse Mazes），MIT，2025-02；官方仓库无 ckpt 下载链接（**不确定**），但 LeWM 的 Google Drive 基线套件含 PLDM 在 PushT/Cube/TwoRooms/Reacher 的 ckpt，stable-worldmodel 也有实现。https://github.com/vladisai/PLDM ；https://latent-planning.github.io/ ；https://arxiv.org/abs/2502.14819

### 2.5 HWM（Hierarchical Planning with Latent World Models）
- 2026-04；在 V-JEPA 2 / DINO-WM / PLDM 上做两层 MPC，真机 Franka pick-place 70% vs 单层 0%，仿真"最多省 3× 规划算力"。**官方只放 PLDM/Diverse Maze 的最小实现**，无新 ckpt。https://github.com/kevinghst/HWM_PLDM ；https://kevinghst.github.io/HWM/ ；https://arxiv.org/abs/2604.03208
- 同类：Hi-LeWM（"Mind the Gap"，2026-07，PushT/Cube，无代码）。https://arxiv.org/abs/2607.12547

### 2.6 IMWM（Intuition Models Complement World Models）
- 2026-06-01；LeWM 骨干 + 检索初始化（|R|=3,000 demo/任务，cos 检索 start/goal key，取回的 action chunk 作 CEM 高斯中心 σ=1）+ 混合代价 + reliability gate；CEM 300×30 同预算。Two-Room 99.2 / Cube 94.7 / Push-T 92.7 / Reacher 83.8。**无公开代码/权重**（附录只写"release plan"）。https://arxiv.org/abs/2606.01626 ；https://arxiv.org/html/2606.01626v1
- 对我们的意义：它的 Retrieval Initialization 就是我们的 warm 档（库条目 → 规划器初始分布），但只有一档、无 RIT、无跨 episode 状态门。

### 2.7 TD-MPC2
- HF `nicklashansen/tdmpc2`（MIT）：目录 `dmcontrol/`、`metaworld/`（mw-<task>-{1,2,3}.pt，50 任务 × 3 seed）、`maniskill2/`（5 任务）、`myosuite/`（10 任务）、`multitask/`（MT30 / MT80，1M/5M/19M/48M/317M）；共 324 个 ckpt。评测 `python evaluate.py task=<task> checkpoint=<pt>`；317M 评测需 ≥24 GB 显存。Meta-World 依赖 MuJoCo 2.1.0 + gym==0.21.0（旧版锁定）。https://huggingface.co/nicklashansen/tdmpc2 ；https://github.com/nicklashansen/tdmpc2 ；https://arxiv.org/abs/2310.16828
- 成本：MPPI 512 样本 × 6 iter × H=3 = 9216 次头评估/步；实测 **20.83 ms/步**（RTX 4090，Dream-MPC 论文）；policy-only ≈ 规划的 1/6。https://arxiv.org/abs/2605.04568
- 配套代码：Dream-MPC（ICML 2026，梯度 MPC，15 次评估追平）https://github.com/jspieler/dream-mpc ；AdaReP（缓存 rollout + 自适应重规划，DMC 30 任务省 54.5% NFE）**无代码** https://arxiv.org/abs/2606.23079
- 注意：输入是状态向量（非像素），"前缀"只是 2 层 MLP，c_pre/c_0 按 FLOPs ≈ 0，wall-clock 需实测。

### 2.8 Diffusion Policy
- 官方 ckpt（MIT）：`image/`：pusht、lift/can/square/transport（ph+mh）、tool_hang_ph；`low_dim/` 另加 kitchen、block_pushing。每次 run 给 best-epoch 与 latest 两份。https://diffusion-policy.cs.columbia.edu/data/experiments/image/ ；https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/ ；https://github.com/real-stanford/diffusion_policy
- 成本：DDIM 10 步 **0.1 s**（RTX 3080）；DDPM 100 步约 10×。https://arxiv.org/html/2303.04137v5
- 与我们 VLA 同构（迭代去噪头），不是搜索型；但 robomimic-mh 数据多峰是 k=1 傻基线可能掉点的候选场地（**需实测**）。

### 2.9 Diffuser（扩散规划器）
- `jannerm/diffuser` 提供 D4RL locomotion 与 maze2d 预训练扩散模型 + value（MIT）；AOR（replandiffuser）基于它做"何时重规划 + 从旧 plan 部分加噪 warm start"，Maze2D +38%。https://github.com/jannerm/diffuser ；https://github.com/rainbow979/replandiffuser ；https://arxiv.org/abs/2310.09629
- 成本：全轨迹 256 步去噪，一次 plan 秒级（**估计**）；依赖 mujoco-py/D4RL 旧栈，一天内装通有风险。

### 2.10 DreamerV3 / Dreamer 4
- DreamerV3 官方仓库不发预训练 ckpt（只有训练产出 `<logdir>/ckpt/`）。https://github.com/danijar/dreamerv3
- Dreamer 4（2025-09）官方**明确不放代码与权重**；社区有 PyTorch/JAX 复现与 `IamCreateAI/Dreamerv4-MC` Minecraft 权重。均属子族 (ii)（想象只用于训练），无操作 benchmark 权重。https://arxiv.org/abs/2509.24527 ；https://huggingface.co/IamCreateAI/Dreamerv4-MC ；https://www.talkrl.com/episodes/danijar-hafner-on-dreamer-v4/transcript

### 2.11 Genie / Oasis 类
- Genie 3：仅产品入口，无权重。https://deepmind.google/models/genie/
- Oasis 500M（Etched/Decart，MIT，2024-11）：Minecraft 键鼠条件帧生成，无机器人 benchmark。https://huggingface.co/Etched/oasis-500m
- Matrix-Game 2.0（Skywork，2025-08，开源权重，25 FPS 流式，Unreal/GTA 数据；3.0 于 2026-04）：同上，无机器人 benchmark。https://arxiv.org/abs/2508.13009
- 1X World Model Challenge：数据 + GENIE 式 baseline，评 PSNR/压缩，无策略闭环。https://github.com/1x-technologies/1xgpt

### 2.12 NWM（Navigation World Models）
- HF `facebook/nwm`（CDiT-XL 1B，CC-BY-NC 4.0，人脸模糊后重训）。规划 CEM 120 样本 × 1 iter × top-5，轨迹长 8 × 0.25 s；**只在 RECON 等录制数据上离线评 ATE/RPE**（1.13/0.35 vs NoMaD 1.93/0.52），无闭环仿真。https://github.com/facebookresearch/nwm ；https://huggingface.co/facebook/nwm ；https://arxiv.org/html/2412.03572v1

### 2.13 Cosmos-Predict2 系（除 Cosmos Policy）
- `Cosmos-Predict2-2B-Sample-Action-Conditioned`（NVIDIA Open Model License）：单图 + 末端位移/夹爪动作 → 3 s 640×480 4 fps 片段；**32.5 GB 显存，25.6 s（GB200）–181 s（RTX 6000 Ada）/段**；无评测数字。https://huggingface.co/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned
- `Cosmos-Predict2.5-2B/robot/action-cond`（2026-02-23 发布，chunk_size 12，不支持多 GPU 推理，时延**不确定**）。https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/inference_robot_action_cond.md ；https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
- `Cosmos-Policy-ALOHA-Planning-Model-Predict2-2B`：真机 ALOHA 规划器（+12.5 pp），属 Cosmos Policy 家族，排除。https://huggingface.co/nvidia/Cosmos-Policy-ALOHA-Planning-Model-Predict2-2B
- 用途都是"策略评测 / 数据引擎"（DreamGen、WorldGym 类），没有"world model + 规划 + sim benchmark"脚本。

### 2.14 其它开权重的动作条件视频世界模型（只做评测器，无规划脚本）
- GE-Sim 2.0（AgiBot，2B 蒸馏，CC BY-NC-SA 4.0，2026-06-25）：闭环视频模拟器，示例评 π0.5；无 MPC/best-of-N。https://github.com/AgibotTech/GE-Sim-V2
- GE-Act（CALVIN 权重）：single-step visual planner + IDM，非迭代搜索。https://github.com/AgibotTech/Genie-Envisioner
- WorldGym / world-model-eval：9 GB 扩散世界模型（Google Drive）+ VLM 打分评策略。https://github.com/world-model-eval/world-model-eval
- Ctrl-World（ICLR 2026，DROID，HF `yjguo/Ctrl-World`）：策略评测 + 想象 SFT。https://github.com/Robert-gyj/Ctrl-World
- iVideoGPT（MIT）：**唯一附带 VP2 规划 ckpt 与 `vp/` 规划代码**的视频世界模型（robosuite/robodesk 64×64 动作条件版）。https://github.com/thuml/iVideoGPT

## 3. 主表：模型 × benchmark × 权重 × 每次决策成本 × 可行性

成本口径：一次"规划调用"的 wall-clock（不是每环境步）；标 ★ 的是文献实测，其余为估计。可行性：A = 一天内 1×H100/4090 + A5000 仿真机能跑通闭环评测；B = 能跑但依赖旧栈/数据大/需改脚本；C = 缺一件或只能真机/离线。

| 模型 | benchmark | 权重（链接） | 许可证 | 每次决策成本 | 三件齐全? | 可行性 |
|---|---|---|---|---|---|---|
| jepa-wms JEPA-WM（+ DINO-WM、V-JEPA-2-AC(fixed) 基线） | Push-T / PointMaze / Wall / Metaworld-Reach、Reach-Wall；DROID→RoboCasa 零样本 | https://huggingface.co/facebook/jepa-wms | CC-BY-NC 4.0 | DINO-WM 级：★47 s/plan（LeWM 测）～ 分钟级；JEPA-WM 同量级（估计） | 是（HF 权重 + HF 数据 + `plan` 脚本 + grid eval） | A（sim 环境轻；DROID 5.6–8.7 TB 不碰） |
| DINO-WM 官方 | PointMaze / PushT / Wall（Rope/Granular 需 PyFleX） | https://github.com/gaoyuezhou/dino_wm（OSF ckpt） | MIT | ★47 s/plan；Sparse Imagination 报 79–297 s（口径不确定） | 是 | A |
| LeWM | PushT / OGBench-Cube / Two-Rooms / Reacher | https://huggingface.co/quentinll/lewm-pusht 等 4 个 | MIT | ★<1 s/plan（CEM 300×30，H=5 宏步=25 环境步） | 是（+ stable-worldmodel） | A（最省事） |
| PLDM | Two-Rooms / Diverse Maze；LeWM 套件里的 4 环境 | https://github.com/vladisai/PLDM ；LeWM Google Drive | MIT | LeWM 同量级（估计） | 部分（官方仓库无 ckpt 链接） | B |
| HWM | Diverse Maze（PLDM） | https://github.com/kevinghst/HWM_PLDM | 未标 | 单层的 1/3（论文） | 否（无 ckpt） | B |
| IMWM | Two-Room / Reacher / Push-T / OGBench-Cube | 无 | — | = LeWM CEM 预算 | 否（无代码） | C |
| TD-MPC2 单任务 5M | Meta-World 50 / DMControl 39 / ManiSkill2 5 / MyoSuite 10 | https://huggingface.co/nicklashansen/tdmpc2 | MIT | ★20.83 ms/步（4090） | 是 | A（MuJoCo 2.1.0 + gym 0.21 旧栈；状态输入） |
| TD-MPC2 多任务 317M | MT80 / MT30 | 同上 `multitask/` | MIT | 不确定（≥24 GB 显存） | 是 | B |
| Diffusion Policy | Push-T / robomimic 5 任务(ph,mh) / kitchen / block_pushing | https://diffusion-policy.cs.columbia.edu/data/experiments/ | MIT | ★0.1 s（DDIM 10 步，3080） | 是 | A |
| Diffuser（+ AOR replandiffuser） | D4RL maze2d / locomotion | https://github.com/jannerm/diffuser | MIT | 秒级/plan（估计） | 是 | B（mujoco-py/D4RL 旧栈） |
| iVideoGPT × VP2 | VP2 robosuite / RoboDesk（11 类 310 实例） | https://github.com/thuml/iVideoGPT ；https://github.com/s-tian/vp2 | MIT | 视频扩散/AR × MPPI 样本数，十秒级/plan（估计） | 是 | B（VP2 benchmark 数据 19.6 GB；robodesk 依赖） |
| V-JEPA 2-AC 官方 | 真机 Franka | https://github.com/facebookresearch/vjepa2 | MIT | ★16 s/action（CEM 800） | 否（无 sim） | C（sim 版走 jepa-wms） |
| NWM CDiT-XL 1B | RECON 等离线轨迹 | https://huggingface.co/facebook/nwm | CC-BY-NC | 120 样本 × 8 帧扩散，分钟级（估计） | 否（无闭环 sim） | C |
| Cosmos-Predict2-2B action-cond / Predict2.5 robot action-cond | 无 benchmark 脚本（Bridge 类数据） | https://huggingface.co/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned | NVIDIA Open Model | ★25.6–181 s/段（单候选） | 否 | C |
| GE-Sim 2.0 / WorldGym / Ctrl-World | 策略评测器 | 见 §2.14 | CC BY-NC-SA / — / — | 秒到十秒级/段（估计） | 否（无规划） | C |
| Oasis 500M / Matrix-Game 2.0 | Minecraft / GTA | https://huggingface.co/Etched/oasis-500m | MIT / 开源 | 实时帧生成 | 否（无机器人 benchmark） | C |
| DreamerV3 / Dreamer 4 | — | 无官方权重 | — | — | 否 | C |
| Cosmos Policy（对照，排除） | LIBERO 40 / RoboCasa | https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B | NVIDIA Open Model | ★0.16/0.61/0.95 s（1/5/10 步，H100）；best-of-8 4.9 s（8×H100） | 是 | B（单卡 best-of-N ≈ 5 s/chunk） |

## 4. "三件齐全"短名单（按每次决策成本从高到低）

1. **jepa-wms（JEPA-WM / DINO-WM / V-JEPA-2-AC(fixed)）× Push-T / PointMaze / Wall / Metaworld-Reach(-Wall)** —— 47 s 级/plan；CC-BY-NC；一个仓库拿到三种世界模型的同环境 ckpt，天然做"贵 vs 便宜"对照；96 集/环境的官方评测协议；ARC-Bench 已用这套 ckpt 证明"少重规划就崩"。
2. **DINO-WM 官方 × PointMaze / PushT / Wall** —— 47 s 级/plan；MIT；CEM 30 iter 是最典型的"迭代头 + 中间状态 (μ,σ)"。
3. **iVideoGPT × VP2（robosuite / RoboDesk）** —— 十秒级/plan（估计）；MIT；唯一"视频世界模型 + 官方规划 benchmark"组合，AdaReP 也用 VP2/RoboDesk 报数。
4. **Diffuser × D4RL maze2d** —— 秒级/plan；MIT；扩散规划器 + AOR 的 warm-start/似然门代码现成，是 warm 档最直接的文献对标。
5. **LeWM × PushT / OGBench-Cube / Two-Rooms / Reacher** —— <1 s/plan；MIT；HF 权重 + stable-worldmodel 求解器全家桶（CEM/iCEM/MPPI/GD），IMWM 的检索初始化在这上面做的。
6. **Diffusion Policy × robomimic (ph/mh) / Push-T** —— 0.1 s；MIT；与 VLA 同构，用来测"多峰数据上 k=1 是否掉点"。
7. **TD-MPC2 × Meta-World 50 / DMControl / ManiSkill2 / MyoSuite** —— 20 ms/步；MIT；最便宜、ckpt 最全，但状态输入且 wall-clock launch-bound。

排除理由汇总：Cosmos-Predict2 系 / GE-Sim / WorldGym / Ctrl-World（有权重无规划脚本，单候选 25–181 s）；NWM（无闭环 sim）；V-JEPA 2-AC 官方（只真机）；IMWM / HWM / AdaReP（无权重或无代码）；Dreamer 4 / Genie 3（无权重）；Oasis / Matrix-Game（无机器人 benchmark）。

## 5. 对 cache 迁移线的直接含义（三条）

- **规划成本跨度四个量级**（TD-MPC2 20 ms → LeWM 1 s → DINO-WM 47 s → Cosmos 分钟级），而 encoder 前缀在 DINO-WM/JEPA-WM 一档是 ViT-S/14 一次前向（~10 ms 级，估计），c_pre/c_0 远小于 VLA 的 15%——这是 hit 档收益最大的场地；LeWM 则相反（前缀 5M ViT-Tiny，头 <1 s），hit 省不了多少但 warm 档验证快。
- **ARC-Bench（2026-08）与 IMWM（2026-06）分别给了两端证据**：前者说 frozen JEPA 规划器的成功率靠每步重规划撑着、降频就崩（= 我们 hit 档不能裸用，需要 RIT 分档）；后者说检索到的 demo chunk 作 CEM 初始化能把 Cube 从 66→95（= warm 档有大 counterfactual）。两篇都没做跨 episode 库 + 多档 + 离线标定。
- **一天跑通的最短路径**：H100 上 `pip install stable-worldmodel` + `quentinll/lewm-pusht`，先复现 CEM 300×30 的 SR 与 <1 s/plan；同机拉 `facebook/jepa-wms` 的 pusht 三套 ckpt 做贵头对照。仿真机（A5000）只在 robomimic/VP2 线才需要。
