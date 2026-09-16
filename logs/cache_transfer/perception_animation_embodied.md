# 跨领域迁移调研：游戏动画 / 流式感知与 3D 视觉 / 非 VLA 具身控制

> 调研日期 2026-09-15。依据 `logs/cache_transfer/briefing.md` §1–§4、`docs/iclr/iclr_paper/arxivd.tex` L64–344（现行方法：tiers + RIT + stateful gate）、`logs/session_handoff.md` §1.2（减步 teacher 基线终态）。
> 三块领域各自打分（§3），其余段落合写、分小节。所有数字均来自引用 URL；查不到的写「不确定」。
> 只读仓库、只写本文件；未跑实验、未装依赖、未碰 git。

**一句话总览**：三块里只有 (c) 非 VLA 具身控制满足「减步会掉点」这一硬条件（DDPM 训练的控制器一步即崩），但它在审稿人眼里不算换领域，且「用自己上一个 chunk 做 warm start」的近亲工作（RTI-DP / Falcon / SDP / STEP）已经把时序局部性吃掉了大半；(a) 动画的 hit 本身就是 Motion Matching（行业基线），且行业正在为了省内存**删库**（Learned Motion Matching），与我们的方向相反；(b) 感知的前缀占 70–80% 成本、hit 地板反而比 VLA 更差，且逐帧复用是最拥挤的赛道。

---

## 1. 最贴合的具体系统

### (a) 游戏角色动画与人体运动生成

| 系统 | 模型 / 结构 | 开源 | benchmark | 成本拆分 |
|---|---|---|---|---|
| **CAMDM**（SIGGRAPH 2024，"Taming Diffusion Probabilistic Models for Character Control"） | Transformer 条件自回归 motion diffusion；输入=历史动作 + 用户控制信号；**训练与推理都只用 8 个 diffusion 步**；权重 20 MB | 是（[aiganimation.github.io/CAMDM](https://aiganimation.github.io/CAMDM/)） | 100STYLE（4M 帧、100 风格） | RTX 3060 上 8 步 = **13 ms/帧**，>60 FPS；Table 5：2 步 6 ms、4 步 8 ms、8 步 13 ms、16 步 25 ms、32 步 47 ms。**没有昂贵的共享前缀**：条件编码就是几层 tokenization，整个模型才 20 MB。[arXiv 2404.15121](https://arxiv.org/html/2404.15121) |
| **A-MDM**（TOG 2024，"Interactive Character Control with Auto-Regressive Motion Diffusion Models"） | 10 层 MLP（1024 宽）逐帧扩散，10–50 步 | 有项目页（[xbpeng.github.io/projects/AMDM](https://xbpeng.github.io/projects/AMDM/index.html)），论文正文未声明代码 | 100STYLE、LaFAN1 | LaFAN1：1 步 0.55 ms / 40 步 20.96 ms / 100 步 52.78 ms；100STYLE：10 步 9 ms / 40 步 21 ms。[arXiv 2306.00416](https://arxiv.org/html/2306.00416v2) |
| **CLoSD / DiP**（ICLR 2025） | 自回归实时 diffusion planner（**10 步**）+ RL 物理跟踪控制器闭环 | 是（[guytevet.github.io/CLoSD-page](https://guytevet.github.io/CLoSD-page/)） | 物理仿真多任务（到达/坐下/击打等），有成功率 | 步数已压到 10；未查到 ms 拆分（不确定）。[arXiv 2410.03441](https://arxiv.org/abs/2410.03441) |
| **Motion Matching / Learned Motion Matching**（Holden et al., SIGGRAPH 2020） | 纯最近邻检索回放；LMM 用 Decompressor / Stepper / Projector 三个小网络替换库查询，把 47 关节、30+ 风格的 590 MB 库压到 8.5 MB；**过渡搜索约 5 Hz**（每 ~200 ms 一次） | 数据集开源（LaFAN1），实现见 Holden 博客 | LaFAN1 / 100STYLE | 无 ms 数字（不确定）。[Ubisoft La Forge](https://www.ubisoft.com/en-us/studio/laforge/news/6xXL85Q3bF2vEj76xmnmIu/introducing-learned-motion-matching) |
| **PDP**（SIGGRAPH Asia 2024） | 物理角色 diffusion policy（DDPM） | 未声明代码 | 运动跟踪 / 扰动恢复 / 文本到动作 | 明确指出「多峰任务 diffusion 57.1% vs MLP 11.9%」，但跟踪单条参考动作时 MLP 98.8% ≈ PDP 98.9%。[arXiv 2406.00960](https://arxiv.org/html/2406.00960) |

### (b) 流式感知与 3D 视觉

| 系统 | 模型 / 结构 | 开源 | benchmark | 成本拆分 |
|---|---|---|---|---|
| **SAM 2**（Hiera 编码器 + memory attention + mask decoder） | 逐帧图像编码器 + 4 层 memory attention（对 6 帧近期记忆 + 首帧）+ 轻量 decoder；head **非迭代** | 是 | SA-V、DAVIS 2017（J&F） | Hiera-B+ 43.8 FPS、Hiera-L 30.2 FPS（A100）；Efficient-SAM2（ICLR 2026）称图像编码器最后一个 stage 就占编码器 60% 时间，SWR/SMR 各带来 1.83×/1.78×，端到端 1.68×、SA-V test J&F 79.8→78.8。[arXiv 2408.00714](https://arxiv.org/html/2408.00714)、[arXiv 2602.08224](https://arxiv.org/html/2602.08224v1)。三方文章估图像编码器占 70–80% 计算、memory attention ~10%（[mlhonk](https://mlhonk.substack.com/p/from-sam-to-sam-3)，非官方数字） |
| **RAFT / SEA-RAFT**（光流迭代精炼） | 前缀 = 特征编码器 + all-pairs 相关体；head = tied-weight GRU 迭代（默认 12–32 步），中间状态 = 当前 flow 场 + hidden | 是（[princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)） | Sintel / KITTI（EPE） | RAFT 论文：1080p 视频 12 步 550 ms/帧，**all-pairs 相关只占 17%** ⇒ 迭代 head 是大头（与 VLA 相反）；论文自带 warm-start（上一帧 flow 前向投影做初始化）并以此在 Sintel 上排第一。SEA-RAFT 把推理迭代 32→12。[arXiv 2003.12039](https://arxiv.org/pdf/2003.12039)、[arXiv 2405.14793](https://arxiv.org/html/2405.14793) |
| **Marigold / Marigold V2**（扩散式深度估计） | Stable Diffusion / DiT 微调为深度预测；V2 **单步 flow-matching** 直接出深度 | 是（[huawei-bayerlab/marigold-v2](https://github.com/huawei-bayerlab/marigold-v2)） | KITTI / ETH3D（AbsRel） | 「Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think」（WACV 2025）证明单步 Marigold 输出清晰深度图；V2 单步 AbsRel 再降 16–26%。⇒ 感知类扩散头**一步无损**。[arXiv 2409.11355](https://arxiv.org/pdf/2409.11355)、[arXiv 2609.08084](https://arxiv.org/html/2609.08084v1) |
| 在线 3DGS SLAM / 流式重建（SplatMAP、StreamGS、RTGS 等） | 关键帧触发的增量优化；「历史帧已重建区域重复处理是冗余」 | 部分开源 | Replica / TUM | 未查到统一成本拆分（不确定）。[arXiv 2510.06644](https://arxiv.org/html/2510.06644v2) |

### (c) 非 VLA 的具身控制

| 系统 | 模型 / 结构 | 开源 | benchmark | 成本拆分 |
|---|---|---|---|---|
| **DiffuseLoco**（CoRL 2024） | 6.8M Transformer encoder-decoder DDPM；输入 8 步 state/action 历史；receding-horizon | 是（[HybridRobotics/DiffuseLoco](https://github.com/HybridRobotics/DiffuseLoco)） | 四足 Go1 sim + 真机；velocity tracking error、稳定性 | **明确拒绝 DDIM**：DDIM-100/10 跑挂 2 次、tracking error +50.69%；DDIM-10/5 +42.04%；DDPM 全步数才稳。3080 上 10 步 DDIM ≈ 0.1 s（旧 tech report）。**无昂贵前缀**（观测编码就是线性投影）。[arXiv 2404.19264](https://arxiv.org/html/2404.19264) |
| **BeyondMimic**（2025，Unitree G1 人形） | 19.95M Transformer decoder，**训练/推理 20 步，约 20 ms**；classifier guidance 做 joystick / waypoint / 避障 | 项目页有，代码状态不确定 | LAFAN1 动作跟踪 + 引导任务，真机 | 前缀 ≈ 0；成本 = 20 步 × 1 ms。[arXiv 2508.08241](https://arxiv.org/html/2508.08241v1) |
| **Diffusion-MPC for locomotion**（2025） | diffusion planner，**warm start = 把上一个 plan 前向加噪再去噪**，10 步即可（冷启动需 20+ 步） | 项目页 [Flexible-Diffusion-MPC.github.io](https://Flexible-Diffusion-MPC.github.io/) | 四足仿真导航/避障 | <100 ms/规划周期。[arXiv 2510.04234](https://arxiv.org/pdf/2510.04234) |
| **PDP / CLoSD**（物理人形，见 (a)） | 同上 | — | — | — |
| **Diffusion Policy 家族**（robomimic/PushT，DP、OneDP、Consistency Policy、RTI-DP、Falcon、SDP、STEP） | 视觉编码器 + DDPM/DDIM U-Net 或 Transformer，100 DDPM / 10 DDIM 步 | 是 | robomimic、PushT、RLBench | V100 上 DDPM-100 660 ms / DDIM-10 66 ms / OneDP 7 ms。[arXiv 2410.21257](https://arxiv.org/html/2410.21257v1) —— 但审稿人会把这块视作与 VLA 同域（见 §6） |
| **RT-Cache**（2025） | 纯回放：DINOv2+SigLIP 2176 维 key，cosine 检索 Open-X 百万轨迹片段 | 未声明代码 | 真机 few-shot | 检索 ~0.1 s；few-shot 96%/93% vs OpenVLA-OFT 0%。[arXiv 2505.09040](https://arxiv.org/html/2505.09040v1/) |

---

## 2. 映射表（对应到我们的符号）

| 我们的符号 | (a) 动画（CAMDM / MM 混合） | (b) 感知（SAM 2 / RAFT） | (c) 非 VLA 具身（DiffuseLoco / BeyondMimic） |
|---|---|---|---|
| 共享前缀 c_pre | 几乎为 0：历史姿态 + 控制信号的 tokenization（20 MB 模型的一小部分） | **占大头**：Hiera 编码器 70–80%（SAM 2）；RAFT 特征+相关体 17% | 几乎为 0：proprio/history 线性投影 |
| tap point | 条件 token 之后 = MM 的 query feature（脚/髋/轨迹特征）本来就现成 | 编码器输出 embedding（SAM 2）/ 相关体（RAFT） | 观测编码后（就是 8 步 state/action 历史向量） |
| 头（迭代生成） | 8 步 DDPM（CAMDM）/ 10–50 步（A-MDM）/ 10 步（DiP） | SAM 2：memory attention + decoder，**非迭代**；RAFT：GRU 12–32 迭代 | DDPM 20 步（BeyondMimic）/ K 步（DiffuseLoco）/ 10 步 DDIM（DP） |
| 中间状态（warm start 起点） | 第 j 步的噪声姿态序列 | RAFT：flow 场 + hidden；SAM 2：无 | 第 j 步的噪声 action chunk |
| tiers | hit = 回放 MM 库片段（=纯 Motion Matching）；warm = 从库片段加噪到 j 步再去噪；miss = 8 步全算 | RAFT：hit = 复用上一帧 flow；warm = 从上一帧 flow 起跑 k<12 步（**论文已做**）；SAM 2：hit = 复用上一帧 mask（Reuse-Gate VOS 已做） | hit = 回放库 chunk；warm = 从库 chunk 的中间状态起跑 ρ_a·K 步；miss = 全步 |
| key 字段 | 历史根轨迹、脚位/速度、控制信号、风格标签（IVF cell = 风格/技能） | 编码器 embedding 池化、上一帧 mask、相机位姿 | proprio 向量、8 步历史、指令/技能 ID（IVF cell = gait/skill） |
| 分数 s_t | 各字段 tanh 标准化后加权（式 norm/fuse 直接可用） | 同上 | 同上（proprio 负欧氏距离为主） |
| 偏差 D_a | tier a 输出与 8 步冷启动输出的姿态/根速度偏差（只比执行帧） | EPE（RAFT）/ IoU（SAM 2） | 执行段 action 偏差（与现行完全同式） |
| 成功度量 SR | **没有天然 SR**：FID、foot sliding、轨迹误差、风格准确率；物理角色（PDP/CLoSD）有摔倒/任务成功 | **没有 SR**：J&F、EPE、AbsRel；闭环漂移 = 跟踪 mask drift | 有：摔倒率、velocity tracking error 阈值、任务成功 |
| 成本模型 | c_a = a·Δ（每步 ~1.6 ms @3060），c_pre≈0 ⇒ hit 地板 ≈ 0 | SAM 2：c_pre/c_0 ≈ 0.7–0.8（**比 VLA 的 15% 差得多**）；RAFT ≈ 0.17 | c_a = a·Δ（每步 ~1 ms @BeyondMimic），c_pre≈0 ⇒ hit 地板 ≈ 0 |
| 闭环性 / 漂移 | 用户交互闭环，错了视觉可见但不「失败」；物理角色会摔 | 跟踪漂移存在，但无「失败」阈值 | **强**：浮基/欠驱动系统，DiffuseLoco 明言「噪声控制信号会破坏本质不稳定的浮基动力学」 |

---

## 3. 打分表 A–G（0–3）

### (a) 游戏角色动画与人体运动生成

| 项 | 分 | 依据 | URL |
|---|---|---|---|
| A 结构匹配 | 2 | 头是 8–50 步 diffusion、有中间状态可 warm；但**没有昂贵共享前缀**（CAMDM 20 MB、13 ms 整模），key 只能取原始 MM 特征——虽然免费，「前缀末端白拿 query」的论点不成立 | [CAMDM](https://arxiv.org/html/2404.15121) |
| B 成本余量 | 2 | hit 地板 ≈ 0（好）；减步有可测退化：CAMDM 2 步 FID 0.938 / foot sliding 0.832 vs 8 步 0.913 / 0.685；A-MDM LaFAN1 1 步 APD 53 vs 40 步 129（多样性减半）。但绝对量只有 13 ms，且 MotionLCM 一步蒸馏已达实时 30 FPS | [CAMDM Table 5](https://arxiv.org/html/2404.15121)、[A-MDM](https://arxiv.org/html/2306.00416v2)、[MotionLCM](https://arxiv.org/abs/2404.19759) |
| C 局部性 | 3 | Motion Matching 本身就是库回放：每 ~200 ms 搜一次、其间连续播放 = 天然 hit 成串；库就是 mocap 数据集 | [LMM](https://www.ubisoft.com/en-us/studio/laforge/news/6xXL85Q3bF2vEj76xmnmIu/introducing-learned-motion-matching) |
| D 容忍度与闭环 | 2 | 近似可接受（视觉），有交互闭环；但无 SR，只有 FID/foot sliding；物理角色（CLoSD）才有成功率 | [CLoSD](https://arxiv.org/abs/2410.03441) |
| E RIT 可用性 | 2 | D_a 离线易算（姿态偏差），单调性合理；gate 对应物 = MM 的搜索节奏；但 δ↔ε 的映射落在感知质量而非成功损失，理论故事变弱 | [arxivd.tex §method:tiers] |
| F novelty | 1 | hit = Motion Matching（行业基线）；retrieval-augmented diffusion 已有（ReMoDiffuse、GenMM）；行业正为省内存把库删掉（LMM 590 MB→8.5 MB），我们要把库加回来；审稿人：「这是 MM + 扩散抛光，算力不是这个领域的痛点」 | [ReMoDiffuse](https://arxiv.org/abs/2304.01116)、[GenMM](https://arxiv.org/abs/2306.00378)、[LMM](https://www.ubisoft.com/en-us/studio/laforge/news/6xXL85Q3bF2vEj76xmnmIu/introducing-learned-motion-matching) |
| G 做得动 | 3 | CAMDM 开源、100STYLE 开源、3060 可跑；每个前沿点 = 离线指标计算，分钟级 | [CAMDM code](https://aiganimation.github.io/CAMDM/) |
| **合计** | **15/21** | | |

### (b) 流式感知与 3D 视觉

| 项 | 分 | 依据 | URL |
|---|---|---|---|
| A 结构匹配 | 1 | SAM 2 的头非迭代、无中间状态；RAFT 有迭代头但前缀只占 17%；前缀昂贵的系统（SAM 2）hit 后仍要付 70–80% | [SAM 2](https://arxiv.org/html/2408.00714)、[RAFT](https://arxiv.org/pdf/2003.12039) |
| B 成本余量 | 1 | SAM 2 hit 地板 ≈ 0.7–0.8（VLA 是 0.15）；RAFT 减迭代会掉 EPE，但**论文自带 warm-start 已把它修好**（5 步 warm ≈ 20 步冷，AV1-MV 论文）；扩散式感知（Marigold）一步无损 | [AV1 MV warm start](https://arxiv.org/pdf/2510.17427)、[Marigold V2](https://arxiv.org/html/2609.08084v1) |
| C 局部性 | 3 | 帧间局部性极强；DAVIS 17 约 73.3% 相邻帧 mask IoU>0.7（据 Reuse-Gate VOS 论文摘要，未核对原表，不确定）；但最近邻永远是 t−1，**库是多余的** | [Reuse Gate VOS](https://arxiv.org/pdf/2012.11655) |
| D 容忍度与闭环 | 1 | 无 SR，无 ε；跟踪漂移存在但是连续误差而非失败事件；「连续替换会漂移」在这里就是普通的误差累积 | — |
| E RIT 可用性 | 2 | D_a = EPE/IoU 离线易算、单调；但没有成功损失可映射；gate 对应物已被 Reuse Gate / Eventful Transformers 占了 | [Eventful Transformers](https://arxiv.org/abs/2308.13494) |
| F novelty | 0 | 逐帧复用是最拥挤的赛道：Deep Feature Flow、Eventful Transformers（2–4×）、Reuse-Gate VOS（2020）、Efficient-SAM2（ICLR 2026）、VLA-Cache（NeurIPS 2025，token 级跨帧复用）、TeaCache（去噪步间复用）；SLAM 关键帧选择更是几十年老题 | [Efficient-SAM2](https://arxiv.org/abs/2602.08224)、[VLA-Cache](https://arxiv.org/abs/2502.02175)、[TeaCache](https://arxiv.org/abs/2411.19108) |
| G 做得动 | 3 | 全部开源、单卡可跑、离线评测 | — |
| **合计** | **11/21** | | |

### (c) 非 VLA 的具身控制

| 项 | 分 | 依据 | URL |
|---|---|---|---|
| A 结构匹配 | 2 | 头 = DDPM 10–20 步、有中间状态；前缀 ≈ 0（6.8M / 20M 小模型），key 取 proprio 历史免费；但「共享前缀」论点消失，成本模型退化为 c_a = a·Δ | [DiffuseLoco](https://arxiv.org/html/2404.19264)、[BeyondMimic](https://arxiv.org/html/2508.08241v1) |
| B 成本余量 | 3 | **三块里唯一硬证据**：DiffuseLoco DDIM-10 跑挂、tracking error +50.69%，DDIM-5 +42%；DP 1 步 DDIM 六个 robomimic 任务全 0.000；RLBench DP 50→1 步 18.7→1.8%；PDP 多峰任务 diffusion 57.1% vs MLP 11.9%。hit 地板 ≈ 0。**注意**：仅对 DDPM 训练的控制器成立，flow-matching 策略一步无损（我们的 π0.5/GR00T 基线 + Dense-Jump 甚至称多步更差） | [DiffuseLoco](https://arxiv.org/html/2404.19264)、[OneDP](https://arxiv.org/html/2410.21257v1)、[From Flow to One Step](https://arxiv.org/html/2603.09415)、[PDP](https://arxiv.org/html/2406.00960)、[Dense-Jump](https://arxiv.org/abs/2509.13574) |
| C 局部性 | 2 | 步态周期性 / LAFAN1 跟踪重复动作 / 车队共享库（RT-Cache 纯回放可用）；但自身上一 chunk 的 warm start（RTI-DP 3 步够、Falcon 2–7×、Diffusion-MPC 步数减半）已吃掉大部分时序局部性；库只在 episode 开头、突变处、跨 agent 共享时有增量。命中率文献无数字（不确定） | [RTI-DP](https://arxiv.org/pdf/2508.05396)、[Falcon](https://arxiv.org/pdf/2503.00339)、[Diffusion-MPC](https://arxiv.org/pdf/2510.04234)、[RT-Cache](https://arxiv.org/html/2505.09040v1/) |
| D 容忍度与闭环 | 3 | 有 SR（摔倒、tracking 阈值、任务成功）；浮基系统闭环漂移比 LIBERO 强得多（DiffuseLoco 明言 DDIM 噪声破坏不稳定动力学）；ε 天然小 ⇒ 反而逼出 warm 档的存在理由 | [DiffuseLoco](https://arxiv.org/html/2404.19264) |
| E RIT 可用性 | 2 | 仿真里 shadow rows 廉价、D_a 同式；RTI-DP 自己提议「离线跑全步得 A0、选 K′ 使初值偏差最小」= RIT 的单 tier 版，说明社区认这个方向；但 action 偏差作为摔倒代理在不稳定系统上可信度未验证（不确定） | [RTI-DP §III Remark](https://arxiv.org/pdf/2508.05396) |
| F novelty | 1 | 近亲密集：RTI-DP、SDP、Falcon、STEP、RNR-DP、BRIDGER（自身 warm start）；Diffusion-MPC（前向加噪旧 plan）；To-the-Noise-and-Back / FlashBack（把用户动作部分加噪再去噪 = 外部样本 warm start）；RT-Cache（库回放）；ActionCache。剩余：跨 episode 库 + 风险索引 ladder + gate；审稿人：「与 VLA 同域」「先和 RTI-DP 比」 | [SDP](https://arxiv.org/abs/2406.04806)、[STEP](https://arxiv.org/pdf/2602.08245)、[RNR-DP](https://arxiv.org/abs/2502.12724)、[To the Noise and Back](https://arxiv.org/abs/2302.12244)、[FlashBack](https://arxiv.org/abs/2505.16892) |
| G 做得动 | 2 | DiffuseLoco 代码+ckpt 开源（Isaac Gym，4090 可跑）；BeyondMimic 代码状态不确定；PDP 无代码；每个前沿点 = 数百次仿真 rollout（秒级/集），远比 LIBERO 便宜；真机无 | [DiffuseLoco GitHub](https://github.com/HybridRobotics/DiffuseLoco) |
| **合计** | **15/21** | | |

---

## 4. 「减迭代次数」傻基线：文献现状

### (a) 动画
- **有人做过，退化温和但可测**。CAMDM Table 5：2/4/8/16/32 步 FID 0.938/0.926/0.913/0.919/0.914，foot sliding 0.832/0.780/0.685/0.675/0.692，时间 6/8/13/25/47 ms；作者选 8 步。[arXiv 2404.15121](https://arxiv.org/html/2404.15121)
- A-MDM LaFAN1：1 步 APD 53.12 vs 40 步 128.91（多样性减半，ADE 几乎不变）；100STYLE 10 步 vs 40 步 ADE 10.44 vs 10.36。即**一步 = 条件均值，丢的是多样性而不是精度**——与我们在 LIBERO 看到的现象同构。[arXiv 2306.00416](https://arxiv.org/html/2306.00416v2)
- 蒸馏把余量抹平：MotionLCM 一步 30+ FPS，4 步 FID 最佳，比 MLD 50 步快 13×。[arXiv 2404.19759](https://arxiv.org/abs/2404.19759)
- 结论：动画领域 warm 档有 counterfactual（多样性/foot sliding），但整条 ladder 只在 6→13 ms 之间挪，绝对收益小。

### (b) 感知
- RAFT：减迭代掉 EPE，但 warm-start 从上一帧 flow 起跑已是论文标配（Sintel 榜首用的就是 warm-start）；AV1 motion vector 做初值 5 步 EPE 1.68 < 冷启动 20 步 2.07，「warm start 不改最终精度只加速收敛」。[arXiv 2003.12039](https://arxiv.org/pdf/2003.12039)、[arXiv 2510.17427](https://arxiv.org/pdf/2510.17427)
- SEA-RAFT 靠更好的初值把推理迭代 32→12。[arXiv 2405.14793](https://arxiv.org/html/2405.14793)
- 扩散式感知：Marigold 单步即清晰、V2 单步 SOTA ⇒ B=0。[arXiv 2409.11355](https://arxiv.org/pdf/2409.11355)
- SAM 2 头不可减步（非迭代）；能减的是编码器，而那是 Efficient-SAM2 / Eventful 的地盘。
- 结论：要么一步无损，要么 warm start 已被 t−1 帧解决，库没有位置。

### (c) 非 VLA 具身
- **有人做过，且崩得彻底**（DDPM 家族）：DP 1 步 DDIM 六任务 0.000（OneDP Table 1）；RLBench DP 50→1 步 18.7→1.8（From Flow to One Step Table I）；DiffuseLoco DDIM-10 挂 2 次、error +50.69%，DDIM-5 +42.04%，只能用全步 DDPM。[arXiv 2410.21257](https://arxiv.org/html/2410.21257v1)、[arXiv 2603.09415](https://arxiv.org/html/2603.09415)、[arXiv 2404.19264](https://arxiv.org/html/2404.19264)
- Consistency Policy Table I：DDiM 9 NFE 在某任务掉到 0.14（DDPM 27 NFE 0.79–1.00）。[arXiv 2405.07503](https://arxiv.org/html/2405.07503v1)
- **反例必须写进论文**：flow-matching 策略一步无损（π0.5 spatial k=1 0.988、GR00T 全平；handoff §1.2），Dense-Jump 甚至称「增加 Euler 步数普遍降低性能」。所以 (c) 的 B 分只对 DDPM/DDIM 控制器成立，选模型时必须避开 flow-matching 头。[arXiv 2509.13574](https://arxiv.org/abs/2509.13574)
- 自身 warm start 后减步：RTI-DP「多数任务 3 步内收敛」（Push-T 状态版 25 ms vs DP 816 ms，分数 0.95 vs 0.92）；Diffusion-MPC 冷启动 20+ 步 → warm 10 步。这是我们的**真正对手基线**，不是 k=1。[arXiv 2508.05396](https://arxiv.org/pdf/2508.05396)

---

## 5. 预期收益（与 VLA 对比）

| | VLA（现状） | (a) 动画 | (b) 感知 | (c) 非 VLA 具身 |
|---|---|---|---|---|
| hit 地板 c_pre/c_0 | 15% | ≈ 0（模型无前缀） | SAM 2 70–80% / RAFT 17% | ≈ 0 |
| 减步地板 (s1+s2)/c_0 | 60.6% / 40.7% | ≈ 0（且蒸馏一步已实时） | 同上 | ≈ 0，但一步会崩 ⇒ 减步曲线在 IR 30–50% 处才回到 teacher（DiffuseLoco 5/10 步都不行 ⇒ 不确定具体拐点） |
| 命中率估计 | libero_10 IR 60 时 SR 0.83 | MM 已是 ~100% 回放（那是基线，不是收益） | t−1 帧几乎总是最近邻（库多余） | 不确定；步态周期性下应高，但要减去自身 warm start 已能拿到的部分 |
| IR 能压到 | GR00T IR 48 → SR 0.61 | 2 步 warm ≈ IR 0.46（6/13 ms），FID 代价 0.02（若 warm 能从 MM 片段起跑保住 8 步质量则更好，未验证） | 不值得算 | 若 hit 能占 50% 决策、warm 3 步占 30%、miss 20%：IR ≈ 0.5·0 + 0.3·0.15 + 0.2·1 ≈ 0.25（纯算术，无实测支撑） |
| SR 损失 | 8–30 pp 视预算 | 无 SR；FID/foot sliding | 无 SR | 未知；不稳定动力学下 hit 档可能很危险，warm 档才是主力 |
| 相比 VLA 的净变化 | — | 地板变好，绝对量变小（13 ms 里省），故事变弱 | 全面变差 | 地板变好、B 变硬、漂移故事变强；但对手从 k=1 换成 RTI-DP |

---

## 6. novelty 地形与最强反驳

### (a) 动画
- **已有近亲**：Motion Matching（检索回放 = 我们的 hit）；Learned Motion Matching 的 Stepper/Projector（stepper 在两次搜索之间自回归 = 我们的 gate skip 模式）；ReMoDiffuse（检索样本作为扩散条件，非初始化）；GenMM（生成式 MM）；MotionLCM（一步蒸馏）。未找到「检索片段 → 部分加噪 → 少步去噪」作为分档系统的论文——这个空位存在，但它就是 SDEdit 套在 MM 上，创意成本低。
- **最强反驳**：(1) 行业痛点是内存不是算力（LMM 的全部动机是删库），我们的库反其道而行；(2) 扩散头本来就 13 ms，省下的是 10 ms/角色，只在人群（crowd）场景有意义，而 crowd 本来就用 MM；(3) 没有成功率，RIT 的 δ↔ε 故事只剩感知指标；(4) 图形学审稿人看视觉质量，ICLR 审稿人看它是 SDEdit。

### (b) 感知
- **已有近亲**：Deep Feature Flow（2017）、Eventful Transformers（ICCV 2023，2–4×）、Reuse-Gate VOS（2020，帧间变化门控 = 我们的 gate）、Efficient-SAM2（ICLR 2026，窗口路由 + 稀疏记忆检索）、VLA-Cache（跨帧 token KV 复用）、TeaCache/DeepCache（去噪步间特征复用）、RAFT warm-start（原论文）。
- **最强反驳**：库检索在流里输给「上一帧」这个免费最近邻；hit 仍要付 70–80% 编码器；无 SR；每一件部件都有人做过。这块直接放弃。

### (c) 非 VLA 具身
- **算不算换领域**：不算。审稿人视角是「diffusion/flow 动作头 + 闭环控制」一族，VLA 只是多了 VLM 前缀；把前缀拿掉后成本模型退化为 a·Δ，反而失去「共享前缀白拿 query」这一现行方法的支点。能辩护的差异只有：(i) 动力学不稳定 ⇒ 漂移更强、ε 更小；(ii) 控制频率 50–100 Hz ⇒ 每步 ms 都算；(iii) 车队/多 agent 共享库是 VLA 没有的设定。若要写成「换领域」，必须以 (i)+(iii) 为主叙事。
- **多峰性 / 闭环漂移是否比 LIBERO 强**：是。多峰：DiffuseLoco 同目标下 trot/pace 两模态、PDP 文本到动作 diffusion 57.1% vs MLP 11.9%；漂移：浮基系统 DDIM 噪声即挂（DiffuseLoco），BeyondMimic 明言推理延迟本身就会影响稳定性。这正是我们在 LIBERO 缺的 counterfactual。
- **已有近亲**：自身 warm start 一整族（RTI-DP、SDP、Falcon、STEP、RNR-DP、BRIDGER、Diffusion-MPC），外部样本 warm start（To the Noise and Back、FlashBack），库回放（RT-Cache、ActionCache）。RTI-DP 甚至在 Remark 里提出「离线全步算 A0、按偏差选 K′」= 单 tier 的 RIT。
- **剩余新东西**：跨 episode 库（vs 自身上一 chunk）、多 tier 风险索引 ladder（vs 手调 K′）、成串 gate、闭环 (IR, SR) 前沿定义。审稿人第一刀：「RTI-DP 3 步不用库就到 teacher，你的库多买了什么？」——只有在 episode 起点、技能切换突变、跨 agent 冷启动三处库才有增量，实验必须直接量这三处。

---

## 7. 首个实验方案（≤ 2 周）

只推荐 (c)；(a) 给一个廉价备选；(b) 不做。

### 主方案：DiffuseLoco（Go1，Isaac Gym）
- **模型/benchmark**：[HybridRobotics/DiffuseLoco](https://github.com/HybridRobotics/DiffuseLoco) 开源 ckpt + 仿真；指标 = velocity tracking error + 摔倒率（SR = 不摔且 tracking error < 阈值）。4090 一卡够（6.8M 模型）。
- **第 1 步（第 1–3 天）傻基线 + 成本拆分**：实测每个 DDPM 步 ms（batch=1，launch-bound 预期每步常数），扫 k = 1…K（从噪声起跑），画 (IR, SR)。预期复现「DDIM-5/10 崩」——若不崩（作者只报了 DDIM 训练变体，DDPM 直接截断未报），本线终止。
- **第 2 步（第 3–6 天）自身 warm start 基线**：RTI-DP 式，用上一 chunk 加噪到 j 步再跑 K−j 步，扫 j。这是对手，不是 k=1。
- **第 3 步（第 6–10 天）库 + shadow + RIT**：collect 成功 episode（不同速度指令/地形）；key = 8 步 proprio 历史 + 速度指令（IVF cell = 指令桶）；shadow rows 算 D_a（执行段 action 偏差）；拟合 q_a(s)，δ 扫 ladder；tiers = hit / warm(ρ=0.25) / warm(ρ=0.5)。
- **第 4 步（第 10–14 天）闭环前沿**：每个 δ 点 ≥ 200 集仿真（秒级/集），对比三条曲线：k 步冷启动、自身 warm、库 RIT（含 gate）。**判决条件**：库 RIT 曲线必须在 IR ≤ 0.3 区间高于自身 warm，且差距集中在 episode 起点/指令切换处；否则 NO。
- 每个前沿点评测成本：200 集 × ~5 s ≈ 20 min（单 4090），10 个 δ 点 + 3 条曲线 ≈ 10 h。

### 备选：CAMDM（100STYLE，3060 即可）
- 基线：Table 5 已有 2/4/8 步；加 1 步。
- 实验：MM 检索片段（key = 当前历史 + 控制信号）→ 加噪到 j 步 → 跑 8−j 步，看能否在 2 步（6 ms）拿到 8 步的 foot sliding 0.685 而非 0.832；离线评 FID/foot sliding/风格准确率，一天出数。
- 只作为「warm 档 counterfactual 是否存在于非机器人领域」的旁证，不作主线。

---

## 8. 结论

- **(a) 游戏动画：NO（偏 MAYBE−）**——形式最像（hit 就是 Motion Matching，warm 档确有多样性 counterfactual），但算力不是该领域痛点、行业正在删库、没有成功率，ICLR 审稿人会把它读成 SDEdit + MM。
- **(b) 流式感知 / 3D 视觉：NO**——前缀占 70–80% 使 hit 地板比 VLA 更差，扩散式感知一步无损，迭代精炼类（RAFT）的 warm start 已被 t−1 帧免费解决，逐帧复用赛道人满为患。
- **(c) 非 VLA 具身控制：MAYBE**——唯一满足「减步会崩」硬条件（DDPM 控制器一步 0%、DiffuseLoco DDIM 即挂）且闭环漂移比 LIBERO 强的领域，hit 地板 ≈ 0；但审稿人不会认它是换领域，而且真正的对手是 RTI-DP/Falcon 那类不用库的自身 warm start——两周实验的唯一目的就是量出「库比上一 chunk 多买了什么」，量不出就停。
