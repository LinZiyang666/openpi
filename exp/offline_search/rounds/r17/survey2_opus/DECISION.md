# R17 二轮调研：LIBERO 和 MetaWorld 都有权重的非 VLA 策略（survey2_opus）

2026-10-08。证据、文件清单和枚举过程见 `EVIDENCE.md`。

## 结论

| 项 | 结论 |
|---|---|
| 5 条硬性要求全过的候选 | **0 个** |
| 最接近 ①（条件入选） | **Cosmos Policy**（视频扩散 WAM，2B）：R2–R5 都过。R1 只过一半：MetaWorld 只有社区权重 `mmurray-ms/...MetaWorld...`，它自带的 T5 嵌入正好是 MT50 的 50 条指令（49 条和 `lerobot/metaworld_mt50` 逐字相同），很可能是 MT50 多任务微调；但只报告了 assembly-v3 一项（冻结基座约 52% / 25 局，作者说刻意留了余量），其余 49 个任务没有数字 |
| 最接近 ② | **DP-T**（brandonyang，社区 Diffusion Policy-Transformer）：两边都有权重（LIBERO 4 套件混训 / MetaWorld ML45），单帧无状态；但没有代码、没有 README、没有成功率，每次调用只要 0.14 TFLOP，CPU 2.4 s（100 步）/ 0.25 s（10 步），R4 不过 |
| WAM / 视频扩散族的格局 | 发布了权重的 WAM 只做 LIBERO（或 LIBERO + RoboTwin / RoboCasa / DROID）；MetaWorld 上的视频扩散策略（VPP、PAD、AVDC、VPDD、S-VAM）都没有 LIBERO 权重。这一族里两个基准**没有交集** |
| 建议 | ① **先做一个便宜的判定实验**：照 cosmos-policy 的 LIBERO 评测脚本写一个 MetaWorld 接口，用社区权重在 MT50 上每任务跑 10 局。成功率可用 → Cosmos 过 R1、作为唯一入选；成功率很低 → 维持"只做 LIBERO"的特例；② 不建议上 DP-T / ACT（太便宜，缓存省不下多少）；③ 不要指望别的 WAM：带权重的都只有 LIBERO |

## 表 1 全部 LIBERO∩MetaWorld 候选 × 硬性要求

R1 两个基准都有可用权重（权重 + 配置齐全、是标准任务集）｜R2 非 VLA｜R3 只依赖当前观测｜R4 缓存有利可图｜R5 族内较新、不与其他入选同族

| # | 候选（LIBERO 仓库 / MetaWorld 仓库） | 族 · 日期 | R1 | R2 | R3 | R4 | R5 | 判定 |
|---|---|---|:-:|:-:|:-:|:-:|:-:|---|
| 1 | Cosmos Policy：`nvidia/Cosmos-Policy-LIBERO-Predict2-2B` / `mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B` | 视频扩散 WAM · 2025-12 / 2026-07 | ◐ LIBERO 官方 98.5%；MetaWorld 社区权重带 MT50 全部 50 条指令嵌入，但只有 assembly-v3 一项数字（52%） | ✓ | ✓ 当前帧 + 本体 + 文本，固定种子 | ✓ 0.61 s 对 ≤0.05 s，≥12× | ✓ 族内唯一带 MetaWorld 的 | **最接近；MT50 冒烟通过即入选** |
| 2 | DP-T：`brandonyang/dp-libero-checkpoints` / `brandonyang/dp-metaworld-checkpoints` | Diffusion Policy（Transformer，DDPM 100 步）· 2026-04 | ◐ 权重 + metadata 在；无代码、无 README、无成功率；LIBERO 只传了 30k / 100k 步 | ✓ | ✓ `n_obs_steps=1` | ✗ 0.14 TFLOP / 调用，CPU 可算；降步后 14 GFLOP | ✓ | 不过（R4） |
| 3 | lerobot Diffusion Policy：`iFaz/diffusion-libero_goal-seed3-15000` / `iFaz/diffusion-metaworld_medium-seed42-half` | DP-UNet · 2026-05 | ✗ MetaWorld 仓库是空的 | ✓ | ✗ `n_obs_steps=2` | ✗ | — | 不过 |
| 4 | ACT：`kunhsiang/act-libero-baseline` / `kunhsiang/act-metaworld-mt10-baseline`；`SongMugeon/Act_finetune_libero_*` / `Act_finetune_meta_*` | ACT · 2026-02/03 | ✓ 无成功率；kunhsiang 的 MetaWorld 版只吃状态、不吃图像 | ✓ | ✓ | ✗ 单次前向、无迭代头，owner 已否 | — | 不过 |
| 5 | CRT：`AIRLab-POLIMI/CRT_LIBERO` / `CRT_MetaWorld` | 不明 · 2026-01 | ◐ 只有 .pth，README 空，无代码、无数字 | ? 内部是 1500→300 维 token 回归器，没有图像编码器和动作头 | ? | ? | — | 不可用 |
| 6 | AVDC 系：`Po-Chen/flowdiffusion`（MetaWorld 11 任务）/ GCP `video-to-action`（LIBERO 8 任务，OneDrive） | 视频扩散规划器（U-Net 201M）· 2023-10 / 2024-11 | ◐ 两个不同项目，任务子集都不标准 | ✓ | ✗ AVDC 的动作来自光流 + 脚本控制器、开环执行；GCP 的低层策略需要生成的目标图 | （一次规划 6–18 s）| — | 不过 |
| 7 | OAT：`hackhackhack66666/aaai27-models`（受限） | 有序动作 token · 2026-07 | ✗ 只有 MetaWorld 4 个单任务，没有 LIBERO | ✓ | ? | ✗ 小模型 | — | 不过 |
| 8 | VLA 系两边都有：π0 / π0.5 / π0-FAST（RLinf 等）、SmolVLA、OpenVLA-OFT、X-VLA、Evo-1 / EVO-Depth / Evo-World（InternVL3-1B）、starVLA、ALAM+π、GR00T N1.7 | VLA | ✓ | ✗ | — | — | — | R2 不过 |
| 9 | 论文两边都报告、但没有两边的权重：QueST、BAKU、PRISE、Premier-TACO、SkillDiffuser、STAR、Cocos（只有 LIBERO 权重）、MGP、ARP、PEWAM（2026-10，接收后才放）、LIFD（有场景记忆）、Sparse2Act、ORCA、CPC、Canonical（HF 仓库 401）、MINT（MetaWorld 权重"计划中"）、RoMAN-Flow（只有 LIBERO） | 多种 | ✗ | 多数 ✓ | 多数 ✓ | 多数 ✗（小模型，或视觉占大头） | — | 不过 |
| 10 | 只有 LIBERO 的 WAM / 扩散：FastWAM、LingBot-VA、DreamZero、OpenWAM、LaWAM、ImageWAM、Long-WAM、mimic-video、Cosmos3、UniWAM、StreamWAM、AED、MoDE | WAM / 扩散 | ✗ 没有 MetaWorld | ✓ | —（R1 已不过，未逐个核查；LingBot-VA 有 KV cache） | — | — | 不过 |
| 11 | 只有 MetaWorld 的视频扩散 / 世界模型：VPP（只放了 CALVIN 权重）、PAD、AVDC、VPDD、S-VAM、TD-MPC2、jepa-wms | 视频扩散 / 世界模型 | ✗ 没有 LIBERO | ✓ | —（R1 已不过；jepa-wms 需要目标图，TD-MPC2 用上次计划热启动） | — | — | 不过 |

## 表 2 最接近的两个

| 项 | Cosmos Policy | DP-T（brandonyang） |
|---|---|---|
| 架构 | Cosmos-Predict2-2B-Video2World DiT；Wan2.1 VAE 编码 224² 图像；proprio / 动作 / 价值以 latent 帧注入；latent 序列 9 帧 = 1764 token | 2× ResNet18（语言 FiLM，96² 裁到 84²）→ DP-T（4 层 cond 编码器 + 12 层解码器，d=512），共 91.8M |
| 检索键 | 当前腕部 + 第三人称两帧的 VAE latent（2×16×28×28，可池化；MetaWorld 版可能只有一路相机） | 两路 ResNet18 特征（2×512） |
| 动作头 / 默认步数 | 联合去噪，5 步（可配，不用重训） | DDPM 100 步（squaredcos），ε 预测 |
| 块长 / 执行 | 16 / 16 | 16 / 16 |
| 无状态 | ✓ `run_libero_eval.py`：队列空才调用 `get_action(当前 obs, seed 固定)`；论文明确"不用历史输入" | ✓ metadata `n_obs_steps 1` |
| 权重 | LIBERO：.pt 3.9 GB + config + 统计量 + T5 嵌入；MetaWorld：consolidated.pt 3.9 GB + 统计量 + T5 嵌入（50 条，= MT50 指令） | 两边各一份 safetensors 367 MB + metadata.pt + norm_stats；无 config.json、无代码 |
| 成功率 | LIBERO 98.5%（98.1 / 100 / 98.2 / 97.6）；MetaWorld 只有 assembly-v3 约 52%（25 局），其余 49 个任务无报告 | 无报告，需要自己测 |
| 许可 | LIBERO：NVIDIA 单向非商用（NSCLv1）；MetaWorld：other | 未声明 |
| 适用层 | 全部：缓存、守卫、修正器、周期旋钮、逐任务下降、减步（5→1，论文 RoboCasa −0.7 个点）、延长执行段 | 全部：减步（DDPM 100 → DDIM 10）也适用 |
| 风险 | MetaWorld 权重只报告了 1 个任务，且作者说刻意削弱；Cosmos 的 MetaWorld runner（FlowDAgger 内部）未公开，需要自己写 MetaWorld 接口（夹爪本体按 `[g, -g]` 编码；动作 4 维填进 7 维，第 3–5 维方差为 0）；lerobot MT50 数据只有一路 480² 图像，相机 / 输入布局要从权重和统计量反推；2B 模型在 4090 上的延迟未知（论文数字是 H100） | 要按 openpi 风格重写 DP-T 加载和推理；成功率未知；太便宜，缓存收益小 |

## 表 3 盈利性（全调用 : 命中路径）

| 模型 | 全调用 | 命中路径（取视觉键） | 全:命中 | 每次命中省 | 来源 |
|---|---|---|---:|---|---|
| π0.5（参照） | 1.00（.152 + .848） | .152 | 6.6×；k=1 时 4.0× | — | R16 口径 |
| GR00T（参照） | 1.00（.148 + .852） | .148 | 6.8×；k=1 时 2.7× | — | R16 口径 |
| Cosmos 5 步 | 0.61 s（H100） | ≤0.0475 s（固定部分：33 帧 VAE + T5 + 预处理） | **≥12.8×** | ≈0.56 s / 块 | 论文：0.61 s@5 步、0.16 s@1 步 |
| Cosmos 1 步 | 0.16 s | ≤0.0475 s | ≥3.4× | ≈0.11 s / 块 | 同上 |
| DP-T 100 步 | 145 GFLOP；CPU 2.36 s；GPU 量级约 0.1 s | 1.2 GFLOP；CPU 12.7 ms | 122×（FLOP）/ 185×（CPU） | GPU 约 0.1 s | CPU 探针；MGP 论文 DP 106 ms |
| DP-T 10 步 | 14.4 GFLOP；CPU 0.25 s | 同上 | 13× / 19× | GPU 约 10–15 ms | CPU 探针 |
| ACT | 单次前向，视觉占大头 | ≈全调用 | ≈1× | ≈0 | — |

- Cosmos 的比值和 π0.5 同一量级：满步约是 π0.5 的 2 倍（≥12.8× 对 6.6×），k=1 时 3.4× 对 4.0×。命中路径是用固定部分取的上界，只编码两帧当前图像的话会更小。
- DP-T 比值很高，但那是 100 步串行去噪撑出来的；我们自己的减步层先把它压到 14 GFLOP，之后缓存每次命中只省约 10 ms，属于 owner 说的"CPU 都能算"。

## 下一步（供 owner 选）

| 选项 | 内容 | 代价 / 风险 |
|---|---|---|
| A（推荐） | Cosmos 社区 MetaWorld 权重 MT50 冒烟：每任务 10 局，共 500 局 | 写 MetaWorld 接口约 1 天；GPU 约 500 局 × 每局约 30 块 × 约 1 s/块 ≈ 4 卡时（4090 延迟按 H100 的约 1.5 倍估算） |
| B | 冒烟不过：维持现状，π0.5 + GR00T 跑两个基准（GR00T 在 MetaWorld 上自己微调），Cosmos 只做 LIBERO 特例 | 无新增 |
| C | 冒烟可用但偏弱：自己在 MT50 上接着微调 Cosmos（从社区权重起步） | 官方 LIBERO 训练 40k 步、批量 1920、64 张 H100 约 48 h；缩小批量能做到多少未知 |
| D | 等 PEWAM（2610.02398，承诺接收后放权重）、MINT（MetaWorld 计划 2026 下半年） | PEWAM 有跨调用程序状态、只有 71.5M；MINT 是 token 策略，都未必过 R3 / R4 |
