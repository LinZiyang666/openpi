# Long-WAM 能否接入推理成本系统：评估

2026-10-09。只做了网页和代码阅读，没有下载权重，也没有用 GPU。唯一跑过的代码是一条 CPU 单行检查 OmegaConf 语义，见 §4 末尾。

**引用缩写**（文件:行号都可以在本机打开）：

| 缩写 | 本机路径 | 来源 |
|---|---|---|
| `P:L` | `/home/weiland/projects/openpi_ext/scratch/longwam_probe/paper/paper.txt`（arXiv HTML 转成的纯文本） | https://arxiv.org/html/2610.10528v1 |
| `W/` | `/home/weiland/projects/openpi_ext/scratch/longwam_probe/web/`（项目页 `data.js` 和 `config.js`，页上所有数字都从这里来） | https://nvlabs.github.io/LongLive/Long-WAM/static/js/data.js |
| `R/` | `/home/weiland/projects/openpi_ext/scratch/longwam_probe/repo/Long-WAM/`（浅克隆，commit `e5fa48cc5b1f`，2026-10-09） | https://github.com/NVlabs/LongLive/tree/e5fa48cc5b1f995637b987195d25b5bd16e26086/Long-WAM |
| `H/` | `/home/weiland/projects/openpi_ext/scratch/longwam_probe/hf/`（API 列表 `models.json`、各仓库的 `tree_*.json`、`README.md`、`config.yaml`） | https://huggingface.co/api/models?author=Efficient-Large-Model&limit=1000 ，以及 `https://huggingface.co/Efficient-Large-Model/<repo>` |

## 0. 结论

| 问题 | 结论 | 依据 |
|---|---|---|
| (a) 能不能原样用 | **不能**。已发布的 LIBERO 权重（IDM 和 COD 两个）都用 P48，也就是 2.4 s 历史：每次调用要输入 13 帧历史，runtime 还在整局里维护一个帧缓冲。没有 MetaWorld 权重 | §1、§2 |
| (b) 有没有零历史的模式或权重 | **LIBERO 和 MT50 都没有**。论文里 LIBERO-Long 94.5% 的那个 0 s 模型没有发布；只有 RoboCasa GR1 发布了 0 s 权重，但 GR1 不是我们的基准。代码里支持 P0 路径，但 P48 权重直接改成 P0 跑，潜帧数和位置都与训练不同，官方不支持。还有一个变通办法 (b')：把当前帧复制 13 份当作历史喂给 P48 权重。这样输出在字面上只依赖当前帧，但没有任何公开结果，必须我们自己测 | §2 |
| (c) 自己微调零历史版本 | 可行，单张 H100 上：**LIBERO P0 约 1.3–1.4 天**（从 LIBERO-IDM 初始化，视频 LoRA，3 个 epoch；照论文 10 个 epoch 则约 4.5 天）；**MT50 P0 约 0.7 天**（冻结视频专家）**到 1.8 天**（视频 LoRA）。还要写代码：Long-WAM 训练器没有冻结和 LoRA 训练模式，要接 MW 数据和适配器，要预算文本嵌入 | §4 |
| (d) 建议 | **不建议移植（倾向 d）**。零历史时，LIBERO-Long 94.5% 对 Fast-WAM 的 95.2%，没有优势；Long-WAM 的增益全部来自历史，而历史正是 owner 禁止的。它与 Fast-WAM 同族：同一个 Wan2.2-TI2V-5B 视频专家加 1024×30 的动作专家（MoT），代码也来自 FastWAM，加进来不增加模型族的多样性。唯一的好处是完整调用更贵（多了 4 步视频想象），完整调用与键路径的 FLOP 比约 34 倍（Fast-WAM 约 4.5 倍），缓存收益会被放大 | §3、§5 |
| 如果 owner 仍想看看 | 便宜探针 (b')：下载 12 GB 的 LIBERO-IDM，用重复当前帧的方式在 LIBERO 长任务和空间任务上各跑 500 局（参考实现，H32/A10），看与原生 P48 差多少。约一个 GPU 晚上，只需要读取已有的 Wan VAE/T5 文件 | §2 |

## 1. 有哪些权重（HF API，2026-10-09）

Efficient-Large-Model 名下共有 162 个模型，其中与 Long-WAM 相关的是 11 个策略权重、2 个机器人视频生成模型和 1 个基座（`H/models.json`）。修订号锁在 `R/configs/checkpoints.yaml:28-60`。

| HF 仓库 | 基准 | 模式 | 历史窗口 | 动作块 / 去噪 | 文件 | 依据 |
|---|---|---|---|---|---|---|
| Long-WAM-LIBERO-IDM | LIBERO 四个套件 | IDM | **P48 = 2.4 s**：过去 48 个控制步加当前帧，每 4 步取 1 帧，共 13 帧，编成 4 个干净潜帧 | H32；视频 4 步到 σ=0.9，动作 10 步 | model.pt **12.042 GB**，另有 config/stats/SHA256SUMS | `H/Long-WAM-LIBERO-IDM/README.md:24-27`，`config.yaml:5,36-39,118-121`；`H/tree_Long-WAM-LIBERO-IDM.json` |
| Long-WAM-LIBERO-COD | LIBERO | 视频和动作联合去噪 | P48 | H32；联合 4 步（σ 从 1 到 0），每 10 步重规划 | 12.042 GB | `H/Long-WAM-LIBERO-COD/README.md:24-27`，`config.yaml:5,121-123` |
| Long-WAM-RoboTwin2.0-IDM / -COD | RoboTwin 2.0 | IDM / COD | P48 | H32，每 24 步重规划 | 各 12.042 GB | `H/Long-WAM-RoboTwin2.0-IDM/README.md:24-27` |
| Long-WAM-RoboCasa365 | RoboCasa365 | IDM | P48 = 2.4 s | H32，每 32 步重规划 | 12.042 GB | `H/Long-WAM-RoboCasa365/README.md:24-27` |
| Long-WAM-YAM | YAM 真机（ABC 130K 预训练） | IDM | P48 = **1.6 s**（30 Hz） | H32 | 12.042 GB | `H/Long-WAM-YAM/README.md:38-40` |
| Long-WAM-RoboCasa-GR1-**0s** | RoboCasa GR1 | IDM | **P0：只用当前帧（1 个干净潜帧）** | H16；视频 4 步加动作 10 步 | 12.042 GB | `H/Long-WAM-RoboCasa-GR1-0s/README.md:21-28`，`config.yaml:28-29,94` |
| Long-WAM-RoboCasa-GR1-2.4s / 4.8s / 9.6s / 19.2s | RoboCasa GR1 | IDM | P48 / P96 / P192 / P384，对应 4 / 7 / 13 / 25 个潜帧 | H16 | 各 12.042 GB | `R/README.md:130-143`；`H/Long-WAM-RoboCasa-GR1-19.2s/README.md:21` |
| LongLive2.0-Robot-S / -M | 视频生成（不是策略） | — | — | — | model.pt 19.999 GB（FP32，约 5.0B 参数） | `R/README.md:145-153`；`H/LongLive2.0-Robot-S/README.md` |
| LongLive-2.0-5B（基座） | 视频生成 | — | — | — | model_bf16.pt 10.000 GB（5.0B） | `H/LongLive-2.0-5B/README.md:38` |
| **MetaWorld** | **没有** | | | | | 论文全文 0 处提到 MetaWorld；仓库里也没有 MetaWorld、Meta-World 或 MT50 |
| **LIBERO P0（论文 94.5% 的那个）** | **没有发布** | | | | | 上面的 HF 列表 |

**许可证**：代码是 Apache-2.0（`R/README.md:670-671`；GitHub API 的 license 字段）。所有策略权重在 HF 上的标签都是 `license: other`，模型卡里没有写具体条款（`H/Long-WAM-*/README.md:2`）。仓库的 NOTICE 明确说代码许可证不覆盖权重（`R/NOTICE:24-26`）。视频模型卡写的是 “Original upstream model terms continue to apply”（`H/LongLive2.0-Robot-S/README.md:35`），基座 LongLive-2.0-5B 用的是 NVIDIA Open Model License（`H/LongLive-2.0-5B/README.md:2-5`）。所有仓库公开、不需要申请（`H/info_*.json` 里 `gated: False`）。

## 2. 有没有状态，历史是什么

| 项目 | 事实 | 依据 |
|---|---|---|
| 历史输入 | P48：过去 48 个控制步加当前帧，共 49 帧原始画面，每 4 步取 1 帧得到 13 帧，经 VAE 编成 4 个干净潜帧；LIBERO 是 20 fps，所以是 2.4 s。干净潜帧数 = P // 16 + 1 | `R/src/longwam/runtime/policy.py:381-395`；`R/src/longwam/models/wan22/long_wam.py:95-107`；`H/libero_meta/*.json`（fps 20） |
| 局首补帧 | 历史不足时重复最早的一帧，训练和推理一致 | `R/src/longwam/runtime/policy.py:355-357,383-386`；`P:L341-342` |
| 整局状态 | runtime 维护长度为 P+1 的帧队列，每个控制步都要喂一帧，`reset` 时清空 | `R/src/longwam/runtime/policy.py:118-120,130-136,139-145`；`R/infra/README.md:243`（“Provide the current observation at every control step”） |
| 跨调用的 KV cache | **没有**。文本和状态的交叉注意力 KV、已观测视频的 KV、RoPE 表都只在一次调用内复用，每次 `infer_joint_ar` 都重新预填 | `P:L424-426`；`P:L1566-1573`（Cache scope：同一次动作求解内复用，与跨决策复用是两回事）；`R/src/longwam/models/wan22/longwam_base.py:1920-1925,1636-1655,1969-1981` |
| 流式 VAE | 可选，LIBERO 评测默认关闭。打开后，因果 VAE 的卷积缓存跨控制步保留（在两次决策之间预先编码下一个窗口），每个决策窗口开始时重置；数值上与整窗编码等价 | `R/configs/eval/libero.yaml:52`；`R/src/longwam/runtime/streaming_vae.py:33-55,79-88`；`R/src/longwam/runtime/policy.py:146-160`；`R/docs/VERIFICATION.md:166-167` |
| 本体感受 | 当前的 q_t 投影成一个上下文 token，两个专家都会读到，所以视频 KV 也依赖 q_t | `R/src/longwam/models/wan22/longwam_base.py:293-327,1861-1864`；`P:L1566-1573` |
| 代码里的零历史路径 | P=0 时只编码一张当前帧，不构造历史窗口 | `R/src/longwam/runtime/libero.py:218-220,254-258`；`longwam_base.py:1822-1827` |
| 能否把 P48 权重改成 P0 跑 | 机械上可以（参数形状与 P 无关），但干净潜帧从 4 个变成 1 个，未来帧的时间位置也变了，与训练分布不符；官方要求权重和配置成对使用 | `R/README.md:143,356`；`long_wam.py:106` |
| (b') 把当前帧复制 13 份 | 这等于每次调用都模拟“局首”的输入，局首正是训练分布里存在的情形；但局中出现静止历史在训练里很少见（LIBERO 用的是去掉空操作的数据集），论文没有这个结果。计算量按 P48 算 | `R/configs/data/libero.yaml:23-26`；`P:L1762-1767` |

**论文报告的最短历史（只有 LIBERO-Long，其他套件没有零历史数字）**：

| 视频初始化 | 0 s（只用当前帧） | 2.4 s | 4.8 s | 9.6 s | 依据 |
|---|---:|---:|---:|---:|---|
| LongLive2.0-Robot（AR，机器人域） | **94.5** | **99.5** | 99.0 | 99.0 | `W/data.js:8-9`；`P:L122-124`；`P:L778-780` |
| LongLive-2.0（AR） | 94.2 | 99.0 | 99.0 | 99.0 | 同上 |
| Wan2.2（双向） | 93.4 | 96.5 | 97.1 | 96.2 | `W/data.js:9` |

- **94.5% 的配置**：每个窗口单独训练一个模型，在训练时的窗口长度下评测；“0 s”是只保留当前观测、不带历史的 P0 模型，初始化用 LongLive2.0-Robot（`P:L339-342`；`W/data.js:4`）。论文没有单独写这个 P0 模型用的去噪模式。它在 2.4 s 处的点（99.5）与主表的 IDM 行完全相同（`P:L537-542`），所以推断是同一套 IDM 配方。
- **容易混淆**：主表里 “w/o V”（不做未来视频去噪，2.4 s 历史）的 LIBERO-Long 恰好也是 94.5（`P:L525-530`），这是另一个配置。

## 3. 架构和每次调用的成本

### 3.1 结构

| 部件 | 规格 | 依据 |
|---|---|---|
| 视频专家 | Wan2.2-TI2V-5B 的 DiT（hidden 3072，30 层，24 头 × 128，输入 48 通道，patch 1×2×2，逐帧因果掩码），用 LongLive2.0-Robot 初始化。LongLive2.0-Robot 是 LongLive-2.0 的 16 s AR 权重在约 1 万窗口小时的机器人和第一人称视频上继续训练的结果；LongLive-2.0 本身是 AR 训练的 Wan2.2-TI2V-5B | `H/Long-WAM-LIBERO-IDM/config.yaml:103,122-146`；`P:L258-265`；`H/LongLive-2.0-5B/README.md:38` |
| 动作专家 | ActionDiT（hidden 1024，ffn 4096，30 层，24 头 × 128，与视频专家做 MoT 混合注意力），约 1.0B。我们 Fast-WAM 用同样配置，可训参数实测 1,020,900,356 | `config.yaml:147-157`；`/home/weiland/projects/openpi/exp/mw_finetune/fastwam/HANDBACK_C.md:59` |
| 总参数 | 12.042 GB（bf16），约 6.0B，与 Fast-WAM LIBERO 权重同样大小 | `H/tree_*.json`；`/home/weiland/projects/openpi/exp/offline_search/rounds/r17/survey3_opus/DECISION.md:7` |
| VAE | Wan2.2 VAE38：48 维潜变量，空间压缩 16 倍，时间压缩 4 倍 | `R/src/longwam/models/wan22/wan_video_vae.py:1436,1463` |
| 文本 | umT5-XXL 嵌入（4096 维，128 个 token），推理时用预先算好的缓存 | `config.yaml:104-106`；`R/src/longwam/inference.py:148-164` |
| 注意力方向 | 视频 query 不读动作 token；动作 query 读历史、部分去噪的未来和整段动作 | `P:L294-298` |
| IDM 推理 | ① 编码干净帧，② 视频专家从噪声去噪 V=4 步到 σ=0.9（不解码成像素），③ 在 [干净帧 \| σ=0.9 的未来] 上预填一次视频 KV，④ 动作专家去噪 A 步，读这份 KV | `longwam_base.py:1603-1697,1927-2009`；`P:L301-328` |
| 步数 | 权重配置是 V4 / A10 / H32；LIBERO 评测脚本默认却是 **H80 / A20**，每 10 步重规划，两者不一致，论文 99.5% 用的是哪一套没写。论文测延迟用的是 V4/A4 | `config.yaml:5,38`；`R/configs/eval/libero.yaml:33-35`；`R/README.md:339,346`；`P:L166-170` |
| token 数（LIBERO 224×448） | 每个潜帧 98 个 token；P48 是 4 个干净帧加 2 个未来帧，共 588；P0 是 1 加 2，共 294。论文测延迟用的就是 LIBERO 网格 [6,7,14]（588 tile，复用后为 392/196） | `config.yaml:40-43`；`R/infra/README.md:181`；`P:L1735-1738` |

### 3.2 实测延迟（论文，RTX 5090，视频 NVFP4、动作 BF16，V4/A4，整段 VAE 计入）

| 配置 | 延迟 | 依据 |
|---|---:|---|
| P0（只用当前帧） | **74.6 ms** | `P:L1755-1776`；`W/data.js:13-17` |
| P48（2.4 s） | 107.4 ms（关掉观测 KV 复用为 112.6 ms） | 同上；`P:L1737-1738` |
| P96 / P192 / P384 | 138.3 / 204.5 / 341.0 ms | 同上 |
| P48 BF16 eager（未优化） | 356.0 ms | `P:L895-897` |
| V2/A2 优化版 | 81.8 ms（RoboTwin 成功率 92.5，V4/A4 为 93.5） | `P:L898-903` |
| Fast-WAM（按它自己的部署配置） | 244.1 ms（与 Long-WAM 的去噪步数不对等） | `P:L891-893,940` |
| **分阶段延迟** | **论文没有给出**（只有总时间和逐项优化后的累计时间）。仅有的一点线索：VAE 卷积调优在 P48 上省了 11.7 ms（119.1 → 107.4），说明 P48 的 VAE 至少占这么多 | `P:L941-1005` |

### 3.3 FLOP 拆分（估算：线性层按 2 × 参数 × token 计；视频专家 5.0B，每 token 约 10 GFLOP；动作专家 1.0B；单帧 VAE 用 survey3 在 meta 设备上实测的 0.47 TFLOP）

| 阶段（LIBERO 224×448，H32，A10，参考实现，没有前缀复用） | Long-WAM P0 | Long-WAM P48 | Fast-WAM |
|---|---:|---:|---:|
| VAE 编码 | 0.47（1 帧） | 约 6.1（13 帧，线性外推） | 0.47 |
| patchify（键的切点） | ≈0 | ≈0 | ≈0 |
| 视频想象 4 步 | 11.8（4 × 294 token） | 23.5（4 × 588） | — |
| 视频预填 | 2.9 | 5.9 | 1.0 |
| 动作 10 步 | 0.64 | 0.64 | 0.64 |
| **合计（TFLOP）** | **约 15.8** | **约 36** | **约 2.1** |
| 完整调用 / 键 | **约 34 倍** | — | 约 4.5 倍 |

Fast-WAM 一列的数字来自 `/home/weiland/projects/openpi/exp/offline_search/rounds/r17/survey3_opus/DECISION.md:57-59`。如果打开硬件 profile 里的干净前缀复用，P0 约 11.9 TFLOP，P48 约 20.4 TFLOP。这个复用只在 `optim/prepare.py` 的硬件 profile 里打开（`R/src/longwam/runtime/optim/prepare.py:311`；`longwam_base.py:1632-1655`）。

### 3.4 当前帧视觉键从哪里取（只适用于 P0 模型）

| 项目 | 内容 | 依据 |
|---|---|---|
| 切点 | 当前帧经 VAE 编码，再经 `video_expert.pre_dit` 的 patchify 得到 98 个 token（维度 3072）。这些 token 与文本、本体感受无关（文本另走 `text_embedding`），切点与我们的 Fast-WAM stage1 完全相同 | `R/src/longwam/models/wan22/wan_video_dit.py:1048,1070-1074`；`/home/weiland/projects/openpi/exp/r17_integration/DESIGN.md:136,145` |
| 键的成本占比 | FLOP 约 3%（0.47 / 15.8）。VAE 是卷积，访存受限，所以按延迟算的占比会高一些，估计 3–8%，需要 GPU 实测 | 上表 |
| P48 模型 | 历史窗口是因果 VAE 编码的，最后一个潜帧依赖历史，取不到只含当前帧的键；要么另算一次单帧 VAE（额外开销），要么用依赖历史的潜帧。而且 P48 模型本身就违反 owner 的规则 | `longwam_base.py:1835-1841` |

## 4. 自己微调零历史版本的成本（单张 H100）

**依据**：我们在 H100 上实测过同架构 Fast-WAM 微调 MT50（单相机 224²，每个样本 147 个视频 token）。冻结视频专家时每次更新 3.50 s（batch 128，36.6 样本/s），16,001 次更新共 15.6 h；视频 LoRA（32×4）每次更新 8.22 s，共 36.5 h（`HANDBACK_C.md:80-87,93`）。Long-WAM 的 P0 窗口是 33 帧原始画面，取 9 帧，编成 3 个潜帧，token 形状与 Fast-WAM 训练完全相同（`R/configs/context/p0.yaml:20-22`；`survey3_opus/DECISION.md:39`）。Long-WAM 每个训练步多一次不求梯度的视频前向，因为第二遍要在 σ=0.9 的未来帧上预填，而且其中的视频 KV 是 detach 的（`longwam_base.py:891-950`），按多 10–30% 计。

| 目标 | 方案 | 估计 | 前提和风险 |
|---|---|---|---|
| **MT50 P0** | 从 LIBERO-IDM 初始化，重置 4 维动作和本体感受层（与我们 Fast-WAM MW 的做法相同），**冻结视频专家**，只算动作损失 | **约 16–20 h**（16k 次更新 × 128） | Long-WAM 训练器没有冻结模式，要改代码（`R/src/longwam/trainer.py` 里没有 freeze 或 LoRA 选项）。视频专家没见过 MW，冻结后 IDM 的想象帧质量没有保证 |
| MT50 P0 | 视频 LoRA r128，视频损失权重 λ_v = 1 | **约 40–44 h** | LoRA 要自己加（仓库只在评测时注入 LoRA：`R/src/longwam/runtime/libero.py:29-83`）。注意动作那一遍的视频 KV 是 detach 的，LoRA 只能从视频损失拿到梯度，所以 λ_v 必须大于 0 |
| **LIBERO P0** | 从 LIBERO-IDM（P48）初始化，形态、归一化都不变，只改窗口长度，视频 LoRA，3 个 epoch（277,713 帧 × 3 ≈ 83 万样本） | **约 31–34 h** | 双相机 224×448 是 294 个 token，每个样本的视频计算约为 MT50 的 2 倍，估计 6.8–7.5 样本/s。P0 下视频专家的时间位置变了，冻结不合适 |
| LIBERO P0 | 照论文配方训 10 个 epoch（约 21.7k 次更新 × 128） | 约 103–113 h | 论文是全参数训练（`R/README.md:484`），6B 参数全参数训练单张 H100 放不下（`survey3_opus/DECISION.md:43`）。论文用的是 16 张 GB200（`P:L600-602`） |
| 两者都需要 | umT5 文本嵌入缓存；LIBERO 数据（`yuanty/LIBERO-fastwam`，共 4.7 GB tar）；MW 数据要转成 Long-WAM 的 LeRobot 格式，或者写适配器；LIBERO 和 MT50 各评 500 局 | — | 训练用的初始化资产下载链接还是 TBD（`R/docs/TRAINING.md:100-105`），只能从已发布的策略权重初始化 |

**移植风险**：这个 release 没有复验 LIBERO 端到端（`R/docs/VERIFICATION.md:183-186,267`）。`native_config()` 先把 EVALUATION 节点挂到 run 上，再往原节点里写 `num_trials` 等字段（`R/src/longwam/evaluation.py:273-278`）。我在 openpi venv 里用 OmegaConf 2.3.0 跑了一条 CPU 单行检查：赋值时节点会被复制，之后写入的字段不会进入 run.EVALUATION。所以审计里发现的字段丢失 bug（`VERIFICATION.md:277-283`）看起来还没修。

## 5. 与 Fast-WAM 相比会失去什么

| 维度 | Fast-WAM（已接入） | Long-WAM（需自训 P0 才合规） | 依据 |
|---|---|---|---|
| LIBERO 只用当前帧的权重 | 有，已发布 | 没有，要自己训 | §1；`survey3_opus/DECISION.md:7,15` |
| MT50 | 我们在训（冻结视频专家，16k 次更新；中途检查 10/16） | 没有，要另训一套 | `HANDBACK_C.md:3`；`/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:19` |
| 零历史时的 LIBERO-Long | 95.2（论文表） | 94.5 | `P:L519-524`；`P:L778-780` |
| 每次调用 | VAE、1 次视频预填、K 步动作 | VAE、4 步视频想象、1 次预填、K 步动作；FLOP 约 7.5 倍 | §3.3 |
| 缓存收益（完整调用 / 键） | 约 4.5 倍（FLOP） | 约 34 倍（FLOP）。这是 Long-WAM 唯一的加分项 | §3.3 |
| 减步旋钮 | 动作步数 K | 视频步数 V 和动作步数 A 两个旋钮（V2/A2 在 RoboTwin 上掉 1.0 个点） | `P:L898-903,939` |
| 模型族多样性 | Wan2.2-5B MoT WAM | **同一族**：代码来自 FastWAM，视频专家和动作专家配置都相同 | `R/README.md:675-677`；`R/licenses/FastWAM-MIT.txt`；§3.1 |
| 接入工作量 | 适配器已写好 | stage1 和 stage3 可以照搬 Fast-WAM 的；stage2 要加 4 步想象和预填；还要接一套新的 MW、LIBERO 运行环境 | `DESIGN.md:129-145` |

## 6. 其他不一致（不影响结论）

- GR1 的 2.4 s 成功率：表 4 写 63.3（`P:L680-681`），但正文和图 6 写 66.3，63.3 是 0 s 的数（`P:L782`；`W/data.js:7`）。摘要里 “0.0 → 19.2 s，63.3 → 78.7” 与图 6 一致（`P:L103-104`）。
- 论文 Table 6 的延迟是在固定输入（LIBERO 网格）上测的，但同一张表的成功率一列用的是 RoboTwin（`P:L877-903`；`R/infra/README.md:181`）。

## 7. 证据存放

本机 `/home/weiland/projects/openpi_ext/scratch/longwam_probe/`：`paper/`（PDF、HTML、纯文本）、`web/`（项目页 HTML 和 JS）、`repo/`（稀疏克隆，不含媒体文件）、`hf/`（API 列表、文件树、模型卡、配置、LIBERO 的 info.json）、`tools/html2txt.py`。
