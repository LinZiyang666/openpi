# 跨领域迁移调研：流式 / 交互式生成媒体与世界模型

> 领域范围：实时 / 自回归视频扩散（StreamDiffusion、CausVid、Self-Forcing 系、Krea Realtime、LTX/Wan 流式变体）、交互式游戏世界模型（GameNGen、Oasis、Matrix-Game 2.0、Genie 系、Solaris）、实时 video-to-video 与 talking head（LiveTalk、StreamAvatar 系）、以及把世界模型当"模拟器"用的 agent 训练 / 策略评测场景（DIAMOND、RoboWorld、WorldGym、Horizon Imagination）。
> 依据：`logs/cache_transfer/briefing.md`、`docs/iclr/iclr_paper/arxivd.tex` L64–344、`logs/session_handoff.md` §1.2。所有外部论断附 URL；没有文献数字的地方明写"不确定"。
> 撰写：2026-09-15。只读仓库，未跑实验。

---

## 0. 先把三类"cache"分清楚（审稿人第一个会问的）

| 类别 | 代表 | 复用发生在 | 与我们的关系 |
|---|---|---|---|
| **同一样本内、跨去噪步的特征缓存** | DeepCache、TeaCache（[arXiv 2411.19108](https://arxiv.org/abs/2411.19108)）、FasterCache、LeMiCa、WorldCache（[arXiv 2603.22286](https://arxiv.org/pdf/2603.22286)，专做视频世界模型，Cosmos-Predict2.5 上 2.3× 且保 99.4% 质量） | 一次去噪链内部，步 t 与步 t+1 之间 | **正交、可叠加**：它们让每个 miss 更便宜（缩小 c_0 内部的 Δ_j），不改变"要不要算这一帧"的决策 |
| **时序 warm-start / 流水线级复用** | StreamDiffusion 的 Stream Batch 与 RCFG（[arXiv 2312.12491](https://arxiv.org/abs/2312.12491)，4090 上 91 fps）；Self-Forcing 的 rolling KV cache（[arXiv 2506.08009](https://arxiv.org/html/2506.08009)）；TempCache 的 KV 压缩（[arXiv 2602.01801](https://arxiv.org/html/2602.01801v1)，长序列 5–10×） | 相邻帧 / 相邻 chunk 之间的上下文与流水线并行 | **正交**：它们决定"上下文怎么带"，我们决定"这一帧的去噪要不要跑、跑几步" |
| **跨样本语义检索 + 跳步（approximate caching）** | NIRVANA（[arXiv 2312.04429](https://arxiv.org/html/2312.04429)，T2I，生产 prompt 上 88% hit、省 21% GPU）；Chorus（[arXiv 2604.04451](https://arxiv.org/html/2604.04451)，视频 DiT 服务，VidProM 上 hit 率 20%→50%，4-step 蒸馏模型 1.23×）；FlexCache（[arXiv 2501.04012](https://arxiv.org/pdf/2501.04012)） | **不同请求**之间：按 prompt embedding 相似度取回中间噪声态、跳过前 K 步 | **这是我们的近亲**。差别：它们按**文本 prompt** 做 batch 级、单次请求的检索；我们是**每帧一个决策**、key 含动作与视觉上下文、有 hit 档（整帧回放）、阈值由 RIT 一个 δ 统一定、有状态门、且必须闭环测漂移。NIRVANA 的"相似度分桶 → K∈{5,…,25}"就是一张**手工写死的 RIT ladder**，这既是我们的 novelty 空间也是最强攻击点（见 §6） |

---

## 1. 领域内最贴合的具体系统

### 1.1 Self-Forcing / CausVid 系（实时自回归视频扩散的事实标准）

- **模型**：Wan2.1-T2V-1.3B 蒸馏成 block-causal 4-step 学生；每次生成 3 个 latent 帧的 chunk；H100 上 17.0 fps、首帧 0.69 s（[arXiv 2506.08009](https://arxiv.org/html/2506.08009)）。代码与权重开源（[github.com/guandeh17/Self-Forcing](https://self-forcing.github.io/)）。CausVid 是前身（4-step，9.4 fps，[github.com/tianweiy/CausVid](https://github.com/tianweiy/CausVid)）。后续 Rolling Forcing、Self-Forcing++、Causal Forcing++、Causal-rCM、Krea Realtime 14B（B200 上 11 fps，4 步，[krea.ai/blog](https://www.krea.ai/blog/krea-realtime-14b)）都是同一骨架。
- **每决策（每 chunk）的成本拆分**：T=4 次 DiT 前向（去噪）+ **1 次 DiT 前向在 t=0 把干净输出写进 KV cache**（论文算法：`x^0 ← G(x_t; t, KV)` 后再 `kv ← G^KV(x^0; 0, KV)`）+ VAE 解码。Flash-VAED 的 Table 5 给出 Self-Forcing-Wan-1.3B 的分段延迟：text encoding 0.598 / denoising 6.254 / video decoding 3.172（单位与"每 chunk 还是每段视频"论文未明说；解码占 **31.6%**，而原版 50 步 Wan 1.3B 里解码只占 2.3%）（[arXiv 2602.19161](https://arxiv.org/html/2602.19161v1)）。
- **含义**：蒸馏之后"头"已经很薄（5 次前向），VAE 解码成了第二大项。后面 §2 的成本模型全部建立在这个拆分上。

### 1.2 交互式游戏世界模型：GameNGen / Oasis / Matrix-Game 2.0 / DIAMOND

- **GameNGen**（DOOM，[arXiv 2408.14837](https://arxiv.org/html/2408.14837)）：SD1.4 骨架，条件 = 过去 64 帧（16 帧后饱和）+ 动作序列，**4 步去噪**；TPU-v5 上"每个去噪步和一次 autoencoder 评估各 10 ms"，合计 50 ms/帧 = 20 fps。**有真实游戏引擎当 oracle**：4 步 PSNR 32.58 / LPIPS 0.198；人类评委仅 58% 能分辨真假。不开源。
- **Oasis 500M**（Minecraft，[github.com/etched-ai/open-oasis](https://github.com/etched-ai/open-oasis)、[HF Etched/oasis-500m](https://huggingface.co/Etched/oasis-500m)）：ViT-VAE + DiT，Diffusion Forcing 训练；`generate.py` 默认 **DDIM 10 步、未蒸馏**（`ddim_noise_steps=10`, `stabilization_level=15`）；全尺寸版 H100 上 20 fps、单帧 47 ms（[oasis-model.github.io](https://oasis-model.github.io/)）。10 步未蒸馏 = 与 π0.5 同构，是我们 K=3 ladder 最自然的落点，但没有 action 对齐的 ground truth。
- **Matrix-Game 2.0**（[arXiv 2508.13009](https://arxiv.org/html/2508.13009v1)、[HF Skywork/Matrix-Game-2.0](https://huggingface.co/Skywork/Matrix-Game-2.0)）：1.8B，SkyReels-V2-I2V-1.3B 加动作模块，few-step 蒸馏；训练 4 步、推理 **3 步**，H100 25 fps；加速表：VAE caching 15.49 fps → 砍一半动作模块 21.03 fps → 4→3 步 25.15 fps，且质量指标不掉（image quality 0.61 / temporal 0.94 不变）。GameWorld Score 基准：视觉质量、时序质量、动作可控性（键鼠准确率）、物理理解。ForgeWM 在它上面蒸出 1/2/4 步三个学生（[arXiv 2608.14022](https://arxiv.org/html/2608.14022)，见 §4）。
- **DIAMOND**（Atari 100k / CS:GO，[arXiv 2405.12399](https://arxiv.org/html/2405.12399)、[diamond-wm.github.io](https://diamond-wm.github.io/)）：像素空间 EDM 去噪 UNet，**3 步**（3 NFE/帧，对比 IRIS 16 NFE），每次观测预测 12.7 ms；agent 完全在世界模型内训练，Atari 100k HNS 1.46。**有真实模拟器当 oracle**，且明确记录了 1 步的多峰塌缩（Boxing 对手动作不可预测 → 1 步"在可能结果之间插值、输出模糊"，多步"驱动到某一个 mode、清晰"）。全开源，硬件需求极低。
- **Solaris**（多玩家 Minecraft，[arXiv 2602.22208](https://arxiv.org/html/2602.22208)）：Matrix-Game 2.0 改多玩家，多个玩家共享同一场景做联合去噪——天然的"多 agent 共享库"场景。

### 1.3 Talking head / 虚拟人流式生成

- **LiveTalk**（[arXiv 2512.23576](https://arxiv.org/html/2512.23576)、[github.com/GAIR-NLP/LiveTalk](https://github.com/GAIR-NLP/LiveTalk)）：Thinker/Talker = Qwen3-Omni 出流式音频 + motion prompt；Performer = OmniAvatar-1.3B（Wan2.1 变体）蒸馏成 4 步、3 latent 帧一块，24.82 fps、首帧 0.33 s；KV 里固定 3 块"identity sink"+2 块滚动。指标 Sync-C/D、FID、FVD。
- **Decoupled Self-Forcing 流式 talking head**（[arXiv 2609.10317](https://arxiv.org/html/2609.10317)）：77M 因果 motion generator + SD 渲染器 4 步，15.4 fps，MEAD FVD 185.9；作者明言瓶颈在渲染器不在 motion。
- 这一支的"前缀"（音频 LLM）虽贵但是**按会话摊销、与视频帧不同步**，不是我们意义上"每决策都跑一遍再出 query"的前缀；视频侧结构同 1.1。

### 1.4 世界模型当模拟器：策略评测 / agent 训练

- **RoboWorld**（[arXiv 2607.01060](https://arxiv.org/html/2607.01060)）：8 个策略复现 RoboArena 排名，Pearson ρ=0.970，耗 100 H100 小时；**WorldGym**（[arXiv 2506.00613](https://arxiv.org/html/2506.00613.pdf)）ρ=0.78（17 任务）；**DreamDojo** r=0.995；**PlayWorld** r=0.877；**Scalable Policy Evaluation with Video World Models**（[arXiv 2511.11520](https://arxiv.org/html/2511.11520v3)，Cosmos-Predict2-2B）ρ=0.83–0.88。这些工作把"世界模型推理成本"当既定事实，没有人做检索跳算。
- **Horizon Imagination**（[arXiv 2602.08032](https://arxiv.org/html/2602.08032)）：直接把"扩散世界模型内 on-policy rollout 太贵"当问题，靠并行去噪 + 稳定动作采样把 Atari 100k 训练从 27 h 压到 19 h，去噪预算减半不掉分——这是该场景最近的"省算力"竞品，路线是调度不是检索。
- **ImageWAM**（[arXiv 2606.19531](https://arxiv.org/abs/2606.19531)）：世界-动作模型不用生成视频只做图像编辑，FLOPs 1/6、延迟 1/4——说明这条线上"省世界模型算力"是活跃需求。

---

## 2. 映射表

| 我们的符号 | VLA（π0.5） | 流式 AR 视频扩散（Self-Forcing / Matrix-Game / LiveTalk） | 像素/轻 VAE 世界模型（GameNGen / DIAMOND） |
|---|---|---|---|
| **共享前缀 c_pre** | 视觉编码器 + VLM 前缀（s1+s2，占 60.6%），每决策必跑 | **几乎不存在**：text encoder 每会话一次；上下文由上一 chunk 已算好的 KV cache 承担。每决策必付的只有"把选定帧写进 KV 的 1 次 t=0 前向"+ VAE 解码 | 历史帧以 latent/像素直接拼接，**无 KV**；每决策必付的只有解码（GameNGen 10 ms/50 ms）或几乎为零（DIAMOND 像素空间） |
| **tap point** | 前缀算完处 | 去噪循环入口：此时手里有上一帧 latent、rolling KV、当前动作/音频/文本 embedding | 同左，且更干净（历史帧 latent 就是 key） |
| **头** | flow-matching 动作专家，10 步 Euler | block-causal DiT，蒸馏后 3–4 步；Oasis 500M 未蒸馏 DDIM 10 步 | UNet 去噪 3–4 步（EDM / DDIM） |
| **中间去噪状态** | 第 j 步的动作噪声态 | 第 j 步的 chunk latent（Solaris 的 checkpointed Self-Forcing 已经在存这个）；NIRVANA/Chorus 存的就是它 | 第 j 步的帧噪声态 |
| **tiers** | tier 0 miss / 中间 warm / tier K hit 回放 chunk | tier 0 全 T 步；warm = 从检索 chunk 的第 j 步噪声态、在**当前 KV + 当前动作**条件下跑剩余步；hit = 直接把库里的干净 latent 当本 chunk 输出，**仍需 1 次 KV 更新前向 + 解码** | hit = 回放库里的帧（可存解码后像素，则连解码也免） |
| **key 字段** | 相机 token 池化、本体状态、指令 embedding | 上一 chunk latent 的池化（=相机字段）、**动作/键鼠/音频块**（=指令，做 IVF cell）、相机位姿/时间戳（=本体，WorldMem 正是按位姿+时间戳检索记忆，[arXiv 2504.12369](https://arxiv.org/abs/2504.12369)） | 同左；Atari 里 (帧, 动作) 对是离散的 |
| **分数 s_t** | 式 norm/fuse | 同式：latent cosine + 位姿距离 + 动作精确匹配，留一估 μ/τ | 同 |
| **偏差 D_a** | tier a 动作 chunk 与纯噪声全算 chunk 的标准化偏差 | tier a 输出帧与全 T 步输出帧的 latent MSE / LPIPS；**有 oracle 时可直接对真模拟器帧算 PSNR/LPIPS**（GameNGen、DIAMOND、Matrix-Game 的 UE 数据） | 同，且 oracle 更便宜 |
| **成功度量 SR / 容差 ε** | 任务成功率 | 媒体线：VBench / FVD / 人类偏好 / Sync-C；世界模型线：PSNR·LPIPS 对 oracle、动作可控性（GameWorld Score）、人类分辨率（GameNGen 58%） | **闭环 agent 指标**：在 cache 化世界模型里训练/评测出的 agent 回报或 SR，与真环境的 Pearson ρ（RoboWorld 0.97、WorldGym 0.78 给出了 ε 的自然口径） |
| **成本模型 c_a** | c_pre + Σ Δ_j | c_hit = KV 前向 + 解码；c_a = c_hit + (T−j_a)·DiT；c_0 = c_hit + T·DiT | c_hit ≈ 解码（或 ≈0）；c_0 = 解码 + T·UNet |
| **闭环性** | 连续替换 → 状态漂移 | **该领域的核心病**：自回归误差累积（Self-Forcing 整篇为此而写；GameNGen 用噪声增强抗漂移；Krea 用 KV 重算 + 首帧锚定）。回放帧会写进 KV，之后所有帧都条件于它 —— 与我们"单步安全 ≠ 连续安全"完全同构 | 同，且 agent 在里面学，漂移直接变成 agent 学坏 |
| **状态门** | 命中成串 | 站着不动 / 重走同一条路 / 多玩家同场景 → 命中成串；连续 L 次 hit 强制 miss ↔ 业界"定期 KV 重算 / 首帧重编码"防漂移惯例 | 同；Atari 里 no-op 序列与确定性段落成串 |

---

## 3. 打分表 A–G

| 项 | 分 | 依据 | 引用 |
|---|---|---|---|
| **A 结构匹配** | **2** | 头（3–10 步、有中间态）、tap point、免费 query（上一帧 latent + 动作 + KV）全齐；但"昂贵共享前缀"在主流 AR 视频扩散里**不存在**——这对我们反而有利（c_pre 小），只是与简报的字面模板不同。SCD（Adobe/MIT）证明可以把 Wan 拆成"25 层因果编码器每帧一次 + 10 层去噪解码器跑 T 次"，形式上与我们完全同构，但那时头只占 10/35 层、hit 省得少；且代码未放 | [Self-Forcing](https://arxiv.org/html/2506.08009)；[SCD 2602.10095](https://arxiv.org/html/2602.10095)；[GameNGen](https://arxiv.org/html/2408.14837) |
| **B 成本余量** | **2** | hit 地板：GameNGen ≈ 10/50 = **20%**（只剩解码）；DIAMOND 像素空间 ≈ **接近 0**（回放帧即可）；Self-Forcing-Wan ≈ (1 KV 前向 + 解码)/(5 前向 + 解码) ≈ (1.25+3.17)/10.0 ≈ **44%**（按 Flash-VAED 分段延迟推算，单位假设见 §1.1；换更快的解码器可压到 ~25%）。傻基线（未重训直接砍到 1 步）**确实掉点**：Self-Forcing 1 步 VBench 84.31→80.62、"严重模糊、2.5 s 起明显误差累积"；GameNGen 4→1 步 PSNR 32.58→25.47；DIAMOND 1 步在随机性游戏里塌成均值。**但**重训的 1 步学生把差距补回大半（ASD 1 步 83.89；GameNGen 蒸馏 1 步 31.10 dB；ForgeWM-1 LPIPS 0.653 vs 4 步 0.617，72 fps）——"训练一次换永久省算力"是审稿人手里的强基线 | [ASD 2511.01419](https://arxiv.org/html/2511.01419)；[GameNGen](https://arxiv.org/html/2408.14837)；[DIAMOND](https://arxiv.org/html/2405.12399)；[ForgeWM](https://arxiv.org/html/2608.14022)；[Flash-VAED](https://arxiv.org/html/2602.19161v1) |
| **C 局部性 / 命中潜力** | **2** | 结构论证很强：游戏世界的 (状态, 动作)→下一帧在确定性引擎里是**函数**（Atari、DOOM 的确定性部分；DIAMOND 指出玩家自己控制的角色 1 步就能准确预测）；重访同一地点是常态到已有专门基准（LoopNav、WorldMem、WorldPack）；多玩家共享场景（Solaris）与 agent 训练从同一批 reset 状态起 rollout 都是"多 agent 共享库"。**但没有任何文献直接测过帧级命中率**；能引用的只有 prompt 级：NIRVANA 生产 T2I 88% hit（阈值 0.65）、Chorus VidProM 1500 请求后 ~50%。帧级数字**不确定，必须实测** | [NIRVANA](https://arxiv.org/html/2312.04429)；[Chorus](https://arxiv.org/html/2604.04451)；[WorldMem](https://arxiv.org/abs/2504.12369)；[LoopNav](https://arxiv.org/pdf/2505.22976)；[Solaris](https://arxiv.org/html/2602.22208) |
| **D 容忍度与闭环** | **3** | 近似帧可接受（PSNR 29–32 ≈ JPEG 有损；人类 58% 分辨率）；有 oracle（真游戏引擎 / Atari 模拟器）；有 SR 型闭环度量（agent 回报、策略评测 Pearson ρ）；**漂移是该领域头号问题**，回放帧写进 KV 后续全体条件于它，"单步替换安全 ≠ 连续替换安全"在这里不是我们要论证的性质而是公认事实 | [GameNGen](https://arxiv.org/html/2408.14837)；[Self-Forcing](https://arxiv.org/html/2506.08009)；[RoboWorld](https://arxiv.org/html/2607.01060)；[WorldGym](https://arxiv.org/html/2506.00613.pdf) |
| **E RIT 可用性** | **3** | D_a = tier a 帧 vs 全步帧的 LPIPS/latent MSE，离线一行代码；有 oracle 时还能对真帧算，比 VLA 的"动作偏差代理成功损失"更直接。偏差随相似度单调是 NIRVANA 分桶表已隐含的经验事实；一个 δ 切所有 cut、按 fps 预算反解都照搬。状态门有对应物（站立/重访/多玩家），强制 miss 对应业界 KV 重算惯例 | [NIRVANA](https://arxiv.org/html/2312.04429)；[Krea Realtime](https://www.krea.ai/blog/krea-realtime-14b) |
| **F novelty 地形** | **2** | 步内缓存（TeaCache/WorldCache）、KV 压缩（TempCache/Forcing-KV）正交可叠加；**跨请求 approximate caching（NIRVANA/Chorus/FlexCache）是近亲**，但全是 batch 文本 prompt 级、无 hit 档、K 靠手工分桶、不测闭环漂移；流式/交互式逐帧、动作为 key、tiers+RIT+gate、闭环评测在文献里**空白**。最强反驳见 §6 | [WorldCache](https://arxiv.org/pdf/2603.22286)；[TempCache](https://arxiv.org/html/2602.01801v1)；[Chorus](https://arxiv.org/html/2604.04451) |
| **G 我们做得动** | **3** | DIAMOND 全开源、Atari 100k 一张 1080/A5000 就能跑、每帧 12.7 ms、有 oracle 与闭环 agent 指标；open-oasis 500M 权重开放且 10 步未蒸馏（K=3 ladder 直接套）；Matrix-Game 2.0 / Self-Forcing 权重开放、1.3–1.8B、单 H100 实时；每个前沿点评测成本 = 几百个 episode 的世界模型 rollout（DIAMOND 分钟级；Matrix-Game 按 GameWorld Score 小时级） | [DIAMOND](https://diamond-wm.github.io/)；[open-oasis](https://github.com/etched-ai/open-oasis)；[Matrix-Game-2.0](https://huggingface.co/Skywork/Matrix-Game-2.0)；[Self-Forcing](https://self-forcing.github.io/) |

**合计 17/21**（VLA 上按同表自评约 A3 B0 C2 D3 E3 F2 G3 = 16，差别在 B 从 0 变 2、A 从 3 变 2）。

---

## 4. "减迭代次数"傻基线：文献已经做过，结论分两层

**第一层：不重训、直接砍步 —— 会掉，而且掉在多峰上。**

- Self-Forcing 4 步学生直接跑 1 步：VBench 84.31 → 80.62，定性"严重模糊、明显误差累积、2.5 s 起出现伪影"（[ASD, arXiv 2511.01419](https://arxiv.org/html/2511.01419) Table）。
- GameNGen 步数消融：1 步 PSNR 25.47 / LPIPS 0.255，4 步 32.58 / 0.198，8 步以上不再涨（[arXiv 2408.14837](https://arxiv.org/html/2408.14837) Table 1）。
- DIAMOND：EDM 参数化下 1 步在 Breakout 这种确定性游戏里"非常稳"，但 Boxing 里对手动作多峰，1 步"在可能结果间插值、输出模糊"，故全局用 3 步（[arXiv 2405.12399](https://arxiv.org/html/2405.12399) Fig. 4）。
- Matrix-Game 2.0：4→3 步无损（[arXiv 2508.13009](https://arxiv.org/html/2508.13009v1) Table 3），再往下没报。

这与 LIBERO 上"k=1 = teacher"的现象**相反**：视频帧的条件分布在有对手/NPC/其他玩家/物理随机性时是多峰的，一步 Euler 取到条件均值就是模糊帧，模糊帧写进 KV 后再累积。所以 warm 档在这里**有 counterfactual**：从检索到的真实样本（清晰、在某个 mode 上）出发跑 1–2 步，理论上比从噪声出发跑 1 步更能落在 mode 上。这是 briefing §2 判断"输出多峰时 warm 档才有存在理由"在本领域的正面证据。

**第二层：重训一个 1 步学生 —— 差距被补回大半，这是真正的强基线。**

- ASD 1 步学生 VBench 83.89（vs 4 步 84.31，差 0.4），且人类偏好 96% 胜过 Self-Forcing 1 步（[arXiv 2511.01419](https://arxiv.org/html/2511.01419)）。
- GameNGen 蒸馏 1 步 31.10 dB（vs 4 步 32.58，差 1.5 dB），作者仍选 4 步作主结果，说明"有可感知代价"（[arXiv 2408.14837](https://arxiv.org/html/2408.14837)）。
- ForgeWM 三个学生：1 步 LPIPS 0.653 / 72 fps，2 步 0.617 / 50 fps，4 步 0.617 / 32 fps；人类视觉偏好仍以 4 步为优（68.8%）（[arXiv 2608.14022](https://arxiv.org/html/2608.14022)）。

**含义**：我们的对照必须同时列两条——训练无关的"砍步"（我们能赢）和"蒸馏 1 步学生"（我们需要证明 hit 档能省下它省不掉的那 1 次前向 + 能修它残余的 1.5 dB / 0.4 VBench）。对蒸馏学生而言，砍步的 IR 地板 = (1 前向 + KV 前向 + 解码)/(T+1 前向 + 解码)：Self-Forcing ≈ 57%，GameNGen（无 KV）= 20/50 = 40%；而 hit 地板分别 ≈ 44% 与 20%。**Wan 系余量只有 ~13 pp，GameNGen/DIAMOND 系余量 20–35 pp**。这决定了 §7 的选型。

---

## 5. 预期收益（文献能撑的部分 + 明确标注的估计）

| 量 | Wan 系流式媒体（Self-Forcing / LiveTalk / Matrix-Game 2.0） | 像素/轻 VAE 世界模型 + agent 在环（DIAMOND / GameNGen 型） |
|---|---|---|
| hit 成本地板 c_K/c_0 | ≈ 44%（Wan VAE）；换 Flash-VAED 类解码器估 ~25%（估计，[Flash-VAED](https://arxiv.org/html/2602.19161v1) 报解码大幅加速但未与 Self-Forcing 合测） | GameNGen 20%；DIAMOND 像素回放 ≈ 检索开销（估 <5%，**不确定**） |
| 砍步地板（蒸馏 1 步学生） | ≈ 57% | 40%（GameNGen）/ 33%（DIAMOND 3→1 步） |
| 帧级命中率 | **不确定**。prompt 级参考：NIRVANA 88%、Chorus ~50%；帧级在 talking head（同一说话人、有限视素）与站立/重访场景可能高，在自由文本驱动视频里可能很低 | **不确定但结构上高**：确定性引擎 + 有限离散动作 + agent 训练从同一批 reset 起；Breakout 类可能极高，Boxing 类取决于对手随机性。必须实测 |
| 可达 IR（估） | 命中 50% 且全 hit 时 ≈ 0.5·0.44 + 0.5·1 = **72%**；不如 1 步学生的 57% | 命中 50% 时 ≈ 0.5·0.05 + 0.5·1 ≈ **53%**（DIAMOND）；命中 80% → **24%**，低于任何砍步方案 |
| SR 损失 | 用 VBench/FVD/人类偏好；有 oracle（UE/Minecraft）时 PSNR/LPIPS。无文献数字 | 用 agent 回报（Atari HNS）或策略评测 ρ；无文献数字 |
| 与 VLA 对比 | 比 VLA 差：hit 地板 44% vs 15%，砍步在这里还能重训补回 | **比 VLA 好**：hit 地板 ≤20% 且砍步真掉点（多峰），warm 档有 counterfactual；且闭环指标（agent 学得好不好）比 SR 更敏感于漂移，故事更硬 |

一句话：**收益能不能"明显"取决于选哪一支**。Wan 系被蒸馏 + VAE 解码 + KV 更新三件事把 hit 地板抬到 ~44%，余量薄；世界模型做模拟器那一支（像素空间或轻解码、有 oracle、多峰、agent 在环）余量大。

---

## 6. novelty 地形与最强反驳

**已有工作占的位置**

1. 步内缓存 / KV 压缩 / 稀疏注意力（TeaCache、WorldCache、TempCache、Light Forcing [arXiv 2602.04789](https://arxiv.org/pdf/2602.04789)）：让每次去噪更便宜。正交，我们应把它们当 c_0 的一部分叠上去，并在论文里明说。
2. 跨请求 approximate caching（NIRVANA、Chorus、FlexCache、以及针对它的攻击研究 [arXiv 2508.20424](https://arxiv.org/pdf/2508.20424) 说明已有部署）：**最近亲**。它们的 K 由相似度分桶手工定（NIRVANA：>0.95→25 步 … >0.65→5 步）、只做 batch 单次生成、无 hit 档、无闭环。我们剩下的新东西：逐帧决策 + 动作/位姿 key；hit 档与"回放会污染 KV"的闭环分析；一个 δ 定所有 cut 并按 fps 预算反解；状态门；R_C(ε) 定义在 agent 指标上。
3. 记忆型世界模型（WorldMem、WorldPack、Composition of Memory Experts [arXiv 2605.18813](https://arxiv.org/html/2605.18813v1)）：按位姿检索过去帧**用来提高一致性**，不省算力。我们可以把 cache 讲成"既是记忆又是算力"，但要防被说成 WorldMem 的推理侧省略版。
4. Horizon Imagination：同一痛点（世界模型内 rollout 太贵），走调度路线。是我们在 agent 训练场景的直接对照。

**审稿人最强的三板斧**

- **"蒸馏一次就够了"**：ASD 1 步学生只差 0.4 VBench，ForgeWM-1 72 fps；训练无关的检索方案在 Wan 系里省不过重训。回应：只在 hit 地板 ≪ 砍步地板的结构（无 KV 或轻解码）上主打；并展示 warm 档从真实样本起跑能修 1 步学生残余的多峰塌缩（要有 Boxing 类的实证）。
- **"这不过是确定性游戏的 memoization 表"**：在 Breakout 上 (帧, 动作)→帧是查表。回应：正因如此才要 tiers 与 RIT——随机成分（对手、NPC、物理）决定 hit 何时失效，δ 把"查表"与"重算"之间的边界用离线偏差量化，并在 agent 回报上闭环验证；纯 memoization 没有这个边界。
- **"命中率没证据"**：文献无帧级命中率。回应：首个实验就测（§7），并报告命中率随库规模、随策略熵的曲线。

---

## 7. 首个实验方案（≤ 2 周）

**选型**：DIAMOND（Atari 100k）为主，理由：全开源、任何一张 GPU 可跑、3 步 EDM 与 10 步 DDIM 均可配置、**真模拟器当 oracle**、Boxing/Breakout 天然给出多峰 vs 确定性对照、闭环指标 = agent 回报、每个前沿点分钟级。备选：open-oasis 500M（10 步未蒸馏，最像 π0.5，但无 oracle、无闭环指标，只做媒体线的 IR 曲线）。

**第一步（第 1–3 天）：傻基线 + 成本拆分实测**
- 在 DIAMOND 官方 ckpt 上测每帧 UNet 单步耗时、（若用 CS:GO 模型）解码耗时，得 c_0 与 Δ_j；测 k=1/2/3 步的 agent 回报（用已训练 agent 在世界模型里评测，不重训）与对 oracle 的 PSNR/LPIPS，Breakout 与 Boxing 各一。
- 同时在 Self-Forcing-Wan-1.3B 上实测 4 次去噪 / 1 次 KV 更新 / VAE 解码的毫秒分段（H100），把 §5 的 44% 变成实测数，决定要不要碰媒体线。

**第二步（第 4–8 天）：collect + shadow + 帧级命中率**
- 用已训练 DIAMOND agent 自跑 N 局入库（key = 帧 latent/像素池化 + 动作 + 可选帧计数），另跑校准局做 shadow 行；报告命中率随库规模、随 ε-greedy 熵的曲线；对每个 tier 离线算 D_a（对全步帧与对 oracle 帧各一），画 q_a(s) 看单调性。

**第三步（第 9–14 天）：第一条闭环前沿**
- K=2（miss / warm 1 步 / hit），扫 δ 5 个点，每点在 cache 化世界模型里评测 agent 回报（≥ 100 局），画 (IR, 回报) 前沿，叠上 k=1/2/3 砍步点与"1 步蒸馏学生"（若时间够，用 DIAMOND 自带脚本蒸一个）。带状态门 vs 不带各一条。
- 判定：若 Boxing 上 warm 档在同 IR 下回报高于砍步、且 hit 地板 <10% 实测成立 → 写 GO 报告；若命中率 <30% 或 warm 无优势 → 转 NO。

---

## 8. 结论

**MAYBE，条件 GO 于"世界模型当模拟器 / agent 在环"这一支；对 Wan 系实时媒体（Self-Forcing / LiveTalk / Matrix-Game 类）为 NO。**

一句话理由：主流流式视频扩散没有昂贵前缀，query 白拿、中间态现成、闭环漂移是公认头号病、傻基线在多峰场景真掉点、离线 RIT 信号比 VLA 更直接——这些都比 VLA 好；但 Wan 系被"few-step 蒸馏 + VAE 解码 + 每帧 KV 更新"三件事把 hit 地板抬到 ~44%、砍步地板 ~57%，余量只有十几个百分点且重训 1 步学生几乎无损；只有像素空间 / 轻解码、有 oracle、agent 在环的世界模型（DIAMOND / GameNGen 型）hit 地板 ≤20% 且砍步真掉点，收益才可能"明显"。帧级命中率文献空白，是首个实验必须先回答的量。
