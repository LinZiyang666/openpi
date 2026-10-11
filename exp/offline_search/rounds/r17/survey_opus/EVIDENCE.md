# 证据：非 VLA 模型与第二个 benchmark 选型

2026-10-08。独立完成：没有打开同轮其他调研目录、提示词目录（除本轮 brief）或 r17_ops。
一手来源 = 论文正文、HF API 文件清单（`/api/models/<repo>?blobs=true`，逐文件字节数）、safetensors 头（HTTP range 只取头部，未下载权重）、GitHub 源码；我们自己的数字来自 `logs/*.log.md` 与 R15/R16 报告。
CPU 探针都在 `taskset -c 26-33,70-77 env CUDA_VISIBLE_DEVICES=` 下跑，没有用 GPU、没有起服务、没有写本目录以外的文件。
标记：【实测】= 我们的日志或本次 CPU 探针；【公开】= 论文 / 官方仓库原文；【估算】= 我按 FLOPs 或比例推的，必须先实测再写进论文。

---

## 0. 总判断（为什么方案长这样）

| 问题 | 答案 | 依据 |
|---|---|---|
| 有没有"贵的非 VLA 模型"同时在 LIBERO 和一个轻量、适合缓存的 benchmark 上有官方权重？ | **没有** | §3 逐个查：Cosmos Policy（LIBERO / RoboCasa / ALOHA）、LingBot-VA（LIBERO-Long / RoboTwin）、Fast-WAM（LIBERO / RoboTwin）、X-WAM（RoboCasa / RoboTwin）、DreamZero（DROID / AgiBot）、Cosmos3 策略（DROID）、GigaWorld-Policy-0.5（AgileX 预训练）、VPP（只有 CALVIN）、JEPA-WM（MetaWorld 只有 Reach / Reach-Wall） |
| MetaWorld 上有哪些可信的非 VLA 权重？ | 只有 JEPA-WM / DINO-WM 的两个 reach 任务（58.2% / 41.6%）；其余是单任务或来路不明的第三方 | §3.9、§3.10 |
| CALVIN 上有哪些贵的非 VLA 权重？ | 只有 VPP（官方，ABC→D 4.33） | §3.8 |
| 自己把 Cosmos 微调到 MetaWorld 可行吗？ | 不可行：它在 LIBERO 的训练是 40K 步、64×H100、48 小时（≈3,072 卡时） | Cosmos Policy 论文；HF config.json `"hardware": "64x H100"` |

所以：非 VLA 模型在 LIBERO 上跑（官方权重齐全），π0.5 在 LIBERO + 第二个 benchmark 上跑；第二个 benchmark 按"适合缓存、轻、π0.5 有权重、已接入"选 MetaWorld。

---

## 1. 参照：我们现有两个 VLA 的"命中能省多少"【实测】

来源：`exp/dispatch_surface/analysis/analytic_cost.py`（π0.5 冻结的 CUDA-graph 分阶段开销）；`exp/libero_groot/config/rit/cost_groot_libero_measured.json`（GR00T，RTX 4090，CUDA graph，nsys 核过）；R16 报告"开销口径"一节的视觉系数 .152 / .148。

| 模型 | 视觉（命中仍要付） | 完整调用（默认步数） | 完整调用（1 步） | 完整:命中 | 每次命中省 GPU |
|---|---|---|---|---|---|
| π0.5（4090） | 10.26 ms | 10.26 + 27.69 + 29.57 = 67.52 ms | 10.26 + 27.69 + 2.96 = 40.90 ms | 6.6× / 4.0× | 57.3 / 30.6 ms |
| GR00T N1.5（4090） | 6.15 ms | 6.15 + 7.19 + 8×3.513 = 41.44 ms | 6.15 + 7.19 + 3.51 = 16.85 ms | 6.7× / 2.7× | 35.3 / 10.7 ms |

这是"有利可图"的现有基线：新模型至少要达到这个量级，最好高出一个数量级。

---

## 2. 本次 CPU 探针：VAE 检索键的 FLOPs【实测（FLOPs 计数）】

方法：diffusers 0.39 的 `AutoencoderKLWan` 按官方 `vae/config.json` 在 meta 设备上实例化（不下权重），`torch.utils.flop_counter.FlopCounterMode` 统计 `encode()`（2 FLOPs/MAC）。配置来源：`Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae/config.json`（Cosmos-Predict2 用的 Wan2.1 VAE，z=16）与 `Wan-AI/Wan2.2-TI2V-5B-Diffusers/vae/config.json`（Fast-WAM / LingBot-VA 用的 Wan2.2 VAE，z=48，16× 压缩）。

| 编码器 | 输入（帧 × 分辨率） | TFLOP | 用途 |
|---|---|---|---|
| Wan2.1 VAE（编码器 53.6M 参数） | 33 × 224² | 8.375 | Cosmos LIBERO 一次完整调用里实际编码的 33 帧（9 个潜帧，多数是占位帧） |
| 同上 | 41 × 224² | 10.385 | Cosmos RoboCasa 完整调用（11 个潜帧） |
| 同上 | 9 × 224² | 2.345 | 只编码"空白 + 腕部 + 主视角"三段 |
| 同上 | 1 × 224² | 0.335 | 单张图单帧编码；两张图 = **0.67 TFLOP**（Cosmos 检索键） |
| Wan2.2 VAE（编码器 149.6M） | 1 × 224×448 | **0.466** | Fast-WAM 检索键（两路相机横拼） |
| 同上 | 1 × 128×256 | **0.152** | LingBot-VA 检索键（两路 128² 横拼） |
| 同上 | 17 × 128×256 | 1.983 | LingBot-VA 一段 16 步动作对应的观测关键帧流式编码量级 |

换算成时间的校准【实测 + 估算】：Cosmos RoboCasa 在 H100 上去掉未来图解码并用 CUDA graph 后，固定开销约 200 ms（`logs/cosmos_nfe_plan.log.md` 22:15–22:35：k=5 470 ms、k=1 295 ms，每次 DiT 前向 64→45 ms；拆出固定 ≈ 172 ms eager / 207 ms graph）。这段固定开销包含 41 帧 VAE 编码（10.4 TFLOP）和 CPU 端 JPEG 压缩、缩放、拼序列，所以 VAE 在 H100 上的等效吞吐 ≥ 50–61 TFLOPS。按此：
- Cosmos 检索键（0.67 TFLOP）≈ **11–13 ms（H100）**；4090 按 20–30 TFLOPS 等效 ≈ **22–34 ms**。
- Fast-WAM 检索键（0.47 TFLOP）≈ 4090 上 **16–23 ms**。
- LingBot-VA 检索键（0.15 TFLOP）≈ 4090 上 **5–8 ms**。
这些都是估算，第一步就要在 GPU 上各测 10 分钟。

---

## 3. 模型逐个核实

### 3.0 汇总：每次命中能省多少 GPU 时间

"命中"只付检索键（策略自己的图像编码器）；检索、修正器在 CPU 上（我们解耦剖析：检索 + 插件约 7.7 ms CPU/请求）。

| 模型 | 完整调用（默认步数） | 完整调用（减步后） | 命中 GPU | 完整:命中 | 每次命中省 | 标记 |
|---|---|---|---|---|---|---|
| GR00T（参照） | 41.4 ms（4090，8 步） | 16.9 ms（1 步） | 6.1 ms | 6.7× / 2.7× | 35 / 11 ms | 实测 |
| π0.5（参照） | 67.5 ms（4090，10 步） | 40.9 ms（1 步） | 10.3 ms | 6.6× / 4.0× | 57 / 31 ms | 实测 |
| Cosmos Policy | 470–576 ms（H100，5 步）；840 ms（4090，5 步，原版含未来图解码）；论文 0.61 s（H100） | 295–306 ms（H100，1 步）；论文 0.16 s | ≈12 ms（H100）/ 22–34 ms（4090） | ≈39–48× / ≈13–25× | ≈0.46–0.56 s / ≈0.15–0.29 s | 完整调用实测+公开；命中估算 |
| LingBot-VA | 6.77 s/段（L40S，20×CFG 视频 + 50 动作步）；2.27 s（H200 级） | ≈1.26 s（L40S，论文部署 3/10 步，按前向次数推） | 键 5–8 ms + KV 补写 ≤74 ms | ≈85× / ≈16× | ≈6.7 s（2.2 s）/ ≈1.2 s | 完整调用公开；其余估算 |
| Fast-WAM（备选） | 110 ms（4090，含文本 + VAE）；190 ms（5090D） | ≈60–80 ms（T5 缓存 + 动作 2 步，估算） | 16–23 ms（4090） | ≈5–7× / ≈3–4× | ≈90 ms / ≈40–60 ms | 完整调用公开；其余估算 |

每 500 局 LIBERO 满步纯推理的 GPU 量级（估算）：Cosmos 我们实测过——空间套件 10 份 H100 副本 20 分钟、长任务 5 份 4090 副本 100 分钟；LingBot-VA 每局 16–22 段 × 2.27–6.77 s ≈ 0.6–2.5 分钟，500 局 ≈ 6–19 卡时（单环境流式、不能合批）；Fast-WAM 每局约 10–16 次 × 0.11 s，500 局十几分钟。

### 3.1 Cosmos Policy（推荐 1）

- 族：视频扩散世界模型做策略（我们已在 LIBERO / RoboCasa-2024 上做过纯减步）。
- 为什么是该族最新的可用成员：NVIDIA 后续的 `nvidia/Cosmos3-Nano-Policy-DROID`、`nvidia/Cosmos3-Edge-Policy-DROID`（HF 2026-09-16 更新，16B，OpenMDW1.1 许可）只有 DROID 真机权重，且带自回归文本塔，没有任何仿真 benchmark 权重（HF 作者 nvidia 下 search=Policy 只有这两个加四个 Cosmos-Policy-*-Predict2-2B）。所以 Cosmos-Policy-Predict2-2B（GitHub/HF 2026-01-22 发布）是该族在仿真上最新的、有权重的成员。
- 结构【公开 + 源码】：`cosmos_policy/experiments/robot/cosmos_utils.py:get_action`——两路 224² 图像先做 JPEG(q95) + 缩放；拼成 33 帧原始序列（空白 1 + 本体占位 4 + 腕部 4 + 主视角 4 + 动作占位 4 + 未来本体 4 + 未来腕部 4 + 未来主视角 4 + 价值 4），Wan2.1 VAE 整段编码成 9 个 28×28 潜帧；本体、动作、价值注入潜帧；2B DiT（Cosmos-Predict2-2B-Video2World）EDM 去噪，默认 5 步。动作段 16 步，整段执行。
  - 我们查到的细节：Cosmos 的"k 步"= k+1 次前向（末尾在 σ_min 上再做一次干净估计）【实测，`logs/cosmos_nfe_plan.log.md` 13:36】。
- 检索键：当前两张图各自单帧过 VAE 的潜特征（16×28×28 每张），空间池化 + PCA。这是策略自己的图像编码器，地位相当于 π0.5 的视觉塔（键不付 DiT，正如 π0.5 的键不付 Gemma 前缀）。
- 开销：

| 来源 | 设置 | 每次调用 |
|---|---|---|
| 论文附录 A.4.2【公开】 | 1×H100，5 / 1 / 10 步 | 0.61 s / 0.16 s / 0.95 s |
| 我们【实测】 | 4090，LIBERO，k=5，原版 eager（含未来图解码） | 840 ms（p50 813），显存 6.5 GB |
| 我们【实测】 | H100，LIBERO，k=5，服务端 | 524 ms |
| 我们【实测】 | H100，RoboCasa，k=5 / k=1，去掉未来图解码 | 576 / 306 ms（eager）；470 / 295 ms（CUDA graph）；动作逐位一致 |

  - 命中路径 GPU【估算】：≈ 12 ms（H100）/ 22–34 ms（4090）。完整:命中（H100）：5 步 ≈ 39–48×，1 步 ≈ 25×；按论文 0.61 / 0.16 s 算是 ≈ 50× / 13×。每次命中省 ≈ 0.46–0.56 s（5 步）/ ≈ 0.15–0.29 s（1 步）。
  - 最坏情况（如果检索键拆不出来、命中要付整段固定开销 172–207 ms）：5 步 2.3–3.4×，1 步 1.4–1.8×。所以"只编码当前两张图"必须先测通，这是 Cosmos 的唯一关键前提。
  - 注意：原版每次都对 33 帧（大半是占位复制帧）做 VAE 编码（8.4 TFLOP），这本身就超过 1 次 DiT 前向（≈7.6 TFLOP，2B×2×1,764 token 再加注意力，估算）。论文里要同时报"原版完整调用"和"只编码一次独特帧的优化版完整调用"，避免被说成占了原版浪费的便宜。优化版 1 步完整调用约 2×37 ms + 12 ms ≈ 90 ms（H100，估算），那时比值只剩 ≈ 7×，仍高于 π0.5 的 4×。
- 适用层：缓存 / 守卫 / 修正器 / 定期旋钮 / 逐任务下降 全部适用（动作空间 7 维，与 LIBERO 的 VLA 相同）。减步：已实测几乎无损——LIBERO 空间 k=5/3/2/1 = 98.6 / 98.4 / 97.4 / 97.4%；长任务 k=5/2 = 98.0 / 97.2%；RoboCasa-2024 k=1 67.25% vs 官方 5 步 67.1%【实测，`logs/cosmos_nfe_plan.log.md`】。延长执行段：动作段 16 步，默认已整段执行，没有再延长的余地（缓存回放可用到 16 步）。
- 权重与文件清单【HF API 核实】：
  - `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`（gated=False，lastModified 2026-02-04）：`Cosmos-Policy-LIBERO-Predict2-2B.pt` 3,913,017,345 B；`config.json` 1,354；`libero_dataset_statistics.json` 2,372；`libero_t5_embeddings.pkl` 41,957,938；`README.md` 11,233。config.json：输入 224² agentview + eye_in_hand、9 维本体，动作 7×16，`denoising_steps: 5`，训练 `40000` 步 / batch `1920` / `64x H100`，LIBERO 四套件 98.1 / 100 / 98.2 / 97.6。
  - 依赖底座 `nvidia/Cosmos-Predict2-2B-Video2World`（gated=auto，NVIDIA Open Model License）：`tokenizer/tokenizer.pth` 507,609,880 B；`model-480p-16fps.pt` 3,913,017,214 B。本机 `/data/cosmos/hf_home/hub/` 下这两个文件还在（owner 9-16 已在 HF 点过同意）。
  - **本机的 Cosmos LIBERO 权重已于 9-16 按 owner 指令删除**（`logs/cosmos_nfe_plan.log.md` §5），要重下约 4 GB；RoboCasa 权重还在 `/data/cosmos/hf_home/models--nvidia--Cosmos-Policy-RoboCasa-Predict2-2B`。本机 cosmos-policy 代码目录 `/home/weiland/cosmos-policy` 也已不在，`exp/cosmos_nfe/` 的服务与分片脚本还在。
  - MetaWorld / CALVIN：无官方权重。唯一的第三方 `mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B`（`consolidated.pt` 3,913,050,026 B）README 自述只针对 `assembly-v3` 单任务、基础策略约 52%（25 局），不能用作 MT50。
- 许可：NVIDIA One-Way Noncommercial License（NSCLv1），研究可用、不可商用。
- 复现风险：低（我们已复现：空间 98.6% vs 论文 98.1%，长任务 98.0% vs 97.6%）。

### 3.2 LingBot-VA（推荐 2）

- 族：基于 Wan2.2 的世界-动作模型（我们已做过 X-WAM 的纯减步）。
- 在该族里的位置（按发布时间）：X-WAM（arXiv 2604.26694，权重 2026-05-28）> Fast-WAM（2603.16666）> DreamZero（2602.15922）> LingBot-VA（2601.21998）。有 LIBERO 官方权重的只有 Fast-WAM 和 LingBot-VA。**按"最新"应选 Fast-WAM；按"有利可图"和"与前两类结构不同"应选 LingBot-VA**。我选 LingBot-VA，Fast-WAM 作备选（§3.3）。
- 结构【公开 + LeRobot 源码】：两路相机 128²（`image_hflip`）→ Wan2.2 VAE（流式、带因果特征缓存）→ 宽度拼接；30 层 3072 维共享骨干（发布的是 "shared backbone" 版：safetensors 头里 `transformer.blocks` 4.910B，视频流和动作流用同一组权重）；每段先去噪 4 个未来视频潜帧（默认 20 步，CFG=5，即每步 2 次前向），再去噪 16 步动作（50 步）；真实观测关键帧和已执行动作写回 KV 缓存（`attn_window=30`）。论文部署配置：视频 3 步（积到 s≈0.6）+ 动作 10 步。
- 检索键：当前关键帧的 VAE 潜特征（48×8×16），池化 + PCA；为避免流式 VAE 的跨帧状态，键用单帧无状态编码（0.15 TFLOP）。
- **跨调用状态（必须处理）**：`modeling_lingbot_va.py:predict_action_chunk` 每段先调 `_compute_kv_cache(self._obs_buffer, self._executed_actions)` 把上一段的观测和执行动作写回 KV，再去噪。缓存命中时我们跳过去噪，但下一次真调用前必须把这段时间的关键帧和实际执行的（缓存回放的）动作补写进 KV——一次前向，可把多段合并成一次。检索只看当前帧、不看历史；它本身不是马尔可夫策略，这是效果上的风险，要用实验回答。
- 开销：

| 来源 | 设置 | 每段（16 步动作） |
|---|---|---|
| Flash-WAM 论文（arXiv 2606.05254）【公开】 | L40S，LIBERO，教师 20 视频步 / 50 动作步 | 6,767 ms |
| 同上 | L40S，RoboTwin，25 / 50 步 | 8.1 s |
| 同上（蒸馏后，不适用于发布权重，只用来给"非去噪部分"定上界） | L40S，LIBERO，1 / 1 步 | 404 ms |
| GeoBoN 论文（arXiv 2607.17454）Table 2【公开】 | LIBERO-Long 基线，H200 / RTX Pro 6000 节点 | 2.27 s |
| LeRobot 文档【公开】 | 显存 | 18–24 GB（UMT5 默认放 CPU） |

  - 每次前向平均 ≈ 6,767 / 91（40 次视频 + 50 次动作 + 1 次 KV 写回）≈ 74 ms（L40S）【估算】。
  - 命中路径 GPU【估算】：键 5–8 ms + KV 补写 ≤ 74 ms（L40S，每段一次；可合并）≈ ≤ 80 ms。
  - 完整:命中：默认配置 ≈ 85×（L40S）/ ≈ 80×（H200 级，按同比例）；论文部署 3/10 步（≈ 17 次前向 ≈ 1.26 s，L40S，估算）≈ 16×。每次命中省 ≈ 6.7 s / 2.2 s / 1.2 s。
- 适用层：缓存 / 守卫 / 修正器 / 旋钮 / 逐任务下降 全部适用（需加 KV 补写）。减步：空间最大（两套步数：视频 20×CFG、动作 50）；但**未蒸馏直接减步**只在真机上测过：教师 3/10 步 66.7% → 1/2 步 40.0% → 1/1 步 23.3%（Flash-WAM Table 3）；LIBERO / RoboTwin 上作者只报了蒸馏后的结果（1/2 步 96.1%、1/1 步 95.1% vs 教师 98.6%）。仿真里未蒸馏减步掉不掉，没人测过——正好是我们减步层要回答的。延长执行段：一段 16 步，默认整段执行。
- 权重与文件清单【HF API + safetensors 头核实】：
  - `robbyant/lingbot-va-posttrain-libero-long`（Apache-2.0，2026-04-24）：`transformer/diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors` 4,821,987,820 / 4,821,655,760 / 535,373,816 B；`vae/diffusion_pytorch_model.safetensors` 2,818,777,808；`text_encoder/model-0000{1,2,3}-of-00003.safetensors` 4,935,812,536 / 4,983,103,192 / 1,442,935,480；tokenizer 若干；共 19 个文件 24.38 GB。
  - LeRobot 转换版 `lerobot/lingbot_va_libero_long`（Apache-2.0，2026-10-06）：`model.safetensors` 10,177,841,732 B（头部核实 5.089B 参数，bf16）、`config.json` 2,274（`num_inference_steps 20`、`action_num_inference_steps 50`、`guidance_scale 5.0`、`action_per_frame 4`、`frame_chunk_size 4`、`attn_window 30`、128×128）；VAE + UMT5 运行时从 robbyant 仓库拉。模型卡写明 "No evaluation results have been provided"。
  - 另有 `robbyant/lingbot-va-base`、`robbyant/lingbot-va-posttrain-robotwin`、`lerobot/lingbot_va_robotwin`、`lerobot/lingbot_va_base`。
  - **LIBERO 空间套件没有官方权重**（只有第三方 `Wjjjh/lingbot-va-libero-spatial` 等，0 下载、无说明，不用）。论文 Table（LIBERO 四套件 98.5 / 99.6 / 97.2 / 98.5）称"在四个套件上训练"，发布的 libero-long 权重能否直接跑空间套件——未核实，50 局即可试出。
  - MetaWorld / CALVIN：无。
- 许可：Apache-2.0（HF 卡 + GitHub）。
- 复现风险：中。① LeRobot 版未公布复现分数；② LeRobot 实现只支持 `--eval.batch_size=1` 的单环境流式推理，服务端不能跨局合批，4090 放 2 份、H100 放 3 份；③ 评测贵：长任务每局约 16–22 段 × 2.27–6.77 s ≈ 0.6–2.5 分钟 GPU，满步纯推理 500 局 ≈ 6–19 卡时（估算）；④ 要实现 KV 补写。

### 3.3 Fast-WAM（备选，替 LingBot-VA）

- 结构【公开 + LeRobot 源码 `wan/modular.py`】：两路 224² 横拼成 224×448 → Wan2.2 VAE 编一帧 → 5.0B 视频骨干只前向一次（`prefill_video_cache`，缓存每层 K/V，**测试时不生成视频**）→ 1.0B 动作专家 10 步流匹配去噪（`num_inference_steps=10`，CFG=1）→ 32 步动作。无跨调用状态（`reset()` 只清动作队列）。safetensors 头核实：`model.mot.mixtures.video` 5.000B + `model.mot.mixtures.action` 1.021B。
- 开销【公开】：GitHub README "End-to-end FastWAM inference is now approximately 2x faster, including text encoding and VAE encoding"：RTX 4090 190 → **110 ms**，H20 470 → 210 ms；论文：RTX 5090D 190 ms（联合生成视频的 IDM 版 810 ms）。
- 命中【估算】：键 16–23 ms（4090）→ 完整:命中 ≈ 5–7×，每次命中省 ≈ 87–94 ms；若 T5 按任务缓存、动作减到 2 步，完整调用约 60–80 ms，比值 ≈ 3–4×，省 ≈ 40–60 ms——**与 π0.5 同一量级**。它测试时就是"大骨干前向一次 + 动作专家去噪"，算力结构和 VLA 一样，这是我没把它当首选的原因。
- 权重【HF API 核实】：`yuanty/fastwam`（作者官方，2026-03-23 建，无 license 字段；GitHub 代码 MIT）：`libero_uncond_2cam224.pt` 12,041,735,140 B、`libero_optional_idm_2cam224.pt` 12,041,735,545、`robotwin_uncond_3cam_384.pt` 12,041,813,092 及各自 stats JSON。LeRobot 版 `lerobot/fastwam_libero_uncond_2cam224`：`model.safetensors` 12,041,631,582 B（6.021B 参数）、`libero_uncond_2cam224_dataset_stats.json` 40,939；**README 列出的 `Wan2.2_VAE.safetensors`、`models_t5_umt5-xxl-enc-bf16.safetensors` 不在仓库里**，代码实际从 `Wan-AI/Wan2.2-TI2V-5B-Diffusers`（Apache-2.0）拉 VAE（2,818,777,808 B）和 UMT5（三片共 11.36 GB）。
- LIBERO：一个权重覆盖四个套件——论文 98.2 / 100 / 97.0 / 95.2（均 97.6）；GitHub 表（first-frame 模式）98.2 / 97.8 / 99.2 / 95.8（均 97.75）。**空间和长任务都有**。MetaWorld / CALVIN：无。
- 复现风险：低到中（无状态、可合批、LeRobot 原生；未公开 LeRobot 版独立复现）。

### 3.4 X-WAM（不选）
- `sharinka0715/X-WAM-checkpoints`（Apache-2.0）：`pretrained/`、`robocasa_sft/`、`robotwin_sft/` 各一个 38.9 GB 的 `mp_rank_00_model_states.pt`。没有 LIBERO / MetaWorld / CALVIN 权重。我们已在 RoboCasa-2024（k=10/5/3/2/1 = 79.2 / 78.1 / 78.6 / 76.8 / 79.3%）和 RoboTwin（k=10/1 = 90.8 / 86.4%）做过纯减步；H100 空载 0.39 s/次（动作 10 步）【实测，`logs/xwam_dp_nfe_plan.log.md`】。

### 3.5 DreamZero（不选）
- 官方 `GEAR-Dreams/DreamZero-DROID`（64.79 GB，CC-BY-NC-4.0，14B）、`DreamZero-AgiBot`：只有真机平台。LIBERO 只有 RLinf 的第三方 `RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000`（25.25 GB，无 README、无成功率），不可核实。14B 版要 2 张卡做 CFG 并行（repo 说明）。

### 3.6 Cosmos3 策略（不选）
- `nvidia/Cosmos3-Nano-Policy-DROID`：16B，MoT（自回归文本塔 + 扩散塔），只给 DROID；无 LIBERO / MetaWorld 权重。DROID 仿真评测（PolaRiS 等）基于 Isaac，GPU 渲染，被规则排除。

### 3.7 GigaWorld-Policy-0.5（不选）
- `open-gigaai/Giga-World-Policy-0.5`（Apache-2.0，24.09 GB，只含 transformer，`num_embodiments: 2`）；GitHub 只有 AgileX 微调配置，无仿真 benchmark 权重。

### 3.8 VPP（Video Prediction Policy，CALVIN 备选时才用）
- 权重【HF API】：`yjguo/svd-robot-calvin-ft`（`unet/diffusion_pytorch_model.safetensors` 6,098,682,464 B、`image_encoder/model.safetensors` 2,528,373,448、`vae/...` 391,017,740）+ `yjguo/dp-calvin`（`last.pt` 1,020,398,302）；GitHub MIT。论文 MetaWorld 平均 0.682，但**没有发布 MetaWorld 权重**；没有 LIBERO 权重。
- 开销【公开】：每个观测视频模型只前向一次，"less than 160 ms"，表 4 约 140 ms，RTX 4090 7–10 Hz；动作头 DiT 10 步、动作段 10。键用 SVD 的 VAE 潜特征时命中约 5–15 ms【估算】，比值 ≈ 10×，省 ≈ 125 ms。减步余地小（视频模型本来就 1 步）。
- CALVIN ABC→D 4.33（1000 条链，Awesome-WAM 排行榜收录）。2024-12 的工作，比 Cosmos Policy 老，属于"视频扩散做策略"一类。

### 3.9 JEPA-WM / DINO-WM / V-JEPA 2-AC（不选）
- `facebook/jepa-wms`（CC-BY-NC-4.0）：`jepa_wm_metaworld.pth.tar` 211,687,551 B、`dino_wm_metaworld.pth.tar` 280,902,699 B 等。论文（arXiv 2512.24497）Table 2：MetaWorld 只评 Reach 与 Reach-Wall，JEPA-WM 58.2 / 41.6%，DINO-WM 44.8 / 35.1%；目标图像条件的 CEM 规划，不是 MT50 语言任务；没有 LIBERO 权重。

### 3.10 其余（都不选）
| 候选 | 理由（证据） |
|---|---|
| MoDE（`mbreuss/MoDE_LIBERO_10`、`MoDE_CALVIN_ABC_1` 等，MIT） | 便宜且视觉占一半：论文附录 A.4 每次 1.53 GFLOPs，A6000 12.2 ms；FiLM-ResNet-50 编码器 8.27 GFLOPs ≥ 10 步去噪 ≈ 7 GFLOPs，命中最多省一半 |
| Diffusion Policy（CNN） | 2023；ε 头减步悬崖（我们实测 DDIM-10/4/2/1 均值 0.88 / 0.81 / 0.60 / 0.03）；DDIM-10 在 4090 上 144 ms 主要是 launch-bound（13 ms/步），CUDA graph 后会很便宜；同族更新的 DP3 无官方权重，MoDE 见上 |
| Seer / GR-1 / MDT | 单次前向、0.3B 级，视觉编码占比大，命中省不下多少 |
| 第三方 `brandonyang/dp-metaworld-checkpoints` | ML45 语言条件 DP，`model.safetensors` 367 MB，来源与评测不明 |
| `MINT-SJTU/Evo-World-Metaworld` | config.json：`vlm_name: OpenGVLab/InternVL3-1B` + 流匹配动作头 = VLA |
| AVDC / UniPi 类 | 2023–24；MetaWorld 只有 11 个任务；权重下落未核实 |
| TD-MPC2 / Newt、LeWM | 状态输入或玩具环境；"只用状态的键直接否" |
| ACT | owner 已否（CPU 可算） |
| 所有 VLA（π0、OpenVLA、X-VLA、SmolVLA、GR00T N1.6/1.7、DM0.5、VLA-JEPA、MolmoAct2…） | 规则排除 |

---

## 4. Benchmark

### 4.1 MetaWorld MT50（推荐）

- π0.5 权重【HF API】：
  - `RLinf/RLinf-Pi05-MetaWorld-SFT`（2025-11-20，无 license 字段）：`model.safetensors` 7,473,091,464 B、`metadata.pt` 3,172、`lerobot/metaworld_mt50/norm_stats.json` 1,134。本机 `/data/ckpt/pi05_metaworld_rlinf/model.safetensors` 同为 7,473,091,464 B，已严格加载、已接入。
  - RL 版 `RLinf/RLinf-Pi05-MetaWorld-RL-FlowNoise`（`model.safetensors` 8,531,807,580 B）、`...-RL-FlowSDE`（8,531,240,580 B）。RLinf 文档：π0.5 SFT 43.8%、π0.5 + PPO 70.7%（四难度组平均）。
- 我们的结果【实测，`logs/session_handoff.md` 中 "MetaWorld π0.5（50 任务 × 20 集）" 一行、`logs/metaworld_warm_reset.log.md`】：50 任务 × 20 局，满步 58.8%、2 步 60.4%、1 步 53.7%（RLinf 口径四组平均 51.4，RLinf 自报 43.8）；6,000 局 9-25 19:4x 开跑、9-26 00:08 完（约 4.4 小时，服务在 4090、worker 在 timan108）。仿真约定已固化（`exp/metaworld/env.py`）：`corner2` 相机 480²、每次推理执行 5 步、≤160 策略步、15 步零动作沉降。
- 适合缓存【实测（读环境源码）】：每个任务一个固定场景（Sawyer 臂 + 固定桌面与物体），没有布局、纹理、物体实例、光照变化；随机的只有物体 / 目标初始位置。从本机 metaworld 3.0.0（v3 环境）的 `envs/sawyer_*_v3.py` 解析：50 个任务里物体初始 xy 随机范围中位 20 cm、最大 20 cm（6 个任务固定不动），目标范围中位 20 cm、最大 30 cm（33 个任务可解析）。每个 (任务, seed) 有 50 个固定变体，评测用 idx 0..19，库可以用 idx 20..49 或换 seed。对比：RoboCasa365 / RoboCasa-2024 每局换厨房布局、风格和物体实例。
- 每局 CPU 开销【实测，本次探针】：物理每步 0.77–2.70 ms（reach / pick-place / door-open / assembly）；建环境 + 沉降 0.9–1.5 s（MT1 构造约 1.2 s，我们代码里按任务缓存）；480×480 渲染在纯 CPU（Mesa llvmpipe，单线程）上 276–333 ms/帧，一局 160 步 32 次渲染 ≈ 9–11 s。生产上 worker 用 GPU 节点的 EGL 渲染（毫秒级，本次不允许测 GPU），所以一局仿真 < 1 s 量级，瓶颈在模型服务。
- 新与影响力【公开】：Meta-World（CoRL 2019）；Meta-World+（arXiv 2505.11289，NeurIPS 2025 Datasets & Benchmarks）重新规范版本与奖励。LeRobot 有 `lerobot/metaworld_mt50` 数据集和 `lerobot/smolvla_metaworld`；RLinf 有 π0 / π0.5 / OpenVLA-OFT 的 SFT 与 RL 权重；VPP（ICML 2025）、OneWM-VLA（arXiv 2605.07931）等也报 MT50。**实话**：OpenMOSS 的 Awesome-WAM 排行榜（`leaderboard/data/leaderboard.json`，2,229 条）收录 18 个 benchmark，LIBERO 748 条、CALVIN 373 条，**没有 MetaWorld**——在 2025–26 的 VLA / 世界模型论文里，MetaWorld 不如 CALVIN 常用。
- 本 benchmark 上有权重的模型：π0.5（SFT + RL）、π0（RLinf）、SmolVLA（LeRobot）、GR00T N1.7（第三方 `namenu/gr00t17-lerobot-metaworld_mt50`，未核实）。**推荐的非 VLA 模型都没有。**

### 4.2 CALVIN ABC→D（备选）

- π0.5 权重【HF API】：`RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT`（Apache-2.0）：`model.safetensors` 7,473,091,464 B、`InternRobotics/InternData-Calvin_ABC/norm_stats.json` 1,755；RL 版 `...-RL-FlowNoise` 8,531,807,580 B。他人用该权重测得 ABC→D 平均链长 3.83（TempoFit 引用 RLinf 权重），其他复现 4.0–4.3（Awesome-WAM 排行榜）。
- 有官方权重的非 VLA：VPP 4.33（贵，§3.8）、MoDE 4.01 / 3.39（便宜）、Seer-Large 4.28、GR-1 3.06（便宜）。Cosmos / LingBot-VA / Fast-WAM 均无。
- 影响力：排行榜 373 条，仅次于 LIBERO；视频 / 世界模型策略（VPP、GR-1、Seer、VidMan、SuSIE、MoWM、S-VAM…）大量使用。新：2022 年，之后没有官方修订。
- 适合缓存：评测只在环境 D（训练没见过）；场景固定，方块位置和抽屉 / 滑门 / 灯的状态随机；1000 条 5 子任务链，子任务起点取决于上一个子任务的结局，起点分布比 MetaWorld 散。中等。
- 轻：PyBullet，可 EGL / 软件渲染。**每局 CPU 开销未核实**（没有装环境；官方评测每子任务上限约 360 步，1000 条链上限约 180 万步，这一步数来自检索结果、未读源码确认）。
- 需要从零接入（环境、链式评测、π0.5 适配、VPP 适配）。

### 4.3 不选的 benchmark
| benchmark | 理由 |
|---|---|
| RoboCasa365 / RoboCasa-2024 | 布局、风格、物体实例都变（owner 已否；两个版本一样） |
| RoboTwin 2.0 | 重：光追渲染 + CuRobo 规划；我们 9-17 实测 4090 上最多 8 个 client（每个 5–7 GB 显存、单核 CPU 绑定），约 160–185 局/小时，OIDN 显存不足会静默喂未去噪帧（`logs/xwam_dp_nfe_plan.log.md` §6–§7） |
| BEHAVIOR | 重（Isaac / OmniGibson） |
| ManiSkill3、SimplerEnv、DROID 仿真（PolaRiS） | GPU 渲染 |
| LIBERO-Plus / LIBERO-Pro | 按设计加扰动（相机、光照、背景、布局…），与"适合缓存"相反；且是 LIBERO 衍生 |
| VLABench | 场景重、变化大 |

---

## 5. 没能核实的（原样列出）

1. 所有"命中路径"的 GPU 时间都是 FLOPs 推算（§2），没有在 GPU 上测。Cosmos 的关键前提"只编码当前两张图做键"尤其要先测。
2. Cosmos 论文 1 步 0.16 s 是 1 次还是 2 次前向，论文没写；与我们 H100 实测 1 步 295–306 ms 差距的来源（JPEG、相机数、占位帧数）没拆。
3. LingBot-VA：LeRobot 版能否复现 LIBERO-Long 98.5%；libero-long 权重能否跑空间套件；论文部署配置 3/10 步的绝对耗时（1.26 s 是按前向次数比例推的）；仿真里未蒸馏减步掉不掉；KV 补写后下一次真调用的输出是否与"一直真调用"一致。
4. Fast-WAM 110 ms（4090）里文本编码、VAE、视频前缀、动作去噪各占多少，官方没给拆分。
5. MetaWorld 在 GPU-EGL 下的单帧渲染耗时（本次只能测 CPU 软件渲染）。
6. CALVIN 每局 CPU 开销；CALVIN 官方评测是否需要下载大体积验证集。
7. GigaWorld-Policy-0.5 的两个 embodiment 具体是什么（只看到 AgileX 配置）。
8. HF 上的搜索是按仓库名关键词做的（metaworld / meta-world / mt50 / ml45 / calvin / 各模型名），可能漏掉名字里不含这些词的权重。

---

## 6. 复现本次检查用的命令（只读）

- 文件清单：`curl -s "https://huggingface.co/api/models/<repo>?blobs=true"`。
- safetensors 参数量：`curl -r 0-7` 取头长 N，`curl -r 8-$((8+N-1))` 取头部 JSON，按键前缀累加 `prod(shape)`。
- VAE FLOPs：lerobot_venv（diffusers 0.39、torch 2.7.1）里 meta 设备实例化 `AutoencoderKLWan(**config)`，`FlopCounterMode` 包 `encode()`。
- MetaWorld CPU 探针：`MUJOCO_GL=egl EGL_PLATFORM=surfaceless __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json`，`exp.metaworld.env.make_env / reset_and_settle / render_image`，随机动作 160 步、每 5 步渲染一次。
- 排行榜统计：`OpenMOSS/Awesome-WAM` 仓库 `leaderboard/data/{benchmarks,leaderboard}.json`。

## 7. 主要来源
- Cosmos Policy：https://arxiv.org/abs/2601.16163（附录 A.4.2 延迟）；https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B ；https://github.com/nvlabs/cosmos-policy （`cosmos_utils.py:get_action`）
- LingBot-VA：https://arxiv.org/abs/2601.21998 ；https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long ；https://huggingface.co/lerobot/lingbot_va_libero_long ；https://huggingface.co/docs/lerobot/main/en/lingbot_va ；LeRobot `src/lerobot/policies/lingbot_va/modeling_lingbot_va.py`
- Flash-WAM（LingBot-VA 延迟与减步）：https://arxiv.org/html/2606.05254 ；GeoBoN：https://arxiv.org/html/2607.17454
- Fast-WAM：https://arxiv.org/abs/2603.16666 ；https://github.com/yuantianyuan01/FastWAM ；https://huggingface.co/yuanty/fastwam ；https://huggingface.co/lerobot/fastwam_libero_uncond_2cam224
- X-WAM：https://huggingface.co/sharinka0715/X-WAM-checkpoints ；DreamZero：https://huggingface.co/GEAR-Dreams/DreamZero-DROID ；Cosmos3：https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID ；GigaWorld：https://github.com/open-gigaai/giga-world-policy
- VPP：https://arxiv.org/abs/2412.14803 ；https://github.com/roboterax/video-prediction-policy
- JEPA-WM：https://arxiv.org/abs/2512.24497 ；https://huggingface.co/facebook/jepa-wms
- MoDE：https://arxiv.org/abs/2412.12953
- MetaWorld：https://arxiv.org/abs/2505.11289 （Meta-World+）；RLinf MetaWorld：https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/metaworld.html
- 排行榜：https://github.com/OpenMOSS/Awesome-WAM （`leaderboard/data/`）
- 我们自己的：`logs/cosmos_nfe_plan.log.md`、`logs/xwam_dp_nfe_plan.log.md`、`logs/nfe_baseline_plan.log.md`、`logs/nfe_baseline_rc365_plan.log.md`、`logs/metaworld_warm_reset.log.md`、`logs/step_diag_metaworld_selfstart_plan.log.md`、`logs/session_handoff.md`、`exp/offline_search/rounds/r15/REPORT.md`、`exp/offline_search/rounds/r16/REPORT.md`
