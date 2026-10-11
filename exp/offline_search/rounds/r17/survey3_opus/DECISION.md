# R17 第三个 MetaWorld 微调模型：结论（survey3_opus）

## 1. 结论

| 角色 | 模型 | LIBERO 权重（已按文件清单核实） | MetaWorld 微调路线 | MT50 微调成本（5 个 epoch ≈ 1.02M 样本） |
|---|---|---|---|---|
| **首选** | **Fast-WAM**（6B：Wan2.2-TI2V-5B 视频专家 + 约 1B 流匹配动作专家，MoT 结构；推理时不想象未来帧） | [`yuanty/fastwam`](https://huggingface.co/yuanty/fastwam) `libero_uncond_2cam224.pt` 12.04 GB；LeRobot 格式副本 [`ZibinDong/fastwam_libero_uncond_2cam224`](https://huggingface.co/ZibinDong/fastwam_libero_uncond_2cam224) | LeRobot `policy.type=fastwam`（原生读取 `lerobot/metaworld_mt50` 的 v3.0 格式，`env.type=metaworld` 直接评测），加 `--policy.freeze_video_expert=true --policy.loss.lambda_video=0` 只训动作专家；动作 / 状态维度不一致的张量自动重新初始化 | 按 Fast-WAM 原代码效率估：H100 约 13 h；4090 约 50 h（太慢）。4090 上更现实的做法：把 epoch 减到约 2.5 个（约 25 h），或改走备选的 LoRA 路线。LeRobot 移植版的实际吞吐没有实测数字 |
| **备选** | **EasyWAM-MoT-Wan22**（与 Fast-WAM 架构相同，换了一套代码；官方给了 LoRA 配方） | [`OpenMOSS-Team/EasyWAM-MoT-Wan22`](https://huggingface.co/OpenMOSS-Team/EasyWAM-MoT-Wan22) `libero_mot_wan22.pt` 12.04 GB + `libero_mot_wan22_lora_128.pt` 2.69 GB | 视频专家加 LoRA（rank 128）+ 动作专家全参；数据走 LeRobot v3。⚠ 文档里没有"从 LIBERO ckpt 换动作维度继续微调"的入口，只有 `resume`，需要补一小段代码来重新初始化动作头 | 按 EasyWAM 效率估：H100 约 6.5 h；4090 约 26 h |
| 为什么备选与首选同族 | 除 Fast-WAM 一系外，没有候选同时满足要求 1–5（见第 2 节）；备选用来对冲微调代码和单卡成本的风险 | — | — | — |

## 2. 候选逐项（✅ 满足 / ❌ 不满足 / ⚠ 有保留）

| 模型 | ① LIBERO 权重 + 报告成功率 | ② 方便在 MT50 上微调 | ③ 非 VLA、与 GR00T / DP 不同族 | ④ 无状态 | ⑤ 缓存划算 | 结论 |
|---|---|---|---|---|---|---|
| **Fast-WAM** | ✅ 四个套件联合训练的单个 ckpt；论文 97.6（Spatial 98.2 / Long 95.2）；LeRobot 复现 96.4 | ✅ 官方代码 + LeRobot 集成；LeRobot 支持冻结视频专家和 PEFT；单卡可训 | ✅ 视频扩散 WAM，umT5 只做文本编码 | ✅ 输入只有当前一帧 + 文本 + proprio；视频 KV 每次调用重新计算；动作集成默认关闭 | ✅ 完整调用 110 ms（4090，README 实测）；检索键用 VAE 潜变量约 6–12 ms（估）；完整 : 命中 ≈ 8–15× | **首选** |
| **EasyWAM-MoT-Wan22** | ✅ 全参 96.7 / LoRA 95.9 | ✅ 官方 LoRA r128 配方；训练吞吐是 Fast-WAM 原代码的 2.68 倍 | ✅ | ✅ 只用当前帧（`infer_action_batch`）；cross-KV 复用不跨重规划 | ✅ 同 Fast-WAM | **备选** |
| EasyWAM-MoT-Cosmos25（2.8B） | ✅ 5.55 GB，97.6 | ✅ 更便宜 | ⚠ 文本编码器是 Cosmos-Reason1-7B（VLM，只离线生成文本嵌入）；骨干与 Cosmos Policy 同系 | ✅ | ⚠ 骨干只有 2B，每次命中省得少 | 不选 |
| Faster-WAM（hustvl） | ✅ `libero/step_021700.pt` 11.1 GB | ✅ 在 Fast-WAM 代码基上改的（全参） | ✅ | ⚠ 每次调用内复用未来表示；未逐行核 | ⚠ 目标就是降延迟 | 不选（Fast-WAM 衍生，与首选重复） |
| RealtimeWAM（lightx2v） | ✅ `realtimewam_libero_fast.pt` | ⚠ | ✅ | ⚠ 未核 | ❌ 蒸馏版，H100 上 12.2 ms，完整调用太便宜 | ❌ |
| Streaming-WAM / FastWAM-Joint-CD（SJTU） | ✅ `ac-stream/`、`joint-cd/` 各 12.04 GB | ⚠ | ✅ | ⚠ stream 版是流式的；未核 | — | 不选 |
| LingBot-VA | ⚠ 只有 LIBERO-Long（`lingbot-va-posttrain-libero-long`） | ✅ | ✅ | ❌ 跨调用 KV cache（client `compute_kv_cache=True`，只在 reset 时清空） | — | ❌ |
| UVA | ⚠ 只有 LIBERO-10（Google Drive `libero10.ckpt`），0.90 / 500 局 | ⚠ README 建议 ≥ 4 张 GPU | ✅ | ❌ `n_obs_steps: 16` 历史帧 | — | ❌ |
| UWM | ❌ 只有 LIBERO-90 预训练权重，目标任务要自己在 LIBERO 上微调 | ✅ | ✅ | ⚠ 未核 | — | ❌ |
| VPP | ❌ 只有 CALVIN 权重（svd-robot-calvin / dp-calvin） | ✅ 有 MetaWorld 代码 | ✅ | — | — | ❌ |
| PAD | ❌ 没找到代码或权重（论文只做了 MetaWorld） | ❌ | ✅ | — | — | ❌ |
| Seer | ❌ 只有 CALVIN / 真机 ckpt（LIBERO-LONG 只放了代码） | ✅ | ✅ | ❌ 历史帧 | — | ❌ |
| MoDE | ✅ `mbreuss/MoDE_LIBERO_10` 3.3 GB | ✅ | ❌ 扩散 Transformer 策略，属 DP 族 | — | ❌ 小模型 | ❌ |
| MDT | 未核 | — | ❌ DP 族 | — | ❌ | ❌ |
| Octo | ❌ 无官方 LIBERO 权重 | — | ❌ 扩散头 | ❌ 2 帧窗口 | — | ❌ |
| GE-Act（Genie Envisioner） | ❌ 只有 AgiBot 的 GE-base / GE-Sim | — | ✅ | ❌ 稀疏记忆历史帧 | — | ❌ |
| Motus | ❌ 只有 RoboTwin 权重 | — | ❌ 含 Qwen3-VL-2B | — | — | ❌ |
| FLOWER | ✅ `mbreuss/flower_libero_*` | — | ❌ VLA（Florence-2） | — | — | ❌ |
| X-WAM | ❌ 只有 RoboCasa / RoboTwin | — | ✅ | — | — | ❌ |
| mimic-video | ⚠ 只有第三方的 libero-object ckpt | — | ✅ | — | — | ❌ |
| Cosmos Policy | owner 已定为 LIBERO 特例 | — | — | — | — | 不在范围内 |

## 3. Fast-WAM 在 MT50 上的微调成本

设定：单相机 480² 缩放到 224²，每个样本 147 个视频 token（9 帧 → 3 个潜帧 × 49）+ 32 个动作 token；5 个 epoch = 1.02M 样本（约 8k step × 全局 batch 128）。成本按 epoch 线性缩放。

| 模式 | 可训参数 | 哪套代码支持 | 单卡显存（估） | TFLOP / 样本 | H100 GPU·h（EasyWAM 效率 / Fast-WAM 原代码效率） | 4090 GPU·h |
|---|---|---|---|---|---|---|
| 全参 | 6.0B | Fast-WAM 官方（默认 8 卡 ZeRO-1）、LeRobot、EasyWAM | 约 96 GB 以上 → 单张 H100 / 4090 都放不下（需 ZeRO-2 offload 或 ≥ 2 张 H100） | 6.7 | 8.3 / 22 | 不可行 |
| LoRA r128（视频）+ 动作专家全参 | 1.34B | EasyWAM 官方配方；LeRobot 有通用的 `use_peft` 入口（目标模块要自己指定，未实测） | 约 35 GB + 激活 → H100 可以；4090 要 batch 4–8 并开梯度检查点 | 5.2 | 6.5 / 17 | 约 26（EasyWAM 效率） |
| 冻结视频专家，只训动作专家 | 约 1.0B | LeRobot `freeze_video_expert`；官方代码要改约 3 行 | 约 30 GB + 激活 → 两种卡都可以 | 3.8 | 4.7 / 12.7 | 约 19 / 51 |

| 吞吐锚点（实测，来自已发布数字） | 数值 |
|---|---|
| EasyWAM-MoT（即 Fast-WAM 架构），LIBERO 双相机，8×H100、每卡 batch 16 | 138.2 样本/s（每卡 17.3）；20k step 约 5 h = 40 H100·h |
| Fast-WAM 原代码，同条件 | 51.5 样本/s（每卡 6.4）；20k step 约 14 h = 112 H100·h |
| 推出的有效算力（LIBERO 每样本 13.2 TFLOP） | EasyWAM 约 228 TFLOPS/卡；原代码约 85 TFLOPS/卡；4090 按 H100 的 1/4 计 |

## 4. 每次调用成本（要求 ⑤）

| 部分 | Fast-WAM LIBERO（双相机 224×448） | Fast-WAM MT50（单相机 224²） | π0.5（参照） |
|---|---|---|---|
| 检索键：VAE 编码当前帧 | 0.47 TFLOP（CPU meta 设备实数），约 6–12 ms（估） | 0.23 TFLOP，约 3–6 ms（估） | SigLIP，约 10 ms |
| 视频专家预填（5B，98 / 49 个 token + 128 个文本 token） | 1.01 TFLOP；batch 1 时受带宽限制，读 10 GB 权重，≥ 10 ms | 0.58 TFLOP，≥ 10 ms | — |
| 动作专家 10 步（1B × 32 token） | 0.64 TFLOP；每步读 2 GB，≥ 20 ms | 同左 | — |
| umT5-XXL 文本编码 | 每个任务缓存一次 | 每个任务缓存一次 | — |
| 完整调用 | 110 ms（4090，torch.compile，含 T5 + VAE；README 实测）；RTX 5090D 上 190 ms（论文） | 约 80–100 ms（估） | 68 ms（4090） |
| 完整 : 命中 | 约 8–15×（按 FLOPs 的下界是 4.5×） | 约 10× 以上（估） | 6.6× |
| 每次命中省下 | 约 85–100 ms | 约 75–90 ms | 约 58 ms |
