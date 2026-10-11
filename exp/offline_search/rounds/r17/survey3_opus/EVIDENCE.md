# R17 survey3_opus — 证据

查询时间 2026-10-08（HF API 文件清单、GitHub API、LeRobot main 源码）。所有探针只用 CPU：`taskset -c 26-33,70-77 env CUDA_VISIBLE_DEVICES=`。没有下载任何权重；代码只做了浅克隆（拉到 /tmp）用来阅读。

## A. Fast-WAM（首选）

### A1. 来源
- 论文：Yuan, Dong, Liu, Zhao，"Fast-WAM: Do World Action Models Need Test-time Future Imagination?"，arXiv [2603.16666](https://arxiv.org/abs/2603.16666)；项目页 https://yuantianyuan01.github.io/FastWAM/
- 官方代码：https://github.com/yuantianyuan01/FastWAM（HEAD `7faa71108368fbb3b6885649f112af607427a2d4`，2026-08-20；1.5k stars）
- LeRobot 集成：文档 https://huggingface.co/docs/lerobot/fastwam；代码 `huggingface/lerobot` main 下的 `src/lerobot/policies/fastwam/`（最新提交 `b863b42283`，2026-10-08）。LeRobot main 里同时有 `src/lerobot/envs/metaworld.py`，可以直接用 `lerobot-eval --env.type=metaworld` 评测。

### A2. LIBERO 权重文件清单（HF API `tree/main`）
```
yuanty/fastwam
  12.042 GB  libero_uncond_2cam224.pt                   <- 论文里的 Fast-WAM
   0.000 GB  libero_uncond_2cam224_dataset_stats.json
  12.042 GB  libero_optional_idm_2cam224.pt             <- 一个模型里可切换 first_frame / IDM 两种模式
   0.000 GB  libero_optional_idm_2cam224_dataset_stats.json
  12.042 GB  robotwin_uncond_3cam_384.pt
ZibinDong/fastwam_libero_uncond_2cam224 （LeRobot 格式；ZibinDong 是论文第二作者）
  12.042 GB  model.safetensors
   2.819 GB  Wan2.2_VAE.safetensors
  11.362 GB  models_t5_umt5-xxl-enc-bf16.safetensors
             config.json, policy_{pre,post}processor*.json/.safetensors, google/umt5-xxl tokenizer
lerobot/fastwam_base （从 Wan 初始化、未见过机器人数据的 FastWAM 起点）
  12.042 GB  model.safetensors
```
12.04 GB 的 bf16 权重约合 6.0B 参数，与论文"约 6B"一致。用 meta 设备实测：Wan2.2 DiT 为 5.000B 参数，动作专家约 1.0B（hidden 1024、30 层）。

### A3. LIBERO 成功率
| 来源 | Spatial | Object | Goal | Long | 平均 |
|---|---|---|---|---|---|
| 论文 Table（Fast-WAM） | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| README，optional-IDM ckpt 的 first-frame 模式 | 98.2 | 99.2 | 97.8 | 95.8 | 97.75 |
| LeRobot 文档复现 `ZibinDong/...`（每个套件 500 局） | 97.6 | 99.0 | 95.0 | 94.0 | 96.4 |

一个 ckpt 覆盖全部四个套件，其中包括我们用的 libero_10 和 libero_spatial。输入是 agentview + 腕部相机，各 224×224、横向拼接，外加 8 维 proprio。用原始 ckpt 时要设 `EVALUATION.sigma_shift=5.0`。

### A4. 无状态（读过的推理代码）
- `src/fastwam/models/wan22/fastwam.py:979-1160` `infer_action(prompt, input_image[1,3,H,W], action_horizon, proprio, ...)`：
  - 只用一帧（各相机拼接后的当前帧）：`first_frame_latents = self._encode_input_image_latents_tensor(input_image)`；
  - `video_cache_k, video_cache_v = prefill_video_cache(...)` 在**每次调用里**由当前帧重新算出，用 `.clone()` 复制后只在本次调用的 10 步动作去噪里用，函数返回时丢弃；
  - 返回 `{"action": latents_action[0]}`，类对象上不保存任何跨调用状态。编译后的函数句柄会缓存，但那只是计算图，不是数据。
- 注意力掩码 `fastwam.py:397-418`：`# action -> first-frame video only`（`mask[video_seq_len:, :first_frame_tokens] = True`）。训练时动作 token 也只看第一帧，所以训练和推理一致，推理时不需要未来帧或历史帧。
- LIBERO 评测 `configs/sim_libero.yaml:25-38`：`replan_steps: 10`，`use_action_ensembler: false`（`experiments/libero/action_ensembler.py` 是可选的跨调用平均，默认关闭）。
- LeRobot `modeling_fastwam.py:154-155, 240-245`：唯一的状态是标准的动作队列（`deque(maxlen=n_action_steps)`，整段执行），这与我们 π0.5 执行 10 步的做法相同；`configuration_fastwam.py:192` `n_obs_steps: int = 1`。（训练时 `observation_delta_indices` 会加载 9 帧未来帧作为视频监督，只在训练中用。）
- 文本：umT5-XXL 只是编码器，每个任务的提示词可以缓存（PR #43 的 prompt caching；LeRobot 也接受预先算好的 `context/context_mask`）。

### A5. 微调相关代码
- 官方训练器 `src/fastwam/trainer.py:82-95, 309-318`：`_apply_dit_only_train_mode` 冻结 VAE / T5，把**整个 MoT DiT**（`model.dit = self.mot`，即视频专家 + 动作专家）和 proprio 编码器设为可训，也就是全参训练。没有冻结视频专家的选项，也没有 LoRA；要只训动作专家，需要把 `model.dit.requires_grad_(True)` 换成 `model.action_expert`，约 3 行改动。
  - 配方 `configs/task/libero_uncond_2cam224_1e-4.yaml`：每卡 batch 16，10 个 epoch，lr 1e-4，cosine；README 写明"LIBERO 用单节点 8 卡"（`scripts/train_zero1.sh 8`，DeepSpeed ZeRO-1）。`scripts/train.py` 也能单进程跑（README 里的 LeRobot v3 示例就是这样用的）。
  - 数据：原生支持 LeRobot 2.1 和 3.0（`configs/data/libero_2cam_lerobot_v30.yaml`；`src/fastwam/datasets/lerobot3/`）。
- LeRobot `configuration_fastwam.py:185-191, 226`：`freeze_video_expert: bool`，文档原文："Freeze the ~5B Wan video expert so only the action expert + proprio encoder train ... (If enabled, also set `loss.lambda_video=0`)"。实现在 `modeling_fastwam.py:80-90`，`get_optim_params` 只返回 `requires_grad` 的参数（`:142-152`）。另有 `use_gradient_checkpointing`、通用的 `use_peft` 入口（`:291`）、FSDP2 包装单元（`_fsdp_wrap_modules = ["MoTLayer"]`）。
- LeRobot 跨本体加载 `modeling_fastwam.py:94-140`：LIBERO 是 7 维动作 / 8 维 proprio，MetaWorld 是 4 维 / 4 维，形状不一致的张量会被丢弃并重新初始化，其余全部加载。
- MT50 数据（`lerobot/metaworld_mt50/meta/info.json`）：codebase v3.0，2,500 局，204,806 帧，`observation.image` 480²，`observation.state` 4 维，`action` 4 维，fps 80。FastWAM 的 LeRobot 适配会把图像缩放到 `image_size`（MT50 设 `(224,224)`）。动作块长度可以缩短，比如 `action_horizon=16, num_video_frames=17` 也满足 `validate_features` 的约束。

### A6. 吞吐锚点（已发布的实测数字）
- EasyWAM README "Efficiency Results"：同一 Fast-WAM 架构，8×H100、每卡 batch 16，**EasyWAM-MoT 138.2 样本/s，FastWAM 原代码 51.5 样本/s**；"On LIBERO, training for 20,000 steps takes approximately 5 hours with EasyWAM, compared with approximately 14 hours using the original FastWAM codebase"。
- 每个 LIBERO 样本的 FLOPs（双相机 224×448）：DiT 6·(5.0B·294 + 1.0B·32) = 9.0 TF，加上 VAE 编码 9 帧 × 0.466 TF = 4.2 TF，共 13.2 TF。由此推出有效算力：EasyWAM 约 228 TFLOPS/H100，原代码约 85 TFLOPS/H100。4090（bf16 密集峰值 165 TFLOPS，H100 是 989）按 H100 的 1/4 计。
- MT50 单相机 224²（147 个视频 token）：
  - 全参 6·(5.0·147 + 1.0·32) + 2.1（VAE）= 6.7 TF；
  - LoRA：2·767 前向 + 2·735 视频激活梯度 + 4·32 动作 + 2.1 = 5.2 TF；
  - 冻结视频：2·735 + 6·32 + 2.1 = 3.8 TF。
  - 乘以 1.02M 样本，再除以上面的有效算力，就得到 DECISION 第 3 节的表。
- 显存：全参用 AdamW 需要 6B × 16 B ≈ 96 GB（bf16 权重 + fp32 主权重 + 两份 Adam 状态 + 梯度），单卡放不下。LoRA 的可训参数是 1.34B（与 EasyWAM `libero_mot_wan22_lora_128.pt` 2.69 GB / 2 B 一致：rank 128 的 LoRA 3.2 亿 + 动作专家约 10 亿），约 21 GB 状态加 10 GB 冻结的 bf16 视频专家。
- 可降成本的点（不是官方选项）：如果 `lambda_video=0`，动作只依赖第一帧，可以只编码 / 前向第一帧，每个样本约 0.9 TF，再便宜约 4 倍。代码里没有这个开关，要自己改。

### A7. 每次调用成本（要求 ⑤）
- README "What's New"：端到端推理（含文本编码和 VAE 编码）在 **RTX 4090 上从 190 ms 降到 110 ms**，H20 上从 470 ms 降到 210 ms（`compile_action_infer=true`，LIBERO 默认开）。论文：190 ms，RTX 5090D V2 32 GB，10 个去噪步。
- CPU meta 设备 FLOP 计数（用 openpi venv 里的 diffusers 0.37.1 和 Wan2.2-TI2V-5B-Diffusers 的 config，随机权重；脚本 `/tmp/r17s3o_probe/scripts/flops.py`）：
```
VAE total params 704.7M, encoder 149.6M
Wan2.2-5B DiT params 5.000B
VAE encode 1 frame 224x224: latent (1, 48, 1, 14, 14)  233.1 GFLOP
  Wan DiT fwd on 1 latent frame (7x7 tokens): 579.5 GFLOP
VAE encode 1 frame 224x448: latent (1, 48, 1, 14, 28)  466.4 GFLOP
  Wan DiT fwd on 1 latent frame (7x14 tokens): 1010.0 GFLOP
```
- 动作专家每步 2·1.0B·32 ≈ 64 GFLOP，10 步约 0.64 TF。batch 1 时受带宽限制：每步读 2 GB 权重；视频专家预填读 10 GB；4090 带宽约 1 TB/s，所以预填 ≥ 10 ms、10 步 ≥ 20 ms。
- 检索键的选择：Wan VAE 对当前帧的潜变量（双相机 48×14×28；作用与 π0.5 的 SigLIP 特征相同）。命中只花 VAE 编码 + 检索，跳过预填和 10 步动作去噪。VAE 编码延迟没有实测，按 conv3d 在 4090 上 50–80 TFLOPS 有效算力估为 6–12 ms。只看 FLOPs 的下界：(0.47 + 1.01 + 0.64) / 0.47 = 4.5×。DiT 在 batch 1 时受带宽限制、VAE 受算力限制，所以按延迟算的比值会更高，估计 8–15×。
- 减步（R16 的做法）：动作头是流匹配 10 步（与 π0.5 / GR00T 同类）。社区有 1 步的 LoRA（`ElysiaTrue/fastwam-dmd-1step`，自报 LIBERO 约 96%）。k = 1 时完整调用只剩 VAE + 预填 + 1 步，缓存每次命中省下的就主要是预填的约 10–15 ms 以上（估）。

## B. EasyWAM-MoT-Wan22（备选）
- 代码：https://github.com/OpenMOSS/EasyWAM（2026-10-02，400 stars）。README 写明 EasyWAM-MoT "is based on FastWAM"、"EasyWAM-MoT and FastWAM use the same model architecture"。支持矩阵：所有模型都支持全参和 LoRA 微调；数据管线统一为 LeRobot v3。
- HF 文件清单：
```
OpenMOSS-Team/EasyWAM-MoT-Wan22      12.042 GB libero_mot_wan22.pt ; 2.687 GB libero_mot_wan22_lora_128.pt
OpenMOSS-Team/EasyWAM-MoT-Cosmos25    5.553 GB libero_mot_cosmos25.pt
OpenMOSS-Team/EasyWAM-Hidden-Wan22   12.067 GB libero_hidden_wan22.pt ; 2.712 GB libero_hidden_wan22_lora_128.pt
OpenMOSS-Team/EasyWAM-Unified-Wan22  10.001 GB libero_unified_wan22.pt ; 0.646 GB libero_unified_wan22_lora_128.pt
（另有 MoT-Joint / MoT-IDM / Hidden / Unified 的 Cosmos25 版本）
```
- LIBERO（README）：MoT-Wan22 全参 97.0 / 99.2 / 96.6 / 94.0，平均 96.7；LoRA r128 96.8 / 99.6 / 97.4 / 90.0，平均 95.9。MoT-Cosmos25 平均 97.6。
- LoRA 配方：`configs/model/lora/video_dit.yaml`（`peft.LoraConfig r=128, alpha=128`，目标模块由各骨干的 `lora_target_modules` 指定）；`configs/task/libero_easywam_mot_wan22_lora.yaml`（batch 16，20k step，lr 1e-4）；`src/model/component/lora.py` `set_video_dit_lora_trainable`，视频 DiT 只训 LoRA，动作 DiT 全训。另有 `scripts/train_zero2_offload.sh`，单卡全参时可以把优化器状态放到 CPU。
- 无状态：`src/model/easywam_mot.py:1441` `infer_action_batch(prompt, input_image, ...)`，由当前帧得到 `first_frame_latents`；效率文档原文："Cross-K/V reuse is inference-only and does not persist across environment replans."
- Cosmos25 骨干的保留意见：`configs/model/backbone/cosmos25.yaml` 里 `reason_model_id: Cosmos-Reason1-7B`、`text_dim: 1024`，文本编码器是 VLM。
- 风险：文档只提供 `resume` 和 `action_dit_pretrained_path` 两种初始化方式，没有"从 LIBERO ckpt 换动作维度继续微调"的说明。

## C. 其他候选的证据
| 模型 | 证据 |
|---|---|
| Faster-WAM | HF `hustvl/FasterWAM`：`libero/step_021700.pt` 11.118 GB、`robotwin/step_029355.pt`；代码 https://github.com/hustvl/FasterWAM（arXiv 2608.04404），README 原文："computes future representations once and selectively reuses them during action denoising"，在 Fast-WAM 代码基上实现 |
| RealtimeWAM | HF `lightx2v/RealtimeWAM`：`realtimewam_libero_fast.pt` 12.04 GB、`..._fast_lora.pt` 0.66 GB、`..._faster.pt` 11.1 GB；单张 H100 上 12.2 ms（来自搜索摘要） |
| Streaming-WAM | HF `SJTU-DENG-Lab/Streaming-WAM`：`ac-stream/model.pt`、`joint-cd/model.pt` 各 12.04 GB |
| LingBot-VA | HF `robbyant/lingbot-va-posttrain-libero-long`（transformer 约 10.2 GB + umT5 + VAE）；`evaluation/libero/client.py:124` `model.infer(dict(obs=key_frame_list, compute_kv_cache=True, ...))`；`wan_va/wan_va_server.py:377-393` 只在 `_reset` 时 `clear_cache`，说明跨调用保留 KV cache |
| UVA | https://github.com/ShuangLI59/unified_video_action README：LIBERO 只有 `gdown 11c2VrmaRp48yw__5A5xpcu8EPzkexHSi -O checkpoints/libero10.ckpt`；`config/task/libero10.yaml`：`n_obs_steps: 16`、`horizon: 32`、`n_action_steps: 8`；README 原文："We recommend using at least 4 GPUs for training"；论文 arXiv 2503.00200，LIBERO-10 500 局 0.90 |
| UWM | https://github.com/WEIRDLabUW/unified-world-model README：只发布了 "pretrained LIBERO-90 checkpoint"（Google Drive），下游任务要自己 finetune（例如 `libero_book_caddy`） |
| VPP | https://github.com/roboterax/video-prediction-policy README 的 Checkpoints 表：clip-vit-base-patch32、svd-robot、svd-robot-calvin、dp-calvin，没有 LIBERO |
| PAD | arXiv 2411.18179（NeurIPS 2024）只做了 MetaWorld + 真机；没找到代码仓库或权重 |
| Seer | https://github.com/OpenRobotLab/Seer README：Checkpoints 只有 CALVIN ABC-D 和 Real-World（DROID 预训练）；LIBERO-LONG 只放了代码 |
| MoDE | HF `mbreuss/MoDE_LIBERO_10`：`model_cleaned.safetensors` 3.317 GB；`mbreuss/MoDE_LIBERO_90` 也存在；架构是扩散 Transformer 策略，属 DP 族 |
| FLOWER | HF `mbreuss/flower_libero_{spatial,object,goal,10,90}`；是 VLA（Florence-2 VLM） |
| GE（Genie Envisioner） | HF `agibot-world/Genie-Envisioner-v1.0`：只有 GE_base_fast / slow；`Genie-Envisioner-Sim-v2.0`：gesim 2B + pi05_gesim，没有 LIBERO；README 原文："takes sparse memory frames as input" |
| Motus | https://github.com/thu-ml/Motus README：VLM 是 Qwen3-VL-2B（约 2.13B）；HF 上只有 `motus-robotics/Motus_robotwin2` 等，没有 LIBERO |
| mimic-video | 只有第三方的 `alicechan/mimic-video-w2a-libero-object-checkpoints`（只覆盖 libero-object，非官方） |
| X-WAM | `sharinka0715/X-WAM-checkpoints`：robocasa_sft / pretrained / robotwin_sft（见 `logs/xwam_dp_nfe_plan.log.md`），没有 LIBERO |

## D. 局限
- 所有 GPU 时间都是用 FLOPs 换算已发布吞吐得到的估计，没有在我们的卡上实测（规则禁止用 GPU）。LeRobot 移植版的训练吞吐没有已发布数字，表里按 Fast-WAM 原代码的效率估。
- "VAE 编码 6–12 ms"是估计值；建议上卡后第一件事是测阶段耗时：VAE 编码、预填、每个动作步各多少 ms。
- MetaWorld 单相机的视觉域与 LIBERO 差别大，冻结视频专家可能掉点（论文"去掉视频协同训练"从 97.6 降到 93.5，不过那是从 Wan 初始化训起的情况）。如果冻结版在 MT50 上明显不如 GR00T / DP，再改用 LoRA 版。
