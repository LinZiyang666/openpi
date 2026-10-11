# 独立选型证据

核验日期：2026-10-08（America/Chicago；网页存档时间跨至 UTC 10-09）。本报告独立阅读指定历史日志、原论文、官方实现和模型仓库；没有读取另外两份研究输出。最终排序：**JEPA-WM + PAD-XL/2，MetaWorld；Cosmos Policy 2B 为唯一、限 LIBERO 的模型备选；不另列 benchmark 备选。** 这是当前可核验资产中的条件推荐，不是所有硬条件已经通过的声明。

原始材料保存在 [sources/](sources/)，逐项 URL、HTTP 状态、时间、SHA256 见对应 `.meta.json`。[checkpoint_inventory.json](checkpoint_inventory.json)保存 HF revision、文件大小和 LFS 哈希。只下载元数据、配置、代码、小型网页，以及 PAD 文件前 256 KiB、Adapt2Act ZIP 尾部 128 KiB；未下载完整大权重、未运行 GPU。

## 1. 先把约束交集讲清楚

| 条件 | JEPA-WM | PAD-XL/2 | Cosmos Policy 2B（备选） |
|---|---|---|---|
| 非 VLA | 是，DINOv2 潜空间动力学 + CEM | 是，VAE + 文本编码器 + 联合 DiT；无 VLM 主干 | 是，视频扩散世界策略；无 VLM 主干 |
| 不重复家族 | 搜索规划，无扩散动作头 | 图像与动作联合扩散 | 与 PAD 都属视频/动作生成，应替换而非并列凑数 |
| 已发布 MetaWorld 权重 | 是；正式评测仅 reach / reach-wall | 是；论文有效结果 48 项 | 未找到 |
| 已发布 LIBERO 权重 | 未找到 | 未找到 | 是 |
| 自己训练 | 不需要 | 不需要 | 不需要 |
| 新颖性 | 2025-12 论文 / 2026 实现，较新 | NeurIPS 2024，明确是弱项 | 2026，较新 |
| 缓存能绕过什么 | CEM 全部候选动力学 rollout | 多步大 DiT；保留原控制器 | 多步视频 DiT；保留当前图像 VAE |
| 盈利证据等级 | 代码级计算量推导，尚无分段计时 | 论文 FLOPs + 代码结构，尚无分段计时 | 有本地全调用和另一环境的分段测量 |

不能声称这三者构成“所有模型 × LIBERO/MetaWorld”的完整矩阵。**若完整矩阵是硬要求，当前调查没有合格解。** 即使只要求每个新增模型在 MetaWorld 有权重，JEPA 的任务覆盖和 PAD 的 v2/v3 兼容仍是实质问题，不应在论文里省略。

## 2. 已有测量怎样用于选型

本地依据：用户指定的 [缓存综合结果](../../r16/REPORT.md)、[前轮结果](../../r15/REPORT.md)、[会话交接 §1](../../../../../logs/session_handoff.md)、[Cosmos 日志](../../../../../logs/cosmos_nfe_plan.log.md)、[X-WAM / DP 日志](../../../../../logs/xwam_dp_nfe_plan.log.md)、[ε/x₀ 修正记录](../../../../../logs/x0_multimodal_plan.log.md)、[VLA 减步日志](../../../../../logs/nfe_baseline_plan.log.md)、[RoboCasa365 日志](../../../../../logs/nfe_baseline_rc365_plan.log.md)、[MetaWorld 记录](../../../../../logs/metaworld_warm_reset.log.md)。这些相对路径指向仓库内原始材料，没有复制或改动它们。

| 已有对象 / 测量口径 | 数字 | 能支持什么，不能支持什么 |
|---|---|---|
| π0.5 LIBERO 既有成本模型 | vision 10.260 + prefix 27.686 + head 29.572 = 67.519 ms | 特征之后有足够计算；不能将其视觉时间赋给新模型 |
| GR00T 既有成本模型 | vision 6.146 + prefix 7.192 + 3.513×k ms，k=8 共 41.442 ms | 既有缓存并非只节省动作头；不能转移成 PAD 延迟 |
| 服务 CPU 剖析 | 检索与插件约 7.7 ms；GR00T 预处理另有 14.8 ms；π0.5 预处理 0.44 ms | 新模型预算暂取 8 ms，必须重测；不能假设检索/修正免费 |
| Cosmos LIBERO，全调用 | H100 空载 524 ms；论文 610 ms；4090 冒烟 840 ms，p50 813 ms | 有绝对成本；缺少 LIBERO 独立 VAE 拆分 |
| Cosmos RoboCasa-2024，H100 | 原始 k5/k1=821/565 ms；去掉未来图像解码后 576/306；再 CUDA graph 后 470/295 | 固定部分约 200 ms，单 DiT 约 45 ms；优化后再算缓存，避免把无用解码记成贡献 |
| X-WAM，H100 空载 | action k10=390 ms | 很值得加速，但公开权重不支持选择的 benchmark |
| X-WAM，排队服务 | k10/5/3/2/1 约 760/430/320/270/220 ms；4090 并发 950 ms | 包含排队，不能拟合成纯 GPU 各层耗时 |
| 官方 CNN Diffusion Policy，4090 空载 | square：DDPM100=1305 ms；DDIM10/4/2/1=144/64/37/24 ms；其余任务 k1=19–34 ms | 全步很贵，但减步后很便宜；不能用 100 步人为制造盈利 |

DP 日志还记录过旧 diffusers 的 DDPM 调度器 `previous_timestep=t-1` 错误，以及 DDIM 单步取 t=0 的口径问题；后续已修复/用 trailing t=99 对照。不能以早期错误结果概括“ε 预测天生不能减步”。已有日志里 X-WAM k1 保持成功率的事实，也要求新实验以有效少步基线比较，而非只对默认全步。

**统一成本定义。** 令 `E` 为当前观察生成检索特征的全部开销，`D` 为从这些特征到动作的可跳过计算，`R` 为检索、PCA/距离和修正等额外成本。理想 GPU 全调用/命中比为 `(E_gpu+D_gpu)/E_gpu`；保守串行服务路径为 `(E+D)/(E+R)`，每命中相对纯推理节省 `D−R`。若每个决策都先检索，平均节省为 `pD−R`，故命中率还须满足 `p>R/D`。额外守卫调用、键的历史更新、预处理必须单列，不能重复算作免费。

8 ms 是**从现有系统借来的待验证 CPU 预算**，不是 GPU 时间。下文涉及它的比值是服务路径估算，不是实测 GPU 占用比。低延迟对象必须同时报告 GPU 时间、服务时间及回合 wall time。

## 3. JEPA-WM：优先验证的不同计算家族

**原始来源。** [论文 2512.24497](https://arxiv.org/abs/2512.24497)、[官方仓库](https://github.com/facebookresearch/jepa-wms)、[MetaWorld 评测配置](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml)、[CEM 实现](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/evals/simu_env_planning/planning/planning/planner.py)。本地对应 `jepa_paper.*`、`jepa_mw_eval.yaml`、`jepa_planner.py` 等。

**具体选择。** 选 `jepa_wm_metaworld.pth.tar`，不是 DROID 版，也不是不存在于发布列表中的“MetaWorld V-JEPA2-fixed”。仓库中 V-JEPA2-AC 的发布条目是 DROID/OSS，不能从论文比较表推断所有变体都有 MetaWorld 权重。相较 DINO-WM，这个版本纳入本体状态、AdaLN、RoPE 和多步 rollout 训练；后续 LeWorldModel 虽更新，官方发布的是 PushT / Cube / TwoRooms / Reacher，没有核验到 MetaWorld 版本，因此不替代本选择。[发布列表](https://huggingface.co/facebook/jepa-wms/tree/9b9c41ef249466630dbf1a20e78391865d07b3b9)、[LeWorldModel 官方说明](https://github.com/lucas-maes/le-wm)

**结构与键。** 单相机 RGB 224² → 冻结 DINOv2 ViT-S/14 → 16×16 个 384 维 patch 特征 → 6 层、384 宽、16 头的动作条件预测 Transformer → CEM 在潜空间选择连续 4 维动作。配置文件名含 `r256`，实际 `img_size=224`，以内容为准。检索键取原生 patch 特征的空间池化/PCA；缓存还须区分目标图像、任务、本体状态及历史，否则“相同当前位置、不同目标”会错误复用。目标编码可每回合一次；命中期间也需正确维护模型历史。

默认 CEM 为 **15 次迭代、300 条候选、6 个规划时刻、10 个 elite**；`frameskip=5, action_skip=1`，一个规划时刻包含 5 个底层动作，完整计划 30 个底层动作，执行前 3 个规划时刻即 15 个动作后重规划。没有扩散步数。采样数量/迭代次数是另一种预算旋钮，不能称为现有 flow/diffusion 的 k=1–2。延长执行时域可从 15 向 30 试，但闭环成功率未知。配置中的逐次图像解码用于展示，公平基线应关闭并避免加载 3.6 GB 可视化 decoder。[模型代码](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py)、[执行代理](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/evals/simu_env_planning/planning/gc_agent.py)

**计算账。** 最小按一帧 256 patch、宽 d=384、6 个标准 Transformer block，只计 QKV/输出投影与 4d MLP，两次浮点运算/乘加：

`F_pred ≥ 2 × 6 × 12 × 256 × 384² = 5.436 GFLOP`

`F_plan ≥ 15 × 300 × 6 × F_pred = 146.767 TFLOP`

这里针对完整 horizon=6 的规划调用；回合尾部若 `steps_left<6`，需按实际 horizon 重算。

未计 attention、动作/本体嵌入、多帧上下文和日志 rollout，因此是保守的代码级计算下界。它说明参数量仅几十 M 不等于每决策便宜：同一预测器被用于大量候选。模型的主要可跳过成本在原生视觉编码之后。[AdaLN 实现](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/models/AdaLN_vit.py)

按 NVIDIA 公布的稠密 BF16/FP16（FP32 累加）峰值，4090 165.2 TFLOP/s、H100 SXM 989.5 TFLOP/s，理论 `D_min` 分别 **888.4 / 148.3 ms**。这不是测量，也不保证软件达到该速度；实际 FP32、上下文增长、内存和调度只会使当前算法更慢，减少候选/算法改写则需重算。[Ada 白皮书，附录规格](https://images.nvidia.cn/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)、[H100 规格，注意标星为稀疏值](https://www.nvidia.com/en-us/data-center/h100/)

为使缺少 `E` 的影响可审查，给**假设检验而非延迟预测**：若 `E=10 ms, R=8 ms`，上述下界对应全调用/命中至少 **49.9× / 8.80×**，单命中净省至少 **880 / 140 ms**；若 `E=30 ms`，比值变为 **24.2× / 4.69×**。没有测得 E，不能将任何一个倍数作为实验结果。理论上满足 2× 的条件为 `E≤D−2R`，两个设备在这个极乐观 D 下分别允许 E≤872 / 132 ms；结构上的盈利余量明显。

**适用层。** 特征缓存、无进展回退、固定执行时域、周期真调用、按任务调参适用；RFF/ridge 可校正连续动作计划，但会改变 CEM 最优序列，需独立验证。延长时域适用。扩散减步不适用；CEM 缩减须作为额外强基线，不能仅对最贵默认搜索量展示缓存收益。

**权重核验。** [不可变文件链接](https://huggingface.co/facebook/jepa-wms/blob/9b9c41ef249466630dbf1a20e78391865d07b3b9/jepa_wm_metaworld.pth.tar)：**211,687,551 bytes**；API public、`gated=false`；发布仓库包含对应 DINO-WM 比较权重，但没有 LIBERO 文件。代码/模型为 **CC-BY-NC-4.0**；DINOv2 本体为 Apache-2.0。见 [API 快照](sources/jepa_hf.json)、[许可证快照](sources/jepa_license.txt)。没有加载权重验证完整 state_dict 或运行成功率。

**复现风险：高于“下载即跑”。** 模型训练数据包含 42 项任务，但论文 MetaWorld 正式规划结果是 reach / reach-wall，不能把训练范围等同有效控制范围。目标来自专家终点图像，和语言条件策略的任务输入不等价；应忠实使用原协议，目标图像生成成本另计。当前代码使用 v3 环境、corner2 摄像机、目标/速度状态；与 PAD v2 不一致。依赖 torch≥2.7、Python3.10、torchcodec 等需要单独环境。若仅这两项简单任务可复现，宜作为“搜索规划也能获益”的机制补充，而非足够广的主实验证据。

## 4. PAD-XL/2：已有完整操控权重，但新颖性和接口有代价

**原始来源。** [NeurIPS 2024 论文](https://arxiv.org/abs/2411.18179)、[官方仓库](https://github.com/Robert-gyj/Prediction_with_Action)、[推理代理](https://github.com/Robert-gyj/Prediction_with_Action/blob/a0bdc12e05d435a77efd830ee6953d556c1a6083/evaluation/agent.py)、[MetaWorld 运行器](https://github.com/Robert-gyj/Prediction_with_Action/blob/a0bdc12e05d435a77efd830ee6953d556c1a6083/run_metaworld.py)。本地 `pad_paper.*`、`pad_agent.py`、`pad_eval.py`。

**为什么仍保留旧模型。** 它不是全局最新的世界动作模型；优势是已验证发布了 MetaWorld 专用完整权重、48 项任务有正式结果，且 VAE 后的大主干可整体跳过。后来的 VPP/S-VAM 公开检查点聚焦 CALVIN；Cosmos/X-WAM 不支持 MetaWorld；Adapt2Act 的可直接运行视觉规划分支弱，Spatial Policy 又依赖远程 GPT-4o。因此这里优先选择训练免除与有效覆盖，明确牺牲新颖性。如果 owner 不接受 2024 年成员，这一位置应留空，不能声称更新成员不存在。

**结构与键。** 当前 RGB256² → SD `sd-vae-ft-mse` encoder → 4×32×32 latent；CLIP ViT-B/32 的**文本分支**给任务条件，文本按指令预编码；当前 latent 和 3 个未来 latent 联合输入 DiT-XL/2（28 层、1152 宽、16 头，约 661M）以及动作 token，联合预测未来图像与动作。可取 VAE encoder 最后 mid-block 的 512 通道特征图，经空间池化/PCA作键；不采用 4 通道最终 latent 的均值冒充语义强特征，也不将随机 posterior sample 当稳定键。这个键的检索效果**尚未验证**。[VAE 配置](https://huggingface.co/stabilityai/sd-vae-ft-mse/blob/31f26fdeee1355a5c34592e401dd41e45d25a493/config.json)

通过 HTTP Range 仅取已发布 `.pt` 的前 256 KiB，读取 ZIP 内 pickle 指令字符串（`pickletools`，没有执行 unpickle），实际参数为 `DiT-XL/2, image_size=256, action_steps=3, action_dim=4, predict_horizon=3, absolute_action=True, skip_step=4, depth=False`。见 [权重前缀](sources/pad_checkpoint_prefix.bin)、[参数指令证据](sources/pad_checkpoint_pickle_ops.txt)。这比用论文真实机器人 7 维动作配置推断 MetaWorld 文件可靠。

**默认步数存在真实冲突。** 发布配置 `denoise_steps=50`；代理调用 `p_sample_loop` 和 respaced diffusion；论文正文称 **75 步 DDIM**。不能混写成“默认 50 步 DDIM”。先复现发布路径，再单独确认论文路径；更换 solver 必须明确标记。未来图像 decoder 仅用于可视化时应从两组基线同时删除，但联合去噪图像 token 对动作计算不可删除。

**动作块不是底层三步。** 文件输出 3 个绝对 XYZ+gripper 目标，采样间隔为 4；运行器只取第一个目标，用反馈运动控制器最多推进 50 个 simulator step，再下一次模型调用，默认最多 20 次模型规划。现有缓存若存原始 delta-action chunk，需适配：缓存/修正绝对目标序列，命中后依然由相同控制器生成底层动作；不要直接重放另一状态下的 delta 序列。延长时域的含义是执行更多预测目标，不能写成 commit=3 个 MuJoCo 步。

**计算账。** 论文表 3/4 报 DiT-XL/2 为 119.1 “Gflops”，50 步合计 5.955 T 论文计数单位；未说明 MAC 的乘加计数约定，也没有 encoder/decoder 分段延迟。为避免与 GPU 规格混用，另外按标准 dense Transformer 的两次运算/乘加计算：

`F_DiT ≥ 2×28×12×257×1152² = 229.196 GFLOP/step`

`F_D(50) ≥ 11.460 TFLOP`

忽略 attention 与条件映射。以同一极乐观稠密半精度峰值折算，4090 **D≥69.37 ms**，H100 SXM **D≥11.58 ms**；原生 FP32、未开启 TF32 时，按普通 FP32 峰值则为 **138.7 / 171.0 ms**。FP32 数字不能作为优化后基线的保证。H100 的低 batch 权重流量/算子开销可能使实际远大于半精度算力下界，但本次不拿未测效率系数编造秒数。

| 假设，仅作敏感性分析 | 4090 半精度计算下界 | H100 SXM 半精度计算下界 |
|---|---:|---:|
| 若 E=10 ms，R=8 ms：全调用/命中 | ≥4.41× | ≥1.20× |
| 同假设：单命中净节省 D−R | ≥61.37 ms | ≥3.58 ms |
| 若 E=30 ms，R=8 ms：全调用/命中 | ≥2.61× | ≥1.09× |

**这些是带有未验证 E 假设的下界，不是 PAD 的实际加速比。** 只凭现有数据无法确认 H100 上可观盈利；即便 4090 结构上更有利，也未测 native encoder、检索键质量和少步最强基线。减到 k=1–2 后上述 50 步下界不成立，必须重新比较。不能将普通 CNN DP 的 13 ms/步移植成 PAD 每步耗时。

**适用层。** 缓存、无进展回退、RFF/ridge（在目标位姿空间）、周期真调用、按任务调参可以做；需要保持绝对目标控制器一致。扩散减步可以测但效果未知；多目标执行时域可以扩展，效果未知。论文主模型 ε 预测和联合视频去噪不能自动继承既有 x₀ 单步结论。

**已发布文件。** 仓库 [yjguo/pad_bridge_pre@bdddfe9e…](https://huggingface.co/yjguo/pad_bridge_pre/tree/bdddfe9e7fa25786061a73b0f2010e81d24a6457)实际列出：

| 文件 | bytes | 用途 |
|---|---:|---|
| [bridge_pre_mw_ft.pt](https://huggingface.co/yjguo/pad_bridge_pre/blob/bdddfe9e7fa25786061a73b0f2010e81d24a6457/bridge_pre_mw_ft.pt) | 2,709,726,746 | 本次选用的 MetaWorld 微调权重 |
| [bridge_pre.pt](https://huggingface.co/yjguo/pad_bridge_pre/blob/bdddfe9e7fa25786061a73b0f2010e81d24a6457/bridge_pre.pt) | 2,698,904,040 | Bridge 预训练；不能拿它代替 MetaWorld 版 |

API `gated=false`，模型卡 MIT，代码 LICENSE.txt 为 MIT；VAE 仓库也标 MIT。没有找到 LIBERO 检查点。未来正式复现仍需下载 SD VAE 与 CLIP 小模型；本次只核验配置/元数据。[API 快照](sources/pad_hf.json)、[许可证](sources/pad_license.md)

**复现风险。** vendored MetaWorld v2 / mujoco-py 老栈，与当前 v3 不等价；paper 摄像机描述与发布配置 `corner3` 有差异；关闭 `visualize_prediction` 后运行器有 `img_all` 未赋值路径，需小修；论文 headline 虽称全 benchmark，附录明确排除两个 handle-pull 任务，72.5% 应标 48 项结果，不能写 MT50 全覆盖。修复这些不是训练，但也不是零成本接入。没有验证该文件恢复论文成功率。

## 5. 唯一备选 Cosmos Policy 2B：成本证据最强，benchmark 不匹配

选 [Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B/tree/cb689ec0e3347c13667d70a78a3447388f5c3bb8)，[官方实现](https://github.com/nvlabs/cosmos-policy)。它是当前已在本项目跑通且有公开 benchmark 权重的视频世界策略；不能声称是整个 Cosmos 家族最新版本。更新 Cosmos3 已有 DROID 及部分社区 LIBERO 权重，未核验到等价的 MetaWorld 发布版，不适合作为本次低风险替换。

结构：当前相机图像 → Cosmos 视频 tokenizer/VAE → 带图像、动作、本体状态和未来状态的 latent 序列 → Predict2 2B DiT → 动作 latent 解码。键应取**当前图像 encoder 的确定性空间特征**，不要为取键先运行完整 DiT。LIBERO action chunk=16、执行16；默认 5 个 solver steps 实际含 `sample_clean`，即 **6 次 DiT NFE**。可减步；LIBERO 默认已执行完整 chunk，不能把 horizon>16 叫“免费延长现有预测”，重复尾动作需另定义。

全调用 LIBERO：H100 524 ms、4090 840 ms；论文约610 ms。LIBERO 视觉编码拆分没有独立测量，故**该 benchmark 的 full:hit 比和净省时仍未知**。RoboCasa-2024 已有优化后拆分可作为可行性证据：full≈470 ms，E_fixed≈200 ms，R假设8 ms → hit≈208 ms，full:hit≈**2.26×**，净省≈**262 ms/命中**；k1 full295ms → **1.42×**、87ms。这里 E_fixed 含预处理/VAE/latent装配，不是纯视觉 kernel；不能跨到 LIBERO、4090 或 RoboCasa365 使用。缓存仍须对比这些优化后基线，不能把 821→470ms 算成本方法收益。

文件：[`Cosmos-Policy-LIBERO-Predict2-2B.pt`](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B/blob/cb689ec0e3347c13667d70a78a3447388f5c3bb8/Cosmos-Policy-LIBERO-Predict2-2B.pt) **3,913,017,345 bytes**；`config.json`1354、`libero_dataset_statistics.json`2372、`libero_t5_embeddings.pkl`41,957,938 bytes。public、ungated；最终 policy 权重模型卡写 **NVIDIA One-Way Noncommercial License (NSCLv1)**，不能误报成基座的 NVIDIA Open Model License。基座获取可能另有条款/授权；本地已有下载历史不等于新用户无门槛。[模型卡快照](sources/cosmos_modelcard.md)、[API](sources/cosmos_hf.json)

缓存、守卫、修正、周期/按任务旋钮均结构可用，缓存特征质量未测。没有 MetaWorld 权重，故只在接受“新增模型不必都跑第二 benchmark”时替换 PAD；不是严格要求下的合格后备。

## 6. MetaWorld：实测轻量性、权重交集与版本边界

**新近性和影响力。** 原始 MetaWorld 属 CoRL2019/2020 时期，不能写成 2025 首发。官方仓库现引用 **Meta-World+: An Improved, Standardized, RL Benchmark，NeurIPS2025 Datasets and Benchmarks**；它具有50个统一机器人任务，并被本次核验的 PAD、JEPA-WM、RLinf、LeRobot 等不同项目实际采用，影响力证据来自这些原始使用记录，不以不稳定的搜索引用数充数。[官方仓库](https://github.com/Farama-Foundation/Metaworld)、[Meta-World+ 论文](https://papers.nips.cc/paper_files/paper/2025/file/da295a7037612bb3bfc1334a6f8fc8bd-Paper-Datasets_and_Benchmarks_Track.pdf)、[LeRobot 官方接入文档](https://huggingface.co/docs/lerobot/en/metaworld)

正式 arXiv 为 [2505.11289](https://arxiv.org/abs/2505.11289)。PyPI 元数据核验：3.0.0 于2025-06-14发布，3.1.1于2026-06-28发布；本次探针固定 **3.0.0**，没有擅自把已有权重搬到最新版。[PyPI 快照](sources/metaworld_pypi.json)

**缓存友好程度。** 每项任务固定机器人、场景资产和操作类型，固定相机下不像 RoboCasa365 跨大量厨房布局/物体资产切换；初始物体和目标位置仍可以变化，不能把“缓存友好”解释为只跑同一个状态。应按任务/目标条件分库，保留位置随机化，使用独立库/调参/评测种子；邻域覆盖、真实命中率和成功率尚未测。JEPA 的目标图像会变，键必须包含目标身份/目标特征。LeRobot 某些数据集约定使用固定物体/目标布局，和官方随机位置协议也不能混为一谈。

**π0.5 权重。** [RLinf/RLinf-Pi05-MetaWorld-SFT@60b2d030…](https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT/tree/60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e)：

| 实际文件 | bytes |
|---|---:|
| [model.safetensors](https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT/blob/60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e/model.safetensors) | 7,473,091,464 |
| metadata.pt | 3,172 |
| lerobot/metaworld_mt50/norm_stats.json | 1,134 |

public、ungated。[RLinf 官方文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html)和本地成功运行记录相互印证。该仓库当前没有可核验的 license 声明；基础 openpi 的许可证不能自动当作这份微调权重的明确授权。输入 corner2 480² 图像、4维本体/动作，既有本地流程旋转图像180°、无 wrist 时 mask=false、10 flow steps、执行5步；这些约定不能为迎合新模型任意统一。

**真实 CPU-only 探针。** 新装隔离环境：`/home/weiland/projects/openpi_ext/lines/r17_survey_astra_cpu/venv`。仅 metaworld3.0.0、mujoco3.15.0、gymnasium1.4.0、numpy2.5.3、PyOpenGL3.1.10 及 scipy/Pillow/packaging 等依赖；未安装模型、没有 CUDA policy。安装缓存保存在本输出目录。完整脚本 [metaworld_cpu.py](probes/metaworld_cpu.py)，结果 [metaworld_cpu.json](probes/metaworld_cpu.json)，汇总 [metaworld_cpu_summary.json](probes/metaworld_cpu_summary.json)。

执行亲和性严格为 `taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES=`，额外强制 Mesa EGL vendor、`LIBGL_ALWAYS_SOFTWARE=1`、`MESA_LOADER_DRIVER_OVERRIDE=llvmpipe`、`LP_NUM_THREADS=1`；实际 GL renderer 返回 **llvmpipe (LLVM20.1.2,256bits)**，vendor Mesa。因此不是隐藏的 NVIDIA EGL 渲染结果。

探针主机 CPU 为 Intel Xeon E5-2696 v4 @ 2.20GHz；没有独占整机，不能把结果外推到所有 CPU。协议：reach、reach-wall、pick-place、drawer-close、assembly，各2个不同初始化；224²/480²各10回合，共20回合。每回合固定500个零动作步骤，每5步渲染一次单视角，共100帧；不运行策略、不判断成功，第一帧和reset单独计时。物理接触负载不等于真实成功轨迹，结果只用于运行成本量级。

| 项目 | 224²（10回合） | 480²（10回合） |
|---|---:|---:|
| 每回合 wall 平均 / 范围 | 26.866 s / 25.223–29.472 | 31.293 s / 29.041–32.898 |
| 每回合 process CPU 平均 | 28.712 CPU-s | 33.229 CPU-s |
| 500步物理合计平均 | 0.759 s | 0.733 s |
| 100帧渲染合计平均 | 26.106 s | 30.558 s |
| 每帧渲染平均 | 261.1 ms | 305.6 ms |
| reset平均 | 28.0 ms | 31.1 ms |
| 暖 env constructor平均 | 39.8 ms | 60.7 ms |

MT1任务集首次构建约0.91–1.14秒，单环境应复用该对象。1000个同长度回合串行仅环境约 **7.46 / 8.69 小时 wall**；16个完美并行 worker 的算术下界约28/33分钟，但**本次没有测并行缩放、共享服务器争用或内存上限**。现有日志6-worker约8–13ep/min含GPU策略/EGL等，不是CPU-only结果，不能替代本探针。

因此 MetaWorld 可以以 CPU worker + 单策略GPU运行；渲染仍可能成为瓶颈。若守卫需更频繁看图，要按实际看图频率增加预算。这里500步/100帧不等于每个模型的原生终止长度，也不包含JEPA专家目标生成成本。

**版本选择不能一句话解决。** π0.5/JEPA 使用现代 MuJoCo/v3；PAD release 基于v2。建议先分别恢复原版基线，在文中明确协议版本，所有加速臂与自己的纯推理臂成对比较。若要求统一同一环境版本、同一任务池横向比较，需要先实证相机、动力学、初始状态和成功判据兼容；本次没有做该验证，也不承诺靠简单改任务后缀即可完成。主任务覆盖可报告π0.5 MT50、PAD48和JEPA2，各自分母明确；**不能把JEPA2宣称为完整第二benchmark覆盖。**

## 7. 排除项：逐项一行理由

| 候选 | 不选的主要理由 / 原始依据 |
|---|---|
| ACT | 一次性小动作网络，owner已明确拒绝；不靠低价头证明缓存价值。 |
| 新增 π0 / OpenVLA / X-VLA / GR00T / SmolVLA | 违反新增非VLA约束。 |
| CNN Diffusion Policy ε/x₀、Consistency Policy | 已有日志说明强减步后约19–34ms；不是以默认100步昂贵为由再选旧小模型。 |
| X-WAM | 5B Wan主干有盈利潜力，但[真实文件列表](sources/xwam_hf.json)只有预训练、RoboCasa、RoboTwin，未核验LIBERO/MetaWorld权重。 |
| Cosmos3 / DreamZero | 更新不等于适配：官方/已查社区权重集中DROID、真实机器人或部分LIBERO；未核验MetaWorld，不用自训补。 |
| WorldScape Policy 2.0 | [官方发布](https://github.com/manifoldai-research/WorldScape-Policy)主要是RoboTwin2.0，所需sim不符合本次轻量限制。 |
| VPP | [官方README](https://github.com/roboterax/video-prediction-policy)公开权重为SVD-CALVIN及DP-CALVIN；没有核验到MetaWorld动作模型，不能把论文实验当权重发布。 |
| S-VAM（2026） | [官方发布](https://github.com/Haodong-Yan/S-VAM)可部署链为CALVIN，许可证写TBD；原生昂贵预测视觉特征还可能落在命中必付路径。 |
| GE-Act / GE-Act2 | GE-Act公开可核验路径为CALVIN；GE-Act2更新但已查[项目页](https://ge-act-v2.github.io/)未核验完整公开权重，不能凭演示准入。 |
| Seer-Large | [官方仓库](https://github.com/InternRobotics/Seer)有CALVIN权重（Drive实际两份各约3.98GB），未核验MetaWorld；高成本MAE历史图像编码不能全算为命中节省。 |
| Moto | [官方实现](https://github.com/TencentARC/Moto)公开路线为CALVIN/SIMPLER，未核验MetaWorld；动作预测不等于测试时必有昂贵逐token生成。 |
| DINO-WM | 同类较旧，已有更新JEPA-WM；两者同时入选不能满足计算家族多样性。 |
| LeWorldModel | [官方检查点](https://github.com/lucas-maes/le-wm#pretrained-checkpoints)确实存在，但四个发布环境无MetaWorld；不拿Reacher误作MetaWorld reach。 |
| TD-MPC2 | [官方模型仓库](https://huggingface.co/nicklashansen/tdmpc2)虽有MT50，但公开通用控制模型主要用状态输入，不能直接提供本系统所需原生视觉检索键。 |
| ScaleDP / ManiFlow | ScaleDP未核验到MetaWorld发布权重；[ManiFlow官方代码](https://github.com/WR-Maox/ManiFlow)给训练路径，未核验可直接评估的目标权重，且1–2步应重新评估盈利。 |
| Canonical（2026） | [代码README](https://github.com/agentic-learning-ai-lab/canonical)声称有双benchmark权重，但所指HF API本次返回401；文件未核验，不能列为可用检查点。 |
| Adapt2Act（ICLR2025） | 权重已验证完整；但[论文表3](https://arxiv.org/html/2504.15369)直接视觉规划9任务最高30.4%，另行RL训练分支的68.3%不能借来，单步重规划也非常慢。 |
| Spatial Policy（2025） | [论文](https://arxiv.org/abs/2508.15874)和[完整文件列表](sources/sp_hf.json)属实，但主要方法依赖在线GPT-4o空间规划/检查，不符合本次禁止远程依赖。 |
| RoboCasa365 | owner已拒绝布局/资产变化；不以固定少数seed偷偷换成容易缓存的非原协议。 |
| RoboCasa-2024 | 有Cosmos/X-WAM权重，但两者同属视频扩散，且π0.5已有365权重不能直接冒充24版支持；不作为本次统一替代。 |
| RoboTwin2.0 / BEHAVIOR / GPU渲染sim | owner明确排除，其GPU渲染、资产和并发成本与本任务不符。 |
| CALVIN | PyBullet且缓存友好；RLinf π0.5权重真实存在，但无法接纳本次JEPA/PAD权重，切换会重新引入模型选择/训练问题；不列为一键备用方案。 |

Adapt2Act额外核验：公开 [Google Drive checkpoints.zip](https://drive.google.com/file/d/1bDoQq_z605cX6czWGKVRLnSgXjR050A8/view)总长11,894,109,015 bytes；HTTP206取最后131,072 bytes并解析ZIP64中央目录，实际含 `animatediff_finetuned/mw_finetuned.ckpt`5,107,230,968、`in_domain/mw/model-24.pt`1,675,231,182、`inv_dyn.ckpt`1,029,848,423 bytes及LoRA。见 [中央目录证据](sources/adapt2act_zip_listing.json)。排除它是控制效果和复现收益的判断，**不是误称没放权重**。Spatial Policy同样有真实视频/动作文件；论文把4090显存写成80GB/40GB的硬件描述也存在矛盾，不采用其延迟数字外推GPU盈利。

## 8. 可执行的准入标准与本次未验证事项

1. 先恢复原生相机、目标条件、动作单位、成功判据；用公开权重测少量成对基线。PAD的75-DDIM/50-respaced-DDPM差异和v2/v3兼容必须先解决。
2. 对优化后的强基线测 E、D、检索/修正R、guard开销与显存；JEPA关闭可视化decoder，PAD关闭未来图像展示，历史/文本能合法预计算的两组都预计算。与最强有效少步/少搜索预算一起比较。
3. 建议准入门槛为 **full:hit≥2×且净省≥50ms/命中**，另核对实际命中率是否超过盈亏线、成功率置信区间是否保持。此门槛是研究选择建议，不是用户原文数值，也不是已完成检查。
4. JEPA目标感知检索和PAD目标位姿控制器是必要适配；先测纯缓存，再加守卫/修正/周期/按任务旋钮，不以改动输入协议制造命中率。

本次**没有验证**：两个主候选的GPU原生视觉时间、优化后全调用/命中实测比、缓存成功率、JEPA全MT50能力、PAD v3兼容/论文SR恢复、所有新模型LIBERO权重、π0.5 MetaWorld权重明确许可证、16-worker软件渲染吞吐。已验证的是公开文件存在、配置/架构、成本计算依据和CPU仿真量级。

结论的强度到此为止：**MetaWorld可选，两个不同计算家族的免训练候选可明确列出；没有证据把它们升级为“所有条件已通过”的主实验承诺。**
