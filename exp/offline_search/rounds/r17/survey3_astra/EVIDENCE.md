核验日期：2026-10-08（Chicago；抓取时 UTC 为 2026-10-09）。只做公开网页、仓库源码、权重目录及指定本地日志的 CPU 阅读；没有下载模型权重、运行 GPU、训练或启动服务。源码快照/Hub API 返回保存在 [sources/](sources/)，计算在 [cost_estimates.json](cost_estimates.json)。未读其他 r17 调研/提示目录。

| 证据类别 | 结论与定位 |
|---|---|
| 首选的发布身份 | 上游 [Fast-WAM](https://github.com/yuantianyuan01/FastWAM) 发布 `.pt`；Hugging Face **LeRobot 主仓集成**提供冻结专家和跨 embodiment 加载；LeRobot 格式权重由 ZibinDong 发布。不能把上游脚本的训练能力与 LeRobot 新接口混为一谈。 |
| 非 VLA 的口径 | Fast-WAM 的视觉/决策主干是 Wan2.2 视频 DiT＋动作 DiT；Light-WAM 是 Wan2.1 视频 DiT＋StateFusion。二者都有**冻结的 UMT5 文本编码器**，用于 task embedding；没有把图像交给 LLM/VLM 主干做动作决策。约6B/2B的策略规模不含这套文本编码器。若“无 LLM”被理解为连独立 T5 编码器也禁止，则两者均须排除。 |
| LIBERO 分数 | Fast-WAM 原论文 [Table 1](https://arxiv.org/html/2603.16666v1)：S/O/G/L＝98.2/100/97/95.2，平均97.6%。选用的 LeRobot 权重对应[集成文档](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/docs/source/fastwam.mdx#L149)：97.6/99/95/94，平均**96.4%**，每套500局，文档以float32/H20运行；不是本次重测，BF16/compiled不保证同分，也不与原论文分数混报。 |
| Light-WAM 分数 | [论文 Table 1](https://arxiv.org/html/2606.08242v1#S4.SS2)：S/O/G/L＝98.2/99.6/97.8/93，平均97.2%。[发布源码中的 spatial 日志](https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/eval_logs/summary_libero_spatial.json)明确 step055000、491/500；论文附录所写 spatial60k 与该发布文件不完全一致，应按实际发布文件评估。 |

| 已实际列出的文件（仅元数据，没有下载权重） | 字节数 | 目录/API 与本地快照 |
|---|---:|---|
| Fast-WAM `libero_uncond_2cam224.pt` | 12,041,735,140 | [yuanty/fastwam 目录](https://huggingface.co/yuanty/fastwam/tree/main)；[API快照](sources/yuanty__fastwam_hf_tree.json) |
| Fast-WAM `libero_optional_idm_2cam224.pt` | 12,041,735,545 | 同上；**本推荐不用该 IDM 分支** |
| LeRobot `model.safetensors` | 12,041,631,550 | [ZibinDong 目录](https://huggingface.co/ZibinDong/fastwam_libero_uncond_2cam224/tree/main)；[API快照](sources/ZibinDong__fastwam_libero_uncond_2cam224_hf_tree.json) |
| LeRobot `Wan2.2_VAE.safetensors`；`models_t5_umt5-xxl-enc-bf16.safetensors` | 2,818,778,392；11,361,845,464 | 同目录；基座配套资产另外占磁盘/内存，不能只算6B策略文件 |
| Light-WAM `lightwam_libero_spatial/checkpoints/weights/step_055000.pt` | 3,720,363,717 | [Light-WAM 目录](https://huggingface.co/l1ziang/lightwam-checkpoints/tree/main)；[API快照](sources/l1ziang__lightwam-checkpoints_hf_tree.json) |
| Light-WAM `lightwam_libero_object/checkpoints/weights/step_012500.pt` | 3,720,363,717 | 同上 |
| Light-WAM `lightwam_libero_goal/checkpoints/weights/step_060000.pt` | 3,720,363,717 | 同上 |
| Light-WAM `lightwam_libero_10/checkpoints/weights/step_080000.pt` | 3,720,363,717 | 同上；四套各自已有权重，无需再训 LIBERO |
| UVA `libero10.ckpt` | 未下载核大小 | [Google Drive 文件](https://drive.google.com/file/d/11c2VrmaRp48yw__5A5xpcu8EPzkexHSi/view)；页面标题确认文件名，[快照](sources/uva_gdrive_listing.html) |
| UWM `libero_90.pt` | 4,044,929,076（Drive 元数据） | [官方链接的 Drive 目录](https://drive.google.com/drive/folders/1M4AuVLMRpSwOf_YAp56bV9AqyZI9ul6g)；[快照](sources/uwm_gdrive_listing.html)；同时有 droid_pretrained/cotrained |
| Seer `33.pth`、`36.pth`、`38.pth` | 未下载核大小 | 官方根目录→Simulation→LIBERO_LONG→[Seer 文件夹](https://drive.google.com/drive/folders/1zwqGvKKtjyuWdDaNSLVGJprJMPoSqAPk)；[快照](sources/seer_libero_doc_listing.html)；另有 finetune 评估日志子目录 |
| MoDE `model_cleaned.safetensors` | 3,317,019,856 | [MoDE_LIBERO_10](https://huggingface.co/mbreuss/MoDE_LIBERO_10/tree/main)；[快照](sources/mbreuss__MoDE_LIBERO_10_hf_tree.json) |
| RIFT `libero/rift/rift_step021700.pt` | 12,060,942,777 | [RIFT 目录](https://huggingface.co/PoopBear/RIFT/tree/main/libero/rift)；[快照](sources/PoopBear__RIFT_hf_tree.json) |
| Octo 社区 `30000/default/checkpoint` | 540,692,595 | [akuramshin/octo-libero-100-finetuned](https://huggingface.co/akuramshin/octo-libero-100-finetuned/tree/main)；[快照](sources/akuramshin__octo-libero-100-finetuned_hf_tree.json)；文件存在，但没有找到对应可核对 SR |

| Fast-WAM 无状态核验 | 精确代码位置与解释 |
|---|---|
| 输入只有当前帧 | [modular.py:1580](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/wan/modular.py#L1580)要求 `[1,3,H,W]` 或 `[3,H,W]`；当前 proprio 为 `[D]`/`[1,D]`；[infer_action:1788](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/wan/modular.py#L1788)没有历史或外部 goal-image 参数。 |
| 无跨调用 KV | 同文件1812新建动作噪声，1815编码本次图像，1825准备本次 video token，**1860局部赋值 `video_kv_cache`**，1885开始动作去噪；只在本次动作多步内部复用。`_compiled_*` 保存编译函数，不保存上一帧内容。 |
| 随机性与 VAE | [config:217](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/configuration_fastwam.py#L217)默认 `inference_seed=42`；[noise:1657](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/wan/modular.py#L1657)每次创建并播种局部 Generator；[VAE adapter:76](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/wan/adapters.py#L76)使用 posterior mode。参考 [diffusers v0.35.1 VAE:1145](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L1145)在每次 encode 前后 clear_cache。 |
| 应用入口边界 | 用 [predict_action_chunk:209](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/modeling_fastwam.py#L209)/core `infer_action`，每次请求重算。**不要把 `select_action:240` 当无状态 API**：它有 `_action_queue`，后续调用会返回旧 chunk 中的动作。队列可留在执行侧，策略请求必须按 chunk 定义。 |
| 训练视频不是推理历史 | [config:413](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/configuration_fastwam.py#L413)的视频采样索引为 `[0,4,…,32]`，全是当前/未来监督；[build_inputs](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/wan/modular.py#L1109)需要未来视频目标，不代表推理需要历史或未来图像。 |
| 检索键建议 | 在本次 video-prefill 后池化某层 K/V 或暴露等价视觉 token；按 task 分区并结合当前 proprio 检查。池化/检索尚未实现或计时；**不是跨调用复用模型 KV**。固定 seed，基座调用本身无跨调用状态；检索系统的外部数据库另论。 |

| Fast-WAM 微调代码核验 | 精确代码位置与适配工作 |
|---|---|
| 官方部分微调 | [modeling:80–90](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/modeling_fastwam.py#L80)设置 `video_expert.requires_grad_(False)`，并冻结重挂到 MoT 下的各 video block；[142–152](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/modeling_fastwam.py#L142)只把 `requires_grad` 参数交给优化器。约5B视频专家冻结，约1B动作专家＋proprio可训。VAE/T5在[modular:876附近](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/wan/modular.py#L860)被冻结。 |
| 上游能力不混报 | [上游 trainer](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/trainer.py#L304)的标准逻辑是冻结其他组件后启用整个 `dit`，包含视频和动作专家；不能据此宣称上游已提供动作专家单独训练。已核实的开关来自 LeRobot；没有拿未经核实的 LoRA 配方充数。 |
| `lambda_video=0` 的真实语义 | [config说明:185](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/configuration_fastwam.py#L185)建议冻结时置零；但[training_loss:1400–1416](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/wan/modular.py#L1400)仍先跑 `_run_training_mot`、算视频损失，最后才乘0。**它不裁掉未来视频前向/VAE，也不保证裁掉反传**；proprio 投影仍经上下文进入视频分支。预算没有按“仅动作头计算”计算。 |
| 跨 embodiment 加载 | [modeling:94–140](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/modeling_fastwam.py#L94)在 `strict=False` 时只丢弃形状不匹配项，保留其余预训练权重；[pretrained.py:168](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/pretrained.py#L168)默认 strict=False。action/proprio改为4时对应输入/输出投影重置；这不是从头训练6B。 |
| 数据契约 | [MT50 meta/info.json](https://huggingface.co/datasets/lerobot/metaworld_mt50/raw/main/meta/info.json)：v3.0，2500 episodes，204806 frames，50 tasks，80 fps；**`observation.image`** 是480² image，state/action各4；另有39维environment_state，本方案不用。快照 [metaworld_info.json](sources/metaworld_info.json)。 |
| 相机与维度 | `rename_map={"observation.image":"observation.images.image"}`；`image_size=[224,224]`、action_dim=4、proprio_dim=4、horizon=32。保留一台真实相机；不复制虚构 wrist。 [factory:300–334](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/factory.py#L300)先重命名元数据，再调用[配置钩子:316](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/configuration_fastwam.py#L316)重建视觉特征；预处理 resize 到目标大小。 |
| 归一化/动作约定 | 使用 MT50 自身统计；[processor:68](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/processor_fastwam.py#L68)的 STATE/ACTION normalizer 接受数据统计；VISUAL保持[0,1]，VAE入口再转[-1,1]。MT50设 `toggle_action_dimensions=[]`，不要直接继承LIBERO的 gripper翻转；先用数据回放确认动作方向。 |
| 训练量的计数陷阱 | [lerobot_train:784–805](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/scripts/lerobot_train.py#L784)明确每个 loop step 是一个 micro-batch。micro4、`accelerator.gradient_accumulation.steps=4` 时，20k优化器更新需要 `steps=80000`；30k需120000。有效batch16；320k/204806≈1.56，480k≈2.34个帧索引等效遍历，**不是已证明足够的收敛轮数**。 |
| 文本/显存 | 50个task可预编码固定文本；[batch转换:157](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/modeling_fastwam.py#L157)接受 `context/context_mask`，core亦接受它们。训练不应重复把同50段文字当额外学习目标。约6.73B策略＋VAE的BF16权重约13.5GB，约1B可训的梯度/Adam/可选FP32主副本另约10–14GB；若文本编码器常驻再加约11GB，**这是静态粗估，不含激活**。H10080＋checkpointing＋micro4合理；不能据此保证4090训练不OOM。 |

| 预算依据与不确定性 | 数值/推导 |
|---|---|
| 已发表训练吞吐 | [Light-WAM Table 3](https://arxiv.org/html/2606.08242v1#S4.SS4)：4×H100、global batch64；Fast-WAM全训31.6 sample/s、70.7GiB/卡；Light-WAM在线VAE消融35.7 sample/s、48.6GiB/卡（micro8×累积2）；缓存＋2×下采样133.3 sample/s、43.1GiB/卡。均非MT50、非本次单卡测量。 |
| 首选小时 | 把31.6÷4＝7.9 sample/s仅作为每卡参考，规划5–8 sample/s；320k窗口需11.1–17.8h，另留1–2h编译/小评测/保存，取约13–20H100 h。30k更新×16需16.7–26.7h＋预留≈18–29h。单卡效率、冻结后反传路径、CPU读图都可能改变该范围。 |
| 备选小时 | 为避免把未知视频预计算耗时写成已知，采用**在线VAE预算**：35.7÷4＝8.925 sample/s参考，按6–9规划；480k窗口14.8–22.2h＋1–2h≈16–24H100 h。单相机分辨率更小可能受益，但未把它算成保证。缓存版纯训练线性参考约4h，离线预计算另计。 |
| 独立实训旁证 | [so101-simstudio 的 FastWAM 实验记录](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md#fastwam-training-lab01-pnp-mi300x--dorobot)：冻结视频专家，MI300X bs4，4k→20k剩余16k步耗4h19m，约4.12 sample/s。**硬件/任务不同，不能替代H100 MT50测量**；其20k闭环仍0%，说明“代码支持微调”不等于跨 embodiment 一定成功。 |
| 风险收口 | 不承诺20k/30k会达到某个MT50 SR。若实际吞吐低于5 sample/s或不能学到任务，Fast-WAM的②不再按通过处理；不擅自延长到多卡、多天。 |

| 缓存证据与口径 | 内容 |
|---|---|
| 已发表同硬件分项 | [Light-WAM Table 4](https://arxiv.org/html/2606.08242v1#S4.SS4)：RTX4090 **48GB**，RoboTwin输入，文本已缓存，排除simulator/I/O。Fast-WAM：VAE11.3、visual36.0、action356.8、overall404.62ms；Light-WAM：12.7/56.5/2.1、overall72.03ms。该表没有证明使用最新编译实现；不能当作MT50单相机测量。 |
| LIBERO 独立分项 | [Fast-WAM issue34](https://github.com/yuantianyuan01/FastWAM/issues/34)：报告者H10080、两相机224×448、horizon32、NFE10，CUDA Events，3warmups/10runs，无compile。T5 18.4、proprio0.1、VAE5.5、prefill30.3、action310ms，总364.3ms。去掉固定T5为345.9ms；这是用户测量，非作者H100复现。 |
| 190ms / 110ms 不混用 | [作者答复](https://github.com/yuantianyuan01/FastWAM/issues/34#issuecomment-4250694653)：论文190ms是RTX5090D V2真实机器人384×320、含T5；不是4090 LIBERO。最新[上游README](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/README.md#L20)另报4090 190→110ms（编译等优化），没有可用于我们检索键的分项，故保留未知。 |
| hit算式 | 取命中时仍需完成的键提取阶段，所有未分摊余量也保留到hit端：Fast4090 `hit=404.62−356.8=47.82ms`；full/hit=8.461。Light深层键 `72.03−2.1=69.93`，仅1.030×；浅VAE键 `72.03−56.5−2.1=13.43`，5.363×。另加未知δ后比值只会下降；命中省时分别是动作/后续分支耗时减δ。 |
| π0.5 对照 | 用户给定68ms/6.6×，hit≈10.30ms、每hit省57.70ms；不是同批实测。Light浅键理论省58.6ms与它相近，但浅VAE特征的检索辨识力没有闭环证据。 |
| 低NFE直接证据 | [issue73](https://github.com/yuantianyuan01/FastWAM/issues/73)对同发布权重报告NFE1/2/5/10总体SR＝95.75/96.35/96.85/96.95%；[作者回复](https://github.com/yuantianyuan01/FastWAM/issues/73#issuecomment-5002503446)认可LIBERO类似现象，同时报告RoboTwin NFE1明显下降。故MT50是否可减步未知。 |
| 敏感性而非测量 | 若粗略把404.62ms中的356.8ms动作项除10，NFE1总83.50ms、深层键hit47.82、比值1.75、省35.68ms；此时后半段已不占多数。**不能用该线性估算预测compiled性能或SR，也不能用10步代替最佳可接受基线。** |

| Light-WAM 备选的代码审计 | 证据 |
|---|---|
| 官方支持部分微调 | [lightwam.py:364–415](https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/src/lightwam/models/wan22/lightwam.py#L364)先冻结全部，再开启backbone LoRA、adapter、video head、StateFusionActionExpert、proprio；[model config](https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/configs/model/lightwam.yaml)含freeze_backbone、LoRA-r64。训练量约0.44B；不是需要自行实现PEFT。 |
| 无状态 | [infer_action:2263–2355](https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/src/lightwam/models/wan22/lightwam.py#L2263)只收当前单图、当前proprio和prompt/context；StateFusion分支不迭代视频、不依赖输入goal image、没有跨调用KV/观察队列。它的训练未来监督不能被误读为推理goal要求。 |
| 数据便利性的限制 | [内置 loader:80](https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/src/lightwam/datasets/lerobot/lerobot/lerobot_dataset.py#L80)标记v2.1；仓库scripts对LIBERO/RoboTwin专用，不能原样吃MT50 v3目录。需要一个dataset/config适配或v3→v2.1转换；从已有LIBERO权重初始化时，4-D动作/状态相关shape mismatch也要选择性处理。没有宣称它已有MT50一条命令。 |
| 在线/离线 VAE | [build_inputs:1072](https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/src/lightwam/models/wan22/lightwam.py#L1072)接受video或video_latents；[预计算脚本](https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/scripts/precompute_video_latents.py#L885)逐样本窗口编码，不能把全部204806帧乘单帧推理延迟当作完整预计算成本。 |
| 为什么只是备选 | 已有深层视觉特征后的动作头仅2.1ms。只有把键前移到当前帧VAE潜变量，才能跳过大部分56.5ms视频主干；这需要验证检索质量，不能直接套用成熟语义特征库的效果。 |

| 其他候选：决定性证据 | 结果 |
|---|---|
| UVA | [论文表II](https://arxiv.org/html/2503.00200v1#S5)：Long93%，230ms；[libero10.yaml:40](https://github.com/ShuangLI59/unified_video_action/blob/fe3dec81af565fbb2d686072722f12340be31293/unified_video_action/config/task/libero10.yaml#L40) `n_obs_steps=16`；[predict_action:221](https://github.com/ShuangLI59/unified_video_action/blob/fe3dec81af565fbb2d686072722f12340be31293/unified_video_action/policy/unified_video_action_policy.py#L221)消费时序obs，runner的MultiStepWrapper维护该历史。重训成单帧会改变已发布LIBERO模型，不能算现成满足④。 |
| UWM | [README:58](https://github.com/WEIRDLabUW/unified-world-model/blob/b7d4067c8859775e5b08f681e76d538015cdd445/README.md#L58)确有LIBERO90发布；[论文表II及IV-C](https://arxiv.org/html/2504.02792v2#S4.SS3)79±11%来自五个LIBERO10任务**各自微调**，不等于发布90预训练权重免训练就有79%。[template:23](https://github.com/WEIRDLabUW/unified-world-model/blob/b7d4067c8859775e5b08f681e76d538015cdd445/configs/template_train_robomimic.yaml#L23)obs_num_frames=2；[eval:27](https://github.com/WEIRDLabUW/unified-world-model/blob/b7d4067c8859775e5b08f681e76d538015cdd445/experiments/uwm/eval_robomimic.py#L27)把它传给obs_horizon；[sample_marginal_action:396](https://github.com/WEIRDLabUW/unified-world-model/blob/b7d4067c8859775e5b08f681e76d538015cdd445/models/uwm/uwm.py#L396)从该观测编码。 |
| Seer | [论文表I](https://arxiv.org/html/2412.15109v1#S4.T1)Long87.7%，top3检查点平均；[eval.sh:40](https://github.com/OpenRobotLab/Seer/blob/ee228e110e098beefa23f94d558dfa8492259e6e/scripts/LIBERO_LONG/Seer/eval.sh#L40)sequence_length7；[wrapper:58](https://github.com/OpenRobotLab/Seer/blob/ee228e110e098beefa23f94d558dfa8492259e6e/utils/eval_utils_libero.py#L58)建立img/state队列，[126–137](https://github.com/OpenRobotLab/Seer/blob/ee228e110e098beefa23f94d558dfa8492259e6e/utils/eval_utils_libero.py#L126)跨调用append再cat。不是无状态PIDM。 |
| MoDE | [论文](https://arxiv.org/html/2412.12953v1)Long92%，LIBERO训练4A6000×2h含评估；[已发布config](https://huggingface.co/mbreuss/MoDE_LIBERO_10/blob/main/config.yaml)obs_seq_len1。 [forward:584](https://github.com/intuitive-robots/MoDE_Diffusion_Policy/blob/72cf8a25eafa54316f0c6c961b620e4b6c9c3e6b/mode/models/mode_agent.py#L584)单帧特征→扩散；任务专家有预计算缓存，`step`还有动作队列。即使包装成stateless，仍属DP式图像条件扩散/MoE，③不符。 |
| VPP / PAD | [VPP发布表](https://github.com/roboterax/video-prediction-policy#checkpoints-)仅列SVD机器人/CALVIN和DP-CALVIN；[PAD源码](https://github.com/Robert-gyj/Prediction_with_Action)及[HF目录](https://huggingface.co/yjguo/pad_bridge_pre/tree/main)列 `bridge_pre.pt` / `bridge_pre_mw_ft.pt`，不是LIBERO。这里结论是**未核实LIBERO发布**，不是证明互联网不存在任何社区模型。 |
| Octo / MDT | Octo有社区LIBERO文件，不能写成“从未发布”；已查[config](https://huggingface.co/akuramshin/octo-libero-100-finetuned/blob/main/config.json)window_size2，未找到对应SR。MDT[官方仓库](https://github.com/intuitive-robots/mdt_policy)指向CALVIN模型；[查到的LIBERO仓](https://huggingface.co/hqfang/nov17-libero-mdt-vit-clip/tree/main)只有`.gitattributes`，见[API](sources/hqfang__nov17-libero-mdt-vit-clip_hf_tree.json)。 |
| GE-Act | [Genie-Envisioner-V1](https://github.com/AgibotTech/Genie-Envisioner-V1)README的GE-Act权重链接为CALVIN；[HF目录](https://huggingface.co/agibot-world/Genie-Envisioner/tree/main)仅基座fast/slow。 [GE-Act V2](https://ge-act-v2.github.io/)未核实可直接用的LIBERO权重＋完整训练代码；不拿演示/计划发布当通过。 |
| RIFT | [论文](https://arxiv.org/html/2608.11521v1#S5.SS2)98.8%、247.9ms/A800；[infer_action:713](https://github.com/ChushanZhang/RIFT/blob/cc5fce8d6f89210765e02c73ba692f01bc5b4b41/rift/models/wan22/rift_model.py#L713)当前图像创建本次anticipation cache，无真实未来帧输入。 [trainer:330–360](https://github.com/ChushanZhang/RIFT/blob/cc5fce8d6f89210765e02c73ba692f01bc5b4b41/rift/trainer.py#L330)resume要求完整训练state目录且启用整个dit；没查到现成冻结专家/LoRA微调入口，不能直接照搬Fast-WAM LeRobot的开关。 |
| 额外扫查 | [FasterWAM](https://github.com/hustvl/FasterWAM)与[HF](https://huggingface.co/hustvl/FasterWAM/tree/main/libero)有11,117,679,929-byte `step_021700.pt`，但8卡全训示例、未核实本需求的部分微调预算，未提升为fallback。[LiLa-WAM](https://github.com/teee000/LiLa-WAM)查到RoboTwin发布，未核实LIBERO。未将“没继续核验”写作某项硬件性能不可能。 |

| 本地已有系统证据 | 本次使用方式 |
|---|---|
| [R16 REPORT](../../r16/REPORT.md) | π0.5/GR00T的1–2步已接近满步SR；缓存视觉系数分别.152/.148。必须将减步基线纳入新模型缓存判断。 |
| [cosmos_nfe_plan.log.md](../../../../../logs/cosmos_nfe_plan.log.md) | Cosmos保留LIBERO特殊案例；减步可显著压缩生成成本，不能把它当新MT50模型。 |
| [xwam_dp_nfe_plan.log.md](../../../../../logs/xwam_dp_nfe_plan.log.md) | XWAM减步结果只作风险提示；DP早期“低步全失败”后来定位到旧diffusers调度bug并修正，不能引用旧结论证明DP不可减步。 |

代码链接固定到本次读取的commit；Hub/Drive目录链接随发布方更新，审计以本地JSON/HTML快照为准。训练小时是明确假设下的算术估算，缓存hit是**每次命中成本**而非整体吞吐保证；还缺命中率、闭环成功率和本机编译后测量。
