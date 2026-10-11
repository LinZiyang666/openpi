| 证据账本 | 口径 |
| --- | --- |
| 日期 / 独立性 | 2026-10-08；仅本输出目录写入。未读其他 survey、prompts、ops 目录；参考系统背景仅 r16/REPORT.md。 |
| 最终结论 | 0 个完成全部认证；DP / brandonyang 为条件候选。见 DECISION.md。 |
| 查询完整度 | 六个指定查询按 HTTP Link rel=next 翻至结束，去重 5014 仓库；不是只取 LIBERO 前1000条。 |
| 范围限制 | 公开可访问、可索引的来源快照；私有/未索引/异名或失效链接可能漏检。缺证不等于不存在。 |
| 安全下载 | 不下载完整策略权重；DP 仅读两份3.6KB metadata和两个128KiB tensor header；BAKU ZIP 仅读64KiB尾部。 |
| 测量限制 | CPU 随机权重结构代理，不是 checkpoint 成功率复现，也不能外推 GPU 毫秒数。 |

**HF 枚举：实际调用及翻页。** [脚本](enumerate_hf.py)、[去重目录（5014行）](HF_CATALOG.tsv)、[MetaWorld 名称筛选及家族归并](METAWORLD_AUDIT.tsv)。

| query | 返回条数 | 页数 | 各页 | API | 本地快照 |
| --- | --- | --- | --- | --- | --- |
| `metaworld` | 103 | 1 | 103 | [首请求](https://huggingface.co/api/models?search=metaworld&limit=1000&full=true) | [全部响应/游标](raw/hf_search_metaworld.json) |
| `meta-world` | 1317 | 2 | 1000+317 | [首请求](https://huggingface.co/api/models?search=meta-world&limit=1000&full=true) | [全部响应/游标](raw/hf_search_meta-world.json) |
| `meta_world` | 1317 | 2 | 1000+317 | [首请求](https://huggingface.co/api/models?search=meta_world&limit=1000&full=true) | [全部响应/游标](raw/hf_search_meta_world.json) |
| `mt50` | 15 | 1 | 15 | [首请求](https://huggingface.co/api/models?search=mt50&limit=1000&full=true) | [全部响应/游标](raw/hf_search_mt50.json) |
| `mt10` | 27 | 1 | 27 | [首请求](https://huggingface.co/api/models?search=mt10&limit=1000&full=true) | [全部响应/游标](raw/hf_search_mt10.json) |
| `libero` | 3667 | 4 | 1000+1000+1000+667 | [首请求](https://huggingface.co/api/models?search=libero&limit=1000&full=true) | [全部响应/游标](raw/hf_search_libero.json) |

`meta-world` / `meta_world` 搜索是宽松词匹配，大量结果是 MetaCLIP worldwide、文本模型或其他噪声。以 `metaworld|meta[-_]world|mt50|mt10` 名称筛选并剔除明显非机器人命中后保留 **110** 个名称线索；这个数仍不是合格模型数。每个线索的文件是否单独核查，已在 TSV 标明。HF 数据集仓库不算 model weights。

| 双边名称共同作者（16） | 家族归并 / 后续核查 |
| --- | --- |
| AIRLab-POLIMI | CRT；是真文件，但不是 action policy。 |
| brandonyang | DP 是核心候选；另有 π0-FAST/openpi，同作者不等于同架构。 |
| iFaz | DP；MW 空仓。 |
| kunhsiang | ACT / inverse / feedback ACT；抽查完整双边权重。 |
| MINT-SJTU | Evo-1、EVO-Depth、Evo-World、StarVLA；均需按 VLM 骨干排除。 |
| RLinf | π / OpenVLA；DreamZero/Wan/OpenSora 的 LIBERO 文件不自动建立 MW 交集。 |
| Mark-ZJTang | ALAM+π0 双边；不能把 action tokenizer 当完整非 VLA。 |
| lerobot | SmolVLA 双边；FastWAM/lingbot/其他 LIBERO 文件未建立 MW 交集。 |
| OpenRAL | rskill 封装；双边 SmolVLA 仍属同家族。 |
| JunnDooChoi | π0.5/SmolVLA 的 UFOS/MLP 等 head 替换，骨干仍 VLA。 |
| jadechoghari | SmolVLA 双边；X-VLA、tokenizer 不能和 SmolVLA 混拼。 |
| msv6 / nikriz / tiantianx | SmolVLA 或 π 派生，合并而不重复计推荐。 |
| zhongweihong | X-VLA 单仓双子目录，完整文件已核查。 |
| Rhx11111 | MW-256 空仓，LIBERO 名称也不构成同家族完整发布。 |

除作者交集外，跨作者也按家族匹配：官方 Cosmos ↔ mmurray-ms；官方 GR00T ↔ namenu；BAKU/RT-1/ACT 在同一 OSF 压缩包。补充 HF query `baku, quest, cocos, canonical, vpp, GEVRM, DiffuView, DynaMo, flowpolicy, diffusion_metaworld` 的原始结果为 `raw/hf_extra_*.json`；这些是补充搜索，**不宣称都完整翻页**（如 quest 命中1000条有截断）。

**公开网页检索及论文交叉。** 使用组合词 `"LIBERO" "MetaWorld" "diffusion" "checkpoints"`、`"LIBERO" "MetaWorld" "world model" weights`、`"LIBERO" "MetaWorld" "2026" "policy" non VLA`；对命中方法追踪官方论文→代码→model/Drive/OSF链接→文件清单。另查询 `VQ BeT pretrained LIBERO MetaWorld checkpoints github`、`PRISE pretrained checkpoints LIBERO MetaWorld github`、`SUREFlow checkpoint github`。依据以论文/作者仓库/HF 原始文件为准，不以聚合站摘要定论。

<a id="dp"></a>

**DP / brandonyang：实际文件、配置和推理链。**

[brandonyang/dp-libero-checkpoints](https://huggingface.co/brandonyang/dp-libero-checkpoints/tree/main) · created `2026-04-22T22:47:39.000Z` · revision `3c48ffaab98faacb119beaf2ee12f943503e6492` · gated `False` · card license `未声明` · [API 快照](raw/hf/brandonyang__dp-libero-checkpoints/metadata.json)


| 文件（完整仓库 listing） | bytes |
| --- | --- |
| `.gitattributes` | 1519 |
| `dp_libero_lang_v1/30000/assets/physical-intelligence/libero/norm_stats.json` | 1955 |
| `dp_libero_lang_v1/30000/metadata.pt` | 3571 |
| `dp_libero_lang_v1/30000/model.safetensors` | 367444556 |
| `dp_libero_lang_v1/30000/optimizer.pt` | 735078521 |

[brandonyang/dp-metaworld-checkpoints](https://huggingface.co/brandonyang/dp-metaworld-checkpoints/tree/main) · created `2026-04-22T22:46:50.000Z` · revision `46c26c0630024f7645f19398cc5be105870ca96f` · gated `False` · card license `未声明` · [API 快照](raw/hf/brandonyang__dp-metaworld-checkpoints/metadata.json)


| 文件（完整仓库 listing） | bytes |
| --- | --- |
| `.gitattributes` | 1519 |
| `dp_metaworld_lang_v1/30000/assets/brandonyang/metaworld_ml45/norm_stats.json` | 1135 |
| `dp_metaworld_lang_v1/30000/metadata.pt` | 3571 |
| `dp_metaworld_lang_v1/30000/model.safetensors` | 367423896 |
| `dp_metaworld_lang_v1/30000/optimizer.pt` | 735037561 |
| `dp_metaworld_lang_v1/85000/assets/brandonyang/metaworld_ml45/norm_stats.json` | 1135 |
| `dp_metaworld_lang_v1/85000/metadata.pt` | 3571 |
| `dp_metaworld_lang_v1/85000/model.safetensors` | 367423896 |
| `dp_metaworld_lang_v1/85000/optimizer.pt` | 735037561 |

| 配置项目 | LIBERO 30k | MetaWorld 85k |
| --- | --- | --- |
| action_dim | 7 | 4 |
| action_horizon | 16 | 16 |
| n_obs_steps | 1 | 1 |
| n_action_steps | 16 | 16 |
| horizon | 16 | 16 |
| lang_emb_dim | 768 | 768 |
| num_inference_steps | 100 | 100 |
| num_train_timesteps | 100 | 100 |
| beta_schedule | squaredcos_cap_v2 | squaredcos_cap_v2 |
| prediction_type | epsilon | epsilon |
| vision_backbone | resnet18 | resnet18 |
| crop_shape | [84, 84] | [84, 84] |
| obs_encoder_group_norm | True | True |
| eval_fixed_crop | True | True |
| n_layer | 12 | 12 |
| n_cond_layers | 4 | 4 |
| n_head | 8 | 8 |
| n_emb | 512 | 512 |
| causal_attn | True | True |
| time_as_cond | True | True |
| obs_as_cond | True | True |

配置证据：[LIBERO 安全解析 JSON](raw/brandon_libero_config.json)、[MW 安全解析 JSON](raw/brandon_metaworld_config.json)、[解析脚本](probe_metadata.py)。`metadata.pt` 内不是只有训练步数，而是完整 model/data/optimizer 配置。用 `pickletools` 看结构后，仅允许 `flax.nnx.filterlib.Nothing` 映射为本地占位；未调用远端 checkpoint 中任意 Python 类、未加载模型参数。保存的 pickle disassembly 可复查。

权重结构证据：[LIBERO safetensors header](raw/brandon_libero_safetensors_header.json)、[MW header](raw/brandon_metaworld_safetensors_header.json)。HTTP Range 仅读前131072 bytes；模型的 `_policy.model.*` 全部 tensor 名称/形状与上游 `TransformerForDiffusion` 一致，代理结果 `model_shape_mismatches={}`。两侧视觉都是 language-FiLM ResNet18，末端 spatial-softmax 32 keypoints 与64维线性层；2×64可作自己的视觉key。语言768维另由外部提供，发布目录没有 VLM/LLM 权重，也没说明具体文本 embedder。

| 代码审查 | 证据 / 限制 |
| --- | --- |
| 每次采样重新初始化 | [上游 policy](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py)：conditional_sample 创建 randn、set_timesteps，并在本次调用内完成 denoising。 |
| 观测窗口 | 同文件 predict_action 按 n_obs_steps 切取输入，assert 无 past_action；发布配置为1。随机采样本身不算观测历史。 |
| key 提取点 | obs_encoder 后、model denoiser 前；[视觉编码器上游](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/vision/multi_image_obs_encoder.py)。checkpoint 多出 FiLM 自定义层，因此视觉代理是重建，不宣称逐算子相同。 |
| 尚缺原样推理 | HF作者主页关联 branyang02；其公开 openpi-eval/main及分支、openpi-trajectory、openpi-cuda、ricl_openpi_latent_action、vvla 的树/README未定位这个 DP wrapper。GitHub搜索 dp_libero_lang_v1、dp_metaworld、DiffusionTransformerHybridImagePolicy 作者范围也未找到。 |
| 复现缺口 | MW 为什么仍有 left_wrist_0_rgb、language embedding生成、裁剪/动作变换和 suite覆盖需专用代码或作者说明。配置可当前帧，不代表任意社区wrapper一定无历史。 |
| 成功率 | 两 repo 无 README、eval metrics；所查关联代码也没给此 DP checkpoint 的 LIBERO/MW SR。不得借用原始 DP、BAKU、QueST 论文 baseline 数字。 |
| license / access | gated=false；两边无 LICENSE/model card。上游 DP MIT 不自动授权社区模型权重。 |

作者溯源快照：`raw/brandon_profile.txt`、`raw/branyang02_repos.json`、`raw/branyang02__*_branches.json`、`raw/brandon_tree_*.json`、`raw/gh_search_dp_metaworld.json`、`raw/gh_search_diffusion_policy_brandon.json`。这些是“尚未找到 wrapper”的搜索范围，不是证明代码不存在。

<a id="cost"></a>

**DP 成本复核。** [可复跑 CPU 脚本](cpu_cost_probe.py)、[全部计时样本/参数与 FLOPs](raw/cpu_cost_proxy.json)、[运行日志](raw/cpu_probe.log)。

```bash
taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python exp/offline_search/rounds/r17/survey2_astra/cpu_cost_probe.py
```

| steps | vision ms | tail ms | full ms | hit ms | full/hit | saved ms |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 20.235239 | 2578.798009 | 2599.033248 | 20.293911 | 128.069612 | 2578.739337 |
| 2 | 20.235239 | 50.351764 | 70.587003 | 20.293911 | 3.478236 | 50.293092 |
| 1 | 20.235239 | 25.144317 | 45.379556 | 20.293911 | 2.236117 | 25.085645 |

B=1 FP32；torch2.7.1、4 CPU intra-op /1 inter-op threads；无 GPU forward。随机权重，2次 warm-up；vision、1/2-step tail 各10样本；100-step tail 3样本（2527.285–2598.302ms）；lookup30样本。full 是 stage medians 相加，未作为端到端装载检查点计时。head 是上游12层 decoder+4层 condition Transformer、DDPMScheduler；视觉仅按 metadata/header 复原。100/2/1 steps 用相同参数，不测 action质量。

参数量：vision28,300,096，head63,539,207；vision FLOPs1,197,098,368，单步 head profiler统计1,469,498,625。FLOPs 采用 multiply/add 各一次的 profiler convention；fused attention、归一化等可能漏计，因此是近似/下界，不写成完整硬件 FLOPs。128维 key 对5000条cache做内积+argmax+chunk取出，median0.058672ms。没有纠错器/guard成本。一般公式：full=E+D；hit=E+L+H；saved=D−L−H；ratio=(E+D)/(E+L+H)。这解释了为何先降步到1/2后缓存收益明显收窄。

<a id="ifaz"></a>

**iFaz：名称交集不是权重交集。**

[iFaz/diffusion-libero_goal-seed3-15000](https://huggingface.co/iFaz/diffusion-libero_goal-seed3-15000/tree/main) · created `2026-05-18T05:49:22.000Z` · revision `895e12782989c939cc2fa7e9450237560f406fd3` · gated `False` · card license `apache-2.0` · [API 快照](raw/hf/iFaz__diffusion-libero_goal-seed3-15000/metadata.json)


| 文件（完整仓库 listing） | bytes |
| --- | --- |
| `.gitattributes` | 1519 |
| `README.md` | 1548 |
| `config.json` | 2282 |
| `model.safetensors` | 1111877924 |
| `policy_postprocessor.json` | 658 |
| `policy_postprocessor_step_0_unnormalizer_processor.safetensors` | 7668 |
| `policy_preprocessor.json` | 1327 |
| `policy_preprocessor_step_3_normalizer_processor.safetensors` | 7676 |
| `train_config.json` | 9375 |

[iFaz/diffusion-metaworld_medium-seed42-half](https://huggingface.co/iFaz/diffusion-metaworld_medium-seed42-half/tree/main) · created `2026-05-18T14:26:00.000Z` · revision `beb1df0d3a2ee58d211acce039304f0d516e12f5` · gated `False` · card license `未声明` · [API 快照](raw/hf/iFaz__diffusion-metaworld_medium-seed42-half/metadata.json)


| 文件（完整仓库 listing） | bytes |
| --- | --- |
| `.gitattributes` | 1519 |

<a id="baku"></a>

**BAKU / RT-1 / MT-ACT：官方 OSF 发布清单。** [官方 README](https://github.com/siddhanthaldar/BAKU)、[OSF project](https://osf.io/3x8v5/?view_only=fb8285f025e84d23a41a0eef683a7e6d)、[OSF API](https://api.osf.io/v2/nodes/3x8v5/files/osfstorage/?view_only=fb8285f025e84d23a41a0eef683a7e6d)。

`weights.zip` 总长2,561,746,368 bytes。HTTP206只取 bytes2561680832–2561746367，共65536 bytes，解析 central directory而非下载2.56GB；[OSF metadata](raw/baku_osf.json)、[Range记录](raw/baku_zip_tail_meta.json)、[ZIP完整目录](raw/baku_zip_listing.json)。以下是其中完整的策略文件列表：

| zip路径 | 解压后bytes | 压缩bytes |
| --- | --- | --- |
| weights/libero/rt1.pt | 356952850 | 321351894 |
| weights/libero/baku.pt | 105742118 | 94337880 |
| weights/libero/mtact.pt | 624545508 | 473870218 |
| weights/dmc/rt1.pt | 347691434 | 311968846 |
| weights/dmc/baku.pt | 89429178 | 78627658 |
| weights/dmc/mtact.pt | 484273942 | 398520950 |
| weights/metaworld/rt1.pt | 347463478 | 312490903 |
| weights/metaworld/baku.pt | 102334342 | 90802372 |
| weights/metaworld/mtact.pt | 624408740 | 479771513 |

| 核查点 | 结果与原始来源 |
| --- | --- |
| 配置不在 ZIP 内 | 官方 [config_eval.yaml](https://github.com/siddhanthaldar/BAKU/blob/main/baku/cfgs/config_eval.yaml)、[LIBERO suite](https://github.com/siddhanthaldar/BAKU/blob/main/baku/cfgs/suite/libero.yaml)、[MW suite](https://github.com/siddhanthaldar/BAKU/blob/main/baku/cfgs/suite/metaworld.yaml) 和 agent yaml；已保存本地。 |
| BAKU骨干/head | ResNet18-FiLM，自身视觉feature→8层/4head/256dim GPT→2层确定性MLP；不是VLM。config_eval是deterministic，num_queries10；代码支持 diffusion 选项不证明发布 baku.pt 是 diffusion。 |
| BAKU历史区别 | suite.history=false，所以 observation_buffer实际最大长度1；但 [act](https://github.com/siddhanthaldar/BAKU/blob/main/baku/agent/baku.py#L654) 写 self.all_time_actions，并对旧预测加权。R3失败来自 action temporal aggregation，不能误说默认10帧历史。 |
| RT-1 | [Instructions](https://github.com/siddhanthaldar/BAKU/blob/main/Instructions.md#L89) 双基准都设置history=true、history_len6；普通 image-token Transformer，并非LLM。 |
| 收益判断 | BAKU tail 是少数当前观测tokens的一次小GPT+MLP；视觉是128×128 ResNet。未运行测量，故不填伪造毫秒/加速比。即使消除temporal aggregation，也缺后段占优的结构。ACT按owner现有排除条件处理。 |
| paper结果 | [BAKU paper v2](https://arxiv.org/html/2406.07539v2) Table1：LIBERO90 / MW30，BAKU90/79、RT-1 16/65、MT-ACT54/13、VQ-BeT90/78（%）。VQ-BeT无ZIP权重。不是40task LIBERO平均或MT50。 |
| 数值与release映射 | 作者发布该实验权重，但ZIP未含逐checkpoint评测日志；上述是论文方法数值，不是本轮复现。 |
| license/access | 官方代码MIT；OSF匿名可读。未找到ZIP独立权重license说明。 |

<a id="canonical"></a>

**Canonical：有 release 声明，但发布文件未核实，且推理不合要求。** [代码](https://github.com/agentic-learning-ai-lab/canonical)、[所链HF](https://huggingface.co/agentic-learning-ai-lab/canonical)、[HTTP401记录](raw/hf_canonical.log)。不能把401直接说成“必须申请”或“仓库不存在”。

[libero_goal/configs/policy.yaml](code/agentic-learning-ai-lab__canonical/libero_goal/configs/policy.yaml)：`# Sequence length: obs_window(10) + goal(1) + t(1) + action_chunk(1) = 13 tokens.; goal_dim: 512       # encoder.output_dim (LIBERO has visual goal); action_chunk: 1; obs_window: 10; n_layers: 4; num_inference_steps: 50; n_action_steps: 1; action_window_size: ${data.action_chunk}   # eval rollout reads top-level cfg.action_window_size (=1); obs_window: 10; action_chunk: 1; use_libero_goal: true; window_size: ${data.obs_window}            # obs context window = 10; action_window_size: ${data.action_chunk}   # action prediction window = 1; vqbet_get_future_action_chunk: false       # DiT: single-step (no VQ-BeT future-chunk slicing); goal_conditional: null                     # goal = dataset last-frame view (use_libero_goal), not future-sampled`

[metaworld_bench/configs/policy.yaml](code/agentic-learning-ai-lab__canonical/metaworld_bench/configs/policy.yaml)：`#   epochs=1000  model.goal_dim=3  model.num_inference_steps=30; state_dim: 11               # ProprioFull: hand_pos(4) + prev_action(4) + EE_velocity(3); goal_dim: 0                 # no goal conditioning (single-task); override to 3 for Hard tier; action_chunk: 4             # per-step action-chunk length; obs_window: 2; n_layers: 8; num_inference_steps: 10     # π0 standard; override to 30 for Hard tier; obs_window: 2; action_chunk: 4             # action-chunk length (n_action_steps=2 below); use_libero_goal: false; n_action_steps: 2`

LIBERO：ResNet18→DiTFlowPolicy（dim224/depth4，50steps），obs_window10，goal_dim512/use_libero_goal=true，action chunk1。MW：ResNet18 spatial→DiTFlowSpatial（dim256/depth8，通常10steps、hard30steps），obs_window2，chunk4/execute2；state11由hand position4+velocity3+prev_action4组成。读取 [LIBERO train/eval](code/agentic-learning-ai-lab__canonical/libero_goal/train_policy.py)、[MW train/eval](code/agentic-learning-ai-lab__canonical/metaworld_bench/train_policy.py)、[policy实现](code/agentic-learning-ai-lab__canonical/models/policy.py)，不仅看模型名称。仓库声称的 `metaworld/encoder.pt` 等仅是文档示例，**不能当实际文件清单**。代码MIT，权重license/access未知。

<a id="cocos"></a>

**Cocos：最值得等待双边发布证明的新 flow 家族。** [论文](https://arxiv.org/abs/2505.11123)、[NeurIPS正式版本](https://papers.nips.cc/paper_files/paper/2025/file/ad6363efb7af02f7db13d087c7e649bd-Paper-Conference.pdf)、[官方代码](https://github.com/ZibinDong/cocos)、[OneDrive原链接](https://1drv.ms/u/c/ba682474b24f6989/EdyFZAAU47BPtZohTYDMHfwBeWB51Z8wSupl6z-iqfqBuw?e=kjBpzM)。

| 项目 | 核实 |
| --- | --- |
| 发布边界 | README只给LIBERO实现和一个压缩包下载链接。OneDrive landing/SharePoint请求失败；shares API未得root文件表（404）。无可验证文件名、bytes/config，故R1不通过，不能说已证实双边发布。raw/cocos_onedrive_share.txt保留响应。 |
| 默认LIBERO推理 | [cocos.py](code/ZibinDong__cocos/cocos.py) To1、Ta16、execute8、sampling_steps20、Euler rectified flow；每次以最新双视角obs重算x1与condition，默认未启用autoencoder EMA更新，无跨调用KV。MW对应代码未找到。 |
| 架构 | DINOv2-base→patch features（可pool为key），T5-base任务embedding→约40M RDT/DiT head；12层/384dim/6heads；condition-conditioned source用小AE。T5独立文本encoder不等于VLM/LLM action backbone。 |
| key成本风险 | 公开推理 process_obs 通过AE的DINO编码；condition.forward又调用vit_model。需要把两个视觉路径/AE算入full与hit，不能只把小AE后的tensor视作免费key。20步head是否主导需实测。 |
| paper LIBERO | Cocos std0.2：Goal93.8±0.3、Spatial97.5±0.7、Object99.1±0.6、Long88.7±0.1，平均94.8（%）；见raw/cocos_paper.html。 |
| paper MW | 30 tasks，74.8±0.9%，baseline59.5±3.8%。不是MT50/ML45；这一自选30任务是否符合owner标准也需单独确认。 |
| license/gating | 所查代码树无LICENSE；OneDrive访问失败不能归类为已知gating。没有可独立认证的发布checkpoint SR。 |

<a id="cosmos"></a>

**Cosmos：仅补充事实，不改变 owner 已定范围。**

[nvidia/Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B/tree/main) · created `2025-12-14T07:12:41.000Z` · revision `cb689ec0e3347c13667d70a78a3447388f5c3bb8` · gated `False` · card license `未声明` · [API 快照](raw/hf/nvidia__Cosmos-Policy-LIBERO-Predict2-2B/metadata.json)


| 文件（完整仓库 listing） | bytes |
| --- | --- |
| `.gitattributes` | 1519 |
| `Cosmos-Policy-LIBERO-Predict2-2B.pt` | 3913017345 |
| `README.md` | 11233 |
| `config.json` | 1354 |
| `libero_dataset_statistics.json` | 2372 |
| `libero_t5_embeddings.pkl` | 41957938 |

[mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B](https://huggingface.co/mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B/tree/main) · created `2026-07-09T00:08:55.000Z` · revision `6546ab82a1d302e4a947bc7b03880ec6031a6109` · gated `False` · card license `other` · [API 快照](raw/hf/mmurray-ms__Cosmos-Policy-MetaWorld-Predict2-2B/metadata.json)


| 文件（完整仓库 listing） | bytes |
| --- | --- |
| `.gitattributes` | 1519 |
| `README.md` | 896 |
| `consolidated.pt` | 3913050026 |
| `dataset_statistics.json` | 1589 |
| `t5_embeddings.pkl` | 52446356 |

官方LIBERO [model card](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)：Cosmos-Predict2-2B视频骨干，5 denoising steps、16×7 action；Spatial98.1/Object100/Goal98.2/Long97.6，平均98.5%。license是 **NVIDIA One-Way Noncommercial License (NSCLv1)**，不是Apache或通用开放商用许可。

社区MW [model card](https://huggingface.co/mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B)：FlowDAgger artifact，仅assembly-v3；base约52%/25episodes，full-mode steering84–88%；后者不能归为base policy。要求gripper proprio `[g,-g]`。仓库无模型config；`dataset_statistics.json`只含归一化，不替代架构配置。card license为other、无明确许可文本。所查 [microsoft/FlowDAgger](https://github.com/microsoft/FlowDAgger) 公开main主要为π0.5 MW与GR00T LIBERO配方，未定位该Cosmos runner。两边匿名可列文件，但MW范围和配套代码均不足。

<a id="other-hf"></a>

**其他 HF 文件核查：排除项也列真实 weights/config。** 下表仅压缩展示代表文件；每行API快照包含完整、未省略的siblings列表与revision。`none`表示所查仓库未声明license，不是许可推断。

| 仓库 | weights（代表） | config（代表） | gated / card license | 全 listing |
| --- | --- | --- | --- | --- |
| [A999baby/metaworld-pick-place-v3-10k](https://huggingface.co/A999baby/metaworld-pick-place-v3-10k/tree/main) | 未列出 | 未列出 | False / None | [全部1项](raw/hf/A999baby__metaworld-pick-place-v3-10k/metadata.json) |
| [AIRLab-POLIMI/CRT_LIBERO](https://huggingface.co/AIRLab-POLIMI/CRT_LIBERO/tree/main) | `CRT_libero.pth` (1557979362 B) | 未列出 | False / apache-2.0 | [全部3项](raw/hf/AIRLab-POLIMI__CRT_LIBERO/metadata.json) |
| [AIRLab-POLIMI/CRT_MetaWorld](https://huggingface.co/AIRLab-POLIMI/CRT_MetaWorld/tree/main) | `CRT_metaworld.pth` (2436833442 B) | 未列出 | False / apache-2.0 | [全部3项](raw/hf/AIRLab-POLIMI__CRT_MetaWorld/metadata.json) |
| [Jiaxin1234/metaworld-test](https://huggingface.co/Jiaxin1234/metaworld-test/tree/main) | 未列出 | 未列出 | False / apache-2.0 | [全部2项](raw/hf/Jiaxin1234__metaworld-test/metadata.json) |
| [MINT-SJTU/EVO-Depth-LIBERO](https://huggingface.co/MINT-SJTU/EVO-Depth-LIBERO/tree/main) | `libero_10/mp_rank_00_model_states.pt` (1829366928 B)<br>`libero_goal/mp_rank_00_model_states.pt` (1829366992 B)<br>`libero_object/mp_rank_00_model_states.pt` (1829366992 B)<br>另1个；详见快照 | `libero_10/config.json` (1062 B)<br>`libero_goal/config.json` (1070 B)<br>`libero_object/config.json` (1070 B)<br>另1个；详见快照 | False / None | [全部17项](raw/hf/MINT-SJTU__EVO-Depth-LIBERO/metadata.json) |
| [MINT-SJTU/EVO-Depth-MetaWorld](https://huggingface.co/MINT-SJTU/EVO-Depth-MetaWorld/tree/main) | `mp_rank_00_model_states.pt` (1829367056 B) | `config.json` (1080 B) | False / None | [全部5项](raw/hf/MINT-SJTU__EVO-Depth-MetaWorld/metadata.json) |
| [MINT-SJTU/Evo-World-Metaworld](https://huggingface.co/MINT-SJTU/Evo-World-Metaworld/tree/main) | `mp_rank_00_model_states.pt` (1742426145 B) | `config.json` (2306 B) | False / None | [全部5项](raw/hf/MINT-SJTU__Evo-World-Metaworld/metadata.json) |
| [MINT-SJTU/Evo1_LIBERO](https://huggingface.co/MINT-SJTU/Evo1_LIBERO/tree/main) | `mp_rank_00_model_states.pt` (1552487120 B) | `config.json` (990 B) | False / None | [全部4项](raw/hf/MINT-SJTU__Evo1_LIBERO/metadata.json) |
| [MINT-SJTU/Evo1_MetaWorld](https://huggingface.co/MINT-SJTU/Evo1_MetaWorld/tree/main) | `mp_rank_00_model_states.pt` (1552487184 B) | `config.json` (1020 B) | False / None | [全部4项](raw/hf/MINT-SJTU__Evo1_MetaWorld/metadata.json) |
| [Mark-ZJTang/alam_plus_pi_libero](https://huggingface.co/Mark-ZJTang/alam_plus_pi_libero/tree/main) | `params/ocdbt.process_0/d/1bf34b1765745b047dcd30b26fc42878` (869592221 B)<br>`params/ocdbt.process_0/d/393f1434b87736a8c145a96c75c7d3f2` (48293584 B)<br>`params/ocdbt.process_0/d/596022c041c375378a42ca7e159883d9` (1553649599 B)<br>另9个；详见快照 | 未列出 | False / None | [全部25项](raw/hf/Mark-ZJTang__alam_plus_pi_libero/metadata.json) |
| [Mark-ZJTang/alam_plus_pi_metaworld_mt50](https://huggingface.co/Mark-ZJTang/alam_plus_pi_metaworld_mt50/tree/main) | `params/ocdbt.process_0/d/08ff7b613001c9a5cc36fc1b007ed5bf` (2204979828 B)<br>`params/ocdbt.process_0/d/2e20b83b5a059636ed0eceb522003011` (2202725011 B)<br>`params/ocdbt.process_0/d/2fd3b330ccf0b5e0a815dfc3cd2886de` (2156848076 B)<br>另4个；详见快照 | 未列出 | False / None | [全部21项](raw/hf/Mark-ZJTang__alam_plus_pi_metaworld_mt50/metadata.json) |
| [Master210/MT10](https://huggingface.co/Master210/MT10/tree/main) | `models/button-press-topdown.pt` (54481418 B)<br>`models/door-open.pt` (54481354 B)<br>`models/drawer-close.pt` (54481418 B)<br>另9个；详见快照 | 未列出 | False / None | [全部33项](raw/hf/Master210__MT10/metadata.json) |
| [Master210/MT50](https://huggingface.co/Master210/MT50/tree/main) | `model.pt` (270461995 B) | 未列出 | False / None | [全部16项](raw/hf/Master210__MT50/metadata.json) |
| [MikeDaisy/zmax_hybrid_metaworld_push_v0.1](https://huggingface.co/MikeDaisy/zmax_hybrid_metaworld_push_v0.1/tree/main) | 未列出 | 未列出 | False / None | [全部1项](raw/hf/MikeDaisy__zmax_hybrid_metaworld_push_v0.1/metadata.json) |
| [Muhammad241198/metaworld-push](https://huggingface.co/Muhammad241198/metaworld-push/tree/main) | `model.safetensors` (206767520 B) | `config.json` (1649 B)<br>`train_config.json` (6621 B) | False / apache-2.0 | [全部9项](raw/hf/Muhammad241198__metaworld-push/metadata.json) |
| [RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial/tree/main) | `model-00001-of-00004.safetensors` (4925122448 B)<br>`model-00002-of-00004.safetensors` (4947392496 B)<br>`model-00003-of-00004.safetensors` (4947417456 B)<br>另1个；详见快照 | `config.json` (64653 B)<br>`generation_config.json` (136 B)<br>`preprocessor_config.json` (1650 B)<br>另2个；详见快照 | False / mit | [全部22项](raw/hf/RLinf__RLinf-OpenVLAOFT-GRPO-LIBERO-spatial/metadata.json) |
| [RLinf/RLinf-OpenVLAOFT-Metaworld-SFT](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-Metaworld-SFT/tree/main) | `lora_adapter/adapter_model.safetensors` (484458600 B)<br>`model-00001-of-00004.safetensors` (4925122448 B)<br>`model-00002-of-00004.safetensors` (4947392496 B)<br>另2个；详见快照 | `config.json` (3607 B)<br>`generation_config.json` (136 B)<br>`lora_adapter/adapter_config.json` (1133 B)<br>另3个；详见快照 | False / mit | [全部23项](raw/hf/RLinf__RLinf-OpenVLAOFT-Metaworld-SFT/metadata.json) |
| [RLinf/RLinf-Pi05-LIBERO-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT/tree/main) | `model.safetensors` (7473091464 B) | 未列出 | False / apache-2.0 | [全部4项](raw/hf/RLinf__RLinf-Pi05-LIBERO-SFT/metadata.json) |
| [RLinf/RLinf-Pi05-MetaWorld-SFT](https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT/tree/main) | `metadata.pt` (3172 B)<br>`model.safetensors` (7473091464 B) | 未列出 | False / None | [全部4项](raw/hf/RLinf__RLinf-Pi05-MetaWorld-SFT/metadata.json) |
| [Rhx11111/metaworld-256](https://huggingface.co/Rhx11111/metaworld-256/tree/main) | 未列出 | 未列出 | False / None | [全部1项](raw/hf/Rhx11111__metaworld-256/metadata.json) |
| [RoboHP/metaworld_mt50_act](https://huggingface.co/RoboHP/metaworld_mt50_act/tree/main) | `pretrained_model/model.safetensors` (206684616 B) | `pretrained_model/config.json` (1509 B)<br>`pretrained_model/train_config.json` (4682 B) | False / mit | [全部9项](raw/hf/RoboHP__metaworld_mt50_act/metadata.json) |
| [abraranwar/spur_metaworld](https://huggingface.co/abraranwar/spur_metaworld/tree/main) | `model-00001-of-00004.safetensors` (4970111944 B)<br>`model-00002-of-00004.safetensors` (4932952128 B)<br>`model-00003-of-00004.safetensors` (4932952216 B)<br>另4个；详见快照 | `config.json` (3383 B)<br>`config.yaml` (2061 B) | False / apache-2.0 | [全部15项](raw/hf/abraranwar__spur_metaworld/metadata.json) |
| [hjlkhjkl/metaworld](https://huggingface.co/hjlkhjkl/metaworld/tree/main) | `1model_step_100352.safetensors` (900901236 B)<br>`1model_step_120320.safetensors` (900901236 B)<br>`1model_step_140288.safetensors` (900901236 B)<br>另7个；详见快照 | 未列出 | False / None | [全部11项](raw/hf/hjlkhjkl__metaworld/metadata.json) |
| [jeffacce/dynamo_ssl](https://huggingface.co/jeffacce/dynamo_ssl/tree/main) | 未列出 | `configs/encoder/identity.yaml` (54 B)<br>`configs/encoder/resnet18_random.yaml` (92 B)<br>`configs/env/block_push_multiview.yaml` (310 B)<br>另29个；详见快照 | False / None | [全部1146项](raw/hf/jeffacce__dynamo_ssl/metadata.json) |
| [ksb21st/metaworld45_100demos_hf](https://huggingface.co/ksb21st/metaworld45_100demos_hf/tree/main) | `1100/500/params/ocdbt.process_0/d/02e273d26af0ca003b1c1d2706de49c0` (2169781658 B)<br>`1100/500/params/ocdbt.process_0/d/09f2714d47ab64b174d9f0918d2fdb67` (463222567 B)<br>`1100/500/params/ocdbt.process_0/d/286710d398319c9208fca750bae31d83` (2523951007 B)<br>另139个；详见快照 | 未列出 | False / None | [全部295项](raw/hf/ksb21st__metaworld45_100demos_hf/metadata.json) |
| [kunhsiang/act-libero-baseline](https://huggingface.co/kunhsiang/act-libero-baseline/tree/main) | `checkpoints/001000/pretrained_model/model.safetensors` (206712028 B)<br>`checkpoints/002000/pretrained_model/model.safetensors` (206712028 B)<br>`checkpoints/003000/pretrained_model/model.safetensors` (206712028 B)<br>另8个；详见快照 | `checkpoints/001000/pretrained_model/config.json` (1696 B)<br>`checkpoints/001000/pretrained_model/train_config.json` (5590 B)<br>`checkpoints/002000/pretrained_model/config.json` (1696 B)<br>另19个；详见快照 | False / mit | [全部119项](raw/hf/kunhsiang__act-libero-baseline/metadata.json) |
| [kunhsiang/act-metaworld-mt10-baseline](https://huggingface.co/kunhsiang/act-metaworld-mt10-baseline/tree/main) | `model.safetensors` (160961760 B) | `config.json` (1488 B)<br>`train_config.json` (13748 B) | False / None | [全部9项](raw/hf/kunhsiang__act-metaworld-mt10-baseline/metadata.json) |
| [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero/tree/main) | `model.safetensors` (906712520 B) | `config.json` (2419 B)<br>`train_config.json` (9631 B) | False / apache-2.0 | [全部9项](raw/hf/lerobot__smolvla_libero/metadata.json) |
| [lerobot/smolvla_metaworld](https://huggingface.co/lerobot/smolvla_metaworld/tree/main) | `model.safetensors` (906712520 B) | `config.json` (2485 B)<br>`train_config.json` (7750 B) | False / apache-2.0 | [全部9项](raw/hf/lerobot__smolvla_metaworld/metadata.json) |
| [limiqsqsqdq/metaworld-test](https://huggingface.co/limiqsqsqdq/metaworld-test/tree/main) | 未列出 | 未列出 | False / None | [全部1项](raw/hf/limiqsqsqdq__metaworld-test/metadata.json) |
| [namenu/gr00t17-lerobot-metaworld_mt50](https://huggingface.co/namenu/gr00t17-lerobot-metaworld_mt50/tree/main) | `model.safetensors` (12576215296 B) | `config.json` (2169 B)<br>`train_config.json` (6753 B) | False / None | [全部8项](raw/hf/namenu__gr00t17-lerobot-metaworld_mt50/metadata.json) |
| [nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B/tree/main) | `model-00001-of-00003.safetensors` (1996859888 B)<br>`model-00002-of-00003.safetensors` (1996145888 B)<br>`model-00003-of-00003.safetensors` (1455454240 B) | `config.json` (1674 B)<br>`experiment_cfg/metadata.json` (72028 B) | False / None | [全部13项](raw/hf/nvidia__GR00T-N1.5-3B/metadata.json) |
| [rishiad/kinitro-metaworld-agent](https://huggingface.co/rishiad/kinitro-metaworld-agent/tree/main) | 未列出 | 未列出 | False / mit | [全部55项](raw/hf/rishiad__kinitro-metaworld-agent/metadata.json) |
| [zhongweihong/xvla-wu-libero70k-metaworld82k](https://huggingface.co/zhongweihong/xvla-wu-libero70k-metaworld82k/tree/main) | `libero-70000/forge.safetensors` (1747186640 B)<br>`metaworld-82000/forge.safetensors` (1747186640 B) | `libero-70000/base_config.json` (9578 B)<br>`libero-70000/checkpoint_manifest.json` (1441 B)<br>`libero-70000/continuation_config.json` (278 B)<br>另13个；详见快照 | False / None | [全部28项](raw/hf/zhongweihong__xvla-wu-libero70k-metaworld82k/metadata.json) |

| 容易误判的项 | 依据 |
| --- | --- |
| SmolVLA | lerobot双边config明确SmolVLM2-500M-Video-Instruct；两个model.safetensors均906,712,520 B。无检查点SR记录。 |
| Evo | Evo1 config的InternVL；EVO-Depth是在同VLM骨干上加入depth；最新Evo-World仓库仍InternVL3，MW权重不构成新增非VLA。 |
| ALAM+π0 | 两个card直接声明pi0 post-training。LIBERO98.05→98.1%；MW85.0是difficulty-macro，episode-weighted为86.8%；当前camera0.71另一次run为86.4%episode/85.366883%macro。不可混合统计口径。 |
| RLinf π0.5 | LIBERO card重复论文SFT77.1→RL98.3%；不得把98.3当这个SFT权重。MW仓库没有README报告成功率。配置部分依赖RLinf代码，不是每个repo都有config.json。 |
| X-VLA | base_config是Florence-2。forge.safetensors仅含全部可训练参数，另需公开基础模型2toINF/X-VLA-pt固定revision c1c4a64a7e03ac5b95c468bf1578f3d03651b53b中的冻结参数。README保留原始历史评测协议（LIBERO2000/MW1000 episodes），未报SR；70k/82k是训练步数。 |
| CRT | [原论文](https://arxiv.org/abs/2602.01158)明确是对VLA输入图像做restoration；模型文件不是完整action策略。 |
| 未知MW-only | Master210、hjlkhjkl等确有文件，但未建立同一非VLA家族的LIBERO对应；Rhx11111、A999baby、limiqsqsqdq、MikeDaisy为空/占位仓。不能根据名字补出架构。 |
| SPUR / TinyVLA | SPUR config是Qwen2.5-VL，TinyVLA本身含VLM；未因名字含diffusion而计入非VLA。 |

**2024–2026 论文/代码发布复查。** 详细 pass/fail 与已摘录数值在 DECISION.md；这里保留排除证据入口。

| 方法 | 主要来源 | 发布/适用性核查 |
| --- | --- | --- |
| QueST (2024) | [论文](https://arxiv.org/abs/2407.15840) / [官方](https://github.com/pairlab/QueST) | README只有数据、train/eval、本地checkpoint；API releases=[]。不能把论文baseline DP的双基准成绩变成社区DP的成绩。 |
| PRISE (2024) | [论文](https://arxiv.org/abs/2402.10450) / [官方](https://github.com/FrankZheng2022/PRISE) | 代码可训练双基准，未核实双边发布权重；论文观测窗口含history。LLM-style不是LLM backbone，但仍不满足R1/R3。 |
| VQ-BeT (2024) | [官方](https://github.com/jayLEE0301/vq_bet_official) | README quick start链接Google Drive的goal-conditioned Kitchen模型。没有理由把此文件标作LIBERO/MetaWorld；BAKU压缩目录也没有vqbet。 |
| STAR (2025) | [官方](https://github.com/iLearn-Lab/ICML25-STAR) | README Checkpoints/Models明确尚不包括预训练权重，release TODO未勾选；releases=[]。 |
| MGP (2026) | [官方](https://github.com/Coondinator/masked_generative_policy) | README仍说完整训练/推理待发布；未找到双边weights，releases=[]。long-horizon版本不断修改全局历史轨迹，非当前obs独立。 |
| FDP (2025/2026) | [论文](https://arxiv.org/abs/2512.21898) / [官方](https://github.com/Chaoqi-LIU/fdp) | LIBERO、MW论文实验存在；README/目录/release未见成对checkpoint下载；releases=[]。 |
| DiffuView (CVPR2026) | [作者项目](https://github.com/zkzwork/diffuview) | index.html的Code为占位链接；HF DiffuView search空；releases=[]。项目成功率是方法表，不是release认证。 |
| AutoSpeed (2026) | [官方](https://github.com/tinda24/autospeed) | 公开sim download为Telon1/autospeed_alohasim_ckpt；real为cobot_magic。训练/eval_libero.py和eval_metaworld.py只是代码，releases=[]。 |
| LIFD (2026-09) | [论文](https://arxiv.org/abs/2609.19796) | recurrent scene memory是方法核心，直接违反当前obs条件；无需借重模型潜力强行通过。 |
| RoMAN-Flow (2026-08) | [官方](https://github.com/konnyaku28/RoMAN-Flow) | SmolVLM 500M，MW release列为后续；VLA排除。 |
| TriVLA (2025) | [论文](https://arxiv.org/abs/2507.01424) | Eagle2 VLM加视频branch；不是非VLA。 |
| FreqPolicy (2025) | [OpenReview](https://openreview.net/forum?id=NKNryrCGYn) | 不同实验接不同policy，MW的3D/flow与LIBERO的OpenVLA不能作为相同非VLA family release。 |
| DataMIL (2025/2026) | [论文](https://arxiv.org/abs/2505.09603) | LIBERO用Octo，MW用环境state MLP；不是同一可视觉cache的大策略。 |
| DILLO (2026) | [论文](https://arxiv.org/abs/2603.23149) | 是根据policy latent和planned actions做steering的LLM student，不是合格无LLM动作模型。 |

GitHub Releases API已核查且空：[pairlab/QueST](raw/releases_pairlab__QueST.json), [iLearn-Lab/ICML25-STAR](raw/releases_iLearn-Lab__ICML25-STAR.json), [Chaoqi-LIU/fdp](raw/releases_Chaoqi-LIU__fdp.json), [Coondinator/masked_generative_policy](raw/releases_Coondinator__masked_generative_policy.json), [ZibinDong/cocos](raw/releases_ZibinDong__cocos.json), [zkzwork/diffuview](raw/releases_zkzwork__diffuview.json), [tinda24/autospeed](raw/releases_tinda24__autospeed.json)。这仅排除GitHub Release assets；同时检查README外链，不能把空release列表当作全网无权重证明。

**WAM / 非 VLA 边界补查（未建立双基准交集，不计候选通过）。**

| 家族 | 来源 / 为什么未晋级 |
| --- | --- |
| PAD / Prediction with Action | [官方](https://github.com/Robert-gyj/Prediction_with_Action)：链接yjguo/pad_bridge_pre，Bridge/MetaWorld配方；未确认LIBERO权重。 |
| VPP | [官方](https://github.com/roboterax/video-prediction-policy)：发布CALVIN action policy和SVD robot，未确认LIBERO+MW成对发布。 |
| Grounding Video Models to Actions | [官方](https://github.com/video-to-action/video-to-action-release)：发布视频模型与探索训练代码，不足以验证完整policy在两边的权重+配置。 |
| DynaMo / Premier-TACO | [DynaMo](https://github.com/jeffacce/dynamo_ssl)、[Premier-TACO](https://github.com/PremierTACO/premier-taco)：表征预训练或encoder checkpoint不是双边完整policy；Canonical另行审查且历史/goal不合。 |
| GEVRM / FastWAM / DreamZero / OpenWAM / FLOWER | 名称/模型枚举和论文关键词搜索未核实双边完整发布；不要以LIBERO checkpoint+MW训练支持代替交集。 |
| SUREFlow / MaIL | [SUREFlow论文](https://arxiv.org/abs/2607.10504)主报LIBERO/LIBERO-PRO；[发布checkpoint issue](https://github.com/tanvirnwu/SUREFlow_IROS_2026/issues)仍开放；未建立MW交集。不能仅因Mamba自动断定有跨调用state，需具体代码，但R1已有缺口。 |

**来源版本与本地审查文件。** 下表树SHA来自实际GitHub API，代码文件保存在对应目录；README/HEAD可能在未来变更，审计应以本地快照为准。

| 官方/作者仓库 | tree SHA | 快照 |
| --- | --- | --- |
| [Chaoqi-LIU/fdp](https://github.com/Chaoqi-LIU/fdp) | 6c7426fa46a4d3d508abb8d42df019ffb7b34935 | [API tree](code/Chaoqi-LIU__fdp/tree.json) |
| [Coondinator/masked_generative_policy](https://github.com/Coondinator/masked_generative_policy) | b6f86df01fb5b0a8a8dcd0b8799bd04b9db46436 | [API tree](code/Coondinator__masked_generative_policy/tree.json) |
| [FrankZheng2022/PRISE](https://github.com/FrankZheng2022/PRISE) | 7860b7163bb5417f8503abfa7e18c7839ba5d6a9 | [API tree](code/FrankZheng2022__PRISE/tree.json) |
| [MINT-SJTU/Evo-1](https://github.com/MINT-SJTU/Evo-1) | 5fd14b015013c4fd0aacf5f8f48f868ca9b870a2 | [API tree](code/MINT-SJTU__Evo-1/tree.json) |
| [MINT-SJTU/Evo-Depth](https://github.com/MINT-SJTU/Evo-Depth) | 617e3f901321573001a02ebb0e8e3ff8802122bd | [API tree](code/MINT-SJTU__Evo-Depth/tree.json) |
| [MINT-SJTU/Evo-World](https://github.com/MINT-SJTU/Evo-World) | 0b3e0b7b917ee6a515c185cedfb9d5efe3d5a04e | [API tree](code/MINT-SJTU__Evo-World/tree.json) |
| [PremierTACO/premier-taco](https://github.com/PremierTACO/premier-taco) | b0d908a51b22967c35ad93e09ea0faf466cf569e | [API tree](code/PremierTACO__premier-taco/tree.json) |
| [Robert-gyj/Prediction_with_Action](https://github.com/Robert-gyj/Prediction_with_Action) | a0bdc12e05d435a77efd830ee6953d556c1a6083 | [API tree](code/Robert-gyj__Prediction_with_Action/tree.json) |
| [ZibinDong/cocos](https://github.com/ZibinDong/cocos) | ccfb8dd862fc24eab2435b9fb0106615329862ef | [API tree](code/ZibinDong__cocos/tree.json) |
| [agentic-learning-ai-lab/canonical](https://github.com/agentic-learning-ai-lab/canonical) | 6338bdea11a4b6e994422c20f898bd9ff264a4fb | [API tree](code/agentic-learning-ai-lab__canonical/tree.json) |
| [branyang02/openpi-cuda](https://github.com/branyang02/openpi-cuda) | 1cade6ad4f3a14c784c21c9b6600447185a41b9a | [API tree](code/branyang02__openpi-cuda/tree.json) |
| [branyang02/openpi-eval](https://github.com/branyang02/openpi-eval) | ff78053a5557517a64e1b7566e8d47214a21adb8 | [API tree](code/branyang02__openpi-eval/tree.json) |
| [branyang02/openpi-trajectory](https://github.com/branyang02/openpi-trajectory) | 2e04ff110e4978666315bb143eb26f361c73a038 | [API tree](code/branyang02__openpi-trajectory/tree.json) |
| [branyang02/ricl_openpi_latent_action](https://github.com/branyang02/ricl_openpi_latent_action) | d01f0885cc82c4fc888335b285b01f87086ad2dd | [API tree](code/branyang02__ricl_openpi_latent_action/tree.json) |
| [branyang02/vvla](https://github.com/branyang02/vvla) | a38a2485439a8165c2e3ea8293cd6a770777e249 | [API tree](code/branyang02__vvla/tree.json) |
| [iLearn-Lab/ICML25-STAR](https://github.com/iLearn-Lab/ICML25-STAR) | fd7ad83ac6a3fe97f1a51823e9c1097a9475ed0c | [API tree](code/iLearn-Lab__ICML25-STAR/tree.json) |
| [jayLEE0301/vq_bet_official](https://github.com/jayLEE0301/vq_bet_official) | 09d4851288ca5deaaa1ab367a208e520f8ee9a84 | [API tree](code/jayLEE0301__vq_bet_official/tree.json) |
| [jeffacce/dynamo_ssl](https://github.com/jeffacce/dynamo_ssl) | 62874039e5d3fc663b0fccd2083b2dd2c8d9935e | [API tree](code/jeffacce__dynamo_ssl/tree.json) |
| [konnyaku28/RoMAN-Flow](https://github.com/konnyaku28/RoMAN-Flow) | e2ea039737cd232436642ab37dc11e614afc4ffd | [API tree](code/konnyaku28__RoMAN-Flow/tree.json) |
| [microsoft/FlowDAgger](https://github.com/microsoft/FlowDAgger) | 3426f246510fd67a9e39c3c08bd5b02d586ad688 | [API tree](code/microsoft__FlowDAgger/tree.json) |
| [pairlab/QueST](https://github.com/pairlab/QueST) | cc25a835a00cf2d91df69704bdd757904d13c061 | [API tree](code/pairlab__QueST/tree.json) |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | 5ba07ac6661db573af695b419a7947ecb704690f | [API tree](code/real-stanford__diffusion_policy/tree.json) |
| [roboterax/video-prediction-policy](https://github.com/roboterax/video-prediction-policy) | 75cf37ad0f537627e1a3aa9a04bab70324ec27ff | [API tree](code/roboterax__video-prediction-policy/tree.json) |
| [siddhanthaldar/BAKU](https://github.com/siddhanthaldar/BAKU) | 47e367059191de30ac4b6353ff5615fdef11d8c5 | [API tree](code/siddhanthaldar__BAKU/tree.json) |
| [tinda24/autospeed](https://github.com/tinda24/autospeed) | 5aaf5e7778106733e20da9d4309e29c2dc7e43ec | [API tree](code/tinda24__autospeed/tree.json) |
| [video-to-action/video-to-action-release](https://github.com/video-to-action/video-to-action-release) | 00ba88ea4d225420584af443e7cc26846cee5682 | [API tree](code/video-to-action__video-to-action-release/tree.json) |
| [zkzwork/diffuview](https://github.com/zkzwork/diffuview) | 4dd4ca2044138105cb0f3121f743d873417e7db4 | [API tree](code/zkzwork__diffuview/tree.json) |

**本轮未执行的事。** 无GPU、无仿真成功率实验、无server/remote/tether、无完整大权重下载、无git commit、无进程终止。所有新文件位于本survey2_astra目录。CPU profiler日志有CUDA设备探测失败提示，但配置为空可见GPU且ProfilerActivity.CPU，模型tensor全部在CPU；没有GPU计算。
