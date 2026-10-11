# R17 二轮调研证据（survey2_opus）

日期 2026-10-08。全部为 Web / HF API / GitHub API 读取和 CPU 探针；没有下载任何权重（只用 HTTP Range 读了 safetensors 头和 torch zip 的 data.pkl，以及两个 3.5 KB 的 metadata.pt）。

## 1. Hugging Face 枚举

### 1.1 按 id 搜索（`/api/models?search=<q>&limit=1000`，跟 Link 头翻页，`full=true`）

| 查询 | 返回条数 | 说明 |
|---|---:|---|
| `metaworld` | 103 | |
| `meta-world` / `meta_world` / `meta world` | 1317 | HF 把它拆成 meta + world，绝大多数是 Llama、metaclip，再用正则过滤 |
| `mt50` / `mt10` / `mt45` / `ml45` | 15 / 27 / 2 / 1 | |
| `mw_` | 1031 | 仅 `abhranil14/mw_VideoAgent*`（只有 MetaWorld）和 `yirantao1000/pi05-mw-*`（VLA）相关 |
| `libero` | 3667 | |

过滤正则：MetaWorld = `meta[-_ ]?world|metaworld|mt50|mt10|ml45`（id + tags），LIBERO = `libero`。得到 **MetaWorld 136 个、LIBERO 3665 个**。

MetaWorld 136 个按族归类：

| 数量 | 族 | 例 |
|---:|---|---|
| 40 | SmolVLA（VLA） | lerobot/smolvla_metaworld、nikriz/smolvla_metaworld_mt50、JunnDooChoi/slurm_smolvla_* |
| 18 | π0.5（VLA） | RLinf/RLinf-Pi05-MetaWorld-SFT、JunnDooChoi/slurm_pi05_*、TongheZhangTH/Pi05-Metaworld-ML45-SFT |
| 7 / 2 | π0 / π0-FAST（VLA） | RLinf/RLinf-Pi0-MetaWorld-*、brandonyang/pi0fast-metaworld-checkpoints |
| 4 / 2 / 1 / 1 / 1 / 1 / 1 | Evo-1·EVO-Depth·Evo-World / starVLA / OpenVLA-OFT / X-VLA / GR00T N1.7 / ALAM+π / TinyVLA（全是 VLA） | MINT-SJTU/*、RLinf/RLinf-OpenVLAOFT-Metaworld-SFT、zhongweihong/xvla-wu-libero70k-metaworld82k、namenu/gr00t17-lerobot-metaworld_mt50 |
| 10 | ACT | kunhsiang/act-metaworld-mt10-*、RoboHP/metaworld_mt50_act、aryannzzz/act-metaworld-shelf-* |
| 2 | Diffusion Policy | brandonyang/dp-metaworld-checkpoints、iFaz/diffusion-metaworld_medium-seed42-half（空仓） |
| 1 | Cosmos Policy（视频扩散 WAM） | mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B |
| 1 | CRT（未知） | AIRLab-POLIMI/CRT_MetaWorld |
| 1 / 1 / 1 | CLIP-BC / 奖励模型 / 状态 RL | TaherFattahi/MetaWorld-VLA-openai-clip-vit、abraranwar/spur_metaworld、rishiad/kinitro-metaworld-agent |
| 12 | 空仓或无权重 | A999baby/*、Master210/MT10、hjlkhjkl/metaworld、ksb21st/* 等 |
| 28+1 | 无关（LLM 等） | TemporalGames/opt-1.3b-*mt10*、tharindu/mt5_* |

### 1.2 补充枚举

| 方法 | 结果 |
|---|---|
| HF 全文检索（`/api/search/full-text?q=...&type=model`，翻页）：`metaworld` 73、`meta-world` 58、`mt50` 10、`MT10 sawyer` 6 | 比 id 搜索多出 44 个；相关的只有 `hackhackhack66666/aaai27-models`（OAT，受限仓库）、`facebook/jepa-wms`（目标图规划）、`nicklashansen/tdmpc2`·`newt`（RL 世界模型）、`jat-project/jat`、`dunnolab/Vintix`（上下文历史）、`SongMugeon/*_meta_*`（ACT / SmolVLA），无新的 LIBERO∩MetaWorld 非 VLA 族 |
| 数据集标签反查：枚举 292 个名字含 metaworld/mt50/mt10/ml45 的数据集，再 `/api/models?filter=dataset:<id>` | 7 个数据集有模型挂标签，共 31 个模型：SmolVLA、ACT、π0.5，无新族 |
| 作者交叉：MetaWorld 136 个的作者 + LIBERO 非 VLA 关键词命中的作者，共 237 个，逐个拉全部模型，看是否两边都有 | 22 个作者两边都有（含 Efficient-Large-Model、nvidia、allenai、twanghcmut 这类 "Meta-Llama" 假阳性）；非 VLA 的只有 AIRLab-POLIMI（CRT）、brandonyang（DP-T）、iFaz（DP，MW 空仓）、kunhsiang / SongMugeon（ACT） |
| 族关键词：`avdc flowdiffusion vpp svd-robot s-vam dp3 3d-diffusion octo uva unified-world uwm mdt hpt dreamzero fastwam lingbot-va motus mimic-video cosmos3` | 全部 MetaWorld = 0；LIBERO：octo 4、mdt 3、dreamzero 9、fastwam 11、lingbot-va 10、mimic-video 2、cosmos3 26 |
| `library=cosmos-policy` / `search=cosmos-policy` | 约 100 个 Cosmos Policy / Cosmos3 policy 仓库，MetaWorld 只有 mmurray-ms 这 1 个 |

官方 WAM / 扩散仓库内部是否藏了 MetaWorld 子目录（recursive tree 后对路径做正则 `meta[-_ ]?world|mt50|mt10|ml45|sawyer`）：nvidia Cosmos-Policy-{LIBERO,RoboCasa,ALOHA}、Cosmos3-{Nano,Edge}-Policy-DROID、ZibinDong/fastwam_{base,libero,robotwin}、lerobot/fastwam_libero、lerobot/lingbot_va_libero_long、robbyant/lingbot-va-posttrain-libero-long、RLinf DreamZero-LIBERO、OpenWAM-Alpha-Sim-LIBERO、Long-WAM-LIBERO-COD、lawam、ImageWAM、VLA-JEPA-LIBERO、MoDE_LIBERO_90、yjguo/pad_bridge_pre、yjguo/dp-calvin、uniwam、mimic-video、StreamWAM、AED、liruiw/hpt-base，共 25 个仓库，**全部 0 命中**。

LIBERO 侧非 VLA 族（id 命中）：Cosmos Policy（nvidia 官方 + AlphaBrainGroup 等）、Cosmos3（社区）、FastWAM（lerobot、ZibinDong）、Long-WAM（Efficient-Large-Model）、DreamZero（RLinf 等）、LingBot-VA（robbyant、lerobot）、OpenWAM、LaWAM、ImageWAM、StreamingWAM、UniWAM、MoGE-WAM、4D-WAM、BadWAM、smolwam、mimic-video、ODEWorld、DART、WorldPilot（引导 VLA）、VLA-JEPA（VLA）、Dreamer-VLA（探针）、MoDE、FLOWER（VLA）、MDT/DiT（hqfang、continuallearning）、Octo（社区）、RDT（社区）、Diffusion Policy（lerobot 格式十余个）。以上除 Cosmos Policy、DP、ACT 外，作者名下都没有 MetaWorld 模型。

## 2. 候选文件清单（API tree，recursive）

| 仓库 | 文件 |
|---|---|
| nvidia/Cosmos-Policy-LIBERO-Predict2-2B（2025-12-14，未门控，NSCLv1 非商用） | `Cosmos-Policy-LIBERO-Predict2-2B.pt` 3913 MB、`config.json`、`libero_dataset_statistics.json`、`libero_t5_embeddings.pkl` 42 MB、README |
| mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B（2026-07-09，license: other） | `consolidated.pt` 3913 MB、`dataset_statistics.json`、`t5_embeddings.pkl` 52 MB、README |
| brandonyang/dp-libero-checkpoints（2026-04-22，无 license、无 README） | `dp_libero_lang_v1/30000/{model.safetensors 367 MB, optimizer.pt, metadata.pt, assets/physical-intelligence/libero/norm_stats.json}` |
| brandonyang/dp-metaworld-checkpoints（2026-04-22，无 license、无 README） | `dp_metaworld_lang_v1/{30000,85000}/{model.safetensors 367 MB, optimizer.pt, metadata.pt, assets/brandonyang/metaworld_ml45/norm_stats.json}` |
| iFaz/diffusion-libero_goal-seed3-15000 | lerobot DP：`config.json`（n_obs_steps=2、horizon 64）、`model.safetensors` 1112 MB、pre/post processor |
| iFaz/diffusion-metaworld_medium-seed42-half | 只有 `.gitattributes`（空仓） |
| AIRLab-POLIMI/CRT_LIBERO / CRT_MetaWorld（2026-01-30，apache-2.0） | 各只有 `CRT_libero.pth` 1558 MB / `CRT_metaworld.pth` 2437 MB + 空 README |
| kunhsiang/act-metaworld-mt10-baseline / act-libero-baseline | lerobot ACT：config（n_obs_steps 1、chunk 100、resnet18）、model.safetensors 161 / 207 MB；MetaWorld 版输入只有 `observation.state` + `observation.environment_state`（无图像） |
| SongMugeon/Act_finetune_meta_260305 / Act_finetune_libero_10_260211 | lerobot ACT，MetaWorld 版输入 480×480 图像 |
| Po-Chen/flowdiffusion（AVDC 官方） | `ckpts/metaworld/model-24.pt` 3219 MB、`metaworld_DA`、`metaworld_flow`、`ithor`、`bridge`、`diffusion_policy/model-10.pt` |
| Renton-Ren/video-to-action-libero-assets（GCP LIBERO 镜像；官方在 OneDrive） | `libero-video-model.zip` 3219 MB、`model-160000.pt` 1396 MB、随机动作 hdf5 ×2、`metaworld_fail.zip` 19983 MB |
| MINT-SJTU/Evo-World-Metaworld（2026-10-04） | `mp_rank_00_model_states.pt` 1742 MB、`config.json`：`vlm_name: OpenGVLab/InternVL3-1B` → VLA |
| hackhackhack66666/aaai27-models（OAT，受限） | 2871 文件、86 个 ckpt；policy ckpt 只有 robomimic can/lift/square、robocasa 4 任务、MetaWorld box-close/coffee-pull/disassemble/stick-pull 单任务；无 LIBERO ckpt |
| wangshaoxuan/RoMAN-Flow | libero_{90,goal,long,object,spatial} + robomimic，无 MetaWorld |

### 2.1 权重内部（只读头部）

- **Cosmos MetaWorld 社区权重覆盖了哪些任务**：下载 52 MB 的 `t5_embeddings.pkl`，用 pickletools 只列字符串、不反序列化，共 50 个键，是 MetaWorld 任务指令。与 `lerobot/metaworld_mt50` 的 `meta/tasks.parquet`（50 条）比对，49 条逐字相同，剩下 1 条是改写（"Push a puck back to a target behind it" 对 "Push the puck back toward the robot to the goal"）。由此判断该权重很可能是在 lerobot MT50 数据（2500 局，单路 480² 图像 + 4 维状态）上做的多任务微调；README 只报告了 assembly-v3 一项。`consolidated.pt` 里只有 DiT（1956 M 参数，687 个张量，`net.*`），VAE 用基座的；没有 config。`dataset_statistics.json`：动作 7 维，其中第 3–5 维方差为 0（MetaWorld 的 4 维动作被填进 LIBERO 的 7 维格式）；本体 9 维，前 5 维 = [g, −g, x, y, z]，其余为常数。

- **brandonyang DP-T**（safetensors 头）：91.8 M 参数；`_policy.model.decoder` 50.5 M（12 层，d=512）、`_policy.model.encoder` 12.6 M（4 层 cond 编码器）、`_policy.obs_encoder.obs_nets.{base_0_rgb,left_wrist_0_rgb}` 28.3 M（ResNet18 + 语言 FiLM，lang_emb 768）；`pos_emb [1,16,512]`、`cond_pos_emb [1,2,512]`、`head [7,512]`（LIBERO）/ `[4,512]`（MetaWorld）。
- **metadata.pt**（pickletools 反汇编，不执行）：`n_obs_steps 1`、`horizon 16`、`n_action_steps 16`、`num_inference_steps 100`、`num_train_timesteps 100`、`beta_schedule squaredcos_cap_v2`、`prediction_type epsilon`、图像 96×96 两路、crop 84×84、`vision_backbone resnet18`、`n_layer 12`、`n_cond_layers 4`、`n_emb 512`、`project_name openpi`；数据 `physical-intelligence/libero`（4 套件混训）/ `brandonyang/metaworld_ml45`（44 任务、4332 局、80 fps）；LIBERO 只上传了 30000 / 100000 步。
- **CRT**（torch zip 头）：609 M（MW）/ 389 M（LIBERO）；`patch_embed.proj [2880,1500]`、6 个 ViT 块（d=2880）、`pos_embed [1,2880,48,48]`、`head [300,2880]`、旋转位置编码 → 1500 维 token 到 300 维的回归器，没有图像编码器、没有动作头，无代码、无说明。

## 3. 无状态核查（读推理代码）

| 模型 | 代码 | 结论 |
|---|---|---|
| Cosmos Policy | `NVlabs/cosmos-policy` `cosmos_policy/experiments/robot/libero/run_libero_eval.py`：`action_queue = deque(maxlen=num_open_loop_steps)`；队列空时 `get_action(cfg, model, dataset_stats, observation, task_description, seed=cfg.seed + query_idx, randomize_seed=False, num_denoising_steps_action=5)`；默认 `chunk_size 16`、`num_open_loop_steps 16`、`num_queries_best_of_n 1`、`ar_future_prediction False`。`cosmos_utils.py::get_action` 每次只用当前帧：序列 `[blank, curr_proprio, curr_wrist, curr_third_person, action, future_proprio, future_wrist, future_third_person, value]`（state_t=9），每帧复制 4 份给 Wan2.1 VAE（1+8×4=33 帧、224²），proprio / action 以 latent 注入；未见 KV cache 或历史。论文原文："does not use input history" | 无状态（当前帧 + 本体 + 文本 + 固定种子） |
| DP-T（brandonyang） | 无公开代码；metadata `n_obs_steps 1`，`cond_pos_emb` 长度 2 = 时间 token + 1 个观测 token | 无状态（单帧） |
| lerobot DP（iFaz、SongMugeon 等 LIBERO 版） | config `n_obs_steps 2` | 需要上一帧，不合格 |
| AVDC | 论文：生成 8 帧视频 → 光流 → 刚体变换 → 抓/推原语 + IK + 位置控制器，开环执行，卡住再重规划 | 没有学到的动作头，动作来自脚本控制器 |
| GCP（video-to-action） | 论文 3.1 节：目标条件 DP 依次以生成视频的每一帧作为目标图 | 低层策略需要目标图、跨调用持有计划 |
| LingBot-VA | 论文 2601.21998："persistent memory through KV cache" | 有 KV cache（且无 MetaWorld） |
| PEWAM（Keep the Effect） | 论文：跨调用保存程序指针和 engagement 标志 | 有跨调用状态（且未发布） |
| MGP-Long | 论文：已执行 token 作为隐状态 | 有状态；MGP-Short 在 MetaWorld 上用 4 帧历史 |

## 4. 成功率（有报告的都在这里）

| 模型 | LIBERO | MetaWorld |
|---|---|---|
| Cosmos Policy | 98.5%（Spatial 98.1 / Object 100.0 / Goal 98.2 / Long 97.6；500 局 × 3 种子，H100）——NVIDIA 模型卡和论文 | 社区 ckpt（T5 嵌入覆盖 MT50 的 50 条指令）：只报告了 assembly-v3，冻结基座约 52%（25 局），FlowDAgger 引导后 84–88%（README，作者说"刻意留余量"）；其余 49 个任务无报告 |
| DP-T（brandonyang） | **无报告** | **无报告** |
| AVDC | — | 11 任务平均 43.1%（带重规划）/ 19.6%（不重规划）/ 37.5%（DDIM 10 步） |
| GCP | 8 任务两场景：42.2% / 62.0% | 6 任务 67.2%（未发布 MetaWorld 权重） |
| ACT（kunhsiang、SongMugeon） | 无报告 | 无报告 |
| CRT | 无 | 无 |

## 5. 成本数据

### 5.1 Cosmos Policy（论文 2601.16163 附录，1×H100）

| 去噪步数 | 每块延迟 | 推导 |
|---|---:|---|
| 5（LIBERO 默认，动作 + 未来态 + 价值并行） | 0.61 s | 每步 ≈ (0.61−0.16)/4 = 0.1125 s |
| 1 | 0.16 s | 固定部分 ≈ 0.16 − 0.1125 = 0.0475 s（含 33 帧 VAE 编码、T5 查表、预处理）= 命中路径上界 |
| 10（ALOHA） | 0.95 s | |

RoboCasa 上 1 步与 5 步成功率 66.4% 对 67.1%。DiT 输入 9 个 latent 帧 × 14×14 = 1764 token，2B 参数，估算每步约 7 TFLOP。

### 5.2 DP-T CPU 探针（随机权重、形状同 brandonyang；`taskset -c 26-33,70-77 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16`，lerobot 环境的 torch 2.11）

```
params: vision 22.4M, denoiser 63.8M
FLOPs: vision(2 cams) 1.18 GF, per denoise step 1.44 GF, 100 steps 143.6 GF, 10 steps 14.4 GF
CPU(16 thr) latency: vision 12.7 ms, step 23.43 ms, full@100 2355 ms, full@10 247 ms
full:hit FLOPs @100 = 122x, @10 = 13x; CPU latency @100 = 185x, @10 = 19.4x
```

脚本：`/tmp/r17s2o/probe/dpt_cost.py`（scratch，不在输出目录）。GPU 上未测；MGP 论文（2512.09101）报告 DP 在 MetaWorld 上每次调用 106 ms，可作量级参考。

### 5.3 参照：我们的系统（R16 REPORT 口径）

π0.5：视觉 .152、满步动作部分 .848、c(1)=.454 → 全调用 : 命中 = 6.6×（满步）/ 4.0×（k=1）。GR00T：视觉 .148、c(1)=.258 → 6.8× / 2.7×。

## 6. 论文检索（2024–2026，同时报告 LIBERO 和 MetaWorld）

方法：
- arXiv HTML 全文统计 "LIBERO" 和 "Meta-?World" 的出现次数。覆盖 OpenMOSS/Awesome-WAM 的 220 篇 + mbreuss/diffusion-literature 的 2024+ 共 252 篇；HF papers 搜索 20 个查询得到 1345 篇，扫了约 1000 篇（标题与机器人相关的 497 篇全扫）。两者都提到、且不是 VLA / 基准 / 综述的，只剩下表中各项。arXiv export API 和 Semantic Scholar 都被限流（429），因此改用下面两种。
- OpenAlex：`abstract.search:LIBERO + abstract.search:Meta-World|MetaWorld`（20 + 17 篇），`fulltext.search:LIBERO + fulltext.search:MetaWorld`（89 篇）。
- WebSearch 若干组；GitHub 代码搜索 README（`metaworld libero checkpoint huggingface` 等 4 组，共约 300 个仓库）。

**结果：Awesome-WAM 列表里的 WAM 论文没有一篇同时评测两个基准。** 视频扩散 / WAM 类按基准分成两堆：只有 LIBERO 的（Cosmos Policy、FastWAM、LingBot-VA、DreamZero、OpenWAM、LaWAM、ImageWAM、ODEWorld、VLA-JEPA、WorldPilot、BadWAM、Mask World Model、SG-WAM、Flex-π、WorldDiT、AcrossWAM），和只有 MetaWorld 的（VPP 2412.14803、PAD 2411.18179、AVDC 2310.08576、VPDD 2402.14407、S-VAM 2603.16195）。

同时报告两个基准的非 VLA 论文及权重情况：

| 论文 | 类型 | 两基准权重 | 依据 |
|---|---|---|---|
| QueST（pairlab/QueST，2024） | 技能 token 自回归 | 无 | README 只给训练脚本 |
| BAKU（siddhanthaldar/BAKU，2024） | 多任务 Transformer | 无 | MetaWorld 数据需邮件索取，无 ckpt |
| PRISE（FrankZheng2022/PRISE，2024） | 动作抽象 | 无 | 只有训练流程 |
| Premier-TACO（2024） | 表征预训练 | 无 | 只放了 DMC 编码器，MetaWorld / LIBERO 代码"coming soon" |
| SkillDiffuser（Liang-ZX，2024） | 技能 + 扩散 | 无 | README 无 ckpt |
| STAR（iLearn-Lab/ICML25-STAR，2025） | RaRSQ 技能 | 无 | README："Pretrained checkpoints are not included" |
| Cocos（ZibinDong/cocos，2025） | DINOv2-B + T5 + 40 M flow DiT | 只有 LIBERO（OneDrive） | README 只讲 LIBERO；且视觉占大头 |
| MGP（2512.09101） | 掩码生成 token | 无 | 文中无代码链接；7 M 参数 |
| ARP（2606.22480） | 量化技能 | 无 | 只有项目页 |
| PEWAM / Keep the Effect（2610.02398） | 71.5 M 四流 DiT WAM | 未发布 | "upon acceptance we will release"；跨调用程序指针；MetaWorld 为 ML10/ML45 单示范 |
| LIFD（2609.19796） | 3D 场景记忆扩散 | 无 | 有记忆（有状态） |
| Sparse2Act（2606.12759） | 点云表征 | 无 | 只有项目页 |
| ORCA（2510.15510，CVPR 26） | 冻结 T2I 扩散作视觉表征 | 无 | 项目页无代码；视觉占大头 |
| CPC（2601.22475） | 持续蒸馏 | 无 | — |
| GCP（video-to-action，ICLR 25） | 视频扩散 + 目标条件 DP | 只有 LIBERO 8 任务 | MetaWorld 未发布；需目标图 |
| Learning Skills from Action-Free Videos（2512.20052） | AVDC + LAPA | 无 | — |
| Canonical（agentic-learning-ai-lab） | ResNet18 + DiT-flow | HF 模型仓库 401 | 小模型 |
| MINT（huangrm） | token 策略 | 只有 LIBERO，MetaWorld "计划中 2026 H2" | README |
| RoMAN-Flow | 自回归归一化流（离线 RL） | 只有 LIBERO + robomimic | HF 文件清单 |
| Much Ado About Noising | 多种生成控制 | 只有 robomimic / kitchen / pusht | README |
| OAT（Chaoqi-LIU/oat；aaai27-models） | 有序动作 token | 官方无 ckpt；第三方只有 MetaWorld 单任务 | README / 文件清单 |
| HPT（liruiw/HPT） | 异构预训练 Transformer | 只有主干，无 MetaWorld / LIBERO 微调权重 | README |
| DILLO / Describe-Then-Act（MaxPappa/DILLO，ECCV 26） | 在 ACT 上加 VLM 蒸馏的引导层 | 底座是 ACT（廉价） | README |
| π_RL（2510.25889，RLinf） | π0 / π0.5 的在线 RL | VLA | — |
| AutoSpeed（tinda24/autospeed，2607.01051） | ACT / BAKU 式 Actor + 速度学习 | 只放了 ALOHA-sim 权重（Telon1/autospeed_alohasim_ckpt），没有 LIBERO / MetaWorld 权重 | README |

VLA 类同时有两基准（R2 直接不合格，不展开）：PolicyTrim、VLA-Corrector、FabriVLA、π0 / π0.5 / π0-FAST（RLinf、brandonyang、JunnDooChoi、SongMugeon）、SmolVLA、OpenVLA-OFT、X-VLA、Evo-1、EVO-Depth、Evo-World、starVLA、ALAM、GR00T N1.7、Spike-driven VLA、RoboTron-Mani、One-Token-Per-Frame（π0 基座）、TinyVLA（只有 MetaWorld）。

## 7. 链接

- Cosmos Policy：https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B ，https://github.com/NVlabs/cosmos-policy ，https://arxiv.org/abs/2601.16163
- Cosmos MetaWorld（社区）：https://huggingface.co/mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B ；FlowDAgger 公开仓库 https://github.com/microsoft/FlowDAgger 只有 π0.5-MetaWorld 和 GR00T-LIBERO 两个后端，Cosmos runner 未公开
- DP-T：https://huggingface.co/brandonyang/dp-libero-checkpoints ，https://huggingface.co/brandonyang/dp-metaworld-checkpoints ；数据 https://huggingface.co/datasets/brandonyang/metaworld_ml45
- AVDC：https://huggingface.co/Po-Chen/flowdiffusion ，https://github.com/flow-diffusion/AVDC_experiments ；GCP：https://github.com/video-to-action/video-to-action-release
- CRT：https://huggingface.co/AIRLab-POLIMI/CRT_LIBERO ，https://huggingface.co/AIRLab-POLIMI/CRT_MetaWorld
- ACT：https://huggingface.co/kunhsiang/act-metaworld-mt10-baseline ，https://huggingface.co/kunhsiang/act-libero-baseline
- FastWAM（只有 LIBERO / RoboTwin）：https://huggingface.co/lerobot/fastwam_libero_uncond_2cam224 ，https://arxiv.org/abs/2603.16666
- 枚举原始数据（scratch）：`/tmp/r17s2o/hf_models_all.json`、`mw_ids.json`、`lib_ids.json`、`author_both.json`、`hf_fulltext.json`、`models_by_mw_dataset.json`、`arxiv_scan_*.tsv`、`openalex.json`
