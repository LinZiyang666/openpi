| 结论（检索日：2026-10-08） | 数量 / 建议 |
|---|---|
| **已完整验证、可直接新增的模型** | **0**：没有同时完成「双基准权重与配置、非 VLA、原版推理无历史、缓存收益」全部验证的新增家族。不是证明不存在。 |
| **最接近：Diffusion Policy Transformer，brandonyang 社区版** | **1 个条件候选**。两边真实权重、完整嵌入式配置和 normalization 均已核实；`n_obs_steps=1`，100 步 DDPM，16 步 chunk；CPU 结构代理显示很大的后段开销。缺专用推理适配代码、语言编码约定和成功率记录，尚不能承诺可复现。 |
| 次优补证对象：Cocos（2025） | 论文同时有 LIBERO、MetaWorld；LIBERO 代码是当前帧 rectified flow。官方 OneDrive 链接存在，但未取得可验证的文件清单，尤其不能确认 MetaWorld 权重。 |
| WAM / 视频扩散 | 未验证到符合全部条件的新增双基准家族。Cosmos 社区 MetaWorld 文件存在，但仅 `assembly-v3`，缺模型配置；继续遵守 owner 的 LIBERO-only 特例安排。 |
| 覆盖范围 | HF 六个关键词完整翻页：合计去重 **5,014** 个仓库；按作者、架构家族交叉，另查 2024–2026 论文、GitHub、OSF、OneDrive/Drive 链接。完整记录见 [EVIDENCE.md](EVIDENCE.md)。 |

| 判定符号 | 含义 |
|---|---|
| ✓ | 本轮有直接证据通过；并不代表实机复现成功。 |
| ✗ | 有直接反证，不符合要求。 |
| ? | 缺证，**不算通过**；不能把未找到写成不存在。 |
| — | 已被前置条件排除，未继续做推理/成本认证。 |
| R1 / R2 / R3 / R4 | 双基准权重及配置 / 非 VLM、LLM 骨干 / 仅当前观测 / 取 key 后的计算主导且调用昂贵。 |

**文件交集、发布声明交集：完整家族表。** 同一家族的不同作者/微调版本合并；假交集也保留。R5 在排序时执行，不把同一家族拆成多个推荐。

| 家族 / 版本、日期 | R1：LIBERO ∩ MetaWorld | R2 | R3 | R4 | 成功率 / 决定 |
|---|---|---|---|---|---|
| **Diffusion Policy Transformer / brandonyang，2026-04-22 上传** | ✓ 两仓库均有 `model.safetensors`、含完整配置的 `metadata.pt`、norm；MW 数据标为 **ML45** | ✓ ResNet + 从头训练的 action Transformer | **? 强证据支持**：配置 1 帧；上游 sampler 无跨调用状态；专用 wrapper 未找到 | **✓ 结构代理**；真实 GPU 延迟 ? | **两个检查点均无已找到的成功率报告**。唯一值得优先补证的候选。[双仓库/代码/测量](EVIDENCE.md#dp) |
| Diffusion Policy / iFaz，2026-05 | **✗** LIBERO 有 1.112 GB 权重与配置；MW 仓库只有 `.gitattributes` | ✓ | — | — | 同属 DP，不另选；不能用仓库名称算发布。[文件](EVIDENCE.md#ifaz) |
| **BAKU，2024-06** | ✓ OSF ZIP 内两边 `baku.pt`；官方 GitHub 有配置 | ✓ ResNet18-FiLM + 小 GPT，默认确定性 head | **✗** 默认 `temporal_agg=true` 聚合跨调用的旧动作预测；观测 history 本身为 false | **✗ 结构不合适**：小单次 head，视觉编码占主要计算 | 论文 LIBERO-90 **90%**；MW **30 任务 79%**，非 MT50。关闭聚合会改变发布策略；不选。[证据](EVIDENCE.md#baku) |
| **RT-1 / BAKU 重实现，2024** | ✓ ZIP 两边 `rt1.pt`，代码配置可查 | ✓ 此 RT-1 是 FiLM 图像编码 + 从头训练 Transformer，不等于 RT-2 | **✗** 6 帧历史配方 | — | 论文 LIBERO-90 **16%** / MW30 **65%**；不选。[证据](EVIDENCE.md#baku) |
| **ACT / MT-ACT，2023–2026** | ✓ BAKU 两边 `mtact.pt`；kunhsiang 另有双边 safetensors + config | ✓ | ? raw chunk 可当前帧，具体 wrapper 各异 | **✗** owner 已排除的低成本家族；无昂贵迭代后段 | BAKU 重实现 **54% / 13%**；kunhsiang 双检查点未找到 SR。按 owner 决策排除。[文件](EVIDENCE.md#other-hf) |
| **Canonical / DiTFlow，2026** | **?** 官方声明双边 encoder+policy；所链 HF API 返回 **401**，未取得文件清单 | ✓ | **✗** LIBERO：10 帧 + goal image；MW：2 帧，state 含前一步动作 | — | 无可核实的发布检查点 SR；即便解锁文件也不合适。[代码](EVIDENCE.md#canonical) |
| **Cocos，2025-05 / NeurIPS 2025** | **?** 官方 OneDrive 无可验证清单；README 只提供 LIBERO 配方 | ✓ DINOv2/T5 独立编码 + action DiT，非 VLM 骨干 | ✓ LIBERO `To=1`；**? MW 代码未发布于所查仓库** | ? 20 步 flow 有潜力；双次视觉编码/AE 需计入 key 成本 | 论文 LIBERO 平均 **94.8%**，MW30 **74.8±0.9%**；未关联到可列出的发布文件。补证第二位。[证据](EVIDENCE.md#cocos) |
| **Cosmos Policy，2025-12 / MW 社区版 2026-07** | **? 不满足完整 R1**：两边有约 3.913 GB 权重；MW 无模型配置，仅 **assembly-v3 单任务** | ✓ 视频扩散骨干 | ? 社区配套 Cosmos runner 未核实 | ? 重后段可能合适；未测 | 官方 LIBERO **98.5%**；社区 MW base **约52% / 25 episodes**，steering 84–88% 不是 base。遵守既定特例，不新增。[文件](EVIDENCE.md#cosmos) |
| **Evo 家族：Evo-1 → EVO-Depth，2025-11 / 2026-05** | ✓ 两版本均有两边完整权重和 JSON 配置；较新 EVO-Depth 优先于 Evo-1 | **✗ InternVL VLM** | — | — | 双基准不改变 VLA 身份；不选。Evo-World（2026-10）也含 InternVL3，未确认 LIBERO release。[文件](EVIDENCE.md#other-hf) |
| **SmolVLA / UFOS 等社区派生，2025–2026** | ✓ lerobot 双边 `model.safetensors + config.json`，另有多作者副本 | **✗ SmolVLM** | — | — | 所抽查双仓 model card 未给检查点 SR；不是新增非 VLA。[文件](EVIDENCE.md#other-hf) |
| **π0 / π0.5 / π0-FAST / ALAM / UFOS 派生，2024–2026** | ✓ 双边权重；部分配置依赖代码配方 | **✗ PaliGemma 等 VLM** | — | — | ALAM 卡片报告 LIBERO **98.1%** / MW **85.0% difficulty-macro**；不选。π0.5 已在系统内。[文件](EVIDENCE.md#other-hf) |
| **OpenVLA-OFT，2025–2026** | ✓ RLinf 双边 sharded safetensors + config | **✗ LLM/VLM** | — | — | 不选；不能因 action head 并行就改判非 VLA。[文件](EVIDENCE.md#other-hf) |
| **X-VLA，2025–2026** | ✓ zhongweihong 单仓含两 benchmark 子目录：forge 权重、base/eval 配置 | **✗ Florence-2 VLM** | — | — | 社区训练轮数不等于 SR；其 README 特别区分历史/当前评测协议。[文件](EVIDENCE.md#other-hf) |
| **GR00T 家族 / N1.7 社区 MW，2026** | ? 查到 MW MT50 12.576 GB 权重 + config；LIBERO 同家族发布另见枚举，未做新双边认证 | **✗ VLM** | — | — | 不构成新增非 VLA；不改变 owner 的 N1.5 MetaWorld 微调安排。[文件](EVIDENCE.md#other-hf) |
| **CRT，2026-02** | **✗ 类别假交集**：两边 `.pth` 是图像修复器；不含完整 action policy/config | **✗ 不是机器人策略**，搭配的是 VLA | — | — | 修复后的 VLA 成功率不能当 CRT policy 成功率。[论文](https://arxiv.org/abs/2602.01158) |

**双基准论文交集：没有验证到合格双边发布的候选。** 数值是论文方法结果，不能冒充发布权重的实测。

| 家族 / 日期 | R1 | R2 | R3 | R4 | 排除依据 / 结果（链接含发布核查） |
|---|---|---|---|---|---|
| QueST，NeurIPS 2024 | ? 未找到双边权重链接；GitHub Releases 空 | ✓ | — | — | README 教训练和加载本地 checkpoint；不等于发布。[官方](https://github.com/pairlab/QueST) |
| PRISE，ICML 2024 | ? 未核实双边发布 | ✓ “LLM-style” 是 tokenization 类比 | ✗ 论文观测编码使用历史 | — | [官方](https://github.com/FrankZheng2022/PRISE)、[论文](https://arxiv.org/abs/2402.10450) |
| VQ-BeT，2024；双基准重实现见 BAKU/QueST | ? 官方现有下载示例是 Kitchen；BAKU ZIP 无 VQ-BeT | ✓ | — | — | 不能拿 Kitchen 权重填交集；确定性离散 skill head 也非首选。[官方](https://github.com/jayLEE0301/vq_bet_official) |
| STAR，ICML 2025 | **✗ 官方明确尚无 pretrained checkpoints** | ✓ | — | — | [官方](https://github.com/iLearn-Lab/ICML25-STAR) |
| MGP，ICLR 2026 | ? 无可列出的双边权重 | ✓ | ✗ long 版本使用历史全局轨迹；short 未认证 | — | 官方方法表：LIBERO90 **88.9%**、LIBERO-long **82.0%**、MW50 **63.7%**；不据此认证某个发布版本。[官方](https://github.com/Coondinator/masked_generative_policy) |
| FDP / Four-Dimensional Diffusion Policy，2025-12 / RA-L 2026 | ? README 和 Releases 未见双边发布 | ✓ | — | ? 多扩散模块可能昂贵 | [代码](https://github.com/Chaoqi-LIU/fdp)、[论文](https://arxiv.org/abs/2512.21898) |
| DiffuView，CVPR 2026 | ? 项目页仍无可用代码/权重入口；HF 无命中 | ✓ | — | ? 扩散计算主要在视觉预训练/编码侧，不能直接算可省后段 | 项目表 LIBERO-100 **92.2%**、MW50 **70.6%**。[项目](https://github.com/zkzwork/diffuview) |
| AutoSpeed / WiseActor，2026-07 | ? 已公开下载为 ALOHA Sim / 真机；未验证 LIBERO+MW 成对权重 | ✓ ACT 分支 | — | ✗ ACT 分支不合 owner 条件 | [官方](https://github.com/tinda24/autospeed) |
| LIFD，2026-09 | ? 未验证双边文件 | ✓ | **✗ recurrent scene memory** | — | 论文 LIBERO **91.6%**、MW **79.8%**；历史依赖已足够排除。[论文](https://arxiv.org/abs/2609.19796) |
| RoMAN-Flow，2026-08 | ? MW release 仍列为后续工作 | **✗ SmolVLM** | — | — | [官方](https://github.com/konnyaku28/RoMAN-Flow) |
| TriVLA，2025-07 | ? 未做双边文件认证 | **✗ Eagle2 VLM** | — | — | 视频/world branch 不改变 VLA 骨干。[论文](https://arxiv.org/abs/2507.01424) |
| FreqPolicy，NeurIPS 2025 | ? 未验证同一非 VLA 策略双边 release | ✗ 双基准不能混拼：MW 3D/flow、LIBERO OpenVLA 分支 | — | — | [论文](https://openreview.net/forum?id=NKNryrCGYn) |
| DataMIL，2025-05 / ICLR 2026 | ? 无同一视觉策略双边认证 | ✓ 算法本身非 VLA | — | ✗ MW state-only MLP 无视觉 key、且便宜 | LIBERO 的 Octo 与 MW 的小 MLP 不能拼成一个合格家族。[论文](https://arxiv.org/abs/2505.09603) |
| DILLO，2026-03 / ECCV 2026 | ✗ 辅助 steering 模块非完整策略交集 | **✗ inference 是 LLM student** | — | — | 名称有 world model 仍不符合非 LLM 约束。[论文](https://arxiv.org/abs/2603.23149) |

**条件候选 1：DP 的具体接入规格。** 来源为发布配置、安全解析的 tensor header、上游 inference code；并未运行该训练权重。

| 项目 | LIBERO / MetaWorld |
|---|---|
| 发布文件 | [LIBERO](https://huggingface.co/brandonyang/dp-libero-checkpoints/tree/main)：30k；[MetaWorld](https://huggingface.co/brandonyang/dp-metaworld-checkpoints/tree/main)：30k、85k。优先审查 MW 85k，但没有证据它质量优于 30k。 |
| benchmark 口径 | LIBERO 数据源 `physical-intelligence/libero`，实际覆盖哪些 suites 尚无 card；MW 数据源 `brandonyang/metaworld_ml45`，**不要写成已验证 MT50**。ML45 是应单独标明的标准变体。 |
| 视觉 → key | 两张同一时刻图像，96→84 crop；ResNet18 + GroupNorm + language FiLM → spatial-softmax → 各 64 维；可拼 **128 维自身视觉 key**。FiLM 后 key 与任务相关，应分任务检索。 |
| backbone / action head | 约 **28.30M** 视觉参数 + **63.54M** action Transformer；12 decoder layers、4 condition layers、width512、8 heads；768 维外部语言 embedding。不是 LLM 预训练骨干。 |
| 默认采样 | epsilon DDPM、cosine schedule、100 training timesteps / **100 inference steps**，每次新噪声；LIBERO action7 / MW action4；horizon16、execute16、`n_obs_steps=1`。 |
| stateless 核验边界 | 上游 `predict_action` 拒绝 `past_action`，每次编码当前观测并重新 sample；step 内迭代不是跨调用 state。发布 tensor 的 action head 名称/形状与上游 **全部匹配**；自定义 FiLM wrapper、语言预处理、MW wrist 输入处理仍缺代码。 |
| 可信度 / license / gating | 匿名访问可列文件、读配置，`gated=false`。真实训练产物结构完整，但**无 model card、无 checkpoint license、无双边 SR**；上游 DP 代码 MIT 不能替代权重授权。 |
| 新旧 / 家族去重 | 本次找到可成对列文件的 DP 最新上传是 2026 社区产物；它是 2023 DP 架构，不伪称新方法。iFaz 版本不能另算第二家族。 |

**缓存成本：CPU 代理，不是已加载权重的延迟，也不是 GPU 速度预测。** B=1、FP32、4 CPU threads、限定核与 `CUDA_VISIBLE_DEVICES=`；随机权重、原版 action head、按 checkpoint 形状重建 FiLM vision。命中路径取同一视觉 key，5000×128 精确内积检索；尚不含 guard/corrector 额外开销 H。[脚本及原始样本](EVIDENCE.md#cost)

| action steps | 视觉/key (ms) | 视觉之后 head+采样 (ms) | full call (ms) | hit (ms，H=0) | full : hit | 每 hit 节省 (ms，H=0) |
|---|---:|---:|---:|---:|---:|---:|
| **100（发布默认）** | 20.235 | 2578.798 | **2599.033** | **20.294** | **128.07×** | **2578.739** |
| 2（仅成本试算，未验证质量） | 20.235 | 50.352 | 70.587 | 20.294 | 3.48× | 50.293 |
| 1（仅成本试算，未验证质量） | 20.235 | 25.144 | 45.380 | 20.294 | 2.24× | 25.086 |

| 成本口径 / 风险 | 数值或处理 |
|---|---|
| FLOPs proxy | 双相机视觉 **1.197 GFLOPs**；每步 action head **≥1.469 GFLOPs**（profiler 对 fused attention 等可能漏计）；100 步 full 约 **≥148.147 GFLOPs**，迭代 head 的统计 FLOPs 占约 **99.2%**。 |
| 实际收益公式 | `T_hit = 20.294 + H`；`saved = T_tail − 0.059 − H`；系统加速仍取决于命中率和成功率。上述只计算单次 call，不等于 rollout 总加速。 |
| 不含项 | 图像读取/resize、设备传输、任务语言编码、归一化、真实 no-progress guard/library corrector。任务 embedding 可按 task 预计算，但需先确认专用 adapter 的语义。 |
| 确定的收益判断 | 默认 100 步在代码结构上高度适合缓存；真实 GPU full/hit ratio、去噪降步后的成功率以及该社区检查点的基础成功率都尚未知。 |

| 我们的层 | DP 适用性 / 具体限制 |
|---|---|
| 自身视觉 key + action-chunk replay | ✓ 插在两视角 encoder 后、action denoiser 前；按 task 分库并保留状态门控。 |
| no-progress guard | ✓ 以当前 obs/proprio 检查；是外部系统状态，不要求补填 policy KV。 |
| library corrector | ✓ 结构可接；需要各 benchmark 的动作尺度、成功轨迹和校验，成本计入 H。 |
| periodic cost knob / per-task knob descent | ✓ 可调阈值、刷新周期、commit horizon；要分别校准两套动作空间。 |
| plain step reduction | ✓ 有显式 steps；**100→1/2 的 DDPM 质量未经证明**。先做合理的步数/采样器扫描；不得套用 π0.5 flow 的低步数结论。 |
| extended commit horizon | ✓ 可把更短执行窗延至已预测的 16 步；发布默认已经 execute16。超过16没有新预测，只能重复/外推，需单列实验。 |

| 下一步优先级（每家族只一个） | 足以升级为推荐的缺口 | 当前决定 |
|---|---|---|
| **1. DP / brandonyang** | 找到/补齐原始 wrapper、语言 embedding 与 MW 相机适配；确认套件与版本；评测两个 base checkpoints；然后测真实设备 full/key/hit。 | **条件入围；当前不计“全部通过”。** 双边权重本身无需再找。 |
| **2. Cocos** | 可访问的 LIBERO+MW 权重及配置清单、MW 推理代码；确认 benchmark 是30还是50任务；计入 DINO/AE 成本。 | 等待发布证据；不能拿论文表格当权重。 |
| 其他 | Canonical 的历史/goal 条件、BAKU/RT-1 的跨调用依赖、ACT 成本、VLA 骨干都不是补一个下载链接能解决的。 | 排除；不凑第二个合格模型。 |
