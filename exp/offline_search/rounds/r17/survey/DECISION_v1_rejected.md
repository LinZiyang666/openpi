# 下一轮策略迁移与基准选择

2026-10-08；供负责人批准、工程启动。事实、原始计时和核查范围见 [EVIDENCE.md](EVIDENCE.md)。

**建议批准三个新策略：OpenVLA-OFT、Diffusion Policy、π0；新增基准选 RoboCasa365 v1.0.1 的官方 Atomic-Seen 18 任务。** π0.5 保留为跨基准对照，不占三个新策略名额。可追加一小组 LIBERO-PRO 位置扰动。

这组选择分别检验：**没有迭代动作头时，五层系统是否仍有效；不能安全降到一步时，检索能否保住性能；同一模型跨 LIBERO 与厨房场景时，收益能否迁移。** 三个新策略中的 π0、Diffusion Policy 都有 RoboCasa365 公开权重，满足“大多数支持”；π0.5 也有基准作者发布的权重和评测实现。两个新策略有公共 LIBERO 权重；Diffusion Policy 是为非 VLA 与 ε 扩散头保留的明确例外，不先花预算训练 LIBERO 权重。

**轻量约束的边界：批准的是 18 个原子任务与按需取图的执行方式，不是完整 50 任务排行榜，更不是全部 365 任务。** 原生逐步软件渲染偏重；本次 CPU 探针验证了必要观测点渲染的可行性。完整 50 任务的长组合任务暂不进入数万局搜索。首轮先做约 1.9 万局诊断批，预算上限 **120 GPU 小时、1,400 CPU 核小时**；未通过下述速度和正确性准入条件，就不启动批量运行。

**1. 三个策略的具体选择**

| 策略与作用 | 自有视觉检索键的接入点 | 动作头、默认采样与动作长度 | 单次调用成本：已知事实与预算假设 |
|---|---|---|---|
| **OpenVLA-OFT，7B；优先验证系统独立于减步的价值** | 两路相机各自的 DINOv2 + SigLIP patch 特征，在 vision_backbone 输出、进入 projector/LLM 之前取出；保留空间池化，重新拟合 PCA 与任务度量 | 公开 LIBERO 配置为**并行连续 L1 回归**；一次前向，无去噪 k。预测 **8**、默认执行 **8**；不是原始 OpenVLA 的逐 token 自回归，也不是仓库里的可选 diffusion 变体 | 原论文对应双相机 + 本体状态版本：**112 ms/块，A100，100 次查询**。这是外部测量，不是本机 4090/H100 成绩；首轮按 **0.10–0.30 s/块**预留。[论文表 II](https://arxiv.org/html/2502.19645v1#S5.T2)、[实现](https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py) |
| **Diffusion Policy，RoboCasa 官方 Transformer 版；非 VLA 主实验** | 三相机 ResNet18/FiLM 视觉编码器的空间特征，来自 policy.obs_encoder；保留**连续两帧**观测历史。CLIP 语言向量按指令缓存，不能把整个视觉编码器省略掉 | ε 预测，12 层、512 宽度的 Transformer。随权重附带配置是 **DDPM-100**；预测 horizon **10**，历史 **2**，默认执行 **8**。采样输出从索引 1 开始，可用未来动作最多 **9**，不是可执行 10/16/64 步 | **这一目标权重没有已核实的本机延迟。** 旧实验是另一个 CNN U-Net：4090 空卡 DDPM-100 为 1.305–1.363 s，DDIM-10 为 0.139–0.155 s，不能移作此 Transformer 的成绩。暂按 **0.5–2.0 s/满步块**编预算，加载后先测；超出则重算预算。[权重配置](https://huggingface.co/robocasa/robocasa365_checkpoints/blob/main/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/.hydra/config.yaml)、[取动作代码](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py) |
| **π0；成本低的跨基准连接策略** | PaliGemma 的 SigLIP So400m/14 图像塔输出；在 VLM 前缀之前取空间特征。与现有接口接近，但重新建库，不借用 π0.5 的特征、PCA 或动作 | flow matching / Euler，默认 **10** 次动作专家前向；所选标准 π0 配置预测 **50**。LIBERO 转换配置默认执行 **10**；RoboCasa 官方客户端默认执行 **5**。扩展执行段有实质空间 | **所选 LIBERO / RoboCasa 权重尚无本机计时。** 文献中另一套 ALOHA π0 配置为 86 ms/次，不能冒充这两个检查点的速度；暂按 **0.06–0.20 s/块**。须分别量视觉、VLM 前缀与 50 步动作块的头开销。[π0 源码](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)、[配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py)、[文献表 III](https://arxiv.org/html/2502.19645v1#S6.T3) |

这里“非 VLA”指没有预训练 VLM 主干、从视觉编码器和动作扩散网络学习的传统 visuomotor policy。所选 Diffusion Policy **仍有语言条件**；不能把“非 VLA”写成“无语言输入”。它使用 CLIP 文本特征调制 ResNet，而不是通过大语言模型推理。[基准论文附录 G](https://arxiv.org/html/2603.04356v1#A7)

**检查点与许可：工程按下表拿权重，不用同家族的其他版本替代。** 本次只验证匿名元数据、文件清单、配置和小文件；没有下载权重或执行 GPU 推理。

| 策略 | LIBERO Spatial / Object / Goal / Long | RoboCasa365 | 许可与访问 |
|---|---|---|---|
| OpenVLA-OFT | 官方分别发布：[Spatial](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial)、[Object](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object)、[Goal](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-goal)、[Long](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-10)。四仓均已核到主权重、动作头和本体投影文件 | **未核到作者发布的对应权重，本轮不承诺迁移到该基准** | 代码与模型卡标 MIT；HF ungated。保留基础 Llama 模型的上游条款与归属说明；不要把仓库标签理解为抹去上游条款 |
| Diffusion Policy | **未核到可直接采用的作者 LIBERO 四套权重**；旧 robomimic 权重不可当作 LIBERO 权重 | 官方 [epoch=0500 检查点](https://huggingface.co/robocasa/robocasa365_checkpoints/blob/main/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/checkpoints/epoch%3D0500-test_mean_score%3D-1.000.ckpt)，供 Atomic / Composite-Seen / Composite-Unseen 共用；首轮只评 Atomic | DP 代码 MIT；基准权重仓标 Apache-2.0、ungated。加载随权重的 EMA、normalizer 与完整配置 |
| π0 | PI 公共存储中确有 **pi0_libero** 权重分片及 [归一化统计](https://storage.googleapis.com/openpi-assets/checkpoints/pi0_libero/assets/physical-intelligence/libero/norm_stats.json)，[对象清单](https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi0_libero/&maxResults=15)。另有 [LeRobot 发布版本](https://huggingface.co/lerobot/pi0_libero_base)，一份 LIBERO 权重，不是四套独立训练。首选 PI 原生权重，逐套冒烟确认 | 官方 [π0 multi-task / 75000](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000)，三类任务共用 | OpenPI 代码 Apache-2.0；LeRobot π0 权重卡标 Gemma；RoboCasa 权重仓标 Apache-2.0。公开存储无需 token，HF ungated；代码、发布权重及底座条款分开保留 |
| π0.5，既有对照 | 官方 [pi05_libero 说明](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero)：联合四套，预测 10、默认 10 次 flow 前向；现有系统执行 10 | 官方 [π0.5 multi-task / 75000](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi05_pretrain_human300/multitask_learning/75000)，[基准作者提交记录](https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions/pi05_2026-04-02.json)；预测 **50**、默认 **10** 次 flow、客户端执行 **5** | 使用与 π0 相同的代码/权重/底座许可区分。HF 仓 ungated。不是仅有论文分数的“支持” |

π0 的 LeRobot 卡片文字较泛，processor 文件也不能单独证明加载后归一化正确，因此不把它的文件可见性当成闭环复现。原生 PI 权重的归一化文件已取回；工程先按原生配置恢复并做动作/图像合同检查。以上检查点身份与修订号保存在证据文档。

**2. 五层全部迁移，哪些层最可能有收益**

以下是待验证的优先级，不是尚未运行的性能结论。所有策略都接入五层；某层最终选择关闭，也必须有消融证据。

| 层 / 额外基线 | OpenVLA-OFT | Diffusion Policy | π0（同时用 π0.5 做基准对照） |
|---|---|---|---|
| 视觉特征检索缓存 | **高**：命中能跳过 7B LLM 与动作读出；但双视觉塔构成较高成本地板 | **高**：视觉编码后可以跳过大量扩散前向；特征偏任务化、跨场景近邻可能较差 | **中高**：跳过 VLM 前缀与动作专家；接口迁移成本最低 |
| 无进展守卫 | **中高**：重复拿同一块动作时及时重新调用策略 | **高**：精细接触容易在缓存轨迹上停滞；需排除两帧历史错位造成的假停滞 | **高于饱和 LIBERO 的预期**：厨房场景的接触、开关和抽屉需要持续纠偏 |
| 库内修正器 | **中**：局部位置残差可能有用，但无迭代头可“修复”不代表动作残差可不受约束 | **高风险、高信息量**：可能补偿近邻偏差；也可能平均不同抓取模式。夹爪与控制模式离散处理，旋转做合法投影 | **中高**：在不改变教师的前提下修正局部偏差；跨布局外推不可信时回退 |
| 周期调用开销旋钮 | **高**：省掉完整 LLM 调用，直接展示五层系统的独立价值 | **高**：保留必要的正常去噪调用，避免用不可用的一步教师救援 | **中高**：即使 k=1 无损，缓存仍能省前缀；厨房中可能必须保留 k≥5 的回退 |
| 逐任务旋钮下降 | **中**：先看任务差异，防止高成功率掩盖个别任务倒退 | **高**：不同任务的少步阈值、库覆盖与接触精度差异大 | **高**：长块执行与精细操作的最优调用频率不同 |
| 单纯减步 | **不适用**；没有去噪循环，不编造 k=1 对照 | **核心竞争基线**：DDPM-100 原生端点，另跑正确 DDIM 网格的 20/10/4/2/1。旧 U-Net 的一步失败是线索，不能当作新 Transformer 已失败 | **必须先测** 10/5/2/1。π0 未证实一步无损；π0.5 的 LIBERO 结果也不能外推到厨房 |
| 扩大执行段 | 扫 1/4/8；官方已执行全部 8，**相对原生没有再扩段的余地**；不重复/填充动作伪造 16 步 | 扫 4/8/9；最多 9。第 9 个预测位置能否闭环使用需实测 | LIBERO 扫 10/20/50，RoboCasa 扫 5/10/20/50；相机观测与守卫的决策频率随执行段一起记账 |

内部证据支持选一个“少步困难”的模型和一个非迭代模型，但不支持承诺具体收益：π0.5 厨房 13 任务、执行 5 步的历史结果为 k=1 **39.1%** 对满步 **55.7%**，k=5 **55.4%**；旧 ε-DP 的 DDIM-1 表现很差，而重训 x₀ 头的一步平坦常伴随更差的满步锚点。LIBERO 上此前的完整系统终表曾根据测试结果调档，不能作为新实验的独立无损证明。[内部核查与勘误](EVIDENCE.md#内部结果核查)

没有可用视觉塔的策略仍可尝试状态/观测历史检索，并沿用守卫、修正器、调用旋钮与逐任务下降；这属于**状态键变体**，不满足“策略自有视觉特征”的同一实验定义。也可附加冻结图像编码器，但须把额外成本算入；状态不足以区分被遮挡物体和目标时，不能期待同等表现。

**3. 新基准如何满足近期、影响力、支持和轻量要求**

RoboCasa365 于 2026 年发布，ICLR 2026，已有基准作者维护的公开榜单与多个独立团队提交。它带来 LIBERO 缺少的厨房布局、物体、接触与移动底座变化；π0.5 / π0 / Diffusion Policy 都有可核查发布物。相比只换一个 LIBERO 扰动集，它能更好检验检索特征与任务进度的迁移。[官网](https://robocasa.ai/)、[排行榜](https://robocasa.ai/leaderboard.html)、[算法支持](https://github.com/robocasa/robocasa/blob/main/docs/benchmarking/policy_learning_algorithms.md)

采用官方 **Atomic-Seen 18 任务**，全量任务名与 450–1,050 步时限见 [证据](EVIDENCE.md#cpu-仿真测量)。固定 v1.0.1，主评测走 multi-task 检查点对应的 **pretrain 场景分布**。不能把 target 后训练的 GR00T 成绩、旧的单布局 13 任务、RoboCasa-2024 或完整 50 任务均分混入同一表。历史 k=1 损失只用于提出假设；新 π0.5 基线重新跑。

本机已有安装，**本次没有安装任何 benchmark 或新依赖**。CPU 只绑定 18–25、62–69，CUDA 隐藏，OSMesa/llvmpipe 单线程渲染。采用三路 **256×256**，不以降分辨率换取“轻量”。只跳过在块执行期间不会消费的图像，物理、控制频率、奖励、成功判定和结束时限都照常推进；DP 还须渲染查询前一帧，不能拿上一次查询图像冒充历史。

| CPU 探针 | 单步或时限成本 | 能说明什么 |
|---|---|---|
| OpenDrawer，pretrain 场景，原生 256² 逐步取图 | 587 ms/步；按 750 步外推 **454 s/局** | 原生全帧 OSMesa 不轻；此前 128² 探针也要 327 s/局 |
| OpenDrawer，256²，每 5 步取图，实际计时 750 个步骤 | 加一次 reset **108 s/局** | 一次完整时限的零动作成本，未含策略推理，也不是成功轨迹 |
| OpenDrawer / CoffeeSetupMug，**pretrain** 随机场景，256²，每 5 步取图 | 短探针外推分别 **132 / 131 s/局** | 与拟采用的场景分布一致，但每任务仅一个 seed |
| PickPlaceCounterToStove，pretrain，256²，每 8 步取连续 2 帧 | 短探针外推 **141 s/局** | 显式计入 DP 的两帧观测；GL 元数据查询有一次告警，随后仿真和取图循环完成，详见证据 |
| 图像/物理一致性 | 24 步回放、6 个保留时间点、18 张图：最终实现图像哈希、qpos、qvel、奖励逐位一致 | 仅证明一个任务的一段轨迹；批量前必须覆盖全部 18 任务 |

预算按 **120–250 CPU 秒/局**、约 **2.8–3.2 GiB/进程**估算：一万局约 **333–694 核小时**；64 个有效单核槽、按 65% 吞吐效率约 **8–17 小时纯仿真墙钟**，再与 GPU 服务瓶颈取最大值。64 进程 RSS 约 180–210 GiB，另留 reset 峰值和库内存；内存不足应降并发，预算按实际槽数重算。这是本地小样本外推，**不是 timan107 的实测吞吐承诺**。RoboTwin 的 160 局/小时来自不同并发和渲染条件，只用来说明不能再选那种部署负担，不宣称严格倍数加速。

准入条件：在 timan107 先做 16→32→64 worker、全 18 任务的无模型回放；按原生分辨率保留观测，验证查询帧与物理一致。稳定吞吐须达到 **≥800 局/小时**、预计一万局纯仿真 ≤12.5 小时、RSS 峰值 <75% 可用总内存；先固定这些标准，再看结果。未达到就停止批量开销，不靠删掉慢任务或缩短时限过关。届时采用已选的 LIBERO-PRO 小批验证继续模型移植，RoboCasa 大跑暂缓；这不会被写成已经满足新增独立场景验证。

可选鲁棒性附加项：**LIBERO-PRO 的温和位置扰动**，π0 / OpenVLA-OFT / π0.5 复用原权重与干净库，固定 10 个 Spatial 任务、两档官方位置强度、每档 20 个新起点，比较纯推理和完整系统，共 **2,400 局**。它的官方 README 列出 π0.5，底层仍是 LIBERO；预计仿真级别接近现有 LIBERO，但本次未安装、未计时该 fork；代码 MIT、数据按其 README 标 CC BY 4.0，未见软件访问门禁。不要从“极强扰动下大家都失败”推断加速有效，也不要将它包装成第二个独立环境。[官方实现](https://github.com/Zxy-MLlab/LIBERO-PRO)

**4. 第一批工程任务与实验协议**

1. **冻结接口与身份。** 每个模型输出自有视觉键、动作块、可复用前缀及分阶段计时。把权重修订、预处理、相机、动作单位/归一化、历史长度、预测长度 C、执行长度 H、求解器网格和实际 NFE 写进 manifest。OFT 使用完整双塔与原生中心裁剪；DP 使用完整两帧和 EMA；π0 优先原生 GCS 权重。键的通道数不照抄旧模型；按空间池化→PCA64→本体状态→任务度量重新拟合，命中/未命中共享本次已经算出的视觉特征。
2. **先建立公平基线。** 模型载入后用 batch 1 和拟部署 batch 做 30 次预热、200 次计时，记录 p50/p95、视觉/前缀/头/检索/通信、峰值显存及有效 NFE。对动作块前后处理做回放核对。保留原生满步/原生 H，以及开发集选出的最佳“仅减步 + 扩段”两条基线；所有臂使用相同仿真取图优化。不将 OSMesa 优化收益算成五层系统收益。
3. **诊断批范围固定。** 新模型先跑 π0 与 OFT 的 LIBERO-Spatial 全 10 任务；RoboCasa 先用 OpenDrawer、CloseFridge、CoffeeSetupMug、PickPlaceSinkToCounter、SlideDishwasherRack、TurnOnSinkFaucet 六任务，跑 π0、π0.5、DP。合计 **38 个模型×任务单元**。六任务由任务类型及既有失败线索预先选定，只用于诊断，不称为官方 18 任务结果。进入正式发布批后必须扩到完整 18 与 LIBERO-Long。
4. **建库不借测试答案。** 每单元最多 50 次满步、原生 H 的独立教师 rollout，目标最多保留 20 条成功轨迹，形成嵌套 5/20 条库。必须存满 C 的教师预测块，而不只存已经执行的前 H 步；全部失败与采集成本也记账。少于 5 条成功的任务先标为覆盖不足并回退纯推理，不额外采样到好看为止、不从测试起点补库。DP 公开 Atomic 成功率较低，库覆盖不足是主要风险；正式报告必须给出所有任务结果及纯推理回退比例。
5. **开发集选参数、测试集只评估。** 库、修正器训练、选档开发集、最终测试集的 episode/seed 分离；没有“留本集但仍参与拟合”的捷径。先在开发集确定 k 与 H，再调周期调用频率；每任务依次降低调用频率；相对同起点纯推理开发基线最多净少1局（20局的5个百分点），首次超出即停止，候选表随后冻结。库内修正器仍使用原有 RFF + 岭回归思路；重新拟合而非迁移权重。周期档以每次缓存决策调用教师的目标比例 {1/2、1/4、1/8、0} 起步，开发阶段仍有无进展守卫；顶部回退为每次正常调用。守卫的进度阈值按 H 换算，失败时调用开发集验证可用的 k，**最高回退必须允许原生满步**。
6. **完整消融与计费。** 配对比较：纯缓存、缓存+守卫、缓存+守卫+修正器、再加统一周期旋钮、最后逐任务下降；另保留纯缓存+修正器用于分清修正器与守卫作用。记录成功率、每任务最差损失、教师调用率、守卫触发、实际执行步数、检索/修正延迟、总 GPU 秒、CPU 核秒和墙钟，同时报告每成功局成本，防止靠提前失败省算力。离线采集与编码成本单列；回本局数 = 离线 GPU 秒 / 每局节省 GPU 秒，分母≤0时明确没有回本点。计算成本按“视觉 + 命中处理 + 未命中剩余阶段”累加；不套用 π0.5 的视觉成本系数、不用 k/K 冒充时间。

诊断批的统计决策：正式配对测试每单元 50 个新起点；对任务等权宏平均做配对 bootstrap，要求完整系统相对**两条基线**的成功率差 95% 区间下界 ≥−3 个百分点，并且相对最佳无缓存基线节省 **≥25% GPU 秒/局**。这是继续扩大的标准，不是每个任务的无损保证。测试后不得改档再引用同一测试集；若需要修改，记录为开发发现，另开新测试池。DP 若满步宏成功率 <20% 或多数任务库覆盖不足，先报告“锚点/覆盖不足，不能验证广泛无损”，不在近零成功率上宣布加速成立。

| 首批工作 | 最大局数 / 测量范围 | 单卡 GPU 小时估算 | CPU 核小时估算 |
|---|---:|---:|---:|
| 适配、计时、全任务仿真准入 | 约 200 冒烟局 + 无模型回放 | 4 | 20–40 |
| 纯减步 / 扩段开发扫描 | π0 LIBERO 6 档、OFT 3 档；厨房 π0/π0.5 各 7 档、DP 8 档；各 20 起点，合计 4,440 局 | 10–25 | 110–220 |
| 满步教师采集与建库 | ≤1,900 局 | 3–15 | 45–110 |
| 五层系统的四档开发搜索 | ≤3,040 局，首次失败停止 | 3–10 | 65–150 |
| 三主臂冻结测试 | 38×50×3 = 5,700 局 | 10–35 | 120–280 |
| 针对层作用的五条额外消融 | 38×20×5 = 3,800 局；预先固定测试池的前20起点；描述性结果，不用来重选主表 | 6–15 | 80–180 |
| 余量、重试、特征/修正器拟合 | — | 约 4 | 100–190 |
| **批准上限** | **约 18,880 局 + 冒烟** | **120** | **1,400** |

扫描不做 k×H 全排列：先在原生 H 上测四档 flow 或六档 DP 采样，再把开发集最佳可用采样配置用于额外 H（π0 LIBERO 20/50，厨房 10/20/50；DP 4/9）。OFT 测 H=1/4/8。上表是资源预算，不是测得的吞吐；GPU 主敏感项是 DP 的 DDPM-100 延迟与失败满时限轨迹。用公式“局数 × 每局调用数 × 实测每次服务成本”在适配阶段替换预算假设。单 GPU 顺序部署模型，复用约 64 个 CPU worker；无需同时装载三个模型。超过上限先交付诊断结果和实测预算，不能默默扩大。通过后，正式扩到 LIBERO-Long 与 RoboCasa 全 18 原子任务，另预留约 40–80 GPU 小时、500–1,000 核小时，按诊断批实测速率重新核算；可选 PRO 再预留 5–10 GPU 小时、20–60 核小时。

**5. 暂不选择的候选**

| 候选 | 本轮不选的理由 |
|---|---|
| GR00T N1.7 | 官方 LIBERO 四套权重确实存在，所读 Spatial 配置是 4 步、预测 40；但不能把 N1.6 的 RoboCasa365 权重当作 N1.7 支持；当前更需要非迭代/非 VLA 证据 |
| GR00T N1.6 | 基准团队的 RoboCasa365 提交及公共权重已核实；本次未核到同等级官方 LIBERO 四套，且与既有 GR00T 实验的新增信息较少 |
| X-VLA | 确有公开 LIBERO 权重，flow 10、块 30；是后备小 VLA 首选，但未核到 RoboCasa365 权重，本轮唯一额外架构名额给非迭代 OFT |
| FLOWER / SmolVLA | 公共 LIBERO 发布线索成立；当前不足以同时改善新基准覆盖与动作头多样性；不因参数小就先假设缓存更划算 |
| Cosmos Policy | LIBERO 权重确有，但 RoboCasa 发布是 2024 版；所谓 k=1 实为两次网络前向，且视觉 VAE 的检索质量未验证 |
| X-WAM | 已验证的是 RoboCasa-2024 / RoboTwin，不能改写成 RoboCasa365；视频模型前缀与世界动作耦合使首轮适配较贵 |
| RDT-1B / MDT / ACT / VQ-BeT | 有视觉能力，均符合理论迁移范围；但本次没有核到能比所选 DP 更直接满足公共 LIBERO + 选定新基准组合的发布物 |
| VLA-Adapter / InternVLA-M1 | LIBERO 候选有价值；非迭代对照先采用实现、分套权重和延迟依据更清楚的 OFT |
| Xiaomi-Robotics-0 / 1、其他新发布模型 | 0 的 LIBERO 与 1 的 RoboCasa365 不算同一模型跨基准；更换底座的新增工作不优于本轮三种机制覆盖 |
| RoboTwin 2.0 / BEHAVIOR-1K | 模拟、渲染或规划部署负担不符本轮大量 rollout 约束 |
| RoboCasa365 完整 50 任务 | 近期且有影响力，但长组合任务更重、基线成功率也低；本轮只批准官方 18 原子任务范围，不宣称完整排行榜成绩 |
| MetaWorld / robomimic | 便宜且有用，可作接口冒烟；不满足“新增且近期”的主要动机，MetaWorld+ 的版本兼容不能仅凭旧 MetaWorld 权重推定 |
| CALVIN / SimplerEnv / ManiSkill3 / VLABench | 本次未完成“所选多数模型 + π0.5 对应权重/接口 + 轻量计时”的完整证据链，不能以未验证支持替代现成方案 |
| LIBERO-Plus | 可作后续鲁棒性研究；官方主表没有 π0.5，且存在扰动文件名污染指令的公开报告，先用 PRO 的窄位置扰动 |

需要负责人批准的是：**上述三个策略、18 原子任务的基准范围，以及先投入有上限的诊断批。** 首个交付物是可复用策略适配器、冻结的模型/环境合同、真实阶段成本表与纯减步/扩段曲线；随后才是五层系统收益。最可能改变决定的事实是 DP 目标权重实际速度/成功率，以及 timan107 的 64 worker 内存和吞吐，已把它们放在昂贵实验之前。
