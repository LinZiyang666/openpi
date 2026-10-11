# 非 VLA 策略与新增基准：重新选型

2026-10-08。替代旧决策；旧版仅保存在 [DECISION_v1_rejected.md](DECISION_v1_rejected.md)。证据、版本、文件清单和计时方法见 [EVIDENCE.md](EVIDENCE.md)。

**建议批准：ACT、DP3、图像 Diffusion Policy（ε 预测、U-Net）；新增 MetaWorld，保留 LIBERO。先做权重复现和六任务诊断，通过后扩大。** 三者分别覆盖 CVAE/Transformer 动作块、点云条件扩散、二维图像条件扩散；均没有 VLM/LLM 策略主干。最后两者虽然都叫 diffusion，但观测表示、预测参数化和减步风险不同。

**批准范围应是先复现、后扩展的方案。** 已核实 ACT 的 MT50-v3 权重与逐任务成绩、DP3 的 12 个 MetaWorld-v2 任务权重；没有核实到同时具备公开文件、匹配配置和可信成绩的二维 ε-DP MetaWorld 权重。按 owner 最新允许的最后手段，为它保留小规模自训预算。DP3 的 v2→v3 兼容、ACT 的相机约定也要先验收。不能把这些缺口写成“已支持”。

## 为什么选 MetaWorld

- **缓存覆盖条件合适。** 同一任务保持 Sawyer、桌面、物体种类、几何和相机不变；变化主要是初始位姿/目标位置，不换厨房、物体实例或布局风格。例如 button-press-wall 的物体位置范围是 10×5 cm，dial-turn 是 20×10 cm；assembly 的螺母初始位置固定，peg 目标在 20×10 cm 范围变化。这仍需要覆盖，不能称完全固定。按任务独立建库，保留原始随机范围，不用锁死一个起点来制造命中率。[任务源码](https://github.com/Farama-Foundation/Metaworld/tree/main/metaworld/envs)
- **有影响力，且有较新的维护版本。** 原 benchmark 是 2019/2020 年，不冒充新任务集；2025 年 Meta-World+ 做了标准化与可复现性更新。本次目标为 Farama MT50-v3，旧权重按原版本先复现，再验收迁移；不把 v2/v3 的成绩合并成一张排名表。[Meta-World+](https://arxiv.org/abs/2505.11289)
- **π0.5 支持已落实。** [RLinf-Pi05-MetaWorld-SFT](https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT/tree/main) 实际列出 `model.safetensors`、`metadata.pt`、`lerobot/metaworld_mt50/norm_stats.json`。RLinf 公布四难度组平均 43.8%；本项目历史 50×20 局按任务平均 full=58.8%、k=2=60.4%、k=1=53.7%。两种平均不能混用。该仓免 gating，但未单列权重许可证，仍须沿用其上游条款核对；它是已有 VLA 对照。[RLinf 结果与协议](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/metaworld.html)
- **CPU 成本满足筛选目标。** 本机 CPU-only、OSMesa/llvmpipe 实测物理步约 0.4–0.9 ms；128² 单视角 RGB 约 0.17–0.18 s。按 200 步、每 5 步观察一次估计约 **7–9 秒/局**；同机 LIBERO 两相机 256²、200 步、每 5 步观察约 **24 秒/局**。这不是含策略推理的真实成功轨迹计时，也不是 timan107 的 64-worker 实测。正式成本须保留 checkpoint 的原始分辨率和观测历史；详见证据中的完整表。

**首批六任务固定为** assembly、basketball、bin-picking、button-press-wall、coffee-push、dial-turn；覆盖抓取、接触、精确对齐和简单重复运动，且均有 DP3 文件。它们是诊断子集，不叫完整 MT50 分数。最终 MT50 要补齐 DP3 剩余任务权重/训练；首批不为凑齐 50 任务而默认训练 38 个模型。

## 三个模型的实施决定

`C` 是网络预测窗口，`H` 是每次执行的未来动作数；一次 call 是“观测编码＋动作生成”，不含仿真。下表的 GPU 毫秒范围是**单卡 batch=1 的预算假设，尚未实测这些目标 checkpoint**，工程第一步分段 profiling 后替换。

| 模型与选择理由 | 自己的视觉特征 → 检索键 | 头、默认步数、C/H | 每 call 预算与权重状态 |
|---|---|---|---|
| **ACT**：无迭代头，检验系统是否能超越 diffusion；也是低剩余开销对照 | 两路 ResNet-18 的末端空间特征，固定空间池化后 PCA64；拼 proprio。任务单独分库，不能只拿全局平均丢掉空间信息 | Transformer/CVAE；推理固定潜变量、一次解码，无 k。RoboHP 配置 **C=100、H=100**，无 temporal ensemble；4 层 encoder、1 层 decoder | 约 **10–50 ms**，512² 双视角可能接近上端。[MetaWorld 权重](https://huggingface.co/RoboHP/metaworld_mt50_act/tree/main)：模型＋config＋train_config 都在；MIT、免 gating；50 任务各 25 局。模型卡“73.1%”是四组平均；重算 50 行任务平均 **77.52%**。LIBERO 可先试 [Deepkar](https://huggingface.co/Deepkar/libero-test-act/tree/main)，但目前只有单种子成功线索，不能当完整复现。 |
| **DP3**：3D 感知，廉价 PointNet 与较重扩散头，最适合检验跨模态检索 | 深度反投影、裁剪、FPS 得 512×3 点；PointNet 的 **64 维**输出，保留两帧历史＋9 维末端/指尖状态；单独拟合度量 | 条件 1D U-Net，**x₀/sample 预测、DDIM 10 步**；C=4、obs=2、**H=3**，不是“可执行 4 步” | 约 **60–200 ms**，不含 CPU 点云前处理。[社区权重](https://huggingface.co/chenhaojun/3D-Diffusion-Policy-ckpt/tree/main) 有 12 个 MetaWorld 任务；assembly eval=100%、button-wall=95%。配置训练集上限 90 episodes，不能冒充原论文 10-demo 成绩。代码 MIT；权重仓未声明独立许可证，公开无 gating。未找到可验收的 LIBERO 成套权重，先训练首批任务。 |
| **图像 ε-DP**：与 x₀-DP3 区分；重点验证“减步掉点时，完整缓存能否继续省” | 原生截断 ResNet-18＋空间池化特征，经 PCA64＋proprio；不借用现有 VLA 的视觉塔 | 使用已发表的 [QueST 中 DP baseline](https://github.com/pairlab/QueST) 配方：ε-U-Net、训练 100 noise levels、**DDIM 10 步**。基础 yaml C=16；发布 LIBERO 训练脚本覆盖为 **C=32、H=2**；MetaWorld 诊断也选此值，明确这是配置选择而非已核论文默认 | 约 **50–160 ms**（10 步）。MetaWorld 未核实合格现成权重，**这是唯一计划中的新增基准自训主项**。LIBERO 的 [ttotmoon](https://huggingface.co/ttotmoon/diffusion-libero-v3/tree/main) 确有模型/normalizer，但没有成绩，且是另一配方：DDPM100、C16/H8、obs2，单call预算约0.5–1.6秒。只作权重验收候选；不能拿它替代已发表 baseline 的复现。代码 MIT，所列 LeRobot 权重模型卡 Apache-2.0、无 gating。 |

LIBERO 从 Spatial 六任务诊断扩展到完整 Spatial/Long，模型与任务逐个注明权重来源；**ACT/ε-DP 的未评估社区权重先验收，失败才自训；DP3 明确需自训。** 用公共 demonstration，重放状态生成 DP3 深度/点云，不靠 GPU 渲染。对采用已发表复现配方的模型，先在原数据量、版本、horizon 下使 SR 距对应论文/发布结果不超过 5 个百分点；若原发布只测 20–25 局，补到至少 100 局并看区间。新六任务训练或 v3 迁移是新设置，不能直接宣称“复现论文总分”。

自训为何仍值得：已有二维 MetaWorld DP 上传中，一个空仓、一个匿名访问 401、一个只有权重和统计量而无完整配置/成绩；不能省掉验收。MoDE 虽有可信 LIBERO 权重，却没有已核实的 MetaWorld 权重/复现配方，仍需新移植；Cosmos 的 MetaWorld release 仅明确 assembly，公开仓当前缺对应 runner。选小型 ε-DP 的训练和复现风险更可控，也提供现有模型缺少的减步失败对照。

## 五层都移植，但收益不预设

| 层/附加优化 | ACT | DP3 | 图像 ε-DP |
|---|---|---|---|
| 视觉检索缓存 | 能接；命中仍需 ResNet，且 H100 原生调用少，净收益可能很小 | PointNet 后命中可绕过 U-Net；CPU 深度/FPS 仍要计费 | 命中绕过完整去噪；最有希望超过纯减步 |
| 无进展守卫 | 跟踪检索轨迹进度＋夹爪/末端位移，防长块卡住 | 接触、插入阶段重点；不能把静止抓稳误判成失败 | 防缓存重复推/抓；召回要独立测 |
| 库内修正器 | 用实际执行的连续动作块学残差，注意抓放边界 | 用同一坐标系拟合局部动作残差，不直接混合点云 | 修正缓存误差，**不能假设能修复 k=1 生成头崩溃** |
| 定期调用旋钮 | 必须相对原生 H100 计费，别强制 H10 后制造加速 | 由任务决定何时重调 10/少步策略 | 允许少量高质量 10 步调用，比每次 1 步可靠 |
| 逐任务旋钮下降 | 低成本策略可能保留全调用档，也是有效结论 | 易重复任务可激进，精确对齐保守 | 按任务同时选择可用 k、H 和调用频率 |
| 纯减步 | 不适用 | 10→4→2→1；x₀ 使 1 步有潜力，**尚无此 checkpoint 的无损证据** | 10→4→2→1；历史 robomimic ε-DP 在少步明显退化，必须在新任务重测 |
| 延长执行 | H10/25/50/100 仅作诊断；默认已到 C100，超过 100 只能用库内长轨迹，不能直接改网络输出 | 原生最多 3 个未来动作；比较库轨迹提交 3/6/9/12。若要网络 C16，需单独训练并列预算 | 原生 H2→4→8→16→32 有真实空间；与缓存分开消融 |

每个模型重新拟合 PCA、距离阈值、进度守卫和修正器；每次命中仍计入视觉编码、检索、修正、仿真与图像成本。无可用视觉头的策略可改用状态/历史编码做检索，但那是**状态检索变体**，不能算本次视觉五层的直接迁移。

## 工程起跑与预算

1. **先复现，不先建大库。** 固定 repo/权重 SHA、软件包、相机翻转、动作/状态坐标、normalizer、成功判定、随机初始化集合。ACT 的配置要求两路 512² 图像，但卡片只明确 corner4 且说需要翻图/动作、eval 代码仍 WIP；其链接的评测 fork 当前返回 404；必须恢复第二路及变换，不能猜。DP3 原 wrapper 是 v2/mujoco-py，FPS 写死 `.cuda()`；改为 CPU FPS 后做几何与采样一致性检查，不能换成随机采样。bin-picking-v3 确实改了初始范围，不能只改环境名。给协议修复 **1–2 工程日、4–8 GPU 小时**验收额度；不过关不做“加速成功”结论。
2. **小训练仅在需要时开。** 六任务/单 seed：二维 DP 每个 benchmark 约 6–18 GPU 小时；ACT LIBERO 约 4–12；DP3 LIBERO 每任务约 2–6，总 12–36；DP3 v3 若需补训再留 12–24 GPU 小时。按两套 LIBERO 社区权重均需重训的情况，合计 **28–84 GPU 小时，若补训 DP3-v3 则 40–108**，属于估算；先测 1,000 updates，用实际迭代耗时与总 updates 重算。完整 45/50-task 或 LIBERO-90 复现另需约 **20–80 GPU 小时/模型/seed**，不混入六任务预算，也不现在启动。
3. **先独立测 k 与 H，再接五层。** 两个 benchmark 各六任务、三个模型；每任务库 25/100 条，按 episode 划分建库/dev/test。先离线筛掉明显无效档，每组合限制 8–12 个 dev 分支（各25起点）：原生、k/H阶梯及逐层消融；选定后冻结。正式用 50 个新起点配对比较原生、最佳纯 k/H、完整系统。预计连同建库 **1.5–2 万局**；只有通过 baseline 门槛的组合才进入这一阶段。
4. **记录可决策指标。** SR 和配对置信区间、每成功局时间、每局 CPU core-seconds/GPU service-seconds、p50/p95 延迟、视觉/头/检索分项、调用率、cache 覆盖距离、守卫误触发/漏救援、修正器收益。用相同种子配对；五层消融固定重规划时刻，H实验单列；不能用 test 结果再改逐任务档位。暂定扩展门槛：相对最佳纯 k/H，成功率下降不超过 3 pp 且区间足以支持该容限，同时端到端至少省 20%；证据不足则加样本，不把“不显著”当无损。

**首轮资源上限建议：180 GPU 小时、500 CPU core-hours，分阶段释放；预计训练 28–108、适配/评估 25–60 GPU 小时，仿真/数据重放 150–400 core-hours。** 64 workers 理想下 500 core-hours≈7.8 小时，但图像带宽、单 GPU 排队与 worker 利用率会拉长墙钟；不能据此承诺吞吐。这里只做了 CPU 探针，没有训练、GPU 推理或服务启动，也没有新增 benchmark 安装。若累计 24 GPU 小时仍修不回某个公共 baseline，停止该分支并回报，禁止用低成功率换漂亮加速比。

## 不选什么

- **所有 VLM/LLM 主干 VLA**：不进入新模型名单；π0.5 只保留已有对照。
- **RoboCasa/365、RoboTwin 2.0、BEHAVIOR、SimplerEnv、ManiSkill3**：遵照 owner 的场景变化、重量或渲染限制排除。
- **CALVIN**：π0.5 权重及 3.838 序列均长已核实；MoDE＋HULC＋MCIL 也能组成有权重组合。但后两者都属潜计划策略、缺已核 LIBERO 权重，MCIL 的长链基线很弱；五任务串联扩大同一指令的状态分布，TinyRenderer 又有纹理差异。本次优先 MetaWorld 的模型差异与单任务库覆盖，接受一次小型 DP 自训；不以“缺 π0.5”或“必然很重”排除 CALVIN。
- **VLABench**：已核实官方 π0.5 primitive-10task 权重及 ID 40.6%；但物体类别/语义/纹理变化是核心评测轴，收缩后又缺当前三模型的同协议权重，不作主基准。
- **robomimic、Adroit**：轻且可缓存，但较旧；本次未核实合格 π0.5 同协议权重＋成绩组合。**DexArt**另有 SAPIEN 渲染与物体实例泛化负担。
- **LIBERO-PRO/Plus**：更新，但核心是扰动/变化泛化，且不能充分提供独立于 LIBERO 的覆盖。可选附加仅用 LIBERO-Plus 的少量低强度图像扰动，库冻结；不把物体/布局变化加回主基准。
- **MolmoSpaces（2026）**：π0.5 支持真实存在，MuJoCo 也可 CPU 渲染；仍因大规模场景/物体变化不合本项目缓存目标。
- **Octo、VQ-BeT/QueST、MoDE**：有价值的下一批；当前没有比上述组合更省训练的 MetaWorld 可验收权重交集。**Canonical** 的公开 README 链接实际匿名访问 401；**CRT_MetaWorld** 是图像修复器，不是动作策略，均不充数。
