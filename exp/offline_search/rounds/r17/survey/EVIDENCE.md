# 重新选型：证据与可复现边界

核查日期：2026-10-08（America/Chicago；快照 UTC 日期为 10-09）。对应 [DECISION.md](DECISION.md)。本文件替代旧调查结论；旧决策仅保存在 [DECISION_v1_rejected.md](DECISION_v1_rejected.md)。

## 1. 方法及结论强度

先读 [INPUT.md](INPUT.md) 和内部原始记录，再按最新要求重新联合筛选模型、benchmark：①没有 VLM/LLM 策略主干；②有自有视觉特征；③场景固定、CPU 渲染可负担；④π0.5 有实际权重及成绩；⑤尽量减少两个 benchmark 上的自训。语言任务 ID 或预先算好的文本条件不等于以 VLM/LLM 为策略主干。

本次推荐 ACT＋DP3＋图像 ε-DP，MetaWorld 为新增基准。**没有找到同时在 LIBERO 和 MetaWorld 拥有三个不同家族、完整配置、已报告成绩且协议一致的公共权重组合。** ACT/DP3 有 MetaWorld 发布物，但有下述复现缺口；ε-DP 为最后手段自训。这个缺口没有被“可训练”“论文做过”替代。CALVIN 确有 MoDE/HULC/MCIL 的权重组合，是认真比较过的备选，见 §7。

证据分级：**文件存在 → 配置可解释 → 该发布物报告过成绩 → 本机复现成功**是四件事。本次做到前三项的部分交集，没有 GPU 复现；匿名 `gated=false` 不保证可直接加载、协议完整或底座许可无限制。所有“未找到”仅限本次核查，不是全网不存在的断言。

取证快照在 [sources_v2/](sources_v2/)，[manifest.json](sources_v2/manifest.json) 记录 URL、时间、字节数、内容 SHA256 与 HTTP 失败。只取元数据、配置、源码；单次文本下载上限 1.8 MB。对 HULC/MCIL 仅 HTTP HEAD 和 128 KiB ZIP 尾部目录，未下载 checkpoint。旧 `sources/`、旧厨房探针为历史材料，**不是本版推荐或新增测量**。

本次未安装任何软件/benchmark；复用现有 MetaWorld、LIBERO Python 环境与已有 OSMesa。没有 GPU 推理/训练、远程连接、服务器、公开端口或进程干预；没有触碰用户禁止目录。CPU 探针均限制 affinity 和空 CUDA_VISIBLE_DEVICES。

## 2. 公共权重核验表

以下 SHA 来自 HF API，不是按模型名猜测。相应 API 原文已存快照。文件名仅列关键文件；没有下载模型参数。

| 模型 / 范围 | 仓库、revision | 实际列出的文件 | 成绩与限制 / 许可 |
|---|---|---|---|
| ACT，MetaWorld 50 tasks | [RoboHP/metaworld_mt50_act](https://huggingface.co/RoboHP/metaworld_mt50_act/tree/main)；`680a3b396860b88adcf4036a735fe124b4d850da` | `pretrained_model/model.safetensors`、`config.json`、`train_config.json`，另有 optimizer/rng/training_step | 25 局/任务、逐任务表；MIT、免 gating。评测 fork 当前 404、相机和动作翻转约定不完整，**发布了成绩但尚不能称可复现基线** |
| DP3，MetaWorld 12 tasks | [chenhaojun/3D-Diffusion-Policy-ckpt](https://huggingface.co/chenhaojun/3D-Diffusion-Policy-ckpt/tree/main)；`c2dc19b2631488d1e34a615c552bf57c83e72fc1` | 每任务 `.hydra/config.yaml`、`checkpoints/latest.ckpt`；部分带分数命名 checkpoint、`eval_results.json` | assembly 1.00、button-wall .95；社区 release，有训练配置/日志/评估文件。免 gating；仓库无独立权重 license 声明，作者代码 MIT |
| π0.5，MetaWorld | [RLinf/RLinf-Pi05-MetaWorld-SFT](https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT/tree/main)；`60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e` | `model.safetensors`、`metadata.pt`、`lerobot/metaworld_mt50/norm_stats.json` | RLinf 文档 SFT 四组平均 .438；本地已有相应闭环历史。免 gating；仓无 README/独立 license 元数据，上游 π0.5/Gemma 条款仍需核对 |
| ACT，LIBERO，候选待验收 | [Deepkar/libero-test-act](https://huggingface.co/Deepkar/libero-test-act/tree/main)；`b6a5253edf0c9d9e458629fdeb489f514ff6300f` | `model.safetensors`、`config.json`、`train_config.json`、pre/postprocessor JSON 和 normalizer safetensors | Apache-2.0 卡片、免 gating。未核到完整 suite SR；OpenRAL 包装卡只有 Spatial task 2 单种子 91 步成功，且写明没有 benchmark 测量，不能据此称复现论文 |
| 图像 DP，LIBERO，另一配方候选 | [ttotmoon/diffusion-libero-v3](https://huggingface.co/ttotmoon/diffusion-libero-v3/tree/main)；`5825af28c585ade6827ea7e8f6234f3ab04e8ab1` | 模型、config、train_config、pre/postprocessor、normalizer 文件齐全 | Apache-2.0 卡片、免 gating；没有 suite SR。训练数据标 `HuggingFaceVLA/libero`，不是 LIBERO-90。DDPM100/C16/H8，不能当 QueST 的 DDIM10/C32 权重 |
| DP3，LIBERO | 未核到合格发布物 | HF 搜索命中以真机/其他任务为主 | 明确自训；优先六任务，公共 demos 重放生成深度，不能声称已复现不存在的 LIBERO 原论文表 |
| ε-DP，MetaWorld | 未核到“模型＋匹配配置＋可信成绩”交集 | 见 §4 的逐仓排除 | 按发表过的配方训练；先计时再锁预算 |

DP3 的 12 个 MetaWorld 文件夹为 assembly、basketball、bin-picking、box-close、button-press、button-press-topdown、button-press-topdown-wall、button-press-wall、coffee-button、coffee-pull、coffee-push、dial-turn；不是 MT50 全集。首轮六任务均在此列表。assembly 的 `epoch=0400-test_mean_score=1.000.ckpt`、basketball 的 `epoch=0200-test_mean_score=1.000.ckpt`、bin-picking 的 `epoch=0400-test_mean_score=0.250.ckpt`、box-close 的 `epoch=0200-test_mean_score=0.650.ckpt`、dial-turn 的 `epoch=0200-test_mean_score=0.800.ckpt` 确实在仓库列表。**文件名不是独立重评成绩**；coffee-push 的 `info.txt` 提到 epoch1200/1.000，但目录列的是 `latest.ckpt`，不凭日志虚构一个已存在的文件。

## 3. 入选模型：架构、取键和默认执行

### ACT

核查 [发布 config](https://huggingface.co/RoboHP/metaworld_mt50_act/blob/main/pretrained_model/config.json)、[train_config](https://huggingface.co/RoboHP/metaworld_mt50_act/blob/main/pretrained_model/train_config.json)、[模型卡](https://huggingface.co/RoboHP/metaworld_mt50_act/blob/main/README.md)、[数据集 info](https://huggingface.co/datasets/RoboHP/metaworld_mt50_v3_50d_512r/blob/main/meta/info.json)。

- 输入为两路 `base`/`wrist` RGB，均 3×512×512，state/action 各 4 维；ResNet18、ImageNet 初始化；Transformer dimension 512、8 heads、4 encoder layers、1 decoder layer，CVAE latent32。
- `chunk_size=100`、`n_action_steps=100`、`temporal_ensemble_coeff=null`。[LeRobot ACT实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py)在推理时取零潜变量，一次解码；无 NFE 减步。模型卡未给评测命令，不能证明报告 SR 当时没有覆盖这些默认值。
- 检索 hook 放在两路 ResNet 空间输出，固定空间池化→PCA64，并加 proprio；不能为了取键先走整个 CVAE/Transformer。PCA/距离度量只在建库/开发集拟合。
- 卡片训练为 200k updates、batch64；数据集实际 2,500 episodes、187,346 frames、50 tasks，两路视频，声明 80 fps。评测写 corner4、翻图与对应动作，但没有说明第二路视角和变换矩阵；链接的 [AoqunJin/Metaworld](https://github.com/AoqunJin/Metaworld) GitHub/API 均返回 404。这是实质性复现风险。
- 模型卡 headline .731 是四个难度组均值。重算 50 行 task SR 算术均值为 **.7752**；首六任务表值 .92/.36/.96/1.00/.96/.92，均值 .8533。不要拿 .731 和按任务平均的其他策略直接比。
- LIBERO Deepkar 配置仍 C100/H100、ResNet18/CVAE，两路 256²、state8/action7。仅验收它的已提供协议；如果找不到正确 preprocessing 或成功率明显不足，启用已列的自训额度，不用单种子证据包装成已发表基线。
- 缓存收益可能很低：原生已一 call 执行 100 步。五层都能实现；H10/25/50 是诊断，不能人为降低 baseline H 后宣称加速。延长超过 100 要用库轨迹，不能把同一网络凭空改成 C200。

### DP3

核查 [随 assembly 权重的 Hydra 配置](https://huggingface.co/chenhaojun/3D-Diffusion-Policy-ckpt/blob/main/metaworld_assembly-dp3-1218_seed0/.hydra/config.yaml)、[策略源码](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py)、[PointNet](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py)、[环境 wrapper 快照](sources_v2/dp3_wrapper.txt)、[runner 快照](sources_v2/dp3_runner.txt)。

- 点云 512×3，无 RGB 通道；9 维状态是末端位置和左右指尖位置。PointNet 输出64维。保存两帧视觉键/状态历史，度量保留时间顺序；这是视觉感知，不是 ground-truth 物体状态。
- 头为条件 1D U-Net，widths512/1024/2048、FiLM 条件；DDIM，训练100 noise levels、squared-cosine、`prediction_type=sample`（x₀）、**推理10步**。C4、obs2、H3，前一槽属于历史对齐，原生可执行未来动作只有3个。
- renderer 128²，camera corner2；作者 wrapper 的相机位置/深度范围、61.4°/−7°旋转、裁剪范围都要保存。源码先反投影再裁剪、FPS 512点，不能把另一相机或 world frame 的点直接输入。
- [FPS 实现](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/3D-Diffusion-Policy/diffusion_policy_3d/gym_util/mjpc_wrapper.py)硬写 `.cuda()`。本次只测独立 CPU NumPy FPS；工程还要用 fixture 核对点坐标、首点、tie-breaking、输出顺序。GPU 点云前处理不符合 worker 约束。
- 原 wrapper 每步额外取 RGB，虽 checkpoint 不用 RGB；应改成 depth-only、只取 query 所需 t−1/t 两帧。这个实现尚未完成，CPU 表的 depth-only 成本是移植预算，不能当原 wrapper 实测。
- wrapper 实际硬停200步；外层 runner 默认1000步不覆盖这个更短的内部上限。复现时首先保持200，不能只看到 runner 就按1000计分。
- 发布配置 `max_train_episodes=90`、batch128、3000epochs；不等于作者原论文“10 demonstrations”。[assembly eval](https://huggingface.co/chenhaojun/3D-Diffusion-Policy-ckpt/blob/main/metaworld_assembly-dp3-1218_seed0/eval_results.json) 为1.00，[button-wall eval](https://huggingface.co/chenhaojun/3D-Diffusion-Policy-ckpt/blob/main/metaworld_button-press-wall-dp3-1218_seed0/eval_results.json) 为.95。assembly `mean_durations=4.825s` 是未知硬件的 rollout 时长，不是 per-call latency。
- [官方 README](https://github.com/YanjieZe/3D-Diffusion-Policy) 给标准 DP3 约3小时/A40、10GB VRAM；Simple DP3 的1–2小时、25FPS不是此标准 DP3 的速度。LIBERO 每任务2–6 GPU小时是据此放宽的预算先验。

### 图像 ε-Diffusion Policy

选 [QueST 官方仓中的 baseline](https://github.com/pairlab/QueST)，不是选择带 VQ 的 QueST 主模型。核查 [DP yaml](https://github.com/pairlab/QueST/blob/main/config/algo/diffusion_policy.yaml)、[DP 实现](https://github.com/pairlab/QueST/blob/main/quest/algos/diffusion_policy.py)、[图像 encoder](https://github.com/pairlab/QueST/blob/main/config/algo/encoder/image/resnet.yaml)、[训练脚本](https://github.com/pairlab/QueST/blob/main/scripts/dp/main.sh)及 [NeurIPS 2024 论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/076c3e48fa502c660902105965fdd9f6-Paper-Conference.pdf)。

- encoder 是截断 ResNet18（`remove_layer_num=4`，非 ImageNet 初始化）＋空间池化/线性投影，image embedding256；策略 U-Net widths256/512/1024。MetaWorld 输入一张128² corner2图、8维 robot state、task onehot；无 LLM/VLM 主干。LIBERO 的离线文本条件也不进入大型视觉语言策略主干。
- 训练代码明确 noise target MSE，ε 参数化；DDIM100训练noise levels、10推理步；obs1。yaml C16、H2；发布的 **LIBERO-90** 训练脚本改为 C32、200epochs。本次诊断选 C32/H2，但不把该脚本说成已核实的 MetaWorld 论文专用命令。
- MetaWorld 配方是 ML45、100 demos/task、生成噪声.2、horizon500；官方支持两基准，但不能把 ML45 表改称 MT50-v3。
- 论文 Table11 的六任务 DP 数字：assembly .88、basketball .78、bin-picking .96、button-wall .98、coffee-push .86、dial-turn1.00。这些来自45任务联合训练；六任务 pilot 不可宣称已复现45任务训练结果。LIBERO-90 DP表值75.1%，也不是 Spatial 的目标分数。
- 先复现原配方可比的小范围目标与训练曲线，再决定是否值得完整 ML45/LIBERO90 三 seed 验收。没有与新设置严格匹配的论文 SR 时，清楚标成新实验；保留发布协议锚点，不捏造“距论文±5pp”。
- 动作 H2→4/8/16/32 有直接执行空间；与减少10→4/2/1步分别消融。旧 ε-DP 少步劣化是选它的研究动机，不是保证这个新 checkpoint 一步必然差。

## 4. 为什么没有用其他公开权重替代自训

| 候选 | 实际检查 | 决定 |
|---|---|---|
| [brandonyang/dp-metaworld-checkpoints](https://huggingface.co/brandonyang/dp-metaworld-checkpoints/tree/main) | 两个 step 子目录有 model.safetensors、metadata.pt、optimizer、norm_stats；无 README/完整架构 config/报告成绩 | 不能仅凭 `dp` 仓名确认 ε-U-Net、steps、C/H 或复现质量；补元数据有希望，但本轮不能算合格发布物 |
| [iFaz/diffusion-metaworld_medium-seed42-half](https://huggingface.co/iFaz/diffusion-metaworld_medium-seed42-half/tree/main) | API 只有 `.gitattributes` | 空权重仓，不计入 |
| abel2341/diffusion_metaworld_mt50 | 搜索索引出现卡片，但匿名 API401；索引卡也没有评测成绩 | 不承诺公开可下载 |
| [kunhsiang/act-metaworld-mt10-baseline](https://huggingface.co/kunhsiang/act-metaworld-mt10-baseline/tree/main) | 实际有文件，但卡片写 LIBERO 数据、5000steps、未 eval | 不是更可信的 MT10 替代品 |
| [MoDE LIBERO-10](https://huggingface.co/mbreuss/MoDE_LIBERO_10/tree/main) | revision `c3482882c9e8eec9c6af739e9bbe5b56d3eae2aa`，config.yaml/model_cleaned.safetensors；MIT、免 gating；官方还有 LIBERO90/CALVIN | 很好的非 VLM ResNet50＋MoE diffusion；但没核到 MetaWorld 权重或对应训练/评测配方，换它仍需新移植训练，且会失去明确的 ε-U-Net 对照 |
| [Cosmos MetaWorld](https://huggingface.co/mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B/tree/main) | consolidated.pt、统计量和T5缓存存在；模型卡明确的是 assembly-v3，25局 base52%、steering84–88%。所链 FlowDAgger 当前公开 backend 是 π0.5/GR00T，缺对应 Cosmos runner | 2B video diffusion、只有一个已报告任务且协议缺口大，首轮不选；不是“没有权重” |
| [Cosmos LIBERO](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B/tree/main) | policy.pt、config/统计量/文本缓存在，当前 gated=false | 纠正旧“必然 gated”；不解决上一行的 MetaWorld 交集问题 |
| [Canonical](https://github.com/agentic-learning-ai-lab/canonical) | README 宣称两基准 bundle；所链 HF 当前匿名 API/raw401 | 发布说明不足以证明可获取文件，暂缓 |
| [CRT_MetaWorld](https://huggingface.co/AIRLab-POLIMI/CRT_MetaWorld/tree/main) | 有 CRT_metaworld.pth；方法是 Corruption Restoration Transformer | 图像修复器不是动作策略，不能充数 |
| [DILLO](https://github.com/MaxPappa/DILLO) | README 链 GDrive；未取到可核完整权重列表；名为 act_policy 的实现是 ResNet18-FiLM＋Transformer/GMM | 不是 CVAE ACT 权重替代；不能只看文件名判断家族 |
| Octo、VQ-BeT、QueST 主模型 | 作者代码/预训练存在；未核到 MetaWorld＋LIBERO 同协议、报告成绩的可用发布物交集 | 本轮不能比小型 ε-DP 更省复现工作；以后补权重可再比较 |

## 5. Benchmark 与 π0.5 的协议核查

[MetaWorld 官方代码](https://github.com/Farama-Foundation/Metaworld)、[Meta-World+ 2025](https://arxiv.org/abs/2505.11289)：原任务集2019/2020年，2025是改进、标准化工作，**不是新发布的50个任务**。代码为[MIT许可](https://github.com/Farama-Foundation/Metaworld/blob/main/LICENSE)、环境无门禁。以其影响力、持续维护和固定任务世界满足本次实际需求；“近期新任务”这项是明确取舍，不隐瞒年龄。

同任务固定机器人、物体种类/形状和场景，无 RoboCasa 式 scene/object catalog 随机抽样；仍有目标/初始位姿随机性，且部分任务范围不小。源码例：button-wall 物体 x∈[−.05,.05]、y∈[.85,.9]；dial-turn x∈[−.1,.1]、y∈[.7,.8]；assembly 螺母初始固定、peg目标 x∈[−.1,.1]、y∈[.75,.85]。因此按任务建库25/100条并测覆盖距离，保留原分布；不把 test reset 固定成训练样本。

[bin-picking-v3 源码](https://github.com/Farama-Foundation/Metaworld/blob/main/metaworld/envs/sawyer_bin_picking_v3.py)已注明初始对象范围调整。DP3 源是 v2/mujoco-py，ACT 的 fork 不可获取，π0.5 当前本地为 metaworld3.0.0/mujoco3.14.0；三者不能只改 env 后缀就声称同协议。路线是**先原版本验收，再迁到固定的 Farama v3；未迁过关时只报告同一模型内的加速对照，不报统一排行榜。** 若要限定一个公共协议用于最终论文，v3过关是后续扩大实验的前置条件。

π0.5 的 [RLinf MetaWorld 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/metaworld.html)直接链接上述 HF 权重，并给 SFT easy/medium/hard/veryhard = .682/.373/.417/.280、四组平均 .438。本地 [MetaWorld adapter](../../../../metaworld/env.py) 与 [交接记录](../../../../../logs/session_handoff.md)已有50×20局 full=.588、k2=.604、k1=.537；分组平均与任务平均不能混算。现有 adapter：corner2位置(.75,.075,.7)、480²、旋转180°、state前4维、settle15步、max160、H5、fullK10；本次没有跑此 GPU 策略。

## 6. CPU 成本：本次实测及外推

**机器与环境。** weilandserver，非 timan107；CPU 型号为已有环境记录中的 Xeon E5-2696 v4。只使用核18–25、62–69。MetaWorld 环境 `/home/weiland/projects/openpi_ext/envs/metaworld_sim`（Python3.11、metaworld3.0.0、mujoco3.14.0、numpy1.26.4）；LIBERO 环境 `/home/weiland/projects/openpi_ext/envs/libero_sim`。OSMesa 库来自已有 `/home/weiland/projects/openpi_ext/lines/dp/env/lib`。**新安装：无。**

设置 `CUDA_VISIBLE_DEVICES=`、`MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa`、`LP_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。GL_RENDERER 实际为 `llvmpipe (LLVM 15.0.7, 256 bits)`，探针对非软件 renderer 会断言失败。用墙钟和 `process_time()` 同时记；RGB CPU时间通常比墙钟略大，预算使用 core-seconds。

测量文件：[MetaWorld 脚本](probes/mw_cpu_probe.py)/[结果](probes/mw_cpu_results.json)，[LIBERO 脚本](probes/libero_cpu_probe.py)/[结果](probes/libero_cpu_results.json)，[FPS 脚本](probes/fps_cpu_probe.py)/[结果](probes/fps_cpu_results.json)，[每局外推](probes/cpu_episode_estimates.json)。

MetaWorld 三任务 assembly/button-wall/box-close，MT1 seed7、固定一个 task 向量，三次reset；每阶段45次、前5次预热，40次计时。分别测物理、单RGB、双RGB、深度。相机沿用本地π0.5位置；双相机为 camera id2/0，**只是 ACT 双视角成本代理，不是已验收的 wrist 映射**。LIBERO Spatial task0/2，agentview＋eye_in_hand，24步去掉前4步；图像 observable 关闭后另测物理。所有动作是零动作/保持夹爪，无视频、模型或网络服务。

| 仿真配置 | 物理 ms/控制步 | 单 RGB ms | 双 RGB ms | 深度 ms | reset s |
|---|---:|---:|---:|---:|---:|
| MetaWorld 128²，三个任务范围 | .38–.68 | 166–178 | 373–404 | 15.2–18.6 | .012–.038 |
| MetaWorld 256² | .36–.58 | 177–203 | 389–446 | 23.7–29.5 | .022–.038 |
| MetaWorld 480² | .44–.86 | 259–275 | 561–588 | 42.5–43.2 | .018–.040 |
| MetaWorld 512² | .50–.59 | 250–277 | 591–620 | 37.2–43.5 | .017–.039 |
| LIBERO 128² | 30.3–32.5 | — | 358–368，扣除物理后 | 未测 | .399–.440 |
| LIBERO 256² | 30.2–31.1 | — | 421–433，扣除物理后 | 未测 | .506–.527 |

分辨率下降不保证成本按像素数缩小：保留默认场景/灯光/阴影，软件 renderer 有较大固定开销。MetaWorld首次构造MT1约.71–1.08s；重复reset很小。LIBERO初始construct另计，不每局重建。进程峰值RSS为串行构建多环境后的累计高水位，不能拿它精确推算64worker内存；正式先做1/8/32worker阶梯。

外推公式：`T = reset + L·t_physics + ceil(L/H)·min(H,n_obs)·t_visual`；LIBERO 的 `t_visual` 从含图像step减去physics。初始化历史帧可能少一帧，此处保守取整。H改变会同时改变观察成本，不能把它只当GPU加速。历史2帧策略必须采t−1/t，不能每H步取一帧后用旧图填充。

| 200步标准化/原生设置 | 外推墙钟 s/局 | 外推 CPU core-s/局 | 解释 |
|---|---:|---:|---|
| MW，128²一RGB，L200/H5/obs1 | 6.73–7.27 | 7.51–8.04 | 基准轻量筛选对照；不含策略 |
| LIBERO，256²两RGB，L200/H5/obs1 | 23.57–23.84 | 24.18–24.43 | 同机、相同L/H，不冒充所有LIBERO任务 |
| π0.5 MW相机设置，480²一RGB，L160/H5 | 8.41–8.96 | 9.09–9.65 | 仅仿真成本，不是π0.5完整rollout计时 |
| 图像DP，128²一RGB，L500/H2/obs1 | 41.72–44.80 | 46.58–49.63 | 原配方长时限；20,000局约259–276 core-hours |
| ACT双视角代理，512²两RGB，L200/H100 | 1.30–1.39 | 1.38–1.47 | 只需2次观察；若guard另加观察必须加计 |
| DP3，128² depth，L200/H3/obs2，CPU FPS | 约4–22 | 约4–22 | 134帧深度＋FPS的预算；未完成整套CPU点云移植 |

FPS 用纯 NumPy greedy、首点0、输出512点，synthetic输入1024/4096/16384点，预热后8次：均值12.9/41.9/139.8ms，p95 14.5/44.0/145.9ms；真实裁剪后点数未知。DP3 depth-only观察约15–19ms，再加FPS/反投影；即使GPU头变快，CPU前处理可能成为瓶颈。应记录裁剪点数及耗时，而不是宣称PointNet便宜就端到端便宜。

**有效范围：**这是短CPU探针外推，非完整成功轨迹、非64进程并发、非 timan107测量。render在静态状态计时；碰撞接触、不同task、内存带宽会变。没有验证软件渲染像素与训练renderer一致；这属于baseline验收。若直接使用原来“每步都渲染”的wrapper，H优化不会自动得到此表收益，工程必须按真实观测依赖稀疏渲染，并把该优化单独列账。

复跑命令（把文件名换成 `libero_cpu_probe.py` 并使用 libero_sim 的 Python 可复跑LIBERO）：

```bash
taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  LD_LIBRARY_PATH=/home/weiland/projects/openpi_ext/lines/dp/env/lib \
  LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python \
  exp/offline_search/rounds/r17/survey/probes/mw_cpu_probe.py
```

## 7. 其他 benchmark 的真实状态

| Benchmark | π0.5 与权重核验 | 缓存、CPU、选择理由 |
|---|---|---|
| **CALVIN** | [RLinf-Pi05-CALVIN-ABC-D-SFT](https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT/tree/main)，sha `ffe18938a37d3855c9ff50d6220d39bcd6434f15`；模型、metadata、norm_stats实存，Apache-2.0卡片、免gating。[RLinf结果](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/calvin.html)：SFT序列均长3.838，完成1–5个任务为.927/.843/.767/.688/.613 | 2021/2022，PyBullet CPU TinyRenderer可用，不能称GPU-only。测试D场景固定，但长链前序状态变化大，库按指令还须区分上下文；[官方FAQ](https://github.com/mees/calvin#faq)明确CPU/GPU纹理有差异。未本地安装或计时，不声称它一定更重 |
| **CALVIN的非VLA组合** | [MoDE_CALVIN_D](https://huggingface.co/mbreuss/MoDE_CALVIN_D/tree/main)有config+model；[HULC下载脚本](https://github.com/lukashermann/hulc/blob/main/checkpoints/download_model_weights.sh)、[MCIL官方链接](https://github.com/mees/calvin#pre-trained-model)的HTTP ZIP实际200。只读ZIP目录已见各`.hydra/config.yaml`和`.ckpt`：[目录取证](sources_v2/calvin_zip_inventory.json) | **可组成三个有权重模型，不能以“没权重”排除。** 但HULC/MCIL都属层级潜计划策略，未核到LIBERO权重，移植/复现工作更大；MCIL原始D→D静态RGB的长链首任务48.9%、完整五链.08%，弱基线不适合证明可靠加速。[原论文](https://arxiv.org/html/2112.03227v1)；因此优先保留MetaWorld的2D/3D/CVAE差异，承认不是单纯按新增基准自训数量最少排序 |
| **VLABench** | [官方 pi05-primitive-10task](https://huggingface.co/VLABench/pi05-primitive-10task/tree/main)有Orbax params、assets/statistics、训练元数据；[卡片](https://huggingface.co/VLABench/pi05-primitive-10task/blob/main/README.md)报ID40.6%、cross-category22.6%、commonsense18.0%、semantic16.1%、texture25.6%；30ksteps、8H100、5000episodes | MuJoCo/dm_control可CPU OSMesa，不因GPU-only排除；但类别、语义、纹理等泛化变化是核心，缩到固定primitive后仍缺当前三模型同协议权重 |
| **robomimic** | DP/ACT/VQ生态成熟；本次搜索到π0.5论文/方法线索，未核到可对应协议的公开π0.5 checkpoint＋报告成绩 | 2021、轻、固定任务，有潜力；强制π0.5证据链尚不够，不入选。不宣称全网不存在 |
| **Adroit / DexArt** | DP3有Adroit部分实际文件；未核实π0.5合格checkpoint＋成绩组合 | Adroit较旧；DexArt另有SAPIEN/物体实例泛化和渲染移植负担，不满足当前最小工作量 |
| **LIBERO-PRO / Plus** | 可复用LIBERO π0.5，社区评测支持不等于独立新任务checkpoint | [Plus官方](https://github.com/sylvestf/LIBERO-plus)重点是相机、机器人、语言、光照、背景、物体等扰动；主基准会回到owner不想要的变化轴，也不是独立于LIBERO的新覆盖。可选只取低强度图像扰动、冻结库，追加≤1000局/≤5GPU小时/≤20core-hours |
| **MolmoSpaces（2026）** | [官方代码](https://github.com/allenai/molmospaces)与[评测 harness](https://github.com/allenai/vla-evaluation-harness)有π0.5支持；MuJoCo也可OSMesa | 新且有影响力，但大量场景/物体组合恰好不合缓存覆盖要求；不能错误写成“不是MuJoCo”或“不支持π0.5” |
| **RoboCasa/365、RoboTwin2、BEHAVIOR、SimplerEnv、ManiSkill3** | 不再争取其权重交集 | 按owner已明确的变化、重量、GPU渲染限制排除 |

额外核查 MDT：公开README链接的是 MDT-V checkpoints（Voltron视觉语言预训练）；在本次严格无VLM主干筛选下不优先选择它来凑CALVIN第三个模型。HULC/MCIL和MoDE可以作为后续候选，但不能把不同split的权重、不同链评测和LIBERO成功率混用。

## 8. 内部经验如何影响五层与实验方法

| 内部原始证据 | 本次采用的结论 |
|---|---|
| [r16 REPORT](../../r16/REPORT.md)、[COORDINATOR_DECISIONS](../../r16/COORDINATOR_DECISIONS.md) | π0.5/GR00T少步在部分任务点估计接近，不推广到ε-DP；不同H的结果不能直接比较。既有终表看test后提档53任务，不当作新实验的无偏独立检验 |
| [r15 REPORT](../../r15/REPORT.md) | 轻量代理视觉键掉点、不同库容量改变最佳旋钮；每模型取自己的视觉特征，分别拟合阈值，建库/开发/测试按episode独立 |
| [DP减步原始日志](../../../../../logs/xwam_dp_nfe_plan.log.md)，39–73行 | DDIM10/2/1混合均值.88/.60/.03包含PushT coverage，不能全称SR；DDPM10失败包含旧scheduler previous_t实现错误，不能拿错误实现证明所有DDPM减步必失败；新实验固定scheduler版本与timestep grid |
| 同一DP日志，空卡计时 | 既有4090模型DDPM100约1.305–1.363s、DDIM10 .139–.155s、k4 .060–.075s、k1 .019–.034s；只提供量级，不是本轮ACT/DP3/QueST checkpoint的计时 |
| [Cosmos日志](../../../../../logs/cosmos_nfe_plan.log.md) | k步另有sample_clean，真实NFE=k+1；不把“k1”自动称一次网络前向；其他家族也记录真实NFE |
| [交接附录](../../../../../logs/session_handoff.md)、[cache transfer历史](../../../../../logs/cache_transfer/summary.md) | MetaWorld已有π0.5少步损失，适合保留对照；推理视觉地板、CPU渲染及worker内存都必须计入 |

五层的收益假设和“不适用”项详见决策表：ACT没有减步；DP3和ε-DP减步都需实测，x₀也不保证无损；ACT/DP3的网络C限制不能靠改H越过，库长轨迹则可尝试。视觉命中仍付encoder，guard与corrector的计算和增量观察都要计费。无视觉头只能做状态检索变体，不能冒称完成本次视觉五层迁移。

## 9. 资源估算、验收和停止条件

单call预算 ACT10–50ms、DP3 60–200ms、图像DP10步50–160ms（LIBERO备选DDPM100另按0.5–1.6s估算），**均是计划假设**；没有对应GPU实测数字。先对目标权重做warmup后分段profiling：视觉、头、检索/修正、CPU点云，batch1及计划并发batch，记录实际precision/NFE/p50/p95。已有SimpleDP3 FPS、未知硬件rollout秒数不能替代。

六任务×两个benchmark的训练预算：图像DP 6–18 GPU小时/benchmark，共12–36；ACT LIBERO仅社区权重失败时4–12；DP3 LIBERO2–6/任务，共12–36。因此需fallback时合计28–84，DP3-v3如再补训12–24则40–108 GPU小时；这些是单seed估算。训练前用1000updates测实际耗时，以`总updates×测得秒/update`重算；完整ML45/MT50/LIBERO90的20–80 GPU小时/模型/seed单独列账，不能把小模型等同于廉价全协议复现。

首轮180 GPU小时/500 CPU core-hours为建议额度，分阶段用：先权重/协议修复4–8GPU小时、1–2工程日；验收后小训练；最后约1.5–2万局、25–60GPU小时的建库/消融/冻结测试。CPU重放/仿真150–400core-hours是覆盖不同H/观察数的预算；测得全500步DP约50core-s/局提供上界量级。数据下载大小、GPU排队、worker内存带宽不在CPUcore-hour墙钟换算里。

验收顺序：

1. 原版本、原数据量、原观察/动作协议复现权重。对应发布结果差≤5pp；20–25局发布物补至≥100局观察区间。新v3和新六任务训练单列，不宣称重现原论文总分。ACT视角/翻转、DP3 CPU几何/采样是先决条件；连续24GPU小时仍修不回baseline则停止该分支。
2. 每任务25/100条建库；每模型/benchmark/task最多8–12个dev分支，各25起点，用于原生、k/H阶梯及逐层消融；不做全笛卡尔积。guard要评估误触发/漏救援，corrector检查动作坐标和夹爪边界；防止视觉近但任务阶段不同的假命中。
3. 冻结阈值/逐任务旋钮后，用50个新起点配对跑原生、最佳纯k/H、完整系统。baseline优化必须和完整系统享有同样的CPU稀疏渲染，不能重复记功。报告成功率、配对区间、每成功局成本、CPU/GPU service时间及端到端吞吐。
4. 扩展条件是相对最佳纯k/H下降不超过3pp且区间支持该容限、端到端至少省20%。50局/任务一般不足以证实3pp非劣性；它是diagnostic，先按模型汇总配对区间、必要时加样本，再决定扩大。ACT收益为零或DP3受CPU地板限制也是有效结果，不靠削弱基线制造加速。

本次额外用GitHub commits API核到的当前源码身份：DP3 `47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`、QueST `cc25a835a00cf2d91df69704bdd757904d13c061`、RLinf `0067f7d5ce525c18bda439ed8dbb638abafd6a06`、MoDE `72cf8a25eafa54316f0c6c961b620e4b6c9c3e6b`、MetaWorld当前main `59fc34d7768af9785e4688c3e1db671424f4a6c3`。本机探针以已安装3.0.0为准，不称运行了这个较新的main。这不是声称checkpoint训练时用了这些新commit；复现先按发布配置恢复版本，源码快照另有内容hash。

当前产物是可审批的研究/移植决策及CPU证据，**不是已跑通三模型的工程验收报告**。
