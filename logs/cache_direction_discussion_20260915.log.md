# cache 方向讨论纪要与后续方向（2026-09-15 → 09-16）

> 记录 owner 与执行 agent 在减步 teacher 基线收工之后的整段讨论：结果解读、novelty 危机、跨领域迁移调研、世界模型 / Cosmos Policy 方向、模型组合、投稿计划。数据出处：`logs/nfe_baseline_plan.log.md`、`logs/nfe_baseline_rc365_plan.log.md`、`logs/cache_transfer/`（7 路调研 + `summary.md` + `plan_A_world_model_planning.md`）。

## 1. 减步基线告诉我们什么

- 纯推理、不带 cache、只把动作头去噪步数从高斯噪声起砍到 k（全 n=500）：π0.5 spatial k=1,2 = 0.988/0.982（锚 0.99）；π0.5 l10 k=1…7 = 0.824/0.828/0.836/0.850/0.842/0.838/0.840（锚 0.92）；GR00T spatial k=1…3 = 0.932/0.918/0.946（锚 0.946）；GR00T l10 k=1…7 = 0.856/0.872/0.862/0.886/0.858/0.842/0.858（锚 0.868）。owner 裁定 π0.5 l10 图中各 +0.04，GR00T 不加；四组点已插进 RIT 四张图的 json。RoboCasa 减步阶梯停在半途（GR00T k=1 main 387/400、π0.5 k=9 main 36/400，journal 保留）。
- 机理：一步 Euler 从噪声出发 ≈ 输出分布的条件均值。单峰（LIBERO）时无损；多峰（ActionCache 的 VLABench 38.8→6.8）时崩。具身路的调研补充：**参数化也算**——DDPM/DDIM 训练的头一步崩，flow-matching 头一步无损。
- 对我们 cache 的含义（"电梯 vs 跳楼"）：减步只动 s3，动不了 s1+s2；地板 = 60.6%（π0.5）/ 40.7%（GR00T）。IR ≥ 60 区间减步与 cache 打平或更好，IR < 60 / < 41 只有 cache 能去。GR00T LIBERO 两张图上 cache 全面落后；π0.5 l10 原始值下 IR 80–95 段 cache 赢、60–70 平。
- novelty 危机（owner）：warm 档在 LIBERO 上没有 counterfactual，系统退化成 hit-or-k=1 二元；我们还独有的只剩"跨 episode 库 + 多档 + 一个 δ 从 shadow 偏差切所有 cut + 按预算反解 + 闭环 (IR,SR) 前沿定义"。

## 2. 跨领域迁移调研（7 路 fable agent，`logs/cache_transfer/summary.md`）

| 方向 | 结论 | 一票否决 |
|---|---|---|
| 驾驶扩散规划 | NO | anchor 词表吞掉多峰，1–2 步 = 全步；前缀占 66% |
| 流式媒体 / 交互世界模型 | MAYBE → DIAMOND/GameNGen 型 GO | 重训 1 步学生追平；帧级命中率空白 |
| LLM/VLM agent | MAYBE → 仅 diffusion-LM agent（LLaDA-UI） | AR 无中间态；vCache 占 hit-or-miss 标定 |
| 语音 | NO | 前缀占 64–70%；蒸馏 1–3 NFE；无闭环 |
| 科学求解（AF3 系） | NO | 不重训 200→2 步无损；无闭环 |
| 动画 / 感知 / 非 VLA 具身 | NO− / NO / MAYBE | 具身不算换领域，对手 RTI-DP 自身 warm start |
| 决策回路世界模型 | MAYBE 偏 GO（TD-MPC2） | AdaReP（2026-06）已做缓存 rollout + 偏差触发重规划 |

跨路结论：① 从头算的廉价近似（减步、蒸馏 1 步学生、自身上一输出 warm start）是每个领域的第一对手，warm 档只在"多峰 ∧ 未蒸馏 ∧ 减步真掉点"里有 counterfactual；② hit 档价值 = c_pre/c_0 地板，VLA 的 15% 反而算好的；③ 闭环漂移叙事只在控制 / 世界模型 / agent 成立；④ 剩余 novelty 见 §1。

排序（P × 收益）：DIAMOND 型世界模型 ≈ Cosmos Policy planning > TD-MPC2 > dLLM agent > 非 VLA 具身 > 动画 > 其余。owner 指出"这几个不是同一领域"后归成两个可成套方向：**A 决策回路里的世界模型**（TD-MPC2 / DINO-WM 受控台 + Cosmos Policy 主秀，复用我们 harness）与 **B 交互式世界模型 / 神经游戏引擎**（DIAMOND / open-oasis / Solaris，真换领域但基础设施从零）。推荐 A，B 只做 DIAMOND 三天探针；A 的展开见 `logs/cache_transfer/plan_A_world_model_planning.md`。

## 3. Cosmos Policy 核实结果（arXiv 2601.16163，`nvlabs/cosmos-policy`）

- 底座 Cosmos-Predict2-2B 视频扩散（Wan2.1 VAE + DiT，EDM）；"一切皆 latent 帧"：11 帧序列 = 占位 / 本体 / 3 图 / 动作 chunk / 未来本体 / 3 未来图 / 未来价值，非图像量归一化后复制填满整帧；策略 / 世界模型 / 价值 = 同一序列上的三种条件掩码（batch 50/25/25）；价值 = 蒙特卡洛回报。推理：整个 DiT 循环 5 步（ALOHA 10），1/5/10 步 = 0.16/0.61/0.95 s（H100）；chunk LIBERO 16 / RoboCasa 32 执行 16 / ALOHA 50。
- 检查点：LIBERO（98.5%，四 suite）、RoboCasa **2024 版 24 任务**（67.1%，非 RoboCasa365，用 `moojink/robocasa-cosmos-policy` fork）、ALOHA 策略 + ALOHA planning model；NSCLv1 非商用。
- 规划模式 best-of-N（每提案 1 + 3 + 15 = 19 次去噪，≈5 s/chunk，8×H100）**只在 ALOHA 真机验证**（+12.5 pp），用 648 条 rollout 再微调的 planning model；仿真上没报数。但 `run_libero_eval.py` 已有 `--num_queries_best_of_n`、`--ar_future_prediction/--ar_value_prediction`、`--planning_model_ckpt_path`（缺省用策略检查点自己当世界模型/价值），**planning 增益一条命令可测**；减步 = `--num_denoising_steps_action k`。
- 硬件：推理 6.8 GB（LIBERO）/ 8.9 GB（RoboCasa）/ 10 GB（规划），**weilandserver 4090 48 GB 可跑**（torch 2.7 cu128、flash-attn 2.7.3、TE 2.2.0，Ada 支持）；训练官方最小 8×80 GB，4090 跑不了配方，h100 单卡靠梯度累积只能做小 refinement；README 警告训练与测试要同款 GPU。4090 上估 1–1.3 s/chunk，LIBERO 500 集一点 ≈ 5–6 h。
- SO-101：没有人公开用 Cosmos Policy 跑过；仓库无 LeRobot 加载器与自定义机器人指南。最近先例 DreamZero-SO101（Wan2.1-I2V-14B LoRA，715 条社区数据，2×H100 127 h）。接 SO-101 = 自己写数据适配器 + 加 LoRA + h100 训。
- 我们的 cache 怎么接：tap point 在 VAE 编码之后（key = 3 帧 latent 池化 + 本体 + T5 指令）；库存动作 chunk、第 j 趟的 6 个目标帧中间状态与 σ_j、预测的未来 latent、价值；策略模式三档（hit 地板估 5–10%；warm = 存的目标帧状态 + 当前条件帧、只跑剩余趟，可整段或只 warm 动作帧；miss）；规划模式按"复用 V / 复用 s′+V / 复用整个提案"分档，D_a 用价值差；两个新信号：hit 后用库条目预测的 s′ 对比真实下一观测做零成本漂移检测（世界模型自验 cache），价值头做 RIT 风险代理。工程 = 照 `serve_pi05_ksweep` 的 monkeypatch 思路挂 key 钩子 / 去噪循环加初始状态入参 / 循环前 dispatcher / shadow 模式。
- 动画：`https://claude.ai/artifact/WGXygng6ys1zTUf6yRqw3a`（模块级数据流，策略 / 规划 / π0.5 对照，英文）。

## 4. 模型组合（限机器人 policy、cache 多档能用的）

| 原理族 | 模型 | 循环体占比 | 减迭代敏感度 | hit 地板 | 角色 |
|---|---|---|---|---|---|
| 流匹配小头 VLA | π0.5、GR00T N1.5 | 44–56% | 低 | 15% | 如实写"减步已到地板"，hit 档仍独占 IR<60%，与 k=1 叠加 |
| DDPM 扩散策略 | Diffusion Policy（robomimic / PushT） | ≈100% | 高（1 步 0%） | ≈0 | warm 档最亮 |
| 视频扩散共去噪 VLA | Cosmos Policy（2B） | ≈90% | 估高，待测 | 5–10% | 主秀 |
| 潜空间 MPC | TD-MPC2（Meta-World / DMControl） | ≈100% | 减样本/迭代会掉；policy prior 是免费地板 | ≈0 | 非扩散分档：warm = 从检索 plan 初始化少跑几轮 |
| 想象式 best-of-N | Cosmos Policy planning | ≈100% | 待测 | ≈0 | 收益量级最大 |

每个入选模型先过 3 天"杀手 ablation 门"：减迭代 ladder 在到目标 IR 前掉 ≥5 pp、hit 地板 ≤ 20%、自身上一输出 warm start 到不了目标。论文主图 = regime map（横轴循环体占比、纵轴减步敏感度），把 π0.5 的失败画成一个角而不是藏起来。

## 5. 投稿

- ICRA 2027（9/15）与 ICLR 2027 已过。可投：AISTATS（10/8）、MLSys（~10/30）、CVPR 2027（11/13）、ICML 2027（~1 月底）、RSS 2027（~2 月初）、IROS（~3 月）、NeurIPS / CoRL 2027（~5 月）、RA-L / TMLR 滚动。
- 建议两条线：现稿 → **RSS 2027**（机器人框架：RoboCasa 减步判决、减步 + cache 联合前沿、真机段、regime map）；下一篇 → **ICML / NeurIPS 2027**（cache for imagination-in-the-loop，Cosmos Policy 主秀）。
- MLSys 需要整套 wall-clock 口径（每决策 p50/p99 含检索、每卡带多少机器人的吞吐曲线、与批处理/减步/量化叠加、内存与库规模扩展、尾延迟、IR-vs-实测校验）；按现结构单机延迟上限 1.3–1.6×，不建议现在投；世界模型线有量级收益后再做 fleet serving 投 MLSys 2028。

## 6. 下一步（顺序）

1. Cosmos Policy 第 0 步：4090 装环境、复现 LIBERO 98.5%、`--num_denoising_steps_action 1/2/3/5` ladder；h100 跑 best-of-N（N=4，先关 3×5 集成）测 planning 增益；RoboCasa-2024 fork 单独装。kill：LIBERO 与 RoboCasa 规划增益都 ≈0 → 主秀换床或放弃 A。
2. RoboCasa365 减步判决（GR00T k=1 续跑 + π0.5 k=1..3）：判"flow-matching × 多峰会不会崩"，决定 tiers 在 VLA 里是否有家。
3. Diffusion Policy 接 robomimic（三天，最便宜的 warm 档证据）；TD-MPC2 接 Meta-World（wall-clock 地板 + AdaReP 式自身复用对照）。
4. π0.5 / GR00T 补 "cache ∘ k=1 teacher" 联合前沿（现有管线一天）。
5. DIAMOND 三天探针作 B 方向备选；SO-101 × Cosmos 等 1–2 出结果后再动。

## 7. 待探索：卡尔曼式「预测–校正」思想进四层判决（2026-09-20 提出，未立项）

> 背景：owner 2026-09-19 裁定不投 ICLR、转入探索，系统改四层 full hit / 单纯减步 / warm start / miss，着墨"何时减步、何时 warm start"（执行线 `step_vs_warmstart_diagnostics_plan.log.md`）。本节记录一个尚未验证的设计思路，只做备忘，不改变 §6 顺序。

卡尔曼滤波可借用的不是"滤波"本身，是三件事：预测再校正、按两侧不确定性比值分配增益、用创新量（预测与观测之差）做验证门。对照现有系统：warm start 的 `start_t` 已经是离散化的增益（缓存 x_t 为先验，剩余去噪步为当前观测的校正）；RIT 的 q̂_a(s) 已经是缓存侧的噪声模型；缺的是策略侧噪声项和跨决策的时间递推。

| # | 思路 | 对应现有部件 | 依赖 / 判据 |
|---|---|---|---|
| 1 | **档位选择改成不确定性比值**：增益 ≈ σ²_policy / (σ²_policy + σ²_cache(s))。σ²_cache(s) = RIT 风险曲线；σ²_policy = step_diag 的 `disp_K`（全步条件离散度）与 `d_k`（减步偏离）。规则：策略确定且减步偏离小 → 单纯减步；策略不确定但 s 高 → warm / full hit；两边都差 → miss。标定从一条曲线变为 (s, disp) 二维表，shadow 表已记录这两个量 | `dispatch_surface` judge、`exp/step_diag` shadow 表 | 等 Q-A / Q-B 出数：`disp_K`/`d_1` 与阶梯缺口的关联须过预注册门（ρ≥0.6），否则策略侧噪声项无依据 |
| 2 | **创新量验证门替代手工计数 hysteresis**：用库 prev/next 链预测下一条应为上次 winner 的后继，与实际 CP1 key 比较得创新量；小则继续盲回放不搜，大则搜索或 miss。key 在 FULL_HIT 下也要建，创新量近乎免费；连续阈值可消 `score_hysteresis` L=6 带来的 IR≈39 硬地板 | `follow_winner` / `score_hysteresis` gate | 不依赖在跑数据，成本最低，建议先写一页 plan。与 `history_verdict.md` 结论一致：历史进打分净负，进门才是正确岗位（gate 线 AUC 0.97+） |
| 3 | **在线 RIT 曲线更新换成递推分位估计**：增益按过程/观测噪声比自适应，附带后验方差可直接喂 `supported / gap_interpolated / unavailable` 分级 | `OnlineRiskCurves`（128 样本滑窗 + PAV） | 优先级最低：在线 RIT 负结果的瓶颈是信号（libero_10 AUROC 0.55），不是估计器 |

**要避开的用法**：在动作空间对缓存 chunk 与策略输出做线性加权。流匹配头输出是多峰样本，两峰平均落在无效动作上；x₀ 线证据（确定性回归头离散度≈0、锚点低 4–8 pp）已说明这一点。融合只能发生在 x_t 噪声空间（即现有 warm start 形式）或只用于决定档位，不碰动作本身。

**2026-09-21 更新（step_diag 出数）**：Q-A 两 policy ρ(d_1,g) = 0.085 / 0.377，均 `no_conclusion`，d_1 同 policy 内跨任务近常量 → 思路 1 的"策略侧噪声项"目前无依据；Q-B 显示无门 top-1 warm start 在 flat 任务上大幅有害且相似度分箱不区分好坏命中 → 支持思路 2（创新量/验证门）优先级前移。见 `exp/step_diag/analysis/step_vs_warmstart.md`。

**边界**：三条都是 Judge / Gate 槽的新实现，interceptor 模式可容纳，不动推理内部；属 L2，立项须走 Plan → G1。高斯假设只作粗近似，实际按分位数版本写。
