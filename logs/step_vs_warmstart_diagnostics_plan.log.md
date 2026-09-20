# 减步 vs warm start：闭环 shadow 诊断与等 NFE 对照（π0.5 / GR00T × LIBERO / RoboCasa365）— Plan v3.1（G1 / G2 APPROVED；Verify 待执行）

> 状态：`In Progress`（2026-09-20，L2；G2 Round 1 APPROVED / code approved，Verify 待执行；本轮修复与偏差裁定见 §9，审查证据见末尾 Review Log）。G1：R1 NEEDS REVISION（8 阻塞 + 2 非阻塞，执行者全部 Accepted → v2）；R2 由 Codex（Review Authority）按 owner override 直接修订至 v3 并 APPROVED；执行者逐条复核 v3 后附一处修订（flat 任务 plain/warm 两臂 100 集，非劣门在 n=50 不可达）→ v3.1。G1 Review Log 已按 execution_authority §3.1 删除（副本在执行者 job tmp）。原始执行者：Claude（Execution Authority）；G2：Codex（Review Authority，按本会话 owner override 直接修复后复测，未另称第二名独立审查者）。
> 上游裁定：owner 2026-09-19「不投 ICLR，转入探索；系统改四层 full hit / 单纯减步 / warm start / miss，着墨何时减步、何时 warm start；Q1（减步是否要按相似度门控）由 calibration 自动回答，不单独验证」（记忆 `project_cache_four_tier_design`）。
> 依据：减步基线线 `nfe_baseline_rc365_plan.log.md`（阶梯数据 `exp/nfe_baseline/data/`）；x₀ 实验线 `x0_multimodal_plan.log.md` + `exp/dp_nfe/analysis/x0_multimodal.md`（一步归零 = ε 参数化；x₀ 头一步无损但为确定性回归器；条件分布诊断方法 §4.3 及其解释边界）。
> 合作者来信（2026-09-19）三问：动作分块、1 步与 10 步轨迹对齐、分套件/分任务失败模式——纳入本线 §2 Q-C。

## 0. 一句话

用一次**教师驱动的 shadow 通道**（执行的动作永远是策略自身的全步输出，额外的减步 / warm start 动作只算不执行）在闭环状态分布上直接量出"减步改变了什么、warm start 补回了什么"，再用**forced warm start 闭环臂**与**同拓扑重跑的减步对照臂**在等 NFE 下逐任务配对，回答：(A) 在这批任务上，逐决策的减步动作偏离与阶梯缺口是否相关（探索性关联，不做新任务筛选结论）；(B) warm start 能否补回同预算减步尚存的 full-vs-reduced 缺口；(C) 合作者的三个事实问题。

## 1. 背景与已知事实（不再重测）

| 事实 | 来源 |
|---|---|
| π0.5 / GR00T（flow 头）LIBERO spatial / libero_10 上 k=1…7 阶梯平坦（π0.5 spatial 0.988/0.982；libero_10 0.824–0.850；GR00T spatial 0.93–0.95、l10 0.84–0.89），逐任务不随 k 变化 | `exp/nfe_baseline/data/runs/{pi05,groot}_libero_*/*_k*_s*.json` |
| RoboCasa365 逐任务阶梯（13 任务 × 50 集，seed 1,000,000+idx，`replan_steps=5`，main lane 8 任务无 pin、pnp lane 5 任务 `config/pnp_pinned_objects.json`）：π0.5 k=1…9 macro 0.391→0.595，10 步参考（v2）0.557；k≤5 在 weilandserver(4090)+timan107、k≥6 在 h100+timan108（`nfe_baseline_rc365_plan.log.md` §6）。逐任务 k=1/2/3 vs v2 参考：CloseFridge 0.00/0.12/0.18 vs 0.62、OpenCabinet 0.28/0.52/0.54 vs 0.70、PickPlaceToasterToCounter 0.00/0.34/0.52 vs 0.50、PickPlaceDrawerToCounter 0.10/0.12/0.26 vs 0.30、PickPlaceSinkToCounter 0.56/1.00/1.00 vs 1.00、OpenDrawer 0.78/0.78/0.72 vs 0.68、PickPlaceCounterToStove 1.00/0.98/0.90 vs 0.84。GR00T（锚 k=4）k=1/2/3 vs 参考：PickPlaceDrawerToCounter 0.38/0.56/0.62 vs 0.68、SlideDishwasherRack 0.24/0.48/0.48 vs 0.48、TurnOnSinkFaucet 0.06/0.24/0.24 vs 0.28、OpenCabinet 0.80/0.90/0.90 vs 0.98、PickPlaceCounterToStove 0.90/0.94/0.94 vs 0.94 | `exp/nfe_baseline/data/agg_{pi05,groot}_rc.json`（逐集 journal/run_plan/summary 在 `exp/nfe_baseline/data/runs/rc/<teacher>/{main,pnp}/k<k>/`）；参考 **`~/tmp_rit/teacher_ref_pi05_v2_50ep.json`（macro 0.5569）**、`~/tmp_rit/teacher_ref_groot_50ep.json`（0.680）。⚠ `teacher_ref_pi05_50ep.json`（0.5954，CloseFridge 0.46）是作废版，不得使用 |
| RoboCasa RIT（带门）K=2/K=3 warm 臂在 IR 65–69 段 macro 0.52–0.55，与 k=3 教师 0.525 打平；teacher-shadow 分布与部署分布不一致已有实证 | `logs/rc365_rit_run_progress.md` §3、§5b |
| LIBERO 上 forced warm start 的 SR ~ start_t 曲线已测（`AlwaysWarmStartJudge`）；RoboCasa 上只有带门 RIT 臂，无 forced warm 臂 | `logs/archive/warm_start_sweep_plan.log.md`；`exp/robocasa365/emit_ws_warmstart_yamls.py`（`build_warm_cell` 逐 teacher 可用；`emit_warm_arms` 对两 teacher 共用一个 `timesteps` 列表，本线不用它） |
| 分块与执行：π0.5 LIBERO `pi05_libero` action_horizon **10**（`training/config.py` `pi05_libero`），π0.5 RoboCasa 50；GR00T 16；客户端执行 5 步（`replan_steps=5`）再推理，1 步与全步相同；每集推理次数 spatial ≈21、libero_10 ≈60（GR00T ≈23 / ≈59）。执行维：LIBERO 7 维（后 25 维 padding，GR00T padding 非常数，按方差会误选 32 维，见 `docs/iclr/modality_weight_selection.md` §1）；RoboCasa 两策略前 12 维，GR00T 字段顺序另作启动断言（§3.0） | 阶梯 episode JSON `client_timing`；`docs/iclr/modality_weight_selection.md` |
| 全步 K 与 schedule：π0.5 两 benchmark 均 K=10（`pi05_v1`，t = 1 − i/10，库快照 `noise_action_1..9`）；GR00T RoboCasa K=4（`groot_n15_k4_v1`，t = i/4）、GR00T LIBERO K=8（`groot_n15_k8_v1`，`serve_groot_libero.py` CLI 覆盖） | `openpi.cache.types.DenoiseSchedule`；`logs/nfe_baseline_plan.log.md` |
| DP 上条件分布诊断（条件离散度 / ΔBIC / 越界 / 最近 demo）与闭环一致，但 ΔBIC 是拟合诊断不是峰数 | `exp/dp_nfe/analysis/dispersion_index.py` docstring；`x0_multimodal_plan.log.md` §4.3 |

未知（本线要测）：LIBERO-Object / LIBERO-Goal 阶梯；同 harness 的 k=10 锚点（libero_10 阶梯 k=7 0.84 与 RIT 线 10 步参考 0.92 不在同一 harness）；逐决策减步动作偏离及其与闭环结果的关系；RoboCasa cliff 任务上 forced warm start 在等 NFE、同拓扑下是否补回减步缺口。

## 2. 问题与预注册判据

### 2.0 距离与执行维（所有量共用）

- 核：`openpi.cache.components.surface_judge.weighted_chunk_deviation(a, b, w, active_mask, h_exec=5)`，即前 5 步、执行维加权 L2 的逐时刻均值。所有 D 都使用此核；不是整个窗口展平后的 L2。
- 度量空间为 staged stage-3 输出、与库 action chunk 一致的模型归一化空间；统一去掉 batch 维为 [H,D]，不把 adapter 反归一化后的环境动作混入。模型/库的空间、维度顺序、horizon 必须在启动时断言。这里度量的是执行相关维上的模型动作偏离，不直接声称物理末端距离。
- `active_mask` 由执行合同冻结，不由方差选维：LIBERO 两策略前 7 维；RoboCasa 两策略前 12 维。GR00T RC 的前 12 维须按 `groot_keys.ACTION_KEYS` 的顺序拼接，宽度 3/3/1/4/1（位置/旋转/夹爪/底盘/模式）；其余 padding 排除。不得按 checkpoint metadata 的字母序推定拼接顺序。
- 尺度来自该环境冻结的同一 RIT 库（RC W13 full；LIBERO 逐 suite 的既有库，见 §3.0）：将库 [N,H,D] 展平 N×H，`σ_d = std(unbiased=False)`；执行维 `w_d=1/σ_d`（σ>1e−6），σ≤1e−6 时 `w_d=1` 并记录退化维，非执行维 w=0。直接按此式计算，不能取 `compute_library_action_weights` 返回的 w 后只换 mask：该 helper 已将低方差维权重置零。冻结 mask、σ、w 与来源 sha。
- `disp_K = mean_{n<n'} D(a_K[n],a_K[n'])` 为全步独立噪声样本的条件离散度；它不是同噪声配对误差的必然下界。

### Q-A：逐决策减步偏离与阶梯缺口的关联（探索性）

采样合同（每个教师决策，独立 generator；不改变生产 RNG）：

1. 主分析固定 N=4 份全步 `a_K[n]`，每个 k 的 4 份 `a_k[n]` 使用与全步相同的初始噪声 z[n]。π0.5 K_set={1,2,3,5}；GR00T RC={1,2,3}；GR00T LIBERO={1,2,4,6}。样本间噪声独立；同一个 n 的所有 k 共享 z。
2. warm 从只读检索 top-1 的缓存 x_t 续跑，t 见 §3.0；无候选记 null 与原因。生产路径全步输出 `a_exec` 使用原生产噪声，仅执行和记录，不充当上述 4 个样本。
3. 主量 `d_k = mean_n D(a_k[n],a_K[n])`，非负，k=K 时为 0；`r_k=d_k/disp_K` 仅在 disp_K>1e−8 时定义，否则 null 并计数，原始 d_k 仍有效。warm 偏离 `d_w(t)=mean_n D(a_w(t),a_K[n])`；warm 没有相同初始噪声，和 d_k 不是同一种配对误差，只并列描述。若列 `d_k−disp_K`，字段必须叫 `excess_vs_spread`，不称“去噪偏离”、不进入主关联/接纳判据。此项纠正 R1/R1 回应沿用的扣 floor 解释。
4. Dense 在线选择：UTF-8 串 `20260919|env_id|task_name|init_idx|decision_idx|dense` 的 SHA256 大端整数 mod 16=0；不含 arm/attempt/launch，因此重试选择一致。命中时另采 28 份**仅全步**样本，合计 32 份用于 PCA-2D/GMM ΔBIC(2 vs 1)，主统计始终只用首 4 份。不承诺每任务恰好 20 个，也不事后按结果补样；报告实际数。拟合前仅保留前 5 步及执行维、按同一 w 缩放，固定 seed 20260919；全常量/秩不足/拟合失败记 null，报告 PCA 解释方差。一个任务有效 dense<8 时不汇总其拟合比例。用偏斜单峰与 off-PC 双峰做解释边界负对照；GMM 分量数不写成峰数。
5. 数组以 float32 保存，避免 fp16 量化把微小差异变成零；所有指标可离线重算。N<4、缺全步/配对数组、shape 不匹配、非有限值拒绝该决策并报告原因。先取每集有效决策中位数，再对集取中位数作为任务指标，避免长集占更大权重；覆盖门见 §3.3。

**探索性预注册**：`g_task = SR(K)−SR(k=1)` 来自 §1 历史阶梯与明确版本的参考（RC π0.5 v2）。分别在每个 policy 的 RC 13 任务上报告 Spearman ρ(median d_1,g)。bootstrap 20000 次、seed 20260919，以任务为簇抽样，保留各任务的整集数据；此区间条件于冻结的历史 g 点估计，不包含旧 SR 的估计误差，也不能排除旧硬件/服务差异。合并 26 点仅作附图，按任务把两个 policy 一起重抽。某 policy ρ≥0.6 且 95% 下界>0.2 → 本批任务上的一致关联；ρ<0.4、g 范围<0.15、d/g 常量或有效任务<10 → 无结论；其余（含未过区间门）→ 灰区。

LOTO 仅为描述性观察：每折在其余任务上拟合“d_1>阈值为 cliff(g≥0.15)”，阈值候选为训练 d 的相邻中点及 ±∞，最大化 balanced accuracy，同分取较低阈值以少误接纳减步；训练仅一类则该折不输出。报告留出任务的混淆矩阵及“预测非 cliff、实际 cliff”计数，不据此部署门控。LIBERO 两 suite 单列，不进 RC ρ。机制解释仅列 disp_K、ΔBIC、d_1 共现及待验证假设。

### Q-B：warm start 能否补回同预算减步尚存的缺口

研究量：同一 policy、固定每次决策 NFE m 下，warm 相对 plain 的成功率增益，以及其恢复 full-vs-plain 缺口的程度。任务清单与分组在读取本线任何闭环结果前冻结，不用本次 plain/full 结果重新筛选。选择依据为历史 seed=1,000,000+idx；本线使用新的固定 seed=2,000,000+idx，避免连用于选择的环境样本也原样复用。

| policy | 主 m / warm t | 固定 cliff 组（历史 gap，作选择依据） | 无明显正向减步折损的对照组 |
|---|---|---|---|
| π0.5 | 2 / 0.2 | CloseFridge(.50)、OpenCabinet(.18)、PickPlaceToasterToCounter(.16)、PickPlaceDrawerToCounter(.18) | PickPlaceSinkToCounter、OpenDrawer、PickPlaceCounterToStove |
| GR00T RC | 1 / 0.75 | PickPlaceDrawerToCounter(.30)、SlideDishwasherRack(.24)、TurnOnSinkFaucet(.22)、OpenCabinet(.18) | PickPlaceCounterToStove |

GR00T k=2 的对应四个 gap 只有 .12/.00/.04/.08，不能沿用 v2 的“主 m=2、预期三个 cliff”。上表只固定本研究样本；本线重跑若不复现 gap，不删任务、不换组，按以下规则给无结论。对照组简称 flat，只表示历史没有明显正向缺口，不预设 full 与 plain 等价。

- 同拓扑重跑所有臂：h100 server、同一固定 timan worker 映射；同代码/服务模式、checkpoint、预处理、replan=5、环境身份、推理精度、并发配置。π0.5 plain k={1,2,3} + full K=10，warm t={.1,.2,.3}；GR00T plain k={1,2} + full K=4，warm t={.75,.5}。每臂每任务 50 集，seed=2,000,000+idx，idx=0…49；**flat 任务的 plain m\* 与 warm t(m\*) 两臂加倍到 100 集（idx=0…99）**——v3 的配对不一致计数区间在 α=.05/12 下，n=50 即使零不一致也只能给 Δ 下界 −0.116，非劣门 −0.10 不可达（执行者复核，Clopper–Pearson 计算：n=100 时零/一/二个净不一致分别给 −0.060/−0.081/−0.099）；其它臂与 cliff 任务仍 50 集。旧阶梯只选组和历史参照，不与新 warm 配对。环境种子配对不保证不同轨迹的模型噪声逐次配对，不作此声称。
- warm 固定 top-1、`always_warm_start`、无门；库、检索权重、normalizer 见 §3.0/§3.2。plain/full 走同一实验服务的 staged 执行路径，绕过 cache verdict；warm 走冻结的 cache 路径。G2/manual parity 必须验证新路径与原基线服务一致。
- **严格等 NFE 准入**：每个 task×arm 的全部 50 个 accepted episode（上述 flat 主 plain/warm 臂为 100 个）均有完整逐决策证据，plain 每次为 m、warm 每次为 WARM 且实际剩余步数为 m。任何 MISS/全步 fallback、额外 stage-3 调用、计数缺失或偏离预算，都令该 cell 为“非等 NFE/证据不足”，保留全部结局作描述，不能删掉 MISS 决策/episode 后继续宣称等 NFE。判决所需任一 cell 不合格，该 policy 主结论为 inconclusive。报告 MISS 比例、逐决策平均实际 NFE、逐集总 NFE；“等”仅指每次决策的 stage-3 前向数，不声称等 episode 总算力或等时延。
- 逐任务按 50 个固定环境身份配对 (F_i,P_i,W_i)：`g=mean(F−P)`、`Δ=mean(W−P)`、`H50=Δ−0.50g`、`H25=Δ−0.25g`。恢复比 Δ/g 仅在 g>0 时描述，否则 null；不把估计出的 g 当无误差常数。单臂 SR 的 Wilson 仅用于展示；差值/恢复量使用同次配对重抽的三臂结果。
- **主估计为固定 cliff 组四任务的等权 macro**（条件于这四个任务，不外推任务总体）。每次 bootstrap 在各任务内独立重抽 50 个环境身份，三臂共用索引，然后求四任务平均 g、Δ、H50、H25。主比较 100000 次、seed 20260919。两个 policy 共 8 个 macro 量，加 4 个 flat 任务 Δ，共 12 个预定区间，各分配 α=.05/12。macro 用 percentile bootstrap 分位点 α/2 和 1−α/2；若某 macro 的 bootstrap 分布完全退化，该 policy 为 inconclusive，不用零宽区间作确定判决。flat 的正式 Δ 区间用配对不一致计数：n10=warm成功/plain失败，n01=相反；各自作 1−α/2 的 Clopper–Pearson 二项区间，再取 [L10−U01,U10−L01]，避免天花板/零不一致样本下 bootstrap 伪零宽。整体为经 Bonferroni 分配的近似同时推断（macro 的 bootstrap 仍为有限样本近似）；报告方法及其局限。逐任务 95% bootstrap 区间和其它 m 仅探索，不用于正式判决。
- **互斥判决，每 policy**（先过数据与等 NFE 门）：
  - supported：cliff macro 的 g、Δ、H50 区间下界均>0，且该 policy 每个 flat 任务 Δ 下界>−0.10。
  - not supported：cliff macro g 下界>0 且 H25 上界<0（在本组尚存缺口中，恢复不足四分之一）。
  - inconclusive：其余，含 gap 未复现、区间过宽或 flat 非劣未证实。
  - 独立 `harmful_on_flat`：任一 flat 任务 Δ 上界<−0.10；可以与 not supported/inconclusive 并存，不覆盖主判决。
- 报告全部 7/5 个任务和其它预算，不用 macro 掩盖个别任务损伤。50 集及多重校正可能导致无结论，尤其 flat 全部配对同结局也不足以在该置信水平确认 10pp 非劣；报告 cliff 恢复证据与 flat 不确定性，不把二者合成“warm 无效”。不事后放宽阈值/换主 m。推断范围限本批任务、本库、固定 t、无门 top-1。

### Q-C：合作者三问（描述性）

1. 分块事实见 §1；RC 每集环境步数与推理次数从 §3.2 新增 episode/per-decision 证据取，不从仅有结局的 journal 猜测。
2. Q-A 的 d_1、r_1、disp_K 逐任务并列 k=1 闭环 SR，区分小模型动作偏离和闭环成功率的关系；小 d_1 本身不能证明所有状态安全或解释闭环补偿机制。
3. π0.5 LIBERO-Object / LIBERO-Goal 阶梯 k={1,2,4,10}；spatial/libero_10 同 harness k=10 锚点。GR00T 无 object/goal checkpoint，不做。

## 3. 设计

### 3.0 环境与产物身份（Code 阶段生成 manifest；启动门拒绝缺项）

| policy × benchmark | checkpoint | H / 执行维 | full K / schedule | warm t（剩余步） | K_set | server / worker |
|---|---|---|---|---|---|---|
| π0.5 × LIBERO | `pi05_libero` | 10 / 前7维 | 10 / `pi05_v1` | .1/.2/.3（1/2/3） | 1,2,3,5 | weilandserver / timan107 |
| π0.5 × RC | RIT π0.5 RC checkpoint | 50 / 前12维 | 10 / `pi05_v1` | .1/.2/.3（1/2/3） | 1,2,3,5 | h100 / timan108 |
| GR00T × LIBERO | `n15_libero_{spatial,10}` | 16 / 前7维 | 8 / `groot_n15_k8_v1` | .875/.75/.5（1/2/4） | 1,2,4,6 | weilandserver / timan107 |
| GR00T × RC | `n15_robocasa_tp … checkpoint-60000` | 16 / 前12维 | 4 / `groot_n15_k4_v1` | .75/.5（1/2） | 1,2,3 | h100 / timan108 |

`envs.json` 实际展开为 6 项（两 policy 各 RC、spatial、libero_10）；Q-C object/goal 的非 shadow 配置另列。每项绑定 checkpoint 绝对路径/sha、code commit、adapter 与 action transform、归一化空间、动作顺序、mask/σ/w、K/schedule、库/检索权重/normalizer 路径与 sha、服务精度/并发、环境版本与 horizon/replan、canonical task 名与 id、seed/init 和 parent pool sha、layout/style/pin、server/worker。无法加载或身份不一致即拒绝启动，不按文件名推定正确。

RC 正式 shadow 使用 base_seed=2,000,000、idx=0…9，与 Q-B 的前10个身份相交；Q-B idx=0…49。smoke 使用独立 experiment/arm 命名空间和 RC base_seed=3,000,000，其数据不进入正式分析。LIBERO 仍按现有 suite 的冻结 init pool 选索引，shadow 0…9；manifest 明记此规则而不套用 RC 的种子段。

- RC 库固定使用 `emit_rit_rc.w13_spec(teacher)`：`/data/robocasa365_cache/cache_artifacts_w13/{teacher}_spatial_pool_16_w13_full.pkl`，teacher 为 pi05/groot_tp；校验 full 快照、live schedule、每任务来源及 50 条的构建证据。normalizer 以该完整 stem 查原 RIT calibration，并冻结原文件 sha。权重 CID：pi05=`grid_vision_1@87_robot_state@12`；groot_tp=`grid3_vision_0@12_vision_2@37_robot_state@50`（`emit_rit_rc.FROZEN_WEIGHT_CID`）。
- 合并 W13 库没有全局 pin_id；main 原本不 pin，PnP 原本 pin。核验逐任务构建来源与 rollout 的 pin，不给合并库伪造全局 expected_pin_id，也不使用仅 PnP 的 `warm_teacher_spec` 默认 spec。
- LIBERO 使用对应 policy×suite 的既有 RIT 模板和库，不把 RC 的 S6/full 名称套到 LIBERO（例如 GR00T libero_10 模板使用 W13 S3）。Code 从既有模板解析并固化真实路径/sha，逐项检查所需 noise_action 快照齐全且 schedule 与上表一致；缺快照拒绝该环境放量，不能静默改库或改 t。
- RC main 8 任务无 pin，PnP 5 任务使用 `exp/robocasa365/config/pnp_pinned_objects.json` 的规范路径及 hash；执行前用既有 resolver 解析。Q-B π0.5 main={CloseFridge,OpenCabinet,OpenDrawer}、PnP={PickPlaceToasterToCounter,PickPlaceDrawerToCounter,PickPlaceSinkToCounter,PickPlaceCounterToStove}；GR00T main={SlideDishwasherRack,TurnOnSinkFaucet,OpenCabinet}、PnP={PickPlaceDrawerToCounter,PickPlaceCounterToStove}。task id 沿用完整 roster 的映射，子集不重新编号。

### 3.1 shadow 服务（Q-A、Q-C.2）

`DiagShadowRecorder` 沿用 X15 的三条不变量：执行动作不变、生产 RNG 不变、shadow 异常只丢标签不丢动作。接入 staged 执行完成处，拿到同一请求的 stage2 与原始 stage3；不重复一次完整生产推理来获取它们。hook/组合均在 exp/step_diag 内，保留生产推理锁。

记录合同（可 JSONL metadata + 分片数组，二者以相对路径和 sha 绑定，必须可离线重算）：

```
experiment_id, launch_id, arm_id, env_id, config_sha,
task_name, task_id, task_uid, attempt, decision_idx, env_seed, init_idx, lane, pin_id,
status=ok|error|finalize, error_reason,
terminal, outcome, n_decisions,                              # server finalize
n_primary=4, n_full=4|32, k_set, noise_ids, dense,
top1_score, top1_entry_id, warm_status,
a_exec, a_full[N_full,H,D], a_k[k][4,H,D], a_warm[t][H,D]|null, # float32
wall_ms, shadow_nfe
```

- 私有噪声身份由实验种子、env/task/seed/attempt/decision/sample 唯一确定并记录；不同 k 共用 sample noise。噪声与模型的 shape/device/dtype 合同一致，不调用全局 seed/reset。
- π0.5：`run_stage3(stage2, noise=z, num_steps=k)`；warm `run_stage3_from(stage2,start_x,start_t,num_steps=10)`；生成器显式传入噪声构造。GR00T：`runner.session()` 内 `staged.denoise_loop(head, runner._head_inputs(stage2), stage2.action_inputs, noise=z, num_steps=k)`，每次重建 head-input mapping；按现有 stage3 输出合同提取 action。warm `run_stage3_from(...,schedule=live)`；live head 的 `num_inference_timesteps` 全程不改，独立 venv 不变。
- top-1 经独立只读 `CacheOrchestrator.check()`（always_hit 取候选但不 apply；与 GrootRitShadow 同法）；不更新库/检索状态、不让 shadow 改写生产 verdict 或 accounting。
- parity 在固定观测、固定生产 RNG 初态及同一确定请求顺序下对比有/无 shadow 的 a_exec；双连接以控制器固定交错顺序覆盖元数据隔离。真实异步 rollout 不要求不同调度下两次运行逐位一致；实际服务仍须保持每条请求原生产动作，比较要排除请求重排混杂。
- 额外 NFE（不含生产全步，warm 全部有候选时）：π0.5 普通 `4×10+4×11+6=90`，dense 加 `28×10` 为 370；GR00T RC 普通 `4×4+4×6+3=43`，dense 155；GR00T LIBERO 普通 `4×8+4×13+7=91`，dense 315。记录实际采样矩阵与计数。RC 13×10×2=260 集；LIBERO 两 suite×10任务×10 init×2 policy=400 集，合计 660 集；吞吐与存储 ETA 由 smoke 实测，不承诺固定小时数。

### 3.2 driver、闭环臂与证据落盘

**入口为新增 `exp/step_diag/run_diag.py` + `worker_entry.py`**，复用 `ConductorDriver`、`RobocasaEpisodeRunner`、`LiberoEpisodeRunner` 和现有 adapter/worker setup；不复制 rollout 循环、不修改旧 runner/driver。新 strategy 负责本实验 task 子集、10/50 集预算、固定 arm_id（如 plain_k2/full_k10/warm_t02）及 config digest。

现成 `run_rc_client.sh → run_ws_search.py` 只作 CLI/环境配置参考，不能直接承载本矩阵：其 pinned 路径调用 `assert_pnp_eval_identity` 强制完整五任务、每任务 50 集，拒绝本线 PnP 子集和 shadow 10 集。新 strategy 使用本计划自己的显式集合校验，复用 pin 解析/realized identity 核验；不关闭或 monkeypatch 旧 PnP guard。LIBERO shadow 也用 conductor；Q-C 阶梯/锚点沿用既有 baseline client 与 `serve_pi05_ksweep.py`，同 suite 所有 k 的 harness 保持一致。

证据链：

1. `experiment_id`/config digest 固定；每次 driver 启动生成 launch_id，driver 创建后将其公开 `driver.run_id` 与 launch_id 的映射持久化到 manifest，再派发任务。task.extra 携带 launch_id/config/arm；轻量 client proxy 在 episode_start 中加入这些字段，不依赖旧 RC runner 会自动转发任意 extra。
2. wrapper 委托原 runner，client proxy 计每次 infer、记录响应 `__hit_meta__`、独立 decision_idx；runner 返回后将 reported_n_steps、n_decisions 和环境身份作为 episode_summary 行加入 EpisodeResult.per_step_rows。RC 通过现有 `gym_make` 注入只计数的 env proxy，按 reset 后真实 env.step 次数另记 n_env_steps（含成功的最后一步）；不直接把 runner 的 step 索引当实际动作次数。新 driver 显式配置 `per_step_writer` 持久化这些行，并使用 conductor 写入的权威 run_id/task_uid/attempt。旧 run_ws_search 未装此 writer，journal 本身也不含逐决策元数据或 n_steps。
3. shadow 数组和 finalize 在 server 本地落盘；proxy 按现有签名转发 `episode_end(success=...)`，不向此接口传 n_steps/extra。server 用本连接记录器的决策计数与 success 写 finalize，worker 的 summary 单独记录计数，分析时相互核对；正常超时失败也封口，异常/断连缺 finalize 则不准入。分析通过 launch_id→driver.run_id 映射将 sidecar 与 journal 的 accepted terminal 精确对齐；不把静态 run prefix 当成 driver UUID。重启读入所有被同一 config manifest 声明的 journal 分片；未终结 attempt 丢弃。跨 launch 重复 accepted 环境身份必须报冲突，不能随意取最新；断点续跑只派发未完成身份。
4. 闭环 NFE 证据含 `hit_type/start_t/schedule_id/executed_steps/n_stage3_calls` 与 infer 序号。π0.5 在实验层实例包装 `denoise_step` 计真实调用次数；GR00T 记录 `GrootStage3Output.steps_run`，并在 manual 样例以实际 denoise_step 调用计数交叉校验；不是只从 YAML 推算。plain/full 也记录同样证据，记录本请求全部 stage-3 调用而不只最后一次。锁内 instrumentation 不改变计算，manual parity 与已知 full/warm/fallback 样例校验计数；若编译/图捕获绕过观测则该配置拒绝准入，不以名义步数代替。
5. terminal journal 作为结局权威，与 episode_summary/finalize 一致；成功或正常超时失败都保留。传输/进程失败按 conductor retry，不能当普通失败填零，也不能无声丢集。日志/数组拉取后校验 hash；per-step writer 失败或证据缺失不能凭 summary 宣称完成。

配置发射：RC 逐 teacher 调 `emit_ws_warmstart_yamls.build_warm_cell/verify_warm_cell`，spec 用 §3.0 的 w13_spec，权重/normalizer 用原 RIT 值；不用共享 timesteps 的 emit_warm_arms。LIBERO shadow 从既有模板构建只读检索配置并做同样 schedule/payload 校验。`config/index.json` 绑定所有 yaml/config/库/normalizer hash 与任务、拓扑和 seed 清单。

跨臂配对键为 `(policy,benchmark,task_name,lane,layout,style,env_seed,init_idx,pin_id)`（LIBERO 加 suite/parent-pool sha）；checkpoint/预处理等由公共 manifest 一致性门约束。task_uid 可以含 arm，不用它跨臂配对。

### 3.3 分析与准入（脚本在 analysis/）

- shadow：先匹配 accepted terminal，再检查该 episode 的 finalize/summary 与连续 decision_idx=0…n_decisions−1。缺行、重复行、冲突身份/sha 或无 finalize 则整集不准入；status=error 保留为显式无效标签，不静默漏行。每集有效主指标决策覆盖率≥90%，每任务≥8/10 集才输出任务指标；完整列出 invalid/error、coverage 与成功/失败分层缺失率，承认标签缺失可能相关于状态。不足的任务不补成零；RC 有效任务<10 则该 policy Q-A 无结论。
- Q-B：先验定每个 cell 的原始 50 个（flat 任务 plain m*/warm t(m*) 臂为 100 个）身份集合与 accepted terminal，再与 episode/per-decision 证据一一核对；不允许仅 common-complete 子集改变研究样本。按固定组、严格等 NFE 门执行 §2 判决；保留全部逐任务表、配对区间、恢复量及计数。图/Markdown 进 analysis/，逐任务 JSON 进 data/。
- 相似度分箱仅覆盖 shadow 的预定 idx=0…9 与 Q-B 相交的子队列，不声称覆盖 50 集。每集分箱分数为先行完成的教师 shadow top-1 score 中位数，记录有效 score 数；两闭环臂共用身份。按 policy×task 在这些分数上取中位数分为两箱（等于边界放低箱，不拆相同值），每箱≥4 个共同有分数的身份才显示 episode Δ；其余不绘制。冻结分箱后再读 warm 结局，报告缺分数、样本量与选择范围。这里只是固定教师轨迹上的共同参考属性和结局关联，不是部署状态的因果风险曲线。

## 4. 文件与接口

| 文件 | 内容 |
|---|---|
| `exp/step_diag/__init__.py` | 包说明 |
| `exp/step_diag/diag_shadow.py` | Recorder + 两策略适配；sample_full/sample_k/resume/top1；独立 RNG、数组/finalize 落盘 |
| `exp/step_diag/serve_diag_pi05.py`、`serve_diag_groot.py` | shadow/plain/full/warm 服务装配，复用 staged/cache；固定 env/config/arm；dense 规则固定，不提供改变主采样数的自由参数 |
| `exp/step_diag/run_diag.py`、`worker_entry.py` | strategy/manifest、conductor 装配、已有 runner 委托、client proxy、per_step_writer 与 resume 校验 |
| `exp/step_diag/emit_arms.py`、`config/` | 逐 teacher 发射、envs/index、固定任务与资源身份 |
| `exp/step_diag/analysis/analyze_shadow.py`、`aggregate_arms.py` | 准入、可重算指标、统计与判决；plot_step_diag.py 为绘图工具 |
| `exp/step_diag/ops/` | server/worker 启动、拉取和汇总；运行产物在 data/（gitignore） |
| `tests/exp/step_diag/test_diag_shadow.py` | 执行动作/RNG 不变、异常 error/finalize、私有同噪声配对、在线 dense、GR00T fresh mapping/live schedule 与双连接隔离 |
| `tests/exp/step_diag/test_metrics.py` | 非执行维/窗外不变、常量执行维不得被删除、逐时刻 vs 展平反例、d_K=0、同噪声有分布离散但零误差反例、disp=0 的 null 比值、样本不足拒绝、PCA 负对照 |
| `tests/exp/step_diag/test_analyze.py` | 配对 triple 重抽、固定组不依新结果筛选、12 区间分配与 flat 不一致计数区间、三分支互斥、天花板/零宽 bootstrap/宽区间/g≤0/flat 损伤；任何 MISS 降级且不删 episode；10/50 分箱覆盖边界 |
| `tests/exp/step_diag/test_driver.py` | 子集保留原 task id/pin、10/50 集精确集合、arm 隔离、重启 run_id 映射、accepted attempt join、重复 accepted 冲突、缺行/finalize/summary、writer 失败与证据不足拒绝；成功最后一步计数与 episode_end 现有签名 |
| `tests/exp/step_diag/test_emit_arms.py` | yaml 过 validate_cache_config；w13_spec/normalizer stem、真实 schedule 与 payload、teacher 各自 t、混合库不能强加全局 pin |
| manual parity（两机、独立依赖环境） | 固定输入/噪声下所有 k 对既有 k-sweep 与 k=K parity；所有 t/fallback 的执行步数计数；重复 stage2 不污染；有/无 shadow 的生产输出及受控双连接隔离 |

不改 src/、exp/nfe_baseline/、exp/robocasa365/ 既有文件，仅导入和实验层组合。此处批准接口与验收合同，实际实现仍须 G2 审查。

## 5. 执行顺序与预算

| 步 | 内容 | 主机 | 门 |
|---|---|---|---|
| 0 | 开跑时重新探查设备、进程/端口与 artifact；先前空闲记录不当作当前事实；仅清理核实属于本线的残留 | 两 server / timan | 资源与身份 manifest 完整 |
| 1 | Code（§4）及 CPU 测试，索引同步 | 本机 | → G2 |
| 2 | G2；Verify（WA §2.7 全仓 uv run pytest、staged API 检查及全部 manual parity） | 本机/两机 | G2 APPROVED + Verify green |
| 3 | 准入 smoke：6 env 各 1 集 shadow、受控双连接；两 teacher 的每个 warm t 各5集、覆盖 main/PnP；driver 重启/证据 join 和零 MISS | 两机 + timan | 全过再放量；重估吞吐/磁盘/ETA |
| 4 | shadow：RC 260 集、LIBERO 400 集；固定 idx=0…9；先冻结分箱身份 | h100+timan108 / weilandserver+timan107 | 660 集及 §3.3 覆盖门 |
| 5 | Q-B：π0.5 对照 7×50×4=1400 + flat 3 任务 plain m\* 加 150，warm 7×50×3=1050 + flat warm t(m\*) 加 150；GR00T 对照 5×50×3=750 + flat 1 任务 加 50，warm 5×50×2=500 + 加 50 | h100+timan108 | 共 4100 集；task→worker 配置在所有臂固定 |
| 6 | π0.5 object/goal 4×2×500=4000；spatial/10 k=10 锚各500 | weilandserver+timan107 | 共5000集，可与步5并行 |
| 7 | 拉取/校验、分析/报告；同步 logs/README、handoff 与记忆 | 本机 | 数据准入和预注册判决 |

worker 只在 timan107/108，server 只在 h100/weilandserver（owner 指定）。服务器按 policy 顺序切换，避免显存互抢；一切并发/拓扑在同一比较内固定。正式预算 9760 集（shadow 660 + Q-B 4100 + Q-C 5000），不含 smoke、重试和已有历史数据；dense 为额外前向，不另算 episode。

## 6. 交付

1. `exp/step_diag/analysis/step_vs_warmstart.md`：Q-A 分 policy 关联及条件区间、逐任务诊断/LOTO/机制假设；Q-B 固定任务组、完整逐任务和 macro 结果、等 NFE 审计、互斥判决/flat 损伤、限定10集的分箱；Q-C 三问；实际成本、缺失与解释边界。
2. `exp/step_diag/data/`：shadow metadata/float32 arrays、run-plan/manifest、全部 journal/per-step/episode-summary、阶梯补档与分析 JSON，均 gitignore；config 中保留可追溯 hash。
3. 四层设计输入仅为候选依据：小 d_1 表示同噪声执行维动作偏离小；它不自动授予任务/状态“减步可整段接纳”。warm 收益以 Q-B 闭环结局为据；离线量替代阶梯尚无本线之外的证据。

## 7. 风险与处置

| 风险 | 处置 |
|---|---|
| 同噪声误差与条件离散度混淆；退化维丢失 | d_k 不扣 floor；零分母显式 null；常量执行维仍有权重；统一模型动作空间 |
| shadow 改动作/RNG、GR00T 输入原地污染或异步调度混杂 | 私有 generator、新 head-input mapping、同锁；固定输入/请求顺序 parity，真实 rollout 不作不受控的逐位跨运行承诺 |
| 阶梯历史跨硬件、任务选择复用结果 | Q-A 限探索；Q-B 固定历史分组，全部臂同拓扑新跑、不重选 |
| journal 无 NFE/步数、重启 UUID 或重试混入 | 新实验 driver 装 writer/client proxy；launch→driver 映射、严格 accepted join、完整集合/计数与冲突拒绝 |
| MISS/额外前向破坏等 NFE | 任何一个即 cell 降级；所有结局仍报告，不做选择性删除 |
| 50 集噪声、多任务/预算尝试 | 主 m/固定组/12 个校正区间冻结；三臂联合重抽恢复量；低功效如实给 inconclusive |
| 混合 W13 库无全局 pin、LIBERO 库 stage 不同 | 逐环境模板/构建证据/sha 冻结；逐任务 pin，schedule 与 payload 启动断言 |
| 诊断缺失、10集分箱不代表50集 | 覆盖门与分层缺失表；分箱限共同固定10集、仅描述 |
| GR00T 无 object/goal checkpoint；资源状态变化 | 仅 π0.5 做补阶梯；开跑重新探查，不全局清理他人进程 |

## 8. 不做

- 不改训练/参数化，不做 CUDA graph 提速或 wall-clock 基准；不重跑已有 LIBERO spatial/10 全阶梯和 RC 13任务全阶梯，仅跑上述新增对照档。
- 不做带门 RIT 臂、不重新标定；Q1（按相似度门控减步）不单独验证。
- 不根据本批相关性、LOTO 或 GMM 分量数声称新任务可跳过阶梯、已识别机制或已证明状态级安全。

## 9. 实现说明（G2 修订；保持 v3.1 研究合同）

本节同步 G2 修复后的工作树。owner ziyanglin 在本会话明确指示「在这轮审查后直接修改到你觉得可以通过的地步」「原本的暂存，把自己的修改留在暂存区外」。依此覆盖本轮 Review Authority 的通常只读与最终 staging 规则：执行者原始提交材料保留在 index，审查者直接修复及记录全部在 index 外；不构成另一次独立审查者复核。§2 的 50/100 集文字已消歧，固定研究任务、统计阈值及预算未改。

### 9.1 文件与证据接口

- §4 的 `diag_shadow.py` 分为 `recorder.py`（进程级落盘、每连接 `DiagSession`）、`pi05.py`、`groot.py`；`envs.py` 冻结环境/任务/资源身份，新增 `evidence.py` 统一校验内容寻址 manifest 与跨臂可比身份。GR00T plain k=1 不构造要求至少两步的 snapshot schedule，直接走 staged loop；每次 shadow 重建 head inputs。shadow 检索失败在教师执行后写 error，不改变执行动作。
- RC 使用 `run_diag.py` + `worker_entry.py`，LIBERO 使用新增 `run_libero_diag.py`。两者均委托生产 runner，安装 client proxy、环境步数计数、worker summary 和 conductor writer；accepted terminal、launch→driver UUID、任务身份、server finalize、数组 digest、逐决策 NFE 必须一致。LIBERO worker 独立验证 held-out 池内容 hash，保持 simulator 与 serving 的依赖环境隔离。
- `emit_arms.py` 逐环境发射并逐条验证库中 action/snapshot 的有限值、shape、完整 schedule；执行 mask 来自 adapter。`--library env=path` 同时改写实际输出 YAML 和校验目标。服务启动绑定实际加载的 checkpoint、模型配置、归一化资产、检索库、代码内容、runtime/device；保留 `manifest_<config_sha>.json`，每行证据引用同一 digest，不能只凭路径或 commit 声称同模型。
- `analysis/analyze_shadow.py` 的正式 CLI 必须提供 `--driver-dir`，先过 accepted/资源/权重库/完整身份门，再按严格 4/32 个全步样本、全部 k、私有噪声身份和 shape/finite/digest 计算标签；RC 有效任务少于 10 个无关联结论。成功/失败/unknown 的缺失表以 driver 的冻结全集为分母，排除未接受重试，整集 server 缺失仍计入；observed error 与 missing decision 分开报告。
- `analysis/aggregate_arms.py` 检查固定 50/100 集原始集合及全部 NFE，不删坏样本后重新配对。flat 主 plain/warm 的正式区间使用全部 100 对，full 的 50 集仅作参考；cliff 保持 50 个三臂联合样本。跨臂必须同 experiment/model/runtime/worker；其它预算输出探索性 paired CI、Wilson、逐集总 NFE。四个 macro 量均检查 bootstrap 退化。分箱先在完整的预干预 shadow 分数集合上冻结阈值，再关联完整环境身份的结局。
- `ops/` 修正 tmux/JSON 参数传递、readiness 失败退出、环境分开的 shadow 输出目录、conductor 启动/拉取/分析链；runbook 给出 CLI、资源门与 manual parity。分析 JSON 进 `data/`，Markdown 进 `analysis/`。

### 9.2 原声明偏差的裁定

1. **服务并发与映射：接受并收紧。** π0.5 所有臂 `--non-concurrent`、GR00T RC 非 concurrent；GR00T LIBERO 保留工厂的 concurrent 支持，本实验同样每 endpoint 只派一个 worker。使用规范 roster task ID modulo 有序 server slot 数分派，子集或 flat 100 集不会重排 task→worker；比较臂保持相同 slot 顺序并核对 worker runtime。每连接元数据隔离由 CPU 测试覆盖，真实服务受控连接/请求顺序的验收仍在 Verify/smoke。
2. **LIBERO 独立 main.py：撤回该偏差，恢复计划。** `run_libero_diag.py` 使用 `ConductorDriver` + `LiberoEpisodeRunner`；直接调用指定 simulator Python，不依赖 conda 启动命令。server finalize 无法单独证明 accepted attempt，故与 RC 同样接入 journal/summary 权威证据，冻结 10×10 身份与池 hash。π0.5 resize=224、GR00T=256、replan=5。
3. **π0.5 plain/full 实例级 stage3 钉步：接受。** `Pi05DiagInterceptor` 实例钉 `_stage3_fn`，记录真实 `denoise_step` 调用数；分析要求 `executed_steps == m`、`n_stage3_calls == 1`。CPU fake model parity 与异常/NFE 负对照通过；真实 checkpoint parity 不由 CPU 结果代替。

### 9.3 测试与 Verify 接口

§4 的测试落为 `test_metrics`、`test_recorder`、`test_pi05_interceptor`、`test_groot_policy`、`test_run_diag`、`test_worker_entry`、`test_analyze`、`test_aggregate`、`test_emit_arms`、`test_serve_parse`。新增 `test_revision_contracts.py` 覆盖实际 GR00T k=1 builder、LIBERO conductor/pool/依赖隔离、完整 recorder→authority→标签准入、资源/身份拒绝、检索异常不干预教师、缺失表分母等回归；独立审查探针放在 gitignored `tests/review_tests/`，不进入 index。

新增 `test_parity_manual.py`：各自 serving Python 在六个环境上读取真实 checkpoint/观测，验证所有 k/full 的生产输出、shadow RNG/输出不变、全部 warm t 的 snapshot resume 和实际调用计数；GR00T 另用 action encoder hook 交叉核对 NFE。指定 `--run-manual` 后缺 GPU/资产必须失败。调用方法及既有 GR00T upstream/HDF5 parity 测试见 `exp/step_diag/ops/README.md`。本轮仅完成 G2 的 CPU/代码审查；真实 GPU parity、WA §2.7 全仓 Verify、两机 smoke 和正式 rollout 未执行。

### 9.4 执行者对 G2 修订的复核（2026-09-20，owner 指示「不能直接接收，亲自审查并修改」）

逐 hunk 复核 Codex 的 24 个改动文件 + 4 个新文件。**接受**：B1（flat Δ 用全部 100 对 plain/warm，原实现经三臂交集截到 50——确为缺陷）、B2/B3/B6/B10（journal↔launch↔summary↔finalize↔decision↔数组的一致性、run_id 含 lane+experiment/config 摘要、dense/噪声身份/shape/finite 复核、按 driver 冻结全集分层、分箱阈值先冻结）、B4（GR00T plain 不构造 snapshot schedule）、B7（`run_libero_diag.py` conductor + 生产 `LiberoEpisodeRunner`，逐集关闭 env 修复原路径的 env 泄漏）、B8（task_id mod 槽位固定映射）、B9（检索异常只丢标签）、B11（ops 引号/readiness）、N1（探索性区间、Wilson、逐集 NFE）。

**改回 / 收窄**（依据 owner 2026-09-12 裁定「provenance / all-or-nothing 门禁对结果零贡献且拖慢开工」，以及一处会让正式分析必然失效的缺陷）：
1. `envs.resource_identity` 原在**每次 server 启动**对整个 checkpoint 目录（7 GB+）做 sha256 并 unpickle 整个库（W13 28 GB）——改为 `checkpoint_digest`（文件清单+大小+小文件内容，不读权重）和 `library_digest`（`<pkl>.sha256` 侧车，只算一次；`freeze_weights` / Step 0 预写）；库的 payload 合同检查只在 `emit_arms --check-libraries`。
2. `RunManifest.config_sha` 原含 `code_commit` 与 runtime（host/GPU/torch/源码摘要）——任何 commit 后重启 server 都会换 config_sha，而 config_sha 已嵌入 run_id/task_uid，等于「重启即整 cell 重跑」；改为只哈希服务合同字段，commit / runtime 记录不入哈希。
3. 跨臂可比性门 `comparison_problems` 原要求所有 cell 的 `worker_identities` 恰为单一值，而 `worker_runtime` 含 `gpu_slot`（CUDA_VISIBLE_DEVICES，按 worker 各异）——真实多 GPU 机队下**每个 cell 都会被判 mismatch，主判决必然 inconclusive**（Codex 的 fixture 只有单一 worker runtime 所以通过）。改为：门只看 checkpoint 身份 + 环境合同 + experiment 命名空间；worker 岛 / 服务 runtime 作 `comparison_notes` 报告不门控；新增测试 `test_runtime_and_worker_facts_are_notes_not_gates`。
4. `evidence.manifest_problems` 不再要求 env 字典（含 server_host/worker_host 部署字段）逐字相等，改比环境合同字段；不再把 runtime 源码摘要缺失当拒绝理由。
5. `verify_library` 的 GR00T `schedule_id` 检查加入库级 `artifact_meta` 回退（in-memory backend 的同一默认规则），避免旧库 payload 无 schedule_id 而误拒。
6. `analyze_all.sh` 生成的拼接表改名 `step_vs_warmstart_tables.md`，不覆盖手写终报 `step_vs_warmstart.md`。
7. `test_parity_manual.py` 两处 E731 改 def（ruff 全过）。

复核后：`uv run pytest tests/exp/step_diag` 84 passed / 1 skipped（manual）；ruff / `git diff --check` / `bash -n` 全过。全仓 Verify 见 §9.5。

### 9.5 Verify（WA §2.7，2026-09-20 CDT）

`uv run pytest -q --continue-on-collection-errors`（裸全量；不加 `--continue-on-collection-errors` 时被两处 HEAD 既有收集错误中断）：**6153 passed / 24 failed / 78 skipped / 2 errors，22m38s**。24 个失败逐条归因：15 个在 gitignored `tests/review_tests/`（其它线的审查探针，不在仓库）；5 个在 HEAD 干净 worktree 同样失败（`test_ws2_evidence_runner` ×2、`test_prebuilt_matrix_backend` ×2、`test_rit_pl::test_sonly_note_compiles`）；2 个为 tokenizer 网络下载（`test_robocasa_policy_config`，既有 GCS 网络失败）；2 个为既有测试顺序干扰（`test_groot_concurrent_serving`，`gr00t.__spec__ is None`，单独运行通过）。2 个收集错误：`test_review_cache_prune_g2.py`（需 `REVIEW_SCRATCH`）、`test_bench_groot_stages.py`（HEAD 缺 `SCHEDULE_ID`）。**本变更集（`exp/step_diag`、`tests/exp/step_diag`）在全量运行中零失败**；`src/` 未改。

## Review Log

### G2 Round 1 — Reviewer — APPROVED — 2026-09-20 11:31 CDT

**Intake / authority**：Review Authority，L2 / G2；目标为本计划 v3.1 对应的 `exp/step_diag/`、`tests/exp/step_diag/` 与文档差异，基线 HEAD `f32982b`。按 `CLAUDE.md`、`WORKING_AGREEMENT.md` 与 `protocols/review_authority.md` 初始化，读取计划、相关架构/证据合同和被审实现、测试后独立验证。本轮 owner 明确授权直接修复及保留原始 index，覆盖审查法通常的只读、交回执行者修复和最终全量 staging 要求；没有 commit、push、部署或真实 rollout。不存在未获授权的章程越权。

**原始材料**：执行者 CPU 测试通过，但独立设计的 20 项合同探针在修复前全部失败，揭示正式分析可能误接纳数据、flat 样本截断以及真实服务入口失败。原始材料已暂存；下述修复、回归测试及本记录均未暂存。初始及收尾 `git diff --cached --binary | sha256sum` 均为 `7d29f5855fa0576317835fc715d4926028cddf8880d7139f87017458e288f676`；`tests/review_tests/` 被 ignore 且未进 index。

**发现与闭环处置**（编号仅属于本次 G2，不沿用 G1 编号）：

- [Blocking] [Concern] G2-B1：flat 主 plain/warm 100 集被 full 的 50 集交集截断 — reasoning: 改变预注册样本，削弱非劣检验，后 50 集结局不能影响正式判决。**Closed**：独立构造全部 100 个 plain/warm pairs；full 50 集单列参考，flat 不混报三臂恢复量；覆盖只改变后 50 集的负对照。
- [Blocking] [Concern] G2-B2：证据链存在 fail-open 路径 — reasoning: 缺 summary、accepted/UUID/arm/config 戳、数组/hash、真实 NFE，或重复 accepted/finalize 时，不能仅凭 server 行给等 NFE 结论。**Closed**：完整 journal→launch→summary→manifest→finalize→decision→array 校验；完整 50/100 原始集合，任何 MISS/fallback/额外 stage3/错误计数整 cell 降级，结局仍保留；负例均先验证完整 fixture 可准入再删改证据。
- [Blocking] [Concern] G2-B3：run-plan/resume 身份与 main/PnP 命名不完整 — reasoning: 跨 lane/config/experiment 复用或重启可能错配环境，非规范 pin 不能用于配对。**Closed**：lane 和 config/experiment digest 入 run identity，正式任务、seed、预算、layout/style、规范 pin、原 task ID 均冻结；配对使用完整环境身份，driver UUID 与本 launch 的 UID 集合都须匹配。
- [Blocking] [Concern] G2-B4：GR00T plain k=1 装配构造了不支持单步的 snapshot schedule — reasoning: 主预算 m=1 的真实服务无法启动，fake policy 测试未覆盖工厂路径。**Closed**：plain/full 直接 staged loop，warm/shadow 才需 snapshot schedule；实际 RC served-builder 回归覆盖 k=1，live K/H/D 断言，fresh mapping 与 batch-one helper 合同复核。
- [Blocking] [Concern] G2-B5：模型/配置/库身份未充分绑定实际内容 — reasoning: 路径或 commit 相同不能证明实际 checkpoint、normalizer、快照、runtime 一致；路径 remap 若只校验不写 YAML 会加载旧库。**Closed**：实际模型资产、代码、配置、runtime/device 与库内容 hash 绑定；不可变 manifest 与逐行 config digest、跨臂比较门；effective YAML remap 和逐 entry 有限值/shape/schedule 检查。
- [Blocking] [Concern] G2-B6：Q-A 可绕过 authority、严格采样和足量任务门 — reasoning: 无 journal、缺样本/错误 shape/hash/噪声身份或不足 10 任务不能给正式标签/关联；LIBERO 不同 init 不能复用同一私有噪声身份。**Closed**：正式 CLI 强制 driver evidence；完整 4/32、k/warm 集、finite/digest、dense 选择与 noise IDs 核验；噪声含 init/pool；任务门与错误拒绝回归通过。
- [Blocking] [Concern] G2-B7：LIBERO 独立 main.py 偏离 conductor 证据合同 — reasoning: 仅 server finalize 不足以证明 accepted attempt，正常失败与基础设施失败不能混淆。**Closed**：撤回偏差，新增实验 strategy/spawn/runner wrapper，复用生产 `LiberoEpisodeRunner` 与 conductor；原始 init/pool hash、summary、terminal 一致，直接 simulator Python 可在无 conda 的主机使用，依赖隔离测试通过。
- [Blocking] [Concern] G2-B8：task→worker 随子集/预算改变，跨臂可比性不足 — reasoning: flat 100 集或任务子集不能改变固定环境任务的 worker，runtime/资源不同不能仍称同拓扑比较。**Closed**：规范 task ID modulo 固定 server slot 数；每 endpoint 一 worker，主/次预算均检查单一且相同的模型与 worker identity、experiment。
- [Blocking] [Concern] G2-B9：shadow 检索异常可能阻断教师动作 — reasoning: 诊断失败必须只丢标签，不能使生产 full rollout 失败。**Closed**：shadow CP1 强制原生产 MISS/full 语义，捕获检索异常并在教师执行后记录 error；CPU 验证教师动作、RNG 与 NFE，warm 正式臂的执行错误仍正常传播。
- [Blocking] [Concern] G2-B10：缺失率与分箱的身份/分母不完整 — reasoning: 重试行不能充当正式集，整集缺失不能从缺失率分母消失；结果交集不能反过来移动先验分箱阈值。**Closed**：缺失率用冻结 driver cohort 和权威 terminal，缺结局单列 unknown；原始 server 计数单列，error/missing 分开；分箱在全部预干预分数上冻结，绑定完整环境与模型/runtime 身份后关联结局。
- [Blocking] [Concern] G2-B11：启动脚本 JSON/tmux quoting 与 readiness 行为不可靠 — reasoning: flat 100 集预算参数可能损坏，陈旧 claim 或超时不应被当成可用服务，产物必须按环境隔离。**Closed**：argv 数组及 `%q` 传递、失败非零退出、分环境输出、LIBERO conductor launcher；实际脚本通过 stub tmux 参数往返测试和 `bash -n`。
- [Non-blocking] [Suggestion] G2-N1：完善完整统计与可执行 parity 入口 — reasoning: 次预算、Wilson、逐集总 NFE、H25 bootstrap 退化及真实依赖验收必须可复核。**Closed**：补齐对应报告/门与回归；新增六环境 manual parity，runbook 明确实际 GPU、双连接/固定请求顺序、上游 HDF5 与 smoke 验收仍在 Verify。

**G2 四项 checklist**：

| 项目 | 裁定 | 依据 |
|---|---|---|
| 与获批计划一致 | PASS | 保留 v3.1 固定组/预算/统计门；修复 flat 100 集，恢复 LIBERO conductor，正式数据门与 §2/§3 一致；三项实现偏差裁定在 §9.2 |
| 测试覆盖及通过 | PASS（G2 范围） | 81 项公开 CPU 测试 + 20 项独立探针通过；真实 GPU manual 明确跳过；有完整正例、删证据/篡改负例与实际入口测试，未用 fake parity 冒充模型 parity |
| 文档及索引 | PASS | 本文 §9 / Review Log、`logs/README.md`、ops runbook 同步；新入口、工况、产物与 Verify 操作已记载 |
| 无本轮引入的回归 | PASS（已测范围） | 扩展套件 2692 passed；未改 `src/`、既有 RoboCasa/LIBERO/阶梯实现；既有测试问题单列并独立复现，未声称全仓 Verify green |

**验证记录**（最终代码）：

```bash
.venv/bin/python -m pytest tests/exp/step_diag tests/review_tests/test_step_diag_g2_contracts.py -q --tb=short --disable-warnings
# 101 passed, 1 skipped, 3 warnings in 10.91s

.venv/bin/python -m pytest tests/cache tests/conductor tests/robocasa365 tests/libero_groot tests/exp/step_diag tests/review_tests/test_step_diag_g2_contracts.py --ignore=tests/robocasa365/test_bench_groot_stages.py -k 'not test_hit_meta_rows_identical_across_runners and not test_frozen_commands_pass_the_new_guards' -q --tb=short --disable-warnings
# 2692 passed, 34 skipped, 4 deselected, 21 warnings in 94.05s

.venv/bin/ruff check exp/step_diag tests/exp/step_diag --select F
.venv/bin/python -m compileall -q exp/step_diag tests/exp/step_diag
# 全部通过；ops/*.sh 全部 bash -n 通过；git diff --check 通过。
```

扩展回归在允许 loopback socket 的环境执行；原沙箱 PermissionError 单独重跑 socket/conductor 相关文件，86 项通过。原始输出留在本会话 `/tmp/step_diag_g2_final_focused.txt`、`/tmp/step_diag_g2_final_regression.txt`、`/tmp/step_diag_g2_socket_regression.txt`。

**已知边界 / 未作为本轮通过证据的项目**：

- 未修改的 `tests/robocasa365/test_bench_groot_stages.py` 引用现模块缺失的 `SCHEDULE_ID`，在收集期失败，扩展运行排除整文件。
- 未修改的 `test_ws2_evidence_runner.py::test_hit_meta_rows_identical_across_runners` 两个参数化例的 exact-key 断言与现有附加字段不符；仅运行旧测试仍复现两失败。`test_groot_concurrent_serving.py::test_frozen_commands_pass_the_new_guards` 两个参数化例受既有 `tests/libero_groot/test_rit_shadow_factory.py` 注入无 `__spec__` 的 gr00t stub 影响，独立运行通过。最终扩展命令明确 deselect 这四例；原始复现见 `/tmp/step_diag_g2_baseline_regression.txt` 与 `/tmp/step_diag_g2_regression2.txt`。
- 本轮使用已安装 `.venv` 的 pytest，未完成 WA §2.7 全仓 `uv run pytest`。真实 checkpoint/GPU parity、六环境实际资源检查、两机 smoke、正式 9760 集和研究结论均未执行；这些仍是后续 Verify/放量门，G2 通过不等于实验已验证或允许越过 smoke。

**Final verdict**：APPROVED — code approved。以上本轮阻塞已修复并复测，无遗留 G2 阻塞。原始暂存快照保持不变；审查者修改（包括新增模块/测试、文档及本记录）全部留在暂存区外。
