# R17：执行 horizon 能否增加而保持成功率？

日期：2026-10-10。状态：已按owner纠正核查历史Cosmos NFE；本次仅修改本文，**本助手未启动GPU任务，未改动已有结果、队列或运行进程**。本文的预注册适用于尚未运行的新 horizon 臂；历史结果已经可见，不能称为盲于全部历史数据的注册。

**给 owner 的回答：不能笼统说三个模型都可以。** 相对本地 R17 默认，DP-T 可测 5→16，Fast-WAM 可测 5→32；是否保持 SR 尚待闭环验证。相对发布配置，DP-T 的执行参数已是 16，Fast-WAM 有作者仓库 10 与 LeRobot 发布配置 32 两种口径；Cosmos 已执行完整 16 步，不能靠调大执行参数再减少推理。依据与限制见第 2 节。


**2026-10-10 owner纠正版：执行范围、去噪臂和两卡排程以第(6)/(7)节为准；第(3)节已同步24个比较。第(4)节仅保留首版四套件预算/命令的追溯记录，不是当前执行队列。Cosmos历史已做；本次仅改文档，不重复提交协调者已排任务。**

## (1) π0.5 / GR00T 实际做过什么

### 1.1 三个量必须分开

仓库远程推理文档将返回的 `action_horizon × action_dim` 预测块与每隔N步调用一次、期间开环执行的客户端安排分开；它本身没有给出SR不变的保证。[docs/upstream/remote_inference.md:60-64](/home/weiland/projects/openpi/docs/upstream/remote_inference.md:60)

| 量 | 历史代码 / 协议中的意思 | 证据 |
|---|---|---|
| 预测长度 H_pred | π0.5 LIBERO 输出 10 步；GR00T 输出 16 步。不是每次都执行全部输出。 | [logs/step_vs_warmstart_diagnostics_plan.log.md:20](/home/weiland/projects/openpi/logs/step_vs_warmstart_diagnostics_plan.log.md:20) |
| 执行长度 H_exec（本文的 C/H） | 早期纯推理客户端 `replan_steps=5`；R4 直接比较客户端 L=5/10。R14–R16 的 C10 是执行动作 0–9 后才再运行模型。 | [r04/k4_eval/INTEGRATION.md:3](/home/weiland/projects/openpi/exp/offline_search/rounds/r04/k4_eval/INTEGRATION.md:3)；[r16/ADAPTER.md:7](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/ADAPTER.md:7) |
| 传输 cadence R | C10 的后期实现仍可 `replan_steps=5`：先执行 0–4，第二次请求只返回同一预测块的 5–9，不做新模型推理。故“请求两次”不等于“推理两次”。 | [r13/astra/SPEC.md:115-116](/home/weiland/projects/openpi/exp/offline_search/rounds/r13/astra/SPEC.md:115)；[r16/RUNBOOK_CONDUCTOR.md:7](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/RUNBOOK_CONDUCTOR.md:7) |
| 去噪 K / NFE | R16 改 K=1/2，π0.5 满步 K=10，GR00T 满步 K=8；H_exec 固定 10。`DP100/Fast10/Cosmos5` 中的数字同样是去噪步数。 | [r16/COORDINATOR_DECISIONS.md:5-12](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/COORDINATOR_DECISIONS.md:5)；[logs/session_handoff.md:95](/home/weiland/projects/openpi/logs/session_handoff.md:95) |

历史 owner IR 以“五个名义控制步”为单位计价：π0.5 `(.152 V+.848 M)/N`，GR00T `(.148 V+.852 M)/N`，V/M 是视觉/完整策略调用数；C10 约 .5，C5 为 1。R4 直接 L 步客户端形式另乘 `5/L`；R14 的 N 已含五步块尾请求，不再重复乘。**这些是归一化计算成本，不是 GPU 墙钟时间**，末尾不足一块会造成 .504/.512 等边界偏差。[r04/ANALYSIS.md:5-10](/home/weiland/projects/openpi/exp/offline_search/rounds/r04/ANALYSIS.md:5)；[r14/descent_astra/TEST_A.md:9](/home/weiland/projects/openpi/exp/offline_search/rounds/r14/descent_astra/TEST_A.md:9)

### 1.2 直接执行长度实验：来源实际追溯到 R4 / R5

R10–R16 的报告引用了更早的纯推理点，因此补读 R4/R5 的分析，不能把 R14 的“逐任务旋钮下降”误认成 horizon sweep。R4 每臂 500 对（10 任务×官方 init 0–49），同 `(task, init)` 配对；精确双侧 McNemar 与 10,000 次配对 bootstrap。R5 同样通常每臂 500 局，重复运行按同一起点聚类，不能把三次重复当作 1,500 个独立初态。[r04/ANALYSIS.md:3-9](/home/weiland/projects/openpi/exp/offline_search/rounds/r04/ANALYSIS.md:3)；[r05/ANALYSIS.md:9-25](/home/weiland/projects/openpi/exp/offline_search/rounds/r05/ANALYSIS.md:9)

| 模型 / 测试套件 | H_exec 与局数 | SR / owner IR | 结论与限制 |
|---|---|---|---|
| π0.5 / libero_10，满步 K10 | C5 三次×500；C10 两次×500 | C5：84.4、84.8、86.0% / 1.000；C10：90.4、90.0% / .500 | C10 明显更好且成本减半；C10(seed3001) 对 C5(seed1001)：+5.6 pp，95% CI [2.2,9.0]，p=.0018。来源表的 discordance 列标题顺序有歧义，本文采用净差方向：52 个新增成功、24 个新增失败。[r04/ANALYSIS.md:146-157](/home/weiland/projects/openpi/exp/offline_search/rounds/r04/ANALYSIS.md:146)；[r04/ANALYSIS.md:230-231](/home/weiland/projects/openpi/exp/offline_search/rounds/r04/ANALYSIS.md:230) |
| π0.5 / libero_spatial，满步 K10 | C5 三次×500；C10 一次×500 | C5：98.6、99.2、99.4% / 1.000；C10：98.6% / .500 | 差 0/−0.6/−0.8 pp；没有检出差异，**不是已证明非劣**。[r04/ANALYSIS.md:157](/home/weiland/projects/openpi/exp/offline_search/rounds/r04/ANALYSIS.md:157)；[r04/ANALYSIS.md:269](/home/weiland/projects/openpi/exp/offline_search/rounds/r04/ANALYSIS.md:269) |
| GR00T / libero_10 | 历史 C5 500；R5 C10 500 | 87.0% / 1 → 86.6% / .5 | −0.4 pp；原报告明确说历史 C5 跨 harness，不能作为独立核验的配对因果结论。[r05/ANALYSIS.md:66-84](/home/weiland/projects/openpi/exp/offline_search/rounds/r05/ANALYSIS.md:66) |
| GR00T / libero_spatial | 历史 C5 500；R5 C10 500 | 94.0% / 1 → 93.8% / .5 | −0.2 pp，同上；不是“所有更长 horizon 都无损”。[r05/ANALYSIS.md:68-84](/home/weiland/projects/openpi/exp/offline_search/rounds/r05/ANALYSIS.md:68) |

π0.5 的 C5 额外运行服务端种子为 1001/2001，C10 为 3001 加一次 unseeded；环境初态配对不等于随机噪声相同。GR00T R5 的 C10 服务端种子 5101，客户端直接 replan10。[r04/ANALYSIS_BRIEF.md:21-22](/home/weiland/projects/openpi/exp/offline_search/rounds/r04/ANALYSIS_BRIEF.md:21)；[r05/q2_groot/HANDBACK.md:107](/home/weiland/projects/openpi/exp/offline_search/rounds/r05/q2_groot/HANDBACK.md:107)

还测过 **GR00T 缓存动作**执行 C5/C10/C15（每格每臂 500，库大小 50/500）。它们不是新模型推理出的块，不能用来声称 GR00T 纯推理 C15 已验证：[r05/ANALYSIS.md:68-84](/home/weiland/projects/openpi/exp/offline_search/rounds/r05/ANALYSIS.md:68)

| GR00T 纯缓存 / 库大小 | C5：SR / IR | C10：SR / IR | C15：SR / IR |
|---|---:|---:|---:|
| libero_10 / 50 | 55.2 / .14800 | 60.8 / .07428 | 57.0 / .05004 |
| libero_10 / 500 | 70.6 / .14800 | 83.0 / .07450 | 80.8 / .05018 |
| spatial / 50 | 88.8 / .14800 | 86.8 / .07529 | 86.4 / .05127 |
| spatial / 500 | 96.6 / .14800 | 96.4 / .07551 | 94.6 / .05175 |

上表全部来自 [r05/ANALYSIS.md:68-80](/home/weiland/projects/openpi/exp/offline_search/rounds/r05/ANALYSIS.md:68)。C10 对 C15 的 SR 差均未在单次运行中分开；越长越好并不成立，空间任务连 C5→C10 都没有正收益。[r05/ANALYSIS.md:82-84](/home/weiland/projects/openpi/exp/offline_search/rounds/r05/ANALYSIS.md:82)

### 1.3 R10–R16 留下的后续证据

| 轮次 | 实际改变什么 / 与 horizon 的关系 | 数据及可用结论 |
|---|---|---|
| R10 | 库大小、守卫、修正器 | 官方 A 每任务 50、每臂 500；测试不拟合。不是独立 horizon sweep。[r10/REPORT.md:5-21](/home/weiland/projects/openpi/exp/offline_search/rounds/r10/REPORT.md:5) |
| R11 | 随机/定期/状态触发的额外调用预算 | 94 个 A 臂各 500；纯推理参考成本 .5；调用预算不是预测块长度。[r11/REPORT.md:3-9](/home/weiland/projects/openpi/exp/offline_search/rounds/r11/REPORT.md:3)；[r11/REPORT.md:43-45](/home/weiland/projects/openpi/exp/offline_search/rounds/r11/REPORT.md:43) |
| R12 | 看一半 / 选相机 | 同样 replan5+一个 policy-tail block；A 各 500。[r12/TEST_A_HANDBACK.md:20](/home/weiland/projects/openpi/exp/offline_search/rounds/r12/TEST_A_HANDBACK.md:20)；[r12/REPORT.md:48](/home/weiland/projects/openpi/exp/offline_search/rounds/r12/REPORT.md:48) |
| R13 | 距离 vs 定期预算；C5 是上下文对照 | 已完成的 GR00T libero_10 C5：500 局，85.4% / 1.000；列为全部七个补充臂之一，未用于选方法。不是 R14 C10 的新预注册配对 horizon 实验。[r13/astra/REPORT.md:69-79](/home/weiland/projects/openpi/exp/offline_search/rounds/r13/astra/REPORT.md:69)；[r13/COORDINATOR_DECISIONS.md:79-87](/home/weiland/projects/openpi/exp/offline_search/rounds/r13/COORDINATOR_DECISIONS.md:79) |
| R14 | B 池逐任务预算下降 | 每任务30局，6档（C10,.45,.40,.32,.25,off）；首次超出5pp点估计容差就停。冻结后 A 每格3臂×500。不是 6 个 horizon。[r14/REPORT.md:9-14](/home/weiland/projects/openpi/exp/offline_search/rounds/r14/REPORT.md:9) |
| R15 | 将同一下降法扩展到库100/200/500 | 复用 R14 A C10，前提是证明与库无关。[r15/COORDINATOR_DECISIONS.md:5-13](/home/weiland/projects/openpi/exp/offline_search/rounds/r15/COORDINATOR_DECISIONS.md:5)；[r15/descent/ADAPTER.md:3-9](/home/weiland/projects/openpi/exp/offline_search/rounds/r15/descent/ADAPTER.md:3) |
| R16 | 去噪 K1/K2；H_exec=10 不变 | 两模型×两套件×两 K，共8个纯推理臂，各500局，配对 R14 C10。不能把这8臂当成 horizon 梯度。[r16/COORDINATOR_DECISIONS.md:5-19](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/COORDINATOR_DECISIONS.md:5) |

R14–R16 复用的满步 C10 参考（每格500局；不是将 R5 数字改写为同一实验）：

| 模型 | libero_10 SR / IR | libero_spatial SR / IR | 来源 |
|---|---:|---:|---|
| π0.5 | 90.8% / .504 | 98.6% / .512 | [r14/REPORT.md:20-25](/home/weiland/projects/openpi/exp/offline_search/rounds/r14/REPORT.md:20) |
| GR00T | 88.0% / .504 | 95.4% / .512 | [r14/REPORT.md:20-25](/home/weiland/projects/openpi/exp/offline_search/rounds/r14/REPORT.md:20) |

R16 K1/K2 纯推理分别为：π0.5 长任务 91.8/.305、92.2/.327；空间 98.4/.310、97.8/.333；GR00T 长任务两者均88.2%，成本 .204/.247；空间均95.0%，成本 .208/.251（格式 SR% / IR）。这些差值的区间跨零只支持“本次精度下未分开”，不补足 horizon 非劣证书。[r16/REPORT.md:20-33](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/REPORT.md:20)

**旧判定规则不是一个现成的“horizon 无损”规则。** R4/R5 用配对区间、McNemar 和重复运行判断改善；R5 明文说保 SR 省成本需要事先指定非劣界，但当时没有指定。R14 的5pp规则是 B 集预算下降的点估计规则；30局允许净少1成功，并非总体非劣证明，测试40个模型×任务仍有11个下降超过5pp。[r05/ANALYSIS.md:23-47](/home/weiland/projects/openpi/exp/offline_search/rounds/r05/ANALYSIS.md:23)；[r14/descent_astra/PROTOCOL.md:13-27](/home/weiland/projects/openpi/exp/offline_search/rounds/r14/descent_astra/PROTOCOL.md:13)；[r14/REPORT.md:32-37](/home/weiland/projects/openpi/exp/offline_search/rounds/r14/REPORT.md:32)

**不纳入 MT50。** 已找到的执行长度实验是上述两个 LIBERO 套件；R16 的正式格子亦仅两套件。R17 虽有 π0.5/GR00T MT50 减去噪步实验，但执行长度一直5，不能据此扩展本研究到 MT50。[r04/ANALYSIS.md:3-6](/home/weiland/projects/openpi/exp/offline_search/rounds/r04/ANALYSIS.md:3)；[r16/COORDINATOR_DECISIONS.md:12](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/COORDINATOR_DECISIONS.md:12)；[RESULTS.md:3-5](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:3)；[RESULTS.md:23-31](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:23)

## (2) 三个新模型的事实、默认值与限制

官方资料本次重新下载，并锁定提交/权重版本；URL、SHA256 在 [r17/horizon/sources/SOURCES.json:1-40](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/SOURCES.json:1)。DP 元数据经仓库已有白名单 inert reader 转为 JSON，没有执行公开 pickle 的任意代码。[dp/checkpoint.py:14-29](/home/weiland/projects/openpi/exp/mw_finetune/dp/checkpoint.py:14)

| 模型 | 原生预测块 | 官方 / 发布的执行 horizon | R17 当前默认与本地代码 | 能否只调执行量？ |
|---|---|---|---|---|
| DP-T | **16×7**（LIBERO）；1个当前观测 | 发布权重元数据 `n_action_steps=16`、`horizon=16`、K100。它是发布配置证据，**不是已找到官方 rollout 的证明**；作者 DP wrapper 未公开，本地是重建实现。 | DP100 / R=5 / 默认C5。`models/dp/serve.py` 的 `plain_config` 默认100、5；layout固定16、5。 | 相对 R17 可测C10/15/16；相对发布默认16已无剩余。>16涉及预测几何/权重，loader拒绝。[r17/horizon/sources/dp_metadata.json:8-35](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/dp_metadata.json:8)；[dp/RECIPE.md:10-16](/home/weiland/projects/openpi/exp/mw_finetune/dp/RECIPE.md:10)；[dp/RECIPE.md:53-58](/home/weiland/projects/openpi/exp/mw_finetune/dp/RECIPE.md:53)；[models/dp/serve.py:17](/home/weiland/projects/openpi/exp/r17_integration/models/dp/serve.py:17)；[models/dp/manifest.py:51-62](/home/weiland/projects/openpi/exp/r17_integration/models/dp/manifest.py:51)；[models/dp/loader.py:26](/home/weiland/projects/openpi/exp/r17_integration/models/dp/loader.py:26) |
| Fast-WAM | **32×7**（LIBERO） | 作者 FastWAM 仓库 `sim_libero.yaml`：**replan10**；LeRobot 发布 checkpoint 与类默认：**n_action_steps32**。两者必须分别报告。 | Fast10 / R=5 / 默认C5；`execution_protocol=system` 为5；`released_libero` profile 为32且同时改 seed/cap。 | 相对本地5可测10/16/20/32；相对作者10可测16/20/32；相对LeRobot32已满。不能切 `released_libero` profile 来冒充“只改 horizon”。[r17/horizon/sources/fastwam_release_config.json:42-45](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/fastwam_release_config.json:42)；[r17/horizon/sources/fastwam_sim_libero.yaml:16-32](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/fastwam_sim_libero.yaml:16)；[r17/horizon/sources/fastwam_lerobot_config.py:193-217](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/fastwam_lerobot_config.py:193)；[models/fastwam/protocols.py:29-44](/home/weiland/projects/openpi/exp/r17_integration/models/fastwam/protocols.py:29) |
| Cosmos Policy | **16×7**，状态9维 | 官方 LIBERO `chunk_size=16`、`num_open_loop_steps=16`，K5；执行完整预测块。 | Cosmos5 / R=C=16；固定seed195；正式launcher使用专属 benchmark profile；checkpoint几何校验7/16/9。 | **没有 C>16 的纯执行 knob 实验**。官方还断言测试chunk_size必须等于训练chunk_size。重复最后动作、重复块、插值或再预测均改变研究问题。[r17/horizon/sources/cosmos_release_config.json:17-27](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/cosmos_release_config.json:17)；[r17/horizon/sources/cosmos_run_libero_eval.py:215-216](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/cosmos_run_libero_eval.py:215)；[r17/horizon/sources/cosmos_run_libero_eval.py:838-839](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/cosmos_run_libero_eval.py:838)；[models/cosmos/protocols.py:8-13](/home/weiland/projects/openpi/exp/r17_integration/models/cosmos/protocols.py:8)；[models/cosmos/loader.py:47-51](/home/weiland/projects/openpi/exp/r17_integration/models/cosmos/loader.py:47)；[models/cosmos/loader.py:88-96](/home/weiland/projects/openpi/exp/r17_integration/models/cosmos/loader.py:88) |

官方直链：[DP 发布 metadata](https://huggingface.co/brandonyang/dp-libero-checkpoints/blob/3c48ffaab98faacb119beaf2ee12f943503e6492/dp_libero_lang_v1/30000/metadata.pt)、[FastWAM 作者执行配置](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/sim_libero.yaml#L25)、[FastWAM LeRobot 发布配置](https://huggingface.co/lerobot/fastwam_libero_uncond_2cam224/blob/53983e1249b4eb4d89ab42b40b8678a882d331ad/config.json)、[NVIDIA 官方 LIBERO runner](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/libero/run_libero_eval.py#L215)。

权重身份也要分开：DP 发布配置来自作者30000步权重；R17当前正式入口使用自己的100000步fine-tune。它们共享H16的架构约束，但本研究的SR结论仅适用于后者，不能替作者权重背书。[dp/RECIPE.md:20](/home/weiland/projects/openpi/exp/mw_finetune/dp/RECIPE.md:20)；[RESULTS.md:39](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:39)；[ops/formal_eval.py:31](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:31)

`exp/mw_finetune` 的独立实现也一致：DP policy默认100，metadata horizon16；Fast-WAM policy默认10，调用 `infer_action(action_horizon=32)`；MT50共有评测用5动作重规划。这不是 Cosmos LIBERO 的16步入口。R17正式launcher从 `model.default_sampler` 和 `model.layout.replan_steps` 建纯推理配置，Fast-WAM的manifest是H32/R5，Cosmos是H16/R16。[dp/policy.py:21-46](/home/weiland/projects/openpi/exp/mw_finetune/dp/policy.py:21)；[dp/policy.py:99-100](/home/weiland/projects/openpi/exp/mw_finetune/dp/policy.py:99)；[fastwam/policy.py:13-39](/home/weiland/projects/openpi/exp/mw_finetune/fastwam/policy.py:13)；[RESULTS.md:3-5](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:3)；[ops/formal_eval.py:151-166](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:151)；[models/fastwam/manifest.py:47-61](/home/weiland/projects/openpi/exp/r17_integration/models/fastwam/manifest.py:47)；[models/cosmos/manifest.py:27-34](/home/weiland/projects/openpi/exp/r17_integration/models/cosmos/manifest.py:27)

**Fast-WAM 官方口径差异不只 horizon。** 作者 YAML 是30步settle；本地system是10步settle、seed7的语义配对、Spatial cap220；本地released profile则固定每次seed42、Spatial cap280。本文只比较同一R17 profile内的执行长度，不报告为完整官方协议复现。[r17/horizon/sources/fastwam_sim_libero.yaml:22-25](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/fastwam_sim_libero.yaml:22)；[models/fastwam/protocols.py:37-44](/home/weiland/projects/openpi/exp/r17_integration/models/fastwam/protocols.py:37)；[models/fastwam/protocols.py:92-99](/home/weiland/projects/openpi/exp/r17_integration/models/fastwam/protocols.py:92)

### 默认关闭的 launcher 参数已添加

| 改动 | 语义与验证点 | 代码 |
|---|---|---|
| `--commit-horizon H`，默认 `None` | 仅纯推理LIBERO的DP/Fast/Cosmos；分别允许5..16、5..32、仅16。未传参仍按原始layout重规划。 | [ops/formal_eval.py:111-120](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:111)；[ops/formal_eval.py:1011-1048](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:1011) |
| 配置→计划→模型块尾 | 更改 `adapter_binding.commit_horizon`，计划检查同一H并写入每局；保持原 benchmark/model/sampler/preprocessing。非5倍数用已有 `allow_partial_commit` 与执行确认协议。 | [ops/formal_driver.py:35-73](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_driver.py:35)；[common/provider.py:72-91](/home/weiland/projects/openpi/exp/r17_integration/common/provider.py:72)；[benchmarks/libero_bridge.py:208-217](/home/weiland/projects/openpi/exp/r17_integration/benchmarks/libero_bridge.py:208) |
| 原框架执行路径 | DP/Fast继续R5，H16=5+5+5+1，H32=5×6+2。尾请求不做模型前向；最后一小段按实际长度执行。不是把预测长度改大，也不是减少WebSocket请求。 | [common/serving.py:298-328](/home/weiland/projects/openpi/exp/r17_integration/common/serving.py:298)；[common/serving.py:362-365](/home/weiland/projects/openpi/exp/r17_integration/common/serving.py:362)；[benchmarks/libero_bridge.py:293-304](/home/weiland/projects/openpi/exp/r17_integration/benchmarks/libero_bridge.py:293) |
| 统计 | 显式H臂验证每局顺序、cursor、policy/尾请求、K/NFE；分别输出 `policy_calls` 与 `policy_tail_requests`。默认臂原检查不变。 | [ops/formal_eval.py:169-185](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:169)；[ops/formal_eval.py:561-595](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:561) |
| CPU验证 | 真实公共ConnectionPolicy＋假模型验证逐动作块尾、16/32余数、完成后才二次前向；计划2000局、默认配置相同、非法长度、错配/缺行/重复记录拒绝、aggregate。没有加载真实权重或调用CUDA。 | [ops/tests/test_formal_horizon.py:1-154](/home/weiland/projects/openpi/exp/r17_integration/ops/tests/test_formal_horizon.py:1)；[r17/horizon/CPU_TESTS.log:1](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/CPU_TESTS.log:1) |

说明：首版CPU回归 **43 passed、25 subtests passed、1 deselected**，包含新增19项horizon测试。[r17/horizon/CPU_TESTS.log:8](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/CPU_TESTS.log:8) CPU整组回归曾暴露既有远程进程模拟测试与当前运维接口不匹配（原mock缺 `replay_safe`；临时适配签名后仍缺批量控制响应，随后还原该测试）。其失败记录保留在 `CPU_TESTS_full_attempt.log`；最终针对本改动的回归排除这个远程模拟测试，不修改运行时运维逻辑。该限制不等于已做GPU/网络资格验证。[ops/tests/test_formal_eval.py:339-368](/home/weiland/projects/openpi/exp/r17_integration/ops/tests/test_formal_eval.py:339)；[r17/horizon/CPU_TESTS_full_attempt.log:1-72](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/CPU_TESTS_full_attempt.log:1)

## (3) 预注册研究设计与判定规则

以下均是**本次设计决定**，不是已测结果。当前范围已按owner纠正同步第(6)/(7)节：保留DP/Fast新臂，并按最新owner决定仅加Cosmos K2 incl. clean step/C16一臂；全部新主比较共24个。

| 项目 | 冻结方案 |
|---|---|
| 主问题 | 同一权重、K、预处理、机器、运行框架下，多执行同一预测块能否减少模型调用并保持SR？预测H_pred不变。 |
| 测试集 | 官方A：每任务50个官方初态idx0..49，每套件500。主分析只跑 `libero_spatial`、`libero_10`；使用默认关闭的 `--suites` 过滤到原2000局计划的1000个身份。不用旧400局或B库成绩作A基线。 |
| 模型 / 执行臂 | DP100：**C5基线、C10、C15、C16**；Fast10：**C5基线、C10、C16、C20、C32**；Cosmos5：**C16已是全块，不新增horizon臂；仅新增K2 incl. clean step减步臂**，复用H100 K5/C16默认。 |
| 对照 | 主比较：DP/Fast各对自己的R17 C5默认；Cosmos新K2/C16对指定H100 K5/C16；旧Cosmos历史不纳入本轮主比较。另报DP相对发布C16、Fast相对作者C10及LeRobot C32的配对差；这些次比较只作描述，不升级为额外的确认性结论。 |
| 配对单位 | `(model, suite, canonical task_name, official init index, init bytes hash, environment seed/profile)`；不同H的task_uid/YAML身份本来不同，禁止直接按uid字符串配对。每格500完整配对，10任务等权。 |
| 冻结量 | DP K100，Fast K10，Cosmos K5；无检索库/守卫/修正器/降NFE/相机消融；相同resize、settle、step cap、checkpoint/norm/sampler；DP同4090×12副本+24 worker；Fast同H100×2+24 worker。 |
| 种子 | 延续各自R17规则：DP/Fast seed7、按任务/初态/decision_seq构造；Cosmos固定195。配对只保证环境身份与种子规则，不声称不同H产生同一条噪声流。 |
| 重复与顺序 | 每H先完整一次，不因中途SR停跑；已有合格完整官方A默认臂可复用，否则同版本重跑默认。新臂固定顺序：DP10→15→16；Fast10→16→20→32。记录顺序/主机效应；不把单次运行的不确定性解释为未来seed全覆盖。 |
| 错误 | 正常任务失败计0；基础设施错误走相同重试规则。缺/多/重复/未验收局、未解决error不得输出最终SR结论；只读已有结果，新目录生成分析。 |
| 样本总量 | horizon候选7臂×1000＝7,000局；step候选5臂×1000＝5,000局（含唯一Cosmos K2）；默认另计，见6.2。不另跑MT50。 |

官方50构造与当前默认计划见 [ops/formal_eval.py:102-108](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:102)、[conductor/manifest.py:31-78](/home/weiland/projects/openpi/exp/r17_integration/conductor/manifest.py:31)；错误400局已明确作废见 [logs/session_handoff.md:59-61](/home/weiland/projects/openpi/logs/session_handoff.md:59)、[RESULTS.md:33-42](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:33)。初态bytes核验及seed构造见 [benchmarks/libero_bridge.py:193-201](/home/weiland/projects/openpi/exp/r17_integration/benchmarks/libero_bridge.py:193)、[benchmarks/libero_bridge.py:275-280](/home/weiland/projects/openpi/exp/r17_integration/benchmarks/libero_bridge.py:275)。

**决策规则（预先冻结）：**

1. 采用“保持到2个百分点以内”的非劣界 **δ=2pp**；另报告δ=0和5pp敏感性，不见结果后改主界。旧horizon实验未预注册非劣界，所以这不是追认其“无损”结论。非劣2pp与严格零损失是不同表述。
2. 每个主格计算 `g=候选成功/默认失败`、`l=候选失败/默认成功`、`Δ=(g−l)/500`；报告两臂SR、g/l、McNemar双侧p（仅描述）及按任务内重采样的10,000次配对bootstrap，seed=20261010。24个主对比＝[DP(3 horizon+2 step)+Fast(4 horizon+2 step)+Cosmos(1 step)]×2套件，使用同一重采样索引携带共享默认臂。
3. 确认性非劣同时控制24个对比：每个对比单侧水平 `.05/24`。注册保守配对二项界 `L_CP=CP_low(g,500,a)−CP_high(l,500,a)`，`a=.05/(2×24)`；再取任务内配对bootstrap的 `.05/24` 下分位，**两种下界均须大于−.02**。CP的解释采用该套件等权任务混合下的二项工作模型；分层bootstrap检视任务异质性。逐任务给出50局SR和损失，**不声称每任务非劣**。这沿用历史 gains/losses 保守界的形式，但本次suite级、多重性和容差为新注册；历史形式见 [r14/descent_astra/PROTOCOL.md:40-49](/home/weiland/projects/openpi/exp/offline_search/rounds/r14/descent_astra/PROTOCOL.md:40)。
4. SR判定与成本判定同时满足才标“可采用”：在**两个主套件分别**通过上述SR门，且各套件实测模型调用/执行动作和总服务时间/执行动作均低于默认。报告按局成本，防止更早失败让总耗时假降；墙钟节省另列。若服务计时缺失或含排队无法分开，成本结论暂缺，不能用理论5/H替代。
5. 从C5沿预定升序检查，第一次不通过即保留前一档；后续已预注册臂仍完整测完、全表报告，不跳过失败挑更长的偶然高点。两套件共用一个H；不在A上拟合逐任务H。结果下界跨−2pp＝**证据不足**；点估计下降但通过＝“2pp非劣”，不能写“SR完全无损”；只有点差≥0才可另写“本次未观察到下降”。
6. 若预算/端口/机器变化，记录基础设施暂停，不以SR自适应加局或换基线。需要更大样本或B集逐任务选择时另注册，并用独立确认集；不把本次挑中的A最优点称为额外独立复现。

样本局限：500对中一个结果改变0.2pp；零discordance时上述CP下界约−1.36pp，具备通过2pp门的可能，但出现较多discordance即容易不够精确。通过只针对这两个固定套件与当前运行条件，不推出“所有模型/任务/horizon无损”。

成本登记：`T=Σ实际执行控制步`（不含settle）、`M=source=policy`次数、`Q_tail=source=policy_tail`次数；主报 `M/T`、`5M/T`（本模型满步推理当单位1）、相对默认比例、`Σ服务ms/T`、服务ms/episode、墙钟GPU占用小时。K保持固定，故不借用π/GR00T的.152/.848系数给新模型计价。尾请求日志虽可能标FULL_HIT，仍是同一策略预测的复用，不是库检索。服务计时是host wall，不能称CUDA event kernel time。[common/serving.py:366-376](/home/weiland/projects/openpi/exp/r17_integration/common/serving.py:366)；[common/serving.py:419-430](/home/weiland/projects/openpi/exp/r17_integration/common/serving.py:419)；[ops/formal_server.py:132-141](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_server.py:132)

## (4) 首版协调者命令与 GPU 时间（已归档；当前只用第7节）

### 4.1 每个臂的规划预算

**这些是预算推算，不是已运行新臂的计时。** 占用一张物理GPU的墙钟小时，不乘副本数；预计误差至少±50%，未保证线性加速。完工官方A默认臂的实际耗时可用于在不查看新臂SR的前提下更新资源预算，不改统计规则。

| 输入 | 只读测量 / 推算依据 |
|---|---|
| DP4090，12副本24worker | 旧400局driver 2375.01s、14687次调用，即6.184 calls/s；该旧receipt PARTIAL有重试，但仅用于粗估计时，不用于SR配对。[r17/horizon/TIMING_INPUTS.json:3-28](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/TIMING_INPUTS.json:3)；[RESULTS.md:39-42](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:39) |
| CosmosH100，2副本24worker | 旧400局driver1362.33s、3991次调用，即2.930 calls/s。[r17/horizon/TIMING_INPUTS.json:31-56](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/TIMING_INPUTS.json:31) |
| FastH100，2副本24worker | B池修复spatial271局2393.90s、l10 446局7951.70s；均约3.09 calls/s，包含debug/图像传输，作保守参照。每500局外推73.6/148.6min。[r17/horizon/TIMING_INPUTS.json:59-112](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/TIMING_INPUTS.json:59) |
| 主套件轨迹量 | DP B库spatial/l10分别12028/31546行（R5），Cosmos3676/8408行（R16）；用于估算典型调用量，不是新H的轨迹长度保证。[RESULTS.md:54-65](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:54) |

冻结估算公式：`t_suite(H)=5min+t_suite(default)×[.25+.75×H_default/H]`；四套件整臂 `t_all(H)=10min+t_all(default)×[...]`。25%表示仿真/传输等不随模型调用数缩放的**规划假设**。DP/Cosmos整臂基准=旧400局driver×5；Fast整臂基准=2×(73.6+148.6)min，即暂把object/goal各估成两个主套件耗时的均值。具体计算保存在 [r17/horizon/TIME_ESTIMATES.json:1-96](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/TIME_ESTIMATES.json:1)。

| 模型 / GPU | C（动作/模型推理） | 理想模型调用密度 / R17默认 | spatial 500局，min | libero_10 500局，min | 下列命令实际四套件2000局，GPU·h |
|---|---:|---:|---:|---:|---:|
| DP / 4090 | 5 默认 | 1.000 | 37.4 | 90.0 | 3.47 |
| DP / 4090 | 10 | .500 | 25.3 | 58.1 | 2.23 |
| DP / 4090 | 15 | .333 | 21.2 | 47.5 | 1.82 |
| DP / 4090 | 16 发布执行长度 | .313 | 20.7 | 46.2 | 1.76 |
| Fast / H100 | 5 默认 | 1.000 | 78.6 | 153.6 | 7.57 |
| Fast / H100 | 10 作者仓库执行长度 | .500 | 51.0 | 97.9 | 4.80 |
| Fast / H100 | 16 | .313 | 40.7 | 77.0 | 3.75 |
| Fast / H100 | 20 | .250 | 37.2 | 70.0 | 3.41 |
| Fast / H100 | 32 LeRobot发布执行长度 | .156 | 32.0 | 59.6 | 2.89 |
| Cosmos / H100 | 16 默认＝完整块 | 1.000 | 25.9 | 52.8 | 2.06（若基线可复用则新增0） |

两主套件分钟列含各自5分钟装载预算，不能简单与四套件列相加对账。新增默认均复用时：DP约**5.81 GPU·h**，Fast约**14.85 GPU·h**，合计**20.66 GPU·h**；若三个默认全重跑，再加13.10 GPU·h。原生块尾执行会保留R5网络/渲染开销，因此“调用减到5/32”不意味着墙钟也减到5/32。

### 4.2 CPU复查命令（前版已执行，本次文档更正未重跑）

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/ops/tests/test_formal_horizon.py \
  exp/r17_integration/ops/tests/test_formal_eval.py \
  exp/r17_integration/tests/benchmarks/test_libero.py \
  -k 'not remote_replica_death_fails_before_lease_renewal_even_on_exit_zero'
```

### 4.3 GPU命令：只交给协调者，今晚基线结束且设备空闲后执行

本次**没有运行**以下命令。先按机器核对没有计算进程，不停任何现有任务；Cosmos/Fast在H100顺序执行，DP在4090。正式launcher及baseline的副本/端口配置参照现行协调者脚本：[/home/weiland/projects/openpi_ext/scratch/r17_ops/chain_official_4090_dp.sh:3-4](/home/weiland/projects/openpi_ext/scratch/r17_ops/chain_official_4090_dp.sh:3)；[/home/weiland/projects/openpi_ext/scratch/r17_ops/chain_official_h100.sh:3-7](/home/weiland/projects/openpi_ext/scratch/r17_ops/chain_official_h100.sh:3)。更正：H100远程锁guard仅用于library分支；普通研究臂由协调者排他安排，勿在本地替H100占4090锁。[ops/formal_eval.py:729-780](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:729)

```bash
# 只读检查；任何计算进程仍在运行时，协调者延期，不执行后面的研究队列。
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
tether exec --safe --timeout 30s h100 -- nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
tether exec --safe --timeout 30s h100 -- nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

每组由协调者在新shell/tmux执行；脚本错误即停，输出目录由launcher生成唯一名字。无需手工同步模型代码：沿用formal launcher自身的准备/密封/部署链。默认臂复用前须核验官方50完整验收、模型/权重/norm/代码及种子profile一致、同机器与副本数；只比较共有官方身份，绝不能拿旧formal400当默认。若今晚基线与新payload的模型身份不同，跑下方默认命令。

```bash
# 4090：约5.81 GPU小时；flock失败就退出，不等候抢占。
set -euo pipefail
cd /home/weiland/projects/openpi
for h in 10 15 16; do
  flock -n /tmp/r17_gpu.lock exp/r17_integration/ops/formal_eval.sh \
    --model dp_libero --checkpoint /data/mw_finetune/dp/libero_policy_100000 \
    --server-node local --worker-node timan107 --replicas 12 --workers 24 \
    --gpu cuda:0 --memory-budget-gib 14 --port 23160 --driver-port 23179 \
    --official-50 --commit-horizon "$h" --seconds 43200 --episode-seconds 1800 --retries 2 \
    --run-name "horizon_dp_c${h}_$(date +%Y%m%d_%H%M%S)" --run
  sleep 45
  sleep 45
done
```

```bash
# H100：约14.85 GPU小时；在Cosmos/Fast默认基线全部结束后再排。
set -euo pipefail
cd /home/weiland/projects/openpi
for h in 10 16 20 32; do
  exp/r17_integration/ops/formal_eval.sh \
    --model fastwam_libero --checkpoint /data/mw_finetune/models/fastwam_libero \
    --server-node h100 --worker-node timan108 --replicas 2 --workers 24 \
    --gpu cuda:0 --memory-budget-gib 64 --port 23173 --driver-port 23177 \
    --official-50 --commit-horizon "$h" --seconds 86400 --episode-seconds 3600 --retries 2 \
    --run-name "horizon_fast_c${h}_$(date +%Y%m%d_%H%M%S)" --run
  sleep 45
  sleep 45
done
```

仅当无法复用合格默认臂时，分别补跑（同样先确认对应GPU空闲；H100两条依次运行）。**不传新flag**，保证测量原默认：

```bash
cd /home/weiland/projects/openpi
flock -n /tmp/r17_gpu.lock exp/r17_integration/ops/formal_eval.sh \
  --model dp_libero --checkpoint /data/mw_finetune/dp/libero_policy_100000 \
  --server-node local --worker-node timan107 --replicas 12 --workers 24 --gpu cuda:0 \
  --memory-budget-gib 14 --port 23160 --driver-port 23179 --official-50 \
  --seconds 43200 --episode-seconds 1800 --retries 2 --run
# 上条约3.47 GPU小时。
exp/r17_integration/ops/formal_eval.sh \
  --model fastwam_libero --checkpoint /data/mw_finetune/models/fastwam_libero \
  --server-node h100 --worker-node timan108 --replicas 2 --workers 24 --gpu cuda:0 \
  --memory-budget-gib 64 --port 23173 --driver-port 23177 --official-50 \
  --seconds 86400 --episode-seconds 3600 --retries 2 --run
# 上条为旧四suite预算；当前两suite命令见7.3。Cosmos补跑命令已按owner决定删除。
```

收工由协调者记录每臂实际run目录、完整 `SUMMARY.json` / `EPISODE_PLAN.json` / accepted journal / per_step / server serving日志；按第3节配对分析，另存新分析目录。GPU端到端尚未验证本flag；首个新臂检查前两局的cursor、尾请求NFE=0与最终执行确认，不因成功/失败选参。任何额外有界smoke只可在GPU空闲时由协调者安排，不能挤占当前基线。

## (5) 留给 owner 的问题

| 问题 | 本文已采用的默认决定，不阻塞准备 |
|---|---|
| “不掉成功率”的容差 | 本轮owner已明确要求沿用2pp非劣＋实测成本，已在6.2冻结；0/5pp仍只作敏感性。不再等待容差确认。 |
| Cosmos是否另做更长预测块研究？ | 本轮不做。若要C32，需要独立验证训练/解码几何及新权重或其他生成方案，不能宣称只是horizon knob。 |
| 是否值得省去附带object/goal以压缩GPU预算？ | 已按本轮owner请求解决：默认关闭的 `--suites` 已实现，原始任务身份CPU证明通过；新命令仅跑spatial/l10，见第(6)/(7)节。 |

交付界限：已完成历史与官方证据核查、预注册设计、默认关闭参数及CPU测试、准确命令和资源预算；**新模型增大H后的SR未知**，等待协调者在设备释放后执行。

## (6) 纯减步研究

Owner 原话：**「单纯减迭代步数的也要研究，DDPM研究减到5和10，其他两个的迭代次数是多少，少的话减到1 2」**。最新纠正：**「我们之前都试过k=1 2的cosmos了在单纯减步实验里面」**。前版未查旧实验就把R17接口的K2限制写成历史不可运行，并安排K3替代，结论与排程均撤回。本次未读取DP/Fast新候选臂SR；Cosmos采用既有结果，协调者已删除其4090默认、K3、K1，这三项继续取消。**随后owner决定「那就只测k=2算上清理步」：仅加回H100的K2 incl. clean step/C16一臂，timan108×12；不新增4090默认或K1/K3。**

### 6.1 明确回答：默认 K 与本轮选择

| 模型 | 默认 K / 本轮减步 K | 固定执行长度 C | 合法采样语义与现有支持 |
|---|---|---:|---|
| **DP-T** | **DDPM100 → DDIM10、DDIM5** | **5**（R17默认） | 保持同一ε预测网络、训练100点β/α、归一化；减步用 **trailing DDIM，η=0**。K10网格 `[99,89,…,9]`，K5 `[99,79,59,39,19]`，最后到clean；不是提前停止原DDPM100循环。`SamplerProfile(auto,K)`、provider、`plain_config(num_steps=K)`均已支持；显式DDPM且K≠100会拒绝。[exp/r17_integration/models/dp/sampler.py:18-52](/home/weiland/projects/openpi/exp/r17_integration/models/dp/sampler.py:18)；[exp/r17_integration/models/dp/sampler.py:73-98](/home/weiland/projects/openpi/exp/r17_integration/models/dp/sampler.py:73)；[exp/r17_integration/models/dp/provider.py:12-28](/home/weiland/projects/openpi/exp/r17_integration/models/dp/provider.py:12)；[exp/r17_integration/models/dp/serve.py:17-42](/home/weiland/projects/openpi/exp/r17_integration/models/dp/serve.py:17) |
| **Fast-WAM** | **K10 → K2、K1** | **5** | 原生shifted flow / Euler更新；重建完整从噪声到终点的K步网格，`shift=5`，K2中间t约833（非均匀500，实际按bf16量化），K1一次完整更新。VAE与video prefill仍各一次，只有action head迭代从10变2/1，因此耗时不会简单变成20%/10%。现有provider和CPU native-reference测试已覆盖K1/2/10。[exp/r17_integration/models/fastwam/manifest.py:30-44](/home/weiland/projects/openpi/exp/r17_integration/models/fastwam/manifest.py:30)；[exp/offline_search/rounds/r17/horizon/sources/fastwam_scheduler_continuous.py:64-96](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/fastwam_scheduler_continuous.py:64)；[exp/r17_integration/models/fastwam/staged.py:284-305](/home/weiland/projects/openpi/exp/r17_integration/models/fastwam/staged.py:284)；[exp/r17_integration/tests/models/fastwam/test_fastwam_native.py:14-55](/home/weiland/projects/openpi/exp/r17_integration/tests/models/fastwam/test_fastwam_native.py:14) |
| **Cosmos Policy** | **默认K5；本轮仅K2 incl. clean step** | **16** | 旧入口直接传官方 `num_denoising_steps_action=K`，没有经过R17的K白名单。保留旧标签与成绩；新臂也使用官方num_steps=2语义并实测actual NFE（含clean调用），不强制把标签K2解释成2次前向，不另加clean或改成K3。H100、C16、timan108×12，见6.1c；旧审计见6.1a。 [exp/cosmos_nfe/serve_cosmos.py:147](/home/weiland/projects/openpi/exp/cosmos_nfe/serve_cosmos.py:147)；[logs/cosmos_nfe_plan.log.md:49](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:49) |

**DP研究名称须完整写成“DDPM100基线 vs 同ε模型 trailing-DDIM K10/K5”**，因为除了NFE也改变了采样器族。E的handback/RECIPE说明原始满100步与单观测；R17资格说明明确将减步定义为trailing DDIM。不能把旧截断DDPM或DP-CNN其他任务的结果当作此次DP-T LIBERO证据。[exp/mw_finetune/dp/HANDBACK_E.md:3-6](/home/weiland/projects/openpi/exp/mw_finetune/dp/HANDBACK_E.md:3)；[exp/mw_finetune/dp/RECIPE.md:71-79](/home/weiland/projects/openpi/exp/mw_finetune/dp/RECIPE.md:71)；[exp/r17_integration/qualification/PLAN.md:193-196](/home/weiland/projects/openpi/exp/r17_integration/qualification/PLAN.md:193)；[logs/session_handoff.md:164](/home/weiland/projects/openpi/logs/session_handoff.md:164)；[logs/session_handoff.md:223](/home/weiland/projects/openpi/logs/session_handoff.md:223)

### 6.1a Cosmos：已完成研究及NFE/GPU限制（保留历史审计）

**如何跑K2。** 2026-09-16使用官方 `Cosmos-Policy-LIBERO-Predict2-2B`、vendor `18a2acc`，`--num_denoising_steps_action 2`；官方50初态×10任务、seed195、deterministic、预测16并执行16，保持horizon。远程wrapper逐请求透传K，JSON顶层/配置中的K2是实际请求；`remote.num_denoising_steps_action=5`只是握手返回的服务器默认，不能误读成K2请求仍跑K5。[logs/cosmos_nfe_plan.log.md:8](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:8)；[logs/cosmos_nfe_plan.log.md:12](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:12)；[exp/cosmos_nfe/serve_cosmos.py:147](/home/weiland/projects/openpi/exp/cosmos_nfe/serve_cosmos.py:147)；[exp/cosmos_nfe/data/results/libero_spatial/k2/shard0.json:7](/home/weiland/projects/openpi/exp/cosmos_nfe/data/results/libero_spatial/k2/shard0.json:7)；[exp/cosmos_nfe/data/results/libero_spatial/k2/shard0.json:162](/home/weiland/projects/openpi/exp/cosmos_nfe/data/results/libero_spatial/k2/shard0.json:162)

| suite / 旧名义K | 成功数 / episodes | SR / 对K5差值 | chunk延迟ms（分片均值的均值） | 当时GPU / 完整性及来源 |
|---|---:|---:|---:|---|
| spatial K5 | 493/500 | 98.6% / 基线 | 1542.72 | H100，完整；[exp/cosmos_nfe/data/aggregate.json:308](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:308)；[exp/cosmos_nfe/data/aggregate.json:364](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:364) |
| spatial K3 | 492/500 | 98.4% / −0.2pp | 1188.11 | H100，完整；[exp/cosmos_nfe/data/aggregate.json:248](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:248)；[exp/cosmos_nfe/data/aggregate.json:304](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:304) |
| spatial K2 | 487/500 | 97.4% / −1.2pp | 847.18 | H100，完整；[exp/cosmos_nfe/data/aggregate.json:188](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:188)；[exp/cosmos_nfe/data/aggregate.json:244](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:244) |
| spatial K1 | 487/500 | 97.4% / −1.2pp | 837.94 | H100，完整；[exp/cosmos_nfe/data/aggregate.json:128](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:128)；[exp/cosmos_nfe/data/aggregate.json:184](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:184) |
| libero_10 K5 | 490/500 | 98.0% / 基线 | 2388.56 | **4090**，完整；[exp/cosmos_nfe/data/aggregate.json:66](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:66)；[exp/cosmos_nfe/data/aggregate.json:122](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:122) |
| libero_10 K2 | 486/500 | 97.2% / −0.8pp | 1075.06 | **H100**，完整，跨卡差值仅描述；[exp/cosmos_nfe/data/aggregate.json:6](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:6)；[exp/cosmos_nfe/data/aggregate.json:62](/home/weiland/projects/openpi/exp/cosmos_nfe/data/aggregate.json:62) |
| libero_10 K3 / K1 | K3 270/277；K1未跑 | K3 97.47%仅partial | 不用于结论 | owner在K2完成后叫停；不能写两个suite的K1都完成。[logs/cosmos_nfe_plan.log.md:48](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:48) |

成本来自wrapper包围 `get_action` 的host wall；远程包括通信，分片等权均值不是全请求加权均值，也不是无竞争CUDA kernel延迟。spatial K2/K1与K5比约 **0.549/0.543**（约快1.82/1.84倍），对应当时10副本并发的观测；l10的0.450比值混合4090与H100，**不能归因于纯减步**。日志记录spatial K5/K3/K2约20/17/16分钟，l10 K5约100分钟，拓扑也不同。[exp/cosmos_nfe/run_libero_shard.py:175](/home/weiland/projects/openpi/exp/cosmos_nfe/run_libero_shard.py:175)；[exp/cosmos_nfe/aggregate_cosmos.py:40](/home/weiland/projects/openpi/exp/cosmos_nfe/aggregate_cosmos.py:40)；[logs/cosmos_nfe_plan.log.md:39](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:39)

**采样器勘误须与“实验确实做过”同时保留。** 旧日志46/49行用基类 `res_sampler.py` 的dummy得出“名义K→K+1 NFE，σ_min=.002”。但其所记固定版本的策略模型实际实例化 `CosmosPolicySampler`；LIBERO配置是σ80→4，Policy子类在K>1先减1，内部1直接一次denoise，所以**按该固定源码推导名义K1/K2同为1 NFE，K3/K5为3/5 NFE**。这是源码推导，旧episode JSON没有实际NFE计数，不能把基类dummy当成当时策略运行的NFE测量。旧spatial K1/K2逐集500项零差异、延迟近同与该分支一致；日志45行临时摘要的K1=.976也被最终聚合487/500纠正。保留名义K2旧结果，不把它说成未运行或独立“真实2 NFE”证据；**不以此安排K3替代；新增K2仅依据owner最新明确决定**。此前R17拒绝K2是接口限制，不能否定旧原生参数实验；D正在解除该限制并分开记录名义K与实际NFE，见6.1c。[logs/cosmos_nfe_plan.log.md:45](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:45)；[/home/weiland/projects/openpi_ext/lines/cosmos-policy/cosmos_policy/models/policy_text2world_model.py:238](/home/weiland/projects/openpi_ext/lines/cosmos-policy/cosmos_policy/models/policy_text2world_model.py:238)；[/home/weiland/projects/openpi_ext/lines/cosmos-policy/cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py:192](/home/weiland/projects/openpi_ext/lines/cosmos-policy/cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py:192)；[exp/offline_search/rounds/r17/horizon/sources/cosmos_sampler.py:85](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/cosmos_sampler.py:85)；[exp/offline_search/rounds/r17/horizon/sources/cosmos_sampler.py:125](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/cosmos_sampler.py:125)；[exp/r17_integration/handbacks/HANDBACK_COSMOS.md:180](/home/weiland/projects/openpi/exp/r17_integration/handbacks/HANDBACK_COSMOS.md:180)

旧结论是“LIBERO减步几乎无损”，本次准确转述为：上述点估计仅下降0.8–1.2pp，有明显spatial观测延迟下降；**不是本轮预注册的24重比较、2pp非劣已经通过**。l10跨GPU、K1缺失及旧NFE标签使证据边界更窄。旧结果保持描述性；最新owner仅指定增加同H100的K2配对臂，不补K1/K3或4090控制。另有RoboCasa-2024 K1完整 **807/1200=67.25%**，对照是论文K5的67.1%而非同次配对；K2仅997集、pooled67.1%，不当完整控制研究。其eager K5/K1的821/565ms是RC24微基准，不能充作LIBERO单流成本。[logs/cosmos_nfe_plan.log.md:49](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:49)；[logs/cosmos_nfe_plan.log.md:72](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:72)；[logs/cosmos_nfe_plan.log.md:60](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:60)

只读核对还覆盖 `/data/cosmos/results/libero_10/k5/shard0..4.json`（合计490/500）与原始日志，例如shard0为99/100；repo `exp/cosmos_nfe/data/results/`保存H100结果及逐集布尔值。未修改任何原始记录。[/data/cosmos/results/libero_10/k5/shard0.json:12](/data/cosmos/results/libero_10/k5/shard0.json:12)；[/data/cosmos/logs/ENV_EVAL-libero_10-cosmos-2026_09_16-11_46_11--k5-s0-seed195.txt:6684](/data/cosmos/logs/ENV_EVAL-libero_10-cosmos-2026_09_16-11_46_11--k5-s0-seed195.txt:6684)

### 6.1b DP-T / Fast-WAM：前序研究是否能省去本轮臂

检索范围：`logs/`、`docs/`、offline_search早期轮次和R17调研/交接、`exp/dp_nfe`、`exp/xwam_nfe`、`exp/mw_finetune`、R17资格/样本，以及本地 `/data/mw_finetune/eval` 的DP/Fast结果目录；以下“未找到”限于这些可读记录，不将代码支持或资格smoke冒充闭环SR研究。

| 对象 / 已有证据 | 已测内容、局数与结果 | 能否替代本轮 / 剩余必要性 |
|---|---|---|
| 早期**DP-CNN**，不是本轮DP-T | robomimic四任务＋PushT，每格50集；DDPM100/DDIM10：square .78/.86、can 1/.98、tool_hang .82/.88、transport .82/.84，PushT .85/.85是coverage而非二值SR。4090空卡四个robomimic任务约1305–1363ms→142–155ms。旧DDPM10失败结论已于9/23撤销（旧diffusers跨步错误），DDIM旧表是leading；后续trailing复测square DDIM10=.68、can=.94、PushT coverage=.865。[logs/xwam_dp_nfe_plan.log.md:39](/home/weiland/projects/openpi/logs/xwam_dp_nfe_plan.log.md:39)；[logs/xwam_dp_nfe_plan.log.md:63](/home/weiland/projects/openpi/logs/xwam_dp_nfe_plan.log.md:63)；[logs/xwam_dp_nfe_plan.log.md:51](/home/weiland/projects/openpi/logs/xwam_dp_nfe_plan.log.md:51)；[logs/x0_multimodal_plan.log.md:371](/home/weiland/projects/openpi/logs/x0_multimodal_plan.log.md:371)；[logs/x0_multimodal_plan.log.md:373](/home/weiland/projects/openpi/logs/x0_multimodal_plan.log.md:373) | 已有减步研究，但模型/权重/任务/网格不同；不能省DP-T LIBERO DDIM10/5，也不能证明其C10/15/16安全。 |
| **DP-T / DP-MW**集成减步 | E记录满100步默认及显式override；P4已给MW/LIBERO合法trailing-DDIM支持、CPU/原生parity。48个LIBERO训练帧proxy中K10对数据MSE .239900 vs DDPM100 .238635，K1 1.948037；这不是episode SR。MT50正式结果只有所核满步锚61.2%（原driver）、61.0%（R17），各500；未找到DP-MW减步完整SR表，不能因题干提到“DP-MW减步”便补造结果。[exp/mw_finetune/dp/RECIPE.md:71](/home/weiland/projects/openpi/exp/mw_finetune/dp/RECIPE.md:71)；[exp/r17_integration/handbacks/HANDBACK_P4.md:265](/home/weiland/projects/openpi/exp/r17_integration/handbacks/HANDBACK_P4.md:265)；[exp/r17_integration/handbacks/HANDBACK_P4.md:287](/home/weiland/projects/openpi/exp/r17_integration/handbacks/HANDBACK_P4.md:287)；[exp/mw_finetune/RESULTS.md:11](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:11)；[exp/mw_finetune/RESULTS.md:50](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:50) | **保留DP全部5候选臂**。先前成文的DP-T LIBERO结果是非标准400局满步锚（新的official-50默认另由协调者执行）；未找到同checkpoint、官方A、配对的K10/5或C梯度研究。[exp/mw_finetune/RESULTS.md:33](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:33) |
| **X-WAM**（独立模型，不能当Fast-WAM） | RC24每档24×50=1200局，K10/5/3/2/1 SR=79.17/78.08/78.58/76.83/79.33%，H100含队列s/query=.76/.43/.32/.27/.22；固定执行32。RoboTwin每档50×20=1000，K10→K1为90.8→86.4%，并非所有benchmark一步都无损。[logs/xwam_dp_nfe_plan.log.md:6](/home/weiland/projects/openpi/logs/xwam_dp_nfe_plan.log.md:6)；[logs/xwam_dp_nfe_plan.log.md:54](/home/weiland/projects/openpi/logs/xwam_dp_nfe_plan.log.md:54)；[logs/xwam_dp_nfe_plan.log.md:106](/home/weiland/projects/openpi/logs/xwam_dp_nfe_plan.log.md:106) | 有相关族减步研究，无本轮Fast-WAM身份；不替代Fast任何臂。 |
| **Fast-WAM公开减步** | 先前R17调研已引用官方仓issue73的发布权重实测：NFE10/2/1总体96.95/96.35/95.75%，各2000局；spatial96.4/97.0/97.2%，l10 **95.2/93.4/92.0%**。发帖者的一手报告，非本组R17收据；回复亦称RoboTwin NFE1明显下降。[exp/offline_search/rounds/r17/survey3_astra/EVIDENCE.md:63](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/survey3_astra/EVIDENCE.md:63)；[原始issue73及回复](https://github.com/yuantianyuan01/FastWAM/issues/73) | **保留Fast全部6候选臂**。公开结果未提供可绑定本轮C5、GPU、seed/init、checkpoint/环境的逐集及成本收据；l10 K1点估计已−3.2pp，更不能用总体平均“近无损”免掉本轮配对。 |
| **本组Fast-WAM / horizon审计** | 本组找到MT50默认500局60.4%、资格K1/2/10原生parity；资格文档明确不认证SR。DP/Fast闭环表的执行C固定5；未找到本组已完成的DP-T C10/15/16或Fast C10/16/20/32官方A研究。[exp/mw_finetune/RESULTS.md:49](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:49)；[exp/r17_integration/handbacks/HANDBACK_P3.md:153](/home/weiland/projects/openpi/exp/r17_integration/handbacks/HANDBACK_P3.md:153)；[exp/mw_finetune/RESULTS.md:3](/home/weiland/projects/openpi/exp/mw_finetune/RESULTS.md:3) | DP/Fast这11候选臂均有必要；另按最新owner决定只加Cosmos K2，不追加MT50，不把“支持K/C”当作“已经测完”。 |

### 6.1c 最新owner决定：仅Cosmos K2 incl. clean step，以及Fast-WAM batching预注册

| 冻结项 | 本轮决定 |
|---|---|
| Cosmos唯一新增臂 | **K2 incl. clean step；官方上游num_steps=2语义，C16**；H100、2 replicas、timan108×12 workers；两主suite各500，共1000局。不新增4090 Cosmos默认，也不运行K1/K3。 |
| K与NFE | 名义K=2包含官方clean-step处理，严格沿用上游分支，不能额外补一次clean来凑两次。封存K、actual NFE、实际sigma序列、clean调用数/是否进入clean分支、sampler及代码身份；计数包含实际执行的clean前向。当前固定源码推导K2落在单次denoise分支，**新臂真实测量优先，绝不把K2标签写成“实测NFE2”**。6.1a旧NFE/GPU限制继续保留。 |
| 实现交付 | D负责 `HANDBACK_COSMOS.md` § **K2 incl. clean step**。本次读取时该新小节尚未落盘，但launcher白名单已出现Cosmos2，错误说明写明native K2 is NFE1；这不是整条链已验收。协调者在运行H7前核对D的最终handback、CPU证明及实际NFE记账；本助手不改D代码。[formal_eval.py:123](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:123)；[HANDBACK_COSMOS.md](/home/weiland/projects/openpi/exp/r17_integration/handbacks/HANDBACK_COSMOS.md) |
| 指定配对控制 | **`formal_cosmos_libero_official2000_20261010_100533_5f558380`，H100 K5/C16**；只提取spatial/l10各500 accepted，按canonical task＋原始init index/hash及环境身份配对，不按run内task_uid配对。默认与候选都用H100、同权重/精度/预处理/种子规则；默认24 workers与候选12的吞吐差异单独登记。控制不完整或身份不符则本比较待定，不私自新增控制臂。 |
| Cosmos成本资格 | 复用控制的SR按同GPU/身份验收；旧默认共卡竞争、worker数量变化可能影响时间。报告actual NFE/控制步及可比服务成本；若无等负载可比计时，只记成本未定，不宣称端到端加速，也不自动补Cosmos控制。 |
| Fast-WAM batching开放问题 | **是否采用P3正在测的server-side batching？** 交付目标 `exp/r17_integration/models/fastwam/BATCHING.md`；本次读取尚不存在，因此不声称已通过或编造batch flag。默认先保留已资格验证的batch配置，是否采用依据P3数值/身份/吞吐结果，在候选结果分析前冻结。 |
| Fast-WAM批配置匹配规则 | **若采用batching，K10/C5控制与全部horizon/step候选必须共享同一批配置**：batch模式/最大batch size、动态聚合与等待窗口、padding/mask、精度/编译、副本与服务并发配置及实现版本全部记录并一致。记录实际batch size分布与排队/服务时间；不把batching收益归到K或C。现有控制配置不匹配则不能沿用作该配置主控制；用匹配配置重测默认，已测候选也须匹配重测，不混合批配置池化。该重测替换原比较，不增加24项族；如要比较两种batch配置本身，另注册，不见SR后择优。 |

### 6.2 协议与决策规则补充

| 项目 | 当前注册（替代首版的执行范围和对比总数） |
|---|---|
| 测试集 | **只跑 `libero_spatial` 与 `libero_10`**。每臂用 `--official-50 --suites libero_spatial libero_10`：每任务原始idx0..49，每suite500，总1000。保留原40任务表、全局task_id、init字节hash、seed/profile；不重编号。不加入MT50。 |
| 纯减步配对 | DP K10/K5分别对同4090的DDPM100/C5；Fast K2/K1分别对同H100的K10/C5；Cosmos K2 incl. clean step/C16对指定H100 K5/C16，见6.1c。所有checkpoint、输入、C、sampler非K参数与环境设置按表冻结。 |
| 对照选择 | Fast优先复用用户指定H100官方A默认臂，须两个主suite均500 accepted、无未解决错误；DP使用Q修复后的完整4090默认重跑。旧DP167局不能作默认；Cosmos仅复用指定H100默认，不新增默认。Fast batching匹配优先于默认复用，见6.1c。触发重跑由完整性/身份/设备证据决定，不按SR高低挑对照。 |
| SR门 | 沿用第(3)节：δ=2pp、每suite500对、g/l、分层配对bootstrap10000次、seed20261010、两种下界都大于−.02、两个主suite分别通过。仅“未显著下降”仍不算通过。 |
| 多重性 | 最新owner只加回Cosmos K2一臂，horizon14个＋step10个＝**24个**：[DP(3+2)+Fast(4+2)+Cosmos1]×2 suite。第(3)节同步`a=.05/(2×24)`的CP差下界、bootstrap单侧分位`.05/24`；δ仍2pp。旧Cosmos历史不进入确认性族。0/5pp仅敏感性描述。 |
| 成本门 | horizon仍要求模型调用/实际控制步及服务时间/控制步都下降。**step臂应要求实际NFE/控制步与服务时间/控制步都下降**；C不变时不要求模型调用数下降。记录M、Σactual_nfe、T、host service/queue/elapsed、GPU占用小时；不把K/K0当作实测耗时。 |
| 对照计时资格 | 当前H100 Cosmos与Fast时间段重叠，服务时间可能受共卡竞争影响。SR对照可按同卡与身份规则复用；成本只能用可证明相同负载或无外部竞争的计时。无法证明时，标“成本未定”，按第(7)节条件命令补隔离的Fast默认；不从较慢的共卡默认直接宣称纯K带来全部墙钟节省。 |
| 选K | 所有臂完整测完；仅DP从100→10→5、Fast从10→2→1，首次不通过则保留前档，仍全表报告后续臂；Cosmos仅检验K5→K2，不延伸K1/K3。horizon与step分开选；本研究不推断“最佳C＋最佳K”联合后仍非劣。联合方案须另做配对确认。 |
| 样本/设备身份 | 主比较按 `(model, GPU type, suite, canonical task, init index/hash, environment profile/seed)` 配对。更换worker节点须原simulator/source/init/probe身份一致；不得把timan节点变化当成新的训练/任务条件。两卡结果不池化、不跨GPU做主配对。 |

五个减步候选臂共5,000局，七个horizon候选臂共7,000局，**共12个候选臂/12,000局/24个主比较**。含必要DP默认重跑为13,000局；DP默认已验收不重复。Fast默认按GPU/身份/批配置匹配复用，否则H0另1,000局（已跑候选如批配置改变也必须重测，单列预算）。**Cosmos仅新增H100 K2/C16一臂1000局**，指定现有K5控制不增加局数；4090 Cosmos默认及K1/K3继续取消。

### 6.3 前版已完成的默认关闭实现与CPU证明（D的K2增补另验收，本次未改代码）

| 新增 / 调整 | 行为与证据 |
|---|---|
| `--denoising-steps K` 默认None | 前版仅纯推理LIBERO：DP5/10/100、Fast1/2/10、Cosmos1/3/5；旧CPU证明包含拒绝Cosmos2。D正在将Cosmos2作为官方名义K支持（6.1c），其最终测试另以D handback为准；仍拒绝和`--commit-horizon`同时传入。未传K时保持原默认。[exp/r17_integration/ops/formal_eval.py:123-134](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:123)；[exp/r17_integration/ops/formal_eval.py:1013-1016](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:1013) |
| 调度封存/执行 | 调用各provider真实 `sampler_for_steps(K)`，保存 `execution_sampler.json`；计划核对其identity/K，任务与endpoint绑定所选K；逐行验证实际stage calls/NFE/sampler identity，防止只改命令标签。模型原default sampler仍保留为参考；服务握手另公布当前execution_sampler及C。[exp/r17_integration/ops/formal_eval.py:151-185](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:151)；[exp/r17_integration/ops/formal_eval.py:213-218](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:213)；[exp/r17_integration/ops/formal_driver.py:37-56](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_driver.py:37)；[exp/r17_integration/ops/formal_eval.py:565-595](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:565) |
| `--suites …` 默认None | 仅完整official50普通臂；先构造原2000局计划，再按suite过滤。benchmark/task表/YAML不因过滤改变；同一campaign下返回任务的**整个EpisodeTask包括task_uid逐字段等于原计划子集**。默认仍四suite；拒绝未知/重复suite、smoke、library、MW组合。[exp/r17_integration/ops/formal_eval.py:137-148](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:137)；[exp/r17_integration/ops/formal_driver.py:70-78](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_driver.py:70)；[exp/r17_integration/ops/tests/test_formal_steps_suites.py:59-81](/home/weiland/projects/openpi/exp/r17_integration/ops/tests/test_formal_steps_suites.py:59) |
| DP graph自检 | 选择减步时，启动自检也比较该K的eager与graph物理动作；仍保留原DDPM100 native自检。无新flag时沿原逻辑。GPU实际检查等协调者启动，CPU只证明选择参数传到自检。[exp/r17_integration/ops/formal_server.py:14-57](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_server.py:14)；[exp/r17_integration/ops/formal_server.py:98-103](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_server.py:98) |
| CPU验证 | 参数→封存→计划→公共ConnectionPolicy实跑NFE、默认不变、suite原身份、错误K/混合干预/伪sampler拒绝；DP5/10与Diffusers更新逐值相同，Fast原生K1/2/10及CosmosK2拒绝也覆盖。最终完整回归 **120 passed、27 subtests passed，无跳过/排除项**（含Q修复后的运维回归）；[CPU_TESTS_STEPS_REGRESSION.log:9](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/CPU_TESTS_STEPS_REGRESSION.log:9)。未加载真实GPU权重、未部署或启动远程任务。[exp/r17_integration/ops/tests/test_formal_steps_suites.py:1-173](/home/weiland/projects/openpi/exp/r17_integration/ops/tests/test_formal_steps_suites.py:1)；[exp/offline_search/rounds/r17/horizon/CPU_TESTS_STEPS.log:9](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/CPU_TESTS_STEPS.log:9) |

以下是前版CPU复查命令，本次未重跑；所有CUDA设备隐藏；Q的新内存准入代码保留。本次没有修改原始模型/采样器或既有结果。

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/ops/tests/test_formal_horizon.py \
  exp/r17_integration/ops/tests/test_formal_steps_suites.py \
  exp/r17_integration/ops/tests/test_formal_eval.py \
  exp/r17_integration/tests/benchmarks/test_libero.py \
  exp/r17_integration/tests/models/dp/test_sampler.py \
  exp/r17_integration/tests/models/fastwam/test_fastwam_native.py \
  exp/r17_integration/models/cosmos/test_cpu.py
```

## (7) 两卡排程

Owner 设备决定：**「本机可以分担cosmos和wam的测试」**。容量按本轮owner提供的 **4090 48GB、H100 80GB**，不是把4090当24GB；单副本约Fast26GB、Cosmos6–8GB、DP0.9GB。Cosmos已有4090烟测实测峰值8.10GiB、H1008.26GiB，说明还需峰值与加载余量。[exp/r17_integration/handbacks/HANDBACK_COSMOS.md:107-109](/home/weiland/projects/openpi/exp/r17_integration/handbacks/HANDBACK_COSMOS.md:107)；[exp/r17_integration/handbacks/HANDBACK_COSMOS.md:237-244](/home/weiland/projects/openpi/exp/r17_integration/handbacks/HANDBACK_COSMOS.md:237)

### 7.1 设备、对照与worker分配

| 资源 | 保留的剩余队列分配 | 内存/配对约束 |
|---|---|---|
| **4090 48GB** | **仅DP：12副本、14GiB总预算** | 默认与5个候选均同4090、同副本数；删除原L6–L8 Cosmos全部安排。内存不通过就停，不静默改变拓扑。 |
| **H100 80GB** | **Fast：2副本、64GiB总预算**，H1–H6；然后**Cosmos K2：2副本、18GiB预算**，H7 | 两副本约52GB；第三副本约78GB，几乎不剩加载/工作区余量，不安排。复用H100 K10/C5的SR对照；如成本计时不合格，追加隔离默认。 |
| timan107 | 只给4090队列：**24 workers**，`--worker-memory-mib 6144` | 6GiB×24=144GiB预留，再加Q的max(8GiB,节点/cgroup的10%)余量；绝不再同时放另一支24-worker fleet。 |
| timan108 | 只给H100队列：**Fast24、Cosmos K2为12 workers**，同6GiB/worker，先后运行不叠加 | 两节点各48核、约250GB RAM按owner输入；容量不是可用内存，必须由Q的MemAvailable/cgroup/RSS准入决定。`--extra-workers`保持0。 |
| 两队列之间 | 每卡一个campaign、每节点一支fleet；两卡同时推进 | 模型副本内部并发保留；不同研究臂不共卡，保证计时可解释。待现有Fast(timan107)和Cosmos(timan108)的fleet退出后再用对应节点。 |

**本机也能跑Fast-WAM：一副本26GB可容纳，两个约52GB则超48GB。** 本次仅在文档末尾加H100 Cosmos K2，不改运行中的队列或重新分配DP/Fast。保留DP→4090、Fast→H100同GPU配对（各节点24 worker），H7 Cosmos使用timan108×12；把Fast移本机会额外需要4090默认及单副本吞吐校准，现有证据不足以证明能缩短墙钟。4090在DP结束后释放，不再补Cosmos控制。Cosmos作者提示GPU差异，旧l10跨卡结果因此只作描述，见6.1a及 [exp/offline_search/rounds/r17/horizon/sources/cosmos_LIBERO.md:80](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/sources/cosmos_LIBERO.md:80)。只按最新owner决定加H100 K2，继续不创建4090控制或K1/K3。

Q已确认今天是timan107两支24-worker fleet重叠导致全局OOM，DP停于167/2000；新准入在准备和fork前检查，失败只拒绝本fleet，不终止其他进程。最新预览连额外8 workers都拒绝，不能因GPU有空就加worker。本文保留Q的默认有界重启，增加显式6GiB预留。[exp/r17_integration/qualification/HANDBACK_Q.md:1107-1137](/home/weiland/projects/openpi/exp/r17_integration/qualification/HANDBACK_Q.md:1107)；[exp/r17_integration/qualification/HANDBACK_Q.md:1163-1168](/home/weiland/projects/openpi/exp/r17_integration/qualification/HANDBACK_Q.md:1163)；[exp/r17_integration/ops/worker_memory.py:48-88](/home/weiland/projects/openpi/exp/r17_integration/ops/worker_memory.py:48)

| 正在运行 / 已失败的默认 | 本研究如何使用 |
|---|---|
| H100 Cosmos K5/C16：`formal_cosmos_libero_official2000_20261010_100533_5f558380` | 原配置2副本、timan108×24；本研究不变更该项目；验收后取两个主suite作H7 K2的指定同GPU控制，对应2个主比较。[exp/offline_search/rounds/r17/horizon/STEP_DEVICE_INPUTS.json:83-97](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/STEP_DEVICE_INPUTS.json:83) |
| H100 Fast K10/C5：`formal_fastwam_libero_official2000_20261010_101044_b0c4e68b` | 当前2副本、timan107×24。完成验收后复用主suite SR；新队列转到timan108，必须核对worker probe与init/source身份一致。计时审查见6.2；禁止按SR择优替换。[exp/offline_search/rounds/r17/horizon/STEP_DEVICE_INPUTS.json:98-112](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/STEP_DEVICE_INPUTS.json:98) |
| 4090 DP K100/C5：旧167局失败；Q修复后重跑 | 旧167局保留但不拼成一次500/suite。新的完整官方A若先完成即复用；否则执行L0。先前控制组缺少的结果不能按失败填0来“凑齐500”。[exp/r17_integration/qualification/HANDBACK_Q.md:1109-1119](/home/weiland/projects/openpi/exp/r17_integration/qualification/HANDBACK_Q.md:1109) |

### 7.2 逐臂预算与顺序（全部只含两主suite，各1000局）

以下是规划预算，不是候选臂实测时间。沿旧只读测量 `B_DP=1.9573、B_Fast=3.7031` driver小时；每臂加10min启动。horizon按`.25+.75×C0/C`；step按`1−.75×f_S3×(1−K/K0)`，f_S3取DP .8213、Fast .6885。预算至少±50%，不是已校准成本。旧 `TWO_GPU_ESTIMATES.json` 保留为前版审计产物，**其Cosmos行/总计已经失效，不是当前队列**；本文仅取其中DP/Fast逐臂值重新求和，不改旧JSON。[exp/offline_search/rounds/r17/horizon/STEP_DEVICE_INPUTS.json:3](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/STEP_DEVICE_INPUTS.json:3)；[exp/offline_search/rounds/r17/horizon/TWO_GPU_ESTIMATES.json:23](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/TWO_GPU_ESTIMATES.json:23)

| 顺序 / 臂 | GPU | worker节点 / 数量 | 副本 | 预计GPU小时 | 同卡对照 |
|---|---|---|---:|---:|---|
| L0 DP K100/C5 | 4090 | timan107 / 24 | 12 | 2.12 | 本行默认 |
| L1 DP K100/C10 | 4090 | timan107 / 24 | 12 | 1.39 | L0 DDPM100/C5 |
| L2 DP K100/C15 | 4090 | timan107 / 24 | 12 | 1.15 | L0 DDPM100/C5 |
| L3 DP K100/C16 | 4090 | timan107 / 24 | 12 | 1.11 | L0 DDPM100/C5 |
| L4 DP DDIM10/C5 | 4090 | timan107 / 24 | 12 | 1.04 | L0 DDPM100/C5 |
| L5 DP DDIM5/C5 | 4090 | timan107 / 24 | 12 | 0.98 | L0 DDPM100/C5 |
| 现有默认 Fast K10/C5 | H100 | timan107 / 24（既有） | 2 | 复用新增0（原估3.87） | 本行默认 |
| H1 Fast K10/C10 | H100 | timan108 / 24 | 2 | 2.48 | H100 K10/C5 |
| H2 Fast K10/C16 | H100 | timan108 / 24 | 2 | 1.96 | H100 K10/C5 |
| H3 Fast K10/C20 | H100 | timan108 / 24 | 2 | 1.79 | H100 K10/C5 |
| H4 Fast K10/C32 | H100 | timan108 / 24 | 2 | 1.53 | H100 K10/C5 |
| H5 Fast K2/C5 | H100 | timan108 / 24 | 2 | 2.34 | H100 K10/C5 |
| H6 Fast K1/C5 | H100 | timan108 / 24 | 2 | 2.15 | H100 K10/C5 |
| **H7 Cosmos K2 incl. clean step/C16** | **H100** | **timan108 / 12** | **2** | **2.46（保守预算）** | 指定现有H100 K5/C16，见6.1c |

Cosmos H7预算 **2.46 GPU小时**：旧H100×2默认driver 1.1458h，24→12 workers按最多2倍时间保守外推，再加10min启动，即`2×1.1458+1/6`；不预支K2/NFE减少的加速，真实NFE与12-worker吞吐未据此推定。输入见 [TWO_GPU_ESTIMATES.json:5](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/TWO_GPU_ESTIMATES.json:5)。同前预算至少±50%；该JSON旧Cosmos4090行仍作废，不改旧产物。

当前总预算（含L0，复用指定Cosmos/Fast默认；Fast批配置不变）：4090 **7.79h**、H100 **14.70h**，合计 **22.49 GPU小时**，并行墙钟约 **14.70h**。L0已验收则4090少2.12h、合计20.37h。H0若由完整性/计时/批配置门触发，再加旧配置估计3.87h，H100/墙钟约18.57h；采用新batch配置后的控制/候选耗时须按P3吞吐重新预算，不能沿用旧速率保证。已经跑过但批配置不匹配的候选重测成本额外列明，不隐藏在H0预算内。这里是总工作量估计，不是运行队列剩余时间；本次不触碰队列。


### 7.3 协调者命令：先核验释放，再在两个独立shell顺序执行

**下面命令是剩余臂的文档化交付，不是重新提交请求；已在队列中的臂不可重复提交。本次均未执行，也未编辑任何运行队列。** 不杀进程，不改现有run目录。Q修复与本flag随新的content-hash payload部署，现有服务不热更新。先只读核对设备、节点容量及现有campaign完整收据；内存拒绝就延期，不绕过Q准入。运行目录用新唯一名字。

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
tether exec --safe --timeout 30s h100 -- nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
tether exec --safe --timeout 30s h100 -- nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
tether exec --safe --timeout 30s timan107 -- free -m
tether exec --safe --timeout 30s timan108 -- free -m
```

**本机队列L0→L5**，只用timan107。下面shell在整个本机队列持有4090锁；flock失败就退出。若Q的完整4090默认已验收，省略且只省略L0那一行。

```bash
set -euo pipefail
cd /home/weiland/projects/openpi
exec 9>/tmp/r17_gpu.lock
flock -n 9
study_stamp=$(date +%Y%m%d_%H%M%S)
run_dp_study() {
  local label=$1
  shift
  exp/r17_integration/ops/formal_eval.sh \
    --model dp_libero --checkpoint /data/mw_finetune/dp/libero_policy_100000 \
    --server-node local --worker-node timan107 --replicas 12 --workers 24 \
    --worker-memory-mib 6144 --worker-restart-limit 2 \
    --gpu cuda:0 --memory-budget-gib 14 --port 23160 --driver-port 23179 \
    --official-50 --suites libero_spatial libero_10 \
    --seconds 43200 --episode-seconds 1800 --retries 2 \
    --run-name "hs2_dp_${label}_${study_stamp}" "$@" --run
}
run_dp_study k100_c5_default          # L0: 2.12h；已有合格Q重跑则省略。
run_dp_study k100_c10 --commit-horizon 10   # L1: 1.39h
run_dp_study k100_c15 --commit-horizon 15   # L2: 1.15h
run_dp_study k100_c16 --commit-horizon 16   # L3: 1.11h
run_dp_study ddim10_c5 --denoising-steps 10 # L4: 1.04h
run_dp_study ddim5_c5 --denoising-steps 5   # L5: 0.98h
# 本shell退出释放锁；所有owned进程须已由launcher验收关闭。
```

**H100队列H1→H6→H7**，只用timan108。协调者保证H100原Cosmos/Fast campaign都已关闭；下面本地queue锁只防重复提交本队列，**不占本机4090锁，也不是远端GPU准入的替代**。现行plain formal的远端GPU-lock guard仅在library分支使用，不能假定它为普通臂排他保留H100。[exp/r17_integration/ops/formal_eval.py:725-766](/home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.py:725)

```bash
set -euo pipefail
cd /home/weiland/projects/openpi
exec 9>/tmp/r17_h100_hs_study_queue.lock
flock -n 9
study_stamp=$(date +%Y%m%d_%H%M%S)
run_fast_h100_study() {
  local label=$1
  shift
  exp/r17_integration/ops/formal_eval.sh \
    --model fastwam_libero --checkpoint /data/mw_finetune/models/fastwam_libero \
    --server-node h100 --worker-node timan108 --replicas 2 --workers 24 \
    --worker-memory-mib 6144 --worker-restart-limit 2 \
    --gpu cuda:0 --memory-budget-gib 64 --port 23173 --driver-port 23177 \
    --official-50 --suites libero_spatial libero_10 \
    --seconds 86400 --episode-seconds 3600 --retries 2 \
    --run-name "hs2_fast_h100_${label}_${study_stamp}" "$@" --run
}
# H0: 对照完整性/身份/计时/批配置不匹配时补；不能按SR选择。
# 以下Fast命令保持当前批配置。若采用P3 batching，冻结匹配的控制与全部臂命令后再交付；不臆造flag。
# run_fast_h100_study k10_c5_control       # +3.87h，无K/C override。
run_fast_h100_study k10_c10 --commit-horizon 10 # H1: 2.48h
run_fast_h100_study k10_c16 --commit-horizon 16 # H2: 1.96h
run_fast_h100_study k10_c20 --commit-horizon 20 # H3: 1.79h
run_fast_h100_study k10_c32 --commit-horizon 32 # H4: 1.53h
run_fast_h100_study k2_c5 --denoising-steps 2   # H5: 2.34h
run_fast_h100_study k1_c5 --denoising-steps 1   # H6: 2.15h
# H7: 前序H100 fleet完成并释放；D的K2 handback/记账验收后执行。约2.46h。
# C16保留Cosmos默认，不同时传 --commit-horizon。
exp/r17_integration/ops/formal_eval.sh \
  --model cosmos_libero --checkpoint /data/mw_finetune/models/cosmos_libero \
  --server-node h100 --worker-node timan108 --replicas 2 --workers 12 \
  --worker-memory-mib 6144 --worker-restart-limit 2 \
  --gpu cuda:0 --memory-budget-gib 18 --port 23240 --driver-port 23249 \
  --official-50 --suites libero_spatial libero_10 --denoising-steps 2 \
  --seconds 43200 --episode-seconds 1800 --retries 2 \
  --run-name "hs2_cosmos_h100_k2_incl_clean_c16_${study_stamp}" --run
```

**Cosmos仅有H7 K2 incl. clean step/C16命令**，没有4090 Cosmos控制或K1/K3；没有K与C同时变化的臂。每臂核验 `execution_sampler.json`（减步臂）、`EPISODE_PLAN.json`中1000原始身份、两suite各500 accepted、actual NFE、实际执行回报及GPU类型后才作配对统计。默认臂不传K/C参数；仅加suite过滤不改变其动作或模型行为。任何GPU smoke/资格复查也由协调者另排，本次没有启动。
