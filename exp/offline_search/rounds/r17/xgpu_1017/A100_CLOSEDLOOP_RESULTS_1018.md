# A100 闭环跨卡结果（2026-10-18，接管 sol 后续跑）

**状态（19:15 CDT）：** π0.5、GR00T、DP、Fast-WAM 基础臂已完成；DP 扩展块 x1/x2 与同卡重复块 r1 已完成，r2 在跑。Cosmos 基础臂首次启动因运行脚本用了 Python 3.11 才有的 `hashlib.file_digest`（Cosmos island 为 3.10）在建 run 目录前退出，已修复并以原 UID 重排在 r2 之后；随后按预注册顺序跑 Cosmos x1 → GR00T x1 → π0.5 x1 → Fast-WAM x1 → GR00T x2 → π0.5 x2（`a100_closedloop_pipeline_1018b.py`，搬运间隙用 DP 同卡重复块填满 GPU）。本文件随每块落地更新。

## 0. 结论

**能说的：** 在已完成的四个模型、九个臂里，A100 与正式参考（4090 或 H100）的配对 SR 差点估计都在 −4 到 +1.4 pp 之间，exact McNemar 无一显著（最小 p=0.29）；能排除的只有大幅差异：DP spatial（各 500 对）的保守 95% CI 约 [−3.4,+2.4]（pure）、[−3.5,+5.5]（cache），DP libero_10 约 [−4.2,+6.2]，其余 100 对的臂 ±6 到 ±13 pp。

**不能说的：** 没有一个臂的 CI 落进预注册 ±2 pp，所以**不能认证任何模型的 A100 部署与正式部署等价**；也不能把差异归因 GPU 架构——每对比较同时换了卡、worker 数（8 vs 24–112）、渲染机和容量臂。

**同卡 floor 不是跨部署的零假设：** DP libero_10 上，A100 两次冷启动只有 10/500 episode 结果不同，而 A100 与正式 4090 不同的有 60/500、62/500（逐集 McNemar p≈3e-11、9e-13）；DP spatial pure 9 vs 17/16（p=0.08/0.14）。也就是成功率几乎不变、单集结果跨部署却明显更不稳定；用同卡重复的不一致率去估跨卡等价需要的样本会严重低估。π0.5/GR00T/Fast-WAM 在 100 对上看不出这种差别（太少）。

**要认证 ±2 pp 需要的配对数**（保守 exact 方法，真差为 0；见下表）：不一致率 2% 约 750 对（期望情形）/ 1,500 对（80% 功效）；3% 约 1,000 / 2,000；5–6%（GR00T cache、DP cache）约 1,500 / 3,000；10%（Fast-WAM cache 同卡、DP libero_10 跨卡 12%）约 2,500 / 5,000。按跨卡而不是同卡不一致率定样本：DP libero_10 需 ≥2,500–5,000 对，而 LIBERO 每 suite 官方只有 500 个 init，单 suite 无法达到，只能多 suite 合并或报“不确定”。

**已有 4090 vs H100 配对**（正式结果按 task/init/seed 对齐，模型/YAML/layout 身份不同，只作描述）：GR00T K4 +0.16 [−1.05,+1.37]、K2 0 [−1.30,+1.30]、DP MT50 +0.20 [−0.30,+0.70] 的区间落在 ±2 pp 内，但身份不一致，不签等价；Fast-WAM K1 spatial +1.2 [−1.5,+3.9]、libero_10 0 [−4.1,+4.1] 区间过宽；Fast-WAM MT50 K10 −1.97 [−4.1,+0.2]、p=0.005 只是 4090 完成的 1,674/2,500 子集，有选择偏差。

## 1. 本轮做了什么

只用了 A100（`tether exec a100`）和本机 CPU；没有碰 4090、H100、timan1、timan107，没有 tether push/pull/expose，没有 sudo，没有碰任何凭据/登录。所有传输都走公网 rsync，`--bwlimit=16384`，每次开一个临时 A100 daemon（端口 29770–29772，不在 23100–23199），传完按 PID 关掉；收据在 `a100_closedloop_1018/transfer/` 和 `CONTINUE_1018b.jsonl`。

设计沿用 sol 的预注册（π0.5/GR00T 同一套）：正式 planner 只筛原 UID、真实正式 worker、client/ACK 协议、源 campaign 的 serving runtime 与 recipe/bundle，8 个 A100 本机 EGL worker，每臂两次独立冷启动，每个模型一个显式 40GB 容量臂。另按协调员 15:00 的要求加了预注册扩展块（同一正式计划、与基础 100 集不重叠、只按原 init 区间选，结果出来前冻结 PLAN/UIDS，固定顺序、只按时间停止），清单见 [extension prereg](evidence/a100_extension_preregistration_1018b.json)。

### 1.1 DP 启动的五处阻塞（都已修，按出现顺序）

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | `PermissionError` 建 LIBERO assets symlink | rsync `-a` 保留了封存目录 `libero/libero` 的 `r-x` 权限 | stage 脚本改成幂等 `symlink_code`：只对这一级目录临时加 owner-write，建链后恢复原 mode；`.pth` 写入分开 |
| 2 | `DP model/architecture/language/preprocess identity mismatch` | DP 正式身份把 torch/transformers/numpy 等版本算进去；sol 把 DP 放在 Fast-WAM island（torch 2.7.1+cu128 / transformers 5.5.4） | A100 CPU 探针：只有 openpi island（`/home/exouser/xgpu_1017/env`，torch 2.7.1 / transformers 4.53.2）复现封存身份 `ed3ae8c6…`；DP server python 改用它（另加同版 websockets overlay） |
| 3 | `Repo id must be in the form …/data/mw_finetune/dp/clip_text` | 冻结 `Policy` 构造参数默认写死 `/data/mw_finetune/dp/clip_text`，loader 只校验迁移后的副本 | 在 A100 研究自己的 canonical 别名树里加两条 symlink 指向已校验副本，[操作记录](evidence/a100_operations_1018b.jsonl) |
| 4 | worker `cv2` 缺 `libpng16-…so`，随后缺 `matplotlib`、`exp.libero_groot` | 本机 island 里实际装的 `cv2` 是 headless wheel；sol 的 worker 闭包漏了它和一批包；而且 LIBERO worker 跑的是 server runtime，不是正式 campaign 的封存 worker 代码 | 用本机 CPU OSMesa 普查冻结 worker 模块的完整 import 闭包补齐（torch/CUDA 仍取 A100 openpi island）；LIBERO worker 改用各源 campaign 自己的 `workers.tar`（逐文件 SHA 校验后才启动）。π0.5/GR00T 的 worker 代码与 runtime 逐字节相同，已跑结果不受影响，见 [对照](evidence/worker_payload_vs_runtime_1018b.json) |
| 5 | DP cache：`DP cache study must match control R2/B8/5ms/MPS on local4090` | 正式 ff1015 DP cache 实际是 R4，经 `filter_failed/launch.py` 的 admission bridge（`server_cli.py`，只给 admission 看 R2，源 `CACHE_CAPACITY_ADMISSION.json` 有授权） | A100 resident 走同一个冻结 bridge（`filter_failed_server_cli.py`，SHA `26f08f96…`），拓扑、batching、recipe 不变 |

Fast-WAM 另有一处：冻结 runtime `1cf76a84`/`b29ef3db` 的 batch treatment 封条绑定 H100 R2、48 个 timan107 worker 和 GPU 型号，A100 上 server 必然拒绝。A100 容量臂不提交这个封条（`fastwam_batching` B16/5ms/native_bf16 不变），CPU 核对 resident 的 batch implementation identity 与封条 `f8bc81d5…` 相同；后来的 FW runtime `205509` 自己也关掉了同一组检查。原封条原样留在 campaign 的 `xgpu_source_batch_treatment`。

### 1.2 磁盘

A100 只有一块 58GB 盘。没有删除任何东西：已完成模型的副本和 A100 上唯一的原始结果都是“移走”（rsync `--remove-source-files` 拉回本机后核对每个文件到齐）。本机原件已有的副本放 `/tmp/xgpu_1017_a100_aside/`，A100 上唯一的结果（sol 的开环 results 2.29GB、π0.5/GR00T/DP 全部 run 目录 1.21GB）放 `/home/weiland/xgpu_1017_a100_aside/`。DP 和 Fast-WAM 共用一个 A100 rootfs（同一 fastwam island + LIBERO 源），省掉 9GB 重复传输。

## 2. 每模型运行与容量臂

| 模型 | 正式参考（卡） pure / cache | 正式拓扑 | A100 容量臂（显式） | runtime pure / cache |
|---|---|---|---|---|
| π0.5 MT50 | tpi7_0536 / pimk1_fuld n250 crg_k50（4090） | R5 / R4 | R2/B8，20GiB | 77681797… / af22613e… |
| GR00T MT50 | tgr7h1_full / ff1015_groot_mt_10 n1000 c（H100） | R8/B1，56GiB | R4/B1，28GiB | 13db208b… / 966ab88a… |
| DP spatial | hs7dpctl_full / ff1015_dp_li_14_u n50 c（4090） | R4/B8，40GiB，CUDA graph，MPS | R4/B8，40GiB（同正式），cache 走正式 R4 admission bridge | 1cf76a84… / 62692e19… |
| Fast-WAM spatial | hs6fw_res_152931 / ff1015_fastwam_li_18 n50 c（H100） | R2/B16，70GiB，48 worker | R1/B16，35GiB，不提交 H100 拓扑封条 | 1cf76a84… / b29ef3db… |
| Cosmos spatial | hs2_cosmos_h100_k2incl_c16_t107 / fa_co_r_u_l n50 crg_k50（H100） | R2 / R8，B1 | R2/B1，19GiB | 41f7d6a4…（旧 archive 逐文件封存）/ a7a387ed… |

所有臂：8 个 A100 本机 EGL worker（正式为 24–112 个远端 worker），每臂两次独立冷启动（新 resident、新私有 MPS），全部原 UID accepted、无 error、driver PASS、MPS 关闭、子进程全部回收；逐 run 收据见 `evidence/a100_closedloop_audit_*_1018*.json`。比较是部署级的（卡、worker 数、拓扑一起变），不能单独归因 GPU 架构。

## 3. `server_timing.clock_unit` 检查

GR00T、DP、Fast-WAM（以及 π0.5）的每一个 A100 run 都**没有** `server_timing.clock_unit == "segment_decision"`：π0.5 18,086、GR00T 16,288、DP 59,540、Fast-WAM 21,144 行 per_step（审计时已有的 run）中 `server_timing` 只有 `infer_ms`/`prev_total_ms`，worker 侧 socket trace 的 `cache_decision.clock_unit` 也全是空。原因：本研究复用的正式参考都是 C5 配方（`commit_horizon=5=replan_steps`），对应的冻结 runtime（13db208b、966ab88a、1cf76a84、62692e19、b29ef3db）早于 segment-boundary 修复，`serving.py`/`websocket_policy_server.py` 里根本没有 `segment_decision`；A100 server 跑的是和正式参考同一份冻结代码，没有套 `longseg_1017/runtime_fix.py`。这些 run 的每一行决策 `cursor=0`、执行 ACK 都是满 5 步，没有 tail delivery，所以 head/delivery 计数在这里是一回事，修复不改变这些 run 的行为；但按要求如实列出：**本研究没有任何一个 GR00T/DP/Fast-WAM 长段（C10，`fa_ls_*`）配方的 A100 run**。长段配方现有正式结果的有 `fa_ls_fw_f_b_l`、`fa_ls_fw_f_p_l`（FW LIBERO）和 `fa_ls_gr_f_p_m`（GR MT50），可作为下一步配 runtime_fix 的 A100 臂参考。

<!-- RESULTS -->
## 4. 闭环配对结果（表由 `a100_closedloop_report_tables_1018b.py` 从 [analysis JSON](evidence/a100_closedloop_analysis_1018b.json) 生成）

ΔSR = A100 − 正式参考（单位 pp）；b/c = 仅 A100 成功 / 仅正式成功（同卡行为 c1 / c2）；p 为 exact 双侧 McNemar；CI 为预注册保守 exact 95% 区间（各 cell Clopper-Pearson 的 Bonferroni 并），等价门槛 ±2 pp。N 列括号内是并入的块（基础 100 + 扩展块）；两次冷启动从不合并成更大的 N。

| 模型 / 臂 | 正式参考卡 | N（块） | 成功 正式/c1/c2 | 同卡 c1−c2（ΔSR pp；b/c；McNemar p；保守95%CI） | A100 c1−正式 | A100 c2−正式 |
|---|---|---:|---|---|---|---|
| π0.5 MT50 pure | 4090 | 100 (100) | 58/58/57 | +1.00；2/1；p=1；[-6.04, +7.86] | +0.00；1/1；p=1；[-6.20, +6.20] | -1.00；0/1；p=1；[-6.21, +4.27] |
| π0.5 MT50 cache | 4090 | 100 (100) | 54/52/52 | +0.00；1/1；p=1；[-6.20, +6.20] | -2.00；0/2；p=0.5；[-7.88, +4.12] | -2.00；0/2；p=0.5；[-7.88, +4.12] |
| GR00T MT50 pure | H100 | 100 (100) | 78/79/79 | +0.00；0/0；p=1；[-4.29, +4.29] | +1.00；1/0；p=1；[-4.27, +6.21] | +1.00；1/0；p=1；[-4.27, +6.21] |
| GR00T MT50 cache | H100 | 98 (98) | 51/52/51 | +1.02；3/2；p=1；[-7.55, +9.43] | +1.02；3/2；p=1；[-7.55, +9.43] | +0.00；3/3；p=1；[-9.11, +9.11] |
| DP spatial pure | 4090 | 500 (100+400) | 451/448/449 | -0.20；4/5；p=1；[-2.36, +1.97] | -0.60；7/10；p=0.629；[-3.42, +2.24] | -0.40；7/9；p=0.804；[-3.15, +2.37] |
| DP spatial cache | 4090 | 500 (100+400) | 351/358/354 | +0.80；17/13；p=0.585；[-2.85, +4.43] | +1.40；23/16；p=0.337；[-2.70, +5.47] | +0.60；21/18；p=0.749；[-3.51, +4.69] |
| DP libero_10 pure | 4090 | 500 (500) | 386/390/392 | -0.40；4/6；p=0.754；[-2.64, +1.86] | +0.80；32/28；p=0.699；[-4.17, +5.76] | +1.20；34/28；p=0.526；[-3.85, +6.23] |
| Fast-WAM spatial pure | H100 | 100 (100) | 92/90/90 | +0.00；1/1；p=1；[-6.20, +6.20] | -2.00；0/2；p=0.5；[-7.88, +4.12] | -2.00；0/2；p=0.5；[-7.88, +4.12] |
| Fast-WAM spatial cache | H100 | 100 (100) | 74/70/74 | -4.00；3/7；p=0.344；[-14.48, +6.93] | -4.00；2/6；p=0.289；[-13.47, +5.97] | +0.00；2/2；p=1；[-7.71, +7.71] |

| 模型 / 臂 | N | 同卡不一致 | 跨卡不一致 c1 / c2 | 跨卡−同卡不一致（逐集 exact McNemar p）c1 / c2 |
|---|---:|---:|---:|---|
| π0.5 MT50 pure | 100 | 3 (3.0%) | 2 / 1 | 0:1 p=1 / 0:2 p=0.5 |
| π0.5 MT50 cache | 100 | 2 (2.0%) | 2 / 2 | 1:1 p=1 / 1:1 p=1 |
| GR00T MT50 pure | 100 | 0 (0.0%) | 1 / 1 | 1:0 p=1 / 1:0 p=1 |
| GR00T MT50 cache | 98 | 5 (5.1%) | 5 / 6 | 3:3 p=1 / 3:2 p=1 |
| DP spatial pure | 500 | 9 (1.8%) | 17 / 16 | 12:4 p=0.0768 / 12:5 p=0.143 |
| DP spatial cache | 500 | 30 (6.0%) | 39 / 39 | 24:15 p=0.2 / 24:15 p=0.2 |
| DP libero_10 pure | 500 | 10 (2.0%) | 60 / 62 | 56:6 p=2.97e-11 / 56:4 p=9.08e-13 |
| Fast-WAM spatial pure | 100 | 2 (2.0%) | 2 / 2 | 1:1 p=1 / 1:1 p=1 |
| Fast-WAM spatial cache | 100 | 10 (10.0%) | 8 / 4 | 1:3 p=0.625 / 1:7 p=0.0703 |

合计 2098 对：同卡不一致 71，跨卡 c1 136，c2 133。

| 4090 / H100 campaign | suite | N | ΔSR(4090−H100) pp | b/c | exact McNemar p | 保守95%CI pp | 身份差异 |
|---|---|---:|---:|---:|---:|---|---|
| hs7fwk1_4090c / hs6fw_k1 | libero_spatial K1 | 500 | +1.20 | 11/5 | 0.21 | [-1.54, +3.90] | model_identity, yaml_identity |
| hs7fwk1_4090c / hs6fw_k1 | libero_10 K1 | 500 | +0.00 | 20/20 | 1 | [-4.15, +4.15] | model_identity, yaml_identity |
| tdp7_0729 / tdp7h1_full2 | metaworld_mt50 K100 | 2500 | +0.20 | 9/4 | 0.267 | [-0.30, +0.70] | model_identity, yaml_identity |
| tgr7_0502 / tgr7h1_full | metaworld_mt50 K4 | 2500 | +0.16 | 45/41 | 0.747 | [-1.05, +1.37] | layout_identity, model_identity, yaml_identity |
| tgr7k2_1044 / tgr7h1_k2 | metaworld_mt50 K2 | 2500 | +0.00 | 50/50 | 1 | [-1.30, +1.30] | layout_identity, model_identity, yaml_identity |
| tfw7_0755 / tfw7full_0944 | metaworld_mt50 K10（4090缺826） | 1674 | -1.97 | 48/81 | 0.00465 | [-4.12, +0.19] | UID namespace |

| 不一致率 d（真差0） | 期望情形 N | 80%功效 N |
|---:|---:|---:|
| 1.0% | 500 | 750 |
| 1.8% | 750 | 1500 |
| 2.0% | 750 | 1500 |
| 3.0% | 1000 | 2000 |
| 5.0% | 1500 | 3000 |
| 5.1% | 1500 | 3000 |
| 6.0% | 2000 | 4000 |
| 10.0% | 2500 | 5000 |

<!-- /RESULTS -->
## 5. DP cache 距离诊断（只诊断，正式配方未改）

首个 head 用 early metric：`y = x·W0f − c0`，`d² = n20 − 2·Z(A0·y) + y·y`，全程 CPU NumPy float32。三组证据：

**(a) 机制：float32 灾难性抵消。** 用 4090 B1 与 longseg 并发共用的封存 C10 fit（`fit.npz` sha `b8bb9bc2…`）和真实 4090 stage1 特征复算：被减的三项最大到 7.15×10⁵（`n20`、`2u·y`、`y·y`），而近邻 `d²≈2.1×10³`，抵消比约 340；float32 在 7×10⁵ 处 ulp=0.0625，对应距离台阶 6.8×10⁻⁴。本机 CPU 的 float32 复算与 4090、A100 记录的距离**逐位一致**，所以 CPU BLAS 不是来源。原 51 个超预算 4090/H100 请求的最坏跨卡配对，16 个 donor 的 d² 差**全部**是 float32 ulp 的整数倍（2⁻⁵ 或 2⁻⁶；816 个差值为 −1/0/+1/+2 个 ulp），最大距离差正好一个台阶 6.83×10⁻⁴。在 4090 特征上加 1e-7–1e-5 的相对扰动（donor 集不变），float32 权重漂移 8.8e-4–1.4e-3，float64 只有 4e-7–3.1e-5。

**(b) 跨卡：4090 vs A100，同工具、同 3 帧输入、同 recipe/fit（native profile）。** A100 B1 回放（GPU 只跑约 2 分钟，见 `collected/b1_dp_a100/`）与已有 4090 B1 比较：stage1 特征相对差 1.9–2.9×10⁻⁴，donor 集与顺序在 float32/float64/无抵消式下都一致；跨卡距离差 2.4–7.4×10⁻³（三种算术几乎一样，说明距离差主要来自特征），但**权重差** float32 4.2–9.6×10⁻⁴，float64 与无抵消式 2.7–8.7×10⁻⁵，缩小 5.5–36 倍。float64 与无抵消式相互一致到 1e-12（保留封存 n20 舍入残差，度量不变）。

**(c) 同卡闭环：A100 DP cache 两次冷启动，100 个 episode 的首个 head。** 正式观察器捕获 stage1 特征、x、y、raw d²（C5 ff1015 fit，与 (a)(b) 不是同一 fit）。61/100 的特征两次完全相同，39/100 有批组成导致的特征漂移（相对中位 1.7e-4，最大 4.6e-4）；最大权重漂移 float32 4.09e-4 → float64/无抵消式 1.84e-4（逐 episode 中位缩小 2.3 倍）；float32 下有 3 个 episode 的 donor 顺序翻转，float64/无抵消式下 0 个。捕获的 y 与 d² 都能在 CPU 逐位复现。

**结论：** 把 early-metric 距离改成 float64 或无抵消式，能去掉 float32 舍入引起的那部分权重抖动和近似并列翻转；在 TF32-off 的 4090/H100 并发（特征几乎不漂、观测到的差全是 ulp 台阶）里，这就是主要部分，所以 float64 很可能消除大部分跨卡权重漂移。但只要 stage1 特征本身跨卡或随批组成漂移（native profile 下约 2×10⁻⁴），float64 后仍剩 1e-5–2e-4 的权重差，**不能消除**。限制：4090/H100 的 TF32-off 特征没有记录、本任务禁止再碰这两张卡，所以 (a) 对那 51 个请求是机制推断，不是用两卡真实特征做的 float64 复算；(b) 是 native profile；正式 recipe 未改，也没做闭环 float64 臂。证据：[CPU 抵消](evidence/dp_cancellation_cpu_1018b.json)、[闭环捕获](evidence/dp_capture_analysis_1018b.json)、[队列内复算](a100_closedloop_1018/collected/dp_DISTANCE.json)。


## 6. 未完成与限制

- 长段（C10，`fa_ls_*`）配方没有 A100 臂，`runtime_fix.py` 未在本研究使用（见 §3）。
- 每个比较是“A100 部署 vs 正式部署”：卡、worker 数与渲染机、容量臂同时变化；同卡重复只在 A100 上做，正式卡的同卡重复不在本任务范围。
- DP cache 基础臂两次冷启动开着首 head 观察器（预注册的诊断 timing 臂）；扩展块与重复块都关掉观察器。
- π0.5/GR00T 基础臂的 worker 用 runtime 代码（与正式 `workers.tar` 逐字节相同）；LIBERO 三个模型改用正式 `workers.tar`。
- Fast-WAM 的 A100 臂不提交 H100 拓扑封条（§1.1）；Fast-WAM 队列内配对因 island 无 scipy 失败一次，改用 openpi island 重算，run 本身 4/4 COMPLETE。

## 7. 文件、命令与回收

- 结果与证据：[analysis](evidence/a100_closedloop_analysis_1018b.json)、`a100_closedloop_1018/collected/*_PAIRED.json`（逐 episode 0/1）、逐 run 审计 `evidence/a100_closedloop_audit_*`、扩展块预注册 [index](evidence/a100_extension_preregistration_1018b.json)、A100 操作 [记录](evidence/a100_operations_1018b.jsonl)、编排日志 `a100_closedloop_1018/CONTINUE_1018b.jsonl`。
- 编排：`a100_closedloop_continue_1018b.py`（stage/aside/push/pull/launch 各步）、`a100_closedloop_pipeline_1018b.py`（Cosmos 起的串行流水线）、`a100_closedloop_extend_1018b.py`（扩展块与重复块预注册）、A100 端 `a100_closedloop_ext_queue_1018b.py`。
- 分析：`.venv/bin/python exp/offline_search/rounds/r17/xgpu_1017/a100_closedloop_analysis_1018b.py`，表格 `a100_closedloop_doc_update_1018b.py`；DP 诊断 `dp_cancellation_cpu_1018b.py`、`dp_capture_analysis_1018b.py`。
- 搬离 A100 的文件：副本在 `/tmp/xgpu_1017_a100_aside/`（本机原件仍在，可删），A100 上唯一的结果在 `/home/weiland/xgpu_1017_a100_aside/`（请保留或归档）；每批的清单与核对见 `a100_closedloop_1018/transfer/aside_*/ASIDE.json`。
