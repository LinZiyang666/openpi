# 减步 vs warm start：闭环 shadow 诊断与等 NFE 对照 — 结果报告

> 计划：`logs/step_vs_warmstart_diagnostics_plan.log.md` v3.1（G1/G2 APPROVED，代码 `d8e464d`）。数据 2026-09-20 13:46 CDT 起跑、2026-09-21 01:38 CDT 收官（America/Chicago）。
> 生成表：`step_vs_warmstart_tables.md`（六环境 `shadow_<env>.md` + `qb_pi05.md`/`qb_groot.md` 拼接）；JSON 在 `exp/step_diag/data/analysis/`；parity 证据 `parity_<env>.json`。
> 本文只解释预注册量，不改判据、不换任务组、不换主预算。

## 0. 一句话结论

- **Q-B（两 policy 各自互斥判决）：均为 `inconclusive`，且均触发独立标记 `harmful_on_flat`。** 无门 top-1 forced warm start 在等决策 NFE 下没有补回减步缺口：π0.5 cliff 四任务 macro Δ = −0.035 [−0.145, +0.075]（H50 上界 −0.055 < 0，H25 上界 +0.006 未过 not-supported 门）；GR00T macro Δ = +0.055 [−0.045, +0.155]，H25 区间跨零。两 policy 各有 flat 任务 Δ 上界远低于 −0.10（π0.5 PickPlaceSinkToCounter −0.63、PickPlaceCounterToStove −0.78；GR00T PickPlaceCounterToStove −0.45），即 **warm start 在历史无缺口的任务上造成大幅损伤**。
- **Q-A：两 policy 的 RC 逐决策减步偏离 d_1 与历史阶梯缺口 g 均 `no_conclusion`**（π0.5 ρ = 0.085 [−0.42, 0.63]；GR00T ρ = 0.377 [−0.35, 0.90]，且 g 范围 0.30 只勉强过 0.15 门）。d_1 在同一 policy 内跨任务几乎常量（π0.5 RC 4.8–8.2、GR00T RC 1.7–2.1，LIBERO 更窄），不能作为任务级"能否减步"的门控量。
- **Q-C.3：π0.5 LIBERO-Object / Goal 阶梯平坦**（object k=1/2/4/10 = .984/.988/.978/.976；goal = .932/.956/.944/.952，各 500 集），与 spatial（k=10 锚 .988）一致；libero_10 同 harness k=10 锚 .836，与旧阶梯 k=7 的 .824–.850 同量级，说明旧线 "k=7 0.84 vs RIT 10 步参考 0.92" 的差异是 harness 差异，不是步数。

对四层设计的含义（候选依据，不是部署结论）：在本批 RoboCasa 任务上，"减步是否可接受"由闭环任务本身决定（cliff 组 g 复现），而逐决策偏离量不预测它；"warm start 是否有益"在无门 top-1 下只在个别任务成立（π0.5 CloseFridge、GR00T TurnOnSinkFaucet），并在 flat 任务上系统性有害——warm start 必须带门，不能作为默认减步替代。

## 1. 数据与准入

| 块 | 集数 | 拓扑 | 准入 |
|---|---|---|---|
| RC shadow（Q-A/Q-C.2） | π0.5 130 + GR00T 130（13 任务 × idx 0…9，seed 2,000,000+idx） | h100 ↔ timan108 | `analyze_shadow` 正式参数（N=4、coverage .9、≥8 集）：两条 130/130 admitted，error rows 0，missing decision rows 0（GR00T 有 4 条 server 端 `not_accepted_terminal` 行被拒，属重试非 accepted） |
| LIBERO shadow（Q-C.2） | 4 env × 100（10 任务 × 10 init） | weilandserver ↔ timan107 | 四条 100/100 admitted，无 rejection |
| Q-B π0.5 | 2,750（7 任务；full/plain_k1/k3/warm_t0.1/t0.3 各 50；plain_k2/warm_t0.2 flat 任务 100） | h100（MPS）↔ timan108 | 主判决三臂 × 7 任务全部 `complete=True, equal_nfe=True, problems={}`；miss_fraction 0.0；executed steps full 10 / plain 2 / warm 2 |
| Q-B GR00T | 1,350（5 任务；full/plain_k1/k2/warm_t0.75/t0.5 各 50；plain_k1/warm_t0.75 flat 任务 100） | weilandserver（MPS）↔ timan107；**full main 三任务改在 h100 ↔ timan108**（见 §5） | 主判决三臂 × 5 任务全部准入；miss 0.0；executed steps full 4 / plain 1 / warm 1；plain_k1 PnP 有 2 个 stray session（记录、不门控，见 §5） |
| Q-C.3 阶梯 | 5,000（object/goal k∈{1,2,4,10} × 500；spatial/libero_10 k=10 × 500） | weilandserver ↔ timan107（`nfe_baseline` 阶梯脚本，信号切档） | 每档 500/500 |

合计 9,760 集正式数据，与计划预算相同；smoke（`sdiag_smoke`、seed 3,000,000+）不入分析。所有 server 行 arrays 按 rows 内 sha256 逐文件校验：π0.5 七臂 2,750/2,750、GR00T 五臂 1,350/1,350 均 0 bad。等 NFE 审计量（主判决三臂，逐 cell 均值）：

| policy | 臂 | 每次决策 stage-3 前向 | 每集决策数 | 每集环境步数 | 每集总 NFE |
|---|---|---|---|---|---|
| π0.5 | full K=10 | 10 | 113 | 565 | 1132 |
| π0.5 | plain_k2 | 2 | 118 | 587 | 210 |
| π0.5 | warm_t0.2 | 2（WARM，剩余 2 步） | 135 | 673 | 258 |
| GR00T | full K=4 | 4 | 89 | 444 | 356 |
| GR00T | plain_k1 | 1 | 95 | 472 | 88 |
| GR00T | warm_t0.75 | 1（WARM，剩余 1 步） | 100 | 501 | 98 |

"等"只指每次决策的 stage-3 前向数；warm 臂每集决策数偏多（失败集跑满 300 步上限），所以每集总 NFE 不相等，且不含检索开销。

## 2. Q-A：逐决策减步偏离与阶梯缺口（探索性）

RC 13 任务、每任务 10 集、每教师决策 N=4 份同噪声样本；d_k = 前 5 步执行维加权 L2 的逐时刻均值，先取每集中位数再取任务中位数。

| policy | ρ(d_1, g) | 95% CI（任务簇 bootstrap 20000） | g 范围 | 判定 | LOTO（描述） |
|---|---|---|---|---|---|
| π0.5 RC | 0.085 | [−0.42, 0.63] | 0.78 | no_conclusion（ρ<0.4） | hits 3 / missed cliffs 4 / false cliffs 4 / correct flat 2 |
| GR00T RC | 0.377 | [−0.35, 0.90] | 0.30 | no_conclusion（ρ<0.4） | hits 3 / missed cliffs 1 / false cliffs 4 / correct flat 5 |

逐任务（d_1 / r_1 = d_1/disp_K / 历史 g / shadow 全步 SR）：

| 任务 | π0.5 d_1 | π0.5 r_1 | π0.5 g | π0.5 SR | GR00T d_1 | GR00T r_1 | GR00T g | GR00T SR |
|---|---|---|---|---|---|---|---|---|
| CloseBlenderLid | 4.76 | 3.32 | .06 | .20 | 1.71 | 1.41 | .04 | .80 |
| CloseFridge | 7.63 | 3.28 | .62 | .60 | 2.09 | 1.17 | .02 | .80 |
| CoffeeSetupMug | 5.68 | 4.17 | .20 | .50 | 1.73 | 1.46 | .12 | .60 |
| OpenCabinet | 6.29 | 2.98 | .42 | .70 | 1.83 | 1.35 | .18 | 1.00 |
| OpenDrawer | 6.24 | 3.67 | −.10 | .40 | 1.76 | 1.42 | .14 | .90 |
| OpenStandMixerHead | 6.08 | 3.87 | −.16 | .30 | 1.78 | 1.21 | .02 | .50 |
| PickPlaceCounterToCabinet | 7.05 | 3.23 | .18 | .40 | 1.71 | 1.39 | .00 | .50 |
| PickPlaceCounterToStove | 6.00 | 3.18 | −.16 | .60 | 1.80 | 1.38 | .04 | .70 |
| PickPlaceDrawerToCounter | 7.52 | 2.86 | .20 | .20 | 1.88 | 1.27 | .30 | .90 |
| PickPlaceSinkToCounter | 5.70 | 3.50 | .44 | 1.00 | 1.77 | 1.42 | .08 | .80 |
| PickPlaceToasterToCounter | 5.74 | 4.65 | .50 | .50 | 1.69 | 1.35 | .00 | .60 |
| SlideDishwasherRack | 8.16 | 5.06 | −.02 | .50 | 1.78 | 1.36 | .24 | .40 |
| TurnOnSinkFaucet | 4.83 | 4.43 | −.02 | .70 | 1.77 | 1.43 | .22 | .10 |

（g 为历史阶梯 SR(K)−SR(k=1)，π0.5 用 v2 参考、GR00T 用 50 集参考，`data/gaps/`。）

读法与机制假设：

- π0.5 的 1 步动作偏离全步 3–5 倍条件离散度（r_1 ≈ 3–5），GR00T 只有 1.2–1.5 倍：π0.5 单步误差远大于其自身噪声间离散度，GR00T 单步已接近"另一份噪声样本"。这与阶梯事实一致（π0.5 RC k=1 macro .39 vs 10 步 .56；GR00T k=1 缺口普遍 <0.3）。
- 但**在同一 policy 内**，d_1 跨任务只在 1.7 倍范围内变化，而 g 从 −0.16 到 +0.62；ρ 接近 0。任务级缺口不是由"平均一步偏离多大"决定的，更可能由该任务对少数关键决策的容错决定（对应 ΔBIC>10 比例：π0.5 CloseFridge .37、PickPlaceCounterToCabinet .57 高，PickPlaceSinkToCounter .13、SlideDishwasherRack .00 低，但也不单调对应 g）。
- 小 d_1 不授予"可整段减步"：GR00T TurnOnSinkFaucet d_1 = 1.77（中位水平）而 g = .22；π0.5 PickPlaceSinkToCounter d_1 = 5.70（偏小）而 g = .44。LOTO 阈值门控在两 policy 上各有 4 个 false cliffs，不能部署。

## 3. Q-B：warm start 能否补回同预算减步尚存的缺口

### 3.1 π0.5（主预算 m\*=2，t\*=0.2；库 W13 `pi05_spatial_pool_16_w13_full`，无门 top-1）

| 任务 | 组 | n | SR full | SR plain_k2 | SR warm_t0.2 | g | Δ = W−P | Δ 95%（探索） | H50 | H25 | 恢复比 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CloseFridge | cliff | 50 | .62 | .08 | .30 | +.54 | **+.22** | [+.06, +.38] | −.05 | +.09 | .41 |
| OpenCabinet | cliff | 50 | .66 | .50 | .42 | +.16 | −.08 | [−.22, +.06] | −.16 | −.12 | −.50 |
| PickPlaceToasterToCounter | cliff | 50 | .40 | .28 | .08 | +.12 | **−.20** | [−.36, −.04] | −.26 | −.23 | −1.67 |
| PickPlaceDrawerToCounter | cliff | 50 | .32 | .18 | .10 | +.14 | −.08 | [−.22, +.06] | −.15 | −.12 | −.57 |
| PickPlaceSinkToCounter | flat | 100 | 1.00 | 1.00 | .37 | – | **−.63** | 正式 [−.77, −.41] | – | – | – |
| OpenDrawer | flat | 100 | .62 | .75 | .71 | – | −.04 | 正式 [−.24, +.17] | – | – | – |
| PickPlaceCounterToStove | flat | 100 | .84 | .94 | .16 | – | **−.78** | 正式 [−.89, −.57] | – | – | – |

cliff macro（4 任务等权、bootstrap 100000、α = .05/12）：g +0.240 [+0.130, +0.350]，Δ −0.035 [−0.145, +0.075]，H50 −0.155 [−0.253, −0.055]，H25 −0.095 [−0.194, +0.006]。

- 历史 cliff 组缺口**复现**：g 点估计 .54/.16/.12/.14 对历史 .50/.18/.16/.18，macro g 下界 > 0。
- 判决：supported 需 Δ、H50 下界 > 0（未达）；not supported 需 H25 上界 < 0（上界 +0.006，差 0.006 未达）→ **inconclusive**。实质上 warm start 平均没有恢复缺口（H50 区间整体 < 0：恢复量显著少于一半），只有 CloseFridge 一个任务恢复 41%。
- `harmful_on_flat`：两个 flat PnP 任务 Δ 上界 −0.41 / −0.57，n10 = 0（warm 成功而 plain 失败的配对为零），n01 = 63 / 78。OpenDrawer 不受损。
- 其它预算（探索）：m=1 时 warm_t0.1 在 cliff 四任务 Δ 全为正（+.20/+.24/+.06/+.04，CloseFridge、OpenCabinet 区间不含 0），但 flat 两任务同样崩（−.36/−.86）；m=3 时 cliff 上 warm 反而更差（PickPlaceToasterToCounter −.32）。warm 的相对收益随预算增加而消失，与"warm start 只在剩余步数极少时才有替代价值"一致。

### 3.2 GR00T（主预算 m\*=1，t\*=0.75；库 W13 `groot_tp_spatial_pool_16_w13_full`）

| 任务 | 组 | n | SR full | SR plain_k1 | SR warm_t0.75 | g | Δ | Δ 95%（探索） | H50 | H25 | 恢复比 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PickPlaceDrawerToCounter | cliff | 50 | .66 | .40 | .48 | +.26 | +.08 | [−.08, +.24] | −.05 | +.02 | .31 |
| SlideDishwasherRack | cliff | 50 | .54 | .40 | .42 | +.14 | +.02 | [−.10, +.14] | −.05 | −.02 | .14 |
| TurnOnSinkFaucet | cliff | 50 | .18 | .04 | .28 | +.14 | **+.24** | [+.12, +.38] | +.17 | +.21 | 1.71 |
| OpenCabinet | cliff | 50 | .90 | .90 | .78 | .00 | −.12 | [−.26, +.02] | −.12 | −.12 | – |
| PickPlaceCounterToStove | flat | 100 | .82 | .94 | .49 | – | **−.45** | 正式 [−.64, −.20] | – | – | – |

cliff macro：g +0.135 [+0.035, +0.235]，Δ +0.055 [−0.045, +0.155]，H50 −0.012 [−0.107, +0.080]，H25 +0.021 [−0.074, +0.115]。

- 历史缺口**部分复现**：PickPlaceDrawerToCounter .26（历史 .30）、SlideDishwasherRack .14（.24）、TurnOnSinkFaucet .14（.22），OpenCabinet .00（.18）未复现。macro g 下界 +0.035 > 0 勉强过门。
- 判决 **inconclusive**：Δ、H50 区间跨零；H25 上界 > 0。TurnOnSinkFaucet 一任务 warm 超过 full（.28 vs .18，Δ 区间不含 0），其余三任务 Δ 在 ±.12 内。
- `harmful_on_flat`：PickPlaceCounterToStove Δ = −.45，正式区间 [−.64, −.20]（n10 = 4、n01 = 49）。
- m=2（探索）：warm_t0.5 相对 plain_k2 在 5 任务 Δ 全 ≤ 0（−.10/−.02/−.06/−.08/−.14），GR00T 在两步预算下 warm 没有任何任务受益。

### 3.3 跨 policy 读法

1. 两个 policy 的 cliff 组 macro Δ 都与 0 无法区分，H50 都不为正：无门 top-1 warm start **不是**减步缺口的通用补救。
2. 收益集中在个别任务：π0.5 CloseFridge（plain .08 → warm .30）、GR00T TurnOnSinkFaucet（.04 → .28）。这两个任务的共同点是 plain 极低（近乎零成功）而全步不高，即"减步后策略完全丢失动作模式"的任务；warm 从库轨迹续跑提供了一个可行的粗动作。
3. 损伤集中在 PnP flat 任务（PickPlaceSinkToCounter、PickPlaceCounterToStove，两 policy 一致），这些任务 plain 已 ≥ .94；warm 的失败是库轨迹与当前场景不匹配时被强制续跑（无门、top-1）。相似度分箱（仅 shadow idx 0…9，描述）没有显示高相似度分箱更安全：π0.5 PickPlaceCounterToStove 低/高相似度箱 Δ = −1.0/−0.8，GR00T 同任务 −.4/−.2。**检索分数（cut ≈ .99）本身不区分好坏命中**，这是给四层门控设计的直接负证据：不能用这一族相似度作为 warm start 的准入门。
4. 上述所有量条件于本批固定任务、W13 库、固定 t、无门 top-1；不外推到带门 RIT、其它库或其它 t。

## 4. Q-C：合作者三问

1. **动作分块与执行**：RC 两 policy 每次推理执行 5 步（replan_steps=5）；π0.5 每集决策数 113–135、环境步数 565–673（主判决三臂均值），GR00T 89–100 / 444–501；1 步与全步的每集决策数相近（差异来自失败集跑满上限）。LIBERO：object/goal 每集推理 24–29 次、150/130 步；spatial 21 次、libero_10 60 次（§1 表）。
2. **1 步与 10 步轨迹对齐**：见 §2 表。RC 上 π0.5 r_1 ≈ 3–5、GR00T ≈ 1.2–1.5；LIBERO 上 π0.5 r_1 ≈ .77–.85（1 步偏离**小于**噪声间离散度）、GR00T spatial ≈ 1.25–1.35、libero_10 ≈ 1.56–1.81。LIBERO 上 π0.5 的一步动作与全步的差别不超过全步自身随噪声的变化，闭环阶梯平坦（Q-C.3）与此一致；RC 上 π0.5 一步偏离远超噪声离散度，闭环出现 cliff。但**跨任务**的 r_1/d_1 不预测哪个任务是 cliff（§2）。
3. **分套件阶梯**（π0.5，同 harness，500 集/档）：

| suite | k=1 | k=2 | k=4 | k=10 |
|---|---|---|---|---|
| libero_object | .984 | .988 | .978 | .976 |
| libero_goal | .932 | .956 | .944 | .952 |
| libero_spatial | （旧线 .988/.982） | | | .988 |
| libero_10 | （旧线 .824–.850） | | | .836 |

四个 LIBERO suite 在 π0.5 上都没有减步折损；GR00T 无 object/goal checkpoint，未做。

## 5. 运行、成本、偏差与解释边界

- **时间线**（CDT）：RC π0.5 shadow 13:46–17:41、GR00T shadow 16:15–17:33（h100↔timan108）；LIBERO 四 env shadow 与 Q-C.3 阶梯在 weilandserver↔timan107 并行至 21:45；Q-B π0.5 18:00–01:34（h100 8 进程、MPS）；Q-B GR00T 21:30–01:38（weilandserver ≤7 进程、MPS）。总墙钟约 12 h，四台机（h100 / weilandserver / timan107 / timan108）全程无空转。
- **拓扑偏差（记录，不门控）**：owner 21:15 裁定"不许任何机器干等"后，GR00T Q-B 整体从 h100 改到 weilandserver↔timan107；GR00T **full main 三任务**（SlideDishwasherRack、TurnOnSinkFaucet、OpenCabinet）又因 weilandserver 槽位不足改在 h100↔timan108 跑。分析端 `comparison_notes` 对这三任务记 `worker_islands=2, serving_runtimes=2`；`comparison_identity`（env 契约 + `checkpoint_sha256`）跨机一致，两机 full 的 `config_sha` 只因 manifest 中 checkpoint 绝对路径字符串不同。π0.5 七臂全部同拓扑。
- **重试/续跑**：π0.5 plain_k2 一个 cell 因 timan108 tmux server 误杀而 INCOMPLETE 后按同参数 resume（只派发缺失身份）；GR00T plain_k1 PnP 首次起跑的 run-plan（server 23156/23157）被删除重起，先前 accepted 的 1 集按同 seed 重跑，旧 server 行成为 2 个 stray session（分析端选 accepted launch 的 occurrence）。均不改变样本。
- **MPS**：两台 server 机在 Q-B 全程开启 MPS（owner 裁定），结束后关闭；MPS 只影响吞吐不影响每次决策 NFE。
- **解释边界**：50 集 + 12 区间 Bonferroni 使 macro 功效有限，两 policy 的 inconclusive 一部分来自区间宽度（π0.5 H25 上界 +0.006）；flat 损伤不受此影响（区间远离 −0.10）。历史 g 作为固定点估计进入 Q-A 相关，不含旧 SR 误差与旧硬件差异；GR00T OpenCabinet 缺口未复现说明历史 g 本身有 ±0.2 量级噪声。分箱只用 10 集、仅描述。
- **产物**：server 行 `exp/step_diag/data/server/<teacher>/<arm>/`（rows/arrays/manifest，含 GR00T full 两机 manifest），driver 产物 `data/rc/<teacher>/<arm>/`（GR00T full 合并两台 timan），shadow `data/analysis/shadow_<env>.json`，Q-B `data/analysis/qb_<policy>.json`，阶梯 `data/qc3/<suite>/`，gaps `data/gaps/`；均 gitignored。运行期代码改动 7 处见 `logs/session_handoff.md` §6。
