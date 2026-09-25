# 减步 vs warm start：闭环 shadow 诊断与等 NFE 对照 — 结果报告

> 计划：`logs/step_vs_warmstart_diagnostics_plan.log.md` v3.1（G1/G2 APPROVED，代码 `d8e464d`）。数据 2026-09-20 13:46 CDT 起跑、2026-09-21 01:38 CDT 收官（America/Chicago）。
> 生成表：`step_vs_warmstart_tables.md`（六环境 `shadow_<env>.md` + `qb_pi05.md`/`qb_groot.md` 拼接）；JSON 在 `exp/step_diag/data/analysis/`；parity 证据 `parity_<env>.json`。
> 本文只解释预注册量，不改判据、不换任务组、不换主预算。

## 0. 一句话结论

- **Q-B（两 policy 各自互斥判决）：均为 `inconclusive`，且均触发独立标记 `harmful_on_flat`。** 无门 top-1 forced warm start 在等决策 NFE 下没有补回减步缺口：π0.5 cliff 四任务 macro Δ = −0.035 [−0.145, +0.075]（H50 上界 −0.055 < 0，H25 上界 +0.006 未过 not-supported 门）；GR00T macro Δ = +0.055 [−0.045, +0.155]，H25 区间跨零。两 policy 各有 flat 任务 Δ 上界远低于 −0.10（π0.5 PickPlaceSinkToCounter −0.63、PickPlaceCounterToStove −0.78；GR00T PickPlaceCounterToStove −0.45），即 **warm start 在历史无缺口的任务上造成大幅损伤**。
- **Q-A：两 policy 的 RC 逐决策减步偏离 d_1 与历史阶梯缺口 g 均 `no_conclusion`**（π0.5 ρ = 0.085 [−0.42, 0.63]；GR00T ρ = 0.377 [−0.35, 0.90]，且 g 范围 0.30 只勉强过 0.15 门）。d_1 在同一 policy 内跨任务几乎常量（π0.5 RC 4.8–8.2、GR00T RC 1.7–2.1，LIBERO 更窄），不能作为任务级"能否减步"的门控量。
- **Q-C.3：π0.5 LIBERO-Object / Goal 阶梯平坦**（object k=1/2/4/10 = .984/.988/.978/.976；goal = .932/.956/.944/.952，各 500 集），与 spatial（k=10 锚 .988）一致；libero_10 同 harness k=10 锚 .836，与旧阶梯 k=7 的 .824–.850 同量级，说明旧线 "k=7 0.84 vs RIT 10 步参考 0.92" 的差异是 harness 差异，不是步数。

- **追加（§6，2026-09-21/22）：reset 式 warm start（t 重置、dt = −1/n，起点为缓存快照或最终动作）的收益是 policy 相关的。** π0.5 宏观 13 任务 2 次前向 0.708，比同预算减步 +0.228 [+0.189, +0.268]、比 10 步 full +0.161 [+0.116, +0.205]（500 集、1M 种子段、外部代码审查均支持）；GR00T 宏观 13 任务 1 次前向 0.562，与减步无差（−0.015 [−0.051, +0.020]）、低于 4 步 full（−0.077 [−0.118, −0.035]），2 步时追平 full。两个 policy 上 reset 式都能止住精确续跑在个别平坦任务上的崩塌，起点用最终动作与用快照无差。
- **喂入点与机制（§6.8–6.10，2026-09-22/23）：最佳喂入点是 policy 相关的。** 缓存最终动作喂在「纯噪声往下一格」（π0.5 t=0.9、GR00T t=0.75）：π0.5 13 任务从 0.708 降到 0.545（t=1 最好），GR00T 1 次前向从 0.573 升到 0.611（1 步里最好，与 4 步 full 差距不显著）；GR00T 2 次前向补齐 13 任务后 reset 式与纯减步、full 三者持平。init_probe 机制探针：t=1 喂入时 π0.5 输出保留缓存差异远多于 GR00T（两步输出比值 0.88/0.56 对 0.32/0.09）。
- **GR00T 起点 × 喂入点 × 步数（§6.11，2026-09-23）：GR00T 上把喂入点从 t=1 下移有益，且与步数耦合。** 1 次前向：缓存最终动作喂 0.5 得 0.646（比纯减步 +0.069 [+0.029, +0.109]，与 4 步 full 持平）；2 次前向：缓存快照喂 0.75 得 0.669，是 GR00T 所有 2 次前向做法里最高的（比纯减步 +0.040 [+0.002, +0.077]，比 full +0.031 区间含 0）；最终动作喂 0.5 走 2 步反而显著变差（0.600）。同一喂入点上快照与最终动作无显著差别；快照喂 0.5 补齐后网格完整：1 步时喂 0.5 最好（快照 .637、最终动作 .646，均追平 full），2 步时喂 0.75 最好、喂 0.5 回落到纯减步水平。

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

## 6. 追加：warm start 续跑变体（教授的 dt = −1/remaining 提案）— 2026-09-21 晚

**来由**。Fan Lai 教授读 `run_stage3_from` 后认为 `dt = -1/num_steps` 有误：既然有好的起点，就该把剩余去噪当作一次新的短 run（dt = −1/remaining_steps），并删去把 t 回放到 start_t 的循环。owner 裁定把他的两种字面读法各做一臂，与正式 Q-B 的 `plain_k2` / `warm_t0.2` / `full` 同种子配对比较（L1，无审查）。三臂都从同一个缓存快照 x₀.₂ 出发、都只做 2 次前向，只有网络看到的 (t, dt) 序列不同：

| 臂 | 起点 | t 序列 | dt | 含义 |
|---|---|---|---|---|
| `warm_t0.2`（我们的，Nirvana 式） | x₀.₂ | 0.2, 0.1 | −0.1 | 跳过前 8 步，其余按原 10 步网格 |
| `warmreset_t0.2`（教授读法 A） | x₀.₂ | 1.0, 0.5 | −0.5 | 把缓存当噪声、t 从 1 重启、无 replay |
| `warmshoot_t0.2`（教授读法 B） | x₀.₂ | 0.2, −0.3 | −0.5 | 保留 start_t，步长放大，t 越过 0 |

实现：`exp/step_diag/pi05.py: warm_variant_stage3`（服务模式 `warmreset` / `warmshoot`，同 `warm_t0.2.yaml`、同库、同 judge `always_warm_start start_t=0.2`），执行路径路由到变体，shadow 括号内仍走精确 resume，步数计数经 `denoise_step` 不变；驱动侧 `run_diag.py` 白名单放行两臂。准入与 §1 同一套（每决策 hit_type=WARM_START、start_t=0.2、executed_steps=2、n_stage3_calls=1、manifest 身份），分析 `exp/step_diag/analysis/warm_variants.py`（配对身份 = 任务/init_idx/env_seed/lane/pin/layout/style，bootstrap 20000，seed 20260919；描述性，不发 verdict）。产物 `data/analysis/warm_variants_pi05.json`、`analysis/warm_variants_pi05.md`。

### 6.1 第一轮：50 集/任务（h100 4 server ↔ timan108 4 cell，22:24–23:23 CDT）

全部 cell 准入（`admissible=ok`，steps=2.0，miss=0）。

| 臂 | CloseFridge（悬崖） | PickPlaceCounterToStove（平坦） |
|---|---|---|
| full（10 步） | 0.62 | 0.84 |
| plain_k2 | 0.08 | 0.94 |
| warm_t0.2 | 0.30 | 0.16 |
| **warmreset_t0.2** | **0.52** | **0.82** |
| warmshoot_t0.2 | 0.00 | 0.00 |

配对差（n=50，95% bootstrap，n10/n01 = 变体成功·参照失败 / 反之）：

| 配对 | CloseFridge | PickPlaceCounterToStove |
|---|---|---|
| warmreset − warm_t0.2 | +0.22 [+0.04, +0.40]（18/7） | **+0.66 [+0.50, +0.80]**（35/2） |
| warmreset − plain_k2 | **+0.44 [+0.30, +0.58]**（23/1） | −0.12 [−0.24, +0.00]（2/8） |
| warmreset − full | −0.10 [−0.30, +0.10] | −0.02 [−0.16, +0.12] |
| warmshoot − plain_k2 | −0.08 [−0.16, −0.02] | −0.94 [−1.00, −0.86] |
| warmshoot − warm_t0.2 | −0.30 [−0.42, −0.18] | −0.16 [−0.26, −0.06] |

**读法**：
- 教授的读法 A（reset）**不是**"把缓存抹掉退化成 plain_k2"：若真抹掉，CloseFridge 应落到 plain_k2 的 0.08，实测 0.52。它同时消掉了我们 warm start 在平坦任务上的灾难性伤害（0.16 → 0.82，与 full 0.84 无差）。两任务上 reset 都与 full 统计不可分。我事先给教授的预测（reset ≈ plain）在 50 集数据上**被证伪**。
- 读法 B（overshoot）全灭：t = −0.3 在训练分布外，网络输出无意义。
- 机制假说（待 500 集与逐决策数据验证）：在 t=1 网络认定输入是纯噪声，v̂ ≈ x − x̂₀(obs)，两步 dt=−0.5 后的结果约为 ¼·x₀.₂ + ¾·x̂₀(当前观测)，即"当前观测的粗网格预测 + 少量缓存内容"——一个由当前观测纠错的**软** warm start；我们的精确 resume 则完全信任缓存位置，好命中时更贴近 full 轨迹，错命中时无处纠错。这与 §3 的 harmful_on_flat 诊断一致：平坦任务里错误检索占多，软版本把它们修回来了。
- 对四层设计的含义：若 500 集复现，"warm start 层"的实现应改为 reset 式（或介于两者之间的 t 起点/步长组合，即未跑的 row 4），且它在悬崖任务上仍保留 warm start 相对 plain 的收益。

### 6.2 第二轮：500 集/任务（进行中，experiment `sdiag_var500`）

owner 批准后于 23:00 CDT 起跑：四臂 `plain_k2 / warm_t0.2 / warmreset_t0.2 / warmshoot_t0.2` × 两任务 × 500 集（种子 2,000,000+idx，前 50/100 身份与 Q-B 同），13–14 个单连接 server（h100 8、weilandserver 6，MPS）↔ timan108/timan107 worker，独立输出根 `data/rc500` / `data/server500`。跑完于 09-22 12:06 CDT（8 个 cell；h100 07:49 事故中断 3 个 cell 后同参数 resume，样本不变）。**准入**：四臂 × 两任务全部 `complete && equal_nfe`（每臂 1000 arrays sha 0 bad，每决策 2 步、WARM_START、单次 stage-3）。产物 `data/analysis/warm_variants_pi05_500{,_idx50}.json`、`analysis/warm_variants_pi05_500{,_idx50}.md`。

**结果（n=500/任务；括号内为 idx 50–499 的 450 集敏感性版）**：

| 臂 | CloseFridge | PickPlaceCounterToStove |
|---|---|---|
| plain_k2 | 0.06 (0.06) | 0.95 (0.95) |
| warm_t0.2 | 0.29 (0.29) | 0.16 (0.17) |
| warmreset_t0.2 | **0.61** (0.62) | **0.79** (0.78) |
| warmshoot_t0.2 | 0.00 | 0.00 |

| 配对差（95% bootstrap） | CloseFridge | PickPlaceCounterToStove |
|---|---|---|
| warmreset − warm_t0.2 | +0.32 [+0.26, +0.38]（213/52） | +0.63 [+0.58, +0.67]（320/6） |
| warmreset − plain_k2 | +0.54 [+0.50, +0.59]（279/7） | −0.16 [−0.20, −0.12]（18/97） |
| warm_t0.2 − plain_k2 | +0.22 [+0.18, +0.27] | −0.79 [−0.82, −0.75] |
| warmshoot − plain_k2 | −0.06 [−0.09, −0.04] | −0.95 [−0.97, −0.93] |
| 两任务均值 warmreset − plain_k2 | +0.19 [+0.16, +0.22] | |
| 两任务均值 warmreset − warm_t0.2 | +0.48 [+0.44, +0.51] | |

**对预先写死判定的打勾**：
1. **H1 成立**：warmreset − warm_t0.2 两任务区间都不含 0 且为正（CloseFridge +0.32、PnP +0.63）。教授的 reset 式续跑优于我们的 Nirvana 式精确 resume。
2. **H2**：CloseFridge 成立（+0.54，区间不含 0）；PickPlaceCounterToStove 上界 −0.12 < 0 → 记「**小代价**」：reset 式 warm start 在平坦任务上比同预算减步低 0.16，50 集时区间碰 0 的边缘性劣势在 500 集上确认为真实但有限的代价。
3. **H3**：full 未跑 500 集，只能与 rc/ 的 50 集 full 锚点配对（§6.1：−0.10 [−0.30, +0.10]、−0.02 [−0.16, +0.12]），在 50 集精度上与 full 不可分，无法在 500 集精度上判定。
4. warmshoot 两任务 500 集均为 0.00，记录。
5. 敏感性：450 集新环境版本与全量版本所有点估计相差 ≤ 0.01、结论一致 → 今晚 50 集的选择效应可以排除。
6. 准入全部通过。

**与 50 集轮的一致性**：四个 SR 点估计与 §6.1 相差 ≤ 0.09（最大是 warmreset CloseFridge 0.52 → 0.61）；1M 段（§6.4）也一致。这两个任务上的结论可以视为定论：reset 式在悬崖任务上大幅优于减步，在平坦任务上小幅劣于减步，两者都远优于精确续跑。

**预先写死的判定（2026-09-21 23:40 CDT，数据落地前）**。500 集轮不是预注册实验（变体在看到 Q-B 后设计、500 集在看到 50 集后决定，任务是 Q-B 里最悬崖/最平坦的两个），因此结论只对这两个任务、π0.5、t=0.2 成立，不外推。为了不留事后解释空间，判定标准在此固定：

1. **H1（主）**：`warmreset − warm_t0.2` 在两任务上的 95% 配对区间都不含 0 且点估计为正 → 教授的 reset 式续跑优于 Nirvana 式精确 resume。
2. **H2**：`warmreset − plain_k2` 在 CloseFridge 区间不含 0 且为正；在 PickPlaceCounterToStove 区间含 0 记「无代价」、上界 < 0 记「小代价」（并报点估计）。
3. **H3**：`warmreset − full` 两任务区间都含 0 → 与 full 统计不可分；任一任务下界 > 0 或上界 < 0 则如实报告。
4. `warmshoot` 只作记录，不设假设。
5. 主分析用 idx 0–499 全部 500 集；敏感性分析只用 idx 50–499（与今晚 50 集轮不重叠的 450 个新环境）。两版结论不一致时以 450 集版为准。
6. 准入不变：任一 cell 不满足 `complete && equal_nfe` 则该任务不判，报未准入原因。

### 6.3 补充 A：变体阶梯（1 / 2 / 3 步）与「起点」消融 resetfinal — 2026-09-22 05:00–07:40 CDT

**设计**。三个步数档 n = 1 / 2 / 3（由 `start_t` = 0.1 / 0.2 / 0.3 定，n = ⌊t·K+½⌋），每档四种同预算做法配对：`plain_k{n}`（噪声起、粗网格）、`warm_t{t}`（Nirvana 精确续跑）、`warmreset_t{t}`（缓存快照 x_t 起、t 重置为 1、dt = −1/n）、**`resetfinal_t{t}`（新增消融：同 warmreset 的循环，但起点换成缓存的最终动作 chunk（t = 0），`start_t` 只定步数）**；`warmshoot_t0.1`（1 步、dt = −1、不越界）作记录，`warmshoot_t0.3` 未跑（两次越过 0，与 t0.2 同理）。每臂每任务 50 集，seed 2M，与 Q-B 逐集配对；全部 cell 准入（6 个新臂 × 100 arrays，sha 0 bad）。产物 `data/analysis/warm_variants_pi05_t{0.1,0.2,0.3}.json`、`analysis/warm_variants_pi05_t*.md`。

成功率（CloseFridge / PickPlaceCounterToStove；full = 0.62 / 0.84）：

| n（前向次数） | plain_k | warm_t（精确续跑） | warmreset（快照起、t 重置） | resetfinal（最终动作起、t 重置） | warmshoot |
|---|---|---|---|---|---|
| 1（t=0.1） | 0.02 / 0.98 | 0.22 / 0.12 | 0.22 / 1.00 | 0.24 / 1.00 | 0.80 / 0.08 |
| 2（t=0.2） | 0.08 / 0.94 | 0.30 / 0.16 | 0.52 / 0.82 | **0.76** / 0.72 | 0.00 / 0.00 |
| 3（t=0.3） | 0.14 / 0.94 | 0.30 / 0.16 | 0.66 / 0.64 | **0.84** / 0.42 | – |

关键配对差（n=50，95% bootstrap）：

| 配对 | n=1 | n=2 | n=3 |
|---|---|---|---|
| resetfinal − warmreset，CloseFridge | +0.02 [−0.10, +0.14] | **+0.24 [+0.10, +0.38]** | +0.18 [0.00, +0.36] |
| resetfinal − warmreset，PnP | 0.00 [0, 0] | −0.10 [−0.22, +0.02] | **−0.22 [−0.36, −0.08]** |
| resetfinal − warmreset，macro(2 任务) | +0.01 [−0.05, +0.07] | +0.07 [−0.03, +0.16] | −0.02 [−0.13, +0.09] |
| warmreset − warm_t，CloseFridge | 0.00 [−0.18, +0.18] | +0.22 [+0.04, +0.40] | +0.36 [+0.20, +0.52] |
| warmreset − warm_t，PnP | +0.88 [+0.78, +0.96] | +0.66 [+0.50, +0.80] | +0.48 [+0.34, +0.62] |
| warmreset − plain_k，CloseFridge | +0.20 [+0.08, +0.32] | +0.44 [+0.30, +0.58] | +0.52 [+0.36, +0.66] |
| warmreset − plain_k，PnP | +0.02 [0.00, +0.06] | −0.12 [−0.24, 0.00] | −0.30 [−0.44, −0.16] |
| resetfinal − full，CloseFridge | −0.38 [−0.56, −0.18] | +0.14 [−0.04, +0.32] | **+0.22 [+0.04, +0.40]** |

**读法**：
1. **两种 reset 式在每个步数档、两个任务上都优于精确续跑 warm_t**（8 个区间全为正且不含 0）。精确续跑在 PnP 上无论几步都是 0.12–0.16，说明它的问题不在步数而在「完全信任缓存位置」。
2. **btw 的判断（快照的 t 与结果无关）只在 n=1 成立**：1 步时 resetfinal 与 warmreset 逐集几乎相同（PnP 完全一致 0/0）。n=2、3 时起点有影响且方向随任务：悬崖任务 CloseFridge 从最终动作起更好（+0.24、+0.18），平坦任务 PnP 从最终动作起更差（−0.10、−0.22）。两任务均值上两者不可分。机制上一致的解释：最终动作比快照离当前观测的正确模态更远（它是教师在别的 episode 里的完整动作），在悬崖任务上这段"距离"提供了 plain 缺的动量，在平坦任务上则把本来正确的预测拉偏；步数越多，起点的影响越大。
3. **对减步的对比按任务分裂**：CloseFridge 上 reset 式随步数单调上升，3 步 resetfinal 0.84 已显著高于 10 步 full 0.62（+0.22 [+0.04, +0.40]）；PnP 上 reset 式随步数单调下降（1.00 → 0.82/0.72 → 0.64/0.42），而 plain_k 三档都是 0.94–0.98。即在平坦任务上 reset 式 warm start 的每一步都在把缓存内容混进来、越混越坏，1 步时与 plain 持平（+0.02）。
4. warmshoot 1 步（x − v，相当于精确续跑步长的 10 倍）在 CloseFridge 0.80、PnP 0.08，与 warm_t0.1 的 PnP 一样坏但 CloseFridge 异常好；2 步全灭。只作记录。
5. 对四层设计的含义（待 500 集与宏观轮确认）：cache 层的「warm start」应实现为 reset 式，且**步数预算要按任务选**——悬崖任务给 3 步且可从最终动作起，平坦任务给 1 步（此时起点无关、与 plain 持平）；库里若只为 reset 式服务，中间态快照可以不存（n=1 时起点无关，n≥2 时最终动作在悬崖任务上反而更好）。

### 6.4 补充 B：种子段交叉验证（seed 1,000,000+idx）— 2026-09-22 07:00–09:19 CDT

**目的**。本线全部数据在 2,000,000 段（plan v3.1 §52：避免复用选组用过的环境）；历史阶梯（nfe_baseline、ws_search）在 1,000,000 段。为排除「种子段效应」，把四臂 `plain_k2 / warm_t0.2 / warmreset_t0.2 / warmshoot_t0.2` × 两任务 × 50 集在 **seed 1,000,000+idx（idx 0…49）** 重跑一遍（experiment `sdiag_xseed1m`，独立 out root `rc_x1m` / `server_x1m`，driver 仅在该 exp id 下放行该种子）。全部 cell 准入（4 臂 × 100 arrays，sha 0 bad）；h100 事故（§6.7）期间 3 个 cell 被中断后同参数 resume，样本不变。产物 `data/analysis/warm_variants_pi05_x1m.json`、`analysis/warm_variants_pi05_x1m.md`。

| 臂 | CloseFridge：1M 段 / 2M 段 50 集 / 2M 段 500 集 | PnP：1M / 2M-50 / 2M-500 |
|---|---|---|
| plain_k2 | 0.04 / 0.08 / 0.064 | 0.92 / 0.94 / 0.948 |
| warm_t0.2 | 0.32 / 0.30 / （§6.2） | 0.14 / 0.16 / 0.162 |
| warmreset_t0.2 | 0.66 / 0.52 / 0.608 | 0.80 / 0.82 / 0.79 |
| warmshoot_t0.2 | 0.00 / 0.00 / （§6.2） | 0.00 / 0.00 / 0.0 |

1M 段配对差（n=50）：warmreset − warm_t0.2：CloseFridge +0.34 [+0.16, +0.52]、PnP +0.66 [+0.52, +0.78]；warmreset − plain_k2：CloseFridge +0.62 [+0.48, +0.76]、PnP −0.12 [−0.26, 0.00]；warm_t0.2 − plain_k2：+0.28 / −0.78；两任务均值 warmreset − plain_k2 +0.25 [+0.15, +0.35]。

**读法**：与 2M 段的 50 集轮逐条同号、同量级（warmreset − plain_k2 在 PnP 上两段都是 −0.12，上界都恰在 0），与 500 集轮的点估计也一致。没有种子段效应；本线沿用 2M 段的选择不影响结论。

### 6.5 宏观：13 任务全体（π0.5）— 2026-09-22 08:40–15:18 CDT

**设计**。experiment `sdiag_macro13`（out root `rc_macro13` / `server_macro13`，seed 2M，每臂每任务 50 集）：`warmreset_t0.2`、`warmshoot_t0.2` 跑全部 13 个 roster 任务；`full`、`plain_k2`、`warm_t0.2` 只补 Q-B 没跑的 6 个任务（CloseBlenderLid、CoffeeSetupMug、OpenStandMixerHead、SlideDishwasherRack、TurnOnSinkFaucet、PickPlaceCounterToCabinet），其余 7 个任务复用 Q-B 正式数据（`rc/`，50–100 集）；owner 10:25 追加 `resetfinal_t0.2` 11 任务（另 2 任务复用 §6.3）与调整组 `resetfinal_t0.1 / t0.3` 各 6 任务。分析端 `warm_variants.py --arms-root rc,rc_macro13 --server-rows server,server_macro13`（跨 root 合并，macro Δ = 各任务配对差均值，episode 在任务内分层 bootstrap）。**准入**：13 任务 × 6 臂全部 `complete && equal_nfe`（server 行 sha 全对；full 目录含 41 条作废 launch 的 stray 行，见 §6.7 事故记录，不门控）。产物 `data/analysis/warm_variants_pi05_macro13.json`、`analysis/warm_variants_pi05_macro13.md`、逐任务表 `analysis/pi05_macro13_success_rates.{md,csv}`、图 `analysis/figures/macro13_four_arms.png`（2026-09-23 按 owner 要求重画：6 臂 = full / plain_k2 / warm_t0.2 / warmreset / resetfinal / midfinal，去掉近乎全 0 的 warmshoot，加 13 任务均值组；图例只把我们的精确续跑叫 warm start，其余叫 warm reset，参数记法 T = 从缓存取出的那份动作的 t（0 = 最终动作）、N = 步数、t = 实际传给模型的 t；数据取自上述 JSON，CloseFridge / PnP 的 warmreset 为同 50 集两轮平均，标 (2 runs)）。

**13 任务均值**（每格为该臂 13 个任务成功率的平均）：

| 臂 | 前向次数/决策 | macro SR |
|---|---|---|
| full | 10 | 0.548 |
| plain_k2 | 2 | 0.481 |
| warm_t0.2（Nirvana 精确续跑） | 2 | 0.277 |
| **warmreset_t0.2** | 2 | **0.708** |
| **resetfinal_t0.2** | 2 | **0.708** |
| warmshoot_t0.2 | 2 | 0.005 |

macro 配对差（13 任务，95% 分层 bootstrap）：warmreset − plain_k2 **+0.228 [+0.189, +0.268]**；warmreset − warm_t0.2 **+0.430 [+0.388, +0.472]**；warmreset − full **+0.161 [+0.116, +0.205]**；resetfinal − warmreset −0.001 [−0.035, +0.033]；warm_t0.2 − plain_k2 −0.204 [−0.242, −0.165]。

**逐任务**（warmreset − plain_k2 / warmreset − full）：13 个任务里 warmreset 对 plain 的配对差 11 个为正且区间不含 0（+0.14 ～ +0.52），1 个持平（PickPlaceSinkToCounter 双 1.00），1 个为负（PickPlaceCounterToStove −0.12 [−0.24, 0.00]）；对 full 的配对差 8 个显著为正（最大 OpenStandMixerHead +0.56、SlideDishwasherRack +0.32、OpenDrawer +0.26、OpenCabinet +0.24），4 个不可分，1 个为负但不显著（CloseFridge −0.12 [−0.30, +0.06]）。resetfinal 与 warmreset 逐任务只在 CloseFridge 分出高下（+0.26 [+0.10, +0.42]），其余 12 个任务区间都含 0。

**调整组（8 任务：6 个补跑 + CloseFridge/PnP 复用 §6.3）**：1 步 `resetfinal_t0.1` macro 0.435，对 plain_k1 +0.147 [+0.103, +0.193]（6 任务），对 full −0.130 [−0.183, −0.078]；3 步 `resetfinal_t0.3` macro 0.595，对 plain_k3 +0.097 [+0.037, +0.157]，对 full +0.030 [−0.025, +0.085]（与 10 步不可分）。即 1 步已优于同预算减步但不及 full，2 步开始超过 full，3 步与 full 不可分（悬崖任务继续升、平坦任务继续降的两种趋势在均值上抵消）。

**读法**：
1. 两个极端任务上的结论在 13 任务上成立且更强：reset 式 warm start 以 2 次前向达到 0.708，比 10 次前向的 full 高 0.16、比同预算减步高 0.23；我们的精确续跑 0.277，比减步还低 0.20。
2. 「平坦任务上 reset 式小输减步」在宏观上只剩 PickPlaceCounterToStove 一个任务（−0.12），另两个平坦任务持平或反超（PickPlaceSinkToCounter 1.00/1.00，OpenDrawer +0.14）；harmful_on_flat 这一条对 reset 式基本消失。
3. resetfinal 与 warmreset 在 13 任务上完全打平：中间态快照对 reset 式没有系统性价值，warm start 层可以只依赖库中的最终动作。
4. 对四层设计：当前证据下「warm start 层」应实现为 reset 式（2 步、从最终动作或快照起均可）；剩余的门控问题缩小为「个别平坦任务上是否改用单纯减步」以及 1 步 vs 2 步的预算选择。

### 6.5.1 外部代码审查与修正 — 2026-09-22 16:16–17:10 CDT

教授怀疑「warmreset 比 full 高约 16 pp」来自代码缺陷，owner 请外部审查者（Codex，只读）独立审查 `exp/step_diag`。审查者从原始 journal 重算得 full 356/650 = 0.548、warmreset 460/650 = 0.708（每任务种子 2,000,000–2,000,049），**未发现 full 被削弱、成功判定偏袒某臂或评测答案泄漏**：full 强制 10 步且与生产策略同一 `_stage3_action_expert`；各臂同一 episode runner、horizon 与 `info["success"]`；检索只用当前观测，库来自 base_seed=0 的离线采集（与评测种子段分离），`NeverWritePolicy` 禁止回写。审查者提出三条统计口径问题（P2），已全部修正于 `analysis/warm_variants.py`：

1. **重复评测的合并**。CloseFridge / PickPlaceCounterToStove 的 warmreset、warmshoot 在 §6.3 与宏观轮各跑一次；旧实现 SR 用两轮、配对差按环境身份建字典时后读入的一轮覆盖前一轮（结果依赖目录顺序）。现改为同一身份的多轮取均值后再配对，SR 用同一口径；交换 `--arms-root` 顺序输出逐字节一致。
2. **跨臂一致性门**。变体分析原先只做逐臂准入。现按正式分析的 `comparison_problems` 逐任务核对各有数据臂的模型 + 环境契约身份，不一致的任务不进 macro；模拟器环境与服务运行时（主机、源码摘要）仅记录。另补证：大权重文件在 manifest 中只按大小记身份，已在 h100 与 weilandserver 上对 checkpoint 做全量内容 sha256，π0.5（`59e60ab8…`）与 GR00T（`2dd52c9c…`）两机逐字节一致，文件修改时间早于实验。
3. **退化区间**。配对差全相同（如 PickPlaceSinkToCounter 各臂同为 1.00）时 bootstrap 塌成 [0, 0]；现改报 点值 ± Wilson 95% 上界 z²/(n+z²)（n = 50 时 ±0.07），表中以 † 标注。

**修正后的数字**：宏观 13 任务 warmreset − full +0.161 [+0.116, +0.205]（原 +0.160 [+0.115, +0.203]），warmreset − plain_k2 +0.228 [+0.189, +0.268]，resetfinal − warmreset −0.001 [−0.035, +0.033]；CloseFridge 的 warmreset − full 由 −0.12 变为 −0.11。其余全部变体表（t0.1/0.2/0.3、500、500_idx50、x1m、macro8_t0.1/0.3）重跑后数值不变，仅新增 † 标注。

**源码版本补查**。审查者指出 full 与 warmreset 的服务端源码摘要不同（摘要覆盖 `exp/step_diag` 等整个目录，任何文件改动都会变化）。按 full 所用源码分组：与 warmreset **同一源码摘要**（`e795fc57`）的 3 个任务（CloseBlenderLid、CoffeeSetupMug、OpenStandMixerHead）上 warmreset − full = +0.20 / +0.02 / +0.56，均值 +0.26；另两组（Q-B 旧版本 7 任务 +0.11、`ba5e05be` 3 任务 +0.18）同号。提升不来自代码版本差。

**读法**：审查未发现能解释主要提升的代码缺陷，三条统计修正不改变任何结论。仍在解释边界内的一点：warm 类臂额外拥有成功经验库，这是方法设定而非 bug——「warmreset 高于 full」的含义是「库中成功动作 + 2 步去噪」优于「无库 10 步去噪」，而不是求解器本身更优。

### 6.6 GR00T 对称实验 — 2026-09-22 10:40–18:50 CDT

**设计**（owner 10:40：与 π0.5 对称，不跑 500 集、不跑 warmshoot）。GR00T N1.5 升序调度 K=4，`warmreset_t{t}` / `resetfinal_t{t}` 的步数 n = K − snapshot_index(t)：t=0.75 → 1 步、t=0.5 → 2 步；实现 `exp/step_diag/groot.py: groot_warm_variant_stage3`（`denoise_loop(noise=起点, num_steps=n, start_index=0)`，起点为缓存快照或缓存最终动作）。三部分，均 seed 2M 除 (b)，每臂每任务 50 集：
- (a) 阶梯：`warmreset / resetfinal × t0.75 / t0.5` × TurnOnSinkFaucet（悬崖）、PickPlaceCounterToStove（平坦），与 Q-B 正式臂逐集配对（exp `sdiag_v1`）。
- (b) 1M 种子段复查：`plain_k1 / warm_t0.75 / warmreset_t0.75 / resetfinal_t0.75` × 同两任务，seed 1,000,000+idx（exp `sdiag_xseed1m`）。
- (c) 宏观 13 任务（exp `sdiag_macro13`）：`warmreset_t0.75`、`resetfinal_t0.75` 跑全部 13 任务；`full / plain_k1 / warm_t0.75` 补 Q-B 以外 8 任务；2 步调整组 `warmreset_t0.5 / resetfinal_t0.5` 6 任务（另 2 任务复用 (a)）。
- 拓扑：h100 ≤ 10 个 GR00T server ↔ timan108，weilandserver ≤ 7 个 ↔ timan107（单连接、MPS）。**准入**：全部 cell `complete && equal_nfe`，逐任务跨臂模型/环境身份一致（§6.5.1 新门），20 个 server 目录 arrays sha 全对（0 bad）。产物 `data/analysis/warm_variants_groot_{t0.75,t0.5,x1m,macro13,macro_t0.5}.json`、`analysis/warm_variants_groot_*.md`、图 `analysis/figures/groot_macro13_five_arms.png`（2026-09-23 按 owner 要求重画为上下两面板：N=1 九臂、N=2 八臂，含 §6.8–6.11 全部 13 任务臂与 13 任务均值组；t 一律换成 π0.5 记法（1 = 噪声、0 = 干净，GR00T 原生 = 1 − t），记法 T / N / t 同 π0.5 图；N=2 的 warm start（ours）用 midreset50_t0.5（T=0.5 快照在 t=0.5 喂入，模型输入与精确续跑相同；warm_t0.5 只跑了 5 任务，逐任务差 ≤0.02）；只有我们的叫 warm start，其余叫 warm reset）。

**(a) 阶梯**（配对差，95% bootstrap）：

| 任务 | full | plain | warm（精确续跑） | warmreset | resetfinal | warmreset − full | warmreset − plain | warmreset − warm |
|---|---|---|---|---|---|---|---|---|
| TurnOnSinkFaucet，1 步 | .18 | .04 | .28 | .24 | .32 | +.06 [−.06, +.18] | **+.20 [+.08, +.32]** | −.04 [−.20, +.12] |
| PickPlaceCounterToStove，1 步 | .82 | .94 | .49 | .94 | .94 | +.12 [.00, +.24] | −.02 [−.08, +.04] | **+.52 [+.36, +.68]** |
| TurnOnSinkFaucet，2 步 | .18 | .36 | .30 | .42 | .38 | **+.24 [+.08, +.40]** | +.06 [−.12, +.22] | +.12 [−.08, +.30] |
| PickPlaceCounterToStove，2 步 | .82 | .96 | .82 | .88 | .86 | +.06 [−.06, +.18] | **−.08 [−.16, −.02]** | +.06 [−.06, +.18] |

resetfinal − warmreset：+.08 / .00 / −.04 / −.02，区间全含 0。

**(b) 1M 段复查**（1 步）：TurnOnSinkFaucet plain .08 / warm .34 / warmreset .14 / resetfinal .20；PickPlaceCounterToStove .90 / .52 / .92 / .86。平坦任务上「reset 式止住精确续跑的崩塌」复现（warmreset − warm +.40 [+.26, +.54]）；悬崖任务上 reset 式**不如**精确续跑（warmreset − warm −.20 [−.36, −.04]），与 2M 段（−.04，含 0）方向一致但更强。

**(c) 宏观 13 任务**（1 步，每格为 13 任务均值）：

| 臂 | 前向次数/决策 | macro SR |
|---|---|---|
| full | 4 | **0.638** |
| plain_k1 | 1 | 0.575 |
| warm_t0.75（精确续跑） | 1 | 0.555 |
| warmreset_t0.75 | 1 | 0.562 |
| resetfinal_t0.75 | 1 | 0.573 |

macro 配对差（13 任务，分层 bootstrap）：warmreset − full **−0.077 [−0.118, −0.035]**；warmreset − plain_k1 −0.015 [−0.051, +0.020]；warmreset − warm_t0.75 +0.012 [−0.031, +0.055]；resetfinal − full −0.065 [−0.105, −0.025]；resetfinal − plain_k1 −0.004 [−0.039, +0.032]；resetfinal − warmreset +0.012 [−0.021, +0.043]；warm_t0.75 − full −0.089 [−0.134, −0.045]。逐任务：三种 1 步 warm 做法都在 PickPlaceDrawerToCounter（full .66 → .36–.48）、CloseBlenderLid（.80 → .56–.76）、PickPlaceCounterToCabinet（.62 → .32–.48）上明显低于 full；reset 式高于 full 的只有 PickPlaceCounterToStove（.94 vs .82）与 PickPlaceSinkToCounter（.90–.96 vs .78）、TurnOnSinkFaucet（.22–.25 vs .18）。

**2 步调整组**（8 任务）：warmreset_t0.5 macro 0.650、resetfinal_t0.5 0.657、full 0.643；warmreset − full +0.007 [−0.040, +0.055]，resetfinal − full +0.015 [−0.033, +0.065]；对 plain_k2（仅 Q-B 5 任务有）−0.024 [−0.080, +0.032]。

**读法**：
1. **π0.5 上的主结果在 GR00T 上不复现。** GR00T 1 步时三种 warm 做法与单纯减步在宏观上无法区分（差值都在 ±0.03 内），且都显著低于 4 步 full（约 −0.08）；2 步时 reset 式追平 full，但也只是与 plain_k2 持平。π0.5 上「reset 式以 2 次前向胜过 10 步 full 16 pp」的现象在 GR00T 上没有对应物。
2. **稳定复现的只有「止损」**：精确续跑在个别平坦任务（PickPlaceCounterToStove）上崩塌，reset 式把它拉回到减步水平，两个种子段都成立。宏观上 warm_t0.75 与 reset 式几乎持平（+0.012），说明这种崩塌在 GR00T 上是局部而非普遍的。
3. **resetfinal ≈ warmreset** 在 GR00T 上同样成立（宏观 +0.012，区间含 0）：起点用缓存最终动作还是中间快照无关紧要，与 π0.5 一致。
4. **可能的解释（未检验）**：GR00T 的 full 只有 4 步、1 步减步本已接近 full（0.575 vs 0.638），留给 warm start 补回的缺口小；π0.5 的 10 步 → 2 步缺口更大、且库中成功动作对 π0.5 的帮助远超对 GR00T。两者 checkpoint、库规模、动作头都不同，本实验不能区分这些因素。
5. 对四层设计：warm start 层采用 reset 式在两个 policy 上都**不比精确续跑差、在个别任务上好得多**，可作为默认实现；但它相对单纯减步的收益是 policy 相关的（π0.5 大、GR00T 约为零），门控不能假设 warm start 必然优于减步。

### 6.7 运行记录与事故

- **h100 进程集体消失（09-22 07:49 CDT）**：exouser 的全部进程（tmux server、MPS、8 个 π0.5 server）同时消失，原因未定（紧随一次 `stop_servers.sh` 与 tether exec 双执行的 launch 之后）。用原端口原配置重起（config sha 确定性相同），受影响 cell 以同参数 ep 级 resume，样本集合不变。
- **π0.5 full main lane 重排**：server 列表变更触发 run-plan mismatch，作废的 launch/journal 移出后重跑；本地曾被部分拉取带回一个作废 launch 文件，分析前已删除；server 端 `server_macro13/pi05/full` 留有 41 条作废 run 的 stray 行（不门控）。
- **GR00T server 的 openpi-client 导入失败**：GR00T venv 的 editable openpi-client 指向已归档旧树，`ops/serve_groot.sh` 的 PYTHONPATH 加入 `packages/openpi-client/src` 后解决。
- **GR00T 补齐 server OOM（09-22 20:55）**：本机两个新 server 加载时 CUDA OOM，当时占满显卡的两个未知进程是 Codex 的 init_probe server；按 PID 清理等待循环与锁后重起。
- **timan107 tmux server 被结束（09-23 00:01）**：midfinal 两个 PnP 格先连败于 `1011 keepalive ping timeout`，随后我们的 tmux server 结束（同账号另一项目 robotwin/X-WAM 当时在同机活动）；两格 INCOMPLETE（GR00T 86/200、π0.5 58/200，journal 无重复），`ops/run_rc_cell.sh` 加 `SD_TMUX_SOCKET`（私有 tmux server）后同参数续跑完成；server 端各留 2 条被重跑覆盖 arrays 的 stray finalize（每个 uid 都有一条有效的，不门控）。
- **外部代码审查**（§6.5.1）：未发现能解释 π0.5 主要提升的缺陷，三条统计修正已落地并重跑全部表。
- 收尾：全部 server 停止，h100 与 weilandserver 的 MPS 关闭、GPU 0 MiB；h100 临时打包文件已删。

### 6.8 喂入点消融 midfinal：缓存最终动作喂在「纯噪声往下一格」— 2026-09-22 19:55 – 09-23 01:25 CDT

**设计**（owner 2026-09-22 晚）。与 `resetfinal` 相同：起点是缓存最终动作、直接喂、不加噪；唯一改动是喂入的 flow time 从 t=1（纯噪声）下移到**完整调度的下一格**：π0.5（K=10）t=0.9，GR00T（K=4）t=0.75（统一用 π0.5 记法，1=噪声、0=干净；GR00T 内部为 0.25）。然后 n 步走到 0，dt = −t_entry/n；臂名里的 t 只定步数预算（π0.5 `midfinal_t0.1/t0.2` = 1/2 步，GR00T `midfinal_t0.75/t0.5` = 1/2 步）。实现 `pi05.py: warm_variant_stage3(variant="mid_final")`、`groot.py: mid_denoise_loop`（上游循环在任意入口网格上的复刻，入口 0 时与 `denoise_loop` 逐位一致，测试钉住），常量 `envs.MID_ENTRY_T`。

**(a) 小阶梯**（每格 50 集，exp `sdiag_v1`，与各臂逐集配对）：

| 模型 / 任务 | 预算 | full | 纯减步 | resetfinal（t=1） | midfinal | midfinal − resetfinal |
|---|---|---|---|---|---|---|
| π0.5 CloseFridge | 1 | .62 | .02 | .24 | .54 | **+.30 [+.12, +.48]** |
| π0.5 CloseFridge | 2 | .62 | .08 | .76 | .58 | −.18 [−.36, .00] |
| π0.5 PickPlaceCounterToStove | 1 | .84 | .98 | 1.00 | .42 | **−.58 [−.72, −.44]** |
| π0.5 PickPlaceCounterToStove | 2 | .84 | .94 | .72 | .32 | **−.40 [−.56, −.24]** |
| GR00T TurnOnSinkFaucet | 1 | .18 | .04 | .32 | .32 | .00 [−.18, +.18] |
| GR00T TurnOnSinkFaucet | 2 | .18 | .36 | .38 | .50 | +.12 [−.06, +.30] |
| GR00T PickPlaceCounterToStove | 1 | .82 | .94 | .94 | .88 | −.06 [−.14, +.02] |
| GR00T PickPlaceCounterToStove | 2 | .82 | .96 | .86 | .86 | .00 [−.12, +.12] |

**(b) 宏观 13 任务**（exp `sdiag_macro13`；每模型只跑主预算：π0.5 2 步 `midfinal_t0.2`、GR00T 1 步 `midfinal_t0.75`，各补 11 任务 × 50，另 2 任务复用 (a)）：

| 13 任务 macro SR | full | 纯减步 | 精确续跑 | warmreset | resetfinal | **midfinal** |
|---|---|---|---|---|---|---|
| π0.5（2 次前向） | .548 | .481 | .277 | .708 | .708 | **.545** |
| GR00T（1 次前向） | .638 | .575 | .555 | .562 | .573 | **.611** |

配对差（分层 bootstrap）：π0.5 midfinal − resetfinal **−0.163 [−0.205, −0.120]**，− 纯减步 +0.065 [+0.022, +0.109]，− full −0.003 [−0.049, +0.045]；GR00T midfinal − resetfinal **+0.038 [0.000, +0.075]**，− warmreset +0.049 [+0.008, +0.089]，− 纯减步 +0.034 [−0.006, +0.074]，− full −0.028 [−0.066, +0.011]。全部格子准入，跨臂身份一致。

**读法**：
1. **喂入点下移一格，两个模型的反应方向相反。** π0.5 从 0.708 掉到 0.545（仍略好于纯减步、与 full 持平）；GR00T 从 0.573 升到 0.611（1 次前向里最好的做法，与 full 的差距缩到不显著）。
2. π0.5 的损失集中在「缓存与当前情形不匹配」的平坦/PnP 任务（PickPlaceCounterToStove .72→.32、PickPlaceCounterToCabinet .64→.30），CloseFridge 这类悬崖任务在 1 步预算下反而受益。t=1 喂入在 π0.5 上起「保护」作用：模型不把缓存当可信草稿，坏缓存带不偏它。
3. GR00T 在 t=1 喂入时基本不保留缓存信息（§6.10），下移到 0.75 后缓存才开始起作用，收益主要来自 CloseFridge（.28→.58）。
4. 对四层设计：warm start 的最佳喂入点是 **policy 相关**的，π0.5 应喂在 t=1，GR00T 应喂得更低；单一固定入口不能同时适配两者。

### 6.9 GR00T 2 次前向补齐到 13 任务 — 2026-09-22 20:50–23:25 CDT

§6.6 的 2 步调整组只覆盖 8 任务、纯减步 2 步只有 Q-B 5 任务。owner 指示补齐：`plain_k2` 补 8 任务、`warmreset_t0.5` / `resetfinal_t0.5` 各补 5 任务（exp `sdiag_macro13`，每任务 50 集，全部准入，arrays sha 0 bad）。

| 13 任务 macro SR（2 次前向） | full（4 步） | 纯减步 | warmreset | resetfinal |
|---|---|---|---|---|
| GR00T | .638 | .629 | .632 | .646 |

配对差：warmreset − 纯减步 +0.003 [−0.034, +0.040]；resetfinal − 纯减步 +0.017 [−0.022, +0.055]；resetfinal − full +0.008 [−0.032, +0.048]。**GR00T 2 步时 reset 式 warm start 与单纯减步无差别，两者都追平 4 步 full**；对照 π0.5 2 步时 reset 式比纯减步高 0.23。表 `analysis/warm_variants_groot_macro13_t0.5.md`。

### 6.10 机制探针：缓存初始化敏感性（init_probe）— 2026-09-22

独立实验，完整报告 `analysis/init_probe_20260922/complete_results.{zh,en}.md`（Codex 启动、中途让出设备，Claude 续跑合并，40/40 集，完整模式分析）。固定 teacher 访问过的观测，把检索缓存最终动作换成另一条同任务缓存 / 零 / 高斯，从 t=1 走到 0，看输出差异保留多少（输出 RMSE / 输入 RMSE）。

| 另一条同任务缓存对照 | GR00T | π0.5 |
|---|---|---|
| CloseFridge，N=2 输出 | 0.32 [0.24, 0.39] | 0.88 [0.81, 0.94] |
| PickPlaceCounterToStove，N=2 输出 | 0.09 [0.08, 0.12] | 0.56 [0.52, 0.59] |

N=2 输出的 6 组比较（2 任务 × 3 对照）π0.5 全部高于 GR00T、区间都不重叠；第一步后两者都约 0.5，分歧发生在第二步。**在 t=1 喂入时，π0.5 的输出保留大量缓存起点信息，GR00T 基本抹掉**——与 §6.6/§6.9（GR00T reset 式≈纯减步）和 §6.8（GR00T 喂得更低才受益）方向一致。它只测敏感性，不测动作质量或成功率；跨模型观测与缓存库不同，不能直接证明成功率差距的因果。

### 6.11 GR00T 起点 × 喂入点 × 步数 — 2026-09-23 09:35–13:00 CDT

**设计**（owner 09-23）。在 §6.8 之上补全 GR00T 的网格：起点 ∈ {缓存最终动作 final，缓存快照 snapshot（warmreset 所用的 x_{start_t}）}，直接喂、不加噪；喂入 flow time（π0.5 记法，1 = 噪声）∈ {1, 0.75, 0.5}；步数 n ∈ {1, 2}，走到 0。新臂：`midfinal50_t0.75/t0.5`（final、喂 0.5）、`midfinal_t0.5`（final、喂 0.75、n=2，补 11 任务）、`midreset_t0.75/t0.5`（snapshot、喂 0.75）、`midreset50_t0.75/t0.5`（snapshot、喂 0.5；审计会话历史后补跑，09-23 14:20–16:05）。臂名里的 t 只定步数预算（t0.75 = 1 步、t0.5 = 2 步）。实现 `envs.MID_ENTRY_T_BY_VARIANT`（`mid_final50`、`mid_snap`、`mid_snap50`），测试钉住时间与步长序列（step_diag 套件 129 passed / 1 skipped）。exp `sdiag_macro13`，每任务 50 集；h100 10 server ↔ timan108（main lane），weilandserver ≤ 7 server ↔ timan107（PnP lane，私有 tmux、避开 GPU3）。七个新臂全部准入、跨臂身份一致、server arrays sha 全对（每臂 650 uid，midfinal_t0.5 为 550）。

**13 任务 macro SR**（参照：full 4 步 .638；纯减步 1 步 .575、2 步 .629）：

| 起点 | 喂入点 | n = 1 | n = 2 |
|---|---|---|---|
| snapshot | t = 1（warmreset） | .562 | .632 |
| snapshot | 0.75（midreset） | .628 | **.669** |
| snapshot | 0.5（midreset50） | .637 | .632 |
| final | t = 1（resetfinal） | .573 | .646 |
| final | 0.75（midfinal） | .611 | .643 |
| final | 0.5（midfinal50） | **.646** | .600 |

配对差（13 任务，分层 bootstrap）：
- n = 1：final 喂 0.5 − 纯减步 **+0.069 [+0.029, +0.109]**，− final 喂 t=1 **+0.073 [+0.032, +0.115]**，− full +0.008 [−0.034, +0.049]；snapshot 喂 0.75 − 纯减步 **+0.051 [+0.012, +0.089]**，− warmreset **+0.066 [+0.029, +0.103]**，− final 喂 0.75 +0.017 [−0.017, +0.051]，− full −0.011 [−0.049, +0.028]。
- n = 1（续）：snapshot 喂 0.5 − 纯减步 **+0.060 [+0.020, +0.100]**，− warmreset **+0.075 [+0.035, +0.116]**，− full −0.002 [−0.042, +0.040]。
- n = 2：snapshot 喂 0.75 − 纯减步 **+0.040 [+0.002, +0.077]**，− warmreset **+0.037 [+0.002, +0.072]**，− final 喂 0.75 +0.026 [−0.011, +0.063]，− full +0.031 [−0.008, +0.069]；final 喂 0.5 − final 喂 t=1 **−0.046 [−0.086, −0.005]**，− final 喂 0.75 **−0.043 [−0.083, −0.005]**；snapshot 喂 0.5 − snapshot 喂 0.75 −0.037 [−0.074, 0.000]，− 纯减步 +0.003 [−0.037, +0.043]，− full −0.006 [−0.046, +0.034]。

**读法**：
1. **GR00T 上把喂入点从 t=1 下移是有益的**：1 步时 final 喂 0.5 与 snapshot 喂 0.75 都显著优于纯减步并追平 4 步 full；2 步时 snapshot 喂 0.75（.669）是 GR00T 所有 2 次前向做法里最高的，显著优于纯减步与 warmreset，比 full 高 0.031（区间含 0）。
2. **喂入点与步数耦合**：1 步时两种起点都是喂得越低越好（final .573 → .611 → .646；snapshot .562 → .628 → .637）；2 步时最佳喂入点在 0.75，喂到 0.5 两种起点都掉回纯减步水平（final .600、snapshot .632）。喂得太低 + 两步时，缓存几乎不被改写，失去了第一步引入当前观测的纠偏作用。
3. **起点 snapshot 与 final 在同一喂入点上无显著差别**（喂 0.75：n=1 +0.017、n=2 +0.026，区间含 0），与 π0.5（§6.3/§6.5：resetfinal ≈ warmreset）一致。
4. 与 π0.5（§6.8：喂 t=1 最好，下移一格 −0.163）对照：**最佳喂入点是 policy 相关、且与步数预算耦合**；GR00T 在 t=1 基本抹掉缓存（§6.10），因此需要喂得更低才能让缓存起作用。

### 6.12 成功轨迹长度：warm reset 是否让成功集更短 — 2026-09-23

详见 `analysis/success_length.md`（脚本 `analysis/success_length.py`，产物 `data/analysis/success_length.json`）。只看成功集，按同一环境身份与 full 配对（两臂都成功的集），长度 = `episode_summary.n_env_steps`。

- **π0.5（N=2）**：warm reset 成功集比 full 短 97–110 环境步（t=1 喂入，约 −23 到 −26%）、157 步（t=0.9 喂入，−38%），**13/13 任务都更短**；纯减步长 33 步（+7.5%），精确续跑无显著差别（+15 [−26, +57]）。
- **GR00T**：N=1 时 t=1 喂入与精确续跑反而长 19–31 步（+6–9%），喂在 t=0.75 / 0.5 与 full 持平；N=2 时最终动作起点的 warm reset 短 13–28 步（−4 到 −7%）。
- 与 §6.10 一致：π0.5 在 t=1 喂入时保留缓存（缓存由成功轨迹建成），成功集更像照着一条示范走；GR00T 抹掉缓存，长度回到 full 附近。
