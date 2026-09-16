# Dispatch Surface Rev 1 — A′ primary 结果与执行方分析（提交 Review Authority）

> **术语统一（2026-08-31，owner 裁定，全库生效）**：**GST = Grid-Searched Threshold（网格搜索阈值）**——本文中的 threshold / thr / T / tgrid / (fh, ws) 网格 / 速率索引均指它；**RIT = Risk-Indexed Threshold（风险索引阈值）**——本文中的 s-only / s0 / 校准分位切 / 风险阶梯 / surface 均指它（(s,v) 版 SV 为 RIT 的消融）。历史正文与 Review Log 按章程不改写，以本注释为准。

> 本文由 Execution Authority 起草，2026-08-29 00:10。**它报告一个负结果，并把执行方对原因的推断和可能的出路列成问题，不主张裁决。**
> 冻结判据（配额 5/5/10/30、δ\*、Gate 1/2、单价、q0.05、`--trials 30`）在执行期**一项都没动**。
> 两个 suite 的 primary 层各 1800 episodes 已跑完并分析；**secondary 两层未启动**（owner 指示暂停）。
> 所有数字均可由 §9 的产物和命令复算。

**一句话摘要**：两个 suite 的 Gate 1 都没过（`line_demoted` × 2）。libero_10 上 SV 在同 SR 下省成本约 5.6%，
但 bootstrap 上界只到 −1.3%，够不到 −5%；libero_spatial 上 SV 被 `t_fh50_ws20` 严格支配（SR 持平、成本贵 4.9%）。
执行方认为这是干净的负结果而非跑坏：数据完整、绑定全部校验通过、插值几乎全部落在 frontier 内部。
但分解到 tier 层面后有几处结构性的东西值得 Review Authority 看一眼（§4、§5），其中两条可能改变「方法完了」这个读法。

---

## 1. 做了什么

### 1.1 执行序列（冻结顺序，未抢跑）

| 步骤 | 状态 | 关键数字 |
|---|---|---|
| l10 primary，6 臂 × 300 | ✅ 1800/1800 | accepted 1800、未接纳 0、重复 uid 0、重试 0 |
| `analyze_precheck`(l10) | ✅ | `line_demoted` |
| spatial primary，6 臂 × 300 | ✅ 1800/1800 | accepted 1800、未接纳 0、重复 uid 0、重试 48（见 §1.3） |
| `analyze_precheck`(spatial) | ✅ | `line_demoted` |
| l10 / spatial secondary | ⏸ 未启动 | owner 指示暂停；它们是描述性层，不影响 Gate |
| `finalize_cross_suite` | ⏸ 未运行 | 两个 primary 都 `line_demoted`，结论不会变 |

拓扑：timan107 48 worker（8× GTX1080 EGL 渲染）→ weilandserver 4090 4 replica（batch 32 / 25 ms），
吞吐 18–23 ep/min，每层约 80–100 min。

### 1.2 数据完整性（执行方自检）

- 每 suite 每臂恰好 300 条，`(task, init)` cell 无空、无重复。
- `analyze_precheck` 的全部内容绑定通过：arm matrix ↔ launch ledger ↔ fit record ↔ artifact sha256、split manifest、A′ pool rollup、policy fingerprint、library sha。
- 两 suite 的 `delta_star` 与冻结值差 < 1e-6（l10 `5.9096355438`、spatial `6.1298200607`）。
- B1（`run_stage3_from` 的 float32 时间戳）已修，影响面验证为只有 `y7` 变（见 handoff §2）；两张标定表在修后重建。
- `branch_shares`：l10 bracket **99.92%**，spatial bracket **86.2%** / high 13.8%——SV 的 SR 几乎总在 threshold frontier 的 SR 范围内，插值是内插不是外推。

### 1.3 执行期出的两个问题（均为执行方环境错误，已修，不影响数据）

1. spatial 首次启动崩：`tether exec` 下 `HOME` 指向 tether-home，LIBERO 的 `~/.libero/config.yaml` 不在那里，包在 import 时 `input()` 问路径，nohup 下 stdin 是 EOF。1200 次 EOFError、零产出。按 PID 逐个 kill 后加 `export LIBERO_CONFIG_PATH=/home/zixuans8/.libero` 重启。
2. spatial 重启后前 48 个 episode 报 `FileNotFoundError: …/assets/scenes/libero_tabletop_base_style.xml`：conda 包里没装 `assets/`，LIBERO 退回从 HuggingFace 下载，48 个 worker 同时冷启动各撞一次。全部由 conductor 重试并接纳（`attempt: {1: 1752, 2: 48}`），之后零错误。
   - 只有 tabletop 场景要这个文件，libero_10 是 KITCHEN/LIVING_ROOM 场景不碰它，故 l10 不受影响。
   - **一个要记录的事实**：spatial 的 arena XML 来自 `jadechoghari/libero-assets` 的当前内容，不是本地固定副本。

---

## 2. 结果

### 2.1 Gate 1 裁决（冻结判据：`q0.05(D_sr) ≥ 0 AND q0.95(D_c) ≤ −0.05`）

| suite | `q0.05(D_sr)` | `q0.95(D_c)` | branch (bracket/high/low) | verdict |
|---|---|---|---|---|
| libero_10 | 0.000 | **−0.0128** | 0.9992 / 0 / 0.0008 | `line_demoted` |
| libero_spatial | 0.000 | **+0.0865** | 0.8623 / 0.1377 / 0 | `line_demoted` |

- `q0.05(D_sr) = 0` 在 bracket 分支下是构造性的（SV 落在包络内时 `D_sr ≡ 0`），**不含信息**；Gate 1 的实质全在成本项。
- Gate 2 按固定时序**未计算**（Gate 1 不过即短路）。此前挂着的「Gate 2 SR 分量是优越性检验」争议在本轮不起作用。

### 2.2 臂级点估计（`analyze_precheck` 的 decision-weighted 成本，单位 ms/decision）

**libero_10**

| 臂 | FULL | WARM | MISS | cost | SR |
|---|---|---|---|---|---|
| `t_fh30_ws20` | 0.232 | 0.194 | 0.574 | 50.20 | 0.697 |
| `t_fh50_ws20` | 0.329 | 0.210 | 0.461 | 44.33 | 0.567 |
| `t_fh70_ws10` | 0.440 | 0.145 | 0.415 | 39.34 | 0.403 |
| `dsp_s0` | 0.327 | 0.257 | 0.416 | 43.47 | 0.583 |
| **`dsp_sv`** | 0.444 | 0.115 | 0.441 | **39.72** | **0.493** |
| `dsp_sv_minus`（描述性） | 0.169 | 0.435 | 0.396 | 48.83 | 0.763 |

**libero_spatial**

| 臂 | FULL | WARM | MISS | cost | SR |
|---|---|---|---|---|---|
| `t_fh30_ws20` | 0.350 | 0.205 | 0.445 | 43.24 | 0.940 |
| `t_fh50_ws20` | 0.491 | 0.188 | 0.322 | **35.54** | **0.937** |
| `t_fh70_ws10` | 0.582 | 0.080 | 0.338 | 32.52 | 0.837 |
| `dsp_s0` | 0.199 | 0.574 | 0.227 | 44.22 | 0.947 |
| **`dsp_sv`** | 0.432 | 0.267 | 0.302 | **37.28** | **0.930** |
| `dsp_sv_minus`（描述性） | 0.153 | 0.575 | 0.272 | 46.84 | 0.947 |

SR 的 cluster bootstrap SE：l10 约 **0.024**、spatial 约 **0.013**（`t_fh70` 0.021）。
tier 单价（冻结）：FULL 10.26、WARM 46.82、MISS 67.52 ms。

### 2.3 同 SR 下的成本差（点估计，与 verdict 一致）

- l10：SV 的 SR 0.493 落在 fh70（0.403）–fh50（0.567）之间，插值对照成本 42.09 → **SV 省 5.6%**。bootstrap p95 只到 −1.3%。
- spatial：SV 的 SR 0.930 落在 fh70（0.837）–fh50（0.937）之间，插值对照 35.34 → **SV 贵 5.5%**。`t_fh50_ws20` 同时 SR 更高（+0.7 pt，在噪声内）且成本更低（−4.7%）。

### 2.4 SV 相对 S0（描述性，Gate 2 未运行）

| suite | ΔSR（配对均值） | 成本变化 | Rev 1 §7.3 离线预检 |
|---|---|---|---|
| l10 | **−0.090** | −8.6% | −6.2% |
| spatial | −0.017 | −15.7% | −12.1% |

成本节省**兑现甚至超出**离线预检，但离线预检没有 SR 轴；线上 SR 在 l10 掉了 9 个点。

---

## 3. 执行方对「为什么」的分解（每条都标了证据强弱）

### 3.1 libero_10：每一个 FULL_HIT 都在花 SR（证据强）

对 6 个臂做 `SR ~ FULL share` 的线性拟合：

| suite | 斜率（每 +10% FULL） | 截距（FULL = 0） |
|---|---|---|
| libero_10 | **−11.6 SR pt** | 0.961 |
| libero_spatial | −1.8 SR pt | 0.990 |

两个截距与 π0.5 官方 LIBERO 数（l10 ≈ 92%、spatial ≈ 98%）相当，说明 harness 本身没问题；**是缓存命中在消耗 SR**。
l10 的斜率意味着：FULL share 从 0.23（fh30）到 0.44（fh70/SV），换 16–22% 成本，付 29 个 SR 点。
在这个斜率下，**任何** dispatch 规则都不可能在 l10 上"同 SR 省 5%"——因为省成本的唯一大头是 FULL（−85%），
而每 10% FULL 要赔 11.6 个 SR 点，远超 5% 成本对应的 SR 当量。

这条不是曲面的问题，是 **library/retrieval 在长程任务上命中质量**的问题：cp1 相似度找到的 winner，其动作块在 l10 上
平均每替换 10% 的决策就让 11.6% 的 episode 失败。曲面只能在给定 winner 后决定"用不用"，救不了 winner 本身不对。

### 3.2 libero_spatial：SV 比 fh50 更保守却没换到 SR（证据中）

spatial 上 FULL 到 0.49 都几乎免费（fh30 0.940 → fh50 0.937），到 0.58 才掉（fh70 0.837）。
SV 停在 FULL 0.432——比 fh50 少 6 个点的 FULL——却把这些决策放进了 WARM（0.267 vs 0.188）。
WARM 单价 46.82 是 MISS 的 69%，所以 SV 多花的钱在 WARM 上，而 SR 没有比 fh50 高（0.930 vs 0.937）。

读法：**在 spatial 上，曲面对"哪些步能 FULL"的排序不比一个标量相似度阈值更好**，而且它把本可以 FULL 的步判成了 WARM。
`v` 这一维的设计意图是"识别可以安全从 WARM 挪到 FULL 的步"，spatial 上没有观察到这个效果（S0 → SV 的 FULL 从 0.199 到 0.432，
但 SR 从 0.947 到 0.930，fh50 用更简单的规则到 0.491 而 SR 0.937）。

### 3.3 δ\* 的机械规则在两个 suite 上都选到了网格顶（证据强，含义待裁）

`fit_record.delta_neighbours = {"minus": …, "plus": null}`——**两个 suite 的 δ\* 都是 δ grid 的最大值（p90）**。
这不是巧合：§4.2 的 accuracy 定义为 `mean(Y_eff ≤ δ | 非 MISS)`，δ 越大越容易满足；hitshare 也随 δ 单调增；
规则是"accuracy 合格者中取 hitshare 最大"——**它结构性地选最大的 δ**，除非 accuracy 门在中途挡住。
两个 suite 的 accuracy 在 δ 顶都是最高（l10 0.980、spatial 类似），门没挡。

后果：SV 被放在曲面**能取到的最激进的工作点**上。而 SV−（p80，l10 δ=5.238）在 l10 上是 (SR 0.763, cost 48.83)——
**同时**比 fh30 (0.697, 50.20) SR 高 6.6 pt、成本低 2.7%，是本轮唯一一个严格支配某个 threshold 臂的曲面工作点。
在 spatial 上 SV− 是 (0.947, 46.84)，与 S0 SR 相同但更贵，没有这个效果。

### 3.4 单个工作点 vs 三点扫描（设计层面，证据弱——只是一个观察）

threshold 基线用三个工作点连成 frontier；曲面只有一个点（δ\*），且由与 SR 无关的规则选定。Gate 1 的插值处理了"工作点不同"，
但没有给曲面一条 frontier。如果把 SV− 和 SV 也连成线（**post-hoc、描述性、只有两点、线性插值宽松**）：

| suite | 匹配成本 | threshold SR | surface SR | 差 |
|---|---|---|---|---|
| l10 | 39.72 | 0.416 | 0.493 | **+0.077** |
| l10 | 44.33 | 0.567 | 0.630 | **+0.063** |
| l10 | 48.83 | 0.666 | 0.763 | **+0.097** |
| spatial | 37.28 | 0.937 | 0.930 | −0.007 |
| spatial | 43.24 | 0.940 | 0.940 | 0.000 |

l10 上曲面线在三个成本点都高 6–10 pt（SE ≈ 2.4 pt/臂）；spatial 上持平，而且 threshold 能到 35.5 以下、曲面到不了。
**执行方不主张这是"曲面赢了"**——它是事后挑了一条对曲面有利的比法。但它与 §3.1/§3.3 一致：在 l10 上，曲面在**同成本下选出的 FULL 步更安全**，
只是 Rev 1 把它放在了一个 SR 已被 FULL 拖到 0.49 的工作点上，那里 frontier 极陡（每 SR 点 ≈ 0.78% 成本），5.6% 的节省在 300 个 episode 下分辨不出来。

### 3.5 Gate 1 的 high 分支会丢掉 SR 优势（Gate 设计属性，请确认是否有意）

`frontier_record` 在 `sr_sv > max(srs)` 时把对照成本钳到 fh30 的，`D_sr` 记正值但 **Gate 1 不奖励它**（只要求 `≥ 0`）。
所以一个"同成本下更安全"的方法**按构造不可能通过 Gate 1**——它省不出 5% 成本，SR 优势又被丢掉。
用 §3.3 的 SV− 试算：high 分支，`D_c = (48.83 − 50.20)/50.20 = −2.7%`，仍不过。
这与 Rev 1 的主张方向（"同 SR 更便宜"）一致，执行方不认为它是错的；但它意味着 **l10 上观察到的那类效应，无论真假，这套 Gate 都看不见**。

### 3.6 追问：l10 是"匹配太远"还是"匹配错了东西"？（owner 提问后补的分解，证据中等偏强）

三组数据把"retrieval 太差"拆开了：

**(a) 匹配并不比 spatial 远。** FULL 命中的 cp1 分数 l10 反而更高（fh30 p10 = 0.99813 vs spatial 0.98549）；
动作空间偏差也一样大——δ\* = p90(y10) 为 l10 5.91 vs spatial 6.13。**同样的距离，l10 的 SR 代价是 spatial 的 6 倍**（−11.6 vs −1.8 pt / +10% FULL）。

**(b) 最严的阈值也不安全。** `SR ~ FULL` 在 0.17–0.44 区间是线性的，没有"前几个命中免费"的凹段。
fh30 只接受相似度最高的 23% 步（p10 = 0.99813），线性外推它已经付了 ≈ 27 pt。
如果只是"边缘匹配差"，曲线应当先平后陡；实际是从头就陡 ⇒ **cp1 相似度在 l10 上不区分安全与不安全的命中**。

**(c) 崩的是"both X and Y"型任务。** SV− → SV 的按 task 跌幅：

| 跌幅 | task |
|---|---|
| −0.53 | 0 put **both** the alphabet soup **and** the tomato sauce in the basket |
| −0.53 | 4 put the white mug on the left plate **and** put the yellow… |
| −0.50 | 6 put the white mug on the plate **and** put the chocolate… |
| −0.37 | 7 put **both** the alphabet soup **and** the cream cheese box… |
| −0.27 | 9 put the … mug in the microwave **and** close it |
| −0.07 / 0.00 | 5 pick up the book…（单目标）/ 1 both cream cheese and butter（全臂 1.00） |

~~失败 episode 的 FULL 命中出现得更早（相对位置中位 0.25 vs 成功 0.40）。~~ **已撤回（§8.3）**：该统计被 episode 长度混杂，within-task 按前 1/4 决策算方向相反。

执行方最初的读法是「不是匹配太远，是 key 量错了东西」；经 §7.8 复审后改为：**效果高度集中在若干顺序/多阶段 task，且高 cp1 similarity 与局部 y10 都没有阻止失败，提示现有 key/Y proxy 可能缺少 task-progress 信息；library coverage 是竞争解释。** 下面保留原推理供参考： 16 帧空间池化的视觉 + 本体状态 key 能找到画面几乎一样的步，
但不编码"现在在做第几个子目标 / 已经放了哪个物体"，于是一个近乎完美的视觉匹配可以是**另一个子目标阶段的动作块**；
在双物体顺序任务上照搬它就走错分支，且越早命中越早走错。spatial 单物体、单阶段，同样的 key 没有这个歧义。

**这份数据分不开的两种解释**：(i) key 缺阶段信息；(ii) dlib 每 task 只有 5 条 episode，对 400 步双物体任务的
(init × 阶段) 覆盖太薄，最近邻本来就不该被信任。两者都会产生上面的全部现象。
一个便宜的判别实验：对 task 0 单独把 dlib 扩到 20–30 条，看 `SR ~ FULL` 斜率是否变平——变平是 (ii)，不变是 (i)。这属于新立项，执行方不启动。

---

## 4. 什么不能救、什么可能救（执行方倾向，不是建议执行）

**在 Rev 1 冻结判据下，这条线已经结束。** 以下每条都需要新的预注册，执行方不会自行启动任何一条。

| 选项 | 内容 | 执行方看法 |
|---|---|---|
| A. 报负结果 | 按 §2 原样报告，附 §3 机制分解 | 应该做，且是本轮唯一不需要新预注册的产出。threshold 基线在两个 suite 上都不弱，这本身是信息 |
| B. 加样本 | A′ 已用满（每 task 30 init 是配额上限）；要加得动 5/5/10/30 | **救不了**：spatial 符号是正的；l10 真效应 ≈ 5.6% 刚好压线，要 `q0.95 ≤ −5%` 需真效应 ≥ 8% 左右，加样本不改变真效应 |
| C. 放宽 Gate | 把 −5% 改小 | **不应做**，且 spatial 放到 0 也不过 |
| D. 曲面 frontier vs threshold frontier | 给曲面也扫 δ（p70/p80/p90 三点），比两条线 | 最便宜的补救：SV− 已存在，只缺一个 p70 点（每 suite +300 ep ≈ 15 min）。但要先回答 §5 Q4——这种比法能否预注册、主张变成什么 |
| E. 换主张 | 从"同 SR 更便宜"改成"同成本更安全" | l10 有信号（+6–10 pt），spatial 无。会是 suite-specific；要新 Gate、新预注册 |
| F. 改 δ 规则 | §4.2 的规则结构性选最大 δ；改成按 D_dev 上的某个 SR 代理选 | 触及冻结项；且 D_dev 没有 episode 级 SR 标签（只有动作偏差 y），代理怎么定是开放问题 |
| G. 去修 retrieval | l10 每 10% FULL 赔 11.6 pt，根子在 winner 质量 | 执行方认为**这是 l10 上真正的瓶颈**，但它是另一条线（library/key builder），不是 dispatch surface 能解决的 |

执行方的倾向：**A 必做；D 作为描述性附录值得跑（成本极低）；E/G 是否立项由 owner 定。** B/C/F 不做。

---

## 5. 请 Review Authority 裁定 / 回答

- **Q1**（§3.3）δ\* 规则在两个 suite 上都选到网格顶，是设计预期吗？如果 accuracy 门的定义使它必然单调，那么"机械选择"实际上等于"取 p90"，
  曲面的工作点就不是被选出来的而是被定义出来的。这是否应该在报告里写明？
- **Q2**（§3.5）Gate 1 high 分支丢弃 SR 优势是有意的吗？如果是，报告应写明"本 Gate 只检验成本方向"，以免读者把 `line_demoted` 读成"曲面在所有方向都输"。
- **Q3**（§3.1）l10 全臂 SR 0.40–0.76 vs 官方 ≈ 0.92。本轮没有 no-cache 对照臂，截距 0.961 是拟合外推。是否需要补一个纯 MISS 臂（300 ep，≈ 15 min）把"缓存本身的 SR 代价"钉死？
- **Q4**（§3.4/§4-D）用已有的 SV− 加一个 p70 点做"曲面 frontier vs threshold frontier"，能否作为**明确标注 post-hoc 的描述性附录**进报告？还是必须重新预注册、用新 A′ 才能报？
- **Q5** secondary 两层（`score_hysteresis`，各 1200 ep，共约 2 h）：跑，还是不跑？它们不改变 Gate 结论，只提供外部效度。
- **Q6** 挂着的三项（Gate 2 裕度 / A1 规格 / D0 容差）：Gate 2 裕度本轮不起作用；A1 与 D0 是协议卫生问题，是否仍需裁定？
- **Q7** §1.3 第 2 条：spatial 场景资产来自外部 HF repo 的当前内容。这是否需要写进 data_authority 台账？

---

## 6. 复现与产物

**timan107**（`/tmp/`，实验后会清）
```
/tmp/dsp_precheck/libero_10_primary/{journal.jsonl,per_step.jsonl,per_step.jsonl.launch.json,verdict.json}
/tmp/dsp_precheck/libero_spatial_primary/{同上}
/tmp/dsp_shared/{libero_10,libero_spatial}/surface_rev1/{fit_record.json,fit_record_s_only.json,*.npz}
/tmp/dsp_shared/config/precheck_<suite>_<layer>/
```
本地副本：`exp/dispatch_surface/data/aprime_rev1/<suite>_primary/`（见 handoff §6）。

**analyze**（在 timan107 `/scratch/zixuans8/openpi_dispatch`）：`./analyze_t107.sh <suite> primary`
**tier 分解**：`/tmp/tier_mix2.py`（本文 §2.2、§3.1、§3.4 的全部数字由它产生；单价与 analyzer 的 `unit_cost` 一致）。

**执行方在本轮犯的错**（记录以便审查）：
- 第一版 tier 分解把 stage 单价当成 tier 单价（把 WARM 算成 27.69 而非 46.82），得出"WARM ≈ MISS、三层经济上退化为两层"的错误推断；对照 `analyze_precheck.unit_cost` 后更正。本文所有成本数字均为更正后的。
- 启动 spatial 时没检查 `HOME` 下的 LIBERO 配置，白跑 1200 次 worker 启动。
- 停 server 时在同一条 shell 里写了未加括号的字面量，`ps | grep` 匹配到自己把 shell 杀了（本线第四次）。

---

## 7. Review Authority 回复与建议（Codex，2026-08-29）

### 7.1 裁决先行

**Rev 1 的确认性主张失败，必须原样报告；但“dispatch surface 方法完了”不成立。** 更精确的结论是：

1. 两个 suite 都没有证明“SV 在 threshold frontier 的同 SR 位置，成本至少节省 5%，且单侧 95% 上界
   仍不高于 −5%”。这是 Rev 1 唯一被预注册并有确认性地位的 Gate 1 claim，现已失败。
2. libero_spatial 当前数据没有 surface 优势；在低成本端 threshold 明显更好，在高 SR 区两者近似持平。
3. libero_10 却存在强而一致的**同成本 SR 优势 / frontier 改善信号**。它不能事后改写 Gate 1，却明确
   反驳“曲面没有用”以及“任何 dispatch 都不可能赢 threshold”的推断。

因此本轮应写成：**confirmatory single-point cost-saving claim failed；exploratory evidence suggests a
better surface Pareto frontier on the harder long-horizon suite, but not on the saturated spatial suite。**

### 7.2 对执行方机制解释的两项重要纠正

#### 纠正 A：§3.1 的跨臂回归不能解释为“每一个 FULL_HIT 都在花 SR”

`SR ~ FULL share` 只有 6 个臂级点；各臂同时改变了 FULL/WARM/MISS 的集合、访问到的状态分布、episode
长度和具体被替换的步骤。它是生态相关，不是 decision-level 因果效应。`FULL=0` 的 0.961 还是样本范围外
外推，不能据此证明本 harness 的 no-cache SR，更不能把 l10 的失败唯一归因于 winner/retrieval。

更直接的反证来自已经保存的配对结果（以下均是 reviewer 的**事后诊断**，不改变正式 Gate）：

| l10 对比 | 成本 | ΔSR | 配对 discordance（surface 胜/负） | task-stratified paired bootstrap q0.05 |
|---|---:|---:|---:|---:|
| SV vs fh70 | 39.72 vs 39.34（仅贵 1.0%） | **+9.0 pt** | 53 / 26 | **+4.33 pt** |
| SV− vs fh30 | 48.83 vs 50.20（便宜 2.7%） | **+6.67 pt** | 46 / 26 | **+2.0 pt** |

未校正的 exact paired 两侧 p 分别约 0.0032 / 0.0245；由于比较是在看过结果后选择的，只能用于定位
信号，不能当新确认性检验。但它们足以否定 §3.1 中“在这个斜率下任何 dispatch 都不可能同成本改善”
的机制结论：当前 surface 已经在两个近似成本锚点选出了比 scalar threshold 更安全的步骤。

#### 纠正 B：失败的核心是 estimand / operating-point 设计，不等于排序无效

l10 的正式 SV 点相对插值 threshold 在同 SR 下点估计已省 **5.6%**；它失败是因为真效应恰贴着
`−5%` 门槛，bootstrap 上界只有 `−1.28%`，所以没有证明“至少省 5%”。同一数据换成近似同成本的
纵向比较，surface 的 SR 优势反而清楚。二者不是矛盾，而是陡峭 frontier 上不同坐标系、不同 margin
与不同统计功效的结果。

所以 `line_demoted` 应解释为“预注册的强 margin claim 未建立”，不能写成“SV 被 threshold 全面支配”。
后一句只在 spatial 的当前 operating point 上接近事实。

### 7.3 δ 规则确有结构性偏置，但执行方提出的 p70 补点方向不对

Q1 的答案是：hitshare 随 δ 单调不减；只要 p90 通过 accuracy 门，规则就必然选择 hitshare 最大的 p90
（并非 accuracy 本身在数学上必然单调，但它把 `Y≤δ` 的容忍标准也随 δ 放宽，使顶点尤其容易通过）。
因此两 suite 选到顶不是一个有辨识力的“最优点选择”，而是当前目标与候选集共同造成的顶点吸附。
报告应明确写出。

但 §4-D 所说“只缺 p70”不是最有价值的补点：p70 比 SV−(p80) 更保守、更贵，大概率跑到 threshold
frontier 的共同成本区间之外。当前真正缺的是 **p90 以上的更激进点（如 p95/p97.5）**，尤其 spatial
的 surface 最低只到 37.28 ms，而 threshold 已覆盖到 32.52/35.54 ms；没有更激进 surface 点，就无法
判断 spatial 是 surface 形状真的差，还是 δ 候选集把它截断了。

这也解释下一版为什么不应再选一个 δ\*：应把 δ 当成用户的风险—成本旋钮，比较完整 frontier。

### 7.4 推荐的 Rev 2：从“单点过强 margin”改为公平的 frontier-vs-frontier

这是我认为最有机会保住论文、也最符合现有证据的主线：

1. **冻结三到五个 surface operating points**。只能用 D_dev 决定，不能再看 A′；候选应覆盖与三个
   threshold 点相同的预测成本/接受率区间，至少包含 p80、p90 和更激进的 p95/p97.5，而不是补 p70。
2. **同样预算比较两条 frontier**。threshold 与 surface 各用相同数量的点、相同 init/task/seed；只在两条
   曲线的共同成本支撑区间内比较，禁止 endpoint 外推。
3. **主 estimand 改为同成本的 ΔSR(c)**。可预注册 2–3 个成本锚点，或预注册共同区间内的 normalized
   AUC / hypervolume；用 task-stratified paired cluster bootstrap，并对多个锚点使用 simultaneous band
   或固定 IUT。这样“同成本更安全”和“同 SR 更便宜”都只是同一 Pareto 改善的两种坐标表示，不会再由
   high branch 把真实 SR 优势压成零。
4. **no-cache / always-MISS 必须成为正式 anchor**。它给出同一模型、同一资产、同一 harness 的 SR 与
   67.52 ms 上界，替代 §3.1 的外推截距和外部官方数字。
5. **S0 也按 matched-cost frontier 比较**。同 δ 的嵌套消融回答的是结构变化，却不保证相同 operating
   point；若论文 claim 是 v 改善 Pareto frontier，应比较 S0/SV 在共同成本上的曲线，而不是只比较同 δ。
6. **确认集必须是新的**。现有 A′ 已用于观察结果和选择 Rev 2；它只能作为 exploratory design set。
   Rev 2 的确认可以落在预先冻结的真实机器人任务、新 suite/新 official init，或独立的新模拟 test pool。
   现有 l10 的 +6–9 pt 只能作为效应量与功效设计依据。

这条叙事比“改小 −5% margin”干净得多：方法本来就是连续风险调度器，论文的自然对象就应是 Pareto
frontier，而不是由一个与 episode SR 无关的离线规则选出的孤立 δ\*。

### 7.5 在改模型之前，先做的诊断

1. **按 task 报 paired 结果**。l10 异质性极强：例如 task 1 多数臂 28–30/30，而 task 8 只有 1–5/30。
   aggregate 结果可能同时混合 ceiling、floor 与真正有分辨力的 task。应报告每 task 的 SV/SV−/threshold
   discordance 和成本，而不是再做 6 点跨臂回归。
2. **运行 always-MISS，但只标作 post-hoc diagnostic**。在现有 A′ 上补跑不会把它变成确认性证据，却能
   立刻回答 cache 是否造成绝对 SR 损失，并验证外部资产/模型/harness 下的真实基线。
3. **如只愿再跑一个 surface 点，应跑更激进点，不跑 p70**。目的只是看 spatial 能否进入 32–36 ms
   的共同成本区间；该结果同样只能设计 Rev 2，不能回填 Rev 1。
4. **检查 episode-level accumulated risk**。当前 y 是逐 decision 相对 full-policy action 的偏差，l10 的
   长时程可能把许多局部“小误差”累积成失败。用现有 A′ 做 post-hoc 关联：consecutive FULL、episode
   FULL/WARM 数、命中发生的相对时刻、task id 与失败的关系。若风险主要由连击/尾段累积驱动，Rev 2
   应加入 history budget（如 consecutive-hit cap / cumulative predicted-risk budget），而不是先重做 retrieval。
5. **只有在 matched-cost surface 仍失败时再改 retriever**。当前 l10 的近成本配对结果已经证明 verdict
   排序有增量价值；直接把整条线转成 retrieval 项目会丢掉已有的正信号。

### 7.6 对 Q1–Q7 的逐项答复

- **Q1：是顶点吸附，不是有信息量的最优选择。** 如实写明，并在 Rev 2 删除单点 δ\* 主张。
- **Q2：high 分支是原 claim 有意的非对称设计，但 `line_demoted` 只能否定该成本-margin claim。** 它不能
  推导“所有 Pareto 方向都输”；Rev 2 应改用共同成本上的纵向 frontier estimand。
- **Q3：需要 exact-stack no-cache anchor。** 现有 A′ 上可先作 post-hoc diagnostic；新的确认集上应预注册
  为正式臂。不能继续用 6 点回归截距替代。
- **Q4：已有 SV−/SV 曲线可以明确标注 post-hoc 后进分析/附录，不能当确认结果。** 若再补点，现有 A′
  仍只能 exploratory；而且优先补 p95/p97.5，不是 p70。正式 frontier claim 必须使用新确认集。
- **Q5：现在不跑 secondary。** 它不改变失败裁决，gate 还会截断最需要辨识的 region；先用同等算力补
  always-MISS 与激进 surface 诊断，等 Rev 2 冻结后再把 secondary 当部署外部效度。
- **Q6：仍需收口但不影响本轮 verdict。** Gate 2 margin 在 Rev 2 重写；A1 保留为“单调模型失配”诊断，
  不能事后变成门；D0 容差/语义按已经修复并重建 table 的实现归档，不再反复解释。
- **Q7：必须进入 data_authority。** 立即记录实际 worker 使用的 HF repo revision、缓存文件 SHA256、下载
  时间、文件列表与 license，并保存不可变副本。只写“当前内容”不足以复现 spatial，若无法恢复精确
  revision/hash，应把它列为 spatial 结果的 provenance limitation。

### 7.7 论文层面的底线与机会

不能把 Rev 1 写成赢，也不要隐藏两个 `line_demoted`。但当前最值得保留的核心发现是：**在更难、长程、
threshold frontier 很陡的 libero_10 上，二维 verdict 在两个近成本 operating points 都把成功率提高了约
7–9 个点；在容易且接近饱和的 spatial 上没有收益。** 这恰好支持一个可检验的新假说：额外 disagreement
信号在复杂任务中改善“哪些步可安全缓存”的排序，但价值取决于任务难度和 operating point。

若新的、预注册的 frontier 实验和真实机器人都复现这一点，方法仍足以形成有竞争力的 ICLR 故事；若
只在现有 A′ 上成立或真实机器人/spatial-like 任务仍无增益，就应把它降为负结果/分析型论文，而不是继续
修改 Gate。现在最不该做的是放宽 −5%、把 post-hoc p 值包装成确认性结果，或用 secondary 掩盖 primary。

### 7.8 对新增 §3.6 的复审：阶段混淆是假说，不是当前数据已经证明的结论

§3.6 提出的“多子目标阶段没有进入 key”是**很好的、可检验的机制假说**，并且比笼统说 retrieval 差更有
研究价值；但现有三组证据还不能把结论写成“不是匹配太远，就是 key 量错了东西”。需要收紧如下：

1. **跨 suite 的绝对 cp1 score 不可直接当成可比距离尺。** 两个 suite 的 library、观测分布和 score
   calibration 都不同；0.998 比 0.985 高只说明 embedding cosine 更饱和，不说明语义 nearest neighbour
   更正确。`p90(y10)` 相近也只说明短 action-chunk 的加权距离相近；它可能正好对“动作方向短期相似、
   但长期子目标错误”不敏感。这一点反而支持检查 proxy，但不能证明匹配不远。
2. **“fh30 从头就陡”仍来自 §3.1 的 6 点生态回归。** 不同臂诱导不同 closed-loop state distribution，
   不能由看似线性推出“最高分的前 23% 也同样不安全”，更不能排除一个被 task mixture 掩盖的 knee。
3. **both-task 关联很有启发，但不是充分条件。** task 1 同为双目标却全臂 1.00，是明确反例；task 8 又
   几乎全臂 floor。SV−→SV 的 task 跌幅同时改变 δ、tier mix 和访问轨迹，不能唯一归因于 phase mismatch。
   “失败 episode 更早 FULL”还是 post-treatment association：早期错误会让 episode 变长/改变后续状态，
   失败难度本身也会同时造成早命中和失败。
4. **把 D_lib 从 5 扩到 20–30 并不是 (i)/(ii) 的干净判别。** 变好既可能是 coverage 增加，也可能是
   更多样本碰巧提供了正确 phase；不变也可能只是 30 条仍不够，而非 key 一定错误。

因此当前允许的表述是：**效果高度集中在若干顺序/多阶段 task，且高 cp1 similarity 与局部 y10 都没有
阻止失败，提示现有 key/Y proxy 可能缺少 task-progress 信息；library coverage 是竞争解释。** 不应提前把
“key 量错了东西”写成机制结论。

更干净、而且先不烧 rollout 的判别顺序如下：

1. 从 query/library H5 为每一步构造**独立 phase label**：优先用 LIBERO goal predicate / object-in-region /
   gripper-object state形成已完成子目标 bitmask；拿不到时才用人工审核的小样本或规范化进度作弱代理。
2. 在固定 query 上审计 top-k：报告 winner phase mismatch rate、top-k 内正确 phase 是否存在、以及 mismatch
   对 y7/y10 与 extreme tail 的条件分布。这样直接测“匹配错阶段”，不靠 episode SR 反推。
3. 做两个正交的离线干预：固定 key 增大 library size 的 coverage curve；固定同一 library 做 phase-consistent
   filter/rerank。前者改善支持 coverage，后者改善支持 key/progress；二者都改善则说明两者共同存在。
4. 特别检查当前的 `v`：若 top-k 全部集中在同一个错误 phase，低 disagreement 会给出虚假确定性；若 top-k
   跨 phase 而 v 高，说明 surface 已能看到歧义，只是激进 δ 没有充分利用。这个结果直接决定下一步是改
   retrieval key，还是改 dispatch/history budget。
5. 只有离线 phase audit 给出明确效应后，再在新确认集上加入 `phase-aware key` 或 `phase-consistent rerank`
   的闭环臂；不要先用 task 0 扩库 rollout 来替代机制识别。

这一补充不改变 §7.4 的主推荐。即使 phase-aware retrieval 最终有效，它也应被视为与 surface 正交的系统
组件：现有 l10 配对数据已经显示 surface verdict 在同一 retriever 下能改善 Pareto 排序；最强论文设计是
分别消融 `threshold / surface / phase-aware retrieval / phase-aware retrieval + surface`，证明二者解决的是
“找对候选”和“在当前风险—成本预算下用不用候选”两个不同问题。

---

## 8. Execution Authority 对 §7 的复核与 Rev 2 计划草案（2026-08-29）

> 执行方逐条验证了 §7 的数字与假说；能复现的标 ✅，补充的标 ➕，不支持的标 ❌。所有新数字由本地副本
> `exp/dispatch_surface/data/aprime_rev1/` 计算，脚本在 §6。§8.3 是给 Review Authority 裁定的计划草案，执行方不自行启动任何 rollout。

### 8.1 对 §7.2 纠正 A/B 的复核

✅ **四个数字全部独立复现**（task-stratified paired cluster bootstrap，10000 reps，seed 20260827）：

| l10 配对 | 成本 | ΔSR | win/lose | exact p | q0.05 | q0.95 |
|---|---:|---:|---:|---:|---:|---:|
| SV vs fh70 | 39.72 vs 39.34 | **+9.00** | 53/26 | 0.0032 | **+4.33** | +13.67 |
| SV− vs fh30 | 48.83 vs 50.20 | **+6.67** | 46/26 | 0.0245 | **+2.00** | +11.33 |

接受"6 点跨臂回归是生态相关"的批评；§3.1 中"任何 dispatch 都不可能同成本改善"一句**撤回**。

➕ **执行方补的第三个同成本配对，codex 没报，但它改变了对 v 的判断**：

| l10 配对 | 成本 | ΔSR | win/lose | exact p | q0.05 |
|---|---:|---:|---:|---:|---:|
| **S0 vs fh50** | 43.47 vs 44.33 | +1.67 | 39/34 | 0.64 | **−3.00** |

s-only 曲面在同成本上对 scalar threshold **没有优势**；同成本优势只出现在两个带 `v` 的臂上。
这是目前"`v` 有排序增量"的最强证据（仍是 post-hoc），也直接支持 §7.4-5（S0 必须按 matched-cost 比而不是同 δ 比）。
按 task 看，SV vs fh70 的胜差分布在 task 0/2/6/7/9（7/1、14/3、5/2、5/3、8/4），不是单个 task 撑起来的。

spatial 的全部同成本配对都在噪声内（|ΔSR| ≤ 1.67，q0.05 ≥ −4.33，p ≥ 0.40），与 §7.1-2 一致。

### 8.2 对 §7.3（δ 补点方向）的复核

✅ 接受"顶点吸附"的表述；✅ 接受 p70 方向错、应补激进点。➕ 但两个 suite 需要的补点**不同**：

- l10 的 surface 成本区间 39.7–48.8 与 threshold 的 39.3–50.2 **已几乎重合**；p95 会把 l10 推到 fh70 以下、threshold 没有点的区域，补了也没有共同支撑。
- spatial 的 surface 只到 37.3，threshold 到 32.5——缺口全在激进端，p95/p97.5 只对 spatial 有意义。
- 离线 hitshare p80→p90 只涨 0.604→0.671（l10），线上 FULL 却 0.169→0.444；**离线分位与线上工作点的映射在两个 suite 上不一样**。

⇒ Rev 2 的 operating points 应按 **D_dev 上的预测成本 / 接受率**定义、与三个 threshold 点对齐，而不是按 y10 分位（§7.4-1 已这么写，这里给出它必要的证据）。

### 8.3 对 §7.5-4（累积风险 / history budget）的检验 — ❌ 现有数据不支持

within-task（每 task 内比失败与成功 episode 的中位数，8 个 task 两类都 ≥ 3 条）：

| 臂 | 特征 | 失败 | 成功 | 失败>成功的 task 数 |
|---|---|---:|---:|---:|
| SV | 最长连续 FULL | 17 | 15 | 3/8 |
| SV | 前 1/4 决策的 FULL share | 0.75 | 0.87 | 3/8 |
| SV | 全程 FULL share | 0.33 | 0.61 | 0/8 |
| fh70 | 最长连续 FULL | 20 | 20 | 3/8 |
| fh70 | 前 1/4 决策的 FULL share | 0.77 | 1.00 | 0/8 |

- 连击不区分失败：SV 300 条里 263 条 maxrun ≥ 11，连击是常态。
- 首个决策几乎总是 FULL（firstF = 0，所有臂）——同 task 的 init 在 t=0 近似相同，库里总有近乎完美的匹配。
- **早命中不但不预示失败，方向反了**：成功 episode 前 1/4 的 FULL share 更高。
- **撤回 §3.6 中"失败 episode 命中更早（0.25 vs 0.40）"**：那是池化在命中上的相对位置，被 episode 长度混杂（失败 104 vs 成功 57 决策）；within-task 按前 1/4 算是反的。codex §7.8-3 的批评成立，实际比他说的更糟。

⇒ consecutive-hit cap / cumulative-risk budget 在现有数据里没有落点，**不建议进 Rev 2**。

➕ **失败的形态**（所有臂）：l10 失败 episode 中位长度 = 104 决策 = 520 步 = LIBERO-10 horizon ⇒ **失败全是超时**；
失败 episode 的 FULL share 只有 0.33（成功 0.61），即失败 episode 大部分时间在 MISS、由策略自己控制。
失败模式是"漂出库覆盖后策略也收不回来"，不是"命中当场撞坏"。这与 phase 假说、coverage 假说、"策略自身在 l10 就有 ~8% 失败"三者都相容——**只有 always-MISS 锚点能拆开**（§7.5-2 正确）。

➕ **线上接受率系统低于离线目标**：SV 0.559 vs 0.671；fh30 0.426 vs 0.50；fh50 0.539 vs 0.70；fh70 0.585 vs 0.80。
线上状态比 D_dev 离库更远，threshold 臂尤甚（它们的目标接受率是按 D_dev 分数分位定的）。
Rev 2 若在成本坐标上冻结 operating points，需要裁定锚点定义在**离线预测成本**还是**线上实测成本**。

### 8.4 对 §7.8（phase 假说）的复核

✅ 接受全部四条收紧；§3.6 的结论句按 codex 的表述改写（"提示 key/Y proxy 可能缺少 task-progress 信息；coverage 是竞争解释"）。
✅ 接受"先离线 audit，不先扩库 rollout"的顺序。

➕ 可行性核查：library H5（`exp/common/data/db/libero_cache/<suite>/episode_*.h5`）每步只有
`clean_action / input_images(base+wrist) / noise_action_* / prompt_emb / robot_state(32) / vision_0`，**没有 sim state、object pose、goal predicate**。
§7.8 第 1 步的"LIBERO goal predicate / object-in-region"标签**不能直接从 H5 得到**，只有两条路：
(a) 从 init state 重放 `clean_action` 恢复 sim state（MuJoCo 确定性，需要验证重放漂移）；(b) 弱代理——`robot_state` 里的夹爪开合周期数 = 已完成的抓放子目标数，对 "both X and Y" 任务是天然的阶段计数。
执行方倾向先用 (b) 做 audit（零 rollout、一小时内能出 top-k mismatch rate），(a) 作为验证。

### 8.5 计划的硬约束（执行方掌握的事实）

1. **官方 init 已用完**：50/task 全部进了 5/5/10/30。新确认集只有三条路：
   - (a) 新 suite（libero_object / libero_goal）：weilandserver 上**没有**这两套的 teacher 语料（只有 `libero_10` / `libero_spatial`），要从 collect → shards → libraries → D_dev 表 → fit → A′ 全链跑，每套约 1–2 天算力；
   - (b) 同 suite、非官方 init（BDDL 区域重采样、新 seed）：复用全部 library / D_dev / surface，只换评测池；主张从 "official init" 变成 "fresh sampled init"，且与 A′ 同分布（统计独立但不是新分布）；
   - (c) 真机：不在执行方手里。
2. 算力：48 worker 跨节点 ≈ 20 ep/min；一个 300-ep 臂 ≈ 15 min；6 臂 primary ≈ 90 min。
3. always-MISS 臂：threshold 判据 `threshold > 1` 即恒 MISS（`judge.py:316`，config 未见上界校验），仍走检索路径故 `searched=true`、成本 67.52。需 emitter 加描述性臂 `dsp_miss`，`run_precheck --arms dsp_miss` 可单独跑；先 `--dry-validate`。
4. 更激进 δ：§4.2 的 grid 是 p10..p90，p95/p97.5 超出冻结 grid，需要 Rev 2 改 `fit_surface` 的 grid 规则。
5. Q7 provenance（已采集，待写入 data_authority）：LIBERO `download_utils.py:132` 把 `jadechoghari/libero-assets` 下载到 `$HOME/.cache/libero/assets`（timan107 实际路径 `/srv/local/zixuans8/tether-home/.cache/libero/assets`），下载时间 2026-08-28 23:08:42 −0500，HF `refs/main` = `90001343cb134b7e26e18fde0fa2416f3ed6e6a3`，1758 个文件 / 362 MB，`scenes/libero_tabletop_base_style.xml` sha256 = `bb118118e3e0…4e133f86`，全目录 (path, sha256) 排序后 rollup = `9bc4f946f44d5aef`。该目录是 worker 进程实际读取的副本；l10 primary 未触碰它。

### 8.6 Rev 2 计划草案（分阶段、每阶段有 go/no-go；请裁定）

**Phase 0 — 现有 A′ 上的 post-hoc 诊断（全部标注 exploratory，不进确认；≈ 1.5 h 算力 + 离线工作）**

| 项 | 内容 | 成本 | 回答什么 |
|---|---|---|---|
| 0a | always-MISS 臂 × 2 suite | 600 ep，≈ 30 min | 同 stack 的无缓存 SR 与 67.52 上锚；l10 缓存的绝对代价 |
| 0b | spatial 激进点 p95、p97.5（l10 不补） | 600 ep，≈ 30 min | spatial 的 surface 能否进入 32–36 ms 区间 |
| 0c | HF 资产 provenance | 0 | Q7 |
| 0d | 离线 phase audit：夹爪周期弱标签 → winner phase mismatch rate → mismatch 对 y7/y10 尾部的条件分布 → top-k 是否跨 phase 且 v 是否响应 | 0 rollout | §7.8 的 (i)/(ii) 判别，以及 v 是否"看见"歧义 |

go/no-go：0b 若 spatial 激进点仍被 threshold 支配 ⇒ Rev 2 主张限定为"难 / 长程任务"；0a 若 l10 无缓存 SR ≈ 0.9 而 SV− 0.76 ⇒ 缓存绝对代价 ≈ 14 pt 必须正面写进论文。

**Phase 1 — 预注册 Rev 2 协议（改文档，不跑）**

- estimand：共同成本支撑区间内的 ΔSR(c)，2–3 个成本锚点 + simultaneous band（或 normalized AUC），task-stratified paired cluster bootstrap 同 Rev 1；
- surface operating points（3–5 个）由 D_dev 上的预测成本定、与 threshold 三点对齐（§8.2）；锚点定义在离线预测成本还是线上实测成本——**待裁**（§8.3）；
- S0 同样按 matched-cost 比（§8.1 已给出动机）；always-MISS 为正式臂；
- 删除单点 δ\* 主张与 Gate 1 high 分支；Gate 2 裕度问题随之消失；
- A1 保留为诊断；D0 按已修实现归档。

**Phase 2 — 新确认集（三选一，待裁）**

执行方倾向 **(b)**：2 suite × 7 臂（3 threshold + 3 surface + MISS）× 300 = 4200 ep ≈ 3.5 h，且不动任何已冻结的 library / D_dev / surface；
代价是主张变成 "fresh sampled init" 且与 A′ 同分布——**这是否够格做确认集，请裁定**。(a) 每套多 1–2 天，但给出真正的新分布。

**Phase 3 — 正交组件（仅在 0d 给出明确 mismatch 效应后）**

phase-aware rerank / phase-consistent filter 作为独立臂，与 surface 做 2×2 消融（§7.8 末段的设计）。

### 8.7 执行方与 codex 的分歧（三处，均待裁）

1. **history budget 不进 Rev 2**（§8.3 数据不支持），codex §7.5-4 建议加入。
2. **0d 离线 audit 应与 Phase 0 并行**而不是排在"matched-cost surface 仍失败之后"（§7.5-5）——它不烧 rollout，且其结果决定 Phase 3 要不要立项。
3. **(b) 同分布新采样 init 是否算独立确认集**：统计上独立，但不是 codex §7.4-6 写的"新 suite / 新 official init"。执行方认为对"Pareto frontier 改善"这个主张够用，对"泛化"主张不够。

---

## 9. Review Authority 对 §8 Rev 2 草案的裁决与收敛方案（Codex，2026-08-29）

### 9.1 总体裁决

**方向通过，但当前草案还不能冻结，需先修四个设计缺口。** 我同意执行方的三项修正：

- history budget **不进入 Rev 2**。§7.5-4 原本就是条件建议；现有 within-task 数据方向相反，条件未触发。
- phase audit 与 Phase 0 **并行执行**。它不消耗确认数据，越早做越能避免把 retrieval 与 verdict 两条线混在一起。
- 独立重采的同分布 init **可以作为新确认集**，但只能确认“在这些已知任务、同一 init 生成分布上的
  frontier 改善”，不能单独支撑跨任务/跨 suite 泛化。真实机器人或新 suite 才承担外部泛化证据。

同时接受 §8.1 的新证据：l10 上 S0 在近成本点没有优势，而两处带 v 的 surface 点均有优势。这使“v 有
增量排序信息”成为很强的 post-hoc 假说。但它还不是干净的 v 确认，因为 S0 只有一个 operating point，
而 SV 有两个点形成曲线；Rev 2 必须补成 matched-cost S0 frontier。

### 9.2 冻结前必须修正的四个缺口

#### R2-B1 — Phase 2 臂数与 S0 claim 自相矛盾

§8.6 Phase 1 要求 S0 matched-cost frontier，Phase 2 却写成 7 臂：`3 threshold + 3 surface + MISS`，
没有任何 S0 臂，因而无法确认“优势来自 v”。若做完整三点嵌套比较，实际是：

`3 threshold + 3 SV + 3 S0 + 1 always-MISS = 10 arms/suite`。

若算力需要压缩，允许预注册 **2 个 S0 matched-cost 点**，形成 9 臂；不能保留 S0 claim 却不运行 S0。
当前 7 臂预算不能冻结。

#### R2-B2 — “预测成本还是实测成本”不能留到确认集决定

正确方案是两层冻结：

1. 把现有 A′ 明确升级为 **Rev 2 development set**。用它只拟合 `δ → realized cost` 的单调映射并选择
   SV/S0 δ；允许使用 cost/tier mix，不再使用 A′ SR 选择点。
2. 在 Rev 2 文档中冻结具体 δ、固定数值成本锚点 `c_1,c_2[,c_3]` 和允许的共同支撑区间。
3. 新确认集上使用每个 bootstrap replicate 的**实际 decision-weighted cost**构造两条曲线，并只在预先
   冻结的 `c_i` 上插值。若任一方法在新数据中不包围锚点，必须 fail closed 为 support miss，不能看过
   新成本后移动锚点或更换 δ。

仅靠 D_dev 预测成本不够：§8.2/§8.3 已证明 closed-loop distribution shift 使离线接受率严重失准。
反过来，在确认集上按实测成本重新挑点又会产生适应性。用旧 A′ 校准点、新独立 init 一次性验证，正好
解决两端问题。

#### R2-B3 — primary estimand 必须唯一，不能把“anchors 或 AUC”留成执行选择

我建议冻结一个主 estimand：

> 在预注册共同成本区间 `[c_L,c_H]` 上，surface 与 threshold 的 piecewise-linear frontier 之
> normalized AUC difference，等价于均匀成本预算下的平均 `ΔSR(c)`。

以 task-stratified paired cluster bootstrap 给出单侧 95% lower bound；`LB > 0` 才确认 frontier 改善。
两个固定成本锚点的 simultaneous intervals 作为关键 secondary，负责说明改善不是单个曲线交叉造成。
这样只有一个 primary test，不需要事后在 hypervolume、AUC、两三个 anchor 中挑最好看的。

若作者更看重“处处不劣”，可把 primary 改成两个 anchor 的 IUT，但会比 AUC 更低功效。二者必须在
Phase 1 二选一；我的推荐是 **AUC primary + simultaneous anchors secondary**。

#### R2-B4 — 复杂任务 claim 需要预先定义，不可直接把 suite 名当复杂度

“l10 复杂、spatial 简单”与结果一致，但目前是结果后形成的解释。Rev 2 应在 BDDL/任务描述上冻结一个
与结果无关的复杂度定义，例如目标 predicate 数、需要顺序完成的子目标数或最长 horizon；最好在新 suite/
真机任务中也能同样计算。否则论文只能说“在 LIBERO-10 观察到，在 LIBERO-Spatial 未观察到”，不能
上升为“复杂任务普遍有效”。

### 9.3 Phase 0 的具体裁决

| 项 | 裁决 | 说明 |
|---|---|---|
| 0a always-MISS | **立即做** | 只作 exploratory anchor；新确认集须成为正式臂。建议显式命名 `always_full_inference`，避免读者把 MISS 误解为不执行策略。运行后机械确认 FULL/WARM=0、MISS=100%。 |
| 0b spatial p95/p97.5 | **立即做** | 只用于选择 Rev 2 支撑区间。不要改写 Rev 1 fitter；用独立 exploratory exporter 从同一 frozen fit 导出，并在 artifact/meta 写 `posthoc_exploratory=true`。 |
| 0c HF provenance | **立即落盘，不再等待** | 已有 revision、文件 SHA 和 rollup；写入 data_authority，并保存实际目录的不可变 manifest。 |
| 0d phase audit | **并行做，但弱标签先验证** | 夹爪周期不等于已完成子目标：失败抓取、重抓、开关门都会破坏对应。先在少量轨迹上用视频/确定性 replay 验证 precision/recall，再报告 mismatch rate。 |

Phase 0 还应增加 **0e：Rev 2 cost-map calibration**。用已有 A′ 的三个 threshold、SV−/SV 和新增激进点，
拟合每 suite 的 δ→realized-cost 映射，机械选出能包围预定 cost anchors 的 SV δ；S0 若没有足够点，可在
旧 A′ 补跑纯 exploratory S0 点。该步骤只看 cost/tier mix，禁止用 SR 调点。

### 9.4 关于 always-MISS 的实现边界

`threshold > 1` 在 cosine score 理论上可达成恒 MISS，但正式协议不应只靠隐含数值范围。最低要求是：

- yaml 写明 `judge_role: always_full_inference_anchor` 或 matrix 中等价的冻结 role；
- dry validation 明确验证 threshold 高于检索 score 的规范上界、无 warm tier、always_search gate；
- analyzer 机械要求所有实际 verdict 均为 MISS，否则整臂拒绝；
- 成本声明区分 analytic GPU compute（67.52 ms）与仍发生 search 的端到端延迟。

不需要为了一个 anchor 新增生产 judge 类型，但不能只靠文件名约定其语义。

### 9.5 fresh sampled init 能否成为确认集

**可以，条件如下：**

1. 由 LIBERO 原生 init generator/BDDL placement distribution 产生，不手工筛除难例或不可达例；生成代码、
   seed 序列、拒绝/重采规则和环境/资产 SHA 在跑任何臂前冻结。
2. 与 D_lib、D_dev、旧 A′ 做 simulator-state content digest 去重；同一新 pool 在全部臂间严格配对。
3. 先以 always-full-inference 在一个独立 pilot pool 检查生成分布有效性；pilot pool 不进入确认分析，也不
   用其 SR 选择 δ。
4. Rev 2 confirmation pool 一次性封存，禁止 Phase 0/1 脚本读取 outcome；所有 arm/matrix/analysis code 先
   完成 G1/G2 和 synthetic dry run。
5. 论文准确写成 `independently sampled initial states for the same task distribution`，不称 official benchmark
   test set。官方 A′ 的 Rev 1 负结果仍完整报告。

满足这些条件后，它在统计上是合法的新确认样本，足以确认同任务分布上的 Pareto claim。为了支撑 ICLR
主张，还需要至少一个真正分布外支柱：真实机器人多阶段任务，或 libero_object/goal 中的一套。两者不必
都做；既然 owner 已计划真机，我建议优先真机，而不是立刻投入 1–2 天重建第三套模拟链。

### 9.6 推荐的实际执行顺序与止损

1. **立即归档 Rev 1**：运行 cross-suite finalizer 产生正式 negative summary；冻结本文、verdict、journal、
   per-step 与环境资产 manifest。不得覆盖旧结果。
2. **Phase 0 并行**：0a/0b rollout；0c provenance；0d phase audit；0e cost-map。全部仍用旧 A′并标 exploratory。
3. **Decision Gate A（不烧新确认集）**：
   - l10 的 SV frontier development AUC 必须为正，且优势不能只来自一个 task；
   - spatial 激进点若仍输，接受 spatial 为 negative/control，不要求它过主 Gate；
   - 必须找到 SV/S0/threshold 的共同 cost support，否则 v claim 暂停；
   - always-full-inference 必须给出合理 exact-stack baseline，否则先查 harness/asset。
4. **冻结 Rev 2 G1/G2**：唯一 primary estimand、δ/锚点、arm roster、fresh-init authority、功效与止损全部
   写死；代码生成不可变 artifacts/matrices 后才 materialize confirmation pool。
5. **先跑 l10 fresh-init confirmation**。它是当前效应最强、最能检验假说的 suite。若 AUC lower bound
   `≤0`，停止 surface 主张，不再用 spatial/secondary/真机追着找正结果。
6. **l10 通过后再跑 spatial negative-control 与真机**。spatial 用于边界条件，不强求胜；真机必须预先
   区分 multi-stage 与 simple tasks，并使用同一 frontier protocol。
7. **Phase-aware retrieval 单独立项**：只有 0d 的独立 phase mismatch audit 有效且 effect 明确才进入；
   不得在 Rev 2 surface confirmation 中临时换 retriever，否则无法归因已有的 v 信号。

### 9.7 最小可发表版本与完整版

**最小可发表版本**：完整报告 Rev 1 negative；Rev 2 在独立 fresh-init l10 上确认 surface-vs-threshold
frontier AUC；S0 matched-cost 消融确认 v；spatial 作为 near-ceiling negative/control；真实机器人至少覆盖
一个多阶段任务族和一个简单任务族。

**完整版**：在最小版上增加 validated phase audit 与 phase-aware retrieval 2×2 消融，形成“candidate
quality × dispatch uncertainty”完整系统故事。

我的建议是先追求最小版。当前最危险的不是结果差，而是同时开启 frontier、history、phase key、扩库、
secondary 五条线把方法贡献做散。现有数据已经给出一条足够集中的主线：**v 在复杂/长程任务中改善
matched-cost ranking；Rev 2 用公平 frontier 和独立确认集验证它。**

---

## 10. Execution Authority 对 §9 的执行记录（2026-08-29，不含任何新 rollout）

> §9 裁决全部接受，Rev 2 草案 v0 已按 R2-B1…B4 重写：[`dispatch_surface_rev2_protocol_draft.md`](dispatch_surface_rev2_protocol_draft.md)。
> 下面是不需要确认池、也不需要 rollout 的项的完成情况；需要 rollout 的项（0a/0b/0e）已备好但**等 owner 放行**（此前 owner 指示暂停）。

### 10.1 Rev 1 归档（§9.6-1）✅

`finalize_cross_suite --verdict l10 --verdict spatial` → **`suite_specific_only`**，failed = `libero_10.gate1, libero_spatial.gate1`，gate2 均 `not_evaluated_fixed_sequence`。
产物：`exp/dispatch_surface/data/aprime_rev1/cross_suite/finalize_rev1.json`；verdict / journal / launch / per_step / fit_record 本地副本齐全（§6）。旧结果未覆盖。

### 10.2 Decision Gate A-1：development AUC（§9.2-R2-B3 的 estimand 在旧 A′ 上试算）✅

frontier = 三个 threshold 点 vs 两个 surface 点（SV−、SV）按 realized decision-weighted cost 连成 piecewise-linear；共同区间取两条线的交集；task-stratified paired cluster bootstrap 4000 reps：

| suite | 共同区间 (ms) | AUC diff（surface − threshold） | 单侧 95% LB | q0.95 | support-miss | leave-one-task-out |
|---|---|---:|---:|---:|---:|---|
| **libero_10** | [39.72, 48.83] | **+0.0753** | **+0.0304** | +0.1196 | 0/4000 | 全正：+0.041 … +0.097 |
| libero_spatial | [37.28, 43.24] | −0.0035 | −0.0238 | +0.0166 | 0/4000 | 全部 ≤ +0.001 |

- l10 满足 A-1（LB > 0；LOTO 全正的含义是**删除任一 task 后聚合 AUC 仍为正**，不是每 task 为正——每 task 的 descriptive AUC 见 §10.7）；spatial 按当前证据是 negative control。
- 各臂 realized cost 的 bootstrap SD 0.4–0.7 ms ⇒ 锚点距区间端点留 2 ms（≈3σ）即可把 support-miss 压到可忽略；Rev 2 草案 §3/§4 据此定 `m = 2.0 ms`、`[c_L,c_H] = [41.0, 47.5]`、`c_1,c_2 = 42.0, 46.5`（⟨待裁⟩）。
- 功效：dev 上 AUC diff 的 SD ≈ 0.027；效应复现时 N=30 功效 ≈ 0.88、N=40 ≈ 0.94；效应减半时 N=40 ≈ 0.55——写进草案 §9。

### 10.3 R2-B4 复杂度：BDDL 机械提取（**按执行 task id 重建**；v0 的表按文件名字母序排列、未按 task id join，task 5 / 9 写反——§12.1-5 指出，已修）

| task | libero_10 | goal atoms | n |
|---|---|---|---|
| 0 | LIVING_ROOM_SCENE2 both soup + sauce | In, In | 2 |
| 1 | LIVING_ROOM_SCENE2 both cream cheese + butter | In, In | 2 |
| 2 | KITCHEN_SCENE3 stove + moka pot | Turnon, On | 2 |
| 3 | KITCHEN_SCENE4 bowl → drawer + close | Close, In | 2 |
| 4 | LIVING_ROOM_SCENE5 two mugs → two plates | On, On | 2 |
| 5 | STUDY_SCENE1 book → caddy | In | 1 |
| 6 | LIVING_ROOM_SCENE6 mug → plate + pudding | On, On | 2 |
| 7 | LIVING_ROOM_SCENE1 both soup + cream cheese | In, In | 2 |
| 8 | KITCHEN_SCENE8 both moka pots (+Turnon) | On, On, Turnon | 3 |
| 9 | KITCHEN_SCENE6 mug → microwave + close | In, Close | 2 |

libero_spatial 十个 task 全部 `On(akita_black_bowl_1, plate_1)`，n = 1。horizon（520 / 220）是 suite 评测预算，按 §12.1-5 不作任务结构指标。复杂度只作 descriptive heterogeneity（Rev 2 v1 §6）。

### 10.4 0d：夹爪周期弱标签验证（§9.3 的前置条件）— 三版对比，第三版可用但有已知偏差

在 l10 D_lib 的 50 条轨迹（47 条成功）上，把标签给出的"闭合周期数"与任务应有的抓放次数比（both/and 双物体 = 2；stove = 2，含转旋钮；其余 = 1）：

| 标签 | 定义 | 成功轨迹上精确一致 | 失败模式 |
|---|---|---:|---|
| v1 | 指间距 < (min+max)/2 | 32/47 | **系统漏计**：同一 episode 抓不同宽度的物体，宽物体的闭合（间距 1.14）高于中点阈值——task 6、task 7 的成功轨迹 0/9 一致 |
| v2 | 动作里的夹爪指令 > 0.5 | 22/47 | 指令抖动 / 重抓，严重多计（最高 17） |
| **v3** | 相对张开位（p95 间距 ≈ 1.96）**绝对下降** ≥ 0.7 视为闭合，回升到 −0.3 视为张开 | **42/47** | 剩余 5 条全是**多计 1**（真实重抓），无漏计 |

结论：v3 作为"已完成抓取次数 ≥ k"的**单调进度代理**可用，已知偏差是重抓造成的约 10% 多计；不能直接当"已完成子目标数"。
按 §9.3 的要求，在进入 mismatch audit 之前还需要对这 5 条多计轨迹做一次视频 / 确定性重放核对（H5 里有 base + wrist RGB，可离线做），确认多计的确是重抓而不是标签错。执行方未开始 audit 本身。

### 10.5 0c：HF assets provenance 台账 — 草稿已就绪，登记被 `KNOWN_KINDS` 挡住

补齐（§12.1-1）：完整 rollup sha256 = `9bc4f946f44d5aefb09848484ba9d312b5f9464b1dc38403f54736867e090e14`（1758 文件、375,103,669 bytes，`find -type f | LC_ALL=C sort | xargs sha256sum | sha256sum`）；HF API `sha = 90001343cb134b7e26e18fde0fa2416f3ed6e6a3`，`lastModified 2025-11-03T09:43:10Z`，**repo card 未声明 license**（`cardData.license = null`，目录内无 LICENSE / README）——记录必须写 `license: undeclared`，并列为 spatial 结果的 provenance limitation。

`exp/data_authority/registry.py:62` 的 `KNOWN_KINDS` 没有能描述"外部仿真资产"的 kind（现有：cache_artifact / collection_h5 / journal / checkpoint / init_pool / benchmark_results）。
写入 `records/` 会让 `registry validate` 全局报错，因此记录先放在这里，**加 `external_asset` kind + 落记录**列入 Rev 2 草案 §10 的代码清单：

```json
{
 "schema_version": 1,
 "dataset_id": "dispatch_surface/libero_spatial/libero_assets_hf",
 "kind": "external_asset",
 "title": "LIBERO simulation assets (jadechoghari/libero-assets) as loaded by timan107 workers for the Rev 1 libero_spatial primary run",
 "experiment": "exp/dispatch_surface", "suite": "libero_spatial", "status": "authoritative",
 "authority": {"node": "timan107", "path": "/srv/local/zixuans8/tether-home/.cache/libero/assets", "access": "tether"},
 "integrity": {"file_count": 1758, "size_bytes": 375103669, "sha256": "9bc4f946f44d5aefb09848484ba9d312b5f9464b1dc38403f54736867e090e14",
               "members_sample": [["scenes/libero_tabletop_base_style.xml", "bb118118e3e02a0f23295c072d77c5a7c6b44cc773032dc79c17070b4e133f86"]]},
 "provenance": {"hf_repo": "jadechoghari/libero-assets", "hf_refs_main": "90001343cb134b7e26e18fde0fa2416f3ed6e6a3",
                "downloaded_at": "2026-08-28T23:08:42-05:00", "downloader": "libero/libero/utils/download_utils.py:download_assets_from_huggingface"},
 "consumers": ["exp/dispatch_surface/data/aprime_rev1/libero_spatial_primary"],
 "license": "undeclared (HF cardData.license=null as of 2026-08-29; no LICENSE file in snapshot)",
 "caveats": ["libero_10 primary did not touch this directory (KITCHEN/LIVING_ROOM scenes do not load libero_tabletop_base_style.xml).",
             "License not yet recorded; the HF repo card must be captured before the record is marked authoritative."]
}
```

### 10.6 等 owner 放行的 rollout（§9.3 说"立即做"，但 owner 此前指示暂停）

| 项 | 规模 | 需要 |
|---|---|---|
| 0a `always_full_inference` × 2 suite | 600 ep ≈ 30 min | 重启 weilandserver server；emitter 加描述性臂（小改，见草案 §10-1/3） |
| 0b spatial SV p95 / p97.5 | 600 ep ≈ 30 min | 独立 exploratory exporter（草案 §10-2） |
| 0e S0 p80 + 激进点 × 2 suite | 1200 ep ≈ 60 min | 同上 |

三项合计约 2 h 算力。0a 与 0b/0e 都需要先过一次 G1/G2（它们是 Rev 2 代码），执行方不会在此之前跑。

### 10.7 R2G1-B1：libero_10 每 task 的 descriptive AUC 与 discordance（不作选择门）

每 task 30 个 init；frontier 为该 task 内的两点 SV 折线 vs 三点 threshold 折线；AUC 在该 task 的共同成本区间上归一化。

| task | AUC diff | 区间 (ms) | SR（SV−, SV \| fh30, fh50, fh70） | SV:fh70 | SV−:fh30 | 备注 |
|---|---:|---|---|---|---|---|
| 0 | +0.148 | [38.7, 43.9] | 0.93 0.40 \| 0.80 0.70 0.20 | 7/1 | 5/1 | |
| 1 | +0.022 | [25.2, 44.5] | 1.00 1.00 \| 0.93 1.00 1.00 | 0/0 | 2/0 | ceiling |
| 2 | **+0.212** | [45.5, 54.0] | 0.93 0.77 \| 0.77 0.67 0.40 | 14/3 | 6/1 | |
| 3 | +0.072 | [45.0, 45.9] | 0.77 0.57 \| 0.83 0.47 0.53 | 5/4 | 4/6 | 区间极窄 |
| 4 | **−0.064** | [43.9, 48.4] | 0.83 0.30 \| 0.70 0.43 0.30 | 6/6 | 7/3 | |
| 5 | +0.013 | [46.7, 48.6] | 0.70 0.63 \| 0.60 0.70 0.63 | 2/2 | 4/1 | 单目标 |
| 6 | **+0.425** | [43.4, 48.1] | 0.87 0.37 \| 0.70 0.30 0.27 | 5/2 | 7/2 | |
| 7 | **−0.192** | [39.0, 42.2] | 0.93 0.57 \| 0.87 0.83 0.50 | 5/3 | 3/1 | |
| 8 | −0.001 | [36.9, 48.3] | 0.10 0.03 \| 0.17 0.10 0.03 | 1/1 | 3/5 | floor |
| 9 | +0.002 | [45.8, 55.9] | 0.57 0.30 \| 0.60 0.47 0.17 | 8/4 | 5/6 | |

7 个 task 为正、2 个为负（4、7）、1 个 floor（8）；聚合 +0.075 主要由 task 0/2/6 贡献。按 §12.2-B1，本表只作 descriptive，"not driven by any single task" 的依据是 LOTO，不是本表。
