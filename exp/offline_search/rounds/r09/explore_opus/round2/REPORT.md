# 长任务上为什么"处处微调"没用，以及"掉队就整段交给大模型"

> 本文件由协调者代存：opus 子代理的会话不允许写报告文件，全文由其交接消息转录，内容未改动。

## 给负责人的结论（白话）

**一句话。** 长任务（LIBERO-10）上，纯缓存失败时自己并不知道失败了：它照着示范的"剧本"按时走完，然后在结尾处原地空转到超时；而它成功的回合又很"脆"，任何一直开着的改动（包括上一轮的动作修正器）都会把其中一成左右弄坏。所以长任务需要的不是"处处微调"，而是"平时一点不碰，一旦掉队就整段交给大模型"。这个方案的闭环实验臂已经冻结、通过本地插件自检，等你安排运行。

**这轮做了什么。** 只用初始状态 0–29；凡是需要定参数的地方只用 0–19，20–29 留作评估（要申请的闭环也只用 20–29）。
1. 把历史上所有轮次里"同一个控制器、同一个初始状态"的重复运行凑到一起：扫描了 1,163 个实验臂、16 万个回合，纯缓存每格有 6–7 次独立重复。这样每个初始状态在纯缓存下的真实成功概率就能估出来——之前的分析都只看单次运行。
2. 用它回答"修正器为什么在短任务有效、在长任务无效"。
3. 找到一个不看任务编号的"掉队"信号，做成可直接上闭环的方法，写了 14 个测试，本地插件自检全部通过。

**发现一：长任务上纯缓存本身就很"抖"。** 两次完全相同的纯缓存运行，300 个初始状态里就有 31 个（π0.5）/ 37 个（GR00T）结果不一样。单次运行里看到的"+38/−34"必须和这个噪声比。修正器改变的结果数（76–81 个）是噪声的两倍多——它不是没起作用，而是有好有坏。

**发现二：修正器在长任务上为什么没用。** 把初始状态分成"纯缓存每次都失败"和"每次都成功"两类来看：
- 短任务（Spatial，π0.5 五十条示范库）：修正器救回了 71% 的"每次都失败"，只弄坏了 3% 的"每次都成功"，净增 11 个百分点。
- 长任务（π0.5 / GR00T 五十条示范库）：救回 46% / 33%，但弄坏了 11% / 12%，净增只剩 3 / 2 个百分点。
- 被弄坏的恰好集中在"随便什么扰动都会弄坏"的初始状态上。一个毫无意义的扰动（把看图时刻平移 5 步）在长任务上也会弄坏 11%、碰巧救回 21%。π0.5 长任务的成功回合里将近一半是"一碰就倒"的；短任务约三分之一，而且修正器在短任务上几乎不碰它们。
- 结论：长任务上，任何对每个回合都生效的改动都要付这笔"弄坏账"。只在出问题时才动手的做法（现有"卡住就叫一次大模型"的保护只弄坏 3–4%）不用付。本轮另外两位研究者的闭环结果也印证了这一点：更强的修正器在长任务上反而掉 9–20 个百分点。

**发现三：长任务失败长什么样。** 失败回合里有 91% 在大约第 230 个控制步（总共 520 步）就走到了示范的结尾帧——和成功回合到达结尾的时间完全一样。也就是说，抓空以后它照样"搬运空气、放下空气"，把剧本走完，然后在结尾处空转到超时。"每次都失败"的初始状态里一半以上是抓空（π0.5 有一半是抓第二个物体时抓空）。成功回合并不比大模型慢（中位数都是 253 步），所以不是"太慢"的问题。

**发现四：一个准、但偏晚的"掉队"信号。** 每次看图时算"现在第几步 − 检索到的示范在这一帧是第几步"，也就是落后示范节奏多少步。成功回合一直跟得上（落后约为 0），失败回合在剧本走完后越落越多。规则：落后达到 12 个决策（60 个控制步），并且不晚于第 80 个决策（第 400 步）。在只用于定参数的初始状态 0–19 上：π0.5 五十条示范库抓到 95% 的失败，误报 15% 的成功回合；GR00T 是 99% / 15%；五百条示范库误报只有 4–6%。缺点是晚：中位数出现在第 190–280 步。现有"卡住保护"的信号早但不准——在 89–98% 本来会成功的回合里也会响——所以不能简单地把它改成"一响就一直交给大模型"。

**主推方案：掉队后整段交给大模型。** 平时完全是纯缓存，一行不改；掉队信号一响，之后每 10 步调一次大模型，直到回合结束。成本只花在 15–45% 掉队的回合上，正常回合完全不碰，也就不付"弄坏账"。
- 唯一的未知数是"接手以后大模型能救回几成"。已有数据：纯缓存在掉队后自己能恢复 17–29%；带三成调用的方法恢复约三到五成；外推到完全接手大约 55–65%。
- 预注册的预测（初始状态 20–29，救回率取 .45–.65）：

| 格子 | 纯缓存 | 掉队接手（成功率） | 推理占比 | 现有做法（初始状态 0–29） |
|---|---|---|---|---|
| π0.5 长任务、五十条示范 | .73 @ .076 | .79–.87 | 约 .16 | 卡住保护 .84 @ .18；纯大模型 .91 @ .50 |
| GR00T 长任务、五十条示范 | .60 @ .074 | .74–.83 | 约 .19 | 卡住保护 .73 @ .22；纯大模型 .87 @ .50 |
| π0.5 长任务、五百条示范 | .83 @ .077 | .88–.92 | 约 .12 | 均匀调用 .92 @ .19；纯大模型 .91 @ .50 |
| GR00T 长任务、五百条示范 | .87 @ .075 | .89–.92 | 约 .12 | 卡住保护 .87 @ .19；纯大模型 .87 @ .50 |

- 置信度：五百条示范的两格和 GR00T 五十条示范这一格较高——只要救回率不低于 .45，就比现有做法更省而且成功率不低；π0.5 五十条示范这一格中等——救回率要到 .6 左右才明显超过现有卡住保护。
- 两个变体也一起冻结了：只接手 120 步然后还给缓存（省钱版，预测成功率低几个点、推理占比约 .13–.14）；叠加在现有卡住保护之上（π0.5 增益较小，GR00T 增益较大，因为 GR00T 的单次调用救回得少）。

**需要跑的闭环。** 16 个臂，每臂 100 对（任务 0–9 × 初始状态 20–29），分三批；均已冻结、通过本地插件自检，依赖清单已生成，未启动。估计每批 25–40 分钟。具体命令见 `HANDBACK.md`。

**不建议做的。** 按"相位"约束检索（回跳是失败之后才出现的现象，不是原因）；在 Spatial 上做掉队接手（信号来时只剩约 50 步，缓存自己恢复率只有 1%）；把现有卡住保护改成"一响就一直接手"（几乎每个回合都会被接手，等于纯大模型）；找更早的节奏信号（在误报不超过 16% 的前提下找不到）。

**和另外两位研究者的关系。** fable 的"抓空就立刻叫两次大模型"在闭环上已经 +5 个百分点、几乎不加成本。它早而便宜，专门对付抓空；我的掉队接手晚但兜底，能接住各种失败。两者可以叠加：用 fable 自己的闭环轨迹模拟，.79 → .79–.86。建议两边的筛选都出结果后再跑叠加版。astra 本轮做的是修正器的共享版、夹爪修正和只用手腕相机，和我的方向不重叠。

**诚实的限制。** 离线模拟只把"信号响之前"的部分算准了（那段轨迹就是纯缓存本身，是实测数据）；"接手以后能救回几成"必须靠闭环测。100 对的筛选只能看方向和救回率，非劣效检验要留给你用初始状态 30–49 做。

---

## Technical summary
Plain names: *pure cache* is the frozen BlindAWM with a ten-control commit. *Pure policy* calls the policy every 10 controls. *Guards* is the deployed four-guard controller; *only no-progress* keeps one of those guards. *Uniform calls* is ρ ≈ .3 plus the stall trigger. *Corrector* is round 1's half-strength residual head. Numbers are in `DATA_ANALYSIS.md`, arms in `PROPOSALS.md`, run instructions in `HANDBACK.md`.

**Tools.** All readers are discovery-only; 14 unit tests.
- `catalog`: scans every run root and catalogs arms and outcomes.
- `episodes`, `serverledger`, `r8ledger`, `forensics`: per-decision ledgers from standard client logs, standard server logs, the R8 debug records and the R8 forensic labels.
- `replicates`: pooled-replicate decomposition (churn, always-fail / always-succeed, stratified gains).
- `anatomy`: rescue/break rates, fragility, failure labels, script exhaustion, recovery after the trigger, onset timing.
- `triggers`: the trouble signals plus the exact-prefix escalation simulator.
- `preregister`: frozen screen predictions.
- `prepare_arms`, `deploy_sources`, `screen_analysis`: artifacts and arms, source install for the coordinator, and the screen readout including measured r.

**What offline evidence cannot say here.** The pre-trigger part of every prediction is measured data. The post-trigger rescue rate is an assumption, bracketed by observed partial-policy recoveries; a full takeover from a messy state may behave differently. No action-error proxy is used anywhere, and the screen reports r directly alongside the paired SR/IR change.
