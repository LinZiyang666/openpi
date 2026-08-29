# ActionCache 之后的论文重定位方案（Codex 版）

> 日期：2026-08-26  
> 范围：不依赖 RoboCasa；假设既定数据最终可完成，并把真实机器人实验纳入正文。  
> 目标：不是从规则或措辞上“防住” ActionCache，而是通过可归因的实验把本文变成一篇回答不同科学问题的论文。

## 0. 总裁决

Claude 的方案抓住了两个正确支点：**在昂贵语言 backbone 之前退出**，以及**利用闭环命中的时间结构**。CP2 对照也值得做。

但以下部分不建议照搬：

1. **不把 arXiv-only 规则当 novelty 保护伞。** 规则也许减轻形式上的引用义务，却不会阻止审稿人据此判断增量性；我们应主动引用并正面对照。
2. **慎用 “independent”。** 只有在能够真实、匿名且可验证地说明两套系统独立形成时才使用；否则只写 “concurrent work”。当前针对 ActionCache 新增的重定位和实验不能暗示为独立于它形成。
3. **不把对方的弱点审计写成论文主线。** “demo 级”“披露真空”“叙事误导”等措辞即使技术上可辩，也会显得敌对。论文只陈述机制差异、匹配实验和实测结果。
4. **CP2 离线同库版本不能叫“忠实复现”。** 它改变了在线建库、淘汰和校准协议，应称为 **ActionCache-style post-backbone baseline**。除非运行作者官方代码或逐项复现协议并报告偏差，否则不要使用 faithful reproduction。
5. **四臂目前混杂多个变量。** CP2→CP1 同时改变了 tap point、表示形式、单/多字段检索和执行策略；即使赢了，也不能知道因为什么赢。
6. **标题中的 “before the vision-language backbone” 不够精确。** CP1 已运行 vision tower、projector 和 token embedding；真正跳过的是后续语言/多模态 transformer 与 action expert。正文和标题应避免让审稿人抓住架构口径错误。

本文应从“谁先做 experience cache”转为：

> **VLA 的闭环动作复用在多早的内部表示上已经可判定？这种可复用性是否具有可利用的时间块结构？**

方法只是对这个发现的利用：在 vision encoding 之后、昂贵 transformer 之前做 stateful early exit，命中时跳过后续 transformer 与 action generation。

---

## 1. 一句话 thesis 与贡献边界

### 1.1 推荐 thesis

> Successful robot rollouts contain temporally clustered action-reuse opportunities that are already identifiable before a VLA's expensive multimodal transformer; a stateful experience-reuse policy converts this early signal into a better closed-loop success--compute frontier.

### 1.2 三条贡献

1. **测量贡献：** 定义并测量成功率容差下的可恢复冗余 $R_{\mathcal C}(\varepsilon)$，同时报告冗余的 run length、stickiness 与 lockstep，而不把一个简单 supremum 包装成独立理论突破。
2. **方法贡献：** 提出 pre-transformer、stateful experience reuse；关键不是“模型自身表征做 key”，而是更早的退出位置和跨时间决策。
3. **实证贡献：** 用受控 tap-point 消融、两种 VLA、仿真和真机闭环实验，证明它相对 post-backbone per-step caching 与 reduced-NFE baseline 改善 success--compute frontier。

不再声称：first VLA experience cache、first own-representation cache、ActionCache 被本文 subsume、形式化 $R$ 本身构成主要理论创新。

### 1.3 标题候选

首选：

> **Muscle Memory for Robot Policies: Stateful Experience Reuse Before the Multimodal Transformer**

备选：

> **How Early Can a Robot Policy Stop Thinking? Measuring and Exploiting Action-Reuse Redundancy in VLAs**

第二个标题科学问题更清楚；第一个更保留项目品牌。

---

## 2. 与 ActionCache 的正确关系

### 2.1 正文定位

ActionCache 是头号 related work 和头号 baseline，而不是附录里的攻击对象。推荐正文口径：

> Concurrent ActionCache work retrieves successful action-generation states using post-backbone VLM representations and either replays or refines them. We study a complementary question: whether reuse decisions can be made before the expensive multimodal transformer, and whether their temporal dependence can be exploited by a stateful scheduler. Our controlled comparison isolates representation timing from key construction and scheduling.

是否保留 “concurrent” 由最终投稿时间线决定；“independent” 默认不写。

### 2.2 只做一张设计空间表

正文或附录用一张中性表比较：tap point、每步必付模块、key、payload、per-step/stateful、校准目标、在线/离线建库、仿真/真机。不给对方列“罪状”，不猜测其数据泄漏，也不跨硬件直接比较速度数字。

### 2.3 两种 baseline 名称严格区分

- **Published ActionCache：** 引用论文的机制和公开数字，仅用于文献定位。
- **ActionCache-style baseline (ours)：** 在我们的模型、库和硬件上实现的 post-backbone single-key per-step caching，用于受控对照；明确列出与原论文在线建库、eviction、投影和 refinement 的异同。

只有官方实现逐项跑通且协议一致时，才称 “reproduction”。

---

## 3. 生死实验：把变量拆开

Claude 四臂可作为工程骨架，但论文主实验应改成以下最小因果链。

| ID | Tap point | Key / search | Scheduler | Payload | 回答的问题 |
|---|---|---|---|---|---|
| T | 无 cache | — | — | teacher | 质量与成本锚点 |
| N | 无 cache | — | — | reduced NFE | 不使用经验时，直接 cheapen teacher 能否支配我们 |
| P2-AC | Stage 2 后 | ActionCache-style 单 key、固定投影、cosine | per-step threshold | direct / matched warm-start | post-backbone caching baseline |
| P1-ISO | Stage 1 后 | 与 P2-AC 尽量同构的单 key、投影、cosine | per-step threshold | 与 P2-AC 相同 | **只隔离 tap point** |
| P1-MF | Stage 1 后 | 当前多字段加权 key | per-step threshold | 与上相同 | 多字段 key 的增量 |
| P1-STATE | Stage 1 后 | 与 P1-MF 相同 | hysteresis/stateful | 与上相同 | 时间状态的增量 |
| P1-BLOCK | Stage 1 后 | 与 P1-MF 相同 | block/lockstep | direct replay | 激进上界与失败模式；结果脆弱则放附录 |

### 3.1 为什么 P1-ISO 必须新增

没有 P1-ISO，P2-AC 与现有 CP1 的差异无法归因：早表示可能更弱，但多字段融合可能更强。P1-ISO 是整篇最重要的消融；它比多跑若干 suite 更能回答 novelty 问题。

### 3.2 统一控制变量

- 相同 teacher checkpoint、成功轨迹、entry 数、payload 和 action horizon；同时报告字节容量。
- 所有阈值只在 calibration split 上选择，最终 test split 冻结后一次性评估。
- 主图按相同 $\varepsilon$ 或相同闭环 SR 对齐，不比较原始 cosine 阈值。
- P2-AC 与 P1-ISO 使用相同校准器；P1-MF/P1-STATE 的新增自由度单独披露。
- 编译档、batch、GPU、检索后端与并发度固定；eager 和 CUDA Graph 都报告，避免只选有利 regime。
- 真正的端到端计时必须包含 key 构建、投影、search、judge、通信和同步。

### 3.3 核心可证伪假设

- **H1（early identifiability）：** P1-ISO 在合理 $\varepsilon$ 下存在非零可恢复冗余；若为零，“早期可判定”主张失败。
- **H2（early-exit value）：** P1-ISO 相对 P2-AC 在同 SR 下减少实测 GPU-time 或提升 serving throughput；只比较命中步理论地板不算通过。
- **H3（key design）：** P1-MF 相对 P1-ISO 提升 $R(\varepsilon)$；若不提升，删除多字段 novelty，采用更简单实现。
- **H4（temporal value）：** P1-STATE 相对 P1-MF 改善端到端 success--compute frontier，而不只是生成更长 hit run。
- **H5（robustness）：** H1/H2/H4 至少在第二个 teacher 或真机上方向一致；不要求所有任务同幅度。

---

## 4. 指标与统计：主图只讲一件事

### 4.1 主图

横轴使用每个环境决策步的实测成本，纵轴使用闭环成功率：

- GPU-ms / decision step 或 GPU-s / successful episode；
- wall-clock latency 的 median、P95、P99；
- 并发 serving 下的 throughput；
- 同一图标出 teacher、NFE baseline、P2-AC、P1-ISO、P1-MF、P1-STATE。

理论 cost floor 仅作机制解释，不作为胜负证据。

### 4.2 冗余结构图

保留 $R_{\mathcal C}(\varepsilon)$、$P(H_t\mid H_{t-1})$、run-length survival 和 lockstep purity，但加入两个防伪检查：

1. 对命中序列做 episode 内 permutation 或保持 marginal hit rate 的几何/null baseline；
2. 分任务阶段或接触/非接触阶段报告，确认长 run 不是 action chunk 重叠或固定 replan 周期的机械产物。

### 4.3 统计

- 配对 init/seed，报告每臂的 episode 数和失败数。
- SR 差使用配对方法或 cluster bootstrap；成本差以 episode 为 cluster。
- 对主假设预先指定少量比较并校正；探索性十二臂不做一张显著性星号墙。
- 报告 effect size 与 CI，不以“未显著”推出等价；若需要 non-inferiority，预先固定 margin $\varepsilon$。

### 4.4 修正 $R(\varepsilon)$ 的表述

$R$ 是受限调度类和固定经验库下的**可达下界/经验前沿**，不是任务内禀常数。正文写清：

- 库、信息集、scheduler class 和 calibration budget 都是 $R$ 的下标或实验条件；
- 不用 unrestricted oracle library 定义“真正的冗余”，避免退化；
- oracle sim-state 只叫 diagnostic upper reference，不声称精确分解 coverage error 与 key error；
- 单调性命题放附录或删除，不把显然的 feasible-set inclusion 当理论贡献。

---

## 5. 真机实验：正文证据，不是演示视频

真机应验证仿真最难替代的三件事：视觉扰动、控制误差累积、连续 replay 后的闭环恢复。

### 5.1 主臂

资源允许时跑四臂：

1. teacher；
2. reduced-NFE teacher；
3. P2-AC；
4. 最终选定的 P1-STATE。

P1-ISO/P1-MF 的因果消融主要留在仿真完成；真机不必把所有工程臂重跑一遍。

### 5.2 协议

- 至少三个具有不同接触模式或时长的任务；每臂每任务尽量 30–50 次，若不足则诚实给 exact binomial CI。
- 初始物体位姿随机化并为各臂配对；运行顺序交错，避免电机温度、光照和操作者熟练度与方法混杂。
- cache 构建 episode 与测试 reset 分离；明确是否使用相同物体实例、背景和相机位姿。
- 报告成功率、干预率、端到端 latency、backbone-call rate、最长连续 replay、失败发生阶段。
- 附失败分类：错误检索、时序漂移、接触失败、感知突变、执行器误差；至少展示典型成功和失败轨迹。

### 5.3 真机 headline 的通过条件

不是“cache 也能完成任务”，而是：

> 在真实扰动下，P1-STATE 相对 P2-AC 或 reduced-NFE baseline，在预设成功率容差内减少后续 transformer 调用和实测端到端成本。

如果只完成少量 trial 或没有受控 baseline，应降级为 qualitative demonstration，不据此抬高主结论。

---

## 6. 论文结构

1. **Introduction：** VLA 成本由 vision encoding、multimodal/language transformer 和 action generation 构成；问题是 reuse decision 能否在昂贵后两段之前做出。
2. **Related work：** VLA acceleration、action caching、KV/token reuse、adaptive compute；ActionCache 放最近工作第一位。
3. **Problem and measure：** 固定库/信息集/调度类下的 $R_{\mathcal C}(\varepsilon)$，作为评价协议。
4. **Where is reuse identifiable?：** P2-AC vs P1-ISO vs P1-MF，回答表示位置问题。
5. **Temporal structure and method：** stickiness/lockstep 发现 → stateful scheduler；先证发现，再介绍利用。
6. **Closed-loop evaluation：** 两种 teacher、LIBERO、真机、端到端与 serving 指标。
7. **Limitations：** cache coverage、分布漂移、接触阶段风险、storage、隐私与失败经验未利用。

把 serving lifecycle、学生模型防守、router、oracle、cache-size 等材料按是否直接支撑三条贡献裁剪；不是每个已有实验都必须进正文。

---

## 7. 执行优先级与 kill criteria

### P0：先锁故事，不先扩实验面

1. 冻结三条贡献和术语：`pre-transformer`, `stateful`, `success--compute frontier`。
2. 将 ActionCache 称为 concurrent work；删除 first/own-representation 优先权主张。
3. 明确 Stage 1 已包含 vision tower，统一成本分解和标题口径。

### P1：最小生死包

1. 接通 P2-AC。
2. 新增 P1-ISO。
3. 在 π0.5 LIBERO Spatial 上完成 T/N/P2-AC/P1-ISO/P1-MF/P1-STATE。
4. 同时完成 eager 与 CUDA Graph 的组件级和端到端计时。

**Kill 1：** 若 P1-ISO/P1-STATE 在同 SR 下均不能优于 P2-AC 或 reduced NFE，则停止宣传 early-exit 方法优势；论文只能退为 redundancy measurement，ICLR 风险很高。

### P2：外部有效性

1. 在 LIBERO-10 或另一 suite 复核主趋势。
2. 在 GR00T 上至少复核 tap-point 成本结构与一个完整闭环 frontier。
3. 完成真机四臂主实验。

**Kill 2：** 若 early-exit 只在一个 teacher/一个短任务成立，结论收窄为特定 architecture/regime，不写一般 VLA 结论。

### P3：加分项

- block/lockstep 激进端点；
- cache-size 曲线；
- concurrent serving；
- transfer；
- online commit/eviction faithfulness check。

这些不能抢占 P1/P2 和真机预算。

---

## 8. 结果分支与论文去留

| 结果 | 论文解释 |
|---|---|
| P1-ISO 比 P2-AC 省成本但 SR frontier 较差，P1-STATE 补回 | 最理想：早退出有信息损失，时间策略使其可用 |
| P1-ISO 已全面优于 P2-AC，stateful 增益小 | 主线改为 tap-point discovery；删除强 stateful claim |
| P2-AC 在高 SR 区更好，P1-STATE 在高压缩区更好 | 诚实画 regime map；贡献是扩展可达区域 |
| reduced NFE 在所有轴支配 caching | 方法主张失败；只能研究经验复用在不同硬件/模型成本结构下何时有价值 |
| 真机方向一致但 CI 宽 | 作为外部有效性证据，不宣称确定提升 |
| 真机失败集中在连续 replay | 将 block replay 定位为风险诊断，突出 stateful reset/fail-closed 机制 |

摘要只写实际通过的假设，不为每个可能结果预先准备“都能赢”的解释。

---

## 9. 对 Claude 方案的采用清单

直接采用：

- pre-transformer early-exit 主轴；
- CP2 post-backbone 对照；
- 端到端成本而非 action-head-only 数字；
- 按 $\varepsilon$ 对齐、配对评测、bootstrap CI；
- temporal run-length/lockstep readout；
- 真机升级为对照实验。

修改后采用：

- 四臂扩为可归因的 P2-AC → P1-ISO → P1-MF → P1-STATE 链；
- “忠实复现”改为 “ActionCache-style baseline”；
- “before VLM backbone”改为 “before multimodal transformer”；
- $R(\varepsilon)$ 从理论 headline 降为评价协议；
- ActionCache 弱点清单改为中性设计空间表和 baseline 结果。

不采用：

- 用 arXiv-only 条款推导“无需打赢它”；
- 默认使用 “independent work”；
- 附录逐项检控对方披露与统计；
- 用理论地板代替同硬件端到端胜负；
- 把 RoboCasa 放在当前关键路径上。

最终判断：Claude 方案可作为**危机识别和工程起点**，但不宜原样成为投稿策略。本文最强的姿态不是“我们为什么没有被 ActionCache 做掉”，而是“ActionCache 之后出现了一个自然但尚未回答的问题，而我们用受控实验和真机证据回答了它”。
