# 论文方向重议纪要（进行中）

> 状态：讨论中，收敛后写正式 outline。旧版提纲/实验设计已冻结为 `.old`。
> 起始：2026-08-22，owner 发起。本文档 2026-08-22 晚经 owner 指示整体重构（原时序式追加 §1–§14 重组为逻辑结构，内容无删减；工作提纲 v0.1 正式入档 §5）。

## 1. Owner 判决：放弃旧 thesis，回归 cache 本身

旧 TIER 叙事（「经验库的价值在索引不在 payload」+ replay/student/teacher 三层派发）被判为**为写 ICLR 而写**的过度承诺：当时发现场景数据微调的小模型（ACT 等）搭配 cache 效果更好，急于把小模型纳入系统显得高大上，导致整篇空洞。新方向 = **回归给 VLA 做的 cache 系统本身**，正面 defense「为什么不用小模型 / 为什么不弱于小模型」。

## 2. Novelty 叙事（owner 已拍：A 为骨架，B 为惊喜章节）

- **A（骨架）— 时间冗余度叙事**：「VLA 闭环推理里有多少步是重复劳动？」主 claim = 闭环 VLA serving 存在大量可回放的重复计算，teacher 自身表征能以近零成本（4ms 检索、7.2e-05 相对成本）检出它们，回放而 SR 不降。novelty 三件套：① **发现**——冗余度高且有结构（块状粘滞 + lockstep + 上一步分数预测本步 AUC 0.97+；gate 线数据现成，已复核见 §6.2）；② **原语**——experience replay 是与蒸馏/量化/减步/token-cache 正交可叠加的第四加速原语；③ **frontier**——SR × compute 曲线 + 库规模数据成本曲线（E3 现成）。
- **B（惊喜章节）— 经验可迁移性**：场景 A 建库、held-out 场景 B 直接用（RoboCasa365 线在跑；owner 报初步观察见 §7.1-4）+ 机制 model-agnostic（pi0.5/GR00T 双 teacher）。结果好可升 headline；不好有退守句（新场景重采重建便宜）。
- C（免训练置信信号）：降为 method 一段，不做主轴——与 UQ/failure-detection 正面撞车，novelty 难守。

**Owner 加粗点——"teacher 自身表征"是与前人 cache 的头号区分**：RT-Cache 等前人做 cache，key 全部来自**外部 feature model**（现成视觉 encoder）；我们**直接从 teacher inference 中间抽 key**。三重红利：① key 是推理的字节副产品（前半段 forward 本来就要跑，key 近乎免费）；② 表征与动作生成同源，是 policy 自己"看世界的方式"，与决策的相关性外部 encoder 不具备（内部 vs CLIP 的 AUROC/$\varepsilon{\to}\delta$ 旧证据可复用）；③ teacher 更新时 key 空间自动跟随，零额外维护。

**必须诚实处理的雷**：强配置下纯回放有 SR 损失（老数据）。A 叙事下转为素材——「冗余度不是 100%，所以 gate/threshold 是核心部件而非附件」，顺势把 gate 线扶正为正文贡献。

**Related-work 必答**：RT-Cache（最近亲，检索回放）的 delta = 闭环 VLA / 内部表征 key / gate+threshold 校准 / SR 约束下的 compute frontier / 跨场景；VLA-Cache = token 级另一物种，一句话划开。血统与竞品划界详见 §4.3。

**"First" claim 纪律（2026-08-22 立，2026-08-26 因 ActionCache 修订为唯一版本）**：全文 first 句统一为——**"To our knowledge, the first formal definition and closed-loop measurement of the recoverable redundancy of VLA inference."**（definitional first，ActionCache 无形式量，无可争议。）旧三限定词系统层 first（serving-layer / closed loop / SR constraint）已被 ActionCache 打掉两个限定词、剩一个太窄，**废止**。系统层差异不再用 first 句表述，改用结构陈述："prior and concurrent caches key on backbone outputs and must run it every step; ours keys before the backbone and skips the entire remaining forward pass"（或 "the only such cache that can skip the backbone itself"）。

## 3. 应用故事、经济账与题目

**核心矛盾**：部署中的机器人是重复性动物（同一厨房/同一批任务/日复一日），VLA 却是无记忆的挥霍者——第 1000 次开同一个抽屉和第 1 次一样贵。**经验本应让推理变便宜，而当前 serving 栈里经验一文不值。** cache 把这笔浪费收回来：teacher 亲测成功的经验被自身表征索引，重复状态 4ms 回放、新状态才付全价。

**两极落地（owner 定调）**：

- **工厂集中 serving（算钱）**：几千台设备大量重复劳作，集中部署共享 GPU 池 + 共享库——中心集群只处理 miss 流量，命中不占 GPU，单卡可服务设备数放大 **$1/(1{-}h)$**（$h$=命中率；工厂重复度最高 → h 最高 → 经济账最漂亮）。提高设备利用率、降低推理成本，不必每台配昂贵边缘推理卡；一台的成功经验 append 入库全厂即刻可用（零梯度，对比 fleet learning 要聚合梯度）；产线换型 = 库重建（便宜），vs 小模型/router 重训（§7 弹药射程内）。
- **家用"越用越快"（算体验）**：家庭环境个性化且固定（你家的抽屉永远在那），h 随使用单调上升，用户可感知延迟改善——命中 = 本地毫秒级回放省掉云端往返；附赠隐私（重复场景不上云）。消费者语言，reviewer 也听得懂。

两极合构：fleet 端算吞吐/利用率，家用端算延迟/隐私，中间同一机制；$1/(1{-}h)$ 进正文当 serving 账本骨架（列定义与实测锚点见 §9）。

**切分修订（owner 拍板 2026-08-25）——正式结构从"工厂/家用"场景切分改为 latency × throughput 双轴**：

- **Per-robot：降延迟**——工厂或家里的单个机器人是同一件事：命中步延迟从"网络往返 + 全推理"塌缩到本地毫秒回放。⚠ 口径纪律：**平均与 P50 大降，miss 步（P99 tail）不降**——延迟分布双峰化而非整体平移，一律写 "lower average / hit-step latency"，禁写裸 "lower latency"。实验背书 = E0 微基准 +（若做）SO-101 真机命中/miss 延迟差。
- **Per-fleet：提吞吐**——结合集中部署趋势（ROSA/Armory 的规范语言），命中不占 GPU，robots-per-GPU = $1/\mathrm{IR}$，上限 6.6×。实验背书 = E11 mini-bench + 推算列。
- 改的理由：① latency × throughput 是 serving 文献的规范双轴，用对方语言说"我们两轴都加分"；② 每维有对应实验（旧"家用体验"无实验对应）；③ 修掉"2026 哪来的家用 VLA 部署"的软肋——单机器人降延迟不依赖应用想象。
- **旧切分的画面感保留为修辞**：抽屉金句、"越用越快"、隐私 bonus 照用于 intro 与叙述色彩，只是不再充当 economics 的正式结构。

**文献支撑（2026-08-22 web 调研）——集中部署经济账不是独木**：

| 文章 | 出处 | 与我们的关系 |
|---|---|---|
| **ROSA: A Robotics Foundation Model Serving System for Robot Factories** | arXiv 2607.01088（2026-07，Stanford+NVIDIA 系：Kozyrakis/Shuran Song/Narang 等） | 工厂论点的学术版：fleet 走网络访问 server-class GPU 池，卖点 = 省电池、提 GPU 利用率、优化「工厂级生产力而非单请求延迟」，报 **12.06× factory productivity**。作者阵容说明工业界在推 |
| **Armory: Action Chunk Scheduling for Batched Robot Policy Serving** | arXiv 2608.00337（2026-08，Georgia Tech） | 一块共享 GPU 服务多机器人、**amortize 计算开销**，10 台 AgileX PiPER 真机 fleet 验证；动机句即「功耗受限硬件跑不动 foundation-scale policy」 |
| **Offload or Overload: A Platform Measurement Study of Mobile Robotic Manipulation Workloads** | arXiv 2603.18284 | 实测：大 workload onboard 跑不动、大 onboard GPU 让电池快几小时耗尽，offload 是出路（带延迟/带宽代价的权衡测量） |
| **Characterizing VLA Models across XPUs** | arXiv 2604.24447 | on-robot 部署被实时 + 成本 + 能耗预算卡死——集中化的反面动机 |
| **Cross-Platform Scaling of VLA Models from Edge to Cloud GPUs** | arXiv 2509.11480（Asilomar 2025） | **对手论点**：右尺寸 edge 设备在功耗约束下可与老数据中心卡竞争 ⇒ edge vs cloud 争论真实存在；我们的位置 = cache 让集中方案的经济账单方面变好（$1/(1{-}h)$ 叠在调度收益上） |
| （经典底座，已在 [`../papers/cloud_edge_deployment.md`](../papers/cloud_edge_deployment.md)） | FogROS2 / SPO 2603.19418 / RoboOS / RaaS / Goldberg cloud robotics | 「lighter, cheaper, smarter」传统 + brain-cerebellum 分层 |

**Related-work 定位金句**：ROSA/Armory 全在优化**怎么调度需求**，没人动**需求本身可以被 cache 消掉**——serving 层有人铺路，我们是他们缺的正交杠杆；cache 命中不占 GPU，$1/(1{-}h)$ 的放大直接叠乘在他们的调度收益上。

**题目候选**（owner 已星标 2）：

1. "Practice Makes Cheap: Training-Free Experience Caching for Vision-Language-Action Models" — 谚语变体，直接说出核心 claim（练习→便宜），成本故事版。
2. ⭐ **"Muscle Memory for Robot Policies: Caching a VLA's Own Experience in Its Own Representations"** — owner 首选（2026-08-22）。隐喻贴合机制（自身经验+自身表征+不过脑），副标题带出 §2 加粗点。
3. "Don't Think Twice: Harvesting Temporal Redundancy in VLA Inference" — 最贴 A 叙事，"redundancy"偏系统味。
4. "Been There, Done That: Experience Replay as a Serving Primitive for Robot Policies" — 原语版，稍油。

## 4. 中心量 $R(\varepsilon)$：定义、数学、创新性、体量

### 4.1 三层定义（§11-2 已拍板）

**① 概念层**：一步推理是冗余的，当它产出的动作已被过去经验决定到"查表即可"的程度——teacher 的全量前向没有产生库里查不到的新信息。类比：**视频编码的帧间冗余**——MISS 步 = I-frame（全量计算），HIT 步 = P-frame（引用过去）。类比自带合法性。

**② 操作层**：**$\varepsilon$-可回收率 $R(\varepsilon)$** = 在「SR 下降 $\le \varepsilon$」约束下，某替换调度能达成的最大回放步比例。曲线不是单数（$\varepsilon$ 换 $h$）；threshold-pareto 数据天然是其采样。真上界不可计算 ⇒ 报告结构 = threshold 系统达成值（下界）+ 受限 oracle 臂（更强下界）+ 上界不可知的诚实声明；gap = 留给后人的空间，本身是贡献姿态。**闭环性内建**：回放改变后续状态分布 ⇒ $R(\varepsilon)$ 只能真实 rollout 测——一句排掉"离线算动作误差就行"的质疑，也解释前人为何没测过。

**③ 归属层**：$R(\varepsilon)$ 是 **(任务, teacher, 库, key) 四元组的属性**，非任务内禀。区分**内禀冗余**（矿藏总量：库无限大时的上确界，引言谈、不测）与**可回收冗余**（手里这张勘探图实际挖得出的量，正文测）；库越大**可回收的**冗余越多（E3 = 爬坡线）。挖矿类比与完整解读见 §4.2 ③。

**结构发现**：冗余不均匀，两种来源两形态——**时间粘滞**（冗余成块）与**轨迹重访**（命中段与库轨迹 lockstep）。两种冗余对应两种回收机制，撑起分析章（实测验证见 §6.2）。

### 4.2 数学表示与逐条解读（v3：公式 + 含义并写，2026-08-22 定稿）

> 风格纪律（owner 定）：每条公式必须带"逐块读 + 白话含义"，只写符号不写含义视为未完成——纪要如此，正文如此。

**§0 设定在干什么**：你有一个很强但很贵的 teacher（每步完整推理的大模型）。它在很多任务上其实在重复自己——同样的情境，反复推导出同样的动作。于是把它成功过的轨迹缓存下来：库 $\mathcal{L}$ 里每条是一个 (key, payload)，key 是 teacher 当时的内部表征 $\varphi(o)$，payload 是它当时的动作。跑新任务时，每步有个**调度器** $\sigma$ 做二选一：查库把近邻 payload 直接放出来（replay，便宜），或老实调 teacher 推理（infer，贵）。核心问题一句话：**在几乎不掉成功率的前提下，最多有多少比例的步能走 replay？** 下面四层公式就是把这句话严格化。

**① 概念层：什么叫"这一步是冗余的"**。设 $s_t \sim d_{\pi_T}$（teacher 自驾的到达分布——前 $t{-}1$ 步都是 teacher 在开车），步 $t$ 冗余当且仅当

$$\Big| P\big(Y{=}1 \mid \mathrm{do}(a_t = \hat{a}_t^{\mathrm{replay}})\big) - P\big(Y{=}1 \mid \mathrm{do}(a_t = a_t^{\pi_T})\big) \Big| \;\le\; \delta$$

逐块读：

- $Y \in \{0,1\}$ 是**整局**的结局（成功/失败），不是这一步的对错。
- $\mathrm{do}(\cdot)$ 是 Pearl 因果干预记号，念作"强行把第 $t$ 步动作掰成这个值，然后放手让后面照常跑完"。它与"观察到动作是这个值"不同——观察有混杂（teacher 出这个动作本身说明局面好），干预没有。
- 一般形式是 TV 距离，但 $Y$ 二值时退化成减法——即**两种做法的最终成功率之差**，直接用平实形式。

整条在说：如果第 $t$ 步偷偷换成库里检索来的动作，整局成功率变化不超过 $\delta$，那这步 teacher 的推理就没产生实质价值——它是冗余的。**为什么必须挂到 $Y$ 上**：前人的离线 metric 通常问"replay 动作和 teacher 动作像不像"，但动作不像可能殊途同归、动作很像可能在关键分叉点毁掉整局——只有把结局放进定义，"冗余"才是真正关心的那个东西。

**Remark（组合性破缺——关键，别跳过）**：**每一步单独可替换，推不出全部一起可替换。** 原因：上面的 do-干预是在 teacher 自己走出来的轨迹上测的——第 $t$ 步之前都是 teacher 在开车；真正部署时前 $t{-}1$ 步可能已全是 replay，你早已漂到另一条轨迹上，到达第 $t$ 步时的局面根本不是原来那个局面，库在那里的检索质量可能完全不同。干预会改变后续状态访问分布，误差会累积、会漂移。这条 remark 一箭双雕：(1) **操作量必须定义在调度器级别而非步级别**——步级冗余是好的概念定义，但不可加，不能把每步结论求和当系统级结论；(2) **必须真跑 rollout**——拿固定的 teacher 轨迹做离线计算永远测不到闭环漂移，前人测不出不是不努力，是方法论上就测不到。

**② 操作层：可回收率**。先看回放率：

$$h(\sigma) \;=\; \mathbb{E}_{\pi_\sigma}\!\left[\frac{1}{T}\sum_{t=1}^{T} \mathbb{1}\{\sigma_t = \mathrm{replay}\}\right]$$

逐块读：$\mathbb{1}\{\cdot\}$ 是指示函数（这步走 replay 记 1、走 infer 记 0）；中间的 $\frac{1}{T}\sum$ 就是这一局里 replay 步的占比；外面对轨迹求期望得平均占比。**下标 $\pi_\sigma$ 是全部重点**：期望取在**混合策略自己产生的轨迹分布**上，不是 teacher 的轨迹分布——你 replay 导致走去了别的地方，那个别的地方的回放率照样计入。闭环性不是外挂检查项，是焊在定义里的。

再看可回收率：

$$R_{\mathcal{C}}(\varepsilon) \;=\; \sup_{\sigma \in \mathcal{C}}\; h(\sigma) \qquad \text{s.t.} \qquad \mathrm{SR}(\pi_\sigma) \;\ge\; \mathrm{SR}_T - \varepsilon$$

带约束的最优化，白话：在调度器类 $\mathcal{C}$ 里，把所有"成功率掉得不超过 $\varepsilon$"的调度器挑出来（可行集），在这些人里找回放率最高的——那个最高回放率就是 $R_{\mathcal{C}}(\varepsilon)$。$\varepsilon$ 是你愿意付的成功率预算（比如 2 个点），$R$ 是该预算下能省掉的推理比例；用 $\sup$ 而非 $\max$ 因为上确界可能取不到、只能无限逼近。**这个形式最漂亮的地方**：既然 $R$ 是 $\sup$，任何实跑出来的调度器的 $h$ **自动是 $R$ 的合法下界**——"我们只报下界"从客套话变成数学事实，测到的数字永远 $\le$ 真值、不可能高估，审稿人挑不了这个刺。

**夹逼结构（按信息集嵌套定义类，不按机制——v2 修复）**。⚠ 外部评论指出过：若按机制定义（"阈值类"vs"oracle 检索类"），$\mathcal{C}_{\mathrm{thr}} \subseteq \mathcal{C}_{\mathrm{oracle}}$ 无天然包含。修复 = 按**信息集**分层：$\mathcal{C}(\mathcal{I})$ = 信息集为 $\mathcal{I}$ 的全体**因果**调度器（只见当前与过去，不见未来与结局）。取 $\mathcal{I}_\varphi$（$\varphi$-key 检索可见的全部信息）$\subseteq \mathcal{I}_{\mathrm{sim}}$（另含 sim 真实状态；sim 内状态可确定性算出观测与 key，故包含成立）$\subseteq \mathcal{I}_{\mathrm{all}}$，则

$$h(\sigma_{\mathrm{thr}}) \;\le\; R_{\mathcal{C}(\mathcal{I}_\varphi)}(\varepsilon) \;\le\; R_{\mathcal{C}(\mathcal{I}_{\mathrm{sim}})}(\varepsilon) \;\le\; R_{\mathcal{C}(\mathcal{I}_{\mathrm{all}})}(\varepsilon) \;=\; R(\varepsilon)$$

逐块读：第一个不等号成立因为 $\sigma_{\mathrm{thr}}$（我们的阈值调度器）是 $\mathcal{C}(\mathcal{I}_\varphi)$ 的一员、而 $R$ 是该类的 sup；后两个由"信息集扩张 + 机制不限 ⇒ 可行集只增 ⇒ sup 只增"保证。**"因果"约束很重要**：少了它，事后诸葛亮调度器可以先知道哪局会成功再决定哪些步 replay，$R$ 虚高到毫无意义。三层的地位：threshold 系统 = 能真部署的那个，threshold-pareto 是它的经验 Pareto 前沿；sim-state 检索臂 = 受限 oracle，部署拿不到但给更紧的下界；$\mathcal{C}(\mathcal{I}_{\mathrm{all}})$ = 真值，不可计算。**gap 就是留给后人的空间**——诚实地说"真值测不到，给你两个逐步收紧的下界，差距是未来工作"。⚠ oracle 臂 $h$ 高于 threshold 臂 $h$ 只是经验现象、不由定理保证（两个数分别下界不同层），图上如实标注。

**③ 归属层：$R$ 到底是谁的性质**。

$$R \;=\; R\big(\varepsilon;\; \mathcal{M},\; \pi_T,\; \mathcal{L},\; \varphi\big)$$

这行不是公式，是**声明范围**：$R$ 不是"任务的性质"，是任务 $\mathcal{M}$、teacher $\pi_T$、库 $\mathcal{L}$、表征函数 $\varphi$ 四者的联合性质——换个 teacher、换个库、换个 embedding，数就变。作用是防过度声称：论文里报的每个数字都必须标清这四样，不能说"这个任务有 60% 冗余"，只能说"在这个配置下测到 $\ge$ 60%"。

**单调性命题**：对"忽略新条目"封闭的类（$\mathcal{C}(\mathcal{I}_{\mathrm{all}})$ 满足），

$$\mathcal{L} \subseteq \mathcal{L}' \;\Longrightarrow\; R(\varepsilon; \mathcal{L}) \;\le\; R(\varepsilon; \mathcal{L}')$$

**库变大，可回收率不会变小。** 证明两行：取任意在 $\mathcal{L}$ 上可行的 $\sigma$，放到 $\mathcal{L}'$ 上让它**忽略所有新条目**原样运行——成功率和回放率逐字不变，仍在可行集里；可行集只增不减，$\sup$ 不降。∎ **注意前提**：证明依赖"调度器可以选择忽略新条目"，只在对忽略封闭的类下成立。**受限类不保证**：threshold 调度器没法忽略——新条目可能检索分数很高但其实是坏匹配，把阈值调度器骗去 replay，成功率反而掉，$R_{\mathrm{thr}}$ 完全可能非单调。

**为什么要把这个命题写进文中**（它不描述我们的系统——价值恰恰在"不适用"上，三个用途）：

1. **把 E3 从"显然"升格为"发现"**。没有这个命题，E3 的"库越大 SR 越高"读起来理所当然、不值一报。命题说明：单调只对最优调度是定理，对 threshold 类**理论上完全可以不单调**——于是 E3 实测的近似单调变成一个理论不保证、但实践中成立的**经验性质**（threshold 调度对库污染的鲁棒性），这才值得一节。对照定理的作用是给经验结果制造落差。
2. **预注册解释权**。若任何数据（E3 尾部、跨场景库、append 场景）出现加库掉 SR，命题提前说了：这不是系统 bug，是受限调度类的已知理论可能性——审稿人指着非单调点问，我们指回这个命题，防御工事在雷响前修好。
3. **给生命周期一个理论注脚**。$R_{\mathrm{all}}$ 单调 = "信息意义上加数据永远不亏"；threshold 下不保证 = "加数据后的收益要靠**重标定**兑现"——这正是 §8 生命周期"库更新 → 阶段 3 重标定"的理论必要性，命题顺手解释了为什么 solver/重标定不是可选项。

成本：两行证明、附录一句话、正文半句——这个价签下三个用途够本。

**内禀冗余**：

$$R^{*}(\varepsilon) \;=\; \sup_{\mathcal{L}}\; R(\varepsilon; \mathcal{L})$$

对所有可能的库再取一次 $\sup$——"经验预算趋于无穷"的极限，即**这个任务本身到底有多少推理可省**，剥离具体库的偶然性。挖矿类比：$R^*$ = **地底下总共埋了多少矿**（任务性质：流水线式任务矿多、每局初始化随机永不重演的任务矿少）；$R(\varepsilon; \mathcal{L})$ = 拿着**手里这张勘探图实际挖得出的量**——某步是不是重复劳动，须在库里找到足够近的经验才能认定，库里没存过，这步再"重复"也认不出来。"库越大冗余越多"的严格含义：任务里的重复劳动客观存在、不随库变，变的是**能认出来**的比例；E3 的增长曲线 = 可回收量随勘探图变全、往内禀天花板爬的过程，饱和处暗示已接近（本 index 下的）天花板。$R^*$ 不可测（库不可穷尽）：**引言谈、实验不测**，作用是给 research program 一个北极星，让读者知道报的数字在往哪个方向逼近。

**④ 结构层：命中不是随机撒的**。

$$s \;=\; P(\mathrm{hit}_t \mid \mathrm{hit}_{t-1}) \;-\; P(\mathrm{hit})$$

逐块读：若命中事件独立，条件概率 = 边际概率，$s = 0$；实测 $s \gg 0$（条件 0.91–0.96 vs 边际低得多，复核见 §6.2）= **命中是扎堆的**——一旦开始 replay 就连着 replay 好几步，一旦 miss 就连着 miss。这层是解释性的，回答"为什么能省这么多"：**不是零散省，是整段跳过**（冗余不是均匀撒的噪声，而是成块的"套路化子轨迹"）；run-length 分布把"块结构"从形容词变成一张图；lockstep 93–98% 说的是命中段内检索到的最优条目持续来自**同一条历史轨迹**——整段在跟随同一个 demonstration 而非每步东拼西凑。机制叙事：系统实际在做的是识别"这段我见过"然后**整段接管**，不是逐步做微观决策。

**一句话串起来**：① 定义什么叫一步冗余（挂在结局上，用干预语义）→ Remark 指出步级不可加，所以 ② 把量搬到调度器级、写成带约束的 $\sup$，让"只报下界"成为定理而非托辞 → ③ 说清这个数属于谁，顺手白捡一个单调性小定理 → ④ 用粘滞系数解释这些冗余长什么样。

**待办（遗留）**：正文写作时确认 $\mathcal{I}_\varphi$ 的精确定义（检索分数向量？含 payload？）与 $\sigma_{\mathrm{thr}} \in \mathcal{C}(\mathcal{I}_\varphi)$ 的表述；oracle 臂在图中的标注措辞（"$\mathcal{C}(\mathcal{I}_{\mathrm{sim}})$ 的下界"）。

**待办（v2 遗留）**：正文写作时确认 $\mathcal{I}_\varphi$ 的精确定义（检索分数向量?含 payload?）与 $\sigma_{\mathrm{thr}} \in \mathcal{C}(\mathcal{I}_\varphi)$ 的表述；oracle 臂在图中的标注措辞（"$\mathcal{C}(\mathcal{I}_{\mathrm{sim}})$ 的下界"）。

### 4.3 创新性核查：守得住，须主动划三条血统线（2026-08-22 web 调研）

**最近竞品（2026 上半年集中涌现，全是方法论文、无一做定义）**：

| 文章 | 出处 | 判定 |
|---|---|---|
| SkiP: When to Skip and When to Refine | arXiv 2605.15536（2026-05） | skip 判据启发式（动作频谱 DCT+分位阈值），无形式化量、无 frontier、无库；跳过=沿用自己的插值动作 |
| ElegantVLA: Learning When to Think | arXiv 2605.29438（2026-05） | 学出的 scheduler 选算力档，无形式化定义、无 cache |
| Denoising Tells When to Replan | arXiv 2606.03847 | 去噪方差自适应 chunk，机制论文 |
| Spatial Attention (execution horizons) | arXiv 2607.04739 | 观测敏感度调 horizon，机制论文 |
| Efficient VLA survey | arXiv 2510.17111 | 明写「时间冗余是互补轴、identify and reuse 是 promising direction」——**社区已点名空位**，可引为背书 |

**概念祖先（必须引用划界）**：**event-triggered control**（几十年的「只在必要时更新控制律」，概念最接近；但已知 dynamics、Lyapunov 保证、无库、无统计 SR 约束——划界句可写成致敬：$R(\varepsilon)$ = ETC 问题在黑盒 policy + 统计成功约束 + 经验回放调度类下的测量版）；RL action-repetition/frame-skip（FiGAR/TempoRL 谱系，学 repetition 不定义量）；LLM 语义 cache hit-rate 理论（对象是请求流非闭环控制）。

**判决**：① 「形式化定义 + 闭环夹逼测量 + 库归属四元组 + 规模增长线」组合无人做过——现有全是「提出跳步机制」，没有「定义量并系统测量」的论文，且我们手握 32k ep frontier 数据可立刻占位。② 关键原语差异：他们跳步 = 沿用**自己**的 chunk 外推；我们回放 = 取**库中经验**——只有我们的版本带 fleet 共享/append 语义。③ 真实风险 = **窗口收窄**：2026-05~07 三个月四篇机制文，截稿前会更挤——定义论文先到先得，快写是最好的防御。

### 4.4 体量裁决与"公式-方法接口"（owner 质询后定）

审计标准：每条公式在后文必须有**消费者**，否则是装饰。

**正文只留三件套（≈0.5 页）**：
1. **$R_{\mathcal{C}}(\varepsilon)$ 定义**——三个真实接口：① **它就是系统标定程序的形式化**（threshold 校准 = SR 约束下最大化回放率 = 经验地解这个优化；公式是方法的目标函数，不是描述）；② §5 主实验语义（threshold-pareto 从"一堆操作点"升格为"对 $R_{\mathrm{thr}}$ 的测量"，A 叙事支点）；③ 「实跑 $h$ 自动是下界」= 报告的诚实结构。
2. **组合性破缺 remark**——消费者 = "必须闭环 rollout / 前人离线 metric 测不出"的方法论辩护 + related-work 划界。
3. **归属四元组 $(\mathcal{M}, \pi_T, \mathcal{L}, \varphi)$**——升格为**全文章节组织原则**：四坐标各对应一章实验（$\mathcal{L}$→库规模 E3、$\mathcal{M}$→跨场景、$\pi_T$→双 teacher、$\varphi$→key 消融）。

**压去附录**：步级 do-定义形式化（正文一句+脚注）、信息集嵌套夹逼的构造与包含论证（正文一行不等式）、单调性命题+证明、内禀 R*（正文各一句）。粘滞系数不算数学，是实验章指标定义。

**红线**：正文必须显式写出三个接口（尤其"校准=解这个优化"）——这句话在，数学是引擎；不在，数学是妆。§4.2 全量保留在纪要作设计记录，不等于进正文。

### 4.5 §3 在后文的呼应地图（逐章审计，2026-08-22 晚）

三件套在每一章的消费位（接口①②③ = §4.4 已录；★ = 本轮新发现）：

- **§4 系统**：threshold 标定 = 解 $R$ 的 $\sup$-s.t.（接口①）；生命周期阶段 2 = 采样可行域、阶段 3 solver = 在线求解同一优化——四阶段可整段用 R 语言复述；warm start 第三档 = $R$ 从二元 $\{\mathrm{replay}, \mathrm{infer}\}$ 到连续谱的推广一句；成本账本的 $h$ 即定义中的回放率（$1/(1{-}h)$ 的 $h$ 有正式出身）。
- **§5 测量**：5.1 曲线语义由 §3 授予（接口②）+ oracle 臂合法性来自信息集夹逼 + 「实跑 $h$ 是下界」报告结构（接口③）；5.2 = L 坐标 + 单调性命题经验对照；Table 1 = $R_{\mathrm{thr}}(\varepsilon)$ 上的具名点，$\varepsilon$ 列即 SR 预算。★ **5.3 lockstep = 组合性破缺的"温和化机制"**：remark 说回放使轨迹漂移、步级冗余不可组合——那为何连续回放几十步不崩？lockstep 给出机制答案：**回放段不是乱漂，是被库中一条完整成功轨迹"轨道化"**，漂移方向恰沿一条亲测走得通的路。§3 的理论隐患与 §5.3 的结构发现互为锁扣（remark 解释为何必须闭环测，lockstep 解释为何闭环测出来还不错）——分析章最好的一段话。
- **§6 跨场景**：整章 = 四元组 **$\mathcal{M}$ 坐标的干预实验**（换 $\mathcal{M}$、冻结 $\pi_T/\mathcal{L}/\varphi$）；双 teacher = $\pi_T$ 坐标——四元组作章节组织原则在此兑现。
- **§7 defense**：★ **trained router = 同一 sup 问题的另一个求解器**——router 也是调度类 $\mathcal{C}$ 中一个 $\sigma$，其 $h$ 同样只是 $R$ 的下界 ⇒ threshold vs router 之争获得干净表述：**争的不是谁对，是"便宜求解器是否已接近该类可达域"**；若 E10 学习型调度器没把下界推高多少，即支持"threshold 已近 $R_{\mathcal{C}(\mathcal{I}_\varphi)}$"（§7.2 增补为第 6 条）。7.3 拉 replan = C 中最笨成员（不看状态的周期调度器），打赢它 = 信号价值的最小证明。7.1 归属声明防"cache 输 = 方法差"。
- **§1/§8**：引言 P2 = R 通俗版、内禀 vs 可回收 = 北极星句；§8 scope 用四元组语言写（sim-only = M 范围、quasi-static = M 内任务子类）；拒稿预埋 ①（"RT-Cache+阈值增量"）的主盾牌 = §3（贡献不是机制是量的定义与测量）。

## 5. 工作提纲 v0.2（2026-08-22 晚定 v0.1；2026-08-24 应教授「方法与发现至少 2–3 页」质疑扩容 §3/§4/§5.3）

**题目（owner 星标）**：Muscle Memory for Robot Policies: Caching a VLA's Own Experience in Its Own Representations（候选清单见 §3）

- **§1 Introduction（~1.25 pp；Fig 1）**：P1 经济矛盾 + serving 热题已起（ROSA/Armory 调度需求，没人消需求）→ P2 中心问题「VLA 闭环推理有多少步是重复劳动？」+ 帧间冗余类比 → P3 加粗点：teacher **自身表征**索引**自身成功经验**（三重红利）→ P4 系统一段（免训练、fail-closed）→ P5 三贡献：① 定义并首次闭环测量 $R(\varepsilon)$，高且有结构；② 系统原语：内部表征经验索引 + 免训练 threshold，与蒸馏/量化/减步/token-cache 正交可叠加；③ frontier + 经济账（跨场景迁移视 B 结果升降）。**Fig 1 = 系统图 + 四阶段生命周期时间轴 + episode 命中条带 teaser**（三合一）。
- **§2 Related Work（~0.75 pp）**：六族——检索回放（RT-Cache：头号 delta = key 来源）；VLA 加速原语（VLA-Cache token 级另一物种）；2026 自适应计算浪潮（SkiP/ElegantVLA 等：机制无定义无库，划界见 §4.3）；policy serving（ROSA/Armory：调度 vs 消需求，正交叠乘）；**跨域 cache 谱系（新增）**——LLM serving 的 cache 已是成熟层（KV/prefix cache = exact reuse、GPTCache/IC-Cache 语义 cache = approximate reuse），我们 = closed-loop approximate reuse；合法性借用（"成熟 serving 原语带进新域"）+ IC-Cache payload 教训（naive 语义回放 win rate 50%→18%）作独立旁证 + 划界（LLM cache 开环请求-响应、有每请求反馈，闭环控制无此奢侈 = 组合性破缺的另一面）；概念祖先 ETC + UQ/失败检测（C 叙事降级于此）。
- **§3 可回收冗余 $R(\varepsilon)$（~0.75–1 pp，v0.2 扩容）**：三件套按 v3 风格写全（每条公式带逐块读 + 白话含义——0.5 页塞不下 v3 写法，教授质疑替我们发现了这个矛盾）；三个公式-方法接口显式成段（§4.4 红线）；其余数学进 App A。
- **§4 系统（~1.5 pp，v0.2 扩容，补三块现成素材）**：
  - **Key 抽取机制细节**（原来只有一句"切点"）：pi0.5 的 prefix_embs 三段布局、pool 降维、切点选择依据 + GR00T 的对应切点（input_embeds）——两模型切点的结构对应本身是"机制 model-agnostic"的证据，加粗点的技术实体，讲细正是差异化；
  - **Warm start 升为一小节**（owner 拍板）：三档执行谱系（全回放 / 经验引导减步 / 全推理）+ 机制（检索动作作 flow 积分起点、跳过前段 denoise）——"冗余是连续谱"的机制兑现，系统不再像"查表"。**分工：机制进正文 §4，E8 三方对照验证仍在 App**；
  - **检索与融合协议**：多模态 key（视觉×2 + robot_state）、逐字段归一化 + 加权融合、LOEO 标定——系统真实的核心工程，有 fusion 机理研究背书；
  - 保留：success-filter 建库、4ms 检索、threshold 标定（= 解 $R$ 的约束优化，预注册面）、**Deployment lifecycle 小节**（四阶段，见 §8）、成本账本（hit 7.2e-05、$1/(1{-}h)$）。
- **§5 测量 $R(\varepsilon)$（~1.5 pp；Fig 2、Fig 3、Table 1）**：5.1 主曲线 **2 benchmark × 2 teacher 四小格**（LIBERO 两 suite + RoboCasa365 pi0.5/GR00T），每格叠 oracle 臂虚线 + teacher 锚点——一般性内建于主图；5.2 库规模线（E3）；5.3 冗余结构**扩为半页小节**（v0.2：粘滞/块长/lockstep 是贡献①的一半，图已画好配得上）+ 分数判别力 AUROC 一句（描述性，caveat 见 §6 表）；**5.4 E7a 来源轴消融升正文**（2026-08-26 owner 拍板：internal vs external encoder 的 AUROC 对比，~0.25 页或并进 Fig 3 一 panel——题目级主张「in its own representations」的直接证据必须在正文，E7b 深度/E7c 融合留 App E；判据 = **证 claim 的消融进正文，证选择的消融进附录**）；Table 1 操作点 × 经济账（列定义见 §9）。AUC 类信号实验以此两处离线小分析为限，**不恢复旧 X3 大章**（那是 C 叙事的主轴，会挤占页面把置信信号抬回主张位）。
- **§6 Does Experience Transfer?（~1 pp；Fig 4）**：唯一主题 = 跨场景（场景 A 库 → held-out B），不重复 §5 故事；**分析小节：迁移率按任务类型分层**（大尺度家具操作 vs 精细抓放，机制假说见 §7.1-4）。fallback：RoboCasa 数据不及时 → 主图退 LIBERO，本章承接 RoboCasa 全部。
- **§6b Real-robot validation（~0.3–0.5 pp，E13，2026-08-26 入编）**：SO-101 上生命周期端到端实走 + 2–3 操作点配对评测；一张小表（任务 × 臂 × {SR, h, 延迟中位}）+ 延迟对比图或命中条带；定位 = 存在性证明非主结果。**入编后 §8 limitations 删除 "sim-only"，改写为 "real-robot validation is small-scale (an existence proof)"**——审稿人的 sim-only 质疑失去靶子。
- **§7 Why Not Alternatives（~0.75 pp；Table 2）**：7.1 小模型——regime 地图叙事 + 获得成本列，SR 如实报输（四道防线见 §7.1）；7.2 trained router——论证段（三层论证 + E10 实证载体）；7.3 teacher-cheapening + warm start——正文一句话（正交可叠加 + 经验引导减步 vs 盲减步），三方对照进 App。
- **§8 Limitations（~0.3 pp）**：真机验证为小规模存在性证明（E13 落地后取代 sim-only）、quasi-static 边界（dynamic 任务对 staleness 敏感）、库内 regime、单 embodiment 类别、robots-per-GPU 为理想化推算（mini-bench 锚定）。
- **§9 Conclusion（~0.1 pp）**。
- **Floats（6+1）**：Fig1 系统+生命周期+条带 teaser · Fig2 主曲线四小格（含 oracle/锚点）· Fig3 规模线+结构（粘滞散点+run-length）· Fig4 跨场景 · Table1 操作点经济账 · Table2 小模型对比（含获得成本列）· §2 对比小表。
- **附录**：A 数学全量（夹逼构造/单调性证明/步级 do 形式化）· B 成本账本 + serving mini-bench + teacher-cheapening/warm-start 三方对照 · C 历史负结果一表 · D 校准协议/预注册/统计 · E per-task 分解 + key 消融 · F 实现/复现。

**页数纪律（owner 令 2026-08-24）**：v0.2 扩容后 §3+§4 ≈ 2.5 pp（响应教授"方法与发现至少 2–3 页"的要求），总页数暂时超预算——**先写全，写作期再删**，不预先自我阉割；§2/§7/§6 是届时的压缩候选。

**新增实验缺口 4 项**：oracle 臂 E2（§5.1）、warm-start 三方对照 E8（App B，兼作 teacher-cheapening）、serving mini-bench E11（App B）、**SO-101 真机验证 E13（§6b）**；在跑收官 3 条（E1-LIBERO 侧、E6、E10）；其余现成数据换叙事。

## 6. 章节-实验对照表（与 §5 提纲对齐）

**编号规约（owner 令 2026-08-22）**：纪要实验一律用 **E 系代号**（E0–E12，按论文出现顺序）；X 系为旧 TIER 台账代号已废止，仅在溯源历史资产/plan 文件时以「旧 X…」形式出现。

主结果在 **2 benchmark × 2 teacher** 网格上出 $R(\varepsilon)$ 曲线，一般性内建于主图；§6 只留跨场景。fallback：RoboCasa 出数不及主图时间线 → 主图退 LIBERO 双 suite、RoboCasa 整体回落 §6。

| 代号 | 章节 | 实验 | 是什么 | 有什么用 | 数据在哪 / 现状 |
|---|---|---|---|---|---|
| — | §1 Fig 1 | $R(\varepsilon)$ teaser + 命中条带 | §5 素材缩略 | 第一页钩子："X% 步免推理、SR 损失 $\le \varepsilon$" | 复用 §5，零额外 |
| **E0** | §4 系统 | 检索/回放微基准 | 六段延迟分解、检索 ~4ms、hit 相对成本 7.2e-05 | 成本账本可信度 + "key 是字节副产品"实证 | **现成**：`exp/cache_latency_bench`（35.49→4.15ms）+ 旧 X14 microbench 冻结成本 |
| **E1** | §5.1 主曲线 | $R_{\mathrm{thr}}(\varepsilon)$，2 benchmark × 2 teacher | LIBERO 两 suite（16 档 × 4 库 × 500 ep）+ RoboCasa365（pi0.5 + GR00T）四小格 | **主结果**：R_thr 首次系统测量 + 一般性内建（四元组 $\mathcal{M}$、$\pi_T$ 坐标在主图兑现） | LIBERO 侧**在跑近完**（threshold-pareto 32,000 ep）；RoboCasa 侧**在跑**（ws_search round-1 → 评测） |
| **E2** | §5.1 oracle 臂 | sim-state 距离检索 | 同协议、检索换真实状态距离，2–3 操作点，画进 Fig 2 各格虚线 | 夹逼上层：$\mathcal{C}(\mathcal{I}_{\mathrm{sim}})$ 下界，覆盖-vs-key 归因（详解见 §6.1） | **需新跑**（E3 plan（旧 X9b）已列 future work；全新实验之一） |
| **E3** | §5.2 规模线 | R 随经验预算 | 库规模 {1,2,5,10,20,45} 轨迹/任务 × 闭环 SR，16 臂 8,000 ep | 四元组 L 坐标；"多少数据买多少冗余"；对照单调性命题（经验单调非定理） | **现成 Verified**：E3 = 旧 X9b `cache_size_ablation_plan`（待 commit）；`always_hit` 无阈值口径 = R 曲线超廉价端点族，与 §5.1 口径有别须注明 |
| **E4** | §5.3 结构 | 粘滞/run-length/lockstep | 命中过程统计（182,899 决策步 always-search 真 verdict） | 冗余"块状+重访"结构发现，解释为什么能省 | **现成且已复核**（§6.2）：`exp/gate_research` + Stage 2 离线分析 |
| — | §5 Table 1 | 操作点表 | 预注册 OP × {SR, h, GPU-s/step, robots-per-GPU, 驻留内存} | 经济账落地成数（详见 §9） | **现成+推算**：threshold-pareto + 微基准换算 + mini-bench 锚点 |
| **E6** | §6 跨场景 | 场景 A 建库 → held-out 场景 B 查询 | RoboCasa365 双厨房、双 teacher；分析小节 = 迁移率按任务类型分层 | **B 惊喜章（Does Experience Transfer?）**；初步观察见 §7.1-4 | **在跑**：robocasa365 线 eval-prep（T6 pkl 建成，待 T7） |
| **E7** | **E7a → 正文 §5.4**；E7b/c → App E | key 空间消融族三轴（2026-08-26 拆分裁决） | ① **E7a 来源**（**正文**）：内部表征 vs CLIP/外部 encoder——题目级主张的直接证据，正文无它则 title claim 悬空；② **E7b 深度**（App）：vision tower 后 vs backbone 中间层（旧 llm_layer_extract 线有现成实验与 runbook）；③ **E7c 融合**（App）：逐模态分开归一化+加权 vs 大一统 key（weighted_sum 两层动机，fusion 机理背书）。**度量 = AUROC**。判据：证 claim 的消融进正文、证选择的进附录 | 四元组 $\varphi$ 坐标；§2 加粗点实证 | **现成可复算**：老 weighted_sum/$\varepsilon{\to}\delta$ + llm_layer_extract + temporal_prune 历史数据离线重放 |
| **E5** | §5.3/App | 分数判别力 AUROC | 被接受命中的 cp1_score 对 episode 成败的 AUROC（threshold-pareto/gate 逐步日志离线算） | 回答"threshold 卡的分数有无判别力"= threshold 能工作的机制解释 + C 叙事段的实数字。⚠ episode 级标签 × 步级分数 = **关联非反事实**（老 phase5 同款 caveat），只作描述性证据 | **零新 rollout**，离线可算；未算 |
| **E8** | §7.3/App B | teacher-cheapening + warm-start 三方对照 | 全 teacher / 盲减步@k / warm-start@k（同跳过积分量比 SR）；两旋钮均为部署期超参，调之合法 | 定位 = **正交可叠加**而非竞争；warm start（cache noisy action 跳 denoise，精度实测不错）= "经验引导的减步" vs 盲减步——若成立升格为又一"经验的价值"论证 | **需补跑但量缩**（~1k ep；旧 X4 卡 + `warm_start_sweep` 数据复用）；正文一句话，图进 App B。全新实验之二 |
| **E9** | §7.1/Table 2 | 小模型对比 | ablation_study 8 臂（hit/miss 槽换 SmolVLA/ACT）+ 获得成本列 | regime 地图叙事的数据面（§7.1） | **现成**：ablation_study（G2 过待 commit） |
| **E10** | §7.2 | trained router | E10（旧 X15 risk_router）代理监督 router vs threshold | §7.2 弹药实证载体；老 X14 不引用（owner 令） | **在跑**：E10 G1 过、待 Code |
| **E11** | App B | serving mini-bench | $N \in \{4,8,16\}$ 并发 episode 压单 GPU，cache on/off 的 steps/s/GPU 比值 | $1/(1{-}h)$ 推算的实测锚点（§9） | **需补跑**（`--replicas`+router+serving_benchmark 工具链现成 = 旧 X13 复活；几百 ep、1–2 天）。全新实验之三 |
| **E13** | §6 后独立小节（0.3–0.5 页） | **SO-101 真机验证**（2026-08-26 入编） | 部署生命周期四阶段在实体臂上端到端实走（采集→shadow→标定→服务）；3 任务按动作尺度分层（抽屉/翻盖、抓放、插入）× 3 臂 × ~20 ep 配对 + 建库 ~150 ep ≈ 330 ep；测 SR、回放率、**端到端延迟**（命中=本地回放 vs miss=网络往返+推理——per-robot 延迟轴的实测) | **存在性证明**：机制在真实感知与物理下成立；**把 sim-only 从 limitations 里整个删掉**；位置模板垫做 init 纪律 | **待跑**（SO-101 硬件在手；前置 = teacher 微调，GR00T so100 官方教程或 pi0.5 lerobot 路径；时间不够则押 rebuttal）。全新实验之四 |
| **E12** | App C | 历史负结果表 | 171 config 逐步权重 + depth{3..6} 扫描压缩一表 | 一句话交代"为何 depth-1 key" | **现成**：34,200+7,200 ep 老数据 |
| — | App D/E | per-task 分解、校准协议、预注册记录 | 逐任务 SR、阈值标定、init 池账本 | 复现性 + 统计纪律 | **现成**（写作期整理） |

### 6.1 oracle 臂详解

**一句话：两套完全一样的系统，只换"怎么判断像不像"这一个零件，其它全同。**

- 我们的臂：判相似靠**模型内部向量余弦**——φ(o) 是对世界的有损压缩，会看走眼（物理不同但向量像 → 错回放；物理近但向量不像 → 白 miss）。
- oracle 臂：同库、同回放、同阈值扫描，唯一换掉相似度一步——掀开仿真器后台，用**物理真值**（物体位置/关节角，MuJoCo 状态）算两局面的真实距离，把"判断像不像"做到理论满分。部署拿不到真值，故它不是系统而是**测量尺**。类比：同一本通讯录找人，我们凭长相描述、oracle 凭身份证号，找到后干的事相同。
- **它在比什么**：同一 SR 约束下，满分判断 vs 向量判断各能安全回放多少步——其它零件全同，差值精确等于"key 看走眼"的损失。
- **为什么非做不可**：闭环回放失败有两个混杂原因——① 库里**根本没有**足够近的经验（覆盖不足，谁判断都没用）；② **有但没找对**（key 有损）。我们的曲线里二者不可分；oracle 把 ② 做到极限，它仍回收不了的就全是 ①。
- **三种读数**：oracle ≫ 我们 → key 的锅，改 index 有大利可图（gap 量化"留给后人"）；oracle ≈ 我们 → **key 没丢东西**，"teacher 自身表征够用"最硬实证（§2 加粗点）；oracle 自己也不高 → 覆盖的锅，谁的 key 都救不了，防全文被读成"检索烂才省这么点"。
- 基建：建库时每条目存 sim 状态（采集 wrapper 存档，旧 X5a 设计可复用）；出处 = E3 plan（旧 X9b）的 identifiability 缺口（`always_hit` SR = f(库内容, 检索质量) 混杂），当时列 future work，现升主线。

### 6.2 §5.3 已实测验证 + 呈现草案（2026-08-22）

三个统计量在 `exp/gate_research` 逐步日志（182,899 决策步，与文档记录逐位吻合）上离线复算，全部成立：粘滞 $P(\mathrm{hit}\mid\mathrm{hit})$ 0.909–0.961 vs 边际 0.278–0.704（粘滞 +0.24~+0.63，7 config 一致）；run-length 中位 9–12 步、P90 18–52 步；lockstep 同轨迹率 0.928–0.983、$\Delta{+}1$ 率 0.694–0.939（**新观察：阈值越宽 $\Delta{+}1$ 率越低**，lockstep 纯度随操作点变化，作次级观察）。

**呈现裁决：图不是表**。三 panel 草图（真数据）：(a) 8 条 episode 时间轴条带（块状一眼见）；(b) 粘滞散点 $P(\mathrm{hit}\mid\mathrm{hit})$ vs $P(\mathrm{hit})$——**每 config 折叠成一点，7 点全离对角线 = "config 太多"问题的解**；(c) run-length 生存曲线 vs 几何基线（差 2–3 个数量级）。正文三句话：命中不独立 / 冗余成块 / 块内跟随同一条库轨迹整段接管——第三句衔接 ★§4.5 的锁扣论证（lockstep = 组合性破缺的温和化机制）。逐 config 表进附录。(a) 代表 config 与 §5.1 主操作点保持一致防挑样质疑。

**页面预算三级制**：A 级 4 个 float 席位（Fig2 一图装四实验；Fig3 规模+结构合并；(a) 条带挪 Fig1 当 teaser；Fig4 跨场景；Table1）；B 级正文一段、数据进别人的 float（oracle 臂/cheapening/小模型/router）；C 级一句话+附录（key 消融 E7b/c/历史负结果/微基准/次级观察；**E7a 来源轴 2026-08-26 升 B 级正文小节**——题目主张的直接证据）。分级口径：**改变读者对 thesis 信念多少**——防御工事不配独立阵地。

产物索引：草图页 [`redundancy_structure_fig.html`](redundancy_structure_fig.html)（自包含，图内嵌）+ [`redundancy_structure_fig.png`](redundancy_structure_fig.png)；线上版 https://claude.ai/code/artifact/30ea8118-5e5d-4367-a0d5-0475319cb53b ；分析/绘图为一次性草稿脚本（~70+120 行，纯离线秒级），正式落地时按流程入 `exp/gate_research/`。

## 7. Defense 弹药库

### 7.1 vs 小模型：八条弹药 + 必输格子的四道防线

**前提如实**：SR@同 teacher 率会输不是风险，是**已经输过**——老数据 hit 槽放蒸馏 student SR 0.888/0.830 > 放回放 0.704（TIER 叙事当年即由此而来）。本节设计以此为前提。

弹药（2026-08-22 首轮，按硬度排）：

1. **训练/维护成本**：cache 零训练——建库 = 纯前向、label-free，teacher 换版本重建库即可；小模型每换 teacher/任务族/embodiment/操作点都要重蒸馏（ACT 还是 per-task ×10 个模型）。乘法维护 vs 自动跟随。
2. **数据成本**（有现成数据）：E3 曲线——每任务几条成功轨迹 cache 就工作；蒸馏 student 用了差集池全量 450 init/suite 的 rollout。数据稀缺侧（恰是真机场景）cache 是唯一选项。
3. **失效模式**：replay 只回放 teacher 亲测成功的动作、仅在近重复状态触发，零外推；不相似 → fail-closed 回落 teacher。小模型 OOD 处是不受控的参数外推，且无自带置信信号——cache 的相似度分数本身就是置信度。
4. **延迟/部署**（有现成数据）：旧 X14 冻结成本 teacher 1.0 / student 0.0557 / **cache 7.2e-05**；检索 ~4ms，可下沉无 GPU 边缘端。
5. **regime 限定反击**：「student 更好」只在蒸馏数据充足、任务族封闭的 LIBERO regime 成立；RoboCasa365 365 任务、每任务数据薄，per-task 蒸馏不可行，cache 机制照常工作（已迁 GR00T N1.5 证明模型无关）。
6. **增量性与可审计**：新经验 append 入库即生效，零梯度零遗忘；每个回放动作有 provenance（哪条轨迹哪步）。小模型两者皆无。
7. **正交而非竞争**：ablation 方向 2（hit→replay, miss→小模型）说明即便有小模型，cache 仍在其上加值——能力放大器，不必二选一。
8. **跨场景鲁棒性**：小模型场景绑定、跨场景须重训；cache 可能具备跨场景鲁棒性（在测）；退守论证：即使失效，新场景重采重建也远比重训便宜。

四道防线（对"必输格子"，按强度排）：

1. **获得成本轴（最强，不是 latency）**：per-step 差 770 倍但两者都远小于 teacher，是二阶差异撑不住防线；数量级差异在**建设侧**（student = 450 init rollout + 蒸馏 + 每变更重付；cache = 每任务 5 条轨迹 + 纯前向 + append 即更新）。Table 2 加**获得成本列**（rollout 数/训练 GPU 时/重训触发条件/增量更新能力），SR 列**如实报输**——输的格子旁必须站着代价列。
2. **Regime 边界**：LIBERO 是蒸馏主场；RoboCasa365 per-task 蒸馏铺不开。若 RoboCasa 侧 cache 工作而蒸馏不可行，LIBERO 的输被框成"蒸馏的舒适区"。
3. **叙事姿态预写（真输了怎么办）**：不打"回放 ≥ 蒸馏"的必输仗，打 **regime 地图**——"同一批经验的两种压缩：参数化插值在数据充足封闭任务族更优；非参数回放在数据稀缺/任务开放/需增量更新场景是唯一选项；我们测量边界"。与全文测量姿态（$R(\varepsilon)$ 同款）自洽——**测量者输一个格子不丢脸，推销员输一个格子全盘崩**。
4. **跨场景维护成本不对称（owner 报初步数据）**：老 cache（场景 A 库）在不同场景表现出一定鲁棒性，且**大尺度任务（fixing furniture 类）上更强**——打在小模型结构性最弱点（换场景=全模型重训 vs cache 介于部分可用与廉价重建之间）。机制假说：大幅度家具操作的动作轨迹由**任务几何**决定、对场景外观细节不敏感 ⇒ 视觉键变了动作模式仍保持；精细抓放对位姿敏感故难迁移。⇒ 跨场景鲁棒性**按任务类型分层**——给 §6 跨场景章送分析小节（per-task 迁移率 × 动作幅度/精细度），可能是 B 惊喜章真正的发现点。⚠ 纪律：①数据状态如实标（初步观察，正式判决待 T7 评测）；②措辞对称（不写"跨场景无损"，写"维护成本不对称"）。

Claim 纪律：headline 不出现任何"cache 优于小模型"字样（弹药限 defense 语境，不进贡献句）；对比表定位正文 §7。

### 7.2 vs trained router（"天花板更高"质疑）

⚠ 纪律（owner 定）：不引用旧 X14 对决负结果当弹药——旧叙事下的设计，"训得烂/拉偏架"一击即溃；SOTA 公平对决还没跑。主轴 = 成本 + 灵活性。

1. **"天花板更高"只在能学到真目标量时成立，而真目标量无监督标签**：路由要的是 Q(s, cache) = "从 s 走 cache、未来继续被路由，最终成不成功"。对它：(a) 标签反事实；(b) 纯执行体 rollout 打标 = visitation/continuation 双错位；(c) Bellman 耦合——学它只有在线 RL。监督能做的是换目标量学单步代理，但学到的不再是 Q——"天花板更高"的理论保证在换目标那一刻就没了。
2. **trained router 本质是当前这个 cache 的函数**：它拟合的是"这个库、这个 key 空间、这个分数分布"下的决策边界——库一变（append/换场景/teacher 更新/规模增长），输入分布就漂，router 面临重训；而 cache 的核心卖点恰恰是随时可变。threshold 只依赖分数的序结构，库变了重标定是轻量离线程序。**训练 router = 把系统里最灵活的部件锁死在最僵硬的部件上。**
3. **threshold 是检索的免费副产品**：cache 档执行本来就要跑检索取 payload，threshold 只是在必产出的分数上加一次比较，边际成本零；trained router 是在必跑检索之外再加一个模型、一条训练管线、一套标签采集。
4. **router 自己就是个小模型**——把"为什么不用小模型"原样继承：OOD 外推不受控 vs threshold 天然 fail-closed（不相似→teacher）。
5. **实证对决留白但姿态摆好**：正文可承诺"与按 SOTA 配方公平训练的 router 对比"（E10 为候选载体），预注册双向措辞——赢了是零训练胜、输了 gap 就是训练的价格标签（且按 #2 每次库变重付）。
6. **R 框架表述（★ §4.5）**：router 也是调度类 $\mathcal{C}$ 中一个 $\sigma$，其 $h$ 同样只是 $R$ 的下界——之争的干净形式 = "便宜求解器是否已接近该类可达域"；E10 若没把下界推高多少，即实证 threshold 已近 $R_{\mathcal{C}(\mathcal{I}_\varphi)}$。

一句话版：threshold 又便宜又有效又随库自由；trained router 天花板未证、成本先付、还把灵活性抵押了。

### 7.3 vs teacher-cheapening：正交叙事 + warm-start 三方对照

审稿人必问"何必搞库？把 teacher 调便宜不就行"。两个旋钮（flow 积分步数、replan 间隔）均为**部署期超参非训练参数**，调之合法，问题成立。处理：

- **定位 = 正交可叠加而非竞争**：cache 的 miss 步照样可用减步 teacher 执行——两者是可组合原语，喂给"第四原语正交可叠加"贡献 claim，竞争框架消解后无须"战胜"。
- **warm start 升级（owner 拍板）**：系统已有 WARM_START 机制（cache 检索的 noisy action 作 flow 积分起点、跳过前段 denoise，精度实测不错）与减步天然同轴——正确实验形态 = **三方对照：全 teacher / 盲减步@k / warm-start@k**（同跳过积分量比 SR）。warm start 赢 = "经验在 denoise 轴也有价值"，防御性 baseline 反手变证据；teacher-cheapening 的减步档就地成为对照组，一个实验两用。
- **落位**：§4 一句话（系统第三档，"冗余是连续谱"）+ 正文 §7.3 一句话 + 三方对照图进 App B；**主 $R(\varepsilon)$ 曲线保持二档口径不动**（threshold-pareto 32k ep 不重跑）。规模 ~1k ep（旧 X4 卡 + `warm_start_sweep` 数据复用）。
- 一致性注记：旧 X14 曾裁"全系统不用 warm start、hit 一律 FULL_HIT"——那是 RL router 实验的口径裁决非机制废除；warm start 以"App 对照 + 系统第三档描述"回归，与主线二档数据不冲突。

### 7.4 审稿人视角补充攻防（2026-08-25 扫描，六条，按危险度排）

1. **"你的冗余是 benchmark 造出来的"（评测协议攻击，最危险）**：LIBERO 每任务固定 50 init ⇒ "init 多样性低冗余当然高，真实世界初始条件连续无限，50% 是 benchmark 伪影"。防线三件：① 四元组归属声明——$R$ 本来就是配置的属性，从不主张普适；② **反转句：家用/工厂部署恰恰是低多样性环境，benchmark 的固定 init 是部署重复性的模拟而非缺陷**；③ 结构发现（粘滞/lockstep）与机制不依赖 init 池大小。须写成正面段落。
2. **"你就是记住了测试集"（泄漏攻击，必查）**：近邻回放系统天然招此疑。弹药已有须归档成正式防线：建库 init 与 A-pool 泄漏实测 **0/50**；init 池纪律；且回放的是**不同 init** 的轨迹段——跨 init 泛化本来就是机制在做的事。
3. **"库无界增长"（系统攻击）**：append-only、无 eviction（`BackendPool` 实况）、O(N) 检索——"部署一年后库爆掉"。防线：Table 1 驻留内存列 + 当前规模检索 4ms + lockstep 结构暗示轨迹级压缩可行（future work）+ 诚实承认 eviction 未做。
4. **"回放错误动作的安全后果"**：sim 掉 1pp = 真机打碎盘子；ε 在安全关键场景怎么定。防线：fail-closed 方向性、ε 可调保守、SR 约束框架本身就是把安全写成预算的语言；limitations 一句。
5. **"自我僵化反馈环"（小众但 AC 级）**：长期回放使系统不再产生新经验、行为固化。防线现成：gate 线 probe/periodic 注入机制即解药 + 生命周期阶段 4 持续 append 新成功。一段话。
6. **AC 元攻击："测量论文还是系统论文"（identity 分散）**：防线 = 写作纪律——§4.4 测量姿态在 intro 钉死，贡献三条全部挂在"测量"动词下。

### 7.5 Appendix「提前防出去」的形式裁决（owner 问 2026-08-25）

**可以做，但不用 "Q\&A" 字面格式**——用 **"Design choices and alternatives considered" / "Extended discussion"** 小节形式：每小节一个问题的陈述式回答（why not a learned router / why not distillation / does the library memorize the eval set / what about unbounded growth / safety of replayed actions…）。实质是 Q&A，形式是 discussion——学术味保住、防御功能不减；正文各处一句话指针（"see App. X"）。

理由与纪律：
- **优点**：攻防前置 = 预注册姿态的延伸；审稿人看到自己的问题已被认真对待，攻击欲下降；继承老 outline 的 rebuttal bank 传统。
- **风险与对策**：① 通篇设防显 defensive ⇒ 用 discussion 语气不用 FAQ 问句体；② **每写出一个攻击 = 替审稿人起草 weakness 清单** ⇒ **只防有实锤防线的问题；没有好答案的问题绝不写进去（写了 = 自曝递刀），留 rebuttal 相机行事**；③ 占 appendix 预算 ⇒ 每问 ≤ 半栏。
- 内部完整版 rebuttal bank 继续在纪要维护（§7 全部），appendix 版是其**有实锤防线的公开子集**。

## 8. 部署生命周期（owner 提出；论文落位 §4 系统章）

四阶段工作流：

1. **数据库采集期**：teacher 独跑，成功轨迹连 key 一起入库（采集即建库，零训练）。
2. **影子数据库运行期**：库上线但**只检索不派发**（shadow / warmup），采集真实分布下的 similarity 数据——为阶段 3 提供标定原料。
3. **参数搜索期**：**online solver** 在阶段 2/持续的 similarity 数据上优化检索权重与 threshold（⚠ **未实现**——当前实验用离线等价物：LOEO 归一化标定 + threshold 网格；solver 是它的在线化）。
4. **稳定运行期**：命中回放、miss 走 teacher；新成功经验持续 append（回到 1 的增量版）。

**落位理由**：① 天然回答两个必问——"threshold/权重哪来的"（阶段 2/3 = 标定的部署形态）与"冷启动怎么办"（阶段 1/2 = 冷启动协议）；② 与 §3 应用故事呼应——"家用越用越快"的机制化版本就是 1→4 推进 + 阶段 4 持续 append。Fig 1 画成四阶段时间轴。

**措辞纪律**：online solver 调的是**几个标量超参**（模态权重 + 阈值），不是学一个函数——必须与 §7.2 trained-router defense 划清（solver ≠ router；免训练主张指"信号与决策规则免训练"，标定自动化不破坏它）。solver 未做：正文描述生命周期、注明实验以离线标定实现阶段 3，online 版列 future work 或排期后补。

## 9. Table 1（操作点表）与 serving mini-bench

**表是什么**：$R(\varepsilon)$ 主图给曲线，Table 1 给曲线上几个**预注册操作点**的完整档案。列定义：

| 列 | 含义 | 口径 |
|---|---|---|
| SR | 该操作点 500 ep 成功率 | 质量轴，"省了但没变笨" |
| 回放率 h | 决策步中走 cache 回放的比例 | 省的量 |
| GPU-s/step | $h\cdot c_{\mathrm{hit}} + (1{-}h)\cdot c_{\mathrm{teacher}}$（$c_{\mathrm{hit}}\approx$4ms 检索） | **实测**，Fig 2 的 x 轴 |
| robots-per-GPU = **$1/\mathrm{IR}$** | 单卡可服务设备数放大倍数；$\mathrm{IR} = 0.152 + 0.848\cdot\mathrm{tr}$（Stage1 key 构建地板） | ⚠ **推算**（无排队损失假设）；上限 $1/0.152 \approx 6.6\times$。$1/(1{-}h)$ 仅作忽略 key 成本的理想式提及（§10.1 口径修正） |
| 库驻留内存 | 库加载后 RAM 占用 | 部署脚注，防"库爆内存"质疑 |

**四个用处**：① 可引用数字——headline 句的正式出处，别人引用抄表不抄图；② 经济账落点——$1/(1{-}h)$ 列是应用故事与数据的唯一硬连接；③ 接拒稿理由 ③（"SR 损失藏在操作点选择里"）的子弹——操作点标定集选定、eval 前冻结、逐点如实报损失；④ 部署者抄作业行（阈值+权重配置）。

**serving 实测裁决**：**不搭框架，跑 mini-bench 锚点**。纯推算 $1/(1{-}h)$ 会被系统审稿人戳（排队/batching/尾延迟全被假设掉）；但完整 serving 框架是 ROSA/Armory 那篇论文，做半吊子反被挑。折中形态基建全现成（`--replicas` + router + `docs/experiments/serving_benchmark.md` 五模式工具链 = 旧 X13 复活）：**$N \in \{4,8,16\}$ 并发 sim episode 压同一 GPU，实测 cache on/off 的 aggregate steps/s/GPU 比值**，与 $1/(1{-}h)$ 推算并排（差值归因排队与共享开销），进 App B 或 Table 1 加列；量级几百 ep、1–2 天。写作口径：claim 重心锚在 per-step compute 实测，serving 层是可信外推；调度优化引 ROSA/Armory 声明正交。

## 10. 主数据读出：已有 vs 缺口（2026-08-22 晚，源 = `exp/data_authority/analysis/gate_threshold_pareto/`）

### 10.1 已有：LIBERO pi0.5 侧主曲线 complete

threshold-pareto 主扫描收官（两 suite × 16 档 f_FH × 2 库 = 64 臂 × 500 ep = 32,000 ep，A-pool 全量、建库泄漏实测 0/50）+ gate-only 消融 2,000 ep。口径：teacher ratio = **决策级** MISS 率（每 5 控制步一决策）；$\mathrm{IR} = 0.152 + 0.848\cdot\mathrm{tr}$（**Stage1 地板 0.152**——全命中也付 key 构建 s1=10.26ms，CUDA-Graph 档实测）。

关键数字：

- **libero_spatial（ws 库）**：tr 0.89→0.33 全程 = teacher 调用省 63%、SR 98.8%→94.0%（−4.8pp）；**甜点 tr 0.50 处 SR 97.9%——省一半 teacher 调用损失 <1pp**；ws 库全程支配 cs 库（+0.7~+1.9pp）。
- **libero_10**：更陡——tr 0.89→0.38 时 SR 84.6%→72.8%（−11.8pp）；**tr 0.70 处 cs 83.5%（−1.1pp）= 省 30% @ ~1pp**；tr 0.50 处 ws 79.7%（−4.9pp）；两库前沿 ~0.67 交叉，pkl 选择依工作点。
- **双 suite 合读 = 冗余随任务难度分层**（四元组 $\mathcal{M}$ 依赖的自然体现），headline 候选句就位：spatial "≈50% 决策免 teacher、SR 损失 <1pp；63% @ <5pp"；l10 "30% @ ~1pp"。
- **gate-only 消融（白捡的部件分解）**：门的固有干预地板 22–25%（probe+滞回+L 锁定所限）；verdict 边际价值 = 再省 9–13pp teacher 但 SR 掉 8–11pp 且是**掉档非前沿延伸**；两库差异在此极端点消失 ⇒ 检索库质量优势需 verdict 在场才兑现。"gate 管何时问缓存、verdict 管这条配不配执行"直接进 §5 分析。

**口径修正（constitutional，改 §9）**：robots-per-GPU ≠ $1/(1{-}h)$——命中也付 Stage1，GPU 占用放大倍数 = **$1/\mathrm{IR}$**：$\mathrm{tr}=0.33 \Rightarrow \mathrm{IR}\approx 0.43$ → **$2.3\times$**；理论上限 $= 1/0.152 \approx$ **$6.6\times$**（非 $\infty$）。这个地板就是 savings-ceiling 的实测版，写对更诚实也更有内容（无穷上限本来就会被审稿人打）；应用故事中 $1/(1{-}h)$ 只作"忽略 key 成本的理想式"出现，正式列一律用 $1/\mathrm{IR}$。

### 10.2 缺口（还没有的，如实标注）

| 缺口 | 状态 |
|---|---|
| **纯 teacher 锚点**：$\varepsilon$ 的参照 $\mathrm{SR}_T$ 目前用 tr=0.89 端点近似，需正式 teacher-only 臂或老数据锚定确认 | 待确认（可能有老数据可引） |
| **oracle 臂**（sim-state 检索，Fig 2 虚线层） | 未跑（全新实验之一） |
| **RoboCasa365 双 teacher 曲线**（主图另外两小格） | 在跑（ws_search round-1 → T7 评测） |
| **跨场景迁移正式数据**（§6 章主结果 + 任务类型分层） | 在跑（初步观察仅记 §7.1-4，待 T7） |
| **warm-start 三方对照**（App B） | 未跑（全新实验之二） |
| **serving mini-bench**（$1/\mathrm{IR}$ 的实测锚点） | 未跑（全新实验之三） |
| **E10 router 对决**（§7.2 实证载体，旧 X15） | 在跑（G1 过，待 Code） |
| **headline 等价判定口径**（TOST 界 / −1pp）与贡献句量词 | 未拍（§11-1/-6，数字已在手可拍） |

## 11. 框架性未决问题（状态更新于 2026-08-22 晚）

| # | 问题 | 状态 |
|---|---|---|
| 1 | **Headline claim 量化口径**：「__% 步可回放，SR 损失 $\le$ __」——X 从哪条数据读、等价判定（TOST 界 / −1pp?）、scope | **数字已在手（§10.1 候选句），等 owner 拍措辞与等价界** |
| 2 | 冗余度定义 | ✅ 已拍（§4） |
| 3 | **主 x 轴**：compute / latency / robots-per-GPU 三选一 primary | **未拍**（§9 倾向 compute 实测为 primary、经济量为推算列，待确认） |
| 4 | 主战场 | ✅ 部分拍：主图 2 benchmark × 2 teacher 网格，fallback 预案在 §6；headline 数字出自哪格未定 |
| 5 | Baseline 集合 | ✅ 已拍：小模型进 Table 2（获得成本列 + regime 地图，§7.1）；cheapening 降 App + 正交叙事（§7.3） |
| 6 | 贡献 bullet 三条最终措辞 | **未拍**（框架在 §5 提纲 P5，动词量词待 headline 口径） |
| 7 | 老数据复用边界 | ✅ 大体清（§6 表现状列）；E3 口径差异须注明 |
| 8 | 诚实边界清单 | ✅ 框架在（§5 提纲 §8 行），写作期成文 |
| 9 | 三大拒稿理由预埋 | ✅ 指定：①"RT-Cache+阈值增量" → §2 划界 + §4.3 + oracle 臂；②"玩具+sim-only" → 2×2 主图 + §8；③"SR 损失藏操作点" → Table 1 预注册 |

## 12. 合作者会议 Deck 结构（20 帧，~35 min + Q&A；owner 定稿 2026-08-25）

会议化三原则：前三张先给结论；TODO 分"纯执行"与"待合作者决策"两类；defense 压一张表、细节进 backup。工作文件 `slides/deck.tex`（Focus 主题自包含，`xelatex` 编译；目录已 gitignore）。**帧 1–4 已制作**。

- **Part 0 — Opening（3 min）**：1. Title（星标题目）；2. The claim, then the numbers——thesis 一句 + headline 三大数字（50%@<1pp / 63%@<5pp / 30%@~1pp），底注口径 skip = hits / total requests（decision level）。
- **Part I — Why this matters（5 min）**：3. Robots repeat themselves; VLAs recompute everything——抽屉金句 + "serving papers schedule demand (ROSA/Armory); nobody deletes demand"；4. Why it matters: latency and throughput——**已按 §3 切分修订制作**：per-robot latency（命中步塌缩本地毫秒，均值/P50 降、tail 双峰化脚注）× per-fleet throughput（集中部署趋势 + robots-per-GPU = $1/\mathrm{IR}$ → 6.6×，"Hits cut GPU time per step (Stage-1 always runs)"）。
- **Part II — What we measure（7 min）**：5. What counts as a redundant step?——do-干预定义、挂结局不挂动作相似度；6. Why closed-loop measurement is forced——组合性破缺 remark、前人离线 metric 结构性测不到；7. $R(\varepsilon)$: a constrained sup——公式 + "every deployed scheduler is automatically a lower bound" + 夹逼三层（oracle 臂标 TODO）；8. Whose property is $R$?——四元组、intrinsic vs recoverable（挖矿类比一句）。
- **Part III — The system（6 min）**：9. Keys from the teacher's own forward pass——三重红利 vs RT-Cache 外部 encoder（加粗点专属帧）；10. Three execution tiers + calibration——replay / warm-start / infer、calibration = empirically solving the $R$ program；11. Deployment lifecycle——collect → shadow → calibrate → run & append 四阶段时间轴。
- **Part IV — Evidence so far（10 min，全真图真数）**：12. Main frontier on LIBERO——帕累托 + 甜点 + gate/verdict 分解一句；13. Redundancy grows with experience（E3 规模线）；14. Redundancy comes in blocks（三 panel 结构图）；15. Does experience transfer?——RoboCasa365 初步观察（大尺度任务迁移最好）+ T7 running；16. Ledger E0–E12 状态板。
- **Part V — Positioning & defenses（4 min）**：17. Why not the alternatives——一张表（student regime map / trained router counterfactual labels / cheaper teacher orthogonal+warm-start）；18. Where this sits in the literature——六族一图 + first 句（2026-08-26 统一版：definitional first）。
- **Part VI — Plan & asks（5 min）**：19. Path to the deadline——三个决定件（E2 oracle / E6 close-out / RoboCasa grid）+ 时间线；20. Decisions I need from you——SO-101（main vs rebuttal）、title 二选一、分工。
- **Backup（不讲被问再翻）**：math walkthrough 逐块读 / monotonicity proof / full defense ammo / reviewer three-bullet 预测 / per-config tables / AUROC 计划 / LLM-cache 谱系。

## 13. 待讨论

（后续讨论逐条追加）
