# 跨领域迁移调研：本项目 cache 系统简报（给调研 agent 读）

> 目的：判断我们这套 **tiered experience cache + 风险索引阈值（RIT）+ 状态门** 能否迁到 VLA 以外的领域，并且在那里 **收益明显**（在 VLA 上收益太小）。
> 读完本文，再读 `docs/iclr/iclr_paper/arxivd.tex` 第 64–344 行（问题定义 + 方法三小节，这是**现行方法**；同目录 `iclr2027_conference.tex` 是另一版，**不要读**），
> 以及 `logs/session_handoff.md` §1.2（今天的减步 teacher 基线终态）。其余文档按需查（`docs/architecture/cache_system.md` 实现细节；`docs/iclr/actioncache_response_plan.md` 竞品 ActionCache）。

## 1. 系统是什么（与领域无关的形式）

被服务的模型是一个 **共享前缀（shared prefix）+ 迭代生成头（head）** 的推理管线。VLA 里：前缀 = 视觉编码器 + VLM 前缀（π0.5：s1=10.3 ms + s2=27.7 ms），头 = flow-matching 动作专家，N 步 Euler 去噪（s3=29.6 ms，10 步）。每个决策（每个 action chunk）都要跑一遍。

cache 插在前缀和头之间（tap point = 前缀算完之处）：

1. **Query 免费**：key 直接取前缀已经算出的特征（各相机 token map 池化、本体状态向量、指令 embedding 均值），零额外成本。
2. **索引 = IVF**：按指令 embedding 分 cell；cell 内逐字段打分（相机 cosine、本体负欧氏距离），每个字段用"库对自身、逐 episode 留一"估出的 μ/τ 做 tanh 标准化到 [0,1]，再加权求和成 **一个分数 s_t**（式 norm/fuse）。
3. **档位（tiers）**：tier 0 = miss（全算）；tier K = hit（直接回放库里存的 action chunk，成本只有 c_pre）；中间 = **warm start**：从库条目存的**中间去噪状态**出发、在当前观测条件下只跑剩下 ρ_a 比例的头步数。成本 c_a = c_pre + Σ_{j≥a} Δ_j，c_pre = c_K < … < c_0。
4. **RIT（Risk-Indexed Threshold）**：档位阈值 θ_1 ≤ … ≤ θ_K 不用 rollout 网格搜，而是**离线**：在 shadow 数据（策略自己跑的校准集，事后离线检索打分、不执行）上，对每个 tier 计算"该 tier 产出的 chunk 与从纯噪声全算的参考 chunk 之间的标准化偏差 D_a"，拟合 **分数条件下的 (1−α) 分位曲线 q_a(s)**（order-restricted quantile regression，随 s 单调不增、高 tier 曲线不低于低 tier），然后**一个容差 δ 同时定所有 cut**：θ_a(δ) = min{s : q_a(s) ≤ δ}，dispatch 取分数够到的最高 tier。δ 扫过去就是一条 ladder；shadow 数据上能预测每个 δ 的 IR，于是给定算力预算 B 可以反解 δ。理论依据：各 tier 在同一风险水平时总代理风险最小。
5. **状态门（stateful gate）**：命中成串（上一决策的分数预测下一次是否命中）。连续 j 次低分就停止检索、每 p 步探一次；连续 L 次 hit 后强制一次 miss 防漂移。
6. **训练无关**：三阶段 collect（策略自跑、成功 episode 入库）→ shadow & calibrate（全部常数离线拟合）→ serve。
7. **评价量**：inference ratio IR = 平均每决策成本 / 全算成本；成功率 SR；可恢复冗余 R_C(ε) = 1 − inf IR s.t. SR ≥ SR_T − ε；对偶 V_C(B)。整个故事是 **(IR, SR) 帕累托前沿**，且必须闭环测（单步替换安全 ≠ 连续替换安全，会漂移）。

## 2. 在 VLA 上的实际数字（为什么说"收益太小"）

- hit 的成本地板 c_pre/c_0 ≈ 15%（π0.5 15.2%、GR00T 14.8%）。
- LIBERO（500 集/臂）：π0.5 libero_10 有门前沿在 IR 60 时 SR≈0.83、IR 80 时≈0.89（teacher 0.92）；GR00T libero_10 RIT 三档在 IR 48 只有 0.61、IR 80 才到 0.86（teacher 0.868）。IR 低于 40 的区间只有 cache 能到，但 SR 掉得厉害（GR00T cache-only 锚点 IR 14.8 / SR 0.434）。
- **今天的减步基线（不带 cache，只把头的去噪步数从噪声起砍到 k）**：π0.5 spatial k=1 就 0.988（=teacher）；π0.5 l10 k=1…7 全在 0.82–0.85（teacher 0.92，8 pp 平台）；GR00T spatial/l10 k=1…7 与 teacher 无差。减步的 IR 地板 = (s1+s2)/c_0 = 60.6%（π0.5）/ 40.7%（GR00T），因为前缀必须跑。
- 含义：**一步 Euler 从高斯噪声出发 ≈ 输出分布的条件均值**。输出分布单峰（LIBERO）时均值无损，warm-start 档就没有 counterfactual，系统退化成 hit-or-miss 二元；输出多峰（ActionCache 报 VLABench 一步 38.8→6.8）时均值崩，warm 档（从检索到的真实样本出发）才有存在理由。
- 同类竞品：ActionCache（arXiv 2607.06370，VLA 语义 cache，hit/warm 两档、key=VLM 输出、成本地板 45–53%）。

## 3. 迁移目标必须同时满足（打分表，每项 0–3）

A. **结构匹配**：有昂贵共享前缀 + 相对便宜、有中间状态可 warm-start 的迭代头（扩散/流匹配/迭代精炼/迭代求解器）；前缀末端有天然 tap point 能白拿 query 特征。
B. **成本余量**：c_pre/c_0 小（hit 省得多）；且"直接减迭代次数"这个傻基线在该领域**会掉点**（输出多峰 / 迭代敏感），否则 warm 档没意义。必须给出该领域的"k=1 基线"文献证据或论证。
C. **局部性 / 命中潜力**：query 分布有重复或近重复（时序成串、同场景多次、多个用户/agent 共享库）；给出文献里能支撑的命中率估计。
D. **容忍度与闭环**：近似输出可接受、错了能被后续纠正或感知不到；存在类似 SR 的成功度量与容差 ε；且存在"连续替换会漂移"的闭环性质（否则只是离线近似计算，故事变弱）。
E. **RIT 可用性**：能离线构造 shadow 信号（tier a 输出与从头算输出的偏差 D_a）、偏差随分数单调、一个 δ 切所有 cut、能按预算反解；状态门有无对应物（命中成串）。
F. **novelty 地形**：该领域已有的 cache / 投机解码 / 特征复用 / warm-start 工作是什么，我们的 tiers + RIT + gate 在那里还剩多少新东西；审稿人会拿什么打我们。
G. **我们做得动**：开源模型 + benchmark + 成本模型可测；算力（1×H100 80G、1×4090 49G、若干 A5000/1080 仿真机）；三周内能出第一条前沿；每个前沿点的评测成本。

## 4. 报告格式（写到 `logs/cache_transfer/<slug>.md`，中文，术语保留英文）

1. 领域内 2–3 个最贴合的具体系统（模型名、代码是否开源、benchmark、成本拆分数字或估计）。
2. **映射表**：前缀 / tap point / 头 / 中间状态 / tiers 定义 / key 字段 / 分数 / 偏差 D_a / 成功度量 / 成本模型 / 闭环性 —— 一一对应到我们的符号。
3. 打分表 A–G（每项 0–3 + 一句依据 + 引用 URL）。
4. 该领域的"减迭代次数"傻基线：文献里有没有人做过，结果如何；如果没有，估计会怎样并说明理由。
5. 预期收益：hit 率、IR 能压到多少、SR 损失多少（尽量给文献数字）；与 VLA 的对比。
6. novelty 地形与最强反驳。
7. 首个实验方案（≤ 2 周）：用什么模型、什么 benchmark、先测什么（第一步永远是傻基线 + 成本拆分实测）。
8. 结论：GO / MAYBE / NO，一句话理由。

规则：只读仓库、只写自己的报告文件；不跑实验、不装东西、不碰 git；引用要有 URL；不确定就写"不确定"，不要编数字。
