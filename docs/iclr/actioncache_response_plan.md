# ActionCache 攻防方案（对 arXiv 2607.06370 的核实、定位与对照实验计划）

> 背景：外部评估指出 ActionCache（2026-07 arXiv）与本文原叙事高度重叠、novelty 被打穿（当前 outline 直投估 8–18%）。本文档 = 2026-08-26 三路专家并行深读原文后的整合方案：逐项核实、写作定位、弹药库、四臂对照实验。
> 结论先行：**原叙事确实被占；但重定位（pre-backbone early exit + temporal reuse）后差异真实、可量化，且对照实验的复现臂 = 激活我们自己搁置的 CP2 checkpoint，核心新增预算仅 ~3,000 ep。**

---

## 1. 核实结论：他们做了什么、没做什么

**ActionCache**（Oi et al., Institute of Science Tokyo, arXiv 2607.06370，*"Training-Free Acceleration for VLA Models with Action Caching and Refinement"*）：

- **Key**：VLM **输出** embedding（backbone 跑完后），逐 token flatten（π0.5 D=1,982,464 / GR00T-N1.6 D=593,408，后者拼 robot-state），固定稀疏三值随机投影 $k_t = Rh_t$ 压到 d=500；**单一向量、单一 cosine top-1，无分模态**。
- **Hit 行为**：缓存去噪轨迹第 $N{-}N_{hit}$ 步的 noisy 中间态，命中后直接执行或跑 $N_{hit}$ 步 refinement——**即我们的 warm start 档**；$T_{hit}$、$N_{hit}$ 均为固定超参。
- **库**：在线 rollout 建库（prefill $T_{hit}=1$ 全收 → pending buffer → **episode 成功才 commit**）+ LRU/LFU 淘汰。
- **评测**：VLABench + LIBERO 四 suite 闭环 + **SO-101 真机**（3 任务 × 50 ep，demo 级）；LIBERO 招牌数字 96.3% SR @ 91.88% hit（NFE=1）。
- **没有的**：形式化冗余量 / SR 约束标定（阈值 = 看 hit-rate 曲线"肩部"逐任务人工调）/ 任何时间结构分析或 stateful gating / 跨场景（只有跨任务）/ serving 经济与生命周期。

**结构性事实（我们全部差异的根）**：key 生在 backbone 出口 ⇒ **backbone 每步必跑**。每步成本地板：他们 ≈ **45–53%**（真机 Table 2 反推 45.1%；我们 CUDA-Graph 档换算 53%），我们 ≈ **14%**。端到端他们实测 **1.26×**、理论天花板 ~2.2×；摘要的 10.44×/40.17× 是 action-head-only 口径。

**措辞纪律（防反杀）**：
- "一定比他们省"只在两层无条件成立——**单命中步成本**（我们省 S2+S3 是他们省部分 S3 的严格超集）与**可达成本区域**（15–45% 区间他们构造上到不了）。**端到端总账是条件性的**：平均节省 = 命中率 × 每步节省，他们命中率 91.88% vs 我们 50%，且依赖编译档——CUDA-Graph 档我们 ~40% vs 他们 ~34%，**eager 档（action head 占 77%）他们反超**。主张一律锚定"结构性更低的地板 + 扩展可达区域"，禁写 "we always save more end-to-end"；总账数字钉死编译档。

---

## 2. 规则保护（ICLR 2027 官方原文，2026-08-26 抓取核实）

- Contemporaneous 窗口 = 截稿前**两个月**内的 **peer-reviewed** 发表；
- **arXiv 不算 peer-reviewed venue**："authors are not required to compare to papers solely on arXiv … the lack of such comparisons **cannot be a basis for rejection**"——ActionCache 目前 arXiv-only，此豁免**无条件**、与时间窗无关；
- 即使它 9 月前被某会录用，也几乎必然落在两个月窗口内。
- **含义**：程序上我们不需要打赢它来自证 novelty；做对照是策略选择（更硬、且防作者本人在审稿池），不是义务。
- ⚠ 官方页面两处截稿日期不一致（9/16 vs 9/25），临投稿前重新核对。

---

## 3. 写作定位策略

1. **门面唯一定位 = "concurrent and independent work"**（intro + related work 各出现一次），不用 "building on"（事实错误），"generalization that subsumes" **不做门面**——降级为设计空间表格那节的附带技术观察，措辞用 "corresponds to one configuration of / is recoverable as"，禁用 "merely / degenerate / subsumed by"，且**每次说必须同段给对方记功**（40× action-head 加速、真机验证、oracle 上界消融）。风险意识：这个窄领域 ActionCache 作者很可能就在审稿池。
2. **Priority 问题**：正文**什么都不放**——毕设脚注绝对禁止（双盲去匿名 + 不可验证），任何"我们更早"的文字暗示全删；"concurrent and independent" 七个字承担全部工作。Camera-ready/录用后再补项目历史，与投稿决策分离。
3. **标题**：保 "Muscle Memory" 品牌，副标题换成点明切点的版本，推荐：*"Muscle Memory for Robot Policies: Early-Exit Caching Before the Vision-Language Backbone"*。摘要第 1–2 句先立 "VLA 推理成本 = backbone + action head 两段" 的分解，让略读者立刻看出我们在 pipeline 更早的位置。
4. **Related work 首段草稿（英文，已备）**：
   > Closest to our work is ActionCache (Oi et al., 2026), concurrent and independent work that accelerates flow-matching VLA inference by caching intermediate denoising states to warm-start future generations. ActionCache retrieves via cosine similarity on a single key extracted from the VLM's output embedding after the backbone runs, using a per-step threshold set by profiling the hit-rate curve, and treats each timestep independently; it reports large, hardware-validated action-head speedups, though its cost floor is necessarily bounded below by the backbone's own cost. Our approach differs in five respects: keys are extracted before the backbone, lowering the cost floor further; retrieval uses weighted per-modality fields rather than one vector; the hit threshold is calibrated under an explicit success-rate constraint via a formal quantity $R(\varepsilon)$, measured with a lower-bound protocol; gating is stateful across a rollout; and we evaluate transfer to a held-out kitchen scene.
5. **三层攻防分工（owner 已裁）**：
   - **正文最小集（3 处，全部中性/自利表述）**：RW 一段（上文草稿）；主表一行（四臂对照的 Arm1 = 头号 baseline，藏附录会被反问）；地板对比 53% vs 14%（这是我们 early-exit 贡献的支柱表述，不是攻防）。
   - **附录攻防章 "Extended comparison with concurrent work"**：逐条弱点审计、复现协议与保真度、还功清单——**检察官内容、公证人语气**（"the LIBERO prefill protocol is not specified"，不写 "they hide"）。与已裁的 "Design choices and alternatives considered" 附录节并列。

---

## 4. 弹药库（按杀伤力排；正文只用 1–3，其余入附录）

1. **LIBERO 上被零工程量基线帕累托支配**（他们 Table 7）：Base NFE=1 = 96.9% SR / 5.84ms **支配** ActionCache NFE=1 = 96.3% / 11.31ms；ActionCache NFE=0（6.57ms）也比裸基线慢且 SR 崩到 92.1%。⇒ flow 基座 NFE=1 几乎不掉点，**action-head 赛道整个 trivial 化——唯一非平凡的省法是跳 backbone**（我们的独占区）。⚠ 对我们同样是警告：E8 三方对照里盲减步臂在 LIBERO 上极强，warm start 别在 action-head 轴上寻求胜利。
2. **端到端 1.26× / 天花板 2.2× / 地板 45.1%**——全部从他们自己的表反推；摘要 10.44×/40.17× 是 action-head-only 叙事框架（数据透明、框架误导——批评叙事不批评数据）。
3. **LIBERO prefill 防泄漏披露真空**：VLABench/真机明确声明 disjoint seeds，唯独 LIBERO（91.88% 招牌数字所在）全文无一字防泄漏说明；LIBERO init 池有限，状态邻近虚高风险。我们的 init 池纪律（0/50 实测）与之成对照。
4. **阈值 = "看肩部" 逐任务人工调**，选取目标（maintain SR）与评测指标重合，无形式化保证、无可复现算法 ⇒ 我们 SR 约束标定协议的价值锚点。
5. **零时间结构**：逐步独立判定，1000+ ep 长跨度测试从未统计 run-length/粘滞 ⇒ E4 + gate 线独占。
6. 无形式量（一切经验曲线）⇒ R(ε) 独占。
7. 跨任务实验条件收紧（T_hit 提到 0.925 + 冻结 cache 的精选正例），无跨场景 ⇒ E6 独占。
8. 单一拼接向量的模态淹没问题（他们自己承认视觉剧变阶段相似度掉）⇒ E7c 直接打靶。
9. 统计不对等（己方 4 seed × 200 vs 对比法 1 seed；真机 50 ep 无误差棒；eviction 消融只报 hit rate 不报 SR）。
10. 真机 demo 级 + "近失败恢复轨迹"是否入 cache 未披露。
11. 陈旧动作在接触敏感阶段（他们承认相似度最低处）无任何保护 ⇒ 我们 stateful gate 的安全叙事。
12. "长跨度无退化" = 自我强化闭环，非真实分布漂移。

**必须还功的八点**（防"无脑贬低"观感）：组件延迟披露透明 / 超参扫描完整（d、p、seed、容量 50–5000）/ eviction 消融含 oracle 上界 / 真机落地 + 近失败恢复轨迹的采集思路 / key 来源消融（output vs input +29.1%）/ 诚实的 limitations / VLABench 与真机确实做了 disjoint-seed / 跨任务方向正确（我们是"更难的延伸"非"他们没碰过"）。

---

## 5. 四臂对照实验方案（核心发现：Arm1 = 激活我们搁置的 CP2）

### 5.1 臂-checkpoint 映射（方案地基，正文必写的一句：对照点由**他们的** key 来源决定，非我们挑选）

| 臂 | Checkpoint | key | hit 省 | 现状 |
|---|---|---|---|---|
| **Arm 1** ActionCache 忠实复现 | **CP2**（S2 后；`types.py:55` 枚举 Reserved 未接线） | S2 输出 flatten + 固定稀疏三值投影 d=500，单 key cosine | 只省 S3 | **需新建**（有骨架） |
| **Arm 2** 早 key + per-step threshold | CP1 | prefix_embs pool，多字段加权 | 省 S2+S3 | **现成**：threshold-pareto 32,000 ep |
| **Arm 3** 早 key + stateful gate | CP1 + `ScoreHysteresisGate`（N1/N4, L=6） | 同上 | 同上 | **现成**：gate 线 live 数据 |
| **Arm 4** 早 key + block/lockstep 盲回放 | CP1 + `FollowWinnerGate`（N2） | 同上 | 另省命中段 search/judge | **现成**：N2 live（含已知 SR 脆弱性，如实报——它是前沿最外端+风险的存在性证明） |

理论 hit 地板（CUDA-Graph 实测换算）：Arm1 ≈ s1+s2+检索 ≈ **42.1ms**（上限 1.72×）；Arm2–4 ≈ s1+检索 ≈ **14.4ms**（上限 5.03×，即 IR 地板 0.152 → 6.6×）。⚠ Arm1 检索成本须独立实测（key 维度/算子不同，不得照抄 4.15ms）。

### 5.2 忠实复现的两个判断调用点

- **单 key 退化**：Arm1 必须显式关掉我们的多字段融合（系统默认会"做得更好"，那就不是在测 ActionCache）——单 key cosine 模式已存在。
- **建库模式**：他们在线建库 vs 我们离线库。**Mode A（主）**：四臂全部用同一批 `libero_cache/<suite>/*.h5` 离线建库——库内容严格同源、零新增 rollout；**Mode B（次，仅 Arm1，附录/rebuttal 级）**：真实现 pending buffer + commit-on-success 做 faithfulness check。

### 5.3 匹配变量与校准（courtesy 2×4）

匹配：库源轨迹 / entry 数（注明不匹配字节数）/ 校准与评测 init 池纪律（0/50 泄漏协议沿用）/ 配对评测 init / CUDA-Graph 编译档。按设计不匹配：检索结构（各系统定义本身）、warm-start 可用性（各提供 $N_{hit}{=}0$ 端点对齐"直接执行"格）。淘汰：主线四臂统一 append-only；LFU 容量 ablation 独立进附录。

**校准 2 协议 × 4 臂全跑**：P1 = 肩点启发式（纯离线后处理，成本≈0）；P2 = 我们的 SR 约束标定。每臂曲线标两种记号——**"Arm1 拿了 P2 仍输"才是最干净的结论**，直接排除"baseline 校准太弱"的头号反击。

### 5.4 统计与 readout

按 ε（SR 损失预算）对齐操作点（不按阈值原值——不同 key 空间尺度不可比）；配对 McNemar（Holm 校正 6 组两两）+ episode 级 cluster bootstrap 对 (cost_A − cost_B) 直接出 CI；每臂独立实测拟合 成本=f(命中率) 曲线（IR 公式是 CP1 专属，不得硬搬 Arm1）。Readout 轴：SR / 实测 GPU-s/step / 端到端延迟（均值与 P99 分开，禁裸写 lower latency）/ backbone-call 占比 / **hit-run-length 分布**（预注册预测：Arm4 偏离几何基线最远、Arm1 最接近、Arm2/3 居中——可证伪假设进图注）/ lockstep purity（诊断 Arm4 失败模式）。

### 5.5 预注册解释分支（六条，摘要措辞提前备好）

① Arm2 只赢成本、同成本 SR 更差 → "早 key 给更高压缩天花板，SR 由 gate/threshold 买回"——坐实 gate 是核心部件；② Arm3 对 Arm2 无增量 → "threshold 已近该信息集可达域"（正面写，但禁过度泛化）；③ **Arm1 同成本 SR 反超**（最危险分支）→ 让步为双 regime 地图（早 key = 高压缩/延迟预算场景，post-backbone key = SR 优先场景），摘要措辞提前写好；④ Arm1 拿 P2 仍全面输 → 最干净，headline 可讲结构性优势；⑤ Arm4 成本最优但漂移大（与 N2 live 一致）→ 前沿外端存在性证明 + 需漂移保护；⑥ 全臂 SR 不可分 → 退守"同质量下结构性更低地板"（地板差异不依赖实验运气）。

### 5.6 工程量与预算

工程（临界路径序）：① CP2 Handler 接线（仿 CP1/CP3）② `ProjectionKeyBuilder` 加 `fixed_sparse_ternary` 模式 ③ 单 key YAML ④ warm-start 挪 CP2（不动 CP1 现行为）⑤ 离线 CP2 builder ⑥ GR00T 侧 S1/S2/S3 微基准。附录级：⑦ LFU ⑧ Mode B pending buffer。

| 层 | 内容 | 新增 ep |
|---|---|---|
| **deadline 前必须** | Arm1 vs Arm2（π0.5 × spatial，P1+P2，含 $N_{hit}$ 4 档）| ~2,500 |
| **deadline 前必须** | teacher-only 锚点臂（顺手补纪要 §10.2 的锚点缺口） | ~500 |
| **必须（零 ep）** | Arm3/4 现成数据按统一 GPU-s/step 口径重画 + 四臂主图 + run-length 图 | 0 |
| 余量应做 | libero_10 扩展 / GR00T 四臂（先跑其微基准，零 rollout） | 各 ~2,500 |
| 附录/rebuttal | RoboCasa 四臂（避与 T7 撞排期）/ LFU 容量 / Mode B / 全 $N_{hit}$ 网格 / 单 key 版 Arm2 / SO-101 真机四臂 | 按需 |

**核心新增 ≈ 3,000 ep**——三臂白嫖现成数据，唯一花钱的是 Arm1。

---

## 6. 对既有文档的影响清单

- **E 系台账**：新增 **E14 = 四臂 ActionCache 对照**（Arm2/3/4 数据复用 E1/E4/gate 线；Arm1 为新采集）；E8 warm start 的角色更新——其机制已被 ActionCache 占位，价值转为三档谱系的中间档与 Arm1 的执行体组件；E13 真机建议从存在性证明升级为真机四臂（他们的 SO-101 是 demo 级，对照式真机直接压过）。
- **提纲**：题目副标题换 pre-backbone 版本；§2 RW 首段换本方案草稿；§5 主表加 Arm1 行；附录加 "Extended comparison with concurrent work" 章；摘要前两句立两段式成本分解。
- **纪要相关章节**（§2 novelty、§7 defense、§11 未决问题）后续按本方案修订。

## 7. 行动清单（按序）

1. Owner 拍板：四臂方案与预算、E14 入编、题目副标题、Arm1 工程排期（CP2 接线走 L2/L3 流程）。
2. 立即项：CP2 Handler + 投影模式（§5.6 ①–⑤）→ Arm1 spatial 首跑。
3. 写作项：RW 首段与附录攻防章按本文档 §3/§4 落墨。
4. arXiv 时间点与导师商定（重定位初稿完成即挂，先占时间戳防下一个撞车者）。
