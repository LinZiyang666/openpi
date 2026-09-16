# TRACER Phase 7 — 负结果整合 ablation / 评测报告（Plan）

> **Status**: Plan（G1 APPROVED R5 2026-07-14；§4 Code DONE；**G2 APPROVED R4** 2026-07-15；§6 Verify green — 2452 passed / 30 skipped，2 个既有环境失败（GCS 网络 + main.py 源码锁 2≠3）经 owner 明示裁决 proceed，均与本 doc-only 改动无关）
> **Level**: L2（走 Understand → Plan → G1 → Code → G2 → Verify）
> **Authority**: Execution
> **上位依据**: [`tracer_retrieval_refinement_roadmap.log.md`](tracer_retrieval_refinement_roadmap.log.md) §4 Phase 7
> **前置**: Phase 4 ✅ / Phase 5 ✅（出场门 FAIL 亦为可用判决）/ Phase 6 ✅（①b NO_GO）

---

## 0. G1 Reviewer 上下文摘要（无需对话历史即可读）

TRACER 检索精炼线（合作者提案 `TRACER_RETRIEVAL_REFINED_PROPOSAL.pdf`）落地为 3 机制：**M1 结果兼容投影**（新 KeyBuilder，Phase 2 骨架 / Phase 6 训练）、**M2 失败感知双检索**（D⁺/D⁻ margin 挡不安全复用，Phase 3 骨架 / Phase 5 标定，Claim 1 最硬）、**M3 动态链深**（Phase 1）。关键路径 = 1→3→4→5→7。

到 Phase 7 为止，**已跑到的运行时/离线判决**如下（下述三条均为「已评估配置/协议内」的判决，**不等于**提案 §11 的 proposal-literal Claim——见本节末「re-scope ↔ proposal-literal」）：

- **Phase 5（M2 双检索标定）运行时出场门 FAIL**（两 suite）——**已评估的 dual full-hit 操作点**（illustrative + calibrated，两 YAML 均 `enable_dual: true`）令 online SR 崩，未过 `SR_cal ≥ SR_base−ε` 门。**直接证据 = 替换 rollout 的真在线 SR 下降**；离线逐步 `L_cal` proxy 与在线 trajectory SR 不相关，是**与 SR collapse 一致的机制解释**（非逐步反事实因果断言）。证据已固化在已提交的 `exp/zixuan_proposal/analysis/phase5_scoresum_findings.md` + 两 suite verdict 卡。
- **Phase 6 ①b 的 reduced 离线 rescue 门 = NO_GO**（两 suite）——混淆-free July 数据训练投影头后，**仅比较** raw A′ vs action-only 投影 B 在 I_cal 上的 safe-reuse 预测 AUROC，投影**零显著增益**（ΔAUROC CI 含 0）。**未执行**提案预注册的 B-vs-C（action-only vs action+denoise）、亦无 downstream SR/IR。证据已固化在已提交的 `exp/zixuan_proposal/analysis/phase6_ib_offline_gate_report.md`。
- **纯 warm-start 成本**：owner 口述既往结论「与全量 inference 开销可比、基本不省算力」，**本 repo 无已提交的 latency/cost .md 产物**，且仓库 warm-start runbook 将 `start_t` 定义为节省一部分 Stage 3（与「不省算力」表述相左）→ 本条**仅作未独立复核的背景**，不承重、不进任何判决。

**Phase 7 的定性转向（owner 裁 2026-07-14，已写入 roadmap §4 Phase 7）**：Phase 7 不再是"证明 Pareto 赢面"，而是把**已跑到的负判决**整合成一份严谨、可追溯、有出场门口径的评测/ablation 报告，并对整条 TRACER 检索精炼线在**已评估边界内**下结论。**评测实质数据大半已在手，本期零新 GPU rollout、零框架/src 改动**——纯 `exp/` 写作 + 已提交证据综合。

**re-scope ↔ proposal-literal（关键区分，G1 Round 1 Blocking 1/2 要求）**：owner 的 re-scope 是「把已成定局的负结果整合成报告」，**不等于**完成提案 §11 定义的 proposal-literal ablation。三者对齐关系（报告须显式声明）：
  - **Claim 1（success-only vs dual）= NOT EVALUATED / de-scoped**：Phase 5 三跑是 cache-OFF baseline（仅供 SR）+ illustrative dual + calibrated dual，**无 success-only comparator** → 现有数只能证「这两个 dual full-hit 操作点未过门」，不能证 dual 相对 success-only 的 Claim 1。
  - **Claim 2（raw vs projection，含 denoise + downstream SR）= 仅 reduced offline rescue 门 NO_GO**：只做了 raw A′ vs action-only 投影 B 的 I_cal AUROC，B-vs-C 与 downstream SR/IR 未做。
  - **Claim 3（fixed vs dynamic depth）= NOT EVALUATED / owner-de-scoped**：无 fixed/oracle/dynamic depth 数据，owner 裁不补（2026-07-14）。**不得写成实证 ablation 的通过/失败。**

本期走 L2 全套门（而非 L0 纯文档直提）：交付物是一条主要研究线的**已评估边界内的负判决**（会影响该线后续投入），其推理链条与证据引用值得独立评审把关；L2 定级源于结论的可审性，而非代码复杂度（本期无代码）。

---

## 1. 目标

在提案 §11 Claim 1/2/3、§14 step 7、Eq 26–28（IR / BHR / FFR）口径下，**如实**交出**已评估边界内**的判决（不把 owner 停止决策/未评估比较写成实证 ablation 的通过或失败）：

1. 逐 Claim 给出**精确定级的**判决矩阵：
   - **Claim 1（M2，success-only vs dual）= NOT EVALUATED / de-scoped**；已评估的两个 dual full-hit 操作点 **FAIL 出场门**（`SR_cal ≥ SR_base−ε` 不满足）。
   - **Claim 2（M1，raw vs projection）= reduced offline rescue 门 NO_GO**（仅 raw A′ vs action-only 投影 B 的 I_cal AUROC；B-vs-C denoise + downstream SR/IR 未评估）。
   - **Claim 3（M3，fixed vs dynamic depth）= NOT EVALUATED / owner-de-scoped**。
2. 在 (SR, inf_ratio) + BHR/FFR 度量空间下呈现整合表：**在已评估配置与协议中，未观察到同时满足「安全（不崩 SR）」与「省算力」的操作点**（inf_ratio/IR 承载 Claim 1 真在线数；Claim 2 的离线 AUROC 单列、不与在线 SR 混算；warm-start 仅列背景不承重）。
3. 对整条 TRACER 检索精炼线给出**收窄到证据边界内**的结论：**在已评估的两 suite、pi05、LIBERO、特定 score-sum/门参数下，失败感知检索 cache 的 full-hit 替换未观察到「既安全又省算力」的工作点**（负结果）；同时明确 re-scope 与 proposal-literal claims 的差异。

**非目标**：本期不跑任何新 rollout、不改任何 src/exp 机制代码、不新采数据、不重训任何模型。所有数字来自已提交的证据报告。

---

## 2. 交付物与改动文件

| # | 文件 | 动作 | 说明 |
|---|---|---|---|
| 1 | `exp/zixuan_proposal/analysis/phase7_tracer_ablation_report.md` | **新建** | 唯一实质交付物——整合 ablation/评测报告（纯 .md，落 `exp/<exp>/analysis/`，符合实验最终报告落点约定） |
| 2 | `logs/tracer_retrieval_refinement_roadmap.log.md` | 编辑 | Phase 7 状态 🟡→✅（=**报告交付完成**，非实证 ablation 通过）回填 + 精确定级判决（Claim 1 de-scoped/dual 操作点 FAIL、Claim 2 reduced 门 NO_GO、Claim 3 owner-de-scoped）+ 顶部 Status header 行同步（roadmap 自带"每期落地后回填其状态与判决"约定） |
| 3 | `logs/README.md` | 编辑 | 加本 plan 文件（`tracer_phase7_...`）索引行（Index Sync Rule 红线：logs/ 改动同 commit 同步索引） |
| 4 | `logs/tracer_phase7_negative_result_integration.log.md` | 新建（=本文件） | 本 plan（G1 APPROVED R5，Review Log 已按 §3.1 删除）；G2 将在此另开新 Review Log |

**明确不触及**：任何 `src/`、任何 `exp/**/*.py`、任何 config、任何 test。`exp/**/analysis/*.md` 不在 `docs/`/`logs/` 索引红线辖域内，故交付报告本身**不需**索引同步（仅 logs/ 侧的 #2/#3 需要）。

---

## 3. 交付报告的章节结构（`phase7_tracer_ablation_report.md`）

1. **Bottom line / 结论先行**：**在已评估配置与协议中**（两 suite、pi05、LIBERO、特定 score-sum/门参数），失败感知检索 cache 的 full-hit 替换未观察到「既安全又省算力」的操作点。**证据链 = Phase 5 在线出场门 FAIL（真在线 SR 下降）+ Phase 6 reduced 离线门 NO_GO（投影零增益）**——两者度量层级不同（在线 SR vs 离线 AUROC），**不并列、不暗示 Phase 6 做过在线验证**。**明确声明**这是已评估边界内的负观察，非提案 §11 proposal-literal claims 的证否。
2. **评测框架**：(SR, inf_ratio) + BHR/FFR/IR 度量空间（Eq 26–28 口径，逐指标一行操作定义，**逐字取自 `analyze_phase5_rollout.py`**：`IR=(n_miss+c_warm·n_ws)/n` 残余推理比 line 82、`BHR=fh_bad/fh_labeled` line 80、`FFR=miss_safe/safe` line 81）；**并显式写出 label proxy 及其粒度上限与因果限制**——`safe_reuse(t):=episode_success==True`、`bad(t):=episode_success==False`（line 6/78），故「FULL_HIT 为 bad 占比」= **失败 episode 内 full-hit 的比例（episode 级关联）**，**不逐步证明该 cached action 导致失败、亦不单独证明对所有精密操作本质不安全**；两 suite（libero_spatial / libero_10）；paired I_val 协议（同 250 held-out episode + seed 7）；各 Claim 的出场门定义。
3. **Claim 1（M2 双检索，success-only vs dual）= NOT EVALUATED / de-scoped；已评估 dual full-hit 操作点 FAIL 出场门**：Phase 5 Pass-3 配对 I_val rollout 表（下 §4.1 数字）；**直接证据 = 替换 rollout 的真在线 SR 下降**；离线 `L_cal` proxy ⊥ 在线 trajectory SR 作为**与 SR collapse 一致的机制解释**（非逐步反事实断言）。显式说明缺 success-only comparator → 不能交付 proposal-literal Claim 1。引 `phase5_scoresum_findings.md` + 两 suite 卡。
4. **Claim 2（M1 投影，raw vs projection）= reduced offline rescue 门 NO_GO**：Phase 6 ①b 离线门表（下 §4.2 数字），混淆-free July 数据、投影零增益。**显式标注**：仅 raw A′ vs action-only 投影 B 的 I_cal AUROC；未执行预注册的 B-vs-C（action+denoise）、无 downstream SR/IR → 非 proposal-literal Claim 2 的完整验证。引 `phase6_ib_offline_gate_report.md`。
5. **Claim 3（M3 动态深度，fixed vs dynamic）= NOT EVALUATED / owner-de-scoped**：无 fixed/oracle/dynamic depth 数据；报告写明逻辑依赖链（M3 价值以 M2 复用可 ship 为前提；M2 未过门 → ablation 三元组在 M2 一腿闭合）+ owner 裁不补 rollout（2026-07-14）。**不写成实证 ablation 的通过/失败。**
6. **warm-start 一腿（仅背景，不承重）**：owner 口述既往结论——纯 warm-start 与全量 inference 开销可比、基本不省算力；**明确标注为既往实验证据、非本期重测、无已提交 cost 产物**，并指出仓库 warm-start runbook 反将 `start_t` 定义为节省一部分 Stage 3（与「不省算力」相左）→ 本条不进任何判决矩阵、不承担 bottom line 的证据重量；指针指向 `exp/warm_start/`（见 §6 风险 R2）。
7. **整合 / ablation 矩阵**：统一表——逐机制列出 (评估状态 / 已评估操作点是否安全 / 是否省算力 / 判决定级)；结论 = 已评估边界内无一兼得「安全 + 省算力」；**同一表列出各 Claim 的 proposal-literal 缺口**（Claim 1 缺 success-only、Claim 2 缺 B-vs-C+downstream、Claim 3 未评估）。
8. **可追溯 / provenance（区分「可复现」与「可追溯」）**：**本报告是对已提交 .md 汇总报告的综合**——头行数字**可追溯**到这些汇总报告（source of truth），但**不可从 repo 独立复现**（原始 Pass-3 JSONL / 投影权重 / gate params 为 gitignored 且在 ziyang10）；每个数字标源报告行 + 承载它们的 commit `2d0e4cc`；设备拓扑。报告显式声明此「可追溯≠可复现」的边界。
9. **局限与何种情形会翻案**：`phase5_scoresum_findings.md` 的 (a) SR-aware 标定 / (b) bounded-divergence 段 / (c) warm-start-only 三条逃逸口；BHR/FFR proxy 的 episode 级粒度与因果上限（§2 已述，此处再收口）；scope 限定（2 suite / pi05 / LIBERO / 特定门参数）；未评估的 proposal-literal 比较清单。

---

## 4. 待整合的判决数字（均来自已提交证据，本 plan 已亲验一致）

### 4.1 Claim 1 = NOT EVALUATED / de-scoped；已评估 dual full-hit 操作点 FAIL 出场门（Phase 5 Pass-3 配对 rollout）

来源：`phase5_scoresum_findings.md` §3 + `phase5_scoresum_report_{spatial,l10}.md`。三跑 = cache-OFF baseline（仅供 SR_base）+ illustrative dual（`enable_dual:true`）+ calibrated dual（`enable_dual:true`）；**无 success-only comparator** → 下表证的是「这两个 dual full-hit 操作点未过门」，**不是** dual vs success-only 的 proposal-literal Claim 1。

| suite | SR_base | SR_calibrated | ΔSR | BHR（calibrated） | FFR | IR | 出场门 |
|---|---|---|---|---|---|---|---|
| libero_spatial | 0.9720 | 0.7760 | −19.6 pp | 0.3670 (1509/4112) | 0.0000 | 0.2831 | FAIL |
| libero_10 | 0.8560 | 0.5360 | −32.0 pp | 0.4893 (5530/11303) | 0.0000 | 0.3171 | FAIL |

出场门 = `BHR↓ ∧ SR_cal ≥ SR_base−ε(0.02) ∧ IR↓/=`；两 suite 的 `SR_cal ≥ SR_base−ε` 均 False → 该 dual 操作点 FAIL。l10 标定门在**每一步**都替换（0 fresh inference，IR 0.317 = 残余推理比）而 SR 崩至 0.536。**直接证据 = 替换 rollout 的真在线 SR 下降**；「~37–49% FULL_HIT 为 bad」= **失败 episode 内 full-hit 占比（episode 级关联，非逐步反事实）**；离线逐步 `L_cal` proxy ⊥ 在线 trajectory SR 作为**与 SR collapse 一致的机制解释**。

### 4.2 Claim 2 = reduced offline rescue 门 NO_GO（Phase 6 ①b；B-vs-C + downstream SR/IR 未评估）

来源：`phase6_ib_offline_gate_report.md` §3。**仅**比较 raw A′ vs action-only 投影 B 在 I_cal 上的 safe-reuse 预测 AUROC；**未执行**提案预注册的 B-vs-C（action-only vs action+denoise）与 downstream SR/IR。

| suite | I_cal ep | succ. | AUROC raw A′ | AUROC proj B | ΔAUROC (B−A′) | 95% CI | 离线门 |
|---|---|---|---|---|---|---|---|
| libero_spatial | 250 | 96.0% | 0.8156 | 0.8194 | +0.0038 | [−0.011, +0.018] | NO_GO |
| libero_10 | 250 | 82.8% | 0.7556 | 0.7600 | +0.0044 | [−0.0031, +0.0115] | NO_GO |

GO 要求 ΔAUROC CI 排除 0 且下界为正；两 suite CI 均含 0 → reduced 离线门 NO_GO。l10 类平衡更好（17.2% 失败 vs spatial 4.0%）却同判决 → 佐证稳。**此为离线 safe-reuse 预测门，非 proposal-literal Claim 2 的完整（含 denoise + 在线 SR）验证。**

### 4.3 Claim 3 = NOT EVALUATED / owner-de-scoped（M3 动态深度）

**无 fixed/oracle/dynamic depth 数据。** 逻辑依赖链：M3 仅优化检索质量/效率，其 Pareto 价值以 M2 复用可 ship 为前提；M2 未过出场门 → ablation 三元组在 M2 一腿闭合。owner 裁不补 depth-ablation rollout（2026-07-14）。**报告不将此写成实证 ablation 的通过/失败，仅记为 owner-de-scoped + 逻辑闭合说明。**

---

## 5. 测试策略

- **无 src / exp 脚本代码改动 → 无新代码 blast radius → 无需新增 test**（WA §6「新特性须有测试」不触发，因本期无特性/代码）。
- **§6 Verify（不在本 plan 预授权任何 §2.7 降级）**：交付为 doc-only 改动（**4 个 `.md`**：交付报告 + roadmap + README + 本 plan）。G1（plan 评审）无权把 WA §2.7 预先降级，Verify 的**范围决定权在 §6 执行时**。执行时：① 先 `git status --porcelain` 留证改动集仅 `.md`（无 `.py`/config/test）；② 依 WA §2.7 跑 **`uv run pytest --ignore=tests/review_tests`**——`--ignore=tests/review_tests` 是**唯一宪法级排除**，必须**显式编码进命令**以守 execution_authority §1 sealed reviewer space：本仓库 `pyproject.toml` `testpaths=["src","scripts","packages","tests"]` 递归收集全 `tests/`、无 `norecursedirs`/`collect_ignore`，`tests/review_tests/` 存 9 个 `test_*.py`，故**裸** `uv run pytest` 会读取并执行 Execution 被禁触的 reviewer probes；除此排除外**其余非-manual 测试全部纳入**，`manual`/`env_dependent` 由根 `conftest.py` 默认 skip（无 `--run-manual`）；③ **已知的环境性挂起（`tests/serving/test_replica_proxy`）不在 plan 内预排除**——该命令仍会收集 `tests/serving`，若执行时触发挂起/环境失败，**留证并当场请求 owner 明示裁决**通过范围，而非静默缩测。范围内的进一步降级由 owner 于 Verify 时定夺（详见 §6 风险 R4）。

---

## 6. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | **三 Claim 出场门口径不同**（Phase 5 = 在线 SR 门 / Phase 6 = 离线 AUROC 门 / warm-start = 成本比），整合时若混用会过度声称 | 各 Claim 保留其原生度量与门定义；跨 Claim 综合段**只做定性归纳、不造新数字**；(SR, inf_ratio) 统一框只承载 Claim 1 的真在线数；Claim 2 明确标注为**离线 safe-reuse 预测门**（AUROC），不与在线 SR 混算；warm-start 仅列背景不承重 |
| R2 | **warm-start「不省算力」在本 repo 无已提交成本/吞吐 .md 产物**（`exp/warm_start/analysis/` 仅 `success_rate_sweep.png`），是 owner 口述结论；**且仓库 warm-start runbook 反将 `start_t` 定义为节省一部分 Stage 3**（与「不省算力」相左）→ 若让它承重会超出可追溯证据 | 报告**仅将 warm-start 列为未独立复核的背景**，不进任何判决矩阵、不承担 bottom line 证据重量；**如实标注为 owner-attested 既往结论、非 Phase 7 重测、无已提交 cost 产物、且与 runbook `start_t` 定义存在表述张力**；不伪造任何成本/吞吐数字；若 owner 给出可追溯成本产物 + 计算口径 + provenance，方可升格承重 |
| R3 | **Claim 3 NOT EVALUATED 可能被读成 ablation 三元组不完整** | 报告显式写出逻辑依赖链（M3 价值以 M2 可 ship 为前提 → M2 未过门 → 三元组在 M2 一腿闭合）+ owner 裁决日期；**标为 owner-de-scoped，不写成实证通过/失败**，避免"漏做"误读 |
| R4 | §6 Verify 的范围张力：**WA §2.7（全部非-manual 通过）** vs **owner 既有裁决（§6 按改动 blast-radius，repo-wide `-m "not manual"` 是 CI 口径且撞 `tests/serving/test_replica_proxy` 挂起）**；**且裸 `uv run pytest` 在本仓库 `testpaths` 递归下会执行 `tests/review_tests/` 违 sealed reviewer space** | Verify 命令固定为 **`uv run pytest --ignore=tests/review_tests`**（唯一宪法级排除显式编码，G1 R3 Blocking 接受；§5/§7 同一命令）；除此不预授权任何降级（G1 R1 Blocking 5 接受）；该命令仍收集 `tests/serving`，环境性挂起**留证 + 请求 owner 明示裁决**通过范围；既有 owner blast-radius 裁决作为**执行时**向 owner 复核的输入，不作 plan 内预降级 |
| R5 | IR / BHR / FFR 语义标注错（Eq 26–28） | 逐指标操作定义**逐字取自 `analyze_phase5_rollout.py`**：`IR=(n_miss+c_warm·n_ws)/n`（line 82，残余推理比，越低省得越多，`c_warm=0.75`）、`BHR=fh_bad/fh_labeled`（line 80）、`FFR=miss_safe/safe`（line 81）；报告注明来源行号，不自行改写 |
| R6 | **BHR/FFR label proxy 的因果强度**：`bad(t):=episode_success==False`（analyzer line 6/78）是 **episode 级关联**，措辞不慎会被误读为逐步反事实安全事实 | 评测框架/风险/局限**三处**写明 proxy 定义、粒度上限（episode 级、非逐步）与因果限制（不证 cached action 导致失败、不证对所有精密操作本质不安全）；**根因表述限定为「与在线 SR collapse 一致的机制解释」**，把替换 rollout 的真 SR 下降作为**直接证据** |
| R7 | **「可复现」被过度声称**：原始 Pass-3 JSONL / 投影权重 / gate params 为 gitignored 且在 ziyang10，repo 内不可独立复现 | 报告严格区分**可追溯（到已提交 .md 汇总报告 + commit `2d0e4cc`）** vs **可复现**；provenance 段显式声明本报告是对已提交汇总的综合，头行数字可追溯不可从 repo 独立复现 |

---

## 7. 完成 / 出场条件

- `phase7_tracer_ablation_report.md` 落 `exp/zixuan_proposal/analysis/`，含 §3 全部 9 节、§4 全部精确定级判决数字、re-scope↔proposal-literal 缺口清单、provenance（可追溯≠可复现 + commit `2d0e4cc`）与局限。
- roadmap Phase 7 状态回填（精确定级：Claim 1 de-scoped/dual 操作点 FAIL、Claim 2 reduced 门 NO_GO、Claim 3 owner-de-scoped）+ 顶部 Status header 同步；`logs/README.md` 加本 plan 行。
- §6 Verify：改动集仅 `.md` 留证 + 依 §5 三步（`git status` → **`uv run pytest --ignore=tests/review_tests`** → 环境挂起留证并请 owner 明示裁决），**不在 plan 内预降级**。
- 无任何 src/exp 机制代码 / config / 数据改动。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-07-14 23:29 CDT

- [Blocking] [Concern] 最终报告与 roadmap 超出了 G1 批准的观察边界：报告将 Claim 1 写成绝对的 “has no operating point”，roadmap 写成“三机制无”安全省算力工作点，综合矩阵又让未经独立复核的 warm-start 承担 “preserves SR” 结论。 — reasoning: 批准计划要求结论严格限定为“在已评估 suites/configs/operating points 中未观察到”，Claim 3 明确 NOT EVALUATED，warm-start 仅可作不承重背景；请将 bottom line、roadmap 和矩阵统一改为观察限定语，并把 M3/warm-start 排除出实证综合结论。
- [Blocking] [Concern] 报告声称 l10 full-hit 点 “up to ~72% inference saved”，但同一报告采用 Eq. 26 定义 IR 为残余推理比例且列出该点 `IR=0.3171`。 — reasoning: 按报告自身定义，节省比例为 `1 - 0.3171 = 0.6829`（68.29%，约 68.3%），不是 72%；请以已提交 verdict card 与所声明公式为准更正或删除派生百分比。
- [Blocking] [Concern] provenance 将 Phase 4 报告与 Phase 5/6 产物一并归到 commit `2d0e4cc`，且设备段把 Phase 5 Pass-1 的 `--non-concurrent` 收集口径与最终 Pass-3 评估拓扑混写。 — reasoning: git 历史显示 `phase4_dual_margin_report_{spatial,l10}.md` 与 `phase4_dminus_provenance.md` 属于 commit `77ed0f5`，Phase 5/6 汇总才属于 `2d0e4cc`；最终 Phase 5 汇总只支持 server `--replicas 1`、client 12 workers、配置串行。请移除未使用的 Phase 4 引用或按真实 commit 分组，并按阶段分别陈述或收窄设备 provenance。
- [Blocking] [Concern] workflow 与 roadmap 状态仍自相矛盾：本计划头部仍为 `§4 Code 进行中`，而索引已记为 `§4 Code DONE` 且当前已进入 G2；roadmap 尾部仍把 “Phase 1” 写成下一步。 — reasoning: G2 入口意味着交付实现已完成，Phase 1–7 也已形成闭环；请同步计划状态，并删除或更新陈旧的下一步说明。
- [Blocking] [Concern] 独立 G2 文档一致性探针共 4 项、4 项失败。 — reasoning: 失败分别对应观察边界、IR 派生百分比、commit provenance、workflow/roadmap 状态；`git diff --check` 通过，但在上述断言全部通过前不满足 G2 的计划一致性、文档准确性与无回归要求。

### G2 Round 2 — Executor — 2026-07-14

- [Blocking 1 — 结论超出 G1 批准的观察边界（报告/roadmap/矩阵 + M3/warm-start 承重）] **Accepted** — ① 报告 §1 bottom line：`has no operating point` → `no operating point ... was observed`（+ "not a proof of non-existence"）；② 报告 §7 矩阵 Reading：明确「实证综合仅覆盖已评估机制（M2 已评估操作点 + M1 reduced 离线门）」，删去 warm-start "preserves SR" 承重句，M3/warm-start 显式排除出实证结论；③ roadmap §4 定性转向 bullet + 顶部 Status header：「三机制无」→「已评估的 M2 操作点 + M1 reduced 离线门均未观察到安全且省算力点；M3 未评估、warm-start 仅背景，皆不进实证综合」。
- [Blocking 2 — l10 "~72%" 与本报告 Eq 26 IR=0.3171 自相矛盾] **Accepted** — 按报告自采 Eq 26（IR 残余推理比），l10 该点省算力 = 1−0.3171 = 0.6829 ≈ 68.3%。已把 §7 矩阵 M2 行 `up to ~72 % inference saved` 改为 `IR↓ to 0.317 on l10; ≈68.3 % of the IR cost-proxy avoided = 1−IR`，去掉与公式不符的 72%。
- [Blocking 3 — provenance 把 phase4 报告误归 `2d0e4cc` + 设备段混写 Pass-1/Pass-3 口径] **Accepted** — 亲验 git：`phase4_dual_margin_report_{spatial,l10}.md`/`phase4_dminus_provenance.md` 属 **`77ed0f5`**，phase5/6 汇总属 **`2d0e4cc`**；且 phase4 报告在本报告正文**未被任何 Claim 引用**。§8 已按真实 commit 分组、并注明 phase4（`77ed0f5`）不在本报告 source set；设备段改为 **Pass-3 eval 拓扑**（server `--replicas 1`、client timan107 12 workers、illustrative/calibrated 串行），并注明 `--non-concurrent` 属 Pass-1 采集非 Pass-3（去混写）。
- [Blocking 4 — workflow/roadmap 状态自相矛盾（plan `§4 Code 进行中` vs 已 G2；roadmap 尾仍写"下一步 Phase 1"）] **Accepted** — plan 头 Status `§4 Code 进行中` → `§4 Code DONE；G2 进行中`（与 README `§4 Code DONE；待 G2` 一致）；roadmap 尾「下一步 Phase 1」改为「Phase 1–7 全部闭环 + Phase 7 G2 进行中；TRACER 线负结果收尾无待跑实验，后续方向待 owner 裁」。
- [Blocking 5 — 4 项文档一致性探针 4 失败] **Accepted** — 该项是 B1–B4 四探针的汇总（观察边界 / IR 派生百分比 / commit provenance / workflow-roadmap 状态）；B1–B4 已逐条修复，四断言应随之全过。`git diff --check` 本就通过（无空白/冲突标记），本轮改动均为 `.md` 文本、不引入回归。

### G2 Round 2 — Reviewer — NEEDS REVISION — 2026-07-14 23:48 CDT

- [Blocking] [Concern] warm-start 仍作为一行保留在 §7 `Integration / ablation matrix`，与报告 §6 的“it does not enter the §7 verdict matrix”以及 G1 批准计划的“不进任何判决矩阵”直接矛盾。 — reasoning: 将该行标成 background/not in verdict 并不能消除它实际进入 verdict matrix 的事实；warm-start 缺少已提交成本产物且只获准作为不承重背景，请从 §7 表格删除，背景与局限可继续保留在 §6。
- [Blocking] [Concern] roadmap Phase 7 的“交付物”段仍写成“在已评估配置与协议内，失败感知检索 cache 的 full-hit 替换无安全且省算力工作点”，未采用本轮其他位置已经修正的“未观察到”口径。 — reasoning: “边界外不外推”限制了适用范围，却没有把范围内的有限操作点观察改成非存在性证明；请将该句同步为“在已评估配置、协议与操作点内，未观察到安全且省算力的 full-hit 替换工作点”或等价的观察限定表达。
- [Blocking] [Concern] 修正后的独立 G2 一致性探针为 5 项中 3 项通过、2 项失败。 — reasoning: 已通过观察限定 bottom line、IR=0.3171→68.29% 计算、Phase 4/5/6 commit provenance 与 plan 状态检查；剩余两项分别精确覆盖上述 warm-start 矩阵矛盾和 roadmap 绝对化措辞。`git diff --check` 与 `git diff --cached --check` 均通过，但在两项断言转绿前不满足 G2 的计划一致性和文档准确性要求。

### G2 Round 3 — Executor — 2026-07-15

- [Blocking 1 — warm-start 仍作 §7 矩阵一行，与 §6「不进 verdict matrix」及 plan「不进任何判决矩阵」矛盾] **Accepted** — 已从 §7 `Integration / ablation matrix` **删除 warm-start 表行**（原「background/not in verdict」标注不能消除其物理进表的矛盾）；表下补一句显式说明「warm-start 刻意不列为行——un-reviewed/un-artifacted 背景，见 §6，不承任何 verdict 重量」。矩阵现仅 M1/M2/M3 三行；warm-start 背景与局限仍留 §6。
- [Blocking 2 — roadmap「交付物」句仍绝对化「无...工作点」] **Accepted** — roadmap §4 Phase 7「交付物」结论句「在已评估配置与协议内…full-hit 替换**无**安全且省算力工作点」→「在已评估配置、协议与操作点内，**未观察到**安全且省算力的 full-hit 替换工作点（观察限定，非非存在性证明，边界外不外推）」，与本轮 bottom line/顶部 header/定性转向 bullet 的观察限定口径统一。
- [Blocking 3 — 5 探针 3 过 2 失] **Accepted** — 该项是汇总；剩余两失（warm-start 矩阵矛盾、roadmap 绝对化）已由 B1/B2 修复，五断言应随之全过。`git diff --check` / `git diff --cached --check` 本就通过，本轮均 `.md` 文本改动、不引入回归。

### G2 Round 3 — Reviewer — NEEDS REVISION — 2026-07-15 00:49 CDT

- [Blocking] [Concern] `logs/README.md` 的 Phase 7 索引摘要仍将 bottom line 写成“已评估配置/协议内无安全且省算力工作点”，没有同步报告与 roadmap 已修正的“未观察到”观察限定口径。 — reasoning: README 是本计划明确要求同步的索引交付物；当前措辞仍把有限操作点的观察写成范围内的非存在性结论，造成索引与 source report/roadmap 不一致。请改为“在已评估配置、协议与操作点内，未观察到安全且省算力的 full-hit 替换工作点”或等价表述。独立 G2 探针现为 6 项中 5 项通过、仅此索引同步项失败；`git diff --check` 与 `git diff --cached --check` 均通过。

### G2 Round 4 — Executor — 2026-07-15

- [Blocking — README 索引摘要仍绝对化「无安全且省算力工作点」] **Accepted** — `logs/README.md` Phase 7 索引行 bottom line「已评估配置/协议内**无**安全且省算力工作点」→「已评估配置/协议/操作点内**未观察到**安全且省算力工作点（观察限定，非非存在性证明）」，与已修正的 report §1 / roadmap 交付物句 / 顶部 header 观察限定口径一致。全改动集(report/roadmap/README/plan 正文非 Review Log 部分)复扫无残留绝对化非存在性措辞；plan §3 第 7 节「已评估边界内无一兼得」为 scoped 观察陈述(仅指已评估的 M1/M2 两机制、非非存在性)，属 G1 已批冻结正文，不在本轮改动范围。

### G2 Round 4 — Reviewer — APPROVED — 2026-07-15 01:10 CDT

- 上轮唯一 Blocking 已闭合：README Phase 7 索引摘要现与报告和 roadmap 一致，明确限定为已评估配置、协议与操作点内“未观察到”，不再声称非存在性结论。
- 一致性复核通过：Claim 1/2/3 定级、M3/warm-start 非承重边界、IR 派生比例、Phase 4/5/6 commit 与设备 provenance、workflow/roadmap 状态均符合 G1 批准计划。
- 独立 G2 文档探针 **6 passed**；`git diff --check` 与 `git diff --cached --check` 通过；改动集仅含计划内 4 个 `.md` 交付物，无 runtime/code blast radius。批准进入 §6 Verify。
