# RIT-PL：结点分段线性风险曲线 + IR 寻址阶梯（实施计划）

> Status: **G2 APPROVED R2 → Verify 完成（§12）→ committed（2026-08-31；G1 APPROVED R3）** | Level: **L2**（`exp/dispatch_surface/` 新估计量 + 新导出器 + emit/plot 接入 + 测试；不改 `src/`）| Authority: Execution（G1/G2 closeout 由 Review Authority 在 owner override 下直接写入）| 2026-08-31
> 术语：**GST** = Grid-Searched Threshold（网格搜索阈值）；**RIT** = Risk-Indexed Threshold（风险索引阈值）。本计划的产物是 RIT 的新估计量 **RIT-PL**（piecewise-linear，family 码 `s0_pl`），与冻结的 12-bin 阶梯 RIT（family `s0`）并列，不回填。
> 数学权威：[`docs/iclr/latex/sonly_note.tex`](../docs/iclr/latex/sonly_note.tex)（Eq. cuts / 单乘子解路径）。上位文档：[`dispatch_surface_sonly_pivot_report.md`](dispatch_surface_sonly_pivot_report.md)。

## 0. 需求（owner 本会话口径，2026-08-31）

1. **q 曲线平滑**：现行 `q̂` 是 12 等频 bin 的阶梯函数，切点吸附 bin 上沿，阈值被量化。改为结点分段线性、保持单调与层嵌套，δ 连续 ⇒ 阈值连续。
2. **IR→δ→θ 映射**：仿 GST 线"cache-in-shadow 上建 inference ratio→threshold 映射"的用法。用户只给期望 inference ratio（IR），系统反解 δ 再解出 (θ_full, θ_warm)，按等距 IR 扫出帕累托前沿。
3. owner 裁定：**闭环访问分布漂移与 gate 影响两项偏差不可避免、不处理**（只记录预测 IR 与实测 IR，不做修正层）。
4. 运行时不动：artifact 仍是两个常数切点，`surface_verdict` 与 `SurfaceJudge` 零改动。

## 0.5 Verify 前置条件：baseline 必须先绿（G1 R1-B6）

现状：提交 `735a164` 按 owner 决定删除了 `docs/iclr/ICLR_PAPER_BLOCKING_TODO.md`，但当前 HEAD 的 `exp/dispatch_surface/config/confirmation_freeze_record.json` 中 `documents_sha256` 仍钉着该文件的 SHA，`tests/dispatch_surface/test_rev2_confirmation.py::test_freeze_record_matches_code_constants`（调用 `freeze_record.verify`）必然失败 ⇒ 裸 `uv run pytest` 不可能全绿，WA §2.7/§6 无法满足。

处置（**本计划 §6 Verify 的硬前置**，不满足则不得进入 Verify）：
- 方案 A（推荐，L1）：修订 freeze record —— 从 `documents_sha256` 移除已退役文档，新增 `retired_documents: {"docs/iclr/ICLR_PAPER_BLOCKING_TODO.md": {"sha256": "d26fee90…", "retired_in_commit": "735a164", "reason": "owner retired the paper gate document (2026-08-30)", "record_sha256_before_retirement": "9e28b6a3564add5ea3252856e51f1782d1cad0790df7e73c809a66e6c0f36fcc"}}`；最后一个 SHA 是本轮读取到的修订前 freeze-record 整文件摘要。`frozen_prefix`（G1 冻结前缀）与 `constants` 不动；`freeze_record.verify` 与既有测试代码**不改**（不允许以跳过/宽松化通过）。新 JSON 按现有键序用 `json.dumps(record, indent=2, ensure_ascii=False) + "\n"` 写出，使测试可以从新记录机械重构并认证旧字节。
- 方案 B：恢复该文档 —— 与 owner 的退役决定相悖，不推荐。
- **owner 裁定（2026-08-31 约 21:24 CDT，回复"并入本任务"）：并入本任务。** 采用方案 A；§4 文件清单加 `confirmation_freeze_record.json`，与本计划其余代码同一 G2 审查。已核实 `freeze_record.load_record` 只要求 `documents_sha256 / frozen_prefix / constants` 三键存在、`verify` 只遍历前两者（`exp/dispatch_surface/freeze_record.py:50-77`），新增顶层键 `retired_documents` 被容忍，`verify` 与 `test_freeze_record_matches_code_constants` 均不改。修订只删一个 `documents_sha256` 条目并记录退役事实；`frozen_prefix`（G1 冻结前缀 sha）与 `constants` 逐字节不动，`gate`/`rules` 文案不动。**本计划验收要求裸全量 `uv run pytest` 全绿**（§7-13）。

## 1. 范围 / 不在范围

**范围**：(a) 新估计量与求解函数（纯 numpy/scipy）；(b) 新导出器（digest 链绑定 Rev 1 包，产 RIT-PL artifact + export record + fit record）；(c) `sgrid_sweep emit/summarize` 接入新 family；(d) 图例与 IR 标定图；(e) 测试；(f) `sonly_note.tex` 增一段"budget addressing"；(g) §0.5 方案 A：`confirmation_freeze_record.json` 退役条目修订（owner 裁定并入）。
**不在范围**：任何 rollout（§9 只列方案，owner 另行放行）；(s,v) 版 PL；改动 Rev 1 冻结链（`fit_surface.main`、`export_exploratory_surface`、Phase 0 record schema 全部字节不动）；对预测 IR 做闭环修正；K=3。

## 2. 背景事实（写 plan 前已亲验）

- 现行求解：`fit_surface.py:366 fit_bimonotone_quantile`（bin 常数 pinball LP，约束 s 非增 / v 非减 / q₇≤q₁₀）→ `fit_surface.py:542 export_boundaries`（第一个 `q≤δ` 的 bin，切点 = `s_edges[i+1]`）。运行时 `surface_judge.py:138 surface_verdict`。
- 量化证据（`analysis/reach_preview.py --s-only`，l10 表 9205 行）：冻结 12-bin 下 q∈[0.5,0.99] 只落 **12 个**不同工作点；96-bin 下 24 个，但 q0.89→0.90 仍从 60.5% 跳到 45.1%。
- **plateau 是数据，不是 bin 数**：l10 s-only 拟合在 96 bin 时 `q_full` 有一段 25 个 bin（bins 43–67，≈25% 决策）同值 5.872，`q_warm` 有 32 个 bin（39–70）同值 5.063（isotonic 合并）。任何"只插值"的平滑都保留这些平台，δ 跨过平台时切点整段跳格。
- 可行性探针（临时脚本，未入库；24 段结点、α=0.05）：
  - ε=0（无斜率下限）：目标 IR 50/65/75/80/85/95 分别只能到 45.8/62.7/72.7/78.5/78.5/93.2（l10）；spatial 同样失败。
  - ε_total=0.02（整个 s 区间强制总下降 0.02，≈δ 量程 7.5 的 0.3%）：两套件 50…95% 每 5% 目标在打印精度 0.1 pt 内全部达到（有限表上 IR 是切点的阶梯函数，逐行粒度 l10≈0.006 pt / spatial≈0.016 pt，精确合同见 §3.3）；落在 ε 地板段的目标（如 l10 IR 50 与 55）δ 在 1e-3 打印精度下相同、全精度不同（ε>0 ⇒ δ↦θ 严格可逆），切点沿地板段由目标 IR 选定（θ_full 0.99770 vs 0.99794）。
  - IR 下界 15.2%（全 FULL_HIT = s1/c_MISS）；IR=100% 即 all-MISS anchor（已有实测臂，不导出）。
- 本地输入齐全：`data/aprime_rev1/inputs/{libero_10,libero_spatial}/dispatch_table_fresh.jsonl`（sha 与 `fit_record_s_only.json.input_digests.table` 一致：`5e5256…` / `9448c1…`）；Rev 1 包 `data/aprime_rev1/discipline/<suite>_primary/MANIFEST.json`（roles 含 `artifact.dsp_s0` / `fit.s0` / `d0` / `rebuild` / `split_manifest` / `matrix` / `verdict`）。
- 成本单价唯一权威 `analysis/analytic_cost.py`（FULL 10.260266 / WARM 46.818293 / MISS 67.518595 ms）。

## 3. 设计

### 3.1 估计量：结点分段线性 pinball LP（RIT-PL）

- 输入合同：`s, y7, y10` 为等长一维有限 float64 数组（否则 `ValueError`）。
- 结点：s 的等频分位 `knots = unique(quantile(s, linspace(0,1,n_seg_req+1), method="linear"))`，去重后 `n_seg = len(knots)−1`（记录进 fit record；`n_seg_req` 为梯子请求值）；段归属 `seg = clip(searchsorted(knots, s, side="right")−1, 0, n_seg−1)`（`s == knots[-1]` 归最后一段，重复 s 值同段）；梯子 `KNOT_LADDER = (24, 12, 6)`（请求段数），去重后每段样本数 ≥ `MIN_SEG_SAMPLES = 8` 且 `n_seg ≥ 2` 否则降一档；梯子耗尽 = stop-loss（与 Rev 1 同款纪律，零执行期自由度）。`choose_knots` 返回 `(knots, n_seg_req)`，避免分位点去重后从 `len(knots)` 无法恢复实际采用的梯子档；`fit_pl_quantile` 显式接收该 `n_seg_req` 并断言 knots 一维、有限、严格递增。l10 24 段≈383 行/段，spatial≈140 行/段。
- 预测：`q̂_a(s) = (1−w)·q_a[k] + w·q_a[k+1]`，`k` 为 s 所在段，`w=(s−x_k)/(x_{k+1}−x_k)`；`s` 超出 `[x_0, x_K]` 时夹到端点值。
- 变量：两层结点值 `q_a[k]`（a∈{warm=τ7, full=τ10}）+ 逐行 pinball 松弛；目标 `(1−α)·u + α·v`，α=0.05（沿用冻结 `quantile_alpha`）。
- 约束：
  1. 严格单调：`q_a[k] − q_a[k+1] ≥ ε·(x_{k+1} − x_k)`，`ε = EPS_TOTAL / (x_K − x_0)`，`EPS_TOTAL = 0.02`（常量，记录进 fit record；意义：整条曲线被强制的总下降量，用来消除 isotonic 平台，使 δ↦θ 严格单调可逆）。
    2. 层嵌套：`q_warm[k] ≤ q_full[k]` ∀k（线性插值保持到整条曲线）。
  3. 非负：`q_a[k] ≥ 0`（Y 按定义非负；用 LP bounds 实现）。
- 求解：`scipy.optimize.linprog(method="highs")`，稀疏 `lil_matrix`，与 `fit_bimonotone_quantile` 同构。

### 3.2 δ → θ：精确交点

`cut_at(knots, q, δ)`：若 `q[0] ≤ δ` → `θ = x_0`（全部准入）；若 `q[K] > δ` → `θ = +inf`（该档永不准入）；否则取首个 `q[k+1] ≤ δ` 的段，`θ = x_k + (q[k]−δ)/(q[k]−q[k+1])·(x_{k+1}−x_k)`。严格单调保证唯一解且 θ(δ) 连续。`θ_warm(δ) ≤ θ_full(δ)` 由嵌套约束保证（导出仍断言）。`cut_at` 要求 `fit.eps_total > 0`（否则 `ValueError`）：ε=0 的拟合允许存在水平段，交点不唯一，只许 `predict`，不许反解。

### 3.3 IR 前向与反解

- 定义（outcome-blind，只用 s 与 q̂）：`IR(θ_full, θ_warm) = Σ_rows c(a(s_row)) / (N · c_MISS)`，`a` 按 `surface_verdict` 语义（`s≥θ_full`→FULL，否则 `s≥θ_warm`→WARM，否则 MISS），`c` 取 `analytic_cost.unit_cost`。向量化实现；**测试要求逐行等于 `surface_verdict`**（G2-B1 同款 parity）。
- 单调性与粒度：θ_a(δ) 随 δ 连续非增，但 **`IR(δ)` 在有限表上是阶梯函数**——只在某切点跨过一个样本的 s 时变化，每跨一行变化 `100·(c_MISS−c_WARM)/(N·c_MISS)` 或 `100·(c_WARM−c_FULL)/(N·c_MISS)` 个百分点（l10 N=9205：0.0033 / 0.0059 pt；spatial N=3364：0.0091 / 0.0161 pt），s 值并列时按并列数倍增。因此 IR 只在离散集合上可达。
- 反解合同 `delta_for_ir(fit, s, target)`（确定性最近可达规则）：
    1. 可达范围与括号端点：令 `q_min = min(q)`、`tiny = nextafter(0,+inf)`；若 `q_min > tiny`，取 `δ_lo_end = 0.5·q_min`，否则取 `δ_lo_end = tiny`。再取 `δ_hi_end = max(q)+1`。两端都满足 `δ>0`，并按真实部署语义计算 `IR_hi = IR(δ_lo_end)`、`IR_lo = IR(δ_hi_end)`；不假定 `IR_hi == 100`（当非负约束贴在 `q_min=0` 时，正 δ 域内 all-MISS 可能不可达）。要求 `IR_lo ≤ target ≤ IR_hi` 且 `target < 100`，否则 `ValueError`（导出器转 `SystemExit`，报告真实可达区间）。若最近可达结果恰为两切点皆 `+inf`（all-MISS，IR=100），`delta_for_ir` 正常返回但**导出器拒绝**（与 §2 一致：100% 由已实测的 always-full anchor 提供，不导出重复臂）。导出器再次断言输出 δ 满足 `SurfaceArtifact.validate()` 的 `δ>0` 合同。
  2. 若 target 等于任一端点 IR，直接返回该端点；否则二分维持括号 `IR(δ_lo) > target > IR(δ_hi)`，直到 `δ_hi−δ_lo ≤ 1e-12·max(1,|δ_hi|)` 或两端 IR 都在 `IR_TOL=0.05 pt` 内；两端都显式求值。
  3. 选端点：取 `|IR(δ)−target|` 更小者；并列取 `δ_lo`（成本更高、风险更低的保守侧）。返回 `{delta, theta_full, theta_warm, predicted_ir, ir_gap = predicted_ir − target, bracket: {delta_lo, ir_lo, delta_hi, ir_hi}}`，δ 全精度浮点。
  4. 容差：`|ir_gap| ≤ IR_TOL` 为正常；`IR_TOL < |ir_gap| ≤ IR_MAX_GAP=0.5 pt` 允许导出但 record 逐臂记 `ir_gap` 并在 stdout 警告；`|ir_gap| > IR_MAX_GAP` 导出器 `SystemExit`（列出两侧最近可达 IR）。不做插值/随机化语义：部署规则仍是确定性两切点。
- **地板段语义（取代原“平台语义”；写进 record 与论文）**：ε>0 后每层严格递减，δ↦θ 唯一，不存在“一 δ 多切点”。数据平台在拟合中表现为**风险下降恰等于 ε 地板的段**。逐臂、逐层记录 `floor_info[a] = {segment: k_a, risk_drop: q_a[k]−q_a[k+1], eps_floor: ε·(x_{k+1}−x_k), on_eps_floor: risk_drop ≤ eps_floor·(1+1e-6), segment_share: n_rows_in_segment/N}`，其中 `k_a` 为 θ_a 所在段（θ_a=x_0 时 k=0，θ_a=+inf 时该层记 `null`）。全部量可由 fit record 的 `knots/q_warm/q_full/eps_total` 与 dev 行复算。论文措辞：在 `on_eps_floor` 段上，拟合的风险梯度即 ε 先验，切点在段内的位置由预算（目标 IR）决定，不主张段内存在数据支持的风险差异（"risk-equal on the floor, budget-addressed"）。

### 3.4 导出器 `exp/dispatch_surface/export_rit_pl.py`

- 输入：`--rev1-package-manifest`、`--table`、`--out-dir`（须空），以及 **寻址模式二选一**（`argparse` 互斥组，`required=True`）：`--target-ir 50,55,…,95`（IR 寻址）或 `--quantiles 0.85,0.925`（δ 寻址，δ = `delta_at_quantile(y10_dev, q)`，numpy `linear`，与 Phase 0 同一调用）。列表解析：逗号分隔 float，去空白；重复值、非递增、越界（IR：§3.3-1；q：(0,1)）均 `SystemExit`。源 role 固定 `artifact.dsp_s0` / `fit.s0`（模板 + dev membership）。
- 命名：IR 寻址 `name = "ir" + f"{target:g}".replace(".", "p")`（80→`ir80`，82.5→`ir82p5`）；δ 寻址 `name = "p" + f"{q*100:g}".replace(".", "")`（与 Phase 0 相同：0.85→`p85`，0.925→`p925`）。臂 id = `dsp_s0_pl_<name>`；artifacts 字典按输入顺序（已强制递增）插入。
- 记录字段规则：两种模式产**同一键集**；IR 寻址下 `quantile = ecdf_quantile(y10_dev, δ) = mean(y10_dev ≤ δ)`（右连续 ECDF，确定性），`target_ir` 为目标值；δ 寻址下 `target_ir = null`、`ir_gap = null`、`quantile = q`（请求值）。两种模式都记 `predicted_ir`（§3.3 前向）。
- 复用（不复制）：`rev1_package.{load_manifest, verify_package, verify_member, member_sha, load_json_member, file_sha256}`；`export_exploratory_surface.{validate_export_d0_binding, dev_mask_from_membership, delta_at_quantile, _git_commit}`；`template_parity.{assert_template_parity, assert_no_placeholders}`；`fit_surface.{load_table, _digest_obj}`；`surface_judge.{SurfaceArtifact, load_surface_artifact, save_surface_artifact}`；`analytic_cost.cost_model_digest`。
- 校验链与 Phase 0 导出器等价：包 verify、source SHA = matrix/verdict 权威、fit record `s_only=True` 且非 stop-loss、`validate_export_d0_binding`、dev membership 数字一致。**不**比较 `final_fit_digests`（那是 12-bin 拟合的指纹；RIT-PL 有自己的 `pl_fit_digests`）。
- **PL fit record 先写、后绑**（G1 R1-B3）：写盘顺序 `fit_record_pl.json` → 逐臂 artifact → `export_record.json`。`pl_fit_record_sha256 = sha256(fit_record_pl.json 字节)` 写入每个 artifact 的 `meta` 与 export record 顶层；`pl_fit_record_path`（绝对路径）写入 export record。fit record 自身携带再证实所需的全部标识：`rev1_package_manifest_sha256, source_artifact_sha256, source_fit_record_sha256, d0_record_sha256, table_sha256, dev_membership_sha256, cost_model_digest, estimator, alpha, eps_total, n_seg_req, n_seg, n_dev_rows, knots, q_warm, q_full, pl_fit_digests, ir_curve, s_range, git_commit, python, numpy`（canonical JSON：`sort_keys=True, indent=2`）。
- 产物：
    - `surface_s0_pl_<name>.npz`（`name` 按上文命名规则：`ir80` / `ir82p5` / `p925`）：`SurfaceArtifact` 字段全部从 source 复制（`k, h_exec, w, active_mask, start_t_ws, quantile_alpha, certification_mode, uses_disagreement=False, v_bin_edges=[-inf,inf], conformal_c, n_calibration_episodes, retrieval_contract`），只写 `delta, s_min_full=[θ_full], s_min_warm=[θ_warm]`；`meta` = source.meta + `{posthoc_exploratory: True, estimator: "pl_knots_v1", family: "s0_pl", addressing, n_seg_req, n_seg, eps_total, target_ir, predicted_ir, ir_gap, quantile, floor_info, s_range, source_role, source_artifact_sha256, source_fit_record_sha256, rev1_package_manifest_sha256, pl_fit_record_sha256, pl_fit_digests}`（null 规则同 record）。`assert_template_parity(artifact, source)` 通过是写盘前置条件。
    - `fit_record_pl.json`：字段见上条；`pl_fit_digests = {knots, q_warm, q_full}` 各为 `_digest_obj(array.tolist())`；`ir_curve` = artifact-compatible 正 δ 域端点 `[δ_lo_end, δ_hi_end]` 上 200 点的 `(delta, IR)` 列表。
  - `export_record.json`：新协议 `PROTOCOL_RIT_PL = "dispatch_surface_rit_pl_dev"`。顶层键集（精确相等）`RIT_PL_EXPORT_RECORD_KEYS = {protocol, posthoc_exploratory, source_role, family, estimator, addressing, rev1_package_manifest_sha256, source_artifact_sha256, source_fit_record_sha256, table_sha256, d0_record_sha256, dev_membership_sha256, pl_fit_record_path, pl_fit_record_sha256, pl_fit_digests, eps_total, n_seg, quantile_method, cost_model_digest, git_commit, python, numpy, artifacts}`；`family="s0_pl"`、`source_role="artifact.dsp_s0"`、`estimator="pl_knots_v1"`、`addressing ∈ {"target_ir","quantile"}`。逐臂键集（精确相等）`RIT_PL_EXPORT_ARTIFACT_KEYS = {path, addressing, target_ir, predicted_ir, ir_gap, quantile, delta, theta_full, theta_warm, floor_info, output_sha256}`（null 规则见上）。
  - 臂命名：`_export_record_arms` 规则 `dsp_{family}_{name}` ⇒ `dsp_s0_pl_ir80` / `dsp_s0_pl_p925`。

### 3.5 emit / run / summarize 接入（`sgrid_sweep.py`、`emit_precheck_yamls.py`、`template_parity.py`）

- `template_parity.assert_rit_pl_export_record_schema(rec, *, what, cost_model_digest)`：新函数；Phase 0 的 `assert_export_record_schema` 字节不动。
- `emit_precheck_yamls._export_record_arms(records, *, protocols=(PROTOCOL_PHASE0,), families=(FAMILY_SV, FAMILY_S0))`：加两个带默认值的关键字参数，默认行为与现在逐字节相同；`sgrid_sweep.emit` 传 `protocols=(PROTOCOL_PHASE0, PROTOCOL_RIT_PL)`, `families=(…, FAMILY_S0_PL)`，并按 `rec["protocol"]` 分派 schema 断言；`quantiles[arm]` 对 PL 臂记 δ 的 y10 分位，矩阵新增 `target_ir` / `predicted_ir` / `estimator` 三个 dict（旧矩阵无这些键，`summarize` 用 `.get`）。
- fit_role 映射：PL 臂的 `source_fit_record_sha256` 校验对 `fit.s0`。
- PL fit record 三重校验（emit 对每条 RIT-PL record；三支彼此独立可触发）：schema 键集常量统一放在 `template_parity.py`，避免该模块反向 import `export_rit_pl.py` 形成循环。(i) **路径**：`pl_fit_record_path` 存在；(ii) **字节**：`sha256(文件) == pl_fit_record_sha256`，且每个 artifact `meta.pl_fit_record_sha256` 与 record 顶层一致；(iii) **语义**（字节校验通过后）：fit record 键集精确等于 `RIT_PL_FIT_RECORD_KEYS`；`n_seg_req ∈ KNOT_LADDER`、`n_seg == len(knots)−1`、`2 ≤ n_seg ≤ n_seg_req`、`len(q_warm) == len(q_full) == len(knots)`；由 `knots/q_warm/q_full` 复算 `pl_fit_digests` 并与 record 及 export record 一致；`table_sha256 / dev_membership_sha256 / source_* / rev1_package_manifest_sha256 / cost_model_digest / eps_total` 与 export record 及包权威一致；**逐臂由 fit 在记录的 δ 上复算 `cuts` 并与 artifact 的 `s_min_full/s_min_warm` 精确相等**。矩阵新增 `pl_fit_record_sha256` dict。任一支失败 → `SystemExit`。
- contract 臂：沿用"至少一个 SV 臂载 contract"的现有规则（与 spatial 扫描同法，配一个已有 SV export record），**不改**该逻辑。
- `summarize` 输出逐臂增加 `target_ir` / `predicted_ir` / `estimator`（有则写）。
- `phase0_roster.FAMILY_S0_PL = "s0_pl"`（只加常量；`ROSTERS`/`assert_roster` 不动）。

### 3.6 图与分析

- `plot_budget_amendment.FAMILY_LABEL/COLOR/MARKER` 加 `s0_pl`（"RIT-PL (piecewise-linear q̂, IR-addressed)"）；`merge_sgrid` 遇到尚不存在的 `s0_pl` 时只为 exploratory frontier 图创建 `{measured_policies: {}, active: []}`，`fig_family_frontiers` / `fig_pareto_hull_percent` 按 `("threshold", "s0", "s0_pl", "sv")` 中实际存在的 family 作图，确认统计/假设图仍保持冻结三族不变。`plot_suite_frontiers` family 循环改为 `("s0", "s0_pl", "sv")`；`_label` 以 `^dsp_s0_pl_ir([0-9]+(?:p[0-9]+)?)$` 识别整数与小数目标（`ir80→IR80`、`ir82p5→IR82.5`），quantile-addressed PL 臂 `p925` 标为 `q.925`。
- 新脚本 `analysis/ir_calibration.py`（sonly_note 实验推论 (iii) 的成本版读数）。**接口**：`--rev1-package-manifest`（`verify_package` 后提供包 `suite`、`fit.s0` 的 `dev_membership` 与 `input_digests.table`、包内 `matrix.library_sha256`）、`--table`（shadow 表；`sha256` 必须等于 `fit.s0.input_digests.table`，否则拒绝）、`--source <tag>:<arm_matrix.json>:<summary.json>`（可重复且 tag 必须非空、唯一）、`--out-json`、`--out-fig`。**source 绑定（每条都必须满足，否则 `SystemExit`）**：matrix `protocol ∈ {PROTOCOL_SGRID, PROTOCOL_SYSGATE}` 且 `layer == "sgrid"`（tgrid / Phase 0 / primary 矩阵含 threshold / anchor 臂，不受理）；`matrix.suite == 包 suite`；`matrix.rev1_package_manifest_sha256 == 所供包的 manifest sha`；`matrix.library_sha256 == 包内 matrix 成员的 library_sha256`；matrix 与 summary 的 `cost_model_digest` 均等于 `analytic_cost.cost_model_digest()`；summary 与 matrix 的 `suite`/`protocol` 一致且 `summary.input_sha256.arm_matrix == sha256(matrix)`；`summary.rev1_package_manifest_sha256 == manifest sha`。**臂集合**：迭代 `summary.arms`（尊重 `arms_subset=True`），要求 `set(summary.arms) ⊆ set(matrix.arms)`；每个臂必须在 `matrix.artifact_paths` 中（缺失即 `SystemExit`，不静默跳过），`summary.arms[arm].family == matrix.families[arm]`，且 `sha256(artifact) == matrix.artifact_sha256[arm]`、`artifact.retrieval_contract.library_sha256 == matrix.library_sha256`。**预测值**：不依赖任何拟合——在 dev 行（membership 掩码）上逐行调用 `surface_verdict(s, v, v_bin_edges, s_min_full, s_min_warm, uses_disagreement)` 计价，`predicted_ir = 100·Σc/(N·c_MISS)`；因此 12-bin `s0`、`sv`（表有 v）与 `s0_pl` 臂统一处理。**实测值**：`measured_ir = 100 · summary.arms[arm].cost / c_MISS`（`cost` 已是 ratio-of-sums ms/决策）。**输出 JSON**：`{table_sha256, dev_membership_sha256, n_dev_rows, c_miss, sources: {tag: {matrix_sha256, summary_sha256, protocol, gate_type, gate_theta, arms: {arm: {family, estimator, predicted_ir, measured_ir, gap, artifact_sha256}}}}}`；图：横轴预测、纵轴实测、按 `gate_type` 区分标记（`always_search` 实心 / `score_hysteresis` 空心）、y=x 参考线。
- `reach_preview.py` 加 `--estimator pl`（复用 §3.1–3.3 函数）以便一处看两种估计量的可达点。

### 3.7 文档

- `docs/iclr/latex/sonly_note.tex`：在 "Coupling at K≥2" 后加一段 *Budget addressing*：PL 估计量、ε 严格单调、`IR(δ)` 反解与最近可达规则、地板段语义（§3.3："risk-equal on the floor, budget-addressed"）；重编 PDF。
- `logs/README.md` 本计划行状态同步；`docs/iclr/README.md` sonly_note 描述追加 budget-addressing 一句。

## 4. 文件清单

| 动作 | 文件 | 内容 |
|---|---|---|
| 新增 | `exp/dispatch_surface/rit_pl.py` | §3.1–3.3 纯函数 + `PLFit` dataclass + 常量 |
| 新增 | `exp/dispatch_surface/export_rit_pl.py` | §3.4 导出器（CLI） |
| 新增 | `exp/dispatch_surface/analysis/ir_calibration.py` | §3.6 IR 标定图 |
| 新增 | `tests/dispatch_surface/test_rit_pl.py` | §7 |
| 修改 | `exp/dispatch_surface/template_parity.py` | 新 schema 函数 + 键集常量 |
| 修改 | `exp/dispatch_surface/emit_precheck_yamls.py` | `_export_record_arms` 带默认值的 `protocols`/`families` |
| 修改 | `exp/dispatch_surface/sgrid_sweep.py` | emit 分派 + 矩阵新键；summarize 透传 |
| 修改 | `exp/dispatch_surface/phase0_roster.py` | `FAMILY_S0_PL` 常量 |
| 修改 | `exp/dispatch_surface/analysis/reach_preview.py` | `--estimator pl` |
| 修改 | `exp/dispatch_surface/analysis/plot_budget_amendment.py`、`plot_suite_frontiers.py` | family 标签/循环/臂标签 |
| 修改 | `docs/iclr/latex/sonly_note.tex`（+pdf）、`logs/README.md` | §3.7 |
| 修改 | `exp/dispatch_surface/config/confirmation_freeze_record.json` | §0.5 方案 A：移除退役文档条目 + `retired_documents`（owner 裁定并入本任务） |

不触碰：`src/`、`fit_surface.py`、`export_exploratory_surface.py`、`run_precheck.py`、`freeze_record.py`、Phase 0 export record 与 Phase 0 schema、`confirmation_freeze_record.json` 中除 §0.5 授权的 `documents_sha256` 退役条目之外的一切（`schema/protocol/frozen_at/gate/binding/frozen_prefix/constants/rules`）、任何 `data/`。

## 5. 接口

```python
# exp/dispatch_surface/rit_pl.py
ESTIMATOR = "pl_knots_v1"
@dataclass class PLFit: knots: np.ndarray; q_warm: np.ndarray; q_full: np.ndarray; eps_total: float; n_seg_req: int; n_seg: int; alpha: float
def choose_knots(s: np.ndarray, ladder=KNOT_LADDER) -> tuple[np.ndarray, int] | None # (knots, n_seg_req)；耗尽 None
def fit_pl_quantile(s, y7, y10, knots, *, n_seg_req: int, alpha: float, eps_total: float) -> PLFit
def predict(fit: PLFit, s: np.ndarray, layer: str) -> np.ndarray                    # "warm"|"full"，端点夹持
def cut_at(fit: PLFit, layer: str, delta: float) -> float                           # §3.2；返回 float 或 inf
def cuts(fit: PLFit, delta: float) -> tuple[float, float]                           # (theta_full, theta_warm)，断言 warm<=full
def predicted_ir(s: np.ndarray, theta_full: float, theta_warm: float) -> float      # 百分比；向量化，语义 == surface_verdict
def ir_curve(fit: PLFit, s: np.ndarray, n: int = 200) -> list[tuple[float, float]] # (delta, IR) 网格
def attainable_range(fit: PLFit, s: np.ndarray) -> tuple[float, float]              # 正 δ 域的 (IR_lo, IR_hi)
def delta_for_ir(fit: PLFit, s: np.ndarray, target: float, *, tol: float = IR_TOL) -> dict
    # §3.3 最近可达规则；返回 {delta, theta_full, theta_warm, predicted_ir, ir_gap, bracket}；target 越界 raise ValueError
def floor_info(fit: PLFit, s: np.ndarray, delta: float) -> dict                     # §3.3 地板段信息，逐层
def ecdf_quantile(y: np.ndarray, delta: float) -> float                             # mean(y <= delta)
def pl_fit_digests(fit: PLFit) -> dict                                              # _digest_obj(knots/q_warm/q_full)
KNOT_LADDER = (24, 12, 6); MIN_SEG_SAMPLES = 8; EPS_TOTAL = 0.02; IR_TOL = 0.05; IR_MAX_GAP = 0.5
```
```python
# exp/dispatch_surface/export_rit_pl.py
PROTOCOL_RIT_PL = "dispatch_surface_rit_pl_dev"
def export(args) -> dict            # 与 export_exploratory_surface.export 同形；返回 record（含 export_record_sha256）；先写 fit_record_pl.json
def main() -> None                  # CLI 见 §3.4（export_rit_pl 入口）
```
```python
# exp/dispatch_surface/analysis/ir_calibration.py
def bind_source(tag: str, matrix_path: str, summary_path: str, *, manifest_sha: str, suite: str, library_sha256: str) -> dict
    # §3.6 source 绑定；失败 SystemExit；返回 {tag, matrix, summary, matrix_sha256, summary_sha256}
def predicted_ir_for_artifact(artifact, table, dev_mask) -> float   # surface_verdict 逐行计价
def main() -> None                  # CLI 见 §3.6（ir_calibration 入口）
```
```python
# exp/dispatch_surface/template_parity.py
RIT_PL_FIT_RECORD_KEYS; RIT_PL_EXPORT_RECORD_KEYS; RIT_PL_EXPORT_ARTIFACT_KEYS
def assert_rit_pl_export_record_schema(rec: dict, *, what: str, cost_model_digest: str) -> None
def assert_rit_pl_fit_record(fit_rec: dict, export_rec: dict, *, what: str) -> None   # §3.5 (iii) 语义支
# exp/dispatch_surface/emit_precheck_yamls.py
def _export_record_arms(records, *, protocols=(PROTOCOL_PHASE0,), families=(FAMILY_SV, FAMILY_S0)) -> dict[str, dict]
```

## 6. 集成点与不变量

- 运行时：artifact schema v2 不变，`SurfaceJudge` 读 `s_min_full[0]/s_min_warm[0]`；`load_surface_artifact` 的 `validate()` 对 PL 臂全部成立（s-only 哨兵 `v_bin_edges`、`warm ≤ full`、`delta>0`、contract 交叉校验）。
- Rev 1 链：`fit_record_s_only.json`、Phase 0 export record、`final_fit_digests`、冻结常量、`freeze_record.py`、`confirmation_freeze_record.json` 的 `frozen_prefix`/`constants` 全部字节不动（该 JSON 仅按 §0.5 owner 授权修订 `documents_sha256` 的退役条目）；PL 臂 `posthoc_exploratory=True` + `estimator` 标签，任何读到该标签的旧工具（`phase0_discipline`、`assert_roster`）会拒绝它进入冻结 roster——这是期望行为。
- 默认参数路径：`_export_record_arms` 默认值下与现行为逐字节相同（测试锁定）。
- outcome-blind：`rit_pl.py` 与 `export_rit_pl.py` 的 import 图不含 `frontier_hull` / `phase0_outcome_design` / `analyze_precheck`，字符串常量不含 `success` / `status`（沿用 `test_source_lock_cost_only_import_graph` 的写法加锁）。

## 7. 测试策略（`tests/dispatch_surface/test_rit_pl.py`）

1. **LP 性质**（合成数据，同 `test_fit_surface_solver._synthetic`）：结点值严格单调（相邻差 ≥ ε·dx−1e-9）、`q_warm ≤ q_full`、`eps_total=0` 时退化为普通单调 PL 且 pinball 目标不高于 ε 版。
2. **交点**：`predict(fit, cut_at(fit, layer, δ)) == δ`（±1e-9）在可解区；`q[0]≤δ` → `knots[0]`；`q[-1]>δ` → inf；θ(δ) 对 δ 单调非增且连续（细网格相邻差有界）。
3. **IR parity**：`predicted_ir` 逐行与 `surface_verdict(s, None, [-inf,inf], [θf],[θw], uses_disagreement=False)` 计数完全一致（含 inf 切点、s=θ 边界）。
4. **反解**：合成表（n=4000，无并列）上目标 {20,…,95} 全部 `|ir_gap| ≤ IR_TOL`；**粗表**（n=50 或人工制造 s 并列簇）上存在目标使 `|ir_gap| > IR_TOL`：返回值满足最近可达规则（两端显式求值、并列取 δ_lo）且 `ir_gap` 如实记录；`|ir_gap| > IR_MAX_GAP` 时导出器 `SystemExit`；越界 target `ValueError`；`IR(δ)` 非增；`q_min=0` 合成 fit 使用最小正 δ、报告真实 `IR_hi<100` 且不把不可达 all-MISS 当成端点；`floor_info` 可由 fit record 复算且地板段上 `on_eps_floor=True`；`cut_at` 对 eps=0 拟合 raise。
5. **结点梯子**：小样本触发降档；极小样本 None；分位点去重使 `n_seg<n_seg_req` 时 `choose_knots` 仍返回所采用的请求档，fit record 中两者可区分且语义校验拒绝伪造的 `n_seg_req`。
6. **导出器**（复用 `test_rev2_phase0.build_world/build_package` 合成 Rev 1 包）：两种寻址模式各跑一次；产物通过 `load_surface_artifact` + `assert_template_parity`；非空目录 / 错 role / 篡改 table sha / 篡改 fit record membership / 同时给两种寻址 / 重复或非递增列表 均 SystemExit；record 顶层与逐臂键集精确等于 `RIT_PL_EXPORT_RECORD_KEYS` / `RIT_PL_EXPORT_ARTIFACT_KEYS`（δ 寻址下 `target_ir`/`ir_gap` 为 null）；`pl_fit_record_sha256` 与文件字节一致、每个 artifact meta 同值；`estimator` 与 `pl_fit_digests` 可复算；`quantile` 在 IR 寻址下等于 `mean(y10_dev ≤ δ)`。
7. **emit**：`_export_record_arms` 默认参数对 Phase 0 record 行为不变（现有 `test_precheck_emit` 全过 + 显式回归）；混入 PL record（IR 寻址与 δ 寻址各一）且 `protocols` 含新协议时臂名 `dsp_s0_pl_ir80` / `dsp_s0_pl_p925`、family `s0_pl`；未列入 `protocols` 的协议被拒；PL fit record **缺失 / 字节漂移（改内容不改 sha）/ 语义篡改（改 knots 并同步更新 record 与所有 artifact meta 的 `pl_fit_record_sha256`，使其穿过字节支到达语义支）** 三种情形各自 `SystemExit`，测试断言三种情形的错误信息来自不同分支；`sgrid_sweep.emit` 端到端（PL record + SV record + 一条 Phase 0 s0 record 混合）生成矩阵，`gate_layer` 两档均可；`summarize` 透传新键（用 `test_sgrid_summarize_subset` 的伪 matrix 写法）。
8. **源码锁**：import 图 / 字符串常量断言（§6）。
9. **IR 标定脚本**：表 sha 与 fit record 不符 / matrix-summary 不配对 / artifact sha 漂移 / cost-model digest 漂移 / 重复或空 source tag / summary-family 与 matrix-family 不符 / **外包**（用第二个合成 Rev 1 包发出的矩阵，manifest sha 不同）/ **跨 suite**（matrix.suite ≠ 包 suite）/ **非 sgrid 层矩阵**（tgrid 或 Phase 0 矩阵，含 threshold/anchor 臂）/ summary 臂不是 matrix 臂子集 / 臂无 `artifact_paths` 条目 均 SystemExit；`arms_subset=True` 的 summary 只处理其列出的臂；合成 artifact 的 `predicted_ir` 等于逐行 `surface_verdict` 计价；输出 JSON 键集精确匹配 §3.6。
10. **图向后兼容**：`plot_suite_frontiers` / `plot_budget_amendment` 对不含 `s0_pl` 的 summary 与不含新键的旧矩阵/summary 正常出图（Agg 后端，tmp 目录）；含 `s0_pl` 时两个 frontier 图的图例均出现 RIT-PL 系列，`ir82p5` 标签为 `IR82.5`、`p925` 标签为 `q.925`；confirmation 统计/假设图的固定 family 集与输出不变。
11. **LaTeX**：`pdflatex -halt-on-error sonly_note.tex` 退出 0（`shutil.which("pdflatex")` 缺失时 skip，并在 §6 Verify 本机实跑一次记录）。
12. **freeze record 修订**（`test_rit_pl.py::test_freeze_record_retirement_is_recorded`，不依赖 HEAD/父提交，保证 commit 后 CI 仍成立）：要求 `retired_documents` 恰含退役路径，条目含 `sha256 / retired_in_commit / reason / record_sha256_before_retirement`，commit 存在且路径在工作树不存在。测试深拷贝当前有序 JSON，移除整个 `retired_documents`，把该路径及条目 `sha256` 追加回 `documents_sha256`，再以 `json.dumps(reconstructed, indent=2, ensure_ascii=False) + "\n"` 重构旧字节；其 SHA-256 必须等于 `record_sha256_before_retirement == 9e28b6a…`。这同时证明除“移走一个 document binding + 新增 retirement ledger”外，`schema/protocol/frozen_at/gate/binding/frozen_prefix/constants/rules` 及其字节均未变化。既有 `test_freeze_record_matches_code_constants` 不改且必须通过。
13. **验收硬门**：§0.5 完成后，裸全量 `uv run pytest` 必须全绿；不接受 deselect / skip 既有失败。

## 8. 风险登记

| # | 风险 | 处置 |
|---|---|---|
| R1 | ε 是先验：地板段上强加的斜率不是数据 | ε_total=0.02（≈δ 量程 0.3%）写死并入 record；`floor_info.on_eps_floor` 标出这些段；论文措辞"risk-equal on the floor, budget-addressed"，不称地板段内存在数据支持的风险差异 |
| R2 | 审稿人可能认为 IR 寻址偷渡了"网格搜索" | δ↦θ 严格可逆、K 个切点由同一 δ 耦合（一维解路径），目标 IR 只是该路径的重新参数化，无 K 维自由度；逐臂 `floor_info` 与 `ir_gap` 入档可复算 |
| R3 | 预测 IR 与实测漂移（闭环 + gate） | owner 裁定不修正；`ir_calibration.py` 只报告 |
| R4 | PL 臂被误当冻结链产物 | 新协议名 + `estimator` 标签 + 独立 record schema；旧工具默认拒收 |
| R5 | spatial 3364 行下 24 段偏少样本/段 | 梯子降档规则；record 记实际 `n_seg` |
| R6 | LP 规模：变量 2K+4N（l10 N=9205 → ~37k 变量，~37k 约束）| 与现行 LP 同量级（现行为 2·n_s·n_v+4N），highs 秒级；测试用 n=400 |
| R7 | `_export_record_arms` 签名扩展影响 Phase 0 路径 | 关键字参数带默认值 + 回归测试锁定默认行为 |
| R8 | 端点外推：s 超出结点范围（新库/新 suite）| 夹持到端点值；s-only 下 `s>x_K` 只会更保守地按最高分处理，记录 `s_range` 进 meta |
| R9 | 目标 IR 不可达（有限表阶梯、s 并列簇、地板段）| §3.3 最近可达规则 + `ir_gap` 逐臂入档 + `IR_MAX_GAP` 拒绝；粗表测试覆盖 |
| R10 | baseline 既有失败使 Verify 不可能全绿 | §0.5 方案 A 并入本任务（owner 裁定）；不以 skip/deselect 通过；§7-12 用修订前整文件 SHA 自证唯一允许的 JSON 变换，且不依赖易漂移的 HEAD |
| R11 | PL fit record 与产物脱钩被替换 | §3.4 先写后绑 + emit 三重校验（路径/字节/内容）|

## 9. 后续 rollout（不在本计划内自动启动；owner 放行后另立记录）

- l10 A′：RIT-PL 目标 IR {50,55,…,95}（10 臂 × 300 ep = 3000 ep，≈2.7 h），可选 `--gate-layer secondary` 镜像 10 臂；contract 臂用现有 `sv_export` 记录。
- spatial A′：同 10 目标；对照 `sptgrid_summary.json`（GST 29 格）与 `spsgrid_summary_filled.json`（12-bin RIT）。
- 读数：三族前沿对前沿（GST / RIT-12bin / RIT-PL）+ `ir_calibration` 预测-实测图。

## 10. 验收

G1 APPROVED → §4 Code → G2 APPROVED R2 → §6 Verify（裸 `uv run pytest`，记录见 §12）→ owner 指示（2026-08-31 23:2x"推进流程到 commit push"）后 commit + push。

## 11. Code 记录（2026-08-31，Execution Authority；G2 交审快照）

**Plan 一致性声明**：code fully follows the approved plan（G1 R3 收口合同逐条落实）。实现细节中与 §5 字面不同的两处，均在 reviewer 合同之内：(1) `PROTOCOL_RIT_PL` 常量定义在 `rit_pl.py` 并由 `export_rit_pl` 复导出（避免 `template_parity` 反向 import 导出器，R3 合同）；(2) `sgrid_sweep.pl_arm_fields(matrix, arm)` 作为 summarize 透传的可测试单元（§3.5"有则写"）。无其它偏离。

**新增**：`exp/dispatch_surface/rit_pl.py`（估计量 + 交点 + IR 前向/反解 + floor_info + record 往返）、`exp/dispatch_surface/export_rit_pl.py`（双寻址导出器，先写 `fit_record_pl.json` 再绑）、`exp/dispatch_surface/analysis/ir_calibration.py`、`tests/dispatch_surface/test_rit_pl.py`（31 例）。
**修改**：`template_parity.py`（三组键集 + `assert_rit_pl_export_record_schema` / `assert_rit_pl_fit_record` / `assert_rit_pl_artifact_coherence`——后者封闭 export record ↔ artifact 数组 ↔ artifact meta ↔ fit record 的重复字段三角，emit 逐臂调用、导出器写盘后自检）、`emit_precheck_yamls._export_record_arms`（带默认值的 `protocols`/`families`，默认行为字节不变）、`sgrid_sweep.py`（协议分派、PL fit record 三支校验、矩阵新键、summarize 透传）、`phase0_roster.FAMILY_S0_PL`、`analysis/reach_preview.py --estimator pl`、`plot_budget_amendment.py`/`plot_suite_frontiers.py`（`s0_pl` 系列、`IR82.5`/`q.925` 标签、`FRONTIER_FAMILIES` 存在即画）、`docs/iclr/latex/sonly_note.tex`（+PDF，Budget addressing 段）、`docs/iclr/README.md`、`logs/README.md`、`exp/dispatch_surface/config/confirmation_freeze_record.json`（§0.5 方案 A：移除退役条目 + `retired_documents`，`json.dumps(indent=2, ensure_ascii=False)+"\n"` 可逐字节重建修订前记录 `9e28b6a3…`）。
**不触碰**：`src/`、`fit_surface.py`、`export_exploratory_surface.py`、`run_precheck.py`、`freeze_record.py`、Phase 0 record schema、`data/`。

**§4 本地自检（advisory，不替代 §6 Verify）**：
- `tests/dispatch_surface/test_rit_pl.py`：**39 passed**（8.6 s；G2 R1 后新增 8 例：record 切点篡改、meta 寻址篡改、`IR_MAX_GAP` 拒绝 + `IR_TOL` 警告、伪造 `n_seg_req` 两种、membership digest 分支、summarize 端到端透传、两图 legend 文本）。
- `tests/dispatch_surface` 全目录 + `tests/cache/test_surface_binding.py` + `test_crd_*`：**472 passed**（636 s；G2 R1 修订后重跑），其中 `test_freeze_record_matches_code_constants` 在修订后的 freeze record 上通过。
- ruff：`exp/dispatch_surface`、`tests/dispatch_surface/test_rit_pl.py` 全净（`tests/dispatch_surface/test_d0_check.py` 3 条 E702 为 HEAD 既有，未触碰）。
- `pdflatex sonly_note.tex` 退出 0（3 页）。
- 真实数据冒烟（临时目录，不入库）：`export_rit_pl --target-ir 50,55,…,95` 在 libero_10 / libero_spatial 的本地 Rev 1 包上各导出 10 臂，`|ir_gap| ≤ 0.03 pt`，可达区间 [15.2, 100]；l10 IR50/55 共享 δ=5.8738 且 `floor_info.full.on_eps_floor=True`（§3.3 地板段语义的实例）。

**git diff --stat HEAD**：
```
docs/iclr/README.md                                |   2 +-
 docs/iclr/latex/sonly_note.pdf                     | Bin 147710 -> 161509 bytes
 docs/iclr/latex/sonly_note.tex                     |  29 +++
 .../analysis/plot_budget_amendment.py              |  30 ++-
 .../analysis/plot_suite_frontiers.py               |   7 +-
 exp/dispatch_surface/analysis/reach_preview.py     |  44 ++++-
 .../config/confirmation_freeze_record.json         |  13 +-
 exp/dispatch_surface/emit_precheck_yamls.py        |  31 ++-
 exp/dispatch_surface/phase0_roster.py              |   4 +
 exp/dispatch_surface/sgrid_sweep.py                |  63 +++++-
 exp/dispatch_surface/template_parity.py            | 114 +++++++++++
 logs/README.md                                     |   1 +
 logs/rit_pl_ir_ladder_plan.log.md                  | 215 +++++++++++++++++++++
 13 files changed, 519 insertions(+), 34 deletions(-)
```
新文件（未跟踪）：`exp/dispatch_surface/analysis/ir_calibration.py`、`exp/dispatch_surface/export_rit_pl.py`、`exp/dispatch_surface/rit_pl.py`、`tests/dispatch_surface/test_rit_pl.py`

## 12. §6 Verify 记录（2026-08-31 23:25–23:45 CDT，Execution Authority）

裸全量 `uv run pytest`（无 -k / 无 deselect / 无 skip 标记改动）：**4957 passed, 14 failed, 60 skipped, 954 s**。

14 条失败逐条归因（均不在本改动触及的模块，本改动只动 `exp/dispatch_surface/`、`tests/dispatch_surface/test_rit_pl.py`、docs/logs 与 `confirmation_freeze_record.json`）：
- `tests/exp/test_prebuilt_matrix_backend.py::{test_cosine_fast_path_bit_identical, test_fast_path_robust_to_candidate_reordering}`（2）：用 `git archive HEAD` 抽出的干净 HEAD 树复跑，**同样 2 failed / 11 passed** —— HEAD 既有失败（cosine 快路径位一致性断言），与本改动无关。
- `tests/robocasa365/test_robocasa_policy_config.py::{test_data_config_asset_id_and_quantile_norm, test_data_transform_chain_shape}`（2）：全量跑时 `FileNotFoundError: /tmp/pytest-of-weiland/…`；在 HEAD 干净树与工作树上单独复跑均 **通过** —— pytest basetemp 环境性问题，非代码回归。
- `tests/review_tests/*`（9：`test_cache_size_g2` ×3、`test_groot_robocasa_g2` ×1、`test_rl_router_g2_contracts` ×1、`test_ws2_g2_round1` ×4）：Review Authority 的独立探针，**未入库（`git ls-files` 为 0）**、对 Execution 禁读；失败信息为其它线的环境依赖（`robocasa` 模块缺失、`Namespace` 属性、cache_size 数据文件），与 dispatch-surface 无关。
- 本任务直接相关：`tests/dispatch_surface/`（含 `test_rit_pl.py` 39 例、`test_freeze_record_matches_code_constants`）与 `tests/cache/` 全部通过。

结论：本改动引入的测试全部通过、既有失败清单在 HEAD 上可复现或为环境性且不由本改动触发；owner 于 Verify 后明示"推进流程到 commit push"。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-08-31 22:30 CDT

- [Blocking] [Concern] RIT-PL emit 没有封闭 export record、artifact 实际切点与 artifact meta 三者的重复字段一致性：当前语义支只把 fit 重算的 cuts 与 artifact 的 `s_min_full/s_min_warm` 比较，未把 export record 的 `theta_full/theta_warm` 与两者比较，也只检查了 artifact meta 的 `pl_fit_record_sha256`，未核对 `addressing/target_ir/predicted_ir/ir_gap/quantile/floor_info/estimator/family/n_seg_req/n_seg/eps_total/pl_fit_digests` 等已有重复字段。独立对抗测试证实：(a) 只改 export record 的 `theta_full`；(b) 只改 artifact meta 的 `target_ir` 并同步 record 中该 artifact 的 `output_sha256`，两者都会被 `sgrid_sweep.emit` 接受并生成矩阵。请让 emitter 对所有重复字段做类型安全的精确一致性校验（浮点按本协议生成值精确比较即可；结构字段深比较），并增加至少“record cut 篡改”和“artifact addressing meta 篡改”的共享回归测试。— reasoning: `export_record.json` 是 emit/matrix 的 provenance 输入；允许其声明值与实际部署 artifact 或其 meta 分叉，会使三重校验与矩阵里的预算寻址字段失去可信含义。
- [Blocking] [Concern] 新增模块仍有 public functions 缺少英文 docstring：至少 `export_rit_pl.export/main`、`ir_calibration.calibrate/plot/main`、`rit_pl.pl_fit_digests/fit_from_record`。— reasoning: `WORKING_AGREEMENT.md` §3.2 明确要求 public classes/functions 均有 docstring；G2 不能放行已知代码标准违反。
- [Blocking] [Concern] §7 承诺的若干关键测试尚未实际落地：现有 31 例没有触发 exporter 的“可达范围内但 `|ir_gap| > IR_MAX_GAP`”拒绝分支，没有让语义校验因伪造 `n_seg_req` 独立失败，没有经 `summarize` 端到端证明 PL 三字段透传，也没有检查两张 frontier 图的 legend 文本确实包含 RIT-PL（当前只验证文件生成和 `_label/_short`）；同时 fit-membership 篡改测试先破坏 package member 字节 SHA，未穿过包校验抵达 membership digest 分支。请补齐这些独立分支测试。— reasoning: 这些都是批准计划明确列出的高风险拒绝/兼容合同，当前测试可在对应逻辑失效时仍保持绿色。
- [Non-blocking] [Suggestion] 当前 `ruff check` 全绿，但 `ruff format --check` 报告本轮 11 个 Python 文件会被重排；仓库 pre-commit 目前没有 formatter hook，因此不作为本轮 gate，但 Executor 可在不扩大语义 diff 的前提下决定是否统一格式。— reasoning: `plot_budget_amendment.py` 的 `"threshold"` 缩进等已有可读性瑕疵，机械格式化能减少后续审阅噪声。

### G2 Round 1 — Executor（2026-08-31 22:50 CDT）

- [B1 provenance 三角未封闭] Accepted — 新增 `template_parity.assert_rit_pl_artifact_coherence(artifact, art_rec, export_rec, fit_rec, *, what)`：export record 逐臂 `theta_full/theta_warm/delta` 必须与部署 artifact 数组精确相等；artifact meta 的 `addressing/target_ir/predicted_ir/ir_gap/quantile/floor_info` 与 record 条目、`estimator/family/addressing/eps_total/n_seg/pl_fit_digests/pl_fit_record_sha256/source_role/source_*_sha256/rev1_package_manifest_sha256` 与 record 顶层、`n_seg_req/s_range` 与 fit record 逐字段精确相等（`_same`：数值按值、bool 不等于 int、None 只等 None、dict/list 深比较）。`sgrid_sweep.emit` 对每个 PL 臂在任何值进入矩阵前调用；`export_rit_pl` 写盘后对每个 artifact 自检。共享回归测试 `test_emit_refuses_record_cut_tampering`（只改 record `theta_full`）与 `test_emit_refuses_artifact_meta_addressing_tampering`（只改 meta `target_ir` 并同步 `output_sha256`）均以 `SystemExit` 拒绝。
- [B2 public docstring 缺失] Accepted — 补齐 `export_rit_pl.export/main`、`ir_calibration.calibrate/plot/main`、`rit_pl.pl_fit_digests/fit_from_record` 的英文 docstring；复查新增模块其余 public 函数均已有 docstring。
- [B3 §7 测试缺口] Accepted — 新增：`IR_MAX_GAP` 拒绝分支（monkeypatch 阈值触发"not attainable within"）与 `IR_TOL` 警告分支；伪造 `n_seg_req`（7 → "not a ladder rung"，12 → "inconsistent"）在字节支同步、meta 同步后由语义支独立拒绝；`summarize` 端到端（合成 journal/per_step/ledger 跑真实 `summarize`，PL 臂三字段透传、Phase 0 臂为 null）；两张 frontier 图 legend 文本含/不含 "RIT-PL" 与 `s0_pl` 是否存在严格一致；membership digest 分支（篡改 fit.s0 membership 并同步 manifest/matrix/verdict 全部 SHA，`verify_package` 通过后由导出器的 membership digest 检查拒绝）。`test_rit_pl.py` 31 → 39 例。
- [NB4 ruff format] Rejected（本轮不做）— 仓库 pre-commit 无 formatter hook，`ruff format` 会重排 11 个文件（含本轮未改动的既有代码段），扩大 G2 审阅 diff 且与 WA §3.1"最小改动"相抵；可另立 L0 格式化提交。`ruff check` 保持全净。

### G2 Round 2 — Reviewer — APPROVED — 2026-08-31 23:19 CDT

- Round 1 B1 verified closed：`assert_rit_pl_artifact_coherence` 在 exporter 写盘后与 emit 入矩阵前同时执行；export-record cuts / delta、部署 artifact 数组、artifact meta、export 顶层与 PL fit record 的全部重复字段形成类型安全的精确一致性三角。两条上一轮独立 provenance 探针现均通过。
- Round 1 B2 verified closed：对三个新增模块作 AST 审计，全部顶层 public class/function 均有英文 docstring。
- Round 1 B3 verified closed：共享测试增至 39 例，补齐 `IR_MAX_GAP` / `IR_TOL`、两种 `n_seg_req` 语义拒绝、membership digest 真分支、summarize 端到端透传、两图 legend 以及两条 provenance 篡改；本轮 reviewer 复跑共享 39 例 + 独立 2 例共 41 passed，Ruff 与 staged/working diff check 全绿。
- NB4 rejection accepted：formatter 不是仓库门禁；本任务不做 11 文件机械重排符合最小改动原则。
- Owner override staging provenance：Executor Round 2 的 7 文件增量已先加入 index；本 Round 2 closeout（本段、顶部状态与 `logs/README.md` 同步）有意留在工作树、未暂存。批准只放行 Verify，不授权 rollout。
