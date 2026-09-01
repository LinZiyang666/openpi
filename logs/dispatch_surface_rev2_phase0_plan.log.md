# Dispatch Surface Rev 2 — Phase 0 实施计划（exploratory 导出 / anchor 臂 / cost-map / outcome-design / 台账）

> **术语统一（2026-08-31，owner 裁定，全库生效）**：**GST = Grid-Searched Threshold（网格搜索阈值）**——本文中的 threshold / thr / T / tgrid / (fh, ws) 网格 / 速率索引均指它；**RIT = Risk-Indexed Threshold（风险索引阈值）**——本文中的 s-only / s0 / 校准分位切 / 风险阶梯 / surface 均指它（(s,v) 版 SV 为 RIT 的消融）。历史正文与 Review Log 按章程不改写，以本注释为准。

> Status: **G1 APPROVED（5 轮）→ Code → G2 APPROVED（3 轮）→ Verify 通过（2026-08-29）→ 待 commit / push**。Level **L2**（`exp/` 多文件；不改 `src/`）。
> 协议权威：[`dispatch_surface_rev2_protocol_draft.md`](dispatch_surface_rev2_protocol_draft.md) v1（§12 预 G1 裁决）；本计划 §3.6 / §3.8 的机械规则同步写回协议 §3.3–§3.4、§7 A-4。
> 结果与证据：[`dispatch_surface_rev1_aprime_result.md`](dispatch_surface_rev1_aprime_result.md) §7–§10。
> 本计划只覆盖 **Phase 0**；确认协议 analyzer（hull AUC / support-miss / fixed sequence）与 fresh-init generator 属第二计划。

---

## 1. 范围与不在范围

**在范围**（一次 G1 → Code → G2 → Verify）：

1. **exploratory 导出器**：以 Rev 1 primary artifact 为不可变模板，复算 final q-grid，导出指定 D_dev-y10 分位 δ 的 artifact + `export_record.json`（完整 provenance chain）。
2. **emitter `exploratory` 层**：suite-specific **精确 roster**（§3.3）、anchor 臂、`contract_anchor_arm`、export record 绑定。
3. **runner `exploratory` 分支**：独立 `validate_exploratory_matrix_artifacts()`；Rev 1 分支逐字不变；CLI / roster / ledger / resume frozen keys / dry validation 均有 phase0 分支与负例。
4. **phase0 discipline validator + `phase0_summary.py`**：anchor 全 MISS 与 A-4 机械判定；surface 臂只出 cost。
5. **cost-only loader + `cost_map.py`**：outcome-blind（`success` / `status` 同时删除或替换后字节等价）；§3.6 非循环机械顺序；冻结 RNG / seed / 分位方法。
6. **`phase0_outcome_design.py`**：只在 `cost_map_frozen.json` 写出并 SHA 冻结后读取 success；输出 A-2、H2 效应 / 方差 / LOTO / N=40 功效、A-6。
7. **Rev 1 discipline package 归档**（matrix / yaml / artifact / fit record / ledger + SHA）进入 data_authority；**roster spec** 文件 + digest。
8. **data_authority**：新 kind `external_asset`（HF assets）与 `task_manifest`（git-tracked，独立记录）；init-pool 记录 integrity **不动**，只在 `consumers/content` 引用。
9. 测试、logs 索引、handoff。

**不在范围**：确认协议 analyzer；fresh-init generator 与 P / C；phase audit（0d）；`src/` 变更；Rev 1 冻结判据；secondary 层；任何 rollout 的自动启动。

**硬约束**：Rev 1 的 `fit_surface.py` 输出、`run_precheck` / `analyze_precheck` 对 primary / secondary 的行为、四个已有 matrix 与两个 verdict **逐字节不变**（sha256 回归）；不得以放宽 Rev 1 校验来兼容新层；`tests/review_tests/` 不读不列。

## 2. 文件清单

| 文件 | 动作 | 内容 |
|---|---|---|
| `exp/dispatch_surface/export_exploratory_surface.py` | 新增 | §3.1 |
| `exp/dispatch_surface/fit_surface.py` | **最小抽取** | `final_fit(table, dev_mask, *, s_only, alpha, ladder) -> FinalFit(q_hat, s_edges, v_edges, sb, vb)`：把 main 中 `choose_grid` → `bin_index` → `fit_bimonotone_quantile` 的最终拟合块抽为纯函数，main 调用它；行为不变，§6-1 字节等价钉住。G1 决定：抽取（不在导出器复制）。 |
| `exp/dispatch_surface/phase0_roster.py` | 新增 | §3.3 两份 suite roster spec 的**代码常量** + spec JSON 写出与 digest 校验 |
| `exp/dispatch_surface/emit_precheck_yamls.py` | 扩展 | `LAYER_EXPLORATORY` 分支（§3.2）；Rev 1 分支不动 |
| `exp/dispatch_surface/run_precheck.py` | 扩展 | §3.4；Rev 1 分支不动 |
| `exp/dispatch_surface/analysis/analytic_cost.py` | 新增（抽取，B4） | **唯一成本 authority**：全精度 `STAGE1_MS/STAGE2_MS/STAGE3_MS`、`PINNED_START_T_WS`、warm 公式、`unit_cost(hit_type, start_t)`、`cost_model_payload()`（canonical JSON）与 `cost_model_digest()`；`analyze_precheck` 改为 import（常数值逐字节不变，§6-4 钉住） |
| `exp/dispatch_surface/analysis/precheck_io.py` | 新增（抽取） | 共享 identity / cost 解析：`parse_task_uid`、`_is_json_int`、cost 行解析、official-init 交叉、stale/fenced 判定、client_timing/infers 纪律。`analyze_precheck` 改为 import（行为不变，§6-4 钉住） |
| `exp/dispatch_surface/analysis/phase0_discipline.py` | 新增 | §3.5 exploratory discipline validator |
| `exp/dispatch_surface/analysis/phase0_summary.py` | 新增 | §3.5 |
| `exp/dispatch_surface/analysis/cost_map.py` | 新增 | §3.6（cost-only；静态 source-lock） |
| `exp/dispatch_surface/analysis/frontier_hull.py` | 新增 | upper concave hull（只被 §3.7 使用） |
| `exp/dispatch_surface/analysis/phase0_outcome_design.py` | 新增 | §3.7 |
| `exp/dispatch_surface/build_task_manifest.py` | 新增 | §3.8-b：`bddl` 包 parser（非正则）→ `exp/dispatch_surface/config/task_manifest_<suite>.json`（git-tracked） |
| `exp/dispatch_surface/archive_rev1_discipline.py` | 新增 | §3.8-c（B3）：**role → package-relative member 映射**的 MANIFEST；从 weilandserver / timan107 拉取 Rev 1 matrix、yaml、artifact、fit record、launch ledger 到 `exp/dispatch_surface/data/aprime_rev1/discipline/<suite>_primary/` 并写 MANIFEST + data_authority 记录 |
| `exp/data_authority/registry.py` | 两行 | `KNOWN_KINDS += ("external_asset", "task_manifest")` |
| `exp/data_authority/records/dispatch_surface__libero_spatial__libero_assets_hf.json` | 新增 | HF assets（result §10.5，`license: undeclared`） |
| `exp/data_authority/records/dispatch_surface__<suite>__task_manifest.json` ×2 | 新增 | kind `task_manifest`，integrity = 该 JSON 文件；content 含 BDDL 文件 SHA、libero 包版本 |
| `exp/data_authority/records/dispatch_surface__<suite>__rev1_discipline.json` ×2 | 新增 | kind `cache_artifact`（既有 kind，语义为"实验判定所需的不可变工件包"）|
| `exp/data_authority/records/dispatch_surface__<suite>__init_pools.json` ×2 | 只改 `consumers`/`content.task_manifest_dataset_id` | integrity 不动（B7-a） |
| `tests/exp/test_dispatch_rev2_phase0.py` | 新增 | §6 |
| `tests/exp/test_precheck_io_extraction.py` | 新增 | §6-4 抽取等价 |
| `tests/data_authority/test_data_authority_registry.py` | 扩展 | 新 kind |
| `logs/README.md`、`logs/session_handoff_dispatch.md` | 更新 | |

## 3. 设计

### 3.1 exploratory 导出器（B2）

```
export_exploratory_surface.py
  --rev1-package-manifest <discipline/<suite>_primary/MANIFEST.json>   # B3：以 role 解析归档成员
  --source-role artifact.dsp_sv | artifact.dsp_s0                        # 不可变模板（由 manifest 解析到成员文件）
  --table <table.jsonl>                                                  # 内容 SHA 必须等于 fit record input_digests.table
  --quantiles 0.85            # 或 0.95,0.975 / 0.80,0.95
  --out-dir <empty dir>
```

1. 绑定（B1 选 **(b)**、B3）：从 package manifest 解析 `matrix`、`fit.sv|s0`、`artifact.<role>`、`d0`、`rebuild`、`split_manifest` 成员并核每个成员 SHA；`sha256(source_artifact)` 必须等于归档 verdict `discipline.artifact_sha256[dsp_sv|dsp_s0]` **且**等于 matrix 中旧绝对路径的**声明 SHA**；`sha256(fit_record)` 等于 matrix `fit_record_sha256`；table 内容 SHA 等于 fit record `input_digests.table`。D0 校验用新增的 `validate_export_d0_binding(d0_record, fit_record, package_manifest, table_sha256, source_artifact)`——**纯 digest-chain 校验，不 live-reopen 任何历史路径**（R3-B1；`d0_check.validate_input_attestation` 会 `resolve()` 并重算 attestation 里绝对路径的文件 / H5 tree SHA，与迁移回归互斥，因此**不调用**）。D0 `inputs.files` 的 exact key set 必须为 `{table, library_pkl, noise_sidecar, cache_yaml, weights_npz}`；fixture 必须来自真实 D0 record 的最小复制，不得增造字段。链条：D0 record 的真实性由 `sha256(d0_record) == fit_record.input_digests.d0_record` → `sha256(fit_record) == matrix.fit_record_sha256` → 归档 verdict `discipline.fit_record_sha256` 认证；在此之上对声明 digest 做交叉绑定：(i) D0 `files.table` 对当前 `table_sha256`，`files.weights_npz/cache_yaml` 对 fit-record 同名 input digest；(ii) D0 `files.library_pkl/noise_sidecar` 对 archive rebuild record 的 `library_sha256/noise_sidecar_sha256`；(iii) package `rebuild` / `split_manifest` member SHA 对 `fit_record.input_digests.rebuild_record/split_manifest`；(iv) rebuild 内 `split_manifest_sha256` 对 package split member；(v) source artifact contract 的 `library_sha256/library_entry_count/policy_fingerprint` 对 rebuild/D0 policy 声明；(vi) D0 schema、`protocol == D0_PROTOCOL`、`PASS`、census、三项 check，以及包含 `query_h5/library_h5/policy` 的 attestation **rollup canonical digest**均有效（只重算 dict，不打开历史路径）。任一不符 ⇒ SystemExit。**不**调用 `fit_surface._validate_d0_record`。
2. dev mask 由 fit record 的**精确 `dev_membership` episode 列表**重建（不是 fit|cal 标记）；`_digest_obj(membership)` 必须等于 `dev_membership_sha256`。
3. `final_fit(table, dev_mask, s_only=…, alpha=quantile_alpha, ladder=Rev1 ladder)` 复算 q-grid；`_digest_obj(q_hat)` / edges digest 必须等于 `final_fit_digests`（`q_deploy` 等），否则 SystemExit。
4. δ_q = `np.percentile(table.y10[dev_mask], 100*q, method="linear")`——与 Rev 1 §4.2 grid（`fit_surface.py:855`）同一调用；p80 / p90 必须与 fit record `delta_grid` 逐值相等（1e-9）。
5. artifact：`w / active_mask / retrieval_contract / quantile_alpha / certification_mode / k / h_exec / start_t_ws / uses_disagreement / v_bin_edges` **逐字段取自 source artifact**；只重算 `s_min_full / s_min_warm`（`export_boundaries(q_hat, s_edges, δ)`）与 `delta`；meta = source meta ∪ `{posthoc_exploratory: true, delta_quantile: q, delta_name: "p85", quantile_method: "linear", source_role, source_artifact_sha256, source_fit_record_sha256, rev1_package_manifest_sha256}`。**meta 中不得出现任何 `pending` / 占位 provenance 值**（B1；哈希方向只有 `export_record → artifact`，export record 的 SHA 由 matrix / ledger 在外部绑定，测试钉死）。
6. `export_record.json`：逐 artifact 记 `source_role`、`rev1_package_manifest_sha256`、`source_artifact_sha256`、`source_fit_record_sha256`、`table_sha256`、`d0_record_sha256`、`dev_membership_sha256`、`final_fit_digests`、`quantile`、`quantile_method`、`delta`、`output_sha256`、`cost_model_digest`（§3.5）、libero/openpi 版本与 git commit；**不记录**只能在原机器解释的 source 绝对路径；文件本身 SHA 写入 matrix（§3.2）。
7. out-dir 非空或含任何 Rev 1 artifact ⇒ SystemExit；不写 fit record。

### 3.2 emitter `exploratory` 层

`--layer exploratory --suite <s> --export-records <sv.json,s0.json> --rev1-package-manifest <MANIFEST.json>`（B3：matrix / fit record / source artifact 全部经 role 从归档包解析；不接收裸 `--rev1-matrix` 路径）

- roster 由 `phase0_roster.py` 常量决定（§3.3），命令行**不能**增删臂；emitter 从 export record 读 δ / quantile / family，与 roster spec 逐项比对（缺失 / 多余 / 重复 quantile / family 与 `uses_disagreement` 不符 ⇒ 拒绝）。
- anchor 臂：`judge: {type: threshold, threshold: 1.5}`，**无** `warm_tiers`；gate `always_search`（`gate_section(LAYER_PRIMARY, theta)`，theta 取自 Rev 1 matrix `gate_theta`，不重算）。
- 三端重放链（emitter 先做一遍）：`artifact.sha256 == export_record.output_sha256` ∧ `export_record.rev1_package_manifest_sha256 == sha256(MANIFEST)` ∧ `export_record.source_artifact_sha256 == manifest.members[source_role].sha256 == rev1_matrix.artifact_sha256[dsp_sv|dsp_s0]`（matrix 原字节不改，只核其**声明 SHA**）∧ `export_record.source_fit_record_sha256 == manifest.members[fit.sv|s0].sha256 == rev1_matrix.fit_record_sha256[sv|s0]` ∧ 从归档成员加载 source artifact，非 δ / boundaries 字段与之逐字段相等 ∧ `cost_model_digest` 一致（§3.5）。
- matrix：`protocol="dispatch_surface_rev2_phase0"`、`layer="exploratory"`、`posthoc_exploratory=true`、`suite`、`roster_spec_sha256`、`rev1_package_manifest_path/sha256`、`export_record_paths/sha256`、`cost_model`（payload + digest，§3.5）、`artifact_paths/sha256`、`judge_role={"always_full_inference":"always_full_inference_anchor"}`、**`contract_anchor_arm`**（本 suite 的 SV exploratory 臂：l10 `dsp_sv_p85`，spatial `dsp_sv_p95`）、`core_arms=[]`、`descriptive_arms=<all>`、`gate_theta`、`library_pkl/sha256`、`template`。

### 3.3 精确 roster（B6，rollout 前冻结为代码常量 + spec JSON + digest）

| suite | 臂 | family | quantile |
|---|---|---|---|
| libero_10 | `always_full_inference` / `dsp_sv_p85` / `dsp_s0_p80` / `dsp_s0_p95` | anchor / SV / S0 / S0 | — / 0.85 / 0.80 / 0.95 |
| libero_spatial | `always_full_inference` / `dsp_sv_p95` / `dsp_sv_p975` / `dsp_s0_p80` / `dsp_s0_p95` | anchor / SV / SV / S0 / S0 | — / 0.95 / 0.975 / 0.80 / 0.95 |

cost-map 的候选集（§3.6）= 上表 ∪ Rev 1 primary 的 `dsp_sv_minus`(p80)、`dsp_sv`(p90)、`dsp_s0`(p90)、三个 threshold 臂；每个 source 的 family / δ **只**来自受认证的 artifact meta / export record / Rev 1 discipline package，不来自命令行。

### 3.4 runner（B1）

- CLI `--layer` choices 加 `exploratory`；`LAYER_EXPECTED_GATE["exploratory"]="always_search"`。
- 分支：`if layer == LAYER_EXPLORATORY: validate_exploratory_matrix_artifacts(matrix)` else 原 `validate_matrix_artifacts(matrix)`（**逐字不动**）。roster 检查：`core_arms == []`、`descriptive_arms == all`、`protocol` 为 phase0 值、`posthoc_exploratory is True`、roster 与 `phase0_roster` 常量完全一致。
- `validate_exploratory_matrix_artifacts()`：从 matrix 的 `rev1_package_manifest_path` 加载归档包并核 SHA，对每个 surface 臂经 role 重放 §3.2 的三端链（**不**依赖任何 `/tmp` 原路径）；对 `expected_k / uses_disagreement` 按 **family**（来自 export record）而非臂名判断；对 anchor 臂：`judge.type == threshold` ∧ `threshold > 1.0` ∧ `warm_tiers` 为空 ∧ gate `always_search`。
- launch contract：**只**从 `matrix["contract_anchor_arm"]` 取 artifact（必须是本 suite 的 SV exploratory 臂且 `uses_disagreement=True`）；不猜臂名。
- launch ledger：`posthoc_exploratory=true`、`roster_spec_sha256`、`rev1_package_manifest_sha256`、`export_record_sha256`、`cost_model_digest`、`contract_anchor_arm` 进入 frozen keys；resume 时逐项比对；subset launch（`--arms`）必须继承同一 frozen matrix / roster ledger。
- `--dry-validate` 覆盖以上全部；负例见 §6。

### 3.5 phase0 discipline validator 与 summary（B4 / B8）

`phase0_discipline.validate(matrix, ledger, split_manifest, *, trials)`：exploratory 版本的 `check_discipline`，**与 Rev 1 同强度**（G2R1-B1）——唯一 run id；逐 entry 校验 `executed_arms ↔ executed_yaml_sha256`、`trials_per_task / replan_steps / env_seed`、`contract_binding`（policy fingerprint、h_exec）、A′ pool attestation（suite / total_inits / rollup / split / state digests）；YAML 字节**重算**；anchor 臂经 `validate_precheck_arms` 重验（threshold > 1、无 warm tier、`always_search`）；返回 `executed_arms_by_run` 与 `roster_complete`。**认领检查** `assert_rows_claimed(accepted, executed_arms_by_run)`：任一 accepted `(run_id, arm)` 未被相应 launch 执行 ⇒ 拒绝——summary、cost-map、outcome-design 在两个 loader 之后统一调用（Rev 1 源亦以其归档 ledger 认领）。其余一致性：suite、A′ content rollup、split manifest SHA、policy fingerprint、library SHA、roster、export record 链、launch ledger；**cost model 交叉校验**（B4）：`analytic_cost.cost_model_payload()` 的 digest 必须等于 matrix / ledger / export record 记录的 `cost_model_digest`，且其三段常数与三档单价逐值（全精度）等于归档 verdict `discipline.cost_inputs.unit_cost_ms`；**不调用** `analyze_precheck.check_discipline`。
`phase0_summary.py` 输出 `phase0_summary.json`（`posthoc_exploratory=true`；`--executed-only` 的部分视图带 `partial_nonadjudicative=true`，正式 summary 要求 `roster_complete`）：
- **A-4 机械判定（anchor 臂）**：`300/300 accepted` ∧ verdict 计数 `{MISS: n, FULL_HIT: 0, WARM_START: 0}` ∧ 每 cell decision 与 `client_timing.infers` 完整一致 ∧ `math.isclose(ratio_of_sums, analytic_cost.unit_cost("MISS", None), rel_tol=0, abs_tol=1e-9)`（期望值 **不写字面量**——全精度 MISS 成本为 67.518595 ms，v2 正文的 `67.5186 ± 1e-6` 会让合法 anchor 必败，B2）；summary 同时记录 `cost_model` payload（全精度三段常数、warm 公式、`PINNED_START_T_WS`）与 `cost_model_digest`；任一不满足 ⇒ `SystemExit`，整臂拒绝。anchor 的 SR **只记录**，不参与任何判定；若 SR 暴露 harness 异常，只能形成有日志的 protocol amendment 并重审，不得静默重跑或换资产。
- surface 臂：realized cost（ratio-of-sums）、tier-mix、每 cell `(cost_sum, n)`；**不输出 SR**。

### 3.6 cost-only loader 与 `cost_map.py`（B3 / B4 / B8）

**loader**：`precheck_io.load_accepted_cells_costonly(journal, grid)` 只依赖 `accepted / task_uid / attempt / run_id / phase`、唯一完整网格、以及 per-step / client_timing 对被采纳 terminal attempt 的完整性确认；**同时忽略 `success` 与 `status`**（R3-B2：生产 schema 中 `done ⇔ success=True`，`status` 是 outcome 的等价编码）；`precheck_io.load_cost_cells(per_step, accepted, officials)` 保留 stale/fenced、official-init、client_timing/infers、decision-count 纪律，去掉 success 比对。原 `analyze_precheck` 的两个 loader 继续校验 status / success，行为不变（§6-4）。AST source-lock 额外禁止 cost-only 路径（`precheck_io` 的 costonly 函数与 `cost_map.py`）出现字符串键 `"success"` / `"status"` 的读取。

**机械顺序（非循环）**：
1. 候选集 = §3.3 roster ∪ Rev 1 primary 点；每个 source 经 §3.5 / Rev 1 discipline package（role 映射）认证；成本单价只从 `analytic_cost` 取（B4）；所有臂共享**同一** task-stratified、init-cluster 的 paired bootstrap 索引（`numpy.random.Generator(PCG64(20260829))`，10000 reps，索引一次生成并写入输出）。
2. **SV / S0 族**（具有数值 δ）：候选严格按**真实 δ** 排序（与分位序一致，否则拒绝），用 point-estimate ratio-of-sums cost 拟合 **decreasing isotonic**（权重 = decision 数；PAV）。**threshold 族不做 isotonic、不伪造 δ**（B6）：按预注册 aggressiveness 顺序固定 endpoints = `fh70`（低成本）/ `fh30`（高成本）、middle = `fh50`。
3. **endpoints**（SV / S0）= 每族 isotonic cost 最低与最高的候选；**并列最低取最大 δ、并列最高取最小 δ**（`family_endpoints`，G2R1-B5），即外侧点（l10 SV: p90 / p80；spatial SV: p975 / p80；S0: p95 / p80）。
4. 用固定 endpoints 与共享索引得到每 rep 的 `L_r = max_family min_endpoint_cost(r)`、`H_r = min_family max_endpoint_cost(r)`；`iL = ceil(0.995·(R−1))`、`iH = floor(0.005·(R−1))` 由 `(q, n, method)` 定义算出零基下标，`qL = sort(L)[iL]`、`qH = sort(H)[iH]`，并断言等于 `np.quantile(·, method="higher"/"lower")`；R = 10000 时强制 `(iL, iH) == (9950, 49)`（ties 时不用值搜索，G2R1-B5）；`c_L = ceil_0.1(qL)`、`c_H = floor_0.1(qH)`；`c_1, c_2` 按协议公式取整。
5. **middle**：l10 SV 唯一候选 p85；S0 唯一候选 p90；T 固定 fh50（非选择）；spatial SV 从 {p90, p95} 取 isotonic cost 最接近 `(c_L+c_H)/2` 者，平局取较小 δ（更保守）。middle 选定后**不再改动**区间。
6. 对所选三点重算 raw point-estimate cost：三点严格不同 ∧ endpoints 包围 `[c_L, c_H]` ∧ `c_H − c_L ≥ 4.0` 否则 **A-3 fail closed**（输出 `a3_pass=false`，仍写出全部中间量）。
7. 写 `cost_map_frozen.json`：**`input_sha256`**（G2R1-B2：Rev 1 的 manifest / matrix / ledger / split / journal / per_step 成员 SHA，Phase 0 的 matrix / ledger / split / journal / per_step / export records SHA）、roster spec SHA、`cost_model` payload + digest、seed = 20260829、R = 10000、索引 SHA、每候选的**真实 δ**（Rev 1 从归档 artifact 读，Phase 0 从受认证 artifact 读；G2R1-B5）、isotonic 表、endpoints、`qL/qH` 与由 `(q, n, method)` 公式算出的零基下标（R = 10000 强制 9950 / 49）、`[c_L,c_H], c_1, c_2`、middle、A-3 结果；写出后由调用方计算 SHA 并登记。
**静态 source-lock**：`cost_map.py` 与 `precheck_io.py` 的 import graph 不含 `frontier_hull`、`phase0_outcome_design`、`analyze_precheck`（成本常数经 `analytic_cost` 共享，不复制、不 import analyzer）；测试用 AST 检查 + 运行期 `sys.modules` 断言。

### 3.7 `phase0_outcome_design.py`（B5）

输入：`cost_map_frozen.json` + 其 SHA（命令行必须给出，且与文件一致）、同一 paired 索引、journal / per_step。**读 outcome 之前**：`validate_cost_map_header`（`seed == 20260829`、`replicates == 10000`（整数、非布尔）、`posthoc_exploratory / outcome_blind / a3_pass` 为真、必填键齐全，G2R2-B1）；然后（G2R1-B2）逐项重算 cost map `input_sha256` 冻结的全部输入（Rev 1 manifest 等于命令行 `--rev1-package-manifest` 的 SHA 且等于 Phase 0 context 绑定的包；Rev 1 matrix / ledger / split / journal / per_step 成员；Phase 0 matrix / ledger / split / journal / per_step / export records），任一不符 ⇒ 拒绝；seed / R 不是冻结值 ⇒ 拒绝；重新加载的每臂 point cost / decision 数必须与冻结 map 一致（`abs_tol 1e-9`）；accepted 行经 `assert_rows_claimed` 认领。输出 `phase0_outcome_design.json`：
- 用冻结的 `[c_L,c_H]` 与三臂 / 族构造协议 §3.2 hull frontier（`frontier_hull.py`：支配 = 协议 §3.2 的 Pareto 规则——另一点成本 ≤ 且 SR ≥ 且至少一项严格；**同 SR 更贵的点被删除，不延伸支撑**；G2R2-B3 恢复冻结语义），AUC 为 piecewise-linear hull 在区间上的**精确梯形积分**（断点 = 区间端点 ∪ 区间内 hull 顶点；G2R1-B3），并调用**与协议 §4 完全相同的** `auc_with_support(hull_a, hull_b, c_L, c_H)` 纯函数（B5）：某 replicate 中任一族 hull 不包围 `[c_L, c_H]` ⇒ 该 replicate `AUC = −1.0` 并**保留**；joint support-miss 率 > 1% ⇒ 输出 `support_miss`，A-2 / A-6 **fail closed**；
- **A-2**：spatial 三族 hull 与 support；SV hull 是否在共同区间被 T 支配（descriptive）；
- **H2 设计量**（R3-B3 唯一口径）：`effect` = 全部 **300 个 development cells** 上、用冻结三点与 `[c_L,c_H]` 得到的 **full-sample plug-in** `AUC(SV) − AUC(S0)`（其三族 hull 必须覆盖区间，否则 A-6 fail；**不是** bootstrap mean，也不含任何 −1）；`sd30` = 10000 个 paired bootstrap AUC difference 的 **sample SD**，其中 support-miss replicate 按协议赋 −1 并保留；joint miss > 1% ⇒ fail。另输出 task-stratified 方差、LOTO、每 task descriptive。N = 40 下的单侧 α = 0.05 功效冻结为 `power = Phi( effect / (sd30 · sqrt(30/40)) − z_0.95 )`；`effect ≤ 0`、`sd30` 非有限或为 0、support gate 失败 ⇒ **A-6 fail**；否则 **A-6** = `power ≥ 0.80`；
- H1 同样重算 development 效应与功效（协议 §9 的更新值）；每个 hypothesis 输出 `support_ok / power_n40 / verdict`，**不带** `a6_pass`。
- **Decision gates 单独输出**（G2R1-B6）：`decision_gates.A2`（仅 spatial 适用的描述量：SV hull 相对 threshold hull 在区间上 SR 差的**精确极值**——在 `{c_L, c_H} ∪ 两 hull 区间内顶点` 上求，记录 `checked_breakpoints / argmin_cost / argmax_cost`，G2R2-B2；是否被支配）、`A5`（H2 support：plug-in hull 覆盖 ∧ joint miss ≤ 1%）、`A6`（仅 H2：`power_n40 ≥ 0.80`，含 reason）。
- **per-task descriptive**（`per_task_descriptive[task]`）：三族 points / hull / support、H1/H2 在该 task 共同支撑上的描述性 AUC、middle 臂配对 discordance（win / lose）、ceiling / floor 标记；附 note 说明 LOTO ≠ 每 task 为正。
不得回写 cost-map、改臂、改区间（文件只读；测试对输入 SHA 失配 ⇒ SystemExit）。

### 3.8 台账与归档（B6 / B7）

- **a. task manifest**（B7 选 a）：`build_task_manifest.py` 用 LIBERO 环境内的 `bddl` 包（`bddl-1.0.1`）解析 goal AST，输出 `exp/dispatch_surface/config/task_manifest_<suite>.json`（git-tracked）：`task_id`（= LIBERO `benchmark.get_task(i)` 执行序）、`task_name`、`bddl_path`、`bddl_sha256`、`goal_atoms`（谓词 + 参数）、`n_goal_atoms`；与 split manifest `assignment[task_id].task_name` **双向**校验。新 kind `task_manifest` 记录；init-pool 记录 integrity **不动**，只加 `consumers` 与 `content.task_manifest_dataset_id`。
- **b. HF assets**：kind `external_asset`；`integrity.sha256` = 完整 rollup；`license: undeclared`；`provenance.produced_by = huggingface:jadechoghari/libero-assets@90001343…`。
- **c. Rev 1 discipline package**（B3）：`archive_rev1_discipline.py` 拉取每 suite primary 的 `arm_matrix_primary.json`、6 个 yaml、3 个 artifact、2 个 fit record、D0 record、rebuild record、split manifest、launch ledger、verdict 到 `data/aprime_rev1/discipline/<suite>_primary/`，**原字节不改**（matrix 内的 `/tmp/...` 绝对路径保留，其 SHA 仍等于 verdict authority）。MANIFEST 冻结**逻辑 role → package-relative member** 映射：`matrix`、`fit.sv`、`fit.s0`、`artifact.dsp_sv`、`artifact.dsp_s0`、`artifact.dsp_sv_minus`、`yaml.<arm>`×6、`d0`、`rebuild`、`split_manifest`、`ledger`、`verdict`，每项 `{member, sha256, declared_path_in_matrix?}`；archive validator 要求每个 role 的成员 SHA 等于 matrix 中对应旧路径的**声明 SHA** 与 verdict `discipline.*`，并要求 `analytic_cost.cost_model_payload()` 逐值等于 verdict `discipline.cost_inputs.unit_cost_ms`。exporter / emitter / runner / discipline / cost-map 全部经 role 解析成员；**迁移回归**：删除（或 chmod 000）原 `/tmp/dsp_shared` 与 `/tmp/dsp_precheck` 树后全链仍通过（§6-14）。登记 data_authority（kind `cache_artifact`）。

## 4. 接口与集成点

- 导出器依赖 `fit_surface.final_fit`（抽取）、`export_boundaries`、`surface_judge.save/load_surface_artifact`；不依赖 emitter。
- emitter ↔ runner ↔ discipline ↔ cost_map 的契约 = arm matrix + export record + roster spec 三个 JSON（均带 SHA，写入 ledger）。
- `analyze_precheck` 不感知 Phase 0：exploratory matrix 被现有 layer 检查拒绝（测试钉住）；其 loader 改为 import `precheck_io`（等价测试）。
- `cost_map_frozen.json` 与 `phase0_outcome_design.json` 是第二计划的唯一数值输入；schema 在本计划冻结。

## 5. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| 抽取 `final_fit` / `precheck_io` 改变 Rev 1 行为 | 冻结产物失效 | §6-1 / §6-4：真实冻结输入重跑，artifact NPZ SHA + 逐字段语义比较；两个 verdict JSON 字节等价 |
| 导出器 δ 定义与 grid 不一致 | 点位不可比 | p80 / p90 逐值回归到 `delta_grid`（1e-9）；`quantile_method` 写入 record |
| provenance 链任一环缺失 | runner 无法证明来源 | 三端重放（emitter / runner / discipline）+ 负例：篡改 source SHA、换 artifact、改 δ |
| anchor 语义靠隐含上界 | 非恒 MISS | dry validation 检查 + A-4 机械判定 100% MISS |
| cost-map 读到 outcome | 违反 outcome-blind | 同时删除/替换 `success` 与 `status` 后字节等价测试 + 静态 source-lock |
| 循环选点 | 区间可被操纵 | §3.6 顺序 + 合成四候选反例测试 |
| exploratory 混入 Rev 1 | 污染确认 | analyzer 拒绝；目录 / matrix / ledger 三处标记 |
| Rev 1 source 只在 `/tmp` | 不可复现 | §3.8-c 归档为 authority 后才允许 cost-map 运行 |
| 远端克隆未同步 | 白跑 | 推送后核 sha256 再 launch |
| LIBERO env / assets 冷启动 | 启动崩 / 重试 | `LIBERO_CONFIG_PATH` 已加；重试由 conductor 吸收，验证 attempt 分布 |

## 6. 测试策略

`tests/exp/test_dispatch_rev2_phase0.py`、`tests/exp/test_precheck_io_extraction.py`、`tests/data_authority/…`：

1. **fitter 字节等价**：合成表跑 `fit_surface` 抽取前后 artifact SHA 相同；Verify 阶段用真实冻结输入重跑，`surface_sv_primary.npz` / `surface_s_only_primary.npz` 的 SHA 等于归档 verdict `discipline.artifact_sha256`，并逐字段语义比较。
2. 导出器：p80 / p90 与 `delta_grid` 一致；`final_fit_digests` 失配 ⇒ exit；source artifact SHA 失配 ⇒ exit；非 δ 字段与 source 逐字段相等；out-dir 非空 ⇒ exit；export record 字段齐全。
3. emitter：roster 精确（缺 / 多 / 重复 quantile / family 错 ⇒ 拒绝）；anchor yaml；三端链负例（改 source SHA / 换 artifact / 改 export record）；`contract_anchor_arm` 必须是 SV exploratory。
4. **`precheck_io` 抽取等价**：Rev 1 两套真实 journal / per_step 经新旧 loader 得到相同 cells 与 cost；`analyze_precheck` 重跑两个 verdict 字节等价。
5. runner：exploratory dry-validate 通过；anchor threshold ≤ 1 / 有 tier ⇒ 拒绝；primary matrix 带 `posthoc_exploratory` ⇒ 拒绝；`--layer` 与 matrix 不符 ⇒ 拒绝；resume frozen keys 含新字段；subset launch 继承 ledger。
6. analyzer：exploratory matrix ⇒ `SystemExit("arm matrix has no valid Rev 1 analysis layer")`。
7. phase0_discipline / summary：anchor 一个 FULL_HIT ⇒ 拒绝；cost ≠ 67.5186 ⇒ 拒绝；cell 不完整 ⇒ 拒绝；surface 臂输出无 `sr`。
8. cost_map：journal/per-step 的 `success` 与 `status` 字段**同时删除 / 任意替换**后字节等价；合成四候选（spatial SV）反例：不同预选会改变区间——按 §3.6 顺序结果唯一；A-3 三个失败条件；isotonic 单调与平局规则；order-statistic 分位；seed / 索引 SHA 复现。
9. 静态 source-lock：AST 检查 `cost_map.py` / `precheck_io.py` 不 import hull / outcome / analyze 模块。
10. outcome_design：输入 SHA 失配 ⇒ exit；不写入 cost-map 文件（mtime / SHA 不变）；A-6 点效应 ≤ 0 ⇒ fail；功效计算对已知合成分布正确。
11. frontier_hull：中点在 hull 上 / 被 chord 支配 / 同 cost 不同 SR 三类回归。
12. task manifest：`bddl` 解析结果与 split manifest 双向校验；task 5 = book、task 9 = microwave（回归 v0 的错配）。
13. data_authority：新 kind 通过 `validate`；init-pool 记录 integrity 未变（SHA 回归）；旧记录不受影响。
14. **迁移回归**（B3）：在临时目录重建归档包、将 `/tmp` 原路径置为不存在（monkeypatch），exporter / emitter / runner dry-validate / discipline / cost_map 全链通过；matrix 声明 SHA 与 role 成员 SHA 不一致 ⇒ 拒绝。
15. **D0 绑定**（B1）：`validate_export_d0_binding` 对篡改 D0 attestation、fit record `input_digests` 不符、rebuild / split / contract 任一项不符 ⇒ exit；正确输入通过；artifact meta 含任何 `pending` 字面值 ⇒ 导出器拒绝写出。
16. **成本 authority**（B2 / B4）：`analyze_precheck` import `analytic_cost` 后两个 Rev 1 verdict 字节等价；`unit_cost("MISS", None) == 67.518595`（全精度比较）；A-4 对 ratio-of-sums = 该值通过、对 `67.5186` 失败（回归 v2 的截断错误）；matrix / ledger / summary / cost-map 的 `cost_model_digest` 一致，且等于归档 verdict `discipline.cost_inputs.unit_cost_ms` 的逐值。
17. **support-miss 纪律**（B5）：合成一个族不包围区间的 replicate ⇒ 该 replicate AUC = −1 且保留；joint miss > 1% ⇒ A-2 / A-6 fail；功效公式对已知 `(effect, sd30)` 的数值断言；`effect ≤ 0` / `sd30 = 0` ⇒ A-6 fail。
23. **G2R2 对抗回归**：(B1) `replicates ∈ {None, 1, 120, 9999, 10001, True}` / 错 seed → header 拒；R = 10000 通过；入口 `run()` 对 R = 120 的 cost map 拒绝；入口级 happy path 改用真实 R = 10000 的 map（模块级 fixture）。(B2) 审查反例：101 点网格 max < 0 而断点 c = 0.024 处 +4e-5 → `frontier_difference_extrema` 给出精确 max 与断点集合；design 的 A-2 记录 `checked_breakpoints`。(B3) `[(0,0),(0.3,1),(1,1)]` 的 hull 为 `[(0,0),(0.3,1)]`、support `[0, 0.3]`、不覆盖 `[0,1]`。(非阻塞) outcome 写出前二次核 cost map SHA，tmp + `os.replace` 原子写。
22. **G2R1 对抗回归**：(B1) 未登记 run id 的 accepted 行 → summary / cost-map 拒绝；登记 run 未执行某臂 → `roster_complete=false`、cost-map 拒绝；重复 run id / pool attestation 漂移 → discipline 拒绝。(B2) outcome `run()`：只翻 success → 拒；只改 verdict tier → 拒；换另一合法 package → 拒；SHA 不符 → 拒；改 seed 后索引 SHA 变。(B3) 精确积分 0.85、断点不在网格、反对称、同 hull 为 0。(B4) `w` / `quantile_alpha` 篡改 → parity 拒；嵌套 placeholder 拒；export record 多余键 → 拒；anchor 加 tier → discipline 拒。(B5) 并列极值的 endpoint 语义；`np.ones(10000)` 下标 9950 / 49。(B6) per-task 块、A2/A5/A6 键、H1/H2 无 `a6_pass`。(B7) runner `main --dry-validate` 成功 + subset + `--layer` 不符 / 带 flag 的 primary matrix / server 指纹不符均拒；完整迁移链 exporter → emitter → runner dry-validate → discipline → cost-map 在历史目录不存在时通过。
18. **分位下标**（B6）：R = 10000 的合成数组断言 `np.quantile(L, .995, method="higher") == np.sort(L)[9950]`、`np.quantile(H, .005, method="lower") == np.sort(H)[49]`；threshold 族不经过 isotonic 路径（AST / 调用计数断言）。
19. **D0 digest-chain**（R3-B1）：迁移回归中 **不 monkeypatch** `validate_export_d0_binding`，在 `/tmp` 原路径不存在时 exporter 仍通过；attestation 内任一声明 digest 被篡改 ⇒ exit；静态断言 exporter import graph 不含 `d0_check.validate_input_attestation`。
20. **status 旁路**（R3-B2）：journal 与 per-step 的 `success` **和** `status` 同时删除 / 任意替换后 `cost_map_frozen.json` 字节不变；AST 断言 cost-only 路径无 `"success"` / `"status"` 键访问。
21. **plug-in vs bootstrap**（R3-B3）：构造含少量 support-miss 的样例，断言 `effect` 等于 full-sample plug-in 值、不等于 bootstrap mean，且 `sd30` 随 −1 replicate 数量增大；plug-in hull 不覆盖区间 ⇒ A-6 fail。

回归：全量 `pytest tests/`（`--ignore` 既有 `test_prebuilt_matrix_backend.py` 两个已知失败）+ ruff。

## 7. 验收与放行

- G2 APPROVED 后：§3.8-c 归档 → 推远端核 SHA → 两份 suite matrix 各一次 `--dry-validate` → owner 放行 rollout（可分 subset launch，每次继承同一 frozen matrix / roster ledger；合计 2700 ep）→ `phase0_summary` → `cost_map` → SHA 冻结 → `phase0_outcome_design` → 协议 §4 数值写回 → 第二计划 G1。
- **Verify recipe（真实冻结产物，远端执行）**：
  1. weilandserver，`cd /data/openpi_dispatch`，同步本分支后用冻结输入重跑 fitter：
     `python -m exp.dispatch_surface.fit_surface --table exp/dispatch_surface/data/libero_10/dispatch_table_fresh.jsonl --cohort-manifest exp/dispatch_surface/data/libero_10/query_cohort_manifest.json --cache-yaml exp/dispatch_surface/config/calibration_retrieval_libero_10.yaml --rebuild-record exp/dispatch_surface/data/libero_10/cache_artifacts/rebuild_record.json --split-manifest exp/dispatch_surface/data/libero_10/init_pools/split_manifest.json --weights-npz exp/dispatch_surface/data/libero_10/dispatch_weights.npz --d0-record exp/dispatch_surface/analysis/d0_libero_10.json --certification-mode empirical_no_certificate --out-dir /tmp/verify_fit/libero_10`（S0 同 + `--s-only --frozen-record`）；期望 `sha256(surface_sv_primary.npz) = dcbd256ac45576be…`、`surface_s_only_primary.npz = 574525c87833382b…`、`surface_sv_minus.npz = 8d9ab2270bd0aa85…`（spatial：`d559510b38c1bba2… / 3fd7a67e72b13efc… / e10ae6c3b0906229…`），逐字段语义比较同时输出。
  2. timan107，重跑 `analyze_t107.sh <suite> primary` → `verdict.json` 与归档副本字节等价（l10 matrix `225430306e1a31f2…`、per_step `d67af61f269e3d87…`；spatial `15dbaf995e1e19b8…` / `d2d15ae25809c3da…`）。
  3. `archive_rev1_discipline.py` 两 suite → `rev1_package.verify_package` 通过；exporter 对真实包导出 l10 p85 / spatial p95、p975 / S0 p80、p95 并核 `final_fit_digests`。
- 本计划不含任何 rollout 自动启动；server 保持停止直至放行。

## 8. Review Log

### G2 Round 1 — Review Authority（2026-08-29）

**Verdict: NEEDS REVISION（不放行 Verify，不放行任何 Phase 0 rollout）。** 新增模块可以导入，现有定向测试与 Ruff 均为绿色；但绿色主要覆盖 happy path。独立白盒审查发现，冻结的 provenance、ledger attribution、AUC estimand 和 cost-map 机械规则仍存在可复现绕过，其中当前 synthetic fixture 本身就在用“不属于 ledger 的 run_id”生成结果，却被 discipline/summary 接受。

**Reviewer 独立验证：**

- `tests/dispatch_surface/test_rev2_phase0.py`：**34 passed**。
- `tests/dispatch_surface tests/data_authority`：**318 passed**；`tests/cache/test_surface_binding.py`：**24 passed**。
- 本线 Python targeted Ruff：**All checks passed**；`git diff --check`：通过。
- 对抗复现 A（ledger attribution）：fixture 的 Phase 0 ledger 写 `runphase0abcd`（`test_rev2_phase0.py:352`），但全部 journal/per-step row 写 `run0123456789`（`:67,125-136`）；`test_phase0_discipline_and_summary` 仍通过。即结果可来自一个 ledger 从未执行的 run。
- 对抗复现 B（AUC）：hull `[(0,0),(0.3,1),(1,1)]` 在 `[0,1]` 的精确 normalized integral 为 `0.85`，当前 `auc_norm()` 返回 `0.8491271820448878`。
- 对抗复现 C（order-statistic metadata）：`L=H=np.ones(10000)` 时当前输出的 `qL_zero_based_index=qH_zero_based_index=0`，冻结合同要求固定为 `9950/49`。
- 对抗复现 D（placeholder）：`_forbid_placeholders({'nested': {'source': 'pending'}})` 被接受，未实现“meta 任意层级不得含 placeholder”。

#### Blocking findings

**G2R1-B1 — Phase 0 结果没有绑定到 launch ledger 实际执行的 `(run_id, arm)`，ledger 自身也未按 Rev 1 同等级纪律验证。** `phase0_discipline.validate()`（`phase0_discipline.py:36-93`）只收集 executed-arm 并返回 run-id 列表：不拒重复 run-id，不验证每 entry 的 `executed_arms ↔ executed_yaml_sha256`，不核 `trials_per_task/replan_steps/env_seed/contract_binding/pool`，也不构造 `executed_arms_by_run`。随后 `phase0_summary`、`cost_map`、`phase0_outcome_design` 只把 journal 的 run-id 与 per-step 自洽 join，从未证明该 run 的 ledger 执行过该 arm。上述 fixture 的两套 run-id 明确不同却全绿，是旧实现必败证据。**放行条件**：Phase 0 discipline 达到 Rev 1 `check_discipline` 的 ledger/contract/pool 强度，要求唯一 run-id、逐 entry YAML/arm/pool/contract 交叉；返回 `executed_arms_by_run`；cost-only 与 outcome loader 后统一拒绝任一 accepted `(run_id, arm)` 未被相应 launch 认领。修正 fixture 为真实一致 run-id，并加“任意一行改成未登记 run-id”“登记 run 但未执行该 arm”两条必拒回归。

**G2R1-B2 — `cost_map_frozen.json` 没有冻结生成它的 journal/per-step 字节，outcome 阶段可换一套结果而 cost-map SHA 不变。** `_load_phase0_source()` 丢弃 cost loader summary（`cost_map.py:185-197`），`build_cost_map()` 输出没有 plan §3.6 要求的 input digests（`:303-327`）。`phase0_outcome_design.main()` 又重新读取命令行 journal/per-step（`phase0_outcome_design.py:145-168`），只核 cost-map 的 Rev 1 manifest SHA 与 Phase 0 matrix context；它既不核这两份日志的 SHA，也不核命令行 `--rev1-package-manifest` 自身 SHA 等于 cost-map，且没有把重新得到的 cells/cost 与冻结 map 对照。因此冻结后替换 Phase 0 outcomes、cost rows，或传入另一份合法 Rev 1 package，仍可能改变 AUC。**放行条件**：cost-map 写入并冻结 Rev 1/Phase 0 的 manifest、matrix、ledger、split、journal、per-step（及必要 export records）内容 SHA；outcome main 在读 outcome 前逐项重算 exact digest，并要求命令行 Rev 1 manifest SHA 同时等于 cost-map 与 Phase 0 context；冻结 seed=20260829、R=10000、cost-model digest/schema；另对重新加载的 point cost/decision count 与 cost-map 做一致性复算。加“只翻 success”“只改 verdict tier”“换另一合法 package”“改 seed/R 后重算 cost-map 文件 SHA”均必拒的 main-level 测试。

**G2R1-B3 — `auc_norm()` 计算的是 401 点算术平均，不是协议冻结的积分。** `frontier_hull.py:78-83` 对包含端点的均匀网格取 `mean`；这既不是 trapezoidal integral，也不对不落在网格上的 hull breakpoint 精确。复现误差约 `8.73e-4`，在接近 0 的单侧判据上足以改变结论；而协议 §4 明确定义连续线性 hull 上的 normalized integral。**放行条件**：在 `[c_L,c_H]` 加入所有内部 hull vertices 后，对 piecewise-linear frontier 做精确梯形积分并除区间宽度；H1/H2 继续共享同一函数。测试至少覆盖 breakpoint 不在固定网格、解析值已知、A−B 反对称、同一 hull 差为 0；禁止用增加 grid 密度近似替代。

**G2R1-B4 — runner/discipline 没有独立重放 exporter 承诺的“除 δ/boundary 外逐字段等同 source artifact”合同。** emitter 做了一部分数组/字段比较，但 `validate_exploratory_matrix_artifacts()`（`run_precheck.py:543-585`）只重放 `certification_mode/retrieval_contract/h_exec/k/uses_disagreement/delta`，未比较 `w/active_mask/v_bin_edges/quantile_alpha/start_t_ws/conformal_c/n_calibration_episodes`；也未拒 export-record 重复覆盖或校验其 protocol/posthoc/cost digest。`phase0_discipline` 复用该弱 validator，且不重算当前 YAML SHA、不调用 anchor/gate shape 校验。`_forbid_placeholders()` 又只检查 meta 顶层字符串（`export_exploratory_surface.py:143-146`），嵌套 placeholder 可通过。**放行条件**：抽一个 emitter/runner/discipline 共用的完整 template-parity helper，逐字段/逐数组核验所有非 boundary 字段；export record exact schema、唯一 arm 映射、protocol/posthoc/cost-model 均 fail closed；placeholder 递归遍历 dict/list；discipline 重算 YAML 字节并验证 anchor threshold>1、无 warm tier、always_search。每种 tamper 均需旧实现必败回归。

**G2R1-B5 — cost-map 的 endpoint/tie 与 order-statistic 记录没有实现冻结机械规则。** 正式规则按数值 δ 排序并将 spatial SV 外端冻结为 p80/p97.5；当前代码明确“不需要”读取 Rev 1 数值 δ、改用 quantile key（`cost_map.py:213-217`），而 extrema 用 `np.argmin/argmax`（`:244-255`）。当 isotonic 产生并列最小值时，`argmin` 选择第一个较小 δ，可把 p95 当低成本 endpoint、把更外侧 p97.5 留作 middle，违背冻结 roster 的 outer endpoint 语义。另 `interval_from_endpoints()` 用 `searchsorted(value)` 记录分位下标（`:109-115`），ties 时不是 method 所对应的 order-statistic index；复现返回 0/0 而非 9950/49。**放行条件**：从受认证 artifact/export/fit record 读取并输出每个 SV/S0 候选的真实 δ，严格按 δ 排序；并列最小成本取最大 δ、并列最大成本取最小 δ（或按 plan 明示的 p80/p97.5 outer endpoint 直接钉死）；分位下标由 `(q,n,method)` 的定义计算并在 R=10000 强制 9950/49，不用 value search。补 isotonic flat/tied extrema 与重复分位值回归。

**G2R1-B6 — outcome-design 缺少 G1 要求的 per-task descriptive 输出与明确 Decision Gate 汇总。** `phase0_outcome_design.design()` 只输出 pooled effect、bootstrap 与 LOTO（`phase0_outcome_design.py:66-122`）；没有 plan §3.7 要求的 per-task descriptive AUC/discordance，也没有清晰的 A-2/A-5/A-6 汇总（当前每个 H1/H2 都写名为 `a6_pass` 的字段）。这会再次诱发协议已专门修正过的“LOTO 全正 = 每 task 全正”误读。**放行条件**：逐 task 输出三族 points/hull/support、H1/H2 descriptive AUC 与 discordance/ceiling/floor 标记；单独输出 Decision Gate A-2（spatial 描述）、A-5（H2 support）和 A-6（仅 H2 power）的机械布尔与原因，H1 不得复用 `a6_pass` 名称；support 不足保持 fail closed。

**G2R1-B7 — G1 冻结的关键回归有多处缺席或为无效断言，无法支撑 G2 放行。** 具体包括：`test_primary_matrix_with_exploratory_flag_is_refused`（`test_rev2_phase0.py:614-618`）只构造 dict 后 `assert` flag 存在，根本没调用被测入口；迁移测试（`:779-797`）只跑 package+exporter，未跑计划要求的 emitter→runner dry-validate→discipline→cost-map；没有 runner `main --dry-validate`/subset-resume 的成功与拒绝测试；没有 outcome `main` 的 input-SHA/只读 cost-map 测试；fitter/precheck-IO 的“真实冻结产物字节等价”也未按 §6-1/§6-4 完整执行。**放行条件**：补齐上述入口级测试，特别是 B1–B6 每项至少一条旧实现必败回归；迁移链不得 monkeypatch provenance validator；若真实冻结输入只在远端，须在 Verify recipe 中给出可执行命令与期望 SHA，而不是以 synthetic helper 替代。

#### Non-blocking observations

- `external_asset` record 的 `integrity.file_count=1758`，但 `members` 只放一个 `size=0` 的示例；当前 registry 不验证 members 与 rollup 一致。完整 rollup SHA 已记录，所以不单独阻断 Phase 0，但建议删除这个伪完整的 `members` 字段或明确改名为 `sentinel_files`，避免读者误以为它是 1758 文件 census。
- `build_task_manifest.py` 的循环验证 benchmark→split，但没有显式拒绝 split assignment 的额外 task key；当前两份实际 manifest 都是 10↔10，属于防御性补强，不单独阻断。
- `phase0_summary --executed-only` 可产出部分 roster 的同名 summary。若保留，应在输出中显式 `partial_nonadjudicative=true` 且禁止作为 A-4/cost-map 输入；正式 summary 仍必须覆盖完整 roster。

**下一 gate**：Execution Authority 逐条响应并修复 **G2R1-B1…B7**，补旧实现必败回归后进入 G2 Round 2。此前 server 保持停止；不得 archive/emit 正式臂矩阵，不得启动旧 A′ rollout，也不得依据当前 outcome-design 输出写回协议数值。

### G2 Round 1 — Executor（2026-08-29）

七条 blocking 与三条 non-blocking 全部核实并修复；正文 §3.5–§3.7、§6、§7 已同步。独立复现了审查的四个对抗例：A（fixture 两套 run id 全绿）、B（`auc_norm` 0.8491 vs 0.85）、C（`np.ones` 下标 0/0）、D（嵌套 placeholder 通过）——旧实现均必败，新实现均拒绝 / 精确。

- **B1 Accepted** — `phase0_discipline.validate` 重写至 Rev 1 强度（唯一 run id、逐 entry YAML / arm / pool / contract 交叉、YAML 字节重算、anchor 重验），返回 `executed_arms_by_run`；新增 `assert_rows_claimed`，summary / cost-map（Rev 1 源亦按归档 ledger 认领）/ outcome-design 在 loader 后统一调用。fixture 改为一致 run id + 真实 A′ pool attestation（`validate_aprime_pool` 产出）；回归：未登记 run id 行（anchor 与 surface 各一）→ 拒；登记 run 未执行某臂 → 拒；重复 run id / pool 漂移 → 拒。
- **B2 Accepted** — cost map 写入 `input_sha256`（Rev 1 六个成员 + Phase 0 六项）；outcome `run()` 在读 outcome 前逐项重算并要求命令行 Rev 1 manifest SHA 同时等于 cost map 与 Phase 0 context，seed / R 非冻结值拒绝，重载 point cost / decisions 与 map 一致（1e-9）。回归：只翻 success / 只改 tier / 换合法 package / SHA 不符均拒；改 seed 索引 SHA 变。
- **B3 Accepted** — `auc_norm` 改为断点（区间端点 ∪ 区间内 hull 顶点）上的精确梯形积分；严格支配改为"成本严格更低且 SR 严格更高"（同 SR 更贵的点留在包络、延伸支撑，与审查例一致）。回归：0.85 精确、断点不在网格、反对称、同 hull 为 0。
- **B4 Accepted** — 新增 `template_parity.py`：`assert_template_parity`（全部非 δ / boundary 标量与数组 + contract）、`assert_no_placeholders`（递归 dict / list）、`assert_export_record_schema`（exact key set、family ↔ source_role、protocol / posthoc / cost digest）；exporter 写前、emitter、runner validator、discipline 共用；runner 拒绝 export record 重复映射；discipline 重算 YAML 字节并重验 anchor。回归：`w` / `quantile_alpha` 篡改、嵌套 placeholder、多余键、anchor 加 tier 均拒。
- **B5 Accepted** — Rev 1 候选的 δ 从归档 artifact 读、Phase 0 从受认证 artifact 读；严格按 δ 排序并核与分位序一致；`family_endpoints`：并列最低取最大 δ、并列最高取最小 δ；`quantile_index(q, n, method)` 按定义算零基下标，R = 10000 强制 9950 / 49，并断言与 NumPy 值一致。回归：并列极值、`np.ones(10000)`。
- **B6 Accepted** — outcome-design 输出 `per_task_descriptive`（三族 points / hull / support、H1/H2 描述性 AUC、middle 臂 discordance、ceiling / floor）与 `decision_gates.{A2, A5, A6}`；hypothesis 块改为 `support_ok / power_n40 / verdict`，删除 `a6_pass`；附 LOTO ≠ 每 task 的 note。
- **B7 Accepted** — 删除无效的自我断言测试，改为 runner `main --dry-validate` 入口级测试（stub server 指纹；成功、subset、`--layer` 不符、带 flag 的 Rev 1 matrix、server 指纹不符）；迁移链改为 exporter → emitter → runner dry-validate → discipline → cost-map，历史目录移走，不 monkeypatch 任何 provenance validator；outcome `run()` 的 SHA / 只读测试；§7 给出真实冻结产物的 Verify recipe（命令 + 期望 SHA）。
- **Non-blocking Accepted** — `external_asset` 记录 `members` 改名 `sentinel_files`（size 置空）；`build_task_manifest` 拒绝 assignment 多余 task key；`--executed-only` summary 标 `partial_nonadjudicative=true`，cost-map 要求 `roster_complete`。

本轮定向测试：`tests/dispatch_surface/test_rev2_phase0.py` 45 passed；`tests/dispatch_surface tests/data_authority tests/cache/test_surface_binding.py` 353 passed；ruff / `git diff --check` / `registry validate` 全净。未进入 Verify，server 停止，无 rollout。

### G2 Round 2 — Review Authority（2026-08-29）

**Verdict: NEEDS REVISION（不放行 Verify，不放行 Phase 0 rollout）。** Round 1 的 ledger attribution、input digest、精确 AUC 积分、template parity、端点/分位下标与 per-task 输出均有实质修复；但独立复核又找到两个可改变结论的可复现实现错误，以及一个在 G2 中未经裁决便改写已冻结 estimand 的问题。当前 45/353 绿测试没有覆盖这三条。

**Reviewer 独立验证：**

- `tests/dispatch_surface/test_rev2_phase0.py`：**45 passed**。
- `tests/dispatch_surface tests/data_authority tests/cache/test_surface_binding.py`：**353 passed**；targeted Ruff：**All checks passed**；`git diff --check`：通过。
- 对抗复现 A（R 冻结旁路）：`seed=20260829` 时，`replicates=1/120/9999` 对 `phase0_outcome_design.py:224` 的布尔式均得到“不拒绝”。现有 `_frozen_cost_map(..., reps=120)` 又在 `test_b2_outcome_stage_refuses_any_frozen_input_drift` 中成功跑完 `run()`，所以这不是理论边角：当前测试本身就证明了 `R=10000` 没有冻结。
- 对抗复现 B（A-2 假支配）：在 `[0,4]` 上，SV hull `[(0,.4998),(.024,.50196),(4,.50196)]` 与 T hull `[(0,.5),(.016,.5016),(4,.66096)]` 的 101 点网格差值最大为 `-0.0002`，但真实折点 `c=.024` 的差值为 `+0.00004`。当前实现会报 `sv_dominated_on_interval=true`，精确判定则为 false。

#### Blocking findings

**G2R2-B1 — `seed/R` fail-closed 条件的逻辑错误让任意非空 R 通过。** `phase0_outcome_design.py:224` 写成 `seed != fixed or reps != 10000 and reps is None`；由于 `and` 优先级与第二个子句的内容，只有 `replicates is None` 才会因 R 被拒绝。这直接违反 G2R1-B2 的放行条件和 Executor “seed / R 非冻结值拒绝”的响应。**放行条件**：使用常量并独立比较 `seed != SEED or replicates != REPLICATES`；提取轻量 header/schema validator，回归分别覆盖 `None/1/120/9999/10001`、错 seed、布尔值及正确 `10000`。main-level 的 happy path 不得继续用 R=120 来假装正式 cost map；如需快速测试，可对纯 `design()` 路径使用小 R，不能绕过入口级冻结检查。

**G2R2-B2 — A-2 用 101 点采样代替精确 piecewise-linear 支配判定，可误报 spatial negative control。** `phase0_outcome_design.py:157-160` 对 `np.linspace(c_L,c_H,101)` 求 min/max；这与本轮已修正的“折点精确积分”同理，两条线性 hull 之差的极值只需、也必须在区间端点与两个 hull 的内部折点并集上精确求。上述反例已证明当前网格能改变 `sv_dominated_on_interval`。**放行条件**：新增共享的 exact frontier-difference extrema helper，断点为 `{c_L,c_H} ∪ vertices(SV) ∪ vertices(T)` 的区间内并集；A-2 记录实际检查断点与精确 min/max。加入上述“网格全非正、折点为正”的旧实现必败回归。

**G2R2-B3 — G2 修复 AUC 时同时改了 G1 已冻结的 dominance/support estimand，且新定义可用“同 SR 但更贵”的被支配臂制造 support。** G1 通过时的 plan 只引用协议 §3.2；协议写的是“删除被另一点严格支配的点，且只在实际 hull support 内积分”。Round 1 的 B3 只要求把 401 点均值改为精确积分，没有授权改 dominance。Executor 却在 `frontier_hull.py:42-50` 与 plan §3.7 新增“必须同时成本严格更低且 SR 严格更高”，因而保留 equal-SR/higher-cost 点并延长 support。这不是纯实现细节：它能把原本的 support-miss 变成可裁决 replicate，而该额外成本没带来任何 SR 收益，与“Pareto frontier”的常规语义也不一致。**放行条件**：本 reviewer 建议恢复常规弱/单边严格支配：`c2 <= c and s2 >= s and (c2 < c or s2 > s)`，因此 equal-SR/higher-cost 点删除，并加回归。如 owner 明确要把横轴解释为“可用 budget”并允许平坦延伸，则必须先做显式 protocol amendment，统一 `logs/dispatch_surface_rev2_protocol_draft.md` §3.2/§12.4、support 定义与回归，再重过 G1；不得以 G2 bugfix 的名义暗改。

#### Non-blocking observation

- `phase0_outcome_design.py:288-291` 先写出 outcome 文件，再做“cost map 在运行期未变”的最后一次 SHA 检查。若恰有并发变更，命令虽失败但磁盘上会留下看似合法的 outcome。建议把二次 SHA 检查移到写文件之前，并用临时文件 + 原子替换避免部分写入；这不是当前单进程正式路径的主要风险，不单独 blocking。

**下一 gate**：Execution Authority 修复 **G2R2-B1/B2**，并对 **B3** 恢复冻结语义或走显式 amendment + G1，补上三条旧实现必败回归后进入 G2 Round 3。此前 server 保持停止；不得 archive/emit 正式臂矩阵，不得启动 A′ rollout，不得依据当前 A-2/outcome-design 输出写回协议数值。

### G2 Round 2 — Executor（2026-08-29）

三条全部核实并修复。复现：A（`seed=20260829, replicates ∈ {1,120,9999}` 均不拒）确认为 `or`/`and` 优先级错误；B（审查的两条 hull 在 101 点网格 max = −0.0002、断点 0.024 处 +4e−5）确认。

- **B1 Accepted** — 新增 `validate_cost_map_header`：`seed == SEED`、`replicates == REPLICATES`（整数、非布尔）独立比较，`posthoc_exploratory / outcome_blind / a3_pass` 与必填键；`run()` 入口调用。回归覆盖 `None/1/120/9999/10001/True`、错 seed、R = 10000 通过；入口级 happy path 改用真实 R = 10000 的 cost map（模块级 fixture），小 R 只用于纯 `design()` / `build_cost_map()` 路径；为此把 endpoint bootstrap 与 replicate (cost, SR) 向量化（同一共享索引，结果定义不变）。
- **B2 Accepted** — 新增共享 `frontier_difference_extrema(hull_a, hull_b, c_L, c_H)`：断点 = `{c_L, c_H} ∪ 两 hull 区间内顶点`，精确 min / max 与 argmin / argmax；A-2 改用它并记录 `checked_breakpoints`。审查反例进回归（网格全非正、断点为正）。
- **B3 Accepted（恢复冻结语义）** — `upper_concave_hull` 恢复协议 §3.2 的 Pareto 支配 `c2 <= c and s2 >= s and (c2 < c or s2 > s)`：同 SR 更贵的点被删除、不延伸支撑；plan §3.7 措辞同步回冻结定义；回归钉住 `[(0,0),(0.3,1),(1,1)] → [(0,0),(0.3,1)]`、support `[0,0.3]`、不覆盖 `[0,1]`。执行方承认 Round 1 修 B3 时越权改了 estimand；本轮不提 amendment，按冻结语义执行。
- **Non-blocking Accepted** — outcome 写出前二次核 cost map SHA，tmp 文件 + `os.replace` 原子替换。

未进入 Verify；server 停止；无 rollout。

### G2 Round 3 — Review Authority（2026-08-29）

**Verdict: APPROVED（代码 gate 放行进入 Verify；正式 rollout 仍需按 §7 完成真实产物核验与 dry-validate）。** G2R2-B1/B2/B3 均按放行条件修复：R/seed 冻结 fail closed，A-2 改为折点并集上的精确极值，Pareto dominance 恢复 G1 冻结的弱/单边严格规则。对抗反例均已进入回归。

**Reviewer 独立验证：**

- `tests/dispatch_surface/test_rev2_phase0.py`：Claude 基线 **57 passed**；reviewer 加固后 **67 passed**。
- `tests/dispatch_surface tests/data_authority tests/cache/test_surface_binding.py`：**372 passed**。
- targeted Ruff：**All checks passed**；`git diff --check`：通过。
- 新向量化 bootstrap 与旧 ratio-of-sums 逐 replicate 实现的独立随机对照：最大绝对差 `1.7763568394002505e-15`（只有浮点求和顺序误差），索引与聚合语义不变。
- 全仓 `tests/ --ignore=tests/exp/test_prebuilt_matrix_backend.py` 在受限 sandbox 中的首个失败为 `tests/ablation_study/test_sidecar_executor.py::test_roundtrip_and_timing_log`，根因是 `socket.socket()` 被 sandbox 以 `PermissionError` 拒绝；该模块与本线无代码交叉。因此不把本次 sandbox 全仓结果冒充为“全量绿”，Verify 在允许 loopback socket/GPU 的正式环境依 §7 重跑。

**Review Authority 放行前加固（owner 授权直接修改，均保留在 staged 基线之外）：**

1. `validate_cost_map_header` 从“键存在”升级为 exact/fail-closed schema：重放 protocol、RNG/分位方法、analytic cost payload+digest、suite roster、Rev1/Phase0 input-digest 键域与 SHA-256 形状、candidate/family selection、`selected == [low,middle,high]`、正成本/决策数、严格递增四锚点，并独立重放 A-3 的区间宽度/三成本不同/端点包围检查与 bootstrap digest；同时在 outcome 输出中保留 `export_records` digest，不再丢失已核验的 provenance 字段。
2. bootstrap RNG/index 生成收敛到 `cost_map_api.shared_index` 单一实现，删除 cost-map 复制体；`index_arrays` 对空索引、replicate 长度不等于冻结网格、网格外 cell 全部 fail closed。
3. `difference_breakpoints` 对非有限/空区间显式拒绝，避免绕过上层时产生伪极值。

**Staging 边界：** Claude/Execution Authority 的 Round 2 修复已全部 staged；上述 reviewer 加固、其回归与本 Round 3 review log 保留 **unstaged**，便于下一会话逐项审阅。

**Verify 入口**：按 §7 三步执行真实冻结产物 SHA/语义等价、Rev 1 verdict 字节等价、两 suite package/export/dry-validate。任一不符即回到 G2；在 Verify 完成前 server 保持停止，不启动 A′ rollout。

## 9. Verify 记录（2026-08-29，Execution Authority）

**§6 全量测试**（`.venv/bin/python -m pytest tests/ --ignore=tests/exp/test_prebuilt_matrix_backend.py --ignore=tests/review_tests`，本地 WSL，允许 loopback socket）：**4594 passed / 45 skipped / 0 failed**（306.7 s）。`tests/exp/test_prebuilt_matrix_backend.py` 单独运行：`test_cosine_fast_path_bit_identical`、`test_fast_path_robust_to_candidate_reordering` 2 failed / 5 passed——G1 认可的既有失败，与本线无代码交叉。定向：`test_rev2_phase0.py` 67 passed（含 reviewer 加固的 10 项）。ruff / `git diff --check` / `registry validate` 全净。

**Reviewer 放行前加固的独立核验**：向量化 endpoint bootstrap 与逐 replicate `ratio_of_sums` 循环在独立随机对照上最大绝对差 1.28e−13；`index_arrays` 对空索引 / 长度不等 / 网格外 cell 三种输入均 `SystemExit`；`validate_cost_map_header` 的 A-3 重放与 `cost_map` 的判定同源。未作修改。

**§7 真实冻结产物核验**（代码经 `tether push` 同步到 weilandserver `/data/openpi_dispatch` 与 timan107 `/scratch/zixuans8/openpi_dispatch`，30 个文件 sha256 逐一核对无差）：

| 步 | 结果 |
|---|---|
| §7-1 fitter 复算（weilandserver，冻结输入 + 原路径字符串；S0 用 `--s-only --top-k 1 --frozen-record /tmp/dsp_shared/<suite>/surface_rev1/fit_record.json`） | 六个 artifact **逐字节一致**：l10 `surface_sv_primary` `dcbd256a…`、`surface_sv_minus` `8d9ab227…`、`surface_s_only_primary` `574525c8…`；spatial `d559510b…` / `e10ae6c3…` / `3fd7a67e…` |
| §7-2 analyzer 复算（timan107，重构后的 `analyze_precheck`） | 两 suite `verdict.json` 与归档副本 `cmp` **逐字节一致** |
| §7-3 归档 + 导出（本地） | `archive_rev1_discipline` 两 suite 各 19 成员，`verify_package` 通过（manifest `48eccb9f…` / `4f9f79b2…`），台账 `dispatch_surface/<suite>/rev1_discipline` 登记并 validate；导出器在真实包上复算 `final_fit_digests` 一致，导出 l10 SV p85（δ 5.5032）、S0 p80/p95（5.2381 / 6.7314）；spatial SV p95/p975（6.7177 / 7.2004）、S0 p80/p95（5.5020 / 6.7177）。交叉核对：l10 p80 = 5.2381 = Rev 1 `delta_neighbours.minus` |

**状态**：Verify 通过。server 停止；未跑任何 rollout；Phase 0 正式 emit / dry-validate / 放行仍按 §7 由 owner 决定。
