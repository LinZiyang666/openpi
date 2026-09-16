# Stage 2 — V2 增益机制离线研究（2a SR 增益分解 / 2b 公平 Pareto / 2c 危险步）Plan

- **Status**: Done（L2 收官：G1 APPROVED R2 / G2 APPROVED R2 2026-07-05；§4 Code + 正式产物 `analysis/stage2_*` 落盘；三线机制判决回填 roadmap §5；§6 Verify + commit；待 owner 确认后归档）
- **Authority**: Execution
- **Level**: L2（三个新分析组件 + 测试；结论承载 Stage 3 N4 设计。纯离线 / 0-GPU）
- **Date**: 2026-07-05
- **上游**: [`logs/gate_exploration_roadmap.log.md`](gate_exploration_roadmap.log.md) §5 Stage 2（owner 已排定的 2a/2b/2c 方法与及格线，本 plan 是其实现级细化）
- **本 plan 目标产物**: `exp/gate_research/analysis/` 下三份 .md 判决 + 两张 Pareto overlay 图；随后回填 roadmap §5 Stage 2 行
- **三线关系（已核实）**: 2a/2b/2c 无"输出→输入"耦合，全部读只读数据，可并行；共享"载入/配对/聚合"底座先建一次（2a、2b 共用）。Stage 3 是唯一同步点（2a 定 N4 注入间隔 L，为其硬前置）。

---

## 0. 背景速览（Stage 1b live 判决 → Stage 2 动机）

Stage 1b live（8 run × 500 ep）判决把 gate 问题从"何时省搜索"升维为"(SR, inf_ratio, 延迟) 三元调度"。核心反常现象（roadmap F10–F12）：**skip 本身带 SR 增益（V2）**——N1 与 matched-periodic 的 SR **都** ≥ always_search baseline。N1 把 skip 集中在预测 MISS 步（动作分布不变 → SR 保真但吃不到 V2）；periodic 均匀 skip（等效限制最大连续缓存执行长度 → 吃到 V2 但延迟崩，因 skip 撞 FULL_HIT 步换全推理）。

Stage 2 = 用**已在盘的数据（0 GPU）**裁决 V2 从何而来，产出可翻译成 N4 gate 规则的证据：

- **2a**：SR 增益分解 + 三假设裁决 —— H1（剂量/截断：SR 增益 ∝ 连续缓存执行 run 被截断程度）/ H2（on-manifold 反馈：定期新推理把轨迹拉回流形 → 后续搜索命中更好）/ H3（WS 执行中毒：SR 损失集中于 WARM_START 部分去噪回放，均匀 skip 顺带打断）。
- **2b**：把 8 live 点 + baseline + d1 前沿 7 config + RPG 锚点放同一 (SR, inf_ratio) 图，正式核对 F11 初判"periodic 抬高既有前沿 ~3-5pp"，判决"gate 是否构成第四设计轴"。
- **2c（可选，scope 受限）**：把 `deviate_score≥5` oracle 危险步标签 join 到 gate_rows，检验危险步能否被 prev_score/其他廉价信号预测 → 为 N4 注入的危险步靶向提供证据。

**评估坐标（C6+C10）**：一律在 (SR, inf_ratio, net 三档 4/34/70ms) 三元报告；"同预算"= 同 inf_ratio 轴，**skip% 已被 C10 判为病态轴**，禁止用作对齐轴。

---

## 1. 数据资产（全部已在盘，勘察已核实）

| 用途 | 路径 | 关键 schema / 数值 |
|---|---|---|
| 8 live run（2a/2b） | `exp/gate_research/data/n1_live/{spatial_fh75_ws10_A,B, l10_fh5_ws40_A,B, spatial_A_periodic,B_periodic, l10_A_periodic,B_periodic}/` | 每 run 3 文件：`manifest.json`（gate_type/theta/j/M/cache_len/inference_len/replan_steps=5/matched_to/yaml_id）、`rows.jsonl`（per-step + `_kind=episode_summary`）、`journal.jsonl`（per-ep success） |
| per-step 字段 | 同上 rows.jsonl | `task_uid="{yaml}:eval:{task_id}:{subset_init_state_idx}"`、`step_idx`(0,5,10…)、`hit_type`∈{FULL_HIT,WARM_START,MISS}、`start_t`(仅 WS 非空=0.5)、`winner_id`、`cp1_score`、**`searched`(bool，N1 权威；periodic 无此字段需闭式重建)**、`success` |
| Stage-0 gate_rows（2a/2b/2c） | `exp/gate_research/data/{libero_spatial,libero_10}/gate_rows.jsonl` | always_search 全 searched=True；libero_spatial 39,136 per_step(3 config)、libero_10 143,763 per_step(4 config)；`collect.robot_state`(dim 32)；同上字段 |
| RPG 锚点（2b） | `exp/random_periodic_gate/analysis/aggregate.csv` | 列：`cfg,gate_type,param_slug,p_inference,seed,cache_len,inference_len,episodes,success_rate,mean_inference_ratio,inference_ratio_source`；**libero_spatial only**；3 keybuilder(clip_w7_d4/max_pool_w3_d5/spatial16_w8_d4)×periodic(k∈{1,2,5,10}×n∈{1,2,3,5,10})+random；`inference_ratio_source=derived`(异 verdict 口径) |
| d1 前沿 7 config 协议匹配锚点（2b） | SR: `exp/gate_research/analysis/gate_research_results.md` §3/§4；精确 inf: `exp/gate_research/analysis/n1_offline_frontier.md` §2/§3 | spatial fh75_ws15 85%/0.270、fh75_ws10 83%/0.287、fh40_ws40 91%/0.351；l10 fh80_ws10 57%/0.314、fh60_ws30 61%/0.369、fh40_ws40 68%/0.417、fh5_ws40 78%/0.636（与 8 live 同协议，可同轴） |
| 异协议对照（2b，仅标注不承重） | `exp/weighted_sum/analysis/{libero_spatial,libero_10}/threshold_pareto/threshold_pareto_per_yaml.csv` | held-out 45-init/100ep-cell，SR/inf 与上表不同（如 spatial fh75_ws10 95%/0.240）；不可同轴混 |
| 8 live 汇总（2b 复用或重算） | `exp/gate_research/analysis/n1_live_final.md` | 8 点 SR/skip/inf/net 全表；baseline SR spatial 82.6 / l10 77.6，inf 0.287/0.636 |
| deviate_score oracle（2c，仅 spatial） | `exp/trajectory_deviation/data/deviate_scores/deviate_score_{clip_w7_d4,spatial16_w8_d4,max_pool_w3_d5}.json` + GT attrs `exp/trajectory_deviation/data/gt/task_*/episode_*.h5` | JSON keyed `task_{TID}/episode_{EP}` → `{deviate_score:[float×num_cycles]}`；oracle 危险 = `deviate_score≥5.0`；GT h5 attr 提供 `(task_id,orig_init_state_idx,replan_steps=5)`；**libero_10 无此数据** |

---

## 2. 可复用 API（勘察已亲验签名 + file:line）

从 `exp/gate_research/analyze_n1_live.py` 直接 import（可复用，无副作用）：

- `load_jsonl(path) -> list[dict]` (L56)
- `parse_task_uid(task_uid) -> tuple[int,int]` (L61) —— 配对键 (task_id, subset_init_state_idx)
- `dedup_episodes(rows, yaml_id=None) -> dict[str,list[dict]]` (L69) —— 丢 episode_summary、取 max-attempt、按 step_idx 排序去重
- `episode_searched(ep, gate_type, manifest) -> list[bool]` (L137) —— **run-length primitive**：N1 读权威 `searched`，periodic 走 `reconstruct_searched` 闭式重建
- `reconstruct_searched(n_steps, cache_len, inference_len) -> list[bool]` (L130)
- `journal_success(rows, yaml_id=None) -> dict[str,bool]` (L114)
- `inf_value(hit_type, start_t) -> float` (L47) / `warm_cost(start_t) -> float` (L42)
- `mcnemar(n1, base, require_equal=True) -> dict` (L240) —— 返回 {n_paired,sr_delta_pp,b,c,mcnemar_chi2}（**连续校正 chi2，非精确 p**；精确 p 由 §3.1 新增 `mcnemar_exact_p(b,c)` 另算）
- `check_complete_decisions(episodes, spacing)` (L96) —— step_idx 网格完整性 fail-fast

**不可复用**：`gate_structure_analysis.py` 是单体脚本（无 `if __name__=="__main__"` 守卫，import 即跑全分析，L21 顶层读 `sys.argv`）。其 `auc(scores,labels)` (L34) 与 `warm_cost` (L29) 需在共享层**重写**（各 ~6 行标准实现），不 import 该文件、亦不改动它（WA §3.1 最小改）。

---

## 3. 落地设计

### 3.1 共享底座（先建一次，2a/2b/2c 共用）

**新文件 `exp/gate_research/stage2_common.py`**：

- `auc(scores, labels) -> float` —— 重写的 Mann-Whitney rank-AUC（不 import 单体脚本）。
- `mcnemar_exact_p(b, c) -> float` —— McNemar 两侧**精确二项检验** p：`n=b+c` 个不一致对，H0 下各对 50/50，`p = min(1, 2·Σ_{k=0}^{min(b,c)} C(n,k)·0.5^n)`。纯 `math.comb`，无 scipy 依赖；与复用的 `mcnemar()`（出 chi2）配套，2a 两者都报。
- `load_run_episodes(manifest_path) -> list[EpisodeRec]`：读一个 n1_live run，复用 `load_jsonl`+`dedup_episodes`+`episode_searched`+`journal_success`，返回每 episode 的 `(task_id, subset_init_state_idx, searched_seq, hit_type_seq, start_t_seq, cp1_score_seq, success)`。
- `action_source_seq(hit_type_seq, searched_seq) -> list[str]`：每步动作来源 ∈ {`CACHE_FH`, `WARM_START`, `NEW_INFER`}，其中 `NEW_INFER` = searched-MISS **或** skip（skip=强制全推理，C10）。WS 单列（H3 需要）。
- `cache_run_lengths(action_source_seq, include_ws: bool) -> list[int]`：最大连续缓存执行 run 长度（`include_ws` 控制 WS 是否计入缓存段，供 R8 敏感性）。
- `load_stage0_episodes(gate_rows_path, yaml_id) -> list[EpisodeRec]`：Stage-0 baseline 版（always_search，全 searched=True）。

### 3.2 Line 2a — SR 增益分解 + H1/H2/H3 裁决

**新文件 `exp/gate_research/stage2a_sr_decomp.py`**（CLI：`python -m exp.gate_research.stage2a_sr_decomp <manifests…> --out analysis/stage2_v2_mechanism.md [--include-ws]`；baseline 从每个 N1 manifest 内的 `baseline_gate_rows_path`/`baseline_yaml_id` 读取，无需单独传 Stage-0 路径）：

- **H1（剂量/截断）**：对 baseline + 各 periodic 剂量算 `cache_run_lengths` 分布 × ep 成败交叉表；periodic 7:1/4:1(spatial)、4:1/2:1(l10) 把 run 截断到 cap≤k；回归 SR 增益 ~ run-cap，检验 F12 的剂量饱和（spatial 7:1 SR 90.4 ≥ 4:1 89.0）。
- **H2（on-manifold 反馈）**：periodic vs baseline 的 **searched-step FH 率**、以及 FH 率 ~ 距上次注入步数曲线；证据坑 = spatial_A periodic inf 0.280<baseline 0.287（searched 步更多 FH）。
- **H3（WS 执行中毒）**：per-ep WS 执行量 × success 相关；periodic 是否缩短 WS 执行 run。
- **Δinf 三分解**：Δinf(periodic−baseline) = skip 转换项（skip 步强制 inf=1.0）+ verdict-mix 迁移项（searched 步 verdict 分布移动）+ ep 长度构成项；验证除 spatial_A 外 skip 转换项主导，spatial_A 为 verdict-mix 迁移主导（H2 佐证）。
- **统计**：复用 `mcnemar()` 出全对照配对的 b/c/连续校正 chi2，并用 §3.1 新增 `mcnemar_exact_p(b,c)` 出**精确二项 p**（N1/periodic vs baseline、N1 vs periodic 各一组；per-task 切片下 b+c 小，精确检验优于 chi2 近似）；per-task ΔSR 排单任务异常。
- **及格线（roadmap §5 2a）**：≥1 假设给出**可翻译成 gate 规则**的证据（如 H1 剂量曲线 → N4 注入间隔 L 取值）。**fallback**：三假设全不成立 → N4 仍按 uniform L 直接 live（periodic 已证 uniform 有效），2a 只影响注入靶向，不阻塞 Stage 3。

### 3.3 Line 2b — 公平 Pareto overlay

**新文件 `exp/gate_research/stage2b_pareto_overlay.py`**（CLI：`--spatial-rows <gate_rows> --l10-rows <gate_rows> --manifests <m.json…> --rpg-csv <aggregate.csv> --out analysis/stage2_fair_pareto.md --fig-dir analysis/ [--extra-anchor <suite:inf:sr:label>…]`）：

- 两张 (SR, inf_ratio) 图（**结构不对称，已定死**）：
  - **libero_spatial**：承重层 = baseline(82.6/0.287) + d1 前沿 2 config(fh75_ws10 83/0.287、fh40_ws40 91/0.351；及 fh75_ws15 85/0.270) + 4 live 点(2 N1+2 periodic)；参照层 = RPG 点云（挖空/灰标记，legend `(different search/judge)`，**不连线/不插值/不进判决**）。
  - **libero_10**：承重层 = baseline(77.6/0.636) + d1 前沿 4 config + 4 live 点；**无 RPG**（本就全同协议，更干净）。
- **F11 正式核对**：仅用**承重层同协议点**判"periodic 是否抬高既有前沿"（spatial_A periodic 0.280/90.4 是否在 RPG 坐标下严格支配 baseline/N1；跨 config 插值口径严格化）。RPG/异协议 threshold_pareto 一律不承重。
- **l10 越界与 `--extra-anchor`（G2 R1 加入）**：l10 四个 live 点 inf 0.642–0.759 超出 d1 锚 inf 上界 0.636 → 默认 OOR 无法判。可选 `--extra-anchor libero_10:1.0:83:pure_inf` 注入 roadmap 用的 l10 纯推理锚（0.83@1.0，`reference_pi05_libero10_baseline`，同 d1 实验协议）后按 fh5_ws40↔纯推理连线判 gain；默认关闭 = 保持 plan 原样诚实 OOR。纯推理锚 SR 协议由调用方声明（前向兼容扩展）。
- **A2 episode 级预算重分配：移出本阶段范围（G1 R1 裁决）**。理由：C8/C11 下按 task 先验重分配 skip 预算会跳命中步 → 反事实 → 离线 SR 不可靠，离线只能给 (inf_ratio, 真 MISS-skip 覆盖) 曲线 + oracle 上界而非 SR 判决，决策价值低于 2a/2b/2c 核心且徒增 scope。降入 Stage 4 押后（如需可另起离线/live 专项）；roadmap §5 2b "A2 并入此处" 于 Phase C 反向调和。
- **及格线**：frontier overlay 图 + "gate 是否构成第四设计轴（keybuilder/judge/search 之外）"判决。

### 3.4 Line 2c — 危险步离线 join（可选，scope=libero_spatial）

**新文件 `exp/gate_research/stage2c_danger_join.py`**（CLI：`--deviate <deviate_score_spatial16_w8_d4.json> --gt-dir <gt/> --gate-rows <libero_spatial/gate_rows.jsonl> [--gate-yaml <yaml_id>] [--early-frac 0.5] --out analysis/stage2_danger_step.md`；keybuilder 由 `--deviate` 文件名推断，默认取 `spatial16_w8_d4`——deviate 三配置中与 gate `cp1_spatial_pool_16` 最近似）：

- 读 GT h5 attr 建 `task_{TID}/episode_{EP}` → `(task_id, orig_init_state_idx)` 映射（复用 trajectory_deviation 的 `_gt_episode_failed_unit` 同款读法），把 `deviate_score[t]≥5` 展成危险步标签。
- join 到 `libero_spatial/gate_rows.jsonl`，键 `(task_id, orig_init_state_idx, step_idx=5·t)`（inner join，仅 GT 成功 ep 覆盖的 init）。
- 算 prev_score / prev_is_MISS / cp1_score / step_idx 等廉价信号对危险步的 AUC（复用 `stage2_common.auc`）。
- **默认 keybuilder = `spatial16_w8_d4`**（deviate 三配置中与 gate `cp1_spatial_pool_16` 最近似）。
- **及格线**：若危险步可由 prev_score/其他信号预测（AUC 显著 >0.5）→ N4 注入可做危险步靶向的证据；否则记录"危险步不可廉价预测"。

### 3.5 输出与文档

- 三份 .md：`exp/gate_research/analysis/stage2_v2_mechanism.md`、`stage2_fair_pareto.md`、`stage2_danger_step.md`（实验报告归 `analysis/` 纯 .md，符合项目惯例）。
- 图：`stage2_pareto_spatial.{png,pdf}`、`stage2_pareto_l10.{png,pdf}` 于 `analysis/`。
- **回填**：分析完成后把 §5 Stage 2 各行"产出/及格线"补上判决（改 `logs/gate_exploration_roadmap.log.md`），订正 roadmap "185,899 决策步" → 实测 182,899（39,136+143,763），并把 §5 2b "A2 预算重分配并入此处" 改为"A2 降 Stage 4 押后"（本 plan G1 R1 裁决）。
- **index sync（WA §4 红线）**：本 plan 新增 → 同 commit 更新 `logs/README.md`。

---

## 4. 涉及文件清单

**新增**：
- `logs/gate_stage2_v2_mechanism.log.md`（本 plan）
- `exp/gate_research/stage2_common.py`
- `exp/gate_research/stage2a_sr_decomp.py`
- `exp/gate_research/stage2b_pareto_overlay.py`
- `exp/gate_research/stage2c_danger_join.py`
- `exp/gate_research/analysis/stage2_v2_mechanism.md` / `stage2_fair_pareto.md` / `stage2_danger_step.md` + 两图
- `tests/exp/test_stage2_common.py`、`test_stage2a_decomp.py`、`test_stage2b_overlay.py`、`test_stage2c_join.py`

**修改**（仅文档，分析完成后）：
- `logs/gate_exploration_roadmap.log.md`（回填 §5 Stage 2 判决 + 订正步数）
- `logs/README.md`（新增本 plan 索引行）

**不改**：`analyze_n1_live.py`、`gate_structure_analysis.py`、任何 src、任何冻结数据（只读）。

---

## 5. 集成点

- 三个 stage2 脚本 import `stage2_common` 与 `analyze_n1_live` 的纯函数（无 src 依赖、无推理路径依赖）。
- 只读 §1 表所列数据文件；不写任何数据目录（仅写 `analysis/` 报告与图）。
- 与推理管线完全解耦（纯离线后处理），符合 WA §2.5。

---

## 6. 测试策略（WA §6：新组件必带测试；tests/exp/，默认 non-manual 小 fixture）

- `test_stage2_common`：(1) `cache_run_lengths` 对合成 searched/hit_type 序列返回已知 run 长度；(2) `reconstruct_searched` 复现 periodic k/n 模式；(3) `action_source_seq` 对 skip 步判 NEW_INFER、FULL_HIT 判 CACHE_FH、WS 单列；(4) `auc` 对可分 fixture ≈1.0、随机 ≈0.5。
- `test_stage2a_decomp`：Δinf 三分解在小 fixture 上三项之和 == 直接算的总 Δinf（代数恒等）；`mcnemar()` 已知 2×2 → 已知 chi2；`mcnemar_exact_p` 确定性值（b=0,c=5 → 2·0.5⁵=0.0625；b=c=3 → 1.0；b=1,c=8 → 2·(C(9,0)+C(9,1))·0.5⁹≈0.0391）。
- `test_stage2b_overlay`：承重/参照分层正确——RPG 行被过滤到 spatial-only 且标 `derived`；l10 overlay 数据集不含任何 RPG 点；协议匹配锚点数值等于 §1 表。
- `test_stage2c_join`：合成 deviate_score JSON + GT attr + gate_rows → join 命中 `(task_id,orig_init,step_idx=5t)`；缺键行被 inner-join 丢弃；AUC 在可分标签上 ≈1.0。
- 完整性不变量（镜像 `verify_gate.py` L42-52）：join/分段前断言无重复 `(task_uid,step_idx)`、robot_state 存在。
- **§6 Verify**：`uv run pytest`（新测试 + 全量 non-manual）全绿方可 commit；读大 jsonl 的集成测试标 `@pytest.mark.manual`。

---

## 7. 风险登记

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | **2c libero_10 离线不可做**（无 deviate_score 无 GT，需新 GPU 采集） | 2c 只能覆盖 libero_spatial | scope 明确锁 spatial；l10 危险步作为 GPU 采集决策押后，plan 内不做（保持 Stage 2 全离线） |
| R2 | **2c 跨配置迁移**：deviate 用 clip/spatial16/max_pool，gate 用 cp1_spatial_pool_16 fh/ws | 2c 是 proxy 非定论 | 取最近似 `spatial16_w8_d4`；报告标注"跨配置迁移假设"，结论定性为 suggestive |
| R3 | **2c 轨迹发散**：deviate 是 GT 开环重放，gate_rows 是 live 闭环 gated rollout | step 对齐早期精确、发散后近似 | join 报告按 episode 相位分段，标注对齐衰减；必要时限早期步 |
| R4 | `gate_structure_analysis.py` import 有副作用 | 不能复用其 auc | 共享层重写 auc（6 行），不 import、不改该脚本 |
| R5 | RPG/threshold_pareto 异协议 | 若误入承重层会污染 F11 判决 | 硬规则：仅同协议点承重，RPG 挖空背景 + legend caveat，threshold_pareto 不上同轴 |
| R6 | roadmap "185,899 步" 与实测 182,899 不符 | 文档数字错误 | 回填时订正为 182,899（39,136+143,763） |
| R7 | H1/H2/H3 三假设全不成立 | 2a 无靶向证据 | fallback：N4 用 uniform L（periodic 已证有效），2a 只影响靶向不阻塞 Stage 3 |
| R8 | WARM_START 动作来源歧义（部分去噪回放算缓存还是推理） | 影响 run-length 口径 | `cache_run_lengths(include_ws)` 双口径敏感性；H3 单独处理 WS |

---

## 8. 执行拓扑（Code 阶段，post-G1）

- **Phase A**：`stage2_common.py` + 其单测（建一次，2a/2b/2c 共用）。
- **Phase B（3 线并行）**：2a / 2b / 2c 各自脚本 + .md + 图 + 单测，互不依赖。
- **Phase C**：回填 roadmap §5 Stage 2 判决 + `logs/README.md` 同步。
- 单 commit 交付（owner 偏好 structured single commit）。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-07-05 05:48 CDT

审查范围：按 G2 口径审查 §4 Code 完成后的实现、测试、正式产物与文档回填；未跑全量 pytest、GPU test 或 manual test。

验证结果：
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check exp/gate_research/stage2_common.py exp/gate_research/stage2a_sr_decomp.py exp/gate_research/stage2b_pareto_overlay.py exp/gate_research/stage2c_danger_join.py tests/exp/test_stage2_common.py tests/exp/test_stage2a_decomp.py tests/exp/test_stage2b_overlay.py tests/exp/test_stage2c_join.py` — PASS。
- `PYTHONPATH=. python -m py_compile exp/gate_research/stage2_common.py exp/gate_research/stage2a_sr_decomp.py exp/gate_research/stage2b_pareto_overlay.py exp/gate_research/stage2c_danger_join.py` — PASS。
- `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/exp/test_stage2_common.py tests/exp/test_stage2a_decomp.py tests/exp/test_stage2b_overlay.py tests/exp/test_stage2c_join.py -q` — PASS，34 passed。
- Reviewer 独立 CLI smoke（输出限定在 `tests/review_tests/stage2_smoke/`）：`stage2a_sr_decomp` 用真实 spatial N1+periodic manifest PASS；`stage2b_pareto_overlay` 用真实 Stage-0/N1/RPG 输入 PASS（仅 matplotlib/font cache warning）；`stage2c_danger_join` 用真实 deviate+GT+gate_rows PASS（join 3254、danger 199）。

Blocking findings：
1. **正式 analysis 产物未落盘**。本 plan §3.5/§4 把 `exp/gate_research/analysis/stage2_v2_mechanism.md`、`stage2_fair_pareto.md`、`stage2_danger_step.md`、`stage2_pareto_spatial.{png,pdf}`、`stage2_pareto_l10.{png,pdf}` 列为目标产物；当前 `exp/gate_research/analysis/` 下没有任何 `stage2*` 文件。Reviewer smoke 只能证明 CLI 可写，不能替代正式交付文件。
2. **roadmap Phase C 回填未完成**。`logs/gate_exploration_roadmap.log.md` 仍保留 `185,899 决策步`，未按 §3.5 订正为 `182,899`；Stage 2 表仍写 `A2 预算重分配并入此处离线合成`，未按 G1 R1 裁决改为 Stage 4 押后；§5 Stage 2 的 2a/2b/2c 也未回填本轮机制判决/输出路径。
3. **index/status sync 未完成**。本 plan 头部仍是 `Status: Plan（G1 APPROVED 2026-07-05；待 §4 Code）`；`logs/README.md` 本 plan 行仍是 `Plan (G1 Round 1 NEEDS REVISION → Executor R1 applied, 待重审 2026-07-05)`。这与“代码编写完毕、进入 G2”状态矛盾，也违反 §3.5 的 index sync 要求。
4. **2a 配对统计缺少全对照 fail-fast**。§2 明确复用 `mcnemar(..., require_equal=True)`，§3.2 要求“全对照配对”；但 `stage2a_sr_decomp.py::paired_sr()` 当前调用 `mcnemar(cond_map, base_map, require_equal=False)`，per-task 切片也静默取交集。若某 run 缺 episode 或 baseline/run unit 不一致，报告会在缩小后的 shared units 上继续产出 ΔSR/p 值，而不是暴露数据完整性问题。请改为 top-level full-pairing fail-fast（或显式校验 set equality 并报告缺失），并补一个 mismatched-unit 测试；per-task 如确需交集，需在报告中暴露 `n_paired`/missing count，不能静默。

Non-blocking concern：
- `stage2a_sr_decomp.py` 的实际 CLI 通过 manifest 内的 baseline path 工作，未实现 §3.2 文档写的 `--stage0 <spatial_rows> <l10_rows>` 参数。真实 smoke 已通过，不单独阻塞；但需要在正式报告/plan 状态说明中消除 CLI 合同不一致，或实现兼容 flag。

结论：NEEDS REVISION。修复以上 blocking 后重新提交 G2；无需跑全量/GPU/manual，保持当前 scoped 测试 + 正式 CLI 产物生成即可。

### G2 Round 1 — Executor — 2026-07-05 CDT

- Accepted — [B1 正式产物落盘] 用修好的代码生成并落盘 `exp/gate_research/analysis/stage2_{v2_mechanism,fair_pareto,danger_step}.md` + `stage2_pareto_{spatial,l10}.{png,pdf}`；2b 以 `--extra-anchor libero_10:1.0:83:pure_inf` 使 l10 可判（默认关闭保持 plan 原样）。
- Accepted — [B2 roadmap Phase C 回填] roadmap 步数 185,899→**182,899**（分解 39,136/143,763）；§5 2b "A2 并入此处"→**A2 降 Stage 4 押后**（G1 R1 理由）；§5 Stage 2 表后新增**判决回填块**（2a H1 达标→N4 L≈6–8；2b gate=第四设计轴，periodic 抬前沿 +6.8/+3.6/+3.0pp；2c 危险步不可廉价预测 AUC~0.52→**否决危险步靶向**）+ 对 Stage 3 净指令；顶部 Status → Stage 2 ✅。README roadmap 行步数同订正。
- Accepted — [B3 status/index sync] plan header Status 与 `logs/README.md` 本 plan 行均同步至"G1 APPROVED R2 / §4 Code 完成 + 产物落盘 / G2 R1 → Executor R1 applied 待重审"。
- Accepted — [B4 配对 fail-fast] `paired_sr` 改 `require_equal=True`（unit 集不等即 ValueError）；`per_task_delta_sr` 加 task-set 相等校验 + 每 task `require_equal=True`，`n` 照报；新增两个 mismatch fail-fast 测试。真实 8-run require_equal=True 复核**无误报**（unit 集确实相等，数值不变）。
- Accepted — [Non-blocking CLI 合同] plan §3.2/§3.3/§3.4 CLI 示例改为与实现一致：2a 从 manifest 内 `baseline_gate_rows_path`/`baseline_yaml_id` 读取（删 `--stage0`）；2b 补全实参并记入 `--extra-anchor`（G2 R1 新增）；2c 补全实参（keybuilder 由 `--deviate` 文件名推断）。

### G2 Round 2 — Reviewer — APPROVED — 2026-07-05 06:32 CDT

复审范围：仅复核 G2 R1 四个 blocking finding 与相关回归风险；未跑全量 pytest、GPU test 或 manual test。

验证结果：
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check exp/gate_research/stage2_common.py exp/gate_research/stage2a_sr_decomp.py exp/gate_research/stage2b_pareto_overlay.py exp/gate_research/stage2c_danger_join.py tests/exp/test_stage2_common.py tests/exp/test_stage2a_decomp.py tests/exp/test_stage2b_overlay.py tests/exp/test_stage2c_join.py` — PASS。
- `PYTHONPATH=. python -m py_compile exp/gate_research/stage2_common.py exp/gate_research/stage2a_sr_decomp.py exp/gate_research/stage2b_pareto_overlay.py exp/gate_research/stage2c_danger_join.py` — PASS。
- `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/exp/test_stage2_common.py tests/exp/test_stage2a_decomp.py tests/exp/test_stage2b_overlay.py tests/exp/test_stage2c_join.py -q` — PASS，36 passed。
- Reviewer 复生成真实数据 CLI 产物到 `tests/review_tests/stage2_r2/`：2a PASS（4 groups），2b PASS（含 `--extra-anchor libero_10:1.0:83:pure_inf`，仅 fontconfig cache warning），2c PASS（join 3254、danger 199）。2a/2c markdown 与正式 `analysis/` 产物完全一致；2b 用正式 manifest 顺序复生成后 markdown 完全一致（不同 manifest 顺序只影响行顺序，不影响数值）。

Blocking closure：
1. B1 关闭：`exp/gate_research/analysis/stage2_v2_mechanism.md`、`stage2_fair_pareto.md`、`stage2_danger_step.md`、`stage2_pareto_spatial.{png,pdf}`、`stage2_pareto_l10.{png,pdf}` 已落盘，且可由当前 CLI 复生成。
2. B2 关闭：roadmap 顶部步数已订正为 182,899；§5 Stage 2 已回填 2a/2b/2c 判决、A2 降 Stage 4、以及 Stage 3 净指令。
3. B3 关闭：plan header 与 `logs/README.md` 已同步到 §4 Code 完成 / G2 R1 Executor applied 待重审状态；本条 Reviewer APPROVED 是最终 G2 裁决。
4. B4 关闭：`paired_sr` 与 per-task 切片均改为 `require_equal=True`/显式 task-set 校验，并新增 mismatch fail-fast 测试；scoped pytest 覆盖通过。

Non-blocking note：
- `logs/gate_exploration_roadmap.log.md` 早期 A2 处置段仍保留部分 Stage-2 历史表述，但 §5 的 Stage 2 回填与 Stage 3 净指令已明确覆盖，不影响本 G2 放行；后续整理路线图时可顺手统一。

结论：APPROVED。可进入 commit/后续 Stage 3。
