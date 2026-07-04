# Plan — CP1 六段延迟 ~ trajectory_depth 受控扫描（libero_10, 3-key）

> **Authority**: Execution · **Level**: L2 · **Status**: Plan / **G1 APPROVED R2** (2026-05-31) / §4 Code
> **复用基础设施**: `exp/cache_latency_bench/`（已 G2 APPROVED, commit `90d31b0`）。本研究**零 src 改动、零 infra 改动**，只新增 4 个派生 yaml + 1 个跨-depth 汇总脚本 + 测试 + 研究 log。
> **前序 plan**: [`cache_latency_bench_plan.log.md`](cache_latency_bench_plan.log.md)（基础设施本体）。

---

## §1 研究问题与假设

**问题**：在固定其余一切的前提下，把 `trajectory_depth` 从 1→3→4→5 调大，CP1 `check()` 的六段延迟（collect / gate / build / search / judge / fetch）各自如何变化？

**主假设 H1**：只有 **search 段**随 depth 单调上升（其余五段与 depth 无关）。
依据（亲验代码）：
- `search` 走 `_search_with_trajectory`，`in_memory_backend.py:693` `for layer_idx in range(L)` 逐层融合，`:671` `L = min(len(history), len(weights))` → 层数 = 有效 depth，**线性 scaling**。
- `collect/gate/build/judge` 不读 trajectory：build 由 `key_builder.build()` 决定（只与 key-set 有关）；judge 是 `judge.py:231` 单次 `top.score >= threshold` 标量比较；gate=`always_search` 恒真；collect 是 stage1 输出装配。
- `fetch` 仅与 hit_type 有关（`orchestrator` 只在 FULL_HIT/WARM_START measure fetch），与 depth 无直接关系。

**H1 即本研究的可证伪内部一致性验收点**（§8）：若 build/gate/judge/collect 也随 depth 显著漂移，说明有泄漏或测量污染，需排查。

---

## §2 实验设计

### 自变量（唯一）
`trajectory_depth ∈ {1, 3, 4, 5}` —— 4 个派生 yaml，**仅** `search_strategy.trajectory_depth` + `trajectory_weights` 不同。

### 受控变量（4 个 yaml 完全一致）
取自现有 3-key 模板 `…/by_base/d4/…__d4__fh10_ws10_quantile.yaml`：
- key-set = **3-key**：`vision_0(0.25)` + `vision_1(0.4375)` + `robot_state(0.3125)`；`vision_2/prompt_emb` disabled。
- `key_builder: cp1_spatial_pool_16`；`gate: always_search`；`top_k:1`；`step_filter: all`。
- `judge: threshold`，**固定** `threshold: 0.997697` + `warm_tiers:[{threshold:0.997403,start_t:0.5}]`。
- `field_similarity`（vision_0/1 cosine、robot_state l2+exp/tau1）+ `score_normalization`（per_field zscore，模板原值）。
- `backend: in_memory` + `preload_path: …/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl` + `brute_force`。
- `write_policy: {type: never}`（harness C2 契约）。
- H5 = `exp/common/data/db/libero_cache/libero_10`（**50 episode 全量**）；`repeats: 1`；`device: cpu`。
- **线程固定**：`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`（4 run 一致），summary 记录 `torch.get_num_threads()`。

`trajectory_weights`（线性递减归一 `[k,k-1,…,1]/Σ`，与模板 d4 同规则；**延迟无关**，仅为规范/与现有一致，因 `:620` weight 只是乘法系数）：
- d1：省略（`trajectory_depth:1` 时 config 允许 weights=None，见 `config.py:1581` 仅 depth>1 才校验）。
- d3：`[0.5, 0.3333, 0.1667]`
- d4：`[0.4, 0.3, 0.2, 0.1]`（= 模板原值）
- d5：`[0.3333, 0.2667, 0.2, 0.1333, 0.0667]`

### 因变量
每 step 的 CP1 六段延迟（ms），由 `replay.py` 经 orchestrator 自带 `SystemTimer` 探针采集，落 `per_step.csv`，按 `hit_type` 分桶统计 median/p50/p95（`summarize.py` 既有）。

---

## §3 亲验的 src 行为（关键陷阱，必须写进报告）

| # | 事实 | file:line | 对研究的影响 |
|---|---|---|---|
| T1 | **depth=1 与 depth≥2 走 backend 两条不同函数** | `search_strategy.py:155` `if self._trajectory_depth <= 1 … return {}` → QuerySpec 无 trajectory 字段 → `in_memory_backend.py:299` 分支不成立 → 走 `:330 _search_weighted_score_sum`（单步）；depth≥2 且 history 够 → `:311 _search_with_trajectory` | **d1 是"单步基线"，不是"1 层 trajectory"**。曲线上 d1→d3 之间有函数级 break，报告必须标注，不能当作连续 depth 效应外推。 |
| T2 | **episode 内 depth 爬升期** | `search_strategy.py:158` `actual_depth = min(self._trajectory_depth, len(self._query_history))`；`in_memory_backend.py:671` `L = min(len(history), len(weights))` | 每 episode 前 `depth-1` 步有效层数未满（1,2,…,depth,depth,…）。**稳态延迟才反映满 depth**。→ §6 汇总额外输出 steady-state（`step_idx >= 4`）分桶，与全量并列。 |
| T3 | **score-memo 跨层/跨步复用** | `in_memory_backend.py:659-663`（session+query_id keyed raw 相似度复用） | 稳态 search 延迟 < naive `depth × 单层`；这是真实 server 行为，**保留**（faithful），但报告说明 search 标度非严格线性。 |
| T4 | threshold 数值不改任一段计算量，仅定 verdict | `judge.py:231` | 固定 threshold 合法；不同 depth 下 score 分布变 → hit 率随 depth 变（depth 真实效应，非混淆），**按 hit_type 分桶**规避对 fetch 段统计的干扰。 |
| T5 | weight 数值=纯乘法、active key 数=循环次数 | `in_memory_backend.py:500` `if w <= 0: continue`、`:620` `final_scores += weight*s*mask` | 4 yaml key-set 完全相同 → build/search 字段循环数恒定，**不引入混淆**。 |
| T6 | config 对 trajectory 的校验 | `config.py:1576-1601`：depth≥1；depth>1 时 weights 必填、`len==depth`、非负、Σ>0 | 派生 yaml 须满足，否则 `ReplayHarness.__init__` 的 `validate_cache_config` fail-fast。 |

---

## §4 产物清单（新增文件）

1. `exp/cache_latency_bench/config/depth_study/depth_{1,3,4,5}.yaml` — 4 个派生 yaml（§2 规格）。
2. `exp/cache_latency_bench/compare_depth.py` — 跨-depth 汇总脚本（§6）。
3. `tests/exp/test_compare_depth.py` — 汇总脚本单元测试（合成 csv，CPU，无模型）。断言覆盖：(a) `ALL` 桶 vs steady-state（`step_idx ≥ 4`）双口径聚合；(b) `hit_counts` 跨 run 正确传播；(c) **no-fetch 行的缺段桶**——MISS-only run 无 `cp1_fetch_ms` 样本时该桶缺失不崩、空桶被跳过；(d) `--runs label=path` 解析，缺 `=`/空 label/路径不存在 fail-fast。
4. `logs/cache_latency_bench_depth_study.log.md` — 研究 log（方法/受控变量/结果表/T1-T3 陷阱标注/H1 结论）。
5. 运行产物（gitignore 的 `data/` 下，不提交）：`exp/cache_latency_bench/data/depth_study/d{1,3,4,5}/{per_step.csv,summary.json}`。

**索引同步（WA §4）**：① 本 plan 文件 `cache_latency_bench_depth_study_plan.log.md` 已在 `logs/README.md` Cache System 表补 active-log 条目（与 plan 同 worktree，满足 index-sync 红线）；② 研究 log `cache_latency_bench_depth_study.log.md` 创建时在**同一 commit** 补 `logs/README.md` 条目；③ 若新增 `docs/` 页则同步 `docs/README.md`。

---

## §5 派生 yaml 生成方式

以模板 d4 yaml 为基底**逐字复制**，仅替换 `search_strategy.trajectory_depth` 与 `trajectory_weights` 两键（d1 删除 `trajectory_weights`）。**不脚本生成、手写 4 份并 diff 校验**（避免生成器引入意外漂移；4 份之间除这两键外 `diff` 必须为空）。每份跑前由 harness `validate_cache_config` 把关。

---

## §6 汇总脚本 `compare_depth.py` 设计

- 输入：4 个 run 的 `per_step.csv`（`--runs d1=path d3=path d4=path d5=path`）。
- 复用 `summarize.summarize()`（既有）逐 run 出分桶统计；**额外**对每 run 计算 steady-state 子集（`step_idx >= 4`）的同口径统计（应对 T2）。
- 输出 `comparison.json` + stdout markdown 表：行=六段，列=depth(1/3/4/5)，值= `ALL` 桶 median(p95)，并另出 steady-state 表 + 各 depth 的 `hit_counts`。
- 纯 stdlib + numpy（与 `summarize.py` 同栈），无新依赖。
- `--runs` label 解析（`label=path` 拆分）显式校验：缺 `=`、空 label、路径不存在均 fail-fast（由 §4 测试覆盖）。

---

## §7 已知偏差（沿用基础设施，报告复述）

1. CPU 模式 `cp1_build` 不含 GPU→CPU D2H（基础设施偏差1）。
2. 仅驱动 CP1（偏差2）。
3. 绝对毫秒与 CPU 负载相关；结论限于**段间相对占比 / search 随 depth 的标度 / 稳态 vs 爬升**，非跨机绝对值（偏差4）。开视频不影响（已固定线程 + 分桶）。
4. T1（d1 函数分叉）、T2（爬升期）、T3（memo）三条本研究特有陷阱。

---

## §8 验收标准

1. 4 个 yaml 通过 `validate_cache_config`（harness 启动不 raise）。
2. 4 个 run 各产出非空 `per_step.csv`，覆盖 50 episode 全部回放，`summary.json.hit_counts` 含 ≥1 个 FULL_HIT 桶（确保 fetch 段有样本）。
3. `compare_depth.py` 跑通，输出六段×depth 表 + steady-state 表。
4. **H1 sanity check**：collect/gate/build/judge 四段的 median 跨 depth 漂移 < search 段漂移的一个量级（否则报告须解释泄漏来源）；search 段稳态 median 随 depth **单调不降，允许相对容差 ε=5%**（host 噪声）。若某相邻 depth 出现 > ε 的逆序，**不直接判失败**——先对该两点把 `repeats` 加大（如 3）重跑 / 诊断 host 负载，复核后再判定。
5. `test_compare_depth.py` 全绿；`ruff check` + `ruff format` 干净。
6. 研究 log 含 T1-T3 标注与 H1 结论。

---

## §9 执行步骤（G1 通过后）

1. 手写 4 个派生 yaml + 互 diff 校验（仅两键差异）。
2. 写 `compare_depth.py` + `test_compare_depth.py`，先跑测试。
3. `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` 依次跑 4 个 `run.py`（前台、单进程；CPU-only 不触发后台审批）。
4. `compare_depth.py` 汇总 → 写研究 log（结果表 + H1 结论 + T1-T3）。
5. Verify（全量 `ruff` + 相关测试）→ G2 → 按 owner 指示提交。
