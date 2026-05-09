# Verdict Phase 4 — Phase 3 Top Recipe 权重扫描

**状态**：`In Progress` — G1 APPROVED Round 3 (2026-05-08); §4 Code 待启动
**等级**：L2 — 多文件特性（**含 src/ 改动**：composer 真权重语义 + solver weight passthrough；exp/ spec + runner + 三轮权重生成器）；按 WA §2.1 走 G1 / G2 双门
**职权**：Execution
**负责人**：LinZiyang666
**日期**：2026-05-08
**关联**：
  - 前置 plan：[`logs/verdict_phase3_threshold_sweep.log.md`](verdict_phase3_threshold_sweep.log.md)
  - 前置数据：`exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl` — 176 cells，已锁定 recipe + threshold winner
  - 前置分析：[`exp/verdict_factor_judge/analysis/phase3_results.md`](../exp/verdict_factor_judge/analysis/phase3_results.md) — recipe 排名

---

## §0 背景

### §0.1 Gate 政策

L2 任务 — G1（plan 审）+ G2（code 审）双门生效，依 WA §2.1 / `protocols/execution_authority.md` §10。

**含 src/ 改动**：phase 3 的 `WeightedSumZeroNanComposer._score_only` 当前实现是**等权平均**（只用 `w != 0` 决定 declared dependencies，无 `w * contrib` 项；solver `reconstruct_scores` 同样 hardcode `{k: 1.0}`）。这让 phase 4 的连续 α / heavy 权重扫描在现有代码下退化为 zero/nonzero ablation。R1 [Blocking 1] 命中此点；扩 scope 在 src/ 加入真加权语义。

Gate 覆盖的 src/ 改动：
- `src/openpi/cache/components/factors/composers/__init__.py` — `WeightedSumZeroNanComposer._score_only` 从等权平均改为 `Σ(w_k · contrib_k) / Σ(w_k)`，零权 key 跳过；NaN raw 仍贡献 0 但保留分母权重（与 zero-NaN 语义一致）。`compose()` 路径不变（仍调 `_score_only`）。**同步更新该类 docstring**（line 282 起，"Equal-weight sum, NaN -> 0 (still counted), two-tier thresholds." 改成 "Weighted sum (Σ w_k · contrib_k / Σ w_k) with NaN -> 0 (still in denominator), two-tier thresholds."）+ `_score_only` 的内嵌 docstring（line 337-343）+ 任何引用旧"equal-weight"的 module / 类注释（reviewer Suggestion 7）。

Gate 覆盖的 exp/ 改动（**phase3_spec.py 不再改动** — G1 R2 Blocking 2 修订：phase4 直接调 `v2_spec.build_eval_yaml(cfg_id, factors, composer)` 与 `v2_spec.build_warmup_yaml(cfg_id, eval_yaml_id, eval_factors=...)`；这两个签名接完整 composer dict / factors list，与 RECIPES_PHASE4 全局无依赖；phase4 在自己进程内构造 composer dict，`phase3_spec.RECIPES` 不被 mutate）：
- `exp/verdict_factor_judge/phase3_threshold_solver.py` — `reconstruct_scores(jsonl_path, recipe, *, composer_weights=None)` 加 keyword-only 参数；None 时维持旧 hardcode `{k: 1.0}` 行为（向后兼容 phase 3）；phase 4 总是显式传
- `exp/verdict_factor_judge/phase4_spec.py` — 2 融合 recipe + 各轮权重生成器 + cell id helper + composer-dict / yaml builder（直接调 v2_spec）
- `exp/verdict_factor_judge/run_phase4.py` — 镜像 run_phase3，含 `preload_normalizer_buffer(eval_yaml_id, buffer)` → `load_cache_config(eval_yaml)` 顺序（phase 3 line 380-382 invariant）；**4-mode CLI**（`emit-warmup-yaml` / `run-warmup` / `emit-eval-yamls` / `run-eval`）拆开"无 server 文件操作 vs 要 server 的 episode 跑"两类工作（修订 G1 R2 Blocking 3）
- `exp/verdict_factor_judge/analysis/plot_alpha_sweep_phase4.py` — α 曲线
- `exp/verdict_factor_judge/analysis/plot_pareto_phase4.py` — phase3 + phase4 同图叠加
- `exp/verdict_factor_judge/analysis/phase4_results.md` — 每跑完一轮追加一节

Gate 覆盖的 tests（按 WA §3.1，**修正路径** — G1 R2 Concern 6）：
- `tests/cache/components/factors/test_composer_zero_nan.py` — **既有文件**（fact-checked，不是 plan R1 误写的 `test_weighted_sum_zero_nan_composer.py`），加 numeric weight sensitivity 测试（同 declared key 集，不同非零权值产生不同 score）；旧"等权"测试改名为 `test_score_only_backward_compat_uniform_weights_match_phase3_behavior`（锁住 weights 全等 → 输出与 phase3 一致的回归不变量）
- `tests/exp/test_phase3_threshold_solver.py` — 加 `composer_weights` passthrough 测试（同 history、不同权重 → 不同 score 序列；None 保持旧行为）
- `tests/exp/test_phase4_spec.py` — recipe 形状 / 权重生成器不变量 / numeric weight sensitivity 整链路（不同 R2 pattern → 不同 weight dict → 不同 solver score）/ **phase3 RECIPES 不变性**（"phase4 import 后 `phase3_spec.RECIPES` 仍是 11 个 key" — 防止全局污染回归）
- `tests/exp/test_phase4_runner.py` — 4-mode CLI dispatch / per-recipe winner CLI 解析 / preload-before-load 顺序 / resume 逻辑 / **round-specific decision_gate 选 winner 与 §3 gate 规则一致**（R1 用 SR−0.5·inf，R2/R3 用"SR 最大且 > uniform.SR + 2pp，否则强制 uniform"）

### §0.2 实验目标

Phase 3 锁定了 recipe 级 winner（g1 / g10 W-FUT × 全 4 desc，16 cell SR_min ≥ 0.90）和次优（g6 online action jerk+disp）。**Phase 4 把 g1+g6、g10+g6 融合成 10-factor recipe，探索加权融合是否能突破任一单 recipe**。

(FH, WS) cell **锁定 per-recipe 的 ultra-cheap anchor**（phase3 实测，per_yaml_summary.jsonl）：

| recipe | locked (FH, WS) | anchor SR | anchor inf |
|---|:---:|:---:|:---:|
| **p1** = g1 + g6 | **(0.5, 0.5)** | 0.950 | 0.372 |
| **p2** = g10 + g6 | **(0.5, 0.4)** | 0.960 | 0.382 |

> **修订理由（G1 R1 Blocking 4）**：原 plan 把 p2 也锁 (0.5, 0.5)，但 phase3 g10 (0.5, 0.5) 实际 SR=0.93 不是 0.96（0.96 是 g10 的 (0.5, 0.4) cell）。继续锁 (0.5, 0.5) 会让 p2 的 α=1 端点显著低于 plan 声称的 anchor，G_R1 sanity 误判融合失败。改成 per-recipe locked cell 是最干净的修法 — 两个 recipe 都对齐到自己的 ultra-cheap anchor。

每 recipe 的唯一变量是 weight 向量。所有轮次的 cell 数（§2）保持不变 — locked cell 改成 per-recipe 后，p1 的所有 cell 在 (0.5, 0.5)，p2 的所有 cell 在 (0.5, 0.4)。

### §0.3 设计依据

**为什么融合**：phase3 g1/g10 全部是 offline，g6 全部是 online。两条信号路径**计算源完全不重叠**（cache pkl 离线静态字段 vs runtime history+chain），所以**误差应该部分独立**。线性加权融合可在不丢任一侧强信号的前提下抑制对方的高噪声模式。如果 α 响应曲线呈非单调（α=0.5 同时优于 α=0 和 α=1），就证实独立性在数据上确实成立。

**为什么 per-recipe 锁 ultra-cheap anchor cell**：
1. Phase 3 已绘出响应面 — p1 在 (0.5, 0.5) SR=0.95 / inf=0.37、p2 在 (0.5, 0.4) SR=0.96 / inf=0.38，都是各自 recipe 的最便宜 winner；
2. Threshold 灵敏度已查清，weight 灵敏度才是下一未知量；
3. 锁 cell 消除一个交互项，α 效应不被 threshold 漂移污染。
4. 两 recipe locked cell 不同**不影响**实验同质性 — 我们对比的是"同一 recipe 内 weight 变化"的 SR/inf 响应，不是 p1 vs p2 直接对比。

**为什么用 grid（不用 Bayesian opt）**：phase 3 测得 100 ep × 1 seed 的 SR 噪声 ±3pp，SNR ≈ 1，让 GP-based acquisition function 在追噪声（见 2026-05-08 用户讨论）。每 cell 成本不贵（~5 min × 6 server 并跑），grid 既可行又更可解释。Bayesian opt 留到 phase 5（1000 ep × 1 seed）若需高维精调时再用。

### §0.4 Scope (in / out)

**In**：
- 单 cfg：spatial16（`spatial16_w8_d4`）
- 2 个融合 recipe：
  - **p1** = g1（offline_state W-FUT 4 desc × 2 win）+ g6（online_action W-K3 2 desc × 1 win）= **10 score**
  - **p2** = g10（offline_action W-FUT 4 desc × 2 win）+ g6（online_action W-K3 2 desc × 1 win）= **10 score**
- (FH, WS) **per-recipe 锁定**：p1 锁 (0.5, 0.5)、p2 锁 (0.5, 0.4) — 每 recipe 每 weight 配置只 1 cell
- Composer：`weighted_sum_zero_nan` **修改后**版本（真加权和；详见 §0.1 src/ 改动），warm_start_t=0.5（同 phase3）
- Threshold 求解：复用 `phase3_threshold_solver`，**每 recipe 一份 warmup yaml**（不是每 weight 配置一份 — 见 §2.5）
- 强制 3 轮（R1 α / R2 offline desc / R3 online desc）+ 1 条件轮（R4 W-FUT window）

**Out**：
- clip / max_pool cfg（等 phase4 spatial16 结果再说）
- 其他 (FH, WS) cell（若 α 效应有 FH/WS 依赖性，留到 phase 5）
- p1, p2 之外的 recipe（不做 g4/g9 衍生融合）
- weighted_sum_zero_nan 之外的 composer
- 跨 recipe weight 迁移实验（如把 p1 的 α\* 套到 p2 — 设计上隐含但不形式化测）

---

## §1 Recipe 定义

### §1.1 p1 — state offline + online action

| score id | factor type | window | source role |
|---|---|---|---|
| s1 | jerk_offline_state | (0, 3) | offline (g1 group) |
| s2 | jerk_offline_state | (0, 5) | offline |
| s3 | direction_offline_state | (0, 3) | offline |
| s4 | direction_offline_state | (0, 5) | offline |
| s5 | dispersion_offline_state | (0, 3) | offline (range:[0.3, 0.7]) |
| s6 | dispersion_offline_state | (0, 5) | offline (range:[0.3, 0.7]) |
| s7 | path_length_offline_state | (0, 3) | offline (high) |
| s8 | path_length_offline_state | (0, 5) | offline (high) |
| s9 | jerk_online_action | (3, 3) | online (g6 group) |
| s10 | dispersion_online_action | (3, 3) | online (range:[0.3, 0.7]) |

### §1.2 p2 — action offline + online action

| score id | factor type | window | source role |
|---|---|---|---|
| s1 | jerk_offline_action | (0, 3) | offline (g10 group) |
| s2 | jerk_offline_action | (0, 5) | offline |
| s3 | direction_offline_action | (0, 3) | offline |
| s4 | direction_offline_action | (0, 5) | offline |
| s5 | dispersion_offline_action | (0, 3) | offline (range:[0.3, 0.7]) |
| s6 | dispersion_offline_action | (0, 5) | offline |
| s7 | path_length_offline_action | (0, 3) | offline (high) |
| s8 | path_length_offline_action | (0, 5) | offline |
| s9 | jerk_online_action | (3, 3) | online (g6 group) |
| s10 | dispersion_online_action | (3, 3) | online (range:[0.3, 0.7]) |

> 注：s9, s10 在 p1 与 p2 中**完全相同**（都是 online_action）。两 recipe 唯一差异在 offline channel。

### §1.3 Pkl 前置条件

Phase 3 已经把 `exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl` 富化补全：8 个 offline_state + 8 个 offline_action key（W-FUT (0,3)+(0,5) 全覆盖）。Phase 4 **不需要新 pkl key**。Online factor (s9, s10) 推理时算，无 pkl 入口。前置检查在 §5.0：R1 启动前先验证 pkl 16 个 key 齐全。

---

## §2 轮次结构

### §2.1 Round 1 — α（offline / online 切分）

可调参数：
```
α ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}    # 7 点
```

权重分配：
```
offline 8 score:  weight = α / 8        （组内均匀）
online  2 score:  weight = (1-α) / 2    （组内均匀）
```

Sanity check：α=0.0 → offline 全清零（退化为 g6）；α=1.0 → online 全清零（退化为 g1 / g10）。两端点都有 phase 3 已知 SR/inf — 用作 pipeline 正确性 sanity。

**Cell 数**：2 recipe × 7 α × 1 (FH, WS) = **14 eval cell** + 2 warmup = 总 16。

**输出 artifact**：`data/phase4/r1_alpha/per_yaml_summary.jsonl` — 14 行。

### §2.2 Round 2 — offline 4-desc 相对权重（α\* 锁定）

前置：R1 跑完，G_R1 选出 α\*（§3.1）。

α\* 内部（online weight 冻结在 (1−α\*)/2 each），把 offline 组的 α\* 总额按 9 种 pattern 分给 4 desc。每 pattern 内每 desc 的两窗 (p0_f3, p0_f5) **等分该 desc 的份额**（窗内切分留到 R4）。

Pattern 列表（按 `(jerk, direction, dispersion, path_length)` 份额）：
| pattern | shares | 说明 |
|---|---|---|
| uniform | (1, 1, 1, 1) | R1 baseline（若 α\* 已在 R1 跑过则不重跑） |
| jerk-heavy | (2, 1, 1, 1) | jerk 加倍 |
| dir-heavy | (1, 2, 1, 1) | direction 加倍 |
| disp-heavy | (1, 1, 2, 1) | dispersion 加倍 |
| path-heavy | (1, 1, 1, 2) | path_length 加倍 |
| jerk-only | (1, 0, 0, 0) | desc 退化 ablation |
| dir-only | (0, 1, 0, 0) | |
| disp-only | (0, 0, 1, 0) | |
| path-only | (0, 0, 0, 1) | |

每 score 权重：`share_desc / sum(shares) * α* / 2`（除 2 因为每 desc 有 2 个窗，窗内均匀）。

**Cell 数**：2 recipe × 9 pattern = **18 eval cell**（无新 warmup，复用 R1 的 per-recipe warmup）。

**输出 artifact**：`data/phase4/r2_offline_desc/per_yaml_summary.jsonl`。

### §2.3 Round 3 — online 2-desc 相对权重（α\*、offline pattern\* 锁定）

前置：R1 + R2 跑完；α\*、offline-pattern\* 由 G_R1、G_R2 选出。

(1−α\*) 总在线预算分给 2 个 online desc：

| pattern | shares (jerk_on, disp_on) |
|---|---|
| uniform | (1, 1) |
| jerk-heavy | (2, 1) |
| disp-heavy | (1, 2) |
| jerk-only | (1, 0) |
| disp-only | (0, 1) |

**Cell 数**：2 recipe × 5 pattern = **10 eval cell**。

**输出 artifact**：`data/phase4/r3_online_desc/per_yaml_summary.jsonl`。

### §2.4 Round 4（条件触发）— W-FUT (0,3) vs (0,5) 切分

**仅当** 该 recipe 的 R3 winner.SR ≥ R3 baseline (uniform online) winner.SR + 2pp（详见 §3.3 G_R3 触发规则）才跑。**Per-recipe 单独判**：p1 / p2 各自满足才触发自己的 R4 5 cell；任一未满足则该 recipe 跳过 R4。

Pattern（在每 offline desc 的份额内切分窗）：
| pattern | shares (p0_f3, p0_f5) |
|---|---|
| uniform | (1, 1) |
| short-heavy | (2, 1) |
| long-heavy | (1, 2) |
| short-only | (1, 0) |
| long-only | (0, 1) |

**Cell 数**：2 recipe × 5 pattern = **10 eval cell**（条件触发）。

### §2.5 Warmup 不变性

**关键**：warmup yaml 只产 `factor_raw`，与 composer weight 完全无关。所以 **每 recipe 一份 warmup（p1.warmup, p2.warmup）就够覆盖所有轮 + 所有 weight pattern**：
- 启动 R1 之前先生成 + 跑 p1.warmup, p2.warmup（4-mode CLI 的 `emit-warmup-yaml` + `run-warmup` 两步，详见 §4.2.1 / §5）；
- 后续所有 eval yaml（R1–R4）都通过修订后的 `phase3_threshold_solver.reconstruct_scores(jsonl_path, recipe, *, composer_weights=...)` 复用同一份 `factor_raw` jsonl 算出 (fh_thr, ws_thr) — **target per-recipe 锁定**（p1: (0.5, 0.5)，p2: (0.5, 0.4)；详见 §0.2 anchor 表 / §4.2.4 `LOCKED_CELLS`）；
- **修订**（G1 R2 Blocking 1 残留）：原 §2.5 写 "solver 已支持任意 composer weight 向量（phase3 验证过）" 是 G1 R1 之前的错误描述 — phase3 solver 实际 hardcode `{k: 1.0}` 不接 weights 入参。phase 4 通过本 plan §0.1 / §4.2.6 列出的 solver 签名扩展显式注入 `composer_weights=cell.weights`。

这避免 7+9+5+5 = 26 次冗余 warmup。

---

## §3 决策门（轮间）

### §3.1 G_R1 — α\* 选择

**输入**：14 cell × (SR, inf, hwm%)。
**决策流程**：

1. 算 `score(α) = SR(α) − 0.5 · inf(α)`（Pareto-leaning 综合指标）
2. α\* = 7 个 α 点中 score 最大者（p1, p2 各算一次，**per-recipe**）
3. **延续规则**（per-recipe，p1 / p2 分别判）：
    - 若 best.SR(α\*) ≥ 该 recipe 的 anchor SR − 2pp → 该 recipe 进 R2
    - **per-recipe anchor**（与 §0.2 anchor 表对齐）：
        - p1 在 locked cell (0.5, 0.5) 的 anchor SR = **0.95**（g1 phase3 实测）
        - p2 在 locked cell (0.5, 0.4) 的 anchor SR = **0.96**（g10 phase3 实测）
    - 若 p1 或 p2 任一 fail → 该 recipe 单独 abort（不强制双 recipe 同时通过）
    - 若两 recipe 均 fail → 整个 phase 4 终止，保留 phase 3 winner

### §3.2 G_R2 — offline pattern 选择

**决策流程**：
1. pattern\* = 9 pattern 中 SR 最大者（per recipe）
2. **延续规则**：
    - 若 pattern\*.SR > uniform.SR + 2pp → 进 R3（offline 内部权重确实有差异）
    - 否则（offline desc 之间 ≤ 2pp 差） → **直接进 R3，pattern\* 用 uniform**（R2 仍作 ablation 价值，但不形成下游 gate）

### §3.3 G_R3 → R4（条件触发）

**决策流程**：
1. R3 跑完后重算 per-recipe 综合 winner（α\*, offline-pat\*, online-pat\*）
2. **R4 触发条件**（per-recipe 单独判，任一 recipe 触发就跑该 recipe 的 R4 5 cell）：
    - 若 combined-winner.SR ≥ R3 baseline (uniform online) winner.SR + 2pp → 跑 R4（窗级权重在该 recipe 上有显著边际收益空间）
    - 否则 → 该 recipe 跳过 R4
3. **R4 baseline**：每 recipe 的 R3 baseline = R2 winner pattern × R3 uniform 那个 cell 的 SR

> **修订理由（G1 R1 Blocking 5）**：原版"combined-winner.SR > phase3 g1/g10 best.SR + 2pp"按字面不可满足 — phase3 g1/g10 max SR = 1.000，1.00 + 0.02 = 1.02 不可达。改成"R3 winner vs R3 baseline (uniform online) Δ ≥ 2pp"，门槛从"绝对天花板"改成"R3 边际收益"，与"R4 是否值得为 W-FUT 双窗精调"的实际命题一致。

---

## §4 代码改动（代码级）

> 设计原则（修订后 — G1 R2 Blocking 1 + 2）：phase4 是 phase3 的"权重扫描扩展"。**最小化触动 phase3 文件**：
>
> 1. **改 src/ composer**（必须 — phase3 旧实现等权平均，无法支撑 weight 数值扫描）
> 2. **改 phase3_threshold_solver `reconstruct_scores`** 加 keyword-only `composer_weights` 参数（向后兼容 None）
> 3. **不动 `phase3_spec.py` / `phase3_spec.RECIPES`** — phase4 直接调 `v2_spec.build_eval_yaml(cfg_id, factors, composer)` 与 `v2_spec.build_warmup_yaml(...)`，自己构造 composer dict / factor list；phase4 RECIPES_PHASE4 是 phase4_spec 内部局部 dict，不与 phase3 RECIPES 合并
> 4. **不动 `run_phase3.py`** — phase4 是独立 runner，沿用 phase3 已确立的 `preload_normalizer_buffer → load_cache_config` 顺序
>
> 这三个接缝点（composer / solver / phase4 自己的 spec+runner）保证：
> - phase 3 既有 yaml 行为按本 plan §4.6 backward compat 测试守住数值一致；
> - phase 3 RECIPES_PHASE4 dict 在同进程内被 phase4 import 后仍然是 11 个 key（test 强制，§4.3.1）；
> - phase4 cell yaml / weight 配置完全 self-contained。

### §4.1 `exp/verdict_factor_judge/phase4_spec.py`（新建，~250 LOC）

#### §4.1.1 Score key 命名（与 phase3 严格一致，不另起 schema）

```python
# 跟 phase3 的 _factor_block 产出的 key 完全同款
# - offline: f"{desc}_offline_{channel}__p{past}_f{future}"
# - online:  f"{desc}_online_{channel}__p{past}_f{future}"

P1_OFFLINE_KEYS: tuple[str, ...] = (
    "jerk_offline_state__p0_f3",        "jerk_offline_state__p0_f5",
    "direction_offline_state__p0_f3",   "direction_offline_state__p0_f5",
    "dispersion_offline_state__p0_f3",  "dispersion_offline_state__p0_f5",
    "path_length_offline_state__p0_f3", "path_length_offline_state__p0_f5",
)
P2_OFFLINE_KEYS: tuple[str, ...] = (
    "jerk_offline_action__p0_f3",        "jerk_offline_action__p0_f5",
    "direction_offline_action__p0_f3",   "direction_offline_action__p0_f5",
    "dispersion_offline_action__p0_f3",  "dispersion_offline_action__p0_f5",
    "path_length_offline_action__p0_f3", "path_length_offline_action__p0_f5",
)
SHARED_ONLINE_KEYS: tuple[str, ...] = (
    "jerk_online_action__p3_f3",
    "dispersion_online_action__p3_f3",
)
DESC_ORDER: tuple[str, ...] = ("jerk", "direction", "dispersion", "path_length")  # R2 pattern axis
WINDOW_ORDER: tuple[tuple[int,int], ...] = ((0, 3), (0, 5))                       # R4 pattern axis
```

#### §4.1.2 Factor block（复用 phase3_spec 的 helper，不新建）

```python
from exp.verdict_factor_judge.phase3_spec import (
    _factor_block,            # produces a single factor dict from (desc, source, channel, windows)
    _W_FUT,                   # = [{"past":0,"future":3}, {"past":0,"future":5}]
    _W_K3,                    # = [{"past":3,"future":3}]
    _multi_desc_factors,      # produces a list of factor dicts for a desc tuple
    _orientations_for,
    _directions_for,
    build_warmup_yaml,        # imported from phase2_spec via phase3_spec
    build_eval_yaml,          # ditto — phase4 reuses this directly
)

_ALL_DESC = ("jerk", "direction", "dispersion", "path_length")
_ONLINE_DESC = ("jerk", "dispersion")

P1_OFFLINE_FACTORS = _multi_desc_factors(_ALL_DESC, "offline", "state",  _W_FUT)
P2_OFFLINE_FACTORS = _multi_desc_factors(_ALL_DESC, "offline", "action", _W_FUT)
SHARED_ONLINE_FACTORS = _multi_desc_factors(_ONLINE_DESC, "online", "action", _W_K3)
```

#### §4.1.3 Recipe 元数据

```python
from typing import TypedDict

class Phase4Recipe(TypedDict):
    recipe_id: str
    factors: list[dict]              # offline factors + online factors concatenated
    offline_keys: tuple[str, ...]    # 8 keys
    online_keys: tuple[str, ...]     # 2 keys (= SHARED_ONLINE_KEYS)
    declared_keys: tuple[str, ...]   # offline_keys + online_keys (10 total)
    orientations: dict[str, str]     # for solver
    directions: dict[str, str]       # for composer

RECIPES_PHASE4: dict[str, Phase4Recipe] = {
    "p1_state_fut_online_act":  _make_p1(),    # private constructors below
    "p2_action_fut_online_act": _make_p2(),
}

def _make_p1() -> Phase4Recipe:
    declared = P1_OFFLINE_KEYS + SHARED_ONLINE_KEYS
    return {
        "recipe_id":     "p1_state_fut_online_act",
        "factors":       list(P1_OFFLINE_FACTORS) + list(SHARED_ONLINE_FACTORS),
        "offline_keys":  P1_OFFLINE_KEYS,
        "online_keys":   SHARED_ONLINE_KEYS,
        "declared_keys": declared,
        "orientations":  _orientations_for(list(declared)),
        "directions":    _directions_for(list(declared)),
    }
# _make_p2 同款，换 P2_OFFLINE_KEYS / P2_OFFLINE_FACTORS。
```

#### §4.1.4 Round pattern 常量

```python
R1_ALPHAS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)

# offline 4-desc pattern: shares assigned to (jerk, direction, dispersion, path_length)
R2_OFFLINE_PATTERNS: dict[str, tuple[int, int, int, int]] = {
    "uniform":     (1, 1, 1, 1),
    "jerk-heavy":  (2, 1, 1, 1),
    "dir-heavy":   (1, 2, 1, 1),
    "disp-heavy":  (1, 1, 2, 1),
    "path-heavy":  (1, 1, 1, 2),
    "jerk-only":   (1, 0, 0, 0),
    "dir-only":    (0, 1, 0, 0),
    "disp-only":   (0, 0, 1, 0),
    "path-only":   (0, 0, 0, 1),
}

# online 2-desc pattern: shares assigned to (jerk_online, dispersion_online)
R3_ONLINE_PATTERNS: dict[str, tuple[int, int]] = {
    "uniform":    (1, 1),
    "jerk-heavy": (2, 1),
    "disp-heavy": (1, 2),
    "jerk-only":  (1, 0),
    "disp-only":  (0, 1),
}

# W-FUT window pattern: shares assigned to (p0_f3, p0_f5) inside each desc
R4_WINDOW_PATTERNS: dict[str, tuple[int, int]] = {
    "uniform":     (1, 1),
    "short-heavy": (2, 1),
    "long-heavy":  (1, 2),
    "short-only":  (1, 0),
    "long-only":   (0, 1),
}

LOCKED_FH: float = 0.5
LOCKED_WS: float = 0.5
```

#### §4.1.5 权重生成器（4 个，纯函数，无 I/O）

每个生成器返回 `dict[score_key, float]`，**键集 = 10 个 declared_keys**，缺一报错。返回的权重 ∑=1.0（FP eps 内）。

```python
def _alloc_offline(
    recipe_id: str,
    alpha: float,                                      # offline group total
    desc_pattern: tuple[int, int, int, int],           # (jerk, dir, disp, path) shares
    window_pattern: tuple[int, int],                   # (p0_f3, p0_f5) shares within desc
) -> dict[str, float]:
    """Distribute `alpha` over 8 offline keys = 4 desc × 2 window.

    For each desc, allocation = alpha * (desc_share / desc_total).
    Within desc, two windows split by window_pattern.

    All-zero shares -> all 0.0 weights for that group (legal).
    """
    keys = RECIPES_PHASE4[recipe_id]["offline_keys"]
    desc_total = sum(desc_pattern)
    win_total = sum(window_pattern)
    out: dict[str, float] = {}
    for d_idx, desc in enumerate(DESC_ORDER):
        desc_share = desc_pattern[d_idx]
        desc_alloc = (alpha * desc_share / desc_total) if desc_total > 0 else 0.0
        for w_idx, (p, f) in enumerate(WINDOW_ORDER):
            key = f"{desc}_offline_{recipe_id_to_channel(recipe_id)}__p{p}_f{f}"
            assert key in keys, f"key {key} not in declared offline_keys"
            win_share = window_pattern[w_idx]
            win_alloc = (desc_alloc * win_share / win_total) if win_total > 0 else 0.0
            out[key] = win_alloc
    return out

def _alloc_online(
    online_total: float,                  # = 1 - alpha
    online_pattern: tuple[int, int],      # (jerk_on, disp_on) shares
) -> dict[str, float]:
    total = sum(online_pattern)
    if total == 0:
        return {k: 0.0 for k in SHARED_ONLINE_KEYS}
    return {
        SHARED_ONLINE_KEYS[0]: online_total * online_pattern[0] / total,
        SHARED_ONLINE_KEYS[1]: online_total * online_pattern[1] / total,
    }

def generate_r1_weights(recipe_id: str, alpha: float) -> dict[str, float]:
    """R1: offline desc uniform (1,1,1,1), online uniform (1,1), windows uniform."""
    off = _alloc_offline(recipe_id, alpha, (1,1,1,1), (1,1))
    on  = _alloc_online(1.0 - alpha, (1, 1))
    return {**off, **on}

def generate_r2_weights(
    recipe_id: str,
    alpha_star: float,
    offline_pattern: tuple[int, int, int, int],
) -> dict[str, float]:
    """R2: alpha locked at α*, online uniform, windows uniform; sweep desc shares."""
    off = _alloc_offline(recipe_id, alpha_star, offline_pattern, (1, 1))
    on  = _alloc_online(1.0 - alpha_star, (1, 1))
    return {**off, **on}

def generate_r3_weights(
    recipe_id: str,
    alpha_star: float,
    offline_pattern: tuple[int, int, int, int],
    online_pattern: tuple[int, int],
) -> dict[str, float]:
    off = _alloc_offline(recipe_id, alpha_star, offline_pattern, (1, 1))
    on  = _alloc_online(1.0 - alpha_star, online_pattern)
    return {**off, **on}

def generate_r4_weights(
    recipe_id: str,
    alpha_star: float,
    offline_pattern: tuple[int, int, int, int],
    online_pattern: tuple[int, int],
    window_pattern: tuple[int, int],
) -> dict[str, float]:
    off = _alloc_offline(recipe_id, alpha_star, offline_pattern, window_pattern)
    on  = _alloc_online(1.0 - alpha_star, online_pattern)
    return {**off, **on}
```

#### §4.1.6 Cell ID 命名（运行时 yaml_id）

```python
def cell_id_r1(cfg_id: str, recipe_id: str, alpha: float) -> str:
    return f"{cfg_id}_phase4_{recipe_id}__r1_a{alpha:.1f}"
def cell_id_r2(cfg_id: str, recipe_id: str, alpha: float, off_pat: str) -> str:
    return f"{cfg_id}_phase4_{recipe_id}__r2_a{alpha:.1f}_off-{off_pat}"
def cell_id_r3(cfg_id: str, recipe_id: str, alpha: float, off_pat: str, on_pat: str) -> str:
    return f"{cfg_id}_phase4_{recipe_id}__r3_a{alpha:.1f}_off-{off_pat}_on-{on_pat}"
def cell_id_r4(cfg_id, recipe_id, alpha, off_pat, on_pat, win_pat) -> str:
    return f"{cfg_id}_phase4_{recipe_id}__r4_a{alpha:.1f}_off-{off_pat}_on-{on_pat}_win-{win_pat}"

def warmup_yaml_id(cfg_id: str, recipe_id: str) -> str:
    return f"{cfg_id}_phase4_{recipe_id}__warmup"
def warmup_eval_yaml_id(cfg_id: str, recipe_id: str) -> str:
    return f"{cfg_id}_phase4_{recipe_id}"   # parent of all (rN, ...) cells, used by warmup buffer key
```

#### §4.1.7 Yaml builder（**不污染 phase3 全局** — G1 R2 Blocking 2 修订）

`v2_spec.build_eval_yaml` 与 `v2_spec.build_warmup_yaml` 实际签名（fact-checked from `v2_spec.py:334, 376`）：

```python
def build_eval_yaml(
    cfg_id: str,
    factors: list[dict],          # full factor blocks list (4-layer architecture)
    composer: dict,               # full composer dict, including type / weights / thresholds / ...
    *,
    export_factor_outputs: bool = True,
    calibration_window_size: int = 50,
) -> dict: ...

def build_warmup_yaml(
    cfg_id: str,
    eval_yaml_id: str,
    *,
    eval_factors: list[dict] | None = None,
) -> dict: ...
```

这两个 builder 接受**完整 composer dict / factor list**，与 `phase3_spec.RECIPES` 全局**完全无依赖**。Phase 4 直接调用，不需要 register 到 phase3 RECIPES：

```python
# In phase4_spec.py
from exp.verdict_factor_judge.v2_spec import build_eval_yaml as v2_build_eval_yaml
from exp.verdict_factor_judge.v2_spec import build_warmup_yaml as v2_build_warmup_yaml

def build_phase4_warmup_yaml(cfg_id: str, recipe_id: str) -> dict:
    """Build a phase4 warmup yaml without touching phase3 RECIPES."""
    r = RECIPES_PHASE4[recipe_id]      # phase4-local dict, never merged into phase3
    return v2_build_warmup_yaml(
        cfg_id=cfg_id,
        eval_yaml_id=warmup_eval_yaml_id(cfg_id, recipe_id),
        eval_factors=r["factors"],     # offline + online factors concatenated
    )

def _build_phase4_composer_dict(
    recipe_id: str,
    composer_weights: dict[str, float],
    fh_thr: float, ws_thr: float,
) -> dict:
    """Self-contained composer dict construction. Mirrors the composer block
    that phase3.build_eval_yaml_for_cell emits, but with phase4-supplied weights.
    """
    r = RECIPES_PHASE4[recipe_id]
    composer: dict = {
        "type": "weighted_sum_zero_nan",
        "weights": dict(composer_weights),
        "tier_thresholds": {"full_hit": float(fh_thr), "warm_start": float(ws_thr)},
        "warm_start_t": 0.5,
    }
    if r["directions"]:
        composer["directions"] = dict(r["directions"])
    return composer

def build_phase4_eval_yaml(
    cfg_id: str, recipe_id: str,
    fh_thr: float, ws_thr: float,
    fh_ratio: float, ws_ratio: float,            # per-recipe locked; for traceability only (not used by builder)
    composer_weights: dict[str, float],          # 10-key dict from generate_r{N}_weights, REQUIRED
) -> dict:
    """Phase 4 builds eval yaml directly via v2_spec; phase3 RECIPES untouched."""
    r = RECIPES_PHASE4[recipe_id]
    composer = _build_phase4_composer_dict(recipe_id, composer_weights, fh_thr, ws_thr)
    return v2_build_eval_yaml(
        cfg_id=cfg_id,
        factors=r["factors"],
        composer=composer,
        export_factor_outputs=True,
        calibration_window_size=50,
    )
```

> **修订理由（G1 R2 Blocking 2）**：R1 修订把 phase3 RECIPES 当作 mutable global mutate（`P3.update(RECIPES_PHASE4)`），会让同进程后续 phase3 manifest / runner / tests 看到 13 个 recipes 而非 11，破坏 phase3/phase4 边界。改用 `v2_spec.build_eval_yaml` / `build_warmup_yaml` 直接接 composer dict — 它们本来就是 RECIPES-agnostic 的下层 helper，phase 4 不需要任何 RECIPES_PHASE4 global 注册。phase3_spec.py 现在完全不动，§4.6 边界表已同步修正。

#### §4.1.8 不变量（测试强制 — `test_phase4_spec.py`）

| ID | 不变量 |
|---|---|
| INV-1 | 每个 `generate_r{1,2,3,4}_weights` 返回 dict 的键集 == `RECIPES_PHASE4[recipe_id]["declared_keys"]`，长度 10 |
| INV-2 | 返回 dict 所有 value ≥ 0，∑ = 1.0（FP eps 1e-9 内） |
| INV-3 | `generate_r1_weights(rid, 0.0)` 把 8 个 offline key 全置 0；只剩 2 个 online key 各 0.5 |
| INV-4 | `generate_r1_weights(rid, 1.0)` 把 2 个 online key 全置 0；offline 8 key 各 1/8 |
| INV-5 | `generate_r2_weights(rid, α, (1,0,0,0))` 把非 jerk 的 6 个 offline key 全置 0；jerk 2 win 平分 α |
| INV-6 | `generate_r3_weights(rid, α, off_pat, (0,1))` 把 jerk_online_action 置 0；只剩 dispersion_online_action = 1−α |
| INV-7 | R4 window-only pattern (1,0) 把每个 desc 的 p0_f5 全置 0；p0_f3 拿全部 desc 配额 |
| INV-8 | `RECIPES_PHASE4["p1_..."]["declared_keys"]` 长度恰好 10，无重复，与 §4.1.1 常量逐字一致 |
| INV-9 | p1 / p2 的 online 部分共享同款 keys（`SHARED_ONLINE_KEYS`） |
| INV-10 | warmup_yaml + eval_yaml 的 `warmup_yaml_id` / `warmup_eval_yaml_id` 字段两侧对齐（防 server 找不到 buffer） |

### §4.2 `exp/verdict_factor_judge/run_phase4.py`（新建，~400 LOC）

#### §4.2.1 CLI（**4-mode 拆分** — G1 R2 Blocking 3 修订）

每次执行 phase4 都需明确**该步要不要 server**。把 R1 修订里模糊的 `--generate-only` 拆成 4 个互斥 mode：

```python
parser.add_argument(
    "--mode", required=True,
    choices=["emit-warmup-yaml", "run-warmup", "emit-eval-yamls", "run-eval"],
    help="Phase 4 4-stage pipeline. The first three are no-server; only run-warmup / run-eval need --serve-host/--serve-port.",
)
parser.add_argument("--round", type=int, choices=[1,2,3,4], required=True)
parser.add_argument("--alpha-star", type=str, default=None,
                    help="Required when round >= 2 AND mode in {emit-eval-yamls, run-eval}. "
                         "Per-recipe map, e.g. 'p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6'. "
                         "Each value must be in R1_ALPHAS.")
parser.add_argument("--offline-pattern", type=str, default=None,
                    help="Required when round >= 3 AND mode in {emit-eval-yamls, run-eval}. "
                         "Per-recipe map; values in R2_OFFLINE_PATTERNS.")
parser.add_argument("--online-pattern", type=str, default=None,
                    help="Required when round == 4 AND mode in {emit-eval-yamls, run-eval}. "
                         "Per-recipe map; values in R3_ONLINE_PATTERNS.")
parser.add_argument("--recipe", type=str, default=None,
                    help="Optional: restrict to one recipe id (e.g. 'p1_state_fut_online_act'). "
                         "Useful when only one recipe needs that round (e.g. R4 conditional trigger fires for one recipe only).")
parser.add_argument("--serve-host", default="155.98.36.13",
                    help="Required when mode in {run-warmup, run-eval}.")
parser.add_argument("--serve-port", type=int, default=9000,
                    help="Required when mode in {run-warmup, run-eval}.")
parser.add_argument("--summary-out", default="exp/verdict_factor_judge/data/phase4")
parser.add_argument("--ep-per-task", type=int, default=10)
parser.add_argument("--n-task", type=int, default=10)
parser.add_argument("--cfg-id", default="spatial16_w8_d4")
```

Mode 的工作划分：

| mode | 要不要 server | 输入 | 输出 |
|---|:---:|---|---|
| **emit-warmup-yaml** | ❌ | recipe ids | `config/spatial16/phase4/warmup/<recipe>__warmup.yaml`（每 recipe 一份；幂等 — 已存在则覆盖以保最新 schema） |
| **run-warmup** | ✅ | recipe ids；前面已 emit 的 warmup yaml | 跑 warmup eval；server-side dump → 客户端 fetch_dump 写 `data/phase4/warmup_factor_raw/<recipe>.jsonl` |
| **emit-eval-yamls** | ❌ | round + per-recipe winner mapping；前面跑出的 `factor_raw.jsonl` | 调 `_solve_thresholds` 算每 cell (fh_thr, ws_thr) → 写 `config/spatial16/phase4/eval/<cell_yaml_id>.yaml`（cell yaml 内嵌完整 composer weights + thresholds） |
| **run-eval** | ✅ | round + per-recipe winner mapping；前面 emit 的 eval yamls | 按 cell 列表逐个 `preload_normalizer_buffer → load_cache_config → run_eval_loop`；append `r{N}_*/per_yaml_summary.jsonl` + `r{N}_*/episode_results/*.json` + `r{N}_*/per_step/*.jsonl` |

Mapping parser helper:

```python
def _parse_per_recipe_map(s: str | None, valid_values: set, recipes: list[str]) -> dict[str, str | float]:
    if s is None: return {}
    out: dict[str, str | float] = {}
    for kv in s.split(","):
        k, _, v = kv.strip().partition("=")
        assert k in recipes, f"unknown recipe id {k!r}; valid: {recipes}"
        v_parsed = _try_float(v) if _try_float(v) in valid_values else v
        assert v_parsed in valid_values, f"value {v!r} not in {valid_values}"
        out[k] = v_parsed
    return out
```

CLI invariants（main 入口 assert）：
- `args.mode == "emit-warmup-yaml"` → 不读 server / mapping；任何 round 都允许（warmup yaml 与 round 无关，但 `--round` 仍必填用作 sanity / consistent CLI shape）
- `args.mode == "run-warmup"` → 必填 `--serve-host` / `--serve-port`；要求 emit-warmup-yaml 输出已存在
- `args.mode == "emit-eval-yamls" or "run-eval"` AND `args.round >= 2` → 必填 `--alpha-star`，解析后含所有未被 `--recipe` 排除的 recipe
- `args.mode == "emit-eval-yamls" or "run-eval"` AND `args.round >= 3` → 上 + `--offline-pattern` 含所有该 round 的 recipe
- `args.mode == "emit-eval-yamls" or "run-eval"` AND `args.round == 4` → 上 + `--online-pattern`
- `args.mode == "emit-eval-yamls"` → 要求每 needed recipe 的 `factor_raw/<recipe>.jsonl` 已存在（即 run-warmup 已跑）
- `args.mode == "run-eval"` → 要求每个该 round/cell 的 eval yaml 文件已存在（emit-eval-yamls 已跑）

> **修订理由（G1 R2 Blocking 3）**：R1 修订写"`--generate-only` 跳过 ctl"，但调用图里 `_generate_yamls_and_thresholds` → `_ensure_warmup` 又用 `ctl.load_cache_config` 跑 warmup eval。两条路径互相矛盾导致 generate-only 路径不可执行。4-mode 拆分把"无 server 文件操作"与"要 server 跑 episode"分到不同命令，每个 mode 自己的前置文件 / server 检查由 main entry 的 assert 强制。

#### §4.2.2 文件路径常量

```python
DATA_ROOT       = Path("exp/verdict_factor_judge/data/phase4")
WARMUP_RAW_DIR  = DATA_ROOT / "warmup_factor_raw"               # <recipe>.jsonl
SUMMARY_DIRS = {
    1: DATA_ROOT / "r1_alpha"            / "per_yaml_summary.jsonl",
    2: DATA_ROOT / "r2_offline_desc"     / "per_yaml_summary.jsonl",
    3: DATA_ROOT / "r3_online_desc"      / "per_yaml_summary.jsonl",
    4: DATA_ROOT / "r4_window"           / "per_yaml_summary.jsonl",
}
PER_STEP_DIRS = {
    1: DATA_ROOT / "r1_alpha"            / "per_step",
    ...
}
EPISODE_DIRS = {
    1: DATA_ROOT / "r1_alpha"            / "episode_results",
    ...
}
PKL_PATH = Path("exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl")
CONFIG_OUT_DIR = Path("exp/verdict_factor_judge/config/spatial16/phase4")  # eval/, warmup/
```

#### §4.2.3 主流程（伪代码 — 4-mode dispatch）

```python
def main(args) -> None:
    _validate_cli_invariants(args)                                # §4.2.1 invariants
    _check_pkl_keys(PKL_PATH, RECIPES_PHASE4)                     # §5.0 pre-flight (still required for any mode that touches yamls/server)
    recipes_to_run = _filter_recipes(args)                        # honor --recipe restriction

    if args.mode == "emit-warmup-yaml":
        for rid in recipes_to_run:
            yaml_path = CONFIG_OUT_DIR / "warmup" / f"{warmup_yaml_id(args.cfg_id, rid)}.yaml"
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            save_yaml(build_phase4_warmup_yaml(args.cfg_id, rid), yaml_path)
        return                                                    # NO server, NO ctl

    if args.mode == "run-warmup":
        for rid in recipes_to_run:
            yaml_path = CONFIG_OUT_DIR / "warmup" / f"{warmup_yaml_id(args.cfg_id, rid)}.yaml"
            assert yaml_path.exists(), f"emit-warmup-yaml not run yet for {rid}: {yaml_path}"
        ctl = WebsocketClientPolicy(args.serve_host, args.serve_port)
        _preload_pkl(ctl, PKL_PATH)
        for rid in recipes_to_run:
            _run_warmup_for_recipe(ctl, rid, args)                # load_cache_config(warmup) + eval loop + fetch_dump
            _extract_finite_factor_raw(...)                       # writes data/phase4/warmup_factor_raw/<rid>.jsonl
        return

    cells = _build_cell_list(args)                                # §4.2.4 — round-aware

    if args.mode == "emit-eval-yamls":
        for rid in recipes_to_run:
            raw_path = WARMUP_RAW_DIR / f"{rid}.jsonl"
            assert raw_path.exists(), f"run-warmup not done for {rid}: {raw_path}"
        for cell in cells:
            fh_thr, ws_thr = _solve_thresholds(cell)
            yaml_path = CONFIG_OUT_DIR / "eval" / f"{cell.yaml_id}.yaml"
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            save_yaml(build_phase4_eval_yaml(
                cfg_id=args.cfg_id, recipe_id=cell.recipe_id,
                fh_thr=fh_thr, ws_thr=ws_thr,
                fh_ratio=cell.fh_ratio, ws_ratio=cell.ws_ratio,
                composer_weights=cell.weights,
            ), yaml_path)
        return                                                    # NO server, NO ctl

    if args.mode == "run-eval":
        for cell in cells:
            assert cell.yaml_path.exists(), f"emit-eval-yamls not done for {cell.yaml_id}"
        summary_path = SUMMARY_DIRS[args.round]
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        done_yaml_ids = _load_done_yaml_ids(summary_path)
        pending = [c for c in cells if c.yaml_id not in done_yaml_ids]
        ctl = WebsocketClientPolicy(args.serve_host, args.serve_port)
        _preload_pkl(ctl, PKL_PATH)
        by_recipe = group_by_recipe(pending)
        for recipe_id, recipe_cells in by_recipe.items():
            for cell in recipe_cells:
                _run_one_cell(ctl, cell, args, summary_path)      # §4.2.3b — preload-before-load invariant
            _recipe_end_cleanup(ctl, recipe_id, recipe_cells, args)
        _dump_decision_gate_table(args.round, summary_path)       # §4.2.7
        return
```

> 注：`_run_warmup_for_recipe` 与 phase3 `run_phase3.py` 的 warmup-run 路径同款（`load_cache_config(warmup_yaml)` + `_run_eval_loop` + 取 dump → finite-only factor_raw jsonl）；细节复用，不再展开。

#### §4.2.3b `_run_one_cell` — preload-before-load 顺序

Phase 3 `run_phase3.py` line 370-382 已确立 invariant：**eval cell 的 `load_cache_config` 之前必须先 `preload_normalizer_buffer(eval_yaml_id, buffer)`**，否则 server-side `_load_calibration_samples(yaml_id=eval_yaml_id)` 找不到 WarmupPool entry，连接时 `build_per_connection_components` 报错 `WarmupPool has no entry for yaml_id=...`。Phase 4 必须沿用同款顺序：

```python
def _run_one_cell(ctl, cell: Cell, args, summary_path: Path) -> None:
    # Step 1: Read warmup buffer from cached factor_raw (per-recipe, written by _ensure_warmup).
    factor_raw_path = WARMUP_RAW_DIR / f"{cell.recipe_id}.jsonl"
    buffer = _load_warmup_buffer_from_factor_raw(
        factor_raw_path, declared_keys=RECIPES_PHASE4[cell.recipe_id]["declared_keys"],
    )

    # Step 2: PRELOAD before load_cache_config — phase3 invariant.
    ack = ctl.preload_normalizer_buffer(cell.yaml_id, buffer)
    assert ack.get("ok"), f"preload failed: {ack}"

    # Step 3: Load eval yaml (composer weights already baked into yaml during _generate_yamls_and_thresholds).
    yaml_text = cell.yaml_path.read_text()
    ctl.load_cache_config(yaml_content=yaml_text, yaml_id=cell.yaml_id)

    # Step 4: Run eval episodes via examples/libero main runner.
    _run_eval_loop(ctl, cell, args)

    # Step 5: Append per_yaml_summary row.
    _append_summary_row(summary_path, cell, eval_results)
```

> **修订理由（G1 R1 Blocking 2.c）**：原版调用图直接 `ctl.load_cache_config(eval_yaml)` 没写 preload；按现有 phase 3 server 实现这必然失败。明确加 §4.2.3b 与 phase3 line 370-382 对齐，让 G2 reviewer 能 grep 到 invariant。

#### §4.2.4 `_build_cell_list(args)`

```python
@dataclass(frozen=True)
class Cell:
    yaml_id: str
    yaml_path: Path
    recipe_id: str
    round_id: int                              # 1..4
    pattern_label: str                         # human-readable, e.g. "α=0.4__off-jerk-heavy"
    weights: dict[str, float]                  # 10-key dict
    fh_thr: float | None                       # filled by solver; None until §4.2.6
    ws_thr: float | None
    fh_ratio: float                            # per-recipe: p1=0.5, p2=0.5
    ws_ratio: float                            # per-recipe: p1=0.5, p2=0.4

# Per-recipe locked cells (see §0.2 anchor table).
LOCKED_CELLS: dict[str, tuple[float, float]] = {
    "p1_state_fut_online_act":  (0.5, 0.5),
    "p2_action_fut_online_act": (0.5, 0.4),
}

def _build_cell_list(args) -> list[Cell]:
    """Build the per-recipe cell list using per-recipe winner mappings."""
    alpha_map        = _parse_per_recipe_map(args.alpha_star,        set(R1_ALPHAS),           list(RECIPES_PHASE4))
    off_pattern_map  = _parse_per_recipe_map(args.offline_pattern,   set(R2_OFFLINE_PATTERNS), list(RECIPES_PHASE4))
    on_pattern_map   = _parse_per_recipe_map(args.online_pattern,    set(R3_ONLINE_PATTERNS),  list(RECIPES_PHASE4))
    recipe_filter = {args.recipe} if args.recipe else set(RECIPES_PHASE4)

    cells = []
    for recipe_id in RECIPES_PHASE4:
        if recipe_id not in recipe_filter: continue
        fh, ws = LOCKED_CELLS[recipe_id]                # per-recipe locked
        if args.round == 1:
            for alpha in R1_ALPHAS:
                cells.append(Cell(
                    yaml_id=cell_id_r1(args.cfg_id, recipe_id, alpha),
                    yaml_path=CONFIG_OUT_DIR / "eval" / f"{cell_id_r1(args.cfg_id, recipe_id, alpha)}.yaml",
                    recipe_id=recipe_id, round_id=1,
                    pattern_label=f"a{alpha:.1f}",
                    weights=generate_r1_weights(recipe_id, alpha),
                    fh_thr=None, ws_thr=None,
                    fh_ratio=fh, ws_ratio=ws,
                ))
        elif args.round == 2:
            alpha_star = alpha_map[recipe_id]                                # per-recipe lookup
            for off_pat_name, off_pat in R2_OFFLINE_PATTERNS.items():
                cells.append(Cell(
                    yaml_id=cell_id_r2(args.cfg_id, recipe_id, alpha_star, off_pat_name),
                    yaml_path=CONFIG_OUT_DIR / "eval" / f"{cell_id_r2(...)}.yaml",
                    recipe_id=recipe_id, round_id=2,
                    pattern_label=f"a{alpha_star:.1f}__off-{off_pat_name}",
                    weights=generate_r2_weights(recipe_id, alpha_star, off_pat),
                    fh_thr=None, ws_thr=None, fh_ratio=fh, ws_ratio=ws,
                ))
        elif args.round == 3:
            alpha_star = alpha_map[recipe_id]
            off_pat_name = off_pattern_map[recipe_id]
            off_pat = R2_OFFLINE_PATTERNS[off_pat_name]
            for on_pat_name, on_pat in R3_ONLINE_PATTERNS.items():
                cells.append(Cell(
                    ...weights=generate_r3_weights(recipe_id, alpha_star, off_pat, on_pat),
                    fh_ratio=fh, ws_ratio=ws,
                ))
        elif args.round == 4:
            alpha_star = alpha_map[recipe_id]
            off_pat = R2_OFFLINE_PATTERNS[off_pattern_map[recipe_id]]
            on_pat  = R3_ONLINE_PATTERNS[on_pattern_map[recipe_id]]
            for win_pat_name, win_pat in R4_WINDOW_PATTERNS.items():
                cells.append(Cell(
                    ...weights=generate_r4_weights(recipe_id, alpha_star, off_pat, on_pat, win_pat),
                    fh_ratio=fh, ws_ratio=ws,
                ))
    return cells
```

#### §4.2.5 `_run_warmup_for_recipe` — `--mode run-warmup` helper

```python
def _run_warmup_for_recipe(ctl, recipe_id: str, args) -> None:
    """Run warmup yaml on the server and extract finite-only factor_raw to disk.
    Caller (the run-warmup mode branch in main, §4.2.3) already verified
    config/.../warmup/<rid>__warmup.yaml exists from emit-warmup-yaml."""
    factor_raw_path = WARMUP_RAW_DIR / f"{recipe_id}.jsonl"
    if factor_raw_path.exists() and factor_raw_path.stat().st_size > 0:
        return                                                      # cached from earlier invocation

    warmup_yaml_path = CONFIG_OUT_DIR / "warmup" / f"{warmup_yaml_id(args.cfg_id, recipe_id)}.yaml"
    ctl.load_cache_config(
        yaml_content=warmup_yaml_path.read_text(),
        yaml_id=warmup_yaml_id(args.cfg_id, recipe_id),
    )
    _run_eval_loop(ctl, n_task=args.n_task, ep_per_task=args.ep_per_task, ...)

    src = DATA_ROOT / "warmup_per_step" / f"{warmup_yaml_id(args.cfg_id, recipe_id)}.jsonl"
    factor_raw_path.parent.mkdir(parents=True, exist_ok=True)
    _extract_finite_factor_raw(
        src, factor_raw_path,
        declared_keys=RECIPES_PHASE4[recipe_id]["declared_keys"],
    )
```

#### §4.2.6 `_solve_thresholds` — `--mode emit-eval-yamls` helper

Phase 3 `reconstruct_scores` 实际签名（fact-checked from `phase3_threshold_solver.py:115`）：
```python
def reconstruct_scores(jsonl_path: Path, recipe: Recipe) -> list[float]: ...
# Internal: composer = WeightedSumZeroNanComposer(weights={k: 1.0 for k in recipe.declared_keys}, ...)
```

Phase 4 src 改动（向后兼容 — keyword-only kwarg；None 维持 phase3 行为）：

```python
# In phase3_threshold_solver.py — extend signature:
def reconstruct_scores(
    jsonl_path: Path,
    recipe: Recipe,
    *,
    composer_weights: dict[str, float] | None = None,                # NEW — None reproduces phase3 behavior
) -> list[float]:
    ...
    weights = composer_weights if composer_weights is not None else {k: 1.0 for k in recipe.declared_keys}
    composer = WeightedSumZeroNanComposer(
        weights=weights,                                              # NEW — passes through, score path uses real weighted sum
        full_hit_threshold=0.0, warm_start_threshold=0.0,
        warm_start_t=0.5, directions=recipe.directions or None,
    )
    ...
```

Phase 4 runner 调用（**仅在 `--mode emit-eval-yamls` 分支调用**，无需 `ctl`）：

```python
# In run_phase4.py:
from exp.verdict_factor_judge.phase3_threshold_solver import (
    load_per_key_finite_history, derive_thresholds, reconstruct_scores,
    Recipe as SolverRecipe,
)

def _solve_thresholds(cell: Cell) -> tuple[float, float]:
    """For a given weight vector, compute (fh_thr, ws_thr) targeting cell's locked (fh_ratio, ws_ratio).
    Pure offline: reads cached warmup factor_raw jsonl, runs solver, returns thr pair.
    No server, no ctl. Called by the emit-eval-yamls mode branch in main (§4.2.3)."""
    raw_path = WARMUP_RAW_DIR / f"{cell.recipe_id}.jsonl"
    r = RECIPES_PHASE4[cell.recipe_id]
    solver_recipe = SolverRecipe(
        recipe_id=cell.recipe_id,
        declared_keys=list(r["declared_keys"]),
        orientations=dict(r["orientations"]),
        directions=dict(r["directions"]),
    )
    scores = reconstruct_scores(
        raw_path,
        recipe=solver_recipe,
        composer_weights=cell.weights,                                # PHASE 4 ENTRY POINT — solver passthrough
    )
    return derive_thresholds(scores, fh_ratio=cell.fh_ratio, ws_ratio=cell.ws_ratio)
```

> **设计要点**：原 R2 修订有一个 `_generate_yamls_and_thresholds(cells, args)` 包装函数，内部还调 `_ensure_warmup(ctl=None, ...)` 同时跑 warmup eval — 与 §4.2.3 的 4-mode 主流程冲突。现 §4.2.5 只跑 warmup（`run-warmup` mode 调用），§4.2.6 只解 thr + 写 yaml（`emit-eval-yamls` mode 调用），两者职责正交、与主流程严格对齐。
>
> **修订理由（G1 R1 Blocking 1 + Blocking 2.b）**：原版"phase3 solver 已支持 composer_weights"是错的 — solver 内部 hardcode `{k: 1.0}`。即使 composer 加了真加权语义，solver 不传 weights 进去，phase 4 R1/R2/R3 解出来的 thr 仍来自 phase3 等权 score。必须显式扩 solver 签名 + 测试覆盖（`test_phase3_threshold_solver.py` 加 weight passthrough 测）。

#### §4.2.7 `_dump_decision_gate_table` — round-specific 与 §3 gate 对齐（G1 R2 Blocking 5 修订）

R1 / R2 / R3 跑完后写 `data/phase4/r{N}_*/decision_gate.json`。**winner 选择规则按 round 不同**，与 §3.1 / §3.2 / §3.3 严格对齐：

```python
def _dump_decision_gate_table(round_id: int, summary_path: Path) -> dict:
    """Round-specific decision logic. Output schema:
    {
        "round": int,
        "rule": str,                              # human-readable rule applied
        "winners": {recipe_id: cell_row},         # per-recipe winner per §3 rules
        "baselines": {recipe_id: cell_row | None}, # uniform baseline row used by R2/R3 gate
        "next_args_suggestion": {                 # exact CLI args for next round's emit-eval-yamls
            "alpha-star": str | None,             # e.g. "p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6"
            "offline-pattern": str | None,
            "online-pattern": str | None,
        },
        "trigger_decisions": {recipe_id: {"continue": bool, "reason": str}},
        "all": [cell_row, ...],                   # raw rows for traceability
    }
    """
    rows = [json.loads(l) for l in summary_path.read_text().splitlines() if l.strip()]
    by_recipe = group_by_recipe(rows)

    if round_id == 1:
        # G_R1 (§3.1): per-recipe argmax over score(α) = SR - 0.5*inf
        rule = "argmax SR - 0.5*inf per recipe; continue iff SR(α*) >= recipe_anchor_SR - 2pp"
        winners, decisions = {}, {}
        for rid, cells in by_recipe.items():
            cells_sorted = sorted(cells, key=lambda c: c["success_rate"] - 0.5 * _compute_inf(c), reverse=True)
            winner = cells_sorted[0]
            winners[rid] = winner
            anchor = ANCHOR_SR[rid]                 # p1 -> 0.95, p2 -> 0.96
            cont = winner["success_rate"] >= anchor - 0.02
            decisions[rid] = {"continue": cont, "reason": f"SR={winner['success_rate']:.3f} vs anchor {anchor:.3f}"}
        next_args = {
            "alpha-star": ",".join(f"{rid}={winners[rid]['alpha']:.1f}" for rid in winners
                                   if decisions[rid]["continue"]) or None,
            "offline-pattern": None, "online-pattern": None,
        }
        baselines = {}                              # R1 has no uniform baseline concept

    elif round_id == 2:
        # G_R2 (§3.2): per-recipe argmax SR; continue to R3 only if SR(pat*) > SR(uniform) + 2pp,
        # else force pattern* = "uniform" for R3.
        rule = "per recipe: argmax SR; if max - uniform <= 2pp, force pattern*=uniform"
        winners, baselines, decisions = {}, {}, {}
        for rid, cells in by_recipe.items():
            uniform_row = next(c for c in cells if c["pattern_label"].endswith("off-uniform"))
            best_row = max(cells, key=lambda c: c["success_rate"])
            baselines[rid] = uniform_row
            sr_delta = best_row["success_rate"] - uniform_row["success_rate"]
            if sr_delta > 0.02:
                winners[rid] = best_row
                decisions[rid] = {"continue": True,
                                  "reason": f"argmax pat={best_row['pattern_label']} SR={best_row['success_rate']:.3f} > uniform SR={uniform_row['success_rate']:.3f} + 2pp"}
            else:
                winners[rid] = uniform_row          # forced uniform
                decisions[rid] = {"continue": True,
                                  "reason": f"argmax SR Δ {sr_delta*100:.1f}pp <= 2pp; forcing pattern*=uniform"}
        # next-round arg names (offline pattern; reuse alpha from R1)
        next_args = {
            "alpha-star": _stringify_alpha_from_winners_self_label(by_recipe),     # carry forward R1 α* embedded in cell label
            "offline-pattern": ",".join(f"{rid}={_extract_offline_pat(winners[rid])}" for rid in winners),
            "online-pattern": None,
        }

    elif round_id == 3:
        # G_R3 (§3.2 + §3.3): per-recipe argmax SR online pattern; trigger R4 iff SR(pat*) > SR(uniform_online) + 2pp.
        rule = "per recipe: argmax SR among 5 online patterns; R4 trigger iff max - uniform_online > 2pp"
        winners, baselines, decisions = {}, {}, {}
        for rid, cells in by_recipe.items():
            uniform_row = next(c for c in cells if c["pattern_label"].endswith("on-uniform"))
            best_row = max(cells, key=lambda c: c["success_rate"])
            baselines[rid] = uniform_row
            sr_delta = best_row["success_rate"] - uniform_row["success_rate"]
            winners[rid] = best_row                 # always selected for completeness; R4 trigger is separate flag
            decisions[rid] = {"continue": sr_delta > 0.02,    # interpreted as "R4 trigger" for R3 gate
                              "reason": f"R3 winner SR={best_row['success_rate']:.3f}; uniform SR={uniform_row['success_rate']:.3f}; Δ {sr_delta*100:+.1f}pp"}
        next_args = {
            "alpha-star": _stringify_alpha_from_winners_self_label(by_recipe),
            "offline-pattern": _stringify_offline_pat_from_winners(winners),
            "online-pattern": ",".join(f"{rid}={_extract_online_pat(winners[rid])}" for rid in winners
                                        if decisions[rid]["continue"]) or None,
        }

    out = {
        "round": round_id, "rule": rule,
        "winners": winners, "baselines": baselines,
        "next_args_suggestion": next_args, "trigger_decisions": decisions,
        "all": rows,
    }
    json.dump(out, open(summary_path.parent / "decision_gate.json", "w"), indent=2)
    print(_format_human_readable(out))               # stdout for user inspection
    return out
```

> **修订理由（G1 R2 Blocking 5）**：R1 修订所有 round 都用 `SR - 0.5*inf`，但 §3.2 R2 / §3.3 R4 触发其实是 "argmax SR 且相对 uniform > 2pp"，与 R1 的 Pareto 综合分不同。若 user 直接 `cat decision_gate.json | jq .winners` 然后复制到下一轮 CLI，会绕过 §3.2 的"小于 2pp 强制 uniform"规则。新版 round-specific logic + `next_args_suggestion` 字段直接给出可粘贴的 CLI 字符串，user 不需要二次手算 gate。`tests/exp/test_phase4_runner.py` 加 `test_decision_gate_r1_picks_argmax_score`、`test_decision_gate_r2_forces_uniform_when_delta_below_2pp`、`test_decision_gate_r3_emits_r4_trigger_flag`、`test_decision_gate_next_args_round_trip` 4 个测试覆盖（详见 §4.3.2）。

### §4.3 测试

#### §4.3.0 `tests/cache/components/factors/test_composer_zero_nan.py`（**既有 file** — 路径修正自 G1 R1 的误写；加 ~5 tests）

针对 src/ composer 改动新增 numeric weight sensitivity 测试：

```python
def test_score_only_uses_real_weighted_sum_not_equal_average():
    """Different non-zero weights must produce different scores on the same factor input.

    Pre-phase4 behavior (BROKEN): _score_only collapsed all non-zero weights to equal
    average. Post-phase4: score = Σ w_k · contrib_k / Σ w_k.
    """
    weights_uniform = {"a": 1.0, "b": 1.0}
    weights_a_heavy = {"a": 2.0, "b": 1.0}
    composer_uni = WeightedSumZeroNanComposer(weights_uniform, ...)
    composer_ah  = WeightedSumZeroNanComposer(weights_a_heavy, ...)
    composer_uni.bind_orientations({"a": "safe", "b": "safe"})
    composer_ah.bind_orientations({"a": "safe", "b": "safe"})
    factors = {"a": 0.9, "b": 0.1}    # divergent
    s_uni = composer_uni._score_only(factors)
    s_ah  = composer_ah._score_only(factors)
    assert abs(s_uni - 0.5) < 1e-9                                       # (0.9+0.1)/2 = 0.5
    assert abs(s_ah  - (2*0.9 + 1*0.1) / 3) < 1e-9                       # = 1.9/3 ≈ 0.633
    assert s_ah > s_uni + 0.05                                            # numerical sensitivity ≥ 5e-2

def test_score_only_zero_weight_excluded_from_denominator():
    """Zero-weight keys do not contribute and do not pad the denominator."""
    weights = {"a": 0.5, "b": 0.0, "c": 0.5}
    ...

def test_score_only_nan_keeps_weight_in_denominator():
    """NaN raw value contributes 0 to numerator BUT keeps its weight in the denominator
    (zero_nan semantics — distinguishes this composer from regular weighted_sum)."""
    weights = {"a": 1.0, "b": 1.0}
    factors = {"a": 0.8, "b": float("nan")}
    composer = WeightedSumZeroNanComposer(weights, ...)
    composer.bind_orientations({"a": "safe", "b": "safe"})
    s = composer._score_only(factors)
    assert abs(s - 0.4) < 1e-9    # (1*0.8 + 1*0) / (1+1) = 0.4

def test_score_only_all_zero_weights_returns_nan():
    """Degenerate construction: empty declared_dependencies → NaN, surfaced as MISS by compose()."""
    ...

def test_score_only_backward_compat_uniform_weights_match_phase3_behavior():
    """For weights = {k: 1.0 for all k}, post-phase4 score equals pre-phase4 equal-average score
    on the same input. Locks in the regression-free path for phase 3 yamls."""
    ...
```

#### §4.3.0b `tests/exp/test_phase3_threshold_solver.py`（既有 file，加 ~3 tests）

```python
def test_reconstruct_scores_default_uses_uniform_weights(tmp_path):
    """composer_weights=None reproduces phase 3 behavior: uniform {k:1.0}."""
    ...

def test_reconstruct_scores_passthrough_changes_score(tmp_path):
    """Same JSONL + recipe, but different composer_weights, produces different score series."""
    history_jsonl = ...   # write 100 rows with diverging factor_raw across 2 keys
    recipe = SolverRecipe(declared_keys=["k1", "k2"], orientations={"k1":"safe","k2":"safe"}, directions={})
    s_uni = reconstruct_scores(history_jsonl, recipe, composer_weights={"k1": 1.0, "k2": 1.0})
    s_k1h = reconstruct_scores(history_jsonl, recipe, composer_weights={"k1": 2.0, "k2": 1.0})
    assert s_uni != s_k1h           # element-wise
    assert max(abs(a-b) for a,b in zip(s_uni, s_k1h)) > 1e-3

def test_reconstruct_scores_zero_weight_excludes_key_from_score(tmp_path):
    """composer_weights={"k1": 1.0, "k2": 0.0} matches reconstruct_scores on declared_keys=["k1"]."""
    ...
```

#### §4.3.1 `tests/exp/test_phase4_spec.py`（~150 LOC，~25 tests）

测试矩阵：

```python
# Group A — Recipe shape (5 tests)
def test_recipes_have_two_entries(): ...
def test_p1_declared_keys_count_10(): ...
def test_p2_declared_keys_count_10(): ...
def test_p1_p2_share_online_keys(): ...
def test_declared_keys_unique(): ...

# Group B — INV-1..INV-7 weight invariants (10 tests)
@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
@pytest.mark.parametrize("alpha", R1_ALPHAS)
def test_r1_weights_sum_to_one(rid, alpha): ...
@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
def test_r1_alpha0_zeros_offline(rid): ...
@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
def test_r1_alpha1_zeros_online(rid): ...
@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
@pytest.mark.parametrize("pat_name,pat", R2_OFFLINE_PATTERNS.items())
def test_r2_weights_sum_to_one(rid, pat_name, pat): ...
@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
def test_r2_jerk_only_zeros_other_desc(rid): ...
@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
@pytest.mark.parametrize("on_pat_name,on_pat", R3_ONLINE_PATTERNS.items())
def test_r3_weights_sum_to_one(rid, on_pat_name, on_pat): ...
@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
def test_r3_disp_only_zeros_jerk_online(rid): ...
@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
@pytest.mark.parametrize("win_pat_name,win_pat", R4_WINDOW_PATTERNS.items())
def test_r4_weights_sum_to_one(rid, win_pat_name, win_pat): ...
@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
def test_r4_short_only_zeros_long_window(rid): ...

# Group C — Cell ID stability (5 tests)
def test_cell_id_r1_format(): ...
def test_cell_id_r2_includes_pattern_name(): ...
def test_cell_id_uniqueness_across_rounds(): ...
def test_warmup_yaml_id_doesnt_clash_with_eval(): ...
def test_warmup_eval_yaml_id_matches_phase3_format(): ...

# Group D — Yaml builder integration (5 tests)
def test_build_warmup_yaml_for_recipe_has_correct_factor_count(): ...
def test_build_eval_yaml_for_round_cell_uses_provided_weights(): ...
def test_build_eval_yaml_locks_warm_start_t_at_0_5(): ...
def test_build_eval_yaml_passes_through_thr(): ...
def test_build_eval_yaml_directions_match_recipe(): ...
```

#### §4.3.2 `tests/exp/test_phase4_runner.py`（~150 LOC，~14 tests）

```python
# CLI invariants
def test_cli_round1_rejects_alpha_star(): ...
def test_cli_round2_requires_alpha_star(): ...
def test_cli_round3_requires_offline_pattern(): ...
def test_cli_round4_requires_online_pattern(): ...

# Per-recipe winner CLI dispatch (G1 R1 Blocking 3)
def test_parse_per_recipe_alpha_map_distinct_values():
    """`--alpha-star "p1=0.4,p2=0.6"` → {p1: 0.4, p2: 0.6}."""
    ...
def test_per_recipe_alpha_distributes_correctly():
    """build_cell_list uses each recipe's own α* — p1 cell weights reflect 0.4, p2 cell weights reflect 0.6."""
    ...
def test_recipe_filter_restricts_to_one():
    """`--recipe p1_state_fut_online_act` → only p1 cells in the list."""
    ...

# Cell list construction
def test_round1_builds_14_cells(): ...               # 2 recipe × 7 α
def test_round2_builds_18_cells(): ...               # 2 recipe × 9 pattern
def test_round3_builds_10_cells(): ...
def test_round4_builds_10_cells(): ...

# Per-recipe locked cell (G1 R1 Blocking 4)
def test_p1_cells_locked_at_05_05():
    """All p1 cells across all rounds have fh_ratio=0.5, ws_ratio=0.5."""
    ...
def test_p2_cells_locked_at_05_04():
    """All p2 cells across all rounds have fh_ratio=0.5, ws_ratio=0.4."""
    ...

# Warmup once-per-recipe
def test_ensure_warmup_skips_when_factor_raw_cached(tmp_path, monkeypatch): ...
def test_ensure_warmup_runs_when_cache_missing(tmp_path, monkeypatch): ...

# Preload-before-load ordering (G1 R1 Blocking 2.c, Suggestion 7)
def test_run_one_cell_calls_preload_before_load(monkeypatch):
    """A mock ctl records call order; assert preload_normalizer_buffer is invoked
    BEFORE load_cache_config for each eval cell."""
    calls = []
    class MockCtl:
        def preload_normalizer_buffer(self, yaml_id, buf):
            calls.append(("preload", yaml_id))
            return {"ok": True}
        def load_cache_config(self, yaml_content, yaml_id):
            calls.append(("load", yaml_id))
            return {"ok": True}
    ...
    _run_one_cell(MockCtl(), cell, args, summary_path)
    assert calls[0][0] == "preload"
    assert calls[1][0] == "load"
    assert calls[0][1] == calls[1][1] == cell.yaml_id

# Decision gate — round-specific (G1 R2 Blocking 5)
def test_decision_gate_r1_picks_argmax_score():
    """R1: winner per recipe = argmax (SR - 0.5*inf); decision.continue iff SR >= anchor - 2pp."""
    ...
def test_decision_gate_r2_forces_uniform_when_delta_below_2pp():
    """R2: when argmax SR - uniform SR <= 2pp, winner is forced to 'uniform' pattern."""
    ...
def test_decision_gate_r2_picks_argmax_when_delta_above_2pp():
    """R2: when argmax SR > uniform SR + 2pp, winner = argmax pattern."""
    ...
def test_decision_gate_r3_emits_r4_trigger_flag():
    """R3: trigger_decisions[recipe].continue = (R3 winner.SR > R3 uniform online.SR + 2pp)."""
    ...
def test_decision_gate_next_args_round_trip():
    """The next_args_suggestion in decision_gate.json round-trips through _parse_per_recipe_map
    without errors and produces a valid Cell list for the next round."""
    ...

# 4-mode CLI dispatch
def test_mode_emit_warmup_yaml_no_server(): ...
def test_mode_run_warmup_requires_serve_host(): ...
def test_mode_emit_eval_yamls_requires_factor_raw_present(): ...
def test_mode_run_eval_requires_eval_yamls_present(): ...

# phase3 RECIPES not polluted (G1 R2 Blocking 2)
def test_phase4_import_does_not_mutate_phase3_recipes():
    """After importing exp.verdict_factor_judge.phase4_spec, phase3_spec.RECIPES still has
    exactly 11 entries (g1..g11), and none of the phase4 keys (p1, p2) are in it."""
    from exp.verdict_factor_judge import phase3_spec
    before = set(phase3_spec.RECIPES.keys())
    from exp.verdict_factor_judge import phase4_spec    # noqa: F401
    after = set(phase3_spec.RECIPES.keys())
    assert before == after
    assert len(after) == 11
    assert not any(k.startswith("p1_") or k.startswith("p2_") for k in after)

# Resume
def test_load_done_yaml_ids_skips_processed_rows(tmp_path): ...
```

### §4.4 分析脚本

#### §4.4.1 `analysis/plot_alpha_sweep_phase4.py`（~120 LOC）

```python
def main(round_id: int = 1) -> None:
    summary = _load(SUMMARY_DIRS[round_id])
    fig, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(20, 8))
    for ax, recipe_id in [(ax_p1, "p1_..."), (ax_p2, "p2_...")]:
        rows = sorted([r for r in summary if r["recipe_id"] == recipe_id],
                      key=lambda r: r["alpha"])
        ax.plot([r["alpha"] for r in rows], [r["success_rate"] for r in rows],
                "o-", color="steelblue", label="SR")
        ax2 = ax.twinx()
        ax2.plot([r["alpha"] for r in rows], [r["inf"] for r in rows],
                 "s--", color="crimson", label="inf")
        # phase3 anchors
        ax.axhline(0.95 if recipe_id.startswith("p1") else 0.96,
                   color="gray", linestyle=":", label="phase3 anchor")
        ax.set_xlabel("α (offline weight share)")
        ax.set_ylabel("SR")
        ax2.set_ylabel("inference_ratio")
        ax.set_title(recipe_id)
    plt.savefig(DATA_ROOT.parent.parent / "analysis" / "phase4_alpha_sweep.png")
```

#### §4.4.2 `analysis/plot_pareto_phase4.py`（~150 LOC）

复用 `plot_pareto_phase3._load_random_periodic` 与 `pareto_upper_frontier`；新增加载 phase4 round 1-3 的 winner cells，并在 phase3 (faded gray dots) 上叠 phase4 (colored markers per round)。

#### §4.4.3 `analysis/phase4_results.md` 章节模板

```
# Verdict Factor Judge — Phase 4 结果分析
## §0 公式 / baseline 同 phase3
## §1 实验目的
## §2 R1 α 扫描结果（14 cell 详表 + α 曲线）
   §2.1 G_R1 决议（日期，α* = ...）
## §3 R2 offline desc 结果（18 cell 详表）
   §3.1 G_R2 决议
## §4 R3 online desc 结果（10 cell 详表）
   §4.1 G_R3 → R4 决议
## §5 R4 window 结果（条件触发；写"未触发"或 10 cell 详表）
## §6 综合 winner / Pareto 比较
## §7 数据局限 + phase 5 候选
## §8 文件索引
```

### §4.5 调用图（**4-mode 拆分 + preload-before-load 顺序**，与 §4.2.1 / §4.2.3 / §4.2.3b 对齐）

```
─── R1 launch sequence ───────────────────────────────────────────────────────
[user] run_phase4 --mode emit-warmup-yaml --round 1
   ├── _check_pkl_keys (§5.0)
   ├── for rid in {p1, p2}:
   │      build_phase4_warmup_yaml(cfg_id, rid) ─► v2_spec.build_warmup_yaml
   │      save_yaml(...) ─► config/spatial16/phase4/warmup/<rid>__warmup.yaml
   └── return  (NO server, NO ctl)

[user] run_phase4 --mode run-warmup --round 1 --serve-host <IP> --serve-port <PORT>
   ├── ctl = WebsocketClientPolicy(host, port)
   ├── _preload_pkl(ctl, PKL_PATH)
   ├── for rid in {p1, p2}:
   │      ctl.load_cache_config(warmup_yaml_text, yaml_id=warmup_yaml_id(rid))
   │      _run_eval_loop(ctl, rid, args)        # writes server-side dump
   │      _extract_finite_factor_raw(...)        # writes data/phase4/warmup_factor_raw/<rid>.jsonl
   └── return

[user] run_phase4 --mode emit-eval-yamls --round 1     # NO --alpha-star (R1 does not need one)
   ├── _build_cell_list(args)  ─► generate_r1_weights × 14
   ├── for cell in cells:
   │      _solve_thresholds(cell)
   │           ├── load_per_key_finite_history(factor_raw.jsonl, ...)
   │           ├── reconstruct_scores(jsonl_path, recipe, composer_weights=cell.weights)  ⬅ NEW kw
   │           └── derive_thresholds(scores, fh_ratio=cell.fh_ratio, ws_ratio=cell.ws_ratio)
   │      build_phase4_eval_yaml(cfg_id, recipe, fh/ws_thr, fh/ws_ratio, weights)
   │           └── v2_spec.build_eval_yaml(cfg_id, factors=r["factors"], composer=...)   ⬅ phase3 RECIPES untouched
   │      save_yaml ─► config/spatial16/phase4/eval/<cell.yaml_id>.yaml
   └── return  (NO server, NO ctl)

[user] run_phase4 --mode run-eval --round 1 --serve-host <IP> --serve-port <PORT>
   ├── ctl + _preload_pkl
   ├── for recipe in {p1, p2}:
   │      for cell in recipe_cells:
   │          ctl.preload_normalizer_buffer(cell.yaml_id, buffer)   ⬅ §4.2.3b INVARIANT (phase3 line 380-382)
   │          ctl.load_cache_config(cell.yaml_path.read_text(), yaml_id=cell.yaml_id)
   │          _run_eval_loop(ctl, cell, args)
   │          _append_summary_row(summary_path, cell, eval_results)
   │      _recipe_end_cleanup(ctl, ...)
   ├── _dump_decision_gate_table(args.round, summary_path)   ⬅ §4.2.7 round-specific
   └── return

─── User reads decision_gate.json → picks per-recipe α* ───────────────────────
$ cat data/phase4/r1_alpha/decision_gate.json | jq .next_args_suggestion
{
  "alpha-star": "p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6",
  "offline-pattern": null,
  "online-pattern": null
}

─── R2 launch sequence (factor_raw cache hit, no re-warmup) ───────────────────
[user] run_phase4 --mode emit-eval-yamls --round 2 \
       --alpha-star "p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6"
   ├── _build_cell_list(args)  ─► generate_r2_weights × 18 (per-recipe alpha lookup)
   ├── for cell in cells:
   │      assert factor_raw/<rid>.jsonl exists       # cached from R1's run-warmup
   │      _solve_thresholds(cell)
   │      build_phase4_eval_yaml(...)
   │      save_yaml ─► .../eval/<cell.yaml_id>.yaml
   └── return

[user] run_phase4 --mode run-eval --round 2 \
       --alpha-star "p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6" \
       --serve-host <IP> --serve-port <PORT>
   └── (same shape as R1 run-eval; preload-before-load on every cell)

─── R3 / R4 — analogous, with per-recipe --offline-pattern / --online-pattern ─
```

> **修订理由（G1 R2 Blocking 4）**：R1 修订调用图仍写 `build_eval_yaml_for_round_cell` + 直接 `ctl.load_cache_config` 缺少 preload；CLI example 仍是单值 `--alpha-star 0.4`。新版调用图把 preload-before-load invariant 显式画出来 + CLI example 全部用 per-recipe mapping。

### §4.6 与 phase3 的边界（修订后 — G1 R2 后保持 phase3_spec 不动）

| phase3 文件 | phase4 是否改？ | 改动方式 / 理由 |
|---|:---:|---|
| **`src/openpi/cache/components/factors/composers/__init__.py`** | ✅ 改 | `WeightedSumZeroNanComposer._score_only` 从等权平均改成 `Σ w_k · contrib_k / Σ w_k`；零权 key 跳过；NaN raw 仍贡献 0 但保留分母权重。`compose()` 路径不变。同步更新该类 / `_score_only` 方法的 docstring 与 module-level 注释 — 旧"Equal-weight sum"描述全替换。**向后兼容**：当所有 weights 相等（如 phase3 的 `{k:1.0 for k}`）时，post-phase4 score 数值上等于 pre-phase4。 |
| `src/openpi/cache/config.py` | ❌ | composer schema 不变（per-key float 已支持） |
| **`exp/verdict_factor_judge/phase3_spec.py`** | ❌（**G1 R2 Blocking 2 修订** — 不再改） | phase4 通过直接调 `v2_spec.build_eval_yaml(cfg_id, factors, composer)` 与 `v2_spec.build_warmup_yaml(...)` 自己构造 yaml；不污染 `phase3_spec.RECIPES`，不扩 `build_eval_yaml_for_cell` 签名 |
| **`exp/verdict_factor_judge/phase3_threshold_solver.py`** | ✅ 改 | `reconstruct_scores` 加 keyword-only `composer_weights: dict[str, float] \| None = None`；None 时维持等权（向后兼容 phase3） |
| `exp/verdict_factor_judge/run_phase3.py` | ❌ | run_phase4 是独立 runner |
| `exp/verdict_factor_judge/analysis/plot_pareto_phase3.py` | ❌（仅 import frontier helper） | |
| **既有 phase 3 测试**：`tests/cache/components/factors/test_composer_zero_nan.py` + `tests/exp/test_phase3_threshold_solver.py` | ✅ 改 | 加 numeric weight sensitivity tests（§4.3.0 / §4.3.0b）；保留 / 改名旧"等权"测试为"backward compat uniform weights"；测试路径修正自 R1 误写（既有文件路径在 `tests/cache/components/factors/`，不是 `tests/cache/`） |

> **回归保证**：phase3 的 11 recipe × 16 cell 实验若用同款 yaml 重跑（weights all=1.0），post-phase4 composer/solver 输出**逐数值一致**（`test_score_only_backward_compat_uniform_weights_match_phase3_behavior` 覆盖）。同时 phase3 RECIPES 在 phase4 import 后仍是 11 个 key（`test_phase4_import_does_not_mutate_phase3_recipes` 覆盖）。这意味着 phase3 results.md 的 baseline 数据无需重跑即可继续作为 phase4 比较锚。

---

## §5 运行命令（草图，G2 后另出 6-batch 教程 `verdict_phase4_run_commands.log.md`）

> 注：所有命令明确 `--mode {emit-warmup-yaml | run-warmup | emit-eval-yamls | run-eval}`。Mode 之间是**强顺序**：emit-warmup → run-warmup → emit-eval-yamls(round=N) → run-eval(round=N) → 看 decision_gate.json → emit-eval-yamls(round=N+1) → run-eval(round=N+1) → ...

### §5.0 Pre-flight

```bash
# 验证 pkl 含 16 个 W-FUT key（p1+p2 offline）
uv run python -m exp.verdict_factor_judge.scripts.verify_phase4_pkl_keys
```

### §5.1 一次性 warmup（pre-R1，所有轮共用）

```bash
# Step 1: 写 warmup yaml（无 server）
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode emit-warmup-yaml --round 1

# Step 2: 启 server（新进程，不继承 phase3 状态）
... （servers/cache_server.py 命令 — 详见 run_commands log）

# Step 3: 跑 warmup（要 server）— 写 data/phase4/warmup_factor_raw/{p1,p2}.jsonl
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-warmup --round 1 \
    --serve-host <IP> --serve-port <PORT>
```

### §5.2 Round 1（α 扫描，14 eval cell）

```bash
# Step 4: 写 14 eval yaml（无 server，调 solver 算 thr）
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode emit-eval-yamls --round 1

# Step 5: 跑 R1 eval（要 server）
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 1 \
    --serve-host <IP> --serve-port <PORT>

# Step 6: 画 α 曲线 + 看 decision_gate.json 选 α*
MPLBACKEND=Agg uv run python -m exp.verdict_factor_judge.analysis.plot_alpha_sweep_phase4 --round 1
cat exp/verdict_factor_judge/data/phase4/r1_alpha/decision_gate.json | jq .next_args_suggestion
# Output e.g.:
# {"alpha-star": "p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6", "offline-pattern": null, "online-pattern": null}
```

### §5.3 Round 2（offline 4-desc，18 eval cell）

```bash
# Step 7: 直接复用 decision_gate.json 给的 next_args_suggestion["alpha-star"]
ALPHA_STAR='p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode emit-eval-yamls --round 2 --alpha-star "$ALPHA_STAR"

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 2 --alpha-star "$ALPHA_STAR" \
    --serve-host <IP> --serve-port <PORT>

# 看 decision_gate.json 选 offline pattern*
cat .../r2_offline_desc/decision_gate.json | jq .next_args_suggestion
```

### §5.4 Round 3（online 2-desc，10 eval cell）

```bash
# 用 R2 给的 offline-pattern + R1 的 alpha-star
ALPHA_STAR='p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6'
OFF_PAT='p1_state_fut_online_act=jerk-heavy,p2_action_fut_online_act=uniform'

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode emit-eval-yamls --round 3 \
    --alpha-star "$ALPHA_STAR" --offline-pattern "$OFF_PAT"

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 3 \
    --alpha-star "$ALPHA_STAR" --offline-pattern "$OFF_PAT" \
    --serve-host <IP> --serve-port <PORT>
```

### §5.5 Round 4（条件触发；可能仅 1 recipe）

```bash
# 仅当 R3 decision_gate.json trigger_decisions[<rid>].continue == true 才跑该 rid
# 例如只有 p1 触发 R4：
uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode emit-eval-yamls --round 4 \
    --alpha-star "$ALPHA_STAR" --offline-pattern "$OFF_PAT" --online-pattern "p1_state_fut_online_act=jerk-heavy" \
    --recipe p1_state_fut_online_act

uv run python -m exp.verdict_factor_judge.run_phase4 \
    --mode run-eval --round 4 \
    --alpha-star "$ALPHA_STAR" --offline-pattern "$OFF_PAT" --online-pattern "p1_state_fut_online_act=jerk-heavy" \
    --recipe p1_state_fut_online_act \
    --serve-host <IP> --serve-port <PORT>
```

### §5.6 终轮 Pareto

```bash
MPLBACKEND=Agg uv run python -m exp.verdict_factor_judge.analysis.plot_pareto_phase4
```

---

## §6 风险

| ID | 风险 | 缓解 |
|---|---|---|
| R-1 | α=0 / α=1 sanity check 与 phase3 已知 SR 不符 | Phase3 anchor SR：g1/g10 (0.5,0.5) 的 SR 为 0.95 / 0.96；R1 端点必须 ±3pp 内复现。否则 = runner / weight-generator bug，停下 debug 不进 R2 |
| R-2 | 融合相对 phase3 winner 没改进（G_R1 fail） | 可接受结局 — phase4_results.md 写"融合无效，保留 g1/g10 winner"。成本：14 cell × 5 min = 70 min，无后续工作 |
| R-3 | Weight grid 错过最优（R2/R3 pattern 太粗） | Pattern 故意离散化。如果 G_R2 选 `disp-heavy`、G_R3 选 `jerk-only`，本身就是值得 phase 5 用细 grid 追的方向。Phase 4 不声明全局最优，只声明方向 |
| R-4 | (FH, WS) = (0.5, 0.5) 不是该锁的 cell | Phase3 显示 g1/g10 跨 16 cell 鲁棒（SR_std < 0.03）。Cell 选择只会平移 α\* 数值，不会消除其存在。若 phase 4 确认 α\* 显著，phase 5 可换 (0.3, 0.5) 复测 |
| R-5 | Online factor (s9, s10) 在高频在线计算路径上意外交互 | s9, s10 phase 3 的 g6 recipe 已实现并测过，无新代码路径 |
| R-6 | 轮间 server 重启导致 WarmupPool 清空 — 必须重新 warm | 缓解：每轮 runner 启动时把 p1.warmup + p2.warmup 作为前 2 cell（幂等）。`factor_raw` jsonl 缓存在 `data/phase4/warmup_factor_raw/<recipe>.jsonl`，solver 从缓存读，不依赖热 warmup buffer |
| R-7 | R4 触发后 yaml 数膨胀 | R4 是条件触发（G_R3）。最坏：14 + 18 + 10 + 10 = 52 cell，6 server 60-70 min wall-clock。可接受 |
| R-8 | composer src/ 改动破坏 phase3 既有 yaml 行为 | §4.3.0 加 `test_score_only_backward_compat_uniform_weights_match_phase3_behavior` 锁住"weights 全等 → 输出与改前一致"的不变量；§4.6 的回归保证表明 phase3 yaml 无需重跑。Code 阶段 G2 时由独立 reviewer 核 phase3 single-cell smoke run（任选一个 phase3 yaml 用新 composer 跑 5 ep，与 phase3 历史 per_step jsonl 同 row 比较 score 数值） |
| R-9 | per-recipe locked cell 让 p1 / p2 直接 (inf, SR) 不可比 | 我们对比的从来不是 p1 vs p2，而是同 recipe 内 weight 变化（§3 决策门均 per-recipe）。Pareto 总图 (§4.4.2) 用 phase3 baseline cloud 作为公共 reference frame，p1/p2 在该 frame 上可视化但不直接互相对比 |
| R-10 | phase4 import 意外 mutate phase3_spec.RECIPES（G1 R2 Blocking 2 隐含风险） | 设计上 phase4_spec 完全本地构造 RECIPES_PHASE4，从 v2_spec.build_eval_yaml 接 composer dict；不 import / update phase3 RECIPES。`tests/exp/test_phase4_spec.py::test_phase4_import_does_not_mutate_phase3_recipes` 强制（§4.3.2）。G2 reviewer 在 code 阶段独立 grep `phase3_spec.RECIPES.update` 确认无该字符串 |
| R-11 | 4-mode CLI 顺序错误（用户跳过 emit-warmup 直接 run-warmup） | main 入口的 invariant assert（§4.2.1）逐 mode 检查前置文件是否存在，缺则 fail-fast 报错指出该跑哪个 mode；`tests/exp/test_phase4_runner.py` 4 个 dispatch 测试覆盖 |
| R-12 | decision_gate.json 的 `next_args_suggestion` 与下一轮 CLI 表达不匹配 | round-trip 测试 `test_decision_gate_next_args_round_trip`（§4.3.2）：把 next_args_suggestion 通过 `_parse_per_recipe_map` 解析 → 喂给 `_build_cell_list` → 检查产生有效 Cell 列表无 KeyError。CI 强制 |

---

## §7 验收标准

- [ ] §4.1 `phase4_spec.py` 4 个生成器（R1/R2/R3/R4）产生的 weight 都满足 ∑=1.0 不变量；权重不变量 100% test 覆盖
- [ ] §4.2 `run_phase4.py` 中途断点能 resume；warmup-once 强制生效
- [ ] §4.3.0 composer numeric weight sensitivity 测试通过（不同非零 weight → 不同 score）
- [ ] §4.3.0b solver `composer_weights` passthrough 测试通过
- [ ] §4.3.2 preload-before-load 顺序 unit 测试通过
- [ ] §4.6 backward compat 测试通过（weights 全等 → 输出逐数值与 phase3 历史一致）
- [ ] §4.3.2 phase3 RECIPES 不变性测试通过（phase4 import 后 `phase3_spec.RECIPES` 仍 11 key）
- [ ] §4.2.7 round-specific decision_gate 4 个测试通过（R1 argmax score / R2 forced-uniform / R3 R4-trigger flag / next-args round-trip）
- [ ] §4.2.1 4-mode CLI invariant 测试通过（每 mode 前置检查正确）
- [ ] §0.1 composer docstring 已更新（grep 旧 "Equal-weight sum" 字符串无残留）
- [ ] R1 端点 sanity：
    - p1 α=1.0 在 (0.5, 0.5) cell 复现 phase3 g1 SR=0.95 ±3pp
    - p1 α=0.0 在 (0.5, 0.5) cell 复现 phase3 g6 同 cell SR ±3pp
    - p2 α=1.0 在 **(0.5, 0.4)** cell 复现 phase3 g10 SR=0.96 ±3pp
    - p2 α=0.0 在 (0.5, 0.4) cell 复现 phase3 g6 同 cell SR ±3pp
- [ ] 每轮 `per_yaml_summary.jsonl` 0 NA 行，cell 数与 §2 预期一致
- [ ] `phase4_results.md` 每跑过的轮都有一节，gate 决议含日期记录
- [ ] phase4 vs phase3 Pareto 叠加图已存
- [ ] `logs/README.md` 已添加本 plan 条目（active table）

---

## §8 设计权衡

1. **score 归一化**：weight 全 10 score 总和 = 1.0；composer 修订后的 `_score_only = Σ w_k · contrib_k / Σ w_k`（§0.1），自动按 active 权重归一（NaN raw 仍贡献 0，但权重保留分母 — zero-NaN 语义）。"α + (1−α) = 1" 是基础不变量（§4.1.8 INV-2）；不需要 per-verdict 再归一化（与 phase 3 zero_nan 行为一致）。
2. **决策门 ΔSR = 2pp**：100 ep × 1 seed 下约 1σ；R2/R3 用 "argmax SR > uniform.SR + 2pp 才选 argmax，否则强制 uniform"（§3.2 / §4.2.7），平衡了"识别真信号"与"避免噪声追逐"。若 phase 5 升到 1000 ep × 1 seed（噪声压到 ~1pp），可收紧到 1pp。
3. **R2 / R3 pattern 集合**：保持 minimal（R2 9 pattern，R3 5 pattern）。如果 R2 winner 是 `disp-heavy` 或 `path-heavy`（非 jerk 主导），phase 5 可加 `(2,2,1,1)` 等联合 heavy pattern 做精调；本 phase 不预投入。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-05-08 16:34 CDT

**Scope reviewed**
- Read G1-approved plan and prior Review Log context under Review Authority.
- Reviewed implementation deltas in `src/openpi/cache/components/factors/composers/__init__.py`, `exp/verdict_factor_judge/phase3_threshold_solver.py`, new `phase4_spec.py`, new `run_phase4.py`, new Phase4 analysis scripts, and new/modified tests.
- Excluded pre-existing Phase3 plot/data artifacts from the Phase4 G2 scope.

**Blocking findings**
1. `run_phase4.py` does not guarantee per-step log collection, so the R1 decision gate can silently use `inf=0` for every cell under the documented/default command path. Evidence: `Args.per_step_log_dir` and argparse default are `""` (`run_phase4.py` lines 135, 163); `_run_one_cell` passes that directly into `_summarize_per_step_log` (`run_phase4.py` line 638); `_summarize_per_step_log` returns zero counts when the dir is empty (`run_phase.py` lines 338-352); `_compute_inf` then returns `0.0` when `n_eval_verdicts` is zero (`run_phase4.py` lines 697-705). This breaks §3.1 / §4.2.7 because R1 is specified as `argmax(SR - 0.5*inf)`, and also contradicts §4.2.2 / §4.2.1 where `PER_STEP_DIRS` and `r{N}_*/per_step/*.jsonl` are planned outputs. Required revision: give `run-eval` a round-specific default per-step log dir (or fail fast unless one is supplied), pass it to `_build_libero_argv`, and add a regression test that the default run-eval path produces nonzero-count-capable summaries / does not allow a decision gate with missing per-step data.
2. `decision_gate.json.next_args_suggestion` is not gate-safe for per-recipe aborts and conditional R4. R1 currently emits `alpha-star` for all winners even when `trigger_decisions[rid]["continue"]` is false (`run_phase4.py` lines 839-845), so copy-pasting the suggestion can continue a recipe §3.1 says must abort. R3 emits `online-pattern` only for triggered recipes but leaves `alpha-star` / `offline-pattern` as all winners and does not include the required `--recipe` restriction (`run_phase4.py` lines 890-900), so the suggested arguments are not a complete round-trip for single-recipe R4. Required revision: make `next_args_suggestion` include only active/continued recipes and include an explicit recipe/active-recipes field or command string; add tests for R1 one-pass/one-fail and R3 one-trigger/one-skip cases.

**Non-blocking concerns**
- The plan/run examples still use `--serve-host` / `--serve-port`, while the implementation exposes `--host` / `--port`; align docs or add aliases before handing commands to operators.
- `src/openpi/cache/config.py` still describes `weighted_sum_zero_nan` as "equal-weight" in comments/docstrings, despite the runtime composer now being true weighted sum.
- `logs/README.md` status text still says §4 Code not started and mentions `phase3_spec.build_eval_yaml_for_cell` receiving `composer_weights`; both are stale relative to the implemented G1-approved design.

**Verification**
- `PYTHONPATH=. uv run pytest tests/cache/components/factors/test_composer_zero_nan.py tests/exp/test_phase3_threshold_solver.py tests/exp/test_phase4_spec.py tests/exp/test_phase4_runner.py -q` → 165 passed, 1 warning. First sandboxed run failed because uv could not write `/home/weiland/.cache/uv`; reran with approved escalation.
- Static grep found no `phase3_spec.RECIPES.update` mutation in Phase4 implementation.

Final verdict: NEEDS REVISION. Code is not approved for G2 until the blocking runner / decision-gate issues above are resolved.

### G2 Round 2 — Executor — 2026-05-08

- Accepted (Blocking 1) — `--mode run-eval` 默认 per-step 路径 + 决策门 fail-fast。事实查证：`Args.per_step_log_dir` 默认空 (`run_phase4.py:135`) -> `_summarize_per_step_log` 返回 zero counts (`run_phase.py:351`) -> `_compute_inf` 返回 0.0 (`run_phase4.py:697`) -> R1 `argmax(SR - 0.5*inf)` 退化为 `argmax(SR)`。修订：(a) 新增 `_apply_default_data_paths(args)` helper，仅在 `--mode run-eval` 自动把空 `per_step_log_dir` / `episode_results_dir` 填成 `_round_data_dir(round) / per_step` 与 `/ episode_results`（plan §4.2.2 layout）；显式传值时不覆盖 (`run_phase4.py:_apply_default_data_paths`)；(b) `main` 在 `_validate_cli_invariants` 后调 `_apply_default_data_paths`；(c) `_dump_decision_gate_table` 加 fail-fast：所有 row 的 `n_eval_verdicts == 0` 时抛 `RuntimeError` 指示 per_step 数据丢失，杜绝静默退化。tests 新增：`test_run_eval_per_step_log_dir_defaults_to_round_specific_path`、`test_run_eval_per_step_log_dir_respects_explicit_value`、`test_apply_default_data_paths_skips_non_run_eval_modes`、`test_decision_gate_raises_when_all_rows_have_zero_verdicts`（共 4 个测试覆盖默认填充 + 显式值不覆盖 + 非 run-eval 模式不填 + fail-fast 守护）。
- Accepted (Blocking 2) — `next_args_suggestion` gate-safe + R3 含 `--recipe`。修订：(a) R1/R2 的 `alpha-star`、`offline-pattern` 现在仅包含 `trigger_decisions[rid]["continue"] == True` 的 recipe（`active_winners` 字典），aborted recipe 不再出现在下一轮 CLI suggestion 里；(b) R3 的 `alpha-star` / `offline-pattern` / `online-pattern` 三个字段全部限制到 `triggered_winners`（continue=True 的 recipe），与 plan §3.3 R4 per-recipe 触发语义一致；(c) 新增 helper `_populate_recipe_restriction_fields` 把 `active_recipes`（active recipe 列表）+ `recipe`（单 recipe 时 = 该 rid，多 recipe 时 = None）+ `cli_command`（含 `--mode emit-eval-yamls --round N` + 必要参数 + 单 recipe 时的 `--recipe <rid>`）写入 `next_args_suggestion`，确保单 recipe R4 / R1 一 pass 一 fail 等场景都能直接 paste 运行；(d) tests 新增：`test_decision_gate_r1_next_args_excludes_aborted_recipe`（R1 一 pass 一 fail，aborted recipe 不在 alpha-star，single-recipe cli_command 含 `--recipe`）、`test_decision_gate_r3_next_args_only_triggered_recipes`（R3 一 trigger 一 skip，三个 pattern 字段都仅含 triggered recipe，cli_command 含 `--round 4 --recipe <rid>`）。
- Accepted (NB-1) — `--serve-host` / `--serve-port` CLI alias。修订：argparse 通过 `add_argument("--host", "--serve-host", dest="host", ...)` 同时接受两种写法（plan §5 与 phase3 风格），新增 `test_serve_host_serve_port_aliases_accepted` 测试覆盖两条解析路径。
- Accepted (NB-2) — `src/openpi/cache/config.py` "equal-weight" 字串清理。修订：`config.py:116-122` 类型枚举 docstring 把 "equal-weight, NaN keys contribute 0 with fixed denominator" 改成 "weighted sum Sum(w_k · contrib_k) / Sum(w_k) over keys with non-zero weight; NaN raw contributes 0 to the numerator but retains its weight in the denominator"；`config.py:2260-2265` factory 内嵌注释同步替换为"weighted sum ... NaN raw → 0 numerator with weight retained"，与 src/composer 改动后语义一致。tests/cache 731 PASS 无回归。
- Accepted (NB-3) — `logs/README.md` 状态行同步。修订：`In Progress` (G1 APPROVED R3 — 2026-05-08; §4 Code 待启动) → `In Progress` (G1 APPROVED R3 / §4 Code 完成 / G2 进行中 — 2026-05-08)；删除"`phase3_spec.build_eval_yaml_for_cell` 加 composer_weights" 错误描述（G1 R2 修订已撤销该接口扩展，phase3_spec.py 不再改动），改写为"phase3_spec.py 不动 — phase4 直接调 v2_spec.build_eval_yaml/build_warmup_yaml，phase3 RECIPES 全程不被 mutate（G1 R2 Blocking 2 修订）"；测试数从 4 个测试文件 → 5 个，165 → 172 tests pass。

修订后等 G2 Round 2 reviewer 复审。

### G2 Round 2 — Reviewer — APPROVED — 2026-05-08 19:29 CDT

**Scope reviewed**
- Re-reviewed the delta since G2 Round 1: `run_phase4.py`, `tests/exp/test_phase4_runner.py`, `src/openpi/cache/config.py`, `logs/README.md`, and executor response above.
- Rechecked the Phase4 implementation for forbidden `phase3_spec.RECIPES` mutation / `build_eval_yaml_for_cell` coupling.

**Blocking findings**
- None. G2 Round 1 Blocking 1 is resolved: `run-eval` now fills round-specific default `per_step` / `episode_results` paths, passes the filled args into `_build_libero_argv`, and `decision_gate` fails fast on all-zero verdict-count summaries.
- G2 Round 1 Blocking 2 is resolved: `next_args_suggestion` now excludes aborted / non-triggered recipes and emits `active_recipes`, `recipe`, and `cli_command` for single-recipe continuation / R4 paths.

**Non-blocking residuals**
- `decision_gate` still only fail-fasts when every row has `n_eval_verdicts=0`; a mixed summary with one missing per-step file would still compute that row's `inf` as 0. This is acceptable for G2 because the default run path now enables per-step logging, but a future hardening pass should reject any completed eval row with zero verdict counts.
- Plan body still contains historical `--serve-host` examples and old reviewer text mentioning "Equal-weight" as quoted prior findings. Runtime now accepts `--serve-host` / `--serve-port` aliases and live config comments are updated, so this is documentation residue only.

**Verification**
- `PYTHONPATH=. uv run pytest tests/cache/components/factors/test_composer_zero_nan.py tests/exp/test_phase3_threshold_solver.py tests/exp/test_phase4_spec.py tests/exp/test_phase4_runner.py -q` -> 172 passed, 1 warning.
- `rg "RECIPES\.update|phase3_spec\.RECIPES\.update|P3\.update|build_eval_yaml_for_cell\(" exp/verdict_factor_judge/phase4_spec.py exp/verdict_factor_judge/run_phase4.py exp/verdict_factor_judge/phase3_threshold_solver.py` -> no matches.

Final verdict: APPROVED. G2 approved; code may proceed to run-command / execution stage.
