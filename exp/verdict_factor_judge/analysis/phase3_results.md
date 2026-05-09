# Verdict Factor Judge — Phase 3 结果分析

数据来源：`exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl`（176 行，0 NA）+ `episode_results/*.json`（11 recipe × 16 cell × 100 ep eval）。
画图脚本：[`plot_pareto_phase3.py`](plot_pareto_phase3.py) → [`phase3_pareto.png`](phase3_pareto.png)（22×14, 1 张总 Pareto 图）；[`plot_3d_bars_phase3.py`](plot_3d_bars_phase3.py) → [`phase3_3d_bars.png`](phase3_3d_bars.png)（40×30, 11 个 recipe-level (FH, WS) 3D 柱图，颜色 = 实际 warm 触发率）。
inf_ratio 公式：见 §0；与 phase2 layer1 不同点是 phase3 的 warm 全部 fire 在 `start_t=0.5`。

---

## 0. inference_ratio 公式

```
miss          → 1.0
full_hit      → 0.0
warm_start@t  → 1 − 0.5 · (1 − t)
```

Phase 3 全部 11 recipes 锁 `composer.warm_start_t = 0.50`（plan §9） → **每 warm verdict 贡献 0.75**（phase2 是 0.85）。

```
inf_ratio = (0·n_full_hit + 0.75·n_warm_start + 1.0·n_miss) / n_eval_verdicts
```

---

## 1. 实验目的与方法

Phase 2 layer 2 在 spatial16 上锁定了 6 个候选 recipe（T-DUAL / start_t / 单 vs 双因子组合），但 thr 仍是手设，**任意 (fh_ratio, ws_ratio) cell 究竟解多少 thr 才合理是黑箱**。Phase 3 改成 **data-driven threshold sweep**：

1. **Warmup 跑出 factor_raw 分布**：每 recipe 一份 warmup yaml，跑 100 ep，把每个 factor 的 raw 输出按 row order finite-only 写入 jsonl。
2. **逆解 thr**：用 `phase3_threshold_solver` 把 (fh_ratio, ws_ratio) 视为目标分位 → 在 warmup factor_raw 的 calibrated score 上算分位数 → 得到 (fh_thr, ws_thr)。
3. **Eval 16 cell**：每 recipe 16 个 cell（fh_ratio × ws_ratio ∈ {0.2, 0.3, 0.4, 0.5}²），用 step 2 解出来的 thr 跑 eval（每 cell ≈ 100 ep）。
4. 跨 11 recipes × 16 cells = **176 eval 点**，统一在 (inf, SR) 平面上比较。

11 recipes 来自 phase2 layer1 / layer2 winner 的扩展（详见 [`phase3_spec.py`](../phase3_spec.py)）：

| recipe_id | factor 配置 | 主要风险/特点 |
|---|---|---|
| g1 | F1b-T × W-FUT × all-4 desc | layer1 spatial16 唯一 "非 NaN-fallback" 突破 |
| g2 | F1b-T × W-LONG-RISK × jerk | layer1 跨 3 cfg 冠军；T median=21 → 长窗高 NaN |
| g3 | F1b-T × W-LONG-RISK × all-4 desc | g2 全 desc 版 |
| g4 | F1b-T × W-SHORT × jerk | layer1 spatial16 inf 最低 winner |
| g5 | F1a-T × jerk + dir pair | F1a 路线 |
| g6 | F1a-A × jerk + curv pair | F1a-A 路线 |
| g7 | F1b-A × W-LONG-RISK × jerk | g2 的 action 域镜像 |
| g8 | F1a-T × curv_only | F1a-T 单 desc 退化 ablation |
| g9 | F1b-T × W-SYM-S × all-4 desc | 对称小窗 |
| g10 | F1b-A × W-FUT × all-4 desc | g1 的 action 域镜像 |
| g11 | F1a-A × curv_only | F1a-A 单 desc 退化 ablation |

Composer 全部用新建的 **`weighted_sum_zero_nan`**（NaN→0，固定分母 N，强制双 tier）：[`composers/__init__.py`](../../../src/openpi/cache/components/factors/composers/__init__.py)。

---

## 2. baseline 对照（spatial16）

| baseline | 来源 | (inf, SR) |
|---|---|---|
| plain inference | warm_start (500 ep) | (1.00, 0.984) |
| always-WARM @ t=0.7 | warm_start (500 ep) | (0.85, 0.976) |
| always-WARM @ t=0.5 | warm_start (500 ep) | **(0.75, 0.952)** ← 与 phase3 warm cost 同点 |
| always-WARM @ t=0.3 | warm_start (500 ep) | (0.65, 0.942) |
| always-FULL_HIT | warm_start (100 ep) | **(0.00, 0.650)** ← 实测，非 phase2 layer1 doc 写的 0.692 |
| random / periodic | random_periodic_gate | 78 散点，前沿在图上 |

> ⚠ **`always-FULL_HIT` baseline 修正**：phase2_layer1_results.md 写的 spatial16 SR=0.692 是 stale 数。Phase 3 重测 100 ep 实测 0.650。phase3 全 FH 退化 cell 的 SR 落在 [0.52, 0.96]，跨 recipe 中位数 0.61，与 0.650 在 100 ep × 1 seed 噪声内。

---

## 3. 11 recipes × 16 cells 主表

格式：`SR / inf / hit·warm·miss% / (fh_thr, ws_thr)`。

### 3.1 Recipe-level summary

| recipe | cells | SR min / med / max | inf min | warm% med | thr=0 退化 (fh/ws) |
|---|---:|:---:|---:|---:|:---:|
| g1 f1b-T W-FUT × all | 16 | 0.90 / 0.97 / **1.00** | 0.37 | 37% | 0 / 3 |
| g2 f1b-T W-LONG-RISK × jerk | 16 | 0.72 / 0.84 / 0.96 | **0.00** | 65% | 4 / 15 |
| g3 f1b-T W-LONG-RISK × all | 16 | 0.74 / 0.91 / 0.97 | 0.42 | 63% | 0 / 13 |
| g4 f1b-T W-SHORT × jerk | 16 | 0.83 / 0.92 / 0.99 | 0.43 | 41% | 0 / 0 |
| g5 f1a-T jerk+dir | 16 | 0.75 / 0.92 / 0.98 | 0.40 | 40% | 0 / 6 |
| g6 f1a-A jerk+curv | 16 | 0.86 / 0.92 / 0.96 | 0.40 | 54% | 0 / 10 |
| g7 f1b-A W-LONG-RISK × jerk | 16 | 0.80 / 0.90 / 0.96 | 0.42 | 65% | 0 / 14 |
| g8 f1a-T curv_only | 16 | **0.52** / 0.88 / 0.93 | 0.00 | 69% | 8 / 16 |
| g9 f1b-T W-SYM-S × all | 16 | 0.79 / 0.93 / 0.99 | 0.42 | 38% | 0 / 3 |
| g10 f1b-A W-FUT × all | 16 | 0.91 / 0.97 / **1.00** | 0.38 | 37% | 0 / 3 |
| g11 f1a-A curv_only | 16 | 0.55 / 0.61 / 0.95 | 0.00 | 0% | **12 / 16** |

### 3.2 每 recipe 最佳 cell（max SR；并列时取 min inf）

| recipe | SR | inf | (fh_ratio, ws_ratio) | (fh_thr, ws_thr) | hit/warm/miss% |
|---|---:|---:|:---:|:---:|:---:|
| g1 | **1.00** | 0.73 | (0.2, 0.3) | (0.603, 0.385) | 19/33/49 |
| g2 | 0.96 | 0.71 | (0.2, 0.2) | (0.410, 0.170) | 25/17/58 |
| g3 | 0.97 | 0.59 | (0.2, 0.4) | (0.412, 0.000) | 21/79/0 |
| g4 | 0.99 | 0.74 | (0.2, 0.3) | (0.627, 0.427) | 18/30/52 |
| g5 | 0.98 | 0.69 | (0.2, 0.4) | (0.660, 0.260) | 21/40/39 |
| g6 | 0.96 | 0.59 | (0.2, 0.5) | (0.660, 0.000) | 21/79/0 |
| g7 | 0.96 | 0.59 | (0.2, 0.3) | (0.420, 0.000) | 21/79/0 |
| g8 | 0.93 | 0.52 | (0.2, 0.5) | (1.000, 0.000) | 31/69/0 |
| g9 | 0.99 | 0.66 | (0.2, 0.5) | (0.597, 0.278) | 21/51/27 |
| g10 | **1.00** | 0.58 | (0.3, 0.5) | (0.515, 0.198) | 29/51/20 |
| g11 | 0.95 | 0.55 | (0.2, 0.5) | (1.000, 0.000) | 27/73/0 |

> 全部 11 recipe **每个都至少有一个 cell 达到 SR ≥ 0.93**。SR=1.00 的 cell 仅出现在 **g1 / g10** （都是 W-FUT 配方）。

---

## 4. 关键发现

### 4.1 Productive winner band（SR ≥ 0.95 且 inf < 0.85）：42 cells

| 子带 | inf 区间 | cells | 代表 cell |
|---|:---:|---:|---|
| ultra-cheap | 0.37 – 0.50 | 5 | g1 fh0.5/ws0.5 (0.95/0.37), g10 fh0.5/ws0.4 (0.96/0.38) |
| cheap | 0.55 – 0.65 | 14 | g10 fh0.3/ws0.5 (**1.00/0.58**), g3 fh0.2/ws0.4 (0.97/0.59), g7 fh0.2/ws0.3 (0.96/0.59) |
| mid | 0.65 – 0.75 | 18 | g1 fh0.2/ws0.4 (0.99/0.70), g4 fh0.2/ws0.3 (0.99/0.74), g5 fh0.2/ws0.4 (0.98/0.69) |
| high-warm | 0.75 – 0.85 | 5 | g3 fh0.2/ws0.2 (0.97/0.78), g7 fh0.2/ws0.2 (0.96/0.79) |

**Top 5 cell（按 inf 升序，SR≥0.95）：**

| inf | SR | recipe | (fh, ws) | hit/warm/miss% |
|---:|---:|---|:---:|:---:|
| **0.37** | 0.95 | g1 | (0.5, 0.5) | 50/50/0 |
| 0.38 | 0.96 | g10 | (0.5, 0.4) | 49/51/0 |
| 0.45 | 0.96 | g10 | (0.4, 0.5) | 40/60/0 |
| 0.46 | 0.95 | g1 | (0.4, 0.5) | 38/62/0 |
| 0.50 | 0.96 | g10 | (0.4, 0.4) | 40/38/21 |

**关键观察**：inf 最低的 5 cell 全部是 **W-FUT 配方（g1 + g10）**，且**几乎没有 miss**（hit + warm ≈ 100%）。这意味着 phase2 layer1 spatial16 上"W-FUT 是唯一非 NaN-fallback 突破"的结论在 phase3 进一步加强：在 data-driven thr 下，W-FUT 在 inf 0.37-0.50 区间仍是**唯一**给出 SR ≥ 0.95 的因子配置。

### 4.2 Trivial gold（inf=0 退化 cell）：24 cells

24 cells 落在 (inf=0, SR ∈ [0.52, 0.96])，其中：

- **g11 全 16 个 cell** + g11 fh0.2 行（4 cells）共 12 退化（thr=0 ratio 12/16）
- **g8 fh ∈ {0.4, 0.5}** 8 cells 退化
- **g2 fh=0.5** 4 cells 退化

退化机制：solver 用 PercentileRollingCalibration 在 warmup score 上找 (1-fh_ratio) 分位作为 fh_thr。当 score 分布在某个值上**饱和**（dispersion factor 退化、curv_radius 在 NaN 后填 0），分位数切到 0 → fh_thr=0 → 全部 verdict 上溯到 full_hit → inf=0。这些 cell 在 Pareto 图上看似 "金圈贴 baseline"，但**不携带 phase3 真实信息**，应当从 winner 候选里排除。

> 后续 Pareto 图建议用 `×` 符号标记 trivial gold，与"真" winner（SR≥0.95 + inf>0）区分。

### 4.3 SR 与 actual warm rate 的关系

```
warm-rate bucket    n   SR mean   SR med    range
─────────────────────────────────────────────────────
   0 - 20%         35    0.721    0.640    [0.52, 0.96]   ← 退化 + 高 FH 主导
  20 - 40%         46    0.936    0.940    [0.81, 1.00]   ← sweet spot
  40 - 60%         41    0.893    0.900    [0.74, 1.00]
  60 - 80%         51    0.896    0.900    [0.80, 0.97]
  80 -100%          3    0.913    0.910    [0.90, 0.93]
```

**SR 曲线呈倒 U：**warm-rate < 20% 区间是退化 + 高 FH 误判堆积（SR 拉到 0.72）；20-40% 区间最高 mean SR (0.936)，那里 hit 信号 + warm 兜底配比最佳；warm-rate > 60% 后 SR 不再上升（warm 已不便宜）。这说明 **phase3 的甜点是 warm 占比 20-40%**，不是越高越好——这与"always-WARM @ t=0.5 SR=0.952"的 baseline 一致：太高 warm 把 inf 拉满 0.75，再无突破。

### 4.4 跨 recipe 系统差：F1b-T W-FUT > F1b-T W-SHORT > F1b-T W-LONG-RISK > F1a 系列 > F1a 单 desc 退化

按 max SR / min inf-at-SR≥0.95 排：

| 排名 | recipe | max SR | best inf @ SR≥0.95 | 综合 |
|:---:|---|---:|---:|---|
| 1 | g10 (F1b-A W-FUT) | 1.00 | 0.38 | **最稳 + 最便宜** |
| 1 | g1 (F1b-T W-FUT) | 1.00 | 0.37 | 同 |
| 3 | g4 (F1b-T W-SHORT × jerk) | 0.99 | 0.74 | 高 SR 但 inf 偏高 |
| 4 | g9 (F1b-T W-SYM-S × all) | 0.99 | 0.66 | mid 区间最强 |
| 5 | g5 (F1a-T jerk+dir) | 0.98 | 0.69 | F1a 系列最佳 |
| 6 | g3 (F1b-T W-LONG-RISK × all) | 0.97 | 0.59 | NaN-warm 主导 |
| 7 | g6 / g7 / g11 | 0.96 / 0.95 | 0.59 / 0.59 / 0.55 | NaN-warm 或退化 |
| 10 | g2 (F1b-T W-LONG-RISK × jerk) | 0.96 | 0.71 | layer1 冠军在 phase3 不再领先 |
| 11 | g8 (F1a-T curv_only) | 0.93 | 0.52 | 最弱 |

**Phase 2 layer1 跨 3 cfg 冠军 g2 在 phase3 上排名第 10**——因为 phase3 把 thr 从 layer1 默认值（NaN-fallback 兜底 50% warm）改成数据驱动，F1b-T × W-LONG-RISK × jerk 的 jerk 信号在主动双 tier 框架下区分度不如 W-FUT。这是 phase3 的核心修正信号：**真正可移植到下游的不是"W-LONG-RISK NaN 兜底大法"，而是 W-FUT 因子下 fh_ratio + ws_ratio 都设较高 (0.4-0.5) 切出来的双 tier**。

---

## 5. Pareto 全局结论

- **Strict-Pareto-positive（不被 r/p + always-WARM 任一点 dominate）**：**67 / 176 cells**。
- 去掉 24 个退化 cell 后，**真 winner = 43 cells**，全部分布在 inf ∈ [0.37, 0.85]、SR ∈ [0.93, 1.00] 的 productive band。
- 与 always-WARM @ t=0.5 baseline（0.75, 0.952）对照：
  - 最便宜 winner cell `g1 fh0.5/ws0.5`：(0.37, 0.95) → **inf 省 0.38（51%），SR 几乎不掉**。
  - 最高 SR winner cell `g10 fh0.3/ws0.5` 与 `g1 fh0.2/ws0.3`：SR=1.00 / inf 0.58 / 0.73 → **inf 省 0.17–0.02，SR 涨 0.05**。

> **Phase 3 核心收益：** 在 always-WARM @ t=0.5 水平上，**两个 W-FUT 配方 (g1 / g10) 把 inf 从 0.75 压到 0.37-0.45 同时维持 SR ≥ 0.95**。这是本项目首次在 spatial16 上**同时**做到"显著省推理 + 不丢 SR"的稳健配置。

---

## 6. 数据局限性

1. **每 cell 100 ep × 1 seed**：SR 标准误约 ±2-3pp，单 cell 之间 ≤ 3pp 的 SR Δ 不可下结论。所有 SR=1.00 的 cell 真值区间是 [0.97, 1.00]。
2. **同 recipe 内 cell 不严格 deterministic**：检查 g11/g8 同 recipe 不同 cell 的 episode 级一致性，发现 19-26/100 ep 不一致（同款 thr=0 退化、同款全 FH 行为，理论上应 100% 一致）。说明 LIBERO sim 在多 worker 异步下 per-episode noise floor ≈ 20%。这为 §4.3 的 SR 低端长尾贡献了一部分。
3. **跨 recipe 全 FH cell SR 系统差 0.13** (g2 0.72 vs g11 0.59)：超 sim noise (~5pp) 但低于经过 1000 ep 复测才能确诊的统计 power。可能机制：长窗 chain walk 触发 InMemoryBackend lazy load 累积副作用 — 未定论。
4. **warmup 与 eval 分布漂移**：solver 用 warmup factor_raw 解的分位 thr，到 eval 上实际 fire 比可能漂离 (fh_ratio, ws_ratio)。3D 柱图的颜色（actual warm rate）就是这种漂移的可视化——多数 recipe 实际 warm 比高于设定 ws_ratio（calibrated score 在 eval 上比 warmup 偏低）。
5. **clip / max_pool 未跑**：phase3 仅 spatial16；外推到 clip / max_pool 需后续实验验证。

---

## 7. Phase 4 决策点

基于 phase3 数据，**推荐 phase 4 / layer 3 复测候选：**

| 优先级 | recipe | cell | 假设 |
|:---:|---|:---:|---|
| ★★★ | g10 (F1b-A W-FUT × all) | (0.3, 0.5) → SR 1.00 / inf 0.58 | **最佳 SR-inf 平衡**，跨 cfg 验证 |
| ★★★ | g1 (F1b-T W-FUT × all) | (0.5, 0.5) → SR 0.95 / inf 0.37 | **最便宜**，验证 ultra-cheap 区间 |
| ★★ | g10 (F1b-A W-FUT × all) | (0.5, 0.4) → SR 0.96 / inf 0.38 | g1 的 action-domain 镜像验证 |
| ★★ | g4 (F1b-T W-SHORT × jerk) | (0.2, 0.3) → SR 0.99 / inf 0.74 | layer1 spatial16 winner 在双 tier 下复测 |
| ★ | g9 (F1b-T W-SYM-S × all) | (0.3, 0.5) → SR 0.96 / inf 0.58 | W-FUT vs W-SYM-S 对照 |

**复测计划：5 cell × 1000 ep × 1 seed = 5000 ep ≈ 1 h（5 server 并跑）。**

复测目标：把 SR 标准误从 ±3pp 压到 ±1pp，确认 g1 / g10 在 SR=1.00 / inf<0.6 上的统计显著性，为下一步 clip / max_pool 扩展挑出真正稳的 recipe。

---

## 8. 文件索引

```
exp/verdict_factor_judge/
├── data/phase3/
│   ├── per_yaml_summary.jsonl              # 176 行（11 recipe × 16 cell）
│   ├── per_step/<yaml_id>.jsonl            # 每 verdict 一行
│   ├── episode_results/<yaml_id>.json      # 每 episode 成功/失败
│   └── warmup_factor_raw/<recipe>.jsonl    # solver 输入：finite-only factor_raw
├── analysis/
│   ├── plot_pareto_phase3.py               # 单张总 Pareto 图
│   ├── plot_3d_bars_phase3.py              # 11 recipes 3D 柱图（FH × WS × SR, color=warm rate）
│   ├── phase3_pareto.png                   # 22×14 Pareto
│   ├── phase3_3d_bars.png                  # 40×30 per-recipe 3D
│   ├── phase3_results.md                   # 本文件
│   ├── phase2_layer1_results.md            # 上一阶段
│   └── phase0_phase1_results.md            # 更早阶段
├── phase3_spec.py                          # 11 recipe 配方 + GRID 定义
├── phase3_threshold_solver.py              # 数据驱动 thr 求解
├── run_phase3.py                           # batch runner
└── config/spatial16/phase3/
    ├── *_phase3_<recipe>__warmup.yaml      # 11 warmup yaml
    └── *_phase3_<recipe>__fh*_ws*.yaml     # 176 eval yaml（每 recipe 16 cell）
```
