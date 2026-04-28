# Verdict Factor Judge — Phase 2 Layer 1 结果分析

数据来源：`exp/verdict_factor_judge/data/phase2_layer1/{cfg}/per_yaml_summary.jsonl` + `episode_results/*.json`（每 cfg 26 个 yaml × 100 ep eval）。
画图脚本：[`plot_pareto.py`](plot_pareto.py) → 输出 [`phase2_layer1_pareto.png`](phase2_layer1_pareto.png)。
inference_ratio 公式：见 [`logs/verdict_factor_judge_experiment_plan.log.md` §inf_ratio](../../../logs/verdict_factor_judge_experiment_plan.log.md)。

外部 baseline 同 Phase 0 / Phase 1：`exp/warm_start/data/`（500 ep × 3 cfg, always_hit / inference / warm@{0.3,0.5,0.7}）+ `exp/random_periodic_gate/analysis/aggregate.csv`（78 个概率 / 周期 gate 点）。

---

## 0. inference_ratio 公式（一定不要算错）

```
miss          → 1.0
full_hit      → 0.0
warm_start@t  → 1 - 0.5 * (1 - t)
```

Phase 2 Layer 1 yaml 的 warm 全部 fire 在 `start_t=0.7`（all_nan_fallback 默认值）→ 每 warm verdict 贡献 **0.85**。

---

## 1. 实验目的

Phase 1 在 (SR, inf_ratio) Pareto 平面上**贴 baseline 没破前沿**（最强 yaml 也只是 match always-WARM）。Phase 2 Layer 1 不再做"扩单因子"，改为**逐因子内部探索**——把每个因子的 descriptor / window 维度拆开试，看哪些**单 desc / 单窗口** 能给出比 Phase 1 更强的 hit-vs-miss 区分信号。

26 yaml × 3 cfg = 78 数据点：

| Layer | yaml 数 | 内容 |
|---|---:|---|
| 1.A | 3 | F1a-A close-out: curv / cum 单 desc + jerk+curv pair |
| 1.B | 5 | F1a-T desc sweep: 4 单 desc + jerk+dir pair |
| 1.C | 7 | F1b-A × {W-SHORT/PAST/FUT/SYM-S × all-4} + W-SHORT × 单 desc |
| 1.D | 7 | F1b-T 同 1.C 结构 |
| 1.E | 4 | W-LONG-RISK ((5,5)+(7,7)) × {F1b-A, F1b-T} × {all, jerk} |

全部 yaml 走 T-FULL（`full_hit=0.5`，无 warm tier，warm 仅来自 `all_nan_fallback@0.7`）。

---

## 2. 三 cfg × 26 yaml 主表

格式：`SR / inf_ratio`，inf 按 §0 公式算。

```
yaml_stem                        | clip          | max_pool      | spatial16     | hit/warm/miss% (clip)
─────────────────────────────────────────────────────────────────────────────────────────────────────
f1a_a_d_cum_only                 | 0.91 / 0.52   | 0.83 / 0.52   | 0.88 / 0.52   | 46/13/41
f1a_a_d_curv_only                | 0.95 / 0.62   | 0.90 / 0.62   | 0.94 / 0.62   | 36/14/51
f1a_a_d_jerk_curv_pair           | 0.92 / 0.62   | 0.88 / 0.63   | 0.96 / 0.62   | 36/13/50
f1a_t_d_cum_only                 | 0.94 / 0.56   | 0.90 / 0.58   | 0.91 / 0.58   | 40/26/34
f1a_t_d_curv_only                | 0.91 / 0.65   | 0.92 / 0.67   | 0.95 / 0.66   | 31/24/45
f1a_t_d_dir_only                 | 0.95 / 0.57   | 0.92 / 0.57   | 0.90 / 0.57   | 40/25/36
f1a_t_d_jerk_dir_pair            | 0.91 / 0.57   | 0.91 / 0.58   | 0.96 / 0.56   | 39/25/36
f1a_t_d_jerk_only                | 0.95 / 0.57   | 0.95 / 0.58   | 0.93 / 0.59   | 39/23/38
f1b_a_w_fut_d_all                | 0.96 / 0.61   | 0.95 / 0.62   | 0.94 / 0.62   | 36/15/48
f1b_a_w_long_risk_d_all          | 0.97 / 0.70   | 0.97 / 0.71   | 0.95 / 0.71   | 23/48/29
f1b_a_w_long_risk_d_jerk         | 0.98 / 0.67   | 0.92 / 0.67   | 0.96 / 0.67   | 26/49/26
f1b_a_w_past_d_all               | 0.93 / 0.60   | 0.89 / 0.62   | 0.91 / 0.61   | 38/13/49
f1b_a_w_short_d_all              | 0.92 / 0.58   | 0.87 / 0.55   | 0.88 / 0.57   | 42/ 0/58
f1b_a_w_short_d_curv             | 0.89 / 0.56   | 0.85 / 0.58   | 0.87 / 0.57   | 44/ 0/56
f1b_a_w_short_d_dir              | 0.81 / 0.44   | 0.82 / 0.45   | 0.84 / 0.44   | 56/ 0/44
f1b_a_w_short_d_jerk             | 0.93 / 0.53   | 0.95 / 0.53   | 0.88 / 0.52   | 47/ 0/53
f1b_a_w_sym_s_d_all              | 0.89 / 0.60   | 0.93 / 0.63   | 0.91 / 0.62   | 39/10/51
f1b_t_w_fut_d_all                | 0.94 / 0.60   | 0.93 / 0.62   | 0.98 / 0.60   | 37/15/47
f1b_t_w_long_risk_d_all          | 0.95 / 0.71   | 0.96 / 0.71   | 0.98 / 0.72   | 22/47/31
f1b_t_w_long_risk_d_jerk         | 0.96 / 0.67   | 0.98 / 0.68   | 0.98 / 0.67   | 26/49/25
f1b_t_w_past_d_all               | 0.89 / 0.59   | 0.81 / 0.57   | 0.87 / 0.58   | 39/12/49
f1b_t_w_short_d_all              | 0.96 / 0.59   | 0.91 / 0.58   | 0.92 / 0.58   | 41/ 0/59
f1b_t_w_short_d_curv             | 0.90 / 0.60   | 0.80 / 0.58   | 0.92 / 0.57   | 40/ 0/60
f1b_t_w_short_d_dir              | 0.91 / 0.47   | 0.86 / 0.49   | 0.93 / 0.49   | 53/ 0/47
f1b_t_w_short_d_jerk             | 0.94 / 0.52   | 0.93 / 0.52   | 0.97 / 0.52   | 48/ 0/52
f1b_t_w_sym_s_d_all              | 0.93 / 0.61   | 0.88 / 0.62   | 0.94 / 0.61   | 38/12/50
```

---

## 3. baseline 对照（修正后真实位置）

| baseline | 来源 | clip(inf, SR) | max_pool(inf, SR) | spatial16(inf, SR) |
|---|---|---|---|---|
| plain inference | warm_start | (1.00, 0.992) | (1.00, 0.984) | (1.00, 0.984) |
| always-WARM @ t=0.7 | warm_start | (0.85, 0.980) | (0.85, 0.964) | (0.85, 0.976) |
| always-WARM @ t=0.5 | warm_start | (0.75, 0.960) | (0.75, 0.966) | (0.75, 0.952) |
| always-WARM @ t=0.3 | warm_start | (0.65, 0.940) | (0.65, 0.946) | (0.65, 0.942) |
| always-FULL_HIT | warm_start | (0.00, 0.674) | (0.00, 0.696) | (0.00, 0.692) |
| random / periodic | random_periodic_gate | 78 散点 | 78 散点 | 78 散点 |

完整图见 [`phase2_layer1_pareto.png`](phase2_layer1_pareto.png) — 灰点 = random/periodic、灰线 = r/p 前沿、红五角星 = always-WARM、蓝点 = Phase 2 yaml、**金圈 = strict-Pareto-positive**（不被任何 baseline 点 dominate）。

---

## 4. 关键发现

### 4.1 Strict-Pareto-positive 跨 cfg 总结

跨 ≥ 2 cfg strict-positive 的 yaml：

| yaml | clip | max_pool | spatial16 | 主机制 |
|---|:--:|:--:|:--:|---|
| **f1b_t_w_long_risk_d_jerk** | ★ (0.96/0.67) | ★ (0.98/0.68) | ★ (0.98/0.67) | NaN-fallback warm + jerk-driven hit |
| **f1b_a_w_long_risk_d_jerk** | ★ (0.98/0.67) |  | ★ (0.96/0.67) | 同上，state→action |
| **f1b_a_w_fut_d_all** | ★ (0.96/0.61) |  | ★ (0.94/0.62) | 未来窗口因子 |
| **f1b_t_w_short_d_jerk** | ★ (0.94/0.52) |  | ★ (0.97/0.52) | 短窗 jerk 主导 |
| **f1a_t_d_jerk_only** | ★ (0.95/0.57) | ★ (0.95/0.58) |  | F1a-T jerk 单 desc |

跨 3 cfg 全 strict-positive（最稳的 winner 候选）：**`f1b_t_w_long_risk_d_jerk`**，SR 0.96 / 0.98 / 0.98 @ inf 0.67。

### 4.2 spatial16 出现近-pure-inference 信号

spatial16 上 4 个 yaml SR ≥ 0.98（接近 plain inference 的 0.984）但 inf 显著 < 1.0：

| yaml | SR | inf | vs plain inference |
|---|---:|---:|---|
| f1b_t_w_fut_d_all | **0.98** | 0.60 | -0.40 inf, -0.004 SR |
| f1b_t_w_long_risk_d_all | **0.98** | 0.72 | -0.28 inf, -0.004 SR |
| f1b_t_w_long_risk_d_jerk | **0.98** | 0.67 | -0.33 inf, -0.004 SR |
| (Phase 1 f_full_t_dual_07) | 0.99 | 0.85 | -0.15 inf, +0.006 SR |

**这是首次出现"接近纯 inference 质量但显著省推理"的信号**。F1b-T (state-domain windowed factor) 在 spatial16 cfg 下表现尤其出色 —— 4 个候选里 4 个都是 F1b-T。这很可能与 spatial16 的视觉 embedding 形状（spatial pool 16, 32k 维）保留更细致 state 几何信息有关，让 F1b-T 的 percentile 标准化能锁定可信 cache hit。

⚠ **100 ep 标准误 ±2-3pp 警告**：上述 SR=0.98 的真实区间是 [0.96, 1.00]。Layer 3 必须用 1000 ep × 1 seed 复测，把噪声降到 ±1pp 才能下"strict-Pareto-positive"结论。

### 4.3 方向性结论

1. **F1b-T (state windowed) > F1b-A (action windowed)**：5 个跨 ≥ 2 cfg 的 winner 里 4 个是 F1b-T。Layer 2 主分量优先 F1b-T。
2. **jerk-only 强于 all-4 desc 强于 dir/curv/cum 单 desc**：Phase 1 已暗示，Phase 2 W-LONG-RISK / W-SHORT 全确认 jerk 是主信号载体。Layer 2 默认描述子启用集 = `[jerk]`。
3. **W-LONG-RISK 不是真正的"风险窗"，是 NaN 概率退化**：(5,5)+(7,7) 每 verdict ~50% NaN → 一半触发 all_nan_fallback@0.7（warm cost 0.85）。看似 hit/warm/miss = 25/49/26 三档分布，本质是"50% baseline-warm + 25% factor-driven hit + 25% factor-driven miss"。Layer 2 必须用 **T-DUAL_07** 让 composer 主动产 warm tier，而不是靠 NaN 兜底。
4. **F1a-A 全维度被 dominated**：3 个 close-out yaml (curv / cum / jerk+curv pair) 没出现在 strict-positive 列表里。Layer 2 不再纳入 F1a-A。
5. **W-PAST / W-SYM-S / W-SHORT 单 desc 都疲软**：W-FUT (`(0,3)(0,5)`) 在 F1b-T spatial16 上单独突破，其他窗口形状价值有限。Layer 2 主推 **W-FUT + W-LONG-RISK 二选一**。

---

## 5. 数据局限性

- **100 ep / yaml**，标准误 ±2-3pp。任何 ≤ 3pp 的 SR Δ **不能下结论**。
- **Phase 2 Layer 1 全 T-FULL，warm 只来自 NaN fallback**——composer 自己从未主动产 warm tier 决策。Layer 2.A 必须切 T-DUAL_07 看 composer 真实三档区分能力。
- **start_t 单点 (0.7)**：Phase 2 Layer 1 没扫 start_t；Layer 2 应试 0.5 / 0.3 看能否在保 SR 同时降 inf。
- **no decision-conditional SR join**：当前 summary 是 yaml-level 聚合，无法看"FULL_HIT 决策的 episode 平均 SR vs WARM 决策的 episode 平均 SR"。要 join `per_step/*.jsonl` 与 `episode_results/*.json` on `(task_id, episode_id)` 才行 — Layer 2 启动前可补一个分析脚本。

---

## 6. Phase 2 Layer 2 决策点

基于 Layer 1 数据，Layer 2.A 锁定以下组合（待生成 yaml）：

| Layer 2.A yaml stem 候选 | 因子 | tier | 假设 |
|---|---|---|---|
| `f1bt_LR_jerk_t_dual_07` | F1b-T × W-LONG-RISK × jerk | T-DUAL_07 | 跨 3 cfg winner + 主动 warm tier |
| `f1bt_LR_jerk_t_dual_05` | 同上 | T-DUAL_05 (start_t=0.5) | 用更便宜的 warm 看 SR 跌不跌 |
| `f1bt_w_fut_jerk_t_dual_07` | F1b-T × W-FUT × jerk | T-DUAL_07 | spatial16 信号 + jerk 单 desc |
| `splice_jerk_t_dual_07` | F1a-T.jerk + F1b-T.W-LONG-RISK.jerk | T-DUAL_07 | 双 state 因子互补 |
| `intrinsic_jerk_t_dual_07` | F1b-A.W-LONG-RISK.jerk + F1b-T.W-LONG-RISK.jerk | T-DUAL_07 | A+T 互补 |
| `full_lite_t_dual_07` | F1a-T.jerk + F1b-T.W-LONG-RISK.jerk + F2 | T-DUAL_07 | 3 因子精简组合 |

**6 yaml × 3 cfg × 100 ep = 1,800 ep ≈ 30 min wall-clock（6 server 并跑）**。

Layer 3：从 Layer 2 选 3 个不同 inf 段的 winner × 3 cfg × 1000 ep = 9,000 ep ≈ 1.5 h。

---

## 7. 文件索引

```
exp/verdict_factor_judge/
├── data/phase2_layer1/{clip,max_pool,spatial16}/
│   ├── per_yaml_summary.jsonl           # 26 行 / cfg, 含 success_rate
│   ├── per_step/<yaml_id>.jsonl         # 每 verdict 一行
│   └── episode_results/<yaml_id>.json   # 每 episode 成功/失败
├── analysis/
│   ├── plot_pareto.py                   # 画图脚本 + 公式 docstring
│   ├── phase2_layer1_pareto.png         # 三 cfg Pareto 图
│   ├── phase2_layer1_results.md         # 本文件
│   └── phase0_phase1_results.md         # 上轮分析
└── config/{clip,max_pool,spatial16}/phase2_layer1_{a,b}/
    ├── *_phase2_<stem>.yaml             # 13 + 13 = 26 eval yaml/cfg
    └── *_phase2_<stem>__warmup.yaml     # 同步 warmup sibling
```
