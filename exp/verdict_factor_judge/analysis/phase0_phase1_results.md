# Verdict Factor Judge — Phase 0 + Phase 1 结果分析

数据来源：

- Phase 0：`exp/verdict_factor_judge/data/phase0/{cfg}/run_state.json`（每 cfg 1 个 yaml × 100 ep AlwaysHit dump）
- Phase 1：`exp/verdict_factor_judge/data/phase1/{cfg}/per_yaml_summary.jsonl` + `episode_results/*.json`（每 cfg 8 个 yaml × 100 ep eval）
- 外部 baseline 对照：`exp/warm_start/data/`（500 ep × 3 cfg，always_hit / inference / warm_t∈{0.3,0.5,0.7}）

任务集统一为 `libero_spatial`，10 task × 10 trial / task = 100 ep（Phase 1）或 500 ep（warm_start baseline）。三个 KeyBuilder cfg 全程一致：

| cfg | KeyBuilder | key weights | trajectory depth |
|---|---|---|---:|
| `clip_w7_d4` | `clip` | `vision_0=0.1`, `vision_1=0.1`, `robot_state=0.8` | 4 |
| `max_pool_w3_d5` | `cp1_max_pool` | `vision_0=0.5`, `robot_state=0.5` | 5 |
| `spatial16_w8_d4` | `cp1_spatial_pool_16` | `vision_0=0.5`, `vision_1=0.25`, `robot_state=0.25` | 4 |

---

## 1. Phase 0 — AlwaysHit + DumpingJudge baseline

**目的**：在所有 verdict 上强制 FULL_HIT（即把 cache 命中的 action chunk 直接当真理执行），同时让 `DumpingJudge` 把 5 个候选因子的 raw 值全量落盘到 calibration JSONL。这套 dump 是 Phase 5 weight calibration 的训练数据来源。

**SR 结果**（100 ep）：

| cfg | SR (Phase 0, 100 ep) | warm_start always_hit (500 ep) | Δ |
|---|---:|---:|---:|
| `clip_w7_d4` | 0.700 | 0.674 | +0.026 |
| `max_pool_w3_d5` | 0.660 | 0.696 | -0.036 |
| `spatial16_w8_d4` | 0.650 | 0.692 | -0.042 |

100-ep 的 ±2pp 标准误覆盖了所有差异 → **Phase 0 SR 与 500-ep AlwaysHit baseline 统计一致**，可作为后续 Phase 1 的 **floor reference**（"如果判错都判 FULL_HIT 会变成什么"）。

---

## 2. Phase 1 — 8 yaml × 3 cfg ablation

**目的**：在 5 个候选因子 (F1a-A / F1a-T / F1b-A / F1b-T / F2) 上做 single-factor 与 full-stack ablation，判断哪些因子组合在最简（uniform weights = 1.0）配置下能产出比 fixed-policy baseline (Ceiling-W) 更好或至少可比的 SR。

每 yaml 流程：

```
warmup (AlwaysWarmStart, 2 trial × 10 task = 20 ep) → DumpingJudge dump
  → preload normalizer buffer (max 200 sample/key)
  → eval (CompositeJudge, 10 trial × 10 task = 100 ep)
```

### 2.1 SR + hit-type 联表（3 cfg 横向对比）

```
yaml_stem               | clip       | max_pool   | spatial16  | hit/warm/miss (clip)
─────────────────────────────────────────────────────────────────────────────────
f1a_t_only              | 0.95       | 0.93       | 0.94       |  724/525/953
f1b_a_only              | 0.91       | 0.88       | 0.89       |  974/  0/1370
f1b_t_only              | 0.93       | 0.93       | 0.88       | 1039/  0/1373
f_full_d_all_t_dual_07  | 0.97       | 1.00       | 0.99       |    0/2146/  10  ← all WARM_START
f_min_a (F1a-A all 4)   | 0.93       | 0.90       | 0.92       |  846/300/1158
f_min_a_dir             | 0.88       | 0.88       | 0.84       | 1028/324/1040
f_min_a_jerk            | 0.94       | 0.90       | 0.93       |  992/300/1051
f_min_cons (F2)         | 0.94       | 0.88       | 0.93       | 1149/  0/1159
```

完整 hit/warm/miss 矩阵见各 cfg 的 `per_yaml_summary.jsonl`。

### 2.2 与外部 baseline 对照（同 KeyBuilder，500 ep）

```
                        | clip   | max_pool | spatial16
─────────────────────────────────────────────────────
plain inference (no $)  | 0.992  | 0.984    | 0.984
always-WARM @ t=0.7     | 0.980  | 0.964    | 0.976   ← Ceiling-W (best fixed)
always-WARM @ t=0.5     | 0.960  | 0.966    | 0.952
always-WARM @ t=0.3     | 0.940  | 0.946    | 0.942
always-FULL_HIT         | 0.674  | 0.696    | 0.692   ← Floor (cache verbatim)
```

**关键参照点**：

- **Ceiling-W @ t=0.7** = 0.964–0.980：固定策略下能达到的最好 SR。任何"加入判断逻辑"的方案的及格线就是它。
- **plain inference** = 0.984–0.992：完全不用 cache，纯 π policy 的 SR。理论上限。
- **always-FULL_HIT** = 0.674–0.696：把 cache 当真理的代价。Phase 1 任何 yaml 跌到这附近 = 判 FULL_HIT 太激进。

---

## 3. 关键发现

### 3.1 F-FULL_T_DUAL_07 是唯一**稳定 ≥ Ceiling-W** 的组合

| cfg | F-FULL SR | Ceiling-W @0.7 | Δ |
|---|---:|---:|---:|
| clip | 0.97 | 0.980 | -0.010 |
| max_pool | **1.00** | 0.964 | **+0.036** |
| spatial16 | **0.99** | 0.976 | **+0.014** |

机制：99.4% 的 verdict 走 WARM_START（要么 composer 落 [0.3, 0.5) 命中 T-DUAL warm_start tier，要么 49 keys 中 NaN 比例过高 → all_nan_fallback@0.7 fire）。剩下 0.5% 是 search 没结果的 MISS。

注意：**0 FULL_HIT**。49 keys 含 jerk(risky) / dir(safe) / curv_radius(non_monotonic) / cum_disp(non_monotonic) 多方向因子，uniform weights 下 composer 永远抓不到 ≥ 0.5 的 composite score。这印证了 Phase 5 weight calibration 的必要性 — 当前 F-FULL 实际上是"伪 always-WARM"，不是真正的多因子判断器。

### 3.2 任何 FULL_HIT-heavy 的 yaml 都付出 SR 代价

```
yaml                    | clip SR | hit% | 比 Ceiling-W (0.98) 差
──────────────────────────────────────────────────────────────────
f1b_a_only              | 0.91    | 41%  | -0.07
f_min_a_dir             | 0.88    | 44%  | -0.10  ← worst
f_min_cons (F2)         | 0.94    | 50%  | -0.04
f_min_a_jerk            | 0.94    | 42%  | -0.04
```

判 FULL_HIT 的代价大、判 MISS 的代价小（让 π 自由走），所以**保守的 judge 优于激进的 judge**。这是 Phase 2 / 5 调权时的核心约束。

### 3.3 单因子排名

| 因子 | 平均 SR | 备注 |
|---|---:|---|
| F1a-T (state-history) all desc | 0.94 | 最强 single-factor，也是有 ~25% WARM_START "保险" |
| F1a-A jerk | 0.92 | 与 F1a all-desc 持平，单 desc 已饱和 |
| F1a-A all 4 desc | 0.92 | desc 加多没收益 |
| F2 (consistency) | 0.92 | 出乎预期平庸 |
| F1b-T (state windowed) | 0.91 | |
| F1b-A (action windowed) | 0.89 | NaN 边界压制 |
| F1a-A dir | 0.87 | 单 desc 最弱 |

F1a-A 4 desc → F1a-A jerk 单 desc：SR 不掉。说明对 F1a-A，**dir / curv_radius / cum_disp 三个 desc 在 uniform weights 下基本是噪声**，jerk 单独承担信号。

### 3.4 跨 cfg 一致性极强

每条 yaml 的 SR 在 3 cfg 间方差 < 3pp（spatial16 整体偏低 1-2pp，可能是 image embedding pool size 16 的 KeyBuilder 在边缘 query 上区分度差）。**KeyBuilder 选择不影响 yaml 排名** → Phase 4 / 5 calibration 可以单 cfg 训练 + 2 cfg 验证（对应原 plan §3.7）。

---

## 4. 待 Phase 2 决定的开放问题

### 4.1 F-FULL_T_DUAL_07 在 max_pool 上 SR=1.00 是真的吗

100/100 的 ±2pp 标准误覆盖了 +0.036 的 Δ，**统计上不能下"显著超过 Ceiling-W"的结论**。
建议 Phase 2 第一步：在 3 cfg 上把 F-FULL_T_DUAL_07 各跑 500 ep 重测，把噪声压到 ±1pp，再判定。

### 4.2 F-FULL 0 FULL_HIT 是 weights 问题还是 threshold 问题

两个假设：

- **A. weights 问题**：uniform 权重让 multi-direction 因子互相稀释 composite。Phase 5 weight calibration 后理论上可破解。
- **B. threshold 问题**：full_hit=0.5 对 49 keys uniform-sum 太高。降到 0.4 / 0.3 可能立刻有 hit。

可用 Phase 4 加一组 T-FULL_THR sweep ∈ {0.3, 0.4, 0.5} × 3 cfg = 9 run（成本约 1.5 hour）来分离两个假设。

### 4.3 单因子 yaml 是否还需要进 Phase 2

全部 8 个单因子 / 简化 yaml 都低于 Ceiling-W (0.88-0.95 vs 0.964-0.980)。建议：

- **不再独立进 Phase 2 ablation**，把 Phase 2 算力压到 F-FULL 变体上
- 单因子数据保留作为 Phase 5 weight calibration 的特征重要性参考

### 4.4 F1b W-MIX NaN 比例对结果的影响

`f1b_a_only` SR = 0.89-0.91，比 `f1a_t_only` 0.94 差 5pp。可能因为 F1b W-MIX (5,0) 和 (0,5) 在 trajectory 边缘 entry 100% NaN（per-project memory: T 中位 21，边缘 5 个 entry NaN 不可避免）。

如果 Phase 2 想"救" F1b：可以试 W-SHORT (`(0,3)(1,1)(3,0)`) 把 NaN 比例从 ~24% 降到 ~13%（plan §3.3b B-1 已铺路），代价是窗口短信号弱。

---

## 5. 数据局限性

- **Phase 1 = 100 ep / yaml**，标准误 ±2-3pp。任何 < 3pp 的 Δ 不能下结论。
- **Phase 1 success_rate 是新写入的字段**（runner SR forwarding 在本次重跑前没接，老数据已废弃，第一次重跑生成的 SR 全部基于 `--save-episode-results` 的 episode-level dump）。
- **success_rate 与 hit/warm/miss 比例没有 verdict-level join**：当前 `per_yaml_summary.jsonl` 是 yaml-level 聚合，不能回答"FULL_HIT 决策的 episode 平均 SR vs WARM_START 决策的 episode 平均 SR"。需要 join `per_step/*.jsonl` 与 `episode_results/*.json` on `(task_id, episode_id)` 才能拿到 decision-conditional SR — Phase 2 之前应该先建这个 join 视图。

---

## 6. 文件索引

```
exp/verdict_factor_judge/
├── data/
│   ├── phase0/{clip,max_pool,spatial16}/
│   │   ├── run_state.json                   # SR + per-task progress
│   │   └── logs/*.episode_results.json      # per-episode success
│   └── phase1/{clip,max_pool,spatial16}/
│       ├── per_yaml_summary.jsonl           # 1 row / yaml: yaml_id, sr, hit/warm/miss
│       ├── per_step/<yaml_id>.jsonl         # 1 row / verdict: phase, hit_type, cp1_score
│       └── episode_results/<yaml_id>.json   # 1 row / episode: task_id, episode_id, success
├── analysis/
│   └── phase0_phase1_results.md             # this file
└── config/{clip,max_pool,spatial16}/
    ├── phase0/                              # 1 yaml each (always_hit_dump)
    └── phase1/                              # 8 eval yaml + 8 sibling __warmup.yaml each
```

外部 baseline：

```
exp/warm_start/data/
├── baseline_failures.json     # always_hit + inference SR + per-failure (task, init) lists
├── state_full_{cfg}.json      # warm_t∈{0.3,0.5,0.7} 的 RunState 含 SR
└── {cfg}/*.episode_results.json
```
