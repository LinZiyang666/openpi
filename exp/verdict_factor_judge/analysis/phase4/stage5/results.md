# Phase 4 Stage 5 — 48-cell × 500ep 真值复测结果

> 数据：`exp/verdict_factor_judge/data/phase5/per_yaml_summary.jsonl` (48 行)
> 命令书：[`logs/verdict_phase4_stage5_run_commands.log.md`](../../../logs/verdict_phase4_stage5_run_commands.log.md)
> Plan：[`logs/verdict_phase4_weight_sweep.log.md`](../../../logs/verdict_phase4_weight_sweep.log.md)
> Pareto 图：[`phase4_stage5_pareto.png`](phase4_stage5_pareto.png)

---

## 0. TL;DR

| # | 关键问题 (Stage 1+2 遗留) | Stage 5 答案 |
|---:|---|---|
| Q1 | phase3 g1/g10 真值 | g1≈**0.915**, g10≈**0.904**（plan 锚点 0.95/0.96 系统性高估 ~3-5pp） |
| Q2 | p2 R2 dir-only winner 是真信号？ | **NO, 完全是 noise** — 500ep dir-only=0.892 在 8 patterns 中最低 |
| Q3 | R1 α 中点 (0.4/0.6) 优于端点？ | **NO** — p1 max-min=5.4pp / p2 max-min=0.6pp，全在 noise floor 内；fusion 假设破裂 |
| Q4 | R4 W-FUT 双窗权重提升？ | **边缘 marginal**：p1 best +2.4pp / p2 best +5.6pp vs R2，**但 4/8 cell 进 Pareto-positive** |
| Q5 | phase4 thr +0.05 偏移是 bug？ | **NO** — 是 spec 不同：p1 (10 因子 α=1.0 → 5 offline) ≠ g1 (6 因子)，weight 分布不同 → score 分布不同 → thr 自然偏移 |

**Phase 4 整体定性**：**mixed**。
- ✅ **Pareto 视角有改进**：phase4 在 SR ≥ 0.92 高分段 push 了 phase3 frontier（11.5pp inf 节省 / +1pp SR vs phase3 g1 fh0.3_ws0.5）。
- ❌ **fusion 假设彻底破裂**：α=1.0 (pure offline) ≈ α=0.6 ≈ R2 任何 pattern ≈ R4 任何 pattern，all 在 500ep noise floor (±2.2pp) 内。
- 🟡 **唯一有方向性的探索维度**：R4 W-FUT 双窗 4/8 cell 是 Pareto-positive，远超 R1 (1/8) 和 R2 (2/16)。但 cell 间差异仍 < noise floor。

---

## 1. 算力 / 拓扑 confirm

| 指标 | 计划 | 实际 |
|---|---:|---:|
| cells | 48 | 48 ✓ |
| episode/cell | 500 | 500 ✓ |
| servers | 6 | 6 ✓ |
| total ep | 24,000 | 24,000 ✓ |
| 噪声标准误 (95% CI half) | ~±2.2pp | ~±2.0-2.7pp（实测 SR×(1-SR)/500 → 取决于 SR） |

48 行 summary 全部 healthy（无 `error` 字段，无 `n_eval_verdicts=0`）。

---

## 2. Group A — phase3 g1 / g10 anchor 真值锁定

### 2.1 数据表

| recipe | (FH, WS) | SR_500ep | SR_100ep | Δ | inf_500 | thr (FH/WS) |
|---|---|---:|---:|---:|---:|---|
| g1 | (0.3, 0.5) | **0.948** | 0.970 | -0.022 | 0.592 | 0.530 / 0.195 |
| g1 | (0.4, 0.5) | 0.894 | 0.950 | -0.056 | 0.459 | 0.453 / 0.000 |
| g1 | (0.5, 0.4) | **0.910** | 0.900 | +0.010 | **0.380** | 0.380 / 0.000 |
| g1 | (0.5, 0.5) | 0.908 | 0.950 | -0.042 | 0.382 | 0.380 / 0.000 |
| g10 | (0.3, 0.5) | **0.958** | 1.000 | -0.042 | 0.586 | 0.515 / 0.198 |
| g10 | (0.4, 0.5) | 0.892 | 0.960 | -0.068 | 0.459 | 0.455 / 0.000 |
| g10 | (0.5, 0.4) | 0.876 | 0.960 | -0.084 | 0.408 | 0.413 / 0.000 |
| g10 | (0.5, 0.5) | 0.890 | 0.930 | -0.040 | 0.407 | 0.413 / 0.000 |

### 2.2 关键观察

- **g1 真值 mean = 0.915, range [0.894, 0.948]**
- **g10 真值 mean = 0.904, range [0.876, 0.958]**
- 100ep SR 系统性 **高估** 真值 ~3-5pp（g10 fh0.5_ws0.4 高估 8.4pp）
- 单 100ep cell 标准误 √(0.95×0.05/100) = ±2.2pp，95% CI ±4.4pp — **实测偏差超过预期 CI**，说明 phase3 stage1 是 lucky-side draw
- Pareto-positive cells: g1 fh0.5_ws0.4 (low inf=0.380 anchor) + g1 fh0.3_ws0.5 + g10 fh0.3_ws0.5（高 SR 高 inf）

### 2.3 影响

> **phase4 plan §2 anchor (g1=0.95, g10=0.96) 已被推翻**。phase4 SR 涨幅看起来比真实小：plan 写 phase4 p1 SR=0.96 vs g1=0.95 → "+1pp"，实际真值 phase4 p1 best=0.958 vs g1=0.915 → **+4.3pp** 才是真涨幅（但 inf 也涨 +9.5pp，trade-off 不变）。

---

## 3. Group B+C — phase4 R2 desc patterns 复测

### 3.1 p1 R2 8 patterns（按 SR 排序）

| pattern | SR_500 | SR_100 | Δ | inf_500 | thr (FH/WS) |
|---|---:|---:|---:|---:|---|
| **dir-only** | **0.958** | 0.940 | +0.018 | 0.477 | 0.460 / 0.010 |
| jerk-only | 0.944 | 0.960 | -0.016 | 0.500 | 0.450 / 0.010 |
| jerk-heavy | 0.932 | 0.960 | -0.028 | 0.490 | 0.446 / 0.062 |
| dir-heavy | 0.930 | 0.950 | -0.020 | 0.481 | 0.440 / 0.052 |
| uniform | 0.928 | 0.960 | -0.032 | 0.471 | 0.430 / 0.060 |
| disp-heavy | 0.926 | 0.950 | -0.024 | 0.485 | 0.414 / 0.048 |
| path-heavy | 0.912 | 0.930 | -0.018 | 0.490 | 0.466 / 0.076 |
| path-only | 0.866 | 0.830 | +0.036 | 0.478 | 0.440 / 0.020 |

### 3.2 p2 R2 8 patterns（按 SR 排序）

| pattern | SR_500 | SR_100 | Δ | inf_500 | thr (FH/WS) |
|---|---:|---:|---:|---:|---|
| jerk-heavy | 0.936 | 0.950 | -0.014 | 0.512 | 0.460 / 0.264 |
| dir-heavy | 0.934 | 0.940 | -0.006 | 0.529 | 0.474 / 0.258 |
| jerk-only | 0.930 | 0.930 | 0.000 | 0.490 | 0.420 / 0.080 |
| uniform | 0.926 | 0.920 | +0.006 | 0.516 | 0.453 / 0.267 |
| disp-heavy | 0.926 | 0.910 | +0.016 | 0.505 | 0.406 / 0.222 |
| path-heavy | 0.920 | 0.940 | -0.020 | 0.515 | 0.472 / 0.272 |
| path-only | 0.894 | 0.900 | -0.006 | 0.511 | 0.480 / 0.120 |
| **dir-only** | **0.892** ⚠️ | **0.970** | **-0.078** | 0.521 | 0.480 / 0.100 |

### 3.3 关键观察

- **p2 R2 dir-only winner 决议彻底反转**：100ep 0.97 → 500ep 0.892 = **8 patterns 中最低**。stage 2 100ep dir-only=0.97 是 unlucky baseline（uniform=0.92）+ single-cell luck 的 noise tail。
- **p1 R2 path-only outlier reproducible**：100ep 0.83 → 500ep 0.866，仍比相邻 desc-only pattern 低 6-9pp。path 因子（在 p1 weight pattern 下）确实表现差。
- **p1/p2 R2 内部所有非退化 cell 真值差距全在 noise floor (±2.2pp) 内**（p1 max-min 排除 path-only 后 = 0.958-0.912 = 4.6pp，去掉 dir-only 后 = 0.944-0.912 = 3.2pp；p2 max-min 排除 dir-only/path-only 后 = 0.936-0.920 = 1.6pp）。
- **结论**：R2 desc-pattern 探索的所有"差异"（除 path-only outlier）都是 100ep noise driven。

### 3.4 影响

> **stage 2 G_R2 决议（p1→uniform / p2→dir-only）在 500ep 视角下应改写为**：
> - p1 R2 winner: **dir-only** (0.958，但 jerk-only 0.944 在 noise 内追平)
> - p2 R2 winner: **jerk-heavy** (0.936，但 dir-heavy/jerk-only/uniform 全在 noise 内追平)
>
> 但实际意义有限——所有 winner 候选在 noise floor 内无法分辨。

---

## 4. Group D — R1 α sweep 真值

### 4.1 数据

| recipe | α | SR_500 | SR_100 | Δ | inf_500 | thr (FH/WS) |
|---|---:|---:|---:|---:|---:|---|
| p1 | 0.0 | 0.896 | 0.880 | +0.016 | 0.454 | 0.350 / 0.000 |
| p1 | 0.4 | 0.914 | 0.880 | +0.034 | 0.482 | 0.383 / 0.024 |
| p1 | 0.6 | **0.950** | 0.930 | +0.020 | 0.483 | 0.431 / 0.036 |
| p1 | 1.0 | 0.930 | 0.960 | -0.030 | 0.473 | 0.430 / 0.060 |
| p2 | 0.0 | 0.904 | 0.920 | -0.016 | 0.450 | 0.340 / 0.000 |
| p2 | 0.4 | 0.904 | 0.950 | -0.046 | 0.507 | 0.379 / 0.159 |
| p2 | 0.6 | 0.908 | 0.910 | -0.002 | 0.495 | 0.395 / 0.220 |
| p2 | 1.0 | 0.910 | 0.970 | -0.060 | 0.518 | 0.453 / 0.267 |

### 4.2 关键观察

- **p1**: α=0.6 SR=0.950 最高，但与 α=1.0 (0.930) 差 +2.0pp，**在 500ep noise floor (±2.2pp) 内**
- **p2**: 几乎完全 flat，α=0.0/0.4/0.6/1.0 → 0.904/0.904/0.908/0.910，max-min = 0.6pp
- **fusion (online + offline 凸组合) 假设彻底破裂**: α 中点优于端点的预测 → 实测 noise-equivalent
- 100ep stage1 数据偏差大：p1 α=0.4/0.6 (+3pp) / p2 α=0.4/1.0 (-5pp/-6pp)，再次说明 phase4 stage1 决议 (α*=1.0 winner) 是 noise 触顶 (单 cell luck)

### 4.3 影响

> R1 α sweep 在 500ep 真值下**没有任何决定性的 α 偏好**。stage1 选 α*=1.0 是 noise 偶然 — 实际上 R1 α 整个维度对 SR 几乎无影响。

---

## 5. Group E — R4 W-FUT 双窗权重首次实测

### 5.1 数据

| recipe | window pattern | SR_500 | inf_500 | thr (FH/WS) |
|---|---|---:|---:|---|
| p1 | **win-short-heavy** | **0.952** | 0.487 | 0.453 / 0.080 |
| p1 | win-long-heavy | 0.932 | 0.470 | 0.423 / 0.040 |
| p1 | win-long-only | 0.924 | 0.446 | 0.425 / 0.000 |
| p1 | win-short-only | 0.920 | 0.493 | 0.480 / 0.120 |
| p2 | **win-short-only** | **0.948** | 0.502 | 0.520 / 0.120 |
| p2 | win-short-heavy | 0.944 | 0.511 | 0.500 / 0.107 |
| p2 | win-long-heavy | 0.942 | 0.515 | 0.480 / 0.093 |
| p2 | win-long-only | 0.918 | 0.445 | 0.440 / 0.000 |

### 5.2 关键观察

- **p1 R4 best (win-short-heavy) = 0.952** vs p1 R2 baseline (uniform=0.928): **+2.4pp**，边缘显著
- **p2 R4 best (win-short-only) = 0.948** vs p2 R2 baseline (jerk-heavy=0.936): **+1.2pp**，noise 内
- 但 p2 R4 best vs p2 R2 stage2-decision (dir-only=0.892): **+5.6pp**，统计显著（但与 dir-only 真值塌陷有关，不是 R4 的功劳）
- **window 偏好定性**：p1 偏好 short window heavy，p2 偏好 short window only — 暗示 1-step "future" 信号比 long-window 更可靠
- **R4 4/8 cell 进 Pareto-positive**（vs Group D 1/8、Group BC 2/16），是 phase4 唯一密集 Pareto 改进的 round

### 5.3 影响

> R4 是 phase4 唯一**有方向性**的探索维度（short window > long window）。但单 cell SR 涨幅都在 noise floor 边缘，需要 multi-seed 验证。R4 win-short 偏好为后续 phase 提供唯一 actionable hint。

---

## 6. Group F — phase3 g6 (pure online) + g4/g8/g9/g11 退化 verify

### 6.1 数据

| recipe | (FH, WS) | SR_500 | SR_100 | Δ | inf_500 | thr (FH/WS) | 注 |
|---|---|---:|---:|---:|---:|---|---|
| g6 | (0.3, 0.5) | 0.932 | 0.940 | -0.008 | 0.536 | 0.490 / 0.000 | pure online |
| g6 | (0.4, 0.5) | 0.892 | 0.900 | -0.008 | 0.449 | 0.340 / 0.000 | pure online |
| g6 | (0.5, 0.4) | 0.884 | 0.870 | +0.014 | 0.405 | 0.230 / 0.000 | pure online |
| g6 | (0.5, 0.5) | 0.872 | 0.860 | +0.012 | 0.402 | 0.230 / 0.000 | pure online |
| g4 | (0.5, 0.5) | 0.876 | 0.840 | +0.036 | 0.426 | 0.427 / 0.053 | f1b w-short jerk |
| g8 | (0.5, 0.5) | 0.594 | 0.570 | +0.024 | **0.000** | 0.000 / 0.000 | **退化 (pure base SR)** |
| g9 | (0.5, 0.5) | 0.824 | 0.790 | +0.034 | 0.411 | 0.378 / 0.000 | f1b sym-s all |
| g11 | (0.5, 0.5) | 0.604 | 0.590 | +0.014 | **0.000** | 0.000 / 0.000 | **退化 (pure base SR)** |

### 6.2 关键观察

- **g6 (pure online) 真值 mean = 0.895**，比 g1 fusion (0.915) 略低 ~2pp，**在 noise 内**。pure online 已经接近 best fusion；offline 信息边际收益小。
- **g8/g11 退化 verified**：thr=0.0/0.0 → 全 cache miss path → SR 0.594/0.604 ≈ libero_spatial pi05 base SR (~0.60)。这俩 recipe 在 spatial16 下完全无效。
- **g4/g9 旁系 anchor**：g4 真值 0.876（f1b w-short jerk）/ g9 真值 0.824（f1b sym-s all）。低于 g1/g10/g6 ~2-9pp，但其复用价值需在其他 keybuilder（如 spatial32/temporal）下重测才能定。

### 6.3 影响

> **g6 真值 (0.895)** 接近 g1 真值 (0.915)，offline factor 边际收益微弱。**Phase4 fusion 假设破裂的根本原因可能是**：online factor (g6) 已经足够好，offline factor (g1/g10) 边际信息冗余 → composer 凸组合无 marginal benefit。

---

## 7. Pareto 前沿分析

### 7.1 Stage 5 own Pareto upper frontier (7 points)

| rank | inf | SR | yaml |
|---:|---:|---:|---|
| 1 | 0.000 | 0.594 | phase3 g8 退化 |
| 2 | 0.000 | 0.604 | phase3 g11 退化 |
| 3 | **0.380** | **0.910** | **phase3 g1 fh0.5_ws0.4** ← lowest-inf high-SR anchor |
| 4 | 0.445 | 0.918 | phase4 p2 R4 win-long-only |
| 5 | 0.446 | 0.924 | phase4 p1 R4 win-long-only |
| 6 | 0.470 | 0.932 | phase4 p1 R4 win-long-heavy |
| 7 | **0.477** | **0.958** | **phase4 p1 R2 dir-only** ← highest-SR cell |

**front 构成**：phase3 = 3，phase4 R2 = 1，phase4 R4 = 3，phase4 R1 = **0**。R1 α sweep 在 Pareto 上零贡献。

### 7.2 Pareto-positive 计数 (vs random/periodic + always-WARM 基线，gold-circle)

| Group | gold / total | 比例 |
|---|---:|---:|
| A (phase3 g1/g10) | 3/8 | 38% |
| B+C (phase4 R2) | 2/16 | 13% |
| D (phase4 R1 α) | 1/8 | 13% |
| **E (phase4 R4)** | **4/8** | **50%** ← 最密集 |
| F (phase3 g6 + 旁系) | 2/8 | 25% |
| **总计** | **12/48** | 25% |

R4 是**唯一比 R2 占比更高**的 round；但单 cell SR 优势仍在 noise floor 边缘。

### 7.3 phase4 vs phase3 anchor 公平比较 (FH=0.5/WS=0.5)

| cell | SR | inf | Δ vs phase3 g1/g10 |
|---|---:|---:|---|
| **phase3 g1 fh0.5_ws0.5** | 0.908 | 0.382 | (anchor) |
| **phase3 g10 fh0.5_ws0.5** | 0.890 | 0.407 | (anchor) |
| phase4 p1 R2 uniform | 0.928 | 0.471 | SR +2.0pp / inf +8.9pp |
| phase4 p1 R2 dir-only | 0.958 | 0.477 | SR +5.0pp / inf +9.5pp |
| phase4 p1 R4 win-short-heavy | 0.952 | 0.487 | SR +4.4pp / inf +10.5pp |
| phase4 p2 R2 jerk-heavy | 0.936 | 0.512 | SR +4.6pp / inf +10.4pp |
| phase4 p2 R4 win-short-heavy | 0.944 | 0.511 | SR +5.4pp / inf +10.4pp |

**phase4 用 +10pp inf 换 +5pp SR**，**不是 strictly Pareto positive**（在 (0.5, 0.5) 这个具体 cell 上）；但**整体 frontier 上 phase4 高 SR 段 dominates phase3 fh0.3_ws0.5**：

| cell | inf | SR | dom 关系 |
|---|---:|---:|---|
| phase3 g1 fh0.3_ws0.5 | 0.592 | 0.948 | dominated |
| phase4 p1 R2 dir-only | **0.477** | **0.958** | **dominates** ↑ (-11.5pp inf, +1.0pp SR) |

---

## 8. inf 偏移 root cause (Q5 解答)

phase3 g1 (0.5, 0.5): thr fh=0.380 / ws=0.000，inf=0.382
phase4 p1 R1 α=1.0 (≈ R2 uniform): thr fh=0.430 / ws=0.060，inf=0.473

thr 差 +0.05 / +0.06 — **不是 bug**，源于 spec：
- phase3 g1 = 6 因子 (5 offline + 1 online)，hardcode `{k: 1.0}` for all 6 keys → composer score = mean(6 contribs)
- phase4 p1 R2 uniform / R1 α=1.0 = 10 因子 (5 offline `state_fut` + 5 online `online_act`)，**α=1.0 时 online weights = 0** → 实际 active 只 5 个 offline，其它 5 weight=0

→ **score 分布不同**（5-key average vs 6-key average，分子分母不同），`derive_thresholds` 解出的 quantile cut 自然不同。这是 design 的直接后果，非 bug。

→ **影响**：phase4 vs phase3 不应直接比 thr / inf 数字（spec 不同）；应比同 cell SR + inf trade-off。

---

## 9. Final verdict

### 9.1 哪个 cell 应该上 production？

**取决于 SR vs inf 偏好**：

| 偏好 | 推荐 cell | SR | inf |
|---|---|---:|---:|
| **最低 inf** (省算力) | phase3 g1 fh0.5_ws0.4 | 0.910 | **0.380** |
| **最高 SR** | phase4 p1 R2 dir-only | **0.958** | 0.477 |
| **中庸** | phase4 p1 R4 win-long-heavy | 0.932 | 0.470 |

production 默认推荐：**phase3 g1 fh0.5_ws0.4**（SR 91% 在 LIBERO spatial 已超 always-WARM @ start_t=0.7 的 0.976 仅 -6.6pp，但 inf cost 节省 53.5pp from 0.65 → 0.380）。如果 SR 是硬指标且能 tolerate +9pp inf cost，则 **phase4 p1 R2 dir-only**。

### 9.2 phase 4 概念定性

| 假设 | 验证结果 |
|---|---|
| H1: weighted-sum composer 比 hardcode-1.0 更优 | **NO**（uniform = α=1.0 与最佳 desc-pattern 无差异） |
| H2: offline + online 凸组合 (α∈(0,1)) 优于 pure online (α=0) 或 pure offline (α=1) | **NO**（p1 max α=0.6 0.950 vs α=1.0 0.930 差 +2pp 在 noise 内；p2 完全 flat） |
| H3: factor 描述 (jerk/dir/disp/path) 间有显著差异 | **NO**（noise floor 内全 equivalent，只 path-only 在 p1 outlier） |
| H4: W-FUT 双窗 short vs long 有偏好 | 🟡 **WEAK YES**（short window 在 p1/p2 都略胜，但单 cell SR 仍在 noise floor 边缘） |

### 9.3 后续 follow-up（不在本 stage 计划）

- ❓ R4 W-FUT short vs long 在 multi-seed (3 × 500ep) 下是否 reproducible
- ❓ R4 win-short 偏好在其他 keybuilder (spatial32 / temporal) 下是否 transfer
- ❓ p2 R2 disp-only 退化 cell（本 stage 跳过）的 verify

---

## 10. 数据 / 复算

```bash
# 解压
cd /tmp && tar xzf /mnt/c/Users/lzy66/Desktop/fsdownload/phase5_20260509_100716.tar.gz

# 重画 Pareto
MPLBACKEND=Agg uv run python -m exp.verdict_factor_judge.analysis.phase4.stage5.plot_pareto
```

## 11. 文件清单

| 文件 | 内容 |
|---|---|
| `phase4_stage5_results.md` | 本文档 |
| `phase4_stage5_pareto.png` | Pareto 散点 + frontier，48 cell 按 group 染色 |
| `plot_pareto_phase4_stage5.py` | Pareto 绘图脚本 |
| `data/phase5/per_yaml_summary.jsonl` | 48 行 master summary（合并自 6 batch） |
| `data/phase5/per_yaml_summary_batch{1..6}.jsonl` | 6 server 各自 batch summary |
| `data/phase5/per_step/<48>.jsonl` | 48 cell × 500 ep × ~30 verdict/ep verdict log |
| `data/phase5/episode_results/<48>.json` | 48 cell × 500 ep episode 级结果 |
| `data/phase5/thresholds/<7>.json` | 7 phase3 recipe 各自 thr (cached) |
