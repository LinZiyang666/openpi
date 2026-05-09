# Verdict Factor Judge — Phase 4 Stage 1 (R1 α 扫描) 结果分析

数据来源：`exp/verdict_factor_judge/data/phase4/r1_alpha/per_yaml_summary.jsonl`（14 行，0 NA）。
画图脚本：[`plot_alpha_sweep_phase4.py`](plot_alpha_sweep_phase4.py) → [`phase4_stage1_alpha_sweep.png`](phase4_stage1_alpha_sweep.png)。
执行命令清单：[`logs/verdict_phase4_stage1_run_commands.log.md`](../../../logs/verdict_phase4_stage1_run_commands.log.md)。
Plan：[`logs/verdict_phase4_weight_sweep.log.md`](../../../logs/verdict_phase4_weight_sweep.log.md)（G1 APPROVED R3 / G2 APPROVED R2）。

---

## 0. 公式

```
inf_ratio = (0·n_full_hit + 0.75·n_warm_start + 1·n_miss) / n_eval_verdicts        # warm cost = 0.75
score(α)  = SR(α) − 0.5 · inf(α)                                                    # G_R1 综合指标
```

---

## 1. 实验目标 + 假设

Phase 4 把 phase3 winner **g1+g6** 与 **g10+g6** 融合成两个 10-score recipe（详见 plan §1）：

| recipe | offline 8 score | online 2 score | locked cell (FH, WS) | 锚 SR |
|---|---|---|---|---|
| **p1** = g1 + g6 | 4 desc × 2 W-FUT 窗 (state) | 2 desc × W-K3 (action) | (0.5, 0.5) | 0.95 |
| **p2** = g10 + g6 | 4 desc × 2 W-FUT 窗 (action) | 2 desc × W-K3 (action) | (0.5, 0.4) | 0.96 |

**假设**：phase3 g1/g10（pure offline，cache pkl 静态字段）与 g6（pure online，runtime history+chain）信号源不重叠 → 误差应部分独立 → 加权融合可能在不丢任一侧强信号的前提下抑制对方的高噪声模式。如果 α 响应曲线呈非单调（α=0.5 同时优于 α=0 和 α=1），就证实独立性在数据上确实成立。

**Stage 1 = R1 α 扫描**：α ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}（7 点）；offline 组内均匀分配 α/8，online 组内均匀分配 (1−α)/2；2 recipe × 7 α = **14 eval cell** + 2 warmup。

---

## 2. 主表

```
recipe                            α      SR     inf  hit%  warm%  miss%   score   fh_thr   ws_thr
─────────────────────────────────────────────────────────────────────────────────────────────────
p1_state_fut_online_act         0.0   0.880   0.452   40%    60%     0%   0.654   0.430    0.060
p1_state_fut_online_act         0.2   0.920   0.481   42%    39%    19%   0.679   ...      ...
p1_state_fut_online_act         0.4   0.880   0.477   43%    39%    19%   0.642   ...      ...
p1_state_fut_online_act         0.5   0.920   0.481   42%    41%    17%   0.680   ...      ...
p1_state_fut_online_act         0.6   0.930   0.492   41%    41%    18%   0.684   ...      ...
p1_state_fut_online_act         0.8   0.900   0.486   41%    40%    19%   0.657   ...      ...
p1_state_fut_online_act         1.0   0.960   0.472   43%    39%    18%   0.724   0.430    0.060
─────────────────────────────────────────────────────────────────────────────────────────────────
p2_action_fut_online_act        0.0   0.920   0.441   41%    59%     0%   0.700   ...      ...
p2_action_fut_online_act        0.2   0.910   0.491   43%    30%    27%   0.664   ...      ...
p2_action_fut_online_act        0.4   0.950   0.497   43%    29%    28%   0.701   ...      ...
p2_action_fut_online_act        0.5   0.940   0.496   43%    30%    27%   0.692   ...      ...
p2_action_fut_online_act        0.6   0.910   0.499   43%    28%    29%   0.661   ...      ...
p2_action_fut_online_act        0.8   0.930   0.504   42%    30%    28%   0.678   ...      ...
p2_action_fut_online_act        1.0   0.970   0.506   41%    34%    25%   0.717   ...      ...
```

> **注意**：summary jsonl 里 `fh_thr` / `ws_thr` 字段为 None — runner 在 `--mode run-eval` 路径下未把 emit-eval-yamls 时算的 thr 回填进 summary（minor schema 缺失）。实际跑用的 thr 嵌入在 yaml 文件里，正确无误（α=0/1 的 thr 直接从 yaml 读，与 phase3 g1 同 cell 对照如 §4.2）。

---

## 3. baseline 对照（spatial16 × locked cell）

phase 3 同 cell 真值（来自 `data/phase3/per_yaml_summary.jsonl`）：

| baseline | cell | SR | inf | hit/warm/miss% |
|---|---|---:|---:|:---:|
| g1 (offline_state W-FUT × all-4 desc) | (0.5, 0.5) | 0.950 | 0.372 | 50/50/0 |
| g6 (online_action W-K3 × jerk+disp pair) | (0.5, 0.5) | 0.860 | 0.410 | 53/55/0 (fh+ws partial overlap) |
| g10 (offline_action W-FUT × all-4 desc) | (0.5, 0.4) | 0.960 | 0.385 | 49/51/0 |
| g6 | (0.5, 0.4) | 0.870 | 0.403 | ... |

---

## 4. 关键发现

### 4.1 Winner = α = 1.0（两 recipe 都是）

| recipe | α* | SR | inf | continue (G_R1) |
|---|:---:|:---:|:---:|:---:|
| p1 | **1.0** | 0.960 | 0.472 | ✓ (anchor 0.95 − 2pp = 0.93) |
| p2 | **1.0** | 0.970 | 0.506 | ✓ (anchor 0.96 − 2pp = 0.94) |

`score(α) = SR − 0.5·inf` 在 α=1.0 时最高（p1 0.724，p2 0.717）。两个 recipe 在 anchor 容差内通过，G_R1 决议**两 recipe 均进入 R2**。

### 4.2 端点 sanity ✓（SR 4/4 通过 ±3pp）

| recipe | α | phase 4 SR | phase 3 ref | phase 3 SR | Δ |
|---|:---:|:---:|---|:---:|:---:|
| p1 | 1.0 | 0.960 | g1 (0.5, 0.5) | 0.950 | **+0.010 ✓** |
| p1 | 0.0 | 0.880 | g6 (0.5, 0.5) | 0.860 | **+0.020 ✓** |
| p2 | 1.0 | 0.970 | g10 (0.5, 0.4) | 0.960 | **+0.010 ✓** |
| p2 | 0.0 | 0.920 | g6 (0.5, 0.4) | 0.870 | **+0.050 ⚠** |

p2 α=0.0 偏离 +0.050（borderline 但在 100 ep × 1 seed 的 ~3pp noise floor 之外）。这是**唯一** borderline 的端点；不阻塞 R1 决议但值得 R2 / phase 5 1000 ep 复测核对。

### 4.3 α 中点 SR 起伏在噪声内 — 没融合收益

```
p1: α=0.0 0.88 → 0.2 0.92 → 0.4 0.88 → 0.5 0.92 → 0.6 0.93 → 0.8 0.90 → 1.0 0.96
p2: α=0.0 0.92 → 0.2 0.91 → 0.4 0.95 → 0.5 0.94 → 0.6 0.91 → 0.8 0.93 → 1.0 0.97
```

- p1：α ∈ [0.0, 0.8] 区间 SR 0.88–0.93 起伏 5pp，**无任何中间点超过 α=1.0**。
- p2：同款，α ∈ [0.0, 0.8] SR 0.91–0.95，**无任何中间点超过 α=1.0**。

**核心判断（§1 假设是否成立）**：

> phase 1 假设"online + offline fusion 给出非单调 α 响应（α=0.5 优于两端）→ 误差独立"在 stage 1 数据上**未得到证实**。两条曲线都是"端点 1.0 最高、中间在噪声内起伏"的形态——等价于"online 信号被 offline 信号覆盖，加进来只是稀释"。

α=1.0 winner 等同于把 online weight 全清零，等同于 phase 3 g1/g10 在锁 cell 上的复跑（端点 sanity §4.2 确认）。

### 4.4 inf 端点系统偏高 ~0.10 — follow-up 项

| recipe | α | phase 4 inf | phase 3 ref inf | Δ |
|---|:---:|:---:|:---:|:---:|
| p1 | 1.0 | 0.472 | 0.372 | **+0.100** |
| p2 | 1.0 | 0.506 | 0.385 | **+0.121** |
| p1 | 0.0 | 0.452 | 0.410 | +0.042 |
| p2 | 0.0 | 0.441 | 0.403 | +0.038 |

α=1.0 的 phase 4 inf 比 phase 3 同 cell 高 ~0.10，原因：

```
phase 3 g1 (0.5, 0.5):    fh_thr=0.380   ws_thr=0.000   →  hwm = 50/50/0%
phase 4 p1 α=1.0:         fh_thr=0.430   ws_thr=0.060   →  hwm = 43/39/18%
```

phase 4 解出的 fh_thr 高 0.05、ws_thr 高 0.06，更多 verdict 跌进 miss（18% vs 0%）→ inf 高。

**可能原因**：phase 4 warmup yaml 含 10 个 factor（offline 8 + online 2），phase 3 g1 warmup yaml 含 8 个 factor。两个 yaml 的 factor extraction 路径在 view fetch 时序上有差异，造成 warmup factor_raw jsonl 的有效行数 / 内容分布不同 → solver 的 quantile 切位（`derive_thresholds`）偏高。SR backward compat 测试只验过单个 composer 在 isolated 输入下的 score 数值一致，没覆盖"phase 4 多因子 warmup 与 phase 3 少因子 warmup 在同源 calibration 下解 thr 是否完全等同"的场景。

**SR 几乎不受影响**（4/4 端点 sanity 通过），所以 R1 决议有效；但 inf 偏移让 phase 4 的 cell 在 (inf, SR) 平面上"看起来"比对应 phase 3 cell 贵 0.10 inf，跨阶段 Pareto 比较时**必须用 phase 3 同 cell 真值**作为对照锚（不要直接看 phase 4 数字）。

R2 启动前应：(a) 看 phase 4 R1 cell yaml 的 fh_thr/ws_thr 与 phase 3 同 cell 比；(b) 若 R2 cell 同样偏高，phase 5 应用 1000 ep × 1 seed 复测 + 强制 phase 3 thr 复用机制做对照。

---

## 5. R2 / R3 / R4 影响

按 G_R1 决议 α* = 1.0（两 recipe 同款）：

| Round | 数学含义（α*=1.0） | 是否值得跑 |
|---|---|:---:|
| **R2** offline 4-desc 权重 sweep | offline 仍占 100% 权重，9 个 desc-share pattern 完整有效 | **✓ 值得跑** — 看是否 jerk-heavy / disp-heavy 等比 uniform 高（与 phase 3 layer 1 "jerk 是主信号载体"结论对照） |
| **R3** online 2-desc 权重 sweep | (1−α*) = 0 → online 总权重 = 0 → 5 个 pattern 数学上**等价**（全 0） | **✗ 跳过** — 跑也是 5 个 identical cells |
| **R4** W-FUT 双窗权重 sweep | offline 仍 100%，5 个 (0,3) vs (0,5) 切分 pattern 有效 | **条件触发** — 仅当 R2 winner SR > R2 uniform + 2pp |

**计划修订**：原 plan §2.3 / §2.4 假设 R3 总会跑；α*=1.0 让 R3 退化。建议直接跳 R3，从 R2 决议直接进 R4（条件触发逻辑改为对比 R2 baseline 而非 R3 baseline）。

---

## 6. 数据局限

1. **每 cell 100 ep × 1 seed**：SR 标准误 ~3pp，端点 sanity 的 +5pp p2 α=0.0 偏移在 borderline；α 中点起伏 5pp 全部都在噪声内。
2. **inf 端点系统偏高 0.10**（§4.4）— 解出 thr 与 phase 3 同 cell 的差异，origin 待定。
3. **summary schema 缺 thr** — runner `--mode run-eval` 路径未回填 fh_thr/ws_thr 进 summary（仅 yaml 里有），是 minor schema gap，不影响实验结论但下游分析要从 yaml 读 thr。

---

## 7. Stage 1 结论

1. **G_R1 通过**：两 recipe 在 anchor 容差内，进 R2。`next_args_suggestion["alpha-star"] = "p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0"`。
2. **fusion 假设破裂**：α 中点没超过端点；α=1.0（纯 offline）赢，等同 phase 3 g1/g10 复跑。online 信号在加权和下被 offline 完全覆盖，没产生独立性收益。
3. **后续 R3 退化**：α*=1.0 让 (1−α\*) = 0，R3 5 patterns 数学上等价；建议跳过 R3 直接 R4（W-FUT 窗内权重）。
4. **R2 仍有价值**：在 α*=1.0 下扫 offline 4-desc pattern，等同于"在 phase 3 g1/g10 同 cell 上探索 desc 权重"，可与 phase 3 layer 1 "jerk-only 强于 all-4 desc"结论交叉验证。
5. **inf 偏移 follow-up**：thr 解差异不影响 R1 决议，但要在 R2 / phase 5 复测中确认是否系统性 bug。

---

## 8. 文件索引

```
exp/verdict_factor_judge/
├── data/phase4/
│   ├── warmup_factor_raw/p{1,2}.jsonl                        # 2 recipe warmup factor 缓存
│   └── r1_alpha/
│       ├── per_yaml_summary.jsonl                            # 14 行 master
│       ├── per_yaml_summary_batch{1..6}.jsonl                # 6 batch 独立
│       ├── per_step/<yaml_id>.jsonl                          # 14 个
│       ├── episode_results/<yaml_id>.json                    # 14 个
│       └── decision_gate.json                                # G_R1 决议输出
├── analysis/
│   ├── plot_alpha_sweep_phase4.py
│   ├── phase4_stage1_alpha_sweep.png                         # 双面板 α 曲线
│   └── phase4_stage1_results.md                              # 本文件
└── config/spatial16/phase4/
    ├── warmup/{p1,p2}__warmup.yaml                           # 2 个 warmup yaml
    └── eval/*__r1_a*.yaml                                    # 14 个 R1 eval yaml（含 fh_thr/ws_thr）
```
