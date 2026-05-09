# Verdict Factor Judge — Phase 4 Stage 2 (R2 offline 4-desc 权重) 结果分析

数据来源：`exp/verdict_factor_judge/data/phase4/r2_offline_desc/per_yaml_summary.jsonl`（18 行，0 NA，thr 字段已 backfilled）。
画图脚本：临时 inline 脚本 → [`phase4_stage2_r2_patterns.png`](phase4_stage2_r2_patterns.png)。
执行命令清单：[`logs/verdict_phase4_stage2_run_commands.log.md`](../../../logs/verdict_phase4_stage2_run_commands.log.md)。
Stage 1 结果：[`phase4_stage1_results.md`](phase4_stage1_results.md)。

---

## 0. 公式与决策门

```
inf_ratio = (0·n_full_hit + 0.75·n_warm_start + 1·n_miss) / n_eval_verdicts        # warm cost = 0.75
score(α)  = SR − 0.5 · inf                                                          # G_R1 综合指标

G_R2 (per recipe):
  argmax_pattern SR；若 argmax SR − uniform SR ≤ 2pp → 强制 pattern* = uniform
```

---

## 1. 实验配置（来自 stage 1 winner）

```
ALPHA_STAR = 'p1_state_fut_online_act=1.0,p2_action_fut_online_act=1.0'
```

α\* = 1.0 在两 recipe 上（stage 1 G_R1 winner）→ R2 在 **纯 offline 配置**下扫 9 个 desc-share pattern。online 总权重 = (1−α\*) = 0，10 个 score 中只有 8 个 offline 起作用。

每 recipe 9 pattern × 2 recipe = **18 eval cell**；locked cell：p1 (0.5, 0.5)、p2 (0.5, 0.4)；warmup factor_raw 复用 stage 1 cache（不重跑）。

---

## 2. 主表

```
recipe                         pattern             SR    inf  hit%  warm%  miss%   score  fh_thr  ws_thr
─────────────────────────────────────────────────────────────────────────────────────────────────────────
p1_state_fut_online_act        uniform          0.960  0.468   43%    39%    17%   0.726   0.430   0.060  ←  R2 winner
p1_state_fut_online_act        jerk-heavy       0.960  0.472   43%    40%    17%   0.724
p1_state_fut_online_act        dir-heavy        0.950  0.476   42%    40%    18%   0.712
p1_state_fut_online_act        disp-heavy       0.950  0.474   43%    40%    17%   0.713
p1_state_fut_online_act        path-heavy       0.930  0.484   41%    41%    18%   0.688
p1_state_fut_online_act        jerk-only        0.960  0.500   40%    42%    19%   0.710
p1_state_fut_online_act        dir-only         0.940  0.480   42%    40%    18%   0.700
p1_state_fut_online_act        disp-only        0.920  0.415   45%    55%     0%   0.713
p1_state_fut_online_act        path-only        0.830  0.482   42%    38%    20%   0.589  ← outlier
─────────────────────────────────────────────────────────────────────────────────────────────────────────
p2_action_fut_online_act       uniform          0.920  0.520   40%    33%    28%   0.660
p2_action_fut_online_act       jerk-heavy       0.950  0.512   41%    32%    27%   0.694
p2_action_fut_online_act       dir-heavy        0.940  0.528   38%    35%    27%   0.676
p2_action_fut_online_act       disp-heavy       0.910  0.508   41%    31%    27%   0.656
p2_action_fut_online_act       path-heavy       0.940  0.509   41%    32%    27%   0.686
p2_action_fut_online_act       jerk-only        0.930  0.492   43%    33%    25%   0.684
p2_action_fut_online_act       dir-only         0.970  0.503   41%    37%    23%   0.719  ← R2 winner
p2_action_fut_online_act       disp-only        0.930  0.410   45%    55%     0%   0.725
p2_action_fut_online_act       path-only        0.900  0.508   41%    35%    25%   0.646
```

---

## 3. R2 决策门（G_R2）

| recipe | uniform SR | argmax pattern | argmax SR | Δ vs uniform | winner | R4 trigger? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| p1 | 0.960 | jerk-heavy / jerk-only / uniform 并列 0.96 | 0.960 | **+0.000** ≤ 2pp | **uniform**（强制） | ❌ |
| p2 | 0.920 | dir-only | 0.970 | **+0.050** > 2pp | **dir-only** | ✓（按字面规则） |

`next_args_suggestion["offline-pattern"] = "p1_state_fut_online_act=uniform,p2_action_fut_online_act=dir-only"`

---

## 4. 关键发现

### 4.1 p1 上 desc 权重无显著影响

p1 9 个 pattern 的 SR 集中在 0.83–0.96（除 path-only 异常 0.83 外，其余 8 个 0.92–0.96，全部在 100 ep 噪声内）。argmax 与 uniform 持平 → G_R2 强制 uniform。**p1 的 4 desc 之间没有可识别的相对权重偏好**。

注意 `disp-only` cell 极特殊：**hwm = 45/55/0，inf=0.415**（最低）。fh_thr=0.500、ws_thr=0.000 退化（与 phase3 g8/g11 dispersion-only 退化同款）— solver 在 dispersion 单 desc 下切到 ws_thr=0，所有 verdict 都进 FH/WS、零 miss。这与 phase3 reults §4.2 的"dispersion 单 desc → thr=0 退化"现象一致。

`path-only` SR=0.83 是个 outlier（比相邻 desc-only 低 9–11 pp），但 thr 范数正常（fh=0.44, ws=0.02），不是退化；可能是 path_length 单 desc 上 score 信号太弱让 hit 决策走偏，或者 100 ep noise outlier。

### 4.2 p2 上 dir-only 看似 winner，但与 R1 α=1.0 的 noise 矛盾

p2 R2 uniform SR = 0.920，dir-only SR = 0.970 → Δ = +0.05 pp，按 plan §3.2 规则 R4 该触发。**但 R1 α=1.0（数学上等价 R2 uniform）SR=0.970**，差 R2 uniform 5pp。下小节 §4.3 量化这个矛盾。

### 4.3 R1 α=1.0 vs R2 uniform 等价性 sanity（重要）

R1 α=1.0 与 R2 uniform 数学上**完全等价**：

```
R1 α=1.0:    offline 8 keys × 1/8 each + online 2 keys × 0 each
R2 uniform:  offline 8 keys × (α/8)/(1)·(1/4) × 2 = 1/8 each + online × 0
              shares (1,1,1,1) × 2 windows uniform
```

实际 sanity（同 cell 跑两次）：

| recipe | variant | fh_thr | ws_thr | SR | inf | hit/warm/miss% |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **p1** | R1 α=1.0 | **0.4300** | **0.0600** | 0.960 | 0.472 | 43/39/18 |
| **p1** | R2 uniform | **0.4300** | **0.0600** | 0.960 | 0.468 | 43/39/17 |
| | Δ | **+0.0000** | **+0.0000** | **+0.000** | −0.004 | |
| **p2** | R1 α=1.0 | **0.4525** | **0.2675** | **0.970** | 0.506 | 41/34/25 |
| **p2** | R2 uniform | **0.4525** | **0.2675** | **0.920** | 0.520 | 40/33/28 |
| | Δ | **+0.0000** | **+0.0000** | **−0.050** | +0.014 | |

**关键结论**：
1. **thr 完全一致**（fh / ws 全部 Δ=0）✓ — solver 在数学等价 weight 下解出相同 thr，stage 1 留下的"thr 反推一致性"open issue 这里**自洽**（thr 部分）。
2. **p1 SR 一致**（同 0.96）→ 行为高度可重复。
3. **p2 SR 差 5pp**（0.97 vs 0.92） → 同 cell 跑两次出现 5pp SR 偏差。

5 pp 在 100 ep × 1 seed 的 SR 标准误（≈3pp）下是 ~1.5σ，**borderline 不显著**。结合 phase3 同 cell 重跑也观察到 ~3-5pp 噪声（phase3_results §6 局限性），LIBERO sim 在 multi-worker async 下不严格 deterministic 是已知问题。

### 4.4 R2 决议的可信度问题

p2 的 R2 winner = dir-only **依赖 R2 uniform 那一次跑的 SR=0.92**。如果 R2 uniform 真值更接近 R1 α=1.0 的 0.97（同 cell 再跑可能给 0.95–0.97），那 dir-only 0.97 vs uniform 真值的 Δ 就会落到 ±2pp 之内，触发"force uniform"——决议反转。

| 场景 | p2 uniform SR（假设） | dir-only Δ | R2 winner |
|---|:---:|:---:|---|
| R2 跑出来的实测 | 0.92 | +0.05 | dir-only |
| 用 R1 α=1.0 复跑值替代 | 0.97 | 0.00 | uniform（force） |
| 取两次平均 | 0.945 | +0.025 | dir-only（边缘） |

**判断**：p2 的 dir-only "胜出"是 100 ep × 1 seed 噪声驱动的边缘决议，**不强可信**。phase 5 1000 ep × 1 seed 复测才能锁。

---

## 5. R3 / R4 影响

### 5.1 R3 跳过（plan §9 已决定）

α\*=1.0 让 (1−α\*)=0 → 5 个 online pattern 数学上完全等价（全 0 weight，5 个 identical cells），跑没意义。**确认跳过 R3**。

### 5.2 R4 触发判断

按 plan §3.3 R4 trigger 规则（reframe 为对比 R2 baseline，详见 stage1 results §5）：

| recipe | R2 winner | R2 uniform | Δ | R4 触发？ |
|---|---|:---:|:---:|:---:|
| p1 | uniform（force） | 0.960 | 0.000 | **❌** — winner = uniform 等于 baseline，没必要扫窗 |
| p2 | dir-only | 0.920 | +0.050 | **⚠ borderline** — 字面触发，但 §4.3 显示 uniform 真值不稳，触发条件可能是 noise |

**建议（per-recipe）**：

- **p1 R4：不跑**。R2 已确认 desc 权重不影响 SR（与 uniform 持平），W-FUT 双窗权重比 desc 更细粒度，更不可能有显著影响。
- **p2 R4：可选，建议跑**——只是为了 sanity（看在 dir-only 下扫双窗能否稳住或拉高 SR）。但**不能**期望 winner 比 dir-only 高 ≥ 2pp（已超出 stage 1/2 数据支持的信号强度）；R4 主要价值是给 phase 5 1000ep 复测候选 cell 扩到双窗维度。

### 5.3 fusion 假设的整体审判

| stage | 关键发现 | 对原始假设的支持 |
|---|---|:---:|
| stage 1 (R1 α 扫描) | α 中点全在噪声内，α=1.0 winner | ❌ 不支持 |
| stage 2 (R2 desc 扫描) | p1 desc 权重无影响；p2 dir-only 字面 winner 但 noise driven | ❌ 不支持 |

phase4 plan §0.3 的核心假设（"online + offline 误差独立 → 加权融合给凸响应"）**两轮都没在数据上得到证实**。R2 进一步显示，即使在"纯 offline"下扫 desc 权重，p1 上 4 desc 间也无可识别偏好（与 phase3 layer1 "jerk-only 强于 all-4 desc" 的结论部分吻合：那次结论也是 100 ep noise borderline）。

---

## 6. inf 端点 follow-up（stage 1 留下的）

stage 1 §4.4 报告 phase4 R1 cell 的 thr 比 phase3 同 cell 高 ~0.05 → inf 偏高 0.10。stage 2 R2 uniform thr = (0.43, 0.06)，与 R1 α=1.0 完全一致（§4.3 证）→ **phase4 内部 self-consistent**。

但 **phase4 vs phase3 的 thr 偏移仍未解释**：

```
phase3 g1 (0.5, 0.5):     fh_thr = 0.380   ws_thr = 0.000   inf = 0.372
phase4 R1/R2 同 cell:     fh_thr = 0.430   ws_thr = 0.060   inf = 0.468
                          ↑ +0.05         ↑ +0.06          ↑ +0.10
```

**剩余 hypothesis**（未排除）：
1. phase4 warmup yaml 含 10 factor block（offline 8 + online 2），phase3 g1 含 8。即使 online weight=0 不影响 score，extract 路径在 view fetch 时序上可能差异 → factor_raw 序列分布不同 → derive_thresholds quantile 切位偏高。
2. PercentileRollingCalibration 的 saturated buffer 在 phase4 (10 keys bind) vs phase3 (8 keys bind) 下 last-50 trim 行为可能微妙差异。

**phase 5 复测建议**：用 1000 ep × 1 seed 重跑 phase 3 g1 / g10 在 (0.5, 0.5) / (0.5, 0.4) cell + phase 4 p1/p2 R2 uniform，对比 thr 差异是否 reproducible，并量化 inf 偏移的真值。

---

## 7. 数据局限

1. **每 cell 100 ep × 1 seed**：SR 标准误 ~3pp。p2 dir-only vs uniform 的 5pp 差是 ~1.5σ borderline，不是显著信号。
2. **R2 uniform 与 R1 α=1.0 SR 5pp 差**（§4.3）：sim 不严格 deterministic 已知；phase 5 1000 ep 才能 disambiguate winner。
3. **path-only outlier**（p1 SR=0.83）：单点 9pp 偏离，不能确定是真信号还是 100 ep noise tail。
4. **inf 偏移 follow-up 未关闭**：phase3 vs phase4 thr 偏 0.05 仍待 phase 5 复测排查。

---

## 8. Stage 2 结论

1. **R2 决议**：
   - p1 → pattern\* = **uniform**（force，4 desc 间无显著权重偏好）。
   - p2 → pattern\* = **dir-only**（SR=0.97 vs uniform 0.92，字面 +5pp；但 noise driven 边缘决议，可信度低）。
2. **fusion 假设破裂确认**：stage 1 + stage 2 均未在数据上证实"online + offline 加权融合"凸性收益。phase4 整体没超过"phase3 g1/g10 单独配置"。
3. **R3 跳过**（α\*=1.0 让 R3 退化）。
4. **R4 建议**：p1 不跑；p2 可选，作为 phase 5 候选扩展（不期望显著 lift）。
5. **stage1 follow-up**：phase4 内部 thr 自洽（R1 α=1.0 与 R2 uniform thr 完全一致）；phase3 vs phase4 跨阶段 thr 偏移仍未关闭。
6. **phase 5 准入候选 cell**（1000 ep × 1 seed 复测）：
   - p1 R2 uniform = R1 α=1.0
   - p2 R2 uniform（确认 SR 是否真在 0.92 还是 0.97）
   - p2 R2 dir-only（验证 winner 是否 reproducible）
   - phase3 g1 (0.5, 0.5)、g10 (0.5, 0.4) 真值复测（核 anchor）

---

## 9. 文件索引

```
exp/verdict_factor_judge/
├── data/phase4/r2_offline_desc/
│   ├── per_yaml_summary.jsonl                            # 18 行 master（thr 已 backfilled by 0e0fb6d）
│   ├── per_yaml_summary_batch{1..6}.jsonl                # 6 batch 独立
│   ├── per_step/<yaml_id>.jsonl                          # 18 个
│   ├── episode_results/<yaml_id>.json                    # 18 个
│   └── decision_gate.json                                # G_R2 决议输出
├── analysis/
│   ├── phase4_stage2_r2_patterns.png                     # 4-panel: SR + inf × p1/p2
│   ├── phase4_stage2_results.md                          # 本文件
│   ├── phase4_stage1_alpha_sweep.png                     # stage1
│   └── phase4_stage1_results.md                          # stage1
└── config/spatial16/phase4/eval/
    └── *__r2_a1.0_off-*.yaml                             # 18 个 R2 eval yaml（commit a0eb43f）
```
