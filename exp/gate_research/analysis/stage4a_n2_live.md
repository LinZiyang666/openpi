# N2 追随赢家门 Live 验证结果（Stage 4a）

**一句话结论**：N2（follow_winner 锁步盲回放）在 6 个操作点中 **1 点通过 pass-line**（libero_10 · budget=3，最低剂量）；其余 5 点 FAIL。N2 的**延迟净收益 net@34 在全部 6 点为正**（稳健价值面），但 **SR 只在「低剂量 × inf-matched periodic 恰为近-baseline」的窄区间竞争性**。复现并扩展 N1「periodic=SR 最优」结论，并印证 roadmap F12（最低剂量甜点）与 R2（难 suite 盲回放漂移伤 SR）。

网格：`lock_streak=3` 固定，`budget ∈ {3,5,8}`，每 suite 500ep（10 task × 50 trial）。及格线（plan §4.2 / roadmap C10-C11）：**SR ≥ 同 inf_ratio 的 matched periodic（|Δinf| ≤ 0.03）∧ SR ≥ baseline − 1pp ∧ net@stock_2.6k ≥ 0**，三条件同真 = pass。盲回放步 `FULL_HIT × searched=False` 记 inf=0（cache-execution）。

---

## libero_spatial（baseline SR 82.6 / inf 0.287）

| run | gate | skip% | SR(run/base) | ΔSR pp | SR-ok | vs-periodic | overall |
|---|---|---|---|---|---|---|---|
| spatial_n2_b3 | follow_winner | 24.5 | 85.4/82.6 | +2.8 | ok | ΔSR −5.0pp \|Δinf\|=0.012 FAIL | **fail** |
| spatial_n2_b5 | follow_winner | 33.2 | 84.4/82.6 | +1.8 | ok | ΔSR −6.0pp \|Δinf\|=0.015 FAIL | **fail** |
| spatial_n2_b8 | follow_winner | 42.3 | 84.2/82.6 | +1.6 | ok | ΔSR −6.2pp \|Δinf\|=0.021 FAIL | **fail** |
| spatial_A_periodic | periodic | 10.9 | 90.4/82.6 | +7.8 | ok | — | — |
| spatial_B_periodic | periodic | 18.3 | 89.0/82.6 | +6.4 | ok | — | — |

| run | inf(live/base) | Δinf | net@4/34/70 | b/c |
|---|---|---|---|---|
| spatial_n2_b3 | 0.269/0.287 | −0.019 | +6.6/+14.0/+22.8 | 25/39 |
| spatial_n2_b5 | 0.266/0.287 | −0.022 | +7.8/+17.8/+29.8 | 28/37 |
| spatial_n2_b8 | 0.260/0.287 | −0.028 | +10.0/+22.7/+37.9 | 32/40 |
| spatial_A_periodic | 0.280/0.287 | −0.007 | +2.6/+5.8/+9.7 | 26/65 |
| spatial_B_periodic | 0.369/0.287 | +0.081 | −23.6/−18.1/−11.5 | 34/66 |

spatial 3 点全 **FAIL vs periodic**：N2 保 SR vs baseline（+1.6~+2.8），但同 inf_ratio(≈0.27) 下 matched periodic_A(inf 0.280) SR=90.4 **远高于** N2(84-85)，ΔSR −5~−6pp。spatial 上 periodic 在 N2 的 inf 区间即可实现强 V2 增益（盲跳撞新推理 > 缓存重放旧错误，F7），N2 追不上。

---

## libero_10（baseline SR 77.6 / inf 0.636）

| run | gate | skip% | SR(run/base) | ΔSR pp | SR-ok | vs-periodic | overall |
|---|---|---|---|---|---|---|---|
| l10_n2_b3 | follow_winner | 10.9 | 78.8/77.6 | +1.2 | ok | ΔSR +1.2pp \|Δinf\|=0.029 PASS | **pass** |
| l10_n2_b5 | follow_winner | 14.0 | 74.4/77.6 | −3.2 | FAIL | ΔSR −3.2pp \|Δinf\|=0.014 FAIL | **fail** |
| l10_n2_b8 | follow_winner | 19.5 | 75.6/77.6 | −2.0 | FAIL | inf 容差内无候选 | **fail**（SR 退化） |
| l10_A_periodic | periodic | 19.2 | 82.2/77.6 | +4.6 | ok | — | — |
| l10_B_periodic | periodic | 32.7 | 82.4/77.6 | +4.8 | ok | — | — |
| l10_periodic_c12 | periodic | 7.1 | 78.4/77.6 | +0.8 | ok | — | — |
| l10_periodic_c30 | periodic | 2.5 | 77.6/77.6 | +0.0 | ok | — | — |

| run | inf(live/base) | Δinf | net@4/34/70 | b/c |
|---|---|---|---|---|
| l10_n2_b3 | 0.621/0.636 | −0.015 | +4.9/+8.2/+12.1 | 42/48 |
| l10_n2_b5 | 0.636/0.636 | −0.000 | +0.7/+4.9/+9.9 | 53/37 |
| l10_n2_b8 | 0.606/0.636 | −0.031 | +9.9/+15.8/+22.8 | 62/52 |
| l10_A_periodic | 0.701/0.636 | +0.064 | −18.6/−12.8/−5.9 | 43/66 |
| l10_B_periodic | 0.759/0.636 | +0.123 | −35.5/−25.7/−14.0 | 38/62 |
| l10_periodic_c12 | 0.663/0.636 | +0.027 | −7.8/−5.7/−3.1 | 44/48 |
| l10_periodic_c30 | 0.650/0.636 | +0.014 | −4.1/−3.4/−2.5 | 34/34 |

**l10_n2_b3 = 全实验唯一 PASS**：SR 78.8 vs baseline 77.6（+1.2，保真）、配对 periodic_c30(inf 0.650, |Δinf|=0.029) ΔSR **+1.2**、net@34 +8.2，三条件全过。budget↑（b5/b8）盲回放变长 → SR 退化至 baseline−1pp 以下（−3.2 / −2.0），印证 R2（难 suite 库轨迹不可靠，盲回放漂移伤 SR）。

> **为何 l10-b3 过而 spatial 全败？** 取决于 N2 的 inf 区间上「有没有一个强 periodic」。l10 baseline inf 高（0.636，cache 命中率低），要匹配 N2 的低 inf(0.62-0.65) 只能用**近-baseline periodic**（c30 skip 2.5%，SR=77.6，几乎不 gate、无 V2 增益）→ N2-b3 微胜。spatial baseline inf 低（0.287，cache 命中率高），一个 skip 10.9% 的 periodic 即可落到 N2 的 inf(0.28) **且**兑现强 V2 增益（SR 90.4）→ N2 败。即：**pass/fail 取决于 N2 的 inf_ratio 上是否存在 V2-gaining 的 periodic**。

---

## Pass-line 分量（N2 三条件）

| run | sr_ok | periodic_pass | net34_ok | overall |
|---|---|---|---|---|
| spatial_n2_b3 | ok | FAIL | ok | fail |
| spatial_n2_b5 | ok | FAIL | ok | fail |
| spatial_n2_b8 | ok | FAIL | ok | fail |
| l10_n2_b3 | ok | **PASS** | ok | **pass** |
| l10_n2_b5 | FAIL | FAIL | ok | fail |
| l10_n2_b8 | FAIL | pending | ok | fail |

---

## 结论与部署含义

1. **机制 live 验证成立**：smoke 检出 51 个盲回放步（`FULL_HIT×searched=False`）+ locked-tail fail-safe 解锁；budget 剂量-响应单调（searched=False 随 budget 升）。锁步→锁定→盲回放→解锁全链路在真实 3-replica H200 server 上正确工作。
2. **N2 非广谱优，但非全败**：6 点 1 过（l10 最低剂量）。与 N1（对 periodic 全败）相比，N2 在「低 inf × 难 suite」找到一个 periodic 弱、N2 微胜的窄甜点。
3. **稳健价值 = 延迟**：net@34 全 6 点为正（N2 盲回放同时省搜索 + 省推理，d_inf 为负）。这是 N2 相对 miss-skip（N1/periodic-skip）的确定性优势面。
4. **SR 价值 = 窄区间**：仅在最低剂量 + inf-matched periodic 恰为近-baseline 时竞争性。剂量升即漂移退化。
5. **部署判据**：N2 适用于「延迟为硬约束 **且** 目标 inf 区间上不存在 V2-gaining periodic（如高 baseline-inf/低命中率大库）**且** 采用最低盲回放剂量」的窄场景；否则 periodic 在 SR 上更优。

## 溯源

- **代码**：commit `0885d25`（Stage 4a N2 服务器化 `eaa4263` + Phase C 使能 `0885d25`）。
- **拓扑**：server = jupyter-ziyang10（3 replica / 2 batch，H200 NVL，expose linziyang.top:14006）；client = timan107（48 worker）。两 suite yaml 不同时跑（server 靠 conductor `load_cache_config` 热切换 pkl 顺序服务）。
- **N2 runs（新，2026-07-07）**：`spatial_n2_{b3,b5,b8}`、`l10_n2_{b3,b5,b8}`，各 500ep。config：cp1_spatial_pool_16 keybuilder，weighted_score_sum_knn，per-field zscore+tanh 归一（spatial fh75_ws10 / l10 fh5_ws40）。
- **matched periodic**：spatial_{A,B}_periodic、l10_{A,B}_periodic、l10_periodic_c12 复用自 N1 Stage 1b（同 config，inf_ratio 轴配对合法）；**l10_periodic_c30（新）** cache30/inf1，为 l10-b3 补一个近-baseline inf 配对。
- **口径**：skip% 由权威 `searched`（server 盖章 `__hit_meta__`；periodic 按 ordinal 重建）；SR/periodic 配对按 `(task_id, subset_init_state_idx)`，三方 unit 等集；baseline_inf_ratio 来自 Stage-0 gate_rows（episode-global max-attempt 去重）；net@{4,34,70} = SEARCH_MS{opt_2.6k,stock_2.6k,opt_50k} × skip% − INFER_MS × Δinf。
