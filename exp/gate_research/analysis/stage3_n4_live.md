# Stage 3a — N4 混合门 live 判决

- **Status**: Done（live 6 run × 500 ep + 1 补档 periodic，2026-07-05）
- **Branch**: Ziyang · **Plan**: `logs/gate_stage3_n4_hybrid.log.md` · **Roadmap**: `logs/gate_exploration_roadmap.log.md` §5 Stage 3
- **拓扑**: server=jupyter-ziyang10（pi05_libero，3 replica / 2 spawn-batch，H200）；client=timan107（48 worker）；broker=racknerd（force_single_active，`--ack-alerts` 放行 expose）

## Headline

**N4（V1 跳预测-MISS ⊕ V2 连续缓存执行 run-length ≥ L 强制注入）在低 L（L=6，频繁注入）下，两个 suite 都胜过 same-inf_ratio 的 matched periodic**，且 6/6 操作点都 SR ≥ baseline、net@34 ≥ 0（从不倒退、延迟净正）。**判决：N4 胜出，赢点 L=6 → 推进 3b 服务器化。** spatial 随 L 增大退化（L8/L12 输给 periodic）；l10 全 L 稳（且 l10 上 N4 对 periodic 是 SR + 延迟双赢）。

## 实验设计

- **N4 客户端状态机**（`exp/gate_research/n4_gate_client.py`，零 src，驱动现成 `ClientControlledGate`）：内嵌复用不改的 `N1GateState`（V1，θ/j/M 沿用 1a A 点）+ V2 分支（连续 FULL_HIT run-length ≥ L → 强制 skip 注入一次新推理，`_last_v2` 冻结使 N1 相位不被污染）。
- **矩阵**：L ∈ {6, 8, 12} × suite ∈ {libero_spatial, libero_10} = 6 run × 500 ep。(θ,j,M)：spatial 0.968929/0.968929/3/3；l10 0.996873/0.996873/3/3。
- **及格线（roadmap C10，三条件同真）**：matched periodic 按 **live inf_ratio 最近**配对（容差 0.03，非 skip%）下 **N4_SR ≥ periodic_SR**（periodic_pass_n4）∧ **SR ≥ baseline − 1pp**（sr_ok）∧ **net@stock_2.6k ≥ 0**（net34_ok）。
- **matched periodic**：复用 1b 的 spatial_A(cache7/inf1)/spatial_B(cache4/inf1)/l10_A(cache4/inf1)/l10_B(cache2/inf1) 4 锚；l10 N4 的 inf≈0.65 落在既有 l10 periodic（0.70/0.76）容差外 → **补档 l10 cache12/inf1**（inf=0.663，matched 全 3 l10 N4，|Δinf|≤0.015）。spatial N4 的 inf≈0.29-0.30 由 spatial_A（0.280）覆盖。

## 判决（overall = 三条件同真）

| run | L | skip% | inf_ratio | SR(run/base) | matched periodic (inf/SR) | ΔSR | net@34 | overall |
|---|---|---|---|---|---|---|---|---|
| spatial_n4_L6 | 6 | 17.2 | 0.289 | **92.4**/82.6 | spatial_A (0.280 / 90.4) | **+2.0** | +5.4 | **✅ pass** |
| spatial_n4_L8 | 8 | 16.7 | 0.300 | 88.8/82.6 | spatial_A (0.280 / 90.4) | −1.6 | +2.0 | ❌ fail |
| spatial_n4_L12 | 12 | 15.1 | 0.303 | 85.8/82.6 | spatial_A (0.280 / 90.4) | −4.6 | +0.5 | ❌ fail |
| l10_n4_L6 | 6 | 21.8 | 0.658 | **81.6**/77.6 | l10_c12 (0.663 / 78.4) | **+3.2** | +0.8 | **✅ pass** |
| l10_n4_L8 | 8 | 21.6 | 0.652 | 81.0/77.6 | l10_c12 (0.663 / 78.4) | +2.6 | +2.4 | **✅ pass** |
| l10_n4_L12 | 12 | 21.2 | 0.648 | 78.4/77.6 | l10_c12 (0.663 / 78.4) | +0.0 | +3.7 | **✅ pass** |

periodic 参照（same 50-init 协议）：spatial_A 90.4@inf0.280(net@34 +5.8) · spatial_B 89.0@0.369(−18.1) · l10_A 82.2@0.701(−12.8) · l10_B 82.4@0.759(−25.7) · **l10_c12 78.4@0.663(−5.7)**。baseline SR：spatial 82.6 / l10 77.6。

**分量分解**：6/6 都 sr_ok ✓ + net34_ok ✓；periodic_pass_n4 = 4 pass / 2 fail（spatial L8/L12）。

## 解读

1. **剂量效应（V2 频率）**：SR 随 L 增大（注入变稀）单调下降——spatial 92.4→88.8→85.8、l10 81.6→81.0→78.4。**低 L（频繁注入）= 高 SR**，与 Stage 2a H1「低剂量注入即达增益」同向、并进一步显示"更频繁更好"（至少到 L=6）。
2. **spatial**：只有 L=6 胜 matched periodic（+2.0）；L=8/12 输给 spatial_A（一个很稀的 cache7/inf1 periodic，SR 90.4@inf0.280 且 net 正）。spatial_A 是强对照——N4 只在最激进档压过它。
3. **l10**：全 L 胜 matched periodic，且是**双赢**——N4 l10 net@34 全正（+0.8/+2.4/+3.7），而任何 l10 periodic net@34 全负（l10_c12 −5.7、l10_A −12.8、l10_B −25.7）。即 l10 上 N4 同时给出更高 SR 和更低延迟成本；periodic 要靠推理暴增换 SR（net 崩），N4 不用。
4. **N4 从不倒退**：6/6 SR ≥ baseline（ΔSR +3.2~+9.8 spatial、+0.8~+4.0 l10），net@34 全正。V1 分支保证延迟，V2 分支加 SR。

## Discussion（反 narrative — 本实验**不**证明什么）

- **不是全面 dominance**：spatial L=8/L=12 输给 periodic（spatial_A）。N4 的 spatial 优势**仅**在 L=6 成立，且对 spatial_A 这个特定稀 periodic 而言优势小（+2.0pp）。
- **单 run、无重复、pi05 随机**：每 cell 仅 1×500ep，pi05 采样方差经验 ~±2-3pp。spatial L6 的 +2.0pp 在噪声量级内，**不能**声称统计显著；l10 的 +3.2/+2.6pp 更稳但同样单 run。McNemar b/c 已在 analyzer 输出（如 spatial L6 b/c=19/68），但未做多 run CI。
- **matched periodic 是插值锚不是穷举前沿**：每个 N4 点只对**最近**一个 periodic（容差 0.03）比，不是对整条 periodic 前沿。spatial 三点都配到 spatial_A（0.280），而 N4 inf（0.289-0.303）在其之上——比较靠 0.03 容差成立（spatial L6 |Δinf|=0.009 紧、L12 |Δinf|=0.023 松）。
- **l10 补档 periodic（cache12）SR 仅 78.4**（勉强过 baseline 77.6）——N4 l10 胜的是一个**弱** periodic；这说明在 inf≈0.66 这个低注入档，periodic 几乎无 SR 增益，而 N4 的 score-targeted V1 + 定向 V2 能挤出 +3pp。不代表 N4 胜过**所有** l10 periodic 配置（高 inf 的 l10_A/B 有 82.2/82.4 但 net 崩）。
- **反事实边界（C8/C11）**：本判决全部基于 live 可观测量（actual SR + inf_ratio），未做离线反事实——符合 Stage 3 定位。
- **数据完整性注记**：spatial_n4_L6 首跑因编排中途 kill/resume 导致 per-step/journal 不一致，已**干净重跑**（journal/rows units 500/500，diff=0）后取值。

## 决策推荐 → 3b

- **N4 胜出，赢点 L=6**（两 suite 唯一都 pass 的档）。**3b 服务器化用 L=6**：扩展 `ScoreHysteresisGate` 加连续缓存执行 run 计数器 + L=6 注入分支（1c 管道现成，L2 小改）。
- **部署档位**：延迟档仍用 N1-A（纯 V1）；**SR 档用 N4 L=6**；l10 尤其推荐（SR + 延迟双赢）。spatial 用 N4 L=6 有小 SR 增益但需注意剂量敏感（勿用高 L）。
- **押后/存疑**：spatial 高 L 退化的机制、多 run CI、N4 是否胜过高-inf l10 periodic（当前只比了 matched-inf 的弱 periodic）。

## Artifact layout

- **代码**（已 commit `251eddc`）：`exp/gate_research/n4_gate_client.py` · `worker_entry_n4.py` · `run_n1_live.py`(--gate-family/--L) · `analyze_n1_live.py`(match_periodic_n4 + n4_overall)。
- **本报告**：`exp/gate_research/analysis/stage3_n4_live.md`（本文件，tracked）。
- **raw data（gitignored，已本地化 `exp/gate_research/data/n1_live/`）**：6 N4 run + l10_periodic_c12 的 journal/rows/manifest（各 500 ep）。本地复跑 `analyze_n1_live` 与 live 流式结果 **byte-identical**（判决可本地复现）。
  - **传输注记**：正常通道当时全堵——broker（force_single_active）JetStream ObjectStore storage 耗尽 → `tether pull` 任何尺寸 bucket_create 失败；本地 curl 外网 egress 被权限层拒。**绕过**：timan107 起 `python -m http.server` + `tether expose` → owner 手动 curl 拉 tarball（sha256 `11fb6c4c…` byte-identical 校验通过）。分析亦在 timan107 上跑、`tether exec` 流式取结果。
- **补档 config（gitignored）**：`exp/gate_research/config/libero_10/periodic_C12/`（cache12/inf1，timan107）。
