# R12 opus — pilot analysis + frozen test-A predictions (2026-10-03 ~20:25 CDT)

（子代理交回原文，由协调者代存。）

## 给 owner 的白话摘要

**三个方法在说什么**
- **方法一**：只看手腕，觉得可疑时再补看第三人称。旧守卫照用。
- **方法二**：不靠第三人称画面的新守卫。两路相机照常看。
- **方法三**：新守卫，平时只看手腕。

**结论**
1. **省视觉这部分完全按预测兑现。**
   - 只看一路相机的比例和事先预测几乎一样，十个用到只看一路的组里每组都差不到 0.05。
   - 每个决策点的推理开销：方法一省 0.009–0.016，方法三的视觉部分省 0.013–0.014。
2. **新守卫叫大模型的次数少了 15–40%，而且几乎都少在本来就会成功的局里。** 失败局里叫的次数和现部署系统一样。
3. **但新守卫第一次叫得更晚。**
   - π0.5 长任务平均晚约 11 个决策点，其他格子晚 2–6 个。
   - 我事先定的停止规则是"失败局里第一次调用晚 3 个以上决策点就停"。它在 π0.5 长任务上明确触发；在 π0.5 空间任务和 GR00T 长任务上勉强触发。
4. **直接看后果，看不到"晚叫"带来的损失。**
   - 没有任何一局失败是从头到尾没叫过大模型。
   - 旧守卫本该很早就叫的那批局，换成新守卫后，失败率并不比同一初始状态下的现部署系统高。
   - 100 局的成功率差别都在噪声以内。
   - 我对 500 局的诚实估计：长任务约少 0.5 个百分点，空间任务约持平。单个格子测试 A 分辨不出 3 个百分点以内的差别。
5. **开旋钮格子的"晚叫"警报是假警报。** 旋钮在第 2 步就叫了大模型；把旋钮的调用算上，各臂第一次调用的时间完全一样。
6. **新发现：GR00T 在本机 4090 和 H100 上表现不同，差距比任何方法的效果都大。**
   - 空间任务上，成功率差约 5 个百分点，开销差约 0.015。
   - 有几组 GR00T 对比混用了两种服务器。按同一服务器重新比：
     - 空间任务上，方法一几乎没省（多出的调用吃掉了视觉的节省）；
     - 方法二、三在空间任务上反而更好；
     - GR00T 长任务的方法二、三没有同服务器的对照。
7. **测试 A 之前方法本身不用改。需要改的是实验安排：**
   - 同一格的所有臂放在同一种服务器上跑；
   - GR00T 长任务补跑一个 H100 上的现部署对照（100 局）；
   - 新守卫的停止规则在三个格子上触发了，是否照样上测试 A 需要你裁定。我的建议是照跑，但记为"越过筛选"。如果严格按规则，就去掉 π0.5 长任务上方法二、三两臂。
8. **测试 A 的预测已冻结（技术部分第 5 节）。**
   - 每个决策点的开销：方法一约省 0.010；方法二约省 0.007，全部来自少叫；方法三约省 0.021。
   - 成功率预测各格都在 −0.6 到 +0.3 个百分点之间，也就是基本持平。

---

## 0. Data and method

**Arms used**
- Method arms: my M1, H2 and M2 arms.
- Deployed replicates, excluding every astra-named arm: fix1 off arms; guard-root off arms (pi05 L10-50, the periodic cell, pi05 Sp-50, groot Sp-50); the old invalid look-half pilot arms (`execution_audit=false`, deployed replicates); the `r12_devlooks_*` recordings.

| cell | deployed replicates |
|---|---|
| pi05 L10-50 | 5 |
| pi05 Sp-50 | 6 |
| pi05 Sp-200 | 4 |
| groot Sp-50 | 5 |
| groot L10-50 | 3 |
| pi05 L10-50 periodic .32 | 2 |

**Not available**: `r12_guard_groot_l10_50_off` never ran; the fix2 pi05 Sp-200 H2/M2 screen had not started.

**Guard replica check.** The recorded `_noprog_span >= 2` reproduces every deployed guard call on all 18 deployed and M1 arms, with 0 mismatches. On H2/M2 arms it is therefore exactly the deployed rule's verdict on the method's own trajectory.

**Serving host.** `server_2310x` = local 4090; `server_2323x` = H100.

| arm group | host |
|---|---|
| H2/M2 pi05 L10-50, pi05 Sp-50 | H100 |
| H2/M2 groot L10-50 | H100 |
| H2/M2 groot Sp-50 | local |
| H2/M2 periodic | local |
| M1 pi05 L10-50, groot L10-50 | local |
| M1 pi05 Sp-50, pi05 Sp-200, groot Sp-50 | H100 |
| R11 test A (all arms) | H100 |

- GR00T Sp-50 deployed, same selection: H100 (2 reps) SR .88, IR .1308, 1.56 calls/ep; local (3 reps) SR .833, IR .1460, 2.09 calls/ep. Within-host spread .02–.06 calls/ep → host effect is real.
- pi05 shows no host effect: L10-50 IR .1685 vs .1688; Sp-200 .103 vs .100.
- GR00T compared within host; pi05 uses the all-host pool.

**Readouts refreshed**: re-ran `rounds/r12/guard/report.py` on the 5 guard roots. Caveats: groot Sp-50 root off arm is H100 while H2/M2 are local (host-confounded); groot L10-50 `off` pending.

**Tools / outputs** (CPU only): `rounds/r12/opus/tools/pilot_{extract,analyse,stats,host}.py`; `rounds/r12/opus/out/` — `pilot/*.npz`, `pilot_{analysis,stats,host}.{txt,json}`, `pilot_ir_decomp.txt`, `pilot_early_stall.txt`, `pilot_h2_misses.txt`.

**Conventions**: ΔIR at owner compute-share prices, split camera / calls; ΔSR paired by init against the per-init deployed mean (SE in parentheses); "first call" in decisions.

## 1. Predictions vs outcome

### M1 (PREDICTION Section B)

| cell (host) | one-camera share, predicted → observed | ΔIR predicted → observed (camera / calls) | ΔSR predicted → observed | calls/episode | verdict |
|---|---|---|---|---|---|
| pi05 L10-50 (local) | .23 [.13,.33] → .228 ✓ | −.011 [−.015,−.006] → −.0131 (−.0087/−.0044) ✓ | −0.2 → +4.4 (2.9) | −7.8% | held |
| pi05 Sp-50 (H100) | .30 [.18,.40] → .259 ✓ | −.012 [−.016,−.007] → −.0141 (−.0101/−.0040) ✓ | −0.2 → +0.7 (3.4) | −6.6% | held |
| pi05 Sp-200 (H100) | .45 [.30,.55] → .398 ✓ | −.018 [−.023,−.012] → −.0106 (−.0155/**+.0048**) ✗ | −0.15 → −1.8 (1.9) | +22% | camera part held; calls missed |
| groot L10-50 (local) | .12 [.06,.20] → .120 ✓ | −.007 [−.010,−.004] → −.0115 (−.0045/−.0072) ✗, larger saving | −0.3 → +5.3 (4.4) | −9.9% | camera part held |
| groot Sp-50 (H100, same host) | .31 [.18,.40] → .265 ✓ | −.015 [−.020,−.009] → **−.0006** (−.0100/+.0094) ✗; −.0097 vs all-host pool | −0.4 → −4.0 (4.0) | +24% | camera part held; calls missed |

- Camera part held in every cell (share within .05 of center; camera savings matched).
- Calls part: calls per failing / winning episode unchanged under M1:

| cell | per failing ep, M1 vs deployed | per winning ep, M1 vs deployed |
|---|---|---|
| pi05 L10-50 | 20.0 vs 20.5 | 3.86 vs 3.51 |
| pi05 Sp-50 | 9.1 vs 9.5 | .92 vs .95 |
| groot L10-50 | 25.3 vs 25.8 | 5.37 vs 5.27 |
| pi05 Sp-200 | 7.2 vs 9.3 | .46 vs .40 |
| groot Sp-50 | 8.3 vs 7.9 | .71 vs .69 |

  Net call deltas track how many episodes failed (noise at 100 eps): Sp-200 4 failures vs 2.2 expected; groot Sp-50 16 vs 12; groot L10-50 25 vs 30.
- No missed-call starvation: first call paired by init over all episodes moved −1.7 / −0.2 / −0.4 / +0.9 / −1.0 decisions.

### H2 (GUARD_PREDICTION)

| cell (host) | calls per look Δ (pred ±.03) | ΔIR (pred ±.015) | share of calls in failing eps (pred +.05..+.15) | first call in failing eps | ΔSR (pred +0.5) |
|---|---|---|---|---|---|
| pi05 L10-50 (H100) | −.034 (edge) | −.0149 ✓ | .59→.65 ✓ | pred ≤ +3 → **+12.8** ✗ | +0.4 (3.2) |
| pi05 Sp-50 (H100) | −.023 ✓ | −.0102 ✓ | .57→.58 ✗ | pred not later → +5.5 ✗ | +1.7 (2.0) |
| groot Sp-50 (local vs local) | −.031 (edge) | −.0133 ✓ | .72→.77 ✓ | pred not later → −1.1 ✓ | +3.7 (2.0) |
| groot L10-50 (H100 vs local pool) | −.004 ✓ | −.0018 ✓ | .68→.74 ✓ | pred ≤ +3 → +5.1 ✗ | −0.7 (4.8); host-confounded |
| periodic .32 (local) | −.005 ✓ | −.0026 ✓ | — | any-call +0.1 ✓ | 0.0 (3.5) |

### M2 (GUARD_PREDICTION)

| cell | one-camera share pred → obs | ΔIR pred → obs (camera / calls) | ΔSR pred → obs |
|---|---|---|---|
| pi05 L10-50 | .35 → .347 | −.013 → −.0226 (−.0133/−.0093) | +0.3 → −0.6 (3.3) |
| pi05 Sp-50 | .35 → .369 | −.013 → −.0382 (−.0143/−.0239) | +0.3 → +3.7 (3.3) |
| groot Sp-50 (local) | .37 → .354 | −.014 → −.0234 (−.0133/−.0101) | +0.3 → −0.3 (4.0) |
| groot L10-50 | .23 → .221 | −.008 → −.0090 (−.0082/−.0009) | +0.3 → −3.7 (4.6); host-confounded |
| periodic .32 | .22 → .224 | −.009 → −.0141 (−.0086/−.0055) | 0 → 0.0 (3.5) |

**Why the H2/M2 call predictions missed.** θ_b was matched on deployed trajectories, assuming H2 keeps the deployed call rate. In its own closed loop H2 calls 15–51% less per winning episode, about the same per failing episode:

| cell | per winning ep, H2 vs deployed | per failing ep, H2 vs deployed |
|---|---|---|
| pi05 L10-50 | 2.46 vs 3.51 | 19.4 vs 20.5 |
| pi05 Sp-50 | .78 vs .95 | 9.5 vs 9.5 |
| groot Sp-50 (local) | .44 vs .70 | 9.7 vs 9.0 |

Mechanism: "sticky but slow to start" — after a call it calls again with probability .63–.70 (deployed .44–.53); after a non-call look it starts a new call only .07–.09 of the time on pi05 (deployed .12–.17). M2 vision savings predicted almost exactly (within .001–.002). ΔSR predictions all inside 100-episode noise.

## 2. Stop rules: the right comparison

Rule (a) = first call later in failing episodes. The guard readout's +17.7 on the periodic cell counts only reason-4 (guard) calls. Below, "first call" counts calls of any kind; "unpaired" = means over failing episodes vs the deployed pool; "paired, all episodes" pairs by init over every episode (no-call episode counts at last look + 2).

| cell | arm | first call in failing eps, unpaired | first call paired, all eps | calls per failing ep, arm vs deployed | calls per winning ep, arm vs deployed | failing eps with no call |
|---|---|---|---|---|---|---|
| periodic .32 | H2 | +0.1 ± 0.3 (deployed 2.2) | 0.0 | 36.3 vs 34.9 | 12.56 vs 12.83 | 0 |
| periodic .32 | M2 | +0.4 ± 0.4 | 0.0 | 32.3 vs 34.9 | 12.61 vs 12.83 | 0 |
| pi05 L10-50 | H2 | **+12.8 ± 5.2** | **+11.5 ± 1.7** | 19.4 vs 20.5 | 2.46 vs 3.51 | 0 |
| pi05 L10-50 | M2 | **+11.3 ± 4.1** | **+10.4 ± 1.7** | 19.8 vs 20.5 | 2.81 vs 3.51 | 0 |
| pi05 L10-50 | M1 | +3.2 ± 3.2 | −1.7 ± 1.2 | 20.0 vs 20.5 | 3.86 vs 3.51 | 0 |
| pi05 Sp-50 | H2 | +5.5 ± 3.1 | +2.5 ± 0.6 | 9.5 vs 9.5 | .78 vs .95 | 0 |
| pi05 Sp-50 | M2 | +4.0 ± 2.3 | +2.0 ± 0.5 | 8.6 vs 9.5 | .57 vs .95 | 0 |
| pi05 Sp-50 | M1 | +0.1 ± 1.9 | −0.2 ± 0.5 | 9.1 vs 9.5 | .92 vs .95 | 0 |
| pi05 Sp-200 | M1 | +3.6 ± 2.5 (4 failures) | −0.4 ± 0.6 | 7.2 vs 9.3 | .46 vs .40 | 0 |
| groot Sp-50 | H2 | −1.1 ± 2.0 | +2.1 ± 0.6 | 9.7 vs 9.0 (local) | .44 vs .70 | 0 |
| groot Sp-50 | M2 | −0.6 ± 1.7 | +2.5 ± 0.6 | 8.9 vs 9.0 | .34 vs .70 | 0 |
| groot Sp-50 | M1 | +1.3 ± 1.8 | +0.9 ± 0.6 (H100) | 8.3 vs 7.9 | .71 vs .69 | 0 |
| groot L10-50 | H2 | +5.1 ± 1.8 | +3.5 ± 1.1 | 26.6 vs 25.8 | 4.12 vs 5.27 | 0 |
| groot L10-50 | M2 | +6.3 ± 2.2 | +6.5 ± 1.2 | 25.3 vs 25.8 | 4.41 vs 5.27 | 0 |
| groot L10-50 | M1 | −1.5 ± 1.6 | −1.0 ± 0.8 | 25.3 vs 25.8 | 5.37 vs 5.27 | 0 |

- **Periodic cell: flag not meaningful.** Knob makes the first call at decision 2 in every arm; when H2 does not flag, knob calls take the guard's place (knob 977 → 1086, guard 445 → 336, total unchanged). Right comparison +0.0, equal calls in failing eps, 7 failures each. **Pass.**
- **Knob-off cells: flag meaningful as a timing fact.** H2/M2 later in all episodes. By the frozen rule (≥ 3 later): **fails** pi05 L10-50 (both arms, robust); **marginal fail** pi05 Sp-50 (+4..+6 failing; +2.0..+2.5 all) and groot L10-50 (+5..+6, host-confounded); **pass** groot Sp-50.
- **Rule (b), call rates.** M1 never falls ≥ 20% (the +22% / +24% are increases from extra failures). **Pass.** H2/M2 at ±.03 per look: edge fails pi05 L10-50 H2 (−.034), groot Sp-50 H2 (−.031); clear fail pi05 Sp-50 M2 (−.055) — fewer calls in winning episodes, not starvation.

## 3. SR risk of H2's fewer and later calls

**Why H2 skips looks the old rule would flag** (`pilot_h2_misses.txt`): 55–65% are looks where the wrist-only top-1 shows progress > .5 library steps while the two-camera top-1 does not (OFFLINE §12.3 agreement problem); 35–45% are stalls with risk still below θ_a (median risk ≈ θ_a); 1–4% lost to the step ≥ 4 eligibility.

**Failures where the deployed rule fired first and H2 had not yet called:**

| arm | failures with old rule first | lead of old rule | deployed SR on those inits |
|---|---|---|---|
| pi05 L10-50 H2 | 10 of 19 | median 19 decisions | .50 over 5 reps (vs .27 for H2's other failures) |
| groot L10-50 H2 | 19 of 31 | — | .56 (vs .36) |
| pi05 Sp-50 H2 | 4 of 10 | — | .12 |
| groot Sp-50 H2 | 3 of 13 | — | .33 |

Same lateness in winning episodes (33 of 81 pi05 L10-50 winners also had the old rule first). On L10 cells the late failures fall on somewhat easier inits: a weak, L10-specific harm signal.

**Consequence checks**: no failing episode lacks a call; calls per failing episode unchanged; last look of winning episodes unchanged (50.7 vs 51.2 decisions, pi05 L10-50). Early-stall conditional test (`pilot_early_stall.txt`; pi05 L10-50, pi05 Sp-50, groot Sp-50, host-matched) — failure excess over the init's deployed rate grouped by the old rule's first-flag time T_old:

| T_old bucket | H2/M2 failure excess | deployed failure excess |
|---|---|---|
| ≤ 10 decisions | −4.0 ± 3.0 pp (n 145) | −0.2 ± 1.3 |
| 12–30 decisions | +2.6 ± 2.4 pp | +1.2 ± 1.1 |
| later or never | −3.6 ± 1.3 pp | −1.2 ± 0.5 |

The 12–30 excess is driven by groot Sp-50 M2 (+.23), and M1 shows the same there (+.31): not guard-specific. Pooled paired ΔSR (knob-off cells, equal weights): H2 +0.8 ± 1.6 pp, M2 −0.7 ± 1.8, M1 +1.5 ± 1.5.

**Honest SR-cost estimate at 500 episodes**: guard swap ≈ −0.5 pp on L10-50 cells, 0 on pi05 Sp-50, +0.3 on groot Sp-50; 80% ≈ ±1.5 pp each; pooled −0.2 [−1.4, +1.0]. A single 500-episode cell sees this within ±2–3 pp; test A cannot confirm or exclude a cost below ~3 pp per cell (~1.3 pp pooled). Matched-SR: H2's call saving ≈ .009 on pi05 L10-50 buys only ≈ 0.3 pp at the knob slope of 29 pp per unit IR; a real 0.5 pp cost would make H2 alone a wash there. The vision saving (M1/M2) does not depend on this.

## 4. Changes before test A

**Methods: no change.** M1, H2, M2 stay frozen. H2's timing shift has no dev-justified fix that would not be fitted to this pilot.

**Experimental design, fix before test A:**
1. Run every arm of a cell on the same host type, H100 preferably (matches R11 test A). For GR00T the host effect (~5 pp SR, ~.015 IR on Sp-50) exceeds any method effect.
2. groot L10-50 H2/M2 screen has no same-host baseline: run `r12_guard_groot_l10_50_off` on H100 (100 eps) before trusting it, or treat those test-A arms as unscreened.
3. Readout: replace "first guard call in failing episodes" with "first call of any kind, paired by init over all episodes", plus calls per failing / winning episode.

**Owner decision needed**: the new guard formally fails the timing rule on pi05 L10-50, marginally on pi05 Sp-50 and groot L10-50; consequence checks clean. Recommendation: run H2/M2 as planned, labelled "screen override". Strict alternative: drop H2/M2 from pi05 L10-50, keep groot Sp-50 and periodic (pass). M1 unaffected.

## 5. TEST_A_PREDICTION (frozen 2026-10-03 20:25 CDT)

Before any R12 test-A result. A pool, 500 eps/arm, all arms of a cell on one host. Reference off levels (GR00T absolute numbers assume H100): R11 test A knob off pi05 L10-50 .844 @ .167; pi05 Sp-50 .922 @ .133; groot Sp-50 .892 @ .128; groot L10-50 .748 @ .209; periodic .32 .908 @ .306; R10 test A pi05 Sp-200 .994 @ .0985. Δ vs same-batch off arm; 80% intervals incl. run noise; Δ host-independent if hosts matched. If GR00T arms are served locally: groot Sp-50 SR −4 pp, IR +.014, calls +0.5; groot L10-50 IR +.005 ± .008, SR −2 ± 3 pp.

### pi05 L10-50
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .840 [.815, .865] | .167 [.162, .172] | 0 | 6.3 [5.8, 6.9] | — | — |
| M1 | .840 [.805, .875] | .157 [.151, .163] | .235 [.20, .27] | 6.2 [5.6, 6.8] | −.010 [−.014, −.006] | 0.0 [−2.8, +2.8] |
| H2 | .835 [.80, .87] | .158 [.151, .165] | 0 | 5.65 [5.0, 6.3] | −.009 [−.015, −.003] | −0.5 [−3.3, +2.3] |
| M2 | .834 [.80, .87] | .145 [.138, .152] | .35 [.31, .39] | 5.7 [5.0, 6.4] | −.022 [−.029, −.015] | −0.6 [−3.4, +2.2] |

### pi05 Sp-50
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .920 [.900, .940] | .133 [.129, .138] | 0 | 1.50 [1.30, 1.72] | — | — |
| M1 | .920 [.895, .945] | .122 [.117, .127] | .27 [.23, .31] | 1.47 [1.25, 1.70] | −.011 [−.015, −.007] | 0.0 [−1.9, +1.9] |
| H2 | .920 [.895, .945] | .126 [.120, .131] | 0 | 1.31 [1.12, 1.52] | −.007 [−.012, −.003] | 0.0 [−2.0, +2.0] |
| M2 | .919 [.892, .944] | .111 [.105, .117] | .38 [.33, .42] | 1.30 [1.10, 1.52] | −.022 [−.028, −.016] | −0.1 [−2.1, +1.9] |

### pi05 Sp-200 (H2/M2 only if their screen passes)
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .990 [.980, .997] | .0985 [.095, .102] | 0 | .51 [.42, .62] | — | — |
| M1 | .988 [.975, .997] | .084 [.080, .088] | .41 [.36, .46] | .53 [.42, .66] | −.015 [−.019, −.011] | −0.2 [−1.3, +0.8] |
| H2 | .990 [.978, .997] | .0955 [.091, .100] | 0 | .43 [.33, .55] | −.003 [−.008, +.002] | 0.0 [−1.1, +1.0] |
| M2 | .989 [.976, .997] | .077 [.071, .083] | .50 [.42, .58] | .43 [.33, .55] | −.022 [−.028, −.015] | −0.1 [−1.2, +0.9] |

### groot Sp-50 (H100)
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .890 [.870, .910] | .128 [.124, .133] | 0 | 1.48 [1.30, 1.68] | — | — |
| M1 | .887 [.86, .91] | .119 [.113, .126] | .27 [.23, .31] | 1.52 [1.28, 1.80] | −.009 [−.015, −.003] | −0.3 [−2.6, +2.0] |
| H2 | .893 [.87, .915] | .120 [.115, .126] | 0 | 1.27 [1.08, 1.48] | −.008 [−.012, −.004] | +0.3 [−1.9, +2.5] |
| M2 | .892 [.865, .915] | .107 [.101, .113] | .36 [.31, .40] | 1.28 [1.06, 1.50] | −.021 [−.027, −.015] | +0.2 [−2.1, +2.5] |

### groot L10-50 (H100)
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .748 [.72, .775] | .209 [.203, .215] | 0 | 10.3 [9.7, 10.9] | — | — |
| M1 | .748 [.71, .785] | .202 [.195, .209] | .125 [.10, .15] | 10.0 [9.2, 10.9] | −.007 [−.012, −.002] | 0.0 [−3.0, +3.0] |
| H2 | .743 [.705, .78] | .204 [.196, .212] | 0 | 9.8 [8.9, 10.7] | −.005 [−.012, +.002] | −0.5 [−3.6, +2.6] |
| M2 | .742 [.705, .78] | .196 [.187, .205] | .225 [.19, .26] | 9.8 [8.9, 10.8] | −.013 [−.021, −.006] | −0.6 [−3.7, +2.5] |

### pi05 L10-50, periodic knob .32
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .905 [.885, .925] | .306 [.301, .311] | 0 | 15.0 [14.5, 15.5] | — | — |
| H2 | .905 [.88, .93] | .303 [.298, .308] | 0 | 14.9 [14.3, 15.5] | −.003 [−.007, +.001] | 0.0 [−2.2, +2.2] |
| M2 | .904 [.88, .93] | .294 [.289, .299] | .225 [.19, .26] | 14.8 [14.2, 15.4] | −.012 [−.016, −.008] | −0.1 [−2.3, +2.1] |

### Pooled over the four knob-off 50-episode cells (equal weights, paired)
| arm | ΔIR | ΔSR (pp) |
|---|---|---|
| M1 | −.009 [−.012, −.006] | −0.1 [−1.4, +1.2] |
| H2 | −.007 [−.011, −.003] | −0.2 [−1.6, +1.2] |
| M2 | −.020 [−.024, −.015] | −0.3 [−1.7, +1.1] |

**Matched-SR ΔIR, central values** (R11 knob slopes 29 pp per unit IR pi05 L10-50, 43 pi05 Sp-50, 44 groot L10-50):

| arm | pi05 L10-50 | pi05 Sp-50 | groot L10-50 |
|---|---|---|---|
| M1 | −.010 | −.011 | −.007 |
| H2 | +.008 | −.007 | +.006 |
| M2 | −.001 | −.020 | +.001 |

80% intervals ≈ ±.10 (L10) and ±.05 (Sp-50); not determinable on flat-slope cells (groot Sp-50, pi05 Sp-200). **Test A can establish the raw IR saving with SR non-inferiority, not a matched-SR saving.**

**Mechanism checks (80%)**: H2/M2 first call of any kind, paired by init over all episodes, vs off: +8..+14 decisions (pi05 L10-50), +1..+4 (both Sp-50 cells, pi05 Sp-200), +2..+8 (groot L10-50), 0 ± 0.5 (periodic). H2/M2 calls per failing episode within ±10% of off; per winning episode −15% to −45%. M1 first call within ±1.5 decisions of off; M1 calls per episode type within ±10%. ≤ 1% of failing episodes without any call, every arm. One-camera share within ±.05 of the centers above.

**Expected outcome**: vision savings as predicted (M1 ≈ −.010, M2 ≈ −.021), SR flat within noise for all arms, no call-starvation signature.
