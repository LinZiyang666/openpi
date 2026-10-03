# R11 layer 4: opus post-sweep analysis — FINAL (94 / 94 test arms)

(Saved by the coordinator from opus's hand-back; the harness blocks report files from subagents.)

**Regenerate every number (read-only on all roots; about 30 s cold, 13 s cached):**

```bash
cd /home/weiland/projects/openpi && taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -W ignore -m exp.offline_search.rounds.r11.opus.analysis.run_analysis
```

- **Full tables:** `analysis/out/TABLES.md` (frozen copy `TABLES_FINAL.md`), machine-readable `results.json`, `arms.csv`. Section numbers in brackets below refer to TABLES.md.
- **Arms per method:**

  | method | arms |
  |---|---|
  | knob off | 8 |
  | random | 17 |
  | periodic | 17 |
  | periodic + post-guard tail | 4 |
  | random two-chunk | 2 |
  | distance | 17 |
  | error hybrid | 17 |
  | adaptive error hybrid | 8 |
  | disagreement | 4 |

- **Roots:**
  - `r11_knob_1` (23 arms) and `r11_knob_2` (24 arms), both H100;
  - `r11_local_k3` (23 arms) and `r11_local_k4` (19 arms), both RTX 4090;
  - `r11_knob_4` (5 arms, H100). It holds the rest of k4's list; k4's aborted partial copy of `pi05_l10_200_random_ir25` is ignored.
- **References:**
  - same-batch R11 knob-off arms for all 8 cell-sizes;
  - R10 GC_dist (same configuration, earlier batch);
  - the 4090 duplicate `r11_local_idg`;
  - R8 pure policy.

---

## 给 owner 的说明（白话）

1. **花费能不能按目标落地。**
   - **随机补调**（每个决策点按固定概率额外调用一次策略）和**定期补调**（距上次调用策略满固定间隔就调用一次，守卫调用也算一次）都能落到目标：
     - 定期 17 个臂全部在目标 ±0.02 以内；
     - 随机 17 个里 16 个在内，唯一超出的差 0.0202。
   - 平均都略低约 0.006–0.008。
   - 误差几乎全部来自一件事：闭环中"无进展守卫"的触发频率与在示范库上离线回放估出的不同。π0.5 实际触发更少，所以花费略低。只要知道闭环下守卫的真实触发率，账本模型的误差只有约 0.0035。
   - **按状态触发的方法系统性地花不满预算**：
     - 距离阈值 17 个臂只有 2 个落在 ±0.02 以内，最差少花 0.19；
     - 误差预测混合 17 个里 6 个落在范围内，平均少花 0.03；
     - 近邻分歧 4 个里 1 个落在范围内；
     - 在线自适应版 8 个里 5 个落在范围内。
   - 原因是闭环中机器人状态比校准时更"贴近"示范库，打分整体偏低，越过离线阈值的比例只有设计值的三分之一到三分之二。
2. **同样的花费换来多少成功率。**
   - 在实际花费相同的点上比较，定期补调比随机补调平均高 **1.5 个百分点**（17 个臂，95% 区间 +0.5 到 +2.3）。
   - 同目标直接配对比较也是 +1.5（8500 对回合，p = 0.0005）。我赛前预测 +0.6，方向对，落在我给的预测区间内。
   - 五类格子里四类都是定期更好，只有 GR00T 空间任务 50 集略差（−0.6）。
   - 误差预测混合换算效率和定期差不多（比定期 +0.4，不显著），但它花不到目标，没法当"旋钮"用。
   - 两个变体都不如定期本身：守卫后再补一次 −1.2；随机触发连调两段 −3.0。
3. **和"拿一部分回合直接全程跑策略"比。**
   - 随机挑一部分回合全程跑策略、其余回合只用缓存，花费和成功率是一条直线。
   - 随机补调就在这条线上（−0.3），等于没有比整回合花钱更聪明。
   - 定期补调在线上方 +1.2（显著），误差预测混合 +1.5。
4. **钱花在哪里有用。**
   - 收益几乎全部来自"缓存本来会失败"的初始状态。在另一次关闭补调的运行里失败过的初始状态上：
     - 随机补调 +28 个百分点；
     - 定期补调 +34；
     - 全程策略 +47。
   - 在缓存本来就能成功的初始状态上，各种补调和全程策略一样基本不变（−1 到 −2，属正常波动）。
   - 定期补调的结构优点：
     - 守卫刚调用完的下一个决策点几乎不再补（11% 对 38%）；
     - 失败回合里补得更少；
     - 纯缓存连续执行最长约 30 个控制步。随机补调偶尔长达 210 个控制步。
5. **硬件。**
   - 同一批次、同一配置，GR00T 长任务 50 集在 H100 和 4090 上成功率 0.748 对 0.750，花费完全一样。π0.5 长任务 50 集 0.844（4090）对 0.826（H100），p = 0.29。
   - 结论：没有系统差异，统计结果可以合并。
   - 但 4090 的视觉编码数值与 H100 略有不同：GR00T 第一步就有约 6% 的回合检索到不同的示范，所以单个回合不能跨卡逐一复现。
   - π0.5 即使在同一张 H100 上重跑，也有 35–41% 的回合在第一次调用策略前就分叉。
   - 最大的差异反而出现在同一张卡的两个批次之间：GR00T 长任务 500 集关闭补调 0.858 对 0.906（p = 0.004），说明单次对照有噪声。我用多次运行平均做了敏感性检查，结论不变。
6. **建议。**
   - 第四层采用**定期补调**。它花费准、同花费下成功率更高，也是唯一稳定高于"整回合混跑"直线的可控方法。随机补调保留作花费校准的参照。两个变体不采用。
   - 按状态触发的方法不作旋钮，除非改成按实际花费在线调节。
   - **示范多、已接近纯策略的格子默认关闭**。建议的纯示范库判据（待验证的假设）是：离线回放时缓存给出的动作和策略自己的动作平均差距小于 0.40。12 个格子里 11 个与实测"离纯策略不到 2 个百分点"一致，包括：
     - π0.5 长任务 200 和 500 集；
     - π0.5 空间任务 200 和 500 集；
     - GR00T 长任务 500 集和空间任务 500 集。
   - 唯一不一致的是 GR00T 空间任务 200 集：判据说"开"，但它其实已经和纯策略持平，开了只是白花钱，不会损失成功率。
   - 这些格子里补调最多只能再挣 2 个百分点左右，实测也确实如此。
   - 下一轮值得试（假设，尚未验证）：
     - 守卫在一个回合里反复触发时加密定期补调，因为收益集中在难的初始状态；
     - 用全库示范估守卫触发率，修正 π0.5 的系统性少花。

---

## Technical section

**Symbols and method labels:**
- v = looks per decision slot.
- g = guard-call share of looks.
- k = knob-call share of non-guard ("eligible") looks.
- f = g + (1 − g)·k + o.
- IR = v·(c_v + c_m·f).
- R / P / P+tail / R2 = opus's random / periodic / periodic + post-guard tail / random two-chunk.
- D / E / AE / Dis = astra's distance / error hybrid / adaptive error hybrid / disagreement.

**Method.** Every comparison is paired on the 500 test-A (task, init) episodes.
- Uncertainty comes from a stratified-by-task episode bootstrap (B = 2000), shared by all arms of a cell-size so contrasts stay paired.
- Discordant-pair tests are exact McNemar.

### 0. Integrity [§0]

All 94 test arms plus the 9 off-like references pass:
- **Ledger and cadence:**
  - decision logs reproduce the ledger's N/V/M exactly;
  - every call is followed by exactly one policy tail and a fresh look (0 cadence violations);
  - no knob call sits on a guard anchor;
  - 0 "other" calls: every miss is reason 4 (the guard) or the arm's own knob reason.
- **Opus methods:** every eligible anchor was re-derived from history.
  - Keyed coins match bit-for-bit.
  - Periodic run/cap match.
  - The call decision equals the SPEC rule in all eligible anchors (0 mismatches).
- **astra's static methods:** the logged probability equals (1 − β)·q + β·1[score > threshold] everywhere.
- **Random:** realized k equals ρ within binomial error over 17 arms (|z| ≤ 2.07) [§1d].

### 1. IR calibration (Q1) [§1, §1b, §1c, §3h]

| method | arms | mean IR − target | max abs | within ±.02 | Δ base guard | Δ guard feedback | Δ knob share | mean abs(IR − informed) | within ±.01 of informed |
|---|---|---|---|---|---|---|---|---|---|
| R | 17 | −.0078 | .0202 | 16/17 | −.0047 | −.0020 | −.0011 | .0035 | 14/17 |
| P | 17 | −.0058 | .0160 | 17/17 | −.0046 | −.0042 | +.0029 | .0035 | 16/17 |
| P+tail | 4 | −.0121 | .0227 | 3/4 | −.0044 | −.0048 | −.0030 | .0079 | 3/4 |
| R2 | 2 | −.0085 | .0162 | 2/2 | −.0030 | −.0017 | −.0037 | .0055 | 2/2 |
| D | 17 | −.0802 | .1916 | 2/17 | −.0078 | +.0006 | **−.0729** | .0743 | 2/17 |
| E | 17 | −.0276 | .0475 | 6/17 | −.0068 | −.0037 | −.0174 | .0214 | 3/17 |
| AE | 8 | −.0141 | .0235 | 5/8 | −.0035 | −.0034 | −.0073 | .0108 | 4/8 |
| Dis | 4 | −.0317 | .0437 | 1/4 | −.0071 | −.0037 | −.0213 | .0255 | 0/4 |

How the table is built:
- The decomposition is an exact Shapley attribution of (realized − library prediction) over five factors: v, the base guard gap (closed-loop knob-off g − library g), guard feedback (arm g − knob-off g), k, and other calls. Residuals are < 1e-15.
- "Informed" = the frozen knob share applied with the measured knob-off g.
- v contributes < .002 everywhere (closed loop .503–.513, the same as the library).

**Base guard gap** [§1c]. Same-batch knob-off g vs library-replay g:

| cell-size | closed-loop g | library g |
|---|---|---|
| π0.5 L10-50 | .212 | .259 |
| π0.5 L10-200 | .170 | .189 |
| π0.5 L10-500 | .168 | .204 |
| π0.5 Sp-50 | .128 | .151 |
| GR00T L10-50 | .314 | .291 |
| GR00T L10-200 | .268 | .271 |
| GR00T L10-500 | .250 | .274 |
| GR00T Sp-50 | .122 | .150 |

- The library over-estimates g on 7 of 8 cell-sizes.
- With ∂IR/∂g = v·c_m·(1 − k), this pulls π0.5 R/P arms down by .006–.016, most at low targets.
- On π0.5 Sp-50, guard feedback dominates instead: knob calls end stalls, and g drops from .128 to .07–.10, so the realized-minus-informed gap is up to −.019 (P+tail).
- GR00T L10 arms sit within −.012 to +.010.

**State knobs: the shortfall is k itself** [§3h]. Closed-loop share of eligible looks above the frozen threshold vs the calibrated dose:

| method | share above threshold | calibrated dose | ratio |
|---|---|---|---|
| D | .06–.36 | .20–.72 | ≈ 0.2–0.7× |
| E | .11–.52 | .14–.72 | — |
| Dis | .33–.41 | .36–.52 | — |

- Closed-loop distance medians sit below the thresholds:

  | cell-size | closed-loop median distance | thresholds |
  |---|---|---|
  | π0.5 L10-50 | 9.0–9.2 | 10.1–15.7 |
  | GR00T Sp-50 | 12.4–12.5 | 15.5–19.3 |
  | π0.5 L10-500 | 6.2 | 7.7 |

- Closed-loop states follow the cached demos; calibration used whole-episode-out donors from 4/5 of the library.
- AE's online dose rises (mean online q .41–.78 vs q0 .16–.65) and recovers most of the shift.

**Answer to Q1.**
- Reliable knobs: random and periodic. Periodic is slightly tighter: max error .016, and 16/17 within ±.01 of informed.
- Adaptive error hybrid is borderline.
- Distance, disagreement and the static error hybrid are not reliable IR knobs on this base.

### 2. Efficiency frontier (Q2) [Overview, §2–§2f]

**Per-cell frontier.** SR @ realized IR; Δ vs same-batch knob-off in pp. The full grid, with every method and its paired b/c, is in TABLES §Overview / §2.

| cell-size | knob-off | pure | arms |
|---|---|---|---|
| GR00T L10-50 | .748 @ .209 | .898 | R .754 / .792 / .818 (+0.6 / +4.4 / +7.0); **P .774 / .826 / .844 (+2.6 / +7.8 / +9.6)**; AE.40 .852 @ .395 (+10.4); E.40 .826 @ .376 |
| π0.5 L10-50 | .844 @ .167 | .908 | R .860 / .860 / .900; **P .870 / .908 @ .306 / .900**; E .888 @ .218 / .896 @ .282 / .902 @ .369 |
| π0.5 Sp-50 | .922 @ .133 | .988 | R .950 / .960 / .968; **P .972 / .980 / .984**; P+tail.32 .990 @ .297 |
| GR00T Sp-50 | .892 @ .128 | .940 | R .894 / .922 / .900; P .876 / .912 / .910; AE .926 / .930; D .862–.868 (−2.4 to −3.0) |
| π0.5 L10-200 | .890 @ .149 (R10 .904) | .908 | R .904, P .916, E .920, D .910 (all @ ≈ .20–.24) |
| GR00T L10-200 | .820 @ .190 (R10 .820) | .898 | R .848 / .836; **P .836 / .884 @ .313**; E .864 / .858 |
| π0.5 L10-500 | .906 @ .148 (R10 .894) | .908 | R .912, P .904, E .924, D .896 |
| GR00T L10-500 | .858 @ .182 (R10 .906) | .898 | R .866, P .900, E .896, D .892 |

**SR advantage at the same realized IR** [§2c]. Arm SR minus the reference method's piecewise-linear curve, which includes the knob-off point; extrapolation ≤ .01 IR. Two references, random's curve and periodic's curve:

| method | arms | vs random's curve, pp [95% CI] | vs periodic's curve, pp [95% CI] |
|---|---|---|---|
| P | 17 | **+1.46 [+0.54, +2.32]** (p .002) | — |
| R | 17 | — | −1.47 [−2.31, −0.52] |
| E | 17 | +1.69 [+0.84, +2.46] | +0.39 [−0.49, +1.15] |
| AE | 8 | +1.30 [+0.07, +2.49] | −0.36 [−1.47, +0.71] |
| Dis | 4 | +1.83 [+0.18, +3.34] | −0.02 [−1.62, +1.48] |
| P+tail | 4 | +1.14 [−0.45, +2.80] | −0.97 [−2.52, +0.62] |
| R2 | 2 | +1.19 [−1.65, +3.97] | −2.82 [−5.81, −0.07] |
| D | 17 | +0.11 [−0.88, +1.10] | −0.87 [−1.98, +0.16] |

**Matched-target P − R** [§2f, §3i]: +1.48 pp, +702/−576, p .0005 over 17 pairs; IR-adjusted +1.48. By family:

| family | P − R (pp) | +b / −c |
|---|---|---|
| π0.5 L10-50 | +1.9 | +124/−95 |
| GR00T L10-50 | +2.7 | +202/−162 |
| π0.5 Sp-50 | +1.9 | +53/−24 |
| GR00T Sp-50 | −0.6 | +115/−124 |
| 200/500 cells | +1.5 | +208/−171 |

- **Opus prediction P4** was +0.6 pp (80% PI −0.9 to +2.1). The realized +1.48 is inside the interval and above the point estimate.
- Other matched-target contrasts:

  | contrast | pp | p | note |
  |---|---|---|---|
  | E − R | +1.38 | .001 | E realizes .020 less IR |
  | D − R | −1.19 | .008 | IR-adjusted +0.11 |
  | AE − R | +1.20 | .05 | |
  | P+tail − P | −1.20 | — | 4 pairs |
  | R2 − P | −3.00 | .04 | |

**Pooled efficiency over knob-off** [§2d], in SR pp per +0.1 IR:
- E 3.40, Dis 3.37, R2 3.31;
- P 2.93 [2.06, 3.85], P+tail 2.89, AE 2.71;
- pure policy (50-cells) 2.35;
- D 2.15, R 1.88 [1.04, 2.74].

These slopes mix cell compositions; the matched-IR advantage above is the fair comparison.

**Chord benchmark** [§2e]. Running knob-off on a share (1 − λ) of episodes and pure policy on the rest traces a line between the two. Advantage over that line at the same IR:

| method | pp [95% CI] |
|---|---|
| R | −0.31 [−1.21, +0.63] |
| **P** | **+1.17 [+0.26, +2.12]** (p .02) |
| E | +1.47 [+0.57, +2.36] |
| Dis | +1.42 |
| P+tail | +0.94 |
| AE | +0.75 |
| D | −0.05 |
| R2 | −0.01 |

- **Sensitivity [§2d′/2e′].** The reference is changed to the per-episode average of all 2–3 knob-off runs of each cell, which damps single-run batch noise such as GR00T L10-500 (.858 vs .906). Every conclusion holds: chord P +1.15 [+0.29, +2.07], E +1.46, R −0.32; efficiency P 2.96, R 1.91.

**Answer to Q2.**
- Periodic converts IR into SR most efficiently among the reliable knobs: +1.5 pp over random at matched IR, the only steerable method clearly above the chord.
- Error hybrid converts equally well (+0.4 vs P, n.s.) but is not steerable.
- Random is no better than whole-episode mixing.
- "Periodic beats random" is confirmed.

### 3. Where the extra calls landed (Q3) [§3a–§3j]

- **Progress and time** [§3a–§3c].
  - R and P spread calls flatly over retrieved-demo progress and normalized time (relative intensity .93–1.05).
  - Guard calls are concentrated late: in knob-off arms, ×0.34 in the first fifth vs ×1.35 in the last.
  - On LIBERO-10 at decision steps ≥ 60 (failure stretches), P's knob intensity drops to ×0.65–0.82 because guard bursts reset its gap; R stays ×1.0.
  - D is ×1.7 at low progress, ×0.6 mid-episode and ×2–3.7 at steps ≥ 60.
  - E, AE and Dis lean late: ×0.5–0.8 early, ×1.2–1.3 late.
- **Around guard calls** [§3d, §3e].
  - P's call rate at the anchor right after a guard call is .11, against .38 overall: a built-in refractory period. R is flat at .36.
  - The next-anchor guard probability is lower after a P knob call than after a look (.163 vs .194). For R it is unchanged (.184 vs .190).
  - This is the mechanism behind the guard-feedback term.
- **Failed vs successful episodes** [§3f, §3i].
  - Failed episodes are long (≈ 89–104 slots vs 35–51) and guard-heavy (g .46–.49 vs .13–.17).
  - P spends less per eligible look in failed episodes than R (.307 vs .353).
  - P puts a smaller share of its knob calls into eventually failed episodes (.118 vs .157); on matched pairs P is lower in 14/17.
  - P also has fewer failed episodes (.106 vs .121).
- **Cache-only stretches** [§3g].
  - P caps the longest stretch at ceil(K) anchors (1–3 anchors = 10–30 controls) in every arm.
  - R's mean longest stretch is 1.7–6.9 anchors, p90 3–11, longest 21 anchors (210 controls).
  - D leaves 6–11-anchor stretches, close to knob-off (9–12.7).
  - Closed-loop stretch statistics match opus's library replay within ≈ 0.5 anchors. Examples: π0.5 L10-50 R @ .25, 6.86 vs 6.33; GR00T L10-500 R @ .25, 5.82 vs 5.78.
- **Where calls pay off** [§3j]. Difficulty is measured in an *independent* knob-off run of the same init, so chance failures of the reference do not leak into the split.

  | split | R | P | E | AE | pure policy |
  |---|---|---|---|---|---|
  | inits that run failed | +27.9 pp | +33.9 pp | +31.3 pp | +40.3 pp | +47.3 pp |
  | high early-guard third | +9.2 pp | +13.5 pp | — | — | +16.0 pp |
  | low early-guard third | +0.6 pp | +0.8 pp | — | — | +2.3 pp |
  | inits it solved | −1.8 pp | −1.2 pp | −0.8 pp | — | −0.8 pp |

  - On solved inits D loses −2.4, comparable to pure policy's −0.8 (regression to the mean), so it is not knob-specific harm.
  - Extra calls pay off almost only where the cache would fail. Periodic captures more of pure policy's gain there than random.
- **State-knob score distribution** [§3h]: see §1.

**Answer to Q3.**
- Random lands uniformly, including right after guard calls and into hopeless failure stretches, and leaves long blind stretches.
- Periodic withholds calls after guard bursts and in failure stretches, and keeps blind stretches ≤ 30 controls. Its matched-IR gain is concentrated on hard inits.

### 4. Hardware check (Q4) [§4]

| pair | type | SR A / B | +b/−c | p | IR A / B | g A / B | step-0 top-1 equal | diverge before 1st policy call |
|---|---|---|---|---|---|---|---|---|
| GR00T L10-50: R11 H100 off vs R11 4090 duplicate (**same batch**) | cross | .748 / .750 | +50/−51 | 1.0 | .209 / .209 | .314 / .313 | .936 | .586 |
| GR00T L10-50: 4090 duplicate vs R10 H100 | cross | .750 / .754 | +54/−56 | .92 | .209 / .207 | .313 / .310 | .936 | .586 |
| π0.5 L10-50: R11 4090 off vs R10 H100 | cross | .844 / .826 | +33/−24 | .29 | .167 / .168 | .212 / .214 | .990 | .420 |
| GR00T L10-50: R11 H100 vs R10 H100 | same | .748 / .754 | +42/−45 | .83 | .209 / .207 | .314 / .310 | 1.000 | .004 |
| GR00T L10-200 | same | .820 / .820 | +33/−33 | 1 | .190 / .191 | .268 / .272 | 1.000 | .000 |
| GR00T L10-500 | same | **.858 / .906** | +21/−45 | **.004** | .182 / .175 | .250 / .235 | 1.000 | .024 |
| GR00T Sp-50 | same | .892 / .896 | +11/−13 | .84 | .128 / .127 | .122 / .119 | .998 | .112 |
| π0.5 Sp-50 | same | .922 / .910 | +19/−13 | .38 | .133 / .135 | .128 / .133 | 1.000 | .350 |
| π0.5 L10-200 | same | .890 / .904 | +20/−27 | .38 | .149 / .150 | .170 / .172 | .988 | .414 |
| π0.5 L10-500 | same | .906 / .894 | +28/−22 | .48 | .148 / .150 | .168 / .173 | .988 | .394 |

- **No systematic SR, IR or guard-rate difference.**
  - The same-batch cross-GPU pair is identical in SR (+50/−51) and IR.
  - Cross-hw mean |ΔSR| is 0.8 pp vs 1.4 pp same-hw.
  - Knob arms show the same IR error on both GPUs: IR − informed −.023 (4090, 41 arms) vs −.022 (H100, 45 arms).
- **GR00T trajectories carry a GPU signature.**
  - On the 4090, 6% of step-0 retrievals flip, and 59% of episodes diverge before the first policy call.
  - Same-hardware GR00T reruns: 0–2% (L10) and 11% (Sp-50).
  - The cause is stage-1 vision-encoder numerics flipping near-tie retrievals.
- **π0.5 has no visible extra signature.** Its pre-call path is already non-reproducible between H100 runs (35–41%), and the cross-GPU pair is at the same level (42%).
- Discordance on GR00T L10-50 is .17 same-hw vs .20–.22 cross-hw.
- **The one significant difference is a same-GPU batch effect.** GR00T L10-500 knob-off was .858 (R11) vs .906 (R10), with an identical cache prefix (2% pre-call divergence). It is policy-noise variance across batches: p .004, or ≈ .04 after a 10-pair Bonferroni.

**Answer to Q4.** 4090 and H100 results are poolable for SR and IR. Single episodes are not reproducible across GPUs (GR00T), nor across processes (π0.5). Single knob-off references carry ±2–5 pp batch noise, which §2d′/2e′ handles by pooling knob-off runs.

### 5. Recommendation for layer 4 (Q5)

1. **Adopt the guard-aware periodic knob** (SPEC §2: a dithered gap cap K per cell-size, guard calls reset the gap).
   - It is IR-reliable: 17/17 within ±.02, mean −.006, ±.0035 given the closed-loop guard rate.
   - It converts IR into SR best among steerable knobs: +1.5 pp over random at matched IR, and +1.2 pp above the whole-episode mixing chord.
   - Keep random as the calibration and reference knob.
   - Do not adopt the post-guard tail or two-chunk variants.
   - Error hybrid matches P's conversion, but only after its IR shortfall. It would need online spend tracking to be a knob, and AE (the tracked version) converts −0.4 vs P.
2. **Knob → IR map from the library alone.**
   - The current whole-episode-out replay with the guard-first overlap rule is adequate: expected realized IR = target − .006 on average, worst −.016 for P.
   - The remaining bias is the library over-estimating g, by .02–.047 on π0.5, plus guard feedback on short tasks.
   - Hypotheses for a future round (library-only fitting, not tested here):
     - (a) estimate g with full-library donors, removing the 4/5-donor sparsity bias;
     - (b) model guard feedback in the replay by thinning guard flags after knob calls by the open-loop "next-anchor stall" rate;
     - (c) a per-episode spend tracker on top of P that charges guard calls against the budget (opus's SigmaDelta fallback), which makes IR insensitive to g;
     - (d) for any state knob, a random floor or online tracking is mandatory, since closed-loop score distributions sit below whole-episode-out calibration.
3. **Where to spend (hypothesis).** Gains concentrate on inits the cache fails, and early guard activity marks them (+13.5 pp for P on the high-early-guard third).
   - A guard-escalating periodic schedule would test this: shorten the gap after repeated guard fires within an episode. It is calibratable on the library guard replay.
   - The post-guard tail, a crude version of it, did not help (−1.2 pp vs P). Any escalation should therefore be gradual, not an immediate re-call.
4. **Default-off on near-pure cells** [§6]. Proposed library-only switch: knob off when the mean whole-episode-out error between the cache's served (corrected) chunk and the policy's own chunk is < .40.
   - It agrees with "measured knob-off-to-pure gap ≤ 2 pp" in **11/12** cell-sizes. Off: π0.5 L10-200/500, π0.5 Sp-200/500, GR00T L10-500, GR00T Sp-500. On: all 50-cells and GR00T L10-200.
   - The miss is GR00T Sp-200 (error .425, already 1.4 pp above pure): the rule leaves it on, wasting IR without SR risk.
   - On the near-pure cells with knob arms, P @ .25 adds +1.9 / +0.4 / +1.8 pp (π0.5 L10-200 / π0.5 L10-500 / GR00T L10-500) vs the pooled knob-off reference, for +.09 IR. That is at most the ≤ 2 pp remaining gap, so off by default with an owner opt-in.
   - The .40 threshold was read off test aggregates. It is a hypothesis to validate on non-test B-pool dev episodes, not a fitted result.

### 6. Opus pre-registered predictions vs measurement (PREDICTION.md, 2026-10-02 21:10 CDT) [§5]

28 of the 38 pre-registered arms are on the final grid; the Spatial targets moved.

| claim | prediction | measured | verdict |
|---|---|---|---|
| **P1** IR within ±.02 of library prediction | ≥ 34/38 | 28/28 | ✓ |
| P1 within .012 of informed prediction | ≥ 32/38 | 28/28 | ✓ |
| P1 mean signed error, π0.5 | ≈ −.008 | −.011 | ✓ |
| P1 mean signed error, GR00T L10-50 | ≈ +.005 | +.002 | ✓ |
| **P2** R/P within .01 IR | ≥ 14/16 | 12/12 | ✓ |
| P2 "P closer to its prediction" | ≥ 10/16 | 7/12 | marginal miss |
| **P3** SR non-decreasing in target per 50-cell | within noise | dips ≤ .022: GR00T Sp-50 R .922 → .900, P .912 → .910; π0.5 L10-50 P .908 → .900 | ✓ |
| P3 SR at .40 | π0.5 L10-50 ≈ .89; GR00T L10-50 ≈ .86; π0.5 Sp-50 ≈ .97 | .900 / .844 / .984 | ✓ |
| P3 no LIBERO-10-50 arm noninferior to pure | — | none | ✓ |
| **P4** P − R pooled | +0.6 pp | +1.48 pp | ✓ (inside the interval) |
| **P5** \|ΔSR\| < 1.5 pp on near-pure cells at .25 | — | 4/6 | partial (exceptions: P on π0.5 L10-200 +2.6 and GR00T L10-500 +4.2 vs a low knob-off draw) |
| **P6** P+tail − P | +0.3 | −1.2 | — (no direction claimed) |
| P6 R2 − R | −0.5 | +1.1 | — (no direction claimed) |
| SR change vs off: realized − predicted | — | −0.5 pp on average (sd 1.9) | random on GR00T L10-50 was over-predicted by 1.5–3.5 pp |

### 7. Scripts (all in `exp/offline_search/rounds/r11/opus/analysis/`)

| file | role |
|---|---|
| `run_analysis.py` | entry point: arm discovery (one complete copy per test arm, incl. `r11_knob_4`), references, stratified paired bootstrap, every table → `out/TABLES.md`, `out/results.json`, `out/arms.csv` |
| `extract.py` | read-only parsing of journals, cost ledgers and server decision logs into per-episode / per-anchor tables; schedule-rule, coin, cadence and ledger audits; per-arm cache (`cache/`) keyed on file size and mtime |
| `stats.py` | bootstrap, McNemar, curve interpolation (≤ .01 IR extrapolation), pooled reference, Shapley decomposition, run-length helpers |
| `out/TABLES_FINAL.md`, `out/results_final.json` | frozen final snapshot (the interim one is `TABLES_INTERIM.md`) |

Library-side inputs (all frozen before the sweep, nothing refitted):
- `../ir_model.py`, used for the cache-stretch replay and the §6 served-chunk error;
- `../out/{curves,predictions}.json`;
- sol's `rounds/r11/knob/calibration/*.json` settings.
