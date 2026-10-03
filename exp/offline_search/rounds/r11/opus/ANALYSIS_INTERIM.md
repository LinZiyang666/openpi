# R11 layer 4: opus post-sweep analysis — INTERIM (40 of 94 test arms)

(Saved by the coordinator from opus's hand-back; the harness blocks report files from subagents.)

**Reproduce / regenerate (single command, read-only on every run root, ~10 s with cache, ~1 min cold for all 94 arms):**

```bash
cd /home/weiland/projects/openpi && taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -W ignore -m exp.offline_search.rounds.r11.opus.analysis.run_analysis
```

The command picks up every arm that has a `summary.json` and ≥ 495 journal rows. It writes `analysis/out/TABLES.md`, `results.json` and `arms.csv`. When all 94 test arms are present, the table header switches from INTERIM to FINAL. The numbers below are frozen in `analysis/out/TABLES_INTERIM.md` (snapshot 2026-10-03 04:00 CDT).

**Status: INTERIM.** 40 of 94 test arms were complete at the snapshot. Arms per method: knob off 2, random 8, periodic 7, periodic + post-guard tail 2, random two-chunk 1, distance 4, error hybrid 8, adaptive error hybrid 5, disagreement 3.
- Five cell-sizes have no same-batch R11 knob-off arm yet: π0.5 Sp-50, GR00T Sp-50, π0.5 L10-200, π0.5 L10-500 and GR00T L10-500. For these the reference is R10's GC_dist arm, which has the same configuration and the same 500 inits but came from an earlier batch.
- The GR00T L10-50 reference is the 4090 duplicate `r11_local_idg`.
- Every number below can move once the remaining arms arrive. The SR contrasts most of all, because many cells have only one or two of their arms so far.

---

## 给 owner 的说明（白话，中期版）

本节基于已跑完的 40 个测试臂（共 94 个）。结论可能随剩余臂变化，其中成功率对比最可能变。

1. **花费能不能按目标落地。**
   - **随机补调**（每个决策点按固定概率额外调用一次策略）和**定期补调**（距上次调用策略满一定间隔就调用）落得很准：15 个臂全部落在目标 ±0.02 以内，平均只低 0.005。
   - 剩下的误差几乎全部来自一个原因：闭环里"无进展守卫"的触发频率，和在示范库里离线回放估出的频率不一样。π0.5 实际触发少，所以花费略低于目标；GR00T 长任务 50 集略高。
   - 只要知道闭环下守卫的真实触发率，账本模型的误差平均只有 0.003。
   - **按状态触发的方法系统性地花不满预算**：
     - 距离阈值平均少花 0.10，4 个臂全部超出 ±0.02；
     - 近邻分歧少花约 0.04，误差预测混合少花约 0.03；
     - 带在线自适应的那一版少花约 0.01。
   - 原因已经从决策日志里直接看到：闭环中机器人状态比校准时更"贴近"示范库，打分整体偏低。超过离线阈值的比例只有预想的三分之一到一半（例：距离法 14%，预想 48%）。
2. **同样的花费，哪种换来的成功率更多。**
   - 在"实际花费相同"的点上比较（沿每种方法自己的几个点插值），定期补调比随机补调平均高 **2.2 个百分点**（5 个臂，95% 区间 +0.4 到 +3.8）。
   - 同一目标直接配对比较，定期比随机高 2.1 个百分点（3 对，p = 0.09）。
   - 我赛前预测是高 0.6 个百分点：方向对，幅度更大，但样本还少。
   - 误差预测混合和近邻分歧也比随机高约 2 个百分点；距离阈值低约 1 个百分点，而且花不出钱。
3. **和"拿一部分回合直接全程跑策略"相比。** 这是个更朴素的花钱办法：随机挑一部分回合全程用策略，其余回合只用缓存。它的花费和成功率是一条直线。
   - 随机补调大致就在这条线上（−0.6 个百分点）。也就是说，把调用均匀撒在回合中间，并不比整回合地花更划算。
   - 定期补调高出约 1 个百分点（还不显著）。"定期 + 守卫触发后再补一次"高出约 2.5 个百分点（只有 2 个臂）。
4. **钱花在哪里有用。**
   - 收益几乎全部来自"缓存本来会失败"的初始状态。在另一次关闭补调的运行里失败过的初始状态上，随机补调提高约 20 个百分点，定期提高约 28 个，全程策略提高约 44 个。
   - 在缓存本来就能成功的初始状态上，补调反而略微有害（−1 到 −4 个百分点）。
   - 定期补调有两个结构特点：守卫刚调用完的下一个决策点不会再补；纯缓存连续执行最长不超过约 30 个控制步。随机补调的最长连续段平均有 2–7 个决策点（20–70 个控制步），最长达 170 个控制步。
5. **硬件。** 同一配置在 4090 和 H100 上，成功率和花费都没有系统差异：
   - π0.5 长任务 50 集：0.844 对 0.826，p = 0.29；
   - GR00T 长任务 50 集：0.750 对 0.754，p = 0.92；
   - 花费相差不到 0.002。

   但 4090 上视觉编码的数值和 H100 略有不同：有 40–60% 的回合在第一次调用策略之前就检索到了不同的示范。H100 两次运行之间这个比例是 0%。所以单个回合不能跨卡逐一复现，统计结果可以合并。
6. **初步建议。**
   - 第四层采用**定期补调**，它的花费和随机补调一样准，同花费下成功率更高。随机补调保留作花费校准的参照。
   - 按状态触发的方法目前不适合当"旋钮"，因为花费跟不上目标；除非改成在线按实际花费调节。
   - 下一轮值得试（假设，尚未验证）：
     - 守卫多次触发的回合里加密补调，因为收益集中在难的初始状态上；
     - 用全库示范来估守卫触发率，修正 π0.5 的系统性少花。
   - 示范多、已接近纯策略的格子是否默认关闭，要等这些格子的臂跑完再定。

---

## Technical section

Symbols:
- **v**: looks per decision slot.
- **g**: guard-call share of looks.
- **k**: knob-call share of non-guard ("eligible") looks.
- **f**: miss share of looks, `f = g + (1 − g)·k + o`.
- **IR** = `v·(c_v + c_m·f)` (owner definition).
- **R / P / P+tail / R2** are opus's random, periodic, periodic + post-guard tail and random two-chunk knobs.
- **D / E / AE / Dis** are astra's distance, error hybrid, adaptive error hybrid and disagreement knobs.

Full tables are in `analysis/out/TABLES_INTERIM.md`, with section numbers in brackets below.

### 0. Data and integrity [§0]

- **Sources:**
  - test roots `r11_knob_1` (H100) and `r11_local_k3`, `r11_local_k4` (4090); `r11_knob_2` had not started at the snapshot;
  - the 4090 duplicate `r11_local_idg`;
  - R10 GC_dist arms (`r10_corr3_*`, H100), used as the knob-off reference where R11's own is pending and as hardware replicates;
  - R8 `r8_*_P10` pure policy.
- **Pairing:** every comparison is paired on the 500 test-A (task, init) episodes. Uncertainty comes from a stratified-by-task episode bootstrap (B = 2000) shared by all arms of a cell-size, so contrasts stay paired. Discordant-pair tests are exact McNemar.
- **Audits.** All 50 parsed arms (40 test, 2 off-like references, 8 R10) pass:
  - the decision logs reproduce the cost ledger's N / V / M exactly;
  - every call is followed by exactly one policy tail and then a fresh look (0 cadence violations);
  - no knob call sits on a guard anchor;
  - "other" calls are 0. Every miss is reason 4, the guard, or the arm's own knob reason (61–68).
- **Opus methods:** every eligible anchor was re-derived from history.
  - Recomputed keyed coins match the logged ones bit-for-bit.
  - Periodic `run` / `cap` match.
  - The call decision equals the SPEC rule in all ≈ 110 k eligible anchors (0 mismatches).
- **astra static methods:** the logged probability equals `(1 − β)·q + β·1[score > threshold]` at every eligible anchor.
- **Random knob:** realized k equals ρ within binomial error, with |z| ≤ 1.7 over 8 arms [§1d].

### 1. IR calibration (Q1) [§1, §1b, §1c, §3h]

**Per-method summary** (realized minus target; the decomposition is exact Shapley over v, the base guard gap, guard feedback, k and other calls):

| method | arms | mean IR − target | max abs | within ±.02 | Δ base guard | Δ guard feedback | Δ knob share | mean abs(IR − informed) |
|---|---|---|---|---|---|---|---|---|
| R | 8 | −.0061 | .0189 | 8/8 | −.0035 | −.0018 | −.0009 | .0029 |
| P | 7 | −.0040 | .0120 | 7/7 | −.0009 | −.0047 | +.0015 | .0036 |
| P+tail | 2 | −.0205 | .0227 | 1/2 | −.0079 | −.0082 | −.0050 | .0133 |
| R2 | 1 | −.0162 | .0162 | 1/1 | −.0118 | −.0004 | −.0036 | .0042 |
| D | 4 | −.1043 | .1362 | 0/4 | −.0114 | +.0021 | **−.0946** | .0945 |
| E | 8 | −.0289 | .0441 | 3/8 | −.0064 | −.0034 | −.0196 | .0233 |
| AE | 5 | −.0122 | .0220 | 4/5 | −.0038 | −.0021 | −.0064 | .0087 |
| Dis | 3 | −.0374 | .0437 | 0/3 | −.0091 | −.0052 | −.0237 | .0294 |

The look rate v contributes < .002 everywhere: v = .503–.513, the same as in the library.

**Reading the decomposition:**
- **Reliable knobs.** R and P are reliable: 15/15 arms within ±.02.
  - The residual is the base guard gap. The library replay's g is higher than the closed-loop knob-off g on π0.5 L10-50 (.259 vs .212). On GR00T L10-50 it is lower (.291 vs .313); on GR00T L10-200 it is equal (.271 vs .268) [§1c].
  - The sensitivity is `∂IR/∂g = v·c_m·(1 − k)`, so the gap matters most at low targets. Example: π0.5 L10-50 R @ .25 realized .231.
  - "Informed" applies the frozen knob share to the measured closed-loop knob-off g. Its mean abs error is .003–.004 for R and P, which confirms IR_MODEL §4's claim that the accounting is accurate to ±.01 *given* the closed-loop guard rate. Guard feedback (knob calls slightly lower the arm's g) adds −.002 to −.005.
- **P+tail and R2** land 1.6–2.3 points low.
  - On π0.5 L10-50 (both variants) the cause is the base guard gap (−.012).
  - On π0.5 Sp-50 (P+tail) it is guard feedback (−.015): the post-guard call ends stalls, and g drops from .133 to .065.
- **State knobs: the shortfall is in k itself** (closed-loop score shift) [§3h].
  - At each static threshold, the share of closed-loop eligible looks above it is .09–.35 for D against a calibrated dose of .30–.70, and .11–.52 for E against .19–.72. Dis is at about .78× its dose.
  - Closed-loop query distances are ≈ 28% below the calibration median where the threshold sits at that median. Examples: π0.5 L10-50 D, closed-loop median 9.0 vs threshold 12.5; GR00T Sp-50 D, 12.5 vs 17.3.
  - This is the dev-pilot finding, now on test A.
  - The hybrids' random half bounds the miss. AE's online dose rises above its q0, with a mean online q of .41–.78 vs q0 .16–.65, and recovers most of it.

**Answer to Q1.**
- Random and periodic are reliable knobs: realized ≈ target within ±.02, with a mean bias of −.005, and that bias is predictable from the guard gap.
- Adaptive error hybrid is nearly reliable: 4/5 within ±.02, worst −.022.
- Distance, disagreement and the static error hybrid are not reliable knobs on this base.

### 2. Efficiency frontier (Q2) [Overview, §2–§2f]

**Per-cell frontier.** SR @ realized IR, Δ vs knob-off in pp:

| cell-size (off; pure) | arms so far |
|---|---|
| GR00T L10-50 (.750 @ .209; .898) | P.25 .774 (+2.4); R.32 .792 (+4.2); AE.32 .778 (+2.8); D.40 .796 @ .310 (+4.6); E.40 .826 @ .376 (+7.6); **P.40 .844 @ .399 (+9.4)**; R.40 .818 @ .403 (+6.8) |
| π0.5 L10-50 (.844 @ .167; .908) | R.25 .860 @ .231 (+1.6); D.32 .870 @ .212; Dis.32 .880; P+tail.32 .874; R2.32 .868; AE.32 .892; E.40 .902 @ .369; **P.40 .900 @ .393 (+5.6)**; AE.40 .898 |
| π0.5 Sp-50 (.910 @ .135 [R10]; .988) | **P.25 .972 @ .238 (+6.2)**; Dis.32 .980; E.32 .972; **P+tail.32 .990 @ .297 (+8.0)**; E.40 .980; AE.40 .974; R.40 .968 @ .387 (+5.8) |
| GR00T Sp-50 (.896 @ .127 [R10]; .940) | D.25 .862 (−3.4); D.32 .868 (−2.8); E.25 .902; R.25 .894; Dis.32 .914; R.32 .922 (+2.6); AE.40 .930 (+3.4); P.40 .910 (+1.4) |
| GR00T L10-200 (.820 @ .190; .898) | R.25 .848; P.25 .836; E.32 .858; **P.32 .884 @ .313 (+6.4)**; R.32 .836 (+1.6) |

**SR advantage over R at the same realized IR** [§2c]. Arm SR minus R's piecewise-linear curve, which includes the knob-off point; no extrapolation:

| method | arms in range | pp [95% CI] |
|---|---|---|
| P | 5 | **+2.15 [+0.44, +3.83]** (bootstrap p .02) |
| E | 5 | +1.84 [+0.46, +3.13] |
| Dis | 2 | +2.14 [+0.20, +3.95] |
| P+tail | 1 | +4.3 (π0.5 Sp-50) |
| AE | 2 | −0.25 [−3.2, +2.0] |
| D | 4 | −0.93 [−2.9, +0.9] |

Measured against P's curve instead, every other method is within ±1.5 pp of P, and none is significant.

**Matched-target paired P − R** [§2f]: +2.07 pp over 3 pairs (+169/−138, McNemar p .09). The IR-adjusted value is +2.08.
- **Opus's prediction P4 was +0.6 pp pooled.** The interim estimate is in the predicted direction but above the 80% prediction interval (−0.9 to +2.1).
- The largest single pair is GR00T L10-200 @ .32: P .884 vs R .836, +55/−31.

**Pooled efficiency over knob-off** [§2d], in SR pp per +0.1 IR:
- P +3.25 [+2.06, +4.51];
- Dis +2.96, E +2.95;
- pure policy on the 50-cells +2.40 [+1.90, +2.92];
- AE +2.34, R +2.27 [+1.11, +3.47];
- D +0.41.

These pooled slopes mix different cell compositions per method. The matched-IR advantage above is the fair comparison.

**Chord benchmark** [§2e]. Running knob-off on a share (1 − λ) of episodes and pure policy on the rest traces a straight line between the two. Advantage over that line at the same IR:
- R −0.59 [−1.86, +0.68];
- P +1.02 [−0.36, +2.51];
- E +0.99 [−0.09, +2.06];
- Dis +1.78 [+0.10, +3.26];
- P+tail +2.54 [+0.72, +4.42] (2 arms);
- D −1.52.

Spreading calls uniformly at random inside episodes buys about what whole-episode mixing buys. Only the structured schedules sit above the chord. This is the strongest single argument for a schedule over random placement.

**Answer to Q2 (interim).**
- Among the reliable knobs, periodic converts IR into SR most efficiently, about +2 pp over random at matched IR.
- E and Dis match P's conversion, but they cannot be steered to a target.
- Opus's +0.6 pp prediction for periodic over random is so far exceeded, not refuted. The estimate rests on only 5 matched arms.

### 3. Where the extra calls landed (Q3) [§3a–§3j]

- **By progress and time** [§3a–§3c].
  - R and P spread calls flatly over retrieved-demo progress and normalized time: relative intensity .95–1.05.
  - Guard calls are concentrated late. In the knob-off arms, guard intensity is ×0.44 in the first fifth and ×1.24 in the last two fifths.
  - So knob calls are front-heavy *relative to need*.
  - D fires about 2× at low progress, where the start of an episode is far from the library, and only about 0.45× in the middle.
- **Around guard calls** [§3d].
  - P's call rate at the anchor right after a guard call is .20 (the K < 1 arms), against .44 overall: a built-in refractory period.
  - R is flat (.38–.40).
  - P+tail calls at 100% right after a guard call and 0% one anchor later.
  - A knob call does not change the next anchor's guard probability for R (.209 vs .204 after a cache look) [§3e].
- **Failed vs successful episodes** [§3f, §3i].
  - Failed episodes run long (≈ 76–104 slots vs 31–51) and are guard-heavy (g ≈ .44–.48 vs .10–.16).
  - P spends less per eligible look in failing episodes than R: .355 vs .384 pooled; on matched pairs, .646/.148/.360 vs .667/.190/.438. The reason is that guard bursts keep resetting P's gap.
  - P also puts a smaller share of its knob calls into failed episodes (.14 vs .18).
- **Cache-only stretches** [§3g].
  - P caps the longest stretch at ceil(K) anchors (1–3 anchors = 10–30 controls).
  - R's mean longest stretch is 1.7–6.9 anchors, with p90 3–11 and a maximum of 17 (170 controls).
  - The closed-loop stretch statistics match the library replay closely. Examples: GR00T L10-200 R @ .25, 5.80 vs 5.71 predicted; π0.5 L10-50 R @ .25, 6.86 vs 6.33.
  - D leaves 6.5–11.3-anchor stretches, about as long as knob-off (9.1–12.3).
- **Where calls pay off** [§3j]. Difficulty is measured in an independent knob-off run of the same init, so chance failures of the reference do not leak into the split.

  | split | R | P | pure policy |
  |---|---|---|---|
  | hard inits (high early-guard third) | +7.2 pp | +12.7 pp | +17.6 pp |
  | inits that run failed | +19.7 pp | +27.7 pp | +44.0 pp |
  | inits it solved | −0.8 pp | −0.8 pp | +1.1 pp |

  - On inits the other run solved, the variants and state knobs lose 1–4 pp.
  - Extra calls help where the cache is failing and slightly perturb episodes where it is not.
- **State-knob score distribution** [§3h]: see §1 above.

### 4. Hardware check (Q4) [§4]

| pair | type | SR | +b/−c | p | IR | g |
|---|---|---|---|---|---|---|
| π0.5 L10-50: 4090 R11 off vs H100 R10 GC_dist | cross | .844 / .826 | +33/−24 | .29 | .167 / .168 | .212 / .214 |
| GR00T L10-50: 4090 idg vs H100 R10 GC_dist | cross | .750 / .754 | +54/−56 | .92 | .209 / .207 | .313 / .310 |
| GR00T L10-200: H100 R11 off vs H100 R10 GC_dist | same | .820 / .820 | +33/−33 | 1 | .190 / .191 | .268 / .272 |

**No systematic SR, IR or guard-rate difference.** Knob arms show the same IR calibration error on both GPUs: IR − informed is −.021 on both.

**Deterministic signature.**
- **Same-hardware replicate.** Every episode has the same top-1 at step 0, and none diverges before the first policy call.
- **Cross-hardware pairs.**
  - The step-0 top-1 agrees in 99.0% (π0.5) and 93.6% (GR00T) of episodes.
  - 42% / 59% of episodes diverge *before* the first policy call.
  - The cause is the stage-1 vision encoding, whose numerics differ by GPU and flip near-tie retrievals.
- After the first policy call, runs diverge even on the same GPU, because the policy noise is not seeded across processes.

**Conclusion (interim):** 4090 and H100 results are poolable for aggregate SR and IR; episode-level trajectories are not reproducible across GPUs.
- Still pending: the h100 R11 offs in `r11_knob_2`, which add three same-hw replicates, and the 4090 offs in `r11_local_k4`, which add three cross-hw pairs.
- When they arrive, they will test whether the discordance rate differs between same-hw and cross-hw pairs. That rate is .132 vs .114/.220 at the interim snapshot, and it is cell-dependent.

### 5. Recommendation for layer 4 (Q5), preliminary

1. **Adopt the guard-aware periodic knob** (SPEC §2, a dithered gap cap K per cell-size).
   - It is as IR-reliable as random: 7/7 within ±.02, and ±.004 once the closed-loop guard rate is known.
   - It is the best-converting reliable knob: +2.2 pp over R at matched IR, and +1.0 pp over the cache ↔ pure chord.
   - Keep R only as the calibration and reference knob.
   - P+tail is the most promising variant (2 arms; +2.5 pp over the chord). Decide after the GR00T L10-50 and GR00T Sp-50 P+tail arms complete.
2. **Knob → IR map from the library alone.**
   - The current replay (IR_MODEL) is adequate for schedule knobs. The residual is the base guard gap, −.005 to −.019 IR on π0.5 at low targets.
   - Hypotheses for a future round, library-only fitting and not tested here:
     - (a) estimate g with full-library donors, removing the 4/5-donor sparsity bias: size_sensitivity predicts part of the π0.5 gap;
     - (b) a per-episode spend tracker on top of P that charges guard calls against the budget (opus's SigmaDelta fallback), making IR insensitive to g;
     - (c) for any state knob, a random floor or online tracking is mandatory, because closed-loop score distributions shift below whole-episode-out calibration. AE's online dose already recovers most of the shift.
3. **Where to spend (hypothesis).** Gains concentrate on inits the cache fails, and calls on inits the cache solves cost about 1 pp.
   - A guard-escalating schedule would test this: shorten P's gap after repeated guard fires within an episode. It is calibratable on the library guard replay.
   - P+tail is a mild version of it.
4. **Knob off by default on near-pure cells.** This is pending: π0.5 L10-200 / 500 and GR00T L10-500 have one state-knob arm each so far, with ΔSR of +1.6 and −1.0 pp, both n.s.
   - Pure policy itself adds only +0.4, +1.4 and −0.8 pp there over the R10 3-layer base [§2].
   - The final version will propose a library-only switch rule.

### 6. Opus pre-registered predictions vs measurement (PREDICTION.md, 2026-10-02 21:10 CDT) [§5]

Only 13 of the 38 pre-registered arms are complete and on the final grid; the coordinator moved the Spatial targets.

- **P1 (IR):**
  - 13/13 within ±.02 of the library prediction;
  - 13/13 within .012 of the informed prediction;
  - π0.5 mean −.013 (predicted ≈ −.008); GR00T L10-50 +.002 (predicted ≈ +.005).
- **P2:** R/P within .01 IR in 3/3 pairs. "P closer to its prediction" holds in 1/3, so far against the claim.
- **P4:** +2.07 pp (predicted +0.6).
- **SR change vs off:** the predictions were optimistic by 1.2 pp on average (sd 1.5). Random on GR00T L10-50 was predicted +7.2 / +10.5 but realized +4.2 / +6.8.

### 7. Scripts (all in `exp/offline_search/rounds/r11/opus/analysis/`)

| file | role |
|---|---|
| `run_analysis.py` | entry point: arm discovery, references, bootstrap, every table → `out/TABLES.md`, `out/results.json`, `out/arms.csv` |
| `extract.py` | read-only parsing of journals, cost ledgers and server decision logs into per-episode / per-anchor tables; schedule-rule, coin, cadence and ledger audits; per-arm cache in `cache/` keyed on file size and mtime |
| `stats.py` | stratified paired bootstrap, McNemar, curve interpolation, Shapley decomposition, run-length helpers |
| `out/TABLES_INTERIM.md`, `out/results_interim.json` | frozen snapshot behind this document |

The library-side predictions reuse `../ir_model.py` (opus methods' cache-stretch replay) and `../out/{curves,predictions}.json`, all frozen before the sweep. Nothing is refitted.
