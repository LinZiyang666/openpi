# R9 explore_fable — ranked proposals

Ranking = expected gain × confidence. Numbers are from `DATA_ANALYSIS.md` (discovery inits 0–29 only).
Dose-simulator numbers are paired mixtures of real closed-loop outcomes; shadow-gap numbers are offline
screening. Cells: π0.5 / GR00T × LIBERO-10 / Spatial × 50 / 500 demos. A = pure cache (IR ≈ .076), CU =
uniform calls ρ .30 + stall trigger, P10 = pure policy every 10 controls (IR .50).

---

## P1. Per-task call budget from a 10-episode shadow calibration (allocation across tasks, not placement in time)

**What.** Keep the uniform-calls controller (random calls + calibrated stall trigger) but give every task its
own call rate ρ_t: the k hardest tasks get ρ ≈ .3 (or pure policy), the others run pure cache (ρ = 0).
Difficulty score = mean served-vs-shadow action gap (or gripper disagreement, or mean retrieval distance) over
~10 pure-cache episodes per task on B-pool inits, with the policy evaluated in shadow afterwards. No success
labels are needed.

**Why.** 3–5 tasks carry the whole P10 − A gap in every sparse cell and 4–5 tasks have A ≥ P10, so uniform
spending wastes about half the budget. The ranking from 10 episodes per task reproduces the 30-init ranking at
Spearman .88–.97 (shadow gap) and is far more stable than the task's own success rate on 10 episodes.

**Evidence (3-fold cross-validated, paired against the uniform-dose family at the same owner IR; real
closed-loop outcomes).**

| cell | rule | SR @ IR | gain vs IR-matched uniform [95% CI] |
|---|---|---|---|
| π0.5 L10-50 | top-3 tasks CU, rest A | .833 @ .139 | **+7.4 pp [+3.8, +10.9]** |
| π0.5 L10-50 | top-4 P10, rest A (gripper signal) | .883 @ .237 | +4.5 [+1.1, +8.2] |
| GR00T L10-50 | top-3 CU, rest A | .753 @ .147 | **+6.4 [+1.8, +11.0]** |
| GR00T L10-50 | top-3 P10, rest A | .810 @ .200 | **+8.1 [+4.6, +11.6]** |
| π0.5 Sp-50 | top-2 CU, rest A | .887 @ .124 | **+4.0 [+1.4, +6.6]** |
| π0.5 Sp-50 | top-5 CU, rest A | .917 @ .194 | +4.8 [+2.7, +7.0] |
| GR00T L10-500 | top-2 CU, rest A | .857 @ .108 | +2.9 [+0.2, +5.3] |
| π0.5 L10-500 | top-5 CU, rest A | .913 @ .131 | +2.5 [+0.7, +4.3] |
| GR00T Sp-50 | top-5 CU, rest A | .923 @ .176 | +3.4 [−0.7, +7.4] |
| Spatial-500 | any | ≈ .96–.99 | ≈ 0 (A already ≈ P10) |

Equal-SR reading: π0.5 Sp-50 reaches .943 at IR .176 instead of CU's .927 at .304 (−42% cost); π0.5 L10-50
.887 at .268 vs .880 at .301; GR00T L10-50 .843 at .336 vs .797 at .303.

**Expected operating points.** π0.5 Sp-50: ≈ P10 at IR .30–.40 (today .44). π0.5 L10-50: .88–.89 at .24–.27
(today .894 needs .458). GR00T L10-50: .84 at .34 (today .868 at .453). GR00T L10-500: .86–.88 at .11–.19.
Dense Spatial: no change. **Confidence: high** on the sparse cells (CIs exclude 0 under cross-validation and
under four different ranking signals), medium on 500-demo LIBERO-10.

**Failure modes.** Over-spending when a saturated task is ranked hard (P10-on-top-k on π0.5 L10-500: −2.6 to
−4.9 pp vs the matched mix): cap ρ_t at .3 unless the task's calibration success is < .8. Task-level knobs are
benchmark-structured; recalibrate on a new benchmark. Realized IR depends on episode lengths per task; the dose
simulator's IR already uses realized costs.

**Exact confirmation plan.**
1. Controller: `CallController` with a per-task ρ table (a thin subclass resetting `rho` / the solved
   `parameter` per task in `reset()`), or two fitted controllers switched by task id. Calibration: 10 pure-cache
   episodes per task on B-pool inits with deferred shadows → `tools/task_alloc.py::task_scores` → top-k.
2. Arms per cell on inits 0–29 (300 pairs), same topology: A, CU ρ .3, per-task {top-3 ρ .3 / rest 0}, per-task
   {top-3 P10 / rest 0}; cells π0.5 L10-50, GR00T L10-50, π0.5 Sp-50 (12 arms ≈ 2 h). Predicted SR/IR of every
   arm is in `out/task_alloc/rules.json`.
3. Bar: per-task arm ≥ CU SR at ≤ 60% of CU's IR (Sp-50) or ≥ CU + 4 pp at ≤ 50% IR (L10-50), paired McNemar;
   then holdout 30–49.

---

## P2. Per-task library choice: shadow-labelled "DAgger rows" only where the original demos are weak

**What.** Grow the library with policy-labelled cache rollouts (`tools/grow_library.py`: every look decision
of a pure-cache episode becomes a row (keys, state) → policy chunk; the frozen A fit is extended with a
registered library; no new serving code; online IR unchanged at ≈ .076) — but use the grown rows **only for
the tasks ranked hard** by the same label-free score as P1, and the original library elsewhere.

**Why / evidence.** Offline, grown rows cut the action gap to the policy by 31–45% on every sparse cell with the
deployed metric (held-out inits). Closed loop (π0.5 L10-50, 100 held-out pairs, same topology) the *uniform*
grown library changed nothing: A .74, grown(A rows) .70, grown(all rows) .73, −1 pp [−11, +9] at IR .0764 vs
.0765. Per task it helped where the 5 demos fail (task 0 .4 → .8, task 4 .4 → .8, task 9 .7 → .9) and hurt where
they succeed (tasks 2, 3: 1.0 → .5/.6; there the policy itself is weaker than the cache). The post-hoc
per-task choice on the same pairs — grown rows on the top-3/4 tasks by mean retrieval distance computed on
inits 0–19, original library otherwise — gives **.83 vs .74, +9 pp [+2, +16]** at identical IR; with the shadow-gap
ranking +7 [+1, +13]; with A-row grown libraries +4 to +8 (CIs include 0). This is a selection among 18 variants
on 100 pairs: a hypothesis with exact numbers, not a result.

**Expected effect if confirmed.** +5 to +9 pp at unchanged IR on π0.5 L10-50; by the offline pattern the same
mechanism applies to GR00T L10-50 and π0.5 Sp-50 (grown artifacts exist: `fits/r9f_grown{A,all}_{g_l10_50,p_sp_50}.pkl`).
Combined with P1 the two per-task decisions use the same calibration run. **Confidence: medium-low** (post hoc,
100 pairs, one cell).

**Failure modes.** Grown rows mimic the policy including its weaknesses (tasks where the policy is worse than
the demos); labelled states include failing cache trajectories; 97% of retrievals come from new rows when
74k are added, so the original demos stop mattering — a per-row prior for original demos or a d1 gate ("use
grown rows only when the nearest demo is far") is the natural refinement.

**Exact confirmation plan.**
1. Preregister the rule: hard set = top-3 by mean retrieval distance of the pure cache on the calibration
   inits; grown rows = all-arm rows (or, for the deployable version, B-pool pure-cache rollouts with shadows).
2. π0.5 L10-50 on inits 0–19 (200 pairs; the grown rows for this check must then come from B-pool or from
   inits 20–29): arms A, per-task-library, P10. Then GR00T L10-50 and π0.5 Sp-50 the same way (6 arms ≈ 40 min).
3. Bar: ≥ A + 5 pp paired at IR within .002 of A.

---

## P3. What to stop pursuing (negative results from this round)

- **Growing the library uniformly with policy-labelled cache rows**: −38% action gap offline, 0 ± 10 pp closed
  loop (above). The same caution applies to the distilled **student MLP** (offline .238 vs .412 gap on π0.5
  L10-50, .282 vs .525 on GR00T L10-50): its premise is the same offline metric; not pursued further without a
  closed-loop test, and then only per task.
- **Re-weighting the retrieved neighbours** (top-1, uniform-k, kernel widths, local linear): ≤ 3% of the gap.
- **Episode-level gating from early disagreement or step-0 retrieval statistics**: AUROC .40–.65.
- **Gripper majority / hysteresis at synthesis**: no change.
- **Placement signals below the task level**: consistent with R6–R8; the predictable heterogeneity is between
  tasks, not between moments.

---

## Summary

| cell | today's cheapest point ≈ P10 SR | P1 per-task calls (CV, real outcomes) | P2 per-task library (hypothesis) |
|---|---|---|---|
| π0.5 L10-50 | not reached (.894 @ .458) | .887 @ .268; .833 @ .139 (+7.4 vs uniform) | .83 vs .74 @ .076 post hoc (+9 [+2, +16]) |
| π0.5 Sp-50 | .986 @ .442 | .943 @ .176; .887 @ .124 (+4.0) | offline gap −37%; not run |
| GR00T L10-50 | .868 @ .453 | .843 @ .336; .753 @ .147 (+6.4); .810 @ .200 (+8.1) | offline gap −36%; not run |
| GR00T Sp-50 | .940 @ .298 | .927 @ .162 (+3.8, CI incl. 0) | — |
| π0.5 L10-500 | .906 @ .197 | .913 @ .131 (+2.5) | — |
| GR00T L10-500 | .866 @ .184 | .857 @ .108 (+2.9); .880 @ .191 (+5.0) | — |
| Spatial-500 | A ≈ P10 (.052–.118) | no change | no change |
