# R6 method selection: configuration C, library-calibrated budgeted rescue

Written 2026-09-29 07:5x CDT by the coordinator after the three preregistered finals (`ideation_Q1/FINAL.md`,
`ideation_Q2/FINAL.md`, `ideation_Q3/FINAL.md`), the three-replicate A/B table (`PAPER_AB.md`), the ablations
(`ABLATIONS.md`) and 25 of the 43 frontier-completion arms (ledger §10, 2026-09-29).

## 1. Decision

The paper keeps A (Commit-Cache) and B (A + guard-triggered committed rescue) as the measured configurations. R6
adds **configuration C**: A plus rescue calls whose **amount** is set by one owner knob and whose **placement** is set by
two library-calibrated signals. C has no hand-set thresholds and no benchmark or robot names.

| Component | Owner question | What C does | Source |
|---|---|---|---|
| Budget | How many MISS | Owner knob **ρ = target owner IR**. The per-anchor base dose d(ρ) comes from the library's decision lengths and the vision cost share c1 (Q2 budget algebra, uniform over tasks). | Q2 §3 steps 1–4, `ideation_Q2/frontier/adapters/budget.py` |
| Placement by disagreement | Where to MISS / library quality | Every bank row stores one scalar r_j, its leave-one-episode-out action reconstruction residual under the deployed kernel. At an anchor, R(s) = Σ w_i r_i over the retrieved rows. It is calibrated to predicted cache–policy disagreement Ehat(s) = a + b·R(s). Call probability p(s) = min(1, λ·Ehat(s)), where λ matches the calibration recordings' mean call rate to d(ρ). | Q1 §4 (selected score R: ρ .468, MAE −15.4%, all 8 cells positive) |
| Stall rescue | Where to MISS | Replaces B's raw no-progress package. Each library template's phase is aligned monotonically to the last W+1 keys. The estimation error (e90) and the normal advance (a10) are calibrated leave-one-out on the library's successful episodes. `slow_confirmed` (delta_hat + e90 < a10) triggers a call. `slow_ambiguous` triggers one extra LOOK instead of a call. Its calls and LOOKs count against the same ρ. | Q3 §3 steps 1–5 |
| Commit / handback | — | Commit min(H, 2R) controls (10 here), hold 1, then a fresh retrieval anchored on the executed tail. This is B's rescue semantics unchanged. **Cooldown applies only after a stall-triggered call** (amended 08:3x, see §7); lottery calls have no cooldown, as in `RiskLottery` / `CacheDose`. | B / C10 |

B's other three guards (stuck, terminal, overtime) are **not** part of C. They are hand-thresholded, and each is
individually removable without a detected SR loss (trigger leave-one-out). Removing them jointly is tested by C itself.

**Calibration inputs (the portability contract).** The inputs are:
- the library;
- per cell, **at most 10 recorded A-controller trajectories** (one per task), carrying the same-observation shadow
  policy chunk at every anchor;
- the policy's measured vision / full-inference cost share c1.

The trajectories are recorded on **non-test initial states**. For LIBERO these come from the B-val pool
(`exp/ablation_study/config/common/split_<suite>.yaml`, `exp/common/data/db_init/libero/<suite>/`), never the official
test inits. Everything else is library-only:
- r_j, and the progress templates, e90 and a10;
- the library lengths used by the budget.

No success label enters any fit. ρ is the only knob.

## 2. Why this and not something else

- **Amount matters more than guards at high budget.** Consider sparse cells with extra random calls on top of B, or
  risk-allocated lotteries on top of A:
  - π0.5 L10-50: B+.5 reaches .880 @ .34 and ρ .45 reaches .874.
  - GR00T L10-50: B+.75 reaches .850 @ .43.
  - π0.5 Spatial-50: B+.75 reaches .978 @ .41 and ρ .45 reaches **.986 @ .44**. That point estimate equals pure L10
    (.986 @ .5), with 7 wins and 7 losses.

  B's guards buy SR cheaply at IR ≈ .15–.2. They do not set the ceiling.
- **The raw no-progress guard is not portable.** It carries all of B's gain on both L10-50 cells. On π0.5 Spatial-50 it
  carries about 5 of the 7 pp. On GR00T Spatial-50 it is harmful: without it B reaches .906 @ .105 (p ≤ .01), and no
  simple library proxy predicts the sign (Q3). The ablation also removed the guard's blind-LOOK veto, so its effect may
  be observation cadence rather than policy takeover. C's stall component targets the same failure (long-task loops)
  with a calibrated estimate. The mechanism arm separates the two explanations.
- **R is the only library score that survived preregistration**, and it is cheap: one stored scalar per row, a weighted
  sum per anchor, and no online shadow. It predicts disagreement, not failure (Q1). Using it for placement is therefore a
  hypothesis tested against uniform placement at matched cost.
- **Task-level coverage risk did not beat uniform.** It point-dominated in 3 of 32 held-out comparisons (Q2). C
  therefore allocates per anchor (R) rather than per task. Uniform allocation is the baseline arm.
- **Minimal IR.** No point cheaper than pure inference passes 2 pp noninferiority (NI) under multiplicity (Q2). C's
  purpose is to lower the crossing IR. The validation reports each cell's crossing, or its absence, honestly.

## 3. Not claimed

- Neither R nor Ehat is a success predictor or a call-value estimate.
- The placement advantage and the stall component are unvalidated hypotheses.
- The method was designed after seeing outcomes on test inits 0–49. The validation below is therefore a conditional
  check on those scenes, and calibration never touches them.
- A clean transfer claim needs another benchmark or robot. RoboCasa365 was named by `ideation_G`.

## 4. Validation: 32 arms × 500 episodes

The episode set is the same 500 (task, init) pairs as every prior arm (test inits, measurement only). Every arm is
paired against A (three replicates), B (three replicates), pure L10 and pure L5.

| Cells | Arms | Question |
|---|---|---|
| π0.5 L10-50, π0.5 Sp-50, GR00T L10-50, GR00T Sp-50 | **U.30**: A + uniform per-anchor calls at ρ=.30 (`RiskLottery` `allocation=uniform`) | Matched-cost control |
| same | **R.30**: A + R-placed calls at ρ=.30 | Placement: R.30 − U.30 at \|ΔIR\| ≤ .01 |
| same | **C.30**: R placement + stall component at ρ=.30 | Stall component: C.30 − R.30 |
| same | **C.45**: as C.30 at ρ=.45 | Reach: C.45 vs pure L10 / L5, 2 pp NI (Q2's conservative Clopper–Pearson bound) |
| same | **Bmech**: B with only the no-progress MISS bit off; blind-LOOK veto and histories kept | Mechanism of the reversal (vs B and vs B-without-NP) |
| π0.5 L10-500, π0.5 Sp-500, GR00T L10-500, GR00T Sp-500 | **U.18, R.18, C.18** (ρ=.18 ≈ B's IR) | Dense libraries: does C match B at B's cost without hand guards? |

**Acceptance and falsification.** Each criterion is declared before outcomes.
- R placement is supported if R.30 − U.30 > 0 pooled over the four sparse cells. The pooled effect uses a paired,
  task-stratified init-cluster bootstrap, and realized IR must match within .01. It is falsified at this operating point
  if the pooled 95% upper bound is below 0, or if realized IRs cannot be matched.
- The stall component is supported if C.30 keeps at least half of B's no-progress benefit on both L10-50 cells and does
  not lose to B-without-NP on GR00T Sp-50 by more than 2 pp. It is falsified if it loses to R.30 on the L10-50 cells, or
  if it rejects nearly every long-task trigger.
- Reach is reported per cell: the lowest C ρ with the 2 pp NI lower bound passing against L10, and separately against
  L5, or "not reached ≤ .45".
- Dense cells: C.18 has SR within 2 pp of B (lower bound) at realized IR ≤ B's.
- Realized IR outside ρ ± .02 falsifies the portable cost calibration for that cell.

## 5. Build order

1. **P4 (codex, Q1 thread)**: the C controller, which also runs U and R.
   - Precompute r_j per bank row, and fit Ehat and λ from recordings.
   - Emit specs for the calibration recordings. These are P3 v2 A-cohort runs with an every-anchor shadow, on B-val
     inits, 10 episodes per cell.
   - Emit all 32 validation specs.
   - Replay tests:
     - ρ → A-floor gives A bit-identical;
     - uniform gives `RiskLottery(uniform)` rates;
     - seeded reproducibility.
2. **P5 (codex, Q3 thread)**:
   - The stall component (Q3 steps 1–4): library windows, the monotone-alignment dynamic program, LOEO tables and an
     online tracker.
   - The Bmech judge.
   - Unit tests.
   - Per-anchor CPU timing.

   It is written against the interface below, so P4 and P5 run in parallel.
3. Coordinator: record the calibration trajectories (80 episodes), fit the calibration artifacts (codex), run a 2-arm
   smoke, then the 32 arms (front of the queue), then the analysis (codex), then commit.

Interface between P4 and P5, in `ideation_Q3/stall/stall.py`:

```python
class StallModel:                      # library-only, built offline
    @classmethod
    def fit(cls, library, *, metric, commit_controls) -> "StallModel": ...
    def save(self, path): ...
    @classmethod
    def load(cls, path) -> "StallModel": ...

class StallTracker:                    # one per episode, pure CPU, no policy calls
    def __init__(self, model: "StallModel", task_key): ...
    def observe(self, key, control_index: int) -> None: ...   # every fresh vision observation (anchor or extra LOOK)
    def status(self) -> dict: ...      # {'state': 'inactive'|'ok'|'slow_confirmed'|'slow_ambiguous',
                                       #  'delta_hat','e90','a10','window_span', ...} logged per decision
```

## 6. Owner decisions (asked in the message body)

1. **Full pilot continuation (+31,680 episodes): recommend not running it as designed.**
   - All three finals agree that it cannot certify a 2 pp margin: pass probability at a true zero gap is 0.1–2% (Q2),
     and the 1 pp margin needs 3,092–19,922 effective pairs (Q3).
   - R's selection is already reached (Q1).
   - Its GPU time is better spent on the 32 C arms plus new-init support.
2. Unsupervised whitening Σ⁻¹ control (8 pure-cache arms): still pending.
3. Display convention for the L10 +0.04 shift: still pending.

## 7. Amendment 2026-09-29 08:3x CDT (before any C outcome)

P4 showed that a one-free-anchor cooldown after *every* call caps C's owner IR at about c1/2 + (1−c1)/4 ≈ .29. That makes
every .45 target, and most .30 targets, infeasible. The cooldown came from Q3's step 5, where it applies to the stall
component only. The frontier's risk lotteries reach pure SR at ρ = .45 without any cooldown. Amended semantics
(controller version `R6-C-v2`):
- the cooldown follows **stall-triggered** calls only;
- R-placed and uniform lottery calls have none;
- everything else is unchanged.

No C outcome has been observed; the acceptance rules in §4 are unchanged.

### 7b. Amendment 2026-09-29 08:4x CDT (before any C outcome)

The v2 dry run (pilot test inits, code check only) still leaves C.45 infeasible with the stall component on both
L10-50 cells: the stall ceiling is about .39. Two rules cause this. `slow_ambiguous` anchors force p = 0, and there
are many confirmed-stall cooldowns. Ambiguity is evidence against a *stall* call, not against a budgeted call.

Amended semantics (still `R6-C-v2`, since no C arm has run):
- at a `slow_ambiguous` anchor, the R / uniform lottery applies exactly as at an `ok` anchor;
- the extra LOOK is scheduled only when that lottery does not call;
- `slow_confirmed` handling and its cooldown are unchanged.

If a target is still infeasible after calibration on the B-val recordings, that arm is reported as infeasible at that ρ
and run at the calibrated ceiling under an explicit `C.max` label. It is never relabelled.
