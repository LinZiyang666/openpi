# R11 shared IR accounting model (opus): knob -> theoretical owner IR on the 3-layer base

Written 2026-10-02 ~21:05 CDT for the R11 explorers (opus, astra) and sol. Code: `ir_model.py` (model, knob
schedules, simulator), `loeo_anchor_table.py` (library replay that feeds it), `offline_curves.py` (curves,
target solves, stress), `r6_overlap_check.py` (validation of the overlap rule on R6 aggregates). Library data
only. Nothing under `trace_runs/os_closed_loop` is read.

## 1. Ledger definition

`IR = (c_v·V + c_m·M) / N`
- `N`: decision slots, one per 5 executed controls. Blind cache tails and policy tails are slots that cost 0.
- `V`: vision decisions, i.e. cache LOOKs plus policy calls. A MISS is also a vision decision.
- `M`: policy calls (MISS).
- `(c_v, c_m)`: π0.5 `(.152, .848)`, GR00T `(.148, .852)`.

**Per ten-control unit.** A LOOK costs `c_v` over 2 slots (≈ .076 per slot). A MISS costs `c_v + c_m = 1` over 2 slots (.5 per slot). Pure policy with a ten-control commit therefore costs .5.

**Short form.** With `v = V/N` and anchor miss fraction `f = M/V`: **`IR = v·(c_v + c_m·f)`**.

## 2. Cadence of the 3-layer base (R10Recipe)

R10Recipe is the R8 `TriggerCommitJudge` (π0.5) or `TriggerGrootCommitJudge` (GR00T) with the anchor-tail BlindAWM plus the GC_dist corrector, under `--os-blind --os-policy-tail --os-judge guard_only`:
- **Anchors** are the vision decisions at decision steps 0, 2, 4, …
- **After a cache LOOK**, the odd slot is the blind tail of the same ten-control chunk.
- **After a MISS**, the odd slot is the committed policy tail (0 cost), followed by a fresh anchor.
- **Extra LOOKs** can come from lifecycle refusals or the no-progress blind veto (`span > 0` at a blind slot). They are rare: across R6's A and B arms, v is .502–.514.

**Library v.** The model uses `v = Σ ceil(n_ep/2) / Σ n_ep` over library episode lengths. This gives .503–.504 on LIBERO-10 and .509–.510 on Spatial, matching R6's ledgers.

## 3. The guard, replayed on the library

**The rule.** The frozen R8 only-no-progress guard (`noprog_span`, `noprog_n = 3`, `prog_eps = .5`) fires at anchor i iff the cache's top-1 library progress has not advanced by more than half a library step for ≥ 2 decision steps:
- each anchor-to-anchor gap counts `(prog_i − prog_{i−1})·(ep_len(top1_i) − 1) ≤ .5`;
- non-advancing gaps accumulate a span, and the guard fires when the span is ≥ 2;
- at the normal two-step cadence, a single non-advancing gap is therefore enough.

`ir_model.guard_flags` is a line-for-line port of R4 `BlindMixedJudge._progress`. `test_ir_model.py` checks it against the frozen function itself on 300 random progress sequences and finds them identical.

**Replay design (`loeo_anchor_table.py`).** For each R10 nested cell-size:
- **Folds.** Five whole-episode-out folds, assigned by permutation position % 5 exactly as the GC_dist distance scale does.
- **Cache refit.** For each fold, the cache is refitted on the training episodes:
  - PCA is refitted on the training fold for sizes < 500; size 500 uses the stored bpool basis;
  - the per-task main and early metric uses nn 3 and λ .1;
  - kref is 5 at 5 episodes per task and 8 otherwise.
- **Serving.** Every held-out episode is served as a new episode at every row.
- **Recorded per row:** the top-1 row's progress and episode length; the uncorrected motion error against the row's own stored policy chunk; the gripper sign mismatch; d1.
- **Corrected error.** R10 opus's library-only k-fold outputs are joined by row id (exact match, max |Δe10| = 0). That brings in the LOEO corrector's dot/cc terms, so the GC_dist-strength corrected error `s = .5·clip((2 − r)/1.25, 0, 1)` is reconstructed exactly (verified to 3e-8 against R10's stored blend outputs).

**Base guard rate g (share of anchors) and base IR from the library replay.** The full table is in `OFFLINE.md` §1. Summary: g is .26 / .29 on L10-50 (π0.5 / GR00T), .19 / .27 on L10-200, .20 / .27 on L10-500, and .15 / .15 on Spatial-50.

## 4. Overlap rule: guard first, knob on the rest

A knob acts only on anchors where the guard did not fire; a guard call is never "double-paid". For an independent knob that calls with share k of the eligible anchors:

`f = g + (1 − g)·k`, so `IR = v·(c_v + c_m·(g + (1 − g)·k))`

- For the random knob, k = ρ exactly.
- For history-dependent knobs (periodic, sigma-delta, tails, refractory), `ir_model.simulate` runs the schedule anchor by anchor on every held-out library episode and returns the slot-weighted IR, f, k and diagnostics.
- The closed-form result and the simulation agree to < .002 for the random knob.

**Validation on closed-loop aggregates (disclosed, nothing fitted).**
- **Data:** `r6_overlap_check.py` reads R6's arm-level ledger table `rounds/r06/frontier_final/frontier_points.csv` (N, V, M per arm; no episode data).
- **Arms:** R6's "B + random anchor dose" arms implement exactly this rule: B's guard first, then a keyed coin on non-guard anchors.
- **Prediction:** from each cell's B replicates alone, `f_pred = g_B + (1 − g_B)·dose`.
- **Result** (`out/r6_overlap_check.md`): over 13 arms with dose .125–.75, predicted − observed owner IR is −.003 to +.011, with median +.003.
- **Cause of the small residual:** extra policy calls mildly lower the closed-loop guard rate. The largest case is π0.5 Spatial-50, where the implied g falls from .145 to .08–.10 at dose .5–.75.
- **Conclusion:** *given the closed-loop base guard rate*, the accounting is accurate to about ±.01.

## 5. What cannot be estimated offline, and how it is bounded

| Unknown | Why offline cannot see it | Effect on predicted IR | How handled |
|---|---|---|---|
| **Closed-loop base guard rate** g_cl | The replay runs the cache on the *policy's* trajectories. In closed loop the cache drives, so stall runs, failure share and length differ. | Random knob: `∂IR/∂g = c_m·v·(1 − ρ)` ≈ .43(1 − ρ). A ±.04 error in g gives ±.017(1 − ρ). The periodic gap cap and the sigma-delta budget absorb part of it, because guard calls reset or consume the knob's budget (see the stress columns in OFFLINE.md). | **Stress test:** the guard flags are thinned or thickened to ×.75 and ×1.25 of g, and the realized IR is recomputed at the knob setting solved on the unperturbed library. **Disclosed check:** see the table below this one. **Dev pilot:** the optional B-pool non-test pilot (BRIEF §5) measures g_cl directly. |
| **Closed-loop episode lengths** (slot weights) | Closed-loop failures run to the step limit (104 / 44 slots). The library holds pure-policy lengths. | v is insensitive (.50–.51). f depends on the weights through g heterogeneity: in the library, failed episodes have g ≈ .3–.55 vs .03–.23 for successful ones. | Inside the ±25% stress band. |
| **Knob feedback on the guard** | Calls change later states. | Over-prediction of ≤ .011 at dose .5–.75 (R6). | State it and accept it. |
| **Extra LOOKs** (lifecycle, veto) | Need closed-loop blind-slot events. | ≤ .003 (v .503 vs .514 at most). | Ignored. |

**Disclosed check against R10's published aggregates.** These are test-set-A IRs of the 3-layer base from `rounds/r10/REPORT.md`. They are used **only** to state the model's accuracy; nothing is tuned to them. The implied closed-loop g is `(IR/v − c_v)/c_m` with library v.

| cell-size | library base IR | R10 3-layer IR (test A, aggregate) | implied closed-loop g | library g |
|---|---|---|---|---|
| π0.5 L10-50 | .187 | .168 | .214 | .259 |
| π0.5 L10-200 | .158 | .150 | .172 | .189 |
| π0.5 L10-500 | .164 | .150 | .172 | .204 |
| π0.5 Sp-50 | .142 | .135 | .133 | .151 |
| GR00T L10-50 | .199 | .207 | .309 | .291 |
| GR00T L10-200 | .191 | .191 | .272 | .271 |
| GR00T L10-500 | .192 | .175 | .234 | .274 |
| GR00T Sp-50 | .141 | .127 | .118 | .150 |

Source: `r10_check.py` → `out/r10_check.md`, which also covers the four Spatial-200/500 context cells.

**Reading the check.** The library replay over-estimates the closed-loop guard rate by .00–.045 on 7 of 8 cell-sizes and under-estimates it by .02 on GR00T L10-50. That is roughly the ×.75–×1.1 range covered by the stress test. Two reasons account for part of it:
- **Sparsity.** On 50-cells the replay serves from 4 of 5 episodes per task. `size_sensitivity.py` shows that this alone raises g by .002–.014, and the bias is predictable from the library itself.
- **Corrector.** In closed loop the GC_dist corrector reduces stalls (R10: IR fell about .01 when the corrector was added).

**Consequence.** Solved on the library, the random knob will tend to land slightly *below* its target on π0.5 cells, by ≈ .43·(g_lib − g_cl)·(1 − ρ) ≈ −.005 to −.02. The periodic gap cap moves less (§6).

## 6. Usage

```python
from exp.offline_search.rounds.r11.opus import ir_model as M
eps = M.load_episodes("pi05", "l10", 50)              # held-out library episodes with guard flags + risk proxy
M.base_stats("pi05", eps)                              # v, g, IR of the knob-off base
M.simulate("pi05", eps, M.Random(.4))                  # IR, f, knob share, capture ratio, cache-run stats
M.simulate("pi05", eps, M.DitheredGapCap(1.5))         # owner's periodic miss, guard-aware
M.solve_rho("pi05", v, g, .32)                         # closed form for the random knob
```

**Plugging in a state-dependent knob (astra).** Subclass `Knob` and return `(guard, knob_call)` per episode from `calls(ep, rep)`. Each `Episode` carries per-anchor `d1norm`, `risk` (corrected), `risk_raw`, `grip` and `guard`, so for example `knob_call = ~guard & (ep.d1norm > tau)`. Then `simulate` gives the IR, f and slot weighting under the same accounting and overlap rule.

**Feeding a different anchor table.** Anything with the same per-row fields (ep, step, ep_len, success, top1_prog, top1_eplen, e10, grip_mis and the joined `r10|…` terms) can be passed through `load_episodes(anchor_dir=…)`.
