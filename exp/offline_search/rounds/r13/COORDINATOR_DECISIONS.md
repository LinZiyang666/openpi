# R13 coordinator decisions (documented deviations from the frozen plan)

All decisions below were made **before any R13 test-A arm started**. None used any pilot or test-A SR.

## D1 (2026-10-04 ~17:00 CDT): the size-500 prerequisite is read per model
**Frozen text (SPEC):** "Size-500 can proceed only after the same controller passes on available smaller-library cells and mechanics are validated for its actual fit."

**astra's reading (`GATES.md`):** all available smaller cells of **both** models. On that reading, GR00T L10-500, and with it the whole `r13_test_groot_l10_rep*` batch, is held until π0.5 L10-50 passes.

**Coordinator ruling:** the prerequisite is applied per model.
- **Reason:** each model's distance_feedback controller carries its own fitted scalars. GR00T's size-500 library-only transfer is informed by GR00T's own smaller cells, not by π0.5's pilot.
- **GR00T:** L10-50 and L10-200 both pass the budget and engagement gates, so GR00T L10-500 and the `r13_test_groot_l10_rep*` batches are released.
- **Retained label:** the one paired-budget miss (GR00T L10-200 at .25, D−P = −.0136) keeps its "not a matched-budget point" label.
- **π0.5:** π0.5 L10-500 still waits for the π0.5 L10-50 pilot verdict.

## D2: the π0.5 L10-50 pilot ran on the local RTX 4090 instead of h100
- **Why:** h100 is lent to another project (I017 wall-clock timing) until about 2026-10-05 07:30 CDT. The reassignment was made for capacity reasons before any of its outcomes existed.
- **Coordinator ruling:** this pilot counts as the π0.5 L10-50 validation.
- **Evidence:** R11 measured no π0.5 host effect (same batch and configuration: .844 vs .826, p = .29, identical IR).
- **Scope:** budget and engagement gates are ledger and score properties, and their dependence on host should be negligible.
- **Analysis:** must report the pilot host per cell.

## D3: replicate blocks are ordered per fleet
**Frozen plan:** complete replicate blocks in order across both fleets.

**Coordinator ruling:**
- GR00T blocks run in frozen order on the local fleet. π0.5 blocks run in frozen order on h100.
- A fleet does not wait for the other fleet's block.
- Within a fleet, the frozen arm order inside each root and the root order are unchanged.

**Reason:**
- h100 is unavailable until about 07:30, and the local fleet would otherwise idle for most of a day.
- All primary contrasts are within model, cell, dose and block, with paired methods adjacent on the same host. Cross-model block alignment carries no estimand.

**Analysis:** report the wall-clock time of each block, and run the frozen leave-one-replicate-block sensitivity per model.

## Application (2026-10-04 18:0x CDT, after the π0.5 L10-50 pilot; still before any test-A arm)
**Re-run of `evaluate_gates.py`:** 64/64 pilot arms included. The first-pass outputs are saved as `tmp/r13/GATES_v1.md` and `gates_v1.json`.

**π0.5 L10-50 pilot result:**
- Every primary arm passes the budget gate:
  - distance_feedback: −.004 to +.002;
  - periodic: −.012 to −.004. Periodic is systematically slightly under; R11 saw the same π0.5 guard-rate effect.
- Engagement passes.
- Mechanics: 0 violations.
- The paired label fails at .32 rep0 only (D−P = +.0125; rep1 is +.0100).

**Verdicts:** astra's only reason for holding the cell is the host. Under D2, π0.5 L10-50 is released, with the .32 rep0 unmatched label retained. Under D1, π0.5 L10-500 is released as a library-only transfer, because all π0.5 smaller cells passed.

**Effect on the batches:** all 16 test-A batch roots are released. The π0.5 batches run on h100 after it is returned.

## D4 (2026-10-04 18:26 CDT, owner-approved): run replicate blocks rep0 and rep1 only
**Ruling:** the test-A grid is cut to rep0 and rep1. Both blocks cover the full dose curve (.25 / .32 / .40 / .45, both methods, plus off and pure controls) in all 8 cells. rep2 and rep3, which only add .32 / .40 precision, are dropped.

**Timing:** made at 18:26, before any test-A arm had finished (0 test summaries existed; the first arm, `r13_groot_l10_pure_c5_rep0`, was still running). No test-A outcome was seen.

**Reason:**
- On B-dev, the pooled distance − periodic SR difference was small: +0.9 pp, p = .24, across 2800 paired episodes.
- Even four blocks could not resolve a per-cell difference this small (frozen MDE 1.4–2.7 pp per cell). Two blocks save about 1.2 days for R14.
- The owner's framing: if the two methods cannot be told apart, that is itself the answer.

**Analysis consequences:**
- PLAN's primary estimand keeps its form, with replicate weights over the two blocks.
- The leave-one-block sensitivity becomes a two-block comparison.
- The precision/MDE statements must be recomputed for two blocks.
- The rep2/rep3 roots stay installed but unrun. They must not be run later in reaction to results.

## D5 (2026-10-04 20:38 CDT, owner directive): stop the test-A grid and run a lean confirmation set
**Owner:** the conclusion "with budget feedback, distance-triggered calls converge to the same cadence as periodic calls" is enough. It is to be shown in the paper; no more data is needed. Move on to R14 quickly.

**Ruling:**
- The rep0/rep1 grid is stopped. The running arm (`r13_groot_l10_50_periodic_ir25_rep0`) was aborted and is discarded.
- The test-A measurement is reduced to a fixed lean set, chosen by rule:
  - the four B-calibrated size-50 cells (GR00T and π0.5, L10-50 and Spatial-50);
  - the primary dose .32 only;
  - distance_feedback, periodic and off;
  - rep0.
  - This is 12 arms: GR00T on the local 4090, π0.5 on h100.
- **Already completed test-A arms:** the 7 arms finished before the stop are reported as supplementary, in full, with nothing selected:
  - GR00T L10-200: D and P at .40;
  - GR00T L10-500: D at .25 and .32, P at .25;
  - GR00T L10 pure C5;
  - one more L10 arm.
  - Two of their SR values were seen incidentally in run logs (GR00T L10-200 .40, D and P). They are outside the lean set and played no part in choosing it.
- **Dropped:**
  - the remaining `r13_devrec_*` arms (diagnostic only; dropped);
  - the size-200/500 test cells, the .25/.40/.45 test doses and rep1–rep3. Their roots stay installed but will not be run.

**The paper claim rests on:**
- B-dev pilot evidence (64 arms; cadence convergence, budget delivery, matched-budget SR indistinguishable);
- `MECHANISM.md`;
- the 12-arm test-A confirmation of cadence and SR at .32.
