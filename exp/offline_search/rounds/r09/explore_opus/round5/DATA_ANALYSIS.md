# Round 5 (opus) — data analysis: wasted guard calls, two gates, residual failures

Lane: IR reduction of fable's leading r3c stacks (`r9f3c_pi05_l10_50_np_corr05_esc` .920 @ .162,
`r9f3c_groot_l10_50_np_corr05` .890 @ .183). Rules kept: inits 30-49 never read (every ledger is admitted through
`iter_jsonl_discovery`, which drops init >= 30 before decoding; holdout roots refused); thresholds fitted on inits 0-19;
inits 20-29 used only to describe the stacks and to cross-check predictions; nothing is task-indexed; nothing launched;
CPUs 2-9,46-53.

Data (`tools/ledger5.py`, per-decision ledgers with the guard's own extras, `DERIVED/r5ledger/`):
fit inits 0-19 — only-no-progress (1 run), B (4 runs; 85-90 % of B's calls are no-progress calls), pure cache
(5 runs) per model, 200 episodes each; screen inits 20-29 — the two leading stacks, their "+ empty grasp" sibling runs,
only-no-progress ± escalation (r9o), corrector-only (2 runs), caches (3 runs), pure policy (5 runs π0.5 / 4 GR00T).
The no-progress statistic is replayed exactly from the logged top-1 rows (0 mismatches against the logged span on
19 019 looks), which also gives *virtual* firings on the pure-cache runs (where the guard would have fired).

## 1. Which guard calls are wasted?

### 1.1 Where the stacks' IR goes (inits 20-29)
| | π0.5 stack | GR00T stack |
|---|---|---|
| looks (cache's own, .152/.148 each) | 47 % of cost | 41 % |
| calls | 53 % (5.5/ep: 3.34 no-progress + 2.16 escalation) | 59 % (7.24/ep, all no-progress) |
| calls spent in the 8 / 11 failed episodes | **36.5 %** of all calls (25.1 per failed ep) | **43.9 %** (28.9 per failed ep) |
| calls per successful episode | 3.8 | 4.6 |
No look is caused by the no-progress look veto in the stacks (looks are budget or lifecycle looks), so only calls are
reducible in this lane; a 25 % IR cut means removing ~47 % (π0.5) / ~42 % (GR00T) of the call cost.

### 1.2 Attributes of the no-progress calls (fractions of no-progress calls)
| | π0.5 stack | GR00T stack | π0.5 fit (only-np / B, 0-19) | GR00T fit |
|---|---|---|---|---|
| on pace (lag <= 2) | **.72** | **.43** | .41-.47 | .28-.33 |
| on pace (lag <= 4) | .84 | .52 | .47-.53 | .34-.40 |
| repeated inside the same unresolved stall | .20 | .39 | .38-.43 | .40-.45 |
| nothing changed after it (no progress at the next look) | .23 | .39 | .38-.43 | .40-.45 |
| robot moved < .02 (normalized) per decision since the previous look | .23 | .36 | | |
| in an episode that failed anyway | .16 | .44 | .45-.49 | .60-.67 |
| in a pair the pure cache always solves (7-10 runs) and that succeeded | .47 | .22 | .28-.29 | .15-.19 |
| ex-post wasted (failed, or cache-always-solves) | .63 | .66 | .72-.78 | .79-.82 |
"Ex-post wasted" is an upper bound (outcome is not observable online). The online-usable question is which of these a
rule can drop without losing successes; two classes answer it:

**(a) On-pace calls do not end stalls.** At the first firing of a stall, the probability that the next look shows
progress again, real call vs the pure cache doing nothing (fit inits 0-19, pooled over both models and replicates;
`out/fit_thresholds.log`):

| lag at firing | -3 | -2 | -1 | 0 | 1 | 2 | **3** | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| after a call | .854 | .787 | .915 | .901 | .866 | .837 | .778 | .781 | .686 | .640 | .701 |
| cache alone | .872 | .879 | .890 | .913 | .837 | .822 | .702 | .628 | .606 | .496 | .531 |
| difference | -.02 | -.09 | +.03 | -.01 | +.03 | +.01 | **+.08** | +.15 | +.08 | +.14 | +.17 |
Up to two decisions behind the demo's pace the cache gets out of the stall on its own as often as with a call
(≈ 85-90 %); from three decisions behind, calls help. Most on-pace "stalls" are retrieval jitter: the top-1 row
switches between demos whose normalized progress differs by less than half a library step while the robot keeps
moving (motion at firings has the same distribution as at all looks).

**Caveat found on the way (π0.5 only):** on-pace calls are not entirely useless downstream. In the fit runs, an
episode whose first stall is on pace later reaches an off-pace stall (lag > 2) in 51.3 % of pure-cache episodes vs
45.8 % with calls (GR00T 68.2 % vs 64.6 %): the policy's 10 controls sometimes leave the robot in a better state. The
prefix-exact simulation (§2.3) turns this into a predicted π0.5 loss of ~2.6 pp for gate P; GR00T shows none.

**(b) The latch: calls after ~20 per episode are almost all in lost episodes.** Fit inits 0-19 (only-no-progress run):
| calls per episode > k | 10 | 12 | 15 | 20 | 25 |
|---|---|---|---|---|---|
| π0.5 episodes / success rate | 37 / .27 | 30 / .20 | 22 / .05 | 13 / .00 | 10 / .00 |
| GR00T episodes / success rate | 66 / .14 | 63 / .10 | 59 / .05 | 51 / .02 | 30 / .03 |
On the stacks the same picture: GR00T's 11 episodes with > 20 calls all failed (44 % of its calls); π0.5's escalated
failures carry 149 of its 216 escalation calls. These calls are "repeated in the same state, after which nothing
changes": the retrieved demo progress does not move again (lag grows by exactly 2 per fresh decision).

## 2. Two gates (task-agnostic; `methods.py`)

- **Gate P, on-pace silence (L0):** a no-progress verdict at a look with pace lag <= L0 is dropped; the decision is a
  cache HIT and the no-progress look veto is lifted for the next decision, so the controller is exactly the cache until
  its next regular look. The span statistic keeps counting; a persisting stall is called at the next look if its lag
  then exceeds L0. Escalation and grasp verdicts are never touched by P.
- **Gate C, call budget (C):** at most C guard calls of any kind per episode (counted from the committed history:
  vision decisions that were MISSes); afterwards every verdict is dropped and the veto lifted (pure corrected cache).
- Implemented as subclasses of fable's frozen r3c classes (`GatedNpGraspEsc3`, `GatedNpGraspStackGroot3`); the gate
  post-processes the final verdict; with both gates off the verdicts are identical (unit test + selftest comparison).

### 2.1 Thresholds (inits 0-19 only; rules stated in `tools/fit_thresholds.py` before applying them)
- L0 = largest lag such that the call-minus-cache recovery difference stays below 5 pp for every lag from -3 up to L0
  (pooled models) -> **L0 = 2** (lag 3: +7.6 pp).
- C = smallest budget on {10, 12, 15, 20, 25} with simulated loss <= 1 pp in both models -> **C = 20**.

### 2.2 Prefix-exact simulation (fit inits 0-19, `tools/gatesim.py`, pair bootstrap 300 draws, 90 % intervals)
Gate P: the gated controller is the pure cache until its first firing with lag > L0, so each pure-cache episode is
kept exactly until then; the suffix success is the judge runs' success after their first call with lag > L0 at the
same (decision, lag) bin. Calibration: L0 = -inf reproduces the judge (π0.5 .861 vs real .855; GR00T .726 vs .729).
| L0 | -1 | 0 | 1 | **2** | 3 | 4 |
|---|---|---|---|---|---|---|
| π0.5 Δ SR (pp) | -0.2 [-1.1,+0.5] | -1.9 [-3.3,-0.7] | -2.0 [-3.8,-0.6] | **-2.6 [-4.7,-1.0]** | -1.3 [-3.3,+0.2] | -1.3 [-3.7,+0.6] |
| GR00T Δ SR (pp) | +0.1 [-0.8,+1.4] | +0.3 [-0.9,+1.3] | +0.6 [-1.0,+2.2] | **+0.6 [-1.4,+2.7]** | -0.2 [-2.5,+2.0] | +0.3 [-1.7,+2.4] |
Gate C: the judge episode is kept exactly up to its (C+1)-th call; a success after that point is replaced by the pure
cache's success after a stall at the same (decision, span) bin; failures are never credited (conservative).
| C | 10 | 12 | 15 | **20** | 25 |
|---|---|---|---|---|---|
| π0.5 Δ SR (pp) | -4.1 | -2.3 | -1.3 | **-0.8 [-1.2,-0.4]** | -0.1 |
| GR00T Δ SR (pp) | -5.6 | -3.5 | -1.8 | **-0.4 [-0.8,-0.1]** | -0.3 |

### 2.3 Prediction for the stacks (inits 20-29; `out/predict.json`; PREDICTION.md written 04:18 CDT before emit)
IR from each stack's own ledger: a dropped on-pace call saves the policy price with the cache's own recovery
probability at that lag (otherwise the call returns one look later); then the budget.
| | π0.5 IR | cut | GR00T IR | cut |
|---|---|---|---|---|
| stack | .162 | | .183 | |
| + P2 | .135 | 16 % | .146 | 20 % |
| + P2 + C20 | .126 | **22 %** | .136 | **26 %** |
SR: fit-sim deltas above; stack-specific cross-check on 20-29 (corrector-only runs as the gated prefix, the stack's own
episodes as the suffix; calibration π0.5 .941 vs real .920, GR00T .890 vs .890): π0.5 P2 -0.6 pp, budget -1.5 pp;
GR00T P2 +1.3 pp, budget 0. Summary: **GR00T reaches the 25 % target at no predicted loss; π0.5 reaches ~22 % at a
predicted loss of ~2-3 pp** (the escalation's rescues need long takeovers, and π0.5's on-pace calls have the small
downstream benefit described in §1.2). A π0.5 25 % cut is available with C = 15 (IR .121) but its predicted loss is ~2-5 pp,
so it is not emitted.

### 2.4 Correctness evidence
- `tests/test_round5.py` (11 pass): gate P drops only on-pace no-progress verdicts; gate C counts committed calls of
  any reason; veto lifted only right after a gated look (or after the budget) and the span statistic is restored;
  gates-off output identical to the parent; quiet state does not leak across episodes; forced verdicts; bad thresholds
  rejected; the real fitted gated stacks equal fable's r3c artifacts field by field (judge burst 0, CorrectedCacheJ
  blend .5, same library steps and actions, same guard / escalation constants); attribute-clash detection.
- CPU plugin selftests (`tools/selftests.py`, `r09_opus_r5/selftest/`, `out/selftest_summary.json`), real judge chain,
  `--blind --policy-tail --judge guard_only` (+ `--policy-tail-blocks 1` GR00T), 4 episodes / 48 decisions each, all
  12 PASS: gates-off vs fable's artifact **0 of 48 decisions differ** (both models); forced no-progress verdicts with
  L0 = 1000: 20/20 gated, 0 calls, every gated look followed by a blind decision (veto lifted); forced verdicts with
  C = 1: at most 1 call per episode, 16 budget-gated decisions, all followed by blind decisions; production P2 / P2C20:
  the stacks' natural verdicts (2 π0.5, 4 GR00T, all on pace) gated, PASS.

## 3. Residual failure anatomy (inits 20-29; `tools/anatomy5.py`, `out/anatomy5.log`)
| | π0.5 stack | GR00T stack |
|---|---|---|
| failures | 8 | 11 |
| of which the pure policy solves (>= half of 5 / 4 runs) | **7** | **10** |
| type: retrieved demo played to its end (peak progress >= .95), episode still times out | 7 | 8 |
| type: stuck in the second half of the demo (peak .84-.89) | 1 | 3 |
| policy in control at the end (escalated, or >= 5 calls in the last 20 decisions) | 8 / 8 (6 escalated) | 11 / 11 |
| last progress of the retrieved demo at decision | 38-86 (median 46) | 40-66 (median 44) |
| calls in those episodes | 11-41 | 24-35 |
| same pair also fails in the sibling stack run | 4 of 8 | 6 of 11 |
| pure-cache success on those pairs (10 runs) | 0-1.0, median .55 | 0-1.0, median .4 |
Kind of failure: **script exhaustion**. The cache follows the retrieved demonstration to its last frame by about
decision 45 without the task being done (an object not where the demo left it); from then on the retrieved progress
never moves again, the no-progress guard fires every other decision (GR00T) or the escalation hands the policy full
control (π0.5), and the policy — which solves these initial states from scratch — does not recover in the remaining
~55 decisions. Half of these pairs are knife-edge (the sibling run succeeds); the other half fail in both runs.
These are exactly the episodes whose calls gate C cuts after the 20th call. Earlier detection of "demo finished
without success" would be the lever for SR (outside this lane: takeover rules).

## 4. Caveats
- The SR predictions are simulations with matched bins, not counterfactual replays; the screen (100 pairs) cannot
  resolve 2-3 pp. IR predictions are much tighter (calls are counted directly).
- The fit data have no corrector and no escalation; the stack-specific cross-check uses inits 20-29 (descriptive only).
- The stacks' screen SRs (.920 / .890) were single runs; their siblings scored .880 / .880, so the same-batch control
  arm in r09_opus_r5 is the reference, not the r3c number.

## 5. Screen outcome (added in round 6; coordinator's timan108 screen, inits 20-29, `tools/readout.py`)
| arm | π0.5 SR @ IR | paired vs stack | GR00T SR @ IR | paired vs stack |
|---|---|---|---|---|
| stack (same batch) | .900 @ .184 | | .870 @ .187 | |
| + P2 | .860 @ .166 | +6 / −10 | .830 @ .153 | +5 / −9 |
| + P2C20 | .870 @ .137 | +4 / −7 | .820 @ .141 | +5 / −10 |
Mechanism as designed (2.4 / 3.3 on-pace verdicts gated per episode, no no-progress call left at lag <= 2, budget
binding 1.3 / 0.9 decisions per episode), IR cuts as predicted for P+C (−26 % / −25 %), but SR fell ~4 pp in all four
gated arms, worse than predicted (π0.5 −0.6…−2.6, GR00T ≈ 0). The π0.5 logs show the reason: with on-pace calls
gated, escalation calls rose from 4.2 to 5.2 per episode — more episodes fell 12 decisions behind. This is the
"protective" downstream effect of on-pace calls (§1.2 caveat), larger in the stacks than the fit simulation
estimated, and present in GR00T too. Conclusion: drop gate P; the budget alone (P2C20 vs P2: +1 / −1 pp) cut a
further .029 / .012 IR at no visible cost and is the only part worth keeping.
