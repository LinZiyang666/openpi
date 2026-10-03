# Frozen round-2b proposals

**Status: executable, locally verified, no round-2b closed-loop launch.** The freeze is [FROZEN_CANDIDATES.json](FROZEN_CANDIDATES.json); exact emitter input is [confirmation_specs.json](confirmation_specs.json). Sixteen candidate artifacts plus six shared controls produce 22 arms. Every arm has exactly 100 task/init pairs, **inits 20–29 only**, in `/home/weiland/trace_runs/os_closed_loop/r09_astra_round2b`.

All fitted components and budget calibration use **0–19 only**. No new parameter is indexed by task, no task hardness is supplied, and no goal predicate is used online. The existing task-conditioned retrieval metric is unchanged. The detector needs additional offline supervision from 400 arm/episodes per cell (200 distinct task/init starts across two controllers); this is not a method learned from the 50 or 500 library demonstrations alone.

No other researcher's round-2 output was read. The current candidates use original cache actions and **do not apply the round-2 residual correction**.

## 1. Bounded policy takeover until observations indicate recovery

**First priority: π0.5 long-50.** Highest expected gain × confidence in this package: the selector flags 27/27 historical evaluation failures and only 1/73 successes, and beats a training-budget-matched clock by roughly 60 controls and ten fewer false-positive episodes. Confidence is moderately high for useful detection on this population, low for recoverability, and very low for preserving P10 SR. GR00T long-50 is the second mechanistic test, with lower confidence and a much larger gap to close. Dense cells rank below both sparse screens because collateral-call harm and weak advantage over elapsed time remain serious concerns.

**Frozen detector.** A 57-input ridge/RFF head shared over ten tasks in a cell. Fit cache and uniform-plus-stall 0–19 trajectories; four init-group folds choose linear vs 128 RFFs and one threshold quantile. The fixed grid chooses 128 RFFs, alpha=100, and the final-training .90 quantile in every cell. It predicts the privileged later-stall/no-progress100 label described in `DATA_ANALYSIS.md`. Neither the label nor the policy shadow is a serving input.

**Frozen response (`_latch`).**

1. Preserve cache retrieval and all ten action controls. Fresh looks occur every ten controls.
2. After at least 60 controls, two consecutive fresh scores at or above the frozen threshold start policy takeover.
3. Every full-policy result commits ten controls, including the exact original policy tail.
4. Make at least three calls. Thereafter return to cache when **two consecutive fresh scores are below half the entry threshold**, or when the episode reaches twelve total full-policy calls.
5. After an early return, require six free fresh anchors of cooldown and two fresh high scores before a new allowed takeover. The twelve-call cap is per episode, across all takeovers. Reset on episode identity change.

There is no extra camera look, gripper override, task router, oracle input, or changed action averaging. The score is not a probability. A lower score may indicate resumed movement without completing the next goal; the exit rule deliberately remains a hypothesis to test.

| Cell | Historical cache SR | Sustained IR, fixed-cache-path estimate | Candidate SR | Illustrative SR if 25% / 50% of flagged failures recover¹ |
|---|---:|---:|---|---:|
| π0.5 long-50 | .730 | .11681 | Unidentified | .7965 / .8640 |
| π0.5 long-500 | .860 | .11134 | Unidentified | .8840 / .9190 |
| GR00T long-50 | .620 | .12422 | Unidentified | .6925 / .7750 |
| GR00T long-500 | .870 | .10458 | Unidentified | .8945 / .9270 |

¹ Assumes 10% of flagged successes are harmed. These are explicitly conditional sensitivity calculations, **not empirical recovery rates or SR forecasts**. Changing paths and episode lengths can change IR as well. For current topology/draws, use the newly measured paired controls, not these R8 point estimates.

**Why sustained control is worth testing:** the learned first-alert one-call effect is not established in randomized data. Long failures can survive a short action improvement and then return to the same cache attractor. Several feedback steps may carry a later transition through. The offline data do not identify that mechanism or establish a minimum necessary burst length.

**Failure modes:** intervention after irreversible damage; false-positive calls on a healthy dense-library path; a score that falls as soon as the arm moves but before subgoal completion; persistence of a wrong policy mode; reaching the twelve-call cap while still stalled; fresh-state distribution shift; mismatch between deferred shadow keys and live keys. Exactly zero score discrepancy in library-query tests does not prove live encoder parity.

**Development objective:** record the full paired comparison, whether positive or negative. Look for a clear SR increase over same-run cache at materially lower cost than P10, then examine whether longer takeover adds wins relative to single/three-call variants. A +5 pp gain is a useful screening target, **not a forecast or a proof threshold**. Even an observed +5 pp may be inconclusive with 100 pairs.

## 2. Three consecutive calls at the same alert, with randomized placement control

**Priority:** run in the same fixed batch as sustained takeover. Lower expected cost and smaller intervention, but possibly insufficient time to complete a transition. It is also the cleanest way to test whether observation-dependent placement helps at approximately matched training budget.

`_burst3`: same entry detector, three consecutive full-policy calls, six free fresh-anchor cooldown, two consecutive high scores to reenter, twelve calls total. `_random3`: after the same warmup, a fixed independent coin starts an otherwise identical three-call burst; its probability is calibrated on 0–19 A/CU observation grids to the learned rule's expected number of calls using an exact finite-state calculation.

| Cell | Three-call IR | Random three-call IR | Frozen start probability |
|---|---:|---:|---:|
| π0.5 long-50 | .10224 | .09695 | .0236478746 |
| π0.5 long-500 | .09994 | .09424 | .0225287378 |
| GR00T long-50 | .09958 | .09244 | .0221313536 |
| GR00T long-500 | .08917 | .09450 | .0246333182 |

These are historical-path costs, and **evaluation budgets do not match exactly**. Report actual SR and IR together; do not call a higher-cost SR improvement an equal-cost win. Randomization uses a fixed seed/domain and identity only to generate independent episode/decision coins, never to set task-specific probabilities. Candidate SR is unidentified; the sensitivity table above describes the common first-alert opportunity set, not equal expected performance of the responses.

Failure modes include handing control back too early, reentering a loop during cooldown, and apparent gains that are explained by cost rather than placement. The response cap prevents unbounded escalation but may itself block recovery.

## 3. Single-call control

`_burst1` retains the same alert and six-anchor cooldown but calls only once per trigger. Expected path IR is **.08702/.08620/.08410/.08028** in the table's cell order. It is a mechanistic control and possible lower-cost option if sustained takeover causes harm, not a promoted success-preserving solution. Randomized first-alert intervals already span both signs, so confidence in a large SR gain is low.

## Exact frozen batches

Run the entire first batch when the assigned fleet is idle; do not stop after a favorable arm:

```text
r9r2b_astra_pi05_l10_50_cache
r9r2b_astra_pi05_l10_50_burst1
r9r2b_astra_pi05_l10_50_burst3
r9r2b_astra_pi05_l10_50_latch
r9r2b_astra_pi05_l10_50_random3
r9r2b_astra_pi05_l10_P10
```

Second fixed mechanistic batch: replace `pi05` by `groot` in those six names. These two batches are recommended together to test portability, not to select a favorable policy family after seeing results.

Lower-priority dense batches have the same five library-specific names with `_500_` replacing `_50_`; reuse each model's same-run P10 comparator. The emitted availability of these arms is not a request to sweep task-specific thresholds. Sparse and dense models share one recipe, while coefficients reflect their different observation/retrieval populations.

All 22 exact names are in `confirmation_specs.json`. Each config explicitly references `manifests/eval100.json`; **never replace it with a 300/500-pair manifest**. Use timan108, at most four H100 servers on ports 23230–23233, at most 32 workers, sync port 23196, one chain at a time. This turn launches nothing; commands are handed back in `HANDBACK.md`.

## Rejected or deferred responses

- Advancing neighbour demonstrations by two rows worsens first-alert motion MSE by 13.8%, 46.5%, and 290.6% in three cells. Gripper disagreement also worsens there. Do not deploy a generic forward jump.
- Gripper-mode filtering and coherent-neighbour selection are inconsistent across cells. Sparse GR00T has a local signal, but action-MSE-only evidence is particularly weak after the newly completed correction failure. No arm is emitted.
- An oracle current-stall trigger and unconditional elapsed-time takeover are diagnostics, not promoted methods. Current-stall calls have no robust terminal-success evidence; clock-only detection is less selective in the strongest cell.
- No task-indexed call budget, routing, correction strength, or library choice is proposed. No Spatial extension is fitted or forecast from these long-task results.

## Final confirmation protocol

The 20–29 screens are development evaluation and cannot establish a simultaneous two-percentage-point noninferiority claim. Report complete paired wins/losses versus cache and P10, pooled owner cost/decision sums, call counts, episode lengths, actual ten-control tails, and query latency. Include every fixed response and all negative results.

Before any coordinator-only locked confirmation, choose and freeze the final candidate family and its number K of claims. The preferred single claim, if the development evidence warrants it, is π0.5 long-50 `_latch`; the current proposal does **not** authorize us to open locked data. For paired candidate/P10 outcomes use `../round2/tools/statistics.py:paired_ni`, margin=.02, family=K, alpha=.05, and require its simultaneous lower bound >−.02 plus lower measured IR. Inconclusive remains inconclusive. Do not refit on 20–29 and then report those same starts as independent validation.
