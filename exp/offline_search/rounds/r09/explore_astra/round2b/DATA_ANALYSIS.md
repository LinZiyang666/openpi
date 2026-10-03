# Round 2b evidence and limitations

## Admission and split

Only the existing discovery-filtered compact files in `derived/r09_astra/compact/` are used. `tools/data.py` reads task/init members first, rejects any init outside 0–29 **before payload loading**, and applies an explicit admission filter. Raw controls are opened by an accepted discovery decision's episode ID, never by enumerating mixed-init episodes or calling `reader.episodes()`. Episode metadata must match the exact allowed task UID before controls are opened. Per-episode source stat digests are retained in `results/labels_*.json`.

Four long-task cells × pure-cache/uniform-plus-stall trajectories, plus independent-call trajectories for the two sparse cells: **10 populations, 3,000 episodes, 94,230 fresh observations**. Each population has 200 fit pairs (tasks 0–9 × inits 0–19) and 100 evaluation pairs (inits 20–29). Source counts and hashes are in `results/populations.json` and `PROVENANCE.json`. Historical P10 ledgers are also validated discovery-only and sliced to 20–29 before aggregation.

The main detector fits **only pure-cache and uniform-plus-stall paths from 0–19**. Independent-call paths do not train it. Nuisance models for randomized effects fit independent-call 0–19 only. The optional existing shared action head used in retrieval falsification also trained only on 0–19. No new fit, score threshold, call probability, or model selection uses 20–29 outcomes. These evaluation starts remain development data, not the coordinator's untouched confirmation set.

No `r09_astra_holdout*`, no 30–49 portion of any historical run, no other researcher's round-2 output, and no `explore_opus` output was opened. No task-specific switch was introduced.

## Labels and model

Control predicates are after-state. A decision's current label uses the last control **strictly before** that decision. The initial truth vector comes from the final settling control. A new goal is the first satisfaction of a predicate that was initially false; reattaining an initially true goal does not count as a new goal. Terminal success within a future horizon counts as forward progress even if it only restores a previously true predicate.

`late_stall`: at least one new goal has previously been reached, at least 60 controls have elapsed since the last new goal, and not all goals are currently true. `late_trap100`: `late_stall` plus no new goal/success within the next 100 controls. Training and discrimination metrics exclude unsuccessful episodes whose remaining horizon is shorter than 100 controls. **Intervention selection and randomized effect estimation do not filter on this future survival indicator.** Their progress outcome stops at the earlier of the horizon and episode termination.

The 57 serving inputs are elapsed count; weighted normalized demonstration phase and spread; kernel entropy/max weight/relative distance; ten-control action moments and gripper pattern; aperture norm and end-effector height; and lag-1/3/6 fresh-observation changes in both camera PCA vectors, state, retrieval phase/overlap, and proposed action. Camera inputs enter only through relative displacements. No predicate, task ID, init ID, teacher shadow, object pose, or future value enters serving features.

Weighted ridge/RFF fitting balances arm/episodes. Four OOF folds are grouped by `init % 4`, keeping the same start out of training across both arms. Fixed candidates: 0 or 128 RFFs, ridge alpha=100. Training OOF average precision chooses capacity. Training OOF F0.5 chooses a threshold quantile from .80/.90/.95. That quantile is applied to final **training** scores; all four select .90 and 128 RFFs. Model standardization and random features fit only within each training fold or final 0–19 fit.

| Cell | Uncensored fit anchors | Train OOF AP: linear / RFF | Final score threshold |
|---|---:|---:|---:|
| π0.5 long-50 | 11,906 | .587 / .649 | .4908529103 |
| π0.5 long-500 | 11,003 | .795 / .836 | .5069724917 |
| GR00T long-50 | 12,561 | .446 / .584 | .3425760984 |
| GR00T long-500 | 11,485 | .543 / .646 | .3629251599 |

These are regression scores, **not calibrated probabilities**. Models and their metadata were written before the evaluation metrics. `screen.log` records the four freeze lines before evaluation output.

## Disjoint evaluation: detection is useful, causal specificity is limited

Pure-cache paths, 100 episodes per cell. AP/AUROC use episode-balanced eligible anchors. Alert counts use every episode and the actual causal controller: warmup 60 controls, two consecutive high fresh scores.

| Cell | Factual SR | Target prevalence / AP / AUROC | Flagged failures / all failures | Flagged successes / all successes | First alert median control / remaining controls |
|---|---:|---|---:|---:|---:|
| π0.5 long-50 | .730 | .110 / .703 / .952 | 27/27 | 1/73 | 290 / 230 |
| π0.5 long-500 | .860 | .060 / .567 / .966 | 14/14 | 11/86 | 270 / 230 |
| GR00T long-50 | .620 | .108 / .701 / .933 | 33/38 | 10/62 | 290 / 170 |
| GR00T long-500 | .870 | .034 / .489 / .958 | 13/13 | 8/87 | 300 / 141 |

First-alert current `late_stall` precision is **60.7%, 64.0%, 55.8%, 57.1%**. First-alert `late_trap100` precision among uncensored alerts is **59.3%, 64.0%, 55.6%, 61.1%**, with denominators 27,25,36,18. Many alerts predict failure without proving a missed later transition. The apparent 100% failure coverage is based on only 13–27 failures in three cells and is not a population guarantee.

On held-out uniform-call paths, AP is .687/.510/.586/.575. On independent-call sparse paths, AP is .661/.564. This supports detection on some recovery-path states, but does not validate the new controller's induced states.

### Is this just elapsed time?

`ablation.py` fits clock-only and no-clock models on 0–19 and matches their **training-path** burst call counts to the full model. Evaluation counts can differ. It does not choose thresholds using evaluation outcomes.

| Cell | Full: flagged fail/success; median time | Clock-only: flagged fail/success; median time | No-clock: flagged fail/success; median time |
|---|---|---|---|
| π0.5 long-50 | 27/1; 290 | 27/11; 350 | 26/2; 290 |
| π0.5 long-500 | 14/11; 270 | 14/12; 280 | 14/10; 250 |
| GR00T long-50 | 33/10; 290 | 38/11; 310 | 28/10; 260 |
| GR00T long-500 | 13/8; 300 | 13/10; 290 | 12/10; 290 |

Observation history clearly helps sparse π0.5 over time alone. It has a much weaker case elsewhere. Do not promote all four cells as equally well-supported transition detectors.

## Randomized one-call effects

The independent-call population passes `eligible=True`, propensity=.25, and treatment=`coin<p` at every selected look. The alert is defined before the current coin. A separately fitted 0–19 outcome nuisance centers the Horvitz–Thompson contrast on evaluation first entries. Non-reaching episodes contribute zero. Bootstrap resamples the ten init IDs jointly over tasks (3,000 draws); all intervals are exploratory, unadjusted, and have few clusters.

Estimand: change **one** call at first alert, then retain the original independent-call controller. It is neither deterministic-gate SR nor sustained-burst value.

| Cell | Reached / treated alerts | Event within 100 controls: effect over all 100 episodes [interval] | Terminal success: effect [interval] |
|---|---:|---|---|
| π0.5 long-50 | 24 / 7 | +.013 [−.061,+.069] | +.038 [−.049,+.133] |
| GR00T long-50 | 37 / 12 | +.018 [−.060,+.096] | −.005 [−.122,+.095] |

In π0.5, neither treatment group had observed forward progress within 50 controls at these first alerts; one treated alert progresses within 100. In GR00T, 50-control progress occurred in 1/12 treated and 1/25 untreated alerts. This is weak, sparse evidence: a lack of observed events does not prove inability to recover. Centered and uncentered estimates, observed group means, and conditional-on-reach intervals are all retained in `causal.json`.

A privileged first-current-stall diagnostic reaches 84/74 episodes, many successful. Its π0.5 100-control progress contrast is +.117 [.022,.202], but terminal success is +.013 [−.097,.120]. GR00T terminal contrast is +.077 [−.010,.166]. The current-stall criterion therefore does not establish a universally useful call rule either. It is **not** an online candidate. No burst effect is inferred by multiplying any one-call estimate.

## Response cost and explicit SR sensitivity

Every look still commits ten controls. One full policy call replaces a cached look and adds .848 (π0.5) or .852 (GR00T) to the owner ledger. Divide summed additional cost by the original count of five-control decisions, including terminal rounding. This is fixed-path accounting, not closed-loop evaluation.

| Cell | Cache IR | Single / three-call / sustained IR | Random three-call IR | Three-call / sustained full calls per 100 episodes |
|---|---:|---|---:|---:|
| π0.5 long-50 | .07645 | .08702 / .10224 / .11681 | .09695 | 200 / 313 |
| π0.5 long-500 | .07669 | .08620 / .09994 / .11134 | .09424 | 159 / 237 |
| GR00T long-50 | .07434 | .08410 / .09958 / .12422 | .09244 | 212 / 419 |
| GR00T long-500 | .07450 | .08028 / .08917 / .10458 | .09450 | 99 / 203 |

Random burst probabilities are .0236478746, .0225287378, .0221313536, .0246333182. They match expected total calls to the learned three-call rule on **training** A/CU grids using exact finite-state dynamic programming. They are not equal-cost on evaluation paths; the table shows the mismatch explicitly. Do not claim an equal-IR win from these arms without accounting for realized cost.

`bounds.py` also implements first-intervention coupling. If a new controller exactly follows cache until its first alert, unalerted sample episodes cannot change outcome. The finite-sample SR bounds are [.72,1], [.75,1], [.52,.95], [.79,1]. These broad bounds assume exact same-path coupling and are **not confidence intervals**. Only sparse π0.5 has a small observed downside opportunity (one alerted success); sampling and encoder parity prevent a general safety guarantee.

Let q be the unknown fraction of alerted failures rescued, and h the unknown fraction of alerted successes harmed. Scenario SR = cache SR + q·flagged_fail/N − h·flagged_success/N. For h=.10:

| Cell | q=.25 | q=.50 | q=.75 |
|---|---:|---:|---:|
| π0.5 long-50 | .7965 | .8640 | .9315 |
| π0.5 long-500 | .8840 | .9190 | .9540 |
| GR00T long-50 | .6925 | .7750 | .8575 |
| GR00T long-500 | .8945 | .9270 | .9595 |

These are **assumption-driven scenarios, not fitted estimates**. Historical P10 on these 20–29 R8 paths is .870/.890 by model, IR≈.5035/.5038. Matching it in sparse GR00T would require rescuing about 85% of flagged failures at h=.10. The current π0.5 P10 screen instead measures .920: historical draws/topologies do not supply the new experiment's comparator.

## Cheap retrieval responses: mostly reject

Advance each neighbour by real consecutive library edges, never across demonstrations, then synthesize its future chunk. Also test gripper-mode reweighting and an entire coherent neighbour nearest the existing disjoint-fit shared-head prediction. Teacher shadows are used only for evaluation, never action selection. At first alerts:

| Cell | Advance 2 rows: relative motion MSE / gripper disagreement change | Gripper-mode relative MSE | Coherent chunk relative MSE |
|---|---|---:|---:|
| π0.5 long-50 | +13.8% / +7.86 pp | +4.5% | −8.0% |
| π0.5 long-500 | +46.5% / +6.40 pp | −18.9% | +7.5% |
| GR00T long-50 | −4.7% / −11.86 pp | −6.3% | −30.4% |
| GR00T long-500 | +290.6% / +13.33 pp | +10.1% | −0.1% |

Advances of 1 and 4 rows, all-anchor and privileged-stall subsets, episode-balanced errors, and clustered differences are retained. Sparse GR00T coherent chunks have a local signal, but the existing motion head improves first-alert MSE by roughly 52% and is already under closed-loop scrutiny. Another action-MSE-only candidate is lower value than testing sustained policy recovery. No retrieval candidate is emitted.

## New closed-loop calibration counterexample

Read-only snapshot of `r09_astra_round2_eval` at **2026-10-02 05:16:56 UTC**. Its manifest contains exactly the 100 pairs on 20–29; journal identity is extracted before outcome payload parsing. Only complete arms receive SR. The coordinator owns this run; no process, remote file, or configuration was touched.

| π0.5 long-50 arm | SR | Paired wins/losses vs cache | Exact two-sided p |
|---|---:|---:|---:|
| cache | .72 | — | — |
| shared motion | .63 | 7/16 | .0931 |
| shared motion + gripper | .63 | 11/20 | .1496 |
| gripper only | .74 | 3/1 | .625 |
| P10 | .92 | — | — |

The 0–19-fit motion head had improved held-out cache-path motion MSE by 37.5%, yet loses nine SR points as a point estimate. This does not prove the population loss is exactly nine points, or that all future correction heads fail. It does demonstrate that low one-step imitation error cannot license an SR claim. Current recovery arms inherit unmodified cache retrieval/actions and do not depend on this head.

## Verification and commands

`tools/reproduce.sh` runs the offline pipeline with CPUs 10–21,54–65 and OMP/OPENBLAS/MKL threads=1. `tools/live_results.py` is separate because its source is a changing coordinator run. No GPU or policy inference is required.

Thirteen unit tests cover admission-before-payload, split disjointness, before-state labels, initial goals, censoring, prefix-causal features, burst/cooldown/cap/hysteresis, exact random-budget expectation against exhaustive coin enumeration, first-entry selection, weighted ranking ties, journal filtering, coupling bounds, and frozen manifests/hashes. Real-library serving probes verify 16 candidates × 70 queries = **1,120 queries and tails**, including **164 full-policy tails**, exact score parity, cache action identity, one-use policy tails, recovery state persistence across plugin invalidation, and clone/reset isolation. Predictor median time is .038–.040 ms locally, excluding feature construction and remote end-to-end overhead.

Development fixes: the environment lacked sklearn, so ranking metrics were implemented and tie-tested without a new dependency. A test initially compared unused cache padding channels; it was corrected to the inherited seven valid action channels, while policy tails remain checked across every channel. Library phase is stored as float64 to match offline arithmetic exactly. Final logs retain passing verification.
