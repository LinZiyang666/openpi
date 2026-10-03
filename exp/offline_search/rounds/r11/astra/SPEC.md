# State-dependent inference knob: implementation specification

Status: frozen candidate algorithms. Numeric arm settings are in `freeze.json` and `arm_grid.csv`; these analysis modules are references, not plugin adapters. The only user setting is target owner IR. Every fitted threshold, CDF and predictor is shared across all tasks within one model × suite × library cell. Layer 4 accepts no task ID as a feature, lookup key, rate, threshold, or schedule selector.

## Execution contract

Call the unchanged R10Recipe query path at each **real vision anchor**. It retrieves and synthesizes the cache proposal, applies the distance-attenuated motion correction, and computes the existing no-progress guard. Compute the layer-4 score from the same retrieval. Never acquire an extra image just to compute the knob and never interrupt an eligible committed blind block.

Let `g` be the existing guard's force-miss decision and `k` the knob's sampled decision. Execute `g OR k`. A guard call consumes budget; there is never a second policy call when both fire. Preserve the guard's reason, diagnostic flags and history when `g` is true. On a knob-only call, set `os_force_miss=1`, reserve `os_reason=11` for the knob, and retain separate scalar diagnostics `r11_knob`, `r11_guard`, `r11_p`, `r11_score`, `r11_dose`. If another R11 adapter already reserves that reason, sol should use one shared R11 knob reason consistently; this is a logging identifier, not an algorithmic parameter. Keep `os_*` fields first so they survive the scalar cap.

A miss executes **one ordinary policy chunk, ten controls total**: five controls now, then the existing one-block lifecycle-approved policy tail. After that, resume normal cache retrieval. No extra policy segment, cooldown, gripper special case or task-specific branch is added. Step zero is eligible like every other real look, using the inherited early retrieval metric.

Do not reset progress memos, `_vision_progress`, `_noprog_span`, the corrector or guard tables, or any committed histories after a knob miss. `_s['flag']` remains the guard/event diagnostic; the outer knob does not forge a guard event. The existing `PolicyTailJudge.query` must have retained `_policy_gate_anchor` before the outgoing force-miss flag is changed. Let the normal plugin miss lifecycle invalidate the cache action anchor. The next policy tail uses the policy chunk, without motion correction. Never serve the discarded corrected cache tail after a policy miss. Preserve all existing invalid-lifecycle fallbacks, and account for any resulting actual look/call in the adaptive ledger.

Episode reset clears the knob RNG/counters/proposal memo and restores its calibrated initial dose. State belongs to the cloned connection and episode. A retried proposal at the same decision index returns the same draw and must not update the controller twice. `controller.py:MissController` is the task-free reference, including the committed-history ledger interface. Sol must synchronize it with **accepted** histories; a query proposal alone is not an executed policy call.

## Scores

All scores use only information already available at the current cache look. No shadow policy evaluation or future query observation is allowed.

**Distance:** nearest geometric distance in the inherited retrieval metric, with step zero's early metric and later anchors' stale metric. In the ordinary ten-control cadence, the previous executed block is a HIT (cache tail or policy tail), so the later metric is stale. Obtain the actual geometric distance from the base retrieval, not from the wrapper's kernel-transformed `res.scores`. This is a raw distance threshold; dividing both the score and threshold by the one cell-wide distance scale would be equivalent.

**Neighbour disagreement:** form the uncorrected top-16 weighted chunk `Abar = Σ w_j A_j`, with normalized float32 AWM weights, `kref=5` when the deployed subset has five episodes per task and `8` otherwise. Define

`D = Σ_j w_j mean_{t<10,c<6} ((A_j[t,c] − Abar[t,c]) / sigma[c])²`.

Here `sigma` is the inherited cell-wide action scale, fitted over the first five controls of the training library. Do not substitute an RMS score in a frozen threshold arm. Gripper and padding are excluded from this motion score.

**Predicted error:** one cell-wide regularized linear model predicts `log1p(pre-corrector motion MSE)` from the following scalar vector, with no task identity:

1. `log1p(nearest geometric distance)`.
2. `log1p(D)`.
3. `log1p(mean((Abar[:10,:6]/sigma[:6])²))`.
4. Weighted fraction of donor chunks with a gripper-sign transition anywhere within the first ten controls.
5. Mean over those controls of `4 p_t (1−p_t)`, where `p_t = Σ_j w_j 1[A_j[t,6]>=0]`.
6. Whether the synthesized uncorrected chunk changes gripper sign within those controls.
7. `log1p((d_16−d_1)/max(d_1,1e−6))`.
8. `1/(16 Σ_j w_j²)`.

These features are built from the **uncorrected** chunk. Using the corrected `res.action` for the energy or disagreement features changes the model. Reconstruct the small weighted chunk from the selected donors or capture it before correction; do not invoke mutating synthesis again. `experiment.py:retrieve` is the exact feature arithmetic. `data/<cell>/predictor.json` stores feature names, mean, std, coefficient vector and intercept-as-first-coefficient. Runtime score is `[1,clip((x−mean)/std,−8,8)] @ coef`. Exponentiation is unnecessary because ranking is monotone.

Gripper-transition and gripper-disagreement scores were evaluated but are **not standalone proposed arms**. Their retained contribution inside the pooled predictor does not introduce a gripper-specific miss rule.

## Static threshold and hybrid algorithms

Let `q` be the calibrated scalar dose. On the entire held-out library score table, choose a common threshold `t` and tie probability `h` so that the **unweighted all-row** mean of

`B_q(s) = 1[s>t] + h·1[s=t]`

equals `q`. Ties share one randomized probability; sorting order must not privilege a task or episode. `signals.py:threshold` implements this exactly, including zero and unit dose.

* `distance`: `p(s)=B_q(distance)`.
* `disagreement`: `p(s)=B_q(D)`; proposed only as the limited cheap-signal ablation.
* `error_hybrid`: `p(s)=q/2 + B_q(predicted_error)/2`.

Draw one Bernoulli at each real look, including guard looks for stable diagnostics, then OR with the guard. The hybrid has a nonzero random floor `q/2` whenever `q>0`. **It is not** an independent floor coin OR a deterministic threshold; that would produce a different probability. One equivalent implementation is to choose a uniform-rate arm with probability one half and the threshold arm otherwise.

The random floor makes the mechanism less dependent on ranking, but it does not guarantee owner IR when guard rates, episode lengths or score distributions change. Thresholds and rate expectations are calibrated with guard overlap included.

For the disabled baseline, bypass layer 4 entirely. At static dose zero the knob never calls, even for a score beyond the library maximum; at static dose one it always calls, even below the library minimum. These endpoint rules override the finite-threshold representation. None of the proposed static arms is at an endpoint. See `POST_FREEZE_NOTES.md` for the reference-code clarification added after the numerical prediction freeze.

## Adaptive error hybrid

Use the same pooled error score and half-random mixture. Retain the sorted held-out score reference from `signals.npz:predicted_error`. At each look convert the current score into its empirical rank `u`: let `l=#reference<s / n`, `h=#reference<=s / n`, then draw `u ~ Uniform(l,h)`; if the interval is degenerate, use its endpoint. This handles tied gripper-related feature patterns without task-specific tie breaking.

Start at the offline-calibrated `q0`. Just before each subsequent real-look proposal, synchronize the committed five-control ledger since the preceding proposal. Let `ΔN` be committed decision slots, `ΔV` vision calls and `ΔM` policy calls. These counts include guard calls, blind slots, policy tails and lifecycle-forced looks. Update

`q ← clip(q − 0.2 * (c_v ΔV + c_m ΔM − target_IR ΔN) / c_m, 0, 1)`.

Then use `p=q/2 + 1[u>1−q]/2`. Increasing measured cost raises the score threshold by decreasing `q`; underspending lowers it. The gain `0.2` and mixture one half are fixed across the whole grid, not selected by task. The controller is initialized independently at each episode. It uses no success, termination estimate, future length or policy-vs-cache error online. Total ledger cost, not the conditional knob-miss fraction, is its feedback quantity. Mandatory guards always take precedence. If they exceed the target, the knob can fall to zero but cannot enforce the target.

Use only the frozen size-50 adaptive arms at targets `.32` and `.40`. Lower-target large-library adaptations were explored and dropped after library simulation found infeasible initialization/overshoot. The adaptive method is not asserted to match IR exactly in a short episode.

For all methods, a nonfinite score conservatively requests a policy call and records a nonfinite diagnostic. It is not included in the nominal calibration claim. Never silently convert it to a cache hit. All actual fallback calls enter the ledger.

## Library-only fitting and calibration

`experiment.py`, `analyze.py`, `signals.py` and `controller.py` are the reference paths; the shared-accounting cross-check is `crosscheck.py`.

1. Open only the selected R10 B-pool `SubsetLibrary`. Retain failed episodes. Use the established subset ordering and assign each task's selected episodes by their permutation position modulo five. This partition is an offline validation device inherited from lower-layer fitting, not a task-indexed serving parameter.
2. Hold out complete episode groups. As in the frozen R10 distance calibration, retain the selected-library PCA basis, but fit the action scale and per-task lower-layer metric only on the outer training episodes. Query the held-out episodes against training donors only. The early metric also excludes all held-out episodes. At large sizes this is grouped whole-episode-out validation with a smaller donor bank, not exact single-episode LOEO.
3. Reuse R10's matching whole-episode-held-out corrector residual terms by parent row ID and assert regenerated distance and uncorrected error agreement. The reported risk is normalized corrected motion MSE: `m0 − 2 a yp + a² pp`, where `a=.5*clip((2−d/scale)/1.25,0,1)` and `a=0` at step zero. Its scale and corrector heads exclude the corresponding validation episode. This is a policy-action difference proxy, not success or recovery utility.
4. For a learned score, hold outer validation episodes out of the error predictor too. On the remaining four groups, hold each group out in turn from metric fitting and donors; generate features and **pre-corrector** motion-MSE labels. No outer validation episode enters any of these inner fits. Fit the one pooled predictor with equal episode mass, weighted feature mean/std (std floor `.05`), z clipping `8`, and ridge penalty `.1*n_rows` on every slope but not the intercept. Predict outer held-out scores. Training on pre-corrector labels avoids another nested corrector fit; evaluation still uses the corrected-cache residual.
5. Fit the deployable pooled predictor once on all outer held-out features and their pre-corrector labels. Retain the outer cross-fitted score reference for thresholds/CDFs. The final head's score distribution on held-out feature tables was audited in `crosscheck.json`; this is a transfer diagnostic, not independent value evidence.
6. Preserve each complete held-out episode sequence. For the ordinary cadence, use vision anchors at steps `0,2,4,...`. At each anchor reconstruct the frozen no-progress span using the retrieved donor's normalized library progress and its episode length: non-advance iff `(prog_now−prog_previous)*(eplen_now−1) <= .5`; add the look-step gap when non-advancing, otherwise clear the span. Guard fires at span `>=2`. Preserve the first-anchor no-guard case. No task-specific guard retuning occurs.
7. Set `N=Σ episode_length`, `V=Σ ceil(episode_length/2)`, and expected `M(q)=Σ_anchor [g+(1−g)p_q(s)]`. The owner ledger is `(c_v V+c_m M)/N`, with `(.152,.848)` for pi05 and `(.148,.852)` for GR00T. Short terminal episodes can make the all-call endpoint slightly exceed `.5`; do not conceal this truncation effect. With regular cadence, a positive stall span at a look is already large enough to force a policy call, so the cache-tail veto creates no extra look. `signals.py:dag` also verifies the general span/cadence branches; it exactly agrees with opus's simpler shared formula on all reachable nominal states.
8. For each static target, solve this library-only expected-IR equation by thirty bisection steps over `q∈[0,1]`, with tie interpolation. Check floor/ceiling feasibility. Save the cell-wide `q,t,h,beta` from `analysis.json`. Never substitute the nominal no-guard formula or add guard costs twice.
9. For adaptive targets, replay the actual controller on exogenous held-out episode sequences with thirty-two independent draws, common seed `20261002`. Solve only the initial `q0` with sixteen bisection steps, comparing endpoints and every tested candidate by absolute mean IR error. The fixed feedback gain is not fitted. Validate sampling sensitivity using one hundred twenty-eight draws and seed `20261003`. An infeasible target is marked infeasible and excluded, not relabeled as a successful match.

All-row weights in quantile fitting and pooled decision counts in cost fitting are intentional. The primary **value** tables use equal episode weights at even-step anchors; those evidence weights do not replace the cost ledger's pooled counts. The matched-IR added-risk table separately measures additional capture after guard calls have been removed.

## What the offline mapping cannot identify

The episode state path, success, length, guard events and future score distribution change when cache or policy actions change. Pure-policy B records cannot identify those counterfactual trajectories. The ordinary-cadence accounting is exact **conditional on its input sequences**, not a calibrated closed-loop success model. Fixed selected-library PCA, reduced fold donor banks, score-model transfer and stochastic policy outputs add uncertainty. Bootstrap intervals describe episode variability on the reused B evidence and do not remove method-selection optimism.

`freeze.json` contains explicit rank-shift and persistent-stall scenario predictions. They are not statistical coverage bounds. Keep the knob-off arm and matched-IR random comparator; do not infer positive SR utility from the captured error alone. No optional non-test recordings were used.
