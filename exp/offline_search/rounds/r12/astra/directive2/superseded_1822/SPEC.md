# DIRECTIVE_2 implementation specification

Status: implementation-ready offline reference and B-pilot proposal, not a deployed plugin or established matched-SR saving. This document supersedes the old primary and its latency accounting. The old specification is preserved in `directive2/prior_delivery/SPEC.md`.

## Method and fit boundary

Use the unchanged full R10 recipe, only-no-progress guard, corrected ten-control commit, and off/periodic R11 knob. The primary is a **wrist-first completion gate with two independently constructed action/progress witnesses**. Every planned look still obtains current vision; there is no extra blind block. All PCA, task metrics, action/state normalization, correction heads and distance scales are fitted from exactly the selected B library, including failed episodes. Task-indexed fits belong to layer 5. Camera gate thresholds are global across tasks, models and sizes.

The reference is `d2_consensus.candidate` plus `d2_screen.gate_mask` and the state-machine equations in `d2_screen.replay_gate`. The replay is an analysis of fixed recorded states; its reads of the recorded full answer are fallback/evaluation inputs, not a substitute for actual acquisition in a serving implementation. A plugin adapter still has to implement the transaction below.

### Library-only ingredients

Keep the selected-library 64-coordinate PCA for each camera and the unchanged full metric/head. Let `p(i)` be the library predecessor exactly two five-control slots earlier in the same episode, clamped at the initial row. Define:

```
F(i) = [P0(i), P1(i), rs(i)[:8]]
H(i) = [P0(p(i)), P1(i), rs(i)[:8], rs(i)[:8] - rs(p(i))[:8]]
W(i) = [P1(i), rs(i)[:8], rs(i)[:8] - rs(p(i))[:8]]
```

Fit independent action-weighted Mahalanobis metrics for H and W per task, using the existing nearest-three other-episode action construction, first five controls/seven standardized action channels, ridge `0.1*trace(Sw)/d`, full-rank Cholesky, stable top-16 retrieval and the original kernel (kref 5 at size 50; 8 otherwise). Candidate payloads retain their current actions/progress and parent row mapping. The last real P0 is memory, not a fresh observation.

Fit H correction heads with selected-library LOEO synthesis, query steps at least two, residual targets on ten controls/six motion channels in selected-library sigma units. Features are H, standardized synthesized ten-control/seven-channel action, `min(step,120)/120`, and the task indicator. Use R10 episode-balanced weights, 384 RFF, seed zero, ridge alpha 100. Leave gripper/padding unchanged. Fit H distance scale with the five-episode-fold selected-library procedure, training-fold metric/sigma, and episode-balanced median noninitial held-out nearest distance. Correction is `.5 * clip((2-d/scale)/1.25,0,1)` times the residual in original action units. `dev_validation.fit_half` implements this scale and head completion; its artifacts now exist for every recorded cell.

Fit W heads identically, with W inputs. W is only a disagreement sensor, never the served action. Its scale is the selected-library training-LOEO median, as implemented in `d2_consensus.fit_wrist`. This sensor-specific calibration is intentional; it is not a claim of production five-fold W correction parity. Changing it requires a fresh B screen.

Retain the original library safety atlas: LOEO raw F/H synthesis at i and raw F at p(i); require standardized motion RMS ≤.15, gripper-sign identity on every committed control, and agreement of the progress-advancement Boolean. Progress advancement is `(p_now-p_previous)*max(ep_len_now-1,1) > .5`. Early rows before step four are unsafe. Save exact same-episode two-slot successors and selected-library state standard deviation, floored at `1e-6`. `final_atlas/*/table.npz` contains these tables and main metric references. Fixed selected-library PCA is transductive across library folds; no B-dev observations enter these fits.

### Before encoding

Maintain connection-private memory of the most recent **genuine full** look: both projected keys, state, step, actual full donor rows/weights, task/episode identity and accepted execution metadata. Reset on identity changes, discontinuity, failed execution, retry ambiguity or missing fields. Do not renew this memory on an accepted half look.

Force full on step <4, memory age other than exactly two slots, a positive previous no-progress span, a known periodic due call, or invalid lifecycle. The narrow valid-policy-tail exception requires the prior full policy look exactly two slots earlier, one successfully executed policy-tail block, both exact five-control execution acknowledgments, no pending span, and unchanged identity. Do not waive generic lifecycle errors.

Advance saved full donors by two library rows. Require successor support mass ≥.99, safe successor mass ≥.75, and RMS error between measured state displacement and weighted expected donor displacement ≤.2 selected-library state-SD units. Missing successors contribute zero support; do not renormalize them into safety. Rejections here choose full immediately. These checks consume no current image features.

For periodic calls, compute the cap from accepted history **before acquisition**: last actual call step, count of look anchors since that call, the saved R11 setting, and `uniform('R11-gap-v1',0,task,init,last_call,'cap')`. Use floor plus the fractional Bernoulli cap. Do not use the recorded fired flag, which is suppressed when the guard already calls. Peeking must not consume or mutate the scheduler. Distance-threshold knobs remain outside this first pilot because full/H distances need separate calibration maps.

### After encoding the wrist, before any commit

1. Construct H from the saved genuine P0, current wrist, current state and measured displacement. Retrieve/correct the proposed half action. Its top row is the proposed guard-progress source.
2. Retrieve/correct W from current wrist, state and displacement. This is the first witness.
3. For H donors `r_k` and weights `w_k`, estimate the missing-view innovation:

```
delta0 = sum_k w_k * (P0(r_k) - P0(p(r_k)))
P0_est = P0_saved + delta0
```

Retrieve with the unchanged full metric on `[P0_est,P1_now,rs_now]` and apply the unchanged full head to these explicitly imputed features. This is the second witness. It is an estimate used only for acceptance; it does not become a true full observation or overwrite the camera memory.
4. Require both corrected witnesses to be within standardized motion RMS `.1` of the proposed corrected H action and to match its gripper sign on every committed control. The strict alternative uses `.05`; everything else stays identical.
5. Require each witness top donor's normalized progress to be within `.5` donor steps of the H top donor: `abs(p_H-p_W)*max(len_H-1,len_W-1,1) <= .5`. Require H to advance more than `.5` donor steps relative to the saved full top donor. Do not substitute episode success, true environment progress or the hidden current view.
6. If any requirement fails, acquire the missing current camera, discard provisional proposals without mutating guard/knob/anchor state, and run ordinary full retrieval/correction once. If all pass, commit H once, then run the ordinary guard and knob once from that chosen donor history. If a policy call is required, complete current vision before the policy and charge completion once.

Keep the selected corrected action as the authoritative cache anchor and preserve its committed tail. Accepted half does not refresh full-camera memory, so the next look is full. On a half-to-policy completion, invalidate donor-planning memory unless a separate stateless genuine-full retrieval has actually been performed; never install H donors as full donors. Preserve mode/availability bits separately from all-or-nothing `has_vision`, and preserve idempotence on duplicate requests.

The analysis computes extra individual-donor innovation witnesses to screen wider ensembles. The selected method uses only W and the weighted-mean innovation witness. Computing only those witnesses is an implementation optimization and earns no additional owner IR saving.

### GR00T numerical path

The same compute prices and candidate method apply to GR00T. Its recorded joint-path key is not automatically a genuine single-camera key. `d2_groot_mismatch.py` uses the specified workspace Python interpreter and the installed deployed GR00T torch/FA2 packages together, loads only the selected checkpoint vision tower/projector, and compares batch-two versus batch-one on identical reconstructed B pixels under bf16 autocast. This measures numerical sensitivity, not original-pixel re-encoding of the library.

An optional numerical-stability filter is implemented by `d2_stress.py`: take the eight largest projected perturbation norms from the measured B-dev sample, use both signs, and require the two-witness gate, H top donor and gripper signs to stay unchanged under all perturbations, with corrected-H motion RMS change ≤.025. This perturbation set and tolerance are **post-freeze B-dev calibration**. They are not a certified bound, and the supplied set is for the L10 checkpoint only.

Before real GR00T serving, obtain exact original/preprocessed B pixels and either re-encode selected-library single-view features or validate a robust representation on true single-camera keys. Validate the fallback/completion prefix too: two separately encoded views need not equal a joint encoding. A full-information separate-camera control must use the same fit/numerical path and the same total compute-share price. Do not silently replay the original full result after a numerically different completion or charge a redundant joint re-encode as if it were free. The required evidence is in REQUESTS.md; no numerical-parity or policy-parity claim is made here.

### Accounting

Let N be requested five-control slots; F full looks; H accepted half looks; C accepted half looks subsequently completed; M policy calls. A rejected partial look plus completion counts in F, not H. `V=F+H`.

```
pi05 work = .152*F + .076*H + .076*C + .848*M
groot work = .148*F + .074*H + .074*C + .852*M
IR = work / N
```

At fixed full-controller trajectories and calls, saving is `c*(H-C)/N`, where c is .076/.074. A policy call always costs the full vision plus policy remainder. No request-dispatch, launch, batching or call-reordering latency enters these equations. Report CPU planning/search time separately.

### Selection status and verification

The two-witness structure and selection of `.1`/`.05` are **post-freeze B-dev-informed design/calibration**, despite all fitted metric/head/table artifacts being library-only. The calibration/evaluation init split in `screen.json` is a development diagnostic, not untouched confirmation: the earlier atlas examination saw these recordings. No task-specific acceptance thresholds were selected. Wider ensembles, tighter atlas thresholds, agentview-first, and donor-mass camera choice are measured alternatives and are not selected for the initial pilot.

`d2_validate.py` checks source manifests/B exclusion, forced-full action and guard parity, periodic cap peeking, legal two-slot memory, hidden-current-view poisoning, unchanged gripper, and additive completion accounting. This does not test a plugin adapter. Before serving, sol must also test actual camera dispatch, full fallback equivalence, exact GR00T prefixes, action padding, tail persistence, independent connections, retries, one guard update and one completion charge.
