# Implementation specification for sol

Status: frozen offline proposal, not a deployed plugin and not a demonstrated SR improvement. Primary is the **pre-look, progress-aware safety atlas with short camera memory and observed displacement**. The reference arithmetic is `atlas_reference.py`; `atlas.py` fits and evaluates the safety labels inside each episode fold. All parameters below are global across tasks/models/suites. Task metric matrices and correction heads remain task-indexed and explicitly belong to layer 5.

## Primary algorithm

Keep the ten-control cache commit, only-no-progress guard, policy-tail lifecycle and layer-4 schedule. Every actual cache look still obtains current wrist vision. A half look combines that current wrist view with the third-person view from the most recent **genuine full look**, exactly two five-control slots earlier. Never extend the number of blind controls.

For library row i let p(i) be its predecessor two slots earlier in the same episode, clamped to episode start. Use the full baseline's selected-library PCA bases, each retaining 64 coordinates. Define:

```
x_full(i) = [P0(i), P1(i), rs(i)[:8]]
x_half(i) = [P0(p(i)), P1(i), rs(i)[:8], rs(i)[:8] - rs(p(i))[:8]]
```

Half query inputs are the saved genuine P0 and state, the current P1 and current state. The current P0 is absent, not a zero vector and not an imputation presented as an observation. Memory is private to an episode/connection. Reset on episode/task changes, discontinuity or invalid execution metadata. Rejected camera plans simply request an ordinary full look; the primary does **not** start a wrist look and then acquire a second camera to reject it.

### Fit the cache and half correction

1. Fit the unchanged R10 full recipe from exactly the selected B library. Retain failed episodes. Retain its PCA, action sigma, guard calibration, full correction heads and full distance scale. Verify selected episode/hash identity; do not import parameters from larger libraries.
2. Fit one half AWM metric per task in the 144-dimensional x_half space. Use the existing action-neighbour construction: first five controls, seven action dimensions standardized by selected-library sigma, nearest three other-episode action heads, ridge `0.1 * trace(Sw)/d`, full-rank Cholesky. Apply the existing stable top-16 kernel; kref is five only when every task has exactly five episodes, otherwise eight. Candidate rows keep their current action/progress metadata; only their camera/context representation changes. Candidates near episode start retain the explicitly clamped p(i) convention.
3. Fit separate half correction heads per task from selected-library LOEO retrieval/synthesis. Query training rows with step at least two; exclude their entire episode from donors. Features are x_half, synthesized ten-control/seven-channel action divided by the same sigma, `min(step,120)/120`, and the existing ten-way task indicator: 225 features. Targets are ten-control/six-motion-channel policy action minus synthesized action, in sigma units. Use R10's episode-balanced anchor weights, 384 RFF, seed zero, ridge alpha 100. Never correct gripper or padding. Preserve the correction across the committed tail.
4. Fit a half-specific distance scale with the R10 five-episode-fold procedure, using half representations and train-fold sigma/metrics. Use the episode-balanced median noninitial held-out nearest distance, and the unchanged correction strength `.5 * clip((2 - d/scale)/1.25, 0, 1)`. Do not compare raw half distances with the full scale. The offline diagnostic used training LOEO scales for tractability; this production calibration is specified explicitly and must be verified on B dev.
5. Fit selected-library state scale as the standard deviation of the valid eight state channels, floored at `1e-6`. This is a per-library normalization, not task-specific camera thresholds.

### Fit the safety atlas

The safety atlas labels a library row by **camera omission safety**, not gripper stages or success labels. Use the same full and half metrics above. For each library row with step at least four, retrieve/synthesize with its entire episode excluded:

- Full raw cache action at i, half raw cache action at i, and full raw retrieval at p(i).
- Motion agreement is RMS over ten controls and six motion channels, divided by selected-library action sigma. Require RMS at most `.15`.
- Require equality of full and half gripper signs at every committed control.
- Compute each current top donor's progress advancement relative to the previous full top donor using the existing equation `(current_progress - previous_progress) * max(current_ep_len - 1, 1) > .5`. Require the two Boolean advancement decisions to agree.

Set `safe[i]` to the conjunction; early rows are unsafe. These are labels from raw cache synthesis, deliberately independent of in-sample correction-head optimism. The main offline assessment separately measures the final corrected action. All fit labels come from the selected library. Whole-episode outer validation in `atlas.py` excludes the validation episodes from label fitting, donor pools, metrics and sigma.

### Decide cameras before encoding the current images

At a full look save its step, true P0, robot state, and actual full-retrieval donor rows/weights. A policy call with a full look can also provide this memory. Camera memory and the cache action anchor are separate objects.

At a planned new cache look:

1. Force full if `step < 4`, if the last genuine full look is not exactly two slots old, if the previous no-progress span is positive, if a guard-aware periodic call is already due, or on an actual lifecycle/metadata failure. A half look never renews the full-camera timestamp.
2. Advance every saved full donor by two same-episode rows. Missing successors contribute zero support. Require support weight at least `.99` and safe successor weight at least `.75`. Do not renormalize missing support into apparent safety.
3. Average supported donors' two-slot state displacements with renormalized supported weights. Compare that expectation with the current measured state displacement from the saved full observation. Require RMS in selected-library state-SD units at most `.2`.
4. If all checks pass request wrist-only; otherwise request full. This planner uses no current image, no task ID threshold, no future action, no true episode progress and no eventual success.
5. For half mode use the half metric/head/scale. For full mode use the byte-preserved full cache branch. Update the ordinary no-progress guard once from the chosen top donor and execute the ordinary layer-4 verdict. On a half MISS, encode the deferred missing camera before the policy runs and charge completion. Do not feed a missing/deferred camera into the policy.

The stored safety Boolean is not a certificate: full and half donor progress still disagree on held-out B rows. Log this in shadow dev and measure actual guard timing in the pilot.

## Integration details that must be implemented deliberately

- R7's wrist wrapper alone is insufficient: it changes retrieval fields but the current R10 corrector requires both current camera projections. Add a mode-specific correction path accepting the explicit memory features. Never temporarily present stale P0 as fresh to generic history consumers.
- Preserve one authoritative action anchor after final synthesis/correction, including mode, rows, weights and corrected tail. Restore full fitted fields after a half query without restoring an older anchor. Request retry must return the same result without appending progress twice.
- Store `has_v0`, `has_v1`, `full_camera_step` independently of the current all-or-nothing `has_vision`. A half look has vision but no current v0. Disabled two-view diagnostic guards must not treat deferred zeros as observations; their absence must not alter the only-no-progress verdict.
- Current plugin camera dispatch forces full on lifecycle reason codes, including the normal fresh look after a committed policy tail. A new capability can admit half **only after explicitly validating**: same episode/task, a real full policy look two slots earlier, the successfully executed single policy-tail block, exact execution counts, finite state, and no pending span. Do not waive all occurrences of reason 6/8. Retain camera memory through valid policy-tail consumption even when the action anchor is invalidated. This is an encoding choice, not permission to execute another blind block.
- If a half MISS is completed, either perform a stateless full retrieval after completion and store its real full donors, or invalidate camera-planning memory until the next ordinary full look. The initial implementation should choose the latter; do not use half-donor rows as if they came from a full retrieval.
- Periodic due status is computable before acquisition from accepted histories and the existing keyed cap. Peeking must not consume/update schedule state. Guard calls still reset the schedule. For the distance knob, retain a separate library-fitted half distance threshold/quantile map under layer 5, or force full whenever that knob is active in the first pilot. Raw half and full distances are incomparable. The frozen primary pilot uses periodic and off; distance is an explicitly deferred secondary experiment.
- Preserve parent-library row IDs and action payload routing. Fit artifacts should contain bases, both metric/head sets, both distance scales, safe rows, successors, state scale and provenance.
- GR00T remains blocked at serving by the missing validated one-camera encoder path. Its offline feature results do not authorize simulated cheap vision or zeroing a camera after encoding both.

## Secondary proposals / controls

**Full vision plus displacement control:** use `[P0(now),P1(now),rs(now),rs(now)-rs(previous full)]`, with a matching full metric/head. In this exploration a fixed two-slot displacement improved the full-view baseline too. This control distinguishes a better state representation from a benefit of camera omission. Its deployed handling of nonstandard look gaps needs the same explicit age contract; fall back to unchanged full on other ages.

**Post-wrist agreement option:** only if sol adds a transactional second-camera acquisition hook before committing query/progress histories. Compare corrected half-memory and corrected wrist-plus-displacement proposals; require motion RMS at most `.1`, identical gripper signs and both proposed top donors advancing relative to the prior genuine full top donor. If rejected, complete the current second camera and rerun the original full proposal exactly once. The quantitative screen is `gates_dynamic.json`; this is secondary because completion/search bookkeeping and measured price nonadditivity complicate a clean comparison. No production evidence is claimed for it.

Do not ship unchanged stale-key substitution, unconditional wrist-only, image-RGB distance, or a new token representation on the strength of these screens. Their measurements and limitations are in OFFLINE.md.

## Required verification before serving

Sol should verify mode-off full action/row/weight parity; hidden-current-v0 poisoning; independent connections and reset; exact two-slot memory age; row mapping; half-head finite output and unmodified gripper/padding; corrected tail identity; no double guard update or duplicate-request charge; completion once on half MISS; unchanged guard/periodic timing when camera mode is forced full; and the narrow valid-policy-tail exception. Those are implementation checks, not experiments on test A. `check.py` validates the explorer's equations/data boundaries and pure planner only; it does not claim plugin parity.
