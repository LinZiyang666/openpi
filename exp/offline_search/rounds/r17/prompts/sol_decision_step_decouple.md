<task>
Coordinator follow-up, 2026-10-17 21:5x. Owner refinement of the fix you just made. Supersede the "defer the look to the next boundary" design.

Owner rule: checkpoints are designed in decision steps, decoupled from environment steps.
- A decision is one whole long segment: a head plus all of its tail deliveries up to `commit_horizon`.
- Tail deliveries are action delivery only. No recipe/gate/knob/stall/corrector evaluation happens there, and no decision counter or clock advances.
- Every clock and schedule counts decisions (segment heads): `decision_seq` as seen by recipes, the knob `periodic` run/cap history (`has_vision`/`hit` histories per decision), the stall gate's call/extra-look/cooldown schedule (`look_due`, cooldown anchors), and the corrector's step index. Environment steps are used only where the environment itself needs them, e.g. episode limits and execution acknowledgements.
- The frozen worker packages keep requesting every `replan_steps`. Do the decoupling on the server side: a tail request returns the next committed actions without touching the recipe or clocks. `per_step` should record a decision index separately from the env-step index.

Invariant: after decoupling, the long-segment and short configurations share exactly the same decision-level code path (cache lookup, gate, knob, stall gate, corrector). The only difference is `commit_horizon`, the number of environment actions delivered per decision. Make that a test: feed the same decision-level observation sequence to a commit_horizon = replan_steps configuration and a commit_horizon > replan_steps configuration. The recipe/clock decisions must be identical; only the delivered action counts differ.

Then:
- Update the tests: a head is always followed by its tails; no force_look or stall evaluation on tails; the recipe-visible decision index is contiguous over heads only.
- Prove it on the real path, as before.
- Report how this changes the stall gate's cadence relative to its Q3 calibration (it previously counted 5-step blocks), and whether any stall/knob artifact would need recalibration. Do not recalibrate; just report.

Continue with the rest of the original task (audit, trash, rebuild, balance). Rebuilt campaigns must use this decoupled code.
</task>

<rules>
Same as the original task: CPU only; no launches; mv only; do not touch running runs or chains.
</rules>
