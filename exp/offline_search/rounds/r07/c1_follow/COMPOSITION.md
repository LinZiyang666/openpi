# Cache continuation component

`FollowExtension(table, extend_blocks=1|2|0, stage_gate=True,
state_valve=True)` is importable from `c1_follow.methods` independently of the
AWM wrapper. It stores only the frozen table and configuration; plans and
anchor cursors belong to each controller connection.

`plan(anchor)` returns an immutable `ExtensionPlan` with `blocks`,
`structural`, `stage_ok`, `extras`. The anchor dictionary is A's original
`_anchor`: original rows and float32 weights, original synthesized action,
anchor proprioception, task/episode/step/last_step. It grants the full requested
cap or zero; E=2 never silently becomes E=1. Every member, even zero-weight
members, must have a true successor chain covering each extension block.
Unanimity and unchanged learned mode hold from anchor through extension heads.

`check(anchor, blind_query_view, age, plan)` returns `(LookReason|None,
extras)`. `age` counts manifest execution blocks since the anchor. Use the
current BlindQueryView state. Eligible SF commitments check the valve at ages
1, 2 (and 3 for E=2). At the cap the component requests an ordinary budget
LOOK. An outside valve requests `follow_state_valve`, code 11. A LOOK enters
ordinary retrieval and the unchanged call judge; it is never itself a call.

`serve(anchor, age, fitted_base.act, fitted_base.cand_name, extras)` returns a
BlindResult. A complete native anchor block is shifted without recomputing:
GR00T age 2 therefore executes controls 10:15. Beyond complete native blocks,
each member advances `next^age`, and its executed head is synthesized with the
anchor's original float32 weights. π0.5 age 2 and both models' age 3 use this
bridge. The wire array is padded from its last available control, but only the
manifest-sized head is authorized for execution. No terminal library row is
clamped and no member is dropped or reweighted.

After an extension update anchor `last_step` to the current decision and
`phase=table.advance(anchor['rows'], age)`. Keep `rows`, `weights`, `action`,
`rs`, and `step` fixed until the next vision anchor. On MISS/reset/invalidation
clear the plan and cache anchor. Controller policy tails keep their existing
hook and cap; they never call this component.

The C3-owned `CallController.install_follow_extension(component)` bridge uses
these signatures. It also retains priority for R6's scheduled stall LOOK and
the policy-tail path. CU/CT calibration prices the original cadence: a future
SA arm needs a separate cadence/budget re-solve and telemetry-cap check.
C1 delivers the component and documents this hook; no SA class/arm is shipped.

The standalone wrapper is identity with `extend_blocks=0`, including Result
extras. With enabled gates, a structural/stage rejection delegates to A;
its executed actions, verdicts and vision flags are unchanged (extra stage
telemetry is present). Eligible anchors can be stopped by the valve at the
first blind check, before any extension is executed. Thus a literal demand
for identity for *every anchor that ultimately serves zero extension blocks*
conflicts with the immediate first-blind valve LOOK requirement. The delivered
implementation chooses that immediate LOOK on eligible commitments and
reports first-blind valve counts explicitly. No identity claim covers those
early valve aborts. Hard/unsupported anchors retain A's commitment and do not
run an effective valve; applying an early LOOK to those anchors would violate
the required rejection identity as well.
