<task>
You are R4 coding agent K1: the method side of "look once, act several steps" plus two pure-cache/guard methods.
Directory: exp/offline_search/rounds/r04/k1_blind/ (create; add __init__.py). Build on
exp/offline_search/rounds/r02/g1_awm/awm.py (AWM), rounds/r03/h1_trap/awm3.py (AWM3) and rounds/r03/h3_judge/judge.py
(MixedJudge) by subclassing/wrapping; never edit them. Implement, following ideation A §1.1–§1.3 and §2 exactly unless
a test forces a change (document any change):
1. Blind-capable AWM methods (usable pure-cache and inside the mixed judge), implementing
   `blind_step(bq) -> BlindResult | LookReason` from the CODING_BRIEF contract (types in
   exp.offline_search.closed_loop.blind, created by K2; if it does not exist yet when you start, define identical
   dataclasses in your package behind a try/except import so both work): serving variants `phase_particles`
   (fixed 16-member anchor kernel and weights, each member advanced along its own library episode, per-member phase
   chosen from offsets {h-1,h,h+1} by the proprioceptive cost with the .05 offset penalty, monotone, ≤2 rows per blind
   step), `kernel_clock`, `top1_clock`, `anchor_tail` (serve the unexecuted part of the anchor chunk; π0.5 H=10 → 1
   blind decision, GR00T H=16 → 2); budget B ∈ {1,2,3,4}; look gates: gripper event ahead (≥.20 anchor mass), near
   terminal (≥.20 mass on the last two rows), two low-motion intervals (library 10th percentile), displacement
   residual > .5 (thresholds .25/1 as variants), lifecycle (first decision / after MISS / invalid anchor); a
   `gates="budget_only"` variant for the ungated tail baseline. State scales/percentiles fitted from the deployed
   library only. Every variant must also run through the ordinary `query()` (vision decisions) exactly like AWM.
2. A gap-aware MixedJudge subclass (`noprog_span`: retrieval progress measured only on vision anchors, accumulated over
   elapsed decisions; motion guard from dense proprioception), wrapping the blind-capable base so the plugin's mixed
   mode can run blind stretches; plus a MixedJudge "memo reset after MISS" variant (ideation B2) and plain `noprog_n`
   passthrough.
3. `control_step_library` (ideation C REPORT.md proposal 1) as a pure-cache method with its ablation G (virtual-distance
   ranking, unshifted actions) and GS (spliced actions).
Tests: offline harness smoke on π0.5 and GR00T cells (inf and cache) at both library sizes; bit-for-bit equality of
every variant's vision `query()` with AWM (or documented, justified differences); a blind-step replay on the store
reproducing ideation A's h=1/h=2 phase-vs-clock numbers to within reconstruction tolerance; a CPU mixed-sequence test of
blind_step with a fake plugin driver (vision → blind → blind → vision → MISS → vision) checking anchors, invalidation
and gate reasons. Deliver `k1_blind/arms_r4.json`: emit_arms spec rows (with `<RUN>` placeholder for fit artifacts) for
the third- and fourth-batch arms in SELECTION.md at both library sizes (π0.5 l10 first, then spatial; GR00T pure-cache
tail/phase), plus prefit commands and measured pickle sizes/fit times.
</task>
