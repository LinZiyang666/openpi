<task>
You are R4 ideation agent A. Your topic is the owner's new direction "look once, act several steps" (看一眼，做几步):
a decision WITH vision (stage-1 keys, AWM retrieval) acts as an anchor; the following decisions are taken WITHOUT running
the vision encoder (stage 1 skipped -> these decisions cost ~0 IR) until a condition sends the loop back to vision.
Today every decision pays stage 1 (IR floor .152); at the best l10 point (500-episode library, guard-only mixed mode,
IR .238) that floor is 64 % of the cost. Design, measure and rank concrete methods for this, including:
- What to serve on a vision-free decision. The measured closed-loop facts (FINDINGS.md) say AWM's consecutive picks hit
  the exact successor row only 18–29 % of the time and stall on the same row 15–40 %, and the kernel mixes 3–4 library
  episodes: so naive "follow the top-1 demo in time" is suspect. Evaluate alternatives such as advancing every kernel
  member along its own library episode and re-synchronising each member's phase with the robot's proprioceptive state
  (robot_state valid dims) — or anything better you find. Quantify on the offline store (all four cells, inf and cache
  query cells, 50- and 500-episode libraries) how the served action degrades with the number of blind decisions
  1..4, split by task phase (near library gripper transitions vs not, early/mid/late), and compare with (a) vision every
  decision (AWM), (b) simply executing a longer part of the anchor's chunk (the "execute more steps" baseline).
- When to look again: bounded blind budget, phase-aware triggers known from the library without vision (e.g. a gripper
  open/close ahead in the members' episodes, near-terminal rows), proprioceptive triggers (no motion, deviation from the
  expected library state), after any MISS. Estimate from the store / closed-loop logs how often each trigger would fire
  and at which phases the pure-cache failures (deadlock spells, grasp/release traps) start relative to them.
- How it combines with the existing mixed HIT/MISS mode (guards / periodic MISS) and with the 500-episode library, and
  the resulting IR (cost model: vision decision .152, vision-free decision ≈ 0, MISS 1.0).
- What the closed-loop server/plugin must do: read exp/offline_search/closed_loop/plugin.py (in particular the
  per-connection policy wrapper `_ConnPolicy.infer` and how `on_search` / bookkeeping work) and state precisely how a
  vision-free decision can be served without the interceptor running stage 1, what the client must send, and how the
  QueryView history stays consistent. Do not implement it; specify it.
Your CPU range: 0-11,44-55 (24 logical CPUs, at most 24 processes). Your letter: A.
</task>
