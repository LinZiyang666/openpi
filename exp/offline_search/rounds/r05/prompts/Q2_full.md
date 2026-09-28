<task>
You are R5 coding agent Q2 (you own the plugin): GR00T mixed mode with chunk-tail execution.
GR00T N1.5 has never been run in mixed HIT/MISS mode in this project; its chunk horizon is H=16 while the client
executes five controls per request. R4 showed on π0.5 that executing more of each chunk is worth several SR points on
libero_10 (pure inference L=10 .904 vs L=5 .85) and that anchor_tail (cache chunk tail) is the robust blind method.
Implement, following rounds/r05/ideation_A/REPORT.md proposal 2 ("GR00T mixed anchor cycles (CycleTail)"):
1. Plugin: generalize K10's `--os-policy-tail` to GR00T (currently π0.5-only) using the GR00T CPU state/output adapter,
   retaining the original transformed wire actions exactly (as K10 does for π0.5), and add `--os-policy-tail-blocks
   <1|2>` so that for H=16 the policy chunk can serve offsets 5 and (with 2) 10 as consecutive blind decisions before
   vision is required; the present "one blind decision after a MISS" rule and buffer consumption must become
   cursor-aware. Keep `--os-policy-tail` alone byte-identical to the current π0.5 behaviour; flags absent ⇒ byte-identical
   logs. Keep K6's per-connection locking contract. Also allow the cache anchor_tail on GR00T to serve up to two blind
   blocks when the method asks (check what K1's BlindAWM already does for H=16).
2. Method: `CycleTail` (rounds/r05/q2_groot/judge.py) wrapping K1 `BlindAWM` for GR00T: at each real vision anchor
   force a MISS every `cycle_k`-th anchor (first anchor of an episode is a MISS), HIT otherwise; after either source,
   serve the source's own chunk tail for `tail_blocks` ∈ {1, 2} blind decisions (10 or 15 executed controls per chunk),
   then require vision. No guards on fabricated visual keys. Handle GR00T's gripper sign convention correctly (see
   memory note in the repo docs / K7 refusal reason: GR00T's gripper sign is reversed relative to π0.5).
3. Tests: fake-policy selftests for GR00T and π0.5 (existing K10 π0.5 tests unchanged), exact wire equality of served
   offsets against an L=10 / L=15 client queue on stored actual GR00T chunks (the store has GR00T inference traces with
   the policy's full chunks), lifecycle edge cases, 8-connection concurrency parity, and every existing selftest matrix
   (K2, K1, K4, K5, K6, K7, K10) against the installed files.
4. Arm specs rounds/r05/q2_groot/arms_q2.json (emit_arms format, `<RUN>` placeholders): GR00T CycleTail with
   cycle_k=4, tail_blocks=1 (G10) at {spatial, l10} × {50, 500} (4 arms; fits: K1 BlindAWM GR00T kwargs lib current kref
   5 / lib big kref 8, full model loaded because MISSes happen); GR00T pure-policy L=10 control at {spatial, l10}
   (client replan 10; library-independent). Exact prefit commands; run the CycleTail prefits into /tmp/q2_fits/.
</task>
