<task>
You are R4 coding agent K2: the serving infrastructure for vision-free ("blind") decisions and the new log fields.
Follow ideation A §3 (server, client and QueryView specification) exactly unless a test forces a change. First create
exp/offline_search/closed_loop/blind.py with the contract dataclasses from CODING_BRIEF.md (LookReason, BlindResult) and
the BlindQueryView, so agent K1 can import them early. Then:
1. In closed_loop/plugin.py add an opt-in pre-inference blind path in the per-connection wrapper (`_ConnPolicy.infer`):
   periodic-MISS check by the global decision counter first; then `method.blind_step(bq)`; on BlindResult serve the
   chunk without calling the inner policy (no interceptor, no stage 1/2/3), apply the model's output transforms
   exactly as the normal path does, commit dense history once (has_vision mask, sentinel keys, prev_a_exec = served
   chunk, prev_hit True), advance the orchestrator's state history / step counter exactly once through a plugin-level
   helper (no src/ edit), broadcast the action once, log once. π0.5 and GR00T paths (see ideation A §3.4 for GR00T's
   adapter/transform chain and gripper conversion). `bq.rs` must be bit-identical to the key builder's robot_state.
2. Log fields from CODING_BRIEF.md on every decision (`vision`, `src`, `hit`, `blind_age`, `look_reason`, `miss_k`,
   `s1_ms`, `s23_ms`, `served_head`), keeping all existing fields; disable/mark the native shadow search on blind rows.
3. Tests: a fake stage-1 that raises if called on a blind decision; stage-call count equals the number of vision
   decisions; CPU selftest (extend selftest.py) through vision HIT → blind → blind → vision → MISS → vision with a toy
   blind-capable probe method (write one in probe.py) and with K1's method if it exists by then; two interleaved
   connections; episode reset; verify_logs covering blind rows; pure-cache and existing mixed selftests unchanged
   (byte-identical logs when the blind option is absent). GPU smoke: one STAGE1_ONLY=1 π0.5 server (and one GR00T if
   possible) on a free port in 23180-23189, driven by replay_client with the toy blind method: prove stage 1 was skipped
   on blind decisions (call counts, s1_ms null) and that served actions/wire outputs match an offline recomputation.
Hand-back must include the exact plugin flag(s) to enable blind serving and a launch example.
</task>
