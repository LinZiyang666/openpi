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

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 4 (R4) of an action-cache exploration for VLA policies
(π0.5, GR00T N1.5) on LIBERO. You are one of four coding agents working in parallel. Read first:
exp/offline_search/rounds/r04/CODING_BRIEF.md (binding: file ownership, live-system rule, the method<->plugin contract,
log fields, cost ledger, rules), then exp/offline_search/rounds/r04/SELECTION.md, the ideation reports in
exp/offline_search/rounds/r04/ideation_{A,B,C}/ (REPORT.md; C also REPORT_2.md), exp/offline_search/closed_loop/README.md,
exp/offline_search/harness/README.md, and the code you build on (named in your task).
Store: /dev/shm/offline_search_store (hot) or /home/weiland/trace_runs/offline_search_store (cold copy, same content).
Python: /home/weiland/projects/openpi/.venv/bin/python from the repo root (GR00T servers use their own venv, see
closed_loop/ops/start_server.sh).
</context>

<hard_constraints>
- Your CPU range: 22-25,66-69. Prefix every python/numpy command with `taskset -c 22-25,66-69` and set
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never more processes than threads in your range.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: allowed for smoke tests only: at most one server at a time, preferably STAGE1_ONLY=1 (about 2.2 GB), never more than 4 GB, and only when nvidia-smi shows at least 8 GB free; stop GPU work at once on any out-of-memory error; release the GPU after each test.
- Edit only the files you own (CODING_BRIEF.md "File ownership"). Develop in a copy, keep existing behaviour
  byte-identical when your new options are absent, install each shared file atomically (temp file + mv), and re-run the
  existing selftests after installing.
- No git. No `rm -rf`. Never `pkill -f`; kill only processes you started, by PID. No timan107, no LIBERO workers, no
  closed-loop chains. Never touch ports, processes or tmux sessions you did not start. Do not read tests/review_tests/.
  Do not modify src/, harness/, profile/ or rounds/r01..r03/.
</hard_constraints>

<completeness_contract>
Resolve your task fully before stopping: implementation, tests, and the hand-back file. Do not stop at the first
working version; check edge cases (episode reset, step 0, after a MISS, both models where required, both library sizes
50 and 500 where required).
</completeness_contract>

<verification_loop>
Before finishing, verify every deliverable against the task and the brief with real runs (selftests, parity checks,
smoke runs as allowed). If a check fails, fix and re-run; report the final numbers, not the first attempt.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; give exact commands, numbers and file paths. Label anything unverified.
</grounding_rules>

<structured_output_contract>
Write rounds/r04/k2_serving/HANDBACK.md (files changed/added with install times; exact switches/arm-spec fields; tests
and their numbers; what the coordinator must do next; caveats), then print a one-paragraph summary as your final
message.
</structured_output_contract>
