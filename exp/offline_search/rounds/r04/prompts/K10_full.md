<task>
You are R4 coding agent K10: let a blind decision right after a policy MISS execute the tail of the policy's own chunk.
Measured motivation (π0.5 libero_10, 500 paired episodes each):
- pure inference, 5 executed control steps per call (L=5): SR .845–.850; pure inference L=10 (the client executes the
  whole 10-step chunk): SR .904. Executing the full chunk alone is worth ≈ +5–6 pp on l10.
- R4 anchor_tail (after a vision HIT, the next decision is served blind from steps 5–9 of the same synthesized chunk)
  with the K7 vision-confirmed guard: SR .880 @ owner IR .203 (500-episode library), .806 @ .242 (50-episode library).
- Today every decision right after a MISS must look again: the plugin forces vision
  (`closed_loop/plugin.py:_try_blind`: `not s.hits[-1]` → LookReason(6, "lifecycle")) and K1's
  `lifecycle_reason` invalidates the anchor after a MISS. So a policy chunk is executed only for 5 steps and then a new
  vision + retrieval (and often another MISS) follows.
Build an opt-in "policy tail" path:
1. Plugin (closed_loop/plugin.py; opt-in flag, e.g. `--os-policy-tail`; without it behaviour and logs stay
   byte-identical): when the previous decision of this connection/episode was a MISS whose policy chunk was executed
   for exactly 5 steps, the next decision may be served blind from that same policy chunk's steps 5..H−1 (π0.5 H=10:
   one blind block), in exactly the action convention the client would have received had it executed the whole chunk
   (verify against an L=10 client path: the executed action sequence must be identical step for step). Ask the method
   first (a new optional method hook, or a BlindQueryView field carrying the previous policy chunk) so the method's
   guards/gates can still request vision; record `src="policy_tail"` (or similar), `vision=false`, `hit` semantics
   documented, `blind_age`, `look_reason`, and keep the R4 cost ledger correct (a policy-tail decision costs 0).
   Episode/task change, invalid state, executed_steps≠5, a second consecutive blind decision, or a MISS anchored in a
   different episode must force vision. Keep K6's per-connection concurrency contract.
2. Method side (new module rounds/r04/k10_policy_tail/judge.py): subclass K7's `VisionConfirmedBlindMixedJudge`
   (rounds/r04/k7_guard/judge.py) so that (a) after a HIT the existing anchor_tail (budget 1, gates budget_only) runs
   unchanged, (b) after a MISS the policy tail is served under the same budget and guards, and (c) the vision-confirmed
   stuck guard and span no-progress guard treat the policy-tail decision as a blind gap. Do not modify K1/K7 files.
3. Tests: extend the existing plugin selftest machinery (closed_loop/selftest.py fake policy, K1 run_plugin_blind.sh,
   K6 concurrency_test.py, K7 checks) to prove: flag off ⇒ byte-identical logs; flag on ⇒ after every MISS the next
   served chunk equals policy_chunk[5:10] (plus the documented padding) exactly; a guard/gate that wants vision wins;
   lifecycle edge cases (step 0, MISS at the last decision, task change, concurrent connections) behave; cost ledger
   counts policy-tail decisions as zero-cost, non-vision, non-MISS.
4. Arm specs rounds/r04/k10_policy_tail/arms_k10.json (emit_arms format, `<RUN>` fit placeholders, full model,
   `--os-blind --os-policy-tail --os-judge guard_only --os-no-shadow-native`, `cost_ledger: true`, K10 MISS — no
   yaml miss patch): π0.5 l10 500-library and 50-library, spatial 500-library, with base kwargs exactly like K7's
   tail1ug arms (anchor_tail, budget 1, gates budget_only; lib big kref 8 at 500, lib current kref 5 at 50), plus the
   exact prefit command per arm (the coordinator runs prefits).
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 4 of an action-cache exploration for VLA policies on
LIBERO: a HIT serves a retrieved/synthesized chunk, a MISS runs the full policy; each decision executes 5 control steps;
R4 added vision-free ("blind") decisions between vision anchors. Read: rounds/r04/CODING_BRIEF.md (binding: live-system
rule, method<->plugin contract, log fields, cost ledger), rounds/r04/k2_serving/HANDBACK.md (blind serving path),
rounds/r04/k6_concurrency/HANDBACK.md (current locking contract), rounds/r04/k5_rand/HANDBACK.md (last plugin owner
before K6), rounds/r04/k1_blind/HANDBACK.md, rounds/r04/k7_guard/HANDBACK.md, closed_loop/README.md.
The coordinator is running closed-loop chains on this code RIGHT NOW: every 10–25 minutes a new server imports
closed_loop/plugin.py. You own closed_loop/plugin.py, closed_loop/selftest.py, closed_loop/verify_logs.py,
closed_loop/blind.py, closed_loop/replay_client.py and the new rounds/r04/k10_policy_tail/. Develop in
rounds/r04/k10_policy_tail/dev/, prove byte-identical behaviour without your flag (including with --os-blind and with
the K5 randomization flags), and install each shared file in ONE atomic step (temp file in the same directory + mv).
Store: /home/weiland/trace_runs/offline_search_store (cold copy). Python: .venv/bin/python from the repo root.
</context>

<hard_constraints>
- Your CPU range: 26-29,70-73. Prefix every python command with `taskset -c 26-29,70-73`, BLAS/OMP threads 1, never
  more processes than threads. CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: none (CUDA_VISIBLE_DEVICES='').
- Edit only files you own. Do not modify src/, harness/, profile/, ops/*, rounds/r01..r03/, or other rounds/r04/k*
  directories. No git, no `rm -rf`, never `pkill -f` (kill only your own PIDs), no servers/ports, no timan107, no
  LIBERO workers, no chains. Do not read tests/review_tests/.
</hard_constraints>

<completeness_contract>
Resolve fully: plugin path, method, tests (existing + new), atomic install, arm specs, hand-back. Check edge cases.
</completeness_contract>

<verification_loop>
Re-run all existing plugin selftests (K2, K1, K4, K5, K6, K7 recipes) and your new tests against the INSTALLED files
and report the final numbers.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; exact commands, numbers, file paths. Label anything unverified.
</grounding_rules>

<structured_output_contract>
Write rounds/r04/k10_policy_tail/HANDBACK.md (design; files with install times and sha256; switches; tests and numbers;
arm specs and prefit commands; what the coordinator must do next; caveats), then print a one-paragraph summary.
</structured_output_contract>
