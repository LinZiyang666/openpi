<task>
You are R5 coding agent Q6 (method side only): combine the wrist-only stage-1 key with blind stepping, which needs a
wrist-only version of the vision-confirmed stuck guard.
Facts: K3's wrist_only stage-1 mode runs only the wrist camera tower on vision HITs (MISS completes the other camera);
its guard-only arms reached π0.5 SR .820 / .734 / .974 / .924 (l10 500 / l10 50 / spatial 500 / spatial 50). K7's
vision-confirmed stuck guard (rounds/r04/k7_guard/judge.py) requires the MIN of both cameras' task-centred cosines to
reach the library 95th percentile, bit-identical to stock at B=0; K1's `BlindWristMixedJudge` (rounds/r04/k1_blind/
wrist.py) still uses K1's dense motion-only stuck guard, which R4 showed fires ≈2.8× too often and costs SR. On wrist HITs
only the wrist key exists, so a wrist-consistent confirmation rule is needed.
1. `WristVisionConfirmedBlindJudge` (rounds/r05/q6_wrist_blind/judge.py): subclass / compose K7's guard with K3's
   WristAWM base and K1's blind machinery so that the stuck guard's visual confirmation uses the wrist camera cosine
   only (task-centred, with its own library 95th percentile calibrated on the deployed library, both scales), and the
   motion part is exactly K7's. Justify the rule; report stuck firing rates on the libraries (successful demos) and on
   replayed decision streams for stock two-camera, K1 dense, K7 two-camera and your wrist rule, at both scales and both
   suites. Keep anchor_tail (budget 1, gates budget_only) as the default serving; phase B=2 as a variant.
2. Tests: unit tests, plugin blind selftests with `--os-blind --os-stage1-mode wrist_only --os-tokens off --os-judge
   guard_only` (K1/K3/K7 recipes as models), 8-connection concurrency parity, and every existing selftest you touch.
3. Arm specs rounds/r05/q6_wrist_blind/arms_q6.json (emit_arms format, `<RUN>` placeholders, full model, K10 MISS,
   cost_ledger): wrist + anchor_tail + wrist-confirmed guard at π0.5 {l10, spatial} × {50, 500} (4 arms). Exact prefit
   commands; run the prefits yourself into /tmp/q6_fits/ (sizes, sha256).
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Read first: exp/offline_search/rounds/r05/CODING_BRIEF.md
(binding rules), rounds/r04/k3_cost/HANDBACK.md (wrist_only mode, WristAWM), rounds/r04/k1_blind/HANDBACK.md and
wrist.py, rounds/r04/k7_guard/HANDBACK.md and judge.py, rounds/r03/h3_judge/judge.py. You own only
rounds/r05/q6_wrist_blind/ (method side; no plugin edits — if a plugin change seems necessary, stop and describe it in
the hand-back instead). Store: /home/weiland/trace_runs/offline_search_store (cold copy).
</context>

<hard_constraints>
- Your CPU range: 10-13,54-57. Prefix every python command with `taskset -c 10-13,54-57`, BLAS/OMP threads 1, never more
  processes than threads. CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: none. Write only under rounds/r05/q6_wrist_blind/ and /tmp/q6_*. No git, no `rm -rf`, never `pkill -f`, no
  servers/ports/timan107/LIBERO workers/chains. Do not read tests/review_tests/.
</hard_constraints>

<completeness_contract>
Resolve fully: guard rule + evidence at both scales, implementation, tests, arm specs, prefits, hand-back.
</completeness_contract>

<verification_loop>
Re-run all checks against the final files and report final numbers.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; exact commands, numbers, file paths. Label hypotheses.
</grounding_rules>

<structured_output_contract>
Write exp/offline_search/rounds/r05/q6_wrist_blind/HANDBACK.md, then print a one-paragraph summary.
</structured_output_contract>
