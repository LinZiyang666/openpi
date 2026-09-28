<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 5 of an action-cache exploration for VLA policies
(π0.5, GR00T N1.5) on LIBERO. Read first: exp/offline_search/rounds/r05/CODING_BRIEF.md (binding rules: file ownership,
live-system rule, cost accounting), then rounds/r05/SELECTION.md, rounds/r05/FINDINGS.md, and the files named in your
task. Store: /home/weiland/trace_runs/offline_search_store (cold copy, read-only in your sandbox). Python:
/home/weiland/projects/openpi/.venv/bin/python from the repo root.
</context>

<hard_constraints>
- Your CPU range: 26-29,70-73. Prefix every python/numpy command with `taskset -c 26-29,70-73` and set OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never more processes than threads in your range. CPUs 38-43 and 82-87 belong
  to another project: never use them.
- GPU: none (CUDA_VISIBLE_DEVICES='').
- Edit only the files you own (CODING_BRIEF.md). Write new files only under exp/offline_search/rounds/r05/q1_commit/ and
  /tmp/q1_*. No git. No `rm -rf`. Never `pkill -f`. No servers, ports, timan107, LIBERO workers or chains.
  Do not read tests/review_tests/.
</hard_constraints>

<completeness_contract>
Resolve the task fully: implementation, tests (existing + new, against the installed / final files), arm specs,
prefits, hand-back. Check edge cases (step 0, episode reset / task change, MISS at the last decision, concurrent
connections, duplicate decision ids).
</completeness_contract>

<verification_loop>
Before finishing, re-run every check against the final files and report the final numbers. If a check fails, fix and
re-run.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; exact commands, numbers and file paths. Label anything unverified.
</grounding_rules>

<structured_output_contract>
Write exp/offline_search/rounds/r05/q1_commit/HANDBACK.md (see CODING_BRIEF.md "Hand-back"), then print a one-paragraph
summary as your final message.
</structured_output_contract>

<task>
You are R5 coding agent Q1 (method side only): policy-plan completion (C10) and the grasp-outcome check (D1).
1. `CommitJudge` (rounds/r05/q1_commit/judge.py): subclass K10's `PolicyTailJudge`
   (rounds/r04/k10_policy_tail/judge.py) with a new kwarg `policy_tail_gate` ∈ {"inherited" (default = K10 behaviour
   bit-for-bit), "lifecycle"}. With "lifecycle", `policy_tail_step(bq)` serves the policy's own remaining chunk after a
   MISS whenever the lifecycle is valid (same episode/task, previous decision a real vision MISS whose chunk executed
   exactly five controls, unconsumed tail, finite state), WITHOUT passing the rejected cache proposal through the
   inherited budget / span no-progress / base gates. Measured reason (rounds/r05/ideation_A/REPORT.md "The MISS path is
   where the next useful saving is concentrated", and K10's own hand-back: guard-only replays served zero policy
   tails): 88–94 % of l10 MISSes occur with a positive cache no-progress span, so the inherited veto blocks almost every
   policy tail. Keep K7's vision-confirmed stuck guard and span no-progress guard at the next real vision anchor; the
   policy-tail row stays a blind gap for those guards; never clear progress memos or fake a HIT in the histories.
   Optional kwarg `monitor` ∈ {"off" (default), "loeo_xyz99"} as specified in ideation A (request vision when the
   actual policy-head displacement residual exceeds the own-library LOEO p99).
2. `GraspCheckJudge` (same module): K7 anchor_tail configuration plus ideation D's D1 rule
   (rounds/r05/ideation_D/REPORT.md §2 D1 and its reference helper `contact_rule.py`, which reproduced 223/223 and 47/47
   historical alarms): on a blind HIT eligible for its continuation, if the command-matched aperture contradiction fires
   (thresholds exactly as D's helper), return a LookReason; at that vision decision force one MISS
   (`extras.os_force_miss=1`, distinct reason code); at most one intervention per episode, consumed only when executed.
   Fit its aperture quantiles from the deployed library only (both scales). π0.5 only (refuse GR00T clearly).
3. Tests: unit tests of both classes on replayed decision streams (use the K7/K10 test machinery), plugin blind
   selftests with `--os-blind --os-policy-tail --os-judge guard_only` (K10's fake policy and `policy_tail_test.py` as
   models) proving: "inherited" == K10 exactly; "lifecycle" serves a policy tail after every eligible MISS (count them)
   and the served chunk equals policy_chunk[5:10] exactly; guards still act at the next anchor; D1 fires exactly on the
   helper's historical decisions when replayed and never more than once per episode; concurrency with 8 connections.
4. Arm specs rounds/r05/q1_commit/arms_q1.json (emit_arms format, `<RUN>` fit placeholders, full model, K10 MISS, L=5,
   `cost_ledger: true`; flags `--os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail
   --os-judge guard_only --os-no-shadow-native` for C10, and without `--os-policy-tail` for D1): C10 lifecycle at
   π0.5 {l10, spatial} × {50, 500} (4 arms, base kwargs as K7 tail1ug: anchor_tail, budget 1, gates budget_only; lib
   current kref 5 at 50, lib big kref 8 at 500); D1 at π0.5 l10 × {50, 500} (2 arms). Give the exact prefit command
   per arm and run the prefits yourself into /tmp/q1_fits/ (report sizes and sha256).
</task>
