<task>
You are R4 coding agent K7: a vision-confirmed stuck guard for the blind (look once, act several steps) judge.
Measured problem (closed loop, π0.5 libero_10, 500-episode library, 500 episodes per arm):
- stock R3 guard-only MixedJudge `r3mx_p_l10_g500`: SR .864, MISS share .101; guard reasons stuck(1) 699,
  terminal(2) 250, overtime(3) 135, no-progress(4) 1,899 forced MISSes.
- R4 `BlindMixedJudge` (rounds/r04/k1_blind/judge.py, progress_guard="noprog_span") with budget B=0 (every decision
  sees vision; arm r4b3_p_l10_500_b0g): SR .842, MISS share .140; stuck 1,952, terminal 257, overtime 150,
  no-progress 1,785.
- Same judge with phase_particles B=2 (r4b3_p_l10_500_ph2g): SR .850, vision share .586, MISS share .144; stuck 1,774,
  no-progress 2,128.
The stock stuck definition (rounds/r03/h3_judge/judge.py header, guard 1) requires BOTH state motion below the
library 10th percentile AND the min-camera task-centred key cosine to the previous decision at or above the library
95th percentile. The blind judge replaced it by dense proprioceptive motion alone (`dense_motion` vs `motion10`),
dropping the visual-stillness confirmation, so the stuck guard fires ~2.8× as often even with B=0, which costs MISS
share and SR. Build the fix:
1. New module rounds/r04/k7_guard/judge.py: a subclass of the K1 BlindMixedJudge (do not edit K1 files) with an
   option (e.g. `stuck_guard="vision_confirmed"`) whose stuck count is exactly the stock definition whenever every
   decision in the window has vision (so with B=0 it must reproduce stock MixedJudge guard flags, reasons, stuck_n and
   V7 confidence bit-for-bit on the same inputs), and which, across blind gaps, only counts low-motion decisions as
   stuck when visual stillness is confirmed by the vision decisions that bound them (e.g. the key cosine between
   consecutive vision anchors against the same library 95th percentile; design and justify the exact rule; never
   invent a key for a blind decision). Keep every other guard, the span no-progress guard and the blind stepping
   unchanged. Also consider whether the V7 `stuck` feature should use the same count (keep the library calibration
   stock).
2. Offline evidence: using the existing offline harness and K1's frozen-path / replay machinery
   (rounds/r04/k1_blind/README.md, run_smokes.py, checks.py; ideation_A's scripts), report per-cell stuck firing
   rates for stock MixedJudge, K1 dense, and your variant on the same decision streams (π0.5 l10 and spatial, both
   library scales 50/500), plus the B=0 parity against stock MixedJudge (all guard fields, verdict inputs, V7
   confidence). If the historical closed-loop logs contain enough inputs (check /home/weiland/trace_runs/
   os_closed_loop/r03_mx/ and r04_blind/ decision logs; they log extras and robot_state but not keys), use them only
   where they suffice and say so.
3. Plugin-level verification: run your judge through the existing plugin blind selftest machinery
   (rounds/r04/k1_blind/run_plugin_blind.sh, rounds/r04/k6_concurrency/concurrency_test.py as a model) with
   `--os-blind` and `--os-judge guard_only`, including concurrent connections, and show decisions match a serialized
   reference.
4. Arm specs rounds/r04/k7_guard/arms_k7.json (emit_arms format, `<RUN>` fit placeholders, full model, `--os-blind`,
   `--os-judge guard_only`, `cost_ledger: true`): π0.5 l10 500-library B=0 (adapter control, expected ≈ stock g500),
   500-library phase_particles B=2 and B=1, 50-library phase_particles B=2; and spatial 500-library B=2. Base kwargs as
   in the K1 rows r4b3_p_l10_500_ph2g / ph1g / r4b3_p_l10_50_ph2g (lib big kref 8 at 500, lib current kref 5 at 50,
   gates "all"). Give the exact prefit command per arm (plugin --os-fit-artifact); run the prefits yourself into
   /home/weiland/trace_runs/os_closed_loop/r04_k7/fits/ (CPU only) and report sizes and SHA256.
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 4 of an action-cache exploration for VLA policies on
LIBERO: a HIT serves a retrieved/synthesized cache chunk, a MISS runs the full policy; guard-only control forces a
MISS when a guard fires; the R4 blind path serves some decisions without running the vision encoder, bounded by a
budget and look gates. Read first: exp/offline_search/rounds/r04/CODING_BRIEF.md (binding rules and the
method<->plugin contract), rounds/r04/k1_blind/HANDBACK.md and README.md, rounds/r03/h3_judge/judge.py (stock guards),
rounds/r04/ideation_A/REPORT.md §1–§3, rounds/r04/k6_concurrency/HANDBACK.md (current plugin concurrency contract:
methods are deep-cloned per connection; fitted arrays are read-only).
You own only the new directory rounds/r04/k7_guard/ and the fit directory /home/weiland/trace_runs/os_closed_loop/
r04_k7/fits/. Closed-loop chains are running on the shared plugin right now; you change no shared file.
Store: /home/weiland/trace_runs/offline_search_store (cold copy; /dev/shm may not be visible).
Python: /home/weiland/projects/openpi/.venv/bin/python from the repo root.
</context>

<hard_constraints>
- Your CPU range: 26-29,70-73. Prefix every python/numpy command with `taskset -c 26-29,70-73` and set
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never more processes than threads in your range.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: none. CUDA_VISIBLE_DEVICES=''.
- Edit only files you own. Do not modify src/, harness/, profile/, closed_loop/, rounds/r01..r03/, or other agents'
  rounds/r04/k1..k6 directories.
- No git. No `rm -rf`. Never `pkill -f`; kill only processes you started, by PID. No timan107, no LIBERO workers, no
  closed-loop chains, no servers, no ports. Do not read tests/review_tests/.
</hard_constraints>

<completeness_contract>
Resolve the task fully: implementation, B=0 bit parity against stock, firing-rate evidence at both library scales,
plugin-level blind and concurrent checks, arm specs, prefits, hand-back. Check edge cases: step 0, first decision after
a MISS, a blind gap at episode start, task change, GR00T refusal or support stated explicitly.
</completeness_contract>

<verification_loop>
Before finishing, re-run your parity and plugin checks against the final files and report the final numbers.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; give exact commands, numbers and file paths. Label anything unverified.
</grounding_rules>

<structured_output_contract>
Write rounds/r04/k7_guard/HANDBACK.md (the exact rule and its justification; parity and firing-rate tables; files with
sha256; arm specs and prefit artifacts; what the coordinator must do next; caveats), then print a one-paragraph summary.
</structured_output_contract>
