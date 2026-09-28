<task>
You are R4 coding agent K5: the randomized CALL-versus-CACHE identification experiment (ideation C second pass,
proposal 1 `causal_rescue_credit`, in exp/offline_search/rounds/r04/ideation_C/REPORT_2.md §2; its reviewable spec is
rounds/r04/ideation_C/p2_causal_pilot.json if present). Implement the experiment side only — the randomized single-landmark
overlay, its logging, arm specs and the estimator — not a learned controller.
1. Plugin overlay (closed_loop/plugin.py, opt-in; absent flags ⇒ byte-identical behaviour and logs). At each episode
   reset, derive from a deterministic RNG keyed by (experiment seed, task id, init/episode index) the landmark class
   (first or third baseline MISS opportunity, p=.5 each) and the treatment (CALL or CACHE, propensity .5); a replicate
   flag flips the treatment for the same (task, init) so two replicate arms are complementary. Count baseline MISS
   opportunities per episode AFTER the ordinary judge verdict (guard-only MixedJudge: guards 1-4 force a MISS) and BEFORE
   the verdict is returned to the interceptor. Override only the assigned opportunity: CALL = run the full policy as
   usual; CACHE = serve that decision's ordinary cache proposal. The baseline controller resumes at the next decision.
   `on_executed` / history must see the actual verdict and action, so q.hist_hit, q.prev_a_exec, fresh/stale selection
   and all later guards stay truthful. At most one randomized decision per episode; episodes that never reach their
   landmark are unexposed and unchanged. Find out how the plugin learns the init/episode index of a LIBERO episode and
   make the assignment depend on the init, not on arrival order or connection (servers serve many workers
   concurrently; episodes arrive in any order; the same (task, init) must get the same class/treatment in every run of
   the same arm). Log per decision: trial_id, eligible, opportunity index, landmark class, assigned treatment,
   propensity, baseline verdict, actual verdict, and the context bins of REPORT_2 step 4 (progress <.5/≥.5, gripper
   sign, confidence >−.3/≤−.3, stall age before/0–9/≥10) captured before randomization; keep all R4 log fields
   (`--os-log-r4`) working, including `served_head`.
2. Estimator (rounds/r04/k5_rand/estimate.py): from two complementary replicate arms (plus optional baseline guard-only
   arms for the same inits) compute intention-to-treat and per-landmark/per-context CALL-minus-CACHE ΔY (success),
   ΔN (requests), ΔM (MISSes) by inverse-propensity means with init-clustered bootstrap intervals, the parent/child
   shrinkage n/(n+100) of REPORT_2 step 5, exposure/compliance/balance tables, and the cost term
   ΔCρ = .848·ΔM + (.152−ρ)·ΔN for ρ in {the g50 and g500 operating points}. Reads collect/kpi outputs or the raw
   decision logs + journals under a run root (the same layout as /home/weiland/trace_runs/os_closed_loop/r03_mx/).
   Validate it on synthetic data with a known planted effect (recovers it; coverage of the interval roughly nominal)
   and on a null (no effect).
3. Arm specs (rounds/r04/k5_rand/arms_rand.json, emit_arms format, `<RUN>` placeholders): π0.5 libero_10, guard-only
   MixedJudge exactly as the R3 arms r3mx_p_l10_g (50-episode library) and r3mx_p_l10_g500 (500-episode library) —
   reuse their fit pickles /home/weiland/trace_runs/os_closed_loop/r03_mx/fits/r3mx_p_l10_g.pkl and
   r3mx_p_l10_g500.pkl when method/kwargs/cell match exactly — × replicate {1, 2}: 4 arms, full model, `--os-log-r4`,
   `cost_ledger: true`. Also give a 2-arm smoke subset recipe (OSCL_EPISODES/OSCL_TASKS) the coordinator can run first.
4. Offline checks you can run without a GPU: assignment determinism across processes and arrival orders, balance of
   class/treatment over the 500 (task, init) pairs, that the override happens exactly once at the assigned opportunity
   (use the plugin's existing replay / fake-policy self-test machinery: closed_loop/selftest.py, replay_client.py,
   rounds/r04/k2_serving/, rounds/r04/k4_eval/run_plugin_selftests.sh, rounds/r04/k1_blind/run_plugin_blind.sh), that
   CACHE serves the ordinary proposal and CALL the policy path, that later guards see the true history, and that
   without the new flags every existing selftest and a before/after log diff are unchanged.
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 4 (R4) of an action-cache exploration for VLA policies
(π0.5, GR00T N1.5) on LIBERO: a HIT serves a retrieved/synthesized cache chunk, a MISS runs the full policy; each
decision executes 5 control steps; IR (owner basis, π0.5) = .152 per vision decision + .848 per MISS, normalized per
decision. The best l10 point so far is 500-library guard-only (SR .864 @ IR .238); 50-library guard-only is SR .740 @
IR .323. Read first: exp/offline_search/rounds/r04/CODING_BRIEF.md (binding rules: live-system rule, log fields,
cost ledger), rounds/r04/SELECTION.md, rounds/r04/ideation_C/REPORT_2.md (your specification, §2 proposal 1),
rounds/r04/k2_serving/HANDBACK.md and rounds/r04/k4_eval/HANDBACK.md (the current plugin and ops state),
exp/offline_search/closed_loop/README.md, rounds/r03/h3_judge/ (MixedJudge and its guards).
The coordinator is running closed-loop chains on this code RIGHT NOW: every ~13 minutes a new server process imports
closed_loop/plugin.py. You now own closed_loop/plugin.py, closed_loop/selftest.py, closed_loop/verify_logs.py and the
new directory rounds/r04/k5_rand/. Develop in rounds/r04/k5_rand/dev/, prove byte-identical behaviour without your
flags, and install each shared file in ONE atomic step (temp file in the same directory, then mv). Never leave a shared
file half-edited.
Store: /home/weiland/trace_runs/offline_search_store (cold copy; /dev/shm is not visible in your sandbox).
Python: /home/weiland/projects/openpi/.venv/bin/python from the repo root.
</context>

<hard_constraints>
- Your CPU range: 26-29,70-73. Prefix every python/numpy command with `taskset -c 26-29,70-73` and set
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never more processes than threads in your range.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: none. Run with CUDA_VISIBLE_DEVICES=''. The coordinator runs the live GPU smoke.
- Edit only the files you own (listed above). Do not modify src/, harness/, profile/, rounds/r01..r03/, other agents'
  rounds/r04/k1..k4 directories, ops/* or the cost table.
- No git. No `rm -rf`. Never `pkill -f`; kill only processes you started, by PID. No timan107, no LIBERO workers, no
  closed-loop chains, no servers on ports 23150-23169. Never touch ports, processes or tmux sessions you did not start.
  Do not read tests/review_tests/.
</hard_constraints>

<completeness_contract>
Resolve the task fully: implementation, offline verification, estimator validation, arm specs, and the hand-back.
Check edge cases: episode reset, step 0, a landmark that coincides with a guard-forced MISS, an episode that ends
before its landmark, concurrent episodes on one server, replicate flip, blind mode (`--os-blind`) either supported or
explicitly refused with a clear error.
</completeness_contract>

<verification_loop>
Before finishing, re-run every existing plugin selftest and your new tests against the INSTALLED files and report the
final numbers. If a check fails, fix and re-run.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; give exact commands, numbers and file paths. Label anything unverified.
</grounding_rules>

<structured_output_contract>
Write rounds/r04/k5_rand/HANDBACK.md (files changed/added with install times and sha256; exact flags / arm-spec fields;
tests and their numbers; the smoke recipe; what the coordinator must do next; caveats), then print a one-paragraph
summary as your final message.
</structured_output_contract>
