<task>
You are R5 coding agent Q3: solve and deploy the causal call-value table (rounds/r05/ideation_B/REPORT.md, Rank 2
`landmark_call_value`) from the completed randomized CALL/CACHE data (K5, rounds/r04/k5_rand/).
Data (500 init clusters × 2 complementary replicates per scale, π0.5 libero_10, guard-only MixedJudge, K10 MISS):
/home/weiland/trace_runs/os_closed_loop/r04_k5/ arms r4k5_p_l10_g500_{r1,r2} and r4k5_p_l10_g50_{r1,r2}; the
coordinator's estimator outputs are k5_g500_estimate.json and k5_g50_estimate.json there. Measured ITT (CALL minus
CACHE at the single randomized landmark): g500 ΔY=+.034 [+.004, +.066], ΔN=−1.58, ΔM=−.24 (calls pay for themselves);
g50 ΔY=−.002 [−.040, +.032], ΔM=+.47, ΔC(ρ=.323)=+.41 [−.45, +1.20] (a call buys nothing measurable).
1. Offline (first; no shared-file edits): run ideation B's `cost_solver.py:causal_fit` (read it; adapt only in your own
   directory) on each scale with its predeclared rules (support ≥30 init clusters and ≥10 per treatment per context,
   SR-loss tolerance .01, cross-fitting on init-mod-5 folds with held-out IPW evaluation, Bonferroni bounds). Report per
   context and parent landmark: supported or not, suppression probability, held-out cost saving and SR change with
   intervals. If no context is supported at a scale, that scale's answer is "baseline CALL everywhere" — accept it.
   Also report whether the sign flip between scales survives a task-held-out sensitivity analysis.
2. Deployment (only for a scale with a supported, held-out-positive table): an opt-in plugin path (e.g.
   `--os-call-table <file>`) that applies the frozen table at the same single assigned landmark per episode as K5
   (reuse K5's identity/assignment/context definitions; deterministic per (task, init) policy coin), serving the
   already-computed cache proposal on suppression and logging baseline vs actual verdict and the table hash. PLUGIN
   OWNERSHIP: agent Q5 may be installing into closed_loop/plugin.py; do NOT install until
   rounds/r05/q5_gpu/HANDBACK.md exists, then rebase on the installed plugin, re-run every existing selftest matrix
   (K2, K1, K4, K5, K6, K7, K10, Q2) and install atomically. If Q5 is not done when you are otherwise finished, leave a
   ready-to-install candidate plus a preimage-checked install script.
3. Arm specs rounds/r05/q3_callvalue/arms_q3.json for each deployable scale: baseline guard-only, frozen table, and an
   equal-number-of-overrides randomized control (as B specifies), π0.5 l10, full 500 inits.
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Read first: exp/offline_search/rounds/r05/CODING_BRIEF.md
(binding rules), rounds/r05/SELECTION.md, rounds/r05/ideation_B/REPORT.md (+ cost_solver.py), rounds/r04/k5_rand/
HANDBACK.md (+ estimate.py, overlay.py), closed_loop/plugin.py. You own rounds/r05/q3_callvalue/ (and, after Q5, the
plugin install described above). Closed-loop runs are read-only for you.
Python: /home/weiland/projects/openpi/.venv/bin/python from the repo root.
</context>

<hard_constraints>
- Your CPU range: 26-29,70-73. Prefix every python command with `taskset -c 26-29,70-73`, BLAS/OMP threads 1, never more
  processes than threads. CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: none. No git, no `rm -rf`, never `pkill -f`, no servers/ports/timan107/LIBERO workers/chains. Do not read
  tests/review_tests/. Write only under rounds/r05/q3_callvalue/ and /tmp/q3_*, plus the atomic plugin install above.
</hard_constraints>

<completeness_contract>
Resolve fully: fits and held-out evaluation at both scales, the deployment decision per scale, implementation and tests
if deployable, arm specs, hand-back.
</completeness_contract>

<verification_loop>
Re-run the fitter's held-out evaluation and (if implemented) all plugin selftests against the final files.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; exact commands, numbers, file paths. A table fitted from these rollouts is
borrowed big-library information — label it.
</grounding_rules>

<structured_output_contract>
Write exp/offline_search/rounds/r05/q3_callvalue/HANDBACK.md, then print a one-paragraph summary.
</structured_output_contract>
