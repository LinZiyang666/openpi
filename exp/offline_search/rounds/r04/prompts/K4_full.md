<task>
You are R4 coding agent K4: evaluation infrastructure and the low-IR frontier specification.
Implement in the files you own (closed_loop/ops/*, ops/remote/*, rounds/r04/k4_eval/):
1. emit_arms.py: generic per-arm yaml patching (`yaml_patch` dict merged into the emitted arm yaml; needed for MISS
   `miss: {num_steps: K}`), per-arm client overrides carried into arms.json and the timan107 matrix (`replan_steps`
   L for the longer-chunk baseline; keep 5 as default), pure-inference arms (every decision MISS; judge
   `periodic:1` on a plain method is acceptable — or a cleaner equivalent), and seed replicates (distinct arm names;
   server seeds differ per process).
2. ops/remote/run_arm.sh and run_gtp_subset.py: accept a manifest file of exact (task_id, episode_idx) pairs (ideation C
   REPORT.md proposal 2; manifests already exist in rounds/r04/ideation_C/pilot_manifest_*.json) and the per-arm
   replan override; chain.sh: EXPECT = number of distinct manifest pairs; manifest-aware DONE markers so a smaller
   manifest never skips a larger run. Keep the old OSCL_EPISODES/OSCL_TASKS behaviour unchanged.
3. collect.py and kpi.py: the cost ledger of CODING_BRIEF.md (vision share v, MISS share m, K per MISS, L per request,
   stage costs from closed_loop/ops/cost_table.json when present, owner constants otherwise; IR per five control steps;
   controls/episode) using the log fields K2 adds (tolerate their absence in old logs: old arms must reproduce their
   recorded IR exactly); weighted estimates for manifest (stratified) pilots: inclusion probabilities from the manifest,
   weighted SR and weighted paired ΔSR with design variance (ideation C REPORT.md §2.2); keep all existing outputs.
   Validate on the R2/R3 run roots under /home/weiland/trace_runs/os_closed_loop/ (read-only): old SR/IR reproduce.
4. rounds/r04/k4_eval/arms_frontier.json: emit_arms spec rows (placeholders `<RUN>`) for the first-batch arms of
   SELECTION.md (pure-inference seed replicates π0.5 l10/sp; 500-library l10 guard noprog 4, periodic 8/12; spatial 500
   guard-only and periodic 12; 50-library l10 periodic 6 and guard noprog 4), the longer-chunk baselines (L=10 with K10
   and K2) and the K2-only pure inference, with prefit notes (existing R2/R3 fit pickles can be reused when
   spec/kwargs/cell match).
Do not run anything on timan107; the coordinator pushes ops/remote/* and runs the smoke.
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
- Your CPU range: 30-33,74-77. Prefix every python/numpy command with `taskset -c 30-33,74-77` and set
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never more processes than threads in your range.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: none. Run with CUDA_VISIBLE_DEVICES=''.
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
Write rounds/r04/k4_eval/HANDBACK.md (files changed/added with install times; exact switches/arm-spec fields; tests
and their numbers; what the coordinator must do next; caveats), then print a one-paragraph summary as your final
message.
</structured_output_contract>
