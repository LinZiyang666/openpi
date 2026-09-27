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
