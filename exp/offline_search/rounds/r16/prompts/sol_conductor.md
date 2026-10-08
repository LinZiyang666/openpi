<task>
R16 is too slow. The owner: "176组要跑到猴年马月；server 有 yaml 热切换功能，conductor 能把任务细致到 episode 级别发放，为什么不用". Today every arm is its own root: plan → sync → start fresh servers bound to one method → run 300 episodes → collect → stop, with restart overhead and an idle tail per arm (~11–15 min per 300-episode arm on the local 4090 + timan107). Rebuild the R16 dev (and later test-A) execution so that servers stay up and workers never wait. You built R16 (exp/offline_search/rounds/r16/, HANDBACK.md, RUNBOOK.md); keep all R16 decisions (COORDINATOR_DECISIONS.md D1–D7) and the frozen protocol.
</task>

<requirements>
1. Multi-method servers with YAML hot switch: one local server process holds every method of a dispatch group resident (the R10/R11/R14/R15 frozen fits + level kwargs per arm); each connection selects its arm's method by yaml_id / bundle via the existing `load_cache_config` hot switch (src/openpi/serving/websocket_policy_server.py) or a minimal default-off plugin extension (exp/offline_search/closed_loop/plugin.py PluginRuntime currently binds one method at startup). Group by k (GR00T's live head K is fixed at launch), so servers restart only between k groups / models. Per-episode decisions and actions must be identical to the single-method path for the same arm and episode identity (CPU replay proof; seeds, identities, per-connection method copies and logs unchanged; each decision log row must still carry its arm).
2. Episode-level dispatch with the existing conductor (src/openpi/conductor/{driver,agent,strategy}.py, docs/experiments/conductor_tutorial.md): one global queue of (arm, task, init) episodes across all arms of a group; timan107 workers pull the next episode. Use the existing worker/client stack (same horizon, seed, manifest, recording off).
3. Adaptive descent: the R14 protocol decides each task by first failure (ir45 → ir40 → ir32 → ir25 → off; 30 dev episodes per task-level against the full-step dev C10). Dispatch a task's next level only after its current level's 30 episodes complete and pass; stop dispatching a task after its first failure. The per-task tables must equal those from running every level on every task. Uniform per-cell tables are dropped (record this; owner speed request). k-step pure C10 dev arms are still needed where the fallback requires them; reuse the 8 already completed (r16_dev_c10_*_50_*) and every completed arm (verify by hash).
4. Outputs consumable by `exp.offline_search.rounds.r16.analyze dev/freeze` and the test emitter (adapt analyze if the layout changes; keep the frozen design hashes honest — record any amendment). Test A: the 32 frozen per-task arms run through the same multi-method/episode dispatch (no adaptivity), 500 episodes each.
5. GPU budget on the local 4090: π0.5 3 servers (measured ≈9.0 GB steady, 9.9 GB peak each), GR00T 4 servers (≈6.0 GB each); resident fits must stay inside that. Report per-server memory for the resident method set.
6. Coordinator-run real-environment smoke commands (one small group, few episodes per arm, both models) with what to check, then the full commands. A driver/runner I can start in tmux, resumable after interruption (completed episodes kept, in-flight ones re-dispatched).
7. HANDBACK_CONDUCTOR.md: what changed, proofs, expected speedup estimate, commands, risks.
</requirements>

<constraints>
CPU only: taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python. No servers, GPU, tether/remote operations, launches, git, pkill, rm -rf. A coordinator line is currently running R16 dev arms on the 4090 and timan107 — do not touch its processes, roots in flight, /tmp/r8_coord logs or openpi_ext/scratch/r16_ops/queue.txt. Shared runtime edits must be default-off, atomic and tested. Test A is measure-only. Refit nothing.
</constraints>

<default_follow_through_policy>
Do not stop for routine questions; choose what is most faithful to R14/R15 and record it. Stop and report only if per-episode equivalence or the frozen protocol cannot be kept.
</default_follow_through_policy>
