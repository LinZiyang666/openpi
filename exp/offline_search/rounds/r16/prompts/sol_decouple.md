<task>
Owner request (2026-10-08 09:3x): decouple CPU work from GPU work at the framework level, so serving throughput is no longer capped by "one Python thread per GPU-model process". Evidence from R16 on weilandserver (44 cores / 88 threads, one RTX 4090): with the cache doing most decisions, each resident GR00T server process sits at 100% of one core while GPU utilization is 29–55% and machine load is a few cores; adding processes scales throughput (4 → 6 GR00T processes: 38 → 58 episodes/min), but every process holds a full model copy (GR00T ≈5.7–6.5 GB, π0.5 ≈9–10.5 GB), so GPU memory, not CPU, limits concurrency. Build a framework-level split: many CPU front processes without model weights do all per-connection CPU work; one (or few) GPU model process(es) per model hold the weights and serve batched model stages.
</task>

<where>
- Work ONLY in the git worktree /home/weiland/projects/openpi_decouple (branch `decouple-cpu-gpu`, from 05a32800). Use the main tree's interpreter: `/home/weiland/projects/openpi/.venv/bin/python` with `PYTHONPATH=/home/weiland/projects/openpi_decouple:/home/weiland/projects/openpi_decouple/src` and cwd the worktree. R16 code that is untracked in the main tree (exp/offline_search/rounds/r16/, e.g. resident.py, conductor*.py) may be read from /home/weiland/projects/openpi and copied into the worktree as needed.
- Do NOT modify anything under /home/weiland/projects/openpi: an R16 test-A run is serving from that tree right now and starts new server processes from it.
</where>

<scope>
1. Map the current serving paths first (short design note): `src/openpi/serving/websocket_policy_server.py` (concurrent mode, per-connection policies, `load_cache_config` / bundle hot switch), `src/openpi/cache/*` (interceptor, orchestrator, BatchingCoordinator, run_stage1/2/3), `scripts/serve_policy.py`, `exp/offline_search/closed_loop/{plugin.py,serve_pi05.py,serve_groot.py}` (offline_search methods, retrieval/corrector CPU work, decision logs), the GR00T server path (`exp/libero_groot/serve_groot_libero.py` and the GR00T venv), and R16's resident multi-method server. Measure (CPU profiling on recorded B-pool inputs, no GPU) where per-decision CPU time goes for both models: transforms/preprocessing, tokenization, retrieval/corrector, serialization, logging, model-stage CPU glue.
2. Design and implement a default-off split: CPU front processes (N configurable, can exceed GPU-process count by a lot) own connections, per-connection state, offline_search method copies and logging; GPU model process(es) own weights and execute model stages (π0.5 stage 1/2/3 incl. k-step stage 3; GR00T backbone + action head incl. k steps) for batched requests from all fronts via an efficient local IPC (shared memory for tensors/images; no pickling of large arrays per call). Keep the existing websocket protocol, YAML hot switch and conductor episode dispatch working unchanged from the worker side.
3. Equivalence: with the split enabled, every decision and action must equal the current single-process path for the same inputs (byte-identical where the current path is deterministic; document and bound any batching-induced numerical difference, which the current concurrent BatchingCoordinator already has). Prove on CPU with model-stage stubs and recorded inputs, both models, interleaved connections, hot switches, k-step MISS, blind tails; plus failure paths (front crash, GPU process restart, backpressure).
4. Throughput model and benchmark plan: CPU-only microbenchmarks of the front path; a coordinator-run GPU benchmark script (single 4090) comparing current vs split at equal GPU memory, reporting episodes/min, GPU utilization, CPU utilization, per-decision latency p50/p99.
5. Deliverables in the worktree: implementation, tests, `docs/` or `exp/.../DECOUPLE.md` design note, `HANDBACK_DECOUPLE.md` (what changed, proofs, how to enable, smoke and benchmark commands for the coordinator, risks). Commit nothing; leave changes in the worktree.
</scope>

<constraints>
CPU only: `taskset -c 10-17,54-61 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. No servers, GPU, tether/remote operations, git commit/push, pkill or rm -rf. Do not touch running processes, /tmp/r8_coord, /home/weiland/trace_runs (read-only for recorded inputs is fine), or the main tree. Shared-runtime changes must be default-off and keep the existing paths byte-for-byte when disabled.
</constraints>

<default_follow_through_policy>
Do not stop for routine questions; record decisions in the handback. Stop and report only if per-decision equivalence cannot be kept or the design needs an owner decision (state the options and your recommendation).
</default_follow_through_policy>
