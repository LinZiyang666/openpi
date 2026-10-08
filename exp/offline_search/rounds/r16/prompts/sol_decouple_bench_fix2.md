<task>
GPU window report for your package (worktree /home/weiland/projects/openpi_decouple; same CPU-only rules: no GPU, servers, remote launches, git commit, pkill, rm -rf; do not touch /home/weiland/projects/openpi or r16_* roots). The real 4090 run exposed four defects your CPU validation missed; I patched three by hand to keep going:
1. `assets/` (gitignored norm stats) missing in the worktree → I symlinked `/home/weiland/projects/openpi_decouple/assets -> /home/weiland/projects/openpi/assets`.
2. `qualify_gpu.py` imported `openpi.serving.batching_coordinator` (needs jax) unconditionally; the GR00T venv has no jax → I moved that import into the π0.5 branch.
3. `package_driver.py:119` launched the timan107 worker fleet with `attempts=1`; tether exec times out at 30 s regularly → I removed `attempts=1` (default 4 retries, like the R16 runner; the worker fleet lock refuses duplicates).
4. NOT fixed: arm `groot_initial_current_1` completed all 300 accepted episodes (`/home/weiland/trace_runs/os_closed_loop/decouple_bench_20261008_132449_a32ab1_groot_initial_current_1_try01`, MEASUREMENT_END accepted=300), then failed with "benchmark has no complete episodes/latencies/utilization samples": every row in `latencies.jsonl` has `uid: null, attempt: null`, so the accepted-cohort filter in `benchmark_gpu.py` (~line 204) drops all latencies.
All four gates passed bit-exact on the 4090 (results under `exp/decouple/results/gpu_comparison/`).
Needed:
a. Fix latency attribution properly (relay learns uid/attempt from the frames or connection metadata the worker actually sends, or timestamps rows and filters by the measurement window); fold fixes 1–3 into the package properly (asset check in preflight, not a manual symlink; review my two edits).
b. A re-analysis command that computes the result of an already completed arm from its raw files without rerunning it; run it on the arm above and report its episodes/min, GPU/CPU utilization, peak VRAM and latency p50/p99.
c. Resume semantics: a new package must not rerun gates whose inputs are unchanged and already passed on this GPU.
d. A real-environment smoke mode for the next window: gates skipped if already passed, then ~20 episodes each of GR00T current/1 and split/8, end to end through collection and SUMMARY, in under 10 minutes; plus a CPU end-to-end dry test that feeds recorded worker frames through the relay and collector so defects like 4 are caught without a GPU. Then regenerate config/hashes, update GPU_PACKAGE.md and the handback with the smoke command, the full command and expected duration.
</task>
