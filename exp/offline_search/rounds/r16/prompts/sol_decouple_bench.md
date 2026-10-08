<task>
Follow-up on your decoupling handback (same worktree /home/weiland/projects/openpi_decouple, same CPU-only rules). Turn the GPU gate and throughput benchmark into a package I can start with one command in a tmux shell once the 4090 is free (after R16, ~13:00; I will hold the whole 48 GB card for this window). Everything you do now is CPU-only preparation; no servers, GPU, remote launches.
</task>

<scope>
1. `exp/decouple/run_gpu_package.sh` that runs, in order, stopping at the first failure with a clear marker line (`PKG_GATE_FAIL`, `PKG_ARM_FAIL <arm>`, `PKG_DONE`):
   a. Numerical gate `qualify_gpu` at batch width 1 for π0.5 k=1,2 and GR00T k=1,2 (pi05_libero_pytorch, /data/ckpt/n15_libero_spatial with the island PYTHONPATH). Must be bit-exact.
   b. Throughput comparison with `benchmark_gpu`, GR00T first (it is the CPU-bound one), then π0.5. Arms per model at one common VRAM budget: current path 1 process; current path with as many processes as the budget fits (GR00T measured 5.7–6.5 GB each, π0.5 9–10.5 GB each); split with one GPU process and N = 4, 8, 16 fronts at batch width 1. Same fresh B-pool episode manifest for every arm of a model (dev pool, not test A), same frozen R16 method (pick one representative cell per model, e.g. spatial 100 k1 at its frozen level), 64 timan107 workers, warmup excluded. One pass per arm, ~300 accepted episodes each; then 2 more alternating paired repeats of the best current arm vs the best split arm. Fresh roots named `decouple_bench_*` under /home/weiland/trace_runs/os_closed_loop; ports and driver port outside 23100–23179.
   c. A summary script that writes `exp/decouple/results/gpu_comparison/SUMMARY.md`: per arm episodes/min, GPU util, host CPU util, peak VRAM, per-decision latency p50/p99, episode failures, and the paired best-vs-best ratio with spread.
2. Generate `exp/decouple/benchmark.local.json` (or a generator) from the real R16 resident/conductor/worker argv so that no hand-editing is needed. Reuse the deployed worker island /scratch/zixuans8/openpi_r16_conductor_v1 on timan107 if its bytes suffice; if worker bytes must change, say exactly which files and give the deploy+sha256 check commands instead of deploying.
3. CPU dry-run: validate the generated config and all argv/paths (bash -n, --help, config schema, manifest identity), without launching anything. Estimate total GPU wall time; keep it under ~2 h.
4. Update HANDBACK_DECOUPLE.md with a short "GPU package" section: the one command, expected duration, outputs, and how to stop/resume.
</scope>

<constraints>
CPU only: `taskset -c 10-17,54-61 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Do not touch /home/weiland/projects/openpi (R16 retests will run from it), any r16_* run roots, running processes, or timan107. No git commit, pkill, rm -rf.
</constraints>

<default_follow_through_policy>
Do not stop for routine questions; record choices in the handback. Stop only if the benchmark cannot be made runnable without an owner decision.
</default_follow_through_policy>
