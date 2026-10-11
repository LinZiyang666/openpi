<task>
Build the formal pure-inference SR campaign launcher inside the R17 framework, so that formal evals use the conductor's per-episode dispatch and resident concurrent server replicas instead of stock `examples/libero/main.py` against a single lock-serialized server. The owner's complaint (verbatim): "你的conductor呢，concurrent server呢，怎么可能这么慢".

Priority 1: DP-LIBERO formal pure inference.
- Checkpoint: `/data/mw_finetune/dp/libero_policy_100000` (exported, pulled locally).
- Arm: DP100 full, no cache, debug capture off.
- Formal set from `exp/r17_integration/qualification/specs/libero_benchmark_spec.json`: 40 tasks × original init 0..9 = 400 episodes. Also support a flag for the official 50-trials/task protocol (init 0..49, 2000 episodes) as a separate campaign.

Priority 2: the same launcher generic over the integrated bundles: dp_mw, fastwam (libero/mw), groot_mw, pi05_mw. The coordinator will use it next for DP-MW MT50 (500) and Fast-WAM MT50 after training ends (about 22:05, H100).
</task>

<design_requirements>
- Server: N resident replicas of the model's R17-qualified serve entry, each on its own backend port, behind the stock `openpi.serving.replica_proxy.run_proxy(public_port, backend_ports)`. This is the same shape as `slot_G2_dp_libero_R_replicas`.
  - DP replicas must pass `--cuda-graph`, which sol E just added. It is bit-exact and self-checked at startup; see `exp/mw_finetune/dp/HANDBACK_E.md` § "CUDA graph". Measured: 1765 → 405 ms per LIBERO infer.
  - N and the GPU are parameters. Default for DP on the local 4090: N=12, about 1 GB each; budget ≤ 14 GB because π0.5 MT50 K evals also hold about 8 GB on the 4090.
- Coordinator measurement on the 4090 (eager, while other evals ran): aggregate DP throughput was 0.59 / 2.15 / 3.46 infer/s at 1 / 4 / 8 processes. Replicas therefore scale.
- Conductor: the R17 conductor driver (`exp.r17_integration.conductor.driver`) with the sealed benchmark spec and the model's sealed manifest. Per-episode dispatch, with W workers on timan107 (EGL, `HOME=/home/zixuans8`, PYTHONPATH as in R16's `conductor_runner.py`) and optionally timan108 (H100 lane uses 149.165.153.233). Default W=24 on timan107. Workers may exceed replicas; the proxy is least-connections sticky.
- Endpoint: public `ziyanglin.com:<port>`. Use free ports 23160–23179 and avoid 23150–23155 / 23250–23255, which are in use by live evals.
- Sync back with the existing SHA-checked `/tmp`-stage transfer, then aggregate. Output per-suite SR and the 40-task mean, with error/retry counts. Results go under `/data/mw_finetune/eval/<run>/`.
- Lifecycle: own all processes (server replicas, proxy, remote driver/workers) with receipt-based cleanup by PID/pgid. No broad pkill.
</design_requirements>

<constraints>
- Do NOT touch the running π0.5 MT50 K eval (port 23150, timan107 tmux r17_pi05) or the Fast-WAM training on H100.
- Do not modify `src/`. Prefer a new file under `exp/r17_integration/ops/` (e.g. `formal_eval.py` plus a thin shell wrapper).
- You may run a smoke (2 workers, about 4 episodes, N=2 replicas) on the 4090 to prove the full path. Do NOT start the full 400/2000 campaign; the coordinator launches it.
</constraints>

<verification_loop>
Smoke must show: (a) replicas pass the CUDA-graph self-check and the proxy serves; (b) conductor dispatches episodes to ≥2 workers that hit ≥2 distinct backends; (c) results sync back and aggregate into a per-suite summary JSON; (d) all owned processes are cleaned up.
</verification_loop>

<compact_output_contract>
Append "## Formal launcher" to `exp/r17_integration/qualification/HANDBACK_Q.md` containing: exact coordinator commands for (1) DP-LIBERO formal400, (2) DP-LIBERO official2000, (3) DP-MW MT50 500, (4) Fast-WAM MT50 500 on H100 + timan108; smoke evidence paths; expected throughput from the smoke. Final message: 6 lines max.
</compact_output_contract>
