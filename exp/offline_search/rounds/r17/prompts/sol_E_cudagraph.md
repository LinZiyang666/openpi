<task>
Add an opt-in CUDA-graph replay of the DP denoiser to `exp/mw_finetune/dp/` to speed up LIBERO/MetaWorld pure-inference evaluation. It must be bit-identical to the current eager path.

Measured on the 4090 (coordinator, 21:10): one `Policy.infer` (100 DDPM steps, B=1) launches about 74,000 CUDA kernels (about 740 per denoising step). GPU kernel time is only about 258 ms, but wall time is 1.7–2.4 s. The workload is launch-bound. The running DP-LIBERO eval (4 servers on ports 23152–23155, 4 suites × 500 episodes) would take more than 20 h at this rate.

Scope:
1. In `policy.py` / `model.py`, add an option (default OFF; e.g. `Policy(..., cuda_graph=False)`) that captures `core.model(trajectory, t, cond)` once per shape into a `torch.cuda.CUDAGraph` with static input buffers (trajectory, timestep, cond) and replays it inside the 100-step loop. The DDPM `scheduler.step` (with the per-call generator noise) stays eager and unchanged. Capturing the whole loop is also fine if you can keep the generator noise bit-identical; the denoiser-only version is acceptable.
2. Add a startup self-check: when the option is on, run eager vs graph on at least 3 fixed random observations (different seeds and prompts) and require `torch.equal` on the final actions. If the check fails, refuse to serve: raise, do not silently fall back.
3. Expose the option as a flag `--cuda-graph` in `serve_libero.py`, and in the MetaWorld DP serve entry if there is one. Record `"cuda_graph": true/false` in server metadata. With the flag off, behaviour, metadata and default path must stay exactly as they are now.
4. Measure eager vs graph wall time per `infer` on cuda:0 (B=1, 100 steps) and report both numbers.
</task>

<constraints>
- Do NOT touch, restart or kill the running DP-LIBERO servers (ports 23152–23155) or any other process. Do not edit `run_eval_dp_libero.sh`; the coordinator handles relaunching.
- GPU: short tests on cuda:0 only (<2 GB, a few minutes). The 4090 is shared with live evals.
- Do not change `src/` or anything under `exp/r17_integration/` except what DP serving strictly needs (ideally nothing).
- Keep the existing CPU tests passing; add a small CPU-skippable test for the self-check logic if cheap.
</constraints>

<verification_loop>
Before reporting done: (a) default-off path produces identical actions to before (compare against the eager path); (b) graph path passes `torch.equal` on ≥3 observations for the LIBERO checkpoint `/data/mw_finetune/dp/libero_policy_100000` and, if present, the MetaWorld DP checkpoint; (c) report eager ms vs graph ms.
</verification_loop>

<compact_output_contract>
Append a section "## CUDA graph (eval throughput)" to `exp/mw_finetune/dp/HANDBACK_E.md` with: files changed, the exact serve flag, bit-exactness evidence, and eager vs graph timing. Final message: 5 lines max.
</compact_output_contract>
