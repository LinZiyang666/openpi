<task>
Prepare (CPU only, do not run on GPU) a batch-1 latency profile for the three new LIBERO models — DP, Fast-WAM, Cosmos — so the coordinator can later compute their owner IR coefficients exactly as was done for π0.5 and GR00T. The coordinator runs it on an exclusive RTX 4090 after the current experiments finish.

Method to copy (owner: "profile 采用 compile 和 cuda graph 编译，batch=1，跟 pi 和 groot 之前测 profile 时一样"):
- GR00T authority: `exp/libero_groot/config/rit/cost_groot_libero_measured.json` and its source cell `exp/libero_groot/data/latency/libero_cg_k8_p0_r0.json`, measured by `exp/libero_groot/serve_groot_libero.py` (measure mode) with `src/openpi/cache/groot/staged.py`: torch.compile `reduce-overhead` (CUDA graphs: stage1, stage2, and the stage-3 step replayed k times), batch 1, warmup 30, 200 iterations, medians by CUDA events, exclusive RTX 4090; certified with an nsys/CUDA trace (cudaGraphLaunch == iters × (2 + k), zero captures after warmup, zero cudagraph skips, compile_count recorded); eager-vs-compiled parity with fixed noise recorded.
- π0.5 authority: `exp/dispatch_surface/analysis/analytic_cost.py` (frozen CUDA-graph stage ms).
- Pricing (see `exp/offline_search/rounds/r16/cost.py`): vision coefficient = stage1 / (stage1 + stage2 + stage3_full); full MISS = (stage2 + stage3_full) / same denominator; a reduced-step MISS = (stage2 + stage3_reduced) / the same full-step denominator.
</task>

<stages_and_steps>
Use the same stage split the R17 server uses (`exp/r17_integration/common/serving.py`): stage1 = adapter `prepare` + `run_stage1` (what a cache HIT pays: the retrieval key), stage2 = `run_stage2`, stage3 = `sample`. Adapters: `exp/r17_integration/models/{dp,fastwam,cosmos}/staged.py`.
Measure the stage-3 step and the full loop at both step counts:
- DP: DDPM 100 (default) and DDIM 10 trailing eta 0 (owner default reduced, same ε network).
- Fast-WAM: K10 (default) and K1 (owner default reduced).
- Cosmos: K5 (default) and K2 including the clean step = 1 actual network evaluation (owner default reduced). Report actual NFE.
Inputs: real batch-1 observations at serving geometry (both cameras, state, prompt) — e.g. rows of the B library under `/data/r17_library/<model>_libero_<suite>/bpool` or a recorded episode; one spatial and one libero_10 prompt.
</stages_and_steps>

<deliverables>
1. One script (e.g. `exp/r17_integration/profile/latency.py --model dp_libero|fastwam_libero|cosmos_libero --out <json>`) that loads the real weights the way the formal server does, compiles the three stages with `reduce-overhead`, runs warmup 30 / 200 iterations, and writes a JSON with the same fields as the GR00T cell (stage ms medians and p10/p90, step and full-loop ms for both K, compile_count, capture_calls_after_warmup, cudagraph_skips, expected vs observed graph launches, eager-vs-compiled parity, GPU/driver/torch, checkpoint identity) plus the derived coefficients (vision, full MISS, reduced MISS).
2. A small wrapper that runs it under nsys (if nsys is installed on weilandserver; otherwise the torch cudagraph counters) to certify graph replay, mirroring GR00T's certification.
3. Check whether each model fits on the 48 GB RTX 4090 at batch 1 (Fast-WAM includes the 5B video DiT; say what must be loaded and whether the text encoder output can be precomputed as the server does).
4. CPU tests with stub modules for the timing/certification/pricing logic; no GPU use.
5. The exact commands the coordinator runs later on the exclusive 4090, and the expected runtime per model.
</deliverables>

<keep_it_simple>
Main function plus a few necessary checks. No gates, seals or admission layers. Do not change serving behaviour; compile only inside the profiling script.
</keep_it_simple>

<action_safety>
Edit only a new `exp/r17_integration/profile/` directory (and tests). GPUs and nodes are running experiments: do not use any GPU, do not start/stop processes or campaigns, no remote nodes. Do not git commit.
</action_safety>

<handback>
Reply with: files and what each does; how each model's three stages map onto the adapter calls; memory fit on the 4090; CPU test results; exact run commands and expected runtime.
</handback>
