<task>
Owner request, 2026-10-17 23:0x. Establish whether our model deployment, including the cache system, is equivalent across GPUs. If it is not, find the operations that restore equivalence.

Background: the owner once ran a server on an A100 and its success rate dropped substantially, possibly from kernel or architecture precision differences. A new A100 (public 149.165.152.232) has arrived but is NOT yet approved as an experiment device. Do not connect to it, probe it or plan anything that touches it until the coordinator says so. Our current GPUs are the RTX 4090 (local, Ada) and the H100 (Hopper). Cross-GPU equivalence between these two also matters now: we want to balance lanes across both GPUs.

Phase 1, now:
1. Find the historical A100 run(s) and the SR drop: model, suite, config, environment versions, numbers vs the reference GPU. Look in `/home/weiland/projects/openpi/logs/` (handoff, *plan.log.md), `dist_experiment_control/docs/devices.md` §2.4 (old a100 149.165.152.105), RUN_CONFIGS, and rounds reports. Cite file:line. Say clearly if no reliable record exists.
2. Inventory per model (π0.5, GR00T, DP, Fast-WAM, Cosmos) the numerics-relevant settings in our frozen serving runtimes:
   - TF32 flags (torch matmul/cudnn, `set_float32_matmul_precision`) and JAX default matmul precision;
   - dtype/autocast (bf16/fp16/fp32);
   - `torch.compile` / max-autotune (e.g. openpi `pytorch_compile_mode` default "max-autotune");
   - attention backend (SDPA flash / mem-efficient / math, flash-attn builds);
   - cudnn deterministic/benchmark flags, `CUBLAS_WORKSPACE_CONFIG`;
   - library/driver versions on each host.
   Mark which of these differ by architecture (Ampere/Ada/Hopper) and which plausibly explain an SR drop.
3. Define the equivalence criteria, in three layers, with concrete thresholds and the noise floor each is compared against:
   - (a) Open-loop numerics on identical recorded observations and identical seeded noise: per-stage tensors (prefix embeddings, each denoising step, final action chunk), max-abs/relative error vs same-GPU repeat drift and vs the drift we already accept (RUN_CONFIGS batching drift, e.g. 0.0156 normalized max).
   - (b) Cache-decision agreement on identical queries: hit/miss, top-1 neighbour, corrector/stall decisions; the agreement rate and the effect near thresholds.
   - (c) Closed-loop paired SR on the same inits and seeds: paired difference CI within a pre-registered margin, McNemar, and the first-divergence step distribution.
4. Build the tooling, CPU-testable:
   - a recorder that captures real observations and seeds from an existing run's per_step logs, or a tiny capture run;
   - a replay harness that runs a given serving runtime on those inputs and dumps per-stage tensors;
   - a comparator that produces the layer (a)/(b) report.
   Reuse existing parity tools where they exist (e.g. `exp/r17_integration/models/*/parity.py`, gpu_diagnostic.py).
5. Write exact GPU commands for the coordinator:
   - a 4090-vs-H100 layer (a)+(b) check, a few minutes each and small GPU memory, runnable beside the running lanes or at a lane boundary;
   - a layer (c) pilot design: episodes per suite, GPU time.
   - Then the A100 plan to run once approved, including which numerics knobs to toggle (TF32 off, highest matmul precision, compile off or fixed, same attention kernel, deterministic flags) and in which order, to isolate the cause.
</task>

<rules>
- CPU only locally. No GPU runs yourself: hand GPU work >2 min to the coordinator as exact commands. tether exec read-only on h100/timan107/timan1 only. Never touch the new A100. Never push, pull or expose.
- Do not modify running runs or chains. Do not edit files other agents are changing: sol is editing `exp/r17_integration/common/serving.py`, `rounds/r17/longseg_1017/*`, `filter_failed_knob/formal_a_1016/*`, `filter_failed/CONTINUATIONS.json`; astra is writing `rounds/r17/autoscale_1017/*`. Put your work under `exp/offline_search/rounds/r17/xgpu_1017/`.
- Do not saturate /data: no parallel bulk reads; sample large logs; check `/proc/pressure/io`.
- No rm, no git commits. Never touch credentials.
</rules>

<compact_output_contract>
Write `exp/offline_search/rounds/r17/xgpu_1017/REPORT.md` (Chinese, tables, citations), plus the tools, their tests and COMMANDS.md. Final message: the historical A100 finding, the criteria, the likely causes, and the first GPU command to run.
</compact_output_contract>
