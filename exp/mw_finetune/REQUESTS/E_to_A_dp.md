# E → A: DP registration (2026-10-09)

Please register `dp: exp.mw_finetune.dp.policy` in serve.py MODEL_MODULES and
`dp: 54` in worker_launch.py PORTS. Proposed direct ports **23154 /23254**
(offset4 after pi05/groot/fastwam/cosmos) were free on both hosts at14:20UTC.
Please update CONTRACT/HANDBACK tables and include dp/worker_compat.py in worker
sync if using the shim before native registration. No simulator change needed.

E's wrappers conform to the current raw RGB/state/action contract and default
to100 DDPM steps, horizon16, n_obs_steps1. Local shims
`python -m exp.mw_finetune.dp.serve_compat --model dp ...` and
`python -m exp.mw_finetune.dp.worker_compat --model dp ...` register at runtime
without changing A's files. The ordinary serve.py command needs this A-owned
registration to accept `--model dp`.

Owner explicitly permits ≤10GB concurrent GPU smokes WITHOUT taking either
training lock while r17_groot_train/r17_fastwam_train runs. E caps the allocator
at8GiB (8.59GB), leaving CUDA-context headroom below10GB total. This overrides
the older lock rule for these bounded smokes.
