Run from `/home/weiland/projects/openpi`. No simulator, server or chain is part of these checks. Python prefixes below pin only CPUs 34–37,78–81, with one BLAS/OMP thread. All scripts retain their expanded subprocess arguments. Fresh `/tmp/q5_*` paths are required for repeated CPU matrices; concurrency and lifecycle tests deliberately refuse existing output directories.

```bash
B=exp/offline_search/rounds/r05/q5_gpu
P=(taskset -c 34-37,78-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B/boot:.:src .venv/bin/python)
# Candidate import uses owned boot/sitecustomize.py; installed import sets Q5_SOURCE=installed.
# Historical development step only: do not re-run prepare.py after installation.
# It rebases onto the installed preimage and records that preimage for install.py.
"${P[@]}" "$B/prepare.py"
taskset -c 34-37,78-81 bash "$B/final_regression/run_early.sh"
taskset -c 34-37,78-81 bash "$B/regression/run_tail.sh"
"${P[@]}" "$B/run_q2.py"
taskset -c 34-37,78-81 bash "$B/run_stack.sh"
```

The original full legacy recipe is `regression/run_installed.sh`. Development separated the K5 file-install audit from the tests, because it asserts that the installed plugin equals the candidate. `final_regression/run_early.sh` reruns K2/K1/K4, all six byte comparisons, K5 replay/overlay/estimator/ledger checks against the frozen final candidate. `regression/run_tail.sh` runs K6/K7/K10 unchanged, with outputs relocated. The audit is executed after installation. Q2 tests use its installed methods and retained CPU inverse-transform helpers; no GR00T model is loaded.

`reuse_verification_fits.py` copied two exact-matching existing K7 fits into the owned verification scratch directory before their selftests started. It checked the full method/kwargs/cell tuple and the source/destination SHA256; see `results/k7_fit_reuse.json`. Its exact invocation was `"${P[@]}" "$B/reuse_verification_fits.py"`. Other required verification fits were created in `/tmp/q5_regression_fits/` by the inherited recipes.

GPU commands (sequential: never run two of these together):

```bash
taskset -c 34-37,78-81 bash "$B/run_gpu.sh"
taskset -c 34-37,78-81 bash "$B/run_stage1.sh"
```

Each GPU child checks `nvidia-smi` for at least 14336 MiB free before CUDA initialization. PyTorch allocation is capped at 5120 MiB, leaving context overhead below the 6144 MiB process limit, with a per-process memory watchdog. Any failed child, including OOM, stops its shell recipe. Each measurement is a fresh process and releases CUDA on exit. No ports are used. `run_gpu.sh` runs two independent processes per family/scale on the same 2200 stored queries; repetitions are not independent queries. `run_stage1.sh` uses the existing checkpoint-validated CPU STAGE1_ONLY weight cache `/tmp/k9_scratch/stage1_policy.pt` read-only, places stage 1 on CUDA and stages 2/3 on meta. It replays six real observations per mode/scale check without starting a server. The final variant uses K9 float64 and preserves the model's global TF32 setting.

Installation and arm validation:

```bash
"${P[@]}" "$B/install.py"
# Install is one-shot, checks Q2's hand-back and exact plugin preimage hash.
"${P[@]}" "$B/make_arms.py"
"${P[@]}" exp/offline_search/closed_loop/ops/emit_arms.py --run-root /tmp/q5_emitted --spec "$B/arms_q5.json"
# Final checks import the actual installed plugin, including eight fake-stack
# configurations, nine byte comparisons, K5's installed-file audit and arm emission.
taskset -c 34-37,78-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 Q5_SOURCE=installed PYTHONPATH=$B/boot:.:src .venv/bin/python "$B/postinstall.py"
"${P[@]}" "$B/summarize.py"
"${P[@]}" "$B/write_handback.py"
```

The final GPU pass ran `run_gpu.sh` followed by `run_stage1.sh` in one sequential shell, with output in `results/gpu_run.log` and `results/stage1_run.log`. Earlier measurements were archived under `results/dev_float32/`, `results/dev_float64/`, and `results/dev_prefinal/`; only the top-level eight replay JSON files and four `stage1_*/report.json` files contribute to the final tables. `results/frozen_sources.json` records the immutable candidate plugin and GPU-helper hashes. The final evidence audit checks those runtime hashes again against the installed files.

No deployment refit is required. `prefit.sh` validates and emits the exact reuse metadata, without fitting. `arms_q5.json` contains only the two runnable AWM shadow arms. K7 exclusions and exact attempted kwargs are recorded in `arms_unsupported.json`.

Coordinator-only smoke: emit the arm specs into `<RUN>`, use the ordinary next π0.5 server start (pure-cache may use STAGE1_ONLY), a five-control client and both 50/500 libraries. For a short input audit, shadow accepts `--os-log-inputs`; GPU serve deliberately rejects it. Aggregate with:

```bash
"${P[@]}" "$B/summarize_shadow.py" --log-dir '<RUN>/server_logs' --out '<RUN>/q5_shadow_summary.json'
```

CPU remains authoritative in shadow. Promotion to `--os-gpu-retrieval serve` is a coordinator decision after inspecting chunk/confidence as well as top-1 agreement; step-zero strict mismatches remain. MixedJudge additionally needs a full model and `--os-judge guard_only`. K7/blind/policy-tail, GR00T, native shadows, token exposure, R4 stage synchronization, shared quantile control, randomized overlays, non-full camera modes, multiple checkpoints and cache writes are explicitly refused. The existing separate retrieval graph is not a combined stage-1 graph.
