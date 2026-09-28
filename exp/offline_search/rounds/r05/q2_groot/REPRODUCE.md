# Q2 reproduction (repository root)

All commands use `/home/weiland/projects/openpi/.venv/bin/python`, CPUs `30-33,74-77`, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1`. No server is started by any Q2 script. The cold store is `/home/weiland/trace_runs/offline_search_store`.

```bash
B=exp/offline_search/rounds/r05/q2_groot
P=(taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
# Already completed once; install.py intentionally refuses a changed preimage.
"${P[@]}" "$B/install.py"
# Final installed-file verification:
bash "$B/regression/run_installed.sh" > "$B/results/installed_regression.log" 2>&1
"${P[@]}" "$B/make_arms.py"
bash "$B/prefit.sh" > "$B/results/prefit.log" 2>&1
"${P[@]}" "$B/run_final.py" > "$B/results/final_checks.log" 2>&1
# Once both test sessions have completed:
"${P[@]}" "$B/summarize.py" > "$B/results/summarize.log" 2>&1
"${P[@]}" "$B/final_audit.py" > /tmp/q2_final_audit.log 2>&1
"${P[@]}" "$B/write_handback.py" > "$B/results/handback.log" 2>&1
"${P[@]}" "$B/final_audit.py" --verify-manifest > /tmp/q2_manifest_verification.log 2>&1
```

The final runs above used separate shell sessions for regression and Q2 checks. Maximum simultaneous Python process fanout is eight (K2's four workers plus Q2's runner, concurrency driver, and worker, or three parity processes). NumPy/BLAS threads remain one. Eight-connection tests use eight threads in one worker process. No GPU work, networking, server, port, LIBERO worker, chain, remote command, git command, or review-test read is part of these commands.

For a repeat, relocate scratch/output paths to fresh `/tmp/q2_*` directories: concurrency/edge drivers refuse existing output directories and the blind driver appends JSONL. Do not rerun the installer to repeat tests. `results/final_commands.json` contains the exact expanded Q2 commands and return codes. The regression shell and its relocated constituent recipes contain the exact existing-matrix commands; K5 replay and K7 runners also persist command JSON. All original K10 test assertions are retained; only output paths, import loaders and CPU masks were relocated. `before/` contains the shared preimages, `dev/` the candidates installed with `os.replace`.

`prefit_commands.json` contains all four exact coordinator command templates with literal `<RUN>` placeholders. `prefit.sh` contains the exact commands executed locally, writing artifacts only to `/tmp/q2_fits/`. Existing matching prefits are loaded, so choose a fresh artifact directory to measure a fresh fit. The four final G10 selftests explicitly load these exact prefits through PluginRuntime.

The K7 regression recipe retains its original artifact-loading behavior. Its first three artifacts were fitted under `/tmp/q2_regression_fits/`; the last two were copied from the exact existing `/tmp/k7_guard_fits/` artifacts after validating method/kwargs/cell. `results/k7_fit_reuse.json` records both source/destination paths, metadata and SHA256. The unchanged verifier still fits its independent replay reference. All four Q2 deployment prefits were fresh fits.

`cpu_transforms.py` loads the actual GR00T inverse action components and checkpoint metadata, omitting the identity GR00T model inverse and unrelated image/state transforms. It extends Python's dependency search path to the installed GR00T source/environment but uses the repository Python/torch. It neither constructs the VLM processor nor loads model weights. The initial full-transform import probes failed on missing `pytorch3d`, then a transformers `VideoInput` incompatibility; those failed probes are retained in `results/transform_probe*.log`. The focused CPU action path is the final tested path.
