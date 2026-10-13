<task>
Repo /home/weiland/projects/openpi (R17 framework under exp/r17_integration). Goal: MetaWorld MT50 runs must get
the same serving features LIBERO has, so the owner can saturate the GPUs: batched concurrent servers, MPS, reduced
denoising steps, long commit horizon, per-episode YAML hot switching for cache experiments. Priorities in order:
1. Batched servers on MetaWorld + capture. `models/dp/formal_batch_server.py:33` and
   `models/fastwam/formal_batch_server.py:44` raise unless `domain == "libero"`, and both force
   `debug_capture=False`. Make both serve MetaWorld (one camera, MT50 wire format; DP DDPM100 CUDA graph path and
   Fast-WAM batching for the one-view MT50 geometry) and accept library teacher capture the same way
   `ops/formal_server.py` does (lines 150-180: `LibraryCaptureFactory` as `capture_sink_factory`,
   `bind_observation(obs)` before inference). Another agent (package A) is changing `debug/library_capture.py` to add a
   no-image switch and one-camera support - call its factory, do not edit that file. In `ops/formal_eval.py` admit
   `--max-batch > 1` and `--mps` for `dp_mw` and `fastwam_mw` (today only Fast-WAM/DP LIBERO; RUN_CONFIGS "Launch and
   pairing rules"). Report B>1 vs B1 max action difference on a few fixed MT50 observations (information only, not a
   gate).
2. Reduced steps and long commit horizon on MetaWorld. `ops/formal_eval.py:172` (`execution_horizon`) and `:185`
   (`execution_steps`) accept only LIBERO. Allow MetaWorld for dp/fastwam (DP 5/10/100 DDIM/DDPM as on LIBERO,
   Fast-WAM 1/2/10) and for pi05/groot if their providers expose `sampler_for_steps` (pi05 K 1/2/10, GR00T K 1/2/4).
   The MetaWorld kernel executes at most 5 actions per call (`exp/metaworld/env.py:run_episode`) and the R17 runner
   requires commit divisible by replan (`benchmarks/metaworld.py` run()); make commit horizon 10 work on MT50
   (cache hits execute 10 steps, model calls 5) the same way LIBERO does it.
3. Cache plans with YAML hot switching on MetaWorld. `ops/cache_study.py:24` MODELS = dp_libero/fastwam_libero/
   cosmos_libero only. Add the MetaWorld bundles so a cache campaign on MT50 registers several YAML bundles on the
   resident server (`load_cache_config`) and the conductor switches them per episode (`conductor/strategy.py`
   ensure_bundle/select_bundle), with the library at `/data/r17_library/<model>_mt50/bpool/` (built by package A,
   seed 8; use the existing Store API). Test-set episodes come from seed 7 idx 0..49. Start with dp_mw and
   fastwam_mw; for pi05_mw/groot_mw report what their R17 providers lack for the cache path, implement it if small.
</task>
<context>
Read first: `exp/r17_integration/ops/RUN_CONFIGS.md` ("Launch and pairing rules", the MetaWorld MT50 sections,
"Pitfalls", "Cache-study topology in use"), `ops/formal_server.py`, `ops/formal_eval.py`, `ops/cache_study.py`,
`models/dp/{formal_batch_server,batch_runtime,provider}.py`, `models/fastwam/{formal_batch_server,formal_batching}.py`,
`conductor/{driver,strategy,manifest}.py`, `benchmarks/metaworld.py`, `exp/metaworld/env.py`.
Owner rule: keep it simple - smallest diff, no new seals, fingerprints, admission gates, receipts or validation
layers; LIBERO-only checks that block MetaWorld may be removed. LIBERO behaviour must not change.
Package A edits concurrently: `benchmarks/spec.py`, `benchmarks/metaworld.py` (seed handling), `exp/metaworld/
episode_runner.py`, `conductor/manifest.py`, `debug/*`, `search/teacher_library.py`, `ops/library_*.py`,
`ops/timan1_workers.py`, `ops/worker_installs.py`, and in `ops/formal_eval.py` only `prepare_workers` and library
plumbing. Package D adds `cosmos_mw` under `models/cosmos/`. Stay out of their regions; if you need a change there,
write it in your handback. Never revert changes you did not make; do not git commit.
</context>
<action_safety>
CPU work only: do NOT start GPU servers or campaigns (`--run`); the coordinator launches every GPU run. You may use the
GPU for at most a few seconds of fixed-observation parity checks only if `nvidia-smi` shows it idle, and must free it.
No sudo, no `rm -rf`, no bare `pkill -f`, no credentials, no `codex exec`, no tether expose.
</action_safety>
<verification_loop>
Minimal CPU tests: batched MW servers build and serve fake requests with capture on/off; formal_eval admits
max-batch/mps/steps/horizon for MW; a 2-bundle MT50 cache plan registers both YAMLs and assigns bundle ids per
episode. Run the existing tests of the files you touched; report pre-existing vs new failures.
</verification_loop>
<compact_output_contract>
Write `exp/r17_integration/handbacks/HANDBACK_MW_B.md`; end with files changed, removed checks, test results, parity
numbers, and exact coordinator commands: batched dp_mw / fastwam_mw MT50 pure-inference smoke (2 tasks x 2 idx,
`--prepare-only` then `--run`) with R/B/MPS as parameters, the same with library capture, and a 2-bundle MT50 cache
smoke.
</compact_output_contract>
