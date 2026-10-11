# E: R17 Diffusion Policy handback

Ready for coordinator training: **MT50 DP from scratch**, and a **four-suite
LIBERO fine-tuning fallback**. Full 100-step epsilon DDPM / one current observation
is the default everywhere. No long GPU training or formal SR run was launched.
Recipe, sources and every deviation: [RECIPE.md](RECIPE.md).

| Verified result | Evidence |
|---|---|
| Exact parameter reconstruction | All 475 tensors strictly load for LIBERO30k and ML45 30k/85k; inventories in reports |
| Native-source denoising | Epsilon MSE: ML45 85k0.00937;30k0.01599; LIBERO30k0.01393;8 frames/one source episode each |
| MetaWorld easy sanity |85k +zero wrist1/16;30k +zero or duplicated wrist0/16 each; no rollout errors; all 100 DDPM steps |
| LIBERO multi-suite smoke |0/4 complete (spatial/object/goal/10 task0 idx0), current two views, OSMesa; no interface errors |
| MT50 scratch full-data200-update probe |6.90391 updates/s;4.80824/5.00391 GiB allocated/reserved;4.02349h compute projection |
| LIBERO partial-data200-update backward probe |7.19309 updates/s;4.80617/5.00586 GiB;3.86173h projection on one cached episode only |
| CPU tests |11 passed, no skips; `reports/cpu_tests.txt`; includes paired source pixels, stateless RNG and interrupted-link recovery |
| Resume |All 475 tensors, Adam state and CPU RNG bitwise equal after2 continuous vs1/save/reload/1 scratch updates |
| Full LIBERO data |273,465 frames /1,693 episodes /40 tasks; all1,693 paired shards pass;12 source samples exact |
| Full MT50 data |204,806 frames /2,500 episodes /50 tasks; all492 SQLite shards audited;12 samples match source pixels/vectors |

Neither smoke is formal SR. Original source cameras are corner4+gripperPOV;
A's MT50 camera is one corner2 view. Zero/duplicate fills degrade native-source
denoising and neither transferred policy is clearly usable. Owner fallback is
scratch MT50; **no ML45 policy weights initialize it**.30k/zero-view confirmation also completed0/16/no errors. LIBERO extrema match nine LIBERO-10 texts,
not all 40 source texts; this does not prove training coverage. The source-frame
check supports the reconstruction, while0/4 supplies no useful four-suite SR.
Retain the original weights for later formal measurement; own training tooling
is ready as requested. Older pre-ReLU MW diagnostics and seedless-global-RNG
LIBERO diagnostics are excluded from these results.

## Paths, API and ownership

Repo: `/home/weiland/projects/openpi`; local/H100 E root: `/data/mw_finetune/dp`.
H100 code: `/data/mw_finetune/dp/code`. Both have E's Python island, compact CLIP,
three original checkpoints and full 96² MT50 data. Islands reuse installed base
packages read-only; package versions are in `<DP_ROOT>/environment.txt`.

`policy.py:load_policy(checkpoint,device,**kw)` returns `reset()/infer(obs)`;
physical float32(16,4) MetaWorld commands. No history or cross-call state.
`domain="libero"` accepts `observation/image`, `observation/wrist_image`,8-D
`observation/state`,prompt; returns(16,7). Fresh private RNG uses obs.seed or0
when absent. `serve_libero.py` uses existing websocket/msgpack lifecycle and
bundle conventions. Metadata includes the actual weight SHA256 and step.

A native `serve.py --model dp` / worker port registration is requested in
[../REQUESTS/E_to_A_dp.md](../REQUESTS/E_to_A_dp.md); A's files were not edited.
DP offset4 = **23154/23254**, previously verified free and used successfully.
Until A registers it, use `dp.serve_compat --model dp`; it adds the entry to A's
actual server at runtime. `worker_compat` supplies the optional port shim.
The architect handoff is [../REQUESTS/E_to_integration_dp.md](../REQUESTS/E_to_integration_dp.md).

Reboot check: no E process survived. Cleanup removed only three E-owned H100
transfer archives after SHA256 parity with retained local copies, under the
owner's cleanup instruction. Extracted assets/data/optimizer checkpoint remain;
H100 E disk is~5.87 GB. Never transfer full LIBERO data there. No shared training
lock was taken by bounded smokes/probes. Per-process allocator cap8 GiB leaves
CUDA-context headroom below 10 GB; Fast-WAM and GR00T processes were untouched.

## Setup, data and CPU checks

```bash
cd /home/weiland/projects/openpi
bash exp/mw_finetune/dp/setup.sh
export PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
DP_PYTHON=/data/mw_finetune/dp/env/bin/python

# Already complete MT50; rerunnable if needed.
"$DP_PYTHON" -m exp.mw_finetune.dp.prepare_data --workers 8
"$DP_PYTHON" -m exp.mw_finetune.dp.cache_language
"$DP_PYTHON" -m exp.mw_finetune.dp.verify_data
"$DP_PYTHON" -m exp.mw_finetune.dp.verify
"$DP_PYTHON" -m pytest -q exp/mw_finetune/dp/tests

# Local-source search found no requested LIBERO copy. Resumes pinned downloads
# and completed per-episode conversion. All files remain in E's local root.
HF_HUB_DISABLE_XET=1 "$DP_PYTHON" -m exp.mw_finetune.dp.libero.prepare_data \
  --download --download-workers 2 --workers 8
"$DP_PYTHON" -m exp.mw_finetune.dp.cache_language \
  --dataset /data/mw_finetune/dp/libero_96
"$DP_PYTHON" -m exp.mw_finetune.dp.libero.verify_data
```

LIBERO source pin: a4336d589d589045d1c56423ffdf3b88a0e19b1f;
1,693 episodes /273,465 frames /40 tasks /10 Hz. Full training refuses incomplete
manifests. One-episode mechanics set is `/data/mw_finetune/dp/libero_probe_96`;
its `partial=true` cannot be used for a long run. Full local preparation is **complete** at `/data/mw_finetune/dp/libero_96`:
1,693 paired PNG SQLite shards,273,465 frames,40 tasks; all quick_checks and
row counts pass. Twelve spread source samples match both resized views and raw
8/7 vectors exactly; [report](reports/libero_data_audit.json). Source uses
34,939,136,789 bytes; prepared set 7,353,751,170 bytes. The anonymous Xet download
hit429 once; HTTP resume with2 download workers completed without credentials.
Frozen40-task training text cache SHA256:
`45ad4a898350d164c767665088b3ed511751d0e371fe58dec375c5b492240986`.
CPU tests11/no skips. A full-data CPU optimizer update passed with finite loss0.010574 and gradient
norm0.578816 (mechanics only; `reports/cpu_libero_full_data_probe.json`).

```bash
# Existing H100 installation: refresh code only (no large data/asset resync).
"$DP_PYTHON" -m exp.mw_finetune.dp.sync_h100 --code-only
# Fresh H100 setup, after checking disk budget:
tether exec --safe --timeout 5m h100 -- env \
  DP_BASE_PYTHON=/data/mw_finetune/envs/fastwam/bin/python \
  bash /data/mw_finetune/dp/code/exp/mw_finetune/dp/setup.sh
```

## Probes and coordinator training

**No 4090 GPU smoke until GR00T finishes (~18:40 CDT)**: it now uses~39.6/48 GB.
H100 probes below are bounded850s, no GPU lock,8 GiB allocator cap. `--probe N`
counts optimizer updates; no save unless `--save-probe`. Logs are JSONL every
update, and probe JSON excludes first5 updates. Timing excludes checkpoint I/O.

```bash
# Exact full-data scratch H100 probe (already measured; new output if repeating):
tether exec --safe --timeout 15m h100 -- env \
  DP_PYTHON=/data/mw_finetune/dp/env/bin/python \
  bash /data/mw_finetune/dp/code/exp/mw_finetune/dp/launch.sh \
  --init scratch --probe 200 --batch-size 256 --workers 4 \
  --output /data/mw_finetune/dp/h100_scratch_mt50_b256_200

# After GR00T finishes, re-probe LOCAL full data; H100 timing is not4090 timing.
bash exp/mw_finetune/dp/launch.sh --init scratch --probe 200 --batch-size 256 \
  --workers 4 --output /data/mw_finetune/dp/4090_scratch_probe
bash exp/mw_finetune/dp/libero/launch.sh --probe 200 --batch-size 256 \
  --workers 4 --output /data/mw_finetune/dp/4090_libero_probe

# COORDINATOR ONLY. launch.sh holds /tmp/r17_gpu.lock for long jobs.
# Each identical command resumes the newest complete checkpoint automatically.
bash exp/mw_finetune/dp/launch.sh --init scratch --steps 100000 \
  --batch-size 256 --workers 4 --save-every 5000 \
  --output /data/mw_finetune/dp/4090_mt50_scratch
bash exp/mw_finetune/dp/libero/launch.sh --steps 100000 \
  --batch-size 256 --workers 4 --save-every 5000 \
  --output /data/mw_finetune/dp/4090_libero_finetune
```

LIBERO launcher defaults to `--init pretrained` with LIBERO30k; override with
`--init scratch` if chosen. MT50 defaults to scratch. Both retain original
100k/global 256, AdamW(.95,.999),1e-4→1e-5/500-warmup/no-EMA recipe. If batch256
fails, use `--batch-size 64 --accumulation 4` and remeasure (old H100 probe11.16h).
Each long run retains20 ~1.10 GB immutable optimizer snapshots (~22.1 GB); run
locally. Full H100 training would exceed E's10 GB disk budget. `latest` is an
atomic link; resume also recovers a complete save before an interrupted link
update. Partial snapshots are preserved/ignored; recipe mismatches fail.

## MetaWorld serve, formal eval, sync-back, aggregate

Commands below use explicit10 trials to pin the published500-episode contract.
A formal run is50 tasks×10, seed7,15zero settle,160policy steps, first5 actions
per16-action chunk; report task mean and equal difficulty-group mean separately.

```bash
# COORDINATOR local server after GPU/port check. Use immutable final checkpoint.
ss -tlnp
nvidia-smi --query-gpu=memory.free,memory.total --format=csv
flock /tmp/r17_gpu.lock /data/mw_finetune/dp/env/bin/python \
  -m exp.mw_finetune.dp.serve_compat --model dp \
  --checkpoint /data/mw_finetune/dp/4090_mt50_scratch/checkpoint-100000 --port 23154

# Timan107 EGL worker lane, server ziyanglin.com:23154.
tether exec --safe --timeout 24h timan107 -- env \
  PYTHONPATH=/scratch/zixuans8/mw_r17/code:/scratch/zixuans8/mw_r17/code/src:/scratch/zixuans8/mw_r17/code/packages/openpi-client/src \
  /scratch/zixuans8/mw_r17/envs/sim/bin/python -m exp.mw_finetune.evaluate \
  --run /scratch/zixuans8/mw_r17/runs/dp_mt50_scratch_100k \
  --model-id dp_mt50_scratch_100k --host ziyanglin.com --port 23154 \
  --renderer egl --workers 12 --trials 10
"$DP_PYTHON" -m exp.mw_finetune.sync_back timan107 dp_mt50_scratch_100k
"$DP_PYTHON" -m exp.mw_finetune.aggregate /data/mw_finetune/eval/dp_mt50_scratch_100k
```

For original85k measurement change checkpoint to `checkpoints/metaworld`, run
ID to `dp_ml45_85k_mt50`, and additionally run `dp.aggregate_coverage RUN` to
report five nominal ML45 held-out tasks and the extra unannotated dial task.
Keep its zero-wrist transfer caveat. For the exact easy sanity, replace formal
worker/trial selection with:

```bash
--workers 2 --trials 2 --tasks button-press-v3 door-close-v3 drawer-close-v3 \
  faucet-open-v3 handle-press-v3 reach-wall-v3 window-close-v3 plate-slide-side-v3
```

H100 lane serving needs only a367MB model export, not an optimizer snapshot.
Check `du -sb` before transfer; one archive plus extracted export fits E's budget.

```bash
"$DP_PYTHON" -m exp.mw_finetune.dp.export \
  /data/mw_finetune/dp/4090_mt50_scratch/checkpoint-100000 \
  /data/mw_finetune/dp/mt50_policy_100000
tar -cf /data/mw_finetune/dp/mt50_policy_100000.tar -C /data/mw_finetune/dp mt50_policy_100000
tether push /data/mw_finetune/dp/mt50_policy_100000.tar \
  h100:/data/mw_finetune/dp/mt50_policy_100000.tar --force
tether exec --safe --timeout 5m h100 -- tar -xf \
  /data/mw_finetune/dp/mt50_policy_100000.tar -C /data/mw_finetune/dp
tether exec --safe --timeout 24h h100 -- env \
  PYTHONPATH=/data/mw_finetune/dp/code:/data/mw_finetune/dp/code/src:/data/mw_finetune/dp/code/packages/openpi-client/src \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  flock /tmp/r17_gpu.lock /data/mw_finetune/dp/env/bin/python \
  -m exp.mw_finetune.dp.serve_compat --model dp \
  --checkpoint /data/mw_finetune/dp/mt50_policy_100000 --port 23254

tether exec --safe --timeout 24h timan108 -- env \
  PYTHONPATH=/scratch/zixuans8/mw_r17/code:/scratch/zixuans8/mw_r17/code/src:/scratch/zixuans8/mw_r17/code/packages/openpi-client/src \
  /scratch/zixuans8/mw_r17/envs/sim/bin/python -m exp.mw_finetune.evaluate \
  --run /scratch/zixuans8/mw_r17/runs/dp_mt50_scratch_100k_h100 \
  --model-id dp_mt50_scratch_100k_h100 --host 149.165.153.233 --port 23254 \
  --renderer egl --workers 12 --trials 10
"$DP_PYTHON" -m exp.mw_finetune.sync_back timan108 dp_mt50_scratch_100k_h100
"$DP_PYTHON" -m exp.mw_finetune.aggregate /data/mw_finetune/eval/dp_mt50_scratch_100k_h100
```

Workers are always on timan; never host a policy there. Drivers are resumable
with the identical manifest after their previous invocation exits. Sync-back
runs only after driver exit. Shared harness/sync tools write their normal result
locations; E modifies none of their code.

## LIBERO serve, smoke and full SR

```bash
# COORDINATOR server; choose original30k or an immutable own final checkpoint.
flock /tmp/r17_gpu.lock "$DP_PYTHON" -m exp.mw_finetune.dp.serve_libero \
  --checkpoint /data/mw_finetune/dp/checkpoints/libero --port 23154

# Reproduce four-suite mechanical smoke using the existing local sim island.
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1
export LD_LIBRARY_PATH=/data/mw_finetune/envs/osmesa/usr/lib/x86_64-linux-gnu:/home/weiland/projects/openpi_ext/envs/libero_sim/lib
/home/weiland/projects/openpi_ext/envs/libero_sim/bin/python \
  -m exp.mw_finetune.dp.smoke_libero --host ziyanglin.com --port 23154 \
  --suites libero_spatial libero_object libero_goal libero_10 --tasks 0 --trials 1 \
  --output /data/mw_finetune/dp/libero_multisuite_repeat.json

# Full SR on an existing EGL LIBERO worker, simulator/assets pinned.
# LIBERO_SIM_PY is that worker's existing interpreter. Run each suite separately.
for DP_SUITE in libero_spatial libero_object libero_goal libero_10; do
  "$LIBERO_SIM_PY" examples/libero/main.py --host ziyanglin.com --port 23154 \
    --task-suite-name "$DP_SUITE" --num-trials-per-task 50 --resize-size 96 \
    --replan-steps 5 --seed 7 \
    --episode-results-path "/data/mw_finetune/dp/${DP_SUITE}_results.json"
done
```

Full LIBERO SR is2,000 identities (4×10×50),10zero settle and native suite limits
220/280/300/520. Preserve every episode row, pin renderer/simulator/assets and
report each suite separately. Never interpret OSMesa0/4 as formal SR. The
source checkpoint is runnable; source-frame convention tests and uncertain
training coverage are disclosed. No reported author SR exists for comparison.

## CUDA graph (eval throughput)

Added opt-in `cuda_graph=False` to `Policy`/`load_policy`. Changed files:
`policy.py`, `serve_libero.py`, `serve_compat.py`, `tests/test_dp.py`, this handback;
added `cuda_graph.py`, `benchmark_cuda_graph.py`, `tests/test_cuda_graph.py` and
the `reports/cuda_graph_*` evidence files. `model.py`, training, `src/`,
`exp/r17_integration/`, and `run_eval_dp_libero.sh` were not edited. No server
was started, contacted, restarted or killed; live ports23152–23155 remain running.

Only `core.model(trajectory,t,cond)` is captured, once per shape/dtype/stride/device.
Its three static inputs are overwritten before every replay. Vision/language
encoding, initial trajectory noise, every `scheduler.step`, private generator
draws and physical action unnormalization remain eager. There is no cached
observation, action or advancing RNG. Replay workspace is serialized per sampler.
The decoder's existing causal-mask detection result is computed before capture
and supplied through a temporary pre-hook, removed after capture. Attention
masks, precision settings and numerical operations remain unchanged. Three side
stream warmups and persistent input/output storage follow the
[PyTorch2.7 CUDA graph recipe](https://docs.pytorch.org/docs/2.7/notes/cuda.html#cuda-graphs).

`--cuda-graph` is available in both E serve entry points. Enabling it performs
three complete eager/graph `infer` comparisons before binding a server port;
different fixed random images/states, seeds11/37/101 and three distinct prompts
are used. It requires `torch.equal` on **final physical actions**, raises on any
failure or capture error, and never falls back. CPU plus this flag raises.
Enabled LIBERO metadata includes `"cuda_graph": true` and the self-check report;
MetaWorld records those in `wrapper_metadata`, plus
`policy_options.cuda_graph=true`. To honor the explicit unchanged-off metadata
requirement, **OFF preserves the legacy metadata exactly**: absence of the new
key means false, rather than adding a false field. The original `model.py` hash
also stays unchanged.

Exact coordinator serve flags (examples only; not launched):

```bash
/data/mw_finetune/dp/env/bin/python -m exp.mw_finetune.dp.serve_libero \
  --checkpoint /data/mw_finetune/dp/libero_policy_100000 \
  --device cuda:0 --port 23154 --denoising-steps 100 --cuda-graph

/data/mw_finetune/dp/env/bin/python -m exp.mw_finetune.dp.serve_compat \
  --model dp --checkpoint /data/mw_finetune/dp/checkpoints/metaworld \
  --device cuda:0 --port 23154 --denoising-steps 100 --cuda-graph
```

Choose the appropriate command for a lane; these examples share a port and
are not simultaneous launches. The MetaWorld shim translates the flag into
A's existing `--policy-option cuda_graph=true`; A's files need no changes.

Measured on local RTX4090 / `cuda:0`, PyTorch2.7.1+cu126, B=1, all100 DDPM steps,
while the four live LIBERO servers shared the GPU. Five synchronized whole
`Policy.infer` calls after warmup, excluding load/capture/startup-check time:

| Checkpoint | Eager median ms | Graph median ms | Speedup | Evidence |
|---|---:|---:|---:|---|
| LIBERO `libero_policy_100000` |1764.81|404.97|4.36×|[JSON](reports/cuda_graph_libero.json)|
| MetaWorld ML45 85k `checkpoints/metaworld` |1718.63|331.40|5.19×|[JSON](reports/cuda_graph_metaworld.json)|

LIBERO graph samples378.22–429.10ms; MetaWorld312.20–370.37ms. An earlier
LIBERO pass measured1754.10→357.64ms; sharing causes timing variation. Each
benchmark process had a1.25GiB allocator cap, reported910MiB total process GPU
memory, ~380.15MiB peak allocated /426MiB peak reserved. Only cuda:0 was used,
one test process at a time, without a GPU lock.

Both checkpoints passed `torch.equal` on all three observations, separately
from the mandatory constructor check. Default-off actions and metadata also
match snapshots taken **before policy edits**, retained under
`/data/mw_finetune/dp/cuda_graph_baseline/{libero,metaworld}.npz`. Interleaved
observations and seedless calls remain deterministic. No trained MT50 export
is present locally; the available MetaWorld85k checkpoint was verified.
CPU suite:15 passed /1 CUDA-only skip ([output](reports/cuda_graph_cpu_tests.txt));
the small CUDA replay test separately passed, checking changed timesteps,
conditions and B=1/B=2 shape caching ([output](reports/cuda_graph_gpu_tests.txt)).
CPU tests also prove one-ULP mismatches fail startup and the MetaWorld flag
preserves default arguments when absent.

Reproduce checks/timing against the retained pre-edit baselines:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /data/mw_finetune/dp/env/bin/python -m pytest exp/mw_finetune/dp/tests -q

/data/mw_finetune/dp/env/bin/python -m exp.mw_finetune.dp.benchmark_cuda_graph \
  --checkpoint /data/mw_finetune/dp/libero_policy_100000 --domain libero \
  --baseline /data/mw_finetune/dp/cuda_graph_baseline/libero.npz --repeats 5 \
  --report /data/mw_finetune/dp/reports/cuda_graph_libero.json

/data/mw_finetune/dp/env/bin/python -m exp.mw_finetune.dp.benchmark_cuda_graph \
  --checkpoint /data/mw_finetune/dp/checkpoints/metaworld --domain metaworld \
  --baseline /data/mw_finetune/dp/cuda_graph_baseline/metaworld.npz --repeats 5 \
  --report /data/mw_finetune/dp/reports/cuda_graph_metaworld.json
```
