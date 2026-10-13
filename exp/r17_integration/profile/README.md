# Batch-1 owner cost profile

Prepared using CPU only. No GPU measurements, CUDA initialization, remote
commands, server changes, process control, or commits were performed.
The GPU compilation/replay and numeric parity results must come from the later
exclusive-device run; the CPU checks do not establish those results.

## Files

- `latency.py`: formal loader, real input selection, three compiled tensor
  programs, native sampling, CUDA-event timings, parity, graph counters and
  full-precision prices. Default warmup is 30 and measurement is 200 iterations
  **per step budget**. Each invocation covers one prompt and both budgets.
- `certify.py`: small launcher that runs both suites in separate model processes
  and certifies the measurement-only nsys trace. If nsys is absent, the same
  profiler counts actual `torch.cuda.CUDAGraph.replay` and `capture_begin` calls
  and records Dynamo/Inductor counters. No launch count is inferred as evidence.
- `cpu_inputs.json`: CPU inspection of row zero in the six existing B pools,
  with original camera geometry, actual prompts, observation hashes and formal
  descriptor paths. This is provenance, not measured latency.
- `tests/test_latency.py`: CPU stub timing, native-update intervals, graph-count
  restoration, nsys parsing, parity, FX specialization, stage cuts and pricing.

## Stage boundaries

All costs follow `common/serving.py`: stage1 includes `prepare` and
`run_stage1`; stage2 is `run_stage2`; stage3 is the complete `sample` call.
Search, `to_wire`, transport, and controller execution are outside the timed
stage costs. CUDA event intervals include host dispatch gaps; synchronized
host-wall totals are also saved. CPU preparation and retrieval-key copies are
inside the appropriate stage interval.

| Model | Stage1 | Stage2 | Stage3 and actual NFE |
| --- | --- | --- | --- |
| DP | Both real cameras resize to 96, CPU CLIP text, external state normalization, checkpoint normalizer and both FiLM vision branches; copy the two keys to CPU | Concatenate native features in checkpoint order | Native epsilon network and diffusers scheduler; DDPM100 = 100 NFE; trailing DDIM10 eta0 = 10 NFE |
| Fast-WAM | Both RGB256 cameras resize to 224, concatenate to 224×448, native Wan VAE, patch embedding and pooled retrieval key | T5 context, proprio token, native pre-DiT conditioning and **full 5B video DiT** KV prefill | Native action expert and shifted-flow update; K10 = 10 NFE; K1 = 1 NFE |
| Cosmos | Both original RGB256 cameras, official frame conversion/augmentation and native 33-frame VAE encoding to `[1,16,9,28,28]`; pool wrist/primary camera keys | Official `get_x0_fn_from_batch`, reusing this observation's latent; native proprio injection and conditioning | Native always-conditional denoise and 2ab solver, including clean step; K5 = 5 NFE; K2 = **1 NFE** |

The tensor cores are extracted with real FX tracing and then compiled using
`torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)`. This
specializes Python structure and fixed serving-protocol branches while keeping
observations, conditioning, timestep/sigma and noise as tensor inputs. VAE
Python caches and Cosmos condition objects are handled inside this extraction.
There is no eager graph-break fallback. Output clones prevent graph replay from
overwriting live conditioning or trajectories. All substitutions are local to
the profiling process, and the native adapter is the eager reference.

The stage3 graph contains the **network evaluation**. The native RNG,
scheduler/solver, validation and output conversion remain in `sample`, and are
included in its full-loop timing. `stage3_network_ms` times each compiled network
call including its output clone. `stage3_step_ms` times each network plus the
native work until the next network starts; the final interval includes sample
output conversion. Initial RNG/setup appears only in full-loop timing. Thus the
step median is diagnostic; pricing always uses the directly measured full loop,
never step median × K. This distinction is explicit because GR00T's tensor step
also contains its small Euler update.

The B capture stores Cosmos's **native** 9-D gripper/xyz/quaternion fields in
float32. The profiler preserves these recorded fields directly and labels their
precision; it does not reconstruct a quaternion from an axis-angle state. For
a recording preserving the original wire float64, pass `--observations FILE.npz`
and `--descriptor LOAD.json`. The NPZ needs a leading observation-row axis on
`observation/image`, `observation/wrist_image`, `prompt`, and either
`observation/state` or `observation/cosmos_proprio`.

## Receipt and certification

Each JSON has a `cells` entry for each budget with raw timings, medians/p10/p90,
actual NFE for every iteration, stage1/stage2/action parity against the unchanged
eager adapter, same-seed repeat equality, different-seed output distinction,
expected/observed replay counts and compilation/capture counters. The same
explicit request seed drives eager and compiled sampling, including DP's
sequential posterior draws. Parity uses rtol=atol=0.02 and also reports maximum
absolute delta and relative Frobenius error; these tolerances are recorded.

The expected count is `iters × (2 + actual_NFE)` for each cell. No separate
timing calls are made inside the certified range, just CUDA events around the
existing step calls. Expected **total launches per suite**, with 200 iterations
of both budgets, are DP 22,800; Fast-WAM 3,000; Cosmos 2,000. Compilation,
warmup, parity and fresh-noise checks precede `cudaProfilerStart`; the unique
NVTX marker binds the nsys export to the JSON. Certification checks zero
captures/instantiations in that range, zero cudagraph skips, no new compiles,
exactly three compiled programs and the exact replay count. With nsys, the API
trace verifies aggregate launches; torch counters retain the per-budget counts.

`certified` describes graph evidence; `valid` additionally requires the recorded
parity/RNG/NFE checks. The wrapper exits nonzero for an invalid receipt and
leaves the JSON and logs for inspection. Coefficients in an invalid receipt
remain diagnostic and are not a measured authority. This writes no cost table
for π0.5, GR00T or the new models.

The coefficients use the full-budget cell's stage1 and stage2 medians:

```text
D = stage1_full + stage2_full + stage3_full_loop
vision       = stage1_full / D
full_miss    = (stage2_full + stage3_full_loop) / D
reduced_miss = (stage2_full + stage3_reduced_loop) / D
```

They are saved without decimal rounding. The two prompts have independent
prices. No prompt average is presented as an authority. Checkpoint/content
identities, source hashes, effective loader descriptor, torch/CUDA/driver/GPU
identity, actual VRAM capacity and load/compile/warmup and measurement memory
peaks are also saved. A Cosmos descriptor pointing at the absent `/data/cosmos`
checkout is relocated to the existing local checkout; the effective path and
actual content hashes are recorded. `--vendor-root` can select it explicitly.

## Memory fit

These are CPU inspection and existing-record estimates, not a new measurement
of graph memory. On the specified **48 GiB** target, all three are expected to
fit at B1 with one model resident at a time. The later receipts report actual
capacity and both memory peaks, since compile/graph pools can add workspace.

- **DP:** 91.85M checkpoint elements, 367,381,916 serialized tensor bytes
  (about 0.342 GiB). Load the complete DP weights and normalization plus the
  formal CLIP text encoder. CLIP stays on CPU, as the adapter does. GPU memory
  should be comfortably below either 24 or 48 GiB.
- **Fast-WAM:** checkpoint header contains 4,999,787,712 video-expert elements
  and 1,020,886,023 action-expert elements. The BF16 tensor payload is
  12,041,421,198 bytes (about 11.21 GiB). The full video DiT must stay resident
  for stage2; loading only the action expert would measure the wrong model.
  Also load the Wan2.2 VAE, tokenizer, UMT5-XXL text encoder, proprio head and
  normalization. Existing `models/fastwam/measurements/local_sweep.json` B1
  records show about **23.4 GiB allocated** with text resident, before this
  new graph compilation. A 48 GiB target has useful headroom. A standard 24 GB
  4090 does not have reliable headroom for this exact resident-text profile.
  `--precompute-text` computes the current prompt's exact native T5 context
  once, moves T5 to CPU, and measures cached-text stage2; this saves roughly
  8.6 GiB of BF16 text weights. It is a separately labeled cost. **The current
  R17 adapter calls T5 in stage2 on each request**, so the default includes it.
  The loader initially loads T5 as the formal server does; the cached option
  therefore does not remove the initial GPU load peak.
- **Text encoders are recomputed per request (checked 2026-10-12).** DP re-encodes the
  constant prompt with CPU CLIP-L/14 inside `prepare` on every request
  (`exp/mw_finetune/dp/preprocess.py::LanguageEncoder`, no cache); the receipt reports it
  separately as `dp_cpu_text_encoder_ms` (already included in `stage1_ms`). Fast-WAM calls
  UMT5-XXL in stage2 on every request; run once more with `--precompute-text` for the
  once-per-episode price. Report both prices; which one enters the formal IR is the owner's call.
- **DP smoke on a shared 4090 (2026-10-12, warmup 3, iters 5, not exclusive):** stage1 134 ms, of which
  CPU CLIP is 130 ms (one OMP thread, as the server runs); stage2 0.4 ms; DDPM100 loop 157 ms; DDIM10 loop
  18.5 ms. All eager-vs-compiled parity checks pass (max delta 6e-4) once stage1 fields keep the adapter's key order.
- **Cosmos:** load the official 2B policy DiT, native tokenizer/VAE, statistics,
  and released `libero_t5_embeddings.pkl`. T5 itself is unnecessary: the
  server already uses the precomputed `[1,512,1024]` prompt embeddings.
  `handbacks/HANDBACK_COSMOS.md` records roughly 6 GiB ready-state memory and
  an 8.26 GiB native B1 peak. Both 24 and 48 GiB should have headroom; the
  later graph memory peak remains to be measured.

[NVIDIA specifies 24 GB for a standard RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).
The 48 GiB fit statement assumes the coordinator's card actually has that
capacity; it is not inferred from the product name.

## Coordinator commands

Run these **after experiments finish**, on the exclusive 4090 with **MPS off** (owner 10-11: same as the π0.5/GR00T profiles; confirm no `nvidia-cuda-mps-server` and no other compute process on the card — the receipt records this as `device_exclusive` and marks the run invalid otherwise). Change the
physical index `0` below if the assigned 4090 has a different index. The visible
device is then `cuda:0`. No campaigns or servers are started by this wrapper.

```bash
cd /home/weiland/projects/openpi
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python -m exp.r17_integration.profile.certify \
  --model dp_libero --out-dir exp/r17_integration/profile/results

.venv/bin/python -m exp.r17_integration.profile.certify \
  --model fastwam_libero --out-dir exp/r17_integration/profile/results

.venv/bin/python -m exp.r17_integration.profile.certify \
  --model cosmos_libero --out-dir exp/r17_integration/profile/results
```

Each command runs spatial and libero_10 sequentially, using the existing model
Python island and the B row's formal `load.json`. Outputs are
`results/<model>_<suite>.json`, `.log`, `.nsys-rep` and `.trace.csv` (the latter
two only when nsys is installed). nsys is currently available locally at
`/usr/local/cuda/bin/nsys`. Use a fresh `--out-dir` for repeated measurements.

A direct single-prompt example, with torch counters only (set `PYTHONPATH` as certify does):

```bash
PYTHONPATH=$PWD:$PWD/src:$PWD/packages/openpi-client/src \
/data/mw_finetune/dp/env/bin/python -m exp.r17_integration.profile.latency \
  --model dp_libero --suite libero_spatial \
  --out exp/r17_integration/profile/results/dp_direct.json
```

Optional separately labeled Fast-WAM cached-text measurement:

```bash
.venv/bin/python -m exp.r17_integration.profile.certify \
  --model fastwam_libero --precompute-text \
  --out-dir exp/r17_integration/profile/results_cached_text
```

Planning estimates for **both prompts and both budgets**, including fresh
loading/compilation, are DP **5–15 min**, Fast-WAM **20–60 min**, Cosmos
**20–60 min**. These are unmeasured allowances, not a timeout or a latency
claim. Compiler-cache state and native kernel compile time can extend them.
The exact measurement work is 800 complete calls/model; warmup adds 120, plus
a few eager/parity/tracing calls. Replays excluding warmup are 45,600 for DP,
6,000 for Fast-WAM and 4,000 for Cosmos across the two prompts.

CPU verification: **25 passed**. Stub tests also forbid CUDA initialization.
The six B-pool input inspections passed. Commands are safe while GPUs are occupied:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /data/mw_finetune/dp/env/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/profile/tests/test_latency.py

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /data/mw_finetune/dp/env/bin/python -m exp.r17_integration.profile.latency \
  --model cosmos_libero --suite libero_10 --inspect-input \
  --out exp/r17_integration/profile/input_check.json
```

`--inspect-input` returns before importing torch or invoking a loader.

## Results (2026-10-12, exclusive RTX 4090 48 GB, MPS off, batch 1, warmup 30, 200 iterations, all certified)

Medians in ms; coefficients share the full-step denominator (stage1 + stage2 + full loop).

| Model | Suite | Stage1 | Stage2 | Full loop | Reduced loop | Vision / full miss / reduced miss |
|---|---|---:|---:|---:|---:|---|
| DP | libero_10 | 78.1 | 0.4 | 160.8 (DDPM100) | 19.4 (DDIM10) | 0.326 / 0.674 / 0.083 |
| DP | spatial | 92.7 | 0.4 | 162.0 | 19.8 | 0.363 / 0.637 / 0.079 |
| Fast-WAM | libero_10 | 9.8 | 32.2 | 79.0 (K10) | 12.3 (K1) | 0.081 / 0.919 / 0.368 |
| Fast-WAM | spatial | 9.8 | 31.9 | 77.6 | 12.3 | 0.082 / 0.918 / 0.371 |
| Cosmos | libero_10 | 103.4 | 1.2 | 280.7 (K5) | 58.5 (K2, 1 NFE) | 0.268 / 0.732 / 0.155 |
| Cosmos | spatial | 109.5 | 1.7 | 284.3 | 59.2 | 0.277 / 0.723 / 0.154 |

Per-request text encoding as served: DP's stage1 includes CPU CLIP (74.5 / 88.6 ms); Fast-WAM's stage2
includes UMT5-XXL (~13 ms; `results_cached_text/`: stage2 19.2–19.5 ms, coefficients 0.093–0.099 /
0.901–0.907 / 0.290–0.292). With text encoded once per episode, DP becomes about 0.02 / 0.98 / 0.12.

Deviations from the pi0.5/GR00T recipe, each found on the real card:
- Fast-WAM: the MoT RoPE tables (complex128) stay on the CPU and traced 0-d scalars are CPU constants;
  both made cudagraph trees skip, so they are moved to the GPU. The uncompiled stage functions equal
  the adapter bit for bit; BF16 compilation moves context ~4% and the KV cache <=2% (relative
  Frobenius), actions ~1%, so stage2 is gated on relative error <=5% and actions at 0.03.
- Cosmos: the DiT uses Transformer Engine kernels (RMSNorm, fused RoPE) that bypass the dispatcher,
  so make_fx records only their output allocations and a traced/inductor denoiser returns
  uninitialised memory. Stage3 is therefore a CUDA graph captured from the native denoiser
  (`use_video_condition`, constant True, is fixed on the host before capture); stages 1–2 are traced
  and compiled as for the other models, with random numbers on the CUDA generator
  (`fallback_random`) and CPU-pinned factory ops moved to the GPU. Two compiled stages are certified.
- The `_local_scalar_dense` guard in `compile_tensor_program` runs after dead-code elimination and can
  miss host decisions; the action parity check is what caught the Cosmos problem.

## MT50

The four MW models run one `mt50` profile each, using row 0 of
`/data/r17_library/<model>_mt50/bpool`, its campaign's `assets/load.json`, and the
canonical prompt in `exp/metaworld/tasks.py`. Read the recorded single RGB480
camera unchanged (already rotated 180°) and state4; model adapters own resizing
and normalization. Batch 1, warmup 30 and 200 iterations **per budget**, CUDA
events, full-loop pricing and the existing certification/receipt schema are
unchanged. Receipts set `profile="mt50"`, `suite="mt50"` and record actual NFE.

| Model | Checkpoint | Stage1 (`prepare` + vision + retrieval key) | Stage2 | Stage3 (complete native `sample`; full / reduced K = NFE) |
| --- | --- | --- | --- | --- |
| `dp_mw` | `/data/mw_finetune/dp/mt50_policy_100000` | RGB480→96, CPU CLIP, state normalization, both checkpoint FiLM branches (native absent-camera policy); copy the single camera key | Concatenate checkpoint features | Epsilon network + DDPM100 / trailing DDIM10 eta0; 100 / 10 |
| `fastwam_mw` | `/data/mw_finetune/fastwam/mt50_frozen_b128/policy_016001` | RGB480→224, native Wan VAE, patch embedding, pooled key | T5, proprio conditioning, full video DiT KV prefill | Action expert + shifted-flow update; 10 / 3 |
| `pi05_mw` | `/data/ckpt/pi05_metaworld_rlinf` | Native input transforms, SigLIP and all three prefix slots + language tokens; native spatial key copied to CPU | Native PaliGemma prefix KV prefill | Native action expert + Euler loop, RNG and output conversion; 10 / 2 |
| `groot_mw` | `/data/mw_finetune/groot/4090_mt50/checkpoint-60000` | Native MW transforms, Eagle vision + language embedding/scatter before LLM; native spatial key copied to CPU | Native Eagle LLM + backbone projection | Native head prologue + Euler loop, RNG and output conversion; 4 / 2 |

All four compile three tensor programs with the same FX/reduce-overhead recipe.
π0.5 retains the native KV-cache container; GR00T's compiled step includes its
Euler update, with tensor `dt` shared by K4/K2. GR00T switches its native endpoint
manifest and live K together outside timing, and restores local substitutions.

Run later on the coordinator's exclusive 4090 (MPS off):

```bash
cd /home/weiland/projects/openpi
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m exp.r17_integration.profile.certify --model dp_mw --out-dir exp/r17_integration/profile/results_mt50
.venv/bin/python -m exp.r17_integration.profile.certify --model fastwam_mw --out-dir exp/r17_integration/profile/results_mt50
.venv/bin/python -m exp.r17_integration.profile.certify --model pi05_mw --out-dir exp/r17_integration/profile/results_mt50
.venv/bin/python -m exp.r17_integration.profile.certify --model groot_mw --out-dir exp/r17_integration/profile/results_mt50
```

Outputs are `<model>_mt50.json` / `.log`, plus nsys artifacts when available.
**Current input limitation:** all four existing MT50 pools declare
`metadata.images="disabled"`; their row-0 source blocks also omit camera bytes.
These default commands need the original observations restored to the pools.
Alternatively pass the existing `--observations FILE.npz --descriptor LOAD.json`
options with a real row-0 wire observation (RGB480, raw state4, prompt; leading
row axis). No camera is synthesized or reconstructed from retrieval keys.

CPU tests cover all budgets, single-camera bytes/prompt/state reading, native
loader and Python-island dispatch, single-profile launching, DP/Fast-WAM MW
dimensions, π0.5/GR00T stage cuts with real CPU FX extraction, retrieval keys,
native cache reconstruction, both GR00T endpoints/restoration, event intervals,
actual NFE and complete receipt/pricing output with stubs. CUDA initialization
is forbidden by the test fixture; GPU compilation/replay remains unmeasured.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/mw_finetune/dp/env/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/profile/tests/test_latency.py
```
