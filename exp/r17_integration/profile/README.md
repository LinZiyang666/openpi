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

Run these **after experiments finish**, on the exclusive 4090. Change the
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

A direct single-prompt example, with torch counters only:

```bash
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
