# Cosmos MT50 fine-tune package

CPU implementation and staging complete. **No GPU load, training, evaluation,
server or campaign was started.** The coordinator owns the first GPU probe and
closed-loop evaluation. A successful MT50 checkpoint does not exist yet.

## Recipe and reasons

See [RECIPE.md](RECIPE.md) for the complete settings and source evidence.
Initialize the full ~1.956B DiT from the existing NVIDIA LIBERO policy, preserving
its documented action prior. The community MW checkpoint's undisclosed protocol
and 0/152 coordinator outcome make it a poor default initialization; the
Predict2 video base would also have to learn the action representation anew.
Reuse only the community release's exact cached T5 prompt embeddings.

Use the native Cosmos Policy training step, hybrid EDM noise .01…200, native
FusedAdamW with BF16 parameters and FP32 master weights/moments, peak LR3e-5,
1000-update warmup, linear decay to .3 at30k then .06 constant. Train40k updates
at effective batch32 (~6.25 frame passes), checkpoint resume state every1k and
export serving weights every5k and at each clean stop. Initial candidates5k/10k,
then20k/40k. Native whole-block activation recomputation is enabled.

Planned H10080GB microbatch4×accumulation8; planned 48GB4090 microbatch1×32.
**These are memory estimates, not measured fits.** Fixed model/gradient/master/
moment storage is ~29.2GiB before VAE and activations. Probe real Adam allocation.
If needed, reduce microbatch/increase accumulation while retaining batch32.

Native 16×7 targets pad our xyz,g actions as `[xyz,0,0,0,g]`; 9-D proprio pads
our obs[:4] as `[g,-g,xyz,0,0,0,1]`. Empirical min/max comes from our complete
dataset, with constant channels centered using bounds mean±.5. No extra action
scale, sign change or adapter clipping. Execute5 predictions under the unchanged
R17 corner2/480/rot180, prompts, settle15, max160 steps and success-at-any-step
protocol. Seven latents/25 raw frames, one camera, three conditional latents.
The image path uses the existing lossless bilinear RGB224 cache; native strong
training augmentation and deterministic 90%-area inference crop; no extra JPEG.

The CPU source audit found **204,806 frames, 2,500 episodes, 2,465 successful
episodes and35 failures**. Use the official50/25/25 sample mixture: successful
demo actions / all-episode future states / all-episode values. Value targets use
measured terminal success and gamma .99, mapped to[-1,1]. Enable the native
per-objective loss mask (an explicit deviation from the LIBERO config default)
to focus demonstration supervision on pure policy actions.

LeRobot's prompt index order differs from R17 task positions. Bind prompt
indices through stored text, with only its push-back description rewritten to
the canonical R17 push wording. There are49 unique R17 prompt embeddings for50
tasks. Statistics on all four wire channels match original dataset metadata.
Neither hidden environment state nor new simulator frames enter training.

## Files and staged assets

Added under `exp/mw_finetune/cosmos/`:

- `mt50_data.py`: CPU statistics/success/prompt preparation; episode-local
  action/future targets; native sample layout; deterministic resumed sampler.
- `train_mt50.py`, `train_mt50.sh`: native model/loss/optimizer, `--probe N`,
  periodic saves, automatic resume, `--max-hours`, SIGTERM/SIGINT clean stop.
- `check_train_cpu.py`, `test_train_mt50.py`: native CPU audit and regressions.
- `RECIPE.md`, this handback.

Updated R17 `models/cosmos/metaworld.py`, `serve.py`, `test_metaworld.py`:
read each fine-tune export's preprocessing/statistics and accept its normal
checkpoint files; retain one-camera keys/debug capture and community defaults
for older directories. No launcher/concurrent-package regions were changed.
No vendor files were modified and no commit was made.

Local prepared assets: `/data/mw_finetune/cosmos/mt50_assets/`:
`dataset_statistics.json`, `t5_embeddings.pkl`, `episode_success.npy`,
`task_prompts.json`, `data_report.json`. No full image/data copy was created.
Both hosts already have `/data/mw_finetune/fastwam/mt50_224` and the LIBERO base.

An isolated training bundle was staged at
**H100 `/data/mw_finetune/cosmos/mt50_runtime`**, with assets in its `assets/`.
Bundle `/data/mw_finetune/cosmos/mt50_training.tar` is ~60MB; transfer used
`exp/r17_integration/ops/direct_transfer.py`, direct public
`149.165.153.233:23118`, rsync `--bwlimit=16384`. Tether carried commands only.
The live collection runtimes were not modified. The H100 raw LeRobot dataset
is absent, but its existing complete RGB224 cache suffices for this launcher.

## CPU verification

Workspace Python and Cosmos Python: **38 tests passed** in each environment.
Tests cover own-statistics action inverse, recorded preprocessing, serving
metadata, fake-model/capture behavior, episode sampler/resume, complete fake
training/probe/export and byte-exact CPU continuation, FP32 moment/master
preservation, bounded stop, and disabling inference-only VAE reseeding before
native training load. Loader worker RNG is isolated from model noise RNG.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:packages/openpi-client/src:. \
  .venv/bin/python -m pytest -q \
  exp/mw_finetune/cosmos/test_train_mt50.py \
  exp/r17_integration/models/cosmos/test_metaworld.py \
  exp/r17_integration/models/cosmos/test_capture.py
```

Native config/loss/sample audits **PASS locally and on H100**:

```json
{"passed":true,"cuda_initialized":false,"frames":204806,
 "native_shape":[3,25,224,224],"action_shape":[16,7],
 "loss_gradient_frames":[[3],[3],[4,5],[6]],
 "conditioning_pixel_parity":true,"task_tail_windows_checked":50,
 "optimizer":{"lr":0.00003,"master_weights":true},
 "sigma_min":0.01,"sigma_max":200,"activation_checkpoint":"block_wise"}
```

The native gradients fall on action/future/value slots as intended. Current
conditioning pixels agree between training and inference preprocessing. All50
task tail windows keep actions/proprio/images within their episode. Local
audit additionally preprocesses the four previously rendered real MT50
480² frames into `[1,3,25,224,224]` on CPU. H100 has no copy of those four
frames, so its real-env-frame check is omitted; full cached task tails pass.

Upstream unused Qwen/UMT5 imports and job-name validation query CUDA even for
a CPU config audit. The audit alone patches those import/byte-tensor calls;
it never instantiates a resident and asserts CUDA stays uninitialized. Training
and inference paths retain the real CUDA functions.

Evidence under local `mt50_assets/`: `cpu_tests.log`, `vendor_cpu_tests.log`,
`native_cpu_check.json/.log`, `h100_native_cpu_check.log`, `data_report.json`.
H100 audit JSON: `mt50_runtime/assets/native_cpu_check.json`.
Shell syntax and `git diff --check` pass. Print-only command parsing was also
checked without CUDA.

## Exact coordinator commands

Run these **only after collection releases the selected GPU**. Probe20 means
20 optimizer updates, including accumulation, real forward/backward and real
Adam allocation. `probe.json` reports every loss/step time, median excluding
the first two updates, action L1, peak allocated/reserved memory and projected
remaining hours. Save time is outside the per-update median but included in
total elapsed time. Use a fresh output directory for independent probes.

Local 4090 probe, from `/home/weiland/projects/openpi`:

```bash
CUDA_VISIBLE_DEVICES=0 exp/mw_finetune/cosmos/train_mt50.sh \
  --output /data/mw_finetune/cosmos/probe_4090 \
  --batch 1 --accum 32 --probe 20 --max-hours 0.5 --resume none
```

Local 4090 full training:

```bash
CUDA_VISIBLE_DEVICES=0 exp/mw_finetune/cosmos/train_mt50.sh \
  --output /data/mw_finetune/cosmos/mt50_4090 \
  --batch 1 --accum 32 --steps 40000 --lr 3e-5 \
  --save-every 1000 --export-every 5000 --max-hours 24 --resume auto
```

H100 probe (runtime and assets are already staged):

```bash
tether exec --timeout 1h --cwd /data/mw_finetune/cosmos/mt50_runtime h100 -- \
  env CUDA_VISIBLE_DEVICES=0 bash exp/mw_finetune/cosmos/train_mt50.sh \
  --assets /data/mw_finetune/cosmos/mt50_runtime/assets \
  --output /data/mw_finetune/cosmos/probe_h100 \
  --batch 4 --accum 8 --probe 20 --max-hours 0.5 --resume none
```

H100 full training:

```bash
tether exec --timeout 26h --cwd /data/mw_finetune/cosmos/mt50_runtime h100 -- \
  env CUDA_VISIBLE_DEVICES=0 bash exp/mw_finetune/cosmos/train_mt50.sh \
  --assets /data/mw_finetune/cosmos/mt50_runtime/assets \
  --output /data/mw_finetune/cosmos/mt50_h100 \
  --batch 4 --accum 8 --steps 40000 --lr 3e-5 \
  --save-every 1000 --export-every 5000 --max-hours 24 --resume auto
```

Reinvoke the full command after each24h allocation to resume the same output.
The time bound includes initialization, is checked before each microstep, and
allows the current microstep plus final save to finish. Partial accumulated
gradients are discarded; resume starts from the last completed optimizer
update. Serving exports and `training_state.pt` contain different payloads:
only the latter carries optimizer/master/scheduler/RNG state. Allow ~90GB peak
additional storage for atomic resume replacement and eight serving milestones.
Each extra bounded-stop export adds ~4GB; each independent probe directory
retains ~31GB of resume/serving state.

If code/assets change, refresh the isolated bundle through the same direct
helper, never a tether file-transfer command. From the workspace:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python - <<'PY'
from pathlib import Path
import tarfile
from exp.r17_integration.qualification.deploy_cpu import runtime_archive
from exp.r17_integration.ops import direct_transfer, formal_eval
root = Path.cwd()
archive = Path('/data/mw_finetune/cosmos/mt50_training.tar')
runtime_archive(root, archive)
with tarfile.open(archive, 'a') as tar:
    for name in ('RECIPE.md', 'train_mt50.sh'):
        path = root / 'exp/mw_finetune/cosmos' / name
        tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
    for name in ('dataset_statistics.json', 't5_embeddings.pkl', 'episode_success.npy',
                 'task_prompts.json', 'data_report.json'):
        tar.add(Path('/data/mw_finetune/cosmos/mt50_assets') / name,
                arcname='assets/' + name, recursive=False)
direct_transfer.transfer(['rsync', 'push', str(archive),
    'h100:/data/mw_finetune/cosmos/mt50_training.tar'],
    remote=formal_eval.remote, execute=formal_eval.execute)
formal_eval.remote('h100', ['python3', '-c',
    "import tarfile; from pathlib import Path; p=Path('/data/mw_finetune/cosmos/mt50_runtime'); "
    "p.mkdir(parents=True,exist_ok=True); "
    "tarfile.open('/data/mw_finetune/cosmos/mt50_training.tar').extractall(p)"], seconds=60)
PY
```

## Estimates, adapter loading and remaining work

Unmeasured budget estimates based on official LIBERO throughput and single-GPU
small-batch/recomputation overhead:

| Device | First5k evaluation candidate | Full40k compute |
|---|---:|---:|
| H10080GB | 10–25h | 80–200h |
| Local48GB4090 | 25–60h | 200–480h |

Add startup/checkpoint/evaluation time. Replace these ranges with probe median
seconds/update × remaining updates /3600. A5k candidate is not a promised
usable policy. Proposed acceptance: at least50% mean MT50 success, then improve
toward75%; use per-task closed-loop results rather than training L1 alone.

R17 loads the exported directory directly; it reads that directory's
`dataset_statistics.json`, prompt cache and `config.json`. No weights/statistics
are substituted from the community directory. Keep training and evaluation on
the same GPU type where practical, as upstream recommends.

Example coordinator evaluation after an actual5k checkpoint exists:

```bash
# H100, using the corresponding trained output on that host.
exp/r17_integration/ops/formal_eval.sh --model cosmos_mw \
  --checkpoint /data/mw_finetune/cosmos/mt50_h100/checkpoint-005000 \
  --server-node h100 --worker-node timan107 --replicas 1 --workers 16 \
  --gpu cuda:0 --memory-budget-gib 20 --port 23160 --driver-port 23169 \
  --seconds 86400 --episode-seconds 1800 --retries 2 --run

# Local4090, using its independently trained output.
exp/r17_integration/ops/formal_eval.sh --model cosmos_mw \
  --checkpoint /data/mw_finetune/cosmos/mt50_4090/checkpoint-005000 \
  --server-node local --worker-node timan107 --replicas 1 --workers 16 \
  --gpu cuda:0 --memory-budget-gib 20 --port 23160 --driver-port 23169 \
  --seconds 86400 --episode-seconds 1800 --retries 2 --run
```

These evaluation commands were not executed. The stock MT50 benchmark seed is7.
The community path remains the launcher's default; always pass the explicit
fine-tuned checkpoint. GPU memory fit, native backward/optimizer execution,
real GPU resume and final success rate remain unverified until the coordinator
runs the probe/training/evaluation. No CPU import or data-preparation blocker
remains on either host.
