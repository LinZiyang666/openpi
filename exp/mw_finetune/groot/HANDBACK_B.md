# B: GR00T N1.5 MT50 handback

Ready for coordinator training on either host. Use **60k optimizer updates**,
LR 1e-4, NVIDIA AdamW/cosine, frozen VLM/vision, tuned projectors **and DiT**,
horizon 16, batch 64/accumulation 1 on 4090 or batch 128/accumulation 1 on H100. Sources and every
deviation are in [RECIPE.md](RECIPE.md). No long training/evaluation was launched.

## Prepared data and measurements

Both hosts have `/data/mw_finetune/datasets/metaworld_mt50_groot` (2,500 episodes,
204,806 frames, 50 tasks, 80 Hz,~8.8 GB) and pinned base weights at
`/data/mw_finetune/models/groot_base`. Raw source (~45.9 GB) is local at
`/data/mw_finetune/datasets/metaworld_mt50`. Conversion preserves raw RGB,
state/action units and episode boundaries; MP4 is lossless, vectors float32,
stats exact over all rows, action tail padding zero. Full training rejects
smoke-only, incomplete or unverified data.

Both full-data probes ran 200 updates with the 60k LR schedule and full head:

| GPU | Batch / accum | Optimizer it/s | s/update | Examples/s | Allocated / reserved GiB | Callback wall | 60k projection |
|---|---|---:|---:|---:|---:|---:|---:|
|4090 48 GB|64 /1|0.7909|1.2643|50.62|30.50 /32.63|293.88s|21.07h|
|H100 80 GB|128 /1|0.8437|1.1853|107.99|44.56 /51.79|252.27s|19.75h|

Timing excludes the first 10 updates and includes data wait; wall projections
exclude saves, validation and queue time. Use our JSON `optimizer_it_per_s`;
HF's early-stop throughput uses the original 60k denominator and is invalid.
Raw JSON/logs are in `reports/`; small-data/frozen-DiT reports are legacy evidence.

CPU: **12 tests passed on both hosts**; full statistics/task counts and action
normalization roundtrips verified on both. Locally, 36 first/middle/last frames
across 12 spread episodes exactly matched source state/action/pixels. All archive
pieces copied to H100 passed SHA256 checks; H100 verification repeated data checks.
Standard 4090 smoke saved/reloaded checkpoint-2 and produced finite 16x4 actions
(23.02 GiB allocated/23.70 GiB reserved, 31.03s callback wall). Actual resume loaded
checkpoint-2, ran updates 3–4, saved/reloaded checkpoint-4 and inferred 16x4
(23.02/23.65 GiB, 40.52s). Both snapshots retained,~16.14 GB each.

4090 batch128 with native allocator OOMed on update 3 (12.09 GiB reserved unused).
64 is the largest passed batch among 64/128; this is a bounded search.
The optional expandable-allocator retry was closed while queued behind A's
formal lane; no CUDA work started. No B GPU job/server remains running.

## Setup, conversion and CPU tests

The matching islands already exist: local vendor
`/home/weiland/projects/openpi_ext/third_party/gr00t_n15`, H100 vendor
`/home/exouser/gr00t_n15`, both pinned commit 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96.
Python 3.11,Torch 2.5.1/CUDA 12.4,Transformers 4.51.3,flash-attn 2.7.4.post1;
relevant package versions are in `reports/local_environment.json`.

```bash
cd /home/weiland/projects/openpi
bash exp/mw_finetune/groot/setup.sh 4090
export PYTHONPATH=/home/weiland/projects/openpi_ext/third_party/gr00t_n15:$PWD
export NO_ALBUMENTATIONS_UPDATE=1
GROOT_PY=/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python
"$GROOT_PY" -m exp.mw_finetune.groot.convert_data --workers 8
"$GROOT_PY" -m exp.mw_finetune.groot.verify_data \
  --dataset /data/mw_finetune/datasets/metaworld_mt50_groot \
  --source /data/mw_finetune/datasets/metaworld_mt50 --samples 12
"$GROOT_PY" -m pytest -q exp/mw_finetune/groot/tests
```

For a fresh H100 copy, choose one transfer path, then setup/verify. Direct HTTP
uses B's assigned port 23151 temporarily; it must be free. `--code-only` refreshes
B code without recopying data. Both paths preserve staging files; completed copy
retained local `/tmp/r17_groot_data.XPsdy6/` and H100
`/tmp/r17_groot_http_recovery/` (~8.8 GB each), plus small code archives.

```bash
bash exp/mw_finetune/groot/sync_h100.sh
# Broker fallback, SHA256 checked and tested successfully:
python3 exp/mw_finetune/groot/sync_h100_http.py
tether exec --cwd /tmp --safe --timeout 10m h100 bash -lc 'cd /home/exouser/openpi; \
  bash exp/mw_finetune/groot/setup.sh h100; \
  PYTHONPATH=/home/exouser/gr00t_n15:$PWD NO_ALBUMENTATIONS_UPDATE=1 \
  /home/exouser/gr00t_n15_venv/.venv/bin/python -m exp.mw_finetune.groot.verify_data \
  --dataset /data/mw_finetune/datasets/metaworld_mt50_groot --samples 12'
```

## Smoke, probe and coordinator training

Launchers acquire `/tmp/r17_gpu.lock`, print/check free memory, and bound
smoke/probe execution to 880s (queue time outside). Smoke data exists on both
hosts at `.../metaworld_mt50_groot_smoke` (32 episodes, 2,852 frames, one task).
Re-running smoke automatically resumes and adds two more updates.

```bash
bash exp/mw_finetune/groot/smoke_4090.sh
bash exp/mw_finetune/groot/smoke_4090.sh --resume required
bash exp/mw_finetune/groot/launch_4090.sh --probe 200 --batch-size 64 --workers 12 \
  --steps 60000 --output /data/mw_finetune/groot/4090_standard_probe_b64
tether exec --cwd /tmp --safe --timeout 30m h100 bash -lc 'cd /home/exouser/openpi; \
  bash exp/mw_finetune/groot/launch_h100.sh --probe 200 --batch-size 128 --workers 12 \
  --steps 60000 --output /data/mw_finetune/groot/h100_standard_probe_b128'

# Coordinator only: identical command/output resumes optimizer, scheduler and RNG.
bash exp/mw_finetune/groot/launch_4090.sh --batch-size 64 --workers 12 \
  --steps 60000 --save-steps 30000 --output /data/mw_finetune/groot/4090_mt50
tether exec --cwd /tmp --safe --timeout 48h h100 bash -lc 'cd /home/exouser/openpi; \
  bash exp/mw_finetune/groot/launch_h100.sh --batch-size 128 --workers 12 \
  --steps 60000 --save-steps 30000 --output /data/mw_finetune/groot/h100_mt50'
```

Keep settings/output identical for resume; a mismatch is rejected. Incomplete
snapshots are renamed/preserved and skipped. Save at 30k/60k; no pruning or
parent-directory duplicate export. Reserve ~34.3 GB for two full snapshots.
H100 shared tree was 85.31 GB after setup; launcher checks 120 GB budget and30 GB
free floor. Two snapshots plus margin project ~119.6 GB; reserve that space across agents.
The quota is rechecked immediately before each save.
Logs: `/data/mw_finetune/groot/logs/`; TensorBoard:`<output>/runs/`; config,
environment and measurements JSON accompany runs. Native allocator is retained.

OOM fallback: first 4090 `--batch-size 32 --accumulation 2` or H100
`--batch-size 64 --accumulation 2`; `--no-tune-diffusion` is a labeled last
fallback. Optional larger 4090 batch requires a passing new probe:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash exp/mw_finetune/groot/launch_4090.sh --probe 200 --batch-size 128 --workers 12 \
  --steps 60000 --output /data/mw_finetune/groot/4090_standard_probe_b128_expandable
```

## A's serving contract

`from exp.mw_finetune.groot.policy import load_policy`; pass checkpoint/device,
then `reset()` and `infer(obs)["actions"]`: fresh physical float32 (16,4),
x/y/z/gripper, four denoising steps, caller-seeded global Torch RNG.
Tag:`metaworld_sawyer`, new-head slot 31; checkpoint normalization required.
No history/queue, clipping, scaling or gripper inversion; vendor files unchanged.

```bash
# weilandserver → ziyanglin.com:23151
bash exp/mw_finetune/groot/serve_local.sh \
  --checkpoint /data/mw_finetune/groot/4090_mt50/checkpoint-60000 --port 23151
# H100 → 149.165.153.233:23251; A's harness/dependencies are present.
tether exec --cwd /tmp --safe --timeout 48h h100 bash -lc 'cd /home/exouser/openpi; \
  GR00T_HARDWARE=h100 bash exp/mw_finetune/groot/serve_local.sh \
  --checkpoint /data/mw_finetune/groot/h100_mt50/checkpoint-60000 --port 23251'
```

A owns 500-episode evaluation and five-action replanning in CONTRACT.md. Smoke
weights are mechanics only. Source push-back text is `Push the puck back toward
the robot to the goal`; A keeps RLinf's `Push the puck to a goal` for that task.
B preserves source annotations and passes A's prompt unchanged; disclose this
one-task prompt shift. Numeric source task IDs are not harness table indices.
