# C — Fast-WAM MT50 handback

Use **frozen_video, microbatch 128, accumulation 1, global 128, 16,001 updates** under the coordinator's ≤24 h choice rule. Both frozen-video and video-LoRA launch modes are implemented. The chosen run trains the full action expert and reset 4-D action/proprio layers from LIBERO, with current-frame-only training and λ_video=0. [RECIPE.md](RECIPE.md) records sources, exact recipes and deviations. The coordinator starts long training and formal evaluation; C has run short probes only.

## Installed artifacts

| Artifact | Location |
|---|---|
| H100 Python 3.12 environment | `/data/mw_finetune/envs/fastwam` |
| Pinned LeRobot / official raw-Wan loader | `/data/mw_finetune/vendor/lerobot`, `/data/mw_finetune/vendor/FastWAM` |
| LIBERO initialization | `/data/mw_finetune/models/fastwam_libero` |
| Existing raw VAE/T5/tokenizer | `/data/xwam/wan22_5b` — read-only, no duplicate conversion/download |
| Full prepared 224² MT50 | `/data/mw_finetune/fastwam/mt50_224` on local and H100 |
| H100 training code | `/data/mw_finetune/fastwam/code/exp/mw_finetune/fastwam` |
| H100 serving code in A's checkout | `/home/exouser/openpi/exp/mw_finetune/fastwam` |

Environment: Torch 2.7.1+cu128, torchvision 0.22.1+cu128, transformers 5.5.4, diffusers 0.39.0. Full package inventory: [reports/h100_environment.txt](reports/h100_environment.txt). Revisions: LeRobot `b9cb121cb7d4e3e26ec5c906d08dda68d151ba52`; FastWAM `7faa71108368fbb3b6885649f112af607427a2d4`; LIBERO HF `53983e1249b4eb4d89ab42b40b8678a882d331ad`; MT50 HF `b46a1c185a2368660650a82ac5a5083a56722f48`. The video DiT comes directly from the LIBERO checkpoint; the raw VAE/T5 loader reuses the existing Wan files strictly, so no raw DiT or converted Diffusers copy is needed.

The complete prepared data is 12,133,230,228 bytes across 492 SQLite shards, plus float32 raw 4-D state/action index and metadata. All 204,806 frames /2,500 episodes /50 task keys are audited; fps 80 preserved. One camera is bilinearly resized 480→224 without rotation/crop/flip. Episode windows never cross boundaries. Index SHA256: `7bd1c896d574117c4d8d7db3ba5c97c6b131b1824e97fb6d4a10b96130bf70f1`. H100 transfer and full audit passed; [reports/h100_full_data.json](reports/h100_full_data.json) matches the local fingerprint. The final full-data batch128 probe is recorded below.

## Rebuild / resync commands

Run from `/home/weiland/projects/openpi` on weilandserver. All mutations target C's subtree or its new `/data/mw_finetune` artifacts.

```bash
tar -czf /tmp/r17_fastwam_code.tar.gz --exclude=__pycache__ --exclude=reports exp/mw_finetune/fastwam
tether push /tmp/r17_fastwam_code.tar.gz h100:/data/mw_finetune/fastwam/code.tar.gz --force
tether exec --safe --timeout 60s h100 /bin/bash -c 'mkdir -p /data/mw_finetune/fastwam/code; tar -xzf /data/mw_finetune/fastwam/code.tar.gz -C /data/mw_finetune/fastwam/code; tar -xzf /data/mw_finetune/fastwam/code.tar.gz -C /home/exouser/openpi'
tether exec --safe --timeout 60m h100 /bin/bash /data/mw_finetune/fastwam/code/exp/mw_finetune/fastwam/setup.sh
```

Data commands run locally, reusing A's pinned source download; H100 receives only the 224² copy. Preparation resumes complete shards and refuses incomplete input unless explicitly marked a smoke subset. Transfer reuses one bounded tar file and deletes nothing.

```bash
bash exp/mw_finetune/fastwam/download_data.sh
/data/mw_finetune/envs/fastwam/bin/python -m exp.mw_finetune.fastwam.prepare_data --workers 8
.venv/bin/python -m exp.mw_finetune.fastwam.sync_data
.venv/bin/python -m exp.mw_finetune.fastwam.verify_data
tether exec --safe --timeout 5m h100 /bin/bash -c 'cd /data/mw_finetune/fastwam/code; /data/mw_finetune/envs/fastwam/bin/python -m exp.mw_finetune.fastwam.verify_data --output /data/mw_finetune/fastwam/full_data_verified.json'
```

`tether exec --safe` avoids observed command-spawn stalls; flags belong before `h100`. C's disk gate preserves 30 GB free and refuses a shared-root projection above 120 GB. No source data or existing checkpoints were deleted.

## Probe and train

The launcher checks free GPU memory, takes `flock /tmp/r17_gpu.lock`, caps probe execution at 900 s including initialization, and writes `train.jsonl` plus `probe.json`. `--probe 200` counts **microsteps**, so 128×1 gives 200 optimizer updates and 16×8 gives 25. The probe-only time cap defaults to 840 s; inspect `completed_microsteps` and `stop_reason`. Peak memory includes actual FP32 Adam moments. Probes save no checkpoint unless `--save-probe` is given.

```bash
tether exec --safe --timeout 30m h100 /bin/bash -c 'cd /data/mw_finetune/fastwam/code; bash exp/mw_finetune/fastwam/launch.sh --mode frozen_video --micro-batch 128 --global-batch 128 --probe 200 --output /data/mw_finetune/fastwam/probe_full_frozen_b128'
tether exec --safe --timeout 30m h100 /bin/bash -c 'cd /data/mw_finetune/fastwam/code; bash exp/mw_finetune/fastwam/launch.sh --mode video_lora --micro-batch 32 --global-batch 128 --probe 200 --output /data/mw_finetune/fastwam/probe_full_lora_b32'
```

Coordinator's selected long run (launch or restart with the same command):

```bash
tether exec --safe --timeout 24h h100 /bin/bash -c 'cd /data/mw_finetune/fastwam/code; bash exp/mw_finetune/fastwam/launch.sh --mode frozen_video --micro-batch 128 --global-batch 128 --steps 16001 --checkpoint-every 2000 --output /data/mw_finetune/fastwam/mt50_frozen_b128 --resume > /data/mw_finetune/fastwam/mt50_frozen_b128.console.log 2>&1'
```

Alternative LoRA run: replace mode with `video_lora`, microbatch with 32, and output with `mt50_lora_b32`. Its 64×2 maximum measured fit needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and has little headroom; 32×4 is its safer default. Both use AdamW (.9,.95), weight decay .01, 800 warmup steps, cosine 1e-4→1e-6, clip 1, activation checkpointing, BF16 activations and FP32 trainables/moments. LoRA adds 300 video linear adapters with rank=alpha=128. Frozen has 1,020,900,356 trainables; LoRA has 1,343,337,476.

`latest.pt` contains trainable tensors, optimizer, CPU/CUDA RNG and exact sampler offset; `--resume` auto-loads it if present. Recipe/data/seed/batch mismatches fail. Every 2k updates save the replaceable serving `policy_latest` and optimizer snapshot. Final immutable `policy_016001` is saved automatically; use it for A's evaluator. Default `--milestone-every 16001` preserves shared disk headroom; additional immutable snapshots require more room. A frozen resume snapshot is about 12.25 GB and serving delta 4.08 GB; LoRA 16.12 GB/5.37 GB. The frozen run adds about 20.42 GB total and needs 12.25 GB transient atomic-write headroom; do not copy raw Wan/T5 into checkpoints.

After the full transfer, shared-root allocated size measured **85.31 GB**, with **116.53 GB free** on `/data`. Selected frozen training plus final snapshot and conservative atomic-write allowance totals about 117.98 GB of the 120 GB shared budget. The LoRA alternative needs about 8.3 GB more budget headroom than currently available to retain its optimizer/latest/final deltas; arrange that space before a long LoRA run. The selected frozen run fits without deleting the probe artifacts. [reports/h100_final_disk_and_code.txt](reports/h100_final_disk_and_code.txt) records this measurement; each checkpoint rechecks current usage.

## Serve through A's harness

`exp.mw_finetune.fastwam.policy.load_policy(checkpoint, device, **kw)` returns a stateless`infer/reset` object. It consumes the 480² RGB frame, raw 4-D state, canonical prompt and per-decision seed; returns 32×4 finite float32 denormalized commands. Ten action steps / CFG1; no gripper inversion, output clipping, queue, cross-request cache or online adaptation. Optional`base`, `wan_root`, `denoising_steps` override metadata paths/defaults. Checkpoint schema is`fastwam-mt50-delta-v1` with strict parameter/shape auditing.

H100 uses direct NAT port 23252 per A's current CONTRACT; check that it is free before launching. `serve_compat` delegates to **A's actual server** and supplies only a read-only task-table snapshot if A's H100 bundle lacks`exp.metaworld.tasks`.

```bash
tether exec --safe --timeout 30s h100 /bin/bash -c 'ss -tlnp | rg ":23252 " || true'
tether exec --safe --timeout 24h h100 /bin/bash -c 'nvidia-smi --query-gpu=name,memory.free --format=csv; cd /home/exouser/openpi; export PYTHONPATH=/home/exouser/openpi:/home/exouser/openpi/src:/home/exouser/openpi/packages/openpi-client/src; exec flock /tmp/r17_gpu.lock /data/mw_finetune/envs/fastwam/bin/python -m exp.mw_finetune.fastwam.serve_compat --model fastwam --checkpoint /data/mw_finetune/fastwam/mt50_frozen_b128/policy_016001 --port 23252'
```

A owns timan108 EGL workers/evaluation. Use its HANDBACK_A commands against`149.165.153.233:23252`,500 episodes, historical 5-action replanning, and difficulty groups 28/11/6/5. Five-action replanning differs from SmolVLA's one-action simulation inference; disclose it. No success-rate evaluation has been claimed for C's smoke checkpoints.

## Measured verification

| H100 run | Microbatch × accumulation | Microsteps/s | Seconds/update | Peak allocated/reserved GB | Compute projection 16,001 updates |
|---|---:|---:|---:|---:|---:|
| frozen200-microstep probe |16×8 |1.3394 |5.9728 |34.49/35.63 |26.55 h |
| LoRA200-microstep probe |16×8 |.7691 |10.4015 |47.18/47.92 |46.23 h |
| frozen largest-batch sweep |128×1 |.2848 |3.5111 |61.96/62.92 |15.61 h |
| **frozen full-data32-update check** |**128×1** |**.2857** |**3.5000** |**61.97/62.91** |**15.56 h** |
| LoRA batch 32 sweep |32×4 |.4867 |8.2186 |58.39/63.51 |36.53 h |
| LoRA fresh 64 expandable-allocator sweep |64×2 |.2372 |8.4313 |80.34/83.92 |37.47 h |

Both 200-microstep probes completed 25 optimizer updates on the 1,331-frame smoke subset. Warmup excluded 10 timed microsteps; LoRA probe total 299 s including 19 s checkpoint write. Batch sweeps time only two steady optimizer updates and exclude first-update state allocation; they are estimates, not accuracy measurements or guaranteed wall-clock durations. LoRA 128 OOMed;64 subsequently OOMed from fragmentation with the default allocator before fitting in a fresh expandable-allocator process. See`reports/h100_*` JSON/logs for original numbers and gradients: zero video-base gradients,824 action tensors, nonzero proprio, and600 LoRA adapter gradients in LoRA mode.

LoRA 32 synchronized VAE profile measured .2809 s VAE work per 2.3193 s microstep. Hypothetical zero-cost cached latents still project36.24 h, excluding preprocessing and cache I/O. This does not meet 24 h; no latent-cache copy was built. [reports/h100_lora_vae_profile.json](reports/h100_lora_vae_profile.json).

Selected frozen full-data check completed **32/32 optimizer updates**, excluding the first 4 from timing: **3.49998 s/update**, **36.5716 samples/s**, **61.9655 GB allocated /62.9146 GB reserved**, **15.5565 h compute projection**. Total131.788 s including initialization; all 50 training instruction embeddings cached. The data manifest is full, not partial. The probe wrote no additional model checkpoint. [reports/h100_full_frozen_b128.json](reports/h100_full_frozen_b128.json).

Exact final full-data smoke command:

```bash
tether exec --safe --timeout 30m h100 /bin/bash -c 'cd /data/mw_finetune/fastwam/code; bash exp/mw_finetune/fastwam/launch.sh --mode frozen_video --micro-batch 128 --global-batch 128 --probe 32 --probe-warmup 4 --max-seconds 500 --output /data/mw_finetune/fastwam/probe_full_frozen_b128_short'
```

Mechanical checks:

- Seven CPU tests passed, including min/max inversion outside training range, episode padding, deterministic stateless wrapper, strict deltas/alias mapping, LoRA gradients, and frozen-path loss/proprio-gradient equivalence to full causal attention. Run:`/data/mw_finetune/envs/fastwam/bin/python -m pytest exp/mw_finetune/fastwam/tests -q`.
- Real frozen and LoRA delta reloads each passed two exact same-seed predictions,32×4 float32 finite. Frozen first/warm inference 1.159/.484 s,25.11 GB peak; LoRA1.210/.455 s,25.75 GB peak. These are training-observation mechanics, not env success.
- A's actual WebSocket server passed`select_bundle`, `episode_start`, `infer`, `episode_end`; repeated reach episode produced exactly identical actions. First/warm request 1.144/.433 s; total 15.21 s. [reports/h100_wire_smoke.json](reports/h100_wire_smoke.json) records A's metadata/checkpoint identity.
- Resume smoke restored the saved 25-update state and completed one more update (26 total), finite loss,33.64 s including load. [reports/h100_resume_smoke.json](reports/h100_resume_smoke.json).

Re-run serving smoke on H100, holding the shared lock:

```bash
tether exec --safe --timeout 5m h100 /bin/bash -c 'cd /home/exouser/openpi; export PYTHONPATH=/home/exouser/openpi:/home/exouser/openpi/src:/home/exouser/openpi/packages/openpi-client/src R17_GPU_LOCK_HELD=1; nvidia-smi --query-gpu=name,memory.free --format=csv; exec flock /tmp/r17_gpu.lock timeout 180 /data/mw_finetune/envs/fastwam/bin/python -m exp.mw_finetune.fastwam.smoke_wire --checkpoint /data/mw_finetune/fastwam/probe_frozen_b16/policy_latest --data /data/mw_finetune/fastwam/smoke_224 --output /data/mw_finetune/fastwam/wire_smoke.json'
```

The saved`probe_frozen_b16/policy_latest` and`probe_lora_b16/policy_latest` use partial data and25 updates. They validate reload/serve only. Formal comparison must use a full-data immutable training milestone.
