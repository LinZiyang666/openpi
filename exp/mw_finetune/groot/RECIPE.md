# GR00T N1.5 on MetaWorld MT50

This is a GR00T recipe adapted to the SmolVLA MT50 dataset. It is not a
reproduction of SmolVLA's model or NVIDIA's eight-GPU LIBERO training.

## Sources and pins

The existing local and H100 checkouts both resolve to NVIDIA N1.5 commit
`4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`. References below use that commit,
not today's main branch (which trains newer GR00T versions).

* [N1.5 README, lines 234–254](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/README.md#L234): fine-tune with `scripts/gr00t_finetune.py`; maximize the batch that fits and train 20k steps. Line 247 advises freezing the diffusion model on a 4090. The VLM remains frozen ([line 72](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/README.md#L72)).
* [Fine-tune CLI, lines 57–119](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/scripts/gr00t_finetune.py#L57): parser defaults batch 32, 10k steps, LR 1e-4, weight decay 1e-5, warmup ratio .05, accumulation 1; `tune_llm=False`, `tune_visual=False`, `tune_projector=True`, `tune_diffusion_model=True`. README's recommended 20k overrides the CLI's 10k default.
* [TrainingArguments, lines 346–382](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/scripts/gr00t_finetune.py#L346): bf16 autocast, TF32, AdamW beta (.95,.999), epsilon 1e-8, cosine LR to zero, no gradient checkpointing. Warmup is 1,000 optimizer updates in a 20k run. `--gradient-accumulation-steps` counts microbatches per optimizer update; `max_steps` counts optimizer updates.
* [NVIDIA LIBERO N1.5 recipe, lines 17–25 and 118–130](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/examples/Libero/README.md#L17): an official recipe exists. Spatial/Object: 20k, batch 128, accumulation 1; Goal: 20k, batch 72, accumulation 4, mean/std action normalization except gripper; 90/Long: 60k, batch 128, accumulation 1. The example launches eight GPUs. The CLI defines batch per GPU, so these are not single-H100 batches or global batch 128. NVIDIA describes these results as minimally tuned demonstrations.
* [LIBERO data config, lines 31–113](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/examples/Libero/custom_data_config.py#L31): horizon 16, padded state/action dimensions 64/32, 224² resize after .95 crop, color jitter, and min/max normalization. We retain these choices, with one camera and four channels.
* [SmolVLA paper §4.1 and §4.3](https://arxiv.org/html/2506.01844v1#S4): MT50 contains 50 demonstrations per task (2,500 episodes), with 10 evaluation trials/task, 500 total; report easy/medium/hard/very-hard groups and the task average. Simulation fine-tuning uses 100k steps, batch 64. The implementation paragraph specifies AdamW (.9,.95), LR 1e-4, cosine to 2.5e-6, 100 warmup steps; that paragraph describes pretraining explicitly, rather than a separately tabulated MT50 optimizer. We use these as the requested MT50 reference settings, not GR00T's defaults.

Base weights: [`nvidia/GR00T-N1.5-3B`](https://huggingface.co/nvidia/GR00T-N1.5-3B),
revision `869830fc749c35f34771aa5209f923ac57e4564e`.
Source: [`lerobot/metaworld_mt50`](https://huggingface.co/datasets/lerobot/metaworld_mt50),
revision `b46a1c185a2368660650a82ac5a5083a56722f48` (main's LeRobot v3 layout).
The release's Python 3.10 recommendation is replaced with the already-validated
Python 3.11 islands, Torch 2.5.1 CUDA 12.4, Transformers 4.51.3,
flash-attn 2.7.4.post1, NumPy 1.26.4, Accelerate 1.2.1, PyAV 12.3.0, Decord 0.6.0.

The official entry point is `python scripts/gr00t_finetune.py`. Our `train.py`
passes its `ArgsConfig` directly with `tune_llm=False`, `tune_visual=False`,
`tune_projector=True`, `tune_diffusion_model=True`, `learning_rate=1e-4`,
`gradient_accumulation_steps=1`, `max_steps=60000`, `num_gpus=1`,
`embodiment_tag="metaworld_sawyer"` and the custom data config. This requires
our in-process tag registration; invoking the unmodified vendor CLI alone
does not know the new Sawyer tag. README's literal 4090 fallback flag is
`--no-tune_diffusion_model`; our equivalent is `--no-tune-diffusion`.

Launchers retain the native allocator used by the successful probes. An
optional retry can set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. A native-allocator 4090 batch-128 attempt
failed on its third update with 12.09 GiB reserved but unused. The setting is
an allocator change, not a change to loss or trainable modules; it aims to
reduce fragmentation from variable shapes. See the
[PyTorch 2.5.1 CUDA memory documentation](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst).
The actual allocator and OOM evidence are recorded in environment JSON/logs.

## Concrete hardware plans

| Setting | One H100 80 GB | Modified RTX 4090 48 GB |
|---|---|---|
| Per-GPU batch | 128 | 64 (full-data probe passed; 128 OOM) |
| Accumulation / effective batch | 1 / 128 | 1 / 64 |
| Optimizer updates | 60,000 | 60,000 |
| LR / warmup / scheduler | 1e-4 / 3,000 / cosine to zero | same |
| AdamW / weight decay | (.95,.999), epsilon 1e-8 / 1e-5 | same |
| LLM and visual tower | frozen | frozen |
| `tune_projector` | true | true |
| `tune_diffusion_model` | true | true |
| Checkpoint interval | 30,000 + final | 30,000 + final |
| Decoding / CPU workers | Decord / 12 | Decord / 12 |

The pinned repo has no NVIDIA MT50 recipe. At the coordinator's request, the
chosen **60k-step proposal** follows the LIBERO-90 multitask recipe because MT50
has 50 tasks, rather than the generic 20k README recommendation. This is an
analogy to task breadth, not an established NVIDIA MT50 result. The original
20k option remains available through `--steps 20000`.

Both cards tune the standard full action head. The README's 4090 warning targets
the normal 24 GB card; this machine has 48 GB, so freezing DiT is not justified
as the default. Use `--no-tune-diffusion` only as an explicitly labeled OOM
fallback, after first lowering microbatch to 32 and accumulating two updates on
4090 to retain effective batch 64. For H100, batch 64/accumulation 2 retains its
effective batch 128. Preliminary frozen-DiT probes remain in `reports/` for
diagnostics; they are not the recommended fine-tune.

Full-MT50 200-update probes with the 60k LR schedule measured:

| GPU / batch / accumulation | Optimizer it/s | Seconds/update | Examples/s | Peak allocated / reserved GiB | Projected 60k hours |
|---|---:|---:|---:|---:|---:|
| H100 / 128 / 1 | 0.8437 | 1.1853 | 107.99 | 44.56 / 51.79 | 19.75 |
| 4090 / 64 / 1 | 0.7909 | 1.2643 | 50.62 | 30.50 / 32.63 | 21.07 |

Times include data wait and exclude the first 10 updates. Projections exclude
checkpoint saves, queue time and validation. These are validated batches,
not an exhaustive memory-limit search. H100's batch 128 matches the official
LIBERO per-GPU batch; increasing it further needs a new full-data probe.
The 4090 native-allocator batch-128 attempt OOMed. The optional allocator retry
was not run because A had reserved the GPU for evaluation; use a fresh output
and re-probe before selecting a larger batch. Accumulation is not assumed to give a larger batch's throughput.
Small-data and frozen-DiT reports remain explicitly preliminary/fallback evidence.

The reserved new-embodiment head must learn Sawyer state/action projections.
`tune_projector` controls state_encoder, action_encoder, action_decoder and
position embeddings. The upstream fallback flag does **not** freeze every other part of
the action head: `vlln` and `vl_self_attention` remain trainable even when
DiT is frozen. See [flow_matching_action_head.py:217–237](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/gr00t/model/action_head/flow_matching_action_head.py#L217).
Measurements list every trainable group. This is not LoRA. The default tuned
parameter count is 1,068,806,144 on both devices; the fallback has 518,419,456.

Hardware and recipe deviations: single GPU instead of LIBERO's eight; smaller
global batch and a bounded batch search; 60k steps instead of SmolVLA's 100k; GR00T optimizer,
warmup and LR endpoint instead of SmolVLA's; horizon 16 instead of SmolVLA's 50;
224² input instead of SmolVLA's 512². DiT freezing would reduce adaptation
capacity and must be reported as a fallback deviation if needed.

## Data and inference semantics

Conversion is CPU-only and leaves the source untouched. The source v3 parquet
shards embed RGB images. We write one parquet and one MP4 per episode in the
v2-compatible layout expected by N1.5. The video codec is lossless RGB H.264
(`libx264rgb`, crf=0), verified pixel-for-pixel against the source's embedded PNGs.
No episode frames are dropped, retimed or reordered. FPS is 80. Source
`observation.environment_state` is retained in parquet but never supplied to
the policy; there is no privileged object state.

`meta/modality.json` splits raw `observation.state` and `action` into
x/y/z/gripper scalars. State is absolute EEF xyz plus raw gripper; action is
Cartesian control xyz plus raw gripper. No EEF conversion, scaling, clipping,
gripper inversion or binarization. Values outside [-1,1] are legitimate source
commands. Marking action `absolute=false` makes N1.5 pad beyond the episode end
with physical zeros, without using frames from another episode. Normalization
uses exact full-dataset min/max and computed population mean/std and q01/q99;
stats are recomputed from every raw state/action row, not averaged per episode.

`metaworld_sawyer` is an actual added Enum value registered in-process, mapped
to the release's reserved non-pretrained `new_embodiment` projector slot 31.
We do not expand or reuse a pretrained robot head, or edit the vendor checkout.
Checkpoints store the named tag in `experiment_cfg/metadata.json`. The loader
requires that metadata, ensuring inference uses training normalization.

`load_policy(checkpoint, device, **kw)` returns fresh 16x4 float32 physical
commands on every call. It uses the base checkpoint's four denoising steps
(also the [release README's inference recommendation](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/README.md#L353)).
`reset()` is a no-op because there is no history/action queue. A supplies the
already-oriented 480² RGB render and seeds the global Torch generator; B does
not rotate it again or preserve any noise between calls.

The SmolVLA paper replans every action in simulation (§4.3). A's requested
historical MetaWorld protocol executes five actions per decision, with
`--replan-steps 1` available as a separately labeled sensitivity evaluation.
Training convention and action units are unchanged by that evaluation choice.

The source's push-back text is `Push the puck back toward the robot to the goal`;
A intentionally uses RLinf's `Push the puck to a goal` for `push-back-v3`.
Every other task text matches by content. B preserves source annotation IDs/text
and passes the harness prompt unchanged; this one-task prompt shift should be
disclosed in evaluation. Numeric source task IDs are not harness table indices.

## Checkpoints, probes and disk budget

`train.py` calls the release's `main(ArgsConfig)` directly. It retains the
official loss, model construction, optimizer and checkpoint-format callback.
Changes are a measurement callback, the added embodiment, TensorBoard logs,
resume/config checks, and no duplicate parent-directory final weight export.
Full checkpoints contain model weights, optimizer, scheduler, RNG and trainer
state plus GR00T's copied normalization metadata. Auto-resume chooses the last
complete checkpoint and rejects changes to the saved settings or normalization.
Incomplete checkpoint directories are renamed with an `incomplete-` prefix and
retained, so a crash during a write cannot block resuming a previous snapshot.

No automatic checkpoint pruning is enabled: the shared rule requires approval
before deletion. The 30k interval keeps two resumable snapshots in a 60k run.
Before training, parameter byte counts estimate remaining checkpoint storage;
on H100 the launcher checks the shared 120 GB tree budget and a 30 GB filesystem
free floor; the tree quota is rechecked before each snapshot write. The
coordinator must reserve space across simultaneous model jobs. Final setup
measured 85.31 GB already in the shared tree; two snapshots plus margin project
about 119.6 GB, so further growth can prevent starting or saving the run.
This long interval is a storage-driven deviation: six full 10k snapshots would
consume about 97 GB for GR00T alone, leaving insufficient room for shared data
and C's model. If a shorter interval is desired, increase storage or arrange an explicitly
approved archival/retention policy; the default does not silently delete files.

`--probe 200` executes 200 **optimizer updates** with the real model, transforms,
dataset and optimizer. The underlying LR schedule remains the selected 60k schedule;
the callback stops early and suppresses checkpoints. Metrics exclude the first
10 updates, include dataloader wait in elapsed update intervals, report both
optimizer/microbatch it/s and examples/s, and CUDA peak allocated/reserved GiB.
The 32-episode probe dataset validates mechanics/compute but is a single-task
subset and cannot be used for long training. Formal timing should be repeated
on the complete MT50 data because mixed task prompts and disk access can alter
throughput. All GPU runs hold `/tmp/r17_gpu.lock`; smoke/probe launchers enforce
an 880-second execution timeout. Long training is coordinator-only.
