# Fast-WAM → MetaWorld MT50, agent C

Two transfer modes share one launcher: `--mode video_lora` keeps video co-training with in-place video LoRA; `--mode frozen_video` trains the action expert from the current frame only. Both start from LIBERO and reset the 4-D action/proprio layers. Per the coordinator’s `REQUESTS/COORD_to_C.md`, the target is ten epochs over MT50 (16,001 optimizer updates), global batch 128, and the official repository schedule. Choose video LoRA if its measured ten-epoch projection is ≤24 hours, otherwise frozen video. Neither is a full-co-training paper reproduction. The coordinator owns long training and formal evaluation.

## Source recipes and discrepancies

The Fast-WAM paper [§4.1–4.2](https://arxiv.org/html/2603.16666v1#S4) specifies Wan2.2-5B, a 1024-wide ~1B action expert, horizon 32, nine video frames (4× temporal subsampling), AdamW at 1e-4 with weight decay .01, cosine scheduling, mixed precision, clipping 1, ten action denoising steps and CFG 1. LIBERO training lasts 20k steps. Its §3.2 objective combines video and action flow-matching regression. The released code resolves details omitted by the paper:

| Setting | Pinned official LIBERO recipe |
|---|---|
| Global batch | Per-GPU 16 × 8 GPUs × accumulation 1 = **128**. README's eight-process command plus task YAML; derived rather than explicitly stated in the paper. |
| Optimizer | AdamW β=(.9,.95), weight decay .01, peak LR 1e-4. |
| Schedule | Warmup 5% of total updates, cosine to 1% of peak = 1e-6. |
| Duration | Paper 20k. Current task YAML instead sets ten epochs and `max_steps: null`; an explicit `max_steps=20000` is needed for the paper duration. |
| Images | Two 224² cameras concatenated to 224×448; 33 raw frames → nine sampled frames. |
| Chunk / inference | 32 actions; ten denoising steps, CFG 1; no future imagination at serving. |
| Loss | Weighted video and action velocity MSE; λ_video=λ_action=1; initial conditioned video latent excluded, episode padding masked. |
| Normalization | Per-channel min/max to [-1,1]; current code additionally caps normalized inputs at ±5. |
| Distributed training | Eight GPUs, DeepSpeed ZeRO-1, no optimizer/parameter offload. |

Reproducible code references (all official repository links use commit `7faa71108368fbb3b6885649f112af607427a2d4`): [task YAML:9–33](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/task/libero_uncond_2cam224_1e-4.yaml#L9), [dataset YAML](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/data/libero_2cam.yaml), [trainer:89–106](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/trainer.py#L89), [schedule:243–275](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/trainer.py#L243), [loss:522–609](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py#L522), [ZeRO-1 config](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/scripts/ds_configs/ds_zero1_config.json). Copies of small config files are in `sources/official/`.

**Scheduler version difference:** current official `configs/model/fastwam.yaml` uses action shift 1. The original LIBERO release uses action train/inference shift 5, as does the LeRobot port's config. The official README explicitly tells original-checkpoint users to use inference shift 5. This plan preserves **5 for both** from the checkpoint, alongside video shift 5 and 1,000 nominal timesteps; it does not silently inherit the newer shift 1.

The [SmolVLA paper §4.2–4.3](https://arxiv.org/html/2506.01844v1#S4) provides the MT50 reference: 100k fine-tuning updates, batch 64, AdamW (.9,.95), LR 1e-4, cosine minimum 2.5e-6 and 100 warmup updates. It uses 512² images, 50-action chunks, ten flow steps, and a frozen VLM; simulation replans every action. MetaWorld evaluation uses ten trials for each of 50 tasks and reports easy/medium/hard/very-hard groups and average. A's historical comparison protocol uses five executed actions per chunk; report that deviation explicitly.

## Single-GPU modes and the coordinator’s choice rule

Full ~6B training with conventional FP32 parameters, gradients and Adam moments is about **96 GB** before activations or the frozen VAE/T5. ZeRO-1 cannot shard that optimizer on one GPU. Our H100 has 80 GB. This arithmetic is a conventional 16 bytes/parameter estimate, not a measured allocation of the upstream trainer.

[LeRobot's pinned config:176–187](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/configuration_fastwam.py#L176) supports `freeze_video_expert` and `lambda_video=0`. Video blocks are reparented into MoT layers. We audit that video base parameters have no gradients and action/proprio gradients are present. The frozen mode uses a dedicated first-frame loss: the original causal mask gives action tokens access only to the current video frame, and that frame cannot see future frames or actions. Thus removing future tokens preserves action predictions and proprio gradients. A tiny CPU equivalence test compares both loss and proprio gradients to the original full-video path. No future VAE encoding, video head, video noise or video loss is evaluated in frozen mode.

The [EasyWAM LoRA recipe](https://github.com/OpenMOSS/EasyWAM/blob/main/configs/task/libero_easywam_mot_wan22_lora.yaml) keeps λ_video=λ_action=1 and uses **rank 128, alpha 128, dropout 0, bias=none**, Kaiming A and zero B. [Its Wan2.2 backbone config](https://github.com/OpenMOSS/EasyWAM/blob/main/configs/model/backbone/wan22.yaml) targets `self_attn.{q,k,v,o}`, `cross_attn.{q,k,v,o}`, `ffn.0`, `ffn.2`. We apply exactly those ten linear targets in all 30 video blocks (300 modules) in place, preserving Fast-WAM’s original action projections and proprio context layout. We borrow adapter settings rather than migrating EasyWAM’s different projector. Video LoRA parameters plus the complete action expert and new 4-D projections train; video base weights, VAE and T5 stay frozen. Full upstream nine-frame video/action losses remain unchanged.

The H100 probes started at micro-batch 16, then tested larger divisors of128 including actual Adam state allocation. **Select frozen_video, micro-batch 128, accumulation 1.** Video LoRA fits64 in a fresh process with expandable CUDA allocator segments, but its 37.5 h estimate exceeds the coordinator’s24 h cutoff. Batch 32 has more memory headroom and was slightly faster in its short sweep. Dropping video co-training is a material deviation and the paper’s ablation reports lower success rates. These probes compare speed and memory, not accuracy.

Measured on H10080 GB with FP32 trainables/Adam and BF16 activations (GB below means decimal10⁹ bytes):

| Mode / microbatch × accumulation | Timed microsteps | Microsteps/s | Seconds / optimizer update | Peak allocated / reserved GB | Ten-epoch compute estimate |
|---|---:|---:|---:|---:|---:|
| frozen_video16×8 | 200, excluding first 10 | 1.3394 | 5.9728 | 34.49 /35.63 | 26.55 h |
| video_lora16×8 | 200, excluding first 10 | .7691 | 10.4015 | 47.18 /47.92 | 46.23 h |
| frozen_video128×1 | 3 updates, excluding first | .2848 | 3.5111 | 61.96 /62.92 | 15.61 h |
| **frozen_video128×1, full MT50** | **32 updates, excluding first 4** | **.2857** | **3.5000** | **61.97 /62.91** | **15.56 h** |
| video_lora32×4 | 3 updates, excluding first | .4867 | 8.2186 | 58.39 /63.51 | 36.53 h |
| video_lora64×2, fresh process, expandable allocator | 3 updates, excluding first | .2372 | 8.4313 | 80.34 /83.92 | 37.47 h |

LoRA 128 OOMed (83.66 GB allocated peak);64 OOMed after that attempt with the default allocator, then fit in a fresh process with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. That recovery distinguishes allocator fragmentation from a fundamental 64 limit. The largest-batch estimates have only two steady optimizer updates and use a1,331-frame smoke subset; they exclude initialization, checkpoint writes and formal evaluation. The200-microstep probes each completed 25 optimizer updates and saved/reloaded real deltas. `HANDBACK_C.md` records the final full-data check and commands for a longer confirmation probe.

The selected mode was additionally checked on all 204,806 MT50 frames with 50 prompt embeddings:32/32 optimizer updates completed in131.79 s including19.49 s initialization, at36.57 samples/s. It loaded1,645 LIBERO tensors, reset exactly six embodiment tensors, and verified zero video-base gradients plus action/proprio gradients. Full H100 data audit matches the local SHA256 and checks every frame key and an RGB sample per task. This is an infrastructure smoke, not a success-rate evaluation or the final trained checkpoint.

**VAE cache check:** a separate synchronized LoRA 32 profile measured2.3193 s/microstep, with .2809 s in VAE encoding. Even a hypothetical zero-cost cache reduces its 41.23 h projection only to 36.24 h. That bound excludes cache construction/I/O and still misses24 h, so latent precomputation is omitted. The synchronized profile is slower than the ordinary sweep; it is reported separately rather than substituted into its throughput.

## Concrete single-H100 configuration

| Setting | This implementation |
|---|---|
| Initialization | `lerobot/fastwam_libero_uncond_2cam224`, revision `53983e1249b4eb4d89ab42b40b8678a882d331ad`; port of `yuanty/fastwam/libero_uncond_2cam224.pt`, not Wan-only or randomly initialized action expert. |
| Trainable | Full action expert (1,020,900,356 parameters including proprio), new action encoder 4→1024, head 1024→4, proprio 4→4096; reset weights **and biases** of those three layers and strictly copy all 1,645 other unique checkpoint tensors. Video mode also trains rank 128 LoRA. |
| Frozen | All video base parameters (4,999,787,712), raw Wan VAE and UMT5. Frozen video computation remains differentiable for proprio gradients. Video LoRA adapters are trainable only in video_lora mode. |
| Precision | Frozen components BF16. Action/proprio/LoRA parameters and Adam moments FP32, BF16 autocast for forward, clipping 1. No quantization or optimizer offload. |
| Batch | Selected frozen_video128×1 gives global 128. Alternative video_lora32×4 is the default for its memory headroom;64×2 is the largest measured fit with expandable allocator segments. Probe includes actual Adam state allocation. |
| Optimizer/schedule | AdamW β=(.9,.95), weight decay .01; linear warmup 5%=800 updates to 1e-4; cosine to 1e-6 at update 16,001. |
| Duration | ceil(204806×10/128)=16,001 optimizer updates. Deterministic epoch permutations sample each frame once/epoch; the final batch covers up to 68 extra windows. Resume state and replaceable serving delta every 2k; final immutable serving snapshot. |
| Activations | Checkpoint both DiTs and mixed attention. Frozen T5/VAE stay in eval and outside optimizer/state snapshots. |
| Data | All 2,500 episodes / 204,806 frames; deterministic shuffled frame/window sampling; fps 80 unchanged. One 480² dataset camera resized to **224²**, no crop/flip/rotation; lossless PNG storage. Four-D state/action unchanged before per-channel MT50 min/max normalization. |
| Targets | Actions at offsets 0..31; images at 0,4,...,32. Duplicate last frame at episode boundary and mask padded action/video targets. No cross-episode windows. |
| Loss | video_lora: unchanged upstream joint loss, λ_video=λ_action=1, nine frames. frozen_video: original action velocity loss, λ_video=0, one current frame only. |
| Serving | Current RGB frame + 4-D state + task prompt, 32×4 denormalized commands, ten denoising steps, CFG1, per-request CPU RNG seed. No gripper sign/binary toggle or queue. |
| Checkpoints | FP32 action/proprio delta plus MT50 stats and immutable base revision; latest optimizer/RNG state is atomically replaced. Existing raw Wan assets are read-only, shared outside checkpoints. |

The launcher resumes only at completed optimizer updates. It validates dataset index fingerprint, global batch, target duration, precision mode and seed. The deterministic sampler derives each epoch’s permutation from `(seed, epoch)` and restores the microstep offset; Torch CPU/CUDA noise RNG is checkpointed. Probe output separates microstep throughput from estimated optimizer-update throughput (divide by accumulation). The smoke's tiny dataset/checkpoint is **not** an MT50-trained evaluation artifact.

## Deviations to disclose in any result

1. Video base frozen in both modes; video LoRA approximates full co-training, or frozen_video removes it. Full action expert fine-tuned from LIBERO rather than Wan-only initialization, as requested. Hardware prevents unsharded full co-training.
2. One camera at 224², four state/action channels, reset embodiment layers, new MT50 stats. LIBERO used two cameras, 7-D action and 8-D proprio. SmolVLA used512²;224 preserves Fast-WAM's camera resolution.
3. One GPU with accumulation preserves the official global 128. Translate the current repository’s ten epochs to 16,001 MT50 updates; the paper used20k LIBERO updates and SmolVLA used100k MT50 updates. This duration/schedule is explicitly chosen by the coordinator.
4. Explicit FP32 action parameters and optimizer states, BF16 activations; activation checkpointing enabled. No DeepSpeed because only one GPU.
5. Cache the 50 fixed training instruction embeddings at setup, then move frozen T5 to CPU to save GPU memory. The checkpoint prompt template is applied both during caching and at serving. Serving encodes each request anew without cross-request cache.
6. Normalized values are not clipped; source training values lie within dataset min/max. This permits out-of-distribution state/action values to invert exactly at serving and respects A's raw-command contract.
7. Formal harness executes five actions before replanning versus SmolVLA's one; no reward feedback, online optimization, retrieval, or future planning.
8. The default keeps only the final immutable serving snapshot to reserve atomic-write headroom within the shared 120 GB disk budget. Resume/replaceable serving checkpoints still save every 2k. Additional immutable snapshots can be requested with`--milestone-every`, subject to the same disk gate.

Measured memory/throughput and exact setup/train/serve commands are in `HANDBACK_C.md` and `reports/`.
