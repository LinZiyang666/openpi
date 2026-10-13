# Cosmos Policy → our MetaWorld MT50 protocol

Initialize the complete DiT from NVIDIA's local LIBERO policy:
`/data/mw_finetune/models/cosmos_libero/Cosmos-Policy-LIBERO-Predict2-2B.pt`.
Its pinned release is `cb689ec0e3347c13667d70a78a3447388f5c3bb8` and its
documented robot policy already predicts 16×7 actions. Transfer this action
prior rather than learn action latents again from the Predict2 video base.
The community MT50 release has no runner/config and failed our reconstruction
probes and the coordinator's seed-7 run (0/152); this provides no evidence for
using it as a reliable initialization. It remains only the source of cached
T5 embeddings for the exact MT50 prompts. This initialization choice is an
inference about transfer efficiency, not an established MT50 success result.

Evidence: [official LIBERO recipe at our vendor commit](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/LIBERO.md),
[NVIDIA model card](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B),
`exp/mw_finetune/cosmos/HANDBACK_D.md`, and
`exp/r17_integration/handbacks/HANDBACK_MW_D.md`.
Vendor commit: `18a2accadf4e7a3531e56754102af5a24d2316da`.

| Setting | MT50 recipe |
|---|---|
| Trainable | All ~1.956B DiT parameters; frozen Wan VAE, precomputed frozen T5; EMA off. No LoRA, quantization or optimizer offload. |
| Loss | Native `CosmosPolicyVideo2WorldModel.training_step`; official hybrid log-normal/uniform EDM noise, sigma .01…200, p_mean ln(4), p_std 1.2, uniform 1…85. Explicit per-sample loss masks: 50% successful-demo action prediction, 25% future-state prediction, 25% value prediction. |
| Data | All 204,806 frames, 2,500 episodes from the same local LeRobot MT50 dataset used by other models; deterministic shuffled frame windows, no temporal subsampling. Numeric statistics recomputed from all original float32 rows in the existing Fast-WAM index. |
| Images | Dataset already has corner2 RGB480 rotated 180°. Reuse Fast-WAM's lossless **bilinear RGB224 PNG** cache on both hosts; no second rotation, no extra JPEG. Native strong augmentation: random 90%-area square crop, ±5° rotation, color jitter, shared current/future transforms. At serving: bilinear224 then native deterministic 90%-area center crop. |
| Language | `exp/metaworld/tasks.py` verbatim, no wrapper template. Resolve LeRobot prompt indices via their stored text, since their order differs from the R17 task table. Push-back uses the shared R17 push wording. Copy the 49 unique exact prompt embeddings into every serving export. |
| State | Wire remains obs[:4]=xyz,g. Native padding `[g,-g,x,y,z,0,0,0,1]` (9-D); no hidden environment state. |
| Actions | Wire/native units remain xyz,g. Pad `[x,y,z,0,0,0,g]` (7-D); normalize with OUR min/max; infer inverse then select columns 0,1,2,6. No sign change, extra action scale or clipping in the adapter. Environment clips as usual. |
| Constant channels | Bounds mean±0.5; fixed orientation/action-rotation channels normalize to zero. Other bounds are empirical min/max; no quantiles or clipping. |
| Chunk / future | 16 consecutive actions, repeat the final action at episode boundaries as upstream; future state/image at min(t+16,last). No window crosses an episode. |
| Native sequence | 25 raw frames → 7 latents: blank, current proprio, current image, action, future proprio, future image, value. Four repeated raw frames per nonblank latent. Three conditional latents. No wrist/second camera. |
| Value | Native sparse-terminal Monte Carlo return, gamma .99, mapped to [-1,1]. Terminal success is read from any `next.success` in the episode; no fabricated failures. Value/future predictions are never used to steer R17 actions. |
| Optimizer | Native FusedAdamW: BF16 parameters/gradients, FP32 master weights and moments, betas (.9,.99), weight decay .1, eps 1e-8, native LR scheduler. |
| LR / schedule | Peak **3e-5**, reduced from official 1e-4 because we transfer a trained policy with effective batch32 rather than1920. Official 1000-update warmup from multiplier1e-6; linear decay to .3 at30k; then .06 constant. This reduced LR is a transfer choice, not a published MT50 setting. |
| Duration | 40,000 optimizer updates, effective batch32 ≈6.25 frame passes. First candidates at5k/10k, then20k/40k. Save resumable state every1k and serving weights every5k, plus at a bounded stop/probe completion. |
| Single H10080GB | Planned microbatch4 × accumulation8. |
| Single local 48GB4090 | Planned microbatch1 × accumulation32. |
| Activations | Native whole-block recomputation for every DiT block; no torch.compile. |
| RNG | Training explicitly disables the inference VAE's `DETERMINISTIC` reseeding flag. Worker augmentation uses isolated CPU RNG; loader worker seeds do not consume model noise RNG. Save/restore CPU/CUDA/NumPy/Python RNG and FP32 optimizer master parameters. |
| Bound/resume | `--max-hours` includes model initialization. Stop checks before every microstep; finish at most the running microstep, discard incomplete accumulated gradients, and save the last completed update. Final checkpoint I/O is additional shutdown time. Reinvoke the same command to resume automatically. SIGTERM/SIGINT request the same clean stop once the loop is running. |
| Serving | Same corner2/480/rot180, seed7, settle15, max160 policy steps, success at any step; native K5 sigma4…80; execute first5 actions then replan. |

The official reference used 40k updates at global batch1920 on64 H100s in
~48h, with a recommended normalized action L1 near .010. Our much smaller
batch and policy initialization make this a transfer recipe, not a reproduction
of that run. The native masks make the intended action/world/value objectives
explicit. The source success audit found2,465 successful episodes and35 failures;
policy samples use successes, while world/value samples use all episodes.
The LIBERO config leaves per-objective loss masking disabled by default. We
enable its existing masking option to focus demonstration supervision on
actions and keep future/value supervision on their respective sample types.
This is an explicit deviation chosen for the pure-inference objective.

The planned batches are conservative estimates, **not measured fits**. The
coordinator must run `--probe 20` after collection releases a GPU. Optimizer
updates allocate real FP32 moments/master weights; the probe includes those
allocations and forward/backward, accumulation, data loading, and synchronization.
Fixed DiT parameter/gradient/master/moment memory is about29.2GiB before the
frozen VAE and activations. If needed, reduce microbatch and increase accumulation
to retain effective batch32. Microbatch changes preserve the sample order and
saved optimizer state; they do not promise numerically identical GPU reductions.

Budget inference from the official throughput (4.32s/update per H100 at local
batch30), adjusted for small microbatches, unsharded state and recomputation:
first5k candidate **10–25h H100**, **25–60h 4090**; full40k **80–200h H100**,
**200–480h 4090**, plus startup/checkpoint/evaluation time. These ranges are
unmeasured. After the probe use `median_step_seconds × remaining_updates / 3600`.
First5k is a candidate for evaluation, not a guaranteed usable policy.

Coordinator acceptance target: at least50% mean MT50 success under our protocol,
then improve toward75%. Evaluate the first candidate across all50 tasks before
committing to the entire budget; report per-task outcomes. Training L1 alone
does not establish closed-loop success. No GPU, training or campaign was started
while preparing this package.

Checkpoint storage: one atomically replaced `training_state.pt` (~27GB including
weights, moments, master parameters and RNG), temporary replacement of similar
size, and ~4GB per serving milestone (8 milestones for40k). Allow ~90GB peak
additional space, plus ~4GB per extra bounded-stop export and ~31GB for each
independent probe output. A serving directory contains `consolidated.pt`, its own
`dataset_statistics.json`, `t5_embeddings.pkl`, `config.json`, and this recipe.
Supply that exact directory to R17 `--model cosmos_mw --checkpoint ...`.
The adapter keeps the old community defaults only when no fine-tune config is
present; it reads the new preprocessing and statistics from our export.
