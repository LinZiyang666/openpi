# Coordinator → C: Fast-WAM recipe decision (2026-10-09 00:1x)

Answer to A's pending recipe question. Build BOTH modes behind one launcher flag; I pick after your H100 probe.

1. `--mode video_lora` (PREFERRED, closest to the paper): keep video co-training exactly as the official recipe
   (`loss.lambda_video` = official value, future frames encoded as upstream), LoRA on the video expert's linear
   layers (EasyWAM-style; state rank/alpha/targets and cite the source you follow), action expert + new 4-D
   action/proprio projections fully trained. Precompute/cache VAE latents of the 224 frames if it saves time.
2. `--mode frozen_video` (FALLBACK): video expert frozen, lambda_video=0, and make the training forward use only
   the current frame (no future-frame encoding / no video loss work) so it costs about what inference costs.

Recipe base = official `configs/task/libero_uncond_2cam224_1e-4.yaml` + `configs/train.yaml` translated to MT50:
global batch 128 (16/GPU × 8 GPUs → use gradient accumulation on one GPU), AdamW lr 1e-4 cosine, weight_decay 1e-2,
10 epochs over the 204,806 MT50 frames (≈16k optimizer steps), bf16, max_grad_norm 1.0, save every 2000 steps,
init from the released LIBERO checkpoint (deviation: paper trains from Wan2.2 base; state it).
Write the per-step memory and it/s for both modes in RECIPES/HANDBACK_C, and the projected wall-clock for 10 epochs.
My rule: use video_lora if its 10-epoch projection on the H100 is ≤ 24 h, else frozen_video.

Addendum 00:1x: micro-batch 1 is far below the official 16/GPU; probe the largest micro-batch that fits on the H100
(try 16, 8, 4) for each mode and accumulate to global 128. Report s/optimizer-step.
