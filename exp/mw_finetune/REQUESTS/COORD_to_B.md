# Coordinator → B: GR00T recipe correction (2026-10-09 00:1x)

Our local RTX 4090 is a **48 GB** card (not the 24 GB retail card the README's `--no-tune_diffusion_model` note targets).
Default to NVIDIA's standard fine-tune (tune_projector + tune_diffusion_model, VLM/LLM and visual frozen) on BOTH GPUs.
Keep `--no-tune_diffusion_model` only as an OOM fallback flag, documented as a deviation.
Probe on the 4090: max per-device batch that fits with diffusion tuning on (try 64, then 32 + accumulation),
report s/step and peak memory; same for the H100 if you have time. Step budget: cite NVIDIA's LIBERO recipe
(20k for Spatial/Goal/Object, 60k for 90/Long); MT50 is closest to LIBERO-90 in task count → propose 60k unless
you find a better-justified source, and give the projected wall-clock on each GPU.
