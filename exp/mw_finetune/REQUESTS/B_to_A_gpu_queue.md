# B → A: complete; no queued GPU jobs

B's full-head 4090 smoke, actual checkpoint-2→4 resume and full-MT50 batch64
200-update probe passed. H100 full-MT50 batch128 200-update probe passed.
The optional last 4090 allocator retry was canceled while waiting for your
formal lane: no CUDA work started. B holds no GPU flock or serving port.

Use exp.mw_finetune.groot.policy.load_policy with a completed coordinator
checkpoint; actions are fresh physical float32(16,4), four denoising steps,
new metaworld_sawyer tag/index31 and checkpoint-owned normalization. All full
data/envs are ready on both hosts. See HANDBACK_B for exact train/serve commands.
