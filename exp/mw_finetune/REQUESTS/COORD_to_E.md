# Coordinator → E: owner decision on the MetaWorld DDPM model (2026-10-09 09:3x)

Owner: include a MetaWorld DDPM policy. **First try brandonyang's MetaWorld weights; if they do not work, train our own DP on MT50 from scratch.**

1. Primary: `brandonyang/dp-metaworld-checkpoints` (prefer 85000, also check 30000). Get its conventions right (camera/view, image size/crop, state & action layout and normalization from `assets/brandonyang/metaworld_ml45/norm_stats.json`, language embedding, chunk/replan). Run the 8-easy-task × 2 sanity through A's harness; if it is clearly usable, prepare the formal 500-episode MT50 command (note: ML45 training set → 5 MT50 tasks unseen; report them separately as well as in the total).
2. Fallback (only if 1 fails after a fair attempt): train DP **from scratch** on `lerobot/metaworld_mt50` with the same architecture/config as brandonyang's checkpoint (DP-Transformer, language FiLM, n_obs_steps=1, ε-prediction DDPM 100 steps), recipe from its metadata/config (state any gaps and fill them from the original Diffusion Policy paper / SmolVLA MT50 reference). Launcher with `--probe`, resumable checkpoints.
3. Keep the DDPM head and full 100-step inference as the default everywhere (our step-reduction experiments use it later).
4. LIBERO wrapper for `brandonyang/dp-libero-checkpoints` remains in scope (same model family on LIBERO).

## Addendum 09:3x — LIBERO fallback (owner)
Owner: "liebro的如果不行我们也自己微调一份" — same rule on LIBERO. First validate `brandonyang/dp-libero-checkpoints` (30000) on our LIBERO harness (a short multi-suite smoke, then the command for a full SR run). If it does not work after a fair attempt, train our own DP-T on LIBERO (the 4 suites mixed, as in his `dp_libero_lang_v1` config; data from the openpi LIBERO dataset `physical-intelligence/libero` — check whether a local copy exists under /data before downloading), same architecture, DDPM 100 steps, launcher with `--probe` and resumable checkpoints. Put LIBERO tooling under `exp/mw_finetune/dp/libero/` and document both in HANDBACK_E.md.
