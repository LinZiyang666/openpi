<update>
Good handback. Next: a cheap **reconstruction-disambiguation probe** so I can decide whether a 500-episode Cosmos-MW run is meaningful.
Target: assembly-v3 (author README ≈52% base over 25 episodes), idx 0..9, bench seed 7; plus reach-v3 idx 0..4 as an easy sanity task.
Configs (each its own immutable result dir; keep A's evaluator and server path):
1. current default (single camera layout, corner2 480² rot180, chunk 16);
2. `libero_duplicate` layout;
3. FlowDAgger viewpoint: corner3 raw 256 as in `metaworld_pi05_adapter.py:32,124` / `task_configs.py:20` — implement as a probe-only render override inside `exp/mw_finetune/cosmos/` (reuse `exp/metaworld` without editing it; if the evaluator cannot take a render override, write a minimal probe runner in your directory that reuses `exp.metaworld.env` and the same server/wrapper) — and also with the author's max_timesteps=300 for assembly;
4. any other single assumption you judge most likely wrong after re-reading the FlowDAgger sources and the checkpoint README/statistics (e.g. chunk length, conditioning-frame count, action column map, rotation).
Also check whether the released statistics/prompt cache indicate the training data was `lerobot/metaworld_mt50` (compare its action/state min/max with the dataset's `meta/stats.json` under `/data/mw_finetune/datasets/metaworld_mt50`) — if they match, the camera is the dataset's convention.
GPU: the 4090 under `flock /tmp/r17_gpu.lock`, total ≤ 20 min, use 4–6 workers per config; B may be probing GR00T — wait for the lock. Report a small table (config → successes/episodes, ms/query) in HANDBACK_D.md and your recommendation for the formal configuration.
</update>
