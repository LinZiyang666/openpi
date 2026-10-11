# R17 MetaWorld MT50 and LIBERO pure-inference results (coordinator)

Protocol: MetaWorld 3.0.0 MT50 v3, `MT1(task, seed=7)` + `train_tasks[idx]`, idx 0..9 per task (500 episodes),
15 settle steps, ≤160 policy steps, 5 actions executed per inference call, corner2 480² rot180 camera
(matches `lerobot/metaworld_mt50`), NVIDIA EGL rendering on timan workers. SR in %.

| Model | Checkpoint | Run | easy (28) | medium (11) | hard (6) | very hard (5) | equal-group mean | per-task mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| π0.5 RLinf SFT | `/data/ckpt/pi05_metaworld_rlinf` | `/data/mw_finetune/eval/pi05_rlinf_mt50_egl` (10-09 00:54–02:43) | 70.0 | 45.5 | 46.7 | 40.0 | 50.5 | 58.8 |
| GR00T N1.5-MW (60k steps, batch 64, diffusion head + projector tuned, VLM frozen, 4 flow steps) | `/data/mw_finetune/groot/4090_mt50/checkpoint-60000` | `/data/mw_finetune/eval/groot_mt50_60k_egl` (4090 + timan107 EGL, 10-09 18:41–19:07, 0 errors) | 91.1 | 58.2 | 71.7 | 52.0 | 68.2 | 77.6 |
| DP-T (from scratch on MT50, ε-DDPM 100 steps) | `/data/mw_finetune/dp/h100_mt50_scratch/checkpoint-100000` (export `mt50_policy_100000`; 100k updates, batch 256) | `/data/mw_finetune/eval/dp_mt50_scratch_100k_h100_merged` (4 disjoint task parts `_p1..p4`, H100 server + timan108 EGL, 10-09 15:10–16:11, 0 errors) | 79.6 | 40.9 | 35.0 | 34.0 | 47.4 | 61.2 |

Reference (published, group mean): DP 10.5, TinyVLA 31.6, SmolVLA 68.2, π0 SFT 50.8, π0.5 SFT 43.8, π0+PPO 78.1, π0.5+PPO 70.7.
Our π0.5 per-task mean matches the September run (58.8%).

## Mid-training sanity checks (not formal results)

| Model | Checkpoint | Tasks × trials | Successes | Notes |
|---|---|---|---:|---|
| Fast-WAM-MW (frozen video) | `mt50_frozen_b128/policy_sanity_004000` (update 4000 / 16001) | 8 easy tasks × 2 (button-press, door-close, drawer-close, faucet-open, handle-press, reach-wall, window-close, plate-slide-side; π0.5 = 100% on all) | 10/16 | H100 + timan108 EGL, 10-09 05:11–05:14; conventions OK. Run `/data/mw_finetune/eval/fastwam_sanity_004000` |
| GR00T N1.5-MW | `4090_mt50/checkpoint-30000` (step 30000 / 60000, batch 64) | same 8 easy tasks × 2 | 14/16 | served on H100 (shared with Fast-WAM training, no lock) + timan108 EGL, 10-09 10:34–10:35; 18.6 eps/min. Run `/data/mw_finetune/eval/groot_sanity_030000` |

## Plain step reduction on MT50 (pure inference, same protocol)

| Model | Denoising steps | Run | easy | medium | hard | very hard | equal-group mean | per-task mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| GR00T N1.5-MW | 4 (default) | `groot_mt50_60k_egl` | 91.1 | 58.2 | 71.7 | 52.0 | 68.2 | 77.6 |
| GR00T N1.5-MW | 2 | `groot_mt50_60k_k2_egl` (10-09 20:07–20:33, 0 errors) | 87.1 | 61.8 | 68.3 | 54.0 | 67.8 | 76.0 |
| GR00T N1.5-MW | 1 | `groot_mt50_60k_k1_egl` (10-09 20:34–20:58, 0 errors) | 89.3 | 51.8 | 65.0 | 46.0 | 63.0 | 73.8 |
| π0.5 RLinf | 10 (default) | `pi05_rlinf_mt50_egl` | 70.0 | 45.5 | 46.7 | 40.0 | 50.5 | 58.8 |
| π0.5 RLinf | 2 | `pi05_rlinf_mt50_k2_egl` (10-09 20:07–21:41, 0 errors) | 68.9 | 46.4 | 48.3 | 40.0 | 50.9 | 58.6 |
| π0.5 RLinf | 1 | `pi05_rlinf_mt50_k1_egl` (10-09 21:41–23:32, 0 errors) | 70.0 | 42.7 | 43.3 | 38.0 | 48.5 | 57.6 |

## LIBERO official test A pure-inference baselines — final, 2026-10-10

正式 A 集：40 tasks × 50 个官方原始初态（idx 0..49），每 suite 500、每模型 2000 局；cache 关闭，10 settle steps，spatial/object/goal/libero_10 控制步上限 220/280/300/520。SR 单位 %，mean 为四 suite 等权均值；这些是 R17 实测，非论文成绩。C=每次模型调用执行动作数，K=采样步数，B=每 replica 的最大 batch。

| Model / checkpoint | K / C / native chunk；部署 | spatial | object | goal | libero_10 | mean；成功数 | 验收与合并来源 |
|---|---|---:|---:|---:|---:|---:|---|
| DP-T；`/data/mw_finetune/dp/libero_policy_100000`（100k） | DDPM100 / C5 / 16；4090 **48GB**，2×B8/5ms + MPS，timan107×32，seed7 | 91.2 | 70.0 | 90.8 | 79.6 | 82.90；1658/2000 | [hs5dp_ctl_142812 SUMMARY](/data/mw_finetune/eval/hs5dp_ctl_142812/SUMMARY.json:1)；[恢复后 PASS 收据](/data/mw_finetune/eval/hs5dp_ctl_142812/RECEIPT.json:1) |
| Fast-WAM；`/data/mw_finetune/models/fastwam_libero` | shifted-flow K10 / C5 / 32；H100 80GB，2×B16/5ms + MPS，timan107×48，seed7 | 90.8 | 95.6 | 97.8 | 91.6 | 93.95；1879/2000 | [hs6fw_res_152931 SUMMARY](/data/mw_finetune/eval/hs6fw_res_152931/SUMMARY.json:1)；[PASS 收据](/data/mw_finetune/eval/hs6fw_res_152931/RECEIPT.json:1)；合并 `hs6fw_ctl_143332` 1839 + resume 161 |
| Cosmos Policy；`/data/mw_finetune/models/cosmos_libero` | K5（实际 NFE5）/ C16 / 16；H100 80GB，2×B1，policy seed195 | 96.4 | 99.6 | 98.4 | 98.6 | 98.25；1965/2000 | [合并 K5 SUMMARY](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/SUMMARY.json:1)；[RECOVERED_RECEIPT PASS](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/RECOVERED_RECEIPT.json:1)；原段 1726 + resume 274 |

DP/Fast 使用 group-commit driver，所有 horizon/step 臂分别配对其同 GPU、同 batch 配置控制；batch cap 相同不等于实际 batch 组成相同。配置证据：[DP campaign](/data/mw_finetune/eval/hs5dp_ctl_142812/campaign.json:1)、[Fast campaign](/data/mw_finetune/eval/hs6fw_res_152931/campaign.json:1)、[共同 A 集协议](/data/mw_finetune/eval/hs5dp_ctl_142812/assets/benchmark.json:1)。

Cosmos 合并控制的 1726 局在 timan108×24（RTX A5000 EGL），274 续局在 timan107×8（GTX 1080 EGL，均属 libero_10）；原始续段 RECEIPT=FAIL 由显式恢复收据和封存文件取代，未掩盖历史状态。[合并清单](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/RESUME_MERGE.json:1)、[封存证据](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/RECOVERED_RESULT_SEAL.json:1)、[Cosmos 配对与节点混杂分析](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/COSMOS_K2.md:1)。

owner 的 horizon/纯减步决策只使用 **libero_spatial + libero_10**；三模型、12 臂最终结论与 cache 的 L 见 [SUMMARY.md](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/SUMMARY.md:1)，全部配对区间和成本见 [DP/Fast 完整分析](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/RESULTS_DP_FASTWAM.md:1)。旧 400 局结果保留于下，仅作 superseded 历史记录，不与正式 A 集合并。

## LIBERO pure inference on a NON-STANDARD 400-episode set (40 tasks × original init 0..9) — superseded

**Superseded; do not compare with R16.** This set came from the R17 integration spec, not the R10/R16 protocol (official test A: 50 official inits per task, 500 per suite). The completed official-A baselines above replace these entries for reporting; do not pool the two sets.

| Model | Checkpoint | Run | spatial | object | goal | libero_10 | mean |
|---|---|---|---:|---:|---:|---:|---:|
| DP-T (own fine-tune, 100k steps, DDPM 100) | `/data/mw_finetune/dp/libero_policy_100000` | `/data/mw_finetune/eval/formal_dp_libero_formal400_20261009_220843_8dca4316` (10-09 22:08–22:52, 12 CUDA-graph replicas on 4090, 24 workers timan107) | 90.0 | 72.0 | 94.0 | 81.0 | 84.3 |
| Cosmos Policy (official LIBERO weights, K5, seed195) | `/data/mw_finetune/models/cosmos_libero` | `formal_cosmos_libero_formal400_20261009_231322_2b680ed5` (H100 2 replicas + timan108 ×24, 23:13–23:40, PASS, 0 errors) | 97.0 | 100.0 | 98.0 | 98.0 | 98.3 |

Historical 400-episode note (superseded): 400/400 completed. 2 libero_10 attempts hit a transient EGL "offscreen framebuffer not complete" at step 0 and were retried successfully. The launcher marked the receipt FAIL/PARTIAL on driver rc=2; `SUMMARY.json` was produced by `formal_eval aggregate`.

## MetaWorld MT50 formal pure inference via R17 framework (conductor per-episode + resident replicas)

| Model | Checkpoint | Run | easy | medium | hard | very hard | group mean | per-task mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Fast-WAM-MW (frozen video, 16001 steps) | `/data/mw_finetune/fastwam/mt50_frozen_b128/policy_016001` | `formal_fastwam_mw_mt50_500_20261009_223917_0b845f00` (H100 2 replicas + timan108 ×24, 22:39–23:39, PASS, 0 errors) | 81.4 | 34.5 | 21.7 | 46.0 | 45.9 | 60.4 |
| DP-T-MW (100k, DDPM 100, CUDA graph) | `/data/mw_finetune/dp/mt50_policy_100000` | `formal_dp_mw_mt50_500_20261009_225821_ef25fe4e` (4090 12 replicas + timan107 ×24, 22:58–23:36, PASS, 0 errors) | 79.3 | 40.9 | 35.0 | 34.0 | 47.3 | 61.0 |

The DP-T framework run reproduces the earlier stock-driver DP-T run (79.6/40.9/35.0/34.0; 47.4/61.2) within sampling noise.

## LIBERO teacher libraries (B pool, 500 inits per suite, aligned with π0.5/GR00T R10 subsets)

Built 10-09/10 via `exp/r17_integration/ops/library_campaign.sh` (R17 conductor + resident replicas, debug capture with allowlist: keys, normalized chunk, state, ids, two uint8 cameras). Full native teacher (DP100 / Fast10 / Cosmos5), failed episodes kept. Parent at `/data/r17_library/<bundle>_<suite>/bpool/`; nested subsets 50/100/200/300/400/500 are R10 row-index views (seed 20261002, same episode ids as π0.5/GR00T); fits per size (Mahalanobis metric, LOEO corrector, distance calibration, caps) under `bpool/fits/size_N/`.

| Library | rows @500 | rows @50 | parent size | notes |
|---|---:|---:|---:|---|
| Cosmos × spatial | 3,676 | 377 | 1.5 GB | 500 eps, 35 failed |
| Cosmos × libero_10 | 8,408 | 847 | 3.3 GB | node-failure salvage 461 + repair 39 |
| DP × spatial | 12,028 | 1,255 | 4.6 GB | repair 4 (step-0 EGL errors) |
| DP × libero_10 | 31,546 | 3,088 | 12 GB | salvage 497 + repair 3 (hung workers) |
| Fast-WAM × spatial | 12,460 | 1,266 | 7.1 GB | salvage 229 + repair 271 |
| Fast-WAM × libero_10 | 27,806 | 2,657 | 16 GB | salvage 54 + repair 446 |
