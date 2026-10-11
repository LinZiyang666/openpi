# E → coordinator: DP tools ready (2026-10-09, after reboot)

Owner decisions implemented: MT50 fallback is **scratch DP-T**, never initialized
from ML45/LIBERO policy tensors. LIBERO fallback fine-tunes the strictly validated
30k tensors on the full four-suite source (scratch override also supported).

Handback: `exp/mw_finetune/dp/HANDBACK_E.md`; sources/deviations: `dp/RECIPE.md`.
All475 tensors strictly load for three original checkpoints. Native-source epsilon
MSE0.00937 (ML45 85k),0.01599 (30k),0.01393 (LIBERO); correct source cameras and
quantile normalization outperform alternatives. All wrappers are importable,
stateless, current-observation only, default100-step epsilon DDPM, horizon16.

Fair MT50 transfer smokes:85k/zero wrist1/16;30k/duplicate0/16;30k/zero0/16;
all complete, no rollout errors, A's timan108 EGL8-easy×2 protocol. Source uses
corner4+gripperPOV rather than A's one corner2 image. Prepare scratch MT50 run.
LIBERO fixed-seed0 multi-suite smoke0/4/no interface errors. Its language extrema
match nine LIBERO-10 texts, not the full40; this does not prove training coverage.
Source native denoising works, but no useful four-suite SR is established.

Local data complete and audited: MT50 204,806frames/2,500episodes/50tasks;
LIBERO273,465frames/1,693episodes/40tasks, paired96² PNGs. Source LIBERO34.94GB,
prepared7.35GB; all SQLite audits and12 spread source image/vector checks pass.
Training text caches are ready. HTTP resume bypassed the anonymous Xet429.

H100 scratch MT50 full-data200-update probe:6.90391it/s,4.80824/5.00391GiB
allocated/reserved,4.02349h compute projection. LIBERO one-episode200-update
backward probe:7.19309it/s,5.00586GiB reserved,3.86173h projection (partial-data
I/O caveat). No local GPU smoke while GR00T occupies39.6/48GB; re-probe after
~18:40CDT. CPU tests11/no skips; release scratch save/reload matches all475
weights, full Adam state and CPU RNG bitwise. Full-data LIBERO CPU update passes.

No E GPU process remains. H100 E root~5.87GB. Cleanup removed only E transfer
archives after SHA parity with local retained copies under owner's instruction.
Do not put the full LIBERO set or20 optimizer snapshots on H100 under10GB.
Long training targets local disk and holds the GPU lock. Coordinator commands:

```bash
# After GR00T finishes; both full datasets are already prepared.
bash exp/mw_finetune/dp/launch.sh --init scratch --probe 200 --batch-size 256 \
  --workers 4 --output /data/mw_finetune/dp/4090_scratch_probe
bash exp/mw_finetune/dp/libero/launch.sh --probe 200 --batch-size 256 \
  --workers 4 --output /data/mw_finetune/dp/4090_libero_probe

# Coordinator only,100k updates; rerun identical command to resume.
bash exp/mw_finetune/dp/launch.sh --init scratch --steps 100000 --batch-size 256 \
  --workers 4 --save-every 5000 --output /data/mw_finetune/dp/4090_mt50_scratch
bash exp/mw_finetune/dp/libero/launch.sh --steps 100000 --batch-size 256 \
  --workers 4 --save-every 5000 --output /data/mw_finetune/dp/4090_libero_finetune
```

Native A model/port registration is requested in `E_to_A_dp.md`: dp offset4,
23154/23254. Meanwhile `python -m exp.mw_finetune.dp.serve_compat --model dp`
works with A's real server. Full formal commands on both timan lanes, serving
exports, sync-back, aggregates and LIBERO4×10×50 SR commands are in HANDBACK_E.
Architect integration note: `E_to_integration_dp.md`. No cache-system code or
other agent's model code was edited; no commit/push or long GPU job was run.
