<task>
Repo /home/weiland/projects/openpi (R17 framework under exp/r17_integration). Add Cosmos Policy on MetaWorld MT50
(`cosmos_mw`) to R17 so the coordinator can measure its pure-inference success rate and later collect a cache library.
Candidate weights: Hugging Face `mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B` (community, v3; the author's
runner is not public). Steps:
1. Download to `/data/mw_finetune/models/cosmos_mw` (public download only; if the repo is gated or needs a token,
   stop and report - do not use or look for credentials). Record size, files, config, any model card text.
2. Work out exactly what it expects: cameras/resolution/crop, conditioning frames, proprio/state, action dimension,
   chunk length, normalization stats, prompt template, denoising steps, and what MetaWorld setup it was trained on
   (camera corner2? MT50 task set? action space). Use the Cosmos Policy code we already have
   (`/home/weiland/projects/openpi_ext/lines/cosmos-policy`; H100 copy `/data/cosmos/cosmos-policy`) and the existing
   R17 Cosmos LIBERO integration (`exp/r17_integration/models/cosmos/`, `handbacks/HANDBACK_COSMOS.md`).
3. Implement `cosmos_mw` following `cosmos_libero`: manifest/provider/adapter for the MT50 wire format used by every
   other MetaWorld model (corner2 480x480 rotated 180 deg, state = obs[:4], prompts from `exp/metaworld/tasks.py`),
   formal selector + default checkpoint in `ops/formal_eval.py` (`DEFAULT_CHECKPOINTS` and model selection only),
   feature spec for retrieval keys (one camera) and debug_capture capability so the library teacher path can capture
   it. Execution: the MT50 kernel executes at most 5 actions per call today (`exp/metaworld/env.py`,
   `benchmarks/metaworld.py`); if the checkpoint's native chunk is longer, report the trained execution horizon and
   implement the native one if small, otherwise execute 5.
4. CPU tests of the adapter (fake model), and a GPU-free shape check of preprocessing on real MT50 frames rendered
   with the MetaWorld env (`/home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python`, MUJOCO_GL=egl may use
   the local GPU only for rendering a handful of frames if `nvidia-smi` shows it idle).
</task>
<context>
Owner rule: keep it simple - smallest diff, no new seals, fingerprints, admission gates or validation layers.
Other agents edit concurrently: package A (`benchmarks/*`, `exp/metaworld/episode_runner.py`, `conductor/manifest.py`,
`debug/*`, `search/teacher_library.py`, `ops/library_*.py`, `ops/timan1_workers.py`, `ops/formal_eval.py`
prepare_workers/library plumbing) and package B (`models/dp|fastwam/formal_batch_server.py`, `ops/cache_study.py`,
`ops/formal_eval.py` execution_horizon/execution_steps/max-batch/mps/seal batching). Stay in `models/cosmos/` plus the
two small `formal_eval.py` selector spots; never revert others' changes; do not git commit.
</context>
<action_safety>
Do NOT start GPU servers or formal campaigns (`--run`); the coordinator runs all GPU jobs, including the first model
load. No sudo, no `rm -rf`, no bare `pkill -f`, no credentials, no `codex exec`, no tether expose. Disk: /data on this
host has ~1.9 TB free; keep downloads under /data/mw_finetune/models/cosmos_mw.
</action_safety>
<compact_output_contract>
Write `exp/r17_integration/handbacks/HANDBACK_MW_D.md`; end with: weights facts, the trained MetaWorld protocol you
inferred (with evidence), files changed, tests, open risks, and exact coordinator commands: a GPU smoke (2 tasks x 2
idx) and the MT50 pure-inference test (seed 7) on `local` 4090 and on `h100` (with the transfer step for weights).
</compact_output_contract>
