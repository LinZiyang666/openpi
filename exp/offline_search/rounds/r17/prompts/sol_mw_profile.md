# sol task: batch-1 latency profile (IR cost) for the four MetaWorld MT50 models

Repo: `/home/weiland/projects/openpi` (weilandserver). Read first:
- `exp/r17_integration/profile/README.md`, `latency.py`, `certify.py`, `tests/test_latency.py` — the LIBERO profiler (dp_libero, fastwam_libero, cosmos_libero) and its stage boundaries.
- `exp/r17_integration/common/serving.py` — stage1 / stage2 / stage3 (sample) boundaries the cache system prices.
- The MT50 adapters: `exp/r17_integration/models/{dp,fastwam,pi05,groot}/` (MW providers/staged adapters) and how `exp/r17_integration/qualification/assets.py::model_assets` loads each MW model.

## Goal
Extend the profiler so `python -m exp.r17_integration.profile.certify --model <m> --out-dir exp/r17_integration/profile/results_mt50`
measures the same three stage costs for the MetaWorld MT50 models, with these step budgets (full, reduced):

| model | checkpoint | full K | reduced K |
|---|---|---|---|
| `dp_mw` | `/data/mw_finetune/dp/mt50_policy_100000` (DDPM100 / trailing DDIM eta0) | 100 | 10 |
| `fastwam_mw` | `/data/mw_finetune/fastwam/mt50_frozen_b128/policy_016001` | 10 | 3 |
| `pi05_mw` | `/data/ckpt/pi05_metaworld_rlinf` | 10 | 2 |
| `groot_mw` | `/data/mw_finetune/groot/4090_mt50/checkpoint-60000` | 4 | 2 |

- Inputs: one real MT50 observation per model from its library parent (`/data/r17_library/<m>_mt50/bpool`, row 0; MT50 has one 480² camera, rotated as in the dataset, 4-D state, task prompt from `exp/metaworld/tasks.py`). There is no suite split: replace the LIBERO `--suite` loop with a single `mt50` profile.
- Stage boundaries: the same as the existing profiler and `common/serving.py` (stage1 = prepare + run_stage1 incl. the retrieval key; stage2 = run_stage2; stage3 = the complete `sample` call at full and reduced K). For π0.5 and GR00T use their MW staged adapters' own stage split; write it into the README table like the LIBERO rows.
- Same measurement method as LIBERO: batch 1, compiled tensor programs where the LIBERO profiler compiles, CUDA-event timings, warmup/iterations defaults, actual NFE recorded. Output receipts in the same schema with `profile="mt50"`.

## Rules
- Only do this. Do not add validation layers, gates, fingerprint/identity checks or admission logic beyond what the LIBERO profiler already does (owner rule: keep it simple).
- **CPU only.** Do not start any GPU process: both GPUs are running experiments. Write CPU tests (extend `tests/test_latency.py`) with stubs. The coordinator runs the GPU measurement on an exclusive 4090.
- Do not change model-owned code under `exp/r17_integration/models/*`; import from it.
- No git commit/push, no `pkill`, no `rm -rf`, do not touch running processes, `/archive`, `/data/i018`, `/home/weiland/idea-sprint`. No `tether push/pull`. Never handle credentials.
- Handback: append a section `## MT50` to `exp/r17_integration/profile/README.md` with the exact GPU commands for the four models, the stage table, and what the CPU tests cover. Keep it short.
