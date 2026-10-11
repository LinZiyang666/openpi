# D: Cosmos Policy MetaWorld adapter

`policy.load_policy(checkpoint, device, **kw)` implements A's `CONTRACT.md` and
imports through `serve.py --model cosmos`. The checkpoint is downloaded and its
full SHA256 is verified. This is a working reconstruction of the released
community policy's inputs, with unresolved camera/config assumptions; it is not
yet a reproduction of the author's reference success rate.

**Follow-up decision:** hold the 500-episode Cosmos-MW evaluation. The
reconstruction probe completed 70 episodes with zero errors in 16m43s of GPU
lock time: assembly 0/10 in every tested configuration, including actual
300-step corner3 rollouts; reach's best result is corner3 1/5. No formal
configuration is validated. Full table and provenance are below.

## Artifacts and provenance

- Checkpoint: `/data/mw_finetune/models/cosmos_metaworld/` on weilandserver,
  [community model](https://huggingface.co/mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B),
  revision `6546ab82a1d302e4a947bc7b03880ec6031a6109`.
- `consolidated.pt`: 3,913,050,026 bytes;
  SHA256 `c8562afc4ee58c20e4c43b8128e72f1e75d9dc1df5a429cde2d112bf6c6436b6`.
  CPU mmap inspection found only 687 `net.*` tensors, with no embedded config.
  See `sources/checkpoint_tensor_metadata.json` and `sources/download_manifest.json`.
- Released statistics and README are preserved in `sources/`. The 50 T5 cache
  entries each have shape `(1,512,1024)`; `sources/t5_cache_metadata.json` records
  their labels and dtypes. We retain the 52 MB pickle beside the large weights,
  outside the code tree.
- Public FlowDAgger: `microsoft/FlowDAgger`
  commit `3426f246510fd67a9e39c3c08bd5b02d586ad688`.
  Its Cosmos runner is absent. Relevant source snapshots are
  `sources/flowdagger_metaworld_pi05_adapter.py` and
  `sources/flowdagger_task_configs.py`; the ignored clone is available locally.
- Runtime: `/home/weiland/projects/openpi_ext/lines/cosmos-policy`, upstream
  commit `18a2accadf4e7a3531e56754102af5a24d2316da`.
  Python 3.10.20, Torch 2.7.0+cu128, TE 2.2, FlashAttention 2.7.3.
  See `sources/environment.json`. GPU inference uses that existing venv;
  evaluation uses A's separate MetaWorld simulator venv and OSMesa.

## Input reconstruction and confidence

| Item | Implemented convention | Evidence / uncertainty |
| --- | --- | --- |
| State | Input `[x,y,z,g]` expands to `[g,-g,x,y,z,0,0,0,1]` | Opposed fingers are explicit in the model README. Statistics show paired means ±0.647938, xyz ranges in channels 2:5, and constant orientation mean `[0,0,0,1]`. Channel order is inferred from these statistics. |
| State normalization | Per-dimension min/max to `[-1,1]`, without clipping | Official `cosmos_utils.py:651`. Released quaternion w bounds are ±0.5 although its mean is 1: identity w becomes **2**, faithfully preserving the published formula. |
| Action | Decode 7-D chunks and select columns `[0,1,2,6]` | Stats rotation channels 3:6 have zero mean/std; gripper channel 6 spans ±1. xyz command ranges exceed ±1, so commands remain unclipped and unscaled. No conversion to target positions. |
| Action normalization | Invert min/max with `0.5*(a+1)*(max-min)+min` | Official `cosmos_utils.py:623`; no mean/std or quantile normalization. |
| Model images | One input view; JPEG quality 95 → resize 224² → center crop 90% area and resize | Official preprocessing `cosmos_utils.py:493,528`. JPEG/augmentation applicability to this checkpoint is assumed; both are options. |
| Camera | A's corner2/id2 at `(0.75,0.075,0.7)`, 480² rotated 180° | Required by A. Public FlowDAgger π0.5 instead uses raw corner3/256 (`metaworld_pi05_adapter.py:32,124`, `shared/task_configs.py:20`). Its Cosmos viewpoint is unknown. Image rotation can undo A's rotation but cannot recover another camera viewpoint. |
| Latent video | Default single-camera slots `[blank,proprio,image,action,future_proprio,future_image,value]`, 7 latents, 25 raw frames, 3 conditioning frames | Generalization of official `cosmos_utils.py:944–1110`. Weights do not encode this count. Explicit duplicate-camera probe options preserve LIBERO (9/33/4) and RoboCasa (11/41/5) slot layouts. |
| Chunk | Default 16×7, projected to **16×4** | Borrowed from official LIBERO inference config (`cosmos_policy_experiment_configs.py:46,183`). Not recoverable from the checkpoint; configurable. |
| Language | Released T5 embeddings only; uncached prompts fail explicitly | All 50 canonical harness prompts are present. `push-back-v3` maps to cache label “Push a puck back to a target behind it”; A's reused push wording is otherwise ambiguous. |
| Sampling | 5 denoising steps; official LIBERO EDM inference sigma max/min 80/4; no CFG or variance scaling | Upstream inference config, not a published MetaWorld recipe. `obs.seed` seeds fresh noise for each request. |
| Execution | First 5 predictions, 15 settle steps, at most 160 control steps | A's harness protocol, not the author's undisclosed Cosmos runner protocol. Public π0.5 assembly registry uses max_timesteps=300. |

Default camera count and chunk length are exposed in stable server metadata as
assumptions. `layout_verified_against_author_runner=false`. Full DiT parameter
coverage is checked during load; incompatible weights raise rather than silently
running partially initialized parameters.

The frozen model generates an action/future/value latent video once, as required
by the model architecture. Only the action latent is read. Future images are not
decoded; predicted futures and values do not steer, select, or rank actions.
There is no warm start, replay/retrieval, online training, or cross-request state.

## Setup and exact commands

Run from `/home/weiland/projects/openpi`. `launch.sh` sets Cosmos's NVIDIA library
paths, existing HF cache, and the workspace/client PYTHONPATH. No token is copied
or printed. The existing NVIDIA VAE cache is reused; on a clean machine that
upstream gated VAE must first be available, or pass a local `tokenizer_path`.

```bash
# Metadata/weight download and full SHA256 verification (no GPU).
exp/mw_finetune/cosmos/launch.sh download --verify

# CPU regression checks (9 tests).
exp/mw_finetune/cosmos/launch.sh test

# CPU audit against official get_action: all 19 data-batch fields identical.
NV=/home/weiland/projects/openpi_ext/lines/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia
CUDA_VISIBLE_DEVICES='' CUDA_HOME="$NV" CUDNN_HOME="$NV/cudnn" \
  HF_HOME=/data/cosmos/hf_home PYTHONPATH="$PWD" \
  /home/weiland/projects/openpi_ext/lines/cosmos-policy/.venv/bin/python \
  -m exp.mw_finetune.cosmos.audit_upstream

# Two episodes through A's actual server/evaluator; obtains GPU lock,
# checks free memory after acquiring it, and bounds its work to <15 minutes.
exp/mw_finetune/cosmos/launch.sh smoke \
  --run exp/mw_finetune/cosmos/results/smoke_single

# Coordinator serving command (long-running; D does not start it permanently).
exp/mw_finetune/cosmos/launch.sh serve \
  --checkpoint /data/mw_finetune/models/cosmos_metaworld --port 8054
```

Optional reconstruction probes use immutable, distinct result directories:

```bash
exp/mw_finetune/cosmos/launch.sh smoke --camera-layout libero_duplicate \
  --run exp/mw_finetune/cosmos/results/smoke_libero_duplicate

# Pass any Options field with A's JSON option mechanism.
exp/mw_finetune/cosmos/launch.sh serve \
  --checkpoint /data/mw_finetune/models/cosmos_metaworld --port 8054 \
  --policy-option 'camera_layout="single"' --policy-option chunk_size=16 \
  --policy-option image_rotation=180 --denoising-steps 5
```

For the coordinator's formal MT50 evaluation, leave the chosen server running
under its lock and start A's CPU evaluator in another terminal:

```bash
export PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src"
export LD_LIBRARY_PATH=/data/mw_finetune/envs/osmesa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
/home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python \
  -m exp.mw_finetune.evaluate --run /data/mw_finetune/reports/cosmos_mw_single \
  --model-id cosmos_mw_single_reconstructed --port 8054 --workers 12 --trials 10
```

No Cosmos fine-tuning or FlowDAgger steering was requested for D. No training
job ran. The SmolVLA 100k/batch64 MT50 recipe in the shared brief applies to the
coordinator's training comparison, not this community frozen-checkpoint probe.
There are no model-size or precision concessions forced by the 48 GB 4090;
bf16 is the upstream runtime convention. The short two-episode sample and
160-step horizon are resource/protocol limits, not a statistically valid SR.

## Verification results

- Download SHA256 matches the model's LFS identity.
- Nine CPU tests pass in both the workspace and Cosmos venvs; the additional
  load-audit regression verifies that missing learned parameters are rejected.
- CPU parity audit passed: duplicate-camera data batch equals upstream
  `get_action` for every key, including video, normalized proprio, T5, and all
  latent indices. Evidence: `results/upstream_cpu_audit.log`.
- **4090 smoke completed**, 2026-10-09 00:10–00:12 CDT: **0/2 successes**, zero
  evaluation errors, A's actual WebSocket serving and evaluation modules.
  Total work under the lock was **104.68 s**, starting with 48,511 MiB free.
  The smoke process shut down its own server and released its GPU lock.
- Both tasks used idx 0, benchmark seed 7, the unchanged corner2/480²/180°
  render, 15 settle steps and first-5 replanning. Both produced 32 full policy
  queries and ended at the 160-step cap:

  | Task | Success | Control steps | Queries | Inference time | Episode wall time |
  | --- | --- | --- | --- | --- | --- |
  | assembly-v3 | No | 160 | 32 | 22.9299 s (716.6 ms/query) | 37.72 s |
  | reach-v3 | No | 160 | 32 | 22.7979 s (712.4 ms/query) | 36.01 s |

- Synthetic WebSocket inference returned finite contiguous float32 `(16,4)`
  chunks. First/warm query latencies were 1.877/0.718 s. Repeating the same
  episode identity/seed gave **max absolute action difference 0**. Its native
  gripper commands were approximately 0.592–0.598, without binarization.
- All learned DiT parameters loaded, with no unexpected learned weights.
  113 TransformerEngine `_extra_state` bookkeeping entries were excluded for
  bf16 inference; remaining missing keys are only those entries, recorded in
  metadata. A first load-only attempt exposed Cosmos's outer
  `load_state_dict(strict=False)` returning None; corrected by auditing/loading
  the net submodule. That failed attempt is preserved under
  `results/load_attempt1/` (102.08 s), and ran no environment episode.
- Evidence: `results/cpu_tests.log`,
  `results/smoke_single/{server.log,eval.log,smoke.json,synthetic_actions.npy}`
  and `results/smoke_single/eval/{manifest.json,episodes.jsonl,summary.json}`.
  The summary's `complete=true` covers these two requested smoke identities,
  not a formal 500-episode MT50 report. No full benchmark or training ran.

The adapter is servable, but this smoke does not establish task competence.
The zero successes should be interpreted alongside the unresolved camera,
latent-layout and horizon reconstruction, rather than as the author's base SR.

## Open questions for the model author / coordinator

1. Original MetaWorld Cosmos config: camera(s), native rendering resolution,
   orientation, spatial crop/JPEG, temporal latent layout, action horizon,
   and inference sigma/sampler settings.
2. Exact padded proprio order/orientation convention and native xyz action
   units. The supplied moments strongly support this reconstruction, but do not
   formally prove its frame conventions.
3. Author's 25-episode assembly benchmark seeds, initializations, horizon and
   replan interval. Its README reports approximately 52% base and 84–88%
   steered SR; those are not MT50 claims, and the 50-prompt cache does not prove
   uniform MT50 training coverage.
4. Resolve the camera distribution before using a large MT50 run as a model
   quality comparison. The current adapter intentionally preserves A's common
   protocol, so unsuccessful rollouts can reflect reconstruction mismatch.

## Reconstruction-disambiguation probe (2026-10-09)

Requested scope: assembly-v3 idx 0..9 plus reach-v3 idx 0..4, benchmark seed 7.
Five CPU workers per config, four independent instances of A's serving path
concurrently on the 4090 under one `flock /tmp/r17_gpu.lock`. All configs retain
five denoising steps, 15 settle steps, opposed gripper encoding, min/max
normalization and first-five replanning. No model weights are changed.

The fourth arm varies **chunk length 16 → 32**. The released nine-dimensional
proprio resembles the padded quaternion convention rather than native 4-D
MetaWorld state, and the upstream RoboCasa inference recipe uses 32 actions
(`cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py:182`);
the checkpoint itself does not encode the horizon. The action-column mapping
has stronger statistics support, and the camera comparison below argues
against undoing A's rotation, so those remain fixed in this arm.

Immutable results root:
`exp/mw_finetune/cosmos/results/reconstruction_20261009_v1/`, with four arms
`01_default`, `02_libero_duplicate`, `03_flowdagger_view_300`, and `04_chunk32`.
Each has independent manifests/journals for assembly and reach, a server log,
CUDA graph parity evidence, and a per-query service-time/action trace.
`plan.json` records all options and source hashes; `report.json` summarizes
actual completed identities, outcomes and timings.

```bash
# CPU statistics/prompt comparison, from the workspace venv (PyArrow).
.venv/bin/python -m exp.mw_finetune.cosmos.compare_dataset

# CPU probe view rendering (private OSMesa, no GPU).
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 \
  PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src" \
  LD_LIBRARY_PATH=/data/mw_finetune/envs/osmesa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
  /home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python \
  -m exp.mw_finetune.cosmos.inspect_views
.venv/bin/python -m exp.mw_finetune.cosmos.inspect_views --compare-only

# All four arms, with one shared lock and a hard total <20-minute GPU bound.
# Historical v1 used the original shim preserved in sources/; its corner3
# horizon is actually 160. Current code fixes that override. Use a fresh
# --run path when reproducing; existing result directories are immutable.
exp/mw_finetune/cosmos/launch.sh probe \
  --run exp/mw_finetune/cosmos/results/reconstruction_20261009_v1

# Actual corrected repetition: same view/policy/seeds, only assembly repeated.
# 422 seconds was the remaining GPU budget after the completed first pass.
exp/mw_finetune/cosmos/launch.sh probe \
  --only-flowdagger --assembly-only --base-port 8074 --seconds 422 \
  --run exp/mw_finetune/cosmos/results/reconstruction_20261009_flow300_v2

# Combined adapter and render/horizon shim regression checks (15 tests).
.venv/bin/python -m pytest -q \
  exp/mw_finetune/cosmos/test_policy.py exp/mw_finetune/cosmos/test_probe.py
```

The existing `exp.cosmos_nfe.cudagraph_net.GraphedNet` accelerates the common
denoising path. Before opening each server, a same-seed eager/graph action
comparison must be bitwise equal. Numerical mismatch stops that arm instead
of silently changing its sampler. Static graph buffers are fully overwritten
per call; they cache execution, not predictions or policy history.

Default, duplicate-camera and chunk32 arms use `exp.mw_finetune.evaluate`
unchanged. The probe-only `probe_evaluate.py` invokes A's `episode()` function
and reuses the MetaWorld episode kernel, modifying only environment creation
for native corner3/256 and the assembly horizon of 300. A local `RawView` proxy
cancels the kernel's 180° image rotation. The benchmark, task identity, settled
state and action execution remain the same. `probe_server.py` invokes A's
`serve.main()` with a process-local allowance for 256² images, plus timing and
graph parity instrumentation. Shared A and `exp/metaworld` source files are
not edited.

The initial v1 corner3 arm actually ran **160 control steps**, despite its
directory name and assembly manifest saying 300. A's episode runner supplies
`max_steps=160` explicitly, overriding the original shim's `functools.partial`
default. We discovered this from the first completed episode journals, kept
the original results and manifests intact, and fixed the shim to replace the
explicit keyword. A regression test covers that exact failure. Treat v1's
corner3 assembly as a **160-step** view comparison; its manifest's 300-step
claim is invalid. The corrected standalone 300-step assembly repetition is
kept in a separate result root, with the same policy/view/seeds and five workers.
Reach remains the valid 160-step view probe from v1.
`sources/reconstruction_v1_probe_evaluate.py` preserves the original shim;
its SHA256 exactly matches the original `plan.json`. The corrected run also
stores executable source snapshots beside its source hashes.

### Training-data comparison

`results/dataset_comparison.json` compares projected checkpoint action columns
`[0,1,2,6]` and reordered proprio columns `[2,3,4,0]` against the local dataset's
`meta/stats.json`. **Neither state nor action min/max matches at atol 1e-6.**

| Native feature | Checkpoint min / max | LeRobot min / max |
| --- | --- | --- |
| action x | -11.369145 / 19.279388 | -11.474599 / 19.455713 |
| action y | -14.908533 / 12.490139 | -14.927049 / 12.468915 |
| action z | -17.274719 / 24.626646 | -17.279015 / 24.625469 |
| state x | -0.499129 / 0.471285 | -0.505536 / 0.486881 |
| state y | 0.387955 / 0.891770 | 0.386600 / 0.903830 |
| state z | 0.044278 / 0.469502 | 0.043383 / 0.508170 |
| state g | 0.262919 / 1 | 0.256266 / 1 |

Action gripper limits are identical at ±1. Means/stds differ too: checkpoint
g mean 0.647938 versus dataset 0.613908; checkpoint xyz action means
`[0.276733,0.558045,-0.054320]` versus dataset
`[0.254733,0.509698,-0.126900]`. The prompt cache matches **49/50** exact dataset
labels; push-back is the sole wording difference. These similarities support
a related MetaWorld collection or conversion, but do not prove the exact
dataset revision, training split or frame filtering.

Direct image evidence supports the dataset camera convention. The first
assembly training frame was extracted from `data/chunk-000/file-000.parquet`.
Against assembly-v3 idx0 after the standard settle, normalized RGB MAE is
**0.01088** for A's corner2/480/180° view, **0.21770** for raw corner2, and
**0.28880** for raw corner3/256 resized for comparison. These frames have
different goal initializations, so this is viewpoint evidence, not a claim of
pixel-perfect replay. The two probe renderers have identical settled 4-D
states. Evidence: `results/viewpoint_evidence/` (original frames, states and
`image_comparison.json`).

### Probe results and recommendation

All 60 episodes completed with zero evaluation errors. The first pass used
757.87 seconds under the 4090 lock; all four eager/CUDA graph audits were
bitwise equal. The table reports mean **server service time per query**, across
assembly and reach, including GPU contention between concurrent servers.
Client queue/round-trip times are separately recorded in `report.json`.
These service timings are not isolated single-model latency measurements.

| Config | Assembly successes / episodes | Reach successes / episodes | ms/query |
| --- | --- | --- | --- |
| Single; corner2 480² rot180; chunk16; 160 steps | 0/10 | 0/5 | 1383 |
| `libero_duplicate`; same camera/chunk/horizon | 0/10 | 0/5 | 1468 |
| Single; corner3 raw256; chunk16; **160 actual steps (v1)** | 0/10 | 1/5 | 1390 |
| Single; corner2 480² rot180; **chunk32**; 160 steps | 0/10 | 0/5 | 1380 |
| Single; corner3 raw256; chunk16; **300 actual assembly steps (v2)** | 0/10 | 1/5 reused | 357* |

The reach success is corner3 idx3, at control step 49. All other first-pass
episodes ended at the 160-step cap. The corrected corner3/300 assembly run is
complete in `results/reconstruction_20261009_flow300_v2/`: all ten episodes
actually executed **300 steps / 60 queries**, and none succeeded. Reach is not
repeated because only assembly's horizon changes. Its 422-second GPU cap was
computed from `1180 - first_pass_wall_s`, reserving cleanup time inside the
total 20-minute budget. Waiting for other agents' locks consumed no GPU budget.

`*` The corrected arm ran alone, so its **357 ms/query** is isolated service
time for assembly; the first four timings include four-server GPU contention.
The corrected eager/graph parity audit is also bitwise exact. All **320**
matching first-160-step query traces reproduce v1's states, first actions and
action extrema exactly (maximum absolute difference zero). Longer episodes
therefore added the intended horizon without changing their earlier behavior.

Total work under the lock: **757.87 + 244.87 = 1002.75 seconds (16m43s)**,
70 distinct episodes, 2,498 policy queries, **zero evaluation errors**.
Both controllers stopped their own servers/workers and released the lock.
The latest CPU checks are **15 passed**. Consolidated evidence and the explicit
v1 metadata correction are in `results/reconstruction_20261009_summary.json`.
Each underlying result directory remains separate and unchanged.

**Recommendation: do not launch the 500-episode run as a validated Cosmos-MW
model-quality comparison yet.** Neither the camera alternative, duplicate
latent layout, longer decoded chunk nor 300-step assembly horizon recovers
the README's base assembly behavior. The single reach success shows that the
policy can produce a successful rollout, but 1/5 does not establish a formal
configuration. The author's reference seeds, replanning and inference setup
remain unknown, so these results do not refute the advertised 52% reference.

For a required **diagnostic** run using A's common protocol, retain the current
default: **single layout, corner2/480/rot180, chunk16, five denoising steps,
opposed gripper, first-five replanning, 160 control steps**. Dataset image
evidence favors this camera and the other layouts/chunk did not improve
assembly. Label its SR as an unvalidated adapter reconstruction. For the next
author-specific sanity check, keep corner3/raw256/single/chunk16/assembly300
available as a probe: it produced the only reach success, but is not a selected
formal MT50 configuration. Before spending 500 episodes, obtain the author's
training/eval config, especially latent slots/conditioning count, image
preprocessing, chunk length, sampler/sigma, proprio padding and replan interval.
