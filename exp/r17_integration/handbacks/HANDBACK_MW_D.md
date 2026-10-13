# Cosmos Policy MetaWorld → R17

`cosmos_mw` is implemented for the stock MT50 wire, pure inference and later
teacher capture. CPU adapter/capture tests and real-frame preprocessing passed.
The coordinator must perform the first policy/VAE GPU load and closed-loop run.
No GPU server or campaign was started; no commit was made.

## Weights facts

Anonymous public download from
[mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B](https://huggingface.co/mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B/tree/6546ab82a1d302e4a947bc7b03880ec6031a6109),
revision `6546ab82a1d302e4a947bc7b03880ec6031a6109`, `gated=false`,
`private=false`. Candidate downloads used Python urllib without authentication.
All release files are under `/data/mw_finetune/models/cosmos_mw`:

| File | Bytes |
| --- | ---: |
| `consolidated.pt` | 3,913,050,026 |
| `t5_embeddings.pkl` | 52,446,356 |
| `dataset_statistics.json` | 1,589 |
| `.gitattributes` | 1,519 |
| `README.md` | 896 |
| **Release total** | **3,965,500,386 (3.97 GB / 3.69 GiB)** |

`download_manifest.json` records sizes and SHA256 for every release file;
both LFS hashes matched public repository metadata. Weight SHA256:
`c8562afc4ee58c20e4c43b8128e72f1e75d9dc1df5a429cde2d112bf6c6436b6`.
Embedding SHA256:
`09e098988b1f442a0043995734faadd475ff82bbeab2c04d5127b760249a735c`.

The exact model card is retained in `README.md`; metadata/license (`other`)
is retained in `public_repo_metadata.json`. The card describes a community
FlowDAgger release recipe for `assembly-v3`, with opposed gripper proprio,
and an approximately 52% frozen-policy reference score on 25 episodes. That
score is not an MT50 result. No `config.json`, training metadata or runner was
released. CPU mmap inspection found 687 `net.*` state-dict entries and
1,956,414,180 tensor elements, mostly bfloat16; no serialized training config.
See `checkpoint_inspection.json` for tensor names/shapes.

## Trained protocol evidence and implemented inference

**The exact trained camera, crop, temporal layout, action chunk and execution
horizon cannot be recovered from this release.** The table distinguishes
checkpoint evidence from the reconstructed protocol. This permits measuring
this adapter on MT50; it does not establish reproduction of the author's run.

| Component | Implementation and evidence |
| --- | --- |
| MetaWorld version/task coverage | Card identifies `assembly-v3`. All 50 cached embeddings are bfloat16 `[1,512,1024]`; they cover all 49 distinct R17 MT50 prompt strings, plus a separate push-back instruction. This supports MT50 language coverage but does not prove MT50 training coverage. Public [FlowDAgger code](https://github.com/microsoft/FlowDAgger) has no Cosmos backend. Training demos, task mixture, seed and camera remain unpublished. |
| Camera | Exactly one stock R17 `corner2` uint8 RGB480 image, camera position `(0.75,0.075,0.7)`, already rotated 180°. Adapter preserves orientation; no wrist view is invented or encoded. Training viewpoint/orientation is an assumption. |
| Resolution/crop | Native helper: JPEG95 → 224×224 → center crop of 90% area (`int(224*sqrt(.9))=212` pixels) → 224×224. Inherited upstream augmentation defaults; unpublished for this checkpoint. |
| Conditioning | Current observation only. Raw video `[1,3,25,224,224]` → VAE latent `[1,16,7,28,28]`. Latent order: blank, proprio, camera, action, future proprio, future camera, value. Three conditional latents; current camera idx2, action idx3. Four copies per noninitial raw frame implement native temporal compression, not four historical observations. This is upstream sequence construction with wrist disabled; trained temporal layout is unknown. |
| Wire state/proprio | Wire remains raw `obs[:4]=[xyz,g]`. Native input is `[g,-g,x,y,z,0,0,0,1]`. Opposed fingers are explicit in the card and stats. Nine-element stats have finger means `±0.647938`, xyz means `[.013373,.630890,.169204]`, quaternion mean/median `[0,0,0,1]` with zero std. Identity quaternion padding is inferred from these stats. |
| Normalization | Released flat arrays `proprio_{min,max,mean,std,median}` (9 each), `actions_{min,max,mean,std,median}` (7 each). Use native min/max, not mean/std: `p_norm=2*(p-min)/(max-min)-1`; `a=.5*(a_norm+1)*(max-min)+min`. No adapter clipping. |
| Actions/action space | Native prediction `[16,7]`; min/max inverse first, then select channels `[0,1,2,6]` → `[16,4]` MetaWorld xyz/gripper. Rotation means/std are zero with artificial bounds `±.5`; xyz bounds exceed `±1`, consistent with unbounded scripted commands subsequently clipped by the environment. Preserve command scale and gripper sign; no added gain or coordinate transform. |
| Chunk/execution horizon | **Trained chunk and execution horizon: unpublished/unknown.** Adopt upstream LIBERO chunk16, execute5 via the existing MT50 kernel, discard the remaining predicted tail and replan. Serving sends `[5,4]`; teacher capture retains normalized `[16,7]`. Do not label five as the trained horizon. |
| Prompt | Exact strings from `exp/metaworld/tasks.py`, no prefix/template rewriting, looked up in released T5 cache. R17's shared push/push-back label is preserved; its separate cached push-back wording is not substituted. |
| Denoising/RNG | Inherited native K5 EDM `2ab`, sigma80→4, final clean step, no CFG/variance scaling/best-of-N steering. Actual K1/K2/K3/K5 NFE is 1/1/3/5. Training/evaluation K is unpublished. Environment seed7; policy uses stock R17 request RNG, with initialization seed7 and per-decision process RNG reset0. Teacher capture accepts the same request seed and does not alter sampling. |
| MT50 evaluation | Unchanged R17 seed7, 15 zero-action settles, at most160 policy controls, success at any step. Current formal spec selects idx0–49 per task: **2,500 test episodes**. Smoke selects assembly/basketball, idx0/1: **four episodes**. |

Native code is commit `18a2accadf4e7a3531e56754102af5a24d2316da` at
`/home/weiland/projects/openpi_ext/lines/cosmos-policy`; H100 vendor is
`/data/cosmos/cosmos-policy`. Evidence:
[image preprocessing](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/cosmos_utils.py#L498),
[normalization](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/cosmos_utils.py#L623),
[native sequence construction](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/cosmos_utils.py#L912),
[inherited chunk/K defaults](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/libero/run_libero_eval.py#L191).
`inferred_config.json` records the resolved overrides: state_t7, condition3,
tokenizer25, no wrist, chunk16, sigma4/80. CPU composition needed temporary
import-only availability/current-device mocks; upstream `config.validate()`
itself invokes CUDA and was left to coordinator model loading.

## Files changed

- `models/cosmos/metaworld.py`: MW manifest, loader, wire policy, one-camera
  VAE keys, adapter and native-generator startup comparison.
- `models/cosmos/{staged,provider,loader,qualify,serve}.py`: small shared hooks
  for MW selection, subclass cloning, dynamic batch/action indices, capture
  seed behavior and truthful metadata. Tokenizer lookup is cached/local and
  explicitly `token=False`. MW loading bypasses upstream experiment
  registration's unrelated HF checkpoint resolution; only local weights and
  tokenizer are loaded. No vendor edit or new admission/fingerprint layer.
- `models/cosmos/test_metaworld.py`: fake-model CPU tests including the real
  library capture path. `models/cosmos/check_metaworld_shapes.py`: four-frame
  rendering and GPU-free native preprocessing comparison.
- `ops/formal_eval.py`: only `DEFAULT_CHECKPOINTS['cosmos_mw']` and the Cosmos
  model selector boolean. Concurrent package A/B changes were preserved.
- This handback. No benchmark, debug, library plumbing or `src/` edits.

## Tests

CPU MW/capture tests: **25 passed** (`test_metaworld.py` + `test_capture.py`).
The MW teacher path captures exactly one original uint8 image, raw four-state,
256-dimensional camera key, four-dimensional state key and native `[16,7]`
normalized chunk; executing five actions is recorded correctly.
Final focused run including the remaining valid LIBERO adapter tests:
**32 passed, one outdated port-default test deselected**.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:packages/openpi-client/src:. .venv/bin/python -m pytest -q exp/r17_integration/models/cosmos/test_metaworld.py exp/r17_integration/models/cosmos/test_capture.py exp/r17_integration/models/cosmos/test_cpu.py -k 'not protocol_does_not_change_existing_r17_identity'
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. /home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python -m exp.r17_integration.models.cosmos.check_metaworld_shapes --render-only
```

For the second shape-check phase, use the existing Cosmos runtime environment
(`server_env` below only constructs environment strings; it launches nothing):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import os, subprocess
from exp.r17_integration.ops.formal_eval import server_env, ROOT
vendor = '/home/weiland/projects/openpi_ext/lines/cosmos-policy'
env = dict(os.environ, **server_env(dict(model='cosmos',server_node='local',vendor=vendor),str(ROOT)))
env['CUDA_VISIBLE_DEVICES'] = ''
subprocess.run([vendor+'/.venv/bin/python', '-m',
               'exp.r17_integration.models.cosmos.check_metaworld_shapes'],env=env,check=True)
PY
```

Real MetaWorld3.0.0 assembly/reach idx0/1, seed7, rendered only while the local
4090 showed 0% utilization / 0 MiB. `real_mt50_frames.{npz,json}` and
`preprocessing_shape_check.json` are beside the weights. All four reconstructed
batches equal upstream preprocessing tensor-for-tensor; no policy/VAE was
loaded and CUDA initialization remained false. Keys in that shape check use
a synthetic latent; genuine VAE key values await coordinator GPU loading.

`cpu_integration_check.json` records the existing formal CPU asset/config/plan
path passing, with the four-episode smoke and 2,500-episode MT50 plan.

Broader existing Cosmos CPU/cache/recovery tests: 38 passed, 15 failed.
Failures concern old driver-port23249 expectations (current default23112),
removed content/admission checks, periodic-gap rejection and old recovery
checks in untouched code. These were not restored under the owner's rule.
The handback's commands use explicit ports. Final focused test log is
`/data/mw_finetune/models/cosmos_mw/cpu_tests.log`.

## Open risks

1. Single-camera layout, camera/crop, chunk16 and K5 remain reconstruction
   choices; exact training configuration, trained execution horizon and MT50
   data coverage are unknown. Model-card assembly SR is not comparable to R17.
2. First GPU policy/VAE load, memory peak, numerical parity and MT50 SR are
   pending on both hosts. Existing native startup comparison is retained,
   using `generate_samples_from_batch` versus stages on the reconstructed MW
   batch; it cannot certify the author's runner.
3. Upstream `get_action` references unset wrist indices when wrist input is
   disabled. MW serving bypasses that helper's broken return/future-decode
   branch; the CPU comparison stops at its model-input boundary.
4. `debug_capture` and generic library teacher capture are implemented/tested;
   CP1 retrieval quality is unverified and MW cache recipes remain unsupported.
   Package A owns collection CLI/plumbing. No library was collected here.

## Exact coordinator commands

Run from `/home/weiland/projects/openpi`. These are proposed commands, **not
executed GPU jobs**. Select vacant ports/resources when coordinating with the
live queues. No `--official-50` is needed for MetaWorld; seed7 is the unchanged
benchmark seed. Each full run below schedules the current 2,500 test cells.

Local 4090 smoke (assembly/basketball × idx0/1), then MT50 pure inference:

```bash
exp/r17_integration/ops/formal_eval.sh --model cosmos_mw --checkpoint /data/mw_finetune/models/cosmos_mw --server-node local --worker-node timan107 --replicas 1 --workers 2 --gpu cuda:0 --memory-budget-gib 10 --port 23140 --driver-port 23112 --smoke --seconds 1200 --episode-seconds 600 --retries 0 --run
exp/r17_integration/ops/formal_eval.sh --model cosmos_mw --checkpoint /data/mw_finetune/models/cosmos_mw --server-node local --worker-node timan107 --replicas 1 --workers 16 --gpu cuda:0 --memory-budget-gib 10 --port 23140 --driver-port 23112 --seconds 86400 --episode-seconds 1800 --retries 2 --run
```

H100 transfer first: use package A's direct rsync helper with a vacant transfer
port23119 and `--bwlimit=16384` KiB/s. Payload goes directly to
`149.165.153.233`; Tether executes only daemon/control commands. No SSH/HF
credentials are required by this transfer code. It transfers only release
assets and their download record, not the local CPU test directory.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from exp.r17_integration.ops import direct_transfer
from exp.r17_integration.ops.formal_eval import remote, execute
from exp.r17_integration.models.cosmos.metaworld import RELEASE_FILES
direct_transfer.PORT = 23119
direct_transfer.BWLIMIT_KIBPS = 16384
root = Path('/data/mw_finetune/models/cosmos_mw')
for name in (*RELEASE_FILES, 'download_manifest.json', 'public_repo_metadata.json'):
    direct_transfer.transfer(['rsync', 'push', str(root/name), 'h100:'+str(root/name)],
                             remote=remote, execute=execute)
PY
```

H100 smoke (same four cells), then MT50 pure inference. The existing Cosmos
runtime/tokenizer on H100 is reused; the launcher deploys the R17 code.

```bash
exp/r17_integration/ops/formal_eval.sh --model cosmos_mw --checkpoint /data/mw_finetune/models/cosmos_mw --server-node h100 --worker-node timan107 --replicas 1 --workers 2 --gpu cuda:0 --memory-budget-gib 10 --port 23240 --driver-port 23112 --smoke --seconds 1200 --episode-seconds 600 --retries 0 --run
exp/r17_integration/ops/formal_eval.sh --model cosmos_mw --checkpoint /data/mw_finetune/models/cosmos_mw --server-node h100 --worker-node timan107 --replicas 1 --workers 16 --gpu cuda:0 --memory-budget-gib 10 --port 23240 --driver-port 23112 --seconds 86400 --episode-seconds 1800 --retries 2 --run
```

## Coordinator GPU check fix — 2026-10-12

Root cause: the blanket `get_checkpoint_path(..., return_value=cfg.ckpt_path)`
override replaced even the already-local tokenizer path with `consolidated.pt`.
WanVAE therefore received the policy's `net.*` state dict and failed with
missing encoder/decoder keys and unexpected policy keys.

Fix in `models/cosmos/metaworld.py`: the resolver now maps only the explicit
base/ALOHA policy registration defaults to the MW policy file, maps the
tokenizer HF URI to `tokenizer_path(spec)`, and preserves local paths.
The real Wan VAE loaded from:

```text
/data/cosmos/hf_home/hub/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/f50c09f5d8ab133a90cac3f4886a6471e9ba3f18/tokenizer/tokenizer.pth
```

Direct MW `cpu_manifest(LoadSpec(...))` also defaults an omitted benchmark
binding to the existing MT50 protocol; this supports the coordinator script's
initial empty-protocol descriptor. Added a CPU regression test for distinct
policy/tokenizer resolution. MW + shared capture tests: **26 passed**.

Ran the coordinator's unchanged script
`/home/weiland/.claude/jobs/35f2869b/tmp/cosmos_mw_gpu_check.py` using the vendor
Python and `formal_eval.server_env(model='cosmos', server_node='local', vendor=...)`.
Added `CUDA_VISIBLE_DEVICES=0`, `HF_HUB_OFFLINE=1` and
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` to the child environment. Exit status **0**.
Log: `/data/mw_finetune/models/cosmos_mw/gpu_check_resolver_fix.log`.

Passing output:

```text
_IncompatibleKeys(missing_keys=[], unexpected_keys=[], incorrect_shapes=[])
loaded 51.6 s; mem GiB 7.55
parity: PASS; K5/K3/K1/K2 byte-exact float64 [16,4]; actual NFE 5/3/1/1; tolerance 0
frames keys ['image', 'state', 'prompt']
K5 call s [0.685, 0.718, 0.690, 0.710, 0.694, 0.699, 0.759, 0.712, 0.716, 0.764, 0.745, 0.796]
median 0.714 actions (16, 4) [0.199 0.030 -0.009 0.589]
peak mem GiB 7.55
```

Memory is PyTorch peak allocated memory, including loading and all inference
calls. Timing uses the first real assembly frame and the script's fixed state
`[0,.6,.2,.6]`; median excludes the first two calls. This establishes native
generator/staged parity on the reconstructed protocol, not closed-loop SR or
parity with the unavailable author's runner. It supersedes the earlier
local-load/parity pending status; H100 GPU verification remains pending.

The check process exited normally. Afterwards `nvidia-smi` showed the 4090 at
0% utilization, 0 MiB used and no compute processes. No server, worker, formal
campaign, file transfer or commit was started.

PASS
