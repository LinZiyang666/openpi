R12 sol prep handback — 2026-10-03 CDT

Both requested parts are complete. All writes are under closed_loop/ or this prep directory. No closed-loop server, worker, simulator rollout, remote setup/sync/chain, or git command was run. No R12 explorer directory was opened. CPU computation used taskset 22-37,66-81 and single-threaded BLAS. GPU use was limited to the bounded latency benchmark processes, which have exited.

**Recording implementation and boundary**

The new module is `exp/offline_search/closed_loop/look_recording.py`. Recording is default off. Enable it with `--os-record-looks`; optional flags are `--os-record-look-tokens`, `--os-record-look-images N` (1–256; zero disables images), and `--os-record-look-shadow`.

The coordinator dependency planner owns `--os-look-contract` and `--os-look-contract-sha256`; do not put those in arms.json yourself. It refuses every recording-related flag on A, missing/implicit pool declarations, non-plugin arms, mixed roots, and overlap with the served B subset. Every arm in a recording root must explicitly declare `dev=true, init_pool=B`. Existing devset checks still bind the actual library-only prefit, constructor/subset, exact B selection, and B init-file digests. The recording planner omits test-A query metadata; the recording runtime never opens it.

The server requires the SHA-bound contract to match model, cell, and method, and binds the output directory to that contract's actual dev arm directory. A B contract cannot be redirected to an A root. The B worker issues a contract-derived `run_id` stamp only after its existing pool, digest, manifest, and prefit checks. Episode metadata must carry that stamp, the matching arm UID, and a selected task/init pair. Unlabelled/test-A or foreign-arm episodes are refused before retrieval. Recording-enabled startup skips the existing git-head probe. No src/ changes or model/retrieval arithmetic changes were made.

Files live at `RUN/runs/ARM/server_PORT/looks/`. Each successful real vision decision gets a compressed NPZ shard; blind decisions get none. Failed episodes' successful decisions are included. A per-process `manifest_UUID.json` contains schema `offline_search.dev_looks.v1`, the full B contract, options, field descriptions, and each shard's SHA256, UID, attempt, step, task, and init. Atomic shard/manifest replacement and unique process UUIDs prevent partial indexed writes and resume overwrites. The reader checks hashes and pool labels and uses `allow_pickle=False`. Interrupted unindexed shards are ignored; errors are surfaced rather than silently treated as complete records.

Persisted arrays and metadata:

- `key_v0`, `key_v1`: exact live `ctx.query_keys[vision_0/vision_1]`, float32 vectors of length 32768. These are the cache key builder's 4×4 adaptive spatial averages of 256×2048 stage-1 tokens, flattened, before R10's PCA/Mahalanobis projection. v0 is agentview/third-person; v1 is wrist. There is no extra encoder call. For an existing partial-camera path the arrays preserve whatever that serving path computed, including missing-view placeholders; `camera_mode` identifies that situation. No missing view is invented.
- `rs`: live normalized robot_state key (π0.5 32-D; GR00T 8-D); `raw_state`: wire physical state converted to float32.
- `step`, `task_id`, `init`, and JSON `meta`: UID, attempt, client episode_id, task text, suite/model, connection/bundle, library, camera mode, and context step. B episodes use index −1 rather than test-A's episode map.
- `action`: actual executed normalized H×32 chunk, including the policy chunk on a MISS. `wire_actions`: actual served client chunk after output transforms. `hit` distinguishes the source.
- `topk`, `scores`, `distances`, `weights`: the complete Result, before the ordinary ten-neighbor log truncation. R10 uses 16 neighbors. Distances are the actual negative Result scores for AWM/R10; the convention is recorded. Generic methods without that convention retain scores and NaN distances. `donor_rows`/`donor_weights` are the actual synthesis anchor. R10 subset donors are mapped once to the returned parent-library row space. Unavailable weights are NaN; no substitute kernel is fitted or recomputed.
- JSON `guard_state`: full decision extras, no-progress span and vision-progress history, guard thresholds/disabled bits and progress/flag/burst/return memos, R11 guard/trigger sets where present, plus judge verdict/tau/run, burst state, blind age, and look reason. Captured before MISS anchor invalidation.
- Optional `tok_v0`, `tok_v1`: exact key-builder stage-1 token slices in float16, 256×2048, independently of method access via `--os-tokens off`.
- Optional `img0`, `img1`: nearest-neighbor downsample of the wire RGB to N×N, uint8, independently of method token access. These are wire images, not another preprocessing/vision pass.
- Optional `shadow_policy`: same-stage-1 full policy chunk. A served MISS supplies its actual chunk; a HIT uses explicit private deterministic noise and the full policy remainder. The served action/history and global RNG are preserved. The extra diagnostic policy work is not charged as an executed policy call in the original cost ledger. Shadow mode requires a full model and is expensive; CPU replay labels use a fake policy, as recorded in the proof.

Reader examples:

```python
from exp.offline_search.closed_loop.look_recording import read_looks
for look in read_looks('/path/to/RUN/runs/ARM/server_PORT/looks'):
    print(look['meta']['uid'], int(look['step']), look['key_v0'].shape)
```

```bash
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
"${P[@]}" -m exp.offline_search.closed_loop.look_recording inspect /path/to/RUN/runs/ARM/server_PORT/looks
```

**Verification**

`evidence/tests.log`: **226 passed**. Tests cover real PluginSession/_ConnPolicy execution with fake observations; cache and MISS served chunks; shadow labels and private RNG; independent token/image options; full neighbors, distances, weights and subset row mapping; blind exclusion; episode/contract/root/output-path refusal; corruption and resume; local and H100 plan mapping; and existing devset/H100/local/robustness regressions.

`default_unchanged.json` and `evidence/default_proof_retry.log`: default A configuration-only dependency plan is **byte-identical**, SHA256 `0b19db389095f79a63ba0f8c2d218598dbb632b1544cb95ce8e32461cf03dad1`, 78 files / 23,470,919,287 bytes. Selections/server-spec serialized bytes and the worker shell script are identical. The historical plugin snapshot test also proves identical default served-action and decision/episode-log bytes with deterministic clocks. No A rollout outcomes were used. The initial proof invocation spelled the evidence directory absolutely instead of relatively; the corrected invocation uses the exact baseline arguments.

`cpu_replay.json` and `evidence/cpu_replay_final.log`: the real per-connection cache orchestrator, R10Recipe prefit, plugin strategy/judge/storage and reader passed on three selected held-out B inits, 30 decisions: **15 real looks, 15 blind decisions**. Served action bytes and live/library key bytes agree; all 16 neighbors, parent donor rows/weights, tokens and downsampled images are verified. No GPU, simulator, actual policy model, or remote operation was used for replay.

Reproduce from the repo root; use a fresh --basetemp on subsequent runs:

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
"${P[@]}" -m pytest -q -p no:cacheprovider --confcutdir=exp/offline_search \
  --basetemp=exp/offline_search/rounds/r12/prep/evidence/pytest_coordinator \
  exp/offline_search/closed_loop/test_look_recording.py \
  exp/offline_search/closed_loop/test_devset.py \
  exp/offline_search/closed_loop/ops/h100/test_h100.py \
  exp/offline_search/closed_loop/ops/h100/test_local.py \
  exp/offline_search/closed_loop/ops/h100/test_robustness.py
"${P[@]}" exp/offline_search/rounds/r12/prep/verify_prep.py default
"${P[@]}" exp/offline_search/rounds/r12/prep/verify_prep.py replay
```

**Exact coordinator commands for recording-enabled dev roots**

These commands are handed back, not executed here. They assume the existing R11 dev/runtime and frozen B files are already installed in the chosen worker island. Use the idle owned dev fleet. Fresh roots preserve the original fits and B selection and contain no inherited DONE/results. Substitute `r11_dev_current`, `r11_dev_size200`, or `r11_devknob_50` for SOURCE as needed. Size-50 is shown. Add `--tokens` and/or `--shadow` to prepare when requested; `--shadow` sets `full_model=true`. No fit is rebuilt.

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS OSCL_INIT_POOL OSCL_RECORD_LOOKS
SOURCE=/home/weiland/trace_runs/os_closed_loop/r11_dev_size50
RUN=/home/weiland/trace_runs/os_closed_loop/r12_devlooks_size50
"${P[@]}" -m exp.offline_search.closed_loop.look_recording prepare "$SOURCE" "$RUN" --images 56
mapfile -t ARMS < <("${P[@]}" -c 'import json,sys; print("\n".join(r["arm"] for r in json.load(open(sys.argv[1]))))' "$RUN/arms.json")

# Install only the five changed/new runtime files into the selected worker.
# Local serving reads the current workspace; no local server launch occurs here.
"${P[@]}" - <<'PYDEPLOY'
from pathlib import Path
from exp.offline_search.closed_loop.ops.h100 import control
pairs = [(Path(line.split(maxsplit=1)[1]), Path(line.split(maxsplit=1)[1]))
         for line in Path('exp/offline_search/rounds/r12/prep/NEW_RUNTIME_SOURCES.sha256').read_text().splitlines()]
control.bundle('timan107', control.island('timan107'), pairs)
PYDEPLOY

SERVER_HOST=local WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker
SERVER_HOST=local WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$RUN" "${ARMS[@]}"
SERVER_HOST=local WORKER_HOST=timan107 SYNC_PORT=23197 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$RUN" "${ARMS[@]}"
# Choose currently free public/forwarded ports; do not stop other listeners.
SERVER_HOST=local WORKER_HOST=timan107 PORTS=23110,23111 WPS=4 OMP=1 LOCAL_SERVER_CPUS=22-37,66-81 \
  MAX_ATTEMPTS=3 POLL_SECONDS=60 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$RUN" "${ARMS[@]}"
SERVER_HOST=local WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control collect "$RUN" "${ARMS[@]}"
```

For H100 serving, additionally install the same five files into its existing serving tree:

```bash
"${P[@]}" - <<'PYDEPLOY'
from pathlib import Path
from exp.offline_search.closed_loop.ops.h100 import control
pairs = [(Path(line.split(maxsplit=1)[1]), Path(line.split(maxsplit=1)[1]))
         for line in Path('exp/offline_search/rounds/r12/prep/NEW_RUNTIME_SOURCES.sha256').read_text().splitlines()]
control.bundle('h100', control.BASE / 'openpi', pairs)
PYDEPLOY
```

Then use `SERVER_HOST=h100` for plan/sync/chain/collect and H100 ports such as `PORTS=23220,23221`. Leave OSCL_MANIFEST unset so each arm uses its own B complement. The planner supplies and transfers recording contracts, and chain automatically stamps the verified B worker tasks. Local recordings are already under each server directory; H100 collect copies the whole arm subtree including looks. A new recording root must be synced after preparation; append-only manual contract flags are refused. No worker-entry or shell-driver change is required beyond the worker_pool hook above.

**Measured single-camera prices**

`bench_latency.py` reuses R7's π0.5 CameraRequest/Pi05StageBatcher method and the unchanged GR00T GrootStagedRunner serving cut. Batch size is one request; GR00T's full tower processes its two camera images together at tower batch 2, while a half look processes one at tower batch 1. π0.5 is eager SDPA, selected bfloat16 with float32 patch embeddings/norms, K10; GR00T is eager bfloat16 under runner.session(), flash_attention_2, K8. `PI05_STACK.json` and `GROOT_STACK.json` record versions/source hashes. The π0.5 raw JSON first-parameter dtype is float32; that field describes one parameter, not its mixed-precision vision/LLM compute.

There are **8 warmups + 40 timed repetitions per observation/path**, CUDA-synchronized wall time. π0.5 uses four B observations across L10/Spatial (160 samples/path); each GR00T checkpoint uses two (80/path); pooled GR00T has 160/path. Path order rotates by observation. CPU policy transforms are outside the timer, as in R7; stage preparation and prefix assembly/scatter are inside. Missing-camera completion encodes the other real camera from the same retained observation and updates its stage-1 prefix. Policy remainder is stages 2+3 with full stage-1 input.

Both policies use permitted π0.5 B-library RGB/prompt examples: GR00T B libraries have no RGB arrays. Stored 224 RGB is resized to 256 before the unchanged GR00T deployment crop/resize transform; zero physical robot state supplies the latency-only state. These are timing observations, not recorded GR00T rollout inputs or accuracy evidence. The first GR00T attempt discovered the missing RGB after loading, exited, and the retry rechecked memory and used the permitted B RGB source. Its original log is retained.

Owner vision/completion price = basis × median(path)/median(stock-full), using **.152** for π0.5 and **.148** for GR00T. π0.5's per-request full has its extra dummy-seeding/preparation cost; normalize to stock full to reproduce the brief's .1656/.0646/.0502 convention. Policy-remainder prices stay **.848/.852** on the owner basis; wall-time measurements do not refit that owner split. The R7 raw script's prose used a different per-request denominator; both raw denominators and the explicit normalization are retained here to remove ambiguity.

Each latency below is median ms (IQR width ms), followed by owner price. Q1/Q3 and every unrounded sample are in `pi05_latency.json`, `groot_l10_latency.json`, and `groot_spatial_latency.json`; `price_table.json` provides all pooled quartiles and prices. Checkpoint-specific prices are preferable to the pooled descriptive column.

| Path | π0.5 ms (IQR), price | GR00T L10 ms (IQR), price | GR00T Spatial ms (IQR), price | GR00T pooled ms (IQR), price |
| --- | ---: | ---: | ---: | ---: |
| stock_full | 68.484 (17.724), 0.152000 | — | — | — |
| full | 79.508 (18.391), 0.176468 | 35.898 (2.568), 0.148000 | 32.426 (0.728), 0.148000 | 33.994 (3.670), 0.148000 |
| wrist_only | 30.186 (7.178), 0.066998 | 36.451 (4.342), 0.150281 | 31.698 (0.603), 0.144681 | 32.703 (4.780), 0.142379 |
| third_only | 30.020 (7.277), 0.066629 | 34.717 (3.570), 0.143130 | 31.802 (0.715), 0.145153 | 32.675 (3.300), 0.142259 |
| completion_after_wrist | 23.062 (6.269), 0.051187 | 33.052 (4.001), 0.136265 | 30.314 (1.475), 0.138360 | 31.576 (4.445), 0.137475 |
| completion_after_third | 28.078 (5.800), 0.062318 | 32.424 (2.874), 0.133677 | 30.245 (1.431), 0.138049 | 31.094 (3.073), 0.135375 |
| policy_remainder | 414.731 (114.470), 0.848000 | 226.089 (12.922), 0.852000 | 203.417 (5.747), 0.852000 | 213.344 (23.188), 0.852000 |

π0.5 third-person-only is approximately the same price as wrist-only here. GR00T half looks do not halve the full-look cost: L10 wrist is slightly more expensive than full; other half paths save only about .003–.006 per look. A subsequent policy call needs the costly missing-camera completion, so the measured GR00T half+completion path exceeds a full look. Broad π0.5 IQRs are retained rather than hidden; no precision beyond these measurements is implied. The original default ledger prices were not changed.

Numerical checks: π0.5 wrist/third completion prefixes are byte-exact to full (max absolute delta 0 on all four observations). GR00T's two separately encoded cameras differ from its batched full encoding (max absolute embedding deltas: L10 1.1171875 and 4.25; Spatial .94140625 and .91796875). The GR00T and π0.5 third-only paths are benchmark-only helpers. This handback does not authorize or implement a GR00T half-camera serving mode; deployment needs its own numerical/accuracy gate.

Before each load, nvidia-smi reported **48,471 MiB free**. The bounded gate required 12,288+4,096 MiB for π0.5 or 10,240+4,096 MiB for GR00T, with a 600-second timeout/30-second retry interval. No memory wait was needed; each model ran in a separate process with an allocator cap at its stated peak budget. Measured peak reserved memory: π0.5 **7,352 MiB**, GR00T L10 **5,678 MiB**, GR00T Spatial **5,516 MiB**. The per-run `*.gpu_checks.json` and benchmark logs retain checks. `evidence/gpu_after.txt` shows only the pre-existing 30 MiB CUDA MPS server; no benchmark process remains. No process was killed.

Exact benchmark reproduction commands (sequential; each script independently performs the memory gate):

```bash
cd /home/weiland/projects/openpi
B=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 "$PWD/.venv/bin/python")
"${B[@]}" -B exp/offline_search/rounds/r12/prep/bench_latency.py --model pi05 --out exp/offline_search/rounds/r12/prep/pi05_recheck.json
G=/home/weiland/projects/openpi_ext/third_party/gr00t_n15
GB=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "PYTHONPATH=$G:$G/examples/Libero:.:src:packages/openpi-client/src" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 /home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python)
"${GB[@]}" -B exp/offline_search/rounds/r12/prep/bench_latency.py --model groot --suite l10 --out exp/offline_search/rounds/r12/prep/groot_l10_recheck.json
"${GB[@]}" -B exp/offline_search/rounds/r12/prep/bench_latency.py --model groot --suite spatial --out exp/offline_search/rounds/r12/prep/groot_spatial_recheck.json
"${P[@]}" exp/offline_search/rounds/r12/prep/price_report.py
```

**Source SHA256**

`NEW_RUNTIME_SOURCES.sha256` is the exact five-file deployment delta for controller/worker/H100; `H100_SOURCES.sha256` is the existing R11 serving closure refreshed with these files and the new recorder; `ALL_SOURCES.sha256` adds tests and prep scripts. No src/, model, checkpoint, R10Recipe, R11Knob, or shell-worker file was edited.

```text
0ff4cc181e4826bbde9295102fc27387537382788fbf205f02a4fa6399058d18  exp/offline_search/closed_loop/plugin.py
88faaf9a2dbf40528b553f1dd54ef119623200fa179284789ff288e427b81805  exp/offline_search/closed_loop/look_recording.py
05c17a0793cadc56afad191fb22d6279ecb94e4e1451be6e5279e2f8f89e8fac  exp/offline_search/closed_loop/worker_pool.py
624a22d60c7a7ac74da7eb18d3e010b2b29be5e44d4f2022de7ad31a47162ef7  exp/offline_search/closed_loop/ops/h100/assets.py
be5f1fa51f0f15014c916be3a08c8bcc377fa9a56c2f49bea171cee37d477d20  exp/offline_search/closed_loop/ops/h100/control.py
```
