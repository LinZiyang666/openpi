# Data Collection Guide

> **AGENT: READ FIRST** — This file is a registered subsystem rule document per [`WORKING_AGREEMENT.md` §8](../../WORKING_AGREEMENT.md#8-subsystem-rules). It carries Working Agreement authority.

## Overview

This data collection path records one HDF5 file per episode during remote inference.

Each file contains:

- Episode metadata: experiment name, task name, episode id, success flag, timestamp
- Per-step embeddings:
  - `vision_0`, `vision_1`, `vision_2`
  - `prompt_emb`
  - `robot_state`
  - `noise_action_*`
  - `clean_action`

The collector is designed as an outer wrapper around the normal policy path:

- `--collect` off: no collection wrapper, no behavior change
- `--collect` on but no active episode: pure delegation, no hooks
- active episode: temporary forward hooks are attached during each `infer()` call and removed immediately afterward

## Output Location

By default, collected files are written under the current working directory:

```bash
./data/<experiment_name>/episode_<episode_id>_<timestamp>.h5
```

For LIBERO, `experiment_name` is usually the task suite name, for example:

```bash
./exp/common/data/libero_spatial/episode_0007_20260331_035410_446588.h5
```

You can override the root directory with `--collect_dir`.

## Server Command

Start the policy server with collection enabled:

```bash
uv run scripts/serve_policy.py --collect --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Useful flags:

- `--collect`: enables per-episode HDF5 recording
- `--collect_dir ./data`: optional custom output root
- `--env LIBERO`: selects the LIBERO policy setup
- `policy:checkpoint`: load a specific checkpoint instead of the environment default
- `--policy.config pi05_libero`: choose the train/inference config
- `--policy.dir ...`: point to the PyTorch checkpoint directory

If you also enable staged cache timing:

```bash
uv run scripts/serve_policy.py --cache --collect --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

## What Gets Saved

Each HDF5 file stores one episode.

Top-level attributes:

- `experiment_name`
- `task`
- `episode_id`
- `num_steps`
- `timestamp`
- `success`

Top-level groups:

- `step_0000`
- `step_0001`
- ...
- `step_NNNN`

Each `step_xxxx` group contains:

- `vision_0`: `float16[256, 2048]`
- `vision_1`: `float16[256, 2048]`
- `vision_2`: `float16[256, 2048]`
- `prompt_emb`: `float16[num_lang_tokens, 2048]`
- `robot_state`: `float32[32]`
- `noise_action_1 ... noise_action_9`: `float32[action_horizon, action_dim]`
- `clean_action`: `float32[action_horizon, action_dim]`

For the released `pi05_libero` checkpoint, this usually means:

- `action_horizon = 10`
- `action_dim = 32`

So the action tensors are typically:

- `noise_action_*`: `(10, 32)`
- `clean_action`: `(10, 32)`

## Minimal Simulator Changes

Your simulator client must send two lifecycle control messages:

1. episode start
2. episode end

The two pieces of metadata that matter most are:

- task name
- episode id

### Task Name

Use a stable human-readable task string. For LIBERO this is usually `task_description`.

Example:

```python
task=str(task_description)
```

This value is stored in the HDF5 file attribute:

```text
attrs["task"]
```

### Episode ID

Use a globally increasing episode counter, not a task-local counter that resets to zero for every task.

Good:

```python
global_episode_id = 0
...
client.episode_start(..., episode_id=global_episode_id)
...
global_episode_id += 1
```

Bad:

```python
client.episode_start(..., episode_id=episode_idx)
```

If `episode_idx` resets inside each task loop, your saved files will repeatedly show `episode_id=0`, `1`, etc. across different tasks, which makes downstream analysis harder.

### Example Client-Side Calls

At the start of each episode:

```python
client.episode_start(
    experiment=args.task_suite_name,
    task=str(task_description),
    episode_id=global_episode_id,
)
```

At the end of each episode:

```python
client.episode_end(success=done)
global_episode_id += 1
```

## Recommended LIBERO Pattern

Inside `examples/libero/main.py`, the intended pattern is:

1. Create one websocket client
2. Maintain one global episode counter across the full evaluation run
3. Before each rollout:
   - send `episode_start(...)`
4. After each rollout:
   - send `episode_end(success=done)`
   - increment the global episode counter

Suggested values:

- `experiment=args.task_suite_name`
- `task=str(task_description)`
- `episode_id=global_episode_id`

## Concise Data Inspection

List saved files:

```bash
find data -name '*.h5'
```

Inspect the newest file:

```bash
python - <<'PY'
import glob
import h5py
paths = sorted(glob.glob('exp/common/data/libero_spatial/*.h5'))
print("latest:", paths[-1])
with h5py.File(paths[-1], 'r') as f:
    print("attrs:", dict(f.attrs))
    print("groups:", list(f.keys())[:5])
    step0 = f['step_0000']
    for key in step0.keys():
        print(key, step0[key].shape, step0[key].dtype)
PY
```

## Important Notes

- After you enable `--collect`, the first real simulator inference may trigger model compilation. This is especially noticeable when `--cache` is also enabled, but the first compiled inference can also happen on the normal PyTorch path.
- During that first compile, the server may look completely stuck for a long time and may not print new progress messages. This is normal. Do not assume it crashed immediately.
- In practice, the most common symptom is: the simulator sends the first episode, and then the server appears to hang before returning the first action chunk.
- The correct response is usually to wait patiently.
- If you want a quick reality check, open another terminal and run `top` or `htop`. If the Python process is still using a lot of CPU, it is often still compiling rather than deadlocked.
- The collector writes one file per episode, not one file per task suite.
- Disconnect during an episode will flush partial data on connection close.
- File names include timestamps, so repeated runs do not overwrite earlier files.

## Gate-research per-step collection (distinct from `--collect`)

The `--collect` mode above uses forward hooks to capture deep model internals and
is **single-connection / single-replica only**. For GATE ("search or not")
research there is a separate, **concurrency-native** collector that records a
lean per-step row — the model input the cache actually keys on (`robot_state` by
default), the verdict (`hit_type` / `cp1_score` / `winner_id` / `start_t`), the
`searched` flag, and the episode `success`. It reuses the `__hit_meta__` wire
channel plus a `__collect_meta__` sibling key and per-connection recorder, so it
works under concurrent serving and (for `robot_state`) under the cross-machine
conductor. The two collectors are **mutually exclusive** (`serve_policy` fails
fast if both are on); use whichever the task needs.

Design & rationale: [`logs/gate_data_collection_plan.log.md`](../../logs/gate_data_collection_plan.log.md).

### 1. Enable it on the server

Off by default → when off, the response wire is **byte-identical** to a normal
serve. Turn it on either in the cache YAML or via CLI (both feed one
`CacheConfig.collection` block):

```yaml
# cache config YAML
collection:
  export_collect_meta: true          # attach __collect_meta__ to each response
  collect_fields: [robot_state]       # default; vision_*/prompt_emb opt-in (see §4)
  wire_frame_cap_kib: 32              # per-step encoded field-byte ceiling
```

```bash
# equivalent CLI override (tri-state: unset = use YAML, set = override YAML)
uv run scripts/serve_policy.py ... \
    --export-collect-meta \
    --collect-fields robot_state
```

Hard requirement (validated at startup, else fail-fast): the CP1 gate must be
`always_search`, so no step is gate-skipped and the collected labels are not
selection-biased. The `collect_fields` encoded byte size must also stay under
`wire_frame_cap_kib`.

### 2. Run the client (standalone LIBERO)

```bash
uv run examples/libero/main.py \
    --host <server> --port <port> \
    --task-suite-name libero_spatial \
    --yaml-id my_cfg \
    --num-trials-per-task 10 \
    --collect-gate-dir out/gate/my_cfg \
    --collect-embeddings none          # 'none' = robot_state only; 'pooled' allows vision
```

- `--collect-gate-dir <dir>` is the **canonical** flag (enables the recorder;
  requires `--yaml-id`). The old `--per-step-log-dir` is a deprecated alias kept
  for backward compatibility.
- `--collect-embeddings` must match the server `collect_fields`: `none` for
  `robot_state`-only, `pooled` when vision embeddings are collected.
- `--num-trials-per-task N` defines the canonical global `episode_id` mapping
  (`task_id * N + episode_idx`); keep it identical to any conductor run you want
  to join against.

Rows are written as JSONL, one file per `yaml_id`, merged across workers.

### 3. Output schema

Two row kinds share the file, distinguished by `_kind`:

**Per-step verdict row** (one per CP1 inference step):

```json
{"yaml_id": "my_cfg", "task_id": 3, "subset_init_state_idx": 5,
 "episode_id": 35, "task_uid": "my_cfg:eval:3:5", "phase": "eval",
 "step_idx": 0, "hit_type": "MISS", "start_t": null, "winner_id": null,
 "cp1_score": 0.81, "searched": true, "collector_schema_version": 1,
 "collect": {"robot_state": [0.12, -0.03, ...]}, "success": true}
```

**Per-episode `episode_summary` row** (one per episode; provenance not derivable
from step rows — `seed`, `kb_id`, `searched_all`):

```json
{"_kind": "episode_summary", "task_uid": "my_cfg:eval:3:5", "episode_id": 35,
 "phase": "eval", "seed": 42, "num_steps": 120, "success": true,
 "searched_all": true, "collect_fields": ["robot_state"], "kb_id": "cp1_mean_pool",
 "collector_schema_version": 1}
```

Model-input arrays arrive as `np.ndarray` over the wire (`robot_state` float32,
`vision_*`/`prompt_emb` float16 to halve frame bytes) and are upcast to float32
and converted to plain lists at the client boundary (conductor `msgpack.packb`
and JSONL `json.dumps` cannot encode ndarrays; the upcast is lossless).

### 4. Fields, scale, and conductor

- `robot_state` (~128 B/step) works **everywhere**, including the cross-machine
  conductor (no NFS — it rides the existing `per_step_rows` central return over
  the wire).
- `vision_*` / `prompt_emb` are large and **standalone-only**; requesting them
  under the conductor fails fast (a per-episode `EpisodeResult` frame is capped
  at 64 MiB). `raw prefix_embs` is out of scope.
- **Conductor producer contract**: a strategy whose episodes run through
  `LiberoEpisodeRunner` must stamp `EpisodeTask.extra["num_trials_per_task"]`
  with the stage's per-phase trial count (warmup and eval differ). The runner
  derives the canonical `episode_id` from it and fails fast if absent — it never
  reads the worker's unrelated default. Both in-repo strategies already do this.

### 5. Offline analysis

`openpi.serving.per_step_recorder.summarize_gate_log(gate_dir, yaml_id)` tallies
eval-phase verdicts from a collected JSONL:

```python
from openpi.serving.per_step_recorder import summarize_gate_log
counts = summarize_gate_log("out/gate/my_cfg", "my_cfg")
# {"n_eval_verdicts": N, "n_full_hit": .., "n_warm_start": .., "n_miss": ..}
# invariant: n_eval_verdicts == n_full_hit + n_warm_start + n_miss
```

It counts only real verdict rows (`hit_type ∈ {FULL_HIT, WARM_START, MISS}`), so
the `episode_summary` provenance row never inflates the inference-ratio
denominator. `searched=False` rows are already dropped at write time in gate
mode, so the file carries no gate-skipped steps.

## RoboCasa365 teacher-library collection (conductor topology)

> Added with the framework re-integration (plan: `logs/robocasa365_framework_integration.log.md`).
> Formal (paper-grade) collection is gated on the unified temporary G2 approval.

### Hard topology constraints

`--collect` is structurally incompatible with concurrency: the embedding
collector attaches module-global forward hooks, so `serve_policy.py` enforces
`--non-concurrent --replicas 1`, and a non-concurrent server rejects a second
connection outright (close code 1013). The collection topology is therefore
**one server process ↔ one connection ↔ one worker**, scaled horizontally by
launching N server processes; the conductor driver's control connection is
replaced with a socket-free no-op ctl so it cannot consume the only slot.
One `run_collect.py` invocation serves exactly ONE teacher (core
`assign_servers` has no model-type notion); the per-teacher endpoint groups
live in the env-config file and are validated before the graph is built.

### Server (pi0.5 example)

```bash
uv run scripts/serve_policy.py \
  --port 8010 --non-concurrent \
  --collect --collect_dir /data/robocasa365_cache/build_l1s1 \
  policy:checkpoint \
  --policy.config pi05_robocasa \
  --policy.dir /home/weiland/ckpt_pi05_robocasa_pytorch
```

`--collect_dir` is the **scene root** (`build_l{L}s{S}`), never the teacher
root: the collector inserts an `<experiment>` (= teacher id) directory level
itself. Final layout:
`<scene-root>/<teacher>/<TaskName>/episode_NNNN_aAA.h5`.

### Driver + workers

```bash
uv run python exp/robocasa365/run_collect.py \
  --role all --teacher pi05 \
  --servers 127.0.0.1:8010 \
  --tasks OpenCabinet:256,CloseDrawer:126 \
  --layout 1 --style 1 --base-seed 0 \
  --collect-root /data/robocasa365_cache/build_l1s1 \
  --env-config exp/robocasa365/config/collect_weilandserver.env \
  --connect-deadline-s 60 --episode-deadline-s 900 --terminate-grace-s 30
```

Before dispatch the exact TaskGraph is frozen into an immutable per-batch
run-plan JSON (`run_plan_<runid>_bNN.json`, containing every expected
`task_uid` plus a `plan_hash`); resumes recompute the hash and refuse to start
if parameters changed. Seeds are `base_seed + episode_idx`; this makes the
*initial state* reproducible, while the rollout itself stays stochastic
(same initial state, fresh flow-matching noise on retry — retries never
overwrite, they write a new `_aAA` attempt file).

### Pinned-object task variants

By default every episode resamples which mesh instance fills each object slot,
so a task's "initial state" is reproducible in pose but not in *identity*. A run
can instead pin every slot to one exact mesh by passing a pin table:

```bash
  --pinned-objects exp/robocasa365/config/pnp_pinned_objects.json
```

The table is `{task: {slot: "objects/<...>/model.xml"}}` plus its own `pin_id`
(sha256 over canonical JSON, domain-separated). Paths are asset-relative because
the three collection hosts keep their asset trees at different prefixes;
robocasa rebases them onto the local `assets_root`.

Two things are worth knowing before using it:

* **The flag must be passed to every driver in the run**, including
  `run_ws_search.py` for a teacher-only arm. A driver without it dispatches
  unpinned episodes that look identical in the journal.
* **Pinning bypasses robocasa's own object filters.** The exact-path branch of
  `sample_kitchen_object_helper` skips `exclude_obj_groups`, the seven
  capability flags and the `obj_instance_split` slice, so a hand-written table
  can seat a pretrain-split or non-graspable object with no error anywhere.
  Generate tables with `select_pinned_objects.py`, which replays those filters
  and only picks from the legal set.

Identity travels with each episode (`pin_id`, `pin_task_id`, and the task's slot
map) and the worker re-reads the table from disk to check both against what it
was dispatched. After `reset()` the episode also records `realized_objects` —
what the scene *actually* built — into its HDF5 attrs. That is what the auditor
judges on: an override that was accepted but never applied produces a correct
`pin_id` and wrong realized objects, and is refused admission.

### Audit + manifest (blocking before any library build)

HDF5 write failures are swallowed server-side (the journal still records the
episode as done), so the auditor is mandatory, and the run-plan is its only
source of the expected-UID set:

```bash
uv run python exp/robocasa365/verify_collection_artifacts.py \
  --root /data/robocasa365_cache/build_l1s1 --teacher pi05 \
  --journal exp/robocasa365/data/journal_collect_l1s1_pi05.jsonl \
  --run-plan exp/robocasa365/data/run_plan_collect_l1s1_pi05_b01.json \
  --target 20 --manifest-out exp/robocasa365/data/manifest_l1s1_pi05.json
```

A journal record is admitted iff `accepted && success && error is None`;
run-plan uids with no journal row at all are reported as `missing_terminal`
(retry exhaustion leaves zero rows — only the run-plan can see it). Library
builds consume the manifest (first `--target` successes per task by
`episode_idx`, sha256-pinned), never directory listings. Per-task episode
counts come from `min_episodes_for_target(sr)` — the smallest N with
`P(Binom(N, sr) ≥ 20) ≥ 0.90` at the SR point estimate.
