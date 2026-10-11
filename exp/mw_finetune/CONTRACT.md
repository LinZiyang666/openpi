# MT50 serving and evaluation contract (A, 2026-10-08)

## Model entry points

`serve.py --model groot|fastwam|cosmos|pi05 --checkpoint DIR --port 8050`
imports `exp.mw_finetune.<model>.policy` and calls
`load_policy(checkpoint: str, device: str, **kw)`. B/C/D own their corresponding
`policy.py`. A owns `exp.mw_finetune.pi05`. No package-level export is required.
`device` defaults to `cuda:0`. `--denoising-steps N` supplies
`kw["denoising_steps"] = N`; when omitted, the wrapper chooses its documented
paper default. `--policy-option KEY=JSON` supplies additional keyword arguments.

The result implements:

```python
class Policy:
    def reset(self) -> None: ...
    def infer(self, obs: dict) -> dict: ...
    # Optional; JSON-safe, stable across server restarts:
    def get_metadata(self) -> dict: ...
```

`load_policy` runs once. All clients share this instance; GPU calls and resets
are serialized by one process-level lock. A calls `reset()` immediately before
each `infer()` so each response is a complete prediction from the current
observation. The wrapper must return a full action chunk, never a queued single
action, and must run pure model inference without retrieval, warm-start state,
cross-request KV reuse, future-imagination planning, or online adaptation.
Ephemeral computation within one inference is permitted. There is no mutable
history shared between episodes or clients.

## Canonical wrapper input/output

```python
obs = {
    "image": np.ndarray,       # uint8, HWC (480,480,3), RGB
    "state": np.ndarray,       # float32 (4,), obs[:4]: EEF x,y,z,gripper
    "prompt": str,             # exp.metaworld.tasks.PROMPTS[task_name]
    "task_name": str,          # canonical MT50 name, e.g. reach-v3
    "seed": int,               # independent policy RNG seed for this decision
    "decision_index": int,     # 0,1,... within episode
}
result = {"actions": np.ndarray}  # float32 (T,4), T >= 5, finite
```

Extra output keys are optional and must be msgpack compatible. A validates and
serves `actions`. Channel order is x,y,z,gripper, exactly the dataset's `action`.
These are MetaWorld Cartesian control commands, not target positions. Do not
scale EEF commands, invert/binarize the gripper, or apply LIBERO transforms.
The training dataset can contain commands outside [-1,1]; return physical
denormalized commands unchanged and let MetaWorld perform its own clipping.
Each wrapper owns its training/checkpoint normalization, image resizing, and
language encoding; A supplies the unresized render and raw state.

`seed = uint32_le(SHA256("mt50:7:<task_name>:<idx>:<decision_index>")[:4])`.
A seeds NumPy/Python/Torch under the model lock. Wrappers with a private
generator must explicitly use `obs["seed"]`; report any different RNG behavior
in metadata. This keeps episode randomness independent of worker scheduling.

## Wire protocol and workers

The server uses the existing `WebsocketPolicyServer` and `msgpack_numpy`
protocol, including initial metadata and `select_bundle(default)`,
`episode_start`, and `episode_end`. Workers continue to send
`observation/image`, `observation/state`, `prompt`; A translates these to the
canonical wrapper input. Episode controls carry task name/index; controls never
become policy observations. Only the default bundle is accepted.

Formal drivers/workers run on timan nodes in the new
`/scratch/zixuans8/mw_r17/` tree, using uv-managed Python 3.11.15 and a matching
MetaWorld 3.0.0/Gymnasium 1.3.0/MuJoCo 3.14.0 environment. Physics runs on CPUs;
**NVIDIA EGL is the renderer for every formal lane**. EGL device indices are
assigned round-robin across all node GPUs. No timan node may run a policy server,
including a zero policy. The evaluator enforces the lane and a30-worker limit.

| Policy server | Driver and workers | Direct client endpoint |
|---|---|---|
| weilandserver (4090) | timan107 | ziyanglin.com:23150 pi05;23151 groot;23152 fastwam;23153 cosmos |
| h100 (H100) | timan108 | 149.165.153.233:23250 pi05;23251 groot;23252 fastwam;23253 cosmos |

Servers listen directly on those ports, 1:1 NAT; no tether expose/broker hop.
Check `ss -tlnp` before launching.23159/23259 are reserved for CPU zero-policy
capacity probes;23180 belongs to an existing rsync service and is untouched.
The harness reuses `exp.metaworld.episode_runner.MetaworldEpisodeRunner` and
`exp.metaworld.env` unchanged. A local env shim serializes environment creation,
first render/context creation and destruction under the shared cross-process
`/tmp/rc365_env_build.lock` (`RC365_ENV_BUILD_LOCK` cannot be disabled).
Each episode records its worker hostname/PID, EGL device index, GL_RENDERER,
GL_VENDOR and GL_VERSION; `worker_metadata.json` collects the worker census.

The driver fsyncs its journal to timan-local disk. `sync_back.py` acquires its
run lock after the driver exits, creates a SHA256 snapshot and transfers it
through tether pull to weilandserver `/data/mw_finetune/eval/<run>`.
The original timan run remains resumable. Tether archives are staged only in
the new `/tmp/mw_r17_A/` tree because transfer allow_roots exclude /scratch.

Local OSMesa CPU workers remain an explicit fallback using
`/home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python` and private
`/data/mw_finetune/envs/osmesa/`; renderer provenance distinguishes these runs.
See HANDBACK_A and `/data/mw_finetune/reports/A_camera_parity/` for measurements
and dataset camera verification. Model wrappers and the obs/action contract
are unchanged by the worker topology update.

Formal protocol: all 50 tasks, `MT1(task, seed=7)` plus
`set_task(train_tasks[idx])`, idx 0..9, **500 episodes**. 15 zero-action settle
steps; at most 160 subsequent policy steps; execute the first 5 actions of each
prediction before replanning. Render corner2/id 2 at (0.75,0.075,0.7), 480²,
rotate 180° as the existing env does. Any step with info.success is binary
success. Difficulty membership is the existing table: 28/11/6/5.
`--smoke` runs reach-v3 idx 0 and 1. `--replan-steps 1` is an explicitly
labeled alternative, not the historical π0.5 protocol. The SmolVLA paper's
simulation inference replans after each action; our default 5 is the requested
historical comparison convention and must be disclosed in reports.

Completed episodes are journaled and fsynced centrally; rerunning an identical
command skips completed identities and retries errors. A validates the server
metadata/checkpoint and protocol on resume. Errors remain missing episodes,
never success-rate failures. Aggregation reports per-task SR, group SR,
overall per-task mean, and separately equal-group mean. A formal report requires
all 500 completed identities; partial reports are marked incomplete.

## Shared resource coordination

Every GPU command, including serving and smoke/probe runs, must hold
`flock /tmp/r17_gpu.lock <command>` on that host; inspect `nvidia-smi` before
starting. The coordinator owns long jobs. B/C/D own model recipe/training docs.
H100 `/data/mw_finetune` has a 120 GB total budget across agents with a 30 GB
free-space floor; existing Wan files remain outside it and are read-only.
No agent deletes anything without coordinator approval.
