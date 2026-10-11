# A: MT50 serving and evaluation handback

A owns the common server, π0.5 wrapper, evaluator, renderer lifecycle shim,
aggregation and worker deployment. B/C/D own their model directories and recipes.
The published model API is [CONTRACT.md](CONTRACT.md). Existing MetaWorld,
R16/offline-search code and results are unchanged. The coordinator runs formal
training/evaluation; A ran only bounded diagnostics.

## Formal lanes and ports

| Server | Driver/workers | Client address | π0.5 | GR00T | Fast-WAM | Cosmos | CPU zero |
|---|---|---|---:|---:|---:|---:|---:|
| weilandserver /4090 | timan107 | ziyanglin.com |23150|23151|23152|23153|23159|
| h100 /H100 | timan108 |149.165.153.233|23250|23251|23252|23253|23259|

These are direct listening ports in the allowed 1:1 NAT ranges. No broker
exposure is needed. Check `ss -tlnp` before starting;23180 is an existing rsync
service and is untouched. **Timan nodes are workers only, never policy servers**
(devices.md§1); the server rejects timan hostnames. The timan evaluator checks
its assigned host/port lane and rejects more than30 workers.

Formal renderer is **NVIDIA EGL on both timan nodes**. Physics is CPU MuJoCo.
Workers spread EGL contexts round-robin across8GTX1080s on timan107 or4A5000s
on timan108. `rendering.py` serializes env creation, first render/context creation
and destruction through `/tmp/rc365_env_build.lock`, sharing the documented
RC365 lock with existing fleets. It refuses a disabled lock or non-NVIDIA EGL.
Every episode records worker hostname/PID, EGL device index, GL_RENDERER,
GL_VENDOR and GL_VERSION; `worker_metadata.json` collects this census.

## Install/sync workers (rerunnable)

Both new trees are installed at `/scratch/zixuans8/mw_r17/`; timan107's scratch
shares its root filesystem (~26GB free before setup), timan108 scratch has9.1TB
free. Each complete tree uses~0.8 GB. No existing openpi/simulator directory or
other user's tree is modified. Managed Python is3.11.15; versions match local:
MetaWorld 3.0.0, Gymnasium 1.3.0, MuJoCo 3.14.0, NumPy 1.26.4, msgpack 1.2.2,
websockets 17.1, PyOpenGL 3.1.10, glfw 2.10.2, Pillow 12.3.0, imageio 2.37.4.
The package list is retained at each tree's `reports/sim_freeze.txt`.

```bash
cd /home/weiland/projects/openpi
.venv/bin/python -m exp.mw_finetune.sync_workers timan107 timan108 --setup
# Code-only refresh (between invocations):
.venv/bin/python -m exp.mw_finetune.sync_workers timan107 timan108
.venv/bin/python -m pytest -q exp/mw_finetune/tests
```

`sync_workers.py` sends exp/metaworld, the common worker modules from
exp/mw_finetune, and packages/openpi-client. It additionally includes only
conductor task.py/protocol.py/worker.py, required by the reused EpisodeRunner,
with a generated empty conductor namespace initializer in the snapshot. It
sends no model wrappers, weights, datasets, venvs or training code. Archives
are SHA256 checked before extraction. Tether allow_roots exclude /scratch, so
small archives stage in the new `/tmp/mw_r17_A/` tree. Existing agent settings
are unchanged. Temporary `too_many_in_flight` transfer saturation is retried
with fresh IDs for up to~122seconds; rerun if another large transfer continues.

## Start a server on weilandserver

Each command holds the shared GPU lock (the B/D scripts already take it).
Set each checkpoint to the immutable final directory chosen in its handback.
Check free GPU memory and listening ports first. Run one server at a time.

```bash
cd /home/weiland/projects/openpi
export PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
ss -tlnp
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# π0.5 RLinf:
flock /tmp/r17_gpu.lock .venv/bin/python -m exp.mw_finetune.serve \
  --model pi05 --checkpoint /data/ckpt/pi05_metaworld_rlinf --port 23150

# GR00T; choose the actual final checkpoint from B's run:
GROOT_MW_CHECKPOINT=/data/mw_finetune/groot/4090_mt50/checkpoint-20000
bash exp/mw_finetune/groot/serve_local.sh \
  --checkpoint "$GROOT_MW_CHECKPOINT" --port 23151

# Fast-WAM; replace the checkpoint path with C's chosen trained delta:
FASTWAM_MW_CHECKPOINT=/data/mw_finetune/fastwam/TRAIN_RUN/checkpoint-STEP
flock /tmp/r17_gpu.lock /data/mw_finetune/envs/fastwam/bin/python \
  -m exp.mw_finetune.serve --model fastwam --checkpoint "$FASTWAM_MW_CHECKPOINT" \
  --port 23152 --policy-option 'base="/data/mw_finetune/models/fastwam_libero"' \
  --policy-option 'wan_root="/data/xwam/wan22_5b"'

# Cosmos community checkpoint; use D's selected reconstruction/options:
bash exp/mw_finetune/cosmos/launch.sh serve \
  --checkpoint /data/mw_finetune/models/cosmos_metaworld --port 23153
```

Omitting `--denoising-steps` keeps each wrapper's default. Additional wrapper
options use `--policy-option 'name=JSON'`; metadata captures all options,
checkpoint identity and the wrapper's stable metadata. Cosmos has unpublished
runner/layout assumptions: include D's chosen reconstruction in its model-id
and report those limitations; see D's handback.

## Start a server on H100

B's island and A's common code are present under /home/exouser/openpi.
C's island is /data/mw_finetune/envs/fastwam. Use a remote terminal/tmux for a
long serving session; A does not launch formal jobs. Examples:

```bash
# Check ports and GPU from weilandserver:
tether exec --safe --timeout 30s h100 -- /usr/bin/ss -tlnp
tether exec --safe --timeout 30s h100 -- nvidia-smi

# GR00T (set final checkpoint using B's handback):
tether exec --timeout 24h h100 -- env GR00T_HARDWARE=h100 bash \
  /home/exouser/openpi/exp/mw_finetune/groot/serve_local.sh \
  --checkpoint /data/mw_finetune/groot/h100_mt50/checkpoint-20000 --port 23251

# Fast-WAM (replace TRAIN_RUN/checkpoint-STEP):
tether exec --timeout 24h h100 -- env \
  PYTHONPATH=/home/exouser/openpi:/home/exouser/openpi/src:/home/exouser/openpi/packages/openpi-client/src \
  OMP_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  flock /tmp/r17_gpu.lock /data/mw_finetune/envs/fastwam/bin/python \
  -m exp.mw_finetune.serve --model fastwam \
  --checkpoint /data/mw_finetune/fastwam/TRAIN_RUN/checkpoint-STEP --port 23252 \
  --policy-option 'base="/data/mw_finetune/models/fastwam_libero"' \
  --policy-option 'wan_root="/data/xwam/wan22_5b"'
```

The same serve module supports π0.5/Cosmos on23250/23253 once their island and
checkpoint are explicitly provisioned on H100; A did not duplicate them there.
The model's training launcher provisions its H100 code/dependencies. Tether
exec timeout bounds the CLI wait, not remote process lifetime; stop the owning
server explicitly when finished. Do not stop other sessions' processes.

## Start formal driver/workers, sync back, aggregate

Pick a unique run-name/model-id for each immutable checkpoint and inference
configuration. All four models use these commands; change `--model` to
groot/fastwam/cosmos and the run-name. Lane ports are selected automatically.
The driver and run directory live on the timan node. Session names are
`r17_pi05`, `r17_groot`, `r17_fastwam`, `r17_cosmos`; an existing session causes
an explicit failure so another driver cannot overwrite it.

```bash
# 4090 lane, after starting pi05 on23150:
tether exec timan107 -- env \
  PYTHONPATH=/scratch/zixuans8/mw_r17/code:/scratch/zixuans8/mw_r17/code/src:/scratch/zixuans8/mw_r17/code/packages/openpi-client/src \
  /scratch/zixuans8/mw_r17/envs/sim/bin/python -m exp.mw_finetune.worker_launch \
  --model pi05 --run-name pi05_rlinf_mt50_egl --workers 12

# H100 lane, after starting fastwam on23252:
tether exec timan108 -- env \
  PYTHONPATH=/scratch/zixuans8/mw_r17/code:/scratch/zixuans8/mw_r17/code/src:/scratch/zixuans8/mw_r17/code/packages/openpi-client/src \
  /scratch/zixuans8/mw_r17/envs/sim/bin/python -m exp.mw_finetune.worker_launch \
  --model fastwam --run-name fastwam_mt50_egl --workers 12

# Monitor one driver:
tether exec timan107 -- tail -n 20 /scratch/zixuans8/mw_r17/runs/pi05_rlinf_mt50_egl/driver.log

# After driver exit; copy to weilandserver and verify every file:
.venv/bin/python -m exp.mw_finetune.sync_back timan107 pi05_rlinf_mt50_egl
.venv/bin/python -m exp.mw_finetune.aggregate /data/mw_finetune/eval/pi05_rlinf_mt50_egl
.venv/bin/python -m exp.mw_finetune.sync_back timan108 fastwam_mt50_egl
.venv/bin/python -m exp.mw_finetune.aggregate /data/mw_finetune/eval/fastwam_mt50_egl
```

Use `--smoke --workers 2` for two reach episodes, or add
`--tasks assembly-v3 reach-v3 --trials 1` for a selected diagnostic. Use
`--foreground` to keep the driver in tether exec instead of tmux. Resume with
the identical worker-launch command after the previous invocation exits;
completed task/idx identities are skipped and errors retried. Worker count
may change between invocations (maximum30). Run manifests require unchanged
checkpoint/server options, simulator versions and env protocol. Torn final
JSONL records are backed up before recovery; malformed interior rows fail.
Journals are fsynced per episode and manifests/summaries replaced atomically.

`sync_back.py` refuses an active driver, snapshots the timan run without
changing its journal, checks the transferred archive SHA256 and every file's
SHA256, and atomically installs files locally under /data/mw_finetune/eval.
It preserves the original timan directory and local run lock, so remote resume
and later sync still work. SHA256SUM-like data are in TRANSFER_SHA256.json;
archive verification records are in /data/mw_finetune/reports/A_transfers/.
All five capacity runs and the real π0.5 lane smoke have passed complete-file
SHA256 rechecks; reports/A_lane_validation.json collects the measured results.
Errors never count as failures. A full result requires all 500 identities.

Formal benchmark: MT1(task,seed7)+set_task(train_tasks[idx]), all 50tasks,
idx0..9;15 settle zero actions then at most160 policy steps; first 5 actions of
each fresh chunk; success at any step. The image is corner2 at
(0.75,0.075,0.7),480²,180° rotation. State is raw obs[:4]; action order xyz,g
with no extra scaling/clipping/gripper transformation. MetaWorld does clipping.
Per-task and difficulty SR are fractions; overall_per_task_mean weights all 50
tasks equally, while equal_group_mean separately weights the28/11/6/5groups
equally. `--replan-steps 1` is a labeled protocol alternative, unlike the
historical π0.5 comparison's5. Server reset before every request and independent
SHA256(task,idx,decision) RNG prohibit cross-episode state/history.

## Capacity probes and validation

Each capacity run used60 episodes (six tasks,10trials each),160 policy steps,
32 requests per episode, full480² raw msgpack images, a CPU zero server and EGL.
Rates include spawned-worker startup, MT1 construction, context lifecycle lock,
network and fsync. All five runs completed with zero errors and are explicitly
marked test_policy/capacity_probe, never formal success rates.

|Lane|Workers|Seconds|Episodes/min|
|---|---:|---:|---:|
|timan107 → weilandserver:23159|12|50.388|71.446|
|timan107 → weilandserver:23159|24|48.167|74.740|
|timan107 → weilandserver:23159|30|48.970|73.515|
|timan108 → H100:23259|12|28.574|125.990|
|timan108 → H100:23259|24|30.180|119.286|

Start at12 workers; increase to24 only if real-model inference benefits. These
zero-policy rates describe simulator/network capacity, not model throughput.
The worker/server nodes also had other authorized setup/probes running.
Artifacts are /data/mw_finetune/eval/A_capacity{107,108}_w{12,24,30}/, with 30
only tested on107. Per-worker GL strings and timing are retained.

To repeat: run a CPU server on weilandserver with
`CUDA_VISIBLE_DEVICES='' .venv/bin/python -m exp.mw_finetune.serve --model zero --port 23159`
(or H10023259 with its root venv, same PYTHONPATH as above). Run worker_launch
on the assigned timan with `--model zero --foreground --capacity-probe`
and `--tasks reach-v3 push-v3 button-press-v3 door-open-v3 drawer-open-v3 window-open-v3 --trials 10`.
Use fresh run-names and workers12/24/30. The zero server needs no GPU or GPU lock.
Stop its owning process after probes.

A CPU suite:24 passed, including schema/action conventions, table/aggregation,
server seeding/reset, torn-journal recovery, renderer lifecycle locking and
SHA256 snapshot/resume checks. C's seven CPU tests also pass in its pinned
Fast-WAM venv; root venv lacks lerobot.policies for its causal-equivalence test.
Timan107 zero two-episode smoke passed; replay skipped completed identities
without duplicates. The resume evidence is reports/A_remote_resume.json;
the source journal SHA256 stayed equal to the copied two-episode journal.
Real π0.5 timan107→4090 smoke completed2 episodes with
0 errors and0/2 success. GPU lock hold was124.961 s including89.886 s server
startup; rollout wall33.629 s. Both reached160 steps/32 decisions. Worker0/1
reported NVIDIA GTX1080 EGL, driver535.183.01. The immutable checkpoint and
options are in /data/mw_finetune/eval/A_pi05_lane107_smoke/manifest.json;
SHA256 archive/file verification passed. Server and worker subprocesses
were stopped by their owning supervisor. Both temporary CPU zero servers
were stopped after the completed capacity probes; no formal job was launched.
Older local π0.5 smoke:2 episodes,0 errors,0/2 success,241.11 s including startup.
A two-episode smoke is a mechanical check, not a quality estimate.

Bounded real lane smoke (waits for the GPU lock, including model startup in
its<300s GPU hold; only own subprocess groups are stopped):

```bash
.venv/bin/python -m exp.mw_finetune.remote_smoke \
  --checkpoint /data/ckpt/pi05_metaworld_rlinf --run-name pi05_lane_smoke_example
```

## Camera parity and local fallback

Evidence: /data/mw_finetune/reports/A_camera_parity/{REPORT.md,pixel_stats.json,
comparison_<task>.png}. Five tasks compare dataset first frames to reset renders:
assembly,basketball,button-press,push,reach. The background is exactly equal;
textured tabletop channels match99.9966% (max 1/255). Correct orientation whole
MAE2.06..5.17/255 vs raw/unrotated52.87..55.29/255. Robot-body MAE3.44..8.35
reflects differing dataset reset poses and is retained in the report. Camera,
orientation and480² resolution match the training dataset convention.
Same benchmark resets are bitwise equal across all three environments.
NVIDIA EGL107/EGL108 whole MAE0.000440..0.000582/255,max1; OSMesa/EGL
MAE0.886..0.993/255,p995,max61..62. Formal comparisons use NVIDIA EGL.

Reproduce capture on each timan:
`.../envs/sim/bin/python -m exp.mw_finetune.camera_parity capture --renderer egl --out .../reports/egl_<node>`
with the worker PYTHONPATH above. On local use the OSMesa environment below.
Pull staged capture tar files via tether, then run local root Python
`-m exp.mw_finetune.camera_parity compare --out /data/mw_finetune/reports/A_camera_parity`
with `--capture LABEL=DIR` for each capture directory (pyarrow is only needed
locally for the dataset comparison).

Fallback workers on weilandserver use CPU OSMesa and are labeled separately:

```bash
bash exp/mw_finetune/setup_workers.sh
export PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src"
export LD_LIBRARY_PATH="/data/mw_finetune/envs/osmesa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
/home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python \
  -m exp.mw_finetune.evaluate --renderer osmesa --host 127.0.0.1 --port 23150 \
  --model-id pi05_rlinf_osmesa_fallback --workers 12 \
  --run /data/mw_finetune/eval/pi05_rlinf_osmesa_fallback
```

Old local zero checks:1worker4.28ep/min;2workers7.63ep/min, two episodes per
measurement including startup. Scripted reach oracle succeeded at50/38steps.
The private OSMesa library is extracted without changing system packages;
its dependencies are the existing local LLVM20/libdrm/libstdc++ libraries.
