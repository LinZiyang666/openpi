<task>
Repo /home/weiland/projects/openpi (R17 framework under exp/r17_integration). Goal of this package: collect a
cache library on MetaWorld MT50 for every R17 MetaWorld model (dp_mw, fastwam_mw, pi05_mw, groot_mw; cosmos_mw is
being added by another agent and must fit the same path later) through the EXISTING formal launcher + teacher-capture
path, WITHOUT images. Owner decisions:
- Initial states are chosen by seed. Test set = `MT1(task, seed=7).train_tasks[idx]`, idx 0..49 (50 tasks x 50 = 2,500
  episodes, the official count). Library/cache = seed 8, idx 0..49 (2,500 episodes). Seeds 9/10 are reserved (do not
  use). Dev set: not decided yet - keep a dev partition definable but do not invent one.
- R17 debug capture is the system-level debug path. It must be able to drop images.

Do exactly these changes (smallest diff that works; no new seals, fingerprints, admission gates, receipts or
validation layers - owner rule: keep it simple; existing checks that block MetaWorld may be removed):
1. Seed per partition. Today `benchmarks/spec.py` BenchmarkSpec has one `seed=7` and splits idx 0-9 test / 10-29
   library / 30-49 dev; `benchmarks/metaworld.py:86` raises unless seed==7; `exp/metaworld/episode_runner.py` uses one
   `self._seed`; `decision_seed()` (`benchmarks/metaworld.py:19`) hashes the seed. Make test = seed 7 idx 0..49 and
   library = seed 8 idx 0..49; carry the episode's seed in the task (e.g. `task.extra["seed"]`, set where
   `conductor/manifest.py:episode_tasks` builds tasks from the partition) and use it for env construction, policy RNG
   (`decision_seed`) and recorded identity. LIBERO behaviour must not change.
2. No-image capture. `debug/library_capture.py` always stores `image_0/1` (lines 29-30), requires two cameras
   (`LibraryCaptureFactory.__call__`, `bind_observation`, `IMAGE_KEYS` includes `observation/wrist_image`). Add one
   campaign switch (e.g. `capture_images`, false for the MetaWorld library) so captures contain no image arrays;
   take the camera list from the model's `feature_spec.real_cameras` / the benchmark wire keys (MT50: one camera,
   `observation/image`). Also make the generic `debug/capture_bridge.py` DecisionSink able to drop image arrays
   (e.g. pi05 `capture_fields` returns `model_input_base_rgb`; drop uint8 HxWx3 model_capture arrays when images
   are off). Keep retrieval keys, robot state, normalized actions, identities, execution acks.
3. Library build without images. `ops/library_campaign.py` only accepts the two LIBERO suites, needs the LIBERO B-pool
   alignment audit, and `finish()` -> `search/teacher_library.build_teacher_parent` builds dense image arrays. Make an
   MT50 library campaign work: models dp_mw/fastwam_mw/pi05_mw/groot_mw, seed 8 idx 0..49, one trial per (task, idx),
   2,500 episodes (smoke = 2 tasks x 2 idx), no LIBERO alignment/B-pool for MT50, parent built without image arrays,
   output `/data/r17_library/<model>_mt50/bpool/`. Library subsets for MT50: nested per-task counts (default
   10/20/30/40/50 episodes per task, all 50 tasks) as row-index views; fits per subset if the existing fit code works
   for 50 tasks - if a fit step is LIBERO-specific, make the parent + subsets work and report what the fit needs.
   Server on `local` (4090) or `h100`; H100 shard streaming path must keep working for MT50.
   Ports: use the model's formal window (DP 23180-23189, Fast-WAM 23170-23179, pi05/GR00T 23160-23169, Cosmos
   23240-23249); remote workers reach a local server through ziyanglin.com (only TCP 23100-23199 forwarded) and an H100
   server at 149.165.153.233 directly; driver port on weilandserver in 23100-23199. Never `tether expose`.
4. Workers on timan107 AND timan1. Owner allows timan1 GPU 3 only (other GPUs belong to other users) for MetaWorld
   workers, max 40. timan1 has no MetaWorld env (timan107 uses `/scratch/zixuans8/mw_r17/envs/sim/bin/python`,
   metaworld 3.0.0 / mujoco 3.14.0 / gymnasium 1.3.0). Provision the same env on timan1 (copy or rebuild; see
   `ops/worker_installs.py`, `ops/formal_eval.py:prepare_workers`), confirm EGL renders on physical GPU 3 only, and
   let `ops/timan1_workers.py` (its lease currently requires LIBERO cache workers) and the launcher accept MetaWorld
   workers on timan1 (`--worker-counts timan107=N,timan1=M`). Render a few seed-8 initial observations on timan1 and
   timan107 and confirm pixels match.
</task>
<context>
Read first: `exp/r17_integration/ops/RUN_CONFIGS.md` (sections "Launch and pairing rules", the four MetaWorld MT50
sections, "Pitfalls"), `ops/LIBERO_LIBRARY.md`, `ops/formal_server.py` (library capture path lines 150-220),
`ops/formal_eval.py` (seal, prepare_workers, plan), `conductor/manifest.py`, `benchmarks/metaworld.py`,
`exp/metaworld/{env,tasks,episode_runner}.py`, `debug/{library_capture,capture_bridge,library_shards}.py`,
`search/teacher_library.py`, `ops/timan1_workers.py`.
Another agent (package B) edits concurrently: `models/dp/formal_batch_server.py`, `models/fastwam/formal_batch_server.py`,
`models/fastwam/formal_batching.py`, `ops/cache_study.py`, and in `ops/formal_eval.py` only `execution_horizon`,
`execution_steps`, the `--max-batch/--mps` admission in `parse_args`, and the batching branch of `seal`. A third agent
(package D) adds `cosmos_mw` under `models/cosmos/`. Do not edit their regions; never revert changes you did not make;
do not git commit (the coordinator reviews and commits).
</context>
<action_safety>
CPU work only: do NOT start GPU servers, formal campaigns or workers fleets (`--run`). The coordinator launches all GPU
runs. tether is allowed for timan1/timan107 provisioning and short render probes (keep calls >=30 s apart, no expose).
On timan1 touch only GPU 3 and /scratch/zixuans8. No sudo, no `rm -rf`, no bare `pkill -f`, no credentials,
no `codex exec`, do not touch /data/i018 or /dev/shm/i018.
</action_safety>
<verification_loop>
Minimal CPU tests for the new behaviour (seed per partition incl. decision_seed; capture with images off has no uint8
image arrays; MT50 library plan = 2,500 unique (task, seed 8, idx); parent build without images on synthetic captures).
Run the existing tests that cover the files you touched and report failures that existed before vs new.
</verification_loop>
<compact_output_contract>
Write `exp/r17_integration/handbacks/HANDBACK_MW_A.md` and end with: files changed (one line each), what was removed,
test results, timan1 provisioning + GPU-3 pin proof + pixel comparison, and the exact commands for the coordinator:
(a) a GPU smoke per model (dp_mw, fastwam_mw, pi05_mw, groot_mw): 2 tasks x 2 idx, seed 8, no images, `--prepare-only`
first then `--run`; (b) the full 2,500-episode MT50 library collection command template with worker counts as
parameters; (c) the `--finish` command.
</compact_output_contract>
