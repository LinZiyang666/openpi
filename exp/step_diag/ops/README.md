# step_diag ops runbook

Servers: **h100** (RoboCasa) and **weilandserver** (LIBERO). Simulator workers:
**timan108** (RoboCasa) and **timan107** (LIBERO). Run from the repository root;
commands below abbreviate `exp/step_diag/ops/` as `ops/` and
`exp/step_diag/config/` as `config/`.

## Before rollout

1. Inspect `nvidia-smi`, listening ports and named tmux sessions on the relevant hosts.
   Stop only processes verified to belong to this experiment. Do not stop other experiments' MPS daemons.
2. Deploy the same working tree to each host (`git rev-parse HEAD` equal on all four). The manifest
   records the commit, a source digest and the runtime (host / GPU / torch) as *notes*; what must be
   identical across the arms of one comparison is the checkpoint identity and the environment contract.
3. Emit YAMLs with `python -m exp.step_diag.emit_arms`. On each serving host run
   `python -m exp.step_diag.emit_arms --out <tmp> --env-ids <local environments> --check-libraries`.
   An optional `--library env=path` remaps the path in the emitted YAML **and** verifies that effective file.
   Deploy those emitted YAMLs when using a remap. Library checks cover every chunk/snapshot, finite values,
   full step count and GR00T schedule. Executed dimensions come from the adapter contract, never library variance.
4. Freeze weights with `ops/freeze_weights.sh <weights-dir>` (LIBERO accepts `env_id=library.pkl` bindings).
   This also writes the library's `<pkl>.sha256` sidecar, which the servers reuse (a library is hashed once,
   never at every start). Each server writes `manifest_<config_sha>.json`: the checkpoint identity (file
   listing + sizes + config/asset contents, no weight reads), cache config sha, library digest, commit,
   source digest and runtime; `config_sha` covers the served contract only (not commit / runtime), so a
   restarted server keeps it and the driver can resume the cell. Every row references that digest.
5. Complete the manual parity gates below and the distinct smoke experiment before formal rollout.

## Servers

```bash
SD_REPO=... SD_PY=... SD_CKPT=... ops/serve_pi05.sh pi05_rc shadow shadow 23240 config/arms/pi05_rc/shadow.yaml
ops/serve_pi05.sh pi05_rc plain plain_k2 23241 2
ops/serve_pi05.sh pi05_rc full full 23242 -
ops/serve_pi05.sh pi05_rc warm warm_t0.2 23243 config/arms/pi05_rc/warm_t0.2.yaml
ops/serve_groot.sh groot_rc plain plain_k1 23250 1
ops/serve_pi05.sh pi05_libero_10 shadow shadow 23140 config/arms/pi05_libero_10/shadow.yaml
ops/serve_groot.sh groot_libero_10 shadow shadow 23150 config/arms/groot_libero_10/shadow.yaml
ops/stop_servers.sh 23240 23241
```

Server output root defaults to `exp/step_diag/data/server`. Cells are under
`<root>/<teacher>/<arm>/`, with teacher `pi05` or `groot_tp`; shadow uses
`<root>/<teacher>/shadow_<env_id>/`. Read `config_sha` from that cell's
`manifest_<arm>.json` and pass it to the driver. Content-addressed manifests are retained across restarts.
A claim that is not listening and a readiness timeout return failure; inspect the named log before retrying.

Pi0.5 and GR00T RC use one connection per server process. GR00T LIBERO supports concurrent connections;
this experiment's launcher conservatively also assigns one worker per endpoint.

## Warm-start continuation variants (2026-09-21 follow-up, pi0.5 RoboCasa only)

`serve_pi05.sh pi05_rc warmreset warmreset_t0.2 <port> config/arms/pi05_rc/warm_t0.2.yaml` and
`... warmshoot warmshoot_t0.2 ...` serve the reviewer's `dt = -1/remaining_steps` continuation
(`exp.step_diag.pi05.warm_variant_stage3`): `warmreset` restarts the flow time at 1 (the cache is fed
as if it were noise), `warmshoot` keeps the cache's start_t (t crosses 0). Same cache yaml, same
Euler-step count as `warm_t0.2`, own manifest / config_sha. Cells run with `run_rc_cell.sh` per task
(50 episodes, formal seeds); `python -m exp.step_diag.analysis.warm_variants` pairs them against the
formal `full` / `plain_k2` / `warm_t0.2` cells. Descriptive only (not in the pre-registered family).

`resetfinal` (2026-09-22 ablation) is the `warmreset` loop started from the payload's final action chunk (t = 0) instead of the snapshot at `start_t`; `start_t` then only sets the step budget `n = floor(start_t*K+0.5)`. Serve it as `serve_pi05.sh pi05_rc resetfinal resetfinal_t0.2 <port> config/arms/pi05_rc/warm_t0.2.yaml`; the diag layer stashes the retrieval result to fetch the chunk.

GR00T (2026-09-22): `serve_groot.sh groot_rc warmreset|resetfinal <arm> <port> config/arms/groot_rc/warm_t0.75.yaml` (or `warm_t0.5.yaml`). `exp.step_diag.groot.groot_warm_variant_stage3` runs upstream's ascending `denoise_loop` as a fresh `n = K - snapshot_index(t)` step loop from `t = 0` with `dt = 1/n` (K = 4: t0.75 -> 1 step, t0.5 -> 2 steps); `install_warm_variant` routes the interceptor's `run_stage3_from` through it before the evidence capture wraps the runner. No overshoot for GR00T.

## RoboCasa cells

```bash
ops/run_rc_cell.sh pi05 plain_k2 main <servers-csv> CloseFridge,OpenCabinet,OpenDrawer \
    50 '{"OpenDrawer": 100}' <config_sha> qb
ops/run_rc_cell.sh pi05 shadow main <servers-csv> <eight-main-tasks-csv> 10 - <config_sha> shadow
```

PnP uses the canonical pinned-object table; main forbids pins. Formal seeds are `2000000+idx`;
cliffs/all secondary arms/full have 50 episodes, while the primary plain/warm flat cells have 100.
Smoke uses `SD_EXP=sdiag_smoke SD_BASE_SEED=3000000` and is excluded from formal analysis.
The driver validates these counts. Main/PnP have distinct run-plan/journal names. Experiment and config
identity enter the resume hash. Ordered server lists must keep the same logical slots across arms:
canonical task ID modulo fleet size selects the slot, independently of task subset or episode budget.
Worker islands and serving runtimes are reported per task (`comparison_notes`), not gated; a different
checkpoint identity or experiment namespace across the arms of one task is.

## LIBERO shadow

```bash
ops/run_libero_shadow.sh pi05_libero_10 <servers-csv> <frozen-pool-dir> <config_sha> shadow
ops/run_libero_shadow.sh groot_libero_spatial <servers-csv> <frozen-pool-dir> <config_sha> shadow
```

`run_libero_diag.py` uses conductor and the production `LiberoEpisodeRunner`. It launches the simulator's
Python directly (`SD_LIB_PY`), so no conda executable is needed. The pool digest, original task/init IDs,
launch/config identity, accepted journal and worker summary join the server evidence. Workers independently
verify pool bytes. The fixed design is 10 tasks × 10 inits; one worker per server serializes tasks assigned
to that server. Pi0.5 uses resize 224, GR00T 256, and both replan after five actions.

## LIBERO self-start round (`sdiag_libero_self`, logs/step_diag_libero_selfstart_plan.log.md)

Arms: `envs.LIBERO_SELF_ARMS_BY_POLICY` (pi0.5 9, GR00T 29) on `libero_spatial` / `libero_10`, the pruned A
pool, 10 tasks × init_idx 0..49 per arm and suite. Warm-family arms serve `config/arms/<env>/warm_t<start_t>.yaml`
(the env's shadow yaml with only the judge replaced). GR00T warm-reset arm ids name their step count,
`<mode>_t<native start_t>_n<N>` (N=1 from the T=0.25 snapshot, N=2 from T=0.5, RoboCasa's K=4 tuples);
`warm_t0.875` / `warm_t0.75` are the exact resume on the native 8-step grid. Server cells land under
`<server root>/<teacher>/<env_id>/<arm_id>/` (one arm id runs on both suites).

```bash
SD_EXP=sdiag_libero_self SD_OUT=<repo>/exp/step_diag/data/server_libero_self \
  ops/serve_groot.sh groot_libero_10 selfmidreset selfmidreset_t0.75_n1 23160 config/arms/groot_libero_10/warm_t0.75.yaml
SD_EXP=sdiag_libero_self SD_OUT=<repo>/exp/step_diag/data/server_libero_self \
  ops/serve_pi05.sh pi05_libero_spatial selfwarmreset selfwarmreset_t0.2 23140 config/arms/pi05_libero_spatial/warm_t0.2.yaml
python -m exp.step_diag.run_libero_diag --env-id groot_libero_10 --arm-id selfmidreset_t0.75_n1 \
    --experiment-id sdiag_libero_self --servers <host>:23160 --pool <pruned A pool dir> --config-sha <sha> \
    --tasks 3 --run-prefix sdq23160t3 --out-root exp/step_diag/data/libero_self --gpu-ids 0
```

One driver invocation is one (arm, suite, task subset) cell of 50 episodes per task; a queue gives every
sequential cell of one arm on one slot its own `--run-prefix`. Re-running the same command resumes the cell
from its journal; the exit code is 1 while the cell is incomplete. A smoke id (containing `smoke`) takes any
arm with `--episodes 1..50`. Analysis: `python -m exp.step_diag.analysis.warm_variants --env-id <env> ...`
(self vs cache / full / plain paired deltas per panel of equal continuation NFE m) and
`python -m exp.step_diag.analysis.success_length --benchmark libero` (successful-episode lengths in decisions
and env steps, settle steps excluded). A panel is equal in the continuation only: a self arm also runs a K-step
direct inference per decision to produce its start (K = 10 pi0.5, 8 GR00T LIBERO, 4 GR00T RoboCasa), so its
action-head cost is K + m per decision against m for its cache arm; the cells report continuation, self-start
and total NFE separately (`nfe_per_decision`, `episode_nfe_mean`). Pairs need the same (task, init_idx, env
seed, initial-state pool). Both reports label a panel `formal` only when every arm covers the frozen 10 tasks ×
init_idx 0..49 of `sdiag_libero_self` with every cell admitted; while the round runs they are `partial`, and the
LIBERO warm-variant report then puts the admitted task subsets in `macro_partial` (progress, not a result).
Length analysis excludes directories with conflicting manifests or a mismatched environment/arm, and binds each
accepted summary to an identity declared by that summary's launch. LIBERO summaries must carry task/index,
seed/pool and a valid settle-step count. A paired length is formal only when both arms are formal; unpaired
success counts include every loaded task, even while `full` is incomplete.

The RoboCasa `self13_queue.py` queue handles binary logs and resumes a cell that finished while the queue was down.
Its exit probe must finish successfully and confirm the tmux session ended before reporting a terminal result;
transport errors or incomplete probe output remain unknown and do not spend retries. Slot-specific orphan workers
are reaped after a confirmed cell exit. These paths are covered by hermetic shell and resume tests without contacting
the experiment machines.

The LIBERO round runs from its own trees (servers `/data/openpi_sdlib` on weilandserver and h100, workers
`/scratch/zixuans8/step_diag/openpi_lib` on timan107 / timan108, each with the frozen
`exp/common/data/db_init/libero/<suite>_apool` pools and, on the servers, the pi0.5 LIBERO libraries under
`exp/common/data/cache_artifacts/`), so the RoboCasa round keeps its deployed code. `libero_queue.py` queues all
760 (env, arm, task) cells behind the RoboCasa queue: a host takes LIBERO servers only once no RoboCasa job waits to
be claimed, fitted into the RoboCasa host budget next to the RoboCasa servers still alive (read from
`/data/step_diag_self13/queue_state.json`). Ports: weilandserver 23160–67 (pi0.5) / 23170–79 (GR00T), h100
23260–67 / 23270–79. Each cell is `ops/run_lib_cell.sh` (one task, 50 episodes, private tmux socket,
`SDCELL_EXIT` marker) with a per-slot, per-task run prefix; server rows land in
`<server tree>/exp/step_diag/data/server_libero_self`, driver artifacts in
`<worker tree>/exp/step_diag/data/libero_self`.

```bash
tmux new -s sdlq -d "cd ~/projects/openpi && python3 -u exp/step_diag/ops/libero_queue.py run 2>&1 | tee -a /data/step_diag_libero_self/queue.out"
python3 exp/step_diag/ops/libero_queue.py status
```

## Manual parity (Verify gate)

On each assigned serving environment, supply a real wire observation saved as an NPZ (no object arrays;
string prompts are scalar unicode arrays), and run with that server's Python:

```bash
SD_ENV_ID=<env-id> SD_CKPT=<checkpoint-dir> SD_OBSERVATION=<observation.npz> \
  python -m pytest tests/exp/step_diag/test_parity_manual.py --run-manual -q
```

Run once for each of the six environment IDs. The test compares every reduced k and full K against the
loaded production policy, checks shadow output/global RNG parity, then checks every warm t against the
full-loop snapshots and counts real denoising calls. GR00T uses action-encoder hooks as an independent
NFE counter. Evidence is written to `exp/step_diag/analysis/parity_<env>.json`. An explicit manual run fails
on missing GPU/assets. Also run the existing GR00T upstream/HDF5 parity test
`tests/robocasa365/test_groot_warmstart_manual.py --run-manual`; CPU tests cover fallback, extra calls,
recorder connection isolation, retrieval failures and accepted-attempt rejection. GPU tests have **not**
been run as part of this G2 repair. Formal rollout still requires the plan's Verify and smoke gates.

## Pull and analysis

```bash
ops/pull_server_rows.sh user@h100:<server-root>/pi05/plain_k2 exp/step_diag/data/server/pi05/plain_k2
ops/pull_server_rows.sh user@h100:<server-root>/pi05/shadow_pi05_rc exp/step_diag/data/server/pi05/shadow_pi05_rc
rsync -a timan108:<repo>/exp/step_diag/data/rc/ exp/step_diag/data/rc/
rsync -a timan107:<repo>/exp/step_diag/data/libero/ exp/step_diag/data/libero/
ops/analyze_all.sh exp/step_diag/data/server <weights-dir> exp/step_diag/data/rc \
    exp/step_diag/data/analysis [gaps-dir] [libero-driver-root]
```

Q-A requires `--driver-dir`; server finalize alone is insufficient. It binds the weight-library digest,
accepted terminal outcome, worker decision counts, array digests and frozen sample/seed sets.
Q-A missing-label strata use the frozen driver cohort: unaccepted retries are excluded, absent server
episodes remain in the denominator, and missing/conflicting terminal outcomes are reported as unknown.
Observed error rows and missing decision rows are separate counts. Q-B retains
outcomes from inadmissible cells for descriptive reporting and gives `inconclusive` when a required cell
fails; flat intervals use all 100 plain/warm pairs independently of the 50 full episodes. Other budgets,
Wilson intervals, paired exploratory intervals and per-episode actual NFE are in the JSON.
JSON outputs go under `data/`; the generated Markdown tables (`shadow_<env>.md`, `qb_<policy>.md`, concatenated
as `step_vs_warmstart_tables.md`) go under `exp/step_diag/analysis/` (override with `SD_REPORTS`); the interpreted
report `step_vs_warmstart.md` is written by hand from them.

## Q-C ladders

Reuse `exp/nfe_baseline/ops/ladder_server.sh` (server lane, one process per port via
`launch_pi05_servers.sh`) and `exp/nfe_baseline/ops/ladder_client.sh` (client lane) for pi0.5
object/goal and the spatial/10 anchors, as specified in the plan. The ladder switches k on the client's
DONE signal: run `ops/kdone_listener.py <port> pi05` on the serving host before the lane and
`ops/rc/sig_client.sh <client lane log> <serving host> <port>` on the client host; the idle timeout is
only a fallback (the idle-only protocol was removed on 2026-09-20). Do not mix the old RC 4090/H100
ladder outcomes into the new paired Q-B comparison.
