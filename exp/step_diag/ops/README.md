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
