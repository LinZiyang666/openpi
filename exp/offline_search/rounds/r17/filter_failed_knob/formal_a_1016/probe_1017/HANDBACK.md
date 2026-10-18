# Worker control timeout repair and continuations — 2026-10-17 18:3x

The 18:19 control reply was a transport error, not a fleet-death report.
`formal_eval.refresh_remote_owned` wrote FAIL with `control_probe_failed=True`;
`Running.check` passed it to `worker_fleet_failed`, immediately retiring both
worker nodes. The driver's normal retirement handling then shut them down.

The coordinator code now gives worker control/lease-status probes **90 seconds**,
as for the capacity probe, and records exceptions as UNKNOWN. The generic
fleet-failure predicate ignores `control_probe_failed`. `Running.check` handles
both the old FAIL-shaped reason and the new UNKNOWN-shaped reply: it retains
the node until an explicit tether OFFLINE report, **300 continuous seconds** of
failed probes, or a successful status reply proving failure/closure. A successful
probe clears the timer. UNKNOWN does not rewrite the node-retirement state or
tell the driver to shut workers down. Existing 600-second leases are unchanged.
Only new launches load this code; active run files and chain scripts were not
edited.

## Stop Cosmos through its native cleanup

The verified launcher is local PID **187415**, start ticks **71735806**, parent
187414. It is `filter_failed_knob.launch --run .../fa_co_f_f_o`. Signal this
single PID, not its process group, wrapper, driver, individual residents or chain.
The native launcher handles SIGTERM, ends its run loop and enters its existing
`finally` cleanup. No signal or stop command was executed by the assistant.

The H100 chain advances after a failed item too. For a controlled handoff, the
coordinator first sets its **existing hold file**, without editing its script:

```bash
touch /home/weiland/projects/openpi_ext/scratch/r17_ops/mw/hold_ff_h100a
kill -TERM 187415
while test -d /proc/187415; do sleep 2; done
```

If executing later, check the pinned identity before signalling:

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python - <<'PY'
from pathlib import Path
from exp.r17_integration.ops.formal_process import ticks
import os, signal
assert ticks(187415) == '71735806', 'launcher identity changed; do not signal'
assert b'/data/mw_finetune/eval/fa_co_f_f_o' in Path('/proc/187415/cmdline').read_bytes().split(b'\0')
os.kill(187415, signal.SIGTERM)
PY
```

These are the native cleanup targets **in order**, for inspection, not a list
of PIDs to signal manually. Remote entries are stopped through their recorded
`formal_process.manage(..., 'stop')` requests; this checks process identity and
closes the supervisor/child. Full identities are in `COSMOS_STOP_ORDER.json`.

| Node | Label | Supervisor / local PID | Child PID |
|---|---|---:|---:|
| h100 | proxy | 1041699 | 1041700 |
| h100 | replica_7 | 1041562 | 1041563 |
| h100 | replica_6 | 1041406 | 1041407 |
| h100 | replica_5 | 1041247 | 1041248 |
| h100 | replica_4 | 1041132 | 1041133 |
| h100 | replica_3 | 1040972 | 1040973 |
| h100 | replica_2 | 1040829 | 1040830 |
| h100 | replica_1 | 1040687 | 1040688 |
| h100 | replica_0 | 1040586 | 1040587 |
| local | workers | 193744 | 193745 |
| timan1 | retired workers | 1643978 | 1643979 |
| timan107 | retired workers | 711961 | 711986 |
| h100 | MPS guard | 1040557 | 1040558 |
| local | driver | 224411 | — |

The guard's native MPS context manager closes its private daemon. Wait for the
launcher to exit and its receipt to settle. The continuation's prepare performs
the existing native read-only ownership proof before any staging; an UNKNOWN
cleanup reply is never claimed to prove closure. To run that proof separately:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python - <<'PY'
from pathlib import Path
from exp.offline_search.rounds.r17.filter_failed.resume import prove_closed
print(prove_closed(Path('/data/mw_finetune/eval/fa_co_f_f_o')))
PY
```

## Continuations

| NAME | Retained valid outcomes at build | Planned continuation | Requested execution workers |
|---|---:|---:|---|
| fa_fw_r_u_lc1 | 16,810 | **13,190 final** | timan1×30 + timan107×40; local capacity 0 |
| fa_co_f_f_oc1 | 2,667 | **3,333 at draft** | timan1×44 + timan107×34; local as native allocator allows |

Fast-WAM has 16,816 accepted rows, six with the known infrastructure error:
13,184 unaccepted planned UIDs plus those six errors = **13,190**. Its old
six-episode, never-prepared/never-launched draft was renamed to
`/data/mw_finetune/eval/fa_fw_r_u_lc1.six_infra_only_20261017T232908Z`.
`REPLACED_DRAFT.json` and `REGISTRY_BEFORE.json` preserve that change. The
registry was updated atomically; the archived draft is not a continuation tip.

**Fast-WAM closure is proved** by the native `library_salvage.prove_closed`
path on local, timan107 and timan1, even though its original FAIL receipt says
`all_owned_closed=false`. All five local driver/server/guard leaders and all
six fleet child/supervisor identities are gone. Additional procfs checks prove
the old launcher 3276308, MPS control daemon 3287352, private CUDA MPS server
3289638 and replicas 3287627/3332360 are absent. See `FW_CLOSURE.json` and
`FW_EXTRA_CLOSURE.json`. The original receipt was not rewritten.

**Cosmos remains running until the coordinator stops it.** Its draft count is
not claimed to be the final missing count. The existing continuation tool's
new `--remaining` mode selects every unaccepted/error UID using the native
terminal acceptance criterion. It builds from the sealed startup snapshot via
`cache_resume.prepare`, with no remote provisioning. On prepare, native closure
must first pass; the draft then reselects from the final parent journal **once**
and reuses `cache_resume.prepare` to update its segment plan and local seal.
Any UIDs accepted after the draft was made are removed from the segment.
Earlier draft plans/selections/seals are preserved by `.before_closure_*`
renames. A CPU regression proves that a newly accepted parent UID is not rerun.

`local_clone` now uses the existing merged timan1_gpu2 context, so the requested
44 workers use its existing 83-worker LIBERO cap rather than the old 40-worker
parser. No worker archive or scientific configuration changes. The requested
node maps are recorded through native `redistribute`; resilience measures and
records actual placement at prepare/launch.
The initial partial Cosmos clone that encountered the old parser was preserved
at `/data/mw_finetune/eval/fa_co_f_f_oc1.before_gpu2_counts_20261017T232939Z`;
it was never staged, launched or registered.

Both runs are registered in `filter_failed/CONTINUATIONS.json`. The formal-A
loader already follows `continued_lanes` and reads each ancestor once; no
change to an original campaign, its assets, plan or outcomes was required.
Only valid terminal outcomes count. Error rows remain missing until repaired.

## Prepare and launch

Both definitions and command pairs exist. Nothing was staged or launched.
The coordinator runs these on free GPU slots, after Cosmos's source has closed:

```bash
cd /home/weiland/projects/openpi
# H100; prepare freezes the final remaining-UID count after source closure.
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_co_f_f_oc1.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_fa_co_f_f_oc1.sh

# 4090; source closure is already proved. Do not overlap another 4090 server lane.
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_fw_r_u_lc1.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_fa_fw_r_u_lc1.sh
```

The build commands used, for audit only; do not repeat against these existing
destinations:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.resume --source /data/mw_finetune/eval/fa_fw_r_u_l --name fa_fw_r_u_lc1 --remaining --verify-closed --worker-counts timan1=30,timan107=40
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.resume --source /data/mw_finetune/eval/fa_co_f_f_o --name fa_co_f_f_oc1 --remaining --worker-counts timan1=44,timan107=34
```

Analysis remains:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a analyze
```

## CPU validation

**19 tests pass**, including the exact 18:19 timeout reason, UNKNOWN retention,
five-minute threshold, successful-probe recovery, explicit OFFLINE, confirmed
fleet failure, 90-second worker control calls, native driver fencing, inherited
outcome merge, and closure-time reselection. Four command scripts pass `bash -n`.
The per-continuation JSON checks pass the native frozen planner, exact original
task objects and **191 / 153 unchanged asset files**, with disjoint UID partitions
of **16,810 + 13,190 = 30,000 / 2,667 + 3,333 = 6,000**, respectively.
Cosmos's partition is its draft snapshot; preparation replaces
it with the final stopped-source partition.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q exp/offline_search/rounds/r17/filter_failed/test_worker_resilience.py exp/offline_search/rounds/r17/filter_failed/test_resume.py exp/offline_search/rounds/r17/filter_failed_knob/test_resume.py exp/offline_search/rounds/r17/filter_failed_knob/test_infra_resume.py exp/offline_search/rounds/r17/filter_failed_knob/test_formal_a.py
```
