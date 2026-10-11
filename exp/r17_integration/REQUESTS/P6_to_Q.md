
## Cache-lane CPU handoff and source-inspection regression (2026-10-10)

P6 full cache lanes are CPU-prepared (csl10d/f/c;20,000 A cells total). Full cache campaigns now use your guarded MPS/startup retry, rc2 recovery and GPU-first async finalize. Existing model/worker native code is inherited from the control with an explicit reviewed R17 admission overlay; original controls remain read-only.

The long composition run loaded formal_eval.py before shared checkout edits shifted its source lines. test_no_transfer_before_gpu_closure_in_normal_path used inspect.getsource on that stale code object and failed at source.index, while a fresh targeted rerun passed. P6 changed only that structural test to parse a single source snapshot by function name and added a stale-firstlineno repro. Behavioral transfer/closure tests are retained. No running process/GPU touched.

## Urgent cache smoke fresh-port routing and liveness fix (2026-10-10 18:3x CDT)

csmd1010 was a backend readiness routing timeout, not a spontaneous worker
crash: local replica0 ready, but launcher waited on ziyanglin.com:23401;
workers waited for an unstarted driver, then were interrupted by owned cleanup.
All34 original remote identities and both local identities are now closed;
P6 sent no signals. Current formal_eval allocates policy/driver tether routes
for opt-in cache studies (allthree models), checks local cache backend listeners
on loopback, and rejects failed/dead/exit0-startup worker fleets even if their
supervisor/lease survives. Cache workers publish their receipt/log locations
and awaiting-driver address after admission. Default native DP/Fast routing
stays unchanged. Cache smoke DP --run now uses the local r17_gpu.lock; the
full4090 cache runner takes the same lock per GPU launch until handoff. H100
never opens the coordinator's4090 lock.

CPU startup27 tests +27 subtests PASS; default-off259 PASS. P6 owns authorized
replacement DP smoke csmdf1010, lock witness LOCAL_GPU_LOCK.json was HELD before
first resident spawn. Generic launcher fixes also apply to queued H100
smokes and already prestaged csl10d/f/c without modifying their native/model
archives: current launch_prepared rebinds endpoint and driver routes at launch.
No H100 GPU work or hs6fw_* process actions by P6. Full smoke and CPU composition
results will be recorded in HANDBACK_P6.md after closure.

Final urgent-smoke handback: csmdf1010 sealed PASS32/32,0 errors,1227 hits,
26stall-only MISS/2600NFE,1253decision/ACK pairs and32 terminalACKs;16YAML
ACKs,eight groups ×two suites,no replica restart. All owned processes and
private MPS daemon closed;4090 lock FREE. Full CPU881+43subtests PASS,
finalops369+43 PASS,legacy259 PASS. H100 GPU smokes/full lanes untouched.
See HANDBACK_P6.md "Urgent DP cache smoke startup routing + failure detection"
and ops/cache_study/evidence/csmdf1010_smoke.json for SHA-bound receipts.

## Cache N4 recycle + accepted-cell continuation (2026-10-10)

The failed csl10d fleet admitted RAM, refused a fresh GPU replacement check
for 120 seconds, and had zero worker crashes. Planned N4 now reaps the old
PID and transfers its continuously held RAM/EGL slot without a second global
admission; same device/UUID/VRAM and reservation-owner identities are checked,
with a 600-second bound and detailed replacement evidence. Cache close uses
a compressed SHA-manifest archive (the previous uncompressed archive also
exceeded Q archive_stage's 512 MiB bound).

cache_lane/queue_retry now explicitly resume complete accepted outcomes under
fresh names, then continue remaining cells/models. Resume checks the actual
structured final server ACK; original csl10d retains4001, missing3999. Only
formal_workers.py and worker_gpu.py refresh in a copied worker payload; native
model, benchmark, simulator, worker client, inputs, YAMLs and driver stay pinned.
Undispatched cache fleets automatically refresh those helpers at launch.

P6 launched the authorized DP lane continuation dpc8513584592 inside csl10dp,
using the existing local4090 GPU lock. Log:
/home/weiland/projects/openpi_ext/scratch/r17_ops/cache_lane_4090_resume_1010.log.
It is to remain running. Original owned PIDs were already closed; P6 sent
no signals and touched no H100 lane/fleet process or active campaign file.

**Live H100 caveat:** its already-imported queue and frozen csl10f fleet retain
the old code; source edits cannot patch those processes. Leave it running.
If it naturally fails and confirms all_owned_closed=true, coordinator should
relaunch the same lane (no local GPU flock):
`.venv/bin/python -u -m exp.r17_integration.ops.cache_lane --lane h100 --directory /data/mw_finetune/eval/csl10h --run`.
The refreshed lane selects complete accepted cells and resumes only missing
ones before continuing Cosmos. An undispatched Cosmos launch through the
current formal_eval gets the recycle-helper refresh automatically.

Final resource addendum: the old supervisor also leaked one log FD per N4
generation under timan107 soft NOFILE1024. New formal_workers closes the old
handle after reap, keeping one handle per slot and preserving all log files.
Regression ran81 genuine CPU generations/80 recycles under NOFILE64 and PASS.
Final framework/regression gate129 PASS; final legacy259 PASS.
For already frozen DP dpc8513584592 only, P6 verified fleet PID2520110,birth,
UID and exact launch campaign, then raised its soft NOFILE to8192 (hard
unchanged). Evidence csl10dp_resume_nofile.json. No H100 limit/process change.
Its live old fleet retains that leak too; if it naturally fails, the lane
restart above installs the log-close fix and resumes accepted outcomes.
DP progress4299/8000 accepted,0 new errors; keep it running.

## RUN_CONFIGS pitfall: worker run filesystem exhaustion (2026-10-10 20:43 CDT)

Owner's confirmed cause for the 20:2x–20:31 DP/Fast continuation failures is
**timan107 root filesystem exhaustion**, including mkdir ENOSPC and unavailable
fleet receipts. `/`, `/tmp`, `/scratch` and our formal runs directory share
the same filesystem (227084713984 bytes, about212GiB). Twelve GB of campaign
run copies/tars plus the9GB timan1 provision copy consumed its headroom.
Never stage bulk model/runtime/provision copies on that filesystem; use an
approved large destination on the intended host. Do not delete unrelated
Isaac-GR00T, openpi_trace*, simulator installs or anyone else's data.

P6 adds generic worker run/stage admission (default8GiB unprivileged available,
`formal_eval.py` and `library_campaign.py --worker-min-free-gib N`) before remote
mkdir/push and before a prepared launch claim. The fleet checks the run FS
before spawn/recycle; replacements retain their RAM/EGL slot while waiting
at most600s for disk recovery. Free bytes, threshold, inode availability and
pause/recovery events are recorded. No recurring tether probes are added.
Apply the same helper to local and future timan1 workers; it is node independent.
Local run root is `/data/mw_finetune/qualification/local_workers/runs`.
T1: the cleanup helper supports timan1 under the existing remote canonical
`/scratch/zixuans8/mw_r17/formal/runs`; if provisioning selects another root,
request an explicit confined-root change. This is not a timan1 fleet launch.

After successful evidence sealing, worker_storage removes ONLY that campaign's
closed run dir/tar/verified transfer sidecar. It rechecks local SHA/size/install,
remote archive SHA/size, exact path/UID, gone PID identities, exclusive journal
lock and every remote file hash. Removal uses plain `rm -- <files>` and bottom-up
`rmdir -- <dir>`, never force/recursive rm. It logs removed paths/bytes and
before/after free space in RECEIPT. Failed/pending/resumable campaigns and
local evidence remain. A cleanup refusal is explicit RETAINED, preserving PASS.

Read-only measurement at20:36: timan107 free6424244224 bytes =5.983GiB,
device64769 for root/tmp/runs. Below the new8GiB admission; a future launch
will refuse until space is recovered. Active lanes are frozen and untouched;
these edits do not retrofit them. A future cache fleet adds worker_disk.py
to the narrowly scoped orchestration refresh, preserving native/client pins.
Please append this pitfall and threshold to RUN_CONFIGS; do not restart live
lanes merely to install the guard. CPU tests and final measurement follow in
HANDBACK_P6.md, "Worker evidence cleanup + disk guard".

Sealing compatibility addendum: immutable `SEALED_EVIDENCE_RECEIPT.json` seals
the verified scientific evidence before deletion. The public RECEIPT remains
PENDING_EVIDENCE during cleanup and publishes PASS once with the cleanup result.
That preserves already-imported queue waiters and stable receipt pins. An old
CPU overlay missing worker_storage retains remote evidence explicitly and still
seals scientific PASS; it never imports unpinned working-tree helpers. Real CPU
background tests passed for both old and current overlays. The new cache fleet
uses the finalizer's bounded8GiB stage in-process (restores Q's512MiB default),
avoiding a second fallback tar for long N4 collections. Native files unchanged.

Q's concurrent timan1 integration now uses the same canonical remote run root,
so P6 disk/cleanup guards apply directly to it. No P6 timan1 operation was run.
Focused final gate167 PASS (including46 storage/disk cases),legacy259 PASS.
Final full ops PASS723 +43 subtests,0 failures/skips,1427.32s; final legacy259
PASS. Receipts worker_storage_fullops_final.xml/worker_storage_legacy_final.xml.
Final read-only timan107 audit at21:32:18 CDT:33328951296 bytes =31.040GiB free,
device64769 for run/stage,8GiB admission PASS. P6 production deletions remain0;
space recovery was external maintenance. Current lanes were not modified,
launched or signalled by P6. No further P6 CPU work pending for this fix.

## DP size500 capacity continuation (2026-10-10 21:57 CDT)

Owner requests replacing dpc3464789869 in csl10dp with roughly60 local4GiB
workers, zero timan107 and3–4 DP replicas. P6 is extending cache-only
continuations to permit a different execution total while preserving the
canonical control/batch/native/YAML seals. Each evidence segment records
fleet/replicas and the changed gated-MISS batch composition. Please avoid
concurrent edits to worker_topology.py/cache_resume.py/cache_lane.py during
this handoff; your same-total redistribution remains supported.

Live admission differs from the owner's snapshot: fastwamc5481985003 has16
local workers (~4GiB RSS each), leaving185GiB MemAvailable and consuming local
renderer VRAM. P6 cannot admit60 DP4GiB workers alongside it while preserving
RAM/VRAM guards. We will select the largest safe DP-only local fleet after
draining its old replicas, retaining4GiB RAM, the10% GPU margin and an explicit
DP-only8GiB growth reserve for4 replicas. This is separate from your five-size
capacity authorization. No H100/timan1 process or campaign change.

Live update22:06: dpc3464789869 drained through all current N4 children, with
1734 accepted, zero errors and zero unresolved dispatches; all owned processes
closed. P6 resumed as dpc8713063796 inside csl10dp under its normal4090 lock,
local33 workers ×4GiB, DP4xB8/5ms MPS, no timan107. GPU admission permits33;
RAM permits37 while Fast-WAM still owns16 local renderer/RAM slots. Requested60
cannot fit alongside that untouched fleet. Future DP continuation flags:
`--redistribute-workers dp_libero:local=N --dp-replicas4`; lane state persists
them for retries. Canonical32-worker control/DPbatch seals stay unchanged.
Each segment records its actual execution workers/replicas and possible
gated-MISS batch composition change. For future H100 boundaries, please
prioritize local renderers for DP as owner directs; do not change its live fleet.
Please also re-admit your prepared local60 size lane:251.75GiB physical RAM
minus32GiB headroom permits at most54×4GiB even when all other fleets close.

## Live fleets — P6 code-only coordination (2026-10-10)
Owner now requests live attach/drain for formal/cache campaigns. I inspected the
current ops tree (untracked, so ordinary git diff does not show it), including
Q's renderer_devices, CACHE_RAM_MIB3584 and local resource defaults. Preserve
those changes. I will add ops/live_fleets.py and ops/live_fleet_driver.py with
small opt-in hooks in formal_eval, formal_workers, driver_release and
formal_finalize; no running campaign, process or sealed directory will change.
Please avoid these hook regions until handback. Registry IDs distinguish fleets
on the same node; attach uses exactly the parent workers.tar, guards and native
inputs. Old frozen campaigns without live-fleet hooks must refuse attach.

Live-fleet integration collision: at CPU round4 the T1 authorize() live hook
was absent again (source had reverted to cache-study-only/max16), causing the
24-worker guarded live formal test to fail. I am reapplying ONLY that hook to
current timan1_workers.py. Please preserve the live_fleet_authorization branch,
local_workers' matching branch, and LIVE_OVERLAY/extra stream hooks. No global
resource default or renderer topology is being reset. Final SHA snapshot will
identify tested source bytes. If your lane reorder touches cache_lane, append
--live-fleets to freshly prepared formal_eval jobs only; old frozen campaigns
must fail closed and cannot be dynamically upgraded by working-tree changes.
The formal_workers.py drain/config-SHA hooks were also overwritten at22:40:19;
the real CPU supervisor regression then made35 replacements instead of draining.
I reapplied its small LIVE_FLEET_PROTOCOL=1 + drain_requested hooks at22:4x.
Please preserve those regions when landing periodic-sync/resource changes.
The initial worker payload will carry a protocol sentinel and prepare fails
closed if the supervisor hook is absent, preventing silent bad deployments.

Broad CPU recheck uncovered a periodic-sync composition gap: sync_back excludes
.incremental_sync.lock from snapshot manifests; worker_storage compared it as
ordinary evidence, retaining every otherwise qualified remote run. I updated
worker_storage's matching ignore list and also acquire the incremental lock
nonblocking before deletion (busy→retain). A real lock regression checks it.
The cold-refresh test fixture now supplies sync_back.py, matching Q's expanded
four-helper refresh; production cache_control was not changed by P6 here.

Live-fleet handback complete (CPU/code only): ops/LIVE_FLEETS.md and
HANDBACK_P6.md "Live fleet attach/drain". Final278 tests +3 subtests and259
legacy/default-off tests PASS. Source snapshot tests/ops/receipts/
live_fleets_sources.json was unchanged after the final suites. Preserve opt-in
formal_workers lifecycle lock/SHA/protocol marker, T1/local authorization branches,
LIVE_OVERLAY and fleet evidence hooks. Q's native worker payload/resource/renderer
and periodic sync logic is retained. cache_lane forwarding --live-fleets for NEW
jobs remains an optional Q integration; direct formal_eval cache jobs support it.
No currently running campaign was changed; old frozen campaigns fail closed.
