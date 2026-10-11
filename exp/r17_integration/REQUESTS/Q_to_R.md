# RUN_CONFIGS pitfall: tether completion and resumed controls (2026-10-10)

Retry explicit tether transport hiccups with bounded backoff. Tether uses rc70
for `waiting for chunk`, `nats: timeout`, and `read pipe: i/o timeout`; rc75
`too_many_in_flight` also needs retry. Require an owned fleet's confirmed exit
receipt to diagnose fleet death. For an ambiguous spawn reply, recover that
exact request's receipt; never replay the spawn blindly.

Use the current Q formal launcher for evidence sync: SHA-checked chunks of at
most4MiB, serialized across campaigns and throttled to2MiB/s, with independent
300s retry windows. A monolithic483MB pull with a60s timeout can fail and starve
another campaign's10s probes. Keep the original remote archive until the whole
archive and its snapshot manifest verify. Sustained outages and checksum or
application failures still fail closed.

A complete, closed campaign whose only failure was the final evidence pull can
be recovered from its preserved SHA-attested archive. Preserve the failed
receipt and require exact accepted-cell completion, owned-process closure,
execution-trace validation, and the native batching audit before publishing PASS.

For a resumed paired control, point `--paired-control` at the completed merged
run and keep `--runtime-from` at the original frozen release. Do not change the
original failed segment's receipt while the resume pins it. Updating a jobs JSON
alone does not update existing preparation seals: use
`formal_queue.sh --prepared-queue <queue> --prepare-only --rebind-control <merged-run>`
before launch; launched runs cannot be rebound. Preparation can defer completion,
but dispatch requires the merged receipt to be closed PASS. Current six Fast-WAM
jobs/preparations were already rebound to `hs6fw_res_152931`, with runtime from
`hs6fw_ctl_143332`; C10 has subsequently been launched by the coordinator.


## RUN_CONFIGS pitfall: private MPS control race / startup retries (2026-10-10)

`hs5dp_c15` and `hs5dp_c15r0` failed in the frozen MPS replica preflight,
before CUDA model initialization or any accepted episode. Their private daemons
had already returned a successful readiness query. At 15:56:02.663 and
15:57:47.319, respectively, two control connections arrived at the same
millisecond, but only one became a valid UI session; the other CLI reported
`Cannot send command to MPS control daemon process`. The same frozen MPS source
worked in C10 when both UI queries completed. Evidence:
`qualification/preflight/mps_queue_20261010/ROOT_CAUSE.json` and each failed run's
`mps/e5681f/l/control.log` / replica log. Both failed runs confirmed owned closure.

Use the current Q launcher. It installs a separately recorded, by-value CLI guard
outside the immutable runtime and prepends only that private CLI directory to the
resident/guard PATH. All control calls share a per-private-pipe flock; only
read-only `get_server_list` / `get_client_list <pid>` queries retry the specific
send error, with backoff and a total 7s deadline (inside the native caller's 10s).
Daemon start, quit, other errors, and command timeouts are never blindly replayed.
`MPS_CONTROL_ORCHESTRATION.json` and `RECEIPT.mps_control_orchestration` record the
guard SHA and original runtime identity. Server/model/worker releases stay intact;
parallel replica startup width and MPS treatment also stay intact. Width 1 remains
a valid conservative workaround. No global MPS pipe, GPU mode, or other PID is used.

New formal_queue processes default to `--startup-retries 2` (allowed 0..3). Retry
only a failed campaign with confirmed owned cleanup and zero accepted journal or
attempt rows; missing/malformed driver evidence or retained resume episodes refuse
retry. Capture sealed startup inputs before the one-shot launch, use a fresh run
name/private worker paths, and retain exact native archives, manifest, YAML,
episode plan/seeds and driver pin. Record the failed and successful physical run
paths in queue evidence. Any accepted episode requires explicit resume; unclosed
processes prevent switching GPUs. Library/cache/resume campaigns are excluded.

A live Python queue parent does not hot-reload its retry loop, but each next
formal_eval child loads the MPS fix from the working orchestration code. For the
already-running hs6 Fast-WAM queue, leave C10 and its parent untouched. A new
`--continue-queue /data/mw_finetune/eval/formal_queue_20261010_155703_e26f7fd4 --run
--startup-retries 2` invocation waits for the old queue to close naturally, skips
closed PASS jobs, then retries a zero-acceptance failed job and runs the remaining
prepared jobs. It launches nothing while the old queue's owned children are alive;
it stops if an old failure has accepted data or incomplete cleanup. If the old
queue succeeds entirely, continuation is an empty PASS. Use one continuation
output directory to prevent duplicate watcher claims; never interrupt the live
queue parent with Ctrl-C or a signal while its arm is active.


## RUN_CONFIGS pitfall: completed driver with missing worker shutdown ACKs (2026-10-10)

Driver rc=2 is not sufficient to fail a completed formal campaign. Require every
planned UID accepted exactly once in the durable closed journal, a matching final
attempt, contiguous/resolved error retries, no driver exception/rejected probes,
and receipt-bound fleet closure (child/supervisor dead, no owned worker survivors,
exit0 or exit2 only for the missing-ACK drain). The normal close path records the
raw `worker_shutdown` gap and resolved errors in RECEIPT/SUMMARY, then seals PASS
only after snapshot/execution/native batching audits. An unclosed fleet, missing
cell, abnormal exit or unresolved error still fails. The native driver receipt
and its rc2 are retained unchanged; this is orchestration, not a model change.

`hs6fw_c10` is recovered PASS with1000/1000, zero error attempts, all owners closed,
and a frozen native batching audit of38931 decisions. Its original PARTIAL is
`/data/mw_finetune/eval/hs6fw_c10/RECEIPT_BEFORE_SHUTDOWN_RECOVERY.json`; proof is
`SHUTDOWN_RECOVERY.json`. There was also a real final-stage size failure: the
645806080-byte H100 server tar exceeded the frozen qualification helper's512MiB
limit. Formal sealing sends a compressed, SHA-checked snapshot builder by value
with an explicit8GiB process-local formal bound; it never edits frozen files.
The identical server files compressed to12545692 bytes and passed their manifest
and native treatment audit. The original size rejection is preserved in the proof.

## RUN_CONFIGS pitfall: GPU handoff precedes evidence sealing (2026-10-10)

Final worker-tar pulls previously held residents/MPS and the launcher's GPU flock
for4-5 minutes after dispatch ended. The current Q launcher confirms owned fleet,
replica/proxy and private MPS closure first. For ordinary formal pure inference
(all integrated model families), it publishes durable `GPU_RELEASED.json`, pins
all closed inputs and a separate immutable CPU evidence orchestration release,
and starts a receipt-bound detached CPU finalizer. It hides CUDA, inherits no GPU
flock FD (`close_fds=True`), and only snapshots/transfers/verifies/aggregates. Its
PID/start ticks/pgid are in `EVIDENCE_FINALIZER_PROCESS.json`; readiness and final
state are in `EVIDENCE_FINALIZATION.json`, logs in `evidence_finalizer.log`.
Model/server/worker archives and the study's selected driver pin stay unchanged.
Chunk SHA retries, the global bulk-only throttle, original-stage retention until
verified install, native frozen batching audit, and durable resume merge all apply.

A launcher rc0 can now mean a successful GPU handoff with RECEIPT status
`PENDING_EVIDENCE`, not a scientific PASS. Finalizer failure becomes FAIL; only
verified final evidence becomes PASS. `--paired-control` refuses pending receipts;
`--resume-from` explicitly refuses a pending source, so its pinned files cannot
race finalization. Smoke/library/cache/native-parity launches seal synchronously,
but their resident cleanup also precedes final pulls. Never price a pending run.

New Python `formal_queue` parents proceed after the verified GPU handoff, check
late evidence failures before the next dispatch, and wait for every final seal
before their QUEUE_RECEIPT can become PASS. Existing coordinator shell queues
load current Q orchestration in each future child and automatically advance on
handoff rc0, including the DP flock wrapper and Fast-WAM retry shell. Currently
running children retain their loaded old close path; do not restart them.
The already-running shell parent cannot acquire a new end-of-queue barrier by
hot reload. After its ALL_DONE line, run this CPU barrier (substitute the actual
physical run names printed by its ARM lines if any zero-episode retry occurred):

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -m exp.r17_integration.ops.formal_finalize --wait   /data/mw_finetune/eval/hs6fw_c10 /data/mw_finetune/eval/hs6fw_c16   /data/mw_finetune/eval/hs6fw_c20 /data/mw_finetune/eval/hs6fw_c32   /data/mw_finetune/eval/hs6fw_k2 /data/mw_finetune/eval/hs6fw_k1
```

This barrier launches/signals nothing and exits nonzero on any incomplete seal.
Do not start a second GPU queue. For the DP shell, use the same barrier with its
actual completed physical run paths. Further transfers can overlap the next arm;
resume/pairing and the final scientific receipt remain behind the seal barrier.


## RUN_CONFIGS pitfall: a closed fleet's native snapshot exceeds stop grace (2026-10-10)

`hs6fw_c20` completed all1000 cells, including a resolved inference error on task34
init48, but its owned native fleet supervisor exhausted the12s stop grace while
snapshotting. The native fleet receipt was already PARTIAL/interrupted with no
worker failures; supervisor exit-9, dead child/supervisor, no owned worker survivors,
and a missing `.transfer.json` describe a close/snapshot gap, not an unresolved
scientific outcome. This supersedes the earlier exit0/2-only restriction solely
for a proven post-completion close gap; abnormal dispatch-time exits still fail.

The normal close path and CPU recovery require the exact closed durable driver
completion (every planned UID once, resolved contiguous retries, no driver/probe
errors), receipt-bound fleet closure, and an explicit `worker_close_gap`. A missing
native snapshot is rebuilt compressed by value only after checking PID/start ticks,
all registered workers gone, the exact launch-manifest SHA, native fleet PASS or
PARTIAL/interrupted without unreplaced failures, and the nonblocking snapshot lock.
All source files are SHA-checked before/after the snapshot. The sealed PASS still
requires verified transfer/install, complete execution traces and the frozen native
batching audit. Original receipts, raw rc2 and the close/error/ACK gaps are retained;
PENDING_EVIDENCE is never a paired control or a resumable scientific result.

## RUN_CONFIGS pitfall: plain bind rejects released TCP TIME_WAIT ports (2026-10-10)

The previous H100 admission check used a plain `bind`, which can raise Errno98 even
when the prior owned listeners are closed. `formal_ports.py` now tests the actual
reusable listener shape (`SO_REUSEADDR`, bind and listen, holding all configured
ports together) and waits at most120s per host for a busy listener to release.
It checks H100 server ports and the local driver port independently, keeps the
configured model window unchanged, never signals a port owner, and writes
`PORT_RELEASE.json`. A real occupied port that does not release remains a failure.
The real CPU regression actively closes a TCP connection, reproduces the old plain
bind Errno98, and verifies admission succeeds while a live unrelated listener is
never disturbed.

New formal_eval child processes in the existing coordinator shell queues pick up
both fixes automatically; do not restart a running parent or arm. Already-loaded
children retain their old close path. Run the documented CPU seal barrier after
closure; incomplete source baselines and zero-episode failed startup attempts are
never promoted to PASS. CPU evidence/tests are under
`exp/r17_integration/qualification/preflight/close_snapshot_ports_20261010/`.

## Cache fleet RAM and cross-node continuation (2026-10-10)

Cache workers actually reach3.1–3.5GiB RSS; the old2048MiB reservation allowed
48 Fast-WAM +32 DP workers to overload timan107's220GiB RAM and trigger kernel
OOM/SIGKILL, followed by worker_restart_resource_timeout. Never pack those two
fleets there. New cache admission reserves at least4096MiB/worker, including
the future footprint of historical live cache fleets, plus node RAM headroom.
T1 requires6144MiB RAM /1024MiB renderer VRAM, at most16 workers, N4 recycling,
and its pinned GPU3 UUID/PCI/minor0 namespace guard; GPUs0–2 are excluded before
NVIDIA queries. Local workers reserve4096/1024MiB with32GiB RAM and16GiB server
growth headroom plus10% physical VRAM headroom.

Closed cache continuations may redistribute the SAME total workers with
`cache_lane --run --redistribute-workers fastwam_libero:timan107=16,local=16,timan1=16`.
Use a separate model-scoped override for the following8-worker Cosmos job,
e.g. `--redistribute-workers cosmos_libero:timan1=8`. Every change is a new
prepared segment with WORKER_REDISTRIBUTION.json; source receipts, native
runtime/client/simulator bytes, full episode UIDs, settings and control pins
stay read-only. Per-worker node/EGL device/UUID is recorded. Retain only complete
decision/attempt/final server-ACK outcomes, independently of success; rerun
missing/unacknowledged cells. Disk admission remains8GiB; redundant provisioning
tar cleanup requires a SHA-verified preserved copy and a closed source, never
deletes simulator installs or a failed/resumable campaign.

Fast-WAM batch-treatment v2 seals the canonical worker topology too. Keep
`worker_nodes` and the original batch treatment/identity exactly pinned;
`execution_worker_nodes` is the separately recorded admission/dispatch map.
Changing the canonical field causes a pre-GPU batch-treatment rejection even
when the total workers and scientific settings are identical. The resident
still attests the original treatment; receipts disclose both maps and every
actual worker node/device.

## T1 namespace wrapper and planned recycle (2026-10-10)

T1 GPU3's pinned bubblewrap namespace forwards native worker exit75, but its
Popen PID is the wrapper, not the Python PID in RECYCLE.json. Comparing those
PIDs spent the crash-restart budget on every successful N4 recycle. Preserve
the GPU3 wrapper/sitecustomize pins; launch the Q timan1_workers relay inside
the namespace. NATIVE_PROCESS.json binds the native PID/birth and wrapper
ancestry to a random, unique generation token. Accept exit75 only when this
record matches that generation, its wrapper PID/birth, and the original native
N4 episode-boundary marker. Such replacements reuse the admitted RAM/EGL slot
and consume no crash-restart budget. Missing/tampered markers and real deaths
remain bounded failures. Refresh only undispatched worker orchestration; never
edit a running immutable payload. Graceful owned-launcher stop + cache resume
retains only complete terminal-ACK cells, with original scientific/server pins.
CPU reproduction and actual T1 wrapper proof:
qualification/preflight/timan1_recycle_20261010/.
