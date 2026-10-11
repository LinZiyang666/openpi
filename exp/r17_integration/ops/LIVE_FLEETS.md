# Live formal/cache worker fleets (v1, default off)

Prepare the campaign with its existing scientific/control/server arguments plus
`--live-fleets` on `formal_eval`. All attachable client routes are pinned in the
initial workers.tar. The launcher and driver publish LIVE_FLEET_CAPABILITY.json
and driver/LIVE_FLEET_DRIVER.json only for that protocol. A campaign frozen
before this protocol cannot accept the command: use a coordinator-scheduled
new segment. Do not edit its campaign.json, runtime or prepared seal in place.
Q's cache_lane currently retains its frozen worker-count CLI; the equivalent
fresh formal_eval cache job can opt in directly. P6_to_Q.md records the lane
forwarding integration point.

The CLI validates a live PID/birth, campaign/worker SHA and driver capability.
Without --run it validates only. With --run it durably queues an operation for
the campaign's owning launcher; it does not create an independent controller.
Attach IDs are never reused, including failed preparations. Identical native
worker/client/simulator bytes and source benchmark inputs are transferred through
the existing bootstrap installer and SHA-checked /tmp JSON stage. The source
worker archive, campaign.json and WORKER_DEPLOYMENT.json remain byte-identical.
Existing RAM, VRAM, disk, namespace, N4 recycle and transient-tether guards apply.
T1 is physical GPU3 only (pinned UUID/PCI/guard);24 workers require its normal
6144MiB RAM/1024MiB VRAM reservations and may be refused by admission.

For a **live-capable H100 campaign** named fw_live01, add24 T1 renderers:

```bash
.venv/bin/python -m exp.r17_integration.ops.live_fleets attach \
  --campaign /data/mw_finetune/eval/fw_live01 \
  --fleet-id t124 --node timan1 --workers 24 --run
.venv/bin/python -m exp.r17_integration.ops.live_fleets status \
  --campaign /data/mw_finetune/eval/fw_live01
.venv/bin/python -m exp.r17_integration.ops.live_fleets detach \
  --campaign /data/mw_finetune/eval/fw_live01 --fleet-id t124 --run
```

Use `--node local` or `--node timan107` for those renderers. To drain an original
fleet, use its registry ID (e.g. `--fleet-id initial_timan107`). Shrink is by
whole fleet; use several small attachment IDs when finer increments are needed.
An attach preserves resident replicas, sampler/batch/MPS, YAMLs, episode UIDs,
seeds and the driver's attempt fences. Added parallel gated MISS calls may have
a different batch composition; fleet/resource timestamps and per-dispatch node
are retained. No node probe is added per episode; owned status/lease probes
remain batched per node at30s. Preparation carries the parent retry journal in
a CPU thread so existing leases continue to be serviced.

Detach first fences new dispatch under the same lock as assignment. The SHA-
checked supervisor marker stops replacements; current episodes report through
the original TCP/result/commit path and receive SHUTDOWN on their next pull.
An already exiting N4 generation is retained without replacement. DRAINED
requires confirmed supervisor, child and descendant closure, with exit0.
No mid-episode signal is used for a requested drain. Episode/transport failures
still fail closed; this is not a lost-episode acceptance exception.

The finalization boundary rejects new/pending attachments. Every started fleet
has its own evidence key even on an already used node. The normal finalizer
verifies SHA/size/manifest for each snapshot before scientific PASS; omitting a
fleet refuses the seal. SEALED_EVIDENCE_RECEIPT.json includes the fleet registry,
input pins, resource history, dispatch node joins and closure. Existing cache
and formal resume selection is unchanged; closed fleet registries are additional
source pins, and live/open source fleets cannot be resumed. Automatic cleanup
uses each fleet's actual confined run path and waits for both eval and periodic
sync locks; failed/resumable evidence remains retained.

## Coordinator's two-minute check (read-only observations after attach)

1. Before submission, run the attach command **without --run**, save the current
   RECEIPT.json resident PID/birth tuples and hashes of campaign.json/workers.tar.
   A missing live capability means the old frozen campaign cannot be upgraded
   by this command. Do not bypass that refusal.
2. Submit attach. At30,60,90 and120 seconds run `live_fleets status`. Require
   request ATTACHED, a T1 GPU3 lease with its pinned UUID, the unchanged source
   SHAs and a fresh driver_observation (normally updated every5 seconds).
   PREPARING means CPU staging/admission is still running, not a successful
   worker attachment. A FAILED request contains fleet ID, node, step and cause.
3. Confirm t124 worker IDs in driver_observation.worker_census have
   worker_node=timan1/fleet_id=t124 and increasing pulls/results. Compare the
   resident PID/birth tuples with the saved ones; all must be identical. Inspect
   driver/fleet_dispatch.jsonl for t124 UIDs and join to accepted rows in
   driver/attempts.jsonl by (task_uid,attempt). Group commit may lag about1s.
   Existing server execution ACK/NFE audits remain the scientific completion
   gate. A quiet/finished campaign may have no new dispatch; report that limit.
4. For a temporary check fleet, submit detach. Require DRAINING, no later
   fleet dispatches, then DRAINED and no replacement generations after the drain
   marker. An episode may last more than2 minutes: leave drain pending and
   inspect later; do not force-stop workers. The final receipt must list t124's
   separate verified archive even after it has drained.

CPU-only implementation and tests have run; production attachment/GPU renderer
qualification has not run in this handback. Local/T1 private installs currently
support LIBERO; MetaWorld live fleets use the existing remote worker nodes.
Library/debug-teacher and native-parity-only campaigns are explicitly excluded.
