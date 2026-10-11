Formal SR queues use immutable server and worker releases and prepare one
campaign ahead. Each job is a JSON array of CLI arguments, with a unique
`--run-name`; the outer file is an array of jobs. All jobs must target the same
`--server-node` and `--gpu`. Every model, checkpoint, batching, horizon, worker
recycling and renderer option is passed through to the formal launcher.

Start a queue after the owner releases its GPU and selected ports:

```bash
exp/r17_integration/ops/formal_queue.sh --jobs exp/r17_integration/ops/formal_queue_fastwam.json --run
```

The example is a **fresh B8/5 ms official-2000 K10/C5 control**, followed by
K10/C10 with the same batching, checkpoint, GPU and worker treatment. Add
remaining arms as more argv arrays. Change run names before another execution.
`--paired-control @earlier_run_name` resolves to that job's result directory.
Preparation may validate the control's treatment and seals while it runs;
launch still requires its completed, closed PASS receipt and batching audit.

Every `--paired-control` arm automatically inherits that control's frozen
runtime and worker payload. For DP or another study without that flag, pass
`--runtime-from /data/mw_finetune/eval/<baseline_run>` explicitly; the DP
`models/dp/formal_eval.sh` shim forwards it. This verifies the preserved
`runtime.tar`, `RUNTIME_RELEASE.json`, worker archive/deployment and native
ModelManifest, then reuses the original release. Missing or changed releases,
weights, code pins, batching, MPS treatment or checkpoint are refused. A running
baseline can supply immutable code for preparation; its receipt is untouched.
Frozen local sealing, resident/proxy and driver imports use the release as
their working directory, so Python cannot prefer the working tree. Each arm
records its source proof in `FROZEN_RUNTIME.json`. GPU allocation and dispatch
remain exclusively launch-time operations.

To prepare a single campaign while another campaign uses the GPU, replace its
`--run` with `--prepare-only`. Busy worker RAM is recorded as a preview and
does not prevent preparation. This action creates no GPU residents, EGL
environments, worker fleet or Cosmos tunnels. Immutable releases are published
atomically under hashes; an existing corrupt release is refused, never repaired
over a live release. Hash-matching worker archives are not pushed again. The
same original LIBERO assets and dependency releases are verified and reused.

Launch the resulting directory directly:

```bash
exp/r17_integration/ops/formal_eval.sh --run --prepared-run /data/mw_finetune/eval/<prepared_run>
```

The preparation seal binds assets, config, episode plan, worker deployment and
runtime. A launch is one-shot and refuses CLI overrides or modified preparation.
It checks immutable bytes and current RAM admission before binding replicas;
the fleet independently reserves current RAM and renderer VRAM before forking.
Cosmos allocates its own tunnels at launch and updates only its campaign's
transport addresses. Resume source pins are checked again. Existing resume
semantics and the honest descendant-closure requirement are preserved.

For a CPU-only queue preparation, use `--prepare-only` instead of `--run` on
the queue helper. These directories can subsequently be launched individually;
the CPU-only queue never dispatches them. `PREPARED.json`, `PHASE_TIMINGS.jsonl`,
and the queue's `QUEUE_*` records provide the preparation and process evidence.
The full queue stops on a failed campaign, failed preparation, or incomplete
owned cleanup. It signals only its receipt-bound launcher PIDs/groups.

Once the paired control has a completed, closed PASS, launch **all already
prepared jobs** in their original order without regenerating names or code:

```bash
exp/r17_integration/ops/formal_queue.sh --prepared-queue /data/mw_finetune/eval/<cpu_queue_output> --run
```

This validates every preparation seal first, claims each campaign once, and
stops before the next campaign on any failure or incomplete cleanup. An arm
still requires the original control's closed PASS and matched batching audit.

LIBERO defaults follow M's measured burst bound: four completed episodes per
worker process, 6144 MiB reservation, two fault replacements. MW recycling
stays off by default. D's per-worker round-robin physical-device/UUID assignment,
global VRAM accounting and timan108 GPU1/2 exclusion stay active. Explicit
`--worker-egl-devices` and `--worker-vram-mib` options are passed unchanged.

DP replicas start in waves of four, Fast-WAM in waves of two; heavy GR00T and
Cosmos loads remain sequential. Use `--replica-startup-width 1` for the prior
load order. Each resident still self-checks and attests its sealed model, DP
still requires CUDA graphs, and existing VRAM caps/accounting remain active.
Workers import and attest concurrently with replica loading; stock reconnects
wait for the driver. The driver starts only after all residents attest and the
proxy acknowledges the full-arm bundle. No episode is dispatched early.

Pure-inference campaigns can opt into `--driver-mode group-commit`. Server and
worker code still comes byte-for-byte from `--runtime-from`; the driver uses a
separate immutable overlay containing only `formal_driver_fast.py` and
`group_commit.py`. `ORCHESTRATION.json`, the campaign receipt and summary record
the source control's driver pin and the selected driver pin. Default mode is
`legacy`; library/capture and parity-only runs cannot use this extension.

The evidence writer groups at one second or 64 events, without filesystem I/O
under the scheduler lock. It fsyncs steps, attempts and dispatch evidence before
writing and fsyncing the terminal journal; only then does the scheduler count
completion. A surviving terminal record always has durable supporting evidence.
Uncommitted assignments/diagnostics can be lost; recovery uses `--resume-from`
into a fresh segment after the source fleet is closed or explicitly isolated.
In-place driver restart is refused. Failed commits and unclosed drains fail
completion even when journal lines remain visible in the page cache.
For resumed grouped campaigns, a SHA-bound raw segment is fsynced under
`durable_segment/` before merging; the composite evidence is then fsynced before
aggregation publishes completion. The source run is never modified.

To update only the driver of an existing CPU-prepared queue:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu exp/r17_integration/ops/formal_queue.sh \
  --prepared-queue /data/mw_finetune/eval/<cpu_queue_output> \
  --prepare-only --refresh-driver group-commit
```

This rechecks original native/worker bytes, import probes and frozen episode
planning, preserves the previous preparation seals, and refuses any launched
campaign. A failed refresh makes the queue unavailable for launch. Refresh and
launch share a campaign-local lock; GPU allocation and dispatch remain off.

Using the preserved 20–22 second DP and 39 second Fast-WAM startup measurements,
the expected prepared switch is about 80–110 seconds for DP12 and 60–100 for
Fast-WAM2, including warm validation, cleanup/control overhead and worker
reconnect. These are estimates, not GPU measurements of the new path. First
acceptance additionally includes real episode execution. Cosmos parity and
unmeasured compilation/model loads can exceed two minutes; prestaging removes
CPU deployment from the critical path but does not promise a universal bound.
