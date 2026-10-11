# P6 qualification and operations (r17.p6.v1 / r17.contracts.v1)

Today: CPU only. Nothing here authorizes use of training GPUs. The coordinator
allocates a future exclusive GPU slot after training has stopped, checks model
G2, and runs one immutable model at a time. No polling CUDA, no automatic launch,
no training-process control, no fixed 23150–54/23250–54/23180 ports.

## Data and fitting

1. `CaptureBridge(..., enabled=False)` is the default and creates no directory,
   writer thread or array copy. Opt in explicitly. Each attempt uses a new output
   directory. `DecisionSink(bridge, episode_metadata_resolver)` consumes P1's
   actual DecisionCapture, native snapshots/model_capture and sampler receipt.
   `PhysicalSink` consumes P2's native event mapping. Record campaign/arm/bundle
   generation/protocol/task/original init/task-data SHA/UID/attempt, never prompt
   as task identity. Actual acknowledged execution is a separate JSONL join;
   absent acknowledgments stay unknown. `CaptureFactory` creates connection-local
   DecisionSink objects over binding-shared writers, so concurrent same-bundle
   seq0 acknowledgments remain isolated. A single shared DecisionSink instead
   requires P1's associated RequestContext additive callback request.
2. The shared `BlockWriter` / numeric NPZ / StreamSink retain their bounded queues,
   failures and spill protocol. Native physical transport maps blocks to the
   existing controls_NNNN.npz allowlist; it does not add another network protocol.
   `CaptureReader` checks closed manifests and hashes. `allow_partial=True` is
   diagnostic only. Torn final lines are reported, completed corrupt lines fail.
3. `export_h5` writes geometry-neutral `r17_manifest`, named arrays/keys and
   records; it is an offline export. Live StepTrace forwarding uses the supplied
   stock H5TraceSink. `read_collected_h5` consumes existing DataCollector/TraceSink
   H5 only with a source content SHA, closed episode attestation, complete binding
   and model-owned feature-builder callback. It does not guess pool geometry.
4. `build_library(sources, destination, binding, episodes, key_fields=...)`
   requires complete contiguous selected source episodes, source bindings and
   exact FeatureSpec fields. READY.json publishes the dense/ragged read-only store.
   Native grid/x/RNG/posterior data retain native dtype and loop metadata. BF16
   observer tensors use float32 value datasets plus exact uint16 datasets named
   `original_name__r17_bfloat16_bits`. `array_serialization` maps each original
   array/key to its companion and native/storage dtypes; reconstruct native
   BF16 from those bits, including NaN payloads and signed zero. Nested snapshot
   noise/RNG tensors use `r17_array` references to named numeric datasets.
   Serialization copies observer buffers without changing inference/resume
   tensors or RNG. Producers that already cast BF16 without native dtype/bits
   cannot acquire that provenance retroactively. Real GPU validation is pending.
   Active action/state arrays exclude padding; normalized originals remain available.
5. MT50 retrieval split: library init10..29, dev30..49, test0..9. Formal500 stays
   50 tasks × init0..9. `nested_episodes` uses deterministic complete per-task
   quotas and explicit all/success-only policy. 50/100 total episode libraries
   lack the >=3/task LOEO corrector minimum; use raw retrieval or choose >=150
   with explicit revised supported sizes. Do not borrow dev/test episodes.
   When benchmark_protocol is a content digest (P2's protocol identity), pass
   `split_policy='mt50_init_v1'` explicitly; no benchmark is inferred from a hash.
6. `fit_recipe` calls P1's LayoutRecipe math. `fit_library(...,
   fitter=common.layers.fit_library_view, config={layout,ordered_tasks,...})`
   is the published alternative. Neither fit view contains outcomes or queries.
   `save_fitted_recipe/load_fitted_recipe` seal arrays and restore C/Fortran order
   to preserve exact BLAS behavior. Per-connection `recipe.fork()` resets mutable
   guard/knob state while sharing fitted arrays.
7. `save_stock_artifact` builds trusted local stock CacheEntry/CachePayload
   pickles and embeds r17_binding inside the top-level model mapping. NPZ never
   permits pickle. `load_stock_artifact` checks its receipt before stock loading.
   Full-hit actions work through the stock backend. Native warm remains in the
   versioned numeric store and requires a certified model executor; legacy
   uniform-flow payloads are never fabricated for DDPM/shifted flow.
8. Teacher library, factor calibration WarmupPool, and network GPU warmup are
   different recipes. Factor fetch/preload and network timing are framework/model
   hooks; these are not teacher episodes or formal outcomes.

## Replay, audit and reports

Fix round 1: R17 server/config imports now come from
`exp.r17_integration.common.config` and `common.websocket_policy_server`.
The stock bundle registry still owns publication; frozen shared source is restored
by P1. Server CLI debug attaches
`make_capture_sink_factory(provider, directory)` to the already-built core,
retaining its recipe and episode metadata. It creates a new DecisionSink per
connection and shares writers by exact model/protocol/actual sampler binding.
`PhysicalCaptureFactory(directory, server_metadata, sampler_resolver=...)`
requires attested `r17_model_spec` plus each event's actual sampler or explicit
num_steps with resolver(k). A reduced-K task cannot inherit the default-K binding.
Both factories expose `.close()`; caller seals them in its owned shutdown path.

Use `replay(method, library, episodes)` for fit/reset/query results and
`compare_replays` for exact action/normalized bytes, donor weights, guard, knob,
calls and cost. Fit/reload parity and actual stock WS staged-action parity have
CPU receipts; real resident recipe+model+WS runs need the later G3 receipt.
`enrich_journal` joins stock journals to sealed episode manifests by exact
UID/attempt. Accepted done/failed rows with error=None are valid; false success
is a completed failure. Error, rejected and late attempts do not enter SR.
Audit requires contiguous accepted decision sequences and all frozen YAML,
model/norm/layout/sampler/fit/library/protocol/split/code identities.
Zero-forward audit covers `tail` and every `*_tail`, including library/cache tails.

Pass the runner's `result.per_step_rows` as `cost_rows` to `summarize` or
`audit_campaign`; `_kind=execution_ack` is split automatically, summary rows are
ignored. `execution_rows` accepts separate client receipts, including final
partial execution on success/end. Canonical aliases are verified, not overwritten.
All joins require full campaign/arm/bundle/protocol/task/init/task-data/UID/attempt
and decision_seq. Sparse rows need a sealed `episode_manifests` entry for the
exact UID+attempt. Server/client control counts and exclusive cursor endpoints
must agree; a missing final acknowledgment remains unsupported.

Reports retain missing/error counts, per-task/difficulty/equal-group means,
actual stage/NFE/cursor/control counts, extra library/calibration/shadow costs,
absolute milliseconds and a within-model normalized baseline. Missing timing
or execution ledgers are unsupported, never zero. Different models' raw NFE
counts do not imply equivalent compute. Paired bootstrap matches campaign,
protocol/model/task/original init/task-data hash before resampling.
P1 `cost_ms`/`stage_timing_ms` or stock `server_timing.infer_ms` are measured host
wall time, including their stated scope; asynchronous GPU stage latency requires
a separate later qualification. `absolute_ms` retains diagnostic observed wall
cost; `pricing_absolute_ms` and normalized pricing require explicitly plain,
eligible timing rows. Any capture-enabled row excludes the whole paired price
total, even if a producer incorrectly marks it eligible. Unknown capture status
or overhead stays unsupported/None. Bridge preparation and original writer queue
wait are measured separately; native capture work is not estimated or subtracted
from an instrumented decision to invent a plain price. B9 GPU synchronization
and CP2-summary overhead need plain-vs-debug fixed-observation GPU measurement
in a future exclusive slot; report both distributions and batch composition.
Full native capture overhead remains unknown unless its producer supplies a
measurement covering that scope.
The full baseline receipt must also explicitly declare capture_enabled=False
and pricing_eligible=True; a captured/unknown reference cannot normalize plain
candidate pricing. Metadata identity matching alone does not certify timings.

Review B6: R17-LIBERO full runs use their sealed R17 commit/seed/wire/step-cap
protocol and are not the published Fast-WAM 97.6% protocol. Reference commit32,
seed42 and official LeRobot sidecar/select_action parity require their separate
P2/P3 receipts. Use DP K>=2 for qualification; its K1 trailing-DDIM profile
remains lossy research, with quality/descent admission pending P4/coordinator.
Native raw/synthetic-view diagnostics are not real paired-camera
qualification (B5/B11). Formal pricing uses B1 unless model owners qualify
B>1 numeric tolerance and record actual batch composition (B10).
HTML/PNG/CSV/JSON contain real camera widths/active channels/gripper3 for MW,
native grids and units. Native physical cards expose raw quantities; goal,
grasp/drop/restore tools retain capability status and pending semantic analyzer
status. They do not manufacture LIBERO predicate truth for MT50.

## Client payload and sync

`ops.build_client_bundle.worker_payload_files(root, debug=True)` consumes P2's
published file profile plus the conductor package's import closure and shared
writer modules. Build into a new SHA-named directory, use Python3.11, and run
the fresh-process P2 `import_probe()` before deployment. NumPy, msgpack and
websockets must already be installed in the client island; no torch/JAX/vendor
or weights enter the payload. Existing EGL build/first-render/close lock remains
P2's worker implementation and has CPU fake regression coverage.

`tether_transfer(argv, lane='tether')` retries only exit75 with
too_many_in_flight, at most8 attempts with capped exponential delays. Callers
retain normal tether transfer request IDs. Ports are held by PortReservation
until startup handoff; stock WS has no inherited-FD option, so the short
release/bind race remains and must trigger a new reservation on bind failure.
Own receipts bind PID/model; stop_owned terminates only that supplied process.
No pkill, name-based process sweep or training-process shutdown.

Formal/library launches explicitly opt into the bounded300-second transport
retry; exit64 `node_offline`/STALE and explicit relay failures are recorded in
`TETHER_BLIPS.jsonl` and the final receipt. Real remote command failures and
SHA failures remain terminal. Remote cleanup uses the same retry and process
identity proof. Default `tether_transfer` callers retain the8-attempt behavior
above.

Resident lease/status controls share one exec per node every30 seconds.
New H100 library campaigns set `h100_batch=true`: one bounded /tmp archive
contains up to8 closed shards, with a per-shard SHA receipt. The next exec
acknowledges the received batch and stages the next cohort. Batches normally
wait for4 shards, flush after120 seconds or at completion, and retain the
20GB capture bound. Both tar copies fit the existing1GiB transfer reserve;
the temporary /data tree uses hardlinks. A verified receipt is required before
any H100 copy is removed. The DP local capture/transfer path is unchanged.
`TETHER_CALLS.jsonl` and `receipt.tether_retry.call_metrics` count actual CLI
calls, including retries, for coordinator comparison with CPU rate estimates.

Large campaign/control JSON is staged under its campaign /tmp directory and
verified by SHA before use; large repair cohorts are never inline command args.
`library_campaign.sh --salvage FAILED_CAMPAIGN` is CPU-only and supports both
local and H100 collections. After proof that owned processes are dead, it
creates a separate `salvage/` collection, sealed provenance and
`QUARANTINE.json`. Local shared bindings become filtered per-episode metadata
views over the original array blocks. Missing worker results never count as
outcomes, even if their capture/acks look complete. Only complete accepted
episodes are retained, including ordinary failed outcomes. Local salvage
refuses errors after a step/decision and refuses to discard a returned teacher
outcome with incomplete evidence. H100 salvage also drains an interrupted
closed batch before remaining shards. Original evidence stays read-only.
Pass `--repair FAILED_CAMPAIGN/salvage` to recollect missing/quarantined UIDs
and finish the parent, subsets and fits; teacher seal/checkpoint/B alignment
checks remain mandatory. Coordinator commands and CPU evidence are in
[P6 handback](../handbacks/HANDBACK_P6.md).

For sync, stop the owned driver normally and call:

```bash
python3.11 -m exp.r17_integration.ops.sync_back snapshot RUN NEW_ARCHIVE.tar
python3.11 -m exp.r17_integration.ops.sync_back install NEW_ARCHIVE.tar DEST --sha256 ARCHIVE_SHA
```

The .eval.lock must be used by the driver. Busy journals fail. Snapshot fsyncs
and hashes immutable bytes, checks for uncooperative mutations, retains partial
inventory, and publishes only the complete archive. Transfer verification happens
before installation. Repeated installation is idempotent; differing local files,
tampering, unsafe tar paths and symlinks fail. Snapshot building streams source
files; installation verifies and writes in two passes with bounded 1MiB buffers.

## Future GPU slot: commands not executed

First render a default dry-run plan (safe on CPU):

```bash
python3.11 -m exp.r17_integration.ops.smoke --model pi05_rlinf_mt50 \
  --checkpoint-sha IMMUTABLE_CHECKPOINT_SHA --receipt NEW_PLAN.json
```

After coordinator scheduling and model G2, supply the model/benchmark
qualification callable (`module:function`) that accepts model_id,
checkpoint_digest, absolute monotonic deadline, closes all owned children in
finally, and returns a passing receipt or an explicit partial/failure. The
coordinator supplies the already-existing exclusive GPU-slot lock:

```bash
MODEL_PYTHON -m exp.r17_integration.ops.smoke --model MODEL_ID \
  --checkpoint-sha IMMUTABLE_CHECKPOINT_SHA --hook QUALIFIED_MODULE:qualify \
  --gpu-slot-token COORDINATOR_GPU_SLOT_LOCK --deadline 900 \
  --receipt NEW_GPU_RECEIPT.json --execute
```

Use one command per π-MW, GR-MW, Fast-WAM-LIBERO/MW and DP-LIBERO/MW immutable
checkpoint. The hook is intentionally not fabricated: models must publish it
after G2. 0–3min load+3warm; 3–8min two mechanical full/retrieval episodes;
8–11min 1–2 saved-obs shadow/native-resume (extra cost); 11–13min closed H5→tiny
artifact→new bundle→forced HIT; 13–15min normal owned shutdown, sync/join/report.
If loading is slow, return partial and stop within the deadline. Full pricing
(>=50 samples), 500-episode SR, throughput/batch/replica/split qualification are
separate later slots. SIGTERM in the owned child raises cancellation so hook
finally blocks can close its workers. If it ignores cancellation, only that
Popen child is force-stopped and the receipt records partial/forced stop; any
owned descendants then need receipt-based cleanup. Do not run a hook without
tested cancellation/finally behavior.

## Cosmos and split plans

Cosmos-LIBERO MVP: a separate adapter translates normal stock WS observations
(two images/proprio8/prompt) into its existing get_action(seed,K) protocol,
returns native H16×7 actions, and forwards episode controls. Seal EDM sampler,
no-future-decode/cuda-graph flag identities and actual measured cost. No CP1 key,
five-layer retrieval, stage batching or Cosmos-MW support is claimed. A later
owner must provide native staged/capture hooks before those bundles are allowed.
Cosmos-MW checkpoint remains unusable, owner coordinator/D.

Split: config-only CPUModelBlueprint must contain no tensors/arrays; choose a
short socket path (<104 bytes), certify serialization, then stage B1/B2 and
replica gates before a GPU-stage RPC smoke. Existing advanced-combination
restrictions (CP2/online-RIT/trace/warmreset/replicas/overrides) stay in force.
No split runtime has been launched by P6.
