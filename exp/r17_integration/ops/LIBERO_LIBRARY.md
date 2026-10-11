# LIBERO teacher campaign

Use `bash exp/r17_integration/ops/library_campaign.sh`, the library sibling of
the formal launcher. It calls the same CPU seal, resident server, stock proxy,
per-episode conductor, worker payload and closed SHA sync infrastructure.
`--run` is required to start GPUs; this implementation was tested on CPU only.
Model admission remains explicit: Fast-WAM requires Q's matching G2 A/B
framework receipts; Cosmos now has sol D's minimal capture hook and passed its
coordinator teacher smoke. π0.5/GR00T are identity references, never collected here.

Fast-WAM CLI takes `--g2-a-receipt PATH --g2-b-receipt PATH` for exact slots
`slot_G2_fastwam_libero_A` and `_B`. Both outer/relayed PASS records must say
GPU executed and bind their actual qualified `model_manifest` (ModelManifest
v1: `spec` plus `content_digests`). All checkpoint/weight/code/norm pins,
layout, features and native sampler must match the fresh CPU teacher seal.
Unbound PASS receipts fail before deployment with exact missing fields. The
owner's 20261009_234134/234224 wrappers currently lack those bindings; Q's
historical native/server sidecars have identities but are not SHA-bound in
the PASS records. The request to publish bound receipts is P6_to_P1 for Q.
`--ab-qualification` plus `--ab-descriptor` is an optional additional P3
loader-parity check, never required and never a substitute for G2 A/B.

For two Fast-WAM replicas, commands use total `--memory-budget-gib 64` (32GiB
per resident), `--replicas 2`, proxy23340, backends23341/23342 and driver23359.
Run the suites sequentially. H100 library startup acquires Q's host-local
`/tmp/r17_gpu.lock` through an owned expiring remote guard; a conflict fails
before any resident loads. The guard stays alive until owned residents close.
There is no coordinator/4090 flock wrapper. See `H100_GPU_LOCK.json` and the
guard's owned-process/cleanup entries in `RECEIPT.json` for actual admission.

Each invocation selects one suite, one full native bundle, and the exact B500
pool: ten tasks times original indices0..49, one attempt/trial per coordinate.
There is no official A input in a fit. The library retains ordinary unsuccessful
episodes; environment/transport errors produce a partial campaign, not a library.
Native protocols stay sealed: DP100/replan5, Fast10/replan5, Cosmos5/replan16
with seed195/environment0. The canonical40-task table is preserved. Suite-local
library IDs0..499 match π/GR even for serving IDs30..39 in libero_10.

The historical input files are under
`exp/common/data/db_init/libero/{libero_spatial,libero_10}`. Their SHA pins are
the `bpool_*.yaml` records under `exp/offline_search/closed_loop/ops/h100/`.
`ops/samples/libero_library_alignment/*.json` prove all coordinate/episode IDs,
all per-array dtype/shape/bytes, both model permutations and six R10 subsets.
GR spatial's464 missing original indices come from the SHA-bound R11 identity
certificate. Pinned files travel unchanged through `/tmp/r17_formal/` into a
private SHA-named worker pool; per-file probe hashes and each selected array's
task-data hash are checked before dispatch/inference. Both B suite directories
are installed because the stock worker's environment probe starts on spatial.

The proof is input-file SHA plus the same original array index, not an invented
full simulator-state comparison: historical H5 records have post-settle simulator
states, not the full reset input. All1,000 original trace reset metadata records
were checked read-only; no simulator or GPU was launched for the audit.

The opt-in `LibraryDecisionSink` saves only FeatureSpec keys, robot state,
normalized teacher actions, identities/native NFE and both real uint8 RGB frames
exactly as passed to the R17 model adapter. `image_sources` marks
`received_wire_uint8`, before any model-owned encoder transforms. It never
resizes/synthesizes a camera. Snapshots, RNG arrays, latent videos, tokenizer
tensors, diagnostics and physical action copies are filtered before the shared
writer. Final execution acknowledgments and closed accepted outcomes supply
success/progress/next/previous links. Captured timing is excluded from pricing.

Results are under `/data/r17_library/<bundle>_<suite>/campaigns/<unique-run>/`.
H100 closes an episode shard after its final decision acknowledgment, then
streams it while residents continue running. The existing SHA-attested `/tmp`
stage, bounded archive pull and snapshot installer verify every file on
weilandserver. A durable receipt precedes removal of the exact H100 shard and
its two archive copies. After owned replicas close, only the remaining server
metadata uses the original final snapshot. A complete collection builds
`<bundle>_<suite>/bpool/READY.json`, dense memory-mapped arrays including images,
then `bpool/subsets/size_{50,100,200,300,400,500}/`. Each subset contains only
`VIEW.json` and `row_indices.npy`; no feature/action/image library arrays are
copied. The builder reads capture blocks with a bounded two-block cache.

H100's campaign-wide limit is **20,000,000,000 bytes**, across all replicas:
256MiB reserved per active episode, 1GiB for the one
in-flight transfer's two copies, and 1GiB for control metadata. The shared
flock quota admits at most66 full active episode reservations. On closure,
the immutable shard retains only its actual bytes plus64KiB rounded up to1MiB,
capped at256MiB, until verified receipt. Closed small episodes therefore do
not consume a full256MiB slot. Array/header
charges happen before writer submissions, including queued buffers. Shards
cannot exceed256MiB. Used capture bytes are released only after verified
receipt; excess reservation is reclaimed when the writer is closed.

At capacity, a new episode waits outside the quota flock for up to120s so the
independent pump can verify/acknowledge pending shards. Active episodes retain
their entire reservation and can finish. Disk/identity failures still stop
immediately; a persistent transfer stall fails closed with shard/reserved/
requested/limit bytes. The20GB policy and free-space thresholds are unchanged.

Every pump failure writes `H100_SHARD_FAILURE.json`, marks the transfer summary
FAIL, prints the full cause, and includes shard id, operation step, archive
paths, exception chain and subprocess stdout/stderr/returncode. The final
campaign receipt includes `shard_transfer_failure`; shutdown failures print
their concrete cause too. If receipt storage itself fails, stderr retains the
transfer and logging errors. Plain DP local does not use this pump/quota.

Before deployment and resident startup, H100 must have at least
20,000,000,000+10GiB free on /data, and3GiB free on /tmp. During collection,
free /data must remain above10GiB+256MiB, and /tmp above3GiB. Checks run before
each capture submission/acknowledgment and in the two-second coordinator
poller. Low space, a full quota, corrupt receipts, failed transfers or missing
terminal acknowledgments stop qualification; data remains for diagnosis.
H100 space checks are future deployment actions; no H100 probe was run by P6.

The parent and all fits are built **only on weilandserver**; the launcher and
CPU finish reject H100 execution by hostname. The prior3x working-space ceiling
is for weilandserver, which holds all verified captures plus the parent/fits.
DP local retains its existing single binding writer, close/sync and build path.

Per-shard receipts are in `campaigns/<run>/shard_transfers/*.receipt.json`,
`H100_SHARD_TRANSFERS.json`, and the final `RECEIPT.json` under `h100_shards`.
`H100_DISK_ADMISSION.json` and `H100_DISK_START.json` record actual future
admission checks. H100 `SHARD_BUDGET.json` records reservation high water,
latest free bytes and any latched failure. A transfer or disk failure returns
a failing/partial campaign; it never drops an episode to finish a500 library.
Interrupted transfer evidence stays bounded and is not automatically retried
into extra remote copies. CPU --finish still requires a PASS, fully drained
campaign. A stalled transfer at shutdown has a60s join deadline; an incomplete
drain prevents parent publication. No running campaigns are migrated.

`bpool/fits/size_N/{READY.json,fit.npz}` seal P1's library-only PCA, Mahalanobis,
early metric, LOEO corrector, five-fold distance calibration, continuity scale
and unchanged corrector/no-progress caps. Guard defaults n3/epsilon0.5 are
fixed shared parameters, not thresholds trained from outcomes. Each size fits
only its selected complete episodes; images/outcomes are inaccessible to fit.
The normalized action/state arrays inside a sealed fitted recipe are its
self-contained serving inputs; the library subset itself remains a row view.

`KNOB_PRODUCTS.json` reports pending IR calibration unless `--ir-costs` supplies
a measured bound `r17.ir_costs.v1`. An optional `--miss-sampler` supplies a
matching sealed native SamplerSpec for a different MISS K. Calibration calls
P1's selected-library LOEO schedule and exact R11 fractional-gap cap solver
for IR25/32/40/45; infeasible targets are explicitly recorded. π/GR cost
coefficients and captured wall latency never substitute for new-model costs.
Fixed guard/corrector products are complete without these measured coefficients.

At500, a B serving query must exclude its complete own library episode
`suite_task_id*50+B-init`. Fits still include that episode, as R15 did. Official
A queries have no own B episode, even when the integer original index is equal.
Current shared `query(exclude_episode=...)` supports raw LOEO retrieval; P1
must add composed corrector/guard/knob LOEO and consume row-index stores before
B500 five-layer serving is admitted. Requests are in `REQUESTS/P6_to_P1.md`.

Smoke uses two episodes per selected suite, the same capture/build path, and a
maximum900s GPU deadline. It writes a tiny `smoke_parent` in its own campaign
directory, with no500 parent or size fits. Full commands require that smoke's
closed capture passes. Commands and disk ceilings are in `HANDBACK_P6.md`.

For a CPU finish/retry after a passing closed collection, use:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash exp/r17_integration/ops/library_campaign.sh --model dp_libero --suite libero_spatial \
  --finish /data/r17_library/dp_libero_libero_spatial/campaigns/EXACT_CLOSED_RUN
```

Ready subsets/fits are verified and reused only with matching seals/settings.

Library attempts now allow at most two infrastructure retries when a worker
errors with zero policy controls, no reported decision/ack rows, and no server
capture or input-intent for that UID/attempt. Ordinary unsuccessful outcomes
are retained once. Errors after a control or decision, unknown progress and
unreported episode timeouts stop the collection. A synchronous, library-only
input-intent ledger precedes model inference; it also fences a failed first
forward whose worker exception would otherwise lose partial progress. The
completed parent rechecks all closed captures and retains every eligible
pre-decision error in its sealed episode/parent provenance. Non-library formal
retry behavior is unchanged.

`--repair FAILED_CAMPAIGN` is an explicit live launch, like `--run`. It reads
the closed failed campaign, keeps every accepted non-error outcome (including
unsuccessful outcomes), and selects only missing UIDs. A fresh CPU seal must
match all six original teacher asset files byte for byte; checkpoint, norm,
weights, code, sampler, full YAML, benchmark, B archive and alignment are
pinned. The original semantic campaign/UID/paired seed/B coordinate is reused;
transport endpoints and worker capacity may change. Original attempts seed
the generation fence, so the two-retry limit also applies across repair.
Original files are read-only; new captures and a derived `merged/` collection
live in a new campaign. Closed capture manifests and every payload file are
SHA-checked before merge and normal parent/subsets/fits publication.

Repair refuses any original post-decision/control error, missing dispatched
worker result, duplicate non-error outcome, exhausted retry budget, changed
inputs or unverified remote capture. Historic campaigns without input-intent
receipts permit only the known zero-step EGL framebuffer initialization error,
with no cost/ack/capture rows. H100 repairs retain the bounded shard pump and
execution-host lock; parent/fits still build only on weilandserver. No running
campaign is changed. Run local DP repair only after the 4090 and its dedicated
library ports become available; `--finish NEW_REPAIR/merged` can repeat a CPU
finish after a successfully collected/merged repair.
An interrupted fit directory is preserved; move that exact partial aside before
retrying. Add `--ir-costs /path/to/measured_profile.json --miss-sampler
/path/to/sealed_sampler.json` to finish later knob qualification on CPU.
Published READY arrays and existing conflicting data are never overwritten.
