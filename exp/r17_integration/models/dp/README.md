# DP-T R17 adapter

This opt-in package consumes agent E's reconstruction unchanged. It uses P1's
`r17.contracts.v1`, stock `CacheOrchestrator`, `WebsocketPolicyServer`, and
`BatchingCore`. Imports of the package register nothing and load no model.
Strict loading establishes compatibility with E's 475-tensor tree, not identity
with an unpublished author wrapper or benchmark quality.

**Cache study (2026-10-10):** real B500 VIEW/fit admission and all eight YAML
groups pass CPU checks on the resident framework path. See [CACHE_STUDY.md](CACHE_STUDY.md)
for exact compatibility evidence, the matched R2/B8/5ms/MPS treatment, CPU
receipts and the coordinator's unexecuted GPU smoke command. Worker RAM
admission must be rerun after the current campaigns finish.

**2026-10-10:** explicit DP-LIBERO batching now supports max1..32 for
DDPM100/trailing DDIM10/5 with private request RNG. See [BATCHING.md](BATCHING.md)
for35 measured configurations, the one unmeasured cell, and the guarded
official50 launch command. Defaults retain B1; older B1 restriction/receipt
descriptions below describe the default and historical releases.

```python
from exp.r17_integration.models.dp import load_adapter, register_provider, make_connection_factory
from exp.r17_integration.models.dp.config import LoadSpec
from exp.r17_integration.models.dp.serve import plain_config
from exp.r17_integration.common.serving import make_bundle_validator
from exp.r17_integration.common.config import build_shared_storage

# CPU is the loader default. Existing ML45 checkpoints are diagnostic only.
adapter = load_adapter(LoadSpec("/data/mw_finetune/dp/checkpoints/metaworld", "metaworld"))
provider = register_provider(adapter)
config = plain_config(provider, num_steps=100, commit_horizon=10)
storage = build_shared_storage(config)
factory = make_connection_factory(adapter, provider=provider,
                                  default_config=config, default_storage=storage)
# Pass factory + make_bundle_validator(provider) to the stock concurrent WS server.
```

The CLI is `python -m exp.r17_integration.models.dp.serve --load-spec SPEC.json
--port PORT`. `SPEC.json` contains `LoadSpec` fields; mutable `latest` paths are
rejected. `--cache-config` loads a real bound YAML; `--batch-size` defaults to1.
Explicit LIBERO batching uses `--max-batch 8 --batch-wait-ms 5 --batch-ledger
DIR/batching.jsonl`; B2..32 requires that membership ledger.
`--manifest-out` / `--yaml-out` export resident identity
and its default bundle. A hot-loaded YAML retains stock component schema and
uses `adapter_binding` / `adapter_native`; use a new bundle generation ID.

Native DDPM100 remains the default and is byte exact with E. Reduced K uses
trailing DDIM, starting at t99: K1 `[99]`, K2 `[99,49]`, K4 `[99,74,49,24]`.
K1 is **lossy** and rejected at binding by default and in cache MISS,
teacher/library, recipe and descent roles. Only a labelled plain step-reduction
study can enable it, at the native replan commit:

```python
study = plain_config(provider, num_steps=1, allow_lossy_sampler=True,
                     sampler_study_label="DP_K1_plain_step_reduction")
```

Both fields live in `adapter_binding`; metadata publishes
`r17_sampler_admission`. K2/K4/K10 retain their reduced-step admission.
Grid construction produces
exactly K points for every integer K1..100; native updates use the actual next
grid alpha even when K does not divide100. DDIM is an explicit
separate baseline: `LoadSpec(sampler_kind="ddim", ddim_eta=...)`, or
`adapter.with_sampler(SamplerProfile("ddim",100,eta))`. Eta, library version,
betas, variance/prediction/clip config and the complete integer grid are bound
into sampler identity. DDIM quality is unqualified.

The scheduler output is the stored internal-affine normalized action. Wire
decoding reverses internal affine then external quantiles exactly once. No
LIBERO gripper inversion or MW action clipping occurs in this package; clipping
remains the benchmark's applied-control operation. LIBERO has H16/action7/state8
and two real ReLU64 keys; MW has H16/action4/state4 and one real key. Its zero
second branch remains language-conditioned in the policy. Language and absent
branch caches are disabled, preserving E's math and honest stage costs.

Both E's MW `image`/`state` and P2's canonical `observation/image` /
`observation/state` are accepted. MW bundles bind `FORMAL_SPEC.identity`, not
only its human-readable name. Server metadata supplies `r17_model_spec` for
client/debug attestation. Formal loading requires E's complete export receipt
and explicit content pins; ML45 transfers cannot be declared formal MT50.

Snapshots save x_t **before** an interior native update, its loop index/native
timestep, model/norm/layout/schedule/conditioning identity, and the private
sequential generator state. `resume` validates these and executes that grid's
suffix. `warm_reset` uses the scheduler's `add_noise`. Floating warm points in
the stock batch envelope are opaque loop-index encodings; no flow arithmetic
is used. Flow-shoot is not applicable, and unsupported stock advanced warm,
CP2/RIT/trace/routing combinations fail closed in P1's generic executor.

Use `--debug-directory DIR` (P6 closed native writer) or
`--capture-sink module:function`, and optional `--recipe-factory module:function`
to deploy shared debug/five-layer hooks through the same factory. By default the
factory consumes P1's `r17.recipe.v1` bundle section: library content/teacher
sampler, corrector, guard, and periodic gap are independent of MISS K. Target IR
requires P1's calibrated cost profile; enabled recipes bind the complete
canonical task table and its identity. Disabled
recipes remain plain at every K. Shared R17 config/server modules wrap the
unchanged sealed stock sources. LIBERO inputs must be paired96 or raw256
uint8 RGB; legacy224 resampling is rejected and wire size is identity-bound.
`fixtures.libero_observation` reads the real independent paired cameras for probes.
`write_libero_qualification_fixture` / `read_libero_qualification_fixture`
seal and consume those recorded96 pairs verbatim as safe NPZ plus PNG/hash
provenance for Q. A missing wrist fails; training-frame mechanics are not
held-out SR or original256 qualification.
Native warm/reset and recipe/library capabilities remain implemented_unverified
for overall qualification, despite their tested CPU hooks. `DPTraceAdapter`
observes existing tensors and emits JSON + numeric arrays. `sample_variants` is
an explicit offline/twin hook: it accounts for every extra NFE and uses matched
private initial streams. It is never invoked by ordinary debug capture.
Snapshots and posterior state roundtrip through P6's closed NPZ/H5 writers.
Formal connections require full task/init/seed context and extended commit
requires `ack_protocol=r17.execution.v1`. Tail reuse requires the preceding
execution acknowledgment; commit is bounded
by H16 and the response is sliced at the configured horizon.

CPU receipts and native capture fixtures are under `samples/`;
`cpu_dynamic_batch_full.xml` and `cpu_dynamic_batch_legacy.xml` contain current
pytest evidence (185 +49 passed, zero skips). Real paired-frame concurrent GPU
timing covers35 configurations, with one missing cell. Real renderer/closed-loop
quality and cache/recipe GPU qualification remain pending. Commands and exact
ownership/status are in `handbacks/HANDBACK_P4.md`; older GPU-fix2 receipts are
historical releases.

Fix round2 introduced the K1 admission gate and paired fixtures. Its
`fix_round2_release_quality.json` and `fix_round2_qualification/` remain
historical quality/fixture evidence. GPU fix1 changes source-bound sample pins;
use the current `gpu_fix1_*` artifacts described below. Historical leading-grid,
round1/2 and B3 receipts remain for audit.

Formal LIBERO LoadSpecs supply `libero_benchmark_specs=[asdict(P2_spec), ...]`.
These bind the full suite/task table, original-init partitions, seed/limits and
wire resolution through P2's BenchmarkSpec identity. `libero_wire_protocols` and
`r17_sampler_specs` in server metadata let the worker select exact protocol and
native sampler bindings. The default diagnostic probe has no official LIBERO
package/task map and retains its explicit diagnostic protocol label.

The bounded native probe also accepts Q's sealed descriptor and recorded input:

```bash
python -m exp.r17_integration.models.dp.probe --load-spec SEALED_LOAD.json \
  --observations WIRE_OBSERVATIONS.npz --device cpu --output OWNED_RUN
```

Only `device` is overridden. Checkpoint, domain, diagnostic/formal mode, protocol,
sampler and all pins remain from the JSON; receipt mode comes from the loaded
adapter. Optional fallback checkpoint/domain arguments must agree with it.
Without `--load-spec`, the existing checkpoint/domain CLI is unchanged.
The NPZ uses aligned leading observation axes and Unicode prompts, loaded with
`allow_pickle=False`. Native checks use row0 verbatim; LIBERO requires its real
wrist key, and the B2 check keeps the existing prompt-variation study. Without
`--observations`, the paired96 LIBERO and diagnostic MW fixtures remain.

GPU fix1 preserves reviewer B3's protocol parser/profile checks. The probe source
edit changes source-bound pins: Q must reseal its assets/runtime before retry.
Old `review_b3_release_*` samples remain historical; current standalone pins are
under `samples/gpu_fix2_release_*_cpu/`, and the official paired256 example is under
`gpu_fix2_q_libero_seal/` + `gpu_fix2_q_libero_cpu/`. Do not silently remove pins
to make an old descriptor load. GPU work remains coordinator-owned.

GPU fix2 disables production B>1 in all three stages: a real4090 measurement
found output drift with batch composition despite identical private RNG states.
Batch capabilities are unsupported and `r17_batch_policy` states max1. Concurrent
connections still use P1's shared executor; every forward stays B1. Optional
stock core injection requires max_batch_size1, with per-submit and stage guards
against hot changes. The serve CLI rejects larger sizes before model loading.
Cache HIT/MISS/periodic/recipe arms must all use B1 so queue composition cannot
change keys/conditioning/actions or masquerade as an experimental effect.

`adapter.for_batch_diagnostic("nonempty label")` creates an offline clone for
explicit batch studies; it cannot register/serve production and does not promote
capabilities. Native probe keeps B2 differences under labelled `diagnostics`,
with magnitude/RNG/finite checks and the old tolerance mismatch count. Its
passing gate is now `real_production_B1_only` plus exact B1 queue/direct parity;
the diagnostic has no numerical parity acceptance tolerance. The one22.026s
pre-fix GPU measurement and tensors are in `samples/gpu_fix2_measurement/`.
The GPU fix2 release itself has no GPU qualification receipt yet.
