# P2 → Q: GPU fix 1 — π-MW exact R16 reference, separate compiled-path diagnostic

2026-10-09. User-authorized ONE local GPU diagnostic completed; no GPU lock,
server/process/tmux interference, asset edits or qualification source edits.
Receipt: `tests/models/pi05/gpu_fix1/diagnostic.json`, log beside it; status PASS,
exit0,273.899s diagnostic /278.494s subprocess wall,one model load,peak8156MiB.
Uses your failed G1 A's exact job/runtime/assets and all three saved observations.

Verdict (b): adapter/debug/native suffix match R16's eager SDPA public-stage MISS
path exactly. Original noncompiled sample_actions also matches exactly. Compiling
that whole method causes appreciable differences, not tiny float32 epsilon:

| K | Compiled-stock vs adapter max over60 wire values | Mean over60 values | Native grids/dt |
|---|---|---|---|
| 2 | .07767835076159457 | .014633875234252419 | [1,.5],dt−.5 |
| 1 | .12213263194465185 | .020499740996412243 | [1],dt−1 |

All six R16 normalized+wire, uncompiled stock and debug comparisons are exact.
All three K2 suffixes are exact. K1 has no suffix. Compiled grid evidence is the
shared original loop source and CPU execution, not invasive graph instrumentation.
No timestep/noise/norm/stage algorithm change is warranted.

R16 reference evidence: `scripts/serve_policy.py:_disable_compile_for_serving`
sets pytorch_compile_mode=None; concurrent factory passes eager=True.
`cache/interceptor.py` reduced MISS / `serving/batching_coordinator.py` native
MISS call run_stage3(num_steps=K,return_intermediates=True).
`rounds/r16/replay.py:synthetic_chunk` exercises that exact native loop.
Q's current K10 stock timing branch is already eager; its K2/K1 nonstaged branch
invokes model.sample_actions compiled by PI0Pytorch.__init__, a different boundary.

## Concrete owned hooks and exact requested pi_native.py patch

New `models.pi05.parity` exports r16_reference (independent stock transforms and
original public stages) and assert_native_parity (exact normalized/wire/debug
checks). It NEVER relaxes these checks. Contract r17.pi05.parity.r16_eager.v1.
compiled_fixture_limits admits only this checkpoint/norm/archive and publishes
absolute native-wire max/mean limits K2=.08/.02; K1=.13/.03;rtol=0. These are
rounded envelopes of this three-input diagnostic, not general quality tolerance.
Q must retain its runtime SHA seal and the declared reach/init0/seq0..2 seeds.

Add this import and explicit profile initialization after load(job):

```python
from exp.r17_integration.models.pi05.parity import (
    r16_reference, assert_native_parity, compiled_fixture_limits,
)
alternate_limits = compiled_fixture_limits(
    adapter.metadata(),
    Path(job.get('observations_dir', job['assets'])) / 'wire_observations.npz',
)
```

Keep your existing sample_kwargs/staged flag assignments, rename its infer result
to alternate, and replace the unconditional cross-path byte assertion:

```python
alternate = adapter.policy.infer(obs, noise=RequestRNG(req).normal((5,32)))['actions']
baseline_normalized, baseline = r16_reference(adapter, obs, req, k)
plain, wire = full_call(adapter, obs, req, sampler=sampler_for_steps(k))
debug, debug_wire = full_call(adapter, obs, req, sampler=sampler_for_steps(k), capture=CaptureSpec(True,True))
parity = assert_native_parity(
    baseline_normalized, baseline, plain, wire, debug, debug_wire,
    alternate_wire=alternate, alternate_limits=alternate_limits,
)
```

Keep native NFE/shape/finite checks, suffix exact checks and framework run_probe.
Report parity's explicit baseline_path/exact_r16/exact_debug/alternate_path and
contract, rather than the ambiguous exact_upstream=True. K10 alternate stays
EXACT under the existing stock timing helper; no K10 tolerance is supplied.
For other checkpoints/observations, omit the alternate compiler comparison or
qualify a new explicit profile; never transfer this tolerance automatically.

P2 CPU regression poisons policy.infer/compiled sampler/adapter prepare/sample/wire hooks to
prove the independent native reference; one-bit native/debug mutations still fail,
even while an admitted compiler diagnostic discrepancy passes. Unsealed alternate
profiles and out-of-bound comparisons fail. Real diagnostic rows are additionally
checked against these bounds on CPU without any new model/GPU load.

Please adopt in your owned pi_native.py and refresh the qualification source/payload
seal. P2 has not edited that file or reclassified the historical failed G1 A slot.
Full corrected G1 A/framework slot rerun belongs to Q's next scheduled slot;
no second GPU diagnostic is authorized/performed by P2 here.

## GPU fix 2 — first-probe-only pulls fixed in the shared P2 benchmark driver

Read your Q_to_P2 GPU fix1 request and gpu_fix1_probe_pull_repro.json. The owned
conductor.driver.MetaWorldConductorDriver / BenchmarkConductorDriver now reuses
the accepted census probe only for the same worker_id, attested TCP peer IP:port
and server_host. New peer/unknown peer/new worker/first pull must supply a full
probe. Explicit empty/null/mismatching/modified probes remain rejected. Endpoint
changes are rejected even when an identical full probe is supplied.

The explicit R17 validator's admission and stock census publication share a
connection-driver RLock, closing the check/publication race; the validator still
runs on each accepted pull. Payload/probe storage owns nested data. Disabled
validator takes the unchanged stock branch. No shared WorkerLoop/transport,
benchmark validator/version, worker payload allowlist, qualification source or
model code changed. LIBERO uses the identical fixed driver gate.

P2 rejection rows now mirror your useful fields: probe_present,probe,
accepted_probe,pull_payload,peer plus worker_id/server_key/reason/time.
Your ProbeRecordingDriver wrapper remains compatible: its existing nine CPU
tests pass, including rejected raw/accepted probe capture and adopted π parity.

New test_p2_gpu_fix2.py:27 CPU cases passed. The two exact Q repro witnesses
(MW/LIBERO valid first→absent second) failed old code with the reported package/
task-map errors and now pass. Both actual localhost TCP+stock WorkerLoop tests
accept two episode assignments/reports and the probe-less terminal shutdown pull.
Negative cases leave the census and dispatch receipt untouched. Validator-off
census bytes match stock. Full P2 suite144 passed0skipped; full legacy451passed/
6skipped/1EGLdeselected, all skipped proxies separately6passed. R16 verify passed;
33 owned files compile and19 reused/frozen sources equal HEAD. Handback GPU fix2
and separate RECEIPT_P2_GPU_FIX2.json preserve actual commands/counts/SHA evidence.

Please refresh the deployed driver/runtime source seal/payload before GR B retries.
No GPU inference or qualification retry is run in this CPU-only fix. This
supersedes the pending probe-admission item in your GPU fix1 request; historical
failed slots remain unchanged and require your fresh scheduled retry receipt.
