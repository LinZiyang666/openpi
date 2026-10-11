# GPU fix 2 — DP production B1 only; native pass criterion changes

P4 chose owner-recommended (a) after the failed
slot_G2_dp_libero_A_20261009_194429_754466. Production B>1 is disabled in
adapter stage hooks, batcher, provider/factory and serve CLI; all three batch
capabilities are unsupported, reason: GPU batch composition changes outputs.
Concurrent connections remain supported with B1 forwards. Stock core is usable
only at max_batch_size=1; hot changes above1 reject before dispatch/forward.
Explicit labelled offline diagnostic clones cannot register/serve production.

**Native pass criterion change:** `real_B1_B2_sampler_equivalence` is replaced
by `real_production_B1_only` (unsupported batch metadata, B2 rejection, exact B1
queue/direct output and RNG). Probe still has13 passing checks/domain. B2
composition measurements are retained under `diagnostics`, labelled
diagnostic_only; their numerical difference no longer gates production B1.
Finite outputs/private RNG remain checked. Do not count these measurements as
batch support/parity certificates, and do not admit B>1 based on a tolerance.

One owner-authorized GPU measurement,22.026s, used your original sealed load
and real paired NPZ without modifying assets. B1 vs B2/B4 end-to-end normalized
max_abs: K1001.078e-4/1.294e-4, K2 3.639e-4/1.210e-3. RNG states were exact;
vision conditioning also shifted. Full measured receipt/tensors are under
models/dp/samples/gpu_fix2_measurement/. This is a diagnostic of the pre-fix
GPU-fix1 release, not qualification of the new release or a quality/cost claim.

Probe/adapter source changes require another asset ModelSpec/model pin and
runtime/payload reseal before retry. Keep production batch-size1 in native,
concurrent, recipe/cache/teacher experiments; throughput batching is unsupported.
Full CPU suite passed153 and legacy/default-off49, no skips. Three current
strict475 native CPU probes passed13 each. Updated official paired256 descriptor
is models/dp/samples/gpu_fix2_q_libero_seal/load.json; its current model pin is
sha256:67331d0af13c19fc3e993e5a89f81a1c89469e68d66049f0ec26122de203a022.
Probe preserves the complete sealed descriptor except device. Final receipts
and commands are in HANDBACK_P4.md, GPU fix2. No further GPU run is planned.
No qualification files or running processes were edited/touched by P4.

Final request scan consumed your appended “GPU fix3 — separate DP-LIBERO A
GPU batch gate failure”: it cites the same failed run. Owner authorized and
recommended the B1-only alternative; P4 selected it, so no production GPU batch
tolerance profile is admitted. Native/reference/debug/suffix equality gates remain exact
and unchanged; a native one-bit mismatch still fails. B2 is an unsupported,
labelled diagnostic only, so there is no measured/unmeasured production batch
profile to consume. Your dependent gates should remain held until fresh sealed
B1 qualification passes. P4 has exhausted the one authorized GPU measurement.
# 2026-10-10 Owner-admitted DP batching

Owner now permits small arithmetic differences with matched batching across
control/arms. Explicit DP-LIBERO BatchConfig admits1..32 for DDPM100 and trailing
eta0 DDIM10/5; default-off remainsB1. models/dp/BATCHING.md and
models/dp/samples/dynamic_batch_summary.json contain the measured handback.
Model identity nowed3ae8c6...; fresh formal seal required (native sources changed
from GPU fix2). Qualification files were not edited. Actual final100000 strict475
weights and Q's real paired256 NPZ are consumed; source hashes match CPU seal.
CPU182/0skip+legacy49/0skip pass. Native B32 private generator receipts352 requests
equalB1. Largest serving committed-action drift0.000345707. Small-frame timings
do not establish SR. Official50 baseline command is prepared, not executed.

Local GPU budget was1199.589115s/1200s underflock, all own contexts closed.
35/36 requested concurrent timing cells measured; **K5/r2/max32 remains empty**.
Final CPU-preload attempt proved no initialized CUDA beforeactivation, then hit
its4.8s GPU cutoff before readiness. Please allocate a later slot for that one
cell; no runtime/parity criterion is silently relaxed. The previous strict B1
native reference/capture/suffix gates still apply. Active batching is a separate
explicit runtime metadata policy; new matched-batch SR is coordinator-owned.
Latest timan108 Tether isOFFLINE; RAM preflight fails closed before launch.

## Final framework launch/test receipt

P3 published the normal DP max1..32 launcher/dispatch seam; preferred command
is now ops/formal_eval.sh in models/dp/BATCHING.md, no DP common-file edits.
Full CPU rerun185passed/0skip and legacy49passed/0skip includes native max8/32
parsing, cuda_graph forwarding and stage environment. Report regenerated with
those XML receipts and current shared launcher source hashes. Exact official50
four-suite args parsed only; no deployment. Default B1 math gates unchanged.

35/36 matrix status remains PARTIAL:K5/r2/B32 timing requires a later GPU slot;
1199.589s authorization fully used. Native B32 private RNG352 requests passes,
not a success-rate certificate. Fresh timan108 admission still exits64/OFFLINE.

