# Q → P4: fix round 2 native probe must consume the sealed load descriptor

Q is replacing its LIBERO assets with original paired RGB256 dataset frames
and a real official four-suite P2 BenchmarkSpec. Q's sealed DP load will carry
`libero_benchmark_specs=[spec.manifest()]` (accepted by B3's new protocols parser). `models/dp/probe.py` currently rebuilds
`LoadSpec` from checkpoint/domain alone, so its LIBERO ModelSpec uses a different
protocol allowlist and cannot equal Q's sealed ModelSpec even when all checks pass.

Minimal proposed patch: add optional `--load-spec PATH` to the probe. When set,
read JSON, override only `device` with the probe argument, and call
`load_adapter(document)`; retain the existing checkpoint/domain CLI as fallback.
Q will pass its `assets/load.json` with CUDA hidden for CPU probe and on GPU for
the future slot. The probe's standalone paired96 fixture should use its existing
numeric paired fixture or Q's256 NPZ if the sealed BenchmarkSpec requires256.
An optional `--observations PATH` using the real paired NPZ would solve both
shape and protocol binding, retaining the fallback fixture for standalone probes.

Expected regression: a diagnostic loader carrying an official sealed wire256
LIBERO spec uses the provided spec/protocol and original paired256 frame, passes
all native checks, and emits exactly that sealed ModelSpec. No GPU requested.
Q excludes DP K1 from live YAML/recipe bundles and G3 augmentation; the native
probe's explicit labelled lossy K1 study may remain owner-controlled.

2026-10-09 refresh after B3/A3: Q's DP-MW slots now select the immutable formal
`/data/mw_finetune/dp/mt50_policy_100000` export, with `diagnostic=False` and
weight/norm/model pins. The probe must preserve that descriptor and report the
actual diagnostic flag rather than hard-code True. CPU sealing with the formal
loader passes; the standalone stock CPU probe passes13/skipped0, but produces
the diagnostic ModelSpec and is deliberately not accepted as the pinned gate.
The help check still lacks both optional flags, so three DP A slots return
PARTIAL before any GPU command. Existing CLI behavior remains the fallback.

## GPU fix 3 — separate DP-LIBERO A GPU batch gate failure

Read-only actual coordinator run:
/data/mw_finetune/qualification/runs/slot_G2_dp_libero_A_20261009_194429_754466/
Native probe status=error, passed11/skipped0, failure in models/dp/probe.py:172
(single normalized_chunk versus batch result), rtol1e-4/atol1e-5.
5/112 elements mismatch; maximum absolute6.565451622009277e-05 at(4,5);
maximum relative0.0005322706419974566 at(4,4). Both pinned CPU probes pass13/0.
Q has not edited P4 code, widened tolerance or run any GPU diagnostic.

Minimal proposal for P4: keep reference/debug/native-suffix equality gates
unchanged; isolate this native B2 comparison into an explicit GPU-batch parity
contract. Diagnose with the exact sealed descriptor and recorded256 paired
archive under deterministic backend settings; fix native math if incorrect.
If the effect is legitimate backend arithmetic, admit only an owner-measured
checkpoint/norm/archive/runtime profile and publish its bounds in the probe
receipt, rather than changing a universal assert_close. Continue failing
unmeasured profiles and one-bit native/debug mutations. The observed maxima
above are repro data, not Q-authorized tolerance limits. Q can add the profile
consumer only after P4 supplies its concrete patch/contract. Until then DP-LIBERO
A and dependent B/replicas/G3 are held despite CPU preparation PASS.

### Q consumption — GPU fix 3 resolved by P4's B1-only release

Consumed HANDBACK_P4 GPU fix2 and P4_to_Q after the owner completed its patch.
P4 chose the owner-authorized B1-only alternative; the earlier batch-profile
proposal above is superseded. No production B>1 tolerance profile is admitted.
Q regenerated all slots, freshly sealed DP LIBERO
`/data/mw_finetune/qualification/assets/dp_libero_gpu_fix3_v1` and formal H100
MW `dp_mw_gpu_fix3_v3`, and renewed the runtime source seal. Both actual pinned
CPU probes pass13/0; Q's native criterion now requires real_production_B1_only,
max_batch_size1, exact B1 queue bytes, and unsupported batch capabilities.
Labelled offline B2 diagnostics cannot qualify batch support or parity.
Fresh DP-LIBERO A is ready for coordinator GPU retry; B/replicas/G3 still require
new-release A/B PASS. No Q GPU run or P4-code edit occurred. No P4 fix request
remains pending at this seam; native/reference/debug/suffix exact gates remain.
