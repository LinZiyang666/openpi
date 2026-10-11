# DP pure-cache study admission (2026-10-10)

CPU model admission is complete for official A (`libero_spatial`, `libero_10`)
with the real B500 teacher views. All eight recipes use P6's existing factory,
fitted-array loader, R6/Q3 controller, resident bundle manager and driver.
No replacement retrieval, corrector or stall math was added in DP.

Match the current control `hs5dp_ctl_142812`: local 48 GB 4090, **two replicas,
max batch eight, 5 ms window, MPS, native DDPM100, group-commit driver,
timan107 ×32 workers at 6144 MiB each**. The earlier one-replica batching
recommendation is historical evidence, not this study's launch treatment.

| Groups | Commit | Corrector | MISS policy | Action-head NFE |
|---|---:|---|---|---|
| c / cr | 5 | off / on | never | 0 |
| ch / chr | 10 | off / on | never | 0 |
| cg / crg | 5 | off / on | calibrated R6/Q3 stall only | 100 only on stall MISS |
| chg / chrg | 10 | off / on | calibrated R6/Q3 stall only | 100 only on stall MISS |

The resident still starts with plain `full.yaml`. P6 subsequently publishes
the 16 suite/group YAMLs and selects them at episode boundaries. The same model,
stage batch queues and prewarmed graph signatures remain resident. Startup
`cache_enabled=False` metadata describes that plain starting configuration;
`cache_study_admitted` declares the additional validated study path.

## Compatibility and fail-closed checks

`library.py` attaches an explicit `AdapterProvider.library_validator` on a
frozen provider copy. Registration does not read production assets or change
provider/ModelSpec identity. Compatibility is checked only for an enabled
library recipe. The receipt uses `r17.teacher_compatibility.v1/PASS`, exact
teacher/resident retrieval identities and a content-bound evidence object.

The audited teacher identity is
`sha256:b3eaa7cc24777f398042e08b6ee0b6b9aeefbf67bcbb3a9edc4558133fe2b3c8`.
The current CPU-sealed resident identity is
`sha256:bd1dd4c1b2e1bfbb4c660ed8ec0a50362f461dc8a631c16edfa9d3097879d0c5`.
These identities differ because adapter admission/serving/batching code was
added after capture. `library_release.py` pins 19 producer sources, 10 weight/
metadata/normalization/CLIP assets and six dependency versions. All E native
producer bytes are unchanged; the current staged adapter is also pinned and
tested against that native implementation. The accepted control's corresponding
producer and asset pins match exactly. Remaining model fields, feature policy
digest, normalization, cameras, preprocessing, geometry and output transform
are checked explicitly; unknown teacher code or changed producer bytes fail.
P6 separately verifies the B-library → official-A physics/task/prompt bridge.

`admission.teacher_binding` verifies `sealed_store_descriptor` metadata,
including actual size-500 `VIEW.json`/row indices, without opening teacher RGB.
The view digest must match; the parent digest cannot stand in for it. K1
teacher rejection examines the actual sealed sampler, including other eta.

## CPU evidence

| Check | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Full DP suite (`cpu_cache_study_full.xml`) | 201 | 0 | 0 |
| Stock legacy/default-off (`cpu_cache_study_legacy.xml`) | 49 | 0 | 0 |
| P6 cache framework (`cpu_cache_study_framework.xml`) | 33 | 0 | 0 |

The real WebSocket/S07 regression executes eight clients × four calls × eight
groups, on one resident PID/generation with weight reload forbidden. Native
small-model E math and a sealed synthetic library isolate the hook path; its
input frame has real paired LIBERO views. CPU window 20 ms ensures stage1 B8
forms; production retains 5 ms. The raw ledger and summary are
`samples/cache_study_batching{.jsonl,_summary.json}`: 160 HITs, 48 library tails,
32 native stall MISSes, 16 policy tails; NFE=3200, entirely in gated groups.
HIT stage counts are `[1,0,0]`, tails `[0,0,0]`, native MISS `[1,1,100]`.
Stage1/2 reached B8; stage3 reached B7. This is CPU correctness evidence, not
a GPU throughput or success-rate receipt.

`samples/cache_study_cpu.py` strictly loads the actual 475 tensors on CPU using
P6's sealed descriptor with only device overridden. One real paired 256×256
frame from each B500 suite proves exact staged/native conditioning, DDPM100
wire actions at seed18 and the two-stage output transform. All 16 real fitted
recipes load; calibrated Q3 reports warming-up/inactive on these fresh anchors.
The suites contain 12,028 and 31,546 selected rows respectively. Stored GPU-B1
versus CPU visual-key max differences were 0.000889719 and 0.000669837; state
keys were exact. Those measurements are diagnostics, not an SR acceptance
bound. See `samples/cache_study_real_cpu.json` for complete producer receipts.

P6's unmodified `smoke.sh dp --prepare-only` passed, CPU only, exit0. Campaign:
`/data/mw_finetune/eval/mps_dp_libero_1010_164432_2c1f7e32`.
It sealed all 16 YAMLs, the 32-episode smoke plan, current runtime and worker
imports/pools; `dispatch_started=false`. Read-only worker RAM admission is
**false** while the existing horizon/step fleets occupy timan107. CPU preparation
PASS does not authorize a launch. Receipt: `samples/cache_study_prepare_summary.json`.

## Coordinator GPU smoke (not executed)

After the existing campaigns release GPU/worker RAM, rerun admission and use:

```bash
cd /home/weiland/projects/openpi
flock -n /tmp/r17_gpu.lock bash exp/r17_integration/ops/cache_study/smoke.sh dp --run
```

Budget **15–25 minutes estimated**, including startup and 32 real episodes over
16 suite/group barriers. No measured GPU duration is available; P6's script
has a 120-minute overall ceiling. This mechanical smoke has two active episodes
per barrier, so it does not establish full-study batch fill or SR. Full study
uses all 50 official A states per task, with the same R2/B8/5 ms/MPS/driver.
The coordinator must require bundle acknowledgments, one resident generation
per replica throughout, actual terminal execution acknowledgments, and P6's
HIT/tail/stall/NFE batch-accounting audit. Ungated NFE must be zero; every gated
policy call must have stall evidence and exactly100 native action-head NFE.

No GPU/simulator episode was run by P4 for this cache task. The formal GPU
graph/cache/MPS combination, actual calibrated stall trajectories, matched SR
and complete A500×16 outcomes remain coordinator/P6 qualification work.
