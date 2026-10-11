# R17 cache study CPU audit

Owner rulings in `exp/offline_search/rounds/r17/cache/OWNER_RULINGS.md` supersede
the initial pre-registration's B-evaluation, legacy no-progress guard and R1
resource proposals: evaluate official A500, fit B500 only, use R6/Q3 calibrated
stall calls with zero ordinary lottery, and retain each historical control's
replica/batch/MPS/worker configuration. No GPU work was performed.

| Gap / location | Change or remaining owner |
|---|---|
| `common/recipe.py:172`: parent checksum/load and online fit, no sealed size500 VIEW/fit admission | Fixed: SHA-bound fitted path/digest, VIEW identity, complete-episode selection, C-specific heads, immutable caches. RGB remains in original B parent; metadata and fitted arrays alone travel to serving. |
| `common/config.py:32`, `common/serving.py:350`: a missing/forced proposal could fall through to a real model call | Fixed: `miss_mode: never` rejects ordinary fallback. `stall_only` accepts only a diagnosed Q3 stall call. Periodic gaps, target IR and the legacy guard are forbidden for these modes. |
| `search/cache_recipe.py:239`: old NoProgressGuard differs from owner R6/R7 stagnation gate | Fixed: original frozen Q3 estimator/tracker and R6-C-v2 call/LOOK/cooldown functions; real executed controls, one fresh-anchor cooldown after a stall call, no lottery. Teacher B successful-trajectory LOEO calibration; all ten task gates available in all ten artifacts. No independent new B validation rollout was collected. |
| `common/recipe.py:214`, `common/layers/recipe.py:311`: frozen fit with corrector off/on must share retrieval but toggle heads | Fixed: shared immutable metric/actions, connection-local `use_corrector`; actual applied/computed correction activity distinguished when a gate discards the cache proposal. C5 gets its own head, never a C10 prefix. |
| DP size500 VIEW admission | **Fixed by P4, CPU admitted**: verified VIEW descriptor retains native DDPM100/K1 rejection. See HANDBACK_P4.md Cache study framework integration. |
| DP/Fast teacher vs current resident identities | **Fixed by P4/P3, CPU admitted**: model-owned immutable-provider hooks validate exact native producer/weights/norm/content and both retrieval identities. See their cache-study handbacks; GPU serving qualification remains pending. |
| Cosmos recipe admission | **Fixed by D, CPU admitted**: explicit K5/C16 shared-recipe path, native key/wire checks, source/content compatibility and four alias groups. Baseline legacy driver retained. GPU qualification remains pending. |
| `search/cache_recipe.py:87`: B library vs official A protocol | Fixed: explicit sealed source/serving LiberoBenchmarkSpec pair; identical physics, task order, prompts, image/state/control/seed semantics. Only cohort lists/protocol label differ. No unconditional identity bypass. |
| `common/provider.py:139; common/serving.py:316`: per-YAML commitment vs stock transport | Fixed/CPU checked: DP/Fast C5/C10 fits, same R5 transport and zero-NFE tails; framework already validates partial commitments. Cosmos C16 long labels alias short labels. DP/Fast CL10 is provisional pending registered NI freeze. |
| `ops/formal_eval.py:1148`, `ops/formal_driver_fast.py:142`: one fixed bundle/plan | Fixed: publish all study bundles through the stock proxy before dispatch, content ACKs, suite/group barriers, one resident fleet per model, per-episode immutable bundle identity, existing durable GroupCommit. No replica restart to switch arms. |
| `benchmarks/execution.py:59`: cost allowlist drops gate/corrector/batch evidence; final server ACK not persisted | Fixed in P6 process-local `ops/cache_study_worker.py` shim; stock worker/renderer loop unchanged. Explicit flag applies across worker recycling; final ACK mismatch fails. Permanent optional P2 interface requested separately. |
| `ops/formal_eval.py:1305; ops/cache_study.py:327` batch audit assumes all policy stages | Fixed opt-in study audit: library anchors run stage1 only; library/policy tails run no stages; diagnosed stall MISS runs the native full sampler. Actual P3/P4 S07 CPU adapters tested at B16/B8 caps. GPU numerical compatibility remains model-owned. |
| `ops/formal_prestage.py:80`: frozen runtime omits Q3 source dependencies | Fixed: append only three original read-only R6 math modules to the study runtime, with immutable archive pin and CPU import regression. No R6/R16/harness edits. |

`ops/cache_study.py audit` uses all six real B500 stores and all 48 frozen
recipes, checks actual fit/query geometry, donor results, corrector/gate
diagnostics and explicit B→A admission. This proves CPU math with teacher
identities. It does not attest current model releases or new GPU quality.

YAML set: `yaml/{dp_libero,fastwam_libero,cosmos_libero}/{sp,l10}/` contains
`c, cr, ch, chr, cg, crg, chg, chrg.yaml`. The generated templates bind B
teacher identities; formal CPU sealing emits resident-bound versions into
the new campaign's `assets/cache_yamls/` and refuses unqualified model hooks.
Cosmos full execution uses four physical groups, with four documented aliases;
smoke switches through all eight labels. Full plan counts are 8,000 DP, 8,000
Fast and 4,000 Cosmos episodes, all A, giving 20,000 independent episodes.

CPU preparation and review:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m exp.r17_integration.ops.cache_study audit \
  --study exp/r17_integration/ops/cache_study/yaml/STUDY.json \
  --output exp/r17_integration/tests/ops/receipts/CACHE_STUDY_REAL_B500_AUDIT.json
bash exp/r17_integration/ops/cache_study/smoke.sh dp --print
bash exp/r17_integration/ops/cache_study/smoke.sh fastwam --print
bash exp/r17_integration/ops/cache_study/smoke.sh cosmos --print
```

Scheduled GPU smoke commands, **not executed**:

```bash
bash exp/r17_integration/ops/cache_study/smoke.sh dp --run
bash exp/r17_integration/ops/cache_study/smoke.sh fastwam --run
bash exp/r17_integration/ops/cache_study/smoke.sh cosmos --run
```

Run Fast and Cosmos sequentially on H100. The script uses the launcher's host
GPU exclusivity; there is no local H100 `flock`. DP uses local 2×B8/5ms/MPS,
32 timan107 workers, budget40GiB, ports23400–23419; Fast uses H100 2×B16/5ms/MPS,
48 timan107 workers, budget70GiB, ports23420–23439; Cosmos uses H100 2×B1,
8 timan107 workers, budget20GiB, ports23440–23459. These match the accepted
controls, including reservations/recycling. Existing campaigns are untouched.

Each smoke plans **32 episodes/model**: two episodes for every group on each
suite. `STUDY_BUNDLE_ACKS.json`, closed resident ready/PID evidence, durable
driver journals and `SUMMARY.json` must verify 32 complete, non-error results,
zero ordinary MISSes, correct native NFE/stage composition on gate calls,
corrector on/off, gate state/LOOK/call evidence, final execution ACKs and a
constant resident generation per PID. `activity` records what occurred;
zero gate calls or correction deltas mean that activity was **unexercised**,
not proven. Do not add outcome-selected A episodes or claim an unobserved call.
Thirty-two episodes plus cold loading may exceed the old ≤15-minute slot;
the command has a bounded two-hour cleanup deadline, not a speed claim.
