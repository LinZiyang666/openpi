# Handback requests for implementation

These are concrete offline implementation requirements for sol and a conditional pilot plan for the coordinator. **Do not start an online run:** the current candidates fail the B-dev gate. `OWNER_ADDENDUM.md`, `SPEC.md` and `GATES.md` take precedence over the original suggestion of a quick pilot.

## 1. Package a CPU-only training/refit pipeline

Use `neural.py --device cpu --objective action --epochs 80 --tag cnn_action_cpu` with the exact per-cell inputs described in SPEC. CPU training works; a GPU is not necessary. Preserve the direct CPU checkpoints and logs produced here. Earlier GPU-trained checkpoints are separate exploratory comparisons, not a justification for new GPU jobs. A GPU execution path must require an actual CPU-infeasibility explanation plus a fresh ≥12-GiB free-memory check, capped allocator and short bounded job.

The implemented starter code includes data extraction, row/subset checks, cell-specific teacher projection, full CPU training, symmetric and asymmetric offline retrieval, direct action and kernel alternatives, LOEO, B-dev evaluation and latency measurement. Reuse the equations, not arbitrary exploratory output paths. All model-selection evidence comes from existing B-dev; no B row outside the cell may train its encoder, metric, normalizer or corrector. Never import `QueryCell` or use harness-wide current-library sigma for this fit.

Produce a build manifest that binds exact library/subset, selected parent rows, image preprocessing, normalized state, teacher artifact, seed, optimizer, 80 completed epochs, model state, library student features and every layer's refit. Reject a checkpoint from another size even if it has the same architecture. Use CPU TorchScript only after parity across all ten task IDs and image/state perturbations; our timing trace is not a ready production export.

## 2. Finish library-only layers 1–3 refits

Primary: symmetric student features → same per-task AWM/early fit → same 16-NN kernel; action payload and parent row map unchanged. Refit confidence scales with complete own-episode donor exclusion. Refit only-no-progress representation access, corrector heads, LOEO distance attenuation scale and any still-used diagnostic calibrations from the same library. Preserve R10's correction channels, blend and step-0 exception. A same-shaped old head is not compatible merely by dimension.

Asymmetric diagnostic: exact teacher library keys plus student queries; retain the teacher coordinate system and update query-dependent calibrations using light pseudoqueries. Make its representation ID distinct. Keep it offline unless it independently clears the frozen gates; the economical pilot has one student variant.

Distinguish donor-LOEO from an encoder trained without the query episode in reports. The existing R10-style fit fixes the representation first. Do not report low in-library error as unbiased unseen performance. All fits and LOEO exclude cross-task donors and apply exclusions before top-k/distance reference selection.

## 3. Add the actual tower-skipping plugin path

Relevant existing boundaries: `closed_loop/plugin.py` `_ConnPolicy._osp_infer`, `_try_blind`, adapter `prepare/output`; AWM `_dist`; `_CorrectedCache._correction`; the progress-span guard; R11 `history` and ledger. Light HIT must be decided **before** falling through to the policy's ordinary stage-1 path. Changing the retrieval class after stage 1 does not save tower work.

Introduce a feature-aware look query and real-look history. Both images are available from the current observation even though teacher token access is disabled. Preserve normalized state and output transforms exactly. Reuse cached/policy-tail blind handling, execution acknowledgments and per-connection locks. Maintain correct episode reset and row-space mapping. Retire disabled token-dependent guard diagnostics from the execution path or explicitly mark them unavailable.

HIT: encoder → retrieval/guard/corrector/knob → output transform, with zero tower calls. MISS: exactly one full policy call with current images/state and no second cache search or provisional commit. Blind continuation: neither image encoder. No teacher-refresh probes, fake teacher keys, fallback server or extra policy call hidden as diagnostics. The default non-QK path must remain identical.

## 4. New ledger and observability

Separate logical observation refresh from expensive work. Required per-decision fields:

| Field | Meaning |
|---|---|
| `look` / `has_look` | Fresh raw-image observation used by retrieval/guard; this advances the anchor history. |
| `key_kind`, `representation_sha256` | Teacher, light symmetric, light asymmetric, or none on blind continuation. |
| `light_encoder_ran`, `light_encoder_ms` | Actual student call and end-to-end key latency. |
| `teacher_tower_ran`, `teacher_tower_ms` | Actual full tower work; expected only on policy calls in QK mode. |
| `policy_full_call`, policy remainder timing | Real served full policy call, preserving model-specific dose. |
| `hit`, `source`, donor parent rows, weights, distance ratio, guard reason, fixed cap | Existing decision semantics and full audit trail. |

For the specified path, `Q = light_encoder_calls`, `T = teacher_tower_calls = M = full_policy_calls`, and `IR_new = (epsilon*Q + M)/N`. The conservative projection uses `Q=V_old`. Store Q/T/M/N explicitly. Old mode remains `(c_v*V+c_m*M)/N`. Do not charge `c_v` for a logical light look, omit the tower on a MISS, or charge it twice. If future forced-policy bypass makes Q<V, report that measured count rather than retaining the projection assumption. Nominal controls are `5*N`; actual terminal controls are separate.

Keep `hist_has_look` distinct from the old cost field named vision. Any legacy alias used for controller compatibility must be documented and must not drive compute accounting. Recording must be B-only, accept light features without fabricated teacher keys, and label any extra shadow work separately from the served-policy ledger.

## 5. Required offline verification before requesting coordinator approval

1. Dataset tests: exact subset/row membership, camera order/resize/state parity, no own-episode donors, B init disjointness, rejected wrong-cell or partial artifact. Instrument fit reads so A/queries, unrelated library rows and B-dev arrays cannot be consumed by a training optimizer.
2. Mathematical replay: library and live feature parity; early/stale metrics, all 16 neighbors, stable ties, synthesis weights, correction and parent mapping. Compare CPU eager/export on all tasks and nontrivial inputs. Reconcile library sigma to selected actions.
3. Real CPU plugin replay with fake model/tower hooks: tower raises on light HIT; one call on MISS; zero student/tower work on legal blind continuation; no double commit; policy-tail/reset/failed receipt/duplicate request tests. Preserve default teacher-path output and log semantics.
4. Ledger fixtures for HIT/MISS/blind/forced-MISS and an epsilon=0 thought experiment. At all-policy looks, the conservative new formula is `M/N + epsilon*Q/N`, so it cannot create fictitious savings. Compare counted work to instrumented calls.
5. Re-run all-row library LOEO, then the same accepted unseen B-dev recordings, with refitted layers where relevant; publish per-task/episode results and gate report. Existing key-only results are a baseline for comparison, not a waiver.

## 6. Coordinator-only next action, conditional

Once the offline report says **pass** and the coordinator approves, prepare the SPEC paired π0.5 L10-50 off/ir25 pilot on B identities outside size 200. Keep the current fixed caps and host/seed matched. Reuse a baseline only after exact equivalence is proven. L10-200 follows only if the first cell passes. Do not launch a GR00T arm without exact build-library RGB recovery. Do not use September GR00T traces: their launcher selects A-pool. Do not run a size-500 unseen-B claim; no such complement exists.

For now the useful next implementation milestone is a complete offline CPU artifact and a gate report, not a closed-loop run. No message, deployment, server, chain, remote operation or test-A emission has been performed by this exploration.
