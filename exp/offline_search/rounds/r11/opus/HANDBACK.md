# R11 opus hand-back (schedule-type IR knobs + shared IR accounting model)

Written 2026-10-02 ~21:20 CDT. Read SPEC.md first; it is the implementation contract.

## What sol implements

1. **`R11KnobRecipe(R10Recipe)` (or an equivalent mixin on the R8 judge).** In `query(q)`:
   - call `super().query(q)` first, which runs the unchanged 3-layer verdict;
   - if the guard forced a MISS, pass the result through;
   - otherwise apply the knob, setting `os_force_miss = 1` and `os_reason` ∈ {61, 62, 63, 64}.
   - **Never** rewrite `os_flags` or any guard / corrector memo. The template is R6 `_ExtraDose` in `rounds/r06/ideation_Q2/frontier/adapters/methods.py`.
2. **Methods** (SPEC §1–§3):
   - `random(ρ)`;
   - `periodic(K)`: guard-aware dithered gap cap;
   - `periodic_pgt1(K)`: periodic plus a one-anchor tail after each guard MISS;
   - `random_tail2(ρ)`: a random trigger plus one forced follow-up anchor.
3. **Keyed uniforms.** Use R6's SHA256 `uniform(key, seed, task, init, step, domain)`:
   - keys `'R11-random-v1'` and `'R11-gap-v1'`;
   - `seed = 0`;
   - periodic cap draws keyed by `last_call_step`.
4. **State.**
   - **Periodic:** `run`, `last_call_step` and `calls_before` are derived from `q.hist_has_vision` / `q.hist_hit` at every anchor, with no mutable state.
   - **Variants:** the two variants need one per-connection list of the steps where the guard (pgt1) or a reason-61 trigger (tail2) fired. Clear it in `reset(episode)` and clone it with the connection, like the guard memo.
5. **Artifact metadata.** Record method, setting, target, library v and g, `pred_IR_lib`, and the SHA of `out/arm_grid.json`. The settings are scalars per cell-size, taken from SPEC §5 (identical to `out/arm_grid.json`).
6. **Arms.**
   - 38 arms in the priority order of `out/arm_grid.md` / PREDICTION.md.
   - Each arm uses the R10 nested subset of its cell-size, the same plugin arguments as the R10 GC_dist arms (`--os-blind --os-policy-tail --os-judge guard_only`, full model, cost ledger) and the A 500-pair manifest.
   - One same-batch knob-off base per cell-size (8 arms; shared with astra).
7. **Audits** (SPEC §6), on decision logs:
   - ledger counts;
   - no knob call on a guard anchor;
   - knob MISS cadence (exactly one policy tail, then a fresh anchor);
   - periodic `run ≤ cap` and call iff `run ≥ cap`;
   - coin determinism;
   - realized IR vs `pred_IR_lib`.

## Files (all in `exp/offline_search/rounds/r11/opus/`)

| file | role |
|---|---|
| `SPEC.md` | implementation spec, calibration procedure, frozen settings |
| `IR_MODEL.md` | shared knob → owner-IR accounting model, assumptions, what is not estimable and how it is bounded |
| `OFFLINE.md` | LOEO knob curves, target solves, guard-rate stress, value evidence and its weight |
| `PREDICTION.md` | per-arm IR / SR predictions and 6 falsifiable statements, timestamped 21:10 CDT |
| `HANDBACK.md` | this file |
| `ir_model.py` | accounting model, frozen guard replay, knob schedules (`Random`, `DitheredGapCap`, `SigmaDelta`, `GapCap`, `FrontLoad`), `simulate`, `solve_rho`, `perturb_guard` |
| `loeo_anchor_table.py` | whole-episode-out 5-fold replay of the deployed retrieval on R10 subsets → `out/anchor/*.npz` (60 files; top-1 progress, d1, errors; joined with R10 opus k-fold head terms, exact match) |
| `offline_curves.py` | curves, target solves (common + proposed levels), stress, profiles → `out/curves.json`, `out/OFFLINE_TABLES.md` |
| `arm_grid.py` | frozen settings + library / informed IR predictions → `out/arm_grid.json`, `out/arm_grid.md` |
| `value_diag.py` | capture ratio, uncovered high-error streaks, cache-run stats → `out/value_diag.{json,md}` |
| `size_sensitivity.py` | guard rate vs episodes per task on the 500 libraries → `out/size_sensitivity.{json,md}` |
| `r6_overlap_check.py` | overlap-rule validation on R6 arm-level ledger aggregates (disclosed) → `out/r6_overlap_check.{json,md}` |
| `r10_check.py` | library base IR vs R10 published 3-layer IR (disclosed check only) → `out/r10_check.{json,md}` |
| `predict.py` | SR / IR predictions → `out/predictions.{json,md}` |
| `test_ir_model.py` | self-tests: the guard port equals the frozen `BlindMixedJudge._progress` on 300 random sequences; random closed form = simulation; gap cap bounds cache runs; no knob call on guard anchors (all pass) |

## Open questions for the coordinator

1. **Dev pilot (recommended).** Run the knob-off base plus the two core arms on about 50 B-pool episodes that are **not** in the subset library, for the four 50-cells and GR00T L10-200 (450 / 300 spare inits available). This measures the closed-loop guard rate, the only material unknown in the IR map.
   - The 500 cells have no spare B-pool inits. Their IR rests on the library model, expected within ±.02.
   - Allowed use of the pilot: check only, or (if the owner approves) re-solve ρ / K with the measured g. This would no longer be "library-only". It stays non-test, and it has to be disclosed.
2. **Settings provenance.** I recommend the frozen library-only settings of SPEC §5.
   - The "informed" prediction column uses R10's published test-A aggregate IR. It must not feed back into settings.
   - Expected consequence: π0.5 arms land about .005–.017 below target.
3. **Portable fit-time solve.** For deployments beyond R11, the guard replay can live inside `recipe/fitting.py: distance_scale`, which already serves every held-out fold. That loop uses fixed selected-library PCA instead of per-fold PCA, which changes g by about .01 (unmeasured). Optional; not needed for R11.
4. **Library-size bias of the guard rate.** On 50-cells the replay serves from 4 of 5 episodes per task, which biases g up by .002–.014 (`out/size_sensitivity.md`). The bias is predictable from the library itself (3 → 4 extrapolation). Not applied, because the effect is ≤ .006 IR. Apply it if the pilot shows a systematic undershoot.
5. **Astra cross-check.**
   - **Interface:** her state-dependent knobs plug into `ir_model.simulate` through a `Knob` subclass. Each `Episode` carries per-anchor `d1norm`, `risk`, `risk_raw`, `grip` and `guard`.
   - **Shared base rates:** her floor and ceiling should match my base v / g per cell-size (OFFLINE §1).
   - **Random floors:** hybrids with a random floor can reuse `Random` with the same key, to share common random numbers with my random arms.
6. **Arm budget.** 38 opus arms plus 8 shared bases. If the budget is tight, the minimum informative set is priority 1 (8 arms: random vs periodic at the middle level on the four 50-cells), then priority 2.
7. **REPORT.md** could not be written to disk because the harness blocked it as a "report file". Its full text is in the final hand-back message to the coordinator, who can save it as `REPORT.md` in this directory.
