# R11 layer 4: schedule-type IR knobs (opus): implementation spec for sol

Frozen 2026-10-02 ~21:25 CDT, before any R11 closed-loop result.

**Scope:**
- the owner's random miss and periodic miss;
- two schedule variants: a post-guard tail, and a two-chunk policy segment after a random trigger.

**Out of scope:**
- the state-dependent knobs, which belong to astra;
- the shared IR accounting model, which is in `IR_MODEL.md`.

**Reference code:** `ir_model.py` contains `Random`, `DitheredGapCap` and `simulate`. Calibration is in `offline_curves.solve` and `arm_grid.py`.

Nothing here is task-indexed. Every knob parameter is a scalar per cell-size (model × suite × library). Task and init enter only as the identity of the episode in keyed coins.

## 0. Common contract (all knobs)

**Where it sits.**
- The knob runs inside the deployed judge's `query(q)`, i.e. at every vision decision (an "anchor").
- It runs **after** the frozen R10Recipe verdict (R8 only-no-progress guard plus the GC_dist corrector), exactly as R6's `_ExtraDose` mixin does in `rounds/r06/ideation_Q2/frontier/adapters/methods.py`.
- Suggested form: a thin `R11KnobRecipe(R10Recipe)` whose `query` calls `super().query(q)` and then applies the knob to `extras`.
- The knob never runs at blind slots or policy-tail slots.

**Guard first.**
- If the guard already forces a MISS (`extras['os_force_miss'] == 1`), the knob does nothing at that anchor.
- Guard flags, `os_reason`, `os_flags` and the guard's progress memo are **never rewritten**.

**What a knob MISS executes.**
- Set `os_force_miss = 1` and `os_reason = 61` (random), `62` (periodic), `63` (post-guard tail) or `64` (random two-chunk follow-up), and keep `os_flags` unchanged.
- The plugin then executes the MISS exactly like a guard MISS:
  1. stage 2/3 policy chunk;
  2. the committed ten-control policy tail under the C10 `policy_tail_gate='lifecycle'`;
  3. a fresh anchor at the next vision decision.
- No extra LOOK, no cooldown slot, no shadow forward.
- Only the random-tail-2 variant (§3) forces more than one consecutive anchor.

**Corrector and guard state after a knob MISS.** Unchanged from a guard MISS:
- the plugin invalidates the cache anchor;
- the next anchor is a fresh retrieval with `prev_hit=False`, and the corrector applies its normal distance gate;
- the guard's progress memo records the knob-MISS anchor's top-1 progress like any query. R6's `_ExtraDose` did the same, and the R6 ledgers validate the resulting accounting (IR_MODEL §4).

**Audit extras** (every anchor):
- `os_r11_version`;
- `os_r11_method` (1 random, 2 periodic, 3 periodic plus post-guard tail, 4 random tail-2);
- `os_r11_guard` (1 if the guard forced this anchor);
- `os_r11_eligible`;
- `os_r11_knob_call`;
- the method-specific state (coin u, run, cap, calls_before);
- `os_r11_setting` (ρ or K).

**Keyed uniforms.** Use R6 Q2's `uniform(key, seed, task, init, step, domain)`: a SHA256 of the JSON payload; bits >> 11; × 2^-53.
- Key strings: `'R11-random-v1'` and `'R11-gap-v1'`.
- `seed = 0` for all arms.
- No UID, arrival order or outcome may enter.
- Common keys across arms give common random numbers across levels and methods.

**History-derived state (clone-safe).** Periodic state is recomputed from the executed history at every anchor, never kept in mutable fields that could drift.

`anchors = [t for t in range(q.step) if q.hist_has_vision[t]]` (past vision decisions)

- `calls_before = #{t in anchors : q.hist_hit[t] == 0}`
- `run = #` anchors since the last `t` with `hist_hit[t] == 0`, i.e. past vision HITs after the most recent past MISS, or since step 0 when none.
- `last_call_step = max{t in anchors : hist_hit[t] == 0}`, or `-1` when none.

Lifecycle-forced or veto LOOKs are vision decisions and therefore count as anchors, which matches what the judge sees.

## 1. Random miss (owner): `R11-RAND(ρ)`

At anchor step s (step 0 included), when the guard did not force:

```
u = uniform('R11-random-v1', 0, task, init, s, 'knob-anchor')
call iff u < ρ
```

This is a one-line change of R6 `_ExtraDose` applied on top of R10Recipe; the key string differs from R6.

**Expected IR:** `IR = v·(c_v + c_m·(g + (1 − g)·ρ))`
- v ≈ .503 (LIBERO-10) or .51 (Spatial);
- g is the base guard share of anchors.

## 2. Periodic miss (owner), guard-aware: `R11-PER(K)`

**Meaning.** "Call the policy at regular intervals since the last policy call." K ≥ 0 is the mean number of cache anchors allowed between two policy calls of any source.

**Rule.** At anchor step s, when the guard did not force:

```
base, frac = floor(K), K - floor(K)
u_c  = uniform('R11-gap-v1', 0, task, init, last_call_step, 'cap')      # one draw per gap
cap  = base + (1 if frac > 0 and u_c < frac else 0)
call iff run >= cap
```

**Properties:**
- Guard calls reset `run` and draw a new cap. A knob call therefore never comes sooner than `cap` anchors after any policy call, which is a built-in refractory period.
- No cache-only stretch exceeds ceil(K) anchors.
- At episode start, `run` counts from step 0 with `last_call_step = −1`, so step 0 is a call only when cap = 0 (K < 1).
- Without the guard, the anchor miss fraction is ≈ 1/(K + 1).

**Reference.** `ir_model.DitheredGapCap.calls` runs the same rule on library episodes. It indexes the cap draw by the anchor index of the last call rather than its step; with two-step anchors this is the same partition of gaps, and sol should use `last_call_step` as specified above.

**Why not the existing `--os-judge-cap`?** It counts request slots including blind tails, so it can force early vision. B cap4 showed v = .569. The anchor-level rule keeps the ten-control cadence.

## 3. Variants (one level each; offline-motivated, exploratory)

**`R11-PER-PGT1(K)`: periodic plus a one-anchor post-guard tail.**
- **Rule:** if the most recent past anchor was a **guard**-forced MISS and the guard does not fire now, call (reason 63). Otherwise apply §2 with the run reset at that tail call.
- **Detecting a guard MISS from history:** the plugin history does not label MISS sources, so the judge needs a per-episode list of guard-forced steps. Keep it as a per-connection list `guard_steps` that is appended inside `query` when the guard fires and cleared in `reset(episode)`. The guard verdict is a deterministic function of the query, so this is clone-safe in the same way as the guard memo.
- **Offline motivation:** anchors right after a guard fire carry 1.02–1.27× the mean LOEO policy-vs-cache error (OFFLINE §4).

**`R11-RAND-T2(ρ)`: random trigger opening a two-chunk policy segment.**
- **Rule:** if the previous anchor was a reason-61 trigger (tracked per connection like `guard_steps`) and the guard does not fire now, call (reason 64). Otherwise apply §1 with ρ.
- **What it tests:** whether longer policy segments at the same IR beat more frequent single chunks.

**Rejected offline (not proposed for closed loop; OFFLINE §4):**
- **Front-loaded schedules:** early anchors have 0.68–0.95× the mean cache error.
- **Refractory after guard calls:** suppressing the knob after a guard call skips the riskier post-guard anchors.
- **Per-episode sigma-delta total budget:** about 40% less sensitive to guard-rate error, but it withholds calls in guard-heavy (failing) episodes and spaces them less evenly than the gap cap.

## 4. Calibration: library → knob setting for a target owner IR (exact procedure)

**Inputs:** the selected library only (R10 subset spec or full library). No closed-loop data, no success labels.

1. **Whole-episode-out replay** (`loeo_anchor_table.py`).
   - **Folds:** position % 5 within each task's episode order, the same folds as the GC_dist distance scale.
   - **Cache refit per fold:**
     - PCA is refitted on the training episodes, or the stored bpool basis is used at 500;
     - the per-task AWM main and early metric on training rows uses nn 3, λ .1 and `heads = action[:, :5, :7]/σ_train`;
     - kref is 5 at 5 episodes per task and 8 otherwise.
   - **Serving:** every held-out row is served with top-16 and the kernel, and the top-1 row's `progress` and `ep_len` are recorded.
2. **Anchors and guard flags** (`ir_model.load_episodes`).
   - Anchors are the even steps of each held-out episode.
   - Guard flags come from the frozen noprog_span rule (`ir_model.guard_flags`; noprog_n 3, prog_eps .5).
   - Slots per episode = the episode length.
3. **Base quantities:**
   - v = Σ anchors / Σ slots;
   - g = Σ guard anchors / Σ anchors.
4. **Solve:**
   - **Random:** closed form `ρ = clip((f* − g)/(1 − g), 0, 1)` with `f* = (T/v − c_v)/c_m` (`ir_model.solve_rho`). The simulated bisection `offline_curves.solve(…, Random)` agrees to < .002; use either.
   - **Periodic and variants:** bisection on K over [0, 40] so that `ir_model.simulate` gives IR = T. Use 8 keyed replicates per episode for the dither; IR is monotone decreasing in K.
5. **Freeze:** store (method, setting, target, library v and g, predicted IR) in the artifact metadata.

**Runtime.** About 4 min for all 12 cell-sizes on 16 CPUs for step 1; seconds per solve. A fit-time port can reuse the builder's existing five-fold distance-calibration loop (`recipe/fitting.py: distance_scale`), which already serves every held-out fold. That loop uses the fixed selected-library PCA instead of per-fold PCA; expect g to change by about .01 (to be measured, not assumed).

## 5. Frozen settings for the proposed grid

From `out/arm_grid.json`; full prediction columns are in PREDICTION.md. ρ for random variants, K for periodic variants.

| cell-size (library) | target IR | R11-RAND ρ | R11-PER K | variants |
|---|---|---|---|---|
| π0.5 LIBERO-10, R10 subset 50 | .25 / .32 / .40 | .1950 / .4177 / .6610 | 2.6125 / 1.0657 / .4386 | PGT1 @ .32: K 1.8009; RAND-T2 @ .32: ρ .2839 |
| GR00T LIBERO-10, subset 50 | .25 / .32 / .40 | .1707 / .3998 / .6555 | 2.7237 / 1.0252 / .4512 | PGT1 @ .32: K 2.3165; RAND-T2 @ .32: ρ .2777 |
| π0.5 Spatial, subset 50 | .22 / .30 / .40 | .2112 / .4292 / .6885 | 2.5795 / 1.0549 / .3931 | PGT1 @ .30: K 1.4050 |
| GR00T Spatial, subset 50 | .22 / .30 | .2144 / .4310 | 2.4123 / 1.0109 | PGT1 @ .30: K 1.4117 |
| GR00T LIBERO-10, subset 200 | .25 / .32 | .1910 / .4221 | 2.4585 / 1.0069 | — |
| π0.5 LIBERO-10, subset 200 | .25 | .2663 | 2.0000 | — |
| π0.5 LIBERO-10, subset 500 | .25 | .2537 | 2.0924 | — |
| GR00T LIBERO-10, subset 500 | .25 | .1869 | 2.4699 | — |

The subsets are the R10 nested subsets (`rounds/r10/subsets/`; seed 20261002 + task permutation, first n/10 per task) of `bpool_cs` (π0.5) and `bpool_all` (GR00T).

## 6. Acceptance audits for sol (decision logs, any dev pilot)

- **Ledger:** `(N, V, M)` equal the decision counts. Knob calls = #reason 61/62/63/64, and no knob call sits on a guard-forced anchor.
- **Cadence:** every knob MISS is followed by exactly one policy tail and then a fresh vision anchor, the same as guard MISSes.
- **Periodic:** at every anchor, `run ≤ cap`, and a knob call happens iff `run ≥ cap` with the guard silent. No knob call within `cap` anchors after any call.
- **Random:** the realized eligible-anchor call share is within binomial error of ρ.
- **Determinism:** the coins match the keyed formula. Re-serving the same history gives the same decision.
- **IR:** realized owner IR vs `pred_IR_lib`.
  - Expected miss band: about ±.02 given the guard-rate uncertainty (IR_MODEL §5).
  - The random and periodic arms of a cell should land within ≈ .01 of each other.
