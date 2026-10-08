# R12 opus — SPEC (for sol): look-half on the five-layer system

opus, frozen 2026-10-03 ~12:25 CDT, before any R12 closed-loop result. Evidence: `OFFLINE.md`. Everything fitted
here is fitted on the selected B-pool library only (the R10 `fit_path` / `library_boundary` guard applies unchanged).

> **Status after DIRECTIVE_2 (owner ruling 2026-10-03 12:5x CDT):** §1 (S, batched camera encode) is **withdrawn** —
> an implementation speed-up, not a method, and not a saving. Prices are compute shares (one camera = half of the
> full vision price). §2–§5 (cascade C, H0) are **superseded by §6–§7** below, which are validated on the closed-loop
> B-dev recordings. Earlier sections are kept unchanged for the record.

Three proposals, in the order they should be decided:

| id | what | layer | status |
|---|---|---|---|
| **S** | embed the two real pi0.5 cameras in ONE vision-tower call; masked third slot from the cached dummy | serving (stage 1) | recommended first; decides whether C is worth building |
| **C** | wrist-first consistency cascade: wrist-only look, complete the agentview only when a self-consistency test fails | layer 1 (+ wrist head in layer 3/5) | build only if S is infeasible or Phase 0 shows a real one-camera saving |
| **H0** | wrist-only at every non-call look (C with an infinite tolerance) | layer 1 | calibration arm only (tests the offline→SR mapping and goal b) |

---

## 1. S — batched camera stage 1 (pi0.5)

**Change.** A stage-1 override in the style of `closed_loop/stage_overrides.py` (no `src` edit): for every stage-1
call (looks and policy calls), stack `images[0]` (base) and `images[1]` (wrist) along the batch axis and call
`paligemma_with_expert.embed_image` once (B = 2 per request); take slot 2 from `Pi05Override._dummy` (already cached
and already validated as all-masked); assemble `prefix_embs`, pad masks, attention masks and position ids exactly as
stock (order base, wrist, dummy, language). Keys are sliced/pooled from the prefix exactly as today.

**Why.** Tower-only microbenchmark (OFFLINE §2): 1 image 23.0 ms, 2 images in one call 23.5 ms, 3 sequential 69.0 ms;
R7 stock stage 1 = 62.1 ms, wrist-only 26.4 ms. Expected batched stage 1 ≈ 27–32 ms → a full look ≈ .066–.078 of a
call instead of .152, with both cameras.

**Parity.** Batched keys differ from per-camera keys by 0.15 % relative L2 (`out/batch_parity.json`), the size of the
existing recomputation noise. Required checks (dev): step-0 top-1 agreement with the per-camera path ≥ 0.90 on the
same states (cf. GR00T cross-GPU 0.936); policy-chunk parity at bf16 tolerance on recorded observations.

**Ledger.** The coordinator measures the batched stage-1 price (REQUESTS 0) and the ledger prices every look and the
stage-1 part of every call at that price. Report IR both on the old basis (.152) and on the new one.

**GR00T.** Nothing to do: `GrootStagedRunner.run_stage1` already embeds both images in one `extract_feature` call.

## 2. C — wrist-first consistency cascade (`LookHalfCascade`)

### 2.1 Fitted state (all from the selected library, at build time)
Everything of the deployed `R10Recipe` fit stays bit-identical (PCA bases `B0T/B1T/muB0/muB1`, per-task F metric,
guard tables, F corrector heads, F distance scale, kref). Added per task t:

1. **Wrist cache W**: AWM per-task metric (`awm.fit_metric`, nn 3, λ .1, main + early) on features
   `x_W(r) = [P1(r), rs8(r)]`, `P1 = B1T·key_v1 − muB1` (the deployed basis; no new PCA). Same kref, k = 16, same
   kernel, same AWM regimes (regime 1 continuity after a MISS uses W's distances; the early metric is never used
   because step 0 is always full). Implementation can reuse R4's `WristAWM` feature hook with the R10 subset view.
2. **Lagged consistency metrics WL_L**, L ∈ {2, 4, 6, 8} decisions: `fit_metric` (nn 3, λ .1, main only) on
   `x_L(r) = [P0(r_L), P1(r), rs8(r)]`, where `r_L` is the row L steps earlier in r's own episode, or r itself when
   step(r) < L (exactly as evaluated offline). Store per-task codes `Z_L` for all candidate rows (4 × 136 floats/row).
3. **Wrist corrector head** (layer 3, task-indexed like the existing heads): R10 `loeo_heads` recipe with the W
   cache — LOEO synthesis of each library row with W (own episode excluded, regime 2), target = stored policy chunk
   minus that synthesis on [:10, :6] in σ units, features `[P1 (64), rs8, synth[:10,:7]/σ (70), min(step,120)/120,
   one-hot task]` (153-d), 384 RFF seed 0, ridge 100, equal episode weights, blend .5.
4. **W distance scale**: R10 `distance_scale` recipe (astra's five folds, fixed library PCA, training-fold σ and
   metric, same-task nearest distances, episode-balanced median of non-step-0 held-out d1) with the W feature set.
   Attenuation exactly as GC_dist: `factor = clip((2 − d_min/scale_W)/1.25, 0, 1)`.
5. **Tolerance** (cell-level constant, task-agnostic): `τ(N) = 0.15·√(50/N)`, N = number of library episodes
   (50 → .150, 100 → .106, 200 → .075, 300 → .061, 400 → .053, 500 → .047). Derived offline: the largest tolerance
   whose library-replay retrieval cost stays ≤ .002 e10 (≈ 0.2 pp) pooled over the four model×suite cells.
6. σ for the consistency score = the deployed cache's action σ (`base.sig`, per channel, executed steps).

### 2.2 Online rule (per decision)
State per episode: `last_agent = (step, P0)` of the most recent look whose agentview was encoded (step 0, a planned
full look, a completion).

- **Blind slots**: unchanged (anchor tail of whatever result was served at the last look, W or F).
- **Step 0**: full look (stock or S path). **Planned policy call** (the layer-4 periodic knob decides from history that
  this look is a call): full look, deployed path, MISS. All other looks, including the lifecycle look right after a
  MISS, are **wrist-first**:
  1. wrist-only stage 1 (existing per-request `wrist_only` path); `P1 = B1T·key_v1 − muB1`.
  2. W retrieval (top-16, kernel, AWM regime rules) → uncorrected kernel mean `sW` [H, 32].
  3. `L = step − last_agent.step`. If `L > 8` → **complete** (reason `lag`).
  4. WL_L retrieval with query `[last_agent.P0, P1, rs8]` against `Z_L` → uncorrected kernel mean `sWL`.
  5. `s = sqrt(mean_{t<10, j<6} ((sW − sWL)[t,j]/σ_j)²) + mean_{t<5} 1[sign sW[t,6] ≠ sign sWL[t,6]]`.
  6. `s > τ(N)` → **complete** (reason `consistency`): embed the base camera (completion path), recompute `key_v0`
     from the completed prefix with the deployed key builder, then run the **deployed** two-camera query (F
     retrieval, F corrector, deployed guard) unchanged. Set `last_agent` to this look.
  7. `s ≤ τ(N)` → **one-camera look**: served chunk = `sW` + .5·factor_W·(wrist head correction) on [:10, :6]
     (step 0 never; gripper and padding unchanged); the anchor memo (blind tail) uses W's rows/weights/corrected
     chunk; the deployed only-no-progress guard runs on **W's top-1** (own guard, unchanged rule). If it flags → MISS:
     the plugin completes the base camera for the policy call (existing MISS completion), and `last_agent` is set
     to this look with the completed `P0`.
- No other change to layers 2–5. Layer-4 knob calls reset the guard gap as in R11.

### 2.3 Plugin / harness contract (pi0.5)
1. `--os-request-cameras` stays the switch. The method declares `next_camera_mode ∈ {full, wrist_only}` before
   stage 1 (as R7). Step 0 stays forced full; **look reason 6 (after a MISS) must no longer force full** for this
   method (declared by a class attribute, e.g. `allows_wrist_after_miss = True`); 7/8 stay forced full.
2. **Completion for retrieval**: `query()` on a wrist-only look may return a sentinel `NeedCamera("base", reason)`
   without committing any state. The plugin then runs the completion (`Pi05Override.complete` at B = 1), rebuilds
   `key_v0` with the key builder from `prefix_embs[:, :256]`, and calls `query()` again on the full view. The second
   call is the only one that commits anchor / guard / history state. At most one completion per look (a later MISS on
   that look reuses it).
3. **Ledger** (`os_sw_*` fields extended): wrist-first look .0646; completion .0502, counted once per look, with
   `completion_reason ∈ {consistency, lag, call}`; planned full look .152 **through the stock stage-1 path**, not the
   per-request full path (.1656 as implemented, R7 Addendum A); policy call .848. Under S all prices are replaced by
   the measured batched ones.
4. **Logs per look**: `camera_plan` (full_planned / wrist_first), `completed`, `completion_reason`, `lh_score`,
   `lh_lag`, `lh_tau`, W and WL top-1 rows, `d_W_min`, `factor_W`. Keep `--os-log-inputs` available.

### 2.4 Tests sol should ship
- **Identity**: τ = −∞ (always complete) reproduces the deployed `R10Recipe` decisions (rows, weights, served bytes,
  guard flags) exactly on the R10 replay harness; the only ledger difference is wrist + completion pricing.
- **H0 path**: τ = +∞ and no lag limit gives wrist-only at every non-call look; completions only for calls.
- **Library replay parity**: on the 50-episode pi0.5 L10 subset, the k-fold one-camera share and completion share at
  τ(50) match `out/sim3_all.json` (`cw0.15`, K0) within ±.03 (my simulation is open-loop and uses k-fold metrics;
  sol's uses deployed full-library metrics, so small differences are expected).
- Fit-boundary audit as R10 (no read outside the selected library).

### 2.5 GR00T
Same algorithm and fitting would apply (v1 = `video.wrist_image`), but GR00T already embeds both images in one call;
build it only if Phase 0 shows a one-image stage 1 at ≤ 0.6 of the two-image stage 1.

## 3. H0 — wrist-only at every non-call look (calibration arm)
C with τ = +∞ and no lag limit (no WL metrics needed): W retrieval, wrist head, own guard, completion only for calls.
Purpose: a closed-loop point at a known, larger offline cost (dq ≈ +.005–.008 on the 50-episode pi0.5 cells) to
calibrate the offline→SR mapping and test goal (b) directly.

## 4. Experiment plan

**Phase 0 — prices (coordinator, local 4090, before any arm is built).** REQUESTS 0: stock vs batched two-camera
stage 1 vs wrist-only vs completion on the pi0.5 serving path; GR00T stage 1 with one vs two images.
Frozen decision rule:
- if batched two-camera stage 1 ≤ 1.25 × wrist-only stage 1 → adopt **S**; C's remaining saving would be ≤ .005 IR
  per slot, so do **not** build C for pi0.5;
- if GR00T one-image stage 1 ≥ 0.6 × two-image → no GR00T look-half;
- otherwise build C (and H0) as specified.

**Phase 1 — dev pilot (B-pool dev inits via `closed_loop/devset.py`, 50 episodes per arm, `--os-log-inputs`).**
- S: pi0.5 L10-50 and Sp-50, S vs baseline in the same batch. Gates: step-0 top-1 agreement ≥ .90 on matched inits;
  realised IR drop within ±.01 of the Phase-0 price prediction; no SR collapse (> 10 pp drop stops the arm).
- C (if built): pi0.5 L10-50, Sp-50, L10-500, knob off, C vs baseline; plus the identity run (τ = −∞, 10 episodes).
  Gates: realised one-camera share inside the 80 % interval of PREDICTION.md; completion reasons as logged;
  guard rate within ±.04 of baseline; no SR collapse. H0 on L10-50 (50 episodes) as a sanity check.
- Nothing from the pilot changes τ, metrics or heads (they are library-only); the pilot only checks implementation
  and the cost model.

**Phase 2 — frozen test-A grid (500 episodes per arm, one run each, same-batch baselines).**
- If S adopted: pi0.5 L10-50, Sp-50, L10-200, L10-500, Sp-500 knob off, and L10-50 / Sp-50 with the periodic knob at
  .32: S vs baseline (14 arms). Claim: pooled paired ΔSR lower 95 % bound > −1.0 pp; IR from the ledger.
- If C built: knob off on the same five pi0.5 cell-sizes: baseline vs C (10 arms); H0 on L10-50 and Sp-50 (2 arms);
  knob on (L10-50, Sp-50): baseline P@.32, C P@.32, and **C retargeted** so that its realised IR equals the baseline
  P@.32 arm (the vision saving re-spent on calls; retarget with the R11 library IR model using C's mean look cost from
  `out/sim3_all.json` `look_owner`) (6 arms). Matched-IR comparison: SR(C retargeted) − SR(baseline P@.32); matched-SR
  total IR: interpolate C's SR-vs-IR curve (P@.32 and retargeted) to the baseline's SR.
- GR00T only after Phase 0 says its one-camera look is cheaper; then knob-off GR00T L10-50, Sp-50, L10-500 × {baseline,
  C}.
- Statistics: pairing by (task, init); exact McNemar per cell; pooled stratified bootstrap over cells (equal weight,
  10 000 draws); report the realised one-camera share, completion share by reason, guard rate, and IR on both price
  bases. π0.5 runs differ bit-for-bit between replicates (R11), so single-cell differences under ~3 pp are noise.

---

## 5. Addendum (2026-10-03 12:28 CDT, dev-calibrated, before any test-A result)
- **Tolerance for C/H0 arms: use τ'(N) = 0.10·√(50/N)** instead of §2.1 item 5. Reason: on the coordinator's
  B-pool dev recordings of the deployed system (pi05 L10-50, shadow policy chunks, 3 098 looks) the frozen τ = .15
  lets through looks whose closed-loop retrieval cost is +.0035 (target ≤ .002); τ = .10 gives +.0014 at a 40 %
  one-camera share (`out/dev_validate_taus.json`). PREDICTION Addendum A has the revised numbers.
- **GR00T**: the coordinator's measured prices (one-camera .143–.150 vs full .148) settle §2.5: no GR00T look-half.
- **Phase 0 remains open for pi0.5**: the batched two-camera stage 1 (S) is still unmeasured; the dummy-cached
  estimate is ≈ .118. The decision rule of §4 is unchanged.

---

## 6. M1 — guarded one-camera look (cache-layer method; DIRECTIVE_2; 2026-10-03 13:10 CDT)

Library-fitted state as §2.1 items 1–4 and 6 (W metric, lagged WL_L metrics, wrist-only corrector head, W distance
scale), plus nothing else. **Dev-calibrated constants** (B-pool dev recordings, labelled as such): τ = 0.10 and
m = 1.0 (π0.5 and GR00T alike; task-agnostic, cell-agnostic).

**Per look** (blind slots unchanged):
1. **Planned two-camera look** (both cameras encoded, deployed path) if: step 0; or the layer-4 knob schedules a call
   at this look; or **the previous look called the policy**.
2. Otherwise encode the **wrist camera only**; P1 = B1T·key_v1 − muB1; W retrieval (AWM regime rules) and, with the
   most recent agentview code (lag L ≤ 8 decisions; L > 8 → complete), WL_L retrieval.
3. **Stall suspicion → complete**: let p_prev = library progress of the previous look's served top-1. Complete if
   `min(adv(W top-1), adv(WL_L top-1)) ≤ .5 + m` library steps, adv(r) = (prog(r) − p_prev)·(ep_len(r) − 1).
4. **Consistency → complete**: complete if `|W − WL_L|` (motion RMS in σ over 10 steps × 6) + mean gripper-sign
   disagreement over the first 5 steps > τ.
5. **Complete** = encode the agentview, refresh the agentview memory, run the deployed two-camera decision unchanged
   (F retrieval, deployed corrector, deployed only-no-progress guard → HIT or call).
6. **Serve one camera** otherwise: the WL_L kernel-mean chunk + .5·factor·(wrist-only head correction) on [:10, :6]
   (factor from W's distance scale, GC_dist form); the anchor memo uses WL_L's rows/weights; the guard memo records
   WL_L's top-1 (by step 3 it cannot flag at this look).
- GR00T: the wrist is encoded alone (not bit-identical to the joint slice: 2 % key difference, top-1 unchanged 96–97 %,
  OFFLINE §12.6); used as is against the joint-encoded library.
- Ledger (compute share): planned/complete look .152 (GR00T .148) = .076 + .076 (.074 + .074); one-camera look .076
  (.074); calls + .848 (.852). Log per look: plan, completion reason (planned / suspicion / consistency / lag),
  score, lag, both top-1 advances.
- Plugin needs (π0.5 and GR00T): a wrist-only stage 1 path, completion-for-retrieval (§2.3 item 2), and the planned
  full look declared before stage 1 (knob schedule, previous-call flag). GR00T needs a one-image `extract_feature`
  stage 1 and a completion of the other image (as in `prep/bench_latency.py`, benchmark-only today).
- Tests: τ = −∞ or m = ∞ reproduces the deployed decisions exactly; the dev replay (`tools/dev_eval3.py`) on the
  recorded arms reproduces the one-camera shares of OFFLINE §12.4 within ±.03.

## 7. Experiment plan for M1 (EXPERIMENTS)

The decisive unknown is the guard in the method's **own** closed loop: dev replays run on two-camera trajectories
and show the one-camera system would miss 1–9 % of the deployed calls after protection; R9 saw delayed guard calls in
its own loop. So the plan measures that first.

**Phase A — dev pilot (B-pool dev inits, `closed_loop/devset.py`, look recording on, knob off).**
- Arms: baseline vs M1 on pi05 L10-50, pi05 Sp-50, groot L10-50, pi05 Sp-200; 100 episodes per arm.
- Measure: realised one-camera share and completion reasons (vs OFFLINE §12.4 ±.10), call rate, **step of the first
  guard call per episode and the call rate in episodes that end in failure** (vs baseline; R9's failure mode),
  IR from the ledger. Stop rule: if M1's first call comes on average ≥ 2 decisions later than baseline, or the call
  rate drops by > 20 % relative, stop (the guard is being starved).
- Nothing is refit on the pilot.

**Phase B — test A (only if Phase A passes; 500 episodes per arm, same-batch baselines, knob off).**
- Cells where the dev replay predicts a gain: pi05 L10-50, pi05 Sp-50, pi05 Sp-200, groot Sp-50 (+ groot L10-50 as
  the predicted-neutral control): baseline vs M1, 10 arms.
- No knob-on arms: with the knob at .32 the method has ≤ 1 % one-camera looks (OFFLINE §12.4), so its matched-SR
  effect there is 0 by construction.
- Matched-SR read-out: ΔSR (paired, pooled bootstrap) converted with the R11 periodic-knob slope of the same cell;
  claim a saving only if ΔIR + (−ΔSR)/slope < 0 with the pooled 80 % interval below 0.

## 8. Guard H2 and look-half M2 (owner task "change the guard's signal source"; 16:14 CDT)
Full design, thresholds and evidence: `GUARD.md` §4–§7; pilot: `GUARD.md` §8; predictions: PREDICTION.md
GUARD_PREDICTION. Summary for sol:
- **Risk model** (library-fitted, per model): logistic on [wrist-only nearest distance / library median, step/100,
  lag of the wrist top-1 behind its demo / episode length, mean xyz speed over the last two look intervals, xyz
  direction reversals over the last three, log(1 + action-vs-state mismatch / library median residual) with the ridge
  map of `tools/guard_table.py:mismatch_map`, clip(wrist-top-1 progress advance, −5, 8)]; coefficients
  `out/guard_lib.json` (`<model>|coef`: feature means, sds, weights with intercept first).
- **H2**: at looks with step ≥ 4, call ⇔ (wrist-top-1 advance ≤ .5 and risk ≥ θ_a) or risk ≥ θ_b; per-cell θ in
  GUARD §6 (dev-calibrated). Replaces the no-progress flag in the judge; everything else in layers 1–5 unchanged.
  The wrist retrieval W (SPEC §2.1 item 1) is computed at every look, full or one-camera.
- **M2**: H2 as the guard; every look that H2 does not call and the knob does not plan is wrist-first; complete the
  agentview only if its memory is > 8 decisions old or |W − WL_lag| + grip > .10; serve the WL_lag chunk with the
  wrist-only head; a call always completes the agentview (the policy needs both cameras). No "previous call → full"
  rule and no suspicion completions.
- Ledger and logs as §6; also log risk, θ_a/θ_b branch, and the wrist advance at every look.
