# TRACER Phase 5 — Failure-Aware Retrieval Cache: Findings Synthesis

**Scope**: TRACER M2 failure-aware dual-retrieval cache (D⁺/D⁻ pools + sigmoid gate
`g = σ(b0 + b1·margin + b2·u_t + b3·delta_pos)`), calibrated offline and evaluated online on
two LIBERO suites (`libero_spatial`, `libero_10`) via paired I_val rollouts
(baseline cache-OFF / illustrative gate / calibrated gate; same 250 held-out episodes + seed 7).

**Bottom line**: **FAIL on both suites, under both fusion designs.** The exit gate
(BHR↓ AND SR_calibrated ≥ SR_base − ε AND IR↓) is not met anywhere. The investigation
produced two distinct findings — one a *fixed implementation bug*, one a *fundamental
limitation of offline calibration for this task*.

---

## 1. Before the fix — `base_fusion = weighted_rrf` produces **0 FULL_HIT** (both suites)

The original dual-retrieval config fused per-field similarities with **weighted RRF**
(reciprocal-rank fusion). RRF is *rank-based*: the top-1 neighbour's fused score is always
`≈ Σ w_f / (60 + 0) ≈ 0.056` regardless of how similar it actually is, and the magnitude
of the raw similarity is discarded.

Consequence: `s_pos` (the positive-pool match score feeding the gate margin) has **no
usable dynamic range** — every query's top match scores the same ~0.056. The gate cannot
discriminate a genuine match from a coincidental one, so with any sane threshold **no step
ever fires FULL_HIT**. A cache that never hits is meaningless — the runtime is just plain
`pi05` inference with overhead.

- Verified this is **not** a pooling problem: raw vision cosines are ~0.99 (anisotropic
  embedding space) for **both** `cp1_mean_pool` and `cp1_spatial_pool_16`. Swapping the
  pooler does not restore discrimination.
- Verified this is **not** a gate-tuning problem: with RRF, `s_pos` range across the whole
  I_cal table is ~0 — no `(b0,b1,b3)` can separate hits from misses.
- **Root cause = the fusion operator collapses the signal.**

## 2. The fix — `weighted_score_sum` + per-field z-score/tanh normalization

Replaced RRF with the proven `exp/weighted_sum` recipe:
**`weighted_score_sum`** fusion + **per-field z-score → tanh** normalization
(`ZScoreNormalizer`, `squash="tanh"` → `0.5·(tanh(z)+1) ∈ [0,1]`) + robot-state-heavy field
weights (`vision_0=0.12 / vision_1=0.31 / robot_state=0.56`; `prompt_emb` enabled at
weight 0 to stay out of the active-field set).

This **restores discrimination**: under score-sum `s_pos` spans ~0.024→0.635 (a 26×
range); after z-score/tanh the score fills `[0,1]` and the FULL_HIT-rate curve becomes
tunable (τ≈0.3 → ~76 %, τ≈0.7 → ~40 % on the offline table). **FULL_HITs are enabled** —
confirmed online on both suites (see §3). *The §1 bug is fixed.*

## 3. After the fix — FULL_HITs fire, but **substitution crashes SR** (both suites)

Paired I_val rollouts (BHR = BadHitRate among FULL_HITs; SR = task success rate):

### libero_spatial (SR_base = 0.972, ε = 0.02)
| run | gate | FULL_HIT rate | SR | BHR |
|---|---|---|---|---|
| baseline | cache-OFF | — | **0.972** | — |
| illustrative | τ≈0.9 conservative | (4443 hits) | **0.776** (−19.6 pp) | 0.431 |
| calibrated | solved (λ=0.5, b2=−0.5) | (4112 hits) | **0.776** (−19.6 pp) | 0.367 |

→ Verdict **FAIL**: BHR↓ True, but SR_cal 0.776 ≪ 0.972−ε. Even the *conservative*
illustrative gate already crashes SR by ~20 pp.

### libero_10 (SR_base = 0.856, ε = 0.02)
| run | gate | FULL_HIT | SR | BHR |
|---|---|---|---|---|
| baseline | cache-OFF | — | **0.856** | — |
| illustrative | τ≈0.9 conservative | 20.6 % (6356/30914) | 0.824 (−3.2 pp) | 0.433 |
| calibrated | solved (λ=1.0, b2=0) | 57.7 % FULL_HIT + 42.3 % WARM, **0 MISS** | **0.536** (−32 pp) | 0.489 |

→ Verdict **FAIL**: BHR **rises** 0.433→0.489, SR_cal 0.536 ≪ 0.856−ε. The calibrated gate
substitutes cache on **every single step** (0 fresh inference) and SR collapses.

## 4. The deep finding — offline calibration cannot see the online SR cost

Across both suites the pattern is identical: **~37–49 % of FULL_HITs are *bad*** (BHR),
i.e. the retrieved cached action is not what a fresh `pi05` forward pass would have produced
at that state. Substituting even a minority of bad actions derails precise manipulation, and
SR — a *trajectory-level* outcome — collapses far more than the per-step BHR would suggest.

The calibration objective `L_cal` minimizes a **per-step proxy** (BHR / IR on the held-out
I_cal table). This proxy **does not correlate with online SR**:

- A retrieved action can pass the margin gate (low predicted badness) yet still nudge a
  contact-rich trajectory off its manifold; the error compounds over the remaining horizon.
- `L_cal` has no term for, and no visibility into, this downstream trajectory divergence —
  the offline table is scored against the recorded action, not against the counterfactual
  rollout that substitution induces.
- Hence offline calibration reliably **over-substitutes**: it drives FULL_HIT/WARM rates up
  to bank inference savings (l10 calibrated: 100 % substitution, ~72 % inference saved) while
  being blind to the SR it is destroying.

**Conclusion.** The score-sum + z-score/tanh fix is *correct and necessary* — it repairs the
RRF signal-collapse bug and makes the cache able to hit. But it exposes the real limitation:
**cache-substitution of policy actions is unsafe for precise manipulation under an
offline-calibrated per-step gate.** To ship this cache one of the following must change:
(a) make the calibration objective SR-aware (requires an online/counterfactual signal
`L_cal` currently cannot observe), (b) restrict substitution to trajectory segments where
divergence is provably bounded, or (c) use the cache for warm-start only (never full action
substitution) and re-measure.

---

## Provenance
- Configs: `exp/zixuan_proposal/config/dual_retrieval_scoresum{,_l10}.yaml` (+ `_calibrated{,_l10}`).
- Normalizers: `exp/zixuan_proposal/data/normalizers_{spatial,l10}.json` (LOEO-fit via
  `exp/common/calibrate_score_normalizers.py`).
- Per-suite verdict cards: `phase5_scoresum_report_{spatial,l10}.md`;
  RRF-era (before-fix) card: `phase5_calibration_report_spatial.md`.
- Pass-3 gate JSONL + episode-results: `exp/zixuan_proposal/data/pass3_scoresum_{spatial,l10}_*`.
- Devices: server = jupyter-ziyang10 (H200, shared/contended → `--replicas 1`);
  client = timan107 (12 workers). Both configs run sequentially (never concurrent).
