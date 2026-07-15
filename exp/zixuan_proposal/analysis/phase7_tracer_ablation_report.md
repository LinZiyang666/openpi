# TRACER Phase 7 — Failure-Aware Retrieval Cache: Integration / Ablation Report

**Date:** 2026-07-14 · **Suites:** libero_spatial, libero_10 · **Model:** pi05 (PyTorch) ·
**Scope:** integration of the committed Phase 5 (M2 calibration) + Phase 6 ①b (M1 projection)
evidence into formal, scope-limited verdicts for the three TRACER mechanisms.

> **Source of truth.** This report is a synthesis of already-committed `.md` evidence
> reports (commit `2d0e4cc`). It does **not** re-run any rollout, re-train any head, or
> re-collect any data. Every headline number is **traceable** to a committed report (cited
> per row) but is **not independently reproducible from the repo** — the raw Pass-3 JSONL,
> projection weights, and gate parameters are gitignored and live on `jupyter-ziyang10`
> (see §8).

---

## 1. Bottom line

**Within the evaluated configurations and protocols** (two suites — libero_spatial /
libero_10; pi05; LIBERO; a specific `weighted_score_sum` + margin-gate parameterization),
**no operating point of the failure-aware retrieval cache's full-hit action substitution was
observed to be simultaneously safe (does not crash SR) and compute-saving.** This is an
observation over the evaluated points, not a proof of non-existence.

The evidence chain is **two findings at two different measurement levels — not a parallel
pair, and Phase 6 did NOT perform any online validation**:

- **Phase 5 (M2 dual-retrieval) — online exit-gate FAIL.** Direct evidence: substitution
  rollouts drop the real online success rate (SR) by 19.6 pp (spatial) / 32.0 pp (l10).
- **Phase 6 ①b (M1 projection) — reduced *offline* rescue-gate NO_GO.** The projection
  gives no statistically significant lift over raw pooled cosine for the failure-aware
  gate's offline safe-reuse AUROC (ΔAUROC CI includes 0 in both suites).

This is a **negative observation bounded by what was evaluated**, not a refutation of the
proposal's literal Claim 1/2/3 (those require comparators that were not run — see §3–§5, §7).

---

## 2. Evaluation framework

**Metric space** = (SR, inf_ratio) + BHR / FFR / IR. Operational definitions are lifted
verbatim from `exp/zixuan_proposal/analyze_phase5_rollout.py` (Eq 26–28):

| Metric | Definition (analyzer) | Line | Reading |
|--------|-----------------------|------|---------|
| **IR** (inference ratio) | `IR = (n_miss + c_warm·n_ws) / n`, `c_warm = 0.75` | `:82` | residual inference fraction; **lower = more inference saved** |
| **BHR** (bad-hit rate) | `BHR = fh_bad / fh_labeled` | `:80` | fraction of FULL_HITs that fall in failed episodes |
| **FFR** | `FFR = miss_safe / safe` | `:81` | among successful-episode steps, fraction that MISS |

**Label proxy and its causal ceiling (load-bearing caveat).** The analyzer defines
`safe_reuse(t) := episode_success == True` and `bad(t) := episode_success == False`
(`analyze_phase5_rollout.py:6`, `:78`). Therefore "**37–49 % of FULL_HITs are bad**" is
**the fraction of full-hits occurring inside failed episodes — an episode-level
association**. It does **not** prove, per step, that the cached action *caused* the failure,
nor that cache substitution is *intrinsically* unsafe for all precise manipulation. The
**direct** safety evidence is the measured online SR drop of the substitution rollouts
(§4.1); the per-step `L_cal ⊥ SR` argument is offered only as a **mechanistic explanation
consistent with** that SR collapse.

**Protocol.** Paired I_val rollouts on the same 250 held-out episodes + seed 7, run
sequentially (never concurrent). Exit gate (Phase 5):
`BHR↓ ∧ SR_calibrated ≥ SR_base − ε (ε = 0.02) ∧ IR↓/=`. Offline rescue gate (Phase 6 ①b):
GO iff the projected-vs-raw ΔAUROC CI excludes 0 with a positive lower bound.

**re-scope ↔ proposal-literal.** The owner's Phase 7 re-scope (2026-07-14) is *"integrate
the already-settled negative results into a report"*, which is **not** the same as
completing the proposal §11 ablations. The alignment (stated per Claim below) is: Claim 1 =
NOT EVALUATED / de-scoped, Claim 2 = reduced-offline-gate NO_GO, Claim 3 = NOT EVALUATED /
owner-de-scoped.

---

## 3. Claim 1 (M2 dual retrieval) — NOT EVALUATED / de-scoped; evaluated dual full-hit operating points FAIL the exit gate

**Proposal-literal Claim 1** = success-only retrieval vs dual (success+failure) retrieval,
decided by bad-hit reduction under matched success / matched inference ratio (§11, §14).
**This was not evaluated:** the Phase 5 runs were a cache-OFF baseline (SR_base only) +
illustrative dual + calibrated dual, and **both** cache-ON YAMLs are `enable_dual: true` —
there is **no success-only comparator**. The data below therefore shows only that **these two
dual full-hit operating points fail the exit gate**, not that dual is worse (or better) than
success-only.

Source: `phase5_scoresum_findings.md` §3 + `phase5_scoresum_report_{spatial,l10}.md`.

| suite | SR_base | SR_calibrated | ΔSR | BHR (calibrated) | FFR | IR | exit gate |
|-------|--------:|--------------:|----:|-----------------:|----:|---:|-----------|
| libero_spatial | 0.9720 | 0.7760 | −19.6 pp | 0.3670 (1509/4112) | 0.0000 | 0.2831 | FAIL |
| libero_10 | 0.8560 | 0.5360 | −32.0 pp | 0.4893 (5530/11303) | 0.0000 | 0.3171 | FAIL |

Both suites fail `SR_cal ≥ SR_base − ε`. On l10 the calibrated gate substitutes on **every
step** (0 fresh inference; IR 0.317 residual) and SR collapses to 0.536. The **direct
evidence** is the online SR drop; the episode-level "37–49 % bad FULL_HIT" figure and the
`L_cal ⊥ online-SR` proxy argument are the mechanistic explanation (§2 caveat), not a
per-step counterfactual proof.

Before this result was trustworthy, a prior `weighted_rrf` signal-collapse bug (top-1 fused
score pinned at ~0.056, `s_pos` with no dynamic range → 0 FULL_HIT) was fixed by switching to
`weighted_score_sum` + per-field z-score/tanh normalization; only after the fix do FULL_HITs
fire, exposing the substitution-safety limitation above (`phase5_scoresum_findings.md` §1–§2).

---

## 4. Claim 2 (M1 projection) — reduced offline rescue-gate NO_GO

**Proposal-literal Claim 2** = raw vs projection including the pre-registered B-vs-C
(action-only vs action+denoise) comparison and downstream SR/IR. **Only the reduced piece
was evaluated:** raw A′ vs action-only projected B on the I_cal offline safe-reuse AUROC.
B-vs-C and downstream SR/IR were **not** run.

Source: `phase6_ib_offline_gate_report.md` §3 (confound-free July-only data; April/July
rendering-drift confound eliminated by drawing D⁺ and D⁻ from the same July run).

| suite | I_cal ep | succ. | AUROC raw A′ | AUROC proj B | ΔAUROC (B−A′) | 95% CI | offline gate |
|-------|---------:|------:|-------------:|-------------:|--------------:|--------|--------------|
| libero_spatial | 250 | 96.0% | 0.8156 | 0.8194 | +0.0038 | [−0.011, +0.018] | NO_GO |
| libero_10 | 250 | 82.8% | 0.7556 | 0.7600 | +0.0044 | [−0.0031, +0.0115] | NO_GO |

Both CIs include 0 → **reduced offline gate NO_GO**: the outcome-compatible projection gives
no significant lift over raw pooled cosine for the gate's offline safe-reuse prediction.
libero_10 is the cleaner test (better class balance: 17.2 % failures vs spatial's 4.0 %) and
reaches the same verdict, so the negative result is corroborated across suites. **This is an
offline safe-reuse prediction gate — not the proposal-literal Claim 2's full (denoise +
online SR) validation.**

---

## 5. Claim 3 (M3 dynamic chain depth) — NOT EVALUATED / owner-de-scoped

**No fixed / oracle / dynamic depth data was collected.** Logical-closure argument: M3 only
optimizes retrieval quality / efficiency; its Pareto value is conditional on M2 reuse being
shippable. Since the evaluated M2 operating points do not pass the exit gate, the
fixed/dynamic depth ablation triple closes at the M2 leg. The owner ruled (2026-07-14) not to
collect a new depth-ablation rollout. This is recorded as **owner-de-scoped + logical
closure — not written as an empirical ablation pass or fail.**

---

## 6. Warm-start (background only — not load-bearing)

The owner attests, from prior experiments, that pure warm-start is cost-comparable to full
inference (≈ no compute savings). This report **does not let that finding bear any evidential
weight**, for two reasons: (a) there is **no committed latency/cost `.md` artifact** in this
repo (`exp/warm_start/analysis/` holds only `success_rate_sweep.png`); (b) the repository's
warm-start runbook actually defines `start_t` as saving **part** of Stage 3, which is in
tension with a flat "no savings" statement. Warm-start is therefore listed here only as
**un-independently-verified background**; it does not enter the §7 verdict matrix or the §1
bottom line. It could be promoted to load-bearing only if the owner supplies a traceable cost
artifact + computation definition + provenance.

---

## 7. Integration / ablation matrix

| Mechanism | Evaluation status | Evaluated operating point: safe? | compute-saving? | Verdict (this report) | proposal-literal gap |
|-----------|-------------------|----------------------------------|-----------------|-----------------------|----------------------|
| **M2** dual retrieval (Claim 1) | dual full-hit points evaluated; success-only comparator **not** run | **No** — SR −19.6 / −32.0 pp | Yes (IR↓ to 0.317 on l10; ≈68.3 % of the IR cost-proxy avoided = 1−IR) | **NOT EVALUATED / de-scoped**; evaluated dual points **FAIL** exit gate | missing success-only matched comparator |
| **M1** projection (Claim 2) | reduced offline gate only | offline gate proxy, no online SR | n/a (offline) | **reduced offline rescue gate NO_GO** | missing B-vs-C (denoise) + downstream SR/IR |
| **M3** dynamic depth (Claim 3) | **not evaluated** | — | — | **NOT EVALUATED / owner-de-scoped** | missing fixed/oracle/dynamic depth data |

(warm-start is deliberately **not** a row here — it is un-reviewed, un-artifacted background only; see §6. It bears no weight in this verdict matrix.)

**Reading of the matrix.** The empirical synthesis covers **only the evaluated mechanisms**
— M2's evaluated dual full-hit operating points and M1's reduced offline gate. Across those,
**no evaluated operating point delivered both safety and compute savings**: M2's full-hit
substitution saves inference (IR↓) but crashes SR; M1's projection does not lift the gate's
offline discrimination. **M3 was not evaluated and warm-start is un-reviewed background —
neither enters this empirical conclusion.** Hence, within the evaluated boundary, no
safe-and-compute-saving operating point was observed (not a proof that none exists).

---

## 8. Provenance — traceable, not independently reproducible

- **Committed source-of-truth reports** used by this synthesis — grouped by the commit that
  added them: **commit `2d0e4cc`** (Phase 5/6): `phase5_scoresum_findings.md`,
  `phase5_scoresum_report_{spatial,l10}.md`, `phase6_ib_offline_gate_report.md`. (The Phase 4
  margin/provenance reports in commit `77ed0f5` establish the D⁺/D⁻ dual artifact but are
  **not** cited by any Claim verdict here, so they are not part of this report's source set.)
- **Metric definitions:** `exp/zixuan_proposal/analyze_phase5_rollout.py:80–82` (BHR/FFR/IR),
  `:6`,`:78` (episode-level `bad`/`safe_reuse` proxy).
- **Raw artifacts (gitignored, on `jupyter-ziyang10`):** Phase-5 Pass-3 gate JSONL +
  episode-results (`exp/zixuan_proposal/data/pass3_scoresum_{spatial,l10}_*`); Phase-6 ①b
  projection weights (`*_july_laneB.pt`), projected/raw I_cal libraries, calibration rows
  (`phase6_july_calib_rows{,_raw}_{spatial,l10}.jsonl`), gate params
  (`{proj,raw}_july_params_{spatial,l10}.json`).
- **Devices (Phase 5 Pass-3 eval topology, per `phase5_scoresum_findings.md`):** server =
  jupyter-ziyang10 (H200, shared/contended → `--replicas 1`); client = timan107 (12 workers);
  the illustrative and calibrated configs are run **sequentially, never concurrent**. (The
  `--non-concurrent` collection flag belongs to the earlier Pass-1 cache-OFF collection, not
  to this Pass-3 eval topology.)
- **Reproducibility boundary:** headline numbers are **traceable** to the committed reports
  above but **cannot be re-derived from the repo alone** (raw inputs are gitignored/remote).

---

## 9. Limitations and what would overturn the verdict

- **Escape hatches** (`phase5_scoresum_findings.md` §4): (a) make calibration SR-aware
  (needs an online/counterfactual signal `L_cal` cannot currently observe); (b) restrict
  substitution to trajectory segments with provably bounded divergence; (c) use the cache for
  warm-start only (never full action substitution) and re-measure.
- **BHR/FFR proxy** is episode-level, not per-step counterfactual (§2) — it cannot by itself
  attribute a failure to a specific cached action or prove universal unsafety.
- **Scope limits:** two suites (libero_spatial, libero_10), pi05, LIBERO, one
  `weighted_score_sum` + margin-gate parameterization. Nothing here generalizes beyond that
  boundary without new evaluation.
- **Unevaluated proposal-literal comparisons** (would be required to close the literal
  Claims): success-only vs dual (Claim 1); B-vs-C denoise + downstream online SR/IR
  (Claim 2); fixed/oracle/dynamic depth (Claim 3).
