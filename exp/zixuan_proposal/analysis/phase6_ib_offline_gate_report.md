# TRACER Phase 6 ①b — Confound-Free Offline GO/NO-GO for Mechanism-1

**Date:** 2026-07-14 · **Suites:** libero_spatial, libero_10 · **Verdict:** NO_GO (both suites)

## 1. Motivation

Phase 6.0's batch-separability gate **FAILED** (April-D+ vs July-D+ vision AUROC 0.850,
CI[0.776, 0.918], PASS threshold 0.55). Root cause was a **structural rendering drift**
between the April-2026 and July-2026 LIBERO collections (per-pixel edge/texture/anti-alias
differences that survive mean-offset removal — not a brightness shift), which the vision
encoder amplifies into linearly-separable embeddings. This confound is not fixable by
normalization: any D+ pool drawn from April is trivially separable from July-env D−.

**Owner ruling ① (2026-07-14):** discard the April D+ pool; draw D+ from the **same July
run** as D− so the failure library and the positive library share one rendering
distribution (confound eliminated). Rebuild the projection data base, projection head, and
recalibration entirely on July-only data. Suite ①b runs this end-to-end for both suites.

## 2. Method

- **Data base (confound-free):** D+ = July successes, D− = July failures, both from the
  `failure_heldout` collection (single July run). Identity resolution uses the uniform gid
  convention `task = gid // 50`, `init = gid % 50`; I_cal = even-init cells, I_val = odd.
- **Projection (lane B):** outcome-compatible dense head (out_dim 2048), masked InfoNCE,
  `eta = 1.0`, 200 epochs, per-vision-stream early selection.
- **Projected library:** fast in-memory transform `z = x Wᵀ` on the pooled dual (no h5
  re-read).
- **Recalibration (Phase-5 machinery, July-only):** LOEO z-score normalizer refit →
  Pass-2 replay through `DualRetrievalKnnStrategy` (chain-aware, depth 5) over the I_cal
  library → `MarginGateCalibrator` solve (c_miss 1.0, c_warm 0.75).
- **Offline GO/NO-GO:** for each I_cal step, `g_t = σ(b0 + margin + b3·Δ⁺)`; compare the
  g_t→`episode_success` AUROC of projected-B vs raw-A′ under an **episode-clustered
  bootstrap**. GO requires the ΔAUROC (B − A′) CI to exclude 0 with a positive lower bound.

A′ (raw) is the identical pipeline with the key builder de-projected to `cp1_mean_pool`
(same depth-5 chain, same margin gate), isolating the projection as the only variable.

## 3. Results

| Suite    | I_cal episodes | succ. rate | calib rows | AUROC raw A′ | AUROC proj B | ΔAUROC (B−A′) | 95% CI            | Verdict |
|----------|---------------:|-----------:|-----------:|-------------:|-------------:|--------------:|-------------------|---------|
| spatial  | 250            | 96.0%      | 5 550      | 0.8156       | 0.8194       | +0.0038       | [−0.011, +0.018]  | NO_GO   |
| libero_10| 250            | 82.8%      | 15 007     | 0.7556       | 0.7600       | +0.0044       | [−0.0031, +0.0115]| NO_GO   |

Supporting signals (libero_10): projection trainset `valid_anchor_frac = 0.999` (vs the
April confounded data's 0.478, which had failed); solved gate betas are near-identical
between B and A′ (`b0 ≈ −0.737`, `b3 = 1.0`), i.e. the projection barely moves the gate;
Miss cost 0.0 for both.

## 4. Conclusion

On **confound-free** July data, with the projection head properly trained and cleanly
converged, **Mechanism-1's outcome-compatible projection provides no statistically
significant benefit over raw pooled cosine** for the failure-aware gate's safe-reuse
prediction — the ΔAUROC CI includes zero in **both** suites. Raw pooled cosine already
reaches the same AUROC (~0.82 spatial, ~0.76 l10).

libero_10 is the cleaner test (better class balance, 17.2% failures vs spatial's 4.0%) and
reaches the same verdict as libero_spatial, so the negative result is corroborated across
suites rather than an artifact of one suite's imbalance. This is **strong negative evidence
for Mechanism-1**. Neither suite's Pass-3 paired rollout was run (NO_GO ⇒ not worth the
GPU).

## 5. Provenance

All artifacts live on `jupyter-ziyang10` under `exp/zixuan_proposal/data/` and
`exp/common/data/cache_artifacts/{libero_spatial,libero_10}/`:
projection weights `*_july_laneB.pt`, projected libraries `cp1_proj_july_dual.pkl`,
I_cal libraries `cp1_{proj,mean_pool}_july_ical_dual.pkl`, calibration rows
`phase6_july_calib_rows{,_raw}_{spatial,l10}.jsonl`, gate params
`{proj,raw}_july_params_{spatial,l10}.json`.

An in-experiment fix to `exp/zixuan_proposal/projection_labels.py::compat_matrix` (the
`‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b` identity, avoiding an `[n,n,D]` OOM at libero_10 scale) and
BLAS thread caps (`OMP/OPENBLAS/MKL=10` on ziyang10's 10-core cgroup) are applied on the
box but not yet committed.
