"""Offline pre-GPU go/no-go for a projected lane (TRACER plan §6.7).

Purpose: before spending Pass-3 GPU on a projected lane, decide offline whether the
lane's CALIBRATED gate separates safe from unsafe reuse better than the raw ablation.

Two distinct signals (plan §6.7 / §C), deliberately NOT conflated:
  * INFERENTIAL GO/NO-GO = AUROC of the calibrated gate score ``g_t`` (equivalently the
    margin ``m_t = s_pos - lambda*s_neg`` at the solved lambda) vs the safe-reuse proxy
    ``episode_success``, computed on the FULL LOEO calibration table (hundreds of
    episode clusters -> adequate power). GO iff the lane beats the raw baseline A with an
    episode-clustered bootstrap CI on delta-AUROC excluding 0.
  * Retrieval@K-compat is DESCRIPTIVE only (5-6 clusters; see §C) and is NEVER a
    threshold here -- reported, not gated on.

The calibrated-``g_t`` AUROC reuses the frozen episode-clustered bootstrap in
``phase6_stats`` (resample by (task_id, init_state_idx)).
"""

from __future__ import annotations

import numpy as np

from exp.zixuan_proposal.phase6_batch_separability import _auroc

SEED = 7
B_REPLICATES = 10_000
CI_LEVEL = 0.95


def _cluster_bootstrap_delta_auroc(
    cluster: np.ndarray,
    label: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    seed: int = SEED,
    b_replicates: int = B_REPLICATES,
    ci_level: float = CI_LEVEL,
) -> dict:
    """Episode-clustered bootstrap of ``AUROC_b - AUROC_a`` over a shared row set.

    ``label`` is the safe-reuse proxy (1 = safe/success). Rows are paired: the same
    resampled clusters index both lanes' scores. Returns point delta + percentile CI +
    ``excludes_zero``.
    """
    uniq = sorted(set(map(tuple, cluster)))
    rows_by_cell: dict = {}
    for i, c in enumerate(map(tuple, cluster)):
        rows_by_cell.setdefault(c, []).append(i)
    point = _auroc(label, score_b) - _auroc(label, score_a)
    rng = np.random.default_rng(seed)
    cidx = np.arange(len(uniq))
    deltas = []
    for _ in range(b_replicates):
        pick = rng.choice(cidx, size=len(uniq), replace=True)
        rows: list[int] = []
        for j in pick:
            rows.extend(rows_by_cell[uniq[j]])
        rows_arr = np.asarray(rows)
        yy = label[rows_arr]
        if len(set(yy.tolist())) < 2:
            continue
        deltas.append(_auroc(yy, score_b[rows_arr]) - _auroc(yy, score_a[rows_arr]))
    alpha = 1.0 - ci_level
    lo, hi = np.quantile(deltas, [alpha / 2, 1.0 - alpha / 2])
    return {
        "point": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "auroc_a": float(_auroc(label, score_a)),
        "auroc_b": float(_auroc(label, score_b)),
    }


def offline_go_no_go(
    cluster: np.ndarray,
    safe_reuse: np.ndarray,
    gate_score_raw_a: np.ndarray,
    gate_score_projected_b: np.ndarray,
    **kw,
) -> dict:
    """GO iff the projected lane's calibrated gate beats A's with the CI excluding 0.

    Returns status in {GO, NO_GO}. A NO_GO is a documented result (no GPU spent), not a
    failure of the pipeline.
    """
    res = _cluster_bootstrap_delta_auroc(
        cluster, safe_reuse, gate_score_raw_a, gate_score_projected_b, **kw
    )
    go = res["excludes_zero"] and res["ci_low"] > 0  # b strictly better than a
    return {"status": "GO" if go else "NO_GO", **res}
