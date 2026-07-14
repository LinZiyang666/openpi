"""Phase 6.0 batch-separability acceptance gate (TRACER plan §6.0, owner-ruled D1=(b)).

Owner ruling (b): break the April(D+)/July(D-) collection-batch confound *at the data
level* by collecting a July-D+ control, then PROVE the confound is broken before any
projection training. This module is the frozen diagnostic.

Design (a) — D+-only matched-batch: among the D+ pool only, can a classifier predict
collection batch (April-D+ vs July-D+)? If not (AUROC ~ chance), batch is not an
exploitable signal. To keep batch from being confounded with task/init, everything is
done WITHIN matched ``(task_id, orig_init_state_idx)`` cells, on episode-level features,
with folds grouped by cell so the classifier must generalise batch ACROSS cells.

Frozen protocol (§6.0):
- features: one standardized episode-level vector per episode (caller mean-pools the
  per-step vision keys);
- folds: deterministic, stratified by batch, grouped by matched cell (a cell never
  splits across train/test); 5 folds by round-robin over sorted cells; a scored fold
  must contain BOTH batches;
- classifier: logistic regression, seed 7;
- min samples: >= 10 independent episodes per batch within matched cells, else INCONCLUSIVE;
- CI: matched-cell clustered bootstrap (resample cells, B=10_000, seed 7) of pooled
  out-of-fold AUROC -> 95% interval;
- verdict: PASS (confound broken) iff the CI upper bound <= 0.55.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

SEED = 7
N_FOLDS = 5
MIN_EPISODES_PER_BATCH = 10
B_REPLICATES = 10_000
AUROC_PASS_UPPER = 0.55
APRIL, JULY = 0, 1  # batch labels (the positive class for the classifier is JULY)


@dataclasses.dataclass(frozen=True)
class Episode:
    """One D+ episode for the batch-separability diagnostic.

    cell:       matched covariate cell (task_id, orig_init_state_idx).
    batch:      APRIL or JULY collection batch (the classifier target).
    features:   episode-level feature vector (already mean-pooled over steps).
    episode_id: the source trajectory id -- an INDEPENDENT-episode identity carried into the
                verdict's per-episode manifest so a consumer can prove the claimed n_april/n_july
                are backed by that many distinct real episodes (not one row reused).
    """

    cell: tuple
    batch: int
    features: np.ndarray
    episode_id: str = ""


# ------------------------------------------------------------------
# Matched cells + deterministic grouped folds
# ------------------------------------------------------------------
def _matched_cells(episodes: list[Episode]) -> list[tuple]:
    """Cells that contain BOTH batches (the only cells where batch is the sole factor)."""
    batches_by_cell: dict = {}
    for e in episodes:
        batches_by_cell.setdefault(e.cell, set()).add(e.batch)
    return sorted(c for c, bs in batches_by_cell.items() if {APRIL, JULY} <= bs)


def episode_feature_digest(features) -> str:
    """CANONICAL SHA-256 of one raw episode-level feature vector.

    Frozen byte spec so the gate and the build-time consumer agree exactly: cast to float64,
    round to 6 decimals, C-contiguous ``tobytes()``. Binds the classifier's ACTUAL input bytes,
    so a run that fed the gate constant/fabricated features (to fake a chance-AUROC PASS) records
    a digest that will not match the real mean-pooled vision reconstructed from the artifact.
    """
    arr = np.ascontiguousarray(np.asarray(features, dtype=np.float64).round(6))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def episode_feature_from_entries(entries) -> np.ndarray:
    """The CANONICAL raw episode-level feature the gate must be run on: the per-``vision_*``-field
    mean over the episode's step rows, concatenated in sorted field order (float64).

    Shared by the gate driver (to build ``Episode.features``) and the build-time verifier (to
    reconstruct the same vector from the supplied artifact), so the ``episode_feature_digest``
    binds the verdict to the artifact's actual vision content. ``entries`` are per-step cache
    entries exposing ``.query_keys``.
    """
    fields = sorted({f for e in entries for f in e.query_keys if f.startswith("vision")})
    if not fields:
        raise ValueError("no vision_* query keys on the episode entries; cannot build canonical feature")
    parts = []
    for f in fields:
        vecs = [np.asarray(e.query_keys[f], dtype=np.float64) for e in entries if f in e.query_keys]
        parts.append(np.mean(np.stack(vecs), axis=0))
    return np.concatenate(parts)


def episode_manifest(episodes: list[Episode]) -> list:
    """Canonical per-episode input manifest: one ``[batch, episode_id, [task, init], feature_digest]``
    row per matched-cell episode, sorted. This is the artifact the digest binds -- it carries the
    INDEPENDENT-episode identities AND the classifier feature bytes, so the claimed n_april/n_july
    cannot be inflated beyond the distinct episode_ids actually present, and the recorded PASS
    cannot have been computed on fabricated features (the feature digest must reconstruct from the
    supplied artifact).
    """
    rows = [
        [int(e.batch), str(e.episode_id), [int(e.cell[0]), int(e.cell[1])], episode_feature_digest(e.features)]
        for e in episodes
    ]
    return sorted(rows)


def gate_input_digest(episodes_manifest: list, n_april: int, n_july: int, ci_high: float) -> str:
    """Tamper-evident SHA-256 over the verdict's canonical PER-EPISODE manifest + counts + CI.

    Unlike a cell-set-only digest, this includes every independent episode's ``(batch, id, cell)``,
    so editing the JSON to inflate n_april/n_july forces forging that many distinct episode rows --
    which the consumer then requires to correspond, id-for-id, to the actually-supplied base/control
    artifacts (``_verify_verdict_binding``). ``episodes_manifest`` is the output of
    ``episode_manifest`` (or the equivalently-shaped list persisted in the verdict).
    """
    payload = json.dumps(
        {
            "episode_manifest": sorted([list(r) for r in episodes_manifest]),
            "n_april": int(n_april),
            "n_july": int(n_july),
            "ci_high": round(float(ci_high), 6),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _fold_of_cell(cells: list[tuple]) -> dict:
    """Assign each cell to a fold by round-robin over the sorted cell list (deterministic)."""
    return {c: (i % N_FOLDS) for i, c in enumerate(cells)}


# ------------------------------------------------------------------
# Minimal deterministic classifier + AUROC (numpy only; no sklearn dep)
# ------------------------------------------------------------------
def _auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), average-rank tie handling."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    rank_sorted = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        rank_sorted[i : j + 1] = (i + j) / 2.0 + 1.0  # average rank (1-based)
        i = j + 1
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = rank_sorted
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fit_logreg(x: np.ndarray, y: np.ndarray, *, l2: float = 1.0, iters: int = 500, lr: float = 0.1):
    """L2-regularized logistic regression by full-batch gradient descent (deterministic)."""
    n, d = x.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(x @ w + b)))
        gw = x.T @ (p - y) / n + l2 * w / n
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def _predict_proba(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x @ w + b)))


# ------------------------------------------------------------------
# Gate
# ------------------------------------------------------------------
def run_gate(
    episodes: list[Episode],
    *,
    seed: int = SEED,
    b_replicates: int = B_REPLICATES,
) -> dict:
    """Run the frozen batch-separability gate. Returns a verdict dict.

    status in {PASS, FAIL, INCONCLUSIVE}. PASS = confound broken (CI upper bound <= 0.55).
    """
    cells = _matched_cells(episodes)
    matched = [e for e in episodes if e.cell in set(cells)]
    n_april = sum(1 for e in matched if e.batch == APRIL)
    n_july = sum(1 for e in matched if e.batch == JULY)
    if n_april < MIN_EPISODES_PER_BATCH or n_july < MIN_EPISODES_PER_BATCH:
        return {
            "status": "INCONCLUSIVE",
            "reason": f"need >= {MIN_EPISODES_PER_BATCH} episodes/batch in matched cells; "
            f"got april={n_april} july={n_july} over {len(cells)} matched cells",
        }

    fold_of = _fold_of_cell(cells)
    X = np.stack([e.features for e in matched]).astype(np.float64)
    y = np.array([e.batch for e in matched])
    fold = np.array([fold_of[e.cell] for e in matched])

    # Out-of-fold predictions; a scored fold must hold both batches in train AND test.
    oof = np.full(len(matched), np.nan)
    scored = np.zeros(len(matched), dtype=bool)
    for f in range(N_FOLDS):
        te = fold == f
        tr = ~te
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue  # fold not scorable; reported via coverage below
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0) + 1e-8
        w, b = _fit_logreg((X[tr] - mu) / sd, y[tr].astype(float))
        oof[te] = _predict_proba((X[te] - mu) / sd, w, b)
        scored[te] = True

    if not scored.any() or len(set(y[scored])) < 2:
        return {"status": "INCONCLUSIVE", "reason": "no scorable fold with both batches"}

    point_auroc = _auroc(y[scored], oof[scored])

    # Matched-cell clustered bootstrap of the pooled OOF AUROC.
    scored_cells = sorted({matched[i].cell for i in np.where(scored)[0]})
    rows_by_cell: dict = {}
    for i in np.where(scored)[0]:
        rows_by_cell.setdefault(matched[i].cell, []).append(i)
    rng = np.random.default_rng(seed)
    aurocs = []
    cidx = np.arange(len(scored_cells))
    for _ in range(b_replicates):
        pick = rng.choice(cidx, size=len(scored_cells), replace=True)
        rows: list[int] = []
        for j in pick:
            rows.extend(rows_by_cell[scored_cells[j]])
        yy, pp = y[rows], oof[rows]
        if len(set(yy)) < 2:
            continue
        aurocs.append(_auroc(yy, pp))
    lo, hi = np.quantile(aurocs, [0.025, 0.975])

    # PASS requires a FINITE upper bound at/under the frozen chance ceiling; a NaN interval
    # (e.g. a degenerate bootstrap) must never be certified as "confound broken".
    status = "PASS" if (np.isfinite(hi) and hi <= AUROC_PASS_UPPER) else "FAIL"
    manifest = episode_manifest(matched)
    return {
        "status": status,
        "auroc": point_auroc,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_matched_cells": len(cells),
        "n_april": n_april,
        "n_july": n_july,
        "matched_cells": [list(c) for c in cells],
        "episode_manifest": manifest,  # canonical per-episode inputs the digest binds
        "input_digest": gate_input_digest(manifest, n_april, n_july, hi),
    }
