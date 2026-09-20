"""Deviation kernel and per-decision quantities of the step-vs-warm-start line (plan §2.0 / Q-A).

Every distance in this line is ``openpi.cache.components.surface_judge.weighted_chunk_deviation``:
the mean over the first ``h_exec`` steps of the weighted L2 norm across the executed dims of two
``[H, D]`` chunks in the model's normalised action space. Chunks are compared in that space (the
staged stage-3 output, the same space the library stores), never after the adapter's
un-normalisation.

Quantities per decision (``N`` private-noise samples ``a_full[n]`` at the full step count K and,
for every reduced step count k, ``a_k[n]`` from the *same* initial noise ``z[n]``):

* ``d_k    = mean_n D(a_k[n], a_full[n])``      paired-noise reduction error (non-negative; 0 at k=K)
* ``disp_K = mean_{n<n'} D(a_full[n], a_full[n'])``  conditional dispersion of the full policy
* ``r_k    = d_k / disp_K``                       only when ``disp_K > 1e-8``, else None
* ``d_w(t) = mean_n D(a_warm(t), a_full[n])``     warm-start continuation vs the full samples

``excess_vs_spread = d_k - disp_K`` may be listed but is descriptive only (it mixes a paired error
with an unpaired spread, which is why v2's "floor subtraction" was dropped).

Action weights are frozen per environment from the library chunks: ``w_d = 1/sigma_d`` on the
executed dims (``w_d = 1`` when ``sigma_d <= 1e-6``, recorded as a degenerate dim), ``w_d = 0`` on
the non-executed dims. ``compute_library_action_weights`` is deliberately not used for ``w``
because it zeroes low-variance dims, which would silently drop a constant executed dim.

The mixture fit reuses ``exp.dp_nfe.analysis.dispersion_index.mixture_diagnostic`` on the
executed window scaled by the same weights; it is a fit diagnostic, not a mode count.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from openpi.cache.components.surface_judge import weighted_chunk_deviation

DEGENERATE_SIGMA = 1e-6
DISP_EPS = 1e-8


# ------------------------------------------------------------------
# Frozen weights
# ------------------------------------------------------------------


def executed_mask(action_dim: int, n_executed: int) -> torch.Tensor:
    """Boolean ``[D]`` mask selecting the first ``n_executed`` dims (the executed contract)."""
    if not 0 < n_executed <= action_dim:
        raise ValueError(f"n_executed must be in (0, {action_dim}], got {n_executed}")
    mask = torch.zeros(action_dim, dtype=torch.bool)
    mask[:n_executed] = True
    return mask


def frozen_action_weights(
    library_chunks: torch.Tensor, active_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """``(w [D], sigma [D], degenerate_dims)`` from ``[N, H, D]`` library chunks and the frozen mask.

    ``sigma`` is the population std over all ``N x H`` rows. Executed dims with
    ``sigma <= DEGENERATE_SIGMA`` keep weight 1 (they are still compared, in normalised units) and
    are returned so the manifest can record them; non-executed dims get weight 0.
    """
    if library_chunks.ndim != 3:
        raise ValueError(f"library chunks must be [N, H, D], got {tuple(library_chunks.shape)}")
    if active_mask.shape != (library_chunks.shape[-1],):
        raise ValueError("active_mask must be [D]")
    flat = library_chunks.reshape(-1, library_chunks.shape[-1]).to(torch.float32)
    sigma = flat.std(dim=0, unbiased=False)
    degenerate = [int(d) for d in torch.nonzero(active_mask & (sigma <= DEGENERATE_SIGMA)).flatten()]
    w = torch.where(sigma > DEGENERATE_SIGMA, 1.0 / sigma.clamp_min(DEGENERATE_SIGMA), torch.ones_like(sigma))
    w = torch.where(active_mask, w, torch.zeros_like(w))
    return w.to(torch.float32), sigma.to(torch.float32), degenerate


# ------------------------------------------------------------------
# Distances
# ------------------------------------------------------------------


def _chunk(x: Any) -> torch.Tensor:
    """``[H, D]`` float32 CPU tensor from a tensor / array, dropping a leading batch dim of 1."""
    t = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x).detach().to("cpu", torch.float32)
    if t.ndim == 3:
        if t.shape[0] != 1:
            raise ValueError(f"expected batch 1, got {tuple(t.shape)}")
        t = t[0]
    if t.ndim != 2:
        raise ValueError(f"chunk must be [H, D], got {tuple(t.shape)}")
    return t


def deviation(a: Any, b: Any, w: torch.Tensor, active_mask: torch.Tensor, h_exec: int) -> float:
    """The frozen kernel on two chunks (see module docstring)."""
    return weighted_chunk_deviation(_chunk(a), _chunk(b), w, active_mask, h_exec)


def paired_deviation(a_k: Sequence[Any], a_full: Sequence[Any], w, active_mask, h_exec: int) -> float:
    """``d_k``: mean over samples of ``D(a_k[n], a_full[n])`` (same initial noise per ``n``)."""
    if len(a_k) != len(a_full) or not a_k:
        raise ValueError("paired samples must be non-empty and equal in count")
    return float(np.mean([deviation(x, y, w, active_mask, h_exec) for x, y in zip(a_k, a_full)]))


def dispersion(a_full: Sequence[Any], w, active_mask, h_exec: int) -> float:
    """``disp_K``: mean pairwise deviation among the full-step samples (needs >= 2 samples)."""
    n = len(a_full)
    if n < 2:
        raise ValueError("dispersion needs at least two samples")
    vals = [deviation(a_full[i], a_full[j], w, active_mask, h_exec) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))


def warm_deviation(a_warm: Any, a_full: Sequence[Any], w, active_mask, h_exec: int) -> float:
    """``d_w(t)``: mean deviation of one warm-start chunk to every full sample (unpaired noise)."""
    if not a_full:
        raise ValueError("warm deviation needs full samples")
    return float(np.mean([deviation(a_warm, y, w, active_mask, h_exec) for y in a_full]))


def ratio(d: float, disp: float) -> Optional[float]:
    """``r_k = d / disp`` when the dispersion is above ``DISP_EPS``; None otherwise (never inf)."""
    if disp is None or not math.isfinite(disp) or disp <= DISP_EPS:
        return None
    return float(d / disp)


# ------------------------------------------------------------------
# Per-decision summary
# ------------------------------------------------------------------


def decision_metrics(
    a_full: Sequence[Any],
    a_k: Dict[int, Sequence[Any]],
    a_warm: Dict[float, Any],
    w: torch.Tensor,
    active_mask: torch.Tensor,
    h_exec: int,
    n_primary: int = 4,
) -> dict:
    """All plan §2 quantities of one decision from its recorded arrays.

    Only the first ``n_primary`` full samples enter ``d_k`` / ``disp_K`` / ``r_k`` (dense
    decisions carry more full samples for the mixture fit only). Raises when fewer than
    ``n_primary`` full samples or paired ``a_k`` samples are present, so the caller can reject
    the decision instead of computing on a partial matrix.
    """
    if len(a_full) < n_primary:
        raise ValueError(f"need {n_primary} full samples, got {len(a_full)}")
    full = list(a_full[:n_primary])
    disp = dispersion(full, w, active_mask, h_exec)
    out: dict = {"disp_K": disp, "n_primary": n_primary, "d": {}, "r": {}, "excess_vs_spread": {}, "d_w": {}}
    for k, samples in a_k.items():
        if len(samples) < n_primary:
            raise ValueError(f"k={k}: need {n_primary} paired samples, got {len(samples)}")
        d = paired_deviation(list(samples[:n_primary]), full, w, active_mask, h_exec)
        out["d"][int(k)] = d
        out["r"][int(k)] = ratio(d, disp)
        out["excess_vs_spread"][int(k)] = d - disp
    for t, chunk in a_warm.items():
        out["d_w"][float(t)] = None if chunk is None else warm_deviation(chunk, full, w, active_mask, h_exec)
    return out


# ------------------------------------------------------------------
# Mixture fit on dense decisions
# ------------------------------------------------------------------


def mixture_fit(
    a_full: Sequence[Any], w: torch.Tensor, active_mask: torch.Tensor, h_exec: int, seed: int = 20260919
) -> dict:
    """PCA-2D / GMM(2 vs 1) delta-BIC of the weighted executed window of the full samples.

    Returns ``dispersion_index.mixture_diagnostic``'s dict (``delta_bic_2_vs_1``,
    ``silhouette_pc1_split``, ``pca_explained_2d``, ``n_samples``) plus ``n_dims``; all-None
    values mean the fit was not attempted (constant / rank-deficient input) — never an exception.
    """
    from exp.dp_nfe.analysis.dispersion_index import mixture_diagnostic

    mask = active_mask.to(torch.bool)
    rows = []
    for x in a_full:
        c = _chunk(x)[:h_exec][:, mask] * w.to(torch.float32)[mask]
        rows.append(c.reshape(1, c.shape[0], c.shape[1]).numpy())
    samples = np.concatenate(rows, axis=0).astype(np.float64)  # [N, h_exec, D_exec]
    n_dims = int(mask.sum())
    flat = samples.reshape(samples.shape[0], -1)
    null = {"delta_bic_2_vs_1": None, "silhouette_pc1_split": None, "pca_explained_2d": None,
            "n_samples": int(samples.shape[0]), "n_dims": n_dims}
    if samples.shape[0] < 4 or float(np.ptp(flat, axis=0).max()) == 0.0:
        return null | {"reason": "constant_or_too_few"}
    if np.linalg.matrix_rank(flat - flat.mean(axis=0)) < 2:
        return null | {"reason": "rank_deficient"}
    try:
        res = mixture_diagnostic(samples, np.ones(n_dims), seed=seed)
    except Exception:  # noqa: BLE001 - a failed fit is a null label, not an error
        res = {"delta_bic_2_vs_1": None, "silhouette_pc1_split": None, "pca_explained_2d": None,
               "n_samples": int(samples.shape[0])}
    res["n_dims"] = n_dims
    return res


def finite(values: Iterable[Any]) -> List[float]:
    """The finite floats of an iterable (None / NaN / inf dropped)."""
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out
