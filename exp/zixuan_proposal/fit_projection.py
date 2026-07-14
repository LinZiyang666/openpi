"""Projection-head training driver + feasibility gate (TRACER Phase 6, §6.2/§8).

Two responsibilities:
  * ``benchmark_epoch`` -- the BLOCKING feasibility gate (§8). A dense 2048x2048 head is
    ~4.19M params/field and the fitter does an N^2 similarity matrix per epoch, so before
    Code-time training we MEASURE one full-batch masked-InfoNCE epoch at the real even-init
    D+ N and check the frozen caps (<= 90 s/epoch AND <= 32 GB peak RSS). Exceeding a cap
    does NOT silently mutate the approved design -- it returns to Plan/G1.
  * ``fit_from_trainset`` -- load a ``<suite>.pt`` trainset (features + symmetric P/N masks
    per projected field) and fit the heads via ``ProjectionKeyBuilder.fit(loss="masked")``,
    saving ``ProjectionParams`` to disk.

Runs offline on CPU (heads are tiny linear maps). The masked InfoNCE it invokes is the
unit-tested ``proj_infonce_loss`` in ``projection_key_builder``.
"""

from __future__ import annotations

import logging
import resource
import time

import torch

from openpi.cache.components.projection_key_builder import proj_infonce_loss

_log = logging.getLogger(__name__)

# Frozen feasibility caps (plan §8).
MAX_SECONDS_PER_EPOCH = 90.0
MAX_PEAK_RSS_GB = 32.0
DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-2
DEFAULT_TEMPERATURE = 0.07


def _peak_rss_gb() -> float:
    """Process peak resident set size in GB (ru_maxrss is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


# ------------------------------------------------------------------
# Feasibility gate (§8)
# ------------------------------------------------------------------
def benchmark_epoch(n: int, in_dim: int, out_dim: int, *, seed: int = 7) -> dict:
    """Time ONE full-batch masked-InfoNCE epoch at (n, in_dim, out_dim) on synthetic data.

    Returns wall-time, peak RSS, and PASS/FAIL vs the frozen caps. A FAIL means the
    approved full-batch design must return to Plan/G1 (not mutate at Code time).
    """
    torch.manual_seed(seed)
    features = torch.randn(n, in_dim)
    # Two-cluster synthetic masks so the loss is well formed (>=1 pos & >=1 neg / anchor).
    half = n // 2
    labels = torch.tensor([0] * half + [1] * (n - half))
    same = labels.view(-1, 1) == labels.view(1, -1)
    eye = torch.eye(n, dtype=torch.bool)
    pos = same & ~eye
    neg = ~same
    weight = (torch.randn(out_dim, in_dim) * (in_dim ** -0.5)).requires_grad_(True)
    opt = torch.optim.Adam([weight], lr=DEFAULT_LR)

    t0 = time.perf_counter()
    opt.zero_grad()
    loss = proj_infonce_loss(features @ weight.t(), pos, neg, DEFAULT_TEMPERATURE)
    loss.backward()
    opt.step()
    secs = time.perf_counter() - t0
    rss = _peak_rss_gb()
    ok = secs <= MAX_SECONDS_PER_EPOCH and rss <= MAX_PEAK_RSS_GB
    return {
        "status": "PASS" if ok else "FAIL",
        "seconds_per_epoch": secs,
        "peak_rss_gb": rss,
        "caps": {"seconds": MAX_SECONDS_PER_EPOCH, "rss_gb": MAX_PEAK_RSS_GB},
    }


# ------------------------------------------------------------------
# Fit driver
# ------------------------------------------------------------------
def _submask(mask: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return mask.index_select(0, idx).index_select(1, idx)


def _has_valid_anchor(pos: torch.Tensor, neg: torch.Tensor) -> bool:
    eye = torch.eye(pos.shape[0], dtype=torch.bool)
    return bool(((pos & ~eye).any(dim=1) & (neg & ~eye).any(dim=1)).any())


def _cross_infonce(z_a: torch.Tensor, z_c: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, temperature: float) -> torch.Tensor:
    """Rectangular masked InfoNCE: anchors ``z_a`` [na,d] retrieve from candidates ``z_c`` [nc,d].

    ``pos``/``neg`` are [na, nc]. Used for the VALIDATION loss so val anchors are scored against
    TRAIN candidates only -- with ``z_c`` detached this makes the val loss carry no gradient into
    the train weights (no validation leakage). Raises if no val anchor is scorable.
    """
    import torch.nn.functional as F

    za = F.normalize(z_a.float(), dim=1)
    zc = F.normalize(z_c.float(), dim=1)
    sim = (za @ zc.t()) / temperature  # [na, nc]
    valid = pos.any(dim=1) & neg.any(dim=1)
    if not bool(valid.any()):
        raise ValueError("no val anchor with both a positive and a negative among train candidates")
    neg_inf = torch.finfo(sim.dtype).min
    num = torch.logsumexp(torch.where(pos, sim, torch.full_like(sim, neg_inf)), dim=1)
    den = torch.logsumexp(torch.where(pos | neg, sim, torch.full_like(sim, neg_inf)), dim=1)
    return -(num - den)[valid].mean()


def fit_from_trainset(
    trainset: dict, *, out_dim: int, epochs: int = DEFAULT_EPOCHS, lr: float = DEFAULT_LR
):
    """Fit projection heads with early-stopping on the early-stop-val fold (plan §6.2).

    ``trainset`` = {"rows": [{"fold": ...}], "fields": {field: {"features": [N, in_dim]}},
    "masks": {"pos": [N,N], "neg": [N,N]}}. The **train gradient uses TRAIN rows only** (train
    anchors + train candidates); the val loss is computed under ``no_grad`` with VAL anchors
    scored against TRAIN candidates, so val features never enter the train gradient (no
    validation leakage). Selects the checkpoint with the lowest val loss; fails loud if train
    or val is unscorable. Returns ``(ProjectionParams, provenance)`` where provenance records
    the selected epoch + val loss per field (machine-readable).
    """
    from openpi.cache.components.projection_key_builder import ProjectionHead, ProjectionParams

    folds = [r["fold"] for r in trainset["rows"]]
    train_sel = torch.tensor([i for i, f in enumerate(folds) if f == "train"], dtype=torch.long)
    val_sel = torch.tensor([i for i, f in enumerate(folds) if f == "val"], dtype=torch.long)
    pos_full = torch.as_tensor(trainset["masks"]["pos"]).bool()
    neg_full = torch.as_tensor(trainset["masks"]["neg"]).bool()
    pos_tr, neg_tr = _submask(pos_full, train_sel), _submask(neg_full, train_sel)
    # val anchors x TRAIN candidates (rectangular)
    pos_vt = pos_full.index_select(0, val_sel).index_select(1, train_sel)
    neg_vt = neg_full.index_select(0, val_sel).index_select(1, train_sel)
    if not _has_valid_anchor(pos_tr, neg_tr):
        raise ValueError("train fold has no anchor with both a positive and a negative")
    if len(val_sel) == 0 or not bool((pos_vt.any(dim=1) & neg_vt.any(dim=1)).any()):
        raise ValueError(
            "early-stop-val fold has no valid anchor vs train candidates; cannot checkpoint-select "
            "per §6.2 (collect more val coverage or return to Plan/G1 for an alternative)"
        )

    heads: dict = {}
    provenance: dict = {}
    for field, d in trainset["fields"].items():
        feats = torch.as_tensor(d["features"]).float()
        x_tr = feats.index_select(0, train_sel)
        x_va = feats.index_select(0, val_sel)
        in_dim = feats.shape[1]
        torch.manual_seed(7)
        w = (torch.randn(out_dim, in_dim) * (in_dim ** -0.5)).requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        best_w, best_val, best_epoch = w.detach().clone(), float("inf"), -1
        for ep in range(epochs):
            opt.zero_grad()
            proj_infonce_loss(x_tr @ w.t(), pos_tr, neg_tr, DEFAULT_TEMPERATURE).backward()  # TRAIN only
            opt.step()
            with torch.no_grad():  # val anchors vs frozen train candidates -> no leakage
                vl = float(_cross_infonce(x_va @ w.t(), x_tr @ w.t(), pos_vt, neg_vt, DEFAULT_TEMPERATURE))
            if vl < best_val:
                best_val, best_w, best_epoch = vl, w.detach().clone(), ep
        _log.info("field %s: early-stop selected epoch %d (val_loss=%.4f)", field, best_epoch, best_val)
        provenance[field] = {"selected_epoch": best_epoch, "val_loss": best_val}
        heads[field] = ProjectionHead(weight=best_w.cpu().float().contiguous(), bias=None)
    return ProjectionParams(heads=heads), provenance
