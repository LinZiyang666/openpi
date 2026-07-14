"""Tests for the projection training driver + feasibility gate (plan §6.2/§8)."""

import numpy as np
import torch

from exp.zixuan_proposal.fit_projection import benchmark_epoch, fit_from_trainset
from openpi.cache.types import VISION_0, VISION_1


def test_benchmark_epoch_passes_small():
    r = benchmark_epoch(n=64, in_dim=32, out_dim=16)
    assert r["status"] == "PASS"
    assert r["seconds_per_epoch"] < 90.0
    assert r["peak_rss_gb"] < 32.0


def test_fit_from_trainset_masked_with_early_stop():
    n, in_dim, out_dim = 40, 8, 4
    rng = np.random.default_rng(0)
    labels = np.array([0] * 20 + [1] * 20)
    same = labels[:, None] == labels[None, :]
    eye = np.eye(n, dtype=bool)
    pos = same & ~eye
    neg = ~same
    feats = np.stack(
        [np.array([3.0, 0] + [0] * (in_dim - 2)) + 0.2 * rng.normal(size=in_dim) for _ in range(20)]
        + [np.array([0, 3.0] + [0] * (in_dim - 2)) + 0.2 * rng.normal(size=in_dim) for _ in range(20)]
    )
    # Interleave train/val within each class so the val fold has both classes (valid anchors).
    folds = []
    for i in range(n):
        folds.append("val" if i % 5 == 0 else "train")
    trainset = {
        "rows": [{"fold": f} for f in folds],
        "fields": {VISION_0: {"features": feats}, VISION_1: {"features": feats}},
        "masks": {"pos": pos, "neg": neg},
    }
    params, prov = fit_from_trainset(trainset, out_dim=out_dim, epochs=20)
    for field in (VISION_0, VISION_1):
        head = params.head(field)
        assert head is not None
        assert head.weight.shape == (out_dim, in_dim)
        assert torch.isfinite(head.weight).all()
        assert prov[field]["selected_epoch"] >= 0  # machine-readable checkpoint provenance


def test_train_gradient_invariant_to_val_feature_changes():
    # No validation leakage: perturbing ONLY the val-fold features must not change the
    # learned weights (train gradient uses train candidates only).
    n, in_dim = 40, 8
    labels = np.array([0] * 20 + [1] * 20)
    same = labels[:, None] == labels[None, :]
    pos = same & ~np.eye(n, dtype=bool)
    neg = ~same
    rng = np.random.default_rng(0)
    feats = np.stack(
        [np.array([3.0, 0] + [0] * (in_dim - 2)) + 0.2 * rng.normal(size=in_dim) for _ in range(20)]
        + [np.array([0, 3.0] + [0] * (in_dim - 2)) + 0.2 * rng.normal(size=in_dim) for _ in range(20)]
    )
    folds = ["val" if i % 5 == 0 else "train" for i in range(n)]
    ts = {"rows": [{"fold": f} for f in folds], "fields": {VISION_0: {"features": feats.copy()}}, "masks": {"pos": pos, "neg": neg}}
    # epochs=1: checkpoint selection is trivial (epoch 0), so any weight difference could only
    # come from the train gradient -> a clean test that the gradient does not see val features.
    p1, _ = fit_from_trainset(ts, out_dim=4, epochs=1)

    feats2 = feats.copy()
    for i, f in enumerate(folds):
        if f == "val":
            feats2[i] = rng.normal(size=in_dim) * 10  # perturb val rows only
    ts2 = {"rows": [{"fold": f} for f in folds], "fields": {VISION_0: {"features": feats2}}, "masks": {"pos": pos, "neg": neg}}
    p2, _ = fit_from_trainset(ts2, out_dim=4, epochs=1)
    assert torch.allclose(p1.head(VISION_0).weight, p2.head(VISION_0).weight)
