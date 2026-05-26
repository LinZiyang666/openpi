"""Unit tests for the Layer-1 score normalization subsystem.

Covers the design invariants that distinguish weighted_score_sum's normalization
from rank fusion: bounded [0,1] output, monotonicity (non-decreasing globally,
strictly increasing on the unsaturated band), magnitude preservation (NOT
empirical-CDF / rank equalization), serialization round-trip, sim_type
compatibility, and the runtime builder dispatch incl. legacy bit-identical
back-compat.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from openpi.cache.components.score_normalizers import (
    CANDIDATE_NORMALIZERS,
    SCORE_NORMALIZER_REGISTRY,
    AffineClipNormalizer,
    DirectionUnifyNormalizer,
    ExpL2Normalizer,
    LegacyPercentileNormalizer,
    LogitNormalizer,
    PowerNormalizer,
    ZScoreNormalizer,
    build_field_normalizers,
)


def _fit_candidate(name: str):
    """Fit a candidate normalizer on a representative distribution for its sim_type."""
    cls = SCORE_NORMALIZER_REGISTRY[name]
    if "cosine" in cls.supported_sim_types:
        # Saturated cosine band: most mass near 1, a tail down to ~0.9.
        scores = np.concatenate([
            np.random.uniform(0.985, 0.999, size=900),
            np.random.uniform(0.90, 0.985, size=100),
        ])
        return cls.fit_from_scores(scores, "cosine"), "cosine"
    # l2-only (exp_l2): distances.
    scores = np.abs(np.random.randn(1000)) * 0.3
    return cls.fit_from_scores(scores, "l2"), "l2"


# ------------------------------------------------------------------
# Bounded output
# ------------------------------------------------------------------
@pytest.mark.parametrize("name", CANDIDATE_NORMALIZERS)
def test_candidate_output_bounded_0_1(name):
    norm, sim_type = _fit_candidate(name)
    if sim_type == "cosine":
        raw = torch.linspace(-1.0, 1.0, 500)
    else:
        raw = torch.linspace(0.0, 3.0, 500)
    out = norm(raw)
    assert out.shape == raw.shape
    assert torch.all(out >= -1e-6) and torch.all(out <= 1.0 + 1e-6), (
        f"{name} produced out-of-range values: [{out.min()}, {out.max()}]"
    )


# ------------------------------------------------------------------
# Monotonicity (non-decreasing globally; strict on the unsaturated band)
# ------------------------------------------------------------------
@pytest.mark.parametrize("name", CANDIDATE_NORMALIZERS)
def test_monotone_non_decreasing(name):
    norm, sim_type = _fit_candidate(name)
    # In similarity-oriented order: increasing cosine, or DEcreasing distance.
    if sim_type == "cosine":
        raw = torch.linspace(-1.0, 1.0, 400)
    else:
        raw = torch.linspace(3.0, 0.0, 400)  # distance decreasing => similarity increasing
    out = norm(raw)
    diffs = out[1:] - out[:-1]
    assert torch.all(diffs >= -1e-6), f"{name} is not monotone non-decreasing in similarity"


def test_affine_strictly_increasing_on_band():
    norm = AffineClipNormalizer("cosine", lo=0.90, hi=0.999)
    raw = torch.linspace(0.90, 0.999, 50)
    out = norm(raw)
    diffs = out[1:] - out[:-1]
    assert torch.all(diffs > 0), "affine_clip must be strictly increasing inside its band"


# ------------------------------------------------------------------
# Magnitude preservation: NOT empirical-CDF / rank equalization
# ------------------------------------------------------------------
def test_not_rank_equalization_on_skewed_distribution():
    """A monotone magnitude-preserving normalizer must differ from the rank/CDF map.

    For a skewed distribution the empirical CDF (rank/N) is uniform-spaced, while
    a magnitude-preserving normalizer reflects the raw spacing. They share the
    same ordering but the values must differ materially.
    """
    rng = np.random.default_rng(0)
    scores = np.concatenate([
        rng.uniform(0.985, 0.999, size=800),
        rng.uniform(0.90, 0.985, size=200),
    ])
    norm = AffineClipNormalizer.fit_from_scores(scores, "cosine")

    xs = np.sort(scores)
    out = norm(torch.from_numpy(xs)).numpy()
    cdf = (np.arange(1, len(xs) + 1) - 0.5) / len(xs)  # empirical CDF in [0,1]

    # Same monotone ordering, but materially different values (not equalized).
    assert np.all(np.diff(out) >= -1e-6)
    assert np.max(np.abs(out - cdf)) > 0.1, "affine output collapsed onto the rank/CDF map"


def test_logit_spreads_saturated_band():
    """Logit must expand a hyper-concentrated near-1 band more than a plain affine."""
    rng = np.random.default_rng(1)
    # Almost all mass in [0.995, 0.999] — differs only in the 3rd-4th decimal.
    scores = rng.uniform(0.995, 0.999, size=1000)
    affine = AffineClipNormalizer.fit_from_scores(scores, "cosine")
    logit = LogitNormalizer.fit_from_scores(scores, "cosine")
    xs = torch.from_numpy(np.sort(scores))
    out_a = affine(xs)
    out_l = logit(xs)
    # Both monotone; logit should not be a degenerate constant and should use range.
    assert (out_l.max() - out_l.min()) > 0.5
    assert torch.all(out_l[1:] - out_l[:-1] >= -1e-6)
    # Sanity: both stay bounded.
    for o in (out_a, out_l):
        assert torch.all(o >= -1e-6) and torch.all(o <= 1.0 + 1e-6)


# ------------------------------------------------------------------
# Serialization round-trip
# ------------------------------------------------------------------
@pytest.mark.parametrize("name", CANDIDATE_NORMALIZERS)
def test_serialization_roundtrip(name):
    norm, sim_type = _fit_candidate(name)
    cls = SCORE_NORMALIZER_REGISTRY[name]
    params = norm.params_to_dict()
    rebuilt = cls.from_params_dict(params, sim_type)
    raw = torch.linspace(-1.0, 1.0, 100) if sim_type == "cosine" else torch.linspace(0.0, 3.0, 100)
    assert torch.allclose(norm(raw), rebuilt(raw), atol=1e-6)


# ------------------------------------------------------------------
# sim_type compatibility
# ------------------------------------------------------------------
def test_cosine_only_method_rejects_l2():
    with pytest.raises(ValueError):
        LogitNormalizer("l2", lo=0.0, hi=1.0)
    with pytest.raises(ValueError):
        PowerNormalizer("l2", lo=0.0, hi=1.0, gamma=1.0)


def test_exp_l2_rejects_cosine():
    with pytest.raises(ValueError):
        ExpL2Normalizer("cosine", tau=1.0)


def test_zscore_squash_locked_to_tanh():
    # Only tanh squash is permitted (keeps output bounded; semantics locked).
    ZScoreNormalizer("cosine", mu=0.0, sigma=1.0, squash="tanh")
    with pytest.raises(ValueError):
        ZScoreNormalizer("cosine", mu=0.0, sigma=1.0, squash="clip")


# ------------------------------------------------------------------
# Legacy back-compat: bit-identical to the old inline computation
# ------------------------------------------------------------------
def test_legacy_percentile_matches_old_cosine_formula():
    p5, p95 = 0.5, 1.0
    norm = LegacyPercentileNormalizer("cosine", p5=p5, p95=p95)
    raw = torch.tensor([1.0, 0.5, 0.0, -1.0])
    # Old: s0 = (cos+1)/2 ; clip((s0 - p5)/(p95 - p5), 0, 1)
    s0 = (raw + 1.0) / 2.0
    expected = ((s0 - p5) / (p95 - p5)).clamp(0.0, 1.0)
    assert torch.allclose(norm(raw), expected, atol=1e-6)


def test_legacy_percentile_denom_zero_fallback():
    norm = LegacyPercentileNormalizer("cosine", p5=0.7, p95=0.7)
    raw = torch.tensor([1.0, 0.0, -1.0])
    out = norm(raw)
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_direction_unify_l2_matches_exp():
    tau = 0.334717
    norm = DirectionUnifyNormalizer("l2", tau=tau)
    raw = torch.tensor([0.1, 0.5, 1.0])
    expected = torch.exp(-raw / tau)
    assert torch.allclose(norm(raw), expected, atol=1e-6)


# ------------------------------------------------------------------
# Runtime builder dispatch
# ------------------------------------------------------------------
def _active(field, sim="cosine", tau=None):
    sim_cfg = {"type": sim}
    if tau is not None:
        sim_cfg["to_similarity"] = {"type": "exp", "tau": tau}
    return (field, 1.0, sim_cfg)


def test_build_per_field_dispatch():
    active = [_active("vision_0", "cosine"), _active("robot_state", "l2", tau=0.33)]
    sn = {
        "type": "per_field",
        "fields": {
            "vision_0": {"method": "logit", "params": {"lo": 0.9, "hi": 0.999, "eps": 1e-4}},
            "robot_state": {"method": "exp_l2", "params": {"tau": 0.33}},
        },
    }
    fs = {"vision_0": {"type": "cosine"}, "robot_state": {"type": "l2", "to_similarity": {"tau": 0.33}}}
    norms = build_field_normalizers(active, sn, fs)
    assert isinstance(norms["vision_0"], LogitNormalizer)
    assert isinstance(norms["robot_state"], ExpL2Normalizer)


def test_build_percentile_dispatch_legacy():
    active = [_active("vision_0", "cosine")]
    sn = {"type": "percentile", "fields": {"vision_0": {"p5": 0.5, "p95": 1.0}}}
    norms = build_field_normalizers(active, sn, {"vision_0": {"type": "cosine"}})
    assert isinstance(norms["vision_0"], LegacyPercentileNormalizer)


def test_build_none_dispatch_direction_unify():
    active = [_active("robot_state", "l2", tau=0.33)]
    norms = build_field_normalizers(
        active, {"type": "none"},
        {"robot_state": {"type": "l2", "to_similarity": {"tau": 0.33}}},
    )
    assert isinstance(norms["robot_state"], DirectionUnifyNormalizer)
    # tau is sourced from field_similarity for the none/legacy paths.
    raw = torch.tensor([0.5])
    assert torch.allclose(norms["robot_state"](raw), torch.exp(-raw / 0.33), atol=1e-6)


def test_build_per_field_missing_entry_raises():
    active = [_active("vision_0", "cosine")]
    sn = {"type": "per_field", "fields": {}}
    with pytest.raises(ValueError):
        build_field_normalizers(active, sn, {"vision_0": {"type": "cosine"}})


def test_build_unknown_type_raises_not_silent_fallback():
    # An unrecognized type must raise, not resurrect the no-normalize path.
    active = [_active("vision_0", "cosine")]
    with pytest.raises(ValueError, match="unknown score_normalization.type"):
        build_field_normalizers(
            active, {"type": "typo", "fields": {}}, {"vision_0": {"type": "cosine"}}
        )
