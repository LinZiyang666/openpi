"""Layer-1 score normalization for the weighted_score_sum search path.

This module is the calibration / normalization layer (Layer 1) of the two-layer
weighted-sum search. It maps a single modality's bottom-level geometric
similarity (a cosine value or an L2 distance) to a normalized, bounded scalar
that is comparable across modalities, so the search layer (Layer 2) can take a
magnitude-faithful weighted sum.

Design contract
---------------
- Each normalizer is **monotone non-decreasing in similarity** and strictly
  increasing on its unsaturated interval; clipping / saturation segments may be
  flat by design (the bounded output is intentional).
- **Candidate** normalizers (those an offline calibration may select) all emit
  values in ``[0, 1]`` so that neither the Layer-2 weighted sum nor the
  calibration ranking metric is dominated by output scale rather than
  discriminative power.
- **No empirical-CDF / rank equalization.** That would discard magnitude and
  collapse the weighted sum into rank fusion (effectively weighted RRF), which
  defeats the purpose of weighted_score_sum.

Direction convention
--------------------
``raw`` passed to a normalizer is the field score as produced by
``InMemoryBackend._batch_field_scores``: a cosine value in ``[-1, 1]`` when
``sim_type == "cosine"`` and an L2 distance in ``[0, inf)`` when
``sim_type == "l2"``. Each normalizer is constructed with its ``sim_type`` and
orients internally so that a higher output always means more similar.

Public interface
----------------
- ``ScoreNormalizer`` — abstract base (callable on a ``torch.Tensor``).
- Concrete normalizers + ``SCORE_NORMALIZER_REGISTRY`` / ``CANDIDATE_NORMALIZERS``.
- ``build_field_normalizers`` — runtime factory consumed by the backend.

Coupling map
------------
  DEPENDS ON:  torch, numpy only (leaf module — no cache.* imports, no cycle).
  CONSUMED BY: in_memory_backend.py (runtime apply), config.py (validation),
               exp/common/calibrate_score_normalizers.py (offline fit).
"""

from __future__ import annotations

import abc
import math
from typing import Any, ClassVar, Optional

import numpy as np
import torch

VALID_SIM_TYPES = frozenset({"cosine", "l2"})


def _orient(raw: torch.Tensor, sim_type: str) -> torch.Tensor:
    """Return a tensor where a higher value means more similar.

    cosine: identity (a larger cosine already means more similar).
    l2:     negate the distance (a smaller distance means more similar).
    """
    if sim_type == "cosine":
        return raw
    if sim_type == "l2":
        return -raw
    raise ValueError(f"Unknown sim_type: {sim_type!r}")


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------
class ScoreNormalizer(abc.ABC):
    """Monotone, bounded map from a field's raw similarity to a comparable scalar.

    Subclasses declare ``name`` and ``supported_sim_types``. Instances hold the
    fitted parameters and are callable on a ``[N]`` float tensor of raw field
    scores, returning a ``[N]`` float tensor of normalized scores.
    """

    name: ClassVar[str]
    supported_sim_types: ClassVar[frozenset]

    def __init__(self, sim_type: str) -> None:
        if sim_type not in self.supported_sim_types:
            raise ValueError(
                f"normalizer {self.name!r} does not support sim_type={sim_type!r}; "
                f"supported: {sorted(self.supported_sim_types)}"
            )
        self.sim_type = sim_type

    @abc.abstractmethod
    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply the normalization to a ``[N]`` tensor of raw field scores."""

    @abc.abstractmethod
    def params_to_dict(self) -> dict[str, Any]:
        """Serialize fitted params (numeric only; ``sim_type`` is supplied at build)."""

    @classmethod
    @abc.abstractmethod
    def from_params_dict(cls, params: dict[str, Any], sim_type: str) -> "ScoreNormalizer":
        """Reconstruct from a serialized params dict + the field's sim_type."""

    @classmethod
    @abc.abstractmethod
    def fit_from_scores(
        cls,
        scores: np.ndarray,
        sim_type: str,
        *,
        robust_pct: tuple[float, float] = (1.0, 99.0),
    ) -> "ScoreNormalizer":
        """Fit params from a 1-D array of raw field scores (offline calibration)."""


# ------------------------------------------------------------------
# Candidate normalizers (selectable by offline calibration; all emit [0, 1])
# ------------------------------------------------------------------
class AffineClipNormalizer(ScoreNormalizer):
    """Linear rescale of the oriented score from ``[lo, hi]`` onto ``[0, 1]``.

    Magnitude- and spacing-preserving within the band. Generalizes the old
    percentile clip but operates in the oriented raw-score space.
    """

    name: ClassVar[str] = "affine_clip"
    supported_sim_types: ClassVar[frozenset] = frozenset({"cosine", "l2"})

    def __init__(self, sim_type: str, lo: float, hi: float) -> None:
        super().__init__(sim_type)
        self.lo = float(lo)
        self.hi = float(hi)

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        x = _orient(raw.float(), self.sim_type)
        denom = self.hi - self.lo
        if denom <= 0:
            return torch.full_like(x, 0.5)
        return ((x - self.lo) / denom).clamp(0.0, 1.0)

    def params_to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi}

    @classmethod
    def from_params_dict(cls, params: dict[str, Any], sim_type: str) -> "AffineClipNormalizer":
        return cls(sim_type, params["lo"], params["hi"])

    @classmethod
    def fit_from_scores(cls, scores, sim_type, *, robust_pct=(1.0, 99.0)):
        x = np.asarray(scores, dtype=np.float64)
        x = x if sim_type == "cosine" else -x
        lo = float(np.percentile(x, robust_pct[0]))
        hi = float(np.percentile(x, robust_pct[1]))
        return cls(sim_type, lo, hi)


class ZScoreNormalizer(ScoreNormalizer):
    """Standardize the oriented score then squash to ``[0, 1]``.

    The squash is mandatory and bounded so the output never escapes ``[0, 1]``;
    only ``"tanh"`` (``0.5*(tanh(z)+1)``) is supported to keep semantics locked.
    """

    name: ClassVar[str] = "zscore"
    supported_sim_types: ClassVar[frozenset] = frozenset({"cosine", "l2"})

    def __init__(self, sim_type: str, mu: float, sigma: float, squash: str = "tanh") -> None:
        super().__init__(sim_type)
        if squash != "tanh":
            raise ValueError(f"ZScoreNormalizer only supports squash='tanh', got {squash!r}")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.squash = squash

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        x = _orient(raw.float(), self.sim_type)
        sigma = self.sigma if self.sigma > 1e-12 else 1.0
        z = (x - self.mu) / sigma
        return 0.5 * (torch.tanh(z) + 1.0)

    def params_to_dict(self) -> dict[str, Any]:
        return {"mu": self.mu, "sigma": self.sigma, "squash": self.squash}

    @classmethod
    def from_params_dict(cls, params, sim_type):
        return cls(sim_type, params["mu"], params["sigma"], params.get("squash", "tanh"))

    @classmethod
    def fit_from_scores(cls, scores, sim_type, *, robust_pct=(1.0, 99.0)):
        x = np.asarray(scores, dtype=np.float64)
        x = x if sim_type == "cosine" else -x
        return cls(sim_type, float(np.mean(x)), float(np.std(x)))


class LogitNormalizer(ScoreNormalizer):
    """Cosine-only: logit of the band-normalized cosine, rescaled to ``[0, 1]``.

    Allocates output resolution to the dense near-1 region while staying
    monotone (a fixed analytic transform, not a distribution-equalizing one).
    """

    name: ClassVar[str] = "logit"
    supported_sim_types: ClassVar[frozenset] = frozenset({"cosine"})

    def __init__(self, sim_type: str, lo: float, hi: float, eps: float = 1e-4) -> None:
        super().__init__(sim_type)
        self.lo = float(lo)
        self.hi = float(hi)
        self.eps = float(eps)

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        x = raw.float()
        denom = self.hi - self.lo
        if denom <= 0:
            return torch.full_like(x, 0.5)
        u = ((x - self.lo) / denom).clamp(self.eps, 1.0 - self.eps)
        t = torch.log(u / (1.0 - u))
        t_lo = math.log(self.eps / (1.0 - self.eps))
        t_hi = math.log((1.0 - self.eps) / self.eps)
        return ((t - t_lo) / (t_hi - t_lo)).clamp(0.0, 1.0)

    def params_to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "eps": self.eps}

    @classmethod
    def from_params_dict(cls, params, sim_type):
        return cls(sim_type, params["lo"], params["hi"], params.get("eps", 1e-4))

    @classmethod
    def fit_from_scores(cls, scores, sim_type, *, robust_pct=(1.0, 99.0)):
        x = np.asarray(scores, dtype=np.float64)
        lo = float(np.percentile(x, robust_pct[0]))
        hi = float(np.percentile(x, robust_pct[1]))
        return cls(sim_type, lo, hi)


class NegLogOneMinusNormalizer(ScoreNormalizer):
    """Cosine-only: ``-log(1 - cos + eps)`` rescaled to ``[0, 1]``.

    Expands the near-1 saturated region (where ``1 - cos`` is tiny) while
    remaining monotone increasing in cosine.
    """

    name: ClassVar[str] = "neg_log_one_minus"
    supported_sim_types: ClassVar[frozenset] = frozenset({"cosine"})

    def __init__(self, sim_type: str, v_lo: float, v_hi: float, eps: float = 1e-4) -> None:
        super().__init__(sim_type)
        self.v_lo = float(v_lo)
        self.v_hi = float(v_hi)
        self.eps = float(eps)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.log((1.0 - x + self.eps).clamp_min(self.eps))

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        v = self._transform(raw.float())
        denom = self.v_hi - self.v_lo
        if denom <= 0:
            return torch.full_like(v, 0.5)
        return ((v - self.v_lo) / denom).clamp(0.0, 1.0)

    def params_to_dict(self) -> dict[str, Any]:
        return {"v_lo": self.v_lo, "v_hi": self.v_hi, "eps": self.eps}

    @classmethod
    def from_params_dict(cls, params, sim_type):
        return cls(sim_type, params["v_lo"], params["v_hi"], params.get("eps", 1e-4))

    @classmethod
    def fit_from_scores(cls, scores, sim_type, *, robust_pct=(1.0, 99.0)):
        x = np.asarray(scores, dtype=np.float64)
        eps = 1e-4
        v = -np.log(np.clip(1.0 - x + eps, eps, None))
        v_lo = float(np.percentile(v, robust_pct[0]))
        v_hi = float(np.percentile(v, robust_pct[1]))
        return cls(sim_type, v_lo, v_hi, eps)


class PowerNormalizer(ScoreNormalizer):
    """Cosine-only: band-normalize cosine to ``[0, 1]`` then raise to ``gamma``.

    ``gamma`` is fit so the distribution median maps to ~0.5, redistributing
    resolution toward the dense region while staying monotone. The median->0.5
    intent holds for the default robust band; if a caller's band excludes the
    median, ``u_med`` clamping keeps ``gamma`` finite but the mapping is no
    longer median-centered.
    """

    name: ClassVar[str] = "power"
    supported_sim_types: ClassVar[frozenset] = frozenset({"cosine"})

    def __init__(self, sim_type: str, lo: float, hi: float, gamma: float) -> None:
        super().__init__(sim_type)
        self.lo = float(lo)
        self.hi = float(hi)
        self.gamma = float(gamma)

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        x = raw.float()
        denom = self.hi - self.lo
        if denom <= 0:
            return torch.full_like(x, 0.5)
        u = ((x - self.lo) / denom).clamp(0.0, 1.0)
        return u.pow(self.gamma)

    def params_to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "gamma": self.gamma}

    @classmethod
    def from_params_dict(cls, params, sim_type):
        return cls(sim_type, params["lo"], params["hi"], params["gamma"])

    @classmethod
    def fit_from_scores(cls, scores, sim_type, *, robust_pct=(1.0, 99.0)):
        x = np.asarray(scores, dtype=np.float64)
        lo = float(np.percentile(x, robust_pct[0]))
        hi = float(np.percentile(x, robust_pct[1]))
        denom = hi - lo
        if denom <= 0:
            return cls(sim_type, lo, hi, 1.0)
        u_med = float(np.clip((np.median(x) - lo) / denom, 1e-3, 1.0 - 1e-3))
        gamma = math.log(0.5) / math.log(u_med)
        gamma = float(np.clip(gamma, 0.1, 10.0))
        return cls(sim_type, lo, hi, gamma)


class ExpL2Normalizer(ScoreNormalizer):
    """L2-only: ``exp(-d / tau)`` mapping a distance to a similarity in ``(0, 1]``."""

    name: ClassVar[str] = "exp_l2"
    supported_sim_types: ClassVar[frozenset] = frozenset({"l2"})

    def __init__(self, sim_type: str, tau: float) -> None:
        super().__init__(sim_type)
        self.tau = float(tau) if float(tau) > 1e-12 else 1.0

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.exp(-raw.float() / self.tau)

    def params_to_dict(self) -> dict[str, Any]:
        return {"tau": self.tau}

    @classmethod
    def from_params_dict(cls, params, sim_type):
        return cls(sim_type, params["tau"])

    @classmethod
    def fit_from_scores(cls, scores, sim_type, *, robust_pct=(1.0, 99.0)):
        d = np.asarray(scores, dtype=np.float64)
        median = float(np.median(d))
        tau = median / math.log(2.0) if median > 0 else 1.0
        return cls(sim_type, tau)


# ------------------------------------------------------------------
# Backward-compatibility normalizers (NOT selectable by calibration)
# ------------------------------------------------------------------
class LegacyPercentileNormalizer(ScoreNormalizer):
    """Faithful reproduction of the old ``percentile`` score normalization.

    Maps the raw score to ``[0, 1]`` first (cosine: ``(cos+1)/2``; l2:
    ``exp(-d/tau)``), then clips ``(s - p5) / (p95 - p5)`` to ``[0, 1]`` with the
    ``denom <= 0 -> 0.5`` fallback. Used only to load legacy YAML / calibration
    artifacts; it is not a Phase-1 candidate.
    """

    name: ClassVar[str] = "legacy_percentile"
    supported_sim_types: ClassVar[frozenset] = frozenset({"cosine", "l2"})

    def __init__(self, sim_type: str, p5: float, p95: float, tau: float = 1.0) -> None:
        super().__init__(sim_type)
        self.p5 = float(p5)
        self.p95 = float(p95)
        self.tau = float(tau) if float(tau) > 1e-12 else 1.0

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        x = raw.float()
        if self.sim_type == "cosine":
            s0 = (x + 1.0) / 2.0
        else:
            s0 = torch.exp(-x / self.tau)
        denom = self.p95 - self.p5
        if denom <= 0:
            return torch.full_like(s0, 0.5)
        return ((s0 - self.p5) / denom).clamp(0.0, 1.0)

    def params_to_dict(self) -> dict[str, Any]:
        return {"p5": self.p5, "p95": self.p95}

    @classmethod
    def from_params_dict(cls, params, sim_type):
        return cls(sim_type, params["p5"], params["p95"], params.get("tau", 1.0))

    @classmethod
    def fit_from_scores(cls, scores, sim_type, *, robust_pct=(1.0, 99.0)):
        # Legacy fit replicates the old script: percentiles on the mapped score.
        x = np.asarray(scores, dtype=np.float64)
        s0 = (x + 1.0) / 2.0 if sim_type == "cosine" else np.exp(-x)
        return cls(sim_type, float(np.percentile(s0, 5)), float(np.percentile(s0, 95)))


class DirectionUnifyNormalizer(ScoreNormalizer):
    """Direction-unify only (the old ``score_normalization.type == "none"`` path).

    cosine -> ``(cos+1)/2``; l2 -> ``exp(-d/tau)``. No band clipping. Backend /
    spec-level back-compat only; not a valid production YAML and not a Phase-1
    candidate (config validation rejects ``type:none`` for weighted_score_sum).
    """

    name: ClassVar[str] = "direction_unify"
    supported_sim_types: ClassVar[frozenset] = frozenset({"cosine", "l2"})

    def __init__(self, sim_type: str, tau: float = 1.0) -> None:
        super().__init__(sim_type)
        self.tau = float(tau) if float(tau) > 1e-12 else 1.0

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        x = raw.float()
        if self.sim_type == "cosine":
            return (x + 1.0) / 2.0
        return torch.exp(-x / self.tau)

    def params_to_dict(self) -> dict[str, Any]:
        return {"tau": self.tau}

    @classmethod
    def from_params_dict(cls, params, sim_type):
        return cls(sim_type, params.get("tau", 1.0))

    @classmethod
    def fit_from_scores(cls, scores, sim_type, *, robust_pct=(1.0, 99.0)):
        return cls(sim_type, 1.0)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
SCORE_NORMALIZER_REGISTRY: dict[str, type[ScoreNormalizer]] = {
    c.name: c
    for c in (
        AffineClipNormalizer,
        ZScoreNormalizer,
        LogitNormalizer,
        NegLogOneMinusNormalizer,
        PowerNormalizer,
        ExpL2Normalizer,
        LegacyPercentileNormalizer,
        DirectionUnifyNormalizer,
    )
}

# Methods an offline calibration may select among (compat normalizers excluded).
CANDIDATE_NORMALIZERS: tuple[str, ...] = (
    "affine_clip",
    "zscore",
    "logit",
    "neg_log_one_minus",
    "power",
    "exp_l2",
)


def candidate_normalizers_for(sim_type: str) -> tuple[str, ...]:
    """Candidate method names compatible with a field's ``sim_type``."""
    return tuple(
        name
        for name in CANDIDATE_NORMALIZERS
        if sim_type in SCORE_NORMALIZER_REGISTRY[name].supported_sim_types
    )


# ------------------------------------------------------------------
# Runtime factory
# ------------------------------------------------------------------
def _field_sim_type_tau(
    field_name: str,
    sim_cfg: Optional[dict[str, Any]],
    field_similarity: Optional[dict[str, Any]],
) -> tuple[str, float]:
    """Resolve a field's sim_type + tau from the per-field similarity config.

    ``sim_cfg`` (carried in active_fields) and ``field_similarity[field]`` are
    the same structure; the latter is the authoritative source when present.
    """
    fs = None
    if field_similarity is not None:
        fs = field_similarity.get(field_name)
    if fs is None:
        fs = sim_cfg or {"type": "cosine"}
    sim_type = fs.get("type", "cosine")
    tau = float((fs.get("to_similarity") or {}).get("tau", 1.0))
    return sim_type, tau


def build_field_normalizers(
    active_fields: list[tuple[str, float, dict[str, Any]]],
    score_normalization: Optional[dict[str, Any]],
    field_similarity: Optional[dict[str, Any]],
) -> dict[str, ScoreNormalizer]:
    """Construct one ``ScoreNormalizer`` per active field.

    - ``type == "per_field"``: build from ``fields[field] = {method, params}``;
      the normalizer owns all of its parameters (including any tau).
    - ``type == "percentile"``: build ``LegacyPercentileNormalizer`` from
      ``fields[field] = {p5, p95}`` with tau taken from ``field_similarity``.
    - ``type == "none"`` / missing: build ``DirectionUnifyNormalizer`` (backend /
      spec-level back-compat only).
    """
    sn = score_normalization or {}
    sn_type = sn.get("type", "none")
    fields_cfg = sn.get("fields") or {}

    out: dict[str, ScoreNormalizer] = {}
    for field_name, _weight, sim_cfg in active_fields:
        sim_type, tau = _field_sim_type_tau(field_name, sim_cfg, field_similarity)

        if sn_type == "per_field":
            entry = fields_cfg.get(field_name)
            if entry is None:
                raise ValueError(
                    f"per_field score_normalization missing entry for field {field_name!r}"
                )
            method = entry.get("method")
            cls = SCORE_NORMALIZER_REGISTRY.get(method)
            if cls is None:
                raise ValueError(
                    f"unknown score normalizer method {method!r} for field "
                    f"{field_name!r}; valid: {sorted(SCORE_NORMALIZER_REGISTRY)}"
                )
            out[field_name] = cls.from_params_dict(entry.get("params", {}), sim_type)
        elif sn_type == "percentile":
            entry = fields_cfg.get(field_name)
            if entry is None:
                raise ValueError(
                    f"percentile score_normalization missing entry for field {field_name!r}"
                )
            out[field_name] = LegacyPercentileNormalizer(
                sim_type, entry["p5"], entry["p95"], tau
            )
        elif sn_type == "none":
            # Backend/spec-level back-compat only (config rejects type:none for
            # weighted_score_sum_knn); direction-unify, no normalization clip.
            out[field_name] = DirectionUnifyNormalizer(sim_type, tau)
        else:
            # Unknown type must NOT silently fall back to the old no-normalize
            # path; fail fast so a YAML typo cannot resurrect the failure mode.
            raise ValueError(
                f"unknown score_normalization.type {sn_type!r}; "
                f"valid: 'per_field' | 'percentile' | 'none'"
            )
    return out
