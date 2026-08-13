"""Retrieval scoring at production parity, plus the E1 feature variants.

The offline experiments only mean something if their ranking is the ranking the
server would produce. This module therefore builds its scorer *from a real eval
yaml* and reuses the production normalizers, rather than re-deriving constants:

  Layer 1  per-field raw similarity (cosine, or L2 distance)
  Layer 2  per-field ScoreNormalizer (z-score + tanh) from the yaml
  Layer 3  weighted sum over fields, restricted to the query's ``task_key``

The ``task_key`` restriction is not optional: production applies
``QueryFilter(task_key=...)`` even when ``step_filter="all"``, so scoring
cross-task candidates offline would inflate the baseline residual.

Public interface: :class:`Scorer`, :func:`build_scorer`, :func:`diff_features`,
:func:`fit_diff_normalizer`, :class:`DiffNormalizer`.

Key dependencies: ``openpi.cache.config.load_cache_config`` (yaml -> config),
``openpi.cache.components.score_normalizers.build_field_normalizers``
(identical Layer-1 objects as the server).
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

# ------------------------------------------------------------------
# Single-frame scorer
# ------------------------------------------------------------------


@dataclasses.dataclass
class Scorer:
    """Production-parity scorer for one query frame against candidate entries.

    ``active_fields`` mirrors the backend's own list of ``(field, weight,
    sim_cfg)`` triples, and ``normalizers`` are the very objects the backend
    would build from the same yaml.
    """

    active_fields: list[tuple[str, float, dict[str, Any]]]
    normalizers: Mapping[str, Any]
    yaml_path: str
    trajectory_depth: int
    trajectory_weights: Optional[list[float]]

    # -- Layer 1 -----------------------------------------------------
    def _raw(self, q: np.ndarray, e: np.ndarray, sim_cfg: dict[str, Any]) -> float:
        sim_type = sim_cfg.get("type", "cosine")
        qt = torch.as_tensor(np.asarray(q), dtype=torch.float32)
        et = torch.as_tensor(np.asarray(e), dtype=torch.float32)
        if sim_type == "cosine":
            return float(torch.nn.functional.cosine_similarity(qt, et, dim=0))
        if sim_type == "l2":
            return float(torch.linalg.vector_norm(qt - et))
        raise ValueError(f"Unknown similarity type: {sim_type!r}")

    # -- Layers 2 + 3 ------------------------------------------------
    def score(self, query_keys: Mapping[str, np.ndarray], entry: Any) -> float:
        """Weighted sum of normalized per-field similarities for one entry."""
        total = 0.0
        for field, weight, sim_cfg in self.active_fields:
            if field not in query_keys or field not in entry.query_keys:
                continue  # production masks absent fields out of the sum
            raw = self._raw(query_keys[field], entry.query_keys[field], sim_cfg)
            s = float(self.normalizers[field](torch.tensor([raw], dtype=torch.float32))[0])
            total += weight * s
        return total

    def score_batch(
        self,
        query_keys: Mapping[str, np.ndarray],
        entries: Sequence[Any],
        cache: Optional[dict] = None,
    ) -> np.ndarray:
        """Vectorised :meth:`score` over many candidates.

        LOEO touches ~10^5 (query, candidate) pairs per key builder, so the
        per-entry Python path is too slow; stacking candidates per field and
        letting torch do one matrix op is what makes E1 tractable. ``cache``
        optionally memoises the stacked candidate matrices across queries
        (keyed by field and by the candidate list's identity).
        """
        n = len(entries)
        totals = torch.zeros(n, dtype=torch.float32)
        for field, weight, sim_cfg in self.active_fields:
            if field not in query_keys:
                continue
            key = (field, id(entries))
            mat = None if cache is None else cache.get(key)
            if mat is None:
                rows, mask = [], []
                dim = len(np.asarray(query_keys[field]))
                for e in entries:
                    vec = e.query_keys.get(field)
                    rows.append(np.asarray(vec, dtype=np.float32) if vec is not None else np.zeros(dim, dtype=np.float32))
                    mask.append(vec is not None)
                mat = (torch.as_tensor(np.stack(rows)), torch.tensor(mask, dtype=torch.float32))
                if cache is not None:
                    cache[key] = mat
            cand, mask_t = mat
            q = torch.as_tensor(np.asarray(query_keys[field], dtype=np.float32))
            sim_type = sim_cfg.get("type", "cosine")
            if sim_type == "cosine":
                raw = torch.nn.functional.cosine_similarity(q.unsqueeze(0), cand, dim=1)
            elif sim_type == "l2":
                raw = torch.linalg.vector_norm(cand - q.unsqueeze(0), dim=1)
            else:
                raise ValueError(f"Unknown similarity type: {sim_type!r}")
            totals += weight * self.normalizers[field](raw) * mask_t
        return totals.numpy()

    def score_trajectory(
        self,
        query_history: Sequence[Mapping[str, np.ndarray]],
        ancestor_entries: Sequence[Optional[Any]],
        weights: Sequence[float],
    ) -> float:
        """``sum_l w_l * Score(anc_l, q_{t-l})`` with missing ancestors at 0.0.

        Matches ``in_memory_backend``'s accumulation, where an absent ancestor
        contributes ``level_scores[l].get(id, 0.0)`` rather than being skipped
        (skipping would silently renormalise the weights).
        """
        if not (len(query_history) == len(ancestor_entries) == len(weights)):
            raise ValueError("query_history, ancestor_entries and weights must align")
        total = 0.0
        for q, entry, w in zip(query_history, ancestor_entries, weights):
            if entry is None or q is None:
                continue
            total += w * self.score(q, entry)
        return total

    def candidates(self, entries: Iterable[Any], task_key: str) -> list[Any]:
        """Restrict candidates to the query's task, as production does."""
        return [e for e in entries if e.payload.task_key == task_key]


def build_scorer(yaml_path: str | pathlib.Path) -> Scorer:
    """Build a :class:`Scorer` from a real eval yaml (no hand-written constants)."""
    from openpi.cache.components.score_normalizers import build_field_normalizers
    from openpi.cache.config import load_cache_config

    yaml_path = pathlib.Path(yaml_path)
    cfg = load_cache_config(yaml_path)
    cp1 = cfg.checkpoints["cp1"]
    ss = cp1.search_strategy

    field_sim_raw = ss.field_similarity or {}
    # build_field_normalizers takes plain dicts (the shape the backend feeds it),
    # while the config layer hands us dataclasses -- convert once, here.
    field_sim = {
        name: (cfg_i if isinstance(cfg_i, dict) else dataclasses.asdict(cfg_i))
        for name, cfg_i in field_sim_raw.items()
    }
    active: list[tuple[str, float, dict[str, Any]]] = []
    for name, field_cfg in [
        ("vision_0", cfg.keys.vision_0),
        ("vision_1", cfg.keys.vision_1),
        ("vision_2", cfg.keys.vision_2),
        ("prompt_emb", cfg.keys.prompt_emb),
        ("robot_state", cfg.keys.robot_state),
    ]:
        if not field_cfg.enabled or field_cfg.weight <= 0:
            continue
        active.append((name, float(field_cfg.weight), field_sim.get(name, {"type": "cosine"})))

    sn = ss.score_normalization
    sn_dict = None if sn is None else (sn if isinstance(sn, dict) else dataclasses.asdict(sn))
    normalizers = build_field_normalizers(active, sn_dict, field_sim)
    return Scorer(
        active_fields=active,
        normalizers=normalizers,
        yaml_path=str(yaml_path),
        trajectory_depth=int(ss.trajectory_depth),
        trajectory_weights=list(ss.trajectory_weights) if ss.trajectory_weights else None,
    )


# ------------------------------------------------------------------
# E1 group C: difference features
# ------------------------------------------------------------------


def diff_features(
    frames: Sequence[Optional[Mapping[str, np.ndarray]]],
    field: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return ``(delta, delta2, is_padding)`` for one field at the newest frame.

    ``frames`` is newest-first: ``[k_t, k_{t-1}, k_{t-2}]``. At an episode
    prefix the missing frames yield zero vectors and ``is_padding=True``; those
    steps are excluded from the primary analysis and reported separately.
    """
    if len(frames) < 3:
        raise ValueError("second-order differences need three frames (newest first)")
    k_t, k_1, k_2 = frames[0], frames[1], frames[2]
    if k_t is None or field not in k_t:
        raise ValueError(f"current frame is missing field {field!r}")
    cur = np.asarray(k_t[field], dtype=np.float32)
    padding = k_1 is None or k_2 is None
    prev = np.asarray(k_1[field], dtype=np.float32) if k_1 is not None else cur
    prev2 = np.asarray(k_2[field], dtype=np.float32) if k_2 is not None else prev
    delta = cur - prev
    delta2 = delta - (prev - prev2)
    if padding:
        delta = np.zeros_like(cur)
        delta2 = np.zeros_like(cur)
    return delta, delta2, padding


def diff_similarity_per_field(
    scorer: "Scorer",
    query_frames: Sequence[Optional[Mapping[str, np.ndarray]]],
    cand_frames: Sequence[Optional[Mapping[str, np.ndarray]]],
) -> dict[str, tuple[float, float]]:
    """Per-modality (delta, delta^2) cosine similarities between two frame stacks.

    The plan's group C weights the difference blocks as ``w_f (x) [1, gamma,
    gamma]`` across **every** active field; scoring a single field would test a
    different feature set than the one that was registered.
    """
    out: dict[str, tuple[float, float]] = {}
    for field, _weight, _sim in scorer.active_fields:
        try:
            dq, d2q, _ = diff_features(query_frames, field)
            dc, d2c, _ = diff_features(cand_frames, field)
        except ValueError:
            continue
        out[field] = (_cosine(dq, dc), _cosine(d2q, d2c))
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return 0.0 if na <= 0.0 or nb <= 0.0 else float(np.dot(a, b) / (na * nb))


@dataclasses.dataclass(frozen=True)
class DiffNormalizer:
    """z-score + tanh normalizer fitted on difference-block similarities.

    Difference similarities live on a different scale than raw-key
    similarities, so reusing the yaml's mu/sigma would conflate "does C help"
    with "is C mis-scaled".
    """

    mu: float
    sigma: float
    n_fit: int
    fit_trajectories: tuple[str, ...]

    def __call__(self, raw: float) -> float:
        sigma = self.sigma if self.sigma > 1e-12 else 1.0
        return 0.5 * (float(np.tanh((raw - self.mu) / sigma)) + 1.0)


def fit_diff_normalizer(
    raw_similarities: Sequence[float],
    fit_trajectories: Sequence[str],
    held_out_trajectory: str,
    *,
    source_trajectories: Optional[Sequence[str]] = None,
) -> DiffNormalizer:
    """Fit the difference-block normalizer on library-side data only.

    Two distinct leaks have to be blocked, and checking only the first is what
    made an earlier version unsafe:

    * the held-out episode must not be among the *fitting* episodes; and
    * every fitted **value** must have been computed from library-side pairs.
      Similarities between the held-out query and the candidates are derived
      from held-out data, so calibrating on them leaks the fold even though no
      held-out trajectory name appears in the fit list.

    ``source_trajectories`` names the episode each value came from and is
    checked element-wise; callers building the sample from library pairs pass
    it, and omitting it is only valid for synthetic data in tests.
    """
    fit = tuple(fit_trajectories)
    if held_out_trajectory in fit:
        raise ValueError(
            f"fold leakage: held-out trajectory {held_out_trajectory!r} is in the fit set"
        )
    if source_trajectories is not None:
        if len(source_trajectories) != len(raw_similarities):
            raise ValueError("source_trajectories must align with raw_similarities")
        if any(t == held_out_trajectory for t in source_trajectories):
            raise ValueError(
                f"fold leakage: a calibration value was computed from held-out "
                f"trajectory {held_out_trajectory!r}"
            )
    values = np.asarray(raw_similarities, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot fit a normalizer on an empty sample")
    return DiffNormalizer(
        mu=float(np.mean(values)),
        sigma=float(np.std(values)),
        n_fit=int(values.size),
        fit_trajectories=fit,
    )
