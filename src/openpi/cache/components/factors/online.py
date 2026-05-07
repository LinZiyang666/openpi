"""Layer 2 — 8 online factors (B2 of the verdict-factor refactor).

Each online factor:
  - Builds a unified splice ``[history[-P:], winner, walk_next(F)]`` of
    length ``P + 1 + F`` (per plan §6.6).
  - Calls Layer 1 (``ctx.normalization``) to z-score the splice.
  - Computes one of the four kinematic descriptors over the normalized
    splice via ``_descriptor_kernel.compute_descriptors``.
  - Supports multi-window via the ``windows: [{past: int, future: int}, ...]``
    construction param; each window emits one keyed scalar.

Naming convention (plan §6.11 #7): ``<descriptor>_online_<channel>``.
Key template: ``<factor_name>__p<P>_f<F>``.

The 8 thin subclasses share two abstract-data bases (action vs state) so
the splice-shape / chain-walk code lives in exactly one place per channel.

Coupling map:
  DEPENDS ON: components/factors/base.py (Factor, FactorContext),
              components/factors/_descriptor_kernel.py (compute_descriptors,
                                                       _DESCRIPTOR_ORIENTATIONS),
              components/factors/_debug.py (verdict logger),
              components/factors/registry.py (@register).
  CONSUMED BY: components/composite_judge.py (B5 Layer 2 instances).
"""

from __future__ import annotations

import torch

from openpi.cache.components.factors._debug import VERDICT_DEBUG, verdict_logger
from openpi.cache.components.factors._descriptor_kernel import (
    _DESCRIPTOR_ORIENTATIONS,
    all_nan_for,
    compute_descriptors,
)
from openpi.cache.components.factors.base import FactorContext, normalize_windows
from openpi.cache.components.factors.registry import register

# Local underscore alias — used by tests + offline.py.
_normalize_windows = normalize_windows


# ----------------------------------------------------------------------
# Shared online base
# ----------------------------------------------------------------------


class _OnlineFactorBase:
    """Common machinery for the 8 online factors.

    Subclasses set:
      - descriptor: one of "jerk" / "direction" / "dispersion" / "path_length"
      - channel:    "action" or "state"

    Both subclasses share:
      - capability flags (``requires_chain_walk = True``,
        ``required_top_k = 0``)
      - ``describe(params)`` classmethod that derives keys from windows
      - ``extract(ctx)`` end-to-end (splice → normalize → kernel)
      - ``_build_splice`` overridden by ``_OnlineActionBase`` /
        ``_OnlineStateBase`` because the per-channel data sources differ.
    """

    # ---- Class-level capability flags ----
    descriptor: str = ""    # set per subclass
    channel:    str = ""    # set per subclass — "action" or "state"
    requires_chain_walk: bool = True
    required_top_k: int = 0

    def __init__(self, *, windows) -> None:
        self._windows = _normalize_windows(windows)
        if not self._windows:
            raise ValueError(f"{type(self).__name__}: at least one window required")
        self.descriptor_orientations: dict[str, str] = self.__class__.describe(
            {"windows": self._windows}
        )

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        if not cls.descriptor or not cls.channel:
            raise TypeError(
                f"{cls.__name__} must set `descriptor` and `channel` class attrs"
            )
        ori = _DESCRIPTOR_ORIENTATIONS[cls.descriptor]
        prefix = f"{cls.descriptor}_online_{cls.channel}"
        return {
            f"{prefix}__p{p}_f{f}": ori
            for (p, f) in _normalize_windows(params["windows"])
        }

    # ------------------------------------------------------------------
    # Per-verdict entrypoint
    # ------------------------------------------------------------------

    def extract(self, ctx: FactorContext) -> dict[str, float]:
        keys = list(self.descriptor_orientations.keys())
        if not ctx.results:
            return all_nan_for(keys)

        winner_id = ctx.results[0].id
        out: dict[str, float] = {}
        for (P, F) in self._windows:
            seq = self._build_splice(winner_id, ctx, P, F)
            if seq is None:
                # Boundary / fork / missing data — emit NaN per plan §6.5 rule 3.
                out[self._key(P, F)] = float("nan")
                continue
            normalized = self._normalize_seq(seq, ctx)            # [P+1+F, D_act]
            if normalized.shape[-1] == 0:
                out[self._key(P, F)] = float("nan")
                continue
            v = normalized[1:] - normalized[:-1]                  # [P+F,   D_act]
            j = normalized[2:] - 2 * normalized[1:-1] + normalized[:-2]  # [P+F-1, D_act]
            raw_dict = compute_descriptors([self.descriptor], normalized, v, j)
            out[self._key(P, F)] = raw_dict[self.descriptor]
        return out

    # ------------------------------------------------------------------
    # Hooks for action / state subclasses
    # ------------------------------------------------------------------

    def _build_splice(
        self,
        winner_id: str,
        ctx: FactorContext,
        P: int,
        F: int,
    ) -> torch.Tensor | None:
        raise NotImplementedError

    def _normalize_seq(self, seq: torch.Tensor, ctx: FactorContext) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, P: int, F: int) -> str:
        return f"{self.descriptor}_online_{self.channel}__p{P}_f{F}"


# ----------------------------------------------------------------------
# Action channel: history.actions + winner.action_chunk[0] + walk_next.action_chunk[0]
# ----------------------------------------------------------------------


class _OnlineActionBase(_OnlineFactorBase):
    channel: str = "action"

    def _build_splice(
        self,
        winner_id: str,
        ctx: FactorContext,
        P: int,
        F: int,
    ) -> torch.Tensor | None:
        history_actions = ctx.history.actions
        if len(history_actions) < P:
            if VERDICT_DEBUG:
                verdict_logger.info(
                    "[vd %s_online_action bail] history.actions=%d < P=%d",
                    self.descriptor, len(history_actions), P,
                )
            return None

        # Winner action_chunk[0] anchor
        winner_payload = ctx.view.get(winner_id)
        if winner_payload.action_chunk is None:
            return None
        anchor = torch.as_tensor(winner_payload.action_chunk[0], dtype=torch.float32)

        # Forward F entries' action_chunk[0]; walk_next may raise
        # NotImplementedError on a real fork (plan §6.7) — treat as boundary.
        try:
            forward_entries = ctx.view.walk_next(winner_id, k=F)
        except NotImplementedError:
            if VERDICT_DEBUG:
                verdict_logger.info(
                    "[vd %s_online_action bail] fork detected at winner=%s",
                    self.descriptor, winner_id,
                )
            return None
        if len(forward_entries) < F:
            if VERDICT_DEBUG:
                verdict_logger.info(
                    "[vd %s_online_action bail] walk_next=%d < F=%d (chain end)",
                    self.descriptor, len(forward_entries), F,
                )
            return None
        forward_actions: list[torch.Tensor] = []
        for e in forward_entries:
            ac = e.payload.action_chunk
            if ac is None:
                return None
            forward_actions.append(torch.as_tensor(ac[0], dtype=torch.float32))

        hist_tail = [torch.as_tensor(a, dtype=torch.float32) for a in history_actions[-P:]] if P > 0 else []
        return torch.stack(hist_tail + [anchor] + forward_actions, dim=0)   # [P+1+F, A]

    def _normalize_seq(self, seq: torch.Tensor, ctx: FactorContext) -> torch.Tensor:
        return ctx.normalization.normalize_action(seq)   # type: ignore[union-attr]


# ----------------------------------------------------------------------
# State channel: history.states + winner.robot_state + walk_next.robot_state
# ----------------------------------------------------------------------


class _OnlineStateBase(_OnlineFactorBase):
    channel: str = "state"

    def _build_splice(
        self,
        winner_id: str,
        ctx: FactorContext,
        P: int,
        F: int,
    ) -> torch.Tensor | None:
        history_states = ctx.history.states
        if len(history_states) < P:
            if VERDICT_DEBUG:
                verdict_logger.info(
                    "[vd %s_online_state bail] history.states=%d < P=%d",
                    self.descriptor, len(history_states), P,
                )
            return None

        winner_entry = ctx.view.get_entry(winner_id)
        winner_state = winner_entry.query_keys.get("robot_state")
        if winner_state is None:
            if VERDICT_DEBUG:
                verdict_logger.info(
                    "[vd %s_online_state bail] winner.robot_state missing winner=%s",
                    self.descriptor, winner_id,
                )
            return None

        try:
            forward_entries = ctx.view.walk_next(winner_id, k=F)
        except NotImplementedError:
            if VERDICT_DEBUG:
                verdict_logger.info(
                    "[vd %s_online_state bail] fork detected at winner=%s",
                    self.descriptor, winner_id,
                )
            return None
        if len(forward_entries) < F:
            if VERDICT_DEBUG:
                verdict_logger.info(
                    "[vd %s_online_state bail] walk_next=%d < F=%d (chain end)",
                    self.descriptor, len(forward_entries), F,
                )
            return None
        forward_states: list[torch.Tensor] = []
        for e in forward_entries:
            rs = e.query_keys.get("robot_state")
            if rs is None:
                return None
            forward_states.append(torch.as_tensor(rs, dtype=torch.float32))

        hist_tail = [torch.as_tensor(s, dtype=torch.float32) for s in history_states[-P:]] if P > 0 else []
        anchor = torch.as_tensor(winner_state, dtype=torch.float32)
        return torch.stack(hist_tail + [anchor] + forward_states, dim=0)   # [P+1+F, S]

    def _normalize_seq(self, seq: torch.Tensor, ctx: FactorContext) -> torch.Tensor:
        return ctx.normalization.normalize_state(seq)   # type: ignore[union-attr]


# ----------------------------------------------------------------------
# 8 thin subclasses: 4 descriptors × 2 channels (action / state)
# ----------------------------------------------------------------------


@register("jerk_online_action")
class JerkOnlineAction(_OnlineActionBase):
    descriptor = "jerk"


@register("jerk_online_state")
class JerkOnlineState(_OnlineStateBase):
    descriptor = "jerk"


@register("direction_online_action")
class DirectionOnlineAction(_OnlineActionBase):
    descriptor = "direction"


@register("direction_online_state")
class DirectionOnlineState(_OnlineStateBase):
    descriptor = "direction"


@register("dispersion_online_action")
class DispersionOnlineAction(_OnlineActionBase):
    descriptor = "dispersion"


@register("dispersion_online_state")
class DispersionOnlineState(_OnlineStateBase):
    descriptor = "dispersion"


@register("path_length_online_action")
class PathLengthOnlineAction(_OnlineActionBase):
    descriptor = "path_length"


@register("path_length_online_state")
class PathLengthOnlineState(_OnlineStateBase):
    descriptor = "path_length"
