"""Layer 1 Normalization Protocol.

A Normalization instance is constructed once at server startup from a
``LibraryStats`` (per-DOF sigma + active mask). Per verdict, Factor
subclasses call ``normalize_action`` / ``normalize_state`` on raw
``[..., A]`` / ``[..., S]`` tensors to obtain z-scored data restricted
to the active subspace.

Coupling map:
  DEPENDS ON:  components/factors/base.py (LibraryStats)
  CONSUMED BY: components/factors/online.py, offline.py
               (factors call ``ctx.normalization.normalize_*`` on demand)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from openpi.cache.components.factors.base import LibraryStats


@runtime_checkable
class Normalization(Protocol):
    """Layer 1 — z-score (or future variants) over raw action / state tensors.

    Constructor must be ``__init__(library_stats: LibraryStats, **params)``.
    Implementations MUST fail-fast at construction time if the required
    sigma / active-mask fields are missing from ``library_stats``.

    There is no runtime mutation: ``library_stats`` is consumed at
    construction and the instance is immutable after that. Per-verdict
    calls are pure functions over input tensors.
    """

    def __init__(self, library_stats: LibraryStats, **params) -> None: ...

    def normalize_action(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw ``[..., A]`` action tensor to normalized ``[..., A_active]``.

        Implementation contract:
          - Apply per-DOF z-score using the action sigma.
          - Restrict to the action active subspace (drop padded DOFs).
          - Return a CPU float32 tensor with the leading dims preserved
            and the trailing dim shrunk to ``A_active = action_active_mask.sum()``.
          - May return a zero-width trailing dim if the active mask is empty;
            callers are responsible for treating that as a NaN signal.
        """
        ...

    def normalize_state(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw ``[..., S]`` robot_state tensor to normalized ``[..., S_active]``.

        Mirror of ``normalize_action`` for the state channel. Same shape
        and dtype contract; uses ``state_sigma`` and ``state_active_mask``.
        """
        ...
