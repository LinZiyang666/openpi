"""Gate: decide whether to perform a cache lookup.

Data flow: KeyBuilder.cached_data (GPU) -> gate() -> bool

Coupling map:
  DEPENDS ON:  nothing (leaf component)
  MAY DEPEND ON: KeyBuilder.cached_data (read-only, for future state-change gates)
  CONSUMED BY: CacheOrchestrator.check() — if False, skip search entirely
  DOES NOT interact with: CacheStorage or any Step 3 component
  IF CHANGED:  Only affects Orchestrator's search frequency, no downstream impact
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from openpi.cache.types import CheckpointID


@runtime_checkable
class GateFunction(Protocol):
    """Decide whether to perform a cache lookup.

    Data flow: KeyBuilder.cached_data (GPU) -> gate() -> bool
    Coupling:
      - MAY DEPEND ON: KeyBuilder.cached_data (read-only, for state-change gates)
      - CONSUMED BY: CacheOrchestrator.check() — if False, skip search entirely
      - DOES NOT interact with: CacheStorage or any Step 3 component
      - IF CHANGED: Only affects Orchestrator's search frequency, no downstream impact
    """

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> bool:
        """Return True if cache should be searched, False to skip.

        Args:
            checkpoint_id: CP1 or CP3.
            cached_data: Raw tensors from KeyBuilder.cached_data (on GPU).
        """
        ...


class AlwaysSearchGate:
    """Always search the cache. Simplest gate for initial development.

    Data flow: (no data consumed) -> always True
    Coupling: None — no dependencies on any other component.
    """

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> bool:
        return True
