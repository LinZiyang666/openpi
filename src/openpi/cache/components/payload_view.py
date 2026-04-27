"""PayloadView: per-check() lazy fetch + chain walk facade for Judge.

Wraps CacheStorage.fetch_payload + CacheStorage.fetch_entry (the latter
duck-types backend's optional fetch_entry capability — see
`cache_storage.py` `fetch_entry` / `library_stats`). Judge gets read-only
access to candidate payload + neighbor entries; never holds CacheStorage
handle directly.

Coupling map:
  DEPENDS ON:  cache_storage.py (CacheStorage.fetch_payload, CacheStorage.fetch_entry)
  CONSUMED BY: CompositeJudge (via Orchestrator.check kw-only injection in B1+)
  IF CHANGED:  CompositeJudge factor extractors may need adaptation

B0 scope: get / get_entry / get_many / walk_prev / walk_next implemented
for the no-fork chain shape currently produced by the write path.
ForkPolicy values other than TRAJECTORY and `cross_trajectory=True` are
parameter placeholders that raise NotImplementedError; explicit failure
is preferred over silently picking an arbitrary branch when a real fork
appears.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Protocol, runtime_checkable

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.storage_types import CacheEntry, CachePayload


# ------------------------------------------------------------------
# Fork handling enum (signature-only in B0)
# ------------------------------------------------------------------


class ForkPolicy(Enum):
    """How chain walks resolve a node with multiple prev/next neighbors.

    Only TRAJECTORY is supported in B0. The other values are reserved so
    callers can write forward-compatible signatures; passing one of them
    raises NotImplementedError until the corresponding semantics land.
    """

    TRAJECTORY = auto()      # Default: stay on chain whose trajectory_id matches anchor
    FIRST = auto()           # Pick prev_ids[0] / next_ids[0] deterministically
    STOP = auto()            # Stop on first fork; return what was collected
    ALL_BRANCHES = auto()    # Return list[list[CacheEntry]] for each branch
    SCORE = auto()           # Pick branch with highest similarity to anchor query


# ------------------------------------------------------------------
# Protocol
# ------------------------------------------------------------------


@runtime_checkable
class PayloadView(Protocol):
    """Read-only Judge-side facade over CacheStorage entries / payloads.

    Returned tensors live on CPU and are not mutated by the view; callers
    must not mutate them either. Lifetime is one Orchestrator.check()
    call: a fresh instance is constructed per call so memo dictionaries
    do not leak across verdicts.
    """

    def get(self, entry_id: str) -> CachePayload: ...
    def get_entry(self, entry_id: str) -> CacheEntry: ...
    def get_many(self, entry_ids: list[str]) -> list[CachePayload]: ...

    def walk_prev(
        self,
        entry_id: str,
        k: int,
        *,
        fork_policy: ForkPolicy = ForkPolicy.TRAJECTORY,
        cross_trajectory: bool = False,
    ) -> list[CacheEntry]: ...

    def walk_next(
        self,
        entry_id: str,
        k: int,
        *,
        fork_policy: ForkPolicy = ForkPolicy.TRAJECTORY,
        cross_trajectory: bool = False,
    ) -> list[CacheEntry]: ...


# ------------------------------------------------------------------
# Default implementation
# ------------------------------------------------------------------


class StoragePayloadView:
    """Default PayloadView implementation backed by CacheStorage.

    Memoizes (entry_id -> payload) and (entry_id -> entry) lookups for
    the lifetime of one Orchestrator.check() call. The memo deduplicates
    fetches across the multi-extractor pipeline (e.g. F1a-T walking the
    state chain plus F2 fetching top-k payloads can share intermediate
    entry lookups).
    """

    def __init__(self, storage: CacheStorage) -> None:
        self._storage = storage
        self._payload_memo: dict[str, CachePayload] = {}
        self._entry_memo: dict[str, CacheEntry] = {}

    # ------------------------------------------------------------------
    # Single-entry / batch fetch
    # ------------------------------------------------------------------

    def get(self, entry_id: str) -> CachePayload:
        if entry_id not in self._payload_memo:
            self._payload_memo[entry_id] = self._storage.fetch_payload(entry_id)
        return self._payload_memo[entry_id]

    def get_entry(self, entry_id: str) -> CacheEntry:
        if entry_id not in self._entry_memo:
            # Goes through `CacheStorage.fetch_entry`, which duck-types the
            # backend's optional `fetch_entry` capability. InMemoryBackend
            # exposes it as a plain public method (the Backend ABC stays
            # unchanged); other backends raise NotImplementedError, which
            # `validate_cache_config` catches up-front for chain-walking
            # composite judges.
            self._entry_memo[entry_id] = self._storage.fetch_entry(entry_id)
        return self._entry_memo[entry_id]

    def get_many(self, entry_ids: list[str]) -> list[CachePayload]:
        return [self.get(eid) for eid in entry_ids]

    # ------------------------------------------------------------------
    # Chain walk
    # ------------------------------------------------------------------

    def walk_prev(
        self,
        entry_id: str,
        k: int,
        *,
        fork_policy: ForkPolicy = ForkPolicy.TRAJECTORY,
        cross_trajectory: bool = False,
    ) -> list[CacheEntry]:
        return self._walk(entry_id, k, "prev_ids", fork_policy, cross_trajectory)

    def walk_next(
        self,
        entry_id: str,
        k: int,
        *,
        fork_policy: ForkPolicy = ForkPolicy.TRAJECTORY,
        cross_trajectory: bool = False,
    ) -> list[CacheEntry]:
        return self._walk(entry_id, k, "next_ids", fork_policy, cross_trajectory)

    def _walk(
        self,
        entry_id: str,
        k: int,
        link_attr: str,
        fork_policy: ForkPolicy,
        cross_trajectory: bool,
    ) -> list[CacheEntry]:
        if cross_trajectory:
            raise NotImplementedError(
                "cross_trajectory=True is not implemented in B0"
            )
        if fork_policy is not ForkPolicy.TRAJECTORY:
            raise NotImplementedError(
                f"fork_policy={fork_policy.name} is not implemented in B0; "
                "only ForkPolicy.TRAJECTORY is supported"
            )
        result: list[CacheEntry] = []
        if k <= 0:
            return result
        cur = self.get_entry(entry_id)
        anchor_traj_id = cur.trajectory_id
        for _ in range(k):
            link_ids = getattr(cur, link_attr)
            if not link_ids:
                break
            if len(link_ids) > 1:
                # Real fork found — refuse to silently pick a branch. When
                # writers start producing semantic-dedup ids that share
                # prev/next chains, the corresponding ForkPolicy semantics
                # will be defined and this branch will dispatch instead of
                # raising.
                raise NotImplementedError(
                    f"fork detected at {cur.id}: {link_attr}={link_ids}; "
                    "ForkPolicy.TRAJECTORY anchor selection is not "
                    "implemented in B0"
                )
            nxt = self.get_entry(link_ids[0])
            if nxt.trajectory_id != anchor_traj_id:
                # Trajectory boundary — TRAJECTORY policy stops here.
                break
            result.append(nxt)
            cur = nxt
        return result
