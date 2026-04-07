"""Write policy: pluggable switch to decide whether to write trajectory at episode end.

Coupling map:
  DEPENDS ON:  storage_types.py (EpisodeRecord), types.py (CheckpointID)
  CONSUMED BY: CacheOrchestrator.on_episode_end()
  IF CHANGED:  Orchestrator write decision logic
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openpi.cache.storage_types import EpisodeRecord
from openpi.cache.types import CheckpointID


@runtime_checkable
class WritePolicy(Protocol):
    """Decides whether to write the episode trajectory to cache at episode end."""

    def should_write(self, episode_record: EpisodeRecord) -> bool: ...


class OnAnyMissWritePolicy:
    """Write if the episode had any CP1 cache miss or gate skip.

    Rationale: episodes with zero misses are fully covered by existing cache;
    only episodes with novel steps benefit from being stored.
    """

    def should_write(self, episode_record: EpisodeRecord) -> bool:
        return episode_record.miss_by_checkpoint.get(CheckpointID.CP1, 0) > 0


class AlwaysWritePolicy:
    """Always write, regardless of cache hit/miss ratio."""

    def should_write(self, episode_record: EpisodeRecord) -> bool:
        return True


class NeverWritePolicy:
    """Never write (read-only mode)."""

    def should_write(self, episode_record: EpisodeRecord) -> bool:
        return False
