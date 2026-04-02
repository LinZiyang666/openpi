"""Abstract base class for vector store backends.

All backends must implement this interface.  CacheStorage (the facade above
this layer) is the only caller; application code never touches backends directly.

Threading
---------
Backends are NOT required to be thread-safe.  CacheStorage serialises all
calls with an RLock.

Score convention
----------------
search() must return scores as normalised cosine similarity ∈ [-1, 1].
Higher is more similar.  Convert your backend's native distance metric before
returning (e.g. for Qdrant Cosine the score is already in this range).
"""

from abc import ABC, abstractmethod

from openpi.cache.storage_types import (
    BatchInsertResult,
    CacheEntry,
    CachePayload,
    QuerySpec,
    SearchResultLite,
)


class VectorStoreBackend(ABC):

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def vector_dims(self) -> dict[str, int]:
        """Field names → embedding dimensions this backend stores and queries.

        Keys must be a subset of CACHE_QUERY_FIELDS (openpi.cache.types).
        CacheStorage reads this once at construction to validate all inputs.

        Example: {"vision_0": 1024, "prompt_emb": 2048, "robot_state": 14}
        """

    @abstractmethod
    def supported_filters(self) -> frozenset[str]:
        """Names of QueryFilter fields this backend can apply server-side (or
        client-side while preserving correct semantics).

        Any field NOT in this set that appears in a QuerySpec will cause
        CacheStorage to raise UnsupportedFilterError before calling the backend.

        Example return value: frozenset({"checkpoint_id", "task_key"})
        """

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    def insert(self, entry: CacheEntry) -> None:
        """Upsert one entry.  Same id → overwrite with latest data, no error."""

    @abstractmethod
    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        """Return at most spec.top_k results ordered by descending score.
        Payload is NOT fetched here; use fetch_payload() for the winner(s).
        """

    @abstractmethod
    def fetch_payload(self, id: str) -> CachePayload:
        """Fetch the full payload for a previously found id.
        Raises KeyError if the id does not exist.
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete entries by id.  Non-existent ids are silently ignored."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of entries currently stored."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def batch_insert(self, entries: list[CacheEntry]) -> BatchInsertResult:
        """Insert multiple entries.  Default: sequential single inserts.
        Backends with a native bulk API should override for better throughput.

        Partial failures are tolerated: failed entries are logged and skipped,
        successful ones are committed.  The caller must not assume atomicity.
        """
        import logging
        logger = logging.getLogger(__name__)

        inserted = 0
        failed_ids: list[str] = []
        for entry in entries:
            try:
                self.insert(entry)
                inserted += 1
            except Exception as exc:
                logger.warning("batch_insert: failed to insert %s: %s", entry.id, exc)
                failed_ids.append(entry.id)
        return BatchInsertResult(inserted=inserted, failed_ids=failed_ids)

    def flush(self) -> None:
        """Persist any buffered writes.  In-memory backends override this;
        remote/file backends are typically no-op.
        """

    def close(self) -> None:
        """Release resources.  Calls flush() first."""
        self.flush()
