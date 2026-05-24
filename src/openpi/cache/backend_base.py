"""Abstract base class for vector store backends.

All backends must implement this interface.  CacheStorage (the facade above
this layer) is the only caller; application code never touches backends directly.

Coupling map:
  DEPENDS ON:  storage_types.py (CacheEntry, CachePayload, QuerySpec, etc.)
  CONSUMED BY: CacheStorage (the only caller), concrete backends (Qdrant, future FAISS)
  IF CHANGED:  CacheStorage internal calls, all concrete backend implementations
  NOTE:        Application code (Orchestrator, Interceptor) must NEVER import this

Runtime write-frozen contract (C2)
-----------------------------------
Server runtime forbids modifying database content. Backends transition to a
read-only state after `freeze()` is called (typically right after
`load_artifact()` in the BackendPool). Once frozen, any mutation entry
(`insert` / `batch_insert` / `delete` / `upsert` / `load_artifact`) raises
`BackendFrozenError`. Read methods (`search` / `fetch_payload` / `fetch_entry`
/ `count`) and derived state mutation (`open_search_session` /
`close_search_session` + per-session in-memory caches) remain allowed —
these are search-path caches, not database content.

With this contract, GIL atomicity of dict lookup is sufficient for concurrent
read safety; no RLock is required. Offline artifact build / enrich must
happen before freeze (e.g. via offline tooling, not through the live server).

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


class BackendFrozenError(RuntimeError):
    """Raised when a mutation entry is called on a frozen backend (C2).

    Mutation entries: insert / batch_insert / delete / upsert / load_artifact.
    Read entries (search / fetch_payload / fetch_entry / count) and derived
    state mutation (open_search_session / close_search_session and per-session
    in-memory caches like `_score_memo`) are NOT considered mutations of
    database content — those remain allowed after freeze.
    """


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

        Trajectory search: if spec.trajectory_history is not None and
        len(spec.trajectory_weights) > 1, the backend should compute
        trajectory-aware similarity in a single pass. Backends that do not
        support this MUST raise NotImplementedError.
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

    # ------------------------------------------------------------------
    # Optional search-session memo capability
    # ------------------------------------------------------------------

    def open_search_session(self, session_id: str) -> None:
        """Register an active per-strategy search session.

        Default no-op. Backends that maintain per-session in-memory caches
        (e.g. InMemoryBackend's per-step score memo) override this to track
        the session in their active set, so that mutation guards and cache
        lookups behave correctly. Backends without internal cache leave the
        default and incur zero cost.
        """

    def close_search_session(self, session_id: str) -> None:
        """Release a previously-opened search session.

        Default no-op. Override symmetrically with `open_search_session` to
        drop session-scoped buckets and remove the session from the active set.
        Idempotent: closing an unknown session_id is a safe no-op.
        """

    # ------------------------------------------------------------------
    # Runtime write-frozen lifecycle (C2)
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Transition this backend to runtime read-only mode (C2).

        After freeze, every mutation entry (insert / batch_insert / delete /
        upsert / load_artifact) must raise BackendFrozenError. Default
        implementation flips a private flag; concrete backends are expected
        to call `self._check_frozen(op)` at the head of each mutation entry.

        Idempotent: calling freeze() repeatedly is safe.
        """
        self._is_frozen = True

    @property
    def is_frozen(self) -> bool:
        """True iff freeze() has been called."""
        return getattr(self, "_is_frozen", False)

    def _check_frozen(self, op_name: str) -> None:
        """Raise BackendFrozenError if this backend has been frozen.

        Concrete backends call this at the head of every mutation entry.
        """
        if self.is_frozen:
            raise BackendFrozenError(
                f"Backend is frozen (runtime write-frozen contract, C2). "
                f"Cannot perform {op_name!r}. "
                "All mutations must happen before freeze() or in offline tooling."
            )
