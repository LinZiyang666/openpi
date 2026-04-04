"""CacheStorage: the single entry point into the vector store layer.

CacheOrchestrator talks only to this class — never to backends directly.

Coupling map:
  DEPENDS ON:  backend_base.py (VectorStoreBackend), storage_types.py (all types)
  CONSUMED BY: CacheOrchestrator (search/insert/fetch_payload) — ONLY consumer
  IF CHANGED:  Orchestrator's search/write calls may need updating
  NOTE:        Step 4+ code must ONLY interact via this facade, never via backend directly

Responsibilities
----------------
1. Thread safety    — RLock around every backend call.
2. Dim validation   — check query_keys shapes on insert and search.
3. Entry validation — call entry.validate() before every insert.
4. Filter checking  — fail-fast if QueryFilter contains unsupported fields.
5. Two-phase search — search() returns lightweight results; fetch_payload()
                      hydrates a single winner to avoid bulk tensor transfer.
6. MetadataDB hook  — optional secondary store; vector DB is source of truth.

MetadataDB consistency
----------------------
When metadata_db is connected: vector is written first, metadata second.
If metadata write fails, the vector entry remains — it carries the full payload
so the cache is still correct.  close() flushes both stores.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Optional

from openpi.cache.backend_base import VectorStoreBackend
from openpi.cache.storage_types import (
    BatchInsertResult,
    CacheEntry,
    CachePayload,
    QuerySpec,
    SearchResult,
    SearchResultLite,
    UnsupportedFilterError,
)


class CacheStorage:

    def __init__(
        self,
        backend: VectorStoreBackend,
        metadata_db=None,   # reserved for future MetadataStore
    ) -> None:
        self._backend = backend
        self._metadata_db = metadata_db
        self._dims: dict[str, int] = backend.vector_dims
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Search (two-phase)
    # ------------------------------------------------------------------

    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        """Vector search.  Returns lightweight results without payload tensors.

        Use fetch_payload() on the winning candidate, or search_and_fetch()
        when you need full results for all top-k entries.
        """
        self._check_query_dims(spec)
        self._check_filters(spec)
        with self._lock:
            return self._backend.search(spec)

    def fetch_payload(self, id: str) -> CachePayload:
        """Fetch the full payload for one candidate id."""
        with self._lock:
            return self._backend.fetch_payload(id)

    def search_and_fetch(self, spec: QuerySpec) -> list[SearchResult]:
        """Convenience: search then fetch payload for every result.

        Suitable when top_k is small (≤ 5).  For larger top_k, prefer
        calling search() then selectively fetch_payload() on the winner.
        """
        lites = self.search(spec)
        results: list[SearchResult] = []
        for lite in lites:
            payload = self.fetch_payload(lite.id)
            results.append(
                SearchResult(
                    id=lite.id,
                    score=lite.score,
                    payload=payload,
                    checkpoint_id=lite.checkpoint_id,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(self, entry: CacheEntry) -> None:
        self._check_entry_dims(entry)
        entry.validate()
        with self._lock:
            self._backend.insert(entry)
            if self._metadata_db is not None:
                # MetadataDB written after vector; vector is source of truth.
                self._metadata_db.insert(entry)

    def batch_insert(self, entries: list[CacheEntry]) -> BatchInsertResult:
        for entry in entries:
            self._check_entry_dims(entry)
            entry.validate()
        with self._lock:
            result = self._backend.batch_insert(entries)
            if self._metadata_db is not None and result.inserted > 0:
                successful = [e for e in entries if e.id not in result.failed_ids]
                self._metadata_db.batch_insert(successful)
            return result

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def delete(self, ids: list[str]) -> None:
        with self._lock:
            self._backend.delete(ids)

    def count(self) -> int:
        with self._lock:
            return self._backend.count()

    def close(self) -> None:
        """Flush and release resources for both vector store and metadata store."""
        with self._lock:
            self._backend.close()   # backend.close() calls flush() internally
            if self._metadata_db is not None:
                self._metadata_db.close()

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    def _check_query_dims(self, spec: QuerySpec) -> None:
        active = set(spec.query_keys.keys()) & set(self._dims.keys())
        if not active:
            raise ValueError(
                f"QuerySpec.query_keys has no fields matching backend. "
                f"Backend fields: {set(self._dims.keys())}, "
                f"query fields: {set(spec.query_keys.keys())}"
            )
        for field_name in active:
            expected = self._dims[field_name]
            shape = tuple(spec.query_keys[field_name].shape)
            if shape != (expected,):
                raise ValueError(
                    f"query_keys[{field_name!r}] shape mismatch: "
                    f"expected ({expected},), got {shape}"
                )

    def _check_entry_dims(self, entry: CacheEntry) -> None:
        # Intersection-based validation: only check fields present in both
        # entry and backend. At least one must overlap.
        active = set(entry.query_keys.keys()) & set(self._dims.keys())
        if not active:
            raise ValueError(
                f"entry.query_keys has no fields matching backend. "
                f"Backend fields: {set(self._dims.keys())}, "
                f"entry fields: {set(entry.query_keys.keys())}"
            )
        for field_name in active:
            expected = self._dims[field_name]
            shape = tuple(entry.query_keys[field_name].shape)
            if shape != (expected,):
                raise ValueError(
                    f"entry.query_keys[{field_name!r}] shape mismatch: "
                    f"expected ({expected},), got {shape}"
                )

    def _check_filters(self, spec: QuerySpec) -> None:
        if spec.filters is None:
            return
        requested = {
            k
            for k, v in dataclasses.asdict(spec.filters).items()
            if v is not None
        }
        if not requested:
            return
        unsupported = requested - self._backend.supported_filters()
        if unsupported:
            raise UnsupportedFilterError(
                f"Backend does not support filter fields: {unsupported}. "
                f"Supported: {self._backend.supported_filters()}"
            )
