"""In-memory vector store backend for development and testing.

Overview
--------
Stores entries in a Python dict, performs brute-force cosine similarity search.
No external dependencies. Suitable for:
  - Unit/integration tests (no Qdrant required)
  - Development and debugging (--cache_config with type: in_memory)
  - Small-scale cache validation (< 10k entries)

Data flow: CacheStorage -> InMemoryBackend.search/insert/... -> in-process dict

Coupling map:
  DEPENDS ON:  backend_base.py (VectorStoreBackend ABC), storage_types.py
  CONSUMED BY: CacheStorage (via VectorStoreBackend interface)
  IF CHANGED:  tests and development cache configs may need updating
  NOTE:        ignores fusion_weights and backend_hints in QuerySpec (single-field cosine)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from openpi.cache.backend_base import VectorStoreBackend
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    QuerySpec,
    SearchResultLite,
)


class InMemoryBackend(VectorStoreBackend):
    """Minimal in-memory backend using brute-force cosine similarity.

    Stores entries in a dict. search() computes cosine similarity
    against stored vectors for the first matching field.
    Score range: [-1, 1] (same as Qdrant cosine).

    Limitations:
      - Does not support step_range or task_key filters (only checkpoint_id).
      - Ignores fusion_weights and backend_hints (single-field cosine only).
      - Not suitable for production workloads (O(n) search).
    """

    def __init__(self, vector_dims: dict[str, int]) -> None:
        self._dims = vector_dims
        self._entries: dict[str, CacheEntry] = {}
        # Counters for call tracking in tests.
        self.search_call_count: int = 0
        self.fetch_payload_call_count: int = 0

    @property
    def vector_dims(self) -> dict[str, int]:
        return self._dims

    def supported_filters(self) -> frozenset[str]:
        return frozenset({"checkpoint_id"})

    def insert(self, entry: CacheEntry) -> None:
        self._entries[entry.id] = entry

    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        self.search_call_count += 1
        if not self._entries:
            return []
        results: list[SearchResultLite] = []
        for eid, entry in self._entries.items():
            if spec.checkpoint_id is not None and entry.checkpoint_id != spec.checkpoint_id:
                continue
            score = 0.0
            for field in spec.query_keys:
                if field in entry.query_keys:
                    q = spec.query_keys[field].float()
                    e = entry.query_keys[field].float()
                    score = float(F.cosine_similarity(q.unsqueeze(0), e.unsqueeze(0)))
                    break
            results.append(
                SearchResultLite(id=eid, score=score, checkpoint_id=entry.checkpoint_id)
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: spec.top_k]

    def fetch_payload(self, id: str) -> CachePayload:
        self.fetch_payload_call_count += 1
        if id not in self._entries:
            raise KeyError(id)
        return self._entries[id].payload

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._entries.pop(i, None)

    def count(self) -> int:
        return len(self._entries)
