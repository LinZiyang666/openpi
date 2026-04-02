"""Qdrant backend for the cache vector store.

Collection management
---------------------
The Qdrant collection must be created manually before use with named-vector
schema matching the fields in QdrantBackendConfig.vector_dims.  This class
does not auto-create collections so that collection schema (distance metric,
quantisation, etc.) remains under human control.

Named vectors
-------------
Each field in vector_dims is stored as a separate named vector in the Qdrant
collection.  This enables per-field KNN with server-side RRF fusion, matching
the retrieval strategy validated in exp/.

Serialisation
-------------
Tensor fields in CachePayload cannot be stored directly in Qdrant's JSON
payload.  They are serialised as torch.save bytes → base64 strings.
Keys for intermediates are stored as f"{t:.4f}" strings to avoid JSON
floating-point round-trip drift.

Two-phase search
----------------
search() fetches only the "checkpoint_id" payload field — the large tensor
fields are NOT transferred.  fetch_payload() does a separate retrieve() call
for the single winner.  This avoids sending ~6–25 KB per candidate when most
candidates are discarded by the judge.

Chunked vectors
---------------
Qdrant limits a single dense vector to 65 535 dimensions.  Raw embeddings
(e.g. vision_0 flattened from [256, 1024] = 262 144 dims) exceed this limit.

The backend handles chunking transparently:
  - On insert: fields wider than _MAX_CHUNK_DIM are split into sequential
    named vectors "field__chunk_000", "field__chunk_001", ...
  - On search: one Prefetch per chunk is built; chunks of the same field share
    the field's total weight divided equally among them.

Callers always work with full-dim tensors.  Chunking is an internal detail.

Fusion strategy
---------------
1 total chunk across all fields → direct query_points (no fusion overhead).
>1 total chunk → all chunks become Prefetch branches + server-side RRF
  via RrfQuery(rrf=Rrf(k=..., weights=...)).
Per-field weights are split equally across that field's chunks so the total
weight contribution of each field is preserved.
The RRF k and per-field weights are configured in QdrantBackendConfig.

Supported filters
-----------------
  checkpoint_id : exact match on the stored checkpoint name (e.g. "CP1")
  task_key      : exact match on the stored task_key string
  step_range    : range filter on the stored integer "step_idx" field
                  (the entry must have step_idx set in payload at write time)
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Prefetch,
    PointStruct,
    Range,
    Rrf,
    RrfQuery,
)

from openpi.cache.backend_base import VectorStoreBackend
from openpi.cache.storage_types import (
    BatchInsertResult,
    CacheEntry,
    CachePayload,
    QuerySpec,
    SearchResultLite,
)
from openpi.cache.types import CheckpointID, VISION_0, PROMPT_EMB, ROBOT_STATE

logger = logging.getLogger(__name__)

# Qdrant hard limit for a single dense named vector.
_MAX_CHUNK_DIM = 65_535


@dataclass
class QdrantBackendConfig:
    url: str = "http://localhost:6333"
    collection_name: str = "openpi_cache"
    # Named vectors this backend stores.  Keys must match collection schema.
    vector_dims: dict[str, int] = field(
        default_factory=lambda: {
            VISION_0: 1024,
            PROMPT_EMB: 2048,
            ROBOT_STATE: 14,
        }
    )
    prefer_grpc: bool = False
    grpc_port: int = 6334
    request_timeout: int = 30
    # RRF fusion parameters (used when query_keys has more than one field)
    rrf_k: int = 60
    candidate_multiplier: int = 5   # prefetch limit = top_k * candidate_multiplier
    fusion_weights: Optional[dict[str, float]] = None  # per-field weight; None = equal


class QdrantVectorStore(VectorStoreBackend):

    _SUPPORTED_FILTERS: frozenset[str] = frozenset(
        {"checkpoint_id", "task_key", "step_range"}
    )

    def __init__(self, config: QdrantBackendConfig) -> None:
        self._config = config
        self._client = QdrantClient(
            url=config.url,
            grpc_port=config.grpc_port,
            prefer_grpc=config.prefer_grpc,
            timeout=config.request_timeout,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def vector_dims(self) -> dict[str, int]:
        return self._config.vector_dims

    def supported_filters(self) -> frozenset[str]:
        return self._SUPPORTED_FILTERS

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def insert(self, entry: CacheEntry) -> None:
        self._client.upsert(
            collection_name=self._config.collection_name,
            points=[self._to_point(entry)],
        )

    def batch_insert(self, entries: list[CacheEntry]) -> BatchInsertResult:
        points: list[PointStruct] = []
        failed_ids: list[str] = []
        for entry in entries:
            try:
                points.append(self._to_point(entry))
            except Exception as exc:
                logger.warning("batch_insert serialisation failed for %s: %s", entry.id, exc)
                failed_ids.append(entry.id)

        if points:
            self._client.upsert(
                collection_name=self._config.collection_name,
                points=points,
            )
        return BatchInsertResult(inserted=len(points), failed_ids=failed_ids)

    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        """Lightweight search: fetches only the checkpoint_id payload field.

        Builds one Prefetch per chunk per active field.
        1 total chunk → direct query (no RRF overhead).
        >1 total chunks → RRF fusion across all prefetches.
        """
        active_fields = sorted(
            f for f in spec.query_keys if f in self._config.vector_dims
        )
        if not active_fields:
            return []

        query_filter = self._make_filter(spec)

        # Build (chunk_name, values, field_name) for every chunk of every field
        all_chunks: list[tuple[str, list[float], str]] = []
        for field_name in active_fields:
            for chunk_name, chunk_values in self._field_to_chunks(
                field_name, spec.query_keys[field_name]
            ):
                all_chunks.append((chunk_name, chunk_values, field_name))

        if len(all_chunks) == 1:
            chunk_name, chunk_values, _ = all_chunks[0]
            response = self._client.query_points(
                collection_name=self._config.collection_name,
                query=chunk_values,
                using=chunk_name,
                query_filter=query_filter,
                limit=spec.top_k,
                with_payload=["checkpoint_id"],
                with_vectors=False,
            )
        else:
            candidate_limit = spec.top_k * self._config.candidate_multiplier
            prefetches = [
                Prefetch(query=values, using=name, filter=query_filter, limit=candidate_limit)
                for name, values, _ in all_chunks
            ]
            fusion_weights = self._build_fusion_weights_for_chunks(active_fields, all_chunks)
            rrf = Rrf(k=self._config.rrf_k, weights=fusion_weights)
            response = self._client.query_points(
                collection_name=self._config.collection_name,
                prefetch=prefetches,
                query=RrfQuery(rrf=rrf),
                limit=spec.top_k,
                with_payload=["checkpoint_id"],
                with_vectors=False,
            )

        results: list[SearchResultLite] = []
        for point in response.points:
            cp_name = (point.payload or {}).get("checkpoint_id")
            try:
                cp = CheckpointID[cp_name] if cp_name else CheckpointID.CP1
            except KeyError:
                logger.warning("Unknown checkpoint_id %r in search result %s", cp_name, point.id)
                continue
            results.append(
                SearchResultLite(id=str(point.id), score=float(point.score), checkpoint_id=cp)
            )
        return results

    def fetch_payload(self, id: str) -> CachePayload:
        results = self._client.retrieve(
            collection_name=self._config.collection_name,
            ids=[id],
            with_payload=True,
        )
        if not results:
            raise KeyError(f"id {id!r} not found in collection {self._config.collection_name!r}")
        return self._deserialize_payload(results[0].payload)

    def delete(self, ids: list[str]) -> None:
        self._client.delete(
            collection_name=self._config.collection_name,
            points_selector=ids,
        )

    def count(self) -> int:
        return self._client.count(
            collection_name=self._config.collection_name
        ).count

    # ------------------------------------------------------------------
    # Internal: point construction
    # ------------------------------------------------------------------

    def _to_point(self, entry: CacheEntry) -> PointStruct:
        vectors: dict[str, list[float]] = {}
        for field_name in self._config.vector_dims:
            if field_name not in entry.query_keys:
                continue
            for chunk_name, chunk_values in self._field_to_chunks(
                field_name, entry.query_keys[field_name]
            ):
                vectors[chunk_name] = chunk_values
        return PointStruct(
            id=entry.id,
            vector=vectors,
            payload={
                "checkpoint_id": entry.checkpoint_id.name,
                "timestamp": entry.timestamp,
                "task_key": entry.payload.task_key,
                **self._serialize_payload(entry.payload),
            },
        )

    @staticmethod
    def _field_to_chunks(
        field_name: str, tensor: torch.Tensor | np.ndarray
    ) -> list[tuple[str, list[float]]]:
        """Split tensor into Qdrant-compatible named chunks.

        Accepts torch.Tensor or numpy.ndarray.
        Returns [(vector_name, values), ...].
        Single chunk → vector_name == field_name (no suffix).
        Multiple chunks → "field__chunk_000", "field__chunk_001", ...
        """
        import numpy as np
        if hasattr(tensor, "numpy"):  # torch.Tensor
            flat = tensor.float().cpu().numpy().astype("float32").flatten().tolist()
        else:
            flat = np.asarray(tensor, dtype=np.float32).flatten().tolist()
        dim = len(flat)
        if dim <= _MAX_CHUNK_DIM:
            return [(field_name, flat)]

        num_chunks = (dim + _MAX_CHUNK_DIM - 1) // _MAX_CHUNK_DIM
        chunks = []
        for i in range(num_chunks):
            start = i * _MAX_CHUNK_DIM
            end = min(start + _MAX_CHUNK_DIM, dim)
            chunks.append((f"{field_name}__chunk_{i:03d}", flat[start:end]))
        return chunks

    def _build_fusion_weights_for_chunks(
        self,
        active_fields: list[str],
        all_chunks: list[tuple[str, list[float], str]],
    ) -> Optional[list[float]]:
        """Per-prefetch weight list with equal split within each field's chunks.

        Returns None when fusion_weights is not configured (equal weights for all
        prefetches, which is Qdrant's default when weights=None).
        """
        cfg = self._config.fusion_weights
        if not cfg:
            return None

        # Count chunks per field from the actual chunk list
        field_chunk_counts: dict[str, int] = {}
        for _, _, field_name in all_chunks:
            field_chunk_counts[field_name] = field_chunk_counts.get(field_name, 0) + 1

        total = sum(cfg.get(f, 1.0) for f in active_fields)
        if total <= 0:
            return None

        weights = []
        for _, _, field_name in all_chunks:
            per_field = cfg.get(field_name, 1.0) / total
            per_chunk = per_field / field_chunk_counts[field_name]
            weights.append(per_chunk)
        return weights

    # ------------------------------------------------------------------
    # Internal: serialisation / deserialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_b64(t: Optional[torch.Tensor]) -> Optional[str]:
        if t is None:
            return None
        import torch
        buf = io.BytesIO()
        torch.save(t, buf)
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _b64_to_tensor(s: Optional[str]) -> Optional[torch.Tensor]:
        if s is None:
            return None
        import torch
        buf = io.BytesIO(base64.b64decode(s))
        return torch.load(buf, weights_only=True)

    @classmethod
    def _serialize_payload(cls, p: CachePayload) -> dict:
        intermediates_b64: Optional[dict[str, Optional[str]]] = None
        if p.intermediates:
            intermediates_b64 = {
                f"{t:.4f}": cls._tensor_to_b64(x_t)
                for t, x_t in p.intermediates.items()
            }

        return {
            "action_chunk": cls._tensor_to_b64(p.action_chunk),
            "intermediates": intermediates_b64,
            "denoising_num_steps": p.denoising_num_steps,
            "next_action_chunk": cls._tensor_to_b64(p.next_action_chunk),
        }

    @classmethod
    def _deserialize_payload(cls, raw: dict) -> CachePayload:
        intermediates: Optional[dict[float, torch.Tensor]] = None
        if raw.get("intermediates"):
            intermediates = {
                float(k): cls._b64_to_tensor(v)
                for k, v in raw["intermediates"].items()
            }

        return CachePayload(
            action_chunk=cls._b64_to_tensor(raw["action_chunk"]),
            intermediates=intermediates,
            denoising_num_steps=raw.get("denoising_num_steps"),
            next_action_chunk=cls._b64_to_tensor(raw.get("next_action_chunk")),
            task_key=raw.get("task_key", ""),
        )

    # ------------------------------------------------------------------
    # Internal: filter construction
    # ------------------------------------------------------------------

    @staticmethod
    def _make_filter(spec: QuerySpec) -> Optional[Filter]:
        conditions: list[FieldCondition] = []

        if spec.checkpoint_id is not None:
            conditions.append(
                FieldCondition(
                    key="checkpoint_id",
                    match=MatchValue(value=spec.checkpoint_id.name),
                )
            )

        if spec.filters is not None:
            f = spec.filters
            if f.task_key is not None:
                conditions.append(
                    FieldCondition(
                        key="task_key",
                        match=MatchValue(value=f.task_key),
                    )
                )
            if f.step_range is not None:
                lo, hi = f.step_range
                conditions.append(
                    FieldCondition(
                        key="step_idx",
                        range=Range(gte=lo, lte=hi),
                    )
                )

        return Filter(must=conditions) if conditions else None
