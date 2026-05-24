"""BackendPool — process-local singleton sharing in-memory backends by fingerprint.

Many sweep yamls share the same on-disk artifact (e.g. all `cp1_spatial_pool_16`
yamls in phase5_systematic preload the same 76 MB pkl). Pre-pool architecture
loaded one Backend instance per yaml → N × 76 MB. With the pool, identical
fingerprints share one Backend instance; only distinct fingerprints incur a
fresh load.

Fingerprint identity
--------------------
Two BackendConfigs share a pool slot iff their ``BackendFingerprint`` is equal.
The fingerprint covers (backend_type, resolved_preload_path, vector_dims tuple,
index_type) — any divergence yields independent backends so each gets its own
``load_artifact`` validation (which fail-fasts on artifact / vector_dims
mismatch).

Lifecycle (G1 R2 Item 5 — build/load responsibility cleanly split)
------------------------------------------------------------------
- Pool path (in_memory + non-empty preload_path):
  ``_build_empty_backend(cfg) → backend.load_artifact(resolved_path) → backend.freeze()``
  Exactly one load per fingerprint, even under concurrent ``get_or_load`` calls.
- Bypass paths (Qdrant; in_memory with empty preload_path):
  ``_build_legacy_complete(cfg)`` — fresh backend, no pool caching, freeze
  applied immediately for C2 contract uniformity.

Threading
---------
``get_or_load`` is safe to call from multiple OS threads concurrently. Two
locks: a coarse instance lock guards the singleton itself and the per-
fingerprint lock registry; a per-fingerprint lock guards the first-load
critical section so identical fingerprints incur exactly one ``load_artifact``
call even under racing callers.

Coupling
--------
DEPENDS ON: cache/backend_base.py (VectorStoreBackend), cache/config.py
            (BackendConfig schema), cache/backends/* (concrete backends)
CONSUMED BY: cache/config.py::build_shared_storage (the only caller)
IF CHANGED: build_shared_storage call sites
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from openpi.cache.backend_base import VectorStoreBackend

if TYPE_CHECKING:
    from openpi.cache.config import BackendConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Fingerprint
# ------------------------------------------------------------------


@dataclass(frozen=True)
class BackendFingerprint:
    """Identity used to decide pool hits.

    Any field differing = a distinct backend that must be independently loaded.
    Two callers with the same fingerprint share one backend instance.

    Fields chosen so that ``load_artifact`` fail-fasts are not silently bypassed:
    same resolved_preload_path but different vector_dims → different fingerprints
    → both load and each surface their own mismatch error.
    """

    backend_type: str
    """e.g. ``"in_memory"`` (only in_memory is pooled today; Qdrant bypass)."""

    resolved_preload_path: str
    """``Path(...).resolve()`` absolute string — eliminates symlink / cwd
    aliasing so "same physical file via different paths" share one slot."""

    vector_dims: tuple[tuple[str, int], ...]
    """Sorted ``tuple(cfg.vector_dims.items())`` for hashability."""

    index_type: str
    """Currently always ``"brute_force"`` for InMemory; reserved for future
    indexing backends."""

    @classmethod
    def from_config(cls, cfg: "BackendConfig") -> "BackendFingerprint":
        """Build a fingerprint from a BackendConfig.

        Pool path callers must guarantee `cfg.type == "in_memory"` and
        `cfg.in_memory.preload_path` is non-empty before invoking this.
        """
        if cfg.type != "in_memory":
            raise ValueError(
                f"BackendFingerprint.from_config only supports in_memory backends; "
                f"got type={cfg.type!r}. Use _build_legacy_complete for others."
            )
        if not cfg.in_memory.preload_path:
            raise ValueError(
                "BackendFingerprint.from_config requires a non-empty preload_path. "
                "Empty preload_path bypasses the pool."
            )
        return cls(
            backend_type=cfg.type,
            resolved_preload_path=str(Path(cfg.in_memory.preload_path).resolve()),
            vector_dims=tuple(sorted(cfg.vector_dims.items())),
            index_type=cfg.in_memory.index_type,
        )


# ------------------------------------------------------------------
# Build helpers — responsibility split (G1 R2 Item 5)
# ------------------------------------------------------------------


def _build_empty_backend(cfg: "BackendConfig") -> VectorStoreBackend:
    """Construct a backend instance WITHOUT calling load_artifact.

    The pool uses this to ensure that ``load_artifact`` is invoked exactly
    once per fingerprint (the pool itself owns the load + freeze sequence).
    Callers outside the pool should use ``_build_legacy_complete``.
    """
    if cfg.type == "in_memory":
        from openpi.cache.backends.in_memory_backend import InMemoryBackend

        return InMemoryBackend(vector_dims=cfg.vector_dims)
    if cfg.type == "qdrant":
        from openpi.cache.backends.qdrant_backend import (
            QdrantBackendConfig,
            QdrantVectorStore,
        )

        qdrant_config = QdrantBackendConfig(
            url=cfg.qdrant.url,
            collection_name=cfg.qdrant.collection_name,
            vector_dims=cfg.vector_dims,
            prefer_grpc=cfg.qdrant.prefer_grpc,
            grpc_port=cfg.qdrant.grpc_port,
            request_timeout=cfg.qdrant.request_timeout,
        )
        return QdrantVectorStore(config=qdrant_config)
    raise ValueError(f"Unknown backend.type {cfg.type!r}. Valid: ['in_memory', 'qdrant']")


def _build_legacy_complete(cfg: "BackendConfig") -> VectorStoreBackend:
    """Build + (optional load) + freeze in one shot for non-pool paths.

    Used for:
      - Qdrant backends (no preload concept; freeze after build for C2 uniformity)
      - in_memory backends with empty preload_path (freeze an empty backend)
    """
    backend = _build_empty_backend(cfg)
    if cfg.type == "in_memory" and cfg.in_memory.preload_path:
        # Defense-in-depth: callers should already route non-empty preload_paths
        # through the pool. Loading here keeps single-shot legacy semantics for
        # any direct invocation (e.g. offline tooling that bypasses the pool).
        backend.load_artifact(cfg.in_memory.preload_path)
    backend.freeze()
    return backend


# ------------------------------------------------------------------
# Pool
# ------------------------------------------------------------------


class BackendPool:
    """Process-local singleton: BackendFingerprint → frozen Backend.

    Lazy load on first ``get_or_load`` for a fingerprint. Concurrent first-load
    is guarded by a per-fingerprint lock + double-check so identical fingerprints
    incur exactly one ``load_artifact`` call.

    All pooled backends are immediately frozen after load (C2 contract).
    Qdrant and empty-preload in_memory backends bypass the pool entirely.
    """

    _instance: Optional["BackendPool"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "BackendPool":
        """Return the process-local singleton, creating it on first call."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop the singleton — used by unit tests to start each test fresh."""
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._backends: dict[BackendFingerprint, VectorStoreBackend] = {}
        self._load_locks: dict[BackendFingerprint, threading.Lock] = {}
        self._load_locks_guard = threading.Lock()

    def get_or_load(self, cfg: "BackendConfig") -> VectorStoreBackend:
        """Return a frozen backend for ``cfg``, loading + caching when first seen.

        - in_memory + non-empty preload_path: pool path (cached, shared by fingerprint)
        - Qdrant or empty preload_path: bypass pool, fresh ``_build_legacy_complete``
        """
        if cfg.type != "in_memory" or not cfg.in_memory.preload_path:
            return _build_legacy_complete(cfg)

        fp = BackendFingerprint.from_config(cfg)
        # Fast path: already loaded.
        existing = self._backends.get(fp)
        if existing is not None:
            return existing

        load_lock = self._get_load_lock(fp)
        with load_lock:
            # Double-check after acquiring the load lock: another thread
            # may have loaded this fingerprint while we waited.
            existing = self._backends.get(fp)
            if existing is not None:
                return existing
            logger.info(
                "BackendPool: loading new fingerprint (path=%s, dims=%s)",
                fp.resolved_preload_path, dict(fp.vector_dims),
            )
            backend = _build_empty_backend(cfg)
            backend.load_artifact(fp.resolved_preload_path)
            backend.freeze()  # C2: immediate freeze after load
            self._backends[fp] = backend
            return backend

    def _get_load_lock(self, fp: BackendFingerprint) -> threading.Lock:
        with self._load_locks_guard:
            lock = self._load_locks.get(fp)
            if lock is None:
                lock = threading.Lock()
                self._load_locks[fp] = lock
            return lock
