"""Injectors that swap a built cache component set's backend / keybuilder for the exp-layer
optimizations (cp1 latency tuning, weighted_sum R1-R5 + weighted_rrf rounds).

Each ``attach_*`` builds via the src ``build_cache_components`` (so the library loads exactly
like the server), transplants the already-populated ``_entries`` dict *reference* into a
fresh optimized subclass (NO 1.1 GB reload), prebuilds the per-bucket matrices, and swaps
the facade's ``_backend`` (or ``components["key_builder"]``). Touches ZERO src — only the
public ``CacheStorage._backend`` / ``components["key_builder"]`` attributes are reassigned.

Backends: attach_prebuilt (R1) / attach_prenorm_dot (R2) / attach_lean_search (R3) /
attach_release_vision (R5) for weighted_sum; attach_lean_rrf / attach_rrf_release for
weighted_rrf. Keybuilder: attach_batched_pool_keybuilder (R4 build, fusion-orthogonal).
"""

from __future__ import annotations

from typing import Any

from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
from openpi.cache.config import build_cache_components

from exp.cache_latency_bench.opt.lean_rrf_backend import LeanRRFBackend
from exp.cache_latency_bench.opt.lean_search_backend import LeanSearchBackend
from exp.cache_latency_bench.opt.prebuilt_matrix_backend import PrebuiltMatrixBackend
from exp.cache_latency_bench.opt.prenorm_dot_backend import PrenormDotBackend
from exp.cache_latency_bench.opt.r4_pool_keybuilder import CP1SpatialPool16BatchedKeyBuilder
from exp.cache_latency_bench.opt.safe_release_backend import ReleaseVisionBackend
from exp.cache_latency_bench.opt.safe_release_rrf_backend import RrfReleaseVisionBackend


def attach_prebuilt(storage) -> PrebuiltMatrixBackend:
    """Swap ``storage._backend`` for a prebuilt subclass sharing the same ``_entries``.

    Returns the new backend. The original InMemoryBackend instance keeps owning the
    ``_entries`` dict; the subclass shares the same reference (no copy), so both see one
    library. Asserts the share + swap held.
    """
    old = storage._backend
    new = PrebuiltMatrixBackend(old.vector_dims)
    # Share every instance attribute, crucially the populated _entries dict reference.
    # _mat / _id2row / _prebuilt set in __init__ survive (old has no such keys).
    new.__dict__.update(old.__dict__)
    new._prebuild_task_matrices()
    assert new._entries is old._entries, "entries reference not shared after transplant"
    # Defensive: confirm the subclass containers survived __dict__.update (old has no
    # such keys, so prebuild output must be intact). Guards a future parent that adds a
    # colliding attribute name.
    assert new._prebuilt and new._mat, "prebuild output lost (clobbered by __dict__.update?)"
    assert not new._has_active_search_sessions(), "no search session may be active at prebuild"
    storage._backend = new
    assert storage._backend is new
    return new


def build_components_with_prebuilt(config) -> dict[str, Any]:
    """Return cache components whose backend is a prebuilt ``PrebuiltMatrixBackend``."""
    components = build_cache_components(config)
    attach_prebuilt(components["storage"])
    return components


def attach_prenorm_dot(storage, **backend_kwargs) -> PrenormDotBackend:
    """Swap ``storage._backend`` for a prebuilt ROUND-2 ``PrenormDotBackend``.

    Mirrors ``attach_prebuilt``: shares the same populated ``_entries`` reference (no
    reload), prebuilds + L2-normalizes the cosine buckets, swaps the facade backend.
    ``backend_kwargs`` forward to PrenormDotBackend (cosine_fields / l2_fields /
    rescore_top_k / keep_raw). Asserts the share + prebuild held.
    """
    old = storage._backend
    # Guard against double-attach: if old were already a Prebuilt/Prenorm backend,
    # __dict__.update would carry its containers and clobber the fresh ones (G2 MINOR).
    assert not isinstance(old, PrebuiltMatrixBackend), "attach_prenorm_dot expects a fresh InMemoryBackend"
    new = PrenormDotBackend(old.vector_dims, **backend_kwargs)
    # old (InMemoryBackend) lacks the subclass containers, so __dict__.update cannot
    # clobber _mat/_id2row/_unit_keys/_raw_mat/_prenorm_hits/_cosine_fields/... .
    new.__dict__.update(old.__dict__)
    new._prebuild_task_matrices()
    assert new._entries is old._entries, "entries reference not shared after transplant"
    assert new._prebuilt and new._mat and new._unit_keys, "prenorm prebuild incomplete"
    assert not new._has_active_search_sessions(), "no search session may be active at prebuild"
    storage._backend = new
    assert storage._backend is new
    return new


def build_components_with_prenorm_dot(config, **backend_kwargs) -> dict[str, Any]:
    """Return cache components whose backend is a prebuilt ``PrenormDotBackend``."""
    components = build_cache_components(config)
    attach_prenorm_dot(components["storage"], **backend_kwargs)
    return components


def attach_lean_search(storage, **backend_kwargs) -> LeanSearchBackend:
    """Swap ``storage._backend`` for a prebuilt ROUND-3 ``LeanSearchBackend``.

    Same transplant/prebuild/swap as ``attach_prenorm_dot`` but with the lean
    steady-state search path. ``backend_kwargs`` forward to PrenormDotBackend.
    """
    old = storage._backend
    assert not isinstance(old, PrebuiltMatrixBackend), "attach_lean_search expects a fresh InMemoryBackend"
    new = LeanSearchBackend(old.vector_dims, **backend_kwargs)
    new.__dict__.update(old.__dict__)
    new._prebuild_task_matrices()
    assert new._entries is old._entries, "entries reference not shared after transplant"
    assert new._prebuilt and new._mat and new._unit_keys, "prenorm prebuild incomplete"
    assert not new._has_active_search_sessions(), "no search session may be active at prebuild"
    storage._backend = new
    assert storage._backend is new
    return new


def build_components_with_lean_search(config, **backend_kwargs) -> dict[str, Any]:
    """Return cache components whose backend is a prebuilt ``LeanSearchBackend``."""
    components = build_cache_components(config)
    attach_lean_search(components["storage"], **backend_kwargs)
    return components


def attach_lean_rrf(storage, **backend_kwargs) -> LeanRRFBackend:
    """Swap ``storage._backend`` for a prebuilt ``LeanRRFBackend`` (RRF lean steady-state)."""
    old = storage._backend
    assert not isinstance(old, PrebuiltMatrixBackend), "attach_lean_rrf expects a fresh InMemoryBackend"
    new = LeanRRFBackend(old.vector_dims, **backend_kwargs)
    new.__dict__.update(old.__dict__)
    new._prebuild_task_matrices()
    assert new._entries is old._entries, "entries reference not shared after transplant"
    assert new._prebuilt and new._mat and new._unit_keys, "prenorm prebuild incomplete"
    assert not new._has_active_search_sessions(), "no search session may be active at prebuild"
    storage._backend = new
    assert storage._backend is new
    return new


def attach_rrf_release(storage, **backend_kwargs) -> RrfReleaseVisionBackend:
    """Swap ``storage._backend`` for a prebuilt ``RrfReleaseVisionBackend`` (RRF lean + release)."""
    old = storage._backend
    assert not isinstance(old, PrebuiltMatrixBackend), "attach_rrf_release expects a fresh InMemoryBackend"
    new = RrfReleaseVisionBackend(old.vector_dims, **backend_kwargs)
    new.__dict__.update(old.__dict__)  # brings _is_frozen=True from the pooled backend
    new._prebuild_task_matrices()
    assert new._entries is old._entries, "entries reference not shared after transplant"
    assert new._prebuilt and new._mat and new._unit_keys, "prenorm prebuild incomplete"
    assert not new._has_active_search_sessions(), "no search session may be active at prebuild"
    new._freed_bytes = new.release_vision()
    assert new._vision_released
    storage._backend = new
    assert storage._backend is new
    return new


def attach_release_vision(storage, **backend_kwargs) -> ReleaseVisionBackend:
    """Swap ``storage._backend`` for a prebuilt ROUND-5 ``ReleaseVisionBackend``.

    Same transplant/prebuild as ``attach_lean_search`` (inherits the R1-R4 search path),
    then frees the per-entry vision copies (~692MB) via ``release_vision()``. The pooled
    backend is already frozen, so ``_is_frozen`` transplants True and the release guard's
    freeze assert passes. Records the bytes freed on ``_freed_bytes``.
    """
    old = storage._backend
    assert not isinstance(old, PrebuiltMatrixBackend), "attach_release_vision expects a fresh InMemoryBackend"
    new = ReleaseVisionBackend(old.vector_dims, **backend_kwargs)
    new.__dict__.update(old.__dict__)  # brings _is_frozen=True from the pooled backend
    new._prebuild_task_matrices()
    assert new._entries is old._entries, "entries reference not shared after transplant"
    assert new._prebuilt and new._mat and new._unit_keys, "prenorm prebuild incomplete"
    assert not new._has_active_search_sessions(), "no search session may be active at prebuild"
    new._freed_bytes = new.release_vision()
    assert new._vision_released
    storage._backend = new
    assert storage._backend is new
    return new


def attach_batched_pool_keybuilder(components):
    """Swap ``components["key_builder"]`` for the ROUND-4 batched-avgpool builder.

    Only valid when the stock builder is a ``CP1SpatialPool16KeyBuilder`` (the
    spatial_pool_16 config). Copies the stock builder's ``_enabled`` set so the active
    field selection is preserved — else vision_2/prompt_emb would re-enter and change the
    active-field set (→ verdict flip). The fresh builder starts with an empty ``_cache``
    (collect() runs per step), so no live state is lost.
    """
    old = components["key_builder"]
    assert isinstance(old, CP1SpatialPool16KeyBuilder), (
        f"attach_batched_pool_keybuilder requires a CP1SpatialPool16KeyBuilder "
        f"(spatial_pool_16 config); got {type(old).__name__}"
    )
    enabled = list(old._enabled) if old._enabled is not None else None
    new = CP1SpatialPool16BatchedKeyBuilder(enabled_fields=enabled)
    components["key_builder"] = new
    assert components["key_builder"] is new
    return new
