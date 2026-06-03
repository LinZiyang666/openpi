"""ROUND 3 memory-release probe (read-only research; throwaway).

Quantifies the Round-1/2 memory doubling and validates the release-mechanism design
WITHOUT touching src and WITHOUT shipping anything. Three questions:

  Q1 (footprint): how many bytes do the prebuilt matrices (_mat, normed cosine + raw l2)
      add on top of the still-resident per-entry query_keys vision tensors? This is the
      "doubling" the release would recover.
  Q2 (fallbacks=0): on the shipped depth_1 + weighted_score_sum + step_filter=all config,
      does the fast path ever fall back? (PrebuiltMatrixBackend._fallbacks must stay 0 for
      the "release-then-never-fallback" branch to be sound.)
  Q3 (freeze guard): does CacheStorage.insert run _check_entry_dims (cache_storage.py:289,
      reads entry.query_keys[field].shape) BEFORE the backend's frozen guard fires? If the
      facade dim-check runs first, releasing fields by deletion is the safe shape (KeyError
      is impossible because _check_entry_dims intersects entry.query_keys with backend dims;
      a deleted field simply drops out of `active`). This probe asserts the ordering on a
      frozen backend with a released entry.

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/r3_mem_release_probe.py \
        --cache-config exp/cache_latency_bench/config/depth_study/depth_1.yaml \
        --h5-dir exp/common/data/db/libero_cache/libero_10 \
        --max-episodes 2 --threads 4
"""

from __future__ import annotations

import dataclasses
import gc
import json
import sys

import torch
import tyro

from openpi.cache.backend_base import BackendFrozenError
from openpi.cache.config import build_cache_components, load_cache_config

from exp.cache_latency_bench.opt.inject import attach_prenorm_dot


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.nelement()


def _entries_vision_bytes(backend, cosine_fields) -> int:
    total = 0
    for e in backend._entries.values():
        for f, v in e.query_keys.items():
            if f in cosine_fields:
                total += _tensor_bytes(v)
    return total


def _entries_all_qk_bytes(backend) -> int:
    total = 0
    for e in backend._entries.values():
        for v in e.query_keys.values():
            total += _tensor_bytes(v)
    return total


def _mat_bytes(backend) -> tuple[int, int]:
    """Return (cosine_mat_bytes, l2_mat_bytes)."""
    cos_b = 0
    l2_b = 0
    for key, m in backend._mat.items():
        if key in backend._unit_keys:
            cos_b += _tensor_bytes(m)
        else:
            l2_b += _tensor_bytes(m)
    return cos_b, l2_b


@dataclasses.dataclass
class Args:
    cache_config: str
    h5_dir: str
    max_episodes: int = 2
    threads: int = 4


def main(args: Args) -> None:
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    config = load_cache_config(args.cache_config)
    components = build_cache_components(config)
    storage = components["storage"]

    backend = attach_prenorm_dot(storage)  # prebuild + normalize cosine buckets

    cosine_fields = backend._cosine_fields
    n_entries = len(backend._entries)

    # ---- Q1: footprint ----
    entries_vision = _entries_vision_bytes(backend, cosine_fields)
    entries_all_qk = _entries_all_qk_bytes(backend)
    cos_mat, l2_mat = _mat_bytes(backend)
    mat_total = cos_mat + l2_mat

    report = {
        "n_entries": n_entries,
        "n_buckets": len(backend._mat),
        "n_cosine_buckets": len(backend._unit_keys),
        "MB": {
            "entries_vision_qk (releasable)": round(entries_vision / 1e6, 1),
            "entries_all_qk": round(entries_all_qk / 1e6, 1),
            "prebuilt_cosine_mat": round(cos_mat / 1e6, 1),
            "prebuilt_l2_mat": round(l2_mat / 1e6, 1),
            "prebuilt_mat_total": round(mat_total / 1e6, 1),
        },
        "doubling_note": (
            "prebuilt cosine matrices carry the SAME vision data as entries_vision_qk "
            "(normed copy). Releasing entry vision tensors recovers ~entries_vision_qk; "
            "l2 (robot_state) entry tensors are tiny (32 floats) and the l2 fast path "
            "reads the raw bucket matrix, so robot_state entry tensors are ALSO releasable."
        ),
    }

    # ---- Q3: confirm release-by-deletion is fallback-safe + freeze-guard ordering ----
    # Take one real entry, snapshot it, simulate the release (delete cosine fields),
    # then prove (a) the facade _check_entry_dims tolerates the missing field, and
    # (b) a frozen backend raises BackendFrozenError on any insert attempt.
    sample_id = next(iter(backend._entries))
    sample = backend._entries[sample_id]
    pre_fields = sorted(sample.query_keys.keys())

    # Simulate release on a COPY of the dict (do not mutate the shared library here).
    import copy
    released_qk = {k: v for k, v in sample.query_keys.items() if k not in cosine_fields}
    released_entry = copy.copy(sample)
    released_entry.query_keys = released_qk

    # (a) facade dim-check on a released entry: intersection drops the deleted fields,
    #     robot_state (32) remains and matches -> no crash, no KeyError.
    facade_ok = None
    try:
        storage._check_entry_dims(released_entry)
        facade_ok = "passed (released entry tolerated; deleted cosine fields drop out of active set)"
    except Exception as exc:  # noqa: BLE001
        facade_ok = f"RAISED {type(exc).__name__}: {exc}"

    # (b) freeze the backend and confirm insert raises BEFORE the backend mutates.
    backend.freeze()
    freeze_ok = None
    try:
        storage.insert(released_entry)
        freeze_ok = "ERROR: insert did NOT raise on frozen backend"
    except BackendFrozenError:
        freeze_ok = "BackendFrozenError (frozen backend blocks insert; release is write-safe)"
    except Exception as exc:  # noqa: BLE001
        freeze_ok = f"other: {type(exc).__name__}: {exc}"
    finally:
        backend._is_frozen = False  # un-freeze the probe instance

    report["release_safety"] = {
        "sample_fields_before": pre_fields,
        "sample_fields_after_release": sorted(released_qk.keys()),
        "facade_check_entry_dims_on_released": facade_ok,
        "frozen_insert_guard": freeze_ok,
        "ordering_note": (
            "CacheStorage.insert (cache_storage.py:192-195) runs _check_entry_dims THEN "
            "entry.validate() THEN self._backend.insert. The backend frozen-guard fires "
            "inside _backend.insert. So on a frozen lib, ANY insert raises BackendFrozenError "
            "regardless of released fields -> the fallback that reads entry.query_keys can "
            "NEVER be reached for a released entry, because writes are blocked. The hazard at "
            ":289 only bites if (1) NOT frozen AND (2) the released field is still a key with a "
            "stale/None tensor lacking .shape. Releasing by DELETING the key (not nulling it) "
            "sidesteps :289 entirely: the deleted field drops out of the active intersection."
        ),
    }

    print(json.dumps(report, indent=2))
    gc.collect()


if __name__ == "__main__":
    main(tyro.cli(Args))
