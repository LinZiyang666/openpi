"""ReleaseVisionBackend — ROUND 5 memory release (frees duplicated entry vision copies).

After prebuild, the per-entry vision tensors in ``_entries[*].query_keys`` (vision_0 /
vision_1, ~692MB total) are byte-duplicates of the normed matrices already living in
``_mat``. On a write-frozen library the fast paths (LEAN search / prenorm GEMV) read ONLY
``_mat``, never the entry tensors, so the entry copies are dead weight. This subclass
replaces each entry's released-field tensor with a SINGLE shared ``torch.empty(0)``
sentinel — keeping the dict key so ``field in query_keys`` stays True and the fast path is
unchanged — reclaiming ~692MB.

A fail-closed guard makes any fallback path (which WOULD read the entry tensors via the
parent ``torch.stack``) raise ``ReleaseUnsafeError`` instead of reading the empty sentinel,
so a config drift out of the depth_1 fast-path window fails loud rather than silently
mis-scoring.

Sentinel choice (Round 3 workflow实测裁决): a shared ``empty(0)`` is fail-loud + catchable
(``stack`` of it gives a shape error, ``_check_entry_dims`` sees ``shape=(0,)`` and raises),
unlike ``untyped_storage().resize_(0)`` (``.shape`` still reports the old size → silently
passes the dim check, and ``stack`` SIGSEGVs uncatchably) or del-key (silently empties
``valid_indices`` → all-zero scores, no counter).

Safety: release only after freeze + prebuild + no active session (all asserted). robot_state
(0.3MB) is NOT released — the l2 path reads the raw ``_mat`` bucket, and freeing it only adds
coupling for negligible gain.

Dependencies: ``exp.cache_latency_bench.opt.lean_search_backend.LeanSearchBackend``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging

import torch

from exp.cache_latency_bench.opt.lean_search_backend import LeanSearchBackend

logger = logging.getLogger(__name__)


class ReleaseUnsafeError(RuntimeError):
    """Raised when a released-vision backend would fall back to reading freed entry tensors."""


class ReleaseVisionBackend(LeanSearchBackend):
    """LeanSearchBackend that frees the per-entry vision copies after prebuild.

    ROUND 5 scope: adds ``release_vision()`` + a fail-closed guard in
    ``_compute_field_scores``; all search machinery (LEAN / prenorm / gather) is inherited
    and reads only ``_mat``, so the steady-state path is unaffected by the release.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vision_released: bool = False

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def release_vision(self, extra_fields=frozenset({"vision_2", "prompt_emb"})) -> int:
        """Replace per-entry vision tensors with a shared empty(0) sentinel; return bytes freed.

        Releases the cosine fields (already normalized into ``_mat``) plus any disabled
        vision fields still carried on entries. Only valid post-freeze + post-prebuild +
        no active session. robot_state is kept (l2 reads ``_mat`` raw bucket).

        ``release_set`` MUST contain only fields whose data has been migrated into ``_mat``
        (cosine buckets) or that are never indexed (disabled vision_2/prompt_emb). Never add
        a field the fast path still reads from the entry, or the sentinel would be scored.
        """
        assert self.is_frozen, "release_vision requires a frozen backend (write_policy:never)"
        assert self._prebuilt and self._unit_keys, "release_vision requires a completed prebuild"
        assert not self._has_active_search_sessions(), "no search session may be active at release"

        release_set = set(self._cosine_fields) | set(extra_fields)
        sentinel = torch.empty(0, dtype=torch.float32)
        freed = 0
        for entry in self._entries.values():
            for f in list(entry.query_keys.keys()):
                if f in release_set:
                    t = entry.query_keys[f]
                    if t is not sentinel:
                        freed += t.untyped_storage().nbytes()
                        entry.query_keys[f] = sentinel
        self._vision_released = True
        self._trim_malloc()
        return freed

    def _trim_malloc(self) -> None:
        """Best-effort return of freed glibc arena to the OS (RSS). Platform-bound; warns on failure."""
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            libc.malloc_trim(0)
        except Exception as e:  # noqa: BLE001 — diagnostics only, never fatal
            logger.warning("malloc_trim unavailable (%s); RSS may not shrink (tensor-bytes freed regardless)", e)

    # ------------------------------------------------------------------
    # Fail-closed guard
    # ------------------------------------------------------------------

    def _compute_field_scores(self, query_vec, candidates, field_name, sim_cfg):
        """After release, a fallback (which reads entry tensors) must raise, not read the sentinel.

        The fast paths (prenorm cosine / l2) read ``_mat`` and are unaffected; only the
        parent ``super()`` fallback would ``torch.stack`` the entry tensors. We detect that
        case with the SAME prenorm_ok/l2_ok predicate as PrenormDotBackend and raise.
        """
        if self._vision_released:
            valid = [i for i, e in enumerate(candidates) if field_name in e.query_keys]
            if valid:
                sim_type = sim_cfg.get("type", "cosine")
                key = self._fast_path_key(candidates, valid, field_name, sim_type)
                prenorm_ok = (
                    key is not None and sim_type == "cosine"
                    and field_name in self._cosine_fields and key in self._unit_keys
                )
                l2_ok = key is not None and sim_type == "l2" and field_name in self._l2_fields
                if not (prenorm_ok or l2_ok):
                    raise ReleaseUnsafeError(
                        f"vision buffers released; fallback for field {field_name!r} would read "
                        "the empty sentinel — config drifted out of the depth_1 fast-path window"
                    )
        return super()._compute_field_scores(query_vec, candidates, field_name, sim_cfg)
