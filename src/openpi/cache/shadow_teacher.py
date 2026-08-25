"""ShadowTeacherRecorder — X15 per-decision teacher labels, recorded not executed.

X14's router learned from one episode-level bit spread over ~54 decisions. X15
replaces that with a dense proxy: at every decision, run the teacher once
*without executing it* and record how far the cached chunk it would have
replayed sits from what the teacher would have done. One rollout then yields a
label per decision instead of a label per episode.

Three invariants make the recording safe to run inside a live episode:

* **The executed action never changes.** The recorder is called after the cache
  payload has been fetched and only ever writes to its sidecar.
* **The RNG stream never changes.** The extra flow-matching forward would
  otherwise advance the global torch generator and silently alter every later
  teacher step, so noise is drawn from a recorder-owned, device-matched
  generator seeded by a stable digest of ``(task_uid, attempt, decision_idx)``.
  A stable digest, not ``hash()``: Python randomises string hashing per
  process, which would make the labels unreproducible.
* **A failure degrades to a missing label, never a missing action.** Any
  exception in the shadow path is recorded as an ``error`` row and swallowed.

Row schema (union; one row per decision plus one terminal row per episode)::

    {"task_uid", "attempt", "decision_idx", "status": "ok"|"error"|"finalize",
     "teacher_chunk": list # ok only  - fp16 chunk, so a label can be recomputed
     "u": float           # ok only  - normalised chunk deviation
     "error_type": str    # error only
     "terminal": bool     # finalize only
     "wall_ms": float}    # ok / error

Key dependencies: ``pi0_pytorch.sample_noise`` (the additive ``generator``
seam) and the per-step dump written by the cache interceptor, joined on
``(task_uid, attempt, decision_idx)``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional

import torch

logger = logging.getLogger("openpi.cache.shadow_teacher")


def stable_seed(task_uid: str, attempt: int, decision_idx: int) -> int:
    """Deterministic 64-bit seed for one decision.

    ``hash()`` is randomised per process (PYTHONHASHSEED), so a run could not
    be replayed; a content digest can.
    """
    digest = hashlib.sha256(
        f"{task_uid}|{attempt}|{decision_idx}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _as_chunk(chunk: torch.Tensor, side: str) -> torch.Tensor:
    """Normalise an action chunk to ``[horizon, action_dim]``.

    The two sides arrive in different shapes: a cache payload's chunk is
    unbatched, while ``Stage3Output.action_chunk`` keeps the unit batch dim
    ``[1, AH, AD]`` from ``split_stage1_output``. Subtracting them directly
    broadcasts ``[1, H, D]`` against ``[H, D]`` into ``[H, H, D]`` — comparing
    every step against every other step — so two IDENTICAL chunks score a large
    deviation instead of zero. Measured: 8.0 where the answer is 0.0. Every
    label would have carried that inflation.
    """
    t = chunk.detach().float()
    if t.dim() == 3:
        if t.shape[0] != 1:
            raise ValueError(
                f"shadow_teacher: {side} chunk has batch size {t.shape[0]}, "
                "expected an unbatched chunk or a unit batch"
            )
        t = t[0]
    if t.dim() != 2:
        raise ValueError(
            f"shadow_teacher: {side} chunk must be [H, D] or [1, H, D], "
            f"got shape {tuple(chunk.shape)}"
        )
    return t


def chunk_deviation(
    cache_chunk: torch.Tensor,
    teacher_chunk: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
) -> float:
    """Mean per-step L2 distance between two action chunks, dimension-normalised.

    ``sigma`` (per-dimension action scale from norm-stats) puts joints with
    different units on a comparable footing; without it the largest-range
    dimension would dominate the label.
    """
    a = _as_chunk(cache_chunk, "cache")
    b = _as_chunk(teacher_chunk, "teacher")
    horizon = min(a.shape[0], b.shape[0])
    if horizon == 0:
        raise ValueError("shadow_teacher: empty action chunk")
    diff = a[:horizon] - b[:horizon]
    if sigma is not None:
        diff = diff / sigma.detach().float().clamp_min(1e-6)
    return float(torch.linalg.vector_norm(diff, dim=-1).mean().item())


class ShadowTeacherRecorder:
    """Record per-decision teacher-vs-cache deviation into a JSONL sidecar.

    Disabled construction (``enabled=False``) makes every method a no-op, which
    is what keeps the production path byte-identical when the recorder is not
    wired.
    """

    def __init__(
        self,
        *,
        path: str,
        enabled: bool = True,
        action_sigma: Optional[torch.Tensor] = None,
    ) -> None:
        self.enabled = enabled
        self._path = path
        self._sigma = action_sigma
        self._fh = None
        self._task_uid: Optional[str] = None
        self._attempt: int = 1
        self._finalized: bool = False
        self.error_count = 0
        self.row_count = 0
        if enabled:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # -- lifecycle ----------------------------------------------------

    def begin_episode(self, task_uid: str, attempt: int) -> None:
        if not self.enabled:
            return
        self._task_uid = task_uid
        self._attempt = int(attempt)
        self._finalized = False
        if self._fh is None:
            self._fh = open(self._path, "a", encoding="utf-8")

    def finalize_episode(self, *, terminal: bool = True) -> None:
        """Write the episode's terminal row and flush.

        Called on both normal completion and abort, so a truncated episode is
        distinguishable from one whose rows are still buffered — the label
        joiner needs that difference to decide whether to drop the episode.
        """
        if not self.enabled or self._task_uid is None or self._finalized:
            # Exactly one terminal row per episode. Both the episode-end hook
            # and the connection-close hook call this, and a normal run hits
            # both; two rows would leave the joiner unable to tell whether the
            # episode completed (the second row says terminal=False).
            return
        self._finalized = True
        self._write({"status": "finalize", "decision_idx": -1, "terminal": bool(terminal)})
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # -- recording ----------------------------------------------------

    def record(
        self,
        *,
        decision_idx: int,
        cache_chunk: torch.Tensor,
        teacher_fn,
        device: Any = "cpu",
        noise_shape: Optional[tuple[int, ...]] = None,
    ) -> Optional[float]:
        """Run the shadow teacher once and record the deviation.

        ``teacher_fn(noise=...)`` must return the teacher's action chunk for the
        current observation. Returns the recorded ``u`` or None when the shadow
        pass failed (already recorded as an ``error`` row).
        """
        if not self.enabled or self._task_uid is None:
            return None
        started = time.perf_counter()
        try:
            noise = None
            if noise_shape is not None:
                # Recorder-owned generator on the sampling device: this is what
                # keeps the main trajectory's RNG stream untouched.
                gen = torch.Generator(device=device)
                gen.manual_seed(stable_seed(self._task_uid, self._attempt, decision_idx))
                noise = torch.normal(
                    mean=0.0, std=1.0, size=noise_shape,
                    dtype=torch.float32, device=device, generator=gen,
                )
            teacher_chunk = teacher_fn(noise=noise)
            u = chunk_deviation(cache_chunk, teacher_chunk, self._sigma)
        except Exception as exc:  # noqa: BLE001 - shadow must never break the episode
            self.error_count += 1
            logger.warning(
                "shadow teacher failed at decision %d: %s", decision_idx, exc
            )
            self._write({
                "status": "error",
                "decision_idx": decision_idx,
                "error_type": type(exc).__name__,
                "wall_ms": (time.perf_counter() - started) * 1e3,
            })
            return None
        self._write({
            "status": "ok",
            "decision_idx": decision_idx,
            # The frozen union schema stores the chunk itself so a label can be
            # recomputed under a different deviation metric without re-running
            # the rollout; ``u`` rides along as the metric in force.
            "teacher_chunk": teacher_chunk.detach().to(torch.float16).tolist(),
            "u": u,
            "wall_ms": (time.perf_counter() - started) * 1e3,
        })
        return u

    # -- internals ----------------------------------------------------

    def _write(self, row: dict) -> None:
        if self._fh is None:
            self._fh = open(self._path, "a", encoding="utf-8")
        payload = {
            "task_uid": self._task_uid,
            "attempt": self._attempt,
            **row,
        }
        self._fh.write(json.dumps(payload) + "\n")
        self.row_count += 1
