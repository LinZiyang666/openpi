"""Vision-free method contract. Actions use the library's normalized H x 32 convention.

``blind_age`` counts consecutive blind decisions BEFORE the current request. Histories
are dense, read-only, and exclude this request. No images, tokens, or visual keys are
available on this facade; a method must return LookReason to request vision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class LookReason:
    code: int
    name: str


@dataclass(frozen=True, slots=True)
class BlindResult:
    action: np.ndarray
    rows: np.ndarray
    weights: np.ndarray
    library: str
    extras: dict[str, float]


@dataclass(frozen=True, slots=True)
class BlindQueryView:
    step: int
    task_id: int
    episode: Any
    rs: np.ndarray
    raw_state: np.ndarray
    prev_hit: bool | None
    prev_a_exec: np.ndarray | None
    hist_a_exec: np.ndarray
    hist_hit: np.ndarray
    hist_rs: np.ndarray
    hist_has_vision: np.ndarray
    blind_age: int
    oracle: Any = None  # populated only by the explicit --os-oracle diagnostic channel


def policy_tail_chunk(chunk, offset=5):
    """Shift a saved H-row chunk by offset; repeat its last row to pad back to H.

    Preserves dtype and every column, for normalized history AND original wire
    actions. Only the first five rows may be executed by a policy-tail request.
    The optional method hook ``policy_tail_step(bq)`` returns a BlindResult with
    this exact action, or LookReason. Member rows/weights describe gate provenance
    (the last vision proposal), not the source of the policy action.
    """
    a = np.asarray(chunk)
    if (offset not in (5, 10) or a.ndim != 2 or len(a) < offset + 5
            or not np.isfinite(a).all()):
        raise ValueError("policy tail requires a finite chunk and a complete five-control block at offset 5 or 10")
    return np.concatenate((a[offset:], np.repeat(a[-1:], offset, axis=0)), axis=0)
