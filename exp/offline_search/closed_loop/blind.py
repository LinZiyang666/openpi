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
