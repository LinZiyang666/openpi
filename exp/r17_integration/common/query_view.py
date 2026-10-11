"""Layout-independent recipe inputs; no evaluation outcome fields."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contracts import RequestContext


@dataclass
class QueryView:
    request: RequestContext
    features: Mapping[str, Any]
    state: Any
    hist_has_vision: Sequence[bool] = ()
    hist_hit: Sequence[int] = ()
    hist_state: Sequence[Any] = ()
    previous_action: Any = None
    prev_hit: bool | None = None
    executed_count: int = 0

    @property
    def step(self):
        return self.request.decision_seq

    @property
    def task_id(self):
        return self.request.task_id


@dataclass
class RecipeResult:
    action: Any
    rows: Any
    weights: Any
    distances: Any
    force_miss: bool = False
    extras: dict = field(default_factory=dict)
