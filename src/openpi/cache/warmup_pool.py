"""Process-wide warmup buffer pool keyed by eval yaml id.

A WarmupPool holds, per ``eval_yaml_id``, a ``dict[factor_key, list[float]]``
of raw factor values previously collected by a sibling
``<eval_yaml_id>__warmup`` run. The verdict_factor_judge runner sets this
from outside the cache system via the ``preload_normalizer_buffer``
WebSocket control message; ``cache.config._load_calibration_samples``
reads it inside ``_build_calibration`` so each new connection's Layer 3
``PercentileRollingCalibration`` pre-fills its rolling buffer at
``bind_keys`` time (no cold-start state — plan §6.3).

Coupling map:
  WRITTEN BY: WebsocketPolicyServer._handle_preload_normalizer_buffer
  READ BY:    cache.config._load_calibration_samples (samples_source=warmup)
  CLEARED BY: WebsocketPolicyServer._handle_unload_warmup_buffer
  MUTATION:   thread-safe (multiple worker connections may build per-conn
              components concurrently while a control thread mutates the pool)
"""

from __future__ import annotations

import threading
from collections import OrderedDict


class WarmupPool:
    """Thread-safe LRU dict[eval_yaml_id, dict[factor_key, list[float]]]."""

    def __init__(self, max_entries: int = 100) -> None:
        self._max_entries = max_entries
        # OrderedDict gives us O(1) LRU semantics via move_to_end / popitem(last=False).
        self._od: OrderedDict[str, dict[str, list[float]]] = OrderedDict()
        # Single lock: writes are rare (one per worker spawn / warmup completion)
        # so contention is negligible and a single lock keeps invariants obvious.
        self._lock = threading.Lock()

    def set(self, eval_yaml_id: str, buffer: dict[str, list[float]]) -> None:
        """Replace the buffer for `eval_yaml_id`. Marks it most-recently-used.

        Evicts the LRU entry when the pool is at capacity.
        """
        # Copy on write so the caller cannot retroactively mutate stored state.
        snapshot = {k: list(v) for k, v in buffer.items()}
        with self._lock:
            self._od[eval_yaml_id] = snapshot
            self._od.move_to_end(eval_yaml_id)
            while len(self._od) > self._max_entries:
                self._od.popitem(last=False)

    def get(self, eval_yaml_id: str) -> dict[str, list[float]] | None:
        """Return a deep copy of the stored buffer or None if absent.

        Marks the entry as most-recently-used on hit. Callers may freely mutate
        the returned dict without contaminating the pool.
        """
        with self._lock:
            stored = self._od.get(eval_yaml_id)
            if stored is None:
                return None
            self._od.move_to_end(eval_yaml_id)
            # Shallow-then-list copy is sufficient: values are list[float].
            return {k: list(v) for k, v in stored.items()}

    def pop(self, eval_yaml_id: str) -> dict[str, list[float]] | None:
        """Remove and return the buffer (deep copy), or None if absent."""
        with self._lock:
            stored = self._od.pop(eval_yaml_id, None)
            if stored is None:
                return None
            return {k: list(v) for k, v in stored.items()}

    def clear(self) -> None:
        """Drop all entries. Called at server start or test teardown."""
        with self._lock:
            self._od.clear()

    def __contains__(self, eval_yaml_id: str) -> bool:
        # Short critical section: no LRU promotion on membership tests.
        with self._lock:
            return eval_yaml_id in self._od

    def __len__(self) -> int:
        with self._lock:
            return len(self._od)


# Module-level singleton — server / config use this directly.
_GLOBAL = WarmupPool()


def get_global_pool() -> WarmupPool:
    """Return the shared process-wide pool."""
    return _GLOBAL
