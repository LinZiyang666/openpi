"""Tests for `openpi.cache.warmup_pool.WarmupPool` (B2.2).

Covers the public contract specified in the verdict_factor_judge plan:
deep-copy semantics on get/pop, LRU eviction + promotion, replace-in-place,
clear, the process-wide singleton accessor, and a small concurrency stress
to surface obvious lock-misuse / torn-write bugs.

Each test instantiates its own `WarmupPool` so the module-level singleton
returned by `get_global_pool()` stays untouched across the suite.
"""

from __future__ import annotations

import threading

from openpi.cache.warmup_pool import WarmupPool, get_global_pool


def test_set_then_get_returns_copy() -> None:
    pool = WarmupPool()
    pool.set("yaml_a", {"factor_x": [1.0, 2.0, 3.0]})

    first = pool.get("yaml_a")
    assert first == {"factor_x": [1.0, 2.0, 3.0]}

    # Mutate the returned dict; the pool should be unaffected.
    first["factor_x"].append(999.0)
    first["factor_y"] = [42.0]

    second = pool.get("yaml_a")
    assert second == {"factor_x": [1.0, 2.0, 3.0]}


def test_set_then_pop_then_get_returns_none() -> None:
    pool = WarmupPool()
    pool.set("yaml_a", {"factor_x": [1.0]})

    popped = pool.pop("yaml_a")
    assert popped == {"factor_x": [1.0]}

    assert pool.get("yaml_a") is None
    assert "yaml_a" not in pool
    assert len(pool) == 0


def test_lru_evicts_oldest_when_capacity_exceeded() -> None:
    pool = WarmupPool(max_entries=3)
    pool.set("A", {"f": [1.0]})
    pool.set("B", {"f": [2.0]})
    pool.set("C", {"f": [3.0]})
    pool.set("D", {"f": [4.0]})

    assert "A" not in pool
    assert "B" in pool
    assert "C" in pool
    assert "D" in pool
    assert len(pool) == 3


def test_get_promotes_to_most_recently_used() -> None:
    pool = WarmupPool(max_entries=3)
    pool.set("A", {"f": [1.0]})
    pool.set("B", {"f": [2.0]})
    pool.set("C", {"f": [3.0]})

    # Touch A so it is no longer the LRU; B becomes the eviction target.
    assert pool.get("A") is not None

    pool.set("D", {"f": [4.0]})

    assert "A" in pool
    assert "B" not in pool
    assert "C" in pool
    assert "D" in pool


def test_set_replaces_existing_value() -> None:
    pool = WarmupPool()
    pool.set("k", {"x": [1.0]})
    pool.set("k", {"x": [2.0]})

    assert pool.get("k") == {"x": [2.0]}
    assert len(pool) == 1


def test_clear_drops_everything() -> None:
    pool = WarmupPool()
    pool.set("a", {"f": [1.0]})
    pool.set("b", {"f": [2.0]})
    pool.set("c", {"f": [3.0]})
    assert len(pool) == 3

    pool.clear()

    assert len(pool) == 0
    assert pool.get("a") is None
    assert "b" not in pool


def test_concurrent_set_get_no_corruption() -> None:
    pool = WarmupPool(max_entries=16)
    n_threads = 8
    iterations = 200
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def writer(tid: int) -> None:
        try:
            for i in range(iterations):
                key = f"yaml_{i % 4}"
                # Distinct value per write so torn reads would surface as
                # missing keys / wrong types in the reader's assertions.
                pool.set(key, {"a": [float(tid * 10_000 + i)]})
        except BaseException as exc:  # pragma: no cover - reported via errors list
            with errors_lock:
                errors.append(exc)

    def reader(tid: int) -> None:
        try:
            for i in range(iterations):
                key = f"yaml_{i % 4}"
                got = pool.get(key)
                if got is None:
                    continue
                # Contract: get returns dict[str, list[float]] with "a" present.
                assert isinstance(got, dict)
                assert "a" in got
                assert isinstance(got["a"], list)
                for v in got["a"]:
                    assert isinstance(v, float)
        except BaseException as exc:  # pragma: no cover - reported via errors list
            with errors_lock:
                errors.append(exc)

    threads: list[threading.Thread] = []
    for tid in range(n_threads // 2):
        threads.append(threading.Thread(target=writer, args=(tid,)))
        threads.append(threading.Thread(target=reader, args=(tid,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_get_global_pool_returns_singleton() -> None:
    assert get_global_pool() is get_global_pool()


def test_pop_returns_copy() -> None:
    pool = WarmupPool()
    pool.set("k", {"x": [1.0, 2.0]})

    popped = pool.pop("k")
    assert popped == {"x": [1.0, 2.0]}

    # Mutating the popped dict must not affect a future, independent set+get.
    popped["x"].append(999.0)
    popped["y"] = [42.0]

    pool.set("k", {"x": [10.0, 20.0]})
    assert pool.get("k") == {"x": [10.0, 20.0]}
