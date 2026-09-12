"""Measure genuine top-1 retrieval using a source-frozen stratified query bank.

Only one library is resident at a time. Each newly loaded point is warmed before
timing; seven seeded randomized rounds separate loading cost from per-call search
latency. Library queries are a microbenchmark, never evidence of closed-loop SR.
"""

from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path
import time

import numpy as np
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage

from ..common import (
    FIELDS,
    REGIMES,
    SEED,
    SUITES,
    VERSION,
    check_identity,
    check_seal,
    context,
    environment,
    new_directory,
    read_json,
    require,
    seal,
    strategy_for_storage,
    template,
    write_json,
)
from ..emit_prune_arms import validate_freeze
from ..prune_library import load_raw


def benchmark_prune(freeze: dict, suite: str, regime: str) -> dict:
    """Measure 100 fixed queries per source over all ten points and seven rounds."""
    validate_freeze(freeze)
    torch.set_num_threads(4)
    arms = sorted(
        (a for a in freeze["arms"] if (a["suite"], a["regime"]) == (suite, regime)),
        key=lambda a: a["point"],
    )
    require(len(arms) == 10, "benchmark requires a complete source curve")
    source = next(
        s for s in freeze["sources"] if (s["suite"], s["regime"]) == (suite, regime)
    )
    raw = load_raw(source["file"]["path"])
    rng = np.random.default_rng(SEED)
    queries = []
    for task in sorted(source["task_counts"]):
        candidates = [e for e in raw["entries"] if e.payload.task_key == task]
        require(len(candidates) >= 10, "source lacks ten queries for a task")
        for index in rng.choice(len(candidates), size=10, replace=False):
            entry = copy.copy(candidates[index])
            entry.query_keys = {
                k: torch.as_tensor(entry.query_keys[k]).clone() for k in FIELDS
            }
            entry.payload = copy.copy(entry.payload)
            entry.payload.action_chunk = None
            entry.payload.intermediates = None
            queries.append(entry)
    del raw, candidates
    gc.collect()
    samples, loads = [], []
    orders = [rng.permutation(10).tolist() for _ in range(7)]
    for round_index, order in enumerate(orders):
        for point_index in order:
            arm = arms[point_index]
            config = template(suite)
            check_identity(arm["artifact"]["file"])
            started = time.perf_counter()
            backend = InMemoryBackend(config["backend"]["vector_dims"])
            backend.load_artifact(arm["artifact"]["file"]["path"])
            backend.freeze()
            strategy = strategy_for_storage(CacheStorage(backend), config)
            loads.append(
                {
                    "round": round_index,
                    "point": arm["point"],
                    "seconds": time.perf_counter() - started,
                }
            )
            contexts = [context(query) for query in queries]
            for query in contexts:
                require(
                    len(strategy.search(query)) == 1,
                    "microbenchmark query has no winner",
                )
            for query, entry in zip(contexts, queries):
                started_ns = time.perf_counter_ns()
                results = strategy.search(query)
                elapsed = (time.perf_counter_ns() - started_ns) / 1e6
                require(
                    len(results) == 1
                    and results[0].id in arm["artifact"]["selection"]["kept_ids"],
                    "microbenchmark returned an invalid winner",
                )
                samples.append(
                    {
                        "round": round_index,
                        "point": arm["point"],
                        "arm": arm["arm"],
                        "query_id": entry.id,
                        "task_key": entry.payload.task_key,
                        "milliseconds": elapsed,
                        "winner_id": results[0].id,
                    }
                )
            del strategy, backend, contexts
            gc.collect()
    summary = []
    for arm in arms:
        values = [r["milliseconds"] for r in samples if r["arm"] == arm["arm"]]
        require(len(values) == 700, "microbenchmark sample coverage mismatch")
        summary.append(
            {
                "arm": arm["arm"],
                "point": arm["point"],
                "samples": len(values),
                "p50_ms": float(np.quantile(values, 0.5)),
                "p95_ms": float(np.quantile(values, 0.95)),
            }
        )
    check_identity(source["file"])
    result = seal(
        {
            "version": VERSION,
            "kind": "microbenchmark",
            "freeze_digest": freeze["digest"],
            "suite": suite,
            "regime": regime,
            "source_digest": source["digest"],
            "query_ids": [q.id for q in queries],
            "seed": SEED,
            "round_orders": orders,
            "samples": samples,
            "summary": summary,
            "loads": loads,
            "environment": environment(),
            "interpretation": "library-query retrieval microbenchmark; load time excluded; no SR or end-to-end speedup inference",
        }
    )
    validate_benchmark(result, freeze)
    return result


def validate_benchmark(result: dict, freeze: dict) -> None:
    """Verify the fixed query bank, seven permutations and all 7,000 raw samples."""
    check_seal(result)
    require(
        result["freeze_digest"] == freeze["digest"]
        and result["seed"] == SEED
        and result["environment"]["cpu_threads"] == 4,
        "microbenchmark protocol differs",
    )
    source = next(
        s
        for s in freeze["sources"]
        if (s["suite"], s["regime"]) == (result["suite"], result["regime"])
    )
    require(
        result["source_digest"] == source["digest"], "microbenchmark source mismatch"
    )
    rng = np.random.default_rng(SEED)
    query_ids = []
    for task in sorted(source["task_counts"]):
        rows = [r for r in source["rows"] if r["task_key"] == task]
        query_ids.extend(
            rows[i]["id"] for i in rng.choice(len(rows), 10, replace=False)
        )
    orders = [rng.permutation(10).tolist() for _ in range(7)]
    require(
        result["query_ids"] == query_ids and result["round_orders"] == orders,
        "query bank or round order changed",
    )
    expected = [
        (round_index, f"P{point:02d}", query)
        for round_index, order in enumerate(orders)
        for point in order
        for query in query_ids
    ]
    require(
        [(r["round"], r["point"], r["query_id"]) for r in result["samples"]]
        == expected,
        "latency samples are missing, duplicated or reordered",
    )
    arms = {
        a["point"]: a
        for a in freeze["arms"]
        if (a["suite"], a["regime"]) == (result["suite"], result["regime"])
    }
    rows_by_id = {r["id"]: r for r in source["rows"]}
    for row in result["samples"]:
        arm = arms[row["point"]]
        require(
            row["arm"] == arm["arm"]
            and row["winner_id"] in arm["artifact"]["selection"]["kept_ids"],
            "latency sample arm/winner mismatch",
        )
        require(
            row["task_key"]
            == rows_by_id[row["query_id"]]["task_key"]
            == rows_by_id[row["winner_id"]]["task_key"],
            "cross-task latency query/winner",
        )
        require(
            np.isfinite(row["milliseconds"]) and row["milliseconds"] >= 0,
            "invalid latency measurement",
        )
    require(len(result["summary"]) == 10, "incomplete latency summary")
    for point, summary in zip(sorted(arms), result["summary"]):
        values = [r["milliseconds"] for r in result["samples"] if r["point"] == point]
        require(
            summary
            == {
                "arm": arms[point]["arm"],
                "point": point,
                "samples": 700,
                "p50_ms": float(np.quantile(values, 0.5)),
                "p95_ms": float(np.quantile(values, 0.95)),
            },
            "latency summary differs from raw samples",
        )


def main() -> None:
    """Write raw retrieval timings and percentiles for one frozen source curve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark_prune(read_json(args.freeze), args.suite, args.regime)
    with new_directory(args.out) as temporary:
        write_json(temporary / "latency.json", result)


if __name__ == "__main__":
    main()
