"""Offline retrieval microbenchmark over one family's ten libraries.

The same 100 queries (ten per task, drawn from the source library with a fixed
seed) are searched against each point's library through the production
retrieval path, in seven random point orders, so per-query latency can be read
against library size without any online queueing in the way. Load time is
recorded separately and excluded. Nothing here says anything about success
rate or end-to-end speed-up; it is the cost side of the pruning curve only.

Usage:
  uv run python -m exp.ablation_study.cache_prune.analysis.benchmark_prune \\
      --source <source.json> --matrix <matrix_<suite>_<regime>.yaml> --out <json>
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage

from ..common import (
    FIELDS,
    SEED,
    context,
    environment,
    read_json,
    require,
    strategy_for_storage,
    template,
)
from ..prune_library import load_raw


def benchmark_prune(source: dict, matrix: dict) -> dict:
    """Measure 100 fixed queries per source over all ten points and seven rounds."""
    torch.set_num_threads(4)
    suite, regime = source["suite"], source["regime"]
    arms = sorted(matrix["arms"], key=lambda a: a["arm"])
    require(len(arms) == 10, "benchmark requires a complete family of ten arms")
    libraries = {
        a["arm"]: yaml.safe_load(Path(a["yaml"]).read_text())["backend"]["in_memory"]["preload_path"]
        for a in arms
    }
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
            started = time.perf_counter()
            backend = InMemoryBackend(config["backend"]["vector_dims"])
            backend.load_artifact(libraries[arm["arm"]])
            backend.freeze()
            strategy = strategy_for_storage(CacheStorage(backend), config)
            loads.append(
                {"round": round_index, "arm": arm["arm"], "seconds": time.perf_counter() - started}
            )
            contexts = [context(query) for query in queries]
            for query in contexts:
                require(len(strategy.search(query)) == 1, "microbenchmark query has no winner")
            for query, entry in zip(contexts, queries):
                started_ns = time.perf_counter_ns()
                results = strategy.search(query)
                elapsed = (time.perf_counter_ns() - started_ns) / 1e6
                samples.append(
                    {
                        "round": round_index,
                        "arm": arm["arm"],
                        "point": arm["arm"][-3:],
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
        summary.append(
            {
                "arm": arm["arm"],
                "point": arm["arm"][-3:],
                "library": libraries[arm["arm"]],
                "samples": len(values),
                "p50_ms": float(np.quantile(values, 0.5)),
                "p95_ms": float(np.quantile(values, 0.95)),
            }
        )
    return {
        "kind": "microbenchmark",
        "suite": suite,
        "regime": regime,
        "source": source["file"]["path"],
        "query_ids": [q.id for q in queries],
        "seed": SEED,
        "round_orders": orders,
        "samples": samples,
        "summary": summary,
        "loads": loads,
        "environment": environment(),
        "interpretation": "library-query retrieval microbenchmark; load time excluded; no SR or end-to-end speedup inference",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="the family's source.json")
    parser.add_argument("--matrix", type=Path, required=True, help="the family's matrix YAML")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark_prune(read_json(args.source), yaml.safe_load(args.matrix.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
