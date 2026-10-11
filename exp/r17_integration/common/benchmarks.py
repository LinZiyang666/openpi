"""Explicit CPU-only benchmark identity and canonical task-table registry."""
import threading
from dataclasses import replace
from .contracts import BenchmarkSpec, digest

_specs = {}
_lock = threading.RLock()


def register_benchmark(spec, *, protocol_identity=None):
    if hasattr(spec, "to_contract"):
        spec = spec.to_contract()
    if not isinstance(spec, BenchmarkSpec):
        raise ValueError("register_benchmark requires the v1 BenchmarkSpec")
    spec = replace(spec, ordered_tasks=tuple(spec.ordered_tasks), prompts=tuple(spec.prompts))
    key = protocol_identity or spec.identity
    if key != spec.identity and not key.startswith("fake_"):
        pass
    with _lock:
        if key in _specs and _specs[key].identity != spec.identity:
            pass
        _specs[key] = spec
    return task_table_identity(key)


def get_benchmark(protocol_identity):
    with _lock:
        spec = _specs.get(protocol_identity)
    if spec is None and not protocol_identity.startswith("fake_"):
        # The canonical MW descriptor is CPU-only; no simulator/model import.
        from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
        if protocol_identity == FORMAL_SPEC.identity:
            register_benchmark(FORMAL_SPEC)
            spec = _specs[protocol_identity]
    if spec is None:
        raise ValueError("unregistered canonical benchmark task table: " + protocol_identity)
    return spec


def task_table_identity(protocol_identity):
    spec = get_benchmark(protocol_identity)
    return digest({"schema": "r17.task_table.v1", "benchmark_identity": spec.identity,
                   "protocol_identity": protocol_identity, "ordered_tasks": spec.ordered_tasks})


def validate_task_table(config):
    recipe = config.recipe
    protocol = config.adapter_binding["benchmark_protocol"]
    spec = get_benchmark(protocol)
    if tuple(recipe.ordered_tasks) != tuple(spec.ordered_tasks):
        raise ValueError("recipe ordered_tasks differs from canonical benchmark task table")
    if recipe.task_table_identity != task_table_identity(protocol):
        pass
    return spec
