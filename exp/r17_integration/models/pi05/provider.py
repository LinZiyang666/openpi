"""Explicit P1 provider registration. Importing this module registers nothing."""
from exp.r17_integration.common.provider import AdapterProvider, NamedFieldsKeyBuilder
from .adapter import sampler_for_steps


def config_document(provider, *, k=10, commit=5, evidence_dir, recipe=None):
    from exp.r17_integration.common.provider import binding_for
    from exp.r17_integration.common.benchmarks import get_benchmark, task_table_identity
    spec = provider.model_spec
    protocol = spec.allowed_benchmark_protocols[0]
    table = get_benchmark(protocol)
    settings = {"ordered_tasks": list(table.ordered_tasks), "task_table_identity": task_table_identity(protocol)}
    if recipe is not None:
        settings.update(dict(recipe))
        if tuple(settings["ordered_tasks"]) != tuple(table.ordered_tasks) or settings["task_table_identity"] != task_table_identity(protocol):
            raise ValueError("recipe canonical task table identity/order mismatch")
    return {"recipe": settings, "enabled": True, "timer": {"enabled": False},
            "adapter_binding": binding_for(provider, spec.allowed_benchmark_protocols[0], num_steps=k, commit_horizon=commit),
            "keys": {name: {"enabled": True} for name in spec.feature_spec.fields},
            "key_builder": {"type": "adapter_native"},
            "backend": {"type": "in_memory", "vector_dims": dict(spec.feature_spec.fields)},
            "checkpoints": {"cp1": {"enabled": True, "gate": {"type": "always_skip"},
                                      "judge": {"type": "always_hit"}, "search_strategy": {"type": "weighted_rrf_knn"}},
                            "cp3": {"enabled": False, "search_strategy": {"type": "weighted_rrf_knn"}}},
            "write_policy": {"type": "never"}, "miss": {"num_steps": k, "evidence_dir": str(evidence_dir)}}


def recipe_from_store(adapter, directory, *, corrector=True, guard=False, periodic_gap=2):
    """Consume P6's sealed library store and P1's sole layer implementation."""
    import numpy as np
    from exp.r17_integration.common.benchmarks import get_benchmark
    from exp.r17_integration.common.layers import LibraryData, fit_library
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.search.fit import LibraryView
    from exp.r17_integration.search.store import Store
    spec = adapter.metadata()
    protocol = spec.allowed_benchmark_protocols[0]
    tasks = get_benchmark(protocol).ordered_tasks
    task_ids = {name:i for i,name in enumerate(tasks)}
    store = Store(directory, expected_binding=model_binding(spec, protocol))
    view = LibraryView(store)
    rows = store.rows
    unique_eps = {e: i for i, e in enumerate(sorted(set(view.episode_ids)))}
    library = LibraryData({name:view.array("key_"+name) for name in spec.feature_spec.fields if name != "robot_state"}, view.array("state"),
                          view.array("normalized_actions"), np.array([task_ids[r["task_name"]] for r in rows]),
                          np.array([unique_eps[e] for e in view.episode_ids]),
                          np.array([r["decision_seq"] for r in rows]), source_digest=store.manifest["content_digest"])
    return fit_library(library, layout=spec.layout, task_count=len(tasks), commit_horizon=spec.layout.replan_steps,
                       corrector=corrector, guard=guard, periodic_gap=periodic_gap)


def register_provider(adapter, *, provider_id=None):
    from exp.r17_integration.common.provider import register_provider as register
    from .adapter import Pi05MetaWorldAdapter
    provider_id = provider_id or ("r17_pi05_libero_v1" if adapter.spec.model_id == "pi05_libero" else "r17_pi05_mt50_v1")
    provider = AdapterProvider(provider_id, adapter.metadata(),
                               lambda: Pi05MetaWorldAdapter(adapter.policy, adapter.spec, observation_factory=adapter.observation_factory),
                               sampler_for_steps, lambda config: NamedFieldsKeyBuilder(adapter.metadata().feature_spec.fields),
                               trace_factory=lambda a: a.make_trace_adapter())
    register(provider)
    return provider


def make_connection_factory(*args, **kwargs):
    from exp.r17_integration.common.serving import make_connection_factory as factory
    capture_sink = kwargs.pop("capture_sink", None)
    capture_sink_factory = kwargs.pop("capture_sink_factory", None)
    core_factory = factory(*args, capture_sink=None, **kwargs)
    def create(base_policy=None, bundle_id="default"):
        core = core_factory(base_policy, bundle_id)
        sink = capture_sink_factory(core) if capture_sink_factory else capture_sink
        if sink:
            from exp.r17_integration.common.contracts import CaptureSpec
            core.capture = CaptureSpec(enabled=True, snapshots=True)
            core.capture_sink = sink
        return core
    return create


def make_capture_sink_factory(directory):
    """P6 owns native sampler-bound capture construction and writing."""
    from exp.r17_integration.debug.capture_bridge import make_capture_sink_factory as factory
    return factory(directory)
