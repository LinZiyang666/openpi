"""P2 LIBERO benchmark documents accepted by the DP-T adapter.

The coordinator's sealed document is the JSON emitted by
``benchmarks.libero_bridge`` (``LiberoBenchmarkSpec.manifest()``), which carries
attestation keys beyond the dataclass fields. Parse it exactly like P2's own
``conductor.manifest.load_benchmark_spec`` (field filter, list->tuple, identity
and canonical task-table checks), then require DP-T's evaluation profile: E ran
``examples/libero/main.py --replan-steps 5 --seed 7`` with the native 10-step
settle and suite caps 220/280/300/520. The Fast-WAM released reference profile
(fixed seed42/replan32/spatial280) is not a DP-T protocol and fails closed.
"""
from dataclasses import fields

DP_LIBERO_PROFILE = {"settle_steps": 10, "seed": 7, "seed_mode": "paired_semantic",
                     "max_steps_by_suite": (220, 280, 300, 520)}
_TUPLE_FIELDS = ("tasks", "formal_indices", "library_indices", "dev_indices", "max_steps_by_suite")


def libero_spec_from_document(document, *, replan_steps):
    from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec
    if not isinstance(document, dict):
        raise ValueError("DP LIBERO benchmark spec must be a P2 JSON document")
    names = {f.name for f in fields(LiberoBenchmarkSpec)}
    args = {k: v for k, v in document.items() if k in names}
    for key in _TUPLE_FIELDS:
        if key in args:
            args[key] = tuple(tuple(r) if isinstance(r, list) else r for r in args[key])
    spec = LiberoBenchmarkSpec(**args)
    if spec.benchmark != "libero":
        raise ValueError("DP LIBERO benchmark document embodiment mismatch")
    if document.get("protocol_identity", spec.identity) != spec.identity:
        pass
    if document.get("task_table_identity", spec.task_table_identity) != spec.task_table_identity:
        pass
    if "ordered_tasks" in document and tuple(document["ordered_tasks"]) != tuple(spec.ordered_tasks):
        raise ValueError("benchmark document ordered_tasks differs from canonical P2 table")
    expected = dict(DP_LIBERO_PROFILE, replan_steps=replan_steps)
    actual = {key: tuple(getattr(spec, key)) if key == "max_steps_by_suite" else getattr(spec, key)
              for key in expected}
    if actual != expected:
        mismatched = sorted(k for k in expected if actual[k] != expected[k])
        raise ValueError(f"LIBERO benchmark execution parameters differ from DP-T evaluation profile: {mismatched}")
    return spec
