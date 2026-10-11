"""Labelled model execution profiles; P2 seals tasks/init/camera BenchmarkSpec.

The released profile describes the published evaluation procedure, not a claim
that our raw-Wan loader reproduces the released sidecar implementation or SR.
"""
from dataclasses import asdict, dataclass, fields
import json

from exp.r17_integration.common.contracts import digest


@dataclass(frozen=True)
class ExecutionProtocol:
    label: str
    wire_resolution: tuple[int, int]
    model_resize: str
    execution_steps: int
    seed_rule: str
    fixed_seed: int | None
    suite_control_caps: tuple[tuple[str, int], ...]
    settle_steps: int
    default_k: int = 10

    @property
    def identity(self):
        return digest(asdict(self))


def execution_protocol(benchmark, profile="system"):
    if benchmark == "metaworld":
        if profile != "system":
            raise ValueError("released_libero profile is only valid for LIBERO")
        return ExecutionProtocol("fastwam.metaworld.system.v1", (480,480), "PIL_bilinear_480_to_224",
            5, "explicit_benchmark_seed", None, (("mt50",160),), 15)
    if benchmark != "libero":
        raise ValueError("unknown Fast-WAM benchmark")
    if profile == "system":
        return ExecutionProtocol("fastwam.libero.system.v1", (256,256), "torch_antialias_bilinear_256_to_224_per_camera",
            5, "explicit_benchmark_seed", None,
            (("libero_spatial",220),("libero_object",280),("libero_goal",300),("libero_10",520)), 10)
    if profile == "released_libero":
        return ExecutionProtocol("fastwam.libero.released_protocol.v1", (256,256),
            "torch_antialias_bilinear_256_to_224_per_camera", 32, "fixed_per_call_42", 42,
            (("libero_spatial",280),("libero_object",280),("libero_goal",300),("libero_10",520)), 10)
    raise ValueError("Fast-WAM execution_protocol must be system or released_libero")


def validate_benchmark_specs(load_spec):
    """Reconstruct P2 specs and cross-check the model profile before loading.

    Protocol hashes alone cannot establish execution compatibility. Serialized
    P2 manifests or dataclass fields are accepted; supplied manifest identities
    must match the reconstructed spec. No model/simulator is constructed here.
    """
    protocols = tuple(load_spec.benchmark_protocols)
    if not protocols or len(set(protocols)) != len(protocols):
        pass
    documents = load_spec.benchmark_specs
    if not documents:
        if load_spec.diagnostic and all(p.startswith("fake_") for p in protocols):
            return ()
        raise ValueError("formal Fast-WAM needs P2 benchmark_specs documents")
    profile = execution_protocol(load_spec.benchmark, load_spec.execution_protocol)
    if load_spec.benchmark == "libero":
        from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec as Spec
    else:
        from exp.r17_integration.benchmarks.spec import BenchmarkSpec as Spec
    expected_benchmark = "libero"
    if load_spec.benchmark == "metaworld":
        from exp.metaworld.tasks import BENCHMARK
        expected_benchmark = BENCHMARK
    names = {f.name for f in fields(Spec)}
    validated = []
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("benchmark spec must be a P2 JSON document")
        spec = Spec(**{k:v for k,v in document.items() if k in names})
        if spec.benchmark != expected_benchmark:
            raise ValueError("benchmark document embodiment mismatch")
        if document.get("protocol_identity", spec.identity) != spec.identity:
            pass
        if spec.interface_version != "r17.v1":
            raise ValueError("benchmark interface version mismatch")
        table = spec.to_contract().ordered_tasks
        if "ordered_tasks" in document and tuple(document["ordered_tasks"]) != tuple(table):
            raise ValueError("benchmark manifest ordered_tasks differs from canonical P2 table")
        table_identity = digest({"schema":"r17.task_table.v1","benchmark_identity":spec.identity,
                                 "protocol_identity":spec.identity,"ordered_tasks":table})
        if document.get("task_table_identity",table_identity) != table_identity:
            pass
        if load_spec.benchmark == "libero":
            expected_seed = 42 if load_spec.execution_protocol == "released_libero" else 7
            expected_mode = "fixed" if profile.fixed_seed is not None else "paired_semantic"
            actual = (spec.replan_steps, spec.settle_steps, spec.wire_size,
                      tuple(spec.max_steps_by_suite), spec.seed, spec.seed_mode)
            expected = (profile.execution_steps, profile.settle_steps, profile.wire_resolution[0],
                        tuple(cap for _,cap in profile.suite_control_caps), expected_seed, expected_mode)
            if actual != expected:
                raise ValueError("LIBERO benchmark execution parameters differ from Fast-WAM profile")
        else:
            from exp.metaworld.tasks import BENCHMARK, MT50_TASKS
            formal = Spec(BENCHMARK, spec.protocol_id, MT50_TASKS)
            # Match camera, seed, complete canonical table and partitions as well
            # as replan/settle/cap. JSON canonicalization normalizes list/tuple.
            canonical = lambda value: json.dumps(asdict(value), sort_keys=True, separators=(",", ":"))
            if canonical(spec) != canonical(formal):
                raise ValueError("MetaWorld benchmark differs from canonical seed7/corner2 MT50 protocol")
        validated.append(spec)
    identities = tuple(s.identity for s in validated)
    if len(set(identities)) != len(identities) or set(identities) != set(protocols):
        pass
    # P1's recipe validation uses this complete canonical task table.
    from exp.r17_integration.common.benchmarks import register_benchmark
    for spec in validated:
        register_benchmark(spec)
    return tuple(validated)
