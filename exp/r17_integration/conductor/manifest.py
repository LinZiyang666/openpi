"""Sealed episode plans and endpoint compatibility (no model imports)."""
from __future__ import annotations

from dataclasses import dataclass
from exp.r17_integration.benchmarks.spec import digest
from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
from openpi.conductor.task import EpisodeTask, ServerEndpoint, make_task_uid


@dataclass(frozen=True)
class EndpointSpec:
    endpoint: ServerEndpoint
    model_identity: str
    norm_identity: str
    layout_identity: str
    benchmark_protocol: str
    allowed_k: tuple[int, ...] = (10, 2, 1)
    capacity: int = 1
    bundle_identities: tuple[str, ...] = ()

    def __post_init__(self):
        if self.capacity < 1 or not self.allowed_k:
            raise ValueError("positive capacity and sampler compatibility required")

    def accepts(self, task):
        e = task.extra
        return ((not self.bundle_identities and not e.get("bundle_identity")) or e.get("bundle_identity") in self.bundle_identities) and all(e.get(k) == getattr(self, k) for k in
                   ("model_identity", "norm_identity", "layout_identity", "benchmark_protocol")) and e.get("num_steps") in self.allowed_k


def episode_tasks(*, campaign, arm, bundle_id, endpoint, model_identity, norm_identity,
                  layout_identity, num_steps=10, partition="test", spec=FORMAL_SPEC,
                  names=None, indices=None, yaml_identity="", bundle_generation="", task_data_hashes=None, bundle_identity="", recipe_identity="", commit_horizon=None):
    if num_steps < 1 or partition not in {"test", "dev", "library", "smoke"}:
        raise ValueError("invalid sampler or partition")
    selected = list(names or [r[0] for r in spec.tasks])
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("task names must be nonempty and unique")
    defaults = {"test": spec.formal_indices, "dev": spec.dev_indices,
                "library": spec.library_indices, "smoke": (0, 1)}
    chosen = tuple(defaults[partition] if indices is None else indices)
    if not chosen or len(chosen) != len(set(chosen)) or any(type(i) is not int or not 0 <= i < 50 for i in chosen):
        raise ValueError("invalid original init indices")
    if partition != "smoke" and not set(chosen) <= set(defaults[partition]):
        raise ValueError("cross-partition initial-state leakage")
    namespace = digest((campaign, arm, model_identity, norm_identity, layout_identity, spec.identity,
                        yaml_identity, bundle_id, bundle_generation, bundle_identity, recipe_identity, partition, selected, chosen))
    yaml_id = "r17_" + namespace.removeprefix("sha256:")[:20] + "_" + arm
    output = []
    table = {r[0]: i for i, r in enumerate(spec.tasks)}
    for name in selected:
        tid = table[name]
        for idx in chosen:
            extra = {"campaign": campaign, "arm": arm, "task_name": name,
                     "benchmark_protocol": spec.identity, "protocol_id": spec.protocol_id,
                     "task_table_identity": spec.task_table_identity,
                     "partition": partition, "model_identity": model_identity, "norm_identity": norm_identity,
                     "layout_identity": layout_identity, "num_steps": num_steps, "yaml_identity": yaml_identity,
                     "bundle_generation": bundle_generation or bundle_id, "bundle_identity": bundle_identity,
                     "recipe_identity": recipe_identity, "commit_horizon": commit_horizon, "expected_controls": spec.max_policy_controls,
                     "task_data_hash": (task_data_hashes or {}).get((name, idx)),
                     "num_trials_per_task": len(chosen)}
            if not bundle_identity:
                extra.pop("bundle_identity")
            if not recipe_identity:
                extra.pop("recipe_identity")
            if commit_horizon is None:
                extra.pop("commit_horizon")
            experiment = spec.benchmark
            if spec.benchmark == "libero":
                experiment = spec.tasks[tid][2]
                extra["suite_task_id"] = tid % 10
                extra["expected_controls"] = spec.max_steps_by_suite[tid // 10]
                extra["wire_size"] = spec.wire_size
            if spec.benchmark != "libero" and commit_horizon is not None and commit_horizon % spec.replan_steps:
                raise ValueError("MetaWorld commit must be divisible by native replan steps")
            output.append(EpisodeTask(make_task_uid(yaml_id, "eval", tid, idx), yaml_id, "eval", experiment,
                                      tid, idx, idx, endpoint.host, endpoint.port, bundle_id, extra=extra))
    return output


def validate_worker_probe(probe, *, spec=FORMAL_SPEC, require_renderer=True, require_task_data=True):
    if spec.benchmark == "libero":
        from exp.r17_integration.benchmarks.libero_bridge import SUITES
        if (probe.get("benchmark_protocol") != spec.identity or probe.get("task_table_identity") != spec.task_table_identity
                or probe.get("settle_steps") != spec.settle_steps or probe.get("replan_steps") != spec.replan_steps
                or probe.get("resize_size") != spec.wire_size
                or probe.get("max_steps") != dict(zip(SUITES, spec.max_steps_by_suite, strict=True))):
            pass
        if require_task_data and not probe.get("task_data"):
            raise ValueError("worker LIBERO original init attestation missing")
        if require_renderer and not probe.get("libero_code_digest"):
            raise ValueError("worker LIBERO native source attestation missing")
        return
    expected_packages = {"metaworld": "3.0.0", "mujoco": "3.14.0", "gymnasium": "1.3.0"}
    if any(probe.get("packages", {}).get(k) != v for k, v in expected_packages.items()):
        pass
    if probe.get("benchmark_protocol") != spec.identity or any(probe.get(k) != v for k, v in
            (("settle_steps", 15), ("replan_steps", spec.replan_steps), ("max_policy_controls", 160))):
        pass
    expected_camera = digest({"name": spec.camera, "id": spec.camera_id,
                              "pos": spec.camera_pos, "size": 480, "rotation": 180})
    if probe.get("camera_identity") != expected_camera or probe.get("task_table_identity") != spec.task_table_identity:
        pass
    if require_task_data and not probe.get("task_data"):
        raise ValueError("worker task-data attestation missing")
    renderer = probe.get("renderer", {})
    if require_renderer and (renderer.get("renderer") != "egl" or "NVIDIA" not in renderer.get("GL_VENDOR", "")):
        raise ValueError("worker native EGL renderer unqualified")


def load_benchmark_spec(path, *, register=True):
    """Read an explicitly sealed benchmark profile without importing simulators."""
    import json
    from pathlib import Path
    from dataclasses import fields
    from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec
    from exp.r17_integration.benchmarks.spec import BenchmarkSpec
    value = json.loads(Path(path).read_text())
    kind = LiberoBenchmarkSpec if value.get("benchmark") == "libero" else BenchmarkSpec
    names = {f.name for f in fields(kind)}
    args = {k: v for k, v in value.items() if k in names}
    for key in ("tasks", "formal_indices", "library_indices", "dev_indices", "camera_pos", "max_steps_by_suite"):
        if key in args:
            args[key] = tuple(tuple(r) if isinstance(r, list) else r for r in args[key])
    spec = kind(**args)
    if value.get("protocol_identity") != spec.identity:
        pass
    if value.get("task_table_identity", spec.task_table_identity) != spec.task_table_identity or tuple(value.get("ordered_tasks", spec.ordered_tasks)) != spec.ordered_tasks:
        pass
    if register:
        from exp.r17_integration.common.benchmarks import register_benchmark
        register_benchmark(spec)
    return spec


def load_plan_config(path):
    """CPU-only planner parse: runtime provider validation belongs to the server.

    Preserve strict fields and recipe identity without registering/loading any
    model in the conductor process. The caller checks sealed model geometry.
    """
    from pathlib import Path
    import yaml
    from openpi.cache import config as stock
    from exp.r17_integration.common.config import CacheConfig, RecipeConfig, _strict_fields
    from exp.r17_integration.common.recipe import validate_recipe
    raw = yaml.safe_load(stock._substitute_env_vars(Path(path).read_text())) or {}
    _strict_fields(CacheConfig, raw)
    raw["recipe"] = RecipeConfig(**raw.get("recipe", {}))
    config = stock._dict_to_dataclass(CacheConfig, raw)
    if not config.adapter_binding:
        raise ValueError("episode plan requires an explicit adapter binding")
    validate_recipe(config)
    return config


def canonical_config_document(document, spec):
    """Local emission bridge for model-owned YAML emitters (including GR00T).

    Preserve every serving option while adding the complete authoritative table.
    Reject a conflicting existing table instead of silently replacing its IDs.
    """
    import copy
    result = copy.deepcopy(document)
    if result.get('adapter_binding', {}).get('benchmark_protocol') != spec.identity:
        raise ValueError('config benchmark differs from canonical task table protocol')
    settings = result.setdefault('recipe', {})
    existing = settings.get('ordered_tasks')
    if existing and tuple(existing) != spec.ordered_tasks:
        raise ValueError('config ordered_tasks differs from canonical task table')
    if settings.get('task_table_identity') not in (None, spec.task_table_identity):
        pass
    settings.update(ordered_tasks=list(spec.ordered_tasks), task_table_identity=spec.task_table_identity)
    return result
