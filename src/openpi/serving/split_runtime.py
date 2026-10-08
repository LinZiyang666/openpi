"""Install split adapters only in explicitly launched child interpreters.

The ordinary serving modules are unmodified. A front loads only an attested
CPU blueprint; its original factories, plugin, lifecycle and websocket handler
still own every connection. The GPU child runs the original model loader, then
serves local stage RPC instead of binding a websocket port.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
from types import FunctionType

import cloudpickle
import torch

from openpi.serving.split_service import (
    RemoteCoordinator, RemoteGrootRunner, RemoteKeyBuilder, RemotePiModel,
    StageClient, StageService, make_groot_coordinator,
)


def _assert_cpu_blueprint(obj, seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        if obj.device.type != "cpu" or isinstance(obj, torch.nn.Parameter):
            raise ValueError("split blueprint contains model parameters or device tensors")
    elif isinstance(obj, torch.nn.Module):
        if any(True for _ in obj.parameters()):
            raise ValueError("split blueprint contains model weights")
    elif isinstance(obj, FunctionType):
        for cell in obj.__closure__ or ():
            _assert_cpu_blueprint(cell.cell_contents, seen)
        _assert_cpu_blueprint(obj.__defaults__, seen)
        _assert_cpu_blueprint(obj.__kwdefaults__, seen)
    elif isinstance(obj, dict):
        for value in obj.values():
            _assert_cpu_blueprint(value, seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _assert_cpu_blueprint(value, seen)
    elif hasattr(obj, "__dict__") and not isinstance(obj, (type, SimpleNamespace)):
        for value in vars(obj).values():
            _assert_cpu_blueprint(value, seen)
    elif isinstance(obj, SimpleNamespace):
        for value in vars(obj).values():
            _assert_cpu_blueprint(value, seen)


def export_blueprint(policy, model_kind, metadata):
    if model_kind == "pi05":
        state = {name: getattr(policy, name) for name in (
            "_input_transform", "_output_transform", "_sample_kwargs", "_metadata",
        )}
        config = policy._model.config
    else:
        state = {k: v for k, v in vars(policy).items() if k != "model"}
        state["device"] = "cpu"
        head = policy.model.action_head
        config = dict(num_inference_timesteps=head.num_inference_timesteps,
                      dtype=head.dtype if hasattr(head, "dtype") else next(head.parameters()).dtype,
                      action_horizon=head.config.action_horizon,
                      action_dim=head.config.action_dim)
    blueprint = dict(kind=model_kind, state=state, config=config, metadata=metadata)
    _assert_cpu_blueprint(blueprint)
    return cloudpickle.dumps(blueprint)


def install_gpu(model_kind, endpoint, blueprint_path, *, batch_size=1, wait_ms=0,
                max_channels=128, capacity=32 << 20):
    from openpi.serving import websocket_policy_server as wps
    isolate_plugin_logs("gpu")

    def serve_gpu(server, **kwargs):
        policy = server._policy
        metadata = dict(server._metadata, split=True, split_role="front", model_processes=1)
        if model_kind == "pi05":
            from openpi.serving.batching_core import get_active_coordinator
            coordinator = get_active_coordinator()
            if coordinator is None:
                raise ValueError("split requires concurrent Pi0.5 serving")
            service = StageService(policy._model, coordinator=coordinator, metadata=metadata,
                                   max_channels=max_channels)
        else:
            from openpi.cache.groot.staged import GrootStagedRunner
            runner = GrootStagedRunner(policy.model)
            coordinator = make_groot_coordinator(runner, max_batch_size=batch_size, max_wait_ms=wait_ms)
            service = StageService(policy.model, runner=runner, coordinator=coordinator,
                                   metadata=metadata, max_channels=max_channels)
        path = Path(blueprint_path)
        temp = path.with_suffix(".pending")
        temp.write_bytes(export_blueprint(policy, model_kind, metadata))
        temp.replace(path)
        service.listen(endpoint)

    wps.WebsocketPolicyServer.serve_forever = serve_gpu


def install_front(model_kind, endpoint, blueprint_path, *, capacity=32 << 20, timeout=60):
    from openpi.serving import websocket_policy_server as wps
    from openpi.cache import config as cc
    isolate_plugin_logs("front_" + os.environ.get("OPENPI_SPLIT_FRONT_ID", "0"))
    client = StageClient(endpoint, capacity=capacity, timeout=timeout)
    blueprint = cloudpickle.loads(Path(blueprint_path).read_bytes())
    _assert_cpu_blueprint(blueprint)
    if blueprint["kind"] != model_kind:
        raise ValueError("split blueprint model mismatch")
    state = blueprint["state"]

    if model_kind == "pi05":
        from openpi.policies import policy, policy_config
        from openpi.serving import batching_coordinator
        def load_front(*args, **kwargs):
            front = object.__new__(policy.Policy)
            vars(front).update(state)
            front._is_pytorch_model = True
            front._pytorch_device = "cpu"
            front._model = RemotePiModel(client, blueprint["config"])
            return front
        policy_config.create_trained_policy = load_front
        batching_coordinator.BatchingCoordinator = lambda *args, **kwargs: RemoteCoordinator(client)
    else:
        from gr00t.model.policy import Gr00tPolicy
        from openpi.cache.groot import staged
        def load_front(front, *args, **kwargs):
            vars(front).update(state)
            config = blueprint["config"]
            front.model = SimpleNamespace(
                device="cpu", action_head=SimpleNamespace(
                    num_inference_timesteps=config["num_inference_timesteps"], dtype=config["dtype"],
                    config=SimpleNamespace(action_horizon=config["action_horizon"], action_dim=config["action_dim"]),
                ),
            )
        Gr00tPolicy.__init__ = load_front
        def get_normalized(front, normalized):
            runner = RemoteGrootRunner(client, front.model)
            return runner.run_stage2(runner.run_stage1(normalized)).action_pred.float()
        Gr00tPolicy._get_action_from_normalized_input = get_normalized
        staged.GrootStagedRunner = lambda model, **kwargs: RemoteGrootRunner(client, model, **kwargs)
        RemoteCoordinator(client).start()

    original_build = cc.build_per_connection_components
    def components(*args, **kwargs):
        result = original_build(*args, **kwargs)
        config = args[0]
        validate_split_config(config)
        result["key_builder"] = RemoteKeyBuilder(result["key_builder"], client)
        return result
    cc.build_per_connection_components = components
    original_load = cc.load_cache_config
    def load_config(*args, **kwargs):
        config = original_load(*args, **kwargs)
        validate_split_config(config)
        return config
    cc.load_cache_config = load_config

    original_init = wps.WebsocketPolicyServer.__init__
    def server_init(server, *args, **kwargs):
        if not kwargs.get("concurrent"):
            raise ValueError("split requires concurrent serving")
        kwargs["host"] = "127.0.0.1"
        kwargs["metadata"] = blueprint["metadata"]
        original_init(server, *args, **kwargs)
        factory = server._connection_policy_factory
        def connection_factory(*args, **kwargs):
            inner = factory(*args, **kwargs)
            return DecisionPolicy(inner, client)
        server._connection_policy_factory = connection_factory
    wps.WebsocketPolicyServer.__init__ = server_init
    return client


class DecisionPolicy:
    def __init__(self, inner, client):
        self.inner, self.client = inner, client
        if hasattr(inner, "prefill_trajectory"):
            self.prefill_trajectory = self._prefill

    def infer(self, *args, **kwargs):
        with self.client.decision():
            return self.inner.infer(*args, **kwargs)

    def _prefill(self, *args, **kwargs):
        with self.client.decision():
            return self.inner.prefill_trajectory(*args, **kwargs)

    def on_task_end(self):
        try:
            if hasattr(self.inner, "on_task_end"):
                self.inner.on_task_end()
        finally:
            current, seen = self.inner, set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                fields = vars(current)
                orchestrator = fields.get("_orchestrator")
                key = getattr(orchestrator, "key_builder", None)
                if isinstance(key, RemoteKeyBuilder):
                    key.close()
                current = next((fields[k] for k in ("_osp_inner", "_inner", "_policy") if k in fields), None)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def validate_split_config(config):
    """Refuse paths requiring device-resident per-connection mutable state.

    Current scope: write-frozen CP1 cache recipes, stateless pooling keys,
    offline_search methods (including correctors and blind/policy tails),
    FULL_HIT/MISS and ordinary WARM_START. Advanced GPU-dependent recipes
    retain their existing single-process entry points.
    """
    if config.write_policy.type != "never":
        raise ValueError("split requires write_policy: never")
    allowed = {"cp1_mean_pool", "cp1_max_pool", "cp1_spatial_pool_4", "cp1_spatial_pool_16",
               "cp1_spatial_pool_64", "groot_cp1_mean_pool", "groot_cp1_max_pool",
               "groot_cp1_spatial_pool_4", "groot_cp1_spatial_pool_16",
               "groot_libero_mean_pool", "groot_libero_max_pool", "groot_libero_spatial_pool_16",
               "groot_libero_spatial_pool_4"}
    allowed.update({"cp1_groot_mean_pool", "cp1_groot_max_pool", "cp1_groot_spatial_pool_16",
                    "cp1_groot_libero_mean_pool", "cp1_groot_libero_spatial_pool_16"})
    if config.key_builder.type not in allowed:
        raise ValueError(f"split: key builder {config.key_builder.type!r} has no equivalence proof")
    if any(name == "cp2" and cp.enabled for name, cp in config.checkpoints.items()):
        raise ValueError("split: CP2 recipes require a separate equivalence proof")
    if any(cp.enabled and cp.judge.type == "online_rit" for cp in config.checkpoints.values()):
        raise ValueError("split: online_rit feedback requires a separate equivalence proof")
    if getattr(config, "warm_reset", None) or getattr(getattr(config, "trace", None), "enabled", False):
        raise ValueError("split: trace/warm_reset recipes require a separate equivalence proof")


def validate_arguments(model_kind, argv):
    refused = ("--trace-out", "--trace-build-cache", "--rit-shadow-out", "--loto-log-out",
               "--os-gpu-retrieval", "--os-request-cameras", "--os-lookhalf", "--os-cache-ablate",
               "--compile-stage1", "--stage1-only", "--os-light-key", "--non-concurrent",
               "--stage1-device", "--stage2-device", "--stage3-device", "--replicas")
    normalized = [arg.replace("_", "-") if arg.startswith("--") else arg for arg in argv]
    for arg in normalized:
        if any(arg == flag or arg.startswith(flag + "=") for flag in refused):
            raise ValueError(f"split: unsupported option {arg}; use the ordinary serving path")
    if "--os-pack-prefix" in normalized:
        raise ValueError("split: prefix packing requires a separate equivalence proof")
    for i, arg in enumerate(normalized):
        if arg == "--os-stage1-mode" or arg.startswith("--os-stage1-mode="):
            mode = arg.split("=", 1)[1] if "=" in arg else normalized[i+1]
            if mode != "full":
                raise ValueError("split: stage1 overrides require a separate equivalence proof")
    if not any(arg == "--cache-config" or arg.startswith("--cache-config=") for arg in normalized):
        raise ValueError("split requires --cache-config (the staged connection factory)")


def isolate_plugin_logs(role):
    from exp.offline_search.closed_loop import plugin
    original = plugin.parse_cli
    def parse(*args, **kwargs):
        opts, rest = original(*args, **kwargs)
        # Keep tags (synthetic winner identities) stable across fronts. Files
        # live in distinct directories, including every resident registry arm.
        opts.os_tag = opts.os_tag or "split"
        root = os.environ.get("OPENPI_SPLIT_LOG_ROOT")
        if root:
            opts.os_log_dir = str(Path(root) / role / opts.os_tag)
        return opts, rest
    plugin.parse_cli = parse
    state_root = os.environ.get("OPENPI_SPLIT_STATE_ROOT")
    if role.startswith("front_") and state_root:
        from openpi.serving.split_state import bind_runtime
        original_init = plugin.PluginRuntime.__init__
        def initialize(runtime, *args, **kwargs):
            original_init(runtime, *args, **kwargs)
            bind_runtime(runtime, state_root)
        plugin.PluginRuntime.__init__ = initialize
