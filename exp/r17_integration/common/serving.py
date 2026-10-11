"""Per-connection staged policy using the stock cache and WebSocket seams."""
from __future__ import annotations

import copy
from dataclasses import fields, replace
import math
import os
import time
import uuid
from typing import Any

from .contracts import CaptureSpec, DecisionCapture, RequestContext, WireChunk
from .provider import validate_artifact, validate_binding, validate_config
from .request_rng import RequestRNG

_RESIDENT_GENERATION = uuid.uuid4().hex


def _array(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _cpu_copy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def _wire_array(value):
    """Preserve the adapter's controller dtype and bytes (including float64)."""
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    out = np.array(value, order="C", copy=True)
    if out.dtype.kind != "f":
        raise ValueError("adapter wire actions must have a floating controller dtype")
    return out


def full_call(adapter, wire_obs, request, *, sampler=None, capture=CaptureSpec()):
    """Conformance helper; no duplicate key/vision stage on a MISS."""
    clean = {k: v for k, v in wire_obs.items() if not k.startswith("__")}
    prepared = adapter.prepare(clean, request)
    s1 = adapter.run_stage1(prepared)
    s2 = adapter.run_stage2(s1, capture=capture.enabled)
    result = adapter.sample(s2, sampler or adapter.metadata().default_sampler, RequestRNG(request), capture)
    wire = adapter.to_wire(result.normalized_chunk, prepared)
    return result, wire


def k_step_call(adapter, wire_obs, request, sampler, *, capture=CaptureSpec()):
    return full_call(adapter, wire_obs, request, sampler=sampler, capture=capture)


def make_bundle_validator(provider, *, recipe_factory=None):
    from .recipe import make_recipe_factory
    recipes = recipe_factory or make_recipe_factory(provider)
    def validate(config, storage=None):
        from .config import validate_cache_config
        validate_cache_config(config)
        p, _ = validate_binding(config.adapter_binding)
        if p is not provider:
            pass
        if config.recipe.enabled:
            recipes(config, storage)
        if storage is not None:
            validate_artifact(config, storage)
    return validate


def make_connection_factory(provider, default_config, default_storage, *, capture_sink=None, recipe_factory=None, adapter_factory=None, capture_spec=None, capture_sink_factory=None):
    """Reuse only resident weights/backend/fit; every mutable wrapper is fresh.

    recipe_factory(config, storage) returns a connection-local fitted recipe.
    adapter_factory defaults to provider.load_adapter (must return a fresh
    wrapper over the shared resident model, never reload GPU weights).
    """
    from .config import build_per_connection_components
    from openpi.cache.orchestrator import CacheOrchestrator

    validator = make_bundle_validator(provider)
    validator(default_config, default_storage)

    def factory(base_policy=None, bundle_id="default"):
        from openpi.serving.websocket_policy_server import get_current_cache_bundle
        config, storage, yaml_id = default_config, default_storage, None
        if bundle_id != "default":
            bundle = get_current_cache_bundle(bundle_id)
            if bundle is None:
                raise ValueError(f"unknown bundle {bundle_id!r}")
            config, storage, yaml_id = bundle.cache_config, bundle.shared_storage, bundle.yaml_id
        validator(config, storage)
        components = build_per_connection_components(config, storage, yaml_id=yaml_id, quiet=True)
        adapter = (adapter_factory or provider.load_adapter)()
        if adapter.metadata().identity != provider.model_spec.identity:
            pass
        recipe = (recipe_factory or bundle_recipes)(config, storage) if config.recipe.enabled else None
        core = ConnectionPolicy(adapter, CacheOrchestrator(**components), config, bundle_id,
                                capture_sink=capture_sink, recipe=recipe, capture_spec=capture_spec,
                                trace_factory=provider.trace_factory)
        if capture_sink_factory is not None:
            core.capture_sink = capture_sink_factory(core)
            if core.capture_sink is not None:
                core.spec.capabilities.require("debug_capture")
                core.capture = capture_spec or CaptureSpec(enabled=True, snapshots=True)
        return core
    from .recipe import make_recipe_factory
    bundle_recipes = make_recipe_factory(provider)
    return factory


class ConnectionPolicy:
    def __init__(self, adapter, orchestrator, config, bundle_id, *, capture_sink=None, recipe=None, capture_spec=None, trace_factory=None):
        self.adapter, self.orchestrator = adapter, orchestrator
        self.config, self.bundle_id = config, bundle_id
        self.spec = adapter.metadata()
        _, self.sampler = validate_binding(config.adapter_binding)
        self.commit_horizon = config.adapter_binding["commit_horizon"]
        self.allow_partial_commit = config.adapter_binding.get("allow_partial_commit", False)
        from .recipe import bundle_identity, recipe_identity
        self.bundle_identity = bundle_identity(config)
        self.recipe_identity = recipe_identity(config)
        self.requires_context = not getattr(config, "diagnostic_context", False)
        self.capture_sink, self.recipe = capture_sink, recipe
        if capture_sink is not None:
            self.spec.capabilities.require("debug_capture")
        self.capture = capture_spec or CaptureSpec(enabled=capture_sink is not None, snapshots=capture_sink is not None)
        self.trace_factory = trace_factory
        self._trace_adapter = None
        self._episode = {}
        self._reset()

    def _reset(self):
        self._seq = 0
        self._chunk = self._wire = None
        self._previous_action = None
        self._last_executed_count = 0
        self._chunk_source = "policy"
        self._cursor = 0
        self._awaiting = None
        self.last_execution_ack = None
        self._has_vision, self._hit, self._states = [], [], []

    def on_task_begin(self):
        self._reset()
        self.orchestrator.on_task_begin()
        hook = getattr(self.adapter, "on_task_begin", None)
        if hook:
            hook()

    def on_episode_start(self, task_id=None, init_state_id=None, episode_id=None, **kwargs):
        self._reset()
        extra = kwargs.get("extra_metadata") or {}
        self._episode = dict(extra, episode_id=episode_id)
        if task_id is not None:
            self._episode.setdefault("task_id", task_id)
        if kwargs.get("task"):
            self._episode.setdefault("task_name", kwargs["task"])
        for name in ("original_init_idx", "orig_init_state_idx", "orig_idx", "orig_init_idx"):
            if name in extra:
                self._episode["original_init_idx"] = extra[name]
                break
        if init_state_id is not None:
            self._episode.setdefault("original_init_idx", init_state_id)
        if extra.get("bundle_identity") not in (None, self.bundle_identity):
            pass
        if self.requires_context and self.commit_horizon > self.spec.layout.replan_steps and extra.get("ack_protocol") != "r17.execution.v1":
            raise ValueError("extended commit requires declared ack_protocol=r17.execution.v1")
        if self.allow_partial_commit and extra.get("partial_tail_protocol") != "r17.partial_tail.v1":
            raise ValueError("partial commit requires declared partial_tail_protocol=r17.partial_tail.v1")
        self.orchestrator.on_episode_start(task_key=extra.get("task_name", kwargs.get("task", "")),
                                           episode_id=str(episode_id), extra_metadata=extra)
        if self.recipe:
            self.recipe.reset(self._episode)
        hook = getattr(self.adapter, "on_episode_start", None)
        if hook:
            hook(task_id=task_id, init_state_id=init_state_id, episode_id=episode_id, **kwargs)

    def on_episode_end(self, **kwargs):
        execution = kwargs.pop("execution", kwargs.pop("__execution__", None))
        if execution is not None:
            self.acknowledge_execution(dict(execution, terminal=True))
        if self.requires_context and self._awaiting is not None:
            raise ValueError("episode_end requires final execution acknowledgment")
        self._reset()
        self.orchestrator.on_episode_end()
        hook = getattr(self.adapter, "on_episode_end", None)
        if hook:
            hook(**kwargs)

    def on_task_end(self):
        self._reset()
        self.orchestrator.on_task_end()
        hook = getattr(self.adapter, "on_task_end", None)
        if hook:
            hook()

    def _request(self, overrides):
        e = self._episode
        if self.requires_context:
            complete = dict(e, **overrides)
            needed = ("campaign", "task_name", "original_init_idx", "task_id", "seed", "explicit_seed")
            missing = [name for name in needed if name not in complete or complete[name] is None]
            if missing or not complete.get("campaign") or not complete.get("task_name"):
                raise ValueError(f"missing R17 request context: {missing or ['campaign/task_name']}")
        r = RequestContext(campaign=e.get("campaign", "r17"),
                           benchmark_protocol=self.config.adapter_binding["benchmark_protocol"],
                           task_name=e.get("task_name", str(e.get("task_id", 0))),
                           original_init_idx=e.get("original_init_idx", 0), decision_seq=self._seq,
                           task_id=e.get("task_id", 0), bundle_generation=e.get("bundle_generation", self.bundle_id),
                           arm=e.get("arm", ""), attempt=e.get("attempt", 0), seed=e.get("seed", 7),
                           explicit_seed=e.get("explicit_seed"))
        unknown = set(overrides) - {f.name for f in fields(RequestContext)}
        if unknown:
            raise ValueError(f"unknown request context fields: {sorted(unknown)}")
        if "decision_seq" in overrides and overrides["decision_seq"] != self._seq:
            raise ValueError("request decision sequence does not match connection")
        if "benchmark_protocol" in overrides and overrides["benchmark_protocol"] != r.benchmark_protocol:
            pass
        request = replace(r, **overrides)
        table_identity = request.task_table_identity or e.get("task_table_identity")
        if table_identity is not None:
            from .benchmarks import get_benchmark, task_table_identity
            if table_identity != task_table_identity(request.benchmark_protocol):
                pass
            tasks = get_benchmark(request.benchmark_protocol).ordered_tasks
            if not 0 <= request.task_id < len(tasks) or tasks[request.task_id] != request.task_name:
                raise ValueError("request task_name/task_id differs from canonical benchmark task table")
            request = replace(request,task_table_identity=table_identity)
        if self.config.recipe.enabled or self.config.recipe.task_table_identity is not None:
            tasks = self.config.recipe.ordered_tasks
            if not 0 <= request.task_id < len(tasks) or tasks[request.task_id] != request.task_name:
                raise ValueError("request task_name/task_id differs from canonical recipe task table")
        return request

    def acknowledge_execution(self, ack):
        if self._awaiting is None:
            raise ValueError("execution acknowledgment has no preceding decision")
        seq, available, request = self._awaiting
        count = ack.get("executed_count")
        if ack.get("decision_seq") != seq or type(count) is not int or not 0 <= count <= available:
            raise ValueError("invalid/stale execution acknowledgment")
        cursor = self._cursor + count
        if "cursor" in ack and ack["cursor"] != cursor:
            raise ValueError("execution acknowledgment cursor mismatch")
        if "cursor_end" in ack and ack["cursor_end"] != cursor:
            raise ValueError("execution acknowledgment cursor_end mismatch")
        if "request" in ack:
            from dataclasses import asdict
            if ack["request"] != asdict(request):
                pass
        self._cursor = cursor
        self._last_executed_count = count
        self._awaiting = None
        execution_hook = getattr(self.recipe, "on_execution", None)
        if execution_hook:
            execution_hook(count)
        hook = getattr(self.capture_sink, "on_execution", None)
        if hook:
            hook({"decision_seq": seq, "executed_count": count, "cursor": cursor, "cursor_end": cursor,
                  "terminal": bool(ack.get("terminal", False)), "bundle_id": self.bundle_id,
                  "request": request})
        from dataclasses import asdict
        self.last_execution_ack = {"request": asdict(request), "decision_seq": seq,
                                   "executed_count": count, "cursor": cursor, "cursor_end": cursor,
                                   "terminal": bool(ack.get("terminal", False)),
                                   "bundle_identity": self.bundle_identity}
        if ack.get("terminal"):
            self._chunk = self._wire = None
        return self.last_execution_ack

    def infer(self, obs: dict) -> dict:
        begin = time.perf_counter()
        timing = {name: 0.0 for name in ("ack", "state", "prepare", "stage1", "retrieval", "recipe", "stage2", "stage3", "to_wire")}
        drain_wait = getattr(self.adapter, "drain_queue_wait_ms", None)
        queue_timing = dict.fromkeys(timing, 0.0) if callable(drain_wait) else None
        if queue_timing is not None:
            drain_wait()  # Discard previous/control work; this ledger is one decision.
        def measured(name, call):
            start = time.perf_counter()
            try:
                return call()
            finally:
                elapsed = (time.perf_counter() - start) * 1000.0
                timing[name] += elapsed
                if queue_timing is not None:
                    wait = drain_wait()
                    if isinstance(wait, bool) or not isinstance(wait, (int,float)) or not math.isfinite(wait) or not 0 <= wait <= elapsed + 1e-6:
                        raise ValueError("adapter queue-wait receipt must be finite and inside measured stage")
                    queue_timing[name] += wait
        import numpy as np
        import torch
        from openpi.cache.types import CheckpointID
        from openpi.cache.components.judge import HitType
        from .query_view import QueryView

        # Control/diagnostic envelopes never reach model preprocessing.
        ack = obs.get("__execution__")
        if ack is not None:
            measured("ack", lambda: self.acknowledge_execution(ack))
        elif self._awaiting is not None:
            if self.requires_context and self.commit_horizon > self.spec.layout.replan_steps:
                raise ValueError("extended commit requires __execution__ acknowledgment; cannot silently downgrade")
            self._chunk = self._wire = None  # no evidence of executed controls
            self._previous_action = None
            self._last_executed_count = 0
            self._awaiting = None
        request = self._request(obs.get("__request__", {}))
        if self.requires_context and self.commit_horizon > self.spec.layout.replan_steps and self._episode.get("ack_protocol") != "r17.execution.v1":
            raise ValueError("extended commit requires declared ack_protocol=r17.execution.v1")
        if self.allow_partial_commit and self._episode.get("partial_tail_protocol") != "r17.partial_tail.v1":
            raise ValueError("partial commit requires declared partial_tail_protocol=r17.partial_tail.v1")
        clean = {k: v for k, v in obs.items() if not k.startswith("__")}
        state_view = measured("state", lambda: self.adapter.extract_state_only(clean))
        state = _array(getattr(state_view, "state", state_view)).reshape(-1)
        calls, nfe, named, diagnostics = (0, 0, 0), 0, {}, {}
        if self.recipe is not None and hasattr(self.recipe, "tail_diagnostics"):
            diagnostics = self.recipe.tail_diagnostics()
        result = None
        check = None
        proposal = None
        source = self._chunk_source + "_tail"
        force_look = self.recipe is not None and self.recipe.force_look(request, self._has_vision, self._hit)
        remaining = self.commit_horizon - self._cursor
        can_tail = self._chunk is not None and remaining >= (1 if self.allow_partial_commit else self.spec.layout.replan_steps) and not force_look
        try:
            if not can_tail:
                self._cursor = 0
                prepared = measured("prepare", lambda: self.adapter.prepare(clean, request))
                s1 = measured("stage1", lambda: self.adapter.run_stage1(prepared))
                named = measured("stage1", lambda: self.adapter.retrieval_fields(s1))
                check = measured("retrieval", lambda: self.orchestrator.check(CheckpointID.CP1, stage1=s1, retrieval_fields=named))
                proposal = None
                if self.recipe is not None:
                    proposal = measured("recipe", lambda: self.recipe.query(QueryView(request, named, state, self._has_vision,
                                                           self._hit, self._states,
                                                           previous_action=self._previous_action.copy() if self._previous_action is not None else None,
                                                           executed_count=self._last_executed_count,
                                                           prev_hit=bool(self._hit[-1]) if self._hit else None)))
                    diagnostics = dict(proposal.extras)
                miss_mode = self.config.recipe.miss_mode
                if miss_mode != "legacy" and (proposal is None or proposal.force_miss and
                        not (miss_mode == "stall_only" and diagnostics.get("stall_call") is True or
                             diagnostics.get("knob_call") is True)):
                    raise ValueError("ordinary MISS forbidden by cache-study YAML; no teacher fallback")
                if proposal is not None and not proposal.force_miss:
                    normalized, source, calls = _array(proposal.action), "library", (1, 0, 0)
                elif proposal is None and check.hit_type == HitType.FULL_HIT:
                    if check.payload is None:
                        raise ValueError("FULL_HIT has no normalized payload")
                    normalized, source, calls = _array(check.payload.action_chunk), "cache", (1, 0, 0)
                else:
                    s2 = measured("stage2", lambda: self.adapter.run_stage2(s1, capture=self.capture.enabled))
                    if check.hit_type == HitType.WARM_START:
                        raise ValueError("native warm payload requires a certified adapter executor")
                    result = measured("stage3", lambda: self.adapter.sample(s2, self.sampler, RequestRNG(request), self.capture))
                    normalized, source, calls, nfe = _array(result.normalized_chunk), "policy", (1, 1, result.actual_nfe), result.actual_nfe
                    diagnostics["rng_receipt"] = dict(result.rng_receipt)
                layout = self.spec.layout
                if normalized.shape != (layout.horizon, layout.action_width) or not np.isfinite(normalized).all():
                    raise ValueError("adapter/cache normalized chunk geometry or finite check failed")
                wire = measured("to_wire", lambda: self.adapter.to_wire(normalized, prepared))
                wire = _wire_array(wire.actions if isinstance(wire, WireChunk) else wire)
                if wire.shape != (layout.horizon, layout.valid_action_dim) or not np.isfinite(wire).all():
                    raise ValueError("adapter wire chunk geometry or finite check failed")
                self._chunk, self._wire = normalized, wire
                self._chunk_source = source
                self.orchestrator.broadcast_action(torch.from_numpy(normalized.copy()))
            available = min(self.spec.layout.replan_steps, self.commit_horizon - self._cursor)
            if available <= 0:
                raise ValueError("no actions remain inside commit horizon")
            actions = self._wire[self._cursor:self.commit_horizon].copy()
            diagnostics.update(cost_ms=(time.perf_counter() - begin) * 1000.0, stage_timing_ms=dict(timing),
                               stage_ms=dict(timing), timing_scope="policy_host_wall_before_capture_sink",
                               timing_kind="host_wall", capture_enabled=self.capture.enabled or self.capture_sink is not None,
                               pricing_eligible=not self.capture.enabled and self.capture_sink is None and queue_timing is not None,
                               bundle_identity=self.bundle_identity, recipe_identity=self.recipe_identity)
            if self.config.recipe.miss_mode != "legacy":
                diagnostics.update(resident_pid=os.getpid(), resident_generation=_RESIDENT_GENERATION)
            queue_wait = sum(queue_timing.values()) if queue_timing is not None else None
            diagnostics.update(queue_wait_ms=queue_wait, stage_queue_wait_ms=queue_timing,
                service_ms=diagnostics["cost_ms"]-queue_wait if queue_wait is not None else None,
                stage_service_ms={name:timing[name]-queue_timing[name] for name in timing} if queue_timing is not None else None,
                service_timing_scope="policy_host_wall_minus_adapter_model_lock_wait_before_capture_sink" if queue_timing is not None else None,
                pricing_time_field="service_ms" if queue_timing is not None else None, gpu_service_ms=None)
            calibration = getattr(self.recipe,"ir_calibration",None)
            if calibration is not None:
                diagnostics["ir_calibration"] = copy.deepcopy(calibration)
            if self.config.adapter_binding.get("allow_lossy_sampler", False):
                diagnostics.update(allow_lossy_sampler=True, sampler_study_label=self.config.adapter_binding["sampler_study_label"],
                                   sampler_quality_status="lossy_study")
            if self.capture_sink:
                capture_begin = time.perf_counter()
                capture_fields = {k: _array(v) for k, v in named.items()}
                capture_fields.setdefault("robot_state", state.copy())
                record = DecisionCapture(request, source, self._cursor, 0, calls, nfe,
                                         self._chunk.copy(), actions.copy(),
                                         fields=capture_fields,
                                         sampler_receipt={"identity": self.sampler.identity, "native_timesteps": self.sampler.native_timesteps},
                                         diagnostics=copy.deepcopy(diagnostics))
                if result is not None:
                    record.native_snapshots = tuple(replace(s, x_t=_cpu_copy(s.x_t), noise_state=_cpu_copy(s.noise_state))
                                                    for s in result.snapshots)
                    record.model_capture = _cpu_copy(result.capture)
                    if self._trace_adapter is None:
                        from .provider import get_trace_provider
                        trace_factory = self.trace_factory or get_trace_provider(self.spec.model_id)
                        self._trace_adapter = trace_factory(self.adapter) if trace_factory else self.adapter.make_trace_adapter()
                    trace = self._trace_adapter
                    hook = getattr(self.adapter, "capture_fields", None)
                    if hook:
                        record.model_capture = hook(s1, s2, result)
                    elif hasattr(trace, "sample_fields"):
                        record.model_capture = trace.sample_fields(s1, s2, result)
                    elif hasattr(trace, "from_stages"):
                        record.model_capture = trace.from_stages(prepared, s1, result, WireChunk(self._wire.copy()))
                diagnostics["capture_overhead_ms"] = (time.perf_counter() - capture_begin) * 1000.0
                diagnostics["capture_overhead_scope"] = "core_capture_preparation_excluding_sink"
                record.diagnostics.update(copy.deepcopy(diagnostics))
                sink_begin = time.perf_counter()
                self.capture_sink(record)
                diagnostics["capture_sink_ms"] = (time.perf_counter() - sink_begin) * 1000.0
            else:
                diagnostics["capture_overhead_ms"] = None if self.capture.enabled else 0.0
            self._awaiting = (self._seq, available, request)
            self._previous_action = self._chunk.copy()
            self._has_vision.append(bool(calls[0]))
            self._hit.append(int(source in {"cache", "library", "cache_tail", "library_tail", "policy_tail"}))
            self._states.append(state.copy())
            self._seq += 1
            hit = source in {"cache", "library", "cache_tail", "library_tail", "policy_tail"}
            meta = {"hit_type": "FULL_HIT" if hit else "MISS", "start_t": None,
                    "winner_id": check.entry_id if check else None,
                    "cp1_score": check.score if check else None, "score": check.score if check else None,
                    "checkpoint": "CP1" if check else None, "searched": check.searched if check else False,
                    "source": source, "cursor": self._cursor,
                    "decision_seq": request.decision_seq,
                    "stage_calls": calls, "actual_nfe": nfe, "sampler_identity": self.sampler.identity,
                    "commit_horizon": self.commit_horizon, **diagnostics}
            if proposal is not None:
                meta.update(donor_rows=proposal.rows.tolist(), donor_weights=proposal.weights.tolist())
            response = {"actions": actions, "__hit_meta__": meta}
            if self.config.collection.export_collect_meta:
                keys = check.query_keys if check is not None else None
                response["__collect_meta__"] = {"searched": bool(check and check.searched),
                    "collect": {name: (_array(keys[name]).astype(np.float32 if name == "robot_state" else np.float16)
                                       if name in keys else None) for name in self.config.collection.collect_fields}
                    if keys else None}
            return response
        except Exception:
            self._chunk = self._wire = None
            raise
        finally:
            self.orchestrator.clear()
