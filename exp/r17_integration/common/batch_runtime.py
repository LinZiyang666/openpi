"""Cross-connection stage queues for adapters with native StageBatchers.

Only native forwards run on the workers. RNG, retrieval, capture and wire
transforms remain on each connection's adapter. DP and Fast-WAM own their
existing runtimes.
"""
from __future__ import annotations

from collections import defaultdict
import contextlib
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import math
import threading
from types import SimpleNamespace


@dataclass(frozen=True)
class BatchConfig:
    max_batch: int = 1
    batch_wait_ms: float = 5.0

    def __post_init__(self):
        if type(self.max_batch) is not int or self.max_batch < 1:
            raise ValueError("max_batch must be a positive integer")
        if (isinstance(self.batch_wait_ms, bool) or not isinstance(self.batch_wait_ms, (int, float))
                or not math.isfinite(self.batch_wait_ms) or self.batch_wait_ms < 0):
            raise ValueError("batch_wait_ms must be finite and nonnegative")

    def metadata(self):
        return asdict(self)


def shape_key(value):
    """Exact shapes only: in particular never pad GR00T language sequences."""
    import torch
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, dict):
        return tuple((k, shape_key(v)) for k, v in value.items())
    if is_dataclass(value):
        return tuple((f.name, shape_key(getattr(value, f.name))) for f in fields(value))
    if isinstance(value, (tuple, list)):
        return tuple(shape_key(v) for v in value)
    # DynamicCache is opaque; the surrounding stage1 fixes its geometry.
    return value if isinstance(value, (str, int, float, bool, type(None))) else type(value).__name__


@dataclass
class _Input:
    batcher: object
    value: object
    lock: object = None
    ready_events: tuple = ()


@dataclass
class _Output:
    value: object
    size: int


class _StageBatcher:
    """Route suite-specific runners and report the actual native batch sizes."""
    @staticmethod
    def _unwrap(payload):
        return replace(payload, stage2_out=payload.stage2_out.value)

    def bucket_key(self, payload):
        item = payload.stage2_out
        x = getattr(payload, 'noise', None)
        if x is None:
            x = payload.start_x
        return (id(item.batcher), shape_key(item.value), shape_key(x),
                item.batcher.bucket_key(self._unwrap(payload)))

    @staticmethod
    def default_save_timesteps():
        return ()

    @staticmethod
    def _run(items, call):
        import torch
        first = items[0]
        # The model lock covers the GPU stages (and compiled vision buffers); GR00T CPU
        # transforms use their own lock and overlap these batches.
        with first.lock if first.lock is not None else contextlib.nullcontext():
            for item in items:
                for event in item.ready_events:
                    torch.cuda.current_stream(event.device).wait_event(event)
            outputs = call(first.batcher, [item.value for item in items])
            if first.lock is not None and first.ready_events:
                # Complete use of shared model buffers before releasing its lock.
                torch.cuda.current_stream(first.ready_events[0].device).synchronize()
        return [_Output(out, len(items)) for out in outputs]

    def _early(self, stage, payloads, **kwargs):
        groups = defaultdict(list)
        for i, item in enumerate(payloads):
            groups[(id(item.batcher), shape_key(item.value))].append((i, item))
        outputs = [None] * len(payloads)
        for group in groups.values():
            replies = self._run([item for _, item in group],
                lambda batcher, values: getattr(batcher, f"run_stage{stage}_batch")(values, **kwargs))
            for (i, _), reply in zip(group, replies, strict=True):
                outputs[i] = reply
        return outputs

    def run_stage1_batch(self, payloads):
        return self._early(1, payloads)

    def run_stage2_batch(self, payloads, *, capture):
        return self._early(2, payloads, capture=capture)

    def _late(self, method, payloads, **kwargs):
        items = [p.stage2_out for p in payloads]
        native = [self._unwrap(p) for p in payloads]
        return self._run(items, lambda batcher, _: getattr(batcher, method)(native, **kwargs))

    def run_stage3_miss(self, payloads, **kwargs):
        return self._late("run_stage3_miss", payloads, **kwargs)

    def run_stage3_warm(self, payloads, **kwargs):
        return self._late("run_stage3_warm", payloads, **kwargs)


class BatchRuntime:
    def __init__(self, adapter, config, *, device):
        from openpi.serving.batching_core import BatchingCore
        self.config, self.device = config, device
        self._batchers = {}
        self._lock = threading.Lock()
        self.core = BatchingCore(_StageBatcher(), device=device,
            max_batch_size=config.max_batch, max_wait_ms=config.batch_wait_ms)
        self.batcher_for(adapter)
        self.core.start()

    def batcher_for(self, adapter):
        group = adapter.batching_group()
        with self._lock:
            if group not in self._batchers:
                self._batchers[group] = adapter.make_stage_batcher()
            return self._batchers[group]

    def close(self):
        self.core.stop()

    def wrap_connection_factory(self, base_factory):
        def factory(*args, **kwargs):
            connection = base_factory(*args, **kwargs)
            original = connection.infer
            def infer(obs):
                connection.adapter.last_batch_sizes = {}
                result = original(obs)
                result['__hit_meta__']['batching'] = dict(config=self.config.metadata(),
                    stages={k: dict(size=v) for k, v in connection.adapter.last_batch_sizes.items()})
                return result
            connection.infer = infer
            return connection
        return factory


class BatchedAdapter:
    """Connection-local bridge; the native adapter still owns sample/resume."""
    # BatchingCore does not expose per-request dispatch timestamps. Native
    # adapter lock wait alone cannot price service excluding the stage queues.
    drain_queue_wait_ms = None

    def __init__(self, adapter, runtime):
        self.adapter, self.runtime = adapter, runtime
        self.last_batch_sizes = {}
        adapter.bind_stage_executor(self)

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def _input(self, value):
        from openpi.serving.batching_core import record_ready_events
        return _Input(self.runtime.batcher_for(self.adapter), value,
            getattr(self.adapter, 'model_lock', None), record_ready_events(self.runtime.device))

    def _submit(self, stage, payload, *, capture=False):
        reply = self.runtime.core.submit_to_stage(stage, "native", payload,
            requires_stage2_capture=capture)
        self.last_batch_sizes[f'stage{stage}'] = reply.size
        return reply.value

    def run_stage1(self, prepared):
        native = self._submit(1, self._input(self.adapter.stage1_batch_input(prepared)))
        return self.adapter.finish_stage1(native, prepared)

    def run_stage2(self, s1, *, capture=False):
        from .contracts import Stage2
        native = self._submit(2, self._input(s1.data), capture=capture)
        return Stage2(native, s1)

    def miss(self, s2, noise, sampler, points):
        from openpi.serving.batching_core import Stage3MissPayload
        item = self._input(self.adapter.stage3_batch_input(s2))
        payload = Stage3MissPayload(item, noise, sampler.num_steps,
                                   None if points is None else tuple(points), item.ready_events)
        out = self._submit(3, payload)
        action = getattr(out, 'action_chunk', None)
        if action is None:
            action = out.action
        return SimpleNamespace(action_chunk=action, action_pred=action, steps_run=sampler.num_steps,
                               intermediates=out.intermediates)

    def warm(self, s2, x, snapshot, sampler, *, capture=False):
        from openpi.serving.batching_core import Stage3WarmStartPayload
        item = self._input(self.adapter.stage3_batch_input(s2))
        payload = Stage3WarmStartPayload(item, x, snapshot.native_timestep, sampler.num_steps,
                                        capture, item.ready_events)
        out = self._submit(3, payload)
        action = getattr(out, 'action_chunk', None)
        if action is None:
            action = out.action
        return SimpleNamespace(action_chunk=action, action_pred=action,
            steps_run=sampler.num_steps-snapshot.loop_index,
            first_step_input=getattr(out, 'first_step_input', None),
            first_step_x=getattr(out, 'first_step_x', None))


def create_runtime(adapter, max_batch=1, batch_wait_ms=5, *, device):
    config = BatchConfig(max_batch, batch_wait_ms)
    return BatchRuntime(adapter, config, device=device) if max_batch > 1 else None
