"""Weight-owning stage service and weight-free client adapters (opt-in only)."""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import os
import socket
import threading
import uuid
from types import SimpleNamespace

import cloudpickle
import torch

from openpi.serving.split_transport import (
    Arena, BackpressureError, Channel, StageRPCError, recv_frame, send_frame,
)


class StageService:
    """Session-scoped GPU handles. EOF/error cleanup waits for outstanding work.

    Channels are bounded at admission, as are handles and tensor arenas. An
    epoch distinguishes every model-process lifetime; no failed call is retried.
    The real Pi0.5 coordinator remains responsible for GPU streams and batching.
    """
    def __init__(self, model, *, coordinator=None, runner=None, metadata=None,
                 max_channels=128, max_handles=32, max_builders=256):
        self.model = model
        self.coordinator = coordinator
        self.runner = runner
        self.metadata = metadata or {}
        self.epoch = uuid.uuid4().hex
        self.admission = threading.BoundedSemaphore(max_channels)
        self.max_handles = max_handles
        self.max_builders = max_builders
        self.model_lock = threading.RLock()
        self.sessions = {}
        self.sessions_lock = threading.Lock()

    def serve_channel(self, sock):
        admitted = self.admission.acquire(blocking=False)
        arena = None
        session = uuid.uuid4().hex
        handles, builders = {}, {}
        try:
            arena = Arena.receive(sock)
            with self.sessions_lock:
                self.sessions[session] = handles
            while True:
                operation, envelope = recv_frame(sock)
                try:
                    if not admitted:
                        raise BackpressureError("GPU channel limit reached")
                    def resolve(ref):
                        epoch, owner, handle = ref
                        if epoch != self.epoch or owner != session or handle not in handles:
                            raise StageRPCError("expired or foreign GPU stage handle; restart the episode")
                        return handles[handle]
                    args = arena.decode(envelope, resolve=resolve)
                    result, retain = self.dispatch(operation, args, builders, handles)
                    if retain:
                        if len(handles) >= self.max_handles:
                            raise BackpressureError("GPU stage handle limit reached")
                        handle = uuid.uuid4().hex
                        handles[handle] = result
                        result = snapshot(result)
                        result._split_ref = (self.epoch, session, handle)
                    encoded = arena.encode(result, slot=1, references=False)
                    send_frame(sock, [True, encoded])
                except Exception as exc:
                    send_frame(sock, [False, [type(exc).__name__, str(exc)]])
        except (EOFError, OSError):
            pass
        finally:
            # dispatch is synchronous in this channel; coordinator errors only
            # return after synchronization (fatal CUDA payloads are retained by
            # BatchingCore until process teardown).
            handles.clear()
            builders.clear()
            with self.sessions_lock:
                self.sessions.pop(session, None)
            if arena is not None:
                arena.close()
            sock.close()
            if admitted:
                self.admission.release()

    def dispatch(self, operation, args, builders, handles):
        if operation == "hello":
            return {"epoch": self.epoch, **self.metadata}, False
        if operation == "release":
            handles.clear()
            return None, False
        if operation == "drop_key":
            builders.pop(args, None)
            return None, False
        if operation == "key":
            from openpi.cache.types import CheckpointID
            identity, blob, checkpoint, outputs = args
            checkpoint = CheckpointID(checkpoint)
            if identity not in builders:
                if len(builders) >= self.max_builders:
                    raise BackpressureError("GPU key-builder limit reached")
                builders[identity] = cloudpickle.loads(blob)
            builder = builders[identity]
            try:
                builder.collect(checkpoint, **outputs)
                return builder.build(checkpoint), False
            finally:
                builder.clear()
        if operation == "batch_params":
            return self.coordinator.update_batch_params(**args), False
        if operation == "pi_stage":
            stage, bundle, payload, kwargs = args
            if stage == 1:
                handles.clear()  # also bounds multi-step prefill lifetime
            return self.coordinator.submit_to_stage(stage, bundle, payload, **kwargs), True
        if operation == "noise":
            shape, kwargs = args
            # Keep the old device and generator algorithm, never substitute CPU
            # randn for CUDA randn. Explicit per-decision noise travels exactly.
            device = next(self.model.parameters()).device
            with self.model_lock:
                return self.model.sample_noise(tuple(shape), device, **kwargs), False
        if operation == "groot":
            name, positional, kwargs = args
            if name == "run_stage1":
                handles.clear()
            if self.coordinator is not None:
                result = self._groot_batched(name, positional, kwargs)
            else:
                with self.model_lock, self.runner.session():
                    result = getattr(self.runner, name)(*positional, **kwargs)
            return result, name in ("run_stage1", "run_stage2_llm")
        raise ValueError(f"unknown GPU operation {operation!r}")

    def _groot_batched(self, name, positional, kwargs):
        from openpi.cache.groot.staged import GrootStage2Output, GrootStage3Output
        from openpi.serving.batching_core import Stage3MissPayload, Stage3WarmStartPayload
        from openpi.cache.groot.batcher import stage3_input
        if name == "run_stage1":
            return self.coordinator.submit_to_stage(1, "split", positional[0])
        if name in ("run_stage2", "run_stage2_llm"):
            s2 = self.coordinator.submit_to_stage(2, "split", positional[0])
            if name == "run_stage2_llm":
                return s2
        else:
            s2 = positional[0]
        if kwargs.get("on_step") is not None:
            raise ValueError("split: stage callbacks must execute in the GPU process")
        with self.model_lock, self.runner.session():
            condition = stage3_input(self.runner, s2)
            if name == "run_stage3_from":
                start_x = positional[1] if len(positional) > 1 else kwargs["start_x"]
                start_t = positional[2] if len(positional) > 2 else kwargs["start_t"]
                schedule = kwargs["schedule"]
                if schedule != self.runner.live_schedule():
                    raise ValueError("split: warm-start schedule differs from the live model")
                payload = Stage3WarmStartPayload(
                    condition, start_x.to(s2.backbone_features.device).squeeze(0),
                    start_t, schedule.num_steps,
                    capture_first_step=kwargs.get("capture_first_step", False),
                )
            else:
                noise = kwargs.get("noise")
                if noise is None:
                    noise = self.runner.sample_noise(s2)
                payload = Stage3MissPayload(
                    condition, noise.to(s2.backbone_features.device).squeeze(0),
                    self.runner.live_schedule().num_steps,
                )
        out = self.coordinator.submit_to_stage(3, "split", payload)
        if name == "run_stage2":
            return GrootStage2Output(s2.backbone_features, s2.attention_mask, s2.action_inputs,
                                     action_pred=out.action)
        return GrootStage3Output(
            action_pred=out.action, start_t=getattr(payload, "start_t", None),
            steps_run=(self.runner.live_schedule().remaining_steps(payload.start_t)
                       if name == "run_stage3_from" else payload.num_steps), first_step_input=out.first_step_input,
            first_step_x=out.first_step_x,
        )

    def listen(self, path):
        """Private AF_UNIX endpoint; only the explicit split launcher calls this."""
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        os.chmod(path, 0o600)
        listener.listen(128)
        try:
            while True:
                sock, _ = listener.accept()
                threading.Thread(target=self.serve_channel, args=(sock,), daemon=True).start()
        finally:
            listener.close()


def snapshot(value):
    """Host view for CPU cache work. KV and attention glue never leave the GPU.

    Key reduction uses the retained original via RPC, not this host view.
    Shallow copying preserves custom stage override classes and audit fields.
    """
    if not dataclasses.is_dataclass(value):
        raise TypeError(f"stage result must be a dataclass, got {type(value)}")
    out = copy.copy(value)
    for name, child in vars(value).items():
        if name == "past_key_values":
            setattr(out, name, None)
        elif name == "stage1":
            setattr(out, name, snapshot(child))
        elif name in ("prefix_att_2d_masks_4d", "prefix_position_ids"):
            setattr(out, name, None)
    return out


class StageClient:
    def __init__(self, endpoint, *, capacity=32 << 20, timeout=60.0, channel_factory=None):
        self.endpoint, self.capacity, self.timeout = endpoint, capacity, timeout
        self.local = threading.local()
        self.channel_factory = channel_factory

    def channel(self):
        channel = getattr(self.local, "channel", None)
        if channel is None or channel.closed:
            if self.channel_factory is not None:
                channel = self.channel_factory()
            else:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                try:
                    sock.connect(str(self.endpoint))
                    channel = Channel(sock, capacity=self.capacity, timeout=self.timeout)
                except BaseException:
                    sock.close()
                    raise
            self.local.channel = channel
        return channel

    def call(self, operation, args=None):
        return self.channel().call(operation, args)

    @contextlib.contextmanager
    def decision(self):
        # Blind decisions do not establish a channel or contact the GPU.
        try:
            yield
        finally:
            channel = getattr(self.local, "channel", None)
            if channel is not None and not channel.closed:
                try:
                    channel.call("release")
                except StageRPCError:
                    channel.close()


class RemoteCoordinator:
    def __init__(self, client):
        self.client = client

    def start(self):
        # Expose the same batch-param control handler in every front.
        from openpi.serving import batching_core
        batching_core._ACTIVE_COORDINATOR = self

    def stop(self, *args, **kwargs):
        pass

    def submit_to_stage(self, stage_id, bundle_id, payload, **kwargs):
        return self.client.call("pi_stage", (stage_id, bundle_id, payload, kwargs))

    def update_batch_params(self, **kwargs):
        return self.client.call("batch_params", kwargs)


class RemotePiModel:
    def __init__(self, client, config):
        self.client, self.config = client, config

    def sample_noise(self, shape, device, **kwargs):
        if kwargs:
            raise ValueError("split: explicit noise required for custom RNG generators")
        return self.client.call("noise", (shape, kwargs))

    def _require_coordinator(self, *args, **kwargs):
        raise RuntimeError("split Pi0.5 requires the staged concurrent path")

    run_stage1 = run_stage2 = run_stage2_capture = run_stage3 = run_stage3_from = _require_coordinator


class RemoteGrootRunner:
    def __init__(self, client, model, **kwargs):
        self.client, self._model = client, model
        self._device_type = "cpu"

    def session(self):
        # Autocast belongs to the GPU runner; CPU cache work stays outside it.
        return contextlib.nullcontext()

    def live_schedule(self):
        from openpi.cache.types import groot_n15_schedule
        return groot_n15_schedule(self._model.action_head.num_inference_timesteps)

    def __getattr__(self, name):
        if name not in ("run_stage1", "run_stage2", "run_stage2_llm", "run_stage3", "run_stage3_from"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.client.call("groot", (name, args, kwargs))


class RemoteKeyBuilder:
    """Same CPU-visible collection; original GPU reductions for exact keys."""
    def __init__(self, inner, client):
        self.inner, self.client = inner, client
        self.identity = uuid.uuid4().hex
        self.blob = cloudpickle.dumps(inner)
        self.outputs = None
        self.channels = set()

    def collect(self, checkpoint_id, **outputs):
        self.outputs = outputs
        self.inner.collect(checkpoint_id, **outputs)

    def build(self, checkpoint_id):
        # Enums are encoded as their string values at this boundary.
        channel = self.client.channel()
        registered = getattr(channel, "registered_keys", None)
        if registered is None:
            registered = channel.registered_keys = set()
        result = channel.call("key", (self.identity, self.blob if self.identity not in registered else b"",
                                      checkpoint_id.value, self.outputs))
        registered.add(self.identity)
        self.channels.add(channel)
        return result

    @property
    def cached_data(self):
        return self.inner.cached_data

    def clear(self):
        self.outputs = None
        self.inner.clear()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def close(self):
        for channel in self.channels:
            if not channel.closed:
                try:
                    channel.call("drop_key", self.identity)
                    channel.registered_keys.discard(self.identity)
                except StageRPCError:
                    channel.close()
        self.channels.clear()


def make_groot_coordinator(runner, *, max_batch_size=1, max_wait_ms=0):
    """All model stages share the existing core. Preserve B=1 stage-1 kernels.

    Backbone requests group by exact shape/dtype before concatenation. The
    existing GR00T adapter supplies action-head bucketing, k-step execution,
    noise dtype and snapshot semantics. Batch width >1 is explicitly opt-in.
    """
    from openpi.cache.groot.batcher import GrootStageBatcher
    from openpi.cache.groot.staged import GrootStage1Output, GrootStage2Output
    from openpi.serving.batching_core import BatchingCore

    class Batcher(GrootStageBatcher):
        def run_stage1_batch(self, payloads):
            with runner.session():
                return [runner.run_stage1(p) for p in payloads]

        def run_stage2_batch(self, payloads, *, capture):
            groups = {}
            for i, p in enumerate(payloads):
                key = (tuple(p.input_embeds.shape[1:]), p.input_embeds.dtype)
                groups.setdefault(key, []).append((i, p))
            replies = [None] * len(payloads)
            with runner.session():
                for group in groups.values():
                    if len(group) == 1:
                        index, payload = group[0]
                        replies[index] = runner.run_stage2_llm(payload)
                        continue
                    # The action inputs belong to each row. Stage 2 only reads
                    # the embeddings and mask, so never concatenate opaque
                    # policy mappings just to run the backbone.
                    first = group[0][1]
                    stacked = GrootStage1Output(
                        torch.cat([p.input_embeds for _, p in group]),
                        torch.cat([p.attention_mask for _, p in group]),
                        torch.cat([p.image_token_mask for _, p in group]), first.action_inputs,
                    )
                    out = runner.run_stage2_llm(stacked)
                    for row, (index, p) in enumerate(group):
                        replies[index] = GrootStage2Output(
                            out.backbone_features[row:row+1], out.attention_mask[row:row+1], p.action_inputs,
                        )
            return replies

    device = getattr(runner._model, "device", "cpu")
    coordinator = BatchingCore(Batcher(runner), device=device,
                               max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
    coordinator.start()
    return coordinator
