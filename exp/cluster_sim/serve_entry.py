"""Single-copy π0.5 server process for the cluster simulation (run by serve.py inside the R16 server tree).

Runs $CLUSTER_SIM_MODULE (the R16 resident for the cache system, scripts.serve_policy --cache for pure inference)
unchanged, after these process-local changes (1-6 in both modes, 7 in cache mode only):

1. Request timing: every frame's arrival (after recv) and its reply (after send) per connection, appended to
   $CLUSTER_SIM_LOG/requests_<pid>.csv as `conn,kind,t_recv,t_sent` (kind = infer / ctrl).
2. Batching: per-stage batch limit and wait (CLUSTER_SIM_BATCH / CLUSTER_SIM_WAIT_MS, e.g. "1:32,2:16,3:16");
   a stage worker takes its first request, drains everything already queued, then waits up to its wait.
   CLUSTER_SIM_UNITS groups stages that take the GPU one batch at a time (e.g. "1,23": stage 1 on its own, stages 2
   and 3 exclusive with stage 3 first); a stage adds whatever arrived while it waited for its unit.
   Stage worker counts stay BATCHING_STAGE{1,2,3}_WORKERS. Per-stage batch counts go to stages_<pid>.json and
   every batch (stage, start, end, size, oldest request's queue wait) to batches_<pid>.csv.
3. Prefix trimming: cameras whose mask is all False (LIBERO's empty third view) are not encoded, and the prompt
   is cut to CLUSTER_SIM_TEXT_TOKENS. Masked tokens take no position (cumsum of the pad mask) and no attention,
   so the computation is the same; only the GEMM shapes shrink (968 -> 512 + text tokens).
4. Request threads: the asyncio default executor gets CLUSTER_SIM_THREADS threads (default 32 otherwise).
5. Memory: the process may allocate at most CLUSTER_SIM_VRAM_GB of the card. Explicit gc.collect() is a no-op and
   gc.freeze() runs at the first infer request (startup objects leave the collector's generations).
6. CUDA graphs (CLUSTER_SIM_GRAPHS=1): stage 1 (preprocess + SigLIP + prefix embedding), stage 2 (Gemma prefix KV)
   and one denoise step each replay a CUDA graph, one per padded batch size (CLUSTER_SIM_BUCKETS); batches are padded
   with copies of their last row and the outputs sliced back. Each stage is FX-traced on real inputs first; with
   CLUSTER_SIM_COMPILE set the traced stages are compiled by inductor before capture (without graphs,
   CLUSTER_SIM_COMPILE=vision,lm,expert compiles those modules).
8. Fused back end (CLUSTER_SIM_FUSE23=1): stage 2 runs inside the stage-3 batch, no hand-off through the request
   thread (section 8 below).
9. Cache mode, CLUSTER_SIM_LEAN=1: the logging-only corrector audit (a second distance pass per query) is skipped.
7. Cache mode (CLUSTER_SIM_KEYS=1): CP1 keys computed per stage-1 batch with one D2H; no device-wide sync in the
   plugin's stage-1 counter. See section 7 below.
Stage timers (SystemTimer) are off in both modes.
"""
import sys

sys.path.pop(0)  # this directory; imports resolve through the server tree's PYTHONPATH

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
import dataclasses
import gc
import itertools
import json
import logging
import os
from pathlib import Path
import queue
import runpy
import threading
import time

import torch
from torch.utils import _pytree
from websockets.asyncio.connection import Connection

LOG = Path(os.environ['CLUSTER_SIM_LOG'])
LOG.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger('cluster_sim')


def per_stage(name, default):
    out = {1: default, 2: default, 3: default}
    for item in filter(None, os.environ.get(name, '').split(',')):
        stage, value = item.split(':')
        out[int(stage)] = float(value)
    return out


# 5. memory --------------------------------------------------------------------------------------------------------
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ.get('CLUSTER_SIM_VRAM_GB', '24')) * 2**30 / total), 0)

# 1. request timing ------------------------------------------------------------------------------------------------
_rows, _rows_lock, _conn_ids = [], threading.Lock(), itertools.count()
_recv, _send = Connection.recv, Connection.send


_frozen = []


async def recv(self, *args, **kwargs):
    data = await _recv(self, *args, **kwargs)
    if not _frozen and isinstance(data, (bytes, bytearray)) and data.find(b'__ctrl__', 0, 64) < 0:
        _frozen.append(True)
        gc.freeze()  # first infer: everything built at startup and preload leaves the collector's generations
    if isinstance(data, (bytes, bytearray)):
        if not hasattr(self, '_cs_id'):
            self._cs_id = next(_conn_ids)
        self._cs_pending = (time.time(), 'ctrl' if data.find(b'__ctrl__', 0, 64) >= 0 else 'infer')
    return data


async def send(self, *args, **kwargs):
    result = await _send(self, *args, **kwargs)
    pending = getattr(self, '_cs_pending', None)
    if pending is not None:
        self._cs_pending = None
        with _rows_lock:
            _rows.append(f'{self._cs_id},{pending[1]},{pending[0]:.6f},{time.time():.6f}\n')
    return result


Connection.recv, Connection.send = recv, send
# The server runs a full gc.collect() on every connection close (and torch.cuda.graph before each capture); with this
# heap one takes seconds while holding the GIL, stalling every other request. The generational collector still runs.
gc.collect = lambda generation=2: 0

# 2. batching ------------------------------------------------------------------------------------------------------
from openpi.serving import batching_coordinator as _bc  # noqa: E402

BATCH = {s: int(v) for s, v in per_stage('CLUSTER_SIM_BATCH', 32).items()}
WAIT = {s: v / 1000.0 for s, v in per_stage('CLUSTER_SIM_WAIT_MS', 0).items()}
_stage_counts = {s: [0, 0, 0.0] for s in (1, 2, 3)}  # batches, requests, busy seconds
_stage_sizes = {s: {} for s in (1, 2, 3)}  # batch size -> [batches, busy seconds]
_batch_rows = []  # stage,t_start,t_end,size,oldest_wait_s -> batches_<pid>.csv


class Gate:
    """Stage batches run side by side (shared); a CUDA graph capture runs with no other stage on the GPU."""

    def __init__(self):
        self.cv, self.active, self.exclusive, self.waiting = threading.Condition(), 0, False, 0

    def enter(self):
        with self.cv:
            self.cv.wait_for(lambda: not self.exclusive and not self.waiting)
            self.active += 1

    def leave(self):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    @contextlib.contextmanager
    def alone(self):  # called from inside a stage batch
        self.leave()
        with self.cv:
            self.waiting += 1
            self.cv.wait_for(lambda: not self.exclusive and self.active == 0)
            self.waiting -= 1
            self.exclusive = True
        try:
            torch.cuda.synchronize()
            yield
        finally:
            with self.cv:
                self.exclusive = False
                self.active += 1  # back in the batch it came from
                self.cv.notify_all()


_gate = Gate()


class Unit:
    """Stages of one unit take the GPU one batch at a time; the waiting stage furthest along goes first (3 > 2 > 1),
    so in-flight requests finish before new ones start."""

    def __init__(self):
        self.cv, self.busy, self.waiting = threading.Condition(), False, {1: 0, 2: 0, 3: 0}

    @contextlib.contextmanager
    def turn(self, stage):
        with self.cv:
            self.waiting[stage] += 1
            self.cv.wait_for(lambda: not self.busy and not any(self.waiting[s] for s in (1, 2, 3) if s > stage))
            self.waiting[stage] -= 1
            self.busy = True
        try:
            yield
        finally:
            with self.cv:
                self.busy = False
                self.cv.notify_all()


# CLUSTER_SIM_UNITS: stage groups that share the GPU exclusively, e.g. "1,2,3" (all side by side), "1,23", "123".
UNITS = {}
for group in os.environ.get('CLUSTER_SIM_UNITS', '1,2,3').split(','):
    unit = Unit() if len(group) > 1 else None
    UNITS.update({int(c): unit for c in group})


class Coordinator(_bc.BatchingCoordinator):
    def __init__(self, model, **kwargs):
        kwargs.update(max_batch_size=max(BATCH.values()), max_wait_ms=1000.0 * max(WAIT.values()))
        super().__init__(model, **kwargs)
        # CLUSTER_SIM_PRIORITY ("1:0,3:-3"): CUDA stream priority per stage (lower runs first; 4090 range 0..-3).
        for stage, priority in per_stage('CLUSTER_SIM_PRIORITY', 0).items() if os.environ.get('CLUSTER_SIM_PRIORITY') else ():
            for key in [k for k in self._worker_streams if k[0] == stage]:
                self._worker_streams[key] = torch.cuda.Stream(device=self._device, priority=int(priority))

    def _stage_loop_inner(self, stage_id, q, stream=None):
        size, wait = BATCH[stage_id], WAIT[stage_id]
        while self.fatal_error is None and (not self._stop.is_set() or not q.empty()):
            try:
                first = q.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + wait
            while len(batch) < size:
                try:
                    batch.append(q.get_nowait())
                    continue
                except queue.Empty:
                    pass
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(q.get(timeout=remaining))
                except queue.Empty:
                    break
            unit = UNITS[stage_id]
            with unit.turn(stage_id) if unit is not None else contextlib.nullcontext():
                while len(batch) < size:  # whatever arrived while waiting for the unit
                    try:
                        batch.append(q.get_nowait())
                    except queue.Empty:
                        break
                t0 = time.monotonic()
                wall = time.time()
                oldest = t0 - min(r.enqueue_t for r in batch)
                _gate.enter()
                try:
                    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                        self._run_batch(stage_id, batch)
                except BaseException as exc:
                    logger.exception('stage %d batch failed', stage_id)
                    self._fail_batch(batch, exc, stream)
                finally:
                    _gate.leave()
            busy = time.monotonic() - t0
            with _rows_lock:
                _batch_rows.append(f'{stage_id},{wall:.6f},{wall + busy:.6f},{len(batch)},{oldest:.6f}\n')
            counts = _stage_counts[stage_id]
            counts[0] += 1
            counts[1] += len(batch)
            counts[2] += busy
            by_size = _stage_sizes[stage_id].setdefault(len(batch), [0, 0.0])
            by_size[0] += 1
            by_size[1] += busy
            for req in batch:
                req.payload = None
            batch.clear()


_bc.BatchingCoordinator = Coordinator

# 3. prefix trimming -----------------------------------------------------------------------------------------------
from openpi.models_pytorch import pi0_pytorch as _pi0  # noqa: E402

TEXT_TOKENS = int(os.environ.get('CLUSTER_SIM_TEXT_TOKENS', '80'))
GRAPHS = os.environ.get('CLUSTER_SIM_GRAPHS') == '1'
_embed_prefix = _pi0.PI0Pytorch.embed_prefix


_keep = []


def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    # With CUDA graphs the camera set is fixed by the first (eager) call: traced graphs hold no host syncs.
    if not (GRAPHS and _keep):
        _keep[:] = [i for i, m in enumerate(img_masks) if bool(m.any())]
        if bool(lang_masks[:, TEXT_TOKENS:].any()):
            raise ValueError(f'prompt longer than CLUSTER_SIM_TEXT_TOKENS={TEXT_TOKENS}')
    return _embed_prefix(self, [images[i] for i in _keep], [img_masks[i] for i in _keep],
                         lang_tokens[:, :TEXT_TOKENS], lang_masks[:, :TEXT_TOKENS])


if TEXT_TOKENS > 0:
    _pi0.PI0Pytorch.embed_prefix = embed_prefix

# 4. request threads -----------------------------------------------------------------------------------------------
from openpi.serving import websocket_policy_server as _wps  # noqa: E402

_run = _wps.WebsocketPolicyServer.run


async def run(self, *args, **kwargs):
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=int(os.environ.get('CLUSTER_SIM_THREADS', '512')), thread_name_prefix='request'))
    return await _run(self, *args, **kwargs)


_wps.WebsocketPolicyServer.run = run


# 6. CUDA graphs and optional torch.compile, the same for both modes ------------------------------------------------
COMPILE = [c for c in os.environ.get('CLUSTER_SIM_COMPILE', '').split(',') if c]
BUCKETS = sorted(int(b) for b in os.environ.get('CLUSTER_SIM_BUCKETS', '1,2,4,8,12,16,24,32,48,64').split(','))


class Graphed:
    """CUDA graphs of one batch-first tensor function, one per padded batch size, replayed on the caller's stream.

    Each graph keeps static inputs and outputs; a call copies its rows into the inputs, replays, and clones the first
    n rows of the outputs (in stream order, before the next replay can overwrite them). Graphs of one function share
    a memory pool; the function is only ever called from one stage thread."""

    def __init__(self, name, fn, limit):
        self.name, self.fn, self.graphs, self.pool = name, fn, {}, torch.cuda.graph_pool_handle()
        self.sizes = [b for b in BUCKETS if b < limit] + [limit]

    def __call__(self, *args):
        leaves, spec = _pytree.tree_flatten(args)  # None fields stay None leaves
        n = next(x for x in leaves if isinstance(x, torch.Tensor)).shape[0]
        if not self.graphs:  # first call: every size at once, from this call's rows, with the GPU to ourselves
            with _gate.alone():
                for size in self.sizes:
                    self.graphs[size] = self.capture(leaves, spec, n, size)
        size = next(b for b in self.sizes if b >= n)
        inputs, outputs, graph, _ = self.graphs[size]
        for static, x in zip(inputs, leaves):
            if isinstance(x, torch.Tensor):
                static[:n].copy_(x)
        graph.replay()
        return _pytree.tree_map(lambda t: t[:n].clone(), outputs)

    def capture(self, leaves, spec, n, size):
        inputs = [x if not isinstance(x, torch.Tensor) else
                  torch.cat([x, x[-1:].expand(size - n, *x.shape[1:])]) if size > n else x[:size].clone() for x in leaves]
        args = _pytree.tree_unflatten(inputs, spec)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self.fn(*args)  # eager first: fixes the camera set before tracing
            fn = trace(self.fn, args)
            for _ in range(3):
                fn(*args)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, stream=side, capture_error_mode='thread_local'):
            outputs = fn(*args)
        logger.info('captured %s batch %d', self.name, size)
        return inputs, outputs, graph, fn  # fn holds the traced graph's GPU constants the capture reads


def trace(fn, args):
    """FX-trace one stage on real inputs (as exp/r17_integration/profile/latency.py does): Python structure and
    host-side tensors (e.g. torch.tensor(list) masks) become graph constants on the GPU, so the graph can be captured.
    With CLUSTER_SIM_COMPILE set the traced graph is compiled by inductor (our own CUDA graph on top)."""
    from torch.fx.experimental.proxy_tensor import make_fx
    graph = make_fx(fn, tracing_mode='real', _error_on_data_dependent_ops=False)(*args)
    device = torch.device('cuda', torch.cuda.current_device())
    for node in graph.graph.nodes:
        value = getattr(graph, node.target, None) if node.op == 'get_attr' else None
        if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
            setattr(graph, node.target, value.to(device))
        target = node.kwargs.get('device') if node.op == 'call_function' else None
        if isinstance(target, torch.device) and target.type == 'cpu':
            node.kwargs = {**node.kwargs, 'device': device}
        if node.target == torch.ops.aten._local_scalar_dense.default:
            raise RuntimeError('host sync left in a traced stage')
    graph.graph.eliminate_dead_code()
    graph.recompile()
    if COMPILE:
        return torch.compile(graph, mode=os.environ.get('CLUSTER_SIM_COMPILE_MODE', 'max-autotune-no-cudagraphs'),
                             fullgraph=True, dynamic=False)
    return graph


def accelerate(model):
    from transformers import DynamicCache
    from openpi.models import model as _model

    pwe = model.paligemma_with_expert
    if not GRAPHS:
        opts = dict(dynamic=True, mode=os.environ.get('CLUSTER_SIM_COMPILE_MODE', 'max-autotune-no-cudagraphs'))
        if 'vision' in COMPILE:
            pwe.embed_image = torch.compile(pwe.embed_image, **opts)
        if 'lm' in COMPILE:
            pwe.paligemma.language_model = torch.compile(pwe.paligemma.language_model, **opts)
        if 'expert' in COMPILE:
            pwe.gemma_expert.model = torch.compile(pwe.gemma_expert.model, **opts)
        return
    torch._dynamo.config.cache_size_limit = 256

    stage1 = Graphed('stage1', lambda fields: model.__class__._stage1_token_prep(model, _model.Observation(**fields)),
                     BATCH[1])
    model._stage1_token_prep = lambda obs: stage1({f.name: getattr(obs, f.name) for f in dataclasses.fields(obs)})

    def backbone(embs, pad, att, pos):
        return tuple(model.__class__._stage2_llm_backbone(model, embs, pad, att, pos))
    stage2 = Graphed('stage2', backbone, BATCH[2])
    model._stage2_llm_backbone = lambda *a: DynamicCache.from_legacy_cache(stage2(*a))

    def backbone_capture(embs, pad, att, pos):
        out, kv = pwe.forward(attention_mask=att, position_ids=pos, past_key_values=None,
                              inputs_embeds=[embs, None], use_cache=True)
        return out[0], tuple(kv)
    stage2c = Graphed('stage2_capture', backbone_capture, BATCH[2])

    def run_stage2_capture(s1):
        out, kv = stage2c(s1.prefix_embs, s1.prefix_pad_masks, s1.prefix_att_2d_masks_4d, s1.prefix_position_ids)
        return _pi0.Stage2Output(stage1=s1, past_key_values=DynamicCache.from_legacy_cache(kv), prefix_out=out)
    model.run_stage2_capture = run_stage2_capture

    def step(state, pad, kv, x, t):
        return model.__class__.denoise_step(model, state, pad, DynamicCache.from_legacy_cache(kv), x, t)
    denoise = Graphed('denoise', step, BATCH[3])
    model.denoise_step = lambda state, pad, kv, x, t: denoise(state, pad, tuple(kv), x, t.contiguous())


if COMPILE or GRAPHS:
    from openpi.policies import policy_config as _pc  # noqa: E402

    _create = _pc.create_trained_policy

    def create_trained_policy(*args, **kwargs):
        policy = _create(*args, **kwargs)
        accelerate(policy._model)
        logger.info('cluster_sim: graphs=%s compile=%s buckets=%s', GRAPHS, COMPILE, BUCKETS)
        return policy

    _pc.create_trained_policy = create_trained_policy

# 8. Fused back end (CLUSTER_SIM_FUSE23=1, both modes): stage 2 and stage 3 of a request run as one batch on the stage-3
# worker. Submitting to stage 2 returns at once with a placeholder Stage2Output (the interceptor only passes it on to
# stage 3); the stage-3 batch first runs stage 2 batched for its placeholders, then denoises. This removes the
# stage-2 -> request thread -> stage-3 hand-off, which split each stage-2 batch into many small stage-3 batches.
if os.environ.get('CLUSTER_SIM_FUSE23') == '1':
    from openpi.serving import batching_coordinator as _bc23

    _submit = _bc.BatchingCoordinator.submit_to_stage
    _miss = _bc23.Pi05StageBatcher.run_stage3_miss

    def submit_to_stage(self, stage_id, bundle_id, payload, *, requires_stage2_capture=False, **kwargs):
        if stage_id == 2 and not requires_stage2_capture:
            out = _pi0.Stage2Output(stage1=payload, past_key_values=None)
            out.cs_pending = True
            return out
        return _submit(self, stage_id, bundle_id, payload, requires_stage2_capture=requires_stage2_capture, **kwargs)

    def run_stage3_miss(self, payloads, **kwargs):
        pending = [p for p in payloads if getattr(p.stage2_out, 'cs_pending', False)]
        if pending:
            for p, real in zip(pending, self.run_stage2_batch([p.stage2_out.stage1 for p in pending], capture=False)):
                p.stage2_out = real
        return _miss(self, payloads, **kwargs)

    _bc.BatchingCoordinator.submit_to_stage = submit_to_stage
    _bc23.Pi05StageBatcher.run_stage3_miss = run_stage3_miss

# 9. Cache mode, CLUSTER_SIM_LEAN=1: R14Knob._diagnostics runs original_corrector_audit on every query, a second
# donor-distance pass (base._dist) only for its log record; it never feeds a decision. Lean mode logs zeros instead.
if os.environ.get('CLUSTER_SIM_LEAN') == '1':
    from exp.offline_search.rounds.r14.knob import recipe as _r14

    _r14.original_corrector_audit = lambda base, q, res: dict(
        audit_reconstruction_ms=0., corrector=dict(distance=0., gate=0., motion_rms=0., norm=0., blend=.5))

# 7. Cache mode only (CLUSTER_SIM_KEYS=1): CP1 keys for a whole stage-1 batch with one D2H copy, and no device-wide sync
# around stage 1. The keys are the CP1SpatialPool16 keys (vision_0/1: 16x16 -> 4x4 pooled tokens, robot_state), the
# same values the key builder computed per request with three synchronous .cpu() copies on the request thread; the
# plugin's stage-1 counter called torch.cuda.synchronize(), which made every request wait for all GPU work (stage 1
# results are already complete when the coordinator publishes them).
if os.environ.get('CLUSTER_SIM_KEYS') == '1':
    import inspect
    import textwrap

    import torch.nn.functional as F
    from openpi.cache.components import key_builder as _kb
    from openpi.models import model as _obs_model
    from openpi.serving import batching_coordinator as _bco
    from openpi.serving import stage_io as _sio

    def cp1_keys(out):
        b = out.prefix_embs.shape[0]
        tokens = out.prefix_embs[:, :512].reshape(b * 2, 16, 16, -1).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(tokens, (4, 4)).permute(0, 2, 3, 1).reshape(b, 2, -1).float()
        vision = torch.empty(pooled.shape, dtype=torch.float32, pin_memory=True)
        vision.copy_(pooled, non_blocking=True)
        state = torch.empty(out.state.shape, dtype=torch.float32, pin_memory=True)
        state.copy_(out.state.float(), non_blocking=True)
        return vision, state  # ready once the stage stream is synchronized, before the batch is published

    def run_stage1_batch(self, payloads):
        batched = _sio.stack_observation(payloads, device=self._device)
        if isinstance(batched, dict) and 'image' in batched:
            batched = _obs_model.Observation.from_dict(batched)
        out = self._model.run_stage1(batched)
        shards = _sio.split_stage1_output(out, len(payloads))
        vision, state = cp1_keys(out)
        for i, shard in enumerate(shards):
            shard.cs_keys = {_kb.VISION_0: vision[i, 0], _kb.VISION_1: vision[i, 1], _kb.ROBOT_STATE: state[i]}
        return shards

    _bco.Pi05StageBatcher.run_stage1_batch = run_stage1_batch
    _collect, _build = _kb._CP1BaseKeyBuilder.collect, _kb._CP1BaseKeyBuilder.build
    _fields = {_kb.VISION_0, _kb.VISION_1, _kb.ROBOT_STATE}

    def collect(self, checkpoint_id, **stage_outputs):
        _collect(self, checkpoint_id, **stage_outputs)
        keys = getattr(stage_outputs.get('stage1'), 'cs_keys', None)
        if keys is not None:
            self._cache['cs_keys'] = keys

    def build(self, checkpoint_id):
        keys = self._cache.get('cs_keys')
        if keys is None or type(self) is not _kb.CP1SpatialPool16KeyBuilder or self._enabled is None \
                or not self._enabled <= _fields:
            return _build(self, checkpoint_id)
        return {k: v for k, v in keys.items() if k in self._enabled}

    _kb._CP1BaseKeyBuilder.collect, _kb._CP1BaseKeyBuilder.build = collect, build

    from exp.offline_search.closed_loop import plugin as _plugin

    source = textwrap.dedent(inspect.getsource(_plugin._BlindAdapter._instrument))
    namespace = {}
    exec(compile(source.replace('torch.cuda.synchronize()', 'pass'), _plugin.__file__, 'exec'), _plugin.__dict__, namespace)
    _plugin._BlindAdapter._instrument = namespace['_instrument']

# Stage timers off in both modes (the cache yaml already sets timer.enabled false; --cache would enable them).
from openpi.cache import timing as _timing  # noqa: E402

_timer_init = _timing.SystemTimer.__init__


def timer_init(self, enabled=True, *args, **kwargs):
    _timer_init(self, False, *args, **kwargs)


_timing.SystemTimer.__init__ = timer_init


# log writer -------------------------------------------------------------------------------------------------------
def flush():
    path = LOG / f'requests_{os.getpid()}.csv'
    while True:
        time.sleep(2)
        with _rows_lock:
            rows, batches = _rows[:], _batch_rows[:]
            del _rows[:], _batch_rows[:]
        if rows:
            with path.open('a') as f:
                f.writelines(rows)
        if batches:
            with (LOG / f'batches_{os.getpid()}.csv').open('a') as f:
                f.writelines(batches)
        (LOG / f'stages_{os.getpid()}.json').write_text(json.dumps(dict(
            batches={s: c[0] for s, c in _stage_counts.items()}, requests={s: c[1] for s, c in _stage_counts.items()},
            busy_s={s: round(c[2], 3) for s, c in _stage_counts.items()},
            by_size={s: {n: [b, round(1000 * t / b, 2)] for n, (b, t) in sorted(list(d.items()))} for s, d in _stage_sizes.items()},
            batch_limit=BATCH, wait_s=WAIT, text_tokens=TEXT_TOKENS, time=time.time())) + '\n')


if __name__ == '__main__':
    threading.Thread(target=flush, daemon=True).start()
    runpy.run_module(os.environ['CLUSTER_SIM_MODULE'], run_name='__main__', alter_sys=True)
