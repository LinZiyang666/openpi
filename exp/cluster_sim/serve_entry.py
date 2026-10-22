"""Single-copy π0.5 server process for the cluster simulation (run by serve.py inside the R16 server tree).

Runs $CLUSTER_SIM_MODULE (the R16 resident for the cache system, scripts.serve_policy --cache for pure inference)
unchanged, after five process-local changes that both modes share:

1. Request timing: every frame's arrival (after recv) and its reply (after send) per connection, appended to
   $CLUSTER_SIM_LOG/requests_<pid>.csv as `conn,kind,t_recv,t_sent` (kind = infer / ctrl).
2. Batching: per-stage batch limit and wait (CLUSTER_SIM_BATCH / CLUSTER_SIM_WAIT_MS, e.g. "1:32,2:16,3:16");
   a stage worker takes its first request, drains everything already queued, then waits up to its wait.
   Stage worker counts stay BATCHING_STAGE{1,2,3}_WORKERS. Per-stage batch counts go to stages_<pid>.json.
3. Prefix trimming: cameras whose mask is all False (LIBERO's empty third view) are not encoded, and the prompt
   is cut to CLUSTER_SIM_TEXT_TOKENS. Masked tokens take no position (cumsum of the pad mask) and no attention,
   so the computation is the same; only the GEMM shapes shrink (968 -> 512 + text tokens).
4. Request threads: the asyncio default executor gets CLUSTER_SIM_THREADS threads (default 32 otherwise).
5. Memory: the process may allocate at most CLUSTER_SIM_VRAM_GB of the card.
Stage timers (SystemTimer) are off in both modes.
"""
import sys

sys.path.pop(0)  # this directory; imports resolve through the server tree's PYTHONPATH

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
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


async def recv(self, *args, **kwargs):
    data = await _recv(self, *args, **kwargs)
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

# 2. batching ------------------------------------------------------------------------------------------------------
from openpi.serving import batching_coordinator as _bc  # noqa: E402

BATCH = {s: int(v) for s, v in per_stage('CLUSTER_SIM_BATCH', 32).items()}
WAIT = {s: v / 1000.0 for s, v in per_stage('CLUSTER_SIM_WAIT_MS', 0).items()}
_stage_counts = {s: [0, 0] for s in (1, 2, 3)}  # batches, requests


class Coordinator(_bc.BatchingCoordinator):
    def __init__(self, model, **kwargs):
        kwargs.update(max_batch_size=max(BATCH.values()), max_wait_ms=1000.0 * max(WAIT.values()))
        super().__init__(model, **kwargs)

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
            counts = _stage_counts[stage_id]
            counts[0] += 1
            counts[1] += len(batch)
            try:
                with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                    self._run_batch(stage_id, batch)
            except BaseException as exc:
                logger.exception('stage %d batch failed', stage_id)
                self._fail_batch(batch, exc, stream)
            for req in batch:
                req.payload = None
            batch.clear()


_bc.BatchingCoordinator = Coordinator

# 3. prefix trimming -----------------------------------------------------------------------------------------------
from openpi.models_pytorch import pi0_pytorch as _pi0  # noqa: E402

TEXT_TOKENS = int(os.environ.get('CLUSTER_SIM_TEXT_TOKENS', '80'))
_embed_prefix = _pi0.PI0Pytorch.embed_prefix


def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    keep = [i for i, m in enumerate(img_masks) if bool(m.any())]
    if bool(lang_masks[:, TEXT_TOKENS:].any()):
        raise ValueError(f'prompt longer than CLUSTER_SIM_TEXT_TOKENS={TEXT_TOKENS}')
    return _embed_prefix(self, [images[i] for i in keep], [img_masks[i] for i in keep],
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
            rows = _rows[:]
            del _rows[:]
        if rows:
            with path.open('a') as f:
                f.writelines(rows)
        (LOG / f'stages_{os.getpid()}.json').write_text(json.dumps(dict(
            batches={s: c[0] for s, c in _stage_counts.items()}, requests={s: c[1] for s, c in _stage_counts.items()},
            batch_limit=BATCH, wait_s=WAIT, text_tokens=TEXT_TOKENS, time=time.time())) + '\n')


threading.Thread(target=flush, daemon=True).start()
runpy.run_module(os.environ['CLUSTER_SIM_MODULE'], run_name='__main__', alter_sys=True)
