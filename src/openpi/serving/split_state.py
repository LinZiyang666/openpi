"""Small process-shared plugin counters; no model or per-connection data.

Preserves the original server-wide decision/connection clock and quantile
window across fronts. flock guards the same short transactions that the
ordinary plugin guards with its RLock. No lock spans retrieval or inference.
"""
from __future__ import annotations

import collections
import dataclasses
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import threading


class ProcessRLock:
    def __init__(self, fd):
        self.fd = fd
        self.local = threading.RLock()
        self.depth = 0

    def __enter__(self):
        self.local.acquire()
        if self.depth == 0:
            fcntl.flock(self.fd, fcntl.LOCK_EX)
        self.depth += 1
        return self

    def __exit__(self, *args):
        self.depth -= 1
        if self.depth == 0:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.local.release()


class SharedState:
    def __init__(self, path, window):
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        self.lock = ProcessRLock(self.fd)
        # Preserve both queue and exact bisect order, including NaN behavior.
        self.window = window
        size = 32 + window * 16
        with self.lock:
            if os.fstat(self.fd).st_size == 0:
                os.ftruncate(self.fd, size)
            elif os.fstat(self.fd).st_size != size:
                raise ValueError("shared plugin state configuration mismatch")
            self.mem = mmap.mmap(self.fd, size)

    def get(self, index):
        with self.lock:
            return struct.unpack_from("<Q", self.mem, index * 8)[0]

    def set(self, index, value):
        with self.lock:
            struct.pack_into("<Q", self.mem, index * 8, value)

    def next(self, index):
        with self.lock:
            value = self.get(index)
            self.set(index, value + 1)
            return value


class SharedConnectionIds:
    def __init__(self, state):
        self.state = state

    def __next__(self):
        return self.state.next(1)


class SharedQuantile:
    def __init__(self, inner, state):
        self.inner, self.shared = inner, state

    def _load(self):
        inner, state = self.inner, self.shared
        n = state.get(3)
        values = struct.unpack_from(f"<{n}d", state.mem, 32)
        inner._q = collections.deque(values)
        inner._sorted = list(struct.unpack_from(f"<{n}d", state.mem, 32 + state.window * 8))
        inner.n_seen = state.get(2)

    def tau(self):
        with self.shared.lock:
            self._load()
            return self.inner.tau()

    def push(self, value):
        with self.shared.lock:
            self._load()
            self.inner.push(value)
            values = tuple(self.inner._q)
            struct.pack_into(f"<{len(values)}d", self.shared.mem, 32, *values)
            struct.pack_into(f"<{len(values)}d", self.shared.mem, 32 + self.shared.window * 8,
                             *self.inner._sorted)
            self.shared.set(3, len(values))
            self.shared.set(2, self.inner.n_seen)

    def state(self):
        with self.shared.lock:
            self._load()
            return self.inner.state()


def bind_runtime(runtime, root, identity=None):
    from exp.offline_search.closed_loop.plugin import PluginRuntime

    if identity is None:
        identity = {k: v for k, v in vars(runtime.opts).items() if k != "os_log_dir"}
    def encode(value):
        if hasattr(value, "as_dict"):
            return value.as_dict()
        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"unstable split runtime identity: {type(value)}")
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=encode).encode()).hexdigest()
    Path(root).mkdir(parents=True, exist_ok=True)
    state = SharedState(Path(root) / digest, runtime.ctrl.W if runtime.ctrl else 0)

    class SharedRuntime(PluginRuntime):
        @property
        def decision_count(self):
            return self._split_state.get(0)

        @decision_count.setter
        def decision_count(self, value):
            self._split_state.set(0, value)

    runtime._split_state = state
    runtime.__class__ = SharedRuntime
    runtime.decision_lock = state.lock
    runtime._conn_ids = SharedConnectionIds(state)
    if runtime.ctrl is not None:
        runtime.ctrl = SharedQuantile(runtime.ctrl, state)
    return runtime
