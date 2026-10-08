"""Local stage RPC: small msgpack envelopes, tensors in anonymous shared memory.

Linux memfd descriptors are passed once per channel with SCM_RIGHTS. Neither
tensor bytes nor numpy arrays ever go through pickle or the socket. Both peers
copy before replying/reusing a slot; a crashed peer leaves no named shm files.
This module has no torch/JAX import until a tensor is encountered.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
import importlib
import mmap
import os
import socket
import struct
import threading
from typing import Any

import msgpack
import numpy as np

MAX_ENVELOPE = 1 << 20
DEFAULT_CAPACITY = 32 << 20


class StageRPCError(RuntimeError):
    pass


class BackpressureError(StageRPCError):
    pass


def _read(sock, count):
    chunks = bytearray()
    while len(chunks) < count:
        part = sock.recv(count - len(chunks))
        if not part:
            raise EOFError("stage peer disconnected")
        chunks.extend(part)
    return chunks


def send_frame(sock, value):
    data = msgpack.packb(value, use_bin_type=True)
    if len(data) > MAX_ENVELOPE:
        raise ValueError("stage envelope exceeds 1 MiB")
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_frame(sock):
    size = struct.unpack("!I", _read(sock, 4))[0]
    if size > MAX_ENVELOPE:
        raise ValueError("stage envelope exceeds 1 MiB")
    return msgpack.unpackb(_read(sock, size), raw=False, strict_map_key=False)


class Arena:
    def __init__(self, fd, capacity):
        if capacity < 1 or os.fstat(fd).st_size != capacity * 2:
            raise ValueError("invalid shared arena size")
        self.capacity = capacity
        self.memory = mmap.mmap(fd, capacity * 2)

    @classmethod
    def create(cls, sock, capacity=DEFAULT_CAPACITY):
        fd = os.memfd_create("openpi-stage", os.MFD_CLOEXEC)
        try:
            os.ftruncate(fd, capacity * 2)
            arena = cls(fd, capacity)
            sock.sendmsg([struct.pack("!Q", capacity)],
                         [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))])
            return arena
        finally:
            os.close(fd)

    @classmethod
    def receive(cls, sock):
        data, ancillary, flags, _ = sock.recvmsg(8, socket.CMSG_SPACE(struct.calcsize("i")))
        if flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC) or len(data) != 8 or len(ancillary) != 1:
            raise ValueError("invalid shared arena handshake")
        level, kind, fds = ancillary[0]
        if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS or len(fds) != struct.calcsize("i"):
            raise ValueError("missing shared arena descriptor")
        fd = struct.unpack("i", fds)[0]
        try:
            return cls(fd, struct.unpack("!Q", data)[0])
        finally:
            os.close(fd)

    def close(self):
        self.memory.close()

    def encode(self, value, slot=0, *, references=True):
        offset = slot * self.capacity
        end = offset + self.capacity

        def array(a, tag, dtype):
            nonlocal offset
            a = np.ascontiguousarray(a)
            offset = (offset + 63) & ~63
            if offset + a.nbytes > end:
                raise BackpressureError(f"stage payload exceeds {self.capacity} byte arena")
            pos = offset
            if a.nbytes:
                self.memory[pos:pos + a.nbytes] = memoryview(a).cast("B")
            offset += a.nbytes
            return [tag, dtype, list(a.shape), pos, a.nbytes]

        def enc(v):
            if v is None or isinstance(v, (bool, str, int, float)):
                return ["scalar", v]
            if isinstance(v, np.generic):
                return enc(v.item())
            if isinstance(v, np.ndarray):
                if v.dtype.hasobject:
                    raise TypeError("object arrays are not stage payloads")
                return array(v, "numpy", v.dtype.str)
            if type(v).__module__.startswith("torch"):
                import torch
                if isinstance(v, torch.dtype):
                    return ["dtype", str(v).removeprefix("torch.")]
                if isinstance(v, torch.Tensor):
                    cpu = v.detach().cpu().contiguous()
                    # uint8 preserves bf16 exactly (numpy cannot represent it).
                    raw = cpu.reshape(-1).view(torch.uint8).numpy()
                    item = array(raw, "tensor", str(cpu.dtype).removeprefix("torch."))
                    item[2] = list(cpu.shape)
                    return item
            ref = getattr(v, "_split_ref", None)
            if ref is not None and references:
                return ["ref", ref]
            if dataclasses.is_dataclass(v):
                return ["object", type(v).__module__, type(v).__qualname__,
                        [[k, enc(x)] for k, x in vars(v).items()]]
            if isinstance(v, Mapping):
                return ["dict", [[enc(k), enc(x)] for k, x in v.items()]]
            if isinstance(v, (list, tuple)):
                return ["tuple" if isinstance(v, tuple) else "list", [enc(x) for x in v]]
            if isinstance(v, bytes):
                if len(v) > MAX_ENVELOPE // 2:
                    return array(np.frombuffer(v, np.uint8), "bytes", "|u1")
                return ["scalar", v]
            raise TypeError(f"unsupported stage payload {type(v).__module__}.{type(v).__qualname__}")

        return enc(value)

    def decode(self, envelope, slot=0, resolve=None):
        start, end = slot * self.capacity, (slot + 1) * self.capacity

        def dec(item):
            tag = item[0]
            if tag == "scalar":
                return item[1]
            if tag in ("numpy", "tensor", "bytes"):
                _, dtype, shape, offset, size = item
                if size < 0 or offset < start or offset + size > end or any(n < 0 for n in shape):
                    raise ValueError("invalid stage array bounds")
                if tag == "tensor":
                    import torch
                    dt = getattr(torch, dtype)
                    expected = int(np.prod(shape, dtype=np.int64)) * torch.empty((), dtype=dt).element_size()
                    if expected != size:
                        raise ValueError("invalid tensor size")
                    raw = np.frombuffer(self.memory, np.uint8, count=size, offset=offset).copy()
                    return torch.from_numpy(raw).view(dt).reshape(shape)
                dt = np.dtype(dtype)
                if dt.hasobject or int(np.prod(shape, dtype=np.int64)) * dt.itemsize != size:
                    raise ValueError("invalid numpy size")
                a = np.ndarray(shape, dtype=dt, buffer=self.memory, offset=offset).copy()
                return a.tobytes() if tag == "bytes" else a
            if tag == "dtype":
                import torch
                return getattr(torch, item[1])
            if tag == "ref":
                if resolve is None:
                    raise ValueError("unexpected remote reference")
                return resolve(item[1])
            if tag == "dict":
                return {dec(k): dec(v) for k, v in item[1]}
            if tag in ("list", "tuple"):
                values = [dec(v) for v in item[1]]
                return tuple(values) if tag == "tuple" else values
            if tag == "object":
                _, module, name, values = item
                if not module.startswith(("openpi.", "exp.", "tests.")):
                    raise ValueError(f"unsupported stage class {module}.{name}")
                cls = importlib.import_module(module)
                for part in name.split("."):
                    cls = getattr(cls, part)
                obj = object.__new__(cls)
                for key, value in values:
                    object.__setattr__(obj, key, dec(value))
                return obj
            raise ValueError(f"unknown stage tag {tag}")

        return dec(envelope)


class Channel:
    """One in-flight call per channel. Slots are never reused after a timeout."""
    def __init__(self, sock, *, capacity=DEFAULT_CAPACITY, timeout=60.0):
        self.sock = sock
        self.sock.settimeout(timeout)
        self.arena = Arena.create(sock, capacity)
        self.lock = threading.Lock()
        self.closed = False

    def call(self, operation, args=None):
        with self.lock:
            if self.closed:
                raise StageRPCError("stage channel closed; reconnect at next decision")
            try:
                envelope = self.arena.encode(args)
                send_frame(self.sock, [operation, envelope])
                ok, response = recv_frame(self.sock)
                if not ok:
                    error, message = response
                    cls = BackpressureError if error == "BackpressureError" else StageRPCError
                    raise cls(message)
                return self.arena.decode(response, slot=1)
            except BackpressureError:
                raise
            except StageRPCError:
                raise
            except (OSError, EOFError, ValueError) as exc:
                self.close()
                raise StageRPCError(f"stage IPC failed: {exc}") from exc

    def close(self):
        if not self.closed:
            self.closed = True
            self.sock.close()
            self.arena.close()
