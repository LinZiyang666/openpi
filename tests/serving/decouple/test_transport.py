import socket
import threading
import time

import numpy as np
import pytest
import torch

from openpi.serving.split_service import StageService
from openpi.serving.split_transport import Arena, BackpressureError, Channel, StageRPCError


def test_arena_bit_exact_and_no_tensor_bytes_in_envelope():
    left, right = socket.socketpair()
    try:
        sender = Arena.create(left, 1 << 20)
        receiver = Arena.receive(right)
        values = {"bf16": torch.arange(100).to(torch.bfloat16).reshape(10, 10),
                  "pixels": np.arange(96, dtype=np.uint8).reshape(4, 8, 3),
                  "empty": np.zeros((0, 2)), "view": np.arange(20)[::2],
                  "inter": {0.7: torch.tensor([float("nan"), float("inf")])}}
        wire = sender.encode(values)
        import msgpack
        assert len(msgpack.packb(wire)) < 500
        restored = receiver.decode(wire)
        assert torch.equal(values["bf16"].view(torch.uint8), restored["bf16"].view(torch.uint8))
        for key in ("pixels", "view", "empty"):
            assert np.array_equal(values[key], restored[key])
        assert torch.equal(values["inter"][0.7].view(torch.uint8), restored["inter"][0.7].view(torch.uint8))
        # A subsequent overwrite cannot mutate a previously returned result.
        sender.encode({"other": np.zeros(4096, np.uint8)})
        assert np.array_equal(values["pixels"], restored["pixels"])
        receiver.close()
        sender.close()
    finally:
        left.close()
        right.close()


def test_capacity_and_malformed_bounds():
    left, right = socket.socketpair()
    sender = Arena.create(left, 128)
    receiver = Arena.receive(right)
    with pytest.raises(BackpressureError):
        sender.encode(np.ones(129, np.uint8))
    with pytest.raises(ValueError):
        receiver.decode(["numpy", "<f4", [8], 120, 32])
    with pytest.raises(ValueError):
        receiver.decode(["numpy", "<f4", [100], 0, 4])
    sender.close()
    receiver.close()
    left.close()
    right.close()


def test_admission_and_front_disconnect_cleanup():
    service = StageService(None, max_channels=1)
    peers = []
    for _ in range(2):
        left, right = socket.socketpair()
        thread = threading.Thread(target=service.serve_channel, args=(right,))
        thread.start()
        peers.append((Channel(left, capacity=4096), thread))
    assert peers[0][0].call("hello")["epoch"] == service.epoch
    with pytest.raises(BackpressureError, match="channel limit"):
        peers[1][0].call("hello")
    for channel, thread in peers:
        channel.close()
        thread.join(2)
        assert not thread.is_alive()
    assert service.sessions == {}


def test_timeout_permanently_closes_channel():
    class Slow(StageService):
        def dispatch(self, *args):
            time.sleep(0.15)
            return super().dispatch(*args)
    left, right = socket.socketpair()
    thread = threading.Thread(target=Slow(None).serve_channel, args=(right,))
    thread.start()
    channel = Channel(left, capacity=4096, timeout=0.02)
    with pytest.raises(StageRPCError, match="IPC failed"):
        channel.call("hello")
    assert channel.closed
    with pytest.raises(StageRPCError, match="closed"):
        channel.call("hello")
    thread.join(2)
    assert not thread.is_alive()
