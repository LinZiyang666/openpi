"""Real process EOF cleanup using inherited socketpairs; no server/listener."""
import multiprocessing as mp
import os
import socket
import threading

import torch

from openpi.serving.batching_coordinator import BatchingCoordinator
from openpi.serving.split_service import StageService
from openpi.serving.split_transport import Channel
from tests.serving.decouple.stubs import PiModel


def _crashing_front(sock):
    channel = Channel(sock, capacity=8 << 20)
    channel.call("pi_stage", (1, "front", {"state": torch.ones(32)}, {}))
    # Deliberately bypass finally/atexit, like a killed frontend.
    os._exit(0)


def test_real_front_process_exit_releases_gpu_handles_and_memfd():
    model = PiModel()
    core = BatchingCoordinator(model, max_batch_size=1, max_wait_ms=0)
    core.start()
    service = StageService(model, coordinator=core)
    left, right = socket.socketpair()
    thread = threading.Thread(target=service.serve_channel, args=(right,))
    thread.start()
    process = mp.get_context("spawn").Process(target=_crashing_front, args=(left,))
    try:
        process.start()
        left.close()
        process.join(20)
        assert process.exitcode == 0
        thread.join(3)
        assert not thread.is_alive()
        assert service.sessions == {}
        assert service.admission.acquire(blocking=False)
        service.admission.release()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(3)
        core.stop()
