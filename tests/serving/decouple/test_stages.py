import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from openpi.serving.batching_coordinator import BatchingCoordinator, Stage3MissPayload
from openpi.serving.split_service import RemoteCoordinator, StageClient, StageService
from openpi.serving.split_transport import Channel, StageRPCError
from tests.serving.decouple.stubs import PiModel


class Harness:
    def __init__(self, service):
        self.service = service
        self.channels, self.threads = [], []
        def channel_factory():
            left, right = socket.socketpair()
            thread = threading.Thread(target=service.serve_channel, args=(right,))
            thread.start()
            self.threads.append(thread)
            channel = Channel(left)
            self.channels.append(channel)
            return channel
        self.client = StageClient(None, channel_factory=channel_factory)

    def close(self):
        for channel in self.channels:
            channel.close()
        for thread in self.threads:
            thread.join(3)
            assert not thread.is_alive()


@pytest.mark.parametrize("k", [1, 2, 10])
def test_pi_exact_interleaved_and_k_miss(k):
    model = PiModel()
    direct = BatchingCoordinator(model, max_batch_size=4, max_wait_ms=1)
    remote_core = BatchingCoordinator(model, max_batch_size=4, max_wait_ms=1)
    direct.start()
    remote_core.start()
    harness = Harness(StageService(model, coordinator=remote_core))
    remote = RemoteCoordinator(harness.client)
    def run(i):
        obs = {"state": torch.arange(32).float() + i}
        noise = torch.arange(320).reshape(10, 32).float() / 320
        s1 = direct.submit_to_stage(1, "old", obs)
        s2 = direct.submit_to_stage(2, "old", s1)
        expected = direct.submit_to_stage(3, "old", Stage3MissPayload(s2, noise, k))
        with harness.client.decision():
            first = remote.submit_to_stage(1, "new", obs)
            assert first.prefix_att_2d_masks_4d is None
            second = remote.submit_to_stage(2, "new", first)
            assert second.past_key_values is None
            actual = remote.submit_to_stage(3, "new", Stage3MissPayload(second, noise, k))
            assert torch.equal(actual.action_chunk, expected.action_chunk)
            for t in expected.intermediates:
                assert torch.equal(actual.intermediates[t], expected.intermediates[t])
        return first
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            stages = list(pool.map(run, range(8)))
        assert all(not handles for handles in harness.service.sessions.values())
        # Foreign/expired handles never resolve after the decision was released.
        with pytest.raises(StageRPCError, match="expired or foreign"):
            remote.submit_to_stage(2, "new", stages[0])
    finally:
        harness.close()
        direct.stop()
        remote_core.stop()


def test_restart_epoch_rejects_old_handle():
    model = PiModel()
    core = BatchingCoordinator(model, max_batch_size=1, max_wait_ms=0)
    core.start()
    old = Harness(StageService(model, coordinator=core))
    new = Harness(StageService(model, coordinator=core))
    try:
        first = RemoteCoordinator(old.client).submit_to_stage(1, "x", {"state": torch.zeros(32)})
        assert old.service.epoch != new.service.epoch
        with pytest.raises(StageRPCError, match="expired or foreign"):
            RemoteCoordinator(new.client).submit_to_stage(2, "x", first)
        # New episode, new stage-1 handle works with the restarted service.
        with new.client.decision():
            remote = RemoteCoordinator(new.client)
            new_first = remote.submit_to_stage(1, "x", {"state": torch.zeros(32)})
            assert remote.submit_to_stage(2, "x", new_first).past_key_values is None
    finally:
        old.close()
        new.close()
        core.stop()
