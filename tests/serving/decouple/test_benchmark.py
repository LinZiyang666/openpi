import asyncio
import io
import json

import numpy as np
from openpi_client import msgpack_numpy

from exp.decouple.latency_proxy import relay


def test_benchmark_relay_preserves_frames_and_records_only_successful_infer():
    pack = msgpack_numpy.Packer().pack
    frames = [pack({"__ctrl__": "episode_start", "__extra__": {"task_uid": "B:1", "attempt": 2}}),
              pack({"state": np.ones(3)}), pack({"__ctrl__": "select_bundle", "bundle_id": "x"}),
              pack({"state": np.zeros(3)})]
    replies = [pack({"concurrent": True}), pack({"__ack__": "episode_start"}),
               pack({"actions": np.zeros((10, 7))}), pack({"__ack__": "select_bundle"}), "failed"]
    class Socket:
        def __init__(self, inbound):
            self.inbound, self.sent = list(inbound), []
        async def send(self, frame):
            self.sent.append(frame)
        async def recv(self):
            return self.inbound.pop(0)
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self.inbound:
                raise StopAsyncIteration
            return await self.recv()
    front, backend, log = Socket(frames), Socket(replies), io.StringIO()
    asyncio.run(relay(front, backend, log))
    assert front.sent == replies and backend.sent == frames
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    assert len(rows) == 2
    assert [r["ok"] for r in rows] == [True, False]
    assert all(r["uid"] == "B:1" and r["attempt"] == 2 for r in rows)
