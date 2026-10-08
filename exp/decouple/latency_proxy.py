"""Coordinator benchmark relay; records each successful serving round trip.

Use the same relay for current and split arms. Worker protocol is unchanged.
Latency spans receipt of a worker frame through sending the backend response;
remote worker network time is outside this boundary. This is benchmark-only.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid

from openpi_client import msgpack_numpy


async def relay(front, backend, log):
    await front.send(await backend.recv())  # original metadata greeting
    uid, attempt = None, None
    connection = uuid.uuid4().hex
    async for frame in front:
        start = time.perf_counter_ns()
        started_t = time.monotonic()
        try:
            request = msgpack_numpy.unpackb(frame)
        except Exception:
            request = None
        if isinstance(request, dict) and request.get("__ctrl__") == "episode_start":
            # The deployed WebsocketClientPolicy writes __extra__ on the wire.
            metadata = request.get("__extra__") or request.get("extra_metadata") or {}
            uid, attempt = metadata.get("task_uid"), metadata.get("attempt")
        infer = isinstance(request, dict) and "__ctrl__" not in request
        await backend.send(frame)
        response = await backend.recv()
        await front.send(response)
        elapsed = (time.perf_counter_ns()-start)/1e6
        if infer:
            try:
                result = msgpack_numpy.unpackb(response)
                ok = isinstance(result, dict) and "actions" in result
            except Exception:
                ok = False
            log.write(json.dumps(dict(uid=uid, attempt=attempt, connection=connection,
                                      started_t=started_t, completed_t=time.monotonic(),
                                      latency_ms=elapsed, ok=ok))+"\n")
            log.flush()
        if isinstance(request, dict) and request.get('__ctrl__') == 'episode_end':
            uid, attempt = None, None


async def run(args):
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve
    from openpi.serving.replica_proxy import _MAX_WS_FRAME_BYTES
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as log:
        async def handler(front):
            async with connect(f"ws://127.0.0.1:{args.backend_port}", max_size=_MAX_WS_FRAME_BYTES,
                               compression=None, ping_interval=None) as backend:
                await relay(front, backend, log)
        async with serve(handler, args.host, args.port, max_size=_MAX_WS_FRAME_BYTES,
                         compression=None, ping_interval=None):
            await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--backend-port", type=int, required=True)
    parser.add_argument("--out", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
