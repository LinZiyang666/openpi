"""WebSocket control-message helpers for cache configuration.

Extracts the ``_send_cache_config`` helper previously inlined in
``exp/cache_experiment/run_cache_experiments.py`` into a shared module.

Public interface
----------------
- ``send_load_cache_config(server_url, yaml_path) -> int``: switch the
  server's cache bundle to the YAML at ``yaml_path``. Returns the new
  bundle version, which the caller can diff to detect a silently dead
  server.

Scope note
----------
Per plan §19.B7, the first-version prefill path is driven exclusively by
``websocket_client_policy.prefill_trajectory`` — which must share the same
connection as subsequent ``infer`` calls so prefill mode lands on the
correct per-connection facade. Standalone ``prefill_begin`` / ``prefill_end``
RPCs would land on an unrelated connection and are therefore deliberately
not implemented here.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import msgpack
import websockets

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Low-level sender
# ------------------------------------------------------------------


async def _send_ctrl(server_url: str, msg: dict, *, ack: str) -> dict:
    """Send one ``__ctrl__`` message and validate the server ack.

    The websocket server sends a metadata packet on connect; we discard it
    before sending the control message. Raises ``RuntimeError`` if the
    server replies with anything other than the expected ack string.

    Public wrappers below drive this coroutine with ``asyncio.run`` and must
    therefore not be called from inside a running event loop.
    """
    async with websockets.connect(server_url) as ws:
        _metadata = await ws.recv()
        await ws.send(msgpack.packb(msg))
        resp = msgpack.unpackb(await ws.recv())
        if resp.get("__ack__") != ack:
            raise RuntimeError(f"Control message failed (expected {ack}): {resp}")
        return resp


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------


def send_load_cache_config(server_url: str, yaml_path: str | Path) -> int:
    """Switch the server cache bundle to the YAML at ``yaml_path``.

    Returns the new bundle version reported by the server (monotonic
    counter). Callers use the version to confirm the bundle actually
    changed (a silently idle server keeps the old version).
    """
    yaml_content = Path(yaml_path).read_text()
    msg = {"__ctrl__": "load_cache_config", "yaml_content": yaml_content}
    resp = asyncio.run(_send_ctrl(server_url, msg, ack="load_cache_config"))
    logger.info("Switched server to bundle v%s: %s", resp.get("version"), yaml_path)
    return int(resp.get("version", -1))
