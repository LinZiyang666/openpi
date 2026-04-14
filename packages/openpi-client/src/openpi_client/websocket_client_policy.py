import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, TimeoutError, OSError):
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def episode_start(
        self,
        experiment: str,
        task: str = "",
        episode_id: int = -1,
        episode_name: str = "",
    ) -> Dict:
        # ``episode_name`` is always sent on the wire. An empty string preserves the
        # legacy server behaviour (server reads it with ``obs.get("__episode_name__", "")``
        # and treats empty as "no override"); a non-empty value lets callers pick the
        # HDF5 filename used by the collection data_collector (plan §3.1 / §3.4).
        self._ws.send(
            self._packer.pack(
                {
                    "__ctrl__": "episode_start",
                    "__experiment__": experiment,
                    "__task__": task,
                    "__episode_id__": episode_id,
                    "__episode_name__": episode_name,
                }
            )
        )
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def episode_end(self, success: bool = False) -> Dict:
        self._ws.send(self._packer.pack({"__ctrl__": "episode_end", "__success__": success}))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def prefill_trajectory(
        self,
        observations: list,
        actions: list,
        *,
        record: bool = False,
        on_miss: str = "error",
    ) -> Dict:
        """Drive the server's cache framework through an (obs, action) sequence.

        Each ``(observations[i], actions[i])`` pair is fed through the server-side
        ``InferenceInterceptor.prefill_trajectory`` as if it were a real inference
        step, so the per-connection cache facade records query keys and stores
        payloads without actually running the model (see ``openpi.cache.interceptor``
        for the in-process semantics). ``observations`` is ``list[Dict]`` and
        ``actions`` is a list of numpy arrays — both travel via ``msgpack_numpy``
        so arrays round-trip without explicit encoding.

        This is the only prefill entry point in the first-version wire protocol;
        standalone ``prefill_begin`` / ``prefill_end`` control messages are
        intentionally not exposed here (plan §19.B7) because they would have to
        share this connection to land on the correct per-connection facade.
        """
        self._ws.send(
            self._packer.pack(
                {
                    "__ctrl__": "prefill_trajectory",
                    "observations": observations,
                    "actions": actions,
                    "record": record,
                    "on_miss": on_miss,
                }
            )
        )
        response = self._ws.recv()
        if isinstance(response, str):
            # Legacy bare-string error path (pre-cleanup/10 servers). Log a
            # warning so operators notice the stale server, then re-raise so
            # callers still fail loudly.
            logging.warning(
                "Server sent a legacy bare-string prefill error. "
                "Upgrade the server — new protocol uses msgpack "
                "{__ack__: 'error', msg: ...}."
            )
            raise RuntimeError(f"Error in inference server:\n{response}")
        decoded = msgpack_numpy.unpackb(response)
        if isinstance(decoded, dict) and decoded.get("__ack__") == "error":
            raise RuntimeError(
                f"Error in inference server:\n{decoded.get('msg', '(no message)')}"
            )
        return decoded

    @override
    def reset(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Lifecycle helpers (plan §19.B4)
    #
    # Phase-2/Phase-3 runners must drive this client inside a ``with`` block so
    # that the per-connection cache facade is released as soon as the worker
    # finishes; without that, the server-side backend reference count can
    # outlive a config-bundle swap and cause stale facade reads.
    # ------------------------------------------------------------------

    def __enter__(self) -> "WebsocketClientPolicy":
        return self

    def __exit__(self, *exc) -> None:
        # Swallow close errors: the websocket may already be half-closed by the
        # server (for example, the bundle was swapped mid-episode). Propagating
        # a close failure would mask the real exception from the ``with`` body.
        try:
            self._ws.close()
        except Exception:
            pass

    def close(self) -> None:
        self.__exit__(None, None, None)
