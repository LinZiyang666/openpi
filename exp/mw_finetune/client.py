"""Existing websocket client protocol with bounded connection and receive waits."""
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import websockets.sync.client


class _TimedSocket:
    def __init__(self, socket, timeout):
        self.socket, self.timeout = socket, timeout

    def recv(self):
        return self.socket.recv(timeout=self.timeout)

    def __getattr__(self, name):
        return getattr(self.socket, name)


class BoundedClient(WebsocketClientPolicy):
    def __init__(self, host="127.0.0.1", port=None, timeout=300):
        self.timeout = timeout
        super().__init__(host=host, port=port)

    def _wait_for_server(self):
        connection = websockets.sync.client.connect(self._uri, compression=None, max_size=None,
                                                   open_timeout=30, close_timeout=5)
        try:
            metadata = msgpack_numpy.unpackb(connection.recv(timeout=30))
        except BaseException:
            connection.close()
            raise
        return _TimedSocket(connection, self.timeout), metadata
