"""Content-bound endpoint registry and held free-port reservation (stdlib)."""
from __future__ import annotations

import socket

RESERVED = set(range(23150, 23155)) | set(range(23250, 23255)) | {23180}


class PortReservation:
    def __init__(self, host="127.0.0.1", port=0, *, socket_factory=socket.socket):
        if port in RESERVED:
            raise ValueError("port occupied or reserved by existing R17 lanes")
        self.socket = socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.bind((host, port))
            self.port = self.socket.getsockname()[1]
            if self.port in RESERVED:
                raise ValueError("allocated a reserved port; retry reservation")
            self.socket.listen(1)
        except Exception:
            self.socket.close()
            raise

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def compatible(endpoint, request):
    fields = ("model_identity", "benchmark_protocol", "norm_identity", "layout_identity", "sampler_identity")
    return all(endpoint.get(key) is not None and endpoint[key] == request.get(key) for key in fields)
