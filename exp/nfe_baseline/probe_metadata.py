"""Print the denoising step count a policy server reports in its handshake.

Runs in the LIBERO client environment (python 3.8) and answers one question
for the lane driver: which loop is the server on ``host:port`` serving right
now? GR00T's LIBERO server reports ``denoising_steps``; the Pi0.5 ksweep
wrapper stamps ``nfe_num_steps``. Prints the integer, ``none`` when the server
is up but carries neither key, or ``down`` when nothing accepts the TCP
connection -- the driver only launches a run once every port prints the k it
is about to evaluate.

Usage:
  python exp/nfe_baseline/probe_metadata.py <host> <port>
"""

import socket
import sys


def probe(host, port, timeout=5.0):
    try:
        socket.create_connection((host, int(port)), timeout=timeout).close()
    except OSError:
        return "down"
    from openpi_client import websocket_client_policy as wcp

    client = wcp.WebsocketClientPolicy(host, int(port))
    try:
        md = client.get_server_metadata() or {}
    finally:
        try:
            client._ws.close()
        except Exception:  # noqa: BLE001 - best-effort close of a probe socket
            pass
    for key in ("denoising_steps", "nfe_num_steps"):
        if key in md:
            return str(int(md[key]))
    return "none"


if __name__ == "__main__":
    print(probe(sys.argv[1], sys.argv[2]))
