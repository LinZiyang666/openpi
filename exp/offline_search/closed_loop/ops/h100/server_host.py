"""Policy-server topology; the historical h100 path remains the default."""
import os


def server_host():
    host = os.environ.get("SERVER_HOST", "h100")
    if host not in ("h100", "local"):
        raise ValueError("SERVER_HOST must be h100 or local")
    return host


def endpoint(port, host=None):
    host = server_host() if host is None else host
    if host not in ("h100", "local"):
        raise ValueError("unknown server host")
    return f"{'ziyanglin.com' if host == 'local' else '149.165.153.233'}:{port}"
