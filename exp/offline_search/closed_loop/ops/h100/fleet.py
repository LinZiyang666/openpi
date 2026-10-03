"""Fixed worker-only topologies; no caller-selected destination trees."""
import os
from pathlib import Path

FLEETS = {
    "timan108": (Path("/scratch/zixuans8/openpi_trace"), tuple(range(4)), 40),
    "timan107": (Path("/scratch/zixuans8/openpi_trace_h100"), tuple(range(8)), 64),
}


def worker_host():
    host = os.environ.get("WORKER_HOST", "timan108")
    if host not in FLEETS:
        raise ValueError("WORKER_HOST must be timan108 or timan107")
    return host


def island(host=None):
    return FLEETS[host or worker_host()][0]
