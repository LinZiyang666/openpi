"""Fixed worker-only topologies; no caller-selected destination trees."""
import os
from pathlib import Path

# host -> (worker island, EGL devices). Worker counts come from the central scheduler.
FLEETS = {
    "timan107": (Path("/scratch/zixuans8/openpi_trace_h100"), tuple(range(8))),
}


def worker_host():
    host = os.environ.get("WORKER_HOST", "timan107")
    if host not in FLEETS:
        raise ValueError("WORKER_HOST must be timan107; timan108 is disabled")
    return host


def island(host=None):
    return FLEETS[host or worker_host()][0]
