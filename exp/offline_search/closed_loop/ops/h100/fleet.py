"""Fixed worker-only topologies; no caller-selected destination trees."""
from exp.offline_search.rounds.r17.autoscale_1017.switch import enabled
import os
from pathlib import Path

from exp.offline_search.rounds.r17.autoscale_1017.defaults import physical_maximum

FLEETS = {
    "timan107": (Path("/scratch/zixuans8/openpi_trace_h100"), tuple(range(8)), physical_maximum("timan107") if enabled() else 64),
}


def worker_host():
    host = os.environ.get("WORKER_HOST", "timan107")
    if host not in FLEETS:
        raise ValueError("WORKER_HOST must be timan107; timan108 is disabled")
    return host


def island(host=None):
    return FLEETS[host or worker_host()][0]
