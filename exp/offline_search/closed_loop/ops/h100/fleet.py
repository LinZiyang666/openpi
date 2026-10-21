"""Fixed worker-only topologies; no caller-selected destination trees."""
import os
from pathlib import Path

# host -> (worker island, EGL devices). Worker counts come from the central scheduler.
# 'h100w' is the H100 as a worker machine ('h100' already names its server helper in control.py).
FLEETS = {
    "timan107": (Path("/scratch/zixuans8/openpi_trace_h100"), tuple(range(8))),
    "timan1": (Path("/scratch/zixuans8/openpi_trace_h100"), (1, 2)),  # our GPU2/GPU1
    "a100": (Path("/data/mw_finetune/qualification/r15_island"), (0,)),
    "h100w": (Path("/data/mw_finetune/qualification/r15_island"), (0,)),
}
# Hosts that can run an arm's driver, preferred first: public address, fleets whose agents can reach it.
# Measured 10-21: timan107 accepts inbound connections from campus (timan1) only; timan1 is behind NAT.
DRIVERS = {"a100": ("149.165.152.232", ("timan107", "timan1", "h100w")),
           "timan107": ("192.17.58.207", ("timan1",))}


def worker_host():
    """The fleet that runs the arm's driver (and its own agent in the same process)."""
    host = os.environ.get("WORKER_HOST", "timan107")
    if host not in FLEETS:
        raise ValueError("WORKER_HOST must be one of " + ", ".join(FLEETS) + "; timan108 is disabled")
    return host


def worker_hosts():
    """All fleets of an arm, driver host first. Opt-in: WORKER_HOSTS (comma list) at sync; unset = one fleet."""
    host = worker_host()
    reach = DRIVERS.get(host, ("", ()))[1]
    others = [h for h in dict.fromkeys(os.environ.get("WORKER_HOSTS", "").split(",")) if h and h != host]
    if any(h not in reach for h in others):
        raise ValueError(f"WORKER_HOSTS: agents on {others} cannot reach a driver on {host} (reachable: {reach})")
    return (host, *others)


def node(host):
    """Tether node and central-scheduler node of a fleet."""
    return "h100" if host == "h100w" else host


def island(host=None):
    return FLEETS[host or worker_host()][0]
