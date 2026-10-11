"""Resource gates shared by launchers; metadata-only checks do not touch CUDA."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import socket
import subprocess
import time
import sys

GPU_AUTHORIZED_AT = datetime(2026, 10, 9, 4, 40, tzinfo=timezone.utc)


def tether_transfer(argv):
    """Retry only tether's temporary shared-transfer saturation, with fresh IDs."""
    for attempt in range(8):
        result = subprocess.run(argv, text=True, capture_output=True)
        if result.returncode == 0:
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            return
        if result.returncode != 75 or "too_many_in_flight" not in result.stderr or attempt == 7:
            raise subprocess.CalledProcessError(result.returncode, argv, result.stdout, result.stderr)
        delay = min(2 ** (attempt + 1), 30)
        print(f"tether transfer busy; retry in{delay}s", file=sys.stderr, flush=True)
        time.sleep(delay)


def gpu_time_gate():
    if datetime.now(timezone.utc) < GPU_AUTHORIZED_AT:
        raise RuntimeError("GPU use predates owner authorization (updated 2026-10-08 23:4x CDT)")


def disk_gate(root, required_gb, floor_gb=None):
    root = Path(root).resolve()
    existing = root
    while not existing.exists():
        existing = existing.parent
    if floor_gb is None:
        floor_gb = 30 if socket.gethostname() != "weilandserver" and str(root).startswith("/data/") else 5
    free = shutil.disk_usage(existing).free / 1e9
    if free < required_gb + floor_gb:
        raise RuntimeError(f"{root}: free={free:.1f} GB; need {required_gb:.1f} GB + {floor_gb:.1f} GB floor")
    shared = Path("/data/mw_finetune")
    if socket.gethostname() != "weilandserver" and root.is_relative_to(shared):
        used = 0.0
        if shared.exists():
            # du counts allocated blocks and does not follow the read-only Wan symlinks.
            used = int(subprocess.check_output(["du", "-s", "-B1", str(shared)], text=True).split()[0]) / 1e9
        if used + required_gb > 120:
            raise RuntimeError(f"H100 shared budget: {used:.1f} GB used + {required_gb:.1f} GB requested exceeds120 GB")
    return free
