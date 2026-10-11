"""Read-only timan108 admission, including owner's 6 GiB/Cosmos reservation.

Older Cosmos receipts omit worker_memory_mib (shared admission defaults4 GiB).
This preflight conservatively reserves6 GiB for each live Cosmos worker, reads
PID start ticks before counting RSS, and never edits receipts/signals processes.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from exp.r17_integration.ops.worker_memory import capacity

MIB = 1024**2


def evaluate(total, available, fleets, workers=12, worker_mib=4096):
    pending = sum(max(0, row["workers"]*max(row["reserve_mib"], 6144 if row["cosmos"] else 0)*MIB-row["rss_bytes"]) for row in fleets)
    headroom = max(8192*MIB, total//10)
    requested = workers*worker_mib*MIB
    return dict(admitted=requested+pending+headroom <= available, total_bytes=total, available_bytes=available,
        other_pending_bytes=pending, headroom_bytes=headroom, requested_bytes=requested,
        workers=workers, worker_memory_mib=worker_mib, cosmos_min_worker_memory_mib=6144, fleets=fleets,
        scope="read-only cooperative admission; rerun immediately before launch")


def remote_observation():
    def rss(entry):
        try:
            stat = Path(f"/proc/{entry['pid']}/stat").read_text().split(") ", 1)[1].split()
            if stat[0] == "Z" or str(entry.get("start_ticks")) != stat[19]:
                return None
            return next(int(line.split()[1])*1024 for line in Path(f"/proc/{entry['pid']}/status").read_text().splitlines() if line.startswith("VmRSS:"))
        except (OSError, KeyError, StopIteration):
            return None
    total, available = capacity()  # unchanged P6 host/cgroup admission function
    fleets = []
    for path in Path("/scratch/zixuans8/mw_r17/formal/runs").glob("*/receipt.json"):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if doc.get("status") != "RUNNING":
            continue
        amounts = [value for entry in doc.get("pids", []) if (value := rss(entry)) is not None]
        owner_live = doc.get("fleet_owner") is not None and rss(doc["fleet_owner"]) is not None
        count = doc.get("workers", len(amounts)) if owner_live else len(amounts)
        if count:
            fleets.append(dict(receipt=str(path), workers=count, rss_bytes=sum(amounts),
                reserve_mib=doc.get("worker_memory_mib") or 4096,
                cosmos=doc.get("model") == "cosmos" or "cosmos" in path.parent.name))
    return total, available, fleets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--worker-memory-mib", type=int, default=4096)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    import inspect
    code = "import json\nfrom pathlib import Path\n" + inspect.getsource(capacity) + "\n" + inspect.getsource(remote_observation) + "\nprint(json.dumps(remote_observation()))"
    probe = subprocess.run(["tether", "exec", "--safe", "--timeout", "30s", "timan108", "--", "python3", "-c", code], capture_output=True, text=True)
    if probe.returncode:
        report = dict(admitted=False, node="timan108", time=time.time(), exit_code=probe.returncode,
            error=probe.stderr.strip() or probe.stdout.strip())
    else:
        report = evaluate(*json.loads(probe.stdout), args.workers, args.worker_memory_mib)
        report.update(node="timan108", time=time.time())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
