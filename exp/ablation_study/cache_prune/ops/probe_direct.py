"""One status line per lane for the direct (freeze-less) run, from the journals.

usage: probe_direct.py <direct-root> <lane> [window-minutes]
"""

from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import time


def listening(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    root, lane = Path(sys.argv[1]) / sys.argv[2], sys.argv[2]
    window = float(sys.argv[3]) * 60 if len(sys.argv) > 3 else 900.0
    now = time.time()
    parts = []
    for family in sorted(root.glob("*_*")):
        journal = family / "journal.jsonl"
        if not journal.exists():
            continue
        rows, recent, infra, ok = 0, 0, 0, 0
        with journal.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") not in ("done", "failed") or row.get("accepted") is False:
                    continue
                rows += 1
                ok += bool(row.get("success"))
                infra += bool(row.get("error"))
                recent += now - row.get("ts", 0) <= window
        arms = 0
        matrix = family / "driver.log"
        if matrix.exists():
            text = matrix.read_text(errors="replace")
            arms = text.count("bundle loaded from")
        done = "DRIVER_EXIT" in (matrix.read_text(errors="replace") if matrix.exists() else "")
        parts.append(
            f"{family.name}: {rows}/5000 ep sr={ok / rows:.3f} " if rows else f"{family.name}: 0/5000 "
        )
        parts[-1] += f"rate={recent / (window / 60):.0f}/min infra_err={infra} arms_loaded={arms}{' DONE' if done else ''}"
    drivers = subprocess.run(
        ["pgrep", "-fc", "run_size_eval --role [d]river"], capture_output=True, text=True
    ).stdout.strip() or "0"
    base = 23280 if lane == "A" else 23150
    up = sum(listening(p) for p in range(base, base + 5))
    print(f"PROBE lane={lane} servers={up}/5 drivers={drivers} | " + " | ".join(parts))


if __name__ == "__main__":
    main()
