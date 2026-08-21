"""Slot scheduler for the RoboCasa365 weighted-sum search (runs on the tether hub).

The GR00T serving stack has no bundle hot-swap, so every search cell needs its
own server process. This orchestrator owns that lifecycle across S parallel
slots: for each cell in a slot's queue it (re)starts the slot's server on
weilandserver with the cell's YAML, verifies IDENTITY (not just a listening
port — see below), launches one ``run_ws_search`` driver invocation on
timan107 (driver + workers in one tmux), polls until the driver exits, checks
the cell summary is COMPLETE (run-plan-reconciled), tears the server down, and
moves on. All remote interaction is short ``tether exec`` calls (the ~10-min
exec cap never bites); the long waits happen locally.

Resume is derived, not stored: a cell whose summary says ``complete`` is
skipped, so re-running the same command after a crash or after INCOMPLETE
cells continues where it left off (the per-cell journal replays done uids).

Trust rules learned the hard way:
- A listening port proves nothing (the 231xx range is shared across sessions
  and a dying predecessor can hold the port). A server counts as up only when
  ITS fresh log (tee truncates per launch) shows the cell's own YAML path on
  the "serving stack" line AND the SERVER-LISTENING banner — provenance from
  live output, never from the fact we just sent a command.
- ``tmux kill-session`` on the driver HUPs the driver process group only;
  workers run in their own sessions. run_ws_search traps SIGHUP/SIGTERM and
  reaps them, but the orchestrator STILL sweeps leftover ``worker_entry``
  processes anchored on this line's unique clone path + the slot's port —
  never anything wider (shared machine).
- Server starts serialize under one global lock: the VRAM triple-read gate is
  check-then-act, and a second slot probing while the first still allocates
  would double-claim the same free memory.

Run inside a local tmux so it survives the operator's session::

    tmux new -s wsorch -d "python exp/robocasa365/orchestrate_ws_search.py \
        --teacher groot_tp --slots 23160,23161,23162 2>&1 | tee /tmp/wsorch_groot_tp.log"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import threading
import time

WEILAND = "weilandserver"
TIMAN = "timan107"
PUBLIC_HOST = "ziyanglin.com"

REMOTE_REPO_WEILAND = "/home/weiland/openpi"
REMOTE_REPO_TIMAN = "/scratch/zixuans8/openpi_rc365"
# The sim-island venv doubles as the driver interpreter: conductor imports are
# light (verified 2026-08-21) and it already carries websockets + msgpack.
TIMAN_PYTHON = (
    "/scratch/zixuans8/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
)

CKPT = {
    "groot_tp": (
        "/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/"
        "target_posttraining/atomic_seen/checkpoint-60000"
    ),
}

# GR00T server stable footprint ~6.25 GiB; require headroom on top before
# claiming (the plan's triple-read discipline).
VRAM_NEED_MIB = 8500

# One server loads at a time (see module docstring).
_LAUNCH_LOCK = threading.Lock()


def tether(node: str, script: str, timeout: int = 300) -> str | None:
    """Run a remote script; None means the CALL failed (timeout/transport), not
    the script — callers must treat None as "unknown", never as a state."""
    try:
        proc = subprocess.run(
            ["tether", "exec", node, "--", "bash", "-lc", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return proc.stdout + proc.stderr


def vram_free_mib() -> int | None:
    out = tether(
        WEILAND,
        "export HOME=/home/weiland; nvidia-smi --query-gpu=memory.free "
        "--format=csv,noheader,nounits",
    )
    if out is None:
        return None
    values = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            values.append(int(line))
    return min(values) if values else None


def vram_gate() -> bool:
    """Three consecutive reads of free VRAM, all >= the per-server need."""
    for _ in range(3):
        free = vram_free_mib()
        if free is None or free < VRAM_NEED_MIB:
            return False
        time.sleep(2)
    return True


class Slot(threading.Thread):
    def __init__(self, *, teacher: str, port: int, cids: list[str], gpu_ids: str,
                 workers: int, episodes: int, log: pathlib.Path, lock: threading.Lock) -> None:
        super().__init__(name=f"slot{port}", daemon=True)
        self.teacher, self.port, self.cids = teacher, port, cids
        self.gpu_ids, self.workers, self.episodes = gpu_ids, workers, episodes
        self.log_path, self.lock = log, lock

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%FT%TZ', time.gmtime())} [{self.port}] {msg}"
        with self.lock:
            print(line, flush=True)
            with self.log_path.open("a") as f:
                f.write(line + "\n")

    # -- remote primitives -------------------------------------------------

    def run_id(self, cid: str) -> str:
        return f"ws1-{cid}__l1s1_{self.teacher}"

    def summary_complete(self, cid: str) -> bool:
        """True only for a summary whose run-plan reconciliation is clean."""
        out = tether(
            TIMAN,
            f"cat {REMOTE_REPO_TIMAN}/exp/robocasa365/data/ws_search/"
            f"{self.teacher}/summary_{self.run_id(cid)}.json 2>/dev/null || echo MISSING",
        )
        if out is None or "MISSING" in out[:200]:
            return False
        try:
            start = out.index("{")
            summary = json.loads(out[start:])
        except (ValueError, json.JSONDecodeError):
            return False
        return bool(summary.get("complete"))

    def start_server(self, cid: str) -> bool:
        yaml_rel = f"exp/robocasa365/config/ws_search/{self.teacher}/{cid}.yaml"
        with _LAUNCH_LOCK:
            # Kill any predecessor and wait for the port to actually free —
            # a lingering listener would make the new bind fail while the
            # port probe still shows "up" (for the WRONG process).
            self.stop_server()
            for _ in range(12):
                out = tether(WEILAND, f"ss -tln | grep -q :{self.port} && echo BUSY || echo FREE")
                if out is not None and "FREE" in out:
                    break
                time.sleep(5)
            else:
                self.log(f"port {self.port} never freed — foreign listener? refusing to start {cid}")
                return False

            for attempt in range(6):
                if vram_gate():
                    break
                self.log(f"VRAM gate not met before {cid} (attempt {attempt + 1}/6); waiting 120s")
                time.sleep(120)
            else:
                self.log(f"VRAM gate failed 6x before {cid}; giving up this start")
                return False

            script = (
                "export HOME=/home/weiland; "
                f"tmux new -s wssrv{self.port} -d \"export HOME=/home/weiland; "
                f"cd {REMOTE_REPO_WEILAND} && OPENPI_MONITOR_LEVEL=BASIC "
                f"PYTHONPATH=/home/weiland/gr00t_n15:{REMOTE_REPO_WEILAND}/src:{REMOTE_REPO_WEILAND} "
                f"/home/weiland/gr00t_n15_venv/.venv/bin/python exp/robocasa365/serve_groot_n15.py "
                f"--checkpoint {CKPT[self.teacher]} --port {self.port} "
                f"--cache-config {yaml_rel} --concurrent 2>&1 | tee /tmp/wssrv{self.port}.log\""
            )
            tether(WEILAND, script)
            # Identity, not liveness: the fresh log must name THIS cell's yaml
            # and print the listening banner. Hold the lock until then so the
            # next slot's VRAM gate sees settled memory.
            for _ in range(72):
                time.sleep(5)
                out = tether(
                    WEILAND,
                    f"grep -l . /tmp/wssrv{self.port}.log >/dev/null 2>&1 && "
                    f"grep -c \"serving stack: concurrent cache -> {yaml_rel}\\|SERVER-LISTENING on 0.0.0.0:{self.port}\" "
                    f"/tmp/wssrv{self.port}.log 2>/dev/null || echo 0",
                )
                if out is not None and out.strip().splitlines() and out.strip().splitlines()[-1] == "2":
                    return True
            tail = tether(WEILAND, f"tail -3 /tmp/wssrv{self.port}.log") or "<no log>"
            self.log(f"server for {cid} failed identity check within 6min; killing. tail: {tail[-300:]}")
            self.stop_server()
            return False

    def stop_server(self) -> None:
        tether(WEILAND, f"export HOME=/home/weiland; tmux kill-session -t wssrv{self.port} 2>/dev/null; echo ok")

    def sweep_orphan_workers(self) -> None:
        """Kill leftover sim workers of THIS slot only: anchored on the unique
        clone path AND this slot's server-key port; workers are their own
        session leaders (start_new_session), so signal the process groups."""
        tether(
            TIMAN,
            "for p in $(pgrep -f '[w]orker_entry.*:%d' || true); do "
            "grep -q openpi_rc365 /proc/$p/cmdline 2>/dev/null && kill -TERM -- -$p 2>/dev/null; "
            "done; echo swept" % self.port,
        )

    def start_driver(self, cid: str) -> None:
        cmd = (
            f"export HOME=/home/zixuans8; cd {REMOTE_REPO_TIMAN} && "
            f"PYTHONPATH={REMOTE_REPO_TIMAN}:{REMOTE_REPO_TIMAN}/src "
            f"{TIMAN_PYTHON} -m exp.robocasa365.run_ws_search "
            f"--teacher {self.teacher} --server {PUBLIC_HOST}:{self.port} --cid '{cid}' "
            f"--episodes {self.episodes} --workers {self.workers} --gpu-ids {self.gpu_ids} "
            f"--env-config exp/robocasa365/config/ws_search_timan107.env "
            f"2>&1 | tee /tmp/wsdrv{self.port}.log"
        )
        script = (
            "export HOME=/home/zixuans8; "
            f"tmux kill-session -t wsdrv{self.port} 2>/dev/null; sleep 1; "
            f"tmux new -s wsdrv{self.port} -d \"{cmd}\""
        )
        tether(TIMAN, script)

    def driver_gone(self) -> bool:
        """Two consecutive confirmed GONE readings; unknown never counts."""
        gone = 0
        for _ in range(2):
            out = tether(TIMAN, f"tmux has-session -t wsdrv{self.port} 2>/dev/null && echo LIVE || echo GONE")
            if out is None or "LIVE" in out:
                return False
            if "GONE" in out:
                gone += 1
            time.sleep(3)
        return gone == 2

    def kill_driver(self) -> None:
        tether(TIMAN, f"export HOME=/home/zixuans8; tmux kill-session -t wsdrv{self.port} 2>/dev/null; echo ok")

    # -- main loop ---------------------------------------------------------

    def run_cell(self, cid: str) -> None:
        if self.summary_complete(cid):
            self.log(f"skip {cid} (summary complete)")
            return
        if not self.start_server(cid):
            self.log(f"SKIP {cid}: server start failed (cell left for a later resume pass)")
            return
        self.log(f"server identity-verified for {cid}; launching driver")
        self.start_driver(cid)
        t0 = time.time()
        while not self.driver_gone():
            if time.time() - t0 > 4.5 * 3600:
                self.log(f"STUCK {cid}: driver over 4.5h; killing — cell stays INCOMPLETE for a later resume pass")
                self.kill_driver()
                break
            time.sleep(60)
        self.sweep_orphan_workers()
        done = self.summary_complete(cid)
        self.log(f"{'DONE' if done else 'INCOMPLETE'} {cid} after {(time.time() - t0) / 60:.0f}min")
        self.stop_server()

    def run(self) -> None:
        for cid in self.cids:
            try:
                self.run_cell(cid)
            except Exception as exc:  # keep the queue alive; the cell reruns on resume
                self.log(f"ERROR on {cid}: {exc!r}; continuing with next cell")
                self.stop_server()
        self.log("queue drained")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True, choices=("groot_tp", "pi05"))
    ap.add_argument("--slots", required=True, help="comma-separated weilandserver ports (231xx)")
    ap.add_argument("--workers-per-slot", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--gpu-count", type=int, default=8, help="timan107 visible GPUs for round-robin")
    ap.add_argument("--configs-index", default="", help="default: config/ws_search/<teacher>/index.json")
    ap.add_argument("--only", default="", help="comma-separated cid subset (smoke/retry)")
    ap.add_argument("--log", default="", help="default: /tmp/wsorch_<teacher>.progress.log")
    args = ap.parse_args()

    if args.teacher not in CKPT:
        raise SystemExit(f"no server recipe for teacher {args.teacher!r} yet (pi05 lands after its P2 gate)")

    root = pathlib.Path(__file__).resolve().parent
    index_path = pathlib.Path(args.configs_index) if args.configs_index else (
        root / "config" / "ws_search" / args.teacher / "index.json")
    cids = sorted(json.loads(index_path.read_text()))
    if args.only:
        keep = {c.strip() for c in args.only.split(",")}
        cids = [c for c in cids if c in keep]

    slots = [int(p) for p in args.slots.split(",")]
    queues = {p: cids[i::len(slots)] for i, p in enumerate(slots)}
    log = pathlib.Path(args.log) if args.log else pathlib.Path(f"/tmp/wsorch_{args.teacher}.progress.log")
    lock = threading.Lock()

    threads = []
    for i, port in enumerate(slots):
        gpus = ",".join(
            str((args.workers_per_slot * i + j) % args.gpu_count)
            for j in range(args.workers_per_slot)
        )
        t = Slot(teacher=args.teacher, port=port, cids=queues[port], gpu_ids=gpus,
                 workers=args.workers_per_slot, episodes=args.episodes, log=log, lock=lock)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    print("ALL SLOTS DRAINED", flush=True)


if __name__ == "__main__":
    main()
