"""Slot scheduler for the RoboCasa365 weighted-sum search (runs on the tether hub).

The GR00T serving stack has no bundle hot-swap, so every search cell needs its
own server process. This orchestrator owns that lifecycle across S parallel
slots. Since the cross-machine pool (owner directive 2026-08-22) each cell is
a three-process affair:

  wssrv<port>   cache server            weilandserver  (per-cell restart)
  wsdrvD<port>  run_ws_search --role driver, fixed pull port 2318x
                                        weilandserver  (public 231xx range,
                                        so agents on any machine can join)
  wsdrvA<port>  --role agent (timan107 sim fleet)      timan107
  wsdrvB<port>  --role agent (local sim island)        weilandserver

The driver owns journal / run-plan / summary on weilandserver; summaries from
the earlier single-machine phase live on timan107, so completeness checks
consult BOTH hosts. All remote interaction is short ``tether exec`` calls; the
long waits happen locally. Resume is derived: a cell whose summary says
``complete`` (either host) is skipped.

Trust rules learned the hard way:
- A listening port proves nothing (shared 231xx range, dying predecessors).
  A server counts as up only when ITS fresh log shows the cell's own YAML
  path AND the websockets bind line; the pre-bind CLI banner lies.
- tmux kill HUPs a process group only; sim workers are their own session
  leaders. run_ws_search traps SIGHUP/SIGTERM and reaps, and the orchestrator
  STILL sweeps leftover ``worker_entry`` processes anchored on the slot's
  port — on BOTH machines, never anything wider.
- Server starts serialize under one global lock (VRAM gate is check-then-act).

Run inside a local tmux so it survives the operator's session::

    tmux new -s wsorch -d "python exp/robocasa365/orchestrate_ws_search.py \
        --teacher groot_tp --slots 23160,23161,23162,23163 2>&1 | tee /tmp/wsorch.log"
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import threading
import time

WEILAND = "weilandserver"
TIMAN = "timan107"
# Third worker host (agent C). timan108 and timan1 carry identical
# /scratch/zixuans8 sim islands (cloned 2026-08-22); timan108 is CUDA-only
# until an admin reboot restores EGL, timan1 passed the full env smoke.
# Overridable per run via --agent-c-host.
TIMAN2 = "timan108"
PUBLIC_HOST = "ziyanglin.com"

REMOTE_REPO_WEILAND = "/home/weiland/openpi"
REMOTE_REPO_TIMAN = "/scratch/zixuans8/openpi_rc365"
# The sim-island venv doubles as the timan107 agent interpreter: conductor
# imports are light (verified 2026-08-21) and it carries websockets + msgpack.
TIMAN_PYTHON = (
    "/scratch/zixuans8/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
)
# Driver + weilandserver agent run in the full repo venv there.
WEILAND_PYTHON = f"{REMOTE_REPO_WEILAND}/.venv/bin/python"

SUMMARY_DIRS = {
    WEILAND: f"{REMOTE_REPO_WEILAND}/exp/robocasa365/data/ws_search",
    TIMAN: f"{REMOTE_REPO_TIMAN}/exp/robocasa365/data/ws_search",
}

CKPT = {
    "groot_tp": (
        "/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/"
        "target_posttraining/atomic_seen/checkpoint-60000"
    ),
    "pi05": "/home/weiland/ckpt_pi05_robocasa_pytorch",
}

# Stable server footprint + headroom, required across a VRAM triple read
# before claiming (plan discipline): GR00T ~6.25 GiB, pi05 ~8 GiB.
VRAM_NEED_MIB = {"groot_tp": 8500, "pi05": 10500}

# Extra server CLI flags per teacher, set by main() from --server-extra-flags
# (e.g. "--compile-stage1" once its real-machine gate has passed).
SERVER_EXTRA_FLAGS: dict[str, str] = {}

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
    values = [int(v) for v in (line.strip() for line in out.splitlines()) if v.isdigit()]
    return min(values) if values else None


def vram_gate(need_mib: int) -> bool:
    """Three consecutive reads of free VRAM, all >= the per-server need."""
    for _ in range(3):
        free = vram_free_mib()
        if free is None or free < need_mib:
            return False
        time.sleep(2)
    return True


def server_launch(teacher: str, port: int, yaml_rel: str) -> tuple[str, list[str]]:
    """Per-teacher (launch command, identity grep patterns).

    Identity patterns are matched against the server's FRESH log (tee
    truncates per launch); every pattern must appear AND the port must listen
    before the slot counts the server as up.
    """
    extra = SERVER_EXTRA_FLAGS.get(teacher, "")
    if teacher == "groot_tp":
        cmd = (
            f"export HOME=/home/weiland; cd {REMOTE_REPO_WEILAND} && OPENPI_MONITOR_LEVEL=BASIC "
            f"PYTHONPATH=/home/weiland/gr00t_n15:{REMOTE_REPO_WEILAND}/src:{REMOTE_REPO_WEILAND} "
            f"/home/weiland/gr00t_n15_venv/.venv/bin/python exp/robocasa365/serve_groot_n15.py "
            f"--checkpoint {CKPT[teacher]} --port {port} "
            f"--cache-config {yaml_rel} --concurrent {extra}".rstrip()
        )
        patterns = [
            f"serving stack: concurrent cache -> {yaml_rel}",
            f"INFO:websockets.server:server listening on 0.0.0.0:{port}",
        ]
    elif teacher == "pi05":
        # tyro ordering: every flag BEFORE the policy:checkpoint subcommand.
        cmd = (
            f"export HOME=/home/weiland; cd {REMOTE_REPO_WEILAND} && OPENPI_MONITOR_LEVEL=BASIC "
            f".venv/bin/python scripts/serve_policy.py --port {port} "
            f"--cache_config {yaml_rel} policy:checkpoint "
            f"--policy.config pi05_robocasa --policy.dir {CKPT[teacher]} {extra}".rstrip()
        )
        patterns = [f"Loading cache config from .*{yaml_rel}"]
    else:
        raise ValueError(f"no server recipe for teacher {teacher!r}")
    return cmd, patterns


def pack_agent_c_gpus(slot_index: int, per_slot: int, gpu_order: str, per_gpu_cap: int) -> str:
    """GPU string for one slot's agent-C fleet: fill one card before opening the next.

    Owner ruling 2026-08-22 for timan1: never spread workers across GPUs --
    pack the first card in ``gpu_order`` to ``per_gpu_cap`` workers, only then
    move on. Each slot's whole fleet lands on a single card (fleets are small
    relative to the cap, and one card per slot keeps the eviction/orphan
    bookkeeping trivial).
    """
    gpus = [g.strip() for g in gpu_order.split(",") if g.strip()]
    workers_before = slot_index * per_slot
    return gpus[min(workers_before // max(per_gpu_cap, 1), len(gpus) - 1)]


def stratify_by_family(cids: list[str]) -> list[str]:
    """Round-robin the cells across weight families (iso/grid2/grid3/...).

    The families are wildly unequal in size (4 iso vs 42 grid2 vs 35 grid4) and
    the cid sort key groups them into contiguous blocks, so an arm that is
    stopped, crashed, or merely read early covers whole families and none of
    the others -- a partial leaderboard that reads as a finding but is really an
    artefact of the queue order. Interleaving makes every prefix a balanced
    sample, so mid-run readings stay honest. Order stays deterministic.
    """
    families: dict[str, list[str]] = {}
    for cid in cids:
        families.setdefault(cid.split("_", 1)[0], []).append(cid)
    order, groups = [], [sorted(families[k]) for k in sorted(families)]
    for i in range(max((len(g) for g in groups), default=0)):
        order.extend(g[i] for g in groups if i < len(g))
    return order


class CellQueue:
    """One shared queue every slot pulls from when it frees up.

    Static per-slot shards (``cids[i::n]``) are balanced in *count* only. Cell
    wall-clock varies with the task mix, and an already-complete cell costs
    seconds while a fresh one costs half an hour, so the shards finish at
    wildly different times -- measured live 2026-08-22: two of four slots drained
    2.5 h before the others and their 20 workers sat idle the whole time, because
    a drained slot cannot take work from a busy one. Pulling on demand removes
    the tail entirely.
    """

    def __init__(self, cids: list[str]) -> None:
        self._items = collections.deque(cids)
        self._lock = threading.Lock()

    def pop(self) -> str | None:
        with self._lock:
            return self._items.popleft() if self._items else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class Slot(threading.Thread):
    def __init__(self, *, teacher: str, port: int, pull_port: int, cids: CellQueue,
                 timan_workers: int, weiland_workers: int, timan_gpus: str,
                 timan2_workers: int = 0, timan2_gpus: str = "0,1,2",
                 episodes: int = 8, run_prefix: str = "ws1",
                 log: pathlib.Path = None, lock: threading.Lock = None) -> None:
        super().__init__(name=f"slot{port}", daemon=True)
        self.teacher, self.port, self.pull_port, self.cids = teacher, port, pull_port, cids
        self.timan_workers, self.weiland_workers = timan_workers, weiland_workers
        self.timan2_workers, self.timan2_gpus = timan2_workers, timan2_gpus
        self.timan_gpus, self.episodes = timan_gpus, episodes
        self.run_prefix = run_prefix
        # Only spell the flag out when it is not the default: a mid-sweep
        # orchestrator restart must still drive remotes whose run_ws_search.py
        # predates the flag, and argparse rejects what it does not know.
        self.prefix_flag = "" if run_prefix == "ws1" else f"--run-prefix {run_prefix} "
        self.log_path, self.lock = log, lock

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%FT%TZ', time.gmtime())} [{self.port}] {msg}"
        with self.lock:
            print(line, flush=True)
            with self.log_path.open("a") as f:
                f.write(line + "\n")

    # -- remote primitives -------------------------------------------------

    def run_id(self, cid: str) -> str:
        return f"{self.run_prefix}-{cid}__l1s1_{self.teacher}"

    def summary_complete(self, cid: str) -> bool:
        """True only for a clean run-plan-reconciled summary, on EITHER host."""
        fname = f"{self.teacher}/summary_{self.run_id(cid)}.json"
        for node in (WEILAND, TIMAN):
            out = tether(
                node, f"cat {SUMMARY_DIRS[node]}/{fname} 2>/dev/null || echo MISSING",
            )
            if out is None or "MISSING" in out[:200]:
                continue
            try:
                summary = json.loads(out[out.index("{"):])
            except (ValueError, json.JSONDecodeError):
                continue
            if summary.get("complete"):
                return True
        return False

    def start_server(self, cid: str) -> bool:
        yaml_rel = f"exp/robocasa365/config/ws_search/{self.teacher}/{cid}.yaml"
        with _LAUNCH_LOCK:
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
                if vram_gate(VRAM_NEED_MIB[self.teacher]):
                    break
                self.log(f"VRAM gate not met before {cid} (attempt {attempt + 1}/6); waiting 120s")
                time.sleep(120)
            else:
                self.log(f"VRAM gate failed 6x before {cid}; giving up this start")
                return False

            launch_cmd, patterns = server_launch(self.teacher, self.port, yaml_rel)
            tether(
                WEILAND,
                "export HOME=/home/weiland; "
                f"tmux new -s wssrv{self.port} -d \"{launch_cmd} 2>&1 | tee /tmp/wssrv{self.port}.log\"",
            )
            grep_expr = "\\|".join(patterns)
            for _ in range(72):
                time.sleep(5)
                out = tether(
                    WEILAND,
                    f"p=$(ss -tln | grep -c :{self.port}); "
                    f"m=$(grep -c \"{grep_expr}\" /tmp/wssrv{self.port}.log 2>/dev/null); "
                    f"echo CHECK:$p:$m",
                )
                if out is None:
                    continue
                for line in out.splitlines():
                    if line.startswith("CHECK:"):
                        _, p, m = line.strip().split(":")
                        if p.isdigit() and int(p) >= 1 and m.isdigit() and int(m) >= len(patterns):
                            return True
            tail = tether(WEILAND, f"tail -3 /tmp/wssrv{self.port}.log") or "<no log>"
            self.log(f"server for {cid} failed identity check within 6min; killing. tail: {tail[-300:]}")
            self.stop_server()
            return False

    def stop_server(self) -> None:
        tether(WEILAND, f"export HOME=/home/weiland; tmux kill-session -t wssrv{self.port} 2>/dev/null; echo ok")

    def sweep_orphan_workers(self) -> None:
        """Kill leftover sim workers of THIS slot only, on both worker hosts:
        anchored on worker_entry + this slot's server-key port; workers are
        session leaders (start_new_session), so signal the process groups."""
        script = (
            "for p in $(pgrep -f '[w]orker_entry.*:%d' || true); do "
            "kill -TERM -- -$p 2>/dev/null; done; echo swept" % self.port
        )
        tether(TIMAN, script)
        tether(TIMAN2, script)
        tether(WEILAND, "export HOME=/home/weiland; " + script)

    # -- driver + agents ---------------------------------------------------

    def start_driver(self, cid: str) -> bool:
        cmd = (
            f"export HOME=/home/weiland; cd {REMOTE_REPO_WEILAND} && "
            f"PYTHONPATH={REMOTE_REPO_WEILAND}:{REMOTE_REPO_WEILAND}/src "
            f"{WEILAND_PYTHON} -m exp.robocasa365.run_ws_search "
            f"--role driver --teacher {self.teacher} --server {PUBLIC_HOST}:{self.port} "
            f"--cid '{cid}' {self.prefix_flag}--episodes {self.episodes} "
            f"--workers {self.timan_workers + self.weiland_workers} "
            f"--bind-host 0.0.0.0 --bind-port {self.pull_port} "
            f"--env-config exp/robocasa365/config/ws_search_weilandserver.env "
            f"2>&1 | tee /tmp/wsdrvD{self.port}.log"
        )
        tether(
            WEILAND,
            "export HOME=/home/weiland; "
            f"tmux kill-session -t wsdrvD{self.port} 2>/dev/null; sleep 1; "
            f"tmux new -s wsdrvD{self.port} -d \"{cmd}\"",
        )
        for _ in range(24):
            time.sleep(5)
            out = tether(WEILAND, f"grep -c 'driver pull port = {self.pull_port}' /tmp/wsdrvD{self.port}.log 2>/dev/null || echo 0")
            if out is not None and out.strip().splitlines() and out.strip().splitlines()[-1] != "0":
                return True
        self.log(f"driver for {cid} never bound pull port {self.pull_port}")
        return False

    def start_agents(self, cid: str) -> None:
        agent_a = (
            f"export HOME=/home/zixuans8; cd {REMOTE_REPO_TIMAN} && "
            f"PYTHONPATH={REMOTE_REPO_TIMAN}:{REMOTE_REPO_TIMAN}/src "
            f"{TIMAN_PYTHON} -m exp.robocasa365.run_ws_search "
            f"--role agent --teacher {self.teacher} --server {PUBLIC_HOST}:{self.port} "
            f"--cid '{cid}' {self.prefix_flag}--workers {self.timan_workers} --gpu-ids {self.timan_gpus} "
            f"--driver-host {PUBLIC_HOST} --driver-port {self.pull_port} "
            f"--env-config exp/robocasa365/config/ws_search_timan107.env "
            f"2>&1 | tee /tmp/wsdrvA{self.port}.log"
        )
        tether(
            TIMAN,
            "export HOME=/home/zixuans8; "
            f"tmux kill-session -t wsdrvA{self.port} 2>/dev/null; sleep 1; "
            f"tmux new -s wsdrvA{self.port} -d \"{agent_a}\"",
        )
        if self.weiland_workers > 0:
            agent_b = (
                f"export HOME=/home/weiland; cd {REMOTE_REPO_WEILAND} && "
                f"PYTHONPATH={REMOTE_REPO_WEILAND}:{REMOTE_REPO_WEILAND}/src "
                f"{WEILAND_PYTHON} -m exp.robocasa365.run_ws_search "
                f"--role agent --teacher {self.teacher} --server {PUBLIC_HOST}:{self.port} "
                f"--cid '{cid}' {self.prefix_flag}--workers {self.weiland_workers} --gpu-ids 0 "
                f"--driver-host 127.0.0.1 --driver-port {self.pull_port} "
                f"--env-config exp/robocasa365/config/ws_search_weilandserver.env "
                f"2>&1 | tee /tmp/wsdrvB{self.port}.log"
            )
            tether(
                WEILAND,
                "export HOME=/home/weiland; "
                f"tmux kill-session -t wsdrvB{self.port} 2>/dev/null; sleep 1; "
                f"tmux new -s wsdrvB{self.port} -d \"{agent_b}\"",
            )

        if self.timan2_workers > 0:
            agent_c = (
                f"export HOME=/home/zixuans8; cd {REMOTE_REPO_TIMAN} && "
                f"PYTHONPATH={REMOTE_REPO_TIMAN}:{REMOTE_REPO_TIMAN}/src "
                f"{TIMAN_PYTHON} -m exp.robocasa365.run_ws_search "
                f"--role agent --teacher {self.teacher} --server {PUBLIC_HOST}:{self.port} "
                f"--cid '{{cid}}' {self.prefix_flag}--workers {self.timan2_workers} --gpu-ids {self.timan2_gpus} "
                f"--driver-host {PUBLIC_HOST} --driver-port {self.pull_port} "
                f"--env-config exp/robocasa365/config/ws_search_timan107.env "
                f"2>&1 | tee /tmp/wsdrvC{self.port}.log"
            ).replace("{cid}", cid)
            tether(
                TIMAN2,
                "export HOME=/home/zixuans8; "
                f"tmux kill-session -t wsdrvC{self.port} 2>/dev/null; sleep 1; "
                f"tmux new -s wsdrvC{self.port} -d \"{agent_c}\"",
            )

    def driver_gone(self) -> bool:
        """Two consecutive confirmed GONE readings; unknown never counts."""
        gone = 0
        for _ in range(2):
            out = tether(WEILAND, f"export HOME=/home/weiland; tmux has-session -t wsdrvD{self.port} 2>/dev/null && echo LIVE || echo GONE")
            if out is None or "LIVE" in out:
                return False
            if "GONE" in out:
                gone += 1
            time.sleep(3)
        return gone == 2

    def kill_cell_processes(self) -> None:
        tether(WEILAND, f"export HOME=/home/weiland; tmux kill-session -t wsdrvD{self.port} 2>/dev/null; tmux kill-session -t wsdrvB{self.port} 2>/dev/null; echo ok")
        tether(TIMAN, f"export HOME=/home/zixuans8; tmux kill-session -t wsdrvA{self.port} 2>/dev/null; echo ok")
        tether(TIMAN2, f"export HOME=/home/zixuans8; tmux kill-session -t wsdrvC{self.port} 2>/dev/null; echo ok")

    # -- main loop ---------------------------------------------------------

    def run_cell(self, cid: str) -> None:
        if self.summary_complete(cid):
            self.log(f"skip {cid} (summary complete)")
            return
        if not self.start_server(cid):
            self.log(f"SKIP {cid}: server start failed (cell left for a later resume pass)")
            return
        if not self.start_driver(cid):
            self.log(f"SKIP {cid}: driver start failed")
            self.stop_server()
            return
        self.start_agents(cid)
        self.log(f"cell {cid} up: server + driver(pull {self.pull_port}) + agents "
                 f"({self.timan_workers} timan / {self.weiland_workers} weiland)")
        t0 = time.time()
        while not self.driver_gone():
            if time.time() - t0 > 4.5 * 3600:
                self.log(f"STUCK {cid}: driver over 4.5h; killing — cell stays INCOMPLETE for a later resume pass")
                break
            time.sleep(60)
        self.kill_cell_processes()
        self.sweep_orphan_workers()
        done = self.summary_complete(cid)
        self.log(f"{'DONE' if done else 'INCOMPLETE'} {cid} after {(time.time() - t0) / 60:.0f}min")
        self.stop_server()

    def run(self) -> None:
        while (cid := self.cids.pop()) is not None:
            try:
                self.run_cell(cid)
            except Exception as exc:  # keep the queue alive; the cell reruns on resume
                self.log(f"ERROR on {cid}: {exc!r}; continuing with next cell")
                self.kill_cell_processes()
                self.stop_server()
        self.log("queue drained")


def main() -> None:
    global TIMAN2
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True, choices=("groot_tp", "pi05"))
    ap.add_argument("--slots", required=True, help="comma-separated weilandserver server ports (231xx)")
    ap.add_argument("--pull-port-base", type=int, default=23180,
                    help="driver pull ports = base + slot index (must be in the public range)")
    ap.add_argument("--timan-workers", type=int, default=5, help="sim workers per slot on timan107")
    ap.add_argument("--weiland-workers", type=int, default=2, help="sim workers per slot on weilandserver")
    ap.add_argument("--timan2-workers", type=int, default=0, help="sim workers per slot on the agent-C host (0 = off)")
    ap.add_argument("--agent-c-host", default=TIMAN2,
                    help="tether node carrying the third worker fleet (timan108 or timan1; "
                         "both hold the same /scratch/zixuans8 island)")
    ap.add_argument("--timan2-gpus", default="",
                    help='explicit per-slot GPU pin for agent-C workers (overrides packing)')
    ap.add_argument("--timan2-gpu-order", default="0,1,2",
                    help='agent-C GPU fill order; one card fills to --timan2-per-gpu-cap before '
                         'the next opens (owner ruling: never spread). timan1: use "0,2,3" — '
                         'GPU1 belongs to another user')
    ap.add_argument("--timan2-per-gpu-cap", type=int, default=24,
                    help="agent-C workers per GPU before overflowing to the next card")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--gpu-count", type=int, default=8, help="timan107 visible GPUs for round-robin")
    ap.add_argument("--configs-index", default="", help="default: config/ws_search/<teacher>/index.json")
    ap.add_argument("--only", default="", help="comma-separated cid subset (smoke/retry)")
    ap.add_argument("--cid-order", default="stratified", choices=("stratified", "sorted"),
                    help="stratified interleaves the weight families so any partial run is a "
                         "balanced sample; sorted keeps the cid sort order (family-blocked)")
    ap.add_argument("--run-prefix", default="ws1",
                    help="round tag in every run_id, and therefore in the skip-if-complete "
                         "check: bump it (e.g. ws2) to re-run cells at a higher trial count")
    ap.add_argument(
        "--server-extra-flags", default="",
        help='extra flags appended to every server launch (e.g. "--compile-stage1")',
    )
    ap.add_argument("--log", default="", help="default: /tmp/wsorch_<teacher>.progress.log")
    args = ap.parse_args()

    # Module-level so every Slot's launch/kill/sweep path follows along.
    TIMAN2 = args.agent_c_host

    if args.teacher not in CKPT:
        raise SystemExit(f"no server recipe for teacher {args.teacher!r}")
    if args.server_extra_flags:
        SERVER_EXTRA_FLAGS[args.teacher] = args.server_extra_flags.strip()

    root = pathlib.Path(__file__).resolve().parent
    index_path = pathlib.Path(args.configs_index) if args.configs_index else (
        root / "config" / "ws_search" / args.teacher / "index.json")
    cids = sorted(json.loads(index_path.read_text()))
    if args.only:
        keep = {c.strip() for c in args.only.split(",")}
        cids = [c for c in cids if c in keep]
    if args.cid_order == "stratified":
        cids = stratify_by_family(cids)

    slots = [int(p) for p in args.slots.split(",")]
    queue = CellQueue(cids)
    log = pathlib.Path(args.log) if args.log else pathlib.Path(f"/tmp/wsorch_{args.teacher}.progress.log")
    lock = threading.Lock()

    threads = []
    for i, port in enumerate(slots):
        gpus = ",".join(
            str((args.timan_workers * i + j) % args.gpu_count)
            for j in range(args.timan_workers)
        )
        t = Slot(teacher=args.teacher, port=port, pull_port=args.pull_port_base + i,
                 cids=queue, timan_workers=args.timan_workers,
                 weiland_workers=args.weiland_workers, timan_gpus=gpus,
                 timan2_workers=args.timan2_workers,
                 timan2_gpus=args.timan2_gpus or pack_agent_c_gpus(
                     i, args.timan2_workers, args.timan2_gpu_order, args.timan2_per_gpu_cap),
                 episodes=args.episodes, run_prefix=args.run_prefix, log=log, lock=lock)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    print("ALL SLOTS DRAINED", flush=True)


if __name__ == "__main__":
    main()
