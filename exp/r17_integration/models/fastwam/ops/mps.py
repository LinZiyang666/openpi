"""CPU-only private MPS lifecycle used by the normal formal launcher.

The guard is an ordinary receipt-bound/leased launcher child, registered before
replicas and closed after them. It never changes compute mode, signals GPU
applications, or addresses the default MPS socket. No CUDA library is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from exp.r17_integration.ops.formal_process import alive, ticks, write_json


def config():
    return dict(schema="r17.mps.v1", enabled=True, private_per_campaign_gpu=True,
                active_thread_percentage=100, compute_mode_change=False)


def bind(campaign, enabled):
    if enabled:
        campaign["mps"] = config()
    return campaign


def check_resume(campaign, source):
    previous = json.loads((Path(source) / "campaign.json").read_text())
    if campaign.get("mps") != previous.get("mps"):
        raise ValueError("resume MPS treatment differs from source")


def comparison_scope(campaign):
    """A timing intervention cannot masquerade as an algorithmic cost gain."""
    source=campaign.get('paired_control')
    previous=json.loads((Path(source)/'campaign.json').read_text()) if source else None
    return dict(changes_model_calls_or_nfe=False, affects_wall_time=True,
                paired_control=source,
                wall_time_comparable_with_control=(campaign.get('mps')==previous.get('mps')) if previous else None,
                sr_qualification='native/batch parity required; throughput is not a success-rate certificate',
                study_cost='STUDY.md requires both call/NFE reductions and comparable service time')


def command(argv, *, env=None, input=None, timeout=10):
    result = subprocess.run(argv, env=env, input=input, text=True,
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"MPS command failed: {argv[0]}: {result.stderr[-1500:]}")
    return result.stdout.strip()


def gpu_snapshot(runner=command):
    text = runner(["nvidia-smi", "--query-gpu=index,uuid,compute_mode,memory.free",
                   "--format=csv,noheader,nounits"])
    rows = []
    for line in text.splitlines():
        index, uuid, mode, free = (x.strip() for x in line.split(","))
        rows.append(dict(index=int(index), uuid=uuid, compute_mode=mode, free_mib=float(free)))
    return rows


def gpu_contexts(uuid, runner=command):
    listing = runner(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    return [dict(pid=int(x), start_ticks=ticks(int(x))) for x in listing.split()]


def select_gpu(device, visible, rows):
    match = re.fullmatch(r"cuda:(\d+)", device)
    if not match:
        raise ValueError("MPS requires an explicit cuda:N device")
    index = int(match[1])
    if visible:
        choices = visible.split(",")
        if index >= len(choices):
            raise ValueError("MPS device outside CUDA_VISIBLE_DEVICES")
        selected = choices[index].strip()
        candidates = [r for r in rows if str(r["index"]) == selected or
                      (selected.startswith("GPU-") and r["uuid"].startswith(selected))]
    else:
        candidates = [r for r in rows if r["index"] == index]
    if len(candidates) != 1:
        raise ValueError("MPS GPU selection is ambiguous or unavailable")
    gpu = candidates[0]
    if gpu["compute_mode"] != "Default":
        raise ValueError("private MPS requires existing Default mode; launcher never changes it")
    return gpu


def paths(campaign_directory, uuid):
    root = Path(campaign_directory).resolve() / "mps" / hashlib.sha256(uuid.encode()).hexdigest()[:6]
    pipe, log = root / "p", root / "l"
    # CUDA's Unix socket path limit also includes internal server/socket names.
    if len(os.fsencode(pipe)) > 85:
        raise ValueError("MPS pipe path too long; use a shorter campaign run name (pipe <=85 bytes)")
    return root, pipe, log


def client_env(state):
    return dict(CUDA_VISIBLE_DEVICES=state["gpu"]["uuid"],
                CUDA_MPS_PIPE_DIRECTORY=state["pipe_directory"],
                CUDA_MPS_LOG_DIRECTORY=state["log_directory"],
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="100")


def private_env(state):
    # Do not inherit partitioning, memory limits, or another campaign's pipe.
    return {**{k: v for k, v in os.environ.items() if not k.startswith("CUDA_MPS_")},
            **client_env(state)}


def replica_argv(argv, ready):
    return [argv[0], '-m', 'exp.r17_integration.models.fastwam.ops.mps', 'exec',
            '--ready', str(ready), '--', *argv]


def exec_replica(ready, argv):
    state=json.loads(Path(ready).read_text())
    control(state,'get_server_list')  # Verify private daemon identity before GPU initialization.
    os.execvpe(argv[0],argv,private_env(state))


def control(state, message, runner=command):
    if not alive(state["daemon"]):
        raise RuntimeError("private MPS daemon PID/start identity is no longer alive")
    pipe = Path(state["pipe_directory"])
    if pipe.is_symlink() or pipe.stat().st_uid != os.getuid() or pipe.stat().st_mode & 0o077:
        raise ValueError("private MPS pipe ownership/permissions changed")
    pidfile = pipe / "nvidia-cuda-mps-control.pid"
    if int(pidfile.read_text()) != state["daemon"]["pid"]:
        raise ValueError("private MPS daemon receipt differs from pipe PID file")
    return runner(["nvidia-cuda-mps-control"], env=private_env(state), input=message+"\n")


def inspect(state, runner=command):
    servers = [int(x) for x in control(state, "get_server_list", runner).split()]
    clients = {}
    for server in servers:
        clients[str(server)] = [int(x) for x in control(state, f"get_client_list {server}", runner).split()]
    return dict(servers=servers, clients=clients, daemon_alive=alive(state["daemon"]))


def account(state, expected_clients, runner=command):
    status = inspect(state, runner)
    clients = {pid for rows in status["clients"].values() for pid in rows}
    if clients != set(expected_clients):
        raise RuntimeError("private MPS clients do not exactly match owned ready replicas")
    owners = set(expected_clients) | set(status["servers"])
    listing = runner(["nvidia-smi", "-i", state["gpu"]["uuid"],
                      "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"])
    memory = {}
    for line in listing.splitlines():
        pid, mib = (x.strip() for x in line.split(","))
        if int(pid) in owners:
            memory[pid] = float(mib)
    # Volta+ drivers normally expose per-client allocations. Older drivers
    # attribute the pool to the server. Never sum both representations twice.
    client_memory = {k: v for k, v in memory.items() if int(k) in expected_clients}
    server_memory = {k: v for k, v in memory.items() if int(k) in status["servers"]}
    if len(client_memory) == len(expected_clients):
        total, scope = sum(client_memory.values()) + sum(server_memory.values()), "clients_plus_private_server_overhead"
    elif not client_memory and len(server_memory) == len(status["servers"]) and server_memory:
        total, scope = sum(server_memory.values()), "private_server_pool"
    else:
        raise RuntimeError("MPS physical VRAM accounting is incomplete")
    return dict(**status, owned_pids=sorted(expected_clients), per_pid_mib=memory,
                owned_total_mib=total, accounting_scope=scope)


class Session:
    def __init__(self, campaign_directory, device="cuda:0", *, runner=command):
        self.directory, self.device, self.runner = Path(campaign_directory), device, runner
        self.state = None

    def start(self):
        gpu = select_gpu(self.device, os.environ.get("CUDA_VISIBLE_DEVICES"), gpu_snapshot(self.runner))
        root, pipe, log = paths(self.directory, gpu["uuid"])
        # O_EXCL-style directory creation: never adopt or overwrite a daemon.
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        pipe.mkdir(mode=0o700)
        log.mkdir(mode=0o700)
        state = dict(**config(), gpu=gpu, directory=str(root), pipe_directory=str(pipe),
                     log_directory=str(log), status="STARTING", uid=os.getuid(), started=time.time())
        state["background_contexts"] = gpu_contexts(gpu["uuid"], self.runner)
        self.state = state
        write_json(root / "state.json", state)
        try:
            self.runner(["nvidia-cuda-mps-control", "-d"], env=private_env(state))
            pidfile = pipe / "nvidia-cuda-mps-control.pid"
            until = time.monotonic()+5
            while not pidfile.exists() and time.monotonic() < until:
                time.sleep(.05)
            pid = int(pidfile.read_text())
            state["daemon"] = dict(pid=pid, start_ticks=ticks(pid))
            state.update(status="RUNNING", initial=inspect(state, self.runner))
            write_json(root / "state.json", state)
            return state
        except BaseException as exc:
            state['startup_error'] = repr(exc)
            pidfile = pipe / "nvidia-cuda-mps-control.pid"
            if "daemon" not in state and pidfile.exists():
                pid = int(pidfile.read_text())
                state["daemon"] = dict(pid=pid, start_ticks=ticks(pid))
            if "daemon" in state:
                self.close()
            else:
                state.update(status="FAIL", daemon_closed=False)
                write_json(root / "state.json", state)
            raise

    def close(self):
        state = self.state
        if state is None or state.get("status") == "CLOSED":
            return state
        try:
            if "daemon" in state and alive(state["daemon"]):
                state["before_close"] = inspect(state, self.runner)
                control(state, "quit", self.runner)
                until = time.monotonic()+5
                while alive(state["daemon"]) and time.monotonic() < until:
                    time.sleep(.05)
            state.update(status="CLOSED", daemon_closed=not ("daemon" in state and alive(state["daemon"])),
                         closed=time.time())
            state['after_contexts'] = gpu_contexts(state['gpu']['uuid'], self.runner)
            after = {(r['pid'],r['start_ticks']) for r in state['after_contexts']}
            state['background_contexts_preserved'] = all((r['pid'],r['start_ticks']) in after
                for r in state.get('background_contexts',()))
        except Exception as exc:
            state.update(status="FAIL", close_error=repr(exc), daemon_closed=False)
        write_json(Path(state["directory"]) / "state.json", state)
        return state


def guard(campaign_directory, device, ready):
    session, stopped = Session(campaign_directory, device), False
    def stop(*unused):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        write_json(Path(ready), session.start())
        while not stopped:
            if not alive(session.state["daemon"]):
                raise RuntimeError("private MPS daemon died")
            time.sleep(.2)
    finally:
        final = session.close()
        if final and not final.get("daemon_closed"):
            raise RuntimeError("private MPS daemon closure was not confirmed")


def main():
    if len(sys.argv)>1 and sys.argv[1]=='exec':
        p=argparse.ArgumentParser()
        p.add_argument('--ready',type=Path,required=True)
        p.add_argument('argv',nargs=argparse.REMAINDER)
        args=p.parse_args(sys.argv[2:])
        argv=args.argv[1:] if args.argv[:1]==['--'] else args.argv
        if not argv:
            p.error('owned replica command required')
        return exec_replica(args.ready,argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("guard", "account", "state"))
    p.add_argument("--campaign-directory", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--clients", type=int, nargs="*", default=[])
    args = p.parse_args()
    if args.mode == "guard":
        return guard(args.campaign_directory, args.device, args.ready)
    initial = json.loads(args.ready.read_text())
    state = json.loads((Path(initial["directory"]) / "state.json").read_text())
    print(json.dumps(account(state, args.clients) if args.mode == "account" else state))


if __name__ == "__main__":
    main()
