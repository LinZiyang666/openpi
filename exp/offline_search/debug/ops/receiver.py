"""Coordinator-only receiver startup/reuse; never started by packaging tests."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

from exp.offline_search.debug.transport.protocol import request
from exp.offline_search.debug.transport.receiver import atomic_json


def ensure(run, port=23199):
    run = Path(run).resolve()
    # timan107 reaches weilandserver only through the public 23100-23199 segment; the coordinator picks a free port.
    if not 1 <= port <= 65535:
        raise ValueError("receiver port out of range")
    state = run / "state"
    state.mkdir(parents=True, exist_ok=True)
    token_path, ready = state / "osdebug_stream.token", state / "osdebug_receiver.json"
    if not token_path.exists():
        fd = os.open(str(token_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_hex(32) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    token = token_path.read_text().strip()
    if ready.exists():
        saved = json.loads(ready.read_text())
        address = "127.0.0.1:" + str(saved["port"])
        try:
            response = request(address, dict(op="health", run=run.name, token=token))
            if response["run"] != run.name:
                raise ValueError("receiver belongs to another run")
            return saved
        except (OSError, RuntimeError, EOFError):
            # An existing receiver of another run is never killed/replaced.
            pass
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=".:src", CUDA_VISIBLE_DEVICES="")
    with (state / "osdebug_receiver.log").open("ab") as log:
        process = subprocess.Popen(["taskset", "-c", os.environ.get("OPS_CPUS", "18-21,62-65"), sys.executable,
                                    "-m", "exp.offline_search.debug.transport.receiver", "--run-root", str(run),
                                    "--port", str(port), "--token-file", str(token_path), "--ready-file", str(ready)],
                                   stdout=log, stderr=log, env=env, start_new_session=True)
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("receiver exited; inspect " + str(state / "osdebug_receiver.log"))
        try:
            request("127.0.0.1:" + str(port), dict(op="health", run=run.name, token=token))
            result = dict(run=run.name, port=port, pid=process.pid, bind="0.0.0.0")
            atomic_json(ready, result)
            return result
        except (OSError, RuntimeError, EOFError):
            time.sleep(.1)
    raise RuntimeError("receiver did not become healthy")


def probe_remote(run):
    """Coordinator-only short TCP reachability check before model startup."""
    saved = json.loads((Path(run) / "state/osdebug_receiver.json").read_text())
    host = os.environ.get("OSDEBUG_RECEIVER_HOST", "ziyanglin.com")
    port = int(saved["port"])
    script = "import socket,sys; sock=socket.create_connection((sys.argv[1],int(sys.argv[2])),timeout=3); sock.close()"
    subprocess.run(["tether", "exec", "timan107", "--", "/scratch/zixuans8/openpi/.venv/bin/python",
                    "-c", script, host, str(port)], check=True, capture_output=True, text=True, timeout=15)
    return dict(reachable=True, from_host="timan107", host=host, port=port)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--port", type=int, default=int(os.environ.get("OSDEBUG_RECEIVER_PORT", "23199")))
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--ensure", action="store_true")
    action.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        try:
            result = probe_remote(a.run_root)
        except (OSError, subprocess.SubprocessError) as exc:
            print("receiver TCP probe from timan107 failed: " + str(exc), file=sys.stderr)
            raise SystemExit(1)
    else:
        result = ensure(a.run_root, a.port)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
