"""Bounded two-episode timan107 -> local GPU smoke; wait for the shared GPU lock."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from exp.mw_finetune.client import BoundedClient


def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["pi05", "groot", "fastwam", "cosmos"], default="pi05")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--port", type=int)
    ap.add_argument("--root", default="/scratch/zixuans8/mw_r17")
    ap.add_argument("--timeout", type=int, default=285, help="including startup; maximum285+10s owned-process cleanup")
    a = ap.parse_args()
    expected_port = 23100 + {"pi05": 50, "groot": 51, "fastwam": 52, "cosmos": 53}[a.model]
    a.port = a.port or expected_port
    if a.port != expected_port:
        ap.error(f"{a.model} uses fixed timan107 port{expected_port}")
    if socket.gethostname().split(".")[0] != "weilandserver" or not 23100 <= a.port <= 23199:
        ap.error("this smoke uses the weilandserver -> timan107 lane")
    if not 1 <= a.timeout <= 285:
        ap.error("GPU lock hold is bounded to <300s")
    if Path(a.run_name).name != a.run_name or a.run_name in (".", ".."):
        ap.error("run-name must be one directory name")
    report = Path("/data/mw_finetune/reports") / a.run_name
    report.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, PYTHONPATH=f"{repo}:{repo}/src:{repo}/packages/openpi-client/src:" + os.environ.get("PYTHONPATH", ""),
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false")
    server, evaluation = None, None
    print("Waiting for /tmp/r17_gpu.lock; smoke clock starts after acquisition", flush=True)
    with open("/tmp/r17_gpu.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
        (report / "gpu_before.txt").write_text(before)
        if int(before.splitlines()[0].split(",")[0]) > 1024:
            raise RuntimeError("GPU memory occupied after lock acquisition; leave other processes untouched")
        with socket.socket() as check:
            check.bind(("0.0.0.0", a.port))
        started = time.monotonic()
        deadline = started + a.timeout
        try:
            with (report / "server.log").open("a") as output:
                server = subprocess.Popen([a.python, "-m", "exp.mw_finetune.serve", "--model", a.model,
                    "--checkpoint", a.checkpoint, "--port", str(a.port)], cwd=repo, env=environment,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"server exited {server.returncode}; inspect {report}/server.log")
                    try:
                        ready = BoundedClient("127.0.0.1", a.port)
                        ready.get_server_metadata()
                        ready.close()
                        break
                    except OSError:
                        time.sleep(0.25)
                else:
                    raise TimeoutError("server startup exceeded smoke deadline")
                server_ready = time.monotonic() - started
                argv = ["tether", "exec", "--timeout", f"{max(1,int(deadline-time.monotonic()))}s", "timan107", "--", "env",
                        f"PYTHONPATH={a.root}/code:{a.root}/code/src:{a.root}/code/packages/openpi-client/src",
                        a.root + "/envs/sim/bin/python", "-m", "exp.mw_finetune.worker_launch", "--model", a.model,
                        "--run-name", a.run_name, "--workers", "2", "--foreground", "--smoke"]
                with (report / "eval.log").open("a") as output_eval:
                    evaluation = subprocess.Popen(argv, stdout=output_eval, stderr=subprocess.STDOUT, start_new_session=True)
                    evaluation.wait(timeout=max(1, deadline - time.monotonic()))
                if evaluation.returncode:
                    raise RuntimeError(f"remote evaluation exited {evaluation.returncode}; inspect {report}/eval.log")
        finally:
            stop_owned(evaluation)
            stop_owned(server)
            result = {"model": a.model, "node": "timan107", "renderer": "egl", "seconds": time.monotonic() - started,
                      "server_ready_s": locals().get("server_ready"),
                      "evaluation_returncode": evaluation.returncode if evaluation else None}
            (report / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    # Transfer verification happens after releasing the GPU.
    subprocess.run([sys.executable, "-m", "exp.mw_finetune.sync_back", "timan107", a.run_name, "--root", a.root], check=True)


if __name__ == "__main__":
    main()
