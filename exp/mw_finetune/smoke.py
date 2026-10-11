"""Own-process supervisor for 2-episode CPU/GPU smoke tests and safe cleanup."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="zero", choices=["zero", "pi05", "groot", "fastwam", "cosmos"])
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--python", default=sys.executable, help="model island Python")
    ap.add_argument("--worker-python", default="/home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18050)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600, help="total smoke wall seconds, maximum 900")
    ap.add_argument("--policy-option", action="append", default=[])
    a = ap.parse_args()
    if not 0 < a.timeout <= 900:
        ap.error("smoke timeout must be 1..900 seconds")
    a.run.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src"), str(root / "packages/openpi-client/src"), env.get("PYTHONPATH", "")])
    env["LD_LIBRARY_PATH"] = "/data/mw_finetune/envs/osmesa/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    env["MUJOCO_GL"] = "osmesa"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    # Limit llvmpipe threads per renderer; 12 processes must not each start 88 threads.
    env["LP_NUM_THREADS"] = "1"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if a.model == "zero":
        env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [a.python, "-m", "exp.mw_finetune.serve", "--model", a.model, "--port", str(a.port)]
    if a.checkpoint:
        cmd += ["--checkpoint", str(a.checkpoint)]
    for option in a.policy_option:
        cmd += ["--policy-option", option]
    lock = None
    if a.model != "zero":
        lock = open("/tmp/r17_gpu.lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
        (a.run / "gpu_before.txt").write_text(check)
        used, total = map(int, check.splitlines()[0].split(","))
        if used > 1024:
            raise RuntimeError(f"GPU not free before smoke: {used}/{total} MiB; leave existing processes untouched")
    started = time.monotonic()
    deadline = started + a.timeout
    server = None
    evaluation = None
    try:
        with (a.run / "server.log").open("a") as out:
            server = subprocess.Popen(cmd, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited {server.returncode}; see {a.run / 'server.log'}")
                try:
                    with socket.create_connection(("127.0.0.1", a.port), timeout=1):
                        break
                except OSError:
                    time.sleep(0.5)
            else:
                raise TimeoutError("server startup exceeded smoke deadline")
            eval_cmd = [a.worker_python, "-m", "exp.mw_finetune.evaluate", "--smoke", "--model-id", a.model + "_smoke",
                        "--port", str(a.port), "--workers", str(a.workers), "--run", str(a.run / "eval")]
            worker_env = dict(env, CUDA_VISIBLE_DEVICES="")
            with (a.run / "eval.log").open("a") as eval_out:
                evaluation = subprocess.Popen(eval_cmd, cwd=root, env=worker_env, stdout=eval_out, stderr=subprocess.STDOUT, start_new_session=True)
                evaluation.wait(timeout=max(1, deadline - time.monotonic()))
            if evaluation.returncode:
                raise RuntimeError(f"evaluation exited {evaluation.returncode}; see {a.run / 'eval.log'}")
            summary = json.loads((a.run / "eval/summary.json").read_text())
            result = {"model": a.model, "workers": a.workers, "seconds": time.monotonic() - started,
                      "completed_episodes": summary["completed_episodes"], "sr": summary["overall_per_task_mean"]}
            (a.run / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        # Only processes created by this supervisor are signaled.
        for process in (evaluation, server):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        if lock:
            lock.close()


if __name__ == "__main__":
    main()
