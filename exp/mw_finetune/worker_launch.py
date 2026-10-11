"""Start/resume the evaluator on a timan node, optionally in r17_<model> tmux."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import shlex
import socket
import subprocess

PORTS = {"pi05": 50, "groot": 51, "fastwam": 52, "cosmos": 53, "zero": 59}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/scratch/zixuans8/mw_r17"))
    ap.add_argument("--model", choices=PORTS, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model-id")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--foreground", action="store_true")
    a, extra = ap.parse_known_args()
    node = socket.gethostname().split(".")[0]
    if node not in ("timan107", "timan108"):
        ap.error("formal driver must run on timan107 or timan108")
    if not 1 <= a.workers <= 30:
        ap.error("timan worker limit is 30")
    if Path(a.run_name).name != a.run_name or a.run_name in (".", ".."):
        ap.error("run-name must be a single directory name")
    host, base = ("ziyanglin.com", 23100) if node == "timan107" else ("149.165.153.233", 23200)
    root = a.root.resolve()
    code = root / "code"
    run = root / "runs" / a.run_name
    run.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONPATH=f"{code}:{code}/src:{code}/packages/openpi-client/src",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", LP_NUM_THREADS="1",
                       CUDA_VISIBLE_DEVICES="", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    argv = [str(root / "envs/sim/bin/python"), "-m", "exp.mw_finetune.evaluate", "--run", str(run),
            "--model-id", a.model_id or a.run_name, "--host", host, "--port", str(base + PORTS[a.model]),
            "--workers", str(a.workers), "--renderer", "egl", *extra]
    if a.foreground:
        os.chdir(code)
        os.execvpe(argv[0], argv, environment)
    session = f"r17_{a.model}"
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        ap.error(f"tmux {session} already exists; inspect it before starting another driver")
    keys = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LP_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "PYOPENGL_PLATFORM")
    command = "cd " + shlex.quote(str(code)) + " && exec env "
    command += " ".join(shlex.quote(f"{k}={environment[k]}") for k in keys) + " " + shlex.join(argv)
    command += " > " + shlex.quote(str(run / "driver.log")) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
    print(f"{node}: tmux {session}; journal {run}; lane {host}:{base + PORTS[a.model]}")


if __name__ == "__main__":
    main()
