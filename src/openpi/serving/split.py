"""Explicit, default-off CPU-front/GPU-stage launcher.

    python -m openpi.serving.split --model pi05 --fronts 8 --port 23110 -- \
        exp/offline_search/closed_loop/serve_pi05.py <ordinary server args>

Only this launcher installs split hooks. It uses fresh execs, never forks a
CUDA interpreter. The ordinary server CLI and worker protocol are unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import time

from openpi.serving.split_socket import POINTER, endpoint_ready, private_endpoint, read_endpoint


def _port_args(argv, port):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--port":
            if i + 1 == len(argv):
                raise ValueError("--port requires a value")
            i += 2
        elif argv[i].startswith("--port="):
            i += 1
        else:
            out.append(argv[i])
            i += 1
    # Tyro's subcommand should follow global flags.
    return ["--port", str(port), *out]


def _child(args, command):
    if args.role == "front" and os.environ.get("HF_MODULES_CACHE"):
        # Blueprints can reference processor classes loaded with HF
        # trust_remote_code. A fresh front must see the GPU's module cache
        # before unpickling, before its own entrypoint imports a processor.
        sys.path.append(str(Path(os.environ["HF_MODULES_CACHE"]).resolve()))
    from openpi.serving.split_runtime import install_front, install_gpu
    endpoint = read_endpoint(args.runtime_dir)
    blueprint = Path(args.runtime_dir) / "blueprint.pkl"
    if args.role == "gpu":
        os.environ["BATCHING_MAX_BATCH_SIZE"] = str(args.batch_size)
        os.environ["BATCHING_MAX_WAIT_MS"] = str(args.wait_ms)
        install_gpu(args.model, endpoint, blueprint, batch_size=args.batch_size,
                    wait_ms=args.wait_ms, max_channels=args.max_channels)
    else:
        install_front(args.model, endpoint, blueprint, capacity=args.arena_mb << 20,
                      timeout=args.timeout)
    if command[0] == "-m":
        sys.argv = [command[1], *command[2:]]
        runpy.run_module(command[1], run_name="__main__")
    else:
        sys.argv = command
        runpy.run_path(command[0], run_name="__main__")


async def _proxy(args, ports, processes):
    from openpi.serving.replica_proxy import ReplicaProxy, _MAX_WS_FRAME_BYTES
    from websockets.asyncio.server import serve
    from websockets.exceptions import WebSocketException
    proxy = ReplicaProxy("127.0.0.1", ports)
    async def ready(port):
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("split child exited while waiting for fronts")
            try:
                ws = await asyncio.wait_for(
                    proxy._open_backend(("127.0.0.1", port), swallow_metadata=True),
                    timeout=min(10, max(0.01, deadline - time.monotonic())))
                await ws.close()
                return
            except (OSError, TimeoutError, WebSocketException):
                await asyncio.sleep(0.25)
        raise TimeoutError(f"split front {port} startup timed out")
    await asyncio.gather(*(ready(p) for p in ports))
    await asyncio.wait_for(proxy.prime_metadata(), timeout=args.startup_timeout)
    async with serve(proxy.handle, args.host, args.port, max_size=_MAX_WS_FRAME_BYTES,
                     compression=None, ping_interval=None):
        while True:
            dead = [p for p in processes if p.poll() is not None]
            if dead:
                raise RuntimeError(f"split child exited: {[(p.pid, p.returncode) for p in dead]}; "
                                   "restart launcher and reload bundles before resuming episodes")
            await asyncio.sleep(0.5)


def _supervise(args, command):
    from openpi.serving.split_runtime import validate_arguments
    server_args = command[2:] if command[0] == "-m" else command[1:]
    validate_arguments(args.model, server_args)
    if args.fronts < 1 or args.batch_size < 1 or args.max_channels < 1 or args.arena_mb < 1:
        raise ValueError("fronts, batch-size, max-channels and arena-mb must be positive")
    if args.wait_ms < 0 or args.timeout <= 0 or args.startup_timeout <= 0:
        raise ValueError("wait-ms must be nonnegative; timeout and startup-timeout must be positive")
    if not 1 <= args.port <= 65535 - args.fronts:
        raise ValueError("split port and all child ports must be in 1..65535")
    processes = []
    # All persistent artifacts stay under the explicitly supplied worktree.
    with tempfile.TemporaryDirectory(prefix=".split-", dir=args.runtime_parent) as runtime, private_endpoint(runtime) as endpoint:
        common = ["-m", "openpi.serving.split", "--model", args.model,
                  "--runtime-dir", runtime, "--batch-size", str(args.batch_size),
                  "--wait-ms", str(args.wait_ms), "--max-channels", str(args.max_channels),
                  "--arena-mb", str(args.arena_mb), "--timeout", str(args.timeout)]
        base_env = dict(os.environ)
        base_env.pop("OPENPI_SPLIT_FRONT_ID", None)
        log_root = Path(args.log_root or Path(args.runtime_parent) / "exp/decouple/serve_logs" / Path(runtime).name)
        log_root.mkdir(parents=True, exist_ok=True)
        # Retain the address for postmortems after the ephemeral runtime exits.
        (log_root / POINTER).write_bytes((Path(runtime) / POINTER).read_bytes())
        (log_root / POINTER).chmod(0o600)
        base_env["OPENPI_SPLIT_LOG_ROOT"] = str(log_root.resolve())
        base_env["OPENPI_SPLIT_STATE_ROOT"] = str(Path(runtime).resolve() / "plugin_state")
        base_env["HF_MODULES_CACHE"] = str(Path(runtime).resolve() / "hf_modules")
        base_env["NO_ALBUMENTATIONS_UPDATE"] = "1"
        try:
            gpu = subprocess.Popen([args.gpu_python or sys.executable, *common, "--role", "gpu", "--", *command],
                                   env=base_env)
            processes.append(gpu)
            deadline = time.monotonic() + args.startup_timeout
            while not endpoint_ready(endpoint, Path(runtime) / "blueprint.pkl"):
                if gpu.poll() is not None:
                    raise RuntimeError(f"GPU child exited during startup: {gpu.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("GPU child startup timed out")
                time.sleep(0.25)
            ports = [args.port + 1 + i for i in range(args.fronts)]
            for index, port in enumerate(ports):
                env = dict(base_env, CUDA_VISIBLE_DEVICES="", OPENPI_SPLIT_FRONT_ID=str(index))
                entry = command[:2] if command[0] == "-m" else command[:1]
                child_args = _port_args(server_args, port)
                p = subprocess.Popen([args.front_python or sys.executable, *common, "--role", "front",
                                      "--", *entry, *child_args], env=env)
                processes.append(p)
            asyncio.run(_proxy(args, ports, processes))
        finally:
            # Signal only our own Popen children. No pkill/process discovery.
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            shutdown_end = time.monotonic() + 10
            survivors = []
            for p in processes:
                try:
                    p.wait(timeout=max(0, shutdown_end - time.monotonic()))
                except subprocess.TimeoutExpired:
                    survivors.append(p)
            for p in survivors:
                if p.poll() is None:
                    p.kill()
            for p in survivors:
                p.wait(timeout=10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi05", "groot"), required=True)
    parser.add_argument("--fronts", type=int, default=4)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gpu-python")
    parser.add_argument("--front-python")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wait-ms", type=float, default=0)
    parser.add_argument("--max-channels", type=int, default=128)
    parser.add_argument("--arena-mb", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--runtime-parent", default=".")
    parser.add_argument("--log-root", help="persistent isolated plugin/resident logs (default: exp/decouple/serve_logs/<run>)")
    parser.add_argument("--runtime-dir", help=argparse.SUPPRESS)
    parser.add_argument("--role", choices=("gpu", "front"), help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an existing serving entry point must follow --")
    if args.role:
        _child(args, command)
    else:
        def stop(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        _supervise(args, command)


if __name__ == "__main__":
    main()
