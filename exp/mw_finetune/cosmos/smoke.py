"""Run A's actual WebSocket harness for assembly-v3 and reach-v3, one episode each.

Launch under flock /tmp/r17_gpu.lock; the owning process bounds and cleans up
only its own child server. Maximum wall time is 850 seconds by default.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="/data/mw_finetune/models/cosmos_metaworld")
    ap.add_argument("--run", type=Path, default=Path("exp/mw_finetune/cosmos/results/smoke_single"))
    ap.add_argument("--sim-python", default="/home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python")
    ap.add_argument("--port", type=int, default=8054)
    ap.add_argument("--seconds", type=int, default=850)
    ap.add_argument("--camera-layout", default="single")
    ap.add_argument("--denoising-steps", type=int, default=5)
    a = ap.parse_args()
    if not 0 < a.seconds <= 850:
        ap.error("seconds must be 1..850")
    a.run.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with socket.socket() as check:
        check.bind(("127.0.0.1", a.port))
    free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    if int(free.strip().splitlines()[0]) < 10000:
        raise RuntimeError(f"4090 has insufficient free memory for smoke: {free.strip()} MiB")
    server_cmd = [sys.executable, "-m", "exp.mw_finetune.serve", "--model", "cosmos", "--checkpoint", a.checkpoint,
                  "--host", "127.0.0.1", "--port", str(a.port), "--denoising-steps", str(a.denoising_steps),
                  "--policy-option", "camera_layout=" + json.dumps(a.camera_layout)]
    result = {"server_command": server_cmd, "tasks": ["assembly-v3", "reach-v3"], "trials": 1,
              "gpu_free_mib_before_start": int(free.strip().splitlines()[0])}
    server = None
    try:
        with (a.run / "server.log").open("w") as server_log:
            server = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited with {server.returncode}; see server.log")
                if time.monotonic() - started > a.seconds:
                    raise TimeoutError("model startup exceeded smoke budget")
                try:
                    with socket.create_connection(("127.0.0.1", a.port), timeout=1):
                        break
                except OSError:
                    time.sleep(.5)
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            from exp.metaworld import tasks as T
            client = WebsocketClientPolicy("127.0.0.1", a.port)
            result["server_metadata"] = client.get_server_metadata()
            if result["server_metadata"].get("model") != "cosmos":
                raise RuntimeError("connected endpoint is not the launched Cosmos server")
            obs = {"observation/image": np.zeros((480, 480, 3), np.uint8),
                   "observation/state": np.asarray([0, .6, .2, .5], np.float32),
                   "prompt": T.PROMPTS["assembly-v3"]}
            outputs, times = [], []
            for _ in range(2):
                client.episode_start(experiment=T.BENCHMARK, task="assembly-v3", episode_id=49,
                                     extra_metadata={"orig_init_state_idx": 49, "seed": 7})
                t = time.monotonic()
                outputs.append(np.asarray(client.infer(obs)["actions"]))
                times.append(time.monotonic() - t)
                client.episode_end(success=False)
            client.close()
            actions = outputs[0]
            np.save(a.run / "synthetic_actions.npy", actions)
            result["synthetic_probe"] = {"shape": list(actions.shape), "dtype": str(actions.dtype),
                                          "finite": bool(np.isfinite(actions).all()), "latency_s": times,
                                          "repeat_max_abs_diff": float(np.max(np.abs(outputs[0] - outputs[1]))),
                                          "per_channel_min": actions.min(0).tolist(),
                                          "per_channel_max": actions.max(0).tolist()}
            eval_cmd = [a.sim_python, "-m", "exp.mw_finetune.evaluate", "--run", str(a.run / "eval"),
                        "--model-id", "cosmos_mw_" + a.camera_layout, "--host", "127.0.0.1", "--port", str(a.port),
                        "--workers", "1", "--tasks", *result["tasks"], "--trials", "1"]
            result["eval_command"] = eval_cmd
            env = dict(os.environ)
            env.update(MUJOCO_GL="osmesa", LIBGL_ALWAYS_SOFTWARE="1", OMP_NUM_THREADS="1", LP_NUM_THREADS="1")
            env["LD_LIBRARY_PATH"] = "/data/mw_finetune/envs/osmesa/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
            remaining = a.seconds - (time.monotonic() - started)
            with (a.run / "eval.log").open("w") as eval_log:
                evaluation = subprocess.run(eval_cmd, env=env, stdout=eval_log, stderr=subprocess.STDOUT,
                                            timeout=max(1, remaining))
            result["eval_returncode"] = evaluation.returncode
            if evaluation.returncode:
                raise RuntimeError(f"evaluation exited with {evaluation.returncode}; see eval.log")
    except Exception as e:
        result["error"] = repr(e)
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        result["wall_s"] = time.monotonic() - started
        (a.run / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
