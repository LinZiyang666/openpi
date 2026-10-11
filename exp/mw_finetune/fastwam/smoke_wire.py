"""Checkpoint reload through A's actual server and msgpack WebSocket protocol.

Only the subprocess created here is terminated. No environment success-rate
claim is made; the observation comes from the prepared training dataset.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .data import Dataset
from .shims import metaworld_tasks as tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("R17_GPU_LOCK_HELD") != "1":
        raise RuntimeError("Acquire /tmp/r17_gpu.lock before this serving smoke")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    begin = time.perf_counter()
    with output.with_suffix(".server.log").open("w") as log:
        process = subprocess.Popen([
            sys.executable, "-m", "exp.mw_finetune.fastwam.serve_compat",
            "--model", "fastwam", "--checkpoint", args.checkpoint,
            "--host", "127.0.0.1", "--port", str(port),
        ], stdout=log, stderr=subprocess.STDOUT)
        client = None
        try:
            deadline = begin + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"A's server exited {process.returncode}; see {log.name}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        break
                except OSError:
                    if time.perf_counter() > deadline:
                        raise TimeoutError("A's server did not listen within 180s")
                    time.sleep(.5)
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            client = WebsocketClientPolicy("127.0.0.1", port)
            metadata = client.get_server_metadata()
            assert metadata["model"] == "fastwam" and metadata["pure_inference"]
            client.select_bundle("default")
            data = Dataset(args.data, allow_partial=True)
            obs = {
                "observation/image": np.asarray(Image.fromarray(data._image(0)).resize((480, 480))),
                "observation/state": data.index["state"][0],
                "prompt": tasks.PROMPTS["reach-v3"],
            }
            actions, timings = [], []
            for _ in range(2):
                client.episode_start(tasks.BENCHMARK, "reach-v3", 0,
                    extra_metadata={"orig_init_state_idx": 0, "seed": 7})
                t0 = time.perf_counter()
                result = client.infer(obs)["actions"]
                timings.append(time.perf_counter() - t0)
                assert result.shape == (32, 4) and result.dtype == np.float32
                assert np.isfinite(result).all()
                actions.append(result)
                client.episode_end(False)
            np.testing.assert_array_equal(*actions)
            report = {"passed": True, "server": "A exp.mw_finetune.serve",
                "wire": "msgpack WebSocket; select_bundle/episode_start/infer/episode_end",
                "actions_shape": [32, 4], "actions_dtype": "float32", "finite": True,
                "episode_restart_same_seed_exact": True, "request_s": timings,
                "wall_s": time.perf_counter() - begin, "metadata": metadata,
                "scope": "serving mechanics on a training observation; not an SR evaluation"}
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in report.items() if k != "metadata"}), flush=True)
        finally:
            if client is not None:
                client._ws.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
