"""Websocket server that exposes the upstream Cosmos Policy ``get_action`` to remote LIBERO clients.

The official LIBERO evaluation runs simulator and model in one process. To put the
model on a server GPU (h100) and the simulators on a client box (timan107), this
server loads the model exactly as ``run_libero_eval.eval_libero`` does (same
``PolicyEvalConfig`` flags, same ``get_model`` / dataset stats / T5 cache) and answers
one request per policy query with the upstream ``get_action``. Nothing about the
denoising path is changed; ``num_denoising_steps_action`` and the sampling ``seed``
come with each request, so one server can serve every k of the ladder.

Protocol (msgpack + msgpack_numpy over websockets, one request per message)::

    request  = {"obs": {"primary_image", "wrist_image", "proprio"}, "task_description": str,
                "seed": int, "randomize_seed": bool, "num_denoising_steps_action": int,
                "generate_future_state_and_value_in_parallel": bool, "deterministic": bool}
    response = {"actions": float32 [K, 7], "value_prediction": float | None,
                "future_image_predictions": {name: uint8 image | None} | None, "server_ms": float}
    {"ping": 1} -> {"pong": 1, "num_denoising_steps_action": <cfg default>, "ckpt_path": ...}

Requests are served one at a time on a single model thread; connections queue.
Usage (h100, cosmos-policy venv, PYTHONPATH containing exp/)::

    python -m exp.cosmos_nfe.serve_cosmos --port 23240 <PolicyEvalConfig flags as in LIBERO.md>
    python -m exp.cosmos_nfe.serve_cosmos --bench robocasa --port 23250 --no-future-decode 1 --cuda-graph 1 <flags>

``--no-future-decode 1`` skips the future-image VAE decode (policy-only serving, bitwise-identical actions);
``--cuda-graph 1`` replays the DiT forward from a CUDA graph (``exp.cosmos_nfe.cudagraph_net``); both are
warmed up before the port opens. Measured on H100 (RoboCasa, k=5): 821 -> 576 -> 470 ms per query.

``--bench`` picks the upstream eval module whose ``PolicyEvalConfig`` / ``validate_config`` are
used (``libero`` default, ``robocasa`` = the 24-task RoboCasa 2024 benchmark; its config module
imports robosuite/robocasa, so the fork must be installed on the server too). ``get_action`` is
the same upstream function for both; a RoboCasa observation carries three images.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import socket
import sys
import threading
import time

import msgpack
import msgpack_numpy
import numpy as np
import websockets.asyncio.server as ws_server
from websockets.exceptions import ConnectionClosed

msgpack_numpy.patch()


def _take(argv: list[str], flag: str, default: str) -> str:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            val = argv[i + 1]
            del argv[i : i + 2]
            return val
        if tok.startswith(flag + "="):
            del argv[i]
            return tok.split("=", 1)[1]
    return default


def _synthetic_obs(bench: str):
    """A random observation of the benchmark's shape plus a cached instruction, only for warm-up / graph capture.

    The instruction must already be in the T5 cache: an unknown one would make upstream load the T5 encoder.
    """
    import cosmos_policy.experiments.robot.cosmos_utils as CU

    task = next(iter(CU.t5_text_embeddings_cache))
    rng = np.random.default_rng(0)
    img = lambda: rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    if bench == "libero":
        return {"primary_image": img(), "wrist_image": img(), "proprio": rng.standard_normal(8).astype(np.float32)}, task
    return {"primary_image": img(), "secondary_image": img(), "wrist_image": img(),
            "proprio": rng.standard_normal(9).astype(np.float32)}, task


def main() -> None:
    argv = sys.argv[1:]
    port = int(_take(argv, "--port", "23240"))
    host = _take(argv, "--host", "0.0.0.0")
    bench = _take(argv, "--bench", "libero")
    no_future = _take(argv, "--no-future-decode", "0") == "1"
    cuda_graph = _take(argv, "--cuda-graph", "0") == "1"

    import draccus

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )
    from cosmos_policy.utils.utils import set_seed_everywhere

    if bench == "libero":
        from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig, validate_config
    elif bench == "robocasa":
        from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import PolicyEvalConfig, validate_config
    else:
        raise SystemExit(f"--bench must be libero or robocasa, got {bench}")

    cfg = draccus.parse(config_class=PolicyEvalConfig, args=argv)
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, cosmos_config = get_model(cfg)
    assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size
    if no_future:
        # policy-only serving: the Wan-VAE decode of the whole 11-frame latent video only feeds the future-image
        # visualisation (~250 ms/query on H100); the action chunk and the value read from the latent are untouched
        import cosmos_policy.experiments.robot.cosmos_utils as CU

        CU.get_future_images_from_generated_samples = lambda *a, **k: {}
        print("COSMOS_SERVER no-future-decode: future-image VAE decode skipped", flush=True)
    if cuda_graph:
        from exp.cosmos_nfe.cudagraph_net import GraphedNet

        model.net = GraphedNet(model.net, warmup=2)
    # warm up (and, with --cuda-graph, capture) before listening so the first client never waits on it
    _warm = _synthetic_obs(bench)
    for i in range(3):
        if cfg.deterministic:
            set_seed_everywhere(0)
        get_action(cfg, model, dataset_stats, _warm[0], _warm[1], seed=cfg.seed, randomize_seed=False,
                   num_denoising_steps_action=cfg.num_denoising_steps_action, generate_future_state_and_value_in_parallel=True)
    print(f"COSMOS_SERVER warm-up done cuda_graph={cuda_graph} no_future_decode={no_future}", flush=True)
    print(f"COSMOS_SERVER model ready bench={bench} ckpt={cfg.ckpt_path} default_k={cfg.num_denoising_steps_action}", flush=True)

    lock = threading.Lock()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    stats = {"n": 0, "ms": 0.0}

    def handle(req: dict) -> dict:
        t0 = time.perf_counter()
        with lock:
            if req.get("deterministic", cfg.deterministic):
                set_seed_everywhere(0)  # mirrors the per-step reset in run_episode
            out = get_action(
                cfg,
                model,
                dataset_stats,
                req["obs"],
                req["task_description"],
                seed=int(req.get("seed", cfg.seed)),
                randomize_seed=bool(req.get("randomize_seed", cfg.randomize_seed)),
                num_denoising_steps_action=int(req.get("num_denoising_steps_action", cfg.num_denoising_steps_action)),
                generate_future_state_and_value_in_parallel=bool(req.get("generate_future_state_and_value_in_parallel", True)),
            )
        ms = (time.perf_counter() - t0) * 1000.0
        stats["n"] += 1
        stats["ms"] += ms
        if stats["n"] % 200 == 0:
            print(f"COSMOS_SERVER served={stats['n']} mean_ms={stats['ms'] / stats['n']:.1f}", flush=True)
        fut = out.get("future_image_predictions")
        if isinstance(fut, dict):
            fut = {k: (np.asarray(v) if v is not None else None) for k, v in fut.items()}
        return {
            "actions": np.stack([np.asarray(a, dtype=np.float32) for a in out["actions"]], axis=0),
            "value_prediction": out.get("value_prediction"),
            "future_image_predictions": fut,
            "server_ms": ms,
        }

    async def serve(conn):
        loop = asyncio.get_running_loop()
        try:
            async for raw in conn:
                req = msgpack.unpackb(raw, raw=False)
                if "ping" in req:
                    resp = {"pong": 1, "num_denoising_steps_action": cfg.num_denoising_steps_action, "ckpt_path": cfg.ckpt_path,
                            "host": socket.gethostname(), "bench": bench}
                else:
                    resp = await loop.run_in_executor(executor, handle, req)
                await conn.send(msgpack.packb(resp, use_bin_type=True))
        except ConnectionClosed:
            pass

    async def run():
        async with ws_server.serve(serve, host, port, max_size=64 * 1024 * 1024, ping_interval=None):
            print(f"COSMOS_SERVER listening on {host}:{port}", flush=True)
            await asyncio.Future()

    asyncio.run(run())


if __name__ == "__main__":
    main()
