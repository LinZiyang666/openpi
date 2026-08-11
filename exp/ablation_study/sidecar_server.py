"""Sidecar policy server for the cache-effectiveness ablation (lerobot venv).

Serves a small student policy (SmolVLA, or a per-task ACT ensemble) over the
openpi websocket msgpack protocol so the main Pi0.5 server's routing hooks
(``CacheConfig.routing``) can forward hit/miss-slot observations here.

Protocol contract (mirrors openpi's WebsocketPolicyServer wire behaviour):
on accept the server SENDS one metadata dict first (the router's client recv's
it during the handshake), then answers each received obs dict with an outputs
dict ``{"actions": float32 [horizon, dim]}``. GPU forwards are serialised by a
lock; per-request queue/forward timings are appended to a JSONL log so the
pure-small latency anchor and the queue-confound split come from sidecar-side
measurements.

The websocket/protocol layer is dependency-light (websockets + openpi-client
msgpack codec) and unit-testable with a fake policy; lerobot imports happen
lazily inside the policy factories only.

Usage (lerobot venv on ziyang10):
    python exp/ablation_study/sidecar_server.py --policy smolvla \
        --checkpoint <dir> --port 7001 --timing-log /tmp/sidecar_smolvla.jsonl
    python exp/ablation_study/sidecar_server.py --policy act \
        --manifest exp/ablation_study/config/act_manifest_<suite>.json --port 7002
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from typing import Any, Callable

import numpy as np
import websockets.sync.server
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)

# Observation keys the router forwards verbatim from the LIBERO client
# (examples/libero/main.py element schema).
OBS_IMAGE = "observation/image"
OBS_WRIST = "observation/wrist_image"
OBS_STATE = "observation/state"
OBS_PROMPT = "prompt"


class SidecarServer:
    """Protocol server around a ``policy_fn(obs dict) -> actions ndarray``."""

    def __init__(
        self,
        policy_fn: Callable[[dict], np.ndarray],
        *,
        host: str = "127.0.0.1",
        port: int = 7001,
        metadata: dict | None = None,
        timing_log: str | None = None,
    ) -> None:
        self._policy_fn = policy_fn
        self._host = host
        self._port = port
        self._metadata = metadata or {"sidecar": True}
        self._timing_log = timing_log
        self._gpu_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._server: websockets.sync.server.Server | None = None

    def _record_timing(self, row: dict) -> None:
        if self._timing_log is None:
            return
        with self._log_lock, open(self._timing_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _handler(self, conn) -> None:
        packer = msgpack_numpy.Packer()
        conn.send(packer.pack(self._metadata))
        while True:
            try:
                data = conn.recv()
            except websockets.exceptions.ConnectionClosed:
                return
            obs = msgpack_numpy.unpackb(data)
            # Control messages (episode_start/end from a standalone main.py
            # client during student-val rollouts) are acknowledged, not
            # inferred: reply with an ack dict mirroring the ctrl name.
            if isinstance(obs, dict) and "__ctrl__" in obs:
                conn.send(packer.pack({"ack": obs["__ctrl__"]}))
                continue
            t_recv = time.perf_counter()
            with self._gpu_lock:
                t_start = time.perf_counter()
                try:
                    actions = np.asarray(self._policy_fn(obs), dtype=np.float32)
                except Exception as e:  # noqa: BLE001 — report as protocol error string
                    logger.exception("sidecar policy error")
                    conn.send(f"sidecar policy error: {e}")
                    continue
                t_end = time.perf_counter()
            self._record_timing(
                {
                    "ts": time.time(),
                    "prompt": str(obs.get(OBS_PROMPT, "")),
                    "queue_ms": (t_start - t_recv) * 1000.0,
                    "forward_ms": (t_end - t_start) * 1000.0,
                }
            )
            try:
                conn.send(packer.pack({"actions": actions}))
            except websockets.exceptions.ConnectionClosed:
                # Client timed out / went away between recv and send: normal
                # teardown, not a policy failure.
                return

    def serve_forever(self) -> None:
        with websockets.sync.server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None
        ) as server:
            self._server = server
            logger.info("sidecar listening on %s:%d", self._host, self._port)
            server.serve_forever()


# ----------------------------------------------------------------------
# Student policy factories (lerobot venv only; lazy imports)
# ----------------------------------------------------------------------


def make_smolvla_policy(checkpoint: str, device: str) -> Callable[[dict], np.ndarray]:
    """Load a finetuned SmolVLA checkpoint; obs dict -> [horizon, dim] chunk."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device).eval()

    def _infer(obs: dict) -> np.ndarray:
        batch = _obs_to_lerobot_batch(obs, device)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch)
        return chunk[0].cpu().numpy()

    return _infer


def route_prompt(policies: dict[str, Any], prompt: str) -> Any:
    """Exact-match prompt routing for the per-task ensemble (unknown -> raise).

    Pure helper so the routing contract is unit-testable without lerobot.
    """
    if prompt not in policies:
        raise KeyError(
            f"ACT manifest has no checkpoint for prompt {prompt!r}; "
            f"known: {sorted(policies)}"
        )
    return policies[prompt]


def make_act_policy(manifest_path: str, device: str) -> Callable[[dict], np.ndarray]:
    """Per-task ACT ensemble routed by exact prompt match (unknown -> raise)."""
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    with open(manifest_path, encoding="utf-8") as f:
        manifest: dict[str, str] = json.load(f)
    policies = {}
    for prompt, ckpt in manifest.items():
        p = ACTPolicy.from_pretrained(ckpt)
        p.to(device).eval()
        policies[prompt] = p

    def _infer(obs: dict) -> np.ndarray:
        policy = route_prompt(policies, str(obs.get(OBS_PROMPT, "")))
        batch = _obs_to_lerobot_batch(obs, device)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch)
        return chunk[0].cpu().numpy()

    return _infer


def _obs_to_lerobot_batch(obs: dict, device: str) -> dict[str, Any]:
    """Map the LIBERO client obs element onto the lerobot batch layout used at
    training time by build_distill_dataset.py (same keys, same preprocessing)."""
    import torch

    def _img(key: str):
        # uint8 HWC (resize_with_pad output, as recorded) -> float CHW in [0,1].
        arr = np.asarray(obs[key])
        t = torch.from_numpy(arr).to(device).permute(2, 0, 1).float() / 255.0
        return t[None]

    return {
        "observation.images.image": _img(OBS_IMAGE),
        "observation.images.wrist_image": _img(OBS_WRIST),
        "observation.state": torch.from_numpy(
            np.asarray(obs[OBS_STATE], dtype=np.float32)
        ).to(device)[None],
        "task": [str(obs.get(OBS_PROMPT, ""))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["smolvla", "act"], required=True)
    parser.add_argument("--checkpoint", default=None, help="SmolVLA checkpoint dir")
    parser.add_argument("--manifest", default=None, help="ACT prompt->checkpoint json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timing-log", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.policy == "smolvla":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --policy smolvla")
        policy_fn = make_smolvla_policy(args.checkpoint, args.device)
    else:
        if not args.manifest:
            raise SystemExit("--manifest is required for --policy act")
        policy_fn = make_act_policy(args.manifest, args.device)

    SidecarServer(
        policy_fn,
        host=args.host,
        port=args.port,
        metadata={"sidecar": True, "policy": args.policy},
        timing_log=args.timing_log,
    ).serve_forever()


if __name__ == "__main__":
    main()
