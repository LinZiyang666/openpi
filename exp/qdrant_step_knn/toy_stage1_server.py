"""Toy Stage-1-only inference server.

Runs the Pi0.5 vision encoder + prompt embedding (Stage 1) and delegates
action retrieval to an external Qdrant query server over HTTP.

The WebSocket protocol is identical to ``scripts/serve_policy.py`` so
existing simulator clients (e.g. ``examples/libero/main.py``) work
without modification.

Usage
-----
::

    uv run exp/qdrant_step_knn/toy_stage1_server.py \\
        --env LIBERO \\
        --port 8000 \\
        --qdrant-server-url http://localhost:8100 \\
        policy:checkpoint \\
        --policy.config pi05_libero \\
        --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
"""

from __future__ import annotations

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import dataclasses
import enum
import logging
import math
import socket
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch
import tyro
from openpi_client import base_policy as _base_policy

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI args (subset of serve_policy.py, no --cache / --collect)
# ---------------------------------------------------------------------------


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(config="pi05_aloha", dir="gs://openpi-assets/checkpoints/pi05_base"),
    EnvMode.ALOHA_SIM: Checkpoint(config="pi05_aloha_sim", dir="gs://openpi-assets/checkpoints/pi05_base"),
    EnvMode.DROID: Checkpoint(config="pi05_droid", dir="gs://openpi-assets/checkpoints/pi05_droid"),
    EnvMode.LIBERO: Checkpoint(config="pi05_libero", dir="gs://openpi-assets/checkpoints/pi05_libero"),
}


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.LIBERO
    default_prompt: str | None = None
    port: int = 8000
    qdrant_server_url: str = "http://localhost:8100"
    stage1_device: str | None = None
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# ---------------------------------------------------------------------------
# Policy creation (reused from serve_policy.py logic)
# ---------------------------------------------------------------------------


def _create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                pytorch_device="cpu",
            )
        case Default():
            checkpoint = DEFAULT_CHECKPOINT.get(args.env)
            if checkpoint is None:
                raise ValueError(f"No default checkpoint for env: {args.env}")
            return _policy_config.create_trained_policy(
                _config.get_config(checkpoint.config),
                checkpoint.dir,
                default_prompt=args.default_prompt,
                pytorch_device="cpu",
            )


# ---------------------------------------------------------------------------
# ToyStage1Policy — runs Stage 1 only, queries external qdrant server
# ---------------------------------------------------------------------------


class ToyStage1Policy(_base_policy.BasePolicy):
    """Runs Stage 1 (vision + prompt embedding) and retrieves actions from
    an external Qdrant query server.

    Implements ``BasePolicy`` so it can be passed directly to
    ``WebsocketPolicyServer``.
    """

    def __init__(self, policy: _policy.Policy, qdrant_server_url: str, stage1_device: str | None = None) -> None:
        if not policy._is_pytorch_model:
            raise ValueError("ToyStage1Policy only supports PyTorch policies.")

        self._policy = policy
        self._model = policy._model
        self._input_transform = policy._input_transform
        self._output_transform = policy._output_transform
        self._qdrant_url = qdrant_server_url.rstrip("/")
        self._step_counter = 0
        self._stage1_device = self._resolve_stage1_device(stage1_device)

        # Lazy imports for HTTP client
        self._requests = None
        self._msgpack = None
        self._msgpack_numpy = None

        self._configure_stage1_device_placement()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def _resolve_stage1_device(self, requested_device: str | None) -> torch.device:
        # Keep device selection local to the toy server so shared policy loading code stays unchanged.
        if requested_device is None:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"stage1_device={requested_device!r} requested, but CUDA is not available.")
        return device

    def _configure_stage1_device_placement(self) -> None:
        # Load the checkpoint on CPU first, then move only the Stage 1 modules to the accelerator.
        paligemma = self._model.paligemma_with_expert.paligemma
        paligemma.vision_tower.to(self._stage1_device)
        paligemma.multi_modal_projector.to(self._stage1_device)
        paligemma.language_model.embed_tokens.to(self._stage1_device)

        # CPU execution is a fallback path; keep Stage 1 compute in float32 there.
        if self._stage1_device.type == "cpu":
            paligemma.vision_tower.to(dtype=torch.float32)
            paligemma.multi_modal_projector.to(dtype=torch.float32)
            paligemma.language_model.embed_tokens.to(dtype=torch.float32)

        logger.info(
            "ToyStage1Policy device placement: stage1_device=%s, model_load_device=%s",
            self._stage1_device,
            self._policy._pytorch_device,
        )

    def _ensure_http_deps(self):
        if self._requests is None:
            import msgpack
            import msgpack_numpy
            import requests

            msgpack_numpy.patch()
            self._requests = requests
            self._msgpack = msgpack
            self._msgpack_numpy = msgpack_numpy

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        # ---- 1. Input transforms (mirrors Policy.infer) ----
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        robot_state_np = np.asarray(inputs["state"], dtype=np.float32).flatten()

        # ---- 2. Convert to torch tensors ----
        torch_inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._stage1_device)[None, ...],
            inputs,
        )
        observation = _model.Observation.from_dict(torch_inputs)

        # ---- 3. Run stage1 with forward hooks ----
        vision_captures: list[torch.Tensor] = []
        lang_capture: list[torch.Tensor | None] = [None]

        def _vision_hook(_module, _inp, out):
            vision_captures.append(out.detach().clone())

        def _lang_hook(_module, _inp, out):
            lang_capture[0] = (out * math.sqrt(out.shape[-1])).detach().clone()

        model = self._model
        handles = [
            model.paligemma_with_expert.paligemma.multi_modal_projector.register_forward_hook(_vision_hook),
            model.paligemma_with_expert.paligemma.language_model.embed_tokens.register_forward_hook(_lang_hook),
        ]
        try:
            with torch.no_grad():
                _stage1 = model.run_stage1(observation)
        finally:
            for h in handles:
                h.remove()

        # ---- 4. Extract embeddings as numpy ----
        vision_embs = [v.squeeze(0).cpu().to(torch.float16).numpy() for v in vision_captures]
        if len(vision_embs) < 3:
            raise RuntimeError(
                f"Expected 3 vision captures, got {len(vision_embs)}. "
                "The model may not have 3 image inputs."
            )
        prompt_emb = lang_capture[0]
        if prompt_emb is None:
            raise RuntimeError("embed_tokens hook did not fire during run_stage1.")
        prompt_emb_np = prompt_emb.squeeze(0).cpu().to(torch.float16).numpy()

        # ---- 5. Query qdrant server ----
        clean_action = self._query_qdrant_server(vision_embs, prompt_emb_np, robot_state_np, self._step_counter)
        self._step_counter += 1

        # ---- 6. Apply output transforms ----
        action_tensor = torch.from_numpy(clean_action)[None, ...]
        outputs: dict[str, Any] = {
            "state": torch_inputs["state"],
            "actions": action_tensor,
        }
        outputs = jax.tree.map(
            lambda x: np.asarray(x[0, ...].detach().cpu()) if isinstance(x, torch.Tensor) else np.asarray(x[0, ...]),
            outputs,
        )
        outputs = self._output_transform(outputs)
        return outputs

    def _query_qdrant_server(
        self,
        vision_embs: list[np.ndarray],
        prompt_emb: np.ndarray,
        robot_state: np.ndarray,
        step_idx: int,
    ) -> np.ndarray:
        """Send embeddings to the qdrant query server and receive clean_action."""
        self._ensure_http_deps()

        payload = {
            "vision_0": vision_embs[0],
            "vision_1": vision_embs[1],
            "vision_2": vision_embs[2],
            "prompt_emb": prompt_emb,
            "robot_state": robot_state,
            "step_idx": step_idx,
        }
        data = self._msgpack.packb(payload, default=self._msgpack_numpy.encode)
        resp = self._requests.post(
            f"{self._qdrant_url}/query",
            data=data,
            headers={"Content-Type": "application/x-msgpack"},
        )
        resp.raise_for_status()
        result = self._msgpack.unpackb(resp.content, object_hook=self._msgpack_numpy.decode)
        clean_action = np.asarray(result["clean_action"], dtype=np.float32)
        return clean_action

    # -- Episode lifecycle (called by WebsocketPolicyServer) --

    def on_episode_start(self, experiment: str, task: str, episode_id: int) -> None:
        self._step_counter = 0
        logger.info("Episode start: experiment=%s task=%s episode_id=%d", experiment, task, episode_id)

    def on_episode_end(self, success: bool) -> None:
        logger.info("Episode end: success=%s steps=%d", success, self._step_counter)

    def on_task_begin(self) -> None:
        logger.info("Task begin (client connected)")

    def on_task_end(self) -> None:
        logger.info("Task end (client disconnected)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _configure_torchinductor_cache_dir() -> None:
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        return
    cache_dir = Path.cwd() / ".cache" / "torch" / "inductor"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)


def main(args: Args) -> None:
    _configure_torchinductor_cache_dir()

    logger.info("Loading policy...")
    policy = _create_policy(args)
    policy_metadata = policy.metadata

    logger.info(
        "Wrapping with ToyStage1Policy (qdrant_server_url=%s, stage1_device=%s)",
        args.qdrant_server_url,
        args.stage1_device,
    )
    toy_policy = ToyStage1Policy(policy, args.qdrant_server_url, args.stage1_device)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=toy_policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
