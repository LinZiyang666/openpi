"""Serve the official RoboCasa365 GR00T N1.5 teacher over a websocket.

This is the second-teacher counterpart to the pi0.5 server used earlier in the
cross-scene cache experiment.  It loads the released checkpoint as-is: there is
no training and no fine-tuning anywhere in this pipeline.

Where everything lives (all paths verified on the host named below)
------------------------------------------------------------------
host            weilandserver, single RTX 4090 (49140 MiB), shared with other
                sessions.  A ``serve_policy.py`` owned by someone else occupies
                port 8000 and ~8.8 GB and must never be shut down; only ever act
                on your own PID / tmux session, never a broad ``pkill``.
port            8020  (8000 = foreign server, 8010 = the pi0.5 teacher)
checkpoint      /home/weiland/ckpt_n15_robocasa/gr00t_n1-5/multitask_learning/checkpoint-120000
                7.2 GB, two safetensors shards, embodiment tag "new_embodiment".
this venv       /home/weiland/gr00t_n15_venv/.venv
                py3.11.15, numpy 1.26.4, transformers 4.51.3, torch 2.5.1+cu124.
gr00t source    NOT installed in that venv -- imported from the n1.5-release
                git worktree at /home/weiland/gr00t_n15, so PYTHONPATH must
                include it.
extra runtime   ``decord`` is required: the parent DataConfig's transform chain
                loads it through a transformers dynamic module, and without it
                construction fails with an ImportError.
extra deps      ``uv pip install -e /home/weiland/openpi/packages/openpi-client``
                (its numpy<2.0.0 pin is satisfied here, so no --no-deps needed --
                unlike the simulation island, where --no-deps is mandatory).
PYTHONPATH      /home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi
                the middle entry pulls in the websocket server only;
                that module imports nothing heavier than openpi_client and
                websockets, and openpi/__init__.py is empty, so no jax or torch
                gets dragged in from this repo.
sim client      runs in a *different* island (py3.12 / numpy 2.2.5), which is why
                this is a server rather than a single in-process script.

Launch::

    tmux new-session -d -s grootsrv "export HOME=/home/weiland; \\
      PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi \\
      /home/weiland/gr00t_n15_venv/.venv/bin/python \\
      /home/weiland/openpi/exp/robocasa365/serve_groot_n15.py 2>&1 | tee /tmp/grootsrv.log"
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Any

import numpy as np

from exp.robocasa365 import groot_keys
from exp.robocasa365.groot_policy_adapter import GrootPolicyAdapter

logging.basicConfig(level=logging.INFO)

DEFAULT_CHECKPOINT = (
    "/home/weiland/ckpt_n15_robocasa/gr00t_n1-5/multitask_learning/checkpoint-120000"
)
DEFAULT_PORT = 8020
EMBODIMENT_TAG = "new_embodiment"


# ------------------------------------------------------------------
# Deterministic-seed wrapper (diagnostics only)
# ------------------------------------------------------------------


class _SeededPolicy:
    """Reset the global torch RNG immediately before each inference.

    The N1.5 action head is flow-matching: every ``get_action`` call starts from
    a fresh ``torch.randn`` sample, so repeated calls on identical input differ.
    Pinning the seed here -- inside the server process, adjacent to the call --
    is what makes the wire-parity check reproducible.  Seeding from the client
    would accomplish nothing: client and server are separate interpreters and do
    not share a global RNG.

    Diagnostics only.  Leaving this on during a real rollout would drive every
    step from the same noise sample.
    """

    def __init__(self, policy: Any, seed: int) -> None:
        self._policy = policy
        self._seed = seed

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        import torch

        torch.manual_seed(self._seed)
        return self._policy.get_action(observations)

    def get_modality_config(self) -> dict[str, Any]:
        return self._policy.get_modality_config()


# ------------------------------------------------------------------
# Start-up handshake
# ------------------------------------------------------------------


def _dummy_observation(checkpoint: pathlib.Path) -> dict[str, Any]:
    """Build a *legal* observation for the start-up self-check.

    Two of the state fields are quaternions.  Zero-filling them -- the obvious
    thing to do for a smoke input -- does not describe a rotation, and the
    quaternion -> matrix -> rotation_6d path turns it into non-finite values, so
    the probe would either mask a real fault or reject a healthy server.  They
    are set to the identity quaternion instead, and the remaining state fields
    are taken from the checkpoint's own per-key means so every value sits inside
    the domain the model was normalized against.
    """
    stats = json.loads((checkpoint / "experiment_cfg" / "metadata.json").read_text())
    state_stats = stats[EMBODIMENT_TAG]["statistics"]["state"]

    obs: dict[str, Any] = {}
    for key in groot_keys.VIDEO_KEYS:
        resolution = groot_keys.MODEL_IMAGE_RESOLUTION
        obs[key] = np.zeros((resolution, resolution, 3), dtype=np.uint8)

    for key in groot_keys.STATE_KEYS:
        vector = np.asarray(
            state_stats[key.removeprefix("state.")]["mean"], dtype=np.float64
        )
        if key in groot_keys.QUATERNION_STATE_KEYS:
            # The mean of a set of quaternions is not itself unit-norm, and a
            # non-unit quaternion is not a rotation.  Renormalising keeps the
            # probe both in-distribution and geometrically valid; falling back
            # to a fixed identity would instead feed two of the five state
            # fields values the model never saw in training.
            norm = float(np.linalg.norm(vector))
            vector = (
                np.asarray(groot_keys.IDENTITY_QUATERNION_WXYZ, dtype=np.float64)
                if norm < 1e-8
                else vector / norm
            )
        obs[key] = vector

    for key in groot_keys.LANGUAGE_KEYS:
        obs[key] = "pick up the object"
    return obs


def _handshake(
    adapter: GrootPolicyAdapter, policy: Any, checkpoint: pathlib.Path
) -> None:
    """Fail loudly at start-up rather than serving a mis-wired policy."""
    modality = policy.get_modality_config()
    for name, expected in (
        ("video", groot_keys.VIDEO_KEYS),
        ("state", groot_keys.STATE_KEYS),
        ("action", groot_keys.ACTION_KEYS),
        ("language", groot_keys.LANGUAGE_KEYS),
    ):
        actual = list(modality[name].modality_keys)
        print(f"  {name:9s} {actual}", flush=True)
        if actual != expected:
            raise RuntimeError(
                f"{name} modality keys/order mismatch: {actual} != {expected}"
            )

    horizon = len(modality["action"].delta_indices)
    if horizon != groot_keys.ACTION_HORIZON:
        raise RuntimeError(f"action horizon {horizon} != {groot_keys.ACTION_HORIZON}")

    result = adapter.infer(_dummy_observation(checkpoint))
    for key, value in result["actions"].items():
        print(f"  {key:38s} shape={value.shape} dtype={value.dtype}", flush=True)
        if not np.isfinite(value).all():
            raise RuntimeError(
                f"handshake inference produced non-finite values for {key}"
            )

    print(f"  checkpoint: {checkpoint}", flush=True)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--diagnostic-seed",
        type=int,
        default=None,
        help="Reset the torch RNG to this seed before every inference. For the "
        "wire-parity check only; must stay unset for real rollouts.",
    )
    args = parser.parse_args()

    from gr00t.model.policy import Gr00tPolicy
    from openpi.serving import websocket_policy_server

    from exp.robocasa365.groot_data_config import RoboCasa365DataConfig

    checkpoint = pathlib.Path(args.checkpoint)
    data_config = RoboCasa365DataConfig()

    print(f"loading policy from {checkpoint}", flush=True)
    policy: Any = Gr00tPolicy(
        model_path=str(checkpoint),
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        device="cuda",
    )
    if args.diagnostic_seed is not None:
        print(
            f"DIAGNOSTIC MODE: seeding torch with {args.diagnostic_seed} per call",
            flush=True,
        )
        policy = _SeededPolicy(policy, args.diagnostic_seed)

    adapter = GrootPolicyAdapter(policy)
    print("running start-up handshake", flush=True)
    _handshake(adapter, policy, checkpoint)
    print("POLICY-READY", flush=True)

    # Advertise the diagnostic seed so a client can refuse to collect success
    # rates from a server whose sampling noise is pinned: that bias is invisible
    # in the resulting numbers and unrecoverable after the fact.
    metadata = {
        "checkpoint": str(checkpoint),
        "diagnostic_seed": args.diagnostic_seed,
        "action_horizon": groot_keys.ACTION_HORIZON,
    }
    server = websocket_policy_server.WebsocketPolicyServer(
        adapter, host="0.0.0.0", port=args.port, metadata=metadata
    )
    print(f"SERVER-LISTENING on 0.0.0.0:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
