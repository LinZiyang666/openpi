"""What the stage benchmark needs to know that is specific to LIBERO.

``exp/robocasa365/bench_groot_stages.py`` owns the measurement -- the three
compiled stages, the CUDA-graph evidence, the record schema -- and must stay a
single implementation, or the two teachers' ledgers are not comparable. Only
three things about it are per-line, and they live here:

*   the prompts, because prompt length is the graph shape under test;
*   how a production-shaped input is built, because LIBERO's adapter takes the
    *wire* observation (two 256x256 frames, an 8-D state, a string) and does
    the GR00T-space translation itself, while RoboCasa's takes an observation
    already in GR00T space;
*   how the policy is constructed, because the data config and the live step
    count differ (``--denoising-steps`` is a CLI property of this server, not
    a checkpoint constant).

A dummy observation is legal here without reading the checkpoint statistics:
LIBERO's state is three positions, three axis-angle components and two gripper
qpos, and zero is a valid value for every one of them -- an axis-angle zero is
the identity rotation. RoboCasa needs the statistics because two of its state
fields are quaternions, and a zero quaternion is not a rotation at all.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from exp.libero_groot import libero_keys as K

#: Task instructions from the two suites, shortest first. Prompt length is the
#: graph shape, so a cell is measured per prompt rather than averaged over
#: prompts -- a single-shape number cannot be extrapolated.
PROMPTS: tuple[str, ...] = (
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "put both the alphabet soup and the tomato sauce in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
)


def dummy_wire_observation(prompt: str) -> dict[str, Any]:
    """One legal wire observation, in the shape the client actually sends.

    Zeros, not noise: the benchmark times compute, which is data-independent,
    and a zero state is in-domain for every LIBERO field. What must be right
    is the *shape* -- ``WIRE_IMAGE_RESOLUTION`` is 256 because the GR00T
    evaluator crops to 224 itself, so feeding 224 here would crop twice and
    change the token count the graph is captured for.
    """
    frame = np.zeros(
        (K.WIRE_IMAGE_RESOLUTION, K.WIRE_IMAGE_RESOLUTION, 3), dtype=np.uint8
    )
    return {
        K.WIRE_IMAGE: frame,
        K.WIRE_WRIST: frame.copy(),
        K.WIRE_STATE: np.zeros(K.WIRE_STATE_DIM, dtype=np.float64),
        K.WIRE_PROMPT: prompt,
    }


def build_input(policy: Any, checkpoint: pathlib.Path, prompt: str) -> Any:
    """Reproduce the serving path's observation shaping, all four steps of it.

    Production is ``build_groot_observation`` (validates the wire contract and
    adds the T=1 axis), then the batch unsqueeze (B=1), then a numpy coercion,
    then ``apply_transforms``. Handing the raw wire observation straight to
    ``apply_transforms`` measures a shape the server never sees, and
    ``run_stage1`` asserts B=1 anyway.
    """
    del checkpoint  # LIBERO's dummy needs no checkpoint statistics
    from openpi.cache.groot.interceptor import _is_batched, _unsqueeze_values

    from exp.libero_groot.policy_adapter import build_groot_observation

    groot_obs = build_groot_observation(dummy_wire_observation(prompt))
    if not _is_batched(groot_obs):
        groot_obs = _unsqueeze_values(groot_obs)
    groot_obs = {
        k: (v if isinstance(v, np.ndarray) else np.array(v))
        for k, v in groot_obs.items()
    }
    return policy.apply_transforms(groot_obs)


def load_policy(checkpoint: pathlib.Path, *, device: str, denoising_steps: int) -> Any:
    """Build the served policy exactly as ``serve_groot_libero.main()`` does.

    ``denoising_steps`` is passed explicitly because for this head it is a
    runtime property -- the server's ``--denoising-steps`` defaults to 8 and
    the checkpoint's own config carries a lower value, so a benchmark that let
    the checkpoint decide would measure a loop the deployment never runs.
    """
    from gr00t.model.policy import Gr00tPolicy

    from custom_data_config import LiberoDataConfig  # examples/Libero on PYTHONPATH

    from exp.libero_groot.serve_groot_libero import EMBODIMENT_TAG

    data_config = LiberoDataConfig()
    return Gr00tPolicy(
        model_path=str(checkpoint),
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=denoising_steps,
        device=device,
    )
