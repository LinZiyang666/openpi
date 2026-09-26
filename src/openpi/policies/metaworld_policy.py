"""MetaWorld MT50 input/output transforms for the RLinf pi0.5 SFT checkpoint.

Mirrors ``libero_policy`` for the ``pi05_metaworld`` inference config
(checkpoint ``RLinf/RLinf-Pi05-MetaWorld-SFT``). The checkpoint was trained on
``lerobot/metaworld_mt50`` with a single third-person camera: only
``base_0_rgb`` carries a real image, both wrist slots are zero images masked
out, and the 4-dim action (xyz delta + gripper) is sliced off the model's
32-dim padded output. The 4-dim state (``obs[:4]``) is forwarded unchanged;
with ``discrete_state_input=False`` the model does not read it, but the
standard pipeline still normalizes and pads it.

Public interface: ``make_metaworld_example``, ``MetaworldInputs``,
``MetaworldOutputs``. Depends only on numpy / einops / ``openpi.transforms``
(no simulator import).
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms

#: Executed action dimensions (xyz delta + gripper).
METAWORLD_ACTION_DIM = 4


def make_metaworld_example() -> dict:
    """Creates a random input example for the MetaWorld policy."""
    return {
        "observation/state": np.random.rand(4).astype(np.float32),
        "observation/image": np.random.randint(256, size=(480, 480, 3), dtype=np.uint8),
        "prompt": "Reach a goal position",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MetaworldInputs(transforms.DataTransformFn):
    """Map a MetaWorld observation onto pi0.5's three image slots.

    The checkpoint saw one camera only, so the two wrist slots are zero images
    with ``image_mask`` False; ``base_0_rgb`` is the (already flipped) corner2
    render. Resizing to 224 happens later in the model transforms.
    """

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        # Actions only exist in training data.
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class MetaworldOutputs(transforms.DataTransformFn):
    """Return the executed 4 action dimensions; the rest is model padding."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :METAWORLD_ACTION_DIM])}
