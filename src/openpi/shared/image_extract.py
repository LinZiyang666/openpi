"""Extract valid model-input images from input_transform output.

Filters by image_mask to skip padding slots (e.g. Libero's right_wrist_0_rgb).
Returns only the canonical image slots that have mask=True.

Note: returned images are post-transform (224x224), not environment-native resolution.
Cameras not mapped to model input slots (e.g. ALOHA cam_low) are not captured.
"""

from __future__ import annotations

import numpy as np

# Fixed canonical image slots (see model.py IMAGE_KEYS)
MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def extract_valid_images(
    inputs: dict,
    image_keys: tuple[str, ...] = MODEL_IMAGE_KEYS,
) -> dict[str, np.ndarray]:
    """Extract image_mask=True images from input_transform output.

    Args:
        inputs: Output of input_transform, containing "image" and "image_mask" dicts.
        image_keys: Canonical image slot names to check.

    Returns:
        {slot_name: (H, W, 3) uint8 ndarray} for slots with mask=True only.
    """
    images = inputs.get("image", {})
    masks = inputs.get("image_mask", {})
    result: dict[str, np.ndarray] = {}
    for key in image_keys:
        if key not in images or not bool(masks.get(key, False)):
            continue
        img = np.asarray(images[key])
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        result[key] = img
    return result
