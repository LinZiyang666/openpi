"""Lossless native proprio; convert R17 rot180 to official vertical flip."""
import numpy as np

from exp.r17_integration.common.contracts import PreparedObs, StateView, WireChunk
from .protocols import PROPRIO_KEY


def native_observation(obs):
    images = [np.asarray(obs[key]) for key in ("observation/image", "observation/wrist_image")]
    if any(image.shape != (256, 256, 3) or image.dtype != np.uint8 for image in images):
        raise ValueError("Cosmos requires both original uint8 RGB256 camera views, already rotated180")
    if PROPRIO_KEY not in obs:
        raise ValueError("Cosmos requires lossless observation/cosmos_proprio [gripper2,xyz3,quat4]; no axis-angle reconstruction")
    state = np.asarray(obs[PROPRIO_KEY])
    if state.shape != (9,) or state.dtype.kind != "f" or not np.isfinite(state).all():
        raise ValueError("Cosmos native proprio must be a finite 9-D floating array")
    prompt = obs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Cosmos requires a nonempty task prompt")
    # Official libero_utils.py:44-57 uses flipud, not rot180. R17's stock
    # worker already rotates both axes, so undo just its horizontal flip.
    return {"primary_image": np.ascontiguousarray(images[0][:, ::-1]), "wrist_image": np.ascontiguousarray(images[1][:, ::-1]),
            "proprio": state.copy()}, prompt


class CosmosWirePolicy:
    def __init__(self, stats, cu):
        self.stats, self.cu = stats, cu

    def prepare(self, obs, request):
        native, prompt = native_observation(obs)
        return PreparedObs({"native": native, "prompt": prompt}, request,
                           native["proprio"].astype(np.float32))

    def extract_state_only(self, obs):
        return StateView(native_observation(obs)[0]["proprio"].astype(np.float32))

    def to_wire(self, normalized_chunk, prepared):
        array = normalized_chunk.detach().float().cpu().numpy() if hasattr(normalized_chunk, "detach") else np.asarray(normalized_chunk)
        if array.shape != (16, 7) or not np.isfinite(array).all():
            raise ValueError("Cosmos normalized action chunk must be finite (16,7)")
        # Preserve official NumPy promotion/controller dtype (normally float64).
        actions = self.cu.unnormalize_actions(array[None], self.stats)[0]
        return WireChunk(np.ascontiguousarray(actions))
