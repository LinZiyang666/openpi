"""Valid-dimension facts of the stored tensors (owner / coordinator 2026-09-26). READ BEFORE CODING.

Action chunks are (H, 32) in the model-normalized action space, but
  * only dims 0..6 are real LIBERO actions (dim 6 = gripper, bimodal at +-1);
      - pi0.5 dims 7..31 are ~0.001 constant padding,
      - GR00T dims 7..31 are noise-like padding with std 0.7-1.0, LARGER than the real dims
        (0.13-0.39) -> any full-32 distance on GR00T is dominated by garbage;
  * only the first 5 steps of a chunk are executed (replan 5); H = 10 (pi0.5) / 16 (GR00T).
robot_state ("rs")
  * pi0.5: 32-d, only dims 0..7 valid (dims 8..31 are exactly 0);
  * GR00T: 8-d, all valid.
  (B0 still uses the raw rs exactly as the online system did: the zero padding does not change L2.)

Never whiten / PCA / z-score over padded dims (zero-variance pi0.5 rs dims make covariances singular)
and never compute action similarity on the full 32 dims. Use the helpers below.
"""
from __future__ import annotations

ACT_VALID = slice(0, 7)          # valid action dims
ACT_DIMS = 7                     # number of valid action dims
EXEC_STEPS = 5                   # executed steps per decision (replan 5)
GRIPPER_DIM = 6                  # gripper dim inside the valid block, bimodal at +-1
HORIZON = {"pi05": 10, "groot": 16}
RS_DIM = {"pi05": 32, "groot": 8}          # stored robot_state width
RS_VALID = {"pi05": 8, "groot": 8}         # valid leading robot_state dims
RAW_STATE_DIM = 8
KEY_DIM = 32768                            # pooled vision key width (4x4 pool of 16x16 grid x 2048)
ACT_FULL_DIMS = 32


def valid_action(a):
    """Executed, valid block of an action chunk: a[..., :EXEC_STEPS, :ACT_DIMS] (a view, no copy)."""
    return a[..., :EXEC_STEPS, ACT_VALID]


def valid_action_chunk(a):
    """Whole horizon on valid dims: a[..., :, :ACT_DIMS] (for methods that look beyond the executed part)."""
    return a[..., ACT_VALID]


def valid_state(rs, model: str):
    """Valid leading dims of a robot_state vector / matrix: rs[..., :RS_VALID[model]]."""
    return rs[..., : RS_VALID[model]]


def gripper(a):
    """Gripper channel of the executed segment: a[..., :EXEC_STEPS, GRIPPER_DIM]."""
    return a[..., :EXEC_STEPS, GRIPPER_DIM]
