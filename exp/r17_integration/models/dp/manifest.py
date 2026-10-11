"""Content identities, excluding mutable paths/mtime and process state."""
import hashlib
import json
from pathlib import Path


def digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_digest(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    return digest({"shape": list(a.shape), "dtype": str(a.dtype),
                   "bytes": hashlib.sha256(a.tobytes()).hexdigest()})


def code_identity():
    here = Path(__file__).parent
    reconstruction = here.parents[2] / "mw_finetune" / "dp"
    # Include every inference dependency of E, plus this complete adapter.
    sources = [reconstruction / name for name in
               ("model.py", "policy.py", "preprocess.py", "checkpoint.py", "RECIPE.md")]
    sources += sorted((reconstruction / "vendor").glob("*.py"))
    sources += sorted(here.glob("*.py"))
    sources += [here.parents[1] / "common" / name for name in
                ("contracts.py", "request_rng.py")]
    return {str(p.relative_to(here.parents[3])): file_digest(p) for p in sources}


def framework_identity():
    """Record framework provenance separately from immutable model math.

    Registry/lifecycle implementation changes should not invalidate a native
    model trajectory. Runtime releases still need their own G0 receipt/seal.
    """
    root = Path(__file__).parents[4]
    paths = [root / "exp/r17_integration/common" / name for name in
             ("provider.py", "serving.py", "config.py", "recipe.py", "websocket_policy_server.py")]
    paths += [root / "src/openpi" / name for name in
              ("cache/config.py", "cache/types.py", "cache/storage_types.py",
               "serving/batching_core.py", "serving/websocket_policy_server.py")]
    return {str(p.relative_to(root)): file_digest(p) for p in paths}


def layout(domain):
    if domain not in ("libero", "metaworld"):
        raise ValueError("domain must be libero or metaworld")
    d, s = (7, 8) if domain == "libero" else (4, 4)
    return {"horizon": 16, "replan_steps": 5, "valid_action_dim": d,
            "model_action_width": d, "valid_action_mask": [True] * d,
            "valid_state_indices": list(range(s)), "motion_channels": list(range(d - 1)),
            "gripper_channel": d - 1, "gripper_closed_convention": "benchmark_native",
            "raw_state_fields": "libero_xyz_axisangle_gripperqpos" if d == 7 else "obs[:4]",
            "action_space_name": "libero_osc_pose" if d == 7 else "metaworld_sawyer_xyz_gripper",
            "units": "benchmark_native; physical scales unqualified", "frame": "benchmark_native",
            "output_transform_id": "dp_internal_affine_then_external_quantile_v1"}


def feature_spec(domain, second_camera):
    return {"cutpoint": "obs_encoder.post_relu", "real_cameras":
            ["base_0_rgb", "left_wrist_0_rgb"] if domain == "libero" else ["base_0_rgb"],
            "absent_branch": None if domain == "libero" else second_camera,
            "hidden_width": 64, "token_grid": None, "pooling": "native_spatialsoftmax_linear_relu",
            "prompt_dependent": True, "key_dtype": "float32", "preprocess": "PIL_bilinear96_center84",
            "wire_image_sizes": [96, 256] if domain == "libero" else None,
            "wire_image_format": "uint8_rgb_hwc",
            "key_fields": ["vision_0", "vision_1", "robot_state"] if domain == "libero"
            else ["vision_0", "robot_state"]}
