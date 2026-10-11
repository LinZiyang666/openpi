"""Pure action inference for mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B.

The release has no training config. Camera layout and chunk length are explicit
reconstruction assumptions; see HANDBACK_D.md before interpreting success rates.
Torch/Cosmos imports are lazy so observation conversions can be tested on CPU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np

REPO_ID = "mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B"
REVISION = "6546ab82a1d302e4a947bc7b03880ec6031a6109"
WEIGHT_SHA256 = "c8562afc4ee58c20e4c43b8128e72f1e75d9dc1df5a429cde2d112bf6c6436b6"


@dataclass(frozen=True)
class Options:
    denoising_steps: int = 5
    chunk_size: int = 16
    camera_layout: str = "single"
    image_rotation: int = 0  # Relative to the already rotated harness image.
    trained_with_image_aug: bool = True
    use_jpeg_compression: bool = True
    gripper_qpos_mode: str = "opposed"
    cuda_graph: bool = False

    def __post_init__(self):
        if self.denoising_steps < 1 or self.chunk_size < 5:
            raise ValueError("denoising_steps must be >=1 and chunk_size >=5")
        if self.camera_layout not in ("single", "libero_duplicate", "robocasa_duplicate"):
            raise ValueError("camera_layout must be single, libero_duplicate or robocasa_duplicate")
        if self.image_rotation not in (0, 180):
            raise ValueError("image_rotation must be 0 or 180")
        if self.gripper_qpos_mode != "opposed":
            raise ValueError("this checkpoint requires gripper_qpos_mode=opposed")

    @property
    def n_cameras(self):
        return {"single": 1, "libero_duplicate": 2, "robocasa_duplicate": 3}[self.camera_layout]


def expand_proprio(state):
    """Stats imply [left_grip,right_grip,EEF xyz,identity quaternion xyzw]."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("state must be finite float32 [x,y,z,g], shape (4,)")
    x, y, z, g = state
    return np.asarray([g, -g, x, y, z, 0, 0, 0, 1], dtype=np.float32)


def project_actions(actions):
    """Drop padded rotations; retain native commands without clipping or scaling."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"expected finite (T,7) Cosmos actions, got {actions.shape}")
    return np.ascontiguousarray(actions[:, [0, 1, 2, 6]])


def validate_stats(stats):
    for prefix, dim in (("actions", 7), ("proprio", 9)):
        lo, hi = (np.asarray(stats[f"{prefix}_{k}"], dtype=np.float32) for k in ("min", "max"))
        if lo.shape != (dim,) or hi.shape != (dim,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
            raise ValueError(f"invalid {prefix} min/max statistics; expected {dim} dimensions")


def load_dit_weights(net, state):
    """Reject absent DiT parameters and extra learned weights, before inference."""
    if not isinstance(state, dict) or not any(k.startswith("net.") for k in state):
        raise ValueError("expected a consolidated Cosmos state_dict with net.* keys")
    ignored_extra_states = sum(k.endswith("_extra_state") for k in state)
    net_state = {k.removeprefix("net."): v for k, v in state.items()
                 if k.startswith("net.") and not k.endswith("_extra_state")}
    # Cosmos's outer load_state_dict returns None in non-strict mode. The net's
    # native PyTorch load returns an auditable report and rejects shape errors.
    result = net.load_state_dict(net_state, strict=False)
    parameter_keys = {k for k, _ in net.named_parameters()}
    missing_parameters = sorted(parameter_keys.intersection(result.missing_keys))
    unexpected_net = [k for k in result.unexpected_keys if not k.endswith("_extra_state")]
    if missing_parameters or unexpected_net:
        raise ValueError(f"incompatible DiT weights: missing={missing_parameters[:10]}, unexpected={unexpected_net[:10]}")
    return {"missing_parameters": missing_parameters, "unexpected_net": unexpected_net,
            "ignored_fp8_extra_states": ignored_extra_states,
            "other_missing_keys": result.missing_keys, "other_unexpected_keys": result.unexpected_keys}


def select_prompt(obs, prompts):
    prompt = obs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")
    # RLinf uses the push-v3 wording for push-back. This cache has its own
    # task-specific wording. Prefer the cache label for that known task.
    if obs.get("task_name") == "push-back-v3":
        prompt = "Push a puck back to a target behind it"
    if prompt not in prompts:
        raise ValueError(f"prompt is absent from the released T5 cache: {prompt!r}")
    return prompt


def unpack_observation(obs, options):
    image = np.asarray(obs["image"] if "image" in obs else obs["observation/image"])
    if image.ndim != 3 or image.shape[0] != image.shape[1] or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be square uint8 RGB HWC")
    if options.image_rotation == 180:
        image = image[::-1, ::-1]
    state = obs["state"] if "state" in obs else obs["observation/state"]
    return np.ascontiguousarray(image), expand_proprio(state)


def latent_layout(n_cameras):
    """Indices in the Cosmos latent video, following upstream cosmos_utils."""
    if n_cameras not in (1, 2, 3):
        raise ValueError("expected 1..3 cameras")
    names = (["blank", "current_proprio"] + [f"current_camera_{i}" for i in range(n_cameras)]
             + ["action", "future_proprio"] + [f"future_camera_{i}" for i in range(n_cameras)] + ["value"])
    return {name: i for i, name in enumerate(names)}


def make_data_batch(image, proprio, embedding, options, stats, device, cu, torch):
    """Generalize upstream's video assembly to a single MetaWorld camera.

    Multi-camera options duplicate the supplied view into upstream slots. They
    are hypotheses for probes, and never claim that additional cameras exist.
    """
    images = cu.prepare_images_for_model([image] * options.n_cameras, options)
    idx = latent_layout(options.n_cameras)
    blank = np.zeros_like(images[0])
    frames = [blank[None]]
    for name in list(idx)[1:]:
        frame = images[int(name.rsplit("_", 1)[1])] if "camera_" in name else blank
        frames.append(np.repeat(frame[None], 4, axis=0))
    video = np.concatenate(frames, axis=0).transpose(3, 0, 1, 2)[None]
    normalized = cu.rescale_proprio(proprio, stats, non_negative_only=False)
    batch = {
        "dataset_name": "video_data",
        "video": torch.as_tensor(np.ascontiguousarray(video), dtype=torch.uint8, device=device),
        "t5_text_embeddings": embedding.to(device=device, dtype=torch.bfloat16),
        "fps": torch.tensor([16], dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros((1, 1, 224, 224), dtype=torch.bfloat16, device=device),
        "num_conditional_frames": options.n_cameras + 2,
        "proprio": torch.as_tensor(normalized[None], dtype=torch.bfloat16, device=device),
    }
    def index(key, value):
        batch[key] = torch.tensor([value], dtype=torch.int64, device=device)
    index("current_proprio_latent_idx", idx["current_proprio"])
    index("future_proprio_latent_idx", idx["future_proprio"])
    index("action_latent_idx", idx["action"])
    index("value_latent_idx", idx["value"])
    # The first image is wrist only in the duplicated two/three-camera probes.
    camera_slots = {"wrist_image": 0 if options.n_cameras > 1 else None,
                    "wrist_image2": None, "image": 0 if options.n_cameras == 1 else 1,
                    "image2": 2 if options.n_cameras == 3 else None}
    for prefix in ("current", "future"):
        for key, slot in camera_slots.items():
            index(f"{prefix}_{key}_latent_idx", -1 if slot is None else idx[f"{prefix}_camera_{slot}"])
    return batch, idx["action"]


class CosmosMetaWorldPolicy:
    def __init__(self, model, options, stats, embeddings, device, cu, torch, metadata):
        self.model, self.options, self.stats = model, options, stats
        self.embeddings, self.device, self.cu, self.torch = embeddings, device, cu, torch
        self.metadata = metadata

    def reset(self):
        # Every request constructs fresh conditioning and sampling noise.
        pass

    def get_metadata(self):
        return dict(self.metadata)

    def infer(self, obs):
        image, proprio = unpack_observation(obs, self.options)
        prompt = select_prompt(obs, self.embeddings)
        seed = int(obs.get("seed", 7))
        with self.torch.cuda.device(self.device), self.torch.inference_mode():
            self.torch.manual_seed(seed)
            self.torch.cuda.manual_seed(seed)
            batch, action_idx = make_data_batch(image, proprio, self.embeddings[prompt], self.options, self.stats,
                                               self.device, self.cu, self.torch)
            generated = self.model.generate_samples_from_batch(
                batch, n_sample=1, num_steps=self.options.denoising_steps, seed=seed,
                is_negative_prompt=False, use_variance_scale=False,
                return_orig_clean_latent_frames=False,
            )
            indices = self.torch.tensor([action_idx], dtype=self.torch.int64, device=self.device)
            normalized = self.cu.extract_action_chunk_from_latent_sequence(
                generated, (self.options.chunk_size, 7), indices).float().cpu().numpy()
            actions = self.cu.unnormalize_actions(normalized, self.stats)[0]
        return {"actions": project_actions(actions)}

    def enable_cuda_graph(self):
        from exp.cosmos_nfe.cudagraph_net import GraphedNet
        if not isinstance(self.model.net, GraphedNet):
            self.model.net = GraphedNet(self.model.net, warmup=2)


def load_policy(checkpoint, device="cuda:0", **kw):
    """Load one frozen bf16 model; requires the existing Cosmos Policy venv.

    Extra options are the Options fields plus tokenizer_path (an offline VAE
    path). Chunk size 16 and single-camera layout are reconstruction defaults.
    """
    tokenizer_path = kw.pop("tokenizer_path", None)
    options = Options(**kw)
    path = Path(checkpoint).expanduser().resolve()
    root, weight = (path.parent, path) if path.is_file() else (path, path / "consolidated.pt")
    for file in (weight, root / "dataset_statistics.json", root / "t5_embeddings.pkl"):
        if not file.is_file():
            raise FileNotFoundError(file)
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in json.loads((root / "dataset_statistics.json").read_text()).items()}
    validate_stats(stats)
    os.environ["DETERMINISTIC"] = "True"
    import torch
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Cosmos Policy inference requires a CUDA device; CPU conversions are independently testable")
    torch.cuda.set_device(device)
    n = options.n_cameras
    overrides = [f"model.config.state_t={2*n+5}", f"model.config.min_num_conditional_frames={n+2}",
                 f"model.config.max_num_conditional_frames={n+2}",
                 f"model.config.tokenizer.chunk_duration={4*(2*n+4)+1}",
                 f"dataloader_train.dataset.chunk_size={options.chunk_size}"]
    if tokenizer_path:
        overrides.append(f"model.config.tokenizer.vae_pth={Path(tokenizer_path).resolve()}")
    model, config = load_model_from_checkpoint(
        experiment_name="cosmos_predict2_2b_480p_libero__inference_only", s3_checkpoint_dir=str(weight),
        config_file="cosmos_policy/config/config.py", instantiate_ema=False, skip_load_model=True,
        experiment_opts=overrides, to_device=str(device),
    )
    # Upstream's loader uses strict=False without inspecting missing weights.
    # Audit all DiT parameters here so a wrong architecture cannot run silently.
    state = torch.load(weight, map_location="cpu", weights_only=True, mmap=True)
    weight_audit = load_dit_weights(model.net, state)
    del state
    model.eval().to(device)
    with (root / "t5_embeddings.pkl").open("rb") as f:
        embeddings = pickle.load(f)
    if len(embeddings) != 50 or any(tuple(v.shape) != (1, 512, 1024) for v in embeddings.values()):
        raise ValueError("expected the release's 50 cached (1,512,1024) T5 embeddings")
    # Keep the prompt cache private; no upstream global T5 fallback or writes.
    embeddings = {k: v.detach().cpu() for k, v in embeddings.items()}
    metadata = {
        "model": "cosmos", "repo_id": REPO_ID, "revision": REVISION,
        "checkpoint": str(weight), "checkpoint_size": weight.stat().st_size,
        "options": asdict(options), "pure_inference": True,
        "policy_rng": "obs.seed; fresh Cosmos sampler noise per query",
        "proprio_mapping": "[g,-g,x,y,z,0,0,0,1]", "action_mapping": "denormalized[:,[0,1,2,6]]",
        "stats_sha256": hashlib.sha256((root / "dataset_statistics.json").read_bytes()).hexdigest(),
        "state_t": int(config.model.config.state_t), "image_size": 224,
        "layout_verified_against_author_runner": False,
        "config_assumptions": ["camera count/layout", "chunk size", "JPEG and center crop", "sampling defaults"],
        "weight_audit": weight_audit,
        "future_decode": False, "planning": False,
    }
    policy = CosmosMetaWorldPolicy(model, options, stats, embeddings, device, cu, torch, metadata)
    if options.cuda_graph:
        policy.enable_cuda_graph()
    return policy
