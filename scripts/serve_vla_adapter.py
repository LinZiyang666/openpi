import dataclasses
import logging
import socket
import sys
import os
import numpy as np
import tyro
import torch

# OpenPI specific imports
from openpi_client import base_policy as _base_policy
from openpi.serving import websocket_policy_server
from openpi.collect.data_collector import InferenceEmbeddings, EpisodeDataCollector

# VLA-Adapter Integration & Monkey-Patches
# --------
# EXTERNAL REPOSITORY ROUTING: We appended the `../third_party/VLA-Adapter` path directly to the system path (`sys.path`). 
# WHY OF CHANGE: The VLA-Adapter codebase isn't a standard, pip-installed Python package—it's a manually cloned Git repository. This hack forces Python to treat that folder like a native library so we can cleanly import their internal functions without throwing module errors.
# -------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../third_party/VLA-Adapter')))

from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.openvla_utils import resize_image_for_policy
from experiments.robot.robot_utils import (
    get_action, 
    get_image_resize_size, 
    normalize_gripper_action, 
    invert_gripper_action
)

@dataclasses.dataclass
class Args:
    port: int = 8001
    collect: bool = False
    collect_dir: str = "./data"
    task_suite: str = "libero_spatial"
    checkpoint_dir: str = "VLA-Adapter/LIBERO-Spatial-Pro"  
    model_family: str = "openvla"


class VLAAdapterPolicy(_base_policy.BasePolicy):
    def __init__(self, cfg: GenerateConfig):
        self.cfg = cfg
        logging.info(f"Initializing VLA-Adapter model from {cfg.pretrained_checkpoint}...")
        
        self.model, self.action_head, self.proprio_projector, self.noisy_action_projector, self.processor = initialize_model(cfg)
        self._model = self.model
        self.resize_size = get_image_resize_size(cfg)

        # 1. Force the model configs to preserve all intermediate reasoning layers
        self.model.config.output_hidden_states = True
        if hasattr(self.model, "language_model"):
            self.model.language_model.config.output_hidden_states = True

        # VLA-Adapter Integration & Monkey-Patches
        # --------
        # THE "BRAIN SCANNER" INTERCEPTION: We forced the configuration to output hidden states and intercepted the base Language Model's `forward` pass to save the `_last_hidden_states_tuple` to memory. 
        # WHY OF CHANGE: To make accurate spatial decisions without hallucinating, the model's Action Head needs to evaluate all the intermediate reasoning layers simultaneously. Hugging Face's default code throws these layers in the trash, which effectively lobotomizes the model's spatial awareness.
        # -------
        original_lm_forward = self.model.language_model.forward
        def patched_lm_forward(*args, **kwargs):
            kwargs['output_hidden_states'] = True
            outputs = original_lm_forward(*args, **kwargs)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                self._last_hidden_states_tuple = outputs.hidden_states
            return outputs
        self.model.language_model.forward = patched_lm_forward

        # VLA-Adapter Integration & Monkey-Patches
        # --------
        # ACTION HEAD MONKEY-PATCH: We hijacked the `predict_action` method to dynamically stack those saved hidden layers, re-inject the `proprio` tensor, and generate a dummy `proprio_projector`. 
        # WHY OF CHANGE: This bypasses three major architectural bugs in the base code without altering the source files. It feeds the Action Head the 4D tensor it needs, stops a `NoneType` crash caused by the model dropping the robot's physical state during the forward pass, and prevents vision-only checkpoints from crashing when they look for a projector layer that doesn't exist.
        # -------
        original_predict = self.action_head.predict_action
        
        def patched_predict_action(hidden_states, proprio=None, proprio_projector=None, **kwargs):
            # Fix 1: Restore the True Multi-Layer Brain
            if hasattr(self, '_last_hidden_states_tuple'):
                hs_tuple = self._last_hidden_states_tuple
                if len(hs_tuple) > 1:
                    hidden_states = torch.stack(hs_tuple, dim=1)
                
            # Fix 2: Re-inject dropped proprioception state
            if proprio is None and hasattr(self, "_current_proprio"):
                proprio = self._current_proprio
            elif proprio is None:
                proprio = torch.zeros((1, 8), dtype=torch.bfloat16, device=hidden_states.device)

            # Fix 3: Mock the projector if missing
            proj = proprio_projector if proprio_projector is not None else self.proprio_projector
            if proj is None:
                dim = hidden_states.shape[-1]
                proj = lambda p: torch.zeros((p.shape[0], dim), dtype=hidden_states.dtype, device=hidden_states.device)
                
            return original_predict(hidden_states, proprio=proprio, proprio_projector=proj, **kwargs)
            
        self.action_head.predict_action = patched_predict_action
        
        logging.info("VLA-Adapter initialized successfully with True Hidden States Patch.")

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        img = np.ascontiguousarray(obs["observation/image"])
        wrist_img = np.ascontiguousarray(obs["observation/wrist_image"])
        
        # VLA-Adapter Integration & Monkey-Patches
        # --------
        # INFERENCE DATA MAPPING: We specifically extracted `observation/state` from the OpenPI simulator and mapped it to a new dictionary key named exactly `"state"`. 
        # WHY OF CHANGE: The VLA-Adapter's internal extraction code is incredibly rigid. If it doesn't see that exact string, it instantly throws a `KeyError` and crashes the server.
        # -------
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        prompt = obs.get("prompt", "")

        img_resized = resize_image_for_policy(img, self.resize_size)
        wrist_img_resized = resize_image_for_policy(wrist_img, self.resize_size)

        # Save tensor for the monkey-patch interception
        self._current_proprio = torch.tensor(state, dtype=torch.bfloat16, device=self.model.device).unsqueeze(0)

        vla_observation = {
            "full_image": img_resized,
            "wrist_image": wrist_img_resized,
            "state": state
        }

        # get_action handles internal normalization using the model's dataset_statistics.json
        actions = get_action(
            self.cfg,
            self.model,
            vla_observation,
            prompt,
            processor=self.processor,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=self.noisy_action_projector,
            use_film=self.cfg.use_film,
            use_minivlm=self.cfg.use_minivlm
        )

        actions = np.array(actions)
        
        # VLA-Adapter Integration & Monkey-Patches
        # --------
        # GRIPPER TIMING & INVERSION: We stripped out our manual numpy thresholding and piped the final actions through the official `normalize_gripper_action` and `invert_gripper_action` functions. 
        # WHY OF CHANGE: The model natively outputs continuous grasp probabilities on a [0, 1] scale where 0 is closed, but the MuJoCo physics engine strictly expects [-1, 1] where 1 is closed. Using their official math fixes the thresholding latency that caused the gripper to miss the bowl on the upswing, and flips the open/close convention so it actually grabs the object.
        # -------
        actions = normalize_gripper_action(actions, binarize=True)
        actions = invert_gripper_action(actions)
        
        return {"actions": actions}

    @property
    def metadata(self) -> dict:
        return {
            "name": "vla-adapter",
            "task_suite": self.cfg.task_suite_name,
            "default_prompt": ""
        }

# VLA-Adapter Integration & Monkey-Patches
# --------
# DATA COLLECTION BYPASS: We built the `VLACollectionPolicy` wrapper to intercept the episode data before it hits the disk, injecting empty lists and arrays of zeros for the embeddings. 
# WHY OF CHANGE: OpenPI's native HDF5 writer was built for diffusion models and expects denoising steps that our autoregressive model simply doesn't have. Handing it dummy zero arrays satisfies the writer's internal checks, preventing a crash while perfectly preserving the actual physical state and action data we need.
# -------
class VLACollectionPolicy:
    def __init__(self, policy: VLAAdapterPolicy, collector: EpisodeDataCollector):
        self._policy = policy
        self._collector = collector
        self._collecting = False

    def infer(self, obs: dict, **kwargs) -> dict:
        result = self._policy.infer(obs, **kwargs)
        if self._collecting:
            state = np.array(obs.get("observation/state", []), dtype=np.float32).flatten()
            actions = np.array(result["actions"], dtype=np.float32)

            embs = InferenceEmbeddings(
                vision_embs=[np.zeros((1, 2048), dtype=np.float16)],
                prompt_emb=np.zeros((1, 2048), dtype=np.float16),
                robot_state=state,
                noise_action_steps=[],
                clean_action=actions
            )
            self._collector.record_inference(embs)
        return result

    def on_episode_start(self, experiment: str, task: str, episode_id: int) -> None:
        self._collector.on_episode_start(experiment, task, episode_id)
        self._collecting = True

    def on_episode_end(self, success: bool) -> None:
        self._collector.on_episode_end(success)
        self._collecting = False
        
    def __getattr__(self, name):
        return getattr(self._policy, name)

# ============================================================================
# Server Execution Entry Point
# ============================================================================
def main(args: Args) -> None:
    cfg = GenerateConfig(
        task_suite_name=args.task_suite,
        pretrained_checkpoint=args.checkpoint_dir,
        model_family=args.model_family
    )
    policy = VLAAdapterPolicy(cfg)
    policy_metadata = policy.metadata

    if args.collect:
        collector = EpisodeDataCollector(base_dir=args.collect_dir)
        policy = VLACollectionPolicy(policy, collector)
        logging.info(f"Data collection enabled -> {args.collect_dir}")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Creating server (host: {hostname}, ip: {local_ip})")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))