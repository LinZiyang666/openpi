"""Rebuild a GR00T stage-1 sequence from a W13 HDF5 step, for the CP2 tooling.

Why a reconstruction at all
---------------------------
The CP2 key is cut from the action head's *encoded* conditioning, which needs
the language model (stage 2) run on the full stage-1 sequence. The W13
collection stores only the sliced fields of that sequence -- the two
image-token runs (``vision_0/1``), the non-image positions (``prompt_emb``) and
the valid state (``robot_state``) -- and no raw images, so stage 2 cannot be
re-run from a stored observation as the Pi0.5 line does. What the collection
does not store is deterministic: the chat-template token ids depend only on the
instruction, so a *template* stage-1 sequence built from a zero-image
observation with the same instruction carries the right text embeddings, mask,
positions and ``action_inputs`` skeleton. Each step is then the template with
its image runs and its valid state overwritten from the file.

Fail-closed by construction (plan §3.6, R1-B3): every step asserts that the
template's text positions equal the stored ``prompt_emb`` (a tokenizer / chat
template drift would show up here first) and that the state written back reads
back exactly as the stored ``robot_state``; a group missing any of the three
fields raises. Nothing is re-normalised: the file already holds the values the
model saw.

Contract: ``reconstruct_stage1`` must be called inside ``runner.session()``
(the same inference / autocast context the online path uses); it returns a
``GrootStage1Output`` whose tensors live in that context, ready for
``run_stage2_llm`` / ``run_cp2_key_source``.

Environment: the GR00T island (torch, no jax). Guarded by
``tests/cache/groot/test_import_isolation.py``.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import numpy as np
import torch

from openpi.cache.groot.interceptor import _is_batched, _unsqueeze_values
from openpi.cache.groot.key_builder import _contiguous_runs
from openpi.cache.groot.staged import GrootStage1Output, GrootStagedRunner
from openpi.cache.types import PROMPT_EMB, ROBOT_STATE, VISION_0, VISION_1

from exp.libero_groot import libero_keys as K
from exp.libero_groot.policy_adapter import build_groot_observation

STAGE1_PATH = "groot_reconstructed_template"
LIBERO_VISION_FIELDS = (VISION_0, VISION_1)
TOKENS_PER_IMAGE = 256
REQUIRED_STEP_FIELDS = (*LIBERO_VISION_FIELDS, PROMPT_EMB, ROBOT_STATE)


@dataclasses.dataclass
class Template:
    """One instruction's stage-1 skeleton (built once, reused per step)."""

    task: str
    input_embeds: torch.Tensor          # [1, N, C]
    attention_mask: torch.Tensor        # [1, N]
    image_token_mask: torch.Tensor      # [1, N] bool
    action_inputs: Any                  # BatchFeature: state / state_mask / embodiment_id
    runs: tuple[tuple[int, int], ...]   # (start, length) per camera, run order

    @property
    def n_tokens(self) -> int:
        """Sequence length ``N`` of this instruction's stage-1 sequence."""
        return int(self.input_embeds.shape[1])


def dummy_wire_observation(task: str) -> dict[str, Any]:
    """A legal zero-image, zero-state wire observation carrying ``task``."""
    frame = np.zeros((K.WIRE_IMAGE_RESOLUTION, K.WIRE_IMAGE_RESOLUTION, 3), dtype=np.uint8)
    return {
        K.WIRE_IMAGE: frame,
        K.WIRE_WRIST: frame.copy(),
        K.WIRE_STATE: np.zeros(K.WIRE_STATE_DIM, dtype=np.float64),
        K.WIRE_PROMPT: str(task),
    }


def normalized_input_for(policy: Any, obs: dict[str, Any]) -> Any:
    """The serving path's observation shaping (adapter -> unsqueeze -> numpy -> transforms)."""
    groot_obs = build_groot_observation(obs)
    if not _is_batched(groot_obs):
        groot_obs = _unsqueeze_values(groot_obs)
    groot_obs = {k: (v if isinstance(v, np.ndarray) else np.array(v)) for k, v in groot_obs.items()}
    return policy.apply_transforms(groot_obs)


def build_template(policy: Any, runner: GrootStagedRunner, task: str) -> Template:
    """Run stage 1 once on a zero-image observation with ``task``; keep the skeleton.

    Must be called inside ``runner.session()``.
    """
    stage1 = runner.run_stage1(normalized_input_for(policy, dummy_wire_observation(task)))
    runs = tuple(_contiguous_runs(stage1.image_token_mask[0]))
    if len(runs) != len(LIBERO_VISION_FIELDS) or any(n != TOKENS_PER_IMAGE for _, n in runs):
        raise RuntimeError(
            f"template for {task!r}: expected {len(LIBERO_VISION_FIELDS)} image runs of "
            f"{TOKENS_PER_IMAGE} tokens, got {runs}"
        )
    return Template(
        task=str(task),
        input_embeds=stage1.input_embeds,
        attention_mask=stage1.attention_mask,
        image_token_mask=stage1.image_token_mask,
        action_inputs=stage1.action_inputs,
        runs=runs,
    )


class TemplateCache:
    """One template per instruction string, built lazily."""

    def __init__(self, policy: Any, runner: GrootStagedRunner) -> None:
        self._policy = policy
        self._runner = runner
        self._by_task: dict[str, Template] = {}

    def get(self, task: str) -> Template:
        """The template for instruction ``task``, built on first use (inside ``runner.session()``)."""
        task = str(task)
        if task not in self._by_task:
            self._by_task[task] = build_template(self._policy, self._runner, task)
        return self._by_task[task]

    def __len__(self) -> int:
        return len(self._by_task)


def _read(group: Any, name: str) -> np.ndarray:
    if name not in group:
        raise RuntimeError(f"H5 step {getattr(group, 'name', '?')} lacks {name!r}")
    return np.asarray(group[name])


def _copy_action_inputs(action_inputs: Any) -> Any:
    """A per-step copy of the BatchFeature so the template is never mutated."""
    data = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in action_inputs.items()}
    try:
        from transformers.feature_extraction_utils import BatchFeature
    except ImportError:  # pragma: no cover
        return data
    return BatchFeature(data=data)


def reconstruct_stage1(template: Template, group: Any) -> GrootStage1Output:
    """``Template`` + one H5 step group -> the step's ``GrootStage1Output``.

    Inside ``runner.session()``. Overwrites the image runs with the stored
    vision embeddings and the valid state slots with the stored normalised
    state; asserts the template's text positions equal the stored ``prompt_emb``
    and that the state reads back exactly.
    """
    for name in REQUIRED_STEP_FIELDS:
        if name not in group:
            raise RuntimeError(f"H5 step {getattr(group, 'name', '?')} lacks {name!r}")
    embeds = template.input_embeds.clone()
    device, dtype = embeds.device, embeds.dtype
    for field, (start, length) in zip(LIBERO_VISION_FIELDS, template.runs, strict=True):
        vis = torch.from_numpy(_read(group, field)).to(device=device, dtype=dtype)
        if tuple(vis.shape) != (length, embeds.shape[2]):
            raise RuntimeError(f"{field}: stored shape {tuple(vis.shape)} != run ({length}, {embeds.shape[2]})")
        embeds[0, start : start + length] = vis
    # Text positions are everything the image runs did not overwrite -- exactly
    # what the collector stored (fp16) as prompt_emb.
    text_mask = ~template.image_token_mask[0]
    stored_text = _read(group, PROMPT_EMB)
    template_text = embeds[0][text_mask].float().to(torch.float16).cpu().numpy()
    if stored_text.shape != template_text.shape or not np.array_equal(
        stored_text.astype(np.float16), template_text
    ):
        raise RuntimeError(
            f"template text positions differ from stored prompt_emb for task {template.task!r} "
            f"(stored {stored_text.shape}, template {template_text.shape}); the chat template, "
            "tokenizer or instruction string is not the one the collection used."
        )
    action_inputs = _copy_action_inputs(template.action_inputs)
    state = action_inputs["state"]
    valid = action_inputs["state_mask"][0, -1]
    stored_state = _read(group, ROBOT_STATE).astype(np.float32).reshape(-1)
    if int(valid.sum()) != stored_state.shape[0]:
        raise RuntimeError(
            f"stored robot_state has {stored_state.shape[0]} values but the state_mask marks "
            f"{int(valid.sum())} valid slots"
        )
    state[0, -1][valid] = torch.from_numpy(stored_state).to(device=state.device, dtype=state.dtype)
    read_back = state[0, -1][valid].float().cpu().numpy()
    if not np.array_equal(read_back, stored_state):
        raise RuntimeError(
            "state written back does not read back exactly: the stored robot_state is not "
            f"representable in the model's state dtype {state.dtype} (max |delta| "
            f"{float(np.max(np.abs(read_back - stored_state)))})"
        )
    return GrootStage1Output(
        input_embeds=embeds,
        attention_mask=template.attention_mask,
        image_token_mask=template.image_token_mask,
        action_inputs=action_inputs,
    )


def h5_task(h5_file: Any) -> str:
    """The instruction (``task.language``) a collector file was conditioned on; raises when absent."""
    task = h5_file.attrs.get("task", None)
    if task is None:
        raise RuntimeError(f"{getattr(h5_file, 'filename', '?')}: no 'task' attribute")
    return str(task)


def load_groot_libero_policy(checkpoint: str | pathlib.Path, *, denoising_steps: int = 8,
                             device: str = "cuda") -> Any:
    """The served policy, built exactly as ``serve_groot_libero.main()`` builds it."""
    from gr00t.model.policy import Gr00tPolicy

    from custom_data_config import LiberoDataConfig  # examples/Libero on PYTHONPATH

    from exp.libero_groot.serve_groot_libero import EMBODIMENT_TAG

    data_config = LiberoDataConfig()
    policy = Gr00tPolicy(
        model_path=str(checkpoint),
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=denoising_steps,
        device=device,
    )
    if int(policy.denoising_steps) != int(denoising_steps):
        raise RuntimeError(f"policy runs {policy.denoising_steps} steps, requested {denoising_steps}")
    return policy


__all__ = [
    "LIBERO_VISION_FIELDS",
    "REQUIRED_STEP_FIELDS",
    "STAGE1_PATH",
    "Template",
    "TemplateCache",
    "build_template",
    "dummy_wire_observation",
    "h5_task",
    "load_groot_libero_policy",
    "normalized_input_for",
    "reconstruct_stage1",
]
