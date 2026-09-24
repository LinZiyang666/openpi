"""GR00T-specific pieces of the trace path (plan §9-2, §9-3, §7.2).

The GR00T counterpart of ``openpi.cache.trace.pi05``: reading the raw
observation after the batch normalisation and before ``apply_transforms``,
slicing the prefix tokens with the key builders' own slice function (the
same tensors the legacy collector recorded), mapping the full inference's
captured loop inputs to ``noise_action_i`` and assembling the ``StepTrace``.

Raw observation keys are GR00T's own vocabulary (``video.*`` images,
``state.*`` vectors, ``annotation.*`` text), not the Pi0.5 wire names; the
camera set is whatever the served input carries, so no camera count is ever
assumed here.

Public interface: ``capture_raw_observation``, ``slice_prefix_tokens``,
``tokenized_prompt``, ``noise_actions_from_caps``, ``build_step_trace``.
Depends on numpy, torch, ``openpi.cache.groot.key_builder``,
``openpi.cache.trace.records`` and ``openpi.cache.trace.types``. Never
imports jax, the models or the Pi0.5 adapter.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from openpi.cache.groot.key_builder import slice_groot_cp1_fields
from openpi.cache.trace.records import action_error_proxies
from openpi.cache.trace.types import SearchTrace, StepTrace, TracePlan
from openpi.cache.types import DenoiseSchedule
from openpi.collect.data_collector import InferenceEmbeddings

VIDEO_PREFIX = "video."
STATE_PREFIX = "state."
ANNOTATION_PREFIX = "annotation."


def _current(key: str, value: np.ndarray, *, ndim: int) -> np.ndarray:
    """``value[0, 0]`` of a ``[B=1, T=1, ...]`` array, refusing any other layout."""
    arr = np.asarray(value)
    if arr.ndim != ndim or arr.shape[0] != 1 or arr.shape[1] != 1:
        raise ValueError(
            f"trace: {key!r} must be [B=1, T=1, ...] with {ndim} axes, got shape {arr.shape}"
        )
    return arr[0, 0]


def capture_raw_observation(
    obs: dict[str, Any],
) -> tuple[dict[str, np.ndarray], Optional[str], Optional[np.ndarray], tuple[tuple[str, int], ...]]:
    """``(raw_images, prompt, raw_state, raw_state_layout)`` of a batched GR00T observation.

    Called after the interceptor's one-time batch normalisation (every value
    is a ``[B, T, ...]`` ``np.ndarray``) and before ``apply_transforms``.
    Every array is copied. ``raw_state`` is the concatenation of the
    ``state.*`` vectors in observation order; ``raw_state_layout`` records
    which key owns which slice.
    """
    images: dict[str, np.ndarray] = {}
    state_parts: list[np.ndarray] = []
    layout: list[tuple[str, int]] = []
    prompts: list[str] = []
    for key, value in obs.items():
        if key.startswith(VIDEO_PREFIX):
            frame = _current(key, value, ndim=5)
            images[key] = np.array(frame, copy=True)
        elif key.startswith(STATE_PREFIX):
            vec = np.asarray(_current(key, value, ndim=3), dtype=np.float32).reshape(-1)
            state_parts.append(np.array(vec, copy=True))
            layout.append((key, int(vec.shape[0])))
        elif key.startswith(ANNOTATION_PREFIX):
            prompts.append(str(_current(key, value, ndim=2)))
    prompt = None
    if prompts:
        prompt = prompts[0] if len(prompts) == 1 else " | ".join(prompts)
    raw_state = np.concatenate(state_parts) if state_parts else None
    return images, prompt, raw_state, tuple(layout)


def slice_prefix_tokens(
    stage1: Any,
    *,
    vision_fields: tuple[str, ...],
    expected_state_index: Optional[torch.Tensor],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """``(vision_embs fp16, prompt_emb fp16, robot_state f32)`` from a GrootStage1Output.

    The key builders' own slice of ``input_embeds``, cut outside the runner
    session (the tensors are inference tensors; the copies below are not).
    ``expected_state_index`` is the episode's first ``state_mask`` row, so a
    mask that changes mid-episode is refused exactly as the builders refuse it.
    """
    fields = slice_groot_cp1_fields(
        stage1.input_embeds,
        stage1.image_token_mask,
        stage1.state,
        stage1.state_mask,
        enabled=None,
        expected_state_index=expected_state_index,
        vision_fields=vision_fields,
    )
    vision = [fields[name].detach().to(torch.float16).cpu().numpy() for name in vision_fields]
    prompt = fields["prompt_emb"].detach().to(torch.float16).cpu().numpy()
    robot_state = fields["robot_state"].detach().to(torch.float32).cpu().numpy().reshape(-1)
    return vision, prompt, robot_state


def tokenized_prompt(normalized_input: dict[str, Any]) -> Optional[np.ndarray]:
    """``eagle_input_ids`` of the normalised input as ``int64 [L]`` (``None`` if absent)."""
    ids = normalized_input.get("eagle_input_ids")
    if ids is None:
        return None
    arr = ids.detach().cpu().numpy() if torch.is_tensor(ids) else np.asarray(ids)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(f"trace: eagle_input_ids must be [1, L], got {arr.shape}")
        arr = arr[0]
    return np.asarray(arr, dtype=np.int64)


def noise_actions_from_caps(
    full: Any, schedule: DenoiseSchedule
) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(noise_action_0, [noise_action_1 .. N-1])`` from the full inference's captures.

    ``full.intermediates`` maps ``snapshot_t(i)`` to the chunk step ``i``
    consumed; step 0 is the noise as the loop actually saw it (cast to the
    head dtype). A build requests the whole schedule, so every step must be
    present.
    """
    inter = getattr(full, "intermediates", None) or {}
    chunks: list[np.ndarray] = []
    for index in range(schedule.num_steps):
        t = schedule.snapshot_t(index)
        if t not in inter:
            raise ValueError(
                f"full inference did not capture step {index} (t={t}); got {sorted(inter)}"
            )
        chunks.append(_unbatch(inter[t]))
    return chunks[0], chunks[1:]


def _unbatch(x: torch.Tensor) -> np.ndarray:
    t = x.detach()
    if t.dim() == 3:
        if t.shape[0] != 1:
            raise ValueError(f"expected a unit batch, got shape {tuple(t.shape)}")
        t = t[0]
    return t.to(torch.float32).cpu().numpy()


def build_step_trace(
    *,
    plan: TracePlan,
    stage1: Any,
    vision_fields: tuple[str, ...],
    expected_state_index: Optional[torch.Tensor],
    raw_images: dict[str, np.ndarray],
    prompt: Optional[str],
    raw_state: Optional[np.ndarray],
    raw_state_layout: tuple[tuple[str, int], ...],
    tokenized: Optional[np.ndarray],
    query_keys: Optional[dict[str, torch.Tensor]],
    search: Optional[SearchTrace],
    full_action: np.ndarray,
    full_output: Any,
    action_full_hit: Optional[np.ndarray],
    action_warm: dict[int, Optional[np.ndarray]],
    action_warm_exec: Optional[np.ndarray],
    action_executed: np.ndarray,
    executed_arm: str,
    verdict: dict[str, Any],
    tier_status: dict[str, str],
    timing_ms: dict[str, float],
    warm_index_map: dict[int, dict[str, Any]],
    real_continuation_json: Optional[str],
    twin_continuation_json: Optional[str],
) -> StepTrace:
    """Assemble the per-decision record from already-computed pieces."""
    if plan.record_prefix_tokens:
        vision, prompt_emb, robot_state = slice_prefix_tokens(
            stage1, vision_fields=vision_fields, expected_state_index=expected_state_index
        )
    else:
        vision = []
        prompt_emb = np.zeros((0, 0), dtype=np.float16)
        valid = stage1.state_mask[0, -1]
        robot_state = (
            stage1.state[0, -1][valid].detach().to(torch.float32).cpu().numpy().reshape(-1)
        )
    if plan.record_noise_actions:
        noise0, noise_steps = noise_actions_from_caps(full_output, plan.schedule)
    else:
        noise0, noise_steps = None, []
    legacy = InferenceEmbeddings(
        vision_embs=vision,
        prompt_emb=prompt_emb,
        robot_state=robot_state,
        noise_action_steps=noise_steps,
        clean_action=np.asarray(full_action, dtype=np.float32),
        input_images=None,
        init_noise=noise0,
    )
    qk = None
    if plan.record_query_keys and query_keys:
        qk = {
            name: t.detach().to(torch.float32).cpu().numpy().reshape(-1)
            for name, t in query_keys.items()
            if torch.is_tensor(t)
        }
    return StepTrace(
        legacy=legacy,
        raw_images=raw_images if plan.record_raw_images else {},
        prompt=prompt,
        raw_state=raw_state,
        model_images=None,
        image_mask=None,
        tokenized_prompt=tokenized if plan.record_tokenized_prompt else None,
        query_keys=qk,
        search=search if plan.record_search else None,
        cp3_twin_json=None,
        action_full_hit=action_full_hit,
        action_warm=dict(action_warm),
        action_warm_exec=action_warm_exec,
        action_executed=np.asarray(action_executed, dtype=np.float32),
        executed_arm=executed_arm,
        verdict=dict(verdict),
        tier_status=dict(tier_status),
        error_proxies=action_error_proxies(full_action, action_full_hit, action_warm, action_warm_exec),
        timing_ms=dict(timing_ms),
        warm_index_map=dict(warm_index_map),
        real_continuation_json=real_continuation_json,
        twin_continuation_json=twin_continuation_json,
        raw_state_layout=raw_state_layout or None,
    )


__all__ = [
    "build_step_trace",
    "capture_raw_observation",
    "noise_actions_from_caps",
    "slice_prefix_tokens",
    "tokenized_prompt",
]
