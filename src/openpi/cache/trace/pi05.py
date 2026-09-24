"""Pi0.5-specific pieces of the trace path (plan §5, §7.2).

The interceptor stays model-agnostic where it can; what is Pi0.5 shaped lives
here: reading the raw observation before the transform chain, slicing the
prefix token runs the legacy collector used to capture with forward hooks
(same source tensor, so the library file stays bit-identical), mapping the
full inference's snapshots to ``noise_action_i`` and assembling the
``StepTrace``. Loaded lazily by the Pi0.5 interceptor only; the GR00T island
never imports it.

Public interface: ``capture_raw_observation``, ``slice_prefix_tokens``,
``noise_actions_from_full``, ``build_step_trace`` (``action_error_proxies`` is
re-exported from ``openpi.cache.trace.records``).
Depends on numpy, torch, ``openpi.cache.components.key_builder._slice_cp1_fields``,
``openpi.cache.trace.records`` and ``openpi.cache.trace.types``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from openpi.cache.components.key_builder import _slice_cp1_fields
from openpi.cache.trace.records import action_error_proxies
from openpi.cache.trace.types import (
    ARM_FULL_INFERENCE,
    SearchTrace,
    StepTrace,
    TracePlan,
)
from openpi.cache.types import DenoiseSchedule
from openpi.collect.data_collector import InferenceEmbeddings


def capture_raw_observation(
    obs: dict, keys: tuple[str, ...]
) -> tuple[dict[str, np.ndarray], Optional[str], Optional[np.ndarray]]:
    """Copy the raw images / prompt / state out of the wire observation.

    Runs before ``_input_transform`` (which pops ``prompt`` and resizes the
    images). Every array is copied so the record never aliases a buffer the
    transform chain or the client codec may reuse. Missing keys are tolerated
    (a client may send fewer cameras).
    """
    images: dict[str, np.ndarray] = {}
    for key in keys:
        if key in obs:
            images[key] = np.array(obs[key], copy=True)
    prompt = obs.get("prompt")
    prompt_text = None if prompt is None else str(prompt)
    state = obs.get("observation/state")
    raw_state = None if state is None else np.array(state, dtype=np.float32, copy=True)
    return images, prompt_text, raw_state


def slice_prefix_tokens(stage1: Any) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """``(vision_embs fp16, prompt_emb fp16, robot_state f32)`` from a Stage1Output.

    Uses the key builders' own slice of ``prefix_embs`` so the recorded tokens
    are the exact tensors the online keys are built from (and, by the GPU
    parity gate, the same values the legacy forward hooks captured).
    """
    fields = _slice_cp1_fields(stage1.prefix_embs, stage1.state, None)
    vision = [
        fields[name].detach().to(torch.float16).cpu().numpy()
        for name in ("vision_0", "vision_1", "vision_2")
        if name in fields
    ]
    prompt = fields["prompt_emb"].detach().to(torch.float16).cpu().numpy()
    robot_state = fields["robot_state"].detach().to(torch.float32).cpu().numpy().reshape(-1)
    return vision, prompt, robot_state


def noise_actions_from_full(
    full: Any, init_noise: torch.Tensor, schedule: DenoiseSchedule
) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(noise_action_0, [noise_action_1 .. N-1])`` of a full inference.

    ``full.intermediates`` must hold every recoverable snapshot of ``schedule``
    (the build request asks for exactly that set); a missing one means the
    request and the loop disagree and the file must not become a library.
    """
    inter = getattr(full, "intermediates", None) or {}
    steps: list[np.ndarray] = []
    for index in range(1, schedule.num_steps):
        t = schedule.snapshot_t(index)
        if t not in inter:
            raise ValueError(
                f"full inference did not return the snapshot for step {index} "
                f"(t={t}); got {sorted(inter)}"
            )
        steps.append(_unbatch(inter[t]))
    return _unbatch(init_noise), steps


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
    input_images: Optional[dict[str, np.ndarray]],
    raw_images: dict[str, np.ndarray],
    prompt: Optional[str],
    raw_state: Optional[np.ndarray],
    model_images: Optional[dict[str, np.ndarray]],
    image_mask: Optional[dict[str, bool]],
    tokenized_prompt: Optional[np.ndarray],
    query_keys: Optional[dict[str, torch.Tensor]],
    search: Optional[SearchTrace],
    cp3_twin_json: Optional[str],
    full_action: np.ndarray,
    init_noise: torch.Tensor,
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
) -> StepTrace:
    """Assemble the per-decision record from already-computed pieces."""
    if plan.record_prefix_tokens:
        vision, prompt_emb, robot_state = slice_prefix_tokens(stage1)
    else:
        vision = []
        prompt_emb = np.zeros((0, 0), dtype=np.float16)
        robot_state = stage1.state[0].detach().to(torch.float32).cpu().numpy().reshape(-1)
    if plan.record_noise_actions:
        noise0, noise_steps = noise_actions_from_full(full_output, init_noise, plan.schedule)
    else:
        noise0, noise_steps = None, []
    legacy = InferenceEmbeddings(
        vision_embs=vision,
        prompt_emb=prompt_emb,
        robot_state=robot_state,
        noise_action_steps=noise_steps,
        clean_action=np.asarray(full_action, dtype=np.float32),
        input_images=input_images or None,
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
        model_images=model_images if plan.record_model_images else None,
        image_mask=image_mask if plan.record_model_images else None,
        tokenized_prompt=tokenized_prompt if plan.record_tokenized_prompt else None,
        query_keys=qk,
        search=search if plan.record_search else None,
        cp3_twin_json=cp3_twin_json,
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
    )


__all__ = [
    "ARM_FULL_INFERENCE",
    "action_error_proxies",
    "build_step_trace",
    "capture_raw_observation",
    "noise_actions_from_full",
    "slice_prefix_tokens",
]
