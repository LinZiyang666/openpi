"""GR00T stage-3 adapter for the model-agnostic batching core (plan §6.3).

All stages concatenate same-shape requests without introducing padding.
Callers must retain exclusion around shared compiled vision buffers.

Same-shape batching, no padding
-------------------------------
The upstream action head conditions the DiT on ``backbone_features`` through
``vl_self_attention`` **without** an attention mask, so zero-padding a
shorter prompt to the longest one in a batch would change every action in
it. Requests are therefore bucketed by the *full* conditioning shape --
sequence length and width, state shape, dtypes, device, embodiment -- and a
bucket is concatenated along the batch axis only. Two connections on the
same task have identical shapes and batch; two tasks with different prompt
lengths never share a forward. The B=1 path and the B>1 path run the same
``denoise_loop``; their numerical agreement is a tested gate, never assumed
from "cat is a view".

Producer-side preparation
-------------------------
``stage3_input`` is called by the producer under its model lock: it copies
the embodiment id to the host (the only ``.item()``-class read) and freezes
the execution domain, so the worker's ``bucket_key`` reads host metadata and
tensor ``.shape`` / ``.dtype`` / ``.device`` only -- never a value that would
synchronise the stream before ``ready_events`` were waited on.

Public interface: ``GrootStage3Input``, ``stage3_input``, ``run_miss``,
``run_warm``, ``split_miss``, ``split_warm``, ``GrootStageBatcher``.
Depends on torch, ``openpi.serving.batching_core`` (jax-free) and
``openpi.cache.groot.staged``. Never imports jax, the models or ``stage_io``.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping
import logging
from typing import Any, Hashable, Optional

import torch

from openpi.cache.groot.staged import (
    GrootStage2Output,
    GrootStage1Output,
    GrootStage3Output,
    GrootStagedRunner,
    _batch_feature,
)
from openpi.cache.types import DenoiseSchedule
from openpi.serving.batching_core import (
    Stage3MissPayload,
    Stage3VariantOutput,
    Stage3WarmStartPayload,
)

logger = logging.getLogger(__name__)


def feature_shape(value):
    if torch.is_tensor(value):
        return (tuple(value.shape[1:]), str(value.dtype), str(value.device))
    if isinstance(value, Mapping):
        return tuple((k, feature_shape(v)) for k, v in value.items())
    return value


def cat_features(values):
    """Native prepared inputs already carry a unit batch axis."""
    first = values[0]
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if isinstance(first, Mapping):
        return {k: cat_features([v[k] for v in values]) for k in first}
    return first


def row_features(value, i):
    if torch.is_tensor(value):
        return value[i:i+1]
    if isinstance(value, Mapping):
        return {k: row_features(v, i) for k, v in value.items()}
    return value


def cat_stage1(values):
    return GrootStage1Output(
        input_embeds=torch.cat([v.input_embeds for v in values]),
        attention_mask=torch.cat([v.attention_mask for v in values]),
        image_token_mask=torch.cat([v.image_token_mask for v in values]),
        action_inputs=_batch_feature(cat_features([v.action_inputs for v in values])))


def split_stage1(value, batch):
    return [GrootStage1Output(value.input_embeds[i:i+1], value.attention_mask[i:i+1],
        value.image_token_mask[i:i+1], _batch_feature(row_features(value.action_inputs, i))) for i in range(batch)]


def split_stage2(value, batch):
    return [GrootStage2Output(value.backbone_features[i:i+1], value.attention_mask[i:i+1],
        _batch_feature(row_features(value.action_inputs, i))) for i in range(batch)]


@dataclass(frozen=True)
class GrootStage3Input:
    """One request's conditioning as the stage-3 payloads carry it (``stage2_out``).

    ``embodiment`` is the host copy of ``action_inputs["embodiment_id"]`` and
    ``exec_domain`` the producer's ``device/autocast`` pair; both are part of
    the bucket key and are read on the worker without touching the GPU.
    """

    stage2: GrootStage2Output
    schedule_id: str
    embodiment: tuple[int, ...]
    exec_domain: str


def stage3_input(runner: GrootStagedRunner, stage2: GrootStage2Output) -> GrootStage3Input:
    """Freeze the host-side bucket identity of ``stage2`` (producer thread, under its lock)."""
    if stage2.action_inputs is None:
        raise ValueError("stage3_input: stage2.action_inputs is None")
    emb = stage2.action_inputs["embodiment_id"]
    embodiment = tuple(int(v) for v in emb.detach().reshape(-1).tolist())
    features = stage2.backbone_features
    exec_domain = f"{features.device}/{runner._device_type}"  # noqa: SLF001
    return GrootStage3Input(
        stage2=stage2,
        schedule_id=runner.live_schedule().schedule_id,
        embodiment=embodiment,
        exec_domain=exec_domain,
    )


def order_timesteps(values) -> tuple[float, ...]:
    """GR00T loops in ascending flow time."""
    return tuple(sorted(set(float(v) for v in values)))


# ---------------------------------------------------------------------------
# Single-forward helpers (shared by the batcher and the direct path)
# ---------------------------------------------------------------------------


def run_miss(
    runner: GrootStagedRunner,
    stage2: GrootStage2Output,
    noise: torch.Tensor,
    *,
    schedule: DenoiseSchedule,
    capture: bool,
) -> tuple[GrootStage3Output, dict[int, torch.Tensor]]:
    """Full loop from ``noise`` (``[B,H,D]``); ``caps[i]`` = x consumed by step ``i``.

    Runs inside ``runner.session()`` (entered here: the worker thread has no
    session of its own). With ``capture=False`` the loop is called without an
    observer, the transcription's original call shape.
    """
    live = runner.live_schedule()
    if schedule != live:
        raise RuntimeError(
            f"run_miss: requested schedule {schedule.schedule_id} but the action head "
            f"is running {live.schedule_id}"
        )
    caps: dict[int, torch.Tensor] = {}
    on_step = None
    if capture:

        def on_step(step: int, x_in: torch.Tensor, x_out: torch.Tensor) -> None:
            del x_out
            caps[step] = x_in.detach().clone()

    with runner.session():
        out = runner.run_stage3(stage2, noise=noise, on_step=on_step)
    if capture and sorted(caps) != list(range(schedule.num_steps)):
        raise RuntimeError(
            f"run_miss: captured steps {sorted(caps)} but the schedule has "
            f"{schedule.num_steps} steps"
        )
    return out, caps


def split_miss(
    out: GrootStage3Output,
    caps: dict[int, torch.Tensor],
    *,
    schedule: DenoiseSchedule,
    save_timesteps_per_request: list[Optional[tuple[float, ...]]],
) -> list[Stage3VariantOutput]:
    """One ``Stage3VariantOutput`` per batch row, carrying exactly its own snapshot set.

    A request that asked for any snapshot also receives step 0 under
    ``snapshot_t(0)``: the noise as the loop consumed it, which is what a
    build records as ``noise_action_0`` (plan §9-3).
    """
    replies: list[Stage3VariantOutput] = []
    for i, wanted in enumerate(save_timesteps_per_request):
        inter = None
        if wanted is not None:
            if 0 not in caps:
                raise RuntimeError("split_miss: step 0 was not captured")
            inter = {schedule.snapshot_t(0): caps[0][i : i + 1]}
            for t in wanted:
                index = schedule.snapshot_index(t)
                if index not in caps:
                    raise RuntimeError(
                        f"split_miss: snapshot t={t} (step {index}) was not captured"
                    )
                inter[float(t)] = caps[index][i : i + 1]
        replies.append(
            Stage3VariantOutput(action=out.action_pred[i : i + 1], intermediates=inter)
        )
    return replies


def run_warm(
    runner: GrootStagedRunner,
    stage2: GrootStage2Output,
    start_x: torch.Tensor,
    start_t: float,
    *,
    schedule: DenoiseSchedule,
    capture_first_step: bool,
) -> GrootStage3Output:
    """Resume the loop from ``start_x`` (``[B,H,D]``) at ``start_t`` under ``schedule``."""
    with runner.session():
        return runner.run_stage3_from(
            stage2,
            start_x,
            start_t,
            schedule=schedule,
            capture_first_step=capture_first_step,
        )


def split_warm(out: GrootStage3Output, batch: int) -> list[Stage3VariantOutput]:
    replies: list[Stage3VariantOutput] = []
    for i in range(batch):
        replies.append(
            Stage3VariantOutput(
                action=out.action_pred[i : i + 1],
                intermediates=None,
                first_step_input=(
                    None if out.first_step_input is None else out.first_step_input[i : i + 1]
                ),
                first_step_x=None if out.first_step_x is None else out.first_step_x[i : i + 1],
            )
        )
    return replies


def cat_stage2(inputs: list[GrootStage3Input]) -> GrootStage2Output:
    """Concatenate same-shape conditioning along the batch axis (no padding)."""
    first = inputs[0].stage2
    keys = list(first.action_inputs.keys())
    for item in inputs[1:]:
        s2 = item.stage2
        if tuple(s2.backbone_features.shape[1:]) != tuple(first.backbone_features.shape[1:]):
            raise ValueError("cat_stage2: conditioning shapes differ inside one bucket")
        if list(s2.action_inputs.keys()) != keys:
            raise ValueError("cat_stage2: action_inputs keys differ inside one bucket")
    if len(inputs) == 1:
        return first
    action_inputs: dict[str, Any] = {}
    for key in keys:
        values = [item.stage2.action_inputs[key] for item in inputs]
        if torch.is_tensor(values[0]):
            action_inputs[key] = torch.cat(values, dim=0)
        else:
            if any(v != values[0] for v in values[1:]):
                raise ValueError(f"cat_stage2: non-tensor action input {key!r} differs")
            action_inputs[key] = values[0]
    return GrootStage2Output(
        backbone_features=torch.cat([i.stage2.backbone_features for i in inputs], dim=0),
        attention_mask=torch.cat([i.stage2.attention_mask for i in inputs], dim=0),
        action_inputs=_batch_feature(action_inputs),
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class GrootStageBatcher:
    """``StageBatcher`` for ``GrootStagedRunner`` with same-shape buckets."""

    def __init__(self, runner: GrootStagedRunner) -> None:
        self._runner = runner

    # -- bucket key -------------------------------------------------------

    @staticmethod
    def bucket_key(payload: Any) -> Hashable:
        cond = payload.stage2_out
        if not isinstance(cond, GrootStage3Input):
            return ("unknown", None, None)
        s2 = cond.stage2
        state = s2.action_inputs["state"]
        state_mask = s2.action_inputs["state_mask"]
        shape_key = (
            cond.schedule_id,
            cond.exec_domain,
            cond.embodiment,
            tuple(s2.backbone_features.shape[1:]),
            str(s2.backbone_features.dtype),
            str(s2.backbone_features.device),
            tuple(s2.attention_mask.shape[1:]),
            tuple(state.shape[1:]),
            str(state.dtype),
            tuple(state_mask.shape[1:]),
        )
        if isinstance(payload, Stage3MissPayload):
            return ("miss", None, payload.num_steps, str(payload.noise.dtype), shape_key)
        if isinstance(payload, Stage3WarmStartPayload):
            return (
                "warm_start",
                payload.start_t,
                payload.num_steps,
                str(payload.start_x.dtype),
                shape_key,
            )
        return ("unknown", None, None)

    @staticmethod
    def default_save_timesteps() -> tuple[float, ...]:
        # The legacy GR00T MISS keeps no snapshot (libraries are built offline).
        return ()

    @staticmethod
    def order_timesteps(values) -> tuple[float, ...]:
        return order_timesteps(values)

    # -- stage 1 / 2: preserve each request's token length ---------------------

    def _early_groups(self, payloads, key, run):
        groups = defaultdict(list)
        for i, payload in enumerate(payloads):
            groups[key(payload)].append((i, payload))
        replies = [None] * len(payloads)
        for group in groups.values():
            with self._runner.session():
                outputs = run([p for _, p in group])
            for (i, _), out in zip(group, outputs, strict=True):
                replies[i] = out
        return replies

    def run_stage1_batch(self, payloads: list) -> list:
        return self._early_groups(payloads, feature_shape,
            lambda values: split_stage1(self._runner.run_stage1(cat_features(values)), len(values)))

    def run_stage2_batch(self, payloads: list, *, capture: bool) -> list:
        del capture
        return self._early_groups(payloads, lambda p: (feature_shape(p.input_embeds), feature_shape(p.action_inputs)),
            lambda values: split_stage2(self._runner.run_stage2_llm(cat_stage1(values)), len(values)))

    # -- stage 3 -------------------------------------------------------------

    def _schedule_for(self, num_steps: int) -> DenoiseSchedule:
        live = self._runner.live_schedule()
        if int(num_steps) != live.num_steps:
            raise RuntimeError(
                f"GR00T stage 3 bucket asks for {num_steps} steps but the action head "
                f"runs {live.num_steps} ({live.schedule_id})"
            )
        return live

    def run_stage3_miss(
        self,
        payloads: list,
        *,
        num_steps: int,
        save_timesteps_per_request: list,
    ) -> list:
        schedule = self._schedule_for(num_steps)
        inputs = [p.stage2_out for p in payloads]
        stage2 = cat_stage2(inputs)
        device = stage2.backbone_features.device
        noise = torch.stack(
            [p.noise if p.noise.dim() == 2 else p.noise[0] for p in payloads], dim=0
        ).to(device)
        capture = any(s is not None for s in save_timesteps_per_request)
        out, caps = run_miss(self._runner, stage2, noise, schedule=schedule, capture=capture)
        return split_miss(
            out, caps, schedule=schedule, save_timesteps_per_request=save_timesteps_per_request
        )

    def run_stage3_warm(
        self,
        payloads: list,
        *,
        start_t: float,
        num_steps: int,
        capture_first_step: bool,
    ) -> list:
        schedule = self._schedule_for(num_steps)
        inputs = [p.stage2_out for p in payloads]
        stage2 = cat_stage2(inputs)
        device = stage2.backbone_features.device
        start_x = torch.stack(
            [p.start_x if p.start_x.dim() == 2 else p.start_x[0] for p in payloads], dim=0
        ).to(device)
        out = run_warm(
            self._runner,
            stage2,
            start_x,
            start_t,
            schedule=schedule,
            capture_first_step=capture_first_step,
        )
        return split_warm(out, len(payloads))


__all__ = [
    "GrootStage3Input",
    "GrootStageBatcher",
    "cat_stage2",
    "order_timesteps",
    "run_miss",
    "run_warm",
    "split_miss",
    "split_warm",
    "stage3_input",
]
