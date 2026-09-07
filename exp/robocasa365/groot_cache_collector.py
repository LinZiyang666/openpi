"""Record GR00T stage-1 embeddings to HDF5 so a cache library can be built offline.

Why a separate wrapper rather than a flag on the interceptor
------------------------------------------------------------
Collection has to run teacher-only. If the cache were active while collecting,
some of the recorded actions would be replayed library entries rather than the
teacher's own, and the resulting library would no longer describe "what the
teacher did in scene A" -- which is the whole independent variable of the
cross-scene experiment. The server therefore refuses ``--collect-hdf5``
together with ``--cache-config``, and this class never touches an orchestrator.

What it writes
--------------
Exactly the schema ``exp/common/build_in_memory_cache_artifact.py`` already
reads, so the offline builder needs no GR00T-specific branch: per step
``vision_0/1/2`` ``[256, emb_dim]``, ``prompt_emb`` ``[num_text_tokens,
emb_dim]``, ``robot_state``, ``clean_action``, the pure-noise start
``noise_action_0`` and the warm-start snapshots ``noise_action_1 .. N-1``
(``[action_horizon, action_dim]`` fp32, the x consumed by Euler step i). The
file-level ``task`` and ``success`` attributes matter as much as the arrays:
the builder drops any episode whose ``success`` attribute is absent or false,
and it copies ``task`` straight into each entry's ``task_key``.

The snapshots are captured with a forward hook on the action head's
``action_encoder`` around the upstream ``get_action`` call, so the library
holds what the real loop consumed rather than a re-implementation of it. The
loop's step count is a runtime property of the served policy, so every file is
stamped with ``denoise_schedule_id`` / ``denoising_num_steps`` read from the
live action head, and a hook count that disagrees with it is an error.

The fields are cut with the same function the online path uses, from the same
stage-1 tensors, inside the same inference/autocast context. That is what
makes the two paths comparable -- pulling ``state`` from the pre-transform
observation instead would silently record fp32 where the model saw bf16.

Environment: the GR00T island. ``openpi.collect.data_collector`` is safe to
import there (h5py + numpy only); ``openpi.collect.collection_policy`` is not,
as it imports jax.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from openpi.cache.groot.interceptor import (
    _is_batched,
    _squeeze_values,
    _unsqueeze_values,
)
from openpi.cache.groot.key_builder import slice_groot_cp1_fields
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.types import DenoiseSchedule
from openpi.collect.data_collector import EpisodeDataCollector, InferenceEmbeddings

logger = logging.getLogger(__name__)

_VISION_FIELDS = ("vision_0", "vision_1", "vision_2")


class GrootCacheCollector:
    """Teacher-only GR00T policy that also records what a cache key would be built from.

    Satisfies the same ``get_action`` protocol as the raw policy and the cache
    interceptor, so the server picks exactly one of the three and the adapter
    above it is unchanged.

    Args:
        policy: a ``Gr00tPolicy``-shaped object, used for its transforms.
        runner: staged runner over the same model.
        out_dir: directory for the per-episode HDF5 files.
        experiment: recorded as the ``experiment_name`` attribute.
    """

    def __init__(
        self,
        policy: Any,
        runner: GrootStagedRunner,
        *,
        out_dir: str,
        experiment: str = "groot_cache",
        vision_fields: tuple[str, ...] | None = None,
    ) -> None:
        self._policy = policy
        self._runner = runner
        self._collector = EpisodeDataCollector(out_dir)
        self._experiment = experiment
        # Camera list in image-token run order; None keeps the slicer's
        # three-camera RoboCasa365 default. LIBERO checkpoints feed two.
        self._vision_fields = vision_fields
        self._state_index = None
        self._schedule: DenoiseSchedule | None = None

    def _live_schedule(self) -> DenoiseSchedule:
        """The schedule the served action head is running right now."""
        return self._runner.live_schedule()

    # -- lifecycle -------------------------------------------------------

    def on_task_begin(self) -> None:
        pass

    def on_task_end(self) -> None:
        pass

    def on_episode_start(
        self,
        experiment: str = "",
        task: str = "",
        episode_id: int = -1,
        episode_name: str = "",
        extra_metadata: dict | None = None,
    ) -> None:
        self._state_index = None
        self._collector.on_episode_start(
            experiment or self._experiment,
            task,
            episode_id,
            episode_name=episode_name,
            extra_metadata=extra_metadata,
        )
        # Re-read per episode, never cached at construction: the count can be
        # changed on the policy after this wrapper exists, and the per-step
        # hook-count assertion below only means something against the value
        # the file claims.
        self._schedule = self._live_schedule()
        self._collector.set_episode_attr(
            "denoise_schedule_id", self._schedule.schedule_id
        )
        self._collector.set_episode_attr(
            "denoising_num_steps", self._schedule.num_steps
        )

    def on_episode_end(self, success: bool) -> None:
        """Flush the episode. ``success`` decides whether the builder will keep it."""
        self._collector.on_episode_end(success=success)

    # -- inference -------------------------------------------------------

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        obs_copy = observations.copy()
        is_batch = _is_batched(obs_copy)
        if not is_batch:
            obs_copy = _unsqueeze_values(obs_copy)
        for key, value in obs_copy.items():
            if not isinstance(value, np.ndarray):
                obs_copy[key] = np.array(value)

        normalized_input = self._policy.apply_transforms(obs_copy)

        # x_t at the input of every Euler step, in loop order. ``action_encoder``
        # is the first module each step feeds the current chunk through, which
        # makes its arg 0 the GR00T analogue of Pi0.5's ``action_in_proj`` hook.
        captures: list[torch.Tensor] = []

        def _action_encoder_hook(module, inp, out):
            del module, out
            captures.append(inp[0].detach())

        action_head = self._runner._model.action_head  # noqa: SLF001
        handle = action_head.action_encoder.register_forward_hook(_action_encoder_hook)
        try:
            with self._runner.session():
                stage1 = self._runner.run_stage1(normalized_input)
                stage2 = self._runner.run_stage2(stage1)
        finally:
            handle.remove()

        schedule = (
            self._schedule if self._schedule is not None else self._live_schedule()
        )
        if len(captures) != schedule.num_steps:
            raise RuntimeError(
                f"GrootCacheCollector: action_encoder fired {len(captures)} times but "
                f"the file is stamped {schedule.schedule_id} ({schedule.num_steps} "
                "steps). The policy's step count changed mid-episode or the "
                "upstream loop no longer feeds the encoder once per step."
            )

        # Cut outside the context: the slices outlive this call inside the
        # episode buffer, and tensors produced under inference_mode stay
        # inference tensors even after .cpu().
        raw = slice_groot_cp1_fields(
            stage1.input_embeds,
            stage1.image_token_mask,
            stage1.state,
            stage1.state_mask,
            enabled=None,
            expected_state_index=self._state_index,
            **(
                {}
                if self._vision_fields is None
                else {"vision_fields": self._vision_fields}
            ),
        )
        if self._state_index is None:
            self._state_index = stage1.state_mask[0, -1].clone()

        action_cpu = stage2.action_pred[0].detach().cpu().float().contiguous()
        if action_cpu.is_inference():
            action_cpu = action_cpu.clone()

        def _snapshot(x: torch.Tensor) -> np.ndarray:
            x = x[0].cpu().float().contiguous()
            if x.is_inference():
                x = x.clone()
            return x.numpy().astype(np.float32)

        init_noise = _snapshot(captures[0])
        noise_action_steps = [_snapshot(x) for x in captures[1:]]

        self._collector.record_inference(
            InferenceEmbeddings(
                # float16 for the token sequences, matching the Pi0.5 collector:
                # they dominate the file size and the offline builder upcasts
                # to fp32 before pooling anyway.
                vision_embs=[
                    raw[name].cpu().to(torch.float16).numpy()
                    for name in (self._vision_fields or _VISION_FIELDS)
                ],
                prompt_emb=raw["prompt_emb"].cpu().to(torch.float16).numpy(),
                robot_state=raw["robot_state"].cpu().float().numpy(),
                noise_action_steps=noise_action_steps,
                clean_action=action_cpu.numpy(),
                init_noise=init_noise,
            )
        )

        unnormalized = self._policy.unapply_transforms(
            {"action": action_cpu[None, ...]}
        )
        if not is_batch:
            unnormalized = _squeeze_values(unnormalized)
        return unnormalized
