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
emb_dim]``, ``robot_state`` and ``clean_action``. The file-level ``task`` and
``success`` attributes matter as much as the arrays: the builder drops any
episode whose ``success`` attribute is absent or false, and it copies ``task``
straight into each entry's ``task_key``.

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

        with self._runner.session():
            stage1 = self._runner.run_stage1(normalized_input)
            stage2 = self._runner.run_stage2(stage1)

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
            **({} if self._vision_fields is None else {"vision_fields": self._vision_fields}),
        )
        if self._state_index is None:
            self._state_index = stage1.state_mask[0, -1].clone()

        action_cpu = stage2.action_pred[0].detach().cpu().float().contiguous()
        if action_cpu.is_inference():
            action_cpu = action_cpu.clone()

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
                noise_action_steps=[],
                clean_action=action_cpu.numpy(),
            )
        )

        unnormalized = self._policy.unapply_transforms(
            {"action": action_cpu[None, ...]}
        )
        if not is_batch:
            unnormalized = _squeeze_values(unnormalized)
        return unnormalized
