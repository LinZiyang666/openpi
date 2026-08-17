"""Cache-aware GR00T inference: drive CacheOrchestrator around the two-stage split.

What this is
------------
The GR00T counterpart of ``openpi.cache.interceptor.InferenceInterceptor``. It
is a parallel implementation rather than a shared base class, for two reasons
that are not stylistic:

* that module imports jax at import time, and the GR00T virtualenv has no jax,
  so it cannot even be loaded there;
* its coordinator routing, sidecar executors, meta-device sentinels, warm start
  and CP3 handling are all unused here, and a common base would have to carry
  Pi0.5's three-stage shape into a model that has two.

What *is* shared is everything that matters: the orchestrator, the storage
facade, the judges, gates and search strategies are model-agnostic — they only
ever see ``stage1=<opaque>`` forwarded to a KeyBuilder.

Where it sits
-------------
It satisfies the same minimal protocol as a raw GR00T policy (``get_action``),
so it is injected *inside* ``GrootPolicyAdapter`` rather than wrapped around
it. The adapter validates the wire contract; that check belongs outermost, so
a malformed observation is rejected before it can reach the key builder.

Verdict handling is binary. WARM_START is impossible by construction — there
is no third stage to resume into — and is refused at config load rather than
here, because the orchestrator silently downgrades an unsatisfiable WARM_START
to MISS, which would make a runtime check unreachable.

Coupling map:
  DEPENDS ON:  GrootStagedRunner, CacheOrchestrator, a Gr00tPolicy-shaped object
  CONSUMED BY: GrootPolicyAdapter (as the injected policy)
  IF CHANGED:  the closed-loop smoke gate must be re-run
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch

from openpi.cache.components.judge import HitType
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Batching helpers (mirror gr00t.model.policy, which we must not import)
# ------------------------------------------------------------------


def _is_batched(obs: dict[str, Any]) -> bool:
    for key, value in obs.items():
        if "state" in key and len(value.shape) < 3:  # (B, Time, Dim)
            return False
    return True


def _unsqueeze_values(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            out[key] = np.expand_dims(value, axis=0)
        elif isinstance(value, list):
            out[key] = np.expand_dims(np.array(value), axis=0)
        elif isinstance(value, torch.Tensor):
            out[key] = value.unsqueeze(0)
        else:
            out[key] = value
    return out


def _squeeze_values(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            out[key] = np.squeeze(value, axis=0)
        elif isinstance(value, torch.Tensor):
            out[key] = value.squeeze(0)
        else:
            out[key] = value
    return out


# ------------------------------------------------------------------
# Interceptor
# ------------------------------------------------------------------


class GrootCacheInterceptor:
    """Drop-in replacement for a GR00T policy that consults the cache at CP1.

    Args:
        policy: a ``Gr00tPolicy``-shaped object. Only its transform pipeline is
            used; inference goes through ``runner`` so the two halves can be
            timed and separated.
        runner: the staged runner wrapping the same policy's model.
        orchestrator: when ``None`` every call runs both stages, which is the
            teacher-only path used for library collection.
        timer: shared with ``runner``; owns the checkpoint and end-to-end
            probes. ``None`` installs a disabled timer.
    """

    def __init__(
        self,
        policy: Any,
        runner: GrootStagedRunner,
        *,
        orchestrator: Optional[CacheOrchestrator] = None,
        timer: Optional[SystemTimer] = None,
    ) -> None:
        self._policy = policy
        self._runner = runner
        self._orchestrator = orchestrator
        self._timer = timer if timer is not None else SystemTimer(enabled=False)
        self._timer.register_probe("total_inference", backend="cpu")
        if orchestrator is not None:
            self._timer.register_probe("cp1_sum", backend="cpu")

    # -- TaskLifecycle ---------------------------------------------------

    def on_task_begin(self) -> None:
        self._timer.on_task_begin()
        if self._orchestrator is not None:
            self._orchestrator.on_task_begin()

    def on_episode_start(
        self,
        experiment: str = "",
        task: str = "",
        episode_id: int = -1,
        episode_name: str = "",
        extra_metadata: dict | None = None,
    ) -> None:
        del experiment, episode_name  # accepted for signature parity
        if self._orchestrator is not None:
            self._orchestrator.on_episode_start(
                task_key=task,
                episode_id=str(episode_id),
                extra_metadata=extra_metadata,
            )

    def on_episode_end(self, success: bool) -> None:
        # `success` is dropped: the orchestrator's episode hook takes no
        # outcome. It still must be called even under write_policy=never,
        # because closing the search session happens in its finally block.
        del success
        if self._orchestrator is not None:
            self._orchestrator.on_episode_end()
        self._timer.on_task_end()
        self._timer.on_task_begin()

    def on_task_end(self) -> None:
        self._timer.on_task_end()
        if self._orchestrator is not None:
            self._orchestrator.on_task_end()

    # -- observability ---------------------------------------------------

    @staticmethod
    def _build_hit_meta(cp1_result) -> dict:
        """Same field set as the Pi0.5 interceptor, so one analysis path reads both.

        ``start_t`` is always None here: it only ever describes a warm start,
        which this two-stage split has no room for.
        """
        if cp1_result is None:
            return {
                "hit_type": "MISS",
                "start_t": None,
                "winner_id": None,
                "cp1_score": None,
                "searched": True,
            }
        return {
            "hit_type": cp1_result.hit_type.name,
            "start_t": None,
            "winner_id": cp1_result.entry_id,
            "cp1_score": cp1_result.score,
            "searched": cp1_result.searched,
        }

    # -- inference -------------------------------------------------------

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        """One cache-aware inference cycle.

        Mirrors ``Gr00tPolicy.get_action`` step for step so that, with the
        orchestrator disabled, the numbers are the unsplit model's. The cache
        check sits between the two stages, deliberately outside the inference
        context: tensors born inside it stay inference tensors even after a
        `.cpu()`, and the cache keeps them across steps.
        """
        with self._timer.measure("total_inference"):
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

            cp1_result = None
            try:
                if self._orchestrator is not None:
                    with self._timer.measure("cp1_sum"):
                        cp1_result = self._orchestrator.check(
                            CheckpointID.CP1, stage1=stage1
                        )
                    if cp1_result.hit_type == HitType.WARM_START:
                        raise RuntimeError(
                            "CP1 returned WARM_START, which a two-stage split "
                            "cannot serve. Config validation should have refused "
                            "this recipe at load time."
                        )

                if cp1_result is not None and cp1_result.hit_type == HitType.FULL_HIT:
                    chunk = cp1_result.payload.action_chunk
                else:
                    with self._runner.session():
                        chunk = self._runner.run_stage2(stage1).action_pred

                action_cpu = self._to_storage_tensor(chunk)

                if self._orchestrator is not None:
                    self._orchestrator.broadcast_action(action_cpu)
                    if cp1_result.query_keys is not None:
                        self._orchestrator.buffer_for_write(
                            cp1_result.query_keys, action_cpu
                        )
            finally:
                if self._orchestrator is not None:
                    self._orchestrator.clear()

            unnormalized = self._policy.unapply_transforms(
                {"action": action_cpu[None, ...]}
            )
            if not is_batch:
                unnormalized = _squeeze_values(unnormalized)

        unnormalized["__hit_meta__"] = self._build_hit_meta(cp1_result)
        return unnormalized

    @staticmethod
    def _to_storage_tensor(chunk: torch.Tensor) -> torch.Tensor:
        """Normalise an action chunk to the storage contract: [H, D] CPU fp32.

        A cache hit hands back an already-unbatched payload tensor while a miss
        produces the model's ``[1, H, D]``; both end up the same shape here.

        The final clone is what actually lets the tensor outlive the inference
        context. Converting inside that context is not enough — the result is
        still an inference tensor, and the first in-place write from the
        storage or normaliser layers would raise.
        """
        if chunk.dim() == 3:
            chunk = chunk[0]
        out = chunk.detach().cpu().float().contiguous()
        if out.is_inference():
            out = out.clone()
        return out
