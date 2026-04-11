"""InferenceInterceptor: cache-aware drop-in replacement for Policy.

Overview
--------
``InferenceInterceptor`` wraps an existing ``Policy`` object and implements
the same ``BasePolicy`` interface, making it a transparent substitute for the
WebSocket server.  When ``--cache`` is passed to ``serve_policy.py``, the
server receives this interceptor instead of the raw ``Policy`` with zero
changes to ``WebsocketPolicyServer`` or any client code.

Cache-aware inference
---------------------
The interceptor routes inference through the *staged* public API
(``run_stage1`` / ``run_stage2`` / ``run_stage3``), times each stage
using ``SystemTimer`` with CUDA Event backends, and optionally
integrates ``CacheOrchestrator`` for cache check and write at CP1
and CP3 checkpoints.

When ``orchestrator`` is provided, CP1 supports three-level judgment:
* FULL_HIT: skip Stage 2 + 3, return cached action.
* WARM_START: run Stage 2, then partial Stage 3 from cached x_t via
  ``run_stage3_from()``.
* MISS: run full Stage 2 + Stage 3 (with intermediates collection for
  future warm starts).

When ``orchestrator=None``: zero-overhead pass-through to compiled stages.

Timing probes:
* Stage 1 (vision + token prep)  -> probe ``"stage1_vision"`` (CUDA backend)
* Stage 2 (LLM backbone)         -> probe ``"stage2_llm"``    (CUDA backend)
* Stage 3 (flow matching)        -> probe ``"stage3_flow"``   (CUDA backend)
* Stage 3 (warm start)           -> probe ``"stage3_warm"``   (CUDA backend)
* End-to-end wall time           -> probe ``"total_inference"`` (CPU backend)
* CP1/CP3 aggregate              -> probes ``"cp1_sum"`` / ``"cp3_sum"`` (CPU)

Task lifecycle
--------------
``InferenceInterceptor`` implements the ``TaskLifecycle`` protocol.
``WebsocketPolicyServer._handler`` calls ``on_task_begin()`` / ``on_task_end()``
when a client connection opens / closes.  ``on_task_end()`` triggers
``SystemTimer.on_task_end()``, which prints a per-probe summary to the
terminal and (optionally) writes a CSV.

External contract (what the client / server sees)
-------------------------------------------------
``infer(obs)`` returns::

    {
        "actions": np.ndarray  [action_horizon, action_dim],
        "state":   np.ndarray  [...],
        "server_timing" is added by the server, not here,
    }

Limitations
-----------
* Only PyTorch policies are supported.  JAX policies do not expose
  ``run_stage1 / run_stage2 / run_stage3``.
* ``SystemTimer`` is created with default settings.  To customise
  ``buffer_size`` or ``output_csv_dir``, pass a pre-configured
  ``SystemTimer`` instance via the ``timer`` argument.

Coupling map:
  DEPENDS ON:  Policy (wrapped), SystemTimer (Step 2),
               CacheOrchestrator + CheckResult + HitType (Step 4, optional),
               CheckpointID (types.py)
  CONSUMED BY: WebsocketPolicyServer (as BasePolicy drop-in)
  IF CHANGED:  Server sees no change (same BasePolicy interface)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import jax
import numpy as np
import torch
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.cache.components.judge import HitType
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.timing import SystemTimer, TaskLifecycle
from openpi.cache.types import CheckpointID
from openpi.models import model as _model
from openpi.policies import policy as _policy

logger = logging.getLogger(__name__)

_NUM_STEPS = 10  # matches pi0_pytorch.run_stage3 default


class InferenceInterceptor(_base_policy.BasePolicy):
    """Drop-in Policy replacement that routes inference through the staged API.

    Wraps a ``Policy`` instance.  All input/output transforms are reused from
    the wrapped policy so normalisation, tokenisation, and action remapping are
    identical to the original path.

    This class implements ``TaskLifecycle`` so that ``WebsocketPolicyServer``
    can call ``on_task_begin`` / ``on_task_end`` without depending on this
    class directly (the server checks ``hasattr(policy, "on_task_begin")``).

    Args:
        policy: A fully initialised ``Policy`` object with
                ``is_pytorch=True``.  JAX policies are not supported.
        timer: Optional pre-configured ``SystemTimer``.  When ``None``
               (default), a timer with default settings is created
               (``enabled=True``, ``buffer_size=10_000``, no CSV output).
               Pass a custom instance to enable CSV export or adjust the
               buffer size::

                   timer = SystemTimer(enabled=True, output_csv_dir="/tmp/t")
                   interceptor = InferenceInterceptor(policy, timer=timer)

    Raises:
        ValueError: If the wrapped policy is not a PyTorch policy.
    """

    def __init__(
        self,
        policy: _policy.Policy,
        timer: Optional[SystemTimer] = None,
        orchestrator: Optional["CacheOrchestrator"] = None,
        eager: bool = False,
        collect_images: bool = False,
    ) -> None:
        if not policy._is_pytorch_model:  # noqa: SLF001
            raise ValueError(
                "InferenceInterceptor only supports PyTorch policies. "
                "The wrapped policy must be initialised with is_pytorch=True."
            )

        self._policy = policy
        # Borrow internals from the wrapped Policy — references only, no copy.
        self._model = policy._model                        # PI0Pytorch instance  # noqa: SLF001
        self._input_transform = policy._input_transform    # composed transform fn  # noqa: SLF001
        self._output_transform = policy._output_transform  # composed transform fn  # noqa: SLF001
        self._pytorch_device = policy._pytorch_device      # e.g. "cuda:0"          # noqa: SLF001

        # ---- Stage functions (eager or compiled) ----
        if eager:
            self._stage1_fn = self._model.run_stage1
            self._stage2_fn = self._model.run_stage2
            self._stage3_fn = self._model.run_stage3
            logger.info("InferenceInterceptor: eager mode (no compile).")
        else:
            self._stage1_fn, self._stage2_fn, self._stage3_fn = (
                self._get_or_compile_stages()
            )

        # ---- SystemTimer setup ----
        # Each probe corresponds to one pipeline component.  The backend
        # ("cuda" vs "cpu") determines the timing method:
        #   "cuda"  → torch.cuda.Event (accurate GPU execution time)
        #   "cpu"   → time.perf_counter_ns (wall time, for CPU-only ops)
        #
        # When Step 4 adds cache checkpoints (CP1 / CP2 / CP3), their probes
        # are registered here as well (using "cpu" or "cuda" as appropriate
        # for the checkpoint's dominant compute location).
        self._timer: SystemTimer = timer if timer is not None else SystemTimer()
        self._timer.register_probe("stage1_vision",   backend="cuda")
        self._timer.register_probe("stage2_llm",      backend="cuda")
        self._timer.register_probe("stage3_flow",     backend="cuda")
        # CPU wall time wrapping all three stages; captures Python + sync
        # overhead in addition to pure GPU time.  Useful as the "felt" latency
        # from the robot's perspective.
        self._timer.register_probe("total_inference", backend="cpu")

        # ---- Image collection for cache key builders ----
        self._collect_images = collect_images

        # ---- CacheOrchestrator ----
        # When orchestrator=None, all cache code paths are skipped (zero overhead).
        # Data flow: Interceptor -> Orchestrator -> CacheStorage facade
        self._orchestrator = orchestrator
        if orchestrator is not None:
            self._timer.register_probe("cp1_sum", backend="cpu")
            self._timer.register_probe("cp3_sum", backend="cpu")
            self._timer.register_probe("stage3_warm", backend="cuda")

    # -----------------------------------------------------------------------
    # Compile-once helpers
    # -----------------------------------------------------------------------

    def _get_or_compile_stages(self) -> tuple[Callable, Callable, Callable]:
        """Compile stage methods once, cache on model for reuse across Interceptors."""
        if hasattr(self._model, "_compiled_stage1_fn"):
            logger.info("InferenceInterceptor: reusing compiled stage methods.")
            return (
                self._model._compiled_stage1_fn,
                self._model._compiled_stage2_fn,
                self._model._compiled_stage3_fn,
            )

        raw_compile_mode: str | None = getattr(
            self._model.config, "pytorch_compile_mode", None
        )
        compile_mode: str | None = raw_compile_mode
        if raw_compile_mode == "max-autotune":
            compile_mode = "max-autotune-no-cudagraphs"
            logger.info(
                "InferenceInterceptor: compile mode '%s' -> '%s' "
                "(avoid CUDAGraph output reuse errors).",
                raw_compile_mode, compile_mode,
            )

        if compile_mode is not None:
            s1 = torch.compile(self._model.run_stage1, mode=compile_mode)
            s2 = torch.compile(self._model.run_stage2, mode=compile_mode)
            s3 = torch.compile(self._model.run_stage3, mode=compile_mode)
            logger.info("InferenceInterceptor: stages compiled (mode='%s').", compile_mode)
        else:
            s1 = self._model.run_stage1
            s2 = self._model.run_stage2
            s3 = self._model.run_stage3
            logger.info("InferenceInterceptor: stages running in eager mode.")

        self._model._compiled_stage1_fn = s1
        self._model._compiled_stage2_fn = s2
        self._model._compiled_stage3_fn = s3
        return s1, s2, s3

    # -----------------------------------------------------------------------
    # TaskLifecycle interface  (called by WebsocketPolicyServer._handler)
    # -----------------------------------------------------------------------

    def on_task_begin(self) -> None:
        """Reset per-task state.  Called when a client connection opens.

        Forwards to ``SystemTimer.on_task_begin()`` and
        ``CacheOrchestrator.on_task_begin()`` (resets step_counter).
        """
        self._timer.on_task_begin()
        if self._orchestrator is not None:
            self._orchestrator.on_task_begin()

    def on_episode_start(self, experiment: str, task: str, episode_id: int) -> None:
        """Reset per-episode state. Called when simulator sends episode_start."""
        if self._orchestrator is not None:
            self._orchestrator.on_episode_start(
                task_key=task,
                episode_id=str(episode_id),
            )

    def on_episode_end(self, success: bool) -> None:
        """Finalise per-episode state. Called when simulator sends episode_end.

        Triggers episode-end write (WritePolicy decides), then resets timer.
        """
        if self._orchestrator is not None:
            self._orchestrator.on_episode_end()
        self._timer.on_task_end()
        self._timer.on_task_begin()

    def on_task_end(self) -> None:
        """Finalise and report timing for the completed task.

        Forwards to ``SystemTimer.on_task_end()`` for summary printing.
        Called when a client WebSocket connection closes.
        """
        self._timer.on_task_end()

    # -----------------------------------------------------------------------
    # BasePolicy interface
    # -----------------------------------------------------------------------

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """Cache-aware inference through the staged API.

        Pipeline:
        1. Input transforms (same as Policy.infer).
        2. Stage 1 (vision) -> CP1 check -> on HIT: early return cached action.
        3. Stage 2 (LLM) -> Stage 3 (flow matching).
        4. CP3 check + broadcast_action + buffer_for_write.
        5. Build outputs.

        When orchestrator=None, cache steps are skipped — identical to base Policy.

        Timing probes: total_inference (CPU), stage1/2/3 (CUDA), cp1_check/write, cp3_check (CPU).

        Args:
            obs: Observation dict for the wrapped policy's input transform.
            noise: Optional initial noise for flow matching, shape [H, D] or [1, H, D].

        Returns:
            Dict with keys ``"actions"`` and ``"state"``.
        """
        # ---- 1. Input transforms (mirrors Policy.infer exactly) ----
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        # Extract valid model-input images (CPU numpy) before GPU transfer.
        # Passed through check() kwargs so KeyBuilder can optionally use them.
        input_images: dict[str, np.ndarray] | None = None
        if self._collect_images:
            from openpi.shared.image_extract import extract_valid_images
            input_images = extract_valid_images(inputs)

        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x))
                           .to(self._pytorch_device)[None, ...],
            inputs,
        )
        observation = _model.Observation.from_dict(inputs)

        # Optional noise forwarding (mirrors Policy.infer sample_kwargs).
        start_noise: torch.Tensor | None = None
        if noise is not None:
            start_noise = torch.from_numpy(noise).to(self._pytorch_device)
            if start_noise.ndim == 2:
                start_noise = start_noise[None, ...]

        # ---- 2. Staged inference with cache checks ----
        torch.compiler.cudagraph_mark_step_begin()
        with self._timer.measure("total_inference"):
            with torch.no_grad():
                with self._timer.measure("stage1_vision"):
                    stage1 = self._stage1_fn(observation)

                # CP1: check cache after Stage 1.
                if self._orchestrator is not None:
                    cp1_kwargs = {"stage1": stage1}
                    if input_images is not None:
                        cp1_kwargs["input_images"] = input_images
                    with self._timer.measure("cp1_sum"):
                        cp1_result = self._orchestrator.check(
                            CheckpointID.CP1, **cp1_kwargs
                        )
                    if cp1_result.hit_type == HitType.FULL_HIT:
                        cached_action = cp1_result.payload.action_chunk
                        # Broadcast action + buffer for trajectory write
                        self._orchestrator.broadcast_action(cached_action)
                        if cp1_result.query_keys is not None:
                            self._orchestrator.buffer_for_write(
                                cp1_result.query_keys, cached_action
                            )
                        outputs = {
                            "state": inputs["state"],
                            "actions": cached_action.to(
                                self._pytorch_device
                            )[None, ...],
                        }
                        outputs = jax.tree.map(
                            lambda x: np.asarray(x[0, ...].detach().cpu()),
                            outputs,
                        )
                        outputs = self._output_transform(outputs)
                        self._orchestrator.clear()
                        return outputs

                with self._timer.measure("stage2_llm"):
                    stage2 = self._stage2_fn(stage1)

                # Stage 3: three-way branch
                if (self._orchestrator is not None
                        and cp1_result.hit_type == HitType.WARM_START):
                    start_t = cp1_result.start_t
                    start_x = cp1_result.payload.intermediates[start_t].to(
                        self._pytorch_device
                    )
                    if start_x.ndim == 2:
                        start_x = start_x[None, ...]
                    with self._timer.measure("stage3_warm"):
                        stage3 = self._model.run_stage3_from(
                            stage2, start_x, start_t,
                            num_steps=cp1_result.payload.denoising_num_steps,
                        )
                elif self._orchestrator is not None:
                    # MISS: eager call with intermediates collection
                    with self._timer.measure("stage3_flow"):
                        stage3 = self._model.run_stage3(
                            stage2, noise=start_noise,
                            num_steps=_NUM_STEPS, return_intermediates=True,
                        )
                else:
                    # No-cache mode: compiled call
                    with self._timer.measure("stage3_flow"):
                        stage3 = self._stage3_fn(stage2, noise=start_noise)

            # Post-inference cache operations.
            if self._orchestrator is not None:
                cp3_kwargs = {"stage1": stage1, "stage3": stage3}
                if input_images is not None:
                    cp3_kwargs["input_images"] = input_images
                with self._timer.measure("cp3_sum"):
                    _cp3_result = self._orchestrator.check(
                        CheckpointID.CP3, **cp3_kwargs
                    )

                action_chunk_cpu = stage3.action_chunk[0].detach().cpu().float().contiguous()

                # Prepare intermediates for write (MISS only; WARM_START has no valid intermediates)
                intermediates_cpu = None
                denoising_num_steps_val = None
                if getattr(stage3, 'intermediates', None):
                    intermediates_cpu = {
                        t: x[0].detach().cpu().float().contiguous()
                        for t, x in stage3.intermediates.items()
                    }
                    denoising_num_steps_val = _NUM_STEPS

                # Broadcast action + buffer for trajectory write
                self._orchestrator.broadcast_action(action_chunk_cpu)
                if cp1_result.query_keys is not None:
                    self._orchestrator.buffer_for_write(
                        cp1_result.query_keys, action_chunk_cpu,
                        intermediates=intermediates_cpu,
                        denoising_num_steps=denoising_num_steps_val,
                    )

                self._orchestrator.clear()

        # ---- 3. Build outputs ----
        # Output format matches Policy.infer so the server and client require
        # no changes.  Timing is reported by SystemTimer at task end.
        outputs = {
            "state":   inputs["state"],
            "actions": stage3.action_chunk,
        }
        outputs = jax.tree.map(
            lambda x: np.asarray(x[0, ...].detach().cpu()), outputs
        )
        outputs = self._output_transform(outputs)
        return outputs

    # -----------------------------------------------------------------------
    # BasePolicy metadata property
    # -----------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """Forward metadata to the wrapped policy (robot type, action shape, etc.)."""
        return self._policy.metadata
