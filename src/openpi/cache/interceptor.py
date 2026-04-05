"""InferenceInterceptor: cache-aware drop-in replacement for Policy.

Overview
--------
``InferenceInterceptor`` wraps an existing ``Policy`` object and implements
the same ``BasePolicy`` interface, making it a transparent substitute for the
WebSocket server.  When ``--cache`` is passed to ``serve_policy.py``, the
server receives this interceptor instead of the raw ``Policy`` with zero
changes to ``WebsocketPolicyServer`` or any client code.

Current state (Step 4)
----------------------
The interceptor routes inference through the *staged* public API
(``run_stage1`` / ``run_stage2`` / ``run_stage3``) introduced in Step 1,
times each stage using ``SystemTimer`` with CUDA Event backends (Step 2),
and optionally integrates ``CacheOrchestrator`` (Step 4) for cache check
and write at CP1 and CP3 checkpoints.

When ``orchestrator`` is provided:
* CP1 check after Stage 1: on FULL_HIT, skip Stage 2 + 3, return cached action.
* CP1 write after normal inference: store state -> action mapping for future lookups.
* CP3 consume at infer() entry: check for pre-scheduled action (stub in Step 4).
* CP3 check after Stage 3: infrastructure validation only (always MISS in Step 4).

When ``orchestrator=None``: behavior is identical to Step 2 (zero overhead).

Timing probes:
* Stage 1 (vision + token prep)  -> probe ``"stage1_vision"`` (CUDA backend)
* Stage 2 (LLM backbone)         -> probe ``"stage2_llm"``    (CUDA backend)
* Stage 3 (flow matching)        -> probe ``"stage3_flow"``   (CUDA backend)
* End-to-end wall time           -> probe ``"total_inference"`` (CPU backend)
* CP1 check/write, CP3 check     -> probes ``"cp1_check"`` etc. (CPU backend)

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
               CachePayload (Step 3, for CP1 write),
               CheckpointID (types.py)
  CONSUMED BY: WebsocketPolicyServer (as BasePolicy drop-in)
  IF CHANGED:  Server sees no change (same BasePolicy interface)
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Callable, Optional

import jax
import numpy as np
import torch
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.cache.components.judge import HitType
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer, TaskLifecycle
from openpi.cache.types import CheckpointID
from openpi.models import model as _model
from openpi.policies import policy as _policy

logger = logging.getLogger(__name__)


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

        # ---- torch.compile for the three stage methods ----
        # sample_actions() is compiled by PI0Pytorch.__init__ using
        # config.pytorch_compile_mode (default "max-autotune").  The stage
        # methods are compiled here with the same mode so the interceptor
        # achieves equivalent throughput.
        #
        # Persistent cache: torch.compile (inductor backend) automatically
        # saves compiled artifacts to disk between restarts:
        #   default location : ~/.cache/torch/inductor/
        #   override via env : TORCHINDUCTOR_CACHE_DIR=/your/path
        # On first run the compilation takes 30–120 s; subsequent starts with
        # the same model and input shapes load from cache and skip recompilation.
        #
        # If pytorch_compile_mode is None (disabled in config), the stage
        # methods run in eager mode — correct but slower.
        raw_compile_mode: str | None = getattr(
            self._model.config, "pytorch_compile_mode", None
        )
        # Staged run_stage1/2/3 execution passes outputs across independently
        # compiled functions. In practice, max-autotune can trigger CUDAGraph
        # output-lifetime issues in this pattern; prefer the no-cudagraph mode.
        compile_mode: str | None = raw_compile_mode
        if raw_compile_mode == "max-autotune":
            compile_mode = "max-autotune-no-cudagraphs"
            logger.info(
                "InferenceInterceptor: overriding compile mode for staged path "
                "from '%s' to '%s' to avoid CUDAGraph output reuse errors.",
                raw_compile_mode,
                compile_mode,
            )
        if compile_mode is not None:
            self._stage1_fn: Callable = torch.compile(
                self._model.run_stage1, mode=compile_mode
            )
            self._stage2_fn: Callable = torch.compile(
                self._model.run_stage2, mode=compile_mode
            )
            self._stage3_fn: Callable = torch.compile(
                self._model.run_stage3, mode=compile_mode
            )
            logger.info(
                "InferenceInterceptor: stage methods compiled "
                "(mode='%s', cache: ~/.cache/torch/inductor or $TORCHINDUCTOR_CACHE_DIR).",
                compile_mode,
            )
        else:
            # Compile disabled — run in eager mode.
            self._stage1_fn = self._model.run_stage1
            self._stage2_fn = self._model.run_stage2
            self._stage3_fn = self._model.run_stage3
            logger.info(
                "InferenceInterceptor: pytorch_compile_mode is None, "
                "stage methods running in eager mode."
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

        # ---- CacheOrchestrator (Step 4) ----
        # When orchestrator=None, all cache code paths are skipped (zero overhead).
        # Data flow: Interceptor -> Orchestrator -> CacheStorage facade (Step 3)
        self._orchestrator = orchestrator
        self._write_executor: concurrent.futures.ThreadPoolExecutor | None = None
        if orchestrator is not None:
            self._timer.register_probe("cp1_sum", backend="cpu")
            self._timer.register_probe("cp1_write", backend="cpu")
            self._timer.register_probe("cp3_sum", backend="cpu")
            # Single-thread pool for non-blocking cache writes.
            # One thread ensures writes are serialised (no concurrent upserts)
            # while not blocking inference.
            self._write_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="cache_write"
            )

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
            self._orchestrator.on_episode_start()

    def on_episode_end(self, success: bool) -> None:
        """Finalise per-episode state. Called when simulator sends episode_end.

        Prints timing summary for the completed episode, then resets the
        timer window so the next episode starts with fresh statistics.
        """
        self._timer.on_task_end()
        self._timer.on_task_begin()

    def on_task_end(self) -> None:
        """Finalise and report timing for the completed task.

        Waits for any pending background writes to finish, then forwards
        to ``SystemTimer.on_task_end()`` for summary printing.

        Called when a client WebSocket connection closes.
        """
        # Drain pending background writes before reporting timing.
        if self._write_executor is not None:
            self._write_executor.shutdown(wait=True)
            # Recreate for next task.
            self._write_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="cache_write"
            )
        self._timer.on_task_end()

    # -----------------------------------------------------------------------
    # BasePolicy interface
    # -----------------------------------------------------------------------

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """Cache-aware inference through the staged API.

        Pipeline:
        1. Input transforms (same as Policy.infer).
        2. CP3 consume: check if previous cycle pre-scheduled an action (stub in Step 4).
        3. Stage 1 (vision) -> CP1 check -> on HIT: early return cached action.
        4. Stage 2 (LLM) -> Stage 3 (flow matching).
        5. CP1 write: store state -> action for future lookups.
        6. CP3 check: infrastructure validation (always MISS in Step 4).
        7. Build outputs.

        When orchestrator=None, steps 2/3-hit/5/6 are skipped — identical to Step 2.

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

        # ---- CP3 consume point (before any stage) ----
        # Check if previous cycle pre-scheduled an action via CP3.
        # Data flow: orchestrator._next_action_scheduled -> action (or None)
        # Stub in Step 4: always None. Step 6 implements actual skip.
        if self._orchestrator is not None:
            scheduled_action = self._orchestrator.should_skip_inference()
            if scheduled_action is not None:
                # scheduled_action is CachePayload.next_action_chunk: CPU float32 [50, 32].
                # inputs["state"] is a batched GPU tensor [1, state_dim].
                # Strip batch dim from state only; action has no batch dim.
                state_np = np.asarray(inputs["state"][0, ...].detach().cpu())
                action_np = np.asarray(scheduled_action.detach().cpu()) \
                    if isinstance(scheduled_action, torch.Tensor) \
                    else np.asarray(scheduled_action)
                outputs: dict[str, Any] = {
                    "state": state_np,
                    "actions": action_np,
                }
                outputs = self._output_transform(outputs)
                self._orchestrator.clear()
                return outputs

        # ---- 2. Staged inference with cache checks ----
        # Mark a fresh compiled-step boundary before invoking staged compiled
        # functions; this prevents CUDAGraph-managed outputs from being read
        # across step boundaries.
        torch.compiler.cudagraph_mark_step_begin()
        with self._timer.measure("total_inference"):
            with torch.no_grad():
                # Stage 1: SigLIP vision encoding + prefix embedding.
                with self._timer.measure("stage1_vision"):
                    stage1 = self._stage1_fn(observation)

                # CP1: check cache after Stage 1.
                # Data flow: stage1 -> orchestrator.check(CP1) -> CheckResult
                # On HIT: skip stage2 + stage3, return cached action.
                if self._orchestrator is not None:
                    with self._timer.measure("cp1_sum"):
                        cp1_result = self._orchestrator.check(
                            CheckpointID.CP1, stage1=stage1
                        )
                    if cp1_result.hit_type == HitType.FULL_HIT:
                        # Build outputs from cached payload.
                        # CachePayload.action_chunk is CPU float32 [50, 32].
                        # Move to device and add batch dim to match normal path.
                        outputs = {
                            "state": inputs["state"],
                            "actions": cp1_result.payload.action_chunk.to(
                                self._pytorch_device
                            )[None, ...],  # [1, 50, 32]
                        }
                        outputs = jax.tree.map(
                            lambda x: np.asarray(x[0, ...].detach().cpu()),
                            outputs,
                        )
                        outputs = self._output_transform(outputs)
                        self._orchestrator.clear()
                        return outputs

                # Stage 2: Gemma 2B backbone forward pass -> KV cache.
                with self._timer.measure("stage2_llm"):
                    stage2 = self._stage2_fn(stage1)

                # CP2 slot: remains commented out (suspended).
                # TODO(Step 7): insert CP2 cache check here.

                # Stage 3: Action Expert — 10-step Euler flow-matching loop.
                with self._timer.measure("stage3_flow"):
                    stage3 = self._stage3_fn(stage2, noise=start_noise)

                # Post-inference cache operations (only on normal path, not on hit path).
                if self._orchestrator is not None:
                    # CP3 check: infrastructure validation only.
                    # No CP3 entries exist in Step 4, so this always returns MISS.
                    # Real CP3 write + skip deferred to Step 6 (DeferredWriter).
                    with self._timer.measure("cp3_sum"):
                        _cp3_result = self._orchestrator.check(
                            CheckpointID.CP3, stage1=stage1, stage3=stage3
                        )

                    # Write CP1 entry for future lookups — non-blocking.
                    # Tensors are materialised to CPU here (in the main thread)
                    # so the background thread never touches GPU state.
                    cp1_payload = CachePayload(
                        action_chunk=stage3.action_chunk[0].detach().cpu().float().contiguous()
                    )
                    cp1_query_keys = self._orchestrator.build_keys(CheckpointID.CP1)
                    self._write_executor.submit(
                        self._bg_write, CheckpointID.CP1, cp1_payload, cp1_query_keys
                    )

                    # Release per-cycle cached data.
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
    # Background write helper
    # -----------------------------------------------------------------------

    def _bg_write(
        self,
        checkpoint_id: CheckpointID,
        payload: CachePayload,
        query_keys: dict[str, torch.Tensor],
    ) -> None:
        """Execute cache write in background thread. All tensors must be CPU."""
        try:
            self._orchestrator.write_with_keys(checkpoint_id, payload, query_keys)
        except Exception:
            logger.exception("Background cache write failed for %s", checkpoint_id.name)

    # -----------------------------------------------------------------------
    # BasePolicy metadata property
    # -----------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """Forward metadata to the wrapped policy (robot type, action shape, etc.)."""
        return self._policy.metadata
