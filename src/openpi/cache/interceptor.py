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
from openpi.models_pytorch.stage_device_placement import StageDeviceConfig
from openpi.policies import policy as _policy

logger = logging.getLogger(__name__)

_NUM_STEPS = 10  # matches pi0_pytorch.run_stage3 default


def _probe_backend(device_str: str | torch.device | None) -> str:
    """Select timer probe backend based on device string or torch.device."""
    if device_str is None:
        return "cpu"
    return "cuda" if str(device_str).startswith("cuda") else "cpu"


def _meta_guard(stage_name: str) -> Callable:
    """Return a sentinel function that raises on call for meta-device stages."""
    def _fn(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            f"{stage_name} is on meta device (not loaded). "
            "Cannot execute forward pass. Check cache config — "
            "meta stages require always_hit at the preceding checkpoint."
        )
    return _fn


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
        stage_config: Optional[StageDeviceConfig] = None,
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

        # ---- Stage device config ----
        sc = stage_config
        self._stage_config = sc
        # Normalize per-stage device: explicit config overrides, otherwise
        # fall back to the wrapped policy's pytorch_device (legacy default).
        _dev = self._pytorch_device
        self._stage1_device = sc.stage1 if sc and not sc.is_legacy_default else _dev
        self._stage2_device = sc.stage2 if sc and not sc.is_legacy_default else _dev
        self._stage3_device = sc.stage3 if sc and not sc.is_legacy_default else _dev

        # ---- Stage functions (eager or compiled, with meta sentinel) ----
        if eager:
            self._stage1_fn = self._model.run_stage1
            self._stage2_fn = self._model.run_stage2
            self._stage3_fn = self._model.run_stage3
            logger.info("InferenceInterceptor: eager mode (no compile).")
        else:
            self._stage1_fn, self._stage2_fn, self._stage3_fn = (
                self._get_or_compile_stages()
            )

        # Meta sentinel: override compiled/eager functions at interceptor level.
        # Sentinel is NOT cached on model — all interceptors share the same
        # device layout (relocate is model-level, done at startup).
        if sc and sc.stage2 == "meta":
            self._stage2_fn = _meta_guard("stage2")
        if sc and sc.stage3 == "meta":
            self._stage3_fn = _meta_guard("stage3")

        # ---- SystemTimer setup ----
        # Probe backend is derived from the normalized per-stage device,
        # so legacy default (stage*=None → pytorch_device) keeps CUDA timing.
        self._timer: SystemTimer = timer if timer is not None else SystemTimer()
        self._timer.register_probe("stage1_vision",   backend=_probe_backend(self._stage1_device))
        if self._stage2_device != "meta":
            self._timer.register_probe("stage2_llm",  backend=_probe_backend(self._stage2_device))
        if self._stage3_device != "meta":
            self._timer.register_probe("stage3_flow", backend=_probe_backend(self._stage3_device))
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
            if self._stage3_device != "meta":
                self._timer.register_probe("stage3_warm", backend=_probe_backend(self._stage3_device))

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

    def on_episode_start(
        self,
        experiment: str = "",
        task: str = "",
        episode_id: int = -1,
        episode_name: str = "",
    ) -> None:
        """Reset per-episode state. Called when simulator sends episode_start.

        ``experiment`` and ``episode_name`` are accepted for wrapper-signature
        alignment (``CollectionPolicy`` forwards both as kwargs) but are not
        propagated to the orchestrator: the orchestrator protocol is limited to
        ``task_key`` / ``episode_id``, and widening it here would leak
        collection-layer concerns into the cache layer. The default values keep
        every argument optional so older callers (``experiment, task,
        episode_id`` positional or kwarg) remain source-compatible.
        """
        del experiment, episode_name  # reserved; avoid unused-arg lint noise
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
    # Prefill API (Step 3 trajectory-deviation spawn runner)
    # -----------------------------------------------------------------------

    def prefill_trajectory(
        self,
        observations: list[dict],
        actions: list[np.ndarray] | None = None,
        *,
        record: bool = False,
        on_miss: str = "error",
    ) -> None:
        """Drive the cache framework along ``(obs, action)`` pairs as if those
        steps had really happened.

        After the call every stateful component (key_builder vision buffer,
        strategy trajectory buffer, orchestrator step_counter, gate/judge
        hooks) is consistent with the supplied trajectory — ready for the
        next "real" inference.

        Implementation: for each step we temporarily enter prefill mode on
        the cache storage facade (which returns a synthetic FULL_HIT
        carrying the ground-truth action) and then call ``infer`` through
        the normal path. The cache framework treats each step as a CP1
        FULL_HIT, records trajectory state, and returns the GT action; the
        caller discards the return value. See
        ``logs/trajectory_deviation_corrective_implementation.log.md`` §7.

        First version scope (other combinations raise ``NotImplementedError``):
          - ``actions`` must be provided (pure cache self-query mode deferred).
          - ``record`` must be False (HDF5 audit of prefill steps deferred).
          - ``on_miss`` must be ``"error"`` (facade synthetic hit cannot miss).
        """
        if actions is None:
            raise NotImplementedError(
                "actions=None (cache self-query mode). Future: run real search "
                "per step and use the cache's own returned actions as history."
            )
        if record:
            raise NotImplementedError(
                "record=True. Future: capture prefill steps into HDF5 tagged "
                "as 'prefill' for audit. Requires CollectionPolicy to "
                "distinguish prefill from real inference steps."
            )
        if on_miss != "error":
            raise NotImplementedError(
                f"on_miss={on_miss!r}. First version uses facade synthetic "
                "hit; MISS cannot happen. Future: 'warn' / 'fallback_infer' "
                "combined with actions=None mode."
            )
        if len(observations) != len(actions):
            raise ValueError(
                f"observations ({len(observations)}) and actions "
                f"({len(actions)}) must have equal length"
            )
        if self._orchestrator is None:
            raise RuntimeError(
                "prefill_trajectory requires a cache orchestrator, but this "
                "interceptor was constructed with orchestrator=None."
            )

        for obs, action in zip(observations, actions, strict=True):
            payload = self._build_prefill_payload(action)
            with self._orchestrator.prefill_mode(payload):
                # Full pipeline: key_builder.collect + build, strategy.search
                # (synthetic hit), judge, fetch_payload, broadcast_action —
                # every side effect runs; the returned action is discarded.
                self.infer(obs)

    @staticmethod
    def _build_prefill_payload(action):
        """Build a minimal ``CachePayload`` carrying only the GT action chunk.

        ``task_key`` defaults to ``""`` and ``intermediates`` /
        ``denoising_num_steps`` default to ``None`` — the prefill path does
        not consult them (the caller discards the returned action, and
        downstream write / warm-start logic is bypassed because the
        orchestrator sees a CP1 FULL_HIT).
        """
        from openpi.cache.storage_types import CachePayload

        if isinstance(action, np.ndarray):
            action_t = torch.from_numpy(action)
        elif isinstance(action, torch.Tensor):
            action_t = action
        else:
            action_t = torch.as_tensor(action)
        return CachePayload(action_chunk=action_t)

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
        # Strip the reserved '__gate_decision__' field BEFORE _input_transform
        # so it never enters the model input pipeline. This is an in-place
        # mutation on obs; the WebSocket path deserialises a fresh dict per
        # call so sharing is not a concern there.
        client_signal = obs.pop("__gate_decision__", None)
        accepts_client_signal = (
            self._orchestrator is not None
            and self._orchestrator.accepts_client_signal
        )
        if client_signal is not None and not accepts_client_signal:
            raise ValueError(
                "obs carries '__gate_decision__' but no ClientControlledGate "
                "is configured at CP1 or CP3. Remove the field from obs, or "
                "load a cache config with gate.type='client_controlled'."
            )
        request_context: dict | None = (
            {"gate_decision": client_signal} if client_signal is not None else None
        )

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
        # Noise goes to stage3 device for flow matching.
        start_noise: torch.Tensor | None = None
        if noise is not None:
            start_noise = torch.from_numpy(noise).to(self._stage3_device)
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
                            CheckpointID.CP1,
                            request_context=request_context,
                            **cp1_kwargs,
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

                # Cross-device transfer (only when stage placement differs)
                if self._stage_config is not None and self._stage_config.needs_relocation:
                    stage1 = stage1.to(self._stage2_device)

                # Meta guard: stage2=meta and not FULL_HIT -> clear error
                if self._stage2_device == "meta":
                    raise RuntimeError(
                        "stage2 is on meta device (not loaded). "
                        "Cannot execute forward pass. Cache CP1 did not return "
                        "FULL_HIT — check cache config (meta stages require "
                        "always_hit at the preceding checkpoint)."
                    )

                with self._timer.measure("stage2_llm"):
                    stage2 = self._stage2_fn(stage1)

                # Meta guard: check before cross-device transfer to avoid
                # moving KV cache to meta device unnecessarily.
                if self._stage3_device == "meta":
                    raise RuntimeError(
                        "stage3 is on meta device (not loaded). "
                        "Cannot execute forward pass. Cache CP1 did not return "
                        "FULL_HIT — check cache config (meta stages require "
                        "always_hit at the preceding checkpoint)."
                    )

                if self._stage_config is not None and self._stage_config.needs_relocation:
                    stage2 = stage2.to(self._stage3_device)

                # Stage 3: three-way branch
                if (self._orchestrator is not None
                        and cp1_result.hit_type == HitType.WARM_START):
                    start_t = cp1_result.start_t
                    start_x = cp1_result.payload.intermediates[start_t].to(
                        self._stage3_device
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
                        CheckpointID.CP3,
                        request_context=request_context,
                        **cp3_kwargs,
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
